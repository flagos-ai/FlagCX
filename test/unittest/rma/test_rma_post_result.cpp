/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include <gtest/gtest.h>

#include <cstdlib>

#include "adaptor.h"
#include "comm.h"
#include "flagcx_hetero.h"
#include "flagcx_net_adaptor.h"
#include "sym_heap.h"

namespace {

flagcxResult_t unusedPut(void *, uint64_t, uint64_t, size_t, int, int, void **,
                         void **, void **) {
  return flagcxInternalError;
}

flagcxResult_t unusedGet(void *, uint64_t, uint64_t, size_t, int, int, void **,
                         void **, void **) {
  return flagcxInternalError;
}

flagcxResult_t unusedPutSignal(void *, uint64_t, uint64_t, size_t, int, int,
                               void **, void **, uint64_t, void **, uint64_t,
                               void **) {
  return flagcxInternalError;
}

int deviceMemcpyCalls = 0;

flagcxResult_t recordDeviceMemcpy(void *, void *, size_t, flagcxMemcpyType_t,
                                  flagcxStream_t, void *) {
  deviceMemcpyCalls++;
  return flagcxSuccess;
}

struct RmaQueueFixture {
  flagcxNetAdaptor_latest net = {};
  flagcxRmaProxyState proxy = {};
  flagcxHeteroComm comm = {};
  flagcxRmaDesc *ring[2] = {};
  volatile uint32_t pi = 0;
  volatile uint32_t ci = 0;
  volatile uint64_t opSeq = 0;
  volatile uint64_t readySeq = 0;
  pthread_mutex_t producerMutex;

  RmaQueueFixture() {
    net.iput = unusedPut;
    net.iget = unusedGet;
    net.iputSignal = unusedPutSignal;

    pthread_mutex_init(&producerMutex, nullptr);
    proxy.queueSize = 2;
    proxy.queueMask = 1;
    proxy.circularBuffers = ring;
    proxy.pis = &pi;
    proxy.cis = &ci;
    proxy.opSeqs = &opSeq;
    proxy.readySeqsCpu = &readySeq;
    proxy.peerProducerMutexes = &producerMutex;

    comm.nRanks = 1;
    comm.netAdaptor = &net;
    comm.rmaProxy = &proxy;
    comm.signalHandle = reinterpret_cast<flagcxOneSideHandleInfo *>(0x1);
  }

  ~RmaQueueFixture() { pthread_mutex_destroy(&producerMutex); }
};

class ForcedNetworkFixture : public ::testing::Test {
protected:
  void SetUp() override {
    savedDeviceAdaptor_ = deviceAdaptor;
    ASSERT_EQ(setenv("FLAGCX_P2P_DISABLE", "1", 1), 0);
    testDeviceAdaptor_ = {};
    testDeviceAdaptor_.deviceMemcpy = recordDeviceMemcpy;
    deviceAdaptor = &testDeviceAdaptor_;
    deviceMemcpyCalls = 0;

    srcWindow_.localBase = reinterpret_cast<void *>(0x1000);
    srcWindow_.heapSize = 64;
    srcWindow_.mrIndex = -1;
    srcWindow_.ipcSlot = 0;
    dstWindow_.localBase = reinterpret_cast<void *>(0x2000);
    dstWindow_.heapSize = 64;
    dstWindow_.mrIndex = -1;
    dstWindow_.ipcSlot = 0;

    queue_.comm.rmaSignalBase = reinterpret_cast<void *>(0x3000);
    queue_.comm.rmaSignalSize = sizeof(uint64_t);
  }

  void TearDown() override { deviceAdaptor = savedDeviceAdaptor_; }

  RmaQueueFixture queue_;
  flagcxSymWindow srcWindow_ = {};
  flagcxSymWindow dstWindow_ = {};
  flagcxDeviceAdaptor_latest testDeviceAdaptor_ = {};
  flagcxDeviceAdaptor_latest *savedDeviceAdaptor_ = nullptr;
};

} // namespace

TEST(RmaPostResult, RetriesOnlyExplicitBackpressure) {
  EXPECT_TRUE(flagcxRmaPostResultIsRetryable(flagcxInProgress));
  EXPECT_FALSE(flagcxRmaPostResultIsRetryable(flagcxSuccess));
  EXPECT_FALSE(flagcxRmaPostResultIsRetryable(flagcxInternalError));
  EXPECT_FALSE(flagcxRmaPostResultIsRetryable(flagcxSystemError));
  EXPECT_FALSE(flagcxRmaPostResultIsRetryable(flagcxRemoteError));
}

TEST(RmaPostResult, ClassifiesBatchPostOutcomes) {
  EXPECT_FALSE(flagcxRmaBatchPostResultIsFatal(flagcxSuccess, 4, 4));
  EXPECT_FALSE(flagcxRmaBatchPostResultIsFatal(flagcxInProgress, 0, 4));
  EXPECT_FALSE(flagcxRmaBatchPostResultIsFatal(flagcxInProgress, 2, 4));

  EXPECT_TRUE(flagcxRmaBatchPostResultIsFatal(flagcxSuccess, 0, 4));
  EXPECT_TRUE(flagcxRmaBatchPostResultIsFatal(flagcxSuccess, 2, 4));
  EXPECT_TRUE(flagcxRmaBatchPostResultIsFatal(flagcxSystemError, 0, 4));
  EXPECT_TRUE(flagcxRmaBatchPostResultIsFatal(flagcxSystemError, 2, 4));
  EXPECT_TRUE(flagcxRmaBatchPostResultIsFatal(flagcxInternalError, 0, 4));
  EXPECT_TRUE(flagcxRmaBatchPostResultIsFatal(flagcxRemoteError, 0, 4));
  EXPECT_TRUE(flagcxRmaBatchPostResultIsFatal(flagcxSuccess, -1, 4));
  EXPECT_TRUE(flagcxRmaBatchPostResultIsFatal(flagcxSuccess, 5, 4));
}

TEST(RmaTransportSelection, MissingNetworkMrsAreRejectedBeforeEnqueue) {
  RmaQueueFixture fixture;

  const size_t offsets[] = {0};
  const size_t sizes[] = {64};
  const int mrIndexes[] = {-1};

  EXPECT_EQ(flagcxHeteroPut(&fixture.comm, 0, 0, 0, 64, -1, -1),
            flagcxNotSupported);
  EXPECT_EQ(flagcxHeteroBatchPut(&fixture.comm, 0, offsets, offsets, sizes,
                                 mrIndexes, mrIndexes, 1),
            flagcxNotSupported);
  EXPECT_EQ(flagcxHeteroGet(&fixture.comm, 0, 0, 0, 64, -1, -1),
            flagcxNotSupported);
  EXPECT_EQ(flagcxHeteroPutSignal(&fixture.comm, 0, 0, 0, 64, 0, -1, -1, 1),
            flagcxNotSupported);

  EXPECT_EQ(fixture.pi, 0u);
  EXPECT_EQ(fixture.opSeq, 0u);
  EXPECT_EQ(fixture.ring[0], nullptr);
  EXPECT_EQ(fixture.ring[1], nullptr);
}

TEST(RmaTransportSelection, SignalOnlyRequiresNetworkSignalRegistration) {
  RmaQueueFixture fixture;
  fixture.comm.signalHandle = nullptr;

  EXPECT_EQ(flagcxHeteroPutSignal(&fixture.comm, 0, 0, 0, 0, 0, -1, -1, 1),
            flagcxNotSupported);
  EXPECT_EQ(fixture.pi, 0u);
  EXPECT_EQ(fixture.opSeq, 0u);
}

TEST_F(ForcedNetworkFixture,
       IpcOnlyWindowsAreRejectedWithoutCopyingOrEnqueueing) {
  flagcxStream_t stream = reinterpret_cast<flagcxStream_t>(0x1);

  EXPECT_EQ(flagcxHeteroPutStream(&queue_.comm, 0, 0, 0, 64, -1, -1,
                                  &srcWindow_, &dstWindow_, stream, nullptr),
            flagcxNotSupported);
  EXPECT_EQ(flagcxHeteroGet(&queue_.comm, 0, 0, 0, 64, -1, -1),
            flagcxNotSupported);
  EXPECT_EQ(flagcxHeteroPutSignalStream(&queue_.comm, 0, 0, 0, 64, 0, -1, -1, 1,
                                        &srcWindow_, &dstWindow_, stream,
                                        nullptr),
            flagcxNotSupported);

  EXPECT_EQ(deviceMemcpyCalls, 0);
  EXPECT_EQ(queue_.pi, 0u);
  EXPECT_EQ(queue_.opSeq, 0u);
  EXPECT_EQ(queue_.ring[0], nullptr);
  EXPECT_EQ(queue_.ring[1], nullptr);
}
