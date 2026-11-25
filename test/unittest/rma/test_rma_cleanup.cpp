#include "adaptor.h"
#include "comm.h"
#include "flagcx_hetero.h"
#include <gtest/gtest.h>

namespace {
int frees, failures, unmaps;
flagcxResult_t release(void *p, void *) {
  if (failures) {
    --failures;
    return flagcxSystemError;
  }
  ++frees;
  free(p);
  return flagcxSuccess;
}
class RmaCleanup : public ::testing::Test {
protected:
  flagcxDeviceAdaptor *saved;
  flagcxDeviceAdaptor device = {};
  flagcxHeteroComm comm = {};
  void SetUp() override {
    saved = deviceAdaptor;
    deviceAdaptor = &device;
    frees = failures = unmaps = 0;
    device.setDevice = [](int) { return flagcxSuccess; };
    device.gdrMemFree = release;
    device.gdrPtrMunmap = [](void *p, size_t) {
      ++unmaps;
      free(p);
      return flagcxSuccess;
    };
    comm.rmaProxy =
        (flagcxRmaProxyState *)calloc(1, sizeof(flagcxRmaProxyState));
    pthread_mutex_init(&comm.rmaProxy->doneMutex, NULL);
    pthread_cond_init(&comm.rmaProxy->doneCond, NULL);
  }
  void TearDown() override {
    failures = 0;
    EXPECT_EQ(flagcxHeteroRmaProxyStop(&comm), flagcxSuccess);
    EXPECT_EQ(flagcxHeteroRmaProxyRelease(&comm), flagcxSuccess);
    deviceAdaptor = saved;
  }
};
TEST_F(RmaCleanup, StopRetainsExportsAndReleaseRetriesOnlyRemainingOwners) {
  auto *proxy = comm.rmaProxy;
  proxy->doneSeqsCpu = (volatile uint64_t *)calloc(1, sizeof(uint64_t));
  proxy->readySeqsCpu = (volatile uint64_t *)calloc(1, sizeof(uint64_t));
  proxy->doneSeqsMapped = true;
  proxy->doneSeqsDev = (uint64_t *)malloc(8);
  proxy->readySeqsDev = (uint64_t *)malloc(8);
  auto *done = proxy->doneSeqsDev;
  EXPECT_EQ(flagcxHeteroRmaProxyRelease(&comm), flagcxInvalidUsage);
  ASSERT_EQ(flagcxHeteroRmaProxyStop(&comm), flagcxSuccess);
  EXPECT_EQ(proxy->doneSeqsCpu, nullptr);
  EXPECT_EQ(proxy->readySeqsCpu, nullptr);
  EXPECT_EQ(proxy->doneSeqsDev, done);
  EXPECT_EQ(frees, 0);
  EXPECT_EQ(unmaps, 1);
  EXPECT_EQ(flagcxHeteroRmaProxyStop(&comm), flagcxSuccess);
  failures = 1;
  EXPECT_EQ(flagcxHeteroRmaProxyRelease(&comm), flagcxSystemError);
  EXPECT_EQ(comm.rmaProxy, proxy);
  EXPECT_EQ(proxy->doneSeqsDev, done);
  EXPECT_EQ(flagcxHeteroRmaProxyRelease(&comm), flagcxSuccess);
  EXPECT_EQ(comm.rmaProxy, nullptr);
  EXPECT_EQ(frees, 2);
  EXPECT_EQ(unmaps, 1);
}
TEST_F(RmaCleanup, HostFunctionModeDoesNotRequireDeviceAllocations) {
  comm.rmaProxy->doneSeqsCpu = (volatile uint64_t *)calloc(1, 8);
  comm.rmaProxy->readySeqsCpu = (volatile uint64_t *)calloc(1, 8);
  ASSERT_EQ(flagcxHeteroRmaProxyStop(&comm), flagcxSuccess);
  EXPECT_TRUE(comm.rmaProxy->stopComplete);
  EXPECT_EQ(flagcxHeteroRmaProxyRelease(&comm), flagcxSuccess);
  EXPECT_EQ(frees, 0);
}
TEST_F(RmaCleanup, OutstandingCallbackRetainsEveryOwner) {
  comm.rmaProxy->callbacks = 1;
  EXPECT_EQ(flagcxHeteroRmaProxyStop(&comm), flagcxInProgress);
  EXPECT_FALSE(comm.rmaProxy->stopComplete);
  EXPECT_EQ(flagcxHeteroRmaProxyRelease(&comm), flagcxInvalidUsage);
  comm.rmaProxy->callbacks = 0;
}
} // namespace
