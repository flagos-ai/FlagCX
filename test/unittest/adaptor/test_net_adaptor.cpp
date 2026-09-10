/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <vector>

#include "adaptor.h"
#include "flagcx.h"
#include "flagcx_net.h"
#include "flagcx_net_adaptor.h"
#include "ib_common.h"
#include "onesided.h"

namespace {

constexpr size_t kBufferSize = 4096;
constexpr int kLocalRank = 0;
constexpr int kRemoteRank = 1;
constexpr int kRequestPoolSize = 256;
constexpr auto kTimeout = std::chrono::seconds(30);

#define SKIP_IF_CALLBACK_NULL(net, callback)                                   \
  do {                                                                         \
    if ((net)->callback == nullptr)                                            \
      GTEST_SKIP() << "Selected net adaptor does not implement " #callback;    \
  } while (0)

flagcxResult_t waitRequest(struct flagcxNetAdaptor *net, void *request,
                           int *size = nullptr) {
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  int done = 0;
  while (!done && std::chrono::steady_clock::now() < deadline) {
    flagcxResult_t result = net->test(request, &done, size);
    if (result != flagcxSuccess)
      return result;
    if (!done)
      std::this_thread::yield();
  }
  return done ? flagcxSuccess : flagcxSystemError;
}

struct ConnectionResult {
  flagcxResult_t connectResult = flagcxSystemError;
  flagcxResult_t acceptResult = flagcxSystemError;
  void *sendComm = nullptr;
  void *recvComm = nullptr;
};

ConnectionResult connectLoopback(struct flagcxNetAdaptor *net, int dev,
                                 const char *listenHandle, void *listenComm) {
  ConnectionResult result;
  char connectHandle[FLAGCX_NET_HANDLE_MAXSIZE] = {};
  memcpy(connectHandle, listenHandle, sizeof(connectHandle));

  std::atomic<bool> cancelled(false);
  std::mutex mutex;
  std::condition_variable completion;
  int completed = 0;
  auto markComplete = [&]() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      ++completed;
    }
    completion.notify_one();
  };

  std::thread connector([&]() {
    const auto deadline = std::chrono::steady_clock::now() + kTimeout;
    result.connectResult = flagcxSuccess;
    while (result.sendComm == nullptr && !cancelled.load() &&
           std::chrono::steady_clock::now() < deadline) {
      result.connectResult = net->connect(dev, connectHandle, &result.sendComm);
      if (result.connectResult != flagcxSuccess) {
        cancelled.store(true);
        break;
      }
      if (result.sendComm == nullptr)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (result.sendComm == nullptr && result.connectResult == flagcxSuccess) {
      result.connectResult = flagcxSystemError;
      cancelled.store(true);
    }
    markComplete();
  });

  std::thread accepter([&]() {
    const auto deadline = std::chrono::steady_clock::now() + kTimeout;
    result.acceptResult = flagcxSuccess;
    while (result.recvComm == nullptr && !cancelled.load() &&
           std::chrono::steady_clock::now() < deadline) {
      result.acceptResult = net->accept(listenComm, &result.recvComm);
      if (result.acceptResult != flagcxSuccess) {
        cancelled.store(true);
        break;
      }
      if (result.recvComm == nullptr)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (result.recvComm == nullptr && result.acceptResult == flagcxSuccess) {
      result.acceptResult = flagcxSystemError;
      cancelled.store(true);
    }
    markComplete();
  });

  {
    std::unique_lock<std::mutex> lock(mutex);
    if (!completion.wait_for(lock, kTimeout + std::chrono::seconds(1),
                             [&]() { return completed == 2; })) {
      fprintf(stderr, "net adaptor loopback handshake timed out\n");
      std::abort();
    }
  }
  connector.join();
  accepter.join();
  return result;
}

struct TestWindow {
  struct flagcxOneSideHandleInfo info = {};
  uintptr_t baseVas[2] = {};
  size_t regionSizes[2] = {};
  struct flagcxNetMrInfo mrInfos[2] = {};

  flagcxResult_t init(struct flagcxNetAdaptor *net, void *buffer, size_t size,
                      int rank, void *mrHandle) {
    if (net == nullptr || net->getMrInfo == nullptr || buffer == nullptr ||
        rank < 0 || rank >= 2 || mrHandle == nullptr)
      return flagcxInvalidArgument;
    FLAGCXCHECK(net->getMrInfo(mrHandle, &mrInfos[rank]));
    baseVas[rank] = reinterpret_cast<uintptr_t>(buffer);
    regionSizes[rank] = size;
    info.baseVas = baseVas;
    info.regionSizes = regionSizes;
    info.mrInfos = mrInfos;
    info.localMrHandle = mrHandle;
    info.nRanks = 2;
    return flagcxSuccess;
  }

  void **opaque() { return reinterpret_cast<void **>(&info); }
};

std::vector<ibv_mr *> deregisterCalls;
ibv_mr *deregisterFailure = nullptr;
int deregisterFailuresRemaining = 0;
int batchPostResult = IBV_SUCCESS;
int batchRejectedIndex = -1;
int batchPostCalls = 0;

flagcxResult_t fakeDeregisterMr(flagcxIbNetCommDevBase *, ibv_mr *mr) {
  deregisterCalls.push_back(mr);
  if (mr == deregisterFailure && deregisterFailuresRemaining > 0) {
    --deregisterFailuresRemaining;
    return flagcxSystemError;
  }
  return flagcxSuccess;
}

int fakeBatchPostSend(ibv_qp *, ibv_send_wr *wr, ibv_send_wr **badWr) {
  ++batchPostCalls;
  ibv_send_wr *rejected = wr;
  for (int i = 0; i < batchRejectedIndex && rejected != nullptr; ++i)
    rejected = rejected->next;
  if (badWr != nullptr)
    *badWr = rejected;
  return batchPostResult;
}

class IbMrCleanupTest : public ::testing::Test {
protected:
  void SetUp() override {
    deregisterCalls.clear();
    deregisterFailure = nullptr;
    deregisterFailuresRemaining = 0;
    comm_ = static_cast<flagcxIbSendComm *>(
        calloc(1, sizeof(struct flagcxIbSendComm)));
    ASSERT_NE(comm_, nullptr);
    comm_->base.isSend = true;
    comm_->base.ndevs = 2;
  }

  void TearDown() override { free(comm_); }

  flagcxIbSendComm *comm_ = nullptr;
};

class NetAdaptorLoopback : public ::testing::Test {
protected:
  void SetUp() override {
    net_ = getNetAdaptor(RDMA);
    ASSERT_NE(net_, nullptr);
    SKIP_IF_CALLBACK_NULL(net_, init);
    SKIP_IF_CALLBACK_NULL(net_, devices);
    SKIP_IF_CALLBACK_NULL(net_, listen);
    SKIP_IF_CALLBACK_NULL(net_, connect);
    SKIP_IF_CALLBACK_NULL(net_, accept);
    SKIP_IF_CALLBACK_NULL(net_, closeSend);
    SKIP_IF_CALLBACK_NULL(net_, closeRecv);
    SKIP_IF_CALLBACK_NULL(net_, closeListen);
    ASSERT_EQ(net_->init(), flagcxSuccess);
    ASSERT_EQ(net_->devices(&nDevs_), flagcxSuccess);
    ASSERT_GT(nDevs_, 0);
    ASSERT_EQ(net_->listen(0, handle_, &listenComm_), flagcxSuccess);
    ASSERT_NE(listenComm_, nullptr);

    ConnectionResult connection =
        connectLoopback(net_, 0, handle_, listenComm_);
    sendComm_ = connection.sendComm;
    recvComm_ = connection.recvComm;
    ASSERT_EQ(connection.connectResult, flagcxSuccess);
    ASSERT_EQ(connection.acceptResult, flagcxSuccess);
    ASSERT_NE(sendComm_, nullptr);
    ASSERT_NE(recvComm_, nullptr);
  }

  void TearDown() override {
    if (deviceBuffer_ != nullptr && deviceAdaptor != nullptr &&
        deviceAdaptor->deviceFree != nullptr) {
      EXPECT_EQ(
          deviceAdaptor->deviceFree(deviceBuffer_, flagcxMemDevice, nullptr),
          flagcxSuccess);
      deviceBuffer_ = nullptr;
    }
    if (net_ == nullptr)
      return;
    if (sendComm_ != nullptr) {
      EXPECT_EQ(net_->closeSend(sendComm_), flagcxSuccess);
    }
    if (recvComm_ != nullptr) {
      EXPECT_EQ(net_->closeRecv(recvComm_), flagcxSuccess);
    }
    if (listenComm_ != nullptr) {
      EXPECT_EQ(net_->closeListen(listenComm_), flagcxSuccess);
    }
  }

  flagcxResult_t registerMr(void *comm, void *buffer, size_t size, int type,
                            void **mrHandle,
                            int flags = FLAGCX_NET_MR_FLAG_NONE) {
    *mrHandle = nullptr;
    return net_->regMr(comm, buffer, size, type, flags, mrHandle);
  }

  struct flagcxNetAdaptor *net_ = nullptr;
  int nDevs_ = 0;
  char handle_[FLAGCX_NET_HANDLE_MAXSIZE] = {};
  void *listenComm_ = nullptr;
  void *sendComm_ = nullptr;
  void *recvComm_ = nullptr;
  void *deviceBuffer_ = nullptr;
};

#define ASSERT_REGISTER_MR(comm, buffer, size, type, handle)                   \
  do {                                                                         \
    ASSERT_EQ(registerMr((comm), (buffer), (size), (type), &(handle)),         \
              flagcxSuccess);                                                  \
    ASSERT_NE((handle), nullptr);                                              \
  } while (0)

#define EXPECT_DEREGISTER_MR(comm, handle)                                     \
  do {                                                                         \
    EXPECT_EQ(net_->deregMr((comm), (handle)), flagcxSuccess);                 \
    (handle) = nullptr;                                                        \
  } while (0)

TEST(NetAdaptorInterface, UpgradeV1ZeroInitializesExtensions) {
  struct flagcxNetAdaptor_v1 legacy = {};
  legacy.name = "legacy";
  struct flagcxNetAdaptor_latest upgraded;
  memset(&upgraded, 0xff, sizeof(upgraded));
  flagcxNetAdaptorUpgrade(&legacy, &upgraded);
  EXPECT_STREQ(upgraded.name, "legacy");
  EXPECT_EQ(upgraded.iputBatch, nullptr);
  EXPECT_EQ(upgraded.testBatch, nullptr);
  EXPECT_EQ(upgraded.igetBatch, nullptr);
  EXPECT_EQ(upgraded.getMrInfo, nullptr);
}

TEST(NetAdaptorInterface, RdmaAdaptorAdvertisesOneSidedContract) {
  struct flagcxNetAdaptor *net = getNetAdaptor(RDMA);
  ASSERT_NE(net, nullptr);
  EXPECT_NE(net->name, nullptr);
  EXPECT_NE(net->regMr, nullptr);
  EXPECT_NE(net->deregMr, nullptr);
  if (net->name == nullptr || strcmp(net->name, "IB") != 0)
    GTEST_SKIP() << "The build-selected RDMA adaptor is not IBRC";
  EXPECT_NE(net->getMrInfo, nullptr);
  EXPECT_NE(net->iput, nullptr);
  EXPECT_NE(net->iget, nullptr);
  EXPECT_NE(net->iputSignal, nullptr);
  EXPECT_NE(net->iputBatch, nullptr);
  EXPECT_NE(net->testBatch, nullptr);
  EXPECT_NE(net->igetBatch, nullptr);
}

TEST_F(IbMrCleanupTest, FailedDeregisterRetainsOnlyUnconsumedNicHandles) {
  auto *wrapper = static_cast<flagcxIbMrHandle *>(
      calloc(1, sizeof(struct flagcxIbMrHandle)));
  ASSERT_NE(wrapper, nullptr);
  auto *first = reinterpret_cast<ibv_mr *>(0x1000);
  auto *second = reinterpret_cast<ibv_mr *>(0x2000);
  wrapper->mrs[0] = first;
  wrapper->mrs[1] = second;
  deregisterFailure = second;
  deregisterFailuresRemaining = 1;

  EXPECT_EQ(
      flagcxIbDeregMrWithCallback(&comm_->base, wrapper, fakeDeregisterMr),
      flagcxSystemError);
  ASSERT_EQ(deregisterCalls.size(), 2u);
  EXPECT_EQ(deregisterCalls[0], first);
  EXPECT_EQ(deregisterCalls[1], second);
  EXPECT_EQ(wrapper->mrs[0], nullptr);
  EXPECT_EQ(wrapper->mrs[1], second);

  deregisterCalls.clear();
  EXPECT_EQ(
      flagcxIbDeregMrWithCallback(&comm_->base, wrapper, fakeDeregisterMr),
      flagcxSuccess);
  ASSERT_EQ(deregisterCalls.size(), 1u);
  EXPECT_EQ(deregisterCalls[0], second);
}

TEST_F(IbMrCleanupTest, DeferredRollbackRemainsQueuedUntilRetrySucceeds) {
  auto *wrapper = static_cast<flagcxIbMrHandle *>(
      calloc(1, sizeof(struct flagcxIbMrHandle)));
  ASSERT_NE(wrapper, nullptr);
  auto *mr = reinterpret_cast<ibv_mr *>(0x3000);
  wrapper->mrs[0] = mr;
  comm_->base.deferredMrHandles = wrapper;
  deregisterFailure = mr;
  deregisterFailuresRemaining = 1;

  EXPECT_EQ(
      flagcxIbDrainDeferredMrsWithCallback(&comm_->base, fakeDeregisterMr),
      flagcxSystemError);
  EXPECT_EQ(comm_->base.deferredMrHandles, wrapper);
  EXPECT_EQ(wrapper->mrs[0], mr);

  EXPECT_EQ(
      flagcxIbDrainDeferredMrsWithCallback(&comm_->base, fakeDeregisterMr),
      flagcxSuccess);
  EXPECT_EQ(comm_->base.deferredMrHandles, nullptr);
}

TEST_F(IbMrCleanupTest, FailedPublicCleanupIsDeferredAndRemainsRetryable) {
  auto *wrapper = static_cast<flagcxIbMrHandle *>(
      calloc(1, sizeof(struct flagcxIbMrHandle)));
  ASSERT_NE(wrapper, nullptr);
  auto *mr = reinterpret_cast<ibv_mr *>(0x4000);
  wrapper->mrs[0] = mr;
  deregisterFailure = mr;
  deregisterFailuresRemaining = 1;

  EXPECT_EQ(flagcxIbDeregMrOrDeferWithCallback(&comm_->base, wrapper,
                                               fakeDeregisterMr),
            flagcxSystemError);
  EXPECT_EQ(comm_->base.deferredMrHandles, wrapper);
  EXPECT_EQ(wrapper->mrs[0], mr);

  deregisterCalls.clear();
  EXPECT_EQ(flagcxIbDeregMrOrDeferWithCallback(&comm_->base, wrapper,
                                               fakeDeregisterMr),
            flagcxSuccess);
  EXPECT_EQ(comm_->base.deferredMrHandles, nullptr);
  ASSERT_EQ(deregisterCalls.size(), 1u);
  EXPECT_EQ(deregisterCalls[0], mr);
}

TEST(IbRequestCompletionTest, SharedCqUpdatesTheWrIdRequestAndDrainsBatch) {
  auto base = std::make_unique<flagcxIbNetCommBase>();
  auto *polled = &base->reqs[0];
  auto *completed = &base->reqs[1];
  polled->type = FLAGCX_NET_IB_REQ_IPUT;
  polled->result = flagcxSuccess;
  polled->events[0] = 1;
  completed->type = FLAGCX_NET_IB_REQ_IGET;
  completed->result = flagcxSuccess;
  completed->events[0] = 2;

  EXPECT_EQ(
      flagcxIbCommonRecordDataCompletion(base.get(), 1, 0, flagcxRemoteError),
      flagcxSuccess);
  EXPECT_EQ(polled->events[0], 1);
  EXPECT_EQ(polled->result, flagcxSuccess);
  EXPECT_EQ(completed->events[0], 1);
  EXPECT_EQ(completed->result, flagcxRemoteError);

  EXPECT_EQ(flagcxIbCommonRecordDataCompletion(base.get(), 1, 0, flagcxSuccess),
            flagcxSuccess);
  EXPECT_EQ(completed->events[0], 0);
  EXPECT_EQ(completed->result, flagcxRemoteError);
}

TEST(IbRequestCompletionTest,
     UnsignaledErrorIsRecordedWithoutConsumingTailEvent) {
  auto base = std::make_unique<flagcxIbNetCommBase>();
  auto *request = &base->reqs[7];
  request->type = FLAGCX_NET_IB_REQ_IPUT;
  request->result = flagcxSuccess;
  request->events[0] = 1;

  uint64_t wrId = flagcxIbUnsignaledWrId(7);
  ASSERT_TRUE(flagcxIbIsUnsignaledWrId(wrId));
  EXPECT_EQ(flagcxIbCommonRecordUnsignaledCompletion(base.get(), wrId,
                                                     flagcxRemoteError),
            flagcxSuccess);
  EXPECT_EQ(request->events[0], 1);
  EXPECT_EQ(request->result, flagcxRemoteError);

  EXPECT_EQ(flagcxIbCommonRecordDataCompletion(base.get(), 7, 0, flagcxSuccess),
            flagcxSuccess);
  EXPECT_EQ(request->events[0], 0);
  EXPECT_EQ(request->result, flagcxRemoteError);
}

TEST(IbOneSidedBatchContractTest,
     PermanentPartialPostRetainsOnlyAcceptedPrefix) {
  struct flagcxNetAdaptor *net = getNetAdaptor(RDMA);
  ASSERT_NE(net, nullptr);
  if (net->name == nullptr || strcmp(net->name, "IB") != 0)
    GTEST_SKIP() << "Partial-post injection is specific to IBRC";
  ASSERT_NE(net->iputBatch, nullptr);

  auto comm = std::make_unique<flagcxIbSendComm>();
  ibv_context context = {};
  ibv_qp qp = {};
  context.ops.post_send = fakeBatchPostSend;
  qp.context = &context;
  comm->base.isSend = true;
  comm->base.ready = 1;
  comm->base.ndevs = 1;
  comm->base.nqps = 1;
  comm->base.qps[0].qp = &qp;
  comm->base.qps[0].devIndex = 0;
  comm->base.qps[0].remDevIdx = 0;

  uint8_t source[3] = {1, 2, 3};
  uint8_t destination[3] = {};
  TestWindow sourceWindow, destinationWindow;
  sourceWindow.baseVas[kLocalRank] = reinterpret_cast<uintptr_t>(source);
  sourceWindow.regionSizes[kLocalRank] = sizeof(source);
  sourceWindow.mrInfos[kLocalRank].nKeys = 1;
  sourceWindow.mrInfos[kLocalRank].lkeys[0] = 11;
  sourceWindow.info.baseVas = sourceWindow.baseVas;
  sourceWindow.info.regionSizes = sourceWindow.regionSizes;
  sourceWindow.info.mrInfos = sourceWindow.mrInfos;
  sourceWindow.info.nRanks = 2;
  destinationWindow.baseVas[kRemoteRank] =
      reinterpret_cast<uintptr_t>(destination);
  destinationWindow.regionSizes[kRemoteRank] = sizeof(destination);
  destinationWindow.mrInfos[kRemoteRank].nKeys = 1;
  destinationWindow.mrInfos[kRemoteRank].rkeys[0] = 22;
  destinationWindow.info.baseVas = destinationWindow.baseVas;
  destinationWindow.info.regionSizes = destinationWindow.regionSizes;
  destinationWindow.info.mrInfos = destinationWindow.mrInfos;
  destinationWindow.info.nRanks = 2;

  const uint64_t offsets[3] = {0, 1, 2};
  const size_t sizes[3] = {1, 1, 1};
  void *requests[3] = {};
  int posted = -1;
  batchPostResult = EINVAL;
  batchRejectedIndex = 1;
  batchPostCalls = 0;

  EXPECT_EQ(net->iputBatch(&comm->base, 3, offsets, offsets, sizes, kLocalRank,
                           kRemoteRank, sourceWindow.opaque(),
                           destinationWindow.opaque(), requests, &posted),
            flagcxSystemError);
  EXPECT_EQ(batchPostCalls, 1);
  EXPECT_EQ(posted, 1);
  ASSERT_NE(requests[0], nullptr);
  EXPECT_EQ(requests[1], nullptr);
  EXPECT_EQ(requests[2], nullptr);
  auto *accepted = static_cast<flagcxIbRequest *>(requests[0]);
  EXPECT_EQ(accepted->events[0], 1);
  EXPECT_EQ(comm->base.reqs[1].type, FLAGCX_NET_IB_REQ_UNUSED);
  EXPECT_EQ(comm->base.reqs[2].type, FLAGCX_NET_IB_REQ_UNUSED);
  EXPECT_EQ(flagcxIbFreeRequest(accepted), flagcxSuccess);
}

TEST_F(NetAdaptorLoopback, RegisterHostMrAndExportMetadata) {
  SKIP_IF_CALLBACK_NULL(net_, regMr);
  SKIP_IF_CALLBACK_NULL(net_, deregMr);
  SKIP_IF_CALLBACK_NULL(net_, getMrInfo);
  std::vector<uint8_t> buffer(kBufferSize);
  void *mr = nullptr;
  ASSERT_REGISTER_MR(sendComm_, buffer.data(), buffer.size(), FLAGCX_PTR_HOST,
                     mr);
  struct flagcxNetMrInfo info = {};
  ASSERT_EQ(net_->getMrInfo(mr, &info), flagcxSuccess);
  EXPECT_GT(info.nKeys, 0u);
  EXPECT_LE(info.nKeys, static_cast<uint32_t>(FLAGCX_NET_MAX_MR_KEYS));
  if (net_->name != nullptr && strcmp(net_->name, "IB") == 0) {
    auto *comm = static_cast<flagcxIbSendComm *>(sendComm_);
    auto *wrapper = static_cast<flagcxIbMrHandle *>(mr);
    ASSERT_EQ(info.nKeys, static_cast<uint32_t>(comm->base.ndevs));
    for (uint32_t i = 0; i < info.nKeys; ++i) {
      ASSERT_NE(wrapper->mrs[i], nullptr);
      EXPECT_EQ(info.lkeys[i], wrapper->mrs[i]->lkey);
      EXPECT_EQ(info.rkeys[i], wrapper->mrs[i]->rkey);
    }
  }
  EXPECT_DEREGISTER_MR(sendComm_, mr);
  EXPECT_EQ(net_->deregMr(sendComm_, nullptr), flagcxSuccess);
}

TEST_F(NetAdaptorLoopback, RegisterGpuMr) {
  SKIP_IF_CALLBACK_NULL(net_, getProperties);
  SKIP_IF_CALLBACK_NULL(net_, regMr);
  SKIP_IF_CALLBACK_NULL(net_, deregMr);
  flagcxNetProperties_t properties = {};
  ASSERT_EQ(net_->getProperties(0, &properties), flagcxSuccess);
  if ((properties.ptrSupport & FLAGCX_PTR_CUDA) == 0)
    GTEST_SKIP() << "Selected RDMA adaptor does not advertise GPU MR support";
  ASSERT_NE(deviceAdaptor, nullptr);
  ASSERT_NE(deviceAdaptor->setDevice, nullptr);
  ASSERT_NE(deviceAdaptor->deviceMalloc, nullptr);
  ASSERT_NE(deviceAdaptor->deviceFree, nullptr);
  ASSERT_EQ(deviceAdaptor->setDevice(0), flagcxSuccess);
  ASSERT_EQ(deviceAdaptor->deviceMalloc(&deviceBuffer_, kBufferSize,
                                        flagcxMemDevice, nullptr),
            flagcxSuccess);
  ASSERT_NE(deviceBuffer_, nullptr);
  void *mr = nullptr;
  ASSERT_REGISTER_MR(sendComm_, deviceBuffer_, kBufferSize, FLAGCX_PTR_CUDA,
                     mr);
  EXPECT_DEREGISTER_MR(sendComm_, mr);
}

TEST_F(NetAdaptorLoopback, RegisterDmaBufMr) {
  SKIP_IF_CALLBACK_NULL(net_, getProperties);
  SKIP_IF_CALLBACK_NULL(net_, regMrDmaBuf);
  SKIP_IF_CALLBACK_NULL(net_, deregMr);
  flagcxNetProperties_t properties = {};
  ASSERT_EQ(net_->getProperties(0, &properties), flagcxSuccess);
  if ((properties.ptrSupport & FLAGCX_PTR_DMABUF) == 0)
    GTEST_SKIP() << "Selected RDMA adaptor does not advertise DMA-BUF support";
  ASSERT_NE(deviceAdaptor, nullptr);
  if (deviceAdaptor->setDevice == nullptr ||
      deviceAdaptor->dmaSupport == nullptr ||
      deviceAdaptor->gdrMemAlloc == nullptr ||
      deviceAdaptor->gdrMemFree == nullptr ||
      deviceAdaptor->getHandleForAddressRange == nullptr)
    GTEST_SKIP() << "Selected device adaptor cannot export DMA-BUF memory";
  ASSERT_EQ(deviceAdaptor->setDevice(0), flagcxSuccess);
  bool supported = false;
  ASSERT_EQ(deviceAdaptor->dmaSupport(&supported), flagcxSuccess);
  if (!supported)
    GTEST_SKIP() << "Selected device does not support DMA-BUF export";

  void *buffer = nullptr;
  ASSERT_EQ(deviceAdaptor->gdrMemAlloc(&buffer, kBufferSize, nullptr),
            flagcxSuccess);
  ASSERT_NE(buffer, nullptr);
  int fd = -1;
  ASSERT_EQ(
      deviceAdaptor->getHandleForAddressRange(&fd, buffer, kBufferSize, 0),
      flagcxSuccess);
  ASSERT_GE(fd, 0);
  void *mr = nullptr;
  EXPECT_EQ(net_->regMrDmaBuf(sendComm_, buffer, kBufferSize, FLAGCX_PTR_DMABUF,
                              0, fd, FLAGCX_NET_MR_FLAG_NONE, &mr),
            flagcxSuccess);
  EXPECT_NE(mr, nullptr);
  if (mr != nullptr) {
    EXPECT_EQ(net_->deregMr(sendComm_, mr), flagcxSuccess);
  }
  close(fd);
  EXPECT_EQ(deviceAdaptor->gdrMemFree(buffer, nullptr), flagcxSuccess);
}

TEST_F(NetAdaptorLoopback, SendRecv) {
  SKIP_IF_CALLBACK_NULL(net_, isend);
  SKIP_IF_CALLBACK_NULL(net_, irecv);
  SKIP_IF_CALLBACK_NULL(net_, test);
  std::vector<uint8_t> source(kBufferSize, 0x5a);
  std::vector<uint8_t> destination(kBufferSize, 0);
  void *sourceMr = nullptr;
  void *destinationMr = nullptr;
  ASSERT_REGISTER_MR(sendComm_, source.data(), source.size(), FLAGCX_PTR_HOST,
                     sourceMr);
  ASSERT_REGISTER_MR(recvComm_, destination.data(), destination.size(),
                     FLAGCX_PTR_HOST, destinationMr);

  void *recvData[1] = {destination.data()};
  size_t recvSizes[1] = {destination.size()};
  int tags[1] = {7};
  void *recvMrs[1] = {destinationMr};
  void *recvRequest = nullptr;
  ASSERT_EQ(net_->irecv(recvComm_, 1, recvData, recvSizes, tags, recvMrs,
                        nullptr, &recvRequest),
            flagcxSuccess);
  ASSERT_NE(recvRequest, nullptr);

  void *sendRequest = nullptr;
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  while (sendRequest == nullptr &&
         std::chrono::steady_clock::now() < deadline) {
    ASSERT_EQ(net_->isend(sendComm_, source.data(), source.size(), tags[0],
                          sourceMr, nullptr, &sendRequest),
              flagcxSuccess);
    if (sendRequest == nullptr)
      std::this_thread::yield();
  }
  ASSERT_NE(sendRequest, nullptr);
  ASSERT_EQ(waitRequest(net_, sendRequest), flagcxSuccess);
  ASSERT_EQ(waitRequest(net_, recvRequest), flagcxSuccess);
  EXPECT_EQ(source, destination);
  EXPECT_DEREGISTER_MR(sendComm_, sourceMr);
  EXPECT_DEREGISTER_MR(recvComm_, destinationMr);
}

TEST_F(NetAdaptorLoopback, IputIgetAndPerRankBounds) {
  SKIP_IF_CALLBACK_NULL(net_, getMrInfo);
  SKIP_IF_CALLBACK_NULL(net_, iput);
  SKIP_IF_CALLBACK_NULL(net_, iget);
  SKIP_IF_CALLBACK_NULL(net_, test);
  std::vector<uint8_t> source(1024);
  std::vector<uint8_t> remote(4096, 0);
  std::vector<uint8_t> destination(2048, 0);
  for (size_t i = 0; i < source.size(); ++i)
    source[i] = static_cast<uint8_t>(i);

  void *sourceMr = nullptr;
  void *remoteMr = nullptr;
  void *destinationMr = nullptr;
  ASSERT_REGISTER_MR(sendComm_, source.data(), source.size(), FLAGCX_PTR_HOST,
                     sourceMr);
  ASSERT_REGISTER_MR(recvComm_, remote.data(), remote.size(), FLAGCX_PTR_HOST,
                     remoteMr);
  ASSERT_REGISTER_MR(sendComm_, destination.data(), destination.size(),
                     FLAGCX_PTR_HOST, destinationMr);
  TestWindow sourceWindow, remoteWindow, destinationWindow;
  ASSERT_EQ(sourceWindow.init(net_, source.data(), source.size(), kLocalRank,
                              sourceMr),
            flagcxSuccess);
  ASSERT_EQ(remoteWindow.init(net_, remote.data(), remote.size(), kRemoteRank,
                              remoteMr),
            flagcxSuccess);
  ASSERT_EQ(destinationWindow.init(net_, destination.data(), destination.size(),
                                   kLocalRank, destinationMr),
            flagcxSuccess);

  void *request = nullptr;
  ASSERT_EQ(net_->iput(sendComm_, 512, 3500, 512, kLocalRank, kRemoteRank,
                       sourceWindow.opaque(), remoteWindow.opaque(), &request),
            flagcxSuccess);
  ASSERT_EQ(waitRequest(net_, request), flagcxSuccess);
  EXPECT_EQ(memcmp(source.data() + 512, remote.data() + 3500, 512), 0);

  request = reinterpret_cast<void *>(1);
  EXPECT_EQ(net_->iput(sendComm_, 800, 0, 512, kLocalRank, kRemoteRank,
                       sourceWindow.opaque(), remoteWindow.opaque(), &request),
            flagcxInvalidArgument);
  EXPECT_EQ(request, nullptr);
  request = reinterpret_cast<void *>(1);
  EXPECT_EQ(net_->iput(sendComm_, 0, 3800, 512, kLocalRank, kRemoteRank,
                       sourceWindow.opaque(), remoteWindow.opaque(), &request),
            flagcxInvalidArgument);
  EXPECT_EQ(request, nullptr);
  request = reinterpret_cast<void *>(1);
  EXPECT_EQ(net_->iput(sendComm_, 0, 0, 1, 2, kRemoteRank,
                       sourceWindow.opaque(), remoteWindow.opaque(), &request),
            flagcxInvalidArgument);
  EXPECT_EQ(request, nullptr);

  request = nullptr;
  ASSERT_EQ(net_->iput(sendComm_, source.size(), remote.size(), 0, kLocalRank,
                       kRemoteRank, sourceWindow.opaque(),
                       remoteWindow.opaque(), &request),
            flagcxSuccess);
  ASSERT_NE(request, nullptr);
  ASSERT_EQ(waitRequest(net_, request), flagcxSuccess);

  request = nullptr;
  ASSERT_EQ(net_->iget(sendComm_, 3500, 1000, 512, kRemoteRank, kLocalRank,
                       remoteWindow.opaque(), destinationWindow.opaque(),
                       &request),
            flagcxSuccess);
  ASSERT_EQ(waitRequest(net_, request), flagcxSuccess);
  EXPECT_EQ(memcmp(remote.data() + 3500, destination.data() + 1000, 512), 0);
  request = reinterpret_cast<void *>(1);
  EXPECT_EQ(net_->iget(sendComm_, 3800, 0, 512, kRemoteRank, kLocalRank,
                       remoteWindow.opaque(), destinationWindow.opaque(),
                       &request),
            flagcxInvalidArgument);
  EXPECT_EQ(request, nullptr);
  request = reinterpret_cast<void *>(1);
  EXPECT_EQ(net_->iget(sendComm_, 0, 1800, 512, kRemoteRank, kLocalRank,
                       remoteWindow.opaque(), destinationWindow.opaque(),
                       &request),
            flagcxInvalidArgument);
  EXPECT_EQ(request, nullptr);

  EXPECT_DEREGISTER_MR(sendComm_, sourceMr);
  EXPECT_DEREGISTER_MR(recvComm_, remoteMr);
  EXPECT_DEREGISTER_MR(sendComm_, destinationMr);
}

TEST_F(NetAdaptorLoopback, RequestPoolBackpressureAndRecovery) {
  SKIP_IF_CALLBACK_NULL(net_, getMrInfo);
  SKIP_IF_CALLBACK_NULL(net_, iput);
  SKIP_IF_CALLBACK_NULL(net_, iputBatch);
  SKIP_IF_CALLBACK_NULL(net_, test);
  std::vector<uint8_t> source(1, 0x6d);
  std::vector<uint8_t> remote(1, 0);
  void *sourceMr = nullptr;
  void *remoteMr = nullptr;
  ASSERT_REGISTER_MR(sendComm_, source.data(), source.size(), FLAGCX_PTR_HOST,
                     sourceMr);
  ASSERT_REGISTER_MR(recvComm_, remote.data(), remote.size(), FLAGCX_PTR_HOST,
                     remoteMr);
  TestWindow sourceWindow, remoteWindow;
  ASSERT_EQ(sourceWindow.init(net_, source.data(), source.size(), kLocalRank,
                              sourceMr),
            flagcxSuccess);
  ASSERT_EQ(remoteWindow.init(net_, remote.data(), remote.size(), kRemoteRank,
                              remoteMr),
            flagcxSuccess);

  std::vector<void *> pending(kRequestPoolSize, nullptr);
  for (void *&request : pending) {
    ASSERT_EQ(net_->iput(sendComm_, 0, 0, 1, kLocalRank, kRemoteRank,
                         sourceWindow.opaque(), remoteWindow.opaque(),
                         &request),
              flagcxSuccess);
    ASSERT_NE(request, nullptr);
  }
  const uint64_t offsets[1] = {0};
  const size_t sizes[1] = {1};
  void *batchRequests[1] = {reinterpret_cast<void *>(1)};
  int posted = -1;
  EXPECT_EQ(net_->iputBatch(sendComm_, 1, offsets, offsets, sizes, kLocalRank,
                            kRemoteRank, sourceWindow.opaque(),
                            remoteWindow.opaque(), batchRequests, &posted),
            flagcxInProgress);
  EXPECT_EQ(posted, 0);
  EXPECT_EQ(batchRequests[0], nullptr);
  for (void *request : pending)
    ASSERT_EQ(waitRequest(net_, request), flagcxSuccess);

  void *request = nullptr;
  ASSERT_EQ(net_->iput(sendComm_, 0, 0, 1, kLocalRank, kRemoteRank,
                       sourceWindow.opaque(), remoteWindow.opaque(), &request),
            flagcxSuccess);
  ASSERT_EQ(waitRequest(net_, request), flagcxSuccess);
  EXPECT_EQ(remote[0], source[0]);
  EXPECT_DEREGISTER_MR(sendComm_, sourceMr);
  EXPECT_DEREGISTER_MR(recvComm_, remoteMr);
}

TEST_F(NetAdaptorLoopback, OneSidedRequestsStayOnOrderedQp) {
  SKIP_IF_CALLBACK_NULL(net_, getMrInfo);
  SKIP_IF_CALLBACK_NULL(net_, iput);
  SKIP_IF_CALLBACK_NULL(net_, test);
  if (net_->name == nullptr || strcmp(net_->name, "IB") != 0)
    GTEST_SKIP() << "QP selection introspection is specific to IBRC";
  auto *sendComm = static_cast<flagcxIbSendComm *>(sendComm_);
  if (sendComm->base.nqps < 2)
    GTEST_SKIP()
        << "Set FLAGCX_IB_QPS_PER_CONNECTION=2 to verify stable QP use";

  std::vector<uint8_t> source(2, 0);
  std::vector<uint8_t> remote(2, 0);
  source[0] = 0x35;
  source[1] = 0x7a;
  void *sourceMr = nullptr;
  void *remoteMr = nullptr;
  ASSERT_REGISTER_MR(sendComm_, source.data(), source.size(), FLAGCX_PTR_HOST,
                     sourceMr);
  ASSERT_REGISTER_MR(recvComm_, remote.data(), remote.size(), FLAGCX_PTR_HOST,
                     remoteMr);
  TestWindow sourceWindow, remoteWindow;
  ASSERT_EQ(sourceWindow.init(net_, source.data(), source.size(), kLocalRank,
                              sourceMr),
            flagcxSuccess);
  ASSERT_EQ(remoteWindow.init(net_, remote.data(), remote.size(), kRemoteRank,
                              remoteMr),
            flagcxSuccess);

  int firstQp = sendComm->base.qpIndex;
  for (uint64_t offset = 0; offset < source.size(); ++offset) {
    void *request = nullptr;
    ASSERT_EQ(net_->iput(sendComm_, offset, offset, 1, kLocalRank, kRemoteRank,
                         sourceWindow.opaque(), remoteWindow.opaque(),
                         &request),
              flagcxSuccess);
    ASSERT_EQ(waitRequest(net_, request), flagcxSuccess);
    EXPECT_EQ(sendComm->base.qpIndex, firstQp);
  }
  EXPECT_EQ(remote, source);

  EXPECT_DEREGISTER_MR(sendComm_, sourceMr);
  EXPECT_DEREGISTER_MR(recvComm_, remoteMr);
}

TEST_F(NetAdaptorLoopback, IputSignal) {
  SKIP_IF_CALLBACK_NULL(net_, getMrInfo);
  SKIP_IF_CALLBACK_NULL(net_, iputSignal);
  SKIP_IF_CALLBACK_NULL(net_, test);
  std::vector<uint8_t> source(kBufferSize, 0x3c);
  std::vector<uint8_t> remote(kBufferSize, 0);
  alignas(uint64_t) uint64_t signal = 0;
  void *sourceMr = nullptr;
  void *remoteMr = nullptr;
  void *signalMr = nullptr;
  ASSERT_REGISTER_MR(sendComm_, source.data(), source.size(), FLAGCX_PTR_HOST,
                     sourceMr);
  ASSERT_REGISTER_MR(recvComm_, remote.data(), remote.size(), FLAGCX_PTR_HOST,
                     remoteMr);
  ASSERT_EQ(registerMr(recvComm_, &signal, sizeof(signal), FLAGCX_PTR_HOST,
                       &signalMr, FLAGCX_NET_MR_FLAG_FORCE_SO),
            flagcxSuccess);
  ASSERT_NE(signalMr, nullptr);
  TestWindow sourceWindow, remoteWindow, signalWindow;
  ASSERT_EQ(sourceWindow.init(net_, source.data(), source.size(), kLocalRank,
                              sourceMr),
            flagcxSuccess);
  ASSERT_EQ(remoteWindow.init(net_, remote.data(), remote.size(), kRemoteRank,
                              remoteMr),
            flagcxSuccess);
  ASSERT_EQ(
      signalWindow.init(net_, &signal, sizeof(signal), kRemoteRank, signalMr),
      flagcxSuccess);

  void *request = nullptr;
  ASSERT_EQ(net_->iputSignal(sendComm_, 0, 0, source.size(), kLocalRank,
                             kRemoteRank, sourceWindow.opaque(),
                             remoteWindow.opaque(), 0, signalWindow.opaque(), 7,
                             &request),
            flagcxSuccess);
  ASSERT_EQ(waitRequest(net_, request), flagcxSuccess);
  EXPECT_EQ(source, remote);
  EXPECT_EQ(signal, 7u);

  request = reinterpret_cast<void *>(1);
  EXPECT_EQ(net_->iputSignal(sendComm_, 0, 0, 0, kLocalRank, kRemoteRank,
                             nullptr, nullptr, 1, signalWindow.opaque(), 1,
                             &request),
            flagcxInvalidArgument);
  EXPECT_EQ(request, nullptr);
  EXPECT_DEREGISTER_MR(sendComm_, sourceMr);
  EXPECT_DEREGISTER_MR(recvComm_, remoteMr);
  EXPECT_DEREGISTER_MR(recvComm_, signalMr);
}

TEST_F(NetAdaptorLoopback, BatchWriteTestAndRead) {
  SKIP_IF_CALLBACK_NULL(net_, getMrInfo);
  SKIP_IF_CALLBACK_NULL(net_, iputBatch);
  SKIP_IF_CALLBACK_NULL(net_, testBatch);
  SKIP_IF_CALLBACK_NULL(net_, igetBatch);
  std::vector<uint8_t> source(kBufferSize);
  std::vector<uint8_t> remote(kBufferSize, 0);
  std::vector<uint8_t> destination(kBufferSize, 0);
  for (size_t i = 0; i < source.size(); ++i)
    source[i] = static_cast<uint8_t>(i);
  void *sourceMr = nullptr;
  void *remoteMr = nullptr;
  void *destinationMr = nullptr;
  ASSERT_REGISTER_MR(sendComm_, source.data(), source.size(), FLAGCX_PTR_HOST,
                     sourceMr);
  ASSERT_REGISTER_MR(recvComm_, remote.data(), remote.size(), FLAGCX_PTR_HOST,
                     remoteMr);
  ASSERT_REGISTER_MR(sendComm_, destination.data(), destination.size(),
                     FLAGCX_PTR_HOST, destinationMr);
  TestWindow sourceWindow, remoteWindow, destinationWindow;
  ASSERT_EQ(sourceWindow.init(net_, source.data(), source.size(), kLocalRank,
                              sourceMr),
            flagcxSuccess);
  ASSERT_EQ(remoteWindow.init(net_, remote.data(), remote.size(), kRemoteRank,
                              remoteMr),
            flagcxSuccess);
  ASSERT_EQ(destinationWindow.init(net_, destination.data(), destination.size(),
                                   kLocalRank, destinationMr),
            flagcxSuccess);

  constexpr int count = 3;
  const uint64_t sourceOffsets[count] = {0, 1024, 2048};
  const uint64_t remoteOffsets[count] = {128, 1152, 2176};
  const uint64_t destinationOffsets[count] = {64, 1088, 2112};
  const size_t sizes[count] = {512, 512, 512};
  void *requests[count] = {};
  int posted = 0;
  ASSERT_EQ(net_->iputBatch(sendComm_, count, sourceOffsets, remoteOffsets,
                            sizes, kLocalRank, kRemoteRank,
                            sourceWindow.opaque(), remoteWindow.opaque(),
                            requests, &posted),
            flagcxSuccess);
  ASSERT_EQ(posted, count);
  int doneFlags[count] = {};
  int doneCount = 0;
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  while (doneCount != count && std::chrono::steady_clock::now() < deadline) {
    ASSERT_EQ(net_->testBatch(requests, count, doneFlags, &doneCount),
              flagcxSuccess);
    if (doneCount != count)
      std::this_thread::yield();
  }
  ASSERT_EQ(doneCount, count);
  for (int i = 0; i < count; ++i)
    EXPECT_EQ(memcmp(source.data() + sourceOffsets[i],
                     remote.data() + remoteOffsets[i], sizes[i]),
              0);

  void *readRequest = nullptr;
  ASSERT_EQ(net_->igetBatch(sendComm_, count, remoteOffsets, destinationOffsets,
                            sizes, kRemoteRank, kLocalRank,
                            remoteWindow.opaque(), destinationWindow.opaque(),
                            &readRequest),
            flagcxSuccess);
  ASSERT_NE(readRequest, nullptr);
  ASSERT_EQ(waitRequest(net_, readRequest), flagcxSuccess);
  for (int i = 0; i < count; ++i)
    EXPECT_EQ(memcmp(remote.data() + remoteOffsets[i],
                     destination.data() + destinationOffsets[i], sizes[i]),
              0);

  int zeroDone = -1;
  EXPECT_EQ(net_->testBatch(nullptr, 0, nullptr, &zeroDone), flagcxSuccess);
  EXPECT_EQ(zeroDone, 0);
  int zeroPosted = -1;
  EXPECT_EQ(net_->iputBatch(sendComm_, 0, nullptr, nullptr, nullptr, kLocalRank,
                            kRemoteRank, nullptr, nullptr, nullptr,
                            &zeroPosted),
            flagcxSuccess);
  EXPECT_EQ(zeroPosted, 0);
  EXPECT_DEREGISTER_MR(sendComm_, sourceMr);
  EXPECT_DEREGISTER_MR(recvComm_, remoteMr);
  EXPECT_DEREGISTER_MR(sendComm_, destinationMr);
}

#undef SKIP_IF_CALLBACK_NULL
#undef ASSERT_REGISTER_MR
#undef EXPECT_DEREGISTER_MR

} // namespace
