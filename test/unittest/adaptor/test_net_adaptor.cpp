/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Contract and loopback tests for the build-selected RDMA network adaptor.
 * USE_ACCL_BAREX=1 selects BAREX through the normal adaptor registry, so
 * these tests exercise the same vtable used by the FlagCX runtime.
 ************************************************************************/

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <mutex>
#include <thread>
#include <unistd.h>
#include <vector>

#include "adaptor.h"
#include "flagcx.h"
#include "flagcx_net.h"
#include "flagcx_net_adaptor.h"
#include "onesided.h"

namespace {

constexpr size_t kBufferSize = 4096;
constexpr int kLocalRank = 0;
constexpr int kRemoteRank = 1;
constexpr auto kTimeout = std::chrono::seconds(30);

#define SKIP_IF_NET_CALLBACK_NULL(net, callback)                               \
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

struct LoopbackConnectionResult {
  flagcxResult_t connectResult = flagcxSystemError;
  flagcxResult_t acceptResult = flagcxSystemError;
  void *sendComm = nullptr;
  void *recvComm = nullptr;
};

LoopbackConnectionResult
establishLoopbackConnection(struct flagcxNetAdaptor *net, int dev,
                            const char *listenHandle, void *listenComm) {
  LoopbackConnectionResult result;
  if (net == nullptr || net->connect == nullptr || net->accept == nullptr ||
      listenHandle == nullptr || listenComm == nullptr) {
    result.connectResult = flagcxInvalidArgument;
    result.acceptResult = flagcxInvalidArgument;
    return result;
  }

  char connectHandle[FLAGCX_NET_HANDLE_MAXSIZE] = {};
  memcpy(connectHandle, listenHandle, sizeof(connectHandle));

  /* IBRC and BAREX advance the two sides of the handshake independently.
     Running connect and accept serially makes progress depend on scheduler and
     TCP timing: one side may need the peer to advance before it can return a
     completed comm.  Keep one thread per endpoint, just like the real proxy
     paths, and never call the same endpoint concurrently from two threads. */
  std::atomic<bool> cancelled(false);
  std::mutex completionMutex;
  std::condition_variable completionCv;
  int completedEndpoints = 0;
  const auto markEndpointComplete = [&]() {
    {
      std::lock_guard<std::mutex> lock(completionMutex);
      ++completedEndpoints;
    }
    completionCv.notify_one();
  };

  std::thread connectThread([&]() {
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
    markEndpointComplete();
  });

  std::thread acceptThread([&]() {
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
    markEndpointComplete();
  });

  /* A callback that is already blocked cannot observe cancelled.  Bound the
     whole helper externally so a broken transport handshake fails the test
     process instead of hanging the CI runner indefinitely. */
  {
    std::unique_lock<std::mutex> lock(completionMutex);
    if (!completionCv.wait_for(lock, kTimeout + std::chrono::seconds(1),
                               [&]() { return completedEndpoints == 2; })) {
      fprintf(stderr, "net adaptor loopback handshake timed out\n");
      std::abort();
    }
  }
  connectThread.join();
  acceptThread.join();
  return result;
}

struct TestWindow {
  struct flagcxOneSideHandleInfo info = {};
  uintptr_t baseVas[2] = {};
  size_t regionSizes[2] = {};
  uint32_t rkeys[2] = {};
  uint32_t lkeys[2] = {};
  struct flagcxNetMrInfo mrInfos[2] = {};

  flagcxResult_t init(struct flagcxNetAdaptor *net, void *buffer, size_t size,
                      int rank, void *mrHandle) {
    if (net == nullptr || net->getMrInfo == nullptr || buffer == nullptr ||
        rank < 0 || rank >= 2 || mrHandle == nullptr)
      return flagcxInvalidArgument;
    flagcxResult_t result = net->getMrInfo(mrHandle, &mrInfos[rank]);
    if (result != flagcxSuccess)
      return result;
    baseVas[rank] = reinterpret_cast<uintptr_t>(buffer);
    regionSizes[rank] = size;
    rkeys[rank] = mrInfos[rank].rkeys[0];
    lkeys[rank] = mrInfos[rank].lkeys[0];
    info.baseVas = baseVas;
    info.regionSize = size;
    info.regionSizes = regionSizes;
    info.rkeys = rkeys;
    info.lkeys = lkeys;
    info.mrInfos = mrInfos;
    info.localMrHandle = mrHandle;
    info.nRanks = 2;
    return flagcxSuccess;
  }

  void **opaque() { return reinterpret_cast<void **>(&info); }
};

class NetAdaptorLoopback : public ::testing::Test {
protected:
  void SetUp() override {
    net_ = getNetAdaptor(RDMA);
    ASSERT_NE(net_, nullptr);
    SKIP_IF_NET_CALLBACK_NULL(net_, init);
    SKIP_IF_NET_CALLBACK_NULL(net_, devices);
    SKIP_IF_NET_CALLBACK_NULL(net_, listen);
    SKIP_IF_NET_CALLBACK_NULL(net_, connect);
    SKIP_IF_NET_CALLBACK_NULL(net_, accept);
    SKIP_IF_NET_CALLBACK_NULL(net_, closeSend);
    SKIP_IF_NET_CALLBACK_NULL(net_, closeRecv);
    SKIP_IF_NET_CALLBACK_NULL(net_, closeListen);
    ASSERT_EQ(net_->init(), flagcxSuccess);
    ASSERT_EQ(net_->devices(&nDevs_), flagcxSuccess);
    ASSERT_GT(nDevs_, 0);

    ASSERT_EQ(net_->listen(0, handle_, &listenComm_), flagcxSuccess);
    ASSERT_NE(listenComm_, nullptr);

    const LoopbackConnectionResult connection =
        establishLoopbackConnection(net_, 0, handle_, listenComm_);
    sendComm_ = connection.sendComm;
    recvComm_ = connection.recvComm;
    ASSERT_EQ(connection.connectResult, flagcxSuccess);
    ASSERT_EQ(connection.acceptResult, flagcxSuccess);
    ASSERT_NE(sendComm_, nullptr) << "connect timed out";
    ASSERT_NE(recvComm_, nullptr) << "accept timed out";
  }

  void TearDown() override {
    if (net_ == nullptr)
      return;
    if (deviceBuffer_ != nullptr && device_ != nullptr) {
      EXPECT_EQ(device_->deviceFree(deviceBuffer_, flagcxMemDevice, nullptr),
                flagcxSuccess);
      deviceBuffer_ = nullptr;
    }
    if (device_ != nullptr) {
      EXPECT_EQ(flagcxDeviceHandleFree(device_), flagcxSuccess);
      device_ = nullptr;
    }
    if (sendComm_ != nullptr && net_->closeSend != nullptr) {
      EXPECT_EQ(net_->closeSend(sendComm_), flagcxSuccess);
    }
    if (recvComm_ != nullptr && net_->closeRecv != nullptr) {
      EXPECT_EQ(net_->closeRecv(recvComm_), flagcxSuccess);
    }
    if (listenComm_ != nullptr && net_->closeListen != nullptr) {
      EXPECT_EQ(net_->closeListen(listenComm_), flagcxSuccess);
    }
  }

  flagcxResult_t registerMr(void *comm, void *buffer, size_t size, int type,
                            void **mrHandle,
                            int mrFlags = FLAGCX_NET_MR_FLAG_NONE) {
    if (net_ == nullptr || net_->regMr == nullptr || comm == nullptr ||
        buffer == nullptr || size == 0 || mrHandle == nullptr)
      return flagcxInvalidArgument;
    *mrHandle = nullptr;
    flagcxResult_t result =
        net_->regMr(comm, buffer, size, type, mrFlags, mrHandle);
    if (result != flagcxSuccess)
      return result;
    if (*mrHandle == nullptr)
      return flagcxInternalError;
    return flagcxSuccess;
  }

  flagcxResult_t deregisterMr(void *comm, void *mrHandle) {
    if (net_ == nullptr || net_->deregMr == nullptr || comm == nullptr ||
        mrHandle == nullptr)
      return flagcxInvalidArgument;
    return net_->deregMr(comm, mrHandle);
  }

  struct flagcxNetAdaptor *net_ = nullptr;
  int nDevs_ = 0;
  char handle_[FLAGCX_NET_HANDLE_MAXSIZE] = {};
  void *listenComm_ = nullptr;
  void *sendComm_ = nullptr;
  void *recvComm_ = nullptr;
  flagcxDeviceHandle_t device_ = nullptr;
  void *deviceBuffer_ = nullptr;
};

class NetAdaptorMemory : public NetAdaptorLoopback {};
class NetAdaptorTwoSided : public NetAdaptorLoopback {};
class NetAdaptorOneSided : public NetAdaptorLoopback {};
class NetAdaptorBatch : public NetAdaptorLoopback {};

#define ASSERT_REGISTER_MR(comm, buffer, size, type, mrHandle)                 \
  do {                                                                         \
    ASSERT_EQ(registerMr((comm), (buffer), (size), (type), &(mrHandle)),       \
              flagcxSuccess);                                                  \
    ASSERT_NE((mrHandle), nullptr);                                            \
  } while (0)

#define EXPECT_DEREGISTER_MR(comm, mrHandle)                                   \
  do {                                                                         \
    EXPECT_EQ(deregisterMr((comm), (mrHandle)), flagcxSuccess);                \
    (mrHandle) = nullptr;                                                      \
  } while (0)

class NetAdaptorReusableListener : public ::testing::Test {
protected:
  void SetUp() override {
    net_ = getNetAdaptor(RDMA);
    ASSERT_NE(net_, nullptr);
    SKIP_IF_NET_CALLBACK_NULL(net_, init);
    SKIP_IF_NET_CALLBACK_NULL(net_, devices);
    SKIP_IF_NET_CALLBACK_NULL(net_, listen);
    SKIP_IF_NET_CALLBACK_NULL(net_, connect);
    SKIP_IF_NET_CALLBACK_NULL(net_, accept);
    SKIP_IF_NET_CALLBACK_NULL(net_, closeSend);
    SKIP_IF_NET_CALLBACK_NULL(net_, closeRecv);
    SKIP_IF_NET_CALLBACK_NULL(net_, closeListen);
    ASSERT_EQ(net_->init(), flagcxSuccess);
    ASSERT_EQ(net_->devices(&nDevs_), flagcxSuccess);
    ASSERT_GT(nDevs_, 0);
    ASSERT_EQ(net_->listen(0, listenHandle_, &listenComm_), flagcxSuccess);
    ASSERT_NE(listenComm_, nullptr);
  }

  void TearDown() override {
    if (net_ == nullptr)
      return;
    for (void *comm : sendComms_) {
      if (comm != nullptr && net_->closeSend != nullptr) {
        EXPECT_EQ(net_->closeSend(comm), flagcxSuccess);
      }
    }
    for (void *comm : recvComms_) {
      if (comm != nullptr && net_->closeRecv != nullptr) {
        EXPECT_EQ(net_->closeRecv(comm), flagcxSuccess);
      }
    }
    if (listenComm_ != nullptr && net_->closeListen != nullptr) {
      EXPECT_EQ(net_->closeListen(listenComm_), flagcxSuccess);
    }
  }

  struct flagcxNetAdaptor *net_ = nullptr;
  int nDevs_ = 0;
  char listenHandle_[FLAGCX_NET_HANDLE_MAXSIZE] = {};
  void *listenComm_ = nullptr;
  std::vector<void *> sendComms_;
  std::vector<void *> recvComms_;
};

TEST(NetAdaptorInterface, AdaptorIsAvailable) {
  struct flagcxNetAdaptor *net = getNetAdaptor(RDMA);
  ASSERT_NE(net, nullptr);
  EXPECT_NE(net->name, nullptr);
}

TEST(NetAdaptorInterface, Init) {
  struct flagcxNetAdaptor *net = getNetAdaptor(RDMA);
  ASSERT_NE(net, nullptr);
  SKIP_IF_NET_CALLBACK_NULL(net, init);
  EXPECT_EQ(net->init(), flagcxSuccess);
}

TEST(NetAdaptorInterface, Devices) {
  struct flagcxNetAdaptor *net = getNetAdaptor(RDMA);
  ASSERT_NE(net, nullptr);
  SKIP_IF_NET_CALLBACK_NULL(net, init);
  SKIP_IF_NET_CALLBACK_NULL(net, devices);
  ASSERT_EQ(net->init(), flagcxSuccess);
  int ndev = 0;
  EXPECT_EQ(net->devices(&ndev), flagcxSuccess);
  EXPECT_GT(ndev, 0);
}

TEST(NetAdaptorInterface, GetProperties) {
  struct flagcxNetAdaptor *net = getNetAdaptor(RDMA);
  ASSERT_NE(net, nullptr);
  SKIP_IF_NET_CALLBACK_NULL(net, init);
  SKIP_IF_NET_CALLBACK_NULL(net, devices);
  SKIP_IF_NET_CALLBACK_NULL(net, getProperties);
  ASSERT_EQ(net->init(), flagcxSuccess);
  int ndev = 0;
  ASSERT_EQ(net->devices(&ndev), flagcxSuccess);
  ASSERT_GT(ndev, 0);
  for (int dev = 0; dev < ndev; ++dev) {
    flagcxNetProperties_t properties = {};
    ASSERT_EQ(net->getProperties(dev, &properties), flagcxSuccess);
    ASSERT_NE(properties.name, nullptr);
    EXPECT_GT(properties.speed, 0);
  }
}

TEST(NetAdaptorInterface, GetDevFromName) {
  struct flagcxNetAdaptor *net = getNetAdaptor(RDMA);
  ASSERT_NE(net, nullptr);
  SKIP_IF_NET_CALLBACK_NULL(net, init);
  SKIP_IF_NET_CALLBACK_NULL(net, devices);
  SKIP_IF_NET_CALLBACK_NULL(net, getProperties);
  SKIP_IF_NET_CALLBACK_NULL(net, getDevFromName);
  ASSERT_EQ(net->init(), flagcxSuccess);
  int ndev = 0;
  ASSERT_EQ(net->devices(&ndev), flagcxSuccess);
  ASSERT_GT(ndev, 0);
  for (int dev = 0; dev < ndev; ++dev) {
    flagcxNetProperties_t properties = {};
    ASSERT_EQ(net->getProperties(dev, &properties), flagcxSuccess);
    ASSERT_NE(properties.name, nullptr);
    int foundDev = -1;
    EXPECT_EQ(net->getDevFromName(properties.name, &foundDev), flagcxSuccess);
    EXPECT_EQ(foundDev, dev);
  }
}

TEST_F(NetAdaptorReusableListener, AcceptsSequentialConnections) {
  for (int connection = 0; connection < 2; ++connection) {
    const LoopbackConnectionResult result =
        establishLoopbackConnection(net_, 0, listenHandle_, listenComm_);
    sendComms_.push_back(result.sendComm);
    recvComms_.push_back(result.recvComm);
    ASSERT_EQ(result.connectResult, flagcxSuccess)
        << "connect failed at iteration " << connection;
    ASSERT_EQ(result.acceptResult, flagcxSuccess)
        << "accept failed at iteration " << connection;
    ASSERT_NE(result.sendComm, nullptr)
        << "connect timed out at iteration " << connection;
    ASSERT_NE(result.recvComm, nullptr)
        << "accept timed out at iteration " << connection;
  }
}

TEST_F(NetAdaptorMemory, RegisterHostMr) {
  SKIP_IF_NET_CALLBACK_NULL(net_, regMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, deregMr);
  std::vector<uint8_t> buffer(kBufferSize);
  void *mrHandle = nullptr;
  ASSERT_REGISTER_MR(sendComm_, buffer.data(), buffer.size(), FLAGCX_PTR_HOST,
                     mrHandle);

  EXPECT_DEREGISTER_MR(sendComm_, mrHandle);
}

TEST_F(NetAdaptorMemory, ExportMrMetadata) {
  SKIP_IF_NET_CALLBACK_NULL(net_, regMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, deregMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, getMrInfo);
  std::vector<uint8_t> buffer(kBufferSize);
  void *mrHandle = nullptr;
  ASSERT_REGISTER_MR(sendComm_, buffer.data(), buffer.size(), FLAGCX_PTR_HOST,
                     mrHandle);

  struct flagcxNetMrInfo info = {};
  ASSERT_EQ(net_->getMrInfo(mrHandle, &info), flagcxSuccess);
  EXPECT_GT(info.nKeys, 0u);
  EXPECT_LE(info.nKeys, static_cast<uint32_t>(FLAGCX_NET_MAX_MR_KEYS));

  EXPECT_DEREGISTER_MR(sendComm_, mrHandle);
}

TEST_F(NetAdaptorMemory, RegisterGpuMr) {
  SKIP_IF_NET_CALLBACK_NULL(net_, getProperties);
  SKIP_IF_NET_CALLBACK_NULL(net_, regMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, deregMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, getMrInfo);
  flagcxNetProperties_t properties = {};
  ASSERT_EQ(net_->getProperties(0, &properties), flagcxSuccess);
  if ((properties.ptrSupport & FLAGCX_PTR_CUDA) == 0)
    GTEST_SKIP() << "Selected net adaptor does not advertise GPU MR support";

  ASSERT_EQ(flagcxDeviceHandleInit(&device_), flagcxSuccess);
  ASSERT_NE(device_, nullptr);
  ASSERT_EQ(device_->setDevice(0), flagcxSuccess);
  ASSERT_EQ(device_->deviceMalloc(&deviceBuffer_, kBufferSize, flagcxMemDevice,
                                  nullptr),
            flagcxSuccess);
  ASSERT_NE(deviceBuffer_, nullptr);

  void *mrHandle = nullptr;
  ASSERT_REGISTER_MR(sendComm_, deviceBuffer_, kBufferSize, FLAGCX_PTR_CUDA,
                     mrHandle);
  struct flagcxNetMrInfo info = {};
  EXPECT_EQ(net_->getMrInfo(mrHandle, &info), flagcxSuccess);
  EXPECT_GT(info.nKeys, 0u);
  EXPECT_DEREGISTER_MR(sendComm_, mrHandle);
}

TEST_F(NetAdaptorMemory, RegMrDmaBufRegistration) {
  SKIP_IF_NET_CALLBACK_NULL(net_, getProperties);
  SKIP_IF_NET_CALLBACK_NULL(net_, regMrDmaBuf);
  SKIP_IF_NET_CALLBACK_NULL(net_, deregMr);

  flagcxNetProperties_t properties = {};
  ASSERT_EQ(net_->getProperties(0, &properties), flagcxSuccess);
  if ((properties.ptrSupport & FLAGCX_PTR_DMABUF) == 0)
    GTEST_SKIP() << "Selected net adaptor does not advertise DMA-BUF support";

  ASSERT_NE(deviceAdaptor, nullptr);
  ASSERT_NE(deviceAdaptor->setDevice, nullptr);
  if (deviceAdaptor->dmaSupport == nullptr ||
      deviceAdaptor->gdrMemAlloc == nullptr ||
      deviceAdaptor->gdrMemFree == nullptr ||
      deviceAdaptor->getHandleForAddressRange == nullptr) {
    GTEST_SKIP() << "Selected device adaptor cannot export DMA-BUF memory";
  }
  ASSERT_EQ(deviceAdaptor->setDevice(0), flagcxSuccess);

  bool dmaBufferSupport = false;
  ASSERT_EQ(deviceAdaptor->dmaSupport(&dmaBufferSupport), flagcxSuccess);
  if (!dmaBufferSupport)
    GTEST_SKIP() << "Selected device does not support DMA-BUF export";

  void *buffer = nullptr;
  ASSERT_EQ(deviceAdaptor->gdrMemAlloc(&buffer, kBufferSize, nullptr),
            flagcxSuccess);
  ASSERT_NE(buffer, nullptr);
  int dmaBufFd = -1;
  ASSERT_EQ(deviceAdaptor->getHandleForAddressRange(&dmaBufFd, buffer,
                                                    kBufferSize, 0),
            flagcxSuccess);
  ASSERT_GE(dmaBufFd, 0);

  void *mrHandle = nullptr;
  EXPECT_EQ(net_->regMrDmaBuf(sendComm_, buffer, kBufferSize, FLAGCX_PTR_DMABUF,
                              0, dmaBufFd, FLAGCX_NET_MR_FLAG_NONE, &mrHandle),
            flagcxSuccess);
  EXPECT_NE(mrHandle, nullptr);
  if (mrHandle != nullptr) {
    EXPECT_EQ(net_->deregMr(sendComm_, mrHandle), flagcxSuccess);
  }
  close(dmaBufFd);
  EXPECT_EQ(deviceAdaptor->gdrMemFree(buffer, nullptr), flagcxSuccess);
}

TEST_F(NetAdaptorTwoSided, SendRecv) {
  SKIP_IF_NET_CALLBACK_NULL(net_, regMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, deregMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, isend);
  SKIP_IF_NET_CALLBACK_NULL(net_, irecv);
  SKIP_IF_NET_CALLBACK_NULL(net_, test);
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
  ASSERT_NE(sendRequest, nullptr) << "isend timed out waiting for recv CTS";

  int sendSize = 0;
  int recvSize = 0;
  ASSERT_EQ(waitRequest(net_, sendRequest, &sendSize), flagcxSuccess);
  ASSERT_EQ(waitRequest(net_, recvRequest, &recvSize), flagcxSuccess);
  EXPECT_EQ(sendSize, static_cast<int>(source.size()));
  EXPECT_EQ(recvSize, static_cast<int>(destination.size()));
  EXPECT_EQ(source, destination);

  EXPECT_DEREGISTER_MR(sendComm_, sourceMr);
  EXPECT_DEREGISTER_MR(recvComm_, destinationMr);
}

TEST_F(NetAdaptorTwoSided, Flush) {
  SKIP_IF_NET_CALLBACK_NULL(net_, regMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, deregMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, iflush);
  SKIP_IF_NET_CALLBACK_NULL(net_, test);
  std::vector<uint8_t> destination(kBufferSize, 0);
  void *destinationMr = nullptr;
  ASSERT_REGISTER_MR(recvComm_, destination.data(), destination.size(),
                     FLAGCX_PTR_HOST, destinationMr);

  void *recvData[1] = {destination.data()};
  int flushSizes[1] = {static_cast<int>(destination.size())};
  void *recvMrs[1] = {destinationMr};
  void *flushRequest = nullptr;
  ASSERT_EQ(
      net_->iflush(recvComm_, 1, recvData, flushSizes, recvMrs, &flushRequest),
      flagcxSuccess);
  if (flushRequest != nullptr) {
    EXPECT_EQ(waitRequest(net_, flushRequest), flagcxSuccess);
  }

  EXPECT_DEREGISTER_MR(recvComm_, destinationMr);
}

TEST_F(NetAdaptorTwoSided, TestNullRequest) {
  SKIP_IF_NET_CALLBACK_NULL(net_, test);
  int nullDone = 0;
  EXPECT_EQ(net_->test(nullptr, &nullDone, nullptr), flagcxSuccess);
  EXPECT_EQ(nullDone, 1);
}

TEST_F(NetAdaptorOneSided, Iput) {
  SKIP_IF_NET_CALLBACK_NULL(net_, regMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, deregMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, getMrInfo);
  SKIP_IF_NET_CALLBACK_NULL(net_, iput);
  SKIP_IF_NET_CALLBACK_NULL(net_, test);
  std::vector<uint8_t> putSource(kBufferSize, 0xa5);
  std::vector<uint8_t> remoteBuffer(kBufferSize, 0);
  void *putSourceMr = nullptr;
  void *remoteMr = nullptr;
  ASSERT_REGISTER_MR(sendComm_, putSource.data(), putSource.size(),
                     FLAGCX_PTR_HOST, putSourceMr);
  ASSERT_REGISTER_MR(recvComm_, remoteBuffer.data(), remoteBuffer.size(),
                     FLAGCX_PTR_HOST, remoteMr);

  TestWindow putSourceWindow;
  TestWindow remoteWindow;
  ASSERT_EQ(putSourceWindow.init(net_, putSource.data(), putSource.size(),
                                 kLocalRank, putSourceMr),
            flagcxSuccess);
  ASSERT_EQ(remoteWindow.init(net_, remoteBuffer.data(), remoteBuffer.size(),
                              kRemoteRank, remoteMr),
            flagcxSuccess);

  void *request = nullptr;
  ASSERT_EQ(net_->iput(sendComm_, 0, 0, putSource.size(), kLocalRank,
                       kRemoteRank, putSourceWindow.opaque(),
                       remoteWindow.opaque(), &request),
            flagcxSuccess);
  ASSERT_NE(request, nullptr);
  ASSERT_EQ(waitRequest(net_, request), flagcxSuccess);
  EXPECT_EQ(putSource, remoteBuffer);

  EXPECT_DEREGISTER_MR(sendComm_, putSourceMr);
  EXPECT_DEREGISTER_MR(recvComm_, remoteMr);
}

TEST_F(NetAdaptorOneSided, Iget) {
  SKIP_IF_NET_CALLBACK_NULL(net_, regMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, deregMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, getMrInfo);
  SKIP_IF_NET_CALLBACK_NULL(net_, iget);
  SKIP_IF_NET_CALLBACK_NULL(net_, test);
  std::vector<uint8_t> remoteBuffer(kBufferSize, 0xa5);
  std::vector<uint8_t> getDestination(kBufferSize, 0);
  void *remoteMr = nullptr;
  void *getDestinationMr = nullptr;
  ASSERT_REGISTER_MR(recvComm_, remoteBuffer.data(), remoteBuffer.size(),
                     FLAGCX_PTR_HOST, remoteMr);
  ASSERT_REGISTER_MR(sendComm_, getDestination.data(), getDestination.size(),
                     FLAGCX_PTR_HOST, getDestinationMr);

  TestWindow remoteWindow;
  TestWindow getDestinationWindow;
  ASSERT_EQ(remoteWindow.init(net_, remoteBuffer.data(), remoteBuffer.size(),
                              kRemoteRank, remoteMr),
            flagcxSuccess);
  ASSERT_EQ(getDestinationWindow.init(net_, getDestination.data(),
                                      getDestination.size(), kLocalRank,
                                      getDestinationMr),
            flagcxSuccess);

  void *request = nullptr;
  ASSERT_EQ(net_->iget(sendComm_, 0, 0, remoteBuffer.size(), kRemoteRank,
                       kLocalRank, remoteWindow.opaque(),
                       getDestinationWindow.opaque(), &request),
            flagcxSuccess);
  ASSERT_NE(request, nullptr);
  ASSERT_EQ(waitRequest(net_, request), flagcxSuccess);
  EXPECT_EQ(remoteBuffer, getDestination);

  EXPECT_DEREGISTER_MR(recvComm_, remoteMr);
  EXPECT_DEREGISTER_MR(sendComm_, getDestinationMr);
}

TEST_F(NetAdaptorOneSided, RequestPoolBackpressureAndRecovery) {
  SKIP_IF_NET_CALLBACK_NULL(net_, regMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, deregMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, getMrInfo);
  SKIP_IF_NET_CALLBACK_NULL(net_, iput);
  SKIP_IF_NET_CALLBACK_NULL(net_, iget);
  SKIP_IF_NET_CALLBACK_NULL(net_, test);

  std::vector<uint8_t> source(kBufferSize, 0x6d);
  std::vector<uint8_t> remote(kBufferSize, 0);
  std::vector<uint8_t> destination(kBufferSize, 0);
  void *sourceMr = nullptr;
  void *remoteMr = nullptr;
  void *destinationMr = nullptr;
  ASSERT_REGISTER_MR(sendComm_, source.data(), source.size(), FLAGCX_PTR_HOST,
                     sourceMr);
  ASSERT_REGISTER_MR(recvComm_, remote.data(), remote.size(), FLAGCX_PTR_HOST,
                     remoteMr);
  ASSERT_REGISTER_MR(sendComm_, destination.data(), destination.size(),
                     FLAGCX_PTR_HOST, destinationMr);

  TestWindow sourceWindow;
  TestWindow remoteWindow;
  TestWindow destinationWindow;
  ASSERT_EQ(sourceWindow.init(net_, source.data(), source.size(), kLocalRank,
                              sourceMr),
            flagcxSuccess);
  ASSERT_EQ(remoteWindow.init(net_, remote.data(), remote.size(), kRemoteRank,
                              remoteMr),
            flagcxSuccess);
  ASSERT_EQ(destinationWindow.init(net_, destination.data(), destination.size(),
                                   kLocalRank, destinationMr),
            flagcxSuccess);

  // BAREX and IB both expose 256 request slots per send comm. Completion alone
  // does not recycle a slot; test() must retire each request first.
  constexpr int requestPoolSize = 256;
  std::vector<void *> requests(requestPoolSize, nullptr);
  for (int i = 0; i < requestPoolSize; ++i) {
    ASSERT_EQ(net_->iput(sendComm_, 0, 0, 1, kLocalRank, kRemoteRank,
                         sourceWindow.opaque(), remoteWindow.opaque(),
                         &requests[i]),
              flagcxSuccess);
    ASSERT_NE(requests[i], nullptr) << "request slot " << i;
  }

  void *request = reinterpret_cast<void *>(1);
  EXPECT_EQ(net_->iput(sendComm_, 0, 0, 1, kLocalRank, kRemoteRank,
                       sourceWindow.opaque(), remoteWindow.opaque(), &request),
            flagcxInternalError);
  EXPECT_EQ(request, nullptr);

  request = reinterpret_cast<void *>(1);
  EXPECT_EQ(net_->iget(sendComm_, 0, 0, 1, kRemoteRank, kLocalRank,
                       remoteWindow.opaque(), destinationWindow.opaque(),
                       &request),
            flagcxInternalError);
  EXPECT_EQ(request, nullptr);

  if (net_->igetBatch != nullptr) {
    const uint64_t offsets[1] = {0};
    const size_t sizes[1] = {1};
    request = reinterpret_cast<void *>(1);
    EXPECT_EQ(net_->igetBatch(sendComm_, 1, offsets, offsets, sizes,
                              kRemoteRank, kLocalRank, remoteWindow.opaque(),
                              destinationWindow.opaque(), &request),
              flagcxInternalError);
    EXPECT_EQ(request, nullptr);
  }

  for (void *pending : requests)
    ASSERT_EQ(waitRequest(net_, pending), flagcxSuccess);

  request = nullptr;
  ASSERT_EQ(net_->iput(sendComm_, 0, 0, 1, kLocalRank, kRemoteRank,
                       sourceWindow.opaque(), remoteWindow.opaque(), &request),
            flagcxSuccess);
  ASSERT_NE(request, nullptr);
  ASSERT_EQ(waitRequest(net_, request), flagcxSuccess);
  EXPECT_EQ(remote[0], source[0]);

  EXPECT_DEREGISTER_MR(sendComm_, sourceMr);
  EXPECT_DEREGISTER_MR(recvComm_, remoteMr);
  EXPECT_DEREGISTER_MR(sendComm_, destinationMr);
}

TEST_F(NetAdaptorOneSided, PerRankRegionBounds) {
  SKIP_IF_NET_CALLBACK_NULL(net_, regMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, deregMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, getMrInfo);
  SKIP_IF_NET_CALLBACK_NULL(net_, iput);
  SKIP_IF_NET_CALLBACK_NULL(net_, iget);
  SKIP_IF_NET_CALLBACK_NULL(net_, test);

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

  TestWindow sourceWindow;
  TestWindow remoteWindow;
  TestWindow destinationWindow;
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
  ASSERT_NE(request, nullptr);
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
  ASSERT_EQ(net_->iget(sendComm_, 3000, 1000, 512, kRemoteRank, kLocalRank,
                       remoteWindow.opaque(), destinationWindow.opaque(),
                       &request),
            flagcxSuccess);
  ASSERT_NE(request, nullptr);
  ASSERT_EQ(waitRequest(net_, request), flagcxSuccess);
  EXPECT_EQ(memcmp(remote.data() + 3000, destination.data() + 1000, 512), 0);

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

  request = reinterpret_cast<void *>(1);
  EXPECT_EQ(net_->iget(sendComm_, 0, 0, 1, 2, kLocalRank, remoteWindow.opaque(),
                       destinationWindow.opaque(), &request),
            flagcxInvalidArgument);
  EXPECT_EQ(request, nullptr);

  if (net_->iputBatch != nullptr) {
    const uint64_t sourceOffsets[1] = {800};
    const uint64_t remoteOffsets[1] = {0};
    const size_t sizes[1] = {512};
    void *requests[1] = {reinterpret_cast<void *>(1)};
    int posted = -1;
    EXPECT_EQ(net_->iputBatch(sendComm_, 1, sourceOffsets, remoteOffsets, sizes,
                              kLocalRank, kRemoteRank, sourceWindow.opaque(),
                              remoteWindow.opaque(), requests, &posted),
              flagcxInvalidArgument);
    EXPECT_EQ(posted, 0);
    EXPECT_EQ(requests[0], nullptr);
  }

  if (net_->iputSignal != nullptr) {
    uint64_t signal = 0;
    void *signalMr = nullptr;
    ASSERT_EQ(registerMr(recvComm_, &signal, sizeof(signal), FLAGCX_PTR_HOST,
                         &signalMr, FLAGCX_NET_MR_FLAG_FORCE_SO),
              flagcxSuccess);
    ASSERT_NE(signalMr, nullptr);

    TestWindow signalWindow;
    ASSERT_EQ(
        signalWindow.init(net_, &signal, sizeof(signal), kRemoteRank, signalMr),
        flagcxSuccess);

    request = reinterpret_cast<void *>(1);
    EXPECT_EQ(net_->iputSignal(sendComm_, 800, 0, 512, kLocalRank, kRemoteRank,
                               sourceWindow.opaque(), remoteWindow.opaque(), 0,
                               signalWindow.opaque(), 1, &request),
              flagcxInvalidArgument);
    EXPECT_EQ(request, nullptr);

    request = reinterpret_cast<void *>(1);
    EXPECT_EQ(net_->iputSignal(sendComm_, 0, 0, 0, kLocalRank, kRemoteRank,
                               nullptr, nullptr, 1, signalWindow.opaque(), 1,
                               &request),
              flagcxInvalidArgument);
    EXPECT_EQ(request, nullptr);

    EXPECT_DEREGISTER_MR(recvComm_, signalMr);
  }

  EXPECT_DEREGISTER_MR(sendComm_, sourceMr);
  EXPECT_DEREGISTER_MR(recvComm_, remoteMr);
  EXPECT_DEREGISTER_MR(sendComm_, destinationMr);
}

TEST_F(NetAdaptorOneSided, IputSignal) {
  SKIP_IF_NET_CALLBACK_NULL(net_, regMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, deregMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, getMrInfo);
  SKIP_IF_NET_CALLBACK_NULL(net_, iputSignal);
  SKIP_IF_NET_CALLBACK_NULL(net_, test);

  std::vector<uint8_t> source(kBufferSize, 0x3c);
  std::vector<uint8_t> remote(kBufferSize, 0);
  uint64_t signal = 0;
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

  TestWindow sourceWindow;
  TestWindow remoteWindow;
  TestWindow signalWindow;
  ASSERT_EQ(sourceWindow.init(net_, source.data(), source.size(), kLocalRank,
                              sourceMr),
            flagcxSuccess);
  ASSERT_EQ(remoteWindow.init(net_, remote.data(), remote.size(), kRemoteRank,
                              remoteMr),
            flagcxSuccess);
  ASSERT_EQ(
      signalWindow.init(net_, &signal, sizeof(signal), kRemoteRank, signalMr),
      flagcxSuccess);

  constexpr uint64_t increment = 7;
  void *request = nullptr;
  ASSERT_EQ(net_->iputSignal(sendComm_, 0, 0, source.size(), kLocalRank,
                             kRemoteRank, sourceWindow.opaque(),
                             remoteWindow.opaque(), 0, signalWindow.opaque(),
                             increment, &request),
            flagcxSuccess);
  ASSERT_NE(request, nullptr);
  ASSERT_EQ(waitRequest(net_, request), flagcxSuccess);
  EXPECT_EQ(source, remote);
  EXPECT_EQ(signal, increment);

  EXPECT_DEREGISTER_MR(sendComm_, sourceMr);
  EXPECT_DEREGISTER_MR(recvComm_, remoteMr);
  EXPECT_DEREGISTER_MR(recvComm_, signalMr);
}

TEST_F(NetAdaptorBatch, IputBatch) {
  SKIP_IF_NET_CALLBACK_NULL(net_, regMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, deregMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, getMrInfo);
  SKIP_IF_NET_CALLBACK_NULL(net_, iputBatch);
  SKIP_IF_NET_CALLBACK_NULL(net_, test);
  std::vector<uint8_t> source(kBufferSize);
  std::vector<uint8_t> remote(kBufferSize, 0);
  for (size_t i = 0; i < source.size(); ++i)
    source[i] = static_cast<uint8_t>(i);

  void *sourceMr = nullptr;
  void *remoteMr = nullptr;
  ASSERT_REGISTER_MR(sendComm_, source.data(), source.size(), FLAGCX_PTR_HOST,
                     sourceMr);
  ASSERT_REGISTER_MR(recvComm_, remote.data(), remote.size(), FLAGCX_PTR_HOST,
                     remoteMr);

  TestWindow sourceWindow;
  TestWindow remoteWindow;
  ASSERT_EQ(sourceWindow.init(net_, source.data(), source.size(), kLocalRank,
                              sourceMr),
            flagcxSuccess);
  ASSERT_EQ(remoteWindow.init(net_, remote.data(), remote.size(), kRemoteRank,
                              remoteMr),
            flagcxSuccess);

  constexpr int count = 3;
  const uint64_t sourceOffsets[count] = {0, 1024, 2048};
  const uint64_t remoteOffsets[count] = {128, 1152, 2176};
  const size_t sizes[count] = {512, 512, 512};
  void *requests[count] = {};
  int posted = 0;
  ASSERT_EQ(net_->iputBatch(sendComm_, count, sourceOffsets, remoteOffsets,
                            sizes, kLocalRank, kRemoteRank,
                            sourceWindow.opaque(), remoteWindow.opaque(),
                            requests, &posted),
            flagcxSuccess);
  ASSERT_EQ(posted, count);

  for (void *request : requests) {
    ASSERT_NE(request, nullptr);
    ASSERT_EQ(waitRequest(net_, request), flagcxSuccess);
  }
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(memcmp(source.data() + sourceOffsets[i],
                     remote.data() + remoteOffsets[i], sizes[i]),
              0);
  }

  EXPECT_DEREGISTER_MR(sendComm_, sourceMr);
  EXPECT_DEREGISTER_MR(recvComm_, remoteMr);
}

TEST_F(NetAdaptorBatch, TestBatch) {
  SKIP_IF_NET_CALLBACK_NULL(net_, regMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, deregMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, getMrInfo);
  SKIP_IF_NET_CALLBACK_NULL(net_, iput);
  SKIP_IF_NET_CALLBACK_NULL(net_, testBatch);
  std::vector<uint8_t> source(kBufferSize, 0x4b);
  std::vector<uint8_t> remote(kBufferSize, 0);
  void *sourceMr = nullptr;
  void *remoteMr = nullptr;
  ASSERT_REGISTER_MR(sendComm_, source.data(), source.size(), FLAGCX_PTR_HOST,
                     sourceMr);
  ASSERT_REGISTER_MR(recvComm_, remote.data(), remote.size(), FLAGCX_PTR_HOST,
                     remoteMr);

  TestWindow sourceWindow;
  TestWindow remoteWindow;
  ASSERT_EQ(sourceWindow.init(net_, source.data(), source.size(), kLocalRank,
                              sourceMr),
            flagcxSuccess);
  ASSERT_EQ(remoteWindow.init(net_, remote.data(), remote.size(), kRemoteRank,
                              remoteMr),
            flagcxSuccess);

  // A single request validates testBatch independently from iputBatch and
  // avoids requiring any other optional batch callback.
  constexpr int count = 1;
  void *requests[count] = {};
  for (int i = 0; i < count; ++i) {
    ASSERT_EQ(net_->iput(sendComm_, i, i, 1, kLocalRank, kRemoteRank,
                         sourceWindow.opaque(), remoteWindow.opaque(),
                         &requests[i]),
              flagcxSuccess);
    ASSERT_NE(requests[i], nullptr);
  }

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
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(doneFlags[i], 1);
    EXPECT_EQ(remote[i], source[i]);
  }

  int zeroDoneCount = -1;
  EXPECT_EQ(net_->testBatch(nullptr, 0, nullptr, &zeroDoneCount),
            flagcxSuccess);
  EXPECT_EQ(zeroDoneCount, 0);

  EXPECT_DEREGISTER_MR(sendComm_, sourceMr);
  EXPECT_DEREGISTER_MR(recvComm_, remoteMr);
}

TEST_F(NetAdaptorBatch, IgetBatch) {
  SKIP_IF_NET_CALLBACK_NULL(net_, regMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, deregMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, getMrInfo);
  SKIP_IF_NET_CALLBACK_NULL(net_, igetBatch);
  SKIP_IF_NET_CALLBACK_NULL(net_, test);
  std::vector<uint8_t> remote(kBufferSize);
  std::vector<uint8_t> destination(kBufferSize, 0);
  for (size_t i = 0; i < remote.size(); ++i)
    remote[i] = static_cast<uint8_t>(i);

  void *remoteMr = nullptr;
  void *destinationMr = nullptr;
  ASSERT_REGISTER_MR(recvComm_, remote.data(), remote.size(), FLAGCX_PTR_HOST,
                     remoteMr);
  ASSERT_REGISTER_MR(sendComm_, destination.data(), destination.size(),
                     FLAGCX_PTR_HOST, destinationMr);

  TestWindow remoteWindow;
  TestWindow destinationWindow;
  ASSERT_EQ(remoteWindow.init(net_, remote.data(), remote.size(), kRemoteRank,
                              remoteMr),
            flagcxSuccess);
  ASSERT_EQ(destinationWindow.init(net_, destination.data(), destination.size(),
                                   kLocalRank, destinationMr),
            flagcxSuccess);

  constexpr int count = 3;
  const uint64_t sourceOffsets[count] = {128, 1152, 2176};
  const uint64_t destinationOffsets[count] = {64, 1088, 2112};
  const size_t sizes[count] = {512, 512, 512};
  void *request = nullptr;
  ASSERT_EQ(net_->igetBatch(sendComm_, count, sourceOffsets, destinationOffsets,
                            sizes, kRemoteRank, kLocalRank,
                            remoteWindow.opaque(), destinationWindow.opaque(),
                            &request),
            flagcxSuccess);
  ASSERT_NE(request, nullptr);
  ASSERT_EQ(waitRequest(net_, request), flagcxSuccess);
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(memcmp(remote.data() + sourceOffsets[i],
                     destination.data() + destinationOffsets[i], sizes[i]),
              0);
  }

  EXPECT_DEREGISTER_MR(recvComm_, remoteMr);
  EXPECT_DEREGISTER_MR(sendComm_, destinationMr);
}

#undef SKIP_IF_NET_CALLBACK_NULL
#undef ASSERT_REGISTER_MR
#undef EXPECT_DEREGISTER_MR

} // namespace
