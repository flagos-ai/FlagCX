/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Contract and loopback tests for the network adaptor selected for IBRC.
 * USE_ACCL_BAREX=1 selects BAREX through the normal adaptor registry, so
 * these tests exercise the same vtable used by the FlagCX runtime.
 ************************************************************************/

#include <chrono>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
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
    net_ = getUnifiedNetAdaptor(IBRC);
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

    const auto deadline = std::chrono::steady_clock::now() + kTimeout;
    while ((sendComm_ == nullptr || recvComm_ == nullptr) &&
           std::chrono::steady_clock::now() < deadline) {
      if (sendComm_ == nullptr) {
        ASSERT_EQ(net_->connect(0, handle_, &sendComm_), flagcxSuccess);
      }
      if (recvComm_ == nullptr) {
        ASSERT_EQ(net_->accept(listenComm_, &recvComm_), flagcxSuccess);
      }
      if (sendComm_ == nullptr || recvComm_ == nullptr)
        std::this_thread::yield();
    }
    ASSERT_NE(sendComm_, nullptr) << "connect timed out";
    ASSERT_NE(recvComm_, nullptr) << "accept timed out";
  }

  void TearDown() override {
    if (net_ == nullptr)
      return;
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

  void registerMr(void *comm, void *buffer, size_t size, int type,
                  void **mrHandle) {
    ASSERT_NE(net_->regMr, nullptr);
    ASSERT_EQ(net_->regMr(comm, buffer, size, type, FLAGCX_NET_MR_FLAG_NONE,
                          mrHandle),
              flagcxSuccess);
    ASSERT_NE(*mrHandle, nullptr);
  }

  void deregisterMr(void *comm, void *mrHandle) {
    ASSERT_NE(net_->deregMr, nullptr);
    ASSERT_EQ(net_->deregMr(comm, mrHandle), flagcxSuccess);
  }

  struct flagcxNetAdaptor *net_ = nullptr;
  int nDevs_ = 0;
  char handle_[FLAGCX_NET_HANDLE_MAXSIZE] = {};
  void *listenComm_ = nullptr;
  void *sendComm_ = nullptr;
  void *recvComm_ = nullptr;
};

class NetAdaptorMemory : public NetAdaptorLoopback {};
class NetAdaptorTwoSided : public NetAdaptorLoopback {};
class NetAdaptorOneSided : public NetAdaptorLoopback {};
class NetAdaptorBatch : public NetAdaptorLoopback {};

TEST(NetAdaptorInterface, AdaptorIsAvailable) {
  struct flagcxNetAdaptor *net = getUnifiedNetAdaptor(IBRC);
  ASSERT_NE(net, nullptr);
  EXPECT_NE(net->name, nullptr);
}

TEST(NetAdaptorInterface, Init) {
  struct flagcxNetAdaptor *net = getUnifiedNetAdaptor(IBRC);
  ASSERT_NE(net, nullptr);
  SKIP_IF_NET_CALLBACK_NULL(net, init);
  EXPECT_EQ(net->init(), flagcxSuccess);
}

TEST(NetAdaptorInterface, Devices) {
  struct flagcxNetAdaptor *net = getUnifiedNetAdaptor(IBRC);
  ASSERT_NE(net, nullptr);
  SKIP_IF_NET_CALLBACK_NULL(net, init);
  SKIP_IF_NET_CALLBACK_NULL(net, devices);
  ASSERT_EQ(net->init(), flagcxSuccess);
  int ndev = 0;
  EXPECT_EQ(net->devices(&ndev), flagcxSuccess);
  EXPECT_GT(ndev, 0);
}

TEST(NetAdaptorInterface, GetProperties) {
  struct flagcxNetAdaptor *net = getUnifiedNetAdaptor(IBRC);
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
  struct flagcxNetAdaptor *net = getUnifiedNetAdaptor(IBRC);
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

TEST_F(NetAdaptorMemory, RegisterHostMr) {
  SKIP_IF_NET_CALLBACK_NULL(net_, regMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, deregMr);
  std::vector<uint8_t> buffer(kBufferSize);
  void *mrHandle = nullptr;
  registerMr(sendComm_, buffer.data(), buffer.size(), FLAGCX_PTR_HOST,
             &mrHandle);

  deregisterMr(sendComm_, mrHandle);
}

TEST_F(NetAdaptorMemory, ExportMrMetadata) {
  SKIP_IF_NET_CALLBACK_NULL(net_, regMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, deregMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, getMrInfo);
  std::vector<uint8_t> buffer(kBufferSize);
  void *mrHandle = nullptr;
  registerMr(sendComm_, buffer.data(), buffer.size(), FLAGCX_PTR_HOST,
             &mrHandle);

  struct flagcxNetMrInfo info = {};
  ASSERT_EQ(net_->getMrInfo(mrHandle, &info), flagcxSuccess);
  EXPECT_GT(info.nKeys, 0u);
  EXPECT_LE(info.nKeys, static_cast<uint32_t>(FLAGCX_NET_MAX_MR_KEYS));

  deregisterMr(sendComm_, mrHandle);
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

  flagcxDeviceHandle_t device = nullptr;
  ASSERT_EQ(flagcxDeviceHandleInit(&device), flagcxSuccess);
  ASSERT_NE(device, nullptr);
  ASSERT_EQ(device->setDevice(0), flagcxSuccess);
  flagcxStream_t stream = nullptr;
  ASSERT_EQ(device->streamCreate(&stream), flagcxSuccess);
  void *buffer = nullptr;
  ASSERT_EQ(device->deviceMalloc(&buffer, kBufferSize, flagcxMemDevice, stream),
            flagcxSuccess);

  void *mrHandle = nullptr;
  registerMr(sendComm_, buffer, kBufferSize, FLAGCX_PTR_CUDA, &mrHandle);
  struct flagcxNetMrInfo info = {};
  EXPECT_EQ(net_->getMrInfo(mrHandle, &info), flagcxSuccess);
  EXPECT_GT(info.nKeys, 0u);
  deregisterMr(sendComm_, mrHandle);

  EXPECT_EQ(device->deviceFree(buffer, flagcxMemDevice, stream), flagcxSuccess);
  EXPECT_EQ(device->streamDestroy(stream), flagcxSuccess);
  EXPECT_EQ(flagcxDeviceHandleFree(device), flagcxSuccess);
}

TEST_F(NetAdaptorMemory, RegMrDmaBufRegistration) {
  SKIP_IF_NET_CALLBACK_NULL(net_, regMrDmaBuf);
  SKIP_IF_NET_CALLBACK_NULL(net_, deregMr);

  ASSERT_NE(deviceAdaptor, nullptr);
  ASSERT_NE(deviceAdaptor->setDevice, nullptr);
  ASSERT_NE(deviceAdaptor->gdrMemAlloc, nullptr);
  ASSERT_NE(deviceAdaptor->gdrMemFree, nullptr);
  ASSERT_NE(deviceAdaptor->getHandleForAddressRange, nullptr);
  ASSERT_EQ(deviceAdaptor->setDevice(0), flagcxSuccess);

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
  registerMr(sendComm_, source.data(), source.size(), FLAGCX_PTR_HOST,
             &sourceMr);
  registerMr(recvComm_, destination.data(), destination.size(), FLAGCX_PTR_HOST,
             &destinationMr);

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

  deregisterMr(sendComm_, sourceMr);
  deregisterMr(recvComm_, destinationMr);
}

TEST_F(NetAdaptorTwoSided, Flush) {
  SKIP_IF_NET_CALLBACK_NULL(net_, regMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, deregMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, iflush);
  SKIP_IF_NET_CALLBACK_NULL(net_, test);
  std::vector<uint8_t> destination(kBufferSize, 0);
  void *destinationMr = nullptr;
  registerMr(recvComm_, destination.data(), destination.size(), FLAGCX_PTR_HOST,
             &destinationMr);

  void *recvData[1] = {destination.data()};
  int flushSizes[1] = {static_cast<int>(destination.size())};
  void *recvMrs[1] = {destinationMr};
  void *flushRequest = nullptr;
  ASSERT_EQ(
      net_->iflush(recvComm_, 1, recvData, flushSizes, recvMrs, &flushRequest),
      flagcxSuccess);
  if (flushRequest != nullptr && flushRequest != reinterpret_cast<void *>(1)) {
    EXPECT_EQ(waitRequest(net_, flushRequest), flagcxSuccess);
  }

  deregisterMr(recvComm_, destinationMr);
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
  registerMr(sendComm_, putSource.data(), putSource.size(), FLAGCX_PTR_HOST,
             &putSourceMr);
  registerMr(recvComm_, remoteBuffer.data(), remoteBuffer.size(),
             FLAGCX_PTR_HOST, &remoteMr);

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

  deregisterMr(sendComm_, putSourceMr);
  deregisterMr(recvComm_, remoteMr);
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
  registerMr(recvComm_, remoteBuffer.data(), remoteBuffer.size(),
             FLAGCX_PTR_HOST, &remoteMr);
  registerMr(sendComm_, getDestination.data(), getDestination.size(),
             FLAGCX_PTR_HOST, &getDestinationMr);

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

  deregisterMr(recvComm_, remoteMr);
  deregisterMr(sendComm_, getDestinationMr);
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
  registerMr(sendComm_, source.data(), source.size(), FLAGCX_PTR_HOST,
             &sourceMr);
  registerMr(recvComm_, remote.data(), remote.size(), FLAGCX_PTR_HOST,
             &remoteMr);
  registerMr(sendComm_, destination.data(), destination.size(), FLAGCX_PTR_HOST,
             &destinationMr);

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

  deregisterMr(sendComm_, sourceMr);
  deregisterMr(recvComm_, remoteMr);
  deregisterMr(sendComm_, destinationMr);
}

TEST_F(NetAdaptorOneSided, PerRankRegionBounds) {
  SKIP_IF_NET_CALLBACK_NULL(net_, regMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, deregMr);
  SKIP_IF_NET_CALLBACK_NULL(net_, getMrInfo);
  SKIP_IF_NET_CALLBACK_NULL(net_, iput);
  SKIP_IF_NET_CALLBACK_NULL(net_, iget);
  SKIP_IF_NET_CALLBACK_NULL(net_, test);
  if (net_->name == nullptr || strcmp(net_->name, "BAREX") != 0)
    GTEST_SKIP() << "Per-rank bounds validation is implemented by BAREX";

  std::vector<uint8_t> source(1024);
  std::vector<uint8_t> remote(4096, 0);
  std::vector<uint8_t> destination(2048, 0);
  for (size_t i = 0; i < source.size(); ++i)
    source[i] = static_cast<uint8_t>(i);

  void *sourceMr = nullptr;
  void *remoteMr = nullptr;
  void *destinationMr = nullptr;
  registerMr(sendComm_, source.data(), source.size(), FLAGCX_PTR_HOST,
             &sourceMr);
  registerMr(recvComm_, remote.data(), remote.size(), FLAGCX_PTR_HOST,
             &remoteMr);
  registerMr(sendComm_, destination.data(), destination.size(), FLAGCX_PTR_HOST,
             &destinationMr);

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

  deregisterMr(sendComm_, sourceMr);
  deregisterMr(recvComm_, remoteMr);
  deregisterMr(sendComm_, destinationMr);
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
  registerMr(sendComm_, source.data(), source.size(), FLAGCX_PTR_HOST,
             &sourceMr);
  registerMr(recvComm_, remote.data(), remote.size(), FLAGCX_PTR_HOST,
             &remoteMr);
  ASSERT_EQ(net_->regMr(recvComm_, &signal, sizeof(signal), FLAGCX_PTR_HOST,
                        FLAGCX_NET_MR_FLAG_FORCE_SO, &signalMr),
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

  deregisterMr(sendComm_, sourceMr);
  deregisterMr(recvComm_, remoteMr);
  deregisterMr(recvComm_, signalMr);
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
  registerMr(sendComm_, source.data(), source.size(), FLAGCX_PTR_HOST,
             &sourceMr);
  registerMr(recvComm_, remote.data(), remote.size(), FLAGCX_PTR_HOST,
             &remoteMr);

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

  deregisterMr(sendComm_, sourceMr);
  deregisterMr(recvComm_, remoteMr);
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
  registerMr(sendComm_, source.data(), source.size(), FLAGCX_PTR_HOST,
             &sourceMr);
  registerMr(recvComm_, remote.data(), remote.size(), FLAGCX_PTR_HOST,
             &remoteMr);

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

  deregisterMr(sendComm_, sourceMr);
  deregisterMr(recvComm_, remoteMr);
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
  registerMr(recvComm_, remote.data(), remote.size(), FLAGCX_PTR_HOST,
             &remoteMr);
  registerMr(sendComm_, destination.data(), destination.size(), FLAGCX_PTR_HOST,
             &destinationMr);

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

  deregisterMr(recvComm_, remoteMr);
  deregisterMr(sendComm_, destinationMr);
}

#undef SKIP_IF_NET_CALLBACK_NULL

} // namespace
