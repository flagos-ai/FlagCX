/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

// Adaptor-level GPU RDMA READ diagnostics for the IB P2P transport.
//
// The P2P Engine tests exercise the same data path through metadata exchange
// and scheduling.  These tests deliberately stop at the net adaptor boundary
// so a GPU-memory failure can be distinguished from an Engine failure.  Each
// allocation/topology case is read twice using independent destination
// buffers: once with an immediate D2H copy after the CQ completion and once
// after deviceSynchronize() has provided the strongest generic visibility
// operation exposed by the device adaptor.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "../adaptor/net_test_utils.h"
#include "adaptor.h"
#include "flagcx.h"
#include "flagcx_net.h"
#include "flagcx_net_adaptor.h"

extern struct flagcxNetAdaptor flagcxNetIbP2p;

namespace {

constexpr size_t kReadBytes = 4096;
constexpr auto kConnectTimeout = std::chrono::seconds(10);
constexpr auto kTransferTimeout = std::chrono::seconds(10);

enum class AllocationKind {
  DeviceMalloc,
  FlagcxMemAlloc,
};

struct GpuReadParam {
  AllocationKind allocation;
  int sourceGpu;
  int destinationGpu;
  const char *name;
};

class ScopedDeviceHandle {
public:
  ScopedDeviceHandle() = default;
  ~ScopedDeviceHandle() {
    if (handle_ != nullptr)
      flagcxDeviceHandleFree(handle_);
  }

  ScopedDeviceHandle(const ScopedDeviceHandle &) = delete;
  ScopedDeviceHandle &operator=(const ScopedDeviceHandle &) = delete;

  flagcxResult_t init() { return flagcxDeviceHandleInit(&handle_); }
  flagcxDeviceHandle_t get() const { return handle_; }

private:
  flagcxDeviceHandle_t handle_ = nullptr;
};

class ScopedDeviceAllocation {
public:
  ScopedDeviceAllocation() = default;
  ~ScopedDeviceAllocation() { reset(); }

  ScopedDeviceAllocation(const ScopedDeviceAllocation &) = delete;
  ScopedDeviceAllocation &operator=(const ScopedDeviceAllocation &) = delete;

  flagcxResult_t allocate(flagcxDeviceHandle_t handle, int device,
                          AllocationKind kind, size_t size) {
    handle_ = handle;
    device_ = device;
    kind_ = kind;
    FLAGCXCHECK(handle_->setDevice(device_));
    if (kind_ == AllocationKind::FlagcxMemAlloc)
      return flagcxMemAlloc(&pointer_, size);
    return handle_->deviceMalloc(&pointer_, size, flagcxMemDevice, nullptr);
  }

  void *get() const { return pointer_; }

  void reset() {
    if (pointer_ == nullptr)
      return;
    if (handle_ != nullptr)
      handle_->setDevice(device_);
    if (kind_ == AllocationKind::FlagcxMemAlloc)
      flagcxMemFree(pointer_);
    else if (handle_ != nullptr)
      handle_->deviceFree(pointer_, flagcxMemDevice, nullptr);
    pointer_ = nullptr;
  }

private:
  flagcxDeviceHandle_t handle_ = nullptr;
  int device_ = -1;
  AllocationKind kind_ = AllocationKind::DeviceMalloc;
  void *pointer_ = nullptr;
};

class ScopedMr {
public:
  ScopedMr() = default;
  ~ScopedMr() { reset(); }

  ScopedMr(const ScopedMr &) = delete;
  ScopedMr &operator=(const ScopedMr &) = delete;

  void set(void *comm, void *mr) {
    comm_ = comm;
    mr_ = mr;
  }

  void *get() const { return mr_; }

  void reset() {
    if (comm_ != nullptr && mr_ != nullptr)
      flagcxNetIbP2p.deregMr(comm_, mr_);
    comm_ = nullptr;
    mr_ = nullptr;
  }

private:
  void *comm_ = nullptr;
  void *mr_ = nullptr;
};

class ScopedP2pConnection {
public:
  ScopedP2pConnection() = default;
  ~ScopedP2pConnection() { reset(); }

  ScopedP2pConnection(const ScopedP2pConnection &) = delete;
  ScopedP2pConnection &operator=(const ScopedP2pConnection &) = delete;

  flagcxResult_t connect(int listenDev, int connectDev) {
    char handle[FLAGCX_NET_HANDLE_MAXSIZE] = {};
    flagcxResult_t result =
        flagcxNetIbP2p.listen(listenDev, handle, &listenComm_);
    if (result != flagcxSuccess)
      return result;

    auto acceptFuture = std::async(std::launch::async, [&]() {
      void *comm = nullptr;
      const flagcxResult_t status = flagcxNetIbP2p.accept(listenComm_, &comm);
      return std::make_pair(status, comm);
    });
    auto connectFuture = std::async(std::launch::async, [&]() {
      void *comm = nullptr;
      const flagcxResult_t status =
          flagcxNetIbP2p.connect(connectDev, handle, &comm);
      return std::make_pair(status, comm);
    });

    if (connectFuture.wait_for(kConnectTimeout) != std::future_status::ready ||
        acceptFuture.wait_for(kConnectTimeout) != std::future_status::ready)
      return flagcxSystemError;

    const auto connectResult = connectFuture.get();
    const auto acceptResult = acceptFuture.get();
    sendComm_ = connectResult.second;
    recvComm_ = acceptResult.second;
    if (connectResult.first != flagcxSuccess)
      return connectResult.first;
    if (acceptResult.first != flagcxSuccess)
      return acceptResult.first;
    if (sendComm_ == nullptr || recvComm_ == nullptr)
      return flagcxSystemError;
    return flagcxSuccess;
  }

  void *sendComm() const { return sendComm_; }
  void *recvComm() const { return recvComm_; }

  void reset() {
    if (sendComm_ != nullptr)
      flagcxNetIbP2p.closeSend(sendComm_);
    if (recvComm_ != nullptr)
      flagcxNetIbP2p.closeRecv(recvComm_);
    if (listenComm_ != nullptr)
      flagcxNetIbP2p.closeListen(listenComm_);
    sendComm_ = nullptr;
    recvComm_ = nullptr;
    listenComm_ = nullptr;
  }

private:
  void *listenComm_ = nullptr;
  void *sendComm_ = nullptr;
  void *recvComm_ = nullptr;
};

flagcxResult_t waitRequest(void *request) {
  const auto deadline = std::chrono::steady_clock::now() + kTransferTimeout;
  int done = 0;
  while (!done && std::chrono::steady_clock::now() < deadline) {
    flagcxResult_t result = flagcxNetIbP2p.test(request, &done, nullptr);
    if (result != flagcxSuccess)
      return result;
    if (!done)
      std::this_thread::yield();
  }
  return done ? flagcxSuccess : flagcxSystemError;
}

const char *allocationName(AllocationKind kind) {
  return kind == AllocationKind::DeviceMalloc ? "deviceMalloc"
                                              : "flagcxMemAlloc/GDR";
}

class P2pGpuReadTest : public ::testing::TestWithParam<GpuReadParam> {};

TEST_P(P2pGpuReadTest, ReadsGpuMrWithAndWithoutVisibilitySync) {
  const GpuReadParam param = GetParam();
  if (param.allocation == AllocationKind::FlagcxMemAlloc) {
    const char *hetero = std::getenv("FLAGCX_USE_HETERO_COMM");
    if (hetero == nullptr || std::strcmp(hetero, "1") != 0) {
      GTEST_SKIP() << "flagcxMemAlloc/GDR diagnostics require "
                      "FLAGCX_USE_HETERO_COMM=1";
    }
  }

  ASSERT_EQ(flagcxNetIbP2p.init(), flagcxSuccess);
  int netDeviceCount = 0;
  ASSERT_EQ(flagcxNetIbP2p.devices(&netDeviceCount), flagcxSuccess);
  ASSERT_GT(netDeviceCount, 0);

  ScopedDeviceHandle deviceHandleGuard;
  ASSERT_EQ(deviceHandleGuard.init(), flagcxSuccess);
  flagcxDeviceHandle_t deviceHandle = deviceHandleGuard.get();
  ASSERT_NE(deviceHandle, nullptr);

  int gpuCount = 0;
  ASSERT_EQ(deviceHandle->getDeviceCount(&gpuCount), flagcxSuccess);
  if (param.sourceGpu >= gpuCount || param.destinationGpu >= gpuCount) {
    GTEST_SKIP() << "GPU topology case requires device "
                 << std::max(param.sourceGpu, param.destinationGpu)
                 << " but only " << gpuCount << " device(s) are visible";
  }

  ASSERT_EQ(deviceHandle->setDevice(param.sourceGpu), flagcxSuccess);
  int sourceNetDev = -1;
  ASSERT_EQ(flagcx_test::getLocalNetDevice(&flagcxNetIbP2p, netDeviceCount,
                                           &sourceNetDev),
            flagcxSuccess);
  ASSERT_GE(sourceNetDev, 0);
  ASSERT_LT(sourceNetDev, netDeviceCount);
  ASSERT_EQ(deviceHandle->setDevice(param.destinationGpu), flagcxSuccess);
  int destinationNetDev = -1;
  ASSERT_EQ(flagcx_test::getLocalNetDevice(&flagcxNetIbP2p, netDeviceCount,
                                           &destinationNetDev),
            flagcxSuccess);
  ASSERT_GE(destinationNetDev, 0);
  ASSERT_LT(destinationNetDev, netDeviceCount);

  flagcxNetProperties_t sourceProperties = {};
  flagcxNetProperties_t destinationProperties = {};
  ASSERT_EQ(flagcxNetIbP2p.getProperties(sourceNetDev, &sourceProperties),
            flagcxSuccess);
  ASSERT_EQ(
      flagcxNetIbP2p.getProperties(destinationNetDev, &destinationProperties),
      flagcxSuccess);
  if ((sourceProperties.ptrSupport & FLAGCX_PTR_CUDA) == 0 ||
      (destinationProperties.ptrSupport & FLAGCX_PTR_CUDA) == 0) {
    GTEST_SKIP()
        << "At least one topology-selected P2P device lacks GPU MR support";
  }

  SCOPED_TRACE(::testing::Message()
               << "allocator=" << allocationName(param.allocation)
               << " sourceGpu=" << param.sourceGpu
               << " destinationGpu=" << param.destinationGpu
               << " sourceNetDev=" << sourceNetDev << " sourceNetName="
               << (sourceProperties.name != nullptr ? sourceProperties.name
                                                    : "<unnamed>")
               << " sourcePciPath="
               << (sourceProperties.pciPath != nullptr
                       ? sourceProperties.pciPath
                       : "<unknown>")
               << " destinationNetDev=" << destinationNetDev
               << " destinationNetName="
               << (destinationProperties.name != nullptr
                       ? destinationProperties.name
                       : "<unnamed>")
               << " destinationPciPath="
               << (destinationProperties.pciPath != nullptr
                       ? destinationProperties.pciPath
                       : "<unknown>"));

  ScopedP2pConnection connection;
  ASSERT_EQ(connection.connect(sourceNetDev, destinationNetDev), flagcxSuccess);
  ASSERT_NE(connection.sendComm(), nullptr);
  ASSERT_NE(connection.recvComm(), nullptr);

  std::vector<unsigned char> expected(kReadBytes);
  std::vector<unsigned char> zeros(kReadBytes, 0);
  for (size_t i = 0; i < expected.size(); ++i)
    expected[i] = static_cast<unsigned char>((i * 37u + 11u) & 0xffu);

  for (bool synchronizeBeforeCopy : {false, true}) {
    SCOPED_TRACE(::testing::Message()
                 << "visibility="
                 << (synchronizeBeforeCopy ? "deviceSynchronize-before-D2H"
                                           : "direct-D2H"));

    ScopedDeviceAllocation remoteSource;
    ScopedDeviceAllocation localDestination;
    ASSERT_EQ(remoteSource.allocate(deviceHandle, param.sourceGpu,
                                    param.allocation, kReadBytes),
              flagcxSuccess);
    ASSERT_EQ(localDestination.allocate(deviceHandle, param.destinationGpu,
                                        param.allocation, kReadBytes),
              flagcxSuccess);
    ASSERT_NE(remoteSource.get(), nullptr);
    ASSERT_NE(localDestination.get(), nullptr);

    ASSERT_EQ(deviceHandle->setDevice(param.sourceGpu), flagcxSuccess);
    ASSERT_EQ(deviceHandle->deviceMemcpy(remoteSource.get(), expected.data(),
                                         kReadBytes, flagcxMemcpyHostToDevice,
                                         nullptr),
              flagcxSuccess);
    std::vector<unsigned char> sourceActual(kReadBytes, 0);
    ASSERT_EQ(deviceHandle->deviceMemcpy(sourceActual.data(),
                                         remoteSource.get(), kReadBytes,
                                         flagcxMemcpyDeviceToHost, nullptr),
              flagcxSuccess);
    ASSERT_EQ(sourceActual, expected)
        << "Remote GPU source was incorrect before the RDMA READ";
    ASSERT_EQ(deviceHandle->setDevice(param.destinationGpu), flagcxSuccess);
    ASSERT_EQ(deviceHandle->deviceMemcpy(localDestination.get(), zeros.data(),
                                         kReadBytes, flagcxMemcpyHostToDevice,
                                         nullptr),
              flagcxSuccess);

    void *remoteMrRaw = nullptr;
    void *localMrRaw = nullptr;
    ASSERT_EQ(deviceHandle->setDevice(param.sourceGpu), flagcxSuccess);
    ASSERT_EQ(flagcxNetIbP2p.regMr(connection.recvComm(), remoteSource.get(),
                                   kReadBytes, FLAGCX_PTR_CUDA,
                                   FLAGCX_NET_MR_FLAG_NONE, &remoteMrRaw),
              flagcxSuccess);
    ScopedMr remoteMr;
    remoteMr.set(connection.recvComm(), remoteMrRaw);
    ASSERT_EQ(deviceHandle->setDevice(param.destinationGpu), flagcxSuccess);
    ASSERT_EQ(flagcxNetIbP2p.regMr(
                  connection.sendComm(), localDestination.get(), kReadBytes,
                  FLAGCX_PTR_CUDA, FLAGCX_NET_MR_FLAG_NONE, &localMrRaw),
              flagcxSuccess);
    ScopedMr localMr;
    localMr.set(connection.sendComm(), localMrRaw);

    void *request = nullptr;
    ASSERT_EQ(flagcxNetIbP2p.iget(connection.sendComm(), 0, 0, kReadBytes, 0, 1,
                                  reinterpret_cast<void **>(remoteMr.get()),
                                  reinterpret_cast<void **>(localMr.get()),
                                  &request),
              flagcxSuccess);
    ASSERT_NE(request, nullptr);
    ASSERT_EQ(waitRequest(request), flagcxSuccess);

    ASSERT_EQ(deviceHandle->setDevice(param.destinationGpu), flagcxSuccess);
    if (synchronizeBeforeCopy) {
      ASSERT_NE(deviceHandle->deviceSynchronize, nullptr);
      ASSERT_EQ(deviceHandle->deviceSynchronize(), flagcxSuccess);
    }

    std::vector<unsigned char> actual(kReadBytes, 0);
    ASSERT_EQ(deviceHandle->deviceMemcpy(actual.data(), localDestination.get(),
                                         kReadBytes, flagcxMemcpyDeviceToHost,
                                         nullptr),
              flagcxSuccess);
    EXPECT_EQ(actual, expected)
        << "GPU RDMA READ completed successfully but the destination data "
           "was not visible to the device runtime";
  }
}

INSTANTIATE_TEST_SUITE_P(
    AllocationAndTopology, P2pGpuReadTest,
    ::testing::Values(
        GpuReadParam{AllocationKind::DeviceMalloc, 0, 0,
                     "DeviceMallocSameGpu0"},
        GpuReadParam{AllocationKind::DeviceMalloc, 1, 1,
                     "DeviceMallocSameGpu1"},
        GpuReadParam{AllocationKind::DeviceMalloc, 0, 1,
                     "DeviceMallocGpu0ToGpu1"},
        GpuReadParam{AllocationKind::DeviceMalloc, 1, 0,
                     "DeviceMallocGpu1ToGpu0"},
        GpuReadParam{AllocationKind::FlagcxMemAlloc, 0, 0, "FlagcxMemSameGpu0"},
        GpuReadParam{AllocationKind::FlagcxMemAlloc, 1, 1, "FlagcxMemSameGpu1"},
        GpuReadParam{AllocationKind::FlagcxMemAlloc, 0, 1,
                     "FlagcxMemGpu0ToGpu1"},
        GpuReadParam{AllocationKind::FlagcxMemAlloc, 1, 0,
                     "FlagcxMemGpu1ToGpu0"}),
    [](const ::testing::TestParamInfo<GpuReadParam> &info) {
      return std::string(info.param.name);
    });

} // namespace
