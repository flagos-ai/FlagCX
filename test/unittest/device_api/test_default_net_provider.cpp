/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Host-only checks for the default Net Provider dispatch contract.
 ************************************************************************/

#include "device_api/comm_traits.h"
#include "device_api/nvidia_net_provider.h"

#include <cassert>
#include <cstdint>
#include <type_traits>

struct OtherPlatform {};
struct MockGdaPlatform {};
static int mockGdaCalls = 0;

template <>
struct DefaultNetGdaProvider<MockGdaPlatform> {
  static constexpr bool compiled(unsigned providerId) {
    return providerId == 7;
  }

  template <typename Operation, typename... Args>
  static auto call(DefaultNetContext ctx, Args &&...args)
      -> decltype(DefaultNetApi<DefaultNetProviderProxy, Operation>::call(
          ctx, static_cast<Args &&>(args)...)) {
    ++mockGdaCalls;
    return DefaultNetApi<DefaultNetProviderProxy, Operation>::call(
        ctx, static_cast<Args &&>(args)...);
  }
};
static_assert(
    std::is_empty<DefaultNetProviderPolicy<OtherPlatform>::WindowState>::value,
    "non-NVIDIA Window must not carry Provider state");
static_assert(
    std::is_empty<DefaultNetProviderPolicy<OtherPlatform>::CommState>::value,
    "non-NVIDIA Comm must not carry Provider state");
static_assert(
    std::is_empty<DefaultNetProviderPolicy<OtherPlatform>::BarrierState>::value,
    "non-NVIDIA Barrier must not carry Provider state");

struct MockNet {
  mutable int calls[11] = {};

  template <typename... Args>
  void proxyPut(Args &&...) const {
    ++calls[0];
  }
  template <typename... Args>
  void proxyGet(Args &&...) const {
    ++calls[1];
  }
  template <typename... Args>
  void proxyPutValue(Args &&...) const {
    ++calls[2];
  }
  template <typename... Args>
  void proxySignal(Args &&...) const {
    ++calls[3];
  }
  template <typename... Args>
  void proxyFlush(Args &&...) const {
    ++calls[4];
  }
  template <typename... Args>
  void proxyWaitSignal(Args &&...) const {
    ++calls[5];
  }
  template <typename... Args>
  uint64_t proxyReadSignal(Args &&...) const {
    ++calls[6];
    return 42;
  }
  template <typename... Args>
  void proxyResetSignal(Args &&...) const {
    ++calls[7];
  }
  template <typename... Args>
  void proxyWaitCounter(Args &&...) const {
    ++calls[8];
  }
  template <typename... Args>
  uint64_t proxyReadCounter(Args &&...) const {
    ++calls[9];
    return 43;
  }
  template <typename... Args>
  void proxyResetCounter(Args &&...) const {
    ++calls[10];
  }
};

int main() {
  typedef DeviceAPI::Net Net;
  typedef DefaultNetProviderPolicy<NvidiaPlatform> Policy;
  DefaultNetContext ctx{DefaultNetProviderProxy, 0, nullptr, 0};
  MockNet mock;

  DefaultNetDispatch<NvidiaPlatform>::call<Net::PutOp>(ctx, mock);
  DefaultNetDispatch<NvidiaPlatform>::call<Net::GetOp>(ctx, mock);
  DefaultNetDispatch<NvidiaPlatform>::call<Net::PutValueOp>(ctx, mock);
  DefaultNetDispatch<NvidiaPlatform>::call<Net::SignalOp>(ctx, mock);
  DefaultNetDispatch<NvidiaPlatform>::call<Net::FlushOp>(ctx, mock);
  DefaultNetDispatch<NvidiaPlatform>::call<Net::WaitSignalOp>(ctx, mock);
  assert(DefaultNetDispatch<NvidiaPlatform>::call<Net::ReadSignalOp>(
             ctx, mock) == 42);
  DefaultNetDispatch<NvidiaPlatform>::call<Net::ResetSignalOp>(ctx, mock);
  DefaultNetDispatch<NvidiaPlatform>::call<Net::WaitCounterOp>(ctx, mock);
  assert(DefaultNetDispatch<NvidiaPlatform>::call<Net::ReadCounterOp>(
             ctx, mock) == 43);
  DefaultNetDispatch<NvidiaPlatform>::call<Net::ResetCounterOp>(ctx, mock);
  for (int i = 0; i != 11; ++i)
    assert(mock.calls[i] == 1);

  assert(defaultNetSelectProvider<NvidiaPlatform>(
             0, DefaultNetGdaRequiredCapabilities, true) ==
         DefaultNetProviderProxy);
  assert(defaultNetSelectProvider<NvidiaPlatform>(
             0, DefaultNetGdaRequiredCapabilities, false) ==
         DefaultNetProviderProxy);
  for (unsigned bit = 1; bit <= DefaultNetGdaBarrier; bit <<= 1) {
    assert(defaultNetSelectProvider<NvidiaPlatform>(
               0, DefaultNetGdaRequiredCapabilities & ~bit, true) ==
           DefaultNetProviderProxy);
  }
  assert(defaultNetSelectProvider<MockGdaPlatform>(
             7, DefaultNetGdaRequiredCapabilities, true) ==
         DefaultNetProviderGda);
  assert(defaultNetSelectProvider<MockGdaPlatform>(
             8, DefaultNetGdaRequiredCapabilities, true) ==
         DefaultNetProviderProxy);
  DefaultNetContext gdaContext{DefaultNetProviderGda, 7,
                               reinterpret_cast<void *>(0x1000), 0};
  DefaultNetDispatch<MockGdaPlatform>::call<Net::PutOp>(gdaContext, mock);
  assert(mockGdaCalls == 1 && mock.calls[0] == 2);

  struct HostComm : Policy::CommState {
  } hostComm;
  struct HostWindow : Policy::WindowState {
  } hostWindow;
  Policy::initComm(hostComm);
  Policy::initWindow(hostWindow);
  assert(hostComm.netProvider == DefaultNetProviderProxy);
  assert(hostComm.netProviderId == 0);
  assert(hostComm.gdaDeviceContext == nullptr);
  assert(hostWindow.netDeviceHandle == nullptr);
  assert(Policy::makeContext(hostComm, reinterpret_cast<void *>(0x1000), 2)
             .handle == reinterpret_cast<void *>(0x1000));
  return 0;
}
