/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * NVIDIA resources for the shared DefaultBackend Net provider contract.
 ************************************************************************/

#ifndef FLAGCX_NVIDIA_NET_PROVIDER_H_
#define FLAGCX_NVIDIA_NET_PROVIDER_H_

#include "default_comm_traits.h"

struct NvidiaPlatform;

// A future NVIDIA GDA implementation specializes
// DefaultNetGdaProvider<NvidiaPlatform>. Its providerId distinguishes native
// implementations while the common Net API remains unchanged.
template <>
struct DefaultNetProviderPolicy<NvidiaPlatform> {
  struct WindowState {
    // Published before the Window device view is materialized. Its host owner
    // releases it through the existing deferred DevMem resource mechanism.
    const void *netDeviceHandle;
  };
  struct CommState {
    DefaultNetProviderKind netProvider;
    unsigned netProviderId;
    void *gdaDeviceContext;
  };
  struct BarrierState {
    DefaultNetProviderKind netProvider;
    unsigned netProviderId;
    void *gdaDeviceContext;
  };

  template <typename Window>
  FLAGCX_HOST_DEVICE_INLINE static void initWindow(Window &win) {
    static_cast<WindowState &>(win).netDeviceHandle = nullptr;
  }

  template <typename Comm>
  FLAGCX_HOST_DEVICE_INLINE static void initComm(Comm &comm) {
    CommState &state = static_cast<CommState &>(comm);
    state.netProvider = defaultNetSelectProvider<NvidiaPlatform>(0, 0, false);
    state.netProviderId = 0;
    state.gdaDeviceContext = nullptr;
  }

  template <typename Comm>
  FLAGCX_HOST_DEVICE_INLINE static DefaultNetContext
  makeContext(const Comm &comm, void *fifo, int contextId) {
    const CommState &state = static_cast<const CommState &>(comm);
    return DefaultNetContext{state.netProvider, state.netProviderId,
                             state.netProvider == DefaultNetProviderProxy
                                 ? fifo
                                 : state.gdaDeviceContext,
                             contextId};
  }

  template <typename Barrier>
  FLAGCX_HOST_DEVICE_INLINE static void initBarrier(Barrier &barrier,
                                                    DefaultNetContext ctx) {
    BarrierState &state = static_cast<BarrierState &>(barrier);
    state.netProvider = ctx.provider;
    state.netProviderId = ctx.providerId;
    state.gdaDeviceContext =
        ctx.provider == DefaultNetProviderGda ? ctx.handle : nullptr;
  }

  template <typename Barrier>
  FLAGCX_HOST_DEVICE_INLINE static DefaultNetContext
  barrierContext(const Barrier &barrier, void *fifo, int contextId) {
    const BarrierState &state = static_cast<const BarrierState &>(barrier);
    return DefaultNetContext{state.netProvider, state.netProviderId,
                             state.netProvider == DefaultNetProviderProxy
                                 ? fifo
                                 : state.gdaDeviceContext,
                             contextId};
  }
};

#endif // FLAGCX_NVIDIA_NET_PROVIDER_H_
