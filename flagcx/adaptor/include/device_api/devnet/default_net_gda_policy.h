/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Device-view storage shared by DefaultBackend platforms with GDA support.
 * Include after default_comm_traits.h; the platform opts in by specializing
 * DefaultNetProviderPolicy<PlatformTag> with this base.
 ************************************************************************/

#ifndef FLAGCX_DEFAULT_NET_GDA_POLICY_H_
#define FLAGCX_DEFAULT_NET_GDA_POLICY_H_

#include "device_api/default_comm_traits.h"

struct DefaultNetGdaPolicy {
  struct WindowState {
    // Host publishes this only after the Window's GDA resources are ready.
    const void *netDeviceHandle;
  };
  struct CommState {
    bool netUseGda;
    void *gdaDeviceContext;
  };
  struct BarrierState {
    bool netUseGda;
    void *gdaDeviceContext;
  };

  template <typename Window>
  FLAGCX_HOST_DEVICE_INLINE static void initWindow(Window &win) {
    static_cast<WindowState &>(win).netDeviceHandle = nullptr;
  }

  template <typename Window>
  FLAGCX_HOST_DEVICE_INLINE static bool hasWindowAccess(const Window &win) {
    return static_cast<const WindowState &>(win).netDeviceHandle != nullptr;
  }

  template <typename Window>
  FLAGCX_HOST_DEVICE_INLINE static void publishWindow(Window &win,
                                                      const void *handle) {
    static_cast<WindowState &>(win).netDeviceHandle = handle;
  }

  template <typename Comm, typename HostComm>
  FLAGCX_HOST_DEVICE_INLINE static void initComm(Comm &comm,
                                                 const HostComm &host) {
    CommState &state = static_cast<CommState &>(comm);
    state.netUseGda = host.netUseGda != 0;
    state.gdaDeviceContext = host.gdaDeviceContext;
  }

  template <typename Comm>
  FLAGCX_HOST_DEVICE_INLINE static DefaultNetContext
  makeContext(const Comm &comm, void *fifo, int contextId) {
    const CommState &state = static_cast<const CommState &>(comm);
    return DefaultNetContext{state.netUseGda ? state.gdaDeviceContext : fifo,
                             contextId, flagcxDevNetResourceSharingDevice,
                             state.netUseGda};
  }

  template <typename Barrier>
  FLAGCX_HOST_DEVICE_INLINE static void initBarrier(Barrier &barrier,
                                                    DefaultNetContext ctx) {
    BarrierState &state = static_cast<BarrierState &>(barrier);
    state.netUseGda = ctx.useGda;
    state.gdaDeviceContext = ctx.useGda ? ctx.handle : nullptr;
  }

  template <typename Barrier>
  FLAGCX_HOST_DEVICE_INLINE static DefaultNetContext
  barrierContext(const Barrier &barrier, void *fifo, int contextId) {
    const BarrierState &state = static_cast<const BarrierState &>(barrier);
    return DefaultNetContext{state.netUseGda ? state.gdaDeviceContext : fifo,
                             contextId, flagcxDevNetResourceSharingDevice,
                             state.netUseGda};
  }
};

#endif // FLAGCX_DEFAULT_NET_GDA_POLICY_H_
