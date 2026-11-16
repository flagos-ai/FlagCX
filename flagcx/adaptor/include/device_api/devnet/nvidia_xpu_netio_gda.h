/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * NVIDIA default GDA dispatch through the installed XPU NetIO device SDK.
 ************************************************************************/

#ifndef FLAGCX_NVIDIA_XPU_NETIO_GDA_H_
#define FLAGCX_NVIDIA_XPU_NETIO_GDA_H_

#include "xpu_netio_gda_view.h"

#if defined(__CUDACC__)
#include <xpu_netio/device_api.cuh>
#endif

#include <type_traits>

template <>
struct DefaultNetGdaProvider<NvidiaPlatform> {
  using Api = CommTraits<DefaultBackend<NvidiaPlatform>>;
  using Net = typename Api::Net;
  using Team = typename Api::Team;
  using Window = typename Api::Window;
  using Atomic = typename PlatformTraits<NvidiaPlatform>::Atomic;
  using Clock = typename PlatformTraits<NvidiaPlatform>::Clock;

  FLAGCX_HOST_DEVICE_INLINE static constexpr bool
  compiled(unsigned providerId) {
    return providerId == 1;
  }

  // The r4 MLX5 profile has no target-visibility or strong-signal guarantee.
  // Only arrival synchronization may be selected by this adapter.
  FLAGCX_DEVICE_INLINE_DECORATOR static bool
  supportsBarrierFence(DefaultNetContext, flagcxDevNetFenceLevel fence) {
    return fence == flagcxDevNetFenceLevel::Relaxed;
  }

  template <typename NetT>
  FLAGCX_DEVICE_INLINE_DECORATOR static bool isValid(DefaultNetContext ctx,
                                                     const NetT &) {
#if defined(__CUDA_ARCH__)
    const auto *view =
        static_cast<const flagcxXpuNetioDeviceContext *>(ctx.handle);
    return compiled(ctx.providerId) && ctx.contextId == 0 && view != nullptr &&
           view->contexts != nullptr && view->signalRemote != nullptr &&
           view->signalBuffer != nullptr && view->errorFlag != nullptr &&
           view->rank < view->nranks;
#else
    return compiled(ctx.providerId) && ctx.handle != nullptr;
#endif
  }

  template <typename Operation, typename... Args>
  FLAGCX_DEVICE_INLINE_DECORATOR static auto call(DefaultNetContext ctx,
                                                  Args &&...args)
      -> decltype(DefaultNetApi<DefaultNetProviderProxy, Operation>::call(
          ctx, static_cast<Args &&>(args)...)) {
#if defined(__CUDACC__)
    if constexpr (std::is_same<Operation, typename Net::PutOp>::value)
      return put(ctx, static_cast<Args &&>(args)...);
    else if constexpr (std::is_same<Operation, typename Net::PutValueOp>::value)
      return putValue(ctx, static_cast<Args &&>(args)...);
    else if constexpr (std::is_same<Operation, typename Net::FlushOp>::value)
      return flush(ctx, static_cast<Args &&>(args)...);
    else if constexpr (std::is_same<Operation, typename Net::SignalOp>::value)
      return signal(ctx, static_cast<Args &&>(args)...);
    else if constexpr (std::is_same<Operation,
                                    typename Net::WaitSignalOp>::value)
      return waitSignal(ctx, static_cast<Args &&>(args)...);
    else {
      typedef decltype(DefaultNetApi<DefaultNetProviderProxy, Operation>::call(
          ctx, static_cast<Args &&>(args)...)) Result;
      return Unsupported<Result>::run();
    }
#else
    typedef decltype(DefaultNetApi<DefaultNetProviderProxy, Operation>::call(
        ctx, static_cast<Args &&>(args)...)) Result;
    return Unsupported<Result>::run();
#endif
  }

private:
  template <typename Result>
  struct Unsupported {
    FLAGCX_DEVICE_INLINE_DECORATOR static Result run() {
      FLAGCX_DEVICE_TRAP();
      return Result{};
    }
  };

#if defined(__CUDACC__)
  struct ThreadCoop {
    __device__ uint32_t thread_rank() const { return 0; }
    __device__ uint32_t size() const { return 1; }
    __device__ uint32_t broadcast(uint32_t value, int) const { return value; }
    __device__ void sync() const {}
  };

  FLAGCX_DEVICE_INLINE_DECORATOR static const flagcxXpuNetioDeviceContext &
  view(DefaultNetContext ctx) {
    return *static_cast<const flagcxXpuNetioDeviceContext *>(ctx.handle);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static void
  fail(const flagcxXpuNetioDeviceContext &state, xpu_netio_status_t status) {
    if (state.errorFlag != nullptr) {
      uint32_t expected = 0;
      Atomic::compareExchange(state.errorFlag, expected,
                              static_cast<uint32_t>(status),
                              flagcxDeviceMemoryOrderAcqRel);
    }
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static bool
  ready(const flagcxXpuNetioDeviceContext &state) {
    return state.contexts != nullptr && state.errorFlag != nullptr &&
           Atomic::load(state.errorFlag, flagcxDeviceMemoryOrderAcquire) == 0;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static bool
  wait(const flagcxXpuNetioDeviceContext &state, xpu_netio_status_t status,
       xpu_netio_request_t &request) {
    if (status == XPU_NETIO_SUCCESS) {
      xpu_netio_request_wait_op_t operation{};
      operation.timeout_ns = state.timeoutNs;
      operation.abort_flag = state.errorFlag;
      status = xpu_netio_request_wait(state.contexts, &request, &operation);
    }
    if (status != XPU_NETIO_SUCCESS)
      fail(state, status);
    return status == XPU_NETIO_SUCCESS;
  }

  template <typename RA, typename LA, typename Coop, typename Desc>
  FLAGCX_DEVICE_INLINE_DECORATOR static void
  put(DefaultNetContext ctx, const Net &net, Team team, int peer, Window dst,
      size_t dstOff, Window src, size_t srcOff, size_t bytes, RA remoteAction,
      LA localAction, Coop coop, Desc, flagcxDeviceScope_t,
      flagcxDeviceScope_t) {
    const auto &state = view(ctx);
    coop.sync();
    if (coop.threadRank() == 0 && ready(state)) {
      const int worldPeer = net.teamRankToWorld(team, peer);
      const auto *source =
          static_cast<const flagcxXpuNetioDeviceWindow *>(src.netDeviceHandle);
      const auto *target =
          static_cast<const flagcxXpuNetioDeviceWindow *>(dst.netDeviceHandle);
      if (net.isSignal(remoteAction) || net.isCounter(localAction) ||
          source == nullptr || target == nullptr || worldPeer < 0 ||
          static_cast<uint32_t>(worldPeer) >= state.nranks ||
          srcOff > source->bytes || bytes > source->bytes - srcOff ||
          dstOff > target->bytes || bytes > target->bytes - dstOff) {
        fail(state, XPU_NETIO_INVALID_ARGUMENT);
      } else {
        xpu_netio_write_op_t operation{};
        operation.peer = worldPeer;
        operation.queue_hint = XPU_NETIO_QUEUE_HINT_NONE;
        operation.flags =
            XPU_NETIO_OP_FLAG_SHARING_THREAD | XPU_NETIO_OP_FLAG_RELEASE_SYSTEM;
        operation.local = source->local;
        operation.remote = target->remote[worldPeer];
        operation.local_offset = srcOff;
        operation.remote_offset = dstOff;
        operation.bytes = bytes;
        operation.wait = {state.timeoutNs, state.errorFlag};
        xpu_netio_request_t request{};
        wait(state,
             xpu_netio_post_write_cooperative(state.contexts, &operation,
                                              &request, ThreadCoop{}),
             request);
      }
    }
    coop.sync();
  }

  template <typename T, typename RA, typename Coop, typename Desc>
  FLAGCX_DEVICE_INLINE_DECORATOR static void
  putValue(DefaultNetContext ctx, const Net &net, Team team, int peer,
           Window dst, size_t dstOff, T value, RA remoteAction, Coop coop, Desc,
           flagcxDeviceScope_t, flagcxDeviceScope_t) {
    const auto &state = view(ctx);
    coop.sync();
    if (coop.threadRank() == 0 && ready(state)) {
      const int worldPeer = net.teamRankToWorld(team, peer);
      const auto *target =
          static_cast<const flagcxXpuNetioDeviceWindow *>(dst.netDeviceHandle);
      if (net.isSignal(remoteAction) || target == nullptr || worldPeer < 0 ||
          static_cast<uint32_t>(worldPeer) >= state.nranks ||
          sizeof(T) > XPU_NETIO_INLINE_VALUE_CAPACITY ||
          dstOff > target->bytes || sizeof(T) > target->bytes - dstOff) {
        fail(state, XPU_NETIO_INVALID_ARGUMENT);
      } else {
        xpu_netio_inline_op_t operation{};
        operation.peer = worldPeer;
        operation.queue_hint = XPU_NETIO_QUEUE_HINT_NONE;
        operation.flags =
            XPU_NETIO_OP_FLAG_SHARING_THREAD | XPU_NETIO_OP_FLAG_RELEASE_SYSTEM;
        operation.remote = target->remote[worldPeer];
        operation.remote_offset = dstOff;
        operation.value_size = sizeof(T);
        const uint8_t *source = reinterpret_cast<const uint8_t *>(&value);
        for (unsigned i = 0; i < sizeof(T); ++i)
          operation.value[i] = source[i];
        operation.wait = {state.timeoutNs, state.errorFlag};
        xpu_netio_request_t request{};
        wait(state,
             xpu_netio_post_inline_cooperative(state.contexts, &operation,
                                               &request, ThreadCoop{}),
             request);
      }
    }
    coop.sync();
  }

  template <typename NetT, typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR static void flush(DefaultNetContext ctx,
                                                   const NetT &, Coop coop,
                                                   flagcxDeviceMemoryOrder_t) {
    const auto &state = view(ctx);
    coop.sync();
    if (coop.threadRank() == 0 && ready(state)) {
      for (uint32_t peer = 0; peer < state.nranks; ++peer) {
        if (peer == state.rank)
          continue;
        xpu_netio_flush_async_op_t operation{};
        operation.peer = peer;
        operation.queue_hint = XPU_NETIO_QUEUE_HINT_NONE;
        operation.flags = XPU_NETIO_OP_FLAG_SHARING_THREAD;
        operation.wait = {state.timeoutNs, state.errorFlag};
        xpu_netio_request_t request{};
        if (!wait(state,
                  xpu_netio_flush_async(state.contexts, &operation, &request),
                  request))
          break;
      }
    }
    coop.sync();
  }

  template <typename Coop, typename Desc>
  FLAGCX_DEVICE_INLINE_DECORATOR static void
  signal(DefaultNetContext ctx,
         const Barrier<DefaultBackend<NvidiaPlatform>, flagcxTeamTagInter, Coop>
             &barrier,
         Team, int peer, flagcxDevNet_SignalInc action,
         typename PlatformTraits<NvidiaPlatform>::CoopThread, Desc,
         flagcxDeviceScope_t, flagcxDeviceScope_t) {
    const auto &state = view(ctx);
    if (!ready(state))
      return;
    const int targetRank = peer;
    const int signalIdx = static_cast<int>(action.signal);
    if (targetRank < 0 || static_cast<uint32_t>(targetRank) >= state.nranks ||
        signalIdx < 0 ||
        static_cast<uint32_t>(signalIdx) >= state.signalCount) {
      fail(state, XPU_NETIO_INVALID_ARGUMENT);
      return;
    }
    const uint64_t epoch = barrier._shadowBuffer[signalIdx];
    xpu_netio_inline_op_t operation{};
    operation.peer = targetRank;
    operation.queue_hint = XPU_NETIO_QUEUE_HINT_NONE;
    operation.flags =
        XPU_NETIO_OP_FLAG_SHARING_THREAD | XPU_NETIO_OP_FLAG_RELEASE_SYSTEM;
    operation.remote = state.signalRemote[targetRank];
    operation.remote_offset =
        static_cast<uint64_t>(signalIdx) * sizeof(uint64_t);
    operation.value_size = sizeof(epoch);
    const uint8_t *bytes = reinterpret_cast<const uint8_t *>(&epoch);
    for (unsigned i = 0; i < sizeof(epoch); ++i)
      operation.value[i] = bytes[i];
    operation.wait = {state.timeoutNs, state.errorFlag};
    xpu_netio_request_t request{};
    wait(state,
         xpu_netio_post_inline_cooperative(state.contexts, &operation, &request,
                                           ThreadCoop{}),
         request);
  }

  template <typename... Args>
  FLAGCX_DEVICE_INLINE_DECORATOR static void signal(DefaultNetContext,
                                                    const Net &, Args &&...) {
    FLAGCX_DEVICE_TRAP();
  }

  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR static void
  waitSignal(DefaultNetContext ctx,
             const Barrier<DefaultBackend<NvidiaPlatform>, flagcxTeamTagInter,
                           Coop> &barrier,
             typename PlatformTraits<NvidiaPlatform>::CoopThread,
             flagcxDevSignal_t signalId, uint64_t expected, int bits,
             flagcxDeviceMemoryOrder_t) {
    const auto &state = view(ctx);
    if (bits != 64 || static_cast<uint32_t>(signalId) >= state.signalCount) {
      fail(state, XPU_NETIO_INVALID_ARGUMENT);
      return;
    }
    uint64_t *slot = &barrier._signalBuffer[signalId];
    const uint64_t start = Clock::now();
    while (Atomic::load(slot, flagcxDeviceMemoryOrderAcquire) < expected) {
      if (!ready(state))
        return;
      if (Clock::now() - start >= state.timeoutNs) {
        fail(state, XPU_NETIO_TIMEOUT);
        return;
      }
      PlatformTraits<NvidiaPlatform>::Intrin::spinBackoff(0);
    }
  }

  template <typename... Args>
  FLAGCX_DEVICE_INLINE_DECORATOR static void
  waitSignal(DefaultNetContext, const Net &, Args &&...) {
    FLAGCX_DEVICE_TRAP();
  }
#endif // __CUDACC__
};

template <>
struct DefaultNetGdaProvider<NvidiaPlatform>::Unsupported<void> {
  FLAGCX_DEVICE_INLINE_DECORATOR static void run() { FLAGCX_DEVICE_TRAP(); }
};

#endif // FLAGCX_NVIDIA_XPU_NETIO_GDA_H_
