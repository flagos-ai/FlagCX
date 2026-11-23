/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * NVIDIA default GDA dispatch through the installed XPU NetIO device SDK.
 ************************************************************************/

#ifndef FLAGCX_NVIDIA_XPU_NETIO_GDA_H_
#define FLAGCX_NVIDIA_XPU_NETIO_GDA_H_

#include "xpu_netio_gda_view.h"

#if defined(__CUDACC__)
#define XPU_NETIO_DEVICE_PLATFORM_HEADER                                       \
  "device_api/devnet/nvidia_xpu_netio_platform.cuh"
#include <xpu_netio/device_api.cuh>
#undef XPU_NETIO_DEVICE_PLATFORM_HEADER
#endif

#include <type_traits>

template <>
struct DefaultNetProvider<NvidiaPlatform> {
  using Api = CommTraits<DefaultBackend<NvidiaPlatform>>;
  using Net = typename Api::Net;
  using Team = typename Api::Team;
  using Window = typename Api::Window;
  using Atomic = typename PlatformTraits<NvidiaPlatform>::Atomic;
  using Clock = typename PlatformTraits<NvidiaPlatform>::Clock;

  static constexpr bool isGda = true;

  // The r4 MLX5 profile has no target-visibility or strong-signal guarantee.
  // Only arrival synchronization may be selected by this adapter.
  FLAGCX_DEVICE_INLINE_DECORATOR static bool
  supportsBarrierFence(DefaultNetContext, flagcxDevNetFenceLevel fence) {
    return fence == flagcxDevNetFenceLevel::Relaxed;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static uint64_t
  beginBarrierWait(DefaultNetContext ctx) {
    return ctx.handle != nullptr ? Clock::now() : 0;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static bool
  continueBarrierWait(DefaultNetContext ctx, uint64_t start) {
#if defined(__CUDACC__)
    if (ctx.handle == nullptr)
      return true;
    const auto &state = view(ctx);
    if (!ready(state))
      return false;
    if (Clock::elapsedNs(start, Clock::now()) >= state.timeoutNs) {
      fail(state, XPU_NETIO_TIMEOUT);
      return false;
    }
#endif
    return true;
  }

  template <typename NetT>
  FLAGCX_DEVICE_INLINE_DECORATOR static bool
  isValid(DefaultNetContext ctx, const NetT &, bool = false) {
#if defined(__CUDA_ARCH__)
    const auto *view =
        static_cast<const flagcxXpuNetioDeviceContext *>(ctx.handle);
    return flagcxDevNetResourceSharingModeValid(ctx.resourceSharingMode) &&
           ctx.contextId == 0 && view != nullptr && view->contexts != nullptr &&
           view->signalRemote != nullptr && view->signalBuffer != nullptr &&
           view->errorFlag != nullptr && view->atomicResults.address != 0 &&
           view->rank < view->nranks;
#else
    return flagcxDevNetResourceSharingModeValid(ctx.resourceSharingMode) &&
           ctx.handle != nullptr;
#endif
  }

  template <typename Operation, typename... Args>
  FLAGCX_DEVICE_INLINE_DECORATOR static auto call(DefaultNetContext ctx,
                                                  Args &&...args)
      -> decltype(DefaultNetProxyApi<Operation>::call(
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
    else if constexpr (std::is_same<Operation,
                                    typename Net::ReadSignalOp>::value)
      return readSignal(ctx, static_cast<Args &&>(args)...);
    else {
      typedef decltype(DefaultNetProxyApi<Operation>::call(
          ctx, static_cast<Args &&>(args)...)) Result;
      return Unsupported<Result>::run();
    }
#else
    typedef decltype(DefaultNetProxyApi<Operation>::call(
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
  // Sharing describes all callers of this QP, independently of the Coop
  // executing one operation. Unsupported narrow modes keep GPU protection.
  FLAGCX_DEVICE_INLINE_DECORATOR static uint32_t
  sharingFlags(DefaultNetContext ctx,
               const flagcxXpuNetioDeviceContext &state) {
    const uint64_t supported = state.contexts[0].sharing_bits;
    switch (ctx.resourceSharingMode) {
      case flagcxDevNetResourceSharingThread:
        if (supported & XPU_NETIO_SHARING_THREAD)
          return XPU_NETIO_OP_FLAG_SHARING_THREAD;
        return XPU_NETIO_OP_FLAG_SHARING_GPU;
      case flagcxDevNetResourceSharingBlock:
        if (supported & XPU_NETIO_SHARING_CTA)
          return XPU_NETIO_OP_FLAG_SHARING_CTA;
        return XPU_NETIO_OP_FLAG_SHARING_GPU;
      case flagcxDevNetResourceSharingDevice:
        return XPU_NETIO_OP_FLAG_SHARING_GPU;
      default:
        FLAGCX_DEVICE_TRAP();
        return 0;
    }
  }

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
    if (state.abortFlag != nullptr &&
        Atomic::load(const_cast<uint32_t *>(state.abortFlag),
                     flagcxDeviceMemoryOrderAcquire) != 0) {
      fail(state, XPU_NETIO_ABORTED);
      return false;
    }
    return state.contexts != nullptr && state.errorFlag != nullptr &&
           Atomic::load(state.errorFlag, flagcxDeviceMemoryOrderAcquire) == 0;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static bool
  wait(const flagcxXpuNetioDeviceContext &state, xpu_netio_status_t status,
       xpu_netio_request_t &request) {
    if (status == XPU_NETIO_SUCCESS) {
      xpu_netio_request_wait_op_t operation{};
      operation.timeout_ns = state.timeoutNs;
      operation.abort_flag =
          state.abortFlag ? state.abortFlag : state.errorFlag;
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
      const bool signalAction = net.isSignal(remoteAction);
      const int signalIdx = net.getSignalIdx(remoteAction);
      if ((!std::is_same<RA, flagcxDevNet_None>::value && !signalAction) ||
          !std::is_same<LA, flagcxDevNet_None>::value ||
          (signalAction && (signalIdx < 0 || static_cast<uint32_t>(signalIdx) >=
                                                 state.userSignalCount)) ||
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
            sharingFlags(ctx, state) | XPU_NETIO_OP_FLAG_RELEASE_SYSTEM;
        operation.local = source->local;
        operation.remote = target->remote[worldPeer];
        operation.local_offset = srcOff;
        operation.remote_offset = dstOff;
        operation.bytes = bytes;
        operation.wait = {state.timeoutNs,
                          state.abortFlag ? state.abortFlag : state.errorFlag};
        xpu_netio_request_t request{};
        const bool written =
            wait(state,
                 xpu_netio_post_write_cooperative(state.contexts, &operation,
                                                  &request, ThreadCoop{}),
                 request);
        // Signal follows every payload chunk on the same logical peer QP.
        // Do not publish successful completion after a failed Write.
        if (written && signalAction)
          postSignal(ctx, state, worldPeer, signalIdx,
                     net.getSignalValue(remoteAction));
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
      const bool signalAction = net.isSignal(remoteAction);
      const int signalIdx = net.getSignalIdx(remoteAction);
      if ((!std::is_same<RA, flagcxDevNet_None>::value && !signalAction) ||
          (signalAction && (signalIdx < 0 || static_cast<uint32_t>(signalIdx) >=
                                                 state.userSignalCount)) ||
          target == nullptr || worldPeer < 0 ||
          static_cast<uint32_t>(worldPeer) >= state.nranks ||
          sizeof(T) > XPU_NETIO_INLINE_VALUE_CAPACITY ||
          dstOff > target->bytes || sizeof(T) > target->bytes - dstOff) {
        fail(state, XPU_NETIO_INVALID_ARGUMENT);
      } else {
        xpu_netio_inline_op_t operation{};
        operation.peer = worldPeer;
        operation.queue_hint = XPU_NETIO_QUEUE_HINT_NONE;
        operation.flags =
            sharingFlags(ctx, state) | XPU_NETIO_OP_FLAG_RELEASE_SYSTEM;
        operation.remote = target->remote[worldPeer];
        operation.remote_offset = dstOff;
        operation.value_size = sizeof(T);
        const uint8_t *source = reinterpret_cast<const uint8_t *>(&value);
        for (unsigned i = 0; i < sizeof(T); ++i)
          operation.value[i] = source[i];
        operation.wait = {state.timeoutNs,
                          state.abortFlag ? state.abortFlag : state.errorFlag};
        xpu_netio_request_t request{};
        const bool written =
            wait(state,
                 xpu_netio_post_inline_cooperative(state.contexts, &operation,
                                                   &request, ThreadCoop{}),
                 request);
        if (written && signalAction)
          postSignal(ctx, state, worldPeer, signalIdx,
                     net.getSignalValue(remoteAction));
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
        operation.flags = sharingFlags(ctx, state);
        operation.wait = {state.timeoutNs,
                          state.abortFlag ? state.abortFlag : state.errorFlag};
        xpu_netio_request_t request{};
        if (!wait(state,
                  xpu_netio_flush_async(state.contexts, &operation, &request),
                  request))
          break;
      }
    }
    coop.sync();
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static void
  postSignal(DefaultNetContext ctx, const flagcxXpuNetioDeviceContext &state,
             int targetRank, int signalIdx, uint64_t increment) {
    if (!ready(state))
      return;
    if (targetRank < 0 || static_cast<uint32_t>(targetRank) >= state.nranks ||
        signalIdx < 0 ||
        static_cast<uint32_t>(signalIdx) >= state.signalCount) {
      fail(state, XPU_NETIO_INVALID_ARGUMENT);
      return;
    }
    // Each arrival increments this source's slot on the destination. Fetch
    // result is discarded, but RC atomic requires a registered return buffer.
    // Separate slots prevent concurrent peers/Barrier channels aliasing it.
    xpu_netio_atomic_op_t operation{};
    operation.peer = targetRank;
    operation.queue_hint = XPU_NETIO_QUEUE_HINT_NONE;
    operation.flags =
        sharingFlags(ctx, state) | XPU_NETIO_OP_FLAG_RELEASE_SYSTEM;
    operation.remote = state.signalRemote[targetRank];
    operation.remote_offset =
        static_cast<uint64_t>(signalIdx) * sizeof(uint64_t);
    operation.opcode = XPU_NETIO_ATOMIC_OPCODE_ADD;
    operation.operand_size = sizeof(uint64_t);
    operation.fetch = 1;
    operation.operand = increment;
    operation.result = state.atomicResults;
    operation.result_offset =
        (static_cast<uint64_t>(signalIdx) * state.nranks + targetRank) *
        sizeof(uint64_t);
    operation.wait = {state.timeoutNs,
                      state.abortFlag ? state.abortFlag : state.errorFlag};
    xpu_netio_request_t request{};
    wait(state,
         xpu_netio_post_atomic_cooperative(state.contexts, &operation, &request,
                                           ThreadCoop{}),
         request);
  }

  template <typename Coop, typename Desc>
  FLAGCX_DEVICE_INLINE_DECORATOR static void
  signal(DefaultNetContext ctx,
         const Barrier<DefaultBackend<NvidiaPlatform>, flagcxTeamTagInter, Coop>
             &barrier,
         Team, int peer, flagcxDevNet_SignalInc action,
         typename PlatformTraits<NvidiaPlatform>::CoopThread, Desc,
         flagcxDeviceScope_t, flagcxDeviceScope_t) {
    const int targetRank = peer * barrier._stride + barrier._localRank;
    postSignal(ctx, view(ctx), targetRank, static_cast<int>(action.signal), 1);
  }

  template <typename RA, typename Coop, typename Desc>
  FLAGCX_DEVICE_INLINE_DECORATOR static void
  signal(DefaultNetContext ctx, const Net &net, Team team, int peer, RA action,
         Coop coop, Desc, flagcxDeviceScope_t, flagcxDeviceScope_t) {
    const auto &state = view(ctx);
    coop.sync();
    if (coop.threadRank() == 0 && ready(state)) {
      const int idx = net.getSignalIdx(action);
      if (!net.isSignal(action) || idx < 0 ||
          static_cast<uint32_t>(idx) >= state.userSignalCount)
        fail(state, XPU_NETIO_INVALID_ARGUMENT);
      else
        postSignal(ctx, state, net.teamRankToWorld(team, peer), idx,
                   net.getSignalValue(action));
    }
    coop.sync();
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
      if (Clock::elapsedNs(start, Clock::now()) >= state.timeoutNs) {
        fail(state, XPU_NETIO_TIMEOUT);
        return;
      }
      PlatformTraits<NvidiaPlatform>::Intrin::spinBackoff(0);
    }
  }

  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR static void
  waitSignal(DefaultNetContext ctx, const Net &, Coop coop,
             flagcxDevSignal_t signalId, uint64_t expected, int bits,
             flagcxDeviceMemoryOrder_t) {
    const auto &state = view(ctx);
    coop.sync();
    if (coop.threadRank() == 0) {
      if (bits != 64 ||
          static_cast<uint32_t>(signalId) >= state.userSignalCount)
        fail(state, XPU_NETIO_INVALID_ARGUMENT);
      else {
        const uint64_t start = Clock::now();
        while (ready(state) &&
               Atomic::load(&state.signalBuffer[signalId],
                            flagcxDeviceMemoryOrderAcquire) < expected) {
          if (Clock::elapsedNs(start, Clock::now()) >= state.timeoutNs) {
            fail(state, XPU_NETIO_TIMEOUT);
            break;
          }
          PlatformTraits<NvidiaPlatform>::Intrin::spinBackoff(0);
        }
      }
    }
    coop.sync();
  }

  FLAGCX_DEVICE_INLINE_DECORATOR static uint64_t
  readSignal(DefaultNetContext ctx, const Net &, flagcxDevSignal_t signalId,
             int bits, flagcxDeviceMemoryOrder_t) {
    const auto &state = view(ctx);
    if (bits != 64 ||
        static_cast<uint32_t>(signalId) >= state.userSignalCount) {
      fail(state, XPU_NETIO_INVALID_ARGUMENT);
      return 0;
    }
    return Atomic::load(&state.signalBuffer[signalId],
                        flagcxDeviceMemoryOrderAcquire);
  }
#endif // __CUDACC__
};

template <>
struct DefaultNetProvider<NvidiaPlatform>::Unsupported<void> {
  FLAGCX_DEVICE_INLINE_DECORATOR static void run() { FLAGCX_DEVICE_TRAP(); }
};

#endif // FLAGCX_NVIDIA_XPU_NETIO_GDA_H_
