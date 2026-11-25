/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 * MUSA-only conservative Proxy completion for diagnostic acceptance runs.
 *
 * What this file adds (all MUSA-only; other vendors are unaffected):
 *   1. flagcxMusaProxyArgument / flagcxMusaProxyCollective
 *      - wrap one proxy op as: fence -> coop.sync -> enqueue -> flush -> fence
 *   2. flagcxMusaProxyOperation
 *      - per-op dispatch: picks the Coop argument (fixed index per op),
 *        Signal/Reset get their special completion handling
 *   3. DefaultNetDispatch<MusaPlatform>
 *      - inherits common validity, Barrier control and GDA dispatch;
 *        only the Proxy provider path is wrapped
 *   4. flagcxMusaCompleteProxy
 *      - rank0 fifoFlush, first-error capture via CAS into the terminal
 *        status word, then a terminal-status check
 *
 * Gated by FLAGCX_MUSA_PROXY_COMPLETION_SYNC in musa_comm_traits.h.
 * Include-order note: this header relies on default_comm_traits.h types
 * (DefaultNetContext, providers, ...) and must be included after it and
 * BEFORE the DeviceAPI alias instantiates the MUSA Default backend, so the
 * DefaultNetDispatch<MusaPlatform> specialization is declared in time.
 ************************************************************************/
#ifndef FLAGCX_MUSA_PROXY_SYNC_H_
#define FLAGCX_MUSA_PROXY_SYNC_H_

#include <type_traits>

// Select a Coop from the existing operation argument list without copying
// windows, actions, or descriptors into a tuple on device.
template <unsigned Index, typename First, typename... Rest>
FLAGCX_DEVICE_INLINE_DECORATOR decltype(auto)
flagcxMusaProxyArgument(First &&first, Rest &&...rest) {
  if constexpr (Index == 0)
    return static_cast<First &&>(first);
  else
    return flagcxMusaProxyArgument<Index - 1>(static_cast<Rest &&>(rest)...);
}

template <typename Net, typename Coop>
FLAGCX_DEVICE_INLINE_DECORATOR bool flagcxMusaCompleteProxy(const Net &net,
                                                            Coop coop);

template <typename Operation, typename Net, typename Coop, typename... Args>
FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxMusaProxyCollective(DefaultNetContext context, const Net &net, Coop coop,
                          Args &&...args) {
  // Direct local signals must follow earlier queued writes in this context.
  if constexpr (std::is_same<Operation, typename Net::SignalOp>::value) {
    if (!flagcxMusaCompleteProxy(net, coop))
      return;
  }
  PlatformTraits<MusaPlatform>::Intrin::threadfenceSystem();
  coop.sync();
  Operation::proxy(context, net, static_cast<Args &&>(args)...);
  if (!flagcxMusaCompleteProxy(net, coop))
    return;
  PlatformTraits<MusaPlatform>::Intrin::threadfenceSystem();
}

// Barrier objects share the operation dispatch but have no Net::PutOp. Their
// participants and completion contract stay with their existing algorithm.
template <typename Net, typename Operation, typename = void>
struct flagcxMusaProxyOperation {
  template <typename... Args>
  FLAGCX_DEVICE_INLINE_DECORATOR static auto
  call(DefaultNetContext context, const Net &net, Args &&...args)
      -> decltype(Operation::proxy(context, net,
                                   static_cast<Args &&>(args)...)) {
    return Operation::proxy(context, net, static_cast<Args &&>(args)...);
  }
};

template <typename Net, typename Operation>
struct flagcxMusaProxyOperation<Net, Operation,
                                std::void_t<typename Net::PutOp>> {
  template <typename... Args>
  FLAGCX_DEVICE_INLINE_DECORATOR static auto
  call(DefaultNetContext context, const Net &net, Args &&...args)
      -> decltype(Operation::proxy(context, net,
                                   static_cast<Args &&>(args)...)) {
    if constexpr (std::is_same<Operation, typename Net::PutOp>::value) {
      auto coop = flagcxMusaProxyArgument<9>(args...);
      return flagcxMusaProxyCollective<Operation>(
          context, net, coop, static_cast<Args &&>(args)...);
    } else if constexpr (std::is_same<Operation, typename Net::GetOp>::value) {
      auto coop = flagcxMusaProxyArgument<7>(args...);
      return flagcxMusaProxyCollective<Operation>(
          context, net, coop, static_cast<Args &&>(args)...);
    } else if constexpr (std::is_same<Operation,
                                      typename Net::PutValueOp>::value) {
      auto coop = flagcxMusaProxyArgument<6>(args...);
      return flagcxMusaProxyCollective<Operation>(
          context, net, coop, static_cast<Args &&>(args)...);
    } else if constexpr (std::is_same<Operation,
                                      typename Net::SignalOp>::value) {
      auto coop = flagcxMusaProxyArgument<3>(args...);
      return flagcxMusaProxyCollective<Operation>(
          context, net, coop, static_cast<Args &&>(args)...);
    } else if constexpr (std::is_same<Operation,
                                      typename Net::ResetSignalOp>::value ||
                         std::is_same<Operation,
                                      typename Net::ResetCounterOp>::value) {
      if (!flagcxMusaCompleteProxy(
              net, typename PlatformTraits<MusaPlatform>::CoopThread{}))
        return;
      Operation::proxy(context, net, static_cast<Args &&>(args)...);
      PlatformTraits<MusaPlatform>::Intrin::threadfenceSystem();
    } else {
      return Operation::proxy(context, net, static_cast<Args &&>(args)...);
    }
  }
};

// Specialize before DeviceAPI instantiates the MUSA Default backend. The
// generic dispatch, operation implementations, and device-view layout are
// shared unchanged. Only the Proxy call is wrapped here.
template <>
struct DefaultNetDispatch<MusaPlatform> : DefaultNetDispatchBase<MusaPlatform> {
  template <typename Operation, typename Net, typename... Args>
  FLAGCX_DEVICE_INLINE_DECORATOR static auto
  call(DefaultNetContext context, const Net &net, Args &&...args)
      -> decltype(Operation::proxy(context, net,
                                   static_cast<Args &&>(args)...)) {
    if (context.useGda)
      return DefaultNetDispatchBase<MusaPlatform>::template call<Operation>(
          context, net, static_cast<Args &&>(args)...);
    if (!flagcxDevNetResourceSharingModeValid(context.resourceSharingMode))
      FLAGCX_DEVICE_TRAP();
    return flagcxMusaProxyOperation<Net, Operation>::call(
        context, net, static_cast<Args &&>(args)...);
  }
};

template <typename Net, typename Coop>
FLAGCX_DEVICE_INLINE_DECORATOR bool flagcxMusaCompleteProxy(const Net &net,
                                                            Coop coop) {
  using Provider = CommTraits<DefaultBackend<MusaPlatform>>::ProxyProvider;
  using Atomic = PlatformTraits<MusaPlatform>::Atomic;
  coop.sync();
  if (coop.threadRank() == 0 && net.fifoBuffer != nullptr) {
    const flagcxResult_t status = Provider::fifoFlush(net.fifoBuffer);
    if (status != flagcxSuccess) {
      // Preserve the first error and let other contexts leave their existing
      // terminal-aware waits. Avoid the platform's unreachable trap mapping.
      for (int i = 0; i < net._dc.contextCount; ++i) {
        if (net._dc.fifoBuffers[i] != nullptr) {
          auto *word = static_cast<uint64_t *>(net._dc.fifoBuffers[i]) +
                       flagcxFifoIdxTerminalStatus;
          uint64_t expected = flagcxSuccess;
          Atomic::compareExchange(word, expected, static_cast<uint64_t>(status),
                                  flagcxDeviceMemoryOrderAcqRel);
        }
      }
    }
  }
  coop.sync();
  return net.fifoBuffer == nullptr ||
         Provider::fifoTerminalStatus(
             static_cast<uint64_t *>(net.fifoBuffer)) == flagcxSuccess;
}

#endif // FLAGCX_MUSA_PROXY_SYNC_H_
