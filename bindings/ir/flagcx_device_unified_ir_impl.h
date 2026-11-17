/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX Unified One-Sided IR — Implementation.
 *
 * Transport-transparent dispatch: checks the peer pointer for a P2P
 * reachability, falls back to Net path otherwise.
 *
 * Included by the bitcode compilation unit via flagcx_device_scalar_ir_impl.h.
 *
 * NOTE: Implementation order matters. Signal/Wait/Flush/Reset (U4-U7) are
 * defined first because Put variants (U1, U3) call them for P2P signal
 * delivery on the data-complete path.
 ************************************************************************/
#ifndef FLAGCX_DEVICE_UNIFIED_IR_IMPL_H_
#define FLAGCX_DEVICE_UNIFIED_IR_IMPL_H_

#include "flagcx_device_unified_ir.h"
#include <stdint.h> // For uint64_t, uint32_t, uint16_t

/* ================================================================
 * Internal helper: scoped memory fence
 * ================================================================ */

static FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxScopedFence(flagcxDevMemoryScope_t scope) {
  switch (scope) {
    case flagcxDeviceScopeSystem:
      DeviceAPI::Intrin::threadfenceSystem();
      break;
    case flagcxDeviceScopeDevice:
      DeviceAPI::Intrin::threadfenceDevice();
      break;
    default:
      break; // Block/Thread: no fence needed
  }
}

/* ================================================================
 * Internal helpers: direct global-memory access (P2P path)
 * ================================================================ */

template <typename Ptr>
static FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxStoreVolatile64Internal(Ptr ptr, uint64_t value) {
  auto dst = FLAGCX_IR_GLOBAL_PTR_CAST(volatile uint64_t, ptr);
  *dst = value;
#if defined(__CCE_IS_AICORE__) && defined(USE_ASCEND_ADAPTOR)
  // Ascend spells the store-release as plain STG + dirty-line sweep; the
  // platform's CceAtomic::store uses the identical pair and the same
  // measured rationale (2026-11-17 host-readback probe, 910C): a bare
  // STG is not guaranteed at a coherent point by kernel tail.
  asc_dcci_entire_out();
#endif
}

// Prefer a platform's optimized cooperative copy when available.
template <typename Intrin, typename DstPtr, typename SrcPtr>
static FLAGCX_DEVICE_INLINE_DECORATOR auto
flagcxCoopCopyBytesInternal(DstPtr dst, SrcPtr src, size_t bytes, int rank,
                            int size, int)
    -> decltype(Intrin::coopCopyBytes(dst, src, bytes, rank, size), void()) {
  Intrin::coopCopyBytes(dst, src, bytes, rank, size);
}

// A correct byte-copy baseline for platforms without an optimized method.
// Address-space spelling is centralized in FLAGCX_IR_GLOBAL_PTR_CAST.
template <typename Intrin, typename DstPtr, typename SrcPtr>
static FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCoopCopyBytesInternal(DstPtr dstIn, SrcPtr srcIn, size_t bytes, int rank,
                            int size, long) {
  auto dst = FLAGCX_IR_GLOBAL_PTR_CAST(unsigned char, dstIn);
  auto src = FLAGCX_IR_GLOBAL_PTR_CAST(const unsigned char, srcIn);
  for (size_t i = (size_t)rank; i < bytes; i += (size_t)size)
    dst[i] = src[i];
}

template <typename DstPtr, typename SrcPtr>
static FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCoopMemcpy(flagcxDevCoopKind_t coopKind, DstPtr dst, SrcPtr src,
                 size_t bytes) {
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  flagcxCoopCopyBytesInternal<DeviceAPI::Intrin>(
      dst, src, bytes, coop.threadRank(), coop.size(), 0);
}

// DefaultBackend owns the completion-word width.  Other backends retain their
// existing uint64_t counter operation through the fallback overload.
template <typename Net>
static FLAGCX_DEVICE_INLINE_DECORATOR auto
flagcxCompleteCounterInternal(const Net &net, flagcxDevCounter_t counter,
                              uint64_t delta, int)
    -> decltype(net.completeCounter(counter, delta), void()) {
  net.completeCounter(counter, delta);
}

template <typename Net>
static FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCompleteCounterInternal(const Net &net, flagcxDevCounter_t counter,
                              uint64_t delta, long) {
  DeviceAPI::Atomic::fetchAdd(net.getCounterPtr(counter), delta,
                              flagcxDeviceMemoryOrderRelease);
}

/* ================================================================
 * Category U4: Unified Signal (2)
 *
 * DEFINED FIRST — Put variants call these for P2P signal delivery.
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevSignalInc(const void FLAGCX_IR_GLOBAL_PTR *commOpaque,
                   flagcxDevTeamKind_t teamKind, int peer,
                   flagcxDevSignal_t signal, flagcxDevContext_t contextId,
                   flagcxDevCoopKind_t coopKind, flagcxDevMemoryScope_t scope) {
  (void)scope;
  FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
  flagcxDevNetSignalSigIncS(FLAGCX_IR_NET_ARG(net), commOpaque, teamKind, peer,
                            coopKind, signal);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevSignalAdd(const void FLAGCX_IR_GLOBAL_PTR *commOpaque,
                   flagcxDevTeamKind_t teamKind, int peer,
                   flagcxDevSignal_t signal, uint64_t value,
                   flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
                   flagcxDevMemoryScope_t scope) {
  (void)scope;
  FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
  flagcxDevNetSignalSigAddS(FLAGCX_IR_NET_ARG(net), commOpaque, teamKind, peer,
                            coopKind, signal, value);
}

/* ================================================================
 * Category U5: Unified Wait (2)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevWaitSignal(const void FLAGCX_IR_GLOBAL_PTR *commOpaque,
                    flagcxDevSignal_t signal, uint64_t least, int bits,
                    flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
                    flagcxDevMemoryOrder_t order) {
  FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
  flagcxDevNetWaitSignalS(FLAGCX_IR_NET_ARG(net), coopKind, signal, least, bits,
                          order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevWaitCounter(const void FLAGCX_IR_GLOBAL_PTR *commOpaque,
                     flagcxDevCounter_t counter, uint64_t least, int bits,
                     flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
                     flagcxDevMemoryOrder_t order) {
  FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
  flagcxDevNetWaitCounterS(FLAGCX_IR_NET_ARG(net), coopKind, counter, least,
                           bits, order);
}

/* ================================================================
 * Category U6: Unified Read (2)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxDevReadSignal(
    const void FLAGCX_IR_GLOBAL_PTR *commOpaque, flagcxDevSignal_t signal,
    int bits, flagcxDevContext_t contextId, flagcxDevMemoryOrder_t order) {
  FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
  return flagcxDevNetReadSignalS(FLAGCX_IR_NET_ARG(net), signal, bits, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxDevReadCounter(
    const void FLAGCX_IR_GLOBAL_PTR *commOpaque, flagcxDevCounter_t counter,
    int bits, flagcxDevContext_t contextId, flagcxDevMemoryOrder_t order) {
  FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
  return flagcxDevNetReadCounterS(FLAGCX_IR_NET_ARG(net), counter, bits, order);
}

/* ================================================================
 * Category U7: Unified Flush / Reset / Shadow (4)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevFlush(const void FLAGCX_IR_GLOBAL_PTR *commOpaque,
               flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
               flagcxDevMemoryOrder_t order) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;

#if FLAGCX_DEVICE_HAS_NET_TRANSPORT
  // A communicator may use IPC for one memory object and Net fallback for
  // another, so flush both domains.  Flushing an empty FIFO is harmless.
  DeviceAPI::Intrin::threadfenceSystem();
  if (comm->_commBase.isOneSidedTransportReady() &&
      flagcxDevCommHasNetContexts(*comm)) {
    FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
    flagcxDevNetFlushS(FLAGCX_IR_NET_ARG(net), coopKind, order);
  }
#else
  // No device net transport on this platform (Ascend 2201 today).  Both
  // statements above reference capabilities this pass declares absent:
  // Intrin::threadfenceSystem is the never-defined trap call (measured:
  // the same undefined-symbol class that killed probe r144 N2 through the
  // put arm), and the net flush arm sits behind a runtime predicate, which
  // DCE cannot remove at instantiation time (probe r145 named the inlined
  // kernel as the referrer; r146 then showed only caller-side CONSTANT
  // scope/kind arguments fold out of flagcxScopedFence - a runtime bool
  // guard is not one).  What cannot run is irrelevant: an IPC-only comm
  // has no net contexts, so the FIFO this flushes is empty by
  // construction.  What flush must still do here is order the calling
  // core's earlier peer stores ahead of the kernel's result-store, and
  // the measured dcci pair behind Intrin::threadfenceDevice is the
  // ordering primitive for exactly that envelope (device scope, single
  // chip; see the fence note in device_utils.h and the rendezvous record
  // in ascend_platform_traits.h, whose two-card peer visibility gate is
  // the hardware premise for the same choice made for put at r144/r146).
  (void)comm;
  (void)contextId;
  (void)coopKind;
  (void)order;
  DeviceAPI::Intrin::threadfenceDevice();
#endif
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevResetSignal(const void FLAGCX_IR_GLOBAL_PTR *commOpaque,
                     flagcxDevContext_t contextId, flagcxDevSignal_t slot) {
  FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
  flagcxDevNetResetSignal(FLAGCX_IR_NET_ARG(net), slot);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevResetCounter(const void FLAGCX_IR_GLOBAL_PTR *commOpaque,
                      flagcxDevContext_t contextId, flagcxDevCounter_t slot) {
  FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
  flagcxDevNetResetCounter(FLAGCX_IR_NET_ARG(net), slot);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevIncreaseSignalShadow(const void FLAGCX_IR_GLOBAL_PTR *commOpaque,
                              flagcxDevContext_t contextId,
                              flagcxDevSignal_t slot, uint64_t delta) {
  FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
  flagcxDevNetIncreaseSignalShadow(FLAGCX_IR_NET_ARG(net), slot, delta);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevWaitSignalMeetShadow(const void FLAGCX_IR_GLOBAL_PTR *commOpaque,
                              flagcxDevContext_t contextId,
                              flagcxDevSignal_t slot, int bits,
                              flagcxDevCoopKind_t coopKind,
                              flagcxDevMemoryOrder_t order) {
  FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
  flagcxDevNetWaitSignalMeetShadowS(FLAGCX_IR_NET_ARG(net), coopKind, slot,
                                    bits, order);
}

/* ================================================================
 * Category U8: Unified Barrier (3)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void flagcxDevBarrierArrive(
    const void FLAGCX_IR_GLOBAL_PTR *commOpaque, flagcxDevTeamKind_t teamKind,
    uint32_t index, flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
    flagcxDevMemoryOrder_t order, flagcxDevMemoryScope_t scope) {
  switch (teamKind) {
    case FLAGCX_TEAM_INTRA:
      flagcxIntraBarrierArriveS(commOpaque, coopKind, index,
                                /*multimem=*/false, order);
      break;
    case FLAGCX_TEAM_INTER: {
      FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
      flagcxInterBarrierArriveS(FLAGCX_IR_NET_ARG(net), coopKind, index, order,
                                flagcxDevNetFenceLevel::Relaxed);
      break;
    }
    case FLAGCX_TEAM_WORLD: {
      FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
      flagcxWorldBarrierArriveS(FLAGCX_IR_NET_ARG(net), coopKind, index,
                                /*multimem=*/false, order,
                                flagcxDevNetFenceLevel::Relaxed);
      break;
    }
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void flagcxDevBarrierWait(
    const void FLAGCX_IR_GLOBAL_PTR *commOpaque, flagcxDevTeamKind_t teamKind,
    uint32_t index, flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
    flagcxDevMemoryOrder_t order, flagcxDevMemoryScope_t scope) {
  switch (teamKind) {
    case FLAGCX_TEAM_INTRA:
      flagcxIntraBarrierWaitS(commOpaque, coopKind, index,
                              /*multimem=*/false, order);
      break;
    case FLAGCX_TEAM_INTER: {
      FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
      flagcxInterBarrierWaitS(FLAGCX_IR_NET_ARG(net), coopKind, index, order,
                              flagcxDevNetFenceLevel::Relaxed);
      break;
    }
    case FLAGCX_TEAM_WORLD: {
      FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
      flagcxWorldBarrierWaitS(FLAGCX_IR_NET_ARG(net), coopKind, index,
                              /*multimem=*/false, order,
                              flagcxDevNetFenceLevel::Relaxed);
      break;
    }
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void flagcxDevBarrierSync(
    const void FLAGCX_IR_GLOBAL_PTR *commOpaque, flagcxDevTeamKind_t teamKind,
    uint32_t index, flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
    flagcxDevMemoryOrder_t order, flagcxDevMemoryScope_t scope) {
  switch (teamKind) {
    case FLAGCX_TEAM_INTRA:
      flagcxIntraBarrierSyncS(commOpaque, coopKind, index,
                              /*multimem=*/false, order);
      break;
    case FLAGCX_TEAM_INTER: {
      FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
      flagcxInterBarrierSyncS(FLAGCX_IR_NET_ARG(net), coopKind, index, order,
                              flagcxDevNetFenceLevel::Relaxed);
      break;
    }
    case FLAGCX_TEAM_WORLD: {
      FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
      flagcxWorldBarrierSyncS(FLAGCX_IR_NET_ARG(net), coopKind, index,
                              /*multimem=*/false, order,
                              flagcxDevNetFenceLevel::Relaxed);
      break;
    }
  }
}

/* ================================================================
 * Category U1: Unified Put (4)
 *
 * These come AFTER signal/wait so that P2P signal delivery calls
 * (flagcxDevSignalInc, flagcxDevSignalAdd) are already defined.
 * ================================================================ */

// Helper: Returns true if peer is on the same node (local)
static FLAGCX_DEVICE_INLINE_DECORATOR bool
flagcxIsPeerLocal(const flagcxDevComm &comm, const flagcxTeam &team, int peer) {
  int worldPeer = flagcxTeamRankToWorld(comm, team, peer);

  // Get my intra base (world rank of rank-0 on my node)
  int myIntraBase = comm._commBase.getRank() - comm._commBase.getIntraRank();

  // Check if peer's world rank is in my node's range
  bool isLocal = (worldPeer >= myIntraBase) &&
                 (worldPeer < myIntraBase + comm._commBase.getIntraSize());

  return isLocal;
}

// Validate team semantics and return whether the peer is a P2P candidate.
// Callers must still check the actual peer pointer: topology alone does not
// guarantee that IPC/VMM setup succeeded.
//
// There is deliberately no caller-name parameter here. It used to be an
// unused `const char *funcName`, and that argument could not bind at all on
// address-space-disjoint compilers: measured on Ascend 910C (bisheng
// 9.1.0-beta.3, probe r139), a string literal in this pass has type
// `const __gm__ char[N]`, and the device image's generic addrspace pass
// rejects the cross-space binding at every call site -- in an extern "C"
// (non-template) inline body, which is type-checked at parse regardless of
// any dead-code elimination. Dropping the dead argument keeps the behaviour
// identical on every platform and makes the whole shared surface bindable
// where literals live in a named address space.
static FLAGCX_DEVICE_INLINE_DECORATOR bool
flagcxValidateAndDispatch(const flagcxDevComm &comm, const flagcxTeam &team,
                          int peer, flagcxDevTeamKind_t teamKind,
                          bool &shouldReturn) {
  shouldReturn = false;
  bool isPeerLocal = flagcxIsPeerLocal(comm, team, peer);

  // Validate team semantics
  if (teamKind == FLAGCX_TEAM_INTRA && !isPeerLocal) {
    shouldReturn = true;
    return false;
  }
  if (teamKind == FLAGCX_TEAM_INTER && isPeerLocal) {
    shouldReturn = true;
    return false;
  }

  // Determine dispatch path
  bool useP2P = (teamKind == FLAGCX_TEAM_INTRA) ||
                (teamKind == FLAGCX_TEAM_WORLD && isPeerLocal);
  return useP2P;
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevPut(const void FLAGCX_IR_GLOBAL_PTR *commOpaque,
             const void FLAGCX_IR_GLOBAL_PTR *dstOpaque, size_t dstOffset,
             const void FLAGCX_IR_GLOBAL_PTR *srcOpaque, size_t srcOffset,
             size_t bytes, flagcxDevTeamKind_t teamKind, int peer,
             flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
             flagcxDevMemoryScope_t scope, flagcxDevMemoryOrder_t order) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);

  bool shouldReturn;
  bool useP2P = flagcxValidateAndDispatch(*comm, team, peer, teamKind,
                                          shouldReturn);
  if (shouldReturn)
    return;
  auto peerPtr =
      useP2P ? flagcxGetPeerPointerWithAccess(*dst, dstOffset, team, peer,
                                              flagcxDevPeerAccessWriteOnly)
             : nullptr;
  useP2P = peerPtr != nullptr;

  if (useP2P) {
    auto localSrc = flagcxGetLocalPointer(*src, srcOffset);
    if (order == flagcxDeviceMemoryOrderRelease ||
        order == flagcxDeviceMemoryOrderAcqRel)
      flagcxScopedFence(scope);
    flagcxCoopMemcpy(coopKind, peerPtr, localSrc, bytes);
  } else {
#if FLAGCX_DEVICE_HAS_NET_TRANSPORT
    FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    FLAGCX_IR_NET_REF(net).put(team, peer, *dst, dstOffset, *src, srcOffset,
                               bytes, flagcxDevNet_None{}, flagcxDevNet_None{},
                               coop);
#else
    // No device net transport on this platform (Ascend 2201 today). The
    // compiler cannot prune this arm - useP2P is a runtime bool - and its
    // never-defined trap calls would fail the in--c device link of EVERY
    // put kernel even though the arm cannot run there: probe r144 N2 lost
    // on exactly that reference and r145's ld.lld context named the
    // referrer as the inlined kernel (n2k_mix_aic / .aiv). __builtin_trap
    // is not an option on that pass (lowers to abort; verifier refuses -
    // T0/T1 in device_utils.h), so the arm lands on the no-symbol
    // FLAGCX_DEVICE_TRAP, the same spelling the GDA provider-switch
    // fallbacks already accept. Why that is defensible rather than a
    // silent no-op: an INTRA-team peer is IPC-mapped, so peerPtr is
    // non-null and this arm is not reached; if the mapping itself failed
    // that is the open T2 alias defect, caught by every test's host-side
    // check of the results buffers, and the Ascend unified runner gates
    // the net-forcing fallback off (Stage D). Reaching it is UB by
    // construction - the same trade, documented in the same header, that
    // the provider-switch sites already make.
    FLAGCX_DEVICE_TRAP();
#endif
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void flagcxDevPut_RSigInc(
    const void FLAGCX_IR_GLOBAL_PTR *commOpaque,
    const void FLAGCX_IR_GLOBAL_PTR *dstOpaque, size_t dstOffset,
    const void FLAGCX_IR_GLOBAL_PTR *srcOpaque, size_t srcOffset, size_t bytes,
    flagcxDevTeamKind_t teamKind, int peer, flagcxDevContext_t contextId,
    flagcxDevCoopKind_t coopKind, flagcxDevMemoryScope_t scope,
    flagcxDevMemoryOrder_t order, flagcxDevSignal_t remoteSignal) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);

  bool shouldReturn;
  bool useP2P = flagcxValidateAndDispatch(*comm, team, peer, teamKind,
                                          shouldReturn);
  if (shouldReturn)
    return;
  auto peerPtr =
      useP2P ? flagcxGetPeerPointerWithAccess(*dst, dstOffset, team, peer,
                                              flagcxDevPeerAccessWriteOnly)
             : nullptr;
  useP2P = peerPtr != nullptr;

  if (useP2P) {
    auto localSrc = flagcxGetLocalPointer(*src, srcOffset);
    if (order == flagcxDeviceMemoryOrderRelease ||
        order == flagcxDeviceMemoryOrderAcqRel)
      flagcxScopedFence(scope);
    flagcxCoopMemcpy(coopKind, peerPtr, localSrc, bytes);
    // All threads fence to flush their own store buffers before signaling
    flagcxScopedFence(flagcxDeviceScopeSystem);
    // Signal after data lands
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    coop.sync();
    if (coop.threadRank() == 0) {
      flagcxDevSignalInc(commOpaque, teamKind, peer, remoteSignal, contextId,
                         FLAGCX_COOP_THREAD, flagcxDeviceScopeSystem);
    }
    coop.sync();
  } else {
    FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    FLAGCX_IR_NET_REF(net).put(team, peer, *dst, dstOffset, *src, srcOffset,
                               bytes, flagcxDevNet_SignalInc{remoteSignal},
                               flagcxDevNet_None{}, coop);
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void flagcxDevPut_RSigAdd(
    const void FLAGCX_IR_GLOBAL_PTR *commOpaque,
    const void FLAGCX_IR_GLOBAL_PTR *dstOpaque, size_t dstOffset,
    const void FLAGCX_IR_GLOBAL_PTR *srcOpaque, size_t srcOffset, size_t bytes,
    flagcxDevTeamKind_t teamKind, int peer, flagcxDevContext_t contextId,
    flagcxDevCoopKind_t coopKind, flagcxDevMemoryScope_t scope,
    flagcxDevMemoryOrder_t order, flagcxDevSignal_t remoteSignal,
    uint64_t signalValue) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);

  bool shouldReturn;
  bool useP2P = flagcxValidateAndDispatch(*comm, team, peer, teamKind,
                                          shouldReturn);
  if (shouldReturn)
    return;
  auto peerPtr =
      useP2P ? flagcxGetPeerPointerWithAccess(*dst, dstOffset, team, peer,
                                              flagcxDevPeerAccessWriteOnly)
             : nullptr;
  useP2P = peerPtr != nullptr;

  if (useP2P) {
    auto localSrc = flagcxGetLocalPointer(*src, srcOffset);
    if (order == flagcxDeviceMemoryOrderRelease ||
        order == flagcxDeviceMemoryOrderAcqRel)
      flagcxScopedFence(scope);
    flagcxCoopMemcpy(coopKind, peerPtr, localSrc, bytes);
    // All threads fence to flush their own store buffers before signaling
    flagcxScopedFence(flagcxDeviceScopeSystem);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    coop.sync();
    if (coop.threadRank() == 0) {
      flagcxDevSignalAdd(commOpaque, teamKind, peer, remoteSignal, signalValue,
                         contextId, FLAGCX_COOP_THREAD,
                         flagcxDeviceScopeSystem);
    }
    coop.sync();
  } else {
    FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    FLAGCX_IR_NET_REF(net).put(
        team, peer, *dst, dstOffset, *src, srcOffset, bytes,
        flagcxDevNet_SignalAdd{remoteSignal, signalValue}, flagcxDevNet_None{},
        coop);
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void flagcxDevPut_LCtrInc(
    const void FLAGCX_IR_GLOBAL_PTR *commOpaque,
    const void FLAGCX_IR_GLOBAL_PTR *dstOpaque, size_t dstOffset,
    const void FLAGCX_IR_GLOBAL_PTR *srcOpaque, size_t srcOffset, size_t bytes,
    flagcxDevTeamKind_t teamKind, int peer, flagcxDevContext_t contextId,
    flagcxDevCoopKind_t coopKind, flagcxDevMemoryScope_t scope,
    flagcxDevMemoryOrder_t order, flagcxDevCounter_t localCounter) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);

  bool shouldReturn;
  bool useP2P = flagcxValidateAndDispatch(*comm, team, peer, teamKind,
                                          shouldReturn);
  if (shouldReturn)
    return;
  auto peerPtr =
      useP2P ? flagcxGetPeerPointerWithAccess(*dst, dstOffset, team, peer,
                                              flagcxDevPeerAccessWriteOnly)
             : nullptr;
  FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
  useP2P = peerPtr != nullptr && comm->_commBase.supportsDirectCounterAccess();

  if (useP2P) {
    auto localSrc = flagcxGetLocalPointer(*src, srcOffset);
    if (order == flagcxDeviceMemoryOrderRelease ||
        order == flagcxDeviceMemoryOrderAcqRel)
      flagcxScopedFence(scope);
    flagcxCoopMemcpy(coopKind, peerPtr, localSrc, bytes);
    // Counter increment after data lands
    // All threads fence to flush their own store buffers before signaling
    flagcxScopedFence(flagcxDeviceScopeSystem);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    coop.sync();
    if (coop.threadRank() == 0) {
      // Counter is local to sender.
      flagcxCompleteCounterInternal(FLAGCX_IR_NET_REF(net), localCounter,
                                    (uint64_t)1, 0);
    }
    coop.sync();
  } else {
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    FLAGCX_IR_NET_REF(net).put(team, peer, *dst, dstOffset, *src, srcOffset,
                               bytes, flagcxDevNet_None{},
                               flagcxDevNet_CounterInc{localCounter}, coop);
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevPut_RSigInc_LCtrInc(
    const void FLAGCX_IR_GLOBAL_PTR *commOpaque,
    const void FLAGCX_IR_GLOBAL_PTR *dstOpaque, size_t dstOffset,
    const void FLAGCX_IR_GLOBAL_PTR *srcOpaque, size_t srcOffset, size_t bytes,
    flagcxDevTeamKind_t teamKind, int peer, flagcxDevContext_t contextId,
    flagcxDevCoopKind_t coopKind, flagcxDevMemoryScope_t scope,
    flagcxDevMemoryOrder_t order, flagcxDevSignal_t remoteSignal,
    flagcxDevCounter_t localCounter) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);

  bool shouldReturn;
  bool useP2P =
      flagcxValidateAndDispatch(*comm, team, peer, teamKind,
                                shouldReturn);
  if (shouldReturn)
    return;
  auto peerPtr =
      useP2P ? flagcxGetPeerPointerWithAccess(*dst, dstOffset, team, peer,
                                              flagcxDevPeerAccessWriteOnly)
             : nullptr;
  FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
  useP2P = peerPtr != nullptr && comm->_commBase.supportsDirectCounterAccess();

  if (useP2P) {
    auto localSrc = flagcxGetLocalPointer(*src, srcOffset);
    if (order == flagcxDeviceMemoryOrderRelease ||
        order == flagcxDeviceMemoryOrderAcqRel)
      flagcxScopedFence(scope);
    flagcxCoopMemcpy(coopKind, peerPtr, localSrc, bytes);
    flagcxScopedFence(flagcxDeviceScopeSystem);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    coop.sync();
    if (coop.threadRank() == 0) {
      // Remote signal increment
      flagcxDevSignalInc(commOpaque, teamKind, peer, remoteSignal, contextId,
                         FLAGCX_COOP_THREAD, flagcxDeviceScopeSystem);
      // Local counter increment
      flagcxCompleteCounterInternal(FLAGCX_IR_NET_REF(net), localCounter,
                                    (uint64_t)1, 0);
    }
    coop.sync();
  } else {
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    FLAGCX_IR_NET_REF(net).put(team, peer, *dst, dstOffset, *src, srcOffset,
                               bytes, flagcxDevNet_SignalInc{remoteSignal},
                               flagcxDevNet_CounterInc{localCounter}, coop);
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevPut_RSigAdd_LCtrInc(
    const void FLAGCX_IR_GLOBAL_PTR *commOpaque,
    const void FLAGCX_IR_GLOBAL_PTR *dstOpaque, size_t dstOffset,
    const void FLAGCX_IR_GLOBAL_PTR *srcOpaque, size_t srcOffset, size_t bytes,
    flagcxDevTeamKind_t teamKind, int peer, flagcxDevContext_t contextId,
    flagcxDevCoopKind_t coopKind, flagcxDevMemoryScope_t scope,
    flagcxDevMemoryOrder_t order, flagcxDevSignal_t remoteSignal,
    uint64_t signalValue, flagcxDevCounter_t localCounter) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);

  bool shouldReturn;
  bool useP2P =
      flagcxValidateAndDispatch(*comm, team, peer, teamKind,
                                shouldReturn);
  if (shouldReturn)
    return;
  auto peerPtr =
      useP2P ? flagcxGetPeerPointerWithAccess(*dst, dstOffset, team, peer,
                                              flagcxDevPeerAccessWriteOnly)
             : nullptr;
  FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
  useP2P = peerPtr != nullptr && comm->_commBase.supportsDirectCounterAccess();

  if (useP2P) {
    auto localSrc = flagcxGetLocalPointer(*src, srcOffset);
    if (order == flagcxDeviceMemoryOrderRelease ||
        order == flagcxDeviceMemoryOrderAcqRel)
      flagcxScopedFence(scope);
    flagcxCoopMemcpy(coopKind, peerPtr, localSrc, bytes);
    flagcxScopedFence(flagcxDeviceScopeSystem);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    coop.sync();
    if (coop.threadRank() == 0) {
      // Remote signal add
      flagcxDevSignalAdd(commOpaque, teamKind, peer, remoteSignal, signalValue,
                         contextId, FLAGCX_COOP_THREAD,
                         flagcxDeviceScopeSystem);
      // Local counter increment
      flagcxCompleteCounterInternal(FLAGCX_IR_NET_REF(net), localCounter,
                                    (uint64_t)1, 0);
    }
    coop.sync();
  } else {
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    FLAGCX_IR_NET_REF(net).put(
        team, peer, *dst, dstOffset, *src, srcOffset, bytes,
        flagcxDevNet_SignalAdd{remoteSignal, signalValue},
        flagcxDevNet_CounterInc{localCounter}, coop);
  }
}

/* ================================================================
 * Category U2: Unified Get (1)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevGet(const void FLAGCX_IR_GLOBAL_PTR *commOpaque,
             const void FLAGCX_IR_GLOBAL_PTR *srcOpaque, size_t srcOffset,
             const void FLAGCX_IR_GLOBAL_PTR *dstOpaque, size_t dstOffset,
             size_t bytes, flagcxDevTeamKind_t teamKind, int peer,
             flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
             flagcxDevMemoryScope_t scope, flagcxDevMemoryOrder_t order) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);

  bool shouldReturn;
  bool useP2P = flagcxValidateAndDispatch(*comm, team, peer, teamKind,
                                          shouldReturn);
  if (shouldReturn)
    return;
  auto peerPtr =
      useP2P ? flagcxGetPeerPointerWithAccess(*src, srcOffset, team, peer,
                                              flagcxDevPeerAccessReadOnly)
             : nullptr;
  useP2P = peerPtr != nullptr;

  if (useP2P) {
    auto localDst = flagcxGetLocalPointer(*dst, dstOffset);
    flagcxCoopMemcpy(coopKind, localDst, peerPtr, bytes);
    if (order == flagcxDeviceMemoryOrderAcquire ||
        order == flagcxDeviceMemoryOrderAcqRel)
      flagcxScopedFence(scope);
  } else {
#if FLAGCX_DEVICE_HAS_NET_TRANSPORT
    FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    FLAGCX_IR_NET_REF(net).get(team, peer, *src, srcOffset, *dst, dstOffset,
                               bytes, coop);
#else
    // Same gate and reasoning as flagcxDevPut.  On Ascend 2201 today this
    // instantiation never happens - the S19 launcher is deliberately not
    // defined there (see test/kernel/ascend/device_ir.asc) - the arm is
    // pruned so the put/get/putValue family stays consistent for any
    // future platform that shares the flag.
    FLAGCX_DEVICE_TRAP();
#endif
  }
}

/* ================================================================
 * Category U3: Unified PutValue (3)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevPutValue(const void FLAGCX_IR_GLOBAL_PTR *commOpaque,
                  const void FLAGCX_IR_GLOBAL_PTR *dstOpaque, size_t dstOffset,
                  uint64_t value, flagcxDevTeamKind_t teamKind, int peer,
                  flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
                  flagcxDevMemoryScope_t scope, flagcxDevMemoryOrder_t order) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);

  bool shouldReturn;
  bool useP2P = flagcxValidateAndDispatch(*comm, team, peer, teamKind,
                                          shouldReturn);
  if (shouldReturn)
    return;
  auto peerPtr =
      useP2P ? flagcxGetPeerPointerWithAccess(*dst, dstOffset, team, peer,
                                              flagcxDevPeerAccessWriteOnly)
             : nullptr;
  useP2P = peerPtr != nullptr;

  if (useP2P) {
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    coop.sync();
    if (coop.threadRank() == 0) {
      if (order == flagcxDeviceMemoryOrderRelease ||
          order == flagcxDeviceMemoryOrderAcqRel)
        flagcxScopedFence(scope);
      flagcxStoreVolatile64Internal(peerPtr, value);
    }
    coop.sync();
  } else {
#if FLAGCX_DEVICE_HAS_NET_TRANSPORT
    FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    FLAGCX_IR_NET_REF(net).putValue(team, peer, *dst, dstOffset, value,
                                    flagcxDevNet_None{}, coop);
#else
    // Same gate and same reasoning as the plain flagcxDevPut above (probe
    // r144/r145: the runtime useP2P guard does not spare these references
    // from codegen on a pass where the net glue's ops are never-defined
    // symbols). An INTRA-team peer is IPC-mapped, so this arm is not
    // reached; reaching it is UB by construction, documented at put.
    FLAGCX_DEVICE_TRAP();
#endif
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevPutValue_RSigInc(const void FLAGCX_IR_GLOBAL_PTR *commOpaque,
                          const void FLAGCX_IR_GLOBAL_PTR *dstOpaque,
                          size_t dstOffset, uint64_t value,
                          flagcxDevTeamKind_t teamKind, int peer,
                          flagcxDevContext_t contextId,
                          flagcxDevCoopKind_t coopKind,
                          flagcxDevMemoryScope_t scope,
                          flagcxDevMemoryOrder_t order,
                          flagcxDevSignal_t remoteSignal) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);

  bool shouldReturn;
  bool useP2P = flagcxValidateAndDispatch(
      *comm, team, peer, teamKind, shouldReturn);
  if (shouldReturn)
    return;
  auto peerPtr =
      useP2P ? flagcxGetPeerPointerWithAccess(*dst, dstOffset, team, peer,
                                              flagcxDevPeerAccessWriteOnly)
             : nullptr;
  useP2P = peerPtr != nullptr;

  if (useP2P) {
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    coop.sync();
    if (coop.threadRank() == 0) {
      if (order == flagcxDeviceMemoryOrderRelease ||
          order == flagcxDeviceMemoryOrderAcqRel)
        flagcxScopedFence(scope);
      flagcxStoreVolatile64Internal(peerPtr, value);
      flagcxScopedFence(flagcxDeviceScopeSystem);
      flagcxDevSignalInc(commOpaque, teamKind, peer, remoteSignal, contextId,
                         FLAGCX_COOP_THREAD, flagcxDeviceScopeSystem);
    }
    coop.sync();
  } else {
    FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    FLAGCX_IR_NET_REF(net).putValue(team, peer, *dst, dstOffset, value,
                                    flagcxDevNet_SignalInc{remoteSignal}, coop);
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevPutValue_RSigAdd(
    const void FLAGCX_IR_GLOBAL_PTR *commOpaque,
    const void FLAGCX_IR_GLOBAL_PTR *dstOpaque, size_t dstOffset,
    uint64_t value, flagcxDevTeamKind_t teamKind, int peer,
    flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
    flagcxDevMemoryScope_t scope, flagcxDevMemoryOrder_t order,
    flagcxDevSignal_t remoteSignal, uint64_t signalValue) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);

  bool shouldReturn;
  bool useP2P = flagcxValidateAndDispatch(
      *comm, team, peer, teamKind, shouldReturn);
  if (shouldReturn)
    return;
  auto peerPtr =
      useP2P ? flagcxGetPeerPointerWithAccess(*dst, dstOffset, team, peer,
                                              flagcxDevPeerAccessWriteOnly)
             : nullptr;
  useP2P = peerPtr != nullptr;

  if (useP2P) {
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    coop.sync();
    if (coop.threadRank() == 0) {
      if (order == flagcxDeviceMemoryOrderRelease ||
          order == flagcxDeviceMemoryOrderAcqRel)
        flagcxScopedFence(scope);
      flagcxStoreVolatile64Internal(peerPtr, value);
      flagcxScopedFence(flagcxDeviceScopeSystem);
      flagcxDevSignalAdd(commOpaque, teamKind, peer, remoteSignal, signalValue,
                         contextId, FLAGCX_COOP_THREAD,
                         flagcxDeviceScopeSystem);
    }
    coop.sync();
  } else {
    FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    FLAGCX_IR_NET_REF(net).putValue(
        team, peer, *dst, dstOffset, value,
        flagcxDevNet_SignalAdd{remoteSignal, signalValue}, coop);
  }
}

#endif // FLAGCX_DEVICE_UNIFIED_IR_IMPL_H_
