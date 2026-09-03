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
 * Internal helper: cooperative memcpy (P2P path)
 *
 * The copy itself lives in the platform layer: choosing the chunk width needs
 * the platform's alignment rules and its address-space qualifier, neither of
 * which this file can spell portably. The pointer types stay deduced so that a
 * platform whose device pointers carry an address-space tag can pass them
 * through unchanged.
 * ================================================================ */

template <typename DstPtr, typename SrcPtr>
static FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCoopMemcpy(flagcxDevCoopKind_t coopKind, DstPtr dst, SrcPtr src,
                 size_t bytes) {
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  DeviceAPI::Intrin::coopCopyBytes(dst, src, bytes, coop.threadRank(),
                                   coop.size());
}

/* ================================================================
 * Category U4: Unified Signal (2)
 *
 * DEFINED FIRST — Put variants call these for P2P signal delivery.
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevSignalInc(const void *commOpaque, flagcxDevTeamKind_t teamKind,
                   int peer, flagcxDevSignal_t signal,
                   flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
                   flagcxDevMemoryScope_t scope) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  // Resolve team-scoped peer to local rank for P2P indexing
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);

  int worldPeer = flagcxTeamRankToWorld(*comm, team, peer);
  int localPeer =
      worldPeer - (comm->_commBase.getRank() - comm->_commBase.getIntraRank());
  if (comm->_commBase.usesDirectP2pSignals() && localPeer >= 0 &&
      localPeer < comm->_commBase.getIntraSize()) {
    // P2P fast path: one atomic per cooperative group.
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    coop.sync();
    if (coop.threadRank() == 0) {
      FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
      auto peerSignal = FLAGCX_IR_NET_REF(net).getPeerSignalPtr(localPeer, signal);
      DeviceAPI::Atomic::fetchAdd(peerSignal, (uint64_t)1,
                                  flagcxDeviceMemoryOrderRelease);
    }
    coop.sync();
  } else {
    // Net FIFO fallback (inter-node or P2P not available)
    FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
    flagcxDevNetSignalSigIncS(FLAGCX_IR_NET_ARG(net), commOpaque, teamKind, peer, coopKind,
                              signal);
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevSignalAdd(const void *commOpaque, flagcxDevTeamKind_t teamKind,
                   int peer, flagcxDevSignal_t signal, uint64_t value,
                   flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
                   flagcxDevMemoryScope_t scope) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  // Resolve team-scoped peer to local rank for P2P indexing
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);

  int worldPeer = flagcxTeamRankToWorld(*comm, team, peer);
  int localPeer =
      worldPeer - (comm->_commBase.getRank() - comm->_commBase.getIntraRank());
  if (comm->_commBase.usesDirectP2pSignals() && localPeer >= 0 &&
      localPeer < comm->_commBase.getIntraSize()) {
    // P2P fast path: one atomic per cooperative group.
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    coop.sync();
    if (coop.threadRank() == 0) {
      FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
      auto peerSignal = FLAGCX_IR_NET_REF(net).getPeerSignalPtr(localPeer, signal);

      DeviceAPI::Atomic::fetchAdd(peerSignal, value,
                                  flagcxDeviceMemoryOrderRelease);
    }
    coop.sync();
  } else {
    // Net FIFO fallback (inter-node or P2P not available)
    FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
    flagcxDevNetSignalSigAddS(FLAGCX_IR_NET_ARG(net), commOpaque, teamKind, peer, coopKind, signal,
                              value);
  }
}

/* ================================================================
 * Category U5: Unified Wait (2)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevWaitSignal(const void *commOpaque, flagcxDevSignal_t signal,
                    uint64_t least, int bits, flagcxDevContext_t contextId,
                    flagcxDevCoopKind_t coopKind,
                    flagcxDevMemoryOrder_t order) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;

  // Send and wait share one communicator-wide signal transport decision.
  if (comm->_commBase.usesDirectP2pSignals()) {
    FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
    auto localSignal = FLAGCX_IR_NET_REF(net).getSignalPtr(signal);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);

    coop.sync();
    if (coop.threadRank() == 0) {
      int iter = 0;
      while (DeviceAPI::Atomic::load(localSignal, order) < least) {
        DeviceAPI::Intrin::spinBackoff(iter++);
      }
    }
    coop.sync();
  } else {
    // Net FIFO path for multi-node
    FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
    flagcxDevNetWaitSignalS(FLAGCX_IR_NET_ARG(net), coopKind, signal, least, bits, order);
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevWaitCounter(const void *commOpaque, flagcxDevCounter_t counter,
                     uint64_t least, int bits, flagcxDevContext_t contextId,
                     flagcxDevCoopKind_t coopKind,
                     flagcxDevMemoryOrder_t order) {
  FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
  flagcxDevNetWaitCounterS(FLAGCX_IR_NET_ARG(net), coopKind, counter, least, bits, order);
}

/* ================================================================
 * Category U6: Unified Read (2)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxDevReadSignal(
    const void *commOpaque, flagcxDevSignal_t signal, int bits,
    flagcxDevContext_t contextId, flagcxDevMemoryOrder_t order) {
  FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
  return flagcxDevNetReadSignalS(FLAGCX_IR_NET_ARG(net), signal, bits, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxDevReadCounter(
    const void *commOpaque, flagcxDevCounter_t counter, int bits,
    flagcxDevContext_t contextId, flagcxDevMemoryOrder_t order) {
  FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
  return flagcxDevNetReadCounterS(FLAGCX_IR_NET_ARG(net), counter, bits, order);
}

/* ================================================================
 * Category U7: Unified Flush / Reset / Shadow (4)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevFlush(const void *commOpaque, flagcxDevContext_t contextId,
               flagcxDevCoopKind_t coopKind, flagcxDevMemoryOrder_t order) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;

  // A communicator may use IPC for one memory object and Net fallback for
  // another, so flush both domains.  Flushing an empty FIFO is harmless.
  DeviceAPI::Intrin::threadfenceSystem();
  if (comm->_commBase.isOneSidedTransportReady() &&
      flagcxDevCommHasNetContexts(*comm)) {
    FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
    flagcxDevNetFlushS(FLAGCX_IR_NET_ARG(net), coopKind, order);
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevResetSignal(const void *commOpaque, flagcxDevContext_t contextId,
                     flagcxDevSignal_t slot) {
  FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
  flagcxDevNetResetSignal(FLAGCX_IR_NET_ARG(net), slot);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevResetCounter(const void *commOpaque, flagcxDevContext_t contextId,
                      flagcxDevCounter_t slot) {
  FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
  flagcxDevNetResetCounter(FLAGCX_IR_NET_ARG(net), slot);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevIncreaseSignalShadow(const void *commOpaque,
                              flagcxDevContext_t contextId,
                              flagcxDevSignal_t slot, uint64_t delta) {
  FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
  flagcxDevNetIncreaseSignalShadow(FLAGCX_IR_NET_ARG(net), slot, delta);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevWaitSignalMeetShadow(const void *commOpaque,
                              flagcxDevContext_t contextId,
                              flagcxDevSignal_t slot, int bits,
                              flagcxDevCoopKind_t coopKind,
                              flagcxDevMemoryOrder_t order) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;

  if (comm->_commBase.usesDirectP2pSignals()) {
    FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
    auto signalPtr = FLAGCX_IR_NET_REF(net).getSignalPtr(slot);
    auto shadowPtr = FLAGCX_IR_NET_REF(net).getSignalShadowPtr(slot);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);

    coop.sync();
    if (coop.threadRank() == 0) {
      uint64_t expectedVal = DeviceAPI::Atomic::load(shadowPtr, order);
      int iter = 0;
      while (DeviceAPI::Atomic::load(signalPtr, order) < expectedVal) {
        DeviceAPI::Intrin::spinBackoff(iter++);
      }
    }
    coop.sync();
  } else {
    // Net FIFO path for multi-node
    FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
    flagcxDevNetWaitSignalMeetShadowS(FLAGCX_IR_NET_ARG(net), coopKind, slot, bits, order);
  }
}

/* ================================================================
 * Category U8: Unified Barrier (3)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void flagcxDevBarrierArrive(
    const void *commOpaque, flagcxDevTeamKind_t teamKind, uint32_t index,
    flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
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
      flagcxWorldBarrierArriveS(FLAGCX_IR_NET_ARG(net), coopKind, index, /*multimem=*/false, order,
                                flagcxDevNetFenceLevel::Relaxed);
      break;
    }
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevBarrierWait(const void *commOpaque, flagcxDevTeamKind_t teamKind,
                     uint32_t index, flagcxDevContext_t contextId,
                     flagcxDevCoopKind_t coopKind, flagcxDevMemoryOrder_t order,
                     flagcxDevMemoryScope_t scope) {
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
      flagcxWorldBarrierWaitS(FLAGCX_IR_NET_ARG(net), coopKind, index, /*multimem=*/false, order,
                              flagcxDevNetFenceLevel::Relaxed);
      break;
    }
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevBarrierSync(const void *commOpaque, flagcxDevTeamKind_t teamKind,
                     uint32_t index, flagcxDevContext_t contextId,
                     flagcxDevCoopKind_t coopKind, flagcxDevMemoryOrder_t order,
                     flagcxDevMemoryScope_t scope) {
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
      flagcxWorldBarrierSyncS(FLAGCX_IR_NET_ARG(net), coopKind, index, /*multimem=*/false, order,
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
static FLAGCX_DEVICE_INLINE_DECORATOR bool
flagcxValidateAndDispatch(const flagcxDevComm &comm, const flagcxTeam &team,
                          int peer, flagcxDevTeamKind_t teamKind,
                          const char *funcName, bool &shouldReturn) {
  (void)funcName;
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
flagcxDevPut(const void *commOpaque, const void *dstOpaque, size_t dstOffset,
             const void *srcOpaque, size_t srcOffset, size_t bytes,
             flagcxDevTeamKind_t teamKind, int peer,
             flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
             flagcxDevMemoryScope_t scope, flagcxDevMemoryOrder_t order) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);

  bool shouldReturn;
  bool useP2P = flagcxValidateAndDispatch(*comm, team, peer, teamKind,
                                          "flagcxDevPut", shouldReturn);
  if (shouldReturn)
    return;
  auto peerPtr =
      useP2P ? flagcxGetPeerPointer(*dst, dstOffset, team, peer,
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
    FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    FLAGCX_IR_NET_REF(net).put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_None{}, flagcxDevNet_None{}, coop);
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevPut_RSigInc(const void *commOpaque, const void *dstOpaque,
                     size_t dstOffset, const void *srcOpaque, size_t srcOffset,
                     size_t bytes, flagcxDevTeamKind_t teamKind, int peer,
                     flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
                     flagcxDevMemoryScope_t scope, flagcxDevMemoryOrder_t order,
                     flagcxDevSignal_t remoteSignal) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);

  bool shouldReturn;
  bool useP2P = flagcxValidateAndDispatch(*comm, team, peer, teamKind,
                                          "flagcxDevPut_RSigInc", shouldReturn);
  if (shouldReturn)
    return;
  auto peerPtr =
      useP2P ? flagcxGetPeerPointer(*dst, dstOffset, team, peer,
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
    FLAGCX_IR_NET_REF(net).put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_SignalInc{remoteSignal}, flagcxDevNet_None{}, coop);
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevPut_RSigAdd(const void *commOpaque, const void *dstOpaque,
                     size_t dstOffset, const void *srcOpaque, size_t srcOffset,
                     size_t bytes, flagcxDevTeamKind_t teamKind, int peer,
                     flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
                     flagcxDevMemoryScope_t scope, flagcxDevMemoryOrder_t order,
                     flagcxDevSignal_t remoteSignal, uint64_t signalValue) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);

  bool shouldReturn;
  bool useP2P = flagcxValidateAndDispatch(*comm, team, peer, teamKind,
                                          "flagcxDevPut_RSigAdd", shouldReturn);
  if (shouldReturn)
    return;
  auto peerPtr =
      useP2P ? flagcxGetPeerPointer(*dst, dstOffset, team, peer,
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
    FLAGCX_IR_NET_REF(net).put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_SignalAdd{remoteSignal, signalValue},
             flagcxDevNet_None{}, coop);
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevPut_LCtrInc(const void *commOpaque, const void *dstOpaque,
                     size_t dstOffset, const void *srcOpaque, size_t srcOffset,
                     size_t bytes, flagcxDevTeamKind_t teamKind, int peer,
                     flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
                     flagcxDevMemoryScope_t scope, flagcxDevMemoryOrder_t order,
                     flagcxDevCounter_t localCounter) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);

  bool shouldReturn;
  bool useP2P = flagcxValidateAndDispatch(*comm, team, peer, teamKind,
                                          "flagcxDevPut_LCtrInc", shouldReturn);
  if (shouldReturn)
    return;
  auto peerPtr =
      useP2P ? flagcxGetPeerPointer(*dst, dstOffset, team, peer,
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
      DeviceAPI::Atomic::fetchAdd(FLAGCX_IR_NET_REF(net).getCounterPtr(localCounter), (uint64_t)1,
                                  flagcxDeviceMemoryOrderRelease);
    }
    coop.sync();
  } else {
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    FLAGCX_IR_NET_REF(net).put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_None{}, flagcxDevNet_CounterInc{localCounter}, coop);
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevPut_RSigInc_LCtrInc(
    const void *commOpaque, const void *dstOpaque, size_t dstOffset,
    const void *srcOpaque, size_t srcOffset, size_t bytes,
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
                                "flagcxDevPut_RSigInc_LCtrInc", shouldReturn);
  if (shouldReturn)
    return;
  auto peerPtr =
      useP2P ? flagcxGetPeerPointer(*dst, dstOffset, team, peer,
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
      DeviceAPI::Atomic::fetchAdd(FLAGCX_IR_NET_REF(net).getCounterPtr(localCounter), (uint64_t)1,
                                  flagcxDeviceMemoryOrderRelease);
    }
    coop.sync();
  } else {
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    FLAGCX_IR_NET_REF(net).put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_SignalInc{remoteSignal},
             flagcxDevNet_CounterInc{localCounter}, coop);
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevPut_RSigAdd_LCtrInc(
    const void *commOpaque, const void *dstOpaque, size_t dstOffset,
    const void *srcOpaque, size_t srcOffset, size_t bytes,
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
                                "flagcxDevPut_RSigAdd_LCtrInc", shouldReturn);
  if (shouldReturn)
    return;
  auto peerPtr =
      useP2P ? flagcxGetPeerPointer(*dst, dstOffset, team, peer,
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
      DeviceAPI::Atomic::fetchAdd(FLAGCX_IR_NET_REF(net).getCounterPtr(localCounter), (uint64_t)1,
                                  flagcxDeviceMemoryOrderRelease);
    }
    coop.sync();
  } else {
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    FLAGCX_IR_NET_REF(net).put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
             flagcxDevNet_SignalAdd{remoteSignal, signalValue},
             flagcxDevNet_CounterInc{localCounter}, coop);
  }
}

/* ================================================================
 * Category U2: Unified Get (1)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevGet(const void *commOpaque, const void *srcOpaque, size_t srcOffset,
             const void *dstOpaque, size_t dstOffset, size_t bytes,
             flagcxDevTeamKind_t teamKind, int peer,
             flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
             flagcxDevMemoryScope_t scope, flagcxDevMemoryOrder_t order) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *src = (const flagcxDevMem *)srcOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);

  bool shouldReturn;
  bool useP2P = flagcxValidateAndDispatch(*comm, team, peer, teamKind,
                                          "flagcxDevGet", shouldReturn);
  if (shouldReturn)
    return;
  auto peerPtr =
      useP2P ? flagcxGetPeerPointer(*src, srcOffset, team, peer) : nullptr;
  useP2P = peerPtr != nullptr;

  if (useP2P) {
    auto localDst = flagcxGetLocalPointer(*dst, dstOffset);
    flagcxCoopMemcpy(coopKind, localDst, peerPtr, bytes);
    if (order == flagcxDeviceMemoryOrderAcquire ||
        order == flagcxDeviceMemoryOrderAcqRel)
      flagcxScopedFence(scope);
  } else {
    FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    FLAGCX_IR_NET_REF(net).get(team, peer, *src, srcOffset, *dst, dstOffset, bytes, coop);
  }
}

/* ================================================================
 * Category U3: Unified PutValue (3)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevPutValue(const void *commOpaque, const void *dstOpaque,
                  size_t dstOffset, uint64_t value,
                  flagcxDevTeamKind_t teamKind, int peer,
                  flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
                  flagcxDevMemoryScope_t scope, flagcxDevMemoryOrder_t order) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);

  bool shouldReturn;
  bool useP2P = flagcxValidateAndDispatch(*comm, team, peer, teamKind,
                                          "flagcxDevPutValue", shouldReturn);
  if (shouldReturn)
    return;
  auto peerPtr =
      useP2P ? flagcxGetPeerPointer(*dst, dstOffset, team, peer,
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
      DeviceAPI::Intrin::storeVolatile64(peerPtr, value);
    }
    coop.sync();
  } else {
    FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    FLAGCX_IR_NET_REF(net).putValue(team, peer, *dst, dstOffset, value, flagcxDevNet_None{},
                  coop);
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevPutValue_RSigInc(const void *commOpaque, const void *dstOpaque,
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
      *comm, team, peer, teamKind, "flagcxDevPutValue_RSigInc", shouldReturn);
  if (shouldReturn)
    return;
  auto peerPtr =
      useP2P ? flagcxGetPeerPointer(*dst, dstOffset, team, peer,
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
      DeviceAPI::Intrin::storeVolatile64(peerPtr, value);
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
    const void *commOpaque, const void *dstOpaque, size_t dstOffset,
    uint64_t value, flagcxDevTeamKind_t teamKind, int peer,
    flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
    flagcxDevMemoryScope_t scope, flagcxDevMemoryOrder_t order,
    flagcxDevSignal_t remoteSignal, uint64_t signalValue) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevMem *dst = (const flagcxDevMem *)dstOpaque;
  flagcxTeam team = flagcxMakeTeamFromKind(*comm, teamKind);

  bool shouldReturn;
  bool useP2P = flagcxValidateAndDispatch(
      *comm, team, peer, teamKind, "flagcxDevPutValue_RSigAdd", shouldReturn);
  if (shouldReturn)
    return;
  auto peerPtr =
      useP2P ? flagcxGetPeerPointer(*dst, dstOffset, team, peer,
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
      DeviceAPI::Intrin::storeVolatile64(peerPtr, value);
      flagcxScopedFence(flagcxDeviceScopeSystem);
      flagcxDevSignalAdd(commOpaque, teamKind, peer, remoteSignal, signalValue,
                         contextId, FLAGCX_COOP_THREAD,
                         flagcxDeviceScopeSystem);
    }
    coop.sync();
  } else {
    FLAGCX_IR_NET_DECL(net, commOpaque, contextId);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    FLAGCX_IR_NET_REF(net).putValue(team, peer, *dst, dstOffset, value,
                  flagcxDevNet_SignalAdd{remoteSignal, signalValue}, coop);
  }
}

#endif // FLAGCX_DEVICE_UNIFIED_IR_IMPL_H_
