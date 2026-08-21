/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX Unified One-Sided IR — Implementation.
 *
 * Transport-transparent dispatch: checks flagcxGetPeerPointer() for P2P
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
      __threadfence_system();
      break;
    case flagcxDeviceScopeDevice:
      __threadfence();
      break;
    default:
      break; // Block/Thread: no fence needed
  }
}

/* ================================================================
 * Internal helper: cooperative memcpy (P2P path)
 *
 * Distributes copy across threads using largest aligned chunks possible.
 * Cascades from 16B vectors down to byte-level for unaligned data.
 * Pattern adopted from NVSHMEM for stronger memory ordering guarantees.
 * ================================================================ */

static FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxCoopMemcpy(flagcxDevCoopKind_t coopKind, void *dst, const void *src,
                 size_t bytes) {
  flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
  int rank = coop.threadRank();
  int size = coop.size();

  // Try 16B aligned vector copy (int4 = 128-bit)
  if (((uintptr_t)dst % 16 == 0) && ((uintptr_t)src % 16 == 0)) {
    int4 *d = (int4 *)dst;
    const int4 *s = (const int4 *)src;
    size_t nelems = bytes / 16;
    for (size_t i = (size_t)rank; i < nelems; i += (size_t)size) {
      d[i] = s[i];
    }
    bytes -= nelems * 16;
    if (bytes == 0)
      return;
    dst = (void *)(d + nelems);
    src = (const void *)(s + nelems);
  }

  // Try 8B aligned copy (uint64_t)
  if (((uintptr_t)dst % 8 == 0) && ((uintptr_t)src % 8 == 0)) {
    uint64_t *d = (uint64_t *)dst;
    const uint64_t *s = (const uint64_t *)src;
    size_t nelems = bytes / 8;
    for (size_t i = (size_t)rank; i < nelems; i += (size_t)size) {
      d[i] = s[i];
    }
    bytes -= nelems * 8;
    if (bytes == 0)
      return;
    dst = (void *)(d + nelems);
    src = (const void *)(s + nelems);
  }

  // Try 4B aligned copy (uint32_t)
  if (((uintptr_t)dst % 4 == 0) && ((uintptr_t)src % 4 == 0)) {
    uint32_t *d = (uint32_t *)dst;
    const uint32_t *s = (const uint32_t *)src;
    size_t nelems = bytes / 4;
    for (size_t i = (size_t)rank; i < nelems; i += (size_t)size) {
      d[i] = s[i];
    }
    bytes -= nelems * 4;
    if (bytes == 0)
      return;
    dst = (void *)(d + nelems);
    src = (const void *)(s + nelems);
  }

  // Try 2B aligned copy (uint16_t)
  if (((uintptr_t)dst % 2 == 0) && ((uintptr_t)src % 2 == 0)) {
    uint16_t *d = (uint16_t *)dst;
    const uint16_t *s = (const uint16_t *)src;
    size_t nelems = bytes / 2;
    for (size_t i = (size_t)rank; i < nelems; i += (size_t)size) {
      d[i] = s[i];
    }
    bytes -= nelems * 2;
    if (bytes == 0)
      return;
    dst = (void *)(d + nelems);
    src = (const void *)(s + nelems);
  }

  // Fallback: byte-level copy for remainder or unaligned data
  unsigned char *d = (unsigned char *)dst;
  const unsigned char *s = (const unsigned char *)src;
  for (size_t i = (size_t)rank; i < bytes; i += (size_t)size) {
    d[i] = s[i];
  }
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
      const void *netOpaque = flagcxDevNetGetFromCommS(commOpaque, contextId);
      const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
      uint64_t *peerSignal = net->getPeerSignalPtr(localPeer, signal);
      DeviceAPI::Atomic::fetchAdd(peerSignal, (uint64_t)1,
                                  flagcxDeviceMemoryOrderRelease);
    }
    coop.sync();
  } else {
    // Net FIFO fallback (inter-node or P2P not available)
    const void *net = flagcxDevNetGetFromCommS(commOpaque, contextId);
    flagcxDevNetSignalSigIncS(net, commOpaque, teamKind, peer, coopKind,
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
      const void *netOpaque = flagcxDevNetGetFromCommS(commOpaque, contextId);
      const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
      uint64_t *peerSignal = net->getPeerSignalPtr(localPeer, signal);

      DeviceAPI::Atomic::fetchAdd(peerSignal, value,
                                  flagcxDeviceMemoryOrderRelease);
    }
    coop.sync();
  } else {
    // Net FIFO fallback (inter-node or P2P not available)
    const void *net = flagcxDevNetGetFromCommS(commOpaque, contextId);
    flagcxDevNetSignalSigAddS(net, commOpaque, teamKind, peer, coopKind, signal,
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
    const void *netOpaque = flagcxDevNetGetFromCommS(commOpaque, contextId);
    const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
    uint64_t *localSignal = net->getSignalPtr(signal);
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
    const void *net = flagcxDevNetGetFromCommS(commOpaque, contextId);
    flagcxDevNetWaitSignalS(net, coopKind, signal, least, bits, order);
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevWaitCounter(const void *commOpaque, flagcxDevCounter_t counter,
                     uint64_t least, int bits, flagcxDevContext_t contextId,
                     flagcxDevCoopKind_t coopKind,
                     flagcxDevMemoryOrder_t order) {
  const void *net = flagcxDevNetGetFromCommS(commOpaque, contextId);
  flagcxDevNetWaitCounterS(net, coopKind, counter, least, bits, order);
}

/* ================================================================
 * Category U6: Unified Read (2)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxDevReadSignal(
    const void *commOpaque, flagcxDevSignal_t signal, int bits,
    flagcxDevContext_t contextId, flagcxDevMemoryOrder_t order) {
  const void *net = flagcxDevNetGetFromCommS(commOpaque, contextId);
  return flagcxDevNetReadSignalS(net, signal, bits, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR uint64_t flagcxDevReadCounter(
    const void *commOpaque, flagcxDevCounter_t counter, int bits,
    flagcxDevContext_t contextId, flagcxDevMemoryOrder_t order) {
  const void *net = flagcxDevNetGetFromCommS(commOpaque, contextId);
  return flagcxDevNetReadCounterS(net, counter, bits, order);
}

/* ================================================================
 * Category U7: Unified Flush / Reset / Shadow (4)
 * ================================================================ */

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevFlush(const void *commOpaque, flagcxDevContext_t contextId,
               flagcxDevCoopKind_t coopKind, flagcxDevMemoryOrder_t order) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;
  const flagcxDevNet *net =
      (const flagcxDevNet *)flagcxDevNetGetFromCommS(commOpaque, contextId);

  // A communicator may use IPC for one memory object and Net fallback for
  // another, so flush both domains.  Flushing an empty FIFO is harmless.
  DeviceAPI::Intrin::threadfenceSystem();
  if (comm->_commBase.isOneSidedTransportReady() && net) {
    flagcxDevNetFlushS((const void *)net, coopKind, order);
  }
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevResetSignal(const void *commOpaque, flagcxDevContext_t contextId,
                     flagcxDevSignal_t slot) {
  const void *net = flagcxDevNetGetFromCommS(commOpaque, contextId);
  flagcxDevNetResetSignal(net, slot);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevResetCounter(const void *commOpaque, flagcxDevContext_t contextId,
                      flagcxDevCounter_t slot) {
  const void *net = flagcxDevNetGetFromCommS(commOpaque, contextId);
  flagcxDevNetResetCounter(net, slot);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevIncreaseSignalShadow(const void *commOpaque,
                              flagcxDevContext_t contextId,
                              flagcxDevSignal_t slot, uint64_t delta) {
  const void *net = flagcxDevNetGetFromCommS(commOpaque, contextId);
  flagcxDevNetIncreaseSignalShadow(net, slot, delta);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevWaitSignalMeetShadow(const void *commOpaque,
                              flagcxDevContext_t contextId,
                              flagcxDevSignal_t slot, int bits,
                              flagcxDevCoopKind_t coopKind,
                              flagcxDevMemoryOrder_t order) {
  const flagcxDevComm *comm = (const flagcxDevComm *)commOpaque;

  if (comm->_commBase.usesDirectP2pSignals()) {
    const void *netOpaque = flagcxDevNetGetFromCommS(commOpaque, contextId);
    const flagcxDevNet *net = (const flagcxDevNet *)netOpaque;
    uint64_t *signalPtr = net->getSignalPtr(slot);
    uint64_t *shadowPtr = net->getSignalShadowPtr(slot);
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
    const void *net = flagcxDevNetGetFromCommS(commOpaque, contextId);
    flagcxDevNetWaitSignalMeetShadowS(net, coopKind, slot, bits, order);
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
      const void *net = flagcxDevNetGetFromCommS(commOpaque, contextId);
      flagcxInterBarrierArriveS(net, coopKind, index, order,
                                flagcxDevNetFenceLevel::Relaxed);
      break;
    }
    case FLAGCX_TEAM_WORLD: {
      const void *net = flagcxDevNetGetFromCommS(commOpaque, contextId);
      flagcxWorldBarrierArriveS(net, coopKind, index, /*multimem=*/false, order,
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
      const void *net = flagcxDevNetGetFromCommS(commOpaque, contextId);
      flagcxInterBarrierWaitS(net, coopKind, index, order,
                              flagcxDevNetFenceLevel::Relaxed);
      break;
    }
    case FLAGCX_TEAM_WORLD: {
      const void *net = flagcxDevNetGetFromCommS(commOpaque, contextId);
      flagcxWorldBarrierWaitS(net, coopKind, index, /*multimem=*/false, order,
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
      const void *net = flagcxDevNetGetFromCommS(commOpaque, contextId);
      flagcxInterBarrierSyncS(net, coopKind, index, order,
                              flagcxDevNetFenceLevel::Relaxed);
      break;
    }
    case FLAGCX_TEAM_WORLD: {
      const void *net = flagcxDevNetGetFromCommS(commOpaque, contextId);
      flagcxWorldBarrierSyncS(net, coopKind, index, /*multimem=*/false, order,
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
  void *peerPtr =
      useP2P ? flagcxGetPeerPointer(*dst, dstOffset, team, peer) : nullptr;
  useP2P = peerPtr != nullptr;

  if (useP2P) {
    void *localSrc = flagcxGetLocalPointer(*src, srcOffset);
    if (order == flagcxDeviceMemoryOrderRelease ||
        order == flagcxDeviceMemoryOrderAcqRel)
      flagcxScopedFence(scope);
    flagcxCoopMemcpy(coopKind, peerPtr, localSrc, bytes);
  } else {
    const flagcxDevNet *net =
        (const flagcxDevNet *)flagcxDevNetGetFromCommS(commOpaque, contextId);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    net->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
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
  void *peerPtr =
      useP2P ? flagcxGetPeerPointer(*dst, dstOffset, team, peer) : nullptr;
  useP2P = peerPtr != nullptr;

  if (useP2P) {
    void *localSrc = flagcxGetLocalPointer(*src, srcOffset);
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
    const flagcxDevNet *net =
        (const flagcxDevNet *)flagcxDevNetGetFromCommS(commOpaque, contextId);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    net->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
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
  void *peerPtr =
      useP2P ? flagcxGetPeerPointer(*dst, dstOffset, team, peer) : nullptr;
  useP2P = peerPtr != nullptr;

  if (useP2P) {
    void *localSrc = flagcxGetLocalPointer(*src, srcOffset);
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
    const flagcxDevNet *net =
        (const flagcxDevNet *)flagcxDevNetGetFromCommS(commOpaque, contextId);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    net->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
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
  void *peerPtr =
      useP2P ? flagcxGetPeerPointer(*dst, dstOffset, team, peer) : nullptr;
  const flagcxDevNet *net =
      (const flagcxDevNet *)flagcxDevNetGetFromCommS(commOpaque, contextId);
  useP2P = peerPtr != nullptr && comm->_commBase.supportsDirectCounterAccess();

  if (useP2P) {
    void *localSrc = flagcxGetLocalPointer(*src, srcOffset);
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
      DeviceAPI::Atomic::fetchAdd(net->getCounterPtr(localCounter), (uint64_t)1,
                                  flagcxDeviceMemoryOrderRelease);
    }
    coop.sync();
  } else {
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    net->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
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
  void *peerPtr =
      useP2P ? flagcxGetPeerPointer(*dst, dstOffset, team, peer) : nullptr;
  const flagcxDevNet *net =
      (const flagcxDevNet *)flagcxDevNetGetFromCommS(commOpaque, contextId);
  useP2P = peerPtr != nullptr && comm->_commBase.supportsDirectCounterAccess();

  if (useP2P) {
    void *localSrc = flagcxGetLocalPointer(*src, srcOffset);
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
      DeviceAPI::Atomic::fetchAdd(net->getCounterPtr(localCounter), (uint64_t)1,
                                  flagcxDeviceMemoryOrderRelease);
    }
    coop.sync();
  } else {
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    net->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
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
  void *peerPtr =
      useP2P ? flagcxGetPeerPointer(*dst, dstOffset, team, peer) : nullptr;
  const flagcxDevNet *net =
      (const flagcxDevNet *)flagcxDevNetGetFromCommS(commOpaque, contextId);
  useP2P = peerPtr != nullptr && comm->_commBase.supportsDirectCounterAccess();

  if (useP2P) {
    void *localSrc = flagcxGetLocalPointer(*src, srcOffset);
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
      DeviceAPI::Atomic::fetchAdd(net->getCounterPtr(localCounter), (uint64_t)1,
                                  flagcxDeviceMemoryOrderRelease);
    }
    coop.sync();
  } else {
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    net->put(team, peer, *dst, dstOffset, *src, srcOffset, bytes,
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
  void *peerPtr =
      useP2P ? flagcxGetPeerPointer(*src, srcOffset, team, peer) : nullptr;
  useP2P = peerPtr != nullptr;

  if (useP2P) {
    void *localDst = flagcxGetLocalPointer(*dst, dstOffset);
    flagcxCoopMemcpy(coopKind, localDst, peerPtr, bytes);
    if (order == flagcxDeviceMemoryOrderAcquire ||
        order == flagcxDeviceMemoryOrderAcqRel)
      flagcxScopedFence(scope);
  } else {
    const flagcxDevNet *net =
        (const flagcxDevNet *)flagcxDevNetGetFromCommS(commOpaque, contextId);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    net->get(team, peer, *src, srcOffset, *dst, dstOffset, bytes, coop);
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
  void *peerPtr =
      useP2P ? flagcxGetPeerPointer(*dst, dstOffset, team, peer) : nullptr;
  useP2P = peerPtr != nullptr;

  if (useP2P) {
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    coop.sync();
    if (coop.threadRank() == 0) {
      if (order == flagcxDeviceMemoryOrderRelease ||
          order == flagcxDeviceMemoryOrderAcqRel)
        flagcxScopedFence(scope);
      *(volatile uint64_t *)peerPtr = value;
    }
    coop.sync();
  } else {
    const flagcxDevNet *net =
        (const flagcxDevNet *)flagcxDevNetGetFromCommS(commOpaque, contextId);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    net->putValue(team, peer, *dst, dstOffset, value, flagcxDevNet_None{},
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
  void *peerPtr =
      useP2P ? flagcxGetPeerPointer(*dst, dstOffset, team, peer) : nullptr;
  useP2P = peerPtr != nullptr;

  if (useP2P) {
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    coop.sync();
    if (coop.threadRank() == 0) {
      if (order == flagcxDeviceMemoryOrderRelease ||
          order == flagcxDeviceMemoryOrderAcqRel)
        flagcxScopedFence(scope);
      *(volatile uint64_t *)peerPtr = value;
      flagcxScopedFence(flagcxDeviceScopeSystem);
      flagcxDevSignalInc(commOpaque, teamKind, peer, remoteSignal, contextId,
                         FLAGCX_COOP_THREAD, flagcxDeviceScopeSystem);
    }
    coop.sync();
  } else {
    const flagcxDevNet *net =
        (const flagcxDevNet *)flagcxDevNetGetFromCommS(commOpaque, contextId);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    net->putValue(team, peer, *dst, dstOffset, value,
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
  void *peerPtr =
      useP2P ? flagcxGetPeerPointer(*dst, dstOffset, team, peer) : nullptr;
  useP2P = peerPtr != nullptr;

  if (useP2P) {
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    coop.sync();
    if (coop.threadRank() == 0) {
      if (order == flagcxDeviceMemoryOrderRelease ||
          order == flagcxDeviceMemoryOrderAcqRel)
        flagcxScopedFence(scope);
      *(volatile uint64_t *)peerPtr = value;
      flagcxScopedFence(flagcxDeviceScopeSystem);
      flagcxDevSignalAdd(commOpaque, teamKind, peer, remoteSignal, signalValue,
                         contextId, FLAGCX_COOP_THREAD,
                         flagcxDeviceScopeSystem);
    }
    coop.sync();
  } else {
    const flagcxDevNet *net =
        (const flagcxDevNet *)flagcxDevNetGetFromCommS(commOpaque, contextId);
    flagcxCoopAny coop = flagcxMakeCoopFromKind(coopKind);
    net->putValue(team, peer, *dst, dstOffset, value,
                  flagcxDevNet_SignalAdd{remoteSignal, signalValue}, coop);
  }
}

#endif // FLAGCX_DEVICE_UNIFIED_IR_IMPL_H_
