#ifndef FLAGCX_KUNLUNXIN_IR_VIEWS_H_
#define FLAGCX_KUNLUNXIN_IR_VIEWS_H_

#include "../flagcx_device_wrapper.h"

using KlxCommPtr = const flagcxDevComm FLAGCX_IR_GLOBAL_PTR *;
using KlxMemPtr = const flagcxDevMem FLAGCX_IR_GLOBAL_PTR *;

static FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopAny
klxMakeCoopFromKind(flagcxCoopKind_t kind) {
  switch (kind) {
    case FLAGCX_COOP_BLOCK:
      return flagcxCoopAny(flagcxCoopBlock());
    case FLAGCX_COOP_WARP:
      return flagcxCoopAny(flagcxCoopWarp());
    case FLAGCX_COOP_THREAD:
    default:
      return flagcxCoopAny(flagcxCoopThread());
  }
}

struct KlxCommView {
  KlxCommPtr p;

  FLAGCX_DEVICE_INLINE_DECORATOR int intraRank() const {
    return p->_commBase.intraRank;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR int intraSize() const {
    return p->_commBase.intraSize;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void arrive(flagcxCoopAny &coop) const {
    if (p->_commBase.gridSyncState && p->_commBase.intraSize > 0) {
      xshmemGridArrive(coop, p->_commBase.gridSyncState, 0,
                       p->_commBase.intraSize, 1, p->_commBase.intraRank,
                       p->_commBase.intraPeMap);
    }
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void wait(flagcxCoopAny &coop) const {
    if (p->_commBase.gridSyncState && p->_commBase.intraSize > 0) {
      xshmemGridWait(coop, p->_commBase.gridSyncState, p->_commBase.intraSize);
    }
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void sync(flagcxCoopAny &coop) const {
    arrive(coop);
    wait(coop);
  }
};

struct KlxMemView {
  KlxMemPtr p;

  FLAGCX_DEVICE_INLINE_DECORATOR int worldPeer(int peer) const {
    return p->_winBase.intraPeMap[peer];
  }

  FLAGCX_DEVICE_INLINE_DECORATOR XSHMEM_FGP char *
  symmetricAddress(FLAGCX_IR_OFFSET_TYPE offset) const {
    return (XSHMEM_FGP char *)p->_winBase.symBase + offset;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR FLAGCX_IR_GLOBAL_RETURN_PTR void *
  intraPointer(FLAGCX_IR_OFFSET_TYPE offset, int peer) const {
    return flagcxXshmemDevice::peerPtr(symmetricAddress(offset),
                                       worldPeer(peer));
  }
};

FLAGCX_IR_EXTERN_C
FLAGCX_DEVICE_INLINE_DECORATOR FLAGCX_IR_GLOBAL_RETURN_PTR void *
flagcxGetIntraPointerWithAccessC(const void FLAGCX_IR_GLOBAL_PTR *opaque,
                                 FLAGCX_IR_OFFSET_TYPE offset, int peer,
                                 flagcxDevPeerAccess_t access) {
  if (access != flagcxDevPeerAccessWriteOnly)
    return nullptr;
  KlxMemView mem{static_cast<KlxMemPtr>(opaque)};
  return flagcxXshmemDevice::peerWritePtr(mem.symmetricAddress(offset),
                                         mem.worldPeer(peer));
}

// flagcxMakeTeamFromKind, which lives in flagcx_device_scalar_ir_impl.h and is
// not visible in this branch. Reads the address-space-1 comm directly; the
// arithmetic is the one in flagcx_device_core.h's team accessors.
static FLAGCX_DEVICE_INLINE_DECORATOR flagcxTeam
klxTeamFromKind(KlxCommPtr p, flagcxDevTeamKind_t kind) {
  int rank = p->_commBase.rank;
  int nRanks = p->_commBase.nRanks;
  int intraSize = p->_commBase.intraSize;
  switch (kind) {
    case FLAGCX_TEAM_WORLD:
      return flagcxTeam(nRanks, rank, 1);
    case FLAGCX_TEAM_INTER:
      return flagcxTeam(nRanks / intraSize, rank / intraSize, intraSize);
    case FLAGCX_TEAM_INTRA:
    default:
      return flagcxTeam(intraSize, p->_commBase.intraRank, 1);
  }
}

// The signal / counter half of DeviceAPI::Net, driven straight from the
// address-space-1 comm so nothing has to be staged in local memory. One for one
// with the private members of CommTraits<XshmemBackend>::Net in
// xshmem_comm_traits.h: that slot layout is the ABI, keep both sides in step.
struct KlxNetView {
  KlxCommPtr p;
  int contextId;

  FLAGCX_DEVICE_INLINE_DECORATOR
  KlxNetView(const void FLAGCX_IR_GLOBAL_PTR *opaque, int context)
      : p(static_cast<KlxCommPtr>(opaque)),
        contextId(p->_contextCount > 0 ? context % p->_contextCount : 0) {}

  FLAGCX_DEVICE_INLINE_DECORATOR int nRanks() const {
    return p->_commBase.nRanks;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR XSHMEM_FGP uint64_t *
  signalSlot(flagcxDevSignal_t signal, int index) const {
    int base = (contextId * p->_commBase.signalCount + (int)signal) *
               FLAGCX_XSHMEM_SIGNAL_SLOTS(nRanks());
    return &p->_commBase.signalBuffer[base + index];
  }

  FLAGCX_DEVICE_INLINE_DECORATOR XSHMEM_FGP uint64_t *
  counterSlot(flagcxDevCounter_t counter, int index) const {
    int base = (contextId * p->_commBase.counterCount + (int)counter) *
               FLAGCX_XSHMEM_SIGNAL_SLOTS(nRanks());
    return &p->_commBase.counterBuffer[base + index];
  }

  // Logical value: every source's ticket, less what a local reset wrote off,
  // plus this PE's own slot.
  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
  sumSignal(flagcxDevSignal_t signal) const {
    int n = nRanks();
    uint64_t total = 0;
    for (int src = 0; src < n; ++src)
      total += *(XshmemGMWord *)signalSlot(signal, src);
    total -=
        *(XshmemGMWord *)signalSlot(signal,
                                    FLAGCX_XSHMEM_RESET_BASELINE_INDEX(n));
    return total + *(XshmemGMWord *)signalSlot(signal, 2 * n);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
  sumCounter(flagcxDevCounter_t counter) const {
    int n = nRanks();
    uint64_t total = 0;
    for (int src = 0; src < n; ++src)
      total += *(XshmemGMWord *)counterSlot(counter, src);
    total -=
        *(XshmemGMWord *)counterSlot(counter,
                                     FLAGCX_XSHMEM_RESET_BASELINE_INDEX(n));
    return total + *(XshmemGMWord *)counterSlot(counter, 2 * n);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR int resolvePE(flagcxTeam team,
                                               int peer) const {
    XSHMEM_FGP int *map = p->_commBase.intraPeMap;
    if (map != nullptr && team.nRanks() == p->_commBase.intraSize &&
        team.rank() == p->_commBase.intraRank && team.stride() == 1)
      return map[peer];
    int base = p->_commBase.rank - team.rank() * team.stride();
    return base + peer * team.stride();
  }

  // SET is the only signal op P800 has, so publish the running total this PE
  // owes `pe`. That ticket slot has a single writer, hence the plain
  // read-add-write.
  FLAGCX_DEVICE_INLINE_DECORATOR void
  sendTicket(XSHMEM_FGP uint64_t *sentSlot, XSHMEM_FGP uint64_t *ownSlot,
             uint64_t delta, int pe) const {
    XshmemGMWord *sent = (XshmemGMWord *)sentSlot;
    uint64_t ticket = *sent + delta;
    *sent = ticket;
    flagcxXshmemDevice::signalOp(ownSlot, ticket, XSHMEM_SIGNAL_SET, pe);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void signal(flagcxCoopAny &coop,
                                             flagcxTeam team, int peer,
                                             flagcxDevSignal_t slot,
                                             uint64_t delta) const {
    coop.sync();
    if (coop.threadRank() == 0) {
      int pe = resolvePE(team, peer);
      sendTicket(signalSlot(slot, nRanks() + pe),
                 signalSlot(slot, p->_commBase.rank), delta, pe);
    }
    coop.sync();
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void spinSignal(flagcxDevSignal_t slot,
                                                 uint64_t least) const {
    int iter = 0;
    while (sumSignal(slot) < least)
      DeviceAPI::Intrin::spinBackoff(iter++);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  waitSignal(flagcxCoopAny &coop, flagcxDevSignal_t slot,
             uint64_t least) const {
    coop.sync();
    if (coop.threadRank() == 0)
      spinSignal(slot, least);
    coop.sync();
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  waitSignalMeetShadow(flagcxCoopAny &coop, flagcxDevSignal_t slot) const {
    coop.sync();
    if (coop.threadRank() == 0) {
      XSHMEM_FGP uint64_t *shadow =
          &p->_commBase.shadowBuffer[contextId * p->_commBase.signalCount +
                                     (int)slot];
      spinSignal(slot, *(XshmemGMWord *)shadow);
    }
    coop.sync();
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  waitCounter(flagcxCoopAny &coop, flagcxDevCounter_t counter,
              uint64_t least) const {
    coop.sync();
    if (coop.threadRank() == 0) {
      int iter = 0;
      while (sumCounter(counter) < least)
        DeviceAPI::Intrin::spinBackoff(iter++);
    }
    coop.sync();
  }
};

// Signal / counter entry points. Defined here and not in the shared
// flagcx_device_wrapper_impl.h, which includes this header; only the symbols
// listed in kunlunxin/Makefile leave the bitcode. bits and order are unused
// because the generic xshmem Net discards them too.
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevSignalInc(const void FLAGCX_IR_GLOBAL_PTR *opaque,
                   flagcxDevTeamKind_t teamKind, int peer,
                   flagcxDevSignal_t signal, flagcxDevContext_t contextId,
                   flagcxDevCoopKind_t coopKind, flagcxDevMemoryScope_t scope) {
  (void)scope;
  KlxNetView net(opaque, (int)contextId);
  flagcxCoopAny coop = klxMakeCoopFromKind(coopKind);
  net.signal(coop, klxTeamFromKind(net.p, teamKind), peer, signal,
             (uint64_t)1);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevSignalAdd(const void FLAGCX_IR_GLOBAL_PTR *opaque,
                   flagcxDevTeamKind_t teamKind, int peer,
                   flagcxDevSignal_t signal, uint64_t value,
                   flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
                   flagcxDevMemoryScope_t scope) {
  (void)scope;
  KlxNetView net(opaque, (int)contextId);
  flagcxCoopAny coop = klxMakeCoopFromKind(coopKind);
  net.signal(coop, klxTeamFromKind(net.p, teamKind), peer, signal, value);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevWaitSignal(const void FLAGCX_IR_GLOBAL_PTR *opaque,
                    flagcxDevSignal_t signal, uint64_t least, int bits,
                    flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
                    flagcxDevMemoryOrder_t order) {
  (void)bits;
  (void)order;
  KlxNetView net(opaque, (int)contextId);
  flagcxCoopAny coop = klxMakeCoopFromKind(coopKind);
  net.waitSignal(coop, signal, least);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevWaitSignalMeetShadow(const void FLAGCX_IR_GLOBAL_PTR *opaque,
                              flagcxDevContext_t contextId,
                              flagcxDevSignal_t slot, int bits,
                              flagcxDevCoopKind_t coopKind,
                              flagcxDevMemoryOrder_t order) {
  (void)bits;
  (void)order;
  KlxNetView net(opaque, (int)contextId);
  flagcxCoopAny coop = klxMakeCoopFromKind(coopKind);
  net.waitSignalMeetShadow(coop, slot);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevWaitCounter(const void FLAGCX_IR_GLOBAL_PTR *opaque,
                     flagcxDevCounter_t counter, uint64_t least, int bits,
                     flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
                     flagcxDevMemoryOrder_t order) {
  (void)bits;
  (void)order;
  KlxNetView net(opaque, (int)contextId);
  flagcxCoopAny coop = klxMakeCoopFromKind(coopKind);
  net.waitCounter(coop, counter, least);
}

// One bind per kernel, per the vendor contract. XTDK's __local__ is only an
// alignment attribute, so the LM scratch must be an alloca in the kernel's own
// frame and the caller passes it in; only the shared halves can be statics.
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevInit(const void FLAGCX_IR_GLOBAL_PTR *opaque, char *localBuf) {
  static __shared__ __simd__ xshmemi_device_host_state_t klxXshmemState;
  static __shared__ __simd__ char klxXshmemFcWindow[XSHMEMI_FC_BUFFER_LEN];

  static_assert(XSHMEMI_LOCAL_BUF_LEN == 4096,
                "device_init lowering allocates XSHMEMI_LOCAL_BUF_LEN bytes");
  KlxCommPtr comm = static_cast<KlxCommPtr>(opaque);
  xshmem_build_xshmemi_device_state(comm->_commBase.devStateHandle, localBuf,
                                    &klxXshmemState, klxXshmemFcWindow);
  XSHMEM_DEVICE_SET_FLOW_CONTROL_THRESHOLD(cluster_num());
}

#endif // FLAGCX_KUNLUNXIN_IR_VIEWS_H_
