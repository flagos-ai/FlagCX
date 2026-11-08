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

// flagcxMakeTeamFromKind, which lives in flagcx_device_scalar_ir_impl.h and is
// not visible in this branch.
static FLAGCX_DEVICE_INLINE_DECORATOR flagcxTeam
klxTeamFromKind(const flagcxDevComm &comm, flagcxDevTeamKind_t kind) {
  switch (kind) {
    case FLAGCX_TEAM_WORLD:
      return flagcxTeamWorld(comm);
    case FLAGCX_TEAM_INTER:
      return flagcxTeamInter(comm);
    case FLAGCX_TEAM_INTRA:
    default:
      return flagcxTeamIntra(comm);
  }
}

// Stage the comm into local memory so the generic flagcxDevNet works here: XTDK
// cannot construct it from the address-space-1 descriptor. Raw storage, because
// GM2LM overwrites all of it and a default-constructed comm zeroes it first.
#define KLX_NET_PROLOGUE(opaque, contextId, coopKind)                          \
  alignas(flagcxDevComm) char klxStagedStorage[sizeof(flagcxDevComm)];         \
  GM2LM((opaque), klxStagedStorage, sizeof(flagcxDevComm));                    \
  const flagcxDevComm &klxStagedComm =                                         \
      *reinterpret_cast<const flagcxDevComm *>(klxStagedStorage);              \
  flagcxDevNet net(klxStagedComm, (int)(contextId));                           \
  flagcxCoopAny coop = klxMakeCoopFromKind(coopKind)

// Signal / counter entry points. Defined here and not in the shared
// flagcx_device_wrapper_impl.h, which includes this header; only the symbols
// listed in kunlunxin/Makefile leave the bitcode.
FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevSignalInc(const void FLAGCX_IR_GLOBAL_PTR *opaque,
                   flagcxDevTeamKind_t teamKind, int peer,
                   flagcxDevSignal_t signal, flagcxDevContext_t contextId,
                   flagcxDevCoopKind_t coopKind, flagcxDevMemoryScope_t scope) {
  (void)scope;
  KLX_NET_PROLOGUE(opaque, contextId, coopKind);
  net.signal(
      klxTeamFromKind(klxStagedComm, teamKind), peer,
      flagcxDevNet_SignalInc{signal}, coop);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevSignalAdd(const void FLAGCX_IR_GLOBAL_PTR *opaque,
                   flagcxDevTeamKind_t teamKind, int peer,
                   flagcxDevSignal_t signal, uint64_t value,
                   flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
                   flagcxDevMemoryScope_t scope) {
  (void)scope;
  KLX_NET_PROLOGUE(opaque, contextId, coopKind);
  net.signal(
      klxTeamFromKind(klxStagedComm, teamKind), peer,
      flagcxDevNet_SignalAdd{signal, value}, coop);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevWaitSignal(const void FLAGCX_IR_GLOBAL_PTR *opaque,
                    flagcxDevSignal_t signal, uint64_t least, int bits,
                    flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
                    flagcxDevMemoryOrder_t order) {
  KLX_NET_PROLOGUE(opaque, contextId, coopKind);
  net.waitSignal(coop, signal, least, bits, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevWaitSignalMeetShadow(const void FLAGCX_IR_GLOBAL_PTR *opaque,
                              flagcxDevContext_t contextId,
                              flagcxDevSignal_t slot, int bits,
                              flagcxDevCoopKind_t coopKind,
                              flagcxDevMemoryOrder_t order) {
  KLX_NET_PROLOGUE(opaque, contextId, coopKind);
  net.waitSignalMeetShadow(coop, slot, bits, order);
}

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevWaitCounter(const void FLAGCX_IR_GLOBAL_PTR *opaque,
                     flagcxDevCounter_t counter, uint64_t least, int bits,
                     flagcxDevContext_t contextId, flagcxDevCoopKind_t coopKind,
                     flagcxDevMemoryOrder_t order) {
  KLX_NET_PROLOGUE(opaque, contextId, coopKind);
  net.waitCounter(coop, counter, least, bits, order);
}

// One bind per kernel, per the vendor contract. XTDK's __local__ is only an
// alignment attribute, so the LM scratch must be an alloca in the kernel's own
// frame and the caller passes it in; only the shared halves can be statics.
static __shared__ __simd__ xshmemi_device_host_state_t klxXshmemState;
static __shared__ __simd__ char klxXshmemFcWindow[XSHMEMI_FC_BUFFER_LEN];

FLAGCX_IR_EXTERN_C FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxDevInit(const void FLAGCX_IR_GLOBAL_PTR *opaque, char *localBuf) {
  static_assert(XSHMEMI_LOCAL_BUF_LEN == 4096,
                "device_init lowering allocates XSHMEMI_LOCAL_BUF_LEN bytes");
  KlxCommPtr comm = static_cast<KlxCommPtr>(opaque);
  xshmem_build_xshmemi_device_state(comm->_commBase.devStateHandle, localBuf,
                                    &klxXshmemState, klxXshmemFcWindow);
  XSHMEM_DEVICE_SET_FLOW_CONTROL_THRESHOLD(cluster_num());
}

#endif // FLAGCX_KUNLUNXIN_IR_VIEWS_H_
