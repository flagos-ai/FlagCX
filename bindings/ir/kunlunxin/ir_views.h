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

#endif // FLAGCX_KUNLUNXIN_IR_VIEWS_H_
