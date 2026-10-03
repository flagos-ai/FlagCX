/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_P2P_VISIBILITY_H_
#define FLAGCX_P2P_VISIBILITY_H_

#include "flagcx_device_adaptor.h"
#include "flagcx_net.h"
#include "flagcx_net_adaptor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Direct P2P READs need an explicit post-CQE visibility stage before a GPU
// destination can be reported complete. This policy helper is exposed
// internally so the fail-closed behavior can be unit-tested without IB
// hardware. completionHasFlush must stay false until the engine actually owns
// and progresses such a stage; merely advertising a provider capability is
// not sufficient.
static inline flagcxResult_t
flagcxP2pValidateReadVisibility(uint32_t requirements, uint32_t capabilities,
                                int destinationPtrType, size_t size,
                                int completionHasFlush) {
  if (size == 0 || destinationPtrType != FLAGCX_PTR_CUDA ||
      (requirements & FLAGCX_GDR_READ_REQUIRES_FLUSH) == 0)
    return flagcxSuccess;
  if ((capabilities & FLAGCX_NET_GDR_FLUSH_READ) == 0)
    return flagcxNotSupported;
  return completionHasFlush ? flagcxSuccess : flagcxNotSupported;
}

#ifdef __cplusplus
}
#endif

#endif // FLAGCX_P2P_VISIBILITY_H_
