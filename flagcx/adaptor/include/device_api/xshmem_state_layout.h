/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Shared host/device layout constants for XSHMEM DevComm-owned state.
 ************************************************************************/

#ifndef FLAGCX_XSHMEM_STATE_LAYOUT_H_
#define FLAGCX_XSHMEM_STATE_LAYOUT_H_

// Per barrier kind:
//   local arrivals [CTA_COUNT]
//   arrive releases [CTA_COUNT]
//   symmetric remote arrival counter [1]
//   wait releases [CTA_COUNT]
#define FLAGCX_XSHMEM_BARRIER_STATE_STRIDE (3 * FLAGCX_DEVICE_CTA_COUNT + 1)
#define FLAGCX_XSHMEM_BARRIER_STATE_WORDS                                      \
  (3 * FLAGCX_XSHMEM_BARRIER_STATE_STRIDE)

#endif // FLAGCX_XSHMEM_STATE_LAYOUT_H_
