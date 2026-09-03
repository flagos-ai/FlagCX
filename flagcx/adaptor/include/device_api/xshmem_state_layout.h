/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Shared host/device layout constants for XSHMEM DevComm-owned state.
 ************************************************************************/

#ifndef FLAGCX_XSHMEM_STATE_LAYOUT_H_
#define FLAGCX_XSHMEM_STATE_LAYOUT_H_

// P800 has no C2C atomic: xshmemi_signal_op only implements XSHMEM_SIGNAL_SET
// and traps on SIGNAL_ADD. A remote increment is therefore emulated the way
// xshmem's own barrier does it — every source owns a private slot in the
// destination and SETs it to a monotonically increasing ticket, and the
// receiver aggregates by summing the slots.
//
// Each (context, signal) owns FLAGCX_XSHMEM_SIGNAL_SLOTS(nRanks) words:
//   [0, nRanks)         received tickets, one per source PE (written remotely)
//   [nRanks, 2*nRanks)  tickets already sent, one per destination PE (local)
//   [2*nRanks]          local-action counter (never written remotely)
// Counters use the identical layout.
#define FLAGCX_XSHMEM_SIGNAL_SLOTS(nRanks) (2 * (nRanks) + 1)

// Upper bound on PEs, used only to size the barrier state at compile time so
// the device pass can fold the per-barrier-kind stride into a constant.
#define FLAGCX_XSHMEM_MAX_PES 256

// Per barrier kind:
//   local arrivals [CTA_COUNT]
//   arrive releases [CTA_COUNT]
//   remote arrival tickets [MAX_PES]  (one slot per source PE, SET remotely)
//   wait releases [CTA_COUNT]
#define FLAGCX_XSHMEM_BARRIER_STATE_STRIDE                                     \
  (3 * FLAGCX_DEVICE_CTA_COUNT + FLAGCX_XSHMEM_MAX_PES)
#define FLAGCX_XSHMEM_BARRIER_STATE_WORDS                                      \
  (3 * FLAGCX_XSHMEM_BARRIER_STATE_STRIDE)

#endif // FLAGCX_XSHMEM_STATE_LAYOUT_H_
