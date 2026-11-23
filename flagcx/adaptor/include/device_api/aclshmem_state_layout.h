/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Shared host/device layout constants for ACLSHMEM DevComm-owned state.
 * Every slot below is one 32-bit word: the ordered read/write channel
 * measured on this soc is asc_dcci plus __gm__ uint32_t, so no wider word
 * is assumed.  Host allocations use the same 4-byte element size.
 *
 * Ascend launches FlagCX kernels with one block on one AI core, so unlike
 * XSHMEM there is no cross-block coordination array.  The state peers write
 * through the MTE engine and poll locally is:
 *
 *   signal:  per (context, signal), 2*MAX_PES+2 words
 *              [0, nRanks)            received tickets, one per source PE
 *              [MAX_PES, MAX_PES+n)   tickets already sent, one per dest PE
 *              [2*MAX_PES]            local-action counter
 *              [2*MAX_PES+1]          reset baseline (received snapshot)
 *            The cann-shmem signal API is int32-only; this backend keeps the
 *            proven XSHMEM ticket scheme instead and writes tickets with MTE
 *            puts, so neither 64-bit signals nor ADD-atomicity is assumed.
 *
 *   barrier: per barrier kind, 1 local generation + one arrival slot per
 *            source PE (the same single-writer ticket idea).
 ************************************************************************/

#ifndef FLAGCX_ACLSHMEM_STATE_LAYOUT_H_
#define FLAGCX_ACLSHMEM_STATE_LAYOUT_H_

// Upper bound on PEs.  Sizes every per-source slot array at compile time.
#define FLAGCX_ACLSHMEM_MAX_PES 256

// Per (context, signal).  The stride uses MAX_PES, not the runtime nRanks,
// so device indexing never needs a runtime-dependent stride.
#define FLAGCX_ACLSHMEM_SIGNAL_SLOTS (2 * FLAGCX_ACLSHMEM_MAX_PES + 2)
#define FLAGCX_ACLSHMEM_SIGNAL_RESET_BASELINE (2 * FLAGCX_ACLSHMEM_MAX_PES + 1)

// Per barrier kind: [0] local generation, [1, MAX_PES] arrival slots.
#define FLAGCX_ACLSHMEM_BARRIER_SLOTS (1 + FLAGCX_ACLSHMEM_MAX_PES)
#define FLAGCX_ACLSHMEM_BARRIER_KINDS 3

#endif // FLAGCX_ACLSHMEM_STATE_LAYOUT_H_
