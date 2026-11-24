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
 *   barrier: per (kind, index), 1 local generation + one arrival slot per
 *            source PE (the same single-writer ticket idea).  The index is
 *            the channel dimension, so concurrent indices never share a
 *            generation plane.
 ************************************************************************/

#ifndef FLAGCX_ACLSHMEM_STATE_LAYOUT_H_
#define FLAGCX_ACLSHMEM_STATE_LAYOUT_H_

// Upper bound on PEs.  Sizes every per-source slot array at compile time.
#define FLAGCX_ACLSHMEM_MAX_PES 256

// Per (context, signal).  The stride uses MAX_PES, not the runtime nRanks,
// so device indexing never needs a runtime-dependent stride.
#define FLAGCX_ACLSHMEM_SIGNAL_SLOTS (2 * FLAGCX_ACLSHMEM_MAX_PES + 2)
#define FLAGCX_ACLSHMEM_SIGNAL_RESET_BASELINE (2 * FLAGCX_ACLSHMEM_MAX_PES + 1)

// Per barrier (kind, index): [0] local generation, [1, MAX_PES] arrival
// slots.  The index names an independent channel - concurrent Coop groups
// use different indices (flagcx_device_core.h, flagcxDevBarrier) - so every
// index owns its own block instead of sharing one generation plane.  The
// plane width per kind is granted at communicator creation from the largest
// requested barrier count, with FLAGCX_ACLSHMEM_BARRIER_INDICES as the floor
// (it covers the unified IR's per-block indices, S16 runs b = 0..3); an index
// beyond the granted capacity traps in the Barrier constructor, never aliased.
#define FLAGCX_ACLSHMEM_BARRIER_SLOTS (1 + FLAGCX_ACLSHMEM_MAX_PES)
#define FLAGCX_ACLSHMEM_BARRIER_KINDS 3
#define FLAGCX_ACLSHMEM_BARRIER_INDICES 4

#endif // FLAGCX_ACLSHMEM_STATE_LAYOUT_H_
