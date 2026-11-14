/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Internal definition of flagcxShmemCommInternal for DUSHMEM (HIP).
 * Shared between dushmem_adaptor.cc and the dushmem device-API backend.
 *
 * The layout must stay identical to CommTraits<DushmemBackend>::Comm in
 * device_api/dushmem_comm_traits.h — the backend verifies it with
 * static_asserts and flagcx_device_core.h raw-casts between the two.
 ************************************************************************/

#ifndef FLAGCX_DUSHMEM_ADAPTOR_H_
#define FLAGCX_DUSHMEM_ADAPTOR_H_

#include <dushmem.h>
#include <stdint.h>

struct flagcxShmemCommInternal {
  int rank, nRanks;
  int intraRank, intraSize;
  dushmem_team_t intraTeam;
  dushmem_team_t interTeam;
  dushmem_team_t worldTeam;

  uint64_t *signalBuffer;
  int signalCount;
  uint64_t *counterBuffer;
  int counterCount;
  uint64_t *shadowBuffer;

  // Grid sync state for multi-block barrier coordination:
  // 3 barriers x (arrive[CTA_COUNT] + release[CTA_COUNT]) = 6*CTA_COUNT
  uint64_t *gridSyncState;
};

#endif // FLAGCX_DUSHMEM_ADAPTOR_H_
