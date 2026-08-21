/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Internal definition of flagcxShmemCommInternal for NVSHMEM.
 * Shared between nvshmem_adaptor.cc and nvshmem_dev_api_backend.cc.
 ************************************************************************/

#ifndef FLAGCX_NVSHMEM_ADAPTOR_H_
#define FLAGCX_NVSHMEM_ADAPTOR_H_

#include <nvshmem.h>
#include <stdint.h>

struct flagcxShmemCommInternal {
  int rank, nRanks;
  int intraRank, intraSize;
  nvshmem_team_t intraTeam;
  nvshmem_team_t interTeam;
  nvshmem_team_t worldTeam;

  uint64_t *signalBuffer;
  int signalCount;
  uint64_t *counterBuffer;
  int counterCount;
  uint64_t *shadowBuffer;

  uint64_t
      *gridSyncState; // 3 barriers x (arrive[CTA_COUNT] + release[CTA_COUNT])
};

#endif // FLAGCX_NVSHMEM_ADAPTOR_H_
