/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Internal definition of flagcxShmemCommInternal for cann-shmem (aclshmem).
 * Shared between aclshmem_adaptor.cc and aclshmem_dev_api_backend.cc.
 ************************************************************************/

#ifndef FLAGCX_ACLSHMEM_ADAPTOR_H_
#define FLAGCX_ACLSHMEM_ADAPTOR_H_

#include <stdint.h>

struct flagcxShmemCommInternal {
  int rank, nRanks;
  int intraRank, intraSize;

  uint64_t *signalBuffer; // symmetric heap; per-source ticket slots
  int signalCount;
  uint64_t *counterBuffer; // local device memory
  int counterCount;
  uint64_t *shadowBuffer; // local device memory, one word per signal

  uint64_t *gridSyncState; // symmetric heap; 3 barrier kinds x slots
  uint64_t *scratchBuffer; // symmetric heap; putValue staging, 8B per ctx
  int barrierCapacity;     // granted (kind, index) channels per barrier kind
};

#endif // FLAGCX_ACLSHMEM_ADAPTOR_H_
