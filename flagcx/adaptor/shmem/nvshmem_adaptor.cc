/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * NVSHMEM Adaptor — implementation of flagcxShmemAdaptor_t for NVSHMEM.
 * Manages NVSHMEM lifecycle, symmetric heap allocations, and device comm
 * state (signals, counters, barriers, teams).
 ************************************************************************/

#include "nvshmem_adaptor.h"
#include "shmem_adaptor.h"

#include "flagcx_kernel_internal.h"
#include "global_comm.h"

#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

// ============================================================
// Internal state for one devComm backed by NVSHMEM
// ============================================================

// ============================================================
// Lifecycle: reference-counted init/finalize
// ============================================================
static int shmemInitRefCount = 0;

static flagcxResult_t nvshmemAdaptorInit(int rank, int nRanks) {
  nvshmem_init();
  if (nvshmemx_init_status() == NVSHMEM_STATUS_NOT_INITIALIZED)
    return flagcxInternalError;
  if (nvshmem_my_pe() != rank || nvshmem_n_pes() != nRanks) {
    nvshmem_finalize();
    return flagcxInternalError;
  }
  shmemInitRefCount++;
  return flagcxSuccess;
}

static flagcxResult_t nvshmemAdaptorFinalize() {
  if (shmemInitRefCount > 0 && --shmemInitRefCount == 0) {
    nvshmem_finalize();
  }
  return flagcxSuccess;
}

// ============================================================
// Symmetric memory management
// ============================================================
static flagcxResult_t nvshmemAdaptorMalloc(void **ptr, size_t size) {
  *ptr = nvshmem_malloc(size);
  if (*ptr == nullptr)
    return flagcxSystemError;
  cudaMemset(*ptr, 0, size);
  return flagcxSuccess;
}

static flagcxResult_t nvshmemAdaptorFree(void *ptr) {
  nvshmem_free(ptr);
  return flagcxSuccess;
}

// ============================================================
// Device Comm Create
// ============================================================
static flagcxResult_t nvshmemAdaptorDevCommDestroy(flagcxShmemComm_t shmemComm);

static flagcxResult_t
nvshmemAdaptorDevCommCreate(flagcxComm_t comm,
                            const struct flagcxDevCommRequirements *reqs,
                            flagcxShmemComm_t *shmemComm) {
  auto *sc = new flagcxShmemCommInternal();
  memset(sc, 0, sizeof(*sc));
  sc->intraTeam = NVSHMEM_TEAM_INVALID;
  sc->interTeam = NVSHMEM_TEAM_INVALID;

  sc->rank = comm->rank;
  sc->nRanks = comm->nranks;
  sc->intraRank = comm->localRank;
  sc->intraSize = comm->localRanks;

  sc->signalCount = reqs->interSignalCount;
  sc->counterCount = reqs->interCounterCount;
  sc->intraBarrierCount = reqs->intraBarrierCount;
  sc->interBarrierCount = reqs->interBarrierCount;
  sc->worldBarrierCount = reqs->barrierCount;

  // Signal buffer (symmetric heap, remote-writable)
  if (sc->signalCount > 0) {
    sc->signalBuffer =
        (uint64_t *)nvshmem_malloc(sc->signalCount * sizeof(uint64_t));
    if (!sc->signalBuffer) {
      delete sc;
      return flagcxSystemError;
    }
    cudaMemset(sc->signalBuffer, 0, sc->signalCount * sizeof(uint64_t));
  }

  // Counter buffer (local device memory)
  if (sc->counterCount > 0) {
    if (cudaMalloc(&sc->counterBuffer, sc->counterCount * sizeof(uint64_t)) !=
        cudaSuccess) {
      goto fail;
    }
    cudaMemset(sc->counterBuffer, 0, sc->counterCount * sizeof(uint64_t));
  }

  // Shadow buffer (local device memory)
  if (sc->signalCount > 0) {
    if (cudaMalloc(&sc->shadowBuffer, sc->signalCount * sizeof(uint64_t)) !=
        cudaSuccess) {
      goto fail;
    }
    cudaMemset(sc->shadowBuffer, 0, sc->signalCount * sizeof(uint64_t));
  }

  // Intra-node barrier signals (symmetric)
  if (sc->intraBarrierCount > 0 && sc->intraSize > 0) {
    size_t sz =
        (size_t)sc->intraBarrierCount * sc->intraSize * sizeof(uint64_t);
    sc->intraBarrierSignals = (uint64_t *)nvshmem_malloc(sz);
    if (!sc->intraBarrierSignals) {
      goto fail;
    }
    cudaMemset(sc->intraBarrierSignals, 0, sz);
  }

  // Inter-node barrier signals (symmetric)
  {
    if (sc->intraSize > 0 && sc->nRanks % sc->intraSize != 0) {
      WARN(
          "nvshmem devCommCreate: nRanks (%d) not divisible by intraSize (%d); "
          "non-uniform topologies are not supported",
          sc->nRanks, sc->intraSize);
      goto fail;
    }
    int interSize = (sc->intraSize > 0) ? sc->nRanks / sc->intraSize : 1;
    if (sc->interBarrierCount > 0 && interSize > 1) {
      size_t sz = (size_t)sc->interBarrierCount * interSize * sizeof(uint64_t);
      sc->interBarrierSignals = (uint64_t *)nvshmem_malloc(sz);
      if (!sc->interBarrierSignals) {
        goto fail;
      }
      cudaMemset(sc->interBarrierSignals, 0, sz);
    }

    // World barrier signals (symmetric)
    if (sc->worldBarrierCount > 0) {
      size_t sz = (size_t)sc->worldBarrierCount * sc->nRanks * sizeof(uint64_t);
      sc->worldBarrierSignals = (uint64_t *)nvshmem_malloc(sz);
      if (!sc->worldBarrierSignals) {
        goto fail;
      }
      cudaMemset(sc->worldBarrierSignals, 0, sz);
    }

    // Barrier usage counters (local)
    int totalBarriers =
        sc->intraBarrierCount + sc->interBarrierCount + sc->worldBarrierCount;
    if (totalBarriers > 0) {
      if (cudaMalloc(&sc->barrierUsage, totalBarriers * sizeof(uint64_t)) !=
          cudaSuccess) {
        goto fail;
      }
      cudaMemset(sc->barrierUsage, 0, totalBarriers * sizeof(uint64_t));
    }

    // Team creation: intra-node
    // NOTE: assumes uniform intra-node GPU count across all nodes.
    // Heterogeneous topologies are not supported by this strided team-split.
    int nodeId = (sc->intraSize > 0) ? sc->rank / sc->intraSize : 0;
    if (nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, nodeId * sc->intraSize,
                                   1, sc->intraSize, NULL, 0,
                                   &sc->intraTeam) != 0) {
      goto fail;
    }

    // Team creation: inter-node
    if (interSize > 1) {
      if (nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, sc->intraRank,
                                     sc->intraSize, interSize, NULL, 0,
                                     &sc->interTeam) != 0) {
        goto fail;
      }
    } else {
      sc->interTeam = NVSHMEM_TEAM_INVALID;
    }

    sc->worldTeam = NVSHMEM_TEAM_WORLD;
  }

  *shmemComm = sc;
  return flagcxSuccess;

fail:
  nvshmemAdaptorDevCommDestroy(sc);
  return flagcxSystemError;
}

// ============================================================
// Device Comm Destroy
// ============================================================
static flagcxResult_t
nvshmemAdaptorDevCommDestroy(flagcxShmemComm_t shmemComm) {
  if (shmemComm == nullptr)
    return flagcxSuccess;

  // Free symmetric heap allocations
  if (shmemComm->signalBuffer)
    nvshmem_free(shmemComm->signalBuffer);
  if (shmemComm->intraBarrierSignals)
    nvshmem_free(shmemComm->intraBarrierSignals);
  if (shmemComm->interBarrierSignals)
    nvshmem_free(shmemComm->interBarrierSignals);
  if (shmemComm->worldBarrierSignals)
    nvshmem_free(shmemComm->worldBarrierSignals);

  // Free local device allocations
  if (shmemComm->counterBuffer)
    cudaFree(shmemComm->counterBuffer);
  if (shmemComm->shadowBuffer)
    cudaFree(shmemComm->shadowBuffer);
  if (shmemComm->barrierUsage)
    cudaFree(shmemComm->barrierUsage);

  // Destroy teams
  if (shmemComm->intraTeam != NVSHMEM_TEAM_INVALID)
    nvshmem_team_destroy(shmemComm->intraTeam);
  if (shmemComm->interTeam != NVSHMEM_TEAM_INVALID)
    nvshmem_team_destroy(shmemComm->interTeam);

  delete shmemComm;
  return flagcxSuccess;
}

// ============================================================
// Global adaptor instance
// ============================================================
static flagcxShmemAdaptor_t nvshmemAdaptorInstance = {
    .name = "nvshmem",
    .init = nvshmemAdaptorInit,
    .finalize = nvshmemAdaptorFinalize,
    .malloc = nvshmemAdaptorMalloc,
    .free = nvshmemAdaptorFree,
    .devCommCreate = nvshmemAdaptorDevCommCreate,
    .devCommDestroy = nvshmemAdaptorDevCommDestroy,
};

flagcxShmemAdaptor_t *shmemAdaptor = &nvshmemAdaptorInstance;
