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

static flagcxResult_t nvshmemAdaptorInit(int rank, int nranks, void *handle) {
  // NVSHMEM is initialized by its launcher and needs no external context.
  (void)handle;
  nvshmem_init();
  if (nvshmemx_init_status() == NVSHMEM_STATUS_NOT_INITIALIZED)
    return flagcxInternalError;
  if (nvshmem_my_pe() != rank || nvshmem_n_pes() != nranks) {
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
  int contextCount =
      (reqs->interContextCount > 0) ? reqs->interContextCount : 1;

  // Signal buffer (symmetric heap, remote-writable), partitioned by logical
  // context as [contextCount][signalCount].
  if (sc->signalCount > 0) {
    size_t signalEntries = (size_t)contextCount * (size_t)sc->signalCount;
    size_t signalBytes = signalEntries * sizeof(uint64_t);
    sc->signalBuffer = (uint64_t *)nvshmem_malloc(signalBytes);
    if (!sc->signalBuffer) {
      delete sc;
      return flagcxSystemError;
    }
    cudaMemset(sc->signalBuffer, 0, signalBytes);
  }

  // Counter buffer (local device memory), partitioned by logical context.
  if (sc->counterCount > 0) {
    size_t counterEntries = (size_t)contextCount * (size_t)sc->counterCount;
    size_t counterBytes = counterEntries * sizeof(uint64_t);
    if (cudaMalloc(&sc->counterBuffer, counterBytes) != cudaSuccess) {
      goto fail;
    }
    cudaMemset(sc->counterBuffer, 0, counterBytes);
  }

  // Shadow buffer (local device memory), partitioned by logical context.
  if (sc->signalCount > 0) {
    size_t shadowEntries = (size_t)contextCount * (size_t)sc->signalCount;
    size_t shadowBytes = shadowEntries * sizeof(uint64_t);
    if (cudaMalloc(&sc->shadowBuffer, shadowBytes) != cudaSuccess) {
      goto fail;
    }
    cudaMemset(sc->shadowBuffer, 0, shadowBytes);
  }

  // Validate topology
  {
    if (sc->intraSize > 0 && sc->nRanks % sc->intraSize != 0) {
      WARN(
          "nvshmem devCommCreate: nRanks (%d) not divisible by intraSize (%d); "
          "non-uniform topologies are not supported",
          sc->nRanks, sc->intraSize);
      goto fail;
    }
    int interSize = (sc->intraSize > 0) ? sc->nRanks / sc->intraSize : 1;

    // Grid sync state for multi-block barrier coordination
    // 3 barriers x (arrive[CTA_COUNT] + release[CTA_COUNT]) = 6*CTA_COUNT
    size_t gridSyncSize = 6 * FLAGCX_DEVICE_CTA_COUNT * sizeof(uint64_t);
    if (cudaMalloc(&sc->gridSyncState, gridSyncSize) != cudaSuccess) {
      goto fail;
    }
    cudaMemset(sc->gridSyncState, 0, gridSyncSize);

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

  // Free local device allocations
  if (shmemComm->counterBuffer)
    cudaFree(shmemComm->counterBuffer);
  if (shmemComm->shadowBuffer)
    cudaFree(shmemComm->shadowBuffer);
  if (shmemComm->gridSyncState)
    cudaFree(shmemComm->gridSyncState);

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
