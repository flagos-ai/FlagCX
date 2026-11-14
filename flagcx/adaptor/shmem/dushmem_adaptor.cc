/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * DUSHMEM Adaptor — implementation of flagcxShmemAdaptor_t for DUSHMEM
 * (Sugon/HIP SHMEM runtime, DTK). Manages DUSHMEM lifecycle, symmetric
 * heap allocations, and device comm state (signals, counters, barriers,
 * teams).
 *
 * Initialization modes:
 *   - handle == nullptr: dushmem_init() with the environment-selected
 *     bootstrap (PMI/PMIx under an MPI launcher, or DUSHMEM_BOOTSTRAP=UID).
 *   - handle != nullptr: `handle` points to a dushmemx_uniqueid_t already
 *     exchanged out-of-band by the caller (e.g. through the FlagCX socket
 *     bootstrap); init uses DUSHMEMX_INIT_WITH_UNIQUEID.
 ************************************************************************/

#include "dushmem_adaptor.h"
#include "shmem_adaptor.h"

#include "flagcx_kernel_internal.h"
#include "global_comm.h"

#include <cstdio>
#include <cstring>
#include <hip/hip_runtime.h>
#include <dushmem.h>
#include <dushmemx.h>

// ============================================================
// Lifecycle: reference-counted init/finalize
// ============================================================
static int shmemInitRefCount = 0;

static flagcxResult_t dushmemAdaptorInit(int rank, int nranks, void *handle) {
  if (shmemInitRefCount > 0) {
    // DUSHMEM has no re-init: additional devComms join the existing world.
    if (dushmem_my_pe() != rank || dushmem_n_pes() != nranks) {
      WARN("dushmem init: already initialized with a different world "
           "(caller %d/%d, dushmem %d/%d)",
           rank, nranks, dushmem_my_pe(), dushmem_n_pes());
      return flagcxInvalidUsage;
    }
    shmemInitRefCount++;
    return flagcxSuccess;
  }
  if (handle != nullptr) {
    // UID mode: the caller borrowed a dushmemx_uniqueid_t obtained from the
    // bootstrap PE (usually rank 0 via dushmemx_get_uniqueid) and exchanged
    // it through FlagCX's own bootstrap.
    dushmemx_init_attr_t attr = DUSHMEMX_INIT_ATTR_INITIALIZER;
    dushmemx_set_attr_uniqueid_args(rank, nranks, (dushmemx_uniqueid_t *)handle,
                                    &attr);
    if (dushmemx_init_attr(DUSHMEMX_INIT_WITH_UNIQUEID, &attr) != 0)
      return flagcxInternalError;
  } else {
    // Default mode: let DUSHMEM pick the bootstrap from the environment
    // (DUSHMEM_BOOTSTRAP / DUSHMEM_BOOTSTRAP_PMI). Works under mpirun with
    // PMI, and with the UID bootstrap when so configured.
    dushmem_init();
  }

  // dushmemx_init_status() returns DUSHMEM_STATUS_NOT_INITIALIZED (0) before
  // init completes; any other value means the runtime is (at least partially)
  // initialized.
  if (dushmemx_init_status() == DUSHMEM_STATUS_NOT_INITIALIZED)
    return flagcxInternalError;
  // DUSHMEM bootstraps in two stages: init returns with IS_BOOTSTRAPPED and
  // the runtime deliberately defers common_init (get_cucontext reports
  // GPU_NOT_SELECTED on the first pass), completing it lazily inside the
  // next collective call. team_split does not trigger that lazy path and
  // would fail with "called before DUSHMEM initialization has completed",
  // so finish the second stage eagerly with the same barrier the DUSHMEM
  // init-status unit test uses.
  if (dushmemx_init_status() < DUSHMEM_STATUS_IS_INITIALIZED) {
    dushmem_barrier_all();
  }
  if (dushmemx_init_status() < DUSHMEM_STATUS_IS_INITIALIZED) {
    WARN("dushmem init: runtime stuck at status %d (expected >= %d)",
         dushmemx_init_status(), DUSHMEM_STATUS_IS_INITIALIZED);
    dushmem_finalize();
    return flagcxInternalError;
  }
  if (dushmem_my_pe() != rank || dushmem_n_pes() != nranks) {
    WARN("dushmem init: caller (rank %d/%d) does not match the initialized "
         "dushmem world (pe %d/%d)",
         rank, nranks, dushmem_my_pe(), dushmem_n_pes());
    dushmem_finalize();
    return flagcxInternalError;
  }
  shmemInitRefCount++;
  return flagcxSuccess;
}

static flagcxResult_t dushmemAdaptorFinalize() {
  if (shmemInitRefCount > 0 && --shmemInitRefCount == 0) {
    dushmem_finalize();
  }
  return flagcxSuccess;
}

// ============================================================
// Symmetric memory management
// ============================================================
static flagcxResult_t dushmemAdaptorMalloc(void **ptr, size_t size) {
  *ptr = dushmem_malloc(size);
  if (*ptr == nullptr)
    return flagcxSystemError;
  hipMemset(*ptr, 0, size);
  return flagcxSuccess;
}

static flagcxResult_t dushmemAdaptorFree(void *ptr) {
  dushmem_free(ptr);
  return flagcxSuccess;
}

// ============================================================
// Device Comm Create
// ============================================================
static flagcxResult_t dushmemAdaptorDevCommDestroy(flagcxShmemComm_t shmemComm);

static flagcxResult_t
dushmemAdaptorDevCommCreate(flagcxComm_t comm,
                            const struct flagcxDevCommRequirements *reqs,
                            size_t reqsSize, flagcxShmemComm_t *shmemComm) {
  if (comm == nullptr || reqs == nullptr || shmemComm == nullptr ||
      reqsSize < FLAGCX_DEV_COMM_REQUIREMENTS_LEGACY_SIZE)
    return flagcxInvalidArgument;
  *shmemComm = nullptr;

  auto *sc = new flagcxShmemCommInternal();
  memset(sc, 0, sizeof(*sc));
  sc->intraTeam = DUSHMEM_TEAM_INVALID;
  sc->interTeam = DUSHMEM_TEAM_INVALID;

  sc->rank = comm->homoRank;
  sc->nRanks = comm->homoRanks;
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
    sc->signalBuffer = (uint64_t *)dushmem_malloc(signalBytes);
    if (!sc->signalBuffer) {
      delete sc;
      return flagcxSystemError;
    }
    hipMemset(sc->signalBuffer, 0, signalBytes);
  }

  // Counter buffer (local device memory), partitioned by logical context.
  // CounterInc is issued with a device-side atomic fetch-add, so the buffer
  // does not need to be symmetric.
  if (sc->counterCount > 0) {
    size_t counterEntries = (size_t)contextCount * (size_t)sc->counterCount;
    size_t counterBytes = counterEntries * sizeof(uint64_t);
    if (hipMalloc(&sc->counterBuffer, counterBytes) != hipSuccess) {
      goto fail;
    }
    hipMemset(sc->counterBuffer, 0, counterBytes);
  }

  // Shadow buffer (local device memory), partitioned by logical context.
  if (sc->signalCount > 0) {
    size_t shadowEntries = (size_t)contextCount * (size_t)sc->signalCount;
    size_t shadowBytes = shadowEntries * sizeof(uint64_t);
    if (hipMalloc(&sc->shadowBuffer, shadowBytes) != hipSuccess) {
      goto fail;
    }
    hipMemset(sc->shadowBuffer, 0, shadowBytes);
  }

  // Validate topology
  {
    if (sc->intraSize > 0 && sc->nRanks % sc->intraSize != 0) {
      WARN(
          "dushmem devCommCreate: nRanks (%d) not divisible by intraSize (%d); "
          "non-uniform topologies are not supported",
          sc->nRanks, sc->intraSize);
      goto fail;
    }
    int interSize = (sc->intraSize > 0) ? sc->nRanks / sc->intraSize : 1;

    // Grid sync state for multi-block barrier coordination
    // 3 barriers x (arrive[CTA_COUNT] + release[CTA_COUNT]) = 6*CTA_COUNT
    size_t gridSyncSize = 6 * FLAGCX_DEVICE_CTA_COUNT * sizeof(uint64_t);
    if (hipMalloc(&sc->gridSyncState, gridSyncSize) != hipSuccess) {
      goto fail;
    }
    hipMemset(sc->gridSyncState, 0, gridSyncSize);

    // Team creation: intra-node
    // NOTE: assumes uniform intra-node GPU count across all nodes.
    // Heterogeneous topologies are not supported by this strided team-split.
    int nodeId = (sc->intraSize > 0) ? sc->rank / sc->intraSize : 0;
    if (dushmem_team_split_strided(DUSHMEM_TEAM_WORLD, nodeId * sc->intraSize,
                                   1, sc->intraSize, NULL, 0,
                                   &sc->intraTeam) != 0) {
      goto fail;
    }

    // Team creation: inter-node
    if (interSize > 1) {
      if (dushmem_team_split_strided(DUSHMEM_TEAM_WORLD, sc->intraRank,
                                     sc->intraSize, interSize, NULL, 0,
                                     &sc->interTeam) != 0) {
        goto fail;
      }
    } else {
      sc->interTeam = DUSHMEM_TEAM_INVALID;
    }

    sc->worldTeam = DUSHMEM_TEAM_WORLD;
  }

  *shmemComm = sc;
  return flagcxSuccess;

fail:
  dushmemAdaptorDevCommDestroy(sc);
  return flagcxSystemError;
}

// ============================================================
// Device Comm Destroy
// ============================================================
static flagcxResult_t
dushmemAdaptorDevCommDestroy(flagcxShmemComm_t shmemComm) {
  if (shmemComm == nullptr)
    return flagcxSuccess;

  // Free symmetric heap allocations
  if (shmemComm->signalBuffer)
    dushmem_free(shmemComm->signalBuffer);

  // Free local device allocations
  if (shmemComm->counterBuffer)
    hipFree(shmemComm->counterBuffer);
  if (shmemComm->shadowBuffer)
    hipFree(shmemComm->shadowBuffer);
  if (shmemComm->gridSyncState)
    hipFree(shmemComm->gridSyncState);

  // Destroy teams
  if (shmemComm->intraTeam != DUSHMEM_TEAM_INVALID)
    dushmem_team_destroy(shmemComm->intraTeam);
  if (shmemComm->interTeam != DUSHMEM_TEAM_INVALID)
    dushmem_team_destroy(shmemComm->interTeam);

  delete shmemComm;
  return flagcxSuccess;
}

// ============================================================
// Global adaptor instance
// ============================================================
static flagcxShmemAdaptor_t dushmemAdaptorInstance = {
    .name = "dushmem",
    .init = dushmemAdaptorInit,
    .finalize = dushmemAdaptorFinalize,
    .malloc = dushmemAdaptorMalloc,
    .free = dushmemAdaptorFree,
    .devCommCreate = dushmemAdaptorDevCommCreate,
    .devCommDestroy = dushmemAdaptorDevCommDestroy,
};

flagcxShmemAdaptor_t *shmemAdaptor = &dushmemAdaptorInstance;
