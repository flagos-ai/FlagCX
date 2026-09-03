/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * XSHMEM Adaptor — implementation of flagcxShmemAdaptor_t for XSHMEM.
 * Manages XSHMEM lifecycle, symmetric heap allocations, and device comm
 * state (signals, counters, barriers, teams).
 ************************************************************************/

#include "xshmem_adaptor.h"
#include "shmem_adaptor.h"

#include "device_api/xshmem_state_layout.h"
#include "flagcx_kernel_internal.h"
#include "global_comm.h"
#include "kunlunxin_adaptor.h"

#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <mutex>
#include <new>
#include <xshmem/xshmem.h>
#include <xshmem/xshmemx.h>

// ============================================================
// Internal state for one devComm backed by XSHMEM
// ============================================================

// ============================================================
// Lifecycle
// ============================================================
namespace {

std::mutex g_shmemInitLock;
int g_shmemUseCount = 0;
bool g_shmemInitDone = false;
#ifdef USE_KUNLUNXIN_ADAPTOR
BKCLContext_t g_shmemInitCtx = nullptr;
#endif
int g_shmemMyPe = -1;
int g_shmemNPes = -1;

} // namespace

static flagcxResult_t xshmemAdaptorInit(int rank, int nranks, void *handle) {
#ifdef USE_KUNLUNXIN_ADAPTOR
  if (handle == nullptr)
    return flagcxInvalidArgument;
  BKCLContext_t ctx = (BKCLContext_t)handle;

  std::lock_guard<std::mutex> lock(g_shmemInitLock);

  if (!g_shmemInitDone) {
    if (xshmem_init(ctx) != 0)
      return flagcxInternalError;
    g_shmemInitDone = true;
    g_shmemInitCtx = ctx;
    g_shmemMyPe = xshmem_my_pe();
    g_shmemNPes = xshmem_n_pes();
  } else if (ctx != g_shmemInitCtx) {
    WARN("xshmem init: already initialized with a different BKCL context; "
         "xshmem has no finalize/re-init, so one process supports one world");
    return flagcxInvalidUsage;
  }

  if (g_shmemMyPe != rank || g_shmemNPes != nranks) {
    WARN("xshmem init: caller (rank %d/%d) does not match the initialized "
         "xshmem world (pe %d/%d)",
         rank, nranks, g_shmemMyPe, g_shmemNPes);
    return flagcxInvalidUsage;
  }

  ++g_shmemUseCount;
  return flagcxSuccess;
#else
  (void)rank;
  (void)nranks;
  (void)handle;
  return flagcxInternalError;
#endif
}

static flagcxResult_t xshmemAdaptorFinalize() {
  std::lock_guard<std::mutex> lock(g_shmemInitLock);
  if (g_shmemUseCount > 0)
    --g_shmemUseCount;
  return flagcxSuccess;
}

// ============================================================
// Symmetric memory management
// ============================================================
static flagcxResult_t xshmemAdaptorMalloc(void **ptr, size_t size) {
  if (ptr == nullptr || size == 0)
    return flagcxInvalidArgument;
  *ptr = xshmem_malloc(size);
  if (*ptr == nullptr)
    return flagcxSystemError;
  if (cudaMemset(*ptr, 0, size) != cudaSuccess) {
    xshmem_free(*ptr);
    *ptr = nullptr;
    return flagcxUnhandledDeviceError;
  }
  return flagcxSuccess;
}

static flagcxResult_t xshmemAdaptorFree(void *ptr) {
  xshmem_free(ptr);
  return flagcxSuccess;
}

// ============================================================
// Device Comm Create
// ============================================================
static flagcxResult_t xshmemAdaptorDevCommDestroy(flagcxShmemComm_t shmemComm);

static flagcxResult_t
xshmemAdaptorDevCommCreate(flagcxComm_t comm,
                           const struct flagcxDevCommRequirements *reqs,
                           flagcxShmemComm_t *shmemComm) {
  if (comm == nullptr || reqs == nullptr || shmemComm == nullptr)
    return flagcxInvalidArgument;
  *shmemComm = nullptr;

  auto *sc = new (std::nothrow) flagcxShmemCommInternal();
  if (sc == nullptr)
    return flagcxSystemError;
  memset(sc, 0, sizeof(*sc));
  sc->intraTeam = XSHMEM_TEAM_INVALID;
  sc->interTeam = XSHMEM_TEAM_INVALID;

  sc->rank = comm->homoRank;
  sc->nRanks = comm->homoRanks;
  sc->intraRank = comm->localRank;
  sc->intraSize = comm->localRanks;

  // FlagCX communicator ranks are not required to be grouped by host. Keep
  // the exact local-rank -> XSHMEM PE mapping in device memory rather than
  // deriving peers from rank-localRank.
  if (sc->intraSize > 0) {
    size_t bytes = (size_t)sc->intraSize * sizeof(int);
    if (comm->localRankToRank == nullptr ||
        cudaMalloc(&sc->intraPeMap, bytes) != cudaSuccess)
      goto fail;
    if (cudaMemcpy(sc->intraPeMap, comm->localRankToRank, bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess)
      goto fail;
  }

  int contextCount = reqs->interContextCount > 0 ? reqs->interContextCount : 1;
  sc->signalCount = reqs->interSignalCount;
  sc->counterCount = reqs->interCounterCount;

  // Signal buffer (symmetric heap, remote-writable)
  if (sc->signalCount > 0) {
    sc->signalBuffer = (uint64_t *)xshmem_malloc(
        (size_t)contextCount * sc->signalCount * sizeof(uint64_t));
    if (!sc->signalBuffer)
      goto fail;
    if (cudaMemset(sc->signalBuffer, 0,
                   (size_t)contextCount * sc->signalCount * sizeof(uint64_t)) !=
        cudaSuccess)
      goto fail;
  }

  // Counter buffer is symmetric because CounterInc can be a remote action.
  if (sc->counterCount > 0) {
    size_t bytes = (size_t)contextCount * sc->counterCount * sizeof(uint64_t);
    sc->counterBuffer = (uint64_t *)xshmem_malloc(bytes);
    if (sc->counterBuffer == nullptr)
      goto fail;
    if (cudaMemset(sc->counterBuffer, 0, bytes) != cudaSuccess)
      goto fail;
  }

  // Shadow buffer (local device memory)
  if (sc->signalCount > 0) {
    size_t bytes = (size_t)contextCount * sc->signalCount * sizeof(uint64_t);
    if (cudaMalloc(&sc->shadowBuffer, bytes) != cudaSuccess) {
      goto fail;
    }
    if (cudaMemset(sc->shadowBuffer, 0, bytes) != cudaSuccess)
      goto fail;
  }

  // Validate topology
  {
    if (sc->intraSize > 0 && sc->nRanks % sc->intraSize != 0) {
      WARN("xshmem devCommCreate: nRanks (%d) not divisible by intraSize (%d); "
           "non-uniform topologies are not supported",
           sc->nRanks, sc->intraSize);
      goto fail;
    }
    int interSize = (sc->intraSize > 0) ? sc->nRanks / sc->intraSize : 1;

    // Symmetric state lets each team implement a scoped barrier with remote
    // signal increments instead of over-synchronizing XSHMEM_TEAM_WORLD.
    size_t gridSyncSize = FLAGCX_XSHMEM_BARRIER_STATE_WORDS * sizeof(uint64_t);
    sc->gridSyncState = (uint64_t *)xshmem_malloc(gridSyncSize);
    if (sc->gridSyncState == nullptr) {
      goto fail;
    }
    if (cudaMemset(sc->gridSyncState, 0, gridSyncSize) != cudaSuccess)
      goto fail;

    (void)interSize;
    sc->intraTeam = XSHMEMX_TEAM_NODE;

    sc->interTeam = XSHMEM_TEAM_INVALID;

    sc->worldTeam = XSHMEM_TEAM_WORLD;
    sc->devStateHandle = xshmem_get_xshmemi_device_state_h();
    if (sc->devStateHandle == nullptr) {
      WARN("xshmem devCommCreate: device state handle is null after init");
      goto fail;
    }
  }

  *shmemComm = sc;
  return flagcxSuccess;

fail:
  xshmemAdaptorDevCommDestroy(sc);
  return flagcxSystemError;
}

// ============================================================
// Device Comm Destroy
// ============================================================
static flagcxResult_t xshmemAdaptorDevCommDestroy(flagcxShmemComm_t shmemComm) {
  if (shmemComm == nullptr)
    return flagcxSuccess;

  // Symmetric allocations are released in reverse allocation order.
  if (shmemComm->gridSyncState)
    xshmem_free(shmemComm->gridSyncState);
  if (shmemComm->counterBuffer)
    xshmem_free(shmemComm->counterBuffer);
  if (shmemComm->signalBuffer)
    xshmem_free(shmemComm->signalBuffer);

  if (shmemComm->shadowBuffer)
    cudaFree(shmemComm->shadowBuffer);
  if (shmemComm->intraPeMap)
    cudaFree(shmemComm->intraPeMap);

  delete shmemComm;
  return flagcxSuccess;
}

// ============================================================
// Global adaptor instance
// ============================================================
static flagcxShmemAdaptor_t xshmemAdaptorInstance = {
    .name = "xshmem",
    .init = xshmemAdaptorInit,
    .finalize = xshmemAdaptorFinalize,
    .malloc = xshmemAdaptorMalloc,
    .free = xshmemAdaptorFree,
    .devCommCreate = xshmemAdaptorDevCommCreate,
    .devCommDestroy = xshmemAdaptorDevCommDestroy,
};

flagcxShmemAdaptor_t *shmemAdaptor = &xshmemAdaptorInstance;
