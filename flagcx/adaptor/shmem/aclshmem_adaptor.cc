/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * ACLSHMEM Adaptor — implementation of flagcxShmemAdaptor_t for cann-shmem
 * on Ascend (910B/910C).  Manages the aclshmem lifecycle, the symmetric
 * heap used by flagcxMemAlloc(..., flagcxMemSHMEM), and device comm state
 * (signals, counters, barrier tickets) that peers update with MTE puts.
 ************************************************************************/

#include "aclshmem_adaptor.h"
#include "aclshmem_state_layout.h"
#include "shmem_adaptor.h"

#include "debug.h"

#include <acl/acl.h>
#include <shmem.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

// ============================================================
// Lifecycle: reference-counted init/finalize
// ============================================================
static int shmemInitRefCount = 0;
static bool shmemInitialized = false;

static void aclshmemDefaultIpPort(char *out, size_t capacity) {
  const char *env = getenv("FLAGCX_ASCEND_SHMEM_IP_PORT");
  snprintf(out, capacity, "%s",
           (env != nullptr && env[0] != '\0') ? env : "tcp://127.0.0.1:8765");
}

// SHMEM_UID_SESSION_ID takes the bootstrap address without the transport
// prefix ("ip:port"), while aclshmemx_init_attr_t.ip_port takes it with one.
static void aclshmemSetSessionId(const char *ipPort) {
  const char *value = ipPort;
  if (strncmp(value, "tcp://", 6) == 0)
    value += 6;
  setenv("SHMEM_UID_SESSION_ID", value, 0);
}

static uint64_t aclshmemLocalHeapBytes() {
  const char *env = getenv("FLAGCX_ASCEND_SHMEM_HEAP");
  if (env != nullptr && env[0] != '\0') {
    unsigned long long parsed = strtoull(env, nullptr, 0);
    if (parsed > 0)
      return (uint64_t)parsed;
  }
  // The unified-IR suite allocates two 18*maxBytes windows (-e 16M gives
  // 288 MiB each) plus the DevComm state; 4 GiB leaves headroom.
  return (uint64_t)4 << 30;
}

static flagcxResult_t aclshmemAdaptorInit(int rank, int nranks, void *handle) {
  (void)handle;
  if (shmemInitialized) {
    if (aclshmem_my_pe() != rank || aclshmem_n_pes() != nranks) {
      WARN("aclshmem is already initialized for a different rank layout "
           "(my_pe=%d n_pes=%d, requested %d/%d)",
           aclshmem_my_pe(), aclshmem_n_pes(), rank, nranks);
      return flagcxInternalError;
    }
    shmemInitRefCount++;
    return flagcxSuccess;
  }

  aclshmemx_init_attr_t attr;
  attr.my_pe = rank;
  attr.n_pes = nranks;
  aclshmemDefaultIpPort(attr.ip_port, sizeof(attr.ip_port));
  attr.local_mem_size = aclshmemLocalHeapBytes();
  attr.comm_args = nullptr;
  aclshmemSetSessionId(attr.ip_port);

  int ret = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr);
  if (ret != 0) {
    WARN("aclshmemx_init_attr failed: rc=%d ip_port=%s local_mem_size=%llu",
         ret, attr.ip_port, (unsigned long long)attr.local_mem_size);
    return flagcxSystemError;
  }
  if (aclshmem_my_pe() != rank || aclshmem_n_pes() != nranks) {
    WARN("aclshmem PE layout mismatch: my_pe=%d n_pes=%d expected %d/%d",
         aclshmem_my_pe(), aclshmem_n_pes(), rank, nranks);
    aclshmem_finalize();
    return flagcxInternalError;
  }
  shmemInitialized = true;
  shmemInitRefCount++;
  return flagcxSuccess;
}

static flagcxResult_t aclshmemAdaptorFinalize() {
  if (shmemInitRefCount > 0 && --shmemInitRefCount == 0) {
    aclshmem_finalize();
    shmemInitialized = false;
  }
  return flagcxSuccess;
}

// ============================================================
// Symmetric memory management
// ============================================================
static flagcxResult_t aclshmemAdaptorMalloc(void **ptr, size_t size) {
  if (!shmemInitialized)
    return flagcxInternalError;
  *ptr = aclshmem_malloc(size);
  if (*ptr == nullptr)
    return flagcxSystemError;
  if (aclrtMemset(*ptr, size, 0, size) != ACL_SUCCESS)
    return flagcxUnhandledDeviceError;
  return flagcxSuccess;
}

static flagcxResult_t aclshmemAdaptorFree(void *ptr) {
  aclshmem_free(ptr);
  return flagcxSuccess;
}

// ============================================================
// Device Comm Create
// ============================================================
static flagcxResult_t aclshmemAdaptorDevCommDestroy(flagcxShmemComm_t comm);

static flagcxResult_t
aclshmemAdaptorDevCommCreate(flagcxComm_t comm,
                             const struct flagcxDevCommRequirements *reqs,
                             size_t reqsSize, flagcxShmemComm_t *shmemComm) {
  if (comm == nullptr || reqs == nullptr || shmemComm == nullptr ||
      reqsSize < FLAGCX_DEV_COMM_REQUIREMENTS_LEGACY_SIZE)
    return flagcxInvalidArgument;
  *shmemComm = nullptr;

  auto *sc = new flagcxShmemCommInternal();
  memset(sc, 0, sizeof(*sc));
  sc->rank = comm->rank;
  sc->nRanks = comm->nranks;
  sc->intraRank = comm->localRank;
  sc->intraSize = comm->localRanks;

  sc->signalCount = reqs->interSignalCount;
  sc->counterCount = reqs->interCounterCount;
  int contextCount = reqs->interContextCount > 0 ? reqs->interContextCount : 1;

  // All state words are 32-bit: that is the only ordered read/write channel
  // measured on this soc (asc_dcci + __gm__ uint32), so no 64-bit word is
  // assumed anywhere below.
  if (sc->signalCount > 0) {
    size_t signalWords =
        (size_t)contextCount * (size_t)sc->signalCount *
        (size_t)FLAGCX_ACLSHMEM_SIGNAL_SLOTS;
    size_t signalBytes = signalWords * sizeof(uint32_t);
    sc->signalBuffer = (uint64_t *)aclshmem_malloc(signalBytes);
    if (sc->signalBuffer == nullptr)
      goto fail;
    if (aclrtMemset(sc->signalBuffer, signalBytes, 0, signalBytes) !=
        ACL_SUCCESS)
      goto fail;

    size_t shadowBytes =
        (size_t)contextCount * (size_t)sc->signalCount * sizeof(uint32_t);
    void *shadow = nullptr;
    if (aclrtMalloc(&shadow, shadowBytes, ACL_MEM_MALLOC_HUGE_FIRST) !=
        ACL_SUCCESS)
      goto fail;
    sc->shadowBuffer = (uint64_t *)shadow;
    if (aclrtMemset(shadow, shadowBytes, 0, shadowBytes) != ACL_SUCCESS)
      goto fail;
  }

  if (sc->counterCount > 0) {
    size_t counterBytes =
        (size_t)contextCount * (size_t)sc->counterCount * sizeof(uint32_t);
    void *counter = nullptr;
    if (aclrtMalloc(&counter, counterBytes, ACL_MEM_MALLOC_HUGE_FIRST) !=
        ACL_SUCCESS)
      goto fail;
    sc->counterBuffer = (uint64_t *)counter;
    if (aclrtMemset(counter, counterBytes, 0, counterBytes) != ACL_SUCCESS)
      goto fail;
  }

  // Barrier tickets are written into each peer's copy of this state, so it
  // must live in the symmetric heap.
  {
    size_t gridWords = (size_t)FLAGCX_ACLSHMEM_BARRIER_KINDS *
                       (size_t)FLAGCX_ACLSHMEM_BARRIER_SLOTS;
    size_t gridBytes = gridWords * sizeof(uint32_t);
    sc->gridSyncState = (uint64_t *)aclshmem_malloc(gridBytes);
    if (sc->gridSyncState == nullptr)
      goto fail;
    if (aclrtMemset(sc->gridSyncState, gridBytes, 0, gridBytes) != ACL_SUCCESS)
      goto fail;
  }

  // putValue staging: MTE sources sit in GM, never on the stack, and the
  // engine translates a symmetric address, so one 8-byte word per context
  // in the symmetric heap.
  {
    size_t scratchBytes = (size_t)contextCount * 2 * sizeof(uint32_t);
    sc->scratchBuffer = (uint64_t *)aclshmem_malloc(scratchBytes);
    if (sc->scratchBuffer == nullptr)
      goto fail;
    if (aclrtMemset(sc->scratchBuffer, scratchBytes, 0, scratchBytes) !=
        ACL_SUCCESS)
      goto fail;
  }

  if (sc->intraSize > 0 && sc->nRanks % sc->intraSize != 0) {
    WARN("aclshmem devCommCreate: nRanks (%d) not divisible by intraSize (%d)",
         sc->nRanks, sc->intraSize);
    goto fail;
  }

  *shmemComm = sc;
  return flagcxSuccess;

fail:
  aclshmemAdaptorDevCommDestroy(sc);
  return flagcxSystemError;
}

static flagcxResult_t
aclshmemAdaptorDevCommDestroy(flagcxShmemComm_t comm) {
  if (comm == nullptr)
    return flagcxSuccess;
  if (comm->signalBuffer)
    aclshmem_free(comm->signalBuffer);
  if (comm->counterBuffer)
    aclrtFree(comm->counterBuffer);
  if (comm->shadowBuffer)
    aclrtFree(comm->shadowBuffer);
  if (comm->gridSyncState)
    aclshmem_free(comm->gridSyncState);
  if (comm->scratchBuffer)
    aclshmem_free(comm->scratchBuffer);
  delete comm;
  return flagcxSuccess;
}

// ============================================================
// Global adaptor instance
// ============================================================
static flagcxShmemAdaptor_t aclshmemAdaptorInstance = {
    .name = "aclshmem",
    .init = aclshmemAdaptorInit,
    .finalize = aclshmemAdaptorFinalize,
    .malloc = aclshmemAdaptorMalloc,
    .free = aclshmemAdaptorFree,
    .devCommCreate = aclshmemAdaptorDevCommCreate,
    .devCommDestroy = aclshmemAdaptorDevCommDestroy,
};

flagcxShmemAdaptor_t *shmemAdaptor = &aclshmemAdaptorInstance;
