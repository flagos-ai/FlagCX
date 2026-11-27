/*************************************************************************
 * Copyright (c) 2025, BAAI. All rights reserved.
 *
 * NCCL API wrapper that delegates to FlagCX internally.
 * This allows frameworks using NCCL to transparently use FlagCX
 * without code modifications.
 ************************************************************************/

#include "comm_completion.h"
#include "flagcx.h"
#include "nccl.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <mutex>

/* ──────────────────────────────────────────────────────────────────────
 * Minimum NCCL version check
 * ────────────────────────────────────────────────────────────────────── */

#if NCCL_VERSION_CODE < NCCL_VERSION(2, 21, 0)
#error                                                                         \
    "NCCL version 2.21.0 or later is required to build the FlagCX NCCL wrapper"
#endif

/* ──────────────────────────────────────────────────────────────────────
 * Compile-time path to the real NCCL shared library.
 * Set via -DREAL_NCCL_PATH=\"...\" in the Makefile.
 * ────────────────────────────────────────────────────────────────────── */

#ifndef REAL_NCCL_PATH
#error                                                                         \
    "REAL_NCCL_PATH must be defined at compile time (e.g. -DREAL_NCCL_PATH=\\\"/usr/local/nccl/lib/libnccl.so.2\\\")"
#endif

/* ──────────────────────────────────────────────────────────────────────
 * TLS recursive guard
 * A thread-local recursive guard prevents infinite recursion when
 * FlagCX's NCCL adaptor calls back into nccl* symbols.  On re-entry
 * the call is forwarded to the real NCCL loaded via dlopen.
 * ────────────────────────────────────────────────────────────────────── */

static thread_local bool inWrapper = false;

struct recursionGuard {
  bool &flag;
  recursionGuard(bool &f) : flag(f) { flag = true; }
  ~recursionGuard() { flag = false; }
  recursionGuard(const recursionGuard &) = delete;
  recursionGuard &operator=(const recursionGuard &) = delete;
};

/* ──────────────────────────────────────────────────────────────────────
 * Real NCCL library loaded via dlopen
 * ────────────────────────────────────────────────────────────────────── */

struct realNccl {
  void *handle;

  /* Function pointers — one per NCCL API that FlagCX may call back into */
  ncclResult_t (*ncclGetVersion)(int *);
  ncclResult_t (*ncclGetUniqueId)(ncclUniqueId *);
  const char *(*ncclGetErrorString)(ncclResult_t);
  const char *(*ncclGetLastError)(ncclComm_t);
  ncclResult_t (*ncclCommInitRank)(ncclComm_t *, int, ncclUniqueId, int);
  ncclResult_t (*ncclCommFinalize)(ncclComm_t);
  ncclResult_t (*ncclCommDestroy)(ncclComm_t);
  ncclResult_t (*ncclCommAbort)(ncclComm_t);
  ncclResult_t (*ncclCommCount)(const ncclComm_t, int *);
  ncclResult_t (*ncclCommCuDevice)(const ncclComm_t, int *);
  ncclResult_t (*ncclCommUserRank)(const ncclComm_t, int *);
  ncclResult_t (*ncclCommGetAsyncError)(ncclComm_t, ncclResult_t *);
  ncclResult_t (*ncclMemAlloc)(void **, size_t);
  ncclResult_t (*ncclMemFree)(void *);
  ncclResult_t (*ncclCommRegister)(const ncclComm_t, void *, size_t, void **);
  ncclResult_t (*ncclCommDeregister)(const ncclComm_t, void *);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 3)
  ncclResult_t (*ncclCommWindowRegister)(ncclComm_t, void *, size_t,
                                         ncclWindow_t *, int);
  ncclResult_t (*ncclCommWindowDeregister)(ncclComm_t, ncclWindow_t);
#endif
  ncclResult_t (*ncclReduce)(const void *, void *, size_t, ncclDataType_t,
                             ncclRedOp_t, int, ncclComm_t, cudaStream_t);
  ncclResult_t (*ncclBroadcast)(const void *, void *, size_t, ncclDataType_t,
                                int, ncclComm_t, cudaStream_t);
  ncclResult_t (*ncclAllReduce)(const void *, void *, size_t, ncclDataType_t,
                                ncclRedOp_t, ncclComm_t, cudaStream_t);
  ncclResult_t (*ncclReduceScatter)(const void *, void *, size_t,
                                    ncclDataType_t, ncclRedOp_t, ncclComm_t,
                                    cudaStream_t);
  ncclResult_t (*ncclAllGather)(const void *, void *, size_t, ncclDataType_t,
                                ncclComm_t, cudaStream_t);
  ncclResult_t (*ncclSend)(const void *, size_t, ncclDataType_t, int,
                           ncclComm_t, cudaStream_t);
  ncclResult_t (*ncclRecv)(void *, size_t, ncclDataType_t, int, ncclComm_t,
                           cudaStream_t);
  ncclResult_t (*ncclGroupStart)();
  ncclResult_t (*ncclGroupEnd)();
};

static realNccl *realNcclInstance = nullptr;
static std::once_flag realNcclOnce;

static void initRealNccl() {
  realNccl *r = new realNccl();
  r->handle = dlopen(REAL_NCCL_PATH, RTLD_LAZY | RTLD_LOCAL);
  if (!r->handle) {
    fprintf(stderr,
            "FlagCX NCCL wrapper: failed to dlopen real NCCL at %s: %s\n",
            REAL_NCCL_PATH, dlerror());
    delete r;
    return;
  }

#define LOAD_SYM(name)                                                         \
  r->name = reinterpret_cast<decltype(r->name)>(dlsym(r->handle, #name));      \
  if (!r->name) {                                                              \
    fprintf(stderr, "FlagCX NCCL wrapper: dlsym failed for " #name ": %s\n",   \
            dlerror());                                                        \
  }

  LOAD_SYM(ncclGetVersion)
  LOAD_SYM(ncclGetUniqueId)
  LOAD_SYM(ncclGetErrorString)
  LOAD_SYM(ncclGetLastError)
  LOAD_SYM(ncclCommInitRank)
  LOAD_SYM(ncclCommFinalize)
  LOAD_SYM(ncclCommDestroy)
  LOAD_SYM(ncclCommAbort)
  LOAD_SYM(ncclCommCount)
  LOAD_SYM(ncclCommCuDevice)
  LOAD_SYM(ncclCommUserRank)
  LOAD_SYM(ncclCommGetAsyncError)
  LOAD_SYM(ncclMemAlloc)
  LOAD_SYM(ncclMemFree)
  LOAD_SYM(ncclCommRegister)
  LOAD_SYM(ncclCommDeregister)
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 3)
  LOAD_SYM(ncclCommWindowRegister)
  LOAD_SYM(ncclCommWindowDeregister)
#endif
  LOAD_SYM(ncclReduce)
  LOAD_SYM(ncclBroadcast)
  LOAD_SYM(ncclAllReduce)
  LOAD_SYM(ncclReduceScatter)
  LOAD_SYM(ncclAllGather)
  LOAD_SYM(ncclSend)
  LOAD_SYM(ncclRecv)
  LOAD_SYM(ncclGroupStart)
  LOAD_SYM(ncclGroupEnd)

#undef LOAD_SYM

  realNcclInstance = r;
}

static realNccl &getRealNccl() {
  std::call_once(realNcclOnce, initRealNccl);
  return *realNcclInstance;
}

/* ──────────────────────────────────────────────────────────────────────
 * Internal ncclComm struct
 * ────────────────────────────────────────────────────────────────────── */

struct ncclComm {
  flagcxDeviceHandle_t devHandle;
  flagcxComm_t comm;
  int rank;
  int nranks;
  flagcxResult_t asyncError;
  struct flagcxNcclCompletion completion;
  bool finalizeRequested, finalizeComplete;
};

static flagcxResult_t advanceNcclFinalize(ncclComm_t comm) {
  if (pthread_mutex_lock(&comm->completion.mutex) != 0)
    return flagcxSystemError;
  bool complete = comm->finalizeComplete;
  pthread_mutex_unlock(&comm->completion.mutex);
  if (complete)
    return flagcxSuccess;
  if (flagcxNcclGroupDepth != 0)
    return flagcxInProgress;
  flagcxResult_t result = flagcxNcclDrain(&comm->completion);
  if (result != flagcxSuccess)
    return result;
  result = flagcxCommFinalize(comm->comm);
  if (result == flagcxSuccess) {
    if (pthread_mutex_lock(&comm->completion.mutex) != 0)
      return flagcxSystemError;
    comm->finalizeComplete = true;
    pthread_mutex_unlock(&comm->completion.mutex);
  }
  return result;
}

/* ──────────────────────────────────────────────────────────────────────
 * Helper: ncclDataType_t -> flagcxDataType_t
 * ────────────────────────────────────────────────────────────────────── */

static ncclResult_t toFlagcxDataType(ncclDataType_t ncclType,
                                     flagcxDataType_t *flagcxType) {
  switch (ncclType) {
    case ncclInt8:
      *flagcxType = flagcxInt8;
      return ncclSuccess;
    case ncclUint8:
      *flagcxType = flagcxUint8;
      return ncclSuccess;
    case ncclInt32:
      *flagcxType = flagcxInt32;
      return ncclSuccess;
    case ncclUint32:
      *flagcxType = flagcxUint32;
      return ncclSuccess;
    case ncclInt64:
      *flagcxType = flagcxInt64;
      return ncclSuccess;
    case ncclUint64:
      *flagcxType = flagcxUint64;
      return ncclSuccess;
    case ncclFloat16:
      *flagcxType = flagcxFloat16;
      return ncclSuccess;
    case ncclFloat32:
      *flagcxType = flagcxFloat32;
      return ncclSuccess;
    case ncclFloat64:
      *flagcxType = flagcxFloat64;
      return ncclSuccess;
    case ncclBfloat16:
      *flagcxType = flagcxBfloat16;
      return ncclSuccess;
    default:
      /* ncclFloat8e4m3 (10), ncclFloat8e5m2 (11) have no FlagCX equivalent */
      return ncclInvalidUsage;
  }
}

/* ──────────────────────────────────────────────────────────────────────
 * Helper: ncclRedOp_t -> flagcxRedOp_t
 * ────────────────────────────────────────────────────────────────────── */

static ncclResult_t toFlagcxRedOp(ncclRedOp_t ncclOp, flagcxRedOp_t *flagcxOp) {
  if ((int)ncclOp >= 0 && (int)ncclOp < (int)ncclNumOps) {
    *flagcxOp = (flagcxRedOp_t)(int)ncclOp;
    return ncclSuccess;
  }
  /* Custom / dynamic reduction ops are not supported */
  return ncclInvalidUsage;
}

/* ──────────────────────────────────────────────────────────────────────
 * Helper: flagcxResult_t -> ncclResult_t
 * ────────────────────────────────────────────────────────────────────── */

static ncclResult_t toNcclResult(flagcxResult_t res) {
  switch (res) {
    case flagcxSuccess:
      return ncclSuccess;
    case flagcxUnhandledDeviceError:
      return ncclUnhandledCudaError;
    case flagcxSystemError:
      return ncclSystemError;
    case flagcxInternalError:
      return ncclInternalError;
    case flagcxInvalidArgument:
      return ncclInvalidArgument;
    case flagcxInvalidUsage:
      return ncclInvalidUsage;
    case flagcxRemoteError:
      return ncclRemoteError;
    case flagcxInProgress:
      return ncclInProgress;
    default:
      return ncclInternalError;
  }
}

/* ──────────────────────────────────────────────────────────────────────
 * Version / Error String APIs
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclGetVersion(int *version) {
  if (inWrapper) {
    return getRealNccl().ncclGetVersion(version);
  }
  recursionGuard guard(inWrapper);
  if (version == nullptr)
    return ncclInvalidArgument;
  *version = NCCL_VERSION_CODE;
  return ncclSuccess;
}

const char *ncclGetErrorString(ncclResult_t result) {
  if (inWrapper) {
    return getRealNccl().ncclGetErrorString(result);
  }
  recursionGuard guard(inWrapper);
  switch (result) {
    case ncclSuccess:
      return "no error";
    case ncclUnhandledCudaError:
      return "unhandled cuda error (run with NCCL_DEBUG=WARN for details)";
    case ncclSystemError:
      return "unhandled system error (run with NCCL_DEBUG=WARN for details)";
    case ncclInternalError:
      return "internal error - please report this issue to the NCCL developers";
    case ncclInvalidArgument:
      return "invalid argument (run with NCCL_DEBUG=WARN for details)";
    case ncclInvalidUsage:
      return "invalid usage (run with NCCL_DEBUG=WARN for details)";
    case ncclRemoteError:
      return "remote process exited or there was a network error";
    case ncclInProgress:
      return "NCCL operation in progress";
    default:
      return "unknown result code";
  }
}

const char *ncclGetLastError(ncclComm_t comm) {
  if (inWrapper) {
    return getRealNccl().ncclGetLastError(comm);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return nullptr;
  return flagcxGetLastError(comm->comm);
}

/* ──────────────────────────────────────────────────────────────────────
 * UniqueId
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclGetUniqueId(ncclUniqueId *uniqueId) {
  if (inWrapper) {
    return getRealNccl().ncclGetUniqueId(uniqueId);
  }
  recursionGuard guard(inWrapper);
  if (uniqueId == nullptr)
    return ncclInvalidArgument;

  flagcxUniqueId flagcxId;
  flagcxResult_t res = flagcxGetUniqueId(&flagcxId);
  if (res != flagcxSuccess) {
    return toNcclResult(res);
  }

  /* flagcxBootstrapHandle fits in NCCL_UNIQUE_ID_BYTES (128).
   * Copy the first 128 bytes of the 256-byte flagcxUniqueId. */
  memset(uniqueId, 0, sizeof(ncclUniqueId));
  memcpy(uniqueId->internal, flagcxId.internal, NCCL_UNIQUE_ID_BYTES);

  return ncclSuccess;
}

/* ──────────────────────────────────────────────────────────────────────
 * Communicator Init / Finalize / Destroy / Abort
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId commId,
                              int rank) {
  if (inWrapper) {
    return getRealNccl().ncclCommInitRank(comm, nranks, commId, rank);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;

  *comm = nullptr;
  ncclComm_t c = (ncclComm_t)calloc(1, sizeof(struct ncclComm));
  if (c == nullptr)
    return ncclSystemError;

  if (pthread_mutex_init(&c->completion.mutex, NULL) != 0) {
    free(c);
    return ncclSystemError;
  }

  /* Init FlagCX device handle */
  flagcxResult_t res = flagcxDeviceHandleInit(&c->devHandle);
  if (res != flagcxSuccess) {
    if (c->devHandle != nullptr)
      *comm = c; // The failed Init retained a cleanup owner.
    else {
      pthread_mutex_destroy(&c->completion.mutex);
      free(c);
    }
    return toNcclResult(res);
  }

  /* Reconstruct a flagcxUniqueId from the NCCL 128-byte id.
   * Zero-init the full 256-byte struct, then copy in the 128-byte handle. */
  flagcxUniqueId uniqueId;
  memset(&uniqueId, 0, sizeof(flagcxUniqueId));
  memcpy(uniqueId.internal, commId.internal, NCCL_UNIQUE_ID_BYTES);

  /* Init the FlagCX communicator */
  res = flagcxCommInitRank(&c->comm, nranks, &uniqueId, rank);
  if (res != flagcxSuccess) {
    if (c->comm != nullptr) {
      *comm = c;
      return toNcclResult(res);
    }
    flagcxResult_t cleanup = flagcxDeviceHandleFree(c->devHandle);
    if (cleanup != flagcxSuccess) {
      *comm = c;
      return toNcclResult(cleanup);
    }
    pthread_mutex_destroy(&c->completion.mutex);
    free(c);
    return toNcclResult(res);
  }

  c->rank = rank;
  c->nranks = nranks;
  c->asyncError = flagcxSuccess;

  *comm = c;
  return ncclSuccess;
}

ncclResult_t ncclCommInitRankConfig(ncclComm_t *comm, int nranks,
                                    ncclUniqueId commId, int rank,
                                    ncclConfig_t *config) {
  /* Config fields are NCCL-specific; FlagCX has no equivalent.
   * Delegate to the non-config version (which handles the guard). */
  (void)config;
  return ncclCommInitRank(comm, nranks, commId, rank);
}

ncclResult_t ncclCommFinalize(ncclComm_t comm) {
  if (inWrapper) {
    return getRealNccl().ncclCommFinalize(comm);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  if (pthread_mutex_lock(&comm->completion.mutex) != 0)
    return ncclSystemError;
  if (comm->finalizeComplete) {
    pthread_mutex_unlock(&comm->completion.mutex);
    return ncclSuccess;
  }
  if (flagcxNcclGroupDepth != 0) {
    pthread_mutex_unlock(&comm->completion.mutex);
    return ncclInvalidUsage;
  }
  comm->finalizeRequested = true;
  pthread_mutex_unlock(&comm->completion.mutex);
  return toNcclResult(advanceNcclFinalize(comm));
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (inWrapper) {
    return getRealNccl().ncclCommDestroy(comm);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  if (comm->comm != nullptr) {
    flagcxResult_t result = flagcxNcclDrain(&comm->completion);
    if (result != flagcxSuccess)
      return toNcclResult(result);
    result = flagcxCommDestroy(comm->comm);
    if (result != flagcxSuccess)
      return toNcclResult(result);
    comm->comm = nullptr;
  }
  flagcxResult_t result = flagcxNcclRelease(&comm->completion);
  if (result != flagcxSuccess)
    return toNcclResult(result);
  result = flagcxDeviceHandleFree(comm->devHandle);
  if (result != flagcxSuccess)
    return toNcclResult(result);
  comm->devHandle = nullptr;
  if (pthread_mutex_destroy(&comm->completion.mutex) != 0)
    return ncclSystemError;
  free(comm);
  return ncclSuccess;
}

ncclResult_t ncclCommAbort(ncclComm_t comm) {
  if (inWrapper) {
    return getRealNccl().ncclCommAbort(comm);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  if (comm->comm != nullptr) {
    flagcxResult_t result = flagcxCommAbort(comm->comm);
    if (result != flagcxSuccess)
      return toNcclResult(result);
    comm->comm = nullptr;
  }
  flagcxResult_t result = flagcxNcclRelease(&comm->completion);
  if (result != flagcxSuccess)
    return toNcclResult(result);
  result = flagcxDeviceHandleFree(comm->devHandle);
  if (result != flagcxSuccess)
    return toNcclResult(result);
  comm->devHandle = nullptr;
  if (pthread_mutex_destroy(&comm->completion.mutex) != 0)
    return ncclSystemError;
  free(comm);
  return ncclSuccess;
}

/* ──────────────────────────────────────────────────────────────────────
 * Communicator Query APIs
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclCommCount(const ncclComm_t comm, int *count) {
  if (inWrapper) {
    return getRealNccl().ncclCommCount(comm, count);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  return toNcclResult(flagcxCommCount(comm->comm, count));
}

ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int *device) {
  if (inWrapper) {
    return getRealNccl().ncclCommCuDevice(comm, device);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  return toNcclResult(flagcxCommGetDeviceNumber(comm->comm, device));
}

ncclResult_t ncclCommUserRank(const ncclComm_t comm, int *rank) {
  if (inWrapper) {
    return getRealNccl().ncclCommUserRank(comm, rank);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  return toNcclResult(flagcxCommUserRank(comm->comm, rank));
}

ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
  if (inWrapper) {
    return getRealNccl().ncclCommGetAsyncError(comm, asyncError);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr || asyncError == nullptr)
    return ncclInvalidArgument;
  if (pthread_mutex_lock(&comm->completion.mutex) != 0)
    return ncclSystemError;
  bool finalizeRequested = comm->finalizeRequested;
  pthread_mutex_unlock(&comm->completion.mutex);
  if (finalizeRequested) {
    *asyncError = toNcclResult(advanceNcclFinalize(comm));
    return ncclSuccess;
  }
  flagcxResult_t flagcxAsync;
  flagcxResult_t res = flagcxCommGetAsyncError(comm->comm, &flagcxAsync);
  if (res != flagcxSuccess)
    return toNcclResult(res);
  if (flagcxAsync == flagcxSuccess && flagcxNcclGroupDepth == 0) {
    if (pthread_mutex_lock(&comm->completion.mutex) != 0)
      return ncclSystemError;
    comm->completion.nativePending = false;
    res = flagcxNcclRecordPending(&comm->completion);
    pthread_mutex_unlock(&comm->completion.mutex);
    if (res != flagcxSuccess)
      return toNcclResult(res);
  }
  *asyncError = toNcclResult(flagcxAsync);
  return ncclSuccess;
}

/* ──────────────────────────────────────────────────────────────────────
 * Buffer Registration
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclCommRegister(const ncclComm_t comm, void *buff, size_t size,
                              void **handle) {
  if (inWrapper) {
    return getRealNccl().ncclCommRegister(comm, buff, size, handle);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  return toNcclResult(flagcxCommRegister(comm->comm, buff, size, handle));
}

ncclResult_t ncclCommDeregister(const ncclComm_t comm, void *handle) {
  if (inWrapper) {
    return getRealNccl().ncclCommDeregister(comm, handle);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  return toNcclResult(flagcxCommDeregister(comm->comm, handle));
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 3)
ncclResult_t ncclCommWindowRegister(ncclComm_t comm, void *buff, size_t size,
                                    ncclWindow_t *win, int winFlags) {
  if (inWrapper) {
    return getRealNccl().ncclCommWindowRegister(comm, buff, size, win,
                                                winFlags);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  return toNcclResult(flagcxCommWindowRegister(
      comm->comm, buff, size, (flagcxWindow_t *)win, winFlags));
}

ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win) {
  if (inWrapper) {
    return getRealNccl().ncclCommWindowDeregister(comm, win);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  return toNcclResult(
      flagcxCommWindowDeregister(comm->comm, (flagcxWindow_t)win));
}
#endif

/* ──────────────────────────────────────────────────────────────────────
 * Memory Allocation
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclMemAlloc(void **ptr, size_t size) {
  if (inWrapper) {
    return getRealNccl().ncclMemAlloc(ptr, size);
  }
  recursionGuard guard(inWrapper);
  return toNcclResult(flagcxMemAlloc(ptr, size));
}

ncclResult_t ncclMemFree(void *ptr) {
  if (inWrapper) {
    return getRealNccl().ncclMemFree(ptr);
  }
  recursionGuard guard(inWrapper);
  return toNcclResult(flagcxMemFree(ptr));
}

/* ──────────────────────────────────────────────────────────────────────
 * Group Semantics
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclGroupStart() {
  if (inWrapper) {
    return getRealNccl().ncclGroupStart();
  }
  recursionGuard guard(inWrapper);
  if (flagcxNcclGroupDepth == INT_MAX)
    return ncclInvalidUsage;
  flagcxResult_t result = flagcxGroupStart(nullptr);
  if (result == flagcxSuccess || result == flagcxInProgress)
    ++flagcxNcclGroupDepth;
  return toNcclResult(result);
}

ncclResult_t ncclGroupEnd() {
  if (inWrapper) {
    return getRealNccl().ncclGroupEnd();
  }
  recursionGuard guard(inWrapper);
  if (flagcxNcclGroupDepth == 0)
    return ncclInvalidUsage;
  flagcxResult_t result = flagcxGroupEnd(nullptr);
  --flagcxNcclGroupDepth;
  if (flagcxNcclGroupDepth == 0)
    result = flagcxNcclGroupComplete(result);
  return toNcclResult(result);
}

/* ──────────────────────────────────────────────────────────────────────
 * Collective Operations
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream) {
  if (inWrapper) {
    return getRealNccl().ncclAllReduce(sendbuff, recvbuff, count, datatype, op,
                                       comm, stream);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  flagcxDataType_t fType;
  flagcxRedOp_t fOp;
  ncclResult_t r;
  if ((r = toFlagcxDataType(datatype, &fType)) != ncclSuccess)
    return r;
  if ((r = toFlagcxRedOp(op, &fOp)) != ncclSuccess)
    return r;
  struct flagcxNcclStream *entry = NULL;
  flagcxResult_t acquired = flagcxNcclBegin(
      &comm->completion, comm->comm, comm->devHandle, (void *)stream, &entry);
  if (acquired != flagcxSuccess)
    return toNcclResult(acquired);
  return toNcclResult(flagcxNcclEnd(
      &comm->completion, flagcxAllReduce(sendbuff, recvbuff, count, fType, fOp,
                                         comm->comm, entry->wrapper)));
}

ncclResult_t ncclBroadcast(const void *sendbuff, void *recvbuff, size_t count,
                           ncclDataType_t datatype, int root, ncclComm_t comm,
                           cudaStream_t stream) {
  if (inWrapper) {
    return getRealNccl().ncclBroadcast(sendbuff, recvbuff, count, datatype,
                                       root, comm, stream);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  flagcxDataType_t fType;
  ncclResult_t r;
  if ((r = toFlagcxDataType(datatype, &fType)) != ncclSuccess)
    return r;
  struct flagcxNcclStream *entry = NULL;
  flagcxResult_t acquired = flagcxNcclBegin(
      &comm->completion, comm->comm, comm->devHandle, (void *)stream, &entry);
  if (acquired != flagcxSuccess)
    return toNcclResult(acquired);
  return toNcclResult(flagcxNcclEnd(
      &comm->completion, flagcxBroadcast(sendbuff, recvbuff, count, fType, root,
                                         comm->comm, entry->wrapper)));
}

ncclResult_t ncclReduce(const void *sendbuff, void *recvbuff, size_t count,
                        ncclDataType_t datatype, ncclRedOp_t op, int root,
                        ncclComm_t comm, cudaStream_t stream) {
  if (inWrapper) {
    return getRealNccl().ncclReduce(sendbuff, recvbuff, count, datatype, op,
                                    root, comm, stream);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  flagcxDataType_t fType;
  flagcxRedOp_t fOp;
  ncclResult_t r;
  if ((r = toFlagcxDataType(datatype, &fType)) != ncclSuccess)
    return r;
  if ((r = toFlagcxRedOp(op, &fOp)) != ncclSuccess)
    return r;
  struct flagcxNcclStream *entry = NULL;
  flagcxResult_t acquired = flagcxNcclBegin(
      &comm->completion, comm->comm, comm->devHandle, (void *)stream, &entry);
  if (acquired != flagcxSuccess)
    return toNcclResult(acquired);
  return toNcclResult(flagcxNcclEnd(
      &comm->completion, flagcxReduce(sendbuff, recvbuff, count, fType, fOp,
                                      root, comm->comm, entry->wrapper)));
}

ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff,
                           size_t sendcount, ncclDataType_t datatype,
                           ncclComm_t comm, cudaStream_t stream) {
  if (inWrapper) {
    return getRealNccl().ncclAllGather(sendbuff, recvbuff, sendcount, datatype,
                                       comm, stream);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  flagcxDataType_t fType;
  ncclResult_t r;
  if ((r = toFlagcxDataType(datatype, &fType)) != ncclSuccess)
    return r;
  struct flagcxNcclStream *entry = NULL;
  flagcxResult_t acquired = flagcxNcclBegin(
      &comm->completion, comm->comm, comm->devHandle, (void *)stream, &entry);
  if (acquired != flagcxSuccess)
    return toNcclResult(acquired);
  return toNcclResult(flagcxNcclEnd(
      &comm->completion, flagcxAllGather(sendbuff, recvbuff, sendcount, fType,
                                         comm->comm, entry->wrapper)));
}

ncclResult_t ncclReduceScatter(const void *sendbuff, void *recvbuff,
                               size_t recvcount, ncclDataType_t datatype,
                               ncclRedOp_t op, ncclComm_t comm,
                               cudaStream_t stream) {
  if (inWrapper) {
    return getRealNccl().ncclReduceScatter(sendbuff, recvbuff, recvcount,
                                           datatype, op, comm, stream);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  flagcxDataType_t fType;
  flagcxRedOp_t fOp;
  ncclResult_t r;
  if ((r = toFlagcxDataType(datatype, &fType)) != ncclSuccess)
    return r;
  if ((r = toFlagcxRedOp(op, &fOp)) != ncclSuccess)
    return r;
  struct flagcxNcclStream *entry = NULL;
  flagcxResult_t acquired = flagcxNcclBegin(
      &comm->completion, comm->comm, comm->devHandle, (void *)stream, &entry);
  if (acquired != flagcxSuccess)
    return toNcclResult(acquired);
  return toNcclResult(
      flagcxNcclEnd(&comm->completion,
                    flagcxReduceScatter(sendbuff, recvbuff, recvcount, fType,
                                        fOp, comm->comm, entry->wrapper)));
}

/* ──────────────────────────────────────────────────────────────────────
 * Point-to-Point Operations
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclSend(const void *sendbuff, size_t count,
                      ncclDataType_t datatype, int peer, ncclComm_t comm,
                      cudaStream_t stream) {
  if (inWrapper) {
    return getRealNccl().ncclSend(sendbuff, count, datatype, peer, comm,
                                  stream);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  flagcxDataType_t fType;
  ncclResult_t r;
  if ((r = toFlagcxDataType(datatype, &fType)) != ncclSuccess)
    return r;
  struct flagcxNcclStream *entry = NULL;
  flagcxResult_t acquired = flagcxNcclBegin(
      &comm->completion, comm->comm, comm->devHandle, (void *)stream, &entry);
  if (acquired != flagcxSuccess)
    return toNcclResult(acquired);
  return toNcclResult(
      flagcxNcclEnd(&comm->completion, flagcxSend(sendbuff, count, fType, peer,
                                                  comm->comm, entry->wrapper)));
}

ncclResult_t ncclRecv(void *recvbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
  if (inWrapper) {
    return getRealNccl().ncclRecv(recvbuff, count, datatype, peer, comm,
                                  stream);
  }
  recursionGuard guard(inWrapper);
  if (comm == nullptr)
    return ncclInvalidArgument;
  flagcxDataType_t fType;
  ncclResult_t r;
  if ((r = toFlagcxDataType(datatype, &fType)) != ncclSuccess)
    return r;
  struct flagcxNcclStream *entry = NULL;
  flagcxResult_t acquired = flagcxNcclBegin(
      &comm->completion, comm->comm, comm->devHandle, (void *)stream, &entry);
  if (acquired != flagcxSuccess)
    return toNcclResult(acquired);
  return toNcclResult(
      flagcxNcclEnd(&comm->completion, flagcxRecv(recvbuff, count, fType, peer,
                                                  comm->comm, entry->wrapper)));
}

/* ──────────────────────────────────────────────────────────────────────
 * Unsupported APIs
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclBcast(void *buff, size_t count, ncclDataType_t datatype,
                       int root, ncclComm_t comm, cudaStream_t stream) {
  return ncclInvalidUsage;
}

ncclResult_t ncclCommInitAll(ncclComm_t *comm, int ndev, const int *devlist) {
  return ncclInvalidUsage;
}

ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key,
                           ncclComm_t *newcomm, ncclConfig_t *config) {
  return ncclInvalidUsage;
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 27, 3)
ncclResult_t ncclCommShrink(ncclComm_t comm, int *excludeRanksList,
                            int excludeRanksCount, ncclComm_t *newcomm,
                            ncclConfig_t *config, int shrinkFlags) {
  return ncclInvalidUsage;
}
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 23, 4)
ncclResult_t ncclCommInitRankScalable(ncclComm_t *newcomm, int nranks,
                                      int myrank, int nId,
                                      ncclUniqueId *commIds,
                                      ncclConfig_t *config) {
  return ncclInvalidUsage;
}
#endif

ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar,
                                      ncclDataType_t datatype,
                                      ncclScalarResidence_t residence,
                                      ncclComm_t comm) {
  return ncclInvalidUsage;
}

ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
  return ncclInvalidUsage;
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 22, 3)
ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t *simInfo) {
  return ncclInvalidUsage;
}
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 24, 3)
void ncclResetDebugInit() { /* Deprecated in NCCL, no-op */
}
#endif
