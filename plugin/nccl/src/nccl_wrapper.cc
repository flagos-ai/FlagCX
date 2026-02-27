/*************************************************************************
 * Copyright (c) 2025, FlagCX CONTRIBUTORS. All rights reserved.
 *
 * NCCL 2.28.3 API wrapper that delegates to FlagCX internally.
 * This allows frameworks using NCCL to transparently use FlagCX
 * without code modifications.
 ************************************************************************/

#include "flagcx.h"
#include "nccl.h"
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

/* ──────────────────────────────────────────────────────────────────────
 * Internal ncclComm struct
 * ────────────────────────────────────────────────────────────────────── */

struct ncclComm {
  flagcxHandlerGroup_t handler; // handler group (uniqueId, comm, devHandle)
  int rank;
  int nranks;
  flagcxResult_t asyncError;
};

/* ──────────────────────────────────────────────────────────────────────
 * Helper: wrap a cudaStream_t into a temporary flagcxStream_t
 *
 * FlagCX wraps CUDA streams in `struct flagcxStream { cudaStream_t base; }`.
 * The struct definition lives in the adaptor headers (not public), so we
 * define a layout-compatible version here for stream wrapping.
 * We allocate a temporary wrapper, call the FlagCX API, then free it.
 * ────────────────────────────────────────────────────────────────────── */

struct flagcxStream {
  cudaStream_t base;
};

struct FlagcxStreamWrapper {
  flagcxStream_t stream;

  FlagcxStreamWrapper(cudaStream_t cudaStream) {
    stream = (flagcxStream_t)malloc(sizeof(struct flagcxStream));
    if (stream) {
      stream->base = cudaStream;
    }
  }

  ~FlagcxStreamWrapper() { free(stream); }

  FlagcxStreamWrapper(const FlagcxStreamWrapper &) = delete;
  FlagcxStreamWrapper &operator=(const FlagcxStreamWrapper &) = delete;
};

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
 * Thread-local group tracking state
 *
 * NCCL's ncclGroupStart/End take no args; FlagCX's require a comm.
 * We track group depth and capture the first communicator used within
 * a group to pass to FlagCX group APIs.
 * ────────────────────────────────────────────────────────────────────── */

static thread_local int groupDepth = 0;
static thread_local flagcxComm_t groupComm = nullptr;

/* ──────────────────────────────────────────────────────────────────────
 * Version / Error String APIs
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclGetVersion(int *version) {
  if (version == nullptr)
    return ncclInvalidArgument;
  *version = NCCL_VERSION_CODE;
  return ncclSuccess;
}

const char *ncclGetErrorString(ncclResult_t result) {
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
  if (comm == nullptr)
    return nullptr;
  return flagcxGetLastError(comm->handler->comm);
}

/* ──────────────────────────────────────────────────────────────────────
 * UniqueId
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclGetUniqueId(ncclUniqueId *uniqueId) {
  if (uniqueId == nullptr)
    return ncclInvalidArgument;

  flagcxUniqueId_t flagcxId = nullptr;
  flagcxResult_t res = flagcxGetUniqueId(&flagcxId);
  if (res != flagcxSuccess) {
    return toNcclResult(res);
  }

  /* flagcxBootstrapHandle fits in NCCL_UNIQUE_ID_BYTES (128).
   * Copy the first 128 bytes of the 256-byte flagcxUniqueId. */
  memset(uniqueId, 0, sizeof(ncclUniqueId));
  memcpy(uniqueId->internal, flagcxId->internal, NCCL_UNIQUE_ID_BYTES);

  free(flagcxId);
  return ncclSuccess;
}

/* ──────────────────────────────────────────────────────────────────────
 * Communicator Init / Finalize / Destroy / Abort
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId commId,
                              int rank) {
  if (comm == nullptr)
    return ncclInvalidArgument;

  ncclComm_t c = (ncclComm_t)calloc(1, sizeof(struct ncclComm));
  if (c == nullptr)
    return ncclSystemError;

  /* Init FlagCX handler group */
  flagcxResult_t res = flagcxHandleInit(&c->handler);
  if (res != flagcxSuccess) {
    free(c);
    return toNcclResult(res);
  }

  /* Reconstruct a flagcxUniqueId from the NCCL 128-byte id.
   * Zero-init the full 256-byte struct, then copy in the 128-byte handle. */
  memset(c->handler->uniqueId, 0, sizeof(flagcxUniqueId));
  memcpy(c->handler->uniqueId->internal, commId.internal, NCCL_UNIQUE_ID_BYTES);

  /* Init the FlagCX communicator */
  res =
      flagcxCommInitRank(&c->handler->comm, nranks, c->handler->uniqueId, rank);
  if (res != flagcxSuccess) {
    flagcxHandleFree(c->handler);
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
   * Delegate to the non-config version. */
  (void)config;
  return ncclCommInitRank(comm, nranks, commId, rank);
}

ncclResult_t ncclCommFinalize(ncclComm_t comm) {
  if (comm == nullptr)
    return ncclInvalidArgument;
  return toNcclResult(flagcxCommFinalize(comm->handler->comm));
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (comm == nullptr)
    return ncclInvalidArgument;
  flagcxResult_t res = flagcxCommDestroy(comm->handler->comm);
  /* flagcxCommDestroy frees internal resources but not the comm struct.
   * flagcxHandleFree will free the comm struct pointer along with handler. */
  flagcxHandleFree(comm->handler);
  free(comm);
  return toNcclResult(res);
}

ncclResult_t ncclCommAbort(ncclComm_t comm) {
  if (comm == nullptr)
    return ncclInvalidArgument;
  flagcxResult_t res = flagcxCommAbort(comm->handler->comm);
  flagcxHandleFree(comm->handler);
  free(comm);
  return toNcclResult(res);
}

/* ──────────────────────────────────────────────────────────────────────
 * Communicator Query APIs
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclCommCount(const ncclComm_t comm, int *count) {
  if (comm == nullptr)
    return ncclInvalidArgument;
  return toNcclResult(flagcxCommCount(comm->handler->comm, count));
}

ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int *device) {
  if (comm == nullptr)
    return ncclInvalidArgument;
  return toNcclResult(flagcxCommGetDeviceNumber(comm->handler->comm, device));
}

ncclResult_t ncclCommUserRank(const ncclComm_t comm, int *rank) {
  if (comm == nullptr)
    return ncclInvalidArgument;
  return toNcclResult(flagcxCommUserRank(comm->handler->comm, rank));
}

ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
  if (comm == nullptr)
    return ncclInvalidArgument;
  flagcxResult_t flagcxAsync;
  flagcxResult_t res =
      flagcxCommGetAsyncError(comm->handler->comm, &flagcxAsync);
  if (res != flagcxSuccess)
    return toNcclResult(res);
  *asyncError = toNcclResult(flagcxAsync);
  return ncclSuccess;
}

/* ──────────────────────────────────────────────────────────────────────
 * Buffer Registration
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclCommRegister(const ncclComm_t comm, void *buff, size_t size,
                              void **handle) {
  if (comm == nullptr)
    return ncclInvalidArgument;
  return toNcclResult(
      flagcxCommRegister(comm->handler->comm, buff, size, handle));
}

ncclResult_t ncclCommDeregister(const ncclComm_t comm, void *handle) {
  if (comm == nullptr)
    return ncclInvalidArgument;
  return toNcclResult(flagcxCommDeregister(comm->handler->comm, handle));
}

ncclResult_t ncclCommWindowRegister(ncclComm_t comm, void *buff, size_t size,
                                    ncclWindow_t *win, int winFlags) {
  if (comm == nullptr)
    return ncclInvalidArgument;
  return toNcclResult(flagcxCommWindowRegister(
      comm->handler->comm, buff, size, (flagcxWindow_t *)win, winFlags));
}

ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win) {
  if (comm == nullptr)
    return ncclInvalidArgument;
  return toNcclResult(
      flagcxCommWindowDeregister(comm->handler->comm, (flagcxWindow_t)win));
}

/* ──────────────────────────────────────────────────────────────────────
 * Memory Allocation
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclMemAlloc(void **ptr, size_t size) {
  return toNcclResult(flagcxMemAlloc(ptr, size));
}

ncclResult_t ncclMemFree(void *ptr) { return toNcclResult(flagcxMemFree(ptr)); }

/* ──────────────────────────────────────────────────────────────────────
 * Group Semantics
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclGroupStart() {
  groupDepth++;
  /* Defer the actual flagcxGroupStart call until the first collective,
   * when we have a communicator to pass. Reset groupComm. */
  if (groupDepth == 1) {
    groupComm = nullptr;
  }
  return ncclSuccess;
}

ncclResult_t ncclGroupEnd() {
  if (groupDepth <= 0)
    return ncclInvalidUsage;
  groupDepth--;
  if (groupDepth == 0 && groupComm != nullptr) {
    flagcxResult_t res = flagcxGroupEnd(groupComm);
    groupComm = nullptr;
    return toNcclResult(res);
  }
  return ncclSuccess;
}

/* Helper called by collective/p2p operations when inside a group.
 * Captures the first communicator and calls flagcxGroupStart. */
static ncclResult_t maybeGroupStart(flagcxComm_t comm) {
  if (groupDepth > 0 && groupComm == nullptr) {
    groupComm = comm;
    return toNcclResult(flagcxGroupStart(comm));
  }
  return ncclSuccess;
}

/* ──────────────────────────────────────────────────────────────────────
 * Collective Operations
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                           ncclDataType_t datatype, ncclRedOp_t op,
                           ncclComm_t comm, cudaStream_t stream) {
  if (comm == nullptr)
    return ncclInvalidArgument;
  flagcxDataType_t fType;
  flagcxRedOp_t fOp;
  ncclResult_t r;
  if ((r = toFlagcxDataType(datatype, &fType)) != ncclSuccess)
    return r;
  if ((r = toFlagcxRedOp(op, &fOp)) != ncclSuccess)
    return r;
  if ((r = maybeGroupStart(comm->handler->comm)) != ncclSuccess)
    return r;
  FlagcxStreamWrapper sw(stream);
  return toNcclResult(flagcxAllReduce(sendbuff, recvbuff, count, fType, fOp,
                                      comm->handler->comm, sw.stream));
}

ncclResult_t ncclBroadcast(const void *sendbuff, void *recvbuff, size_t count,
                           ncclDataType_t datatype, int root, ncclComm_t comm,
                           cudaStream_t stream) {
  if (comm == nullptr)
    return ncclInvalidArgument;
  flagcxDataType_t fType;
  ncclResult_t r;
  if ((r = toFlagcxDataType(datatype, &fType)) != ncclSuccess)
    return r;
  if ((r = maybeGroupStart(comm->handler->comm)) != ncclSuccess)
    return r;
  FlagcxStreamWrapper sw(stream);
  return toNcclResult(flagcxBroadcast(sendbuff, recvbuff, count, fType, root,
                                      comm->handler->comm, sw.stream));
}

ncclResult_t ncclReduce(const void *sendbuff, void *recvbuff, size_t count,
                        ncclDataType_t datatype, ncclRedOp_t op, int root,
                        ncclComm_t comm, cudaStream_t stream) {
  if (comm == nullptr)
    return ncclInvalidArgument;
  flagcxDataType_t fType;
  flagcxRedOp_t fOp;
  ncclResult_t r;
  if ((r = toFlagcxDataType(datatype, &fType)) != ncclSuccess)
    return r;
  if ((r = toFlagcxRedOp(op, &fOp)) != ncclSuccess)
    return r;
  if ((r = maybeGroupStart(comm->handler->comm)) != ncclSuccess)
    return r;
  FlagcxStreamWrapper sw(stream);
  return toNcclResult(flagcxReduce(sendbuff, recvbuff, count, fType, fOp, root,
                                   comm->handler->comm, sw.stream));
}

ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff,
                           size_t sendcount, ncclDataType_t datatype,
                           ncclComm_t comm, cudaStream_t stream) {
  if (comm == nullptr)
    return ncclInvalidArgument;
  flagcxDataType_t fType;
  ncclResult_t r;
  if ((r = toFlagcxDataType(datatype, &fType)) != ncclSuccess)
    return r;
  if ((r = maybeGroupStart(comm->handler->comm)) != ncclSuccess)
    return r;
  FlagcxStreamWrapper sw(stream);
  return toNcclResult(flagcxAllGather(sendbuff, recvbuff, sendcount, fType,
                                      comm->handler->comm, sw.stream));
}

ncclResult_t ncclReduceScatter(const void *sendbuff, void *recvbuff,
                               size_t recvcount, ncclDataType_t datatype,
                               ncclRedOp_t op, ncclComm_t comm,
                               cudaStream_t stream) {
  if (comm == nullptr)
    return ncclInvalidArgument;
  flagcxDataType_t fType;
  flagcxRedOp_t fOp;
  ncclResult_t r;
  if ((r = toFlagcxDataType(datatype, &fType)) != ncclSuccess)
    return r;
  if ((r = toFlagcxRedOp(op, &fOp)) != ncclSuccess)
    return r;
  if ((r = maybeGroupStart(comm->handler->comm)) != ncclSuccess)
    return r;
  FlagcxStreamWrapper sw(stream);
  return toNcclResult(flagcxReduceScatter(sendbuff, recvbuff, recvcount, fType,
                                          fOp, comm->handler->comm, sw.stream));
}

ncclResult_t ncclAlltoAll(const void *sendbuff, void *recvbuff, size_t count,
                          ncclDataType_t datatype, ncclComm_t comm,
                          cudaStream_t stream) {
  if (comm == nullptr)
    return ncclInvalidArgument;
  flagcxDataType_t fType;
  ncclResult_t r;
  if ((r = toFlagcxDataType(datatype, &fType)) != ncclSuccess)
    return r;
  if ((r = maybeGroupStart(comm->handler->comm)) != ncclSuccess)
    return r;
  FlagcxStreamWrapper sw(stream);
  return toNcclResult(flagcxAlltoAll(sendbuff, recvbuff, count, fType,
                                     comm->handler->comm, sw.stream));
}

ncclResult_t ncclGather(const void *sendbuff, void *recvbuff, size_t count,
                        ncclDataType_t datatype, int root, ncclComm_t comm,
                        cudaStream_t stream) {
  if (comm == nullptr)
    return ncclInvalidArgument;
  flagcxDataType_t fType;
  ncclResult_t r;
  if ((r = toFlagcxDataType(datatype, &fType)) != ncclSuccess)
    return r;
  if ((r = maybeGroupStart(comm->handler->comm)) != ncclSuccess)
    return r;
  FlagcxStreamWrapper sw(stream);
  return toNcclResult(flagcxGather(sendbuff, recvbuff, count, fType, root,
                                   comm->handler->comm, sw.stream));
}

ncclResult_t ncclScatter(const void *sendbuff, void *recvbuff, size_t count,
                         ncclDataType_t datatype, int root, ncclComm_t comm,
                         cudaStream_t stream) {
  if (comm == nullptr)
    return ncclInvalidArgument;
  flagcxDataType_t fType;
  ncclResult_t r;
  if ((r = toFlagcxDataType(datatype, &fType)) != ncclSuccess)
    return r;
  if ((r = maybeGroupStart(comm->handler->comm)) != ncclSuccess)
    return r;
  FlagcxStreamWrapper sw(stream);
  return toNcclResult(flagcxScatter(sendbuff, recvbuff, count, fType, root,
                                    comm->handler->comm, sw.stream));
}

/* ──────────────────────────────────────────────────────────────────────
 * Point-to-Point Operations
 * ────────────────────────────────────────────────────────────────────── */

ncclResult_t ncclSend(const void *sendbuff, size_t count,
                      ncclDataType_t datatype, int peer, ncclComm_t comm,
                      cudaStream_t stream) {
  if (comm == nullptr)
    return ncclInvalidArgument;
  flagcxDataType_t fType;
  ncclResult_t r;
  if ((r = toFlagcxDataType(datatype, &fType)) != ncclSuccess)
    return r;
  if ((r = maybeGroupStart(comm->handler->comm)) != ncclSuccess)
    return r;
  FlagcxStreamWrapper sw(stream);
  return toNcclResult(
      flagcxSend(sendbuff, count, fType, peer, comm->handler->comm, sw.stream));
}

ncclResult_t ncclRecv(void *recvbuff, size_t count, ncclDataType_t datatype,
                      int peer, ncclComm_t comm, cudaStream_t stream) {
  if (comm == nullptr)
    return ncclInvalidArgument;
  flagcxDataType_t fType;
  ncclResult_t r;
  if ((r = toFlagcxDataType(datatype, &fType)) != ncclSuccess)
    return r;
  if ((r = maybeGroupStart(comm->handler->comm)) != ncclSuccess)
    return r;
  FlagcxStreamWrapper sw(stream);
  return toNcclResult(
      flagcxRecv(recvbuff, count, fType, peer, comm->handler->comm, sw.stream));
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

ncclResult_t ncclCommShrink(ncclComm_t comm, int *excludeRanksList,
                            int excludeRanksCount, ncclComm_t *newcomm,
                            ncclConfig_t *config, int shrinkFlags) {
  return ncclInvalidUsage;
}

ncclResult_t ncclCommInitRankScalable(ncclComm_t *newcomm, int nranks,
                                      int myrank, int nId,
                                      ncclUniqueId *commIds,
                                      ncclConfig_t *config) {
  return ncclInvalidUsage;
}

ncclResult_t ncclRedOpCreatePreMulSum(ncclRedOp_t *op, void *scalar,
                                      ncclDataType_t datatype,
                                      ncclScalarResidence_t residence,
                                      ncclComm_t comm) {
  return ncclInvalidUsage;
}

ncclResult_t ncclRedOpDestroy(ncclRedOp_t op, ncclComm_t comm) {
  return ncclInvalidUsage;
}

ncclResult_t ncclGroupSimulateEnd(ncclSimInfo_t *simInfo) {
  return ncclInvalidUsage;
}

void ncclResetDebugInit() { /* Deprecated in NCCL, no-op */
}
