#include "nvidia_adaptor.h"
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 3)
#include "nccl_device.h"
#endif
#include <cstring>

#ifdef USE_NVIDIA_ADAPTOR

#define NCCL_ADAPTOR_DEVICE_CTA_COUNT 36
#define NCCL_ADAPTOR_MAX_STAGED_BUFFER_SIZE (8*1024*1024)

static bool loaded = false;
typedef void (*ncclCollFunc_t)(ncclWindow_t send_win, ncclWindow_t recv_win,
                               void *recvbuffer, size_t count, flagcxDataType_t datatype,
                               int nRanks, ncclDevComm &devComm,
                               cudaStream_t cudaStream);
static ncclCollFunc_t localAllReduce = NULL;
static ncclCollFunc_t interleavedAllReduce = NULL;

ncclResult_t loadCollFuncSymbol(const char *path, const char *name,
                                ncclCollFunc_t *fn) {
  void *handle = flagcxOpenLib(
      path, RTLD_LAZY, [](const char *p, int err, const char *msg) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
      });

  if (!handle)
    return ncclSystemError;

  void *sym = dlsym(handle, name);
  if (!sym) {
    fprintf(stderr, "dlsym failed: %s\n", dlerror());
    return ncclSystemError;
  }

  *fn = (ncclCollFunc_t)sym;
  return ncclSuccess;
}

flagcxResult_t ncclAdaptorGetVersion(int *version) {
  return (flagcxResult_t)ncclGetVersion(version);
}

flagcxResult_t ncclAdaptorGetUniqueId(flagcxUniqueId_t *uniqueId) {
  if (*uniqueId == NULL) {
    flagcxCalloc(uniqueId, 1);
  }
  return (flagcxResult_t)ncclGetUniqueId((ncclUniqueId *)(*uniqueId));
}

const char *ncclAdaptorGetErrorString(flagcxResult_t result) {
  return ncclGetErrorString((ncclResult_t)result);
}

const char *ncclAdaptorGetLastError(flagcxInnerComm_t comm) {
  return ncclGetLastError(comm->base);
}

flagcxResult_t ncclAdaptorCommWindowRegister(const flagcxInnerComm_t comm,
                                              void *buff, size_t size,
                                              void **win, int flags) {
  return (flagcxResult_t)ncclCommWindowRegister(comm->base, buff, size,
                                                 (ncclWindow_t *)win, flags);
}

flagcxResult_t ncclAdaptorCommWindowDeregister(const flagcxInnerComm_t comm,
                                                void *win) {
  return (flagcxResult_t)ncclCommWindowDeregister(comm->base,
                                                   (ncclWindow_t)win);
}

flagcxResult_t ncclAdaptorGetStagedBuffer(const flagcxInnerComm_t comm,
                                          void **buff, size_t /*size*/,
                                          int isRecv) {
  ncclResult_t res;
  if (isRecv && comm->recvStagedBuff == NULL) {
    FLAGCXCHECK(flagcxCalloc(&comm->recvStagedBuff, 1));
    res = ncclMemAlloc(&comm->recvStagedBuff->buff, NCCL_ADAPTOR_MAX_STAGED_BUFFER_SIZE);
    if (res != ncclSuccess) {
      return (flagcxResult_t)res;
    }
    res = ncclCommWindowRegister(comm->base, comm->recvStagedBuff->buff,
                                 NCCL_ADAPTOR_MAX_STAGED_BUFFER_SIZE,
                                 &comm->recvStagedBuff->win,
                                 NCCL_WIN_COLL_SYMMETRIC);
    if (res != ncclSuccess) {
      return (flagcxResult_t)res;
    }
  } else if (!isRecv && comm->sendStagedBuff == NULL) {
    FLAGCXCHECK(flagcxCalloc(&comm->sendStagedBuff, 1));
    res = ncclMemAlloc(&comm->sendStagedBuff->buff, NCCL_ADAPTOR_MAX_STAGED_BUFFER_SIZE);
    if (res != ncclSuccess) {
      return (flagcxResult_t)res;
    }
    res = ncclCommWindowRegister(comm->base, comm->sendStagedBuff->buff,
                                 NCCL_ADAPTOR_MAX_STAGED_BUFFER_SIZE,
                                 &comm->sendStagedBuff->win,
                                 NCCL_WIN_COLL_SYMMETRIC);
    if (res != ncclSuccess) {
      return (flagcxResult_t)res;
    }
  }

  if (buff) {
    if (isRecv) {
      *buff = comm->recvStagedBuff->buff;
    } else {
      *buff = comm->sendStagedBuff->buff;
    }
  }

  return flagcxSuccess;
}

flagcxResult_t ncclAdaptorCommInitRank(flagcxInnerComm_t *comm, int nranks,
                                       flagcxUniqueId_t commId, int rank,
                                       bootstrapState * /*bootstrap*/) {
  ncclResult_t res;
  if (*comm == NULL) {
    void *p = malloc(sizeof(struct flagcxInnerComm));
    memset(p, 0, sizeof(struct flagcxInnerComm));
    (*comm) = (struct flagcxInnerComm *)p;
  }
  res = ncclCommInitRank(&(*comm)->base, nranks, *(ncclUniqueId *)commId, rank);
  if (res != ncclSuccess) {
    return (flagcxResult_t)res;
  }

  if ((*comm)->devBase == NULL) {
    flagcxCalloc(&(*comm)->devBase, 1);
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = NCCL_ADAPTOR_DEVICE_CTA_COUNT;
    reqs.lsaMultimem = true;
    using pncclDevCommCreate_t =
        ncclResult_t (*)(ncclComm_t comm, ncclDevCommRequirements *,
                         ncclDevComm *);
    void *handle = dlopen("libnccl.so", RTLD_NOW | RTLD_GLOBAL);
    auto fn =
        reinterpret_cast<pncclDevCommCreate_t>(dlsym(handle, "pncclDevCommCreate"));
    if (handle == NULL || fn == NULL) {
      return flagcxUnhandledDeviceError;
    }
    res = fn((*comm)->base, &reqs, (*comm)->devBase);
    if (res != ncclSuccess) {
      return (flagcxResult_t)res;
    }
    FLAGCXCHECK(ncclAdaptorGetStagedBuffer(*comm, NULL, 0, 1));
    FLAGCXCHECK(ncclAdaptorGetStagedBuffer(*comm, NULL, 0, 0));
  }

  return flagcxSuccess;
}

flagcxResult_t ncclAdaptorCommFinalize(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclCommFinalize(comm->base);
}

flagcxResult_t ncclAdaptorCommDestroy(flagcxInnerComm_t comm) {
  ncclResult_t res = ncclSuccess;
  if (comm->sendStagedBuff != NULL) {
    res = ncclCommWindowDeregister(comm->base, comm->sendStagedBuff->win);
    res = ncclMemFree(comm->sendStagedBuff->buff);
    free(comm->sendStagedBuff);
    comm->sendStagedBuff = NULL;
  }
  if (comm->recvStagedBuff != NULL) {
    res = ncclCommWindowDeregister(comm->base, comm->recvStagedBuff->win);
    res = ncclMemFree(comm->recvStagedBuff->buff);
    free(comm->recvStagedBuff);
    comm->recvStagedBuff = NULL;
  }
  if (comm->devBase != NULL) {
    free(comm->devBase);
    comm->devBase = NULL;
  }
  res = ncclCommDestroy(comm->base);
  free(comm);
  return (flagcxResult_t)res;
}

flagcxResult_t ncclAdaptorCommAbort(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclCommAbort(comm->base);
}

flagcxResult_t ncclAdaptorCommResume(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclInvalidUsage;
}

flagcxResult_t ncclAdaptorCommSuspend(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclInvalidUsage;
}

flagcxResult_t ncclAdaptorCommCount(const flagcxInnerComm_t comm, int *count) {
  return (flagcxResult_t)ncclCommCount(comm->base, count);
}

flagcxResult_t ncclAdaptorCommCuDevice(const flagcxInnerComm_t comm,
                                       int *device) {
  return (flagcxResult_t)ncclCommCuDevice(comm->base, device);
}

flagcxResult_t ncclAdaptorCommUserRank(const flagcxInnerComm_t comm,
                                       int *rank) {
  return (flagcxResult_t)ncclCommUserRank(comm->base, rank);
}

flagcxResult_t ncclAdaptorCommGetAsyncError(flagcxInnerComm_t comm,
                                            flagcxResult_t *asyncError) {
  return (flagcxResult_t)ncclCommGetAsyncError(comm->base,
                                               (ncclResult_t *)asyncError);
}

flagcxResult_t ncclAdaptorMemAlloc(void **ptr, size_t size) {
  return (flagcxResult_t)ncclMemAlloc(ptr, size);
}

flagcxResult_t ncclAdaptorMemFree(void *ptr) {
  return (flagcxResult_t)ncclMemFree(ptr);
}

flagcxResult_t ncclAdaptorCommRegister(const flagcxInnerComm_t comm, void *buff,
                                       size_t size, void **handle) {
  return (flagcxResult_t)ncclCommRegister(comm->base, buff, size, handle);
}

flagcxResult_t ncclAdaptorCommDeregister(const flagcxInnerComm_t comm,
                                         void *handle) {
  return (flagcxResult_t)ncclCommDeregister(comm->base, handle);
}

flagcxResult_t ncclAdaptorReduce(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 flagcxRedOp_t op, int root,
                                 flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
  return (flagcxResult_t)ncclReduce(sendbuff, recvbuff, count,
                                    (ncclDataType_t)datatype, (ncclRedOp_t)op,
                                    root, comm->base, stream->base);
}

flagcxResult_t ncclAdaptorGather(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 int root, flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
  int rank, nranks;
  ncclResult_t res = ncclSuccess;
  res = ncclCommUserRank(comm->base, &rank);
  res = ncclCommCount(comm->base, &nranks);

  size_t size = count * getFlagcxDataTypeSize(datatype);
  char *buffer = static_cast<char *>(recvbuff);

  res = ncclGroupStart();
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      res = ncclRecv(static_cast<void *>(buffer + r * size), size, ncclChar, r,
                     comm->base, stream->base);
    }
  }
  res = ncclSend(sendbuff, size, ncclChar, root, comm->base, stream->base);
  res = ncclGroupEnd();

  return (flagcxResult_t)res;
}

flagcxResult_t ncclAdaptorScatter(const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  int root, flagcxInnerComm_t comm,
                                  flagcxStream_t stream) {
  int rank, nranks;
  ncclResult_t res = ncclSuccess;
  res = ncclCommUserRank(comm->base, &rank);
  res = ncclCommCount(comm->base, &nranks);

  size_t size = count * getFlagcxDataTypeSize(datatype);
  const char *buffer = static_cast<const char *>(sendbuff);

  res = ncclGroupStart();
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      res = ncclSend(static_cast<const void *>(buffer + r * size), size,
                     ncclChar, r, comm->base, stream->base);
    }
  }
  res = ncclRecv(recvbuff, size, ncclChar, root, comm->base, stream->base);
  res = ncclGroupEnd();

  return (flagcxResult_t)res;
}

flagcxResult_t ncclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    int root, flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return (flagcxResult_t)ncclBroadcast(sendbuff, recvbuff, count,
                                       (ncclDataType_t)datatype, root,
                                       comm->base, stream->base);
}

flagcxResult_t ncclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    flagcxRedOp_t op, flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  if (!loaded) {
    const char *customAllreducePathEnv = flagcxGetEnv("FLAGCX_CUSTOM_ALLREDUCE_PATH");
    if (customAllreducePathEnv) {
      loadCollFuncSymbol(customAllreducePathEnv, "flagcxLocalAllReduce", &localAllReduce);
      loadCollFuncSymbol(customAllreducePathEnv, "flagcxInterleavedAllReduce", &interleavedAllReduce);
    }
    loaded = true;
  }
  size_t size = count * getFlagcxDataTypeSize(datatype);
  int nranks;
  ncclCommCount(comm->base, &nranks);

  if (localAllReduce == NULL || interleavedAllReduce == NULL ||
      size >= NCCL_ADAPTOR_MAX_STAGED_BUFFER_SIZE || comm->devBase == NULL) {
    return (flagcxResult_t)ncclAllReduce(
        sendbuff, recvbuff, count, (ncclDataType_t)datatype, (ncclRedOp_t)op,
        comm->base, stream->base);
  }

  cudaMemcpyAsync(comm->sendStagedBuff->buff, sendbuff, size,
                  cudaMemcpyDeviceToDevice, stream->base);
  if ((nranks <= 4 && size < 512 * 1024) ||
      (nranks <= 8 && size < 256 * 1024)) {
    localAllReduce(comm->sendStagedBuff->win, comm->recvStagedBuff->win,
                   recvbuff, count, datatype, nranks, *comm->devBase,
                   stream->base);
  } else {
    interleavedAllReduce(comm->sendStagedBuff->win, comm->recvStagedBuff->win,
                         recvbuff, count, datatype, nranks, *comm->devBase,
                         stream->base);
    cudaMemcpyAsync(recvbuff, comm->recvStagedBuff->buff, size,
                    cudaMemcpyDeviceToDevice, stream->base);
  }
  return flagcxSuccess;
}

flagcxResult_t
ncclAdaptorReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                         flagcxDataType_t datatype, flagcxRedOp_t op,
                         flagcxInnerComm_t comm, flagcxStream_t stream) {
  return (flagcxResult_t)ncclReduceScatter(
      sendbuff, recvbuff, recvcount, (ncclDataType_t)datatype, (ncclRedOp_t)op,
      comm->base, stream->base);
}

flagcxResult_t ncclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                    size_t sendcount, flagcxDataType_t datatype,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return (flagcxResult_t)ncclAllGather(sendbuff, recvbuff, sendcount,
                                       (ncclDataType_t)datatype, comm->base,
                                       stream->base);
}

flagcxResult_t ncclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                   size_t count, flagcxDataType_t datatype,
                                   flagcxInnerComm_t comm,
                                   flagcxStream_t stream) {
  int nranks;
  ncclResult_t res = ncclSuccess;
  res = ncclCommCount(comm->base, &nranks);

  size_t size = count * getFlagcxDataTypeSize(datatype);
  const char *buffer_in = static_cast<const char *>(sendbuff);
  char *buffer_out = static_cast<char *>(recvbuff);

  res = ncclGroupStart();
  for (int r = 0; r < nranks; r++) {
    res = ncclSend(static_cast<const void *>(buffer_in + r * size), size,
                   ncclChar, r, comm->base, stream->base);
    res = ncclRecv(static_cast<void *>(buffer_out + r * size), size, ncclChar,
                   r, comm->base, stream->base);
  }
  res = ncclGroupEnd();

  return (flagcxResult_t)res;
}

flagcxResult_t ncclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                    size_t *sdispls, void *recvbuff,
                                    size_t *recvcounts, size_t *rdispls,
                                    flagcxDataType_t datatype,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  int nranks;
  ncclResult_t res = ncclSuccess;
  res = ncclCommCount(comm->base, &nranks);

  size_t size = getFlagcxDataTypeSize(datatype);
  const char *buffer_in = static_cast<const char *>(sendbuff);
  char *buffer_out = static_cast<char *>(recvbuff);

  res = ncclGroupStart();
  for (int r = 0; r < nranks; r++) {
    if (flagcxCCLAdaptorNeedSendrecv(sendcounts[r])) {
      res = ncclSend(static_cast<const void *>(buffer_in + sdispls[r] * size),
                     sendcounts[r], (ncclDataType_t)datatype, r, comm->base,
                     stream->base);
    }
    if (flagcxCCLAdaptorNeedSendrecv(recvcounts[r])) {
      res = ncclRecv(static_cast<void *>(buffer_out + rdispls[r] * size),
                     recvcounts[r], (ncclDataType_t)datatype, r, comm->base,
                     stream->base);
    }
  }
  res = ncclGroupEnd();

  return (flagcxResult_t)res;
}

flagcxResult_t ncclAdaptorSend(const void *sendbuff, size_t count,
                               flagcxDataType_t datatype, int peer,
                               flagcxInnerComm_t comm, flagcxStream_t stream) {
  return (flagcxResult_t)ncclSend(sendbuff, count, (ncclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

flagcxResult_t ncclAdaptorRecv(void *recvbuff, size_t count,
                               flagcxDataType_t datatype, int peer,
                               flagcxInnerComm_t comm, flagcxStream_t stream) {
  return (flagcxResult_t)ncclRecv(recvbuff, count, (ncclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

flagcxResult_t ncclAdaptorGroupStart() {
  return (flagcxResult_t)ncclGroupStart();
}

flagcxResult_t ncclAdaptorGroupEnd() { return (flagcxResult_t)ncclGroupEnd(); }

struct flagcxCCLAdaptor ncclAdaptor = {
    "NCCL",
    // Basic functions
    ncclAdaptorGetVersion, ncclAdaptorGetUniqueId, ncclAdaptorGetErrorString,
    ncclAdaptorGetLastError,
    // Symmetric operations
    ncclAdaptorCommWindowRegister, ncclAdaptorCommWindowDeregister,
    ncclAdaptorGetStagedBuffer,
    // Communicator functions
    ncclAdaptorCommInitRank, ncclAdaptorCommFinalize, ncclAdaptorCommDestroy,
    ncclAdaptorCommAbort, ncclAdaptorCommResume, ncclAdaptorCommSuspend,
    ncclAdaptorCommCount, ncclAdaptorCommCuDevice, ncclAdaptorCommUserRank,
    ncclAdaptorCommGetAsyncError, ncclAdaptorMemAlloc, ncclAdaptorMemFree,
    ncclAdaptorCommRegister, ncclAdaptorCommDeregister,
    // Communication functions
    ncclAdaptorReduce, ncclAdaptorGather, ncclAdaptorScatter,
    ncclAdaptorBroadcast, ncclAdaptorAllReduce, ncclAdaptorReduceScatter,
    ncclAdaptorAllGather, ncclAdaptorAlltoAll, ncclAdaptorAlltoAllv,
    ncclAdaptorSend, ncclAdaptorRecv,
    // Group semantics
    ncclAdaptorGroupStart, ncclAdaptorGroupEnd};

#endif // USE_NVIDIA_ADAPTOR