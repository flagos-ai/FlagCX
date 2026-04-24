#include "nvidia_adaptor.h"

#ifdef USE_NVIDIA_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "comm.h"

static bool checkIsAllCudaP2p(ncclComm_t comm) {
  int gpuCount;
  if (cudaGetDeviceCount(&gpuCount) != cudaSuccess) {
    return false;
  }

  for (int i = 0; i < gpuCount; ++i) {
    for (int j = i + 1; j < gpuCount; ++j) {
      int canAccess = 0;
      if (cudaDeviceCanAccessPeer(&canAccess, i, j) != cudaSuccess ||
          !canAccess) {
        return false;
      }
    }
  }
  return true;
}
static bool checkNvlsSupport() {
  int driverVersion, currentDevice;
  CUdevice dev;
  int multicastSupported = 0;
  if (cudaDriverGetVersion(&driverVersion) != cudaSuccess ||
      driverVersion < 12010 || cudaGetDevice(&currentDevice) != cudaSuccess ||
      cuDeviceGet(&dev, currentDevice) != CUDA_SUCCESS ||
      cuDeviceGetAttribute(&multicastSupported,
                           CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
                           dev) != CUDA_SUCCESS) {
    return false;
  }
  return (multicastSupported != 0);
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

flagcxResult_t ncclAdaptorGetStagedBuffer(const flagcxInnerComm_t comm,
                                          void **buff, size_t /*size*/,
                                          int isRecv) {
  return flagcxNotSupported;
}

#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
static flagcxResult_t ncclDevCommCreateHelper(ncclComm_t comm,
                                              ncclDevCommRequirements *reqs,
                                              ncclDevComm *devComm) {
  using pncclDevCommCreate_t =
      flagcxCustomOpFunc_t<ncclResult_t, ncclComm_t, ncclDevCommRequirements *,
                           ncclDevComm *>;
  void *handle = dlopen("libnccl.so", RTLD_NOW | RTLD_GLOBAL);
  if (!handle) {
    return flagcxInternalError;
  }
  auto fn = reinterpret_cast<pncclDevCommCreate_t>(
      dlsym(handle, "pncclDevCommCreate"));
  if (!fn) {
    dlclose(handle);
    return flagcxInternalError;
  }
  ncclResult_t ret = fn(comm, reqs, devComm);
  dlclose(handle);
  return (flagcxResult_t)ret;
}

static flagcxResult_t ncclDevCommDestroyHelper(ncclComm_t comm,
                                               const ncclDevComm *devComm) {
  using pncclDevCommDestroy_t =
      flagcxCustomOpFunc_t<ncclResult_t, ncclComm_t, const ncclDevComm *>;
  void *handle = dlopen("libnccl.so", RTLD_NOW | RTLD_GLOBAL);
  if (!handle) {
    return flagcxInternalError;
  }
  auto fn = reinterpret_cast<pncclDevCommDestroy_t>(
      dlsym(handle, "pncclDevCommDestroy"));
  if (!fn) {
    dlclose(handle);
    return flagcxInternalError;
  }
  ncclResult_t ret = fn(comm, devComm);
  dlclose(handle);
  return (flagcxResult_t)ret;
}
#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)

const char *ncclAdaptorGetErrorString(flagcxResult_t result) {
  return ncclGetErrorString((ncclResult_t)result);
}

const char *ncclAdaptorGetLastError(flagcxInnerComm_t comm) {
  return ncclGetLastError(comm->base);
}

flagcxResult_t ncclAdaptorCommInitRank(flagcxInnerComm_t *comm, int nranks,
                                       flagcxUniqueId_t commId, int rank,
                                       bootstrapState * /*bootstrap*/) {
  if (*comm == NULL) {
    void *p = malloc(sizeof(struct flagcxInnerComm));
    memset(p, 0, sizeof(struct flagcxInnerComm));
    (*comm) = (struct flagcxInnerComm *)p;
  }
  FLAGCXCHECK((flagcxResult_t)ncclCommInitRank(&(*comm)->base, nranks,
                                               *(ncclUniqueId *)commId, rank));
  return flagcxSuccess;
}

flagcxResult_t ncclAdaptorCommFinalize(flagcxInnerComm_t comm) {
  FLAGCXCHECK((flagcxResult_t)ncclCommFinalize(comm->base));
  free(comm);
  return flagcxSuccess;
}

flagcxResult_t ncclAdaptorCommDestroy(flagcxInnerComm_t comm) {
  FLAGCXCHECK((flagcxResult_t)ncclCommDestroy(comm->base));
  free(comm);
  return flagcxSuccess;
}

flagcxResult_t ncclAdaptorCommAbort(flagcxInnerComm_t comm) {
  FLAGCXCHECK((flagcxResult_t)ncclCommAbort(comm->base));
  free(comm);
  return flagcxSuccess;
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

flagcxResult_t ncclAdaptorCommWindowRegister(flagcxInnerComm_t comm, void *buff,
                                             size_t size,
                                             flagcxSymWindow_t *win,
                                             int winFlags) {
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 27, 0)
  if (*win == NULL) {
    FLAGCXCHECK(flagcxCalloc(win, 1));
  }
  flagcxResult_t res = (flagcxResult_t)ncclCommWindowRegister(
      comm->base, buff, size, &(*win)->base, winFlags);
  if (res == flagcxSuccess) {
    (*win)->winFlags = winFlags;
  }
  return res;
#else
  return flagcxNotSupported;
#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 27, 0)
}

flagcxResult_t ncclAdaptorCommWindowDeregister(flagcxInnerComm_t comm,
                                               flagcxSymWindow_t win) {
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 27, 0)
  flagcxResult_t res = flagcxSuccess;
  res = (flagcxResult_t)ncclCommWindowDeregister(comm->base, win->base);
  free(win);
  return res;
#else
  return flagcxNotSupported;
#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 27, 0)
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
  FLAGCXCHECK((flagcxResult_t)ncclCommUserRank(comm->base, &rank));
  FLAGCXCHECK((flagcxResult_t)ncclCommCount(comm->base, &nranks));

  size_t size = count * getFlagcxDataTypeSize(datatype);
  char *buffer = static_cast<char *>(recvbuff);

  flagcxResult_t res = flagcxSuccess;
  FLAGCXCHECK((flagcxResult_t)ncclGroupStart());
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      FLAGCXCHECKGOTO(
          (flagcxResult_t)ncclRecv(static_cast<void *>(buffer + r * size), size,
                                   ncclChar, r, comm->base, stream->base),
          res, group_exit);
    }
  }
  FLAGCXCHECKGOTO((flagcxResult_t)ncclSend(sendbuff, size, ncclChar, root,
                                           comm->base, stream->base),
                  res, group_exit);
group_exit:
  FLAGCXCHECK((flagcxResult_t)ncclGroupEnd());
  return res;
}

flagcxResult_t ncclAdaptorScatter(const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  int root, flagcxInnerComm_t comm,
                                  flagcxStream_t stream) {
  int rank, nranks;
  FLAGCXCHECK((flagcxResult_t)ncclCommUserRank(comm->base, &rank));
  FLAGCXCHECK((flagcxResult_t)ncclCommCount(comm->base, &nranks));

  size_t size = count * getFlagcxDataTypeSize(datatype);
  const char *buffer = static_cast<const char *>(sendbuff);

  flagcxResult_t res = flagcxSuccess;
  FLAGCXCHECK((flagcxResult_t)ncclGroupStart());
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      FLAGCXCHECKGOTO(
          (flagcxResult_t)ncclSend(static_cast<const void *>(buffer + r * size),
                                   size, ncclChar, r, comm->base, stream->base),
          res, group_exit);
    }
  }
  FLAGCXCHECKGOTO((flagcxResult_t)ncclRecv(recvbuff, size, ncclChar, root,
                                           comm->base, stream->base),
                  res, group_exit);
group_exit:
  FLAGCXCHECK((flagcxResult_t)ncclGroupEnd());
  return res;
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
  FLAGCXCHECK((flagcxResult_t)ncclAllReduce(
      sendbuff, recvbuff, count, (ncclDataType_t)datatype, (ncclRedOp_t)op,
      comm->base, stream->base));
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
  const char *bufferIn = static_cast<const char *>(sendbuff);
  char *bufferOut = static_cast<char *>(recvbuff);

  res = ncclGroupStart();
  for (int r = 0; r < nranks; r++) {
    res = ncclSend(static_cast<const void *>(bufferIn + r * size), size,
                   ncclChar, r, comm->base, stream->base);
    res = ncclRecv(static_cast<void *>(bufferOut + r * size), size, ncclChar, r,
                   comm->base, stream->base);
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
  const char *bufferIn = static_cast<const char *>(sendbuff);
  char *bufferOut = static_cast<char *>(recvbuff);

  res = ncclGroupStart();
  for (int r = 0; r < nranks; r++) {
    if (flagcxCCLAdaptorNeedSendrecv(sendcounts[r])) {
      res = ncclSend(static_cast<const void *>(bufferIn + sdispls[r] * size),
                     sendcounts[r], (ncclDataType_t)datatype, r, comm->base,
                     stream->base);
    }
    if (flagcxCCLAdaptorNeedSendrecv(recvcounts[r])) {
      res = ncclRecv(static_cast<void *>(bufferOut + rdispls[r] * size),
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

flagcxResult_t ncclAdaptorDevCommCreate(flagcxInnerComm_t comm,
                                        const flagcxDevCommRequirements *reqs,
                                        flagcxInnerDevComm_t *devComm) {
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
  flagcxInnerDevComm_t inner =
      (flagcxInnerDevComm_t)malloc(sizeof(struct flagcxInnerDevComm));
  if (!inner)
    return flagcxSystemError;

  // Map generic requirements to NCCL-specific requirements
  ncclDevCommRequirements ncclReqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  ncclReqs.lsaBarrierCount = reqs->intraBarrierCount;
  ncclReqs.lsaMultimem = reqs->intraMulticast;
  ncclReqs.barrierCount = reqs->barrierCount;
  ncclReqs.lsaLLA2ABlockCount = reqs->intraLLA2ABlockCount;
  ncclReqs.lsaLLA2ASlotCount = reqs->intraLLA2ASlotCount;
  ncclReqs.railGinBarrierCount = reqs->interBarrierCount;
  ncclReqs.ginSignalCount = reqs->interSignalCount;
  ncclReqs.ginForceEnable = reqs->interForceEnable;
  ncclReqs.ginContextCount = reqs->interContextCount;
  ncclReqs.ginCounterCount = reqs->interCounterCount;

  flagcxResult_t ret =
      ncclDevCommCreateHelper(comm->base, &ncclReqs, &inner->base);
  if (ret != flagcxSuccess) {
    free(inner);
    return ret;
  }

  *devComm = inner;
  return flagcxSuccess;
#else
  return flagcxNotSupported;
#endif
}

flagcxResult_t ncclAdaptorDevCommDestroy(flagcxInnerComm_t comm,
                                         flagcxInnerDevComm_t devComm) {
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
  if (!devComm)
    return flagcxSuccess;
  flagcxResult_t ret = ncclDevCommDestroyHelper(comm->base, &devComm->base);
  free(devComm);
  return ret;
#else
  return flagcxNotSupported;
#endif
}

flagcxResult_t ncclAdaptorDevCommReqsInit(flagcxInnerComm_t comm,
                                          flagcxDevCommRequirements *reqs) {
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
  const char *winEnv = flagcxGetEnv("NCCL_WIN_ENABLE");
  const char *cuMemEnv = flagcxGetEnv("NCCL_CUMEM_ENABLE");
  const char *crossNicEnv = flagcxGetEnv("NCCL_CROSS_NIC");
  const char *ibDisableEnv = flagcxGetEnv("NCCL_IB_DISABLE");
  const char *ibMergeNicsEnv = flagcxGetEnv("NCCL_IB_MERGE_NICS");
  int winEnable = winEnv ? atoi(winEnv) : 1;
  int cuMemEnable = cuMemEnv ? atoi(cuMemEnv) : -2;
  int crossNic = crossNicEnv ? atoi(crossNicEnv) : 2;
  int ibDisable = ibDisableEnv ? atoi(ibDisableEnv) : 0;
  int ibMergeNics = ibMergeNicsEnv ? atoi(ibMergeNicsEnv) : 0;
  bool symmetricSupport = (crossNic > 0) && (ibDisable == 0) &&
                          (ibMergeNics == 0) && checkIsAllCudaP2p(comm->base);
  if (!winEnable || cuMemEnable == 0 || !symmetricSupport)
    return flagcxNotSupported;

  *reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs->intraBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs->intraMulticast = checkNvlsSupport();
  return flagcxSuccess;
#else
  return flagcxNotSupported;
#endif
}

struct flagcxCCLAdaptor ncclAdaptor = {
    "NCCL",
    // Basic functions
    ncclAdaptorGetVersion, ncclAdaptorGetUniqueId, ncclAdaptorGetErrorString,
    ncclAdaptorGetLastError, ncclAdaptorGetStagedBuffer,
    // Communicator functions
    ncclAdaptorCommInitRank, ncclAdaptorCommFinalize, ncclAdaptorCommDestroy,
    ncclAdaptorCommAbort, ncclAdaptorCommResume, ncclAdaptorCommSuspend,
    ncclAdaptorCommCount, ncclAdaptorCommCuDevice, ncclAdaptorCommUserRank,
    ncclAdaptorCommGetAsyncError, ncclAdaptorMemAlloc, ncclAdaptorMemFree,
    ncclAdaptorCommRegister, ncclAdaptorCommDeregister,
    // Symmetric functions
    ncclAdaptorCommWindowRegister, ncclAdaptorCommWindowDeregister,
    // Communication functions
    ncclAdaptorReduce, ncclAdaptorGather, ncclAdaptorScatter,
    ncclAdaptorBroadcast, ncclAdaptorAllReduce, ncclAdaptorReduceScatter,
    ncclAdaptorAllGather, ncclAdaptorAlltoAll, ncclAdaptorAlltoAllv,
    ncclAdaptorSend, ncclAdaptorRecv,
    // Group semantics
    ncclAdaptorGroupStart, ncclAdaptorGroupEnd,
    // Device API
    ncclAdaptorDevCommReqsInit, ncclAdaptorDevCommCreate,
    ncclAdaptorDevCommDestroy};

#endif // USE_NVIDIA_ADAPTOR
