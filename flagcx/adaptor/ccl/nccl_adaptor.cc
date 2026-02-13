#include "nvidia_adaptor.h"

#ifdef USE_NVIDIA_ADAPTOR

static bool checkIsAllCudaP2p(ncclComm_t comm, int nranks) {
  int gpuCount;
  if (cudaGetDeviceCount(&gpuCount) != cudaSuccess) {
    return false;
  }

  int checkCount = (gpuCount < nranks) ? gpuCount : nranks;
  for (int i = 0; i < checkCount; ++i) {
    for (int j = i + 1; j < checkCount; ++j) {
      int canAccess = 0;
      if (cudaDeviceCanAccessPeer(&canAccess, i, j) != cudaSuccess || !canAccess) {
        return false;
      }
    }
  }
  return true;
}
static bool checkNvlsSupport(int nranks) {
  if (nranks < 2) {
    return false;
  }
  int driverVersion, currentDevice;
  CUdevice dev;
  int multicastSupported = 0;
  if (cudaDriverGetVersion(&driverVersion) != cudaSuccess || driverVersion < 12010 ||
      cudaGetDevice(&currentDevice) != cudaSuccess ||
      cuDeviceGet(&dev, currentDevice) != CUDA_SUCCESS ||
      cuDeviceGetAttribute(&multicastSupported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev) != CUDA_SUCCESS) {
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
  flagcxResult_t res = flagcxSuccess;
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
  if (isRecv && comm->recvStagedBuff == NULL) {
    FLAGCXCHECK(flagcxCalloc(&comm->recvStagedBuff, 1));
    res = (flagcxResult_t)ncclMemAlloc(&comm->recvStagedBuff->buff,
                                       NCCL_ADAPTOR_MAX_STAGED_BUFFER_SIZE);
    if (res != flagcxSuccess) {
      free(comm->recvStagedBuff);
      comm->recvStagedBuff = NULL;
      return res;
    }
    res = (flagcxResult_t)ncclCommWindowRegister(
        comm->base, comm->recvStagedBuff->buff,
        NCCL_ADAPTOR_MAX_STAGED_BUFFER_SIZE, &comm->recvStagedBuff->win,
        NCCL_WIN_COLL_SYMMETRIC);
    if (res != flagcxSuccess) {
      (void)ncclMemFree(comm->recvStagedBuff->buff);
      free(comm->recvStagedBuff);
      comm->recvStagedBuff = NULL;
      return res;
    }
  } else if (!isRecv && comm->sendStagedBuff == NULL) {
    FLAGCXCHECK(flagcxCalloc(&comm->sendStagedBuff, 1));
    res = (flagcxResult_t)ncclMemAlloc(&comm->sendStagedBuff->buff,
                                       NCCL_ADAPTOR_MAX_STAGED_BUFFER_SIZE);
    if (res != flagcxSuccess) {
      free(comm->sendStagedBuff);
      comm->sendStagedBuff = NULL;
      return res;
    }
    res = (flagcxResult_t)ncclCommWindowRegister(
        comm->base, comm->sendStagedBuff->buff,
        NCCL_ADAPTOR_MAX_STAGED_BUFFER_SIZE, &comm->sendStagedBuff->win,
        NCCL_WIN_COLL_SYMMETRIC);
    if (res != flagcxSuccess) {
      (void)ncclMemFree(comm->sendStagedBuff->buff);
      free(comm->sendStagedBuff);
      comm->sendStagedBuff = NULL;
      return res;
    }
  }
  if (buff) {
    if (isRecv) {
      *buff = comm->recvStagedBuff->buff;
    } else {
      *buff = comm->sendStagedBuff->buff;
    }
  }
#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
  return res;
}

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

#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
  if ((*comm)->devBase == NULL) {
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
    bool symmetricSupport = (crossNic > 0) && (ibDisable == 0) && (ibMergeNics == 0) && checkIsAllCudaP2p((*comm)->base, nranks);
    if (winEnable && cuMemEnable != 0 && symmetricSupport) {
      FLAGCXCHECK(flagcxCalloc(&(*comm)->devBase, 1));
      ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
      reqs.lsaBarrierCount = NCCL_ADAPTOR_DEVICE_CTA_COUNT;
      reqs.lsaMultimem = checkNvlsSupport(nranks);
      reqs.railGinBarrierCount = NCCL_ADAPTOR_DEVICE_CTA_COUNT;
      reqs.ginSignalCount = 1;
      using pncclDevCommCreate_t = ncclResult_t (*)(
          ncclComm_t comm, ncclDevCommRequirements *, ncclDevComm *);
      void *handle = dlopen("libnccl.so", RTLD_NOW | RTLD_GLOBAL);
      if (handle) {
        auto fn = reinterpret_cast<pncclDevCommCreate_t>(
            dlsym(handle, "pncclDevCommCreate"));
        if (fn) {
          FLAGCXCHECK((flagcxResult_t)fn((*comm)->base, &reqs, (*comm)->devBase));
        }
        dlclose(handle);
      }
      if ((*comm)->devBase == NULL) {
        WARN("ncclDevComm is not initialized succefully");
      }
    }
  }
#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
  FLAGCXCHECK(ncclAdaptorGetStagedBuffer(*comm, NULL, 0, 1));
  FLAGCXCHECK(ncclAdaptorGetStagedBuffer(*comm, NULL, 0, 0));
  return flagcxSuccess;
}

flagcxResult_t ncclAdaptorCommFinalize(flagcxInnerComm_t comm) {
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
  if (comm->sendStagedBuff != NULL) {
    FLAGCXCHECK((flagcxResult_t)ncclCommWindowDeregister(
        comm->base, comm->sendStagedBuff->win));
    FLAGCXCHECK((flagcxResult_t)ncclMemFree(comm->sendStagedBuff->buff));
    free(comm->sendStagedBuff);
  }
  if (comm->recvStagedBuff != NULL) {
    FLAGCXCHECK((flagcxResult_t)ncclCommWindowDeregister(
        comm->base, comm->recvStagedBuff->win));
    FLAGCXCHECK((flagcxResult_t)ncclMemFree(comm->recvStagedBuff->buff));
    free(comm->recvStagedBuff);
  }
  if (comm->devBase != NULL) {
    free(comm->devBase);
  }
#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
  FLAGCXCHECK((flagcxResult_t)ncclCommFinalize(comm->base));
  free(comm);
  return flagcxSuccess;
}

flagcxResult_t ncclAdaptorCommDestroy(flagcxInnerComm_t comm) {
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
  if (comm->sendStagedBuff != NULL) {
    FLAGCXCHECK((flagcxResult_t)ncclCommWindowDeregister(
        comm->base, comm->sendStagedBuff->win));
    FLAGCXCHECK((flagcxResult_t)ncclMemFree(comm->sendStagedBuff->buff));
    free(comm->sendStagedBuff);
  }
  if (comm->recvStagedBuff != NULL) {
    FLAGCXCHECK((flagcxResult_t)ncclCommWindowDeregister(
        comm->base, comm->recvStagedBuff->win));
    FLAGCXCHECK((flagcxResult_t)ncclMemFree(comm->recvStagedBuff->buff));
    free(comm->recvStagedBuff);
  }
  if (comm->devBase != NULL) {
    free(comm->devBase);
  }
#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
  FLAGCXCHECK((flagcxResult_t)ncclCommDestroy(comm->base));
  free(comm);
  return flagcxSuccess;
}

flagcxResult_t ncclAdaptorCommAbort(flagcxInnerComm_t comm) {
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
  if (comm->sendStagedBuff != NULL) {
    FLAGCXCHECK((flagcxResult_t)ncclCommWindowDeregister(
        comm->base, comm->sendStagedBuff->win));
    FLAGCXCHECK((flagcxResult_t)ncclMemFree(comm->sendStagedBuff->buff));
    free(comm->sendStagedBuff);
  }
  if (comm->recvStagedBuff != NULL) {
    FLAGCXCHECK((flagcxResult_t)ncclCommWindowDeregister(
        comm->base, comm->recvStagedBuff->win));
    FLAGCXCHECK((flagcxResult_t)ncclMemFree(comm->recvStagedBuff->buff));
    free(comm->recvStagedBuff);
  }
  if (comm->devBase != NULL) {
    free(comm->devBase);
  }
#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
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
                                             size_t size, flagcxWindow_t *win,
                                             int winFlags) {
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 27, 0)
  if (*win == NULL) {
    FLAGCXCHECK(flagcxCalloc(win, 1));
  }
  return (flagcxResult_t)ncclCommWindowRegister(comm->base, buff, size,
                                                &(*win)->base, winFlags);
#else
  return flagcxNotSupported;
#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 27, 0)
}

flagcxResult_t ncclAdaptorCommWindowDeregister(flagcxInnerComm_t comm,
                                               flagcxWindow_t win) {
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
#if defined(COMPILE_KERNEL_HOST) &&                                            \
    (NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)) &&                            \
    !defined(NVCC_GENCODE_MULTICAST_UNSUPPORTED)
  size_t size = count * getFlagcxDataTypeSize(datatype);
  int nranks;
  FLAGCXCHECK((flagcxResult_t)ncclCommCount(comm->base, &nranks));
  if (size >= NCCL_ADAPTOR_MAX_STAGED_BUFFER_SIZE) {
    FLAGCXCHECK((flagcxResult_t)ncclAllReduce(
        sendbuff, recvbuff, count, (ncclDataType_t)datatype, (ncclRedOp_t)op,
        comm->base, stream->base));
  } else {
    DEVCHECK(cudaMemcpyAsync(comm->sendStagedBuff->buff, sendbuff, size,
                             cudaMemcpyDeviceToDevice, stream->base));
    if ((nranks <= 4 && size < 512 * 1024) ||
        (nranks <= 8 && size < 256 * 1024)) {
      FLAGCXCHECK((flagcxResult_t)ncclAdaptorLocalAllReduce(
          sendbuff, recvbuff, comm->sendStagedBuff->win,
          comm->recvStagedBuff->win, count, (ncclDataType_t)datatype,
          (ncclRedOp_t)op, *comm->devBase, stream->base));
    } else {
      FLAGCXCHECK((flagcxResult_t)ncclAdaptorInterleavedAllReduce(
          sendbuff, recvbuff, comm->sendStagedBuff->win,
          comm->recvStagedBuff->win, count, (ncclDataType_t)datatype,
          (ncclRedOp_t)op, *comm->devBase, stream->base));
      DEVCHECK(cudaMemcpyAsync(recvbuff, comm->recvStagedBuff->buff, size,
                               cudaMemcpyDeviceToDevice, stream->base));
    }
  }
#else
  FLAGCXCHECK((flagcxResult_t)ncclAllReduce(
      sendbuff, recvbuff, count, (ncclDataType_t)datatype, (ncclRedOp_t)op,
      comm->base, stream->base));
#endif
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
    ncclAdaptorGroupStart, ncclAdaptorGroupEnd};

#endif // USE_NVIDIA_ADAPTOR