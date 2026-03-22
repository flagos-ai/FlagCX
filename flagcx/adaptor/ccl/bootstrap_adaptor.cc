#include "bootstrap_adaptor.h"
#include "bootstrap.h"

#ifdef USE_BOOTSTRAP_ADAPTOR

static int groupDepth = 0;
static std::vector<stagedBuffer_t> sendStagedBufferList;
static std::vector<stagedBuffer_t> recvStagedBufferList;

// TODO: unsupported
flagcxResult_t bootstrapAdaptorGetVersion(int *version) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t bootstrapAdaptorGetUniqueId(flagcxUniqueId_t *uniqueId) {
  return flagcxNotSupported;
}

flagcxResult_t bootstrapAdaptorGetStagedBuffer(const flagcxInnerComm_t comm,
                                               void **buff, size_t size,
                                               int isRecv) {
  stagedBuffer *sbuff = NULL;
  if (isRecv) {
    for (auto it = recvStagedBufferList.begin();
         it != recvStagedBufferList.end(); it++) {
      if ((*it)->size - (*it)->offset >= size) {
        sbuff = (*it);
        break;
      }
    }
  } else {
    for (auto it = sendStagedBufferList.begin();
         it != sendStagedBufferList.end(); it++) {
      if ((*it)->size - (*it)->offset >= size) {
        sbuff = (*it);
        break;
      }
    }
  }
  if (sbuff == NULL) {
    FLAGCXCHECK(flagcxCalloc(&sbuff, 1));
    sbuff->offset = 0;
    int newSize = BOOTSTRAP_ADAPTOR_MAX_STAGED_BUFFER_SIZE;
    while (newSize < size) {
      newSize *= 2;
    }
    sbuff->buffer = malloc(newSize);
    if (sbuff->buffer == NULL) {
      return flagcxSystemError;
    }
    sbuff->size = newSize;
    if (isRecv) {
      recvStagedBufferList.push_back(sbuff);
    } else {
      sendStagedBufferList.push_back(sbuff);
    }
  }
  *buff = (void *)((char *)sbuff->buffer + sbuff->offset);
  return flagcxSuccess;
}

// TODO: unsupported
const char *bootstrapAdaptorGetErrorString(flagcxResult_t result) {
  return "Not Implemented";
}

// TODO: unsupported
const char *bootstrapAdaptorGetLastError(flagcxInnerComm_t comm) {
  return "Not Implemented";
}

flagcxResult_t bootstrapAdaptorCommInitRank(flagcxInnerComm_t *comm, int nranks,
                                            flagcxUniqueId_t /*commId*/,
                                            int rank,
                                            bootstrapState *bootstrap) {
  if (*comm == NULL) {
    FLAGCXCHECK(flagcxCalloc(comm, 1));
  }
  (*comm)->base = bootstrap;

  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorCommFinalize(flagcxInnerComm_t comm) {
  for (size_t i = sendStagedBufferList.size() - 1; i >= 0; --i) {
    stagedBuffer *buff = sendStagedBufferList[i];
    free(buff->buffer);
    free(buff);
  }
  for (size_t i = recvStagedBufferList.size() - 1; i >= 0; --i) {
    stagedBuffer *buff = recvStagedBufferList[i];
    free(buff->buffer);
    free(buff);
  }
  sendStagedBufferList.clear();
  recvStagedBufferList.clear();
  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorCommDestroy(flagcxInnerComm_t comm) {
  for (size_t i = sendStagedBufferList.size() - 1; i >= 0; --i) {
    stagedBuffer *buff = sendStagedBufferList[i];
    free(buff->buffer);
    free(buff);
  }
  for (size_t i = recvStagedBufferList.size() - 1; i >= 0; --i) {
    stagedBuffer *buff = recvStagedBufferList[i];
    free(buff->buffer);
    free(buff);
  }
  sendStagedBufferList.clear();
  recvStagedBufferList.clear();
  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorCommAbort(flagcxInnerComm_t comm) {
  for (size_t i = sendStagedBufferList.size() - 1; i >= 0; --i) {
    stagedBuffer *buff = sendStagedBufferList[i];
    free(buff->buffer);
    free(buff);
  }
  for (size_t i = recvStagedBufferList.size() - 1; i >= 0; --i) {
    stagedBuffer *buff = recvStagedBufferList[i];
    free(buff->buffer);
    free(buff);
  }
  sendStagedBufferList.clear();
  recvStagedBufferList.clear();
  return flagcxSuccess;
}

// TODO: unsupported
flagcxResult_t bootstrapAdaptorCommResume(flagcxInnerComm_t comm) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t bootstrapAdaptorCommSuspend(flagcxInnerComm_t comm) {
  return flagcxNotSupported;
}

flagcxResult_t bootstrapAdaptorCommCount(const flagcxInnerComm_t comm,
                                         int *count) {
  *count = comm->base->nranks;
  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorCommCuDevice(const flagcxInnerComm_t comm,
                                            int *device) {
  device = NULL;
  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorCommUserRank(const flagcxInnerComm_t comm,
                                            int *rank) {
  *rank = comm->base->rank;
  return flagcxSuccess;
}

// TODO: unsupported
flagcxResult_t bootstrapAdaptorCommGetAsyncError(flagcxInnerComm_t comm,
                                                 flagcxResult_t *asyncError) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t bootstrapAdaptorMemAlloc(void **ptr, size_t size) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t bootstrapAdaptorMemFree(void *ptr) { return flagcxNotSupported; }

// TODO: unsupported
flagcxResult_t bootstrapAdaptorCommRegister(flagcxInnerComm_t comm, void *buff,
                                            size_t size, void **handle) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t bootstrapAdaptorCommDeregister(flagcxInnerComm_t comm,
                                              void *handle) {
  return flagcxNotSupported;
}

flagcxResult_t bootstrapAdaptorCommWindowRegister(flagcxInnerComm_t comm,
                                                  void *buff, size_t size,
                                                  flagcxWindow_t *win,
                                                  int winFlags) {
  return flagcxNotSupported;
}

flagcxResult_t bootstrapAdaptorCommWindowDeregister(flagcxInnerComm_t comm,
                                                    flagcxWindow_t win) {
  return flagcxNotSupported;
}

flagcxResult_t bootstrapAdaptorGather(const void *sendbuff, void *recvbuff,
                                      size_t count, flagcxDataType_t datatype,
                                      int root, flagcxInnerComm_t comm,
                                      flagcxStream_t /*stream*/) {
  FLAGCXCHECK(
      GatherBootstrap(comm->base, sendbuff, recvbuff, count, datatype, root));
  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorScatter(const void *sendbuff, void *recvbuff,
                                       size_t count, flagcxDataType_t datatype,
                                       int root, flagcxInnerComm_t comm,
                                       flagcxStream_t /*stream*/) {
  FLAGCXCHECK(
      ScatterBootstrap(comm->base, sendbuff, recvbuff, count, datatype, root));
  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                         size_t count,
                                         flagcxDataType_t datatype, int root,
                                         flagcxInnerComm_t comm,
                                         flagcxStream_t /*stream*/) {
  FLAGCXCHECK(BroadcastBootstrap(comm->base, sendbuff, recvbuff, count,
                                 datatype, root));
  return flagcxSuccess;
}

flagcxResult_t
bootstrapAdaptorAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                          flagcxDataType_t datatype, flagcxRedOp_t op,
                          flagcxInnerComm_t comm, flagcxStream_t /*stream*/) {
  FLAGCXCHECK(
      AllReduceBootstrap(comm->base, sendbuff, recvbuff, count, datatype, op));
  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorReduce(const void *sendbuff, void *recvbuff,
                                      size_t count, flagcxDataType_t datatype,
                                      flagcxRedOp_t op, int root,
                                      flagcxInnerComm_t comm,
                                      flagcxStream_t /*stream*/) {
  FLAGCXCHECK(ReduceBootstrap(comm->base, sendbuff, recvbuff, count, datatype,
                              op, root));
  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorReduceScatter(const void *sendbuff,
                                             void *recvbuff, size_t recvcount,
                                             flagcxDataType_t datatype,
                                             flagcxRedOp_t op,
                                             flagcxInnerComm_t comm,
                                             flagcxStream_t /*stream*/) {
  FLAGCXCHECK(ReduceScatterBootstrap(comm->base, sendbuff, recvbuff, recvcount,
                                     datatype, op));
  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                         size_t sendcount,
                                         flagcxDataType_t datatype,
                                         flagcxInnerComm_t comm,
                                         flagcxStream_t /*stream*/) {
  FLAGCXCHECK(
      AllGatherBootstrap(comm->base, sendbuff, recvbuff, sendcount, datatype));
  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                        size_t count, flagcxDataType_t datatype,
                                        flagcxInnerComm_t comm,
                                        flagcxStream_t /*stream*/) {
  FLAGCXCHECK(
      AlltoAllBootstrap(comm->base, sendbuff, recvbuff, count, datatype));
  return flagcxSuccess;
}

flagcxResult_t
bootstrapAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                          size_t *sdispls, void *recvbuff, size_t *recvcounts,
                          size_t *rdispls, flagcxDataType_t datatype,
                          flagcxInnerComm_t comm, flagcxStream_t /*stream*/) {
  FLAGCXCHECK(AlltoAllvBootstrap(comm->base, sendbuff, sendcounts, sdispls,
                                 recvbuff, recvcounts, rdispls, datatype));
  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorSend(const void *sendbuff, size_t count,
                                    flagcxDataType_t datatype, int peer,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t /*stream*/) {
  // TODO(MC952-arch): implement out-of-order sends
  size_t size = count * getFlagcxDataTypeSize(datatype);
  FLAGCXCHECK(bootstrapSend(comm->base, peer, BOOTSTRAP_ADAPTOR_SEND_RECV_TAG,
                            (void *)sendbuff, size));
  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorRecv(void *recvbuff, size_t count,
                                    flagcxDataType_t datatype, int peer,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t /*stream*/) {
  // TODO(MC952-arch): implement out-of-order recvs
  size_t size = count * getFlagcxDataTypeSize(datatype);
  FLAGCXCHECK(bootstrapRecv(comm->base, peer, BOOTSTRAP_ADAPTOR_SEND_RECV_TAG,
                            recvbuff, size));
  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorGroupStart() {
  groupDepth++;
  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorGroupEnd() {
  groupDepth--;
  if (groupDepth == 0) {
    for (size_t i = 0; i < sendStagedBufferList.size(); ++i) {
      stagedBuffer *buff = sendStagedBufferList[i];
      buff->offset = 0;
    }
    for (size_t i = 0; i < recvStagedBufferList.size(); ++i) {
      stagedBuffer *buff = recvStagedBufferList[i];
      buff->offset = 0;
    }
  }
  return flagcxSuccess;
}

struct flagcxCCLAdaptor bootstrapAdaptor = {
    "BOOTSTRAP",
    // Basic functions
    bootstrapAdaptorGetVersion, bootstrapAdaptorGetUniqueId,
    bootstrapAdaptorGetErrorString, bootstrapAdaptorGetLastError,
    bootstrapAdaptorGetStagedBuffer,
    // Communicator functions
    bootstrapAdaptorCommInitRank, bootstrapAdaptorCommFinalize,
    bootstrapAdaptorCommDestroy, bootstrapAdaptorCommAbort,
    bootstrapAdaptorCommResume, bootstrapAdaptorCommSuspend,
    bootstrapAdaptorCommCount, bootstrapAdaptorCommCuDevice,
    bootstrapAdaptorCommUserRank, bootstrapAdaptorCommGetAsyncError,
    bootstrapAdaptorMemAlloc, bootstrapAdaptorMemFree,
    bootstrapAdaptorCommRegister, bootstrapAdaptorCommDeregister,
    // Symmetric functions
    bootstrapAdaptorCommWindowRegister, bootstrapAdaptorCommWindowDeregister,
    // Communication functions
    bootstrapAdaptorReduce, bootstrapAdaptorGather, bootstrapAdaptorScatter,
    bootstrapAdaptorBroadcast, bootstrapAdaptorAllReduce,
    bootstrapAdaptorReduceScatter, bootstrapAdaptorAllGather,
    bootstrapAdaptorAlltoAll, bootstrapAdaptorAlltoAllv, bootstrapAdaptorSend,
    bootstrapAdaptorRecv,
    // Group semantics
    bootstrapAdaptorGroupStart, bootstrapAdaptorGroupEnd};

#endif // USE_BOOTSTRAP_ADAPTOR