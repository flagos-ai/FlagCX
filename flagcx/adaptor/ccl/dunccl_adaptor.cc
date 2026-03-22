#include "du_adaptor.h"

#ifdef USE_DU_ADAPTOR

flagcxResult_t duncclAdaptorGetVersion(int *version) {
  return (flagcxResult_t)ncclGetVersion(version);
}

flagcxResult_t duncclAdaptorGetUniqueId(flagcxUniqueId_t *uniqueId) {
  if (*uniqueId == NULL) {
    flagcxCalloc(uniqueId, 1);
  }
  return (flagcxResult_t)ncclGetUniqueId((ncclUniqueId *)(*uniqueId));
}

flagcxResult_t duncclAdaptorGetStagedBuffer(const flagcxInnerComm_t comm,
                                            void **buff, size_t size,
                                            int isRecv) {
  return flagcxNotSupported;
}

const char *duncclAdaptorGetErrorString(flagcxResult_t result) {
  return ncclGetErrorString((ncclResult_t)result);
}

const char *duncclAdaptorGetLastError(flagcxInnerComm_t comm) {
  return ncclGetLastError(comm->base);
}

flagcxResult_t duncclAdaptorCommInitRank(flagcxInnerComm_t *comm, int nranks,
                                         flagcxUniqueId_t commId, int rank,
                                         bootstrapState * /*bootstrap*/) {
  if (*comm == NULL) {
    flagcxCalloc(comm, 1);
  }
  return (flagcxResult_t)ncclCommInitRank(&(*comm)->base, nranks,
                                          *(ncclUniqueId *)commId, rank);
}

flagcxResult_t duncclAdaptorCommFinalize(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclCommFinalize(comm->base);
}

flagcxResult_t duncclAdaptorCommDestroy(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclCommDestroy(comm->base);
}

flagcxResult_t duncclAdaptorCommAbort(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclCommAbort(comm->base);
}

flagcxResult_t duncclAdaptorCommResume(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclInvalidUsage;
}

flagcxResult_t duncclAdaptorCommSuspend(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclInvalidUsage;
}

flagcxResult_t duncclAdaptorCommCount(const flagcxInnerComm_t comm,
                                      int *count) {
  return (flagcxResult_t)ncclCommCount(comm->base, count);
}

flagcxResult_t duncclAdaptorCommCuDevice(const flagcxInnerComm_t comm,
                                         int *device) {
  return (flagcxResult_t)ncclCommCuDevice(comm->base, device);
}

flagcxResult_t duncclAdaptorCommUserRank(const flagcxInnerComm_t comm,
                                         int *rank) {
  return (flagcxResult_t)ncclCommUserRank(comm->base, rank);
}

flagcxResult_t duncclAdaptorCommGetAsyncError(flagcxInnerComm_t comm,
                                              flagcxResult_t *asyncError) {
  return (flagcxResult_t)ncclCommGetAsyncError(comm->base,
                                               (ncclResult_t *)asyncError);
}

// TODO: unsupported
flagcxResult_t duncclAdaptorMemAlloc(void **ptr, size_t size) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t duncclAdaptorMemFree(void *ptr) { return flagcxNotSupported; }

// TODO: unsupported
flagcxResult_t duncclAdaptorCommRegister(flagcxInnerComm_t comm, void *buff,
                                         size_t size, void **handle) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t duncclAdaptorCommDeregister(flagcxInnerComm_t comm,
                                           void *handle) {
  return flagcxNotSupported;
}

flagcxResult_t duncclAdaptorCommWindowRegister(flagcxInnerComm_t comm,
                                               void *buff, size_t size,
                                               flagcxWindow_t *win,
                                               int winFlags) {
  return flagcxNotSupported;
}

flagcxResult_t duncclAdaptorCommWindowDeregister(flagcxInnerComm_t comm,
                                                 flagcxWindow_t win) {
  return flagcxNotSupported;
}

flagcxResult_t duncclAdaptorReduce(const void *sendbuff, void *recvbuff,
                                   size_t count, flagcxDataType_t datatype,
                                   flagcxRedOp_t op, int root,
                                   flagcxInnerComm_t comm,
                                   flagcxStream_t stream) {
  return (flagcxResult_t)ncclReduce(sendbuff, recvbuff, count,
                                    (ncclDataType_t)datatype, (ncclRedOp_t)op,
                                    root, comm->base, stream->base);
}

flagcxResult_t duncclAdaptorGather(const void *sendbuff, void *recvbuff,
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

flagcxResult_t duncclAdaptorScatter(const void *sendbuff, void *recvbuff,
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

flagcxResult_t duncclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                      size_t count, flagcxDataType_t datatype,
                                      int root, flagcxInnerComm_t comm,
                                      flagcxStream_t stream) {
  return (flagcxResult_t)ncclBroadcast(sendbuff, recvbuff, count,
                                       (ncclDataType_t)datatype, root,
                                       comm->base, stream->base);
}

flagcxResult_t duncclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                      size_t count, flagcxDataType_t datatype,
                                      flagcxRedOp_t op, flagcxInnerComm_t comm,
                                      flagcxStream_t stream) {
  return (flagcxResult_t)ncclAllReduce(
      sendbuff, recvbuff, count, (ncclDataType_t)datatype, (ncclRedOp_t)op,
      comm->base, stream->base);
}

flagcxResult_t duncclAdaptorReduceScatter(const void *sendbuff, void *recvbuff,
                                          size_t recvcount,
                                          flagcxDataType_t datatype,
                                          flagcxRedOp_t op,
                                          flagcxInnerComm_t comm,
                                          flagcxStream_t stream) {
  return (flagcxResult_t)ncclReduceScatter(
      sendbuff, recvbuff, recvcount, (ncclDataType_t)datatype, (ncclRedOp_t)op,
      comm->base, stream->base);
}

flagcxResult_t duncclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                      size_t sendcount,
                                      flagcxDataType_t datatype,
                                      flagcxInnerComm_t comm,
                                      flagcxStream_t stream) {
  return (flagcxResult_t)ncclAllGather(sendbuff, recvbuff, sendcount,
                                       (ncclDataType_t)datatype, comm->base,
                                       stream->base);
}

flagcxResult_t duncclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
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

flagcxResult_t duncclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
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

flagcxResult_t duncclAdaptorSend(const void *sendbuff, size_t count,
                                 flagcxDataType_t datatype, int peer,
                                 flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
  return (flagcxResult_t)ncclSend(sendbuff, count, (ncclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

flagcxResult_t duncclAdaptorRecv(void *recvbuff, size_t count,
                                 flagcxDataType_t datatype, int peer,
                                 flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
  return (flagcxResult_t)ncclRecv(recvbuff, count, (ncclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

flagcxResult_t duncclAdaptorGroupStart() {
  return (flagcxResult_t)ncclGroupStart();
}

flagcxResult_t duncclAdaptorGroupEnd() {
  return (flagcxResult_t)ncclGroupEnd();
}

struct flagcxCCLAdaptor duncclAdaptor = {
    "DUNCCL",
    // Basic functions
    duncclAdaptorGetVersion, duncclAdaptorGetUniqueId,
    duncclAdaptorGetErrorString, duncclAdaptorGetLastError,
    duncclAdaptorGetStagedBuffer,
    // Communicator functions
    duncclAdaptorCommInitRank, duncclAdaptorCommFinalize,
    duncclAdaptorCommDestroy, duncclAdaptorCommAbort, duncclAdaptorCommResume,
    duncclAdaptorCommSuspend, duncclAdaptorCommCount, duncclAdaptorCommCuDevice,
    duncclAdaptorCommUserRank, duncclAdaptorCommGetAsyncError,
    duncclAdaptorMemAlloc, duncclAdaptorMemFree, duncclAdaptorCommRegister,
    duncclAdaptorCommDeregister,
    // Symmetric functions
    duncclAdaptorCommWindowRegister, duncclAdaptorCommWindowDeregister,
    // Communication functions
    duncclAdaptorReduce, duncclAdaptorGather, duncclAdaptorScatter,
    duncclAdaptorBroadcast, duncclAdaptorAllReduce, duncclAdaptorReduceScatter,
    duncclAdaptorAllGather, duncclAdaptorAlltoAll, duncclAdaptorAlltoAllv,
    duncclAdaptorSend, duncclAdaptorRecv,
    // Group semantics
    duncclAdaptorGroupStart, duncclAdaptorGroupEnd};

#endif // USE_DU_ADAPTOR
