#include "iluvatar_corex_adaptor.h"

#ifdef USE_ILUVATAR_COREX_ADAPTOR

flagcxResult_t ixncclAdaptorGetVersion(int *version) {
  return (flagcxResult_t)ncclGetVersion(version);
}

flagcxResult_t ixncclAdaptorGetUniqueId(flagcxUniqueId_t *uniqueId) {
  if (*uniqueId == NULL) {
    flagcxCalloc(uniqueId, 1);
  }
  return (flagcxResult_t)ncclGetUniqueId((ncclUniqueId *)(*uniqueId));
}

flagcxResult_t ixncclAdaptorGetStagedBuffer(const flagcxInnerComm_t comm,
                                            void **buff, size_t size,
                                            int isRecv) {
  return flagcxNotSupported;
}

const char *ixncclAdaptorGetErrorString(flagcxResult_t result) {
  return ncclGetErrorString((ncclResult_t)result);
}

const char *ixncclAdaptorGetLastError(flagcxInnerComm_t comm) {
  return ncclGetLastError(comm->base);
}

flagcxResult_t ixncclAdaptorCommInitRank(flagcxInnerComm_t *comm, int nranks,
                                         flagcxUniqueId_t commId, int rank,
                                         bootstrapState * /*bootstrap*/) {
  if (*comm == NULL) {
    flagcxCalloc(comm, 1);
  }
  return (flagcxResult_t)ncclCommInitRank(&(*comm)->base, nranks,
                                          *(ncclUniqueId *)commId, rank);
}

flagcxResult_t ixncclAdaptorCommFinalize(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclCommFinalize(comm->base);
}

flagcxResult_t ixncclAdaptorCommDestroy(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclCommDestroy(comm->base);
}

flagcxResult_t ixncclAdaptorCommAbort(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclCommAbort(comm->base);
}

flagcxResult_t ixncclAdaptorCommResume(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclInvalidUsage;
}

flagcxResult_t ixncclAdaptorCommSuspend(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclInvalidUsage;
}

flagcxResult_t ixncclAdaptorCommCount(const flagcxInnerComm_t comm,
                                      int *count) {
  return (flagcxResult_t)ncclCommCount(comm->base, count);
}

flagcxResult_t ixncclAdaptorCommCuDevice(const flagcxInnerComm_t comm,
                                         int *device) {
  return (flagcxResult_t)ncclCommCuDevice(comm->base, device);
}

flagcxResult_t ixncclAdaptorCommUserRank(const flagcxInnerComm_t comm,
                                         int *rank) {
  return (flagcxResult_t)ncclCommUserRank(comm->base, rank);
}

flagcxResult_t ixncclAdaptorCommGetAsyncError(flagcxInnerComm_t comm,
                                              flagcxResult_t *asyncError) {
  return (flagcxResult_t)ncclCommGetAsyncError(comm->base,
                                               (ncclResult_t *)asyncError);
}

// TODO: unsupported
flagcxResult_t ixncclAdaptorMemAlloc(void **ptr, size_t size) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t ixncclAdaptorMemFree(void *ptr) { return flagcxNotSupported; }

// TODO: unsupported
flagcxResult_t ixncclAdaptorCommRegister(flagcxInnerComm_t comm, void *buff,
                                         size_t size, void **handle) {
  return flagcxNotSupported;
}

flagcxResult_t ixncclAdaptorCommWindowRegister(flagcxInnerComm_t comm,
                                               void *buff, size_t size,
                                               flagcxWindow_t *win,
                                               int winFlags) {
  return flagcxNotSupported;
}

flagcxResult_t ixncclAdaptorCommWindowDeregister(flagcxInnerComm_t comm,
                                                 flagcxWindow_t win) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t ixncclAdaptorCommDeregister(flagcxInnerComm_t comm,
                                           void *handle) {
  return flagcxNotSupported;
}

flagcxResult_t ixncclAdaptorReduce(const void *sendbuff, void *recvbuff,
                                   size_t count, flagcxDataType_t datatype,
                                   flagcxRedOp_t op, int root,
                                   flagcxInnerComm_t comm,
                                   flagcxStream_t stream) {
  return (flagcxResult_t)ncclReduce(sendbuff, recvbuff, count,
                                    (ncclDataType_t)datatype, (ncclRedOp_t)op,
                                    root, comm->base, stream->base);
}

flagcxResult_t ixncclAdaptorGather(const void *sendbuff, void *recvbuff,
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

flagcxResult_t ixncclAdaptorScatter(const void *sendbuff, void *recvbuff,
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

flagcxResult_t ixncclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                      size_t count, flagcxDataType_t datatype,
                                      int root, flagcxInnerComm_t comm,
                                      flagcxStream_t stream) {
  return (flagcxResult_t)ncclBroadcast(sendbuff, recvbuff, count,
                                       (ncclDataType_t)datatype, root,
                                       comm->base, stream->base);
}

flagcxResult_t ixncclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                      size_t count, flagcxDataType_t datatype,
                                      flagcxRedOp_t op, flagcxInnerComm_t comm,
                                      flagcxStream_t stream) {
  return (flagcxResult_t)ncclAllReduce(
      sendbuff, recvbuff, count, (ncclDataType_t)datatype, (ncclRedOp_t)op,
      comm->base, stream->base);
}

flagcxResult_t ixncclAdaptorReduceScatter(const void *sendbuff, void *recvbuff,
                                          size_t recvcount,
                                          flagcxDataType_t datatype,
                                          flagcxRedOp_t op,
                                          flagcxInnerComm_t comm,
                                          flagcxStream_t stream) {
  return (flagcxResult_t)ncclReduceScatter(
      sendbuff, recvbuff, recvcount, (ncclDataType_t)datatype, (ncclRedOp_t)op,
      comm->base, stream->base);
}

flagcxResult_t ixncclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                      size_t sendcount,
                                      flagcxDataType_t datatype,
                                      flagcxInnerComm_t comm,
                                      flagcxStream_t stream) {
  return (flagcxResult_t)ncclAllGather(sendbuff, recvbuff, sendcount,
                                       (ncclDataType_t)datatype, comm->base,
                                       stream->base);
}

flagcxResult_t ixncclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                     size_t count, flagcxDataType_t datatype,
                                     flagcxInnerComm_t comm,
                                     flagcxStream_t stream) {
  int rank, nranks;
  ncclResult_t res = ncclSuccess;
  res = ncclCommUserRank(comm->base, &rank);
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

flagcxResult_t ixncclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
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

flagcxResult_t ixncclAdaptorSend(const void *sendbuff, size_t count,
                                 flagcxDataType_t datatype, int peer,
                                 flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
  return (flagcxResult_t)ncclSend(sendbuff, count, (ncclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

flagcxResult_t ixncclAdaptorRecv(void *recvbuff, size_t count,
                                 flagcxDataType_t datatype, int peer,
                                 flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
  return (flagcxResult_t)ncclRecv(recvbuff, count, (ncclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

flagcxResult_t ixncclAdaptorGroupStart() {
  return (flagcxResult_t)ncclGroupStart();
}

flagcxResult_t ixncclAdaptorGroupEnd() {
  return (flagcxResult_t)ncclGroupEnd();
}

struct flagcxCCLAdaptor ixncclAdaptor = {
    "IXNCCL",
    // Basic functions
    ixncclAdaptorGetVersion, ixncclAdaptorGetUniqueId,
    ixncclAdaptorGetErrorString, ixncclAdaptorGetLastError,
    ixncclAdaptorGetStagedBuffer,
    // Communicator functions
    ixncclAdaptorCommInitRank, ixncclAdaptorCommFinalize,
    ixncclAdaptorCommDestroy, ixncclAdaptorCommAbort, ixncclAdaptorCommResume,
    ixncclAdaptorCommSuspend, ixncclAdaptorCommCount, ixncclAdaptorCommCuDevice,
    ixncclAdaptorCommUserRank, ixncclAdaptorCommGetAsyncError,
    ixncclAdaptorMemAlloc, ixncclAdaptorMemFree, ixncclAdaptorCommRegister,
    ixncclAdaptorCommDeregister,
    // Symmetric functions
    ixncclAdaptorCommWindowRegister, ixncclAdaptorCommWindowDeregister,
    // Communication functions
    ixncclAdaptorReduce, ixncclAdaptorGather, ixncclAdaptorScatter,
    ixncclAdaptorBroadcast, ixncclAdaptorAllReduce, ixncclAdaptorReduceScatter,
    ixncclAdaptorAllGather, ixncclAdaptorAlltoAll, ixncclAdaptorAlltoAllv,
    ixncclAdaptorSend, ixncclAdaptorRecv,
    // Group semantics
    ixncclAdaptorGroupStart, ixncclAdaptorGroupEnd};

#endif // USE_ILUVATAR_COREX_ADAPTOR
