#include "amd_adaptor.h"

#ifdef USE_AMD_ADAPTOR

flagcxResult_t rcclAdaptorGetVersion(int *version) {
  return (flagcxResult_t)ncclGetVersion(version);
}

flagcxResult_t rcclAdaptorGetUniqueId(flagcxUniqueId_t *uniqueId) {
  if (*uniqueId == NULL) {
    flagcxCalloc(uniqueId, 1);
  }
  return (flagcxResult_t)ncclGetUniqueId((ncclUniqueId *)(*uniqueId));
}

flagcxResult_t rcclAdaptorGetStagedBuffer(const flagcxInnerComm_t comm,
                                          void **buff, size_t size,
                                          int isRecv) {
  return flagcxNotSupported;
}

const char *rcclAdaptorGetErrorString(flagcxResult_t result) {
  return ncclGetErrorString((ncclResult_t)result);
}

const char *rcclAdaptorGetLastError(flagcxInnerComm_t comm) {
  return ncclGetLastError(comm->base);
}

flagcxResult_t rcclAdaptorCommInitRank(flagcxInnerComm_t *comm, int nranks,
                                       flagcxUniqueId_t commId, int rank,
                                       bootstrapState * /*bootstrap*/) {
  if (*comm == NULL) {
    flagcxCalloc(comm, 1);
  }
  return (flagcxResult_t)ncclCommInitRank(&(*comm)->base, nranks,
                                          *(ncclUniqueId *)commId, rank);
}

flagcxResult_t rcclAdaptorCommFinalize(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclCommFinalize(comm->base);
}

flagcxResult_t rcclAdaptorCommDestroy(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclCommDestroy(comm->base);
}

flagcxResult_t rcclAdaptorCommAbort(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclCommAbort(comm->base);
}

flagcxResult_t rcclAdaptorCommResume(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclInvalidUsage;
}

flagcxResult_t rcclAdaptorCommSuspend(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclInvalidUsage;
}

flagcxResult_t rcclAdaptorCommCount(const flagcxInnerComm_t comm, int *count) {
  return (flagcxResult_t)ncclCommCount(comm->base, count);
}

flagcxResult_t rcclAdaptorCommCuDevice(const flagcxInnerComm_t comm,
                                       int *device) {
  return (flagcxResult_t)ncclCommCuDevice(comm->base, device);
}

flagcxResult_t rcclAdaptorCommUserRank(const flagcxInnerComm_t comm,
                                       int *rank) {
  return (flagcxResult_t)ncclCommUserRank(comm->base, rank);
}

flagcxResult_t rcclAdaptorCommGetAsyncError(flagcxInnerComm_t comm,
                                            flagcxResult_t *asyncError) {
  return (flagcxResult_t)ncclCommGetAsyncError(comm->base,
                                               (ncclResult_t *)asyncError);
}

// TODO: unsupported
flagcxResult_t rcclAdaptorMemAlloc(void **ptr, size_t size) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t rcclAdaptorMemFree(void *ptr) { return flagcxNotSupported; }

// TODO: unsupported
flagcxResult_t rcclAdaptorCommRegister(flagcxInnerComm_t comm, void *buff,
                                       size_t size, void **handle) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t rcclAdaptorCommDeregister(flagcxInnerComm_t comm, void *handle) {
  return flagcxNotSupported;
}

flagcxResult_t rcclAdaptorCommWindowRegister(flagcxInnerComm_t comm, void *buff,
                                             size_t size, flagcxWindow_t *win,
                                             int winFlags) {
  return flagcxNotSupported;
}

flagcxResult_t rcclAdaptorCommWindowDeregister(flagcxInnerComm_t comm,
                                               flagcxWindow_t win) {
  return flagcxNotSupported;
}

flagcxResult_t rcclAdaptorReduce(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 flagcxRedOp_t op, int root,
                                 flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
  return (flagcxResult_t)ncclReduce(sendbuff, recvbuff, count,
                                    (ncclDataType_t)datatype, (ncclRedOp_t)op,
                                    root, comm->base, stream->base);
}

flagcxResult_t rcclAdaptorGather(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 int root, flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
  int rank, nranks;
  size_t size = count * getFlagcxDataTypeSize(datatype);
  char *buffer = static_cast<char *>(recvbuff);
  ncclResult_t res = ncclSuccess;

  CCLCHECKGOTO(ncclCommUserRank(comm->base, &rank), res, fail);
  CCLCHECKGOTO(ncclCommCount(comm->base, &nranks), res, fail);
  CCLCHECKGOTO(ncclGroupStart(), res, fail);
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      CCLCHECKGOTO(ncclRecv(static_cast<void *>(buffer + r * size), size,
                            ncclChar, r, comm->base, stream->base),
                   res, fail);
    }
  }
  CCLCHECKGOTO(
      ncclSend(sendbuff, size, ncclChar, root, comm->base, stream->base), res,
      fail);
  CCLCHECKGOTO(ncclGroupEnd(), res, fail);

  return flagcxSuccess;
fail:
  return (flagcxResult_t)res;
}

flagcxResult_t rcclAdaptorScatter(const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  int root, flagcxInnerComm_t comm,
                                  flagcxStream_t stream) {
  int rank, nranks;
  size_t size = count * getFlagcxDataTypeSize(datatype);
  const char *buffer = static_cast<const char *>(sendbuff);
  ncclResult_t res = ncclSuccess;

  CCLCHECKGOTO(ncclCommUserRank(comm->base, &rank), res, fail);
  CCLCHECKGOTO(ncclCommCount(comm->base, &nranks), res, fail);
  CCLCHECKGOTO(ncclGroupStart(), res, fail);
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      CCLCHECKGOTO(ncclSend(static_cast<const void *>(buffer + r * size), size,
                            ncclChar, r, comm->base, stream->base),
                   res, fail);
    }
  }
  CCLCHECKGOTO(
      ncclRecv(recvbuff, size, ncclChar, root, comm->base, stream->base), res,
      fail);
  CCLCHECKGOTO(ncclGroupEnd(), res, fail);

  return flagcxSuccess;
fail:
  return (flagcxResult_t)res;
}

flagcxResult_t rcclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    int root, flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return (flagcxResult_t)ncclBroadcast(sendbuff, recvbuff, count,
                                       (ncclDataType_t)datatype, root,
                                       comm->base, stream->base);
}

flagcxResult_t rcclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    flagcxRedOp_t op, flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return (flagcxResult_t)ncclAllReduce(
      sendbuff, recvbuff, count, (ncclDataType_t)datatype, (ncclRedOp_t)op,
      comm->base, stream->base);
}

flagcxResult_t
rcclAdaptorReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                         flagcxDataType_t datatype, flagcxRedOp_t op,
                         flagcxInnerComm_t comm, flagcxStream_t stream) {
  return (flagcxResult_t)ncclReduceScatter(
      sendbuff, recvbuff, recvcount, (ncclDataType_t)datatype, (ncclRedOp_t)op,
      comm->base, stream->base);
}

flagcxResult_t rcclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                    size_t sendcount, flagcxDataType_t datatype,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return (flagcxResult_t)ncclAllGather(sendbuff, recvbuff, sendcount,
                                       (ncclDataType_t)datatype, comm->base,
                                       stream->base);
}

flagcxResult_t rcclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                   size_t count, flagcxDataType_t datatype,
                                   flagcxInnerComm_t comm,
                                   flagcxStream_t stream) {
  int nranks;
  size_t size = count * getFlagcxDataTypeSize(datatype);
  const char *bufferIn = static_cast<const char *>(sendbuff);
  char *bufferOut = static_cast<char *>(recvbuff);
  ncclResult_t res = ncclSuccess;

  CCLCHECKGOTO(ncclCommCount(comm->base, &nranks), res, fail);
  CCLCHECKGOTO(ncclGroupStart(), res, fail);
  for (int r = 0; r < nranks; r++) {
    CCLCHECKGOTO(ncclSend(static_cast<const void *>(bufferIn + r * size), size,
                          ncclChar, r, comm->base, stream->base),
                 res, fail);
    CCLCHECKGOTO(ncclRecv(static_cast<void *>(bufferOut + r * size), size,
                          ncclChar, r, comm->base, stream->base),
                 res, fail);
  }
  CCLCHECKGOTO(ncclGroupEnd(), res, fail);

  return flagcxSuccess;
fail:
  return (flagcxResult_t)res;
}

flagcxResult_t rcclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                    size_t *sdispls, void *recvbuff,
                                    size_t *recvcounts, size_t *rdispls,
                                    flagcxDataType_t datatype,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  int nranks;
  size_t size = getFlagcxDataTypeSize(datatype);
  const char *bufferIn = static_cast<const char *>(sendbuff);
  char *bufferOut = static_cast<char *>(recvbuff);
  ncclResult_t res = ncclSuccess;

  CCLCHECKGOTO(ncclCommCount(comm->base, &nranks), res, fail);
  CCLCHECKGOTO(ncclGroupStart(), res, fail);
  for (int r = 0; r < nranks; r++) {
    if (flagcxCCLAdaptorNeedSendrecv(sendcounts[r])) {
      CCLCHECKGOTO(
          ncclSend(static_cast<const void *>(bufferIn + sdispls[r] * size),
                   sendcounts[r], (ncclDataType_t)datatype, r, comm->base,
                   stream->base),
          res, fail);
    }
    if (flagcxCCLAdaptorNeedSendrecv(recvcounts[r])) {
      CCLCHECKGOTO(ncclRecv(static_cast<void *>(bufferOut + rdispls[r] * size),
                            recvcounts[r], (ncclDataType_t)datatype, r,
                            comm->base, stream->base),
                   res, fail);
    }
  }
  CCLCHECKGOTO(ncclGroupEnd(), res, fail);

  return flagcxSuccess;
fail:
  return (flagcxResult_t)res;
}

flagcxResult_t rcclAdaptorSend(const void *sendbuff, size_t count,
                               flagcxDataType_t datatype, int peer,
                               flagcxInnerComm_t comm, flagcxStream_t stream) {
  return (flagcxResult_t)ncclSend(sendbuff, count, (ncclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

flagcxResult_t rcclAdaptorRecv(void *recvbuff, size_t count,
                               flagcxDataType_t datatype, int peer,
                               flagcxInnerComm_t comm, flagcxStream_t stream) {
  return (flagcxResult_t)ncclRecv(recvbuff, count, (ncclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

flagcxResult_t rcclAdaptorGroupStart() {
  return (flagcxResult_t)ncclGroupStart();
}

flagcxResult_t rcclAdaptorGroupEnd() { return (flagcxResult_t)ncclGroupEnd(); }

struct flagcxCCLAdaptor rcclAdaptor = {
    "RCCL",
    // Basic functions
    rcclAdaptorGetVersion, rcclAdaptorGetUniqueId, rcclAdaptorGetErrorString,
    rcclAdaptorGetLastError, rcclAdaptorGetStagedBuffer,
    // Communicator functions
    rcclAdaptorCommInitRank, rcclAdaptorCommFinalize, rcclAdaptorCommDestroy,
    rcclAdaptorCommAbort, rcclAdaptorCommResume, rcclAdaptorCommSuspend,
    rcclAdaptorCommCount, rcclAdaptorCommCuDevice, rcclAdaptorCommUserRank,
    rcclAdaptorCommGetAsyncError, rcclAdaptorMemAlloc, rcclAdaptorMemFree,
    rcclAdaptorCommRegister, rcclAdaptorCommDeregister,
    // Symmetric functions
    rcclAdaptorCommWindowRegister, rcclAdaptorCommWindowDeregister,
    // Communication functions
    rcclAdaptorReduce, rcclAdaptorGather, rcclAdaptorScatter,
    rcclAdaptorBroadcast, rcclAdaptorAllReduce, rcclAdaptorReduceScatter,
    rcclAdaptorAllGather, rcclAdaptorAlltoAll, rcclAdaptorAlltoAllv,
    rcclAdaptorSend, rcclAdaptorRecv,
    // Group semantics
    rcclAdaptorGroupStart, rcclAdaptorGroupEnd};

#endif // USE_AMD_ADAPTOR