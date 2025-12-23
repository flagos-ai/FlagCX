/*************************************************************************
 * Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd.
   All Rights Reserved.
 ************************************************************************/

#include "metax_adaptor.h"

#ifdef USE_METAX_ADAPTOR

flagcxResult_t mcclAdaptorGetVersion(int *version) {
  return (flagcxResult_t)mcclGetVersion(version);
}

flagcxResult_t mcclAdaptorGetUniqueId(flagcxUniqueId_t *uniqueId) {
  if (*uniqueId == NULL) {
    flagcxCalloc(uniqueId, 1);
  }
  return (flagcxResult_t)mcclGetUniqueId((mcclUniqueId *)(*uniqueId));
}

const char *mcclAdaptorGetErrorString(flagcxResult_t result) {
  return mcclGetErrorString((mcclResult_t)result);
}

const char *mcclAdaptorGetLastError(flagcxInnerComm_t comm) {
  return mcclGetLastError(comm->base);
}

flagcxResult_t mcclAdaptorCommInitRank(flagcxInnerComm_t *comm, int nranks,
                                       flagcxUniqueId_t commId, int rank,
                                       bootstrapState * /*bootstrap*/) {
  if (*comm == NULL) {
    flagcxCalloc(comm, 1);
  }
  return (flagcxResult_t)mcclCommInitRank(&(*comm)->base, nranks,
                                          *(mcclUniqueId *)commId, rank);
}

flagcxResult_t mcclAdaptorCommFinalize(flagcxInnerComm_t comm) {
  return (flagcxResult_t)mcclCommFinalize(comm->base);
}

flagcxResult_t mcclAdaptorCommDestroy(flagcxInnerComm_t comm) {
  return (flagcxResult_t)mcclCommDestroy(comm->base);
}

flagcxResult_t mcclAdaptorCommAbort(flagcxInnerComm_t comm) {
  return (flagcxResult_t)mcclCommAbort(comm->base);
}

flagcxResult_t mcclAdaptorCommResume(flagcxInnerComm_t comm) {
  return (flagcxResult_t)mcclInvalidUsage;
}

flagcxResult_t mcclAdaptorCommSuspend(flagcxInnerComm_t comm) {
  return (flagcxResult_t)mcclInvalidUsage;
}

flagcxResult_t mcclAdaptorCommCount(const flagcxInnerComm_t comm, int *count) {
  return (flagcxResult_t)mcclCommCount(comm->base, count);
}

flagcxResult_t mcclAdaptorCommMcDevice(const flagcxInnerComm_t comm,
                                       int *device) {
  return (flagcxResult_t)mcclCommMcDevice(comm->base, device);
}

flagcxResult_t mcclAdaptorCommUserRank(const flagcxInnerComm_t comm,
                                       int *rank) {
  return (flagcxResult_t)mcclCommUserRank(comm->base, rank);
}

flagcxResult_t mcclAdaptorCommGetAsyncError(flagcxInnerComm_t comm,
                                            flagcxResult_t *asyncError) {
  return (flagcxResult_t)mcclCommGetAsyncError(comm->base,
                                               (mcclResult_t *)asyncError);
}

// TODO: unsupported
flagcxResult_t mcclAdaptorMemAlloc(void **ptr, size_t size) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t mcclAdaptorMemFree(void *ptr) { return flagcxNotSupported; }

// TODO: unsupported
flagcxResult_t mcclAdaptorCommRegister(flagcxInnerComm_t comm, void *buff,
                                       size_t size, void **handle) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t mcclAdaptorCommDeregister(flagcxInnerComm_t comm, void *handle) {
  return flagcxNotSupported;
}

flagcxResult_t mcclAdaptorReduce(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 flagcxRedOp_t op, int root,
                                 flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
  return (flagcxResult_t)mcclReduce(sendbuff, recvbuff, count,
                                    (mcclDataType_t)datatype, (mcclRedOp_t)op,
                                    root, comm->base, stream->base);
}

flagcxResult_t mcclAdaptorGather(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 int root, flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
  int rank, nranks;
  mcclResult_t res = mcclSuccess;
  res = mcclCommUserRank(comm->base, &rank);
  res = mcclCommCount(comm->base, &nranks);

  size_t size = count * getFlagcxDataTypeSize(datatype);
  char *buffer = static_cast<char *>(recvbuff);

  res = mcclGroupStart();
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      res = mcclRecv(static_cast<void *>(buffer + r * size), size, mcclChar, r,
                     comm->base, stream->base);
    }
  }
  res = mcclSend(sendbuff, size, mcclChar, root, comm->base, stream->base);
  res = mcclGroupEnd();

  return (flagcxResult_t)res;
}

flagcxResult_t mcclAdaptorScatter(const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  int root, flagcxInnerComm_t comm,
                                  flagcxStream_t stream) {
  int rank, nranks;
  mcclResult_t res = mcclSuccess;
  res = mcclCommUserRank(comm->base, &rank);
  res = mcclCommCount(comm->base, &nranks);

  size_t size = count * getFlagcxDataTypeSize(datatype);
  const char *buffer = static_cast<const char *>(sendbuff);

  res = mcclGroupStart();
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      res = mcclSend(static_cast<const void *>(buffer + r * size), size,
                     mcclChar, r, comm->base, stream->base);
    }
  }
  res = mcclRecv(recvbuff, size, mcclChar, root, comm->base, stream->base);
  res = mcclGroupEnd();

  return (flagcxResult_t)res;
}

flagcxResult_t mcclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    int root, flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return (flagcxResult_t)mcclBroadcast(sendbuff, recvbuff, count,
                                       (mcclDataType_t)datatype, root,
                                       comm->base, stream->base);
}

flagcxResult_t mcclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    flagcxRedOp_t op, flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return (flagcxResult_t)mcclAllReduce(
      sendbuff, recvbuff, count, (mcclDataType_t)datatype, (mcclRedOp_t)op,
      comm->base, stream->base);
}

flagcxResult_t
mcclAdaptorReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                         flagcxDataType_t datatype, flagcxRedOp_t op,
                         flagcxInnerComm_t comm, flagcxStream_t stream) {
  return (flagcxResult_t)mcclReduceScatter(
      sendbuff, recvbuff, recvcount, (mcclDataType_t)datatype, (mcclRedOp_t)op,
      comm->base, stream->base);
}

flagcxResult_t mcclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                    size_t sendcount, flagcxDataType_t datatype,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return (flagcxResult_t)mcclAllGather(sendbuff, recvbuff, sendcount,
                                       (mcclDataType_t)datatype, comm->base,
                                       stream->base);
}

flagcxResult_t mcclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                   size_t count, flagcxDataType_t datatype,
                                   flagcxInnerComm_t comm,
                                   flagcxStream_t stream) {
  int rank, nranks;
  mcclResult_t res = mcclSuccess;
  res = mcclCommUserRank(comm->base, &rank);
  res = mcclCommCount(comm->base, &nranks);

  size_t size = count * getFlagcxDataTypeSize(datatype);
  const char *buffer_in = static_cast<const char *>(sendbuff);
  char *buffer_out = static_cast<char *>(recvbuff);

  res = mcclGroupStart();
  for (int r = 0; r < nranks; r++) {
    res = mcclSend(static_cast<const void *>(buffer_in + r * size), size,
                   mcclChar, r, comm->base, stream->base);
    res = mcclRecv(static_cast<void *>(buffer_out + r * size), size, mcclChar,
                   r, comm->base, stream->base);
  }
  res = mcclGroupEnd();

  return (flagcxResult_t)res;
}

flagcxResult_t mcclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                    size_t *sdispls, void *recvbuff,
                                    size_t *recvcounts, size_t *rdispls,
                                    flagcxDataType_t datatype,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  int nranks;
  mcclResult_t res = mcclSuccess;
  res = mcclCommCount(comm->base, &nranks);

  size_t size = getFlagcxDataTypeSize(datatype);
  const char *buffer_in = static_cast<const char *>(sendbuff);
  char *buffer_out = static_cast<char *>(recvbuff);

  res = mcclGroupStart();
  for (int r = 0; r < nranks; r++) {
    if (flagcxCCLAdaptorNeedSendrecv(sendcounts[r])) {
      res = mcclSend(static_cast<const void *>(buffer_in + sdispls[r] * size),
                     sendcounts[r], (mcclDataType_t)datatype, r, comm->base,
                     stream->base);
    }
    if (flagcxCCLAdaptorNeedSendrecv(recvcounts[r])) {
      res = mcclRecv(static_cast<void *>(buffer_out + rdispls[r] * size),
                     recvcounts[r], (mcclDataType_t)datatype, r, comm->base,
                     stream->base);
    }
  }
  res = mcclGroupEnd();

  return (flagcxResult_t)res;
}

flagcxResult_t mcclAdaptorSend(const void *sendbuff, size_t count,
                               flagcxDataType_t datatype, int peer,
                               flagcxInnerComm_t comm, flagcxStream_t stream) {
  return (flagcxResult_t)mcclSend(sendbuff, count, (mcclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

flagcxResult_t mcclAdaptorRecv(void *recvbuff, size_t count,
                               flagcxDataType_t datatype, int peer,
                               flagcxInnerComm_t comm, flagcxStream_t stream) {
  return (flagcxResult_t)mcclRecv(recvbuff, count, (mcclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

flagcxResult_t mcclAdaptorGroupStart() {
  return (flagcxResult_t)mcclGroupStart();
}

flagcxResult_t mcclAdaptorGroupEnd() { return (flagcxResult_t)mcclGroupEnd(); }

struct flagcxCCLAdaptor mcclAdaptor = {
    "MCCL",
    // Basic functions
    mcclAdaptorGetVersion, mcclAdaptorGetUniqueId, mcclAdaptorGetErrorString,
    mcclAdaptorGetLastError,
    // Communicator functions
    mcclAdaptorCommInitRank, mcclAdaptorCommFinalize, mcclAdaptorCommDestroy,
    mcclAdaptorCommAbort, mcclAdaptorCommResume, mcclAdaptorCommSuspend,
    mcclAdaptorCommCount, mcclAdaptorCommMcDevice, mcclAdaptorCommUserRank,
    mcclAdaptorCommGetAsyncError, mcclAdaptorMemAlloc, mcclAdaptorMemFree,
    mcclAdaptorCommRegister, mcclAdaptorCommDeregister,
    // Communication functions
    mcclAdaptorReduce, mcclAdaptorGather, mcclAdaptorScatter,
    mcclAdaptorBroadcast, mcclAdaptorAllReduce, mcclAdaptorReduceScatter,
    mcclAdaptorAllGather, mcclAdaptorAlltoAll, mcclAdaptorAlltoAllv,
    mcclAdaptorSend, mcclAdaptorRecv,
    // Group semantics
    mcclAdaptorGroupStart, mcclAdaptorGroupEnd};

#endif // USE_METAX_ADAPTOR
