#include "du_adaptor.h"

#ifdef USE_DU_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "comm.h"

static_assert(sizeof(ncclUniqueId) <= sizeof(flagcxInnerUniqueId),
              "ncclUniqueId does not fit inside flagcxInnerUniqueId");

flagcxResult_t durcclAdaptorGetVersion(int *version) {
  return (flagcxResult_t)ncclGetVersion(version);
}

flagcxResult_t durcclAdaptorGetUniqueId(flagcxInnerUniqueId_t *uniqueId) {
  if (*uniqueId == NULL) {
    flagcxCalloc(uniqueId, 1);
  }
  return (flagcxResult_t)ncclGetUniqueId((ncclUniqueId *)(*uniqueId));
}

flagcxResult_t durcclAdaptorGetStagedBuffer(const flagcxInnerComm_t comm,
                                            void **buff, size_t size,
                                            int isRecv) {
  return flagcxNotSupported;
}

const char *durcclAdaptorGetErrorString(flagcxResult_t result) {
  return ncclGetErrorString((ncclResult_t)result);
}

const char *durcclAdaptorGetLastError(flagcxInnerComm_t comm) {
  return ncclGetLastError(comm->base);
}

flagcxResult_t
durcclAdaptorCommInitRank(flagcxInnerComm_t *comm, int nranks,
                          flagcxInnerUniqueId_t commId, int rank,
                          struct bootstrapState * /*bootstrap*/) {
  if (*comm == NULL) {
    flagcxCalloc(comm, 1);
  }
  return (flagcxResult_t)ncclCommInitRank(&(*comm)->base, nranks,
                                          *(ncclUniqueId *)commId, rank);
}

flagcxResult_t durcclAdaptorCommFinalize(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclCommFinalize(comm->base);
}

flagcxResult_t durcclAdaptorCommDestroy(flagcxInnerComm_t comm) {
  if (comm == NULL)
    return flagcxInvalidArgument;
  flagcxResult_t result = (flagcxResult_t)ncclCommDestroy(comm->base);
  if (result != flagcxSuccess)
    return result;
  free(comm);
  return flagcxSuccess;
}

flagcxResult_t durcclAdaptorCommAbort(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclCommAbort(comm->base);
}

flagcxResult_t durcclAdaptorCommResume(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclInvalidUsage;
}

flagcxResult_t durcclAdaptorCommSuspend(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclInvalidUsage;
}

flagcxResult_t durcclAdaptorCommCount(const flagcxInnerComm_t comm,
                                      int *count) {
  return (flagcxResult_t)ncclCommCount(comm->base, count);
}

flagcxResult_t durcclAdaptorCommCuDevice(const flagcxInnerComm_t comm,
                                         int *device) {
  return (flagcxResult_t)ncclCommCuDevice(comm->base, device);
}

flagcxResult_t durcclAdaptorCommUserRank(const flagcxInnerComm_t comm,
                                         int *rank) {
  return (flagcxResult_t)ncclCommUserRank(comm->base, rank);
}

flagcxResult_t durcclAdaptorCommGetAsyncError(flagcxInnerComm_t comm,
                                              flagcxResult_t *asyncError) {
  return (flagcxResult_t)ncclCommGetAsyncError(comm->base,
                                               (ncclResult_t *)asyncError);
}

// TODO: unsupported
flagcxResult_t durcclAdaptorMemAlloc(void **ptr, size_t size) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t durcclAdaptorMemFree(void *ptr) { return flagcxNotSupported; }

// TODO: unsupported
flagcxResult_t durcclAdaptorCommRegister(flagcxInnerComm_t comm, void *buff,
                                         size_t size, void **handle) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t durcclAdaptorCommDeregister(flagcxInnerComm_t comm,
                                           void *handle) {
  return flagcxNotSupported;
}

flagcxResult_t durcclAdaptorCommWindowRegister(flagcxInnerComm_t comm,
                                               void *buff, size_t size,
                                               flagcxInnerWindow_t *win,
                                               int winFlags) {
  return flagcxNotSupported;
}

flagcxResult_t durcclAdaptorCommWindowDeregister(flagcxInnerComm_t comm,
                                                 flagcxInnerWindow_t win) {
  return flagcxNotSupported;
}

flagcxResult_t durcclAdaptorReduce(const void *sendbuff, void *recvbuff,
                                   size_t count, flagcxDataType_t datatype,
                                   flagcxRedOp_t op, int root,
                                   flagcxInnerComm_t comm,
                                   flagcxStream_t stream) {
  return (flagcxResult_t)ncclReduce(sendbuff, recvbuff, count,
                                    (ncclDataType_t)datatype, (ncclRedOp_t)op,
                                    root, comm->base, stream->base);
}

flagcxResult_t durcclAdaptorGather(const void *sendbuff, void *recvbuff,
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

flagcxResult_t durcclAdaptorScatter(const void *sendbuff, void *recvbuff,
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

flagcxResult_t durcclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                      size_t count, flagcxDataType_t datatype,
                                      int root, flagcxInnerComm_t comm,
                                      flagcxStream_t stream) {
  return (flagcxResult_t)ncclBroadcast(sendbuff, recvbuff, count,
                                       (ncclDataType_t)datatype, root,
                                       comm->base, stream->base);
}

flagcxResult_t durcclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                      size_t count, flagcxDataType_t datatype,
                                      flagcxRedOp_t op, flagcxInnerComm_t comm,
                                      flagcxStream_t stream) {
  return (flagcxResult_t)ncclAllReduce(
      sendbuff, recvbuff, count, (ncclDataType_t)datatype, (ncclRedOp_t)op,
      comm->base, stream->base);
}

flagcxResult_t durcclAdaptorReduceScatter(const void *sendbuff, void *recvbuff,
                                          size_t recvcount,
                                          flagcxDataType_t datatype,
                                          flagcxRedOp_t op,
                                          flagcxInnerComm_t comm,
                                          flagcxStream_t stream) {
  return (flagcxResult_t)ncclReduceScatter(
      sendbuff, recvbuff, recvcount, (ncclDataType_t)datatype, (ncclRedOp_t)op,
      comm->base, stream->base);
}

flagcxResult_t durcclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                      size_t sendcount,
                                      flagcxDataType_t datatype,
                                      flagcxInnerComm_t comm,
                                      flagcxStream_t stream) {
  return (flagcxResult_t)ncclAllGather(sendbuff, recvbuff, sendcount,
                                       (ncclDataType_t)datatype, comm->base,
                                       stream->base);
}

flagcxResult_t durcclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
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

flagcxResult_t durcclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
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

flagcxResult_t durcclAdaptorSend(const void *sendbuff, size_t count,
                                 flagcxDataType_t datatype, int peer,
                                 flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
  return (flagcxResult_t)ncclSend(sendbuff, count, (ncclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

flagcxResult_t durcclAdaptorRecv(void *recvbuff, size_t count,
                                 flagcxDataType_t datatype, int peer,
                                 flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
  return (flagcxResult_t)ncclRecv(recvbuff, count, (ncclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

flagcxResult_t durcclAdaptorGroupStart() {
  return (flagcxResult_t)ncclGroupStart();
}

flagcxResult_t durcclAdaptorGroupEnd() {
  return (flagcxResult_t)ncclGroupEnd();
}

flagcxResult_t
durcclAdaptorDevCommReqsInit(flagcxInnerComm_t /*comm*/,
                             flagcxDevCommRequirements * /*reqs*/) {
  return flagcxNotSupported;
}

flagcxResult_t
durcclAdaptorDevCommCreate(flagcxInnerComm_t /*comm*/,
                           const flagcxDevCommRequirements * /*reqs*/,
                           flagcxInnerDevComm_t * /*devComm*/) {
  return flagcxNotSupported;
}

flagcxResult_t durcclAdaptorDevCommDestroy(flagcxInnerComm_t /*comm*/,
                                           flagcxInnerDevComm_t /*devComm*/) {
  return flagcxNotSupported;
}

struct flagcxCCLAdaptor durcclAdaptor = {
    "DURCCL",
    // Basic functions
    durcclAdaptorGetVersion, durcclAdaptorGetUniqueId,
    durcclAdaptorGetErrorString, durcclAdaptorGetLastError,
    durcclAdaptorGetStagedBuffer,
    // Communicator functions
    durcclAdaptorCommInitRank, durcclAdaptorCommFinalize,
    durcclAdaptorCommDestroy, durcclAdaptorCommAbort, durcclAdaptorCommResume,
    durcclAdaptorCommSuspend, durcclAdaptorCommCount, durcclAdaptorCommCuDevice,
    durcclAdaptorCommUserRank, durcclAdaptorCommGetAsyncError,
    durcclAdaptorMemAlloc, durcclAdaptorMemFree, durcclAdaptorCommRegister,
    durcclAdaptorCommDeregister,
    // Symmetric functions
    durcclAdaptorCommWindowRegister, durcclAdaptorCommWindowDeregister,
    // Communication functions
    durcclAdaptorReduce, durcclAdaptorGather, durcclAdaptorScatter,
    durcclAdaptorBroadcast, durcclAdaptorAllReduce, durcclAdaptorReduceScatter,
    durcclAdaptorAllGather, durcclAdaptorAlltoAll, durcclAdaptorAlltoAllv,
    durcclAdaptorSend, durcclAdaptorRecv,
    // Group semantics
    durcclAdaptorGroupStart, durcclAdaptorGroupEnd,
    // Device API
    durcclAdaptorDevCommReqsInit, durcclAdaptorDevCommCreate,
    durcclAdaptorDevCommDestroy};

#endif // USE_DU_ADAPTOR
