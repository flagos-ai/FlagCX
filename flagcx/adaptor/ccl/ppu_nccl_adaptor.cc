#include "ppu_adaptor.h"

#ifdef USE_PPU_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "comm.h"

flagcxResult_t ppuncclAdaptorGetVersion(int *version) {
  return (flagcxResult_t)ncclGetVersion(version);
}

flagcxResult_t ppuncclAdaptorGetUniqueId(flagcxUniqueId_t *uniqueId) {
  if (*uniqueId == NULL) {
    flagcxCalloc(uniqueId, 1);
  }
  return (flagcxResult_t)ncclGetUniqueId((ncclUniqueId *)(*uniqueId));
}

flagcxResult_t ppuncclAdaptorGetStagedBuffer(const flagcxInnerComm_t comm,
                                             void **buff, size_t size,
                                             int isRecv) {
  return flagcxNotSupported;
}

const char *ppuncclAdaptorGetErrorString(flagcxResult_t result) {
  return ncclGetErrorString((ncclResult_t)result);
}

const char *ppuncclAdaptorGetLastError(flagcxInnerComm_t comm) {
  return ncclGetLastError(comm->base);
}

flagcxResult_t
ppuncclAdaptorCommInitRank(flagcxInnerComm_t *comm, int nranks,
                           flagcxUniqueId_t commId, int rank,
                           struct bootstrapState * /*bootstrap*/) {
  if (*comm == NULL) {
    flagcxCalloc(comm, 1);
  }
  return (flagcxResult_t)ncclCommInitRank(&(*comm)->base, nranks,
                                          *(ncclUniqueId *)commId, rank);
}

flagcxResult_t ppuncclAdaptorCommFinalize(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclCommFinalize(comm->base);
}

flagcxResult_t ppuncclAdaptorCommDestroy(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclCommDestroy(comm->base);
}

flagcxResult_t ppuncclAdaptorCommAbort(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ncclCommAbort(comm->base);
}

flagcxResult_t ppuncclAdaptorCommResume(flagcxInnerComm_t comm) {
  return flagcxNotSupported;
}

flagcxResult_t ppuncclAdaptorCommSuspend(flagcxInnerComm_t comm) {
  return flagcxNotSupported;
}

flagcxResult_t ppuncclAdaptorCommCount(const flagcxInnerComm_t comm,
                                       int *count) {
  return (flagcxResult_t)ncclCommCount(comm->base, count);
}

flagcxResult_t ppuncclAdaptorCommCuDevice(const flagcxInnerComm_t comm,
                                          int *device) {
  return (flagcxResult_t)ncclCommCuDevice(comm->base, device);
}

flagcxResult_t ppuncclAdaptorCommUserRank(const flagcxInnerComm_t comm,
                                          int *rank) {
  return (flagcxResult_t)ncclCommUserRank(comm->base, rank);
}

flagcxResult_t ppuncclAdaptorCommGetAsyncError(flagcxInnerComm_t comm,
                                               flagcxResult_t *asyncError) {
  return (flagcxResult_t)ncclCommGetAsyncError(comm->base,
                                               (ncclResult_t *)asyncError);
}

flagcxResult_t ppuncclAdaptorMemAlloc(void **ptr, size_t size) {
  return flagcxNotSupported;
}

flagcxResult_t ppuncclAdaptorMemFree(void *ptr) { return flagcxNotSupported; }

flagcxResult_t ppuncclAdaptorCommRegister(flagcxInnerComm_t comm, void *buff,
                                          size_t size, void **handle) {
  return flagcxNotSupported;
}

flagcxResult_t ppuncclAdaptorCommDeregister(flagcxInnerComm_t comm,
                                            void *handle) {
  return flagcxNotSupported;
}

flagcxResult_t ppuncclAdaptorCommWindowRegister(flagcxInnerComm_t comm,
                                                void *buff, size_t size,
                                                flagcxInnerWindow_t *win,
                                                int winFlags) {
  return flagcxNotSupported;
}

flagcxResult_t ppuncclAdaptorCommWindowDeregister(flagcxInnerComm_t comm,
                                                  flagcxInnerWindow_t win) {
  return flagcxNotSupported;
}

flagcxResult_t ppuncclAdaptorReduce(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    flagcxRedOp_t op, int root,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return (flagcxResult_t)ncclReduce(sendbuff, recvbuff, count,
                                    (ncclDataType_t)datatype, (ncclRedOp_t)op,
                                    root, comm->base, stream->base);
}

flagcxResult_t ppuncclAdaptorGather(const void *sendbuff, void *recvbuff,
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

flagcxResult_t ppuncclAdaptorScatter(const void *sendbuff, void *recvbuff,
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

flagcxResult_t ppuncclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                       size_t count, flagcxDataType_t datatype,
                                       int root, flagcxInnerComm_t comm,
                                       flagcxStream_t stream) {
  return (flagcxResult_t)ncclBroadcast(sendbuff, recvbuff, count,
                                       (ncclDataType_t)datatype, root,
                                       comm->base, stream->base);
}

flagcxResult_t ppuncclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                       size_t count, flagcxDataType_t datatype,
                                       flagcxRedOp_t op, flagcxInnerComm_t comm,
                                       flagcxStream_t stream) {
  return (flagcxResult_t)ncclAllReduce(
      sendbuff, recvbuff, count, (ncclDataType_t)datatype, (ncclRedOp_t)op,
      comm->base, stream->base);
}

flagcxResult_t
ppuncclAdaptorReduceScatter(const void *sendbuff, void *recvbuff, size_t count,
                            flagcxDataType_t datatype, flagcxRedOp_t op,
                            flagcxInnerComm_t comm, flagcxStream_t stream) {
  return (flagcxResult_t)ncclReduceScatter(
      sendbuff, recvbuff, count, (ncclDataType_t)datatype, (ncclRedOp_t)op,
      comm->base, stream->base);
}

flagcxResult_t ppuncclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                       size_t count, flagcxDataType_t datatype,
                                       flagcxInnerComm_t comm,
                                       flagcxStream_t stream) {
  return (flagcxResult_t)ncclAllGather(sendbuff, recvbuff, count,
                                       (ncclDataType_t)datatype, comm->base,
                                       stream->base);
}

flagcxResult_t ppuncclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                      size_t count, flagcxDataType_t datatype,
                                      flagcxInnerComm_t comm,
                                      flagcxStream_t stream) {
  int nranks;
  FLAGCXCHECK((flagcxResult_t)ncclCommCount(comm->base, &nranks));

  size_t size = count * getFlagcxDataTypeSize(datatype);
  const char *sendBuffer = static_cast<const char *>(sendbuff);
  char *recvBuffer = static_cast<char *>(recvbuff);

  flagcxResult_t res = flagcxSuccess;
  FLAGCXCHECK((flagcxResult_t)ncclGroupStart());
  for (int r = 0; r < nranks; r++) {
    FLAGCXCHECKGOTO((flagcxResult_t)ncclSend(
                        static_cast<const void *>(sendBuffer + r * size), size,
                        ncclChar, r, comm->base, stream->base),
                    res, group_exit);
    FLAGCXCHECKGOTO(
        (flagcxResult_t)ncclRecv(static_cast<void *>(recvBuffer + r * size),
                                 size, ncclChar, r, comm->base, stream->base),
        res, group_exit);
  }
group_exit:
  FLAGCXCHECK((flagcxResult_t)ncclGroupEnd());
  return res;
}

flagcxResult_t ppuncclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                       size_t *sdispls, void *recvbuff,
                                       size_t *recvcounts, size_t *rdispls,
                                       flagcxDataType_t datatype,
                                       flagcxInnerComm_t comm,
                                       flagcxStream_t stream) {
  int nranks;
  FLAGCXCHECK((flagcxResult_t)ncclCommCount(comm->base, &nranks));

  size_t typeSize = getFlagcxDataTypeSize(datatype);
  const char *sendBuffer = static_cast<const char *>(sendbuff);
  char *recvBuffer = static_cast<char *>(recvbuff);

  flagcxResult_t res = flagcxSuccess;
  FLAGCXCHECK((flagcxResult_t)ncclGroupStart());
  for (int r = 0; r < nranks; r++) {
    if (flagcxCCLAdaptorNeedSendrecv(sendcounts[r])) {
      FLAGCXCHECKGOTO(
          (flagcxResult_t)ncclSend(
              static_cast<const void *>(sendBuffer + sdispls[r] * typeSize),
              sendcounts[r] * typeSize, ncclChar, r, comm->base, stream->base),
          res, group_exit);
    }
    if (flagcxCCLAdaptorNeedSendrecv(recvcounts[r])) {
      FLAGCXCHECKGOTO(
          (flagcxResult_t)ncclRecv(
              static_cast<void *>(recvBuffer + rdispls[r] * typeSize),
              recvcounts[r] * typeSize, ncclChar, r, comm->base, stream->base),
          res, group_exit);
    }
  }
group_exit:
  FLAGCXCHECK((flagcxResult_t)ncclGroupEnd());
  return res;
}

flagcxResult_t ppuncclAdaptorSend(const void *sendbuff, size_t count,
                                  flagcxDataType_t datatype, int peer,
                                  flagcxInnerComm_t comm,
                                  flagcxStream_t stream) {
  return (flagcxResult_t)ncclSend(sendbuff, count, (ncclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

flagcxResult_t ppuncclAdaptorRecv(void *recvbuff, size_t count,
                                  flagcxDataType_t datatype, int peer,
                                  flagcxInnerComm_t comm,
                                  flagcxStream_t stream) {
  return (flagcxResult_t)ncclRecv(recvbuff, count, (ncclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

flagcxResult_t ppuncclAdaptorGroupStart() {
  return (flagcxResult_t)ncclGroupStart();
}

flagcxResult_t ppuncclAdaptorGroupEnd() {
  return (flagcxResult_t)ncclGroupEnd();
}

flagcxResult_t
ppuncclAdaptorDevCommReqsInit(flagcxInnerComm_t /*comm*/,
                              flagcxDevCommRequirements * /*reqs*/) {
  return flagcxNotSupported;
}

flagcxResult_t
ppuncclAdaptorDevCommCreate(flagcxInnerComm_t /*comm*/,
                            const flagcxDevCommRequirements * /*reqs*/,
                            flagcxInnerDevComm_t * /*devComm*/) {
  return flagcxNotSupported;
}

flagcxResult_t ppuncclAdaptorDevCommDestroy(flagcxInnerComm_t /*comm*/,
                                            flagcxInnerDevComm_t /*devComm*/) {
  return flagcxNotSupported;
}

struct flagcxCCLAdaptor ppuncclAdaptor = {
    "PPU_NCCL",
    // Basic functions
    ppuncclAdaptorGetVersion, ppuncclAdaptorGetUniqueId,
    ppuncclAdaptorGetErrorString, ppuncclAdaptorGetLastError,
    ppuncclAdaptorGetStagedBuffer,
    // Communicator functions
    ppuncclAdaptorCommInitRank, ppuncclAdaptorCommFinalize,
    ppuncclAdaptorCommDestroy, ppuncclAdaptorCommAbort,
    ppuncclAdaptorCommResume, ppuncclAdaptorCommSuspend,
    ppuncclAdaptorCommCount, ppuncclAdaptorCommCuDevice,
    ppuncclAdaptorCommUserRank, ppuncclAdaptorCommGetAsyncError,
    ppuncclAdaptorMemAlloc, ppuncclAdaptorMemFree, ppuncclAdaptorCommRegister,
    ppuncclAdaptorCommDeregister,
    // Symmetric functions
    ppuncclAdaptorCommWindowRegister, ppuncclAdaptorCommWindowDeregister,
    // Communication functions
    ppuncclAdaptorReduce, ppuncclAdaptorGather, ppuncclAdaptorScatter,
    ppuncclAdaptorBroadcast, ppuncclAdaptorAllReduce,
    ppuncclAdaptorReduceScatter, ppuncclAdaptorAllGather,
    ppuncclAdaptorAlltoAll, ppuncclAdaptorAlltoAllv, ppuncclAdaptorSend,
    ppuncclAdaptorRecv,
    // Group semantics
    ppuncclAdaptorGroupStart, ppuncclAdaptorGroupEnd,
    // Device API
    ppuncclAdaptorDevCommReqsInit, ppuncclAdaptorDevCommCreate,
    ppuncclAdaptorDevCommDestroy};

#endif // USE_PPU_ADAPTOR
