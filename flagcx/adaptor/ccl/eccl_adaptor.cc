/*************************************************************************
 * Copyright (c) 2025, ENFLAME CORPORATION. All rights reserved.
 ************************************************************************/

#include "enflame_adaptor.h"

#ifdef USE_ENFLAME_ADAPTOR

flagcxResult_t ecclAdaptorGetVersion(int *version) {
  return (flagcxResult_t)ecclGetVersion(version);
}

flagcxResult_t ecclAdaptorGetUniqueId(flagcxUniqueId_t *uniqueId) {
  if (*uniqueId == NULL) {
    flagcxCalloc(uniqueId, 1);
  }
  return (flagcxResult_t)ecclGetUniqueId((ecclUniqueId *)(*uniqueId));
}

const char *ecclAdaptorGetErrorString(flagcxResult_t result) {
  return ecclGetErrorString((ecclResult_t)result);
}

const char *ecclAdaptorGetLastError(flagcxInnerComm_t comm) {
  return ecclGetLastError(comm->base);
}

flagcxResult_t ecclAdaptorCommInitRank(flagcxInnerComm_t *comm, int nranks,
                                       flagcxUniqueId_t commId, int rank,
                                       bootstrapState * /*bootstrap*/) {
  if (*comm == NULL) {
    flagcxCalloc(comm, 1);
  }
  return (flagcxResult_t)ecclCommInitRank(&(*comm)->base, nranks,
                                          *(ecclUniqueId *)commId, rank);
}

flagcxResult_t ecclAdaptorCommFinalize(flagcxInnerComm_t comm) {
  // ECCL does not have a separate finalize function, use destroy
  return flagcxSuccess;
}

flagcxResult_t ecclAdaptorCommDestroy(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ecclCommDestroy(comm->base);
}

flagcxResult_t ecclAdaptorCommAbort(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ecclCommAbort(comm->base);
}

flagcxResult_t ecclAdaptorCommResume(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ecclInvalidUsage;
}

flagcxResult_t ecclAdaptorCommSuspend(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ecclInvalidUsage;
}

flagcxResult_t ecclAdaptorCommCount(const flagcxInnerComm_t comm, int *count) {
  return (flagcxResult_t)ecclCommCount(comm->base, count);
}

flagcxResult_t ecclAdaptorCommCuDevice(const flagcxInnerComm_t comm,
                                       int *device) {
  return (flagcxResult_t)ecclCommDevice(comm->base, device);
}

flagcxResult_t ecclAdaptorCommUserRank(const flagcxInnerComm_t comm,
                                       int *rank) {
  return (flagcxResult_t)ecclCommUserRank(comm->base, rank);
}

flagcxResult_t ecclAdaptorCommGetAsyncError(flagcxInnerComm_t comm,
                                            flagcxResult_t *asyncError) {
  return (flagcxResult_t)ecclCommGetAsyncError(comm->base,
                                               (ecclResult_t *)asyncError);
}

flagcxResult_t ecclAdaptorMemAlloc(void **ptr, size_t size) {
  topsError_t err = topsMalloc(ptr, size);
  if (err != topsSuccess) {
    return flagcxUnhandledDeviceError;
  }
  return flagcxSuccess;
}

flagcxResult_t ecclAdaptorMemFree(void *ptr) {
  topsError_t err = topsFree(ptr);
  if (err != topsSuccess) {
    return flagcxUnhandledDeviceError;
  }
  return flagcxSuccess;
}

flagcxResult_t ecclAdaptorCommRegister(const flagcxInnerComm_t comm, void *buff,
                                       size_t size, void **handle) {
  // ECCL does not have memory registration, return success
  *handle = buff;
  return flagcxSuccess;
}

flagcxResult_t ecclAdaptorCommDeregister(const flagcxInnerComm_t comm,
                                         void *handle) {
  // ECCL does not have memory deregistration, return success
  return flagcxSuccess;
}

flagcxResult_t ecclAdaptorReduce(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 flagcxRedOp_t op, int root,
                                 flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
  return (flagcxResult_t)ecclReduce(sendbuff, recvbuff, count,
                                    (ecclDataType_t)datatype, (ecclRedOp_t)op,
                                    root, comm->base, stream->base);
}

flagcxResult_t ecclAdaptorGather(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 int root, flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
  return (flagcxResult_t)ecclGather(sendbuff, recvbuff, count,
                                    (ecclDataType_t)datatype, root,
                                    comm->base, stream->base);
}

flagcxResult_t ecclAdaptorScatter(const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  int root, flagcxInnerComm_t comm,
                                  flagcxStream_t stream) {
  return (flagcxResult_t)ecclScatter(sendbuff, recvbuff, count,
                                     (ecclDataType_t)datatype, root,
                                     comm->base, stream->base);
}

flagcxResult_t ecclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    int root, flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return (flagcxResult_t)ecclBroadcast(sendbuff, recvbuff, count,
                                       (ecclDataType_t)datatype, root,
                                       comm->base, stream->base);
}

flagcxResult_t ecclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    flagcxRedOp_t op, flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return (flagcxResult_t)ecclAllReduce(
      sendbuff, recvbuff, count, (ecclDataType_t)datatype, (ecclRedOp_t)op,
      comm->base, stream->base);
}

flagcxResult_t
ecclAdaptorReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                         flagcxDataType_t datatype, flagcxRedOp_t op,
                         flagcxInnerComm_t comm, flagcxStream_t stream) {
  return (flagcxResult_t)ecclReduceScatter(
      sendbuff, recvbuff, recvcount, (ecclDataType_t)datatype, (ecclRedOp_t)op,
      comm->base, stream->base);
}

flagcxResult_t ecclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                    size_t sendcount, flagcxDataType_t datatype,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return (flagcxResult_t)ecclAllGather(sendbuff, recvbuff, sendcount,
                                       (ecclDataType_t)datatype, comm->base,
                                       stream->base);
}

flagcxResult_t ecclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                   size_t count, flagcxDataType_t datatype,
                                   flagcxInnerComm_t comm,
                                   flagcxStream_t stream) {
  return (flagcxResult_t)ecclAlltoall(sendbuff, count, (ecclDataType_t)datatype,
                                      recvbuff, count, (ecclDataType_t)datatype,
                                      comm->base, stream->base);
}

flagcxResult_t ecclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                    size_t *sdispls, void *recvbuff,
                                    size_t *recvcounts, size_t *rdispls,
                                    flagcxDataType_t datatype,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return (flagcxResult_t)ecclAlltoAllv(
      (void *)sendbuff, sendcounts, sdispls, (ecclDataType_t)datatype,
      recvbuff, recvcounts, rdispls, (ecclDataType_t)datatype,
      comm->base, stream->base);
}

flagcxResult_t ecclAdaptorSend(const void *sendbuff, size_t count,
                               flagcxDataType_t datatype, int peer,
                               flagcxInnerComm_t comm, flagcxStream_t stream) {
  return (flagcxResult_t)ecclSend(sendbuff, count, (ecclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

flagcxResult_t ecclAdaptorRecv(void *recvbuff, size_t count,
                               flagcxDataType_t datatype, int peer,
                               flagcxInnerComm_t comm, flagcxStream_t stream) {
  return (flagcxResult_t)ecclRecv(recvbuff, count, (ecclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

flagcxResult_t ecclAdaptorGroupStart() {
  return (flagcxResult_t)ecclGroupStart();
}

flagcxResult_t ecclAdaptorGroupEnd() { return (flagcxResult_t)ecclGroupEnd(); }

struct flagcxCCLAdaptor ecclAdaptor = {
    "ECCL",
    // Basic functions
    ecclAdaptorGetVersion, ecclAdaptorGetUniqueId, ecclAdaptorGetErrorString,
    ecclAdaptorGetLastError,
    // Communicator functions
    ecclAdaptorCommInitRank, ecclAdaptorCommFinalize, ecclAdaptorCommDestroy,
    ecclAdaptorCommAbort, ecclAdaptorCommResume, ecclAdaptorCommSuspend,
    ecclAdaptorCommCount, ecclAdaptorCommCuDevice, ecclAdaptorCommUserRank,
    ecclAdaptorCommGetAsyncError, ecclAdaptorMemAlloc, ecclAdaptorMemFree,
    ecclAdaptorCommRegister, ecclAdaptorCommDeregister,
    // Communication functions
    ecclAdaptorReduce, ecclAdaptorGather, ecclAdaptorScatter,
    ecclAdaptorBroadcast, ecclAdaptorAllReduce, ecclAdaptorReduceScatter,
    ecclAdaptorAllGather, ecclAdaptorAlltoAll, ecclAdaptorAlltoAllv,
    ecclAdaptorSend, ecclAdaptorRecv,
    // Group semantics
    ecclAdaptorGroupStart, ecclAdaptorGroupEnd};

#endif // USE_ENFLAME_ADAPTOR

