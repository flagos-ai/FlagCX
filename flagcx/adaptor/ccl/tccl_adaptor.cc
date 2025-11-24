#include "tsmicro_adaptor.h"

#ifdef USE_TSM_ADAPTOR

#include <cstring>
#include <map>

static const std::map<tcclResult_t, flagcxResult_t> tccl_to_flagcx_result_map = {
  {tcclSuccess, flagcxSuccess},
  {tcclUnhandledDeviceError, flagcxUnhandledDeviceError},
  {tcclSystemError, flagcxSystemError},
  {tcclInvalidArgument, flagcxInvalidArgument},
  {tcclInvalidUsage, flagcxInvalidUsage},
  {tcclRemoteError, flagcxRemoteError},
  {tcclInProgress, flagcxInProgress},
  {tcclUnhandledCCLError, flagcxUnhandledCCLError},
  {tcclNotSupported, flagcxNotSupported},
  {tcclNumResults, flagcxNumResults},
  {tcclInternalError, flagcxInternalError}
};

// Data type mapping
static const std::map<flagcxDataType_t, tcclDataType_t> flagcx_to_tccl_datatype_map = {
  {flagcxInt8, tcclInt8},
  {flagcxChar, tcclChar},
  {flagcxUint8, tcclUint8},
  {flagcxInt32, tcclInt32},
  {flagcxInt, tcclInt},
  {flagcxUint32, tcclUint32},
  {flagcxInt64, tcclInt64},
  {flagcxUint64, tcclUint64},
  {flagcxFloat16, tcclFloat16},
  {flagcxHalf, tcclHalf},
  {flagcxFloat32, tcclFloat32},
  {flagcxFloat, tcclFloat},
  {flagcxFloat64, tcclFloat64},
  {flagcxDouble, tcclDouble},
  {flagcxBfloat16, tcclBfloat16},
  {flagcxNumTypes, tcclNumTypes}
};

// Reduction operation mapping
static const std::map<flagcxRedOp_t, tcclRedOp_t> flagcx_to_tccl_redop_map = {
  {flagcxSum, tcclSum},
  {flagcxProd, tcclProd},
  {flagcxMax, tcclMax},
  {flagcxMin, tcclMin},
  {flagcxAvg, tcclAvg},
  {flagcxNumRedOps, tcclNumRedOps},
  {flagcxMaxRedOp, tcclMaxRedOp},
  {flagcxRedNoOp, tcclRedNoOp}
};

// Type conversion functions using maps
static inline flagcxResult_t fromTcclResult(tcclResult_t result) {
  auto it = tccl_to_flagcx_result_map.find(result);
  if (it != tccl_to_flagcx_result_map.end()) {
    return it->second;
  }
  return flagcxInternalError; // Default error if not found
}

static inline tcclDataType_t toTcclDataType(flagcxDataType_t dtype) {
  auto it = flagcx_to_tccl_datatype_map.find(dtype);
  if (it != flagcx_to_tccl_datatype_map.end()) {
    return it->second;
  }
  return tcclNumTypes; // Default enum value if not found
}

static inline tcclRedOp_t toTcclRedOp(flagcxRedOp_t op) {
  auto it = flagcx_to_tccl_redop_map.find(op);
  if (it != flagcx_to_tccl_redop_map.end()) {
    return it->second;
  }
  return tcclRedNoOp; // Default enum value if not found
}

flagcxResult_t tcclAdaptorGetVersion(int *version) {
  tcclResult_t result = tcclGetVersion(version);
  return fromTcclResult(result);
}

flagcxResult_t tcclAdaptorGetUniqueId(flagcxUniqueId_t *uniqueId) {
  if (*uniqueId == NULL) {
    flagcxCalloc(uniqueId, 1);
  }
  tcclResult_t result = tcclGetUniqueId((tcclUniqueId *)(*uniqueId));
  return fromTcclResult(result);
}

const char *tcclAdaptorGetErrorString(flagcxResult_t result) {
  // TODO: supported later
  return "Not Implemented";
}

const char *tcclAdaptorGetLastError(flagcxInnerComm_t comm) {
  if (!comm) {
    return "flagcxInvalidArgument";
  }
  return tcclGetLastError(comm->base);
}

flagcxResult_t tcclAdaptorCommInitRank(flagcxInnerComm_t *comm, int nranks,
                                       flagcxUniqueId_t commId, int rank,
                                       bootstrapState * /*bootstrap*/) {
  if (*comm == NULL) {
    flagcxCalloc(comm, 1);
  }
  tcclResult_t result = tcclCommInitRank(&(*comm)->base, nranks,
                                          *(tcclUniqueId *)commId, rank);
  return fromTcclResult(result);
}

flagcxResult_t tcclAdaptorCommFinalize(flagcxInnerComm_t comm) {
  if (!comm) {
    return flagcxInvalidArgument;
  }
  tcclResult_t result = tcclCommFinalize(comm->base);
  return fromTcclResult(result);
}

flagcxResult_t tcclAdaptorCommDestroy(flagcxInnerComm_t comm) {
  if (!comm) {
    return flagcxInvalidArgument;
  }
  tcclResult_t result = tcclCommDestroy(comm->base);
  return fromTcclResult(result);
}

flagcxResult_t tcclAdaptorCommAbort(flagcxInnerComm_t comm) {
  if (!comm) {
    return flagcxInvalidArgument;
  }
  tcclResult_t result = tcclCommAbort(comm->base);
  return fromTcclResult(result);
}

flagcxResult_t tcclAdaptorCommResume(flagcxInnerComm_t comm) {
  if (!comm) {
    return flagcxInvalidArgument;
  }
  tcclResult_t result = tcclCommResume(comm->base);
  return fromTcclResult(result);
}

flagcxResult_t tcclAdaptorCommSuspend(flagcxInnerComm_t comm) {
  if (!comm) {
    return flagcxInvalidArgument;
  }
  tcclResult_t result = tcclCommSuspend(comm->base);
  return fromTcclResult(result);
}

flagcxResult_t tcclAdaptorCommCount(const flagcxInnerComm_t comm, int *count) {
  if (!comm) {
    return flagcxInvalidArgument;
  }
  tcclResult_t result = tcclCommCount(comm->base, count);
  return fromTcclResult(result);
}

flagcxResult_t tcclAdaptorCommCuDevice(const flagcxInnerComm_t comm,
                                       int *device) {
  if (!comm) {
    return flagcxInvalidArgument;
  }
  tcclResult_t result = tcclCommGetDeviceNumber(comm->base, device);
  return fromTcclResult(result);
}

flagcxResult_t tcclAdaptorCommUserRank(const flagcxInnerComm_t comm,
                                       int *rank) {
  if (!comm) {
    return flagcxInvalidArgument;
  }
  tcclResult_t result = tcclCommUserRank(comm->base, rank);
  return fromTcclResult(result);
}

flagcxResult_t tcclAdaptorCommGetAsyncError(flagcxInnerComm_t comm,
                                            flagcxResult_t *asyncError) {
  if (!comm) {
    return flagcxInvalidArgument;
  }
  tcclResult_t tccl_error;
  tcclResult_t result = tcclCommGetAsyncError(comm->base, &tccl_error);
  if (asyncError) {
    *asyncError = fromTcclResult(tccl_error);
  }
  return fromTcclResult(result);
}

flagcxResult_t tcclAdaptorMemAlloc(void **ptr, size_t size) {
  return flagcxNotSupported;
}

flagcxResult_t tcclAdaptorMemFree(void *ptr) {
  return flagcxNotSupported;
}

flagcxResult_t tcclAdaptorCommRegister(const flagcxInnerComm_t comm, void *buff,
                                       size_t size, void **handle) {
  if (!comm) {
    return flagcxInvalidArgument;
  }
  tcclResult_t result = tcclCommRegister(comm->base, buff, size, handle);
  return fromTcclResult(result);
}

flagcxResult_t tcclAdaptorCommDeregister(const flagcxInnerComm_t comm,
                                         void *handle) {
  if (!comm) {
    return flagcxInvalidArgument;
  }
  tcclResult_t result = tcclCommDeregister(comm->base, handle);
  return fromTcclResult(result);
}

flagcxResult_t tcclAdaptorReduce(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 flagcxRedOp_t op, int root,
                                 flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
  if (!comm || !stream) {
    return flagcxInvalidArgument;
  }
  tcclResult_t result = tcclReduce(sendbuff, recvbuff, count, toTcclDataType(datatype),
                                    toTcclRedOp(op), root, comm->base, stream->base);
  return fromTcclResult(result);
}

flagcxResult_t tcclAdaptorGather(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 int root, flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
    if (!comm || !stream) {
    return flagcxInvalidArgument;
  }
  tcclResult_t result = tcclGather(sendbuff, recvbuff, count, toTcclDataType(datatype),
                                    root, comm->base, stream->base);
  return fromTcclResult(result);
}

flagcxResult_t tcclAdaptorScatter(const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  int root, flagcxInnerComm_t comm,
                                  flagcxStream_t stream) {
  if (!comm || !stream) {
    return flagcxInvalidArgument;
  }
  tcclResult_t result = tcclScatter(sendbuff, recvbuff, count, toTcclDataType(datatype),
                                     root, comm->base, stream->base);
  return fromTcclResult(result);
}

flagcxResult_t tcclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    int root, flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  if (!comm || !stream) {
    return flagcxInvalidArgument;
  }
  tcclResult_t result = tcclBroadcast(sendbuff, recvbuff, count, toTcclDataType(datatype),
                                       root, comm->base, stream->base);
  return fromTcclResult(result);
}

flagcxResult_t tcclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    flagcxRedOp_t op, flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  if (!comm || !stream) {
    return flagcxInvalidArgument;
  }
  tcclResult_t result = tcclAllReduce(sendbuff, recvbuff, count, toTcclDataType(datatype),
                                       toTcclRedOp(op), comm->base, stream->base);
  return fromTcclResult(result);
}

flagcxResult_t
tcclAdaptorReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                         flagcxDataType_t datatype, flagcxRedOp_t op,
                         flagcxInnerComm_t comm, flagcxStream_t stream) {
  if (!comm || !stream) {
    return flagcxInvalidArgument;
  }
  tcclResult_t result = tcclReduceScatter(sendbuff, recvbuff, recvcount,
                                           toTcclDataType(datatype), toTcclRedOp(op),
                                           comm->base, stream->base);
  return fromTcclResult(result);
}

flagcxResult_t tcclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                    size_t sendcount, flagcxDataType_t datatype,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  if (!comm || !stream) {
    return flagcxInvalidArgument;
  }
  tcclResult_t result = tcclAllGather(sendbuff, recvbuff, sendcount,
                                       toTcclDataType(datatype), comm->base, stream->base);
  return fromTcclResult(result);
}

flagcxResult_t tcclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                   size_t count, flagcxDataType_t datatype,
                                   flagcxInnerComm_t comm,
                                   flagcxStream_t stream) {
  if (!comm || !stream) {
    return flagcxInvalidArgument;
  }
  tcclResult_t result = tcclAlltoAll(sendbuff, recvbuff, count, toTcclDataType(datatype),
                                      comm->base, stream->base);
  return fromTcclResult(result);
}

flagcxResult_t tcclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                    size_t *sdispls, void *recvbuff,
                                    size_t *recvcounts, size_t *rdispls,
                                    flagcxDataType_t datatype,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  if (!comm || !stream) {
    return flagcxInvalidArgument;
  }
  tcclResult_t result = tcclAlltoAllv(sendbuff, sendcounts, sdispls, recvbuff,
                                       recvcounts, rdispls, toTcclDataType(datatype),
                                       comm->base, stream->base);
  return fromTcclResult(result);
}

flagcxResult_t tcclAdaptorSend(const void *sendbuff, size_t count,
                               flagcxDataType_t datatype, int peer,
                               flagcxInnerComm_t comm, flagcxStream_t stream) {
  if (!comm || !stream) {
    return flagcxInvalidArgument;
  }
  tcclResult_t result = tcclSend(sendbuff, count, toTcclDataType(datatype), peer,
                                  comm->base, stream->base);
  return fromTcclResult(result);
}

flagcxResult_t tcclAdaptorRecv(void *recvbuff, size_t count,
                               flagcxDataType_t datatype, int peer,
                               flagcxInnerComm_t comm, flagcxStream_t stream) {
  if (!comm || !stream) {
    return flagcxInvalidArgument;
  }
  tcclResult_t result = tcclRecv(recvbuff, count, toTcclDataType(datatype), peer,
                                  comm->base, stream->base);
  return fromTcclResult(result);
}

flagcxResult_t tcclAdaptorGroupStart() {
  tcclResult_t result = tcclGroupStart();
  return fromTcclResult(result);
}

flagcxResult_t tcclAdaptorGroupEnd() {
  tcclResult_t result = tcclGroupEnd();
  return fromTcclResult(result);
}

struct flagcxCCLAdaptor tcclAdaptor = {
    "TCCL",
    // Basic functions
    tcclAdaptorGetVersion, tcclAdaptorGetUniqueId, tcclAdaptorGetErrorString,
    tcclAdaptorGetLastError,
    // Communicator functions
    tcclAdaptorCommInitRank, tcclAdaptorCommFinalize, tcclAdaptorCommDestroy,
    tcclAdaptorCommAbort, tcclAdaptorCommResume, tcclAdaptorCommSuspend,
    tcclAdaptorCommCount, tcclAdaptorCommCuDevice, tcclAdaptorCommUserRank,
    tcclAdaptorCommGetAsyncError, tcclAdaptorMemAlloc, tcclAdaptorMemFree,
    tcclAdaptorCommRegister, tcclAdaptorCommDeregister,
    // Communication functions
    tcclAdaptorReduce, tcclAdaptorGather, tcclAdaptorScatter,
    tcclAdaptorBroadcast, tcclAdaptorAllReduce, tcclAdaptorReduceScatter,
    tcclAdaptorAllGather, tcclAdaptorAlltoAll, tcclAdaptorAlltoAllv,
    tcclAdaptorSend, tcclAdaptorRecv,
    // Group semantics
    tcclAdaptorGroupStart, tcclAdaptorGroupEnd};

#endif // USE_TSM_ADAPTOR