#include "sunrise_adaptor.h"

#ifdef USE_SUNRISE_ADAPTOR
#include <map>

// Datatype and reduction op mappings
std::map<flagcxDataType_t, pcclDataType_t> f2p_datatype_map = {
    {flagcxInt8, pcclInt8},        {flagcxUint8, pcclUint8},
    {flagcxInt, pcclInt},          {flagcxInt32, pcclInt32},
    {flagcxUint32, pcclUint32},    {flagcxInt64, pcclInt64},
    {flagcxUint64, pcclUint64},    {flagcxHalf, pcclHalf},
    {flagcxFloat16, pcclFloat16},  {flagcxBfloat16, pcclBfloat16},
    {flagcxFloat32, pcclFloat32},  {flagcxFloat, pcclFloat},
    {flagcxNumTypes, pcclTypesNum}};

std::map<flagcxRedOp_t, pcclRedOp_t> f2p_reduceop_map = {{flagcxSum, pcclSum},
                                                         {flagcxProd, pcclProd},
                                                         {flagcxMax, pcclMax},
                                                         {flagcxMin, pcclMin},
                                                         {flagcxAvg, pcclAvg}};

std::map<pcclResult_t, flagcxResult_t> p2f_ret_map = {
    {pcclSuccess, flagcxSuccess},
    {pcclUnhandledTangError, flagcxUnhandledDeviceError},
    {pcclSystemError, flagcxSystemError},
    {pcclInternalError, flagcxInternalError},
    {pcclInvalidArgument, flagcxInvalidArgument},
    {pcclInvalidUsage, flagcxInvalidUsage},
    {pcclRemoteError, flagcxRemoteError},
    {pcclInProgress, flagcxInProgress},
    // TODO: no match flagcx error for pcclInvalidDeviceIndex, use
    // flagcxUnhandledDeviceError temporarily
    {pcclInvalidDeviceIndex, flagcxUnhandledDeviceError},
    {pccl_NUM_RESULTS, flagcxNumResults}};

std::map<flagcxResult_t, pcclResult_t> f2p_ret_map = {
    {flagcxSuccess, pcclSuccess},
    {flagcxUnhandledDeviceError, pcclUnhandledTangError},
    {flagcxSystemError, pcclSystemError},
    {flagcxInternalError, pcclInternalError},
    {flagcxInvalidArgument, pcclInvalidArgument},
    {flagcxInvalidUsage, pcclInvalidUsage},
    {flagcxRemoteError, pcclRemoteError},
    {flagcxInProgress, pcclInProgress},
    {flagcxNumResults, pccl_NUM_RESULTS}};

flagcxResult_t pcclAdaptorGetVersion(int *version) {
  return (flagcxResult_t)p2f_ret_map[pcclGetVersion(version)];
}

flagcxResult_t pcclAdaptorGetUniqueId(flagcxUniqueId_t *uniqueId) {
  if (*uniqueId == NULL) {
    flagcxCalloc(uniqueId, 1);
  }
  return (
      flagcxResult_t)p2f_ret_map[pcclGetUniqueId((pcclUniqueId *)(*uniqueId))];
}

const char *pcclAdaptorGetErrorString(flagcxResult_t result) {
  return pcclGetErrorString(f2p_ret_map[result]);
}

const char *pcclAdaptorGetLastError(flagcxInnerComm_t comm) {
  return pcclGetLastError(comm->base);
}

// TODO: unsupported
flagcxResult_t pcclAdaptorGetStagedBuffer(const flagcxInnerComm_t comm,
                                          void **buff, size_t size,
                                          int isRecv) {
  return flagcxUnhandledDeviceError;
}

// Communicator functions
flagcxResult_t pcclAdaptorCommInitRank(flagcxInnerComm_t *comm, int nranks,
                                       flagcxUniqueId_t commId, int rank,
                                       bootstrapState * /*bootstrap*/) {
  if (*comm == NULL) {
    flagcxCalloc(comm, 1);
  }
  return (flagcxResult_t)p2f_ret_map[pcclCommInitRank(
      &(*comm)->base, nranks, *(pcclUniqueId *)commId, rank)];
}

// Not exposed in flagcxCCLAdaptor, while pccl has this function
flagcxResult_t pcclAdaptorCommInitAll(flagcxInnerComm_t *comm, int nranks,
                                      const int *devlist) {
  if (*comm == NULL) {
    flagcxCalloc(comm, 1);
  }
  return (flagcxResult_t)
      p2f_ret_map[pcclCommInitAll(&(*comm)->base, nranks, devlist)];
}

// TODO: unsupported
flagcxResult_t pcclAdaptorCommFinalize(flagcxInnerComm_t comm) {
  return flagcxUnhandledDeviceError;
}

flagcxResult_t pcclAdaptorCommDestroy(flagcxInnerComm_t comm) {
  return (flagcxResult_t)p2f_ret_map[pcclCommDestroy(comm->base)];
}

flagcxResult_t pcclAdaptorCommAbort(flagcxInnerComm_t comm) {
  return (flagcxResult_t)p2f_ret_map[pcclCommAbort(comm->base)];
}

// TODO: unsupported
flagcxResult_t pcclAdaptorCommResume(flagcxInnerComm_t comm) {
  return flagcxUnhandledDeviceError;
}

// TODO: unsupported
flagcxResult_t pcclAdaptorCommSuspend(flagcxInnerComm_t comm) {
  return flagcxUnhandledDeviceError;
}

flagcxResult_t pcclAdaptorCommCount(const flagcxInnerComm_t comm, int *count) {
  return (flagcxResult_t)p2f_ret_map[pcclCommCount(comm->base, count)];
}

flagcxResult_t pcclAdaptorCommCuDevice(const flagcxInnerComm_t comm,
                                       int *device) {
  return (flagcxResult_t)p2f_ret_map[pcclCommCuDevice(comm->base, device)];
}

flagcxResult_t pcclAdaptorCommUserRank(const flagcxInnerComm_t comm,
                                       int *rank) {
  return (flagcxResult_t)p2f_ret_map[pcclCommUserRank(comm->base, rank)];
}

flagcxResult_t pcclAdaptorCommGetAsyncError(flagcxInnerComm_t comm,
                                            flagcxResult_t *asyncError) {
  return (flagcxResult_t)
      p2f_ret_map[pcclCommGetAsyncError(comm->base, &f2p_ret_map[*asyncError])];
}

// TODO: unsupported
flagcxResult_t pcclAdaptorMemAlloc(void **ptr, size_t size) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t pcclAdaptorMemFree(void *ptr) { return flagcxNotSupported; }

// TODO: unsupported
flagcxResult_t pcclAdaptorCommRegister(flagcxInnerComm_t comm, void *buff,
                                       size_t size, void **handle) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t pcclAdaptorCommDeregister(flagcxInnerComm_t comm, void *handle) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t pcclAdaptorCommWindowRegister(flagcxInnerComm_t comm, void *buff,
                                             size_t size,
                                             flagcxInnerWindow_t *win,
                                             int winFlags) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t pcclAdaptorCommWindowDeregister(flagcxInnerComm_t comm,
                                               flagcxInnerWindow_t win) {
  return flagcxNotSupported;
}

flagcxResult_t pcclAdaptorReduce(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 flagcxRedOp_t op, int root,
                                 flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
  return (flagcxResult_t)p2f_ret_map[pcclReduce(
      sendbuff, recvbuff, count, f2p_datatype_map[datatype],
      f2p_reduceop_map[op], root, comm->base, stream->base)];
}

// TODO: unsupported
flagcxResult_t pcclAdaptorGather(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 int root, flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t pcclAdaptorScatter(const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  int root, flagcxInnerComm_t comm,
                                  flagcxStream_t stream) {
  return flagcxNotSupported;
}

flagcxResult_t pcclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    int root, flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return (flagcxResult_t)p2f_ret_map[pcclBroadcast(
      sendbuff, recvbuff, count, f2p_datatype_map[datatype], root, comm->base,
      stream->base)];
}

flagcxResult_t pcclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    flagcxRedOp_t op, flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return (flagcxResult_t)p2f_ret_map[pcclAllReduce(
      sendbuff, recvbuff, count, f2p_datatype_map[datatype],
      f2p_reduceop_map[op], comm->base, stream->base)];
}

flagcxResult_t
pcclAdaptorReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                         flagcxDataType_t datatype, flagcxRedOp_t op,
                         flagcxInnerComm_t comm, flagcxStream_t stream) {
  return (flagcxResult_t)p2f_ret_map[pcclReduceScatter(
      sendbuff, recvbuff, recvcount, f2p_datatype_map[datatype],
      f2p_reduceop_map[op], comm->base, stream->base)];
}

flagcxResult_t pcclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                    size_t sendcount, flagcxDataType_t datatype,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return (flagcxResult_t)p2f_ret_map[pcclAllGather(
      sendbuff, recvbuff, sendcount, f2p_datatype_map[datatype], comm->base,
      stream->base)];
}

// TODO: unsupported
flagcxResult_t pcclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                   size_t count, flagcxDataType_t datatype,
                                   flagcxInnerComm_t comm,
                                   flagcxStream_t stream) {
  return flagcxUnhandledDeviceError;
}

// TODO: unsupported
flagcxResult_t pcclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                    size_t *sdispls, void *recvbuff,
                                    size_t *recvcounts, size_t *rdispls,
                                    flagcxDataType_t datatype,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return flagcxUnhandledDeviceError;
}

flagcxResult_t pcclAdaptorSend(const void *sendbuff, size_t count,
                               flagcxDataType_t datatype, int peer,
                               flagcxInnerComm_t comm, flagcxStream_t stream) {
  return (flagcxResult_t)
      p2f_ret_map[pcclSend(sendbuff, count, f2p_datatype_map[datatype], peer,
                           comm->base, stream->base)];
}

flagcxResult_t pcclAdaptorRecv(void *recvbuff, size_t count,
                               flagcxDataType_t datatype, int peer,
                               flagcxInnerComm_t comm, flagcxStream_t stream) {
  return (flagcxResult_t)
      p2f_ret_map[pcclRecv(recvbuff, count, f2p_datatype_map[datatype], peer,
                           comm->base, stream->base)];
}

flagcxResult_t pcclAdaptorGroupStart(void) {
  return (flagcxResult_t)p2f_ret_map[pcclGroupStart()];
}

flagcxResult_t pcclAdaptorGroupEnd(void) {
  return (flagcxResult_t)p2f_ret_map[pcclGroupEnd()];
}

// Adaptor descriptor
struct flagcxCCLAdaptor pcclAdaptor = {
    "PCCL",
    // Basic functions
    pcclAdaptorGetVersion, pcclAdaptorGetUniqueId, pcclAdaptorGetErrorString,
    pcclAdaptorGetLastError, pcclAdaptorGetStagedBuffer,
    // Communicator functions
    pcclAdaptorCommInitRank, pcclAdaptorCommFinalize, pcclAdaptorCommDestroy,
    pcclAdaptorCommAbort, pcclAdaptorCommResume, pcclAdaptorCommSuspend,
    pcclAdaptorCommCount, pcclAdaptorCommCuDevice, pcclAdaptorCommUserRank,
    pcclAdaptorCommGetAsyncError, pcclAdaptorMemAlloc, pcclAdaptorMemFree,
    pcclAdaptorCommRegister, pcclAdaptorCommDeregister,
    // Symmetric functions
    pcclAdaptorCommWindowRegister, pcclAdaptorCommWindowDeregister,
    // Communication functions
    pcclAdaptorReduce, pcclAdaptorGather, pcclAdaptorScatter,
    pcclAdaptorBroadcast, pcclAdaptorAllReduce, pcclAdaptorReduceScatter,
    pcclAdaptorAllGather, pcclAdaptorAlltoAll, pcclAdaptorAlltoAllv,
    pcclAdaptorSend, pcclAdaptorRecv,
    // Group semantics
    pcclAdaptorGroupStart, pcclAdaptorGroupEnd};

#endif // USE_SUNRISE_ADAPTOR
