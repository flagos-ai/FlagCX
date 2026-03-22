#ifdef USE_CAMBRICON_ADAPTOR

#include "cambricon_adaptor.h"
#include <map>

std::map<flagcxDataType_t, cnclDataType_t> f2c_datatype_map = {
    {flagcxInt8, cnclInt8},       {flagcxUint8, cnclUint8},
    {flagcxInt, cnclInt},         {flagcxInt32, cnclInt32},
    {flagcxInt64, cnclInt64},     {flagcxHalf, cnclHalf},
    {flagcxFloat16, cnclFloat16}, {flagcxBfloat16, cnclBfloat16},
    {flagcxFloat32, cnclFloat32}, {flagcxFloat, cnclFloat},
    {flagcxDouble, cnclFloat},
};

std::map<flagcxRedOp_t, cnclReduceOp_t> f2c_reduceop_map = {
    {flagcxSum, cnclSum},
    {flagcxProd, cnclProd},
    {flagcxMax, cnclMax},
    {flagcxMin, cnclMin}};

// TODO: not match fully
std::map<cnclResult_t, flagcxResult_t> c2f_ret_map = {
    {CNCL_RET_SUCCESS, flagcxSuccess},
    {CNCL_RET_ERR_UNSUPPORTED, flagcxUnhandledDeviceError},
    {CNCL_RET_ASYNC_ERROR, flagcxRemoteError}};

std::map<flagcxResult_t, cnclResult_t> f2c_ret_map = {
    {flagcxSuccess, CNCL_RET_SUCCESS},
    {flagcxUnhandledDeviceError, CNCL_RET_ERR_UNSUPPORTED}};

// TODO: unsupported
flagcxResult_t cnclAdaptorGetVersion(int *version) {
  // return (flagcxResult_t)cnclGetVersion(version);
  return flagcxUnhandledDeviceError;
}

flagcxResult_t cnclAdaptorGetUniqueId(flagcxUniqueId_t *uniqueId) {
  if (*uniqueId == NULL) {
    flagcxCalloc(uniqueId, 1);
  }
  return (
      flagcxResult_t)c2f_ret_map[cnclGetCliqueId((cnclCliqueId *)(*uniqueId))];
}

flagcxResult_t cnclAdaptorGetStagedBuffer(const flagcxInnerComm_t comm,
                                          void **buff, size_t size,
                                          int isRecv) {
  return flagcxNotSupported;
}

const char *cnclAdaptorGetErrorString(flagcxResult_t result) {
  return cnclGetErrorStr((cnclResult_t)f2c_ret_map[result]);
}

// TODO: unsupported
const char *cnclAdaptorGetLastError(flagcxInnerComm_t comm) {
  // return cnclGetLastError(comm->base);
  return "Not Implemented";
}

flagcxResult_t cnclAdaptorCommInitRank(flagcxInnerComm_t *comm, int nranks,
                                       flagcxUniqueId_t commId, int rank,
                                       bootstrapState * /*bootstrap*/) {
  if (*comm == NULL) {
    flagcxCalloc(comm, 1);
  }
  int devId = 0;
  DEVCHECK(cnrtGetDevice(&devId));
  return (flagcxResult_t)c2f_ret_map[cnclInitComms(
      &(*comm)->base, 1 /*num_comm*/, &devId /*dev_list*/, &rank /*rank_list*/,
      nranks, (cnclCliqueId *)commId)];
}

// TODO: unsupported
flagcxResult_t cnclAdaptorCommFinalize(flagcxInnerComm_t comm) {
  // return (flagcxResult_t)cnclCommFinalize(comm->base);
  return flagcxUnhandledDeviceError;
}

flagcxResult_t cnclAdaptorCommDestroy(flagcxInnerComm_t comm) {
  return (flagcxResult_t)
      c2f_ret_map[cnclDestroyComms(&(comm->base), 1 /*num_comm*/)];
}

flagcxResult_t cnclAdaptorCommAbort(flagcxInnerComm_t comm) {
  return (flagcxResult_t)c2f_ret_map[cnclAbortComm(comm->base)];
}

// TODO: not match
flagcxResult_t cnclAdaptorCommResume(flagcxInnerComm_t comm) {
  // return (flagcxResult_t)ncclInvalidUsage;
  return (flagcxResult_t)c2f_ret_map[CNCL_RET_ERR_ARGUMENTS];
}

// TODO: not match
flagcxResult_t cnclAdaptorCommSuspend(flagcxInnerComm_t comm) {
  // return (flagcxResult_t)ncclInvalidUsage;
  return (flagcxResult_t)c2f_ret_map[CNCL_RET_ERR_ARGUMENTS];
}

flagcxResult_t cnclAdaptorCommCount(const flagcxInnerComm_t comm, int *count) {
  return (flagcxResult_t)c2f_ret_map[cnclGetCommCount(count, comm->base)];
}

flagcxResult_t cnclAdaptorCommCuDevice(const flagcxInnerComm_t comm,
                                       int *device) {
  return (flagcxResult_t)c2f_ret_map[cnclGetCommDevice(device, comm->base)];
}

flagcxResult_t cnclAdaptorCommUserRank(const flagcxInnerComm_t comm,
                                       int *rank) {
  return (flagcxResult_t)c2f_ret_map[cnclGetCommRank(rank, comm->base)];
}

flagcxResult_t cnclAdaptorCommGetAsyncError(flagcxInnerComm_t comm,
                                            flagcxResult_t *asyncError) {
  *asyncError = c2f_ret_map[cnclGetCommAsyncError(comm->base)];
  return flagcxSuccess;
}

// TODO: unsupported
flagcxResult_t cnclAdaptorMemAlloc(void **ptr, size_t size) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t cnclAdaptorMemFree(void *ptr) { return flagcxNotSupported; }

// TODO: unsupported
flagcxResult_t cnclAdaptorCommRegister(flagcxInnerComm_t comm, void *buff,
                                       size_t size, void **handle) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t cnclAdaptorCommDeregister(flagcxInnerComm_t comm, void *handle) {
  return flagcxNotSupported;
}

flagcxResult_t cnclAdaptorCommWindowRegister(flagcxInnerComm_t comm, void *buff,
                                             size_t size, flagcxWindow_t *win,
                                             int winFlags) {
  return flagcxNotSupported;
}

flagcxResult_t cnclAdaptorCommWindowDeregister(flagcxInnerComm_t comm,
                                               flagcxWindow_t win) {
  return flagcxNotSupported;
}

flagcxResult_t cnclAdaptorReduce(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 flagcxRedOp_t op, int root,
                                 flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
  return (flagcxResult_t)c2f_ret_map[cnclReduce(
      sendbuff, recvbuff, count, (cnclDataType_t)f2c_datatype_map[datatype],
      (cnclReduceOp_t)f2c_reduceop_map[op], root, comm->base, stream->base)];
}

flagcxResult_t cnclAdaptorGather(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 int root, flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
  int rank, nranks;
  cnclResult_t res = CNCL_RET_SUCCESS;
  res = cnclGetCommRank(&rank, comm->base);
  res = cnclGetCommCount(&nranks, comm->base);

  size_t size = count * getFlagcxDataTypeSize(datatype);
  char *buffer = static_cast<char *>(recvbuff);

  res = cnclGroupStart();
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      res = cnclRecv(static_cast<void *>(buffer + r * size), size, cnclChar, r,
                     comm->base, stream->base);
    }
  }
  res = cnclSend(const_cast<void *>(sendbuff), size, cnclChar, root, comm->base,
                 stream->base);
  res = cnclGroupEnd();

  return (flagcxResult_t)c2f_ret_map[res];
}

flagcxResult_t cnclAdaptorScatter(const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  int root, flagcxInnerComm_t comm,
                                  flagcxStream_t stream) {
  int rank, nranks;
  cnclResult_t res = CNCL_RET_SUCCESS;
  res = cnclGetCommRank(&rank, comm->base);
  res = cnclGetCommCount(&nranks, comm->base);

  size_t size = count * getFlagcxDataTypeSize(datatype);
  const char *buffer = static_cast<const char *>(sendbuff);

  res = cnclGroupStart();
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      res = cnclSend(
          const_cast<void *>(static_cast<const void *>(buffer + r * size)),
          size, cnclChar, r, comm->base, stream->base);
    }
  }
  res = cnclRecv(recvbuff, size, cnclChar, root, comm->base, stream->base);
  res = cnclGroupEnd();

  return (flagcxResult_t)c2f_ret_map[res];
}

flagcxResult_t cnclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    int root, flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return (flagcxResult_t)c2f_ret_map[cnclBroadcast(
      sendbuff, recvbuff, count, (cnclDataType_t)f2c_datatype_map[datatype],
      root, comm->base, stream->base)];
}

flagcxResult_t cnclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    flagcxRedOp_t op, flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return (flagcxResult_t)c2f_ret_map[cnclAllReduce(
      sendbuff, recvbuff, count, (cnclDataType_t)f2c_datatype_map[datatype],
      (cnclReduceOp_t)f2c_reduceop_map[op], comm->base, stream->base)];
}

flagcxResult_t
cnclAdaptorReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                         flagcxDataType_t datatype, flagcxRedOp_t op,
                         flagcxInnerComm_t comm, flagcxStream_t stream) {
  return (flagcxResult_t)c2f_ret_map[cnclReduceScatter(
      sendbuff, recvbuff, recvcount, (cnclDataType_t)f2c_datatype_map[datatype],
      (cnclReduceOp_t)f2c_reduceop_map[op], comm->base, stream->base)];
}

flagcxResult_t cnclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                    size_t sendcount, flagcxDataType_t datatype,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return (flagcxResult_t)c2f_ret_map[cnclAllGather(
      sendbuff, recvbuff, sendcount, (cnclDataType_t)f2c_datatype_map[datatype],
      comm->base, stream->base)];
}

flagcxResult_t cnclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                   size_t count, flagcxDataType_t datatype,
                                   flagcxInnerComm_t comm,
                                   flagcxStream_t stream) {
  int rank, nranks;
  cnclResult_t res = CNCL_RET_SUCCESS;
  res = cnclGetCommRank(&rank, comm->base);
  res = cnclGetCommCount(&nranks, comm->base);

  size_t size = count * getFlagcxDataTypeSize(datatype);
  const char *bufferIn = static_cast<const char *>(sendbuff);
  char *bufferOut = static_cast<char *>(recvbuff);

  res = cnclGroupStart();
  for (int r = 0; r < nranks; r++) {
    res = cnclSend(
        const_cast<void *>(static_cast<const void *>(bufferIn + r * size)),
        size, cnclChar, r, comm->base, stream->base);
    res = cnclRecv(static_cast<void *>(bufferOut + r * size), size, cnclChar, r,
                   comm->base, stream->base);
  }
  res = cnclGroupEnd();

  return (flagcxResult_t)c2f_ret_map[res];
}

flagcxResult_t cnclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                    size_t *sdispls, void *recvbuff,
                                    size_t *recvcounts, size_t *rdispls,
                                    flagcxDataType_t datatype,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  int nranks;
  cnclResult_t res = CNCL_RET_SUCCESS;
  res = cnclGetCommCount(&nranks, comm->base);

  size_t size = getFlagcxDataTypeSize(datatype);
  const char *bufferIn = static_cast<const char *>(sendbuff);
  char *bufferOut = static_cast<char *>(recvbuff);

  res = cnclGroupStart();
  for (int r = 0; r < nranks; r++) {
    if (flagcxCCLAdaptorNeedSendrecv(sendcounts[r])) {
      res = cnclSend(const_cast<void *>(static_cast<const void *>(
                         bufferIn + sdispls[r] * size)),
                     sendcounts[r], f2c_datatype_map[datatype], r, comm->base,
                     stream->base);
    }
    if (flagcxCCLAdaptorNeedSendrecv(recvcounts[r])) {
      res = cnclRecv(static_cast<void *>(bufferOut + rdispls[r] * size),
                     recvcounts[r], f2c_datatype_map[datatype], r, comm->base,
                     stream->base);
    }
  }
  res = cnclGroupEnd();

  return (flagcxResult_t)c2f_ret_map[res];
}

flagcxResult_t cnclAdaptorSend(const void *sendbuff, size_t count,
                               flagcxDataType_t datatype, int peer,
                               flagcxInnerComm_t comm, flagcxStream_t stream) {
  // TODO: const_cast will be removed in the future
  return (flagcxResult_t)
      c2f_ret_map[cnclSend(const_cast<void *>(sendbuff), count,
                           (cnclDataType_t)f2c_datatype_map[datatype], peer,
                           comm->base, stream->base)];
}

flagcxResult_t cnclAdaptorRecv(void *recvbuff, size_t count,
                               flagcxDataType_t datatype, int peer,
                               flagcxInnerComm_t comm, flagcxStream_t stream) {
  return (flagcxResult_t)c2f_ret_map[cnclRecv(
      recvbuff, count, (cnclDataType_t)f2c_datatype_map[datatype], peer,
      comm->base, stream->base)];
}

flagcxResult_t cnclAdaptorGroupStart() {
  return (flagcxResult_t)c2f_ret_map[cnclGroupStart()];
}

flagcxResult_t cnclAdaptorGroupEnd() {
  return (flagcxResult_t)c2f_ret_map[cnclGroupEnd()];
}

struct flagcxCCLAdaptor cnclAdaptor = {
    "CNCL",
    // Basic functions
    cnclAdaptorGetVersion, cnclAdaptorGetUniqueId, cnclAdaptorGetErrorString,
    cnclAdaptorGetLastError, cnclAdaptorGetStagedBuffer,
    // Communicator functions
    cnclAdaptorCommInitRank, cnclAdaptorCommFinalize, cnclAdaptorCommDestroy,
    cnclAdaptorCommAbort, cnclAdaptorCommResume, cnclAdaptorCommSuspend,
    cnclAdaptorCommCount, cnclAdaptorCommCuDevice, cnclAdaptorCommUserRank,
    cnclAdaptorCommGetAsyncError, cnclAdaptorMemAlloc, cnclAdaptorMemFree,
    cnclAdaptorCommRegister, cnclAdaptorCommDeregister,
    // Symmetric functions
    cnclAdaptorCommWindowRegister, cnclAdaptorCommWindowDeregister,
    // Communication functions
    cnclAdaptorReduce, cnclAdaptorGather, cnclAdaptorScatter,
    cnclAdaptorBroadcast, cnclAdaptorAllReduce, cnclAdaptorReduceScatter,
    cnclAdaptorAllGather, cnclAdaptorAlltoAll, cnclAdaptorAlltoAllv,
    cnclAdaptorSend, cnclAdaptorRecv,
    // Group semantics
    cnclAdaptorGroupStart, cnclAdaptorGroupEnd};

#endif // USE_CAMBRICON_ADAPTOR
