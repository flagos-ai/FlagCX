#include "ppu_adaptor.h"

#ifdef USE_PPU_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "comm.h"

// Native PCCL enums are not a FlagCX ABI: use explicit conversions, including
// errors introduced by newer SDKs (e.g. timeout) and unknown future values.
static flagcxResult_t ppupcclResultToFlagcx(pcclResult_t result) {
  switch (result) {
    case pcclSuccess:
      return flagcxSuccess;
    case pcclUnhandledHggcError:
      return flagcxUnhandledDeviceError;
    case pcclSystemError:
      return flagcxSystemError;
    case pcclInternalError:
      return flagcxInternalError;
    case pcclInvalidArgument:
      return flagcxInvalidArgument;
    case pcclInvalidUsage:
      return flagcxInvalidUsage;
    case pcclRemoteError:
      return flagcxRemoteError;
    case pcclInProgress:
      return flagcxInProgress;
    default:
      return flagcxUnhandledCCLError;
  }
}

static flagcxResult_t ppupcclDataTypeToNative(flagcxDataType_t type,
                                              pcclDataType_t *native) {
  switch (type) {
    case flagcxInt8:
      *native = pcclInt8;
      break;
    case flagcxUint8:
      *native = pcclUint8;
      break;
    case flagcxInt32:
      *native = pcclInt32;
      break;
    case flagcxUint32:
      *native = pcclUint32;
      break;
    case flagcxInt64:
      *native = pcclInt64;
      break;
    case flagcxUint64:
      *native = pcclUint64;
      break;
    case flagcxFloat16:
      *native = pcclFloat16;
      break;
    case flagcxFloat32:
      *native = pcclFloat32;
      break;
    case flagcxFloat64:
      *native = pcclFloat64;
      break;
    case flagcxBfloat16:
      *native = pcclBfloat16;
      break;
    default:
      return flagcxInvalidArgument;
  }
  return flagcxSuccess;
}

static flagcxResult_t ppupcclRedOpToNative(flagcxRedOp_t op,
                                           pcclRedOp_t *native) {
  switch (op) {
    case flagcxSum:
      *native = pcclSum;
      break;
    case flagcxProd:
      *native = pcclProd;
      break;
    case flagcxMax:
      *native = pcclMax;
      break;
    case flagcxMin:
      *native = pcclMin;
      break;
    case flagcxAvg:
      *native = pcclAvg;
      break;
    default:
      return flagcxInvalidArgument;
  }
  return flagcxSuccess;
}

static_assert(sizeof(pcclUniqueId) <= sizeof(flagcxInnerUniqueId),
              "pcclUniqueId does not fit inside flagcxInnerUniqueId");

flagcxResult_t ppupcclAdaptorGetVersion(int *version) {
  return ppupcclResultToFlagcx(pcclGetVersion(version));
}

flagcxResult_t ppupcclAdaptorGetUniqueId(flagcxInnerUniqueId_t *uniqueId) {
  if (*uniqueId == NULL) {
    FLAGCXCHECK(flagcxCalloc(uniqueId, 1));
  }
  return ppupcclResultToFlagcx(pcclGetUniqueId((pcclUniqueId *)(*uniqueId)));
}

flagcxResult_t ppupcclAdaptorGetStagedBuffer(const flagcxInnerComm_t comm,
                                             void **buff, size_t size,
                                             int isRecv) {
  return flagcxNotSupported;
}

const char *ppupcclAdaptorGetErrorString(flagcxResult_t result) {
  switch (result) {
    case flagcxSuccess:
      return pcclGetErrorString(pcclSuccess);
    case flagcxUnhandledDeviceError:
      return pcclGetErrorString(pcclUnhandledHggcError);
    case flagcxSystemError:
      return pcclGetErrorString(pcclSystemError);
    case flagcxInternalError:
      return pcclGetErrorString(pcclInternalError);
    case flagcxInvalidArgument:
      return pcclGetErrorString(pcclInvalidArgument);
    case flagcxInvalidUsage:
      return pcclGetErrorString(pcclInvalidUsage);
    case flagcxRemoteError:
      return pcclGetErrorString(pcclRemoteError);
    case flagcxInProgress:
      return pcclGetErrorString(pcclInProgress);
    case flagcxNotSupported:
      return "PPU PCCL adaptor operation is not supported";
    default:
      return "PPU PCCL operation failed; consult pcclGetLastError";
  }
}

const char *ppupcclAdaptorGetLastError(flagcxInnerComm_t comm) {
  return pcclGetLastError(comm->base);
}

flagcxResult_t
ppupcclAdaptorCommInitRank(flagcxInnerComm_t *comm, int nranks,
                           flagcxInnerUniqueId_t commId, int rank,
                           struct bootstrapState * /*bootstrap*/) {
  if (*comm == NULL) {
    FLAGCXCHECK(flagcxCalloc(comm, 1));
  }
  return ppupcclResultToFlagcx(
      pcclCommInitRank(&(*comm)->base, nranks, *(pcclUniqueId *)commId, rank));
}

flagcxResult_t ppupcclAdaptorCommFinalize(flagcxInnerComm_t comm) {
  return ppupcclResultToFlagcx(pcclCommFinalize(comm->base));
}

flagcxResult_t ppupcclAdaptorCommDestroy(flagcxInnerComm_t comm) {
  return ppupcclResultToFlagcx(pcclCommDestroy(comm->base));
}

flagcxResult_t ppupcclAdaptorCommAbort(flagcxInnerComm_t comm) {
  return ppupcclResultToFlagcx(pcclCommAbort(comm->base));
}

flagcxResult_t ppupcclAdaptorCommResume(flagcxInnerComm_t comm) {
  return flagcxNotSupported;
}

flagcxResult_t ppupcclAdaptorCommSuspend(flagcxInnerComm_t comm) {
  return flagcxNotSupported;
}

flagcxResult_t ppupcclAdaptorCommCount(const flagcxInnerComm_t comm,
                                       int *count) {
  return ppupcclResultToFlagcx(pcclCommCount(comm->base, count));
}

flagcxResult_t ppupcclAdaptorCommCuDevice(const flagcxInnerComm_t comm,
                                          int *device) {
  return ppupcclResultToFlagcx(pcclCommCuDevice(comm->base, device));
}

flagcxResult_t ppupcclAdaptorCommUserRank(const flagcxInnerComm_t comm,
                                          int *rank) {
  return ppupcclResultToFlagcx(pcclCommUserRank(comm->base, rank));
}

flagcxResult_t ppupcclAdaptorCommGetAsyncError(flagcxInnerComm_t comm,
                                               flagcxResult_t *asyncError) {
  if (comm == nullptr || asyncError == nullptr)
    return flagcxInvalidArgument;
  pcclResult_t nativeError = pcclSuccess;
  const flagcxResult_t result =
      ppupcclResultToFlagcx(pcclCommGetAsyncError(comm->base, &nativeError));
  if (result == flagcxSuccess)
    *asyncError = ppupcclResultToFlagcx(nativeError);
  return result;
}

flagcxResult_t ppupcclAdaptorMemAlloc(void **ptr, size_t size) {
  return flagcxNotSupported;
}

flagcxResult_t ppupcclAdaptorMemFree(void *ptr) { return flagcxNotSupported; }

flagcxResult_t ppupcclAdaptorCommRegister(flagcxInnerComm_t comm, void *buff,
                                          size_t size, void **handle) {
  return flagcxNotSupported;
}

flagcxResult_t ppupcclAdaptorCommDeregister(flagcxInnerComm_t comm,
                                            void *handle) {
  return flagcxNotSupported;
}

flagcxResult_t ppupcclAdaptorCommWindowRegister(flagcxInnerComm_t comm,
                                                void *buff, size_t size,
                                                flagcxInnerWindow_t *win,
                                                int winFlags) {
  return flagcxNotSupported;
}

flagcxResult_t ppupcclAdaptorCommWindowDeregister(flagcxInnerComm_t comm,
                                                  flagcxInnerWindow_t win) {
  return flagcxNotSupported;
}

flagcxResult_t ppupcclAdaptorReduce(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    flagcxRedOp_t op, int root,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  pcclRedOp_t nativeOp;
  FLAGCXCHECK(ppupcclRedOpToNative(op, &nativeOp));
  return ppupcclResultToFlagcx(pcclReduce(sendbuff, recvbuff, count,
                                          nativeDatatype, nativeOp, root,
                                          comm->base, stream->base));
}

flagcxResult_t ppupcclAdaptorGather(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    int root, flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  int rank, nranks;
  FLAGCXCHECK(ppupcclResultToFlagcx(pcclCommUserRank(comm->base, &rank)));
  FLAGCXCHECK(ppupcclResultToFlagcx(pcclCommCount(comm->base, &nranks)));

  size_t size = count * getFlagcxDataTypeSize(datatype);
  char *buffer = static_cast<char *>(recvbuff);

  flagcxResult_t res = flagcxSuccess;
  FLAGCXCHECK(ppupcclResultToFlagcx(pcclGroupStart()));
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      FLAGCXCHECKGOTO(ppupcclResultToFlagcx(
                          pcclRecv(static_cast<void *>(buffer + r * size), size,
                                   pcclChar, r, comm->base, stream->base)),
                      res, group_exit);
    }
  }
  FLAGCXCHECKGOTO(ppupcclResultToFlagcx(pcclSend(sendbuff, size, pcclChar, root,
                                                 comm->base, stream->base)),
                  res, group_exit);
group_exit:
  FLAGCXCHECK(ppupcclResultToFlagcx(pcclGroupEnd()));
  return res;
}

flagcxResult_t ppupcclAdaptorScatter(const void *sendbuff, void *recvbuff,
                                     size_t count, flagcxDataType_t datatype,
                                     int root, flagcxInnerComm_t comm,
                                     flagcxStream_t stream) {
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  int rank, nranks;
  FLAGCXCHECK(ppupcclResultToFlagcx(pcclCommUserRank(comm->base, &rank)));
  FLAGCXCHECK(ppupcclResultToFlagcx(pcclCommCount(comm->base, &nranks)));

  size_t size = count * getFlagcxDataTypeSize(datatype);
  const char *buffer = static_cast<const char *>(sendbuff);

  flagcxResult_t res = flagcxSuccess;
  FLAGCXCHECK(ppupcclResultToFlagcx(pcclGroupStart()));
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      FLAGCXCHECKGOTO(ppupcclResultToFlagcx(pcclSend(
                          static_cast<const void *>(buffer + r * size), size,
                          pcclChar, r, comm->base, stream->base)),
                      res, group_exit);
    }
  }
  FLAGCXCHECKGOTO(ppupcclResultToFlagcx(pcclRecv(recvbuff, size, pcclChar, root,
                                                 comm->base, stream->base)),
                  res, group_exit);
group_exit:
  FLAGCXCHECK(ppupcclResultToFlagcx(pcclGroupEnd()));
  return res;
}

flagcxResult_t ppupcclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                       size_t count, flagcxDataType_t datatype,
                                       int root, flagcxInnerComm_t comm,
                                       flagcxStream_t stream) {
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  return ppupcclResultToFlagcx(pcclBroadcast(sendbuff, recvbuff, count,
                                             nativeDatatype, root, comm->base,
                                             stream->base));
}

flagcxResult_t ppupcclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                       size_t count, flagcxDataType_t datatype,
                                       flagcxRedOp_t op, flagcxInnerComm_t comm,
                                       flagcxStream_t stream) {
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  pcclRedOp_t nativeOp;
  FLAGCXCHECK(ppupcclRedOpToNative(op, &nativeOp));
  return ppupcclResultToFlagcx(pcclAllReduce(sendbuff, recvbuff, count,
                                             nativeDatatype, nativeOp,
                                             comm->base, stream->base));
}

flagcxResult_t
ppupcclAdaptorReduceScatter(const void *sendbuff, void *recvbuff, size_t count,
                            flagcxDataType_t datatype, flagcxRedOp_t op,
                            flagcxInnerComm_t comm, flagcxStream_t stream) {
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  pcclRedOp_t nativeOp;
  FLAGCXCHECK(ppupcclRedOpToNative(op, &nativeOp));
  return ppupcclResultToFlagcx(pcclReduceScatter(sendbuff, recvbuff, count,
                                                 nativeDatatype, nativeOp,
                                                 comm->base, stream->base));
}

flagcxResult_t ppupcclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                       size_t count, flagcxDataType_t datatype,
                                       flagcxInnerComm_t comm,
                                       flagcxStream_t stream) {
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  return ppupcclResultToFlagcx(pcclAllGather(
      sendbuff, recvbuff, count, nativeDatatype, comm->base, stream->base));
}

flagcxResult_t ppupcclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                      size_t count, flagcxDataType_t datatype,
                                      flagcxInnerComm_t comm,
                                      flagcxStream_t stream) {
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  int nranks;
  FLAGCXCHECK(ppupcclResultToFlagcx(pcclCommCount(comm->base, &nranks)));

  size_t size = count * getFlagcxDataTypeSize(datatype);
  const char *sendBuffer = static_cast<const char *>(sendbuff);
  char *recvBuffer = static_cast<char *>(recvbuff);

  flagcxResult_t res = flagcxSuccess;
  FLAGCXCHECK(ppupcclResultToFlagcx(pcclGroupStart()));
  for (int r = 0; r < nranks; r++) {
    FLAGCXCHECKGOTO(ppupcclResultToFlagcx(pcclSend(
                        static_cast<const void *>(sendBuffer + r * size), size,
                        pcclChar, r, comm->base, stream->base)),
                    res, group_exit);
    FLAGCXCHECKGOTO(ppupcclResultToFlagcx(
                        pcclRecv(static_cast<void *>(recvBuffer + r * size),
                                 size, pcclChar, r, comm->base, stream->base)),
                    res, group_exit);
  }
group_exit:
  FLAGCXCHECK(ppupcclResultToFlagcx(pcclGroupEnd()));
  return res;
}

flagcxResult_t ppupcclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                       size_t *sdispls, void *recvbuff,
                                       size_t *recvcounts, size_t *rdispls,
                                       flagcxDataType_t datatype,
                                       flagcxInnerComm_t comm,
                                       flagcxStream_t stream) {
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  int nranks;
  FLAGCXCHECK(ppupcclResultToFlagcx(pcclCommCount(comm->base, &nranks)));

  size_t typeSize = getFlagcxDataTypeSize(datatype);
  const char *sendBuffer = static_cast<const char *>(sendbuff);
  char *recvBuffer = static_cast<char *>(recvbuff);

  flagcxResult_t res = flagcxSuccess;
  FLAGCXCHECK(ppupcclResultToFlagcx(pcclGroupStart()));
  for (int r = 0; r < nranks; r++) {
    if (flagcxCCLAdaptorNeedSendrecv(sendcounts[r])) {
      FLAGCXCHECKGOTO(
          ppupcclResultToFlagcx(pcclSend(
              static_cast<const void *>(sendBuffer + sdispls[r] * typeSize),
              sendcounts[r] * typeSize, pcclChar, r, comm->base, stream->base)),
          res, group_exit);
    }
    if (flagcxCCLAdaptorNeedSendrecv(recvcounts[r])) {
      FLAGCXCHECKGOTO(
          ppupcclResultToFlagcx(pcclRecv(
              static_cast<void *>(recvBuffer + rdispls[r] * typeSize),
              recvcounts[r] * typeSize, pcclChar, r, comm->base, stream->base)),
          res, group_exit);
    }
  }
group_exit:
  FLAGCXCHECK(ppupcclResultToFlagcx(pcclGroupEnd()));
  return res;
}

flagcxResult_t ppupcclAdaptorSend(const void *sendbuff, size_t count,
                                  flagcxDataType_t datatype, int peer,
                                  flagcxInnerComm_t comm,
                                  flagcxStream_t stream) {
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  return ppupcclResultToFlagcx(pcclSend(sendbuff, count, nativeDatatype, peer,
                                        comm->base, stream->base));
}

flagcxResult_t ppupcclAdaptorRecv(void *recvbuff, size_t count,
                                  flagcxDataType_t datatype, int peer,
                                  flagcxInnerComm_t comm,
                                  flagcxStream_t stream) {
  pcclDataType_t nativeDatatype;
  FLAGCXCHECK(ppupcclDataTypeToNative(datatype, &nativeDatatype));
  return ppupcclResultToFlagcx(pcclRecv(recvbuff, count, nativeDatatype, peer,
                                        comm->base, stream->base));
}

flagcxResult_t ppupcclAdaptorGroupStart() {
  return ppupcclResultToFlagcx(pcclGroupStart());
}

flagcxResult_t ppupcclAdaptorGroupEnd() {
  return ppupcclResultToFlagcx(pcclGroupEnd());
}

flagcxResult_t
ppupcclAdaptorDevCommReqsInit(flagcxInnerComm_t /*comm*/,
                              flagcxDevCommRequirements * /*reqs*/) {
  return flagcxNotSupported;
}

flagcxResult_t
ppupcclAdaptorDevCommCreate(flagcxInnerComm_t /*comm*/,
                            const flagcxDevCommRequirements * /*reqs*/,
                            flagcxInnerDevComm_t * /*devComm*/) {
  return flagcxNotSupported;
}

flagcxResult_t ppupcclAdaptorDevCommDestroy(flagcxInnerComm_t /*comm*/,
                                            flagcxInnerDevComm_t /*devComm*/) {
  return flagcxNotSupported;
}

struct flagcxCCLAdaptor ppupcclAdaptor = {
    "PPU_PCCL",
    // Basic functions
    ppupcclAdaptorGetVersion, ppupcclAdaptorGetUniqueId,
    ppupcclAdaptorGetErrorString, ppupcclAdaptorGetLastError,
    ppupcclAdaptorGetStagedBuffer,
    // Communicator functions
    ppupcclAdaptorCommInitRank, ppupcclAdaptorCommFinalize,
    ppupcclAdaptorCommDestroy, ppupcclAdaptorCommAbort,
    ppupcclAdaptorCommResume, ppupcclAdaptorCommSuspend,
    ppupcclAdaptorCommCount, ppupcclAdaptorCommCuDevice,
    ppupcclAdaptorCommUserRank, ppupcclAdaptorCommGetAsyncError,
    ppupcclAdaptorMemAlloc, ppupcclAdaptorMemFree, ppupcclAdaptorCommRegister,
    ppupcclAdaptorCommDeregister,
    // Symmetric functions
    ppupcclAdaptorCommWindowRegister, ppupcclAdaptorCommWindowDeregister,
    // Communication functions
    ppupcclAdaptorReduce, ppupcclAdaptorGather, ppupcclAdaptorScatter,
    ppupcclAdaptorBroadcast, ppupcclAdaptorAllReduce,
    ppupcclAdaptorReduceScatter, ppupcclAdaptorAllGather,
    ppupcclAdaptorAlltoAll, ppupcclAdaptorAlltoAllv, ppupcclAdaptorSend,
    ppupcclAdaptorRecv,
    // Group semantics
    ppupcclAdaptorGroupStart, ppupcclAdaptorGroupEnd,
    // Device API
    ppupcclAdaptorDevCommReqsInit, ppupcclAdaptorDevCommCreate,
    ppupcclAdaptorDevCommDestroy};

#endif // USE_PPU_ADAPTOR
