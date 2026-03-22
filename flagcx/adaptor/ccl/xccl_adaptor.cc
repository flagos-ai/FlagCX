#include "kunlunxin_adaptor.h"
#include <iostream>

#ifdef USE_KUNLUNXIN_ADAPTOR

BKCLDataType flagcxToXcclDataType(flagcxDataType_t type) {
  // use BKCL_UINT8 as unknown data type
  static const struct {
    flagcxDataType_t flagcxType;
    BKCLDataType bkclType;
  } typeMap[] = {
      {flagcxInt8, BKCL_UINT8},        {flagcxChar, BKCL_UINT8},
      {flagcxUint8, BKCL_UINT8},       {flagcxInt32, BKCL_INT32},
      {flagcxInt, BKCL_INT32},         {flagcxUint32, BKCL_INT32},
      {flagcxUint64, BKCL_INT64},      {flagcxInt64, BKCL_INT64},
      {flagcxFloat16, BKCL_FLOAT16},   {flagcxHalf, BKCL_FLOAT16},
      {flagcxFloat32, BKCL_FLOAT},     {flagcxFloat, BKCL_FLOAT},
      {flagcxFloat64, BKCL_FLOAT64},   {flagcxDouble, BKCL_FLOAT64},
      {flagcxBfloat16, BKCL_BFLOAT16},
  };

  const size_t mapSize = sizeof(typeMap) / sizeof(typeMap[0]);

  for (size_t i = 0; i < mapSize; ++i) {
    if (typeMap[i].flagcxType == type) {
      return typeMap[i].bkclType;
    }
  }

  // return unknown data type if not found
  return BKCL_UINT8;
}

BKCLOp flagcxRedOpToBKCLOp(flagcxRedOp_t op) {
  switch (op) {
    case flagcxSum:
      return BKCLOp::BKCL_ADD;
    case flagcxProd:
      return BKCLOp::BKCL_PRODUCT;
    case flagcxMax:
      return BKCLOp::BKCL_MAX;
    case flagcxMin:
      return BKCLOp::BKCL_MIN;
    default:
      // return BKCLOp::BKCL_NUM_OPS to account for unknown redOp type
      return BKCLOp::BKCL_NUM_OPS;
  }
}

// Unsupported
flagcxResult_t xcclAdaptorGetVersion(int *version) {
  return flagcxNotSupported;
}

flagcxResult_t xcclAdaptorGetUniqueId(flagcxUniqueId_t *uniqueId) {
  if (*uniqueId == NULL) {
    flagcxCalloc(uniqueId, 1);
  }
  // Note that when performing heterogeneous communication between Kunlunxin and
  // other devices, XCCL must be used to generate the unique ID.
  return (flagcxResult_t)bkcl_get_unique_id(
      (BKCLUniqueId *)(((char *)*uniqueId) + sizeof(int)));
}

flagcxResult_t xcclAdaptorGetStagedBuffer(const flagcxInnerComm_t comm,
                                          void **buff, size_t size,
                                          int isRecv) {
  return flagcxNotSupported;
}

// Unsupported
const char *xcclAdaptorGetErrorString(flagcxResult_t result) {
  return "flagcxNotSupported";
}

// Unsupported
const char *xcclAdaptorGetLastError(flagcxInnerComm_t comm) {
  return "flagcxNotSupported";
}

flagcxResult_t xcclAdaptorCommInitRank(flagcxInnerComm_t *comm, int nranks,
                                       flagcxUniqueId_t commId, int rank,
                                       bootstrapState * /*bootstrap*/) {
  if (*comm == NULL) {
    flagcxCalloc(comm, 1);
  }
  return (flagcxResult_t)bkcl_init_rank(
      &(*comm)->base, rank, nranks,
      (BKCLUniqueId *)((char *)commId + sizeof(int)));
}

// Unsupported
flagcxResult_t xcclAdaptorCommFinalize(flagcxInnerComm_t comm) {
  return flagcxNotSupported;
}

flagcxResult_t xcclAdaptorCommDestroy(flagcxInnerComm_t comm) {
  return (flagcxResult_t)bkcl_destroy_context(comm->base);
}

flagcxResult_t xcclAdaptorCommAbort(flagcxInnerComm_t comm) {
  return (flagcxResult_t)bkcl_comm_abort(comm->base);
}

// Unsupported
flagcxResult_t xcclAdaptorCommResume(flagcxInnerComm_t comm) {
  return flagcxNotSupported;
}

// Unsupported
flagcxResult_t xcclAdaptorCommSuspend(flagcxInnerComm_t comm) {
  return flagcxNotSupported;
}

flagcxResult_t xcclAdaptorCommCount(const flagcxInnerComm_t comm, int *count) {
  return (flagcxResult_t)bkcl_comm_count(comm->base, count);
}

// Unsupported
flagcxResult_t xcclAdaptorCommCuDevice(const flagcxInnerComm_t comm,
                                       int *device) {
  return flagcxNotSupported;
}

flagcxResult_t xcclAdaptorCommUserRank(const flagcxInnerComm_t comm,
                                       int *rank) {
  return (flagcxResult_t)bkcl_comm_user_rank(comm->base, rank);
}

// TODO: unsupported
flagcxResult_t xcclAdaptorMemAlloc(void **ptr, size_t size) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t xcclAdaptorMemFree(void *ptr) { return flagcxNotSupported; }

// Unsupported
flagcxResult_t xcclAdaptorCommGetAsyncError(flagcxInnerComm_t comm,
                                            flagcxResult_t *asyncError) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t xcclAdaptorCommRegister(flagcxInnerComm_t comm, void *buff,
                                       size_t size, void **handle) {
  return flagcxNotSupported;
}

flagcxResult_t xcclAdaptorCommWindowRegister(flagcxInnerComm_t comm, void *buff,
                                             size_t size, flagcxWindow_t *win,
                                             int winFlags) {
  return flagcxNotSupported;
}

flagcxResult_t xcclAdaptorCommWindowDeregister(flagcxInnerComm_t comm,
                                               flagcxWindow_t win) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t xcclAdaptorCommDeregister(flagcxInnerComm_t comm, void *handle) {
  return flagcxNotSupported;
}

flagcxResult_t xcclAdaptorReduce(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 flagcxRedOp_t op, int root,
                                 flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
  return (flagcxResult_t)bkcl_reduce(
      comm->base, sendbuff, recvbuff, count, flagcxToXcclDataType(datatype),
      flagcxRedOpToBKCLOp(op), root, stream->base);
}

// Unsupported
flagcxResult_t xcclAdaptorGather(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 int root, flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
  return flagcxNotSupported;
}

flagcxResult_t xcclAdaptorScatter(const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  int root, flagcxInnerComm_t comm,
                                  flagcxStream_t stream) {
  int rank, nranks;
  BKCLResult_t res = BKCL_SUCCESS;
  res = bkcl_comm_user_rank(comm->base, &rank);
  res = bkcl_comm_count(comm->base, &nranks);

  size_t size = count * getFlagcxDataTypeSize(datatype);
  const char *buffer = static_cast<const char *>(sendbuff);

  res = bkcl_group_start();
  if (rank == root) {
    for (int r = 0; r < nranks; r++) {
      res = bkcl_send(comm->base, static_cast<const void *>(buffer + r * size),
                      size, r, BKCL_UINT8, stream->base);
    }
  }
  res = bkcl_recv(comm->base, recvbuff, size, root, BKCL_UINT8, stream->base);
  res = bkcl_group_end();

  return (flagcxResult_t)res;
}

flagcxResult_t xcclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    int root, flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return (flagcxResult_t)bkcl_broadcast(comm->base, sendbuff, recvbuff, count,
                                        flagcxToXcclDataType(datatype), root,
                                        stream->base);
}

flagcxResult_t xcclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    flagcxRedOp_t op, flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return (flagcxResult_t)bkcl_all_reduce(comm->base, sendbuff, recvbuff, count,
                                         flagcxToXcclDataType(datatype),
                                         flagcxRedOpToBKCLOp(op), stream->base);
}

flagcxResult_t
xcclAdaptorReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                         flagcxDataType_t datatype, flagcxRedOp_t op,
                         flagcxInnerComm_t comm, flagcxStream_t stream) {
  return (flagcxResult_t)bkcl_reduce_scatter(
      comm->base, sendbuff, recvbuff, recvcount, flagcxToXcclDataType(datatype),
      flagcxRedOpToBKCLOp(op), stream->base);
}

flagcxResult_t xcclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                    size_t sendcount, flagcxDataType_t datatype,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return (flagcxResult_t)bkcl_all_gather(
      comm->base, sendbuff, sendcount, recvbuff, flagcxToXcclDataType(datatype),
      stream->base);
}

flagcxResult_t xcclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                   size_t count, flagcxDataType_t datatype,
                                   flagcxInnerComm_t comm,
                                   flagcxStream_t stream) {
  return (flagcxResult_t)bkcl_all_to_all(comm->base, sendbuff, count, recvbuff,
                                         flagcxToXcclDataType(datatype),
                                         stream->base);
}

flagcxResult_t xcclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                    size_t *sdispls, void *recvbuff,
                                    size_t *recvcounts, size_t *rdispls,
                                    flagcxDataType_t datatype,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  int nranks;
  bkcl_comm_count(comm->base, &nranks);

  size_t *sendcountsDev = NULL;
  size_t *sdisplsDev = NULL;
  size_t *recvcountsDev = NULL;
  size_t *rdisplsDev = NULL;

  xpu_malloc((void **)(&sendcountsDev), nranks * sizeof(size_t));
  xpu_malloc((void **)(&sdisplsDev), nranks * sizeof(size_t));
  xpu_malloc((void **)(&recvcountsDev), nranks * sizeof(size_t));
  xpu_malloc((void **)(&rdisplsDev), nranks * sizeof(size_t));
  xpu_memcpy_async((void *)sendcountsDev, (void *)sendcounts,
                   nranks * sizeof(size_t), XPUMemcpyKind::XPU_HOST_TO_DEVICE,
                   stream->base);
  xpu_memcpy_async((void *)sdisplsDev, (void *)sdispls, nranks * sizeof(size_t),
                   XPUMemcpyKind::XPU_HOST_TO_DEVICE, stream->base);
  xpu_memcpy_async((void *)recvcountsDev, (void *)recvcounts,
                   nranks * sizeof(size_t), XPUMemcpyKind::XPU_HOST_TO_DEVICE,
                   stream->base);
  xpu_memcpy_async((void *)rdisplsDev, (void *)rdispls, nranks * sizeof(size_t),
                   XPUMemcpyKind::XPU_HOST_TO_DEVICE, stream->base);

  flagcxResult_t res = (flagcxResult_t)bkcl_all_to_all_v(
      comm->base, sendbuff, sendcountsDev, sdisplsDev,
      flagcxToXcclDataType(datatype), recvbuff, recvcountsDev, rdisplsDev,
      flagcxToXcclDataType(datatype), stream->base);
  cudaStreamSynchronize(stream->base);
  xpu_free(sendcountsDev);
  xpu_free(sdisplsDev);
  xpu_free(recvcountsDev);
  xpu_free(rdisplsDev);
  return res;
}

flagcxResult_t xcclAdaptorSend(const void *sendbuff, size_t count,
                               flagcxDataType_t datatype, int peer,
                               flagcxInnerComm_t comm, flagcxStream_t stream) {
  return (flagcxResult_t)bkcl_send(comm->base, sendbuff, count, peer,
                                   flagcxToXcclDataType(datatype),
                                   stream->base);
}

flagcxResult_t xcclAdaptorRecv(void *recvbuff, size_t count,
                               flagcxDataType_t datatype, int peer,
                               flagcxInnerComm_t comm, flagcxStream_t stream) {
  return (flagcxResult_t)bkcl_recv(comm->base, recvbuff, count, peer,
                                   flagcxToXcclDataType(datatype),
                                   stream->base);
}

flagcxResult_t xcclAdaptorGroupStart() {
  return (flagcxResult_t)bkcl_group_start();
}

flagcxResult_t xcclAdaptorGroupEnd() {
  return (flagcxResult_t)bkcl_group_end();
}

struct flagcxCCLAdaptor xcclAdaptor = {
    "XCCL",
    // Basic functions
    xcclAdaptorGetVersion, xcclAdaptorGetUniqueId, xcclAdaptorGetErrorString,
    xcclAdaptorGetLastError, xcclAdaptorGetStagedBuffer,
    // Communicator functions
    xcclAdaptorCommInitRank, xcclAdaptorCommFinalize, xcclAdaptorCommDestroy,
    xcclAdaptorCommAbort, xcclAdaptorCommResume, xcclAdaptorCommSuspend,
    xcclAdaptorCommCount, xcclAdaptorCommCuDevice, xcclAdaptorCommUserRank,
    xcclAdaptorCommGetAsyncError, xcclAdaptorMemAlloc, xcclAdaptorMemFree,
    xcclAdaptorCommRegister, xcclAdaptorCommDeregister,
    // Symmetric functions
    xcclAdaptorCommWindowRegister, xcclAdaptorCommWindowDeregister,
    // Communication functions
    xcclAdaptorReduce, xcclAdaptorGather, xcclAdaptorScatter,
    xcclAdaptorBroadcast, xcclAdaptorAllReduce, xcclAdaptorReduceScatter,
    xcclAdaptorAllGather, xcclAdaptorAlltoAll, xcclAdaptorAlltoAllv,
    xcclAdaptorSend, xcclAdaptorRecv,
    // Group semantics
    xcclAdaptorGroupStart, xcclAdaptorGroupEnd};

#endif // USE_KUNLUNXIN_ADAPTOR