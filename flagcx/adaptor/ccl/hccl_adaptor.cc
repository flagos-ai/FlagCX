#include "ascend_adaptor.h"

#ifdef USE_ASCEND_ADAPTOR
#include <map>
#include <vector>
std::map<flagcxDataType_t, HcclDataType> f2h_datatype_map = {
    {flagcxInt8, HCCL_DATA_TYPE_INT8},
    {flagcxUint8, HCCL_DATA_TYPE_UINT8},
    {flagcxInt, HCCL_DATA_TYPE_INT32},
    {flagcxInt32, HCCL_DATA_TYPE_INT32},
    {flagcxInt64, HCCL_DATA_TYPE_INT64},
    {flagcxHalf, HCCL_DATA_TYPE_FP16},
    {flagcxFloat16, HCCL_DATA_TYPE_FP16},
    {flagcxBfloat16, HCCL_DATA_TYPE_BFP16},
    {flagcxFloat32, HCCL_DATA_TYPE_FP32},
    {flagcxFloat, HCCL_DATA_TYPE_FP32},
    {flagcxDouble, HCCL_DATA_TYPE_FP64},
};

std::map<flagcxRedOp_t, HcclReduceOp> f2h_reduceop_map = {
    {flagcxSum, HCCL_REDUCE_SUM},
    {flagcxProd, HCCL_REDUCE_PROD},
    {flagcxMax, HCCL_REDUCE_MAX},
    {flagcxMin, HCCL_REDUCE_MIN}};

// TODO: not match fully
std::map<HcclResult, flagcxResult_t> h2f_ret_map = {
    {HCCL_SUCCESS, flagcxSuccess},
    {HCCL_E_PARA, flagcxInvalidArgument},
    {HCCL_E_PTR, flagcxUnhandledDeviceError},
    {HCCL_E_MEMORY, flagcxUnhandledDeviceError},
    {HCCL_E_INTERNAL, flagcxInternalError},
    {HCCL_E_NOT_SUPPORT, flagcxNotSupported},
    {HCCL_E_NOT_FOUND, flagcxUnhandledDeviceError},
    {HCCL_E_UNAVAIL, flagcxUnhandledDeviceError},
    {HCCL_E_SYSCALL, flagcxUnhandledDeviceError},
    {HCCL_E_TIMEOUT, flagcxUnhandledDeviceError},
    {HCCL_E_OPEN_FILE_FAILURE, flagcxUnhandledDeviceError},
    {HCCL_E_TCP_CONNECT, flagcxUnhandledDeviceError},
    {HCCL_E_ROCE_CONNECT, flagcxUnhandledDeviceError},
    {HCCL_E_TCP_TRANSFER, flagcxUnhandledDeviceError},
    {HCCL_E_ROCE_TRANSFER, flagcxUnhandledDeviceError},
    {HCCL_E_RUNTIME, flagcxUnhandledDeviceError},
    {HCCL_E_DRV, flagcxUnhandledDeviceError},
    {HCCL_E_PROFILING, flagcxUnhandledDeviceError},
    {HCCL_E_CCE, flagcxUnhandledDeviceError},
    {HCCL_E_NETWORK, flagcxUnhandledDeviceError},
    {HCCL_E_AGAIN, flagcxUnhandledDeviceError},
    {HCCL_E_REMOTE, flagcxRemoteError},
    {HCCL_E_SUSPENDING, flagcxUnhandledDeviceError},
    {HCCL_E_RESERVED, flagcxUnhandledDeviceError}};

std::map<flagcxResult_t, HcclResult> f2h_ret_map = {
    {flagcxSuccess, HCCL_SUCCESS},
    {flagcxInternalError, HCCL_E_INTERNAL},
    {flagcxNotSupported, HCCL_E_NOT_SUPPORT},
    {flagcxInvalidArgument, HCCL_E_PARA},
    {flagcxRemoteError, HCCL_E_REMOTE},
    {flagcxUnhandledDeviceError, HCCL_E_RESERVED}};

struct HcclSendRecvItemEx {
  flagcxInnerComm_t comm;
  flagcxStream_t stream;
};
HcclSendRecvItemEx item;
std::vector<HcclSendRecvItem> sendRecvInfo;

// TODO: unsupported
flagcxResult_t hcclAdaptorGetVersion(int *version) {
  return flagcxNotSupported;
}

flagcxResult_t hcclAdaptorGetUniqueId(flagcxUniqueId_t *uniqueId) {
  return (
      flagcxResult_t)h2f_ret_map[HcclGetRootInfo((HcclRootInfo *)(*uniqueId))];
}

flagcxResult_t hcclAdaptorGetStagedBuffer(const flagcxInnerComm_t comm,
                                          void **buff, size_t size,
                                          int isRecv) {
  return flagcxNotSupported;
}

const char *hcclAdaptorGetErrorString(flagcxResult_t result) {
  return HcclGetErrorString((HcclResult)f2h_ret_map[result]);
}

// TODO: unsupported
const char *hcclAdaptorGetLastError(flagcxInnerComm_t comm) {
  return "Not Implemented";
}

flagcxResult_t hcclAdaptorCommInitRank(flagcxInnerComm_t *comm, int nranks,
                                       flagcxUniqueId_t commId, int rank,
                                       bootstrapState * /*bootstrap*/) {
  if (*comm == NULL) {
    flagcxCalloc(comm, 1);
  }
  return (flagcxResult_t)h2f_ret_map[HcclCommInitRootInfo(
      nranks, (HcclRootInfo *)commId, rank, &(*comm)->base)];
}

// TODO: unsupported
flagcxResult_t hcclAdaptorCommFinalize(flagcxInnerComm_t comm) {
  return flagcxUnhandledDeviceError;
}

flagcxResult_t hcclAdaptorCommDestroy(flagcxInnerComm_t comm) {
  return (flagcxResult_t)h2f_ret_map[HcclCommDestroy(comm->base)];
}

// TODO: unsupported
flagcxResult_t hcclAdaptorCommAbort(flagcxInnerComm_t comm) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t hcclAdaptorCommResume(flagcxInnerComm_t comm) {
  return flagcxUnhandledDeviceError;
}

// TODO: unsupported
flagcxResult_t hcclAdaptorCommSuspend(flagcxInnerComm_t comm) {
  return flagcxUnhandledDeviceError;
}

flagcxResult_t hcclAdaptorCommCount(const flagcxInnerComm_t comm, int *count) {
  return (flagcxResult_t)
      h2f_ret_map[HcclGetRankSize(comm->base, (uint32_t *)count)];
}

// TODO: unsupported
flagcxResult_t hcclAdaptorCommCuDevice(const flagcxInnerComm_t comm,
                                       int *device) {
  return flagcxUnhandledDeviceError;
}

flagcxResult_t hcclAdaptorCommUserRank(const flagcxInnerComm_t comm,
                                       int *rank) {
  return (
      flagcxResult_t)h2f_ret_map[HcclGetRankId(comm->base, (uint32_t *)rank)];
}

// TODO: unsupported
flagcxResult_t hcclAdaptorCommGetAsyncError(flagcxInnerComm_t comm,
                                            flagcxResult_t *asyncError) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t hcclAdaptorMemAlloc(void **ptr, size_t size) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t hcclAdaptorMemFree(void *ptr) { return flagcxNotSupported; }

// TODO: unsupported
flagcxResult_t hcclAdaptorCommRegister(flagcxInnerComm_t comm, void *buff,
                                       size_t size, void **handle) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t hcclAdaptorCommDeregister(flagcxInnerComm_t comm, void *handle) {
  return flagcxNotSupported;
}

flagcxResult_t hcclAdaptorCommWindowRegister(flagcxInnerComm_t comm, void *buff,
                                             size_t size, flagcxWindow_t *win,
                                             int winFlags) {
  return flagcxNotSupported;
}

flagcxResult_t hcclAdaptorCommWindowDeregister(flagcxInnerComm_t comm,
                                               flagcxWindow_t win) {
  return flagcxNotSupported;
}

flagcxResult_t hcclAdaptorReduce(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 flagcxRedOp_t op, int root,
                                 flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
  void *sendbuffptr = (void *)sendbuff;
  return (flagcxResult_t)h2f_ret_map[HcclReduce(
      sendbuffptr, recvbuff, count, (HcclDataType)f2h_datatype_map[datatype],
      (HcclReduceOp)f2h_reduceop_map[op], root, comm->base, stream->base)];
}

// TODO: unsupported
flagcxResult_t hcclAdaptorGather(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 int root, flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
  uint32_t rank, nranks;
  HcclResult res = HCCL_SUCCESS;
  res = HcclGetRankSize(comm->base, &nranks);
  res = HcclGetRankId(comm->base, &rank);
  size_t size = count * getFlagcxDataTypeSize(datatype);
  char *buffer = static_cast<char *>(recvbuff);
  std::vector<HcclSendRecvItem> sendRecvInfo;
  if (rank == root) {
    for (uint32_t r = 0; r < nranks; r++) {
      sendRecvInfo.emplace_back(HcclSendRecvItem{
          HcclSendRecvType::HCCL_RECV, static_cast<void *>(buffer + r * size),
          size, HcclDataType::HCCL_DATA_TYPE_INT8, r});
    }
  }
  void *sendbuffptr = (void *)sendbuff;
  sendRecvInfo.emplace_back(
      HcclSendRecvItem{HcclSendRecvType::HCCL_SEND, sendbuffptr, size,
                       HcclDataType::HCCL_DATA_TYPE_INT8, (uint32_t)root});
  uint32_t itemNum = sendRecvInfo.size();
  HcclBatchSendRecv(sendRecvInfo.data(), itemNum, comm->base, stream->base);

  return (flagcxResult_t)h2f_ret_map[res];
}

flagcxResult_t hcclAdaptorScatter(const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  int root, flagcxInnerComm_t comm,
                                  flagcxStream_t stream) {
  void *sendbuffptr = (void *)sendbuff;
  return (flagcxResult_t)h2f_ret_map[HcclScatter(
      sendbuffptr, recvbuff, count, (HcclDataType)f2h_datatype_map[datatype],
      root, comm->base, stream->base)];
}

flagcxResult_t hcclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    int root, flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {

  uint32_t rank;
  HcclGetRankId(comm->base, &rank);
  if (rank == root) {
    aclrtMemcpy(recvbuff, count, sendbuff, count, ACL_MEMCPY_DEVICE_TO_DEVICE);
  }
  void *buffer = (rank == root) ? const_cast<void *>(sendbuff) : recvbuff;
  return (flagcxResult_t)h2f_ret_map[HcclBroadcast(
      buffer, count, (HcclDataType)f2h_datatype_map[datatype], root, comm->base,
      stream->base)];
}

flagcxResult_t hcclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    flagcxRedOp_t op, flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  void *sendbuffptr = (void *)sendbuff;
  return (flagcxResult_t)h2f_ret_map[HcclAllReduce(
      sendbuffptr, recvbuff, count, (HcclDataType)f2h_datatype_map[datatype],
      (HcclReduceOp)f2h_reduceop_map[op], comm->base, stream->base)];
}

flagcxResult_t
hcclAdaptorReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                         flagcxDataType_t datatype, flagcxRedOp_t op,
                         flagcxInnerComm_t comm, flagcxStream_t stream) {
  void *sendbuffptr = (void *)sendbuff;
  return (flagcxResult_t)h2f_ret_map[HcclReduceScatter(
      sendbuffptr, recvbuff, recvcount,
      (HcclDataType)f2h_datatype_map[datatype],
      (HcclReduceOp)f2h_reduceop_map[op], comm->base, stream->base)];
}

flagcxResult_t hcclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                    size_t sendcount, flagcxDataType_t datatype,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  void *sendbuffptr = (void *)sendbuff;
  return (flagcxResult_t)h2f_ret_map[HcclAllGather(
      sendbuffptr, recvbuff, sendcount,
      (HcclDataType)f2h_datatype_map[datatype], comm->base, stream->base)];
}

flagcxResult_t hcclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                   size_t count, flagcxDataType_t datatype,
                                   flagcxInnerComm_t comm,
                                   flagcxStream_t stream) {
  void *sendbuffptr = (void *)sendbuff;
  return (flagcxResult_t)h2f_ret_map[HcclAlltoAll(
      sendbuffptr, count, (HcclDataType)f2h_datatype_map[datatype], recvbuff,
      count, (HcclDataType)f2h_datatype_map[datatype], comm->base,
      stream->base)];
}

flagcxResult_t hcclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                    size_t *sdispls, void *recvbuff,
                                    size_t *recvcounts, size_t *rdispls,
                                    flagcxDataType_t datatype,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  void *sendbuffptr = (void *)sendbuff;
  return (flagcxResult_t)h2f_ret_map[HcclAlltoAllV(
      sendbuffptr, sendcounts, sdispls,
      (HcclDataType)f2h_datatype_map[datatype], recvbuff, recvcounts, rdispls,
      (HcclDataType)f2h_datatype_map[datatype], comm->base, stream->base)];
}

flagcxResult_t hcclAdaptorSend(const void *sendbuff, size_t count,
                               flagcxDataType_t datatype, int peer,
                               flagcxInnerComm_t comm, flagcxStream_t stream) {
  void *sendbuffptr = (void *)sendbuff;
  item.comm = comm;
  item.stream = stream;
  sendRecvInfo.emplace_back(HcclSendRecvItem{
      HcclSendRecvType::HCCL_SEND, sendbuffptr, count,
      (HcclDataType)f2h_datatype_map[datatype], (uint32_t)peer});
  return flagcxSuccess;
}

flagcxResult_t hcclAdaptorRecv(void *recvbuff, size_t count,
                               flagcxDataType_t datatype, int peer,
                               flagcxInnerComm_t comm, flagcxStream_t stream) {
  sendRecvInfo.emplace_back(HcclSendRecvItem{
      HcclSendRecvType::HCCL_RECV, recvbuff, count,
      (HcclDataType)f2h_datatype_map[datatype], (uint32_t)peer});
  return flagcxSuccess;
}

flagcxResult_t hcclAdaptorGroupStart() {
  sendRecvInfo.clear();
  return flagcxSuccess;
}

flagcxResult_t hcclAdaptorGroupEnd() {
  uint32_t itemNum = 0;
  itemNum = sendRecvInfo.size();
  if (itemNum > 0) {
    return (flagcxResult_t)h2f_ret_map[HcclBatchSendRecv(
        sendRecvInfo.data(), itemNum, item.comm->base, item.stream->base)];
  }
  return flagcxSuccess;
}

struct flagcxCCLAdaptor hcclAdaptor = {
    "HCCL",
    // Basic functions
    hcclAdaptorGetVersion, hcclAdaptorGetUniqueId, hcclAdaptorGetErrorString,
    hcclAdaptorGetLastError, hcclAdaptorGetStagedBuffer,
    // Communicator functions
    hcclAdaptorCommInitRank, hcclAdaptorCommFinalize, hcclAdaptorCommDestroy,
    hcclAdaptorCommAbort, hcclAdaptorCommResume, hcclAdaptorCommSuspend,
    hcclAdaptorCommCount, hcclAdaptorCommCuDevice, hcclAdaptorCommUserRank,
    hcclAdaptorCommGetAsyncError, hcclAdaptorMemAlloc, hcclAdaptorMemFree,
    hcclAdaptorCommRegister, hcclAdaptorCommDeregister,
    // Symmetric functions
    hcclAdaptorCommWindowRegister, hcclAdaptorCommWindowDeregister,
    // Communication functions
    hcclAdaptorReduce, hcclAdaptorGather, hcclAdaptorScatter,
    hcclAdaptorBroadcast, hcclAdaptorAllReduce, hcclAdaptorReduceScatter,
    hcclAdaptorAllGather, hcclAdaptorAlltoAll, hcclAdaptorAlltoAllv,
    hcclAdaptorSend, hcclAdaptorRecv,
    // Group semantics
    hcclAdaptorGroupStart, hcclAdaptorGroupEnd};

#endif // USE_ASCEND_ADAPTOR
