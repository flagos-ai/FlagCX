/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 *
 * Example CCL adaptor plugin for FlagCX.
 * This is a minimal skeleton: all operations return flagcxInternalError,
 * so this plugin is only useful for verifying that the loading mechanism
 * works. A real plugin would wrap a CCL library (e.g. NCCL, RCCL).
 ************************************************************************/

#include "flagcx/flagcx_ccl_adaptor.h"
#include "flagcx/nvidia_adaptor.h"

static flagcxResult_t pluginGetVersion(int *version) {
  return flagcxInternalError;
}

static flagcxResult_t pluginGetUniqueId(flagcxUniqueId_t *uniqueId) {
  return flagcxInternalError;
}

static const char *pluginGetErrorString(flagcxResult_t result) {
  return "Example CCL plugin: not implemented";
}

static const char *pluginGetLastError(flagcxInnerComm_t comm) {
  return "Example CCL plugin: not implemented";
}

static flagcxResult_t pluginGetStagedBuffer(const flagcxInnerComm_t comm,
                                            void **buff, size_t size,
                                            int isRecv) {
  return flagcxInternalError;
}

static flagcxResult_t pluginCommInitRank(flagcxInnerComm_t *comm, int nranks,
                                         flagcxUniqueId *commId, int rank,
                                         struct bootstrapState *bootstrap) {
  return flagcxInternalError;
}

static flagcxResult_t pluginCommFinalize(flagcxInnerComm_t comm) {
  return flagcxInternalError;
}

static flagcxResult_t pluginCommDestroy(flagcxInnerComm_t comm) {
  return flagcxInternalError;
}

static flagcxResult_t pluginCommAbort(flagcxInnerComm_t comm) {
  return flagcxInternalError;
}

static flagcxResult_t pluginCommResume(flagcxInnerComm_t comm) {
  return flagcxInternalError;
}

static flagcxResult_t pluginCommSuspend(flagcxInnerComm_t comm) {
  return flagcxInternalError;
}

static flagcxResult_t pluginCommCount(const flagcxInnerComm_t comm,
                                      int *count) {
  return flagcxInternalError;
}

static flagcxResult_t pluginCommGetDeviceNumber(const flagcxInnerComm_t comm,
                                                int *device) {
  return flagcxInternalError;
}

static flagcxResult_t pluginCommUserRank(const flagcxInnerComm_t comm,
                                         int *rank) {
  return flagcxInternalError;
}

static flagcxResult_t pluginCommGetAsyncError(flagcxInnerComm_t comm,
                                              flagcxResult_t *asyncError) {
  return flagcxInternalError;
}

static flagcxResult_t pluginMemAlloc(void **ptr, size_t size) {
  return flagcxInternalError;
}

static flagcxResult_t pluginMemFree(void *ptr) { return flagcxInternalError; }

static flagcxResult_t pluginCommRegister(const flagcxInnerComm_t comm,
                                         void *buff, size_t size,
                                         void **handle) {
  return flagcxInternalError;
}

static flagcxResult_t pluginCommDeregister(const flagcxInnerComm_t comm,
                                           void *handle) {
  return flagcxInternalError;
}

static flagcxResult_t pluginCommWindowRegister(flagcxInnerComm_t comm,
                                               void *buff, size_t size,
                                               flagcxWindow_t *win,
                                               int winFlags) {
  return flagcxInternalError;
}

static flagcxResult_t pluginCommWindowDeregister(flagcxInnerComm_t comm,
                                                 flagcxWindow_t win) {
  return flagcxInternalError;
}

static flagcxResult_t pluginReduce(const void *sendbuff, void *recvbuff,
                                   size_t count, flagcxDataType_t datatype,
                                   flagcxRedOp_t op, int root,
                                   flagcxInnerComm_t comm,
                                   flagcxStream_t stream) {
  return flagcxInternalError;
}

static flagcxResult_t pluginGather(const void *sendbuff, void *recvbuff,
                                   size_t count, flagcxDataType_t datatype,
                                   int root, flagcxInnerComm_t comm,
                                   flagcxStream_t stream) {
  return flagcxInternalError;
}

static flagcxResult_t pluginScatter(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    int root, flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return flagcxInternalError;
}

static flagcxResult_t pluginBroadcast(const void *sendbuff, void *recvbuff,
                                      size_t count, flagcxDataType_t datatype,
                                      int root, flagcxInnerComm_t comm,
                                      flagcxStream_t stream) {
  return flagcxInternalError;
}

static flagcxResult_t pluginAllReduce(const void *sendbuff, void *recvbuff,
                                      size_t count, flagcxDataType_t datatype,
                                      flagcxRedOp_t op, flagcxInnerComm_t comm,
                                      flagcxStream_t stream) {
  return flagcxInternalError;
}

static flagcxResult_t
pluginReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                    flagcxDataType_t datatype, flagcxRedOp_t op,
                    flagcxInnerComm_t comm, flagcxStream_t stream) {
  return flagcxInternalError;
}

static flagcxResult_t pluginAllGather(const void *sendbuff, void *recvbuff,
                                      size_t sendcount,
                                      flagcxDataType_t datatype,
                                      flagcxInnerComm_t comm,
                                      flagcxStream_t stream) {
  return flagcxInternalError;
}

static flagcxResult_t pluginAlltoAll(const void *sendbuff, void *recvbuff,
                                     size_t count, flagcxDataType_t datatype,
                                     flagcxInnerComm_t comm,
                                     flagcxStream_t stream) {
  return flagcxInternalError;
}

static flagcxResult_t pluginAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                      size_t *sdispls, void *recvbuff,
                                      size_t *recvcounts, size_t *rdispls,
                                      flagcxDataType_t datatype,
                                      flagcxInnerComm_t comm,
                                      flagcxStream_t stream) {
  return flagcxInternalError;
}

static flagcxResult_t pluginSend(const void *sendbuff, size_t count,
                                 flagcxDataType_t datatype, int peer,
                                 flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
  return flagcxInternalError;
}

static flagcxResult_t pluginRecv(void *recvbuff, size_t count,
                                 flagcxDataType_t datatype, int peer,
                                 flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
  return flagcxInternalError;
}

static flagcxResult_t pluginGroupStart() { return flagcxInternalError; }

static flagcxResult_t pluginGroupEnd() { return flagcxInternalError; }

__attribute__((visibility("default"))) struct flagcxCCLAdaptor_v1
    FLAGCX_CCL_ADAPTOR_PLUGIN_SYMBOL_V1 = {
        "Example",
        pluginGetVersion,
        pluginGetUniqueId,
        pluginGetErrorString,
        pluginGetLastError,
        pluginGetStagedBuffer,
        pluginCommInitRank,
        pluginCommFinalize,
        pluginCommDestroy,
        pluginCommAbort,
        pluginCommResume,
        pluginCommSuspend,
        pluginCommCount,
        pluginCommGetDeviceNumber,
        pluginCommUserRank,
        pluginCommGetAsyncError,
        pluginMemAlloc,
        pluginMemFree,
        pluginCommRegister,
        pluginCommDeregister,
        pluginCommWindowRegister,
        pluginCommWindowDeregister,
        pluginReduce,
        pluginGather,
        pluginScatter,
        pluginBroadcast,
        pluginAllReduce,
        pluginReduceScatter,
        pluginAllGather,
        pluginAlltoAll,
        pluginAlltoAllv,
        pluginSend,
        pluginRecv,
        pluginGroupStart,
        pluginGroupEnd,
};
