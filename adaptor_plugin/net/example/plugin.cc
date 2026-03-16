/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 *
 * Example net adaptor plugin for FlagCX.
 * This is a minimal skeleton: it reports 0 devices so the runtime
 * will fall back to a built-in adaptor (IBRC or Socket).
 ************************************************************************/

#include "flagcx/flagcx_net_adaptor.h"

static flagcxResult_t pluginInit() { return flagcxSuccess; }

static flagcxResult_t pluginDevices(int *ndev) {
  *ndev = 0;
  return flagcxSuccess;
}

static flagcxResult_t pluginGetProperties(int dev, void *props) {
  return flagcxInternalError;
}

static flagcxResult_t pluginReduceSupport(flagcxDataType_t dataType,
                                          flagcxRedOp_t redOp, int *supported) {
  return flagcxInternalError;
}

static flagcxResult_t pluginGetDeviceMr(void *comm, void *mhandle,
                                        void **dptr_mhandle) {
  return flagcxInternalError;
}

static flagcxResult_t pluginIrecvConsumed(void *recvComm, int n,
                                          void *request) {
  return flagcxInternalError;
}

static flagcxResult_t pluginListen(int dev, void *handle, void **listenComm) {
  return flagcxInternalError;
}

static flagcxResult_t pluginConnect(int dev, void *handle, void **sendComm) {
  return flagcxInternalError;
}

static flagcxResult_t pluginAccept(void *listenComm, void **recvComm) {
  return flagcxInternalError;
}

static flagcxResult_t pluginCloseSend(void *sendComm) {
  return flagcxInternalError;
}

static flagcxResult_t pluginCloseRecv(void *recvComm) {
  return flagcxInternalError;
}

static flagcxResult_t pluginCloseListen(void *listenComm) {
  return flagcxInternalError;
}

static flagcxResult_t pluginRegMr(void *comm, void *data, size_t size, int type,
                                  void **mhandle) {
  return flagcxInternalError;
}

static flagcxResult_t pluginRegMrDmaBuf(void *comm, void *data, size_t size,
                                        int type, uint64_t offset, int fd,
                                        void **mhandle) {
  return flagcxInternalError;
}

static flagcxResult_t pluginDeregMr(void *comm, void *mhandle) {
  return flagcxInternalError;
}

static flagcxResult_t pluginIsend(void *sendComm, void *data, size_t size,
                                  int tag, void *mhandle, void *phandle,
                                  void **request) {
  return flagcxInternalError;
}

static flagcxResult_t pluginIrecv(void *recvComm, int n, void **data,
                                  size_t *sizes, int *tags, void **mhandles,
                                  void **phandles, void **request) {
  return flagcxInternalError;
}

static flagcxResult_t pluginIflush(void *recvComm, int n, void **data,
                                   int *sizes, void **mhandles,
                                   void **request) {
  return flagcxInternalError;
}

static flagcxResult_t pluginTest(void *request, int *done, int *sizes) {
  return flagcxInternalError;
}

static flagcxResult_t pluginPut(void *sendComm, uint64_t srcOff,
                                uint64_t dstOff, size_t size, int srcRank,
                                int dstRank, void **gHandles, void **request) {
  return flagcxInternalError;
}

static flagcxResult_t pluginPutSignal(void *sendComm, uint64_t dstOff,
                                      int dstRank, void **gHandles,
                                      void **request) {
  return flagcxInternalError;
}

static flagcxResult_t pluginWaitValue(void **gHandles, int rank,
                                      uint64_t offset, uint64_t expected) {
  return flagcxInternalError;
}

static flagcxResult_t pluginGetDevFromName(char *name, int *dev) {
  return flagcxInternalError;
}

__attribute__((visibility(
    "default"))) struct flagcxNetAdaptor FLAGCX_NET_ADAPTOR_PLUGIN_SYMBOL = {
    "Example",           pluginInit,          pluginDevices,
    pluginGetProperties, pluginReduceSupport, pluginGetDeviceMr,
    pluginIrecvConsumed, pluginListen,        pluginConnect,
    pluginAccept,        pluginCloseSend,     pluginCloseRecv,
    pluginCloseListen,   pluginRegMr,         pluginRegMrDmaBuf,
    pluginDeregMr,       pluginIsend,         pluginIrecv,
    pluginIflush,        pluginTest,          pluginPut,
    pluginPutSignal,     pluginWaitValue,     pluginGetDevFromName,
};
