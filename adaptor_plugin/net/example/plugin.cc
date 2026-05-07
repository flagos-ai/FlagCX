/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 *
 * Example net adaptor plugin for FlagCX.
 * This is a minimal skeleton: it reports 0 devices so the runtime
 * will fall back to a built-in adaptor (IBRC or Socket).
 ************************************************************************/

#include "flagcx/flagcx_net.h"
#include "flagcx/flagcx_net_adaptor.h"

#include <string.h>

static flagcxResult_t pluginInit() { return flagcxSuccess; }

static flagcxResult_t pluginDevices(int *ndev) {
  *ndev = 0;
  return flagcxSuccess;
}

// Reference implementation showing the structure of flagcxNetProperties_t
// and the usage of different memory pointer types.
static flagcxResult_t pluginGetProperties(int dev, void *props) {
  flagcxNetProperties_t *p = (flagcxNetProperties_t *)props;
  memset(p, 0, sizeof(*p));

  // Human-readable device name (used in log messages)
  p->name = (char *)"ExampleNet0";

  // sysfs PCI path, e.g. "/sys/devices/pci0000:00/0000:00:01.0"
  // Set to NULL when no PCI device backs this network interface.
  p->pciPath = NULL;

  // Globally unique identifier for the NIC chip.
  // Important for multi-function cards (physical or SR-IOV virtual).
  p->guid = 0;

  // Bitmask of memory pointer types this device can send/recv from directly:
  //   FLAGCX_PTR_HOST   - host (CPU) pinned memory
  //   FLAGCX_PTR_CUDA   - device (GPU) memory (requires GPUDirect / peer
  //   access) FLAGCX_PTR_DMABUF - DMA-BUF imported memory (requires kernel
  //   DMA-BUF support)
  // A pure CPU-based transport typically supports only FLAGCX_PTR_HOST.
  // An RDMA transport with GPUDirect would set FLAGCX_PTR_HOST |
  // FLAGCX_PTR_CUDA.
  p->ptrSupport = FLAGCX_PTR_HOST;

  // Set to 1 if regMr registrations are global (not tied to a particular comm).
  // When true, a single registration can be reused across multiple connections.
  p->regIsGlobal = 0;

  // Link speed in Mbps (e.g. 100000 for 100 Gbps)
  p->speed = 0;

  // Physical port number (0-based)
  p->port = 0;

  // One-way network latency in microseconds
  p->latency = 0;

  // Maximum number of concurrent comm objects (send + recv) this device
  // supports. Use 1 if the device has no such limit.
  p->maxComms = 1;

  // Maximum number of grouped receive operations in a single irecv call.
  // Use 1 if grouped receives are not supported.
  p->maxRecvs = 1;

  // Network device offload type:
  //   FLAGCX_NET_DEVICE_HOST   - all processing on host CPU (default)
  //   FLAGCX_NET_DEVICE_UNPACK - device-side unpack offload supported
  p->netDeviceType = FLAGCX_NET_DEVICE_HOST;

  // Version number for the device offload protocol.
  // Set to FLAGCX_NET_DEVICE_INVALID_VERSION when netDeviceType is HOST.
  p->netDeviceVersion = FLAGCX_NET_DEVICE_INVALID_VERSION;

  return flagcxSuccess;
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
                                  int mrFlags, void **mhandle) {
  return flagcxInternalError;
}

static flagcxResult_t pluginRegMrDmaBuf(void *comm, void *data, size_t size,
                                        int type, uint64_t offset, int fd,
                                        int mrFlags, void **mhandle) {
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

static flagcxResult_t pluginIput(void *sendComm, uint64_t srcOff,
                                 uint64_t dstOff, size_t size, int srcRank,
                                 int dstRank, void **srcHandles,
                                 void **dstHandles, void **request) {
  return flagcxInternalError;
}

static flagcxResult_t pluginIget(void *sendComm, uint64_t srcOff,
                                 uint64_t dstOff, size_t size, int srcRank,
                                 int dstRank, void **srcHandles,
                                 void **dstHandles, void **request) {
  return flagcxInternalError;
}

static flagcxResult_t pluginIputSignal(void *sendComm, uint64_t srcOff,
                                       uint64_t dstOff, size_t size,
                                       int srcRank, int dstRank,
                                       void **srcHandles, void **dstHandles,
                                       uint64_t signalOff, void **signalHandles,
                                       uint64_t signalValue, void **request) {
  return flagcxInternalError;
}

static flagcxResult_t pluginGetDevFromName(char *name, int *dev) {
  return flagcxInternalError;
}

__attribute__((visibility("default"))) struct flagcxNetAdaptor_v1
    FLAGCX_NET_ADAPTOR_PLUGIN_SYMBOL_V1 = {
        "Example",           pluginInit,       pluginDevices,
        pluginGetProperties, pluginListen,     pluginConnect,
        pluginAccept,        pluginCloseSend,  pluginCloseRecv,
        pluginCloseListen,   pluginRegMr,      pluginRegMrDmaBuf,
        pluginDeregMr,       pluginIsend,      pluginIrecv,
        pluginIflush,        pluginTest,       pluginIput,
        pluginIget,          pluginIputSignal, pluginGetDevFromName,
};
