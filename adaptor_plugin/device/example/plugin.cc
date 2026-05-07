/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 *
 * Example device adaptor plugin for FlagCX.
 * This is a minimal skeleton: all operations return flagcxInternalError,
 * so this plugin is only useful for verifying that the loading mechanism
 * works. A real plugin would wrap a device runtime (e.g. CUDA, CANN).
 ************************************************************************/

#include "flagcx/flagcx_device_adaptor.h"
#include "flagcx/nvidia_adaptor.h"

static flagcxResult_t pluginDeviceSynchronize() { return flagcxInternalError; }

static flagcxResult_t pluginDeviceMemcpy(void *dst, void *src, size_t size,
                                         flagcxMemcpyType_t type,
                                         flagcxStream_t stream, void *args) {
  return flagcxInternalError;
}

static flagcxResult_t pluginDeviceMemset(void *ptr, int value, size_t size,
                                         flagcxMemType_t type,
                                         flagcxStream_t stream) {
  return flagcxInternalError;
}

static flagcxResult_t pluginDeviceMalloc(void **ptr, size_t size,
                                         flagcxMemType_t type,
                                         flagcxStream_t stream) {
  return flagcxInternalError;
}

static flagcxResult_t pluginDeviceFree(void *ptr, flagcxMemType_t type,
                                       flagcxStream_t stream) {
  return flagcxInternalError;
}

static flagcxResult_t pluginSetDevice(int dev) { return flagcxInternalError; }

static flagcxResult_t pluginGetDevice(int *dev) { return flagcxInternalError; }

static flagcxResult_t pluginGetDeviceCount(int *count) {
  return flagcxInternalError;
}

static flagcxResult_t pluginGetVendor(char *vendor) {
  return flagcxInternalError;
}

static flagcxResult_t pluginHostGetDevicePointer(void **pDevice, void *pHost) {
  return flagcxInternalError;
}

// GDR functions
static flagcxResult_t pluginMemHandleInit(int dev_id, void **memHandle) {
  return flagcxInternalError;
}

static flagcxResult_t pluginMemHandleDestroy(int dev, void *memHandle) {
  return flagcxInternalError;
}

static flagcxResult_t pluginGdrMemAlloc(void **ptr, size_t size,
                                        void *memHandle) {
  return flagcxInternalError;
}

static flagcxResult_t pluginGdrMemFree(void *ptr, void *memHandle) {
  return flagcxInternalError;
}

static flagcxResult_t pluginHostShareMemAlloc(void **ptr, size_t size,
                                              void *memHandle) {
  return flagcxInternalError;
}

static flagcxResult_t pluginHostShareMemFree(void *ptr, void *memHandle) {
  return flagcxInternalError;
}

static flagcxResult_t pluginGdrPtrMmap(void **pcpuptr, void *devptr,
                                       size_t sz) {
  return flagcxInternalError;
}

static flagcxResult_t pluginGdrPtrMunmap(void *cpuptr, size_t sz) {
  return flagcxInternalError;
}

// Stream functions
static flagcxResult_t pluginStreamCreate(flagcxStream_t *stream) {
  return flagcxInternalError;
}

static flagcxResult_t pluginStreamDestroy(flagcxStream_t stream) {
  return flagcxInternalError;
}

static flagcxResult_t pluginStreamCopy(flagcxStream_t *newStream,
                                       void *oldStream) {
  return flagcxInternalError;
}

static flagcxResult_t pluginStreamFree(flagcxStream_t stream) {
  return flagcxInternalError;
}

static flagcxResult_t pluginStreamSynchronize(flagcxStream_t stream) {
  return flagcxInternalError;
}

static flagcxResult_t pluginStreamQuery(flagcxStream_t stream) {
  return flagcxInternalError;
}

static flagcxResult_t pluginStreamWaitEvent(flagcxStream_t stream,
                                            flagcxEvent_t event) {
  return flagcxInternalError;
}

static flagcxResult_t pluginStreamWaitValue64(flagcxStream_t stream, void *addr,
                                              uint64_t value, int flags) {
  return flagcxInternalError;
}

static flagcxResult_t pluginStreamWriteValue64(flagcxStream_t stream,
                                               void *addr, uint64_t value,
                                               int flags) {
  return flagcxInternalError;
}

// Event functions
static flagcxResult_t pluginEventCreate(flagcxEvent_t *event,
                                        flagcxEventType_t eventType) {
  return flagcxInternalError;
}

static flagcxResult_t pluginEventDestroy(flagcxEvent_t event) {
  return flagcxInternalError;
}

static flagcxResult_t pluginEventRecord(flagcxEvent_t event,
                                        flagcxStream_t stream) {
  return flagcxInternalError;
}

static flagcxResult_t pluginEventSynchronize(flagcxEvent_t event) {
  return flagcxInternalError;
}

static flagcxResult_t pluginEventQuery(flagcxEvent_t event) {
  return flagcxInternalError;
}

static flagcxResult_t pluginEventElapsedTime(float *ms, flagcxEvent_t start,
                                             flagcxEvent_t end) {
  return flagcxInternalError;
}

// IpcMemHandle functions
static flagcxResult_t pluginIpcMemHandleCreate(flagcxIpcMemHandle_t *handle,
                                               size_t *size) {
  return flagcxInternalError;
}

static flagcxResult_t pluginIpcMemHandleGet(flagcxIpcMemHandle_t handle,
                                            void *devPtr) {
  return flagcxInternalError;
}

static flagcxResult_t pluginIpcMemHandleOpen(flagcxIpcMemHandle_t handle,
                                             void **devPtr) {
  return flagcxInternalError;
}

static flagcxResult_t pluginIpcMemHandleClose(void *devPtr) {
  return flagcxInternalError;
}

static flagcxResult_t pluginIpcMemHandleFree(flagcxIpcMemHandle_t handle) {
  return flagcxInternalError;
}

// Kernel launch
static flagcxResult_t
pluginLaunchKernel(void *func, unsigned int block_x, unsigned int block_y,
                   unsigned int block_z, unsigned int grid_x,
                   unsigned int grid_y, unsigned int grid_z, void **args,
                   size_t share_mem, void *stream, void *memHandle) {
  return flagcxInternalError;
}

static flagcxResult_t pluginCopyArgsInit(void **args) {
  return flagcxInternalError;
}

static flagcxResult_t pluginCopyArgsFree(void *args) {
  return flagcxInternalError;
}

static flagcxResult_t pluginLaunchDeviceFunc(flagcxStream_t stream,
                                             flagcxLaunchFunc_t fn,
                                             void *args) {
  return flagcxInternalError;
}

// Others
static flagcxResult_t pluginGetDeviceProperties(struct flagcxDevProps *props,
                                                int dev) {
  return flagcxInternalError;
}

static flagcxResult_t pluginGetDevicePciBusId(char *pciBusId, int len,
                                              int dev) {
  return flagcxInternalError;
}

static flagcxResult_t pluginGetDeviceByPciBusId(int *dev,
                                                const char *pciBusId) {
  return flagcxInternalError;
}

// HostFunc launch
static flagcxResult_t pluginLaunchHostFunc(flagcxStream_t stream,
                                           void (*fn)(void *), void *args) {
  return flagcxInternalError;
}

// DMA buffer
static flagcxResult_t pluginDmaSupport(bool *dmaBufferSupport) {
  return flagcxInternalError;
}

static flagcxResult_t pluginGetHandleForAddressRange(void *handleOut,
                                                     void *buffer, size_t size,
                                                     unsigned long long flags) {
  return flagcxInternalError;
}

__attribute__((visibility("default"))) struct flagcxDeviceAdaptor
    FLAGCX_DEVICE_ADAPTOR_PLUGIN_SYMBOL_V1 = {
        "Example",
        // Basic functions
        pluginDeviceSynchronize,
        pluginDeviceMemcpy,
        pluginDeviceMemset,
        pluginDeviceMalloc,
        pluginDeviceFree,
        pluginSetDevice,
        pluginGetDevice,
        pluginGetDeviceCount,
        pluginGetVendor,
        pluginHostGetDevicePointer,
        // GDR functions
        pluginMemHandleInit,
        pluginMemHandleDestroy,
        pluginGdrMemAlloc,
        pluginGdrMemFree,
        pluginHostShareMemAlloc,
        pluginHostShareMemFree,
        pluginGdrPtrMmap,
        pluginGdrPtrMunmap,
        // Stream functions
        pluginStreamCreate,
        pluginStreamDestroy,
        pluginStreamCopy,
        pluginStreamFree,
        pluginStreamSynchronize,
        pluginStreamQuery,
        pluginStreamWaitEvent,
        pluginStreamWaitValue64,
        pluginStreamWriteValue64,
        // Event functions
        pluginEventCreate,
        pluginEventDestroy,
        pluginEventRecord,
        pluginEventSynchronize,
        pluginEventQuery,
        pluginEventElapsedTime,
        // IpcMemHandle functions
        pluginIpcMemHandleCreate,
        pluginIpcMemHandleGet,
        pluginIpcMemHandleOpen,
        pluginIpcMemHandleClose,
        pluginIpcMemHandleFree,
        // Kernel launch
        pluginLaunchKernel,
        pluginCopyArgsInit,
        pluginCopyArgsFree,
        pluginLaunchDeviceFunc,
        // Others
        pluginGetDeviceProperties,
        pluginGetDevicePciBusId,
        pluginGetDeviceByPciBusId,
        // HostFunc launch
        pluginLaunchHostFunc,
        // DMA buffer
        pluginDmaSupport,
        pluginGetHandleForAddressRange,
};
