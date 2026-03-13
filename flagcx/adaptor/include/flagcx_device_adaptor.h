/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_DEVICE_ADAPTOR_H_
#define FLAGCX_DEVICE_ADAPTOR_H_

#include "flagcx.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration for opaque types not in flagcx.h
// (flagcxStream_t, flagcxEvent_t, flagcxIpcMemHandle_t are already typedef'd in
// flagcx.h)
struct flagcxDevProps;

// C-compatible typedef matching the C++ using alias in dlsymbols.h.
typedef void (*flagcxLaunchFunc_t)(flagcxStream_t, void *);

struct flagcxDeviceAdaptor {
  char name[32];
  // Basic functions
  flagcxResult_t (*deviceSynchronize)();
  flagcxResult_t (*deviceMemcpy)(void *dst, void *src, size_t size,
                                 flagcxMemcpyType_t type, flagcxStream_t stream,
                                 void *args);
  flagcxResult_t (*deviceMemset)(void *ptr, int value, size_t size,
                                 flagcxMemType_t type, flagcxStream_t stream);
  flagcxResult_t (*deviceMalloc)(void **ptr, size_t size, flagcxMemType_t type,
                                 flagcxStream_t stream);
  flagcxResult_t (*deviceFree)(void *ptr, flagcxMemType_t type,
                               flagcxStream_t stream);
  flagcxResult_t (*setDevice)(int dev);
  flagcxResult_t (*getDevice)(int *dev);
  flagcxResult_t (*getDeviceCount)(int *count);
  flagcxResult_t (*getVendor)(char *vendor);
  flagcxResult_t (*hostGetDevicePointer)(void **pDevice, void *pHost);

  // GDR functions
  flagcxResult_t (*memHandleInit)(int dev_id, void **memHandle);
  flagcxResult_t (*memHandleDestroy)(int dev, void *memHandle);
  flagcxResult_t (*gdrMemAlloc)(void **ptr, size_t size, void *memHandle);
  flagcxResult_t (*gdrMemFree)(void *ptr, void *memHandle);
  flagcxResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void *memHandle);
  flagcxResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
  flagcxResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t sz);
  flagcxResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);

  // Stream functions
  flagcxResult_t (*streamCreate)(flagcxStream_t *stream);
  flagcxResult_t (*streamDestroy)(flagcxStream_t stream);
  flagcxResult_t (*streamCopy)(flagcxStream_t *newStream, void *oldStream);
  flagcxResult_t (*streamFree)(flagcxStream_t stream);
  flagcxResult_t (*streamSynchronize)(flagcxStream_t stream);
  flagcxResult_t (*streamQuery)(flagcxStream_t stream);
  flagcxResult_t (*streamWaitEvent)(flagcxStream_t stream, flagcxEvent_t event);

  // Event functions
  flagcxResult_t (*eventCreate)(flagcxEvent_t *event,
                                flagcxEventType_t eventType);
  flagcxResult_t (*eventDestroy)(flagcxEvent_t event);
  flagcxResult_t (*eventRecord)(flagcxEvent_t event, flagcxStream_t stream);
  flagcxResult_t (*eventSynchronize)(flagcxEvent_t event);
  flagcxResult_t (*eventQuery)(flagcxEvent_t event);

  // IpcMemHandle functions
  flagcxResult_t (*ipcMemHandleCreate)(flagcxIpcMemHandle_t *handle,
                                       size_t *size);
  flagcxResult_t (*ipcMemHandleGet)(flagcxIpcMemHandle_t handle, void *devPtr);
  flagcxResult_t (*ipcMemHandleOpen)(flagcxIpcMemHandle_t handle,
                                     void **devPtr);
  flagcxResult_t (*ipcMemHandleClose)(void *devPtr);
  flagcxResult_t (*ipcMemHandleFree)(flagcxIpcMemHandle_t handle);

  // Kernel launch
  flagcxResult_t (*launchKernel)(void *func, unsigned int block_x,
                                 unsigned int block_y, unsigned int block_z,
                                 unsigned int grid_x, unsigned int grid_y,
                                 unsigned int grid_z, void **args,
                                 size_t share_mem, void *stream,
                                 void *memHandle);
  flagcxResult_t (*copyArgsInit)(void **args);
  flagcxResult_t (*copyArgsFree)(void *args);
  flagcxResult_t (*launchDeviceFunc)(flagcxStream_t stream,
                                     flagcxLaunchFunc_t fn, void *args);

  // Others
  flagcxResult_t (*getDeviceProperties)(struct flagcxDevProps *props, int dev);
  flagcxResult_t (*getDevicePciBusId)(char *pciBusId, int len, int dev);
  flagcxResult_t (*getDeviceByPciBusId)(int *dev, const char *pciBusId);

  // HostFunc launch
  flagcxResult_t (*launchHostFunc)(flagcxStream_t stream, void (*fn)(void *),
                                   void *args);
  // DMA buffer
  flagcxResult_t (*dmaSupport)(bool *dmaBufferSupport);
  flagcxResult_t (*getHandleForAddressRange)(void *handleOut, void *buffer,
                                             size_t size,
                                             unsigned long long flags);
  // Event elapsed time
  flagcxResult_t (*eventElapsedTime)(float *ms, flagcxEvent_t start,
                                     flagcxEvent_t end);
};

// Device adaptor plugin API version (independent of CCL/Net versions)
#define FLAGCX_DEVICE_ADAPTOR_PLUGIN_VERSION 1

// Versioned export symbol name
#define FLAGCX_DEVICE_ADAPTOR_PLUGIN_SYMBOL flagcxDeviceAdaptorPlugin_v1

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // FLAGCX_DEVICE_ADAPTOR_H_
