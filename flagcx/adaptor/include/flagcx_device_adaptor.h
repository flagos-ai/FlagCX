/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_DEVICE_ADAPTOR_H_
#define FLAGCX_DEVICE_ADAPTOR_H_

#include "flagcx.h"

#ifdef __cplusplus
extern "C" {
#endif

// Device properties — defined here so plugin authors have the full layout.
struct flagcxDevProps {
  char name[256];
  int pciBusId;
  int pciDeviceId;
  int pciDomainId;
};

// C-compatible typedef matching the C++ using alias in dlsymbols.h.
typedef void (*flagcxLaunchFunc_t)(flagcxStream_t, void *);

// Version history:
//   v1 — Initial version with basic device functions, GDR functions,
//         stream/event/IPC functions, kernel launch, device properties,
//         host func launch, DMA buffer, event elapsed time, and
//         stream memory operations.
struct flagcxDeviceAdaptor_v1 {
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
  flagcxResult_t (*streamWaitValue64)(flagcxStream_t stream, void *addr,
                                      uint64_t value, int flags);
  flagcxResult_t (*streamWriteValue64)(flagcxStream_t stream, void *addr,
                                       uint64_t value, int flags);

  // Event functions
  flagcxResult_t (*eventCreate)(flagcxEvent_t *event,
                                flagcxEventType_t eventType);
  flagcxResult_t (*eventDestroy)(flagcxEvent_t event);
  flagcxResult_t (*eventRecord)(flagcxEvent_t event, flagcxStream_t stream);
  flagcxResult_t (*eventSynchronize)(flagcxEvent_t event);
  flagcxResult_t (*eventQuery)(flagcxEvent_t event);
  flagcxResult_t (*eventElapsedTime)(float *ms, flagcxEvent_t start,
                                     flagcxEvent_t end);

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
};

// Latest version — extends v1 with host memory registration hooks used by
// the shm barrier path (FLAGCX_BARRIER_IPC_DISABLE=1).
struct flagcxDeviceAdaptor_latest {
  // All v1 fields (must stay layout-compatible with flagcxDeviceAdaptor_v1)
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
  flagcxResult_t (*streamWaitValue64)(flagcxStream_t stream, void *addr,
                                      uint64_t value, int flags);
  flagcxResult_t (*streamWriteValue64)(flagcxStream_t stream, void *addr,
                                       uint64_t value, int flags);
  // Event functions
  flagcxResult_t (*eventCreate)(flagcxEvent_t *event,
                                flagcxEventType_t eventType);
  flagcxResult_t (*eventDestroy)(flagcxEvent_t event);
  flagcxResult_t (*eventRecord)(flagcxEvent_t event, flagcxStream_t stream);
  flagcxResult_t (*eventSynchronize)(flagcxEvent_t event);
  flagcxResult_t (*eventQuery)(flagcxEvent_t event);
  flagcxResult_t (*eventElapsedTime)(float *ms, flagcxEvent_t start,
                                     flagcxEvent_t end);
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

  // Added beyond v1: host memory registration for the shm barrier path.
  // Registers/unregisters an mmap'd buffer as pinned memory so
  // hostGetDevicePointer returns a valid per-process GPU VA.
  flagcxResult_t (*hostRegister)(void *ptr, size_t size);
  flagcxResult_t (*hostUnregister)(void *ptr);

  // ---- Symmetric memory VMM functions (NULL if not supported) ----

  // Phase 1: Export existing VMM allocation as shareable handle.
  // ptr must have been allocated by gdrMemAlloc (VMM-backed).
  // physHandle: out — opaque handle for map/multicast/free
  // shareableHandle: out — buffer for IPC-exportable handle
  // handleSize: in/out — buffer size / actual size
  // allocSize: out — actual physical allocation size (granularity-aligned)
  flagcxResult_t (*symPhysAlloc)(void *ptr, size_t size, void **physHandle,
                                 void *shareableHandle, size_t *handleSize,
                                 size_t *allocSize);
  flagcxResult_t (*symPhysFree)(void *physHandle);

  // Phase 2: Import peer handles + reserve flat VA + map all peers.
  // peerHandles[]: shareable handles from all local peers
  // nPeers: number of local peers (including self)
  // selfIndex: this rank's index in peerHandles[]
  // selfPhysHandle: this rank's physical handle (avoids re-import)
  // allocSize: physical allocation size per peer (granularity-aligned)
  // flatBase: out — contiguous VA base (allocSize * nPeers)
  flagcxResult_t (*symFlatMap)(void *peerHandles[], int nPeers, int selfIndex,
                               void *selfPhysHandle, size_t allocSize,
                               void **flatBase);
  flagcxResult_t (*symFlatUnmap)(void *flatBase, size_t allocSize, int nPeers);

  // Phase 3: Multicast (NVLS). All function pointers are non-NULL.
  // Non-CUDA platforms use stubs that return flagcxNotSupported.
  flagcxResult_t (*symMulticastSupported)(int *supported);
  flagcxResult_t (*symMulticastCreate)(size_t allocSize, int nLocalDevices,
                                       const int *localDeviceOrdinals,
                                       void **mcHandle, int *shareableFd);
  flagcxResult_t (*symMulticastBind)(void *mcHandle, int importFd,
                                     void *physHandle, size_t allocSize,
                                     int localRank, int nLocalDevices,
                                     void **mcBase, size_t *mcMapSize);
  flagcxResult_t (*symMulticastTeardown)(void *mcBase, size_t mcMapSize);
  // Release the multicast object handle returned by symMulticastCreate.
  // Must be called after all ranks have torn down their mappings.
  flagcxResult_t (*symMulticastFree)(void *mcHandle);
};

#define flagcxDeviceAdaptor flagcxDeviceAdaptor_latest

// Upgrade a v1 plugin struct to latest in-place into dst.
// Fields added beyond v1 (hostRegister, hostUnregister) are zeroed (NULL).
static inline void
flagcxDeviceAdaptorUpgradeV1(const struct flagcxDeviceAdaptor_v1 *src,
                             struct flagcxDeviceAdaptor_latest *dst) {
  memset(dst, 0, sizeof(*dst));
  memcpy(dst, src, sizeof(struct flagcxDeviceAdaptor_v1));
}

// Device adaptor plugin API version (independent of CCL/Net versions)
#define FLAGCX_DEVICE_ADAPTOR_PLUGIN_VERSION 1

// Versioned export symbol name
#define FLAGCX_DEVICE_ADAPTOR_PLUGIN_SYMBOL_V1 flagcxDeviceAdaptorPlugin_v1

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // FLAGCX_DEVICE_ADAPTOR_H_
