/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_DEVICE_ADAPTOR_H_
#define FLAGCX_DEVICE_ADAPTOR_H_

#include "flagcx.h"
#include <string.h>

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

// CUDA-compatible APIs are implemented by several vendors, but NVIDIA
// compute-capability policy must only be applied to NVIDIA CUDA devices.
typedef enum {
  FLAGCX_GDR_DEVICE_UNKNOWN = 0,
  FLAGCX_GDR_DEVICE_CUDA = 1,
  FLAGCX_GDR_DEVICE_METAX = 2,
  FLAGCX_GDR_DEVICE_DU = 3,
  FLAGCX_GDR_DEVICE_PPU = 4,
} flagcxGdrDeviceFamily_t;

// C-compatible typedef matching the C++ using alias in dlsymbols.h.
typedef void (*flagcxLaunchFunc_t)(flagcxStream_t, void *);

// streamWaitValue64 always waits for *addr >= value. Callers add
// FLUSH_REMOTE_WRITES when the counter publishes completion of payload writes
// issued by another GPU, the network, or a host proxy. Adaptors must not
// silently ignore this flag: return flagcxNotSupported when the backend cannot
// provide the requested visibility guarantee.
typedef enum {
  FLAGCX_STREAM_WAIT_VALUE_DEFAULT = 0,
  FLAGCX_STREAM_WAIT_VALUE_FLUSH_REMOTE_WRITES = 1 << 0,
} flagcxStreamWaitValueFlags_t;

// Candidate routes for registering a VMM-backed allocation. The common layer
// intersects these device capabilities with the selected net adaptor and its
// runtime probes, preferring DMA-BUF over a provider-validated VA route.
typedef enum {
  FLAGCX_VMM_MR_CAP_NONE = 0,
  FLAGCX_VMM_MR_CAP_VA = 1 << 0,
  FLAGCX_VMM_MR_CAP_DMABUF = 1 << 1,
} flagcxVmmMrCaps_t;

typedef enum {
  FLAGCX_VMM_MR_ROUTE_NONE = 0,
  FLAGCX_VMM_MR_ROUTE_VA = 1,
  FLAGCX_VMM_MR_ROUTE_DMABUF = 2,
} flagcxVmmMrRoute_t;

// Internal metadata attached only to the latest in-process representation.
// These bits are not part of the frozen v1 plugin ABI.
typedef enum {
  FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE = 0,
  FLAGCX_DEVICE_ADAPTOR_INTERNAL_LEGACY_V1 = 1 << 0,
  // Transitional opt-in for built-in adaptors whose runtime pointer query has
  // not been wired yet. Remove this bit from each adaptor as it gains an
  // authoritative getPointerType implementation.
  FLAGCX_DEVICE_ADAPTOR_INTERNAL_IPC_POINTER_INFERENCE = 1 << 1,
} flagcxDeviceAdaptorInternalFlags_t;

// End-to-end GPUDirect visibility requirements advertised only by the latest
// in-process adaptor. READ covers an RDMA READ whose local destination is this
// device; WRITE covers an RDMA WRITE targeting this device. The bits describe
// when an acquire/flush is required, not which transport performs it, and are
// deliberately independent of ordinary, DMA-BUF, or VMM registration routes.
typedef enum {
  FLAGCX_GDR_FLUSH_NONE = 0,
  FLAGCX_GDR_READ_REQUIRES_FLUSH = 1 << 0,
  FLAGCX_GDR_WRITE_REQUIRES_FLUSH = 1 << 1,
} flagcxGdrFlushRequirements_t;

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

// Latest version — extends v1 with host registration, symmetric-memory,
// error-query, and pointer-introspection capabilities.
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
  // Release this process's multicast object reference after this process has
  // torn down its mapping. The provider object remains alive until every
  // imported/created reference has been released.
  flagcxResult_t (*symMulticastFree)(void *mcHandle);

  flagcxResult_t (*getLastError)();

  // Classify an address independently from IPC-export capability. Managed
  // and device allocations report FLAGCX_PTR_CUDA; ordinary/pinned host
  // allocations report FLAGCX_PTR_HOST. Unsupported backends return
  // flagcxNotSupported.
  flagcxResult_t (*getPointerType)(const void *ptr, int *ptrType);

  // Return the allocation containing ptr. IPC runtimes may export the whole
  // allocation and map its base even when ptr refers to an interior address.
  // This optional callback lets common IPC code preserve that user offset.
  flagcxResult_t (*getAddressRange)(const void *ptr, void **base, size_t *size);

  // Candidate registration routes for memory returned by gdrMemAlloc while
  // FLAGCX_VMM_ENABLE=1. Built-in adaptors must opt in explicitly.
  uint32_t vmmMrCaps;

  // Loader-owned metadata. A v1 plugin cannot advertise the latest allocation
  // capabilities, so common code uses this flag to preserve its historical
  // environment-driven MR routing without changing the v1 ABI.
  uint32_t internalFlags;

  // Added only to the latest in-process representation. Import and retain a
  // process-local reference to a multicast object. Each rank must keep this
  // reference until its own multicast mapping has been torn down; releasing
  // the final reference destroys the provider object without requiring a
  // communicator rendezvous during rank-local destruction.
  flagcxResult_t (*symMulticastImport)(int importFd, void **mcHandle);

  // Retry-safe VMM teardown primitives. These are latest-only: v1 stays
  // frozen. Common code records completion of each operation separately so a
  // VA-free failure never causes a successful unmap to be issued twice.
  flagcxResult_t (*symFlatMappingUnmap)(void *flatBase, size_t allocSize,
                                        int nPeers);
  flagcxResult_t (*symFlatVaFree)(void *flatBase, size_t allocSize, int nPeers);
  flagcxResult_t (*symMulticastMappingUnmap)(void *mcBase, size_t mcMapSize);
  flagcxResult_t (*symMulticastVaFree)(void *mcBase, size_t mcMapSize);

  // Default visibility requirements for GPUDirect operations targeting
  // allocations owned by this device adaptor. v1 plugins inherit their legacy
  // WRITE acquire contract; common code may apply explicit environment
  // overrides.
  uint32_t gdrFlushRequirements;

  // Device family used by the per-connection visibility resolver. Legacy v1
  // plugins remain UNKNOWN and therefore never receive CUDA-only exemptions.
  flagcxGdrDeviceFamily_t gdrDeviceFamily;

  // Latest-only architecture query. CUDA returns major * 10 + minor; other
  // and legacy adaptors leave this NULL so policy remains conservative. This
  // must not be added to flagcxDevProps, which is part of the frozen v1 ABI.
  flagcxResult_t (*getDeviceArchitecture)(int dev, int *architecture);
};

#define flagcxDeviceAdaptor flagcxDeviceAdaptor_latest

static inline bool flagcxDeviceAdaptorNativeAllocIsVmm(
    const struct flagcxDeviceAdaptor_latest *adaptor, bool vmmEnabled) {
  if (adaptor == NULL || !vmmEnabled)
    return false;
  if ((adaptor->internalFlags & FLAGCX_DEVICE_ADAPTOR_INTERNAL_LEGACY_V1) != 0)
    return true;
  return adaptor->vmmMrCaps != FLAGCX_VMM_MR_CAP_NONE;
}

static inline flagcxResult_t
flagcxDeviceAdaptorGetPointerTypeNotSupported(const void *ptr, int *ptrType) {
  (void)ptr;
  (void)ptrType;
  return flagcxNotSupported;
}

static inline flagcxResult_t
flagcxDeviceAdaptorGetAddressRangeNotSupported(const void *ptr, void **base,
                                               size_t *size) {
  (void)ptr;
  (void)base;
  (void)size;
  return flagcxNotSupported;
}

// Upgrade a v1 plugin struct to latest in-place into dst. New optional function
// pointers and VMM capabilities remain zero; visibility policy preserves the
// legacy signal-wait contract below.
static inline void
flagcxDeviceAdaptorUpgradeV1(const struct flagcxDeviceAdaptor_v1 *src,
                             struct flagcxDeviceAdaptor_latest *dst) {
  memset(dst, 0, sizeof(*dst));
  memcpy(dst, src, sizeof(struct flagcxDeviceAdaptor_v1));
  dst->internalFlags |= FLAGCX_DEVICE_ADAPTOR_INTERNAL_LEGACY_V1;
  // Preserve the v1 flagcxWaitSignal contract: before visibility requirements
  // became explicit, every signal wait requested a remote-write acquire. READ
  // remains unset because the v1 ABI never promised a post-GET flush.
  dst->gdrFlushRequirements = FLAGCX_GDR_WRITE_REQUIRES_FLUSH;
}

// Device adaptor plugin API version (independent of CCL/Net versions)
#define FLAGCX_DEVICE_ADAPTOR_PLUGIN_VERSION 1

// Versioned export symbol name
#define FLAGCX_DEVICE_ADAPTOR_PLUGIN_SYMBOL_V1 flagcxDeviceAdaptorPlugin_v1

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // FLAGCX_DEVICE_ADAPTOR_H_
