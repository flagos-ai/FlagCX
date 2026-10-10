#include "du_adaptor.h"

#ifdef USE_DU_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "param.h"
#include <limits>
#include <mutex>
#include <new>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

struct DucudaVmmAllocation {
  CUmemGenericAllocationHandle handle;
  size_t size;
  bool mappingOwned;
  bool vaOwned;
  bool handleOwned;
};
static std::mutex gDucudaVmmAllocationMtx;
static std::unordered_map<void *, DucudaVmmAllocation> gDucudaVmmAllocations;

struct DucudaSymPhysHandle {
  CUmemGenericAllocationHandle handle;
  CUdeviceptr allocationBase;
  size_t allocationSize;
  bool releaseOwned;
};

struct DucudaFlatMapping {
  size_t allocSize;
  int nPeers;
  std::vector<unsigned char> mappedSlots;
  std::vector<CUmemGenericAllocationHandle> importedHandles;
  std::vector<unsigned char> importedHandleOwned;
  bool vaOwned;
};
static std::mutex gDucudaFlatMappingMtx;
static std::unordered_map<void *, DucudaFlatMapping> gDucudaFlatMappings;

static bool ducudaFlatMappingMatches(const DucudaFlatMapping &mapping,
                                     size_t allocSize, int nPeers) {
  return mapping.allocSize == allocSize && mapping.nPeers == nPeers;
}

std::map<flagcxMemcpyType_t, cudaMemcpyKind> memcpy_type_map = {
    {flagcxMemcpyHostToDevice, cudaMemcpyHostToDevice},
    {flagcxMemcpyDeviceToHost, cudaMemcpyDeviceToHost},
    {flagcxMemcpyDeviceToDevice, cudaMemcpyDeviceToDevice},
};

flagcxResult_t ducudaAdaptorDeviceSynchronize() {
  DEVCHECK(cudaDeviceSynchronize());
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
                                         flagcxMemcpyType_t type,
                                         flagcxStream_t stream, void *args) {
  if (stream == NULL) {
    DEVCHECK(cudaMemcpy(dst, src, size, memcpy_type_map[type]));
  } else {
    DEVCHECK(
        cudaMemcpyAsync(dst, src, size, memcpy_type_map[type], stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorDeviceMemset(void *ptr, int value, size_t size,
                                         flagcxMemType_t type,
                                         flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    memset(ptr, value, size);
  } else {
    if (stream == NULL) {
      DEVCHECK(cudaMemset(ptr, value, size));
    } else {
      DEVCHECK(cudaMemsetAsync(ptr, value, size, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorDeviceMalloc(void **ptr, size_t size,
                                         flagcxMemType_t type,
                                         flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    DEVCHECK(cudaHostAlloc(ptr, size, cudaHostAllocMapped));
  } else if (type == flagcxMemManaged) {
    DEVCHECK(cudaMallocManaged(ptr, size, cudaMemAttachGlobal));
  } else {
    if (stream == NULL) {
      DEVCHECK(cudaMalloc(ptr, size));
    } else {
      DEVCHECK(cudaMallocAsync(ptr, size, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorDeviceFree(void *ptr, flagcxMemType_t type,
                                       flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    DEVCHECK(cudaFreeHost(ptr));
  } else if (type == flagcxMemManaged) {
    DEVCHECK(cudaFree(ptr));
  } else {
    if (stream == NULL) {
      DEVCHECK(cudaFree(ptr));
    } else {
      DEVCHECK(cudaFreeAsync(ptr, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorSetDevice(int dev) {
  DEVCHECK(cudaSetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorGetDevice(int *dev) {
  DEVCHECK(cudaGetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorGetDeviceCount(int *count) {
  DEVCHECK(cudaGetDeviceCount(count));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "DU");
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorCanAccessPeer(int srcDev, int dstDev,
                                          int *canAccess) {
  if (canAccess == NULL)
    return flagcxInvalidArgument;
  DEVCHECK(cudaDeviceCanAccessPeer(canAccess, srcDev, dstDev));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorGetPointerType(const void *ptr, int *ptrType) {
  if (ptr == NULL || ptrType == NULL)
    return flagcxInvalidArgument;

  cudaPointerAttributes attrs = {};
  cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
  if (err == cudaErrorInvalidValue) {
    // Ordinary host allocations are not tracked by the DU runtime. Clear the
    // probe error so it cannot affect a later runtime call.
    cudaGetLastError();
    *ptrType = FLAGCX_PTR_HOST;
    return flagcxSuccess;
  }
  if (err != cudaSuccess) {
    // Runtime/device errors must not silently turn a GPU pointer into host
    // memory, which would register it through the wrong transport path.
    cudaGetLastError();
    return flagcxUnhandledDeviceError;
  }
#if CUDART_VERSION >= 10000
  *ptrType = (attrs.type == cudaMemoryTypeDevice ||
              attrs.type == cudaMemoryTypeManaged)
                 ? FLAGCX_PTR_CUDA
                 : FLAGCX_PTR_HOST;
#else
  *ptrType = (attrs.memoryType == cudaMemoryTypeDevice || attrs.isManaged)
                 ? FLAGCX_PTR_CUDA
                 : FLAGCX_PTR_HOST;
#endif
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorHostGetDevicePointer(void **pDevice, void *pHost) {
  if (pDevice == NULL || pHost == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaHostGetDevicePointer(pDevice, pHost, 0));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorGdrMemAlloc(void **ptr, size_t size,
                                        void *memHandle) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
  if (!flagcxParamVmmEnable()) {
    DEVCHECK(cudaMalloc(ptr, size));
    cudaPointerAttributes attrs;
    DEVCHECK(cudaPointerGetAttributes(&attrs, *ptr));
    unsigned flags = 1;
    DEVCHECK(cuPointerSetAttribute(&flags, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                   (CUdeviceptr)attrs.devicePointer));
    return flagcxSuccess;
  }

  int device = 0;
  CUdevice cuDevice;
  DEVCHECK(cudaGetDevice(&device));
  DEVCHECK(cuDeviceGet(&cuDevice, device));

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = cuDevice;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  // The DCU CUDA compatibility layer aborts on NVIDIA-only capability enum
  // 110 instead of returning CUDA_ERROR_NOT_SUPPORTED.  Do not query it here:
  // DMA-BUF export and provider VA registration are the authoritative probes
  // for this allocation and are exercised independently by the strict CI
  // modes.

  size_t granularity = 0;
  DEVCHECK(cuMemGetAllocationGranularity(&granularity, &prop,
                                         CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  size_t allocSize = size;
  ALIGN_SIZE(allocSize, granularity);

  CUmemGenericAllocationHandle handle;
  DEVCHECK(cuMemCreate(&handle, allocSize, &prop, 0));
  CUdeviceptr address = 0;
  CUresult result = cuMemAddressReserve(&address, allocSize, granularity, 0, 0);
  if (result != CUDA_SUCCESS) {
    cuMemRelease(handle);
    return flagcxUnhandledDeviceError;
  }
  result = cuMemMap(address, allocSize, 0, handle, 0);
  if (result != CUDA_SUCCESS) {
    cuMemAddressFree(address, allocSize);
    cuMemRelease(handle);
    return flagcxUnhandledDeviceError;
  }
  CUmemAccessDesc access = {};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = cuDevice;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  result = cuMemSetAccess(address, allocSize, &access, 1);
  if (result != CUDA_SUCCESS) {
    cuMemUnmap(address, allocSize);
    cuMemAddressFree(address, allocSize);
    cuMemRelease(handle);
    return flagcxUnhandledDeviceError;
  }
  *ptr = (void *)address;
  bool tracked = false;
  try {
    std::lock_guard<std::mutex> lock(gDucudaVmmAllocationMtx);
    tracked = gDucudaVmmAllocations
                  .emplace(*ptr, DucudaVmmAllocation{handle, allocSize, true,
                                                     true, true})
                  .second;
  } catch (const std::bad_alloc &) {
  }
  if (!tracked) {
    cuMemUnmap(address, allocSize);
    cuMemAddressFree(address, allocSize);
    cuMemRelease(handle);
    *ptr = NULL;
    return flagcxSystemError;
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return flagcxSuccess;
  }
  std::lock_guard<std::mutex> lock(gDucudaVmmAllocationMtx);
  auto it = gDucudaVmmAllocations.find(ptr);
  if (it == gDucudaVmmAllocations.end()) {
    DEVCHECK(cudaFree(ptr));
    return flagcxSuccess;
  }
  DucudaVmmAllocation &allocation = it->second;
  if (allocation.mappingOwned) {
    if (cuMemUnmap((CUdeviceptr)ptr, allocation.size) != CUDA_SUCCESS)
      return flagcxUnhandledDeviceError;
    allocation.mappingOwned = false;
  }
  if (allocation.vaOwned) {
    if (cuMemAddressFree((CUdeviceptr)ptr, allocation.size) != CUDA_SUCCESS)
      return flagcxUnhandledDeviceError;
    allocation.vaOwned = false;
  }
  // Keep the handle returned by cuMemCreate alive for the whole allocation
  // lifetime.  The DCU compatibility layer does not reliably preserve the
  // backing allocation across repeated retain/release cycles when only VA
  // mappings hold references. Window registrations borrow this root handle;
  // only VMM allocations created outside this adaptor retain a private handle.
  if (allocation.handleOwned) {
    if (cuMemRelease(allocation.handle) != CUDA_SUCCESS)
      return flagcxUnhandledDeviceError;
    allocation.handleOwned = false;
  }
  gDucudaVmmAllocations.erase(it);
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorStreamCreate(flagcxStream_t *stream) {
  (*stream) = NULL;
  flagcxCalloc(stream, 1);
  DEVCHECK(cudaStreamCreateWithFlags((cudaStream_t *)(*stream),
                                     cudaStreamNonBlocking));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorStreamDestroy(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(cudaStreamDestroy(stream->base));
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorStreamCopy(flagcxStream_t *newStream,
                                       void *oldStream) {
  (*newStream) = NULL;
  flagcxCalloc(newStream, 1);
  (*newStream)->base = (cudaStream_t)oldStream;
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorStreamFree(flagcxStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorStreamSynchronize(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(cudaStreamSynchronize(stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorStreamQuery(flagcxStream_t stream) {
  flagcxResult_t res = flagcxSuccess;
  if (stream != NULL) {
    cudaError error = cudaStreamQuery(stream->base);
    if (error == cudaSuccess) {
      res = flagcxSuccess;
    } else if (error == cudaErrorNotReady) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

flagcxResult_t ducudaAdaptorStreamWaitEvent(flagcxStream_t stream,
                                            flagcxEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(
        cudaStreamWaitEvent(stream->base, event->base, cudaEventWaitDefault));
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorEventCreate(flagcxEvent_t *event,
                                        flagcxEventType_t eventType) {
  (*event) = NULL;
  flagcxCalloc(event, 1);
  const unsigned int flags = (eventType == flagcxEventDefault)
                                 ? cudaEventDefault
                                 : cudaEventDisableTiming;
  DEVCHECK(cudaEventCreateWithFlags(&((*event)->base), flags));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorEventDestroy(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(cudaEventDestroy(event->base));
    free(event);
    event = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorEventRecord(flagcxEvent_t event,
                                        flagcxStream_t stream) {
  if (event != NULL) {
    if (stream != NULL) {
      DEVCHECK(cudaEventRecordWithFlags(event->base, stream->base,
                                        cudaEventRecordDefault));
    } else {
      DEVCHECK(cudaEventRecordWithFlags(event->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorEventSynchronize(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(cudaEventSynchronize(event->base));
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorEventQuery(flagcxEvent_t event) {
  flagcxResult_t res = flagcxSuccess;
  if (event != NULL) {
    cudaError error = cudaEventQuery(event->base);
    if (error == cudaSuccess) {
      res = flagcxSuccess;
    } else if (error == cudaErrorNotReady) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

flagcxResult_t ducudaAdaptorIpcMemHandleCreate(flagcxIpcMemHandle_t *handle,
                                               size_t *size) {
  flagcxCalloc(handle, 1);
  if (size != NULL) {
    *size = sizeof(cudaIpcMemHandle_t);
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorIpcMemHandleGet(flagcxIpcMemHandle_t handle,
                                            void *devPtr) {
  if (handle == NULL || devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaIpcGetMemHandle(&handle->base, devPtr));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorIpcMemHandleOpen(flagcxIpcMemHandle_t handle,
                                             void **devPtr) {
  if (handle == NULL || devPtr == NULL || *devPtr != NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaIpcOpenMemHandle(devPtr, handle->base,
                                cudaIpcMemLazyEnablePeerAccess));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorIpcMemHandleClose(void *devPtr) {
  if (devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaIpcCloseMemHandle(devPtr));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorIpcMemHandleFree(flagcxIpcMemHandle_t handle) {
  if (handle != NULL) {
    free(handle);
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorLaunchHostFunc(flagcxStream_t stream,
                                           void (*fn)(void *), void *args) {
  if (stream != NULL) {
    DEVCHECK(cudaLaunchHostFunc(stream->base, fn, args));
  }
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorDmaSupport(bool *dmaBufferSupport) {
  if (dmaBufferSupport == NULL)
    return flagcxInvalidArgument;

  // Enum 124 is also unsupported by the DCU compatibility layer and aborts
  // the process.  Advertise the implemented export path and let
  // cuMemGetHandleForAddressRange provide the per-allocation result.  Auto
  // mode falls back to VA only on an explicit flagcxNotSupported result.
  *dmaBufferSupport = true;
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorMemGetHandleForAddressRange(
    void *handleOut, void *buffer, size_t size, unsigned long long flags) {
  if (handleOut == NULL || buffer == NULL || size == 0)
    return flagcxInvalidArgument;
  CUresult result =
      cuMemGetHandleForAddressRange(handleOut, (CUdeviceptr)buffer, size,
                                    CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, flags);
  if (result == CUDA_ERROR_NOT_SUPPORTED)
    return flagcxNotSupported;
  return result == CUDA_SUCCESS ? flagcxSuccess : flagcxUnhandledDeviceError;
}

flagcxResult_t ducudaAdaptorGetDeviceProperties(struct flagcxDevProps *props,
                                                int dev) {
  if (props == NULL) {
    return flagcxInvalidArgument;
  }

  cudaDeviceProp devProp;
  DEVCHECK(cudaGetDeviceProperties(&devProp, dev));
  strncpy(props->name, devProp.name, sizeof(props->name) - 1);
  props->name[sizeof(props->name) - 1] = '\0';
  props->pciBusId = devProp.pciBusID;
  props->pciDeviceId = devProp.pciDeviceID;
  props->pciDomainId = devProp.pciDomainID;
  // TODO: see if there's another way to get this info. In some cuda versions,
  // cudaDeviceProp does not have `gpuDirectRDMASupported` field
  // props->gdrSupported = devProp.gpuDirectRDMASupported;

  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorGetDevicePciBusId(char *pciBusId, int len,
                                              int dev) {
  if (pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaDeviceGetPCIBusId(pciBusId, len, dev));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorGetDeviceByPciBusId(int *dev,
                                                const char *pciBusId) {
  if (dev == NULL || pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaDeviceGetByPCIBusId(dev, pciBusId));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorStreamWaitValue64(flagcxStream_t stream, void *addr,
                                              uint64_t value, int flags) {
  if (stream == NULL || addr == NULL)
    return flagcxInvalidArgument;
  if (flags & ~FLAGCX_STREAM_WAIT_VALUE_FLUSH_REMOTE_WRITES)
    return flagcxInvalidArgument;

  // The current DU driver supports ordinary stream memory waits but not the
  // acquire/flush guarantee required after a NIC or peer device publishes the
  // waited-on value. Report the missing capability without invoking a driver
  // flag that can surface as a generic error on this CUDA-compatible runtime.
  if (flags & FLAGCX_STREAM_WAIT_VALUE_FLUSH_REMOTE_WRITES)
    return flagcxNotSupported;

  unsigned int waitFlags = CU_STREAM_WAIT_VALUE_GEQ;

  CUstream cuStream = (CUstream)(stream->base);
  CUresult err =
      cuStreamWaitValue64(cuStream, (CUdeviceptr)addr, value, waitFlags);
  if (err == CUDA_SUCCESS)
    return flagcxSuccess;
  if (err == CUDA_ERROR_NOT_SUPPORTED)
    return flagcxNotSupported;
  return flagcxUnhandledDeviceError;
}
flagcxResult_t ducudaAdaptorStreamWriteValue64(flagcxStream_t stream,
                                               void *addr, uint64_t value,
                                               int flags) {
  (void)flags;
  if (stream == NULL || addr == NULL)
    return flagcxInvalidArgument;
  CUstream cuStream = (CUstream)(stream->base);
  CUresult err = cuStreamWriteValue64(cuStream, (CUdeviceptr)addr, value,
                                      CU_STREAM_WRITE_VALUE_DEFAULT);
  return (err == CUDA_SUCCESS) ? flagcxSuccess : flagcxUnhandledDeviceError;
}
flagcxResult_t ducudaAdaptorEventElapsedTime(float *ms, flagcxEvent_t start,
                                             flagcxEvent_t end) {
  if (ms == NULL || start == NULL || end == NULL) {
    return flagcxInvalidArgument;
  }
  cudaError_t error = cudaEventElapsedTime(ms, start->base, end->base);
  if (error == cudaSuccess) {
    return flagcxSuccess;
  } else if (error == cudaErrorNotReady) {
    return flagcxInProgress;
  } else {
    return flagcxUnhandledDeviceError;
  }
}

flagcxResult_t ducudaAdaptorHostRegister(void *ptr, size_t size) {
  if (ptr == NULL || size == 0) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaHostRegister(ptr, size, cudaHostRegisterMapped));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorHostUnregister(void *ptr) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaHostUnregister(ptr));
  return flagcxSuccess;
}

// Symmetric memory VMM handle exchange and flat mapping.
static flagcxResult_t
ducudaSymPhysHandleDestroy(DucudaSymPhysHandle *physHandle) {
  if (physHandle == NULL)
    return flagcxSuccess;
  if (physHandle->releaseOwned) {
    CUresult result = cuMemRelease(physHandle->handle);
    if (result != CUDA_SUCCESS) {
      WARN("DU VMM physical handle release failed: result=%d "
           "allocationBase=%p size=%zu",
           (int)result, (void *)physHandle->allocationBase,
           physHandle->allocationSize);
      return flagcxUnhandledDeviceError;
    }
    physHandle->releaseOwned = false;
  }
  free(physHandle);
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorSymPhysAlloc(void *ptr, size_t size,
                                         void **physHandle,
                                         void *shareableHandle,
                                         size_t *handleSize,
                                         size_t *allocSize) {
  if (ptr == NULL || physHandle == NULL || shareableHandle == NULL ||
      handleSize == NULL || allocSize == NULL)
    return flagcxInvalidArgument;
  *physHandle = NULL;
  *allocSize = 0;
  if (*handleSize < sizeof(int))
    return flagcxInvalidArgument;

  // Discover actual physical allocation size (already granularity-aligned)
  CUdeviceptr allocationBase = 0;
  size_t actualAllocSize = 0;
  CUresult result =
      cuMemGetAddressRange(&allocationBase, &actualAllocSize, (CUdeviceptr)ptr);
  if (result != CUDA_SUCCESS)
    return flagcxUnhandledDeviceError;
  const CUdeviceptr address = (CUdeviceptr)ptr;
  if (allocationBase == 0 || actualAllocSize == 0 || address < allocationBase ||
      address - allocationBase > actualAllocSize ||
      size > actualAllocSize - (address - allocationBase))
    return flagcxInvalidUsage;

  DucudaSymPhysHandle *handle =
      (DucudaSymPhysHandle *)malloc(sizeof(DucudaSymPhysHandle));
  if (handle == NULL)
    return flagcxSystemError;
  handle->allocationBase = allocationBase;
  handle->allocationSize = actualAllocSize;
  handle->releaseOwned = false;

  bool rootAllocationFound = false;
  {
    std::lock_guard<std::mutex> lock(gDucudaVmmAllocationMtx);
    auto it = gDucudaVmmAllocations.find((void *)allocationBase);
    if (it != gDucudaVmmAllocations.end()) {
      const DucudaVmmAllocation &allocation = it->second;
      if (allocation.size != actualAllocSize || !allocation.mappingOwned ||
          !allocation.vaOwned || !allocation.handleOwned) {
        free(handle);
        return flagcxInvalidUsage;
      }
      // The allocation registry owns the handle returned by cuMemCreate for
      // the whole allocation lifetime.  Borrow it for window registration;
      // repeatedly retaining and releasing this root handle invalidates the
      // backing allocation on GalaxyHIP after several windows.
      handle->handle = allocation.handle;
      rootAllocationFound = true;
    }
  }
  if (!rootAllocationFound) {
    // VMM allocations not created by ducudaAdaptorGdrMemAlloc still require a
    // private retained reference, released when this window is destroyed.
    result = cuMemRetainAllocationHandle(&handle->handle, ptr);
    if (result != CUDA_SUCCESS) {
      free(handle);
      if (result == CUDA_ERROR_INVALID_VALUE ||
          result == CUDA_ERROR_NOT_SUPPORTED)
        return flagcxNotSupported;
      return flagcxUnhandledDeviceError;
    }
    handle->releaseOwned = true;
  }

  // Export as POSIX fd for IPC sharing.
  result =
      cuMemExportToShareableHandle(shareableHandle, handle->handle,
                                   CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
  if (result != CUDA_SUCCESS) {
    flagcxResult_t cleanupResult = ducudaSymPhysHandleDestroy(handle);
    if (cleanupResult != flagcxSuccess) {
      // Preserve the retained provider resource for the common rollback path.
      *physHandle = handle;
      return cleanupResult;
    }
    return flagcxUnhandledDeviceError;
  }
  *allocSize = actualAllocSize;
  *handleSize = sizeof(int); // POSIX fd is an int
  *physHandle = handle;
  return flagcxSuccess;
}
flagcxResult_t ducudaAdaptorSymPhysFree(void *physHandle) {
  return ducudaSymPhysHandleDestroy((DucudaSymPhysHandle *)physHandle);
}
flagcxResult_t ducudaAdaptorSymFlatMappingUnmap(void *flatBase,
                                                size_t allocSize, int nPeers);
flagcxResult_t ducudaAdaptorSymFlatVaFree(void *flatBase, size_t allocSize,
                                          int nPeers);
flagcxResult_t ducudaAdaptorSymFlatMap(void *peerHandles[], int nPeers,
                                       int selfIndex, void *selfPhysHandle,
                                       size_t allocSize, void **flatBase) {
  if (peerHandles == NULL || selfPhysHandle == NULL || flatBase == NULL ||
      nPeers <= 0 || selfIndex < 0 || selfIndex >= nPeers || allocSize == 0 ||
      allocSize > std::numeric_limits<size_t>::max() / (size_t)nPeers)
    return flagcxInvalidArgument;
  *flatBase = NULL;

  CUmemGenericAllocationHandle selfHandle =
      ((DucudaSymPhysHandle *)selfPhysHandle)->handle;

  // allocSize is already granularity-aligned (from cuMemGetAddressRange)
  size_t totalSize = allocSize * nPeers;

  DucudaFlatMapping initialMapping;
  try {
    initialMapping.allocSize = allocSize;
    initialMapping.nPeers = nPeers;
    initialMapping.mappedSlots.assign(nPeers, 0);
    initialMapping.importedHandles.resize(nPeers);
    initialMapping.importedHandleOwned.assign(nPeers, 0);
    initialMapping.vaOwned = true;
  } catch (const std::bad_alloc &) {
    return flagcxSystemError;
  }

  // Reserve the full VA range
  CUdeviceptr base = 0;
  CUresult result = cuMemAddressReserve(&base, totalSize, 0, 0, 0);
  if (result != CUDA_SUCCESS) {
    WARN("DU VMM flat map failed: stage=address-reserve result=%d "
         "allocSize=%zu nPeers=%d",
         (int)result, allocSize, nPeers);
    return flagcxUnhandledDeviceError;
  }

  // Publish adaptor-private ownership before the first fallible map step.  On
  // failure the common symmetric-window state machine receives flatBase and
  // calls the split mapping/VA cleanup callbacks; it does not need to know
  // which DU slots were mapped successfully.
  std::unique_lock<std::mutex> lock(gDucudaFlatMappingMtx);
  bool tracked = false;
  try {
    tracked =
        gDucudaFlatMappings.emplace((void *)base, std::move(initialMapping))
            .second;
  } catch (const std::bad_alloc &) {
  }
  if (!tracked) {
    lock.unlock();
    result = cuMemAddressFree(base, totalSize);
    if (result != CUDA_SUCCESS)
      WARN("DU VMM flat map cleanup failed: stage=address-free result=%d "
           "base=%p size=%zu",
           (int)result, (void *)base, totalSize);
    return flagcxSystemError;
  }
  DucudaFlatMapping &mapping = gDucudaFlatMappings.at((void *)base);
  *flatBase = (void *)base;

  // Import and map each peer's physical memory
  int cudaDev;
  cudaError_t cudaResult = cudaGetDevice(&cudaDev);
  if (cudaResult != cudaSuccess) {
    WARN("DU VMM flat map failed: stage=get-device result=%d base=%p",
         (int)cudaResult, (void *)base);
    return flagcxUnhandledDeviceError;
  }
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  for (int i = 0; i < nPeers; i++) {
    CUmemGenericAllocationHandle peerHandle;
    if (i == selfIndex) {
      peerHandle = selfHandle;
    } else {
      int fd = *(int *)peerHandles[i];
      result = cuMemImportFromShareableHandle(
          &peerHandle, (void *)(uintptr_t)fd,
          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
      if (result != CUDA_SUCCESS) {
        WARN("DU VMM flat map failed: stage=import result=%d base=%p "
             "slot=%d fd=%d",
             (int)result, (void *)base, i, fd);
        return flagcxUnhandledDeviceError;
      }
      mapping.importedHandles[i] = peerHandle;
      mapping.importedHandleOwned[i] = 1;
    }
    CUdeviceptr slot = base + (CUdeviceptr)i * allocSize;
    result = cuMemMap(slot, allocSize, 0, peerHandle, 0);
    if (result != CUDA_SUCCESS) {
      WARN("DU VMM flat map failed: stage=map result=%d base=%p slot=%d "
           "slotAddress=%p size=%zu",
           (int)result, (void *)base, i, (void *)slot, allocSize);
      return flagcxUnhandledDeviceError;
    }
    mapping.mappedSlots[i] = 1;
    result = cuMemSetAccess(slot, allocSize, &accessDesc, 1);
    if (result != CUDA_SUCCESS) {
      WARN("DU VMM flat map failed: stage=set-access result=%d base=%p "
           "slot=%d slotAddress=%p size=%zu",
           (int)result, (void *)base, i, (void *)slot, allocSize);
      return flagcxUnhandledDeviceError;
    }
  }

  return flagcxSuccess;
}
flagcxResult_t ducudaAdaptorSymFlatUnmap(void *flatBase, size_t allocSize,
                                         int nPeers) {
  flagcxResult_t result =
      ducudaAdaptorSymFlatMappingUnmap(flatBase, allocSize, nPeers);
  return result == flagcxSuccess
             ? ducudaAdaptorSymFlatVaFree(flatBase, allocSize, nPeers)
             : result;
}
flagcxResult_t ducudaAdaptorSymFlatMappingUnmap(void *flatBase,
                                                size_t allocSize, int nPeers) {
  if (flatBase == NULL)
    return flagcxSuccess;

  std::lock_guard<std::mutex> lock(gDucudaFlatMappingMtx);
  auto it = gDucudaFlatMappings.find(flatBase);
  if (it == gDucudaFlatMappings.end()) {
    WARN("DU VMM flat unmap failed: untracked base=%p", flatBase);
    return flagcxInvalidUsage;
  }
  DucudaFlatMapping &mapping = it->second;
  if (!ducudaFlatMappingMatches(mapping, allocSize, nPeers)) {
    WARN("DU VMM flat unmap failed: geometry mismatch base=%p "
         "expectedSize=%zu actualSize=%zu expectedPeers=%d actualPeers=%d",
         flatBase, mapping.allocSize, allocSize, mapping.nPeers, nPeers);
    return flagcxInvalidArgument;
  }

  CUdeviceptr base = (CUdeviceptr)flatBase;
  for (int i = 0; i < nPeers; i++) {
    if (!mapping.mappedSlots[i])
      continue;
    CUdeviceptr slot = base + (CUdeviceptr)i * allocSize;
    CUresult result = cuMemUnmap(slot, allocSize);
    if (result != CUDA_SUCCESS) {
      WARN("DU VMM flat unmap failed: stage=slot-unmap result=%d base=%p "
           "slot=%d slotAddress=%p size=%zu",
           (int)result, flatBase, i, (void *)slot, allocSize);
      return flagcxUnhandledDeviceError;
    }
    mapping.mappedSlots[i] = 0;
  }

  // GalaxyHIP does not reliably keep an imported allocation alive after its
  // handle is released while an alias mapping still exists.  Keep every
  // imported handle until all slots have been unmapped.  Returning above on an
  // unmap failure deliberately preserves every handle for a later retry.
  for (int i = 0; i < nPeers; i++) {
    if (!mapping.importedHandleOwned[i])
      continue;
    CUresult result = cuMemRelease(mapping.importedHandles[i]);
    if (result != CUDA_SUCCESS) {
      WARN("DU VMM flat unmap failed: stage=release-import result=%d "
           "base=%p slot=%d",
           (int)result, flatBase, i);
      return flagcxUnhandledDeviceError;
    }
    mapping.importedHandleOwned[i] = 0;
  }
  return flagcxSuccess;
}
flagcxResult_t ducudaAdaptorSymFlatVaFree(void *flatBase, size_t allocSize,
                                          int nPeers) {
  if (flatBase == NULL)
    return flagcxSuccess;

  std::lock_guard<std::mutex> lock(gDucudaFlatMappingMtx);
  auto it = gDucudaFlatMappings.find(flatBase);
  if (it == gDucudaFlatMappings.end()) {
    WARN("DU VMM flat VA free failed: untracked base=%p", flatBase);
    return flagcxInvalidUsage;
  }
  DucudaFlatMapping &mapping = it->second;
  if (!ducudaFlatMappingMatches(mapping, allocSize, nPeers)) {
    WARN("DU VMM flat VA free failed: geometry mismatch base=%p "
         "expectedSize=%zu actualSize=%zu expectedPeers=%d actualPeers=%d",
         flatBase, mapping.allocSize, allocSize, mapping.nPeers, nPeers);
    return flagcxInvalidArgument;
  }
  for (int i = 0; i < nPeers; i++) {
    if (mapping.mappedSlots[i] || mapping.importedHandleOwned[i]) {
      WARN("DU VMM flat VA free blocked by owned slot: base=%p slot=%d "
           "mapped=%d importedHandle=%d",
           flatBase, i, (int)mapping.mappedSlots[i],
           (int)mapping.importedHandleOwned[i]);
      return flagcxInvalidUsage;
    }
  }
  if (mapping.vaOwned) {
    CUresult result =
        cuMemAddressFree((CUdeviceptr)flatBase, allocSize * (size_t)nPeers);
    if (result != CUDA_SUCCESS) {
      WARN("DU VMM flat VA free failed: stage=address-free result=%d "
           "base=%p size=%zu",
           (int)result, flatBase, allocSize * (size_t)nPeers);
      return flagcxUnhandledDeviceError;
    }
    mapping.vaOwned = false;
  }
  gDucudaFlatMappings.erase(it);
  return flagcxSuccess;
}
flagcxResult_t ducudaAdaptorSymMulticastSupported(int *supported) {
  // not supported on dcu
  if (supported == NULL)
    return flagcxInvalidArgument;

  if (supported)
    *supported = 0;
  return flagcxSuccess;
}
flagcxResult_t ducudaAdaptorSymMulticastCreate(size_t allocSize,
                                               int nLocalDevices,
                                               const int *localDeviceOrdinals,
                                               void **mcHandle,
                                               int *shareableFd) {
  // not supported on dcu
  if (mcHandle)
    *mcHandle = NULL;

  if (shareableFd)
    *shareableFd = -1;

  return flagcxNotSupported;
}
flagcxResult_t ducudaAdaptorSymMulticastBind(void *mcHandle, int importFd,
                                             void *physHandle, size_t allocSize,
                                             int localRank, int nLocalDevices,
                                             void **mcBase, size_t *mcMapSize) {
  // not supported on dcu
  if (mcBase)
    *mcBase = NULL;

  if (mcMapSize)
    *mcMapSize = 0;

  return flagcxNotSupported;
}
flagcxResult_t ducudaAdaptorSymMulticastTeardown(void *mcBase,
                                                 size_t mcMapSize) {
  // not supported on dcu
  return flagcxSuccess;
}
flagcxResult_t ducudaAdaptorSymMulticastMappingUnmap(void *mcBase,
                                                     size_t mcMapSize) {
  if (mcBase == NULL)
    return flagcxSuccess;
  DEVCHECK(cuMemUnmap((CUdeviceptr)mcBase, mcMapSize));
  return flagcxSuccess;
}
flagcxResult_t ducudaAdaptorSymMulticastVaFree(void *mcBase, size_t mcMapSize) {
  if (mcBase == NULL)
    return flagcxSuccess;
  DEVCHECK(cuMemAddressFree((CUdeviceptr)mcBase, mcMapSize));
  return flagcxSuccess;
}
flagcxResult_t ducudaAdaptorSymMulticastFree(void *mcHandle) {
  // not supported on dcu
  return flagcxSuccess;
}
flagcxResult_t ducudaAdaptorSymMulticastImport(int, void **mcHandle) {
  if (mcHandle != NULL)
    *mcHandle = NULL;
  return flagcxNotSupported;
}

flagcxResult_t ducudaAdaptorGetAddressRange(const void *ptr, void **base,
                                            size_t *size) {
  if (ptr == NULL || base == NULL || size == NULL)
    return flagcxInvalidArgument;

  CUdeviceptr allocationBase = 0;
  CUresult result =
      cuMemGetAddressRange(&allocationBase, size, (CUdeviceptr)ptr);
  if (result != CUDA_SUCCESS)
    return flagcxUnhandledDeviceError;
  *base = (void *)allocationBase;
  return flagcxSuccess;
}

struct flagcxDeviceAdaptor ducudaAdaptor {
  "DUCUDA",
      // Basic functions
      ducudaAdaptorDeviceSynchronize, ducudaAdaptorDeviceMemcpy,
      ducudaAdaptorDeviceMemset, ducudaAdaptorDeviceMalloc,
      ducudaAdaptorDeviceFree, ducudaAdaptorSetDevice, ducudaAdaptorGetDevice,
      ducudaAdaptorGetDeviceCount, ducudaAdaptorGetVendor,
      ducudaAdaptorHostGetDevicePointer,
      // GDR functions
      NULL, // flagcxResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // flagcxResult_t (*memHandleDestroy)(int dev, void *memHandle);
      ducudaAdaptorGdrMemAlloc, ducudaAdaptorGdrMemFree,
      NULL, // flagcxResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // flagcxResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      NULL, // flagcxResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t
            // sz);
      NULL, // flagcxResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);
      // Stream functions
      ducudaAdaptorStreamCreate, ducudaAdaptorStreamDestroy,
      ducudaAdaptorStreamCopy, ducudaAdaptorStreamFree,
      ducudaAdaptorStreamSynchronize, ducudaAdaptorStreamQuery,
      ducudaAdaptorStreamWaitEvent, ducudaAdaptorStreamWaitValue64,
      ducudaAdaptorStreamWriteValue64,
      // Event functions
      ducudaAdaptorEventCreate, ducudaAdaptorEventDestroy,
      ducudaAdaptorEventRecord, ducudaAdaptorEventSynchronize,
      ducudaAdaptorEventQuery, ducudaAdaptorEventElapsedTime,
      // IpcMemHandle functions
      ducudaAdaptorIpcMemHandleCreate, ducudaAdaptorIpcMemHandleGet,
      ducudaAdaptorIpcMemHandleOpen, ducudaAdaptorIpcMemHandleClose,
      ducudaAdaptorIpcMemHandleFree,
      // Kernel launch
      NULL, // flagcxResult_t (*launchKernel)(void *func, unsigned int block_x,
            // unsigned int block_y, unsigned int block_z, unsigned int grid_x,
            // unsigned int grid_y, unsigned int grid_z, void **args, size_t
            // share_mem, void *stream, void *memHandle);
      NULL, // flagcxResult_t (*copyArgsInit)(void **args);
      NULL, // flagcxResult_t (*copyArgsFree)(void *args);
      NULL, // flagcxResult_t
            // (*launchDeviceFunc)(flagcxStream_t stream,
            // void *args);
      // Others
      ducudaAdaptorGetDeviceProperties, // flagcxResult_t
                                        // (*getDeviceProperties)(struct
                                        // flagcxDevProps *props, int dev);
      ducudaAdaptorGetDevicePciBusId,   // flagcxResult_t
                                        // (*getDevicePciBusId)(char *pciBusId,
                                        // int len, int dev);
      ducudaAdaptorGetDeviceByPciBusId, // flagcxResult_t
                                        // (*getDeviceByPciBusId)(
                                        // int
                                        // *dev, const char *pciBusId);
      ducudaAdaptorLaunchHostFunc,
      // DMA buffer
      ducudaAdaptorDmaSupport, // flagcxResult_t (*dmaSupport)(bool
                               // *dmaBufferSupport);
      ducudaAdaptorMemGetHandleForAddressRange, // flagcxResult_t
                                                // (*memGetHandleForAddressRange)(void
                                                // *handleOut, void *buffer,
                                                // size_t size, unsigned long
                                                // long flags);
      ducudaAdaptorHostRegister,   // flagcxResult_t (*hostRegister)(void *,
                                   // size_t);
      ducudaAdaptorHostUnregister, // flagcxResult_t (*hostUnregister)(void *);
      // Symmetric memory VMM functions
      ducudaAdaptorSymPhysAlloc, ducudaAdaptorSymPhysFree,
      ducudaAdaptorSymFlatMap, ducudaAdaptorSymFlatUnmap,
      ducudaAdaptorSymMulticastSupported, ducudaAdaptorSymMulticastCreate,
      ducudaAdaptorSymMulticastBind, ducudaAdaptorSymMulticastTeardown,
      ducudaAdaptorSymMulticastFree,
      NULL, // flagcxResult_t (*getLastError)();
      ducudaAdaptorGetPointerType, ducudaAdaptorGetAddressRange,
      FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA,
      FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE, ducudaAdaptorSymMulticastImport,
      ducudaAdaptorSymFlatMappingUnmap, ducudaAdaptorSymFlatVaFree,
      ducudaAdaptorSymMulticastMappingUnmap, ducudaAdaptorSymMulticastVaFree,
      // Hygon requires the local post-READ flush for ordinary allocations as
      // well as VMM. Its current DU runtime cannot provide stream acquire for
      // incoming WRITEs, so strong WRITE consumers fail safely as unsupported.
      FLAGCX_GDR_READ_REQUIRES_FLUSH | FLAGCX_GDR_WRITE_REQUIRES_FLUSH,
      FLAGCX_GDR_DEVICE_DU, NULL, ducudaAdaptorCanAccessPeer,
};

#endif // USE_DU_ADAPTOR
