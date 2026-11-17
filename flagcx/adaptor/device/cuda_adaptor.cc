#include "nvidia_adaptor.h"

#ifdef USE_NVIDIA_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "param.h"
#include "segmented_vmm_allocation.h"
#include <limits>
#include <mutex>
#include <new>
#include <unistd.h>
#include <unordered_map>

static std::mutex gVmmHandleMapMtx;
struct CudaVmmAllocation {
  CUmemGenericAllocationHandle handle;
  size_t size;
  bool mappingOwned;
  bool vaOwned;
  bool handleOwned;
};
static std::unordered_map<void *, CudaVmmAllocation> gVmmHandleMap;

#if CUDART_VERSION >= 12010
struct CudaSegmentOps {
  typedef CUmemGenericAllocationHandle Handle;
  typedef CUmemAllocationProp Prop;

  static flagcxResult_t deviceOrdinal(int *device) {
    return cudaGetDevice(device) == cudaSuccess ? flagcxSuccess
                                                : flagcxUnhandledDeviceError;
  }
  static flagcxResult_t createHandle(size_t size, const Prop &prop,
                                     Handle *handle) {
    return cuMemCreate(handle, size, &prop, 0) == CUDA_SUCCESS
               ? flagcxSuccess
               : flagcxUnhandledDeviceError;
  }
  static flagcxResult_t reserveAddress(size_t size, size_t alignment,
                                       uintptr_t *base) {
    CUdeviceptr address = 0;
    if (cuMemAddressReserve(&address, size, alignment, 0, 0) != CUDA_SUCCESS)
      return flagcxUnhandledDeviceError;
    *base = static_cast<uintptr_t>(address);
    return flagcxSuccess;
  }
  static flagcxResult_t freeAddress(uintptr_t base, size_t size) {
    return cuMemAddressFree(static_cast<CUdeviceptr>(base), size) ==
                   CUDA_SUCCESS
               ? flagcxSuccess
               : flagcxUnhandledDeviceError;
  }
  static flagcxResult_t mapSegment(uintptr_t address, size_t size,
                                   Handle handle) {
    return cuMemMap(static_cast<CUdeviceptr>(address), size, 0, handle, 0) ==
                   CUDA_SUCCESS
               ? flagcxSuccess
               : flagcxUnhandledDeviceError;
  }
  static flagcxResult_t unmapSegment(uintptr_t address, size_t size) {
    return cuMemUnmap(static_cast<CUdeviceptr>(address), size) == CUDA_SUCCESS
               ? flagcxSuccess
               : flagcxUnhandledDeviceError;
  }
  static flagcxResult_t setAccess(uintptr_t address, size_t size) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
      return flagcxUnhandledDeviceError;
    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    return cuMemSetAccess(static_cast<CUdeviceptr>(address), size, &access,
                          1) == CUDA_SUCCESS
               ? flagcxSuccess
               : flagcxUnhandledDeviceError;
  }
  static flagcxResult_t importHandle(int fd, Handle *handle) {
    return cuMemImportFromShareableHandle(
               handle, reinterpret_cast<void *>(static_cast<uintptr_t>(fd)),
               CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) == CUDA_SUCCESS
               ? flagcxSuccess
               : flagcxUnhandledDeviceError;
  }
  static flagcxResult_t exportHandle(Handle handle, int *fd) {
    return cuMemExportToShareableHandle(
               fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0) ==
                   CUDA_SUCCESS
               ? flagcxSuccess
               : flagcxUnhandledDeviceError;
  }
  static flagcxResult_t releaseHandle(Handle handle) {
    return cuMemRelease(handle) == CUDA_SUCCESS ? flagcxSuccess
                                                : flagcxUnhandledDeviceError;
  }
};
typedef flagcxSegmentedVmmAllocation<CudaSegmentOps> CudaSegmentAllocation;
typedef flagcxSegmentedVmmMap<CudaSegmentOps> CudaSegmentMap;
#endif

constexpr uint32_t cudaVmmMrCapsForVersion(int cudartVersion) {
  return cudartVersion >= 12010
             ? FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA
             : FLAGCX_VMM_MR_CAP_NONE;
}

static_assert(cudaVmmMrCapsForVersion(12000) == FLAGCX_VMM_MR_CAP_NONE,
              "pre-12.1 CUDA allocations must not advertise VMM MR routes");
static_assert(cudaVmmMrCapsForVersion(12010) ==
                  (FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA),
              "CUDA 12.1+ allocations must advertise their VMM MR routes");

std::map<flagcxMemcpyType_t, cudaMemcpyKind> memcpy_type_map = {
    {flagcxMemcpyHostToDevice, cudaMemcpyHostToDevice},
    {flagcxMemcpyDeviceToHost, cudaMemcpyDeviceToHost},
    {flagcxMemcpyDeviceToDevice, cudaMemcpyDeviceToDevice},
};

flagcxResult_t cudaAdaptorDeviceSynchronize() {
  DEVCHECK(cudaDeviceSynchronize());
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
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

flagcxResult_t cudaAdaptorDeviceMemset(void *ptr, int value, size_t size,
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

flagcxResult_t cudaAdaptorDeviceMalloc(void **ptr, size_t size,
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

flagcxResult_t cudaAdaptorDeviceFree(void *ptr, flagcxMemType_t type,
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

flagcxResult_t cudaAdaptorSetDevice(int dev) {
  DEVCHECK(cudaSetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorGetDevice(int *dev) {
  DEVCHECK(cudaGetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorGetDeviceCount(int *count) {
  DEVCHECK(cudaGetDeviceCount(count));
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "NVIDIA");
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorCanAccessPeer(int srcDev, int dstDev,
                                        int *canAccess) {
  if (canAccess == NULL)
    return flagcxInvalidArgument;
  DEVCHECK(cudaDeviceCanAccessPeer(canAccess, srcDev, dstDev));
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorHostGetDevicePointer(void **pDevice, void *pHost) {
  if (pDevice == NULL || pHost == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaHostGetDevicePointer(pDevice, pHost, 0));
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorGdrMemAlloc(void **ptr, size_t size,
                                      void *memHandle) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
#if CUDART_VERSION >= 12010
  if (!flagcxParamVmmEnable()) {
    DEVCHECK(cudaMalloc(ptr, size));
    return flagcxSuccess;
  }

  size_t memGran = 0;
  CUdevice currentDev;
  CUmemAllocationProp memprop = {};
  CUmemGenericAllocationHandle handle = (CUmemGenericAllocationHandle)-1;
  int cudaDev;
  int flag;
  CUresult cuRes;

  DEVCHECK(cudaGetDevice(&cudaDev));
  FLAGCXCHECK(CudaSegmentAllocation::retryPending(cudaDev));
  DEVCHECK(cuDeviceGet(&currentDev, cudaDev));

  size_t handleSize = size;
  // Symmetric-memory handle exchange and DMA-BUF export both use POSIX file
  // descriptors.  Do not add FABRIC merely because the device advertises it:
  // creating a FABRIC-exportable allocation also requires an IMEX channel,
  // which is independent of the device capability and is not needed here.
  const int requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  memprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  memprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  memprop.requestedHandleTypes =
      (CUmemAllocationHandleType)requestedHandleTypes;
  memprop.location.id = currentDev;
  // Query device to see if RDMA support is available
  flag = 0;
  DEVCHECK(cuDeviceGetAttribute(
      &flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED,
      currentDev));
  INFO(FLAGCX_INIT,
       "[gdrMemAlloc] dev=%d GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED=%d "
       "size=%zu",
       cudaDev, flag, size);
  if (flag)
    memprop.allocFlags.gpuDirectRDMACapable = 1;
  DEVCHECK(cuMemGetAllocationGranularity(&memGran, &memprop,
                                         CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  ALIGN_SIZE(handleSize, memGran);
  size_t segmentBytes = 0;
  FLAGCXCHECK(flagcxVmmSegmentBytes(memGran, &segmentBytes));
  if (segmentBytes != 0 && segmentBytes < handleSize) {
    bool multiSegmentEnabled = true;
    FLAGCXCHECK(flagcxVmmMultiSegmentEnabled(true, &multiSegmentEnabled));
    if (!multiSegmentEnabled) {
      *ptr = NULL;
      return flagcxNotSupported;
    }
    return CudaSegmentAllocation::allocate(ptr, handleSize, memGran,
                                           segmentBytes, memprop);
  }
  INFO(FLAGCX_INIT,
       "[gdrMemAlloc] memGran=%zu handleSize=%zu gpuDirectRDMACapable=%d",
       memGran, handleSize, (int)memprop.allocFlags.gpuDirectRDMACapable);
  /* Allocate the physical memory on the device */
  DEVCHECK(cuMemCreate(&handle, handleSize, &memprop, 0));
  /* Reserve a virtual address range */
  cuRes = cuMemAddressReserve((CUdeviceptr *)ptr, handleSize, memGran, 0, 0);
  if (cuRes != CUDA_SUCCESS) {
    WARN("[gdrMemAlloc] cuMemAddressReserve FAILED: %d", (int)cuRes);
    cuMemRelease(handle);
    return flagcxUnhandledDeviceError;
  }
  /* Map the virtual address range to the physical allocation */
  cuRes = cuMemMap((CUdeviceptr)*ptr, handleSize, 0, handle, 0);
  if (cuRes != CUDA_SUCCESS) {
    WARN("[gdrMemAlloc] cuMemMap FAILED: %d", (int)cuRes);
    cuMemAddressFree((CUdeviceptr)*ptr, handleSize);
    cuMemRelease(handle);
    *ptr = NULL;
    return flagcxUnhandledDeviceError;
  }
  /* Set access for the current device */
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = currentDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  cuRes = cuMemSetAccess((CUdeviceptr)*ptr, handleSize, &accessDesc, 1);
  if (cuRes != CUDA_SUCCESS) {
    WARN("[gdrMemAlloc] cuMemSetAccess FAILED: %d", (int)cuRes);
    cuMemUnmap((CUdeviceptr)*ptr, handleSize);
    cuMemAddressFree((CUdeviceptr)*ptr, handleSize);
    cuMemRelease(handle);
    *ptr = NULL;
    return flagcxUnhandledDeviceError;
  }
  INFO(FLAGCX_INIT, "[gdrMemAlloc] VMM alloc OK: ptr=%p size=%zu", *ptr,
       handleSize);
  /* Retain the handle so cuMemGetHandleForAddressRange can export DMA-BUF fds.
     Released in cudaAdaptorGdrMemFree. */
  bool tracked = false;
  try {
    std::lock_guard<std::mutex> lk(gVmmHandleMapMtx);
    tracked = gVmmHandleMap
                  .emplace(*ptr, CudaVmmAllocation{handle, handleSize, true,
                                                   true, true})
                  .second;
  } catch (const std::bad_alloc &) {
  }
  if (!tracked) {
    cuMemUnmap((CUdeviceptr)*ptr, handleSize);
    cuMemAddressFree((CUdeviceptr)*ptr, handleSize);
    cuMemRelease(handle);
    *ptr = NULL;
    return flagcxSystemError;
  }
#else
  DEVCHECK(cudaMalloc(ptr, size));
  cudaPointerAttributes attrs;
  DEVCHECK(cudaPointerGetAttributes(&attrs, *ptr));
  unsigned flags = 1;
  DEVCHECK(cuPointerSetAttribute(&flags, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                 (CUdeviceptr)attrs.devicePointer));
#endif
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return flagcxSuccess;
  }
#if CUDART_VERSION >= 12010
  bool segmented = false;
  FLAGCXCHECK(CudaSegmentAllocation::freeAllocation(ptr, &segmented));
  if (segmented)
    return flagcxSuccess;
  std::lock_guard<std::mutex> lk(gVmmHandleMapMtx);
  auto it = gVmmHandleMap.find(ptr);
  if (it == gVmmHandleMap.end()) {
    DEVCHECK(cudaFree(ptr));
    return flagcxSuccess;
  }

  CudaVmmAllocation &allocation = it->second;
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
  if (allocation.handleOwned) {
    if (cuMemRelease(allocation.handle) != CUDA_SUCCESS)
      return flagcxUnhandledDeviceError;
    allocation.handleOwned = false;
  }
  gVmmHandleMap.erase(it);
#else
  DEVCHECK(cudaFree(ptr));
#endif
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorStreamCreate(flagcxStream_t *stream) {
  (*stream) = NULL;
  flagcxCalloc(stream, 1);
  DEVCHECK(cudaStreamCreateWithFlags((cudaStream_t *)(*stream),
                                     cudaStreamNonBlocking));
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorStreamDestroy(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(cudaStreamDestroy(stream->base));
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorStreamCopy(flagcxStream_t *newStream,
                                     void *oldStream) {
  (*newStream) = NULL;
  flagcxCalloc(newStream, 1);
  (*newStream)->base = (cudaStream_t)oldStream;
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorStreamFree(flagcxStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorStreamSynchronize(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(cudaStreamSynchronize(stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorStreamQuery(flagcxStream_t stream) {
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

flagcxResult_t cudaAdaptorStreamWaitEvent(flagcxStream_t stream,
                                          flagcxEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(
        cudaStreamWaitEvent(stream->base, event->base, cudaEventWaitDefault));
  }
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorStreamWaitValue64(flagcxStream_t stream, void *addr,
                                            uint64_t value, int flags) {
  if (stream == NULL || addr == NULL)
    return flagcxInvalidArgument;
  if (flags & ~FLAGCX_STREAM_WAIT_VALUE_FLUSH_REMOTE_WRITES)
    return flagcxInvalidArgument;

  unsigned int waitFlags = CU_STREAM_WAIT_VALUE_GEQ;
  if (flags & FLAGCX_STREAM_WAIT_VALUE_FLUSH_REMOTE_WRITES) {
    // The symbol being present in the CUDA headers is not a runtime guarantee.
    // Match NVSHMEM's capability check before requesting the stream acquire.
    CUdevice device;
    int canFlushRemoteWrites = 0;
    CUresult probe = cuCtxGetDevice(&device);
    if (probe != CUDA_SUCCESS)
      return probe == CUDA_ERROR_NOT_SUPPORTED ? flagcxNotSupported
                                               : flagcxUnhandledDeviceError;
    probe = cuDeviceGetAttribute(&canFlushRemoteWrites,
                                 CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES,
                                 device);
    if (probe != CUDA_SUCCESS || canFlushRemoteWrites == 0)
      return flagcxNotSupported;
    waitFlags |= CU_STREAM_WAIT_VALUE_FLUSH;
  }

  CUstream cuStream = (CUstream)(stream->base);
  CUresult err =
      cuStreamWaitValue64(cuStream, (CUdeviceptr)addr, value, waitFlags);
  if (err == CUDA_SUCCESS)
    return flagcxSuccess;
  if (err == CUDA_ERROR_NOT_SUPPORTED)
    return flagcxNotSupported;
  return flagcxUnhandledDeviceError;
}

flagcxResult_t cudaAdaptorStreamWriteValue64(flagcxStream_t stream, void *addr,
                                             uint64_t value, int flags) {
  (void)flags;
  if (stream == NULL || addr == NULL)
    return flagcxInvalidArgument;
  CUstream cuStream = (CUstream)(stream->base);
  CUresult err = cuStreamWriteValue64(cuStream, (CUdeviceptr)addr, value,
                                      CU_STREAM_WRITE_VALUE_DEFAULT);
  if (err == CUDA_SUCCESS)
    return flagcxSuccess;
  if (err == CUDA_ERROR_NOT_SUPPORTED)
    return flagcxNotSupported;
  return flagcxUnhandledDeviceError;
}

flagcxResult_t cudaAdaptorEventCreate(flagcxEvent_t *event,
                                      flagcxEventType_t eventType) {
  (*event) = NULL;
  flagcxCalloc(event, 1);
  const unsigned int flags = (eventType == flagcxEventDefault)
                                 ? cudaEventDefault
                                 : cudaEventDisableTiming;
  DEVCHECK(cudaEventCreateWithFlags(&((*event)->base), flags));
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorEventDestroy(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(cudaEventDestroy(event->base));
    free(event);
    event = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorEventRecord(flagcxEvent_t event,
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

flagcxResult_t cudaAdaptorEventSynchronize(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(cudaEventSynchronize(event->base));
  }
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorEventQuery(flagcxEvent_t event) {
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

flagcxResult_t cudaAdaptorEventElapsedTime(float *ms, flagcxEvent_t start,
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

flagcxResult_t cudaAdaptorIpcMemHandleCreate(flagcxIpcMemHandle_t *handle,
                                             size_t *size) {
  flagcxCalloc(handle, 1);
  if (size != NULL) {
    *size = sizeof(cudaIpcMemHandle_t);
  }
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorIpcMemHandleGet(flagcxIpcMemHandle_t handle,
                                          void *devPtr) {
  if (handle == NULL || devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaIpcGetMemHandle(&handle->base, devPtr));
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorIpcMemHandleOpen(flagcxIpcMemHandle_t handle,
                                           void **devPtr) {
  if (handle == NULL || devPtr == NULL || *devPtr != NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaIpcOpenMemHandle(devPtr, handle->base,
                                cudaIpcMemLazyEnablePeerAccess));
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorIpcMemHandleClose(void *devPtr) {
  if (devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaIpcCloseMemHandle(devPtr));
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorIpcMemHandleFree(flagcxIpcMemHandle_t handle) {
  if (handle != NULL) {
    free(handle);
  }
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorLaunchHostFunc(flagcxStream_t stream,
                                         void (*fn)(void *), void *args) {
  if (stream != NULL) {
    DEVCHECK(cudaLaunchHostFunc(stream->base, fn, args));
  }
  return flagcxSuccess;
}
flagcxResult_t cudaAdaptorLaunchDeviceFunc(flagcxStream_t stream,
                                           flagcxLaunchFunc_t fn, void *args) {
  if (stream != NULL) {
    fn(stream, args);
  }
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorGetDeviceProperties(struct flagcxDevProps *props,
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

flagcxResult_t cudaAdaptorGetDeviceArchitecture(int dev, int *architecture) {
  if (architecture == NULL)
    return flagcxInvalidArgument;
  cudaDeviceProp devProp;
  DEVCHECK(cudaGetDeviceProperties(&devProp, dev));
  *architecture = devProp.major * 10 + devProp.minor;
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorGetDevicePciBusId(char *pciBusId, int len, int dev) {
  if (pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaDeviceGetPCIBusId(pciBusId, len, dev));
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorGetDeviceByPciBusId(int *dev, const char *pciBusId) {
  if (dev == NULL || pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaDeviceGetByPCIBusId(dev, pciBusId));
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorDmaSupport(bool *dmaBufferSupport) {
  if (dmaBufferSupport == NULL)
    return flagcxInvalidArgument;

#if CUDA_VERSION >= 11070
  int flag = 0;
  CUdevice dev;
  int cudaDriverVersion = 0;

  CUresult cuRes = cuDriverGetVersion(&cudaDriverVersion);
  if (cuRes != CUDA_SUCCESS || cudaDriverVersion < 11070) {
    *dmaBufferSupport = false;
    return flagcxSuccess;
  }

  int deviceId = 0;
  if (cudaGetDevice(&deviceId) != cudaSuccess) {
    *dmaBufferSupport = false;
    return flagcxSuccess;
  }

  CUresult devRes = cuDeviceGet(&dev, deviceId);
  if (devRes != CUDA_SUCCESS) {
    *dmaBufferSupport = false;
    return flagcxSuccess;
  }

  CUresult attrRes =
      cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, dev);
  if (attrRes != CUDA_SUCCESS || flag == 0) {
    *dmaBufferSupport = false;
    return flagcxSuccess;
  }

  *dmaBufferSupport = true;
  return flagcxSuccess;

#else
  *dmaBufferSupport = false;
  return flagcxSuccess;
#endif
}

flagcxResult_t
cudaAdaptorMemGetHandleForAddressRange(void *handleOut, void *buffer,
                                       size_t size, unsigned long long flags) {
  if (handleOut == NULL || buffer == NULL || size == 0)
    return flagcxInvalidArgument;
  const long hostPageSize = sysconf(_SC_PAGESIZE);
  if (hostPageSize <= 0)
    return flagcxSystemError;
  const size_t pageSize = static_cast<size_t>(hostPageSize);
  if (reinterpret_cast<uintptr_t>(buffer) % pageSize != 0 ||
      size % pageSize != 0)
    return flagcxInvalidArgument;

  CUdeviceptr dptr = (CUdeviceptr)buffer;
  CUresult err = cuMemGetHandleForAddressRange(
      handleOut, dptr, size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, flags);
  // A page-aligned range may still be ineligible for DMA-BUF export. The MR
  // caller can then use VA only when that allocation has a safe VA route.
  if (err == CUDA_ERROR_NOT_SUPPORTED ||
      (err == CUDA_ERROR_INVALID_VALUE && flags == 0)) {
    INFO(FLAGCX_REG,
         "CUDA DMA-BUF export unavailable: ptr=%p size=%zu result=%d", buffer,
         size, static_cast<int>(err));
    return flagcxNotSupported;
  }
  if (err != CUDA_SUCCESS)
    WARN("CUDA DMA-BUF export failed: ptr=%p size=%zu result=%d", buffer, size,
         static_cast<int>(err));
  return err == CUDA_SUCCESS ? flagcxSuccess : flagcxUnhandledDeviceError;
}

flagcxResult_t cudaAdaptorHostRegister(void *ptr, size_t size) {
  if (ptr == NULL || size == 0) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaHostRegister(ptr, size, cudaHostRegisterMapped));
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorHostUnregister(void *ptr) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaHostUnregister(ptr));
  return flagcxSuccess;
}

// ==========================================================================
// Symmetric memory VMM functions
// ==========================================================================

#if CUDART_VERSION >= 12010

flagcxResult_t cudaAdaptorSymPhysAlloc(void *ptr, size_t size,
                                       void **physHandle, void *shareableHandle,
                                       size_t *handleSize, size_t *allocSize) {
  if (ptr == NULL || physHandle == NULL || shareableHandle == NULL ||
      handleSize == NULL || allocSize == NULL)
    return flagcxInvalidArgument;

  CUmemGenericAllocationHandle *cuHandle =
      (CUmemGenericAllocationHandle *)malloc(
          sizeof(CUmemGenericAllocationHandle));
  if (cuHandle == NULL)
    return flagcxSystemError;

  // Retain the physical allocation handle from the VMM-backed pointer
  CUresult retainRes = cuMemRetainAllocationHandle(cuHandle, ptr);
  if (retainRes != CUDA_SUCCESS) {
    WARN("[symPhysAlloc] cuMemRetainAllocationHandle FAILED: %d ptr=%p",
         (int)retainRes, ptr);
    free(cuHandle);
    if (retainRes == CUDA_ERROR_INVALID_VALUE ||
        retainRes == CUDA_ERROR_NOT_SUPPORTED)
      return flagcxNotSupported;
    return flagcxUnhandledDeviceError;
  }

  // Discover actual physical allocation size (already granularity-aligned)
  CUdeviceptr allocationBase = 0;
  size_t actualAllocSize = 0;
  CUresult rangeRes =
      cuMemGetAddressRange(&allocationBase, &actualAllocSize, (CUdeviceptr)ptr);
  if (rangeRes != CUDA_SUCCESS) {
    cuMemRelease(*cuHandle);
    free(cuHandle);
    return flagcxUnhandledDeviceError;
  }
  const CUdeviceptr address = (CUdeviceptr)ptr;
  if (allocationBase == 0 || actualAllocSize == 0 || address < allocationBase ||
      address - allocationBase > actualAllocSize ||
      size > actualAllocSize - (address - allocationBase)) {
    cuMemRelease(*cuHandle);
    free(cuHandle);
    return flagcxInvalidUsage;
  }
  *allocSize = actualAllocSize;

  // Export as POSIX fd for IPC sharing
  if (*handleSize < sizeof(int)) {
    cuMemRelease(*cuHandle);
    free(cuHandle);
    return flagcxInvalidArgument;
  }
  CUresult exportRes = cuMemExportToShareableHandle(
      shareableHandle, *cuHandle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
  if (exportRes != CUDA_SUCCESS) {
    WARN("[symPhysAlloc] cuMemExportToShareableHandle FAILED: %d",
         (int)exportRes);
    cuMemRelease(*cuHandle);
    free(cuHandle);
    return flagcxUnhandledDeviceError;
  }
  INFO(FLAGCX_INIT, "[symPhysAlloc] ptr=%p allocSize=%zu fd=%d", ptr,
       actualAllocSize, *(int *)shareableHandle);
  *handleSize = sizeof(int); // POSIX fd is an int
  *physHandle = cuHandle;
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorSymPhysFree(void *physHandle) {
  if (physHandle == NULL)
    return flagcxSuccess;
  CUmemGenericAllocationHandle *cuHandle =
      (CUmemGenericAllocationHandle *)physHandle;
  if (cuMemRelease(*cuHandle) != CUDA_SUCCESS)
    return flagcxUnhandledDeviceError;
  free(cuHandle);
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorSymPhysDescribe(void *ptr, size_t size,
                                          void **segmentBundle,
                                          const flagcxSymSegmentMeta **segments,
                                          int *segmentCount, size_t *userOffset,
                                          size_t *slotAlignment) {
  if (ptr == NULL || size == 0 || segmentBundle == NULL || segments == NULL ||
      segmentCount == NULL || userOffset == NULL || slotAlignment == NULL)
    return flagcxInvalidArgument;
  *segmentBundle = NULL;
  *segments = NULL;
  *segmentCount = 0;
  *slotAlignment = 0;
  const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  if (size > std::numeric_limits<uintptr_t>::max() - address)
    return flagcxInvalidArgument;

  CudaSegmentMap::Bundle *bundle = new (std::nothrow) CudaSegmentMap::Bundle();
  if (bundle == NULL)
    return flagcxSystemError;
  *segmentBundle = bundle;
  bool found = false;
  flagcxResult_t result =
      CudaSegmentAllocation::describe(ptr, size, bundle, userOffset, &found);
  if (result != flagcxSuccess)
    return result;
  if (!found) {
    std::lock_guard<std::mutex> lock(gVmmHandleMapMtx);
    for (std::unordered_map<void *, CudaVmmAllocation>::const_iterator it =
             gVmmHandleMap.begin();
         it != gVmmHandleMap.end(); ++it) {
      const uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
      const CudaVmmAllocation &allocation = it->second;
      if (address < base || address - base > allocation.size ||
          size > allocation.size - (address - base))
        continue;
      found = true;
      if (!allocation.mappingOwned || !allocation.vaOwned ||
          !allocation.handleOwned)
        return flagcxInvalidUsage;
      try {
        bundle->segments.push_back(flagcxSymSegmentMeta{0, allocation.size});
        bundle->handleOwned.push_back(0);
        bundle->handles.push_back(allocation.handle);
      } catch (const std::bad_alloc &) {
        return flagcxSystemError;
      }
      *userOffset = address - base;
      break;
    }
  }
  if (!found) {
    // An external VMM mapping has unknown physical offsets and cannot be
    // described safely. Only a successful legacy IPC export proves that an
    // untracked allocation may use the ordinary IPC path.
    CUdeviceptr allocationBase = 0;
    size_t allocationSize = 0;
    if (cuMemGetAddressRange(&allocationBase, &allocationSize,
                             (CUdeviceptr)ptr) == CUDA_SUCCESS &&
        address >= allocationBase &&
        address - allocationBase <= allocationSize &&
        size <= allocationSize - (address - allocationBase)) {
      cudaIpcMemHandle_t ipcHandle;
      if (cudaIpcGetMemHandle(&ipcHandle, (void *)allocationBase) ==
          cudaSuccess) {
        delete bundle;
        *segmentBundle = NULL;
      } else {
        cudaGetLastError();
      }
    }
    // Keep an empty bundle for an unknown mapping to block IPC fallback.
    return flagcxNotSupported;
  }
  if (bundle->segments.size() > 1) {
    bool multiSegmentEnabled = true;
    FLAGCXCHECK(flagcxVmmMultiSegmentEnabled(true, &multiSegmentEnabled));
    if (!multiSegmentEnabled)
      return flagcxNotSupported;
  }
  // Ordinary allocations have already returned without requiring VMM support.
  int cudaDev = 0;
  CUdevice device = 0;
  if (cudaGetDevice(&cudaDev) != cudaSuccess ||
      cuDeviceGet(&device, cudaDev) != CUDA_SUCCESS)
    return flagcxUnhandledDeviceError;
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  if (cuMemGetAllocationGranularity(slotAlignment, &prop,
                                    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) !=
          CUDA_SUCCESS ||
      *slotAlignment == 0)
    return flagcxUnhandledDeviceError;
  if (bundle->segments.empty() ||
      bundle->segments.size() >
          static_cast<size_t>(std::numeric_limits<int>::max()))
    return flagcxInvalidUsage;
  *segments = bundle->segments.data();
  *segmentCount = static_cast<int>(bundle->segments.size());
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorSymPhysExportSegment(void *bundle, int index,
                                               int *fd) {
  return CudaSegmentMap::exportSegment(bundle, index, fd);
}
flagcxResult_t cudaAdaptorSymPhysFreeSegments(void *bundle) {
  return CudaSegmentMap::freeBundle(bundle);
}
flagcxResult_t cudaAdaptorSymPhysSegmentCount(const void *ptr, size_t size,
                                              int *count) {
  flagcxResult_t result = CudaSegmentAllocation::segmentCount(ptr, size, count);
  if (result != flagcxNotSupported)
    return result;
  const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  std::lock_guard<std::mutex> lock(gVmmHandleMapMtx);
  for (std::unordered_map<void *, CudaVmmAllocation>::const_iterator it =
           gVmmHandleMap.begin();
       it != gVmmHandleMap.end(); ++it) {
    const uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
    if (address < base || address - base > it->second.size ||
        size > it->second.size - (address - base))
      continue;
    *count = 1;
    return flagcxSuccess;
  }
  return flagcxNotSupported;
}
flagcxResult_t cudaAdaptorSymFlatMapSegments(
    const int *fds, int nPeers, int self, void *bundle,
    const flagcxSymRankLayout *layouts, const flagcxSymSegmentMeta *segments,
    int totalSegments, size_t stride, size_t alignment, void **flatBase) {
  return CudaSegmentMap::map(fds, nPeers, self, bundle, layouts, segments,
                             totalSegments, stride, alignment, flatBase);
}
flagcxResult_t cudaAdaptorSymFlatMapSegmentsBegin(
    int nPeers, int self, void *bundle, const flagcxSymRankLayout *layouts,
    const flagcxSymSegmentMeta *segments, int totalSegments, size_t stride,
    size_t alignment, void **flatBase) {
  return CudaSegmentMap::begin(nPeers, self, bundle, layouts, segments,
                               totalSegments, stride, alignment, flatBase);
}
flagcxResult_t cudaAdaptorSymFlatMapSegmentOne(void *flatBase, int index,
                                               int fd) {
  return CudaSegmentMap::mapOne(flatBase, index, fd);
}
flagcxResult_t cudaAdaptorSymFlatMappingUnmapSegments(void *flatBase) {
  return CudaSegmentMap::unmap(flatBase);
}
flagcxResult_t cudaAdaptorSymFlatVaFreeSegments(void *flatBase) {
  return CudaSegmentMap::freeVa(flatBase);
}

flagcxResult_t cudaAdaptorSymLsaGetCapacity(size_t *totalBytes,
                                            size_t *alignment) {
  if (totalBytes == NULL || alignment == NULL)
    return flagcxInvalidArgument;
  size_t freeBytes = 0;
  if (cudaMemGetInfo(&freeBytes, totalBytes) != cudaSuccess)
    return flagcxUnhandledDeviceError;
  *alignment = 512ULL * 1024 * 1024;
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorSymLsaVaReserve(size_t size, size_t alignment,
                                          void **base) {
  if (base == NULL)
    return flagcxInvalidArgument;
  uintptr_t address = 0;
  flagcxResult_t result =
      CudaSegmentOps::reserveAddress(size, alignment, &address);
  *base = result == flagcxSuccess ? reinterpret_cast<void *>(address) : NULL;
  return result;
}

flagcxResult_t cudaAdaptorSymLsaVaFree(void *base, size_t size) {
  return CudaSegmentOps::freeAddress(reinterpret_cast<uintptr_t>(base), size);
}

flagcxResult_t cudaAdaptorSymLsaMapBeginAt(
    void *windowBase, int nPeers, int self, void *bundle,
    const flagcxSymRankLayout *layouts, const flagcxSymSegmentMeta *segments,
    int totalSegments, size_t stride, size_t alignment, void **mappingBase) {
  return CudaSegmentMap::beginAt(windowBase, nPeers, self, bundle, layouts,
                                 segments, totalSegments, stride, alignment,
                                 mappingBase);
}

flagcxResult_t cudaAdaptorSymFlatMap(void *peerHandles[], int nPeers,
                                     int selfIndex, void *selfPhysHandle,
                                     size_t allocSize, void **flatBase) {
  if (peerHandles == NULL || selfPhysHandle == NULL || flatBase == NULL ||
      nPeers <= 0 || allocSize == 0)
    return flagcxInvalidArgument;

  CUmemGenericAllocationHandle selfHandle =
      *(CUmemGenericAllocationHandle *)selfPhysHandle;

  // allocSize is already granularity-aligned (from cuMemGetAddressRange)
  size_t totalSize = allocSize * nPeers;

  // Reserve the full VA range
  CUdeviceptr base = 0;
  CUresult result = cuMemAddressReserve(&base, totalSize, 0, 0, 0);
  if (result != CUDA_SUCCESS)
    return flagcxUnhandledDeviceError;

  // Import and map each peer's physical memory
  int cudaDev;
  if (cudaGetDevice(&cudaDev) != cudaSuccess) {
    cuMemAddressFree(base, totalSize);
    return flagcxUnhandledDeviceError;
  }
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  int mappedPeers = 0;
  for (int i = 0; i < nPeers; i++) {
    CUmemGenericAllocationHandle peerHandle;
    bool imported = i != selfIndex;
    if (i == selfIndex) {
      peerHandle = selfHandle;
    } else {
      int fd = *(int *)peerHandles[i];
      result = cuMemImportFromShareableHandle(
          &peerHandle, (void *)(uintptr_t)fd,
          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
      if (result != CUDA_SUCCESS)
        goto rollback;
    }
    CUdeviceptr slot = base + (CUdeviceptr)i * allocSize;
    result = cuMemMap(slot, allocSize, 0, peerHandle, 0);
    if (result != CUDA_SUCCESS) {
      if (imported)
        cuMemRelease(peerHandle);
      goto rollback;
    }
    mappedPeers++;
    result = cuMemSetAccess(slot, allocSize, &accessDesc, 1);
    if (imported)
      cuMemRelease(peerHandle);
    if (result != CUDA_SUCCESS)
      goto rollback;
  }

  *flatBase = (void *)base;
  return flagcxSuccess;

rollback:
  for (int i = 0; i < mappedPeers; i++)
    cuMemUnmap(base + (CUdeviceptr)i * allocSize, allocSize);
  cuMemAddressFree(base, totalSize);
  return flagcxUnhandledDeviceError;
}

flagcxResult_t cudaAdaptorSymFlatUnmap(void *flatBase, size_t allocSize,
                                       int nPeers) {
  if (flatBase == NULL)
    return flagcxSuccess;
  CUdeviceptr base = (CUdeviceptr)flatBase;
  size_t totalSize = allocSize * nPeers;
  DEVCHECK(cuMemUnmap(base, totalSize));
  DEVCHECK(cuMemAddressFree(base, totalSize));
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorSymFlatMappingUnmap(void *flatBase, size_t allocSize,
                                              int nPeers) {
  if (flatBase == NULL)
    return flagcxSuccess;
  DEVCHECK(cuMemUnmap((CUdeviceptr)flatBase, allocSize * nPeers));
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorSymFlatVaFree(void *flatBase, size_t allocSize,
                                        int nPeers) {
  if (flatBase == NULL)
    return flagcxSuccess;
  DEVCHECK(cuMemAddressFree((CUdeviceptr)flatBase, allocSize * nPeers));
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorSymMulticastSupported(int *supported) {
  if (supported == NULL)
    return flagcxInvalidArgument;
  *supported = 0;
  int cudaDev;
  DEVCHECK(cudaGetDevice(&cudaDev));
  CUdevice dev;
  DEVCHECK(cuDeviceGet(&dev, cudaDev));
  CUresult res = cuDeviceGetAttribute(
      supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev);
  if (res != CUDA_SUCCESS)
    *supported = 0;
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorSymMulticastCreate(size_t allocSize,
                                             int nLocalDevices,
                                             const int *localDeviceOrdinals,
                                             void **mcHandle,
                                             int *shareableFd) {
  if (mcHandle == NULL || shareableFd == NULL || nLocalDevices <= 0 ||
      localDeviceOrdinals == NULL)
    return flagcxInvalidArgument;
  *mcHandle = NULL;
  *shareableFd = -1;

  CUmemGenericAllocationHandle handle = 0;
  int fd = -1;
  CUresult err;

  // Get multicast granularity and align size
  CUmulticastObjectProp mcProp = {};
  mcProp.numDevices = (unsigned int)nLocalDevices;
  mcProp.size = allocSize;
  mcProp.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  size_t mcGran = 0;
  err = cuMulticastGetGranularity(&mcGran, &mcProp,
                                  CU_MULTICAST_GRANULARITY_RECOMMENDED);
  if (err != CUDA_SUCCESS)
    return flagcxUnhandledDeviceError;
  mcProp.size = ((allocSize + mcGran - 1) / mcGran) * mcGran;

  err = cuMulticastCreate(&handle, &mcProp);
  if (err != CUDA_SUCCESS)
    return flagcxUnhandledDeviceError;

  // Add all local devices using explicit ordinals
  for (int i = 0; i < nLocalDevices; i++) {
    CUdevice peerDev;
    err = cuDeviceGet(&peerDev, localDeviceOrdinals[i]);
    if (err != CUDA_SUCCESS)
      goto cleanup_handle;
    err = cuMulticastAddDevice(handle, peerDev);
    if (err != CUDA_SUCCESS)
      goto cleanup_handle;
  }

  // Export as POSIX FD for sharing with peers
  err = cuMemExportToShareableHandle(
      &fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
  if (err != CUDA_SUCCESS)
    goto cleanup_handle;

  // Store handle as heap-allocated value
  {
    CUmemGenericAllocationHandle *handlePtr =
        (CUmemGenericAllocationHandle *)malloc(
            sizeof(CUmemGenericAllocationHandle));
    if (handlePtr == NULL)
      goto cleanup_fd;

    *handlePtr = handle;
    *mcHandle = handlePtr;
    *shareableFd = fd;
  }
  return flagcxSuccess;

cleanup_fd:
  close(fd);
cleanup_handle:
  cuMemRelease(handle);
  return flagcxUnhandledDeviceError;
}

flagcxResult_t cudaAdaptorSymMulticastImport(int importFd, void **mcHandle) {
  if (importFd < 0 || mcHandle == NULL)
    return flagcxInvalidArgument;
  *mcHandle = NULL;

  CUmemGenericAllocationHandle handle = 0;
  CUresult res =
      cuMemImportFromShareableHandle(&handle, (void *)(intptr_t)importFd,
                                     CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
  if (res != CUDA_SUCCESS) {
    WARN("symMulticastImport: cuMemImportFromShareableHandle failed: %d", res);
    return flagcxUnhandledDeviceError;
  }

  CUmemGenericAllocationHandle *handlePtr =
      (CUmemGenericAllocationHandle *)malloc(
          sizeof(CUmemGenericAllocationHandle));
  if (handlePtr == NULL) {
    cuMemRelease(handle);
    return flagcxSystemError;
  }
  *handlePtr = handle;
  *mcHandle = handlePtr;
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorSymMulticastBind(void *mcHandle, int importFd,
                                           void *physHandle, size_t allocSize,
                                           int localRank, int nLocalDevices,
                                           void **mcBase, size_t *mcMapSize) {
  if (mcBase == NULL || physHandle == NULL || mcMapSize == NULL)
    return flagcxInvalidArgument;
  *mcBase = NULL;
  *mcMapSize = 0;

  CUmemGenericAllocationHandle cuMcHandle;
  bool imported = (mcHandle == NULL);

  if (mcHandle != NULL) {
    // Rank 0: already has the handle from symMulticastCreate
    cuMcHandle = *(CUmemGenericAllocationHandle *)mcHandle;
  } else {
    // Other ranks: import from FD
    if (importFd < 0)
      return flagcxInvalidArgument;
    CUresult res = cuMemImportFromShareableHandle(
        &cuMcHandle, (void *)(intptr_t)importFd,
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    if (res != CUDA_SUCCESS) {
      WARN("symMulticastBind: cuMemImportFromShareableHandle failed: %d", res);
      return flagcxUnhandledDeviceError;
    }
  }

  CUmemGenericAllocationHandle cuPhysHandle =
      *(CUmemGenericAllocationHandle *)physHandle;

  // Bind this rank's physical allocation to the multicast object.
  // Use cuMulticastBindMem (takes physical handle), not cuMulticastBindAddr
  // (which takes a virtual address).
  CUresult res =
      cuMulticastBindMem(cuMcHandle, 0, cuPhysHandle, 0, allocSize, 0);
  if (res != CUDA_SUCCESS) {
    WARN("symMulticastBind: cuMulticastBindMem failed: %d (localRank=%d "
         "allocSize=%zu)",
         res, localRank, allocSize);
    if (imported)
      cuMemRelease(cuMcHandle);
    return flagcxUnhandledDeviceError;
  }

  // Get multicast granularity to compute aligned total size
  CUmulticastObjectProp mcProp = {};
  mcProp.numDevices = (unsigned int)nLocalDevices;
  mcProp.size = allocSize;
  mcProp.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  size_t mcGran = 0;
  res = cuMulticastGetGranularity(&mcGran, &mcProp,
                                  CU_MULTICAST_GRANULARITY_RECOMMENDED);
  if (res != CUDA_SUCCESS) {
    WARN("symMulticastBind: cuMulticastGetGranularity failed: %d", res);
    if (imported)
      cuMemRelease(cuMcHandle);
    return flagcxUnhandledDeviceError;
  }
  size_t alignedSize = ((allocSize + mcGran - 1) / mcGran) * mcGran;

  // Reserve VA and map the multicast handle
  CUdeviceptr mcVa = 0;
  res = cuMemAddressReserve(&mcVa, alignedSize, mcGran, 0, 0);
  if (res != CUDA_SUCCESS) {
    WARN("symMulticastBind: cuMemAddressReserve failed: %d", res);
    if (imported)
      cuMemRelease(cuMcHandle);
    return flagcxUnhandledDeviceError;
  }

  res = cuMemMap(mcVa, alignedSize, 0, cuMcHandle, 0);
  if (res != CUDA_SUCCESS) {
    WARN("symMulticastBind: cuMemMap failed: %d", res);
    cuMemAddressFree(mcVa, alignedSize);
    if (imported)
      cuMemRelease(cuMcHandle);
    return flagcxUnhandledDeviceError;
  }

  // Set access for the current device
  int cudaDev;
  DEVCHECK(cudaGetDevice(&cudaDev));
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  res = cuMemSetAccess(mcVa, alignedSize, &accessDesc, 1);
  if (res != CUDA_SUCCESS) {
    WARN("symMulticastBind: cuMemSetAccess failed: %d", res);
    cuMemUnmap(mcVa, alignedSize);
    cuMemAddressFree(mcVa, alignedSize);
    if (imported)
      cuMemRelease(cuMcHandle);
    return flagcxUnhandledDeviceError;
  }

  *mcBase = (void *)mcVa;
  *mcMapSize = alignedSize;
  if (imported)
    cuMemRelease(cuMcHandle);
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorSymLsaMulticastGranularity(size_t objectSize,
                                                     int nLocalDevices,
                                                     size_t *granularity) {
  if (objectSize == 0 || nLocalDevices <= 0 || granularity == NULL)
    return flagcxInvalidArgument;
  CUmulticastObjectProp prop = {};
  prop.numDevices = (unsigned int)nLocalDevices;
  prop.size = objectSize;
  prop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  CUresult result = cuMulticastGetGranularity(
      granularity, &prop, CU_MULTICAST_GRANULARITY_RECOMMENDED);
  return result == CUDA_SUCCESS && *granularity != 0
             ? flagcxSuccess
             : flagcxUnhandledDeviceError;
}

flagcxResult_t cudaAdaptorSymLsaMulticastMap(void *mcHandle, size_t size,
                                             size_t alignment, void **base,
                                             bool *mappingOwned,
                                             bool *vaOwned) {
  if (mcHandle == NULL || size == 0 || base == NULL || mappingOwned == NULL ||
      vaOwned == NULL)
    return flagcxInvalidArgument;
  *base = NULL;
  *mappingOwned = false;
  *vaOwned = false;
  CUdeviceptr address = 0;
  CUresult result = cuMemAddressReserve(&address, size, alignment, 0, 0);
  if (result != CUDA_SUCCESS)
    return flagcxUnhandledDeviceError;
  *base = reinterpret_cast<void *>(address);
  *vaOwned = true;
  result = cuMemMap(address, size, 0,
                    *static_cast<CUmemGenericAllocationHandle *>(mcHandle), 0);
  if (result != CUDA_SUCCESS)
    return flagcxUnhandledDeviceError;
  *mappingOwned = true;
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess)
    return flagcxUnhandledDeviceError;
  CUmemAccessDesc access = {};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = device;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  return cuMemSetAccess(address, size, &access, 1) == CUDA_SUCCESS
             ? flagcxSuccess
             : flagcxUnhandledDeviceError;
}

flagcxResult_t cudaAdaptorSymLsaMulticastBind(void *mcHandle, size_t offset,
                                              const void *source, size_t size) {
  if (mcHandle == NULL || source == NULL || size == 0)
    return flagcxInvalidArgument;
  return cuMulticastBindAddr(
             *static_cast<CUmemGenericAllocationHandle *>(mcHandle), offset,
             reinterpret_cast<CUdeviceptr>(source), size, 0) == CUDA_SUCCESS
             ? flagcxSuccess
             : flagcxUnhandledDeviceError;
}

flagcxResult_t cudaAdaptorSymLsaMulticastUnbind(void *mcHandle, size_t offset,
                                                size_t size) {
  if (mcHandle == NULL || size == 0)
    return flagcxInvalidArgument;
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess)
    return flagcxUnhandledDeviceError;
  return cuMulticastUnbind(
             *static_cast<CUmemGenericAllocationHandle *>(mcHandle), device,
             offset, size) == CUDA_SUCCESS
             ? flagcxSuccess
             : flagcxUnhandledDeviceError;
}

flagcxResult_t cudaAdaptorSymMulticastTeardown(void *mcBase, size_t mcMapSize) {
  if (mcBase == NULL)
    return flagcxSuccess;
  CUdeviceptr va = (CUdeviceptr)mcBase;
  DEVCHECK(cuMemUnmap(va, mcMapSize));
  DEVCHECK(cuMemAddressFree(va, mcMapSize));
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorSymMulticastMappingUnmap(void *mcBase,
                                                   size_t mcMapSize) {
  if (mcBase == NULL)
    return flagcxSuccess;
  DEVCHECK(cuMemUnmap((CUdeviceptr)mcBase, mcMapSize));
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorSymMulticastVaFree(void *mcBase, size_t mcMapSize) {
  if (mcBase == NULL)
    return flagcxSuccess;
  DEVCHECK(cuMemAddressFree((CUdeviceptr)mcBase, mcMapSize));
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorSymMulticastFree(void *mcHandle) {
  if (mcHandle == NULL)
    return flagcxSuccess;
  CUmemGenericAllocationHandle handle =
      *(CUmemGenericAllocationHandle *)mcHandle;
  DEVCHECK(cuMemRelease(handle));
  free(mcHandle);
  return flagcxSuccess;
}

#else // CUDART_VERSION < 12010

flagcxResult_t cudaAdaptorSymPhysAlloc(void *, size_t, void **, void *,
                                       size_t *, size_t *) {
  return flagcxNotSupported;
}
flagcxResult_t cudaAdaptorSymPhysFree(void *) { return flagcxNotSupported; }
flagcxResult_t cudaAdaptorSymFlatMap(void *[], int, int, void *, size_t,
                                     void **) {
  return flagcxNotSupported;
}
flagcxResult_t cudaAdaptorSymFlatUnmap(void *, size_t, int) {
  return flagcxNotSupported;
}
flagcxResult_t cudaAdaptorSymFlatMappingUnmap(void *, size_t, int) {
  return flagcxNotSupported;
}
flagcxResult_t cudaAdaptorSymFlatVaFree(void *, size_t, int) {
  return flagcxNotSupported;
}
flagcxResult_t cudaAdaptorSymMulticastSupported(int *supported) {
  if (supported)
    *supported = 0;
  return flagcxSuccess;
}
flagcxResult_t cudaAdaptorSymMulticastCreate(size_t, int, const int *, void **,
                                             int *) {
  return flagcxNotSupported;
}
flagcxResult_t cudaAdaptorSymMulticastBind(void *, int, void *, size_t, int,
                                           int, void **, size_t *) {
  return flagcxNotSupported;
}
flagcxResult_t cudaAdaptorSymMulticastTeardown(void *, size_t) {
  return flagcxSuccess;
}
flagcxResult_t cudaAdaptorSymMulticastMappingUnmap(void *, size_t) {
  return flagcxNotSupported;
}
flagcxResult_t cudaAdaptorSymMulticastVaFree(void *, size_t) {
  return flagcxNotSupported;
}
flagcxResult_t cudaAdaptorSymMulticastFree(void *) { return flagcxSuccess; }
flagcxResult_t cudaAdaptorSymMulticastImport(int, void **) {
  return flagcxNotSupported;
}

#endif // CUDART_VERSION >= 12010

flagcxResult_t cudaAdaptorGetLastError() {
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? flagcxSuccess : flagcxSystemError;
}

flagcxResult_t cudaAdaptorGetPointerType(const void *ptr, int *ptrType) {
  if (ptr == NULL || ptrType == NULL)
    return flagcxInvalidArgument;

  cudaPointerAttributes attrs = {};
  cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
  if (err == cudaErrorInvalidValue) {
    // Ordinary host allocations are not tracked by the CUDA runtime. Clear
    // the probe error so it cannot affect a later runtime call.
    cudaGetLastError();
    *ptrType = FLAGCX_PTR_HOST;
    return flagcxSuccess;
  }
  if (err != cudaSuccess) {
    // Do not misclassify runtime initialization, device-loss, or asynchronous
    // execution errors as host memory.
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

flagcxResult_t cudaAdaptorGetAddressRange(const void *ptr, void **base,
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

struct flagcxDeviceAdaptor cudaAdaptor {
  "CUDA",
      // Basic functions
      cudaAdaptorDeviceSynchronize, cudaAdaptorDeviceMemcpy,
      cudaAdaptorDeviceMemset, cudaAdaptorDeviceMalloc, cudaAdaptorDeviceFree,
      cudaAdaptorSetDevice, cudaAdaptorGetDevice, cudaAdaptorGetDeviceCount,
      cudaAdaptorGetVendor, cudaAdaptorHostGetDevicePointer,
      // GDR functions
      NULL, // flagcxResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // flagcxResult_t (*memHandleDestroy)(int dev, void *memHandle);
      cudaAdaptorGdrMemAlloc, cudaAdaptorGdrMemFree,
      NULL, // flagcxResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // flagcxResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      NULL, // flagcxResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t
            // sz);
      NULL, // flagcxResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);
      // Stream functions
      cudaAdaptorStreamCreate, cudaAdaptorStreamDestroy, cudaAdaptorStreamCopy,
      cudaAdaptorStreamFree, cudaAdaptorStreamSynchronize,
      cudaAdaptorStreamQuery, cudaAdaptorStreamWaitEvent,
      cudaAdaptorStreamWaitValue64, cudaAdaptorStreamWriteValue64,
      // Event functions
      cudaAdaptorEventCreate, cudaAdaptorEventDestroy, cudaAdaptorEventRecord,
      cudaAdaptorEventSynchronize, cudaAdaptorEventQuery,
      cudaAdaptorEventElapsedTime,
      // IpcMemHandle functions
      cudaAdaptorIpcMemHandleCreate, cudaAdaptorIpcMemHandleGet,
      cudaAdaptorIpcMemHandleOpen, cudaAdaptorIpcMemHandleClose,
      cudaAdaptorIpcMemHandleFree,
      // Kernel launch
      NULL, // flagcxResult_t (*launchKernel)(void *func, unsigned int block_x,
            // unsigned int block_y, unsigned int block_z, unsigned int grid_x,
            // unsigned int grid_y, unsigned int grid_z, void **args, size_t
            // share_mem, void *stream, void *memHandle);
      NULL, // flagcxResult_t (*copyArgsInit)(void **args);
      NULL, // flagcxResult_t (*copyArgsFree)(void *args);
      cudaAdaptorLaunchDeviceFunc, // flagcxResult_t
                                   // (*launchDeviceFunc)(flagcxStream_t stream,
                                   // void *args);
      // Others
      cudaAdaptorGetDeviceProperties, // flagcxResult_t
                                      // (*getDeviceProperties)(struct
                                      // flagcxDevProps *props, int dev);
      cudaAdaptorGetDevicePciBusId, // flagcxResult_t (*getDevicePciBusId)(char
                                    // *pciBusId, int len, int dev);
      cudaAdaptorGetDeviceByPciBusId, // flagcxResult_t
                                      // (*getDeviceByPciBusId)(int
                                      // *dev, const char *pciBusId);
      cudaAdaptorLaunchHostFunc,
      // DMA buffer
      cudaAdaptorDmaSupport, // flagcxResult_t (*dmaSupport)(bool
                             // *dmaBufferSupport);
      cudaAdaptorMemGetHandleForAddressRange, // flagcxResult_t
                                              // (*memGetHandleForAddressRange)(void
                                              // *handleOut, void *buffer,
                                              // size_t size, unsigned long long
                                              // flags);
      cudaAdaptorHostRegister,   // flagcxResult_t (*hostRegister)(void *,
                                 // size_t);
      cudaAdaptorHostUnregister, // flagcxResult_t (*hostUnregister)(void *);
                                 // Symmetric memory VMM functions
      cudaAdaptorSymPhysAlloc, cudaAdaptorSymPhysFree, cudaAdaptorSymFlatMap,
      cudaAdaptorSymFlatUnmap, cudaAdaptorSymMulticastSupported,
      cudaAdaptorSymMulticastCreate, cudaAdaptorSymMulticastBind,
      cudaAdaptorSymMulticastTeardown, cudaAdaptorSymMulticastFree,
      cudaAdaptorGetLastError, cudaAdaptorGetPointerType,
      cudaAdaptorGetAddressRange, cudaVmmMrCapsForVersion(CUDART_VERSION),
      FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE, cudaAdaptorSymMulticastImport,
      cudaAdaptorSymFlatMappingUnmap, cudaAdaptorSymFlatVaFree,
      cudaAdaptorSymMulticastMappingUnmap, cudaAdaptorSymMulticastVaFree,
      // Conservative default, matching NCCL's explicit GET acquire and its
      // pre-Hopper receive policy. A future per-topology resolver may clear a
      // bit only after the device/NIC path is known to be coherent.
      FLAGCX_GDR_READ_REQUIRES_FLUSH | FLAGCX_GDR_WRITE_REQUIRES_FLUSH,
      FLAGCX_GDR_DEVICE_CUDA, cudaAdaptorGetDeviceArchitecture,
      cudaAdaptorCanAccessPeer,
#if CUDART_VERSION >= 12010
      cudaAdaptorSymPhysDescribe, cudaAdaptorSymPhysExportSegment,
      cudaAdaptorSymPhysFreeSegments, cudaAdaptorSymFlatMapSegments,
      cudaAdaptorSymFlatMappingUnmapSegments, cudaAdaptorSymFlatVaFreeSegments,
      cudaAdaptorSymPhysSegmentCount, cudaAdaptorSymFlatMapSegmentsBegin,
      cudaAdaptorSymFlatMapSegmentOne, cudaAdaptorSymLsaGetCapacity,
      cudaAdaptorSymLsaVaReserve, cudaAdaptorSymLsaVaFree,
      cudaAdaptorSymLsaMapBeginAt, cudaAdaptorSymLsaMulticastGranularity,
      cudaAdaptorSymLsaMulticastMap, cudaAdaptorSymLsaMulticastBind,
      cudaAdaptorSymLsaMulticastUnbind,
#else
      NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
      NULL, NULL, NULL, NULL, NULL,
#endif
};

#endif // USE_NVIDIA_ADAPTOR
