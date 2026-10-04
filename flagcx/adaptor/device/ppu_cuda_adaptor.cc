#include "ppu_adaptor.h"

#ifdef USE_PPU_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "param.h"
#include <mutex>
#include <new>
#include <unistd.h>
#include <unordered_map>

struct PpuVmmAllocation {
  size_t size;
  bool mappingOwned;
  bool vaOwned;
};
static std::mutex gPpuVmmAllocationMtx;
static std::unordered_map<void *, PpuVmmAllocation> gPpuVmmAllocations;

std::map<flagcxMemcpyType_t, cudaMemcpyKind> memcpy_type_map = {
    {flagcxMemcpyHostToDevice, cudaMemcpyHostToDevice},
    {flagcxMemcpyDeviceToHost, cudaMemcpyDeviceToHost},
    {flagcxMemcpyDeviceToDevice, cudaMemcpyDeviceToDevice},
};

flagcxResult_t ppucudaAdaptorDeviceSynchronize() {
  DEVCHECK(cudaDeviceSynchronize());
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
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

flagcxResult_t ppucudaAdaptorDeviceMemset(void *ptr, int value, size_t size,
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

flagcxResult_t ppucudaAdaptorDeviceMalloc(void **ptr, size_t size,
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

flagcxResult_t ppucudaAdaptorDeviceFree(void *ptr, flagcxMemType_t type,
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

flagcxResult_t ppucudaAdaptorSetDevice(int dev) {
  DEVCHECK(cudaSetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorGetDevice(int *dev) {
  DEVCHECK(cudaGetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorGetDeviceCount(int *count) {
  DEVCHECK(cudaGetDeviceCount(count));
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "PPU");
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorHostGetDevicePointer(void **pDevice, void *pHost) {
  if (pDevice == NULL || pHost == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaHostGetDevicePointer(pDevice, pHost, 0));
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorGdrMemAlloc(void **ptr, size_t size,
                                         void *memHandle) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
  // PPU SDK version numbering differs from NVIDIA (e.g. CUDART_VERSION=13000
  // does not imply identical API availability). Use runtime toggle instead
  // of compile-time CUDART_VERSION guards.
  if (!flagcxParamVmmEnable()) {
    DEVCHECK(cudaMalloc(ptr, size));
    cudaPointerAttributes attrs;
    DEVCHECK(cudaPointerGetAttributes(&attrs, *ptr));
    unsigned flags = 1;
    DEVCHECK(cuPointerSetAttribute(&flags, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                   (CUdeviceptr)attrs.devicePointer));
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
  DEVCHECK(cuDeviceGet(&currentDev, cudaDev));

  size_t handleSize = size;
  int requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
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
  if (flag)
    memprop.allocFlags.gpuDirectRDMACapable = 1;
  DEVCHECK(cuMemGetAllocationGranularity(&memGran, &memprop,
                                         CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  ALIGN_SIZE(handleSize, memGran);
  /* Allocate the physical memory on the device */
  DEVCHECK(cuMemCreate(&handle, handleSize, &memprop, 0));
  /* Reserve a virtual address range */
  cuRes = cuMemAddressReserve((CUdeviceptr *)ptr, handleSize, memGran, 0, 0);
  if (cuRes != CUDA_SUCCESS) {
    cuMemRelease(handle);
    return flagcxUnhandledDeviceError;
  }
  /* Map the virtual address range to the physical allocation */
  cuRes = cuMemMap((CUdeviceptr)*ptr, handleSize, 0, handle, 0);
  if (cuRes != CUDA_SUCCESS) {
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
    cuMemUnmap((CUdeviceptr)*ptr, handleSize);
    cuMemAddressFree((CUdeviceptr)*ptr, handleSize);
    cuMemRelease(handle);
    *ptr = NULL;
    return flagcxUnhandledDeviceError;
  }
  /* Release the create-time handle reference; the mapping holds its own. */
  if (cuMemRelease(handle) != CUDA_SUCCESS) {
    cuMemUnmap((CUdeviceptr)*ptr, handleSize);
    cuMemAddressFree((CUdeviceptr)*ptr, handleSize);
    *ptr = NULL;
    return flagcxUnhandledDeviceError;
  }
  bool tracked = false;
  try {
    std::lock_guard<std::mutex> lock(gPpuVmmAllocationMtx);
    tracked = gPpuVmmAllocations
                  .emplace(*ptr, PpuVmmAllocation{handleSize, true, true})
                  .second;
  } catch (const std::bad_alloc &) {
  }
  if (!tracked) {
    cuMemUnmap((CUdeviceptr)*ptr, handleSize);
    cuMemAddressFree((CUdeviceptr)*ptr, handleSize);
    *ptr = NULL;
    return flagcxSystemError;
  }
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return flagcxSuccess;
  }
  std::lock_guard<std::mutex> lock(gPpuVmmAllocationMtx);
  auto it = gPpuVmmAllocations.find(ptr);
  if (it == gPpuVmmAllocations.end()) {
    DEVCHECK(cudaFree(ptr));
    return flagcxSuccess;
  }
  PpuVmmAllocation &allocation = it->second;
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
  gPpuVmmAllocations.erase(it);
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorStreamCreate(flagcxStream_t *stream) {
  (*stream) = NULL;
  flagcxCalloc(stream, 1);
  DEVCHECK(cudaStreamCreateWithFlags((cudaStream_t *)(*stream),
                                     cudaStreamNonBlocking));
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorStreamDestroy(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(cudaStreamDestroy(stream->base));
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorStreamCopy(flagcxStream_t *newStream,
                                        void *oldStream) {
  (*newStream) = NULL;
  flagcxCalloc(newStream, 1);
  (*newStream)->base = (cudaStream_t)oldStream;
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorStreamFree(flagcxStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorStreamSynchronize(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(cudaStreamSynchronize(stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorStreamQuery(flagcxStream_t stream) {
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

flagcxResult_t ppucudaAdaptorStreamWaitEvent(flagcxStream_t stream,
                                             flagcxEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(
        cudaStreamWaitEvent(stream->base, event->base, cudaEventWaitDefault));
  }
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorStreamWaitValue64(flagcxStream_t stream,
                                               void *addr, uint64_t value,
                                               int flags) {
  if (stream == NULL || addr == NULL)
    return flagcxInvalidArgument;
  if (flags & ~FLAGCX_STREAM_WAIT_VALUE_FLUSH_REMOTE_WRITES)
    return flagcxInvalidArgument;

  unsigned int waitFlags = CU_STREAM_WAIT_VALUE_GEQ;
  // PPU's automatic GDR requirement is intentionally NONE while BAREX lacks a
  // real visibility operation. Keep the existing direct strong-wait behavior
  // for compatibility; add a runtime capability probe before enabling PPU's
  // default WRITE requirement in a follow-up.
  if (flags & FLAGCX_STREAM_WAIT_VALUE_FLUSH_REMOTE_WRITES)
    waitFlags |= CU_STREAM_WAIT_VALUE_FLUSH;

  CUstream cuStream = (CUstream)(stream->base);
  CUresult err =
      cuStreamWaitValue64(cuStream, (CUdeviceptr)addr, value, waitFlags);
  if (err == CUDA_SUCCESS)
    return flagcxSuccess;
  if (err == CUDA_ERROR_NOT_SUPPORTED)
    return flagcxNotSupported;
  return flagcxUnhandledDeviceError;
}

flagcxResult_t ppucudaAdaptorStreamWriteValue64(flagcxStream_t stream,
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

flagcxResult_t ppucudaAdaptorEventCreate(flagcxEvent_t *event,
                                         flagcxEventType_t eventType) {
  (*event) = NULL;
  flagcxCalloc(event, 1);
  const unsigned int flags = (eventType == flagcxEventDefault)
                                 ? cudaEventDefault
                                 : cudaEventDisableTiming;
  DEVCHECK(cudaEventCreateWithFlags(&((*event)->base), flags));
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorEventDestroy(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(cudaEventDestroy(event->base));
    free(event);
    event = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorEventRecord(flagcxEvent_t event,
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

flagcxResult_t ppucudaAdaptorEventSynchronize(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(cudaEventSynchronize(event->base));
  }
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorEventQuery(flagcxEvent_t event) {
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

flagcxResult_t ppucudaAdaptorEventElapsedTime(float *ms, flagcxEvent_t start,
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

flagcxResult_t ppucudaAdaptorIpcMemHandleCreate(flagcxIpcMemHandle_t *handle,
                                                size_t *size) {
  flagcxCalloc(handle, 1);
  if (size != NULL) {
    *size = sizeof(cudaIpcMemHandle_t);
  }
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorIpcMemHandleGet(flagcxIpcMemHandle_t handle,
                                             void *devPtr) {
  if (handle == NULL || devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaIpcGetMemHandle(&handle->base, devPtr));
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorIpcMemHandleOpen(flagcxIpcMemHandle_t handle,
                                              void **devPtr) {
  if (handle == NULL || devPtr == NULL || *devPtr != NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaIpcOpenMemHandle(devPtr, handle->base,
                                cudaIpcMemLazyEnablePeerAccess));
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorIpcMemHandleClose(void *devPtr) {
  if (devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaIpcCloseMemHandle(devPtr));
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorIpcMemHandleFree(flagcxIpcMemHandle_t handle) {
  if (handle != NULL) {
    free(handle);
  }
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorLaunchHostFunc(flagcxStream_t stream,
                                            void (*fn)(void *), void *args) {
  if (stream != NULL) {
    DEVCHECK(cudaLaunchHostFunc(stream->base, fn, args));
  }
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorDmaSupport(bool *dmaBufferSupport) {
  if (dmaBufferSupport == NULL)
    return flagcxInvalidArgument;

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
}

flagcxResult_t ppucudaAdaptorMemGetHandleForAddressRange(
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

flagcxResult_t ppucudaAdaptorGetDeviceProperties(struct flagcxDevProps *props,
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

  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorGetDevicePciBusId(char *pciBusId, int len,
                                               int dev) {
  if (pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaDeviceGetPCIBusId(pciBusId, len, dev));
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorGetDeviceByPciBusId(int *dev,
                                                 const char *pciBusId) {
  if (dev == NULL || pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaDeviceGetByPCIBusId(dev, pciBusId));
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorHostRegister(void *ptr, size_t size) {
  if (ptr == NULL || size == 0) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaHostRegister(ptr, size, cudaHostRegisterMapped));
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorHostUnregister(void *ptr) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaHostUnregister(ptr));
  return flagcxSuccess;
}

// Symmetric memory VMM — handle export/import and flat mapping
flagcxResult_t ppucudaAdaptorSymPhysAlloc(void *ptr, size_t size,
                                          void **physHandle,
                                          void *shareableHandle,
                                          size_t *handleSize,
                                          size_t *allocSize) {
  if (ptr == NULL || physHandle == NULL || shareableHandle == NULL ||
      handleSize == NULL || allocSize == NULL)
    return flagcxInvalidArgument;

  CUmemGenericAllocationHandle *cuHandle =
      (CUmemGenericAllocationHandle *)malloc(
          sizeof(CUmemGenericAllocationHandle));
  if (cuHandle == NULL)
    return flagcxSystemError;

  CUresult result = cuMemRetainAllocationHandle(cuHandle, ptr);
  if (result != CUDA_SUCCESS) {
    free(cuHandle);
    if (result == CUDA_ERROR_INVALID_VALUE ||
        result == CUDA_ERROR_NOT_SUPPORTED)
      return flagcxNotSupported;
    return flagcxUnhandledDeviceError;
  }

  CUdeviceptr allocationBase = 0;
  size_t actualAllocSize = 0;
  result =
      cuMemGetAddressRange(&allocationBase, &actualAllocSize, (CUdeviceptr)ptr);
  if (result != CUDA_SUCCESS) {
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

  if (*handleSize < sizeof(int)) {
    cuMemRelease(*cuHandle);
    free(cuHandle);
    return flagcxInvalidArgument;
  }
  result = cuMemExportToShareableHandle(
      shareableHandle, *cuHandle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
  if (result != CUDA_SUCCESS) {
    cuMemRelease(*cuHandle);
    free(cuHandle);
    return flagcxUnhandledDeviceError;
  }
  *handleSize = sizeof(int);
  *physHandle = cuHandle;
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorSymPhysFree(void *physHandle) {
  if (physHandle == NULL)
    return flagcxSuccess;
  CUmemGenericAllocationHandle *cuHandle =
      (CUmemGenericAllocationHandle *)physHandle;
  if (cuMemRelease(*cuHandle) != CUDA_SUCCESS)
    return flagcxUnhandledDeviceError;
  free(cuHandle);
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorSymFlatMap(void *peerHandles[], int nPeers,
                                        int selfIndex, void *selfPhysHandle,
                                        size_t allocSize, void **flatBase) {
  if (peerHandles == NULL || selfPhysHandle == NULL || flatBase == NULL ||
      nPeers <= 0 || allocSize == 0)
    return flagcxInvalidArgument;

  CUmemGenericAllocationHandle selfHandle =
      *(CUmemGenericAllocationHandle *)selfPhysHandle;

  size_t totalSize = allocSize * nPeers;

  CUdeviceptr base = 0;
  CUresult result = cuMemAddressReserve(&base, totalSize, 0, 0, 0);
  if (result != CUDA_SUCCESS)
    return flagcxUnhandledDeviceError;

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

flagcxResult_t ppucudaAdaptorSymFlatUnmap(void *flatBase, size_t allocSize,
                                          int nPeers) {
  if (flatBase == NULL)
    return flagcxSuccess;
  CUdeviceptr base = (CUdeviceptr)flatBase;
  size_t totalSize = allocSize * nPeers;
  DEVCHECK(cuMemUnmap(base, totalSize));
  DEVCHECK(cuMemAddressFree(base, totalSize));
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorSymFlatMappingUnmap(void *flatBase,
                                                 size_t allocSize, int nPeers) {
  if (flatBase == NULL)
    return flagcxSuccess;
  DEVCHECK(cuMemUnmap((CUdeviceptr)flatBase, allocSize * nPeers));
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorSymFlatVaFree(void *flatBase, size_t allocSize,
                                           int nPeers) {
  if (flatBase == NULL)
    return flagcxSuccess;
  DEVCHECK(cuMemAddressFree((CUdeviceptr)flatBase, allocSize * nPeers));
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorSymMulticastSupported(int *supported) {
  if (supported == NULL)
    return flagcxInvalidArgument;

  *supported = 0;
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorSymMulticastCreate(size_t allocSize,
                                                int nLocalDevices,
                                                const int *localDeviceOrdinals,
                                                void **mcHandle,
                                                int *shareableFd) {
  if (mcHandle)
    *mcHandle = NULL;
  if (shareableFd)
    *shareableFd = -1;
  return flagcxNotSupported;
}

flagcxResult_t ppucudaAdaptorSymMulticastBind(void *mcHandle, int importFd,
                                              void *physHandle,
                                              size_t allocSize, int localRank,
                                              int nLocalDevices, void **mcBase,
                                              size_t *mcMapSize) {
  if (mcBase)
    *mcBase = NULL;
  if (mcMapSize)
    *mcMapSize = 0;
  return flagcxNotSupported;
}

flagcxResult_t ppucudaAdaptorSymMulticastTeardown(void *mcBase,
                                                  size_t mcMapSize) {
  return flagcxSuccess;
}
flagcxResult_t ppucudaAdaptorSymMulticastMappingUnmap(void *mcBase,
                                                      size_t mcMapSize) {
  if (mcBase == NULL)
    return flagcxSuccess;
  DEVCHECK(cuMemUnmap((CUdeviceptr)mcBase, mcMapSize));
  return flagcxSuccess;
}
flagcxResult_t ppucudaAdaptorSymMulticastVaFree(void *mcBase,
                                                size_t mcMapSize) {
  if (mcBase == NULL)
    return flagcxSuccess;
  DEVCHECK(cuMemAddressFree((CUdeviceptr)mcBase, mcMapSize));
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorSymMulticastFree(void *mcHandle) {
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorSymMulticastImport(int, void **mcHandle) {
  if (mcHandle != NULL)
    *mcHandle = NULL;
  return flagcxNotSupported;
}

flagcxResult_t ppucudaAdaptorGetPointerType(const void *ptr, int *ptrType) {
  if (ptr == NULL || ptrType == NULL)
    return flagcxInvalidArgument;

  cudaPointerAttributes attrs = {};
  cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
  if (err == cudaErrorInvalidValue) {
    cudaGetLastError();
    *ptrType = FLAGCX_PTR_HOST;
    return flagcxSuccess;
  }
  if (err != cudaSuccess) {
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

flagcxResult_t ppucudaAdaptorGetAddressRange(const void *ptr, void **base,
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

struct flagcxDeviceAdaptor ppucudaAdaptor {
  "PPU_CUDA",
      // Basic functions
      ppucudaAdaptorDeviceSynchronize, ppucudaAdaptorDeviceMemcpy,
      ppucudaAdaptorDeviceMemset, ppucudaAdaptorDeviceMalloc,
      ppucudaAdaptorDeviceFree, ppucudaAdaptorSetDevice,
      ppucudaAdaptorGetDevice, ppucudaAdaptorGetDeviceCount,
      ppucudaAdaptorGetVendor, ppucudaAdaptorHostGetDevicePointer,
      // GDR functions
      NULL, // memHandleInit
      NULL, // memHandleDestroy
      ppucudaAdaptorGdrMemAlloc, ppucudaAdaptorGdrMemFree,
      NULL, // hostShareMemAlloc
      NULL, // hostShareMemFree
      NULL, // gdrPtrMmap
      NULL, // gdrPtrMunmap
      // Stream functions
      ppucudaAdaptorStreamCreate, ppucudaAdaptorStreamDestroy,
      ppucudaAdaptorStreamCopy, ppucudaAdaptorStreamFree,
      ppucudaAdaptorStreamSynchronize, ppucudaAdaptorStreamQuery,
      ppucudaAdaptorStreamWaitEvent, ppucudaAdaptorStreamWaitValue64,
      ppucudaAdaptorStreamWriteValue64,
      // Event functions
      ppucudaAdaptorEventCreate, ppucudaAdaptorEventDestroy,
      ppucudaAdaptorEventRecord, ppucudaAdaptorEventSynchronize,
      ppucudaAdaptorEventQuery, ppucudaAdaptorEventElapsedTime,
      // IpcMemHandle functions
      ppucudaAdaptorIpcMemHandleCreate, ppucudaAdaptorIpcMemHandleGet,
      ppucudaAdaptorIpcMemHandleOpen, ppucudaAdaptorIpcMemHandleClose,
      ppucudaAdaptorIpcMemHandleFree,
      // Kernel launch
      NULL, // launchKernel
      NULL, // copyArgsInit
      NULL, // copyArgsFree
      NULL, // launchDeviceFunc
      // Others
      ppucudaAdaptorGetDeviceProperties, ppucudaAdaptorGetDevicePciBusId,
      ppucudaAdaptorGetDeviceByPciBusId, ppucudaAdaptorLaunchHostFunc,
      // DMA buffer
      ppucudaAdaptorDmaSupport, ppucudaAdaptorMemGetHandleForAddressRange,
      ppucudaAdaptorHostRegister, ppucudaAdaptorHostUnregister,
      // Symmetric memory VMM functions
      ppucudaAdaptorSymPhysAlloc, ppucudaAdaptorSymPhysFree,
      ppucudaAdaptorSymFlatMap, ppucudaAdaptorSymFlatUnmap,
      ppucudaAdaptorSymMulticastSupported, ppucudaAdaptorSymMulticastCreate,
      ppucudaAdaptorSymMulticastBind, ppucudaAdaptorSymMulticastTeardown,
      ppucudaAdaptorSymMulticastFree,
      NULL, // flagcxResult_t (*getLastError)();
      ppucudaAdaptorGetPointerType, ppucudaAdaptorGetAddressRange,
      FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA,
      FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE, ppucudaAdaptorSymMulticastImport,
      ppucudaAdaptorSymFlatMappingUnmap, ppucudaAdaptorSymFlatVaFree,
      ppucudaAdaptorSymMulticastMappingUnmap, ppucudaAdaptorSymMulticastVaFree,
      // Transitional BAREX compatibility policy: NONE preserves the existing
      // PPU CI behavior while ACCL has no GPU-visibility flush API. This is not
      // a documented coherence guarantee. Do not enable READ/WRITE requirements
      // until BAREX advertises a real capability backed by a hardware test.
      FLAGCX_GDR_FLUSH_NONE, FLAGCX_GDR_DEVICE_PPU, NULL,
};

#endif // USE_PPU_ADAPTOR
