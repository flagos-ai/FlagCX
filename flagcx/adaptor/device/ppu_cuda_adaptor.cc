#include "ppu_adaptor.h"

#ifdef USE_PPU_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "param.h"
#include <unistd.h>

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
  cuMemRelease(handle);
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return flagcxSuccess;
  }
  // PPU SDK version numbering differs from NVIDIA. Use runtime toggle instead
  // of compile-time CUDART_VERSION guards.
  if (!flagcxParamVmmEnable()) {
    DEVCHECK(cudaFree(ptr));
    return flagcxSuccess;
  }

  size_t size = 0;
  DEVCHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
  DEVCHECK(cuMemUnmap((CUdeviceptr)ptr, size));
  DEVCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
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
  (void)flags;
  if (stream == NULL || addr == NULL)
    return flagcxInvalidArgument;
  CUstream cuStream = (CUstream)(stream->base);
  CUresult err = cuStreamWaitValue64(cuStream, (CUdeviceptr)addr, value,
                                     CU_STREAM_WAIT_VALUE_GEQ);
  return (err == CUDA_SUCCESS) ? flagcxSuccess : flagcxUnhandledDeviceError;
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
  CUdeviceptr dptr = (CUdeviceptr)buffer;
  DEVCHECK(cuMemGetHandleForAddressRange(
      handleOut, dptr, size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, flags));
  return flagcxSuccess;
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
  DEVCHECK(cudaHostRegister(ptr, size, cudaHostRegisterMapped));
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorHostUnregister(void *ptr) {
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

  DEVCHECK(cuMemRetainAllocationHandle(cuHandle, ptr));

  size_t actualAllocSize = 0;
  DEVCHECK(cuMemGetAddressRange(NULL, &actualAllocSize, (CUdeviceptr)ptr));
  *allocSize = actualAllocSize;

  if (*handleSize < sizeof(int)) {
    free(cuHandle);
    return flagcxInvalidArgument;
  }
  DEVCHECK(cuMemExportToShareableHandle(
      shareableHandle, *cuHandle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
  *handleSize = sizeof(int);
  *physHandle = cuHandle;
  return flagcxSuccess;
}

flagcxResult_t ppucudaAdaptorSymPhysFree(void *physHandle) {
  if (physHandle == NULL)
    return flagcxSuccess;
  CUmemGenericAllocationHandle *cuHandle =
      (CUmemGenericAllocationHandle *)physHandle;
  cuMemRelease(*cuHandle);
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
  DEVCHECK(cuMemAddressReserve(&base, totalSize, 0, 0, 0));

  int cudaDev;
  DEVCHECK(cudaGetDevice(&cudaDev));
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
      DEVCHECK(cuMemImportFromShareableHandle(
          &peerHandle, (void *)(uintptr_t)fd,
          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    }
    CUdeviceptr slot = base + (CUdeviceptr)i * allocSize;
    DEVCHECK(cuMemMap(slot, allocSize, 0, peerHandle, 0));
    DEVCHECK(cuMemSetAccess(slot, allocSize, &accessDesc, 1));
    if (i != selfIndex) {
      cuMemRelease(peerHandle);
    }
  }

  *flatBase = (void *)base;
  return flagcxSuccess;
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

flagcxResult_t ppucudaAdaptorSymMulticastFree(void *mcHandle) {
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
};

#endif // USE_PPU_ADAPTOR
