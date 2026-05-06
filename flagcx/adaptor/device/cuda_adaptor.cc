#include "nvidia_adaptor.h"

#ifdef USE_NVIDIA_ADAPTOR

#include "adaptor.h"
#include "alloc.h"

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

flagcxResult_t cudaAdaptorHostGetDevicePointer(void **pDevice, void *pHost) {
  DEVCHECK(cudaHostGetDevicePointer(pDevice, pHost, 0));
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorGdrMemAlloc(void **ptr, size_t size,
                                      void *memHandle) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
#if CUDART_VERSION >= 12010
  size_t memGran = 0;
  CUdevice currentDev;
  CUmemAllocationProp memprop = {};
  CUmemGenericAllocationHandle handle = (CUmemGenericAllocationHandle)-1;
  int cudaDev;
  int flag;

  DEVCHECK(cudaGetDevice(&cudaDev));
  DEVCHECK(cuDeviceGet(&currentDev, cudaDev));

  size_t handleSize = size;
  int requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  // Query device to see if FABRIC handle support is available
#if CUDART_VERSION >= 12040
  flag = 0;
  DEVCHECK(cuDeviceGetAttribute(
      &flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, currentDev));
  if (flag)
    requestedHandleTypes |= CU_MEM_HANDLE_TYPE_FABRIC;
#endif
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
  DEVCHECK(cuMemAddressReserve((CUdeviceptr *)ptr, handleSize, memGran, 0, 0));
  /* Map the virtual address range to the physical allocation */
  DEVCHECK(cuMemMap((CUdeviceptr)*ptr, handleSize, 0, handle, 0));
  /* Set access for the current device */
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = currentDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  DEVCHECK(cuMemSetAccess((CUdeviceptr)*ptr, handleSize, &accessDesc, 1));
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
  CUmemGenericAllocationHandle handle;
  size_t size = 0;
  DEVCHECK(cuMemRetainAllocationHandle(&handle, ptr));
  DEVCHECK(cuMemRelease(handle));
  DEVCHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
  DEVCHECK(cuMemUnmap((CUdeviceptr)ptr, size));
  DEVCHECK(cuMemRelease(handle));
  DEVCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
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
  (void)flags;
  if (stream == NULL || addr == NULL)
    return flagcxInvalidArgument;
  CUstream cuStream = (CUstream)(stream->base);
  CUresult err = cuStreamWaitValue64(cuStream, (CUdeviceptr)addr, value,
                                     CU_STREAM_WAIT_VALUE_GEQ);
  return (err == CUDA_SUCCESS) ? flagcxSuccess : flagcxUnhandledDeviceError;
}

flagcxResult_t cudaAdaptorStreamWriteValue64(flagcxStream_t stream, void *addr,
                                             uint64_t value, int flags) {
  (void)flags;
  if (stream == NULL || addr == NULL)
    return flagcxInvalidArgument;
  CUstream cuStream = (CUstream)(stream->base);
  CUresult err = cuStreamWriteValue64(cuStream, (CUdeviceptr)addr, value,
                                      CU_STREAM_WRITE_VALUE_DEFAULT);
  return (err == CUDA_SUCCESS) ? flagcxSuccess : flagcxUnhandledDeviceError;
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
  CUdeviceptr dptr = (CUdeviceptr)buffer;
  DEVCHECK(cuMemGetHandleForAddressRange(
      handleOut, dptr, size, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, flags));
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorHostRegister(void *ptr, size_t size) {
  DEVCHECK(cudaHostRegister(ptr, size, cudaHostRegisterMapped));
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorHostUnregister(void *ptr) {
  DEVCHECK(cudaHostUnregister(ptr));
  return flagcxSuccess;
}

// ==========================================================================
// Symmetric memory VMM functions
// ==========================================================================

#if CUDART_VERSION >= 12010

flagcxResult_t cudaAdaptorSymPhysAlloc(void *ptr, size_t size,
                                       void **physHandle, void *shareableHandle,
                                       size_t *handleSize) {
  if (ptr == NULL || physHandle == NULL || shareableHandle == NULL ||
      handleSize == NULL)
    return flagcxInvalidArgument;

  CUmemGenericAllocationHandle *cuHandle =
      (CUmemGenericAllocationHandle *)malloc(
          sizeof(CUmemGenericAllocationHandle));
  if (cuHandle == NULL)
    return flagcxSystemError;

  // Retain the physical allocation handle from the VMM-backed pointer
  DEVCHECK(cuMemRetainAllocationHandle(cuHandle, ptr));

  // Export as POSIX fd for IPC sharing
  if (*handleSize < sizeof(int)) {
    free(cuHandle);
    return flagcxInvalidArgument;
  }
  DEVCHECK(cuMemExportToShareableHandle(
      shareableHandle, *cuHandle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
  *handleSize = sizeof(int); // POSIX fd is an int
  *physHandle = cuHandle;
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorSymPhysFree(void *physHandle) {
  if (physHandle == NULL)
    return flagcxSuccess;
  CUmemGenericAllocationHandle *cuHandle =
      (CUmemGenericAllocationHandle *)physHandle;
  cuMemRelease(*cuHandle);
  free(cuHandle);
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorSymFlatMap(void *peerHandles[], int nPeers,
                                     int selfIndex, void *selfPhysHandle,
                                     size_t heapSize, size_t maxHeapSize,
                                     void **flatBase) {
  if (peerHandles == NULL || selfPhysHandle == NULL || flatBase == NULL ||
      nPeers <= 0)
    return flagcxInvalidArgument;

  CUmemGenericAllocationHandle selfHandle =
      *(CUmemGenericAllocationHandle *)selfPhysHandle;

  // Get granularity for alignment
  CUmemAllocationProp memprop = {};
  memprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  memprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  int cudaDev;
  DEVCHECK(cudaGetDevice(&cudaDev));
  memprop.location.id = cudaDev;
  size_t memGran = 0;
  DEVCHECK(cuMemGetAllocationGranularity(&memGran, &memprop,
                                         CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

  size_t totalSize = maxHeapSize * nPeers;
  // Reserve the full VA range
  CUdeviceptr base = 0;
  DEVCHECK(cuMemAddressReserve(&base, totalSize, memGran, 0, 0));

  // Import and map each peer's physical memory
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  for (int i = 0; i < nPeers; i++) {
    CUmemGenericAllocationHandle peerHandle;
    if (i == selfIndex) {
      peerHandle = selfHandle;
    } else {
      // Import from POSIX fd
      int fd = *(int *)peerHandles[i];
      DEVCHECK(cuMemImportFromShareableHandle(
          &peerHandle, (void *)(uintptr_t)fd,
          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    }
    CUdeviceptr slot = base + (CUdeviceptr)i * maxHeapSize;
    DEVCHECK(cuMemMap(slot, heapSize, 0, peerHandle, 0));
    DEVCHECK(cuMemSetAccess(slot, heapSize, &accessDesc, 1));
    // Release the imported handle (mapping holds a ref)
    if (i != selfIndex) {
      cuMemRelease(peerHandle);
    }
  }

  *flatBase = (void *)base;
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorSymFlatUnmap(void *flatBase, size_t maxHeapSize,
                                       int nPeers) {
  if (flatBase == NULL)
    return flagcxSuccess;
  CUdeviceptr base = (CUdeviceptr)flatBase;
  size_t totalSize = maxHeapSize * nPeers;
  DEVCHECK(cuMemUnmap(base, totalSize));
  DEVCHECK(cuMemAddressFree(base, totalSize));
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorSymMulticastSetup(void *physHandle, size_t heapSize,
                                            int nLocalDevices, void **mcBase) {
  if (physHandle == NULL || mcBase == NULL || nLocalDevices <= 0)
    return flagcxInvalidArgument;
  *mcBase = NULL;

  // Check multicast support at runtime
  int cudaDev;
  DEVCHECK(cudaGetDevice(&cudaDev));
  CUdevice dev;
  DEVCHECK(cuDeviceGet(&dev, cudaDev));
  int multicastSupported = 0;
  CUresult res = cuDeviceGetAttribute(
      &multicastSupported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev);
  if (res != CUDA_SUCCESS || !multicastSupported)
    return flagcxNotSupported;

  CUmemGenericAllocationHandle cuHandle =
      *(CUmemGenericAllocationHandle *)physHandle;

  // Create multicast object
  CUmulticastObjectProp mcProp = {};
  mcProp.numDevices = (unsigned int)nLocalDevices;
  mcProp.size = heapSize;
  mcProp.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  CUmemGenericAllocationHandle mcHandle;
  DEVCHECK(cuMulticastCreate(&mcHandle, &mcProp));

  // Add all local devices
  for (int i = 0; i < nLocalDevices; i++) {
    CUdevice peerDev;
    DEVCHECK(cuDeviceGet(&peerDev, i));
    DEVCHECK(cuMulticastAddDevice(mcHandle, peerDev));
  }

  // Bind the physical allocation to the multicast object
  DEVCHECK(cuMulticastBindAddr(mcHandle, 0, cuHandle, heapSize, 0));

  // Reserve VA and map the multicast handle
  CUmemAllocationProp memprop = {};
  memprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  memprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  memprop.location.id = cudaDev;
  size_t memGran = 0;
  DEVCHECK(cuMemGetAllocationGranularity(&memGran, &memprop,
                                         CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

  CUdeviceptr mcVa = 0;
  DEVCHECK(cuMemAddressReserve(&mcVa, heapSize, memGran, 0, 0));
  DEVCHECK(cuMemMap(mcVa, heapSize, 0, mcHandle, 0));

  // Set access for the current device
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  DEVCHECK(cuMemSetAccess(mcVa, heapSize, &accessDesc, 1));

  *mcBase = (void *)mcVa;
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorSymMulticastTeardown(void *mcBase,
                                               size_t maxHeapSize) {
  if (mcBase == NULL)
    return flagcxSuccess;
  CUdeviceptr va = (CUdeviceptr)mcBase;
  DEVCHECK(cuMemUnmap(va, maxHeapSize));
  DEVCHECK(cuMemAddressFree(va, maxHeapSize));
  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorSymHeapGrow(void *flatBase, void *peerNewHandles[],
                                      int nPeers, int selfIndex,
                                      void *selfNewPhysHandle, size_t oldSize,
                                      size_t newSize, size_t maxHeapSize) {
  if (flatBase == NULL || peerNewHandles == NULL || selfNewPhysHandle == NULL ||
      newSize <= oldSize)
    return flagcxInvalidArgument;

  CUmemGenericAllocationHandle selfHandle =
      *(CUmemGenericAllocationHandle *)selfNewPhysHandle;
  CUdeviceptr base = (CUdeviceptr)flatBase;

  int cudaDev;
  DEVCHECK(cudaGetDevice(&cudaDev));
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  size_t deltaSize = newSize - oldSize;

  for (int i = 0; i < nPeers; i++) {
    CUmemGenericAllocationHandle peerHandle;
    if (i == selfIndex) {
      peerHandle = selfHandle;
    } else {
      int fd = *(int *)peerNewHandles[i];
      DEVCHECK(cuMemImportFromShareableHandle(
          &peerHandle, (void *)(uintptr_t)fd,
          CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    }
    CUdeviceptr slot = base + (CUdeviceptr)i * maxHeapSize + oldSize;
    DEVCHECK(cuMemMap(slot, deltaSize, 0, peerHandle, 0));
    DEVCHECK(cuMemSetAccess(slot, deltaSize, &accessDesc, 1));
    if (i != selfIndex) {
      cuMemRelease(peerHandle);
    }
  }

  return flagcxSuccess;
}

flagcxResult_t cudaAdaptorSymMulticastGrow(void *mcBase, void *newPhysHandle,
                                           size_t oldSize, size_t newSize) {
  if (mcBase == NULL || newPhysHandle == NULL || newSize <= oldSize)
    return flagcxInvalidArgument;

  CUdeviceptr va = (CUdeviceptr)mcBase;
  CUmemGenericAllocationHandle cuHandle =
      *(CUmemGenericAllocationHandle *)newPhysHandle;
  size_t deltaSize = newSize - oldSize;

  // Retrieve the multicast handle from the existing mapping
  CUmemGenericAllocationHandle mcHandle;
  DEVCHECK(cuMemRetainAllocationHandle(&mcHandle, mcBase));

  // Bind new physical pages into the multicast object at the growth offset
  DEVCHECK(cuMulticastBindAddr(mcHandle, oldSize, cuHandle, deltaSize, 0));

  // Map the new region into the existing VA
  DEVCHECK(cuMemMap(va + oldSize, deltaSize, 0, mcHandle, oldSize));

  // Set access
  int cudaDev;
  DEVCHECK(cudaGetDevice(&cudaDev));
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = cudaDev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  DEVCHECK(cuMemSetAccess(va + oldSize, deltaSize, &accessDesc, 1));

  cuMemRelease(mcHandle);
  return flagcxSuccess;
}

#endif // CUDART_VERSION >= 12010

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
#if CUDART_VERSION >= 12010
      cudaAdaptorSymPhysAlloc, cudaAdaptorSymPhysFree, cudaAdaptorSymFlatMap,
      cudaAdaptorSymFlatUnmap, cudaAdaptorSymMulticastSetup,
      cudaAdaptorSymMulticastTeardown, cudaAdaptorSymHeapGrow,
      cudaAdaptorSymMulticastGrow,
#else
      NULL, // symPhysAlloc
      NULL, // symPhysFree
      NULL, // symFlatMap
      NULL, // symFlatUnmap
      NULL, // symMulticastSetup
      NULL, // symMulticastTeardown
      NULL, // symHeapGrow
      NULL, // symMulticastGrow
#endif
};

#endif // USE_NVIDIA_ADAPTOR
