#include "nvidia_adaptor.h"

#ifdef USE_NVIDIA_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include <unistd.h>

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
  DEVCHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
  DEVCHECK(cuMemUnmap((CUdeviceptr)ptr, size));
  DEVCHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
  DEVCHECK(cuMemRelease(handle));
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
  DEVCHECK(cuMemRetainAllocationHandle(cuHandle, ptr));

  // Discover actual physical allocation size (already granularity-aligned)
  size_t actualAllocSize = 0;
  DEVCHECK(cuMemGetAddressRange(NULL, &actualAllocSize, (CUdeviceptr)ptr));
  *allocSize = actualAllocSize;

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
  DEVCHECK(cuMemAddressReserve(&base, totalSize, 0, 0, 0));

  // Import and map each peer's physical memory
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

flagcxResult_t cudaAdaptorSymMulticastTeardown(void *mcBase, size_t mcMapSize) {
  if (mcBase == NULL)
    return flagcxSuccess;
  CUdeviceptr va = (CUdeviceptr)mcBase;
  DEVCHECK(cuMemUnmap(va, mcMapSize));
  DEVCHECK(cuMemAddressFree(va, mcMapSize));
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
flagcxResult_t cudaAdaptorSymMulticastFree(void *) { return flagcxSuccess; }

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
      cudaAdaptorSymPhysAlloc, cudaAdaptorSymPhysFree, cudaAdaptorSymFlatMap,
      cudaAdaptorSymFlatUnmap, cudaAdaptorSymMulticastSupported,
      cudaAdaptorSymMulticastCreate, cudaAdaptorSymMulticastBind,
      cudaAdaptorSymMulticastTeardown, cudaAdaptorSymMulticastFree,
};

#endif // USE_NVIDIA_ADAPTOR
