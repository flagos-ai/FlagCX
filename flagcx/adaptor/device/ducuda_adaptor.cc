#include "du_adaptor.h"

#ifdef USE_DU_ADAPTOR

#include "adaptor.h"
#include "alloc.h"

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

flagcxResult_t ducudaAdaptorHostGetDevicePointer(void **pDevice, void *pHost) {
  DEVCHECK(cudaHostGetDevicePointer(pDevice, pHost, 0));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorGdrMemAlloc(void **ptr, size_t size,
                                        void *memHandle) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaMalloc(ptr, size));
  cudaPointerAttributes attrs;
  DEVCHECK(cudaPointerGetAttributes(&attrs, *ptr));
  unsigned flags = 1;
  DEVCHECK(cuPointerSetAttribute(&flags, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                 (CUdeviceptr)attrs.devicePointer));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return flagcxSuccess;
  }
  DEVCHECK(cudaFree(ptr));
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

  *dmaBufferSupport = false;
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorMemGetHandleForAddressRange(
    void *handleOut, void *buffer, size_t size, unsigned long long flags) {
  // unsupportted on dcu
  return flagcxNotSupported;
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
  (void)flags;
  if (stream == NULL || addr == NULL)
    return flagcxInvalidArgument;
  CUstream cuStream = (CUstream)(stream->base);
  CUresult err = cuStreamWaitValue64(cuStream, (CUdeviceptr)addr, value,
                                     CU_STREAM_WAIT_VALUE_GEQ);
  return (err == CUDA_SUCCESS) ? flagcxSuccess : flagcxUnhandledDeviceError;
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
  DEVCHECK(cudaHostRegister(ptr, size, cudaHostRegisterMapped));
  return flagcxSuccess;
}

flagcxResult_t ducudaAdaptorHostUnregister(void *ptr) {
  DEVCHECK(cudaHostUnregister(ptr));
  return flagcxSuccess;
}

// Symmetric memory VMM stubs (not supported)
flagcxResult_t ducudaAdaptorSymPhysAlloc(void *ptr, size_t size,
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
flagcxResult_t ducudaAdaptorSymPhysFree(void *physHandle) {
  if (physHandle == NULL)
    return flagcxSuccess;
  CUmemGenericAllocationHandle *cuHandle =
      (CUmemGenericAllocationHandle *)physHandle;
  cuMemRelease(*cuHandle);
  free(cuHandle);
  return flagcxSuccess;
}
flagcxResult_t ducudaAdaptorSymFlatMap(void *peerHandles[], int nPeers,
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
flagcxResult_t ducudaAdaptorSymFlatUnmap(void *flatBase, size_t allocSize,
                                         int nPeers) {
  if (flatBase == NULL)
    return flagcxSuccess;
  CUdeviceptr base = (CUdeviceptr)flatBase;
  size_t totalSize = allocSize * nPeers;
  DEVCHECK(cuMemUnmap(base, totalSize));
  DEVCHECK(cuMemAddressFree(base, totalSize));
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
flagcxResult_t ducudaAdaptorSymMulticastFree(void *mcHandle) {
  // not supported on dcu
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
      // Symmetric memory VMM functions (not supported)
      ducudaAdaptorSymPhysAlloc, ducudaAdaptorSymPhysFree,
      ducudaAdaptorSymFlatMap, ducudaAdaptorSymFlatUnmap,
      ducudaAdaptorSymMulticastSupported, ducudaAdaptorSymMulticastCreate,
      ducudaAdaptorSymMulticastBind, ducudaAdaptorSymMulticastTeardown,
      ducudaAdaptorSymMulticastFree,
};

#endif // USE_DU_ADAPTOR
