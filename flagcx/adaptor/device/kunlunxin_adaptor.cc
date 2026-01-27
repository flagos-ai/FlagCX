#ifdef USE_KUNLUNXIN_ADAPTOR

#include "kunlunxin_adaptor.h"

std::map<flagcxMemcpyType_t, cudaMemcpyKind> memcpy_type_map = {
    {flagcxMemcpyHostToDevice, cudaMemcpyHostToDevice},
    {flagcxMemcpyDeviceToHost, cudaMemcpyDeviceToHost},
    {flagcxMemcpyDeviceToDevice, cudaMemcpyDeviceToDevice},
};

flagcxResult_t kunlunAdaptorDeviceSynchronize() {
  DEVCHECK(cudaDeviceSynchronize());
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
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

flagcxResult_t kunlunAdaptorDeviceMemset(void *ptr, int value, size_t size,
                                         flagcxMemType_t type,
                                         flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    memset(ptr, value, size);
  } else {
    if (stream == NULL) {
      DEVCHECK(cudaMemset(ptr, value, size));
    } else {
      // The underlying interface here is synchronous, not an asynchronous
      // implementation.
      DEVCHECK(cudaMemsetAsync(ptr, value, size, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorDeviceMalloc(void **ptr, size_t size,
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
      // The underlying interface here is synchronous, not an asynchronous
      // implementation.
      DEVCHECK(cudaMallocAsync(ptr, size, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorDeviceFree(void *ptr, flagcxMemType_t type,
                                       flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    DEVCHECK(cudaFreeHost(ptr));
  } else if (type == flagcxMemManaged) {
    DEVCHECK(cudaFree(ptr));
  } else {
    if (stream == NULL) {
      DEVCHECK(cudaFree(ptr));
    } else {
      // The underlying interface here is synchronous, not an asynchronous
      // implementation.
      DEVCHECK(cudaFreeAsync(ptr, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorSetDevice(int dev) {
  DEVCHECK(cudaSetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorGetDevice(int *dev) {
  DEVCHECK(cudaGetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorGetDeviceCount(int *count) {
  DEVCHECK(cudaGetDeviceCount(count));
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "KUNLUNXIN");
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorHostGetDevicePointer(void **pDevice, void *pHost) {
  DEVCHECK(cudaHostGetDevicePointer(pDevice, pHost, 0));
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorGdrMemAlloc(void **ptr, size_t size,
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

flagcxResult_t kunlunAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return flagcxSuccess;
  }
  DEVCHECK(cudaFree(ptr));
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorGdrPtrMmap(void **pcpuptr, void *devptr,
                                       size_t sz) {
  if (pcpuptr == NULL || devptr == NULL || sz == 0) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(baidu::xpu::bkcl::xccl_mmap(pcpuptr, devptr, sz));
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorGdrPtrMunmap(void *cpuptr, size_t sz) {
  if (cpuptr == NULL || sz == 0) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(baidu::xpu::bkcl::xccl_munmap(cpuptr, sz));
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorStreamCreate(flagcxStream_t *stream) {
  (*stream) = NULL;
  flagcxCalloc(stream, 1);
  DEVCHECK(cudaStreamCreateWithFlags((cudaStream_t *)(*stream),
                                     cudaStreamNonBlocking));
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorStreamDestroy(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(cudaStreamDestroy(stream->base));
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorStreamCopy(flagcxStream_t *newStream,
                                       void *oldStream) {
  (*newStream) = NULL;
  flagcxCalloc(newStream, 1);
  (*newStream)->base = (cudaStream_t)oldStream;
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorStreamFree(flagcxStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorStreamSynchronize(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(cudaStreamSynchronize(stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorStreamQuery(flagcxStream_t stream) {
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

flagcxResult_t kunlunAdaptorStreamWaitEvent(flagcxStream_t stream,
                                            flagcxEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(
        cudaStreamWaitEvent(stream->base, event->base, cudaEventWaitDefault));
  }
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorEventCreate(flagcxEvent_t *event,
                                        flagcxEventType_t eventType) {
  (*event) = NULL;
  flagcxCalloc(event, 1);
  const unsigned int flags = (eventType == flagcxEventDefault)
                                 ? cudaEventDefault
                                 : cudaEventDisableTiming;
  DEVCHECK(cudaEventCreateWithFlags(&((*event)->base), flags));
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorEventDestroy(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(cudaEventDestroy(event->base));
    free(event);
    event = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorEventRecord(flagcxEvent_t event,
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

flagcxResult_t kunlunAdaptorEventSynchronize(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(cudaEventSynchronize(event->base));
  }
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorEventQuery(flagcxEvent_t event) {
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

flagcxResult_t kunlunAdaptorIpcMemHandleCreate(flagcxIpcMemHandle_t *handle,
                                               size_t *size) {
  flagcxCalloc(handle, 1);
  if (size != NULL) {
    *size = sizeof(cudaIpcMemHandle_t);
  }
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorIpcMemHandleGet(flagcxIpcMemHandle_t handle,
                                            void *devPtr) {
  if (handle == NULL || devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaIpcGetMemHandle(&handle->base, devPtr));
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorIpcMemHandleOpen(flagcxIpcMemHandle_t handle,
                                             void **devPtr) {
  if (handle == NULL || devPtr == NULL || *devPtr != NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaIpcOpenMemHandle(devPtr, handle->base,
                                cudaIpcMemLazyEnablePeerAccess));
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorIpcMemHandleClose(void *devPtr) {
  if (devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaIpcCloseMemHandle(devPtr));
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorIpcMemHandleFree(flagcxIpcMemHandle_t handle) {
  if (handle != NULL) {
    free(handle);
  }
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorLaunchHostFunc(flagcxStream_t stream,
                                           void (*fn)(void *), void *args) {
  if (stream != NULL) {
    DEVCHECK(cudaLaunchHostFunc(stream->base, fn, args));
  }
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorLaunchDeviceFunc(flagcxStream_t stream,
                                             flagcxLaunchFunc_t fn,
                                             void *args) {
  if (stream != NULL) {
    fn(stream, args);
  }
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorGetDeviceProperties(struct flagcxDevProps *props,
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

flagcxResult_t kunlunAdaptorGetDevicePciBusId(char *pciBusId, int len,
                                              int dev) {
  if (pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaDeviceGetPCIBusId(pciBusId, len, dev));
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorGetDeviceByPciBusId(int *dev,
                                                const char *pciBusId) {
  if (dev == NULL || pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaDeviceGetByPCIBusId(dev, pciBusId));
  return flagcxSuccess;
}

struct flagcxDeviceAdaptor kunlunAdaptor {
  "KUNLUN",
      // Basic functions
      kunlunAdaptorDeviceSynchronize, kunlunAdaptorDeviceMemcpy,
      kunlunAdaptorDeviceMemset, kunlunAdaptorDeviceMalloc,
      kunlunAdaptorDeviceFree, kunlunAdaptorSetDevice, kunlunAdaptorGetDevice,
      kunlunAdaptorGetDeviceCount, kunlunAdaptorGetVendor,
      kunlunAdaptorHostGetDevicePointer,
      // GDR functions
      NULL, // flagcxResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // flagcxResult_t (*memHandleDestroy)(int dev, void *memHandle);
      kunlunAdaptorGdrMemAlloc, kunlunAdaptorGdrMemFree,
      NULL, // flagcxResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // flagcxResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      kunlunAdaptorGdrPtrMmap,   // flagcxResult_t (*gdrPtrMmap)(void **pcpuptr,
                                 // void *devptr, size_t sz);
      kunlunAdaptorGdrPtrMunmap, // flagcxResult_t (*gdrPtrMunmap)(void *cpuptr,
                                 // size_t sz);
      // Stream functions
      kunlunAdaptorStreamCreate, kunlunAdaptorStreamDestroy,
      kunlunAdaptorStreamCopy, kunlunAdaptorStreamFree,
      kunlunAdaptorStreamSynchronize, kunlunAdaptorStreamQuery,
      kunlunAdaptorStreamWaitEvent,
      // Event functions
      kunlunAdaptorEventCreate, kunlunAdaptorEventDestroy,
      kunlunAdaptorEventRecord, kunlunAdaptorEventSynchronize,
      kunlunAdaptorEventQuery,
      // IpcMemHandle functions
      kunlunAdaptorIpcMemHandleCreate, kunlunAdaptorIpcMemHandleGet,
      kunlunAdaptorIpcMemHandleOpen, kunlunAdaptorIpcMemHandleClose,
      kunlunAdaptorIpcMemHandleFree,
      // Kernel launch
      NULL, // flagcxResult_t (*launchKernel)(void *func, unsigned int block_x,
            // unsigned int block_y, unsigned int block_z, unsigned int grid_x,
            // unsigned int grid_y, unsigned int grid_z, void **args, size_t
            // share_mem, void *stream, void *memHandle);
      NULL, // flagcxResult_t (*copyArgsInit)(void **args);
      NULL, // flagcxResult_t (*copyArgsFree)(void *args);
      kunlunAdaptorLaunchDeviceFunc,
      // Others
      kunlunAdaptorGetDeviceProperties, // flagcxResult_t
                                        // (*getDeviceProperties)(struct
                                        // flagcxDevProps *props, int dev);
      kunlunAdaptorGetDevicePciBusId,   // flagcxResult_t
                                        // (*getDevicePciBusId)(char *pciBusId,
                                        // int len, int dev);
      kunlunAdaptorGetDeviceByPciBusId, // flagcxResult_t
                                        // (*getDeviceByPciBusId)(int
                                        // *dev, const char *pciBusId);
      kunlunAdaptorLaunchHostFunc,
      // DMA buffer
      NULL, // flagcxResult_t (*dmaSupport)(bool *dmaBufferSupport);
      NULL, // flagcxResult_t (*memGetHandleForAddressRange)(void *handleOut,
            // void *buffer, size_t size, unsigned long long flags);
      NULL, // flagcxResult_t (*eventElapsedTime)(float *ms, flagcxEvent_t
            // start, flagcxEvent_t end);
};

#endif // USE_KUNLUNXIN_ADAPTOR