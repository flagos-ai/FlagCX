#include "iluvatar_corex_adaptor.h"

#ifdef USE_ILUVATAR_COREX_ADAPTOR

std::map<flagcxMemcpyType_t, cudaMemcpyKind> memcpy_type_map = {
    {flagcxMemcpyHostToDevice, cudaMemcpyHostToDevice},
    {flagcxMemcpyDeviceToHost, cudaMemcpyDeviceToHost},
    {flagcxMemcpyDeviceToDevice, cudaMemcpyDeviceToDevice},
};

flagcxResult_t ixcudaAdaptorDeviceSynchronize() {
  DEVCHECK(cudaDeviceSynchronize());
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
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

flagcxResult_t ixcudaAdaptorDeviceMemset(void *ptr, int value, size_t size,
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

flagcxResult_t ixcudaAdaptorDeviceMalloc(void **ptr, size_t size,
                                         flagcxMemType_t type,
                                         flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    DEVCHECK(cudaMallocHost(ptr, size));
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

flagcxResult_t ixcudaAdaptorDeviceFree(void *ptr, flagcxMemType_t type,
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

flagcxResult_t ixcudaAdaptorSetDevice(int dev) {
  DEVCHECK(cudaSetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorGetDevice(int *dev) {
  DEVCHECK(cudaGetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorGetDeviceCount(int *count) {
  DEVCHECK(cudaGetDeviceCount(count));
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "ILUVATAR_COREX");
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorGdrMemAlloc(void **ptr, size_t size,
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

flagcxResult_t ixcudaAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return flagcxSuccess;
  }
  DEVCHECK(cudaFree(ptr));
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorStreamCreate(flagcxStream_t *stream) {
  (*stream) = NULL;
  flagcxCalloc(stream, 1);
  DEVCHECK(cudaStreamCreateWithFlags((cudaStream_t *)(*stream),
                                     cudaStreamNonBlocking));
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorStreamDestroy(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(cudaStreamDestroy(stream->base));
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorStreamCopy(flagcxStream_t *newStream,
                                       void *oldStream) {
  (*newStream) = NULL;
  flagcxCalloc(newStream, 1);
  (*newStream)->base = (cudaStream_t)oldStream;
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorStreamFree(flagcxStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorStreamSynchronize(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(cudaStreamSynchronize(stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorStreamQuery(flagcxStream_t stream) {
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

flagcxResult_t ixcudaAdaptorStreamWaitEvent(flagcxStream_t stream,
                                            flagcxEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(cudaStreamWaitEvent(stream->base, event->base, 0));
  }
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorEventCreate(flagcxEvent_t *event,
                                        flagcxEventType_t eventType) {
  (*event) = NULL;
  flagcxCalloc(event, 1);
  const unsigned int flags = (eventType == flagcxEventDefault)
                                 ? cudaEventDefault
                                 : cudaEventDisableTiming;
  DEVCHECK(cudaEventCreateWithFlags(&((*event)->base), flags));

  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorEventDestroy(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(cudaEventDestroy(event->base));
    free(event);
    event = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorEventRecord(flagcxEvent_t event,
                                        flagcxStream_t stream) {
  if (event != NULL) {
    if (stream != NULL) {
      DEVCHECK(cudaEventRecord(event->base, stream->base));
    } else {
      DEVCHECK(cudaEventRecord(event->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorEventSynchronize(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(cudaEventSynchronize(event->base));
  }
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorEventQuery(flagcxEvent_t event) {
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

flagcxResult_t ixcudaAdaptorIpcMemHandleCreate(flagcxIpcMemHandle_t *handle,
                                               size_t *size) {
  flagcxCalloc(handle, 1);
  if (size != NULL) {
    *size = sizeof(cudaIpcMemHandle_t);
  }
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorIpcMemHandleGet(flagcxIpcMemHandle_t handle,
                                            void *devPtr) {
  if (handle == NULL || devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaIpcGetMemHandle(&handle->base, devPtr));
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorIpcMemHandleOpen(flagcxIpcMemHandle_t handle,
                                             void **devPtr) {
  if (handle == NULL || devPtr == NULL || *devPtr != NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaIpcOpenMemHandle(devPtr, handle->base,
                                cudaIpcMemLazyEnablePeerAccess));
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorIpcMemHandleClose(void *devPtr) {
  if (devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaIpcCloseMemHandle(devPtr));
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorIpcMemHandleFree(flagcxIpcMemHandle_t handle) {
  if (handle != NULL) {
    free(handle);
  }
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorLaunchHostFunc(flagcxStream_t stream,
                                           void (*fn)(void *), void *args) {
  if (stream != NULL) {
    DEVCHECK(cudaLaunchHostFunc(stream->base, fn, args));
  }
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorGetDeviceProperties(struct flagcxDevProps *props,
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

flagcxResult_t ixcudaAdaptorGetDevicePciBusId(char *pciBusId, int len,
                                              int dev) {
  if (pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaDeviceGetPCIBusId(pciBusId, len, dev));
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorGetDeviceByPciBusId(int *dev,
                                                const char *pciBusId) {
  if (dev == NULL || pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaDeviceGetByPCIBusId(dev, pciBusId));
  return flagcxSuccess;
}

static flagcxResult_t ixcudaAdaptorStreamWaitValue64(flagcxStream_t, void *,
                                                     uint64_t, int) {
  return flagcxNotSupported;
}

struct flagcxDeviceAdaptor ixcudaAdaptor {
  "IXCUDA",
      // Basic functions
      ixcudaAdaptorDeviceSynchronize, ixcudaAdaptorDeviceMemcpy,
      ixcudaAdaptorDeviceMemset, ixcudaAdaptorDeviceMalloc,
      ixcudaAdaptorDeviceFree, ixcudaAdaptorSetDevice, ixcudaAdaptorGetDevice,
      ixcudaAdaptorGetDeviceCount, ixcudaAdaptorGetVendor, NULL,
      // GDR functions
      NULL, // flagcxResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // flagcxResult_t (*memHandleDestroy)(int dev, void *memHandle);
      ixcudaAdaptorGdrMemAlloc, ixcudaAdaptorGdrMemFree,
      NULL, // flagcxResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // flagcxResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      NULL, // flagcxResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t
            // sz);
      NULL, // flagcxResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);
      // Stream functions
      ixcudaAdaptorStreamCreate, ixcudaAdaptorStreamDestroy,
      ixcudaAdaptorStreamCopy, ixcudaAdaptorStreamFree,
      ixcudaAdaptorStreamSynchronize, ixcudaAdaptorStreamQuery,
      ixcudaAdaptorStreamWaitEvent,
      // Event functions
      ixcudaAdaptorEventCreate, ixcudaAdaptorEventDestroy,
      ixcudaAdaptorEventRecord, ixcudaAdaptorEventSynchronize,
      ixcudaAdaptorEventQuery,
      // IpcMemHandle functions
      ixcudaAdaptorIpcMemHandleCreate, ixcudaAdaptorIpcMemHandleGet,
      ixcudaAdaptorIpcMemHandleOpen, ixcudaAdaptorIpcMemHandleClose,
      ixcudaAdaptorIpcMemHandleFree,
      // Kernel launch
      NULL, // flagcxResult_t (*launchKernel)(void *func, unsigned int block_x,
            // unsigned int block_y, unsigned int block_z, unsigned int grid_x,
            // unsigned int grid_y, unsigned int grid_z, void **args, size_t
            // share_mem, void *stream, void *memHandle);
      NULL, // flagcxResult_t (*copyArgsInit)(void **args);
      NULL, // flagcxResult_t (*copyArgsFree)(void *args);
      NULL, // flagcxResult_t (*launchDeviceFunc)(flagcxStream_t stream, void
            // *args);
      // Others
      ixcudaAdaptorGetDeviceProperties, // flagcxResult_t
                                        // (*getDeviceProperties)(struct
                                        // flagcxDeviceProps *props, int dev);
      ixcudaAdaptorGetDevicePciBusId,   // flagcxResult_t
                                        // (*getDevicePciBusId)(char *pciBusId,
                                        // int len, int dev);
      ixcudaAdaptorGetDeviceByPciBusId, // flagcxResult_t
                                        // (*getDeviceByPciBusId)(int *dev,
                                        // const char *pciBusId);
      ixcudaAdaptorLaunchHostFunc,

      // DMA buffer
      NULL, // flagcxResult_t (*dmaSupport)(bool *dmaBufferSupport);
      NULL, // flagcxResult_t (*memGetHandleForAddressRange)(void *handleOut,
            // void *buffer, size_t size, unsigned long long flags);
      NULL, // flagcxResult_t (*eventElapsedTime)(float *ms, flagcxEvent_t
            // start,
            // flagcxEvent_t end);
      ixcudaAdaptorStreamWaitValue64,
};

#endif // USE_ILUVATAR_COREX_ADAPTOR