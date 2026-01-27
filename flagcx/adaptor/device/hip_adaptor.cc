#include "amd_adaptor.h"

#ifdef USE_AMD_ADAPTOR

std::map<flagcxMemcpyType_t, hipMemcpyKind> memcpy_type_map = {
    {flagcxMemcpyHostToDevice, hipMemcpyHostToDevice},
    {flagcxMemcpyDeviceToHost, hipMemcpyDeviceToHost},
    {flagcxMemcpyDeviceToDevice, hipMemcpyDeviceToDevice},
};

flagcxResult_t hipAdaptorDeviceSynchronize() {
  DEVCHECK(hipDeviceSynchronize());
  return flagcxSuccess;
}

flagcxResult_t hipAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
                                      flagcxMemcpyType_t type,
                                      flagcxStream_t stream, void *args) {
  if (stream == NULL) {
    DEVCHECK(hipMemcpy(dst, src, size, memcpy_type_map[type]));
  } else {
    DEVCHECK(
        hipMemcpyAsync(dst, src, size, memcpy_type_map[type], stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t hipAdaptorDeviceMemset(void *ptr, int value, size_t size,
                                      flagcxMemType_t type,
                                      flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    memset(ptr, value, size);
  } else {
    if (stream == NULL) {
      DEVCHECK(hipMemset(ptr, value, size));
    } else {
      DEVCHECK(hipMemsetAsync(ptr, value, size, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t hipAdaptorDeviceMalloc(void **ptr, size_t size,
                                      flagcxMemType_t type,
                                      flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    DEVCHECK(hipHostMalloc(ptr, size));
  } else if (type == flagcxMemManaged) {
    DEVCHECK(hipMallocManaged(ptr, size, hipMemAttachGlobal));
  } else {
    if (stream == NULL) {
      DEVCHECK(hipMalloc(ptr, size));
    } else {
      DEVCHECK(hipMallocAsync(ptr, size, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t hipAdaptorDeviceFree(void *ptr, flagcxMemType_t type,
                                    flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    DEVCHECK(hipFreeHost(ptr));
  } else if (type == flagcxMemManaged) {
    DEVCHECK(hipFree(ptr));
  } else {
    if (stream == NULL) {
      DEVCHECK(hipFree(ptr));
    } else {
      DEVCHECK(hipFreeAsync(ptr, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t hipAdaptorSetDevice(int dev) {
  DEVCHECK(hipSetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t hipAdaptorGetDevice(int *dev) {
  DEVCHECK(hipGetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t hipAdaptorGetDeviceCount(int *count) {
  DEVCHECK(hipGetDeviceCount(count));
  return flagcxSuccess;
}

flagcxResult_t hipAdaptorGetVendor(char *vendor) {
  strncpy(vendor, "AMD", MAX_VENDOR_LEN - 1);
  vendor[MAX_VENDOR_LEN - 1] = '\0';
  return flagcxSuccess;
}

flagcxResult_t hipAdaptorGdrMemAlloc(void **ptr, size_t size, void *memHandle) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(hipMalloc(ptr, size));
  hipPointerAttribute_t attrs;
  DEVCHECK(hipPointerGetAttributes(&attrs, *ptr));
  unsigned flags = 1;
  DEVCHECK(hipPointerSetAttribute(&flags, HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                  (hipDeviceptr_t)attrs.devicePointer));
  return flagcxSuccess;
}

flagcxResult_t hipAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return flagcxSuccess;
  }
  DEVCHECK(hipFree(ptr));
  return flagcxSuccess;
}

flagcxResult_t hipAdaptorStreamCreate(flagcxStream_t *stream) {
  (*stream) = NULL;
  flagcxCalloc(stream, 1);
  DEVCHECK(
      hipStreamCreateWithFlags((hipStream_t *)(*stream), hipStreamNonBlocking));
  return flagcxSuccess;
}

flagcxResult_t hipAdaptorStreamDestroy(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(hipStreamDestroy(stream->base));
    free(stream);
  }
  return flagcxSuccess;
}

flagcxResult_t hipAdaptorStreamCopy(flagcxStream_t *newStream,
                                    void *oldStream) {
  (*newStream) = NULL;
  flagcxCalloc(newStream, 1);
  (*newStream)->base = (hipStream_t)oldStream;
  return flagcxSuccess;
}

flagcxResult_t hipAdaptorStreamFree(flagcxStream_t stream) {
  if (stream != NULL) {
    free(stream);
  }
  return flagcxSuccess;
}

flagcxResult_t hipAdaptorStreamSynchronize(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(hipStreamSynchronize(stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t hipAdaptorStreamQuery(flagcxStream_t stream) {
  flagcxResult_t res = flagcxSuccess;
  if (stream != NULL) {
    hipError_t error = hipStreamQuery(stream->base);
    if (error == hipSuccess) {
      res = flagcxSuccess;
    } else if (error == hipErrorNotReady) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

flagcxResult_t hipAdaptorStreamWaitEvent(flagcxStream_t stream,
                                         flagcxEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(hipStreamWaitEvent(stream->base, event->base, 0));
  }
  return flagcxSuccess;
}

flagcxResult_t hipAdaptorEventCreate(flagcxEvent_t *event,
                                     flagcxEventType_t eventType) {
  (*event) = NULL;
  flagcxCalloc(event, 1);
  const unsigned int flags = (eventType == flagcxEventDefault)
                                 ? hipEventDefault
                                 : hipEventDisableTiming;
  DEVCHECK(hipEventCreateWithFlags(&((*event)->base), flags));
  return flagcxSuccess;
}

flagcxResult_t hipAdaptorEventDestroy(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(hipEventDestroy(event->base));
    free(event);
  }
  return flagcxSuccess;
}

flagcxResult_t hipAdaptorEventRecord(flagcxEvent_t event,
                                     flagcxStream_t stream) {
  if (event != NULL && stream != NULL) {
    DEVCHECK(hipEventRecord(event->base, stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t hipAdaptorEventSynchronize(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(hipEventSynchronize(event->base));
  }
  return flagcxSuccess;
}

flagcxResult_t hipAdaptorEventQuery(flagcxEvent_t event) {
  flagcxResult_t res = flagcxSuccess;
  if (event != NULL) {
    hipError_t error = hipEventQuery(event->base);
    if (error == hipSuccess) {
      res = flagcxSuccess;
    } else if (error == hipErrorNotReady) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

flagcxResult_t hipAdaptorIpcMemHandleCreate(flagcxIpcMemHandle_t *handle,
                                            size_t *size) {
  // to be implemented
  return flagcxNotSupported;
}

flagcxResult_t hipAdaptorIpcMemHandleGet(flagcxIpcMemHandle_t handle,
                                         void *devPtr) {
  // to be implemented
  return flagcxNotSupported;
}

flagcxResult_t hipAdaptorIpcMemHandleOpen(flagcxIpcMemHandle_t handle,
                                          void **devPtr) {
  // to be implemented
  return flagcxNotSupported;
}

flagcxResult_t hipAdaptorIpcMemHandleClose(void *devPtr) {
  // to be implemented
  return flagcxNotSupported;
}

flagcxResult_t hipAdaptorIpcMemHandleFree(flagcxIpcMemHandle_t handle) {
  // to be implemented
  return flagcxNotSupported;
}

flagcxResult_t hipAdaptorLaunchHostFunc(flagcxStream_t stream,
                                        void (*fn)(void *), void *args) {
  if (stream != NULL) {
    DEVCHECK(hipLaunchHostFunc(stream->base, fn, args));
  }
  return flagcxSuccess;
}
flagcxResult_t hipAdaptorLaunchDeviceFunc(flagcxStream_t stream,
                                          flagcxLaunchFunc_t fn, void *args) {
  if (stream != NULL) {
    fn(stream, args);
  }
  return flagcxSuccess;
}

flagcxResult_t hipAdaptorGetDeviceProperties(struct flagcxDevProps *props,
                                             int dev) {
  if (props == NULL) {
    return flagcxInvalidArgument;
  }

  hipDeviceProp_t devProp;
  DEVCHECK(hipGetDeviceProperties(&devProp, dev));
  strncpy(props->name, devProp.name, sizeof(props->name) - 1);
  props->name[sizeof(props->name) - 1] = '\0';
  props->pciBusId = devProp.pciBusID;
  props->pciDeviceId = devProp.pciDeviceID;
  props->pciDomainId = devProp.pciDomainID;
  // props->gdrSupported = devProp.gpuDirectRDMASupported;

  return flagcxSuccess;
}

flagcxResult_t hipAdaptorGetDevicePciBusId(char *pciBusId, int len, int dev) {
  if (pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(hipDeviceGetPCIBusId(pciBusId, len, dev));
  return flagcxSuccess;
}

flagcxResult_t hipAdaptorGetDeviceByPciBusId(int *dev, const char *pciBusId) {
  if (dev == NULL || pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(hipDeviceGetByPCIBusId(dev, pciBusId));
  return flagcxSuccess;
}

flagcxResult_t hipAdaptorDmaSupport(bool *dmaBufferSupport) {
  *dmaBufferSupport = false;
  return flagcxSuccess;
}

flagcxResult_t hipAdaptorMemGetHandleForAddressRange(void *handleOut,
                                                     void *buffer, size_t size,
                                                     unsigned long long flags) {
  return flagcxNotSupported;
}

struct flagcxDeviceAdaptor hipAdaptor {
  "HIP",
      // Basic functions
      hipAdaptorDeviceSynchronize, hipAdaptorDeviceMemcpy,
      hipAdaptorDeviceMemset, hipAdaptorDeviceMalloc, hipAdaptorDeviceFree,
      hipAdaptorSetDevice, hipAdaptorGetDevice, hipAdaptorGetDeviceCount,
      hipAdaptorGetVendor,
      NULL, // flagcxResult_t (*hostGetDevicePointer)(void **pDevice, void
            // *pHost);
      // GDR functions
      NULL, // flagcxResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // flagcxResult_t (*memHandleDestroy)(int dev, void *memHandle);
      hipAdaptorGdrMemAlloc, hipAdaptorGdrMemFree,
      NULL, // flagcxResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // flagcxResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      NULL, // flagcxResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t
            // sz);
      NULL, // flagcxResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);
      // Stream functions
      hipAdaptorStreamCreate, hipAdaptorStreamDestroy, hipAdaptorStreamCopy,
      hipAdaptorStreamFree, hipAdaptorStreamSynchronize, hipAdaptorStreamQuery,
      hipAdaptorStreamWaitEvent,
      // Event functions
      hipAdaptorEventCreate, hipAdaptorEventDestroy, hipAdaptorEventRecord,
      hipAdaptorEventSynchronize, hipAdaptorEventQuery,
      // IpcMemHandle functions
      hipAdaptorIpcMemHandleCreate, hipAdaptorIpcMemHandleGet,
      hipAdaptorIpcMemHandleOpen, hipAdaptorIpcMemHandleClose,
      hipAdaptorIpcMemHandleFree,
      // Kernel launch
      NULL, // flagcxResult_t (*launchKernel)(void *func, unsigned int block_x,
            // unsigned int block_y, unsigned int block_z, unsigned int grid_x,
            // unsigned int grid_y, unsigned int grid_z, void **args, size_t
            // share_mem, void *stream, void *memHandle);
      NULL, // flagcxResult_t (*copyArgsInit)(void **args);
      NULL, // flagcxResult_t (*copyArgsFree)(void *args);
      hipAdaptorLaunchDeviceFunc, // flagcxResult_t
                                  // (*launchDeviceFunc)(flagcxStream_t stream,
                                  // void *args);
      // Others
      hipAdaptorGetDeviceProperties, // flagcxResult_t
                                     // (*getDeviceProperties)(struct
                                     // flagcxDevProps *props, int dev);
      hipAdaptorGetDevicePciBusId,   // flagcxResult_t (*getDevicePciBusId)(char
                                     // *pciBusId, int len, int dev);
      hipAdaptorGetDeviceByPciBusId, // flagcxResult_t
                                     // (*getDeviceByPciBusId)(int
                                     // *dev, const char *pciBusId);
      hipAdaptorLaunchHostFunc,
      // DMA buffer
      hipAdaptorDmaSupport, // flagcxResult_t (*dmaSupport)(bool
                            // *dmaBufferSupport);
      hipAdaptorMemGetHandleForAddressRange, // flagcxResult_t
                                             // (*memGetHandleForAddressRange)(void
                                             // *handleOut, void *buffer,
                                             // size_t size, unsigned long long
                                             // flags);
      NULL, // flagcxResult_t (*eventElapsedTime)(float *ms, flagcxEvent_t
            // start, flagcxEvent_t end);
};

#endif // USE_AMD_ADAPTOR
