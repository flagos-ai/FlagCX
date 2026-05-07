#include "musa_adaptor.h"

#ifdef USE_MUSA_ADAPTOR

#include "adaptor.h"
#include "alloc.h"

std::map<flagcxMemcpyType_t, musaMemcpyKind> memcpy_type_map = {
    {flagcxMemcpyHostToDevice, musaMemcpyHostToDevice},
    {flagcxMemcpyDeviceToHost, musaMemcpyDeviceToHost},
    {flagcxMemcpyDeviceToDevice, musaMemcpyDeviceToDevice},
};

flagcxResult_t musaAdaptorDeviceSynchronize() {
  DEVCHECK(musaDeviceSynchronize());
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
                                       flagcxMemcpyType_t type,
                                       flagcxStream_t stream, void *args) {
  if (stream == NULL) {
    DEVCHECK(musaMemcpy(dst, src, size, memcpy_type_map[type]));
  } else {
    DEVCHECK(
        musaMemcpyAsync(dst, src, size, memcpy_type_map[type], stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorDeviceMemset(void *ptr, int value, size_t size,
                                       flagcxMemType_t type,
                                       flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    memset(ptr, value, size);
  } else {
    if (stream == NULL) {
      DEVCHECK(musaMemset(ptr, value, size));
    } else {
      DEVCHECK(musaMemsetAsync(ptr, value, size, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorDeviceMalloc(void **ptr, size_t size,
                                       flagcxMemType_t type,
                                       flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    DEVCHECK(musaMallocHost(ptr, size));
  } else if (type == flagcxMemManaged) {
    DEVCHECK(musaMallocManaged(ptr, size, musaMemAttachGlobal));
  } else {
    if (stream == NULL) {
      DEVCHECK(musaMalloc(ptr, size));
    } else {
      DEVCHECK(musaMalloc(ptr, size));
      // MUSA currently does not support async malloc
      // DEVCHECK(musaMallocAsync(ptr, size, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorDeviceFree(void *ptr, flagcxMemType_t type,
                                     flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    DEVCHECK(musaFreeHost(ptr));
  } else if (type == flagcxMemManaged) {
    DEVCHECK(musaFree(ptr));
  } else {
    if (stream == NULL) {
      DEVCHECK(musaFree(ptr));
    } else {
      DEVCHECK(musaFree(ptr));
      // MUSA currently does not support async malloc
      // DEVCHECK(musaFreeAsync(ptr, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorSetDevice(int dev) {
  DEVCHECK(musaSetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorGetDevice(int *dev) {
  DEVCHECK(musaGetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorGetDeviceCount(int *count) {
  DEVCHECK(musaGetDeviceCount(count));
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "MUSA");
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorGdrMemAlloc(void **ptr, size_t size,
                                      void *memHandle) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(musaMalloc(ptr, size));
  musaPointerAttributes attrs;
  DEVCHECK(musaPointerGetAttributes(&attrs, *ptr));
  unsigned flags = 1;
  DEVCHECK(muPointerSetAttribute(&flags, MU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                 (MUdeviceptr)attrs.devicePointer));
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return flagcxSuccess;
  }
  DEVCHECK(musaFree(ptr));
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorStreamCreate(flagcxStream_t *stream) {
  (*stream) = NULL;
  flagcxCalloc(stream, 1);
  DEVCHECK(musaStreamCreateWithFlags((musaStream_t *)(*stream),
                                     musaStreamNonBlocking));
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorStreamDestroy(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(musaStreamDestroy(stream->base));
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorStreamCopy(flagcxStream_t *newStream,
                                     void *oldStream) {
  (*newStream) = NULL;
  flagcxCalloc(newStream, 1);
  (*newStream)->base = (musaStream_t)oldStream;
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorStreamFree(flagcxStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorStreamSynchronize(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(musaStreamSynchronize(stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorStreamQuery(flagcxStream_t stream) {
  flagcxResult_t res = flagcxSuccess;
  if (stream != NULL) {
    musaError error = musaStreamQuery(stream->base);
    if (error == musaSuccess) {
      res = flagcxSuccess;
    } else if (error == musaErrorNotReady) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

flagcxResult_t musaAdaptorStreamWaitEvent(flagcxStream_t stream,
                                          flagcxEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(
        musaStreamWaitEvent(stream->base, event->base, musaEventWaitDefault));
  }
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorEventCreate(flagcxEvent_t *event,
                                      flagcxEventType_t eventType) {
  (*event) = NULL;
  flagcxCalloc(event, 1);
  const unsigned int flags = (eventType == flagcxEventDefault)
                                 ? musaEventDefault
                                 : musaEventDisableTiming;
  DEVCHECK(musaEventCreateWithFlags(&((*event)->base), flags));
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorEventDestroy(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(musaEventDestroy(event->base));
    free(event);
    event = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorEventRecord(flagcxEvent_t event,
                                      flagcxStream_t stream) {
  if (event != NULL) {
    if (stream != NULL) {
      DEVCHECK(musaEventRecordWithFlags(event->base, stream->base,
                                        musaEventRecordDefault));
    } else {
      DEVCHECK(musaEventRecordWithFlags(event->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorEventSynchronize(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(musaEventSynchronize(event->base));
  }
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorEventQuery(flagcxEvent_t event) {
  flagcxResult_t res = flagcxSuccess;
  if (event != NULL) {
    musaError error = musaEventQuery(event->base);
    if (error == musaSuccess) {
      res = flagcxSuccess;
    } else if (error == musaErrorNotReady) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

flagcxResult_t musaAdaptorIpcMemHandleCreate(flagcxIpcMemHandle_t *handle,
                                             size_t *size) {
  // to be implemented
  return flagcxNotSupported;
}

flagcxResult_t musaAdaptorIpcMemHandleGet(flagcxIpcMemHandle_t handle,
                                          void *devPtr) {
  // to be implemented
  return flagcxNotSupported;
}

flagcxResult_t musaAdaptorIpcMemHandleOpen(flagcxIpcMemHandle_t handle,
                                           void **devPtr) {
  // to be implemented
  return flagcxNotSupported;
}

flagcxResult_t musaAdaptorIpcMemHandleClose(void *devPtr) {
  // to be implemented
  return flagcxNotSupported;
}

flagcxResult_t musaAdaptorIpcMemHandleFree(flagcxIpcMemHandle_t handle) {
  // to be implemented
  return flagcxNotSupported;
}

flagcxResult_t musaAdaptorLaunchHostFunc(flagcxStream_t stream,
                                         void (*fn)(void *), void *args) {
  if (stream != NULL) {
    DEVCHECK(musaLaunchHostFunc(stream->base, fn, args));
  }
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorGetDeviceProperties(struct flagcxDevProps *props,
                                              int dev) {
  if (props == NULL) {
    return flagcxInvalidArgument;
  }

  musaDeviceProp devProp;
  DEVCHECK(musaGetDeviceProperties(&devProp, dev));
  strncpy(props->name, devProp.name, sizeof(props->name) - 1);
  props->name[sizeof(props->name) - 1] = '\0';
  props->pciBusId = devProp.pciBusID;
  props->pciDeviceId = devProp.pciDeviceID;
  props->pciDomainId = devProp.pciDomainID;
  // TODO: see if there's another way to get this info. In some musa versions,
  // musaDeviceProp does not have `gpuDirectRDMASupported` field
  // props->gdrSupported = devProp.gpuDirectRDMASupported;

  return flagcxSuccess;
}

flagcxResult_t musaAdaptorGetDevicePciBusId(char *pciBusId, int len, int dev) {
  if (pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(musaDeviceGetPCIBusId(pciBusId, len, dev));
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorGetDeviceByPciBusId(int *dev, const char *pciBusId) {
  if (dev == NULL || pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(musaDeviceGetByPCIBusId(dev, pciBusId));
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorStreamWaitValue64(flagcxStream_t, void *, uint64_t,
                                            int) {
  return flagcxNotSupported;
}
flagcxResult_t musaAdaptorStreamWriteValue64(flagcxStream_t, void *, uint64_t,
                                             int) {
  return flagcxNotSupported;
}
flagcxResult_t musaAdaptorEventElapsedTime(float *, flagcxEvent_t,
                                           flagcxEvent_t) {
  return flagcxNotSupported;
}

flagcxResult_t musaAdaptorHostRegister(void *, size_t) {
  return flagcxNotSupported;
}
flagcxResult_t musaAdaptorHostUnregister(void *) { return flagcxNotSupported; }

struct flagcxDeviceAdaptor musaAdaptor {
  "MUSA",
      // Basic functions
      musaAdaptorDeviceSynchronize, musaAdaptorDeviceMemcpy,
      musaAdaptorDeviceMemset, musaAdaptorDeviceMalloc, musaAdaptorDeviceFree,
      musaAdaptorSetDevice, musaAdaptorGetDevice, musaAdaptorGetDeviceCount,
      musaAdaptorGetVendor, NULL,
      // GDR functions
      NULL, // flagcxResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // flagcxResult_t (*memHandleDestroy)(int dev, void *memHandle);
      musaAdaptorGdrMemAlloc, musaAdaptorGdrMemFree,
      NULL, // flagcxResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // flagcxResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      NULL, // flagcxResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t
            // sz);
      NULL, // flagcxResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);
      // Stream functions
      musaAdaptorStreamCreate, musaAdaptorStreamDestroy, musaAdaptorStreamCopy,
      musaAdaptorStreamFree, musaAdaptorStreamSynchronize,
      musaAdaptorStreamQuery, musaAdaptorStreamWaitEvent,
      musaAdaptorStreamWaitValue64, musaAdaptorStreamWriteValue64,
      // Event functions
      musaAdaptorEventCreate, musaAdaptorEventDestroy, musaAdaptorEventRecord,
      musaAdaptorEventSynchronize, musaAdaptorEventQuery,
      musaAdaptorEventElapsedTime,
      // IpcMemHandle functions
      musaAdaptorIpcMemHandleCreate, musaAdaptorIpcMemHandleGet,
      musaAdaptorIpcMemHandleOpen, musaAdaptorIpcMemHandleClose,
      musaAdaptorIpcMemHandleFree,
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
      musaAdaptorGetDeviceProperties, // flagcxResult_t
                                      // (*getDeviceProperties)(struct
                                      // flagcxDevProps *props, int dev);
      musaAdaptorGetDevicePciBusId, // flagcxResult_t (*getDevicePciBusId)(char
                                    // *pciBusId, int len, int dev);
      musaAdaptorGetDeviceByPciBusId, // flagcxResult_t
                                      // (*getDeviceByPciBusId)(int
                                      // *dev, const char *pciBusId);
      musaAdaptorLaunchHostFunc,
      // DMA buffer
      NULL, // flagcxResult_t (*dmaSupport)(bool *dmaBufferSupport);
      NULL, // flagcxResult_t (*memGetHandleForAddressRange)(void *handleOut,
            // void *buffer, size_t size, unsigned long long flags);
      musaAdaptorHostRegister,   // flagcxResult_t (*hostRegister)(void *,
                                 // size_t);
      musaAdaptorHostUnregister, // flagcxResult_t (*hostUnregister)(void *);
};

#endif // USE_MUSA_ADAPTOR
