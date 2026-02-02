/*************************************************************************
 * Copyright (c) 2025, ENFLAME CORPORATION. All rights reserved.
 ************************************************************************/

#include "enflame_adaptor.h"

#ifdef USE_ENFLAME_ADAPTOR

std::map<flagcxMemcpyType_t, topsMemcpyKind> memcpy_type_map = {
    {flagcxMemcpyHostToDevice, topsMemcpyHostToDevice},
    {flagcxMemcpyDeviceToHost, topsMemcpyDeviceToHost},
    {flagcxMemcpyDeviceToDevice, topsMemcpyDeviceToDevice},
};

flagcxResult_t topsAdaptorDeviceSynchronize() {
  DEVCHECK(topsDeviceSynchronize());
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
                                       flagcxMemcpyType_t type,
                                       flagcxStream_t stream, void *args) {
  if (stream == NULL) {
    DEVCHECK(topsMemcpy(dst, src, size, memcpy_type_map[type]));
  } else {
    DEVCHECK(
        topsMemcpyAsync(dst, src, size, memcpy_type_map[type], stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorDeviceMemset(void *ptr, int value, size_t size,
                                       flagcxMemType_t type,
                                       flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    memset(ptr, value, size);
  } else {
    if (stream == NULL) {
      DEVCHECK(topsMemset(ptr, value, size));
    } else {
      DEVCHECK(topsMemsetAsync(ptr, value, size, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorDeviceMalloc(void **ptr, size_t size,
                                       flagcxMemType_t type,
                                       flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    DEVCHECK(topsHostMalloc(ptr, size, topsHostMallocMapped));
  } else if (type == flagcxMemManaged) {
    DEVCHECK(topsMallocManaged(ptr, size, topsMemAttachGlobal));
  } else {
    if (stream == NULL) {
      DEVCHECK(topsMalloc(ptr, size));
    } else {
      DEVCHECK(topsMallocAsync(ptr, size, stream->base, 0));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorDeviceFree(void *ptr, flagcxMemType_t type,
                                     flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    DEVCHECK(topsHostFree(ptr));
  } else if (type == flagcxMemManaged) {
    DEVCHECK(topsFree(ptr));
  } else {
    if (stream == NULL) {
      DEVCHECK(topsFree(ptr));
    } else {
      DEVCHECK(topsFreeAsync(ptr, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorSetDevice(int dev) {
  DEVCHECK(topsSetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorGetDevice(int *dev) {
  DEVCHECK(topsGetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorGetDeviceCount(int *count) {
  DEVCHECK(topsGetDeviceCount(count));
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "ENFLAME");
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorHostGetDevicePointer(void **pDevice, void *pHost) {
  DEVCHECK(topsHostGetDevicePointer(pDevice, pHost, 0));
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorGdrMemAlloc(void **ptr, size_t size,
                                      void *memHandle) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(topsMalloc(ptr, size));
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return flagcxSuccess;
  }
  DEVCHECK(topsFree(ptr));
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorStreamCreate(flagcxStream_t *stream) {
  (*stream) = NULL;
  flagcxCalloc(stream, 1);
  DEVCHECK(topsStreamCreateWithFlags((topsStream_t *)(*stream),
                                     topsStreamNonBlocking));
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorStreamDestroy(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(topsStreamDestroy(stream->base));
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorStreamCopy(flagcxStream_t *newStream,
                                     void *oldStream) {
  (*newStream) = NULL;
  flagcxCalloc(newStream, 1);
  (*newStream)->base = (topsStream_t)oldStream;
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorStreamFree(flagcxStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorStreamSynchronize(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(topsStreamSynchronize(stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorStreamQuery(flagcxStream_t stream) {
  flagcxResult_t res = flagcxSuccess;
  if (stream != NULL) {
    topsError_t error = topsStreamQuery(stream->base);
    if (error == topsSuccess) {
      res = flagcxSuccess;
    } else if (error == topsErrorNotReady) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

flagcxResult_t topsAdaptorStreamWaitEvent(flagcxStream_t stream,
                                          flagcxEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(topsStreamWaitEvent(stream->base, event->base, 0));
  }
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorEventCreate(flagcxEvent_t *event,
                                      flagcxEventType_t eventType) {
  (*event) = NULL;
  flagcxCalloc(event, 1);
  const unsigned int flags = (eventType == flagcxEventDefault)
                                 ? topsEventDefault
                                 : topsEventDisableTiming;
  DEVCHECK(topsEventCreateWithFlags(&((*event)->base), flags));
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorEventDestroy(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(topsEventDestroy(event->base));
    free(event);
    event = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorEventRecord(flagcxEvent_t event,
                                      flagcxStream_t stream) {
  if (event != NULL) {
    if (stream != NULL) {
      DEVCHECK(topsEventRecord(event->base, stream->base));
    } else {
      DEVCHECK(topsEventRecord(event->base, NULL));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorEventSynchronize(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(topsEventSynchronize(event->base));
  }
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorEventQuery(flagcxEvent_t event) {
  flagcxResult_t res = flagcxSuccess;
  if (event != NULL) {
    topsError_t error = topsEventQuery(event->base);
    if (error == topsSuccess) {
      res = flagcxSuccess;
    } else if (error == topsErrorNotReady) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

flagcxResult_t topsAdaptorIpcMemHandleCreate(flagcxIpcMemHandle_t *handle,
                                             size_t *size) {
  flagcxCalloc(handle, 1);
  if (size != NULL) {
    *size = sizeof(topsIpcMemHandle_t);
  }
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorIpcMemHandleGet(flagcxIpcMemHandle_t handle,
                                          void *devPtr) {
  if (handle == NULL || devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(topsIpcGetMemHandle(&handle->base, devPtr));
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorIpcMemHandleOpen(flagcxIpcMemHandle_t handle,
                                           void **devPtr) {
  if (handle == NULL || devPtr == NULL || *devPtr != NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(topsIpcOpenMemHandle(devPtr, handle->base,
                                topsIpcMemLazyEnablePeerAccess));
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorIpcMemHandleClose(void *devPtr) {
  if (devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(topsIpcCloseMemHandle(devPtr));
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorIpcMemHandleFree(flagcxIpcMemHandle_t handle) {
  if (handle != NULL) {
    free(handle);
  }
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorLaunchHostFunc(flagcxStream_t stream,
                                         void (*fn)(void *), void *args) {
  if (stream != NULL) {
    DEVCHECK(topsLaunchHostFunc(stream->base, fn, args));
  }
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorLaunchDeviceFunc(flagcxStream_t stream,
                                           flagcxLaunchFunc_t fn, void *args) {
  if (stream != NULL) {
    fn(stream, args);
  }
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorGetDeviceProperties(struct flagcxDevProps *props,
                                              int dev) {
  if (props == NULL) {
    return flagcxInvalidArgument;
  }

  topsDeviceProp_t devProp;
  DEVCHECK(topsGetDeviceProperties(&devProp, dev));
  strncpy(props->name, devProp.name, sizeof(props->name) - 1);
  props->name[sizeof(props->name) - 1] = '\0';
  props->pciBusId = devProp.pciBusID;
  props->pciDeviceId = devProp.pciDeviceID;
  props->pciDomainId = devProp.pciDomainID;

  return flagcxSuccess;
}

flagcxResult_t topsAdaptorGetDevicePciBusId(char *pciBusId, int len, int dev) {
  if (pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  // TOPS uses topsGetDeviceProperties to get PCI bus ID
  topsDeviceProp_t devProp;
  DEVCHECK(topsGetDeviceProperties(&devProp, dev));
  snprintf(pciBusId, len, "%04x:%02x:%02x.%01x", devProp.pciDomainID,
           devProp.pciBusID, devProp.pciDeviceID, devProp.pciFunctionID);
  return flagcxSuccess;
}

flagcxResult_t topsAdaptorGetDeviceByPciBusId(int *dev, const char *pciBusId) {
  if (dev == NULL || pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  // Search for device by PCI bus ID
  int count;
  DEVCHECK(topsGetDeviceCount(&count));
  for (int i = 0; i < count; i++) {
    char busId[64];
    topsAdaptorGetDevicePciBusId(busId, sizeof(busId), i);
    if (strcasecmp(busId, pciBusId) == 0) {
      *dev = i;
      return flagcxSuccess;
    }
  }
  return flagcxInvalidArgument;
}

flagcxResult_t topsAdaptorDmaSupport(bool *dmaBufferSupport) {
  if (dmaBufferSupport == NULL)
    return flagcxInvalidArgument;

  // TOPS/GCU may not support DMA buffer in the same way as CUDA
  *dmaBufferSupport = false;
  return flagcxSuccess;
}

flagcxResult_t
topsAdaptorMemGetHandleForAddressRange(void *handleOut, void *buffer,
                                       size_t size, unsigned long long flags) {
  // Not supported for TOPS
  return flagcxNotSupported;
}

flagcxResult_t topsAdaptorEventElapsedTime(float *ms, flagcxEvent_t start,
                                           flagcxEvent_t end) {
  if (ms == NULL || start == NULL || end == NULL) {
    return flagcxInvalidArgument;
  }
  topsError_t error = topsEventElapsedTime(ms, start->base, end->base);
  if (error == topsSuccess) {
    return flagcxSuccess;
  } else if (error == topsErrorNotReady) {
    return flagcxInProgress;
  } else {
    return flagcxUnhandledDeviceError;
  }
}

struct flagcxDeviceAdaptor topsAdaptor {
  "TOPS",
      // Basic functions
      topsAdaptorDeviceSynchronize, topsAdaptorDeviceMemcpy,
      topsAdaptorDeviceMemset, topsAdaptorDeviceMalloc, topsAdaptorDeviceFree,
      topsAdaptorSetDevice, topsAdaptorGetDevice, topsAdaptorGetDeviceCount,
      topsAdaptorGetVendor, topsAdaptorHostGetDevicePointer,
      // GDR functions
      NULL, // flagcxResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // flagcxResult_t (*memHandleDestroy)(int dev, void *memHandle);
      topsAdaptorGdrMemAlloc, topsAdaptorGdrMemFree,
      NULL, // flagcxResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // flagcxResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      NULL, // flagcxResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t
            // sz);
      NULL, // flagcxResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);
      // Stream functions
      topsAdaptorStreamCreate, topsAdaptorStreamDestroy, topsAdaptorStreamCopy,
      topsAdaptorStreamFree, topsAdaptorStreamSynchronize,
      topsAdaptorStreamQuery, topsAdaptorStreamWaitEvent,
      // Event functions
      topsAdaptorEventCreate, topsAdaptorEventDestroy, topsAdaptorEventRecord,
      topsAdaptorEventSynchronize, topsAdaptorEventQuery,
      // IpcMemHandle functions
      topsAdaptorIpcMemHandleCreate, topsAdaptorIpcMemHandleGet,
      topsAdaptorIpcMemHandleOpen, topsAdaptorIpcMemHandleClose,
      topsAdaptorIpcMemHandleFree,
      // Kernel launch
      NULL, // flagcxResult_t (*launchKernel)(void *func, unsigned int block_x,
            // unsigned int block_y, unsigned int block_z, unsigned int grid_x,
            // unsigned int grid_y, unsigned int grid_z, void **args, size_t
            // share_mem, void *stream, void *memHandle);
      NULL, // flagcxResult_t (*copyArgsInit)(void **args);
      NULL, // flagcxResult_t (*copyArgsFree)(void *args);
      topsAdaptorLaunchDeviceFunc, // flagcxResult_t
                                   // (*launchDeviceFunc)(flagcxStream_t stream,
                                   // void *args);
      // Others
      topsAdaptorGetDeviceProperties, // flagcxResult_t
                                      // (*getDeviceProperties)(struct
                                      // flagcxDevProps *props, int dev);
      topsAdaptorGetDevicePciBusId, // flagcxResult_t (*getDevicePciBusId)(char
                                    // *pciBusId, int len, int dev);
      topsAdaptorGetDeviceByPciBusId, // flagcxResult_t
                                      // (*getDeviceByPciBusId)(int
                                      // *dev, const char *pciBusId);
      topsAdaptorLaunchHostFunc,
      // DMA buffer
      topsAdaptorDmaSupport, // flagcxResult_t (*dmaSupport)(bool
                             // *dmaBufferSupport);
      topsAdaptorMemGetHandleForAddressRange, // flagcxResult_t
                                              // (*memGetHandleForAddressRange)(void
                                              // *handleOut, void *buffer,
                                              // size_t size, unsigned long long
                                              // flags);
      topsAdaptorEventElapsedTime, // flagcxResult_t
};

#endif // USE_ENFLAME_ADAPTOR
