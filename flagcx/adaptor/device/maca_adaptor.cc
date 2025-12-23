/*************************************************************************
 * Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd.
   All Rights Reserved.
 ************************************************************************/

#include "metax_adaptor.h"

#ifdef USE_METAX_ADAPTOR

std::map<flagcxMemcpyType_t, mcMemcpyKind> memcpy_type_map = {
    {flagcxMemcpyHostToDevice, mcMemcpyHostToDevice},
    {flagcxMemcpyDeviceToHost, mcMemcpyDeviceToHost},
    {flagcxMemcpyDeviceToDevice, mcMemcpyDeviceToDevice},
};

flagcxResult_t macaAdaptorDeviceSynchronize() {
  DEVCHECK(mcDeviceSynchronize());
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
                                       flagcxMemcpyType_t type,
                                       flagcxStream_t stream, void *args) {
  if (stream == NULL) {
    DEVCHECK(mcMemcpy(dst, src, size, memcpy_type_map[type]));
  } else {
    DEVCHECK(
        mcMemcpyAsync(dst, src, size, memcpy_type_map[type], stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorDeviceMemset(void *ptr, int value, size_t size,
                                       flagcxMemType_t type,
                                       flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    memset(ptr, value, size);
  } else {
    if (stream == NULL) {
      DEVCHECK(mcMemset(ptr, value, size));
    } else {
      DEVCHECK(mcMemsetAsync(ptr, value, size, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorDeviceMalloc(void **ptr, size_t size,
                                       flagcxMemType_t type,
                                       flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    DEVCHECK(mcMallocHost(ptr, size));
  } else if (type == flagcxMemManaged) {
    DEVCHECK(mcMallocManaged(ptr, size, mcMemAttachGlobal));
  } else {
    if (stream == NULL) {
      DEVCHECK(mcMalloc(ptr, size));
    } else {
      DEVCHECK(mcMallocAsync(ptr, size, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorDeviceFree(void *ptr, flagcxMemType_t type,
                                     flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    DEVCHECK(mcFreeHost(ptr));
  } else if (type == flagcxMemManaged) {
    DEVCHECK(mcFree(ptr));
  } else {
    if (stream == NULL) {
      DEVCHECK(mcFree(ptr));
    } else {
      DEVCHECK(mcFreeAsync(ptr, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorSetDevice(int dev) {
  DEVCHECK(mcSetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorGetDevice(int *dev) {
  DEVCHECK(mcGetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorGetDeviceCount(int *count) {
  DEVCHECK(mcGetDeviceCount(count));
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "METAX");
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorGdrMemAlloc(void **ptr, size_t size,
                                      void *memHandle) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(mcMalloc(ptr, size));
  mcPointerAttribute_t attrs;
  DEVCHECK(mcPointerGetAttributes(&attrs, *ptr));
  unsigned flags = 1;
  DEVCHECK(mcPointerSetAttribute(&flags, mcPointerAttributeSyncMemops,
                                 (mcDeviceptr_t)attrs.devicePointer));
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return flagcxSuccess;
  }
  DEVCHECK(mcFree(ptr));
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorStreamCreate(flagcxStream_t *stream) {
  (*stream) = NULL;
  flagcxCalloc(stream, 1);
  DEVCHECK(
      mcStreamCreateWithFlags((mcStream_t *)(*stream), mcStreamNonBlocking));
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorStreamDestroy(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(mcStreamDestroy(stream->base));
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorStreamCopy(flagcxStream_t *newStream,
                                     void *oldStream) {
  (*newStream) = NULL;
  flagcxCalloc(newStream, 1);
  (*newStream)->base = (mcStream_t)oldStream;
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorStreamFree(flagcxStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorStreamSynchronize(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(mcStreamSynchronize(stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorStreamQuery(flagcxStream_t stream) {
  flagcxResult_t res = flagcxSuccess;
  if (stream != NULL) {
    mcError_t error = mcStreamQuery(stream->base);
    if (error == mcSuccess) {
      res = flagcxSuccess;
    } else if (error == mcErrorNotReady) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

flagcxResult_t macaAdaptorStreamWaitEvent(flagcxStream_t stream,
                                          flagcxEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(mcStreamWaitEvent(stream->base, event->base, mcEventWaitDefault));
  }
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorEventCreate(flagcxEvent_t *event,
                                      flagcxEventType_t eventType) {
  (*event) = NULL;
  flagcxCalloc(event, 1);
  const unsigned int flags =
      (eventType == flagcxEventDefault) ? mcEventDefault : mcEventDisableTiming;
  DEVCHECK(mcEventCreateWithFlags(&((*event)->base), flags));
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorEventDestroy(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(mcEventDestroy(event->base));
    free(event);
    event = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorEventRecord(flagcxEvent_t event,
                                      flagcxStream_t stream) {
  if (event != NULL) {
    if (stream != NULL) {
      DEVCHECK(mcEventRecordWithFlags(event->base, stream->base,
                                      mcEventRecordDefault));
    } else {
      DEVCHECK(mcEventRecordWithFlags(event->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorEventSynchronize(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(mcEventSynchronize(event->base));
  }
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorEventQuery(flagcxEvent_t event) {
  flagcxResult_t res = flagcxSuccess;
  if (event != NULL) {
    mcError_t error = mcEventQuery(event->base);
    if (error == mcSuccess) {
      res = flagcxSuccess;
    } else if (error == mcErrorNotReady) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

flagcxResult_t macaAdaptorIpcMemHandleCreate(flagcxIpcMemHandle_t *handle,
                                             size_t *size) {
  flagcxCalloc(handle, 1);
  if (size != NULL) {
    *size = sizeof(mcIpcMemHandle_t);
  }
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorIpcMemHandleGet(flagcxIpcMemHandle_t handle,
                                          void *devPtr) {
  if (handle == NULL || devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(mcIpcGetMemHandle(&handle->base, devPtr));
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorIpcMemHandleOpen(flagcxIpcMemHandle_t handle,
                                           void **devPtr) {
  if (handle == NULL || devPtr == NULL || *devPtr != NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(
      mcIpcOpenMemHandle(devPtr, handle->base, mcIpcMemLazyEnablePeerAccess));
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorIpcMemHandleClose(void *devPtr) {
  if (devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(mcIpcCloseMemHandle(devPtr));
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorIpcMemHandleFree(flagcxIpcMemHandle_t handle) {
  if (handle != NULL) {
    free(handle);
  }
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorLaunchHostFunc(flagcxStream_t stream,
                                         void (*fn)(void *), void *args) {
  if (stream != NULL) {
    DEVCHECK(mcLaunchHostFunc(stream->base, fn, args));
  }
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorGetDeviceProperties(struct flagcxDevProps *props,
                                              int dev) {
  if (props == NULL) {
    return flagcxInvalidArgument;
  }

  mcDeviceProp_t devProp;
  DEVCHECK(mcGetDeviceProperties(&devProp, dev));
  strncpy(props->name, devProp.name, sizeof(props->name) - 1);
  props->name[sizeof(props->name) - 1] = '\0';
  props->pciBusId = devProp.pciBusID;
  props->pciDeviceId = devProp.pciDeviceID;
  props->pciDomainId = devProp.pciDomainID;
  // TODO: see if there's another way to get this info. In some mc versions,
  // mcDeviceProp_t does not have `gpuDirectRDMASupported` field
  // props->gdrSupported = devProp.gpuDirectRDMASupported;

  return flagcxSuccess;
}

flagcxResult_t macaAdaptorGetDevicePciBusId(char *pciBusId, int len, int dev) {
  if (pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(mcDeviceGetPCIBusId(pciBusId, len, dev));
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorGetDeviceByPciBusId(int *dev, const char *pciBusId) {
  if (dev == NULL || pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(mcDeviceGetByPCIBusId(dev, pciBusId));
  return flagcxSuccess;
}

struct flagcxDeviceAdaptor macaAdaptor {
  "MACA",
      // Basic functions
      macaAdaptorDeviceSynchronize, macaAdaptorDeviceMemcpy,
      macaAdaptorDeviceMemset, macaAdaptorDeviceMalloc, macaAdaptorDeviceFree,
      macaAdaptorSetDevice, macaAdaptorGetDevice, macaAdaptorGetDeviceCount,
      macaAdaptorGetVendor, NULL,
      // GDR functions
      NULL, // flagcxResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // flagcxResult_t (*memHandleDestroy)(int dev, void *memHandle);
      macaAdaptorGdrMemAlloc, macaAdaptorGdrMemFree,
      NULL, // flagcxResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // flagcxResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      NULL, // flagcxResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t
            // sz);
      NULL, // flagcxResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);
      // Stream functions
      macaAdaptorStreamCreate, macaAdaptorStreamDestroy, macaAdaptorStreamCopy,
      macaAdaptorStreamFree, macaAdaptorStreamSynchronize,
      macaAdaptorStreamQuery, macaAdaptorStreamWaitEvent,
      // Event functions
      macaAdaptorEventCreate, macaAdaptorEventDestroy, macaAdaptorEventRecord,
      macaAdaptorEventSynchronize, macaAdaptorEventQuery,
      // IpcMemHandle functions
      macaAdaptorIpcMemHandleCreate, macaAdaptorIpcMemHandleGet,
      macaAdaptorIpcMemHandleOpen, macaAdaptorIpcMemHandleClose,
      macaAdaptorIpcMemHandleFree,
      // Kernel launch
      NULL, // flagcxResult_t (*launchKernel)(void *func, unsigned int block_x,
            // unsigned int block_y, unsigned int block_z, unsigned int grid_x,
            // unsigned int grid_y, unsigned int grid_z, void **args, size_t
            // share_mem, void *stream, void *memHandle);
      NULL, // flagcxResult_t (*copyArgsInit)(void **args);
      NULL, // flagcxResult_t (*copyArgsFree)(void *args);
      NULL, // flagcxResult_t (*launchDeviceFunc)(flagcxStream_t stream,
            // void *args);
      // Others
      macaAdaptorGetDeviceProperties, // flagcxResult_t
                                      // (*getDeviceProperties)(struct
                                      // flagcxDevProps *props, int dev);
      macaAdaptorGetDevicePciBusId, // flagcxResult_t (*getDevicePciBusId)(char
                                    // *pciBusId, int len, int dev);
      macaAdaptorGetDeviceByPciBusId, // flagcxResult_t
                                      // (*getDeviceByPciBusId)(int
                                      // *dev, const char *pciBusId);
      macaAdaptorLaunchHostFunc,
      // DMA buffer
      NULL, // flagcxResult_t (*dmaSupport)(bool *dmaBufferSupport);
      NULL, // flagcxResult_t (*memGetHandleForAddressRange)(void *handleOut,
            // void *buffer, size_t size, unsigned long long flags);
      NULL, // flagcxResult_t (*eventElapsedTime)(float *ms, flagcxEvent_t
            // start, flagcxEvent_t end);
};

#endif // USE_METAX_ADAPTOR
