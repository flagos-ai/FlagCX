#include "sunrise_adaptor.h"

#ifdef USE_SUNRISE_ADAPTOR

std::map<flagcxMemcpyType_t, tangMemcpyKind> memcpy_type_map = {
    {flagcxMemcpyHostToDevice, tangMemcpyHostToDevice},
    {flagcxMemcpyDeviceToHost, tangMemcpyDeviceToHost},
    {flagcxMemcpyDeviceToDevice, tangMemcpyDeviceToDevice},
};

flagcxResult_t ptpuAdaptorDeviceSynchronize() {
  DEVCHECK(tangDeviceSynchronize());
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
                                       flagcxMemcpyType_t type,
                                       flagcxStream_t stream, void *args) {
  (void)args;
  if (stream == NULL) {
    DEVCHECK(tangMemcpy(dst, src, size, memcpy_type_map[type]));
  } else {
    DEVCHECK(tangMemcpyAsync(dst, src, size, memcpy_type_map[type], stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorDeviceMemset(void *ptr, int value, size_t size,
                                       flagcxMemType_t type,
                                       flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    memset(ptr, value, size);
  } else {
    if (stream == NULL) {
      DEVCHECK(tangMemset(ptr, value, size));
    } else {
      DEVCHECK(tangMemsetAsync(ptr, value, size, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorDeviceMalloc(void **ptr, size_t size,
                                       flagcxMemType_t type,
                                       flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    DEVCHECK(tangMallocHost(ptr, size));
  } else {
    if (stream == NULL) {
      DEVCHECK(tangMalloc(ptr, size));
    } else {
      DEVCHECK(tangMallocAsync(ptr, size, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorDeviceFree(void *ptr, flagcxMemType_t type,
                                     flagcxStream_t stream) {
  (void)stream;
  if (type == flagcxMemHost) {
    DEVCHECK(tangFreeHost(ptr));
  } else {
    DEVCHECK(tangFree(ptr));
  }
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorSetDevice(int dev) {
  DEVCHECK(tangSetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorGetDevice(int *dev) {
  DEVCHECK(tangGetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorGetDeviceCount(int *count) {
  DEVCHECK(tangGetDeviceCount(count));
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "SUNRISE");
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorHostGetDevicePointer(void **pDevice, void *pHost) {
  // by now flags in tangHostGetDevicePointer input must be 0
  DEVCHECK(tangHostGetDevicePointer(pDevice, pHost, 0));
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorGdrMemAlloc(void **ptr, size_t size,
                                      void *memHandle) {
  (void)memHandle;
  tangStream_t allocStream;
  tangStreamCaptureMode mode = tangStreamCaptureModeRelaxed;
  DEVCHECK(tangThreadExchangeStreamCaptureMode(&mode));

  DEVCHECK(tangStreamCreateWithFlags(&allocStream, tangStreamNonBlocking));
  DEVCHECK(tangMallocAsync(ptr, size, allocStream));
  DEVCHECK(tangStreamSynchronize(allocStream));
  DEVCHECK(tangStreamDestroy(allocStream));

  mode = tangStreamCaptureModeGlobal;
  DEVCHECK(tangThreadExchangeStreamCaptureMode(&mode));
  return flagcxSuccess;
}

// TODO: by now no tangGdrMemFree function, handle it by the ref of mlu_adaptor
flagcxResult_t ptpuAdaptorGdrMemFree(void *ptr, void *memHandle) {
  (void)memHandle;
  if (ptr == NULL) {
    return flagcxSuccess;
  }
  DEVCHECK(tangFree(ptr));
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorStreamCreate(flagcxStream_t *stream) {
  (*stream) = NULL;
  flagcxCalloc(stream, 1);
  DEVCHECK(tangStreamCreateWithFlags((tangStream_t *)(*stream),
                                     tangStreamNonBlocking));
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorStreamDestroy(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(tangStreamDestroy(stream->base));
    free(stream);
  }
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorStreamCopy(flagcxStream_t *newStream,
                                     void *oldStream) {
  (*newStream) = NULL;
  flagcxCalloc(newStream, 1);
  (*newStream)->base = (tangStream_t)oldStream;
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorStreamFree(flagcxStream_t stream) {
  if (stream != NULL) {
    free(stream);
  }
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorStreamSynchronize(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(tangStreamSynchronize(stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorStreamQuery(flagcxStream_t stream) {
  flagcxResult_t res = flagcxSuccess;
  if (stream != NULL) {
    tangError_t error = tangStreamQuery(stream->base);
    if (error == tangSuccess) {
      res = flagcxSuccess;
    } else if (error == tangErrorNotReady) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

flagcxResult_t ptpuAdaptorStreamWaitEvent(flagcxStream_t stream,
                                          flagcxEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(tangStreamWaitEvent(stream->base, event->base, tangEventWaitDefault));
  }
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorStreamWaitValue64(flagcxStream_t, void *, uint64_t,
                                            int) {
  return flagcxNotSupported;
}

flagcxResult_t ptpuAdaptorStreamWriteValue64(flagcxStream_t, void *, uint64_t,
                                             int) {
  return flagcxNotSupported;
}

// TODO:ref of cuda_adaptor.cc, ignore flags with tangEventBlockingSync, tangEventInterprocess
flagcxResult_t ptpuAdaptorEventCreate(flagcxEvent_t *event,
                                      flagcxEventType_t eventType) {
  (*event) = NULL;
  flagcxCalloc(event, 1);
  const unsigned int flags = (eventType == flagcxEventDefault)
                                 ? tangEventDefault
                                 : tangEventDisableTiming;
  DEVCHECK(tangEventCreateWithFlags(&((*event)->base), flags));
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorEventDestroy(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(tangEventDestroy(event->base));
    free(event);
  }
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorEventRecord(flagcxEvent_t event,
                                      flagcxStream_t stream) {
  if (event != NULL) {
    if (stream != NULL) {
      DEVCHECK(tangEventRecordWithFlags(event->base, stream->base, tangEventRecordDefault));
    } else {
      DEVCHECK(tangEventRecordWithFlags(event->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorEventSynchronize(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(tangEventSynchronize(event->base));
  }
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorEventQuery(flagcxEvent_t event) {
  flagcxResult_t res = flagcxSuccess;
  if (event != NULL) {
    tangError_t error = tangEventQuery(event->base);
    if (error == tangSuccess) {
      res = flagcxSuccess;
    } else if (error == tangErrorNotReady) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

flagcxResult_t ptpuAdaptorEventElapsedTime(float *ms, flagcxEvent_t start,
                                           flagcxEvent_t end) {
  if (ms == NULL || start == NULL || end == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(tangEventElapsedTime(ms, start->base, end->base));
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorIpcMemHandleCreate(flagcxIpcMemHandle_t *handle,
                                             size_t *size) {
  flagcxCalloc(handle, 1);
  if (size != NULL) {
    *size = sizeof(tangIpcMemHandle_t);
  }
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorIpcMemHandleGet(flagcxIpcMemHandle_t handle,
                                          void *devPtr) {
  if (handle == NULL || devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(tangIpcGetMemHandle(&handle->base, devPtr));
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorIpcMemHandleOpen(flagcxIpcMemHandle_t handle,
                                           void **devPtr) {
  if (handle == NULL || devPtr == NULL || *devPtr != NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(tangIpcOpenMemHandle(devPtr, handle->base, tangIpcMemLazyEnablePeerAccess));
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorIpcMemHandleClose(void *devPtr) {
  if (devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(tangIpcCloseMemHandle(devPtr));
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorIpcMemHandleFree(flagcxIpcMemHandle_t handle) {
  if (handle != NULL) {
    free(handle);
  }
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorLaunchDeviceFunc(flagcxStream_t stream,
                                           flagcxLaunchFunc_t fn, void *args) {
  if (stream != NULL) {
    fn(stream, args);
  }
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorGetDeviceProperties(struct flagcxDevProps *props,
                                              int dev) {
  if (props == NULL) {
    return flagcxInvalidArgument;
  }

  tangDeviceProp devProp;
  DEVCHECK(tangGetDeviceProperties(&devProp, dev));
  strncpy(props->name, devProp.name, sizeof(props->name) - 1);
  props->name[sizeof(props->name) - 1] = '\0';
  props->pciBusId = devProp.pciBusID;
  props->pciDeviceId = devProp.pciDeviceID;
  props->pciDomainId = devProp.pciDomainID;
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorGetDevicePciBusId(char *pciBusId, int len, int dev) {
  if (pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(tangDeviceGetPCIBusId(pciBusId, len, dev));
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorGetDeviceByPciBusId(int *dev, const char *pciBusId) {
  if (dev == NULL || pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(tangDeviceGetByPCIBusId(dev, pciBusId));
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorLaunchHostFunc(flagcxStream_t stream,
                                         void (*fn)(void *), void *args) {
  if (stream != NULL) {
    DEVCHECK(tangLaunchHostFunc(stream->base, fn, args));
  }
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorDmaSupport(bool *dmaBufferSupport) {
  if (dmaBufferSupport == NULL) {
    return flagcxInvalidArgument;
  }
  *dmaBufferSupport = false;
  return flagcxSuccess;
}

flagcxResult_t ptpuAdaptorGetHandleForAddressRange(void *handleOut, void *buffer,
                                                   size_t size,
                                                   unsigned long long flags) {
  (void)handleOut;
  (void)buffer;
  (void)size;
  (void)flags;
  return flagcxNotSupported;
}

flagcxResult_t ptpuAdaptorHostRegister(void *ptr, size_t size) {
  (void)ptr;
  (void)size;
  return flagcxNotSupported;
}

flagcxResult_t ptpuAdaptorHostUnregister(void *ptr) {
  (void)ptr;
  return flagcxNotSupported;
}

struct flagcxDeviceAdaptor ptpuAdaptor {
  "PTPU",
      // Basic functions
      ptpuAdaptorDeviceSynchronize,
      ptpuAdaptorDeviceMemcpy,
      ptpuAdaptorDeviceMemset,
      ptpuAdaptorDeviceMalloc,
      ptpuAdaptorDeviceFree,
      ptpuAdaptorSetDevice,
      ptpuAdaptorGetDevice,
      ptpuAdaptorGetDeviceCount,
      ptpuAdaptorGetVendor,
      ptpuAdaptorHostGetDevicePointer,
      // GDR functions
      NULL, // flagcxResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // flagcxResult_t (*memHandleDestroy)(int dev, void *memHandle);
      ptpuAdaptorGdrMemAlloc,
      ptpuAdaptorGdrMemFree,
      NULL, // flagcxResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // flagcxResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      NULL, // flagcxResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t
            // sz);
      NULL, // flagcxResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);
      // Stream functions
      ptpuAdaptorStreamCreate,
      ptpuAdaptorStreamDestroy,
      ptpuAdaptorStreamCopy,
      ptpuAdaptorStreamFree,
      ptpuAdaptorStreamSynchronize,
      ptpuAdaptorStreamQuery,
      ptpuAdaptorStreamWaitEvent,
      ptpuAdaptorStreamWaitValue64,
      ptpuAdaptorStreamWriteValue64,
      // Event functions
      ptpuAdaptorEventCreate,
      ptpuAdaptorEventDestroy,
      ptpuAdaptorEventRecord,
      ptpuAdaptorEventSynchronize,
      ptpuAdaptorEventQuery,
      ptpuAdaptorEventElapsedTime,
      // IpcMemHandle functions
      ptpuAdaptorIpcMemHandleCreate,
      ptpuAdaptorIpcMemHandleGet,
      ptpuAdaptorIpcMemHandleOpen,
      ptpuAdaptorIpcMemHandleClose,
      ptpuAdaptorIpcMemHandleFree,
      // Kernel launch
      NULL, // flagcxResult_t (*launchKernel)(void *func, unsigned int block_x,
            // unsigned int block_y, unsigned int block_z, unsigned int grid_x,
            // unsigned int grid_y, unsigned int grid_z, void **args, size_t
            // share_mem, void *stream, void *memHandle);
      NULL, // flagcxResult_t (*copyArgsInit)(void **args);
      NULL, // flagcxResult_t (*copyArgsFree)(void *args);
      ptpuAdaptorLaunchDeviceFunc,
      // Others
      ptpuAdaptorGetDeviceProperties,
      ptpuAdaptorGetDevicePciBusId,
      ptpuAdaptorGetDeviceByPciBusId,
      // HostFunc launch
      ptpuAdaptorLaunchHostFunc,
      // DMA buffer
      ptpuAdaptorDmaSupport,
      ptpuAdaptorGetHandleForAddressRange,
      ptpuAdaptorHostRegister,
      ptpuAdaptorHostUnregister,
};
#endif // USE_SUNRISE_ADAPTOR
