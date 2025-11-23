#include "tsmicro_adaptor.h"

#ifdef USE_TSM_ADAPTOR

std::map<flagcxMemcpyType_t, txMemcpyKind> memcpy_type_map = {
    {flagcxMemcpyHostToDevice, txMemcpyHostToDevice},
    {flagcxMemcpyDeviceToHost, txMemcpyDeviceToHost},
    {flagcxMemcpyDeviceToDevice, txMemcpyDeviceToDevice},
};

flagcxResult_t tsmicroAdaptorDeviceSynchronize() {
  DEVCHECK(txDeviceSynchronize());
  return flagcxSuccess;
}

flagcxResult_t tsmicroAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
                                       flagcxMemcpyType_t type,
                                       flagcxStream_t stream, void *args) {
  if (stream == NULL) {
    DEVCHECK(txMemcpy(dst, src, size, memcpy_type_map[type]));
  } else {
    DEVCHECK(txMemcpyAsync(dst, src, size, memcpy_type_map[type], stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t tsmicroAdaptorDeviceMemset(void *ptr, int value, size_t size,
                                       flagcxMemType_t type,
                                       flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    memset(ptr, value, size);
  } else {
    if (stream == NULL) {
      // TODO: supported later
      // DEVCHECK(txMemset(ptr, value, size));
    } else {
      // DEVCHECK(txMemset(ptr, value, size));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t tsmicroAdaptorDeviceMalloc(void **ptr, size_t size,
                                       flagcxMemType_t type,
                                       flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    DEVCHECK(txMallocHost(ptr, size));
  } else if (type == flagcxMemDevice) {
    DEVCHECK(txMalloc(ptr, size));
  }
  return flagcxSuccess;
}

flagcxResult_t tsmicroAdaptorDeviceFree(void *ptr, flagcxMemType_t type,
                                     flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    DEVCHECK(txFreeHost(ptr));
  } else if (type == flagcxMemManaged) {
    DEVCHECK(txFree(ptr));
  } else {
    if (stream == NULL) {
      DEVCHECK(txFree(ptr));
    } else {
      DEVCHECK(txFree(ptr));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t tsmicroAdaptorSetDevice(int dev) {
  DEVCHECK(txSetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t tsmicroAdaptorGetDevice(int *dev) {
  DEVCHECK(txGetDevice((uint32_t*)dev));
  return flagcxSuccess;
}

flagcxResult_t tsmicroAdaptorGetDeviceCount(int *count) {
  DEVCHECK(txGetDeviceCount((uint32_t *)count));
  return flagcxSuccess;
}

flagcxResult_t tsmicroAdaptorGetVendor(char *vendor) {
  if(vendor != NULL){
    strcpy(vendor, "TSMICRO");
  }
  return flagcxSuccess;
}

flagcxResult_t tsmicroAdaptorHostGetDevicePointer(void **pDevice, void *pHost) {
  return flagcxNotSupported;
}

flagcxResult_t tsmicroAdaptorGdrMemAlloc(void **ptr, size_t size,
                                      void *memHandle) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(txMalloc(ptr, size));
  return flagcxSuccess;
}

flagcxResult_t tsmicroAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return flagcxSuccess;
  }
  return flagcxNotSupported;
}

flagcxResult_t tsmicroAdaptorStreamCreate(flagcxStream_t *stream) {
  (*stream) = NULL;
  flagcxCalloc(stream, 1);
  DEVCHECK(txStreamCreate((txStream_t *)(*stream)));
  return flagcxSuccess;
}

flagcxResult_t tsmicroAdaptorStreamDestroy(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(txStreamDestroy(stream->base));
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t tsmicroAdaptorStreamCopy(flagcxStream_t *newStream,
                                     void *oldStream) {
  (*newStream) = NULL;
  flagcxCalloc(newStream, 1);
  memcpy((void *)*newStream, oldStream, sizeof(txStream_t));
  return flagcxSuccess;
}

flagcxResult_t tsmicroAdaptorStreamFree(flagcxStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t tsmicroAdaptorStreamSynchronize(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(txStreamSynchronize(stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t tsmicroAdaptorStreamQuery(flagcxStream_t stream) {
  flagcxResult_t res = flagcxSuccess;
  if (stream != NULL) {
    txError_t error = txStreamQuery(stream->base);
    if (error == TX_SUCCESS) {
      res = flagcxSuccess;
    } else if (error == TX_ERROR_NOT_READY) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

flagcxResult_t tsmicroAdaptorStreamWaitEvent(flagcxStream_t stream,
                                          flagcxEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(txStreamWaitEvent(stream->base, event->base));
  }
  return flagcxSuccess;
}

flagcxResult_t tsmicroAdaptorEventCreate(flagcxEvent_t *event,
                                      flagcxEventType_t eventType) {
  (*event) = NULL;
  flagcxCalloc(event, 1);
  DEVCHECK(txEventCreate(&(*event)->base));
  return flagcxSuccess;
}

flagcxResult_t tsmicroAdaptorEventDestroy(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(txEventDestroy(event->base));
    free(event);
    event = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t tsmicroAdaptorEventRecord(flagcxEvent_t event,
                                      flagcxStream_t stream) {
  if (event != NULL) {
      DEVCHECK(txEventRecord(event->base, stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t tsmicroAdaptorEventSynchronize(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(txEventSynchronize(event->base));
  }
  return flagcxSuccess;
}

flagcxResult_t tsmicroAdaptorEventQuery(flagcxEvent_t event) {
  // TODO: supported later
  return flagcxNotSupported;
}

flagcxResult_t tsmicroAdaptorIpcMemHandleCreate(flagcxIpcMemHandle_t *handle,
                                             size_t *size) {
  flagcxCalloc(handle, 1);
  if (size != NULL) {
    *size = sizeof(txIpcMemHandle_t);
  }
  return flagcxSuccess;
}

flagcxResult_t tsmicroAdaptorIpcMemHandleGet(flagcxIpcMemHandle_t handle,
                                          void *devPtr) {
  if (handle == NULL || devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(txIpcGetMemHandle(&handle->base, devPtr));
  return flagcxSuccess;
}

flagcxResult_t tsmicroAdaptorIpcMemHandleOpen(flagcxIpcMemHandle_t handle,
                                           void **devPtr) {
  if (handle == NULL || devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(txIpcOpenMemHandle(devPtr, handle->base));
  return flagcxSuccess;
}

flagcxResult_t tsmicroAdaptorIpcMemHandleClose(void *devPtr) {
  if (devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(txIpcCloseMemHandle(devPtr));
  return flagcxSuccess;
}

flagcxResult_t tsmicroAdaptorIpcMemHandleFree(flagcxIpcMemHandle_t handle) {
  if (handle != NULL) {
    free(handle);
  }
  return flagcxSuccess;
}

flagcxResult_t tsmicroAdaptorLaunchHostFunc(flagcxStream_t stream,
                                         void (*fn)(void *), void *args) {
  return flagcxNotSupported;
}

flagcxResult_t tsmicroAdaptorLaunchDeviceFunc(flagcxStream_t stream,
                                           flagcxLaunchFunc_t fn, void *args) {
  if (stream != NULL) {
    fn(stream, args);
  }
  return flagcxSuccess;
}

flagcxResult_t tsmicroAdaptorGetDeviceProperties(struct flagcxDevProps *props,
                                              int dev) {
  if (props == NULL) {
    return flagcxInvalidArgument;
  }
  txDeviceProperty devProp;
  txGetDeviceProperty(dev, &devProp);

  strncpy(props->name, devProp.devProp.devName, sizeof(props->name) - 1);
  props->name[sizeof(props->name) - 1] = '\0';
  props->pciBusId = devProp.devProp.busId;
  props->pciDeviceId = devProp.devProp.deviceId;
  props->pciDomainId = devProp.devProp.domainId;
  return flagcxSuccess;
}

flagcxResult_t tsmicroAdaptorGetDevicePciBusId(char *pciBusId, int len, int dev) {
  if (pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  // TODO: supported later
  return flagcxNotSupported;
}

flagcxResult_t tsmicroAdaptorGetDeviceByPciBusId(int *dev, const char *pciBusId) {
  if (dev == NULL || pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(txGetDeviceByPCIBusId((uint32_t *)dev, pciBusId));
  return flagcxSuccess;
}


flagcxResult_t tsmicroAdaptorDmaSupport(bool *dmaBufferSupport) {
  if (dmaBufferSupport == NULL)
    return flagcxInvalidArgument;
  // TODO: supported later
  return flagcxNotSupported;
}

flagcxResult_t
tsmicroAdaptorMemGetHandleForAddressRange(void *handleOut, void *buffer,
                                       size_t size, unsigned long long flags) {
  return flagcxNotSupported;
}

flagcxResult_t tsmicroAdaptorEventElapsedTime(float *ms, flagcxEvent_t start,
                                           flagcxEvent_t end) {
  if (ms == NULL || start == NULL || end == NULL) {
    return flagcxInvalidArgument;
  }
  txError_t error = txEventElapsedTime(ms, start->base, end->base);
  if (error == TX_SUCCESS) {
    return flagcxSuccess;
  } else if (error == TX_ERROR_INVALID_HANDLE) {
    return flagcxInvalidArgument;
  } else if (error == TX_ERROR_NOT_READY) {
    return flagcxInProgress;
  } else {
    return flagcxUnhandledDeviceError;
  }
}
struct flagcxDeviceAdaptor tsmicroAdaptor {
  "TSM",
      // Basic functions
      tsmicroAdaptorDeviceSynchronize, tsmicroAdaptorDeviceMemcpy,
      tsmicroAdaptorDeviceMemset, tsmicroAdaptorDeviceMalloc, tsmicroAdaptorDeviceFree,
      tsmicroAdaptorSetDevice, tsmicroAdaptorGetDevice, tsmicroAdaptorGetDeviceCount,
      tsmicroAdaptorGetVendor, tsmicroAdaptorHostGetDevicePointer,
      // GDR functions
      NULL, // flagcxResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // flagcxResult_t (*memHandleDestroy)(int dev, void *memHandle);
      tsmicroAdaptorGdrMemAlloc, tsmicroAdaptorGdrMemFree,
      NULL, // flagcxResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // flagcxResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      NULL, // flagcxResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t
            // sz);
      NULL, // flagcxResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);
      // Stream functions
      tsmicroAdaptorStreamCreate, tsmicroAdaptorStreamDestroy, tsmicroAdaptorStreamCopy,
      tsmicroAdaptorStreamFree, tsmicroAdaptorStreamSynchronize,
      tsmicroAdaptorStreamQuery, tsmicroAdaptorStreamWaitEvent,
      // Event functions
      tsmicroAdaptorEventCreate, tsmicroAdaptorEventDestroy, tsmicroAdaptorEventRecord,
      tsmicroAdaptorEventSynchronize, tsmicroAdaptorEventQuery,
      // IpcMemHandle functions
      tsmicroAdaptorIpcMemHandleCreate, tsmicroAdaptorIpcMemHandleGet,
      tsmicroAdaptorIpcMemHandleOpen, tsmicroAdaptorIpcMemHandleClose,
      tsmicroAdaptorIpcMemHandleFree,
      // Kernel launch
      NULL, // flagcxResult_t (*launchKernel)(void *func, unsigned int block_x,
            // unsigned int block_y, unsigned int block_z, unsigned int grid_x,
            // unsigned int grid_y, unsigned int grid_z, void **args, size_t
            // share_mem, void *stream, void *memHandle);
      NULL, // flagcxResult_t (*copyArgsInit)(void **args);
      NULL, // flagcxResult_t (*copyArgsFree)(void *args);
      tsmicroAdaptorLaunchDeviceFunc, // flagcxResult_t
                                   // (*launchDeviceFunc)(flagcxStream_t stream,
                                   // void *args);
      // Others
      tsmicroAdaptorGetDeviceProperties, // flagcxResult_t
                                      // (*getDeviceProperties)(struct
                                      // flagcxDevProps *props, int dev);
      tsmicroAdaptorGetDevicePciBusId, // flagcxResult_t (*getDevicePciBusId)(char
                                    // *pciBusId, int len, int dev);
      tsmicroAdaptorGetDeviceByPciBusId, // flagcxResult_t
                                      // (*getDeviceByPciBusId)(int
                                      // *dev, const char *pciBusId);
      tsmicroAdaptorLaunchHostFunc,
      // DMA buffer
      tsmicroAdaptorDmaSupport, // flagcxResult_t (*dmaSupport)(bool
                             // *dmaBufferSupport);
      tsmicroAdaptorMemGetHandleForAddressRange, // flagcxResult_t
                                              // (*memGetHandleForAddressRange)(void
                                              // *handleOut, void *buffer,
                                              // size_t size, unsigned long long
                                              // flags);
      tsmicroAdaptorEventElapsedTime, // flagcxResult_t
};

#endif // USE_TSM_ADAPTOR
