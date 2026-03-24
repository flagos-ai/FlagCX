#include "cambricon_adaptor.h"

#ifdef USE_CAMBRICON_ADAPTOR

std::map<flagcxMemcpyType_t, cnrtMemTransDir_t> memcpy_type_map = {
    {flagcxMemcpyHostToDevice, cnrtMemcpyHostToDev},
    {flagcxMemcpyDeviceToHost, cnrtMemcpyDevToHost},
    {flagcxMemcpyDeviceToDevice, cnrtMemcpyDevToDev},
};

flagcxResult_t mluAdaptorDeviceSynchronize() {
  DEVCHECK(cnrtSyncDevice());
  return flagcxSuccess;
}

flagcxResult_t mluAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
                                      flagcxMemcpyType_t type,
                                      flagcxStream_t stream, void *args) {
  if (stream == NULL) {
    DEVCHECK(cnrtMemcpy(dst, src, size, memcpy_type_map[type]));
  } else {
    DEVCHECK(cnrtMemcpyAsync_V2(dst, src, size, stream->base,
                                memcpy_type_map[type]));
  }
  return flagcxSuccess;
}

flagcxResult_t mluAdaptorDeviceMemset(void *ptr, int value, size_t size,
                                      flagcxMemType_t type,
                                      flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    memset(ptr, value, size);
  } else {
    if (stream == NULL) {
      DEVCHECK(cnrtMemset(ptr, value, size));
    } else {
      DEVCHECK(cnrtMemsetAsync(ptr, value, size, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t mluAdaptorDeviceMalloc(void **ptr, size_t size,
                                      flagcxMemType_t type,
                                      flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    DEVCHECK(cnrtHostMalloc(ptr, size));
  } else {
    DEVCHECK(cnrtMalloc(ptr, size));
    // DEVCHECK(cnrtMallocAsync(ptr, size, stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t mluAdaptorDeviceFree(void *ptr, flagcxMemType_t type,
                                    flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    DEVCHECK(cnrtFreeHost(ptr));
  } else {
    DEVCHECK(cnrtFree(ptr));
    // DEVCHECK(cnrtFreeAsync(ptr, stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t mluAdaptorSetDevice(int dev) {
  DEVCHECK(cnrtSetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t mluAdaptorGetDevice(int *dev) {
  DEVCHECK(cnrtGetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t mluAdaptorGetDeviceCount(int *count) {
  DEVCHECK(cnrtGetDeviceCount(reinterpret_cast<unsigned int *>(count)));
  return flagcxSuccess;
}

flagcxResult_t mluAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "MLU");
  return flagcxSuccess;
}

flagcxResult_t mluAdaptorGdrMemAlloc(void **ptr, size_t size, void *memHandle) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cnrtMalloc(ptr, size));
  return flagcxSuccess;
}

flagcxResult_t mluAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return flagcxSuccess;
  }
  DEVCHECK(cnrtFree(ptr));
  return flagcxSuccess;
}

flagcxResult_t mluAdaptorStreamCreate(flagcxStream_t *stream) {
  (*stream) = NULL;
  flagcxCalloc(stream, 1);
  DEVCHECK(cnrtQueueCreate((cnrtQueue_t *)(*stream)));
  return flagcxSuccess;
}

flagcxResult_t mluAdaptorStreamDestroy(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(cnrtQueueDestroy(stream->base));
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t mluAdaptorStreamCopy(flagcxStream_t *newStream,
                                    void *oldStream) {
  (*newStream) = NULL;
  flagcxCalloc(newStream, 1);
  (*newStream)->base = (cnrtQueue_t)oldStream;
  return flagcxSuccess;
}

flagcxResult_t mluAdaptorStreamFree(flagcxStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t mluAdaptorStreamSynchronize(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(cnrtQueueSync(stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t mluAdaptorStreamQuery(flagcxStream_t stream) {
  flagcxResult_t res = flagcxSuccess;
  if (stream != NULL) {
    cnrtRet_t error = cnrtQueueQuery(stream->base);
    if (error == cnrtSuccess) {
      res = flagcxSuccess;
    } else if (error == cnrtErrorNotReady) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

flagcxResult_t mluAdaptorLaunchHostFunc(flagcxStream_t stream,
                                        void (*fn)(void *), void *args) {
  if (stream != NULL) {
    DEVCHECK(cnrtInvokeHostFunc(stream->base, fn, args));
  }
  return flagcxSuccess;
}

flagcxResult_t mluAdaptorGetDeviceProperties(struct flagcxDevProps *props,
                                             int dev) {
  if (props == NULL) {
    return flagcxInvalidArgument;
  }
  cnrtDeviceProp_t devProp;
  DEVCHECK(cnrtGetDeviceProperties(&devProp, dev));

  strncpy(props->name, devProp.name, sizeof(props->name) - 1);
  props->name[sizeof(props->name) - 1] = '\0';
  props->pciBusId = devProp.pciBusID;
  props->pciDeviceId = devProp.pciDeviceID;
  props->pciDomainId = devProp.pciDomainID;

  return flagcxSuccess;
}

flagcxResult_t mluAdaptorGetDevicePciBusId(char *pciBusId, int len, int dev) {
  if (pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cnrtDeviceGetPCIBusId(pciBusId, len, dev));
  return flagcxSuccess;
}

flagcxResult_t mluAdaptorGetDeviceByPciBusId(int *dev, const char *pciBusId) {
  if (dev == NULL || pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cnrtDeviceGetByPCIBusId(dev, pciBusId));
  return flagcxSuccess;
}

flagcxResult_t mluAdaptorStreamWaitEvent(flagcxStream_t stream,
                                         flagcxEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(cnrtQueueWaitNotifier(event->base, stream->base,
                                   CNRT_NOTIFIER_WAIT_DEFAULT);)
  }
  return flagcxSuccess;
}

flagcxResult_t mluAdaptorEventCreate(flagcxEvent_t *event,
                                     flagcxEventType_t eventType) {
  (*event) = NULL;
  flagcxCalloc(event, 1);
  const unsigned int flags = (eventType == flagcxEventDefault)
                                 ? CNRT_NOTIFIER_DEFAULT
                                 : CNRT_NOTIFIER_DISABLE_TIMING_ALL;
  DEVCHECK(cnrtNotifierCreateWithFlags(&((*event)->base), flags));
  return flagcxSuccess;
}

flagcxResult_t mluAdaptorEventDestroy(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(cnrtNotifierDestroy(event->base));
    free(event);
    event = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t mluAdaptorEventRecord(flagcxEvent_t event,
                                     flagcxStream_t stream) {
  if (event != NULL) {
    if (stream != NULL) {
      DEVCHECK(cnrtPlaceNotifierWithFlags(event->base, stream->base,
                                          CNRT_NOTIFIER_PLACE_DEFAULT));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t mluAdaptorEventSynchronize(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(cnrtWaitNotifier(event->base));
  }
  return flagcxSuccess;
}

flagcxResult_t mluAdaptorEventQuery(flagcxEvent_t event) {
  flagcxResult_t res = flagcxSuccess;
  if (event != NULL) {
    cnrtRet_t error = cnrtQueryNotifier(event->base);
    if (error == cnrtSuccess) {
      res = flagcxSuccess;
    } else if (error == cnrtErrorBusy) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

flagcxResult_t mluAdaptorIpcMemHandleCreate(flagcxIpcMemHandle_t *handle,
                                            size_t *size) {
  // to be implemented
  return flagcxNotSupported;
}

flagcxResult_t mluAdaptorIpcMemHandleGet(flagcxIpcMemHandle_t handle,
                                         void *devPtr) {
  // to be implemented
  return flagcxNotSupported;
}

flagcxResult_t mluAdaptorIpcMemHandleOpen(flagcxIpcMemHandle_t handle,
                                          void **devPtr) {
  // to be implemented
  return flagcxNotSupported;
}

flagcxResult_t mluAdaptorIpcMemHandleClose(void *devPtr) {
  // to be implemented
  return flagcxNotSupported;
}

flagcxResult_t mluAdaptorIpcMemHandleFree(flagcxIpcMemHandle_t handle) {
  // to be implemented
  return flagcxNotSupported;
}

flagcxResult_t mluAdaptorStreamWaitValue64(flagcxStream_t, void *, uint64_t,
                                           int) {
  return flagcxNotSupported;
}
flagcxResult_t mluAdaptorStreamWriteValue64(flagcxStream_t, void *, uint64_t,
                                            int) {
  return flagcxNotSupported;
}
flagcxResult_t mluAdaptorEventElapsedTime(float *, flagcxEvent_t,
                                          flagcxEvent_t) {
  return flagcxNotSupported;
}

struct flagcxDeviceAdaptor mluAdaptor {
  "MLU",
      // Basic functions
      mluAdaptorDeviceSynchronize, mluAdaptorDeviceMemcpy,
      mluAdaptorDeviceMemset, mluAdaptorDeviceMalloc, mluAdaptorDeviceFree,
      mluAdaptorSetDevice, mluAdaptorGetDevice, mluAdaptorGetDeviceCount,
      mluAdaptorGetVendor, NULL,
      // GDR functions
      NULL, // flagcxResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // flagcxResult_t (*memHandleDestroy)(int dev, void *memHandle);
      mluAdaptorGdrMemAlloc, mluAdaptorGdrMemFree,
      NULL, // flagcxResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // flagcxResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      NULL, // flagcxResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t
            // sz);
      NULL, // flagcxResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);
      // Stream functions
      mluAdaptorStreamCreate, mluAdaptorStreamDestroy, mluAdaptorStreamCopy,
      mluAdaptorStreamFree, mluAdaptorStreamSynchronize, mluAdaptorStreamQuery,
      mluAdaptorStreamWaitEvent, mluAdaptorStreamWaitValue64,
      mluAdaptorStreamWriteValue64,
      // Event functions
      mluAdaptorEventCreate, mluAdaptorEventDestroy, mluAdaptorEventRecord,
      mluAdaptorEventSynchronize, mluAdaptorEventQuery,
      mluAdaptorEventElapsedTime,
      // IpcMemHandle functions
      mluAdaptorIpcMemHandleCreate, mluAdaptorIpcMemHandleGet,
      mluAdaptorIpcMemHandleOpen, mluAdaptorIpcMemHandleClose,
      mluAdaptorIpcMemHandleFree,
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
      mluAdaptorGetDeviceProperties, // flagcxResult_t
                                     // (*getDeviceProperties)(struct
                                     // flagcxDevProps *props, int dev);
      mluAdaptorGetDevicePciBusId,   // flagcxResult_t (*getDevicePciBusId)(char
                                     // *pciBusId, int len, int dev);
      mluAdaptorGetDeviceByPciBusId, // flagcxResult_t
                                     // (*getDeviceByPciBusId)(int
                                     // *dev, const char *pciBusId);
      mluAdaptorLaunchHostFunc,
      // DMA buffer
      NULL, // flagcxResult_t (*dmaSupport)(bool *dmaBufferSupport);
      NULL, // flagcxResult_t (*memGetHandleForAddressRange)(void *handleOut,
            // void *buffer, size_t size, unsigned long long flags);
};

#endif // USE_CAMBRICON_ADAPTOR
