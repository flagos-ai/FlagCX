#ifdef USE_KUNLUNXIN_ADAPTOR

#include "kunlunxin_adaptor.h"

#include "adaptor.h"
#include "alloc.h"

#include <xpu/runtime.h>

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

/*
 * GdrMem lifecycle:
 *   memHandleInit
 *   gdrMemAlloc -> xpu_malloc + xccl_mmap
 *   caller registers hostMappedPtr and completes communication
 *   caller deregisters the MR
 *   gdrMemFree -> xccl_munmap + xpu_free
 *   memHandleDestroy
 *
 * memHandle is mandatory. It is passed by value, so gdrMemAlloc cannot safely
 * create and return an implicit handle when the argument is null.
 *
 * Production upper-layer wiring is outside this implementation: the upper
 * layer must own and propagate the same handle through allocation and cleanup.
 */

flagcxResult_t kunlunAdaptorMemHandleInit(int dev_id, void **memHandle) {
  (void)dev_id;
  if (memHandle == NULL) return flagcxInvalidArgument;
  auto *h = new (std::nothrow) KunlunXinGdrMemHandle{};
  if (h == NULL) return flagcxSystemError;
  *memHandle = h;
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorMemHandleDestroy(int dev, void *memHandle) {
  (void)dev;
  if (memHandle == NULL) return flagcxSuccess;
  auto *h = static_cast<KunlunXinGdrMemHandle *>(memHandle);
  // Refuse to destroy a handle that still has a live mapping or unreleased device pointer.
  if (h->mapped || h->devPtr != NULL) {
    return flagcxInvalidArgument;
  }
  delete h;
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorGdrMemAlloc(void **ptr, size_t size,
                                        void *memHandle) {
  if (ptr == NULL || size == 0 || memHandle == NULL) {
    return flagcxInvalidArgument;
  }

  auto *handle = static_cast<KunlunXinGdrMemHandle *>(memHandle);
  // Reject if handle already holds a live mapping.
  if (handle->mapped || handle->devPtr != NULL) {
    return flagcxInvalidArgument;
  }

  // Step 1: allocate device memory.
  void *devPtr = NULL;
  int ret = xpu_malloc(&devPtr, size);
  if (ret != 0 || devPtr == NULL) {
    return flagcxUnhandledDeviceError;
  }

  // Step 2: map device VA to host VA (requires BKCL_USE_PEERMEM_XDR=1).
  void *hostPtr = NULL;
  ret = baidu::xpu::bkcl::xccl_mmap(&hostPtr, devPtr, size);
  if (ret != 0 || hostPtr == NULL) {
    xpu_free(devPtr);
    return flagcxUnhandledDeviceError;
  }

  handle->devPtr        = devPtr;
  handle->hostMappedPtr = hostPtr;
  handle->size          = size;
  handle->mapped        = true;
  *ptr = devPtr;
  return flagcxSuccess;
}

flagcxResult_t kunlunAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return flagcxSuccess;
  }
  if (memHandle == NULL) {
    return flagcxInvalidArgument;
  }

  auto *handle = static_cast<KunlunXinGdrMemHandle *>(memHandle);

  // Sanity check: ptr must match what we allocated.
  if (handle->devPtr != ptr) {
    return flagcxInvalidArgument;
  }

  // Step 1: unmap host mapping — must pass hostMappedPtr, not devPtr.
  if (handle->mapped && handle->hostMappedPtr != NULL) {
    int ret = baidu::xpu::bkcl::xccl_munmap(handle->hostMappedPtr,
                                             handle->size);
    if (ret != 0) return flagcxUnhandledDeviceError;
  }

  // Step 2: free device memory.
  int ret = xpu_free(ptr);
  if (ret != 0) return flagcxUnhandledDeviceError;

  // Clear all handle fields.
  *handle = {};
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

flagcxResult_t kunlunAdaptorStreamWaitValue64(flagcxStream_t, void *, uint64_t,
                                              int) {
  return flagcxNotSupported;
}
flagcxResult_t kunlunAdaptorStreamWriteValue64(flagcxStream_t, void *, uint64_t,
                                               int) {
  return flagcxNotSupported;
}
flagcxResult_t kunlunAdaptorEventElapsedTime(float *, flagcxEvent_t,
                                             flagcxEvent_t) {
  return flagcxNotSupported;
}

flagcxResult_t kunlunAdaptorHostRegister(void *, size_t) {
  return flagcxNotSupported;
}
flagcxResult_t kunlunAdaptorHostUnregister(void *) {
  return flagcxNotSupported;
}

// Symmetric memory VMM stubs (not supported)
flagcxResult_t kunlunxinAdaptorSymPhysAlloc(void *, size_t, void **, void *,
                                            size_t *, size_t *) {
  return flagcxNotSupported;
}
flagcxResult_t kunlunxinAdaptorSymPhysFree(void *) {
  return flagcxNotSupported;
}
flagcxResult_t kunlunxinAdaptorSymFlatMap(void *[], int, int, void *, size_t,
                                          void **) {
  return flagcxNotSupported;
}
flagcxResult_t kunlunxinAdaptorSymFlatUnmap(void *, size_t, int) {
  return flagcxNotSupported;
}
flagcxResult_t kunlunxinAdaptorSymMulticastSupported(int *supported) {
  if (supported)
    *supported = 0;
  return flagcxSuccess;
}
flagcxResult_t kunlunxinAdaptorSymMulticastCreate(size_t, int, const int *,
                                                  void **, int *) {
  return flagcxNotSupported;
}
flagcxResult_t kunlunxinAdaptorSymMulticastBind(void *, int, void *, size_t,
                                                int, int, void **, size_t *) {
  return flagcxNotSupported;
}
flagcxResult_t kunlunxinAdaptorSymMulticastTeardown(void *, size_t) {
  return flagcxSuccess;
}
flagcxResult_t kunlunxinAdaptorSymMulticastFree(void *) {
  return flagcxNotSupported;
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
      kunlunAdaptorMemHandleInit,    // flagcxResult_t (*memHandleInit)(int dev_id, void **memHandle);
      kunlunAdaptorMemHandleDestroy, // flagcxResult_t (*memHandleDestroy)(int dev, void *memHandle);
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
      kunlunAdaptorStreamWaitEvent, kunlunAdaptorStreamWaitValue64,
      kunlunAdaptorStreamWriteValue64,
      // Event functions
      kunlunAdaptorEventCreate, kunlunAdaptorEventDestroy,
      kunlunAdaptorEventRecord, kunlunAdaptorEventSynchronize,
      kunlunAdaptorEventQuery, kunlunAdaptorEventElapsedTime,
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
      kunlunAdaptorHostRegister,   // flagcxResult_t (*hostRegister)(void *,
                                   // size_t);
      kunlunAdaptorHostUnregister, // flagcxResult_t (*hostUnregister)(void *);
      // Symmetric memory VMM functions (not supported)
      kunlunxinAdaptorSymPhysAlloc, kunlunxinAdaptorSymPhysFree,
      kunlunxinAdaptorSymFlatMap, kunlunxinAdaptorSymFlatUnmap,
      kunlunxinAdaptorSymMulticastSupported, kunlunxinAdaptorSymMulticastCreate,
      kunlunxinAdaptorSymMulticastBind, kunlunxinAdaptorSymMulticastTeardown,
      kunlunxinAdaptorSymMulticastFree,
};
#endif // USE_KUNLUNXIN_ADAPTOR
