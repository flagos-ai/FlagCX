/*************************************************************************
 * Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd.
   All Rights Reserved.
 ************************************************************************/

#include "metax_adaptor.h"

#ifdef USE_METAX_ADAPTOR

#include "adaptor.h"
#include "alloc.h"

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

flagcxResult_t macaAdaptorHostGetDevicePointer(void **pDevice, void *pHost) {
  DEVCHECK(mcHostGetDevicePointer(pDevice, pHost, 0));
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

flagcxResult_t macaAdaptorEventElapsedTime(float *ms, flagcxEvent_t start,
                                           flagcxEvent_t end) {
  if (ms == NULL || start == NULL || end == NULL) {
    return flagcxInvalidArgument;
  }
  mcError_t error = mcEventElapsedTime(ms, start->base, end->base);
  if (error == mcSuccess) {
    return flagcxSuccess;
  } else if (error == mcErrorNotReady) {
    return flagcxInProgress;
  } else {
    return flagcxUnhandledDeviceError;
  }
}

flagcxResult_t macaAdaptorStreamWaitValue64(flagcxStream_t, void *, uint64_t,
                                            int) {
  return flagcxNotSupported;
}
flagcxResult_t macaAdaptorStreamWriteValue64(flagcxStream_t, void *, uint64_t,
                                             int) {
  return flagcxNotSupported;
}

flagcxResult_t
macaAdaptorMemGetHandleForAddressRange(void *handleOut, void *buffer,
                                       size_t size, unsigned long long flags) {
  // MCdeviceptr dptr = (MCdeviceptr)buffer;
  DEVCHECK(mcMemGetHandleForAddressRange(handleOut, buffer, size, 0x1, flags));
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorHostRegister(void *ptr, size_t size) {
  DEVCHECK(mcHostRegister(ptr, size, mcHostRegisterMapped));
  return flagcxSuccess;
}
flagcxResult_t macaAdaptorHostUnregister(void *ptr) {
  DEVCHECK(mcHostUnregister(ptr));
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorSymPhysAlloc(void *ptr, size_t size,
                                       void **physHandle, void *shareableHandle,
                                       size_t *handleSize, size_t *allocSize) {
  if (ptr == NULL || physHandle == NULL || shareableHandle == NULL ||
      handleSize == NULL || allocSize == NULL)
    return flagcxInvalidArgument;

  mcMemGenericAllocationHandle *mcHandle =
      (mcMemGenericAllocationHandle *)malloc(
          sizeof(mcMemGenericAllocationHandle));
  if (mcHandle == NULL)
    return flagcxSystemError;

  // Retain the physical allocation handle from the VMM-backed pointer
  DEVCHECK(mcMemRetainAllocationHandle(mcHandle, ptr));

  // Discover actual physical allocation size (already granularity-aligned)
  size_t actualAllocSize = 0;
  DEVCHECK(mcMemGetAddressRange(NULL, &actualAllocSize, ptr));
  *allocSize = actualAllocSize;

  // Export as POSIX fd for IPC sharing
  if (*handleSize < sizeof(int)) {
    free(mcHandle);
    return flagcxInvalidArgument;
  }
  DEVCHECK(mcMemExportToShareableHandle(shareableHandle, *mcHandle,
                                        mcMemHandleTypePosixFileDescriptor, 0));
  *handleSize = sizeof(int); // POSIX fd is an int
  *physHandle = mcHandle;
  return flagcxSuccess;
}

// flagcxResult_t macaAdaptorSymPhysFree(void *) { return flagcxNotSupported; }
flagcxResult_t macaAdaptorSymPhysFree(void *physHandle) {
  if (physHandle == NULL)
    return flagcxSuccess;
  mcMemGenericAllocationHandle *mcHandle =
      (mcMemGenericAllocationHandle *)physHandle;
  mcMemRelease(*mcHandle);
  free(mcHandle);
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorSymFlatMap(void *peerHandles[], int nPeers,
                                     int selfIndex, void *selfPhysHandle,
                                     size_t allocSize, void **flatBase) {
  if (peerHandles == NULL || selfPhysHandle == NULL || flatBase == NULL ||
      nPeers <= 0 || allocSize == 0)
    return flagcxInvalidArgument;

  mcMemGenericAllocationHandle selfHandle =
      *(mcMemGenericAllocationHandle *)selfPhysHandle;

  // allocSize is already granularity-aligned (from cuMemGetAddressRange)
  size_t totalSize = allocSize * nPeers;

  // Reserve the full VA range
  mcDeviceptr_t base = 0;
  DEVCHECK(mcMemAddressReserve(&base, totalSize, 0, 0, 0));

  // Import and map each peer's physical memory
  int macaDev;
  DEVCHECK(mcGetDevice(&macaDev));
  mcMemAccessDesc accessDesc = {};
  accessDesc.location.type = mcMemLocationTypeDevice;
  accessDesc.location.id = macaDev;
  accessDesc.flags = mcMemAccessFlagsProtReadWrite;

  for (int i = 0; i < nPeers; i++) {
    mcMemGenericAllocationHandle peerHandle;
    if (i == selfIndex) {
      peerHandle = selfHandle;
    } else {
      int fd = *(int *)peerHandles[i];
      DEVCHECK(
          mcMemImportFromShareableHandle(&peerHandle, (void *)(uintptr_t)fd,
                                         mcMemHandleTypePosixFileDescriptor));
    }
    mcDeviceptr_t slot =
        (mcDeviceptr_t)((uintptr_t)base + (uint64_t)i * allocSize);
    DEVCHECK(mcMemMap(slot, allocSize, 0, peerHandle, 0));
    DEVCHECK(mcMemSetAccess(slot, allocSize, &accessDesc, 1));
    if (i != selfIndex) {
      mcMemRelease(peerHandle);
    }
  }

  *flatBase = (void *)base;
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorSymFlatUnmap(void *flatBase, size_t allocSize,
                                       int nPeers) {
  if (flatBase == NULL)
    return flagcxSuccess;
  mcDeviceptr_t base = (mcDeviceptr_t)flatBase;
  size_t totalSize = allocSize * nPeers;
  DEVCHECK(mcMemUnmap(base, totalSize));
  DEVCHECK(mcMemAddressFree(base, totalSize));
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorSymMulticastSupported(int *supported) {
  if (supported == NULL)
    return flagcxInvalidArgument;
  *supported = 0;
  int macaDev;
  DEVCHECK(mcGetDevice(&macaDev));
  MCdevice dev;
  DEVCHECK(mcDeviceGet(&dev, macaDev));
  mcError_t res =
      mcDeviceGetAttribute(supported, mcDeviceAttributeMulticastSupported, dev);
  if (res != mcSuccess)
    *supported = 0;
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorSymMulticastCreate(size_t allocSize,
                                             int nLocalDevices,
                                             const int *localDeviceOrdinals,
                                             void **mcHandle,
                                             int *shareableFd) {
  if (mcHandle == NULL || shareableFd == NULL || nLocalDevices <= 0 ||
      localDeviceOrdinals == NULL)
    return flagcxInvalidArgument;
  *mcHandle = NULL;
  *shareableFd = -1;

  mcMemGenericAllocationHandle handle = 0;
  int fd = -1;
  mcError_t err;

  // Get multicast granularity and align size
  mcMulticastObjectProp mcProp = {};
  mcProp.numDevices = (unsigned int)nLocalDevices;
  mcProp.size = allocSize;
  mcProp.handleTypes = mcMemHandleTypePosixFileDescriptor;

  size_t mcGran = 0;
  err = mcMulticastGetGranularity(&mcGran, &mcProp,
                                  MC_MULTICAST_GRANULARITY_RECOMMENDED);
  if (err != mcSuccess)
    return flagcxUnhandledDeviceError;
  mcProp.size = ((allocSize + mcGran - 1) / mcGran) * mcGran;

  err = mcMulticastCreate(&handle, &mcProp);
  if (err != mcSuccess)
    return flagcxUnhandledDeviceError;

  // Add all local devices using explicit ordinals
  for (int i = 0; i < nLocalDevices; i++) {
    MCdevice peerDev;
    err = mcDeviceGet(&peerDev, localDeviceOrdinals[i]);
    if (err != mcSuccess)
      goto cleanup_handle;
    err = mcMulticastAddDevice(handle, peerDev);
    if (err != mcSuccess)
      goto cleanup_handle;
  }

  // Export as POSIX FD for sharing with peers
  err = mcMemExportToShareableHandle(&fd, handle,
                                     mcMemHandleTypePosixFileDescriptor, 0);
  if (err != mcSuccess)
    goto cleanup_handle;

  // Store handle as heap-allocated value
  {
    mcMemGenericAllocationHandle *handlePtr =
        (mcMemGenericAllocationHandle *)malloc(
            sizeof(mcMemGenericAllocationHandle));
    if (handlePtr == NULL)
      goto cleanup_fd;

    *handlePtr = handle;
    *mcHandle = handlePtr;
    *shareableFd = fd;
  }
  return flagcxSuccess;

cleanup_fd:
  close(fd);
cleanup_handle:
  mcMemRelease(handle);
  return flagcxUnhandledDeviceError;
}

flagcxResult_t macaAdaptorSymMulticastBind(void *mcHandle, int importFd,
                                           void *physHandle, size_t allocSize,
                                           int localRank, int nLocalDevices,
                                           void **mcBase, size_t *mcMapSize) {
  if (mcBase == NULL || physHandle == NULL || mcMapSize == NULL)
    return flagcxInvalidArgument;
  *mcBase = NULL;
  *mcMapSize = 0;

  mcMemGenericAllocationHandle mcMcHandle;
  bool imported = (mcHandle == NULL);

  if (mcHandle != NULL) {
    // Rank 0: already has the handle from symMulticastCreate
    mcMcHandle = *(mcMemGenericAllocationHandle *)mcHandle;
  } else {
    // Other ranks: import from FD
    if (importFd < 0)
      return flagcxInvalidArgument;
    mcError_t res =
        mcMemImportFromShareableHandle(&mcMcHandle, (void *)(intptr_t)importFd,
                                       mcMemHandleTypePosixFileDescriptor);
    if (res != mcSuccess) {
      WARN("symMulticastBind: cuMemImportFromShareableHandle failed: %d", res);
      return flagcxUnhandledDeviceError;
    }
  }

  mcMemGenericAllocationHandle mcPhysHandle =
      *(mcMemGenericAllocationHandle *)physHandle;

  // Bind this rank's physical allocation to the multicast object.
  // Use cuMulticastBindMem (takes physical handle), not cuMulticastBindAddr
  // (which takes a virtual address).
  mcError_t res =
      mcMulticastBindMem(mcMcHandle, 0, mcPhysHandle, 0, allocSize, 0);
  if (res != mcSuccess) {
    WARN("symMulticastBind: mcMulticastBindMem failed: %d (localRank=%d "
         "allocSize=%zu)",
         res, localRank, allocSize);
    if (imported)
      mcMemRelease(mcMcHandle);
    return flagcxUnhandledDeviceError;
  }

  // Get multicast granularity to compute aligned total size
  mcMulticastObjectProp mcProp = {};
  mcProp.numDevices = (unsigned int)nLocalDevices;
  mcProp.size = allocSize;
  mcProp.handleTypes = mcMemHandleTypePosixFileDescriptor;
  size_t mcGran = 0;
  res = mcMulticastGetGranularity(&mcGran, &mcProp,
                                  MC_MULTICAST_GRANULARITY_RECOMMENDED);
  if (res != mcSuccess) {
    WARN("symMulticastBind: mcMulticastGetGranularity failed: %d", res);
    if (imported)
      mcMemRelease(mcMcHandle);
    return flagcxUnhandledDeviceError;
  }
  size_t alignedSize = ((allocSize + mcGran - 1) / mcGran) * mcGran;

  // Reserve VA and map the multicast handle
  mcDeviceptr_t mcVa = 0;
  res = mcMemAddressReserve(&mcVa, alignedSize, mcGran, 0, 0);
  if (res != mcSuccess) {
    WARN("symMulticastBind: mcMemAddressReserve failed: %d", res);
    if (imported)
      mcMemRelease(mcMcHandle);
    return flagcxUnhandledDeviceError;
  }

  res = mcMemMap(mcVa, alignedSize, 0, mcMcHandle, 0);
  if (res != mcSuccess) {
    WARN("symMulticastBind: mcMemMap failed: %d", res);
    mcMemAddressFree(mcVa, alignedSize);
    if (imported)
      mcMemRelease(mcMcHandle);
    return flagcxUnhandledDeviceError;
  }

  // Set access for the current device
  int macaDev;
  DEVCHECK(mcGetDevice(&macaDev));
  mcMemAccessDesc accessDesc = {};
  accessDesc.location.type = mcMemLocationTypeDevice;
  accessDesc.location.id = macaDev;
  accessDesc.flags = mcMemAccessFlagsProtReadWrite;
  res = mcMemSetAccess(mcVa, alignedSize, &accessDesc, 1);
  if (res != mcSuccess) {
    WARN("symMulticastBind: cuMemSetAccess failed: %d", res);
    mcMemUnmap(mcVa, alignedSize);
    mcMemAddressFree(mcVa, alignedSize);
    if (imported)
      mcMemRelease(mcMcHandle);
    return flagcxUnhandledDeviceError;
  }

  *mcBase = (void *)mcVa;
  *mcMapSize = alignedSize;
  if (imported)
    mcMemRelease(mcMcHandle);
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorSymMulticastTeardown(void *mcBase, size_t mcMapSize) {
  if (mcBase == NULL)
    return flagcxSuccess;
  mcDeviceptr_t va = (mcDeviceptr_t)mcBase;
  DEVCHECK(mcMemUnmap(va, mcMapSize));
  DEVCHECK(mcMemAddressFree(va, mcMapSize));
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorSymMulticastFree(void *mcHandle) {
  if (mcHandle == NULL)
    return flagcxSuccess;
  mcMemGenericAllocationHandle handle =
      *(mcMemGenericAllocationHandle *)mcHandle;
  DEVCHECK(mcMemRelease(handle));
  free(mcHandle);
  return flagcxSuccess;
}

struct flagcxDeviceAdaptor macaAdaptor {
  "MACA",
      // Basic functions
      macaAdaptorDeviceSynchronize, macaAdaptorDeviceMemcpy,
      macaAdaptorDeviceMemset, macaAdaptorDeviceMalloc, macaAdaptorDeviceFree,
      macaAdaptorSetDevice, macaAdaptorGetDevice, macaAdaptorGetDeviceCount,
      macaAdaptorGetVendor, macaAdaptorHostGetDevicePointer,
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
      macaAdaptorStreamWaitValue64, macaAdaptorStreamWriteValue64,
      // Event functions
      macaAdaptorEventCreate, macaAdaptorEventDestroy, macaAdaptorEventRecord,
      macaAdaptorEventSynchronize, macaAdaptorEventQuery,
      macaAdaptorEventElapsedTime,
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
      macaAdaptorMemGetHandleForAddressRange, // flagcxResult_t
                                              // (*memGetHandleForAddressRange)(void
                                              // *handleOut, void *buffer,
                                              // size_t size, unsigned long long
                                              // flags);
      macaAdaptorHostRegister,   // flagcxResult_t (*hostRegister)(void *,
                                 // size_t);
      macaAdaptorHostUnregister, // flagcxResult_t (*hostUnregister)(void *);
      // Symmetric memory VMM functions (not supported)
      macaAdaptorSymPhysAlloc, macaAdaptorSymPhysFree, macaAdaptorSymFlatMap,
      macaAdaptorSymFlatUnmap, macaAdaptorSymMulticastSupported,
      macaAdaptorSymMulticastCreate, macaAdaptorSymMulticastBind,
      macaAdaptorSymMulticastTeardown, macaAdaptorSymMulticastFree,
};

#endif // USE_METAX_ADAPTOR
