#include "ppu_adaptor.h"

#ifdef USE_PPU_ADAPTOR

#include "adaptor.h"
#include "hggc_sdk_features.h"
#include "hggc_vmm.h"
#include "param.h"
#include <stdio.h>
#include <unistd.h>

flagcxResult_t hggcAdaptorDeviceSynchronize() {
  HGGCCHECK(hggcDeviceSynchronize());
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
                                       flagcxMemcpyType_t type,
                                       flagcxStream_t stream, void *args) {
  enum hggcMemcpyKind kind;
  switch (type) {
    case flagcxMemcpyHostToDevice:
      kind = hggcMemcpyHostToDevice;
      break;
    case flagcxMemcpyDeviceToHost:
      kind = hggcMemcpyDeviceToHost;
      break;
    case flagcxMemcpyDeviceToDevice:
      kind = hggcMemcpyDeviceToDevice;
      break;
    default:
      return flagcxInvalidArgument;
  }
  if (stream == NULL) {
    HGGCCHECK(hggcMemcpy(dst, src, size, kind));
  } else {
    HGGCCHECK(hggcMemcpyAsync(dst, src, size, kind, stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorDeviceMemset(void *ptr, int value, size_t size,
                                       flagcxMemType_t type,
                                       flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    memset(ptr, value, size);
  } else {
    if (stream == NULL) {
      HGGCCHECK(hggcMemset(ptr, value, size));
    } else {
      HGGCCHECK(hggcMemsetAsync(ptr, value, size, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorDeviceMalloc(void **ptr, size_t size,
                                       flagcxMemType_t type,
                                       flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    HGGCCHECK(hggcHostAlloc(ptr, size, hggcHostAllocMapped));
  } else if (type == flagcxMemManaged) {
#if FLAGCX_HGGC_HAS_MANAGED
    HGGCCHECK(hggcMallocManaged(ptr, size, hggcMemAttachGlobal));
#else
    return flagcxNotSupported;
#endif
  } else {
    if (stream == NULL) {
      HGGCCHECK(hggcMalloc(ptr, size));
    } else {
      HGGCCHECK(hggcMallocAsync(ptr, size, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorDeviceFree(void *ptr, flagcxMemType_t type,
                                     flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    HGGCCHECK(hggcFreeHost(ptr));
  } else if (type == flagcxMemManaged) {
    HGGCCHECK(hggcFree(ptr));
  } else {
    if (stream == NULL) {
      HGGCCHECK(hggcFree(ptr));
    } else {
      HGGCCHECK(hggcFreeAsync(ptr, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorSetDevice(int dev) {
  HGGCCHECK(hggcSetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorGetDevice(int *dev) {
  HGGCCHECK(hggcGetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorGetDeviceCount(int *count) {
  HGGCCHECK(hggcGetDeviceCount(count));
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "PPU");
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorCanAccessPeer(int srcDev, int dstDev,
                                        int *canAccess) {
  if (canAccess == NULL)
    return flagcxInvalidArgument;
  HGGCCHECK(hggcDeviceCanAccessPeer(canAccess, srcDev, dstDev));
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorHostGetDevicePointer(void **pDevice, void *pHost) {
  if (pDevice == NULL || pHost == NULL) {
    return flagcxInvalidArgument;
  }
  HGGCCHECK(hggcHostGetDevicePointer(pDevice, pHost, 0));
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorGdrMemAlloc(void **ptr, size_t size,
                                      void *memHandle) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
  // Native HGGC capability is independent of NVIDIA toolkit version numbers.
  if (!flagcxParamVmmEnable()) {
    HGGCCHECK(hggcMalloc(ptr, size));
    hggcPointerAttributes attrs;
    HGGCCHECK(hggcPointerGetAttributes(&attrs, *ptr));
    unsigned flags = 1;
    HGDRVCHECK(hgPointerSetAttribute(&flags, HG_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                     (HGdeviceptr)attrs.devicePointer));
    return flagcxSuccess;
  }

  *ptr = NULL;
  size_t memGran = 0, segmentBytes = 0, handleSize;
  HGdevice currentDev;
  HGmemAllocationProp memprop;
  memset(&memprop, 0, sizeof(memprop));
  int hggcDev;
  HGGCCHECK(hggcGetDevice(&hggcDev));
  FLAGCXCHECK(hggcVmmRetryPending(hggcDev));
  HGDRVCHECK(hgDeviceGet(&currentDev, hggcDev));
  memprop.type = HG_MEM_ALLOCATION_TYPE_PINNED;
  memprop.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
  memprop.requestedHandleTypes = HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  memprop.location.id = currentDev;
#if FLAGCX_HGGC_HAS_VMM_RDMA
  int rdmaSupported = 0;
  if (hgDeviceGetAttribute(&rdmaSupported,
                           HG_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED,
                           currentDev) == HGGC_SUCCESS &&
      rdmaSupported)
    memprop.allocFlags.gpuDirectRDMACapable = 1;
#endif
  HGDRVCHECK(hgMemGetAllocationGranularity(
      &memGran, &memprop, HG_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  if (memGran == 0 || size == 0 || size > SIZE_MAX - (memGran - 1))
    return flagcxInvalidArgument;
  handleSize = ((size + memGran - 1) / memGran) * memGran;
  FLAGCXCHECK(hggcVmmSegmentBytes(memGran, &segmentBytes));
  if (segmentBytes != 0 && segmentBytes < handleSize) {
    int enabled;
    FLAGCXCHECK(hggcVmmMultiSegmentEnabled(&enabled));
    if (!enabled)
      return flagcxNotSupported;
  } else {
    segmentBytes = handleSize;
  }
  return hggcVmmAllocate(ptr, handleSize, memGran, segmentBytes, &memprop,
                         hggcDev);
}

flagcxResult_t hggcAdaptorGdrMemFree(void *ptr, void *memHandle) {
  int found = 0;
  if (ptr == NULL)
    return flagcxSuccess;
  FLAGCXCHECK(hggcVmmFreeAllocation(ptr, &found));
  if (!found)
    HGGCCHECK(hggcFree(ptr));
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorStreamCreate(flagcxStream_t *stream) {
  (*stream) = NULL;
  *stream = (flagcxStream_t)calloc(1, sizeof(**stream));
  if (*stream == NULL)
    return flagcxSystemError;
  const hggcError_t result =
      hggcStreamCreateWithFlags(&(*stream)->base, hggcStreamNonBlocking);
  if (result != hggcSuccess) {
    free(*stream);
    *stream = NULL;
    return hggcResultToFlagcx(result);
  }
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorStreamDestroy(flagcxStream_t stream) {
  if (stream != NULL) {
    HGGCCHECK(hggcStreamDestroy(stream->base));
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorStreamCopy(flagcxStream_t *newStream,
                                     void *oldStream) {
  (*newStream) = NULL;
  *newStream = (flagcxStream_t)calloc(1, sizeof(**newStream));
  if (*newStream == NULL)
    return flagcxSystemError;
  (*newStream)->base = (hggcStream_t)oldStream;
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorStreamFree(flagcxStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorStreamSynchronize(flagcxStream_t stream) {
  if (stream != NULL) {
    HGGCCHECK(hggcStreamSynchronize(stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorStreamQuery(flagcxStream_t stream) {
  flagcxResult_t res = flagcxSuccess;
  if (stream != NULL) {
    hggcError_t error = hggcStreamQuery(stream->base);
    if (error == hggcSuccess) {
      res = flagcxSuccess;
    } else if (error == hggcErrorNotReady) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

flagcxResult_t hggcAdaptorStreamWaitEvent(flagcxStream_t stream,
                                          flagcxEvent_t event) {
  if (stream != NULL && event != NULL) {
    HGGCCHECK(hggcStreamWaitEvent(stream->base, event->base, 0));
  }
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorStreamWaitValue64(flagcxStream_t stream, void *addr,
                                            uint64_t value, int flags) {
  if (stream == NULL || addr == NULL)
    return flagcxInvalidArgument;
  if (flags & ~FLAGCX_STREAM_WAIT_VALUE_FLUSH_REMOTE_WRITES)
    return flagcxInvalidArgument;

#if FLAGCX_HGGC_HAS_WAIT_VALUE64
  unsigned int waitFlags = HG_STREAM_WAIT_VALUE_GEQ;
  if (flags & FLAGCX_STREAM_WAIT_VALUE_FLUSH_REMOTE_WRITES) {
#if FLAGCX_HGGC_HAS_WAIT_VALUE64_FLUSH
    waitFlags |= HG_STREAM_WAIT_VALUE_FLUSH;
#else
    return flagcxNotSupported;
#endif
  }
  return hggcDriverResultToFlagcx(hgStreamWaitValue64(
      (HGstream)(stream->base), (HGdeviceptr)addr, value, waitFlags));
#else
  return flagcxNotSupported;
#endif
}

flagcxResult_t hggcAdaptorStreamWriteValue64(flagcxStream_t stream, void *addr,
                                             uint64_t value, int flags) {
  (void)flags;
  if (stream == NULL || addr == NULL)
    return flagcxInvalidArgument;
#if FLAGCX_HGGC_HAS_WRITE_VALUE64
  return hggcDriverResultToFlagcx(
      hgStreamWriteValue64((HGstream)(stream->base), (HGdeviceptr)addr, value,
                           HG_STREAM_WRITE_VALUE_DEFAULT));
#else
  return flagcxNotSupported;
#endif
}

flagcxResult_t hggcAdaptorEventCreate(flagcxEvent_t *event,
                                      flagcxEventType_t eventType) {
  if (event == NULL)
    return flagcxInvalidArgument;
  if (event == NULL)
    return flagcxInvalidArgument;
  (*event) = NULL;
  *event = (flagcxEvent_t)calloc(1, sizeof(**event));
  if (*event == NULL)
    return flagcxSystemError;
  const unsigned int flags = (eventType == flagcxEventDefault)
                                 ? hggcEventDefault
                                 : hggcEventDisableTiming;
  const hggcError_t result = hggcEventCreateWithFlags(&(*event)->base, flags);
  if (result != hggcSuccess) {
    free(*event);
    *event = NULL;
    return hggcResultToFlagcx(result);
  }
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorEventDestroy(flagcxEvent_t event) {
  if (event != NULL) {
    HGGCCHECK(hggcEventDestroy(event->base));
    free(event);
    event = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorEventRecord(flagcxEvent_t event,
                                      flagcxStream_t stream) {
  if (event != NULL) {
    if (stream != NULL) {
      HGGCCHECK(hggcEventRecord(event->base, stream->base));
    } else {
      HGGCCHECK(hggcEventRecord(event->base, NULL));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorEventSynchronize(flagcxEvent_t event) {
  if (event != NULL) {
    HGGCCHECK(hggcEventSynchronize(event->base));
  }
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorEventQuery(flagcxEvent_t event) {
  flagcxResult_t res = flagcxSuccess;
  if (event != NULL) {
    hggcError_t error = hggcEventQuery(event->base);
    if (error == hggcSuccess) {
      res = flagcxSuccess;
    } else if (error == hggcErrorNotReady) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

flagcxResult_t hggcAdaptorEventElapsedTime(float *ms, flagcxEvent_t start,
                                           flagcxEvent_t end) {
  if (ms == NULL || start == NULL || end == NULL) {
    return flagcxInvalidArgument;
  }
  hggcError_t error = hggcEventElapsedTime(ms, start->base, end->base);
  if (error == hggcSuccess) {
    return flagcxSuccess;
  } else if (error == hggcErrorNotReady) {
    return flagcxInProgress;
  } else {
    return flagcxUnhandledDeviceError;
  }
}

flagcxResult_t hggcAdaptorIpcMemHandleCreate(flagcxIpcMemHandle_t *handle,
                                             size_t *size) {
  *handle = (flagcxIpcMemHandle_t)calloc(1, sizeof(**handle));
  if (*handle == NULL)
    return flagcxSystemError;
  if (size != NULL) {
    *size = sizeof(hggcIpcMemHandle_t);
  }
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorIpcMemHandleGet(flagcxIpcMemHandle_t handle,
                                          void *devPtr) {
  if (handle == NULL || devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  HGGCCHECK(hggcIpcGetMemHandle(&handle->base, devPtr));
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorIpcMemHandleOpen(flagcxIpcMemHandle_t handle,
                                           void **devPtr) {
  if (handle == NULL || devPtr == NULL || *devPtr != NULL) {
    return flagcxInvalidArgument;
  }
  HGGCCHECK(hggcIpcOpenMemHandle(devPtr, handle->base,
                                 hggcIpcMemLazyEnablePeerAccess));
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorIpcMemHandleClose(void *devPtr) {
  if (devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  HGGCCHECK(hggcIpcCloseMemHandle(devPtr));
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorIpcMemHandleFree(flagcxIpcMemHandle_t handle) {
  if (handle != NULL) {
    free(handle);
  }
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorLaunchHostFunc(flagcxStream_t stream,
                                         void (*fn)(void *), void *args) {
  if (stream != NULL) {
    HGGCCHECK(hggcLaunchHostFunc(stream->base, fn, args));
  }
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorDmaSupport(bool *dmaBufferSupport) {
  if (dmaBufferSupport == NULL)
    return flagcxInvalidArgument;

  *dmaBufferSupport = false;
#if FLAGCX_HGGC_HAS_DMA_BUF
  int ordinal = 0, supported = 0;
  HGdevice device;
  HGGCCHECK(hggcGetDevice(&ordinal));
  HGDRVCHECK(hgDeviceGet(&device, ordinal));
  const HGresult result = hgDeviceGetAttribute(
      &supported, HG_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, device);
  // An unavailable attribute is a capability miss, not a CUDA version check.
  if (result == HGGC_SUCCESS)
    *dmaBufferSupport = supported != 0;
#endif
  return flagcxSuccess;
}

flagcxResult_t
hggcAdaptorMemGetHandleForAddressRange(void *handleOut, void *buffer,
                                       size_t size, unsigned long long flags) {
  if (handleOut == NULL || buffer == NULL || size == 0)
    return flagcxInvalidArgument;
#if FLAGCX_HGGC_HAS_DMA_BUF
  HGresult result =
      hgMemGetHandleForAddressRange(handleOut, (HGdeviceptr)buffer, size,
                                    HG_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, flags);
  if (result == HGGC_ERROR_NOT_SUPPORTED || result == HGGC_ERROR_INVALID_VALUE)
    return flagcxNotSupported;
  return hggcDriverResultToFlagcx(result);
#else
  return flagcxNotSupported;
#endif
}

flagcxResult_t hggcAdaptorGetDeviceProperties(struct flagcxDevProps *props,
                                              int dev) {
  if (props == NULL) {
    return flagcxInvalidArgument;
  }

  hggcDeviceProp devProp;
  HGGCCHECK(hggcGetDeviceProperties(&devProp, dev));
  strncpy(props->name, devProp.name, sizeof(props->name) - 1);
  props->name[sizeof(props->name) - 1] = '\0';
  char pci[32] = {0};
  HGGCCHECK(hggcDeviceGetPCIBusId(pci, sizeof(pci), dev));
  unsigned domain = 0, bus = 0, device = 0, function = 0;
  if (sscanf(pci, "%x:%x:%x.%x", &domain, &bus, &device, &function) != 4)
    return flagcxUnhandledDeviceError;
  props->pciBusId = bus;
  props->pciDeviceId = device;
  props->pciDomainId = domain;

  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorGetDevicePciBusId(char *pciBusId, int len, int dev) {
  if (pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  HGGCCHECK(hggcDeviceGetPCIBusId(pciBusId, len, dev));
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorGetDeviceByPciBusId(int *dev, const char *pciBusId) {
  if (dev == NULL || pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  HGGCCHECK(hggcDeviceGetByPCIBusId(dev, pciBusId));
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorHostRegister(void *ptr, size_t size) {
  if (ptr == NULL || size == 0) {
    return flagcxInvalidArgument;
  }
  HGGCCHECK(hggcHostRegister(ptr, size, hggcHostRegisterMapped));
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorHostUnregister(void *ptr) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
  HGGCCHECK(hggcHostUnregister(ptr));
  return flagcxSuccess;
}

// Symmetric memory VMM — handle export/import and flat mapping
flagcxResult_t hggcAdaptorSymPhysAlloc(void *ptr, size_t size,
                                       void **physHandle, void *shareableHandle,
                                       size_t *handleSize, size_t *allocSize) {
  if (ptr == NULL || physHandle == NULL || shareableHandle == NULL ||
      handleSize == NULL || allocSize == NULL)
    return flagcxInvalidArgument;

  HGmemGenericAllocationHandle *hgHandle =
      (HGmemGenericAllocationHandle *)malloc(
          sizeof(HGmemGenericAllocationHandle));
  if (hgHandle == NULL)
    return flagcxSystemError;

  HGresult result = hgMemRetainAllocationHandle(hgHandle, ptr);
  if (result != HGGC_SUCCESS) {
    free(hgHandle);
    if (result == HGGC_ERROR_INVALID_VALUE ||
        result == HGGC_ERROR_NOT_SUPPORTED)
      return flagcxNotSupported;
    return flagcxUnhandledDeviceError;
  }

  HGdeviceptr allocationBase = 0;
  size_t actualAllocSize = 0;
  result =
      hgMemGetAddressRange(&allocationBase, &actualAllocSize, (HGdeviceptr)ptr);
  if (result != HGGC_SUCCESS) {
    hgMemRelease(*hgHandle);
    free(hgHandle);
    return flagcxUnhandledDeviceError;
  }
  const HGdeviceptr address = (HGdeviceptr)ptr;
  if (allocationBase == 0 || actualAllocSize == 0 || address < allocationBase ||
      address - allocationBase > actualAllocSize ||
      size > actualAllocSize - (address - allocationBase)) {
    hgMemRelease(*hgHandle);
    free(hgHandle);
    return flagcxInvalidUsage;
  }
  *allocSize = actualAllocSize;

  if (*handleSize < sizeof(int)) {
    hgMemRelease(*hgHandle);
    free(hgHandle);
    return flagcxInvalidArgument;
  }
  result = hgMemExportToShareableHandle(
      shareableHandle, *hgHandle, HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
  if (result != HGGC_SUCCESS) {
    hgMemRelease(*hgHandle);
    free(hgHandle);
    return flagcxUnhandledDeviceError;
  }
  *handleSize = sizeof(int);
  *physHandle = hgHandle;
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorSymPhysFree(void *physHandle) {
  if (physHandle == NULL)
    return flagcxSuccess;
  HGmemGenericAllocationHandle *hgHandle =
      (HGmemGenericAllocationHandle *)physHandle;
  if (hgMemRelease(*hgHandle) != HGGC_SUCCESS)
    return flagcxUnhandledDeviceError;
  free(hgHandle);
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorSymPhysDescribe(void *ptr, size_t size,
                                          void **segmentBundle,
                                          const flagcxSymSegmentMeta **segments,
                                          int *segmentCount, size_t *userOffset,
                                          size_t *slotAlignment) {
  if (ptr == NULL || size == 0 || segmentBundle == NULL || segments == NULL ||
      segmentCount == NULL || userOffset == NULL || slotAlignment == NULL)
    return flagcxInvalidArgument;
  *segmentBundle = NULL;
  *segments = NULL;
  *segmentCount = 0;
  *slotAlignment = 0;
  const uintptr_t address = (uintptr_t)ptr;
  if (size > UINTPTR_MAX - address)
    return flagcxInvalidArgument;

  struct hggcVmmBundle *bundle =
      (struct hggcVmmBundle *)calloc(1, sizeof(*bundle));
  if (bundle == NULL)
    return flagcxSystemError;
  *segmentBundle = bundle;
  int found = 0;
  FLAGCXCHECK(hggcVmmDescribe(ptr, size, bundle, userOffset, &found));
  if (!found) {
    // Only a successful legacy IPC export proves that an untracked pointer
    // may use that path. External VMM offsets are unknown, so no retain or
    // remapping is safe for any other untracked pointer.
    HGdeviceptr allocationBase = 0;
    size_t allocationSize = 0;
    if (hgMemGetAddressRange(&allocationBase, &allocationSize,
                             (HGdeviceptr)ptr) == HGGC_SUCCESS &&
        address >= allocationBase &&
        address - allocationBase <= allocationSize &&
        size <= allocationSize - (address - allocationBase)) {
      hggcIpcMemHandle_t ipcHandle;
      if (hggcIpcGetMemHandle(&ipcHandle, (void *)allocationBase) ==
          hggcSuccess) {
        hggcVmmFreeBundle(bundle);
        *segmentBundle = NULL;
      } else {
        hggcGetLastError();
      }
    }
    // Keep an empty bundle for an unknown mapping to block IPC fallback.
    return flagcxNotSupported;
  }
  if (bundle->count > 1) {
    int multiSegmentEnabled = 1;
    FLAGCXCHECK(hggcVmmMultiSegmentEnabled(&multiSegmentEnabled));
    if (!multiSegmentEnabled)
      return flagcxNotSupported;
  }
  // Ordinary allocations have already returned without requiring VMM support.
  int device = 0;
  HGdevice hgDevice;
  if (hggcGetDevice(&device) != hggcSuccess ||
      hgDeviceGet(&hgDevice, device) != HGGC_SUCCESS)
    return flagcxUnhandledDeviceError;
  HGmemAllocationProp prop;
  memset(&prop, 0, sizeof(prop));
  prop.type = HG_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = hgDevice;
  prop.requestedHandleTypes = HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  if (hgMemGetAllocationGranularity(slotAlignment, &prop,
                                    HG_MEM_ALLOC_GRANULARITY_RECOMMENDED) !=
          HGGC_SUCCESS ||
      *slotAlignment == 0)
    return flagcxUnhandledDeviceError;
  if (bundle->count == 0 || bundle->count > INT_MAX)
    return flagcxInvalidUsage;
  *segments = bundle->segments;
  *segmentCount = (int)bundle->count;
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorSymPhysExportSegment(void *bundle, int index,
                                               int *fd) {
  return hggcVmmExportSegment(bundle, index, fd);
}
flagcxResult_t hggcAdaptorSymPhysFreeSegments(void *bundle) {
  return hggcVmmFreeBundle(bundle);
}
flagcxResult_t hggcAdaptorSymPhysSegmentCount(const void *ptr, size_t size,
                                              int *count) {
  return hggcVmmSegmentCount(ptr, size, count);
}
flagcxResult_t hggcAdaptorSymFlatMapSegments(
    const int *fds, int nPeers, int self, void *bundle,
    const flagcxSymRankLayout *layouts, const flagcxSymSegmentMeta *segments,
    int totalSegments, size_t stride, size_t alignment, void **flatBase) {
  return hggcVmmFlatMap(fds, nPeers, self, bundle, layouts, segments,
                        totalSegments, stride, alignment, flatBase);
}
flagcxResult_t hggcAdaptorSymFlatMapSegmentsBegin(
    int nPeers, int self, void *bundle, const flagcxSymRankLayout *layouts,
    const flagcxSymSegmentMeta *segments, int totalSegments, size_t stride,
    size_t alignment, void **flatBase) {
  return hggcVmmFlatBegin(NULL, 1, nPeers, self, bundle, layouts, segments,
                          totalSegments, stride, alignment, flatBase);
}
flagcxResult_t hggcAdaptorSymFlatMapSegmentOne(void *flatBase, int index,
                                               int fd) {
  return hggcVmmFlatMapOne(flatBase, index, fd);
}
flagcxResult_t hggcAdaptorSymFlatMappingUnmapSegments(void *flatBase) {
  return hggcVmmFlatUnmap(flatBase);
}
flagcxResult_t hggcAdaptorSymFlatVaFreeSegments(void *flatBase) {
  return hggcVmmFlatFreeVa(flatBase);
}

flagcxResult_t hggcAdaptorSymLsaGetCapacity(size_t *totalBytes,
                                            size_t *alignment) {
  if (totalBytes == NULL || alignment == NULL)
    return flagcxInvalidArgument;
  size_t freeBytes = 0;
  if (hggcMemGetInfo(&freeBytes, totalBytes) != hggcSuccess)
    return flagcxUnhandledDeviceError;
  *alignment = 512ULL * 1024 * 1024;
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorSymLsaVaReserve(size_t size, size_t alignment,
                                          void **base) {
  if (base == NULL)
    return flagcxInvalidArgument;
  uintptr_t address = 0;
  flagcxResult_t result = hggcVmmReserveAddress(size, alignment, &address);
  *base = result == flagcxSuccess ? (void *)(address) : NULL;
  return result;
}

flagcxResult_t hggcAdaptorSymLsaVaFree(void *base, size_t size) {
  return hggcVmmFreeAddress((uintptr_t)(base), size);
}

flagcxResult_t hggcAdaptorSymLsaMapBeginAt(
    void *windowBase, int nPeers, int self, void *bundle,
    const flagcxSymRankLayout *layouts, const flagcxSymSegmentMeta *segments,
    int totalSegments, size_t stride, size_t alignment, void **mappingBase) {
  return hggcVmmFlatBegin(windowBase, 0, nPeers, self, bundle, layouts,
                          segments, totalSegments, stride, alignment,
                          mappingBase);
}

flagcxResult_t hggcAdaptorSymFlatMap(void *peerHandles[], int nPeers,
                                     int selfIndex, void *selfPhysHandle,
                                     size_t allocSize, void **flatBase) {
  if (peerHandles == NULL || selfPhysHandle == NULL || flatBase == NULL ||
      nPeers <= 0 || allocSize == 0)
    return flagcxInvalidArgument;

  HGmemGenericAllocationHandle selfHandle =
      *(HGmemGenericAllocationHandle *)selfPhysHandle;

  size_t totalSize = allocSize * nPeers;

  HGdeviceptr base = 0;
  HGresult result = hgMemAddressReserve(&base, totalSize, 0, 0, 0);
  if (result != HGGC_SUCCESS)
    return flagcxUnhandledDeviceError;

  int hggcDev;
  if (hggcGetDevice(&hggcDev) != hggcSuccess) {
    hgMemAddressFree(base, totalSize);
    return flagcxUnhandledDeviceError;
  }
  HGmemAccessDesc accessDesc;
  memset(&accessDesc, 0, sizeof(accessDesc));
  accessDesc.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = hggcDev;
  accessDesc.flags = HG_MEM_ACCESS_FLAGS_PROT_READWRITE;

  int mappedPeers = 0;
  for (int i = 0; i < nPeers; i++) {
    HGmemGenericAllocationHandle peerHandle;
    bool imported = i != selfIndex;
    if (i == selfIndex) {
      peerHandle = selfHandle;
    } else {
      int fd = *(int *)peerHandles[i];
      result = hgMemImportFromShareableHandle(
          &peerHandle, (void *)(uintptr_t)fd,
          HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
      if (result != HGGC_SUCCESS)
        goto rollback;
    }
    HGdeviceptr slot = base + (HGdeviceptr)i * allocSize;
    result = hgMemMap(slot, allocSize, 0, peerHandle, 0);
    if (result != HGGC_SUCCESS) {
      if (imported)
        hgMemRelease(peerHandle);
      goto rollback;
    }
    mappedPeers++;
    result = hgMemSetAccess(slot, allocSize, &accessDesc, 1);
    if (imported)
      hgMemRelease(peerHandle);
    if (result != HGGC_SUCCESS)
      goto rollback;
  }

  *flatBase = (void *)base;
  return flagcxSuccess;

rollback:
  for (int i = 0; i < mappedPeers; i++)
    hgMemUnmap(base + (HGdeviceptr)i * allocSize, allocSize);
  hgMemAddressFree(base, totalSize);
  return flagcxUnhandledDeviceError;
}

flagcxResult_t hggcAdaptorSymFlatUnmap(void *flatBase, size_t allocSize,
                                       int nPeers) {
  if (flatBase == NULL)
    return flagcxSuccess;
  HGdeviceptr base = (HGdeviceptr)flatBase;
  size_t totalSize = allocSize * nPeers;
  HGDRVCHECK(hgMemUnmap(base, totalSize));
  HGDRVCHECK(hgMemAddressFree(base, totalSize));
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorSymFlatMappingUnmap(void *flatBase, size_t allocSize,
                                              int nPeers) {
  if (flatBase == NULL)
    return flagcxSuccess;
  HGDRVCHECK(hgMemUnmap((HGdeviceptr)flatBase, allocSize * nPeers));
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorSymFlatVaFree(void *flatBase, size_t allocSize,
                                        int nPeers) {
  if (flatBase == NULL)
    return flagcxSuccess;
  HGDRVCHECK(hgMemAddressFree((HGdeviceptr)flatBase, allocSize * nPeers));
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorSymMulticastSupported(int *supported) {
  if (supported == NULL)
    return flagcxInvalidArgument;

  *supported = 0;
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorSymMulticastCreate(size_t allocSize,
                                             int nLocalDevices,
                                             const int *localDeviceOrdinals,
                                             void **mcHandle,
                                             int *shareableFd) {
  if (mcHandle)
    *mcHandle = NULL;
  if (shareableFd)
    *shareableFd = -1;
  return flagcxNotSupported;
}

flagcxResult_t hggcAdaptorSymMulticastBind(void *mcHandle, int importFd,
                                           void *physHandle, size_t allocSize,
                                           int localRank, int nLocalDevices,
                                           void **mcBase, size_t *mcMapSize) {
  if (mcBase)
    *mcBase = NULL;
  if (mcMapSize)
    *mcMapSize = 0;
  return flagcxNotSupported;
}

flagcxResult_t hggcAdaptorSymMulticastTeardown(void *mcBase, size_t mcMapSize) {
  return flagcxSuccess;
}
flagcxResult_t hggcAdaptorSymMulticastMappingUnmap(void *mcBase,
                                                   size_t mcMapSize) {
  if (mcBase == NULL)
    return flagcxSuccess;
  HGDRVCHECK(hgMemUnmap((HGdeviceptr)mcBase, mcMapSize));
  return flagcxSuccess;
}
flagcxResult_t hggcAdaptorSymMulticastVaFree(void *mcBase, size_t mcMapSize) {
  if (mcBase == NULL)
    return flagcxSuccess;
  HGDRVCHECK(hgMemAddressFree((HGdeviceptr)mcBase, mcMapSize));
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorSymMulticastFree(void *mcHandle) {
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorSymMulticastImport(int, void **mcHandle) {
  if (mcHandle != NULL)
    *mcHandle = NULL;
  return flagcxNotSupported;
}

flagcxResult_t hggcAdaptorGetPointerType(const void *ptr, int *ptrType) {
  if (ptr == NULL || ptrType == NULL)
    return flagcxInvalidArgument;

  hggcPointerAttributes attrs;
  memset(&attrs, 0, sizeof(attrs));
  hggcError_t err = hggcPointerGetAttributes(&attrs, ptr);
  if (err == hggcErrorInvalidValue) {
    hggcGetLastError();
    *ptrType = FLAGCX_PTR_HOST;
    return flagcxSuccess;
  }
  if (err != hggcSuccess) {
    hggcGetLastError();
    return flagcxUnhandledDeviceError;
  }
#if FLAGCX_HGGC_HAS_POINTER_TYPE
  *ptrType = (attrs.type == hggcMemoryTypeDevice ||
              attrs.type == hggcMemoryTypeManaged)
                 ? FLAGCX_PTR_CUDA
                 : FLAGCX_PTR_HOST;
#else
  *ptrType = (attrs.memoryType == hggcMemoryTypeDevice || attrs.isManaged)
                 ? FLAGCX_PTR_CUDA
                 : FLAGCX_PTR_HOST;
#endif
  return flagcxSuccess;
}

flagcxResult_t hggcAdaptorGetAddressRange(const void *ptr, void **base,
                                          size_t *size) {
  if (ptr == NULL || base == NULL || size == NULL)
    return flagcxInvalidArgument;

  HGdeviceptr allocationBase = 0;
  HGresult result =
      hgMemGetAddressRange(&allocationBase, size, (HGdeviceptr)ptr);
  if (result != HGGC_SUCCESS)
    return flagcxUnhandledDeviceError;
  *base = (void *)allocationBase;
  return flagcxSuccess;
}

struct flagcxDeviceAdaptor hggcAdaptor {
  "HGGC",
      // Basic functions
      hggcAdaptorDeviceSynchronize, hggcAdaptorDeviceMemcpy,
      hggcAdaptorDeviceMemset, hggcAdaptorDeviceMalloc, hggcAdaptorDeviceFree,
      hggcAdaptorSetDevice, hggcAdaptorGetDevice, hggcAdaptorGetDeviceCount,
      hggcAdaptorGetVendor, hggcAdaptorHostGetDevicePointer,
      // GDR functions
      NULL, // memHandleInit
      NULL, // memHandleDestroy
      hggcAdaptorGdrMemAlloc, hggcAdaptorGdrMemFree,
      NULL, // hostShareMemAlloc
      NULL, // hostShareMemFree
      NULL, // gdrPtrMmap
      NULL, // gdrPtrMunmap
      // Stream functions
      hggcAdaptorStreamCreate, hggcAdaptorStreamDestroy, hggcAdaptorStreamCopy,
      hggcAdaptorStreamFree, hggcAdaptorStreamSynchronize,
      hggcAdaptorStreamQuery, hggcAdaptorStreamWaitEvent,
      hggcAdaptorStreamWaitValue64, hggcAdaptorStreamWriteValue64,
      // Event functions
      hggcAdaptorEventCreate, hggcAdaptorEventDestroy, hggcAdaptorEventRecord,
      hggcAdaptorEventSynchronize, hggcAdaptorEventQuery,
      hggcAdaptorEventElapsedTime,
      // IpcMemHandle functions
      hggcAdaptorIpcMemHandleCreate, hggcAdaptorIpcMemHandleGet,
      hggcAdaptorIpcMemHandleOpen, hggcAdaptorIpcMemHandleClose,
      hggcAdaptorIpcMemHandleFree,
      // Kernel launch
      NULL, // launchKernel
      NULL, // copyArgsInit
      NULL, // copyArgsFree
      NULL, // launchDeviceFunc
      // Others
      hggcAdaptorGetDeviceProperties, hggcAdaptorGetDevicePciBusId,
      hggcAdaptorGetDeviceByPciBusId, hggcAdaptorLaunchHostFunc,
      // DMA buffer
      hggcAdaptorDmaSupport, hggcAdaptorMemGetHandleForAddressRange,
      hggcAdaptorHostRegister, hggcAdaptorHostUnregister,
      // Symmetric memory VMM functions
      hggcAdaptorSymPhysAlloc, hggcAdaptorSymPhysFree, hggcAdaptorSymFlatMap,
      hggcAdaptorSymFlatUnmap, hggcAdaptorSymMulticastSupported,
      hggcAdaptorSymMulticastCreate, hggcAdaptorSymMulticastBind,
      hggcAdaptorSymMulticastTeardown, hggcAdaptorSymMulticastFree,
      NULL, // flagcxResult_t (*getLastError)();
      hggcAdaptorGetPointerType, hggcAdaptorGetAddressRange,
      FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA,
      FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE, hggcAdaptorSymMulticastImport,
      hggcAdaptorSymFlatMappingUnmap, hggcAdaptorSymFlatVaFree,
      hggcAdaptorSymMulticastMappingUnmap, hggcAdaptorSymMulticastVaFree,
      // Transitional BAREX compatibility policy: NONE preserves the existing
      // PPU CI behavior while ACCL has no GPU-visibility flush API. This is not
      // a documented coherence guarantee. Do not enable READ/WRITE requirements
      // until BAREX advertises a real capability backed by a hardware test.
      FLAGCX_GDR_FLUSH_NONE, FLAGCX_GDR_DEVICE_PPU, NULL,
      hggcAdaptorCanAccessPeer, hggcAdaptorSymPhysDescribe,
      hggcAdaptorSymPhysExportSegment, hggcAdaptorSymPhysFreeSegments,
      hggcAdaptorSymFlatMapSegments, hggcAdaptorSymFlatMappingUnmapSegments,
      hggcAdaptorSymFlatVaFreeSegments, hggcAdaptorSymPhysSegmentCount,
      hggcAdaptorSymFlatMapSegmentsBegin, hggcAdaptorSymFlatMapSegmentOne,
      hggcAdaptorSymLsaGetCapacity, hggcAdaptorSymLsaVaReserve,
      hggcAdaptorSymLsaVaFree, hggcAdaptorSymLsaMapBeginAt, NULL, NULL, NULL,
      NULL,
};

#endif // USE_PPU_ADAPTOR
