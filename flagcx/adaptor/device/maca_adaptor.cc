/*************************************************************************
 * Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd.
   All Rights Reserved.
 ************************************************************************/

#include "metax_adaptor.h"

#ifdef USE_METAX_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "param.h"
#include "segmented_vmm_allocation.h"
#include <dlfcn.h>
#include <limits>
#include <mutex>
#include <new>
#include <unistd.h>
#include <unordered_map>

struct MacaVmmAllocation {
  size_t size;
  bool mappingOwned;
  bool vaOwned;
};
static std::mutex gMacaVmmAllocationMtx;
static std::unordered_map<void *, MacaVmmAllocation> gMacaVmmAllocations;

struct MacaSegmentOps {
  typedef mcMemGenericAllocationHandle Handle;
  typedef mcMemAllocationProp Prop;
  static flagcxResult_t deviceOrdinal(int *device) {
    return mcGetDevice(device) == mcSuccess ? flagcxSuccess
                                            : flagcxUnhandledDeviceError;
  }
  static flagcxResult_t createHandle(size_t size, const Prop &prop,
                                     Handle *handle) {
    return mcMemCreate(handle, size, &prop, 0) == mcSuccess
               ? flagcxSuccess
               : flagcxUnhandledDeviceError;
  }
  static flagcxResult_t reserveAddress(size_t size, size_t alignment,
                                       uintptr_t *base) {
    mcDeviceptr_t address = 0;
    if (mcMemAddressReserve(&address, size, alignment, 0, 0) != mcSuccess)
      return flagcxUnhandledDeviceError;
    *base = reinterpret_cast<uintptr_t>(address);
    return flagcxSuccess;
  }
  static flagcxResult_t freeAddress(uintptr_t base, size_t size) {
    return mcMemAddressFree(reinterpret_cast<mcDeviceptr_t>(base), size) ==
                   mcSuccess
               ? flagcxSuccess
               : flagcxUnhandledDeviceError;
  }
  static flagcxResult_t mapSegment(uintptr_t address, size_t size,
                                   Handle handle) {
    return mcMemMap(reinterpret_cast<mcDeviceptr_t>(address), size, 0, handle,
                    0) == mcSuccess
               ? flagcxSuccess
               : flagcxUnhandledDeviceError;
  }
  static flagcxResult_t unmapSegment(uintptr_t address, size_t size) {
    return mcMemUnmap(reinterpret_cast<mcDeviceptr_t>(address), size) ==
                   mcSuccess
               ? flagcxSuccess
               : flagcxUnhandledDeviceError;
  }
  static flagcxResult_t setAccess(uintptr_t address, size_t size) {
    int device = 0;
    MCdevice mcDevice;
    if (mcGetDevice(&device) != mcSuccess ||
        mcDeviceGet(&mcDevice, device) != mcSuccess)
      return flagcxUnhandledDeviceError;
    mcMemAccessDesc access = {};
    access.location.type = mcMemLocationTypeDevice;
    access.location.id = mcDevice;
    access.flags = mcMemAccessFlagsProtReadWrite;
    return mcMemSetAccess(reinterpret_cast<mcDeviceptr_t>(address), size,
                          &access, 1) == mcSuccess
               ? flagcxSuccess
               : flagcxUnhandledDeviceError;
  }
  static flagcxResult_t importHandle(int fd, Handle *handle) {
    return mcMemImportFromShareableHandle(
               handle, reinterpret_cast<void *>(static_cast<uintptr_t>(fd)),
               mcMemHandleTypePosixFileDescriptor) == mcSuccess
               ? flagcxSuccess
               : flagcxUnhandledDeviceError;
  }
  static flagcxResult_t exportHandle(Handle handle, int *fd) {
    return mcMemExportToShareableHandle(
               fd, handle, mcMemHandleTypePosixFileDescriptor, 0) == mcSuccess
               ? flagcxSuccess
               : flagcxUnhandledDeviceError;
  }
  static flagcxResult_t releaseHandle(Handle handle) {
    return mcMemRelease(handle) == mcSuccess ? flagcxSuccess
                                             : flagcxUnhandledDeviceError;
  }
};
typedef flagcxSegmentedVmmAllocation<MacaSegmentOps> MacaSegmentAllocation;
typedef flagcxSegmentedVmmMap<MacaSegmentOps> MacaSegmentMap;

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

flagcxResult_t macaAdaptorCanAccessPeer(int srcDev, int dstDev,
                                        int *canAccess) {
  if (canAccess == NULL)
    return flagcxInvalidArgument;
  DEVCHECK(mcDeviceCanAccessPeer(canAccess, srcDev, dstDev));
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorHostGetDevicePointer(void **pDevice, void *pHost) {
  if (pDevice == NULL || pHost == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(mcHostGetDevicePointer(pDevice, pHost, 0));
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorGdrMemAlloc(void **ptr, size_t size,
                                      void *memHandle) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
  if (!flagcxParamVmmEnable()) {
    DEVCHECK(mcMalloc(ptr, size));
    mcPointerAttribute_t attrs;
    DEVCHECK(mcPointerGetAttributes(&attrs, *ptr));
    unsigned flags = 1;
    DEVCHECK(mcPointerSetAttribute(&flags, mcPointerAttributeSyncMemops,
                                   (mcDeviceptr_t)attrs.devicePointer));
    return flagcxSuccess;
  }

  int device = 0;
  MCdevice mcDevice;
  DEVCHECK(mcGetDevice(&device));
  FLAGCXCHECK(MacaSegmentAllocation::retryPending(device));
  DEVCHECK(mcDeviceGet(&mcDevice, device));

  mcMemAllocationProp prop = {};
  prop.type = mcMemAllocationTypePinned;
  prop.location.type = mcMemLocationTypeDevice;
  prop.location.id = mcDevice;
  prop.requestedHandleTypes = mcMemHandleTypePosixFileDescriptor;

  size_t granularity = 0;
  DEVCHECK(mcMemGetAllocationGranularity(&granularity, &prop,
                                         MC_MEM_ALLOC_GRANULARITY_MINIMUM));
  size_t allocSize = size;
  ALIGN_SIZE(allocSize, granularity);

  size_t segmentBytes = 0;
  FLAGCXCHECK(flagcxVmmSegmentBytes(granularity, &segmentBytes));
  if (segmentBytes != 0 && segmentBytes < allocSize)
    return MacaSegmentAllocation::allocate(ptr, allocSize, granularity,
                                           segmentBytes, prop);

  mcMemGenericAllocationHandle handle;
  DEVCHECK(mcMemCreate(&handle, allocSize, &prop, 0));
  mcDeviceptr_t address = 0;
  mcError_t error = mcMemAddressReserve(&address, allocSize, granularity, 0, 0);
  if (error != mcSuccess) {
    mcMemRelease(handle);
    return flagcxUnhandledDeviceError;
  }
  error = mcMemMap(address, allocSize, 0, handle, 0);
  if (error != mcSuccess) {
    mcMemAddressFree(address, allocSize);
    mcMemRelease(handle);
    return flagcxUnhandledDeviceError;
  }
  mcMemAccessDesc access = {};
  access.location.type = mcMemLocationTypeDevice;
  access.location.id = mcDevice;
  access.flags = mcMemAccessFlagsProtReadWrite;
  error = mcMemSetAccess(address, allocSize, &access, 1);
  if (error != mcSuccess) {
    mcMemUnmap(address, allocSize);
    mcMemAddressFree(address, allocSize);
    mcMemRelease(handle);
    return flagcxUnhandledDeviceError;
  }
  if (mcMemRelease(handle) != mcSuccess) {
    mcMemUnmap(address, allocSize);
    mcMemAddressFree(address, allocSize);
    return flagcxUnhandledDeviceError;
  }
  *ptr = (void *)(uintptr_t)address;
  bool tracked = false;
  try {
    std::lock_guard<std::mutex> lock(gMacaVmmAllocationMtx);
    tracked = gMacaVmmAllocations
                  .emplace(*ptr, MacaVmmAllocation{allocSize, true, true})
                  .second;
  } catch (const std::bad_alloc &) {
  }
  if (!tracked) {
    mcMemUnmap(address, allocSize);
    mcMemAddressFree(address, allocSize);
    *ptr = NULL;
    return flagcxSystemError;
  }
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return flagcxSuccess;
  }
  bool segmented = false;
  FLAGCXCHECK(MacaSegmentAllocation::freeAllocation(ptr, &segmented));
  if (segmented)
    return flagcxSuccess;
  std::lock_guard<std::mutex> lock(gMacaVmmAllocationMtx);
  auto it = gMacaVmmAllocations.find(ptr);
  if (it == gMacaVmmAllocations.end()) {
    DEVCHECK(mcFree(ptr));
    return flagcxSuccess;
  }
  MacaVmmAllocation &allocation = it->second;
  if (allocation.mappingOwned) {
    if (mcMemUnmap((mcDeviceptr_t)ptr, allocation.size) != mcSuccess)
      return flagcxUnhandledDeviceError;
    allocation.mappingOwned = false;
  }
  if (allocation.vaOwned) {
    if (mcMemAddressFree((mcDeviceptr_t)ptr, allocation.size) != mcSuccess)
      return flagcxUnhandledDeviceError;
    allocation.vaOwned = false;
  }
  gMacaVmmAllocations.erase(it);
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorDmaSupport(bool *supported) {
  if (supported == NULL)
    return flagcxInvalidArgument;
  *supported = false;
  int device = 0;
  MCdevice mcDevice;
  if (mcGetDevice(&device) != mcSuccess ||
      mcDeviceGet(&mcDevice, device) != mcSuccess)
    return flagcxSuccess;
  // A POSIX shareable-handle attribute is only a prerequisite for DMA-BUF
  // export, not proof that every VMM allocation can be exported. The
  // per-allocation callback below performs the definitive range-export probe.
  int value = 0;
  if (mcDeviceGetAttribute(
          &value, mcDeviceAttributeHandleTypePosixFileDescriptorSupported,
          mcDevice) == mcSuccess)
    *supported = value != 0;
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
  int device = -1;
  mcError_t getDeviceResult = mcGetDevice(&device);
  TRACE(FLAGCX_INIT,
        "MetaX IPC open source=device-adaptor device=%d getDeviceResult=%d "
        "rawImportedBase=%p",
        device, (int)getDeviceResult, *devPtr);
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorIpcMemHandleClose(void *devPtr) {
  if (devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  int device = -1;
  mcError_t getDeviceResult = mcGetDevice(&device);
  TRACE(FLAGCX_INIT,
        "MetaX IPC close source=device-adaptor device=%d getDeviceResult=%d "
        "rawImportedBase=%p",
        device, (int)getDeviceResult, devPtr);
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

flagcxResult_t macaAdaptorStreamWaitValue64(flagcxStream_t stream, void *addr,
                                            uint64_t value, int flags) {
  if (stream == NULL || addr == NULL)
    return flagcxInvalidArgument;
  if (flags & ~FLAGCX_STREAM_WAIT_VALUE_FLUSH_REMOTE_WRITES)
    return flagcxInvalidArgument;

  // The current MACA runtime does not implement the acquire/flush semantics
  // required after a NIC or peer device writes the waited-on value.  Reject
  // the stronger contract before calling mcStreamWaitValue64 so callers can
  // select another path without causing a runtime error or poisoning the
  // device context.
  if (flags & FLAGCX_STREAM_WAIT_VALUE_FLUSH_REMOTE_WRITES)
    return flagcxNotSupported;

  mcError_t error =
      mcStreamWaitValue64(stream->base, addr, value,
                          mcStreamWaitValue_flags::MC_STREAM_WAIT_VALUE_GEQ);
  if (error == mcSuccess)
    return flagcxSuccess;
  if (error == mcErrorNotSupported)
    return flagcxNotSupported;
  return flagcxUnhandledDeviceError;
}

flagcxResult_t macaAdaptorStreamWriteValue64(flagcxStream_t stream, void *addr,
                                             uint64_t value, int flags) {
  (void)flags;
  if (stream == NULL || addr == NULL)
    return flagcxInvalidArgument;

  mcError_t error = mcStreamWriteValue64(
      stream->base, addr, value,
      mcStreamWriteValue_flags::MC_STREAM_WRITE_VALUE_DEFAULT);
  return error == mcSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

flagcxResult_t
macaAdaptorMemGetHandleForAddressRange(void *handleOut, void *buffer,
                                       size_t size, unsigned long long flags) {
  if (handleOut == NULL || buffer == NULL || size == 0)
    return flagcxInvalidArgument;
  mcError_t result =
      mcMemGetHandleForAddressRange(handleOut, buffer, size, 0x1, flags);
  // MetaX reports mcErrorInvalidDevicePointer when a valid VMM allocation
  // cannot be exported as DMA-BUF.  Treat that allocation-specific rejection
  // as an unavailable route so the common MR policy can probe VA. Invalid
  // caller arguments have already been rejected above, and the VMM MR path
  // validates the native address range before reaching this callback.
  if (result == mcErrorNotSupported || result == mcErrorInvalidDevicePointer)
    return flagcxNotSupported;
  return result == mcSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

flagcxResult_t macaAdaptorHostRegister(void *ptr, size_t size) {
  if (ptr == NULL || size == 0) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(mcHostRegister(ptr, size, mcHostRegisterMapped));
  return flagcxSuccess;
}
flagcxResult_t macaAdaptorHostUnregister(void *ptr) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
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
  mcError_t result = mcMemRetainAllocationHandle(mcHandle, ptr);
  if (result != mcSuccess) {
    free(mcHandle);
    if (result == mcErrorInvalidValue || result == mcErrorNotSupported)
      return flagcxNotSupported;
    return flagcxUnhandledDeviceError;
  }

  // Discover actual physical allocation size (already granularity-aligned)
  mcDeviceptr_t allocationBase = 0;
  size_t actualAllocSize = 0;
  result = mcMemGetAddressRange(&allocationBase, &actualAllocSize, ptr);
  if (result != mcSuccess) {
    mcMemRelease(*mcHandle);
    free(mcHandle);
    return flagcxUnhandledDeviceError;
  }
  const uintptr_t addressValue = (uintptr_t)ptr;
  const uintptr_t allocationBaseValue = (uintptr_t)allocationBase;
  if (allocationBaseValue == 0 || actualAllocSize == 0 ||
      addressValue < allocationBaseValue) {
    mcMemRelease(*mcHandle);
    free(mcHandle);
    return flagcxInvalidUsage;
  }
  const uintptr_t allocationOffset = addressValue - allocationBaseValue;
  if (allocationOffset > actualAllocSize ||
      size > actualAllocSize - allocationOffset) {
    mcMemRelease(*mcHandle);
    free(mcHandle);
    return flagcxInvalidUsage;
  }
  *allocSize = actualAllocSize;

  // Export as POSIX fd for IPC sharing
  if (*handleSize < sizeof(int)) {
    mcMemRelease(*mcHandle);
    free(mcHandle);
    return flagcxInvalidArgument;
  }
  result = mcMemExportToShareableHandle(shareableHandle, *mcHandle,
                                        mcMemHandleTypePosixFileDescriptor, 0);
  if (result != mcSuccess) {
    mcMemRelease(*mcHandle);
    free(mcHandle);
    return flagcxUnhandledDeviceError;
  }
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
  if (mcMemRelease(*mcHandle) != mcSuccess)
    return flagcxUnhandledDeviceError;
  free(mcHandle);
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorSymPhysDescribe(void *ptr, size_t size,
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
  const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  if (size > std::numeric_limits<uintptr_t>::max() - address)
    return flagcxInvalidArgument;

  MacaSegmentMap::Bundle *bundle = new (std::nothrow) MacaSegmentMap::Bundle();
  if (bundle == NULL)
    return flagcxSystemError;
  *segmentBundle = bundle;
  bool found = false;
  flagcxResult_t result =
      MacaSegmentAllocation::describe(ptr, size, bundle, userOffset, &found);
  if (result != flagcxSuccess)
    return result;
  if (!found) {
    std::lock_guard<std::mutex> lock(gMacaVmmAllocationMtx);
    for (std::unordered_map<void *, MacaVmmAllocation>::const_iterator it =
             gMacaVmmAllocations.begin();
         it != gMacaVmmAllocations.end(); ++it) {
      const uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
      const MacaVmmAllocation &allocation = it->second;
      if (address < base || address - base > allocation.size ||
          size > allocation.size - (address - base))
        continue;
      found = true;
      if (!allocation.mappingOwned || !allocation.vaOwned)
        return flagcxInvalidUsage;
      mcMemGenericAllocationHandle handle = 0;
      if (mcMemRetainAllocationHandle(&handle, ptr) != mcSuccess)
        return flagcxUnhandledDeviceError;
      try {
        bundle->segments.push_back(flagcxSymSegmentMeta{0, allocation.size});
        bundle->handleOwned.push_back(1);
        bundle->handles.push_back(handle);
      } catch (const std::bad_alloc &) {
        if (bundle->handles.empty())
          mcMemRelease(handle);
        return flagcxSystemError;
      }
      *userOffset = address - base;
      break;
    }
  }
  if (!found) {
    // Only a successful legacy IPC export proves that an untracked pointer
    // may use that path. External VMM offsets are unknown, so no retain or
    // remapping is safe for any other untracked pointer.
    mcDeviceptr_t allocationBase = 0;
    size_t allocationSize = 0;
    if (mcMemGetAddressRange(&allocationBase, &allocationSize,
                             (mcDeviceptr_t)ptr) == mcSuccess &&
        address >= (uintptr_t)allocationBase &&
        address - (uintptr_t)allocationBase <= allocationSize &&
        size <= allocationSize - (address - (uintptr_t)allocationBase)) {
      mcIpcMemHandle_t ipcHandle;
      if (mcIpcGetMemHandle(&ipcHandle, (void *)allocationBase) == mcSuccess) {
        delete bundle;
        *segmentBundle = NULL;
      } else {
        mcGetLastError();
      }
    }
    // Keep an empty bundle for an unknown mapping to block IPC fallback.
    return flagcxNotSupported;
  }
  // Ordinary allocations have already returned without requiring VMM support.
  int device = 0;
  MCdevice mcDevice;
  if (mcGetDevice(&device) != mcSuccess ||
      mcDeviceGet(&mcDevice, device) != mcSuccess)
    return flagcxUnhandledDeviceError;
  mcMemAllocationProp prop = {};
  prop.type = mcMemAllocationTypePinned;
  prop.location.type = mcMemLocationTypeDevice;
  prop.location.id = mcDevice;
  prop.requestedHandleTypes = mcMemHandleTypePosixFileDescriptor;
  if (mcMemGetAllocationGranularity(slotAlignment, &prop,
                                    MC_MEM_ALLOC_GRANULARITY_MINIMUM) !=
          mcSuccess ||
      *slotAlignment == 0)
    return flagcxUnhandledDeviceError;
  if (bundle->segments.empty() ||
      bundle->segments.size() >
          static_cast<size_t>(std::numeric_limits<int>::max()))
    return flagcxInvalidUsage;
  *segments = bundle->segments.data();
  *segmentCount = static_cast<int>(bundle->segments.size());
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorSymPhysExportSegment(void *bundle, int index,
                                               int *fd) {
  return MacaSegmentMap::exportSegment(bundle, index, fd);
}
flagcxResult_t macaAdaptorSymPhysFreeSegments(void *bundle) {
  return MacaSegmentMap::freeBundle(bundle);
}
flagcxResult_t macaAdaptorSymPhysSegmentCount(const void *ptr, size_t size,
                                              int *count) {
  flagcxResult_t result = MacaSegmentAllocation::segmentCount(ptr, size, count);
  if (result != flagcxNotSupported)
    return result;
  const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  std::lock_guard<std::mutex> lock(gMacaVmmAllocationMtx);
  for (std::unordered_map<void *, MacaVmmAllocation>::const_iterator it =
           gMacaVmmAllocations.begin();
       it != gMacaVmmAllocations.end(); ++it) {
    const uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
    if (address < base || address - base > it->second.size ||
        size > it->second.size - (address - base))
      continue;
    *count = 1;
    return flagcxSuccess;
  }
  return flagcxNotSupported;
}
flagcxResult_t macaAdaptorSymFlatMapSegments(
    const int *fds, int nPeers, int self, void *bundle,
    const flagcxSymRankLayout *layouts, const flagcxSymSegmentMeta *segments,
    int totalSegments, size_t stride, size_t alignment, void **flatBase) {
  return MacaSegmentMap::map(fds, nPeers, self, bundle, layouts, segments,
                             totalSegments, stride, alignment, flatBase);
}
flagcxResult_t macaAdaptorSymFlatMapSegmentsBegin(
    int nPeers, int self, void *bundle, const flagcxSymRankLayout *layouts,
    const flagcxSymSegmentMeta *segments, int totalSegments, size_t stride,
    size_t alignment, void **flatBase) {
  return MacaSegmentMap::begin(nPeers, self, bundle, layouts, segments,
                               totalSegments, stride, alignment, flatBase);
}
flagcxResult_t macaAdaptorSymFlatMapSegmentOne(void *flatBase, int index,
                                               int fd) {
  return MacaSegmentMap::mapOne(flatBase, index, fd);
}
flagcxResult_t macaAdaptorSymFlatMappingUnmapSegments(void *flatBase) {
  return MacaSegmentMap::unmap(flatBase);
}
flagcxResult_t macaAdaptorSymFlatVaFreeSegments(void *flatBase) {
  return MacaSegmentMap::freeVa(flatBase);
}

flagcxResult_t macaAdaptorSymLsaGetCapacity(size_t *totalBytes,
                                            size_t *alignment) {
  if (totalBytes == NULL || alignment == NULL)
    return flagcxInvalidArgument;
  int device = 0;
  mcDeviceProp_t properties = {};
  if (mcGetDevice(&device) != mcSuccess ||
      mcGetDeviceProperties(&properties, device) != mcSuccess)
    return flagcxUnhandledDeviceError;
  *totalBytes = properties.totalGlobalMem;
  *alignment = 512ULL * 1024 * 1024;
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorSymLsaVaReserve(size_t size, size_t alignment,
                                          void **base) {
  if (base == NULL)
    return flagcxInvalidArgument;
  uintptr_t address = 0;
  flagcxResult_t result =
      MacaSegmentOps::reserveAddress(size, alignment, &address);
  *base = result == flagcxSuccess ? reinterpret_cast<void *>(address) : NULL;
  return result;
}

flagcxResult_t macaAdaptorSymLsaVaFree(void *base, size_t size) {
  return MacaSegmentOps::freeAddress(reinterpret_cast<uintptr_t>(base), size);
}

flagcxResult_t macaAdaptorSymLsaMapBeginAt(
    void *windowBase, int nPeers, int self, void *bundle,
    const flagcxSymRankLayout *layouts, const flagcxSymSegmentMeta *segments,
    int totalSegments, size_t stride, size_t alignment, void **mappingBase) {
  return MacaSegmentMap::beginAt(windowBase, nPeers, self, bundle, layouts,
                                 segments, totalSegments, stride, alignment,
                                 mappingBase);
}

flagcxResult_t macaAdaptorSymFlatMap(void *peerHandles[], int nPeers,
                                     int selfIndex, void *selfPhysHandle,
                                     size_t allocSize, void **flatBase) {
  if (peerHandles == NULL || selfPhysHandle == NULL || flatBase == NULL ||
      nPeers <= 0 || allocSize == 0)
    return flagcxInvalidArgument;

  mcMemGenericAllocationHandle selfHandle =
      *(mcMemGenericAllocationHandle *)selfPhysHandle;

  // allocSize is already granularity-aligned (from mcMemGetAddressRange)
  size_t totalSize = allocSize * nPeers;

  // Reserve the full VA range
  mcDeviceptr_t base = 0;
  mcError_t result = mcMemAddressReserve(&base, totalSize, 0, 0, 0);
  if (result != mcSuccess)
    return flagcxUnhandledDeviceError;

  // Import and map each peer's physical memory
  int macaDev;
  if (mcGetDevice(&macaDev) != mcSuccess) {
    mcMemAddressFree(base, totalSize);
    return flagcxUnhandledDeviceError;
  }
  mcMemAccessDesc accessDesc = {};
  accessDesc.location.type = mcMemLocationTypeDevice;
  accessDesc.location.id = macaDev;
  accessDesc.flags = mcMemAccessFlagsProtReadWrite;

  int mappedPeers = 0;
  for (int i = 0; i < nPeers; i++) {
    mcMemGenericAllocationHandle peerHandle;
    bool imported = i != selfIndex;
    if (i == selfIndex) {
      peerHandle = selfHandle;
    } else {
      int fd = *(int *)peerHandles[i];
      result =
          mcMemImportFromShareableHandle(&peerHandle, (void *)(uintptr_t)fd,
                                         mcMemHandleTypePosixFileDescriptor);
      if (result != mcSuccess)
        goto rollback;
    }
    mcDeviceptr_t slot =
        (mcDeviceptr_t)((uintptr_t)base + (uint64_t)i * allocSize);
    result = mcMemMap(slot, allocSize, 0, peerHandle, 0);
    if (result != mcSuccess) {
      if (imported)
        mcMemRelease(peerHandle);
      goto rollback;
    }
    mappedPeers++;
    result = mcMemSetAccess(slot, allocSize, &accessDesc, 1);
    if (imported)
      mcMemRelease(peerHandle);
    if (result != mcSuccess)
      goto rollback;
  }

  *flatBase = (void *)base;
  return flagcxSuccess;

rollback:
  for (int i = 0; i < mappedPeers; i++) {
    mcDeviceptr_t slot =
        (mcDeviceptr_t)((uintptr_t)base + (uint64_t)i * allocSize);
    mcMemUnmap(slot, allocSize);
  }
  mcMemAddressFree(base, totalSize);
  return flagcxUnhandledDeviceError;
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

flagcxResult_t macaAdaptorSymFlatMappingUnmap(void *flatBase, size_t allocSize,
                                              int nPeers) {
  if (flatBase == NULL)
    return flagcxSuccess;
  DEVCHECK(mcMemUnmap((mcDeviceptr_t)flatBase, allocSize * nPeers));
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorSymFlatVaFree(void *flatBase, size_t allocSize,
                                        int nPeers) {
  if (flatBase == NULL)
    return flagcxSuccess;
  DEVCHECK(mcMemAddressFree((mcDeviceptr_t)flatBase, allocSize * nPeers));
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

flagcxResult_t macaAdaptorSymMulticastImport(int importFd, void **mcHandle) {
  if (importFd < 0 || mcHandle == NULL)
    return flagcxInvalidArgument;
  *mcHandle = NULL;

  mcMemGenericAllocationHandle handle = 0;
  mcError_t res = mcMemImportFromShareableHandle(
      &handle, (void *)(intptr_t)importFd, mcMemHandleTypePosixFileDescriptor);
  if (res != mcSuccess) {
    WARN("symMulticastImport: mcMemImportFromShareableHandle failed: %d", res);
    return flagcxUnhandledDeviceError;
  }

  mcMemGenericAllocationHandle *handlePtr =
      (mcMemGenericAllocationHandle *)malloc(
          sizeof(mcMemGenericAllocationHandle));
  if (handlePtr == NULL) {
    mcMemRelease(handle);
    return flagcxSystemError;
  }
  *handlePtr = handle;
  *mcHandle = handlePtr;
  return flagcxSuccess;
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
      WARN("symMulticastBind: mcMemImportFromShareableHandle failed: %d", res);
      return flagcxUnhandledDeviceError;
    }
  }

  mcMemGenericAllocationHandle mcPhysHandle =
      *(mcMemGenericAllocationHandle *)physHandle;

  // Bind this rank's physical allocation to the multicast object.
  // Use mcMulticastBindMem (takes physical handle), not mcMulticastBindAddr
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
    WARN("symMulticastBind: mcMemSetAccess failed: %d", res);
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

flagcxResult_t macaAdaptorSymLsaMulticastGranularity(size_t objectSize,
                                                     int nLocalDevices,
                                                     size_t *granularity) {
  if (objectSize == 0 || nLocalDevices <= 0 || granularity == NULL)
    return flagcxInvalidArgument;
  // Older MACA SDKs expose multicast objects but not address binding. Keep
  // those builds on unicast LSA without requiring the newer declarations.
  if (dlsym(RTLD_DEFAULT, "mcMulticastBindAddr") == NULL ||
      dlsym(RTLD_DEFAULT, "mcMulticastUnbind") == NULL)
    return flagcxNotSupported;
  mcMulticastObjectProp prop = {};
  prop.numDevices = (unsigned int)nLocalDevices;
  prop.size = objectSize;
  prop.handleTypes = mcMemHandleTypePosixFileDescriptor;
  mcError_t result = mcMulticastGetGranularity(
      granularity, &prop, MC_MULTICAST_GRANULARITY_RECOMMENDED);
  return result == mcSuccess && *granularity != 0 ? flagcxSuccess
                                                  : flagcxUnhandledDeviceError;
}

flagcxResult_t macaAdaptorSymLsaMulticastMap(void *mcHandle, size_t size,
                                             size_t alignment, void **base,
                                             bool *mappingOwned,
                                             bool *vaOwned) {
  if (mcHandle == NULL || size == 0 || base == NULL || mappingOwned == NULL ||
      vaOwned == NULL)
    return flagcxInvalidArgument;
  *base = NULL;
  *mappingOwned = false;
  *vaOwned = false;
  mcDeviceptr_t address = 0;
  mcError_t result = mcMemAddressReserve(&address, size, alignment, 0, 0);
  if (result != mcSuccess)
    return flagcxUnhandledDeviceError;
  *base = reinterpret_cast<void *>(address);
  *vaOwned = true;
  result = mcMemMap(address, size, 0,
                    *static_cast<mcMemGenericAllocationHandle *>(mcHandle), 0);
  if (result != mcSuccess)
    return flagcxUnhandledDeviceError;
  *mappingOwned = true;
  int device = -1;
  if (mcGetDevice(&device) != mcSuccess)
    return flagcxUnhandledDeviceError;
  mcMemAccessDesc access = {};
  access.location.type = mcMemLocationTypeDevice;
  access.location.id = device;
  access.flags = mcMemAccessFlagsProtReadWrite;
  return mcMemSetAccess(address, size, &access, 1) == mcSuccess
             ? flagcxSuccess
             : flagcxUnhandledDeviceError;
}

flagcxResult_t macaAdaptorSymLsaMulticastBind(void *mcHandle, size_t offset,
                                              const void *source, size_t size) {
  if (mcHandle == NULL || source == NULL || size == 0)
    return flagcxInvalidArgument;
  typedef mcError_t (*BindAddr)(mcMemGenericAllocationHandle, size_t,
                                mcDeviceptr_t, size_t, unsigned long long);
  BindAddr bind =
      reinterpret_cast<BindAddr>(dlsym(RTLD_DEFAULT, "mcMulticastBindAddr"));
  if (bind == NULL)
    return flagcxNotSupported;
  return bind(*static_cast<mcMemGenericAllocationHandle *>(mcHandle), offset,
              (mcDeviceptr_t)(uintptr_t)source, size, 0) == mcSuccess
             ? flagcxSuccess
             : flagcxUnhandledDeviceError;
}

flagcxResult_t macaAdaptorSymLsaMulticastUnbind(void *mcHandle, size_t offset,
                                                size_t size) {
  if (mcHandle == NULL || size == 0)
    return flagcxInvalidArgument;
  int device = -1;
  if (mcGetDevice(&device) != mcSuccess)
    return flagcxUnhandledDeviceError;
  typedef mcError_t (*Unbind)(mcMemGenericAllocationHandle, int, size_t,
                              size_t);
  Unbind unbind =
      reinterpret_cast<Unbind>(dlsym(RTLD_DEFAULT, "mcMulticastUnbind"));
  if (unbind == NULL)
    return flagcxNotSupported;
  return unbind(*static_cast<mcMemGenericAllocationHandle *>(mcHandle), device,
                offset, size) == mcSuccess
             ? flagcxSuccess
             : flagcxUnhandledDeviceError;
}

flagcxResult_t macaAdaptorSymMulticastTeardown(void *mcBase, size_t mcMapSize) {
  if (mcBase == NULL)
    return flagcxSuccess;
  mcDeviceptr_t va = (mcDeviceptr_t)mcBase;
  DEVCHECK(mcMemUnmap(va, mcMapSize));
  DEVCHECK(mcMemAddressFree(va, mcMapSize));
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorSymMulticastMappingUnmap(void *mcBase,
                                                   size_t mcMapSize) {
  if (mcBase == NULL)
    return flagcxSuccess;
  DEVCHECK(mcMemUnmap((mcDeviceptr_t)mcBase, mcMapSize));
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorSymMulticastVaFree(void *mcBase, size_t mcMapSize) {
  if (mcBase == NULL)
    return flagcxSuccess;
  DEVCHECK(mcMemAddressFree((mcDeviceptr_t)mcBase, mcMapSize));
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

flagcxResult_t macaAdaptorGetPointerType(const void *ptr, int *ptrType) {
  if (ptr == NULL || ptrType == NULL)
    return flagcxInvalidArgument;

  mcPointerAttribute_t attrs = {};
  mcError_t err = mcPointerGetAttributes(&attrs, ptr);
  if (err == mcErrorInvalidValue) {
    // Ordinary host allocations are not tracked by the MACA runtime.
    mcGetLastError();
    *ptrType = FLAGCX_PTR_HOST;
    return flagcxSuccess;
  }
  if (err != mcSuccess) {
    mcGetLastError();
    return flagcxUnhandledDeviceError;
  }
#if CUDART_VERSION >= 10000
  *ptrType =
      (attrs.type == mcMemoryTypeDevice || attrs.type == mcMemoryTypeManaged)
          ? FLAGCX_PTR_CUDA
          : FLAGCX_PTR_HOST;
#else
  *ptrType = (attrs.memoryType == mcMemoryTypeDevice || attrs.isManaged)
                 ? FLAGCX_PTR_CUDA
                 : FLAGCX_PTR_HOST;
#endif
  return flagcxSuccess;
}

flagcxResult_t macaAdaptorGetAddressRange(const void *ptr, void **base,
                                          size_t *size) {
  if (ptr == NULL || base == NULL || size == NULL)
    return flagcxInvalidArgument;

  mcDeviceptr_t allocationBase = 0;
  DEVCHECK(mcMemGetAddressRange(&allocationBase, size, (mcDeviceptr_t)ptr));
  *base = (void *)(uintptr_t)allocationBase;
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
      macaAdaptorDmaSupport,
      macaAdaptorMemGetHandleForAddressRange, // flagcxResult_t
                                              // (*memGetHandleForAddressRange)(void
                                              // *handleOut, void *buffer,
                                              // size_t size, unsigned long long
                                              // flags);
      macaAdaptorHostRegister,   // flagcxResult_t (*hostRegister)(void *,
                                 // size_t);
      macaAdaptorHostUnregister, // flagcxResult_t (*hostUnregister)(void *);
      // Symmetric memory VMM functions
      macaAdaptorSymPhysAlloc, macaAdaptorSymPhysFree, macaAdaptorSymFlatMap,
      macaAdaptorSymFlatUnmap, macaAdaptorSymMulticastSupported,
      macaAdaptorSymMulticastCreate, macaAdaptorSymMulticastBind,
      macaAdaptorSymMulticastTeardown, macaAdaptorSymMulticastFree,
      NULL, // flagcxResult_t (*getLastError)();
      macaAdaptorGetPointerType, macaAdaptorGetAddressRange,
      FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA,
      FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE, macaAdaptorSymMulticastImport,
      macaAdaptorSymFlatMappingUnmap, macaAdaptorSymFlatVaFree,
      macaAdaptorSymMulticastMappingUnmap, macaAdaptorSymMulticastVaFree,
      // Stay conservative until the MACA runtime and NIC provider document a
      // coherent path. WRITE-dependent stream waits currently fail safely as
      // unsupported instead of silently weakening the acquire contract.
      FLAGCX_GDR_READ_REQUIRES_FLUSH | FLAGCX_GDR_WRITE_REQUIRES_FLUSH,
      FLAGCX_GDR_DEVICE_METAX, NULL, macaAdaptorCanAccessPeer,
      macaAdaptorSymPhysDescribe, macaAdaptorSymPhysExportSegment,
      macaAdaptorSymPhysFreeSegments, macaAdaptorSymFlatMapSegments,
      macaAdaptorSymFlatMappingUnmapSegments, macaAdaptorSymFlatVaFreeSegments,
      macaAdaptorSymPhysSegmentCount, macaAdaptorSymFlatMapSegmentsBegin,
      macaAdaptorSymFlatMapSegmentOne, macaAdaptorSymLsaGetCapacity,
      macaAdaptorSymLsaVaReserve, macaAdaptorSymLsaVaFree,
      macaAdaptorSymLsaMapBeginAt, macaAdaptorSymLsaMulticastGranularity,
      macaAdaptorSymLsaMulticastMap, macaAdaptorSymLsaMulticastBind,
      macaAdaptorSymLsaMulticastUnbind,
};

#endif // USE_METAX_ADAPTOR
