#include "du_adaptor.h"

#ifdef USE_DU_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "param.h"
#include "segmented_vmm_allocation.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

struct DuhipVmmAllocation {
  hipMemGenericAllocationHandle_t handle;
  size_t size;
  bool mappingOwned;
  bool vaOwned;
  bool handleOwned;
};
static std::mutex gDuhipVmmAllocationMtx;
static std::unordered_map<void *, DuhipVmmAllocation> gDuhipVmmAllocations;

struct DuhipSymPhysHandle {
  hipMemGenericAllocationHandle_t handle;
  hipDeviceptr_t allocationBase;
  size_t allocationSize;
  bool releaseOwned;
};

struct DuhipFlatMapping {
  size_t allocSize;
  int nPeers;
  std::vector<unsigned char> mappedSlots;
  std::vector<hipMemGenericAllocationHandle_t> importedHandles;
  std::vector<unsigned char> importedHandleOwned;
  bool vaOwned;
};
static std::mutex gDuhipFlatMappingMtx;
static std::unordered_map<void *, DuhipFlatMapping> gDuhipFlatMappings;

// Keep the physical handles private to the HIP adaptor. The allocation and
// mapping helpers own partial rollback, including imported handle lifetime.
static thread_local size_t duhipFailAfterMap =
    std::numeric_limits<size_t>::max();
static flagcxResult_t duhipVmmDeviceError(const char *stage, hipError_t error,
                                          size_t size = 0,
                                          size_t alignment = 0) {
  WARN("DU VMM failed: stage=%s hipError=%d size=%zu alignment=%zu", stage,
       static_cast<int>(error), size, alignment);
  return flagcxUnhandledDeviceError;
}

struct DuhipSegmentOps {
  typedef hipMemGenericAllocationHandle_t Handle;
  typedef hipMemAllocationProp Prop;

  static flagcxResult_t deviceOrdinal(int *device) {
    hipError_t error = hipGetDevice(device);
    return error == hipSuccess ? flagcxSuccess
                               : duhipVmmDeviceError("get-device", error);
  }
  static flagcxResult_t createHandle(size_t size, const Prop &prop,
                                     Handle *handle) {
    hipError_t error = hipMemCreate(handle, size, &prop, 0);
    return error == hipSuccess
               ? flagcxSuccess
               : duhipVmmDeviceError("create-handle", error, size);
  }
  static flagcxResult_t reserveAddress(size_t size, size_t alignment,
                                       uintptr_t *base,
                                       const char *stage = "address-reserve") {
    void *address = NULL;
    hipError_t error = hipMemAddressReserve(&address, size, alignment, NULL, 0);
    if (error != hipSuccess)
      return duhipVmmDeviceError(stage, error, size, alignment);
    *base = reinterpret_cast<uintptr_t>(address);
    return flagcxSuccess;
  }
  static flagcxResult_t freeAddress(uintptr_t base, size_t size) {
    hipError_t error = hipMemAddressFree(reinterpret_cast<void *>(base), size);
    return error == hipSuccess
               ? flagcxSuccess
               : duhipVmmDeviceError("address-free", error, size);
  }
  static flagcxResult_t mapSegment(uintptr_t address, size_t size,
                                   Handle handle) {
    if (duhipFailAfterMap == 0)
      return flagcxUnhandledDeviceError;
    if (duhipFailAfterMap != std::numeric_limits<size_t>::max())
      --duhipFailAfterMap;
    hipError_t error =
        hipMemMap(reinterpret_cast<void *>(address), size, 0, handle, 0);
    return error == hipSuccess
               ? flagcxSuccess
               : duhipVmmDeviceError("map-segment", error, size);
  }
  static flagcxResult_t unmapSegment(uintptr_t address, size_t size) {
    hipError_t error = hipMemUnmap(reinterpret_cast<void *>(address), size);
    return error == hipSuccess
               ? flagcxSuccess
               : duhipVmmDeviceError("unmap-segment", error, size);
  }
  static flagcxResult_t setAccess(uintptr_t address, size_t size) {
    int device = 0;
    hipError_t error = hipGetDevice(&device);
    if (error != hipSuccess)
      return duhipVmmDeviceError("set-access-get-device", error, size);
    hipMemAccessDesc access = {};
    access.location.type = hipMemLocationTypeDevice;
    access.location.id = device;
    access.flags = hipMemAccessFlagsProtReadWrite;
    error =
        hipMemSetAccess(reinterpret_cast<void *>(address), size, &access, 1);
    return error == hipSuccess ? flagcxSuccess
                               : duhipVmmDeviceError("set-access", error, size);
  }
  static flagcxResult_t importHandle(int fd, Handle *handle) {
    hipError_t error = hipMemImportFromShareableHandle(
        handle, reinterpret_cast<void *>(static_cast<uintptr_t>(fd)),
        hipMemHandleTypePosixFileDescriptor);
    return error == hipSuccess ? flagcxSuccess
                               : duhipVmmDeviceError("import-handle", error);
  }
  static flagcxResult_t exportHandle(Handle handle, int *fd) {
    hipError_t error = hipMemExportToShareableHandle(
        fd, handle, hipMemHandleTypePosixFileDescriptor, 0);
    return error == hipSuccess ? flagcxSuccess
                               : duhipVmmDeviceError("export-handle", error);
  }
  static flagcxResult_t releaseHandle(Handle handle) {
    hipError_t error = hipMemRelease(handle);
    return error == hipSuccess ? flagcxSuccess
                               : duhipVmmDeviceError("release-handle", error);
  }
};
typedef flagcxSegmentedVmmAllocation<DuhipSegmentOps> DuhipSegmentAllocation;
typedef flagcxSegmentedVmmMap<DuhipSegmentOps> DuhipSegmentMap;

static bool duhipFlatMappingMatches(const DuhipFlatMapping &mapping,
                                    size_t allocSize, int nPeers) {
  return mapping.allocSize == allocSize && mapping.nPeers == nPeers;
}

std::map<flagcxMemcpyType_t, hipMemcpyKind> memcpy_type_map = {
    {flagcxMemcpyHostToDevice, hipMemcpyHostToDevice},
    {flagcxMemcpyDeviceToHost, hipMemcpyDeviceToHost},
    {flagcxMemcpyDeviceToDevice, hipMemcpyDeviceToDevice},
};

static hipError_t duhipIpcAtomicAlloc(void **ptr, size_t size) {
  int device = 0;
  hipError_t result = hipGetDevice(&device);
  if (result != hipSuccess)
    return result;
  hipDeviceProp_t props = {};
  result = hipGetDeviceProperties(&props, device);
  if (result != hipSuccess)
    return result;
  const unsigned flags = strncmp(props.gcnArchName, "gfx942", 6) == 0 ||
                                 strncmp(props.gcnArchName, "gfx950", 6) == 0
                             ? hipDeviceMallocUncached
                             : hipDeviceMallocFinegrained;
  return hipExtMallocWithFlags(ptr, size, flags);
}

flagcxResult_t duhipAdaptorDeviceSynchronize() {
  DEVCHECK(hipDeviceSynchronize());
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
                                        flagcxMemcpyType_t type,
                                        flagcxStream_t stream, void *args) {
  if (stream == NULL) {
    DEVCHECK(hipMemcpy(dst, src, size, memcpy_type_map[type]));
  } else {
    DEVCHECK(
        hipMemcpyAsync(dst, src, size, memcpy_type_map[type], stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorDeviceMemset(void *ptr, int value, size_t size,
                                        flagcxMemType_t type,
                                        flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    memset(ptr, value, size);
  } else {
    if (stream == NULL) {
      DEVCHECK(hipMemset(ptr, value, size));
    } else {
      DEVCHECK(hipMemsetAsync(ptr, value, size, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorDeviceMalloc(void **ptr, size_t size,
                                        flagcxMemType_t type,
                                        flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    DEVCHECK(hipHostAlloc(ptr, size, hipHostAllocMapped));
  } else if (type == flagcxMemManaged) {
    DEVCHECK(hipMallocManaged(ptr, size, hipMemAttachGlobal));
  } else {
    if (stream == NULL) {
      DEVCHECK(duhipIpcAtomicAlloc(ptr, size));
    } else {
      DEVCHECK(hipMallocAsync(ptr, size, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorDeviceFree(void *ptr, flagcxMemType_t type,
                                      flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    DEVCHECK(hipFreeHost(ptr));
  } else if (type == flagcxMemManaged) {
    DEVCHECK(hipFree(ptr));
  } else {
    if (stream == NULL) {
      DEVCHECK(hipFree(ptr));
    } else {
      DEVCHECK(hipFreeAsync(ptr, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorSetDevice(int dev) {
  DEVCHECK(hipSetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorGetDevice(int *dev) {
  DEVCHECK(hipGetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorGetDeviceCount(int *count) {
  DEVCHECK(hipGetDeviceCount(count));
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "DU");
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorCanAccessPeer(int srcDev, int dstDev,
                                         int *canAccess) {
  if (canAccess == NULL)
    return flagcxInvalidArgument;
  DEVCHECK(hipDeviceCanAccessPeer(canAccess, srcDev, dstDev));
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorGetPointerType(const void *ptr, int *ptrType) {
  if (ptr == NULL || ptrType == NULL)
    return flagcxInvalidArgument;

  hipPointerAttribute_t attrs = {};
  hipError_t err = hipPointerGetAttributes(&attrs, ptr);
  if (err == hipErrorInvalidValue) {
    // Ordinary host allocations are not tracked by the DU runtime. Clear the
    // probe error so it cannot affect a later runtime call.
    hipGetLastError();
    *ptrType = FLAGCX_PTR_HOST;
    return flagcxSuccess;
  }
  if (err != hipSuccess) {
    // Runtime/device errors must not silently turn a GPU pointer into host
    // memory, which would register it through the wrong transport path.
    hipGetLastError();
    return flagcxUnhandledDeviceError;
  }
  *ptrType =
      (attrs.type == hipMemoryTypeDevice || attrs.type == hipMemoryTypeManaged)
          ? FLAGCX_PTR_CUDA
          : FLAGCX_PTR_HOST;
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorHostGetDevicePointer(void **pDevice, void *pHost) {
  if (pDevice == NULL || pHost == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(hipHostGetDevicePointer(pDevice, pHost, 0));
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorGdrMemAlloc(void **ptr, size_t size,
                                       void *memHandle) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
  *ptr = NULL;
  if (!flagcxParamVmmEnable()) {
    hipError_t result = duhipIpcAtomicAlloc(ptr, size);
    if (result != hipSuccess)
      return flagcxUnhandledDeviceError;
    hipPointerAttribute_t attrs = {};
    result = hipPointerGetAttributes(&attrs, *ptr);
    unsigned flags = 1;
    if (result == hipSuccess)
      result = hipPointerSetAttribute(&flags, HIP_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                      attrs.devicePointer);
    if (result != hipSuccess) {
      hipFree(*ptr);
      *ptr = nullptr;
      return flagcxUnhandledDeviceError;
    }
    return flagcxSuccess;
  }

  int device = 0;
  DEVCHECK(hipGetDevice(&device));
  FLAGCXCHECK(DuhipSegmentAllocation::retryPending(device));

  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;
  prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;

  // The DCU HIP compatibility layer aborts on NVIDIA-only capability enum
  // 110 instead of returning hipErrorNotSupported.  Do not query it here:
  // DMA-BUF export and provider VA registration are the authoritative probes
  // for this allocation and are exercised independently by the strict CI
  // modes.

  size_t granularity = 0;
  DEVCHECK(hipMemGetAllocationGranularity(
      &granularity, &prop, hipMemAllocationGranularityRecommended));
  size_t allocSize = size;
  ALIGN_SIZE(allocSize, granularity);

  size_t segmentBytes = 0;
  flagcxResult_t segmentResult =
      flagcxVmmSegmentBytes(granularity, &segmentBytes);
  if (segmentResult != flagcxSuccess) {
    *ptr = NULL;
    return segmentResult;
  }
  if (segmentBytes != 0 && segmentBytes < allocSize) {
    // The test hook fails the second map after the first segment is live.
    // Scope it to this allocation so ordinary window maps cannot see it.
    const char *failSetting = std::getenv("FLAGCX_CI_DU_VMM_FAIL_AFTER_MAP");
    if (std::getenv("FLAGCX_CI_REQUIRE_MULTI_SEGMENT") != NULL &&
        failSetting != NULL && failSetting[0] >= '0' && failSetting[0] <= '9') {
      char *end = NULL;
      unsigned long long parsed = std::strtoull(failSetting, &end, 10);
      if (*end == '\0' && parsed < allocSize / segmentBytes + 1)
        duhipFailAfterMap = static_cast<size_t>(parsed);
    }
    flagcxResult_t result = DuhipSegmentAllocation::allocate(
        ptr, allocSize, granularity, segmentBytes, prop);
    duhipFailAfterMap = std::numeric_limits<size_t>::max();
    return result;
  }

  hipMemGenericAllocationHandle_t handle;
  DEVCHECK(hipMemCreate(&handle, allocSize, &prop, 0));
  void *address = nullptr;
  hipError_t result =
      hipMemAddressReserve(&address, allocSize, granularity, 0, 0);
  if (result != hipSuccess) {
    hipMemRelease(handle);
    return flagcxUnhandledDeviceError;
  }
  result = hipMemMap(address, allocSize, 0, handle, 0);
  if (result != hipSuccess) {
    hipMemAddressFree(address, allocSize);
    hipMemRelease(handle);
    return flagcxUnhandledDeviceError;
  }
  hipMemAccessDesc access = {};
  access.location.type = hipMemLocationTypeDevice;
  access.location.id = device;
  access.flags = hipMemAccessFlagsProtReadWrite;
  result = hipMemSetAccess(address, allocSize, &access, 1);
  if (result != hipSuccess) {
    hipMemUnmap(address, allocSize);
    hipMemAddressFree(address, allocSize);
    hipMemRelease(handle);
    return flagcxUnhandledDeviceError;
  }
  *ptr = address;
  bool tracked = false;
  try {
    std::lock_guard<std::mutex> lock(gDuhipVmmAllocationMtx);
    tracked = gDuhipVmmAllocations
                  .emplace(*ptr, DuhipVmmAllocation{handle, allocSize, true,
                                                    true, true})
                  .second;
  } catch (const std::bad_alloc &) {
  }
  if (!tracked) {
    hipMemUnmap(address, allocSize);
    hipMemAddressFree(address, allocSize);
    hipMemRelease(handle);
    *ptr = NULL;
    return flagcxSystemError;
  }
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return flagcxSuccess;
  }
  bool segmented = false;
  flagcxResult_t segmentResult =
      DuhipSegmentAllocation::freeAllocation(ptr, &segmented);
  if (segmented || segmentResult != flagcxSuccess)
    return segmentResult;
  std::lock_guard<std::mutex> lock(gDuhipVmmAllocationMtx);
  auto it = gDuhipVmmAllocations.find(ptr);
  if (it == gDuhipVmmAllocations.end()) {
    DEVCHECK(hipFree(ptr));
    return flagcxSuccess;
  }
  DuhipVmmAllocation &allocation = it->second;
  if (allocation.mappingOwned) {
    if (hipMemUnmap((hipDeviceptr_t)ptr, allocation.size) != hipSuccess)
      return flagcxUnhandledDeviceError;
    allocation.mappingOwned = false;
  }
  if (allocation.vaOwned) {
    if (hipMemAddressFree((hipDeviceptr_t)ptr, allocation.size) != hipSuccess)
      return flagcxUnhandledDeviceError;
    allocation.vaOwned = false;
  }
  // Keep the handle returned by hipMemCreate alive for the whole allocation
  // lifetime.  The DCU compatibility layer does not reliably preserve the
  // backing allocation across repeated retain/release cycles when only VA
  // mappings hold references. Window registrations borrow this root handle;
  // only VMM allocations created outside this adaptor retain a private handle.
  if (allocation.handleOwned) {
    if (hipMemRelease(allocation.handle) != hipSuccess)
      return flagcxUnhandledDeviceError;
    allocation.handleOwned = false;
  }
  gDuhipVmmAllocations.erase(it);
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorStreamCreate(flagcxStream_t *stream) {
  (*stream) = NULL;
  flagcxCalloc(stream, 1);
  DEVCHECK(
      hipStreamCreateWithFlags((hipStream_t *)(*stream), hipStreamNonBlocking));
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorStreamDestroy(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(hipStreamDestroy(stream->base));
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorStreamCopy(flagcxStream_t *newStream,
                                      void *oldStream) {
  (*newStream) = NULL;
  flagcxCalloc(newStream, 1);
  (*newStream)->base = (hipStream_t)oldStream;
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorStreamFree(flagcxStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorStreamSynchronize(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(hipStreamSynchronize(stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorStreamQuery(flagcxStream_t stream) {
  flagcxResult_t res = flagcxSuccess;
  if (stream != NULL) {
    hipError_t error = hipStreamQuery(stream->base);
    if (error == hipSuccess) {
      res = flagcxSuccess;
    } else if (error == hipErrorNotReady) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

flagcxResult_t duhipAdaptorStreamWaitEvent(flagcxStream_t stream,
                                           flagcxEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(
        hipStreamWaitEvent(stream->base, event->base, hipEventWaitDefault));
  }
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorEventCreate(flagcxEvent_t *event,
                                       flagcxEventType_t eventType) {
  (*event) = NULL;
  flagcxCalloc(event, 1);
  const unsigned int flags = (eventType == flagcxEventDefault)
                                 ? hipEventDefault
                                 : hipEventDisableTiming;
  DEVCHECK(hipEventCreateWithFlags(&((*event)->base), flags));
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorEventDestroy(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(hipEventDestroy(event->base));
    free(event);
    event = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorEventRecord(flagcxEvent_t event,
                                       flagcxStream_t stream) {
  if (event != NULL) {
    if (stream != NULL) {
      DEVCHECK(hipEventRecordWithFlags(event->base, stream->base,
                                       hipEventRecordDefault));
    } else {
      DEVCHECK(hipEventRecordWithFlags(event->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorEventSynchronize(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(hipEventSynchronize(event->base));
  }
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorEventQuery(flagcxEvent_t event) {
  flagcxResult_t res = flagcxSuccess;
  if (event != NULL) {
    hipError_t error = hipEventQuery(event->base);
    if (error == hipSuccess) {
      res = flagcxSuccess;
    } else if (error == hipErrorNotReady) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

flagcxResult_t duhipAdaptorIpcMemHandleCreate(flagcxIpcMemHandle_t *handle,
                                              size_t *size) {
  flagcxCalloc(handle, 1);
  if (size != NULL) {
    *size = sizeof(hipIpcMemHandle_t);
  }
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorIpcMemHandleGet(flagcxIpcMemHandle_t handle,
                                           void *devPtr) {
  if (handle == NULL || devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(hipIpcGetMemHandle(&handle->base, devPtr));
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorIpcMemHandleOpen(flagcxIpcMemHandle_t handle,
                                            void **devPtr) {
  if (handle == NULL || devPtr == NULL || *devPtr != NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(
      hipIpcOpenMemHandle(devPtr, handle->base, hipIpcMemLazyEnablePeerAccess));
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorIpcMemHandleClose(void *devPtr) {
  if (devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(hipIpcCloseMemHandle(devPtr));
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorIpcMemHandleFree(flagcxIpcMemHandle_t handle) {
  if (handle != NULL) {
    free(handle);
  }
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorLaunchHostFunc(flagcxStream_t stream,
                                          void (*fn)(void *), void *args) {
  if (stream != NULL) {
    DEVCHECK(hipLaunchHostFunc(stream->base, fn, args));
  }
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorDmaSupport(bool *dmaBufferSupport) {
  if (dmaBufferSupport == NULL)
    return flagcxInvalidArgument;

  // Enum 124 is also unsupported by the DCU compatibility layer and aborts
  // the process.  Advertise the implemented export path and let
  // hipMemGetHandleForAddressRange provide the per-allocation result.  Auto
  // mode falls back to VA only on an explicit flagcxNotSupported result.
  *dmaBufferSupport = true;
  return flagcxSuccess;
}

flagcxResult_t
duhipAdaptorMemGetHandleForAddressRange(void *handleOut, void *buffer,
                                        size_t size, unsigned long long flags) {
  if (handleOut == NULL || buffer == NULL || size == 0)
    return flagcxInvalidArgument;
  hipError_t result = hipMemGetHandleForAddressRange(
      handleOut, (hipDeviceptr_t)buffer, size,
      HIP_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, flags);
  if (result == hipErrorNotSupported)
    return flagcxNotSupported;
  return result == hipSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

flagcxResult_t duhipAdaptorGetDeviceProperties(struct flagcxDevProps *props,
                                               int dev) {
  if (props == NULL) {
    return flagcxInvalidArgument;
  }

  hipDeviceProp_t devProp;
  DEVCHECK(hipGetDeviceProperties(&devProp, dev));
  strncpy(props->name, devProp.name, sizeof(props->name) - 1);
  props->name[sizeof(props->name) - 1] = '\0';
  props->pciBusId = devProp.pciBusID;
  props->pciDeviceId = devProp.pciDeviceID;
  props->pciDomainId = devProp.pciDomainID;
  // HIP device properties do not provide a portable GDR support field.
  // props->gdrSupported = devProp.gpuDirectRDMASupported;

  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorGetDevicePciBusId(char *pciBusId, int len, int dev) {
  if (pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(hipDeviceGetPCIBusId(pciBusId, len, dev));
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorGetDeviceByPciBusId(int *dev, const char *pciBusId) {
  if (dev == NULL || pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(hipDeviceGetByPCIBusId(dev, pciBusId));
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorStreamWaitValue64(flagcxStream_t stream, void *addr,
                                             uint64_t value, int flags) {
  if (stream == NULL || addr == NULL)
    return flagcxInvalidArgument;
  if (flags & ~FLAGCX_STREAM_WAIT_VALUE_FLUSH_REMOTE_WRITES)
    return flagcxInvalidArgument;

  // The current DU driver supports ordinary stream memory waits but not the
  // acquire/flush guarantee required after a NIC or peer device publishes the
  // waited-on value. Report the missing capability without invoking a driver
  // flag that can surface as a generic error on this HIP-compatible runtime.
  if (flags & FLAGCX_STREAM_WAIT_VALUE_FLUSH_REMOTE_WRITES)
    return flagcxNotSupported;

  unsigned int waitFlags = hipStreamWaitValueGte;

  hipStream_t hipStream = (hipStream_t)(stream->base);
  hipError_t err =
      hipStreamWaitValue64(hipStream, (hipDeviceptr_t)addr, value, waitFlags);
  if (err == hipSuccess)
    return flagcxSuccess;
  if (err == hipErrorNotSupported)
    return flagcxNotSupported;
  return flagcxUnhandledDeviceError;
}
flagcxResult_t duhipAdaptorStreamWriteValue64(flagcxStream_t stream, void *addr,
                                              uint64_t value, int flags) {
  (void)flags;
  if (stream == NULL || addr == NULL)
    return flagcxInvalidArgument;
  hipStream_t hipStream = (hipStream_t)(stream->base);
  hipError_t err =
      hipStreamWriteValue64(hipStream, (hipDeviceptr_t)addr, value, 0);
  return (err == hipSuccess) ? flagcxSuccess : flagcxUnhandledDeviceError;
}
flagcxResult_t duhipAdaptorEventElapsedTime(float *ms, flagcxEvent_t start,
                                            flagcxEvent_t end) {
  if (ms == NULL || start == NULL || end == NULL) {
    return flagcxInvalidArgument;
  }
  hipError_t error = hipEventElapsedTime(ms, start->base, end->base);
  if (error == hipSuccess) {
    return flagcxSuccess;
  } else if (error == hipErrorNotReady) {
    return flagcxInProgress;
  } else {
    return flagcxUnhandledDeviceError;
  }
}

flagcxResult_t duhipAdaptorHostRegister(void *ptr, size_t size) {
  if (ptr == NULL || size == 0) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(hipHostRegister(ptr, size, hipHostRegisterMapped));
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorHostUnregister(void *ptr) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(hipHostUnregister(ptr));
  return flagcxSuccess;
}

// Symmetric memory VMM handle exchange and flat mapping.
static flagcxResult_t
duhipSymPhysHandleDestroy(DuhipSymPhysHandle *physHandle) {
  if (physHandle == NULL)
    return flagcxSuccess;
  if (physHandle->releaseOwned) {
    hipError_t result = hipMemRelease(physHandle->handle);
    if (result != hipSuccess) {
      WARN("DU VMM physical handle release failed: result=%d "
           "allocationBase=%p size=%zu",
           (int)result, (void *)physHandle->allocationBase,
           physHandle->allocationSize);
      return flagcxUnhandledDeviceError;
    }
    physHandle->releaseOwned = false;
  }
  free(physHandle);
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorSymPhysAlloc(void *ptr, size_t size,
                                        void **physHandle,
                                        void *shareableHandle,
                                        size_t *handleSize, size_t *allocSize) {
  if (ptr == NULL || physHandle == NULL || shareableHandle == NULL ||
      handleSize == NULL || allocSize == NULL)
    return flagcxInvalidArgument;
  *physHandle = NULL;
  *allocSize = 0;
  if (*handleSize < sizeof(int))
    return flagcxInvalidArgument;

  // Discover actual physical allocation size (already granularity-aligned)
  hipDeviceptr_t allocationBase = 0;
  size_t actualAllocSize = 0;
  hipError_t result = hipMemGetAddressRange(&allocationBase, &actualAllocSize,
                                            (hipDeviceptr_t)ptr);
  if (result != hipSuccess)
    return flagcxUnhandledDeviceError;
  const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  const uintptr_t baseAddress = reinterpret_cast<uintptr_t>(allocationBase);
  if (allocationBase == nullptr || actualAllocSize == 0 ||
      address < baseAddress || address - baseAddress > actualAllocSize ||
      size > actualAllocSize - (address - baseAddress))
    return flagcxInvalidUsage;

  DuhipSymPhysHandle *handle =
      (DuhipSymPhysHandle *)malloc(sizeof(DuhipSymPhysHandle));
  if (handle == NULL)
    return flagcxSystemError;
  handle->allocationBase = allocationBase;
  handle->allocationSize = actualAllocSize;
  handle->releaseOwned = false;

  bool rootAllocationFound = false;
  {
    std::lock_guard<std::mutex> lock(gDuhipVmmAllocationMtx);
    auto it = gDuhipVmmAllocations.find((void *)allocationBase);
    if (it != gDuhipVmmAllocations.end()) {
      const DuhipVmmAllocation &allocation = it->second;
      if (allocation.size != actualAllocSize || !allocation.mappingOwned ||
          !allocation.vaOwned || !allocation.handleOwned) {
        free(handle);
        return flagcxInvalidUsage;
      }
      // The allocation registry owns the handle returned by hipMemCreate for
      // the whole allocation lifetime.  Borrow it for window registration;
      // repeatedly retaining and releasing this root handle invalidates the
      // backing allocation on GalaxyHIP after several windows.
      handle->handle = allocation.handle;
      rootAllocationFound = true;
    }
  }
  if (!rootAllocationFound) {
    // VMM allocations not created by duhipAdaptorGdrMemAlloc still require a
    // private retained reference, released when this window is destroyed.
    result = hipMemRetainAllocationHandle(&handle->handle, ptr);
    if (result != hipSuccess) {
      free(handle);
      if (result == hipErrorInvalidValue || result == hipErrorNotSupported)
        return flagcxNotSupported;
      return flagcxUnhandledDeviceError;
    }
    handle->releaseOwned = true;
  }

  // Export as POSIX fd for IPC sharing.
  result = hipMemExportToShareableHandle(
      shareableHandle, handle->handle, hipMemHandleTypePosixFileDescriptor, 0);
  if (result != hipSuccess) {
    flagcxResult_t cleanupResult = duhipSymPhysHandleDestroy(handle);
    if (cleanupResult != flagcxSuccess) {
      // Preserve the retained provider resource for the common rollback path.
      *physHandle = handle;
      return cleanupResult;
    }
    return flagcxUnhandledDeviceError;
  }
  *allocSize = actualAllocSize;
  *handleSize = sizeof(int); // POSIX fd is an int
  *physHandle = handle;
  return flagcxSuccess;
}
flagcxResult_t duhipAdaptorSymPhysFree(void *physHandle) {
  return duhipSymPhysHandleDestroy((DuhipSymPhysHandle *)physHandle);
}
flagcxResult_t duhipAdaptorSymFlatMappingUnmap(void *flatBase, size_t allocSize,
                                               int nPeers);
flagcxResult_t duhipAdaptorSymFlatVaFree(void *flatBase, size_t allocSize,
                                         int nPeers);
flagcxResult_t duhipAdaptorSymFlatMap(void *peerHandles[], int nPeers,
                                      int selfIndex, void *selfPhysHandle,
                                      size_t allocSize, void **flatBase) {
  if (peerHandles == NULL || selfPhysHandle == NULL || flatBase == NULL ||
      nPeers <= 0 || selfIndex < 0 || selfIndex >= nPeers || allocSize == 0 ||
      allocSize > std::numeric_limits<size_t>::max() / (size_t)nPeers)
    return flagcxInvalidArgument;
  *flatBase = NULL;

  hipMemGenericAllocationHandle_t selfHandle =
      ((DuhipSymPhysHandle *)selfPhysHandle)->handle;

  // allocSize is already granularity-aligned (from hipMemGetAddressRange)
  size_t totalSize = allocSize * nPeers;

  DuhipFlatMapping initialMapping;
  try {
    initialMapping.allocSize = allocSize;
    initialMapping.nPeers = nPeers;
    initialMapping.mappedSlots.assign(nPeers, 0);
    initialMapping.importedHandles.resize(nPeers);
    initialMapping.importedHandleOwned.assign(nPeers, 0);
    initialMapping.vaOwned = true;
  } catch (const std::bad_alloc &) {
    return flagcxSystemError;
  }

  // Reserve the full VA range
  void *base = nullptr;
  hipError_t result = hipMemAddressReserve(&base, totalSize, 0, 0, 0);
  if (result != hipSuccess) {
    WARN("DU VMM flat map failed: stage=address-reserve result=%d "
         "allocSize=%zu nPeers=%d",
         (int)result, allocSize, nPeers);
    return flagcxUnhandledDeviceError;
  }

  // Publish adaptor-private ownership before the first fallible map step.  On
  // failure the common symmetric-window state machine receives flatBase and
  // calls the split mapping/VA cleanup callbacks; it does not need to know
  // which DU slots were mapped successfully.
  std::unique_lock<std::mutex> lock(gDuhipFlatMappingMtx);
  bool tracked = false;
  try {
    tracked =
        gDuhipFlatMappings.emplace((void *)base, std::move(initialMapping))
            .second;
  } catch (const std::bad_alloc &) {
  }
  if (!tracked) {
    lock.unlock();
    result = hipMemAddressFree(base, totalSize);
    if (result != hipSuccess)
      WARN("DU VMM flat map cleanup failed: stage=address-free result=%d "
           "base=%p size=%zu",
           (int)result, (void *)base, totalSize);
    return flagcxSystemError;
  }
  DuhipFlatMapping &mapping = gDuhipFlatMappings.at((void *)base);
  *flatBase = (void *)base;

  // Import and map each peer's physical memory
  int hipDev;
  hipError_t hipResult = hipGetDevice(&hipDev);
  if (hipResult != hipSuccess) {
    WARN("DU VMM flat map failed: stage=get-device result=%d base=%p",
         (int)hipResult, (void *)base);
    return flagcxUnhandledDeviceError;
  }
  hipMemAccessDesc accessDesc = {};
  accessDesc.location.type = hipMemLocationTypeDevice;
  accessDesc.location.id = hipDev;
  accessDesc.flags = hipMemAccessFlagsProtReadWrite;

  for (int i = 0; i < nPeers; i++) {
    hipMemGenericAllocationHandle_t peerHandle;
    if (i == selfIndex) {
      peerHandle = selfHandle;
    } else {
      int fd = *(int *)peerHandles[i];
      result =
          hipMemImportFromShareableHandle(&peerHandle, (void *)(uintptr_t)fd,
                                          hipMemHandleTypePosixFileDescriptor);
      if (result != hipSuccess) {
        WARN("DU VMM flat map failed: stage=import result=%d base=%p "
             "slot=%d fd=%d",
             (int)result, (void *)base, i, fd);
        return flagcxUnhandledDeviceError;
      }
      mapping.importedHandles[i] = peerHandle;
      mapping.importedHandleOwned[i] = 1;
    }
    void *slot = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(base) +
                                          static_cast<size_t>(i) * allocSize);
    result = hipMemMap(slot, allocSize, 0, peerHandle, 0);
    if (result != hipSuccess) {
      WARN("DU VMM flat map failed: stage=map result=%d base=%p slot=%d "
           "slotAddress=%p size=%zu",
           (int)result, (void *)base, i, (void *)slot, allocSize);
      return flagcxUnhandledDeviceError;
    }
    mapping.mappedSlots[i] = 1;
    result = hipMemSetAccess(slot, allocSize, &accessDesc, 1);
    if (result != hipSuccess) {
      WARN("DU VMM flat map failed: stage=set-access result=%d base=%p "
           "slot=%d slotAddress=%p size=%zu",
           (int)result, (void *)base, i, (void *)slot, allocSize);
      return flagcxUnhandledDeviceError;
    }
  }

  return flagcxSuccess;
}
flagcxResult_t duhipAdaptorSymFlatUnmap(void *flatBase, size_t allocSize,
                                        int nPeers) {
  flagcxResult_t result =
      duhipAdaptorSymFlatMappingUnmap(flatBase, allocSize, nPeers);
  return result == flagcxSuccess
             ? duhipAdaptorSymFlatVaFree(flatBase, allocSize, nPeers)
             : result;
}
flagcxResult_t duhipAdaptorSymFlatMappingUnmap(void *flatBase, size_t allocSize,
                                               int nPeers) {
  if (flatBase == NULL)
    return flagcxSuccess;

  std::lock_guard<std::mutex> lock(gDuhipFlatMappingMtx);
  auto it = gDuhipFlatMappings.find(flatBase);
  if (it == gDuhipFlatMappings.end()) {
    WARN("DU VMM flat unmap failed: untracked base=%p", flatBase);
    return flagcxInvalidUsage;
  }
  DuhipFlatMapping &mapping = it->second;
  if (!duhipFlatMappingMatches(mapping, allocSize, nPeers)) {
    WARN("DU VMM flat unmap failed: geometry mismatch base=%p "
         "expectedSize=%zu actualSize=%zu expectedPeers=%d actualPeers=%d",
         flatBase, mapping.allocSize, allocSize, mapping.nPeers, nPeers);
    return flagcxInvalidArgument;
  }

  void *base = flatBase;
  for (int i = 0; i < nPeers; i++) {
    if (!mapping.mappedSlots[i])
      continue;
    void *slot = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(base) +
                                          static_cast<size_t>(i) * allocSize);
    hipError_t result = hipMemUnmap(slot, allocSize);
    if (result != hipSuccess) {
      WARN("DU VMM flat unmap failed: stage=slot-unmap result=%d base=%p "
           "slot=%d slotAddress=%p size=%zu",
           (int)result, flatBase, i, (void *)slot, allocSize);
      return flagcxUnhandledDeviceError;
    }
    mapping.mappedSlots[i] = 0;
  }

  // GalaxyHIP does not reliably keep an imported allocation alive after its
  // handle is released while an alias mapping still exists.  Keep every
  // imported handle until all slots have been unmapped.  Returning above on an
  // unmap failure deliberately preserves every handle for a later retry.
  for (int i = 0; i < nPeers; i++) {
    if (!mapping.importedHandleOwned[i])
      continue;
    hipError_t result = hipMemRelease(mapping.importedHandles[i]);
    if (result != hipSuccess) {
      WARN("DU VMM flat unmap failed: stage=release-import result=%d "
           "base=%p slot=%d",
           (int)result, flatBase, i);
      return flagcxUnhandledDeviceError;
    }
    mapping.importedHandleOwned[i] = 0;
  }
  return flagcxSuccess;
}
flagcxResult_t duhipAdaptorSymFlatVaFree(void *flatBase, size_t allocSize,
                                         int nPeers) {
  if (flatBase == NULL)
    return flagcxSuccess;

  std::lock_guard<std::mutex> lock(gDuhipFlatMappingMtx);
  auto it = gDuhipFlatMappings.find(flatBase);
  if (it == gDuhipFlatMappings.end()) {
    WARN("DU VMM flat VA free failed: untracked base=%p", flatBase);
    return flagcxInvalidUsage;
  }
  DuhipFlatMapping &mapping = it->second;
  if (!duhipFlatMappingMatches(mapping, allocSize, nPeers)) {
    WARN("DU VMM flat VA free failed: geometry mismatch base=%p "
         "expectedSize=%zu actualSize=%zu expectedPeers=%d actualPeers=%d",
         flatBase, mapping.allocSize, allocSize, mapping.nPeers, nPeers);
    return flagcxInvalidArgument;
  }
  for (int i = 0; i < nPeers; i++) {
    if (mapping.mappedSlots[i] || mapping.importedHandleOwned[i]) {
      WARN("DU VMM flat VA free blocked by owned slot: base=%p slot=%d "
           "mapped=%d importedHandle=%d",
           flatBase, i, (int)mapping.mappedSlots[i],
           (int)mapping.importedHandleOwned[i]);
      return flagcxInvalidUsage;
    }
  }
  if (mapping.vaOwned) {
    hipError_t result =
        hipMemAddressFree((hipDeviceptr_t)flatBase, allocSize * (size_t)nPeers);
    if (result != hipSuccess) {
      WARN("DU VMM flat VA free failed: stage=address-free result=%d "
           "base=%p size=%zu",
           (int)result, flatBase, allocSize * (size_t)nPeers);
      return flagcxUnhandledDeviceError;
    }
    mapping.vaOwned = false;
  }
  gDuhipFlatMappings.erase(it);
  return flagcxSuccess;
}
flagcxResult_t duhipAdaptorSymMulticastSupported(int *supported) {
  // not supported on dcu
  if (supported == NULL)
    return flagcxInvalidArgument;

  if (supported)
    *supported = 0;
  return flagcxSuccess;
}
flagcxResult_t duhipAdaptorSymMulticastCreate(size_t allocSize,
                                              int nLocalDevices,
                                              const int *localDeviceOrdinals,
                                              void **mcHandle,
                                              int *shareableFd) {
  // not supported on dcu
  if (mcHandle)
    *mcHandle = NULL;

  if (shareableFd)
    *shareableFd = -1;

  return flagcxNotSupported;
}
flagcxResult_t duhipAdaptorSymMulticastBind(void *mcHandle, int importFd,
                                            void *physHandle, size_t allocSize,
                                            int localRank, int nLocalDevices,
                                            void **mcBase, size_t *mcMapSize) {
  // not supported on dcu
  if (mcBase)
    *mcBase = NULL;

  if (mcMapSize)
    *mcMapSize = 0;

  return flagcxNotSupported;
}
flagcxResult_t duhipAdaptorSymMulticastTeardown(void *mcBase,
                                                size_t mcMapSize) {
  // not supported on dcu
  return flagcxSuccess;
}
flagcxResult_t duhipAdaptorSymMulticastMappingUnmap(void *mcBase,
                                                    size_t mcMapSize) {
  if (mcBase == NULL)
    return flagcxSuccess;
  DEVCHECK(hipMemUnmap((hipDeviceptr_t)mcBase, mcMapSize));
  return flagcxSuccess;
}
flagcxResult_t duhipAdaptorSymMulticastVaFree(void *mcBase, size_t mcMapSize) {
  if (mcBase == NULL)
    return flagcxSuccess;
  DEVCHECK(hipMemAddressFree((hipDeviceptr_t)mcBase, mcMapSize));
  return flagcxSuccess;
}
flagcxResult_t duhipAdaptorSymMulticastFree(void *mcHandle) {
  // not supported on dcu
  return flagcxSuccess;
}
flagcxResult_t duhipAdaptorSymMulticastImport(int, void **mcHandle) {
  if (mcHandle != NULL)
    *mcHandle = NULL;
  return flagcxNotSupported;
}

flagcxResult_t duhipAdaptorGetAddressRange(const void *ptr, void **base,
                                           size_t *size) {
  if (ptr == NULL || base == NULL || size == NULL)
    return flagcxInvalidArgument;

  hipDeviceptr_t allocationBase = 0;
  hipError_t result =
      hipMemGetAddressRange(&allocationBase, size, (hipDeviceptr_t)ptr);
  if (result != hipSuccess)
    return flagcxUnhandledDeviceError;
  *base = (void *)allocationBase;
  return flagcxSuccess;
}

flagcxResult_t
duhipAdaptorSymPhysDescribe(void *ptr, size_t size, void **segmentBundle,
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

  DuhipSegmentMap::Bundle *bundle =
      new (std::nothrow) DuhipSegmentMap::Bundle();
  if (bundle == NULL)
    return flagcxSystemError;
  *segmentBundle = bundle;
  bool found = false;
  flagcxResult_t result =
      DuhipSegmentAllocation::describe(ptr, size, bundle, userOffset, &found);
  if (result != flagcxSuccess) {
    WARN("DU VMM describe failed: stage=tracked-allocation result=%d size=%zu",
         static_cast<int>(result), size);
    return result;
  }
  if (!found) {
    std::lock_guard<std::mutex> lock(gDuhipVmmAllocationMtx);
    for (std::unordered_map<void *, DuhipVmmAllocation>::const_iterator it =
             gDuhipVmmAllocations.begin();
         it != gDuhipVmmAllocations.end(); ++it) {
      const uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
      const DuhipVmmAllocation &allocation = it->second;
      if (address < base || address - base > allocation.size ||
          size > allocation.size - (address - base))
        continue;
      found = true;
      if (!allocation.mappingOwned || !allocation.vaOwned ||
          !allocation.handleOwned)
        return flagcxInvalidUsage;
      try {
        bundle->segments.push_back(flagcxSymSegmentMeta{0, allocation.size});
        bundle->handleOwned.push_back(0);
        bundle->handles.push_back(allocation.handle);
      } catch (const std::bad_alloc &) {
        return flagcxSystemError;
      }
      *userOffset = address - base;
      break;
    }
  }
  if (!found) {
    // A successful legacy IPC export is the only positive proof that an
    // untracked allocation may use IPC. Never retain an ordinary HIP pointer:
    // some DU drivers fault instead of returning an error from that VMM API.
    hipDeviceptr_t allocationBase = NULL;
    size_t allocationSize = 0;
    if (hipMemGetAddressRange(&allocationBase, &allocationSize,
                              (hipDeviceptr_t)ptr) == hipSuccess &&
        address >= reinterpret_cast<uintptr_t>(allocationBase) &&
        address - reinterpret_cast<uintptr_t>(allocationBase) <=
            allocationSize &&
        size <= allocationSize -
                    (address - reinterpret_cast<uintptr_t>(allocationBase))) {
      hipIpcMemHandle_t ipcHandle;
      if (hipIpcGetMemHandle(&ipcHandle, allocationBase) == hipSuccess) {
        delete bundle;
        *segmentBundle = NULL;
      } else {
        hipGetLastError();
      }
    } else {
      hipGetLastError();
    }
    return flagcxNotSupported;
  }

  int device = 0;
  hipError_t error = hipGetDevice(&device);
  if (error != hipSuccess)
    return duhipVmmDeviceError("describe-get-device", error, size);
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = device;
  prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
  error = hipMemGetAllocationGranularity(
      slotAlignment, &prop, hipMemAllocationGranularityRecommended);
  if (error != hipSuccess)
    return duhipVmmDeviceError("describe-granularity", error, size);
  if (*slotAlignment == 0) {
    WARN("DU VMM describe failed: stage=zero-granularity size=%zu", size);
    return flagcxUnhandledDeviceError;
  }
  if (bundle->segments.empty() ||
      bundle->segments.size() >
          static_cast<size_t>(std::numeric_limits<int>::max()))
    return flagcxInvalidUsage;
  *segments = bundle->segments.data();
  *segmentCount = static_cast<int>(bundle->segments.size());
  return flagcxSuccess;
}

flagcxResult_t duhipAdaptorSymPhysExportSegment(void *bundle, int index,
                                                int *fd) {
  return DuhipSegmentMap::exportSegment(bundle, index, fd);
}
flagcxResult_t duhipAdaptorSymPhysFreeSegments(void *bundle) {
  return DuhipSegmentMap::freeBundle(bundle);
}
flagcxResult_t duhipAdaptorSymPhysSegmentCount(const void *ptr, size_t size,
                                               int *count) {
  if (ptr == NULL || size == 0 || count == NULL)
    return flagcxInvalidArgument;
  flagcxResult_t result =
      DuhipSegmentAllocation::segmentCount(ptr, size, count);
  if (result != flagcxNotSupported)
    return result;
  const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  std::lock_guard<std::mutex> lock(gDuhipVmmAllocationMtx);
  for (std::unordered_map<void *, DuhipVmmAllocation>::const_iterator it =
           gDuhipVmmAllocations.begin();
       it != gDuhipVmmAllocations.end(); ++it) {
    const uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
    if (address < base || address - base > it->second.size ||
        size > it->second.size - (address - base))
      continue;
    *count = 1;
    return flagcxSuccess;
  }
  return flagcxNotSupported;
}

flagcxResult_t duhipAdaptorSymFlatMapSegments(
    const int *fds, int nPeers, int self, void *bundle,
    const flagcxSymRankLayout *layouts, const flagcxSymSegmentMeta *segments,
    int totalSegments, size_t stride, size_t alignment, void **flatBase) {
  return DuhipSegmentMap::map(fds, nPeers, self, bundle, layouts, segments,
                              totalSegments, stride, alignment, flatBase);
}
flagcxResult_t duhipAdaptorSymFlatMapSegmentsBegin(
    int nPeers, int self, void *bundle, const flagcxSymRankLayout *layouts,
    const flagcxSymSegmentMeta *segments, int totalSegments, size_t stride,
    size_t alignment, void **flatBase) {
  return DuhipSegmentMap::begin(nPeers, self, bundle, layouts, segments,
                                totalSegments, stride, alignment, flatBase);
}
flagcxResult_t duhipAdaptorSymFlatMapSegmentOne(void *flatBase, int index,
                                                int fd) {
  return DuhipSegmentMap::mapOne(flatBase, index, fd);
}
flagcxResult_t duhipAdaptorSymFlatMappingUnmapSegments(void *flatBase) {
  return DuhipSegmentMap::unmap(flatBase);
}
flagcxResult_t duhipAdaptorSymFlatVaFreeSegments(void *flatBase) {
  return DuhipSegmentMap::freeVa(flatBase);
}
flagcxResult_t duhipAdaptorSymLsaGetCapacity(size_t *totalBytes,
                                             size_t *alignment) {
  if (totalBytes == NULL || alignment == NULL)
    return flagcxInvalidArgument;
  size_t freeBytes = 0;
  hipError_t error = hipMemGetInfo(&freeBytes, totalBytes);
  if (error != hipSuccess)
    return duhipVmmDeviceError("lsa-get-capacity", error);
  // The common LSA path raises this floor to the current window's VMM
  // allocation granularity before reserving or mapping its VA.
  *alignment = 1024;
  return flagcxSuccess;
}
flagcxResult_t duhipAdaptorSymLsaVaReserve(size_t size, size_t alignment,
                                           void **base) {
  if (base == NULL)
    return flagcxInvalidArgument;
  *base = NULL;
  hipError_t error = hipMemAddressReserve(base, size, alignment, NULL, 0);
  if (error == hipSuccess)
    return flagcxSuccess;
  duhipVmmDeviceError("lsa-address-reserve", error, size, alignment);
  return error == hipErrorOutOfMemory ? flagcxNotSupported
                                      : flagcxUnhandledDeviceError;
}
flagcxResult_t duhipAdaptorSymLsaVaFree(void *base, size_t size) {
  return DuhipSegmentOps::freeAddress(reinterpret_cast<uintptr_t>(base), size);
}
flagcxResult_t duhipAdaptorSymLsaMapBeginAt(
    void *windowBase, int nPeers, int self, void *bundle,
    const flagcxSymRankLayout *layouts, const flagcxSymSegmentMeta *segments,
    int totalSegments, size_t stride, size_t alignment, void **mappingBase) {
  return DuhipSegmentMap::beginAt(windowBase, nPeers, self, bundle, layouts,
                                  segments, totalSegments, stride, alignment,
                                  mappingBase);
}

struct flagcxDeviceAdaptor duhipAdaptor {
  "DUHIP",
      // Basic functions
      duhipAdaptorDeviceSynchronize, duhipAdaptorDeviceMemcpy,
      duhipAdaptorDeviceMemset, duhipAdaptorDeviceMalloc,
      duhipAdaptorDeviceFree, duhipAdaptorSetDevice, duhipAdaptorGetDevice,
      duhipAdaptorGetDeviceCount, duhipAdaptorGetVendor,
      duhipAdaptorHostGetDevicePointer,
      // GDR functions
      NULL, // flagcxResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // flagcxResult_t (*memHandleDestroy)(int dev, void *memHandle);
      duhipAdaptorGdrMemAlloc, duhipAdaptorGdrMemFree,
      NULL, // flagcxResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // flagcxResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      NULL, // flagcxResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t
            // sz);
      NULL, // flagcxResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);
      // Stream functions
      duhipAdaptorStreamCreate, duhipAdaptorStreamDestroy,
      duhipAdaptorStreamCopy, duhipAdaptorStreamFree,
      duhipAdaptorStreamSynchronize, duhipAdaptorStreamQuery,
      duhipAdaptorStreamWaitEvent, duhipAdaptorStreamWaitValue64,
      duhipAdaptorStreamWriteValue64,
      // Event functions
      duhipAdaptorEventCreate, duhipAdaptorEventDestroy,
      duhipAdaptorEventRecord, duhipAdaptorEventSynchronize,
      duhipAdaptorEventQuery, duhipAdaptorEventElapsedTime,
      // IpcMemHandle functions
      duhipAdaptorIpcMemHandleCreate, duhipAdaptorIpcMemHandleGet,
      duhipAdaptorIpcMemHandleOpen, duhipAdaptorIpcMemHandleClose,
      duhipAdaptorIpcMemHandleFree,
      // Kernel launch
      NULL, // flagcxResult_t (*launchKernel)(void *func, unsigned int block_x,
            // unsigned int block_y, unsigned int block_z, unsigned int grid_x,
            // unsigned int grid_y, unsigned int grid_z, void **args, size_t
            // share_mem, void *stream, void *memHandle);
      NULL, // flagcxResult_t (*copyArgsInit)(void **args);
      NULL, // flagcxResult_t (*copyArgsFree)(void *args);
      NULL, // flagcxResult_t
            // (*launchDeviceFunc)(flagcxStream_t stream,
            // void *args);
      // Others
      duhipAdaptorGetDeviceProperties, // flagcxResult_t
                                       // (*getDeviceProperties)(struct
                                       // flagcxDevProps *props, int dev);
      duhipAdaptorGetDevicePciBusId,   // flagcxResult_t
                                       // (*getDevicePciBusId)(char *pciBusId,
                                       // int len, int dev);
      duhipAdaptorGetDeviceByPciBusId, // flagcxResult_t
                                       // (*getDeviceByPciBusId)(
                                       // int
                                       // *dev, const char *pciBusId);
      duhipAdaptorLaunchHostFunc,
      // DMA buffer
      duhipAdaptorDmaSupport, // flagcxResult_t (*dmaSupport)(bool
                              // *dmaBufferSupport);
      duhipAdaptorMemGetHandleForAddressRange, // flagcxResult_t
                                               // (*memGetHandleForAddressRange)(void
                                               // *handleOut, void *buffer,
                                               // size_t size, unsigned long
                                               // long flags);
      duhipAdaptorHostRegister,   // flagcxResult_t (*hostRegister)(void *,
                                  // size_t);
      duhipAdaptorHostUnregister, // flagcxResult_t (*hostUnregister)(void *);
      // Symmetric memory VMM functions
      duhipAdaptorSymPhysAlloc, duhipAdaptorSymPhysFree, duhipAdaptorSymFlatMap,
      duhipAdaptorSymFlatUnmap, duhipAdaptorSymMulticastSupported,
      duhipAdaptorSymMulticastCreate, duhipAdaptorSymMulticastBind,
      duhipAdaptorSymMulticastTeardown, duhipAdaptorSymMulticastFree,
      NULL, // flagcxResult_t (*getLastError)();
      duhipAdaptorGetPointerType, duhipAdaptorGetAddressRange,
      FLAGCX_VMM_MR_CAP_DMABUF | FLAGCX_VMM_MR_CAP_VA,
      FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE, duhipAdaptorSymMulticastImport,
      duhipAdaptorSymFlatMappingUnmap, duhipAdaptorSymFlatVaFree,
      duhipAdaptorSymMulticastMappingUnmap, duhipAdaptorSymMulticastVaFree,
      // Hygon requires the local post-READ flush for ordinary allocations as
      // well as VMM. Its current DU runtime cannot provide stream acquire for
      // incoming WRITEs, so strong WRITE consumers fail safely as unsupported.
      FLAGCX_GDR_READ_REQUIRES_FLUSH | FLAGCX_GDR_WRITE_REQUIRES_FLUSH,
      FLAGCX_GDR_DEVICE_DU, NULL, duhipAdaptorCanAccessPeer,
      duhipAdaptorSymPhysDescribe, duhipAdaptorSymPhysExportSegment,
      duhipAdaptorSymPhysFreeSegments, duhipAdaptorSymFlatMapSegments,
      duhipAdaptorSymFlatMappingUnmapSegments,
      duhipAdaptorSymFlatVaFreeSegments, duhipAdaptorSymPhysSegmentCount,
      duhipAdaptorSymFlatMapSegmentsBegin, duhipAdaptorSymFlatMapSegmentOne,
      duhipAdaptorSymLsaGetCapacity, duhipAdaptorSymLsaVaReserve,
      duhipAdaptorSymLsaVaFree, duhipAdaptorSymLsaMapBeginAt, NULL, NULL, NULL,
      NULL,
};

#endif // USE_DU_ADAPTOR
