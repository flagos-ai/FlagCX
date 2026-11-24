#include "musa_adaptor.h"

#ifdef USE_MUSA_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <unistd.h>

std::map<flagcxMemcpyType_t, musaMemcpyKind> memcpy_type_map = {
    {flagcxMemcpyHostToDevice, musaMemcpyHostToDevice},
    {flagcxMemcpyDeviceToHost, musaMemcpyDeviceToHost},
    {flagcxMemcpyDeviceToDevice, musaMemcpyDeviceToDevice},
};

flagcxResult_t musaAdaptorDeviceSynchronize() {
  DEVCHECK(musaDeviceSynchronize());
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorGetLastError() {
  DEVCHECK(musaGetLastError());
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
                                       flagcxMemcpyType_t type,
                                       flagcxStream_t stream, void *args) {
  if (stream == NULL) {
    DEVCHECK(musaMemcpy(dst, src, size, memcpy_type_map[type]));
  } else {
    DEVCHECK(
        musaMemcpyAsync(dst, src, size, memcpy_type_map[type], stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorDeviceMemset(void *ptr, int value, size_t size,
                                       flagcxMemType_t type,
                                       flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    memset(ptr, value, size);
  } else {
    if (stream == NULL) {
      DEVCHECK(musaMemset(ptr, value, size));
    } else {
      DEVCHECK(musaMemsetAsync(ptr, value, size, stream->base));
    }
  }
  return flagcxSuccess;
}

// DMA-BUF requires the native allocation itself to be page aligned.
// Keep rejected native allocations alive while filling their alignment gap;
// never return an interior pointer (IPC and musaFree need the original base).
static flagcxResult_t musaAllocExportable(void **ptr, size_t size) {
  if (ptr == nullptr)
    return flagcxInvalidArgument;
  *ptr = nullptr;
  const char *enabled = std::getenv("FLAGCX_DMABUF_ENABLE");
  if (size == 0 || enabled == nullptr || std::strcmp(enabled, "1") != 0) {
    DEVCHECK(musaMalloc(ptr, size));
    return flagcxSuccess;
  }
  const long nativePage = sysconf(_SC_PAGESIZE);
  if (nativePage <= 0)
    return flagcxSystemError;
  const size_t page = static_cast<size_t>(nativePage);
  if (size > std::numeric_limits<size_t>::max() - (page - 1))
    return flagcxInvalidArgument;
  const size_t rounded = ((size + page - 1) / page) * page;
  static std::mutex allocationMutex;
  std::lock_guard<std::mutex> lock(allocationMutex);
  void *held[32] = {};
  int count = 0;
  musaError_t status = musaSuccess;
  for (int attempt = 0; attempt < 16; ++attempt) {
    void *candidate = nullptr;
    status = musaMalloc(&candidate, rounded);
    if (status != musaSuccess)
      break;
    const size_t remainder = reinterpret_cast<uintptr_t>(candidate) % page;
    if (remainder == 0) {
      *ptr = candidate;
      break;
    }
    held[count++] = candidate;
    void *padding = nullptr;
    status = musaMalloc(&padding, page - remainder);
    if (status != musaSuccess)
      break;
    held[count++] = padding;
  }
  for (int i = 0; i < count; ++i) {
    const musaError_t freed = musaFree(held[i]);
    if (status == musaSuccess && freed != musaSuccess)
      status = freed;
  }
  if (status != musaSuccess) {
    if (*ptr != nullptr) {
      musaFree(*ptr);
      *ptr = nullptr;
    }
    return flagcxUnhandledDeviceError;
  }
  return *ptr != nullptr ? flagcxSuccess : flagcxNotSupported;
}

flagcxResult_t musaAdaptorDeviceMalloc(void **ptr, size_t size,
                                       flagcxMemType_t type,
                                       flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    // FIFOs and semaphores need a device-visible alias, not only pinned memory.
    DEVCHECK(musaHostAlloc(ptr, size, musaHostAllocMapped));
  } else if (type == flagcxMemManaged) {
    DEVCHECK(musaMallocManaged(ptr, size, musaMemAttachGlobal));
  } else {
    if (stream == NULL) {
      FLAGCXCHECK(musaAllocExportable(ptr, size));
    } else {
      FLAGCXCHECK(musaAllocExportable(ptr, size));
      // MUSA currently does not support async malloc
      // DEVCHECK(musaMallocAsync(ptr, size, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorDeviceFree(void *ptr, flagcxMemType_t type,
                                     flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    DEVCHECK(musaFreeHost(ptr));
  } else if (type == flagcxMemManaged) {
    DEVCHECK(musaFree(ptr));
  } else {
    if (stream == NULL) {
      DEVCHECK(musaFree(ptr));
    } else {
      DEVCHECK(musaFree(ptr));
      // MUSA currently does not support async malloc
      // DEVCHECK(musaFreeAsync(ptr, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorSetDevice(int dev) {
  DEVCHECK(musaSetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorGetDevice(int *dev) {
  DEVCHECK(musaGetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorGetDeviceCount(int *count) {
  DEVCHECK(musaGetDeviceCount(count));
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "MUSA");
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorHostGetDevicePointer(void **pDevice, void *pHost) {
  if (pDevice == nullptr)
    return flagcxInvalidArgument;
  *pDevice = nullptr;
  if (pHost == nullptr)
    return flagcxInvalidArgument;
  DEVCHECK(musaHostGetDevicePointer(pDevice, pHost, 0));
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorGdrMemAlloc(void **ptr, size_t size,
                                      void *memHandle) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
  FLAGCXCHECK(musaAllocExportable(ptr, size));
  musaPointerAttributes attrs;
  DEVCHECK(musaPointerGetAttributes(&attrs, *ptr));
  unsigned flags = 1;
  DEVCHECK(muPointerSetAttribute(&flags, MU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                 (MUdeviceptr)attrs.devicePointer));
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return flagcxSuccess;
  }
  DEVCHECK(musaFree(ptr));
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorStreamCreate(flagcxStream_t *stream) {
  (*stream) = NULL;
  flagcxCalloc(stream, 1);
  DEVCHECK(musaStreamCreateWithFlags((musaStream_t *)(*stream),
                                     musaStreamNonBlocking));
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorStreamDestroy(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(musaStreamDestroy(stream->base));
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorStreamCopy(flagcxStream_t *newStream,
                                     void *oldStream) {
  (*newStream) = NULL;
  flagcxCalloc(newStream, 1);
  (*newStream)->base = (musaStream_t)oldStream;
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorStreamFree(flagcxStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorStreamSynchronize(flagcxStream_t stream) {
  if (stream != NULL) {
    const musaError_t error = musaStreamSynchronize(stream->base);
    if (error != musaSuccess) {
      WARN("musaStreamSynchronize: native=%d (%s)", (int)error,
           musaGetErrorString(error));
      return flagcxUnhandledDeviceError;
    }
  }
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorStreamQuery(flagcxStream_t stream) {
  flagcxResult_t res = flagcxSuccess;
  if (stream != NULL) {
    musaError error = musaStreamQuery(stream->base);
    if (error == musaSuccess) {
      res = flagcxSuccess;
    } else if (error == musaErrorNotReady) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

flagcxResult_t musaAdaptorStreamWaitEvent(flagcxStream_t stream,
                                          flagcxEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(
        musaStreamWaitEvent(stream->base, event->base, musaEventWaitDefault));
  }
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorEventCreate(flagcxEvent_t *event,
                                      flagcxEventType_t eventType) {
  if (event == NULL)
    return flagcxInvalidArgument;
  *event = NULL;
  flagcxEvent_t wrapper = (flagcxEvent_t)calloc(1, sizeof(struct flagcxEvent));
  if (wrapper == NULL)
    return flagcxSystemError;
  const unsigned int flags = (eventType == flagcxEventDefault)
                                 ? musaEventDefault
                                 : musaEventDisableTiming;
  if (musaEventCreateWithFlags(&wrapper->base, flags) != musaSuccess) {
    free(wrapper);
    return flagcxUnhandledDeviceError;
  }
  *event = wrapper;
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorEventDestroy(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(musaEventDestroy(event->base));
    free(event);
    event = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorEventRecord(flagcxEvent_t event,
                                      flagcxStream_t stream) {
  if (event != NULL) {
    if (stream != NULL) {
      DEVCHECK(musaEventRecordWithFlags(event->base, stream->base,
                                        musaEventRecordDefault));
    } else {
      DEVCHECK(musaEventRecordWithFlags(event->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorEventSynchronize(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(musaEventSynchronize(event->base));
  }
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorEventQuery(flagcxEvent_t event) {
  flagcxResult_t res = flagcxSuccess;
  if (event != NULL) {
    musaError error = musaEventQuery(event->base);
    if (error == musaSuccess) {
      res = flagcxSuccess;
    } else if (error == musaErrorNotReady) {
      res = flagcxInProgress;
    } else {
      res = flagcxUnhandledDeviceError;
    }
  }
  return res;
}

// The wrapper owns the exported handle bytes, not the device allocation.
flagcxResult_t musaAdaptorIpcMemHandleCreate(flagcxIpcMemHandle_t *handle,
                                             size_t *size) {
  if (handle == nullptr)
    return flagcxInvalidArgument;
  *handle = nullptr;
  if (size != nullptr)
    *size = sizeof(musaIpcMemHandle_t);
  static_assert(sizeof(flagcxIpcMemHandle) == sizeof(musaIpcMemHandle_t),
                "IPC wrapper must contain only the native transferable handle");
  *handle = static_cast<flagcxIpcMemHandle_t>(calloc(1, sizeof(**handle)));
  return *handle != nullptr ? flagcxSuccess : flagcxSystemError;
}

flagcxResult_t musaAdaptorIpcMemHandleGet(flagcxIpcMemHandle_t handle,
                                          void *devPtr) {
  if (handle == nullptr || devPtr == nullptr)
    return flagcxInvalidArgument;
  DEVCHECK(musaIpcGetMemHandle(&handle->base, devPtr));
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorIpcMemHandleOpen(flagcxIpcMemHandle_t handle,
                                           void **devPtr) {
  if (devPtr == nullptr)
    return flagcxInvalidArgument;
  *devPtr = nullptr;
  if (handle == nullptr)
    return flagcxInvalidArgument;
  DEVCHECK(musaIpcOpenMemHandle(devPtr, handle->base,
                                musaIpcMemLazyEnablePeerAccess));
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorIpcMemHandleClose(void *devPtr) {
  if (devPtr == nullptr)
    return flagcxInvalidArgument;
  DEVCHECK(musaIpcCloseMemHandle(devPtr));
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorIpcMemHandleFree(flagcxIpcMemHandle_t handle) {
  free(handle);
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorLaunchHostFunc(flagcxStream_t stream,
                                         void (*fn)(void *), void *args) {
  if (stream != NULL) {
    DEVCHECK(musaLaunchHostFunc(stream->base, fn, args));
  }
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorGetDeviceProperties(struct flagcxDevProps *props,
                                              int dev) {
  if (props == NULL) {
    return flagcxInvalidArgument;
  }

  musaDeviceProp devProp;
  DEVCHECK(musaGetDeviceProperties(&devProp, dev));
  strncpy(props->name, devProp.name, sizeof(props->name) - 1);
  props->name[sizeof(props->name) - 1] = '\0';
  props->pciBusId = devProp.pciBusID;
  props->pciDeviceId = devProp.pciDeviceID;
  props->pciDomainId = devProp.pciDomainID;
  // TODO: see if there's another way to get this info. In some musa versions,
  // musaDeviceProp does not have `gpuDirectRDMASupported` field
  // props->gdrSupported = devProp.gpuDirectRDMASupported;

  return flagcxSuccess;
}

flagcxResult_t musaAdaptorGetDevicePciBusId(char *pciBusId, int len, int dev) {
  if (pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(musaDeviceGetPCIBusId(pciBusId, len, dev));
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorGetDeviceByPciBusId(int *dev, const char *pciBusId) {
  if (dev == NULL || pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(musaDeviceGetByPCIBusId(dev, pciBusId));
  return flagcxSuccess;
}

// Adapt the native Stream MemOp path used by MooreThreads/mccl commit
// 23edec5afda741efe0119410000dd80fd03092ef:
//   src/ce_coll.cc: 64-bit MemOp capability and ordered signal writes;
//   src/rma/rma_ce.cc: GEQ waits and remote-write flush capability;
//   src/include/musawrap.h: native muStreamWait/WriteValue64 entry points.
// Keep DefaultBackend; these operations do not require an MCCL communicator.
namespace {

flagcxResult_t musaStreamMemOpResult(MUresult result) {
  if (result == MUSA_SUCCESS)
    return flagcxSuccess;
  if (result == MUSA_ERROR_NOT_SUPPORTED)
    return flagcxNotSupported;
  return flagcxUnhandledDeviceError;
}

flagcxResult_t musaStreamMemOpCapability(MUdevice device,
                                         MUdevice_attribute attribute) {
  int supported = 0;
  const MUresult result = muDeviceGetAttribute(&supported, attribute, device);
  // Older drivers can reject an attribute they do not implement. Do not
  // confuse context/device failures with a missing optional capability.
  if (result == MUSA_ERROR_INVALID_VALUE || result == MUSA_ERROR_NOT_SUPPORTED)
    return flagcxNotSupported;
  if (result != MUSA_SUCCESS)
    return flagcxUnhandledDeviceError;
  return supported != 0 ? flagcxSuccess : flagcxNotSupported;
}

flagcxResult_t musaStreamMemOpCheck(bool flushRemoteWrites) {
  MUdevice device;
  const MUresult result = muCtxGetDevice(&device);
  if (result != MUSA_SUCCESS)
    return musaStreamMemOpResult(result);
  flagcxResult_t capability = musaStreamMemOpCapability(
      device, MU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS);
  if (capability != flagcxSuccess)
    return capability;
  if (flushRemoteWrites)
    return musaStreamMemOpCapability(
        device, MU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES);
  return flagcxSuccess;
}

} // namespace

flagcxResult_t musaAdaptorStreamWaitValue64(flagcxStream_t stream, void *addr,
                                            uint64_t value, int flags) {
  if (stream == nullptr || addr == nullptr ||
      (reinterpret_cast<uintptr_t>(addr) % alignof(uint64_t)) != 0)
    return flagcxInvalidArgument;
  if (flags & ~FLAGCX_STREAM_WAIT_VALUE_FLUSH_REMOTE_WRITES)
    return flagcxInvalidArgument;

  const bool flushRemoteWrites =
      (flags & FLAGCX_STREAM_WAIT_VALUE_FLUSH_REMOTE_WRITES) != 0;
  const flagcxResult_t capability = musaStreamMemOpCheck(flushRemoteWrites);
  if (capability != flagcxSuccess)
    return capability;

  // Native GEQ follows the driver's cyclic sequence-number comparison,
  // consistent with MCCL. Use counters with an outstanding distance < 2^63.
  unsigned int waitFlags = MU_STREAM_WAIT_VALUE_GEQ;
  if (flushRemoteWrites)
    waitFlags |= MU_STREAM_WAIT_VALUE_FLUSH;
  return musaStreamMemOpResult(
      muStreamWaitValue64(reinterpret_cast<MUstream>(stream->base),
                          reinterpret_cast<MUdeviceptr>(addr),
                          static_cast<muuint64_t>(value), waitFlags));
}

flagcxResult_t musaAdaptorStreamWriteValue64(flagcxStream_t stream, void *addr,
                                             uint64_t value, int flags) {
  if (stream == nullptr || addr == nullptr ||
      (reinterpret_cast<uintptr_t>(addr) % alignof(uint64_t)) != 0 ||
      flags != 0)
    return flagcxInvalidArgument;
  const flagcxResult_t capability = musaStreamMemOpCheck(false);
  if (capability != flagcxSuccess)
    return capability;

  // DEFAULT retains the native stream-scoped pre-write memory barrier.
  // Do not select NO_MEMORY_BARRIER or replace this with a host-side store.
  return musaStreamMemOpResult(muStreamWriteValue64(
      reinterpret_cast<MUstream>(stream->base),
      reinterpret_cast<MUdeviceptr>(addr), static_cast<muuint64_t>(value),
      MU_STREAM_WRITE_VALUE_DEFAULT));
}
flagcxResult_t musaAdaptorEventElapsedTime(float *ms, flagcxEvent_t start,
                                           flagcxEvent_t end) {
  if (ms == NULL || start == NULL || end == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(musaEventElapsedTime(ms, start->base, end->base));
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorHostRegister(void *ptr, size_t bytes) {
  if (ptr == nullptr || bytes == 0)
    return flagcxInvalidArgument;
  const musaError_t result =
      musaHostRegister(ptr, bytes, musaHostRegisterMapped);
  if (result == musaErrorNotSupported)
    return flagcxNotSupported;
  DEVCHECK(result);
  return flagcxSuccess;
}
flagcxResult_t musaAdaptorHostUnregister(void *ptr) {
  if (ptr == nullptr)
    return flagcxInvalidArgument;
  const musaError_t result = musaHostUnregister(ptr);
  if (result == musaErrorNotSupported)
    return flagcxNotSupported;
  DEVCHECK(result);
  return flagcxSuccess;
}

// Symmetric memory VMM stubs (not supported)
flagcxResult_t musaAdaptorSymPhysAlloc(void *, size_t, void **, void *,
                                       size_t *, size_t *) {
  return flagcxNotSupported;
}
flagcxResult_t musaAdaptorSymPhysFree(void *) { return flagcxNotSupported; }
flagcxResult_t musaAdaptorSymFlatMap(void *[], int, int, void *, size_t,
                                     void **) {
  return flagcxNotSupported;
}
flagcxResult_t musaAdaptorSymFlatUnmap(void *, size_t, int) {
  return flagcxNotSupported;
}
flagcxResult_t musaAdaptorSymMulticastSupported(int *supported) {
  if (supported)
    *supported = 0;
  return flagcxSuccess;
}
flagcxResult_t musaAdaptorSymMulticastCreate(size_t, int, const int *, void **,
                                             int *) {
  return flagcxNotSupported;
}
flagcxResult_t musaAdaptorSymMulticastBind(void *, int, void *, size_t, int,
                                           int, void **, size_t *) {
  return flagcxNotSupported;
}
flagcxResult_t musaAdaptorSymMulticastTeardown(void *, size_t) {
  return flagcxSuccess;
}
flagcxResult_t musaAdaptorSymMulticastFree(void *) {
  return flagcxNotSupported;
}

// Ordinary GPU memory registration uses the native DMA-BUF route when
// the platform and the specific allocation both support exporting it.
flagcxResult_t musaAdaptorDmaSupport(bool *supported) {
  if (supported == nullptr)
    return flagcxInvalidArgument;
  *supported = false;
  int ordinal = 0;
  DEVCHECK(musaGetDevice(&ordinal));
  MUdevice device;
  if (muDeviceGet(&device, ordinal) != MUSA_SUCCESS)
    return flagcxUnhandledDeviceError;
  int value = 0;
  MUresult result = muDeviceGetAttribute(
      &value, MU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, device);
  if (result == MUSA_ERROR_NOT_SUPPORTED || result == MUSA_ERROR_INVALID_VALUE)
    return flagcxSuccess;
  if (result != MUSA_SUCCESS)
    return flagcxUnhandledDeviceError;
  *supported = value != 0;
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorGetAddressRange(const void *ptr, void **base,
                                          size_t *size) {
  if (ptr == nullptr || base == nullptr || size == nullptr)
    return flagcxInvalidArgument;
  *base = nullptr;
  *size = 0;
  MUdeviceptr address = 0;
  MUresult result =
      muMemGetAddressRange(&address, size, reinterpret_cast<MUdeviceptr>(ptr));
  if (result == MUSA_ERROR_NOT_SUPPORTED || result == MUSA_ERROR_INVALID_VALUE)
    return flagcxNotSupported;
  if (result != MUSA_SUCCESS)
    return flagcxUnhandledDeviceError;
  *base = reinterpret_cast<void *>(address);
  return flagcxSuccess;
}

flagcxResult_t musaAdaptorGetPointerType(const void *ptr, int *ptrType) {
  if (ptr == nullptr || ptrType == nullptr)
    return flagcxInvalidArgument;
  musaPointerAttributes attrs = {};
  const musaError_t result = musaPointerGetAttributes(&attrs, ptr);
  if (result == musaErrorInvalidValue) {
    // Ordinary host allocations need not be tracked by the runtime.
    musaGetLastError();
    *ptrType = FLAGCX_PTR_HOST;
    return flagcxSuccess;
  }
  if (result != musaSuccess) {
    musaGetLastError();
    return flagcxUnhandledDeviceError;
  }
#if MUSART_VERSION >= 38300
  *ptrType = (attrs.type == musaMemoryTypeDevice ||
              attrs.type == musaMemoryTypeManaged)
                 ? FLAGCX_PTR_CUDA
                 : FLAGCX_PTR_HOST;
#else
  *ptrType = (attrs.memoryType == musaMemoryTypeDevice || attrs.isManaged)
                 ? FLAGCX_PTR_CUDA
                 : FLAGCX_PTR_HOST;
#endif
  return flagcxSuccess;
}

flagcxResult_t
musaAdaptorMemGetHandleForAddressRange(void *handleOut, void *buffer,
                                       size_t size, unsigned long long flags) {
  if (handleOut == nullptr || buffer == nullptr || size == 0)
    return flagcxInvalidArgument;
  *static_cast<int *>(handleOut) = -1;
  const long page = sysconf(_SC_PAGESIZE);
  if (page <= 0)
    return flagcxSystemError;
  if (reinterpret_cast<uintptr_t>(buffer) % page != 0 || size % page != 0)
    return flagcxInvalidArgument;
  MUresult result = muMemGetHandleForAddressRange(
      handleOut, reinterpret_cast<MUdeviceptr>(buffer), size,
      MU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, flags);
  if (result == MUSA_ERROR_NOT_SUPPORTED ||
      (result == MUSA_ERROR_INVALID_VALUE && flags == 0))
    return flagcxNotSupported;
  return result == MUSA_SUCCESS ? flagcxSuccess : flagcxUnhandledDeviceError;
}

struct flagcxDeviceAdaptor musaAdaptor {
  "MUSA",
      // Basic functions
      musaAdaptorDeviceSynchronize, musaAdaptorDeviceMemcpy,
      musaAdaptorDeviceMemset, musaAdaptorDeviceMalloc, musaAdaptorDeviceFree,
      musaAdaptorSetDevice, musaAdaptorGetDevice, musaAdaptorGetDeviceCount,
      musaAdaptorGetVendor, musaAdaptorHostGetDevicePointer,
      // GDR functions
      NULL, // flagcxResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // flagcxResult_t (*memHandleDestroy)(int dev, void *memHandle);
      musaAdaptorGdrMemAlloc, musaAdaptorGdrMemFree,
      NULL, // flagcxResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // flagcxResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      NULL, // flagcxResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t
            // sz);
      NULL, // flagcxResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);
      // Stream functions
      musaAdaptorStreamCreate, musaAdaptorStreamDestroy, musaAdaptorStreamCopy,
      musaAdaptorStreamFree, musaAdaptorStreamSynchronize,
      musaAdaptorStreamQuery, musaAdaptorStreamWaitEvent,
      musaAdaptorStreamWaitValue64, musaAdaptorStreamWriteValue64,
      // Event functions
      musaAdaptorEventCreate, musaAdaptorEventDestroy, musaAdaptorEventRecord,
      musaAdaptorEventSynchronize, musaAdaptorEventQuery,
      musaAdaptorEventElapsedTime,
      // IpcMemHandle functions
      musaAdaptorIpcMemHandleCreate, musaAdaptorIpcMemHandleGet,
      musaAdaptorIpcMemHandleOpen, musaAdaptorIpcMemHandleClose,
      musaAdaptorIpcMemHandleFree,
      // Kernel launch
      NULL, // flagcxResult_t (*launchKernel)(void *func, unsigned int block_x,
            // unsigned int block_y, unsigned int block_z, unsigned int grid_x,
            // unsigned int grid_y, unsigned int grid_z, void **args, size_t
            // share_mem, void *stream, void *memHandle);
      NULL, // flagcxResult_t (*copyArgsInit)(void **args);
      NULL, // flagcxResult_t (*copyArgsFree)(void *args);
      NULL, // flagcxResult_t (*launchDeviceFunc)(flagcxStream_t stream, void
            // *args);
      // Others
      musaAdaptorGetDeviceProperties, // flagcxResult_t
                                      // (*getDeviceProperties)(struct
                                      // flagcxDevProps *props, int dev);
      musaAdaptorGetDevicePciBusId, // flagcxResult_t (*getDevicePciBusId)(char
                                    // *pciBusId, int len, int dev);
      musaAdaptorGetDeviceByPciBusId, // flagcxResult_t
                                      // (*getDeviceByPciBusId)(int
                                      // *dev, const char *pciBusId);
      musaAdaptorLaunchHostFunc,
      // DMA buffer
      musaAdaptorDmaSupport, musaAdaptorMemGetHandleForAddressRange,
      musaAdaptorHostRegister,   // flagcxResult_t (*hostRegister)(void *,
                                 // size_t);
      musaAdaptorHostUnregister, // flagcxResult_t (*hostUnregister)(void *);
      // Symmetric memory VMM functions (not supported)
      musaAdaptorSymPhysAlloc, musaAdaptorSymPhysFree, musaAdaptorSymFlatMap,
      musaAdaptorSymFlatUnmap, musaAdaptorSymMulticastSupported,
      musaAdaptorSymMulticastCreate, musaAdaptorSymMulticastBind,
      musaAdaptorSymMulticastTeardown, musaAdaptorSymMulticastFree,
      musaAdaptorGetLastError, musaAdaptorGetPointerType,
      musaAdaptorGetAddressRange, FLAGCX_VMM_MR_CAP_NONE,
      FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE, NULL, NULL, NULL, NULL, NULL,
      // A NIC completion alone does not establish GPU visibility. Require
      // the provider's acquire for both incoming WRITE and local GET results.
      FLAGCX_GDR_READ_REQUIRES_FLUSH | FLAGCX_GDR_WRITE_REQUIRES_FLUSH,
      FLAGCX_GDR_DEVICE_MUSA,
};

#endif // USE_MUSA_ADAPTOR
