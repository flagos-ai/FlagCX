#include "iluvatar_adaptor.h"

#ifdef USE_ILUVATAR_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "param.h"
#include "segmented_vmm_allocation.h"
#include <climits>
#include <cstdint>
#include <dlfcn.h>
#include <unistd.h>

// Header availability is only a compile prerequisite. Resolve each driver
// entry point at runtime so older host drivers fail explicitly, not at dlopen.
#ifndef FLAGCX_ILUVATAR_HAS_VMM
#define FLAGCX_ILUVATAR_HAS_VMM 0
#endif
#if FLAGCX_ILUVATAR_HAS_VMM
static flagcxResult_t ixcudaVmmResult(CUresult result, const char *operation) {
  INFO(FLAGCX_ALLOC, "IXCUDA VMM end: %s result=%d", operation, (int)result);
  if (result == CUDA_SUCCESS)
    return flagcxSuccess;
  return result == CUDA_ERROR_NOT_SUPPORTED ? flagcxNotSupported
                                            : flagcxUnhandledDeviceError;
}
#define IXCUDA_VMM_CALL(fn, ...)                                               \
  do {                                                                         \
    static auto entry =                                                        \
        reinterpret_cast<decltype(&fn)>(dlsym(RTLD_DEFAULT, #fn));             \
    if (entry == nullptr) {                                                    \
      INFO(FLAGCX_ALLOC, "IXCUDA VMM: missing driver symbol %s", #fn);         \
      return flagcxNotSupported;                                               \
    }                                                                          \
    return ixcudaVmmResult(entry(__VA_ARGS__), #fn);                           \
  } while (0)

struct IxcudaVmmOps {
  using Handle = CUmemGenericAllocationHandle;
  using Prop = CUmemAllocationProp;
  static flagcxResult_t deviceOrdinal(int *device) {
    return cudaGetDevice(device) == cudaSuccess ? flagcxSuccess
                                                : flagcxUnhandledDeviceError;
  }
  static flagcxResult_t createHandle(size_t size, const Prop &prop, Handle *h) {
    IXCUDA_VMM_CALL(cuMemCreate, h, size, &prop, 0);
  }
  static flagcxResult_t reserveAddress(size_t size, size_t alignment,
                                       uintptr_t *base) {
    CUdeviceptr address = 0;
    static auto entry = reinterpret_cast<decltype(&cuMemAddressReserve)>(
        dlsym(RTLD_DEFAULT, "cuMemAddressReserve"));
    if (entry == nullptr)
      return flagcxNotSupported;
    auto result = ixcudaVmmResult(entry(&address, size, alignment, 0, 0),
                                  "cuMemAddressReserve");
    if (result == flagcxSuccess)
      *base = static_cast<uintptr_t>(address);
    return result;
  }
  static flagcxResult_t freeAddress(uintptr_t address, size_t size) {
    IXCUDA_VMM_CALL(cuMemAddressFree, (CUdeviceptr)address, size);
  }
  static flagcxResult_t mapSegment(uintptr_t address, size_t size, Handle h) {
    IXCUDA_VMM_CALL(cuMemMap, (CUdeviceptr)address, size, 0, h, 0);
  }
  static flagcxResult_t unmapSegment(uintptr_t address, size_t size) {
    IXCUDA_VMM_CALL(cuMemUnmap, (CUdeviceptr)address, size);
  }
  static flagcxResult_t setAccess(uintptr_t address, size_t size) {
    int device = 0;
    FLAGCXCHECK(deviceOrdinal(&device));
    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    static auto entry = reinterpret_cast<decltype(&cuMemSetAccess)>(
        dlsym(RTLD_DEFAULT, "cuMemSetAccess"));
    if (entry == nullptr)
      return flagcxNotSupported;
    FLAGCXCHECK(ixcudaVmmResult(entry((CUdeviceptr)address, size, &access, 1),
                                "cuMemSetAccess"));
    // Apply PR 1's ordering contract to the VMM VA, not cudaMalloc alone.
    unsigned sync = 1;
    return ixcudaVmmResult(
        cuPointerSetAttribute(&sync, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                              (CUdeviceptr)address),
        "cuPointerSetAttribute(SYNC_MEMOPS)");
  }
  static flagcxResult_t importHandle(int fd, Handle *h) {
    IXCUDA_VMM_CALL(cuMemImportFromShareableHandle, h,
                    reinterpret_cast<void *>(static_cast<uintptr_t>(fd)),
                    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
  }
  static flagcxResult_t exportHandle(Handle h, int *fd) {
    if (fd == nullptr)
      return flagcxInvalidArgument;
    *fd = -1;
    IXCUDA_VMM_CALL(cuMemExportToShareableHandle, fd, h,
                    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0);
  }
  static flagcxResult_t releaseHandle(Handle h) {
    IXCUDA_VMM_CALL(cuMemRelease, h);
  }
};
using IxcudaVmmAllocation = flagcxSegmentedVmmAllocation<IxcudaVmmOps>;
using IxcudaVmmMap = flagcxSegmentedVmmMap<IxcudaVmmOps>;

static flagcxResult_t ixcudaVmmProperties(CUmemAllocationProp *prop,
                                          size_t *granularity) {
  int device = 0;
  FLAGCXCHECK(IxcudaVmmOps::deviceOrdinal(&device));
  *prop = {};
  prop->type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop->location.id = device;
  prop->requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  IXCUDA_VMM_CALL(cuMemGetAllocationGranularity, granularity, prop,
                  CU_MEM_ALLOC_GRANULARITY_MINIMUM);
}

static flagcxResult_t ixcudaVmmAllocate(void **ptr, size_t size) {
  *ptr = nullptr;
  if (size == 0)
    return flagcxInvalidArgument;
  int device = 0;
  FLAGCXCHECK(IxcudaVmmOps::deviceOrdinal(&device));
  FLAGCXCHECK(IxcudaVmmAllocation::retryPending(device));
  CUmemAllocationProp prop = {};
  size_t granularity = 0;
  FLAGCXCHECK(ixcudaVmmProperties(&prop, &granularity));
  if (granularity == 0 || size > SIZE_MAX - (granularity - 1))
    return flagcxInvalidArgument;
  const size_t aligned = ((size + granularity - 1) / granularity) * granularity;
  size_t segmentBytes = 0;
  FLAGCXCHECK(flagcxVmmSegmentBytes(granularity, &segmentBytes));
  if (segmentBytes == 0)
    segmentBytes = aligned;
  bool multiSegment = true;
  FLAGCXCHECK(flagcxVmmMultiSegmentEnabled(true, &multiSegment));
  if (!multiSegment && segmentBytes < aligned)
    return flagcxNotSupported;
  return IxcudaVmmAllocation::allocate(ptr, aligned, granularity, segmentBytes,
                                       prop);
}

static flagcxResult_t ixcudaVmmFree(void *ptr, bool *found) {
  void *base = nullptr;
  size_t size = 0;
  int owner = -1, saved = -1;
  *found = false;
  auto range = IxcudaVmmAllocation::addressRange(ptr, &base, &size, &owner);
  if (range == flagcxNotSupported)
    return flagcxSuccess;
  if (range != flagcxSuccess)
    return range;
  *found = true;
  if (ptr != base)
    return flagcxInvalidArgument;
  FLAGCXCHECK(IxcudaVmmOps::deviceOrdinal(&saved));
  if (owner != saved && cudaSetDevice(owner) != cudaSuccess)
    return flagcxUnhandledDeviceError;
  auto result = IxcudaVmmAllocation::freeAllocation(ptr, found);
  if (owner != saved && cudaSetDevice(saved) != cudaSuccess &&
      result == flagcxSuccess)
    result = flagcxUnhandledDeviceError;
  return result;
}

flagcxResult_t
ixcudaAdaptorSymPhysDescribe(void *ptr, size_t size, void **opaque,
                             const flagcxSymSegmentMeta **segments, int *count,
                             size_t *offset, size_t *alignment) {
  if (!ptr || !size || !opaque || !segments || !count || !offset ||
      !alignment || size > UINTPTR_MAX - reinterpret_cast<uintptr_t>(ptr))
    return flagcxInvalidArgument;
  *opaque = nullptr;
  *segments = nullptr;
  *count = 0;
  *offset = *alignment = 0;
  auto *bundle = new (std::nothrow) IxcudaVmmMap::Bundle();
  if (!bundle)
    return flagcxSystemError;
  *opaque = bundle; // Includes partial description ownership on failure.
  bool found = false;
  FLAGCXCHECK(IxcudaVmmAllocation::describe(ptr, size, bundle, offset, &found));
  if (!found) {
    // Only a successful legacy IPC export proves ordinary allocation. Leave
    // an empty bundle for unknown VMM to prevent unsafe legacy fallback.
    CUdeviceptr base = 0;
    size_t bytes = 0;
    cudaIpcMemHandle_t ipc;
    INFO(FLAGCX_ALLOC, "IXCUDA IPC probe range begin: ptr=%p size=%zu", ptr,
         size);
    if (flagcxDebugLevel >= FLAGCX_LOG_INFO)
      fflush(flagcxDebugFile);
    const CUresult rangeResult =
        cuMemGetAddressRange(&base, &bytes, (CUdeviceptr)ptr);
    INFO(FLAGCX_ALLOC,
         "IXCUDA IPC probe range end: result=%d ptr=%p base=0x%llx bytes=%zu "
         "requested=%zu",
         (int)rangeResult, ptr, (unsigned long long)base, bytes, size);
    if (rangeResult == CUDA_SUCCESS &&
        reinterpret_cast<uintptr_t>(ptr) >= base &&
        reinterpret_cast<uintptr_t>(ptr) - base < bytes &&
        size <= bytes - (reinterpret_cast<uintptr_t>(ptr) - base)) {
      INFO(FLAGCX_ALLOC,
           "IXCUDA IPC probe export begin: base=0x%llx handleBytes=%zu",
           (unsigned long long)base, sizeof(ipc));
      if (flagcxDebugLevel >= FLAGCX_LOG_INFO)
        fflush(flagcxDebugFile);
      const auto ipcResult =
          cudaIpcGetMemHandle(&ipc, reinterpret_cast<void *>(base));
      INFO(FLAGCX_ALLOC, "IXCUDA IPC probe export end: result=%d",
           (int)ipcResult);
      if (ipcResult == cudaSuccess) {
        delete bundle;
        *opaque = nullptr;
      } else {
        cudaGetLastError();
      }
    } else {
      cudaGetLastError();
    }
    return flagcxNotSupported;
  }
  // Match CUDA: the allocation owns the original create handles. The bundle
  // borrows them and must be freed before the source allocation is released.
  bool multiSegment = true;
  FLAGCXCHECK(flagcxVmmMultiSegmentEnabled(true, &multiSegment));
  if (!multiSegment && bundle->segments.size() > 1)
    return flagcxNotSupported;
  CUmemAllocationProp prop;
  FLAGCXCHECK(ixcudaVmmProperties(&prop, alignment));
  if (*alignment == 0 || bundle->segments.empty() ||
      bundle->segments.size() > static_cast<size_t>(INT_MAX))
    return flagcxInvalidUsage;
  *count = static_cast<int>(bundle->segments.size());
  *segments = bundle->segments.data();
  return flagcxSuccess;
}
flagcxResult_t ixcudaAdaptorSymPhysExportSegment(void *bundle, int index,
                                                 int *fd) {
  return IxcudaVmmMap::exportSegment(bundle, index, fd);
}
flagcxResult_t ixcudaAdaptorSymPhysFreeSegments(void *bundle) {
  return IxcudaVmmMap::freeBundle(bundle);
}
flagcxResult_t ixcudaAdaptorSymPhysSegmentCount(const void *ptr, size_t size,
                                                int *count) {
  return IxcudaVmmAllocation::segmentCount(ptr, size, count);
}
flagcxResult_t ixcudaAdaptorSymFlatMapSegments(
    const int *fds, int peers, int self, void *bundle,
    const flagcxSymRankLayout *layouts, const flagcxSymSegmentMeta *segments,
    int count, size_t stride, size_t alignment, void **base) {
  return IxcudaVmmMap::map(fds, peers, self, bundle, layouts, segments, count,
                           stride, alignment, base);
}
flagcxResult_t ixcudaAdaptorSymFlatMapSegmentsBegin(
    int peers, int self, void *bundle, const flagcxSymRankLayout *layouts,
    const flagcxSymSegmentMeta *segments, int count, size_t stride,
    size_t alignment, void **base) {
  return IxcudaVmmMap::begin(peers, self, bundle, layouts, segments, count,
                             stride, alignment, base);
}
flagcxResult_t ixcudaAdaptorSymFlatMapSegmentOne(void *base, int index,
                                                 int fd) {
  return IxcudaVmmMap::mapOne(base, index, fd);
}
flagcxResult_t ixcudaAdaptorSymFlatMappingUnmapSegments(void *base) {
  return IxcudaVmmMap::unmap(base);
}
flagcxResult_t ixcudaAdaptorSymFlatVaFreeSegments(void *base) {
  return IxcudaVmmMap::freeVa(base);
}
flagcxResult_t ixcudaAdaptorSymLsaGetCapacity(size_t *bytes,
                                              size_t *alignment) {
  if (!bytes || !alignment)
    return flagcxInvalidArgument;
  size_t freeBytes = 0;
  if (cudaMemGetInfo(&freeBytes, bytes) != cudaSuccess)
    return flagcxUnhandledDeviceError;
  CUmemAllocationProp prop;
  return ixcudaVmmProperties(&prop, alignment);
}
flagcxResult_t ixcudaAdaptorSymLsaVaReserve(size_t size, size_t alignment,
                                            void **base) {
  if (!base || !size || !alignment || (alignment & (alignment - 1)) != 0 ||
      size % alignment != 0)
    return flagcxInvalidArgument;
  *base = nullptr;
  uintptr_t address = 0;
  auto result = IxcudaVmmOps::reserveAddress(size, alignment, &address);
  if (result == flagcxSuccess)
    *base = reinterpret_cast<void *>(address);
  return result;
}
flagcxResult_t ixcudaAdaptorSymLsaVaFree(void *base, size_t size) {
  if (!base || !size)
    return flagcxInvalidArgument;
  return IxcudaVmmOps::freeAddress(reinterpret_cast<uintptr_t>(base), size);
}
flagcxResult_t
ixcudaAdaptorSymLsaMapBeginAt(void *window, int peers, int self, void *bundle,
                              const flagcxSymRankLayout *layouts,
                              const flagcxSymSegmentMeta *segments, int count,
                              size_t stride, size_t alignment, void **base) {
  return IxcudaVmmMap::beginAt(window, peers, self, bundle, layouts, segments,
                               count, stride, alignment, base);
}

// Legacy single-segment callbacks share the same retry-safe mapping records.
flagcxResult_t ixcudaAdaptorSymPhysAlloc(void *ptr, size_t size,
                                         void **physical, void *shared,
                                         size_t *handleSize,
                                         size_t *allocSize) {
  if (!physical || !shared || !handleSize || !allocSize ||
      *handleSize < sizeof(int))
    return flagcxInvalidArgument;
  *physical = nullptr;
  *static_cast<int *>(shared) = -1;
  void *bundle = nullptr;
  const flagcxSymSegmentMeta *segments = nullptr;
  int count = 0;
  size_t offset = 0, alignment = 0;
  auto result = ixcudaAdaptorSymPhysDescribe(ptr, size, &bundle, &segments,
                                             &count, &offset, &alignment);
  if (result == flagcxSuccess && count != 1)
    result = flagcxNotSupported;
  if (result == flagcxSuccess) {
    *allocSize = segments[0].size;
    result = ixcudaAdaptorSymPhysExportSegment(bundle, 0,
                                               static_cast<int *>(shared));
  }
  if (result != flagcxSuccess) {
    ixcudaAdaptorSymPhysFreeSegments(bundle);
    return result;
  }
  *handleSize = sizeof(int);
  *physical = bundle;
  return flagcxSuccess;
}
flagcxResult_t ixcudaAdaptorSymPhysFree(void *bundle) {
  return IxcudaVmmMap::freeBundle(bundle);
}
flagcxResult_t ixcudaAdaptorSymFlatMap(void *handles[], int peers, int self,
                                       void *bundle, size_t size, void **base) {
  if (!handles || !bundle || !base || peers <= 0 || self < 0 || self >= peers ||
      !size)
    return flagcxInvalidArgument;
  try {
    std::vector<flagcxSymRankLayout> layouts(peers);
    std::vector<flagcxSymSegmentMeta> segments(peers, {0, size});
    std::vector<int> fds(peers, -1);
    for (int i = 0; i < peers; i++) {
      if (i != self && !handles[i])
        return flagcxInvalidArgument;
      layouts[i] = flagcxSymRankLayout{size, 0, size, i, 1};
      if (i != self)
        fds[i] = *static_cast<int *>(handles[i]);
    }
    CUmemAllocationProp prop;
    size_t alignment = 0;
    FLAGCXCHECK(ixcudaVmmProperties(&prop, &alignment));
    return IxcudaVmmMap::map(fds.data(), peers, self, bundle, layouts.data(),
                             segments.data(), peers, size, alignment, base);
  } catch (const std::bad_alloc &) {
    return flagcxSystemError;
  }
}
flagcxResult_t ixcudaAdaptorSymFlatMappingUnmap(void *base, size_t, int) {
  return IxcudaVmmMap::unmap(base);
}
flagcxResult_t ixcudaAdaptorSymFlatVaFree(void *base, size_t, int) {
  return IxcudaVmmMap::freeVa(base);
}
flagcxResult_t ixcudaAdaptorSymFlatUnmap(void *base, size_t size, int peers) {
  FLAGCXCHECK(ixcudaAdaptorSymFlatMappingUnmap(base, size, peers));
  return ixcudaAdaptorSymFlatVaFree(base, size, peers);
}
#undef IXCUDA_VMM_CALL
#endif // FLAGCX_ILUVATAR_HAS_VMM

flagcxResult_t ixcudaAdaptorGetAddressRange(const void *ptr, void **base,
                                            size_t *size) {
  if (!ptr || !base || !size)
    return flagcxInvalidArgument;
  *base = nullptr;
  *size = 0;
#if FLAGCX_ILUVATAR_HAS_VMM
  auto result = IxcudaVmmAllocation::addressRange(ptr, base, size);
  if (result != flagcxNotSupported)
    return result;
#endif
  CUdeviceptr address = 0;
  CUresult resultCode = cuMemGetAddressRange(&address, size, (CUdeviceptr)ptr);
  if (resultCode != CUDA_SUCCESS)
    return flagcxUnhandledDeviceError;
  *base = reinterpret_cast<void *>(address);
  return flagcxSuccess;
}

std::map<flagcxMemcpyType_t, cudaMemcpyKind> memcpy_type_map = {
    {flagcxMemcpyHostToDevice, cudaMemcpyHostToDevice},
    {flagcxMemcpyDeviceToHost, cudaMemcpyDeviceToHost},
    {flagcxMemcpyDeviceToDevice, cudaMemcpyDeviceToDevice},
};

// Give ordinary device allocations the same RDMA memory ordering as GDR
// allocations before either kind of buffer is registered with a NIC.
static flagcxResult_t ixcudaAdaptorSetSyncMemops(void *ptr) {
  cudaPointerAttributes attrs;
  if (cudaPointerGetAttributes(&attrs, ptr) != cudaSuccess ||
      attrs.devicePointer == NULL)
    return flagcxUnhandledDeviceError;
  unsigned flags = 1;
  if (cuPointerSetAttribute(&flags, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                            (CUdeviceptr)attrs.devicePointer) != CUDA_SUCCESS)
    return flagcxUnhandledDeviceError;
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorDeviceSynchronize() {
  DEVCHECK(cudaDeviceSynchronize());
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorDeviceMemcpy(void *dst, void *src, size_t size,
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

flagcxResult_t ixcudaAdaptorDeviceMemset(void *ptr, int value, size_t size,
                                         flagcxMemType_t type,
                                         flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    memset(ptr, value, size);
  } else {
    if (stream == NULL) {
      DEVCHECK(cudaMemset(ptr, value, size));
    } else {
      DEVCHECK(cudaMemsetAsync(ptr, value, size, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorDeviceMalloc(void **ptr, size_t size,
                                         flagcxMemType_t type,
                                         flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    DEVCHECK(cudaMallocHost(ptr, size));
  } else if (type == flagcxMemManaged) {
    DEVCHECK(cudaMallocManaged(ptr, size, cudaMemAttachGlobal));
  } else {
    if (stream == NULL) {
      DEVCHECK(cudaMalloc(ptr, size));
    } else {
      DEVCHECK(cudaMallocAsync(ptr, size, stream->base));
    }
    // A zero-byte allocation may succeed with a null pointer.
    flagcxResult_t result =
        *ptr == NULL ? flagcxSuccess : ixcudaAdaptorSetSyncMemops(*ptr);
    if (result != flagcxSuccess) {
      if (stream == NULL)
        cudaFree(*ptr);
      else
        cudaFreeAsync(*ptr, stream->base);
      *ptr = NULL;
      return result;
    }
  }
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorDeviceFree(void *ptr, flagcxMemType_t type,
                                       flagcxStream_t stream) {
  if (type == flagcxMemHost) {
    DEVCHECK(cudaFreeHost(ptr));
  } else if (type == flagcxMemManaged) {
    DEVCHECK(cudaFree(ptr));
  } else {
    if (stream == NULL) {
      DEVCHECK(cudaFree(ptr));
    } else {
      DEVCHECK(cudaFreeAsync(ptr, stream->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorSetDevice(int dev) {
  DEVCHECK(cudaSetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorGetDevice(int *dev) {
  DEVCHECK(cudaGetDevice(dev));
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorGetDeviceCount(int *count) {
  DEVCHECK(cudaGetDeviceCount(count));
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorGetVendor(char *vendor) {
  strcpy(vendor, "ILUVATAR_COREX");
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorHostGetDevicePointer(void **pDevice, void *pHost) {
  if (pDevice == NULL || pHost == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaHostGetDevicePointer(pDevice, pHost, 0));
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorGdrMemAlloc(void **ptr, size_t size,
                                        void *memHandle) {
  if (ptr == NULL) {
    return flagcxInvalidArgument;
  }
  *ptr = NULL;
#if FLAGCX_ILUVATAR_HAS_VMM
  if (flagcxParamVmmEnable())
    return ixcudaVmmAllocate(ptr, size);
#endif
  // Match CUDA's legacy-SDK path: VMM_ENABLE selects VMM only when the
  // allocation APIs were compiled in. Keep ordinary GDR allocations usable.
  DEVCHECK(cudaMalloc(ptr, size));
  if (ixcudaAdaptorSetSyncMemops(*ptr) != flagcxSuccess) {
    cudaFree(*ptr);
    *ptr = NULL;
    return flagcxUnhandledDeviceError;
  }
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorGdrMemFree(void *ptr, void *memHandle) {
  if (ptr == NULL) {
    return flagcxSuccess;
  }
#if FLAGCX_ILUVATAR_HAS_VMM
  bool vmm = false;
  FLAGCXCHECK(ixcudaVmmFree(ptr, &vmm));
  if (vmm)
    return flagcxSuccess;
#endif
  DEVCHECK(cudaFree(ptr));
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorStreamCreate(flagcxStream_t *stream) {
  (*stream) = NULL;
  flagcxCalloc(stream, 1);
  DEVCHECK(cudaStreamCreateWithFlags((cudaStream_t *)(*stream),
                                     cudaStreamNonBlocking));
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorStreamDestroy(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(cudaStreamDestroy(stream->base));
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorStreamCopy(flagcxStream_t *newStream,
                                       void *oldStream) {
  (*newStream) = NULL;
  flagcxCalloc(newStream, 1);
  (*newStream)->base = (cudaStream_t)oldStream;
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorStreamFree(flagcxStream_t stream) {
  if (stream != NULL) {
    free(stream);
    stream = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorStreamSynchronize(flagcxStream_t stream) {
  if (stream != NULL) {
    DEVCHECK(cudaStreamSynchronize(stream->base));
  }
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorStreamQuery(flagcxStream_t stream) {
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

flagcxResult_t ixcudaAdaptorStreamWaitEvent(flagcxStream_t stream,
                                            flagcxEvent_t event) {
  if (stream != NULL && event != NULL) {
    DEVCHECK(cudaStreamWaitEvent(stream->base, event->base, 0));
  }
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorEventCreate(flagcxEvent_t *event,
                                        flagcxEventType_t eventType) {
  if (event == NULL)
    return flagcxInvalidArgument;
  *event = NULL;
  flagcxEvent_t wrapper = (flagcxEvent_t)calloc(1, sizeof(struct flagcxEvent));
  if (wrapper == NULL)
    return flagcxSystemError;
  const unsigned int flags = (eventType == flagcxEventDefault)
                                 ? cudaEventDefault
                                 : cudaEventDisableTiming;
  if (cudaEventCreateWithFlags(&wrapper->base, flags) != cudaSuccess) {
    free(wrapper);
    return flagcxUnhandledDeviceError;
  }
  *event = wrapper;
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorEventDestroy(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(cudaEventDestroy(event->base));
    free(event);
    event = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorEventRecord(flagcxEvent_t event,
                                        flagcxStream_t stream) {
  if (event != NULL) {
    if (stream != NULL) {
      DEVCHECK(cudaEventRecord(event->base, stream->base));
    } else {
      DEVCHECK(cudaEventRecord(event->base));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorEventSynchronize(flagcxEvent_t event) {
  if (event != NULL) {
    DEVCHECK(cudaEventSynchronize(event->base));
  }
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorEventQuery(flagcxEvent_t event) {
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

flagcxResult_t ixcudaAdaptorIpcMemHandleCreate(flagcxIpcMemHandle_t *handle,
                                               size_t *size) {
  flagcxCalloc(handle, 1);
  if (size != NULL) {
    *size = sizeof(cudaIpcMemHandle_t);
  }
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorIpcMemHandleGet(flagcxIpcMemHandle_t handle,
                                            void *devPtr) {
  if (handle == NULL || devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaIpcGetMemHandle(&handle->base, devPtr));
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorIpcMemHandleOpen(flagcxIpcMemHandle_t handle,
                                             void **devPtr) {
  if (handle == NULL || devPtr == NULL || *devPtr != NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaIpcOpenMemHandle(devPtr, handle->base,
                                cudaIpcMemLazyEnablePeerAccess));
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorIpcMemHandleClose(void *devPtr) {
  if (devPtr == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaIpcCloseMemHandle(devPtr));
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorIpcMemHandleFree(flagcxIpcMemHandle_t handle) {
  if (handle != NULL) {
    free(handle);
  }
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorLaunchHostFunc(flagcxStream_t stream,
                                           void (*fn)(void *), void *args) {
  if (stream != NULL) {
    DEVCHECK(cudaLaunchHostFunc(stream->base, fn, args));
  }
  return flagcxSuccess;
}

flagcxResult_t
ixcudaAdaptorLaunchKernel(void *func, unsigned int block_x,
                          unsigned int block_y, unsigned int block_z,
                          unsigned int grid_x, unsigned int grid_y,
                          unsigned int grid_z, void **args, size_t share_mem,
                          void *stream, void *memHandle) {
  // Host-side validation: report the precise reason rather than a generic
  // device error. A NULL stream is legal (legacy default stream), so it is not
  // rejected here.
  if (func == NULL || args == NULL || block_x == 0 || block_y == 0 ||
      block_z == 0 || grid_x == 0 || grid_y == 0 || grid_z == 0) {
    return flagcxInvalidArgument;
  }
  (void)memHandle; // Unused anywhere in the tree; kept for ABI parity.

  // FlagCX orders the dimensions block-first while CUDA orders them grid-first.
  // Assembling them explicitly is mandatory: a positional copy would transpose
  // the two and still "succeed" while producing wrong results or a hang.
  dim3 grid(grid_x, grid_y, grid_z);
  dim3 block(block_x, block_y, block_z);

  // The opaque stream handle wraps the vendor stream; NULL selects the legacy
  // default stream.
  cudaStream_t cudaStream =
      (stream == NULL) ? (cudaStream_t)0 : ((flagcxStream_t)stream)->base;

  // Deliberately not DEVCHECK: that macro collapses every vendor error into
  // flagcxUnhandledDeviceError and would discard the real cause. The specific
  // error is surfaced through the log instead.
  // The sticky error state is intentionally left untouched so that
  // getLastError() keeps reporting it later: kernel launch and error handling
  // are separate capabilities and must stay orthogonal.
  cudaError_t err = cudaLaunchKernel((const void *)func, grid, block, args,
                                     share_mem, cudaStream);
  if (err != cudaSuccess) {
    WARN("launchKernel FAILED: %s (%d) block=(%u,%u,%u) grid=(%u,%u,%u)",
         cudaGetErrorString(err), (int)err, block_x, block_y, block_z, grid_x,
         grid_y, grid_z);
    return flagcxUnhandledDeviceError;
  }
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorGetDeviceProperties(struct flagcxDevProps *props,
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

flagcxResult_t ixcudaAdaptorGetDevicePciBusId(char *pciBusId, int len,
                                              int dev) {
  if (pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaDeviceGetPCIBusId(pciBusId, len, dev));
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorGetDeviceByPciBusId(int *dev,
                                                const char *pciBusId) {
  if (dev == NULL || pciBusId == NULL) {
    return flagcxInvalidArgument;
  }
  DEVCHECK(cudaDeviceGetByPCIBusId(dev, pciBusId));
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorCanAccessPeer(int srcDev, int dstDev,
                                          int *canAccess) {
  if (canAccess == NULL)
    return flagcxInvalidArgument;
  *canAccess = 0;
  DEVCHECK(cudaDeviceCanAccessPeer(canAccess, srcDev, dstDev));
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorStreamWaitValue64(flagcxStream_t stream, void *addr,
                                              uint64_t value, int flags) {
  if (stream == NULL || addr == NULL ||
      (reinterpret_cast<uintptr_t>(addr) & (alignof(uint64_t) - 1)) != 0 ||
      (flags & ~FLAGCX_STREAM_WAIT_VALUE_FLUSH_REMOTE_WRITES) != 0)
    return flagcxInvalidArgument;

  // CoreX has not passed the remote-payload visibility test. A working local
  // wait does not establish the stronger acquire guarantee.
  if (flags & FLAGCX_STREAM_WAIT_VALUE_FLUSH_REMOTE_WRITES)
    return flagcxNotSupported;

  CUresult result =
      cuStreamWaitValue64((CUstream)stream->base, (CUdeviceptr)addr, value,
                          CU_STREAM_WAIT_VALUE_GEQ);
  if (result == CUDA_SUCCESS)
    return flagcxSuccess;
  return result == CUDA_ERROR_NOT_SUPPORTED ? flagcxNotSupported
                                            : flagcxUnhandledDeviceError;
}

flagcxResult_t ixcudaAdaptorStreamWriteValue64(flagcxStream_t stream,
                                               void *addr, uint64_t value,
                                               int flags) {
  if (stream == NULL || addr == NULL || flags != 0 ||
      (reinterpret_cast<uintptr_t>(addr) & (alignof(uint64_t) - 1)) != 0)
    return flagcxInvalidArgument;
  CUresult result =
      cuStreamWriteValue64((CUstream)stream->base, (CUdeviceptr)addr, value,
                           CU_STREAM_WRITE_VALUE_DEFAULT);
  if (result == CUDA_SUCCESS)
    return flagcxSuccess;
  return result == CUDA_ERROR_NOT_SUPPORTED ? flagcxNotSupported
                                            : flagcxUnhandledDeviceError;
}
flagcxResult_t ixcudaAdaptorEventElapsedTime(float *, flagcxEvent_t,
                                             flagcxEvent_t) {
  return flagcxNotSupported;
}

flagcxResult_t ixcudaAdaptorHostRegister(void *ptr, size_t size) {
  if (ptr == NULL || size == 0)
    return flagcxInvalidArgument;
  DEVCHECK(cudaHostRegister(ptr, size, cudaHostRegisterMapped));
  return flagcxSuccess;
}
flagcxResult_t ixcudaAdaptorHostUnregister(void *ptr) {
  if (ptr == NULL)
    return flagcxInvalidArgument;
  DEVCHECK(cudaHostUnregister(ptr));
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorDmaSupport(bool *supported) {
  if (supported == NULL)
    return flagcxInvalidArgument;
  *supported = false;
#if CUDA_VERSION >= 11070
  int runtimeDevice = -1;
  if (cudaGetDevice(&runtimeDevice) != cudaSuccess)
    return flagcxUnhandledDeviceError;
  CUdevice device;
  CUresult result = cuDeviceGet(&device, runtimeDevice);
  if (result != CUDA_SUCCESS)
    return flagcxUnhandledDeviceError;
  int attribute = 0;
  result = cuDeviceGetAttribute(&attribute,
                                CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, device);
  if (result == CUDA_ERROR_NOT_SUPPORTED || result == CUDA_ERROR_INVALID_VALUE)
    return flagcxSuccess;
  if (result != CUDA_SUCCESS)
    return flagcxUnhandledDeviceError;
  *supported = attribute != 0;
#endif
  return flagcxSuccess;
}

flagcxResult_t ixcudaAdaptorGetHandleForAddressRange(void *handleOut,
                                                     void *buffer, size_t size,
                                                     unsigned long long flags) {
  if (handleOut == NULL || buffer == NULL || size == 0 || flags != 0)
    return flagcxInvalidArgument;
  *static_cast<int *>(handleOut) = -1;
  if (size > UINTPTR_MAX - reinterpret_cast<uintptr_t>(buffer))
    return flagcxInvalidArgument;
  const long pageSize = sysconf(_SC_PAGESIZE);
  if (pageSize <= 0)
    return flagcxSystemError;
  if (reinterpret_cast<uintptr_t>(buffer) % (size_t)pageSize != 0 ||
      size % (size_t)pageSize != 0)
    return flagcxInvalidArgument;
#if CUDA_VERSION >= 11070
  cudaPointerAttributes attrs = {};
  cudaError_t pointerError = cudaPointerGetAttributes(&attrs, buffer);
  if (pointerError != cudaSuccess) {
    cudaGetLastError();
    return pointerError == cudaErrorInvalidValue ? flagcxInvalidArgument
                                                 : flagcxUnhandledDeviceError;
  }
#if CUDART_VERSION >= 10000
  if (attrs.type != cudaMemoryTypeDevice)
#else
  if (attrs.memoryType != cudaMemoryTypeDevice)
#endif
    return flagcxInvalidArgument;
  int fd = -1;
  CUresult result =
      cuMemGetHandleForAddressRange(&fd, (CUdeviceptr)buffer, size,
                                    CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, flags);
  if (result == CUDA_SUCCESS && fd >= 0) {
    *static_cast<int *>(handleOut) = fd;
    return flagcxSuccess;
  }
  if (fd >= 0)
    (void)close(fd);
  if (result == CUDA_ERROR_NOT_SUPPORTED ||
      result == CUDA_ERROR_INVALID_VALUE) {
    INFO(FLAGCX_REG,
         "IXCUDA DMA-BUF export unavailable: ptr=%p size=%zu result=%d", buffer,
         size, static_cast<int>(result));
    return flagcxNotSupported;
  }
  WARN("IXCUDA DMA-BUF export failed: ptr=%p size=%zu result=%d", buffer, size,
       static_cast<int>(result));
  return flagcxUnhandledDeviceError;
#else
  return flagcxNotSupported;
#endif
}

// Compile-time fallback for SDKs without VMM declarations.
#if !FLAGCX_ILUVATAR_HAS_VMM
flagcxResult_t ixcudaAdaptorSymPhysAlloc(void *, size_t, void **, void *,
                                         size_t *, size_t *) {
  return flagcxNotSupported;
}
flagcxResult_t ixcudaAdaptorSymPhysFree(void *) { return flagcxNotSupported; }
flagcxResult_t ixcudaAdaptorSymFlatMap(void *[], int, int, void *, size_t,
                                       void **) {
  return flagcxNotSupported;
}
flagcxResult_t ixcudaAdaptorSymFlatUnmap(void *, size_t, int) {
  return flagcxNotSupported;
}
#endif
flagcxResult_t ixcudaAdaptorSymMulticastSupported(int *supported) {
  if (supported)
    *supported = 0;
  return flagcxSuccess;
}
flagcxResult_t ixcudaAdaptorSymMulticastCreate(size_t, int, const int *,
                                               void **, int *) {
  return flagcxNotSupported;
}
flagcxResult_t ixcudaAdaptorSymMulticastBind(void *, int, void *, size_t, int,
                                             int, void **, size_t *) {
  return flagcxNotSupported;
}
flagcxResult_t ixcudaAdaptorSymMulticastTeardown(void *, size_t) {
  return flagcxSuccess;
}
flagcxResult_t ixcudaAdaptorSymMulticastFree(void *) {
  return flagcxNotSupported;
}

// Consumes the pending device error: cudaGetLastError() both reads the record
// and clears it, so a later caller cannot mistake this failure for its own. The
// counterpart is ixcudaAdaptorLaunchKernel(), which deliberately leaves the
// sticky state set and reports the failure itself -- there is nothing left to
// log here, and logging again would report one failure twice.
// Measured on CoreX 5.1.0: the record is the per-thread slot that every runtime
// call overwrites, so a reader only sees it before the next runtime call.
flagcxResult_t ixcudaAdaptorGetLastError() {
  cudaError_t err = cudaGetLastError();
  return err == cudaSuccess ? flagcxSuccess : flagcxUnhandledDeviceError;
}

flagcxResult_t ixcudaAdaptorGetPointerType(const void *ptr, int *ptrType) {
  if (ptr == NULL || ptrType == NULL)
    return flagcxInvalidArgument;

#if FLAGCX_ILUVATAR_HAS_VMM
  void *allocationBase = nullptr;
  size_t allocationBytes = 0;
  const auto tracked =
      IxcudaVmmAllocation::addressRange(ptr, &allocationBase, &allocationBytes);
  if (tracked == flagcxSuccess) {
    *ptrType = FLAGCX_PTR_CUDA;
    return flagcxSuccess;
  }
  if (tracked != flagcxNotSupported)
    return tracked;
#endif
  cudaPointerAttributes attrs = {};
  cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
  if (err == cudaErrorInvalidValue) {
    cudaGetLastError();
    *ptrType = FLAGCX_PTR_HOST;
    return flagcxSuccess;
  }
  if (err != cudaSuccess) {
    cudaGetLastError();
    return flagcxUnhandledDeviceError;
  }
#if CUDART_VERSION >= 10000
  *ptrType = (attrs.type == cudaMemoryTypeDevice ||
              attrs.type == cudaMemoryTypeManaged)
                 ? FLAGCX_PTR_CUDA
                 : FLAGCX_PTR_HOST;
#else
  *ptrType = (attrs.memoryType == cudaMemoryTypeDevice || attrs.isManaged)
                 ? FLAGCX_PTR_CUDA
                 : FLAGCX_PTR_HOST;
#endif
  return flagcxSuccess;
}

struct flagcxDeviceAdaptor ixcudaAdaptor {
  "IXCUDA",
      // Basic functions
      ixcudaAdaptorDeviceSynchronize, ixcudaAdaptorDeviceMemcpy,
      ixcudaAdaptorDeviceMemset, ixcudaAdaptorDeviceMalloc,
      ixcudaAdaptorDeviceFree, ixcudaAdaptorSetDevice, ixcudaAdaptorGetDevice,
      ixcudaAdaptorGetDeviceCount, ixcudaAdaptorGetVendor,
      ixcudaAdaptorHostGetDevicePointer,
      // GDR functions
      NULL, // flagcxResult_t (*memHandleInit)(int dev_id, void **memHandle);
      NULL, // flagcxResult_t (*memHandleDestroy)(int dev, void *memHandle);
      ixcudaAdaptorGdrMemAlloc, ixcudaAdaptorGdrMemFree,
      NULL, // flagcxResult_t (*hostShareMemAlloc)(void **ptr, size_t size, void
            // *memHandle);
      NULL, // flagcxResult_t (*hostShareMemFree)(void *ptr, void *memHandle);
      NULL, // flagcxResult_t (*gdrPtrMmap)(void **pcpuptr, void *devptr, size_t
            // sz);
      NULL, // flagcxResult_t (*gdrPtrMunmap)(void *cpuptr, size_t sz);
      // Stream functions
      ixcudaAdaptorStreamCreate, ixcudaAdaptorStreamDestroy,
      ixcudaAdaptorStreamCopy, ixcudaAdaptorStreamFree,
      ixcudaAdaptorStreamSynchronize, ixcudaAdaptorStreamQuery,
      ixcudaAdaptorStreamWaitEvent, ixcudaAdaptorStreamWaitValue64,
      ixcudaAdaptorStreamWriteValue64,
      // Event functions
      ixcudaAdaptorEventCreate, ixcudaAdaptorEventDestroy,
      ixcudaAdaptorEventRecord, ixcudaAdaptorEventSynchronize,
      ixcudaAdaptorEventQuery, ixcudaAdaptorEventElapsedTime,
      // IpcMemHandle functions
      ixcudaAdaptorIpcMemHandleCreate, ixcudaAdaptorIpcMemHandleGet,
      ixcudaAdaptorIpcMemHandleOpen, ixcudaAdaptorIpcMemHandleClose,
      ixcudaAdaptorIpcMemHandleFree,
      // Kernel launch
      ixcudaAdaptorLaunchKernel, // flagcxResult_t (*launchKernel)(void *func,
                                 // unsigned int block_x, unsigned int block_y,
                                 // unsigned int block_z, unsigned int grid_x,
                                 // unsigned int grid_y, unsigned int grid_z,
                                 // void **args, size_t share_mem, void *stream,
                                 // void *memHandle);
      NULL,                      // flagcxResult_t (*copyArgsInit)(void **args);
      NULL,                      // flagcxResult_t (*copyArgsFree)(void *args);
      NULL, // flagcxResult_t (*launchDeviceFunc)(flagcxStream_t stream, void
            // *args);
      // Others
      ixcudaAdaptorGetDeviceProperties, // flagcxResult_t
                                        // (*getDeviceProperties)(struct
                                        // flagcxDeviceProps *props, int dev);
      ixcudaAdaptorGetDevicePciBusId,   // flagcxResult_t
                                        // (*getDevicePciBusId)(char *pciBusId,
                                        // int len, int dev);
      ixcudaAdaptorGetDeviceByPciBusId, // flagcxResult_t
                                        // (*getDeviceByPciBusId)(int *dev,
                                        // const char *pciBusId);
      ixcudaAdaptorLaunchHostFunc,

      // DMA buffer
      ixcudaAdaptorDmaSupport, ixcudaAdaptorGetHandleForAddressRange,
      ixcudaAdaptorHostRegister,   // flagcxResult_t (*hostRegister)(void *,
                                   // size_t);
      ixcudaAdaptorHostUnregister, // flagcxResult_t (*hostUnregister)(void *);
      // Symmetric memory VMM functions
      ixcudaAdaptorSymPhysAlloc, ixcudaAdaptorSymPhysFree,
      ixcudaAdaptorSymFlatMap, ixcudaAdaptorSymFlatUnmap,
      ixcudaAdaptorSymMulticastSupported, ixcudaAdaptorSymMulticastCreate,
      ixcudaAdaptorSymMulticastBind, ixcudaAdaptorSymMulticastTeardown,
      ixcudaAdaptorSymMulticastFree,
      ixcudaAdaptorGetLastError, // flagcxResult_t (*getLastError)();
      ixcudaAdaptorGetPointerType, ixcudaAdaptorGetAddressRange,
#if FLAGCX_ILUVATAR_HAS_VMM
      FLAGCX_VMM_MR_CAP_VA, // Candidate only; CoreX CI remains gated.
      FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE, NULL,
      ixcudaAdaptorSymFlatMappingUnmap, ixcudaAdaptorSymFlatVaFree, NULL, NULL,
#else
      FLAGCX_VMM_MR_CAP_NONE, FLAGCX_DEVICE_ADAPTOR_INTERNAL_NONE, NULL, NULL,
      NULL, NULL, NULL,
#endif
      // Preserve the legacy collective receive acquire until this platform
      // explicitly documents a coherent GPUDirect WRITE path.
      FLAGCX_GDR_WRITE_REQUIRES_FLUSH, FLAGCX_GDR_DEVICE_ILUVATAR, NULL,
      ixcudaAdaptorCanAccessPeer,
#if FLAGCX_ILUVATAR_HAS_VMM
      ixcudaAdaptorSymPhysDescribe, ixcudaAdaptorSymPhysExportSegment,
      ixcudaAdaptorSymPhysFreeSegments, ixcudaAdaptorSymFlatMapSegments,
      ixcudaAdaptorSymFlatMappingUnmapSegments,
      ixcudaAdaptorSymFlatVaFreeSegments, ixcudaAdaptorSymPhysSegmentCount,
      ixcudaAdaptorSymFlatMapSegmentsBegin, ixcudaAdaptorSymFlatMapSegmentOne,
      ixcudaAdaptorSymLsaGetCapacity, ixcudaAdaptorSymLsaVaReserve,
      ixcudaAdaptorSymLsaVaFree, ixcudaAdaptorSymLsaMapBeginAt, NULL, NULL,
      NULL, NULL,
#else
      NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
      NULL, NULL, NULL, NULL, NULL,
#endif
};
#endif // USE_ILUVATAR_ADAPTOR
