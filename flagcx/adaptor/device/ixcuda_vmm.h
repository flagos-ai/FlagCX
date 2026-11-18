#ifndef FLAGCX_IXCUDA_VMM_H_
#define FLAGCX_IXCUDA_VMM_H_

#include "param.h"
#include "segmented_vmm_allocation.h"
#include <climits>
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
    if (cuMemGetAddressRange(&base, &bytes, (CUdeviceptr)ptr) == CUDA_SUCCESS &&
        reinterpret_cast<uintptr_t>(ptr) >= base &&
        reinterpret_cast<uintptr_t>(ptr) - base < bytes &&
        size <= bytes - (reinterpret_cast<uintptr_t>(ptr) - base) &&
        cudaIpcGetMemHandle(&ipc, reinterpret_cast<void *>(base)) ==
            cudaSuccess) {
      delete bundle;
      *opaque = nullptr;
    } else {
      cudaGetLastError();
    }
    return flagcxNotSupported;
  }
  // A description owns references independently from the source allocation.
  // Export/import a reference rather than assuming CoreX declares the newer
  // cuMemRetainAllocationHandle API. At most one temporary FD is live.
  for (size_t i = 0; i < bundle->handles.size(); ++i) {
    int fd = -1;
    auto result = IxcudaVmmOps::exportHandle(bundle->handles[i], &fd);
    IxcudaVmmOps::Handle retained = {};
    if (result == flagcxSuccess)
      result = IxcudaVmmOps::importHandle(fd, &retained);
    if (fd >= 0)
      close(fd);
    if (result != flagcxSuccess)
      return result;
    bundle->handles[i] = retained;
    bundle->handleOwned[i] = 1;
  }
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
#endif // FLAGCX_IXCUDA_VMM_H_
