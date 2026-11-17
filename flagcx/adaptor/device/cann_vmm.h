#ifndef FLAGCX_CANN_VMM_H_
#define FLAGCX_CANN_VMM_H_

// CANN exports a uint64_t VMM token, whereas SymMem exchanges file
// descriptors. A private memfd carries that token through the existing FD
// exchange. The receiving process imports the token, never the memfd itself.
#include "segmented_vmm_allocation.h"
#include <cerrno>
#include <cstdint>
#include <limits>
#include <new>
#include <set>
#include <sys/syscall.h>
#include <unistd.h>
#include <unordered_map>

static flagcxResult_t cannVmmResult(aclError result, const char *operation) {
  if (result == ACL_SUCCESS)
    return flagcxSuccess;
  WARN("CANN VMM %s failed: ACL error %d", operation, (int)result);
  return result == ACL_ERROR_RT_FEATURE_NOT_SUPPORT
             ? flagcxNotSupported
             : flagcxUnhandledDeviceError;
}

struct CannVmmShareToken {
  uint64_t token;
  int32_t ownerDevice;
  uint32_t reserved;
};

struct CannSegmentOps {
  typedef aclrtDrvMemHandle Handle;
  typedef aclrtPhysicalMemProp Prop;

  static std::mutex &exportLock() {
    static std::mutex lock;
    return lock;
  }
  static std::unordered_map<Handle, CannVmmShareToken> &exports() {
    static std::unordered_map<Handle, CannVmmShareToken> cache;
    return cache;
  }
  struct ImportRecord {
    Handle handle;
    size_t refs;
  };
  static std::mutex &importLock() {
    static std::mutex lock;
    return lock;
  }
  static std::unordered_map<uint64_t, ImportRecord> &imports() {
    static std::unordered_map<uint64_t, ImportRecord> cache;
    return cache;
  }

  static flagcxResult_t deviceOrdinal(int *device) {
    return cannVmmResult(aclrtGetDevice(device), "aclrtGetDevice");
  }
  static flagcxResult_t createHandle(size_t size, const Prop &prop,
                                     Handle *handle) {
    return cannVmmResult(aclrtMallocPhysical(handle, size, &prop, 0),
                         "aclrtMallocPhysical");
  }
  static flagcxResult_t reserveAddress(size_t size, size_t alignment,
                                       uintptr_t *base) {
    (void)alignment; // CANN currently requires the reserved parameter to be 0.
    void *address = NULL;
    flagcxResult_t result =
        cannVmmResult(aclrtReserveMemAddress(&address, size, 0, NULL, 0),
                      "aclrtReserveMemAddress");
    if (result == flagcxSuccess)
      *base = reinterpret_cast<uintptr_t>(address);
    return result;
  }
  static flagcxResult_t freeAddress(uintptr_t base, size_t) {
    return cannVmmResult(aclrtReleaseMemAddress(reinterpret_cast<void *>(base)),
                         "aclrtReleaseMemAddress");
  }
  static flagcxResult_t mapSegment(uintptr_t address, size_t size,
                                   Handle handle) {
    return cannVmmResult(
        aclrtMapMem(reinterpret_cast<void *>(address), size, 0, handle, 0),
        "aclrtMapMem");
  }
  static flagcxResult_t unmapSegment(uintptr_t address, size_t) {
    return cannVmmResult(aclrtUnmapMem(reinterpret_cast<void *>(address)),
                         "aclrtUnmapMem");
  }
  static flagcxResult_t setAccess(uintptr_t address, size_t size) {
    int device = 0;
    FLAGCXCHECK(deviceOrdinal(&device));
    aclrtMemAccessDesc access = {};
    access.flags = ACL_RT_MEM_ACCESS_FLAGS_READWRITE;
    access.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device;
    return cannVmmResult(
        aclrtMemSetAccess(reinterpret_cast<void *>(address), size, &access, 1),
        "aclrtMemSetAccess");
  }
  static flagcxResult_t exportHandle(Handle handle, int *fd) {
    if (fd == NULL)
      return flagcxInvalidArgument;
    *fd = -1;
    CannVmmShareToken payload = {};
    {
      std::lock_guard<std::mutex> lock(exportLock());
      std::unordered_map<Handle, CannVmmShareToken>::const_iterator cached =
          exports().find(handle);
      if (cached != exports().end()) {
        payload = cached->second;
      } else {
        FLAGCXCHECK(deviceOrdinal(&payload.ownerDevice));
        FLAGCXCHECK(cannVmmResult(
            aclrtMemExportToShareableHandle(handle, ACL_MEM_HANDLE_TYPE_NONE,
                                            1 /* disable PID validation */,
                                            &payload.token),
            "aclrtMemExportToShareableHandle"));
        try {
          exports().emplace(handle, payload);
        } catch (const std::bad_alloc &) {
          return flagcxSystemError;
        }
      }
    }
#if defined(__linux__) && defined(SYS_memfd_create)
    int carrier = static_cast<int>(syscall(SYS_memfd_create, "flagcx-cann-vmm",
                                           0x0001U /* MFD_CLOEXEC */));
    if (carrier < 0)
      return flagcxSystemError;
    if (pwrite(carrier, &payload, sizeof(payload), 0) != sizeof(payload)) {
      close(carrier);
      return flagcxSystemError;
    }
    *fd = carrier;
    return flagcxSuccess;
#else
    return flagcxNotSupported;
#endif
  }
  static flagcxResult_t importHandle(int fd, Handle *handle) {
    if (fd < 0 || handle == NULL)
      return flagcxInvalidArgument;
    *handle = NULL;
    CannVmmShareToken payload = {};
    if (pread(fd, &payload, sizeof(payload), 0) != sizeof(payload))
      return flagcxSystemError;
    int device = 0;
    FLAGCXCHECK(deviceOrdinal(&device));
    if (payload.ownerDevice != device) {
      static std::mutex peerLock;
      static std::set<uint64_t> enabledPeers;
      const uint64_t pair =
          (static_cast<uint64_t>(static_cast<uint32_t>(device)) << 32) |
          static_cast<uint32_t>(payload.ownerDevice);
      std::lock_guard<std::mutex> lock(peerLock);
      if (enabledPeers.find(pair) == enabledPeers.end()) {
        int32_t canAccess = 0;
        FLAGCXCHECK(cannVmmResult(
            aclrtDeviceCanAccessPeer(&canAccess, device, payload.ownerDevice),
            "aclrtDeviceCanAccessPeer"));
        if (canAccess == 0)
          return flagcxNotSupported;
        FLAGCXCHECK(
            cannVmmResult(aclrtDeviceEnablePeerAccess(payload.ownerDevice, 0),
                          "aclrtDeviceEnablePeerAccess"));
        enabledPeers.insert(pair);
      }
    }
    std::lock_guard<std::mutex> lock(importLock());
    std::unordered_map<uint64_t, ImportRecord>::iterator cached =
        imports().find(payload.token);
    if (cached != imports().end()) {
      ++cached->second.refs;
      *handle = cached->second.handle;
      return flagcxSuccess;
    }
    flagcxResult_t imported = cannVmmResult(
        aclrtMemImportFromShareableHandle(payload.token, device, handle),
        "aclrtMemImportFromShareableHandle");
    if (imported != flagcxSuccess) {
      if (*handle != NULL)
        (void)aclrtFreePhysical(*handle);
      *handle = NULL;
      return imported;
    }
    try {
      imports().emplace(payload.token, ImportRecord{*handle, 1});
    } catch (const std::bad_alloc &) {
      (void)aclrtFreePhysical(*handle);
      *handle = NULL;
      return flagcxSystemError;
    }
    return flagcxSuccess;
  }
  static flagcxResult_t releaseHandle(Handle handle) {
    {
      std::lock_guard<std::mutex> lock(importLock());
      for (std::unordered_map<uint64_t, ImportRecord>::iterator it =
               imports().begin();
           it != imports().end(); ++it) {
        if (it->second.handle != handle)
          continue;
        if (it->second.refs > 1) {
          --it->second.refs;
          return flagcxSuccess;
        }
        FLAGCXCHECK(cannVmmResult(aclrtFreePhysical(handle),
                                  "aclrtFreePhysical(imported)"));
        imports().erase(it);
        return flagcxSuccess;
      }
    }
    FLAGCXCHECK(cannVmmResult(aclrtFreePhysical(handle), "aclrtFreePhysical"));
    std::lock_guard<std::mutex> lock(exportLock());
    exports().erase(handle);
    return flagcxSuccess;
  }
};

typedef flagcxSegmentedVmmAllocation<CannSegmentOps> CannSegmentAllocation;
typedef flagcxSegmentedVmmMap<CannSegmentOps> CannSegmentMap;

static flagcxResult_t cannVmmGranularity(aclrtPhysicalMemProp prop,
                                         size_t *granularity) {
  return cannVmmResult(
      aclrtMemGetAllocationGranularity(
          &prop, ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM, granularity),
      "aclrtMemGetAllocationGranularity");
}

static flagcxResult_t cannVmmAllocationProp(aclrtPhysicalMemProp *prop,
                                            size_t *granularity) {
  if (prop == NULL || granularity == NULL)
    return flagcxInvalidArgument;
  int device = 0;
  FLAGCXCHECK(CannSegmentOps::deviceOrdinal(&device));
  *prop = {};
  prop->handleType = ACL_MEM_HANDLE_TYPE_NONE;
  prop->allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
  prop->location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  prop->location.id = device;
  prop->memAttr = ACL_HBM_MEM_NORMAL;
  return cannVmmGranularity(*prop, granularity);
}

static flagcxResult_t cannVmmAllocate(void **ptr, size_t size) {
  if (ptr == NULL || size == 0)
    return flagcxInvalidArgument;
  *ptr = NULL;
  aclrtPhysicalMemProp prop = {};
  size_t granularity = 0;
  FLAGCXCHECK(cannVmmAllocationProp(&prop, &granularity));
  if (granularity == 0 ||
      size > std::numeric_limits<size_t>::max() - (granularity - 1))
    return flagcxInvalidArgument;
  size_t alignedSize = ((size + granularity - 1) / granularity) * granularity;
  size_t segmentBytes = 0;
  FLAGCXCHECK(flagcxVmmSegmentBytes(granularity, &segmentBytes));
  bool multiEnabled = true;
  FLAGCXCHECK(flagcxVmmMultiSegmentEnabled(true, &multiEnabled));
  if (segmentBytes != 0 && segmentBytes < alignedSize && !multiEnabled)
    return flagcxNotSupported;
  if (segmentBytes == 0 || segmentBytes > alignedSize)
    segmentBytes = alignedSize;
  int device = 0;
  FLAGCXCHECK(CannSegmentOps::deviceOrdinal(&device));
  FLAGCXCHECK(CannSegmentAllocation::retryPending(device));
  return CannSegmentAllocation::allocate(ptr, alignedSize, granularity,
                                         segmentBytes, prop);
}

static flagcxResult_t cannVmmDescribe(void *ptr, size_t size, void **opaque,
                                      const flagcxSymSegmentMeta **segments,
                                      int *count, size_t *userOffset,
                                      size_t *alignment) {
  if (ptr == NULL || size == 0 || opaque == NULL || segments == NULL ||
      count == NULL || userOffset == NULL || alignment == NULL)
    return flagcxInvalidArgument;
  *opaque = NULL;
  *segments = NULL;
  *count = 0;
  *userOffset = 0;
  *alignment = 0;
  CannSegmentMap::Bundle *bundle = new (std::nothrow) CannSegmentMap::Bundle();
  if (bundle == NULL)
    return flagcxSystemError;
  bool found = false;
  flagcxResult_t result =
      CannSegmentAllocation::describe(ptr, size, bundle, userOffset, &found);
  if (!found && result == flagcxSuccess) {
    delete bundle;
    return flagcxNotSupported;
  }
  *opaque = bundle;
  if (result != flagcxSuccess)
    return result;
  aclrtPhysicalMemProp prop = {};
  result = cannVmmAllocationProp(&prop, alignment);
  if (result != flagcxSuccess)
    return result;
  if (bundle->segments.empty() ||
      bundle->segments.size() >
          static_cast<size_t>(std::numeric_limits<int>::max()))
    return flagcxInvalidUsage;
  *segments = bundle->segments.data();
  *count = static_cast<int>(bundle->segments.size());
  return flagcxSuccess;
}

#endif // FLAGCX_CANN_VMM_H_
