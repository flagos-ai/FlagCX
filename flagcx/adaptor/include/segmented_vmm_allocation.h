#ifndef FLAGCX_SEGMENTED_VMM_ALLOCATION_H_
#define FLAGCX_SEGMENTED_VMM_ALLOCATION_H_

#include "segmented_vmm_map.h"
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>

// Read this for each allocation/description: tests and applications may
// change the setting after an allocation has already been created. DU defaults
// to a single physical VMM allocation until multi-segment mapping is proven
// on its supported drivers; other providers keep their current default.
static inline flagcxResult_t flagcxVmmMultiSegmentEnabled(bool enabledByDefault,
                                                          bool *enabled) {
  if (enabled == NULL)
    return flagcxInvalidArgument;
  const char *setting = std::getenv("FLAGCX_VMM_MULTI_SEGMENT_ENABLE");
  if (setting == NULL || setting[0] == '\0') {
    *enabled = enabledByDefault;
    return flagcxSuccess;
  }
  if (std::strcmp(setting, "0") == 0 || std::strcmp(setting, "1") == 0) {
    *enabled = setting[0] == '1';
    return flagcxSuccess;
  }
  return flagcxInvalidArgument;
}

// A disabled setting leaves the existing single-allocation path unchanged.
static inline flagcxResult_t flagcxVmmSegmentBytes(size_t granularity,
                                                   size_t *segmentBytes) {
  if (segmentBytes == NULL || granularity == 0)
    return flagcxInvalidArgument;
  *segmentBytes = 0;
  const char *setting = std::getenv("FLAGCX_VMM_SEGMENT_BYTES");
  if (setting == NULL || setting[0] == '\0')
    return flagcxSuccess;
  errno = 0;
  char *end = NULL;
  unsigned long long parsed = std::strtoull(setting, &end, 10);
  if (setting[0] == '-' || errno != 0 || end == setting || *end != '\0' ||
      parsed == 0 || parsed > std::numeric_limits<size_t>::max() ||
      parsed < granularity || parsed % granularity != 0)
    return flagcxInvalidArgument;
  *segmentBytes = static_cast<size_t>(parsed);
  return flagcxSuccess;
}

template <typename Ops>
class flagcxSegmentedVmmAllocation {
public:
  typedef typename Ops::Handle Handle;
  typedef typename Ops::Prop Prop;
  typedef flagcxSegmentedVmmBundle<Handle> Bundle;

  static flagcxResult_t allocate(void **ptr, size_t size, size_t granularity,
                                 size_t segmentBytes, const Prop &prop) {
    if (ptr == NULL || size == 0 || granularity == 0 ||
        segmentBytes < granularity || segmentBytes % granularity != 0)
      return flagcxInvalidArgument;
    *ptr = NULL;
    int device = 0;
    flagcxResult_t result = Ops::deviceOrdinal(&device);
    if (result != flagcxSuccess)
      return result;
    Allocation initial;
    initial.size = size;
    initial.device = device;
    initial.vaOwned = true;
    try {
      for (size_t offset = 0; offset < size;) {
        Segment segment;
        segment.offset = offset;
        segment.size = std::min(segmentBytes, size - offset);
        initial.segments.push_back(segment);
        offset += segment.size;
      }
    } catch (const std::bad_alloc &) {
      return flagcxSystemError;
    }
    uintptr_t base = 0;
    result = Ops::reserveAddress(size, granularity, &base);
    if (result != flagcxSuccess)
      return result;
    std::lock_guard<std::mutex> lock(mutex());
    bool inserted = false;
    try {
      inserted =
          allocations()
              .emplace(reinterpret_cast<void *>(base), std::move(initial))
              .second;
    } catch (const std::bad_alloc &) {
    }
    if (!inserted) {
      Ops::freeAddress(base, size);
      return flagcxSystemError;
    }
    typename std::unordered_map<void *, Allocation>::iterator it =
        allocations().find(reinterpret_cast<void *>(base));
    Allocation &allocation = it->second;
    for (size_t i = 0; i < allocation.segments.size(); i++) {
      Segment &segment = allocation.segments[i];
      result = Ops::createHandle(segment.size, prop, &segment.handle);
      if (result != flagcxSuccess)
        break;
      segment.handleOwned = true;
      result =
          Ops::mapSegment(base + segment.offset, segment.size, segment.handle);
      if (result != flagcxSuccess)
        break;
      segment.mapped = true;
      result = Ops::setAccess(base + segment.offset, segment.size);
      if (result != flagcxSuccess)
        break;
    }
    if (result != flagcxSuccess) {
      allocation.pendingCleanup = true;
      if (release(base, allocation) == flagcxSuccess)
        allocations().erase(it);
      return result;
    }
    *ptr = reinterpret_cast<void *>(base);
    return flagcxSuccess;
  }

  static flagcxResult_t freeAllocation(void *ptr, bool *found) {
    if (found == NULL)
      return flagcxInvalidArgument;
    *found = false;
    std::lock_guard<std::mutex> lock(mutex());
    typename std::unordered_map<void *, Allocation>::iterator it =
        allocations().find(ptr);
    if (it == allocations().end())
      return flagcxSuccess;
    *found = true;
    flagcxResult_t result =
        release(reinterpret_cast<uintptr_t>(ptr), it->second);
    if (result == flagcxSuccess)
      allocations().erase(it);
    return result;
  }

  static flagcxResult_t retryPending(int device) {
    std::lock_guard<std::mutex> lock(mutex());
    for (typename std::unordered_map<void *, Allocation>::iterator it =
             allocations().begin();
         it != allocations().end();) {
      Allocation &allocation = it->second;
      if (!allocation.pendingCleanup || allocation.device != device) {
        ++it;
        continue;
      }
      flagcxResult_t result =
          release(reinterpret_cast<uintptr_t>(it->first), allocation);
      if (result != flagcxSuccess)
        return result;
      it = allocations().erase(it);
    }
    return flagcxSuccess;
  }

  static flagcxResult_t describe(void *ptr, size_t size, Bundle *bundle,
                                 size_t *userOffset, bool *found) {
    if (ptr == NULL || size == 0 || bundle == NULL || userOffset == NULL ||
        found == NULL)
      return flagcxInvalidArgument;
    *found = false;
    const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
    std::lock_guard<std::mutex> lock(mutex());
    for (typename std::unordered_map<void *, Allocation>::const_iterator it =
             allocations().begin();
         it != allocations().end(); ++it) {
      const uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
      const Allocation &allocation = it->second;
      if (address < base || address - base > allocation.size ||
          size > allocation.size - (address - base))
        continue;
      *found = true;
      if (allocation.pendingCleanup)
        return flagcxInvalidUsage;
      const size_t begin = address - base;
      const size_t end = begin + size;
      size_t firstOffset = allocation.size;
      size_t coveredTo = begin;
      try {
        for (size_t i = 0; i < allocation.segments.size(); i++) {
          const Segment &segment = allocation.segments[i];
          if (segment.offset > allocation.size ||
              segment.size > allocation.size - segment.offset)
            return flagcxInvalidUsage;
          const size_t segmentEnd = segment.offset + segment.size;
          if (segment.offset >= end || segmentEnd <= begin)
            continue;
          if (segment.offset > coveredTo)
            return flagcxInvalidUsage;
          if (!segment.mapped || !segment.handleOwned)
            return flagcxInvalidUsage;
          if (bundle->segments.empty())
            firstOffset = segment.offset;
          if (segment.offset < firstOffset)
            return flagcxInvalidUsage;
          bundle->segments.push_back(
              flagcxSymSegmentMeta{segment.offset - firstOffset, segment.size});
          bundle->handleOwned.push_back(0);
          bundle->handles.push_back(segment.handle);
          if (segmentEnd > coveredTo)
            coveredTo = segmentEnd;
        }
      } catch (const std::bad_alloc &) {
        return flagcxSystemError;
      }
      if (bundle->segments.empty() || coveredTo < end)
        return flagcxInvalidUsage;
      *userOffset = begin - firstOffset;
      return flagcxSuccess;
    }
    return flagcxSuccess;
  }

  static flagcxResult_t segmentCount(const void *ptr, size_t size, int *count) {
    if (ptr == NULL || size == 0 || count == NULL)
      return flagcxInvalidArgument;
    *count = 0;
    const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
    std::lock_guard<std::mutex> lock(mutex());
    for (typename std::unordered_map<void *, Allocation>::const_iterator it =
             allocations().begin();
         it != allocations().end(); ++it) {
      const uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
      const Allocation &allocation = it->second;
      if (address < base || address - base > allocation.size ||
          size > allocation.size - (address - base))
        continue;
      if (allocation.pendingCleanup ||
          allocation.segments.size() >
              static_cast<size_t>(std::numeric_limits<int>::max()))
        return flagcxInvalidUsage;
      const size_t begin = address - base;
      const size_t end = begin + size;
      for (size_t i = 0; i < allocation.segments.size(); ++i) {
        const Segment &segment = allocation.segments[i];
        if (segment.offset > allocation.size ||
            segment.size > allocation.size - segment.offset)
          return flagcxInvalidUsage;
        if (segment.offset < end && segment.offset + segment.size > begin)
          ++*count;
      }
      if (*count == 0)
        return flagcxInvalidUsage;
      return flagcxSuccess;
    }
    return flagcxNotSupported;
  }

  // Return the lengths of the pieces of a tracked range that lie in each
  // physical segment. The caller may issue one provider operation per piece
  // without holding the allocation registry lock during provider calls.
  static flagcxResult_t segmentChunks(const void *ptr, size_t size,
                                      std::vector<size_t> *chunks,
                                      bool *found) {
    if (ptr == NULL || size == 0 || chunks == NULL || found == NULL)
      return flagcxInvalidArgument;
    chunks->clear();
    *found = false;
    const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
    std::lock_guard<std::mutex> lock(mutex());
    for (typename std::unordered_map<void *, Allocation>::const_iterator it =
             allocations().begin();
         it != allocations().end(); ++it) {
      const uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
      const Allocation &allocation = it->second;
      if (address < base || address - base >= allocation.size)
        continue;
      *found = true;
      size_t cursor = address - base;
      if (allocation.pendingCleanup || size > allocation.size - cursor)
        return flagcxInvalidUsage;
      size_t remaining = size;
      try {
        for (size_t i = 0; i < allocation.segments.size() && remaining != 0;
             i++) {
          const Segment &segment = allocation.segments[i];
          if (segment.offset > allocation.size ||
              segment.size > allocation.size - segment.offset)
            return flagcxInvalidUsage;
          const size_t end = segment.offset + segment.size;
          if (end <= cursor)
            continue;
          if (segment.offset > cursor || !segment.mapped)
            return flagcxInvalidUsage;
          const size_t chunk = std::min(remaining, end - cursor);
          chunks->push_back(chunk);
          cursor += chunk;
          remaining -= chunk;
        }
      } catch (const std::bad_alloc &) {
        return flagcxSystemError;
      }
      return remaining == 0 ? flagcxSuccess : flagcxInvalidUsage;
    }
    return flagcxSuccess;
  }

private:
  struct Segment {
    size_t offset = 0;
    size_t size = 0;
    Handle handle = Handle();
    bool mapped = false;
    bool handleOwned = false;
  };
  struct Allocation {
    size_t size = 0;
    int device = -1;
    bool vaOwned = false;
    bool pendingCleanup = false;
    std::vector<Segment> segments;
  };
  static flagcxResult_t release(uintptr_t base, Allocation &allocation) {
    for (size_t i = 0; i < allocation.segments.size(); i++) {
      Segment &segment = allocation.segments[i];
      if (!segment.mapped)
        continue;
      flagcxResult_t result =
          Ops::unmapSegment(base + segment.offset, segment.size);
      if (result != flagcxSuccess)
        return result;
      segment.mapped = false;
    }
    if (allocation.vaOwned) {
      flagcxResult_t result = Ops::freeAddress(base, allocation.size);
      if (result != flagcxSuccess)
        return result;
      allocation.vaOwned = false;
    }
    for (size_t i = 0; i < allocation.segments.size(); i++) {
      Segment &segment = allocation.segments[i];
      if (!segment.handleOwned)
        continue;
      flagcxResult_t result = Ops::releaseHandle(segment.handle);
      if (result != flagcxSuccess)
        return result;
      segment.handleOwned = false;
    }
    return flagcxSuccess;
  }
  static std::mutex &mutex() {
    static std::mutex instance;
    return instance;
  }
  static std::unordered_map<void *, Allocation> &allocations() {
    static std::unordered_map<void *, Allocation> instance;
    return instance;
  }
};

#endif // FLAGCX_SEGMENTED_VMM_ALLOCATION_H_
