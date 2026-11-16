#ifndef FLAGCX_SEGMENTED_VMM_MAP_H_
#define FLAGCX_SEGMENTED_VMM_MAP_H_

// Adaptor-private bookkeeping for providers with a CUDA-style VMM API. Ops
// supplies the provider calls; no provider handle enters the symmetric heap.
#include "flagcx_device_adaptor.h"
#include <cstdint>
#include <limits>
#include <mutex>
#include <new>
#include <unordered_map>
#include <utility>
#include <vector>

template <typename Handle>
struct flagcxSegmentedVmmBundle {
  std::vector<flagcxSymSegmentMeta> segments;
  std::vector<Handle> handles;
  std::vector<unsigned char> handleOwned;
};

template <typename Ops>
class flagcxSegmentedVmmMap {
public:
  typedef typename Ops::Handle Handle;
  typedef flagcxSegmentedVmmBundle<Handle> Bundle;

  static flagcxResult_t exportSegment(void *opaque, int index, int *fd) {
    Bundle *bundle = static_cast<Bundle *>(opaque);
    if (bundle == NULL || fd == NULL || index < 0 ||
        static_cast<size_t>(index) >= bundle->handles.size())
      return flagcxInvalidArgument;
    *fd = -1;
    return Ops::exportHandle(bundle->handles[index], fd);
  }

  static flagcxResult_t freeBundle(void *opaque) {
    Bundle *bundle = static_cast<Bundle *>(opaque);
    if (bundle == NULL)
      return flagcxSuccess;
    for (size_t i = 0;
         i < bundle->handles.size() && i < bundle->handleOwned.size(); i++) {
      if (!bundle->handleOwned[i])
        continue;
      flagcxResult_t result = Ops::releaseHandle(bundle->handles[i]);
      if (result != flagcxSuccess)
        return result;
      bundle->handleOwned[i] = 0;
    }
    delete bundle;
    return flagcxSuccess;
  }

  static flagcxResult_t begin(int nPeers, int self, void *opaque,
                              const flagcxSymRankLayout *layouts,
                              const flagcxSymSegmentMeta *segments,
                              int totalSegments, size_t stride,
                              size_t alignment, void **flatBase) {
    return beginImpl(NULL, true, nPeers, self, opaque, layouts, segments,
                     totalSegments, stride, alignment, flatBase);
  }

  // The communicator owns the surrounding VA reservation. This mapping
  // record owns only physical aliases and imported handles in that range.
  static flagcxResult_t beginAt(void *reservedWindowBase, int nPeers, int self,
                                void *opaque,
                                const flagcxSymRankLayout *layouts,
                                const flagcxSymSegmentMeta *segments,
                                int totalSegments, size_t stride,
                                size_t alignment, void **flatBase) {
    if (reservedWindowBase == NULL)
      return flagcxInvalidArgument;
    return beginImpl(reservedWindowBase, false, nPeers, self, opaque, layouts,
                     segments, totalSegments, stride, alignment, flatBase);
  }

  static flagcxResult_t beginImpl(void *requestedBase, bool reserveVa,
                                  int nPeers, int self, void *opaque,
                                  const flagcxSymRankLayout *layouts,
                                  const flagcxSymSegmentMeta *segments,
                                  int totalSegments, size_t stride,
                                  size_t alignment, void **flatBase) {
    Bundle *bundle = static_cast<Bundle *>(opaque);
    if (bundle == NULL || layouts == NULL || segments == NULL ||
        flatBase == NULL || nPeers <= 0 || self < 0 || self >= nPeers ||
        totalSegments <= 0 || stride == 0 || alignment == 0 ||
        (alignment & (alignment - 1)) != 0 || stride % alignment != 0 ||
        stride >
            std::numeric_limits<size_t>::max() / static_cast<size_t>(nPeers) ||
        bundle->handles.size() >
            static_cast<size_t>(std::numeric_limits<int>::max()) ||
        layouts[self].segmentCount != static_cast<int>(bundle->handles.size()))
      return flagcxInvalidArgument;
    *flatBase = NULL;
    Flat initial;
    initial.totalSize = stride * static_cast<size_t>(nPeers);
    initial.vaOwned = reserveVa;
    initial.self = self;
    initial.bundle = bundle;
    try {
      initial.segments.resize(totalSegments);
    } catch (const std::bad_alloc &) {
      return flagcxSystemError;
    }
    for (int rank = 0; rank < nPeers; rank++) {
      const flagcxSymRankLayout &layout = layouts[rank];
      if (layout.firstSegment < 0 || layout.segmentCount <= 0 ||
          layout.firstSegment > totalSegments ||
          layout.segmentCount > totalSegments - layout.firstSegment ||
          static_cast<size_t>(rank) >
              std::numeric_limits<size_t>::max() / stride)
        return flagcxInvalidArgument;
      for (int i = 0; i < layout.segmentCount; i++) {
        const int index = layout.firstSegment + i;
        const flagcxSymSegmentMeta &segment = segments[index];
        if (segment.size == 0 || segment.offset > stride ||
            segment.size > stride - segment.offset)
          return flagcxInvalidArgument;
        Segment &entry = initial.segments[index];
        entry.offset = static_cast<size_t>(rank) * stride + segment.offset;
        entry.size = segment.size;
        entry.ownerRank = rank;
        entry.ownerLocalIndex = i;
      }
    }

    uintptr_t base = reinterpret_cast<uintptr_t>(requestedBase);
    if (reserveVa) {
      flagcxResult_t result =
          Ops::reserveAddress(initial.totalSize, alignment, &base);
      if (result != flagcxSuccess)
        return result;
    }
    if (base == 0 || base % alignment != 0 ||
        base > std::numeric_limits<uintptr_t>::max() - initial.totalSize) {
      if (reserveVa)
        Ops::freeAddress(base, initial.totalSize);
      return flagcxInvalidArgument;
    }
    std::lock_guard<std::mutex> lock(mutex());
    bool inserted = false;
    try {
      inserted =
          mappings()
              .emplace(reinterpret_cast<void *>(base), std::move(initial))
              .second;
    } catch (const std::bad_alloc &) {
    }
    if (!inserted) {
      if (reserveVa)
        Ops::freeAddress(base, initial.totalSize);
      return flagcxSystemError;
    }
    // Publish the VA before any fallible import/map call. The common rollback
    // path then has a token for every partial provider operation.
    *flatBase = reinterpret_cast<void *>(base);
    return flagcxSuccess;
  }

  static flagcxResult_t mapOne(void *flatBase, int index, int fd) {
    if (flatBase == NULL || index < 0)
      return flagcxInvalidArgument;
    std::lock_guard<std::mutex> lock(mutex());
    typename std::unordered_map<void *, Flat>::iterator it =
        mappings().find(flatBase);
    if (it == mappings().end() ||
        static_cast<size_t>(index) >= it->second.segments.size())
      return flagcxInvalidArgument;
    Flat &mapping = it->second;
    Segment &entry = mapping.segments[index];
    if (entry.mapped || entry.importedOwned)
      return flagcxInvalidUsage;
    flagcxResult_t result = flagcxSuccess;
    if (entry.ownerRank == mapping.self) {
      if (fd != -1)
        return flagcxInvalidArgument;
      entry.handle = mapping.bundle->handles[entry.ownerLocalIndex];
    } else {
      if (fd < 0)
        return flagcxInvalidArgument;
      result = Ops::importHandle(fd, &entry.handle);
      if (result != flagcxSuccess)
        return result;
      entry.importedOwned = true;
    }
    result =
        Ops::mapSegment(reinterpret_cast<uintptr_t>(flatBase) + entry.offset,
                        entry.size, entry.handle);
    if (result != flagcxSuccess)
      return result;
    entry.mapped = true;
    result = Ops::setAccess(
        reinterpret_cast<uintptr_t>(flatBase) + entry.offset, entry.size);
    if (result != flagcxSuccess)
      return result;
    return flagcxSuccess;
  }

  static flagcxResult_t map(const int *fds, int nPeers, int self, void *opaque,
                            const flagcxSymRankLayout *layouts,
                            const flagcxSymSegmentMeta *segments,
                            int totalSegments, size_t stride, size_t alignment,
                            void **flatBase) {
    if (fds == NULL)
      return flagcxInvalidArgument;
    flagcxResult_t result = begin(nPeers, self, opaque, layouts, segments,
                                  totalSegments, stride, alignment, flatBase);
    if (result != flagcxSuccess)
      return result;
    for (int rank = 0; rank < nPeers; rank++) {
      const flagcxSymRankLayout &layout = layouts[rank];
      for (int i = 0; i < layout.segmentCount; i++) {
        const int index = layout.firstSegment + i;
        result = mapOne(*flatBase, index, rank == self ? -1 : fds[index]);
        if (result != flagcxSuccess)
          return result;
      }
    }
    return flagcxSuccess;
  }

  static flagcxResult_t unmap(void *flatBase) {
    if (flatBase == NULL)
      return flagcxSuccess;
    std::lock_guard<std::mutex> lock(mutex());
    typename std::unordered_map<void *, Flat>::iterator it =
        mappings().find(flatBase);
    if (it == mappings().end())
      return flagcxInvalidUsage;
    Flat &mapping = it->second;
    for (size_t i = 0; i < mapping.segments.size(); i++) {
      Segment &entry = mapping.segments[i];
      if (!entry.mapped)
        continue;
      flagcxResult_t result = Ops::unmapSegment(
          reinterpret_cast<uintptr_t>(flatBase) + entry.offset, entry.size);
      if (result != flagcxSuccess)
        return result;
      entry.mapped = false;
    }
    // A failed unmap leaves all imported handles alive for a later retry.
    for (size_t i = 0; i < mapping.segments.size(); i++) {
      Segment &entry = mapping.segments[i];
      if (!entry.importedOwned)
        continue;
      flagcxResult_t result = Ops::releaseHandle(entry.handle);
      if (result != flagcxSuccess)
        return result;
      entry.importedOwned = false;
    }
    return flagcxSuccess;
  }

  static flagcxResult_t freeVa(void *flatBase) {
    if (flatBase == NULL)
      return flagcxSuccess;
    std::lock_guard<std::mutex> lock(mutex());
    typename std::unordered_map<void *, Flat>::iterator it =
        mappings().find(flatBase);
    if (it == mappings().end())
      return flagcxInvalidUsage;
    Flat &mapping = it->second;
    for (size_t i = 0; i < mapping.segments.size(); i++) {
      if (mapping.segments[i].mapped || mapping.segments[i].importedOwned)
        return flagcxInvalidUsage;
    }
    if (mapping.vaOwned) {
      flagcxResult_t result = Ops::freeAddress(
          reinterpret_cast<uintptr_t>(flatBase), mapping.totalSize);
      if (result != flagcxSuccess)
        return result;
      mapping.vaOwned = false;
    }
    mappings().erase(it);
    return flagcxSuccess;
  }

private:
  struct Segment {
    size_t offset = 0;
    size_t size = 0;
    int ownerRank = -1;
    int ownerLocalIndex = -1;
    Handle handle = Handle();
    bool mapped = false;
    bool importedOwned = false;
  };
  struct Flat {
    size_t totalSize = 0;
    bool vaOwned = false;
    int self = -1;
    Bundle *bundle = NULL;
    std::vector<Segment> segments;
  };
  static std::mutex &mutex() {
    static std::mutex instance;
    return instance;
  }
  static std::unordered_map<void *, Flat> &mappings() {
    static std::unordered_map<void *, Flat> instance;
    return instance;
  }
};

#endif // FLAGCX_SEGMENTED_VMM_MAP_H_
