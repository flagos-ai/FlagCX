#ifndef FLAGCX_HGGC_VMM_H_
#define FLAGCX_HGGC_VMM_H_

// PPU-private VMM bookkeeping. Every successful provider operation records
// ownership before the next fallible call; cleanup keeps the record on error.
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct hggcVmmSegment {
  size_t offset;
  size_t size;
  HGmemGenericAllocationHandle handle;
  int mapped;
  int handleOwned;
  int ownerRank;
  int ownerLocalIndex;
};

struct hggcVmmAllocation {
  struct hggcVmmAllocation *next;
  uintptr_t base;
  size_t size;
  size_t count;
  int device;
  int vaOwned;
  int pendingCleanup;
  struct hggcVmmSegment *segments;
};

struct hggcVmmBundle {
  size_t count;
  struct flagcxSymSegmentMeta *segments;
  HGmemGenericAllocationHandle *handles;
  unsigned char *handleOwned;
};

struct hggcVmmFlat {
  struct hggcVmmFlat *next;
  uintptr_t base;
  size_t size;
  int count;
  int self;
  int vaOwned;
  struct hggcVmmBundle *bundle;
  struct hggcVmmSegment *segments;
};

static pthread_mutex_t hggcVmmAllocationMutex = PTHREAD_MUTEX_INITIALIZER;
static struct hggcVmmAllocation *hggcVmmAllocations;
static pthread_mutex_t hggcVmmFlatMutex = PTHREAD_MUTEX_INITIALIZER;
static struct hggcVmmFlat *hggcVmmFlats;

static flagcxResult_t hggcVmmMultiSegmentEnabled(int *enabled) {
  const char *setting = getenv("FLAGCX_VMM_MULTI_SEGMENT_ENABLE");
  *enabled = 1;
  if (setting == NULL || *setting == '\0')
    return flagcxSuccess;
  if (strcmp(setting, "0") != 0 && strcmp(setting, "1") != 0)
    return flagcxInvalidArgument;
  *enabled = setting[0] == '1';
  return flagcxSuccess;
}

static flagcxResult_t hggcVmmSegmentBytes(size_t granularity, size_t *bytes) {
  const char *setting = getenv("FLAGCX_VMM_SEGMENT_BYTES");
  char *end = NULL;
  unsigned long long parsed;
  if (granularity == 0 || bytes == NULL)
    return flagcxInvalidArgument;
  *bytes = 0;
  if (setting == NULL || *setting == '\0')
    return flagcxSuccess;
  errno = 0;
  parsed = strtoull(setting, &end, 10);
  if (*setting == '-' || errno != 0 || end == setting || *end != '\0' ||
      parsed == 0 || parsed > SIZE_MAX || parsed < granularity ||
      parsed % granularity != 0)
    return flagcxInvalidArgument;
  *bytes = (size_t)parsed;
  return flagcxSuccess;
}

static flagcxResult_t hggcVmmReserveAddress(size_t size, size_t alignment,
                                            uintptr_t *base) {
  HGdeviceptr address = 0;
  HGDRVCHECK(hgMemAddressReserve(&address, size, alignment, 0, 0));
  *base = (uintptr_t)address;
  return flagcxSuccess;
}

static flagcxResult_t hggcVmmFreeAddress(uintptr_t base, size_t size) {
  HGDRVCHECK(hgMemAddressFree((HGdeviceptr)base, size));
  return flagcxSuccess;
}

static flagcxResult_t hggcVmmSetAccess(uintptr_t address, size_t size) {
  int device = 0;
  HGmemAccessDesc access;
  memset(&access, 0, sizeof(access));
  HGGCCHECK(hggcGetDevice(&device));
  access.location.type = HG_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = device;
  access.flags = HG_MEM_ACCESS_FLAGS_PROT_READWRITE;
  HGDRVCHECK(hgMemSetAccess((HGdeviceptr)address, size, &access, 1));
  return flagcxSuccess;
}

// Caller holds the allocation mutex. Unmap all aliases before releasing VA
// or handles, and clear each ownership flag only after its release succeeds.
static flagcxResult_t hggcVmmReleaseAllocation(struct hggcVmmAllocation *a) {
  for (size_t i = 0; i < a->count; i++) {
    struct hggcVmmSegment *s = &a->segments[i];
    if (s->mapped) {
      HGDRVCHECK(hgMemUnmap((HGdeviceptr)(a->base + s->offset), s->size));
      s->mapped = 0;
    }
  }
  if (a->vaOwned) {
    FLAGCXCHECK(hggcVmmFreeAddress(a->base, a->size));
    a->vaOwned = 0;
  }
  for (size_t i = 0; i < a->count; i++) {
    struct hggcVmmSegment *s = &a->segments[i];
    if (s->handleOwned) {
      HGDRVCHECK(hgMemRelease(s->handle));
      s->handleOwned = 0;
    }
  }
  return flagcxSuccess;
}

static void hggcVmmDeleteAllocation(struct hggcVmmAllocation **link) {
  struct hggcVmmAllocation *a = *link;
  *link = a->next;
  free(a->segments);
  free(a);
}

static flagcxResult_t hggcVmmRetryPending(int device) {
  flagcxResult_t result = flagcxSuccess;
  pthread_mutex_lock(&hggcVmmAllocationMutex);
  struct hggcVmmAllocation **link = &hggcVmmAllocations;
  while (*link != NULL) {
    struct hggcVmmAllocation *a = *link;
    if (!a->pendingCleanup || a->device != device) {
      link = &a->next;
      continue;
    }
    result = hggcVmmReleaseAllocation(a);
    if (result != flagcxSuccess)
      break;
    hggcVmmDeleteAllocation(link);
  }
  pthread_mutex_unlock(&hggcVmmAllocationMutex);
  return result;
}

static flagcxResult_t hggcVmmAllocate(void **ptr, size_t size,
                                      size_t granularity, size_t segmentBytes,
                                      const HGmemAllocationProp *prop,
                                      int device) {
  struct hggcVmmAllocation *a;
  flagcxResult_t result;
  if (ptr == NULL || size == 0 || granularity == 0 || segmentBytes == 0 ||
      size % granularity != 0 || segmentBytes % granularity != 0)
    return flagcxInvalidArgument;
  *ptr = NULL;
  size_t count = 1 + (size - 1) / segmentBytes;
  if (count > INT_MAX || count > SIZE_MAX / sizeof(struct hggcVmmSegment))
    return flagcxInvalidArgument;
  a = (struct hggcVmmAllocation *)calloc(1, sizeof(*a));
  if (a == NULL)
    return flagcxSystemError;
  a->segments = (struct hggcVmmSegment *)calloc(count, sizeof(*a->segments));
  if (a->segments == NULL) {
    free(a);
    return flagcxSystemError;
  }
  a->size = size;
  a->count = count;
  a->device = device;
  result = hggcVmmReserveAddress(size, granularity, &a->base);
  if (result != flagcxSuccess) {
    free(a->segments);
    free(a);
    return result;
  }
  a->vaOwned = 1;
  pthread_mutex_lock(&hggcVmmAllocationMutex);
  a->next = hggcVmmAllocations;
  hggcVmmAllocations = a;
  for (size_t i = 0, offset = 0; i < count; i++) {
    struct hggcVmmSegment *s = &a->segments[i];
    s->offset = offset;
    s->size = size - offset < segmentBytes ? size - offset : segmentBytes;
    offset += s->size;
    result =
        hggcDriverResultToFlagcx(hgMemCreate(&s->handle, s->size, prop, 0));
    if (result != flagcxSuccess)
      break;
    s->handleOwned = 1;
    result = hggcDriverResultToFlagcx(
        hgMemMap((HGdeviceptr)(a->base + s->offset), s->size, 0, s->handle, 0));
    if (result != flagcxSuccess)
      break;
    s->mapped = 1;
    result = hggcVmmSetAccess(a->base + s->offset, s->size);
    if (result != flagcxSuccess)
      break;
  }
  // Keep the existing single-segment reference policy. Its mapping holds a
  // reference; descriptions retain their own. Multi-segment bundles borrow.
  if (result == flagcxSuccess && count == 1) {
    result = hggcDriverResultToFlagcx(hgMemRelease(a->segments[0].handle));
    if (result == flagcxSuccess)
      a->segments[0].handleOwned = 0;
  }
  if (result == flagcxSuccess) {
    *ptr = (void *)a->base;
  } else {
    a->pendingCleanup = 1;
    if (hggcVmmReleaseAllocation(a) == flagcxSuccess)
      hggcVmmDeleteAllocation(&hggcVmmAllocations);
  }
  pthread_mutex_unlock(&hggcVmmAllocationMutex);
  return result;
}

static flagcxResult_t hggcVmmFreeAllocation(void *ptr, int *found) {
  flagcxResult_t result = flagcxSuccess;
  *found = 0;
  pthread_mutex_lock(&hggcVmmAllocationMutex);
  struct hggcVmmAllocation **link = &hggcVmmAllocations;
  while (*link != NULL && (*link)->base != (uintptr_t)ptr)
    link = &(*link)->next;
  if (*link != NULL) {
    *found = 1;
    // Do not mark user frees pending: the caller owns the token and retries.
    result = hggcVmmReleaseAllocation(*link);
    if (result == flagcxSuccess)
      hggcVmmDeleteAllocation(link);
  }
  pthread_mutex_unlock(&hggcVmmAllocationMutex);
  return result;
}

static struct hggcVmmAllocation *hggcVmmFindRange(const void *ptr,
                                                  size_t size) {
  uintptr_t address = (uintptr_t)ptr;
  for (struct hggcVmmAllocation *a = hggcVmmAllocations; a != NULL; a = a->next)
    if (address >= a->base && address - a->base <= a->size &&
        size <= a->size - (address - a->base))
      return a;
  return NULL;
}

static flagcxResult_t hggcVmmBundleStorage(struct hggcVmmBundle *bundle,
                                           size_t count) {
  if (count == 0 || count > INT_MAX ||
      count > SIZE_MAX / sizeof(*bundle->segments) ||
      count > SIZE_MAX / sizeof(*bundle->handles))
    return flagcxInvalidArgument;
  bundle->segments =
      (struct flagcxSymSegmentMeta *)calloc(count, sizeof(*bundle->segments));
  bundle->handles =
      (HGmemGenericAllocationHandle *)calloc(count, sizeof(*bundle->handles));
  bundle->handleOwned =
      (unsigned char *)calloc(count, sizeof(*bundle->handleOwned));
  if (bundle->segments == NULL || bundle->handles == NULL ||
      bundle->handleOwned == NULL)
    return flagcxSystemError;
  return flagcxSuccess;
}

static flagcxResult_t hggcVmmFreeBundle(void *opaque) {
  struct hggcVmmBundle *bundle = (struct hggcVmmBundle *)opaque;
  if (bundle == NULL)
    return flagcxSuccess;
  for (size_t i = 0; i < bundle->count; i++) {
    if (bundle->handleOwned[i]) {
      HGDRVCHECK(hgMemRelease(bundle->handles[i]));
      bundle->handleOwned[i] = 0;
    }
  }
  free(bundle->segments);
  free(bundle->handles);
  free(bundle->handleOwned);
  free(bundle);
  return flagcxSuccess;
}

static flagcxResult_t hggcVmmDescribeLocked(void *ptr, size_t size,
                                            struct hggcVmmBundle *bundle,
                                            size_t *userOffset, int *found) {
  struct hggcVmmAllocation *a = hggcVmmFindRange(ptr, size);
  *found = a != NULL;
  if (a == NULL)
    return flagcxSuccess;
  if (a->pendingCleanup || !a->vaOwned)
    return flagcxInvalidUsage;
  size_t begin = (uintptr_t)ptr - a->base;
  size_t end = begin + size;
  size_t count = 0, firstOffset = 0, coveredTo = begin;
  for (size_t i = 0; i < a->count; i++) {
    struct hggcVmmSegment *s = &a->segments[i];
    if (s->offset > a->size || s->size > a->size - s->offset)
      return flagcxInvalidUsage;
    if (s->offset >= end || s->offset + s->size <= begin)
      continue;
    if (s->offset > coveredTo || !s->mapped ||
        (a->count > 1 && !s->handleOwned))
      return flagcxInvalidUsage;
    coveredTo = s->offset + s->size;
    count++;
  }
  if (count == 0 || coveredTo < end)
    return flagcxInvalidUsage;
  FLAGCXCHECK(hggcVmmBundleStorage(bundle, count));
  for (size_t i = 0; i < a->count; i++) {
    struct hggcVmmSegment *s = &a->segments[i];
    if (s->offset >= end || s->offset + s->size <= begin)
      continue;
    size_t index = bundle->count;
    if (index == 0)
      firstOffset = s->offset;
    bundle->segments[index].offset = s->offset - firstOffset;
    bundle->segments[index].size = s->size;
    if (a->count == 1) {
      HGDRVCHECK(hgMemRetainAllocationHandle(&bundle->handles[index], ptr));
      bundle->handleOwned[index] = 1;
    } else {
      bundle->handles[index] = s->handle;
    }
    bundle->count++;
  }
  *userOffset = begin - firstOffset;
  return flagcxSuccess;
}

static flagcxResult_t hggcVmmDescribe(void *ptr, size_t size,
                                      struct hggcVmmBundle *bundle,
                                      size_t *userOffset, int *found) {
  pthread_mutex_lock(&hggcVmmAllocationMutex);
  flagcxResult_t result =
      hggcVmmDescribeLocked(ptr, size, bundle, userOffset, found);
  pthread_mutex_unlock(&hggcVmmAllocationMutex);
  return result;
}

static flagcxResult_t hggcVmmSegmentCount(const void *ptr, size_t size,
                                          int *count) {
  flagcxResult_t result = flagcxNotSupported;
  if (ptr == NULL || size == 0 || count == NULL ||
      size > UINTPTR_MAX - (uintptr_t)ptr)
    return flagcxInvalidArgument;
  *count = 0;
  pthread_mutex_lock(&hggcVmmAllocationMutex);
  struct hggcVmmAllocation *a = hggcVmmFindRange(ptr, size);
  if (a != NULL) {
    result =
        a->pendingCleanup || !a->vaOwned ? flagcxInvalidUsage : flagcxSuccess;
    size_t begin = (uintptr_t)ptr - a->base;
    for (size_t i = 0; result == flagcxSuccess && i < a->count; i++) {
      struct hggcVmmSegment *s = &a->segments[i];
      if (s->offset < begin + size && s->offset + s->size > begin) {
        if (!s->mapped)
          result = flagcxInvalidUsage;
        else
          ++*count;
      }
    }
  }
  pthread_mutex_unlock(&hggcVmmAllocationMutex);
  return result;
}

static flagcxResult_t hggcVmmExportSegment(void *opaque, int index, int *fd) {
  struct hggcVmmBundle *bundle = (struct hggcVmmBundle *)opaque;
  if (bundle == NULL || fd == NULL || index < 0 ||
      (size_t)index >= bundle->count)
    return flagcxInvalidArgument;
  *fd = -1;
  HGDRVCHECK(hgMemExportToShareableHandle(
      fd, bundle->handles[index], HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
  return flagcxSuccess;
}

static struct hggcVmmFlat *hggcVmmFindFlat(void *ptr) {
  for (struct hggcVmmFlat *f = hggcVmmFlats; f != NULL; f = f->next)
    if (f->base == (uintptr_t)ptr)
      return f;
  return NULL;
}

static flagcxResult_t
hggcVmmFlatBegin(void *requestedBase, int reserveVa, int nPeers, int self,
                 void *opaque, const struct flagcxSymRankLayout *layouts,
                 const struct flagcxSymSegmentMeta *segments, int totalSegments,
                 size_t stride, size_t alignment, void **flatBase) {
  struct hggcVmmBundle *bundle = (struct hggcVmmBundle *)opaque;
  struct hggcVmmFlat *f = NULL;
  flagcxResult_t result = flagcxInvalidArgument;
  if (flatBase == NULL)
    return flagcxInvalidArgument;
  *flatBase = NULL;
  if (bundle == NULL || layouts == NULL || segments == NULL || nPeers <= 0 ||
      self < 0 || self >= nPeers || totalSegments <= 0 || stride == 0 ||
      alignment == 0 || (alignment & (alignment - 1)) != 0 ||
      stride % alignment != 0 || stride > SIZE_MAX / (size_t)nPeers ||
      (size_t)totalSegments > SIZE_MAX / sizeof(*f->segments) ||
      bundle->count > INT_MAX ||
      layouts[self].segmentCount != (int)bundle->count ||
      (!reserveVa && requestedBase == NULL))
    return flagcxInvalidArgument;
  f = (struct hggcVmmFlat *)calloc(1, sizeof(*f));
  if (f == NULL)
    return flagcxSystemError;
  f->segments = (struct hggcVmmSegment *)calloc((size_t)totalSegments,
                                                sizeof(*f->segments));
  if (f->segments == NULL) {
    free(f);
    return flagcxSystemError;
  }
  f->size = stride * (size_t)nPeers;
  f->count = totalSegments;
  f->self = self;
  f->bundle = bundle;
  // Match the common rank ordering and reject overlapping descriptors. Slot
  // alignment constrains rank bases, not each physical segment's granularity.
  int nextIndex = 0;
  for (int rank = 0; rank < nPeers; rank++) {
    const struct flagcxSymRankLayout *layout = &layouts[rank];
    if (layout->firstSegment != nextIndex || layout->segmentCount <= 0 ||
        layout->segmentCount > totalSegments - nextIndex)
      goto fail;
    size_t end = 0;
    for (int i = 0; i < layout->segmentCount; i++) {
      int index = nextIndex + i;
      const struct flagcxSymSegmentMeta *s = &segments[index];
      struct hggcVmmSegment *entry = &f->segments[index];
      if (s->size == 0 || s->offset < end || s->offset > stride ||
          s->size > stride - s->offset)
        goto fail;
      end = s->offset + s->size;
      entry->offset = (size_t)rank * stride + s->offset;
      entry->size = s->size;
      entry->ownerRank = rank;
      entry->ownerLocalIndex = i;
    }
    nextIndex += layout->segmentCount;
  }
  if (nextIndex != totalSegments)
    goto fail;
  f->base = (uintptr_t)requestedBase;
  if (reserveVa) {
    result = hggcVmmReserveAddress(f->size, alignment, &f->base);
    if (result != flagcxSuccess)
      goto fail;
    f->vaOwned = 1;
  }
  if (f->base == 0 || f->base % alignment != 0 ||
      f->base > UINTPTR_MAX - f->size) {
    result = flagcxInvalidArgument;
    goto fail;
  }
  pthread_mutex_lock(&hggcVmmFlatMutex);
  if (hggcVmmFindFlat((void *)f->base) != NULL) {
    pthread_mutex_unlock(&hggcVmmFlatMutex);
    result = flagcxInvalidUsage;
    goto fail;
  }
  f->next = hggcVmmFlats;
  hggcVmmFlats = f;
  pthread_mutex_unlock(&hggcVmmFlatMutex);
  // Publish before import/map so common rollback always has a cleanup token.
  *flatBase = (void *)f->base;
  return flagcxSuccess;
fail:
  if (f->vaOwned)
    hggcVmmFreeAddress(f->base, f->size);
  free(f->segments);
  free(f);
  return result;
}

static flagcxResult_t hggcVmmFlatMapOneLocked(void *flatBase, int index,
                                              int fd) {
  struct hggcVmmFlat *f = hggcVmmFindFlat(flatBase);
  if (f == NULL || index < 0 || index >= f->count)
    return flagcxInvalidArgument;
  struct hggcVmmSegment *s = &f->segments[index];
  if (s->mapped || s->handleOwned)
    return flagcxInvalidUsage;
  if (s->ownerRank == f->self) {
    if (fd != -1)
      return flagcxInvalidArgument;
    s->handle = f->bundle->handles[s->ownerLocalIndex];
  } else {
    if (fd < 0)
      return flagcxInvalidArgument;
    HGDRVCHECK(hgMemImportFromShareableHandle(
        &s->handle, (void *)(uintptr_t)fd,
        HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    s->handleOwned = 1;
  }
  HGDRVCHECK(
      hgMemMap((HGdeviceptr)(f->base + s->offset), s->size, 0, s->handle, 0));
  s->mapped = 1;
  return hggcVmmSetAccess(f->base + s->offset, s->size);
}

static flagcxResult_t hggcVmmFlatMapOne(void *flatBase, int index, int fd) {
  if (flatBase == NULL)
    return flagcxInvalidArgument;
  pthread_mutex_lock(&hggcVmmFlatMutex);
  flagcxResult_t result = hggcVmmFlatMapOneLocked(flatBase, index, fd);
  pthread_mutex_unlock(&hggcVmmFlatMutex);
  return result;
}

static flagcxResult_t
hggcVmmFlatMap(const int *fds, int nPeers, int self, void *bundle,
               const struct flagcxSymRankLayout *layouts,
               const struct flagcxSymSegmentMeta *segments, int totalSegments,
               size_t stride, size_t alignment, void **flatBase) {
  if (fds == NULL)
    return flagcxInvalidArgument;
  FLAGCXCHECK(hggcVmmFlatBegin(NULL, 1, nPeers, self, bundle, layouts, segments,
                               totalSegments, stride, alignment, flatBase));
  for (int rank = 0; rank < nPeers; rank++)
    for (int i = 0; i < layouts[rank].segmentCount; i++) {
      int index = layouts[rank].firstSegment + i;
      FLAGCXCHECK(
          hggcVmmFlatMapOne(*flatBase, index, rank == self ? -1 : fds[index]));
    }
  return flagcxSuccess;
}

static flagcxResult_t hggcVmmFlatUnmapLocked(struct hggcVmmFlat *f) {
  for (int i = 0; i < f->count; i++) {
    struct hggcVmmSegment *s = &f->segments[i];
    if (s->mapped) {
      HGDRVCHECK(hgMemUnmap((HGdeviceptr)(f->base + s->offset), s->size));
      s->mapped = 0;
    }
  }
  for (int i = 0; i < f->count; i++) {
    struct hggcVmmSegment *s = &f->segments[i];
    if (s->handleOwned) {
      HGDRVCHECK(hgMemRelease(s->handle));
      s->handleOwned = 0;
    }
  }
  return flagcxSuccess;
}

static flagcxResult_t hggcVmmFlatUnmap(void *flatBase) {
  if (flatBase == NULL)
    return flagcxSuccess;
  pthread_mutex_lock(&hggcVmmFlatMutex);
  struct hggcVmmFlat *f = hggcVmmFindFlat(flatBase);
  flagcxResult_t result =
      f == NULL ? flagcxInvalidUsage : hggcVmmFlatUnmapLocked(f);
  pthread_mutex_unlock(&hggcVmmFlatMutex);
  return result;
}

static flagcxResult_t hggcVmmFlatFreeVaLocked(void *flatBase) {
  struct hggcVmmFlat **link = &hggcVmmFlats;
  while (*link != NULL && (*link)->base != (uintptr_t)flatBase)
    link = &(*link)->next;
  if (*link == NULL)
    return flagcxInvalidUsage;
  struct hggcVmmFlat *f = *link;
  for (int i = 0; i < f->count; i++)
    if (f->segments[i].mapped || f->segments[i].handleOwned)
      return flagcxInvalidUsage;
  if (f->vaOwned)
    FLAGCXCHECK(hggcVmmFreeAddress(f->base, f->size));
  *link = f->next;
  free(f->segments);
  free(f);
  return flagcxSuccess;
}

static flagcxResult_t hggcVmmFlatFreeVa(void *flatBase) {
  if (flatBase == NULL)
    return flagcxSuccess;
  pthread_mutex_lock(&hggcVmmFlatMutex);
  flagcxResult_t result = hggcVmmFlatFreeVaLocked(flatBase);
  pthread_mutex_unlock(&hggcVmmFlatMutex);
  return result;
}

#endif // FLAGCX_HGGC_VMM_H_
