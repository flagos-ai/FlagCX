"""Compile PPU's private bookkeeping as C11/C++11 and inject failures.

The fake driver tracks VA, mappings, and handle references independently so a
lost ownership record, a double release, or an early release fails the test.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
CC = os.environ.get("CC") or shutil.which("cc")
CXX = os.environ.get("CXX") or shutil.which("c++")

SOURCE = r"""
#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "flagcx_device_adaptor.h"

typedef int HGresult;
typedef int HGdevice;
#define hggcSuccess 0
typedef int hggcIpcMemHandle_t;
typedef uintptr_t HGdeviceptr;
typedef int HGmemGenericAllocationHandle;
typedef struct { int type, id; } HGmemLocation;
typedef struct { HGmemLocation location; int flags; } HGmemAccessDesc;
typedef struct { int type; HGmemLocation location; int requestedHandleTypes; } HGmemAllocationProp;
enum { HGGC_SUCCESS, HGGC_ERROR_UNKNOWN,
       HG_MEM_LOCATION_TYPE_DEVICE, HG_MEM_ACCESS_FLAGS_PROT_READWRITE,
       HG_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, HG_MEM_ALLOCATION_TYPE_PINNED,
       HG_MEM_ALLOC_GRANULARITY_RECOMMENDED };
enum { HEAP, RESERVE, CREATE, MAP, ACCESS, UNMAP, FREE_VA, RELEASE,
       IMPORT, EXPORT, RETAIN, OP_COUNT };
static int calls[OP_COUNT], failures[OP_COUNT];
static int refs[512], nextHandle;
static uintptr_t nextVa;
static struct { uintptr_t base; size_t size; } reservations[256];
static struct { uintptr_t base; size_t size; int handle; } mappings[256];
static int currentDevice;

static int fail(int op) { return ++calls[op] == failures[op]; }
static void *testCalloc(size_t n, size_t size) {
  return fail(HEAP) ? NULL : calloc(n, size);
}
static flagcxResult_t hggcDriverResultToFlagcx(HGresult result) {
  return result == HGGC_SUCCESS ? flagcxSuccess : flagcxUnhandledDeviceError;
}
#define HGDRVCHECK(call) do { HGresult r_ = (call); if (r_ != HGGC_SUCCESS) return flagcxUnhandledDeviceError; } while (0)
#define HGGCCHECK(call) HGDRVCHECK(call)
#define FLAGCXCHECK(call) do { flagcxResult_t r_ = (call); if (r_ != flagcxSuccess) return r_; } while (0)
static int hggcGetDevice(int *device) { *device = currentDevice; return 0; }
static HGresult hgMemAddressReserve(HGdeviceptr *p, size_t size, size_t alignment,
                                    HGdeviceptr requested, unsigned long long flags) {
  (void)alignment; (void)requested; (void)flags;
  if (fail(RESERVE)) return HGGC_ERROR_UNKNOWN;
  // Reuse freed VA, but never overlap a reservation still owned by a caller.
  int occupied;
  do {
    occupied = 0;
    for (int i = 0; i < 256; i++)
      if (reservations[i].base && nextVa < reservations[i].base + reservations[i].size &&
          reservations[i].base < nextVa + size) {
        nextVa += 0x100000; occupied = 1; break;
      }
  } while (occupied);
  for (int i = 0; i < 256; i++) if (!reservations[i].base) {
    *p = nextVa; nextVa += 0x100000;
    reservations[i].base = *p; reservations[i].size = size; return 0;
  }
  abort();
}
static HGresult hgMemAddressFree(HGdeviceptr p, size_t size) {
  if (fail(FREE_VA)) return HGGC_ERROR_UNKNOWN;
  for (int i = 0; i < 256; i++)
    assert(!mappings[i].base || mappings[i].base < p || mappings[i].base >= p + size);
  for (int i = 0; i < 256; i++) if (reservations[i].base == p) {
    assert(reservations[i].size == size); reservations[i].base = 0;
    if (p < nextVa) nextVa = p;
    return 0;
  }
  abort();
}
static HGresult hgMemCreate(HGmemGenericAllocationHandle *handle, size_t size,
                            const HGmemAllocationProp *prop, unsigned long long flags) {
  (void)size; (void)prop; (void)flags;
  if (fail(CREATE)) return HGGC_ERROR_UNKNOWN;
  *handle = ++nextHandle; assert(nextHandle < 512); refs[*handle] = 1; return 0;
}
static HGresult hgMemRelease(HGmemGenericAllocationHandle handle) {
  if (fail(RELEASE)) return HGGC_ERROR_UNKNOWN;
  assert(refs[handle] > 0); --refs[handle]; return 0;
}
static HGresult hgMemMap(HGdeviceptr p, size_t size, size_t offset,
                         HGmemGenericAllocationHandle handle, unsigned long long flags) {
  (void)offset; (void)flags;
  if (fail(MAP)) return HGGC_ERROR_UNKNOWN;
  assert(refs[handle] > 0);
  for (int i = 0; i < 256; i++) if (!mappings[i].base) {
    mappings[i].base = p; mappings[i].size = size; mappings[i].handle = handle;
    ++refs[handle]; return 0;
  }
  abort();
}
static HGresult hgMemUnmap(HGdeviceptr p, size_t size) {
  if (fail(UNMAP)) return HGGC_ERROR_UNKNOWN;
  for (int i = 0; i < 256; i++) if (mappings[i].base == p) {
    assert(mappings[i].size == size && refs[mappings[i].handle] > 0);
    --refs[mappings[i].handle]; mappings[i].base = 0; return 0;
  }
  abort();
}
static HGresult hgMemSetAccess(HGdeviceptr p, size_t size,
                               const HGmemAccessDesc *access, size_t count) {
  (void)p; (void)size; (void)access; (void)count;
  return fail(ACCESS) ? HGGC_ERROR_UNKNOWN : 0;
}
static HGresult hgMemRetainAllocationHandle(HGmemGenericAllocationHandle *handle, void *ptr) {
  if (fail(RETAIN)) return HGGC_ERROR_UNKNOWN;
  for (int i = 0; i < 256; i++)
    if (mappings[i].base && (uintptr_t)ptr >= mappings[i].base &&
        (uintptr_t)ptr - mappings[i].base < mappings[i].size) {
      *handle = mappings[i].handle; ++refs[*handle]; return 0;
    }
  abort();
}
static HGresult hgMemImportFromShareableHandle(HGmemGenericAllocationHandle *handle,
                                               void *fd, int type) {
  (void)fd; (void)type;
  if (fail(IMPORT)) return HGGC_ERROR_UNKNOWN;
  *handle = ++nextHandle; assert(nextHandle < 512); refs[*handle] = 1; return 0;
}
static HGresult hgMemExportToShareableHandle(void *fd, HGmemGenericAllocationHandle handle,
                                            int type, unsigned long long flags) {
  (void)type; (void)flags;
  if (fail(EXPORT)) return HGGC_ERROR_UNKNOWN;
  assert(refs[handle] > 0); *(int *)fd = handle + 100; return 0;
}
static int pointerKind, clearedErrors;
static HGresult hgDeviceGet(HGdevice *device, int ordinal) { *device = ordinal; return 0; }
static HGresult hgMemGetAllocationGranularity(size_t *size, const HGmemAllocationProp *prop, int flags) {
  (void)prop; (void)flags; *size = 4096; return 0;
}
static HGresult hgMemGetAddressRange(HGdeviceptr *base, size_t *size, HGdeviceptr ptr) {
  if (pointerKind == 0) return HGGC_ERROR_UNKNOWN;
  *base = ptr & ~(HGdeviceptr)4095; *size = 4096; return 0;
}
static int hggcIpcGetMemHandle(hggcIpcMemHandle_t *handle, void *ptr) {
  (void)handle; (void)ptr; return pointerKind == 1 ? 0 : HGGC_ERROR_UNKNOWN;
}
static int hggcGetLastError(void) { ++clearedErrors; return 0; }
#define calloc testCalloc
#include "hggc_vmm.h"
#undef calloc

/* PRODUCTION_DESCRIBE */

static void clean(void) {
  assert(hggcVmmAllocations == NULL && hggcVmmFlats == NULL);
  for (int i = 0; i < 512; i++) assert(refs[i] == 0);
  for (int i = 0; i < 256; i++) assert(!reservations[i].base && !mappings[i].base);
}
static void reset(void) {
  clean(); memset(calls, 0, sizeof(calls)); memset(failures, 0, sizeof(failures));
  currentDevice = 0; nextVa = 0x100000; nextHandle = 0;
}
static void releaseAllocation(void *ptr) {
  int found = 0;
  assert(hggcVmmFreeAllocation(ptr, &found) == flagcxSuccess && found);
}
static int vaReserved(void *ptr) {
  for (int i = 0; i < 256; i++)
    if (reservations[i].base == (uintptr_t)ptr) return 1;
  return 0;
}
static void failedFreeKeepsAllocationIdentity(void) {
  HGmemAllocationProp prop = {0};
  for (int failAt = 1; failAt <= 2; failAt++) {
    reset(); void *a = NULL, *b = NULL, *reused = NULL;
    int found = 0, count = 0;
    assert(hggcVmmAllocate(&a, 12288, 4096, 4096, &prop, 0) == flagcxSuccess);
    failures[RELEASE] = calls[RELEASE] + failAt;
    assert(hggcVmmFreeAllocation(a, &found) == flagcxUnhandledDeviceError && found);
    assert(vaReserved(a) && calls[FREE_VA] == 0);
    // User-owned failed frees must not be collected by allocation rollback.
    int releases = calls[RELEASE];
    assert(hggcVmmRetryPending(0) == flagcxSuccess && calls[RELEASE] == releases);
    assert(hggcVmmAllocate(&b, 12288, 4096, 4096, &prop, 0) == flagcxSuccess);
    assert(b != a && vaReserved(a) && vaReserved(b));
    // Retry A while B is live. Successful releases must not be repeated.
    releaseAllocation(a);
    assert(calls[RELEASE] - releases == 4 - failAt);
    assert(!vaReserved(a) && vaReserved(b));
    assert(hggcVmmSegmentCount(b, 12288, &count) == flagcxSuccess && count == 3);
    for (int i = 1; i <= 3; i++) assert(refs[i] == 0);
    for (int i = 4; i <= 6; i++) assert(refs[i] == 2);
    // Reuse is safe only after A's cleanup has completed.
    assert(hggcVmmAllocate(&reused, 12288, 4096, 4096, &prop, 0) == flagcxSuccess);
    assert(reused == a && reused != b);
    releaseAllocation(reused); releaseAllocation(b); clean();
  }
}
static void vaFreeRetryDoesNotReleaseHandlesAgain(void) {
  reset(); HGmemAllocationProp prop = {0}; void *a = NULL, *b = NULL;
  int found = 0, count = 0;
  assert(hggcVmmAllocate(&a, 12288, 4096, 4096, &prop, 0) == flagcxSuccess);
  failures[FREE_VA] = calls[FREE_VA] + 1;
  assert(hggcVmmFreeAllocation(a, &found) == flagcxUnhandledDeviceError && found);
  assert(vaReserved(a));
  for (int i = 1; i <= 3; i++) assert(refs[i] == 0);
  assert(hggcVmmAllocate(&b, 12288, 4096, 4096, &prop, 0) == flagcxSuccess && b != a);
  int releases = calls[RELEASE], unmaps = calls[UNMAP], frees = calls[FREE_VA];
  releaseAllocation(a);
  assert(calls[RELEASE] == releases && calls[UNMAP] == unmaps && calls[FREE_VA] == frees + 1);
  assert(hggcVmmSegmentCount(b, 12288, &count) == flagcxSuccess && count == 3);
  releaseAllocation(b); clean();
}
static void failedRollbackKeepsVaUntilHandleCleanup(void) {
  reset(); HGmemAllocationProp prop = {0}; void *a = (void *)1, *b = NULL, *reused = NULL;
  failures[MAP] = 2; failures[RELEASE] = 2;
  assert(hggcVmmAllocate(&a, 12288, 4096, 4096, &prop, 0) == flagcxUnhandledDeviceError);
  assert(a == NULL && hggcVmmAllocations && hggcVmmAllocations->pendingCleanup);
  void *failedBase = (void *)hggcVmmAllocations->base;
  assert(vaReserved(failedBase) && calls[FREE_VA] == 0 && refs[1] == 0 && refs[2] == 1);
  // A different device's allocation cannot collect this pending cleanup.
  assert(hggcVmmRetryPending(1) == flagcxSuccess && vaReserved(failedBase));
  assert(hggcVmmAllocate(&b, 4096, 4096, 4096, &prop, 1) == flagcxSuccess && b != failedBase);
  int releases = calls[RELEASE];
  assert(hggcVmmRetryPending(0) == flagcxSuccess && calls[RELEASE] == releases + 1);
  assert(!vaReserved(failedBase) && vaReserved(b) && refs[2] == 0 && refs[3] == 1);
  assert(hggcVmmAllocate(&reused, 4096, 4096, 4096, &prop, 0) == flagcxSuccess);
  assert(reused == failedBase);
  releaseAllocation(reused); releaseAllocation(b); clean();
}
static struct hggcVmmBundle *describe(void *ptr, size_t size, size_t expectedOffset,
                                      size_t expectedCount) {
  struct hggcVmmBundle *bundle = (struct hggcVmmBundle *)calloc(1, sizeof(*bundle));
  int found = 0; size_t offset = 0;
  assert(bundle != NULL);
  assert(hggcVmmDescribe(ptr, size, bundle, &offset, &found) == flagcxSuccess);
  assert(found && offset == expectedOffset && bundle->count == expectedCount);
  return bundle;
}
static void allocationFailures(void) {
  const int ops[] = {HEAP, RESERVE, CREATE, MAP, ACCESS};
  HGmemAllocationProp prop = {0};
  for (size_t i = 0; i < sizeof(ops)/sizeof(ops[0]); i++) {
    for (int nth = 1; nth <= (ops[i] == RESERVE ? 1 : 2); nth++) {
      reset(); failures[ops[i]] = nth;
      void *ptr = (void *)1;
      assert(hggcVmmAllocate(&ptr, 12288, 4096, 4096, &prop, 0) != flagcxSuccess);
      assert(ptr == NULL); clean();
    }
  }
  reset(); failures[RELEASE] = 1;
  void *ptr = (void *)1;
  assert(hggcVmmAllocate(&ptr, 4096, 4096, 4096, &prop, 0) != flagcxSuccess);
  assert(ptr == NULL); clean();
}
static void allocationRollbackRetry(void) {
  const int ops[] = {UNMAP, FREE_VA, RELEASE};
  HGmemAllocationProp prop = {0};
  for (size_t i = 0; i < sizeof(ops)/sizeof(ops[0]); i++) {
    reset(); failures[MAP] = 2; failures[ops[i]] = 1;
    void *ptr = (void *)1;
    assert(hggcVmmAllocate(&ptr, 12288, 4096, 4096, &prop, 0) == flagcxUnhandledDeviceError);
    assert(ptr == NULL && hggcVmmAllocations && hggcVmmAllocations->pendingCleanup);
    assert(hggcVmmRetryPending(1) == flagcxSuccess && hggcVmmAllocations);
    assert(hggcVmmRetryPending(0) == flagcxSuccess); clean();
  }
}
static void freeRetryAndDescriptions(void) {
  const int ops[] = {UNMAP, FREE_VA, RELEASE};
  HGmemAllocationProp prop = {0};
  for (int multi = 0; multi <= 1; multi++) {
    for (size_t i = 0; i < sizeof(ops)/sizeof(ops[0]); i++) {
      reset(); void *ptr = NULL; int found = 0;
      assert(hggcVmmAllocate(&ptr, 12288, 4096, multi ? 4096 : 12288, &prop, 0) == flagcxSuccess);
      struct hggcVmmBundle *b = describe((void *)((uintptr_t)ptr + 5120), 6144,
                                         multi ? 1024 : 5120, multi ? 2 : 1);
      assert(b->segments[0].offset == 0);
      if (multi) assert(b->segments[1].offset == 4096);
      int fd = -1, count = -1;
      assert(hggcVmmExportSegment(b, 0, &fd) == flagcxSuccess && fd >= 0);
      assert(hggcVmmSegmentCount((void *)((uintptr_t)ptr + 5120), 6144, &count) == flagcxSuccess);
      assert(count == (multi ? 2 : 1));
      // Borrowed multi-segment references must never be released by bundles.
      if (!multi) {
        failures[RELEASE] = calls[RELEASE] + 1;
        assert(hggcVmmFreeBundle(b) == flagcxUnhandledDeviceError);
      }
      assert(hggcVmmFreeBundle(b) == flagcxSuccess);
      failures[ops[i]] = calls[ops[i]] + 1;
      // Single allocations no longer own a handle after successful create.
      if (!multi && ops[i] == RELEASE) { releaseAllocation(ptr); clean(); continue; }
      assert(hggcVmmFreeAllocation(ptr, &found) == flagcxUnhandledDeviceError && found);
      assert(hggcVmmAllocations != NULL);
      b = (struct hggcVmmBundle *)calloc(1, sizeof(*b));
      size_t offset = 0;
      assert(hggcVmmDescribe(ptr, 12288, b, &offset, &found) ==
             (ops[i] == UNMAP ? flagcxSuccess : flagcxInvalidUsage) && found);
      assert(hggcVmmFreeBundle(b) == flagcxSuccess);
      releaseAllocation(ptr); clean();
    }
  }
  for (int heap = 1; heap <= 3; heap++) {
    reset(); void *ptr = NULL; int found = 0; size_t offset = 0;
    assert(hggcVmmAllocate(&ptr, 4096, 4096, 4096, &prop, 0) == flagcxSuccess);
    struct hggcVmmBundle *b = (struct hggcVmmBundle *)calloc(1, sizeof(*b));
    failures[HEAP] = calls[HEAP] + heap;
    assert(hggcVmmDescribe(ptr, 4096, b, &offset, &found) == flagcxSystemError && found);
    assert(hggcVmmFreeBundle(b) == flagcxSuccess);
    releaseAllocation(ptr); clean();
  }
}
static void flatMappingRetry(void) {
  const struct flagcxSymRankLayout layouts[] = {{4096, 0, 4096, 0, 1}, {4096, 0, 4096, 1, 1}};
  const struct flagcxSymSegmentMeta segments[] = {{0, 4096}, {0, 4096}};
  const int failureOps[] = {IMPORT, MAP, ACCESS, UNMAP, RELEASE, FREE_VA};
  HGmemAllocationProp prop = {0};
  for (size_t i = 0; i < sizeof(failureOps)/sizeof(failureOps[0]); i++) {
    reset(); void *ptr = NULL, *base = NULL;
    assert(hggcVmmAllocate(&ptr, 4096, 4096, 4096, &prop, 0) == flagcxSuccess);
    struct hggcVmmBundle *b = describe(ptr, 4096, 0, 1);
    assert(hggcVmmFlatBegin(NULL, 1, 2, 0, b, layouts, segments, 2, 4096, 4096, &base) == flagcxSuccess);
    assert(hggcVmmFlatMapOne(base, 0, -1) == flagcxSuccess);
    int op = failureOps[i]; failures[op] = calls[op] + 1;
    flagcxResult_t mapped = hggcVmmFlatMapOne(base, 1, 9);
    assert(mapped == ((op == IMPORT || op == MAP || op == ACCESS) ? flagcxUnhandledDeviceError : flagcxSuccess));
    if (op != IMPORT) assert(hggcVmmFlatFreeVa(base) == flagcxInvalidUsage);
    if (op == UNMAP || op == RELEASE)
      assert(hggcVmmFlatUnmap(base) == flagcxUnhandledDeviceError);
    assert(hggcVmmFlatUnmap(base) == flagcxSuccess);
    if (op == FREE_VA) assert(hggcVmmFlatFreeVa(base) == flagcxUnhandledDeviceError);
    assert(hggcVmmFlatFreeVa(base) == flagcxSuccess);
    assert(hggcVmmFreeBundle(b) == flagcxSuccess);
    releaseAllocation(ptr); clean();
  }
  reset(); void *ptr = NULL, *arena = NULL, *base = NULL;
  assert(hggcVmmAllocate(&ptr, 4096, 4096, 4096, &prop, 0) == flagcxSuccess);
  struct hggcVmmBundle *b = describe(ptr, 4096, 0, 1);
  assert(hggcVmmReserveAddress(8192, 4096, (uintptr_t *)&arena) == flagcxSuccess);
  assert(hggcVmmFlatBegin(arena, 0, 2, 0, b, layouts, segments, 2, 4096, 4096, &base) == flagcxSuccess);
  assert(hggcVmmFlatMapOne(base, 0, -1) == flagcxSuccess);
  assert(hggcVmmFlatMapOne(base, 1, 9) == flagcxSuccess);
  assert(hggcVmmFlatUnmap(base) == flagcxSuccess);
  int frees = calls[FREE_VA];
  assert(hggcVmmFlatFreeVa(base) == flagcxSuccess && calls[FREE_VA] == frees);
  assert(hggcVmmFreeAddress((uintptr_t)arena, 8192) == flagcxSuccess);
  assert(hggcVmmFreeBundle(b) == flagcxSuccess); releaseAllocation(ptr); clean();
}
static void segmentAndSlotAlignment(void) {
  reset(); HGmemAllocationProp prop = {0}; void *ptr = NULL, *base = NULL;
  assert(hggcVmmAllocate(&ptr, 8192, 4096, 4096, &prop, 0) == flagcxSuccess);
  struct hggcVmmBundle *b = describe(ptr, 8192, 0, 2);
  const struct flagcxSymRankLayout layout = {12288, 8192, 4096, 0, 2};
  const struct flagcxSymSegmentMeta segments[] = {{0, 4096}, {8192, 4096}};
  int fds[] = {-1, -1};
  assert(hggcVmmFlatMap(fds, 1, 0, b, &layout, segments, 2, 16384, 16384, &base) == flagcxSuccess);
  assert(hggcVmmFlatUnmap(base) == flagcxSuccess);
  assert(hggcVmmFlatFreeVa(base) == flagcxSuccess);
  assert(hggcVmmFreeBundle(b) == flagcxSuccess); releaseAllocation(ptr); clean();
}
static void policyAndValidation(void) {
  reset(); size_t bytes = 0; int enabled = 0;
  assert(hggcVmmMultiSegmentEnabled(&enabled) == flagcxSuccess && enabled);
  assert(setenv("FLAGCX_VMM_MULTI_SEGMENT_ENABLE", "0", 1) == 0);
  assert(hggcVmmMultiSegmentEnabled(&enabled) == flagcxSuccess && !enabled);
  assert(setenv("FLAGCX_VMM_MULTI_SEGMENT_ENABLE", "bad", 1) == 0);
  assert(hggcVmmMultiSegmentEnabled(&enabled) == flagcxInvalidArgument);
  unsetenv("FLAGCX_VMM_MULTI_SEGMENT_ENABLE");
  const char *invalid[] = {"0", "-4096", "4097", "1024", "x", "18446744073709551616"};
  for (size_t i = 0; i < sizeof(invalid)/sizeof(invalid[0]); i++) {
    setenv("FLAGCX_VMM_SEGMENT_BYTES", invalid[i], 1);
    assert(hggcVmmSegmentBytes(4096, &bytes) == flagcxInvalidArgument);
  }
  setenv("FLAGCX_VMM_SEGMENT_BYTES", "8192", 1);
  assert(hggcVmmSegmentBytes(4096, &bytes) == flagcxSuccess && bytes == 8192);
  unsetenv("FLAGCX_VMM_SEGMENT_BYTES");
  int count = -1;
  assert(hggcVmmSegmentCount((void *)UINTPTR_MAX, 2, &count) == flagcxInvalidArgument);
  assert(hggcVmmSegmentCount((void *)0x10000, 4096, &count) == flagcxNotSupported && count == 0);
  clean();
}
static void pointerProvenance(void) {
  reset();
  for (pointerKind = 0; pointerKind <= 2; pointerKind++) {
    void *bundle = (void *)1;
    const struct flagcxSymSegmentMeta *meta = NULL;
    int count = -1; size_t offset = 0, alignment = 0;
    int retains = calls[RETAIN], clears = clearedErrors;
    assert(hggcAdaptorSymPhysDescribe((void *)0x100100, 1024, &bundle, &meta,
                                     &count, &offset, &alignment) == flagcxNotSupported);
    assert((bundle == NULL) == (pointerKind == 1));
    assert(meta == NULL && count == 0 && calls[RETAIN] == retains);
    assert(clearedErrors == clears + (pointerKind == 2));
    assert(hggcVmmFreeBundle(bundle) == flagcxSuccess);
  }
  clean();
}
int main(int argc, char **argv) {
  assert(argc == 2);
  if (!strcmp(argv[1], "allocation_failures")) allocationFailures();
  else if (!strcmp(argv[1], "allocation_rollback")) allocationRollbackRetry();
  else if (!strcmp(argv[1], "free_and_describe")) freeRetryAndDescriptions();
  else if (!strcmp(argv[1], "flat_mapping")) flatMappingRetry();
  else if (!strcmp(argv[1], "alignment")) segmentAndSlotAlignment();
  else if (!strcmp(argv[1], "policy")) policyAndValidation();
  else if (!strcmp(argv[1], "provenance")) pointerProvenance();
  else if (!strcmp(argv[1], "free_identity")) failedFreeKeepsAllocationIdentity();
  else if (!strcmp(argv[1], "va_free_retry")) vaFreeRetryDoesNotReleaseHandlesAgain();
  else if (!strcmp(argv[1], "rollback_identity")) failedRollbackKeepsVaUntilHandleCleanup();
  else abort();
  clean(); return 0;
}
"""


@unittest.skipUnless(CC and CXX, "C and C++ compilers required")
class PpuVmmHostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        root = Path(cls.directory.name)
        source = root / "test.c"
        production = (ROOT / "flagcx/adaptor/device/hggc_adaptor.cc").read_text()
        start = production.index("flagcxResult_t hggcAdaptorSymPhysDescribe(")
        end = production.index("flagcxResult_t hggcAdaptorSymPhysExportSegment(", start)
        describe = production[start:end].replace("const flagcxSymSegmentMeta", "const struct flagcxSymSegmentMeta")
        source.write_text(SOURCE.replace("/* PRODUCTION_DESCRIBE */", describe))
        cls.binaries = []
        for compiler, language, standard in ((CC, "c", "c11"), (CXX, "c++", "c++11")):
            binary = root / ("test-" + language)
            result = subprocess.run([
                compiler, "-x", language, "-std=" + standard, "-pthread",
                "-I" + str(ROOT / "flagcx/include"),
                "-I" + str(ROOT / "flagcx/adaptor/include"),
                "-I" + str(ROOT / "flagcx/adaptor/device"),
                str(source), "-o", str(binary),
            ], capture_output=True, text=True)
            if result.returncode:
                raise AssertionError(result.stderr)
            cls.binaries.append(binary)

    def run_case(self, case):
        for binary in self.binaries:
            with self.subTest(compiler=binary.name):
                env = {k: v for k, v in os.environ.items()
                       if k not in ("FLAGCX_VMM_MULTI_SEGMENT_ENABLE", "FLAGCX_VMM_SEGMENT_BYTES")}
                result = subprocess.run([str(binary), case], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_allocation_failures_release_owned_resources(self):
        self.run_case("allocation_failures")

    def test_failed_allocation_cleanup_retries_on_owning_device(self):
        self.run_case("allocation_rollback")

    def test_failed_free_keeps_identity_until_cleanup_completes(self):
        self.run_case("free_identity")

    def test_va_free_retry_does_not_repeat_handle_release(self):
        self.run_case("va_free_retry")

    def test_failed_rollback_keeps_va_until_owned_handles_are_released(self):
        self.run_case("rollback_identity")

    def test_interior_descriptions_and_free_retries(self):
        self.run_case("free_and_describe")

    def test_import_map_cleanup_retries_and_borrowed_arena(self):
        self.run_case("flat_mapping")

    def test_unknown_vmm_does_not_fall_back_to_legacy_ipc(self):
        self.run_case("provenance")

    def test_segment_sizes_are_independent_of_slot_alignment(self):
        self.run_case("alignment")

    def test_policy_and_invalid_ranges(self):
        self.run_case("policy")


if __name__ == "__main__":
    unittest.main()
