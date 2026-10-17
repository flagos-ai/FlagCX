#include "../../../flagcx/adaptor/device/segmented_vmm_allocation.h"
#include "../../../flagcx/adaptor/device/segmented_vmm_map.h"
#include <gtest/gtest.h>

namespace {
struct FakeVmmOps {
  typedef int Handle;
  typedef int Prop;
  static int maps;
  static int unmaps;
  static int releases;
  static int frees;
  static bool failMap;
  static bool failUnmap;

  static flagcxResult_t deviceOrdinal(int *device) {
    *device = 0;
    return flagcxSuccess;
  }
  static flagcxResult_t createHandle(size_t, const Prop &, Handle *handle) {
    *handle = 100 + maps;
    return flagcxSuccess;
  }

  static flagcxResult_t exportHandle(Handle, int *fd) {
    *fd = 9;
    return flagcxSuccess;
  }
  static flagcxResult_t releaseHandle(Handle) {
    releases++;
    return flagcxSuccess;
  }
  static flagcxResult_t reserveAddress(size_t, size_t, uintptr_t *base) {
    *base = 0x100000;
    return flagcxSuccess;
  }
  static flagcxResult_t freeAddress(uintptr_t, size_t) {
    frees++;
    return flagcxSuccess;
  }
  static flagcxResult_t importHandle(int, Handle *handle) {
    *handle = 17;
    return flagcxSuccess;
  }
  static flagcxResult_t mapSegment(uintptr_t, size_t, Handle) {
    maps++;
    if (failMap && maps == 2)
      return flagcxUnhandledDeviceError;
    return flagcxSuccess;
  }
  static flagcxResult_t setAccess(uintptr_t, size_t) { return flagcxSuccess; }
  static flagcxResult_t unmapSegment(uintptr_t, size_t) {
    unmaps++;
    if (failUnmap && unmaps == 1)
      return flagcxUnhandledDeviceError;
    return flagcxSuccess;
  }
};
int FakeVmmOps::maps = 0;
int FakeVmmOps::unmaps = 0;
int FakeVmmOps::releases = 0;
int FakeVmmOps::frees = 0;
bool FakeVmmOps::failMap = false;
bool FakeVmmOps::failUnmap = false;

TEST(SegmentedVmmMap, PartialMapKeepsImportedHandleUntilUnmapSucceeds) {
  typedef flagcxSegmentedVmmMap<FakeVmmOps> Mapper;
  Mapper::Bundle bundle;
  bundle.handles.push_back(7);
  bundle.handleOwned.push_back(0);
  const flagcxSymRankLayout layouts[2] = {{4096, 0, 4096, 0, 1},
                                          {4096, 0, 4096, 1, 1}};
  const flagcxSymSegmentMeta segments[2] = {{0, 4096}, {0, 4096}};
  FakeVmmOps::maps = FakeVmmOps::unmaps = FakeVmmOps::releases =
      FakeVmmOps::frees = 0;
  FakeVmmOps::failMap = true;
  FakeVmmOps::failUnmap = true;
  void *base = NULL;
  ASSERT_EQ(flagcxSuccess, Mapper::begin(2, 0, &bundle, layouts, segments, 2,
                                         4096, 4096, &base));
  ASSERT_NE(static_cast<void *>(NULL), base);
  EXPECT_EQ(flagcxSuccess, Mapper::mapOne(base, 0, -1));
  EXPECT_EQ(flagcxUnhandledDeviceError, Mapper::mapOne(base, 1, 9));
  EXPECT_EQ(flagcxInvalidUsage, Mapper::freeVa(base));
  EXPECT_EQ(flagcxUnhandledDeviceError, Mapper::unmap(base));
  EXPECT_EQ(0, FakeVmmOps::releases);
  EXPECT_EQ(flagcxSuccess, Mapper::unmap(base));
  EXPECT_EQ(1, FakeVmmOps::releases);
  EXPECT_EQ(flagcxSuccess, Mapper::freeVa(base));
  EXPECT_EQ(1, FakeVmmOps::frees);
  FakeVmmOps::failMap = FakeVmmOps::failUnmap = false;
}

TEST(SegmentedVmmMap, MappingInsideSharedArenaDoesNotFreeArenaVa) {
  typedef flagcxSegmentedVmmMap<FakeVmmOps> Mapper;
  Mapper::Bundle bundle;
  bundle.handles.push_back(7);
  bundle.handleOwned.push_back(0);
  const flagcxSymRankLayout layout = {4096, 0, 4096, 0, 1};
  const flagcxSymSegmentMeta segment = {0, 4096};
  FakeVmmOps::maps = FakeVmmOps::unmaps = FakeVmmOps::releases =
      FakeVmmOps::frees = 0;
  FakeVmmOps::failMap = FakeVmmOps::failUnmap = false;
  void *base = NULL;
  void *reserved = reinterpret_cast<void *>(static_cast<uintptr_t>(0x200000));
  ASSERT_EQ(flagcxSuccess, Mapper::beginAt(reserved, 1, 0, &bundle, &layout,
                                           &segment, 1, 4096, 4096, &base));
  ASSERT_EQ(reserved, base);
  EXPECT_EQ(flagcxSuccess, Mapper::mapOne(base, 0, -1));
  EXPECT_EQ(flagcxSuccess, Mapper::unmap(base));
  EXPECT_EQ(flagcxSuccess, Mapper::freeVa(base));
  EXPECT_EQ(0, FakeVmmOps::frees);
}

TEST(SegmentedVmmAllocation, PartialMapReleasesEveryOwnedHandle) {
  typedef flagcxSegmentedVmmAllocation<FakeVmmOps> Allocation;
  FakeVmmOps::maps = FakeVmmOps::unmaps = FakeVmmOps::releases =
      FakeVmmOps::frees = 0;
  FakeVmmOps::failMap = true;
  FakeVmmOps::failUnmap = false;
  void *ptr = reinterpret_cast<void *>(1);
  EXPECT_EQ(flagcxUnhandledDeviceError,
            Allocation::allocate(&ptr, 8192, 4096, 4096, 0));
  EXPECT_EQ(static_cast<void *>(NULL), ptr);
  EXPECT_EQ(1, FakeVmmOps::unmaps);
  EXPECT_EQ(2, FakeVmmOps::releases);
  EXPECT_EQ(1, FakeVmmOps::frees);
  FakeVmmOps::failMap = false;

  ASSERT_EQ(flagcxSuccess, Allocation::allocate(&ptr, 8192, 4096, 4096, 0));
  int count = 0;
  EXPECT_EQ(flagcxSuccess, Allocation::segmentCount(ptr, 8192, &count));
  EXPECT_EQ(2, count);
  bool found = false;
  EXPECT_EQ(flagcxSuccess, Allocation::freeAllocation(ptr, &found));
  EXPECT_TRUE(found);
}
} // namespace
