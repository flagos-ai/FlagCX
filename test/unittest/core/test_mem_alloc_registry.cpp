/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "mem_alloc_registry.h"

#include <gtest/gtest.h>
#include <thread>
#include <vector>

namespace {

static_assert(flagcxMemAllocBackendGDR == 0,
              "GDR must remain the first allocation backend");
static_assert(flagcxMemAllocBackendCCL == 1,
              "CCL allocation backend order changed");
static_assert(flagcxMemAllocBackendSHMEM == 2,
              "SHMEM allocation backend order changed");

void *addr(uintptr_t value) { return reinterpret_cast<void *>(value); }

flagcxMemAllocationInfo allocation(uintptr_t base, size_t size,
                                   flagcxMemAllocator_t allocator,
                                   flagcxMemAllocBackend backend) {
  return {addr(base), size, allocator, backend};
}

TEST(MemAllocRegistry, PreservesAllocatorAndBackend) {
  flagcxMemAllocRegistry registry;
  const flagcxMemAllocationInfo entries[] = {
      allocation(0x10000, 0x1000, flagcxMemCCL, flagcxMemAllocBackendGDR),
      allocation(0x20000, 0x1000, flagcxMemCCL, flagcxMemAllocBackendCCL),
      allocation(0x30000, 0x1000, flagcxMemSHMEM, flagcxMemAllocBackendSHMEM),
  };
  for (const auto &entry : entries)
    ASSERT_EQ(registry.insert(entry), flagcxSuccess);

  for (const auto &entry : entries) {
    flagcxMemAllocationInfo found;
    ASSERT_EQ(registry.findExact(entry.base, &found), flagcxSuccess);
    EXPECT_EQ(found.base, entry.base);
    EXPECT_EQ(found.size, entry.size);
    EXPECT_EQ(found.allocator, entry.allocator);
    EXPECT_EQ(found.backend, entry.backend);
  }
}

TEST(MemAllocRegistry, AcceptsContainedSubrange) {
  flagcxMemAllocRegistry registry;
  ASSERT_EQ(registry.insert(allocation(0x10000, 0x1000, flagcxMemSHMEM,
                                       flagcxMemAllocBackendSHMEM)),
            flagcxSuccess);
  flagcxMemAllocationInfo found;
  EXPECT_EQ(registry.findRange(addr(0x10100), 0x200, &found), flagcxSuccess);
  EXPECT_EQ(found.base, addr(0x10000));
}

TEST(MemAllocRegistry, RejectsRangeOutsideAllocation) {
  flagcxMemAllocRegistry registry;
  ASSERT_EQ(registry.insert(allocation(0x10000, 0x1000, flagcxMemSHMEM,
                                       flagcxMemAllocBackendSHMEM)),
            flagcxSuccess);
  flagcxMemAllocationInfo found;
  EXPECT_EQ(registry.findRange(addr(0x10f00), 0x200, &found),
            flagcxInvalidUsage);
  EXPECT_EQ(registry.findRange(addr(0x11000), 1, &found), flagcxInvalidUsage);
}

TEST(MemAllocRegistry, RejectsInvalidAndOverlappingAllocations) {
  flagcxMemAllocRegistry registry;
  EXPECT_EQ(registry.insert(
                allocation(0, 0x1000, flagcxMemCCL, flagcxMemAllocBackendGDR)),
            flagcxInvalidArgument);
  EXPECT_EQ(registry.insert(
                allocation(0x10000, 0, flagcxMemCCL, flagcxMemAllocBackendGDR)),
            flagcxInvalidArgument);
  ASSERT_EQ(registry.insert(allocation(0x10000, 0x1000, flagcxMemCCL,
                                       flagcxMemAllocBackendGDR)),
            flagcxSuccess);
  EXPECT_EQ(registry.insert(allocation(0x10000, 0x1000, flagcxMemCCL,
                                       flagcxMemAllocBackendGDR)),
            flagcxInvalidUsage);
  EXPECT_EQ(registry.insert(allocation(0x10800, 0x1000, flagcxMemCCL,
                                       flagcxMemAllocBackendGDR)),
            flagcxInvalidUsage);
}

TEST(MemAllocRegistry, KeepsSamePageAllocationsIndependent) {
  flagcxMemAllocRegistry registry;
  ASSERT_EQ(registry.insert(allocation(0x10100, 0x100, flagcxMemCCL,
                                       flagcxMemAllocBackendGDR)),
            flagcxSuccess);
  ASSERT_EQ(registry.insert(allocation(0x10300, 0x100, flagcxMemCCL,
                                       flagcxMemAllocBackendCCL)),
            flagcxSuccess);
  flagcxMemAllocationInfo first, second;
  ASSERT_EQ(registry.findExact(addr(0x10100), &first), flagcxSuccess);
  ASSERT_EQ(registry.findExact(addr(0x10300), &second), flagcxSuccess);
  EXPECT_EQ(first.backend, flagcxMemAllocBackendGDR);
  EXPECT_EQ(second.backend, flagcxMemAllocBackendCCL);
}

TEST(MemAllocRegistry, EraseRequiresExactBase) {
  flagcxMemAllocRegistry registry;
  ASSERT_EQ(registry.insert(allocation(0x10000, 0x1000, flagcxMemCCL,
                                       flagcxMemAllocBackendCCL)),
            flagcxSuccess);
  EXPECT_EQ(registry.erase(addr(0x10100)), flagcxInvalidUsage);
  EXPECT_EQ(registry.erase(addr(0x10000)), flagcxSuccess);
  flagcxMemAllocationInfo found;
  EXPECT_EQ(registry.findExact(addr(0x10000), &found), flagcxInvalidUsage);
  EXPECT_EQ(registry.erase(addr(0x10000)), flagcxInvalidUsage);
}

TEST(MemAllocRegistry, SupportsConcurrentIndependentAllocations) {
  flagcxMemAllocRegistry registry;
  constexpr int kThreads = 8;
  constexpr int kEntries = 128;
  std::vector<std::thread> threads;
  for (int thread = 0; thread < kThreads; ++thread) {
    threads.emplace_back([&, thread] {
      uintptr_t threadBase = 0x100000 + (uintptr_t)thread * 0x100000;
      for (int i = 0; i < kEntries; ++i) {
        void *base = addr(threadBase + (uintptr_t)i * 0x1000);
        EXPECT_EQ(registry.insert(
                      {base, 0x100, flagcxMemCCL, flagcxMemAllocBackendGDR}),
                  flagcxSuccess);
        flagcxMemAllocationInfo found;
        EXPECT_EQ(registry.findRange((char *)base + 8, 16, &found),
                  flagcxSuccess);
      }
    });
  }
  for (auto &thread : threads)
    thread.join();

  for (int thread = 0; thread < kThreads; ++thread) {
    uintptr_t threadBase = 0x100000 + (uintptr_t)thread * 0x100000;
    for (int i = 0; i < kEntries; ++i)
      EXPECT_EQ(registry.erase(addr(threadBase + (uintptr_t)i * 0x1000)),
                flagcxSuccess);
  }
}

} // namespace
