// Unit tests for flagcxRegPool — buffer registration pool.
// Source: flagcx/core/reg_pool.cc + flagcx/core/include/reg_pool.h
// Links against libflagcx. No MPI, no GPU required.

#include <gtest/gtest.h>

#include "reg_pool.h"
#include <cstring>
#include <vector>

// Helper: create a fake comm pointer from an integer
static void *fakeComm(uintptr_t id) { return reinterpret_cast<void *>(id); }

// Helper: create a fake proxyConn pointer
static struct flagcxProxyConnector *fakeProxy(uintptr_t id) {
  return reinterpret_cast<struct flagcxProxyConnector *>(id);
}

class RegPoolTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Use a fresh pool for each test
    pool = new flagcxRegPool();
    pageSize = sysconf(_SC_PAGESIZE);
  }
  void TearDown() override { delete pool; }

  flagcxRegPool *pool;
  uintptr_t pageSize;

  // Allocate a page-aligned buffer of given size
  void *alignedAddr(uintptr_t base) {
    return reinterpret_cast<void *>(base * pageSize);
  }
};

// =============================================================================
// 1. Basic Registration
// =============================================================================

TEST_F(RegPoolTest, RegisterBuffer_NullData_NoOp) {
  EXPECT_EQ(pool->registerBuffer(fakeComm(1), nullptr, 1024), flagcxSuccess);
  EXPECT_EQ(pool->registerBuffer(fakeComm(1), alignedAddr(10), 0),
            flagcxSuccess);
}

TEST_F(RegPoolTest, RegisterBuffer_SingleBuffer) {
  void *data = alignedAddr(10);
  ASSERT_EQ(pool->registerBuffer(fakeComm(1), data, pageSize), flagcxSuccess);

  flagcxRegItem *item = pool->getItem(fakeComm(1), data);
  ASSERT_NE(item, nullptr);
  EXPECT_EQ(item->refCount, 1);
}

TEST_F(RegPoolTest, RegisterBuffer_PageAlignment) {
  // Use an address in the middle of a page
  void *data = reinterpret_cast<void *>(10 * pageSize + 100);
  size_t length = pageSize + 200; // spans 2+ pages

  ASSERT_EQ(pool->registerBuffer(fakeComm(1), data, length), flagcxSuccess);

  flagcxRegItem *item = pool->getItem(fakeComm(1), data);
  ASSERT_NE(item, nullptr);
  EXPECT_EQ(item->beginAddr, 10 * pageSize);
  // endAddr should be page-aligned ceiling
  uintptr_t expectedEnd =
      (10 * pageSize + 100 + length + pageSize - 1) & ~(pageSize - 1);
  EXPECT_EQ(item->endAddr, expectedEnd);
}

// =============================================================================
// 2. Refcounting
// =============================================================================

TEST_F(RegPoolTest, RegisterBuffer_SameBufferTwice_IncrementsRefCount) {
  void *data = alignedAddr(20);
  ASSERT_EQ(pool->registerBuffer(fakeComm(1), data, pageSize), flagcxSuccess);
  ASSERT_EQ(pool->registerBuffer(fakeComm(1), data, pageSize), flagcxSuccess);

  flagcxRegItem *item = pool->getItem(fakeComm(1), data);
  ASSERT_NE(item, nullptr);
  EXPECT_EQ(item->refCount, 2);
}

TEST_F(RegPoolTest, DeregisterBuffer_DecrementsRefCount) {
  void *data = alignedAddr(30);
  ASSERT_EQ(pool->registerBuffer(fakeComm(1), data, pageSize), flagcxSuccess);
  ASSERT_EQ(pool->registerBuffer(fakeComm(1), data, pageSize), flagcxSuccess);

  flagcxRegItem *item = pool->getItem(fakeComm(1), data);
  ASSERT_NE(item, nullptr);
  EXPECT_EQ(item->refCount, 2);

  ASSERT_EQ(pool->deregisterBuffer(fakeComm(1), item), flagcxSuccess);

  // Item should still exist with refCount=1
  flagcxRegItem *item2 = pool->getItem(fakeComm(1), data);
  ASSERT_NE(item2, nullptr);
  EXPECT_EQ(item2->refCount, 1);
}

TEST_F(RegPoolTest, DeregisterBuffer_LastRef_RemovesItem) {
  void *data = alignedAddr(40);
  ASSERT_EQ(pool->registerBuffer(fakeComm(1), data, pageSize), flagcxSuccess);

  flagcxRegItem *item = pool->getItem(fakeComm(1), data);
  ASSERT_NE(item, nullptr);

  ASSERT_EQ(pool->deregisterBuffer(fakeComm(1), item), flagcxSuccess);

  // Item should be gone
  EXPECT_EQ(pool->getItem(fakeComm(1), data), nullptr);
}

// =============================================================================
// 3. Pointer Stability (Issue #1 validation)
// =============================================================================

TEST_F(RegPoolTest, PointerStability_ManyInsertions) {
  constexpr int N = 1000;
  std::vector<flagcxRegItem *> items(N);

  // Register N distinct buffers
  for (int i = 0; i < N; i++) {
    void *data = alignedAddr(100 + i);
    ASSERT_EQ(pool->registerBuffer(nullptr, data, pageSize), flagcxSuccess);
    items[i] = pool->getItem(nullptr, data);
    ASSERT_NE(items[i], nullptr);
  }

  // Verify all pointers are still valid (no dangling after rehash)
  for (int i = 0; i < N; i++) {
    void *data = alignedAddr(100 + i);
    flagcxRegItem *current = pool->getItem(nullptr, data);
    EXPECT_EQ(current, items[i])
        << "Pointer mismatch at index " << i << " — likely rehash invalidation";
    EXPECT_EQ(current->beginAddr, (100 + i) * pageSize);
  }
}

// =============================================================================
// 4. Multi-Comm Semantics (Issue #2 validation)
// =============================================================================

TEST_F(RegPoolTest, RegisterBuffer_TwoComms_SameBuffer) {
  void *data = alignedAddr(50);
  void *commA = fakeComm(0x1000);
  void *commB = fakeComm(0x2000);

  ASSERT_EQ(pool->registerBuffer(commA, data, pageSize), flagcxSuccess);
  ASSERT_EQ(pool->registerBuffer(commB, data, pageSize), flagcxSuccess);

  flagcxRegItem *itemA = pool->getItem(commA, data);
  flagcxRegItem *itemB = pool->getItem(commB, data);
  ASSERT_NE(itemA, nullptr);
  ASSERT_NE(itemB, nullptr);
  // Same underlying item
  EXPECT_EQ(itemA, itemB);
  EXPECT_EQ(itemA->refCount, 2);
}

TEST_F(RegPoolTest, DeregisterBuffer_OneComm_OtherStillValid) {
  void *data = alignedAddr(60);
  void *commA = fakeComm(0x3000);
  void *commB = fakeComm(0x4000);

  ASSERT_EQ(pool->registerBuffer(commA, data, pageSize), flagcxSuccess);
  ASSERT_EQ(pool->registerBuffer(commB, data, pageSize), flagcxSuccess);

  flagcxRegItem *item = pool->getItem(commA, data);
  ASSERT_NE(item, nullptr);

  // Deregister from commA
  ASSERT_EQ(pool->deregisterBuffer(commA, item), flagcxSuccess);

  // commB should still find it
  flagcxRegItem *itemB = pool->getItem(commB, data);
  ASSERT_NE(itemB, nullptr);
  EXPECT_EQ(itemB->refCount, 1);

  // commA's mapping is gone, but global fallback still works
  flagcxRegItem *itemA = pool->getItem(commA, data);
  EXPECT_NE(itemA, nullptr); // found via global fallback
}

TEST_F(RegPoolTest, RegisterBuffer_NullComm_GlobalOnly) {
  void *data = alignedAddr(70);

  ASSERT_EQ(pool->registerBuffer(nullptr, data, pageSize), flagcxSuccess);

  // Null comm query finds it
  flagcxRegItem *item = pool->getItem(nullptr, data);
  ASSERT_NE(item, nullptr);

  // Non-null comm also finds it via global fallback
  flagcxRegItem *item2 = pool->getItem(fakeComm(0x5000), data);
  EXPECT_EQ(item2, item);
}

// =============================================================================
// 5. Handle Management (Issue #3 validation)
// =============================================================================

TEST_F(RegPoolTest, AddNetHandle_StoresOwnerComm) {
  void *data = alignedAddr(80);
  void *commA = fakeComm(0x6000);
  ASSERT_EQ(pool->registerBuffer(commA, data, pageSize), flagcxSuccess);
  flagcxRegItem *item = pool->getItem(commA, data);
  ASSERT_NE(item, nullptr);

  void *fakeHandle = reinterpret_cast<void *>(0xDEAD);
  auto *proxy = fakeProxy(0xBEEF);
  ASSERT_EQ(pool->addNetHandle(commA, item, fakeHandle, proxy), flagcxSuccess);

  ASSERT_EQ(item->handles.size(), 1u);
  EXPECT_EQ(item->handles[0].first.handle, fakeHandle);
  EXPECT_EQ(item->handles[0].first.proxyConn, proxy);
  EXPECT_EQ(item->handles[0].first.ownerComm, commA);
}

TEST_F(RegPoolTest, AddP2pHandle_StoresOwnerComm) {
  void *data = alignedAddr(81);
  void *commA = fakeComm(0x7000);
  ASSERT_EQ(pool->registerBuffer(commA, data, pageSize), flagcxSuccess);
  flagcxRegItem *item = pool->getItem(commA, data);
  ASSERT_NE(item, nullptr);

  flagcxIpcRegInfo fakeHandle = {};
  fakeHandle.peerRank = 1;
  fakeHandle.allocationBase = reinterpret_cast<void *>(0xCAFE0000);
  auto *proxy = fakeProxy(0xFACE);
  ASSERT_EQ(pool->addP2pHandle(commA, item, &fakeHandle, proxy), flagcxSuccess);

  ASSERT_EQ(item->handles.size(), 1u);
  EXPECT_EQ(item->handles[0].second.handle, &fakeHandle);
  EXPECT_EQ(item->handles[0].second.proxyConn, proxy);
  EXPECT_EQ(item->handles[0].second.ownerComm, commA);
}

TEST_F(RegPoolTest, AddP2pHandle_SameProxyDifferentAllocationKeepsBoth) {
  void *data = alignedAddr(82);
  void *commA = fakeComm(0x8000);
  ASSERT_EQ(pool->registerBuffer(commA, data, pageSize), flagcxSuccess);
  flagcxRegItem *item = pool->getItem(commA, data);
  ASSERT_NE(item, nullptr);

  auto *proxy = fakeProxy(0xAAAA);
  flagcxIpcRegInfo first = {};
  first.peerRank = 1;
  first.allocationBase = reinterpret_cast<void *>(0x100000);
  flagcxIpcRegInfo second = {};
  second.peerRank = 1;
  second.allocationBase = reinterpret_cast<void *>(0x200000);

  ASSERT_EQ(pool->addP2pHandle(commA, item, &first, proxy), flagcxSuccess);
  ASSERT_EQ(pool->addP2pHandle(commA, item, &second, proxy), flagcxSuccess);

  ASSERT_EQ(item->handles.size(), 2u);
  EXPECT_EQ(item->handles[0].second.handle, &first);
  EXPECT_EQ(item->handles[1].second.handle, &second);
}

TEST_F(RegPoolTest, AddP2pHandle_DuplicateMappingIsRejected) {
  void *data = alignedAddr(83);
  void *commA = fakeComm(0x9000);
  ASSERT_EQ(pool->registerBuffer(commA, data, pageSize), flagcxSuccess);
  flagcxRegItem *item = pool->getItem(commA, data);
  ASSERT_NE(item, nullptr);

  auto *proxy = fakeProxy(0xBBBB);
  flagcxIpcRegInfo first = {};
  first.peerRank = 2;
  first.allocationBase = reinterpret_cast<void *>(0x300000);
  flagcxIpcRegInfo duplicate = first;

  ASSERT_EQ(pool->addP2pHandle(commA, item, &first, proxy), flagcxSuccess);
  EXPECT_EQ(pool->addP2pHandle(commA, item, &duplicate, proxy),
            flagcxInvalidUsage);
  ASSERT_EQ(item->handles.size(), 1u);
  EXPECT_EQ(item->handles[0].second.handle, &first);
}

TEST_F(RegPoolTest, AddP2pHandle_SharesAllocationMappingAcrossRegistrations) {
  void *commA = fakeComm(0xA000);
  void *firstData = alignedAddr(84);
  void *secondData = alignedAddr(86);
  ASSERT_EQ(pool->registerBuffer(commA, firstData, pageSize), flagcxSuccess);
  ASSERT_EQ(pool->registerBuffer(commA, secondData, pageSize), flagcxSuccess);
  flagcxRegItem *firstItem = pool->getItem(commA, firstData);
  flagcxRegItem *secondItem = pool->getItem(commA, secondData);
  ASSERT_NE(firstItem, nullptr);
  ASSERT_NE(secondItem, nullptr);
  ASSERT_NE(firstItem, secondItem);

  auto *proxy = fakeProxy(0xCCCC);
  flagcxIpcRegInfo mapping = {};
  mapping.peerRank = 3;
  mapping.allocationBase = reinterpret_cast<void *>(0x400000);

  ASSERT_EQ(pool->addP2pHandle(commA, firstItem, &mapping, proxy),
            flagcxSuccess);
  EXPECT_EQ(mapping.refCount, 1);
  ASSERT_EQ(pool->addP2pHandle(commA, firstItem, &mapping, proxy),
            flagcxSuccess);
  EXPECT_EQ(mapping.refCount, 1);
  ASSERT_EQ(pool->addP2pHandle(commA, secondItem, &mapping, proxy),
            flagcxSuccess);
  EXPECT_EQ(mapping.refCount, 2);
  EXPECT_EQ(pool->findP2pHandle(commA, 3, mapping.allocationBase), &mapping);
}

TEST_F(RegPoolTest, DeregisterP2pHandle_ReleasesSharedMappingAfterLastUse) {
  void *commA = fakeComm(0xB000);
  void *firstData = alignedAddr(88);
  void *secondData = alignedAddr(90);
  ASSERT_EQ(pool->registerBuffer(commA, firstData, pageSize), flagcxSuccess);
  ASSERT_EQ(pool->registerBuffer(commA, secondData, pageSize), flagcxSuccess);
  flagcxRegItem *firstItem = pool->getItem(commA, firstData);
  flagcxRegItem *secondItem = pool->getItem(commA, secondData);
  ASSERT_NE(firstItem, nullptr);
  ASSERT_NE(secondItem, nullptr);

  auto *mapping =
      static_cast<flagcxIpcRegInfo *>(calloc(1, sizeof(flagcxIpcRegInfo)));
  ASSERT_NE(mapping, nullptr);
  mapping->peerRank = 4;
  mapping->allocationBase = reinterpret_cast<void *>(0x500000);
  mapping->sameProcess = true;
  mapping->impInfo.importedBase = mapping->allocationBase;
  mapping->impInfo.legacyIpcCap = true;
  auto *proxy = fakeProxy(0xDDDD);

  ASSERT_EQ(pool->addP2pHandle(commA, firstItem, mapping, proxy),
            flagcxSuccess);
  ASSERT_EQ(pool->addP2pHandle(commA, secondItem, mapping, proxy),
            flagcxSuccess);
  ASSERT_EQ(mapping->refCount, 2);

  ASSERT_EQ(pool->deregisterBuffer(commA, firstItem), flagcxSuccess);
  ASSERT_EQ(mapping->refCount, 1);
  EXPECT_EQ(pool->findP2pHandle(commA, 4, mapping->allocationBase), mapping);

  void *allocationBase = mapping->allocationBase;
  ASSERT_EQ(pool->deregisterBuffer(commA, secondItem), flagcxSuccess);
  EXPECT_EQ(pool->findP2pHandle(commA, 4, allocationBase), nullptr);
}

TEST_F(RegPoolTest, AddNetHandle_DuplicateProxyConn_Updates) {
  void *data = alignedAddr(82);
  void *commA = fakeComm(0x8000);
  ASSERT_EQ(pool->registerBuffer(commA, data, pageSize), flagcxSuccess);
  flagcxRegItem *item = pool->getItem(commA, data);
  ASSERT_NE(item, nullptr);

  auto *proxy = fakeProxy(0xAAAA);
  void *handle1 = reinterpret_cast<void *>(0x1111);
  void *handle2 = reinterpret_cast<void *>(0x2222);

  ASSERT_EQ(pool->addNetHandle(commA, item, handle1, proxy), flagcxSuccess);
  ASSERT_EQ(pool->addNetHandle(commA, item, handle2, proxy), flagcxSuccess);

  // Should update in-place, not add a second entry
  EXPECT_EQ(item->handles.size(), 1u);
  EXPECT_EQ(item->handles[0].first.handle, handle2);
}

TEST_F(RegPoolTest, AddNetHandle_NullReg_NoOp) {
  EXPECT_EQ(pool->addNetHandle(fakeComm(1), nullptr, nullptr, nullptr),
            flagcxSuccess);
}

// =============================================================================
// 6. Page Mapping
// =============================================================================

TEST_F(RegPoolTest, GetItem_DifferentOffsetSamePage) {
  // Register at page boundary
  void *data = alignedAddr(90);
  ASSERT_EQ(pool->registerBuffer(fakeComm(1), data, pageSize), flagcxSuccess);

  // Query with an offset within the same page
  void *offsetData = reinterpret_cast<void *>(90 * pageSize + 128);
  flagcxRegItem *item = pool->getItem(fakeComm(1), offsetData);
  ASSERT_NE(item, nullptr);
  EXPECT_EQ(item->beginAddr, 90 * pageSize);
}

TEST_F(RegPoolTest, GetItem_CommSpecificFallsToGlobal) {
  void *data = alignedAddr(91);
  // Register with null comm (global only)
  ASSERT_EQ(pool->registerBuffer(nullptr, data, pageSize), flagcxSuccess);

  // Query with a comm that never registered — should fall through to global
  flagcxRegItem *item = pool->getItem(fakeComm(0x9000), data);
  EXPECT_NE(item, nullptr);
}

// =============================================================================
// 7. Edge Cases
// =============================================================================

TEST_F(RegPoolTest, DeregisterBuffer_NullHandle_NoOp) {
  EXPECT_EQ(pool->deregisterBuffer(fakeComm(1), nullptr), flagcxSuccess);
}

TEST_F(RegPoolTest, DeregisterBuffer_InvalidHandle_ReturnsError) {
  // Create a stack-local regItem that's not in the pool
  flagcxRegItem fakeItem;
  fakeItem.beginAddr = 999 * pageSize;
  fakeItem.endAddr = 1000 * pageSize;

  EXPECT_EQ(pool->deregisterBuffer(fakeComm(1), &fakeItem), flagcxInvalidUsage);
}

// =============================================================================
// homoRegHandles (per-comm storage in flagcxRegItem)
// =============================================================================

TEST_F(RegPoolTest, HomoRegHandles_PerCommStorage) {
  void *data = alignedAddr(100);
  ASSERT_EQ(pool->registerBuffer(nullptr, data, pageSize), flagcxSuccess);
  flagcxRegItem *item = pool->getItem(nullptr, data);
  ASSERT_NE(item, nullptr);

  uintptr_t commKeyA = 0xA000;
  uintptr_t commKeyB = 0xB000;
  void *handleA = reinterpret_cast<void *>(0xAAAA);
  void *handleB = reinterpret_cast<void *>(0xBBBB);

  item->homoRegHandles[commKeyA] = handleA;
  item->homoRegHandles[commKeyB] = handleB;

  EXPECT_EQ(item->homoRegHandles.size(), 2u);
  EXPECT_EQ(item->homoRegHandles[commKeyA], handleA);
  EXPECT_EQ(item->homoRegHandles[commKeyB], handleB);
}

TEST_F(RegPoolTest, HomoRegHandles_EraseOneComm_OtherRemains) {
  void *data = alignedAddr(101);
  ASSERT_EQ(pool->registerBuffer(nullptr, data, pageSize), flagcxSuccess);
  flagcxRegItem *item = pool->getItem(nullptr, data);
  ASSERT_NE(item, nullptr);

  uintptr_t commKeyA = 0xC000;
  uintptr_t commKeyB = 0xD000;
  item->homoRegHandles[commKeyA] = reinterpret_cast<void *>(0x1);
  item->homoRegHandles[commKeyB] = reinterpret_cast<void *>(0x2);

  item->homoRegHandles.erase(commKeyA);

  EXPECT_EQ(item->homoRegHandles.count(commKeyA), 0u);
  EXPECT_EQ(item->homoRegHandles.count(commKeyB), 1u);
  EXPECT_EQ(item->homoRegHandles[commKeyB], reinterpret_cast<void *>(0x2));
}

// =============================================================================
// 10. Register/Deregister Symmetry (API contract scenarios)
// =============================================================================

TEST_F(RegPoolTest, RegisterNullComm_DeregisterNullComm_Works) {
  // Register(nullptr) + Deregister(nullptr) → pool-only, works
  void *data = alignedAddr(200);
  ASSERT_EQ(pool->registerBuffer(nullptr, data, pageSize), flagcxSuccess);

  flagcxRegItem *item = pool->getItem(nullptr, data);
  ASSERT_NE(item, nullptr);
  EXPECT_EQ(item->refCount, 1);

  ASSERT_EQ(pool->deregisterBuffer(nullptr, item), flagcxSuccess);
  EXPECT_EQ(pool->getItem(nullptr, data), nullptr);
}

TEST_F(RegPoolTest, RegisterComm_DeregisterComm_Works) {
  // Register(comm) + Deregister(comm) → full cleanup, works
  void *data = alignedAddr(201);
  void *commA = fakeComm(0xA000);
  ASSERT_EQ(pool->registerBuffer(commA, data, pageSize), flagcxSuccess);

  flagcxRegItem *item = pool->getItem(commA, data);
  ASSERT_NE(item, nullptr);
  EXPECT_EQ(item->refCount, 1);

  ASSERT_EQ(pool->deregisterBuffer(commA, item), flagcxSuccess);
  EXPECT_EQ(pool->getItem(commA, data), nullptr);
  EXPECT_EQ(pool->getItem(nullptr, data), nullptr);
}

TEST_F(RegPoolTest, RegisterComm_DeregisterNullComm_PoolCleanupOnly) {
  // Register(comm) + Deregister(nullptr) → pool removes item but cannot
  // clean backend handles. At pool level this succeeds (pool doesn't know
  // about backend handles). The flagcxCommDeregister layer guards this.
  void *data = alignedAddr(202);
  void *commA = fakeComm(0xB000);
  ASSERT_EQ(pool->registerBuffer(commA, data, pageSize), flagcxSuccess);

  flagcxRegItem *item = pool->getItem(commA, data);
  ASSERT_NE(item, nullptr);

  // Pool-level deregister with nullptr succeeds (no backend awareness)
  ASSERT_EQ(pool->deregisterBuffer(nullptr, item), flagcxSuccess);
  // Item removed from global pool
  EXPECT_EQ(pool->getItem(nullptr, data), nullptr);
  // Comm-specific mappings also cleaned up (no dangling pointers)
  EXPECT_EQ(pool->getItem(commA, data), nullptr);
}
