// Unit tests for the unified MR Registry (flagcx_mr_registry.h).
// Pure C data structure — no MPI, no GPU, no network needed.

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

extern "C" {
#include "flagcx_mr_registry.h"
}

class MrRegistryTest : public ::testing::Test {
protected:
  struct flagcxMrRegistry *reg = nullptr;

  void SetUp() override {
    ASSERT_EQ(flagcxMrRegistryCreate(&reg), flagcxSuccess);
    ASSERT_NE(reg, nullptr);
  }

  void TearDown() override {
    if (reg) {
      flagcxMrRegistryDestroy(reg);
      reg = nullptr;
    }
  }

  // Helper: create a P2P extension with given ipc state
  struct flagcxMrP2pExt *makeP2pExt(bool hasIpc = false) {
    auto *ext =
        (struct flagcxMrP2pExt *)calloc(1, sizeof(struct flagcxMrP2pExt));
    ext->mrId = 0; // let registry assign
    ext->hasIpc = hasIpc;
    return ext;
  }
};

TEST_F(MrRegistryTest, CreateAndDestroy) {
  // SetUp already verified creation; this tests the empty state
  ASSERT_EQ(flagcxMrRegistryRdLock(reg), flagcxSuccess);
  EXPECT_EQ(flagcxMrRegistryCount(reg), 0);
  EXPECT_EQ(flagcxMrRegistryEntries(reg), nullptr);
  ASSERT_EQ(flagcxMrRegistryRdUnlock(reg), flagcxSuccess);
}

TEST_F(MrRegistryTest, RegisterSingleEntry) {
  auto *ext = makeP2pExt();
  uint64_t mrId = 0;

  ASSERT_EQ(flagcxMrRegistryRegister(reg, 0x1000, 4096, 0, 0,
                                     FLAGCX_MR_OWNER_P2P,
                                     (void *)(uintptr_t)0xDEAD, ext, &mrId),
            flagcxSuccess);

  EXPECT_GT(mrId, 0u);
  EXPECT_EQ(flagcxMrRegistryCount(reg), 1);
}

TEST_F(MrRegistryTest, LookupContainment) {
  auto *ext = makeP2pExt();
  uint64_t mrId = 0;
  ASSERT_EQ(flagcxMrRegistryRegister(reg, 0x1000, 4096, 0, 0,
                                     FLAGCX_MR_OWNER_P2P,
                                     (void *)(uintptr_t)0xDEAD, ext, &mrId),
            flagcxSuccess);

  // Addr within range
  struct flagcxMrEntry found;
  struct flagcxMrExtension p2pExt;
  struct flagcxMrExtension *exts[FLAGCX_MR_OWNER_COUNT] = {&p2pExt, NULL, NULL};
  ASSERT_EQ(flagcxMrRegistryLookup(reg, 0x1500, &found, exts), flagcxSuccess);
  EXPECT_EQ(found.baseAddr, (uintptr_t)0x1000);
  EXPECT_EQ(found.size, 4096u);
  EXPECT_EQ(p2pExt.type, (uint32_t)FLAGCX_MR_OWNER_P2P);
  EXPECT_EQ(p2pExt.p2p.mrId, mrId);

  // Addr at base
  ASSERT_EQ(flagcxMrRegistryLookup(reg, 0x1000, &found, NULL), flagcxSuccess);

  // Addr past end — should fail
  EXPECT_NE(flagcxMrRegistryLookup(reg, 0x2000, &found, NULL), flagcxSuccess);

  // Addr before — should fail
  EXPECT_NE(flagcxMrRegistryLookup(reg, 0x0FFF, &found, NULL), flagcxSuccess);
}

TEST_F(MrRegistryTest, FindExact) {
  auto *ext = makeP2pExt();
  ASSERT_EQ(flagcxMrRegistryRegister(reg, 0x2000, 8192, 1, 2,
                                     FLAGCX_MR_OWNER_P2P,
                                     (void *)(uintptr_t)0xBEEF, ext, nullptr),
            flagcxSuccess);

  struct flagcxMrEntry found;
  ASSERT_EQ(flagcxMrRegistryFindExact(reg, 0x2000, &found, NULL),
            flagcxSuccess);
  EXPECT_EQ(found.ibDevN, 1);
  EXPECT_EQ(found.ptrType, 2);

  // Non-exact addr inside range should fail for FindExact
  EXPECT_NE(flagcxMrRegistryFindExact(reg, 0x2001, &found, NULL),
            flagcxSuccess);
}

TEST_F(MrRegistryTest, LookupById) {
  auto *ext = makeP2pExt();
  uint64_t mrId = 0;
  ASSERT_EQ(flagcxMrRegistryRegister(reg, 0x3000, 4096, 0, 0,
                                     FLAGCX_MR_OWNER_P2P,
                                     (void *)(uintptr_t)0xCAFE, ext, &mrId),
            flagcxSuccess);

  struct flagcxMrEntry found;
  ASSERT_EQ(flagcxMrRegistryLookupById(reg, mrId, &found, NULL), flagcxSuccess);
  EXPECT_EQ(found.baseAddr, (uintptr_t)0x3000);

  // Non-existent ID
  EXPECT_NE(flagcxMrRegistryLookupById(reg, mrId + 100, &found, NULL),
            flagcxSuccess);
}

TEST_F(MrRegistryTest, IdempotentRegistration) {
  auto *ext1 = makeP2pExt();
  uint64_t mrId1 = 0, mrId2 = 0;
  ASSERT_EQ(flagcxMrRegistryRegister(reg, 0x4000, 4096, 0, 0,
                                     FLAGCX_MR_OWNER_P2P,
                                     (void *)(uintptr_t)0x1, ext1, &mrId1),
            flagcxSuccess);

  // Same addr+size, same owner → idempotent, returns same mrId
  ASSERT_EQ(flagcxMrRegistryRegister(reg, 0x4000, 4096, 0, 0,
                                     FLAGCX_MR_OWNER_P2P,
                                     (void *)(uintptr_t)0x2, nullptr, &mrId2),
            flagcxSuccess);
  EXPECT_EQ(mrId1, mrId2);
  EXPECT_EQ(flagcxMrRegistryCount(reg), 1);
}

TEST_F(MrRegistryTest, MhandleUpdate) {
  auto *ext1 = makeP2pExt();
  ext1->mrId = 0; // let registry assign
  uint64_t mrId1 = 0;
  ASSERT_EQ(flagcxMrRegistryRegister(reg, 0x9000, 4096, 0, 0,
                                     FLAGCX_MR_OWNER_P2P,
                                     (void *)(uintptr_t)0x10, ext1, &mrId1),
            flagcxSuccess);
  ASSERT_NE(mrId1, 0u);

  // Re-register with different mhandle AND new ext → ext replaced
  auto *ext2 = makeP2pExt();
  ext2->hasIpc = true;
  uint64_t mrId2 = 0;
  ASSERT_EQ(flagcxMrRegistryRegister(reg, 0x9000, 4096, 0, 0,
                                     FLAGCX_MR_OWNER_P2P,
                                     (void *)(uintptr_t)0x20, ext2, &mrId2),
            flagcxSuccess);
  // mrId preserved from original registration
  EXPECT_EQ(mrId2, mrId1);
  EXPECT_EQ(flagcxMrRegistryCount(reg), 1);

  // Verify mhandle and ext are updated
  struct flagcxMrEntry found;
  struct flagcxMrExtension p2pExt;
  struct flagcxMrExtension *exts[FLAGCX_MR_OWNER_COUNT] = {&p2pExt, NULL, NULL};
  ASSERT_EQ(flagcxMrRegistryFindExact(reg, 0x9000, &found, exts),
            flagcxSuccess);
  EXPECT_EQ(found.mhandles[FLAGCX_MR_OWNER_IDX_P2P], (void *)(uintptr_t)0x20);
  EXPECT_EQ(p2pExt.type, (uint32_t)FLAGCX_MR_OWNER_P2P);
  EXPECT_EQ(p2pExt.p2p.hasIpc, true);

  // Re-register with different mhandle but NULL ext → mhandle updated, ext
  // preserved
  uint64_t mrId3 = 0;
  ASSERT_EQ(flagcxMrRegistryRegister(reg, 0x9000, 4096, 0, 0,
                                     FLAGCX_MR_OWNER_P2P,
                                     (void *)(uintptr_t)0x30, nullptr, &mrId3),
            flagcxSuccess);
  EXPECT_EQ(mrId3, mrId1);
  ASSERT_EQ(flagcxMrRegistryFindExact(reg, 0x9000, &found, exts),
            flagcxSuccess);
  EXPECT_EQ(found.mhandles[FLAGCX_MR_OWNER_IDX_P2P], (void *)(uintptr_t)0x30);
  // ext2 still present (not freed since new ext was NULL)
  EXPECT_EQ(p2pExt.type, (uint32_t)FLAGCX_MR_OWNER_P2P);
  EXPECT_EQ(p2pExt.p2p.hasIpc, true);
}

TEST_F(MrRegistryTest, MultiOwnerRegistration) {
  auto *p2pExt = makeP2pExt();
  auto *collExt =
      (struct flagcxMrCollExt *)calloc(1, sizeof(struct flagcxMrCollExt));
  collExt->channelId = 7;

  uint64_t mrId = 0;
  ASSERT_EQ(flagcxMrRegistryRegister(reg, 0x5000, 4096, 0, 0,
                                     FLAGCX_MR_OWNER_P2P,
                                     (void *)(uintptr_t)0xA, p2pExt, &mrId),
            flagcxSuccess);
  ASSERT_EQ(flagcxMrRegistryRegister(reg, 0x5000, 4096, 0, 0,
                                     FLAGCX_MR_OWNER_COLL,
                                     (void *)(uintptr_t)0xB, collExt, nullptr),
            flagcxSuccess);

  EXPECT_EQ(flagcxMrRegistryCount(reg), 1);

  struct flagcxMrEntry found;
  struct flagcxMrExtension collExt2;
  struct flagcxMrExtension *exts[FLAGCX_MR_OWNER_COUNT] = {NULL, &collExt2,
                                                           NULL};
  ASSERT_EQ(flagcxMrRegistryFindExact(reg, 0x5000, &found, exts),
            flagcxSuccess);
  EXPECT_EQ(found.ownerMask,
            (uint32_t)(FLAGCX_MR_OWNER_P2P | FLAGCX_MR_OWNER_COLL));
  EXPECT_EQ(found.mhandles[FLAGCX_MR_OWNER_IDX_P2P], (void *)(uintptr_t)0xA);
  EXPECT_EQ(found.mhandles[FLAGCX_MR_OWNER_IDX_COLL], (void *)(uintptr_t)0xB);
  EXPECT_EQ(collExt2.type, (uint32_t)FLAGCX_MR_OWNER_COLL);
  EXPECT_EQ(collExt2.coll.channelId, 7);
}

TEST_F(MrRegistryTest, SizeMismatchRejected) {
  auto *ext = makeP2pExt();
  ASSERT_EQ(flagcxMrRegistryRegister(reg, 0x6000, 4096, 0, 0,
                                     FLAGCX_MR_OWNER_P2P,
                                     (void *)(uintptr_t)0x1, ext, nullptr),
            flagcxSuccess);

  auto *collExt =
      (struct flagcxMrCollExt *)calloc(1, sizeof(struct flagcxMrCollExt));
  // Different size → error
  EXPECT_NE(flagcxMrRegistryRegister(reg, 0x6000, 8192, 0, 0,
                                     FLAGCX_MR_OWNER_COLL,
                                     (void *)(uintptr_t)0x2, collExt, nullptr),
            flagcxSuccess);
  free(collExt); // caller retains ownership on failure
}

TEST_F(MrRegistryTest, OverlapRejected) {
  auto *ext1 = makeP2pExt();
  ASSERT_EQ(flagcxMrRegistryRegister(reg, 0x7000, 4096, 0, 0,
                                     FLAGCX_MR_OWNER_P2P,
                                     (void *)(uintptr_t)0x1, ext1, nullptr),
            flagcxSuccess);

  auto *ext2 = makeP2pExt();
  // Overlapping region
  EXPECT_NE(flagcxMrRegistryRegister(reg, 0x7800, 4096, 0, 0,
                                     FLAGCX_MR_OWNER_P2P,
                                     (void *)(uintptr_t)0x2, ext2, nullptr),
            flagcxSuccess);
  free(ext2);
}

TEST_F(MrRegistryTest, Deregister) {
  auto *ext = makeP2pExt();
  uint64_t mrId = 0;
  ASSERT_EQ(flagcxMrRegistryRegister(reg, 0x8000, 4096, 0, 0,
                                     FLAGCX_MR_OWNER_P2P,
                                     (void *)(uintptr_t)0xF00D, ext, &mrId),
            flagcxSuccess);

  void *removedExt = nullptr;
  struct flagcxMrEntry removedEntry;
  ASSERT_EQ(flagcxMrRegistryDeregister(reg, 0x8000, FLAGCX_MR_OWNER_P2P,
                                       &removedEntry, &removedExt),
            flagcxSuccess);
  EXPECT_EQ(removedEntry.baseAddr, (uintptr_t)0x8000);
  EXPECT_EQ(removedEntry.mhandles[FLAGCX_MR_OWNER_IDX_P2P],
            (void *)(uintptr_t)0xF00D);
  EXPECT_NE(removedExt, nullptr);
  free(removedExt);

  // Entry should be gone
  EXPECT_EQ(flagcxMrRegistryCount(reg), 0);
}

TEST_F(MrRegistryTest, DeregisterMultiOwnerPartial) {
  auto *p2pExt = makeP2pExt();
  auto *rmaExt =
      (struct flagcxMrRmaExt *)calloc(1, sizeof(struct flagcxMrRmaExt));
  rmaExt->oneSideHandleIdx = 42;

  ASSERT_EQ(flagcxMrRegistryRegister(reg, 0x9000, 4096, 0, 0,
                                     FLAGCX_MR_OWNER_P2P,
                                     (void *)(uintptr_t)0x1, p2pExt, nullptr),
            flagcxSuccess);
  ASSERT_EQ(flagcxMrRegistryRegister(reg, 0x9000, 4096, 0, 0,
                                     FLAGCX_MR_OWNER_RMA,
                                     (void *)(uintptr_t)0x2, rmaExt, nullptr),
            flagcxSuccess);

  // Remove P2P owner — entry should remain with RMA
  void *removedExt = nullptr;
  ASSERT_EQ(flagcxMrRegistryDeregister(reg, 0x9000, FLAGCX_MR_OWNER_P2P,
                                       nullptr, &removedExt),
            flagcxSuccess);
  free(removedExt);

  EXPECT_EQ(flagcxMrRegistryCount(reg), 1);
  struct flagcxMrEntry found;
  struct flagcxMrExtension p2pExt2, rmaExt2;
  struct flagcxMrExtension *exts[FLAGCX_MR_OWNER_COUNT] = {&p2pExt2, NULL,
                                                           &rmaExt2};
  ASSERT_EQ(flagcxMrRegistryFindExact(reg, 0x9000, &found, exts),
            flagcxSuccess);
  EXPECT_EQ(found.ownerMask, (uint32_t)FLAGCX_MR_OWNER_RMA);
  EXPECT_EQ(p2pExt2.type, (uint32_t)FLAGCX_MR_OWNER_NONE);
  EXPECT_EQ(rmaExt2.type, (uint32_t)FLAGCX_MR_OWNER_RMA);
  EXPECT_EQ(rmaExt2.rma.oneSideHandleIdx, 42);

  // Remove RMA — entry should be fully removed
  void *removedRma = nullptr;
  ASSERT_EQ(flagcxMrRegistryDeregister(reg, 0x9000, FLAGCX_MR_OWNER_RMA,
                                       nullptr, &removedRma),
            flagcxSuccess);
  free(removedRma);
  EXPECT_EQ(flagcxMrRegistryCount(reg), 0);
}

TEST_F(MrRegistryTest, SortedInsertionOrder) {
  // Insert in reverse order — internal array should still be sorted
  for (int i = 9; i >= 0; i--) {
    auto *ext = makeP2pExt();
    uintptr_t addr = 0x10000 + (uintptr_t)i * 0x1000;
    ASSERT_EQ(flagcxMrRegistryRegister(reg, addr, 0x800, 0, 0,
                                       FLAGCX_MR_OWNER_P2P, nullptr, ext,
                                       nullptr),
              flagcxSuccess);
  }

  EXPECT_EQ(flagcxMrRegistryCount(reg), 10);

  ASSERT_EQ(flagcxMrRegistryRdLock(reg), flagcxSuccess);
  struct flagcxMrEntry *entries = flagcxMrRegistryEntries(reg);
  for (int i = 1; i < 10; i++) {
    EXPECT_LT(entries[i - 1].baseAddr, entries[i].baseAddr);
  }
  ASSERT_EQ(flagcxMrRegistryRdUnlock(reg), flagcxSuccess);
}

TEST_F(MrRegistryTest, MrIdMonotonicallyIncreasing) {
  uint64_t ids[5];
  for (int i = 0; i < 5; i++) {
    auto *ext = makeP2pExt();
    uintptr_t addr = 0x20000 + (uintptr_t)i * 0x2000;
    ASSERT_EQ(flagcxMrRegistryRegister(reg, addr, 0x1000, 0, 0,
                                       FLAGCX_MR_OWNER_P2P, nullptr, ext,
                                       &ids[i]),
              flagcxSuccess);
  }
  for (int i = 1; i < 5; i++) {
    EXPECT_GT(ids[i], ids[i - 1]);
  }
}

TEST_F(MrRegistryTest, ConcurrentLookup) {
  // Register several entries, then hammer with concurrent reads
  const int N = 100;
  for (int i = 0; i < N; i++) {
    auto *ext = makeP2pExt();
    uintptr_t addr = 0x100000 + (uintptr_t)i * 0x10000;
    ASSERT_EQ(flagcxMrRegistryRegister(reg, addr, 0x8000, 0, 0,
                                       FLAGCX_MR_OWNER_P2P, nullptr, ext,
                                       nullptr),
              flagcxSuccess);
  }

  // Spawn 4 reader threads, each doing lookups
  std::vector<std::thread> threads;
  std::atomic<int> errors{0};
  for (int t = 0; t < 4; t++) {
    threads.emplace_back([&, t]() {
      for (int iter = 0; iter < 1000; iter++) {
        int idx = (iter + t * 251) % N;
        uintptr_t addr = 0x100000 + (uintptr_t)idx * 0x10000 + 0x100;
        struct flagcxMrEntry found;
        if (flagcxMrRegistryLookup(reg, addr, &found, NULL) != flagcxSuccess)
          errors++;
      }
    });
  }
  for (auto &th : threads)
    th.join();

  EXPECT_EQ(errors.load(), 0);
}

TEST_F(MrRegistryTest, FindByHandle) {
  auto *ext = makeP2pExt();
  void *handle = (void *)(uintptr_t)0xBEEF;
  uint64_t mrId = 0;

  ASSERT_EQ(flagcxMrRegistryRegister(reg, 0x5000, 8192, 0, 0,
                                     FLAGCX_MR_OWNER_P2P, handle, ext, &mrId),
            flagcxSuccess);

  // Find by P2P handle
  struct flagcxMrEntry found;
  ASSERT_EQ(flagcxMrRegistryFindByHandle(reg, FLAGCX_MR_OWNER_IDX_P2P, handle,
                                         &found, NULL),
            flagcxSuccess);
  EXPECT_EQ(found.baseAddr, (uintptr_t)0x5000);
  EXPECT_EQ(found.size, 8192u);
  EXPECT_EQ(found.mhandles[FLAGCX_MR_OWNER_IDX_P2P], handle);

  // Wrong owner index — should not find
  EXPECT_NE(flagcxMrRegistryFindByHandle(reg, FLAGCX_MR_OWNER_IDX_COLL, handle,
                                         &found, NULL),
            flagcxSuccess);

  // Non-existent handle — should not find
  EXPECT_NE(flagcxMrRegistryFindByHandle(reg, FLAGCX_MR_OWNER_IDX_P2P,
                                         (void *)(uintptr_t)0xDEADDEAD, &found,
                                         NULL),
            flagcxSuccess);

  // NULL handle — should return error
  EXPECT_NE(flagcxMrRegistryFindByHandle(reg, FLAGCX_MR_OWNER_IDX_P2P, NULL,
                                         &found, NULL),
            flagcxSuccess);
}

TEST_F(MrRegistryTest, RmaOwnerRegistration) {
  // Simulate RMA one-sided registration with flagcxMrRmaExt
  struct flagcxMrRmaExt *ext =
      (struct flagcxMrRmaExt *)calloc(1, sizeof(struct flagcxMrRmaExt));
  ext->oneSideHandleIdx = 0; // data buffer slot 0
  void *handle = (void *)(uintptr_t)0xABCD;
  uint64_t mrId = 0;

  ASSERT_EQ(flagcxMrRegistryRegister(reg, 0x8000, 16384, 0, 2,
                                     FLAGCX_MR_OWNER_RMA, handle, ext, &mrId),
            flagcxSuccess);
  EXPECT_GT(mrId, 0u);
  EXPECT_EQ(flagcxMrRegistryCount(reg), 1);

  // Lookup should find it
  struct flagcxMrEntry found;
  struct flagcxMrExtension rmaExt2;
  struct flagcxMrExtension *exts[FLAGCX_MR_OWNER_COUNT] = {NULL, NULL,
                                                           &rmaExt2};
  ASSERT_EQ(flagcxMrRegistryLookup(reg, 0x8100, &found, exts), flagcxSuccess);
  EXPECT_EQ(found.baseAddr, (uintptr_t)0x8000);
  EXPECT_EQ(found.size, 16384u);
  EXPECT_EQ(rmaExt2.type, (uint32_t)FLAGCX_MR_OWNER_RMA);
  EXPECT_EQ(rmaExt2.rma.oneSideHandleIdx, 0);
  EXPECT_EQ(found.mhandles[FLAGCX_MR_OWNER_IDX_RMA], handle);

  // FindByHandle with RMA owner
  ASSERT_EQ(flagcxMrRegistryFindByHandle(reg, FLAGCX_MR_OWNER_IDX_RMA, handle,
                                         &found, NULL),
            flagcxSuccess);
  EXPECT_EQ(found.baseAddr, (uintptr_t)0x8000);

  // Deregister
  void *removedExt = NULL;
  ASSERT_EQ(flagcxMrRegistryDeregister(reg, 0x8000, FLAGCX_MR_OWNER_RMA, NULL,
                                       &removedExt),
            flagcxSuccess);
  EXPECT_EQ(removedExt, ext);
  free(removedExt);
  EXPECT_EQ(flagcxMrRegistryCount(reg), 0);
}

TEST_F(MrRegistryTest, MultipleRmaBuffers) {
  // Register data, signal, and staging buffers (different sentinel indices)
  struct {
    uintptr_t addr;
    size_t size;
    int handleIdx;
  } buffers[] = {
      {0x10000, 4096, 0}, // data buffer
      {0x20000, 256, -1}, // signal buffer
      {0x30000, 512, -2}, // staging buffer
  };

  for (int i = 0; i < 3; i++) {
    struct flagcxMrRmaExt *ext =
        (struct flagcxMrRmaExt *)calloc(1, sizeof(struct flagcxMrRmaExt));
    ext->oneSideHandleIdx = buffers[i].handleIdx;
    ASSERT_EQ(flagcxMrRegistryRegister(reg, buffers[i].addr, buffers[i].size, 0,
                                       2, FLAGCX_MR_OWNER_RMA,
                                       (void *)(uintptr_t)(0xF00 + i), ext,
                                       NULL),
              flagcxSuccess);
  }
  EXPECT_EQ(flagcxMrRegistryCount(reg), 3);

  // Each is independently findable by address
  struct flagcxMrEntry found;
  struct flagcxMrExtension rmaExt2;
  struct flagcxMrExtension *exts[FLAGCX_MR_OWNER_COUNT] = {NULL, NULL,
                                                           &rmaExt2};
  ASSERT_EQ(flagcxMrRegistryLookup(reg, 0x20080, &found, exts), flagcxSuccess);
  EXPECT_EQ(rmaExt2.rma.oneSideHandleIdx, -1); // signal

  // Deregister all
  for (int i = 0; i < 3; i++) {
    void *removedExt = NULL;
    ASSERT_EQ(flagcxMrRegistryDeregister(
                  reg, buffers[i].addr, FLAGCX_MR_OWNER_RMA, NULL, &removedExt),
              flagcxSuccess);
    free(removedExt);
  }
  EXPECT_EQ(flagcxMrRegistryCount(reg), 0);
}

TEST_F(MrRegistryTest, GlobalInitIdempotent) {
  ASSERT_EQ(flagcxMrRegistryGlobalInit(), flagcxSuccess);
  ASSERT_NE(flagcxGlobalMrRegistry, nullptr);

  // Second call is safe (refcount incremented)
  ASSERT_EQ(flagcxMrRegistryGlobalInit(), flagcxSuccess);

  // Release both refs
  ASSERT_EQ(flagcxMrRegistryGlobalRelease(), flagcxSuccess);
  ASSERT_EQ(flagcxMrRegistryGlobalRelease(), flagcxSuccess);
  EXPECT_EQ(flagcxGlobalMrRegistry, nullptr);
}

TEST_F(MrRegistryTest, ConcurrentWriters) {
  // Spawn multiple threads each registering and deregistering their own
  // non-overlapping address ranges to stress the write-lock path.
  const int kNumThreads = 8;
  const int kOpsPerThread = 100;
  const uintptr_t kPageSize = 4096;

  std::atomic<int> errors{0};
  auto worker = [&](int tid) {
    for (int i = 0; i < kOpsPerThread; i++) {
      // Each thread uses a unique address range: tid * (1<<20) + i * page
      uintptr_t addr = ((uintptr_t)(tid + 1) << 20) + (uintptr_t)i * kPageSize;
      size_t size = kPageSize;

      struct flagcxMrP2pExt *ext =
          (struct flagcxMrP2pExt *)calloc(1, sizeof(struct flagcxMrP2pExt));
      uint64_t mrId = 0;

      flagcxResult_t res =
          flagcxMrRegistryRegister(reg, addr, size, 0, 1, FLAGCX_MR_OWNER_P2P,
                                   (void *)(uintptr_t)(tid + 1), ext, &mrId);
      if (res != flagcxSuccess) {
        errors++;
        free(ext);
        continue;
      }

      // Deregister immediately
      struct flagcxMrEntry outEntry;
      void *outExt = nullptr;
      res = flagcxMrRegistryDeregister(reg, addr, FLAGCX_MR_OWNER_P2P,
                                       &outEntry, &outExt);
      if (res != flagcxSuccess)
        errors++;
      free(outExt);
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int t = 0; t < kNumThreads; t++) {
    threads.emplace_back(worker, t);
  }
  for (auto &th : threads) {
    th.join();
  }

  EXPECT_EQ(errors.load(), 0);
  // All entries should have been deregistered
  EXPECT_EQ(flagcxMrRegistryCount(reg), 0);
}

TEST_F(MrRegistryTest, LookupByIdWithIndex) {
  // Register several P2P entries, verify LookupById works for all
  const int kCount = 5;
  uint64_t mrIds[kCount];

  for (int i = 0; i < kCount; i++) {
    auto *ext =
        (struct flagcxMrP2pExt *)calloc(1, sizeof(struct flagcxMrP2pExt));
    ext->hasIpc = (i % 2 == 0);
    uintptr_t addr = 0x10000 + (uintptr_t)i * 0x1000;
    ASSERT_EQ(flagcxMrRegistryRegister(
                  reg, addr, 0x1000, 0, 1, FLAGCX_MR_OWNER_P2P,
                  (void *)(uintptr_t)(0xA00 + i), ext, &mrIds[i]),
              flagcxSuccess);
    EXPECT_NE(mrIds[i], (uint64_t)0);
  }

  // Lookup each by mrId
  for (int i = 0; i < kCount; i++) {
    struct flagcxMrEntry found;
    struct flagcxMrExtension p2pExt;
    struct flagcxMrExtension *exts[FLAGCX_MR_OWNER_COUNT] = {&p2pExt, NULL,
                                                             NULL};
    ASSERT_EQ(flagcxMrRegistryLookupById(reg, mrIds[i], &found, exts),
              flagcxSuccess);
    EXPECT_EQ(found.baseAddr, 0x10000 + (uintptr_t)i * 0x1000);
    EXPECT_EQ(found.size, (size_t)0x1000);
    EXPECT_EQ(p2pExt.type, (uint32_t)FLAGCX_MR_OWNER_P2P);
    EXPECT_EQ(p2pExt.p2p.mrId, mrIds[i]);
    EXPECT_EQ(p2pExt.p2p.hasIpc, (i % 2 == 0));
  }

  // Deregister entries [1] and [3], verify they become not-found
  for (int i : {1, 3}) {
    void *removedExt = nullptr;
    uintptr_t addr = 0x10000 + (uintptr_t)i * 0x1000;
    ASSERT_EQ(flagcxMrRegistryDeregister(reg, addr, FLAGCX_MR_OWNER_P2P, NULL,
                                         &removedExt),
              flagcxSuccess);
    free(removedExt);
  }

  // Removed entries should not be found
  for (int i : {1, 3}) {
    struct flagcxMrEntry found;
    EXPECT_NE(flagcxMrRegistryLookupById(reg, mrIds[i], &found, NULL),
              flagcxSuccess);
  }

  // Remaining entries still findable
  for (int i : {0, 2, 4}) {
    struct flagcxMrEntry found;
    struct flagcxMrExtension p2pExt;
    struct flagcxMrExtension *exts[FLAGCX_MR_OWNER_COUNT] = {&p2pExt, NULL,
                                                             NULL};
    ASSERT_EQ(flagcxMrRegistryLookupById(reg, mrIds[i], &found, exts),
              flagcxSuccess);
    EXPECT_EQ(found.baseAddr, 0x10000 + (uintptr_t)i * 0x1000);
    EXPECT_EQ(p2pExt.p2p.mrId, mrIds[i]);
  }

  // Cleanup remaining
  for (int i : {0, 2, 4}) {
    void *removedExt = nullptr;
    uintptr_t addr = 0x10000 + (uintptr_t)i * 0x1000;
    ASSERT_EQ(flagcxMrRegistryDeregister(reg, addr, FLAGCX_MR_OWNER_P2P, NULL,
                                         &removedExt),
              flagcxSuccess);
    free(removedExt);
  }
  EXPECT_EQ(flagcxMrRegistryCount(reg), 0);
}

TEST_F(MrRegistryTest, RejectWraparoundRegion) {
  auto *ext = makeP2pExt();
  // addr + size wraps uintptr_t — should be rejected
  EXPECT_NE(flagcxMrRegistryRegister(reg, UINTPTR_MAX - 10, 100, 0, 0,
                                     FLAGCX_MR_OWNER_P2P,
                                     (void *)(uintptr_t)0x1, ext, nullptr),
            flagcxSuccess);
  free(ext);
  EXPECT_EQ(flagcxMrRegistryCount(reg), 0);
}
