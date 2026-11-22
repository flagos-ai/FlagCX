#include "rma_init_rollback.hpp"

namespace {
class RmaOptionalSyncTest : public RmaInitRollbackTest {
protected:
  // Low bits select successful allocations; high bits select CPU mappings.
  unsigned resources = 15;
  void SetUp() override {
    RmaInitRollbackTest::SetUp();
    device.gdrMemAlloc = [](void **pointer, size_t, void *) {
      auto *self = static_cast<RmaOptionalSyncTest *>(current);
      unsigned index = self->allocations++;
      *pointer = nullptr;
      if (!(self->resources & (1u << index)))
        return flagcxNotSupported;
      *pointer = (void *)(uintptr_t)((index + 1) * 0x10000);
      return flagcxSuccess;
    };
    device.gdrPtrMmap = [](void **pointer, void *allocation, size_t) {
      auto *self = static_cast<RmaOptionalSyncTest *>(current);
      unsigned bit = allocation == (void *)0x10000 ? 4u : 8u;
      *pointer = nullptr;
      if (!(self->resources & bit))
        return flagcxNotSupported;
      *pointer = allocation;
      return flagcxSuccess;
    };
  }
  void start() {
    makeOwner(false);
    ASSERT_EQ(flagcxHeteroCommStartRma(&owner), flagcxSuccess);
    ASSERT_NE(owner, nullptr);
    ASSERT_NE(owner->rmaProxy, nullptr);
    ASSERT_NE(owner->rmaProxy->worker, nullptr);
    ASSERT_NE(owner->rmaProxy->doneSeqsCpu, nullptr);
    ASSERT_NE(owner->rmaProxy->readySeqsCpu, nullptr);
  }
  flagcxResult_t destroy() {
    flagcxResult_t result = flagcxHeteroCommDestroy(owner);
    if (result == flagcxSuccess)
      owner = nullptr;
    return result;
  }
};
class RmaOptionalSyncResources
    : public RmaOptionalSyncTest,
      public ::testing::WithParamInterface<unsigned> {};
} // namespace

TEST_P(RmaOptionalSyncResources,
       FinalizeStopsViewsAndRetainsActualAllocations) {
  resources = GetParam();
  ASSERT_NO_FATAL_FAILURE(start());
  auto *proxy = owner->rmaProxy;
  auto *done = proxy->doneSeqsDev;
  auto *ready = proxy->readySeqsDev;
  EXPECT_EQ(done != nullptr, (resources & 1u) != 0);
  EXPECT_EQ(ready != nullptr, (resources & 2u) != 0);
  int mapped = proxy->doneSeqsCpuMapped + proxy->readySeqsCpuMapped;
  if ((resources & 15u) != 15u) {
    EXPECT_FALSE(proxy->useStreamOps);
  }
  // Select the valid STREAM_OPS teardown case independently of the process's
  // cached RMA_STREAM_OPS environment. No GPU operation is issued here.
  if (resources & 16u) {
    // Do not change a synchronization mode observed by a live worker.
    ASSERT_EQ(owner->workerPool->retire(&proxy->worker), flagcxSuccess);
    proxy->useStreamOps = 1;
  }
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  EXPECT_EQ(owner->rmaProxy, proxy);
  EXPECT_TRUE(proxy->stopComplete);
  EXPECT_TRUE(proxy->quiesced);
  EXPECT_EQ(proxy->worker, nullptr);
  EXPECT_EQ(proxy->ipcState, nullptr);
  EXPECT_EQ(proxy->doneSeqsCpu, nullptr);
  EXPECT_EQ(proxy->readySeqsCpu, nullptr);
  EXPECT_FALSE(proxy->doneSeqsCpuMapped);
  EXPECT_FALSE(proxy->readySeqsCpuMapped);
  EXPECT_EQ(proxy->doneSeqsDev, done);
  EXPECT_EQ(proxy->readySeqsDev, ready);
  EXPECT_EQ(unmaps, mapped);
  EXPECT_EQ(doneFrees + readyFrees, 0);
  EXPECT_EQ(drains, 0);
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  EXPECT_EQ(unmaps, mapped);
  EXPECT_EQ(destroy(), flagcxSuccess);
  EXPECT_EQ(owner, nullptr);
  EXPECT_EQ(doneFrees, done != nullptr ? 1 : 0);
  EXPECT_EQ(readyFrees, ready != nullptr ? 1 : 0);
  EXPECT_EQ(unmaps, mapped);
}

INSTANTIATE_TEST_SUITE_P(AllocationAndMapping, RmaOptionalSyncResources,
                         ::testing::Values(0u, 1u, 2u, 3u, 5u, 7u, 10u, 11u,
                                           15u, 31u));

TEST_F(RmaOptionalSyncTest, PartialMappingFailuresKeepRemainingOwnersForRetry) {
  resources = 7; // Both allocations, one mapping and one host fallback.
  ASSERT_NO_FATAL_FAILURE(start());
  auto *proxy = owner->rmaProxy;
  auto *doneCpu = proxy->doneSeqsCpu;
  auto *readyCpu = proxy->readySeqsCpu;
  unmapFailures = 1;
  EXPECT_EQ(flagcxHeteroCommFinalize(owner), flagcxSystemError);
  EXPECT_FALSE(proxy->stopComplete);
  EXPECT_EQ(proxy->worker, nullptr);
  EXPECT_EQ(proxy->doneSeqsCpu, doneCpu);
  EXPECT_EQ(proxy->readySeqsCpu, readyCpu);
  EXPECT_TRUE(proxy->doneSeqsCpuMapped);
  EXPECT_FALSE(proxy->readySeqsCpuMapped);
  EXPECT_EQ(doneFrees + readyFrees, 0);
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  EXPECT_EQ(proxy->doneSeqsCpu, nullptr);
  EXPECT_EQ(proxy->readySeqsCpu, nullptr);
  EXPECT_EQ(unmaps, 2);
  freeFailures = 1;
  EXPECT_EQ(destroy(), flagcxUnhandledDeviceError);
  ASSERT_NE(owner, nullptr);
  EXPECT_EQ(proxy->doneSeqsDev, nullptr);
  EXPECT_NE(proxy->readySeqsDev, nullptr);
  EXPECT_EQ(doneFrees, 1);
  EXPECT_EQ(readyFrees, 1);
  EXPECT_EQ(destroy(), flagcxSuccess);
  EXPECT_EQ(owner, nullptr);
  EXPECT_EQ(unmaps, 2);
  EXPECT_EQ(doneFrees, 1);
  EXPECT_EQ(readyFrees, 2);
}
