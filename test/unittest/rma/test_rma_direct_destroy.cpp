#include "rma_init_rollback.hpp"

namespace {
class RmaDirectDestroyTest : public RmaInitRollbackTest {
protected:
  void SetUp() override {
    RmaInitRollbackTest::SetUp();
    device.deviceSynchronize = [] { return flagcxSuccess; };
  }
  static flagcxResult_t orderedCleanup(flagcxProxyConnection *connection,
                                       int phase) {
    auto *fixture = static_cast<RmaDirectDestroyTest *>(current);
    if (phase == flagcxTransportCleanupCloseImports) {
      EXPECT_NE(fixture->owner->rmaProxy, nullptr);
      EXPECT_TRUE(fixture->owner->rmaProxy->stopComplete);
      EXPECT_EQ(fixture->owner->rmaProxy->worker, nullptr);
      EXPECT_EQ(fixture->doneFrees + fixture->readyFrees, 0);
      fixture->closes++;
    } else {
      EXPECT_EQ(fixture->owner->rmaProxy, nullptr);
      fixture->releases++;
      connection->transportResources = nullptr;
    }
    return flagcxSuccess;
  }
  flagcxResult_t destroy() {
    flagcxResult_t result = flagcxHeteroCommDestroy(owner);
    if (result == flagcxSuccess)
      owner = nullptr;
    return result;
  }
  void start() {
    makeOwner(false);
    ASSERT_EQ(flagcxHeteroCommStartRma(&owner), flagcxSuccess);
    ASSERT_NE(owner, nullptr);
    ASSERT_NE(owner->rmaProxy, nullptr);
    ASSERT_NE(owner->rmaProxy->worker, nullptr);
    attachFallibleTransport();
  }
};

TEST_F(RmaDirectDestroyTest, StopsLiveWorkerBeforeTransportShutdown) {
  ASSERT_NO_FATAL_FAILURE(start());
  ASSERT_NE(owner, nullptr);
  transport.cleanupProxyConnection = orderedCleanup;
  EXPECT_EQ(destroy(), flagcxSuccess);
  EXPECT_EQ(owner, nullptr);
  EXPECT_EQ(unmaps, 2);
  EXPECT_EQ(doneFrees, 1);
  EXPECT_EQ(readyFrees, 1);
  EXPECT_EQ(closes, 1);
  EXPECT_EQ(releases, 1);
}

TEST_F(RmaDirectDestroyTest, RetiredOperationErrorDoesNotLeakResources) {
  ASSERT_NO_FATAL_FAILURE(start());
  __atomic_store_n(&owner->rmaProxy->rmaError, 1, __ATOMIC_RELEASE);
  EXPECT_EQ(flagcxHeteroDrainProducers(owner), flagcxRemoteError);
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  ASSERT_NE(owner->rmaProxy, nullptr);
  EXPECT_EQ(owner->rmaProxy->rmaError, 1);
  EXPECT_TRUE(owner->rmaProxy->stopComplete);
  EXPECT_EQ(doneFrees + readyFrees, 0);
  EXPECT_EQ(destroy(), flagcxSuccess);
  EXPECT_EQ(unmaps, 2);
  EXPECT_EQ(doneFrees, 1);
  EXPECT_EQ(readyFrees, 1);
  EXPECT_EQ(closes, 1);
  EXPECT_EQ(releases, 1);
}

TEST_F(RmaDirectDestroyTest, UnmapFailurePreservesOwnerAndTransportForRetry) {
  ASSERT_NO_FATAL_FAILURE(start());
  unmapFailures = 1;
  EXPECT_EQ(destroy(), flagcxSystemError);
  ASSERT_NE(owner, nullptr);
  ASSERT_NE(owner->rmaProxy, nullptr);
  EXPECT_EQ(owner->rmaProxy->worker, nullptr);
  EXPECT_FALSE(owner->destroyProxyComplete);
  EXPECT_EQ(closes + releases, 0);
  EXPECT_EQ(doneFrees + readyFrees, 0);
  EXPECT_EQ(destroy(), flagcxSuccess);
  EXPECT_EQ(owner, nullptr);
  EXPECT_EQ(unmaps, 3);
  EXPECT_EQ(doneFrees, 1);
  EXPECT_EQ(readyFrees, 1);
  EXPECT_EQ(closes, 1);
  EXPECT_EQ(releases, 1);
}

TEST_F(RmaDirectDestroyTest, FreeFailureDoesNotRepeatReleasedPrefix) {
  ASSERT_NO_FATAL_FAILURE(start());
  freeFailures = 1;
  EXPECT_EQ(destroy(), flagcxUnhandledDeviceError);
  ASSERT_NE(owner, nullptr);
  ASSERT_NE(owner->rmaProxy, nullptr);
  EXPECT_EQ(owner->rmaProxy->worker, nullptr);
  EXPECT_EQ(owner->rmaProxy->doneSeqsDev, nullptr);
  EXPECT_NE(owner->rmaProxy->readySeqsDev, nullptr);
  EXPECT_EQ(doneFrees, 1);
  EXPECT_EQ(closes, 1);
  EXPECT_EQ(releases, 0);
  EXPECT_EQ(destroy(), flagcxSuccess);
  EXPECT_EQ(unmaps, 2);
  EXPECT_EQ(doneFrees, 1);
  EXPECT_EQ(readyFrees, 2);
}

TEST_F(RmaDirectDestroyTest, DeviceFailurePreservesResourcesUntilRetry) {
  ASSERT_NO_FATAL_FAILURE(start());
  failDevice = true;
  EXPECT_EQ(destroy(), flagcxRemoteError);
  ASSERT_NE(owner, nullptr);
  EXPECT_NE(owner->rmaProxy, nullptr);
  EXPECT_EQ(unmaps + doneFrees + readyFrees + closes + releases, 0);
  failDevice = false;
  EXPECT_EQ(destroy(), flagcxSuccess);
  EXPECT_EQ(unmaps, 2);
  EXPECT_EQ(doneFrees, 1);
  EXPECT_EQ(readyFrees, 1);
}

TEST_F(RmaDirectDestroyTest, PrepareAndDestroyAfterExplicitStopAreIdempotent) {
  ASSERT_NO_FATAL_FAILURE(start());
  ASSERT_EQ(flagcxHeteroRmaProxyStop(owner), flagcxSuccess);
  ASSERT_NE(owner->rmaProxy, nullptr);
  EXPECT_TRUE(owner->rmaProxy->stopComplete);
  EXPECT_EQ(doneFrees + readyFrees, 0);
  EXPECT_EQ(flagcxHeteroCommPrepareDestroy(owner), flagcxSuccess);
  EXPECT_EQ(flagcxHeteroCommPrepareDestroy(owner), flagcxSuccess);
  EXPECT_EQ(destroy(), flagcxSuccess);
  EXPECT_EQ(unmaps, 2);
  EXPECT_EQ(doneFrees, 1);
  EXPECT_EQ(readyFrees, 1);
  EXPECT_EQ(closes, 1);
  EXPECT_EQ(releases, 1);
}

TEST_F(RmaDirectDestroyTest,
       FinalizeRetainsBothSyncAllocationsAndRejectsRestart) {
  ASSERT_NO_FATAL_FAILURE(start());
  auto *proxy = owner->rmaProxy;
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  ASSERT_EQ(owner->rmaProxy, proxy);
  EXPECT_TRUE(proxy->stopComplete);
  EXPECT_EQ(proxy->worker, nullptr);
  EXPECT_EQ(proxy->doneSeqsCpu, nullptr);
  EXPECT_EQ(proxy->readySeqsCpu, nullptr);
  EXPECT_NE(proxy->doneSeqsDev, nullptr);
  EXPECT_NE(proxy->readySeqsDev, nullptr);
  EXPECT_EQ(doneFrees + readyFrees, 0);
  EXPECT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  EXPECT_EQ(flagcxHeteroRmaProxyStart(owner), flagcxInvalidUsage);
  EXPECT_EQ(flagcxHeteroCommStartRma(&owner), flagcxInvalidUsage);
  EXPECT_EQ(owner->rmaProxy, proxy);
  EXPECT_FALSE(owner->finalizeState.rollbackPending);
  EXPECT_EQ(destroy(), flagcxSuccess);
  EXPECT_EQ(unmaps, 2);
  EXPECT_EQ(doneFrees, 1);
  EXPECT_EQ(readyFrees, 1);
}

TEST_F(RmaDirectDestroyTest, LiveAndStoppedOwnersCannotBeOverwrittenByStart) {
  ASSERT_NO_FATAL_FAILURE(start());
  auto *proxy = owner->rmaProxy;
  EXPECT_EQ(flagcxHeteroRmaProxyStart(owner), flagcxInvalidUsage);
  EXPECT_EQ(flagcxHeteroCommStartRma(&owner), flagcxInvalidUsage);
  EXPECT_EQ(owner->rmaProxy, proxy);
  EXPECT_NE(proxy->worker, nullptr);
  ASSERT_EQ(flagcxHeteroRmaProxyStop(owner), flagcxSuccess);
  EXPECT_EQ(flagcxHeteroRmaProxyStart(owner), flagcxInvalidUsage);
  EXPECT_EQ(flagcxHeteroCommStartRma(&owner), flagcxInvalidUsage);
  EXPECT_EQ(owner->rmaProxy, proxy);
  EXPECT_EQ(allocations, 2);
}

TEST_F(RmaDirectDestroyTest,
       UnmapInProgressCannotCompleteStopOrCloseTransports) {
  ASSERT_NO_FATAL_FAILURE(start());
  unmapFailures = 1;
  unmapFailureResult = flagcxInProgress;
  EXPECT_EQ(flagcxHeteroCommFinalize(owner), flagcxInProgress);
  ASSERT_NE(owner->rmaProxy, nullptr);
  EXPECT_FALSE(owner->rmaProxy->stopComplete);
  EXPECT_NE(owner->rmaProxy->doneSeqsCpu, nullptr);
  EXPECT_EQ(owner->finalizeState.phase, flagcxHeteroFinalizeRelay);
  EXPECT_EQ(closes + releases + doneFrees + readyFrees, 0);
  EXPECT_EQ(flagcxHeteroRmaProxyRelease(owner), flagcxInvalidUsage);
  EXPECT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  EXPECT_EQ(doneFrees + readyFrees, 0);
  EXPECT_EQ(destroy(), flagcxSuccess);
  EXPECT_EQ(unmaps, 3);
}

TEST_F(RmaDirectDestroyTest, ReleaseInProgressIsALocalRetryWithRetainedOwner) {
  ASSERT_NO_FATAL_FAILURE(start());
  freeFailures = 1;
  freeFailureResult = flagcxInProgress;
  EXPECT_EQ(destroy(), flagcxInProgress);
  ASSERT_NE(owner, nullptr);
  EXPECT_EQ(owner->finalizeState.phase, flagcxHeteroFinalizeComplete);
  ASSERT_NE(owner->rmaProxy, nullptr);
  EXPECT_TRUE(owner->rmaProxy->stopComplete);
  EXPECT_EQ(owner->rmaProxy->doneSeqsDev, nullptr);
  EXPECT_NE(owner->rmaProxy->readySeqsDev, nullptr);
  EXPECT_EQ(closes, 1);
  EXPECT_EQ(releases, 0);
  EXPECT_EQ(destroy(), flagcxSuccess);
  EXPECT_EQ(unmaps, 2);
  EXPECT_EQ(doneFrees, 1);
  EXPECT_EQ(readyFrees, 2);
  EXPECT_EQ(closes, 1);
}
} // namespace
