#include "rma_init_rollback.hpp"

TEST_F(RmaInitRollbackTest, StartupAndUnmapFailuresRetainOwnerUntilRetry) {
  makeOwner();
  unmapFailures = 2; // Start's internal cleanup and InitRank's rollback.
  EXPECT_EQ(flagcxHeteroCommStartRma(&owner), flagcxSystemError);
  ASSERT_NE(owner, nullptr);
  ASSERT_NE(owner->rmaProxy, nullptr);
  EXPECT_NE(owner->rmaProxy->doneSeqsCpu, nullptr);
  EXPECT_FALSE(owner->destroyProxyComplete);
  EXPECT_EQ(doneFrees + readyFrees, 0);
  EXPECT_EQ(flagcxHeteroCommRollbackInit(&owner), flagcxSuccess);
  EXPECT_EQ(owner, nullptr);
  EXPECT_EQ(unmaps, 4);
  EXPECT_EQ(doneFrees, 1);
  EXPECT_EQ(readyFrees, 1);
  EXPECT_EQ(flagcxHeteroCommRollbackInit(&owner), flagcxSuccess);
  EXPECT_EQ(unmaps, 4);
}

TEST_F(RmaInitRollbackTest, StartupAndFreeFailuresDoNotRepeatSuccessfulPrefix) {
  makeOwner();
  freeFailures = 1;
  EXPECT_EQ(flagcxHeteroCommStartRma(&owner), flagcxUnhandledDeviceError);
  ASSERT_NE(owner, nullptr);
  ASSERT_NE(owner->rmaProxy, nullptr);
  EXPECT_EQ(owner->rmaProxy->doneSeqsCpu, nullptr);
  EXPECT_EQ(owner->rmaProxy->readySeqsCpu, nullptr);
  EXPECT_EQ(owner->rmaProxy->doneSeqsDev, nullptr);
  EXPECT_NE(owner->rmaProxy->readySeqsDev, nullptr);
  EXPECT_EQ(doneFrees, 1);
  EXPECT_EQ(flagcxHeteroCommRollbackInit(&owner), flagcxSuccess);
  EXPECT_EQ(owner, nullptr);
  EXPECT_EQ(unmaps, 2);
  EXPECT_EQ(doneFrees, 1);
  EXPECT_EQ(readyFrees, 2);
}

TEST_F(RmaInitRollbackTest, SuccessfulRollbackReturnsOriginalStartupError) {
  makeOwner();
  EXPECT_EQ(flagcxHeteroCommStartRma(&owner), flagcxInvalidUsage);
  EXPECT_EQ(owner, nullptr);
  EXPECT_EQ(unmaps, 2);
  EXPECT_EQ(doneFrees, 1);
  EXPECT_EQ(readyFrees, 1);
}

TEST_F(RmaInitRollbackTest,
       LaterDestroyFailureKeepsOutputUntilReleaseSucceeds) {
  makeOwner();
  attachFallibleTransport();
  releaseFailures = 1;
  EXPECT_EQ(flagcxHeteroCommStartRma(&owner), flagcxRemoteError);
  ASSERT_NE(owner, nullptr);
  EXPECT_EQ(owner->rmaProxy, nullptr);
  EXPECT_TRUE(owner->destroyProxyComplete);
  EXPECT_EQ(closes, 1);
  EXPECT_EQ(releases, 1);
  EXPECT_EQ(flagcxHeteroCommRollbackInit(&owner), flagcxSuccess);
  EXPECT_EQ(owner, nullptr);
  EXPECT_EQ(closes, 1);
  EXPECT_EQ(releases, 2);
  EXPECT_EQ(doneFrees, 1);
  EXPECT_EQ(readyFrees, 1);
}

TEST_F(RmaInitRollbackTest, DeviceSelectionFailureDoesNotTouchRetainedOwner) {
  makeOwner();
  unmapFailures = 2;
  ASSERT_EQ(flagcxHeteroCommStartRma(&owner), flagcxSystemError);
  failDevice = true;
  EXPECT_EQ(flagcxHeteroCommRollbackInit(&owner), flagcxRemoteError);
  EXPECT_NE(owner, nullptr);
  EXPECT_EQ(unmaps, 2);
  failDevice = false;
  EXPECT_EQ(flagcxHeteroCommRollbackInit(&owner), flagcxSuccess);
  EXPECT_EQ(owner, nullptr);
}
