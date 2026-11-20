#include "rma_init_rollback.hpp"

namespace {
class CommInitRollbackTest : public RmaInitRollbackTest {};
} // namespace

TEST_F(CommInitRollbackTest, OuterDestroyRetriesRmaWithoutFinalizeOrBootstrap) {
  makeOwner();
  unmapFailures = 3;
  ASSERT_EQ(flagcxHeteroCommStartRma(&owner), flagcxSystemError);
  transferToOuter();
  EXPECT_EQ(flagcxCommFinalize(outer), flagcxInvalidUsage);
  EXPECT_EQ(flagcxCommRequireActive(outer), flagcxInvalidUsage);
  EXPECT_EQ(flagcxCommDestroy(outer), flagcxSystemError);
  EXPECT_NE(outer->heteroComm, nullptr);
  EXPECT_NE(outer->heteroComm->rmaProxy, nullptr);
  EXPECT_EQ(outer->destroyPhase, flagcxDestroyBuffers);
  EXPECT_EQ(outer->finalizeState.commonPhase, flagcxFinalizeDrain);
  EXPECT_EQ(nativeDestroys, 0);
  EXPECT_EQ(flagcxCommDestroy(outer), flagcxSuccess);
  outer = nullptr;
  EXPECT_EQ(nativeFinalizes, 0);
  EXPECT_EQ(drains, 0);
  EXPECT_EQ(nativeDestroys, 1);
  EXPECT_EQ(doneFrees, 1);
  EXPECT_EQ(readyFrees, 1);
}

TEST_F(CommInitRollbackTest,
       OuterNativeDestroyFailureRetainsLocalReleaseCursor) {
  makeOwner();
  freeFailures = 1;
  ASSERT_EQ(flagcxHeteroCommStartRma(&owner), flagcxUnhandledDeviceError);
  transferToOuter();
  nativeFailures = 1;
  EXPECT_EQ(flagcxCommDestroy(outer), flagcxRemoteError);
  EXPECT_EQ(outer->heteroComm, nullptr);
  EXPECT_NE(outer->homoComm, nullptr);
  EXPECT_EQ(outer->destroyPhase, flagcxDestroyVendors);
  EXPECT_TRUE(outer->initRollbackPending);
  EXPECT_EQ(flagcxCommDestroy(outer), flagcxSuccess);
  outer = nullptr;
  EXPECT_EQ(nativeDestroys, 2);
  EXPECT_EQ(nativeFinalizes, 0);
  EXPECT_EQ(drains, 0);
  EXPECT_EQ(doneFrees, 1);
  EXPECT_EQ(readyFrees, 2);
}

TEST_F(CommInitRollbackTest,
       OuterRollbackWorksAfterInnerAlreadyFreedOnFailure) {
  makeOwner();
  ASSERT_EQ(flagcxHeteroCommStartRma(&owner), flagcxInvalidUsage);
  ASSERT_EQ(owner, nullptr);
  transferToOuter();
  EXPECT_EQ(flagcxCommDestroy(outer), flagcxSuccess);
  outer = nullptr;
  EXPECT_EQ(nativeDestroys, 1);
  EXPECT_EQ(nativeFinalizes, 0);
  EXPECT_EQ(drains, 0);
}

TEST_F(CommInitRollbackTest, PendingRollbackCannotEnterHeteroFinalize) {
  makeOwner();
  unmapFailures = 2;
  ASSERT_EQ(flagcxHeteroCommStartRma(&owner), flagcxSystemError);
  EXPECT_TRUE(owner->finalizeState.rollbackPending);
  EXPECT_EQ(flagcxHeteroCommFinalize(owner), flagcxInvalidUsage);
  EXPECT_EQ(drains, 0);
}

TEST_F(CommInitRollbackTest,
       RmaFreeInProgressRetainsInnerAndOuterRollbackOwners) {
  makeOwner();
  freeFailures = 2;
  freeFailureResult = flagcxInProgress;
  ASSERT_EQ(flagcxHeteroCommStartRma(&owner), flagcxInProgress);
  ASSERT_NE(owner, nullptr);
  transferToOuter();
  auto *inner = outer->heteroComm;
  EXPECT_EQ(flagcxCommDestroy(outer), flagcxInProgress);
  ASSERT_EQ(outer->heteroComm, inner);
  ASSERT_NE(inner->rmaProxy, nullptr);
  EXPECT_TRUE(inner->rmaProxy->stopComplete);
  EXPECT_EQ(inner->rmaProxy->doneSeqsDev, nullptr);
  EXPECT_NE(inner->rmaProxy->readySeqsDev, nullptr);
  EXPECT_EQ(outer->destroyPhase, flagcxDestroyBuffers);
  EXPECT_EQ(nativeDestroys + nativeFinalizes, 0);
  EXPECT_EQ(flagcxCommDestroy(outer), flagcxSuccess);
  outer = nullptr;
  EXPECT_EQ(doneFrees, 1);
  EXPECT_EQ(readyFrees, 3);
  EXPECT_EQ(nativeDestroys, 1);
  EXPECT_EQ(unmaps, 2);
  EXPECT_EQ(drains + nativeFinalizes, 0);
}
