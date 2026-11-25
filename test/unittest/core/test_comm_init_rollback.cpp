#include "rma_init_rollback.hpp"
namespace {
class CommInitRollback : public RmaInitRollbackTest {};
} // namespace
TEST_F(CommInitRollback, RollbackRetainsOuterUntilNativeOwnerSucceeds) {
  makeOwner();
  unmapFailures = 2;
  ASSERT_EQ(flagcxHeteroCommStartRma(&owner), flagcxSystemError);
  ASSERT_NE(owner, nullptr);
  transferToOuter();
  nativeFailures = 1;
  EXPECT_EQ(flagcxCommFinalize(outer), flagcxInvalidUsage);
  EXPECT_EQ(flagcxAllReduce(nullptr, nullptr, 0, flagcxChar, flagcxSum, outer,
                            nullptr),
            flagcxInvalidUsage);
  EXPECT_EQ(flagcxCommDestroy(outer), flagcxRemoteError);
  EXPECT_EQ(outer->heteroComm, nullptr);
  EXPECT_NE(outer->homoComm, nullptr);
  EXPECT_EQ(nativeFinalizes, 0);
  EXPECT_EQ(doneFrees, 1);
  EXPECT_EQ(readyFrees, 1);
  EXPECT_EQ(flagcxCommDestroy(outer), flagcxSuccess);
  outer = nullptr;
  EXPECT_EQ(nativeDestroys, 2);
  EXPECT_EQ(doneFrees + readyFrees, 2);
}
TEST_F(CommInitRollback, CachedProxyErrorDoesNotLeaveADanglingInnerOwner) {
  makeOwner();
  owner->proxyState->workerCleanupResult = flagcxRemoteError;
  transferToOuter();
  EXPECT_EQ(flagcxCommDestroy(outer), flagcxRemoteError);
  EXPECT_EQ(outer->heteroComm, nullptr);
  EXPECT_EQ(nativeDestroys, 0);
  EXPECT_EQ(flagcxCommDestroy(outer), flagcxSuccess);
  outer = nullptr;
  EXPECT_EQ(nativeDestroys, 1);
  EXPECT_EQ(nativeFinalizes, 0);
}
