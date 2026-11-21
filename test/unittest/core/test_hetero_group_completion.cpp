#include "hetero_group_completion.hpp"

namespace {
class HeteroGroupBoundary : public HeteroGroupCompletionTest {};
} // namespace

TEST_F(HeteroGroupBoundary, HostCompletionCoversEveryCommunicatorStream) {
  checkMode(0);
}
TEST_F(HeteroGroupBoundary,
       StreamValueCompletionCoversEveryCommunicatorStream) {
  checkMode(2);
}
TEST_F(HeteroGroupBoundary, DeviceCompletionCoversEveryCommunicatorStream) {
  // The device semaphore uses a process-lifetime pinned pool. Isolate the
  // fake allocator so the production static destructor never sees fake memory.
  EXPECT_EXIT(
      {
        checkMode(1);
        TearDown();
        _exit(HasFailure() ? 1 : 0);
      },
      ::testing::ExitedWithCode(0), "");
}
TEST_F(HeteroGroupBoundary, FailedWaitCannotPublishAnOriginalStreamFence) {
  ASSERT_EQ(flagcxHeteroGroupStart(), flagcxSuccess);
  ASSERT_NO_FATAL_FAILURE(submit(&a));
  ASSERT_NO_FATAL_FAILURE(submit(&b));
  completionResult = flagcxInProgress;
  EXPECT_EQ(flagcxGroupEndInternal(), flagcxInProgress);
  EXPECT_EQ(a.record->state, flagcxWorkUncertain);
  EXPECT_EQ(b.record->state, flagcxWorkUncertain);
  EXPECT_EQ(flagcxWorkWait(&a.comm.submittedWork), flagcxInProgress);
  EXPECT_EQ(flagcxWorkRelease(&a.comm.submittedWork), flagcxInvalidUsage);
}
TEST_F(HeteroGroupBoundary, PartialProxyPublishFailureRetainsEveryOwner) {
  ASSERT_EQ(flagcxHeteroGroupStart(), flagcxSuccess);
  ASSERT_NO_FATAL_FAILURE(submit(&a));
  ASSERT_NO_FATAL_FAILURE(submit(&b));
  a.proxy.asyncResult = flagcxRemoteError;
  EXPECT_EQ(flagcxGroupEndInternal(), flagcxRemoteError);
  EXPECT_NE(flagcxIntruQueueHead(&b.proxy.proxyOps[0].prodPeers.sendQueue),
            nullptr);
  EXPECT_EQ(completionCalls, 0);
  EXPECT_EQ(a.record->state, flagcxWorkUncertain);
  EXPECT_EQ(b.record->state, flagcxWorkUncertain);
  EXPECT_EQ(flagcxWorkWait(&a.comm.submittedWork), flagcxRemoteError);
}
TEST_F(HeteroGroupBoundary, CompletionStreamCaptureCannotGrantRelease) {
  flagcxWorkGroupBoundary boundary = {(flagcxStream_t)&b.stream, 1, 1};
  ASSERT_EQ(flagcxHeteroGroupStart(), flagcxSuccess);
  ASSERT_NO_FATAL_FAILURE(submit(&a));
  b.stream.capture = flagcxStreamCaptureActive;
  EXPECT_EQ(flagcxWorkGroupCompleteAtBoundary(flagcxWorkHeteroGroup,
                                              flagcxSuccess, &boundary),
            flagcxInvalidUsage);
  EXPECT_EQ(a.record->state, flagcxWorkUncertain);
  b.stream.capture = flagcxStreamCaptureNone;
}

TEST_F(HeteroGroupBoundary, MarkerRecordFailureCannotCompleteItsOwner) {
  failNextMarker = true;
  ASSERT_EQ(flagcxHeteroGroupStart(), flagcxSuccess);
  ASSERT_NO_FATAL_FAILURE(submit(&a));
  ASSERT_NO_FATAL_FAILURE(submit(&b));
  EXPECT_EQ(flagcxGroupEndInternal(), flagcxRemoteError);
  EXPECT_EQ(b.record->state, flagcxWorkUncertain);
  EXPECT_EQ(a.record->state, flagcxWorkRecorded);
  EXPECT_EQ(flagcxWorkWait(&b.comm.submittedWork), flagcxRemoteError);
  EXPECT_EQ(flagcxWorkWait(&a.comm.submittedWork), flagcxInProgress);
}
TEST_F(HeteroGroupBoundary, PendingStreamValueWaitCannotGrantRelease) {
  ASSERT_NO_FATAL_FAILURE(selectMode(2));
  ASSERT_EQ(flagcxHeteroGroupStart(), flagcxSuccess);
  ASSERT_NO_FATAL_FAILURE(submit(&a));
  ASSERT_NO_FATAL_FAILURE(submit(&b));
  completionResult = flagcxInProgress;
  EXPECT_EQ(flagcxGroupEndInternal(true), flagcxInProgress);
  EXPECT_EQ(a.record->state, flagcxWorkUncertain);
  EXPECT_EQ(b.record->state, flagcxWorkUncertain);
}

TEST_F(HeteroGroupBoundary, ReadinessEventCreateFailureSubmitsNoProxyWork) {
  ASSERT_EQ(flagcxHeteroGroupStart(), flagcxSuccess);
  ASSERT_NO_FATAL_FAILURE(submit(&a));
  ASSERT_NO_FATAL_FAILURE(submit(&b));
  createFailures = 1;
  EXPECT_EQ(flagcxGroupEndInternal(), flagcxSystemError);
  EXPECT_EQ(completionCalls, 0);
  EXPECT_EQ(flagcxIntruQueueHead(&a.proxy.proxyOps[0].prodPeers.sendQueue),
            nullptr);
  EXPECT_EQ(flagcxIntruQueueHead(&b.proxy.proxyOps[0].prodPeers.sendQueue),
            nullptr);
  EXPECT_EQ(flagcxIntruQueueHead(&a.task.sendQueue), nullptr);
  EXPECT_EQ(flagcxIntruQueueHead(&b.task.recvQueue), nullptr);
  EXPECT_EQ(flagcxWorkWait(&a.comm.submittedWork), flagcxSuccess);
}
