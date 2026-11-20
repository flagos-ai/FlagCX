#include "work_completion.hpp"

namespace {
class WorkCancellationTest : public WorkCompletionTest {};
} // namespace

TEST_F(WorkCancellationTest,
       CancelUnsubmittedRecordDoesNotRecordOrWaitForStream) {
  auto *record = prepare();
  stream.submitted =
      2; // Independent work must not become a cancellation fence.
  EXPECT_EQ(flagcxWorkCancel(record), flagcxSuccess);
  EXPECT_EQ(flagcxWorkCancel(nullptr), flagcxSuccess);
  EXPECT_EQ(flagcxWorkCancel(record), flagcxInvalidUsage);
  EXPECT_EQ(flagcxWorkWait(&tracker), flagcxSuccess);
  EXPECT_EQ(recorded + queried + waited, 0);
  EXPECT_EQ(stream.completed, 0u);
  EXPECT_EQ(flagcxWorkRelease(&tracker), flagcxSuccess);
  EXPECT_EQ(destroyed, 1);
}

TEST_F(WorkCancellationTest, CanceledEventsStayOwnedInsteadOfBeingRecycled) {
  auto *canceled = prepare();
  ASSERT_EQ(flagcxWorkCancel(canceled), flagcxSuccess);
  auto *next = prepare();
  EXPECT_NE(next, canceled);
  EXPECT_EQ(created, 2);
  EXPECT_EQ(queried, 0);
  ASSERT_EQ(flagcxWorkCancel(next), flagcxSuccess);
  EXPECT_EQ(flagcxWorkWait(&tracker), flagcxSuccess);
  EXPECT_EQ(flagcxWorkRelease(&tracker), flagcxSuccess);
  EXPECT_EQ(destroyed, 2);
}

TEST_F(WorkCancellationTest, SubmittedOrUncertainRecordsCannotBeCanceled) {
  auto *record = prepare();
  ASSERT_EQ(finish(record, &stream, flagcxSuccess, 1), flagcxSuccess);
  EXPECT_EQ(flagcxWorkCancel(record), flagcxInvalidUsage);
  EXPECT_EQ(record->state, flagcxWorkGroupPending);
  ASSERT_EQ(flagcxWorkGroupComplete(flagcxWorkNativeGroup, flagcxSuccess),
            flagcxSuccess);
  EXPECT_EQ(flagcxWorkCancel(record), flagcxInvalidUsage);
  EXPECT_EQ(record->state, flagcxWorkRecorded);
  record = prepare();
  recordFailures = 1;
  EXPECT_EQ(finish(record), flagcxRemoteError);
  EXPECT_EQ(flagcxWorkCancel(record), flagcxInvalidUsage);
  EXPECT_EQ(record->state, flagcxWorkUncertain);
}

TEST_F(WorkCancellationTest,
       CanceledEventReleaseRetainsOwnerOnErrorAndProgress) {
  auto *record = prepare();
  auto event = record->event;
  ASSERT_EQ(flagcxWorkCancel(record), flagcxSuccess);
  ASSERT_EQ(flagcxWorkWait(&tracker), flagcxSuccess);
  auto destroy = device.eventDestroy;
  device.eventDestroy = [](flagcxEvent_t) { return flagcxInProgress; };
  EXPECT_EQ(flagcxWorkRelease(&tracker), flagcxInProgress);
  EXPECT_EQ(tracker.head, record);
  EXPECT_EQ(record->event, event);
  device.eventDestroy = destroy;
  destroyFailures = 1;
  EXPECT_EQ(flagcxWorkRelease(&tracker), flagcxSystemError);
  EXPECT_EQ(tracker.head, record);
  EXPECT_EQ(record->event, event);
  EXPECT_EQ(flagcxWorkRelease(&tracker), flagcxSuccess);
  EXPECT_EQ(destroyed, 1);
  EXPECT_EQ(recorded + queried + waited, 0);
}
