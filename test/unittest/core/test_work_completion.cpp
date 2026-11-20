#include "work_completion.hpp"

TEST_F(WorkCompletionTest, WaitsOnlyForRecordedOperationNotLaterStreamWork) {
  auto *record = prepare();
  ASSERT_NE(record, nullptr);
  stream.submitted = 1;
  ASSERT_EQ(finish(record), flagcxSuccess);
  stream.submitted = 2; // Unrelated work queued after this operation.
  stream.completed = 1;
  otherStream.submitted = 1;
  EXPECT_EQ(flagcxWorkWait(&tracker), flagcxSuccess);
  EXPECT_EQ(waited, 1);
  EXPECT_EQ(flagcxWorkPrepare(&tracker, (flagcxStream_t)&stream, 1, &record),
            flagcxInvalidUsage);
}

TEST_F(WorkCompletionTest, PendingWorkAndWaitFailureRetainSuccessfulPrefixes) {
  stream.submitted = 1;
  auto *first = prepare();
  ASSERT_EQ(finish(first), flagcxSuccess);
  auto *second = prepare();
  otherStream.submitted = 1;
  ASSERT_EQ(finish(second, &otherStream), flagcxSuccess);
  stream.completed = 1;
  EXPECT_EQ(flagcxWorkWait(&tracker), flagcxInProgress);
  EXPECT_EQ(first->state, flagcxWorkComplete);
  waitFailures = 1;
  EXPECT_EQ(flagcxWorkWait(&tracker), flagcxUnhandledDeviceError);
  otherStream.completed = 1;
  EXPECT_EQ(flagcxWorkWait(&tracker), flagcxSuccess);
  EXPECT_EQ(waited,
            4); // The already-complete first record was not waited twice.
}

TEST_F(WorkCompletionTest, FailedRecordCannotBeRepairedAtFinalizeStreamTail) {
  auto *record = prepare();
  recordFailures = 1;
  EXPECT_EQ(finish(record), flagcxRemoteError);
  EXPECT_EQ(flagcxWorkWait(&tracker), flagcxRemoteError);
  EXPECT_EQ(recorded, 1);
  EXPECT_EQ(waited, 0);
  EXPECT_EQ(flagcxWorkRelease(&tracker), flagcxInvalidUsage);
}

TEST_F(WorkCompletionTest, PartialSubmissionErrorStillRecordsAcceptedPrefix) {
  auto *record = prepare();
  stream.submitted = 1;
  EXPECT_EQ(finish(record, &stream, flagcxSystemError), flagcxSystemError);
  EXPECT_EQ(record->state, flagcxWorkRecorded);
  EXPECT_EQ(flagcxWorkWait(&tracker), flagcxInProgress);
  stream.completed = 1;
  EXPECT_EQ(flagcxWorkWait(&tracker), flagcxSuccess);
}

TEST_F(WorkCompletionTest, CompletedEventsAreReusedBeforeAnotherSubmission) {
  for (int i = 0; i < 100; i++) {
    auto *record = prepare();
    ASSERT_NE(record, nullptr);
    stream.submitted++;
    ASSERT_EQ(finish(record), flagcxSuccess);
    stream.completed = stream.submitted;
  }
  EXPECT_EQ(created, 1);
  EXPECT_EQ(flagcxWorkWait(&tracker), flagcxSuccess);
  destroyFailures = 1;
  EXPECT_EQ(flagcxWorkRelease(&tracker), flagcxSystemError);
  EXPECT_NE(tracker.head, nullptr);
  EXPECT_EQ(flagcxWorkRelease(&tracker), flagcxSuccess);
  EXPECT_EQ(destroyed, 1);
}

TEST_F(WorkCompletionTest, CreateFailurePublishesNoWorkAndCanBeCleanedLocally) {
  flagcxWorkRecord *record = (flagcxWorkRecord *)1;
  createFailures = 1;
  EXPECT_EQ(flagcxWorkPrepare(&tracker, (flagcxStream_t)&stream, 1, &record),
            flagcxSystemError);
  EXPECT_EQ(record, nullptr);
  EXPECT_EQ(flagcxWorkWait(&tracker), flagcxSuccess);
  EXPECT_EQ(flagcxWorkRelease(&tracker), flagcxSuccess);
  EXPECT_EQ(created, 0);
}

TEST_F(WorkCompletionTest, PartialNativeCreateFailureLeavesNoInvalidEvent) {
  partialCreateFailures = 1;
  flagcxWorkRecord *record = nullptr;
  EXPECT_EQ(flagcxWorkPrepare(&tracker, (flagcxStream_t)&stream, 0, &record),
            flagcxSystemError);
  EXPECT_EQ(record, nullptr);
  ASSERT_NE(tracker.head, nullptr);
  EXPECT_EQ(tracker.head->event, nullptr);
  EXPECT_EQ(wrappersAllocated, 1);
  EXPECT_EQ(wrappersFreed, 1);
  record = prepare();
  ASSERT_NE(record, nullptr);
  stream.submitted = stream.completed = 1;
  EXPECT_EQ(finish(record), flagcxSuccess);
  EXPECT_EQ(flagcxWorkWait(&tracker), flagcxSuccess);
  EXPECT_EQ(flagcxWorkRelease(&tracker), flagcxSuccess);
  EXPECT_EQ(created, 1);
  EXPECT_EQ(destroyed, 1);
  EXPECT_EQ(wrappersAllocated, wrappersFreed);
}

TEST_F(WorkCompletionTest, FailedNativeGroupLeavesUncertainOwner) {
  auto *record = prepare();
  EXPECT_EQ(finish(record, &stream, flagcxSuccess, 1), flagcxSuccess);
  EXPECT_EQ(recorded, 0);
  EXPECT_EQ(flagcxWorkGroupComplete(flagcxWorkNativeGroup, flagcxRemoteError),
            flagcxRemoteError);
  EXPECT_EQ(flagcxWorkWait(&tracker), flagcxRemoteError);
  EXPECT_EQ(recorded, 0);
}

TEST_F(WorkCompletionTest, CapturedCallsNeverInsertOrQueryCompletionEvents) {
  stream.capture = flagcxStreamCaptureActive;
  for (int i = 0; i < 100; i++) {
    auto *record = prepare();
    ASSERT_NE(record, nullptr);
    ASSERT_EQ(finish(record), flagcxSuccess);
    EXPECT_EQ(record->state, flagcxWorkExternalGraph);
  }
  EXPECT_EQ(created, 0);
  EXPECT_EQ(recorded, 0);
  EXPECT_EQ(queried, 0);
  stream.capture = flagcxStreamCaptureNone;
  // Graph replay completion is a caller precondition, not inferred by these
  // records. A later ordinary operation gets a real completion event again.
  auto *ordinary = prepare();
  stream.submitted = stream.completed = 1;
  ASSERT_EQ(finish(ordinary), flagcxSuccess);
  EXPECT_EQ(created, 1);
  EXPECT_EQ(flagcxWorkWait(&tracker), flagcxSuccess);
  EXPECT_EQ(waited, 1);
}

TEST_F(WorkCompletionTest, CaptureDoesNotQueryEarlierOrdinaryWork) {
  auto *ordinary = prepare();
  stream.submitted = 1;
  ASSERT_EQ(finish(ordinary), flagcxSuccess);
  stream.capture = flagcxStreamCaptureActive;
  int previousQueries = queried;
  auto *captured = prepare();
  ASSERT_EQ(finish(captured), flagcxSuccess);
  EXPECT_EQ(queried, previousQueries);
  EXPECT_EQ(recorded, 1);
  stream.capture = flagcxStreamCaptureNone;
  EXPECT_EQ(flagcxWorkWait(&tracker), flagcxInProgress);
  stream.completed = 1;
  EXPECT_EQ(flagcxWorkWait(&tracker), flagcxSuccess);
  EXPECT_EQ(captured->state, flagcxWorkExternalGraph);
}

TEST_F(WorkCompletionTest, UnsupportedCaptureAndInvalidatedStateSubmitNothing) {
  flagcxWorkRecord *record = nullptr;
  stream.capture = flagcxStreamCaptureActive;
  EXPECT_EQ(flagcxWorkPrepare(&tracker, (flagcxStream_t)&stream, 0, &record),
            flagcxNotSupported);
  EXPECT_EQ(record, nullptr);
  EXPECT_EQ(tracker.initialized, 0);
  stream.capture = flagcxStreamCaptureInvalidated;
  EXPECT_EQ(flagcxWorkPrepare(&tracker, (flagcxStream_t)&stream, 1, &record),
            flagcxInvalidUsage);
  stream.capture = flagcxStreamCaptureNone;
  captureResult = flagcxRemoteError;
  EXPECT_EQ(flagcxWorkPrepare(&tracker, (flagcxStream_t)&stream, 1, &record),
            flagcxRemoteError);
  device.streamGetCaptureStatus = nullptr;
  EXPECT_EQ(flagcxWorkPrepare(&tracker, (flagcxStream_t)&stream, 1, &record),
            flagcxNotSupported);
  EXPECT_EQ(created + recorded + queried, 0);
}

TEST_F(WorkCompletionTest, DeferredCaptureTransitionKeepsUncertainOwner) {
  stream.capture = flagcxStreamCaptureActive;
  auto *record = prepare();
  ASSERT_EQ(finish(record, &stream, flagcxSuccess, 1), flagcxSuccess);
  stream.capture = flagcxStreamCaptureNone;
  EXPECT_EQ(flagcxWorkGroupComplete(flagcxWorkNativeGroup, flagcxSuccess),
            flagcxInvalidUsage);
  EXPECT_EQ(record->state, flagcxWorkUncertain);
  EXPECT_EQ(flagcxWorkWait(&tracker), flagcxInvalidUsage);
  EXPECT_EQ(recorded, 0);
  EXPECT_EQ(flagcxWorkGroupComplete(flagcxWorkNativeGroup, flagcxSuccess),
            flagcxSuccess);
}

