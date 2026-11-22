#include "group.h"
#include "launch_kernel.h"
#include "proxy.h"
#include "work_completion.hpp"

namespace {
class WorkGroupBoundary : public WorkCompletionTest {
protected:
  flagcxLaunchFunc_t savedKernel;
  int launchedJobs = 0;
  static flagcxResult_t failJob(flagcxAsyncJob *) {
    static_cast<WorkGroupBoundary *>(current)->launchedJobs++;
    return flagcxInternalError;
  }
  void SetUp() override {
    WorkCompletionTest::SetUp();
    savedKernel = deviceAsyncKernel;
    deviceAsyncKernel = nullptr;
    // These host boundary tests do not submit stream-value work. Do not
    // inherit real SDK probe callbacks from the platform adaptor.
    device.streamWaitValue64 = nullptr;
    device.streamWriteValue64 = nullptr;
    ASSERT_EQ(flagcxGetGroupDepth(), 0);
  }
  void TearDown() override {
    while (flagcxGetGroupDepth() > 0)
      flagcxGroupEndInternal();
    EXPECT_EQ(flagcxWorkGroupComplete(flagcxWorkHeteroGroup, flagcxSuccess),
              flagcxSuccess);
    deviceAsyncKernel = savedKernel;
    WorkCompletionTest::TearDown();
  }
  flagcxWorkRecord *pending() {
    auto *record = prepare();
    EXPECT_EQ(flagcxWorkFinish(record, (flagcxStream_t)&stream, flagcxSuccess,
                               flagcxWorkHeteroGroup, 1),
              flagcxSuccess);
    return record;
  }
};
} // namespace

TEST_F(WorkGroupBoundary, InternalOutermostEndRecordsOnlyOnce) {
  ASSERT_EQ(flagcxHeteroGroupStart(), flagcxSuccess);
  ASSERT_EQ(flagcxHeteroGroupStart(), flagcxSuccess);
  auto *record = pending();
  stream.submitted = 1;
  ASSERT_EQ(flagcxGroupEndInternal(true), flagcxSuccess);
  EXPECT_EQ(recorded, 0);
  EXPECT_EQ(record->state, flagcxWorkGroupPending);
  ASSERT_EQ(flagcxGroupEndInternal(true), flagcxSuccess);
  EXPECT_EQ(recorded, 1);
  EXPECT_EQ(record->state, flagcxWorkRecorded);
  EXPECT_EQ(flagcxWorkWait(&tracker), flagcxInProgress);
  stream.completed = 1;
  EXPECT_EQ(flagcxWorkWait(&tracker), flagcxSuccess);
  ASSERT_EQ(flagcxHeteroGroupStart(), flagcxSuccess);
  ASSERT_EQ(flagcxHeteroGroupEnd(), flagcxSuccess);
  EXPECT_EQ(recorded, 1);
}

TEST_F(WorkGroupBoundary, EventRecordFailureClearsPendingButKeepsOwner) {
  ASSERT_EQ(flagcxHeteroGroupStart(), flagcxSuccess);
  auto *record = pending();
  recordFailures = 1;
  EXPECT_EQ(flagcxGroupEndInternal(true), flagcxRemoteError);
  EXPECT_EQ(record->state, flagcxWorkUncertain);
  EXPECT_EQ(flagcxWorkWait(&tracker), flagcxRemoteError);
  ASSERT_EQ(flagcxHeteroGroupStart(), flagcxSuccess);
  EXPECT_EQ(flagcxHeteroGroupEnd(), flagcxSuccess);
  EXPECT_EQ(recorded, 1);
}

TEST_F(WorkGroupBoundary, FailedLaunchPreservesBoundaryForSafeCleanup) {
  flagcxHeteroComm comm = {};
  flagcxProxyState proxy = {};
  uint64_t connectSend = 0, connectRecv = 0;
  volatile uint32_t abort = 0;
  comm.nRanks = comm.localRanks = 1;
  comm.proxyState = &proxy;
  comm.connectSend = &connectSend;
  comm.connectRecv = &connectRecv;
  comm.abortFlag = &abort;
  proxy.initialized = 1;
  auto *job = (flagcxAsyncJob *)calloc(1, sizeof(flagcxAsyncJob));
  ASSERT_NE(job, nullptr);
  job->func = failJob;
  job->destructor = free;
  job->comm = &comm;
  ASSERT_EQ(flagcxHeteroGroupStart(), flagcxSuccess);
  auto *record = pending();
  flagcxIntruQueueEnqueue(&flagcxAsyncJobs, job);
  flagcxGroupCommPreconnectHead = &comm;
  EXPECT_EQ(flagcxGroupEndInternal(true), flagcxInternalError);
  EXPECT_EQ(launchedJobs, 1);
  EXPECT_EQ(deviceQueries, 0);
  EXPECT_EQ(record->state, flagcxWorkRecorded);
  EXPECT_EQ(flagcxWorkWait(&tracker), flagcxSuccess);
  EXPECT_EQ(recorded, 1);
  ASSERT_EQ(flagcxHeteroGroupStart(), flagcxSuccess);
  EXPECT_EQ(flagcxHeteroGroupEnd(), flagcxSuccess);
  EXPECT_EQ(recorded, 1);
}
