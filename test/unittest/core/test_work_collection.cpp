#include "adaptor.h"
#include "work.h"
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <thread>

namespace {
struct CollectionEvent {
  bool ready = false;
  flagcxResult_t result = flagcxSuccess;
};

class WorkCollection : public ::testing::Test {
protected:
  inline static WorkCollection *active;
  flagcxDeviceAdaptor *saved;
  flagcxDeviceAdaptor device{};
  flagcxWorkTracker tracker{};
  std::atomic<int> creates{0}, destroys{0}, queries{0};
  std::atomic<bool> entered{false}, resume{false};
  CollectionEvent *blocking = nullptr;
  int failDestroy = 0;
  flagcxStream_t slow = (flagcxStream_t)1, fast = (flagcxStream_t)2;

  void SetUp() override {
    active = this;
    saved = deviceAdaptor;
    device.eventCreate = [](flagcxEvent_t *out, flagcxEventType_t) {
      *out = (flagcxEvent_t) new CollectionEvent;
      ++active->creates;
      return flagcxSuccess;
    };
    device.eventRecord = [](flagcxEvent_t event, flagcxStream_t stream) {
      ((CollectionEvent *)event)->ready = stream == active->fast;
      return flagcxSuccess;
    };
    device.eventQuery = device.eventSynchronize = [](flagcxEvent_t event) {
      ++active->queries;
      auto *e = (CollectionEvent *)event;
      if (e == active->blocking) {
        active->entered = true;
        while (!active->resume)
          std::this_thread::yield();
      }
      if (e->result != flagcxSuccess)
        return e->result;
      return e->ready ? flagcxSuccess : flagcxInProgress;
    };
    device.eventDestroy = [](flagcxEvent_t event) {
      if (active->failDestroy) {
        --active->failDestroy;
        return flagcxSystemError;
      }
      ++active->destroys;
      delete (CollectionEvent *)event;
      return flagcxSuccess;
    };
    deviceAdaptor = &device;
    flagcxWorkNativeGroupResult(flagcxSuccess);
  }
  flagcxWorkRecord *submit(flagcxStream_t stream) {
    flagcxWorkRecord *record = nullptr;
    EXPECT_EQ(flagcxWorkPrepare(&tracker, stream, &record), flagcxSuccess);
    if (record)
      EXPECT_EQ(flagcxWorkFinish(record, stream, flagcxSuccess,
                                 flagcxWorkNativeGroup, 0),
                flagcxSuccess);
    return record;
  }
  unsigned activeRecords() {
    unsigned count = 0;
    for (auto *r = tracker.head; r; r = r->next)
      ++count;
    return count;
  }
  bool waitForQuery() {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!entered && std::chrono::steady_clock::now() < deadline)
      std::this_thread::yield();
    return entered;
  }
  void TearDown() override {
    resume = true;
    for (auto *r = tracker.head; r; r = r->next) {
      if (r->event) {
        auto *e = (CollectionEvent *)r->event;
        e->ready = true;
        e->result = flagcxSuccess;
      }
      EXPECT_NE(r->state, flagcxWorkChecking);
      EXPECT_EQ(r->pins, 0);
    }
    failDestroy = 0;
    EXPECT_EQ(flagcxWorkWait(&tracker), flagcxSuccess);
    EXPECT_EQ(flagcxWorkRelease(&tracker), flagcxSuccess);
    EXPECT_EQ(tracker.collectAfter, 0u);
    EXPECT_EQ(creates.load(), destroys.load());
    deviceAdaptor = saved;
  }
};

TEST_F(WorkCollection, EightPendingRecordsDoNotStarveCompletedRecords) {
  for (int i = 0; i < 8; ++i)
    ASSERT_NE(submit(slow), nullptr);
  for (int i = 0; i < 2048; ++i) {
    int before = queries;
    ASSERT_NE(submit(fast), nullptr);
    EXPECT_LE(queries.load() - before, 8);
    ASSERT_LE(activeRecords(), 12u);
    ASSERT_LE(creates.load() - destroys.load(), 12);
  }
  // The eight original markers really are still pending; the test has not
  // completed old work to make room for newer records.
  int pending = 0;
  for (auto *r = tracker.head; r; r = r->next)
    pending += !((CollectionEvent *)r->event)->ready;
  EXPECT_EQ(pending, 8);
}

TEST_F(WorkCollection, DeletedCursorWrapsAndReusesCompletedEvents) {
  auto *old = submit(slow);
  ASSERT_NE(old, nullptr);
  auto *oldEvent = (CollectionEvent *)old->event;
  for (int i = 0; i < 64; ++i)
    ASSERT_NE(submit(fast), nullptr);
  oldEvent->ready = true;
  for (int i = 0; i < 16; ++i)
    ASSERT_NE(submit(fast), nullptr);
  EXPECT_LE(activeRecords(), 1u);
  EXPECT_LE(creates.load(), 4);
}

TEST_F(WorkCollection, SkipsPinnedAndCheckingRecordsThenRevisitsThem) {
  auto *pinned = submit(fast);
  ASSERT_NE(pinned, nullptr);
  flagcxWorkRetain(pinned);
  auto *checking = submit(fast);
  ASSERT_NE(checking, nullptr);
  checking->state = flagcxWorkChecking; // Another completion observer owns it.
  for (int i = 0; i < 64; ++i)
    ASSERT_NE(submit(fast), nullptr);
  EXPECT_EQ(pinned->pins, 1);
  EXPECT_EQ(checking->state, flagcxWorkChecking);
  EXPECT_LE(activeRecords(), 3u);
  flagcxWorkDrop(pinned);
  checking->state = flagcxWorkRecorded;
  for (int i = 0; i < 16; ++i)
    ASSERT_NE(submit(fast), nullptr);
  EXPECT_LE(activeRecords(), 1u);
}

TEST_F(WorkCollection, QueryErrorRetainsOwnerAndAdvancesToOtherRecords) {
  auto *bad = submit(slow);
  ASSERT_NE(bad, nullptr);
  auto *event = (CollectionEvent *)bad->event;
  ASSERT_NE(submit(slow), nullptr);
  tracker.collectAfter = 0;
  event->result = flagcxUnhandledDeviceError;
  flagcxWorkRecord *out = nullptr;
  EXPECT_EQ(flagcxWorkPrepare(&tracker, fast, &out),
            flagcxUnhandledDeviceError);
  EXPECT_EQ(out, nullptr);
  EXPECT_EQ(tracker.head, bad);
  EXPECT_EQ(bad->state, flagcxWorkRecorded);
  EXPECT_EQ(tracker.collectAfter, bad->id);
  EXPECT_EQ(tracker.collecting, 0);
  event->result = flagcxSuccess;
  for (int i = 0; i < 32; ++i)
    ASSERT_NE(submit(fast), nullptr);
  EXPECT_LE(activeRecords(), 3u);
}

TEST_F(WorkCollection, EventDestroyFailureRetainsItsOwnerForRetry) {
  auto *record = submit(fast);
  ASSERT_NE(record, nullptr);
  // A failed, physically completed operation cannot reuse its event.
  record->reusable = 0;
  failDestroy = 1;
  flagcxWorkRecord *out = nullptr;
  EXPECT_EQ(flagcxWorkPrepare(&tracker, fast, &out), flagcxSystemError);
  EXPECT_EQ(out, nullptr);
  EXPECT_EQ(tracker.retired, record);
  EXPECT_NE(record->event, nullptr);
  EXPECT_EQ(tracker.collecting, 0);
  ASSERT_NE(submit(fast), nullptr);
  EXPECT_EQ(tracker.retired, nullptr);
  EXPECT_EQ(destroys.load(), 1);
}

TEST_F(WorkCollection, ConcurrentSubmissionDoesNotRestartCollection) {
  auto *record = submit(slow);
  ASSERT_NE(record, nullptr);
  blocking = (CollectionEvent *)record->event;
  std::thread collector([&] {
    flagcxWorkRecord *out = nullptr;
    EXPECT_EQ(flagcxWorkPrepare(&tracker, fast, &out), flagcxSuccess);
    EXPECT_EQ(flagcxWorkCancel(out), flagcxSuccess);
  });
  if (!waitForQuery()) {
    resume = true;
    collector.join();
    FAIL() << "Collector did not reach the completion query";
  }
  EXPECT_EQ(record->state, flagcxWorkChecking);
  EXPECT_NE(submit(fast), nullptr);
  EXPECT_EQ(queries.load(), 1);
  resume = true;
  collector.join();
  EXPECT_EQ(tracker.collecting, 0);
}

TEST_F(WorkCollection, ReleaseWaitsForCollectorBeforeDestroyingTracker) {
  auto *record = submit(slow);
  ASSERT_NE(record, nullptr);
  blocking = (CollectionEvent *)record->event;
  std::thread collector([&] {
    flagcxWorkRecord *out = nullptr;
    EXPECT_EQ(flagcxWorkPrepare(&tracker, fast, &out), flagcxInvalidUsage);
    EXPECT_EQ(out, nullptr);
    if (out)
      flagcxWorkCancel(out);
  });
  if (!waitForQuery()) {
    resume = true;
    collector.join();
    FAIL() << "Collector did not reach the completion query";
  }
  EXPECT_EQ(flagcxWorkWait(&tracker), flagcxInProgress);
  EXPECT_EQ(flagcxWorkRelease(&tracker), flagcxInProgress);
  EXPECT_EQ(tracker.initialized, 2);
  resume = true;
  collector.join();
  EXPECT_EQ(tracker.collecting, 0);
}
} // namespace
