/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "worker_pool.h"

#include <gtest/gtest.h>
#include <pthread.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>

namespace {

struct Probe {
  std::atomic<int> initCalls{0};
  std::atomic<int> progressCalls{0};
  std::atomic<int> rmaProgressCalls{0};
  std::atomic<int> finishCalls{0};
  pthread_t initThread{};
  pthread_t progressThread{};
  pthread_t finishThread{};
  flagcxResult_t initResult = flagcxSuccess;
  flagcxResult_t progressResult = flagcxSuccess;
  int drainPasses = 0;
};

flagcxResult_t init(void *state) {
  auto *probe = static_cast<Probe *>(state);
  probe->initThread = pthread_self();
  probe->initCalls.fetch_add(1);
  return probe->initResult;
}

flagcxResult_t progress(void *state, bool stopping, bool *madeProgress,
                        bool *outstanding) {
  auto *probe = static_cast<Probe *>(state);
  probe->progressThread = pthread_self();
  probe->progressCalls.fetch_add(1);
  if (probe->progressResult != flagcxSuccess)
    return probe->progressResult;
  if (stopping && probe->drainPasses > 0) {
    --probe->drainPasses;
    *madeProgress = true;
    *outstanding = probe->drainPasses != 0;
  } else {
    *madeProgress = false;
    *outstanding = !stopping;
  }
  return flagcxSuccess;
}

void finish(void *state) {
  auto *probe = static_cast<Probe *>(state);
  probe->finishThread = pthread_self();
  probe->finishCalls.fetch_add(1);
}

flagcxResult_t rmaProgress(void *state, bool stopping, bool *madeProgress,
                           bool *outstanding) {
  auto *probe = static_cast<Probe *>(state);
  probe->rmaProgressCalls.fetch_add(1);
  return progress(state, stopping, madeProgress, outstanding);
}

const flagcxWorkerOps kProbeOps = {init, progress, finish};
const flagcxWorkerOps kRmaOps = {init, rmaProgress, finish};

struct Gate {
  std::mutex mutex;
  std::condition_variable cond;
  bool entered = false;
  bool released = false;

  void block() {
    std::unique_lock<std::mutex> lock(mutex);
    entered = true;
    cond.notify_all();
    cond.wait(lock, [this] { return released; });
  }

  bool waitUntilEntered() {
    std::unique_lock<std::mutex> lock(mutex);
    return cond.wait_for(lock, std::chrono::seconds(5),
                         [this] { return entered; });
  }

  void release() {
    std::lock_guard<std::mutex> lock(mutex);
    released = true;
    cond.notify_all();
  }
};

flagcxResult_t blockedInit(void *state) {
  static_cast<Gate *>(state)->block();
  return flagcxSystemError;
}

flagcxResult_t successfulBlockedInit(void *state) {
  static_cast<Gate *>(state)->block();
  return flagcxSuccess;
}

flagcxResult_t idleProgress(void *, bool stopping, bool *madeProgress,
                            bool *outstanding) {
  *madeProgress = false;
  *outstanding = !stopping;
  return flagcxSuccess;
}

flagcxResult_t blockedDrain(void *state, bool stopping, bool *madeProgress,
                            bool *outstanding) {
  if (stopping)
    static_cast<Gate *>(state)->block();
  *madeProgress = false;
  *outstanding = !stopping;
  return flagcxSuccess;
}

const flagcxWorkerOps kBlockedInitOps = {blockedInit, idleProgress, nullptr};
const flagcxWorkerOps kSuccessfulBlockedInitOps = {successfulBlockedInit,
                                                   idleProgress, nullptr};
const flagcxWorkerOps kBlockedDrainOps = {nullptr, blockedDrain, nullptr};

TEST(WorkerPool, EachRegistrationOwnsOneThreadAndDrainsOnStop) {
  flagcxWorkerPool pool;
  Probe coll;
  Probe rma;
  rma.drainPasses = 3;
  flagcxWorkerPool::Worker *collWorker = nullptr;
  flagcxWorkerPool::Worker *rmaWorker = nullptr;

  ASSERT_EQ(flagcxSuccess, pool.start(&kProbeOps, &coll, &collWorker));
  ASSERT_EQ(flagcxSuccess, pool.start(&kRmaOps, &rma, &rmaWorker));
  ASSERT_EQ(2u, pool.workerCount());
  ASSERT_NE(nullptr, collWorker);
  ASSERT_NE(nullptr, rmaWorker);
  EXPECT_EQ(flagcxInProgress, pool.status(collWorker));
  EXPECT_EQ(flagcxInProgress, pool.status(rmaWorker));

  EXPECT_EQ(flagcxSuccess, pool.stopAndJoin(collWorker));
  EXPECT_EQ(flagcxSuccess, pool.stopAndJoin(rmaWorker));
  EXPECT_EQ(flagcxSuccess, pool.status(collWorker));
  EXPECT_EQ(flagcxSuccess, pool.status(rmaWorker));
  EXPECT_EQ(flagcxSuccess, pool.joinAll());
  EXPECT_EQ(1, coll.initCalls.load());
  EXPECT_EQ(1, rma.initCalls.load());
  EXPECT_GE(coll.progressCalls.load(), 1);
  EXPECT_GE(rma.progressCalls.load(), 3);
  EXPECT_EQ(0, coll.rmaProgressCalls.load());
  EXPECT_GE(rma.rmaProgressCalls.load(), 3);
  EXPECT_EQ(0, rma.drainPasses);
  EXPECT_EQ(1, coll.finishCalls.load());
  EXPECT_EQ(1, rma.finishCalls.load());
  EXPECT_FALSE(pthread_equal(pthread_self(), coll.initThread));
  EXPECT_FALSE(pthread_equal(coll.initThread, rma.initThread));
  EXPECT_TRUE(pthread_equal(coll.initThread, coll.progressThread));
  EXPECT_TRUE(pthread_equal(coll.initThread, coll.finishThread));
  EXPECT_TRUE(pthread_equal(rma.initThread, rma.progressThread));
  EXPECT_TRUE(pthread_equal(rma.initThread, rma.finishThread));
  EXPECT_EQ(flagcxSuccess, pool.retire(&collWorker));
  EXPECT_EQ(nullptr, collWorker);
  EXPECT_EQ(1u, pool.workerCount());
  EXPECT_EQ(flagcxSuccess, pool.retire(&rmaWorker));
  EXPECT_EQ(0u, pool.workerCount());
}

TEST(WorkerPool, FailedInitializationRollsBackRegistration) {
  flagcxWorkerPool pool;
  Probe failed;
  failed.initResult = flagcxSystemError;
  flagcxWorkerPool::Worker *worker = nullptr;
  EXPECT_EQ(flagcxSystemError, pool.start(&kProbeOps, &failed, &worker));
  EXPECT_EQ(nullptr, worker);
  EXPECT_EQ(0u, pool.workerCount());
  EXPECT_EQ(0, failed.progressCalls.load());
  EXPECT_EQ(1, failed.finishCalls.load());
  EXPECT_TRUE(pthread_equal(failed.initThread, failed.finishThread));

  Probe recovered;
  ASSERT_EQ(flagcxSuccess, pool.start(&kProbeOps, &recovered, &worker));
  EXPECT_EQ(flagcxSuccess, pool.stopAndJoin(worker));
}

TEST(WorkerPool, ProgressFailureIsReturnedByJoin) {
  flagcxWorkerPool pool;
  Probe failed;
  failed.progressResult = flagcxRemoteError;
  flagcxWorkerPool::Worker *worker = nullptr;
  ASSERT_EQ(flagcxSuccess, pool.start(&kProbeOps, &failed, &worker));
  EXPECT_EQ(flagcxRemoteError, pool.join(worker));
  EXPECT_EQ(flagcxRemoteError, pool.status(worker));
  EXPECT_EQ(1, failed.finishCalls.load());
}

TEST(WorkerPool, RetiringOneWorkerKeepsOtherRegistrationsRunning) {
  flagcxWorkerPool pool;
  Probe first;
  Probe second;
  Probe replacement;
  flagcxWorkerPool::Worker *firstWorker = nullptr;
  flagcxWorkerPool::Worker *secondWorker = nullptr;
  flagcxWorkerPool::Worker *replacementWorker = nullptr;
  ASSERT_EQ(flagcxSuccess, pool.start(&kProbeOps, &first, &firstWorker));
  ASSERT_EQ(flagcxSuccess, pool.start(&kRmaOps, &second, &secondWorker));

  EXPECT_EQ(flagcxSuccess, pool.retire(&firstWorker));
  EXPECT_EQ(nullptr, firstWorker);
  EXPECT_EQ(flagcxInProgress, pool.status(secondWorker));
  ASSERT_EQ(flagcxSuccess,
            pool.start(&kProbeOps, &replacement, &replacementWorker));
  EXPECT_EQ(2u, pool.workerCount());
  EXPECT_EQ(flagcxSuccess, pool.retire(&secondWorker));
  EXPECT_EQ(flagcxSuccess, pool.retire(&replacementWorker));
  EXPECT_EQ(0u, pool.workerCount());
}

TEST(WorkerPool, DestructorStopsAllRegisteredWorkers) {
  Probe first;
  Probe second;
  {
    flagcxWorkerPool pool;
    flagcxWorkerPool::Worker *worker = nullptr;
    ASSERT_EQ(flagcxSuccess, pool.start(&kProbeOps, &first, &worker));
    ASSERT_EQ(flagcxSuccess, pool.start(&kProbeOps, &second, &worker));
  }
  EXPECT_EQ(1, first.finishCalls.load());
  EXPECT_EQ(1, second.finishCalls.load());
}

TEST(WorkerPool, BlockedInitializationDoesNotBlockOtherRegistrations) {
  flagcxWorkerPool pool;
  Gate gate;
  Probe other;
  flagcxWorkerPool::Worker *blocked = nullptr;
  flagcxWorkerPool::Worker *independent = nullptr;
  auto firstStart = std::async(std::launch::async, [&] {
    return pool.start(&kBlockedInitOps, &gate, &blocked);
  });
  if (!gate.waitUntilEntered()) {
    gate.release();
    FAIL() << "blocked worker did not enter initialization";
  }

  auto secondStart = std::async(std::launch::async, [&] {
    return pool.start(&kProbeOps, &other, &independent);
  });
  const bool secondReady = secondStart.wait_for(std::chrono::seconds(2)) ==
                           std::future_status::ready;
  if (secondReady) {
    EXPECT_EQ(flagcxSuccess, secondStart.get());
    EXPECT_EQ(flagcxSuccess, pool.retire(&independent));
  }
  gate.release();
  EXPECT_TRUE(secondReady) << "another registration waited for pool mutex";
  if (!secondReady) {
    EXPECT_EQ(flagcxSuccess, secondStart.get());
    EXPECT_EQ(flagcxSuccess, pool.retire(&independent));
  }
  EXPECT_EQ(flagcxSystemError, firstStart.get());
  EXPECT_EQ(nullptr, blocked);
  EXPECT_EQ(0u, pool.workerCount());
}

TEST(WorkerPool, RetiringOneWorkerDoesNotBlockAnother) {
  flagcxWorkerPool pool;
  Gate gate;
  Probe other;
  flagcxWorkerPool::Worker *blocked = nullptr;
  flagcxWorkerPool::Worker *independent = nullptr;
  ASSERT_EQ(flagcxSuccess, pool.start(&kBlockedDrainOps, &gate, &blocked));
  ASSERT_EQ(flagcxSuccess, pool.start(&kProbeOps, &other, &independent));

  auto firstRetire =
      std::async(std::launch::async, [&] { return pool.retire(&blocked); });
  if (!gate.waitUntilEntered()) {
    gate.release();
    FAIL() << "blocked worker did not enter drain";
  }
  auto secondRetire =
      std::async(std::launch::async, [&] { return pool.retire(&independent); });
  const bool secondReady = secondRetire.wait_for(std::chrono::seconds(2)) ==
                           std::future_status::ready;
  gate.release();
  EXPECT_TRUE(secondReady) << "another retirement waited for pool mutex";
  EXPECT_EQ(flagcxSuccess, secondRetire.get());
  EXPECT_EQ(flagcxSuccess, firstRetire.get());
  EXPECT_EQ(0u, pool.workerCount());
}

TEST(WorkerPool, ShutdownDuringInitializationRollsBackRegistration) {
  flagcxWorkerPool pool;
  Gate gate;
  flagcxWorkerPool::Worker *worker = nullptr;
  auto starting = std::async(std::launch::async, [&] {
    return pool.start(&kSuccessfulBlockedInitOps, &gate, &worker);
  });
  if (!gate.waitUntilEntered()) {
    gate.release();
    FAIL() << "worker did not enter initialization";
  }

  auto stopping =
      std::async(std::launch::async, [&] { pool.requestStopAll(); });
  const bool stopReady =
      stopping.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
  gate.release();
  EXPECT_TRUE(stopReady) << "pool stop waited for initialization";
  stopping.get();
  EXPECT_EQ(flagcxInvalidUsage, starting.get());
  EXPECT_EQ(flagcxSuccess, pool.joinAll());
  EXPECT_EQ(nullptr, worker);
  if (worker != nullptr)
    EXPECT_EQ(flagcxSuccess, pool.retire(&worker));
  EXPECT_EQ(0u, pool.workerCount());
}

TEST(WorkerPool, JoinAllAllowsIndependentRetirement) {
  flagcxWorkerPool pool;
  Gate gate;
  Probe other;
  flagcxWorkerPool::Worker *blocked = nullptr;
  flagcxWorkerPool::Worker *independent = nullptr;
  ASSERT_EQ(flagcxSuccess, pool.start(&kBlockedDrainOps, &gate, &blocked));
  ASSERT_EQ(flagcxSuccess, pool.start(&kProbeOps, &other, &independent));

  auto allJoined =
      std::async(std::launch::async, [&] { return pool.joinAll(); });
  if (!gate.waitUntilEntered()) {
    gate.release();
    FAIL() << "blocked worker did not enter drain";
  }
  auto independentRetire =
      std::async(std::launch::async, [&] { return pool.retire(&independent); });
  const bool retireReady =
      independentRetire.wait_for(std::chrono::seconds(2)) ==
      std::future_status::ready;
  auto blockedRetire =
      std::async(std::launch::async, [&] { return pool.retire(&blocked); });
  gate.release();
  EXPECT_TRUE(retireReady) << "retirement waited for joinAll's pool mutex";
  EXPECT_EQ(flagcxSuccess, independentRetire.get());
  EXPECT_EQ(flagcxSuccess, allJoined.get());
  EXPECT_EQ(flagcxSuccess, blockedRetire.get());
  EXPECT_EQ(0u, pool.workerCount());
}

} // namespace
