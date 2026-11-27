#include "adaptor.h"
#include "group.h"
#include "launch_kernel.h"
#include "proxy.h"
#include "transport.h"
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <future>
#include <gtest/gtest.h>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

namespace {
class GroupNotification : public ::testing::Test {
public:
  inline static GroupNotification *active;
  flagcxDeviceAdaptor fake = {};
  flagcxDeviceAdaptor *saved;
  flagcxLaunchFunc_t savedKernel;
  flagcxHeteroComm comms[2] = {};
  flagcxProxyState states[2] = {};
  flagcxTasks::Peer peers[2][2] = {};
  flagcxChannelPeer channelPeers[2] = {};
  flagcxChannelPeer *channelArrays[2][2] = {};
  flagcxHeteroComm::P2pSchedulePair schedules[2][2] = {};
  flagcxTransportComm transport = {};
  flagcxProxyConnection connections[2] = {};
  int input[2] = {11, 22}, output[2] = {};
  int streamTokens[2] = {};
  struct Callback {
    flagcxStream_t stream;
    void (*fn)(void *);
    void *arg;
    bool ran;
  };
  std::vector<Callback> callbacks;
  std::vector<char> trace;
  int launchCalls = 0, failLaunch = 0, wrapperFrees = 0, eventFrees = 0;
  int memoryAllocations = 0, memoryFrees = 0;
  bool failAlias = false, failMemoryFree = false;
  bool failEventDestroy = false;
  flagcxResult_t queryResult = flagcxSuccess;
  int destroyAttempts = 0, failDestroyAttempt = 0;
  std::set<flagcxEvent_t> destroyedEvents;

  void SetUp() override {
    active = this;
    saved = deviceAdaptor;
    savedKernel = deviceAsyncKernel;
    deviceAdaptor = &fake;
    deviceAsyncKernel = nullptr;
    fake.setDevice = [](int) { return flagcxSuccess; };
    fake.deviceMalloc = [](void **p, size_t bytes, flagcxMemType_t,
                           flagcxStream_t) {
      *p = calloc(1, bytes);
      ++active->memoryAllocations;
      return *p ? flagcxSuccess : flagcxSystemError;
    };
    fake.deviceFree = [](void *p, flagcxMemType_t, flagcxStream_t) {
      if (active->failMemoryFree)
        return flagcxInProgress;
      ++active->memoryFrees;
      free(p);
      return flagcxSuccess;
    };
    fake.hostGetDevicePointer = [](void **device, void *host) {
      if (active->failAlias)
        return flagcxRemoteError;
      *device = host;
      return flagcxSuccess;
    };
    fake.eventQuery = [](flagcxEvent_t event) {
      EXPECT_EQ(active->destroyedEvents.count(event), 0u);
      return active->queryResult;
    };
    fake.eventCreate = [](flagcxEvent_t *event, flagcxEventType_t) {
      *event = (flagcxEvent_t)calloc(1, 8);
      active->destroyedEvents.erase(*event);
      return *event ? flagcxSuccess : flagcxSystemError;
    };
    fake.eventRecord = [](flagcxEvent_t, flagcxStream_t stream) {
      EXPECT_NE(stream, nullptr);
      active->trace.push_back('R');
      return flagcxSuccess;
    };
    fake.eventDestroy = [](flagcxEvent_t event) {
      if (++active->destroyAttempts == active->failDestroyAttempt)
        return flagcxInProgress;
      if (active->failEventDestroy)
        return flagcxRemoteError;
      active->destroyedEvents.insert(event);
      ++active->eventFrees;
      free(event);
      return flagcxSuccess;
    };
    fake.streamCopy = [](flagcxStream_t *stream, void *native) {
      EXPECT_EQ(native, nullptr);
      *stream = (flagcxStream_t)calloc(1, 8);
      return *stream ? flagcxSuccess : flagcxSystemError;
    };
    fake.streamFree = [](flagcxStream_t stream) {
      ++active->wrapperFrees;
      free(stream);
      return flagcxSuccess;
    };
    fake.streamWaitEvent = [](flagcxStream_t stream, flagcxEvent_t) {
      EXPECT_NE(stream, nullptr);
      active->trace.push_back('W');
      return flagcxSuccess;
    };
    fake.launchHostFunc = [](flagcxStream_t stream, void (*fn)(void *),
                             void *arg) {
      EXPECT_NE(stream, nullptr);
      active->trace.push_back('H');
      if (++active->launchCalls == active->failLaunch)
        return flagcxRemoteError;
      active->callbacks.push_back({stream, fn, arg, false});
      return flagcxSuccess;
    };
    fake.streamQuery = [](flagcxStream_t stream) {
      if (active->queryResult != flagcxSuccess)
        return active->queryResult;
      // A cancelled callback returns immediately. Emulate its completion
      // before publishing a successful SDK stream-query result.
      active->runStream(stream);
      return flagcxSuccess;
    };
    fake.streamSynchronize = [](flagcxStream_t) {
      ADD_FAILURE() << "normal group notification must stay asynchronous";
      return flagcxInternalError;
    };
    for (int i = 0; i < 2; ++i) {
      auto &comm = comms[i];
      comm.rank = 0;
      comm.nRanks = comm.localRanks = 1;
      comm.tasks.peers = peers[i];
      comm.proxyState = &states[i];
      channelArrays[i][0] = &channelPeers[i];
      comm.channels[0].peers = channelArrays[i];
      channelPeers[i].send[0].proxyConn.connection = &connections[i];
      pthread_mutex_init(&states[i].mutex, nullptr);
      pthread_cond_init(&states[i].cond, nullptr);
      flagcxIntruQueueConstruct(&peers[i][0].sendQueue);
      flagcxIntruQueueConstruct(&peers[i][0].recvQueue);
      flagcxIntruQueueConstruct(&states[i].proxyOps[0].prodPeers.sendQueue);
      flagcxIntruQueueConstruct(&states[i].proxyOps[0].prodPeers.recvQueue);
    }
    flagcxGroupDepth = 1;
    flagcxGroupCommHead = flagcxGroupCommPreconnectHead = nullptr;
    flagcxGroupJobMainPtr = nullptr;
    flagcxGroupJobMain = {};
    flagcxIntruQueueConstruct(&flagcxAsyncJobs);
  }
  void runStream(flagcxStream_t stream) {
    for (auto &callback : callbacks) {
      if (callback.ran || callback.stream != stream)
        continue;
      callback.ran = true;
      callback.fn(callback.arg);
    }
  }
  void runCallback(size_t index) {
    callbacks[index].ran = true;
    callbacks[index].fn(callbacks[index].arg);
  }
  void pair(int commIndex, flagcxStream_t sendStream,
            flagcxStream_t recvStream) {
    auto *send = (flagcxTaskP2p *)calloc(1, sizeof(flagcxTaskP2p));
    auto *recv = (flagcxTaskP2p *)calloc(1, sizeof(flagcxTaskP2p));
    send->buff = &input[commIndex];
    recv->buff = &output[commIndex];
    send->bytes = recv->bytes = sizeof(int);
    send->dtype = recv->dtype = flagcxInt;
    send->opId = recv->opId = INT_MAX;
    send->step = recv->step = -1;
    send->stream = sendStream;
    recv->stream = recvStream;
    flagcxIntruQueueEnqueue(&peers[commIndex][0].sendQueue, send);
    flagcxIntruQueueEnqueue(&peers[commIndex][0].recvQueue, recv);
    comms[commIndex].tasks.nTasksP2p += 2;
  }
  auto &queue(int index) {
    return states[index].proxyOps[0].prodPeers.sendQueue;
  }
  void completeCopies() {
    for (int i = 0; i < 2; ++i)
      for (auto *op = flagcxIntruQueueHead(&queue(i)); op; op = op->next) {
        *reinterpret_cast<int *>(op->recvbuff) =
            *reinterpret_cast<int *>(op->sendbuff);
        op->args.semaphore->subCounter(op->args.opId);
        op->args.done = 1;
      }
  }
  void TearDown() override {
    failEventDestroy = false;
    queryResult = flagcxSuccess;
    for (int i = 0; i < 2; ++i)
      for (auto *op = flagcxIntruQueueHead(&queue(i)); op; op = op->next)
        if (op->args.semaphore)
          op->args.semaphore->cancelNotifications();
    for (size_t i = 0; i < callbacks.size(); ++i)
      if (!callbacks[i].ran)
        runCallback(i);
    for (int i = 0; i < 2; ++i) {
      flagcxProxyFailProgressQueue(&states[i], &queue(i), flagcxRemoteError);
      // Retry concrete owners retained by the production failure-retirement
      // path.
      while (states[i].deferredRelayOps) {
        auto *op = states[i].deferredRelayOps;
        states[i].deferredRelayOps = op->next;
        flagcxIntruQueueEnqueue(&queue(i), op);
        flagcxProxyFailProgressQueue(&states[i], &queue(i), flagcxRemoteError);
      }
      EXPECT_EQ(flagcxGroupDrainRejected(&comms[i]), flagcxSuccess);
      for (int peer = 0; peer < 2; ++peer) {
        while (!flagcxIntruQueueEmpty(&peers[i][peer].sendQueue))
          free(flagcxIntruQueueDequeue(&peers[i][peer].sendQueue));
        while (!flagcxIntruQueueEmpty(&peers[i][peer].recvQueue))
          free(flagcxIntruQueueDequeue(&peers[i][peer].recvQueue));
      }
      pthread_cond_destroy(&states[i].cond);
      pthread_mutex_destroy(&states[i].mutex);
    }
    flagcxGroupDepth = 0;
    flagcxGroupCommHead = flagcxGroupCommPreconnectHead = nullptr;
    deviceAdaptor = saved;
    deviceAsyncKernel = savedKernel;
  }
};

TEST_F(GroupNotification, DefaultStreamGetsARealNotificationAndRetryableOwner) {
  pair(0, nullptr, nullptr);
  flagcxGroupCommHead = &comms[0];
  ASSERT_EQ(flagcxGroupEndInternal(), flagcxSuccess);
  ASSERT_EQ(callbacks.size(), 1u);
  EXPECT_NE(callbacks[0].stream, nullptr);
  auto semaphore = flagcxIntruQueueHead(&queue(0))->args.semaphore;
  auto future = std::async(std::launch::async, [&] { runCallback(0); });
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!semaphore->pollStart(0, 0) &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  EXPECT_TRUE(semaphore->pollStart(0, 0));
  completeCopies();
  if (future.wait_for(std::chrono::seconds(2)) != std::future_status::ready)
    semaphore->cancelNotifications();
  future.get();
  EXPECT_EQ(output[0], input[0]);
  EXPECT_TRUE(semaphore->pollEnd());
  flagcxProxyFailProgressQueue(&states[0], &queue(0), flagcxRemoteError);
  failEventDestroy = true;
  EXPECT_EQ(flagcxReleaseSemaphore(&semaphore), flagcxRemoteError);
  EXPECT_NE(semaphore, nullptr);
  EXPECT_EQ(wrapperFrees, 0);
  failEventDestroy = false;
  EXPECT_EQ(flagcxReleaseSemaphore(&semaphore), flagcxSuccess);
  EXPECT_EQ(wrapperFrees, 1);
}

TEST_F(GroupNotification,
       BothSelfCopyStreamsGetOrderedCallbacksWithoutTailEvents) {
  auto a = (flagcxStream_t)&streamTokens[0],
       b = (flagcxStream_t)&streamTokens[1];
  pair(0, a, b);
  flagcxGroupCommHead = &comms[0];
  ASSERT_EQ(flagcxGroupEndInternal(), flagcxSuccess);
  ASSERT_EQ(callbacks.size(), 2u);
  auto semaphore = flagcxIntruQueueHead(&queue(0))->args.semaphore;
  auto first = std::async(std::launch::async, [&] { runCallback(0); });
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!semaphore->pollStart(0, 0) &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  completeCopies();
  if (first.wait_for(std::chrono::seconds(2)) != std::future_status::ready)
    semaphore->cancelNotifications();
  first.get();
  EXPECT_FALSE(
      semaphore->pollEnd()); // The other callback still owns a reference.
  runCallback(1);
  EXPECT_TRUE(semaphore->pollEnd());
  bool notified = false;
  for (char entry : trace) {
    if (entry == 'H')
      notified = true;
    if (notified)
      EXPECT_NE(entry, 'R');
  }
}

TEST_F(GroupNotification, OneCallbackCanStartAllCommsBeforeOtherCallbacksRun) {
  for (int i = 0; i < 2; ++i)
    pair(i, (flagcxStream_t)&streamTokens[i], (flagcxStream_t)&streamTokens[i]);
  comms[0].groupNext = &comms[1];
  flagcxGroupCommHead = &comms[0];
  ASSERT_EQ(flagcxGroupEndInternal(), flagcxSuccess);
  ASSERT_EQ(callbacks.size(), 2u);
  auto semaphore = flagcxIntruQueueHead(&queue(0))->args.semaphore;
  EXPECT_EQ(semaphore, flagcxIntruQueueHead(&queue(1))->args.semaphore);
  auto first = std::async(std::launch::async, [&] { runCallback(0); });
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!semaphore->pollStart(0, 0) &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  completeCopies();
  if (first.wait_for(std::chrono::seconds(2)) != std::future_status::ready)
    semaphore->cancelNotifications();
  first.get();
  runCallback(1);
  EXPECT_TRUE(semaphore->pollEnd());
}

TEST_F(GroupNotification,
       FailedNotificationCancelsEveryPostedCommAndKeepsOwners) {
  for (int i = 0; i < 2; ++i)
    pair(i, (flagcxStream_t)&streamTokens[i], (flagcxStream_t)&streamTokens[i]);
  comms[0].groupNext = &comms[1];
  flagcxGroupCommHead = &comms[0];
  failLaunch = 2;
  EXPECT_EQ(flagcxGroupEndInternal(), flagcxRemoteError);
  for (int i = 0; i < 2; ++i) {
    EXPECT_EQ(states[i].asyncResult, flagcxRemoteError);
    ASSERT_FALSE(flagcxIntruQueueEmpty(&queue(i)));
  }
  queryResult = flagcxInProgress;
  flagcxProxyFailProgressQueue(&states[0], &queue(0), flagcxRemoteError);
  EXPECT_NE(states[0].deferredRelayOps, nullptr);
  EXPECT_NE(states[0].deferredRelayOps->args.semaphore, nullptr);
  queryResult = flagcxSuccess;
}

TEST_F(GroupNotification, RejectedReadinessRetainsEventOwnerForCleanupRetry) {
  pair(0, (flagcxStream_t)&streamTokens[0], (flagcxStream_t)&streamTokens[0]);
  flagcxGroupCommHead = &comms[0];
  fake.eventRecord = [](flagcxEvent_t, flagcxStream_t) {
    return flagcxRemoteError;
  };
  EXPECT_EQ(flagcxGroupEndInternal(), flagcxRemoteError);
  EXPECT_NE(comms[0].rejectedGroupOps, nullptr);
  failEventDestroy = true;
  EXPECT_EQ(flagcxGroupDrainRejected(&comms[0]), flagcxRemoteError);
  EXPECT_NE(comms[0].rejectedGroupOps, nullptr);
  failEventDestroy = false;
  EXPECT_EQ(flagcxGroupDrainRejected(&comms[0]), flagcxSuccess);
  EXPECT_EQ(comms[0].rejectedGroupOps, nullptr);
  EXPECT_TRUE(callbacks.empty());
}

TEST_F(GroupNotification, UnmatchedEndDoesNotPoisonNextBalancedGroup) {
  flagcxGroupDepth = 0;
  EXPECT_EQ(flagcxGroupEndInternal(), flagcxInvalidUsage);
  EXPECT_EQ(flagcxGetGroupDepth(), 0);
  ASSERT_EQ(flagcxGroupStartInternal(), flagcxSuccess);
  EXPECT_EQ(flagcxGroupEndInternal(), flagcxSuccess);
  EXPECT_EQ(flagcxGetGroupDepth(), 0);
}

TEST_F(GroupNotification, FailedPreparationDiscardsAllUnsubmittedTasks) {
  pair(0, nullptr, nullptr);
  flagcxGroupCommHead = &comms[0];
  auto *job = (flagcxAsyncJob *)calloc(1, sizeof(flagcxAsyncJob));
  job->comm = &comms[0];
  job->func = [](flagcxAsyncJob *) { return flagcxRemoteError; };
  flagcxIntruQueueEnqueue(&flagcxAsyncJobs, job);
  EXPECT_EQ(flagcxGroupEndInternal(), flagcxRemoteError);
  EXPECT_TRUE(flagcxIntruQueueEmpty(&peers[0][0].sendQueue));
  EXPECT_TRUE(flagcxIntruQueueEmpty(&peers[0][0].recvQueue));
  EXPECT_EQ(comms[0].tasks.nTasksP2p, 0);
}

TEST_F(GroupNotification, PoolWrapCannotRecycleAStillOwnedSemaphoreSlot) {
  flagcxDeviceSemaphoreBufferPool pool;
  std::vector<int> ids(1);
  ASSERT_EQ(pool.getSlotId(&ids[0]), flagcxSuccess);
  auto *first = pool.getHostPtr(ids[0]);
  first[0] = 123;
  int originalCapacity = pool.capacity;
  for (int i = 0; i <= originalCapacity; ++i) {
    int id;
    ASSERT_EQ(pool.getSlotId(&id), flagcxSuccess);
    ids.push_back(id);
  }
  EXPECT_EQ(std::set<int>(ids.begin(), ids.end()).size(), ids.size());
  EXPECT_EQ(first[0], 123);
  EXPECT_EQ(pool.clear(), flagcxInProgress);
  for (int id : ids)
    ASSERT_EQ(pool.releaseSlot(id), flagcxSuccess);
  EXPECT_EQ(pool.clear(), flagcxSuccess);
}

TEST_F(GroupNotification, ConcurrentPoolGrowthLeasesUniqueStableSlots) {
  flagcxDeviceSemaphoreBufferPool pool;
  std::vector<int> ids;
  std::mutex mutex;
  std::vector<std::thread> threads;
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&] {
      for (int n = 0; n < 16; ++n) {
        int id = -1;
        EXPECT_EQ(pool.getSlotId(&id), flagcxSuccess);
        auto *host = pool.getHostPtr(id);
        EXPECT_EQ(pool.getDevicePtr(id), host);
        host[0] = id;
        std::lock_guard<std::mutex> lock(mutex);
        ids.push_back(id);
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  ASSERT_EQ(ids.size(), 128u);
  EXPECT_EQ(std::set<int>(ids.begin(), ids.end()).size(), ids.size());
  for (int id : ids) {
    EXPECT_EQ(pool.getHostPtr(id)[0], id);
    ASSERT_EQ(pool.releaseSlot(id), flagcxSuccess);
  }
  EXPECT_EQ(pool.clear(), flagcxSuccess);
}

TEST_F(GroupNotification, PartialPoolGrowthRetainsFailedAllocationOwner) {
  flagcxDeviceSemaphoreBufferPool pool;
  failAlias = true;
  int id = -1;
  EXPECT_EQ(pool.getSlotId(&id), flagcxRemoteError);
  EXPECT_EQ(id, -1);
  ASSERT_NE(pool.pending, nullptr);
  auto owner = pool.pending->signals;
  EXPECT_NE(owner, nullptr);
  failMemoryFree = true;
  EXPECT_EQ(pool.clear(), flagcxInProgress);
  EXPECT_EQ(pool.pending->signals, owner);
  failMemoryFree = false;
  EXPECT_EQ(pool.clear(), flagcxSuccess);
  EXPECT_EQ(memoryFrees, 1);
}

TEST_F(GroupNotification, DeviceNotificationNeedsDeviceAndHostRetirement) {
  ASSERT_EQ(deviceSemaphoreBufferPool.clear(), flagcxSuccess);
  flagcxDeviceSemaphore semaphore;
  ASSERT_EQ(semaphore.initResult, flagcxSuccess);
  ASSERT_NE(semaphore.getEvent(), nullptr);
  semaphore.notificationStreams.push_back((flagcxStream_t)&streamTokens[0]);
  semaphore.completionArmed = true;
  queryResult = flagcxInProgress;
  EXPECT_FALSE(semaphore.pollEnd());
  EXPECT_EQ(semaphore.release(), flagcxInProgress);
  EXPECT_EQ(deviceSemaphoreBufferPool.clear(), flagcxInProgress);
  queryResult = flagcxSuccess;
  EXPECT_TRUE(semaphore.pollEnd());
  EXPECT_EQ(semaphore.release(), flagcxSuccess);
  EXPECT_EQ(deviceSemaphoreBufferPool.clear(), flagcxSuccess);
}

TEST_F(GroupNotification, DeviceCancellationCannotRaceCounterBelowZero) {
  ASSERT_EQ(deviceSemaphoreBufferPool.clear(), flagcxSuccess);
  flagcxDeviceSemaphore semaphore;
  ASSERT_EQ(semaphore.initResult, flagcxSuccess);
  for (int i = 0; i < FLAGCX_OPS_PER_SEMAPHORE; ++i) {
    semaphore.addCounter(i);
    semaphore.signals[semaphore.curStep[i]] = 0;
  }
  std::vector<std::thread> threads;
  std::atomic<bool> start{false};
  for (int t = 0; t < 8; ++t)
    threads.emplace_back([&, t] {
      while (!start.load())
        std::this_thread::yield();
      for (int i = t; i < FLAGCX_OPS_PER_SEMAPHORE; i += 8)
        semaphore.subCounter(i);
    });
  start = true;
  semaphore.cancelNotifications();
  for (auto &thread : threads)
    thread.join();
  EXPECT_EQ(semaphore.getCounter(), 0);
  EXPECT_EQ(semaphore.release(), flagcxSuccess);
  EXPECT_EQ(deviceSemaphoreBufferPool.clear(), flagcxSuccess);
}

TEST_F(GroupNotification,
       StreamValueCannotRetireBeforeBuilderSealsNotification) {
  ASSERT_EQ(streamValueBufferPool.clear(), flagcxSuccess);
  flagcxStreamValueSemaphore semaphore;
  ASSERT_EQ(semaphore.initResult, flagcxSuccess);
  semaphore.prepareNotifications();
  semaphore.markCompletionDrained();
  EXPECT_FALSE(semaphore.pollEnd());
  semaphore.sealNotifications();
  EXPECT_TRUE(semaphore.pollEnd());
  EXPECT_EQ(semaphore.release(), flagcxSuccess);
  EXPECT_EQ(streamValueBufferPool.clear(), flagcxSuccess);
}

TEST_F(GroupNotification, DeviceCompletionErrorIsVisibleToProxyRetirement) {
  ASSERT_EQ(deviceSemaphoreBufferPool.clear(), flagcxSuccess);
  flagcxDeviceSemaphore semaphore;
  ASSERT_EQ(semaphore.initResult, flagcxSuccess);
  ASSERT_NE(semaphore.getEvent(), nullptr);
  semaphore.notificationStreams.push_back((flagcxStream_t)&streamTokens[0]);
  semaphore.completionArmed = true;
  queryResult = flagcxRemoteError;
  EXPECT_FALSE(semaphore.pollEnd());
  EXPECT_EQ(semaphore.getNotificationError(), flagcxRemoteError);
  semaphore.cancelNotifications(flagcxRemoteError);
  queryResult = flagcxSuccess;
  EXPECT_EQ(semaphore.release(), flagcxSuccess);
  EXPECT_EQ(deviceSemaphoreBufferPool.clear(), flagcxSuccess);
}

TEST_F(GroupNotification, StepProgressIsScopedToEachComm) {
  for (int i = 0; i < 2; ++i) {
    pair(i, (flagcxStream_t)&streamTokens[i], (flagcxStream_t)&streamTokens[i]);
    flagcxIntruQueueHead(&peers[i][0].sendQueue)->opId = 7;
    flagcxIntruQueueHead(&peers[i][0].recvQueue)->opId = 7;
  }
  comms[0].groupNext = &comms[1];
  flagcxGroupCommHead = &comms[0];
  ASSERT_EQ(flagcxGroupEndInternal(), flagcxSuccess);
  auto *a = flagcxIntruQueueHead(&queue(0));
  auto *b = flagcxIntruQueueHead(&queue(1));
  auto semaphore = a->args.semaphore;
  EXPECT_NE(a->args.opId, b->args.opId);
  auto first = std::async(std::launch::async, [&] { runCallback(0); });
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (!semaphore->pollStart(a->args.opId, 0) &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  semaphore->subCounter(a->args.opId);
  a->args.done = 1;
  EXPECT_FALSE(semaphore->pollStart(b->args.opId, 1));
  semaphore->subCounter(b->args.opId);
  b->args.done = 1;
  if (first.wait_for(std::chrono::seconds(2)) != std::future_status::ready)
    semaphore->cancelNotifications();
  first.get();
  runCallback(1);
  EXPECT_TRUE(semaphore->pollEnd());
}

TEST_F(GroupNotification,
       AcceptedCopyEventPinsTransportOwnerAfterCallbackCancellation) {
  pair(0, (flagcxStream_t)&streamTokens[0], (flagcxStream_t)&streamTokens[1]);
  flagcxGroupCommHead = &comms[0];
  failLaunch = 2;
  ASSERT_EQ(flagcxGroupEndInternal(), flagcxRemoteError);
  auto *op = flagcxIntruQueueHead(&queue(0));
  ASSERT_NE(op, nullptr);
  ASSERT_EQ(states[0].pendingTransportOps, 1u);
  auto copyEvent = (flagcxEvent_t)calloc(1, 8);
  op->args.pendingCopyEvents[0] = copyEvent;
  fake.eventQuery = [](flagcxEvent_t) { return flagcxInProgress; };
  flagcxProxyFailProgressQueue(&states[0], &queue(0), flagcxRemoteError);
  EXPECT_EQ(states[0].deferredRelayOps, op);
  EXPECT_EQ(states[0].pendingTransportOps, 1u);
  EXPECT_EQ(flagcxProxyScratchReady(&comms[0]), flagcxInProgress);
  fake.eventQuery = [](flagcxEvent_t) { return flagcxSuccess; };
  states[0].deferredRelayOps = nullptr;
  flagcxIntruQueueEnqueue(&queue(0), op);
  flagcxProxyFailProgressQueue(&states[0], &queue(0), flagcxRemoteError);
  EXPECT_EQ(states[0].deferredRelayOps, nullptr);
  EXPECT_EQ(states[0].pendingTransportOps, 0u);
  EXPECT_EQ(flagcxProxyScratchReady(&comms[0]), flagcxSuccess);
  free(copyEvent);
}
class RemoteGroupNotification : public GroupNotification,
                                public ::testing::WithParamInterface<bool> {
public:
  void SetUp() override {
    GroupNotification::SetUp();
    comms[0].nRanks = 2;
    comms[0].p2pSchedule = schedules[0];
    schedules[0][1] = {1, 1};
    channelArrays[0][1] = &channelPeers[0];
    channelPeers[0].recv[0].proxyConn.connection = &connections[0];
    transport.prepareProxyOp = [](flagcxHeteroComm *, flagcxProxyOp *, void *,
                                  size_t, int,
                                  flagcxDataType_t) { return flagcxSuccess; };
    connections[0].tcomm = &transport;
    pair(0, (flagcxStream_t)&streamTokens[0], (flagcxStream_t)&streamTokens[0]);
    auto *task = (flagcxTaskP2p *)calloc(1, sizeof(flagcxTaskP2p));
    task->buff = &input[0];
    task->bytes = sizeof(int);
    task->dtype = flagcxInt;
    task->opId = INT_MAX;
    task->step = -1;
    task->stream = (flagcxStream_t)&streamTokens[0];
    if (GetParam())
      flagcxIntruQueueEnqueue(&peers[0][1].sendQueue, task);
    else
      flagcxIntruQueueEnqueue(&peers[0][1].recvQueue, task);
    ++comms[0].tasks.nTasksP2p;
    flagcxGroupCommHead = &comms[0];
  }
};

TEST_P(RemoteGroupNotification, EventCreationFailureKeepsBothRollbackOwners) {
  fake.eventCreate = [](flagcxEvent_t *event, flagcxEventType_t) {
    if (!active->trace.empty())
      return flagcxRemoteError; // Self-copy readiness was already recorded.
    *event = (flagcxEvent_t)calloc(1, 8);
    return flagcxSuccess;
  };
  failEventDestroy = true;
  EXPECT_EQ(flagcxGroupEndInternal(), flagcxRemoteError);
  ASSERT_NE(comms[0].rejectedGroupOps, nullptr);
  ASSERT_NE(comms[0].rejectedGroupOps->next, nullptr);
  EXPECT_EQ(flagcxGroupDrainRejected(&comms[0]), flagcxRemoteError);
  EXPECT_EQ(comms[0].tasks.nTasksP2p, 0);
  EXPECT_TRUE(flagcxIntruQueueEmpty(&peers[0][1].sendQueue));
  EXPECT_TRUE(flagcxIntruQueueEmpty(&peers[0][1].recvQueue));
  EXPECT_EQ(launchCalls, 0);
  failEventDestroy = false;
  EXPECT_EQ(flagcxGroupDrainRejected(&comms[0]), flagcxSuccess);
  EXPECT_EQ(comms[0].rejectedGroupOps, nullptr);
  EXPECT_EQ(eventFrees, 1);
}

TEST_P(RemoteGroupNotification, MissingConnectionRollsBackPreparedPrefix) {
  if (GetParam())
    channelPeers[0].send[0].proxyConn.connection = nullptr;
  else
    channelPeers[0].recv[0].proxyConn.connection = nullptr;
  EXPECT_EQ(flagcxGroupEndInternal(), flagcxInternalError);
  ASSERT_NE(comms[0].rejectedGroupOps, nullptr);
  ASSERT_NE(comms[0].rejectedGroupOps->next, nullptr);
  EXPECT_EQ(comms[0].tasks.nTasksP2p, 0);
  EXPECT_TRUE(flagcxIntruQueueEmpty(&peers[0][1].sendQueue));
  EXPECT_TRUE(flagcxIntruQueueEmpty(&peers[0][1].recvQueue));
  EXPECT_EQ(launchCalls, 0);
  EXPECT_EQ(flagcxGroupDrainRejected(&comms[0]), flagcxSuccess);
  EXPECT_EQ(comms[0].rejectedGroupOps, nullptr);
  EXPECT_EQ(eventFrees, 1);
}

INSTANTIATE_TEST_SUITE_P(SendAndRecv, RemoteGroupNotification,
                         ::testing::Bool());
TEST_F(GroupNotification, DeviceCompletionProofSurvivesPartialEventDestroy) {
  ASSERT_EQ(deviceSemaphoreBufferPool.clear(), flagcxSuccess);
  auto derived = std::make_shared<flagcxDeviceSemaphore>();
  ASSERT_EQ(derived->initResult, flagcxSuccess);
  ASSERT_NE(derived->getEvent(), nullptr);
  ASSERT_NE(derived->getEvent(), nullptr);
  derived->notificationStreams.push_back((flagcxStream_t)&streamTokens[0]);
  derived->completionArmed = true;
  std::shared_ptr<flagcxSemaphore> owner = derived;
  derived.reset();
  failDestroyAttempt = 2;
  EXPECT_EQ(flagcxReleaseSemaphore(&owner), flagcxInProgress);
  ASSERT_NE(owner, nullptr);
  EXPECT_EQ(eventFrees, 1);
  EXPECT_TRUE(owner->pollEnd()); // Must not query the destroyed head event.
  EXPECT_EQ(flagcxReleaseSemaphore(&owner), flagcxSuccess);
  EXPECT_EQ(owner, nullptr);
  EXPECT_EQ(eventFrees, 2);
  EXPECT_EQ(deviceSemaphoreBufferPool.clear(), flagcxSuccess);
}

TEST_F(GroupNotification, StreamValueCompletionProofSurvivesPartialCleanup) {
  ASSERT_EQ(streamValueBufferPool.clear(), flagcxSuccess);
  fake.streamWriteValue64 = fake.streamWaitValue64 =
      [](flagcxStream_t, void *, uint64_t, int) { return flagcxSuccess; };
  auto derived = std::make_shared<flagcxStreamValueSemaphore>();
  ASSERT_EQ(derived->initResult, flagcxSuccess);
  ASSERT_NE(derived->getEvent(), nullptr);
  ASSERT_EQ(derived->enqueueCompletion((flagcxStream_t)&streamTokens[0]),
            flagcxSuccess);
  std::shared_ptr<flagcxSemaphore> owner = derived;
  derived.reset();
  failDestroyAttempt = 2;
  EXPECT_EQ(flagcxReleaseSemaphore(&owner), flagcxInProgress);
  ASSERT_NE(owner, nullptr);
  EXPECT_EQ(eventFrees, 1);
  EXPECT_TRUE(owner->pollEnd()); // Completion event was already destroyed.
  EXPECT_EQ(flagcxReleaseSemaphore(&owner), flagcxSuccess);
  EXPECT_EQ(owner, nullptr);
  EXPECT_EQ(eventFrees, 2);
  EXPECT_EQ(streamValueBufferPool.clear(), flagcxSuccess);
}
} // namespace
