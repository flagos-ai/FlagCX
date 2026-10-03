/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include <gtest/gtest.h>

#include "adaptor.h"
#include "net.h"
#include "net_transport.h"
#include "proxy.h"

#include <memory>

namespace {

struct MockSemaphore final : flagcxSemaphore {
  int subCount = 0;

  flagcxEvent_t getEvent() override { return nullptr; }
  void signalStart() override {}
  void *getSignals() override { return nullptr; }
  void subCounter(int = 0) override { subCount++; }
  void addCounter(int = 0) override {}
  int getCounter() override { return 0; }
  int pollStart(int = 0, int = 0) override { return 1; }
  int pollEnd() override { return subCount != 0; }
  void wait() override {}
};

struct MockConcurrencyTracker {
  struct Domain {
    uint64_t key = 0;
    int outstanding = 0;
  };

  void accept(uint64_t key) {
    Domain *domain = find(key, true);
    ASSERT_NE(domain, nullptr);
    if (domain == nullptr)
      return;
    if (domain->outstanding++ == 0) {
      currentDomains++;
      if (currentDomains > maxConcurrentDomains)
        maxConcurrentDomains = currentDomains;
    }
  }

  void complete(uint64_t key) {
    Domain *domain = find(key, false);
    ASSERT_NE(domain, nullptr);
    ASSERT_GT(domain == nullptr ? 0 : domain->outstanding, 0);
    if (domain == nullptr || domain->outstanding == 0)
      return;
    if (--domain->outstanding == 0)
      currentDomains--;
  }

  int outstanding(uint64_t key) {
    Domain *domain = find(key, false);
    return domain == nullptr ? 0 : domain->outstanding;
  }

  Domain *find(uint64_t key, bool create) {
    Domain *freeDomain = nullptr;
    for (Domain &domain : domains) {
      if (domain.key == key)
        return &domain;
      if (freeDomain == nullptr && domain.key == 0)
        freeDomain = &domain;
    }
    if (create && freeDomain != nullptr) {
      freeDomain->key = key;
      return freeDomain;
    }
    return nullptr;
  }

  Domain domains[4] = {};
  int currentDomains = 0;
  int maxConcurrentDomains = 0;
};

struct MockRequest {
  int done = 0;
  flagcxResult_t testResult = flagcxSuccess;
  MockConcurrencyTracker *tracker = nullptr;
  uint64_t orderingKey = 0;
  bool accepted = false;
  bool completionObserved = false;
};

struct MockNetState {
  flagcxResult_t postResults[4] = {};
  int acceptRequest[4] = {};
  MockRequest requests[4] = {};
  flagcxNetSubmitContext submits[4] = {};
  int postCount = 0;
  int flushCount = 0;
  MockConcurrencyTracker *tracker = nullptr;
};

thread_local MockNetState *activeNetState = nullptr;

flagcxResult_t mockIsend(void *, void *, size_t, int, void *, void *,
                         void **request) {
  if (activeNetState == nullptr || request == nullptr)
    return flagcxInvalidArgument;
  const int index = activeNetState->postCount++;
  if (index >= 4)
    return flagcxInternalError;
  *request = nullptr;
  if (flagcxNetGetSubmitContext(&activeNetState->submits[index]) !=
      flagcxSuccess)
    return flagcxInternalError;
  if (activeNetState->acceptRequest[index]) {
    MockRequest *mock = &activeNetState->requests[index];
    mock->tracker = activeNetState->tracker;
    mock->orderingKey = activeNetState->submits[index].orderingKey;
    if (mock->tracker != nullptr && !mock->accepted) {
      mock->tracker->accept(mock->orderingKey);
      mock->accepted = true;
    }
    *request = mock;
  }
  return activeNetState->postResults[index];
}

flagcxResult_t mockIrecv(void *, int, void **, size_t *, int *, void **,
                         void **, void **request) {
  return mockIsend(nullptr, nullptr, 0, 0, nullptr, nullptr, request);
}

flagcxResult_t mockIflush(void *, int, void **, int *, void **,
                          void **request) {
  if (activeNetState == nullptr || request == nullptr)
    return flagcxInvalidArgument;
  activeNetState->flushCount++;
  *request = reinterpret_cast<void *>(0x1);
  return flagcxSuccess;
}

flagcxResult_t mockTest(void *request, int *done, int *sizes) {
  if (request == nullptr || done == nullptr)
    return flagcxInvalidArgument;
  auto *mock = static_cast<MockRequest *>(request);
  *done = mock->done;
  if (mock->done && mock->tracker != nullptr && !mock->completionObserved) {
    mock->tracker->complete(mock->orderingKey);
    mock->completionObserved = true;
  }
  if (sizes != nullptr)
    *sizes = 0;
  return mock->testResult;
}

struct ScopedNetChunkConfig {
  ScopedNetChunkConfig() {
    oldChunks = flagcxNetChunks;
    oldChunkSize = flagcxNetChunkSize;
    flagcxNetChunks = 2;
    flagcxNetChunkSize = 1;
  }
  ~ScopedNetChunkConfig() {
    flagcxNetChunks = oldChunks;
    flagcxNetChunkSize = oldChunkSize;
    activeNetState = nullptr;
  }

  int64_t oldChunks;
  int64_t oldChunkSize;
};

struct ScopedDeviceAdaptor {
  explicit ScopedDeviceAdaptor(uint32_t requirements) {
    saved = deviceAdaptor;
    adaptor.gdrFlushRequirements = requirements;
    deviceAdaptor = &adaptor;
  }
  ~ScopedDeviceAdaptor() { deviceAdaptor = saved; }

  flagcxDeviceAdaptor_latest adaptor{};
  flagcxDeviceAdaptor_latest *saved = nullptr;
};

struct SendProgressFixture {
  explicit SendProgressFixture(int channelId = 3,
                               MockConcurrencyTracker *tracker = nullptr) {
    adaptor.name = "MOCK";
    adaptor.isend = mockIsend;
    adaptor.test = mockTest;
    resources.netSendComm = this;
    resources.netAdaptor = &adaptor;
    semaphore = std::make_shared<MockSemaphore>();
    args.semaphore = semaphore;
    args.chunkSteps = 2;
    args.chunkSize = 1;
    args.sendStepMask = 1;
    args.regBufFlag = 1;
    args.regHandle = this;
    EXPECT_EQ(flagcxCollProxyTransportInit(
                  &args.collTransport, 2, 7,
                  flagcxCollProxyOrderingKey(channelId, 0, 1),
                  FLAGCX_NET_SUBMIT_INDEPENDENT),
              flagcxSuccess);
    state.tracker = tracker;
    activeNetState = &state;
  }

  flagcxResult_t progress() {
    activeNetState = &state;
    return flagcxProxySend(&resources, data, sizeof(data), &args);
  }

  flagcxNetAdaptor adaptor{};
  sendNetResources resources{};
  flagcxProxyArgs args{};
  std::shared_ptr<MockSemaphore> semaphore;
  MockNetState state{};
  char data[2] = {};
};

struct RecvProgressFixture {
  RecvProgressFixture() {
    adaptor.name = "MOCK";
    adaptor.irecv = mockIrecv;
    adaptor.test = mockTest;
    resources.netRecvComm = this;
    resources.netAdaptor = &adaptor;
    resources.ptrSupport = 0;
    semaphore = std::make_shared<MockSemaphore>();
    args.semaphore = semaphore;
    args.chunkSteps = 2;
    args.chunkSize = 1;
    args.sendStepMask = 1;
    args.regBufFlag = 1;
    args.regHandle = this;
    EXPECT_EQ(flagcxCollProxyTransportInit(&args.collTransport, 2, 8,
                                           flagcxCollProxyOrderingKey(3, 1, 0),
                                           FLAGCX_NET_SUBMIT_INDEPENDENT),
              flagcxSuccess);
    activeNetState = &state;
  }

  flagcxNetAdaptor adaptor{};
  recvNetResources resources{};
  flagcxProxyArgs args{};
  std::shared_ptr<MockSemaphore> semaphore;
  MockNetState state{};
  char data[2] = {};
};

TEST(CollProxyProgressTest, OutOfOrderCqeDoesNotAdvanceSendStep) {
  ScopedNetChunkConfig chunkConfig;
  SendProgressFixture fixture;
  fixture.state.postResults[0] = flagcxSuccess;
  fixture.state.postResults[1] = flagcxSuccess;
  fixture.state.acceptRequest[0] = 1;
  fixture.state.acceptRequest[1] = 1;
  fixture.state.requests[0] = {0, flagcxSuccess};
  fixture.state.requests[1] = {1, flagcxSuccess};

  ASSERT_EQ(flagcxProxySend(&fixture.resources, fixture.data,
                            sizeof(fixture.data), &fixture.args),
            flagcxSuccess);
  ASSERT_EQ(flagcxProxySend(&fixture.resources, fixture.data,
                            sizeof(fixture.data), &fixture.args),
            flagcxSuccess);

  EXPECT_EQ(fixture.args.posted, 2);
  EXPECT_EQ(fixture.args.transmitted, 0);
  EXPECT_EQ(fixture.state.submits[0].sequence, 0u);
  EXPECT_EQ(fixture.state.submits[1].sequence, 1u);
  EXPECT_EQ(fixture.args.subs[1].requests[0], nullptr);

  fixture.state.requests[0].done = 1;
  ASSERT_EQ(flagcxProxySend(&fixture.resources, fixture.data,
                            sizeof(fixture.data), &fixture.args),
            flagcxSuccess);
  EXPECT_EQ(fixture.args.transmitted, 2);
}

TEST(CollProxyProgressTest, BackpressureCancelsReservationBeforeRetry) {
  ScopedNetChunkConfig chunkConfig;
  SendProgressFixture fixture;
  fixture.state.postResults[0] = flagcxInProgress;
  fixture.state.postResults[1] = flagcxSuccess;
  fixture.state.acceptRequest[0] = 0;
  fixture.state.acceptRequest[1] = 1;
  fixture.state.requests[1] = {0, flagcxSuccess};

  ASSERT_EQ(flagcxProxySend(&fixture.resources, fixture.data,
                            sizeof(fixture.data), &fixture.args),
            flagcxSuccess);
  EXPECT_EQ(fixture.args.posted, 0);
  EXPECT_EQ(fixture.args.collTransport.nextSubmit, 0u);
  uint64_t next = 0;
  uint32_t inFlight = 0;
  flagcxResult_t firstError = flagcxSuccess;
  ASSERT_EQ(
      flagcxNetCompletionScoreboardQuery(&fixture.args.collTransport.scoreboard,
                                         &next, &inFlight, &firstError),
      flagcxSuccess);
  EXPECT_EQ(next, 0u);
  EXPECT_EQ(inFlight, 0u);

  ASSERT_EQ(flagcxProxySend(&fixture.resources, fixture.data,
                            sizeof(fixture.data), &fixture.args),
            flagcxSuccess);
  EXPECT_EQ(fixture.args.posted, 1);
  EXPECT_EQ(fixture.state.submits[1].sequence, 0u);
}

TEST(CollProxyProgressTest,
     DifferentOrderingDomainsCanRemainInflightAndRetireIndependently) {
  ScopedNetChunkConfig chunkConfig;
  MockConcurrencyTracker tracker;
  SendProgressFixture domainA(3, &tracker);
  SendProgressFixture domainB(4, &tracker);

  domainA.state.postResults[0] = flagcxSuccess;
  domainA.state.postResults[1] = flagcxSuccess;
  domainA.state.acceptRequest[0] = 1;
  domainA.state.acceptRequest[1] = 1;
  domainA.state.requests[0] = {0, flagcxSuccess};
  domainA.state.requests[1] = {1, flagcxSuccess};

  domainB.state.postResults[0] = flagcxSuccess;
  domainB.state.acceptRequest[0] = 1;
  domainB.state.requests[0] = {0, flagcxSuccess};

  // Domain A accepts two sequences. Sequence 1 completes first, but its
  // scoreboard cannot retire it past the still-pending sequence 0.
  ASSERT_EQ(domainA.progress(), flagcxSuccess);
  ASSERT_EQ(domainA.progress(), flagcxSuccess);
  ASSERT_EQ(domainA.args.posted, 2);
  EXPECT_EQ(domainA.args.transmitted, 0);
  EXPECT_EQ(domainA.state.submits[0].orderingKey,
            domainA.state.submits[1].orderingKey);
  EXPECT_EQ(domainA.state.submits[0].sequence, 0u);
  EXPECT_EQ(domainA.state.submits[1].sequence, 1u);

  // A different ordering domain must be accepted while A remains inflight.
  ASSERT_EQ(domainB.progress(), flagcxSuccess);
  const uint64_t keyA = domainA.state.submits[0].orderingKey;
  const uint64_t keyB = domainB.state.submits[0].orderingKey;
  ASSERT_NE(keyA, keyB);
  EXPECT_EQ(tracker.currentDomains, 2);
  EXPECT_EQ(tracker.maxConcurrentDomains, 2);
  EXPECT_EQ(tracker.outstanding(keyA), 1);
  EXPECT_EQ(tracker.outstanding(keyB), 1);

  uint64_t next = 0;
  uint32_t inFlight = 0;
  flagcxResult_t firstError = flagcxSuccess;
  ASSERT_EQ(
      flagcxNetCompletionScoreboardQuery(&domainA.args.collTransport.scoreboard,
                                         &next, &inFlight, &firstError),
      flagcxSuccess);
  EXPECT_EQ(next, 0u);
  EXPECT_EQ(inFlight, 2u);

  // Domain B can retire without waiting for A's sequence 0.
  domainB.state.requests[0].done = 1;
  ASSERT_EQ(domainB.progress(), flagcxSuccess);
  EXPECT_EQ(domainB.args.transmitted, 1);
  EXPECT_EQ(domainA.args.transmitted, 0);
  EXPECT_EQ(tracker.currentDomains, 1);
  EXPECT_EQ(tracker.outstanding(keyA), 1);

  // Once A's sequence 0 completes, its already-completed sequence 1 retires
  // in the same contiguous advance.
  domainA.state.requests[0].done = 1;
  ASSERT_EQ(domainA.progress(), flagcxSuccess);
  EXPECT_EQ(domainA.args.transmitted, 2);
  EXPECT_EQ(tracker.currentDomains, 0);
}

TEST(CollProxyProgressTest, OutOfOrderReceiveCqeDoesNotAdvanceCopyStep) {
  ScopedNetChunkConfig chunkConfig;
  RecvProgressFixture fixture;
  fixture.state.postResults[0] = flagcxSuccess;
  fixture.state.postResults[1] = flagcxSuccess;
  fixture.state.acceptRequest[0] = 1;
  fixture.state.acceptRequest[1] = 1;
  fixture.state.requests[0] = {0, flagcxSuccess};
  fixture.state.requests[1] = {1, flagcxSuccess};

  ASSERT_EQ(flagcxProxyRecv(&fixture.resources, fixture.data,
                            sizeof(fixture.data), &fixture.args),
            flagcxSuccess);
  ASSERT_EQ(flagcxProxyRecv(&fixture.resources, fixture.data,
                            sizeof(fixture.data), &fixture.args),
            flagcxSuccess);
  ASSERT_EQ(flagcxProxyRecv(&fixture.resources, fixture.data,
                            sizeof(fixture.data), &fixture.args),
            flagcxSuccess);

  EXPECT_EQ(fixture.args.posted, 2);
  EXPECT_EQ(fixture.args.netCompleted[1], 1);
  EXPECT_EQ(fixture.args.postFlush, 0);
  EXPECT_EQ(fixture.args.copied, 0);

  fixture.state.requests[0].done = 1;
  for (int i = 0; i < 6 && fixture.args.copied != 2; ++i) {
    ASSERT_EQ(flagcxProxyRecv(&fixture.resources, fixture.data,
                              sizeof(fixture.data), &fixture.args),
              flagcxSuccess);
  }
  EXPECT_EQ(fixture.args.postFlush, 2);
  EXPECT_EQ(fixture.args.copied, 2);
}

TEST(CollProxyProgressTest, NoWriteRequirementSkipsAutomaticFlush) {
  ScopedNetChunkConfig chunkConfig;
  ScopedDeviceAdaptor device(FLAGCX_GDR_FLUSH_NONE);
  RecvProgressFixture fixture;
  fixture.resources.ptrSupport = FLAGCX_PTR_CUDA;
  fixture.adaptor.iflush = mockIflush;
  fixture.adaptor.gdrFlushCaps = FLAGCX_NET_GDR_FLUSH_NONE;
  fixture.state.postResults[0] = flagcxSuccess;
  fixture.state.acceptRequest[0] = 1;
  fixture.state.requests[0] = {1, flagcxSuccess};

  for (int i = 0; i < 6 && fixture.args.copied != 1; ++i) {
    ASSERT_EQ(
        flagcxProxyRecv(&fixture.resources, fixture.data, 1, &fixture.args),
        flagcxSuccess);
  }

  EXPECT_EQ(fixture.state.flushCount, 0);
  EXPECT_EQ(fixture.args.postFlush, 1);
  EXPECT_EQ(fixture.args.copied, 1);
}

TEST(CollProxyProgressTest, RequiredWriteWithoutCapabilityFailsClosed) {
  ScopedNetChunkConfig chunkConfig;
  ScopedDeviceAdaptor device(FLAGCX_GDR_WRITE_REQUIRES_FLUSH);
  RecvProgressFixture fixture;
  fixture.resources.ptrSupport = FLAGCX_PTR_CUDA;
  fixture.adaptor.iflush = mockIflush;
  fixture.adaptor.gdrFlushCaps = FLAGCX_NET_GDR_FLUSH_NONE;
  fixture.state.postResults[0] = flagcxSuccess;
  fixture.state.acceptRequest[0] = 1;
  fixture.state.requests[0] = {1, flagcxSuccess};

  ASSERT_EQ(flagcxProxyRecv(&fixture.resources, fixture.data, 1, &fixture.args),
            flagcxSuccess);
  EXPECT_EQ(flagcxProxyRecv(&fixture.resources, fixture.data, 1, &fixture.args),
            flagcxNotSupported);
  EXPECT_EQ(fixture.state.flushCount, 0);
  EXPECT_EQ(fixture.args.postFlush, 0);
}

TEST(CollProxyProgressTest, PermanentErrorWakesEveryQueuedWaiter) {
  ScopedNetChunkConfig chunkConfig;
  SendProgressFixture fixture;
  fixture.state.postResults[0] = flagcxSuccess;
  fixture.state.acceptRequest[0] = 1;
  fixture.state.requests[0] = {1, flagcxRemoteError};

  flagcxResult_t progressResult = flagcxProxySend(
      &fixture.resources, fixture.data, sizeof(fixture.data), &fixture.args);
  ASSERT_EQ(progressResult, flagcxRemoteError);

  uint32_t abort = 0;
  flagcxProxyState proxyState{};
  proxyState.abortFlag = &abort;
  flagcxIntruQueue<flagcxProxyOp, &flagcxProxyOp::next> queue;
  flagcxIntruQueueConstruct(&queue);
  auto waiter = std::make_shared<MockSemaphore>();
  for (int i = 0; i < 2; ++i) {
    flagcxProxyOp *op = nullptr;
    ASSERT_EQ(flagcxCalloc(&op, 1), flagcxSuccess);
    op->args.semaphore = waiter;
    op->args.opId = i;
    flagcxIntruQueueEnqueue(&queue, op);
  }

  EXPECT_EQ(flagcxProxyFailProgressQueue(&proxyState, &queue, progressResult),
            flagcxRemoteError);
  EXPECT_EQ(proxyState.asyncResult, flagcxRemoteError);
  EXPECT_EQ(abort, 1u);
  EXPECT_EQ(waiter->subCount, 2);
  EXPECT_TRUE(flagcxIntruQueueEmpty(&queue));
}

} // namespace
