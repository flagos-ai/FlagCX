#include "flagcx_hetero.h"
#include "ipc_lifecycle_mock.hpp"
#include "onesided.h"
#include "onesided_types.h"
#include <atomic>
#include <chrono>
#include <thread>

class FlushOwnership : public IpcLifecycleMock {
public:
  struct Request {};
  flagcxNetAdaptor net = {};
  flagcxOneSideHandleInfo info = {};
  Request *request = nullptr;
  flagcxResult_t postResult = flagcxSuccess, pollResult = flagcxRemoteError;
  bool outputRequest = true;
  std::atomic<bool> done{false};
  int posts = 0, polls = 0, retired = 0, deregistrations = 0;
  std::atomic<bool> entered{false}, hold{false};
  static FlushOwnership *self() {
    return static_cast<FlushOwnership *>(active);
  }
  static flagcxResult_t poll(void *ptr, int *done, int *) {
    auto *test = self();
    EXPECT_EQ(ptr, test->request);
    ++test->polls;
    test->entered = true;
    while (test->hold)
      std::this_thread::yield();
    *done = test->done;
    if (*done) {
      delete test->request;
      test->request = nullptr;
      ++test->retired;
    }
    return test->pollResult;
  }
  void SetUp() override {
    IpcLifecycleMock::SetUp();
    addTransport();
    comm->heteroComm->netAdaptor = &net;
    net.iflush = [](void *, int, void **, int *, void **, void **out) {
      ++self()->posts;
      if (self()->outputRequest) {
        EXPECT_EQ(self()->request, nullptr);
        self()->request = new Request;
        *out = self()->request;
      }
      return self()->postResult;
    };
    net.test = poll;
    net.deregMr = [](void *, void *) {
      ++self()->deregistrations;
      return flagcxSuccess;
    };
    info.localRecvComm = (void *)0x100;
    info.localMrHandle = (void *)0x200;
  }
  void TearDown() override {
    hold = false;
    done = true;
    pollResult = flagcxSuccess;
    net.test = poll;
    IpcLifecycleMock::TearDown();
    EXPECT_EQ(request, nullptr);
  }
  flagcxResult_t flush() {
    return flagcxHeteroFlush(comm->heteroComm, (void *)0x300, 8, &info);
  }
};

TEST_F(FlushOwnership, PollErrorRetainsRequestBeforeMrDeregistration) {
  auto *child = comm->heteroComm;
  auto *owner =
      (flagcxOneSideHandleInfo *)calloc(1, sizeof(flagcxOneSideHandleInfo));
  *owner = info;
  owner->windowRefs = 1;
  child->oneSideHandles = (flagcxOneSideHandleInfo **)calloc(1, sizeof(owner));
  child->oneSideHandles[0] = owner;
  child->oneSideHandleCount = child->oneSideHandleCapacity = 1;
  EXPECT_EQ(flush(), flagcxRemoteError);
  EXPECT_NE(request, nullptr);
  pollResult = flagcxSuccess;
  EXPECT_EQ(flagcxOneSideDeregisterInternal(child, 0), flagcxInProgress);
  EXPECT_EQ(deregistrations, 0);
  EXPECT_EQ(owner->windowRefs, 1u);
  done = true;
  EXPECT_EQ(flagcxOneSideDeregisterInternal(child, 0), flagcxSuccess);
  EXPECT_EQ(deregistrations, 1);
  EXPECT_EQ(retired, 1);
  EXPECT_EQ(child->oneSideHandles[0], nullptr);
}

TEST_F(FlushOwnership, FinalizeRetainsImportsAndConnectionsWhileFlushPending) {
  addBackingAndImport();
  EXPECT_EQ(flush(), flagcxRemoteError);
  pollResult = flagcxSuccess;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxInProgress);
  EXPECT_EQ(transportCloses, 0);
  EXPECT_TRUE(closes.empty());
  EXPECT_TRUE(frees.empty());
  flagcxHeteroComm independent = {};
  EXPECT_EQ(flagcxProxyTransportReady(&independent), flagcxSuccess);
  done = true;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(retired, 1);
  EXPECT_EQ(transportCloses, 1);
  EXPECT_EQ(closes.size(), 1u);
  destroySuccessfully();
}

TEST_F(FlushOwnership, FailedPostWithRequestKeepsAcceptedNativeOwner) {
  postResult = flagcxRemoteError;
  EXPECT_EQ(flush(), flagcxRemoteError);
  EXPECT_EQ(polls, 0);
  EXPECT_NE(request, nullptr);
  pollResult = flagcxSuccess;
  EXPECT_EQ(flagcxHeteroDrainFlushRequests(comm->heteroComm), flagcxInProgress);
  done = true;
  EXPECT_EQ(flagcxHeteroDrainFlushRequests(comm->heteroComm), flagcxSuccess);
  EXPECT_EQ(retired, 1);
  EXPECT_EQ(flagcxHeteroDrainFlushRequests(comm->heteroComm), flagcxSuccess);
  EXPECT_EQ(polls, 2);
}

TEST_F(FlushOwnership, NullRequestPreservesBackpressureAndLegacyNoop) {
  outputRequest = false;
  postResult = flagcxInProgress;
  EXPECT_EQ(flush(), flagcxInProgress);
  postResult = flagcxSuccess;
  EXPECT_EQ(flush(), flagcxSuccess);
  EXPECT_EQ(polls, 0);
  EXPECT_EQ(flagcxHeteroReleaseFlushState(comm->heteroComm), flagcxSuccess);
  EXPECT_EQ(comm->heteroComm->flushState, nullptr);
}

TEST_F(FlushOwnership, ErrorWithPhysicalCompletionNeverRepollsFreedRequest) {
  done = true;
  EXPECT_EQ(flush(), flagcxRemoteError);
  EXPECT_EQ(retired, 1);
  pollResult = flagcxSuccess;
  EXPECT_EQ(flagcxHeteroDrainFlushRequests(comm->heteroComm), flagcxSuccess);
  EXPECT_EQ(polls, 1);
}

TEST_F(FlushOwnership, MissingPollerRetainsRequestUntilItCanBeRetired) {
  net.test = nullptr;
  EXPECT_EQ(flush(), flagcxNotSupported);
  EXPECT_NE(request, nullptr);
  net.test = poll;
  done = true;
  pollResult = flagcxSuccess;
  EXPECT_EQ(flagcxHeteroDrainFlushRequests(comm->heteroComm), flagcxSuccess);
  EXPECT_EQ(retired, 1);
}

TEST_F(FlushOwnership, ConcurrentDrainReportsBusyWithoutBlockingOtherComm) {
  hold = true;
  flagcxResult_t result = flagcxSuccess;
  std::thread worker([&] { result = flush(); });
  auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!entered && std::chrono::steady_clock::now() < until)
    std::this_thread::yield();
  EXPECT_TRUE(entered);
  EXPECT_EQ(flagcxHeteroDrainFlushRequests(comm->heteroComm), flagcxInProgress);
  flagcxHeteroComm independent = {};
  EXPECT_EQ(flagcxProxyTransportReady(&independent), flagcxSuccess);
  hold = false;
  worker.join();
  EXPECT_EQ(result, flagcxRemoteError);
  done = true;
  pollResult = flagcxSuccess;
  EXPECT_EQ(flagcxHeteroDrainFlushRequests(comm->heteroComm), flagcxSuccess);
  EXPECT_EQ(retired, 1);
}

TEST_F(FlushOwnership, ProxyErrorInterruptsAcceptedFlushWithoutLosingRequest) {
  postResult = flagcxInProgress;
  pollResult = flagcxSuccess;
  comm->heteroComm->proxyState->asyncResult = flagcxRemoteError;
  std::atomic<bool> returned{false};
  flagcxResult_t result = flagcxSuccess;
  std::thread worker([&] {
    result = flush();
    returned = true;
  });
  auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!returned && std::chrono::steady_clock::now() < until)
    std::this_thread::yield();
  EXPECT_TRUE(returned);
  if (!returned)
    done = true; // Bound a regression instead of leaving an infinite poll.
  worker.join();
  EXPECT_EQ(result, flagcxRemoteError);
  EXPECT_NE(request, nullptr);
  done = true;
  EXPECT_EQ(flagcxHeteroDrainFlushRequests(comm->heteroComm), flagcxSuccess);
  EXPECT_EQ(retired, 1);
}
