#include "adaptor.h"
#include "flagcx_hetero.h"
#include "group.h"
#include "launch_kernel.h"
#include "net.h"
#include "proxy.h"
#include "transport.h"
#include "work.h"
#include <cstdlib>
#include <gtest/gtest.h>
#include <memory>

namespace {
struct CompletionEvent {};

class GroupCompletionFailure : public ::testing::Test {
protected:
  inline static GroupCompletionFailure *active;
  flagcxDeviceAdaptor *savedAdaptor;
  flagcxLaunchFunc_t savedKernel;
  flagcxDeviceAdaptor device{};
  flagcxHeteroComm comm{};
  flagcxProxyState proxy{};
  flagcxChannelPeer peer{};
  flagcxChannelPeer *peers[1] = {&peer};
  flagcxTasks::Peer taskPeers[1]{};
  flagcxProxyOps &proxyOps = proxy.proxyOps[0];
  flagcxProxyConnection connection{};
  sendNetResources resources{};
  flagcxNetAdaptor net{};
  flagcxWorkRecord *parent = nullptr, *previous = nullptr;
  int order[4]{}, source = 1, destination = 0;
  int records = 0, failRecordAt = 0, destroyed = 0, created = 0;
  int launches = 0, failDestroy = 0, requestTests = 0;
  bool fenceReady = false, requestDone = false, failQuery = false;
  flagcxResult_t launchResult = flagcxUnhandledDeviceError;
  flagcxSemaphore *acceptedSemaphore = nullptr;

  static flagcxResult_t createEvent(flagcxEvent_t *out, flagcxEventType_t) {
    *out = (flagcxEvent_t) new CompletionEvent;
    ++active->created;
    return flagcxSuccess;
  }
  static flagcxResult_t recordEvent(flagcxEvent_t event, flagcxStream_t) {
    return ++active->records == active->failRecordAt
               ? flagcxUnhandledDeviceError
               : flagcxSuccess;
  }
  static flagcxResult_t queryEvent(flagcxEvent_t) {
    if (active->failQuery)
      return flagcxUnhandledDeviceError;
    return active->fenceReady ? flagcxSuccess : flagcxInProgress;
  }
  static flagcxResult_t destroyEvent(flagcxEvent_t event) {
    if (active->failDestroy) {
      --active->failDestroy;
      return flagcxSystemError;
    }
    delete (CompletionEvent *)event;
    ++active->destroyed;
    return flagcxSuccess;
  }
  static flagcxResult_t launchHost(flagcxStream_t, void (*)(void *),
                                   void *arg) {
    ++active->launches;
    if (active->launchResult == flagcxSuccess)
      active->acceptedSemaphore = (flagcxSemaphore *)arg;
    return active->launchResult;
  }
  static flagcxResult_t launchDevice(flagcxStream_t, flagcxLaunchFunc_t,
                                     void *) {
    ++active->launches;
    return active->launchResult;
  }
  static flagcxResult_t testRequest(void *, int *done, int *) {
    ++active->requestTests;
    *done = active->requestDone;
    return *done ? flagcxSuccess : flagcxInProgress;
  }
  void SetUp() override {
    active = this;
    savedAdaptor = deviceAdaptor;
    savedKernel = deviceAsyncKernel;
    ASSERT_EQ(deviceSemaphoreBufferPool.clear(), flagcxSuccess);
    device.eventCreate = createEvent;
    device.eventRecord = recordEvent;
    device.eventQuery = device.eventSynchronize = queryEvent;
    device.eventDestroy = destroyEvent;
    device.launchHostFunc = launchHost;
    device.launchDeviceFunc = launchDevice;
    device.deviceMalloc = [](void **ptr, size_t bytes, flagcxMemType_t,
                             flagcxStream_t) {
      *ptr = std::calloc(1, bytes);
      return *ptr ? flagcxSuccess : flagcxSystemError;
    };
    device.deviceFree = [](void *ptr, flagcxMemType_t, flagcxStream_t) {
      std::free(ptr);
      return flagcxSuccess;
    };
    device.hostGetDevicePointer = [](void **out, void *ptr) {
      *out = ptr;
      return flagcxSuccess;
    };
    device.setDevice = [](int) { return flagcxSuccess; };
    device.streamWaitEvent = [](flagcxStream_t, flagcxEvent_t) {
      return flagcxSuccess;
    };
    deviceAdaptor = &device;
    deviceAsyncKernel = nullptr;
    comm.nRanks = comm.localRanks = 1;
    comm.proxyState = &proxy;
    comm.groupNext = (flagcxHeteroComm *)1;
    comm.channels[0].peers = peers;
    comm.tasks.peers = taskPeers;
    comm.tasks.p2pOrder = order;
    peer.send[0].connected = peer.recv[0].connected = 1;
    peer.send[0].proxyConn.connection = &connection;
    connection.transport = TRANSPORT_NET;
    connection.send = 1;
    connection.transportResources = &resources;
    net.test = testRequest;
    resources.netAdaptor = &net;
    pthread_mutex_init(&proxy.mutex, nullptr);
    pthread_mutex_init(&proxy.rpcMutex, nullptr);
    pthread_mutex_init(&proxy.relayImportMutex, nullptr);
    pthread_cond_init(&proxy.cond, nullptr);
    ASSERT_EQ(flagcxWorkPrepareHost(&comm.submittedWork, &parent),
              flagcxSuccess);
    previous = flagcxWorkSetCurrent(parent);
  }
  flagcxResult_t submit(bool deviceKernel = false) {
    deviceAsyncKernel = deviceKernel ? +[](flagcxStream_t, void *) {} : nullptr;
    flagcxHeteroGroupStart();
    auto stream = (flagcxStream_t)this;
    EXPECT_EQ(flagcxHeteroSend(&source, 1, flagcxInt, 0, &comm, stream),
              flagcxSuccess);
    EXPECT_EQ(flagcxHeteroRecv(&destination, 1, flagcxInt, 0, &comm, stream),
              flagcxSuccess);
    auto result = flagcxHeteroGroupEnd();
    flagcxWorkSetCurrent(previous);
    previous = nullptr;
    EXPECT_EQ(flagcxWorkCancel(parent), flagcxSuccess);
    parent = nullptr;
    return result;
  }
  flagcxProxyOp *posted() {
    return flagcxIntruQueueHead(&proxyOps.prodPeers.sendQueue);
  }
  void retirePosted() {
    EXPECT_EQ(flagcxProxyFailProgressQueue(&proxy,
                                           &proxyOps.prodPeers.sendQueue,
                                           flagcxUnhandledDeviceError),
              flagcxUnhandledDeviceError);
  }
  void retryDeferred() {
    // Feed the retained op through the same production retirement checks.
    auto *op = proxy.deferredRelayOps;
    ASSERT_NE(op, nullptr);
    proxy.deferredRelayOps = op->next;
    op->next = nullptr;
    flagcxIntruQueueEnqueue(&proxyOps.prodPeers.sendQueue, op);
    retirePosted();
  }
  void TearDown() override {
    flagcxWorkSetCurrent(previous);
    if (parent)
      EXPECT_EQ(flagcxWorkCancel(parent), flagcxSuccess);
    fenceReady = requestDone = true;
    failQuery = false;
    failDestroy = 0;
    // The mock never schedules a real callback/kernel. Complete uncertain
    // markers here only to release fixture allocations after safety assertions.
    for (auto *r = comm.submittedWork.head; r; r = r->next)
      if (r->state == flagcxWorkUncertain)
        r->state = flagcxWorkComplete;
    if (posted())
      retirePosted();
    if (acceptedSemaphore) {
      // An accepted notification is completed explicitly by the mock runtime.
      if (auto *host = dynamic_cast<flagcxHostSemaphore *>(acceptedSemaphore)) {
        host->signalStart();
        for (auto &entry : host->signals)
          entry.first = entry.second;
        host->wait();
      }
      acceptedSemaphore = nullptr;
    }
    (void)flagcxGroupDrainWork(&comm, 0);
    for (int i = 0; i < 3 && proxy.deferredRelayOps; ++i)
      retryDeferred();
    EXPECT_EQ(proxy.deferredRelayOps, nullptr);
    EXPECT_EQ(flagcxGroupDrainWork(&comm, 1), flagcxSuccess);
    EXPECT_EQ(comm.pendingGroupWork, nullptr);
    EXPECT_EQ(flagcxWorkWait(&comm.submittedWork), flagcxSuccess);
    EXPECT_EQ(flagcxWorkRelease(&comm.submittedWork), flagcxSuccess);
    if (comm.proxyWork.initialized) {
      EXPECT_EQ(flagcxWorkWait(&comm.proxyWork), flagcxSuccess);
      EXPECT_EQ(flagcxWorkRelease(&comm.proxyWork), flagcxSuccess);
    }
    EXPECT_EQ(created, destroyed);
    EXPECT_EQ(deviceSemaphoreBufferPool.clear(), flagcxSuccess);
    pthread_cond_destroy(&proxy.cond);
    pthread_mutex_destroy(&proxy.mutex);
    pthread_mutex_destroy(&proxy.rpcMutex);
    pthread_mutex_destroy(&proxy.relayImportMutex);
    deviceAsyncKernel = savedKernel;
    deviceAdaptor = savedAdaptor;
  }
};

TEST_F(GroupCompletionFailure, RejectedHostWaitsForFenceAndNativeRequest) {
  ASSERT_EQ(submit(), flagcxUnhandledDeviceError);
  ASSERT_NE(posted(), nullptr);
  auto *semaphore = posted()->args.semaphore.get();
  posted()->args.subs[0].requests[0] = (void *)2;
  retirePosted();
  EXPECT_NE(proxy.deferredRelayOps, nullptr);
  EXPECT_EQ(flagcxHeteroCommFinalize(&comm), flagcxUnhandledDeviceError);
  EXPECT_EQ(flagcxGroupDrainWork(&comm, 0), flagcxUnhandledDeviceError);
  EXPECT_FALSE(semaphore->isCompletionCancelled());
  fenceReady = true;
  EXPECT_EQ(flagcxGroupDrainWork(&comm, 0), flagcxUnhandledDeviceError);
  EXPECT_TRUE(semaphore->isCompletionCancelled());
  EXPECT_TRUE(semaphore->pollEnd());
  retryDeferred();
  ASSERT_NE(proxy.deferredRelayOps, nullptr);
  EXPECT_NE(proxy.deferredRelayOps->args.subs[0].requests[0], nullptr);
  requestDone = true;
  retryDeferred();
  EXPECT_EQ(proxy.deferredRelayOps, nullptr);
  EXPECT_EQ(flagcxGroupDrainWork(&comm, 1), flagcxSuccess);
  EXPECT_EQ(flagcxGroupDrainWork(&comm, 1), flagcxSuccess);
  EXPECT_EQ(flagcxHeteroCommFinalize(&comm), flagcxSuccess);
  EXPECT_EQ(flagcxHeteroCommFinalize(&comm), flagcxSuccess);
}

TEST_F(GroupCompletionFailure, RejectedDeviceReleasesOnlyAfterFence) {
  ASSERT_EQ(submit(true), flagcxUnhandledDeviceError);
  ASSERT_NE(posted(), nullptr);
  auto *semaphore = posted()->args.semaphore.get();
  retirePosted();
  EXPECT_EQ(flagcxGroupDrainWork(&comm, 0), flagcxUnhandledDeviceError);
  EXPECT_FALSE(semaphore->pollEnd());
  fenceReady = true;
  EXPECT_EQ(flagcxGroupDrainWork(&comm, 0), flagcxUnhandledDeviceError);
  EXPECT_TRUE(semaphore->pollEnd());
  retryDeferred();
  EXPECT_EQ(proxy.deferredRelayOps, nullptr);
  EXPECT_EQ(flagcxGroupDrainWork(&comm, 1), flagcxSuccess);
}

TEST_F(GroupCompletionFailure, AcceptedHostWithFailedFenceIsNotCancelled) {
  launchResult = flagcxSuccess;
  failRecordAt = 2; // Readiness succeeds; the original last-use fence fails.
  ASSERT_EQ(submit(), flagcxUnhandledDeviceError);
  ASSERT_NE(acceptedSemaphore, nullptr);
  retirePosted();
  fenceReady = true;
  EXPECT_EQ(flagcxGroupDrainWork(&comm, 0), flagcxUnhandledDeviceError);
  EXPECT_FALSE(acceptedSemaphore->isCompletionCancelled());
  EXPECT_FALSE(acceptedSemaphore->pollEnd());
  EXPECT_NE(proxy.deferredRelayOps, nullptr);
}

TEST_F(GroupCompletionFailure, AcceptedDeviceIsNotCancelledAfterRecordFailure) {
  launchResult = flagcxSuccess;
  failRecordAt = 2; // Kernel accepted, then its readiness-event record fails.
  ASSERT_EQ(submit(true), flagcxUnhandledDeviceError);
  ASSERT_NE(posted(), nullptr);
  auto *semaphore = posted()->args.semaphore.get();
  retirePosted();
  fenceReady = true;
  EXPECT_EQ(flagcxGroupDrainWork(&comm, 0), flagcxUnhandledDeviceError);
  EXPECT_FALSE(semaphore->isCompletionCancelled());
  // Simulate the already accepted mock device kernel reaching its final store.
  auto *deviceSem = dynamic_cast<flagcxDeviceSemaphore *>(semaphore);
  ASSERT_NE(deviceSem, nullptr);
  __atomic_store_n(deviceSem->signals + FLAGCX_SIGNAL_COUNTER_OFFSET, 0,
                   __ATOMIC_RELEASE);
  retryDeferred();
  EXPECT_EQ(flagcxGroupDrainWork(&comm, 1), flagcxSuccess);
}

TEST_F(GroupCompletionFailure, EventDestroyFailureRetainsGroupForRetry) {
  ASSERT_EQ(submit(), flagcxUnhandledDeviceError);
  retirePosted();
  fenceReady = true;
  (void)flagcxGroupDrainWork(&comm, 0);
  retryDeferred();
  failDestroy = 1;
  EXPECT_EQ(flagcxGroupDrainWork(&comm, 1), flagcxSystemError);
  EXPECT_NE(comm.pendingGroupWork, nullptr);
  EXPECT_EQ(flagcxGroupDrainWork(&comm, 1), flagcxSuccess);
  EXPECT_EQ(comm.pendingGroupWork, nullptr);
}

TEST_F(GroupCompletionFailure,
       FailedFallbackRecordDoesNotAuthorizeCancellation) {
  failRecordAt = 2;
  ASSERT_EQ(submit(), flagcxUnhandledDeviceError);
  ASSERT_NE(posted(), nullptr);
  auto *semaphore = posted()->args.semaphore.get();
  retirePosted();
  fenceReady = true;
  EXPECT_EQ(flagcxGroupDrainWork(&comm, 0), flagcxUnhandledDeviceError);
  EXPECT_FALSE(semaphore->isCompletionCancelled());
  EXPECT_NE(comm.pendingGroupWork, nullptr);
  EXPECT_NE(proxy.deferredRelayOps, nullptr);
}

TEST_F(GroupCompletionFailure, FailedFenceQueryRetainsOwnerUntilQuerySucceeds) {
  ASSERT_EQ(submit(), flagcxUnhandledDeviceError);
  ASSERT_NE(posted(), nullptr);
  auto *semaphore = posted()->args.semaphore.get();
  retirePosted();
  fenceReady = failQuery = true;
  EXPECT_EQ(flagcxGroupDrainWork(&comm, 0), flagcxUnhandledDeviceError);
  EXPECT_FALSE(semaphore->isCompletionCancelled());
  failQuery = false;
  EXPECT_EQ(flagcxGroupDrainWork(&comm, 0), flagcxUnhandledDeviceError);
  EXPECT_TRUE(semaphore->isCompletionCancelled());
  retryDeferred();
  EXPECT_EQ(flagcxGroupDrainWork(&comm, 1), flagcxSuccess);
}

TEST_F(GroupCompletionFailure, PreparationFailureDoesNotLaunchNotification) {
  failRecordAt = 1;
  ASSERT_EQ(submit(), flagcxUnhandledDeviceError);
  EXPECT_EQ(launches, 0);
  EXPECT_EQ(posted(), nullptr);
  fenceReady = true;
  EXPECT_EQ(flagcxGroupDrainWork(&comm, 1), flagcxSuccess);
}
} // namespace
