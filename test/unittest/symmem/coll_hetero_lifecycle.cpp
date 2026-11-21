#include "bootstrap.h"
#include "p2p.h"
#include "relay_shutdown.hpp"
#include "rma_init_rollback.hpp"
#include <chrono>
#include <mpi.h>
#include <thread>
#include <vector>

namespace {
class HeteroFinalizeMpi : public RmaInitRollbackTest {
protected:
  inline static HeteroFinalizeMpi *active;
  int rank = 0, nranks = 0, fifoFrees = 0, importCloses = 0;
  int closeDelayMs = 0;
  bool failImport = false, failFifo = false;
  flagcxResult_t importResult = flagcxSuccess, fifoResult = flagcxSuccess;
  struct Fence {
    bool ready;
    flagcxStream_t stream = nullptr;
  };
  int waitFailures = 0, workWaits = 0, eventsCreated = 0, eventsFreed = 0;
  std::vector<int> localRanks;
  bootstrapState *bootstrap = nullptr;
  volatile uint32_t abort = 0;

  void SetUp() override {
    RmaInitRollbackTest::SetUp();
    active = this;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    if (nranks < 3)
      GTEST_SKIP()
          << "Requires three one-rank processes, including an empty rank";
    device.deviceSynchronize = [] {
      active->drains++;
      return flagcxUnhandledDeviceError;
    };
    device.streamGetCaptureStatus = [](flagcxStream_t,
                                       flagcxStreamCaptureStatus_t *status) {
      *status = flagcxStreamCaptureNone;
      return flagcxSuccess;
    };
    device.eventCreate = [](flagcxEvent_t *event, flagcxEventType_t) {
      *event = (flagcxEvent_t) new Fence{true};
      active->eventsCreated++;
      return flagcxSuccess;
    };
    device.eventRecord = [](flagcxEvent_t event, flagcxStream_t stream) {
      ((Fence *)event)->stream = stream;
      return flagcxSuccess;
    };
    device.eventQuery = [](flagcxEvent_t event) {
      auto *fence = (Fence *)event;
      bool ready = fence->stream != nullptr ? ((Fence *)fence->stream)->ready
                                            : fence->ready;
      return ready ? flagcxSuccess : flagcxInProgress;
    };
    device.eventSynchronize = [](flagcxEvent_t event) {
      active->workWaits++;
      if (active->waitFailures) {
        active->waitFailures--;
        return flagcxUnhandledDeviceError;
      }
      return active->device.eventQuery(event);
    };
    device.eventDestroy = [](flagcxEvent_t event) {
      delete (Fence *)event;
      active->eventsFreed++;
      return flagcxSuccess;
    };
    device.ipcMemHandleClose = [](void *) {
      active->importCloses++;
      std::this_thread::sleep_for(
          std::chrono::milliseconds(active->closeDelayMs));
      return active->failImport ? flagcxRemoteError : active->importResult;
    };
    device.deviceFree = [](void *, flagcxMemType_t, flagcxStream_t) {
      active->fifoFrees++;
      return active->failFifo ? flagcxSystemError : active->fifoResult;
    };
    ASSERT_EQ(bootstrapNetInit(), flagcxSuccess);
    flagcxBootstrapHandle handle = {};
    if (rank == 0)
      ASSERT_EQ(bootstrapGetUniqueId(&handle), flagcxSuccess);
    MPI_Bcast(&handle, sizeof(handle), MPI_BYTE, 0, MPI_COMM_WORLD);
    ASSERT_EQ(bootstrapCollInit(&handle, rank, nranks, handle.magic, &abort,
                                &bootstrap),
              flagcxSuccess);
    makeOwner();
    for (int channel = 0; channel < MAXCHANNELS; channel++) {
      free(owner->channels[channel].peers);
      owner->channels[channel].peers =
          (flagcxChannelPeer **)calloc(nranks, sizeof(flagcxChannelPeer *));
    }
    owner->rank = owner->localRank = rank;
    owner->nRanks = owner->localRanks = nranks;
    for (int peer = 0; peer < nranks; peer++)
      localRanks.push_back(peer);
    owner->localRankToRank = localRanks.data();
    owner->bootstrap = bootstrap;
    if (rank <= 1) {
      attachFallibleTransport();
      transport.cleanupProxyConnection = flagcxP2pCleanupProxyConnection;
      auto *connection = owner->proxyState->connectionPool->pools[0];
      connection->transport = TRANSPORT_P2P;
      connection->send = rank == 1;
      auto *resource =
          (flagcxP2pResources *)calloc(1, sizeof(flagcxP2pResources));
      connection->transportResources = resource;
      if (rank == 0)
        resource->localRecvFifo = (void *)0x44000;
      else
        resource->importedRecvFifoBase = (void *)0x55000;
    }
  }
  flagcxResult_t destroy() {
    flagcxResult_t result = flagcxHeteroCommDestroy(owner);
    if (result == flagcxSuccess)
      owner = nullptr;
    return result;
  }
  flagcxWorkRecord *attachWork(flagcxWorkTracker *tracker, bool ready) {
    flagcxWorkRecord *record = nullptr;
    EXPECT_EQ(flagcxWorkPrepare(tracker, nullptr, 0, &record), flagcxSuccess);
    if (record != nullptr) {
      EXPECT_EQ(flagcxWorkFinish(record, nullptr, flagcxSuccess,
                                 flagcxWorkNativeGroup, 0),
                flagcxSuccess);
      ((Fence *)record->event)->ready = ready;
    }
    return record;
  }
  void destroyInRankOrder() {
    int released = 1;
    if (rank != 0)
      MPI_Recv(&released, 1, MPI_INT, 0, 0x6d60, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    EXPECT_EQ(destroy(), flagcxSuccess);
    if (rank == 0)
      for (int peer = 1; peer < nranks; peer++)
        MPI_Send(&released, 1, MPI_INT, peer, 0x6d60, MPI_COMM_WORLD);
  }
  void TearDown() override {
    failImport = failFifo = false;
    importResult = fifoResult = flagcxSuccess;
    closeDelayMs = 0;
    failDevice = false;
    if (owner != nullptr)
      EXPECT_EQ(destroy(), flagcxSuccess);
    EXPECT_EQ(eventsCreated, eventsFreed);
    EXPECT_EQ(drains, 0);
    if (bootstrap != nullptr)
      EXPECT_EQ(bootstrapClose(bootstrap), flagcxSuccess);
    RmaInitRollbackTest::TearDown();
  }
};

TEST_F(HeteroFinalizeMpi, WorkWaitFailureConvergesBeforeAnyImportCloses) {
  attachWork(&owner->submittedWork, true);
  waitFailures = rank == 1 ? 1 : 0;
  EXPECT_EQ(flagcxHeteroCommFinalize(owner), flagcxUnhandledDeviceError);
  EXPECT_EQ(owner->finalizeState.phase, flagcxHeteroFinalizeDrain);
  EXPECT_EQ(importCloses + fifoFrees, 0);
  EXPECT_FALSE(owner->destroyProxyComplete);
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  EXPECT_EQ(workWaits, rank == 1 ? 2 : 1);
  destroyInRankOrder();
  EXPECT_EQ(fifoFrees, rank == 0 ? 1 : 0);
}

TEST_F(HeteroFinalizeMpi, IndependentCommunicatorWorkDoesNotDelayClosure) {
  flagcxWorkTracker other = {};
  auto *pending = attachWork(&other, false);
  ASSERT_NE(pending, nullptr);
  attachWork(&owner->submittedWork, true);
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  EXPECT_EQ(workWaits, 1);
  EXPECT_FALSE(((Fence *)pending->event)->ready);
  destroyInRankOrder();
  ((Fence *)pending->event)->ready = true;
  EXPECT_EQ(flagcxWorkWait(&other), flagcxSuccess);
  EXPECT_EQ(flagcxWorkRelease(&other), flagcxSuccess);
}

TEST_F(HeteroFinalizeMpi, RmaSyncAllocationsReleaseLocallyAfterFinalize) {
  if (rank == 0) {
    // Other ranks intentionally have no RMA allocations, but participate in
    // the communicator's existing close protocol.
    delete owner->workerPool;
    owner->workerPool = new flagcxWorkerPool;
    ASSERT_EQ(flagcxHeteroCommStartRma(&owner), flagcxSuccess);
  }
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  EXPECT_EQ(doneFrees + readyFrees, 0);
  if (rank == 0) {
    ASSERT_NE(owner->rmaProxy, nullptr);
    EXPECT_TRUE(owner->rmaProxy->stopComplete);
    EXPECT_NE(owner->rmaProxy->doneSeqsDev, nullptr);
    EXPECT_NE(owner->rmaProxy->readySeqsDev, nullptr);
    freeFailures = 1;
    EXPECT_EQ(destroy(), flagcxUnhandledDeviceError);
    ASSERT_NE(owner, nullptr);
    EXPECT_EQ(owner->finalizeState.phase, flagcxHeteroFinalizeComplete);
    EXPECT_EQ(owner->rmaProxy->doneSeqsDev, nullptr);
    EXPECT_NE(owner->rmaProxy->readySeqsDev, nullptr);
  }
  destroyInRankOrder(); // Peers wait in MPI during rank zero's local retry.
  EXPECT_EQ(doneFrees, rank == 0 ? 1 : 0);
  EXPECT_EQ(readyFrees, rank == 0 ? 2 : 0);
}

TEST_F(HeteroFinalizeMpi, PendingWorkKeepsDestroyOwnerUntilCompletion) {
  auto *pending = attachWork(&owner->submittedWork, rank != 1);
  ASSERT_NE(pending, nullptr);
  EXPECT_EQ(destroy(), flagcxInProgress);
  ASSERT_NE(owner, nullptr);
  EXPECT_EQ(importCloses + fifoFrees, 0);
  EXPECT_FALSE(owner->destroyProxyComplete);
  ((Fence *)pending->event)->ready = true;
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  destroyInRankOrder();
}

TEST_F(HeteroFinalizeMpi, FailedGroupPrefixConvergesBeforeImportClosure) {
  flagcxWorkRecord *record = nullptr;
  ASSERT_EQ(flagcxWorkPrepare(&owner->submittedWork, nullptr, 0, &record),
            flagcxSuccess);
  ASSERT_EQ(flagcxWorkFinish(record, nullptr, flagcxSuccess,
                             flagcxWorkNativeGroup, 1),
            flagcxSuccess);
  EXPECT_EQ(flagcxWorkGroupComplete(flagcxWorkNativeGroup, flagcxRemoteError),
            flagcxRemoteError);
  ((Fence *)record->event)->ready = rank != 1;
  EXPECT_EQ(flagcxHeteroCommFinalize(owner), flagcxInProgress);
  EXPECT_EQ(importCloses + fifoFrees, 0);
  EXPECT_FALSE(owner->finalizeState.importsClosedConfirmed);
  ((Fence *)record->event)->ready = true;
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  destroyInRankOrder();
  EXPECT_EQ(eventsCreated, eventsFreed);
}

TEST_F(HeteroFinalizeMpi, ProxyWorkWaitConvergesAndLocalReleaseCanRetry) {
  auto *pending = attachWork(&owner->proxyWork, rank != 1);
  ASSERT_NE(pending, nullptr);
  EXPECT_EQ(flagcxHeteroCommFinalize(owner), flagcxInProgress);
  EXPECT_EQ(importCloses + fifoFrees, 0);
  EXPECT_EQ(owner->submittedWork.closed, 1);
  EXPECT_EQ(owner->proxyWork.closed, 1);
  EXPECT_FALSE(owner->destroyProxyComplete);
  ((Fence *)pending->event)->ready = true;
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  destroyInRankOrder();
}

TEST_F(HeteroFinalizeMpi, DelayedImportAndEmptyRankConfirmBeforeLocalDestroy) {
  closeDelayMs = rank == 1 ? 300 : 0;
  auto started = std::chrono::steady_clock::now();
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  if (rank == 0)
    EXPECT_GE(std::chrono::steady_clock::now() - started,
              std::chrono::milliseconds(250));
  EXPECT_EQ(fifoFrees, 0);
  EXPECT_TRUE(owner->finalizeState.importsClosedConfirmed);
  EXPECT_EQ(importCloses, rank == 1 ? 1 : 0);
  EXPECT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  EXPECT_EQ(drains, 0);
  destroyInRankOrder(); // Peers wait in MPI while rank zero releases locally.
  EXPECT_EQ(fifoFrees, rank == 0 ? 1 : 0);
}

TEST_F(HeteroFinalizeMpi, FailedImportRetainsExporterAndLocalReleaseCanRetry) {
  failImport = rank == 1;
  EXPECT_EQ(destroy(), flagcxRemoteError);
  ASSERT_NE(owner, nullptr);
  EXPECT_FALSE(owner->finalizeState.importsClosedConfirmed);
  EXPECT_EQ(fifoFrees, 0);
  failImport = false;
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  EXPECT_EQ(importCloses, rank == 1 ? 2 : 0);
  EXPECT_EQ(drains, 0);
  if (rank == 0) {
    failFifo = true;
    EXPECT_EQ(destroy(), flagcxSystemError);
    ASSERT_NE(owner, nullptr);
    EXPECT_EQ(owner->finalizeState.phase, flagcxHeteroFinalizeComplete);
    failFifo = false;
  }
  destroyInRankOrder();
  EXPECT_EQ(fifoFrees, rank == 0 ? 2 : 0);
  EXPECT_EQ(importCloses, rank == 1 ? 2 : 0);
}

TEST_F(HeteroFinalizeMpi, PendingImportRetainsExporterAndEmptyRankUntilRetry) {
  importResult = rank == 1 ? flagcxInProgress : flagcxSuccess;
  EXPECT_EQ(destroy(), flagcxInProgress);
  ASSERT_NE(owner, nullptr);
  EXPECT_EQ(owner->finalizeState.phase, flagcxHeteroFinalizeProxy);
  EXPECT_FALSE(owner->finalizeState.importsClosedConfirmed);
  EXPECT_EQ(owner->destroyProxyComplete, rank != 1);
  EXPECT_EQ(fifoFrees, 0);
  if (rank == 1) {
    ASSERT_NE(owner->proxyState->connectionPool, nullptr);
    auto *resource =
        (flagcxP2pResources *)owner->proxyState->connectionPool->pools[0][0]
            .transportResources;
    ASSERT_NE(resource, nullptr);
    EXPECT_EQ(resource->importedRecvFifoBase, (void *)0x55000);
  }
  importResult = flagcxSuccess;
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  EXPECT_TRUE(owner->finalizeState.importsClosedConfirmed);
  EXPECT_EQ(importCloses, rank == 1 ? 2 : 0);
  EXPECT_EQ(fifoFrees, 0);
  destroyInRankOrder();
  EXPECT_EQ(fifoFrees, rank == 0 ? 1 : 0);
}

TEST_F(HeteroFinalizeMpi, PendingLocalReleaseRetriesWhilePeersWait) {
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  if (rank == 0) {
    fifoResult = flagcxInProgress;
    EXPECT_EQ(destroy(), flagcxInProgress);
    ASSERT_NE(owner, nullptr);
    EXPECT_EQ(owner->finalizeState.phase, flagcxHeteroFinalizeComplete);
    EXPECT_TRUE(owner->finalizeState.importsClosedConfirmed);
    ASSERT_NE(owner->proxyState->connectionPool, nullptr);
    auto *resource =
        (flagcxP2pResources *)owner->proxyState->connectionPool->pools[0][0]
            .transportResources;
    ASSERT_NE(resource, nullptr);
    EXPECT_EQ(resource->localRecvFifo, (void *)0x44000);
    fifoResult = flagcxSuccess;
  }
  // Other ranks wait in MPI while rank zero retries only local destruction.
  destroyInRankOrder();
  EXPECT_EQ(fifoFrees, rank == 0 ? 2 : 0);
  EXPECT_EQ(importCloses, rank == 1 ? 1 : 0);
}

TEST_F(HeteroFinalizeMpi, ClosureDoesNotWaitForAnotherNode) {
  if (rank < 2)
    localRanks = {0, 1};
  else
    localRanks = {rank};
  owner->localRank = rank < 2 ? rank : 0;
  owner->localRanks = (int)localRanks.size();
  owner->localRankToRank = localRanks.data();
  int released = 1;
  if (rank >= 2)
    MPI_Recv(&released, 1, MPI_INT, 0, 0x6d61, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  EXPECT_EQ(destroy(), flagcxSuccess);
  if (rank == 0)
    for (int peer = 2; peer < nranks; peer++)
      MPI_Send(&released, 1, MPI_INT, peer, 0x6d61, MPI_COMM_WORLD);
}

TEST_F(HeteroFinalizeMpi, DeviceSelectionFailureConvergesBeforeClosure) {
  failDevice = rank == 1;
  EXPECT_EQ(flagcxHeteroCommFinalize(owner), flagcxRemoteError);
  EXPECT_FALSE(owner->destroyProxyComplete);
  EXPECT_EQ(importCloses + fifoFrees, 0);
  failDevice = false;
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  destroyInRankOrder();
}
// Each process has one rank. The dummy service only models stop/join; close
// and release acknowledgement go through the actual socket RPC implementation.
TEST_F(HeteroFinalizeMpi, RelayCloseFailureKeepsEveryServiceAliveUntilRetry) {
  int peerFd = -1;
  flagcxProxyConnector *connector = nullptr;
  if (rank == 1) {
    connector = attachRelayShutdownEndpoint(owner, &peerFd);
    ASSERT_NE(connector, nullptr);
    ASSERT_TRUE(replyRelayShutdown(peerFd, connector, flagcxSuccess));
  }
  owner->proxyState->initialized = 1;
  ASSERT_EQ(pthread_create(
                &owner->proxyState->thread, nullptr,
                [](void *argument) -> void * {
                  auto *proxy = (flagcxProxyState *)argument;
                  while (!__atomic_load_n(&proxy->stop, __ATOMIC_ACQUIRE))
                    std::this_thread::yield();
                  return nullptr;
                },
                owner->proxyState),
            0);
  failImport = rank == 1;
  EXPECT_EQ(flagcxHeteroCommFinalize(owner), flagcxRemoteError);
  EXPECT_EQ(owner->finalizeState.phase, flagcxHeteroFinalizeRelay);
  EXPECT_EQ(owner->proxyState->initialized, 1);
  EXPECT_EQ(owner->proxyState->stop, 0);
  EXPECT_EQ(owner->proxyState->rpcStopping, 0);
  EXPECT_FALSE(owner->destroyProxyComplete);
  EXPECT_EQ(fifoFrees, 0);
  failImport = false;
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  EXPECT_EQ(owner->proxyState->initialized, 0);
  if (rank == 1) {
    EXPECT_TRUE(connector->connection->relayReleaseAcked);
    EXPECT_EQ(importCloses,
              3); // Failed relay close, successful retry, P2P close.
    close(peerFd);
  }
  destroyInRankOrder();
}

} // namespace

TEST_F(HeteroFinalizeMpi, GroupCompletionStreamProtectsNonLaunchCommunicator) {
  Fence original = {true}, completion = {rank != 1};
  flagcxWorkTracker other = {};
  flagcxWorkRecord *first = nullptr, *second = nullptr;
  ASSERT_EQ(flagcxWorkPrepare(&owner->submittedWork, (flagcxStream_t)&original,
                              0, &first),
            flagcxSuccess);
  ASSERT_EQ(flagcxWorkPrepare(&other, (flagcxStream_t)&completion, 0, &second),
            flagcxSuccess);
  ASSERT_EQ(flagcxWorkFinish(first, (flagcxStream_t)&original, flagcxSuccess,
                             flagcxWorkHeteroGroup, 1),
            flagcxSuccess);
  ASSERT_EQ(flagcxWorkFinish(second, (flagcxStream_t)&completion, flagcxSuccess,
                             flagcxWorkHeteroGroup, 1),
            flagcxSuccess);
  flagcxWorkGroupBoundary boundary = {(flagcxStream_t)&completion, 1, 1};
  ASSERT_EQ(flagcxWorkGroupCompleteAtBoundary(flagcxWorkHeteroGroup,
                                              flagcxSuccess, &boundary),
            flagcxSuccess);
  EXPECT_EQ(((Fence *)first->event)->stream, (flagcxStream_t)&completion);
  EXPECT_EQ(flagcxHeteroCommFinalize(owner), flagcxInProgress);
  EXPECT_EQ(importCloses + fifoFrees, 0);
  EXPECT_FALSE(owner->finalizeState.importsClosedConfirmed);
  completion.ready = true;
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  destroyInRankOrder();
  EXPECT_EQ(flagcxWorkWait(&other), flagcxSuccess);
  EXPECT_EQ(flagcxWorkRelease(&other), flagcxSuccess);
}
