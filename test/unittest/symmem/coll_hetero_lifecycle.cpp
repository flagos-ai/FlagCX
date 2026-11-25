#include "bootstrap.h"
#include "comm_lifecycle_check.hpp"
#include "p2p.h"
#include "rma_init_rollback.hpp"
#include <chrono>
#include <mpi.h>
#include <thread>
#include <vector>

namespace {
class HeteroFinalizeMpi : public RmaInitRollbackTest {
protected:
  inline static HeteroFinalizeMpi *active;
  int rank, nranks, importCloses = 0, fifoFrees = 0;
  bool failImport = false, failFree = false;
  int delayMs = 0;
  volatile uint32_t abortFlag = 0;
  std::vector<int> members;
  void SetUp() override {
    RmaInitRollbackTest::SetUp();
    active = this;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    ASSERT_GE(nranks, 2);
    device.ipcMemHandleClose = [](void *) {
      ++active->importCloses;
      std::this_thread::sleep_for(std::chrono::milliseconds(active->delayMs));
      return active->failImport ? flagcxRemoteError : flagcxSuccess;
    };
    device.deviceFree = [](void *, flagcxMemType_t, flagcxStream_t) {
      ++active->fifoFrees;
      return active->failFree ? flagcxSystemError : flagcxSuccess;
    };
    makeOwner();
    owner->rank = owner->localRank = rank;
    owner->nRanks = owner->localRanks = nranks;
    for (int i = 0; i < nranks; ++i)
      members.push_back(i);
    owner->localRankToRank = members.data();
    for (int c = 0; c < MAXCHANNELS; ++c) {
      free(owner->channels[c].peers);
      owner->channels[c].peers =
          (flagcxChannelPeer **)calloc(nranks, sizeof(flagcxChannelPeer *));
    }
    ASSERT_EQ(bootstrapNetInit(), flagcxSuccess);
    flagcxBootstrapHandle handle = {};
    if (rank == 0)
      ASSERT_EQ(bootstrapGetUniqueId(&handle), flagcxSuccess);
    MPI_Bcast(&handle, sizeof(handle), MPI_BYTE, 0, MPI_COMM_WORLD);
    ASSERT_EQ(bootstrapCollInit(&handle, rank, nranks, handle.magic, &abortFlag,
                                &owner->bootstrap),
              flagcxSuccess);
    if (rank < 2)
      attachFifo(rank == 0);
  }
  void attachFifo(bool exporter) {
    attachFallibleTransport();
    transport.cleanupProxyConnection = flagcxP2pCleanupProxyConnection;
    auto *connection = owner->proxyState->connectionPool->pools[0];
    connection->transport = TRANSPORT_P2P;
    connection->send = !exporter;
    auto *resources =
        (flagcxP2pResources *)calloc(1, sizeof(flagcxP2pResources));
    connection->transportResources = resources;
    if (exporter)
      resources->localRecvFifo = (void *)0x44000;
    else
      resources->importedRecvFifoBase = (void *)0x55000;
  }
  void destroyWhilePeersWait(bool retryFree = false) {
    MPI_Barrier(MPI_COMM_WORLD);
    int result = 0;
    if (rank == 0) {
      failFree = retryFree;
      if (retryFree) {
        EXPECT_EQ(flagcxHeteroCommDestroy(&owner), flagcxSystemError);
        ASSERT_NE(owner, nullptr);
        EXPECT_TRUE(owner->finalizeState.importsClosedConfirmed);
        failFree = false;
      }
      EXPECT_EQ(flagcxHeteroCommDestroy(&owner), flagcxSuccess);
      EXPECT_EQ(owner, nullptr);
      for (int peer = 1; peer < nranks; ++peer)
        MPI_Send(&result, 1, MPI_INT, peer, 0x6c31, MPI_COMM_WORLD);
    } else {
      MPI_Recv(&result, 1, MPI_INT, 0, 0x6c31, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      EXPECT_EQ(flagcxHeteroCommDestroy(&owner), flagcxSuccess);
      EXPECT_EQ(owner, nullptr);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  void TearDown() override {
    failImport = failFree = failDevice = false;
    RmaInitRollbackTest::TearDown();
  }
};
} // namespace

TEST_F(HeteroFinalizeMpi, DelayedImportAndEmptyRanksConfirmBeforeLocalDestroy) {
  if (rank == 1)
    delayMs = 25;
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  EXPECT_TRUE(owner->finalizeState.importsClosedConfirmed);
  EXPECT_EQ(fifoFrees, 0);
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  EXPECT_EQ(importCloses, rank == 1 ? 1 : 0);
  destroyWhilePeersWait();
  EXPECT_EQ(fifoFrees, rank == 0 ? 1 : 0);
}

TEST_F(HeteroFinalizeMpi, FailedImportRetainsExporterUntilCollectiveRetry) {
  failImport = rank == 1;
  EXPECT_EQ(flagcxHeteroCommFinalize(owner), flagcxRemoteError);
  EXPECT_FALSE(owner->finalizeState.importsClosedConfirmed);
  EXPECT_EQ(fifoFrees, 0);
  failImport = false;
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  EXPECT_EQ(importCloses, rank == 1 ? 2 : 0);
  destroyWhilePeersWait(true);
  EXPECT_EQ(fifoFrees, rank == 0 ? 2 : 0);
}

TEST_F(HeteroFinalizeMpi, DeviceSelectionFailureConvergesBeforeImportClosure) {
  failDevice = rank == 1;
  EXPECT_EQ(flagcxHeteroCommFinalize(owner), flagcxRemoteError);
  EXPECT_EQ(owner->finalizeState.phase, flagcxHeteroFinalizeDrain);
  EXPECT_EQ(importCloses + fifoFrees, 0);
  failDevice = false;
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  destroyWhilePeersWait();
}

TEST_F(HeteroFinalizeMpi, SparseNodeClosureDoesNotWaitForAnotherNode) {
  if (nranks < 4)
    GTEST_SKIP() << "Requires two node groups";
  // Move the importer into rank zero's sparse node group; the other node has
  // no imports or exports and waits outside FlagCX until rank zero finishes.
  if (rank == 1) {
    owner->finalizeState.rollbackPending = 1;
    ASSERT_EQ(flagcxProxyDestroy(owner), flagcxSuccess);
    owner->finalizeState.rollbackPending = 0;
  }
  if (rank == 2)
    attachFifo(false);
  members.clear();
  for (int i = rank % 2; i < nranks; i += 2)
    members.push_back(i);
  owner->localRank = rank / 2;
  owner->localRanks = members.size();
  owner->localRankToRank = members.data();
  int finished = 0;
  if (rank % 2)
    MPI_Recv(&finished, 1, MPI_INT, 0, 0x6c32, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  ASSERT_EQ(flagcxHeteroCommDestroy(&owner), flagcxSuccess);
  if (rank == 0)
    for (int peer = 1; peer < nranks; peer += 2)
      MPI_Send(&finished, 1, MPI_INT, peer, 0x6c32, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(HeteroFinalizeMpi, InitializationDeclaresIpcAndNativeScopeOnEmptyRanks) {
  flagcxComm policy = {};
  policy.rank = rank;
  policy.nranks = nranks;
  policy.localRank = rank;
  policy.localRanks = nranks;
  policy.localRankToRank = members.data();
  policy.bootstrap = owner->bootstrap;
  policy.heteroComm = rank == 0 ? owner : nullptr;
  policy.hostComm = rank == nranks - 1 ? (flagcxInnerComm_t)1 : nullptr;
  ASSERT_EQ(flagcxCommInitFinalizePolicy(&policy), flagcxSuccess);
  EXPECT_TRUE(policy.ipcClosePolicyConfirmed);
  EXPECT_TRUE(policy.internalBuffersNeedPeerClose);
  EXPECT_EQ(policy.nativeFinalizeScope, flagcxNativeFinalizeComm);
  EXPECT_EQ(policy.internalBuffers, nullptr);
  policy.bootstrap = nullptr;
  EXPECT_EQ(flagcxCommRequireIpcClose(&policy, 0), flagcxSuccess);
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  destroyWhilePeersWait();
}

TEST_F(HeteroFinalizeMpi, CollectiveCreationDeclaresClosureBeforePublication) {
  flagcxComm policy = {};
  policy.rank = rank;
  policy.nranks = nranks;
  policy.localRank = rank;
  policy.localRanks = nranks;
  policy.localRankToRank = members.data();
  policy.bootstrap = owner->bootstrap;
  ASSERT_EQ(flagcxCommRequireIpcClose(&policy, rank == 0), flagcxSuccess);
  EXPECT_TRUE(policy.internalBuffersNeedPeerClose);
  EXPECT_TRUE(policy.ipcClosePolicyConfirmed);
  EXPECT_EQ(policy.internalBuffers, nullptr);
  policy.bootstrap = nullptr;
  EXPECT_EQ(flagcxCommRequireIpcClose(&policy, 0), flagcxSuccess);
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  destroyWhilePeersWait();
}

TEST_F(HeteroFinalizeMpi,
       OuterFinalizeRetriesClosureThenEveryRankDestroysLocally) {
  outer = (flagcxComm_t)calloc(1, sizeof(flagcxComm));
  ASSERT_NE(outer, nullptr);
  outer->commType = flagcxCommunicatorHybrid;
  outer->rank = outer->localRank = rank;
  outer->nranks = outer->localRanks = nranks;
  outer->localRankToRank = (int *)calloc(nranks, sizeof(int));
  ASSERT_NE(outer->localRankToRank, nullptr);
  for (int i = 0; i < nranks; ++i)
    outer->localRankToRank[i] = i;
  outer->ipcTable = outer->ipcTableInline;
  outer->ipcTableCapacity = FLAGCX_MAX_IPC_ENTRIES;
  // Transfer this fixture's bootstrap to its outer owner. All child protocol
  // steps are coordinated by the outer Finalize, with no second agreement.
  outer->bootstrap = owner->bootstrap;
  owner->bootstrap = nullptr;
  outer->heteroComm = owner;
  owner->ownerComm = outer;
  owner->workOwner = &outer->submittedWork;
  auto *child = owner;
  owner = nullptr;
  ASSERT_EQ(flagcxCommInitFinalizePolicy(outer), flagcxSuccess);
  failImport = rank == 1;
  EXPECT_EQ(flagcxCommFinalize(outer), flagcxRemoteError);
  EXPECT_FALSE(child->finalizeState.importsClosedConfirmed);
  EXPECT_EQ(fifoFrees, 0);
  failImport = false;
  ASSERT_EQ(flagcxCommFinalize(outer), flagcxSuccess);
  EXPECT_TRUE(child->finalizeState.importsClosedConfirmed);
  EXPECT_EQ(flagcxTestFinalizeAndDestroyRankLocally(&outer, rank, nranks),
            flagcxSuccess);
  EXPECT_EQ(outer, nullptr);
  EXPECT_EQ(fifoFrees, rank == 0 ? 1 : 0);
}
