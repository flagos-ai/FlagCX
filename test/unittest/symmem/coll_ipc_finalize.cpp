#include "comm_lifecycle_check.hpp"
#include "ipc_lifecycle_mock.hpp"
#include <chrono>
#include <cstring>
#include <thread>

// Real bootstrap coordination with deterministic local device failures. Ranks
// without an import/export must still take part in the same close protocol.
class IpcFinalizeBootstrapTest : public IpcLifecycleMock {
public:
  int rank = 0, nranks = 0;
  volatile uint32_t abortFlag = 0;

  void SetUp() override {
    IpcLifecycleMock::SetUp();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    if (nranks < 2)
      GTEST_SKIP() << "Requires at least two ranks";
    ASSERT_EQ(bootstrapNetInit(), flagcxSuccess);
    flagcxBootstrapHandle handle = {};
    if (rank == 0)
      ASSERT_EQ(bootstrapGetUniqueId(&handle), flagcxSuccess);
    MPI_Bcast(&handle, sizeof(handle), MPI_BYTE, 0, MPI_COMM_WORLD);
    ASSERT_EQ(bootstrapCollInit(&handle, rank, nranks, handle.magic, &abortFlag,
                                &comm->bootstrap),
              flagcxSuccess);
    comm->rank = rank;
    comm->nranks = nranks;
    comm->internalBuffersNeedPeerClose = true;
    if (rank == 0) {
      void *buffer = nullptr;
      ASSERT_EQ(flagcxCommAllocInternalBuffer(comm, &buffer, 64,
                                              flagcxInternalBufferDevice),
                flagcxSuccess);
    } else if (rank == 1) {
      auto &entry = comm->ipcTable[0];
      entry.ownership = flagcxBufferInternal;
      entry.inUse = entry.ownerHeld = true;
      entry.nPeers = 1;
      entry.hostPeerBasePtrs = (void **)calloc(1, sizeof(void *));
      entry.hostPeerBasePtrs[0] = (void *)0x55000;
    }
  }
  void expectCommon(flagcxResult_t result, flagcxResult_t expected) {
    int local = result, low = 0, high = 0;
    MPI_Allreduce(&local, &low, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local, &high, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_EQ(low, expected);
    EXPECT_EQ(high, expected);
  }
  void destroyOneRankAtATime() {
    ASSERT_EQ(flagcxTestFinalizeAndDestroyRankLocally(&comm, rank, nranks),
              flagcxSuccess);
    EXPECT_EQ(syncs, 0);
  }
};

TEST_F(IpcFinalizeBootstrapTest,
       EmptyRanksAndDelayedImporterJoinBeforeRelease) {
  if (rank == 1)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  expectCommon(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_TRUE(frees.empty());
  EXPECT_TRUE(comm->internalImportsConfirmed);
  if (rank == 1)
    EXPECT_EQ(closes.size(), 1u);
  // A cached successful Finalize must also be rank-local.
  if (rank == 0)
    EXPECT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  MPI_Barrier(MPI_COMM_WORLD);
  destroyOneRankAtATime();
  if (rank == 0)
    EXPECT_EQ(frees.size(), 1u);
}

TEST_F(IpcFinalizeBootstrapTest,
       FailedImporterKeepsAllExportersUntilCommonRetry) {
  failImport = rank == 1;
  expectCommon(flagcxCommFinalize(comm), flagcxRemoteError);
  EXPECT_FALSE(comm->internalImportsConfirmed);
  EXPECT_TRUE(frees.empty());
  if (rank == 0)
    EXPECT_NE(comm->internalBuffers, nullptr);
  failImport = false;
  expectCommon(flagcxCommFinalize(comm), flagcxSuccess);
  if (rank == 1)
    EXPECT_EQ(closes.size(), 2u);
  destroyOneRankAtATime();
}

TEST_F(IpcFinalizeBootstrapTest, DeviceSelectionFailureIsAgreedBeforeAnyClose) {
  failDevice = rank == 1;
  expectCommon(flagcxCommFinalize(comm), flagcxRemoteError);
  EXPECT_TRUE(closes.empty());
  EXPECT_EQ(comm->finalizePhase, 0);
  failDevice = false;
  expectCommon(flagcxCommFinalize(comm), flagcxSuccess);
  destroyOneRankAtATime();
}

TEST_F(IpcFinalizeBootstrapTest,
       BackingReleaseFailureRetriesWhilePeersStayOutsideFlagcx) {
  expectCommon(flagcxCommFinalize(comm), flagcxSuccess);
  if (rank == 0) {
    failFree = true;
    EXPECT_EQ(flagcxCommDestroy(comm), flagcxRemoteError);
    EXPECT_NE(comm->internalBuffers, nullptr);
    failFree = false;
    destroySuccessfully();
    EXPECT_EQ(frees.size(), 1u);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  for (int selected = 1; selected < nranks; ++selected) {
    if (rank == selected)
      destroySuccessfully();
    MPI_Barrier(MPI_COMM_WORLD);
  }
}
