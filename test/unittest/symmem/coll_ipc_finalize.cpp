#include "comm_lifecycle_check.hpp"
#include "ipc_lifecycle_mock.hpp"
#include "reg_pool.h"
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
    comm->magic = handle.magic;
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

TEST_F(IpcFinalizeBootstrapTest,
       NativePreparationConvergesBeforeEnteringCollectiveDestroy) {
  comm->homoComm = (flagcxInnerComm_t)0x9000;
  comm->nativeDestroyFinalize = true;
  comm->nativeCloseNeedsAgreement = true;
  // This mimics a collective compatibility close. Premature first entry
  // returns an error instead of hanging the negative regression forever.
  sdk.commDestroy = [](flagcxInnerComm_t) {
    ++active->destroyCalls;
    if (active->pendingPolls == 0)
      return flagcxRemoteError;
    return MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS ? flagcxSuccess
                                                      : flagcxSystemError;
  };
  void *data = (void *)0x67890000;
  ASSERT_EQ(globalRegPool.registerBuffer(nullptr, data, 64), flagcxSuccess);
  auto *token = globalRegPool.getItem(nullptr, data);
  std::unique_ptr<flagcxRegItem, void (*)(flagcxRegItem *)> reader(
      nullptr, [](flagcxRegItem *item) { globalRegPool.releaseItem(item); });
  if (rank == 0)
    reader.reset(globalRegPool.acquireItem(nullptr, data));
  expectCommon(flagcxCommFinalize(comm), flagcxInProgress);
  EXPECT_EQ(comm->finalizePhase, 6);
  EXPECT_EQ(destroyCalls, 0);
  EXPECT_FALSE(comm->internalImportsConfirmed);
  EXPECT_TRUE(frees.empty());
  reader.reset();
  // A rank that already prepared its owners must not repeat rank-local
  // registration work inside collective native close on the next attempt.
  if (rank == 1)
    reader.reset(globalRegPool.acquireItem(nullptr, data));
  pendingPolls = 1;
  expectCommon(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(destroyCalls, 1);
  EXPECT_EQ(comm->homoComm, nullptr);
  reader.reset();
  EXPECT_EQ(globalRegPool.deregisterBuffer(nullptr, token), flagcxSuccess);
  destroyOneRankAtATime();
}

TEST_F(IpcFinalizeBootstrapTest,
       PendingEarlierNativeOwnerCompletesBeforeNextCollective) {
  comm->homoComm = (flagcxInnerComm_t)0x9000;
  comm->homoInterComm = (flagcxInnerComm_t)0xa000;
  comm->nativeCloseNeedsAgreement = true;
  pendingPolls = rank == 0 ? 2 : 0;
  sdk.commFinalize = [](flagcxInnerComm_t) {
    ++active->finalizeCalls;
    EXPECT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
    return flagcxInProgress;
  };
  expectCommon(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(finalizeCalls, 2);
  EXPECT_EQ(asyncPolls, rank == 0 ? 4 : 2);
  EXPECT_TRUE(frees.empty());
  destroyOneRankAtATime();
}

TEST_F(IpcFinalizeBootstrapTest,
       PendingEarlierCompatibilityCloseCompletesBeforeNextCollective) {
  comm->homoComm = (flagcxInnerComm_t)0x9000;
  comm->homoInterComm = (flagcxInnerComm_t)0xa000;
  comm->nativeCloseNeedsAgreement = true;
  comm->nativeDestroyFinalize = true;
  pendingPolls = rank == 0 ? 2 : 0;
  sdk.commDestroy = [](flagcxInnerComm_t handle) {
    ++active->destroyCalls;
    if (handle == (flagcxInnerComm_t)0x9000 && active->pendingPolls-- > 0)
      return flagcxInProgress;
    return MPI_Barrier(MPI_COMM_WORLD) == MPI_SUCCESS ? flagcxSuccess
                                                      : flagcxSystemError;
  };
  expectCommon(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(destroyCalls, rank == 0 ? 4 : 2);
  EXPECT_TRUE(frees.empty());
  destroyOneRankAtATime();
}

TEST_F(IpcFinalizeBootstrapTest, HostShmGenerationIsIssuedByRootAndRetained) {
  setenv("FLAGCX_SIGNAL_HOST_ENABLE", "1", 1);
  comm->commType = flagcxCommunicatorHybrid;
  comm->localRanks = nranks;
  comm->localRank = rank;
  comm->shmGeneration = rank == 0 ? 3 : 100;
  comm->localRankToRank = (int *)calloc(nranks, sizeof(int));
  for (int i = 0; i < nranks; ++i)
    comm->localRankToRank[i] = i;
  device.hostRegister = [](void *, size_t) { return flagcxSuccess; };
  device.hostUnregister = [](void *) { return flagcxSuccess; };
  device.hostGetDevicePointer = [](void **out, void *ptr) {
    *out = ptr;
    return flagcxSuccess;
  };
  device.deviceMemcpy = [](void *dst, void *src, size_t bytes,
                           flagcxMemcpyType_t, flagcxStream_t, void *) {
    memcpy(dst, src, bytes);
    return flagcxSuccess;
  };
  device.deviceMemset = [](void *ptr, int value, size_t bytes, flagcxMemType_t,
                           flagcxStream_t) {
    memset(ptr, value, bytes);
    return flagcxSuccess;
  };
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.intraBarrierCount = 1;
  flagcxDevComm_t first = nullptr, second = nullptr;
  ASSERT_EQ(flagcxDevCommCreate(comm, &reqs, &first), flagcxSuccess);
  EXPECT_EQ(first->shmGeneration, 4u);
  auto *oldStorage = (uint64_t *)first->localBarrierShmPtr;
  ASSERT_NE(oldStorage, nullptr);
  *oldStorage = 0x1234 + rank;
  MPI_Barrier(MPI_COMM_WORLD);
  ASSERT_EQ(flagcxDevCommDestroy(comm, first), flagcxSuccess);
  if (rank == 1)
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  ASSERT_EQ(flagcxDevCommCreate(comm, &reqs, &second), flagcxSuccess);
  EXPECT_EQ(second->shmGeneration, 5u);
  EXPECT_EQ(*oldStorage, 0x1234u + rank);
  EXPECT_EQ(*(uint64_t *)second->localBarrierShmPtr, 0u);
  ASSERT_EQ(flagcxDevCommDestroy(comm, second), flagcxSuccess);
  expectCommon(flagcxCommFinalize(comm), flagcxSuccess);
  destroyOneRankAtATime();
}

TEST_F(IpcFinalizeBootstrapTest,
       CounterOnlyCreationLeavesNextShmGenerationAligned) {
  setenv("FLAGCX_SIGNAL_HOST_ENABLE", "1", 1);
  comm->commType = flagcxCommunicatorHybrid;
  comm->localRanks = nranks;
  comm->localRank = rank;
  comm->localRankToRank = (int *)calloc(nranks, sizeof(int));
  for (int i = 0; i < nranks; ++i)
    comm->localRankToRank[i] = i;
  device.hostRegister = [](void *, size_t) { return flagcxSuccess; };
  device.hostUnregister = [](void *) { return flagcxSuccess; };
  device.deviceMemcpy = [](void *dst, void *src, size_t bytes,
                           flagcxMemcpyType_t, flagcxStream_t, void *) {
    memcpy(dst, src, bytes);
    return flagcxSuccess;
  };
  comm->shmGeneration = rank == 0 ? 8 : 100;
  device.hostGetDevicePointer = [](void **out, void *ptr) {
    *out = ptr;
    return flagcxSuccess;
  };
  device.deviceMemset = [](void *ptr, int value, size_t bytes, flagcxMemType_t,
                           flagcxStream_t) {
    memset(ptr, value, bytes);
    return flagcxSuccess;
  };
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.interCounterCount = 1;
  flagcxDevComm_t view = nullptr;
  ASSERT_EQ(flagcxDevCommCreate(comm, &reqs, &view), flagcxSuccess);
  EXPECT_EQ(view->shmGeneration, 0u);
  ASSERT_NE(view->counterBuffer, nullptr);
  EXPECT_EQ(view->counterBuffer[0], 0u);
  ASSERT_EQ(flagcxDevCommDestroy(comm, view), flagcxSuccess);
  // Every rank skips the counter-only reservation. The next creation opens
  // shared mappings and must receive its own fresh generation, not a stale one.
  reqs.interCounterCount = 0;
  reqs.intraBarrierCount = 1;
  view = nullptr;
  ASSERT_EQ(flagcxDevCommCreate(comm, &reqs, &view), flagcxSuccess);
  EXPECT_EQ(view->shmGeneration, 9u);
  EXPECT_NE(view->localBarrierShmPtr, nullptr);
  ASSERT_EQ(flagcxDevCommDestroy(comm, view), flagcxSuccess);
  expectCommon(flagcxCommFinalize(comm), flagcxSuccess);
  destroyOneRankAtATime();
}
