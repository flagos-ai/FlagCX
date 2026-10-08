// MPI tests for symmetric window register / deregister.
// Requires MPI + GPUs.

#include "adaptor.h"
#include "flagcx_net_adaptor.h"
#include "global_comm.h"
#include "onesided.h"
#include "sym_heap.h"
#include "symmem_test.hpp"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool envEnabled(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && std::strcmp(value, "0") != 0;
}

class ScopedEnvVar {
public:
  ScopedEnvVar(const char *name, const char *value) : name_(name) {
    const char *old = std::getenv(name);
    if (old != nullptr) {
      hadOldValue_ = true;
      oldValue_ = old;
    }
    set(value);
  }

  void set(const char *value) {
    if (value != nullptr)
      setenv(name_, value, 1);
    else
      unsetenv(name_);
  }

  ~ScopedEnvVar() {
    if (hadOldValue_)
      setenv(name_, oldValue_.c_str(), 1);
    else
      unsetenv(name_);
  }

private:
  const char *name_;
  bool hadOldValue_ = false;
  std::string oldValue_;
};

::testing::AssertionResult vmmMrRouteMatches(uint8_t actual) {
  const char *expected = std::getenv("FLAGCX_CI_EXPECT_VMM_MR_ROUTE");
  if (expected == nullptr || expected[0] == '\0') {
    if (actual != FLAGCX_VMM_MR_ROUTE_NONE)
      return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure()
           << "VMM MR did not record a registration route";
  }

  flagcxVmmMrRoute_t expectedRoute = FLAGCX_VMM_MR_ROUTE_NONE;
  if (std::strcmp(expected, "dmabuf") == 0)
    expectedRoute = FLAGCX_VMM_MR_ROUTE_DMABUF;
  else if (std::strcmp(expected, "va") == 0)
    expectedRoute = FLAGCX_VMM_MR_ROUTE_VA;
  else
    return ::testing::AssertionFailure()
           << "invalid FLAGCX_CI_EXPECT_VMM_MR_ROUTE=" << expected;

  if (actual == static_cast<uint8_t>(expectedRoute))
    return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure()
         << "expected VMM MR route " << expected << " ("
         << static_cast<int>(expectedRoute) << "), got "
         << static_cast<int>(actual);
}

bool allRanksSucceeded(flagcxResult_t result) {
  int local = result == flagcxSuccess ? 1 : 0;
  int global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  return global != 0;
}

void allRankResultRange(flagcxResult_t result, int *minimum, int *maximum) {
  int local = static_cast<int>(result);
  MPI_Allreduce(&local, minimum, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local, maximum, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
}

flagcxResult_t failSymPhysAlloc(void *, size_t, void **physHandle,
                                void *shareableHandle, size_t *, size_t *) {
  if (physHandle != nullptr)
    *physHandle = nullptr;
  if (shareableHandle != nullptr)
    *static_cast<int *>(shareableHandle) = -1;
  return flagcxSystemError;
}

flagcxResult_t failRegMr(void *, void *, size_t, int, int, void **mhandle) {
  if (mhandle != nullptr)
    *mhandle = nullptr;
  return flagcxSystemError;
}

flagcxResult_t failRegMrDmaBuf(void *, void *, size_t, int, uint64_t, int, int,
                               void **mhandle) {
  if (mhandle != nullptr)
    *mhandle = nullptr;
  return flagcxSystemError;
}

flagcxResult_t failDeregMr(void *, void *) { return flagcxSystemError; }

flagcxResult_t (*savedFullMeshConnect)(int, void *, void **) = nullptr;
flagcxResult_t (*savedFullMeshResetConnect)(void *) = nullptr;
void *firstFullMeshHandle = nullptr;
int injectedFullMeshConnectFailures = 0;
int injectedFullMeshResetFailures = 0;

flagcxResult_t failSecondFullMeshConnect(int dev, void *handle,
                                         void **sendComm) {
  if (firstFullMeshHandle == nullptr)
    firstFullMeshHandle = handle;
  else if (handle != firstFullMeshHandle &&
           injectedFullMeshConnectFailures++ == 0) {
    if (sendComm != nullptr)
      *sendComm = nullptr;
    return flagcxSystemError;
  }
  return savedFullMeshConnect(dev, handle, sendComm);
}

flagcxResult_t failFirstFullMeshReset(void *handle) {
  if (injectedFullMeshResetFailures++ == 0)
    return flagcxSystemError;
  return savedFullMeshResetConnect(handle);
}

flagcxResult_t failIpcMemHandleGet(flagcxIpcMemHandle_t, void *) {
  return flagcxSystemError;
}

flagcxResult_t (*savedSymFlatMap)(void *[], int, int, void *, size_t,
                                  void **) = nullptr;
flagcxResult_t (*savedSymFlatUnmap)(void *, size_t, int) = nullptr;
int injectedFlatUnmapFailures = 0;

flagcxResult_t failSymFlatMap(void *[], int, int, void *, size_t,
                              void **flatBase) {
  if (flatBase != nullptr)
    *flatBase = nullptr;
  return flagcxSystemError;
}

flagcxResult_t failSymFlatUnmapOnce(void *flatBase, size_t allocSize,
                                    int nPeers) {
  if (injectedFlatUnmapFailures++ == 0)
    return flagcxRemoteError;
  return savedSymFlatUnmap != nullptr
             ? savedSymFlatUnmap(flatBase, allocSize, nPeers)
             : flagcxInternalError;
}

flagcxResult_t createTestComm(flagcxComm_t *testComm) {
  int rank = 0;
  int nranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  flagcxUniqueId uniqueId = {};
  flagcxResult_t result =
      rank == 0 ? flagcxGetUniqueId(&uniqueId) : flagcxSuccess;
  if (!allRanksSucceeded(result))
    return result;
  MPI_Bcast(&uniqueId, sizeof(uniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
  return flagcxCommInitRank(testComm, nranks, &uniqueId, rank);
}

int findLocalMrIndex(flagcxHeteroComm_t comm, const void *buffer) {
  if (comm == nullptr)
    return -1;
  for (int i = 0; i < comm->oneSideHandleCount; i++) {
    auto *handle = comm->oneSideHandles[i];
    if (handle != nullptr && handle->baseVas != nullptr &&
        handle->baseVas[comm->rank] == reinterpret_cast<uintptr_t>(buffer))
      return i;
  }
  return -1;
}

} // namespace

// ---------------------------------------------------------------------------
// Register with FLAGCX_WIN_COLL_SYMMETRIC
// ---------------------------------------------------------------------------

TEST_F(SymMemTest, RankLocalStatusFailureConvergesDeterministically) {
  int rank = 0;
  int nranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  if (nranks < 2)
    GTEST_SKIP() << "requires at least two ranks";

  flagcxResult_t localStatus = flagcxSuccess;
  if (rank == 1)
    localStatus = flagcxSystemError;
  else if (rank == 2)
    localStatus = flagcxRemoteError;

  flagcxResult_t commonStatus = flagcxSuccess;
  flagcxResult_t result =
      flagcxSymConvergeStatus(comm->heteroComm, localStatus, &commonStatus);
  ASSERT_TRUE(allRanksSucceeded(result));
  int minimum = 0;
  int maximum = 0;
  allRankResultRange(commonStatus, &minimum, &maximum);
  EXPECT_EQ(minimum, static_cast<int>(flagcxSystemError));
  EXPECT_EQ(maximum, static_cast<int>(flagcxSystemError));

  // The same tags are reusable after the failure result has converged; no
  // stale status may leak into a later registration/cleanup phase.
  commonStatus = flagcxSystemError;
  result =
      flagcxSymConvergeStatus(comm->heteroComm, flagcxSuccess, &commonStatus);
  ASSERT_TRUE(allRanksSucceeded(result));
  allRankResultRange(commonStatus, &minimum, &maximum);
  EXPECT_EQ(minimum, static_cast<int>(flagcxSuccess));
  EXPECT_EQ(maximum, static_cast<int>(flagcxSuccess));
}

TEST_F(SymMemTest, RemotePeersWithoutNetworkDoNotPublishWindow) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_REMOTE_NO_NET"))
    GTEST_SKIP()
        << "Runs in the logical multi-node network-disabled invocation";
  ASSERT_TRUE(envEnabled("FLAGCX_VMM_ENABLE"));
  ASSERT_TRUE(envEnabled("FLAGCX_IB_DISABLE"));
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(comm->heteroComm, nullptr);
  ASSERT_LT(comm->heteroComm->localRanks, comm->heteroComm->nRanks);

  void *buffer = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&buffer, size)));
  const int handlesBefore = comm->heteroComm->oneSideHandleCount;
  const uint64_t fdExchangesBefore = comm->heteroComm->symWindowFdExchangeCount;
  const uint64_t mrExchangesBefore =
      comm->heteroComm->oneSideDataMetadataExchangeCount;

  flagcxWindow_t window = nullptr;
  flagcxResult_t result = flagcxCommWindowRegister(comm, buffer, size, &window,
                                                   FLAGCX_WIN_COLL_SYMMETRIC);
  int minimum = 0;
  int maximum = 0;
  allRankResultRange(result, &minimum, &maximum);
  EXPECT_EQ(minimum, static_cast<int>(flagcxNotSupported));
  EXPECT_EQ(maximum, static_cast<int>(flagcxNotSupported));
  EXPECT_EQ(window, nullptr);
  EXPECT_EQ(comm->heteroComm->symWindows, nullptr);
  EXPECT_EQ(comm->heteroComm->pendingSymCleanup, nullptr);
  EXPECT_EQ(comm->heteroComm->oneSideHandleCount, handlesBefore);
  EXPECT_EQ(comm->heteroComm->symWindowFdExchangeCount, fdExchangesBefore);
  EXPECT_EQ(comm->heteroComm->oneSideDataMetadataExchangeCount,
            mrExchangesBefore);
  EXPECT_EQ(flagcxMemFree(buffer), flagcxSuccess);
}

TEST_F(SymMemTest, RankLocalIpcFailureUsesNetworkMrFallback) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_LOCAL_IPC_NET_FALLBACK"))
    GTEST_SKIP() << "Runs in the local IPC + NET fallback invocation";
  ASSERT_FALSE(envEnabled("FLAGCX_VMM_ENABLE"));
  ASSERT_FALSE(envEnabled("FLAGCX_IB_DISABLE"));
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(comm->heteroComm, nullptr);
  ASSERT_EQ(comm->heteroComm->localRanks, comm->heteroComm->nRanks);
  ASSERT_GT(comm->nranks, 1);
  ASSERT_NE(deviceAdaptor, nullptr);
  ASSERT_NE(deviceAdaptor->ipcMemHandleGet, nullptr);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  void *buffer = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&buffer, size)));

  auto originalIpcMemHandleGet = deviceAdaptor->ipcMemHandleGet;
  if (rank == 1)
    deviceAdaptor->ipcMemHandleGet = failIpcMemHandleGet;
  flagcxWindow_t window = nullptr;
  flagcxResult_t result = flagcxCommWindowRegister(comm, buffer, size, &window,
                                                   FLAGCX_WIN_COLL_SYMMETRIC);
  if (rank == 1)
    deviceAdaptor->ipcMemHandleGet = originalIpcMemHandleGet;

  ASSERT_TRUE(allRanksSucceeded(result));
  ASSERT_NE(window, nullptr);
  ASSERT_NE(window->defaultBase, nullptr);
  EXPECT_EQ(window->defaultBase->ipcSlot, -1);
  EXPECT_TRUE(window->defaultBase->hasNetworkMrRef);
  const int mrIndex = window->defaultBase->mrIndex;
  ASSERT_GE(mrIndex, 0);
  ASSERT_LT(mrIndex, comm->heteroComm->oneSideHandleCount);
  ASSERT_NE(comm->heteroComm->oneSideHandles[mrIndex], nullptr);

  constexpr size_t transferSize = 128;
  constexpr size_t sourceOffset = 64;
  constexpr size_t destinationOffset = 4096;
  std::vector<uint8_t> pattern(transferSize, static_cast<uint8_t>(rank + 41));
  result = devHandle->deviceMemset(buffer, 0, size, flagcxMemDevice, stream);
  if (result == flagcxSuccess)
    result = devHandle->deviceMemcpy(static_cast<char *>(buffer) + sourceOffset,
                                     pattern.data(), transferSize,
                                     flagcxMemcpyHostToDevice, stream);
  if (result == flagcxSuccess)
    result = devHandle->streamSynchronize(stream);
  ASSERT_TRUE(allRanksSucceeded(result));
  MPI_Barrier(MPI_COMM_WORLD);

  const int peer = (rank + 1) % comm->nranks;
  uint64_t before = 0;
  result = flagcxReadCounter(comm, &before);
  if (result == flagcxSuccess)
    result = flagcxGet(comm, peer, sourceOffset, destinationOffset,
                       transferSize, mrIndex, mrIndex);
  if (result == flagcxSuccess)
    result = flagcxWaitCounter(comm, before + 1);
  ASSERT_TRUE(allRanksSucceeded(result));

  std::vector<uint8_t> readBack(transferSize, 0);
  result = devHandle->deviceMemcpy(
      readBack.data(), static_cast<char *>(buffer) + destinationOffset,
      transferSize, flagcxMemcpyDeviceToHost, nullptr);
  ASSERT_TRUE(allRanksSucceeded(result));
  EXPECT_EQ(readBack, std::vector<uint8_t>(transferSize,
                                           static_cast<uint8_t>(peer + 41)));

  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowDeregister(comm, window)));
  EXPECT_EQ(flagcxMemFree(buffer), flagcxSuccess);
}

TEST_F(SymMemTest, P2pDisabledLocalWindowAcquiresNetworkMrBeforePublish) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_LOCAL_IPC_NET_FALLBACK"))
    GTEST_SKIP() << "Runs in the local P2P-disabled NET invocation";
  ASSERT_FALSE(envEnabled("FLAGCX_VMM_ENABLE"));
  ASSERT_FALSE(envEnabled("FLAGCX_IB_DISABLE"));
  ASSERT_TRUE(envEnabled("FLAGCX_P2P_DISABLE"));
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(comm->heteroComm, nullptr);
  ASSERT_EQ(comm->heteroComm->localRanks, comm->heteroComm->nRanks);
  ASSERT_GT(comm->nranks, 1);

  void *buffer = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&buffer, size)));

  flagcxWindow_t window = nullptr;
  flagcxResult_t result = flagcxCommWindowRegister(comm, buffer, size, &window,
                                                   FLAGCX_WIN_COLL_SYMMETRIC);
  ASSERT_TRUE(allRanksSucceeded(result));
  ASSERT_NE(window, nullptr);
  ASSERT_NE(window->defaultBase, nullptr);
  EXPECT_TRUE(window->defaultBase->hasNetworkMrRef);
  EXPECT_GE(window->defaultBase->mrIndex, 0);

  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowDeregister(comm, window)));
  EXPECT_EQ(flagcxMemFree(buffer), flagcxSuccess);
}

TEST_F(SymMemTest, RegisterSymmetricWindow) {
  flagcxWindow_t win = nullptr;

  flagcxResult_t res = flagcxCommWindowRegister(comm, devBuff, size, &win,
                                                FLAGCX_WIN_COLL_SYMMETRIC);
  ASSERT_EQ(res, flagcxSuccess);
  ASSERT_NE(win, nullptr);
  EXPECT_EQ(win->isSymmetricDefault, 1);
  EXPECT_NE(win->defaultBase, nullptr);

  MPI_Barrier(MPI_COMM_WORLD);

  res = flagcxCommWindowDeregister(comm, win);
  EXPECT_EQ(res, flagcxSuccess);
}

// ---------------------------------------------------------------------------
// Register with FLAGCX_WIN_DEFAULT (non-symmetric)
// ---------------------------------------------------------------------------

TEST_F(SymMemTest, RegisterDefaultFlag) {
  flagcxWindow_t win = nullptr;

  flagcxResult_t res =
      flagcxCommWindowRegister(comm, devBuff, size, &win, FLAGCX_WIN_DEFAULT);
  // On non-homo path with default flag, win may be NULL (no sym heap created)
  EXPECT_EQ(res, flagcxSuccess);

  MPI_Barrier(MPI_COMM_WORLD);

  if (win != nullptr) {
    flagcxCommWindowDeregister(comm, win);
  }
}

// ---------------------------------------------------------------------------
// Deregister after register — basic lifecycle
// ---------------------------------------------------------------------------

TEST_F(SymMemTest, DeregisterAfterRegister) {
  flagcxWindow_t win = nullptr;

  ASSERT_EQ(flagcxCommWindowRegister(comm, devBuff, size, &win,
                                     FLAGCX_WIN_COLL_SYMMETRIC),
            flagcxSuccess);
  ASSERT_NE(win, nullptr);

  MPI_Barrier(MPI_COMM_WORLD);

  EXPECT_EQ(flagcxCommWindowDeregister(comm, win), flagcxSuccess);

  MPI_Barrier(MPI_COMM_WORLD);
}

// ---------------------------------------------------------------------------
// Deregister NULL — should be safe
// ---------------------------------------------------------------------------

TEST_F(SymMemTest, DeregisterNullWindow) {
  EXPECT_EQ(flagcxCommWindowDeregister(comm, nullptr), flagcxSuccess);
}

// ---------------------------------------------------------------------------
// Verify sym window fields after registration
// ---------------------------------------------------------------------------

TEST_F(SymMemTest, SymWindowFields) {
  flagcxWindow_t win = nullptr;

  ASSERT_EQ(flagcxCommWindowRegister(comm, devBuff, size, &win,
                                     FLAGCX_WIN_COLL_SYMMETRIC),
            flagcxSuccess);
  ASSERT_NE(win, nullptr);
  ASSERT_NE(win->defaultBase, nullptr);

  flagcxSymWindow_t d = win->defaultBase;
  EXPECT_EQ(d->heapSize, size);
  EXPECT_GT(d->localRanks, 0);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxCommWindowDeregister(comm, win);
}

// ---------------------------------------------------------------------------
// Register multiple windows on different buffers
// ---------------------------------------------------------------------------

TEST_F(SymMemTest, RegisterMultipleWindows) {
  flagcxWindow_t win1 = nullptr;
  flagcxWindow_t win2 = nullptr;

  ASSERT_EQ(flagcxCommWindowRegister(comm, devBuff, size, &win1,
                                     FLAGCX_WIN_COLL_SYMMETRIC),
            flagcxSuccess);
  ASSERT_EQ(flagcxCommWindowRegister(comm, devBuff2, size, &win2,
                                     FLAGCX_WIN_COLL_SYMMETRIC),
            flagcxSuccess);

  ASSERT_NE(win1, nullptr);
  ASSERT_NE(win2, nullptr);
  EXPECT_NE(win1, win2);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxCommWindowDeregister(comm, win1);
  flagcxCommWindowDeregister(comm, win2);
}

TEST_F(SymMemTest, RepeatedRegisterDeregister) {
  int stableHandleCount = -1;
  for (int iteration = 0; iteration < 3; iteration++) {
    flagcxWindow_t win = nullptr;
    ASSERT_EQ(flagcxCommWindowRegister(comm, devBuff, size, &win,
                                       FLAGCX_WIN_COLL_SYMMETRIC),
              flagcxSuccess);
    ASSERT_NE(win, nullptr);
    ASSERT_NE(win->defaultBase, nullptr);
    EXPECT_TRUE(win->defaultBase->published);
    MPI_Barrier(MPI_COMM_WORLD);
    ASSERT_EQ(flagcxCommWindowDeregister(comm, win), flagcxSuccess);
    EXPECT_EQ(comm->heteroComm->symWindows, nullptr);
    if (envEnabled("FLAGCX_CI_REQUIRE_NET_MR")) {
      if (stableHandleCount < 0)
        stableHandleCount = comm->heteroComm->oneSideHandleCount;
      else
        EXPECT_EQ(comm->heteroComm->oneSideHandleCount, stableHandleCount)
            << "deregistered MR slots should be reused";
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

TEST_F(SymMemTest,
       RankLocalVmmPrepareFailureConvergesBeforeFdExchangeAndRetries) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_VMM") ||
      envEnabled("FLAGCX_CI_REQUIRE_NET_MR"))
    GTEST_SKIP() << "Runs in the required VMM-local invocation";

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  flagcxComm_t testComm = nullptr;
  ASSERT_TRUE(allRanksSucceeded(createTestComm(&testComm)));
  ASSERT_NE(testComm, nullptr);
  ASSERT_NE(testComm->heteroComm, nullptr);
  void *buffer = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&buffer, size)));

  auto originalSymPhysAlloc = deviceAdaptor->symPhysAlloc;
  ASSERT_NE(originalSymPhysAlloc, nullptr);
  if (rank == 1)
    deviceAdaptor->symPhysAlloc = failSymPhysAlloc;

  flagcxWindow_t fallbackWindow = nullptr;
  flagcxResult_t result = flagcxCommWindowRegister(
      testComm, buffer, size, &fallbackWindow, FLAGCX_WIN_COLL_SYMMETRIC);
  if (rank == 1)
    deviceAdaptor->symPhysAlloc = originalSymPhysAlloc;

  int minimum = 0;
  int maximum = 0;
  allRankResultRange(result, &minimum, &maximum);
  ASSERT_EQ(minimum, maximum);
  EXPECT_EQ(minimum, static_cast<int>(flagcxSystemError));
  EXPECT_EQ(testComm->heteroComm->symWindowFdExchangeCount, 0u);
  EXPECT_EQ(fallbackWindow, nullptr);
  EXPECT_EQ(testComm->heteroComm->symWindows, nullptr);
  EXPECT_EQ(testComm->heteroComm->pendingSymCleanup, nullptr);

  flagcxWindow_t retryWindow = nullptr;
  result = flagcxCommWindowRegister(testComm, buffer, size, &retryWindow,
                                    FLAGCX_WIN_COLL_SYMMETRIC);
  ASSERT_TRUE(allRanksSucceeded(result));
  ASSERT_NE(retryWindow, nullptr);
  ASSERT_NE(retryWindow->defaultBase, nullptr);
  EXPECT_TRUE(retryWindow->defaultBase->hasFlatMapping);
  EXPECT_EQ(testComm->heteroComm->symWindowFdExchangeCount, 1u);
  ASSERT_TRUE(
      allRanksSucceeded(flagcxCommWindowDeregister(testComm, retryWindow)));
  ASSERT_TRUE(allRanksSucceeded(flagcxCommDestroy(testComm)));
  EXPECT_EQ(flagcxMemFree(buffer), flagcxSuccess);
}

TEST_F(SymMemTest, VmmFlatFallbackPreservesMrRoute) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_VMM") ||
      !envEnabled("FLAGCX_CI_REQUIRE_NET_MR"))
    GTEST_SKIP() << "Runs in the required VMM + NET invocation";

  flagcxComm_t testComm = nullptr;
  ASSERT_TRUE(allRanksSucceeded(createTestComm(&testComm)));
  ASSERT_NE(testComm, nullptr);
  ASSERT_NE(testComm->heteroComm, nullptr);
  ASSERT_GT(testComm->heteroComm->localRanks, 1);
  void *buffer = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&buffer, size)));

  auto originalSymFlatMap = deviceAdaptor->symFlatMap;
  ASSERT_NE(originalSymFlatMap, nullptr);
  if (testComm->heteroComm->localRank == 1)
    deviceAdaptor->symFlatMap = failSymFlatMap;

  flagcxWindow_t fallbackWindow = nullptr;
  flagcxResult_t result = flagcxCommWindowRegister(
      testComm, buffer, size, &fallbackWindow, FLAGCX_WIN_COLL_SYMMETRIC);
  deviceAdaptor->symFlatMap = originalSymFlatMap;

  ASSERT_TRUE(allRanksSucceeded(result));
  ASSERT_NE(fallbackWindow, nullptr);
  ASSERT_NE(fallbackWindow->defaultBase, nullptr);
  EXPECT_TRUE(fallbackWindow->defaultBase->isVmmAllocation);
  EXPECT_FALSE(fallbackWindow->defaultBase->hasFlatMapping);
  EXPECT_EQ(fallbackWindow->defaultBase->ipcSlot, -1);
  EXPECT_TRUE(fallbackWindow->defaultBase->hasNetworkMrRef);
  const int mrIndex = fallbackWindow->defaultBase->mrIndex;
  ASSERT_GE(mrIndex, 0);
  ASSERT_LT(mrIndex, testComm->heteroComm->oneSideHandleCount);
  auto *handle = testComm->heteroComm->oneSideHandles[mrIndex];
  ASSERT_NE(handle, nullptr);
  EXPECT_TRUE(vmmMrRouteMatches(handle->registrationRoute));

  ASSERT_TRUE(
      allRanksSucceeded(flagcxCommWindowDeregister(testComm, fallbackWindow)));
  ASSERT_TRUE(allRanksSucceeded(flagcxCommDestroy(testComm)));
  EXPECT_EQ(flagcxMemFree(buffer), flagcxSuccess);
}

TEST_F(SymMemTest, DirectGdrVmmPreservesMrRoute) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_VMM") ||
      !envEnabled("FLAGCX_CI_REQUIRE_NET_MR"))
    GTEST_SKIP() << "Runs in the required VMM + NET invocation";

  ASSERT_NE(deviceAdaptor, nullptr);
  ASSERT_NE(deviceAdaptor->gdrMemAlloc, nullptr);
  ASSERT_NE(deviceAdaptor->gdrMemFree, nullptr);

  // Custom-op staged buffers are allocated directly through gdrMemAlloc and
  // intentionally do not belong to the public flagcxMemAlloc registry.
  void *directBuffer = nullptr;
  flagcxResult_t result =
      deviceAdaptor->gdrMemAlloc(&directBuffer, size, nullptr);
  ASSERT_TRUE(allRanksSucceeded(result));
  ASSERT_NE(directBuffer, nullptr);
  EXPECT_FALSE(flagcxOneSideRegistryRangeIsVmm(directBuffer, size));

  flagcxWindow_t window = nullptr;
  result = flagcxCommWindowRegister(comm, directBuffer, size, &window,
                                    FLAGCX_WIN_COLL_SYMMETRIC);
  ASSERT_TRUE(allRanksSucceeded(result));
  ASSERT_NE(window, nullptr);
  ASSERT_NE(window->defaultBase, nullptr);
  EXPECT_TRUE(window->defaultBase->isVmmAllocation);
  const int mrIndex = window->defaultBase->mrIndex;
  ASSERT_GE(mrIndex, 0);
  ASSERT_LT(mrIndex, comm->heteroComm->oneSideHandleCount);
  auto *handle = comm->heteroComm->oneSideHandles[mrIndex];
  ASSERT_NE(handle, nullptr);
  EXPECT_TRUE(vmmMrRouteMatches(handle->registrationRoute));

  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowDeregister(comm, window)));
  ASSERT_TRUE(
      allRanksSucceeded(deviceAdaptor->gdrMemFree(directBuffer, nullptr)));
}

TEST_F(SymMemTest, FullMeshRoundFailureConvergesAndRetries) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_NET_MR"))
    GTEST_SKIP() << "Runs in the required NET invocations";

  int rank = 0;
  int nranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  if (nranks < 4)
    GTEST_SKIP() << "requires four ranks to exercise overlapping rounds";

  flagcxComm_t testComm = nullptr;
  ASSERT_TRUE(allRanksSucceeded(createTestComm(&testComm)));
  ASSERT_NE(testComm, nullptr);
  ASSERT_NE(testComm->heteroComm, nullptr);
  void *buffer = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&buffer, size)));

  auto *net = testComm->heteroComm->netAdaptor;
  ASSERT_NE(net, nullptr);
  ASSERT_NE(net->connect, nullptr);
  ASSERT_NE(net->resetConnect, nullptr);
  if (rank == 1) {
    savedFullMeshConnect = net->connect;
    savedFullMeshResetConnect = net->resetConnect;
    firstFullMeshHandle = nullptr;
    injectedFullMeshConnectFailures = 0;
    injectedFullMeshResetFailures = 0;
    net->connect = failSecondFullMeshConnect;
    net->resetConnect = failFirstFullMeshReset;
  }

  flagcxResult_t result = flagcxOneSideRegister(testComm, buffer, size);
  if (rank == 1) {
    net->connect = savedFullMeshConnect;
    net->resetConnect = savedFullMeshResetConnect;
  }

  int minimum = 0;
  int maximum = 0;
  allRankResultRange(result, &minimum, &maximum);
  EXPECT_EQ(minimum, static_cast<int>(flagcxSystemError));
  EXPECT_EQ(maximum, minimum);
  int injected = rank == 1 && injectedFullMeshConnectFailures > 0 ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &injected, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  EXPECT_EQ(injected, 1);
  EXPECT_EQ(testComm->heteroComm->pendingOneSideMeshCleanup != nullptr,
            rank == 1);
  EXPECT_EQ(testComm->heteroComm->oneSideHandleCount, 0);
  EXPECT_EQ(testComm->heteroComm->oneSideHandles, nullptr);
  EXPECT_EQ(testComm->heteroComm->oneSideDataMetadataExchangeCount, 0u);

  MPI_Barrier(MPI_COMM_WORLD);
  result = flagcxOneSideRegister(testComm, buffer, size);
  ASSERT_TRUE(allRanksSucceeded(result));
  EXPECT_EQ(testComm->heteroComm->pendingOneSideMeshCleanup, nullptr);
  EXPECT_EQ(testComm->heteroComm->oneSideHandleCount, 1);
  ASSERT_TRUE(allRanksSucceeded(flagcxCommDestroy(testComm)));
  EXPECT_EQ(flagcxMemFree(buffer), flagcxSuccess);
}

TEST_F(SymMemTest, RankLocalMrFailureDoesNotPublishPartialWindow) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_NET_MR"))
    GTEST_SKIP() << "Runs in the required NET invocations";

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  flagcxComm_t testComm = nullptr;
  ASSERT_TRUE(allRanksSucceeded(createTestComm(&testComm)));
  ASSERT_NE(testComm, nullptr);
  ASSERT_NE(testComm->heteroComm, nullptr);
  void *buffer = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&buffer, size)));

  auto *net = testComm->heteroComm->netAdaptor;
  ASSERT_NE(net, nullptr);
  auto originalRegMr = net->regMr;
  auto originalRegMrDmaBuf = net->regMrDmaBuf;
  ASSERT_NE(originalRegMr, nullptr);
  if (rank == 1) {
    net->regMr = failRegMr;
    net->regMrDmaBuf = failRegMrDmaBuf;
  }

  flagcxWindow_t failedWindow = nullptr;
  flagcxResult_t result = flagcxCommWindowRegister(
      testComm, buffer, size, &failedWindow, FLAGCX_WIN_COLL_SYMMETRIC);
  if (rank == 1) {
    net->regMr = originalRegMr;
    net->regMrDmaBuf = originalRegMrDmaBuf;
  }

  int minimum = 0;
  int maximum = 0;
  allRankResultRange(result, &minimum, &maximum);
  EXPECT_EQ(minimum, maximum);
  EXPECT_EQ(minimum, static_cast<int>(flagcxSystemError));
  EXPECT_EQ(failedWindow, nullptr);
  EXPECT_EQ(testComm->heteroComm->symWindows, nullptr);
  EXPECT_EQ(testComm->heteroComm->oneSideHandleCount, 0);
  EXPECT_EQ(testComm->heteroComm->oneSideDataMetadataExchangeCount, 0u);

  flagcxWindow_t retryWindow = nullptr;
  result = flagcxCommWindowRegister(testComm, buffer, size, &retryWindow,
                                    FLAGCX_WIN_COLL_SYMMETRIC);
  ASSERT_TRUE(allRanksSucceeded(result));
  ASSERT_NE(retryWindow, nullptr);
  EXPECT_TRUE(retryWindow->defaultBase->hasNetworkMrRef);
  if (retryWindow->defaultBase->hasFlatMapping) {
    const int mrIndex = retryWindow->defaultBase->mrIndex;
    ASSERT_GE(mrIndex, 0);
    ASSERT_LT(mrIndex, testComm->heteroComm->oneSideHandleCount);
    auto *handle = testComm->heteroComm->oneSideHandles[mrIndex];
    ASSERT_NE(handle, nullptr);

    EXPECT_TRUE(vmmMrRouteMatches(handle->registrationRoute));
  }

  auto originalDeregMr = net->deregMr;
  ASSERT_NE(originalDeregMr, nullptr);
  if (rank == 1)
    net->deregMr = failDeregMr;
  result = flagcxCommWindowDeregister(testComm, retryWindow);
  if (rank == 1)
    net->deregMr = originalDeregMr;
  allRankResultRange(result, &minimum, &maximum);
  EXPECT_EQ(minimum, static_cast<int>(flagcxSystemError));
  EXPECT_EQ(maximum, static_cast<int>(flagcxSystemError));

  // The public operation is collective: every rank reports the same failure
  // and retries the same window. Ranks whose local MR was already released
  // retain the mapping but skip the MR release on retry; the failing rank
  // retains the exact MR handle and releases it after fault injection clears.
  ASSERT_NE(retryWindow->defaultBase, nullptr);
  if (rank == 1) {
    EXPECT_TRUE(retryWindow->defaultBase->hasNetworkMrRef);
  } else {
    EXPECT_FALSE(retryWindow->defaultBase->hasNetworkMrRef);
  }
  EXPECT_EQ(testComm->heteroComm->symWindows, retryWindow->defaultBase);
  flagcxResult_t retryResult =
      flagcxCommWindowDeregister(testComm, retryWindow);
  ASSERT_TRUE(allRanksSucceeded(retryResult));
  ASSERT_TRUE(allRanksSucceeded(flagcxCommDestroy(testComm)));
  EXPECT_EQ(flagcxMemFree(buffer), flagcxSuccess);
}

TEST_F(SymMemTest, MrRollbackFailureRetainsWindowLeaseUntilRetry) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_NET_MR"))
    GTEST_SKIP() << "Runs in the required NET invocations";

  int rank = 0;
  int nranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  if (nranks < 2)
    GTEST_SKIP() << "requires at least two ranks";

  flagcxComm_t testComm = nullptr;
  ASSERT_TRUE(allRanksSucceeded(createTestComm(&testComm)));
  ASSERT_NE(testComm, nullptr);
  ASSERT_NE(testComm->heteroComm, nullptr);
  void *buffer = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&buffer, size)));

  auto *net = testComm->heteroComm->netAdaptor;
  ASSERT_NE(net, nullptr);
  auto originalRegMr = net->regMr;
  auto originalRegMrDmaBuf = net->regMrDmaBuf;
  auto originalDeregMr = net->deregMr;
  ASSERT_NE(originalRegMr, nullptr);
  ASSERT_NE(originalDeregMr, nullptr);
  if (rank == 1) {
    net->regMr = failRegMr;
    net->regMrDmaBuf = failRegMrDmaBuf;
  }
  if (rank == 0)
    net->deregMr = failDeregMr;

  flagcxWindow_t cleanupToken = nullptr;
  flagcxResult_t result = flagcxCommWindowRegister(
      testComm, buffer, size, &cleanupToken, FLAGCX_WIN_COLL_SYMMETRIC);
  net->regMr = originalRegMr;
  net->regMrDmaBuf = originalRegMrDmaBuf;
  net->deregMr = originalDeregMr;

  int minimum = 0;
  int maximum = 0;
  allRankResultRange(result, &minimum, &maximum);
  EXPECT_EQ(minimum, static_cast<int>(flagcxSystemError));
  EXPECT_EQ(maximum, static_cast<int>(flagcxSystemError));
  ASSERT_NE(cleanupToken, nullptr);
  ASSERT_NE(cleanupToken->defaultBase, nullptr);
  EXPECT_TRUE(cleanupToken->defaultBase->hasPendingNetworkCleanup);
  EXPECT_EQ(cleanupToken->defaultBase->state, flagcxSymWindowCleanupRequired);
  EXPECT_EQ(testComm->heteroComm->symWindows, nullptr);
  EXPECT_NE(testComm->heteroComm->pendingSymCleanup, nullptr);
  EXPECT_EQ(testComm->heteroComm->oneSideHandleCount, 0);
  EXPECT_EQ(testComm->heteroComm->oneSideDataMetadataExchangeCount, 0u);

  // The failed MR still refers to this allocation. Neither a direct free nor
  // a new collective registration may pass until every rank retries the
  // cleanup token after the injected deregistration error is removed.
  EXPECT_EQ(flagcxMemFree(buffer), flagcxInvalidUsage);
  flagcxWindow_t blockedWindow = nullptr;
  result = flagcxCommWindowRegister(testComm, buffer, size, &blockedWindow,
                                    FLAGCX_WIN_COLL_SYMMETRIC);
  EXPECT_EQ(result, flagcxInvalidUsage);
  EXPECT_EQ(blockedWindow, nullptr);

  ASSERT_TRUE(
      allRanksSucceeded(flagcxCommWindowDeregister(testComm, cleanupToken)));
  EXPECT_EQ(testComm->heteroComm->pendingSymCleanup, nullptr);

  flagcxWindow_t retryWindow = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowRegister(
      testComm, buffer, size, &retryWindow, FLAGCX_WIN_COLL_SYMMETRIC)));
  ASSERT_NE(retryWindow, nullptr);
  ASSERT_TRUE(
      allRanksSucceeded(flagcxCommWindowDeregister(testComm, retryWindow)));
  ASSERT_TRUE(allRanksSucceeded(flagcxCommDestroy(testComm)));
  EXPECT_EQ(flagcxMemFree(buffer), flagcxSuccess);
}

TEST_F(SymMemTest, StrictDmaBufUnsupportedDoesNotPublishWindow) {
  if (!envEnabled("FLAGCX_CI_EXPECT_VMM_DMABUF_UNSUPPORTED"))
    GTEST_SKIP() << "Runs only on a provider without DMA-BUF MR registration";
  ASSERT_TRUE(envEnabled("FLAGCX_CI_REQUIRE_VMM"));
  ASSERT_TRUE(envEnabled("FLAGCX_CI_REQUIRE_NET_MR"));
  const char *mode = std::getenv("FLAGCX_VMM_MR_MODE");
  ASSERT_NE(mode, nullptr);
  ASSERT_STREQ(mode, "dmabuf");

  flagcxComm_t testComm = nullptr;
  ASSERT_TRUE(allRanksSucceeded(createTestComm(&testComm)));
  ASSERT_NE(testComm, nullptr);
  ASSERT_NE(testComm->heteroComm, nullptr);
  ASSERT_NE(testComm->heteroComm->netAdaptor, nullptr);
  EXPECT_EQ(testComm->heteroComm->netAdaptor->regMrDmaBuf, nullptr);

  void *buffer = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&buffer, size)));
  const uint64_t exchangesBefore =
      testComm->heteroComm->oneSideDataMetadataExchangeCount;
  flagcxWindow_t window = nullptr;
  flagcxResult_t result = flagcxCommWindowRegister(
      testComm, buffer, size, &window, FLAGCX_WIN_COLL_SYMMETRIC);

  int minimum = 0;
  int maximum = 0;
  allRankResultRange(result, &minimum, &maximum);
  EXPECT_EQ(minimum, static_cast<int>(flagcxNotSupported));
  EXPECT_EQ(maximum, static_cast<int>(flagcxNotSupported));
  EXPECT_EQ(window, nullptr);
  EXPECT_EQ(testComm->heteroComm->symWindows, nullptr);
  EXPECT_EQ(testComm->heteroComm->oneSideHandleCount, 0);
  EXPECT_EQ(testComm->heteroComm->oneSideDataMetadataExchangeCount,
            exchangesBefore);

  ASSERT_TRUE(allRanksSucceeded(flagcxCommDestroy(testComm)));
  EXPECT_EQ(flagcxMemFree(buffer), flagcxSuccess);
}

TEST_F(SymMemTest, VmmNetRouteCapabilityUnion) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_VMM_ROUTE_UNION"))
    GTEST_SKIP() << "Runs in the VMM NET route-capability invocation";
  ASSERT_TRUE(envEnabled("FLAGCX_CI_REQUIRE_VMM"));
  ASSERT_TRUE(envEnabled("FLAGCX_CI_REQUIRE_NET_MR"));

  ScopedEnvVar routeMode("FLAGCX_VMM_MR_MODE", "auto");
  const char *modes[] = {"dmabuf", "va"};
  const flagcxVmmMrRoute_t expectedRoutes[] = {FLAGCX_VMM_MR_ROUTE_DMABUF,
                                               FLAGCX_VMM_MR_ROUTE_VA};
  bool routeSucceeded[2] = {false, false};

  for (int routeIndex = 0; routeIndex < 2; routeIndex++) {
    SCOPED_TRACE(modes[routeIndex]);
    routeMode.set(modes[routeIndex]);

    flagcxComm_t testComm = nullptr;
    ASSERT_TRUE(allRanksSucceeded(createTestComm(&testComm)));
    ASSERT_NE(testComm, nullptr);
    ASSERT_NE(testComm->heteroComm, nullptr);
    ASSERT_NE(testComm->heteroComm->netAdaptor, nullptr);

    void *buffer = nullptr;
    ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&buffer, size)));
    const int handlesBefore = testComm->heteroComm->oneSideHandleCount;
    const uint64_t exchangesBefore =
        testComm->heteroComm->oneSideDataMetadataExchangeCount;
    flagcxWindow_t window = nullptr;
    flagcxResult_t result = flagcxCommWindowRegister(
        testComm, buffer, size, &window, FLAGCX_WIN_COLL_SYMMETRIC);

    int minimum = 0;
    int maximum = 0;
    allRankResultRange(result, &minimum, &maximum);
    ASSERT_EQ(minimum, maximum)
        << modes[routeIndex] << " result diverged across ranks";
    ASSERT_TRUE(result == flagcxSuccess || result == flagcxNotSupported)
        << modes[routeIndex] << " returned unexpected result "
        << static_cast<int>(result);

    if (result == flagcxSuccess) {
      routeSucceeded[routeIndex] = true;
      ASSERT_NE(window, nullptr);
      ASSERT_NE(window->defaultBase, nullptr);
      ASSERT_TRUE(window->defaultBase->hasNetworkMrRef);
      const int mrIndex = window->defaultBase->mrIndex;
      ASSERT_GE(mrIndex, 0);
      ASSERT_LT(mrIndex, testComm->heteroComm->oneSideHandleCount);
      auto *handle = testComm->heteroComm->oneSideHandles[mrIndex];
      ASSERT_NE(handle, nullptr);
      EXPECT_EQ(handle->registrationRoute,
                static_cast<uint8_t>(expectedRoutes[routeIndex]));
      ASSERT_TRUE(
          allRanksSucceeded(flagcxCommWindowDeregister(testComm, window)));
    } else {
      EXPECT_EQ(window, nullptr);
      EXPECT_EQ(testComm->heteroComm->symWindows, nullptr);
      EXPECT_EQ(testComm->heteroComm->pendingSymCleanup, nullptr);
      EXPECT_EQ(testComm->heteroComm->oneSideHandleCount, handlesBefore);
      EXPECT_EQ(testComm->heteroComm->oneSideDataMetadataExchangeCount,
                exchangesBefore);
    }

    ASSERT_TRUE(allRanksSucceeded(flagcxCommDestroy(testComm)));
    EXPECT_EQ(flagcxMemFree(buffer), flagcxSuccess);
  }

  const int localMask =
      (routeSucceeded[0] ? 1 : 0) | (routeSucceeded[1] ? 2 : 0);
  int minimumMask = 0;
  int maximumMask = 0;
  MPI_Allreduce(&localMask, &minimumMask, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&localMask, &maximumMask, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  ASSERT_EQ(minimumMask, maximumMask);
  if (!envEnabled("FLAGCX_CI_ALLOW_VMM_NET_UNSUPPORTED")) {
    EXPECT_NE(minimumMask, 0)
        << "neither DMA-BUF nor VA can register a VMM allocation";
  }
}

TEST_F(SymMemTest, AsymmetricDeregisterUsesCollectivePublishSlot) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_NET_MR"))
    GTEST_SKIP() << "Runs in the required NET invocations";

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  flagcxComm_t testComm = nullptr;
  ASSERT_TRUE(allRanksSucceeded(createTestComm(&testComm)));
  ASSERT_NE(testComm, nullptr);
  auto *net = testComm->heteroComm->netAdaptor;
  ASSERT_NE(net, nullptr);
  ASSERT_NE(net->deregMr, nullptr);

  void *anchorBuffer = nullptr;
  void *victimBuffer = nullptr;
  void *replacementBuffer = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&anchorBuffer, size)));
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&victimBuffer, size)));
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&replacementBuffer, size)));

  flagcxWindow_t anchor = nullptr;
  flagcxWindow_t victim = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowRegister(
      testComm, anchorBuffer, size, &anchor, FLAGCX_WIN_COLL_SYMMETRIC)));
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowRegister(
      testComm, victimBuffer, size, &victim, FLAGCX_WIN_COLL_SYMMETRIC)));
  ASSERT_NE(anchor, nullptr);
  ASSERT_NE(victim, nullptr);
  ASSERT_EQ(anchor->defaultBase->mrIndex, 0);
  ASSERT_EQ(victim->defaultBase->mrIndex, 1);

  auto originalDeregMr = net->deregMr;
  if (rank == 1)
    net->deregMr = failDeregMr;
  flagcxResult_t result = flagcxCommWindowDeregister(testComm, victim);
  if (rank == 1)
    net->deregMr = originalDeregMr;
  int minimum = 0;
  int maximum = 0;
  allRankResultRange(result, &minimum, &maximum);
  ASSERT_EQ(minimum, static_cast<int>(flagcxSystemError));
  ASSERT_EQ(maximum, static_cast<int>(flagcxSystemError));

  // The failed rank still owns slot 1 while the other rank has released it.
  // Public/non-window registration must converge that mismatch instead of
  // letting one rank return early while the other enters metadata collectives.
  result = flagcxOneSideRegister(testComm, victimBuffer, size);
  allRankResultRange(result, &minimum, &maximum);
  ASSERT_EQ(minimum, static_cast<int>(flagcxInvalidUsage));
  ASSERT_EQ(maximum, static_cast<int>(flagcxInvalidUsage));

  flagcxWindow_t replacement = nullptr;
  result = flagcxCommWindowRegister(testComm, replacementBuffer, size,
                                    &replacement, FLAGCX_WIN_COLL_SYMMETRIC);
  ASSERT_TRUE(allRanksSucceeded(result));
  ASSERT_NE(replacement, nullptr);
  int minimumSlot = 0;
  int maximumSlot = 0;
  int replacementSlot = replacement->defaultBase->mrIndex;
  MPI_Allreduce(&replacementSlot, &minimumSlot, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(&replacementSlot, &maximumSlot, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  EXPECT_EQ(minimumSlot, maximumSlot);
  EXPECT_EQ(minimumSlot, 2);

  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowDeregister(testComm, victim)));
  ASSERT_TRUE(
      allRanksSucceeded(flagcxCommWindowDeregister(testComm, replacement)));
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowDeregister(testComm, anchor)));
  ASSERT_TRUE(allRanksSucceeded(flagcxCommDestroy(testComm)));
  EXPECT_EQ(flagcxMemFree(replacementBuffer), flagcxSuccess);
  EXPECT_EQ(flagcxMemFree(victimBuffer), flagcxSuccess);
  EXPECT_EQ(flagcxMemFree(anchorBuffer), flagcxSuccess);
}

TEST_F(SymMemTest, PublicRegistrationUsesVmmMrRouting) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_VMM") ||
      !envEnabled("FLAGCX_CI_REQUIRE_NET_MR"))
    GTEST_SKIP() << "Runs in the required VMM + NET invocation";

  flagcxComm_t testComm = nullptr;
  ASSERT_TRUE(allRanksSucceeded(createTestComm(&testComm)));
  ASSERT_NE(testComm, nullptr);
  void *oneSideBuffer = nullptr;
  void *commBuffer = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&oneSideBuffer, size)));
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&commBuffer, size)));

  ASSERT_TRUE(
      allRanksSucceeded(flagcxOneSideRegister(testComm, oneSideBuffer, size)));
  int oneSideIndex = findLocalMrIndex(testComm->heteroComm, oneSideBuffer);
  ASSERT_GE(oneSideIndex, 0);
  auto *oneSideHandle = testComm->heteroComm->oneSideHandles[oneSideIndex];
  ASSERT_NE(oneSideHandle, nullptr);
  EXPECT_TRUE(vmmMrRouteMatches(oneSideHandle->registrationRoute));

  void *registration = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxCommRegister(
      testComm, commBuffer, size, &registration, flagcxMemCCL)));
  ASSERT_NE(registration, nullptr);
  int commIndex = findLocalMrIndex(testComm->heteroComm, commBuffer);
  ASSERT_GE(commIndex, 0);
  auto *commHandle = testComm->heteroComm->oneSideHandles[commIndex];
  ASSERT_NE(commHandle, nullptr);
  EXPECT_TRUE(vmmMrRouteMatches(commHandle->registrationRoute));

  EXPECT_EQ(flagcxCommDeregister(testComm, registration), flagcxSuccess);
  ASSERT_TRUE(allRanksSucceeded(flagcxCommDestroy(testComm)));
  EXPECT_EQ(flagcxMemFree(commBuffer), flagcxSuccess);
  EXPECT_EQ(flagcxMemFree(oneSideBuffer), flagcxSuccess);
}

TEST_F(SymMemTest, SignalRegistrationUsesAllocationProvenance) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_VMM") ||
      !envEnabled("FLAGCX_CI_REQUIRE_NET_MR"))
    GTEST_SKIP() << "Runs in the required VMM + NET invocation";

  flagcxComm_t testComm = nullptr;
  ASSERT_TRUE(allRanksSucceeded(createTestComm(&testComm)));
  ASSERT_NE(testComm, nullptr);
  ASSERT_NE(testComm->heteroComm, nullptr);
  auto *net = testComm->heteroComm->netAdaptor;
  ASSERT_NE(net, nullptr);
  if (net->iputSignal == nullptr) {
    ASSERT_TRUE(allRanksSucceeded(flagcxCommDestroy(testComm)));
    GTEST_SKIP() << "selected provider has no atomic signal capability";
  }

  void *trackedVmmBuffer = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&trackedVmmBuffer, size)));
  ASSERT_TRUE(allRanksSucceeded(flagcxOneSideSignalRegister(
      testComm, trackedVmmBuffer, size, FLAGCX_PTR_CUDA)));
  ASSERT_NE(testComm->heteroComm->signalHandle, nullptr);
  EXPECT_TRUE(
      vmmMrRouteMatches(testComm->heteroComm->signalHandle->registrationRoute));
  ASSERT_TRUE(allRanksSucceeded(flagcxOneSideSignalDeregister(testComm)));

  // This allocation intentionally bypasses flagcxMemAlloc. VMM being enabled
  // globally must not make an ordinary external device allocation take the
  // DMA-BUF path.
  void *externalBuffer = nullptr;
  flagcxResult_t result =
      devHandle->deviceMalloc(&externalBuffer, size, flagcxMemDevice, nullptr);
  if (result == flagcxSuccess)
    result = devHandle->deviceMemset(externalBuffer, 0, size, flagcxMemDevice,
                                     nullptr);
  if (result == flagcxSuccess)
    result = devHandle->deviceSynchronize();
  ASSERT_TRUE(allRanksSucceeded(result));
  ASSERT_NE(externalBuffer, nullptr);
  ASSERT_TRUE(allRanksSucceeded(flagcxOneSideSignalRegister(
      testComm, externalBuffer, size, FLAGCX_PTR_CUDA)));
  ASSERT_NE(testComm->heteroComm->signalHandle, nullptr);
  EXPECT_EQ(testComm->heteroComm->signalHandle->registrationRoute,
            static_cast<uint8_t>(FLAGCX_VMM_MR_ROUTE_NONE));
  ASSERT_TRUE(allRanksSucceeded(flagcxOneSideSignalDeregister(testComm)));

  ASSERT_TRUE(allRanksSucceeded(flagcxCommDestroy(testComm)));
  EXPECT_EQ(devHandle->deviceFree(externalBuffer, flagcxMemDevice, nullptr),
            flagcxSuccess);
  EXPECT_EQ(flagcxMemFree(trackedVmmBuffer), flagcxSuccess);
}

TEST_F(SymMemTest, RankLocalSignalMrFailurePreservesErrorAndRetries) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_VMM") ||
      !envEnabled("FLAGCX_CI_REQUIRE_NET_MR"))
    GTEST_SKIP() << "Runs in the required VMM + NET invocation";

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  flagcxComm_t testComm = nullptr;
  ASSERT_TRUE(allRanksSucceeded(createTestComm(&testComm)));
  ASSERT_NE(testComm, nullptr);
  ASSERT_NE(testComm->heteroComm, nullptr);
  auto *net = testComm->heteroComm->netAdaptor;
  ASSERT_NE(net, nullptr);
  if (net->iputSignal == nullptr) {
    ASSERT_TRUE(allRanksSucceeded(flagcxCommDestroy(testComm)));
    GTEST_SKIP() << "selected provider has no atomic signal capability";
  }

  void *signalBuffer = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&signalBuffer, size)));
  ASSERT_TRUE(flagcxOneSideRegistryRangeIsVmm(signalBuffer, size));

  auto originalRegMr = net->regMr;
  auto originalRegMrDmaBuf = net->regMrDmaBuf;
  ASSERT_NE(originalRegMr, nullptr);
  if (rank == 1) {
    net->regMr = failRegMr;
    net->regMrDmaBuf = failRegMrDmaBuf;
  }

  flagcxResult_t result = flagcxOneSideSignalRegister(testComm, signalBuffer,
                                                      size, FLAGCX_PTR_CUDA);
  if (rank == 1) {
    net->regMr = originalRegMr;
    net->regMrDmaBuf = originalRegMrDmaBuf;
  }

  int minimum = 0;
  int maximum = 0;
  allRankResultRange(result, &minimum, &maximum);
  EXPECT_EQ(minimum, static_cast<int>(flagcxSystemError));
  EXPECT_EQ(maximum, static_cast<int>(flagcxSystemError));
  EXPECT_EQ(testComm->heteroComm->signalHandle, nullptr);
  EXPECT_EQ(testComm->heteroComm->rmaSignalBase, nullptr);
  EXPECT_EQ(testComm->heteroComm->rmaSignalSize, 0u);
  EXPECT_EQ(testComm->heteroComm->rmaSignalIpcSlot, -1);

  ASSERT_TRUE(allRanksSucceeded(flagcxOneSideSignalRegister(
      testComm, signalBuffer, size, FLAGCX_PTR_CUDA)));
  ASSERT_NE(testComm->heteroComm->signalHandle, nullptr);
  EXPECT_TRUE(
      vmmMrRouteMatches(testComm->heteroComm->signalHandle->registrationRoute));
  ASSERT_TRUE(allRanksSucceeded(flagcxOneSideSignalDeregister(testComm)));
  ASSERT_TRUE(allRanksSucceeded(flagcxCommDestroy(testComm)));
  EXPECT_EQ(flagcxMemFree(signalBuffer), flagcxSuccess);
}

TEST_F(SymMemTest, FlatMapRollbackFailureRetainsOwnershipForRetry) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_VMM") ||
      envEnabled("FLAGCX_CI_REQUIRE_NET_MR"))
    GTEST_SKIP() << "Runs in the required VMM-local invocation";

  flagcxComm_t testComm = nullptr;
  ASSERT_TRUE(allRanksSucceeded(createTestComm(&testComm)));
  ASSERT_NE(testComm, nullptr);
  ASSERT_GT(testComm->heteroComm->localRanks, 1);
  void *buffer = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&buffer, size)));

  savedSymFlatMap = deviceAdaptor->symFlatMap;
  savedSymFlatUnmap = deviceAdaptor->symFlatMappingUnmap;
  ASSERT_NE(savedSymFlatMap, nullptr);
  ASSERT_NE(savedSymFlatUnmap, nullptr);
  injectedFlatUnmapFailures = 0;
  if (testComm->heteroComm->localRank == 1)
    deviceAdaptor->symFlatMap = failSymFlatMap;
  if (testComm->heteroComm->localRank == 0)
    deviceAdaptor->symFlatMappingUnmap = failSymFlatUnmapOnce;

  flagcxWindow_t failedWindow = nullptr;
  flagcxResult_t result = flagcxCommWindowRegister(
      testComm, buffer, size, &failedWindow, FLAGCX_WIN_COLL_SYMMETRIC);
  deviceAdaptor->symFlatMap = savedSymFlatMap;
  deviceAdaptor->symFlatMappingUnmap = savedSymFlatUnmap;

  int minimum = 0;
  int maximum = 0;
  allRankResultRange(result, &minimum, &maximum);
  EXPECT_EQ(minimum, maximum);
  EXPECT_NE(minimum, static_cast<int>(flagcxSuccess));
  ASSERT_NE(failedWindow, nullptr);
  EXPECT_EQ(failedWindow->defaultBase->state, flagcxSymWindowCleanupRequired);
  EXPECT_EQ(testComm->heteroComm->symWindows, nullptr);
  EXPECT_NE(testComm->heteroComm->pendingSymCleanup, nullptr);
  EXPECT_EQ(flagcxMemFree(buffer), flagcxInvalidUsage);

  flagcxWindow_t retryWindow = nullptr;
  result = flagcxCommWindowRegister(testComm, buffer, size, &retryWindow,
                                    FLAGCX_WIN_COLL_SYMMETRIC);
  EXPECT_EQ(result, flagcxInvalidUsage);
  EXPECT_EQ(retryWindow, nullptr);
  ASSERT_TRUE(
      allRanksSucceeded(flagcxCommWindowDeregister(testComm, failedWindow)));
  EXPECT_EQ(testComm->heteroComm->pendingSymCleanup, nullptr);
  result = flagcxCommWindowRegister(testComm, buffer, size, &retryWindow,
                                    FLAGCX_WIN_COLL_SYMMETRIC);
  ASSERT_TRUE(allRanksSucceeded(result));
  ASSERT_NE(retryWindow, nullptr);
  EXPECT_EQ(testComm->heteroComm->pendingSymCleanup, nullptr);
  EXPECT_TRUE(retryWindow->defaultBase->hasFlatMapping);
  ASSERT_TRUE(
      allRanksSucceeded(flagcxCommWindowDeregister(testComm, retryWindow)));
  ASSERT_TRUE(allRanksSucceeded(flagcxCommDestroy(testComm)));
  EXPECT_EQ(flagcxMemFree(buffer), flagcxSuccess);
}

TEST_F(SymMemTest, VmmRollbackFailureConvergesBeforeMrMetadataExchange) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_VMM") ||
      !envEnabled("FLAGCX_CI_REQUIRE_NET_MR"))
    GTEST_SKIP() << "Runs in the required VMM + NET invocation";

  flagcxComm_t testComm = nullptr;
  ASSERT_TRUE(allRanksSucceeded(createTestComm(&testComm)));
  ASSERT_NE(testComm, nullptr);
  ASSERT_NE(testComm->heteroComm, nullptr);
  auto *hetero = testComm->heteroComm;
  ASSERT_GT(hetero->localRanks, 1);
  ASSERT_LT(hetero->localRanks, hetero->nRanks)
      << "test requires more than one logical node";

  void *buffer = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&buffer, size)));
  savedSymFlatMap = deviceAdaptor->symFlatMap;
  savedSymFlatUnmap = deviceAdaptor->symFlatMappingUnmap;
  ASSERT_NE(savedSymFlatMap, nullptr);
  ASSERT_NE(savedSymFlatUnmap, nullptr);
  injectedFlatUnmapFailures = 0;

  const int failingNode = hetero->rankToNode[0];
  const bool onFailingNode = hetero->node == failingNode;
  if (onFailingNode && hetero->localRank == 1)
    deviceAdaptor->symFlatMap = failSymFlatMap;
  if (onFailingNode && hetero->localRank == 0)
    deviceAdaptor->symFlatMappingUnmap = failSymFlatUnmapOnce;

  const uint64_t metadataExchangesBefore =
      hetero->oneSideDataMetadataExchangeCount;
  flagcxWindow_t failedWindow = nullptr;
  flagcxResult_t result = flagcxCommWindowRegister(
      testComm, buffer, size, &failedWindow, FLAGCX_WIN_COLL_SYMMETRIC);
  deviceAdaptor->symFlatMap = savedSymFlatMap;
  deviceAdaptor->symFlatMappingUnmap = savedSymFlatUnmap;

  int minimum = 0;
  int maximum = 0;
  allRankResultRange(result, &minimum, &maximum);
  EXPECT_EQ(minimum, static_cast<int>(flagcxRemoteError));
  EXPECT_EQ(maximum, static_cast<int>(flagcxRemoteError));
  ASSERT_NE(failedWindow, nullptr);
  EXPECT_EQ(failedWindow->defaultBase->state, flagcxSymWindowCleanupRequired);
  EXPECT_EQ(hetero->symWindows, nullptr);
  EXPECT_EQ(hetero->oneSideHandleCount, 0);
  EXPECT_EQ(hetero->oneSideDataMetadataExchangeCount, metadataExchangesBefore);
  EXPECT_NE(hetero->pendingSymCleanup, nullptr);
  EXPECT_EQ(flagcxMemFree(buffer), flagcxInvalidUsage);

  flagcxWindow_t retryWindow = nullptr;
  result = flagcxCommWindowRegister(testComm, buffer, size, &retryWindow,
                                    FLAGCX_WIN_COLL_SYMMETRIC);
  EXPECT_EQ(result, flagcxInvalidUsage);
  EXPECT_EQ(retryWindow, nullptr);
  ASSERT_TRUE(
      allRanksSucceeded(flagcxCommWindowDeregister(testComm, failedWindow)));
  EXPECT_EQ(hetero->pendingSymCleanup, nullptr);
  result = flagcxCommWindowRegister(testComm, buffer, size, &retryWindow,
                                    FLAGCX_WIN_COLL_SYMMETRIC);
  ASSERT_TRUE(allRanksSucceeded(result));
  ASSERT_NE(retryWindow, nullptr);
  EXPECT_EQ(hetero->pendingSymCleanup, nullptr);
  EXPECT_TRUE(retryWindow->defaultBase->hasFlatMapping);
  EXPECT_TRUE(retryWindow->defaultBase->hasNetworkMrRef);

  ASSERT_TRUE(
      allRanksSucceeded(flagcxCommWindowDeregister(testComm, retryWindow)));
  ASSERT_TRUE(allRanksSucceeded(flagcxCommDestroy(testComm)));
  EXPECT_EQ(flagcxMemFree(buffer), flagcxSuccess);
}

TEST_F(SymMemTest, CrossNodeCleanupFailureConvergesBeforeRelease) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_VMM") ||
      !envEnabled("FLAGCX_CI_REQUIRE_NET_MR"))
    GTEST_SKIP() << "Runs in the required VMM + NET invocation";

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  flagcxComm_t testComm = nullptr;
  ASSERT_TRUE(allRanksSucceeded(createTestComm(&testComm)));
  ASSERT_NE(testComm, nullptr);
  ASSERT_NE(testComm->heteroComm, nullptr);
  ASSERT_LT(testComm->heteroComm->localRanks, testComm->heteroComm->nRanks)
      << "test requires more than one logical node";
  void *buffer = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&buffer, size)));

  flagcxWindow_t window = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowRegister(
      testComm, buffer, size, &window, FLAGCX_WIN_COLL_SYMMETRIC)));
  ASSERT_NE(window, nullptr);
  ASSERT_NE(window->defaultBase, nullptr);
  ASSERT_TRUE(window->defaultBase->hasFlatMapping);

  savedSymFlatUnmap = deviceAdaptor->symFlatMappingUnmap;
  ASSERT_NE(savedSymFlatUnmap, nullptr);
  injectedFlatUnmapFailures = 0;
  if (rank == 0)
    deviceAdaptor->symFlatMappingUnmap = failSymFlatUnmapOnce;

  flagcxResult_t result = flagcxCommWindowDeregister(testComm, window);
  deviceAdaptor->symFlatMappingUnmap = savedSymFlatUnmap;

  int minimum = 0;
  int maximum = 0;
  allRankResultRange(result, &minimum, &maximum);
  EXPECT_EQ(minimum, static_cast<int>(flagcxRemoteError));
  EXPECT_EQ(maximum, static_cast<int>(flagcxRemoteError));
  ASSERT_NE(window->defaultBase, nullptr);
  EXPECT_TRUE(window->defaultBase->published);
  EXPECT_EQ(testComm->heteroComm->symWindows, window->defaultBase);
  if (rank == 0) {
    EXPECT_NE(window->defaultBase->flatBase, nullptr);
  }

  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowDeregister(testComm, window)));
  EXPECT_EQ(testComm->heteroComm->symWindows, nullptr);
  ASSERT_TRUE(allRanksSucceeded(flagcxCommDestroy(testComm)));
  EXPECT_EQ(flagcxMemFree(buffer), flagcxSuccess);
}

TEST_F(SymMemTest, DuplicateWindowsShareMrUntilLastDeregister) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_NET_MR"))
    GTEST_SKIP() << "Runs in the required NET invocations";

  flagcxWindow_t first = nullptr;
  flagcxWindow_t second = nullptr;
  ASSERT_EQ(flagcxCommWindowRegister(comm, devBuff, size, &first,
                                     FLAGCX_WIN_COLL_SYMMETRIC),
            flagcxSuccess);
  ASSERT_EQ(flagcxCommWindowRegister(comm, devBuff, size, &second,
                                     FLAGCX_WIN_COLL_SYMMETRIC),
            flagcxSuccess);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_NE(first->defaultBase, nullptr);
  ASSERT_NE(second->defaultBase, nullptr);
  EXPECT_EQ(first->defaultBase->mrIndex, second->defaultBase->mrIndex);

  const int mrIndex = first->defaultBase->mrIndex;
  ASSERT_GE(mrIndex, 0);
  ASSERT_LT(mrIndex, comm->heteroComm->oneSideHandleCount);
  auto *handle = comm->heteroComm->oneSideHandles[mrIndex];
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(handle->windowRefs, 2u);

  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowDeregister(comm, first)));
  handle = comm->heteroComm->oneSideHandles[mrIndex];
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(handle->windowRefs, 1u);
  EXPECT_NE(second->defaultBase, nullptr);
  EXPECT_TRUE(second->defaultBase->hasNetworkMrRef);

  // Prove that the remaining reference is not merely bookkeeping: use the
  // shared MR for a real cross-node GET after the first window is gone.
  int remotePeer = -1;
  for (int peer = 0; peer < comm->nranks; peer++) {
    if (comm->heteroComm->rankToNode[peer] != comm->heteroComm->node) {
      remotePeer = peer;
      break;
    }
  }
  ASSERT_GE(remotePeer, 0);
  constexpr size_t transferSize = 128;
  constexpr size_t sourceOffset = 64;
  constexpr size_t destinationOffset = 4096;
  std::vector<uint8_t> pattern(transferSize,
                               static_cast<uint8_t>(comm->rank + 31));
  flagcxResult_t result =
      devHandle->deviceMemset(devBuff, 0, size, flagcxMemDevice, stream);
  if (result == flagcxSuccess)
    result = devHandle->deviceMemcpy(
        static_cast<char *>(devBuff) + sourceOffset, pattern.data(),
        transferSize, flagcxMemcpyHostToDevice, stream);
  if (result == flagcxSuccess)
    result = devHandle->streamSynchronize(stream);
  ASSERT_TRUE(allRanksSucceeded(result));
  MPI_Barrier(MPI_COMM_WORLD);

  uint64_t before = 0;
  result = flagcxReadCounter(comm, &before);
  if (result == flagcxSuccess)
    result = flagcxGet(comm, remotePeer, sourceOffset, destinationOffset,
                       transferSize, mrIndex, mrIndex);
  if (result == flagcxSuccess)
    result = flagcxWaitCounter(comm, before + 1);
  ASSERT_TRUE(allRanksSucceeded(result));

  std::vector<uint8_t> readBack(transferSize, 0);
  result = devHandle->deviceMemcpy(
      readBack.data(), static_cast<char *>(devBuff) + destinationOffset,
      transferSize, flagcxMemcpyDeviceToHost, nullptr);
  ASSERT_TRUE(allRanksSucceeded(result));
  EXPECT_EQ(readBack, std::vector<uint8_t>(
                          transferSize, static_cast<uint8_t>(remotePeer + 31)));

  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowDeregister(comm, second)));
  handle = comm->heteroComm->oneSideHandles[mrIndex];
  if (mrIndex == 0) {
    ASSERT_NE(handle, nullptr);
    EXPECT_EQ(handle->windowRefs, 0u);
    EXPECT_EQ(handle->localMrHandle, nullptr);
  } else {
    EXPECT_EQ(handle, nullptr);
  }
}

TEST_F(SymMemTest, CommDestroyReleasesLiveWindow) {
  flagcxComm_t testComm = nullptr;
  ASSERT_TRUE(allRanksSucceeded(createTestComm(&testComm)));
  ASSERT_NE(testComm, nullptr);
  void *buffer = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&buffer, size)));
  flagcxWindow_t liveWindow = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowRegister(
      testComm, buffer, size, &liveWindow, FLAGCX_WIN_COLL_SYMMETRIC)));
  ASSERT_NE(liveWindow, nullptr);
  ASSERT_NE(testComm->heteroComm->symWindows, nullptr);

  int multicastSupported = 0;
  int multicastQueryOk = 1;
  if (envEnabled("FLAGCX_CI_REQUIRE_VMM") &&
      deviceAdaptor->symMulticastSupported != nullptr) {
    flagcxResult_t supportResult =
        deviceAdaptor->symMulticastSupported(&multicastSupported);
    multicastQueryOk = supportResult == flagcxSuccess ? 1 : 0;
  }
  int allMulticastQueriesOk = 0;
  MPI_Allreduce(&multicastQueryOk, &allMulticastQueriesOk, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  int allMulticastSupported = 0;
  MPI_Allreduce(&multicastSupported, &allMulticastSupported, 1, MPI_INT,
                MPI_MIN, MPI_COMM_WORLD);
  if (allMulticastQueriesOk != 0 && allMulticastSupported != 0 &&
      liveWindow->defaultBase->hasFlatMapping) {
    ASSERT_NE(deviceAdaptor->symMulticastImport, nullptr);
    EXPECT_NE(liveWindow->defaultBase->mcBase, nullptr);
    EXPECT_NE(liveWindow->defaultBase->mcHandle, nullptr);
  }

  // Deliberately omit flagcxCommWindowDeregister. Communicator destruction is
  // collective for some device providers, so every rank must enter it
  // concurrently. The local-only cleanup behavior is covered by the mock
  // ownership tests in test_sym_window_struct.cpp.
  flagcxResult_t result = flagcxCommDestroy(testComm);
  ASSERT_TRUE(allRanksSucceeded(result));
  testComm = nullptr;
  EXPECT_EQ(flagcxMemFree(buffer), flagcxSuccess);
}
