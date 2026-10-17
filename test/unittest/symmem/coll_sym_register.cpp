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

#ifdef __linux__
#include <dirent.h>
#include <sys/resource.h>
#endif

#if defined(USE_NVIDIA_ADAPTOR) || defined(USE_PPU_ADAPTOR)
#include <cuda.h>
#include <cuda_runtime.h>
#elif defined(USE_DU_ADAPTOR)
#include <hip/hip_runtime.h>
#elif defined(USE_METAX_ADAPTOR)
#include <mcr/mc_runtime.h>
#endif

namespace {

flagcxResult_t (*savedProviderMapOne)(void *, int, int) = nullptr;

flagcxResult_t failAfterProviderMapOne(void *flatBase, int index, int fd) {
  flagcxResult_t result = savedProviderMapOne(flatBase, index, fd);
  return result == flagcxSuccess ? flagcxUnhandledDeviceError : result;
}

bool envEnabled(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && std::strcmp(value, "0") != 0;
}

#ifdef __linux__
size_t openFdCount() {
  DIR *directory = opendir("/proc/self/fd");
  if (directory == nullptr)
    return 0;
  size_t count = 0;
  while (readdir(directory) != nullptr)
    count++;
  closedir(directory);
  return count >= 2 ? count - 2 : 0;
}

class ScopedFdLimit {
public:
  bool lowerTo(rlim_t limit) {
    if (getrlimit(RLIMIT_NOFILE, &original_) != 0)
      return false;
    if (limit >= original_.rlim_cur)
      return false;
    struct rlimit lowered = original_;
    lowered.rlim_cur = limit;
    if (setrlimit(RLIMIT_NOFILE, &lowered) != 0)
      return false;
    active_ = true;
    return true;
  }
  ~ScopedFdLimit() {
    if (active_)
      setrlimit(RLIMIT_NOFILE, &original_);
  }

private:
  struct rlimit original_ = {};
  bool active_ = false;
};
#endif

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

int injectedFlatUnmapFailures = 0;

flagcxResult_t (*savedSymLsaMapBeginAt)(void *, int, int, void *,
                                        const flagcxSymRankLayout *,
                                        const flagcxSymSegmentMeta *, int,
                                        size_t, size_t, void **) = nullptr;
flagcxResult_t (*savedSymSegmentUnmap)(void *) = nullptr;
flagcxResult_t (*savedSymLsaVaReserve)(size_t, size_t, void **) = nullptr;
int injectedSymLsaVaReserveCalls = 0;

flagcxResult_t failSymLsaVaReserve(size_t, size_t, void **base) {
  ++injectedSymLsaVaReserveCalls;
  if (base != nullptr)
    *base = nullptr;
  return flagcxUnhandledDeviceError;
}

flagcxResult_t failSymLsaMapBeginAt(void *, int, int, void *,
                                    const flagcxSymRankLayout *,
                                    const flagcxSymSegmentMeta *, int, size_t,
                                    size_t, void **mappingBase) {
  if (mappingBase != nullptr)
    *mappingBase = nullptr;
  return flagcxUnhandledDeviceError;
}

flagcxResult_t failSymSegmentUnmapOnce(void *mappingBase) {
  if (injectedFlatUnmapFailures++ == 0)
    return flagcxRemoteError;
  return savedSymSegmentUnmap != nullptr ? savedSymSegmentUnmap(mappingBase)
                                         : flagcxInternalError;
}

flagcxResult_t failSymPhysDescribe(void *, size_t, void **bundle,
                                   const flagcxSymSegmentMeta **segments,
                                   int *count, size_t *userOffset,
                                   size_t *alignment) {
  if (bundle != nullptr)
    *bundle = nullptr;
  if (segments != nullptr)
    *segments = nullptr;
  if (count != nullptr)
    *count = 0;
  if (userOffset != nullptr)
    *userOffset = 0;
  if (alignment != nullptr)
    *alignment = 0;
  return flagcxSystemError;
}

flagcxResult_t disableMulticastForLsaTest(int *supported) {
  if (supported == nullptr)
    return flagcxInvalidArgument;
  *supported = 0;
  return flagcxSuccess;
}

class ScopedMulticastDisabled {
public:
  ScopedMulticastDisabled()
      : original_(deviceAdaptor != nullptr
                      ? deviceAdaptor->symMulticastSupported
                      : nullptr) {
    if (deviceAdaptor != nullptr)
      deviceAdaptor->symMulticastSupported = disableMulticastForLsaTest;
  }
  ~ScopedMulticastDisabled() {
    if (deviceAdaptor != nullptr)
      deviceAdaptor->symMulticastSupported = original_;
  }

private:
  flagcxResult_t (*original_)(int *);
};

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

TEST_F(SymMemTest, RankLocalWindowRangeErrorConvergesBeforeSetup) {
  if (comm == nullptr || comm->heteroComm == nullptr ||
      comm->heteroComm->nRanks < 2)
    GTEST_SKIP() << "Requires at least two ranks";
  void *buffer = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&buffer, size)));
  const uint64_t fdExchangesBefore = comm->heteroComm->symWindowFdExchangeCount;
  void *requested = comm->heteroComm->rank == 1
                        ? (void *)((uintptr_t)buffer + size - 1)
                        : buffer;
  flagcxWindow_t window = nullptr;
  flagcxResult_t result = flagcxCommWindowRegister(
      comm, requested, size, &window, FLAGCX_WIN_COLL_SYMMETRIC);
  int minimum = 0;
  int maximum = 0;
  allRankResultRange(result, &minimum, &maximum);
  EXPECT_EQ(minimum, static_cast<int>(flagcxInvalidUsage));
  EXPECT_EQ(maximum, static_cast<int>(flagcxInvalidUsage));
  EXPECT_EQ(window, nullptr);
  EXPECT_EQ(comm->heteroComm->symWindows, nullptr);
  EXPECT_EQ(comm->heteroComm->symWindowFdExchangeCount, fdExchangesBefore);
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

  auto originalSymPhysDescribe = deviceAdaptor->symPhysDescribe;
  ASSERT_NE(originalSymPhysDescribe, nullptr);
  if (rank == 1)
    deviceAdaptor->symPhysDescribe = failSymPhysDescribe;

  flagcxWindow_t fallbackWindow = nullptr;
  flagcxResult_t result = flagcxCommWindowRegister(
      testComm, buffer, size, &fallbackWindow, FLAGCX_WIN_COLL_SYMMETRIC);
  if (rank == 1)
    deviceAdaptor->symPhysDescribe = originalSymPhysDescribe;

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

TEST_F(SymMemTest, VmmLsaMapFailureDoesNotPublishNetworkOnlyWindow) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_VMM") ||
      !envEnabled("FLAGCX_CI_REQUIRE_NET_MR"))
    GTEST_SKIP() << "Runs in the required VMM + NET invocation";
  ScopedMulticastDisabled noMulticast;

  flagcxComm_t testComm = nullptr;
  ASSERT_TRUE(allRanksSucceeded(createTestComm(&testComm)));
  ASSERT_NE(testComm, nullptr);
  ASSERT_NE(testComm->heteroComm, nullptr);
  ASSERT_GT(testComm->heteroComm->localRanks, 1);
  void *buffer = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&buffer, size)));

  auto originalSymLsaMap = deviceAdaptor->symLsaMapBeginAt;
  ASSERT_NE(originalSymLsaMap, nullptr);
  if (testComm->heteroComm->localRank == 1)
    deviceAdaptor->symLsaMapBeginAt = failSymLsaMapBeginAt;

  const int handlesBefore = testComm->heteroComm->oneSideHandleCount;
  flagcxWindow_t failedWindow = nullptr;
  flagcxResult_t result = flagcxCommWindowRegister(
      testComm, buffer, size, &failedWindow, FLAGCX_WIN_COLL_SYMMETRIC);
  deviceAdaptor->symLsaMapBeginAt = originalSymLsaMap;

  int minimum = 0;
  int maximum = 0;
  allRankResultRange(result, &minimum, &maximum);
  EXPECT_EQ(minimum, maximum);
  EXPECT_NE(result, flagcxSuccess);
  EXPECT_EQ(failedWindow, nullptr);
  EXPECT_EQ(testComm->heteroComm->symWindows, nullptr);
  EXPECT_EQ(testComm->heteroComm->oneSideHandleCount, handlesBefore);

  flagcxWindow_t retryWindow = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowRegister(
      testComm, buffer, size, &retryWindow, FLAGCX_WIN_COLL_SYMMETRIC)));
  ASSERT_NE(retryWindow, nullptr);
  EXPECT_EQ(retryWindow->defaultBase->resourcePath, flagcxSymResourceLsa);
  ASSERT_TRUE(
      allRanksSucceeded(flagcxCommWindowDeregister(testComm, retryWindow)));
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

TEST_F(SymMemTest, DmaBufUnavailableWithoutVaDoesNotPublishWindow) {
  if (!envEnabled("FLAGCX_CI_EXPECT_VMM_DMABUF_UNSUPPORTED"))
    GTEST_SKIP() << "Runs only on a provider without DMA-BUF MR registration";
  ASSERT_TRUE(envEnabled("FLAGCX_CI_REQUIRE_VMM"));
  ASSERT_TRUE(envEnabled("FLAGCX_CI_REQUIRE_NET_MR"));
  ASSERT_TRUE(envEnabled("FLAGCX_DMABUF_ENABLE"));

  flagcxComm_t testComm = nullptr;
  ASSERT_TRUE(allRanksSucceeded(createTestComm(&testComm)));
  ASSERT_NE(testComm, nullptr);
  ASSERT_NE(testComm->heteroComm, nullptr);
  ASSERT_NE(testComm->heteroComm->netAdaptor, nullptr);
  EXPECT_EQ(testComm->heteroComm->netAdaptor->regMrDmaBuf, nullptr);
  ASSERT_EQ(testComm->heteroComm->netAdaptor->vmmMrCaps & FLAGCX_VMM_MR_CAP_VA,
            0u);

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

  ScopedEnvVar routeMode("FLAGCX_DMABUF_ENABLE", "1");
  const char *modes[] = {"1", "0"};
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
      if (routeIndex == 1)
        EXPECT_EQ(handle->registrationRoute,
                  static_cast<uint8_t>(FLAGCX_VMM_MR_ROUTE_VA));
      else
        EXPECT_TRUE(handle->registrationRoute == FLAGCX_VMM_MR_ROUTE_DMABUF ||
                    handle->registrationRoute == FLAGCX_VMM_MR_ROUTE_VA);
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

  // This allocation bypasses flagcxMemAlloc. VMM being enabled globally must
  // not classify this ordinary device allocation as VMM. DMA-BUF eligibility
  // is controlled separately by FLAGCX_DMABUF_ENABLE and the export result.
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
  EXPECT_TRUE(testComm->heteroComm->signalHandle->registrationRoute ==
                  FLAGCX_VMM_MR_ROUTE_VA ||
              testComm->heteroComm->signalHandle->registrationRoute ==
                  FLAGCX_VMM_MR_ROUTE_DMABUF);
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

TEST_F(SymMemTest, AsymmetricPendingCleanupWithoutLocalTokenConverges) {
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(comm->heteroComm, nullptr);
  flagcxHeteroComm_t hetero = comm->heteroComm;
  if (hetero->nRanks < 2)
    GTEST_SKIP() << "Requires at least two ranks";
  ASSERT_EQ(hetero->pendingSymCleanup, nullptr);

  // Only one rank owns a failed-registration token. All other ranks must
  // still pass through every cleanup convergence stage with a null token.
  flagcxWindow_t token = nullptr;
  flagcxSymWindow_t pending = nullptr;
  if (hetero->rank == 0) {
    token = static_cast<flagcxWindow_t>(calloc(1, sizeof(*token)));
    pending = static_cast<flagcxSymWindow_t>(calloc(1, sizeof(*pending)));
  }
  const bool allocated =
      hetero->rank != 0 || (token != nullptr && pending != nullptr);
  if (!allRanksSucceeded(allocated ? flagcxSuccess : flagcxSystemError)) {
    free(pending);
    free(token);
    FAIL() << "Could not allocate the pending cleanup token";
  }
  flagcxResult_t retain = flagcxSuccess;
  if (hetero->rank == 0) {
    token->defaultBase = pending;
    pending->owner = token;
    pending->mrIndex = -1;
    pending->ipcSlot = -1;
    retain = flagcxSymRetainPendingCleanup(hetero, token);
  }
  if (!allRanksSucceeded(retain)) {
    free(pending);
    free(token);
    FAIL() << "Could not retain the pending cleanup token";
  }

  EXPECT_EQ(flagcxSymRetryPendingCleanup(hetero, flagcxSymCleanupCollective),
            flagcxSuccess);
  EXPECT_EQ(hetero->pendingSymCleanup, nullptr);
}

TEST_F(SymMemTest, FlatMapRollbackFailureRetainsOwnershipForRetry) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_VMM") ||
      envEnabled("FLAGCX_CI_REQUIRE_NET_MR"))
    GTEST_SKIP() << "Runs in the required VMM-local invocation";
  ScopedMulticastDisabled noMulticast;

  flagcxComm_t testComm = nullptr;
  ASSERT_TRUE(allRanksSucceeded(createTestComm(&testComm)));
  ASSERT_NE(testComm, nullptr);
  ASSERT_GT(testComm->heteroComm->localRanks, 1);
  void *buffer = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&buffer, size)));

  savedSymLsaMapBeginAt = deviceAdaptor->symLsaMapBeginAt;
  savedSymSegmentUnmap = deviceAdaptor->symFlatMappingUnmapSegments;
  ASSERT_NE(savedSymLsaMapBeginAt, nullptr);
  ASSERT_NE(savedSymSegmentUnmap, nullptr);
  injectedFlatUnmapFailures = 0;
  if (testComm->heteroComm->localRank == 1)
    deviceAdaptor->symLsaMapBeginAt = failSymLsaMapBeginAt;
  if (testComm->heteroComm->localRank == 0)
    deviceAdaptor->symFlatMappingUnmapSegments = failSymSegmentUnmapOnce;

  flagcxWindow_t failedWindow = nullptr;
  flagcxResult_t result = flagcxCommWindowRegister(
      testComm, buffer, size, &failedWindow, FLAGCX_WIN_COLL_SYMMETRIC);
  deviceAdaptor->symLsaMapBeginAt = savedSymLsaMapBeginAt;
  deviceAdaptor->symFlatMappingUnmapSegments = savedSymSegmentUnmap;

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
  ScopedMulticastDisabled noMulticast;

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
  savedSymLsaMapBeginAt = deviceAdaptor->symLsaMapBeginAt;
  savedSymSegmentUnmap = deviceAdaptor->symFlatMappingUnmapSegments;
  ASSERT_NE(savedSymLsaMapBeginAt, nullptr);
  ASSERT_NE(savedSymSegmentUnmap, nullptr);
  injectedFlatUnmapFailures = 0;

  const int failingNode = hetero->rankToNode[0];
  const bool onFailingNode = hetero->node == failingNode;
  if (onFailingNode && hetero->localRank == 1)
    deviceAdaptor->symLsaMapBeginAt = failSymLsaMapBeginAt;
  if (onFailingNode && hetero->localRank == 0)
    deviceAdaptor->symFlatMappingUnmapSegments = failSymSegmentUnmapOnce;

  const uint64_t metadataExchangesBefore =
      hetero->oneSideDataMetadataExchangeCount;
  flagcxWindow_t failedWindow = nullptr;
  flagcxResult_t result = flagcxCommWindowRegister(
      testComm, buffer, size, &failedWindow, FLAGCX_WIN_COLL_SYMMETRIC);
  deviceAdaptor->symLsaMapBeginAt = savedSymLsaMapBeginAt;
  deviceAdaptor->symFlatMappingUnmapSegments = savedSymSegmentUnmap;

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
  ScopedMulticastDisabled noMulticast;

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

  savedSymSegmentUnmap = deviceAdaptor->symFlatMappingUnmapSegments;
  ASSERT_NE(savedSymSegmentUnmap, nullptr);
  injectedFlatUnmapFailures = 0;
  if (rank == 0)
    deviceAdaptor->symFlatMappingUnmapSegments = failSymSegmentUnmapOnce;

  flagcxResult_t result = flagcxCommWindowDeregister(testComm, window);
  deviceAdaptor->symFlatMappingUnmapSegments = savedSymSegmentUnmap;

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

TEST_F(SymMemTest, CrossNodeMixedVmmAndOrdinarySkipsLegacyIpc) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_MIXED_NODE_VMM"))
    GTEST_SKIP() << "Requires a multi-node VMM + NET invocation";
  ASSERT_TRUE(envEnabled("FLAGCX_VMM_ENABLE"));
  ASSERT_NE(deviceAdaptor, nullptr);
  ASSERT_NE(deviceAdaptor->symPhysDescribe, nullptr);
  ASSERT_NE(deviceAdaptor->symPhysFreeSegments, nullptr);

  flagcxComm_t testComm = nullptr;
  ASSERT_TRUE(allRanksSucceeded(createTestComm(&testComm)));
  ASSERT_NE(testComm, nullptr);
  flagcxHeteroComm_t hetero = testComm->heteroComm;
  ASSERT_NE(hetero, nullptr);
  ASSERT_LT(hetero->localRanks, hetero->nRanks);
  const bool vmmNode = hetero->localRankToRank[0] != 0;

  void *probe = nullptr;
  ASSERT_TRUE(
      allRanksSucceeded(deviceAdaptor->gdrMemAlloc(&probe, 1, nullptr)));
  void *probeBundle = nullptr;
  const flagcxSymSegmentMeta *probeSegments = nullptr;
  int probeCount = 0;
  size_t probeOffset = 0;
  size_t granularity = 0;
  flagcxResult_t result =
      deviceAdaptor->symPhysDescribe(probe, 1, &probeBundle, &probeSegments,
                                     &probeCount, &probeOffset, &granularity);
  ASSERT_TRUE(allRanksSucceeded(result));
  ASSERT_GT(granularity, 0u);
  ASSERT_LE(granularity, SIZE_MAX / 3);
  ASSERT_EQ(deviceAdaptor->symPhysFreeSegments(probeBundle), flagcxSuccess);
  ASSERT_EQ(deviceAdaptor->gdrMemFree(probe, nullptr), flagcxSuccess);

  // Keep the required-success case within one physical segment. The separate
  // multi-segment case checks whether this NET provider accepts the full VA.
  const size_t allocationSize =
      granularity < (1U << 20) ? granularity : static_cast<size_t>(1U << 20);
  const std::string segmentSetting = std::to_string(granularity);
  ScopedEnvVar segmentSize("FLAGCX_VMM_SEGMENT_BYTES",
                           vmmNode ? segmentSetting.c_str() : nullptr);
  void *buffer = nullptr;
  result = vmmNode ? flagcxMemAlloc(&buffer, allocationSize)
                   : deviceAdaptor->deviceMalloc(&buffer, allocationSize,
                                                 flagcxMemDevice, nullptr);
  ASSERT_TRUE(allRanksSucceeded(result));
  segmentSize.set(nullptr);

  auto originalIpcGet = deviceAdaptor->ipcMemHandleGet;
  if (!vmmNode)
    deviceAdaptor->ipcMemHandleGet = failIpcMemHandleGet;
  flagcxWindow_t window = nullptr;
  result = flagcxCommWindowRegister(testComm, buffer, allocationSize, &window,
                                    FLAGCX_WIN_COLL_SYMMETRIC);
  if (!vmmNode)
    deviceAdaptor->ipcMemHandleGet = originalIpcGet;
  ASSERT_TRUE(allRanksSucceeded(result));
  ASSERT_NE(window, nullptr);
  ASSERT_NE(window->defaultBase, nullptr);
  EXPECT_EQ(window->defaultBase->ipcSlot, -1);
  EXPECT_TRUE(window->defaultBase->hasNetworkMrRef);
  EXPECT_EQ(window->defaultBase->resourcePath == flagcxSymResourceLsa ||
                window->defaultBase->resourcePath == flagcxSymResourceLegacyV1,
            vmmNode);
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowDeregister(testComm, window)));

  auto *net = hetero->netAdaptor;
  ASSERT_NE(net, nullptr);
  auto originalRegMr = net->regMr;
  auto originalRegMrDmaBuf = net->regMrDmaBuf;
  if (hetero->rank == 0) {
    net->regMr = failRegMr;
    net->regMrDmaBuf = failRegMrDmaBuf;
  }
  window = nullptr;
  result = flagcxCommWindowRegister(testComm, buffer, allocationSize, &window,
                                    FLAGCX_WIN_COLL_SYMMETRIC);
  if (hetero->rank == 0) {
    net->regMr = originalRegMr;
    net->regMrDmaBuf = originalRegMrDmaBuf;
  }
  int minimum = 0;
  int maximum = 0;
  allRankResultRange(result, &minimum, &maximum);
  EXPECT_EQ(minimum, maximum);
  EXPECT_EQ(minimum, static_cast<int>(flagcxSystemError));
  EXPECT_EQ(window, nullptr);
  EXPECT_EQ(hetero->symWindows, nullptr);

  ASSERT_TRUE(allRanksSucceeded(flagcxCommDestroy(testComm)));
  result = vmmNode
               ? flagcxMemFree(buffer)
               : deviceAdaptor->deviceFree(buffer, flagcxMemDevice, nullptr);
  EXPECT_EQ(result, flagcxSuccess);
}

TEST_F(SymMemTest, CrossNodeMixedMultiSegmentVmmConvergesMrFailure) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_MIXED_NODE_VMM"))
    GTEST_SKIP() << "Requires a multi-node VMM + NET invocation";
  ASSERT_TRUE(envEnabled("FLAGCX_VMM_ENABLE"));
  ASSERT_NE(deviceAdaptor, nullptr);
  ASSERT_NE(deviceAdaptor->symPhysDescribe, nullptr);
  ASSERT_NE(deviceAdaptor->symPhysFreeSegments, nullptr);

  flagcxComm_t testComm = nullptr;
  ASSERT_TRUE(allRanksSucceeded(createTestComm(&testComm)));
  ASSERT_NE(testComm, nullptr);
  flagcxHeteroComm_t hetero = testComm->heteroComm;
  ASSERT_NE(hetero, nullptr);
  ASSERT_LT(hetero->localRanks, hetero->nRanks);
  const bool vmmNode = hetero->localRankToRank[0] != 0;

  void *probe = nullptr;
  ASSERT_TRUE(
      allRanksSucceeded(deviceAdaptor->gdrMemAlloc(&probe, 1, nullptr)));
  void *probeBundle = nullptr;
  const flagcxSymSegmentMeta *probeSegments = nullptr;
  int probeCount = 0;
  size_t probeOffset = 0;
  size_t granularity = 0;
  flagcxResult_t result =
      deviceAdaptor->symPhysDescribe(probe, 1, &probeBundle, &probeSegments,
                                     &probeCount, &probeOffset, &granularity);
  ASSERT_TRUE(allRanksSucceeded(result));
  ASSERT_GT(granularity, 0u);
  ASSERT_LE(granularity, SIZE_MAX / 3);
  ASSERT_EQ(deviceAdaptor->symPhysFreeSegments(probeBundle), flagcxSuccess);
  ASSERT_EQ(deviceAdaptor->gdrMemFree(probe, nullptr), flagcxSuccess);

  const size_t allocationSize = granularity * 3;
  const std::string segmentSetting = std::to_string(granularity);
  ScopedEnvVar segmentSize("FLAGCX_VMM_SEGMENT_BYTES",
                           vmmNode ? segmentSetting.c_str() : nullptr);
  void *buffer = nullptr;
  result = vmmNode ? flagcxMemAlloc(&buffer, allocationSize)
                   : deviceAdaptor->deviceMalloc(&buffer, allocationSize,
                                                 flagcxMemDevice, nullptr);
  ASSERT_TRUE(allRanksSucceeded(result));
  segmentSize.set(nullptr);

  if (vmmNode) {
    void *bundle = nullptr;
    const flagcxSymSegmentMeta *segments = nullptr;
    int count = 0;
    size_t userOffset = 0;
    size_t alignment = 0;
    result = deviceAdaptor->symPhysDescribe(buffer, allocationSize, &bundle,
                                            &segments, &count, &userOffset,
                                            &alignment);
    if (result == flagcxSuccess && count != 3)
      result = flagcxInvalidUsage;
    flagcxResult_t release = deviceAdaptor->symPhysFreeSegments(bundle);
    if (result == flagcxSuccess)
      result = release;
  }
  ASSERT_TRUE(allRanksSucceeded(result));

  flagcxWindow_t window = nullptr;
  result = flagcxCommWindowRegister(testComm, buffer, allocationSize, &window,
                                    FLAGCX_WIN_COLL_SYMMETRIC);
  int minimum = 0;
  int maximum = 0;
  allRankResultRange(result, &minimum, &maximum);
  EXPECT_EQ(minimum, maximum);
  EXPECT_TRUE(minimum == static_cast<int>(flagcxSuccess) ||
              minimum == static_cast<int>(flagcxNotSupported) ||
              minimum == static_cast<int>(flagcxSystemError));
  if (minimum == static_cast<int>(flagcxSuccess)) {
    ASSERT_NE(window, nullptr);
    ASSERT_NE(window->defaultBase, nullptr);
    EXPECT_EQ(window->defaultBase->ipcSlot, -1);
    EXPECT_TRUE(window->defaultBase->hasNetworkMrRef);
    ASSERT_TRUE(
        allRanksSucceeded(flagcxCommWindowDeregister(testComm, window)));
  } else {
    EXPECT_EQ(window, nullptr);
    EXPECT_EQ(hetero->symWindows, nullptr);
    EXPECT_EQ(hetero->pendingSymCleanup, nullptr);
    EXPECT_EQ(hetero->pendingOneSideCleanup, nullptr);
    EXPECT_EQ(hetero->pendingOneSideMeshCleanup, nullptr);
  }

  // Disabling DMA-BUF selects the complete VA range on providers that offer
  // that route. Registration may still fail on a particular driver/NIC.
  {
    ScopedEnvVar noDmaBuf("FLAGCX_DMABUF_ENABLE", "0");
    window = nullptr;
    result = flagcxCommWindowRegister(testComm, buffer, allocationSize, &window,
                                      FLAGCX_WIN_COLL_SYMMETRIC);
  }
  allRankResultRange(result, &minimum, &maximum);
  EXPECT_EQ(minimum, maximum);
  EXPECT_TRUE(minimum == static_cast<int>(flagcxSuccess) ||
              minimum == static_cast<int>(flagcxNotSupported) ||
              minimum == static_cast<int>(flagcxSystemError));
  if (minimum == static_cast<int>(flagcxSuccess)) {
    ASSERT_NE(window, nullptr);
    ASSERT_NE(window->defaultBase, nullptr);
    const int mrIndex = window->defaultBase->mrIndex;
    ASSERT_GE(mrIndex, 0);
    ASSERT_LT(mrIndex, hetero->oneSideHandleCount);
    ASSERT_NE(hetero->oneSideHandles[mrIndex], nullptr);
    EXPECT_EQ(hetero->oneSideHandles[mrIndex]->registrationRoute,
              FLAGCX_VMM_MR_ROUTE_VA);

    int remotePeer = -1;
    for (int peer = 0; peer < hetero->nRanks; peer++) {
      if (hetero->rankToNode[peer] != hetero->node) {
        remotePeer = peer;
        break;
      }
    }
    ASSERT_GE(remotePeer, 0);
    const size_t transferSize = granularity < 128 ? granularity / 2 : 64;
    ASSERT_GT(transferSize, 0u);
    result = devHandle->deviceMemset(buffer, 0, allocationSize, flagcxMemDevice,
                                     stream);
    for (int segment = 0; segment < 3 && result == flagcxSuccess; segment++) {
      std::vector<uint8_t> pattern(
          transferSize, static_cast<uint8_t>(hetero->rank + segment + 31));
      result = devHandle->deviceMemcpy(
          static_cast<char *>(buffer) + segment * granularity, pattern.data(),
          transferSize, flagcxMemcpyHostToDevice, stream);
    }
    if (result == flagcxSuccess)
      result = devHandle->streamSynchronize(stream);
    ASSERT_TRUE(allRanksSucceeded(result));
    MPI_Barrier(MPI_COMM_WORLD);

    for (int segment = 0; segment < 3; segment++) {
      const size_t sourceOffset = segment * granularity;
      const size_t destinationOffset = sourceOffset + granularity / 2;
      uint64_t before = 0;
      result = flagcxReadCounter(testComm, &before);
      if (result == flagcxSuccess)
        result = flagcxGet(testComm, remotePeer, sourceOffset,
                           destinationOffset, transferSize, mrIndex, mrIndex);
      if (result == flagcxSuccess)
        result = flagcxWaitCounter(testComm, before + 1);
      ASSERT_TRUE(allRanksSucceeded(result));

      std::vector<uint8_t> readBack(transferSize, 0);
      result = devHandle->deviceMemcpy(
          readBack.data(), static_cast<char *>(buffer) + destinationOffset,
          transferSize, flagcxMemcpyDeviceToHost, nullptr);
      ASSERT_TRUE(allRanksSucceeded(result));
      EXPECT_EQ(readBack, std::vector<uint8_t>(
                              transferSize,
                              static_cast<uint8_t>(remotePeer + segment + 31)));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    ASSERT_TRUE(
        allRanksSucceeded(flagcxCommWindowDeregister(testComm, window)));
  } else {
    EXPECT_EQ(window, nullptr);
    EXPECT_EQ(hetero->symWindows, nullptr);
    EXPECT_EQ(hetero->pendingSymCleanup, nullptr);
    EXPECT_EQ(hetero->pendingOneSideCleanup, nullptr);
    EXPECT_EQ(hetero->pendingOneSideMeshCleanup, nullptr);
  }

  ASSERT_TRUE(allRanksSucceeded(flagcxCommDestroy(testComm)));
  result = vmmNode
               ? flagcxMemFree(buffer)
               : deviceAdaptor->deviceFree(buffer, flagcxMemDevice, nullptr);
  EXPECT_EQ(result, flagcxSuccess);
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
      liveWindow->defaultBase->mcBase != nullptr &&
      liveWindow->defaultBase->resourcePath == flagcxSymResourceLsa) {
    EXPECT_TRUE(liveWindow->defaultBase->mcLsaBound);
    EXPECT_EQ(liveWindow->defaultBase->mcHandle, nullptr);
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

TEST_F(SymMemTest, SegmentedVmmRegistrationUsesBoundedFds) {
#ifndef __linux__
  GTEST_SKIP() << "Requires Linux RLIMIT_NOFILE and /proc/self/fd";
#else
  if (!envEnabled("FLAGCX_CI_REQUIRE_LOW_FD_MULTI_SEGMENT"))
    GTEST_SKIP() << "Requires the low-FD multi-segment VMM CI invocation";
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(comm->heteroComm, nullptr);
  ASSERT_GT(comm->heteroComm->localRanks, 1);
  ASSERT_EQ(comm->heteroComm->localRanks, comm->heteroComm->nRanks);
  ASSERT_NE(deviceAdaptor, nullptr);
  ASSERT_TRUE(envEnabled("FLAGCX_VMM_ENABLE"));
  ASSERT_NE(deviceAdaptor->symPhysDescribe, nullptr);
  ASSERT_NE(deviceAdaptor->symPhysFreeSegments, nullptr);
  ASSERT_NE(deviceAdaptor->symFlatMapSegmentsBegin, nullptr);
  ASSERT_NE(deviceAdaptor->symFlatMapSegmentOne, nullptr);

  void *probe = nullptr;
  ASSERT_TRUE(
      allRanksSucceeded(deviceAdaptor->gdrMemAlloc(&probe, 1, nullptr)));
  void *bundle = nullptr;
  const flagcxSymSegmentMeta *segments = nullptr;
  int count = 0;
  size_t userOffset = 0;
  size_t granularity = 0;
  ASSERT_TRUE(allRanksSucceeded(deviceAdaptor->symPhysDescribe(
      probe, 1, &bundle, &segments, &count, &userOffset, &granularity)));
  ASSERT_EQ(deviceAdaptor->symPhysFreeSegments(bundle), flagcxSuccess);
  ASSERT_EQ(deviceAdaptor->gdrMemFree(probe, nullptr), flagcxSuccess);
  const bool affordable = granularity > 0 && granularity <= 4 * 1024 * 1024;
  if (!allRanksSucceeded(affordable ? flagcxSuccess : flagcxNotSupported))
    GTEST_SKIP() << "VMM granularity is too large for the 64-segment test";

  const std::string setting = std::to_string(granularity);
  ScopedEnvVar segmentSize("FLAGCX_VMM_SEGMENT_BYTES", setting.c_str());
  void *allocation = nullptr;
  const size_t allocationSize = granularity * 64;
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&allocation, allocationSize)));
  segmentSize.set(nullptr);

  flagcxResult_t result = flagcxSuccess;
  flagcxWindow_t window = nullptr;
  {
    ScopedFdLimit fdLimit;
    const size_t baseline = openFdCount();
    const bool lowered =
        baseline > 0 && fdLimit.lowerTo(static_cast<rlim_t>(baseline + 32));
    if (!allRanksSucceeded(lowered ? flagcxSuccess : flagcxSystemError)) {
      EXPECT_EQ(flagcxMemFree(allocation), flagcxSuccess);
      GTEST_SKIP() << "Could not lower RLIMIT_NOFILE on every rank";
    }
    result = flagcxCommWindowRegister(comm, allocation, allocationSize, &window,
                                      FLAGCX_WIN_COLL_SYMMETRIC);
  }
  ASSERT_TRUE(allRanksSucceeded(result));
  ASSERT_NE(window, nullptr);
  ASSERT_NE(window->defaultBase, nullptr);
  EXPECT_EQ(window->defaultBase->resourcePath, flagcxSymResourceLsa);
  EXPECT_EQ(window->defaultBase->rankLayouts[comm->heteroComm->localRank]
                .segmentCount,
            64);
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowDeregister(comm, window)));
  EXPECT_EQ(flagcxMemFree(allocation), flagcxSuccess);
#endif
}

TEST_F(SymMemTest, LsaWindowsShareArenaAndReuseReleasedOffset) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_VMM"))
    GTEST_SKIP() << "Requires the VMM LSA path";
  ScopedMulticastDisabled noMulticast;
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(comm->heteroComm, nullptr);
  ASSERT_NE(deviceAdaptor->symLsaMapBeginAt, nullptr);

  void *firstAllocation = nullptr;
  void *secondAllocation = nullptr;
  void *thirdAllocation = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&firstAllocation, size)));
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&secondAllocation, size)));
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&thirdAllocation, size)));

  flagcxWindow_t first = nullptr;
  flagcxWindow_t second = nullptr;
  flagcxWindow_t third = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowRegister(
      comm, firstAllocation, size, &first, FLAGCX_WIN_COLL_SYMMETRIC)));
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowRegister(
      comm, secondAllocation, size, &second, FLAGCX_WIN_COLL_SYMMETRIC)));
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_EQ(first->defaultBase->resourcePath, flagcxSymResourceLsa);
  ASSERT_EQ(second->defaultBase->resourcePath, flagcxSymResourceLsa);
  const uintptr_t firstBase =
      reinterpret_cast<uintptr_t>(first->defaultBase->flatBase);
  const uintptr_t secondBase =
      reinterpret_cast<uintptr_t>(second->defaultBase->flatBase);
  const size_t firstOffset = first->defaultBase->lsaWindowOffset;
  const size_t secondOffset = second->defaultBase->lsaWindowOffset;
  EXPECT_EQ(first->defaultBase->slotStride, second->defaultBase->slotStride);
  EXPECT_EQ(firstBase - firstOffset, secondBase - secondOffset);
  EXPECT_NE(firstOffset, secondOffset);

  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowDeregister(comm, first)));
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowRegister(
      comm, thirdAllocation, size, &third, FLAGCX_WIN_COLL_SYMMETRIC)));
  ASSERT_NE(third, nullptr);
  ASSERT_EQ(third->defaultBase->resourcePath, flagcxSymResourceLsa);
  EXPECT_EQ(third->defaultBase->lsaWindowOffset, firstOffset);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(third->defaultBase->flatBase),
            firstBase);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(second->defaultBase->flatBase),
            secondBase);
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowDeregister(comm, second)));
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowDeregister(comm, third)));
  EXPECT_EQ(flagcxMemFree(firstAllocation), flagcxSuccess);
  EXPECT_EQ(flagcxMemFree(secondAllocation), flagcxSuccess);
  EXPECT_EQ(flagcxMemFree(thirdAllocation), flagcxSuccess);
}

TEST_F(SymMemTest, LsaReserveFailureDoesNotPublishWindow) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_VMM") ||
      envEnabled("FLAGCX_CI_REQUIRE_NET_MR"))
    GTEST_SKIP() << "Runs in the VMM-local invocation";
  ScopedMulticastDisabled noMulticast;
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(comm->heteroComm, nullptr);
  ASSERT_GT(comm->heteroComm->localRanks, 1);
  flagcxComm_t testComm = nullptr;
  ASSERT_TRUE(allRanksSucceeded(createTestComm(&testComm)));
  ASSERT_NE(testComm, nullptr);
  ASSERT_NE(testComm->heteroComm, nullptr);
  ASSERT_EQ(testComm->heteroComm->symLsaArena, nullptr);
  savedSymLsaVaReserve = deviceAdaptor->symLsaVaReserve;
  ASSERT_NE(savedSymLsaVaReserve, nullptr);

  void *allocation = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&allocation, size)));
  injectedSymLsaVaReserveCalls = 0;
  if (testComm->heteroComm->localRank == 1)
    deviceAdaptor->symLsaVaReserve = failSymLsaVaReserve;
  flagcxWindow_t window = nullptr;
  flagcxResult_t result = flagcxCommWindowRegister(
      testComm, allocation, size, &window, FLAGCX_WIN_COLL_SYMMETRIC);
  deviceAdaptor->symLsaVaReserve = savedSymLsaVaReserve;
  EXPECT_TRUE(
      allRanksSucceeded(injectedSymLsaVaReserveCalls ==
                                (testComm->heteroComm->localRank == 1 ? 1 : 0)
                            ? flagcxSuccess
                            : flagcxSystemError));
  int minimum = 0;
  int maximum = 0;
  allRankResultRange(result, &minimum, &maximum);
  EXPECT_EQ(minimum, maximum);
  EXPECT_NE(result, flagcxSuccess);
  EXPECT_EQ(window, nullptr);
  EXPECT_EQ(testComm->heteroComm->symWindows, nullptr);
  EXPECT_EQ(testComm->heteroComm->symLsaArena, nullptr);

  if (minimum == maximum && minimum != flagcxSuccess &&
      allRanksSucceeded(window == nullptr ? flagcxSuccess
                                          : flagcxInvalidUsage)) {
    result = flagcxCommWindowRegister(testComm, allocation, size, &window,
                                      FLAGCX_WIN_COLL_SYMMETRIC);
    const bool retrySucceeded = allRanksSucceeded(result);
    EXPECT_TRUE(retrySucceeded);
    if (retrySucceeded && window != nullptr) {
      EXPECT_EQ(window->defaultBase->resourcePath, flagcxSymResourceLsa);
    }
  }
  if (allRanksSucceeded(window != nullptr ? flagcxSuccess : flagcxNotSupported))
    EXPECT_TRUE(
        allRanksSucceeded(flagcxCommWindowDeregister(testComm, window)));
  EXPECT_TRUE(allRanksSucceeded(flagcxCommDestroy(testComm)));
  EXPECT_EQ(flagcxMemFree(allocation), flagcxSuccess);
}

TEST_F(SymMemTest, SegmentedVmmSubrangeMapsOnlyTouchedSegment) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_MULTI_SEGMENT"))
    GTEST_SKIP() << "Requires the multi-segment VMM CI invocation";
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(comm->heteroComm, nullptr);
  ASSERT_GT(comm->heteroComm->localRanks, 1);

  void *probe = nullptr;
  ASSERT_TRUE(
      allRanksSucceeded(deviceAdaptor->gdrMemAlloc(&probe, 1, nullptr)));
  void *probeBundle = nullptr;
  const flagcxSymSegmentMeta *probeSegments = nullptr;
  int probeCount = 0;
  size_t probeOffset = 0;
  size_t granularity = 0;
  ASSERT_TRUE(allRanksSucceeded(
      deviceAdaptor->symPhysDescribe(probe, 1, &probeBundle, &probeSegments,
                                     &probeCount, &probeOffset, &granularity)));
  ASSERT_EQ(deviceAdaptor->symPhysFreeSegments(probeBundle), flagcxSuccess);
  ASSERT_EQ(deviceAdaptor->gdrMemFree(probe, nullptr), flagcxSuccess);
  if (!allRanksSucceeded(granularity >= 4 && granularity <= 16 * 1024 * 1024
                             ? flagcxSuccess
                             : flagcxNotSupported))
    GTEST_SKIP() << "VMM granularity is too large for the subrange test";

  const std::string setting = std::to_string(granularity);
  ScopedEnvVar segmentSize("FLAGCX_VMM_SEGMENT_BYTES", setting.c_str());
  void *allocation = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&allocation, granularity * 5)));
  segmentSize.set(nullptr);
  const size_t userOffset = granularity * 2 + granularity / 4;
  const size_t userSize = granularity / 2;
  void *buffer = static_cast<char *>(allocation) + userOffset;
  const int pattern = comm->heteroComm->rank % 251 + 1;
  ASSERT_TRUE(allRanksSucceeded(deviceAdaptor->deviceMemset(
      buffer, pattern, userSize, flagcxMemDevice, nullptr)));

  flagcxWindow_t window = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowRegister(
      comm, buffer, userSize, &window, FLAGCX_WIN_COLL_SYMMETRIC)));
  ASSERT_NE(window, nullptr);
  ASSERT_NE(window->defaultBase, nullptr);
  flagcxSymWindow_t state = window->defaultBase;
  const int localRank = comm->heteroComm->localRank;
  EXPECT_EQ(state->resourcePath, flagcxSymResourceLsa);
  ASSERT_NE(state->rankLayouts, nullptr);
  EXPECT_EQ(state->rankLayouts[localRank].segmentCount, 1);
  EXPECT_EQ(state->rankLayouts[localRank].mappedSize, granularity);
  EXPECT_EQ(state->rankLayouts[localRank].userOffset, granularity / 4);

  MPI_Barrier(MPI_COMM_WORLD);
  const int peerLocalRank = (localRank + 1) % comm->heteroComm->localRanks;
  const int peerRank = comm->heteroComm->localRankToRank[peerLocalRank];
  void *peerPtr = nullptr;
  ASSERT_EQ(flagcxSymWindowResolveIpcPeerPtr(comm->heteroComm, state, peerRank,
                                             0, 1, &peerPtr),
            flagcxSuccess);
  unsigned char value = 0;
  ASSERT_EQ(deviceAdaptor->deviceMemcpy(
                &value, peerPtr, 1, flagcxMemcpyDeviceToHost, nullptr, nullptr),
            flagcxSuccess);
  EXPECT_EQ(value, peerRank % 251 + 1);
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowDeregister(comm, window)));

  // A range that straddles the next physical segment must retain both
  // touched segments, with offsets relative to the first one.
  const size_t crossingStart = granularity * 3 / 4;
  const size_t crossingSize = granularity / 2 + 1;
  buffer = static_cast<char *>(allocation) + granularity * 2 + crossingStart;
  ASSERT_TRUE(allRanksSucceeded(deviceAdaptor->deviceMemset(
      buffer, pattern, crossingSize, flagcxMemDevice, nullptr)));
  int touchedSegments = 0;
  ASSERT_TRUE(allRanksSucceeded(deviceAdaptor->symPhysSegmentCount(
      buffer, crossingSize, &touchedSegments)));
  EXPECT_EQ(touchedSegments, 2);

  window = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowRegister(
      comm, buffer, crossingSize, &window, FLAGCX_WIN_COLL_SYMMETRIC)));
  ASSERT_NE(window, nullptr);
  ASSERT_NE(window->defaultBase, nullptr);
  state = window->defaultBase;
  EXPECT_EQ(state->resourcePath, flagcxSymResourceLsa);
  ASSERT_NE(state->rankLayouts, nullptr);
  const flagcxSymRankLayout &crossingLayout = state->rankLayouts[localRank];
  EXPECT_EQ(crossingLayout.segmentCount, 2);
  EXPECT_EQ(crossingLayout.mappedSize, granularity * 2);
  EXPECT_EQ(crossingLayout.userOffset, crossingStart);
  ASSERT_NE(state->segments, nullptr);
  if (crossingLayout.firstSegment >= 0 && crossingLayout.segmentCount >= 2 &&
      crossingLayout.firstSegment + 1 < state->totalSegments) {
    EXPECT_EQ(state->segments[crossingLayout.firstSegment].offset, 0u);
    EXPECT_EQ(state->segments[crossingLayout.firstSegment + 1].offset,
              granularity);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  for (size_t offset : {size_t(0), crossingSize - 1}) {
    ASSERT_EQ(flagcxSymWindowResolveIpcPeerPtr(comm->heteroComm, state,
                                               peerRank, offset, 1, &peerPtr),
              flagcxSuccess);
    ASSERT_EQ(deviceAdaptor->deviceMemcpy(&value, peerPtr, 1,
                                          flagcxMemcpyDeviceToHost, nullptr,
                                          nullptr),
              flagcxSuccess);
    EXPECT_EQ(value, peerRank % 251 + 1);
  }
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowDeregister(comm, window)));
  EXPECT_EQ(flagcxMemFree(allocation), flagcxSuccess);
}

TEST_F(SymMemTest, SegmentedVmmWindowMapsDifferentPeerLayouts) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_MULTI_SEGMENT"))
    GTEST_SKIP() << "Requires the multi-segment VMM CI invocation";
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(comm->heteroComm, nullptr);
  ASSERT_GT(comm->heteroComm->localRanks, 1);
  ASSERT_EQ(comm->heteroComm->localRanks, comm->heteroComm->nRanks);
  ASSERT_NE(deviceAdaptor, nullptr);
  ASSERT_NE(deviceAdaptor->symPhysDescribe, nullptr);
  ASSERT_NE(deviceAdaptor->symPhysFreeSegments, nullptr);
  ASSERT_NE(deviceAdaptor->symFlatMapSegments, nullptr);
  ASSERT_NE(deviceAdaptor->symPhysSegmentCount, nullptr);

  void *probe = nullptr;
  ASSERT_TRUE(
      allRanksSucceeded(deviceAdaptor->gdrMemAlloc(&probe, 1, nullptr)));
  void *probeBundle = nullptr;
  const flagcxSymSegmentMeta *probeSegments = nullptr;
  int probeCount = 0;
  size_t probeOffset = 0;
  size_t granularity = 0;
  flagcxResult_t result =
      deviceAdaptor->symPhysDescribe(probe, 1, &probeBundle, &probeSegments,
                                     &probeCount, &probeOffset, &granularity);
  ASSERT_TRUE(allRanksSucceeded(result));
  ASSERT_EQ(probeCount, 1);
  ASSERT_GT(granularity, 0u);
  ASSERT_LE(granularity, SIZE_MAX / 3);
  ASSERT_EQ(deviceAdaptor->symPhysFreeSegments(probeBundle), flagcxSuccess);
  ASSERT_EQ(deviceAdaptor->gdrMemFree(probe, nullptr), flagcxSuccess);

  const int localRank = comm->heteroComm->localRank;
  const size_t allocationSize = granularity * 3;
  const size_t segmentBytes =
      localRank % 2 == 0 ? granularity * 2 : granularity;
  const std::string setting = std::to_string(segmentBytes);
  ScopedEnvVar segmentSize("FLAGCX_VMM_SEGMENT_BYTES", setting.c_str());
  void *allocation = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&allocation, allocationSize)));
  segmentSize.set(nullptr);
  const size_t userOffset =
      localRank % 2 == 0 ? granularity / 2 : granularity / 4;
  const size_t userSize = granularity * 2;
  void *buffer = static_cast<char *>(allocation) + userOffset;
  const int pattern = comm->heteroComm->rank % 251 + 1;
  ASSERT_TRUE(allRanksSucceeded(deviceAdaptor->deviceMemset(
      buffer, pattern, userSize, flagcxMemDevice, nullptr)));

  flagcxWindow_t window = nullptr;
  result = flagcxCommWindowRegister(comm, buffer, userSize, &window,
                                    FLAGCX_WIN_COLL_SYMMETRIC);
  ASSERT_TRUE(allRanksSucceeded(result));
  ASSERT_NE(window, nullptr);
  ASSERT_NE(window->defaultBase, nullptr);
  EXPECT_EQ(window->defaultBase->resourcePath, flagcxSymResourceLsa);
  ASSERT_NE(window->defaultBase->rankLayouts, nullptr);
  EXPECT_EQ(window->defaultBase->rankLayouts[localRank].segmentCount,
            localRank % 2 == 0 ? 2 : 3);
  EXPECT_EQ(window->defaultBase->rankLayouts[localRank].userOffset, userOffset);

  MPI_Barrier(MPI_COMM_WORLD);
  const int peerLocalRank = (localRank + 1) % comm->heteroComm->localRanks;
  const int peerRank = comm->heteroComm->localRankToRank[peerLocalRank];
  void *peerPtr = nullptr;
  ASSERT_EQ(flagcxSymWindowResolveIpcPeerPtr(comm->heteroComm,
                                             window->defaultBase, peerRank,
                                             granularity, 1, &peerPtr),
            flagcxSuccess);
  unsigned char value = 0;
  ASSERT_EQ(deviceAdaptor->deviceMemcpy(
                &value, peerPtr, 1, flagcxMemcpyDeviceToHost, nullptr, nullptr),
            flagcxSuccess);
  EXPECT_EQ(value, peerRank % 251 + 1);
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowDeregister(comm, window)));

  const int handlesBefore = comm->heteroComm->oneSideHandleCount;
  savedProviderMapOne = deviceAdaptor->symFlatMapSegmentOne;
  if (localRank == 1)
    deviceAdaptor->symFlatMapSegmentOne = failAfterProviderMapOne;
  window = nullptr;
  result = flagcxCommWindowRegister(comm, buffer, userSize, &window,
                                    FLAGCX_WIN_COLL_SYMMETRIC);
  deviceAdaptor->symFlatMapSegmentOne = savedProviderMapOne;
  int minimum = 0;
  int maximum = 0;
  allRankResultRange(result, &minimum, &maximum);
  EXPECT_EQ(minimum, static_cast<int>(flagcxUnhandledDeviceError));
  EXPECT_EQ(maximum, static_cast<int>(flagcxUnhandledDeviceError));
  EXPECT_EQ(window, nullptr);
  EXPECT_EQ(comm->heteroComm->symWindows, nullptr);
  EXPECT_EQ(comm->heteroComm->oneSideHandleCount, handlesBefore);

  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowRegister(
      comm, buffer, userSize, &window, FLAGCX_WIN_COLL_SYMMETRIC)));
  ASSERT_NE(window, nullptr);
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowDeregister(comm, window)));
  EXPECT_EQ(flagcxMemFree(allocation), flagcxSuccess);
}

TEST_F(SymMemTest, SegmentedMapFailureDoesNotPublishNetworkOnlyWindow) {
  if (!envEnabled("FLAGCX_CI_CHECK_SEGMENTED_NET_FALLBACK"))
    GTEST_SKIP() << "Requires multi-segment VMM and NET";
  ASSERT_TRUE(envEnabled("FLAGCX_VMM_ENABLE"));
  ASSERT_FALSE(envEnabled("FLAGCX_IB_DISABLE"));
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(comm->heteroComm, nullptr);
  ASSERT_GT(comm->heteroComm->localRanks, 1);
  if (deviceAdaptor->symPhysDescribe == nullptr ||
      deviceAdaptor->symFlatMapSegmentOne == nullptr)
    GTEST_SKIP() << "Adaptor has no segmented VMM callbacks";

  void *probe = nullptr;
  ASSERT_TRUE(
      allRanksSucceeded(deviceAdaptor->gdrMemAlloc(&probe, 1, nullptr)));
  void *probeBundle = nullptr;
  const flagcxSymSegmentMeta *probeSegments = nullptr;
  int probeCount = 0;
  size_t probeOffset = 0;
  size_t granularity = 0;
  ASSERT_TRUE(allRanksSucceeded(
      deviceAdaptor->symPhysDescribe(probe, 1, &probeBundle, &probeSegments,
                                     &probeCount, &probeOffset, &granularity)));
  ASSERT_GT(granularity, 0u);
  ASSERT_LE(granularity, SIZE_MAX / 3);
  ASSERT_EQ(deviceAdaptor->symPhysFreeSegments(probeBundle), flagcxSuccess);
  ASSERT_EQ(deviceAdaptor->gdrMemFree(probe, nullptr), flagcxSuccess);

  const std::string setting = std::to_string(granularity);
  ScopedEnvVar segmentSize("FLAGCX_VMM_SEGMENT_BYTES", setting.c_str());
  void *allocation = nullptr;
  const size_t allocationSize = granularity * 3;
  ASSERT_TRUE(allRanksSucceeded(flagcxMemAlloc(&allocation, allocationSize)));
  segmentSize.set(nullptr);
  void *allocationBundle = nullptr;
  const flagcxSymSegmentMeta *allocationSegments = nullptr;
  int allocationCount = 0;
  size_t allocationOffset = 0;
  size_t allocationAlignment = 0;
  ASSERT_TRUE(allRanksSucceeded(deviceAdaptor->symPhysDescribe(
      allocation, allocationSize, &allocationBundle, &allocationSegments,
      &allocationCount, &allocationOffset, &allocationAlignment)));
  ASSERT_GT(allocationCount, 1);
  ASSERT_EQ(deviceAdaptor->symPhysFreeSegments(allocationBundle),
            flagcxSuccess);

  // Register this exact allocation first. General DMA-BUF capability does
  // not guarantee that its entire mapped range is exportable by this driver.
  flagcxWindow_t baselineWindow = nullptr;
  flagcxResult_t baseline =
      flagcxCommWindowRegister(comm, allocation, allocationSize,
                               &baselineWindow, FLAGCX_WIN_COLL_SYMMETRIC);
  int baselineMinimum = 0;
  int baselineMaximum = 0;
  allRankResultRange(baseline, &baselineMinimum, &baselineMaximum);
  ASSERT_EQ(baselineMinimum, baselineMaximum);
  if (baseline == flagcxSuccess) {
    ASSERT_NE(baselineWindow, nullptr);
    ASSERT_NE(baselineWindow->defaultBase, nullptr);
    ASSERT_TRUE(baselineWindow->defaultBase->hasNetworkMrRef);
    const int mrIndex = baselineWindow->defaultBase->mrIndex;
    ASSERT_GE(mrIndex, 0);
    const uint8_t route =
        comm->heteroComm->oneSideHandles[mrIndex]->registrationRoute;
    EXPECT_TRUE(route == FLAGCX_VMM_MR_ROUTE_DMABUF ||
                route == FLAGCX_VMM_MR_ROUTE_VA);
    ASSERT_TRUE(
        allRanksSucceeded(flagcxCommWindowDeregister(comm, baselineWindow)));
  } else {
    EXPECT_EQ(baselineWindow, nullptr);
    EXPECT_EQ(comm->heteroComm->symWindows, nullptr);
  }

  savedProviderMapOne = deviceAdaptor->symFlatMapSegmentOne;
  if (comm->heteroComm->localRank == 1)
    deviceAdaptor->symFlatMapSegmentOne = failAfterProviderMapOne;
  flagcxWindow_t window = nullptr;
  flagcxResult_t result = flagcxCommWindowRegister(
      comm, allocation, allocationSize, &window, FLAGCX_WIN_COLL_SYMMETRIC);
  deviceAdaptor->symFlatMapSegmentOne = savedProviderMapOne;

  int minimum = 0;
  int maximum = 0;
  allRankResultRange(result, &minimum, &maximum);
  ASSERT_EQ(minimum, maximum);
  EXPECT_NE(result, flagcxSuccess);
  EXPECT_EQ(window, nullptr);
  EXPECT_EQ(comm->heteroComm->symWindows, nullptr);
  EXPECT_EQ(flagcxMemFree(allocation), flagcxSuccess);
}

TEST_F(SymMemTest, OrdinaryAllocationUsesIpcWithVmmEnabled) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_VMM"))
    GTEST_SKIP() << "Requires the VMM-enabled MPI invocation";
  ASSERT_TRUE(envEnabled("FLAGCX_VMM_ENABLE"));
  ASSERT_NE(comm, nullptr);
  ASSERT_NE(comm->heteroComm, nullptr);
  if (comm->heteroComm->localRanks < 2)
    GTEST_SKIP() << "Requires at least two local ranks";
  if (comm->heteroComm->localRanks != comm->heteroComm->nRanks)
    GTEST_SKIP() << "Requires all ranks on one node";
  ASSERT_NE(deviceAdaptor, nullptr);
  ASSERT_NE(deviceAdaptor->symPhysDescribe, nullptr);
  ASSERT_NE(deviceAdaptor->symPhysAlloc, nullptr);

  void *buffer = nullptr;
  ASSERT_TRUE(allRanksSucceeded(
      deviceAdaptor->deviceMalloc(&buffer, size, flagcxMemDevice, nullptr)));
  for (size_t offset : {size_t(0), size_t(256)}) {
    void *requested = static_cast<char *>(buffer) + offset;
    size_t requestedSize = size - offset;
    void *bundle = nullptr;
    const flagcxSymSegmentMeta *segments = nullptr;
    int segmentCount = 0;
    size_t userOffset = 0;
    size_t alignment = 1;
    EXPECT_EQ(deviceAdaptor->symPhysDescribe(requested, requestedSize, &bundle,
                                             &segments, &segmentCount,
                                             &userOffset, &alignment),
              flagcxNotSupported);
    EXPECT_EQ(bundle, nullptr);
    EXPECT_EQ(segments, nullptr);
    EXPECT_EQ(segmentCount, 0);
    EXPECT_EQ(alignment, 0u);
    if (bundle != nullptr)
      EXPECT_EQ(deviceAdaptor->symPhysFreeSegments(bundle), flagcxSuccess);

    flagcxWindow_t window = nullptr;
    auto originalSymPhysAlloc = deviceAdaptor->symPhysAlloc;
    deviceAdaptor->symPhysAlloc = failSymPhysAlloc;
    flagcxResult_t result = flagcxCommWindowRegister(
        comm, requested, requestedSize, &window, FLAGCX_WIN_COLL_SYMMETRIC);
    deviceAdaptor->symPhysAlloc = originalSymPhysAlloc;
    ASSERT_TRUE(allRanksSucceeded(result));
    ASSERT_NE(window, nullptr);
    ASSERT_NE(window->defaultBase, nullptr);
    EXPECT_NE(window->defaultBase->resourcePath, flagcxSymResourceSegments);
    EXPECT_FALSE(window->defaultBase->hasFlatMapping);
    EXPECT_GE(window->defaultBase->ipcSlot, 0);
    ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowDeregister(comm, window)));
  }
  EXPECT_EQ(deviceAdaptor->deviceFree(buffer, flagcxMemDevice, nullptr),
            flagcxSuccess);
}

TEST_F(SymMemTest, OrdinaryAllocationDoesNotClaimSegmentedVmm) {
  ASSERT_NE(deviceAdaptor, nullptr);
  int minimum = 0;
  int maximum = 0;
  allRankResultRange(deviceAdaptor->symPhysDescribe != nullptr
                         ? flagcxSuccess
                         : flagcxNotSupported,
                     &minimum, &maximum);
  ASSERT_EQ(minimum, maximum);
  if (minimum != static_cast<int>(flagcxSuccess))
    GTEST_SKIP() << "Adaptor has no segmented VMM description callback";
  ASSERT_NE(deviceAdaptor->symPhysFreeSegments, nullptr);
  void *buffer = nullptr;
  ASSERT_TRUE(allRanksSucceeded(
      deviceAdaptor->deviceMalloc(&buffer, size, flagcxMemDevice, nullptr)));
  void *bundle = nullptr;
  const flagcxSymSegmentMeta *segments = nullptr;
  int segmentCount = 0;
  size_t userOffset = 0;
  size_t alignment = 0;
  flagcxResult_t result = deviceAdaptor->symPhysDescribe(
      buffer, size, &bundle, &segments, &segmentCount, &userOffset, &alignment);
  EXPECT_EQ(result, flagcxNotSupported);
  EXPECT_EQ(bundle, nullptr);
  EXPECT_EQ(segments, nullptr);
  EXPECT_EQ(segmentCount, 0);
  EXPECT_EQ(alignment, 0u);
  if (bundle != nullptr)
    EXPECT_EQ(deviceAdaptor->symPhysFreeSegments(bundle), flagcxSuccess);
  EXPECT_EQ(deviceAdaptor->deviceFree(buffer, flagcxMemDevice, nullptr),
            flagcxSuccess);
}

#if defined(USE_NVIDIA_ADAPTOR) || defined(USE_DU_ADAPTOR) ||                  \
    defined(USE_METAX_ADAPTOR) || defined(USE_PPU_ADAPTOR)
TEST_F(SymMemTest, UntrackedVmmMappingIsRejected) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_MULTI_SEGMENT"))
    GTEST_SKIP() << "Requires the provider VMM CI invocation";
  ASSERT_TRUE(envEnabled("FLAGCX_VMM_ENABLE"));
#if defined(USE_METAX_ADAPTOR)
  size_t granularity = 0;
  struct ScopedMapping {
    mcMemGenericAllocationHandle handle = 0;
    mcDeviceptr_t base = nullptr;
    size_t size = 0;
    bool mapped = false;
    ~ScopedMapping() {
      if (mapped)
        mcMemUnmap(base, size);
      if (base != nullptr)
        mcMemAddressFree(base, size);
      if (handle != 0)
        mcMemRelease(handle);
    }
  } mapping;
  const bool setupOk = [&]() -> bool {
    int device = 0;
    if (mcGetDevice(&device) != mcSuccess)
      return false;
    MCdevice mcDevice = 0;
    if (mcDeviceGet(&mcDevice, device) != mcSuccess)
      return false;
    mcMemAllocationProp prop = {};
    prop.type = mcMemAllocationTypePinned;
    prop.location.type = mcMemLocationTypeDevice;
    prop.location.id = mcDevice;
    prop.requestedHandleTypes = mcMemHandleTypePosixFileDescriptor;
    if (mcMemGetAllocationGranularity(&granularity, &prop,
                                      MC_MEM_ALLOC_GRANULARITY_MINIMUM) !=
            mcSuccess ||
        granularity == 0)
      return false;
    mapping.size = granularity;
    if (mcMemCreate(&mapping.handle, granularity, &prop, 0) != mcSuccess ||
        mcMemAddressReserve(&mapping.base, granularity, granularity, 0, 0) !=
            mcSuccess ||
        mcMemMap(mapping.base, granularity, 0, mapping.handle, 0) != mcSuccess)
      return false;
    mapping.mapped = true;
    mcMemAccessDesc access = {};
    access.location.type = mcMemLocationTypeDevice;
    access.location.id = mcDevice;
    access.flags = mcMemAccessFlagsProtReadWrite;
    return mcMemSetAccess(mapping.base, granularity, &access, 1) == mcSuccess;
  }();
#elif defined(USE_DU_ADAPTOR)
  size_t granularity = 0;
  struct ScopedMapping {
    hipMemGenericAllocationHandle_t handle = 0;
    hipDeviceptr_t base = nullptr;
    size_t size = 0;
    bool mapped = false;
    ~ScopedMapping() {
      if (mapped)
        hipMemUnmap(base, size);
      if (base != nullptr)
        hipMemAddressFree(base, size);
      if (handle != 0)
        hipMemRelease(handle);
    }
  } mapping;
  const bool setupOk = [&]() -> bool {
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess)
      return false;
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;
    prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
    if (hipMemGetAllocationGranularity(
            &granularity, &prop, hipMemAllocationGranularityRecommended) !=
            hipSuccess ||
        granularity == 0)
      return false;
    mapping.size = granularity;
    if (hipMemCreate(&mapping.handle, granularity, &prop, 0) != hipSuccess ||
        hipMemAddressReserve(&mapping.base, granularity, granularity, nullptr,
                             0) != hipSuccess ||
        hipMemMap(mapping.base, granularity, 0, mapping.handle, 0) !=
            hipSuccess)
      return false;
    mapping.mapped = true;
    hipMemAccessDesc access = {};
    access.location.type = hipMemLocationTypeDevice;
    access.location.id = device;
    access.flags = hipMemAccessFlagsProtReadWrite;
    return hipMemSetAccess(mapping.base, granularity, &access, 1) == hipSuccess;
  }();
#elif defined(USE_NVIDIA_ADAPTOR) || defined(USE_PPU_ADAPTOR)
  size_t granularity = 0;
  struct ScopedMapping {
    CUmemGenericAllocationHandle handle = 0;
    CUdeviceptr base = 0;
    size_t size = 0;
    bool mapped = false;
    ~ScopedMapping() {
      if (mapped)
        cuMemUnmap(base, size);
      if (base != 0)
        cuMemAddressFree(base, size);
      if (handle != 0)
        cuMemRelease(handle);
    }
  } mapping;
  const bool setupOk = [&]() -> bool {
    int cudaDev = 0;
    if (cudaGetDevice(&cudaDev) != cudaSuccess)
      return false;
    CUdevice cuDevice = 0;
    if (cuDeviceGet(&cuDevice, cudaDev) != CUDA_SUCCESS)
      return false;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = cuDevice;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    if (cuMemGetAllocationGranularity(&granularity, &prop,
                                      CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) !=
            CUDA_SUCCESS ||
        granularity == 0)
      return false;
    mapping.size = granularity;
    if (cuMemCreate(&mapping.handle, granularity, &prop, 0) != CUDA_SUCCESS ||
        cuMemAddressReserve(&mapping.base, granularity, granularity, 0, 0) !=
            CUDA_SUCCESS ||
        cuMemMap(mapping.base, granularity, 0, mapping.handle, 0) !=
            CUDA_SUCCESS)
      return false;
    mapping.mapped = true;
    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = cuDevice;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    return cuMemSetAccess(mapping.base, granularity, &access, 1) ==
           CUDA_SUCCESS;
  }();
#endif

  EXPECT_TRUE(setupOk);
  ASSERT_TRUE(
      allRanksSucceeded(setupOk ? flagcxSuccess : flagcxUnhandledDeviceError));

  void *bundle = nullptr;
  const flagcxSymSegmentMeta *segments = nullptr;
  int segmentCount = 0;
  size_t userOffset = 0;
  size_t alignment = 0;
  flagcxResult_t description = deviceAdaptor->symPhysDescribe(
      (void *)mapping.base, granularity, &bundle, &segments, &segmentCount,
      &userOffset, &alignment);
  EXPECT_EQ(description, flagcxNotSupported);
  EXPECT_NE(bundle, nullptr);
  EXPECT_EQ(segments, nullptr);
  EXPECT_EQ(segmentCount, 0);
  const bool hasBundle = bundle != nullptr;
  flagcxResult_t cleanup = flagcxSuccess;
  if (bundle != nullptr)
    cleanup = deviceAdaptor->symPhysFreeSegments(bundle);
  EXPECT_EQ(cleanup, flagcxSuccess);
  ASSERT_TRUE(allRanksSucceeded(description == flagcxNotSupported &&
                                        hasBundle && cleanup == flagcxSuccess
                                    ? flagcxSuccess
                                    : flagcxInvalidUsage));

  // An untracked VMM mapping must not be registered through ordinary IPC.
  flagcxWindow_t window = nullptr;
  flagcxResult_t result =
      flagcxCommWindowRegister(comm, (void *)mapping.base, granularity, &window,
                               FLAGCX_WIN_COLL_SYMMETRIC);
  int minimum = 0;
  int maximum = 0;
  allRankResultRange(result, &minimum, &maximum);
  EXPECT_EQ(minimum, maximum);
  EXPECT_NE(result, flagcxSuccess);
  EXPECT_EQ(window, nullptr);
  ASSERT_NE(comm->heteroComm, nullptr);
  EXPECT_EQ(comm->heteroComm->symWindows, nullptr);
  if (minimum == maximum && window != nullptr)
    EXPECT_TRUE(allRanksSucceeded(flagcxCommWindowDeregister(comm, window)));
}
#endif

#ifdef USE_DU_ADAPTOR
TEST_F(SymMemTest, DuSegmentedAllocFailureRollsBackAndCanRetry) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_MULTI_SEGMENT"))
    GTEST_SKIP() << "Requires the DU multi-segment VMM CI invocation";
  ASSERT_NE(deviceAdaptor, nullptr);
  ASSERT_NE(deviceAdaptor->gdrMemAlloc, nullptr);
  ASSERT_NE(deviceAdaptor->gdrMemFree, nullptr);

  int hipDevice = 0;
  ASSERT_EQ(hipGetDevice(&hipDevice), hipSuccess);
  hipMemAllocationProp prop = {};
  prop.type = hipMemAllocationTypePinned;
  prop.location.type = hipMemLocationTypeDevice;
  prop.location.id = hipDevice;
  prop.requestedHandleTypes = hipMemHandleTypePosixFileDescriptor;
  size_t granularity = 0;
  ASSERT_EQ(hipMemGetAllocationGranularity(
                &granularity, &prop, hipMemAllocationGranularityRecommended),
            hipSuccess);
  ASSERT_GT(granularity, 0u);
  ASSERT_LE(granularity, SIZE_MAX / 3);
  const size_t allocationSize = granularity * 3;
  const std::string segmentSetting = std::to_string(granularity);
  ScopedEnvVar segmentSize("FLAGCX_VMM_SEGMENT_BYTES", segmentSetting.c_str());
  void *allocation = (void *)1;
  segmentSize.set("-1");
  EXPECT_EQ(deviceAdaptor->gdrMemAlloc(&allocation, allocationSize, nullptr),
            flagcxInvalidArgument);
  EXPECT_EQ(allocation, nullptr);

  if (granularity > 1) {
    ASSERT_LT(allocationSize, SIZE_MAX);
    const std::string invalidLargeSegment = std::to_string(allocationSize + 1);
    segmentSize.set(invalidLargeSegment.c_str());
    allocation = (void *)1;
    EXPECT_EQ(deviceAdaptor->gdrMemAlloc(&allocation, allocationSize, nullptr),
              flagcxInvalidArgument);
    EXPECT_EQ(allocation, nullptr);
  }

  segmentSize.set(segmentSetting.c_str());
  ScopedEnvVar failAfterMap("FLAGCX_CI_DU_VMM_FAIL_AFTER_MAP", "1");
  allocation = (void *)1;
  EXPECT_EQ(deviceAdaptor->gdrMemAlloc(&allocation, allocationSize, nullptr),
            flagcxUnhandledDeviceError);
  EXPECT_EQ(allocation, nullptr);

  failAfterMap.set(nullptr);
  ASSERT_EQ(deviceAdaptor->gdrMemAlloc(&allocation, allocationSize, nullptr),
            flagcxSuccess);
  ASSERT_NE(allocation, nullptr);
  EXPECT_EQ(hipMemset(allocation, 0x5a, allocationSize), hipSuccess);
  EXPECT_EQ(deviceAdaptor->gdrMemFree(allocation, nullptr), flagcxSuccess);
}
#endif
