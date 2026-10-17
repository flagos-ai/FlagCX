// MPI tests for symmetric memory cross-GPU access.
// Verifies that the flat VA mapping allows direct peer reads/writes.
// Requires MPI + GPUs with P2P support.

#include "adaptor.h"
#include "global_comm.h"
#include "onesided_types.h"
#include "sym_heap.h"
#include "symmem_test.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

bool allRanksReady(bool localReady) {
  int local = localReady ? 1 : 0;
  int global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  return global != 0;
}

bool allRanksSucceeded(flagcxResult_t localResult) {
  return allRanksReady(localResult == flagcxSuccess);
}

bool envEnabled(const char *name) {
  const char *value = std::getenv(name);
  return value != nullptr && std::strcmp(value, "0") != 0;
}

flagcxResult_t (*savedLsaMulticastBind)(void *, size_t, const void *,
                                        size_t) = nullptr;
bool failThisLocalRankLsaMulticastBind = false;
flagcxResult_t (*savedLsaMulticastUnbind)(void *, size_t, size_t) = nullptr;
bool failThisLocalRankLsaMulticastUnbind = false;

flagcxResult_t failLsaMulticastBindOnLocalRankZero(void *handle, size_t offset,
                                                   const void *source,
                                                   size_t size) {
  if (failThisLocalRankLsaMulticastBind)
    return flagcxUnhandledDeviceError;
  return savedLsaMulticastBind != nullptr
             ? savedLsaMulticastBind(handle, offset, source, size)
             : flagcxInternalError;
}

flagcxResult_t failLsaMulticastUnbindOnLocalRankZero(void *handle,
                                                     size_t offset,
                                                     size_t size) {
  if (failThisLocalRankLsaMulticastUnbind)
    return flagcxUnhandledDeviceError;
  return savedLsaMulticastUnbind != nullptr
             ? savedLsaMulticastUnbind(handle, offset, size)
             : flagcxInternalError;
}

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

} // namespace

// ---------------------------------------------------------------------------
// IPC-fallback coverage.  Unlike the VMM tests below, these tests resolve the
// peer pointer through the window's IPC table and require no network MR.
// ---------------------------------------------------------------------------

TEST_F(SymMemTest, CrossGpuReadViaIpcPeerPtr) {
  flagcxWindow_t win = nullptr;
  flagcxResult_t res = flagcxCommWindowRegister(comm, devBuff, size, &win,
                                                FLAGCX_WIN_COLL_SYMMETRIC);
  ASSERT_TRUE(allRanksSucceeded(res));

  const bool requireNetMr = envEnabled("FLAGCX_CI_REQUIRE_NET_MR");
  bool localReady = win != nullptr && win->defaultBase != nullptr &&
                    !win->defaultBase->hasFlatMapping &&
                    win->defaultBase->localRanks >= 2 &&
                    win->defaultBase->ipcSlot >= 0 &&
                    (requireNetMr ? win->defaultBase->mrIndex >= 0
                                  : win->defaultBase->mrIndex < 0) &&
                    hasHeteroComm();
  ASSERT_TRUE(allRanksReady(localReady))
      << "IPC mode has the wrong IPC locator or network-MR state";

  flagcxSymWindow_t window = win->defaultBase;
  int localRank = comm->heteroComm->localRank;
  int peerLocalRank = (localRank + 1) % window->localRanks;
  int peerGlobalRank = comm->localRankToRank[peerLocalRank];

  std::vector<float> pattern(count, static_cast<float>(localRank + 1));
  res = devHandle->deviceMemcpy(devBuff, pattern.data(), size,
                                flagcxMemcpyHostToDevice, stream);
  ASSERT_TRUE(allRanksSucceeded(res));
  ASSERT_TRUE(allRanksSucceeded(devHandle->streamSynchronize(stream)));
  MPI_Barrier(MPI_COMM_WORLD);

  void *peerPtr = nullptr;
  res = flagcxSymWindowResolveIpcPeerPtr(comm->heteroComm, window,
                                         peerGlobalRank, 0, size, &peerPtr);
  ASSERT_TRUE(allRanksSucceeded(res));
  ASSERT_TRUE(allRanksReady(peerPtr != nullptr));

  res = devHandle->deviceMemcpy(devBuff2, peerPtr, size,
                                flagcxMemcpyDeviceToDevice, stream);
  ASSERT_TRUE(allRanksSucceeded(res));
  std::vector<float> readBack(count, 0.0f);
  res = devHandle->deviceMemcpy(readBack.data(), devBuff2, size,
                                flagcxMemcpyDeviceToHost, stream);
  ASSERT_TRUE(allRanksSucceeded(res));
  ASSERT_TRUE(allRanksSucceeded(devHandle->streamSynchronize(stream)));

  const float expected = static_cast<float>(peerLocalRank + 1);
  EXPECT_EQ(std::count(readBack.begin(), readBack.end(), expected), count);
  MPI_Barrier(MPI_COMM_WORLD);
  EXPECT_EQ(flagcxCommWindowDeregister(comm, win), flagcxSuccess);
}

TEST_F(SymMemTest, CrossGpuWriteViaIpcPeerPtr) {
  flagcxWindow_t win = nullptr;
  flagcxResult_t res = flagcxCommWindowRegister(comm, devBuff, size, &win,
                                                FLAGCX_WIN_COLL_SYMMETRIC);
  ASSERT_TRUE(allRanksSucceeded(res));

  const bool requireNetMr = envEnabled("FLAGCX_CI_REQUIRE_NET_MR");
  bool localReady = win != nullptr && win->defaultBase != nullptr &&
                    !win->defaultBase->hasFlatMapping &&
                    win->defaultBase->localRanks >= 2 &&
                    win->defaultBase->ipcSlot >= 0 &&
                    (requireNetMr ? win->defaultBase->mrIndex >= 0
                                  : win->defaultBase->mrIndex < 0) &&
                    hasHeteroComm();
  ASSERT_TRUE(allRanksReady(localReady))
      << "IPC mode has the wrong IPC locator or network-MR state";

  flagcxSymWindow_t window = win->defaultBase;
  int localRank = comm->heteroComm->localRank;
  int targetLocalRank = (localRank + 1) % window->localRanks;
  int targetGlobalRank = comm->localRankToRank[targetLocalRank];

  res = devHandle->deviceMemset(devBuff, 0, size, flagcxMemDevice, stream);
  ASSERT_TRUE(allRanksSucceeded(res));
  std::vector<float> pattern(count, static_cast<float>(localRank + 100));
  res = devHandle->deviceMemcpy(devBuff2, pattern.data(), size,
                                flagcxMemcpyHostToDevice, stream);
  ASSERT_TRUE(allRanksSucceeded(res));
  ASSERT_TRUE(allRanksSucceeded(devHandle->streamSynchronize(stream)));
  MPI_Barrier(MPI_COMM_WORLD);

  void *peerPtr = nullptr;
  res = flagcxSymWindowResolveIpcPeerPtr(comm->heteroComm, window,
                                         targetGlobalRank, 0, size, &peerPtr);
  ASSERT_TRUE(allRanksSucceeded(res));
  ASSERT_TRUE(allRanksReady(peerPtr != nullptr));
  res = devHandle->deviceMemcpy(peerPtr, devBuff2, size,
                                flagcxMemcpyDeviceToDevice, stream);
  ASSERT_TRUE(allRanksSucceeded(res));
  ASSERT_TRUE(allRanksSucceeded(devHandle->streamSynchronize(stream)));
  MPI_Barrier(MPI_COMM_WORLD);

  int writerLocalRank =
      (localRank + window->localRanks - 1) % window->localRanks;
  const float expected = static_cast<float>(writerLocalRank + 100);
  std::vector<float> readBack(count, 0.0f);
  res = devHandle->deviceMemcpy(readBack.data(), devBuff, size,
                                flagcxMemcpyDeviceToHost, stream);
  ASSERT_TRUE(allRanksSucceeded(res));
  ASSERT_TRUE(allRanksSucceeded(devHandle->streamSynchronize(stream)));
  EXPECT_EQ(std::count(readBack.begin(), readBack.end(), expected), count);

  MPI_Barrier(MPI_COMM_WORLD);
  EXPECT_EQ(flagcxCommWindowDeregister(comm, win), flagcxSuccess);
}

// ---------------------------------------------------------------------------
// Each rank writes a pattern, then reads from the next peer's region
// via the flat VA and verifies correctness.
// ---------------------------------------------------------------------------

TEST_F(SymMemTest, CrossGpuReadViaPeerPtr) {
  flagcxWindow_t win = nullptr;

  ASSERT_EQ(flagcxCommWindowRegister(comm, devBuff, size, &win,
                                     FLAGCX_WIN_COLL_SYMMETRIC),
            flagcxSuccess);
  ASSERT_NE(win, nullptr);
  ASSERT_NE(win->defaultBase, nullptr);

  flagcxSymWindow_t d = win->defaultBase;
  if (!d->hasFlatMapping || d->flatBase == nullptr) {
    flagcxCommWindowDeregister(comm, win);
    if (envEnabled("FLAGCX_CI_REQUIRE_VMM"))
      FAIL() << "FLAGCX_CI_REQUIRE_VMM forbids IPC fallback";
    GTEST_SKIP() << "VMM not available, cannot test flat VA access";
  }
  ASSERT_EQ(d->mrIndex >= 0, envEnabled("FLAGCX_CI_REQUIRE_NET_MR"));
  if (!hasHeteroComm()) {
    flagcxCommWindowDeregister(comm, win);
    GTEST_SKIP() << "heteroComm not available";
  }

  int localRank = comm->heteroComm->localRank;
  int localRanks = d->localRanks;

  // Each rank fills its own region with (localRank + 1) as a float pattern
  float fillValue = (float)(localRank + 1);
  std::vector<float> pattern(count, fillValue);
  flagcxResult_t setupResult = devHandle->deviceMemcpy(
      devBuff, pattern.data(), size, flagcxMemcpyHostToDevice, stream);
  ASSERT_TRUE(allRanksSucceeded(setupResult));
  ASSERT_TRUE(allRanksSucceeded(devHandle->streamSynchronize(stream)));

  MPI_Barrier(MPI_COMM_WORLD);

  // Read from the next peer's region in the flat VA
  int peerLocalRank = (localRank + 1) % localRanks;
  void *peerRegion = nullptr;
  ASSERT_EQ(flagcxSymWindowResolveIpcPeerPtr(
                comm->heteroComm, d,
                comm->heteroComm->localRankToRank[peerLocalRank], 0, size,
                &peerRegion),
            flagcxSuccess);

  // Stage through local device memory: peer VMM -> devBuff2 -> host
  std::vector<float> readBack(count, 0.0f);
  flagcxResult_t copyResult = devHandle->deviceMemcpy(
      devBuff2, peerRegion, size, flagcxMemcpyDeviceToDevice, stream);
  if (copyResult == flagcxSuccess)
    copyResult = devHandle->deviceMemcpy(readBack.data(), devBuff2, size,
                                         flagcxMemcpyDeviceToHost, stream);
  if (copyResult == flagcxSuccess)
    copyResult = devHandle->streamSynchronize(stream);
  ASSERT_TRUE(allRanksSucceeded(copyResult));

  // Verify: peer's region should contain (peerLocalRank + 1)
  float expected = (float)(peerLocalRank + 1);
  int mismatches = 0;
  for (size_t i = 0; i < count && mismatches < 10; i++) {
    if (readBack[i] != expected) {
      mismatches++;
      if (mismatches == 1) {
        EXPECT_FLOAT_EQ(readBack[i], expected)
            << "Mismatch at index " << i << " reading from peer "
            << peerLocalRank;
      }
    }
  }
  EXPECT_EQ(mismatches, 0) << "Total mismatches reading peer " << peerLocalRank
                           << "'s region";

  MPI_Barrier(MPI_COMM_WORLD);

  EXPECT_EQ(flagcxCommWindowDeregister(comm, win), flagcxSuccess);
}

// ---------------------------------------------------------------------------
// Each rank writes to the next peer's region, then verifies its own
// region was written by the previous peer.
// ---------------------------------------------------------------------------

TEST_F(SymMemTest, CrossGpuWriteViaPeerPtr) {
  flagcxWindow_t win = nullptr;

  ASSERT_EQ(flagcxCommWindowRegister(comm, devBuff, size, &win,
                                     FLAGCX_WIN_COLL_SYMMETRIC),
            flagcxSuccess);
  ASSERT_NE(win, nullptr);
  ASSERT_NE(win->defaultBase, nullptr);

  flagcxSymWindow_t d = win->defaultBase;
  if (!d->hasFlatMapping || d->flatBase == nullptr) {
    flagcxCommWindowDeregister(comm, win);
    if (envEnabled("FLAGCX_CI_REQUIRE_VMM"))
      FAIL() << "FLAGCX_CI_REQUIRE_VMM forbids IPC fallback";
    GTEST_SKIP() << "VMM not available, cannot test flat VA access";
  }
  ASSERT_EQ(d->mrIndex >= 0, envEnabled("FLAGCX_CI_REQUIRE_NET_MR"));
  if (!hasHeteroComm()) {
    flagcxCommWindowDeregister(comm, win);
    GTEST_SKIP() << "heteroComm not available";
  }

  int localRank = comm->heteroComm->localRank;
  int localRanks = d->localRanks;

  // Zero out own region first
  flagcxResult_t setupResult =
      devHandle->deviceMemset(devBuff, 0, size, flagcxMemDevice, stream);
  ASSERT_TRUE(allRanksSucceeded(setupResult));
  ASSERT_TRUE(allRanksSucceeded(devHandle->streamSynchronize(stream)));

  MPI_Barrier(MPI_COMM_WORLD);

  // Write pattern (localRank + 100) into the NEXT peer's region.
  // Use devBuff2 as staging: host -> local device -> peer VMM region.
  // Direct host-to-peer-VMM may not work with cudaMemcpyHostToDevice.
  int targetLocalRank = (localRank + 1) % localRanks;
  void *targetRegion = nullptr;
  ASSERT_EQ(flagcxSymWindowResolveIpcPeerPtr(
                comm->heteroComm, d,
                comm->heteroComm->localRankToRank[targetLocalRank], 0, size,
                &targetRegion),
            flagcxSuccess);

  float writeValue = (float)(localRank + 100);
  std::vector<float> pattern(count, writeValue);
  flagcxResult_t copyResult = devHandle->deviceMemcpy(
      devBuff2, pattern.data(), size, flagcxMemcpyHostToDevice, stream);
  if (copyResult == flagcxSuccess)
    copyResult = devHandle->deviceMemcpy(targetRegion, devBuff2, size,
                                         flagcxMemcpyDeviceToDevice, stream);
  if (copyResult == flagcxSuccess)
    copyResult = devHandle->streamSynchronize(stream);
  ASSERT_TRUE(allRanksSucceeded(copyResult));

  MPI_Barrier(MPI_COMM_WORLD);

  // Now read back our own region — it should have been written by prev peer
  int writerLocalRank = (localRank + localRanks - 1) % localRanks;
  float expected = (float)(writerLocalRank + 100);

  std::vector<float> readBack(count, 0.0f);
  copyResult = devHandle->deviceMemcpy(readBack.data(), devBuff, size,
                                       flagcxMemcpyDeviceToHost, stream);
  if (copyResult == flagcxSuccess)
    copyResult = devHandle->streamSynchronize(stream);
  ASSERT_TRUE(allRanksSucceeded(copyResult));

  int mismatches = 0;
  for (size_t i = 0; i < count && mismatches < 10; i++) {
    if (readBack[i] != expected) {
      mismatches++;
      if (mismatches == 1) {
        EXPECT_FLOAT_EQ(readBack[i], expected)
            << "Mismatch at index " << i << ", expected write from rank "
            << writerLocalRank;
      }
    }
  }
  EXPECT_EQ(mismatches, 0) << "Total mismatches in own region after peer write";

  MPI_Barrier(MPI_COMM_WORLD);

  EXPECT_EQ(flagcxCommWindowDeregister(comm, win), flagcxSuccess);
}

// A four-rank/two-logical-node invocation must expose both paths from the same
// window: flat/IPC for a local peer and an MR-backed GET for a remote peer.
TEST_F(SymMemTest, HybridLocalAndRemoteAccess) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_NET_MR"))
    GTEST_SKIP() << "Runs only in the required NET invocation";

  flagcxWindow_t win = nullptr;
  flagcxResult_t res = flagcxCommWindowRegister(comm, devBuff, size, &win,
                                                FLAGCX_WIN_COLL_SYMMETRIC);
  ASSERT_TRUE(allRanksSucceeded(res));
  ASSERT_NE(win, nullptr);
  ASSERT_NE(win->defaultBase, nullptr);
  flagcxSymWindow_t window = win->defaultBase;

  if (envEnabled("FLAGCX_CI_REQUIRE_VMM")) {
    ASSERT_TRUE(
        allRanksReady(window->hasFlatMapping && window->flatBase != nullptr))
        << "required VMM invocation fell back to IPC";
  } else {
    ASSERT_TRUE(allRanksReady(!window->hasFlatMapping && window->ipcSlot >= 0))
        << "required IPC invocation has no local mapping";
  }
  ASSERT_TRUE(allRanksReady(window->mrIndex >= 0))
      << "required NET invocation did not register an MR";
  ASSERT_LT(window->mrIndex, comm->heteroComm->oneSideHandleCount);
  auto *mrHandle = comm->heteroComm->oneSideHandles[window->mrIndex];
  ASSERT_NE(mrHandle, nullptr);
  if (envEnabled("FLAGCX_CI_REQUIRE_VMM")) {
    EXPECT_TRUE(vmmMrRouteMatches(mrHandle->registrationRoute));
  }
  const char *expectedAdaptor = std::getenv("FLAGCX_CI_EXPECT_NET_ADAPTOR");
  bool adaptorMatches =
      expectedAdaptor == nullptr ||
      (comm->heteroComm->netAdaptor != nullptr &&
       comm->heteroComm->netAdaptor->name != nullptr &&
       std::strcmp(comm->heteroComm->netAdaptor->name, expectedAdaptor) == 0);
  ASSERT_TRUE(allRanksReady(adaptorMatches))
      << "required network adaptor was not selected";
  ASSERT_TRUE(allRanksReady(window->localRanks >= 2 &&
                            window->localRanks < comm->nranks));

  const size_t transferSize = 256;
  const size_t srcOffset = 128;
  const size_t dstOffset = 4096;
  std::vector<uint8_t> pattern(transferSize,
                               static_cast<uint8_t>(comm->rank + 17));
  res = devHandle->deviceMemset(devBuff, 0, size, flagcxMemDevice, stream);
  if (res == flagcxSuccess)
    res =
        devHandle->deviceMemcpy((char *)devBuff + srcOffset, pattern.data(),
                                transferSize, flagcxMemcpyHostToDevice, stream);
  if (res == flagcxSuccess)
    res = devHandle->streamSynchronize(stream);
  ASSERT_TRUE(allRanksSucceeded(res));
  MPI_Barrier(MPI_COMM_WORLD);

  const int localRank = comm->heteroComm->localRank;
  const int localPeerRank = (localRank + 1) % window->localRanks;
  const int localPeer = comm->localRankToRank[localPeerRank];
  int remotePeer = -1;
  for (int peer = 0; peer < comm->nranks; peer++) {
    if (comm->heteroComm->rankToNode[peer] != comm->heteroComm->node) {
      remotePeer = peer;
      break;
    }
  }
  ASSERT_TRUE(allRanksReady(remotePeer >= 0));

  void *localPeerPtr = nullptr;
  res =
      flagcxSymWindowResolveIpcPeerPtr(comm->heteroComm, window, localPeer,
                                       srcOffset, transferSize, &localPeerPtr);
  if (res == flagcxSuccess)
    res = devHandle->deviceMemcpy(devBuff2, localPeerPtr, transferSize,
                                  flagcxMemcpyDeviceToDevice, stream);
  std::vector<uint8_t> localRead(transferSize, 0);
  if (res == flagcxSuccess)
    res = devHandle->deviceMemcpy(localRead.data(), devBuff2, transferSize,
                                  flagcxMemcpyDeviceToHost, stream);
  if (res == flagcxSuccess)
    res = devHandle->streamSynchronize(stream);
  ASSERT_TRUE(allRanksSucceeded(res));
  EXPECT_EQ(localRead, std::vector<uint8_t>(
                           transferSize, static_cast<uint8_t>(localPeer + 17)));

  uint64_t before = 0;
  res = flagcxReadCounter(comm, &before);
  if (res == flagcxSuccess)
    res = flagcxGet(comm, remotePeer, srcOffset, dstOffset, transferSize,
                    window->mrIndex, window->mrIndex);
  if (res == flagcxSuccess)
    res = flagcxWaitCounter(comm, before + 1);
  ASSERT_TRUE(allRanksSucceeded(res));

  // Do not add an extra deviceSynchronize after flagcxWaitCounter: the host
  // copy must observe data based on the RMA completion contract itself.
  std::vector<uint8_t> remoteRead(transferSize, 0);
  res =
      devHandle->deviceMemcpy(remoteRead.data(), (char *)devBuff + dstOffset,
                              transferSize, flagcxMemcpyDeviceToHost, nullptr);
  ASSERT_TRUE(allRanksSucceeded(res));
  EXPECT_EQ(remoteRead,
            std::vector<uint8_t>(transferSize,
                                 static_cast<uint8_t>(remotePeer + 17)));

  MPI_Barrier(MPI_COMM_WORLD);
  EXPECT_EQ(flagcxCommWindowDeregister(comm, win), flagcxSuccess);
}

// ---------------------------------------------------------------------------
// Verify multicast base is set when VMM succeeds
// ---------------------------------------------------------------------------

TEST_F(SymMemTest, MulticastMappingWhenSupported) {
  flagcxWindow_t win = nullptr;
  flagcxResult_t res = flagcxCommWindowRegister(comm, devBuff, size, &win,
                                                FLAGCX_WIN_COLL_SYMMETRIC);
  ASSERT_TRUE(allRanksSucceeded(res));
  ASSERT_NE(win, nullptr);
  ASSERT_NE(win->defaultBase, nullptr);

  if (!win->defaultBase->hasFlatMapping) {
    EXPECT_EQ(flagcxCommWindowDeregister(comm, win), flagcxSuccess);
    if (envEnabled("FLAGCX_CI_REQUIRE_VMM"))
      FAIL() << "required VMM invocation fell back to IPC";
    GTEST_SKIP() << "multicast requires VMM";
  }

  int localSupported = 0;
  res = deviceAdaptor->symMulticastSupported != nullptr
            ? deviceAdaptor->symMulticastSupported(&localSupported)
            : flagcxNotSupported;
  if (res == flagcxNotSupported) {
    localSupported = 0;
    res = flagcxSuccess;
  }
  ASSERT_TRUE(allRanksSucceeded(res));
  bool supportedEverywhere = allRanksReady(localSupported != 0);
  if (!supportedEverywhere) {
    EXPECT_EQ(flagcxCommWindowDeregister(comm, win), flagcxSuccess);
    if (envEnabled("FLAGCX_CI_REQUIRE_MULTICAST"))
      FAIL() << "multicast capability was required but unavailable";
    GTEST_SKIP() << "multicast capability is not present on every rank";
  }

  bool mappingReady = allRanksReady(
      win->defaultBase->resourcePath == flagcxSymResourceLsa &&
      win->defaultBase->mcBase != nullptr && win->defaultBase->mcLsaBound);
  if (!mappingReady) {
    EXPECT_EQ(flagcxCommWindowDeregister(comm, win), flagcxSuccess);
    if (envEnabled("FLAGCX_CI_REQUIRE_MULTICAST"))
      FAIL() << "multicast LSA binding was required but unavailable";
    GTEST_SKIP() << "multicast is unsupported by this device topology";
  }
  EXPECT_EQ(win->defaultBase->mcHandle, nullptr);
  EXPECT_EQ(flagcxCommWindowDeregister(comm, win), flagcxSuccess);
}

TEST_F(SymMemTest, LsaMulticastOffsetsReuseAndBindFailureFallsBack) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_VMM"))
    GTEST_SKIP() << "requires VMM LSA";
  flagcxWindow_t first = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowRegister(
      comm, devBuff, size, &first, FLAGCX_WIN_COLL_SYMMETRIC)));
  ASSERT_NE(first, nullptr);
  if (!allRanksReady(first->defaultBase->mcLsaBound)) {
    ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowDeregister(comm, first)));
    if (envEnabled("FLAGCX_CI_REQUIRE_MULTICAST"))
      FAIL() << "multicast LSA binding was required but unavailable";
    GTEST_SKIP() << "communicator multicast arena is unavailable";
  }

  flagcxWindow_t second = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowRegister(
      comm, devBuff2, size, &second, FLAGCX_WIN_COLL_SYMMETRIC)));
  ASSERT_NE(second, nullptr);
  ASSERT_TRUE(allRanksReady(second->defaultBase->mcLsaBound));
  const size_t firstOffset = first->defaultBase->lsaWindowOffset;
  const uintptr_t firstMc =
      reinterpret_cast<uintptr_t>(first->defaultBase->mcBase);
  const uintptr_t secondMc =
      reinterpret_cast<uintptr_t>(second->defaultBase->mcBase);
  EXPECT_NE(firstOffset, second->defaultBase->lsaWindowOffset);
  EXPECT_EQ(firstMc - firstOffset -
                first->defaultBase->rankLayouts[comm->heteroComm->localRank]
                    .userOffset,
            secondMc - second->defaultBase->lsaWindowOffset -
                second->defaultBase->rankLayouts[comm->heteroComm->localRank]
                    .userOffset);

  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowDeregister(comm, first)));
  first = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowRegister(
      comm, devBuff, size, &first, FLAGCX_WIN_COLL_SYMMETRIC)));
  ASSERT_NE(first, nullptr);
  ASSERT_TRUE(allRanksReady(first->defaultBase->mcLsaBound));
  EXPECT_EQ(first->defaultBase->lsaWindowOffset, firstOffset);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(first->defaultBase->mcBase), firstMc);
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowDeregister(comm, first)));
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowDeregister(comm, second)));

  ASSERT_NE(deviceAdaptor->symLsaMulticastBind, nullptr);
  savedLsaMulticastBind = deviceAdaptor->symLsaMulticastBind;
  failThisLocalRankLsaMulticastBind = comm->heteroComm->localRank == 0;
  deviceAdaptor->symLsaMulticastBind = failLsaMulticastBindOnLocalRankZero;
  flagcxWindow_t fallback = nullptr;
  flagcxResult_t result = flagcxCommWindowRegister(
      comm, devBuff, size, &fallback, FLAGCX_WIN_COLL_SYMMETRIC);
  deviceAdaptor->symLsaMulticastBind = savedLsaMulticastBind;
  savedLsaMulticastBind = nullptr;
  failThisLocalRankLsaMulticastBind = false;
  ASSERT_TRUE(allRanksSucceeded(result));
  ASSERT_NE(fallback, nullptr);
  EXPECT_EQ(fallback->defaultBase->resourcePath, flagcxSymResourceLsa);
  EXPECT_TRUE(fallback->defaultBase->hasFlatMapping);
  EXPECT_FALSE(fallback->defaultBase->mcLsaBound);
  EXPECT_EQ(fallback->defaultBase->mcBase, nullptr);
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowDeregister(comm, fallback)));
}

TEST_F(SymMemTest, LsaMulticastUnbindFailureRetainsWindowForRetry) {
  if (!envEnabled("FLAGCX_CI_REQUIRE_VMM"))
    GTEST_SKIP() << "requires VMM LSA";
  flagcxWindow_t window = nullptr;
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowRegister(
      comm, devBuff, size, &window, FLAGCX_WIN_COLL_SYMMETRIC)));
  ASSERT_NE(window, nullptr);
  if (!allRanksReady(window->defaultBase->mcLsaBound)) {
    ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowDeregister(comm, window)));
    if (envEnabled("FLAGCX_CI_REQUIRE_MULTICAST"))
      FAIL() << "multicast LSA binding was required but unavailable";
    GTEST_SKIP() << "communicator multicast arena is unavailable";
  }

  ASSERT_NE(deviceAdaptor->symLsaMulticastUnbind, nullptr);
  savedLsaMulticastUnbind = deviceAdaptor->symLsaMulticastUnbind;
  failThisLocalRankLsaMulticastUnbind = comm->heteroComm->localRank == 0;
  deviceAdaptor->symLsaMulticastUnbind = failLsaMulticastUnbindOnLocalRankZero;
  flagcxResult_t result = flagcxCommWindowDeregister(comm, window);
  deviceAdaptor->symLsaMulticastUnbind = savedLsaMulticastUnbind;
  savedLsaMulticastUnbind = nullptr;
  failThisLocalRankLsaMulticastUnbind = false;

  int local = static_cast<int>(result);
  int minimum = 0;
  int maximum = 0;
  MPI_Allreduce(&local, &minimum, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local, &maximum, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  EXPECT_EQ(minimum, maximum);
  EXPECT_NE(result, flagcxSuccess);
  ASSERT_TRUE(allRanksSucceeded(flagcxCommWindowDeregister(comm, window)));
}
