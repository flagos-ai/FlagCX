// MPI tests for symmetric memory cross-GPU access.
// Verifies that the flat VA mapping allows direct peer reads/writes.
// Requires MPI + GPUs with P2P support.

#include "global_comm.h"
#include "sym_heap.h"
#include "symmem_test.hpp"
#include <algorithm>
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

  bool localReady = win != nullptr && win->defaultBase != nullptr &&
                    !win->defaultBase->isVMM &&
                    win->defaultBase->localRanks >= 2 &&
                    win->defaultBase->ipcSlot >= 0 &&
                    win->defaultBase->mrIndex < 0 && hasHeteroComm();
  ASSERT_TRUE(allRanksReady(localReady))
      << "IPC fallback requires ipcSlot >= 0 and mrIndex < 0 on every rank";

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

  bool localReady = win != nullptr && win->defaultBase != nullptr &&
                    !win->defaultBase->isVMM &&
                    win->defaultBase->localRanks >= 2 &&
                    win->defaultBase->ipcSlot >= 0 &&
                    win->defaultBase->mrIndex < 0 && hasHeteroComm();
  ASSERT_TRUE(allRanksReady(localReady))
      << "IPC fallback requires ipcSlot >= 0 and mrIndex < 0 on every rank";

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
  if (!d->isVMM || d->flatBase == nullptr) {
    flagcxCommWindowDeregister(comm, win);
    GTEST_SKIP() << "VMM not available, cannot test flat VA access";
  }
  if (!hasHeteroComm()) {
    flagcxCommWindowDeregister(comm, win);
    GTEST_SKIP() << "heteroComm not available";
  }

  int localRank = comm->heteroComm->localRank;
  int localRanks = d->localRanks;
  size_t allocSize = d->allocSize;

  // Each rank fills its own region with (localRank + 1) as a float pattern
  float fillValue = (float)(localRank + 1);
  std::vector<float> pattern(count, fillValue);
  devHandle->deviceMemcpy(devBuff, pattern.data(), size,
                          flagcxMemcpyHostToDevice, stream);
  devHandle->streamSynchronize(stream);

  MPI_Barrier(MPI_COMM_WORLD);

  // Read from the next peer's region in the flat VA
  int peerLocalRank = (localRank + 1) % localRanks;
  void *peerRegion = (char *)d->flatBase + (size_t)peerLocalRank * allocSize;

  // Stage through local device memory: peer VMM -> devBuff2 -> host
  std::vector<float> readBack(count, 0.0f);
  devHandle->deviceMemcpy(devBuff2, peerRegion, size,
                          flagcxMemcpyDeviceToDevice, stream);
  devHandle->deviceMemcpy(readBack.data(), devBuff2, size,
                          flagcxMemcpyDeviceToHost, stream);
  devHandle->streamSynchronize(stream);

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

  flagcxCommWindowDeregister(comm, win);
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
  if (!d->isVMM || d->flatBase == nullptr) {
    flagcxCommWindowDeregister(comm, win);
    GTEST_SKIP() << "VMM not available, cannot test flat VA access";
  }
  if (!hasHeteroComm()) {
    flagcxCommWindowDeregister(comm, win);
    GTEST_SKIP() << "heteroComm not available";
  }

  int localRank = comm->heteroComm->localRank;
  int localRanks = d->localRanks;
  size_t allocSize = d->allocSize;

  // Zero out own region first
  devHandle->deviceMemset(devBuff, 0, size, flagcxMemDevice, stream);
  devHandle->streamSynchronize(stream);

  MPI_Barrier(MPI_COMM_WORLD);

  // Write pattern (localRank + 100) into the NEXT peer's region.
  // Use devBuff2 as staging: host -> local device -> peer VMM region.
  // Direct host-to-peer-VMM may not work with cudaMemcpyHostToDevice.
  int targetLocalRank = (localRank + 1) % localRanks;
  void *targetRegion =
      (char *)d->flatBase + (size_t)targetLocalRank * allocSize;

  float writeValue = (float)(localRank + 100);
  std::vector<float> pattern(count, writeValue);
  devHandle->deviceMemcpy(devBuff2, pattern.data(), size,
                          flagcxMemcpyHostToDevice, stream);
  devHandle->deviceMemcpy(targetRegion, devBuff2, size,
                          flagcxMemcpyDeviceToDevice, stream);
  devHandle->streamSynchronize(stream);

  MPI_Barrier(MPI_COMM_WORLD);

  // Now read back our own region — it should have been written by prev peer
  int writerLocalRank = (localRank + localRanks - 1) % localRanks;
  float expected = (float)(writerLocalRank + 100);

  std::vector<float> readBack(count, 0.0f);
  devHandle->deviceMemcpy(readBack.data(), devBuff, size,
                          flagcxMemcpyDeviceToHost, stream);
  devHandle->streamSynchronize(stream);

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

  flagcxCommWindowDeregister(comm, win);
}

// ---------------------------------------------------------------------------
// Verify multicast base is set when VMM succeeds
// ---------------------------------------------------------------------------
