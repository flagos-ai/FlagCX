// MPI tests for symmetric memory cross-GPU access.
// Verifies that the flat VA mapping allows direct peer reads/writes.
// Requires MPI + GPUs with P2P support.

#include "adaptor.h"
#include "param.h"
#include "sym_heap.h"
#include "symmem_test.hpp"
#include <cstring>
#include <vector>

namespace {

bool allRanksHaveConfiguredVmm(flagcxComm_t comm) {
  int localReady = comm != nullptr && comm->heteroComm != nullptr &&
                           flagcxParamVmmEnable() && deviceAdaptor != nullptr &&
                           deviceAdaptor->symPhysAlloc != nullptr &&
                           deviceAdaptor->symFlatMap != nullptr
                       ? 1
                       : 0;
  int allReady = 0;
  MPI_Allreduce(&localReady, &allReady, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  return allReady != 0;
}

bool allRanksMappedVmm(flagcxSymWindow_t window) {
  int localMapped =
      window != nullptr && window->isVMM && window->flatBase != nullptr ? 1 : 0;
  int allMapped = 0;
  MPI_Allreduce(&localMapped, &allMapped, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  return allMapped != 0;
}

} // namespace

// ---------------------------------------------------------------------------
// Each rank writes a pattern, then reads from the next peer's region
// via the flat VA and verifies correctness.
// ---------------------------------------------------------------------------

TEST_F(SymMemTest, CrossGpuReadViaPeerPtr) {
  if (!allRanksHaveConfiguredVmm(comm))
    GTEST_SKIP() << "VMM is disabled or unavailable on at least one rank";

  flagcxWindow_t win = nullptr;

  ASSERT_EQ(flagcxCommWindowRegister(comm, devBuff, size, &win,
                                     FLAGCX_WIN_COLL_SYMMETRIC),
            flagcxSuccess);
  ASSERT_NE(win, nullptr);
  ASSERT_NE(win->defaultBase, nullptr);

  flagcxSymWindow_t d = win->defaultBase;
  if (!allRanksMappedVmm(d)) {
    flagcxCommWindowDeregister(comm, win);
    GTEST_SKIP() << "VMM mapping is unavailable on at least one rank";
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
  if (!allRanksHaveConfiguredVmm(comm))
    GTEST_SKIP() << "VMM is disabled or unavailable on at least one rank";

  flagcxWindow_t win = nullptr;

  ASSERT_EQ(flagcxCommWindowRegister(comm, devBuff, size, &win,
                                     FLAGCX_WIN_COLL_SYMMETRIC),
            flagcxSuccess);
  ASSERT_NE(win, nullptr);
  ASSERT_NE(win->defaultBase, nullptr);

  flagcxSymWindow_t d = win->defaultBase;
  if (!allRanksMappedVmm(d)) {
    flagcxCommWindowDeregister(comm, win);
    GTEST_SKIP() << "VMM mapping is unavailable on at least one rank";
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
