// MPI correctness tests for flagcxGet (RDMA READ).
// Requires 2 ranks with hetero communicator and RDMA-capable net adaptor.

#include "rma_test.hpp"
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// GetSmall: rank 1 reads 64 bytes from rank 0's buffer
// ---------------------------------------------------------------------------
TEST_F(RmaTest, GetSmall) {
  if (nranks < 2)
    GTEST_SKIP() << "Requires at least 2 ranks";

  const size_t testSize = 64;
  flagcxStream_t s;
  devHandle->streamCreate(&s);

  // Rank 0 fills its buffer with known pattern
  if (rank == 0) {
    std::vector<uint8_t> pattern(testSize, 0xCD);
    devHandle->deviceMemcpy(dataBuff, pattern.data(), testSize,
                            flagcxMemcpyHostToDevice, nullptr);
  } else {
    devHandle->deviceMemset(dataBuff, 0, size, flagcxMemDevice, nullptr);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxResult_t opRes = flagcxSuccess;
  if (rank == 1) {
    uint64_t cntBefore;
    opRes = flagcxReadCounter(comm, &cntBefore);
    if (opRes == flagcxSuccess)
      opRes = flagcxGet(comm, 0, 0, 0, testSize, 0, 0);
    if (opRes == flagcxSuccess)
      opRes = flagcxWaitCounter(comm, cntBefore + 1);
  }

  int localFailed = opRes != flagcxSuccess;
  int anyFailed = 0;
  MPI_Allreduce(&localFailed, &anyFailed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (opRes != flagcxSuccess)
    ADD_FAILURE() << "GetSmall RMA operation failed with result " << opRes;
  if (anyFailed) {
    devHandle->streamDestroy(s);
    return;
  }

  if (rank == 1) {
    // Verify
    std::vector<uint8_t> received(testSize, 0);
    devHandle->deviceMemcpy(received.data(), dataBuff, testSize,
                            flagcxMemcpyDeviceToHost, nullptr);

    int mismatches = 0;
    for (size_t i = 0; i < testSize; ++i) {
      if (received[i] != 0xCD) {
        mismatches++;
        if (mismatches == 1) {
          EXPECT_EQ(received[i], 0xCD) << "Mismatch at byte " << i;
        }
      }
    }
    EXPECT_EQ(mismatches, 0);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  devHandle->streamDestroy(s);
}

// ---------------------------------------------------------------------------
// GetLarge: rank 1 reads 1 MB from rank 0's buffer
// ---------------------------------------------------------------------------
TEST_F(RmaTest, GetLarge) {
  if (nranks < 2)
    GTEST_SKIP() << "Requires at least 2 ranks";

  const size_t testSize = RMA_TEST_SIZE;
  flagcxStream_t s;
  devHandle->streamCreate(&s);

  if (rank == 0) {
    std::vector<uint8_t> pattern(testSize);
    for (size_t i = 0; i < testSize; ++i)
      pattern[i] = static_cast<uint8_t>((i * 7) & 0xFF);
    devHandle->deviceMemcpy(dataBuff, pattern.data(), testSize,
                            flagcxMemcpyHostToDevice, nullptr);
  } else {
    devHandle->deviceMemset(dataBuff, 0, size, flagcxMemDevice, nullptr);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxResult_t opRes = flagcxSuccess;
  if (rank == 1) {
    uint64_t cntBefore;
    opRes = flagcxReadCounter(comm, &cntBefore);
    if (opRes == flagcxSuccess)
      opRes = flagcxGet(comm, 0, 0, 0, testSize, 0, 0);
    if (opRes == flagcxSuccess)
      opRes = flagcxWaitCounter(comm, cntBefore + 1);
  }

  int localFailed = opRes != flagcxSuccess;
  int anyFailed = 0;
  MPI_Allreduce(&localFailed, &anyFailed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (opRes != flagcxSuccess)
    ADD_FAILURE() << "GetLarge RMA operation failed with result " << opRes;
  if (anyFailed) {
    devHandle->streamDestroy(s);
    return;
  }

  if (rank == 1) {
    // Verify
    std::vector<uint8_t> received(testSize, 0);
    devHandle->deviceMemcpy(received.data(), dataBuff, testSize,
                            flagcxMemcpyDeviceToHost, nullptr);

    int mismatches = 0;
    for (size_t i = 0; i < testSize && mismatches < 10; ++i) {
      uint8_t expected = static_cast<uint8_t>((i * 7) & 0xFF);
      if (received[i] != expected) {
        mismatches++;
        if (mismatches == 1) {
          EXPECT_EQ(received[i], expected) << "Mismatch at byte " << i;
        }
      }
    }
    EXPECT_EQ(mismatches, 0);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  devHandle->streamDestroy(s);
}

// ---------------------------------------------------------------------------
// GetBidirectional: both ranks read from each other simultaneously
// ---------------------------------------------------------------------------
TEST_F(RmaTest, GetBidirectional) {
  if (nranks < 2)
    GTEST_SKIP() << "Requires at least 2 ranks";

  const size_t testSize = 4096;
  flagcxStream_t s;
  devHandle->streamCreate(&s);

  // Each rank fills its buffer with rank-specific pattern
  {
    std::vector<uint8_t> pattern(testSize);
    for (size_t i = 0; i < testSize; ++i)
      pattern[i] = static_cast<uint8_t>((rank + 1 + i) & 0xFF);
    devHandle->deviceMemcpy(dataBuff, pattern.data(), testSize,
                            flagcxMemcpyHostToDevice, nullptr);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Both ranks issue Get from peer. Use offset to avoid overwriting source
  // data.
  int peer = (rank == 0) ? 1 : 0;
  size_t dstOffset = testSize; // read into second half of buffer

  uint64_t cntBefore;
  flagcxResult_t opRes = flagcxReadCounter(comm, &cntBefore);
  if (opRes == flagcxSuccess)
    opRes = flagcxGet(comm, peer, 0, dstOffset, testSize, 0, 0);
  if (opRes == flagcxSuccess)
    opRes = flagcxWaitCounter(comm, cntBefore + 1);

  int localFailed = opRes != flagcxSuccess;
  int anyFailed = 0;
  MPI_Allreduce(&localFailed, &anyFailed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (opRes != flagcxSuccess)
    ADD_FAILURE() << "GetBidirectional RMA operation failed with result "
                  << opRes;
  if (anyFailed) {
    devHandle->streamDestroy(s);
    return;
  }

  // Verify peer's data at dstOffset
  std::vector<uint8_t> received(testSize, 0);
  devHandle->deviceMemcpy(received.data(), (char *)dataBuff + dstOffset,
                          testSize, flagcxMemcpyDeviceToHost, nullptr);

  int mismatches = 0;
  for (size_t i = 0; i < testSize && mismatches < 10; ++i) {
    uint8_t expected = static_cast<uint8_t>((peer + 1 + i) & 0xFF);
    if (received[i] != expected) {
      mismatches++;
      if (mismatches == 1) {
        EXPECT_EQ(received[i], expected)
            << "Mismatch at byte " << i << " reading from rank " << peer;
      }
    }
  }
  EXPECT_EQ(mismatches, 0);

  MPI_Barrier(MPI_COMM_WORLD);
  devHandle->streamDestroy(s);
}
