// MPI correctness tests for flagcxPut / flagcxPutSignal.
// Requires 2 ranks with hetero communicator and RDMA-capable net adaptor.

#include "rma_test.hpp"
#include <cstring>
#include <vector>

static int collectiveOpStatus(flagcxResult_t res) {
  int localStatus =
      (res == flagcxSuccess) ? 0 : (res == flagcxNotSupported ? 1 : 2);
  int globalStatus = 0;
  MPI_Allreduce(&localStatus, &globalStatus, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  return globalStatus;
}

// ---------------------------------------------------------------------------
// PutSignal: rank 0 writes known pattern to rank 1, rank 1 verifies
// ---------------------------------------------------------------------------
TEST_F(RmaTest, PutSignalSmall) {
  if (nranks < 2)
    GTEST_SKIP() << "Requires at least 2 ranks";
  if (!signalRmaAvailable)
    GTEST_SKIP() << signalRmaSkipReason;

  const size_t testSize = 64;
  flagcxStream_t s;
  devHandle->streamCreate(&s);

  // Reset data buffer
  devHandle->deviceMemset(dataBuff, 0, size, flagcxMemDevice, nullptr);
  devHandle->deviceMemset(signalBuff, 0, signalSize, flagcxMemDevice, nullptr);
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxResult_t opRes = flagcxSuccess;
  if (rank == 0) {
    // Fill source with 0xAB pattern
    std::vector<uint8_t> pattern(testSize, 0xAB);
    devHandle->deviceMemcpy(dataBuff, pattern.data(), testSize,
                            flagcxMemcpyHostToDevice, nullptr);

    opRes = flagcxPutSignal(dataBuff, testSize, flagcxChar, 1, dataWin, 0, 0,
                            comm, s);
  }

  int globalStatus = collectiveOpStatus(opRes);
  if (globalStatus == 1) {
    devHandle->streamDestroy(s);
    GTEST_SKIP() << "flagcxPutSignal is not supported by this backend";
  }
  ASSERT_EQ(globalStatus, 0);

  flagcxResult_t waitRes = flagcxSuccess;
  if (rank == 0) {
    waitRes = devHandle->streamSynchronize(s);
  } else if (rank == 1) {
    // Wait for signal from rank 0
    flagcxWaitSignalDesc_t desc = {1, 0};
    waitRes = flagcxWaitSignal(1, &desc, comm, s);
    if (waitRes == flagcxSuccess)
      waitRes = devHandle->streamSynchronize(s);
  }

  int globalWaitStatus = collectiveOpStatus(waitRes);
  if (globalWaitStatus == 1) {
    devHandle->streamDestroy(s);
    GTEST_SKIP() << "flagcxWaitSignal is not supported by this backend";
  }
  ASSERT_EQ(globalWaitStatus, 0);

  if (rank == 1) {
    // Verify data
    std::vector<uint8_t> received(testSize, 0);
    devHandle->deviceMemcpy(received.data(), dataBuff, testSize,
                            flagcxMemcpyDeviceToHost, nullptr);

    int mismatches = 0;
    for (size_t i = 0; i < testSize; ++i) {
      if (received[i] != 0xAB) {
        mismatches++;
        if (mismatches == 1) {
          EXPECT_EQ(received[i], 0xAB) << "Mismatch at byte " << i;
        }
      }
    }
    EXPECT_EQ(mismatches, 0);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  devHandle->streamDestroy(s);
}

// ---------------------------------------------------------------------------
// PutSignal large (1 MB)
// ---------------------------------------------------------------------------
TEST_F(RmaTest, PutSignalLarge) {
  if (nranks < 2)
    GTEST_SKIP() << "Requires at least 2 ranks";
  if (!signalRmaAvailable)
    GTEST_SKIP() << signalRmaSkipReason;

  const size_t testSize = RMA_TEST_SIZE;
  flagcxStream_t s;
  devHandle->streamCreate(&s);

  devHandle->deviceMemset(dataBuff, 0, size, flagcxMemDevice, nullptr);
  devHandle->deviceMemset(signalBuff, 0, signalSize, flagcxMemDevice, nullptr);
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxResult_t opRes = flagcxSuccess;
  if (rank == 0) {
    // Fill with ascending byte pattern
    std::vector<uint8_t> pattern(testSize);
    for (size_t i = 0; i < testSize; ++i)
      pattern[i] = static_cast<uint8_t>(i & 0xFF);
    devHandle->deviceMemcpy(dataBuff, pattern.data(), testSize,
                            flagcxMemcpyHostToDevice, nullptr);

    opRes = flagcxPutSignal(dataBuff, testSize, flagcxChar, 1, dataWin, 0, 0,
                            comm, s);
  }

  int globalStatus = collectiveOpStatus(opRes);
  if (globalStatus == 1) {
    devHandle->streamDestroy(s);
    GTEST_SKIP() << "flagcxPutSignal is not supported by this backend";
  }
  ASSERT_EQ(globalStatus, 0);

  flagcxResult_t waitRes = flagcxSuccess;
  if (rank == 0) {
    waitRes = devHandle->streamSynchronize(s);
  } else if (rank == 1) {
    flagcxWaitSignalDesc_t desc = {1, 0};
    waitRes = flagcxWaitSignal(1, &desc, comm, s);
    if (waitRes == flagcxSuccess)
      waitRes = devHandle->streamSynchronize(s);
  }

  int globalWaitStatus = collectiveOpStatus(waitRes);
  if (globalWaitStatus == 1) {
    devHandle->streamDestroy(s);
    GTEST_SKIP() << "flagcxWaitSignal is not supported by this backend";
  }
  ASSERT_EQ(globalWaitStatus, 0);

  if (rank == 1) {
    std::vector<uint8_t> received(testSize, 0);
    devHandle->deviceMemcpy(received.data(), dataBuff, testSize,
                            flagcxMemcpyDeviceToHost, nullptr);

    int mismatches = 0;
    for (size_t i = 0; i < testSize && mismatches < 10; ++i) {
      uint8_t expected = static_cast<uint8_t>(i & 0xFF);
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
