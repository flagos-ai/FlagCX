// MPI correctness tests for flagcxPut / flagcxPutSignal.
// Requires 2 ranks with hetero communicator and RDMA-capable net adaptor.

#include "rma_test.hpp"
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Helper: establish connection between rank 0 and rank 1 via dummy send/recv
// (required before one-sided ops can be issued)
// ---------------------------------------------------------------------------
static void establishConnection(flagcxComm_t comm,
                                flagcxDeviceHandle_t devHandle, int rank,
                                int nranks) {
  flagcxStream_t s;
  devHandle->streamCreate(&s);
  void *dummy = nullptr;
  devHandle->deviceMalloc(&dummy, 1, flagcxMemDevice, nullptr);

  // All-to-all dummy exchange to establish connections
  flagcxGroupStart(comm);
  for (int peer = 0; peer < nranks; ++peer) {
    if (peer == rank)
      continue;
    flagcxSend(dummy, 1, flagcxChar, peer, comm, s);
    flagcxRecv(dummy, 1, flagcxChar, peer, comm, s);
  }
  flagcxGroupEnd(comm);

  devHandle->streamSynchronize(s);
  devHandle->deviceFree(dummy, flagcxMemDevice, nullptr);
  devHandle->streamDestroy(s);
  MPI_Barrier(MPI_COMM_WORLD);
}

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

  establishConnection(comm, devHandle, rank, nranks);

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

  if (rank == 0) {
    devHandle->streamSynchronize(s);
  } else if (rank == 1) {
    // Wait for signal from rank 0
    flagcxWaitSignalDesc_t desc = {1, 0};
    flagcxResult_t res = flagcxWaitSignal(1, &desc, comm, s);
    ASSERT_EQ(res, flagcxSuccess);
    devHandle->streamSynchronize(s);

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

  establishConnection(comm, devHandle, rank, nranks);

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

  if (rank == 0) {
    devHandle->streamSynchronize(s);
  } else if (rank == 1) {
    flagcxWaitSignalDesc_t desc = {1, 0};
    flagcxResult_t res = flagcxWaitSignal(1, &desc, comm, s);
    ASSERT_EQ(res, flagcxSuccess);
    devHandle->streamSynchronize(s);

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

// ---------------------------------------------------------------------------
// Separate PUT + signal: the signal is a release boundary for all earlier
// writes to the same peer, even when the IB adaptor stripes operations across
// multiple QPs.
// ---------------------------------------------------------------------------
TEST_F(RmaTest, PutThenSignalOrdersPayload) {
  if (nranks < 2)
    GTEST_SKIP() << "Requires at least 2 ranks";

  establishConnection(comm, devHandle, rank, nranks);

  constexpr int iterations = 8;
  const size_t testSize = RMA_TEST_SIZE;
  flagcxStream_t waitStream;
  devHandle->streamCreate(&waitStream);

  devHandle->deviceMemset(signalBuff, 0, signalSize, flagcxMemDevice, nullptr);
  MPI_Barrier(MPI_COMM_WORLD);

  for (int iteration = 0; iteration < iterations; ++iteration) {
    const uint8_t expected = static_cast<uint8_t>(0x31 + iteration);
    if (rank == 0) {
      std::vector<uint8_t> pattern(testSize, expected);
      devHandle->deviceMemcpy(dataBuff, pattern.data(), testSize,
                              flagcxMemcpyHostToDevice, nullptr);
    } else if (rank == 1) {
      devHandle->deviceMemset(dataBuff, 0, testSize, flagcxMemDevice, nullptr);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    uint64_t counterBefore = 0;
    flagcxResult_t opRes = flagcxSuccess;
    if (rank == 0) {
      opRes = flagcxReadCounter(comm, &counterBefore);
      if (opRes == flagcxSuccess)
        opRes = flagcxPut(comm, 1, 0, 0, testSize, 0, 0);
      // A null stream keeps this signal on the inter-rank RMA proxy path.
      if (opRes == flagcxSuccess)
        opRes = flagcxSignal(1, 0, comm, nullptr);
    }

    int globalStatus = collectiveOpStatus(opRes);
    if (globalStatus == 1) {
      devHandle->streamDestroy(waitStream);
      GTEST_SKIP() << "Separate PUT + signal is not supported by this backend";
    }
    ASSERT_EQ(globalStatus, 0);

    if (rank == 1) {
      flagcxWaitSignalDesc_t desc = {static_cast<uint64_t>(iteration + 1), 0};
      ASSERT_EQ(flagcxWaitSignal(1, &desc, comm, waitStream), flagcxSuccess);
      ASSERT_EQ(devHandle->streamSynchronize(waitStream), flagcxSuccess);

      std::vector<uint8_t> received(testSize, 0);
      devHandle->deviceMemcpy(received.data(), dataBuff, testSize,
                              flagcxMemcpyDeviceToHost, nullptr);
      EXPECT_EQ(received, std::vector<uint8_t>(testSize, expected));
    } else if (rank == 0) {
      ASSERT_EQ(flagcxWaitCounter(comm, counterBefore + 2), flagcxSuccess);
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }

  devHandle->streamDestroy(waitStream);
}

// ---------------------------------------------------------------------------
// Batch PUT + signal: all WRs in the batch must be remotely visible before
// the signal ending the epoch can be observed.
// ---------------------------------------------------------------------------
TEST_F(RmaTest, BatchPutThenSignalOrdersPayload) {
  if (nranks < 2)
    GTEST_SKIP() << "Requires at least 2 ranks";

  establishConnection(comm, devHandle, rank, nranks);

  constexpr size_t batchCount = 4;
  const size_t chunkSize = RMA_TEST_SIZE / batchCount;
  size_t srcOffsets[batchCount];
  size_t dstOffsets[batchCount];
  size_t sizes[batchCount];
  int srcMrIdxs[batchCount];
  int dstMrIdxs[batchCount];
  for (size_t i = 0; i < batchCount; ++i) {
    srcOffsets[i] = i * chunkSize;
    dstOffsets[i] = i * chunkSize;
    sizes[i] = chunkSize;
    srcMrIdxs[i] = 0;
    dstMrIdxs[i] = 0;
  }

  flagcxStream_t waitStream;
  devHandle->streamCreate(&waitStream);
  devHandle->deviceMemset(signalBuff, 0, signalSize, flagcxMemDevice, nullptr);

  std::vector<uint8_t> expected(RMA_TEST_SIZE);
  for (size_t i = 0; i < expected.size(); ++i)
    expected[i] = static_cast<uint8_t>((i * 13 + 7) & 0xff);
  if (rank == 0) {
    devHandle->deviceMemcpy(dataBuff, expected.data(), expected.size(),
                            flagcxMemcpyHostToDevice, nullptr);
  } else if (rank == 1) {
    devHandle->deviceMemset(dataBuff, 0, expected.size(), flagcxMemDevice,
                            nullptr);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  uint64_t counterBefore = 0;
  flagcxResult_t opRes = flagcxSuccess;
  if (rank == 0) {
    opRes = flagcxReadCounter(comm, &counterBefore);
    if (opRes == flagcxSuccess) {
      opRes = flagcxBatchPut(comm, 1, srcOffsets, dstOffsets, sizes, srcMrIdxs,
                             dstMrIdxs, batchCount);
    }
    if (opRes == flagcxSuccess)
      opRes = flagcxSignal(1, 0, comm, nullptr);
  }

  int globalStatus = collectiveOpStatus(opRes);
  if (globalStatus == 1) {
    devHandle->streamDestroy(waitStream);
    GTEST_SKIP() << "Batch PUT + signal is not supported by this backend";
  }
  ASSERT_EQ(globalStatus, 0);

  if (rank == 1) {
    flagcxWaitSignalDesc_t desc = {1, 0};
    ASSERT_EQ(flagcxWaitSignal(1, &desc, comm, waitStream), flagcxSuccess);
    ASSERT_EQ(devHandle->streamSynchronize(waitStream), flagcxSuccess);

    std::vector<uint8_t> received(expected.size(), 0);
    devHandle->deviceMemcpy(received.data(), dataBuff, received.size(),
                            flagcxMemcpyDeviceToHost, nullptr);
    EXPECT_EQ(received, expected);
  } else if (rank == 0) {
    ASSERT_EQ(flagcxWaitCounter(comm, counterBefore + batchCount + 1),
              flagcxSuccess);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  devHandle->streamDestroy(waitStream);
}
