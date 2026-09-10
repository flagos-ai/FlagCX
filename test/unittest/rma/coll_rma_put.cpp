// MPI correctness tests for flagcxPut / flagcxPutSignal.
// Requires 2 local ranks with a hetero communicator; individual invocations
// select either IPC or RDMA explicitly.

#include "flagcx_hetero.h"
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

// The IPC data path must use the symmetric window's IPC locator directly. It
// must not require a network MR index or an initialized RDMA sendComm.
TEST_F(RmaTest, IpcPutWithoutNetworkMr) {
  if (!requireIpc)
    GTEST_SKIP() << "Runs only in the explicit IPC invocation";
  int localMrAbsent = dataWin->defaultBase->mrIndex < 0 ? 1 : 0;
  int allMrsAbsent = 0;
  MPI_Allreduce(&localMrAbsent, &allMrsAbsent, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  ASSERT_EQ(allMrsAbsent, 1)
      << "IPC invocation unexpectedly registered a network MR";

  const size_t testSize = 64;
  flagcxStream_t s = nullptr;
  flagcxResult_t setupRes = devHandle->streamCreate(&s);
  if (setupRes == flagcxSuccess && s == nullptr)
    setupRes = flagcxInternalError;
  ASSERT_EQ(collectiveOpStatus(setupRes), 0);

  setupRes =
      devHandle->deviceMemset(dataBuff, 0, size, flagcxMemDevice, nullptr);
  ASSERT_EQ(collectiveOpStatus(setupRes), 0);
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxResult_t opRes = flagcxSuccess;
  if (rank == 0) {
    std::vector<uint8_t> pattern(testSize, 0x5A);
    opRes = devHandle->deviceMemcpy(dataBuff, pattern.data(), testSize,
                                    flagcxMemcpyHostToDevice, nullptr);
    if (opRes == flagcxSuccess) {
      uint64_t opSeq = 0;
      opRes = flagcxHeteroPutStream(comm->heteroComm, 1, 0, 0, testSize, -1, -1,
                                    dataWin->defaultBase, dataWin->defaultBase,
                                    s, &opSeq);
    }
    if (opRes == flagcxSuccess)
      opRes = devHandle->streamSynchronize(s);
  }

  int globalOpStatus = collectiveOpStatus(opRes);
  ASSERT_EQ(globalOpStatus, 0) << "IPC PUT failed without network MR state";
  MPI_Barrier(MPI_COMM_WORLD);

  bool localDataValid = true;
  if (rank == 1) {
    std::vector<uint8_t> received(testSize, 0);
    flagcxResult_t copyRes = devHandle->deviceMemcpy(
        received.data(), dataBuff, testSize, flagcxMemcpyDeviceToHost, nullptr);
    localDataValid = copyRes == flagcxSuccess &&
                     received == std::vector<uint8_t>(testSize, 0x5A);
  }

  int localValid = localDataValid ? 1 : 0;
  int allDataValid = 0;
  MPI_Allreduce(&localValid, &allDataValid, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  EXPECT_EQ(allDataValid, 1);
  EXPECT_EQ(devHandle->streamDestroy(s), flagcxSuccess);
}

// ---------------------------------------------------------------------------
// PutSignal: rank 0 writes known pattern to rank 1, rank 1 verifies
// ---------------------------------------------------------------------------
TEST_F(RmaTest, PutSignalSmall) {
  if (nranks < 2)
    GTEST_SKIP() << "Requires at least 2 ranks";
  ASSERT_FALSE(signalRmaSetupFailed) << signalRmaSkipReason;
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
    GTEST_SKIP() << "RMA signal operations are not supported";
  }
  if (globalStatus == 2) {
    devHandle->streamDestroy(s);
    FAIL() << "flagcxPutSignal failed collectively with status "
           << globalStatus;
  }

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
    FAIL() << "Remote-write visibility flush became unavailable after the "
              "suite capability check";
  }
  if (globalWaitStatus == 2) {
    devHandle->streamDestroy(s);
    FAIL() << "flagcxWaitSignal failed collectively with status "
           << globalWaitStatus;
  }

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
  ASSERT_FALSE(signalRmaSetupFailed) << signalRmaSkipReason;
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
    GTEST_SKIP() << "RMA signal operations are not supported";
  }
  if (globalStatus == 2) {
    devHandle->streamDestroy(s);
    FAIL() << "flagcxPutSignal failed collectively with status "
           << globalStatus;
  }

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
    FAIL() << "Remote-write visibility flush became unavailable after the "
              "suite capability check";
  }
  if (globalWaitStatus == 2) {
    devHandle->streamDestroy(s);
    FAIL() << "flagcxWaitSignal failed collectively with status "
           << globalWaitStatus;
  }

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
