// MPI correctness tests for flagcxSignal / flagcxWaitSignal.
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
// SignalOnly: rank 0 sends signal without data, rank 1 waits
// ---------------------------------------------------------------------------
TEST_F(RmaTest, SignalOnlyNoData) {
  if (nranks < 2)
    GTEST_SKIP() << "Requires at least 2 ranks";
  ASSERT_TRUE(signalRmaAvailable) << signalRmaSkipReason;

  flagcxStream_t s;
  devHandle->streamCreate(&s);

  devHandle->deviceMemset(signalBuff, 0, signalSize, flagcxMemDevice, nullptr);
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxResult_t opRes = flagcxSuccess;
  if (rank == 0) {
    opRes = flagcxSignal(1, 0, comm, s);
  }

  int globalStatus = collectiveOpStatus(opRes);
  if (globalStatus != 0) {
    devHandle->streamDestroy(s);
    FAIL() << "flagcxSignal failed collectively with status " << globalStatus;
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
  if (globalWaitStatus != 0) {
    devHandle->streamDestroy(s);
    FAIL() << "flagcxWaitSignal failed collectively with status "
           << globalWaitStatus;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  devHandle->streamDestroy(s);
}

// ---------------------------------------------------------------------------
// MultipleSignals: rank 0 sends 4 signals, rank 1 batch-waits for all
// ---------------------------------------------------------------------------
TEST_F(RmaTest, MultipleSignals) {
  if (nranks < 2)
    GTEST_SKIP() << "Requires at least 2 ranks";
  ASSERT_TRUE(signalRmaAvailable) << signalRmaSkipReason;

  flagcxStream_t s;
  devHandle->streamCreate(&s);

  devHandle->deviceMemset(signalBuff, 0, signalSize, flagcxMemDevice, nullptr);
  MPI_Barrier(MPI_COMM_WORLD);

  const int numSignals = 4;

  flagcxResult_t opRes = flagcxSuccess;
  if (rank == 0) {
    for (int i = 0; i < numSignals; ++i) {
      opRes = flagcxSignal(1, 0, comm, s);
      if (opRes != flagcxSuccess) {
        break;
      }
    }
  }

  int globalStatus = collectiveOpStatus(opRes);
  if (globalStatus != 0) {
    devHandle->streamDestroy(s);
    FAIL() << "flagcxSignal failed collectively with status " << globalStatus;
  }

  flagcxResult_t waitRes = flagcxSuccess;
  if (rank == 0) {
    waitRes = devHandle->streamSynchronize(s);
  } else if (rank == 1) {
    // Wait for all 4 signals from rank 0
    flagcxWaitSignalDesc_t desc = {(uint64_t)numSignals, 0};
    waitRes = flagcxWaitSignal(1, &desc, comm, s);
    if (waitRes == flagcxSuccess)
      waitRes = devHandle->streamSynchronize(s);
  }

  int globalWaitStatus = collectiveOpStatus(waitRes);
  if (globalWaitStatus != 0) {
    devHandle->streamDestroy(s);
    FAIL() << "flagcxWaitSignal failed collectively with status "
           << globalWaitStatus;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  devHandle->streamDestroy(s);
}
