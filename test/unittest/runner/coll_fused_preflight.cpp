#include "bootstrap.h"
#include "comm.h"
#include "global_comm.h"
#include "proxy.h"
#include "runner_fixtures.hpp"
#include "work_completion.hpp"

namespace {
class FusedPreflightMpi : public WorkCompletionTest {
protected:
  int rank = 0, nranks = 0;
  volatile uint32_t abortFlag = 0;
  flagcxComm comm = {};
  flagcxHeteroComm hetero = {};
  flagcxProxyState proxy = {};
  bootstrapState *bootstrap = nullptr;
  float input = 0, weight = 0, output = 0;
  enum Failure {
    NativeEventCreate,
    CaptureQuery,
    ActiveCapture,
    PreparePending
  };
  void SetUp() override {
    WorkCompletionTest::SetUp();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    ASSERT_GE(nranks, 2);
    device.deviceMalloc = [](void **, size_t, flagcxMemType_t, flagcxStream_t) {
      ADD_FAILURE() << "fused work submitted before shared preflight succeeded";
      return flagcxSystemError;
    };
    ASSERT_EQ(bootstrapNetInit(), flagcxSuccess);
    flagcxBootstrapHandle handle = {};
    if (rank == 0)
      ASSERT_EQ(bootstrapGetUniqueId(&handle), flagcxSuccess);
    MPI_Bcast(&handle, sizeof(handle), MPI_BYTE, 0, MPI_COMM_WORLD);
    ASSERT_EQ(bootstrapCollInit(&handle, rank, nranks, handle.magic, &abortFlag,
                                &bootstrap),
              flagcxSuccess);
    comm.commType = flagcxCommunicatorHybrid;
    comm.rank = rank;
    comm.nranks = nranks;
    comm.bootstrap = bootstrap;
    comm.heteroComm = &hetero;
    comm.uniRunnerKernelSupported = true;
    hetero.proxyState = &proxy;
    hetero.nRanks = nranks;
  }
  flagcxResult_t invoke(bool reduceScatter, bool validInput = true) {
    const void *source = validInput ? &input : nullptr;
    return reduceScatter
               ? flagcxGemmReduceScatter(source, &weight, &output, 1, 1, 1,
                                         flagcxFloat, flagcxSum, &comm,
                                         (flagcxStream_t)&stream)
               : flagcxAllGatherGemm(source, &weight, &output, 1, 1, 1,
                                     flagcxFloat, &comm,
                                     (flagcxStream_t)&stream);
  }
  void fail(bool reduceScatter, Failure failure, int failedRank) {
    flagcxResult_t expected = flagcxSuccess;
    switch (failure) {
      case NativeEventCreate:
        expected = flagcxSystemError;
        partialCreateFailures = rank == failedRank ? 1 : 0;
        break;
      case CaptureQuery:
        expected = flagcxUnhandledDeviceError;
        captureResult = rank == failedRank ? expected : flagcxSuccess;
        break;
      case ActiveCapture:
        expected = flagcxNotSupported;
        stream.capture = rank == failedRank ? flagcxStreamCaptureActive
                                            : flagcxStreamCaptureNone;
        break;
      case PreparePending:
        expected = flagcxInProgress;
        captureResult = rank == failedRank ? expected : flagcxSuccess;
        break;
    }
    EXPECT_EQ(invoke(reduceScatter), expected);
    captureResult = flagcxSuccess;
    stream.capture = flagcxStreamCaptureNone;
    EXPECT_EQ(recorded + queried + waited, 0);
    EXPECT_EQ(comm.submittedWork.closed, 0);
    for (auto *record = comm.submittedWork.head; record; record = record->next)
      EXPECT_EQ(record->state, flagcxWorkComplete);
  }
  void retry(bool reduceScatter) {
    fail(reduceScatter, CaptureQuery, 1);
    // Invalid input stops the real Impl before workspace allocation on every
    // platform, after a successful retry of the shared completion preflight.
    EXPECT_EQ(invoke(reduceScatter, false), flagcxInvalidArgument);
    EXPECT_EQ(recorded, 1);
    EXPECT_EQ(comm.submittedWork.closed, 0);
  }
  void TearDown() override {
    destroyFailures = 0;
    captureResult = flagcxSuccess;
    stream.capture = flagcxStreamCaptureNone;
    EXPECT_EQ(flagcxWorkWait(&comm.submittedWork), flagcxSuccess);
    EXPECT_EQ(flagcxWorkRelease(&comm.submittedWork), flagcxSuccess);
    if (bootstrap != nullptr)
      EXPECT_EQ(bootstrapClose(bootstrap), flagcxSuccess);
    WorkCompletionTest::TearDown();
  }
};
} // namespace

TEST_F(FusedPreflightMpi, EventCreateFailureConvergesForBothApis) {
  fail(false, NativeEventCreate, 1);
  fail(true, NativeEventCreate, 0);
}
TEST_F(FusedPreflightMpi, CaptureQueryFailureConvergesForBothApis) {
  fail(false, CaptureQuery, 1);
  fail(true, CaptureQuery, 0);
}
TEST_F(FusedPreflightMpi, ActiveCaptureRejectsBothApisCollectively) {
  fail(false, ActiveCapture, 1);
  fail(true, ActiveCapture, 0);
}
TEST_F(FusedPreflightMpi, InProgressConvergesWithoutSubmittingEitherApi) {
  fail(false, PreparePending, 1);
  fail(true, PreparePending, 0);
}
TEST_F(FusedPreflightMpi, CanceledRecordReleaseCanRetryWhilePeersWait) {
  fail(false, CaptureQuery, 1);
  int released = 1;
  if (rank == 0) {
    ASSERT_EQ(flagcxWorkWait(&comm.submittedWork), flagcxSuccess);
    destroyFailures = 1;
    EXPECT_EQ(flagcxWorkRelease(&comm.submittedWork), flagcxSystemError);
    ASSERT_NE(comm.submittedWork.head, nullptr);
    EXPECT_NE(comm.submittedWork.head->event, nullptr);
    EXPECT_EQ(flagcxWorkRelease(&comm.submittedWork), flagcxSuccess);
    for (int peer = 1; peer < nranks; peer++)
      MPI_Send(&released, 1, MPI_INT, peer, 0x6e30, MPI_COMM_WORLD);
  } else {
    MPI_Recv(&released, 1, MPI_INT, 0, 0x6e30, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }
  EXPECT_EQ(recorded + queried + waited, 0);
}
TEST_F(FusedPreflightMpi, AllGatherGemmCanRetryAfterPreflightFailure) {
  retry(false);
}
TEST_F(FusedPreflightMpi, GemmReduceScatterCanRetryAfterPreflightFailure) {
  retry(true);
}
