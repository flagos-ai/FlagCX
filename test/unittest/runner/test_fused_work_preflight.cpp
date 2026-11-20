#include "global_comm.h"
#include "runner.h"
#include "work_completion.hpp"

namespace {
class FusedWorkPreflightTest : public WorkCompletionTest {
protected:
  flagcxComm comm = {};
  void SetUp() override {
    WorkCompletionTest::SetUp();
    comm.nranks = 1;
    device.deviceMalloc = [](void **, size_t, flagcxMemType_t, flagcxStream_t) {
      ADD_FAILURE() << "fused Impl submitted allocation after failed preflight";
      return flagcxSystemError;
    };
  }
  flagcxResult_t invoke(bool reduceScatter) {
    float storage = 0;
    return reduceScatter
               ? flagcxGemmReduceScatter(&storage, &storage, &storage, 1, 1, 1,
                                         flagcxFloat, flagcxSum, &comm,
                                         (flagcxStream_t)&stream)
               : flagcxAllGatherGemm(&storage, &storage, &storage, 1, 1, 1,
                                     flagcxFloat, &comm,
                                     (flagcxStream_t)&stream);
  }
  void TearDown() override {
    for (auto *record = comm.submittedWork.head; record; record = record->next)
      EXPECT_EQ(record->state, flagcxWorkComplete);
    releaseFake(&comm.submittedWork);
    WorkCompletionTest::TearDown();
  }
};
} // namespace

TEST_F(FusedWorkPreflightTest, EventCreateErrorsReturnBeforeEitherImpl) {
  for (bool reduceScatter : {false, true}) {
    partialCreateFailures = 1;
    EXPECT_EQ(invoke(reduceScatter), flagcxSystemError);
  }
  EXPECT_EQ(recorded + queried + waited, 0);
}

TEST_F(FusedWorkPreflightTest, CaptureErrorsAndProgressReturnBeforeEitherImpl) {
  for (bool reduceScatter : {false, true}) {
    captureResult = flagcxUnhandledDeviceError;
    EXPECT_EQ(invoke(reduceScatter), flagcxUnhandledDeviceError);
    captureResult = flagcxInProgress;
    EXPECT_EQ(invoke(reduceScatter), flagcxInProgress);
    captureResult = flagcxSuccess;
    stream.capture = flagcxStreamCaptureActive;
    EXPECT_EQ(invoke(reduceScatter), flagcxNotSupported);
    stream.capture = flagcxStreamCaptureNone;
  }
  EXPECT_EQ(created + recorded + queried + waited, 0);
}

TEST_F(FusedWorkPreflightTest, AgreementFailureCancelsEveryReadyMarker) {
  comm.nranks = 2; // No bootstrap: the agreement itself must fail locally.
  for (bool reduceScatter : {false, true})
    EXPECT_EQ(invoke(reduceScatter), flagcxInvalidArgument);
  EXPECT_EQ(created, 2);
  EXPECT_EQ(recorded + queried + waited + destroyed, 0);
  EXPECT_EQ(comm.submittedWork.closed, 0);
}
