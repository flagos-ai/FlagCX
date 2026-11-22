#include "ppu_comm_lifecycle.hpp"

using namespace PpuLifecycleShim;

TEST(PpuLifecycleTest, FinalizeClosesNativeAndRetainsWrapper) {
  NativeContext native;
  Wrapper wrapper{&native};
  EXPECT_EQ(ppupcclAdaptorCommFinalize(&wrapper), flagcxSuccess);
  EXPECT_EQ(native.finalizes, 1);
  EXPECT_EQ(native.destroys, 1);
  EXPECT_EQ(wrapper.base, nullptr);
  EXPECT_EQ(wrapper.frees, 0);
  EXPECT_EQ(ppupcclAdaptorCommFinalize(&wrapper), flagcxSuccess);
  flagcxResult_t status = flagcxRemoteError;
  EXPECT_EQ(ppupcclAdaptorCommGetAsyncError(&wrapper, &status), flagcxSuccess);
  EXPECT_EQ(status, flagcxSuccess);
  EXPECT_EQ(ppupcclAdaptorCommAbort(&wrapper), flagcxSuccess);
  EXPECT_EQ(ppupcclAdaptorCommDestroy(&wrapper), flagcxSuccess);
  EXPECT_EQ(wrapper.frees, 1);
  EXPECT_EQ(native.finalizes, 1);
  EXPECT_EQ(native.destroys, 1);
  EXPECT_EQ(native.queries + native.aborts, 0);
}

TEST(PpuLifecycleTest, AsyncFinalizeOnlyCompletesAfterNativeClose) {
  NativeContext native;
  native.finalizeResult = native.asyncResult = flagcxInProgress;
  Wrapper wrapper{&native};
  EXPECT_EQ(ppupcclAdaptorCommFinalize(&wrapper), flagcxInProgress);
  EXPECT_EQ(ppupcclAdaptorCommFinalize(&wrapper), flagcxInProgress);
  EXPECT_EQ(native.finalizes, 1);
  EXPECT_EQ(native.destroys, 0);
  flagcxResult_t status = flagcxSuccess;
  EXPECT_EQ(ppupcclAdaptorCommGetAsyncError(&wrapper, &status), flagcxSuccess);
  EXPECT_EQ(status, flagcxInProgress);
  native.asyncResult = flagcxSuccess;
  native.destroyResult = flagcxInProgress;
  EXPECT_EQ(ppupcclAdaptorCommGetAsyncError(&wrapper, &status), flagcxSuccess);
  EXPECT_EQ(status, flagcxInProgress);
  EXPECT_EQ(wrapper.base, &native);
  EXPECT_EQ(wrapper.frees, 0);
  int queries = native.queries;
  native.destroyResult = flagcxSuccess;
  EXPECT_EQ(ppupcclAdaptorCommGetAsyncError(&wrapper, &status), flagcxSuccess);
  EXPECT_EQ(status, flagcxSuccess);
  EXPECT_EQ(wrapper.base, nullptr);
  EXPECT_EQ(native.finalizes, 1);
  EXPECT_EQ(native.queries, queries);
  EXPECT_EQ(native.destroys, 2);
  EXPECT_EQ(ppupcclAdaptorCommDestroy(&wrapper), flagcxSuccess);
  EXPECT_EQ(wrapper.frees, 1);
}

TEST(PpuLifecycleTest, CloseFailureRetainsNativeAndCompletedFinalize) {
  NativeContext native;
  native.destroyResult = flagcxRemoteError;
  Wrapper wrapper{&native};
  EXPECT_EQ(ppupcclAdaptorCommFinalize(&wrapper), flagcxRemoteError);
  EXPECT_EQ(wrapper.base, &native);
  EXPECT_EQ(wrapper.frees, 0);
  native.destroyResult = flagcxSuccess;
  EXPECT_EQ(ppupcclAdaptorCommFinalize(&wrapper), flagcxSuccess);
  EXPECT_EQ(native.finalizes, 1);
  EXPECT_EQ(native.destroys, 2);
  EXPECT_EQ(ppupcclAdaptorCommDestroy(&wrapper), flagcxSuccess);
}

TEST(PpuLifecycleTest, FailedAsyncQueryRetainsAcceptedFinalize) {
  NativeContext native;
  native.finalizeResult = flagcxInProgress;
  native.queryResult = flagcxSystemError;
  Wrapper wrapper{&native};
  ASSERT_EQ(ppupcclAdaptorCommFinalize(&wrapper), flagcxInProgress);
  flagcxResult_t status = flagcxSuccess;
  EXPECT_EQ(ppupcclAdaptorCommGetAsyncError(&wrapper, &status), flagcxSuccess);
  EXPECT_EQ(status, flagcxSystemError);
  EXPECT_EQ(wrapper.base, &native);
  EXPECT_EQ(native.destroys, 0);
  native.queryResult = flagcxSuccess;
  native.asyncResult = flagcxRemoteError;
  EXPECT_EQ(ppupcclAdaptorCommFinalize(&wrapper), flagcxRemoteError);
  EXPECT_EQ(native.destroys, 0);
  native.asyncResult = flagcxSuccess;
  EXPECT_EQ(ppupcclAdaptorCommFinalize(&wrapper), flagcxSuccess);
  EXPECT_EQ(native.finalizes, 1);
  EXPECT_EQ(ppupcclAdaptorCommDestroy(&wrapper), flagcxSuccess);
}

TEST(PpuLifecycleTest, InitialFinalizeFailureRetainsOpenNative) {
  NativeContext native;
  native.finalizeResult = flagcxRemoteError;
  Wrapper wrapper{&native};
  EXPECT_EQ(ppupcclAdaptorCommFinalize(&wrapper), flagcxRemoteError);
  EXPECT_EQ(wrapper.closePhase, ppupcclCloseOpen);
  EXPECT_EQ(wrapper.base, &native);
  EXPECT_EQ(native.destroys, 0);
  native.finalizeResult = flagcxSuccess;
  EXPECT_EQ(ppupcclAdaptorCommFinalize(&wrapper), flagcxSuccess);
  EXPECT_EQ(native.finalizes, 2);
  EXPECT_EQ(ppupcclAdaptorCommDestroy(&wrapper), flagcxSuccess);
}

TEST(PpuLifecycleTest, DirectDestroyRetainsRollbackOwnerOnFailure) {
  NativeContext native;
  Wrapper wrapper{&native};
  for (flagcxResult_t result : {flagcxRemoteError, flagcxInProgress}) {
    native.destroyResult = result;
    EXPECT_EQ(ppupcclAdaptorCommDestroy(&wrapper), result);
    EXPECT_EQ(wrapper.base, &native);
    EXPECT_EQ(wrapper.frees, 0);
  }
  native.destroyResult = flagcxSuccess;
  EXPECT_EQ(ppupcclAdaptorCommDestroy(&wrapper), flagcxSuccess);
  EXPECT_EQ(native.finalizes, 0);
  EXPECT_EQ(native.destroys, 3);
  EXPECT_EQ(wrapper.frees, 1);
}

TEST(PpuLifecycleTest, ActiveErrorQueriesAndAbortUseNativeContext) {
  NativeContext native;
  native.asyncResult = flagcxRemoteError;
  Wrapper wrapper{&native};
  flagcxResult_t status = flagcxSuccess;
  EXPECT_EQ(ppupcclAdaptorCommGetAsyncError(&wrapper, &status), flagcxSuccess);
  EXPECT_EQ(status, flagcxRemoteError);
  EXPECT_EQ(native.destroys + native.finalizes, 0);
  EXPECT_EQ(ppupcclAdaptorCommAbort(&wrapper), flagcxSuccess);
  EXPECT_EQ(native.aborts, 1);
  EXPECT_EQ(ppupcclAdaptorCommDestroy(&wrapper), flagcxSuccess);
  EXPECT_EQ(ppupcclAdaptorCommFinalize(nullptr), flagcxInvalidArgument);
  EXPECT_EQ(ppupcclAdaptorCommDestroy(nullptr), flagcxInvalidArgument);
  EXPECT_EQ(ppupcclAdaptorCommAbort(nullptr), flagcxInvalidArgument);
  EXPECT_EQ(ppupcclAdaptorCommGetAsyncError(nullptr, &status),
            flagcxInvalidArgument);
  EXPECT_EQ(ppupcclAdaptorCommGetAsyncError(&wrapper, nullptr),
            flagcxInvalidArgument);
}
