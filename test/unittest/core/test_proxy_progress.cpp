#include <gtest/gtest.h>

#include "proxy.h"

TEST(ProxyProgressState, SuccessAndInProgressDoNotAbort) {
  uint32_t abort = 0;
  flagcxProxyState state{};
  state.abortFlag = &abort;

  EXPECT_EQ(flagcxProxyRecordAsyncError(&state, flagcxSuccess), flagcxSuccess);
  EXPECT_EQ(flagcxProxyRecordAsyncError(&state, flagcxInProgress),
            flagcxInProgress);
  EXPECT_EQ(state.asyncResult, flagcxSuccess);
  EXPECT_EQ(abort, 0u);
}

TEST(ProxyProgressState, HardErrorRecordsFirstFailureAndAborts) {
  uint32_t abort = 0;
  flagcxProxyState state{};
  state.abortFlag = &abort;

  EXPECT_EQ(flagcxProxyRecordAsyncError(&state, flagcxUnhandledDeviceError),
            flagcxUnhandledDeviceError);

  EXPECT_EQ(state.asyncResult, flagcxUnhandledDeviceError);
  EXPECT_EQ(abort, 1u);

  EXPECT_EQ(flagcxProxyRecordAsyncError(&state, flagcxSystemError),
            flagcxSystemError);
  EXPECT_EQ(state.asyncResult, flagcxUnhandledDeviceError);
}

TEST(ProxyProgressState, NullStateIsSafe) {
  EXPECT_EQ(flagcxProxyRecordAsyncError(nullptr, flagcxSystemError),
            flagcxSystemError);
}
