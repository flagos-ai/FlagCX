#include "runtime_capture.h"
#include <gtest/gtest.h>

namespace {
int captureState, captureError, calls;
void *nativeStream;
int queryCapture(void *stream, int *status) {
  EXPECT_EQ(stream, nativeStream);
  calls++;
  *status = captureState;
  return captureError;
}
} // namespace

TEST(RuntimeCapture, MissingCaptureEntrypointsMeanNoRuntimeCapture) {
  flagcxStreamCaptureStatus_t status = flagcxStreamCaptureInvalidated;
  EXPECT_EQ(flagcxRuntimeQueryCapture(nullptr, nullptr, nullptr, &status),
            flagcxSuccess);
  EXPECT_EQ(status, flagcxStreamCaptureNone);
  EXPECT_EQ(flagcxRuntimeQueryCapture(nullptr, (void *)1, nullptr, &status),
            flagcxNotSupported);
  EXPECT_EQ(status, flagcxStreamCaptureInvalidated);
}

TEST(RuntimeCapture, QueryPreservesEveryDefinedStateAndNativeStream) {
  captureError = calls = 0;
  nativeStream = (void *)0x1234;
  for (captureState = 0; captureState <= 2; captureState++) {
    flagcxStreamCaptureStatus_t status;
    EXPECT_EQ(flagcxRuntimeQueryCapture(queryCapture, (void *)1, nativeStream,
                                        &status),
              flagcxSuccess);
    EXPECT_EQ((int)status, captureState);
  }
  EXPECT_EQ(calls, 3);
}

TEST(RuntimeCapture, QueryErrorsAreNeverTreatedAsAnInactiveCapture) {
  flagcxStreamCaptureStatus_t status = flagcxStreamCaptureNone;
  captureState = 0;
  captureError = 1;
  EXPECT_EQ(
      flagcxRuntimeQueryCapture(queryCapture, (void *)1, nativeStream, &status),
      flagcxUnhandledDeviceError);
  EXPECT_EQ(status, flagcxStreamCaptureInvalidated);
  captureError = 0;
  captureState = 42;
  EXPECT_EQ(
      flagcxRuntimeQueryCapture(queryCapture, (void *)1, nativeStream, &status),
      flagcxInternalError);
  EXPECT_EQ(status, flagcxStreamCaptureInvalidated);
}

TEST(RuntimeCapture, V1UpgradePreservesPrefixWithoutInventingCaptureSupport) {
  static_assert(sizeof(flagcxDeviceAdaptor_v1) == 416, "v1 ABI changed");
  flagcxDeviceAdaptor_v1 old = {};
  old.streamQuery = [](flagcxStream_t) { return flagcxRemoteError; };
  flagcxDeviceAdaptor latest = {};
  flagcxDeviceAdaptorUpgradeV1(&old, &latest);
  EXPECT_EQ(latest.streamQuery, old.streamQuery);
  EXPECT_EQ(latest.streamGetCaptureStatus, nullptr);
}
