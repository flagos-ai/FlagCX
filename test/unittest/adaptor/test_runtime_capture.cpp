#include "flagcx_device_adaptor.h"
#include <gtest/gtest.h>
#include <string>
namespace CaptureProbe {
static int calls, state, error;
static void *stream;
} // namespace CaptureProbe
struct CaptureContract {
  const char *name;
  bool available, ppu;
  flagcxResult_t (*query)(bool, flagcxStreamCaptureStatus_t *);
};
#include "adaptor-contracts/capture.h"

TEST(CaptureContract, StatesAreMappedBySdkEnumsAndPreserveDefaultStream) {
  for (const auto &sdk : captureContracts) {
    SCOPED_TRACE(sdk.name);
    CaptureProbe::calls = CaptureProbe::error = 0;
    flagcxStreamCaptureStatus_t output = flagcxStreamCaptureNone;
    EXPECT_EQ(sdk.query(false, nullptr), flagcxInvalidArgument);
    for (bool nonnull : {false, true}) {
      CaptureProbe::stream = nonnull ? (void *)42 : nullptr;
      for (int state = 0; state < 3; ++state) {
        const int native[] = {11, 23, 41};
        CaptureProbe::state = native[state];
        EXPECT_EQ(sdk.query(nonnull, &output),
                  sdk.available ? flagcxSuccess : flagcxNotSupported);
        EXPECT_EQ(output, sdk.available ? (flagcxStreamCaptureStatus_t)state
                                        : flagcxStreamCaptureInvalidated);
      }
    }
    EXPECT_EQ(CaptureProbe::calls, sdk.available ? 6 : 0);
  }
}

TEST(CaptureContract, UnknownEnumsAndErrorsNeverBecomeInactiveCapture) {
  for (const auto &sdk : captureContracts) {
    SCOPED_TRACE(sdk.name);
    CaptureProbe::calls = CaptureProbe::error = 0;
    CaptureProbe::stream = nullptr;
    CaptureProbe::state = 127;
    flagcxStreamCaptureStatus_t output = flagcxStreamCaptureNone;
    EXPECT_EQ(sdk.query(false, &output),
              sdk.available ? flagcxInternalError : flagcxNotSupported);
    EXPECT_EQ(output, flagcxStreamCaptureInvalidated);
    if (sdk.available && (std::string(sdk.name) == "cudaQuery" ||
                          std::string(sdk.name) == "musaQuery" ||
                          std::string(sdk.name) == "hggcQuery" ||
                          std::string(sdk.name) == "cannQuery")) {
      CaptureProbe::error = 81;
      EXPECT_EQ(sdk.query(false, &output), flagcxNotSupported);
      EXPECT_EQ(output, flagcxStreamCaptureInvalidated);
    }
    if (sdk.available && std::string(sdk.name) == "cannQuery") {
      CaptureProbe::error = 82;
      EXPECT_EQ(sdk.query(false, &output), flagcxNotSupported);
      EXPECT_EQ(output, flagcxStreamCaptureInvalidated);
    }
    CaptureProbe::error = 99;
    EXPECT_EQ(sdk.query(false, &output),
              sdk.available ? flagcxUnhandledDeviceError : flagcxNotSupported);
    EXPECT_EQ(output, flagcxStreamCaptureInvalidated);
    if (sdk.ppu && sdk.available) {
      CaptureProbe::error = 70;
      EXPECT_EQ(sdk.query(false, &output), flagcxInProgress);
    }
  }
}

TEST(CaptureContract, V1UpgradeDoesNotInventIntrospectionOrMoveOldFields) {
  static_assert(sizeof(flagcxDeviceAdaptor_v1) == 416, "v1 layout changed");
  static_assert(
      offsetof(flagcxDeviceAdaptor_latest, streamGetCaptureStatus) ==
          offsetof(flagcxDeviceAdaptor_latest, symLsaMulticastUnbind) +
              sizeof(void *),
      "query is not appended");
  flagcxDeviceAdaptor_v1 v1 = {};
  flagcxDeviceAdaptor latest = {};
  v1.streamQuery = [](flagcxStream_t) { return flagcxRemoteError; };
  flagcxDeviceAdaptorUpgradeV1(&v1, &latest);
  EXPECT_EQ(latest.streamQuery, v1.streamQuery);
  EXPECT_EQ(latest.streamGetCaptureStatus, nullptr);
}
