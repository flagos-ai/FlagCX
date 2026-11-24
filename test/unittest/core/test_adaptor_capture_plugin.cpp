#include "adaptor.h"
#include "adaptor_plugin_load.h"
#include <cstdlib>
#include <gtest/gtest.h>
#include <string>

TEST(AdaptorCapturePlugin, V1WithAndWithoutSidecarKeepOrdinaryStreamDispatch) {
  const std::string saved = ::testing::GTEST_FLAG(death_test_style);
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  for (bool sidecar : {false, true}) {
    std::string file = std::string(FLAGCX_TEST_CAPTURE_PLUGIN_DIR) +
                       (sidecar ? "/with_query.so" : "/legacy.so");
    ASSERT_EXIT(
        {
          setenv("FLAGCX_DEVICE_ADAPTOR_PLUGIN", file.c_str(), 1);
          if (flagcxDeviceAdaptorPluginLoad() != flagcxSuccess)
            std::_Exit(1);
          if (deviceAdaptor->streamQuery((flagcxStream_t)(uintptr_t)0x13579) !=
              flagcxRemoteError)
            std::_Exit(2);
          if (sidecar) {
            if (deviceAdaptor->streamGetCaptureStatus == NULL)
              std::_Exit(3);
            flagcxStreamCaptureStatus_t status;
            if (deviceAdaptor->streamGetCaptureStatus(
                    (flagcxStream_t)(uintptr_t)0x13579, &status) !=
                    flagcxSuccess ||
                status != flagcxStreamCaptureActive)
              std::_Exit(4);
          } else if (deviceAdaptor->streamGetCaptureStatus != NULL)
            std::_Exit(5);
          if (flagcxDeviceAdaptorPluginUnload() != flagcxSuccess)
            std::_Exit(6);
          std::_Exit(0);
        },
        ::testing::ExitedWithCode(0), "");
  }
  ::testing::GTEST_FLAG(death_test_style) = saved;
}
