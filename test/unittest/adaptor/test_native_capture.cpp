#include "flagcx_device_adaptor.h"
#include <gtest/gtest.h>

extern "C" struct flagcxDeviceAdaptor *deviceAdaptor;

#if defined(USE_NVIDIA_ADAPTOR)
#include <cuda_runtime_api.h>
#define NATIVE_API(name) cuda##name
#define NATIVE_SUCCESS cudaSuccess
#define NATIVE_CAPTURE_MODE cudaStreamCaptureModeThreadLocal
#elif defined(USE_MUSA_ADAPTOR)
#include <musa_runtime_api.h>
#define NATIVE_API(name) musa##name
#define NATIVE_SUCCESS musaSuccess
#define NATIVE_CAPTURE_MODE musaStreamCaptureModeThreadLocal
#endif

#if defined(USE_NVIDIA_ADAPTOR) || defined(USE_MUSA_ADAPTOR)
// Identical to the built-in adaptor's single native-handle wrapper.
struct flagcxStream {
  NATIVE_API(Stream_t) base;
};
namespace {
struct CaptureOwners {
  flagcxStream stream{};
  NATIVE_API(Graph_t) graph = nullptr;
  bool capturing = false;
  ~CaptureOwners() {
    if (capturing)
      NATIVE_API(StreamEndCapture)(stream.base, &graph);
    if (graph)
      EXPECT_EQ(NATIVE_API(GraphDestroy)(graph), NATIVE_SUCCESS);
    if (stream.base)
      EXPECT_EQ(NATIVE_API(StreamDestroy)(stream.base), NATIVE_SUCCESS);
  }
};
} // namespace
#endif

TEST(NativeCapture, ActualAdaptorObservesBeginAndEndWithoutCommLifecycle) {
#if !defined(USE_NVIDIA_ADAPTOR) && !defined(USE_MUSA_ADAPTOR)
  GTEST_SKIP() << "Direct native Graph test currently covers CUDA and MUSA";
#else
  int count = 0;
  ASSERT_EQ(NATIVE_API(GetDeviceCount)(&count), NATIVE_SUCCESS);
  ASSERT_GT(count, 0);
  ASSERT_EQ(NATIVE_API(SetDevice)(0), NATIVE_SUCCESS);
  ASSERT_NE(deviceAdaptor->streamGetCaptureStatus, nullptr);
  CaptureOwners owner;
  ASSERT_EQ(NATIVE_API(StreamCreate)(&owner.stream.base), NATIVE_SUCCESS);
  flagcxStreamCaptureStatus_t status = flagcxStreamCaptureInvalidated;
  flagcxResult_t query =
      deviceAdaptor->streamGetCaptureStatus(&owner.stream, &status);
  if (query == flagcxNotSupported) {
    RecordProperty("capture", "query_unavailable");
    return;
  }
  ASSERT_EQ(query, flagcxSuccess);
  EXPECT_EQ(status, flagcxStreamCaptureNone);
  auto begun =
      NATIVE_API(StreamBeginCapture)(owner.stream.base, NATIVE_CAPTURE_MODE);
  if (begun == NATIVE_API(ErrorNotSupported)) {
    RecordProperty("capture", "runtime_not_supported");
    return;
  }
  ASSERT_EQ(begun, NATIVE_SUCCESS);
  owner.capturing = true;
  ASSERT_EQ(deviceAdaptor->streamGetCaptureStatus(&owner.stream, &status),
            flagcxSuccess);
  EXPECT_EQ(status, flagcxStreamCaptureActive);
  ASSERT_EQ(NATIVE_API(StreamEndCapture)(owner.stream.base, &owner.graph),
            NATIVE_SUCCESS);
  owner.capturing = false;
  ASSERT_EQ(deviceAdaptor->streamGetCaptureStatus(&owner.stream, &status),
            flagcxSuccess);
  EXPECT_EQ(status, flagcxStreamCaptureNone);
#endif
}
