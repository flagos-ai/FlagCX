#include "flagcx.h"
#include "flagcx_device_adaptor.h"
#include <gtest/gtest.h>

#ifdef USE_NVIDIA_ADAPTOR
#include "nvidia_adaptor.h"
#include <dlfcn.h>

namespace {
// SDK error injection is thread-local; other adaptor tests and native workers
// continue to use the real runtime functions.
thread_local bool injectSdk = false;
thread_local cudaError_t createResult = cudaSuccess;
thread_local cudaError_t captureResult = cudaSuccess;
thread_local cudaStreamCaptureStatus captureState = cudaStreamCaptureStatusNone;
thread_local int creates = 0, destroys = 0;
cudaEvent_t fakeEvent() { return (cudaEvent_t)0x1234; }
struct SdkInjection {
  SdkInjection() {
    creates = destroys = 0;
    createResult = captureResult = cudaSuccess;
    captureState = cudaStreamCaptureStatusNone;
    injectSdk = true;
  }
  ~SdkInjection() { injectSdk = false; }
};
} // namespace

extern "C" cudaError_t CUDARTAPI cudaEventCreateWithFlags(cudaEvent_t *event,
                                                          unsigned flags) {
  if (injectSdk) {
    creates++;
    if (createResult == cudaSuccess)
      *event = fakeEvent();
    return createResult;
  }
  typedef cudaError_t (*Real)(cudaEvent_t *, unsigned);
  static Real real = (Real)dlsym(RTLD_NEXT, "cudaEventCreateWithFlags");
  return real ? real(event, flags) : cudaErrorUnknown;
}
extern "C" cudaError_t CUDARTAPI cudaEventDestroy(cudaEvent_t event) {
  if (injectSdk) {
    destroys++;
    EXPECT_EQ(event, fakeEvent())
        << "destroy called for an invalid native handle";
    return event == fakeEvent() ? cudaSuccess : cudaErrorInvalidResourceHandle;
  }
  typedef cudaError_t (*Real)(cudaEvent_t);
  static Real real = (Real)dlsym(RTLD_NEXT, "cudaEventDestroy");
  return real ? real(event) : cudaErrorUnknown;
}
extern "C" cudaError_t CUDARTAPI
cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus *status) {
  if (injectSdk) {
    *status = captureState;
    return captureResult;
  }
  typedef cudaError_t (*Real)(cudaStream_t, cudaStreamCaptureStatus *);
  static Real real = (Real)dlsym(RTLD_NEXT, "cudaStreamIsCapturing");
  return real ? real(stream, status) : cudaErrorUnknown;
}

extern flagcxResult_t cudaAdaptorEventCreate(flagcxEvent_t *,
                                             flagcxEventType_t);
extern flagcxResult_t cudaAdaptorEventDestroy(flagcxEvent_t);
extern flagcxResult_t
cudaAdaptorStreamGetCaptureStatus(flagcxStream_t,
                                  flagcxStreamCaptureStatus_t *);
#endif

TEST(CudaCompletionContract,
     NativeCreateFailurePublishesNullAndRetryCanDestroy) {
#ifndef USE_NVIDIA_ADAPTOR
  GTEST_SKIP() << "Requires CUDA headers and the built-in adaptor";
#else
  SdkInjection injection;
  flagcxEvent_t owner = (flagcxEvent_t)1;
  createResult = cudaErrorMemoryAllocation;
  EXPECT_EQ(cudaAdaptorEventCreate(&owner, flagcxEventDisableTiming),
            flagcxUnhandledDeviceError);
  EXPECT_EQ(owner, nullptr);
  EXPECT_EQ(destroys, 0);
  createResult = cudaSuccess;
  ASSERT_EQ(cudaAdaptorEventCreate(&owner, flagcxEventDisableTiming),
            flagcxSuccess);
  ASSERT_NE(owner, nullptr);
  EXPECT_EQ(cudaAdaptorEventDestroy(owner), flagcxSuccess);
  EXPECT_EQ(creates, 2);
  EXPECT_EQ(destroys, 1);
#endif
}

TEST(CudaCompletionContract,
     ActualAdaptorPreservesCaptureStatesAndQueryErrors) {
#ifndef USE_NVIDIA_ADAPTOR
  GTEST_SKIP() << "Requires CUDA headers and the built-in adaptor";
#else
  SdkInjection injection;
  flagcxStreamCaptureStatus_t output = flagcxStreamCaptureInvalidated;
  for (int state = 0; state <= 2; state++) {
    captureState = (cudaStreamCaptureStatus)state;
    EXPECT_EQ(cudaAdaptorStreamGetCaptureStatus(nullptr, &output),
              flagcxSuccess);
    EXPECT_EQ((int)output, state);
  }
  captureResult = cudaErrorStreamCaptureImplicit;
  EXPECT_EQ(cudaAdaptorStreamGetCaptureStatus(nullptr, &output),
            flagcxUnhandledDeviceError);
  EXPECT_EQ(output, flagcxStreamCaptureInvalidated);
#endif
}
