#include "flagcx_device_adaptor.h"

// V1 plugin fixture: all required callbacks exist; stream layout is opaque.
namespace {
template <typename R, typename... Args>
R noop(Args...) {
  return R();
}
template <typename R, typename... Args>
void bind(R (**function)(Args...)) {
  *function = &noop<R, Args...>;
}
flagcxDeviceAdaptor_v1 descriptor() {
  flagcxDeviceAdaptor_v1 adaptor = {};
  strcpy(adaptor.name, "handle-device-test");
  bind(&adaptor.deviceSynchronize);
  bind(&adaptor.deviceMemcpy);
  bind(&adaptor.deviceMemset);
  bind(&adaptor.deviceMalloc);
  bind(&adaptor.deviceFree);
  bind(&adaptor.setDevice);
  bind(&adaptor.getDevice);
  bind(&adaptor.getDeviceCount);
  bind(&adaptor.getVendor);
  bind(&adaptor.gdrMemAlloc);
  bind(&adaptor.gdrMemFree);
  bind(&adaptor.streamCreate);
  bind(&adaptor.streamDestroy);
  bind(&adaptor.streamCopy);
  bind(&adaptor.streamFree);
  bind(&adaptor.streamSynchronize);
  bind(&adaptor.streamQuery);
  bind(&adaptor.streamWaitEvent);
  bind(&adaptor.streamWaitValue64);
  bind(&adaptor.streamWriteValue64);
  bind(&adaptor.eventCreate);
  bind(&adaptor.eventDestroy);
  bind(&adaptor.eventRecord);
  bind(&adaptor.eventSynchronize);
  bind(&adaptor.eventQuery);
  bind(&adaptor.eventElapsedTime);
  bind(&adaptor.ipcMemHandleCreate);
  bind(&adaptor.ipcMemHandleGet);
  bind(&adaptor.ipcMemHandleOpen);
  bind(&adaptor.ipcMemHandleClose);
  bind(&adaptor.ipcMemHandleFree);
  bind(&adaptor.launchHostFunc);
  adaptor.streamQuery = [](flagcxStream_t) { return flagcxRemoteError; };
  return adaptor;
}
} // namespace
extern "C" {
flagcxDeviceAdaptor_v1 flagcxDeviceAdaptorPlugin_v1 = descriptor();
}

#ifdef FLAGCX_TEST_CAPTURE_SIDECAR
extern "C" flagcxResult_t
flagcxDeviceAdaptorStreamCaptureStatus_v1(flagcxStream_t stream,
                                          flagcxStreamCaptureStatus_t *status) {
  if (stream != (flagcxStream_t)(uintptr_t)0x13579 || status == NULL)
    return flagcxInvalidArgument;
  *status = flagcxStreamCaptureActive;
  return flagcxSuccess;
}
#endif
