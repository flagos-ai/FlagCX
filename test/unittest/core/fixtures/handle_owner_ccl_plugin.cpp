#include "flagcx_ccl_adaptor.h"

// Load a real v1 descriptor so the test can observe actual CCL refcounting
// through adaptor restoration. No communicator or device operation is needed.
namespace {
template <typename R, typename... Args>
R noop(Args...) {
  return R();
}
template <typename R, typename... Args>
void bind(R (**function)(Args...)) {
  *function = &noop<R, Args...>;
}
flagcxCCLAdaptor_v1 descriptor() {
  flagcxCCLAdaptor_v1 adaptor = {};
  adaptor.name = "handle-owner-test";
#define BIND(field) bind(&adaptor.field)
  BIND(getVersion);
  BIND(getUniqueId);
  BIND(getErrorString);
  BIND(getLastError);
  BIND(getStagedBuffer);
  BIND(commInitRank);
  BIND(commFinalize);
  BIND(commDestroy);
  BIND(commAbort);
  BIND(commResume);
  BIND(commSuspend);
  BIND(commCount);
  BIND(commGetDeviceNumber);
  BIND(commUserRank);
  BIND(commGetAsyncError);
  BIND(memAlloc);
  BIND(memFree);
  BIND(commRegister);
  BIND(commDeregister);
  BIND(commWindowRegister);
  BIND(commWindowDeregister);
  BIND(reduce);
  BIND(gather);
  BIND(scatter);
  BIND(broadcast);
  BIND(allReduce);
  BIND(reduceScatter);
  BIND(allGather);
  BIND(alltoAll);
  BIND(alltoAllv);
  BIND(send);
  BIND(recv);
  BIND(groupStart);
  BIND(groupEnd);
#undef BIND
  return adaptor;
}
} // namespace

extern "C" {
flagcxCCLAdaptor_v1 flagcxCCLAdaptorPlugin_v1 = descriptor();
}
