// Compiled by adaptor for SDK contracts and by core for outer-Comm tests.
// Both suites share the mock owners and native callbacks below.
#include "check.h"
#include "flagcx.h"
#include "flagcx_ccl_adaptor.h"
#include <gtest/gtest.h>

#ifdef FLAGCX_TEST_CCL_LIFECYCLE_CORE
#include "adaptor.h"
#include "comm_lifecycle.h"
#include "global_comm.h"
#endif

namespace CclLifecycleShim {
struct NativeContext {
  int finalizeCalls = 0, destroyCalls = 0, queryCalls = 0, abortCalls = 0;
  int pendingDestroys = 0;
  flagcxResult_t finalizeResult = flagcxSuccess;
  flagcxResult_t destroyResult = flagcxSuccess;
  flagcxResult_t queryResult = flagcxSuccess;
  flagcxResult_t asyncResult = flagcxSuccess;
};
struct Wrapper {
  NativeContext *base;
  int frees = 0;
  int closePhase = 0;
};
typedef Wrapper *flagcxInnerComm_t;
static void free(void *pointer) { ((Wrapper *)pointer)->frees++; }
static flagcxResult_t finalizeNative(NativeContext *native) {
  native->finalizeCalls++;
  return native->finalizeResult;
}
static flagcxResult_t destroyNative(NativeContext *native) {
  native->destroyCalls++;
  if (native->pendingDestroys > 0) {
    native->pendingDestroys--;
    return flagcxInProgress;
  }
  return native->destroyResult;
}
} // namespace CclLifecycleShim

namespace NcclLifecycleShim {
using CclLifecycleShim::flagcxInnerComm_t;
using CclLifecycleShim::free;
using CclLifecycleShim::NativeContext;
using CclLifecycleShim::Wrapper;
static flagcxResult_t ncclCommFinalize(NativeContext *native) {
  return CclLifecycleShim::finalizeNative(native);
}
static flagcxResult_t ncclCommDestroy(NativeContext *native) {
  return CclLifecycleShim::destroyNative(native);
}
#include "../../../flagcx/adaptor/ccl/nccl_comm_lifecycle.h"
} // namespace NcclLifecycleShim

namespace HcclLifecycleShim {
using CclLifecycleShim::flagcxInnerComm_t;
using CclLifecycleShim::free;
#define FLAGCX_COMPAT_FINALIZE hcclAdaptorCommFinalize
#define FLAGCX_COMPAT_DESTROY hcclAdaptorCommDestroy
#define FLAGCX_COMPAT_CLOSE_NATIVE(comm)                                       \
  CclLifecycleShim::destroyNative((comm)->base)
#include "../../../flagcx/adaptor/ccl/compat_comm_lifecycle.h"
} // namespace HcclLifecycleShim

namespace CnclLifecycleShim {
using CclLifecycleShim::flagcxInnerComm_t;
using CclLifecycleShim::free;
#define FLAGCX_COMPAT_FINALIZE cnclAdaptorCommFinalize
#define FLAGCX_COMPAT_DESTROY cnclAdaptorCommDestroy
#define FLAGCX_COMPAT_CLOSE_NATIVE(comm)                                       \
  CclLifecycleShim::destroyNative((comm)->base)
#include "../../../flagcx/adaptor/ccl/compat_comm_lifecycle.h"
} // namespace CnclLifecycleShim

namespace PcclLifecycleShim {
using CclLifecycleShim::flagcxInnerComm_t;
using CclLifecycleShim::free;
#define FLAGCX_COMPAT_FINALIZE pcclAdaptorCommFinalize
#define FLAGCX_COMPAT_DESTROY pcclAdaptorCommDestroy
#define FLAGCX_COMPAT_CLOSE_NATIVE(comm)                                       \
  CclLifecycleShim::destroyNative((comm)->base)
#include "../../../flagcx/adaptor/ccl/compat_comm_lifecycle.h"
} // namespace PcclLifecycleShim

namespace XcclLifecycleShim {
using CclLifecycleShim::flagcxInnerComm_t;
using CclLifecycleShim::free;
#define FLAGCX_COMPAT_FINALIZE xcclAdaptorCommFinalize
#define FLAGCX_COMPAT_DESTROY xcclAdaptorCommDestroy
#define FLAGCX_COMPAT_CLOSE_NATIVE(comm)                                       \
  CclLifecycleShim::destroyNative((comm)->base)
#include "../../../flagcx/adaptor/ccl/compat_comm_lifecycle.h"
} // namespace XcclLifecycleShim

namespace MusaLifecycleShim {
using CclLifecycleShim::flagcxInnerComm_t;
using CclLifecycleShim::free;
#define FLAGCX_COMPAT_FINALIZE mcclAdaptorCommFinalize
#define FLAGCX_COMPAT_DESTROY mcclAdaptorCommDestroy
#define FLAGCX_COMPAT_CLOSE_NATIVE(comm)                                       \
  CclLifecycleShim::destroyNative((comm)->base)
#include "../../../flagcx/adaptor/ccl/compat_comm_lifecycle.h"
} // namespace MusaLifecycleShim

namespace EcclLifecycleShim {
using CclLifecycleShim::flagcxInnerComm_t;
using CclLifecycleShim::free;
#define FLAGCX_COMPAT_FINALIZE ecclAdaptorCommFinalize
#define FLAGCX_COMPAT_DESTROY ecclAdaptorCommDestroy
#define FLAGCX_COMPAT_CLOSE_NATIVE(comm)                                       \
  CclLifecycleShim::destroyNative((comm)->base)
#include "../../../flagcx/adaptor/ccl/compat_comm_lifecycle.h"
} // namespace EcclLifecycleShim

// PCCL uses the same owners and error injection as the other SDK shims.
namespace PpuLifecycleShim {
using CclLifecycleShim::flagcxInnerComm_t;
using CclLifecycleShim::free;
using CclLifecycleShim::NativeContext;
using CclLifecycleShim::Wrapper;
typedef flagcxResult_t pcclResult_t;
static const pcclResult_t pcclSuccess = flagcxSuccess;
struct ppupcclSymbols {
  pcclResult_t (*pcclCommFinalize)(NativeContext *);
  pcclResult_t (*pcclCommGetAsyncError)(NativeContext *, pcclResult_t *);
  pcclResult_t (*pcclCommDestroy)(NativeContext *);
  pcclResult_t (*pcclCommAbort)(NativeContext *);
};
static const ppupcclSymbols api = {
    CclLifecycleShim::finalizeNative,
    [](NativeContext *ctx, pcclResult_t *status) {
      ctx->queryCalls++;
      *status = ctx->asyncResult;
      return ctx->queryResult;
    },
    CclLifecycleShim::destroyNative,
    [](NativeContext *ctx) {
      ctx->abortCalls++;
      return flagcxSuccess;
    }};
static flagcxResult_t ppupcclLoadApi(const ppupcclSymbols **out) {
  *out = &api;
  return flagcxSuccess;
}
static flagcxResult_t ppupcclResultToFlagcx(pcclResult_t result) {
  return result;
}
#include "../../../flagcx/adaptor/ccl/ppu_comm_lifecycle.h"
} // namespace PpuLifecycleShim

#ifdef FLAGCX_TEST_CCL_LIFECYCLE_CORE
namespace {
struct PpuProviderGuard {
  flagcxCCLAdaptor *saved = cclAdaptors[flagcxCCLAdaptorDevice];
  flagcxCCLAdaptor provider = *saved;
  PpuProviderGuard() {
    provider.commFinalize = [](flagcxInnerComm_t inner) {
      return PpuLifecycleShim::ppupcclAdaptorCommFinalize(
          (PpuLifecycleShim::flagcxInnerComm_t)inner);
    };
    provider.commGetAsyncError = [](flagcxInnerComm_t inner,
                                    flagcxResult_t *status) {
      return PpuLifecycleShim::ppupcclAdaptorCommGetAsyncError(
          (PpuLifecycleShim::flagcxInnerComm_t)inner, status);
    };
    cclAdaptors[flagcxCCLAdaptorDevice] = &provider;
  }
  ~PpuProviderGuard() { cclAdaptors[flagcxCCLAdaptorDevice] = saved; }
};
} // namespace

TEST(CommLifecycleTest, PpuAsyncFinalizeWaitsForNativeDestroy) {
  PpuLifecycleShim::NativeContext native;
  native.finalizeResult = flagcxInProgress;
  native.pendingDestroys = 1;
  PpuLifecycleShim::Wrapper wrapper{&native};
  PpuProviderGuard guard;
  flagcxComm comm = {};
  comm.commType = flagcxCommunicatorHomo;
  comm.homoComm = (flagcxInnerComm_t)&wrapper;
  ASSERT_EQ(flagcxCommFinalize(&comm), flagcxSuccess);
  EXPECT_EQ(comm.finalizeState.commonPhase, flagcxFinalizeComplete);
  EXPECT_EQ(wrapper.base, nullptr);
  EXPECT_EQ(wrapper.frees, 0);
  EXPECT_EQ(native.finalizeCalls, 1);
  EXPECT_EQ(native.queryCalls, 1);
  EXPECT_EQ(native.destroyCalls, 2);
  EXPECT_EQ(PpuLifecycleShim::ppupcclAdaptorCommDestroy(&wrapper),
            flagcxSuccess);
  EXPECT_EQ(wrapper.frees, 1);
}

TEST(CommLifecycleTest, PpuFailedCloseResumesThroughOuterAsyncPoll) {
  PpuLifecycleShim::NativeContext native;
  native.finalizeResult = flagcxInProgress;
  native.destroyResult = flagcxRemoteError;
  PpuLifecycleShim::Wrapper wrapper{&native};
  PpuProviderGuard guard;
  flagcxComm comm = {};
  comm.commType = flagcxCommunicatorHomo;
  comm.homoComm = (flagcxInnerComm_t)&wrapper;
  EXPECT_EQ(flagcxCommFinalize(&comm), flagcxRemoteError);
  EXPECT_NE(comm.finalizeState.commonPhase, flagcxFinalizeComplete);
  EXPECT_EQ(wrapper.base, &native);
  native.destroyResult = flagcxSuccess;
  ASSERT_EQ(flagcxCommFinalize(&comm), flagcxSuccess);
  EXPECT_EQ(wrapper.base, nullptr);
  EXPECT_EQ(native.finalizeCalls, 1);
  EXPECT_EQ(native.queryCalls, 1);
  EXPECT_EQ(native.destroyCalls, 2);
  EXPECT_EQ(PpuLifecycleShim::ppupcclAdaptorCommDestroy(&wrapper),
            flagcxSuccess);
}

#else
TEST(CclLifecycleTest, NcclFinalizeRetainsWrapperUntilDestroy) {
  using namespace NcclLifecycleShim;
  NativeContext context = {};
  Wrapper wrapper = {&context, 0};
  EXPECT_EQ(ncclAdaptorCommFinalize(&wrapper), flagcxSuccess);
  EXPECT_EQ(wrapper.frees, 0);
  EXPECT_EQ(wrapper.base, &context);
  EXPECT_EQ(ncclAdaptorCommDestroy(&wrapper), flagcxSuccess);
  EXPECT_EQ(context.finalizeCalls, 1);
  EXPECT_EQ(context.destroyCalls, 1);
  EXPECT_EQ(wrapper.frees, 1);
}

TEST(CclLifecycleTest, NcclFinalizeErrorAndProgressRetainWrapper) {
  using namespace NcclLifecycleShim;
  NativeContext context = {};
  Wrapper wrapper = {&context, 0};
  context.finalizeResult = flagcxInProgress;
  EXPECT_EQ(ncclAdaptorCommFinalize(&wrapper), flagcxInProgress);
  EXPECT_EQ(wrapper.frees, 0);
  EXPECT_EQ(wrapper.base, &context);
  context.finalizeResult = flagcxRemoteError;
  EXPECT_EQ(ncclAdaptorCommFinalize(&wrapper), flagcxRemoteError);
  EXPECT_EQ(wrapper.frees, 0);
  EXPECT_EQ(ncclAdaptorCommDestroy(&wrapper), flagcxSuccess);
  EXPECT_EQ(wrapper.frees, 1);
}

TEST(CclLifecycleTest, NcclDestroyFailureRetainsWrapperForRetry) {
  using namespace NcclLifecycleShim;
  NativeContext context = {};
  Wrapper wrapper = {&context, 0};
  ASSERT_EQ(ncclAdaptorCommFinalize(&wrapper), flagcxSuccess);
  context.destroyResult = flagcxRemoteError;
  EXPECT_EQ(ncclAdaptorCommDestroy(&wrapper), flagcxRemoteError);
  EXPECT_EQ(wrapper.frees, 0);
  context.destroyResult = flagcxSuccess;
  EXPECT_EQ(ncclAdaptorCommDestroy(&wrapper), flagcxSuccess);
  EXPECT_EQ(context.destroyCalls, 2);
  EXPECT_EQ(wrapper.frees, 1);
}

TEST(CclLifecycleTest, UnsupportedNativeFinalizeClosesOnceAndRetainsWrapper) {
  typedef flagcxResult_t (*Operation)(NcclLifecycleShim::flagcxInnerComm_t);
  Operation finalizers[] = {HcclLifecycleShim::hcclAdaptorCommFinalize,
                            CnclLifecycleShim::cnclAdaptorCommFinalize,
                            PcclLifecycleShim::pcclAdaptorCommFinalize,
                            XcclLifecycleShim::xcclAdaptorCommFinalize,
                            MusaLifecycleShim::mcclAdaptorCommFinalize,
                            EcclLifecycleShim::ecclAdaptorCommFinalize};
  Operation destructors[] = {HcclLifecycleShim::hcclAdaptorCommDestroy,
                             CnclLifecycleShim::cnclAdaptorCommDestroy,
                             PcclLifecycleShim::pcclAdaptorCommDestroy,
                             XcclLifecycleShim::xcclAdaptorCommDestroy,
                             MusaLifecycleShim::mcclAdaptorCommDestroy,
                             EcclLifecycleShim::ecclAdaptorCommDestroy};
  for (int i = 0; i < 6; i++) {
    NcclLifecycleShim::NativeContext native = {};
    NcclLifecycleShim::Wrapper wrapper = {&native, 0};
    native.destroyResult = flagcxRemoteError;
    EXPECT_EQ(finalizers[i](&wrapper), flagcxRemoteError);
    EXPECT_EQ(wrapper.base, &native);
    EXPECT_EQ(wrapper.frees, 0);
    native.destroyResult = flagcxSuccess;
    EXPECT_EQ(finalizers[i](&wrapper), flagcxSuccess);
    EXPECT_EQ(wrapper.base, nullptr);
    EXPECT_EQ(wrapper.frees, 0);
    EXPECT_EQ(finalizers[i](&wrapper), flagcxSuccess);
    EXPECT_EQ(destructors[i](&wrapper), flagcxSuccess);
    EXPECT_EQ(native.destroyCalls, 2);
    EXPECT_EQ(wrapper.frees, 1);
  }
}

TEST(CclLifecycleTest, LegacyPluginKeepsV1LayoutAndUsesConservativeScope) {
  flagcxCCLAdaptor_v1 old = {};
  flagcxCCLAdaptor_latest current = {};
  old.name = "legacy";
  flagcxCCLAdaptorUpgradeV1(&old, &current);
  EXPECT_EQ(current.name, old.name);
  EXPECT_EQ(current.devCommDestroy, nullptr);
  EXPECT_EQ(current.finalizeScope, flagcxCclFinalizeWorld);
  EXPECT_EQ(offsetof(flagcxCCLAdaptor_v1, commFinalize),
            offsetof(flagcxCCLAdaptor_latest, commFinalize));
}

namespace {
using namespace PpuLifecycleShim;

TEST(PpuLifecycleTest, FinalizeClosesNativeAndRetainsWrapper) {
  NativeContext native;
  Wrapper wrapper{&native};
  EXPECT_EQ(ppupcclAdaptorCommFinalize(&wrapper), flagcxSuccess);
  EXPECT_EQ(native.finalizeCalls, 1);
  EXPECT_EQ(native.destroyCalls, 1);
  EXPECT_EQ(wrapper.base, nullptr);
  EXPECT_EQ(wrapper.frees, 0);
  EXPECT_EQ(ppupcclAdaptorCommFinalize(&wrapper), flagcxSuccess);
  flagcxResult_t status = flagcxRemoteError;
  EXPECT_EQ(ppupcclAdaptorCommGetAsyncError(&wrapper, &status), flagcxSuccess);
  EXPECT_EQ(status, flagcxSuccess);
  EXPECT_EQ(ppupcclAdaptorCommAbort(&wrapper), flagcxSuccess);
  EXPECT_EQ(ppupcclAdaptorCommDestroy(&wrapper), flagcxSuccess);
  EXPECT_EQ(wrapper.frees, 1);
  EXPECT_EQ(native.finalizeCalls, 1);
  EXPECT_EQ(native.destroyCalls, 1);
  EXPECT_EQ(native.queryCalls + native.abortCalls, 0);
}

TEST(PpuLifecycleTest, AsyncFinalizeOnlyCompletesAfterNativeClose) {
  NativeContext native;
  native.finalizeResult = native.asyncResult = flagcxInProgress;
  Wrapper wrapper{&native};
  EXPECT_EQ(ppupcclAdaptorCommFinalize(&wrapper), flagcxInProgress);
  EXPECT_EQ(ppupcclAdaptorCommFinalize(&wrapper), flagcxInProgress);
  EXPECT_EQ(native.finalizeCalls, 1);
  EXPECT_EQ(native.destroyCalls, 0);
  flagcxResult_t status = flagcxSuccess;
  EXPECT_EQ(ppupcclAdaptorCommGetAsyncError(&wrapper, &status), flagcxSuccess);
  EXPECT_EQ(status, flagcxInProgress);
  native.asyncResult = flagcxSuccess;
  native.destroyResult = flagcxInProgress;
  EXPECT_EQ(ppupcclAdaptorCommGetAsyncError(&wrapper, &status), flagcxSuccess);
  EXPECT_EQ(status, flagcxInProgress);
  EXPECT_EQ(wrapper.base, &native);
  EXPECT_EQ(wrapper.frees, 0);
  int queries = native.queryCalls;
  native.destroyResult = flagcxSuccess;
  EXPECT_EQ(ppupcclAdaptorCommGetAsyncError(&wrapper, &status), flagcxSuccess);
  EXPECT_EQ(status, flagcxSuccess);
  EXPECT_EQ(wrapper.base, nullptr);
  EXPECT_EQ(native.finalizeCalls, 1);
  EXPECT_EQ(native.queryCalls, queries);
  EXPECT_EQ(native.destroyCalls, 2);
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
  EXPECT_EQ(native.finalizeCalls, 1);
  EXPECT_EQ(native.destroyCalls, 2);
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
  EXPECT_EQ(native.destroyCalls, 0);
  native.queryResult = flagcxSuccess;
  native.asyncResult = flagcxRemoteError;
  EXPECT_EQ(ppupcclAdaptorCommFinalize(&wrapper), flagcxRemoteError);
  EXPECT_EQ(native.destroyCalls, 0);
  native.asyncResult = flagcxSuccess;
  EXPECT_EQ(ppupcclAdaptorCommFinalize(&wrapper), flagcxSuccess);
  EXPECT_EQ(native.finalizeCalls, 1);
  EXPECT_EQ(ppupcclAdaptorCommDestroy(&wrapper), flagcxSuccess);
}

TEST(PpuLifecycleTest, InitialFinalizeFailureRetainsOpenNative) {
  NativeContext native;
  native.finalizeResult = flagcxRemoteError;
  Wrapper wrapper{&native};
  EXPECT_EQ(ppupcclAdaptorCommFinalize(&wrapper), flagcxRemoteError);
  EXPECT_EQ(wrapper.closePhase, ppupcclCloseOpen);
  EXPECT_EQ(wrapper.base, &native);
  EXPECT_EQ(native.destroyCalls, 0);
  native.finalizeResult = flagcxSuccess;
  EXPECT_EQ(ppupcclAdaptorCommFinalize(&wrapper), flagcxSuccess);
  EXPECT_EQ(native.finalizeCalls, 2);
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
  EXPECT_EQ(native.finalizeCalls, 0);
  EXPECT_EQ(native.destroyCalls, 3);
  EXPECT_EQ(wrapper.frees, 1);
}

TEST(PpuLifecycleTest, ActiveErrorQueriesAndAbortUseNativeContext) {
  NativeContext native;
  native.asyncResult = flagcxRemoteError;
  Wrapper wrapper{&native};
  flagcxResult_t status = flagcxSuccess;
  EXPECT_EQ(ppupcclAdaptorCommGetAsyncError(&wrapper, &status), flagcxSuccess);
  EXPECT_EQ(status, flagcxRemoteError);
  EXPECT_EQ(native.destroyCalls + native.finalizeCalls, 0);
  EXPECT_EQ(ppupcclAdaptorCommAbort(&wrapper), flagcxSuccess);
  EXPECT_EQ(native.abortCalls, 1);
  EXPECT_EQ(ppupcclAdaptorCommDestroy(&wrapper), flagcxSuccess);
  EXPECT_EQ(ppupcclAdaptorCommFinalize(nullptr), flagcxInvalidArgument);
  EXPECT_EQ(ppupcclAdaptorCommDestroy(nullptr), flagcxInvalidArgument);
  EXPECT_EQ(ppupcclAdaptorCommAbort(nullptr), flagcxInvalidArgument);
  EXPECT_EQ(ppupcclAdaptorCommGetAsyncError(nullptr, &status),
            flagcxInvalidArgument);
  EXPECT_EQ(ppupcclAdaptorCommGetAsyncError(&wrapper, nullptr),
            flagcxInvalidArgument);
}

} // namespace
#endif
