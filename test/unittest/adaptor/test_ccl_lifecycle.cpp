#include "check.h"
#include "flagcx.h"
#include "flagcx_ccl_adaptor.h"
#include <gtest/gtest.h>

namespace NcclLifecycleShim {
struct NativeContext {
  int finalizeCalls, destroyCalls;
  flagcxResult_t finalizeResult, destroyResult;
};
struct Wrapper {
  NativeContext *base;
  int frees;
};
typedef Wrapper *flagcxInnerComm_t;
static flagcxResult_t ncclCommFinalize(NativeContext *native) {
  native->finalizeCalls++;
  return native->finalizeResult;
}
static flagcxResult_t ncclCommDestroy(NativeContext *native) {
  native->destroyCalls++;
  return native->destroyResult;
}
static void free(void *pointer) { ((Wrapper *)pointer)->frees++; }
#include "../../../flagcx/adaptor/ccl/nccl_comm_lifecycle.h"
} // namespace NcclLifecycleShim

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

namespace HcclLifecycleShim {
using NcclLifecycleShim::flagcxInnerComm_t;
using NcclLifecycleShim::free;
#define FLAGCX_COMPAT_FINALIZE hcclAdaptorCommFinalize
#define FLAGCX_COMPAT_DESTROY hcclAdaptorCommDestroy
#define FLAGCX_COMPAT_CLOSE_NATIVE(comm)                                       \
  NcclLifecycleShim::ncclCommDestroy((comm)->base)
#include "../../../flagcx/adaptor/ccl/compat_comm_lifecycle.h"
} // namespace HcclLifecycleShim

namespace CnclLifecycleShim {
using NcclLifecycleShim::flagcxInnerComm_t;
using NcclLifecycleShim::free;
#define FLAGCX_COMPAT_FINALIZE cnclAdaptorCommFinalize
#define FLAGCX_COMPAT_DESTROY cnclAdaptorCommDestroy
#define FLAGCX_COMPAT_CLOSE_NATIVE(comm)                                       \
  NcclLifecycleShim::ncclCommDestroy((comm)->base)
#include "../../../flagcx/adaptor/ccl/compat_comm_lifecycle.h"
} // namespace CnclLifecycleShim

namespace PcclLifecycleShim {
using NcclLifecycleShim::flagcxInnerComm_t;
using NcclLifecycleShim::free;
#define FLAGCX_COMPAT_FINALIZE pcclAdaptorCommFinalize
#define FLAGCX_COMPAT_DESTROY pcclAdaptorCommDestroy
#define FLAGCX_COMPAT_CLOSE_NATIVE(comm)                                       \
  NcclLifecycleShim::ncclCommDestroy((comm)->base)
#include "../../../flagcx/adaptor/ccl/compat_comm_lifecycle.h"
} // namespace PcclLifecycleShim

namespace XcclLifecycleShim {
using NcclLifecycleShim::flagcxInnerComm_t;
using NcclLifecycleShim::free;
#define FLAGCX_COMPAT_FINALIZE xcclAdaptorCommFinalize
#define FLAGCX_COMPAT_DESTROY xcclAdaptorCommDestroy
#define FLAGCX_COMPAT_CLOSE_NATIVE(comm)                                       \
  NcclLifecycleShim::ncclCommDestroy((comm)->base)
#include "../../../flagcx/adaptor/ccl/compat_comm_lifecycle.h"
} // namespace XcclLifecycleShim

namespace MusaLifecycleShim {
using NcclLifecycleShim::flagcxInnerComm_t;
using NcclLifecycleShim::free;
#define FLAGCX_COMPAT_FINALIZE mcclAdaptorCommFinalize
#define FLAGCX_COMPAT_DESTROY mcclAdaptorCommDestroy
#define FLAGCX_COMPAT_CLOSE_NATIVE(comm)                                       \
  NcclLifecycleShim::ncclCommDestroy((comm)->base)
#include "../../../flagcx/adaptor/ccl/compat_comm_lifecycle.h"
} // namespace MusaLifecycleShim

namespace EcclLifecycleShim {
using NcclLifecycleShim::flagcxInnerComm_t;
using NcclLifecycleShim::free;
#define FLAGCX_COMPAT_FINALIZE ecclAdaptorCommFinalize
#define FLAGCX_COMPAT_DESTROY ecclAdaptorCommDestroy
#define FLAGCX_COMPAT_CLOSE_NATIVE(comm)                                       \
  NcclLifecycleShim::ncclCommDestroy((comm)->base)
#include "../../../flagcx/adaptor/ccl/compat_comm_lifecycle.h"
} // namespace EcclLifecycleShim

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
