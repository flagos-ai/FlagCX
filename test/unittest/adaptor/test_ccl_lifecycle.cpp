#include "flagcx.h"
#include <cstdlib>
#include <gtest/gtest.h>
#include <set>
#define FLAGCXCHECK(call)                                                      \
  do {                                                                         \
    flagcxResult_t r = call;                                                   \
    if (r != flagcxSuccess && r != flagcxInProgress)                           \
      return r;                                                                \
  } while (0)
namespace NativeOwnerProbe {
static std::set<void *> owners;
static int finalizes, destroys, aborts, frees;
static int result;
static void reset() {
  ASSERT_TRUE(owners.empty());
  finalizes = destroys = aborts = frees = 0;
  result = 0;
}
static void release(void *p) {
  EXPECT_EQ(owners.erase(p), 1u);
  ++frees;
  std::free(p);
}
} // namespace NativeOwnerProbe
struct NativeOwnerContract {
  const char *name;
  bool nativeFinalize;
  void *(*make)();
  flagcxResult_t (*finalize)(void *);
  flagcxResult_t (*destroy)(void *);
  flagcxResult_t (*abort)(void *);
  bool (*hasNative)(void *);
};
#include "adaptor-contracts/native_owners.h"
#undef FLAGCXCHECK

TEST(CclOwnerContract, FinalizeRetainsWrapperAndNeverCallsNativeDestroy) {
  for (const auto &sdk : nativeOwners) {
    SCOPED_TRACE(sdk.name);
    NativeOwnerProbe::reset();
    void *p = sdk.make();
    flagcxResult_t result = sdk.finalize(p);
    if (sdk.nativeFinalize) {
      EXPECT_EQ(result, flagcxSuccess);
    } else {
      EXPECT_EQ(result, flagcxNotSupported);
    }
    EXPECT_EQ(NativeOwnerProbe::owners.count(p), 1u);
    EXPECT_EQ(NativeOwnerProbe::finalizes, sdk.nativeFinalize ? 1 : 0);
    EXPECT_EQ(NativeOwnerProbe::destroys + NativeOwnerProbe::frees, 0);
    EXPECT_EQ(sdk.destroy(p), flagcxSuccess);
    EXPECT_EQ(NativeOwnerProbe::frees, 1);
  }
}

TEST(CclOwnerContract, PendingThenErrorThenSuccessRetainsOwnerUntilRelease) {
  for (const auto &sdk : nativeOwners) {
    SCOPED_TRACE(sdk.name);
    NativeOwnerProbe::reset();
    void *p = sdk.make();
    for (int result : {flagcxInProgress, flagcxRemoteError}) {
      NativeOwnerProbe::result = result;
      EXPECT_EQ(sdk.destroy(p), result);
      EXPECT_EQ(NativeOwnerProbe::owners.count(p), 1u);
      EXPECT_EQ(NativeOwnerProbe::frees, 0);
    }
    NativeOwnerProbe::result = 0;
    EXPECT_EQ(sdk.destroy(p), flagcxSuccess);
    EXPECT_EQ(NativeOwnerProbe::frees, 1);
    EXPECT_EQ(NativeOwnerProbe::destroys, 3);
  }
}

TEST(CclOwnerContract, NullDestroySkipsNativeSdk) {
  for (const auto &sdk : nativeOwners) {
    SCOPED_TRACE(sdk.name);
    NativeOwnerProbe::reset();
    EXPECT_EQ(sdk.destroy(nullptr), flagcxInvalidArgument);
    EXPECT_EQ(NativeOwnerProbe::destroys + NativeOwnerProbe::frees, 0);
  }
}

TEST(CclOwnerContract, HcclFailedInitWrapperDoesNotDestroyANullNativeContext) {
  NativeOwnerProbe::reset();
  auto *owner = (hcclOwnerContract::flagcxInnerComm_t)hcclOwnerContract::make();
  owner->base = nullptr;
  EXPECT_EQ(hcclOwnerContract::hcclAdaptorCommDestroy(owner), flagcxSuccess);
  EXPECT_EQ(NativeOwnerProbe::destroys, 0);
  EXPECT_EQ(NativeOwnerProbe::frees, 1);
  EXPECT_TRUE(NativeOwnerProbe::owners.empty());
}

TEST(CclOwnerContract, AbortRetainsWrapperAndDestroyDoesNotCloseNativeTwice) {
  for (const auto &sdk : nativeOwners) {
    SCOPED_TRACE(sdk.name);
    NativeOwnerProbe::reset();
    void *owner = sdk.make();
    if (sdk.abort(owner) == flagcxNotSupported) {
      EXPECT_TRUE(sdk.hasNative(owner));
      EXPECT_EQ(sdk.destroy(owner), flagcxSuccess);
      continue;
    }
    EXPECT_FALSE(sdk.hasNative(owner));
    EXPECT_EQ(NativeOwnerProbe::owners.count(owner), 1u);
    EXPECT_EQ(NativeOwnerProbe::aborts, 1);
    EXPECT_EQ(sdk.abort(owner), flagcxSuccess);
    EXPECT_EQ(NativeOwnerProbe::aborts, 1);
    EXPECT_EQ(sdk.destroy(owner), flagcxSuccess);
    EXPECT_EQ(NativeOwnerProbe::destroys, 0);
    EXPECT_EQ(NativeOwnerProbe::frees, 1);
  }
}

TEST(CclOwnerContract, AbortFailureOrInProgressRetainsNativeOwner) {
  for (const auto &sdk : nativeOwners) {
    SCOPED_TRACE(sdk.name);
    NativeOwnerProbe::reset();
    void *owner = sdk.make();
    for (int result : {flagcxInProgress, flagcxRemoteError}) {
      NativeOwnerProbe::result = result;
      flagcxResult_t aborted = sdk.abort(owner);
      EXPECT_TRUE(aborted == flagcxNotSupported || aborted == result);
      EXPECT_TRUE(sdk.hasNative(owner));
      EXPECT_EQ(NativeOwnerProbe::owners.count(owner), 1u);
      EXPECT_EQ(NativeOwnerProbe::frees, 0);
    }
    NativeOwnerProbe::result = 0;
    EXPECT_EQ(sdk.destroy(owner), flagcxSuccess);
  }
}

TEST(CclOwnerContract, MetaxRegistrationCleanupRetriesWithLiveNativeContext) {
  using namespace mcclOwnerContract;
  NativeOwnerProbe::reset();
  deregistrations = windowDeregistrations = 0;
  auto *comm = (flagcxInnerComm_t)make();
  void *handle = nullptr;
  mcclOwnerContract::flagcxInnerWindow_t win = nullptr;
  ASSERT_EQ(mcclAdaptorCommRegister(comm, (void *)45, 16, &handle),
            flagcxSuccess);
  ASSERT_EQ(mcclAdaptorCommWindowRegister(comm, (void *)45, 16, &win, 0),
            flagcxSuccess);
  for (int result : {flagcxInProgress, flagcxRemoteError}) {
    NativeOwnerProbe::result = result;
    EXPECT_EQ(mcclAdaptorCommDeregister(comm, handle), result);
    EXPECT_EQ(mcclAdaptorCommWindowDeregister(comm, win), result);
    EXPECT_EQ(NativeOwnerProbe::owners.count(win), 1u);
    EXPECT_EQ(win->base, (void *)44);
    EXPECT_EQ(comm->base, (void *)42);
    EXPECT_EQ(NativeOwnerProbe::frees, 0);
  }
  NativeOwnerProbe::result = 0;
  EXPECT_EQ(mcclAdaptorCommDeregister(comm, handle), flagcxSuccess);
  EXPECT_EQ(mcclAdaptorCommWindowDeregister(comm, win), flagcxSuccess);
  EXPECT_EQ(deregistrations, 3);
  EXPECT_EQ(windowDeregistrations, 3);
  EXPECT_EQ(NativeOwnerProbe::frees, 1);
  EXPECT_EQ(mcclAdaptorCommDestroy(comm), flagcxSuccess);
  EXPECT_TRUE(NativeOwnerProbe::owners.empty());
}

TEST(CclOwnerContract, MetaxAbortCleanupDoesNotReenterRetiredNativeContext) {
  using namespace mcclOwnerContract;
  NativeOwnerProbe::reset();
  deregistrations = windowDeregistrations = 0;
  auto *comm = (flagcxInnerComm_t)make();
  void *handle = nullptr;
  mcclOwnerContract::flagcxInnerWindow_t win = nullptr;
  ASSERT_EQ(mcclAdaptorCommRegister(comm, (void *)45, 16, &handle),
            flagcxSuccess);
  ASSERT_EQ(mcclAdaptorCommWindowRegister(comm, (void *)45, 16, &win, 0),
            flagcxSuccess);
  ASSERT_EQ(mcclAdaptorCommAbort(comm), flagcxSuccess);
  NativeOwnerProbe::result = flagcxRemoteError;
  EXPECT_EQ(mcclAdaptorCommDeregister(comm, handle), flagcxSuccess);
  EXPECT_EQ(mcclAdaptorCommWindowDeregister(comm, win), flagcxSuccess);
  EXPECT_EQ(deregistrations + windowDeregistrations, 0);
  EXPECT_EQ(mcclAdaptorCommDestroy(comm), flagcxSuccess);
  EXPECT_EQ(NativeOwnerProbe::destroys, 0);
  EXPECT_EQ(NativeOwnerProbe::frees, 2);
  EXPECT_TRUE(NativeOwnerProbe::owners.empty());
}
