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
static int finalizes, destroys, frees;
static int result;
static void reset() {
  ASSERT_TRUE(owners.empty());
  finalizes = destroys = frees = 0;
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
