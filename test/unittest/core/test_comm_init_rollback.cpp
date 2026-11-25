#include "rma_init_rollback.hpp"
#include <cstring>
#include <map>
#include <optional>
#include <string>
namespace {
class CommInitRollback : public RmaInitRollbackTest {
protected:
  std::map<std::string, std::optional<std::string>> environment;
  bool failUniqueId = false;
  int nativeInits = 0;
  void SetUp() override {
    RmaInitRollbackTest::SetUp();
    for (const char *name :
         {"FLAGCX_USE_HETERO_COMM", "FLAGCX_USE_TUNER",
          "FLAGCX_DEVICE_ADAPTOR_PLUGIN", "FLAGCX_CCL_ADAPTOR_PLUGIN"}) {
      const char *value = getenv(name);
      environment[name] =
          value ? std::optional<std::string>(value) : std::nullopt;
      unsetenv(name);
    }
    setenv("FLAGCX_USE_HETERO_COMM", "0", 1);
    setenv("FLAGCX_USE_TUNER", "0", 1);
    device.getDevice = [](int *device) {
      *device = 7;
      return flagcxSuccess;
    };
    device.getVendor = [](char *vendor) {
      strcpy(vendor, "MOCK");
      return flagcxSuccess;
    };
    ccl.getUniqueId = [](flagcxInnerUniqueId_t *id) {
      if (static_cast<CommInitRollback *>(current)->failUniqueId)
        return flagcxRemoteError;
      memset(*id, 0, sizeof(**id));
      return flagcxSuccess;
    };
    ccl.commInitRank = [](flagcxInnerComm_t *comm, int, flagcxInnerUniqueId_t,
                          int, bootstrapState *) {
      ++static_cast<CommInitRollback *>(current)->nativeInits;
      *comm = (flagcxInnerComm_t)0x1000;
      return flagcxRemoteError;
    };
    ccl.commFinalize = [](flagcxInnerComm_t) {
      ++static_cast<CommInitRollback *>(current)->nativeFinalizes;
      return flagcxNotSupported;
    };
  }
  void TearDown() override {
    RmaInitRollbackTest::TearDown();
    for (const auto &entry : environment) {
      if (entry.second)
        setenv(entry.first.c_str(), entry.second->c_str(), 1);
      else
        unsetenv(entry.first.c_str());
    }
  }
};
} // namespace
TEST_F(CommInitRollback, RollbackRetainsOuterUntilNativeOwnerSucceeds) {
  makeOwner();
  unmapFailures = 2;
  ASSERT_EQ(flagcxHeteroCommStartRma(&owner), flagcxSystemError);
  ASSERT_NE(owner, nullptr);
  transferToOuter();
  nativeFailures = 1;
  EXPECT_EQ(flagcxCommFinalize(outer), flagcxInvalidUsage);
  EXPECT_EQ(flagcxAllReduce(nullptr, nullptr, 0, flagcxChar, flagcxSum, outer,
                            nullptr),
            flagcxInvalidUsage);
  EXPECT_EQ(flagcxCommDestroy(outer), flagcxRemoteError);
  EXPECT_EQ(outer->heteroComm, nullptr);
  EXPECT_NE(outer->homoComm, nullptr);
  EXPECT_EQ(nativeFinalizes, 0);
  EXPECT_EQ(doneFrees, 1);
  EXPECT_EQ(readyFrees, 1);
  EXPECT_EQ(flagcxCommDestroy(outer), flagcxSuccess);
  outer = nullptr;
  EXPECT_EQ(nativeDestroys, 2);
  EXPECT_EQ(doneFrees + readyFrees, 2);
}
TEST_F(CommInitRollback, CachedProxyErrorDoesNotLeaveADanglingInnerOwner) {
  makeOwner();
  owner->proxyState->workerCleanupResult = flagcxRemoteError;
  transferToOuter();
  EXPECT_EQ(flagcxCommDestroy(outer), flagcxRemoteError);
  EXPECT_EQ(outer->heteroComm, nullptr);
  EXPECT_EQ(nativeDestroys, 0);
  EXPECT_EQ(flagcxCommDestroy(outer), flagcxSuccess);
  outer = nullptr;
  EXPECT_EQ(nativeDestroys, 1);
  EXPECT_EQ(nativeFinalizes, 0);
}

TEST_F(CommInitRollback,
       NativeInitFailureRetainsRollbackOwnerBeforePolicyInit) {
  flagcxUniqueId id = {};
  ASSERT_EQ(flagcxGetUniqueId(&id), flagcxSuccess);
  EXPECT_EQ(flagcxCommInitRank(&outer, 1, &id, 0), flagcxRemoteError);
  ASSERT_NE(outer, nullptr);
  EXPECT_EQ(nativeInits, 1);
  EXPECT_TRUE(outer->initRollbackPending);
  EXPECT_FALSE(outer->allowNativeDestroyFinalize);
  EXPECT_FALSE(outer->initRollbackNeedsAgreement);
  EXPECT_STREQ(flagcxGetLastError(outer),
               flagcxGetErrorString(flagcxRemoteError));
  EXPECT_EQ(flagcxCommFinalize(outer), flagcxInvalidUsage);
  EXPECT_EQ(flagcxOneSideStagingRegister(outer, (void *)0x2000, 16),
            flagcxInvalidUsage);
  nativeFailures = 1;
  EXPECT_EQ(flagcxCommDestroy(outer), flagcxRemoteError);
  ASSERT_NE(outer, nullptr);
  EXPECT_NE(outer->homoComm, nullptr);
  EXPECT_EQ(nativeFinalizes, 0);
  EXPECT_EQ(flagcxCommDestroy(outer), flagcxSuccess);
  outer = nullptr;
  EXPECT_EQ(nativeDestroys, 2);
}

TEST_F(CommInitRollback,
       UniqueIdFailureRetainsRollbackOwnerWithoutNativeContext) {
  failUniqueId = true;
  flagcxUniqueId id = {};
  ASSERT_EQ(flagcxGetUniqueId(&id), flagcxSuccess);
  EXPECT_EQ(flagcxCommInitRank(&outer, 1, &id, 0), flagcxRemoteError);
  ASSERT_NE(outer, nullptr);
  EXPECT_TRUE(outer->initRollbackPending);
  EXPECT_EQ(nativeInits, 0);
  EXPECT_EQ(flagcxCommDestroy(outer), flagcxSuccess);
  outer = nullptr;
  EXPECT_EQ(nativeFinalizes + nativeDestroys, 0);
}

TEST_F(CommInitRollback, InvalidInputDoesNotRetagAnExistingOwner) {
  flagcxComm existing = {};
  flagcxComm_t output = &existing;
  flagcxUniqueId id = {};
  EXPECT_EQ(flagcxCommInitRank(&output, 0, &id, 0), flagcxInvalidArgument);
  EXPECT_EQ(output, &existing);
  EXPECT_FALSE(existing.initRollbackPending);
  EXPECT_EQ(flagcxCommInitRank(&output, 1, nullptr, 0), flagcxInvalidArgument);
  EXPECT_EQ(output, &existing);
  EXPECT_FALSE(existing.initRollbackPending);
}
