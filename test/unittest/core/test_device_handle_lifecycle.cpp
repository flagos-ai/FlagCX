#include "adaptor.h"
#include "adaptor_plugin_load.h"
#include "flagcx.h"
#include "global_comm.h"
#include "launch_kernel.h"
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <set>
#include <string>

namespace {
class HandleOwnerRetry : public ::testing::Test {
protected:
  inline static HandleOwnerRetry *current;
  flagcxDeviceAdaptor *savedDevice = nullptr;
  flagcxDeviceAdaptor device = {};
  flagcxDeviceHandle_t handle = nullptr, other = nullptr;
  flagcxHandlerGroup_t group = nullptr;
  flagcxComm_t comm = nullptr;
  flagcxResult_t freeFailure = flagcxSystemError;
  uint64_t *held = nullptr;
  int allocations = 0, frees = 0, freeCalls = 0, failures = 0;
  std::set<void *> live;
  std::string savedDeviceEnv, savedCclEnv;
  bool hadDeviceEnv = false, hadCclEnv = false;

  void SetUp() override {
    current = this;
    ASSERT_EQ(streamValueBufferPool.clear(), flagcxSuccess);
    savedDevice = deviceAdaptor;
    device = *savedDevice;
    deviceAdaptor = &device;
    device.getDevice = [](int *ordinal) {
      *ordinal = 0;
      return flagcxSuccess;
    };
    device.deviceMalloc = [](void **pointer, size_t bytes, flagcxMemType_t,
                             flagcxStream_t) {
      *pointer = malloc(bytes);
      if (*pointer == nullptr)
        return flagcxSystemError;
      current->allocations++;
      current->live.insert(*pointer);
      return flagcxSuccess;
    };
    device.hostGetDevicePointer = [](void **pointer, void *host) {
      *pointer = host;
      return flagcxSuccess;
    };
    device.deviceFree = [](void *pointer, flagcxMemType_t, flagcxStream_t) {
      current->freeCalls++;
      if (current->failures > 0) {
        current->failures--;
        return current->freeFailure;
      }
      EXPECT_EQ(current->live.erase(pointer), 1u) << "duplicate pool free";
      current->frees++;
      free(pointer);
      return flagcxSuccess;
    };
    const char *value = getenv("FLAGCX_DEVICE_ADAPTOR_PLUGIN");
    hadDeviceEnv = value != nullptr;
    if (value)
      savedDeviceEnv = value;
    value = getenv("FLAGCX_CCL_ADAPTOR_PLUGIN");
    hadCclEnv = value != nullptr;
    if (value)
      savedCclEnv = value;
    setenv("FLAGCX_DEVICE_ADAPTOR_PLUGIN", "none", 1);
    setenv("FLAGCX_CCL_ADAPTOR_PLUGIN", "none", 1);
  }
  void acquireSignal() {
    void *devicePointer = nullptr;
    ASSERT_EQ(streamValueBufferPool.acquire(&held, &devicePointer),
              flagcxSuccess);
    ASSERT_NE(held, nullptr);
  }
  void releaseSignal() {
    streamValueBufferPool.release(held);
    held = nullptr;
  }
  flagcxResult_t freeHandle(flagcxDeviceHandle_t *pointer) {
    flagcxResult_t result = flagcxDeviceHandleFree(*pointer);
    if (result == flagcxSuccess)
      *pointer = nullptr;
    return result;
  }
  flagcxResult_t freeGroup() {
    flagcxResult_t result = flagcxHandleFree(group);
    if (result == flagcxSuccess)
      group = nullptr;
    return result;
  }
  flagcxResult_t freeComm() {
    flagcxResult_t result = flagcxCommDestroy(comm);
    if (result == flagcxSuccess)
      comm = nullptr;
    return result;
  }
  void makeComm(bool rollback) {
    ASSERT_EQ(flagcxDeviceAdaptorPluginInit(), flagcxSuccess);
    ASSERT_EQ(flagcxCCLAdaptorPluginInit(), flagcxSuccess);
    comm = (flagcxComm_t)calloc(1, sizeof(flagcxComm));
    ASSERT_NE(comm, nullptr);
    comm->commType = flagcxCommunicatorHomo;
    comm->finalizeState.commonPhase = flagcxFinalizeComplete;
    comm->finalizeState.lifecycle = flagcxCommFinalized;
    comm->initRollbackPending = rollback;
    // No native communicator or work was created in this host fixture.
  }
  void TearDown() override {
    failures = 0;
    if (held != nullptr)
      releaseSignal();
    if (handle != nullptr) {
      EXPECT_EQ(freeHandle(&handle), flagcxSuccess);
    }
    if (other != nullptr) {
      EXPECT_EQ(freeHandle(&other), flagcxSuccess);
    }
    if (group != nullptr) {
      EXPECT_EQ(freeGroup(), flagcxSuccess);
    }
    if (comm != nullptr) {
      EXPECT_EQ(freeComm(), flagcxSuccess);
    }
    // Check before a manual clear so an orphaned ref cannot hide a leak.
    EXPECT_EQ(allocations, frees);
    EXPECT_TRUE(live.empty());
    EXPECT_EQ(streamValueBufferPool.clear(), flagcxSuccess);
    if (hadDeviceEnv)
      setenv("FLAGCX_DEVICE_ADAPTOR_PLUGIN", savedDeviceEnv.c_str(), 1);
    else
      unsetenv("FLAGCX_DEVICE_ADAPTOR_PLUGIN");
    if (hadCclEnv)
      setenv("FLAGCX_CCL_ADAPTOR_PLUGIN", savedCclEnv.c_str(), 1);
    else
      unsetenv("FLAGCX_CCL_ADAPTOR_PLUGIN");
    deviceAdaptor = savedDevice;
  }
};

TEST_F(HandleOwnerRetry, ActiveSignalRetainsPublicHandleUntilRetry) {
  ASSERT_EQ(flagcxDeviceHandleInit(&handle), flagcxSuccess);
  ASSERT_NO_FATAL_FAILURE(acquireSignal());
  auto originalGetDevice = handle->getDevice;
  EXPECT_EQ(freeHandle(&handle), flagcxInvalidUsage);
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(handle->getDevice, originalGetDevice);
  EXPECT_EQ(frees, 0);
  releaseSignal();
  EXPECT_EQ(freeHandle(&handle), flagcxSuccess);
  EXPECT_EQ(frees, allocations);
}

TEST_F(HandleOwnerRetry, FailedPoolFreeRetainsHandleAcrossRepeatedFailures) {
  ASSERT_EQ(flagcxDeviceHandleInit(&handle), flagcxSuccess);
  ASSERT_NO_FATAL_FAILURE(acquireSignal());
  releaseSignal();
  failures = 2;
  EXPECT_EQ(freeHandle(&handle), flagcxSystemError);
  EXPECT_NE(handle, nullptr);
  EXPECT_EQ(freeHandle(&handle), flagcxSystemError);
  EXPECT_NE(handle, nullptr);
  EXPECT_EQ(frees, 0);
  EXPECT_EQ(freeHandle(&handle), flagcxSuccess);
  EXPECT_EQ(freeCalls, 3);
  EXPECT_EQ(frees, 1);
}

TEST_F(HandleOwnerRetry, GroupFreePropagatesErrorAndRetainsAllFields) {
  ASSERT_EQ(flagcxHandleInit(&group), flagcxSuccess);
  auto original = group->devHandle;
  group->uniqueId = (flagcxUniqueId_t)0x1234;
  group->comm = (flagcxComm_t)0x5678;
  ASSERT_NO_FATAL_FAILURE(acquireSignal());
  EXPECT_EQ(freeGroup(), flagcxInvalidUsage);
  ASSERT_NE(group, nullptr);
  EXPECT_EQ(group->devHandle, original);
  EXPECT_EQ(group->uniqueId, (flagcxUniqueId_t)0x1234);
  EXPECT_EQ(group->comm, (flagcxComm_t)0x5678);
  releaseSignal();
  failures = 1;
  EXPECT_EQ(freeGroup(), flagcxSystemError);
  EXPECT_EQ(group->devHandle, original);
  EXPECT_EQ(freeGroup(), flagcxSuccess);
  EXPECT_EQ(frees, 1);
}

TEST_F(HandleOwnerRetry, NewOwnerCanCloseAfterAnOlderOwnersFailedFree) {
  ASSERT_EQ(flagcxDeviceHandleInit(&handle), flagcxSuccess);
  ASSERT_NO_FATAL_FAILURE(acquireSignal());
  releaseSignal();
  failures = 1;
  EXPECT_EQ(freeHandle(&handle), flagcxSystemError);
  ASSERT_EQ(flagcxDeviceHandleInit(&other), flagcxSuccess);
  EXPECT_EQ(freeHandle(&handle), flagcxSuccess);
  EXPECT_EQ(frees, 0); // The newer owner still holds the device reference.
  EXPECT_EQ(freeHandle(&other), flagcxSuccess);
  EXPECT_EQ(frees, 1);
  ASSERT_EQ(flagcxDeviceHandleInit(&handle), flagcxSuccess);
  ASSERT_NO_FATAL_FAILURE(acquireSignal());
  releaseSignal();
  EXPECT_EQ(freeHandle(&handle), flagcxSuccess);
  EXPECT_EQ(frees,
            2); // No phantom reference survives a second init/free cycle.
}

TEST_F(HandleOwnerRetry, RetryDoesNotReleaseANewerOwnersCclReference) {
  auto *originalCcl = cclAdaptors[flagcxCCLAdaptorDevice];
  setenv("FLAGCX_CCL_ADAPTOR_PLUGIN", FLAGCX_TEST_HANDLE_CCL_PLUGIN, 1);
  ASSERT_EQ(flagcxDeviceHandleInit(&handle), flagcxSuccess);
  ASSERT_STREQ(cclAdaptors[flagcxCCLAdaptorDevice]->name, "handle-owner-test");
  ASSERT_NO_FATAL_FAILURE(acquireSignal());
  releaseSignal();
  failures = 1;
  EXPECT_EQ(freeHandle(&handle), flagcxSystemError);
  EXPECT_EQ(cclAdaptors[flagcxCCLAdaptorDevice], originalCcl);
  ASSERT_EQ(flagcxDeviceHandleInit(&other), flagcxSuccess);
  ASSERT_STREQ(cclAdaptors[flagcxCCLAdaptorDevice]->name, "handle-owner-test");
  EXPECT_EQ(freeHandle(&handle), flagcxSuccess);
  // A repeated CCL decrement would unload the only reference held by other.
  EXPECT_STREQ(cclAdaptors[flagcxCCLAdaptorDevice]->name, "handle-owner-test");
  EXPECT_EQ(freeHandle(&other), flagcxSuccess);
  EXPECT_EQ(cclAdaptors[flagcxCCLAdaptorDevice], originalCcl);
  EXPECT_EQ(frees, 1);
}

TEST_F(HandleOwnerRetry, NullArgumentsDoNotAcquirePluginReferences) {
  EXPECT_EQ(flagcxDeviceHandleInit(nullptr), flagcxInvalidArgument);
  EXPECT_EQ(flagcxHandleInit(nullptr), flagcxInvalidArgument);
  EXPECT_EQ(flagcxDeviceHandleFree(nullptr), flagcxSuccess);
  EXPECT_EQ(flagcxHandleFree(nullptr), flagcxSuccess);
  ASSERT_EQ(flagcxDeviceHandleInit(&handle), flagcxSuccess);
  ASSERT_NO_FATAL_FAILURE(acquireSignal());
  releaseSignal();
  EXPECT_EQ(freeHandle(&handle), flagcxSuccess);
  EXPECT_EQ(frees, 1);
}

TEST_F(HandleOwnerRetry, PendingPoolFreeRetainsPublicHandleAcrossRetries) {
  ASSERT_EQ(flagcxDeviceHandleInit(&handle), flagcxSuccess);
  ASSERT_NO_FATAL_FAILURE(acquireSignal());
  releaseSignal();
  freeFailure = flagcxInProgress;
  failures = 2;
  auto original = handle;
  EXPECT_EQ(freeHandle(&handle), flagcxInProgress);
  ASSERT_EQ(handle, original);
  EXPECT_EQ(freeHandle(&handle), flagcxInProgress);
  ASSERT_EQ(handle, original);
  EXPECT_EQ(frees, 0);
  EXPECT_EQ(freeHandle(&handle), flagcxSuccess);
  EXPECT_EQ(freeCalls, 3);
  EXPECT_EQ(frees, 1);
}

TEST_F(HandleOwnerRetry, PendingGroupFreeKeepsEveryFieldUntilSuccess) {
  ASSERT_EQ(flagcxHandleInit(&group), flagcxSuccess);
  auto original = group->devHandle;
  group->uniqueId = (flagcxUniqueId_t)0x1234;
  group->comm = (flagcxComm_t)0x5678;
  ASSERT_NO_FATAL_FAILURE(acquireSignal());
  releaseSignal();
  freeFailure = flagcxInProgress;
  failures = 1;
  EXPECT_EQ(freeGroup(), flagcxInProgress);
  ASSERT_NE(group, nullptr);
  EXPECT_EQ(group->devHandle, original);
  EXPECT_EQ(group->uniqueId, (flagcxUniqueId_t)0x1234);
  EXPECT_EQ(group->comm, (flagcxComm_t)0x5678);
  EXPECT_EQ(frees, 0);
  EXPECT_EQ(freeGroup(), flagcxSuccess);
  EXPECT_EQ(frees, 1);
}

TEST_F(HandleOwnerRetry, PendingCommPluginUnloadRetainsLocalDestroyPhase) {
  ASSERT_NO_FATAL_FAILURE(makeComm(false));
  ASSERT_NO_FATAL_FAILURE(acquireSignal());
  releaseSignal();
  failures = 2;
  freeFailure = flagcxInProgress;
  EXPECT_EQ(freeComm(), flagcxInProgress);
  ASSERT_NE(comm, nullptr);
  EXPECT_EQ(comm->destroyPhase, flagcxDestroyDevicePlugin);
  EXPECT_EQ(freeComm(), flagcxInProgress);
  EXPECT_EQ(comm->destroyPhase, flagcxDestroyDevicePlugin);
  EXPECT_EQ(frees, 0);
  EXPECT_EQ(freeComm(), flagcxSuccess);
  EXPECT_EQ(freeCalls, 3);
  EXPECT_EQ(frees, 1);
}

TEST_F(HandleOwnerRetry, PendingInitRollbackKeepsCommReachableWithoutFinalize) {
  ASSERT_NO_FATAL_FAILURE(makeComm(true));
  comm->finalizeState.commonPhase = flagcxFinalizeDrain;
  ASSERT_NO_FATAL_FAILURE(acquireSignal());
  releaseSignal();
  failures = 1;
  freeFailure = flagcxInProgress;
  EXPECT_EQ(freeComm(), flagcxInProgress);
  ASSERT_NE(comm, nullptr);
  EXPECT_TRUE(comm->initRollbackPending);
  EXPECT_EQ(comm->destroyPhase, flagcxDestroyDevicePlugin);
  EXPECT_EQ(comm->finalizeState.commonPhase, flagcxFinalizeDrain);
  EXPECT_EQ(freeComm(), flagcxSuccess);
  EXPECT_EQ(frees, 1);
}

TEST_F(HandleOwnerRetry, PendingDeviceFreeCannotReleaseNewerCclReference) {
  auto *originalCcl = cclAdaptors[flagcxCCLAdaptorDevice];
  setenv("FLAGCX_CCL_ADAPTOR_PLUGIN", FLAGCX_TEST_HANDLE_CCL_PLUGIN, 1);
  ASSERT_EQ(flagcxDeviceHandleInit(&handle), flagcxSuccess);
  ASSERT_NO_FATAL_FAILURE(acquireSignal());
  releaseSignal();
  failures = 1;
  freeFailure = flagcxInProgress;
  EXPECT_EQ(freeHandle(&handle), flagcxInProgress);
  EXPECT_EQ(cclAdaptors[flagcxCCLAdaptorDevice], originalCcl);
  ASSERT_EQ(flagcxDeviceHandleInit(&other), flagcxSuccess);
  EXPECT_EQ(freeHandle(&handle), flagcxSuccess);
  EXPECT_STREQ(cclAdaptors[flagcxCCLAdaptorDevice]->name, "handle-owner-test");
  EXPECT_EQ(frees, 0);
  EXPECT_EQ(freeHandle(&other), flagcxSuccess);
  EXPECT_EQ(cclAdaptors[flagcxCCLAdaptorDevice], originalCcl);
  EXPECT_EQ(frees, 1);
}

TEST_F(HandleOwnerRetry, PendingPluginLoadCannotReportGroupInitSuccess) {
  ASSERT_EQ(flagcxDeviceHandleInit(&handle), flagcxSuccess);
  ASSERT_NO_FATAL_FAILURE(acquireSignal());
  releaseSignal();
  setenv("FLAGCX_DEVICE_ADAPTOR_PLUGIN", FLAGCX_TEST_HANDLE_DEVICE_PLUGIN, 1);
  freeFailure = flagcxInProgress;
  failures = 2;
  EXPECT_EQ(flagcxDeviceHandleInit(&other), flagcxInProgress);
  EXPECT_EQ(other, nullptr); // No new plugin reference was acquired.
  EXPECT_EQ(flagcxHandleInit(&group), flagcxInProgress);
  EXPECT_EQ(group, nullptr);
  EXPECT_EQ(deviceAdaptor, &device);
  EXPECT_EQ(frees, 0);
  setenv("FLAGCX_DEVICE_ADAPTOR_PLUGIN", "none", 1);
  EXPECT_EQ(freeHandle(&handle), flagcxSuccess);
  EXPECT_EQ(frees, 1);
}

} // namespace
