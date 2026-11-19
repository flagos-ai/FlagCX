#include "adaptor.h"
#include "comm_lifecycle.h"
#include "device_api/flagcx_device.h"
#include "global_comm.h"
#include <atomic>
#include <chrono>
#include <future>
#include <gtest/gtest.h>
#include <thread>

TEST(CommLifecycleTest, FinalizingRejectsNewDeviceOwners) {
  flagcxComm comm = {};
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  char buffer[32] = {};
  for (int state = flagcxCommFinalizing; state <= flagcxCommDestroying;
       state++) {
    comm.finalizeState.lifecycle = (enum flagcxCommLifecycle)state;
    flagcxDevComm_t devComm = (flagcxDevComm_t)1;
    flagcxDevMem_t devMem = (flagcxDevMem_t)1;
    EXPECT_EQ(flagcxDevCommCreate(&comm, &reqs, &devComm), flagcxInvalidUsage);
    EXPECT_EQ(devComm, nullptr);
    EXPECT_EQ(
        flagcxDevMemCreate(&comm, buffer, sizeof(buffer), nullptr, &devMem),
        flagcxInvalidUsage);
    EXPECT_EQ(devMem, nullptr);
    EXPECT_EQ(comm.activeDevComms, nullptr);
    EXPECT_EQ(comm.activeDevMems, nullptr);
  }
}

TEST(CommLifecycleTest, AdmissionDoesNotReopenAfterCleanupFailure) {
  flagcxComm comm = {};
  EXPECT_EQ(flagcxCommRequireActive(&comm), flagcxSuccess);
  comm.finalizeState.lifecycle = flagcxCommFinalizing;
  comm.finalizeState.lastResult = flagcxRemoteError;
  EXPECT_EQ(flagcxCommRequireActive(&comm), flagcxInvalidUsage);
  EXPECT_EQ(comm.finalizeState.lastResult, flagcxRemoteError);
  EXPECT_EQ(flagcxCommRequireActive(nullptr), flagcxInvalidArgument);
}

TEST(CommLifecycleTest, VendorProgressDoesNotCompleteFlagcxFinalizeEarly) {
  static std::atomic<bool> polling, complete;
  static int calls;
  polling = complete = false;
  calls = 0;
  struct ProviderGuard {
    flagcxCCLAdaptor *saved = cclAdaptors[flagcxCCLAdaptorDevice];
    flagcxCCLAdaptor provider = *saved;
    ProviderGuard() { cclAdaptors[flagcxCCLAdaptorDevice] = &provider; }
    ~ProviderGuard() { cclAdaptors[flagcxCCLAdaptorDevice] = saved; }
  } guard;
  guard.provider.commFinalize = [](flagcxInnerComm_t) {
    calls++;
    return flagcxInProgress;
  };
  guard.provider.commGetAsyncError = [](flagcxInnerComm_t,
                                        flagcxResult_t *status) {
    polling = true;
    *status = complete ? flagcxSuccess : flagcxInProgress;
    return flagcxSuccess;
  };
  flagcxComm comm = {};
  comm.commType = flagcxCommunicatorHomo;
  comm.homoComm = (flagcxInnerComm_t)1;
  auto work =
      std::async(std::launch::async, [&] { return flagcxCommFinalize(&comm); });
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!polling && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  EXPECT_TRUE(polling);
  flagcxResult_t status = flagcxSuccess;
  EXPECT_EQ(flagcxCommGetAsyncError(&comm, &status), flagcxSuccess);
  EXPECT_EQ(status, flagcxInProgress);
  complete = true;
  EXPECT_EQ(work.get(), flagcxSuccess);
  EXPECT_EQ(flagcxCommFinalize(&comm), flagcxSuccess);
  EXPECT_EQ(calls, 1);
  EXPECT_EQ(comm.finalizeState.commonPhase, flagcxFinalizeComplete);
}
