#include "comm_lifecycle.h"
#include "device_api/flagcx_device.h"
#include "global_comm.h"
#include <gtest/gtest.h>

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
