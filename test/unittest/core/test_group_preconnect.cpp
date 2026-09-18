/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include <gtest/gtest.h>

#include "adaptor.h"
#include "group.h"
#include "launch_kernel.h"
#include "proxy.h"

#include <cstdlib>
#include <cstring>

namespace {

flagcxResult_t setDeviceSuccess(int) { return flagcxSuccess; }

int failingAsyncJobCalls = 0;

flagcxResult_t failAsyncJob(flagcxAsyncJob *) {
  failingAsyncJobCalls++;
  return flagcxInternalError;
}

class ScopedGroupGlobals {
public:
  ScopedGroupGlobals()
      : savedDeviceAdaptor_(deviceAdaptor),
        savedDeviceKernel_(deviceAsyncKernel) {
    std::memset(&fakeDeviceAdaptor_, 0, sizeof(fakeDeviceAdaptor_));
    std::strncpy(fakeDeviceAdaptor_.name, "group-test",
                 sizeof(fakeDeviceAdaptor_.name) - 1);
    fakeDeviceAdaptor_.setDevice = setDeviceSuccess;
    deviceAdaptor = &fakeDeviceAdaptor_;
    deviceAsyncKernel = nullptr;

    flagcxGroupDepth = 1;
    flagcxGroupCommHead = nullptr;
    flagcxGroupCommPreconnectHead = nullptr;
    flagcxGroupJobMainPtr = nullptr;
    std::memset(&flagcxGroupJobMain, 0, sizeof(flagcxGroupJobMain));
    flagcxIntruQueueConstruct(&flagcxAsyncJobs);
    failingAsyncJobCalls = 0;
  }

  ~ScopedGroupGlobals() {
    deviceAdaptor = savedDeviceAdaptor_;
    deviceAsyncKernel = savedDeviceKernel_;
    flagcxGroupDepth = 0;
    flagcxGroupCommHead = nullptr;
    flagcxGroupCommPreconnectHead = nullptr;
    flagcxGroupJobMainPtr = nullptr;
    std::memset(&flagcxGroupJobMain, 0, sizeof(flagcxGroupJobMain));
    flagcxIntruQueueConstruct(&flagcxAsyncJobs);
  }

private:
  flagcxDeviceAdaptor_latest *savedDeviceAdaptor_;
  flagcxLaunchFunc_t savedDeviceKernel_;
  flagcxDeviceAdaptor_latest fakeDeviceAdaptor_;
};

} // namespace

TEST(GroupPreconnectOwnership, FailedLaunchDoesNotRevisitConsumedList) {
  ScopedGroupGlobals globals;

  flagcxHeteroComm comm = {};
  flagcxProxyState proxy = {};
  volatile uint32_t abortFlag = 0;
  uint64_t connectSend[1] = {};
  uint64_t connectRecv[1] = {};

  proxy.initialized = 1;
  comm.proxyState = &proxy;
  comm.abortFlag = &abortFlag;
  comm.rank = 0;
  comm.nRanks = 1;
  comm.localRanks = 1;
  comm.connectSend = connectSend;
  comm.connectRecv = connectRecv;

  // Run one independent failing job alongside a successful preconnect. The
  // failure enters groupCleanup() only after the preconnect list was consumed,
  // reproducing the old cleanup traversal through preconnectNext=0x1.
  flagcxAsyncJob *failingJob = nullptr;
  ASSERT_EQ(flagcxCalloc(&failingJob, 1), flagcxSuccess);
  failingJob->func = failAsyncJob;
  failingJob->destructor = std::free;
  failingJob->state = flagcxGroupJobRunning;
  failingJob->comm = &comm;
  flagcxIntruQueueEnqueue(&flagcxAsyncJobs, failingJob);

  comm.preconnectNext = nullptr;
  flagcxGroupCommPreconnectHead = &comm;

  EXPECT_EQ(flagcxGroupEndInternal(), flagcxInternalError);
  EXPECT_EQ(failingAsyncJobCalls, 1);
  EXPECT_EQ(comm.preconnectNext, reinterpret_cast<flagcxHeteroComm *>(0x1));
  EXPECT_EQ(flagcxGroupCommPreconnectHead, nullptr);
  EXPECT_TRUE(flagcxIntruQueueEmpty(&flagcxAsyncJobs));
}
