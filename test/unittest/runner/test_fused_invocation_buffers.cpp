#include <gtest/gtest.h>

#include "adaptor.h"
#include "alloc.h"
#include "comm.h"
#include "global_comm.h"
#include "proxy.h"
#include "runner.h"

#include <cstdlib>

namespace {

flagcxResult_t syncResult = flagcxSuccess;
int freeCalls = 0;

flagcxResult_t setDevice(int) { return flagcxSuccess; }
flagcxResult_t synchronizeDevice() {
  ADD_FAILURE() << "device-wide synchronization";
  return flagcxInternalError;
}
flagcxResult_t createEvent(flagcxEvent_t *event, flagcxEventType_t) {
  *event = (flagcxEvent_t)std::malloc(1);
  return flagcxSuccess;
}
flagcxResult_t recordEvent(flagcxEvent_t, flagcxStream_t) {
  return flagcxSuccess;
}
flagcxResult_t queryEvent(flagcxEvent_t) { return syncResult; }
flagcxResult_t destroyEvent(flagcxEvent_t event) {
  std::free(event);
  return flagcxSuccess;
}
flagcxResult_t freeDevice(void *ptr, flagcxMemType_t, flagcxStream_t) {
  ++freeCalls;
  std::free(ptr);
  return flagcxSuccess;
}

class FusedInvocationBuffersTest : public ::testing::Test {
protected:
  flagcxDeviceAdaptor *savedAdaptor = nullptr;
  flagcxDeviceAdaptor fakeAdaptor{};

  void SetUp() override {
    savedAdaptor = deviceAdaptor;
    fakeAdaptor.setDevice = setDevice;
    fakeAdaptor.deviceSynchronize = synchronizeDevice;
    fakeAdaptor.deviceFree = freeDevice;
    fakeAdaptor.eventCreate = createEvent;
    fakeAdaptor.eventRecord = recordEvent;
    fakeAdaptor.eventQuery = queryEvent;
    fakeAdaptor.eventSynchronize = queryEvent;
    fakeAdaptor.eventDestroy = destroyEvent;
    deviceAdaptor = &fakeAdaptor;
    syncResult = flagcxSuccess;
    freeCalls = 0;
  }

  void TearDown() override { deviceAdaptor = savedAdaptor; }
};

TEST_F(FusedInvocationBuffersTest, KeepsBuffersUntilInvocationCompletes) {
  flagcxComm comm{};
  flagcxHeteroComm hcomm{};
  flagcxProxyState proxy{};
  comm.heteroComm = &hcomm;
  hcomm.proxyState = &proxy;

  flagcxFusedInvocationBuffers *buffers = nullptr;
  ASSERT_EQ(flagcxCalloc(&buffers, 1), flagcxSuccess);
  ASSERT_NE(buffers, nullptr);
  buffers->workspace = std::malloc(64);
  buffers->counters = std::malloc(64);
  ASSERT_NE(buffers->workspace, nullptr);
  ASSERT_NE(buffers->counters, nullptr);
  ASSERT_EQ(
      flagcxWorkPrepare(&comm.submittedWork, nullptr, &buffers->workspaceWork),
      flagcxSuccess);
  flagcxWorkRetain(buffers->workspaceWork);
  ASSERT_EQ(flagcxWorkFinish(buffers->workspaceWork, nullptr, flagcxSuccess,
                             flagcxWorkNativeGroup, 0, 1),
            flagcxSuccess);
  proxy.uniRunnerState.pendingFusedBuffers = buffers;

  syncResult = flagcxUnhandledDeviceError;
  EXPECT_EQ(flagcxFusedDrainPendingBuffers(&comm), flagcxUnhandledDeviceError);
  EXPECT_EQ(freeCalls, 0);
  EXPECT_EQ(proxy.uniRunnerState.pendingFusedBuffers, buffers);

  syncResult = flagcxSuccess;
  EXPECT_EQ(flagcxFusedDrainPendingBuffers(&comm), flagcxSuccess);
  EXPECT_EQ(freeCalls, 2);
  EXPECT_EQ(proxy.uniRunnerState.pendingFusedBuffers, nullptr);
  EXPECT_EQ(flagcxWorkWait(&comm.submittedWork), flagcxSuccess);
  EXPECT_EQ(flagcxWorkRelease(&comm.submittedWork), flagcxSuccess);
}

} // namespace
