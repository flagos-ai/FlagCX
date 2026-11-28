#include "flagcx_hetero.h"
#include "ipc_lifecycle_mock.hpp"

class StandaloneDeviceContext : public IpcLifecycleMock {
public:
  inline static StandaloneDeviceContext *context;
  flagcxHeteroComm_t child = nullptr;
  int selectedDevice = 0;
  flagcxResult_t selectionResult = flagcxSuccess;

  void SetUp() override {
    IpcLifecycleMock::SetUp();
    context = this;
    addTransport();
    child = comm->heteroComm;
    child->ownerComm = nullptr;
    child->cudaDev = 4;
    child->proxyState->uniRunnerState.groupedArScratch = calloc(1, 64);
    device.setDevice = [](int ordinal) {
      EXPECT_EQ(ordinal, 4);
      if (context->selectionResult == flagcxSuccess)
        context->selectedDevice = ordinal;
      return context->selectionResult;
    };
    device.deviceFree = [](void *pointer, flagcxMemType_t, flagcxStream_t) {
      EXPECT_EQ(context->selectedDevice, 4);
      if (context->failFree)
        return context->freeError;
      context->frees.push_back(pointer);
      free(pointer);
      return flagcxSuccess;
    };
  }
  void destroyed() {
    child = comm->heteroComm = nullptr;
    comm->commType = flagcxCommunicatorHomo;
    // The remaining parent fixture has no device allocations.
    device.setDevice = [](int) { return flagcxSuccess; };
  }
  void TearDown() override {
    selectionResult = flagcxSuccess;
    device.setDevice = [](int ordinal) {
      context->selectedDevice = ordinal;
      return flagcxSuccess;
    };
    if (child != nullptr)
      child->ownerComm = comm;
    IpcLifecycleMock::TearDown();
  }
};

TEST_F(StandaloneDeviceContext, CachedFinalizeDoesNotReplaceDestroySelection) {
  ASSERT_EQ(flagcxHeteroCommFinalize(child), flagcxSuccess);
  EXPECT_TRUE(frees.empty());
  // Model a retry that has already released the proxy. Its empty fast path
  // does not select a device, while the persistent runner owner still must.
  ASSERT_EQ(flagcxProxyDestroy(child), flagcxSuccess);
  ASSERT_EQ(child->proxyState->connectionPool, nullptr);
  selectedDevice = 1;
  ASSERT_EQ(flagcxHeteroCommDestroy(child), flagcxSuccess);
  EXPECT_EQ(selectedDevice, 4);
  EXPECT_EQ(frees.size(), 1u);
  destroyed();
}

TEST_F(StandaloneDeviceContext, SelectionAndReleaseFailuresKeepOwnerForRetry) {
  ASSERT_EQ(flagcxHeteroCommFinalize(child), flagcxSuccess);
  auto &scratch = child->proxyState->uniRunnerState.groupedArScratch;
  auto *allocation = scratch;
  selectedDevice = 1;
  selectionResult = flagcxRemoteError;
  EXPECT_EQ(flagcxHeteroCommDestroy(child), flagcxRemoteError);
  EXPECT_EQ(scratch, allocation);
  EXPECT_EQ(selectedDevice, 1);
  selectionResult = flagcxInProgress;
  EXPECT_EQ(flagcxHeteroCommDestroy(child), flagcxInProgress);
  EXPECT_EQ(scratch, allocation);
  EXPECT_TRUE(frees.empty());
  selectionResult = flagcxSuccess;
  failFree = true;
  EXPECT_EQ(flagcxHeteroCommDestroy(child), flagcxRemoteError);
  EXPECT_EQ(scratch, allocation);
  EXPECT_EQ(selectedDevice, 4);
  selectedDevice = 2;
  failFree = false;
  ASSERT_EQ(flagcxHeteroCommDestroy(child), flagcxSuccess);
  EXPECT_EQ(selectedDevice, 4);
  EXPECT_EQ(frees.size(), 1u);
  destroyed();
}
