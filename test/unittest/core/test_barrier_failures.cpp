#include "group.h"
#include "ipc_lifecycle_mock.hpp"
#include "runner.h"
#include <thread>

class BarrierFailureTest : public IpcLifecycleMock {
public:
  flagcxRunner runner = {};
  flagcxRunner *savedRunner = nullptr;
  flagcxResult_t allocationResult = flagcxSuccess;
  flagcxResult_t memsetResult = flagcxSuccess;
  flagcxResult_t collectiveResult = flagcxSuccess;
  flagcxResult_t syncResult = flagcxSuccess;
  bool allocateOnFailure = false;
  int allocations = 0, memsets = 0, collectives = 0;

  static BarrierFailureTest *self() {
    return static_cast<BarrierFailureTest *>(active);
  }
  void SetUp() override {
    IpcLifecycleMock::SetUp();
    savedRunner = flagcxRunners[flagcxHomoRunner];
    flagcxRunners[flagcxHomoRunner] = &runner;
    device.deviceMalloc = [](void **p, size_t bytes, flagcxMemType_t,
                             flagcxStream_t) {
      ++self()->allocations;
      if (self()->allocationResult == flagcxSuccess ||
          self()->allocateOnFailure)
        *p = calloc(1, bytes);
      return self()->allocationResult;
    };
    device.deviceMemset = [](void *p, int, size_t, flagcxMemType_t,
                             flagcxStream_t) {
      EXPECT_NE(p, nullptr);
      ++self()->memsets;
      return self()->memsetResult;
    };
    runner.allReduce = [](const void *input, void *output, size_t,
                          flagcxDataType_t, flagcxRedOp_t, flagcxComm_t,
                          flagcxStream_t) {
      EXPECT_NE(input, nullptr);
      EXPECT_EQ(input, output);
      ++self()->collectives;
      return self()->collectiveResult;
    };
    device.streamSynchronize = [](flagcxStream_t) {
      ++self()->syncs;
      return self()->syncResult;
    };
  }
  void TearDown() override {
    syncResult = flagcxSuccess;
    runner.groupEnd = [] { return flagcxSuccess; };
    while (flagcxGroupEnd(comm) == flagcxSuccess) {
    }
    flagcxGroupDepth = 0;
    flagcxRunners[flagcxHomoRunner] = savedRunner;
    IpcLifecycleMock::TearDown();
  }
};

TEST_F(BarrierFailureTest, AllocationFailureNeverUsesMissingBuffer) {
  for (auto error : {flagcxRemoteError, flagcxInProgress}) {
    allocationResult = error;
    EXPECT_EQ(flagcxBarrier(comm, nullptr), error);
    EXPECT_EQ(comm->pendingBarrierResources, nullptr);
  }
  EXPECT_EQ(allocations, 2);
  EXPECT_EQ(memsets, 0);
  EXPECT_EQ(collectives, 0);
  EXPECT_EQ(syncs, 0);
  EXPECT_TRUE(frees.empty());
}

TEST_F(BarrierFailureTest, PartialAllocationRetainsOwnerAcrossFailedFree) {
  allocationResult = flagcxRemoteError;
  allocateOnFailure = true;
  failFree = true;
  EXPECT_EQ(flagcxBarrier(comm, nullptr), flagcxRemoteError);
  ASSERT_NE(comm->pendingBarrierResources, nullptr);
  EXPECT_NE(comm->pendingBarrierResources->buffer, nullptr);
  EXPECT_TRUE(comm->pendingBarrierResources->streamDrained);
  EXPECT_EQ(memsets, 0);
  EXPECT_EQ(collectives, 0);
  EXPECT_TRUE(frees.empty());
  failFree = false;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(comm->pendingBarrierResources, nullptr);
  EXPECT_EQ(syncs, 1);
  EXPECT_EQ(frees.size(), 1u);
}

TEST_F(BarrierFailureTest, MemsetFailureStopsCollectiveAndReleasesScratch) {
  for (auto error : {flagcxRemoteError, flagcxInProgress}) {
    memsetResult = error;
    EXPECT_EQ(flagcxBarrier(comm, nullptr), error);
    EXPECT_EQ(comm->pendingBarrierResources, nullptr);
  }
  EXPECT_EQ(collectives, 0);
  EXPECT_EQ(syncs, 2);
  EXPECT_EQ(frees.size(), 2u);
}

TEST_F(BarrierFailureTest, CollectiveFailureAndFailedDrainKeepScratch) {
  collectiveResult = flagcxRemoteError;
  syncResult = flagcxUnhandledDeviceError;
  EXPECT_EQ(flagcxBarrier(comm, nullptr), flagcxRemoteError);
  ASSERT_NE(comm->pendingBarrierResources, nullptr);
  EXPECT_FALSE(comm->pendingBarrierResources->streamDrained);
  EXPECT_TRUE(frees.empty());
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxUnhandledDeviceError);
  EXPECT_TRUE(frees.empty());
  syncResult = flagcxSuccess;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(comm->pendingBarrierResources, nullptr);
  EXPECT_EQ(frees.size(), 1u);
}

TEST_F(BarrierFailureTest, SynchronizationFailureIsReportedAndRetryable) {
  syncResult = flagcxRemoteError;
  EXPECT_EQ(flagcxBarrier(comm, nullptr), flagcxRemoteError);
  ASSERT_NE(comm->pendingBarrierResources, nullptr);
  EXPECT_TRUE(frees.empty());
  syncResult = flagcxSuccess;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(frees.size(), 1u);
}

TEST_F(BarrierFailureTest, PendingCollectiveRetainsScratchUntilNativeProgress) {
  comm->homoComm = (flagcxInnerComm_t)0x9000;
  collectiveResult = flagcxInProgress;
  pendingPolls = 2;
  EXPECT_EQ(flagcxBarrier(comm, nullptr), flagcxInProgress);
  ASSERT_NE(comm->pendingBarrierResources, nullptr);
  EXPECT_TRUE(comm->pendingBarrierResources->collectivePending);
  EXPECT_EQ(syncs, 0);
  EXPECT_TRUE(frees.empty());
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxInProgress);
  EXPECT_TRUE(frees.empty());
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(comm->pendingBarrierResources, nullptr);
  EXPECT_EQ(frees.size(), 1u);
}

TEST_F(BarrierFailureTest, FreeFailureRetainsCompletedDrainProgress) {
  failFree = true;
  freeError = flagcxInProgress;
  EXPECT_EQ(flagcxBarrier(comm, nullptr), flagcxInProgress);
  ASSERT_NE(comm->pendingBarrierResources, nullptr);
  EXPECT_TRUE(comm->pendingBarrierResources->streamDrained);
  EXPECT_EQ(syncs, 1);
  failFree = false;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(syncs, 1);
  EXPECT_EQ(frees.size(), 1u);
  destroySuccessfully();
  EXPECT_EQ(frees.size(), 1u);
}

TEST_F(BarrierFailureTest, SuccessfulBarrierDrainsBeforeFreeingScratch) {
  EXPECT_EQ(flagcxBarrier(comm, nullptr), flagcxSuccess);
  EXPECT_EQ(comm->pendingBarrierResources, nullptr);
  EXPECT_EQ(collectives, 1);
  EXPECT_EQ(syncs, 1);
  EXPECT_EQ(frees.size(), 1u);
}

TEST_F(BarrierFailureTest, HeterogeneousGroupRejectsBeforeAllocation) {
  flagcxGroupDepth = 1;
  EXPECT_EQ(flagcxBarrier(comm, nullptr), flagcxInvalidUsage);
  flagcxGroupDepth = 0;
  EXPECT_EQ(allocations, 0);
}

TEST_F(BarrierFailureTest, NativeNestedGroupsRejectBarrierAndFinalize) {
  runner.groupStart = runner.groupEnd = [] { return flagcxSuccess; };
  ASSERT_EQ(flagcxGroupStart(NULL), flagcxSuccess);
  ASSERT_EQ(flagcxGroupStart(comm), flagcxSuccess);
  EXPECT_EQ(flagcxGetGroupDepth(), 0);
  EXPECT_EQ(flagcxBarrier(comm, nullptr), flagcxInvalidUsage);
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxInvalidUsage);
  EXPECT_FALSE(comm->closing);
  EXPECT_EQ(allocations, 0);
  ASSERT_EQ(flagcxGroupEnd(comm), flagcxSuccess);
  EXPECT_EQ(flagcxBarrier(comm, nullptr), flagcxInvalidUsage);
  ASSERT_EQ(flagcxGroupEnd(NULL), flagcxSuccess);
  EXPECT_EQ(flagcxBarrier(comm, nullptr), flagcxSuccess);
  EXPECT_EQ(allocations, 1);
}

TEST_F(BarrierFailureTest, NativeGroupDepthBelongsToCallingThread) {
  runner.groupStart = runner.groupEnd = [] { return flagcxSuccess; };
  ASSERT_EQ(flagcxGroupStart(comm), flagcxSuccess);
  std::thread another(
      [&] { EXPECT_EQ(flagcxGroupEnd(comm), flagcxInvalidUsage); });
  another.join();
  EXPECT_EQ(flagcxBarrier(comm, nullptr), flagcxInvalidUsage);
  EXPECT_EQ(allocations, 0);
  ASSERT_EQ(flagcxGroupEnd(comm), flagcxSuccess);
  EXPECT_EQ(flagcxBarrier(comm, nullptr), flagcxSuccess);
}

TEST_F(BarrierFailureTest, FailedNativeEndUnwindsTheMatchedBoundary) {
  runner.groupStart = [] { return flagcxSuccess; };
  runner.groupEnd = [] { return flagcxRemoteError; };
  ASSERT_EQ(flagcxGroupStart(comm), flagcxSuccess);
  EXPECT_EQ(flagcxGroupEnd(comm), flagcxRemoteError);
  EXPECT_EQ(flagcxGroupEnd(comm), flagcxInvalidUsage);
  EXPECT_EQ(flagcxBarrier(comm, nullptr), flagcxSuccess);
}

TEST_F(BarrierFailureTest, HybridGroupsReuseExistingHeterogeneousDepth) {
  sdk.groupStart = sdk.groupEnd = [] { return flagcxSuccess; };
  comm->commType = flagcxCommunicatorHybrid;
  ASSERT_EQ(flagcxGroupStart(comm), flagcxSuccess);
  EXPECT_EQ(flagcxGetGroupDepth(), 1);
  EXPECT_EQ(flagcxBarrier(comm, nullptr), flagcxInvalidUsage);
  EXPECT_EQ(allocations, 0);
  ASSERT_EQ(flagcxGroupEnd(comm), flagcxSuccess);
  EXPECT_EQ(flagcxGetGroupDepth(), 0);
  comm->commType = flagcxCommunicatorHomo;
  EXPECT_EQ(flagcxBarrier(comm, nullptr), flagcxSuccess);
}
