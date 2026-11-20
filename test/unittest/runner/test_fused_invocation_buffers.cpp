#include "alloc.h"
#include "comm.h"
#include "global_comm.h"
#include "proxy.h"
#include "runner.h"
#include "uni_runner_impl.h"
#include "work_completion.hpp"
#include <cstdlib>

namespace {
class FusedInvocationBuffersTest : public WorkCompletionTest {
protected:
  inline static FusedInvocationBuffersTest *active;
  flagcxComm comm = {};
  flagcxHeteroComm hetero = {};
  flagcxProxyState proxy = {};
  int freeCalls = 0, freed = 0;
  void *failFree = nullptr;
  bool failPrivateStream = false;
  void SetUp() override {
    WorkCompletionTest::SetUp();
    active = this;
    comm.heteroComm = &hetero;
    hetero.proxyState = &proxy;
    device.deviceFree = [](void *pointer, flagcxMemType_t, flagcxStream_t) {
      active->freeCalls++;
      if (pointer == active->failFree)
        return flagcxSystemError;
      active->freed++;
      free(pointer);
      return flagcxSuccess;
    };
    device.streamSynchronize = [](flagcxStream_t input) {
      // Only private streams may be synchronized; the borrowed user stream
      // may already contain unrelated future work.
      EXPECT_NE(input, (flagcxStream_t)&active->stream);
      return active->failPrivateStream ? flagcxRemoteError : flagcxSuccess;
    };
    device.streamDestroy = [](flagcxStream_t) { return flagcxSuccess; };
  }
  flagcxFusedInvocationBuffers *attach() {
    auto *buffers = (flagcxFusedInvocationBuffers *)calloc(
        1, sizeof(flagcxFusedInvocationBuffers));
    buffers->workspace = malloc(64);
    buffers->counters = malloc(64);
    proxy.uniRunnerState.pendingFusedBuffers = buffers;
    buffers->workspaceWork = prepare(&comm.submittedWork);
    flagcxWorkRetain(buffers->workspaceWork);
    stream.submitted = 1;
    EXPECT_EQ(finish(buffers->workspaceWork), flagcxSuccess);
    return buffers;
  }
  void TearDown() override {
    failFree = nullptr;
    failPrivateStream = false;
    stream.completed = stream.submitted;
    EXPECT_EQ(flagcxWorkWait(&comm.submittedWork), flagcxSuccess);
    EXPECT_EQ(flagcxFusedDrainPendingBuffers(&comm), flagcxSuccess);
    if (proxy.uniRunnerState.borrowedStreamWork != nullptr) {
      EXPECT_EQ(flagcxUniRunnerDrainWork(&comm, false), flagcxSuccess);
    }
    EXPECT_EQ(flagcxWorkRelease(&comm.submittedWork), flagcxSuccess);
    comm.finalizeState.commonPhase = flagcxFinalizeComplete;
    EXPECT_EQ(flagcxCommFreeInternalBuffers(&comm), flagcxSuccess);
    WorkCompletionTest::TearDown();
  }
};
} // namespace

TEST_F(FusedInvocationBuffersTest, KeepsBuffersUntilInvocationFenceCompletes) {
  auto *buffers = attach();
  EXPECT_EQ(flagcxFusedDrainPendingBuffers(&comm), flagcxInProgress);
  EXPECT_EQ(freeCalls, 0);
  EXPECT_EQ(proxy.uniRunnerState.pendingFusedBuffers, buffers);
  stream.completed = 1;
  stream.submitted = 2; // Later unrelated work must not extend the fence.
  EXPECT_EQ(flagcxFusedDrainPendingBuffers(&comm), flagcxSuccess);
  EXPECT_EQ(freed, 2);
  EXPECT_EQ(proxy.uniRunnerState.pendingFusedBuffers, nullptr);
}

TEST_F(FusedInvocationBuffersTest, FailedRunnerDrainCannotFreeWorkspace) {
  auto *buffers = attach();
  stream.completed = 1;
  buffers->runnerCleanupPending = true;
  proxy.uniRunnerState.redStream = (flagcxStream_t)&otherStream;
  failPrivateStream = true;
  EXPECT_EQ(flagcxFusedDrainPendingBuffers(&comm), flagcxRemoteError);
  EXPECT_EQ(freeCalls, 0);
  EXPECT_EQ(proxy.uniRunnerState.pendingFusedBuffers, buffers);
  failPrivateStream = false;
  EXPECT_EQ(flagcxFusedDrainPendingBuffers(&comm), flagcxSuccess);
  EXPECT_EQ(freed, 2);
}

TEST_F(FusedInvocationBuffersTest,
       PartialFreeRetryDoesNotRepeatCounterRelease) {
  auto *buffers = attach();
  stream.completed = 1;
  failFree = buffers->workspace;
  EXPECT_EQ(flagcxFusedDrainPendingBuffers(&comm), flagcxSystemError);
  EXPECT_EQ(buffers->counters, nullptr);
  EXPECT_EQ(buffers->workspaceWork, nullptr);
  EXPECT_EQ(freed, 1);
  failFree = nullptr;
  EXPECT_EQ(flagcxFusedDrainPendingBuffers(&comm), flagcxSuccess);
  EXPECT_EQ(freed, 2);
  EXPECT_EQ(freeCalls, 3);
}

TEST_F(FusedInvocationBuffersTest, DeferredBorrowedStreamUsesSavedBoundary) {
  auto *state = &proxy.uniRunnerState;
  state->ipcCommStream = (flagcxStream_t)&stream;
  state->borrowedStreamWork = prepare(&comm.submittedWork);
  flagcxWorkRetain(state->borrowedStreamWork);
  stream.submitted = 1;
  EXPECT_EQ(flagcxUniRunnerDrainWork(&comm, true), flagcxInProgress);
  EXPECT_EQ(recorded, 1);
  stream.submitted = 2;
  stream.completed = 1;
  EXPECT_EQ(flagcxUniRunnerDrainWork(&comm, false), flagcxSuccess);
  EXPECT_EQ(recorded, 1);
  EXPECT_EQ(state->borrowedStreamWork, nullptr);
}

TEST_F(FusedInvocationBuffersTest, DefaultStreamMarkerIsAlsoRetired) {
  auto *state = &proxy.uniRunnerState;
  state->borrowedStreamWork = prepare(&comm.submittedWork);
  flagcxWorkRetain(state->borrowedStreamWork);
  EXPECT_EQ(flagcxUniRunnerDrainWork(&comm, true), flagcxSuccess);
  EXPECT_EQ(recorded, 1);
  EXPECT_EQ(state->borrowedStreamWork, nullptr);
}

TEST_F(FusedInvocationBuffersTest,
       FinalizeTransfersPrivateBuffersForLocalFree) {
  auto *buffers = attach();
  stream.completed = 1;
  ASSERT_EQ(flagcxWorkWait(&comm.submittedWork), flagcxSuccess);
  comm.finalizeState.commonPhase = flagcxFinalizeOwners;
  void *workspace = buffers->workspace;
  EXPECT_EQ(flagcxFusedDrainPendingBuffers(&comm), flagcxSuccess);
  EXPECT_EQ(proxy.uniRunnerState.pendingFusedBuffers, nullptr);
  EXPECT_EQ(freeCalls, 0);
  EXPECT_EQ(deviceSyncs, 0);
  ASSERT_NE(comm.internalBuffers, nullptr);
  EXPECT_EQ(comm.internalBuffers->pointer, workspace);
  comm.finalizeState.commonPhase = flagcxFinalizeComplete;
  failFree = comm.internalBuffers->next->pointer;
  EXPECT_EQ(flagcxCommFreeInternalBuffers(&comm), flagcxSystemError);
  EXPECT_EQ(freed, 1);
  failFree = nullptr;
  EXPECT_EQ(flagcxCommFreeInternalBuffers(&comm), flagcxSuccess);
  EXPECT_EQ(freed, 2);
  EXPECT_EQ(freeCalls, 3);
}

TEST_F(FusedInvocationBuffersTest, PersistentPrivateCachesTransferToParent) {
  auto *state = &proxy.uniRunnerState;
  // Resources may also be prepared before the first invocation sets the
  // lifetimeOwner. Keep ipcOwner until the final transfer succeeds.
  state->ipcOwner = &comm;
  state->streamFlagsPool = (uint64_t *)malloc(64);
  state->redCompletionCountersPool = (uint64_t *)malloc(64);
  state->groupedArScratch = malloc(64);
  state->ipcParentFlagsDevice = (uint64_t *)malloc(64);
  comm.finalizeState.commonPhase = flagcxFinalizeOwners;
  ASSERT_EQ(cleanupUniRunnerPersistentState(state), flagcxSuccess);
  EXPECT_EQ(freeCalls, 0);
  ASSERT_EQ(cleanupUniRunnerPersistentState(state), flagcxSuccess);
  comm.finalizeState.commonPhase = flagcxFinalizeComplete;
  EXPECT_EQ(flagcxCommFreeInternalBuffers(&comm), flagcxSuccess);
  EXPECT_EQ(freed, 4);
}

TEST_F(FusedInvocationBuffersTest, FinalizeRetainsInvocationFifoAllocations) {
  auto *state = &proxy.uniRunnerState;
  state->fifo = new flagcxFifo;
  state->ipcFifo = new flagcxFifo;
  state->fifo->buffer = (uint64_t *)malloc(64);
  state->ipcFifo->buffer = (uint64_t *)malloc(64);
  comm.finalizeState.commonPhase = flagcxFinalizeOwners;
  EXPECT_EQ(cleanupUniRunner(&comm, false), flagcxSuccess);
  EXPECT_EQ(state->fifo, nullptr);
  EXPECT_EQ(state->ipcFifo, nullptr);
  EXPECT_EQ(freeCalls, 0);
  comm.finalizeState.commonPhase = flagcxFinalizeComplete;
  EXPECT_EQ(flagcxCommFreeInternalBuffers(&comm), flagcxSuccess);
  EXPECT_EQ(freed, 2);
}
