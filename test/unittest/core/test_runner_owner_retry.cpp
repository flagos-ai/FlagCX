#include "c2c_algo.h"
#include "flagcx_tuner.h"
#include "ipc_lifecycle_mock.hpp"
#include "runner.h"
#include "uni_runner_impl.h"

TEST_F(IpcLifecycleMock, C2cPlannerRejectsNativeOuterGroupBeforePreparation) {
  sdk.groupStart = sdk.groupEnd = [] { return flagcxSuccess; };
  device.deviceMalloc = [](void **, size_t, flagcxMemType_t, flagcxStream_t) {
    ++active->syncs;
    return flagcxRemoteError;
  };
  flagcxC2cPlanner planner{};
  flagcxC2cPlanner cached = planner;
  ASSERT_EQ(flagcxGroupStart(NULL), flagcxSuccess);
  ASSERT_EQ(flagcxGroupStart(comm), flagcxSuccess);
  EXPECT_EQ(flagcxGetGroupDepth(), 0);
  EXPECT_TRUE(flagcxGroupIsOpen());
  EXPECT_EQ(planner.execute(nullptr, nullptr, flagcxFloat, -1, nullptr),
            flagcxInvalidUsage);
  ASSERT_EQ(flagcxGroupEnd(comm), flagcxSuccess);
  EXPECT_EQ(cached.execute(nullptr, nullptr, flagcxFloat, -1, nullptr),
            flagcxInvalidUsage);
  EXPECT_EQ(comm->pendingC2cResources, nullptr);
  EXPECT_EQ(syncs, 0);
  EXPECT_TRUE(frees.empty());
  ASSERT_EQ(flagcxGroupEnd(NULL), flagcxSuccess);
  EXPECT_FALSE(flagcxGroupIsOpen());
}

TEST_F(IpcLifecycleMock, TunerReleaseRetainsAliasesUntilStrictSuccess) {
  auto inner = (flagcxInnerComm_t)0x9000;
  comm->homoComm = comm->homoInterComm = comm->tunerInnerComm = inner;
  comm->homoCommMap[TunerCollCategory{}] = inner;
  comm->homoBestCommMap[TunerCollCategory{}] = inner;
  comm->commMap[flagcxCommTag{}] = inner;
  sdk.commDestroy = [](flagcxInnerComm_t) {
    ++active->destroyCalls;
    return active->nativeResult;
  };
  for (auto status : {flagcxInProgress, flagcxRemoteError}) {
    nativeResult = status;
    EXPECT_EQ(flagcxDestroyHomoCommOwner(comm, inner), status);
    EXPECT_EQ(comm->homoComm, inner);
    EXPECT_EQ(comm->commMap.begin()->second, inner);
    EXPECT_EQ(comm->homoCommMap.begin()->second, inner);
    EXPECT_EQ(comm->homoBestCommMap.begin()->second, inner);
  }
  nativeResult = flagcxSuccess;
  ASSERT_EQ(flagcxDestroyHomoCommOwner(comm, inner), flagcxSuccess);
  EXPECT_EQ(comm->homoComm, nullptr);
  EXPECT_EQ(comm->homoInterComm, nullptr);
  EXPECT_EQ(comm->tunerInnerComm, nullptr);
  EXPECT_EQ(comm->commMap.begin()->second, nullptr);
  EXPECT_EQ(comm->homoCommMap.begin()->second, nullptr);
  EXPECT_EQ(comm->homoBestCommMap.begin()->second, nullptr);
}

TEST_F(IpcLifecycleMock, FifoFreeIsRetryableAndIdempotent) {
  flagcxFifo fifo;
  fifo.buffer = (uint64_t *)calloc(1, 64);
  auto owner = fifo.buffer;
  failFree = true;
  freeError = flagcxInProgress;
  EXPECT_EQ(fifo.flagcxFifoDestroy(), flagcxInProgress);
  EXPECT_EQ(fifo.buffer, owner);
  failFree = false;
  ASSERT_EQ(fifo.flagcxFifoDestroy(), flagcxSuccess);
  EXPECT_EQ(fifo.buffer, nullptr);
  ASSERT_EQ(fifo.flagcxFifoDestroy(), flagcxSuccess);
  EXPECT_EQ(frees.size(), 1u);
}

TEST_F(IpcLifecycleMock, C2cFailedDrainRetainsScratchAndBothStreamProgress) {
  auto owner = (flagcxC2cResources *)calloc(1, sizeof(flagcxC2cResources));
  owner->scratch = calloc(1, 64);
  owner->userStream = (flagcxStream_t)0x1000;
  owner->privateStream = (flagcxStream_t)0x2000;
  owner->needDrain = true;
  comm->pendingC2cResources = owner;
  device.streamSynchronize = [](flagcxStream_t stream) {
    ++active->syncs;
    return stream == (flagcxStream_t)0x2000 && active->failDevice
               ? flagcxRemoteError
               : flagcxSuccess;
  };
  device.streamDestroy = [](flagcxStream_t) { return flagcxSuccess; };
  failDevice = true;
  EXPECT_EQ(flagcxC2cDrainPendingResources(comm), flagcxRemoteError);
  EXPECT_EQ(comm->pendingC2cResources, owner);
  EXPECT_TRUE(owner->userDrained);
  EXPECT_FALSE(owner->privateDrained);
  EXPECT_TRUE(frees.empty());
  failDevice = false;
  failFree = true;
  freeError = flagcxInProgress;
  EXPECT_EQ(flagcxC2cDrainPendingResources(comm), flagcxInProgress);
  EXPECT_TRUE(owner->privateDrained);
  failFree = false;
  ASSERT_EQ(flagcxC2cDrainPendingResources(comm), flagcxSuccess);
  EXPECT_EQ(comm->pendingC2cResources, nullptr);
  EXPECT_EQ(frees.size(), 1u);
  EXPECT_EQ(syncs, 3);
}

class C2cScratchCompletion : public IpcLifecycleMock {
public:
  bool scratchInUse = false;

  void SetUp() override {
    IpcLifecycleMock::SetUp();
    comm->nclusters = comm->homoRanks = comm->homoInterRanks = 1;
    comm->clusterIds = (int *)calloc(1, sizeof(int));
    comm->clusterSizes = (int *)calloc(1, sizeof(int));
    comm->clusterSizes[0] = 1;
    comm->globalRank2HomoRank = (int *)calloc(1, sizeof(int));
    comm->clusterInterRankList = {{0}};
    sdk.groupStart = sdk.groupEnd = [] { return flagcxSuccess; };
    sdk.reduce = [](const void *, void *, size_t, flagcxDataType_t,
                    flagcxRedOp_t, int, flagcxInnerComm_t,
                    flagcxStream_t) { return flagcxSuccess; };
    sdk.allReduce = [](const void *, void *, size_t, flagcxDataType_t,
                       flagcxRedOp_t, flagcxInnerComm_t,
                       flagcxStream_t) { return flagcxSuccess; };
    sdk.reduceScatter = [](const void *, void *, size_t, flagcxDataType_t,
                           flagcxRedOp_t, flagcxInnerComm_t, flagcxStream_t) {
      ++active->backendCleanups;
      static_cast<C2cScratchCompletion *>(active)->scratchInUse = true;
      return flagcxSuccess;
    };
    device.streamCreate = [](flagcxStream_t *stream) {
      *stream = (flagcxStream_t)0x1000;
      return flagcxSuccess;
    };
    device.streamSynchronize = [](flagcxStream_t stream) {
      ++active->syncs;
      auto *test = static_cast<C2cScratchCompletion *>(active);
      if (stream == (flagcxStream_t)0x2000 && test->scratchInUse) {
        if (test->nativeResult != flagcxSuccess)
          return test->nativeResult;
        test->scratchInUse = false;
      }
      return flagcxSuccess;
    };
    device.deviceFree = [](void *pointer, flagcxMemType_t, flagcxStream_t) {
      EXPECT_FALSE(static_cast<C2cScratchCompletion *>(active)->scratchInUse)
          << "immediate free must wait for the final native scratch reader";
      active->frees.push_back(pointer);
      free(pointer);
      return flagcxSuccess;
    };
    device.streamDestroy = [](flagcxStream_t stream) {
      EXPECT_EQ(stream, (flagcxStream_t)0x1000);
      ++active->destroyCalls;
      return flagcxSuccess;
    };
  }

  flagcxResult_t execute() {
    int input = 0, output = 0;
    flagcxC2cPlanner planner(1, 1, 0, comm, flagcxCommOpReduceScatter,
                             flagcxMin);
    return planner.execute(&input, &output, flagcxInt, 0,
                           (flagcxStream_t)0x2000);
  }
};

TEST_F(C2cScratchCompletion, SuccessfulSubmissionDrainsBeforeImmediateFree) {
  ASSERT_EQ(execute(), flagcxSuccess);
  EXPECT_EQ(backendCleanups, 1);
  EXPECT_FALSE(scratchInUse);
  EXPECT_EQ(frees.size(), 1u);
  EXPECT_EQ(destroyCalls, 1);
  EXPECT_EQ(comm->pendingC2cResources, nullptr);
}

TEST_F(C2cScratchCompletion, SuccessfulSubmissionRetainsFailedDrainForRetry) {
  for (auto status : {flagcxInProgress, flagcxRemoteError}) {
    auto freed = frees.size();
    nativeResult = status;
    ASSERT_EQ(execute(), status);
    auto *owner = comm->pendingC2cResources;
    ASSERT_NE(owner, nullptr);
    EXPECT_NE(owner->scratch, nullptr);
    EXPECT_TRUE(owner->needDrain);
    EXPECT_FALSE(owner->userDrained);
    EXPECT_TRUE(scratchInUse);
    EXPECT_EQ(frees.size(), freed);
    EXPECT_EQ(destroyCalls, freed);
    nativeResult = flagcxSuccess;
    ASSERT_EQ(flagcxC2cDrainPendingResources(comm), flagcxSuccess);
    EXPECT_FALSE(scratchInUse);
    EXPECT_EQ(frees.size(), freed + 1);
    EXPECT_EQ(destroyCalls, freed + 1);
    EXPECT_EQ(comm->pendingC2cResources, nullptr);
  }
  EXPECT_EQ(backendCleanups,
            2); // Retrying cleanup never resubmits the collective.
}

TEST_F(IpcLifecycleMock, UniRunnerScratchIsRetainedWhenFreeFails) {
  addTransport();
  comm->uniRunnerKernelSupported = true;
  device.deviceMalloc = [](void **p, size_t bytes, flagcxMemType_t,
                           flagcxStream_t) {
    *p = calloc(1, bytes);
    return flagcxRemoteError;
  };
  failFree = true;
  freeError = flagcxInProgress;
  EXPECT_EQ(
      uniRunner.reduce(NULL, NULL, 16, flagcxFloat, flagcxSum, 0, comm, NULL),
      flagcxRemoteError);
  auto &state = comm->heteroComm->proxyState->uniRunnerState;
  ASSERT_NE(state.pendingFusedBuffers, nullptr);
  EXPECT_NE(state.pendingFusedBuffers->workspace, nullptr);
  failFree = false;
  EXPECT_EQ(flagcxFusedDrainPendingBuffers(comm), flagcxSuccess);
  EXPECT_EQ(state.pendingFusedBuffers, nullptr);
  EXPECT_EQ(frees.size(), 1u);
}

TEST_F(IpcLifecycleMock, UniRunnerPartialStreamCreationRetainsFirstOwner) {
  addTransport();
  comm->uniRunnerKernelSupported = true;
  device.streamCreate = [](flagcxStream_t *stream) {
    if (++active->syncs == 2)
      return flagcxNotSupported;
    *stream = (flagcxStream_t)0x1000;
    return flagcxSuccess;
  };
  device.streamSynchronize = [](flagcxStream_t) { return flagcxSuccess; };
  device.streamDestroy = [](flagcxStream_t) {
    return active->failDevice ? flagcxRemoteError : flagcxSuccess;
  };
  device.setDevice = [](int) { return flagcxSuccess; };
  failDevice = true;
  EXPECT_EQ(initUniRunner(comm, nullptr), flagcxRemoteError);
  auto &state = comm->heteroComm->proxyState->uniRunnerState;
  EXPECT_EQ(state.redStream, (flagcxStream_t)0x1000);
  failDevice = false;
  EXPECT_EQ(cleanupUniRunner(comm), flagcxSuccess);
  EXPECT_EQ(state.redStream, nullptr);
}

TEST_F(IpcLifecycleMock,
       CancelledCopyKeepsRunnerWorkspaceUntilTransportRetires) {
  addTransport();
  auto &state = comm->heteroComm->proxyState->uniRunnerState;
  auto *buffers = (flagcxFusedInvocationBuffers *)calloc(
      1, sizeof(flagcxFusedInvocationBuffers));
  buffers->workspace = calloc(1, 64);
  buffers->runnerCleanupPending = true;
  state.pendingFusedBuffers = buffers;
  state.redStream = (flagcxStream_t)0x1000;
  device.streamSynchronize = [](flagcxStream_t) { return flagcxSuccess; };
  device.streamDestroy = [](flagcxStream_t) {
    ++active->syncs;
    return flagcxSuccess;
  };
  comm->heteroComm->proxyState->asyncResult = flagcxRemoteError;
  comm->heteroComm->proxyState->pendingTransportOps = 1;
  EXPECT_EQ(flagcxFusedDrainPendingBuffers(comm), flagcxInProgress);
  EXPECT_EQ(state.pendingFusedBuffers, buffers);
  EXPECT_NE(buffers->workspace, nullptr);
  EXPECT_TRUE(frees.empty());
  EXPECT_EQ(syncs, 0);
  comm->heteroComm->proxyState->pendingTransportOps = 0;
  EXPECT_EQ(flagcxFusedDrainPendingBuffers(comm), flagcxSuccess);
  EXPECT_EQ(state.pendingFusedBuffers, nullptr);
  EXPECT_EQ(syncs, 1);
  EXPECT_EQ(frees.size(), 1u);
}

TEST_F(IpcLifecycleMock, CancelledCopyKeepsC2cScratchUntilTransportRetires) {
  addTransport();
  auto *owner = (flagcxC2cResources *)calloc(1, sizeof(flagcxC2cResources));
  owner->scratch = calloc(1, 64);
  comm->pendingC2cResources = owner;
  comm->heteroComm->proxyState->asyncResult = flagcxRemoteError;
  comm->heteroComm->proxyState->pendingTransportOps = 1;
  EXPECT_EQ(flagcxC2cDrainPendingResources(comm), flagcxInProgress);
  EXPECT_EQ(comm->pendingC2cResources, owner);
  EXPECT_TRUE(frees.empty());
  comm->heteroComm->proxyState->pendingTransportOps = 0;
  EXPECT_EQ(flagcxC2cDrainPendingResources(comm), flagcxSuccess);
  EXPECT_EQ(comm->pendingC2cResources, nullptr);
  EXPECT_EQ(frees.size(), 1u);
}

TEST_F(IpcLifecycleMock,
       FusedFinishRetainsWorkspaceUntilAcceptedRequestsRetire) {
  addTransport();
  auto &state = comm->heteroComm->proxyState->uniRunnerState;
  auto *buffers = (flagcxFusedInvocationBuffers *)calloc(
      1, sizeof(flagcxFusedInvocationBuffers));
  buffers->workspace = calloc(1, 64);
  buffers->counters = calloc(1, 64);
  comm->heteroComm->proxyState->asyncResult = flagcxRemoteError;
  comm->heteroComm->proxyState->pendingTransportOps = 1;
  device.deviceSynchronize = [] {
    ADD_FAILURE() << "device synchronization cannot retire accepted requests";
    return flagcxSuccess;
  };
  EXPECT_EQ(flagcxFusedFinishInvocation(flagcxRemoteError, true, comm, buffers,
                                        nullptr, false),
            flagcxRemoteError);
  ASSERT_EQ(state.pendingFusedBuffers, buffers);
  EXPECT_TRUE(buffers->runnerCleanupPending);
  EXPECT_NE(buffers->workspace, nullptr);
  EXPECT_NE(buffers->counters, nullptr);
  EXPECT_TRUE(frees.empty());
  EXPECT_EQ(flagcxFusedDrainPendingBuffers(comm), flagcxInProgress);
  comm->heteroComm->proxyState->pendingTransportOps = 0;
  EXPECT_EQ(flagcxFusedDrainPendingBuffers(comm), flagcxSuccess);
  EXPECT_EQ(state.pendingFusedBuffers, nullptr);
  EXPECT_EQ(frees.size(), 2u);
}

TEST_F(IpcLifecycleMock, FusedPreparationRetainsFailedStreamDrainForRetry) {
  addTransport();
  auto &state = comm->heteroComm->proxyState->uniRunnerState;
  auto *buffers = (flagcxFusedInvocationBuffers *)calloc(
      1, sizeof(flagcxFusedInvocationBuffers));
  buffers->workspace = calloc(1, 64);
  device.deviceSynchronize = [] {
    ADD_FAILURE() << "failure cleanup must not synchronize the entire device";
    return flagcxSuccess;
  };
  device.streamSynchronize = [](flagcxStream_t stream) {
    EXPECT_EQ(stream, (flagcxStream_t)0x1000);
    ++active->syncs;
    return active->nativeResult;
  };
  nativeResult = flagcxInProgress;
  EXPECT_EQ(flagcxFusedFinishInvocation(flagcxRemoteError, false, comm, buffers,
                                        (flagcxStream_t)0x1000, true),
            flagcxRemoteError);
  ASSERT_EQ(state.pendingFusedBuffers, buffers);
  EXPECT_TRUE(buffers->streamDrainPending);
  EXPECT_TRUE(frees.empty());
  nativeResult = flagcxRemoteError;
  EXPECT_EQ(flagcxFusedDrainPendingBuffers(comm), flagcxRemoteError);
  EXPECT_TRUE(buffers->streamDrainPending);
  EXPECT_TRUE(frees.empty());
  nativeResult = flagcxSuccess;
  failFree = true;
  freeError = flagcxInProgress;
  EXPECT_EQ(flagcxFusedDrainPendingBuffers(comm), flagcxInProgress);
  EXPECT_FALSE(buffers->streamDrainPending);
  EXPECT_EQ(syncs, 3);
  failFree = false;
  EXPECT_EQ(flagcxFusedDrainPendingBuffers(comm), flagcxSuccess);
  EXPECT_EQ(syncs, 3); // Do not repeat the completed stream drain.
  EXPECT_EQ(frees.size(), 1u);
  EXPECT_EQ(state.pendingFusedBuffers, nullptr);
}

TEST_F(IpcLifecycleMock,
       C2cCreationFailureRetainsReturnedStreamUntilDestroyRetry) {
  comm->nclusters = comm->homoRanks = comm->homoInterRanks = 1;
  comm->clusterIds = (int *)calloc(1, sizeof(int));
  comm->clusterSizes = (int *)calloc(1, sizeof(int));
  comm->clusterSizes[0] = 1;
  comm->globalRank2HomoRank = (int *)calloc(1, sizeof(int));
  comm->clusterInterRankList = {{0}};
  device.streamCreate = [](flagcxStream_t *stream) {
    *stream = (flagcxStream_t)0x1000;
    return flagcxRemoteError;
  };
  device.streamSynchronize = [](flagcxStream_t) { return flagcxSuccess; };
  device.streamDestroy = [](flagcxStream_t stream) {
    EXPECT_EQ(stream, (flagcxStream_t)0x1000);
    ++active->backendCleanups;
    return active->nativeResult;
  };
  nativeResult = flagcxInProgress;
  int input = 0, output = 0;
  flagcxC2cPlanner planner(1, 1, 0, comm, flagcxCommOpAllGather, flagcxRedNoOp);
  EXPECT_EQ(
      planner.execute(&input, &output, flagcxInt, 0, (flagcxStream_t)0x2000),
      flagcxRemoteError);
  ASSERT_NE(comm->pendingC2cResources, nullptr);
  EXPECT_EQ(comm->pendingC2cResources->privateStream, (flagcxStream_t)0x1000);
  EXPECT_EQ(backendCleanups, 1);
  nativeResult = flagcxSuccess;
  EXPECT_EQ(flagcxC2cDrainPendingResources(comm), flagcxSuccess);
  EXPECT_EQ(comm->pendingC2cResources, nullptr);
  EXPECT_EQ(backendCleanups, 2);
}
