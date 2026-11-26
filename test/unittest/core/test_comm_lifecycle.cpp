#include "adaptor.h"
#include "comm.h"
#include "comm_lifecycle.h"
#include "global_comm.h"
#include <gtest/gtest.h>

TEST(CommLifecycleState, ClosingAdmissionCannotReopenOnFailure) {
  flagcxComm comm = {};
  EXPECT_EQ(flagcxCommRequireActive(&comm), flagcxSuccess);
  for (int state = flagcxCommFinalizing; state <= flagcxCommDestroying;
       ++state) {
    comm.finalizeState.lifecycle = (flagcxCommLifecycle)state;
    comm.finalizeState.lastResult = flagcxRemoteError;
    EXPECT_EQ(flagcxCommRequireActive(&comm), flagcxInvalidUsage);
    EXPECT_EQ(comm.finalizeState.lastResult, flagcxRemoteError);
  }
  EXPECT_EQ(flagcxCommRequireActive(NULL), flagcxInvalidArgument);
}
TEST(CommLifecyclePolicy, EmptyRankKeepsItsDeclaredIpcDomain) {
  flagcxComm comm = {};
  comm.nranks = comm.localRanks = 1;
  comm.internalBuffersNeedPeerClose = true;
  EXPECT_EQ(flagcxCommInitFinalizePolicy(&comm), flagcxSuccess);
  EXPECT_TRUE(comm.ipcClosePolicyConfirmed);
  EXPECT_TRUE(comm.internalBuffersNeedPeerClose);
  EXPECT_EQ(comm.internalBuffers, nullptr);
}
TEST(CommLifecyclePolicy, MissingPeerMetadataDoesNotGrantPermission) {
  flagcxComm comm = {};
  comm.nranks = 2;
  EXPECT_EQ(flagcxCommInitFinalizePolicy(&comm), flagcxInvalidUsage);
  EXPECT_FALSE(comm.ipcClosePolicyConfirmed);
}

TEST(CommLifecycleState, PublicOneSideRegistrationRejectsClosedOwners) {
  flagcxHeteroComm hetero = {};
  flagcxComm comm = {};
  char buffer[16] = {};
  comm.commType = flagcxCommunicatorHybrid;
  comm.heteroComm = &hetero;
  for (int state = flagcxCommFinalizing; state <= flagcxCommDestroying;
       ++state) {
    comm.finalizeState.lifecycle = (flagcxCommLifecycle)state;
    EXPECT_EQ(flagcxOneSideRegister(&comm, buffer, sizeof(buffer)),
              flagcxInvalidUsage);
    EXPECT_EQ(flagcxOneSideSignalRegister(&comm, buffer, sizeof(buffer),
                                          FLAGCX_PTR_HOST),
              flagcxInvalidUsage);
    EXPECT_EQ(flagcxOneSideStagingRegister(&comm, buffer, sizeof(buffer)),
              flagcxInvalidUsage);
    EXPECT_EQ(hetero.oneSideHandleCount, 0);
    EXPECT_EQ(hetero.signalHandle, nullptr);
    EXPECT_EQ(hetero.stagingHandle, nullptr);
  }
  EXPECT_EQ(flagcxOneSideRegister(nullptr, buffer, sizeof(buffer)),
            flagcxInvalidArgument);
  EXPECT_EQ(flagcxOneSideSignalRegister(nullptr, buffer, sizeof(buffer),
                                        FLAGCX_PTR_HOST),
            flagcxInvalidArgument);
  EXPECT_EQ(flagcxOneSideStagingRegister(nullptr, buffer, sizeof(buffer)),
            flagcxInvalidArgument);
}

#include "device_api/flagcx_device.h"
#include "reg_pool.h"
#include <cstdlib>
#include <new>
#include <set>
#include <string>
#include <vector>

TEST(CommLifecycleDiagnostics, ErrorCodesHaveStableSdkIndependentMessages) {
  std::set<std::string> messages;
  for (int result = 0; result < flagcxNumResults; ++result) {
    const char *message = flagcxGetErrorString((flagcxResult_t)result);
    ASSERT_NE(message, nullptr);
    EXPECT_NE(*message, '\0');
    EXPECT_STRNE(message, "Not implemented.");
    messages.insert(message);
  }
  EXPECT_EQ(messages.size(), (size_t)flagcxNumResults);
  EXPECT_STREQ(flagcxGetErrorString((flagcxResult_t)-1), "Unknown error");
  EXPECT_STREQ(flagcxGetErrorString(flagcxNumResults), "Unknown error");
}

namespace {
class CommFinalizeOwnerTest : public ::testing::Test {
protected:
  inline static CommFinalizeOwnerTest *current;
  flagcxDeviceAdaptor device = {};
  flagcxCCLAdaptor sdk = {}, host = {};
  flagcxDeviceAdaptor *savedDevice;
  flagcxCCLAdaptor *savedSdk, *savedHost;
  flagcxComm_t comm = nullptr;
  int finalizes = 0, destroys = 0, polls = 0, frees = 0, aborts = 0;
  int failAbortAt = 0;
  int lastErrorQueries = 0;
  int hostFinalizes = 0, hostDestroys = 0;
  int pendingPolls = 0, eventFailures = 0, syncFailures = 0, eventDestroys = 0;
  flagcxResult_t finalizeResult = flagcxSuccess, asyncResult = flagcxSuccess;
  flagcxResult_t destroyResult = flagcxSuccess, freeResult = flagcxSuccess;
  flagcxResult_t abortResult = flagcxSuccess;
  std::set<void *> live;
  std::set<void *> aborted;
  std::vector<void *> finalized;
  void SetUp() override {
    current = this;
    savedDevice = deviceAdaptor;
    savedSdk = cclAdaptors[flagcxCCLAdaptorDevice];
    savedHost = cclAdaptors[flagcxCCLAdaptorHost];
    device.setDevice = [](int) { return flagcxSuccess; };
    device.deviceSynchronize = [] {
      ADD_FAILURE() << "device-wide drain";
      return flagcxRemoteError;
    };
    device.deviceFree = [](void *, flagcxMemType_t, flagcxStream_t) {
      ++current->frees;
      return current->freeResult;
    };
    device.eventCreate = [](flagcxEvent_t *event, flagcxEventType_t) {
      *event = (flagcxEvent_t)malloc(1);
      return *event != NULL ? flagcxSuccess : flagcxSystemError;
    };
    device.eventRecord = [](flagcxEvent_t, flagcxStream_t) {
      return flagcxSuccess;
    };
    device.eventQuery = [](flagcxEvent_t) { return flagcxSuccess; };
    device.eventSynchronize = [](flagcxEvent_t) {
      if (current->syncFailures > 0) {
        --current->syncFailures;
        return flagcxRemoteError;
      }
      return flagcxSuccess;
    };
    device.eventDestroy = [](flagcxEvent_t event) {
      ++current->eventDestroys;
      if (current->eventFailures > 0) {
        --current->eventFailures;
        return flagcxRemoteError;
      }
      free(event);
      return flagcxSuccess;
    };
    sdk.commFinalize = [](flagcxInnerComm_t inner) {
      ++current->finalizes;
      current->finalized.push_back(inner);
      return current->finalizeResult;
    };
    sdk.commGetAsyncError = [](flagcxInnerComm_t, flagcxResult_t *result) {
      ++current->polls;
      *result =
          current->pendingPolls-- > 0 ? flagcxInProgress : current->asyncResult;
      return flagcxSuccess;
    };
    sdk.getLastError = [](flagcxInnerComm_t inner) {
      ++current->lastErrorQueries;
      EXPECT_EQ(current->live.count(inner), 1u);
      EXPECT_EQ(current->aborted.count(inner), 0u);
      return "native diagnostic";
    };
    sdk.commDestroy = [](flagcxInnerComm_t inner) {
      ++current->destroys;
      if (current->destroyResult != flagcxSuccess)
        return current->destroyResult;
      EXPECT_EQ(current->live.erase(inner), 1u);
      free(inner);
      return flagcxSuccess;
    };
    sdk.commAbort = [](flagcxInnerComm_t inner) {
      ++current->aborts;
      EXPECT_EQ(current->live.count(inner), 1u);
      EXPECT_EQ(current->aborted.count(inner), 0u);
      if (current->abortResult != flagcxSuccess)
        return current->abortResult;
      if (current->aborts == current->failAbortAt)
        return flagcxRemoteError;
      current->aborted.insert(inner);
      return flagcxSuccess; // The adaptor retains the wrapper until Destroy.
    };
    host.commFinalize = [](flagcxInnerComm_t) {
      ++current->hostFinalizes;
      return flagcxSuccess;
    };
    host.commDestroy = [](flagcxInnerComm_t inner) {
      ++current->hostDestroys;
      EXPECT_EQ(current->live.erase(inner), 1u);
      free(inner);
      return flagcxSuccess;
    };
    deviceAdaptor = &device;
    cclAdaptors[flagcxCCLAdaptorDevice] = &sdk;
    cclAdaptors[flagcxCCLAdaptorHost] = &host;
    comm = (flagcxComm_t)calloc(1, sizeof(flagcxComm));
    new (&comm->homoCommMap) decltype(comm->homoCommMap);
    new (&comm->homoBestCommMap) decltype(comm->homoBestCommMap);
    new (&comm->commMap) decltype(comm->commMap);
    comm->commType = flagcxCommunicatorHomo;
    comm->nranks = comm->localRanks = 1;
    comm->ipcTable = comm->ipcTableInline;
    comm->ipcTableCapacity = FLAGCX_MAX_IPC_ENTRIES;
  }
  flagcxInnerComm_t makeOwner() {
    void *ptr = calloc(1, 16);
    live.insert(ptr);
    return (flagcxInnerComm_t)ptr;
  }
  void TearDown() override {
    finalizeResult = asyncResult = destroyResult = freeResult = abortResult =
        flagcxSuccess;
    if (comm != nullptr && comm->finalizeState.abortStarted &&
        !comm->finalizeState.abortComplete) {
      EXPECT_EQ(flagcxCommAbort(comm), flagcxSuccess);
      comm = nullptr;
    }
    if (comm != nullptr)
      EXPECT_EQ(flagcxCommDestroy(comm), flagcxSuccess);
    EXPECT_TRUE(live.empty());
    deviceAdaptor = savedDevice;
    cclAdaptors[flagcxCCLAdaptorDevice] = savedSdk;
    cclAdaptors[flagcxCCLAdaptorHost] = savedHost;
  }
};
} // namespace

TEST_F(CommFinalizeOwnerTest, NativeFinalizePollsOnceAcceptedAndIsIdempotent) {
  comm->homoComm = makeOwner();
  finalizeResult = flagcxInProgress;
  pendingPolls = 2;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(finalizes, 1);
  EXPECT_EQ(polls, 3);
  EXPECT_EQ(destroys, 0);
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(finalizes, 1);
}

TEST_F(CommFinalizeOwnerTest, AsyncErrorRetainsStartedNativeOwnerForRetry) {
  comm->homoComm = makeOwner();
  asyncResult = flagcxRemoteError;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxRemoteError);
  EXPECT_EQ(comm->finalizeState.commonPhase, flagcxFinalizeVendors);
  EXPECT_EQ(live.size(), 1u);
  asyncResult = flagcxSuccess;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(finalizes, 1);
  EXPECT_EQ(polls, 2);
}

TEST_F(CommFinalizeOwnerTest, LastErrorUsesNativeOnlyWhileActive) {
  comm->homoComm = makeOwner();
  EXPECT_STREQ(flagcxGetLastError(comm), "native diagnostic");
  EXPECT_EQ(lastErrorQueries, 1);
  asyncResult = flagcxRemoteError;
  ASSERT_EQ(flagcxCommFinalize(comm), flagcxRemoteError);
  EXPECT_STREQ(flagcxGetLastError(comm),
               flagcxGetErrorString(flagcxRemoteError));
  EXPECT_EQ(lastErrorQueries, 1);
}

TEST_F(CommFinalizeOwnerTest, LastErrorWithoutNativeOwnerStaysLocal) {
  EXPECT_STREQ(flagcxGetLastError(comm), flagcxGetErrorString(flagcxSuccess));
  EXPECT_EQ(lastErrorQueries, 0);
}

TEST_F(CommFinalizeOwnerTest, AbortAfterNativeAsyncErrorConsumesHandleOnce) {
  comm->homoComm = makeOwner();
  asyncResult = flagcxRemoteError;
  ASSERT_EQ(flagcxCommFinalize(comm), flagcxRemoteError);
  EXPECT_EQ(flagcxCommAbort(comm), flagcxSuccess);
  comm = nullptr;
  EXPECT_EQ(aborts, 1);
  EXPECT_EQ(finalizes, 1);
  EXPECT_EQ(polls, 1);
  EXPECT_EQ(destroys, 1); // Releases the retained adaptor wrapper.
  EXPECT_TRUE(live.empty());
}

TEST_F(CommFinalizeOwnerTest, AbortAfterDrainFailureAllowsCleanupRetry) {
  comm->homoComm = makeOwner();
  flagcxWorkRecord *work = nullptr;
  ASSERT_EQ(flagcxWorkPrepare(&comm->submittedWork, nullptr, &work),
            flagcxSuccess);
  ASSERT_EQ(
      flagcxWorkFinish(work, nullptr, flagcxSuccess, flagcxWorkNativeGroup, 0),
      flagcxSuccess);
  syncFailures = 1;
  ASSERT_EQ(flagcxCommFinalize(comm), flagcxRemoteError);
  freeResult = flagcxRemoteError;
  auto *buffer =
      (flagcxInternalBuffer *)calloc(1, sizeof(flagcxInternalBuffer));
  buffer->pointer = (void *)0x123000;
  comm->internalBuffers = buffer;
  EXPECT_EQ(flagcxCommAbort(comm), flagcxRemoteError);
  EXPECT_TRUE(comm->finalizeState.abortComplete);
  EXPECT_EQ(aborts, 1);
  EXPECT_EQ(flagcxCommRequireActive(comm), flagcxInvalidUsage);
  freeResult = flagcxSuccess;
  EXPECT_EQ(flagcxCommAbort(comm), flagcxSuccess);
  comm = nullptr;
  EXPECT_EQ(aborts, 1);
  EXPECT_EQ(finalizes, 0);
}

TEST_F(CommFinalizeOwnerTest, AbortRetainsPartialPrefixAndEveryAlias) {
  auto a = makeOwner(), b = makeOwner(), c = makeOwner();
  comm->homoCommMap[TunerCollCategory{}] = a;
  comm->commMap[flagcxCommTag{}] = b;
  comm->homoComm = comm->homoInterComm = comm->tunerInnerComm = c;
  failAbortAt = 2;
  EXPECT_EQ(flagcxCommAbort(comm), flagcxRemoteError);
  EXPECT_EQ(comm->finalizeState.abortIndex, 1u);
  EXPECT_STREQ(flagcxGetLastError(comm),
               flagcxGetErrorString(flagcxRemoteError));
  EXPECT_EQ(lastErrorQueries, 0);
  EXPECT_EQ(comm->homoComm, c);
  EXPECT_EQ(comm->commMap.begin()->second, b);
  EXPECT_EQ(live.size(), 3u);
  EXPECT_EQ(flagcxCommDestroy(comm), flagcxInvalidUsage);
  EXPECT_EQ(flagcxCommAbort(comm), flagcxSuccess);
  comm = nullptr;
  EXPECT_EQ(aborts, 4);
  EXPECT_EQ(finalizes, 0);
  EXPECT_EQ(destroys, 3);
}

TEST_F(CommFinalizeOwnerTest, AbortInProgressAndBusyLifecyclePreserveOwners) {
  auto owner = comm->homoComm = makeOwner();
  comm->finalizeState.running = 1;
  EXPECT_EQ(flagcxCommAbort(comm), flagcxInProgress);
  EXPECT_EQ(aborts, 0);
  comm->finalizeState.running = 0;
  abortResult = flagcxInProgress;
  EXPECT_EQ(flagcxCommAbort(comm), flagcxInProgress);
  EXPECT_EQ(comm->homoComm, owner);
  EXPECT_EQ(comm->finalizeState.abortIndex, 0u);
  abortResult = flagcxSuccess;
  EXPECT_EQ(flagcxCommAbort(comm), flagcxSuccess);
  comm = nullptr;
  EXPECT_EQ(aborts, 2);
}

TEST_F(CommFinalizeOwnerTest, UnsupportedAbortKeepsNormalFinalizeAvailable) {
  comm->homoComm = makeOwner();
  abortResult = flagcxNotSupported;
  EXPECT_EQ(flagcxCommAbort(comm), flagcxNotSupported);
  EXPECT_FALSE(comm->finalizeState.abortStarted);
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(finalizes, 1);
}

TEST_F(CommFinalizeOwnerTest, AbortDoesNotInventIpcClosurePermission) {
  comm->homoComm = makeOwner();
  comm->nranks = comm->localRanks = 2;
  comm->ipcClosePolicyConfirmed = comm->internalBuffersNeedPeerClose = true;
  auto *buffer =
      (flagcxInternalBuffer *)calloc(1, sizeof(flagcxInternalBuffer));
  buffer->pointer = (void *)0x456000;
  comm->internalBuffers = buffer;
  EXPECT_EQ(flagcxCommAbort(comm), flagcxInProgress);
  EXPECT_EQ(aborts, 1);
  EXPECT_EQ(frees, 0);
  EXPECT_EQ(comm->finalizeState.commonPhase, flagcxFinalizeDrain);
  EXPECT_NE(comm->homoComm, nullptr);
  // Simulate the common import-close result. No bootstrap is available in this
  // fixture; Abort above must return instead of attempting a peer exchange.
  comm->finalizeState.commonPhase = flagcxFinalizeVendors;
  comm->nranks = comm->localRanks = 1;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(finalizes, 0);
  EXPECT_EQ(frees, 0);
  EXPECT_EQ(flagcxCommAbort(comm), flagcxSuccess);
  comm = nullptr;
  EXPECT_EQ(aborts, 1);
  EXPECT_EQ(frees, 1);
}

TEST_F(CommFinalizeOwnerTest, UnknownProviderNotSupportedNeverDestroysNative) {
  comm->homoComm = makeOwner();
  finalizeResult = flagcxNotSupported;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxNotSupported);
  EXPECT_EQ(destroys, 0);
  EXPECT_EQ(live.size(), 1u);
  EXPECT_EQ(flagcxAllReduce(nullptr, nullptr, 0, flagcxChar, flagcxSum, comm,
                            nullptr),
            flagcxInvalidUsage);
}

TEST_F(CommFinalizeOwnerTest,
       CompatibilityDestroyConsumesEveryAliasExactlyOnce) {
  auto inner = makeOwner();
  comm->allowNativeDestroyFinalize = true;
  comm->homoComm = comm->homoInterComm = comm->tunerInnerComm = inner;
  comm->homoCommMap[TunerCollCategory{}] = inner;
  comm->homoBestCommMap[TunerCollCategory{}] = inner;
  comm->commMap[flagcxCommTag{}] = inner;
  finalizeResult = flagcxNotSupported;
  destroyResult = flagcxInProgress;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxInProgress);
  EXPECT_EQ(comm->homoComm, inner);
  destroyResult = flagcxRemoteError;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxRemoteError);
  destroyResult = flagcxSuccess;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(finalizes, 1);
  EXPECT_EQ(destroys, 3);
  EXPECT_EQ(comm->homoComm, nullptr);
  EXPECT_EQ(comm->homoInterComm, nullptr);
  EXPECT_EQ(comm->tunerInnerComm, nullptr);
  EXPECT_EQ(comm->homoCommMap.begin()->second, nullptr);
  EXPECT_STREQ(flagcxGetLastError(comm), flagcxGetErrorString(flagcxSuccess));
  EXPECT_EQ(lastErrorQueries, 0);
  flagcxResult_t async = flagcxInternalError;
  EXPECT_EQ(flagcxCommGetAsyncError(comm, &async), flagcxSuccess);
  EXPECT_EQ(async, flagcxSuccess);
  EXPECT_EQ(flagcxCommAbort(comm), flagcxSuccess);
  comm = nullptr;
  EXPECT_EQ(destroys, 3);
  EXPECT_EQ(aborts, 0);
}

TEST_F(CommFinalizeOwnerTest, OrdinaryErrorDoesNotTriggerCompatibilityDestroy) {
  comm->homoComm = makeOwner();
  comm->allowNativeDestroyFinalize = true;
  finalizeResult = flagcxRemoteError;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxRemoteError);
  EXPECT_EQ(destroys, 0);
}

TEST_F(CommFinalizeOwnerTest, AllNativeOwnersAndAutomaticHostFallbackFinalize) {
  auto first = makeOwner(), second = makeOwner(), third = makeOwner();
  comm->homoCommMap[TunerCollCategory{}] = second;
  comm->commMap[flagcxCommTag{}] = first;
  comm->homoBestCommMap[TunerCollCategory{}] = second;
  comm->homoComm = first;
  comm->homoInterComm = comm->tunerInnerComm = third;
  comm->hostComm = makeOwner();
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(finalized, (std::vector<void *>{second, first, third}));
  EXPECT_EQ(hostFinalizes, 1);
  EXPECT_EQ(destroys + hostDestroys, 0);
}

TEST_F(CommFinalizeOwnerTest, FreeFailureRetriesLocallyWithoutNativeFinalize) {
  comm->homoComm = makeOwner();
  ASSERT_EQ(
      flagcxCommReleaseBuffer(nullptr, nullptr, flagcxInternalBufferDevice),
      flagcxSuccess);
  comm->internalBuffers =
      (flagcxInternalBuffer *)calloc(1, sizeof(flagcxInternalBuffer));
  comm->internalBuffers->pointer = (void *)0x44000;
  freeResult = flagcxRemoteError;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(frees, 0);
  EXPECT_EQ(flagcxCommDestroy(comm), flagcxRemoteError);
  EXPECT_NE(comm->internalBuffers, nullptr);
  EXPECT_STREQ(flagcxGetLastError(comm),
               flagcxGetErrorString(flagcxRemoteError));
  EXPECT_EQ(lastErrorQueries, 0);
  EXPECT_EQ(finalizes, 1);
  freeResult = flagcxSuccess;
  EXPECT_EQ(flagcxCommDestroy(comm), flagcxSuccess);
  comm = nullptr;
  EXPECT_EQ(frees, 2);
  EXPECT_EQ(finalizes, 1);
  EXPECT_EQ(destroys, 1);
}

TEST_F(CommFinalizeOwnerTest,
       DeferredAllocationRetainsOwnerAndDeduplicatesAliases) {
  comm->finalizeState.lifecycle = flagcxCommFinalizing;
  comm->finalizeState.commonPhase = flagcxFinalizeOwners;
  EXPECT_EQ(flagcxCommReleaseBuffer(comm, (void *)0x44000,
                                    flagcxInternalBufferDevice),
            flagcxSuccess);
  EXPECT_EQ(flagcxCommReleaseBuffer(comm, (void *)0x44000,
                                    flagcxInternalBufferDevice),
            flagcxSuccess);
  EXPECT_EQ(
      flagcxCommReleaseBuffer(comm, (void *)0x44000, flagcxInternalBufferHost),
      flagcxInvalidUsage);
  EXPECT_EQ(frees, 0);
  EXPECT_EQ(flagcxCommFreeInternalBuffers(comm), flagcxInvalidUsage);
  EXPECT_EQ(comm->internalBuffers->next, nullptr);
}

TEST_F(CommFinalizeOwnerTest, ClosingRejectsNewDeviceViewsWithoutAllocating) {
  comm->finalizeState.lifecycle = flagcxCommFinalizing;
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  flagcxDevComm_t devComm = (flagcxDevComm_t)1;
  flagcxDevMem_t devMem = (flagcxDevMem_t)1;
  EXPECT_EQ(flagcxDevCommCreate(comm, &reqs, &devComm), flagcxInvalidUsage);
  EXPECT_EQ(devComm, nullptr);
  EXPECT_EQ(flagcxDevMemCreate(comm, (void *)1, 8, nullptr, &devMem),
            flagcxInvalidUsage);
  EXPECT_EQ(devMem, nullptr);
}

TEST_F(CommFinalizeOwnerTest, ZeroInitializedUntunedMapsAreSafe) {
  // This is the initializer's untuned allocation pattern, without map nodes.
  auto *untuned = (flagcxComm_t)calloc(1, sizeof(flagcxComm));
  untuned->commType = flagcxCommunicatorHomo;
  untuned->nranks = untuned->localRanks = 1;
  untuned->ipcTable = untuned->ipcTableInline;
  untuned->ipcTableCapacity = FLAGCX_MAX_IPC_ENTRIES;
  untuned->homoComm = makeOwner();
  EXPECT_EQ(flagcxCommFinalize(untuned), flagcxSuccess);
  EXPECT_EQ(flagcxCommDestroy(untuned), flagcxSuccess);
  EXPECT_EQ(finalizes, 1);
  EXPECT_EQ(destroys, 1);
}

TEST_F(CommFinalizeOwnerTest,
       EventDestroyFailureRetriesAfterFinalizationLocally) {
  flagcxWorkRecord *record = nullptr;
  ASSERT_EQ(flagcxWorkPrepare(&comm->submittedWork, nullptr, &record),
            flagcxSuccess);
  ASSERT_EQ(flagcxWorkFinish(record, nullptr, flagcxSuccess,
                             flagcxWorkNativeGroup, 0),
            flagcxSuccess);
  ASSERT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  eventFailures = 1;
  EXPECT_EQ(flagcxCommDestroy(comm), flagcxRemoteError);
  EXPECT_EQ(comm->finalizeState.commonPhase, flagcxFinalizeComplete);
  EXPECT_NE(comm->submittedWork.head, nullptr);
  EXPECT_EQ(flagcxCommDestroy(comm), flagcxSuccess);
  comm = nullptr;
  EXPECT_EQ(eventDestroys, 2);
}

TEST_F(CommFinalizeOwnerTest, OwnCompletionFailureKeepsNativeAndExportBacking) {
  comm->homoComm = makeOwner();
  comm->internalBuffers =
      (flagcxInternalBuffer *)calloc(1, sizeof(flagcxInternalBuffer));
  comm->internalBuffers->pointer = (void *)0x44000;
  flagcxWorkRecord *record = nullptr;
  ASSERT_EQ(flagcxWorkPrepare(&comm->submittedWork, nullptr, &record),
            flagcxSuccess);
  ASSERT_EQ(flagcxWorkFinish(record, nullptr, flagcxSuccess,
                             flagcxWorkNativeGroup, 0),
            flagcxSuccess);
  syncFailures = 1;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxRemoteError);
  EXPECT_EQ(comm->finalizeState.commonPhase, flagcxFinalizeDrain);
  EXPECT_NE(comm->internalBuffers, nullptr);
  EXPECT_EQ(finalizes + destroys + frees, 0);
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(finalizes, 1);
  EXPECT_EQ(frees, 0);
}
