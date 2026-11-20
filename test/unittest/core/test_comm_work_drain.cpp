#include "comm_lifecycle.h"
#include "dev_api_backend.h"
#include "global_comm.h"
#include "work_completion.hpp"
#include <cstdlib>

namespace {
class CommWorkDrain : public WorkCompletionTest {
protected:
  inline static CommWorkDrain *active;
  flagcxDevApiBackend *savedBackend = nullptr;
  flagcxDevApiBackend backend = {};
  int freeCalls = 0, freed = 0;
  void *failFree = nullptr;
  flagcxResult_t ownerResult = flagcxSuccess;
  flagcxComm_t owned = nullptr;
  void SetUp() override {
    WorkCompletionTest::SetUp();
    active = this;
    savedBackend = devApiBackend;
    backend = *savedBackend;
    backend.devMemDestroy = [](flagcxComm_t, flagcxDevMem_t) {
      return flagcxSuccess;
    };
    backend.devCommDestroy = [](flagcxComm_t, flagcxDevComm_t) {
      return active->ownerResult;
    };
    devApiBackend = &backend;
    device.deviceMalloc = [](void **output, size_t bytes, flagcxMemType_t,
                             flagcxStream_t) {
      *output = malloc(bytes);
      return *output ? flagcxSuccess : flagcxSystemError;
    };
    device.deviceFree = [](void *pointer, flagcxMemType_t, flagcxStream_t) {
      active->freeCalls++;
      if (pointer == active->failFree)
        return flagcxRemoteError;
      active->freed++;
      free(pointer);
      return flagcxSuccess;
    };
    owned = (flagcxComm_t)calloc(1, sizeof(flagcxComm));
    owned->commType = flagcxCommunicatorHomo;
    owned->deviceOrdinalValid = 1;
  }
  void TearDown() override {
    destroyFailures = waitFailures = 0;
    failFree = nullptr;
    ownerResult = flagcxSuccess;
    if (owned != nullptr) {
      for (auto *node = owned->submittedWork.head; node; node = node->next)
        node->state = flagcxWorkComplete;
      EXPECT_EQ(flagcxCommDestroy(owned), flagcxSuccess);
    }
    devApiBackend = savedBackend;
    WorkCompletionTest::TearDown();
  }
  void attachOwners() {
    auto *devComm = (flagcxDevComm_t)calloc(1, sizeof(flagcxDevCommInternal));
    pthread_mutex_init(&devComm->cachedPtrMutex, nullptr);
    devComm->barrierIpcIndex = devComm->signalIpcSlot = -1;
    devComm->cachedDevicePtr = malloc(8);
    devComm->cachedNetContextsPtr = malloc(8);
    devComm->cachedGridBarrierPtr = malloc(8);
    owned->activeDevComms = devComm;
    auto *devMem = (flagcxDevMem_t)calloc(1, sizeof(flagcxDevMemInternal));
    pthread_mutex_init(&devMem->cachedPtrMutex, nullptr);
    devMem->ipcIndex = -1;
    devMem->cachedDevicePtr = malloc(8);
    devMem->segmentedPeerLayouts = malloc(8);
    owned->activeDevMems = devMem;
  }
  size_t retained() {
    size_t n = 0;
    for (auto *node = owned->internalBuffers; node; node = node->next)
      n++;
    return n;
  }
};
} // namespace

TEST_F(CommWorkDrain, FinalizeDoesNotWaitForAnotherCommunicatorsStream) {
  auto *work = prepare(&owned->submittedWork);
  stream.submitted = stream.completed = 1;
  ASSERT_EQ(finish(work), flagcxSuccess);
  auto *other = prepare();
  otherStream.submitted = 1;
  ASSERT_EQ(finish(other, &otherStream), flagcxSuccess);
  EXPECT_EQ(flagcxCommFinalize(owned), flagcxSuccess);
  EXPECT_EQ(waited, 1);
  EXPECT_EQ(other->state, flagcxWorkRecorded);
  EXPECT_EQ(otherStream.completed, 0u);
}

TEST_F(CommWorkDrain, IncompleteAndFailedWorkWaitKeepExporterUntilRetry) {
  void *exporter = nullptr;
  ASSERT_EQ(flagcxCommAllocInternalBuffer(owned, &exporter, 64,
                                          flagcxInternalBufferDevice),
            flagcxSuccess);
  auto *work = prepare(&owned->submittedWork);
  stream.submitted = 1;
  ASSERT_EQ(finish(work), flagcxSuccess);
  waitFailures = 1;
  EXPECT_EQ(flagcxCommFinalize(owned), flagcxUnhandledDeviceError);
  EXPECT_EQ(owned->finalizeState.commonPhase, flagcxFinalizeDrain);
  EXPECT_EQ(owned->internalBuffers->pointer, exporter);
  EXPECT_EQ(flagcxCommFinalize(owned), flagcxInProgress);
  EXPECT_EQ(flagcxCommDestroy(owned), flagcxInProgress);
  EXPECT_EQ(owned->internalBuffers->pointer, exporter);
  EXPECT_EQ(flagcxCommRequireActive(owned), flagcxInvalidUsage);
  stream.completed = 1;
  EXPECT_EQ(flagcxCommFinalize(owned), flagcxSuccess);
  EXPECT_EQ(owned->internalBuffers->pointer, exporter);
}

TEST_F(CommWorkDrain, EventReleaseFailureIsALocalDestroyRetry) {
  auto *work = prepare(&owned->submittedWork);
  stream.submitted = stream.completed = 1;
  ASSERT_EQ(finish(work), flagcxSuccess);
  ASSERT_EQ(flagcxCommFinalize(owned), flagcxSuccess);
  destroyFailures = 1;
  EXPECT_EQ(flagcxCommDestroy(owned), flagcxSystemError);
  EXPECT_EQ(owned->finalizeState.commonPhase, flagcxFinalizeComplete);
  EXPECT_EQ(owned->destroyPhase, flagcxDestroyBuffers);
  EXPECT_EQ(owned->submittedWork.head, work);
  EXPECT_EQ(destroyed, 0);
  EXPECT_EQ(flagcxCommDestroy(owned), flagcxSuccess);
  owned = nullptr;
  EXPECT_EQ(destroyed, 1);
  EXPECT_EQ(recorded, 1);
}

TEST_F(CommWorkDrain, PartialEventCreateFailureDoesNotPreventDestroy) {
  partialCreateFailures = 1;
  flagcxWorkRecord *record = nullptr;
  EXPECT_EQ(flagcxWorkPrepare(&owned->submittedWork, (flagcxStream_t)&stream, 0,
                              &record),
            flagcxSystemError);
  EXPECT_EQ(record, nullptr);
  EXPECT_EQ(flagcxCommFinalize(owned), flagcxSuccess);
  EXPECT_EQ(flagcxCommDestroy(owned), flagcxSuccess);
  owned = nullptr;
  EXPECT_EQ(created + destroyed, 0);
  EXPECT_EQ(wrappersAllocated, 1);
  EXPECT_EQ(wrappersFreed, 1);
}

TEST_F(CommWorkDrain, LiveOwnersUseDrainProofAndDeferCachedFreesUntilDestroy) {
  attachOwners();
  auto *work = prepare(&owned->submittedWork);
  stream.submitted = 1;
  ASSERT_EQ(finish(work), flagcxSuccess);
  EXPECT_EQ(flagcxCommFinalize(owned), flagcxInProgress);
  EXPECT_NE(owned->activeDevMems, nullptr);
  EXPECT_NE(owned->activeDevComms, nullptr);
  EXPECT_EQ(retained(), 0u);
  stream.completed = 1;
  ASSERT_EQ(flagcxCommFinalize(owned), flagcxSuccess);
  EXPECT_EQ(owned->activeDevMems, nullptr);
  EXPECT_EQ(owned->activeDevComms, nullptr);
  EXPECT_EQ(deviceSyncs, 0);
  EXPECT_EQ(freeCalls, 0); // Includes implicitly synchronizing SDK frees.
  EXPECT_EQ(retained(), 5u);
  failFree = owned->internalBuffers->next->pointer;
  EXPECT_EQ(flagcxCommDestroy(owned), flagcxRemoteError);
  EXPECT_EQ(freed, 1);
  EXPECT_EQ(retained(), 4u);
  failFree = nullptr;
  EXPECT_EQ(flagcxCommDestroy(owned), flagcxSuccess);
  owned = nullptr;
  EXPECT_EQ(freed, 5);
  EXPECT_EQ(freeCalls, 6);
}

TEST_F(CommWorkDrain,
       OwnerRetryKeepsSuccessfulTransfersWithoutDuplicatingThem) {
  attachOwners();
  ownerResult = flagcxInProgress;
  EXPECT_EQ(flagcxCommFinalize(owned), flagcxInProgress);
  EXPECT_EQ(owned->finalizeState.commonPhase, flagcxFinalizeOwners);
  EXPECT_EQ(owned->activeDevMems, nullptr);
  ASSERT_NE(owned->activeDevComms, nullptr);
  EXPECT_FALSE(owned->activeDevComms->backendDestroyed);
  EXPECT_EQ(retained(), 2u);
  EXPECT_EQ(flagcxCommFinalize(owned), flagcxInProgress);
  EXPECT_EQ(retained(), 2u);
  ownerResult = flagcxSuccess;
  EXPECT_EQ(flagcxCommFinalize(owned), flagcxSuccess);
  EXPECT_EQ(retained(), 5u);
  EXPECT_EQ(freeCalls, 0);
  EXPECT_EQ(deviceSyncs, 0);
}
