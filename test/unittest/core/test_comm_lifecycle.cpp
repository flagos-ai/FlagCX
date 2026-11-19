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

namespace {
struct LifecycleProviderGuard {
  flagcxCCLAdaptor *saved = cclAdaptors[flagcxCCLAdaptorDevice];
  flagcxCCLAdaptor provider = *saved;
  LifecycleProviderGuard() { cclAdaptors[flagcxCCLAdaptorDevice] = &provider; }
  ~LifecycleProviderGuard() { cclAdaptors[flagcxCCLAdaptorDevice] = saved; }
};
flagcxComm_t makeFinalizedComm(uintptr_t token) {
  flagcxComm_t comm = (flagcxComm_t)calloc(1, sizeof(flagcxComm));
  comm->commType = flagcxCommunicatorHomo;
  comm->homoComm = (flagcxInnerComm_t)token;
  comm->finalizeState.lifecycle = flagcxCommFinalized;
  comm->finalizeState.commonPhase = flagcxFinalizeComplete;
  return comm;
}
static std::atomic<int> lifecycleCalls;
static int createCalls;
static int failSecondCreate(pthread_t *thread, const pthread_attr_t *attr,
                            void *(*fn)(void *), void *arg) {
  return ++createCalls == 2 ? EAGAIN : pthread_create(thread, attr, fn, arg);
}
static flagcxResult_t countLifecycleJob(flagcxComm_t) {
  lifecycleCalls++;
  return flagcxSuccess;
}
} // namespace

TEST(CommLifecycleTest, WorkerCreationFailureCancelsBeforeAnyCollective) {
  flagcxComm storage[3] = {};
  flagcxComm_t members[] = {&storage[0], &storage[1], &storage[2]};
  lifecycleCalls = createCalls = 0;
  EXPECT_EQ(flagcxCommRunLifecycleJobs(members, 3, countLifecycleJob,
                                       failSecondCreate),
            flagcxSystemError);
  EXPECT_EQ(lifecycleCalls, 0);
  EXPECT_EQ(flagcxCommRunLifecycleJobs(members, 3, countLifecycleJob, nullptr),
            flagcxSuccess);
  EXPECT_EQ(lifecycleCalls, 3);
}

TEST(CommLifecycleTest, SequentialDestroyRetainsCliqueUntilLastRequest) {
  LifecycleProviderGuard guard;
  static int destroys;
  destroys = 0;
  guard.provider.commDestroy = [](flagcxInnerComm_t) {
    destroys++;
    return flagcxSuccess;
  };
  flagcxComm_t members[] = {makeFinalizedComm(1), makeFinalizedComm(2)};
  ASSERT_EQ(flagcxCommMakeIntraProcGroup(members, 2), flagcxSuccess);
  EXPECT_EQ(flagcxCommDestroy(members[0]), flagcxSuccess);
  EXPECT_EQ(destroys, 0);
  EXPECT_EQ(flagcxCommRequireActive(members[0]), flagcxInvalidUsage);
  EXPECT_EQ(flagcxCommDestroy(members[0]), flagcxInvalidUsage);
  EXPECT_EQ(members[1]->intraLifecycle->destroyRequested, 1u);
  EXPECT_EQ(flagcxCommDestroy(members[1]), flagcxSuccess);
  EXPECT_EQ(destroys, 2);
}

TEST(CommLifecycleTest, FailedLastDestroyRetainsCallerAndSuccessfulPrefix) {
  LifecycleProviderGuard guard;
  static int calls[4];
  static bool fail;
  memset(calls, 0, sizeof(calls));
  fail = true;
  guard.provider.commDestroy = [](flagcxInnerComm_t inner) {
    uintptr_t id = (uintptr_t)inner;
    calls[id]++;
    return id == 2 && fail ? flagcxUnhandledDeviceError : flagcxSuccess;
  };
  flagcxComm_t members[] = {makeFinalizedComm(1), makeFinalizedComm(2),
                            makeFinalizedComm(3)};
  ASSERT_EQ(flagcxCommMakeIntraProcGroup(members, 3), flagcxSuccess);
  EXPECT_EQ(flagcxCommDestroy(members[0]), flagcxSuccess);
  EXPECT_EQ(flagcxCommDestroy(members[1]), flagcxSuccess);
  EXPECT_EQ(flagcxCommDestroy(members[2]), flagcxUnhandledDeviceError);
  EXPECT_EQ(calls[1], 1);
  EXPECT_EQ(calls[2], 1);
  EXPECT_EQ(calls[3], 0);
  EXPECT_EQ(members[2]->intraLifecycle->members, members[1]);
  EXPECT_EQ(members[2]->intraLifecycle->retryOwner, members[2]);
  fail = false;
  EXPECT_EQ(flagcxCommDestroy(members[2]), flagcxSuccess);
  EXPECT_EQ(calls[1], 1);
  EXPECT_EQ(calls[2], 2);
  EXPECT_EQ(calls[3], 1);
}

TEST(CommLifecycleTest, IndependentCliquesDoNotShareDestroyRequests) {
  flagcxComm_t a[] = {makeFinalizedComm(0), makeFinalizedComm(0)};
  flagcxComm_t b[] = {makeFinalizedComm(0), makeFinalizedComm(0)};
  ASSERT_EQ(flagcxCommMakeIntraProcGroup(a, 2), flagcxSuccess);
  ASSERT_EQ(flagcxCommMakeIntraProcGroup(b, 2), flagcxSuccess);
  EXPECT_EQ(flagcxCommDestroy(a[0]), flagcxSuccess);
  EXPECT_EQ(b[0]->intraLifecycle->destroyRequested, 0u);
  EXPECT_EQ(flagcxCommDestroy(b[0]), flagcxSuccess);
  EXPECT_EQ(flagcxCommDestroy(a[1]), flagcxSuccess);
  EXPECT_EQ(b[1]->intraLifecycle->destroyRequested, 1u);
  EXPECT_EQ(flagcxCommDestroy(b[1]), flagcxSuccess);
}

TEST(CommLifecycleTest,
     NestedFinalizeGroupDefersDeduplicatesAndClosesAdmission) {
  flagcxComm a = {}, b = {};
  a.commType = b.commType = flagcxCommunicatorHomo;
  flagcxCommLifecycleGroupStart();
  flagcxCommLifecycleGroupStart();
  EXPECT_EQ(flagcxCommFinalizeEnqueue(&a), flagcxSuccess);
  EXPECT_EQ(flagcxCommFinalizeEnqueue(&a), flagcxSuccess);
  EXPECT_EQ(flagcxCommFinalizeEnqueue(&b), flagcxSuccess);
  EXPECT_EQ(flagcxCommRequireActive(&a), flagcxInvalidUsage);
  EXPECT_EQ(a.finalizeState.commonPhase, flagcxFinalizeDrain);
  EXPECT_EQ(flagcxCommLifecycleGroupEnd(flagcxSuccess), flagcxSuccess);
  EXPECT_EQ(a.finalizeState.commonPhase, flagcxFinalizeDrain);
  EXPECT_EQ(flagcxCommLifecycleGroupEnd(flagcxSuccess), flagcxSuccess);
  EXPECT_EQ(a.finalizeState.commonPhase, flagcxFinalizeComplete);
  EXPECT_EQ(b.finalizeState.commonPhase, flagcxFinalizeComplete);
  EXPECT_EQ(flagcxCommLifecycleGroupDepth(), 0);
}

TEST(CommLifecycleTest, GroupErrorRetainsFinalizationForExplicitRetry) {
  flagcxComm comm = {};
  comm.commType = flagcxCommunicatorHomo;
  flagcxCommLifecycleGroupStart();
  EXPECT_EQ(flagcxCommFinalizeEnqueue(&comm), flagcxSuccess);
  EXPECT_EQ(flagcxCommLifecycleGroupEnd(flagcxRemoteError), flagcxRemoteError);
  EXPECT_EQ(comm.finalizeState.commonPhase, flagcxFinalizeDrain);
  EXPECT_EQ(comm.finalizeState.lastResult, flagcxRemoteError);
  EXPECT_EQ(flagcxCommFinalize(&comm), flagcxSuccess);
}

TEST(CommLifecycleTest,
     DeviceDrainFailureClosesAdmissionAndRetriesBeforeImports) {
  auto *saved = deviceAdaptor;
  flagcxDeviceAdaptor provider = *saved;
  deviceAdaptor = &provider;
  struct Restore {
    flagcxDeviceAdaptor *saved;
    ~Restore() { deviceAdaptor = saved; }
  } restore{saved};
  static int syncs;
  syncs = 0;
  provider.setDevice = [](int) { return flagcxSuccess; };
  provider.deviceSynchronize = [] {
    return ++syncs == 1 ? flagcxUnhandledDeviceError : flagcxSuccess;
  };
  flagcxComm comm = {};
  comm.commType = flagcxCommunicatorHomo;
  comm.deviceOrdinalValid = 1;
  EXPECT_EQ(flagcxCommFinalize(&comm), flagcxUnhandledDeviceError);
  EXPECT_EQ(comm.finalizeState.commonPhase, flagcxFinalizeDrain);
  EXPECT_EQ(flagcxCommRequireActive(&comm), flagcxInvalidUsage);
  EXPECT_EQ(flagcxCommFinalize(&comm), flagcxSuccess);
  EXPECT_EQ(syncs, 2);
}

TEST(CommLifecycleTest, TunerContextsFinalizeAndFreeOnceAcrossAllAliases) {
  LifecycleProviderGuard guard;
  static int finalized[4], destroyed[4];
  memset(finalized, 0, sizeof(finalized));
  memset(destroyed, 0, sizeof(destroyed));
  guard.provider.commFinalize = [](flagcxInnerComm_t inner) {
    finalized[(uintptr_t)inner]++;
    return flagcxSuccess;
  };
  guard.provider.commDestroy = [](flagcxInnerComm_t inner) {
    destroyed[(uintptr_t)inner]++;
    return flagcxSuccess;
  };
  flagcxTuner tuner = {};
  tuner.destroy = [](void *) { return flagcxSuccess; };
  flagcxComm_t comm = (flagcxComm_t)calloc(1, sizeof(flagcxComm));
  comm->commType = flagcxCommunicatorHomo;
  comm->tuner = &tuner;
  comm->commMap.clear();
  comm->homoCommMap.clear();
  comm->homoBestCommMap.clear();
  flagcxCommTag tag = {};
  TunerCollCategory category = {};
  comm->commMap[tag] = (flagcxInnerComm_t)1;
  comm->homoCommMap[category] = (flagcxInnerComm_t)2;
  comm->homoBestCommMap[category] = (flagcxInnerComm_t)2;
  comm->homoComm = (flagcxInnerComm_t)1;
  comm->homoInterComm = (flagcxInnerComm_t)2;
  comm->tunerInnerComm = (flagcxInnerComm_t)3;
  comm->clusterInterRankList.resize(2);
  comm->clusterInterRankList[0].push_back(1);
  comm->clusterVendorMap.resize(2);
  ASSERT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  for (int i = 1; i <= 3; i++)
    EXPECT_EQ(finalized[i], 1);
  ASSERT_EQ(flagcxCommDestroy(comm), flagcxSuccess);
  for (int i = 1; i <= 3; i++)
    EXPECT_EQ(destroyed[i], 1);
}
