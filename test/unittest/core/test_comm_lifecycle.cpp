#include "adaptor.h"
#include "comm_lifecycle.h"
#include "device_api/flagcx_device.h"
#include "global_comm.h"
#include "ppu_comm_lifecycle.hpp"
#include "reg_pool.h"
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
struct PpuProviderGuard {
  flagcxCCLAdaptor *saved = cclAdaptors[flagcxCCLAdaptorDevice];
  flagcxCCLAdaptor provider = *saved;
  PpuProviderGuard() {
    provider.commFinalize = [](flagcxInnerComm_t inner) {
      return PpuLifecycleShim::ppupcclAdaptorCommFinalize(
          (PpuLifecycleShim::flagcxInnerComm_t)inner);
    };
    provider.commGetAsyncError = [](flagcxInnerComm_t inner,
                                    flagcxResult_t *status) {
      return PpuLifecycleShim::ppupcclAdaptorCommGetAsyncError(
          (PpuLifecycleShim::flagcxInnerComm_t)inner, status);
    };
    cclAdaptors[flagcxCCLAdaptorDevice] = &provider;
  }
  ~PpuProviderGuard() { cclAdaptors[flagcxCCLAdaptorDevice] = saved; }
};
} // namespace

TEST(CommLifecycleTest, PpuAsyncFinalizeWaitsForNativeDestroy) {
  PpuLifecycleShim::NativeContext native;
  native.finalizeResult = flagcxInProgress;
  native.pendingDestroys = 1;
  PpuLifecycleShim::Wrapper wrapper{&native};
  PpuProviderGuard guard;
  flagcxComm comm = {};
  comm.commType = flagcxCommunicatorHomo;
  comm.homoComm = (flagcxInnerComm_t)&wrapper;
  ASSERT_EQ(flagcxCommFinalize(&comm), flagcxSuccess);
  EXPECT_EQ(comm.finalizeState.commonPhase, flagcxFinalizeComplete);
  EXPECT_EQ(wrapper.base, nullptr);
  EXPECT_EQ(wrapper.frees, 0);
  EXPECT_EQ(native.finalizes, 1);
  EXPECT_EQ(native.queries, 1);
  EXPECT_EQ(native.destroys, 2);
  EXPECT_EQ(PpuLifecycleShim::ppupcclAdaptorCommDestroy(&wrapper),
            flagcxSuccess);
  EXPECT_EQ(wrapper.frees, 1);
}

TEST(CommLifecycleTest, PpuFailedCloseResumesThroughOuterAsyncPoll) {
  PpuLifecycleShim::NativeContext native;
  native.finalizeResult = flagcxInProgress;
  native.destroyResult = flagcxRemoteError;
  PpuLifecycleShim::Wrapper wrapper{&native};
  PpuProviderGuard guard;
  flagcxComm comm = {};
  comm.commType = flagcxCommunicatorHomo;
  comm.homoComm = (flagcxInnerComm_t)&wrapper;
  EXPECT_EQ(flagcxCommFinalize(&comm), flagcxRemoteError);
  EXPECT_NE(comm.finalizeState.commonPhase, flagcxFinalizeComplete);
  EXPECT_EQ(wrapper.base, &native);
  native.destroyResult = flagcxSuccess;
  ASSERT_EQ(flagcxCommFinalize(&comm), flagcxSuccess);
  EXPECT_EQ(wrapper.base, nullptr);
  EXPECT_EQ(native.finalizes, 1);
  EXPECT_EQ(native.queries, 1);
  EXPECT_EQ(native.destroys, 2);
  EXPECT_EQ(PpuLifecycleShim::ppupcclAdaptorCommDestroy(&wrapper),
            flagcxSuccess);
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
} // namespace

TEST(CommLifecycleTest, IndependentCommunicatorsReleaseImmediately) {
  LifecycleProviderGuard guard;
  static int destroys;
  destroys = 0;
  guard.provider.commDestroy = [](flagcxInnerComm_t) {
    destroys++;
    return flagcxSuccess;
  };
  auto *first = makeFinalizedComm(1);
  auto *second = makeFinalizedComm(2);
  EXPECT_EQ(flagcxCommDestroy(first), flagcxSuccess);
  EXPECT_EQ(destroys, 1);
  EXPECT_EQ(flagcxCommDestroy(second), flagcxSuccess);
  EXPECT_EQ(destroys, 2);
}

TEST(CommLifecycleTest, FailedNativeDestroyKeepsOwnerForLocalRetry) {
  LifecycleProviderGuard guard;
  static int calls;
  static flagcxResult_t firstResult;
  guard.provider.commDestroy = [](flagcxInnerComm_t) {
    return ++calls == 1 ? firstResult : flagcxSuccess;
  };
  for (auto result : {flagcxRemoteError, flagcxInProgress}) {
    calls = 0;
    firstResult = result;
    auto *comm = makeFinalizedComm(1);
    EXPECT_EQ(flagcxCommDestroy(comm), result);
    EXPECT_EQ(comm->homoComm, (flagcxInnerComm_t)1);
    EXPECT_EQ(comm->destroyPhase, flagcxDestroyVendors);
    EXPECT_EQ(flagcxCommDestroy(comm), flagcxSuccess);
    EXPECT_EQ(calls, 2);
  }
}

TEST(CommLifecycleTest, GroupedFinalizePreservesCallerTlsAndFifoOrder) {
  LifecycleProviderGuard guard;
  static thread_local int wrapperGuard;
  wrapperGuard = 37;
  static int order[2], count;
  count = 0;
  guard.provider.commFinalize = [](flagcxInnerComm_t inner) {
    EXPECT_EQ(wrapperGuard, 37);
    order[count++] = (int)(uintptr_t)inner;
    return flagcxSuccess;
  };
  flagcxComm first = {}, second = {};
  first.commType = second.commType = flagcxCommunicatorHomo;
  first.homoComm = (flagcxInnerComm_t)1;
  second.homoComm = (flagcxInnerComm_t)2;
  flagcxCommLifecycleGroupStart();
  ASSERT_EQ(flagcxCommFinalizeEnqueue(&first), flagcxSuccess);
  ASSERT_EQ(flagcxCommFinalizeEnqueue(&second), flagcxSuccess);
  ASSERT_EQ(flagcxCommFinalizeEnqueue(&first), flagcxSuccess);
  EXPECT_EQ(count, 0);
  ASSERT_EQ(flagcxCommLifecycleGroupEnd(flagcxSuccess), flagcxSuccess);
  EXPECT_EQ(count, 2);
  EXPECT_EQ(order[0], 1);
  EXPECT_EQ(order[1], 2);
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

TEST(CommLifecycleTest, NativeOnlyFinalizeDoesNotRequireBootstrapPeers) {
  LifecycleProviderGuard guard;
  static int calls;
  calls = 0;
  guard.provider.commFinalize = [](flagcxInnerComm_t) {
    calls++;
    return flagcxSuccess;
  };
  flagcxComm comm = {};
  comm.commType = flagcxCommunicatorHomo;
  comm.homoComm = (flagcxInnerComm_t)1;
  comm.nranks = comm.localRanks = 4;
  // No peer/bootstrap exists. The native SDK owns all communicator resources.
  EXPECT_FALSE(comm.internalBuffersNeedPeerClose);
  EXPECT_EQ(flagcxCommFinalize(&comm), flagcxSuccess);
  EXPECT_EQ(calls, 1);
}

TEST(CommLifecycleTest, StandaloneIpcDeclarationIsStickyAcrossEmptyCalls) {
  flagcxComm comm = {};
  EXPECT_EQ(flagcxCommRequireIpcClose(nullptr, 1), flagcxInvalidArgument);
  EXPECT_EQ(flagcxCommRequireIpcClose(&comm, 0), flagcxSuccess);
  EXPECT_FALSE(comm.internalBuffersNeedPeerClose);
  EXPECT_EQ(flagcxCommRequireIpcClose(&comm, 1), flagcxSuccess);
  EXPECT_TRUE(comm.internalBuffersNeedPeerClose);
  EXPECT_EQ(flagcxCommRequireIpcClose(&comm, 0), flagcxSuccess);
  EXPECT_TRUE(comm.internalBuffersNeedPeerClose);
}

TEST(CommLifecycleTest, NativeOnlyDrainDoesNotSynchronizeTheDevice) {
  auto *saved = deviceAdaptor;
  flagcxDeviceAdaptor provider = *saved;
  deviceAdaptor = &provider;
  static int syncs;
  syncs = 0;
  provider.setDevice = [](int) { return flagcxSuccess; };
  provider.deviceSynchronize = [] {
    syncs++;
    return flagcxUnhandledDeviceError;
  };
  flagcxComm comm = {};
  comm.commType = flagcxCommunicatorHomo;
  comm.deviceOrdinalValid = 1;
  EXPECT_EQ(flagcxCommFinalize(&comm), flagcxSuccess);
  EXPECT_EQ(syncs, 0);
  EXPECT_EQ(flagcxCommRequireActive(&comm), flagcxInvalidUsage);
  EXPECT_EQ(flagcxWorkRelease(&comm.submittedWork), flagcxSuccess);
  deviceAdaptor = saved;
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

TEST(CommLifecycleTest, DeviceWorkDrainsBeforeVendorStopsItsProxy) {
  LifecycleProviderGuard guard;
  static bool nativeProxyAlive;
  nativeProxyAlive = true;
  guard.provider.commFinalize = [](flagcxInnerComm_t) {
    nativeProxyAlive = false;
    return flagcxSuccess;
  };
  auto *saved = deviceAdaptor;
  flagcxDeviceAdaptor provider = *saved;
  deviceAdaptor = &provider;
  struct Restore {
    flagcxDeviceAdaptor *saved;
    ~Restore() { deviceAdaptor = saved; }
  } restore{saved};
  provider.setDevice = [](int) { return flagcxSuccess; };
  provider.deviceSynchronize = [] {
    // A Device API kernel can still need the vendor's proxy even when no
    // vendor host collective is pending on its internal streams.
    return nativeProxyAlive ? flagcxSuccess : flagcxRemoteError;
  };
  flagcxComm comm = {};
  comm.commType = flagcxCommunicatorHomo;
  comm.homoComm = (flagcxInnerComm_t)1;
  comm.deviceOrdinalValid = 1;
  EXPECT_EQ(flagcxCommFinalize(&comm), flagcxSuccess);
  EXPECT_FALSE(nativeProxyAlive);
}

TEST(CommLifecycleTest, NativeFinalizeRunsAfterDependentOwnersAreClosed) {
  LifecycleProviderGuard guard;
  static flagcxComm_t owner;
  guard.provider.commFinalize = [](flagcxInnerComm_t) {
    EXPECT_EQ(owner->activeDevMems, nullptr);
    EXPECT_TRUE(owner->finalizeState.localDoneMask &
                (1u << flagcxFinalizeImports));
    return flagcxSuccess;
  };
  auto *savedDevice = deviceAdaptor;
  flagcxDeviceAdaptor device = *savedDevice;
  device.deviceSynchronize = [] { return flagcxSuccess; };
  deviceAdaptor = &device;
  flagcxComm comm = {};
  owner = &comm;
  comm.commType = flagcxCommunicatorHomo;
  comm.homoComm = (flagcxInnerComm_t)1;
  auto *mem = (flagcxDevMemInternal *)calloc(1, sizeof(flagcxDevMemInternal));
  pthread_mutex_init(&mem->cachedPtrMutex, nullptr);
  mem->ipcIndex = -1;
  comm.activeDevMems = mem;
  EXPECT_EQ(flagcxCommFinalize(&comm), flagcxSuccess);
  deviceAdaptor = savedDevice;
}

TEST(CommLifecycleTest, VendorFailureRetainsCompletedCleanupForRetry) {
  LifecycleProviderGuard guard;
  static int calls;
  calls = 0;
  guard.provider.commFinalize = [](flagcxInnerComm_t) {
    return ++calls == 1 ? flagcxRemoteError : flagcxSuccess;
  };
  flagcxComm comm = {};
  comm.commType = flagcxCommunicatorHomo;
  comm.homoComm = (flagcxInnerComm_t)1;
  comm.rank = 2;
  comm.nranks = 4;
  comm.deviceOrdinalValid = 1;
  comm.deviceOrdinal = 3;
  auto *savedDevice = deviceAdaptor;
  flagcxDeviceAdaptor device = *savedDevice;
  device.setDevice = [](int) { return flagcxSuccess; };
  device.deviceSynchronize = [] { return flagcxSuccess; };
  deviceAdaptor = &device;
  EXPECT_EQ(flagcxCommFinalize(&comm), flagcxRemoteError);
  EXPECT_EQ(comm.finalizeState.commonPhase, flagcxFinalizeVendors);
  EXPECT_TRUE(comm.finalizeState.localDoneMask & (1u << flagcxFinalizeImports));
  EXPECT_EQ(flagcxCommFinalize(&comm), flagcxSuccess);
  int value;
  EXPECT_EQ(flagcxCommCount(&comm, &value), flagcxSuccess);
  EXPECT_EQ(value, 4);
  EXPECT_EQ(flagcxCommUserRank(&comm, &value), flagcxSuccess);
  EXPECT_EQ(value, 2);
  EXPECT_EQ(flagcxCommGetDeviceNumber(&comm, &value), flagcxSuccess);
  EXPECT_EQ(value, 3);
  deviceAdaptor = savedDevice;
}

class NativeRegistrationRetry
    : public ::testing::TestWithParam<flagcxResult_t> {};
TEST_P(NativeRegistrationRetry, ClosesBeforeNativeContext) {
  LifecycleProviderGuard guard;
  static int deregisters, finalizers, destructors;
  static flagcxResult_t firstResult;
  firstResult = GetParam();
  deregisters = finalizers = destructors = 0;
  guard.provider.commDeregister = [](flagcxInnerComm_t, void *) {
    return ++deregisters == 1 ? firstResult : flagcxSuccess;
  };
  guard.provider.commFinalize = [](flagcxInnerComm_t) {
    EXPECT_EQ(deregisters, 2);
    finalizers++;
    return flagcxSuccess;
  };
  guard.provider.commDestroy = [](flagcxInnerComm_t) {
    destructors++;
    return flagcxSuccess;
  };
  auto *comm = (flagcxComm_t)calloc(1, sizeof(flagcxComm));
  comm->commType = flagcxCommunicatorHomo;
  comm->homoComm = (flagcxInnerComm_t)0x5533;
  flagcxRegSnapshot snapshot = {};
  void *data = (void *)0x990000;
  ASSERT_EQ(
      globalRegPool.registerAndAcquire(comm->homoComm, data, 64, &snapshot),
      flagcxSuccess);
  void *handle = nullptr;
  int create = 0;
  ASSERT_EQ(globalRegPool.beginHomoHandle(comm->homoComm, snapshot.item,
                                          &handle, &create),
            flagcxSuccess);
  ASSERT_EQ(create, 1);
  ASSERT_EQ(globalRegPool.finishHomoHandle(comm->homoComm, snapshot.item,
                                           (void *)1, flagcxSuccess),
            flagcxSuccess);
  globalRegPool.releaseItem(&snapshot);
  EXPECT_EQ(flagcxCommFinalize(comm), firstResult);
  EXPECT_EQ(comm->finalizeState.commonPhase, flagcxFinalizeOwners);
  EXPECT_FALSE(comm->finalizeState.localDoneMask &
               (1u << flagcxFinalizeOwners));
  EXPECT_EQ(comm->homoComm, (flagcxInnerComm_t)0x5533);
  EXPECT_EQ(globalRegPool.releaseOwnerReferences(comm->homoComm),
            flagcxInvalidUsage);
  EXPECT_EQ(finalizers + destructors, 0);
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(flagcxCommDestroy(comm), flagcxSuccess);
  EXPECT_EQ(deregisters, 2);
  EXPECT_EQ(finalizers, 1);
  EXPECT_EQ(destructors, 1);
  ASSERT_EQ(globalRegPool.acquireItem(nullptr, data, &snapshot), flagcxSuccess);
  EXPECT_EQ(snapshot.item, nullptr);
  globalRegPool.releaseItem(&snapshot);
}

TEST(CommLifecycleTest, ConservativeRequiredBitCannotBypassPolicyAgreement) {
  flagcxComm comm = {};
  bootstrapState unavailable = {};
  unavailable.mode = FLAGCX_BOOTSTRAP_P2P;
  int ranks[] = {0, 1};
  comm.bootstrap = &unavailable;
  comm.nranks = comm.localRanks = 2;
  comm.localRankToRank = ranks;
  comm.internalBuffersNeedPeerClose = true;
  EXPECT_EQ(flagcxCommRequireIpcClose(&comm, 1), flagcxInvalidArgument);
  EXPECT_TRUE(comm.internalBuffersNeedPeerClose);
  EXPECT_FALSE(comm.ipcClosePolicyConfirmed);
  EXPECT_EQ(flagcxCommRequireIpcClose(&comm, 1), flagcxInvalidArgument);
  comm.ipcClosePolicyConfirmed = true;
  EXPECT_EQ(flagcxCommRequireIpcClose(&comm, 1), flagcxSuccess);
  EXPECT_EQ(flagcxCommRequireIpcClose(&comm, 0), flagcxSuccess);
}

TEST(CommLifecycleTest, ConfirmedNativeOnlyPolicyMustAgreeBeforeFirstExport) {
  flagcxComm comm = {};
  bootstrapState unavailable = {};
  unavailable.mode = FLAGCX_BOOTSTRAP_P2P;
  int ranks[] = {0, 1};
  comm.bootstrap = &unavailable;
  comm.nranks = comm.localRanks = 2;
  comm.localRankToRank = ranks;
  comm.ipcClosePolicyConfirmed = true;
  EXPECT_FALSE(comm.internalBuffersNeedPeerClose);
  EXPECT_EQ(flagcxCommRequireIpcClose(&comm, 1), flagcxInvalidArgument);
  EXPECT_FALSE(comm.ipcClosePolicyConfirmed);
}

INSTANTIATE_TEST_SUITE_P(Provider, NativeRegistrationRetry,
                         ::testing::Values(flagcxRemoteError,
                                           flagcxInProgress));

TEST(CommLifecycleTest, NativeRegistrationRetryKeepsSuccessfulPrefix) {
  LifecycleProviderGuard guard;
  static int calls, successful[3], finalizers;
  calls = finalizers = 0;
  memset(successful, 0, sizeof(successful));
  guard.provider.commDeregister = [](flagcxInnerComm_t, void *handle) {
    int index = (uintptr_t)handle;
    EXPECT_EQ(successful[index], 0);
    if (++calls == 2)
      return flagcxInProgress;
    successful[index]++;
    return flagcxSuccess;
  };
  guard.provider.commFinalize = [](flagcxInnerComm_t) {
    EXPECT_EQ(successful[1] + successful[2], 2);
    finalizers++;
    return flagcxSuccess;
  };
  guard.provider.commDestroy = [](flagcxInnerComm_t) { return flagcxSuccess; };
  auto *comm = makeFinalizedComm(0x5534);
  comm->finalizeState = {};
  for (int i = 1; i <= 2; i++) {
    flagcxRegSnapshot snapshot = {};
    ASSERT_EQ(globalRegPool.registerAndAcquire(
                  comm->homoComm, (void *)(uintptr_t)(0x991000 + i * 0x1000),
                  64, &snapshot),
              flagcxSuccess);
    void *handle = nullptr;
    int create = 0;
    ASSERT_EQ(globalRegPool.beginHomoHandle(comm->homoComm, snapshot.item,
                                            &handle, &create),
              flagcxSuccess);
    ASSERT_EQ(create, 1);
    ASSERT_EQ(globalRegPool.finishHomoHandle(comm->homoComm, snapshot.item,
                                             (void *)(uintptr_t)i,
                                             flagcxSuccess),
              flagcxSuccess);
    globalRegPool.releaseItem(&snapshot);
  }
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxInProgress);
  EXPECT_EQ(calls, 2);
  EXPECT_EQ(finalizers, 0);
  EXPECT_EQ(successful[1] + successful[2], 1);
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(calls, 3);
  EXPECT_EQ(finalizers, 1);
  EXPECT_EQ(successful[1], 1);
  EXPECT_EQ(successful[2], 1);
  EXPECT_EQ(flagcxCommDestroy(comm), flagcxSuccess);
}

TEST(CommLifecycleTest, PendingNativeOwnerReferencesPreventNativeDestroy) {
  LifecycleProviderGuard guard;
  static int destroys;
  destroys = 0;
  guard.provider.commDestroy = [](flagcxInnerComm_t) {
    destroys++;
    return flagcxSuccess;
  };
  auto *comm = makeFinalizedComm(0x5535);
  void *buffer = (void *)0x995000;
  ASSERT_EQ(globalRegPool.registerBuffer(comm->homoComm, buffer, 64),
            flagcxSuccess);
  auto *item = globalRegPool.getItem(comm->homoComm, buffer);
  ASSERT_NE(item, nullptr);
  ASSERT_NE(item->owners, nullptr);
  item->owners->closing = 1;
  EXPECT_EQ(flagcxCommDestroy(comm), flagcxInProgress);
  EXPECT_EQ(comm->homoComm, (flagcxInnerComm_t)0x5535);
  EXPECT_EQ(comm->destroyPhase, flagcxDestroyVendors);
  EXPECT_EQ(destroys, 0);
  item->owners->closing = 0;
  EXPECT_EQ(flagcxCommDestroy(comm), flagcxSuccess);
  EXPECT_EQ(destroys, 1);
}

TEST(CommLifecycleTest, PendingHostDestroyRetainsAutomaticFallbackOwner) {
  auto *saved = cclAdaptors[flagcxCCLAdaptorHost];
  flagcxCCLAdaptor provider = *saved;
  cclAdaptors[flagcxCCLAdaptorHost] = &provider;
  struct Restore {
    flagcxCCLAdaptor *saved;
    ~Restore() { cclAdaptors[flagcxCCLAdaptorHost] = saved; }
  } restore{saved};
  static int calls;
  calls = 0;
  provider.commDestroy = [](flagcxInnerComm_t) {
    return ++calls == 1 ? flagcxInProgress : flagcxSuccess;
  };
  auto *comm = makeFinalizedComm(0);
  comm->commType = flagcxCommunicatorHybrid;
  comm->hasSingleRankHomoComm = 1;
  comm->hostComm = (flagcxInnerComm_t)0x5599;
  EXPECT_EQ(flagcxCommDestroy(comm), flagcxInProgress);
  ASSERT_EQ(comm->hostComm, (flagcxInnerComm_t)0x5599);
  EXPECT_EQ(comm->destroyPhase, flagcxDestroyHost);
  EXPECT_EQ(flagcxCommDestroy(comm), flagcxSuccess);
  EXPECT_EQ(calls, 2);
}
