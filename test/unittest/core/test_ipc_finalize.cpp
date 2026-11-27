#include "ipc_lifecycle_mock.hpp"
#include "launch_kernel.h"
#include "reg_pool.h"
#include "sym_heap.h"

TEST_F(IpcLifecycleMock, FinalizeClosesImportsAndDestroyAloneFreesBacking) {
  addBackingAndImport();
  comm->homoComm = (flagcxInnerComm_t)0x9000;
  ASSERT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_TRUE(comm->finalized);
  EXPECT_TRUE(comm->internalImportsConfirmed);
  EXPECT_EQ(closes.size(), 1u);
  EXPECT_TRUE(frees.empty());
  EXPECT_EQ(destroyCalls, 0);
  ASSERT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(finalizeCalls, 1);
  destroySuccessfully();
  EXPECT_EQ(frees.size(), 2u);
  EXPECT_EQ(destroyCalls, 1);
  EXPECT_EQ(syncs, 0);
}

TEST_F(IpcLifecycleMock, FailedImportKeepsAdmissionClosedAndOwnerRetryable) {
  addBackingAndImport();
  failImport = true;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxRemoteError);
  EXPECT_FALSE(comm->finalized);
  EXPECT_FALSE(comm->internalImportsConfirmed);
  EXPECT_TRUE(frees.empty());
  EXPECT_EQ(flagcxAllGather(nullptr, nullptr, 0, flagcxFloat, comm, nullptr),
            flagcxInvalidUsage);
  EXPECT_EQ(flagcxGroupStart(comm), flagcxInvalidUsage);
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  flagcxDevComm_t view = nullptr;
  EXPECT_EQ(flagcxDevCommCreate(comm, &reqs, &view), flagcxInvalidUsage);
  EXPECT_EQ(view, nullptr);
  failImport = false;
  ASSERT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(closes.size(), 2u);
}

TEST_F(IpcLifecycleMock, FreeFailureRetriesWithoutRepeatingFinalize) {
  addBackingAndImport();
  comm->homoComm = (flagcxInnerComm_t)0x9000;
  ASSERT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  failFree = true;
  EXPECT_EQ(flagcxCommDestroy(comm), flagcxRemoteError);
  EXPECT_NE(comm->internalBuffers, nullptr);
  EXPECT_EQ(finalizeCalls, 1);
  EXPECT_EQ(destroyCalls, 0);
  failFree = false;
  destroySuccessfully();
  EXPECT_EQ(finalizeCalls, 1);
  EXPECT_EQ(closes.size(), 1u);
}

TEST_F(IpcLifecycleMock, EveryNativeAliasIsFinalizedAndDestroyedOnce) {
  auto first = (flagcxInnerComm_t)0x9000, second = (flagcxInnerComm_t)0xa000;
  comm->homoComm = comm->tunerInnerComm = first;
  comm->homoCommMap[TunerCollCategory{}] = first;
  comm->commMap[flagcxCommTag{}] = second;
  comm->homoInterComm = second;
  comm->hostComm = (flagcxInnerComm_t)0xb000;
  ASSERT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(finalizeCalls, 3);
  destroySuccessfully();
  EXPECT_EQ(destroyCalls, 3);
}

TEST_F(IpcLifecycleMock, AcceptedNativeFinalizeIsPolledWithoutResubmission) {
  comm->homoComm = (flagcxInnerComm_t)0x9000;
  nativeResult = flagcxInProgress;
  pendingPolls = 1;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxInProgress);
  EXPECT_FALSE(comm->finalized);
  EXPECT_EQ(finalizeCalls, 1);
  ASSERT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(finalizeCalls, 1);
  EXPECT_EQ(asyncPolls, 2);
}

TEST_F(IpcLifecycleMock, SynchronousNativeFinalizeDoesNotRequireAsyncPoller) {
  comm->hostComm = (flagcxInnerComm_t)0x9000;
  host.commGetAsyncError = nullptr;
  ASSERT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(finalizeCalls, 1);
  EXPECT_EQ(asyncPolls, 0);
  destroySuccessfully();
  EXPECT_EQ(destroyCalls, 1);
}

TEST_F(IpcLifecycleMock, NativePendingWithoutPollerNeverInventsCompletion) {
  comm->homoComm = (flagcxInnerComm_t)0x9000;
  nativeResult = flagcxInProgress;
  sdk.commGetAsyncError = nullptr;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxNotSupported);
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxNotSupported);
  EXPECT_FALSE(comm->finalized);
  EXPECT_EQ(finalizeCalls, 1);
  EXPECT_EQ(destroyCalls, 0);
}

TEST_F(IpcLifecycleMock,
       ProxyCloseFailureAndLocalFreeFailureKeepTransportOwner) {
  addTransport();
  failTransportClose = true;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxRemoteError);
  EXPECT_EQ(transportCloses, 1);
  EXPECT_EQ(transportFrees, 0);
  failTransportClose = false;
  ASSERT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(transportCloses, 2);
  failTransportFree = true;
  EXPECT_EQ(flagcxCommDestroy(comm), flagcxRemoteError);
  EXPECT_NE(comm->heteroComm, nullptr);
  EXPECT_EQ(transportCloses, 2);
  failTransportFree = false;
  destroySuccessfully();
  EXPECT_EQ(transportCloses, 2);
  EXPECT_EQ(transportFrees, 2);
}

TEST_F(IpcLifecycleMock, DeviceSelectionFailureDoesNotStartImportClosure) {
  addBackingAndImport();
  failDevice = true;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxRemoteError);
  EXPECT_TRUE(closes.empty());
  EXPECT_EQ(comm->finalizePhase, 0);
  failDevice = false;
  ASSERT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
}

TEST_F(IpcLifecycleMock, PendingPrivateScratchFreeKeepsChildForLocalRetry) {
  addTransport();
  auto &state = comm->heteroComm->proxyState->uniRunnerState;
  state.groupedArScratch = calloc(1, 64);
  ASSERT_NE(state.groupedArScratch, nullptr);
  ASSERT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  failFree = true;
  freeError = flagcxInProgress;
  EXPECT_EQ(flagcxCommDestroy(comm), flagcxInProgress);
  EXPECT_NE(comm->heteroComm, nullptr);
  EXPECT_NE(state.groupedArScratch, nullptr);
  failFree = false;
  destroySuccessfully();
  EXPECT_EQ(frees.size(), 1u);
  EXPECT_EQ(transportFrees, 1);
}

TEST_F(IpcLifecycleMock, CancelledTransportRetiresBeforeAnyImportsClose) {
  addBackingAndImport();
  addTransport();
  auto *state = comm->heteroComm->proxyState;
  auto backing = comm->ipcTable[0].basePtr;
  state->asyncResult = flagcxRemoteError;
  state->pendingTransportOps = 1; // Includes deferred collective retirement.
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxInProgress);
  EXPECT_EQ(comm->finalizePhase, 0);
  EXPECT_FALSE(comm->finalized);
  EXPECT_EQ(backendCleanups, 0);
  EXPECT_EQ(transportCloses, 0);
  EXPECT_TRUE(closes.empty());
  EXPECT_TRUE(frees.empty());
  state->pendingTransportOps = 0;
  ASSERT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(closes.size(), 1u);
  for (auto pointer : frees)
    EXPECT_NE(pointer, backing);
  destroySuccessfully();
  EXPECT_EQ(transportFrees, 1);
}

TEST_F(IpcLifecycleMock, HealthyTransportAlsoRetiresBeforeRegistrationCleanup) {
  addTransport();
  comm->heteroComm->proxyState->pendingTransportOps = 1;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxInProgress);
  EXPECT_EQ(backendCleanups, 0);
  EXPECT_EQ(transportCloses, 0);
  comm->heteroComm->proxyState->pendingTransportOps = 0;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
}

TEST_F(IpcLifecycleMock,
       PublicDeregistrationRetainsOwnersDuringTransportDrain) {
  addTransport();
  flagcxWindow window = {};
  window.ownerComm = comm;
  window.vendorBase = (flagcxInnerWindow_t)0x1000;
  auto *state = comm->heteroComm->proxyState;
  state->asyncResult = flagcxRemoteError;
  state->pendingTransportOps = 1;
  EXPECT_EQ(flagcxCommWindowDeregister(comm, &window, flagcxMemCCL),
            flagcxInProgress);
  EXPECT_EQ(window.vendorBase, (flagcxInnerWindow_t)0x1000);
  EXPECT_TRUE(window.cleanupStarted);
  EXPECT_EQ(flagcxCommDeregister(comm, (void *)0x2000, flagcxMemCCL),
            flagcxInProgress);
  EXPECT_EQ(backendCleanups, 0);
  EXPECT_TRUE(closes.empty());
  state->pendingTransportOps = 0;
}

TEST_F(IpcLifecycleMock, NativeRegistrationPreparationPrecedesSdkClose) {
  comm->homoComm = (flagcxInnerComm_t)0x9000;
  comm->nativeDestroyFinalize = true;
  void *data = (void *)0x98765000;
  ASSERT_EQ(globalRegPool.registerBuffer(nullptr, data, 64), flagcxSuccess);
  auto *token = globalRegPool.getItem(nullptr, data);
  std::unique_ptr<flagcxRegItem, void (*)(flagcxRegItem *)> reader(
      globalRegPool.acquireItem(nullptr, data),
      [](flagcxRegItem *item) { globalRegPool.releaseItem(item); });
  ASSERT_NE(reader, nullptr);
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxInProgress);
  EXPECT_EQ(comm->finalizePhase, 6);
  EXPECT_TRUE(comm->nativeCatalogReady);
  EXPECT_FALSE(comm->nativeOwners[0].registrationsReleased);
  EXPECT_EQ(finalizeCalls, 0);
  EXPECT_EQ(destroyCalls, 0);
  reader.reset();
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(destroyCalls, 1);
  EXPECT_EQ(comm->homoComm, nullptr);
  EXPECT_EQ(globalRegPool.deregisterBuffer(nullptr, token), flagcxSuccess);
  destroySuccessfully();
}

TEST_F(IpcLifecycleMock,
       StandaloneFinalizeDrainsRejectedOwnersBeforeReadiness) {
  addTransport();
  auto *child = comm->heteroComm;
  child->ownerComm = nullptr;
  auto *op = new flagcxProxyOp{};
  op->args.semaphore = std::make_shared<flagcxHostSemaphore>();
  op->args.semaphore->cancelNotifications();
  child->rejectedGroupOps = op;
  EXPECT_EQ(flagcxHeteroCommFinalize(child), flagcxSuccess);
  EXPECT_EQ(child->rejectedGroupOps, nullptr);
  EXPECT_TRUE(child->importsClosedConfirmed);
  EXPECT_TRUE(child->closeStarted);
  EXPECT_EQ(flagcxHeteroCommFinalize(child), flagcxSuccess);
}

TEST_F(IpcLifecycleMock, StandalonePartialFinalizeRejectsNewP2pSubmissions) {
  addTransport();
  auto *child = comm->heteroComm;
  child->ownerComm = nullptr;
  child->proxyState->pendingTransportOps = 1;
  ASSERT_EQ(flagcxHeteroCommFinalize(child), flagcxInProgress);
  EXPECT_TRUE(child->closeStarted);
  int buffer = 0;
  EXPECT_EQ(flagcxHeteroSend(&buffer, 1, flagcxInt, 0, child, nullptr),
            flagcxInvalidUsage);
  EXPECT_EQ(flagcxHeteroRecv(&buffer, 1, flagcxInt, 0, child, nullptr),
            flagcxInvalidUsage);
  EXPECT_EQ(flagcxGetGroupDepth(), 0);
  EXPECT_EQ(flagcxHeteroPut(child, 0, 0, 0, 1, 0, 0), flagcxInvalidUsage);
  EXPECT_EQ(flagcxHeteroGet(child, 0, 0, 0, 1, 0, 0), flagcxInvalidUsage);
  device.eventDestroy = [](flagcxEvent_t event) {
    EXPECT_EQ(event, (flagcxEvent_t)0x3000);
    ++active->backendCleanups;
    return flagcxSuccess;
  };
  EXPECT_EQ(
      flagcxHeteroGet(child, 0, 0, 0, 1, 0, 0, 0, false, (flagcxEvent_t)0x3000),
      flagcxInvalidUsage);
  EXPECT_EQ(backendCleanups, 1); // Rejected Get still consumes its event owner.
  child->proxyState->pendingTransportOps = 0;
  child->ownerComm = comm;
}
