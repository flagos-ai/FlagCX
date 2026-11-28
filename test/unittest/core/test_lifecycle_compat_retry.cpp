#include "ipc_lifecycle_mock.hpp"
#include <unistd.h>

TEST_F(IpcLifecycleMock, BuiltinCompatFinalizeDestroysNativeButKeepsBacking) {
  addBackingAndImport();
  comm->homoComm = comm->tunerInnerComm = (flagcxInnerComm_t)0x9000;
  comm->nativeDestroyFinalize = true;
  sdk.commFinalize = nullptr;
  sdk.commDestroy = [](flagcxInnerComm_t) {
    ++active->destroyCalls;
    return active->nativeResult;
  };
  nativeResult = flagcxInProgress;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxInProgress);
  EXPECT_NE(comm->homoComm, nullptr);
  EXPECT_TRUE(frees.empty());
  nativeResult = flagcxSuccess;
  ASSERT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(comm->homoComm, nullptr);
  EXPECT_EQ(comm->tunerInnerComm, nullptr);
  EXPECT_TRUE(frees.empty());
  destroySuccessfully();
  EXPECT_EQ(destroyCalls, 2);
  EXPECT_EQ(finalizeCalls, 0);
  EXPECT_EQ(frees.size(), 2u);
}

TEST_F(IpcLifecycleMock, UnknownNativeUnsupportedFinalizeRetainsOwner) {
  comm->homoComm = (flagcxInnerComm_t)0x9000;
  nativeResult = flagcxNotSupported;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxNotSupported);
  EXPECT_EQ(destroyCalls, 0);
  EXPECT_NE(comm->homoComm, nullptr);
  nativeResult = flagcxSuccess;
}

TEST_F(IpcLifecycleMock, InitRollbackDestroysHalfBuiltNativeWithoutFinalize) {
  comm->commType = flagcxCommunicatorUnknown;
  comm->homoComm = (flagcxInnerComm_t)0x9000;
  comm->initRollbackPending = true;
  comm->closing = true;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxInvalidUsage);
  destroySuccessfully();
  EXPECT_EQ(finalizeCalls, 0);
  EXPECT_EQ(destroyCalls, 1);
}

TEST_F(IpcLifecycleMock, FailedNativeAbortRetainsOwnerForRetry) {
  comm->homoComm = (flagcxInnerComm_t)0x9000;
  sdk.commAbort = [](flagcxInnerComm_t) {
    ++active->backendCleanups;
    return active->nativeResult;
  };
  nativeResult = flagcxRemoteError;
  EXPECT_EQ(flagcxCommAbort(comm), flagcxRemoteError);
  EXPECT_NE(comm->homoComm, nullptr);
  EXPECT_EQ(destroyCalls, 0);
  EXPECT_EQ(flagcxAllGather(nullptr, nullptr, 0, flagcxInt, comm, nullptr),
            flagcxInvalidUsage);
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxInProgress);
  EXPECT_EQ(finalizeCalls, 0);
  nativeResult = flagcxSuccess;
  ASSERT_EQ(flagcxCommAbort(comm), flagcxSuccess);
  comm = nullptr;
  EXPECT_EQ(backendCleanups, 2);
  EXPECT_EQ(finalizeCalls, 0);
  EXPECT_EQ(destroyCalls, 1);
}

TEST_F(IpcLifecycleMock,
       UnsupportedFirstAbortLeavesFinalizeAndNewAliasesUsable) {
  comm->homoComm = (flagcxInnerComm_t)0x9000;
  sdk.commAbort = [](flagcxInnerComm_t) { return flagcxNotSupported; };
  EXPECT_EQ(flagcxCommAbort(comm), flagcxNotSupported);
  EXPECT_FALSE(comm->closing);
  EXPECT_FALSE(comm->nativeCatalogReady);
  comm->homoInterComm = (flagcxInnerComm_t)0xa000;
  ASSERT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  EXPECT_EQ(finalizeCalls, 2);
}

TEST_F(IpcLifecycleMock, HeterogeneousAbortDoesNotConsumeNativeOwner) {
  addTransport();
  comm->homoComm = (flagcxInnerComm_t)0x9000;
  sdk.commAbort = [](flagcxInnerComm_t) {
    ++active->backendCleanups;
    return flagcxSuccess;
  };
  EXPECT_EQ(flagcxCommAbort(comm), flagcxNotSupported);
  EXPECT_EQ(backendCleanups, 0);
  EXPECT_NE(comm->homoComm, nullptr);
}

TEST_F(IpcLifecycleMock, LocalAbortDoesNotInventPeerImportClosePermission) {
  comm->nranks = 2;
  comm->internalBuffersNeedPeerClose = true;
  comm->homoComm = (flagcxInnerComm_t)0x9000;
  sdk.commAbort = [](flagcxInnerComm_t) { return flagcxSuccess; };
  EXPECT_EQ(flagcxCommAbort(comm), flagcxInProgress);
  EXPECT_FALSE(comm->internalImportsConfirmed);
  EXPECT_EQ(destroyCalls, 0);
  // Mock teardown only: there is no bootstrap peer in this unit test.
  comm->nranks = 1;
}

TEST_F(IpcLifecycleMock, ClosedDiagnosticsNeverTouchRetiredNativeState) {
  comm->homoComm = (flagcxInnerComm_t)0x9000;
  comm->closing = true;
  comm->lastResult = flagcxRemoteError;
  sdk.getLastError = [](flagcxInnerComm_t) -> const char * {
    ADD_FAILURE() << "retired SDK state was queried";
    return "unexpected";
  };
  sdk.commGetAsyncError = [](flagcxInnerComm_t, flagcxResult_t *) {
    ADD_FAILURE() << "retired SDK state was queried";
    return flagcxInternalError;
  };
  EXPECT_STREQ(flagcxGetLastError(comm),
               flagcxGetErrorString(flagcxRemoteError));
  flagcxResult_t result;
  EXPECT_EQ(flagcxCommGetAsyncError(comm, &result), flagcxSuccess);
  EXPECT_EQ(result, flagcxRemoteError);
}

TEST_F(IpcLifecycleMock, RetiredProxyErrorIsReportedBeforeRetryCompletes) {
  addTransport();
  comm->heteroComm->proxyState->workerCleanupResult = flagcxRemoteError;
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxRemoteError);
  EXPECT_NE(comm->heteroComm, nullptr);
  EXPECT_EQ(transportFrees, 0);
  EXPECT_EQ(flagcxCommFinalize(comm), flagcxSuccess);
  destroySuccessfully();
  EXPECT_EQ(transportFrees, 1);
}

TEST_F(IpcLifecycleMock, ClosedFusedAndAsyncGetEntriesRejectBeforePreparation) {
  comm->closing = true;
  EXPECT_EQ(flagcxAllGatherGemm(nullptr, nullptr, nullptr, 1, 1, 1, flagcxFloat,
                                comm, nullptr),
            flagcxInvalidUsage);
  EXPECT_EQ(flagcxGemmReduceScatter(nullptr, nullptr, nullptr, 1, 1, 1,
                                    flagcxFloat, flagcxSum, comm, nullptr),
            flagcxInvalidUsage);
  EXPECT_EQ(
      flagcxGetAsync(nullptr, 0, flagcxFloat, 0, nullptr, 0, 0, comm, nullptr),
      flagcxInvalidUsage);
}

TEST_F(IpcLifecycleMock, HostMappedBarrierRollbackRetainsPinnedMappingOwner) {
  // Re-exec isolates the cached runtime parameter from other test cases.
  std::string previous = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  flagcxDevCommRequirements requirements =
      FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  requirements.intraBarrierCount = 1;
  EXPECT_EXIT(
      {
        setenv("FLAGCX_SIGNAL_HOST_ENABLE", "1", 1);
        auto check = [](bool ok) {
          if (!ok)
            _exit(90);
        };
        comm->magic = (uint64_t)getpid();
        comm->localRankToRank = (int *)calloc(1, sizeof(int));
        device.hostRegister = [](void *, size_t) {
          ++active->transportCloses;
          return flagcxSuccess;
        };
        device.hostUnregister = [](void *) {
          ++active->backendCleanups;
          return active->failDevice ? flagcxInProgress : flagcxSuccess;
        };
        device.hostGetDevicePointer = [](void **, void *) {
          return flagcxRemoteError;
        };
        flagcxDevCommInternal view = {};
        view.barrierIpcIndex = view.signalIpcSlot = -1;
        check(backend.devCommCreate(comm, &requirements, sizeof(requirements),
                                    &view) == flagcxRemoteError);
        check(transportCloses == 1 && backendCleanups == 0);
        check(view.registeredBarrierShmCount == 1);
        check(view.myShmHandle != nullptr &&
              view.localBarrierShmPtr != nullptr);
        auto owner = view.myShmHandle;
        auto mapping = view.localBarrierShmPtr;
        failDevice = true;
        check(backend.devCommDestroy(comm, &view) == flagcxInProgress);
        check(view.myShmHandle == owner && view.localBarrierShmPtr == mapping);
        check(view.registeredBarrierShmCount == 1);
        failDevice = false;
        check(backend.devCommDestroy(comm, &view) == flagcxSuccess);
        check(view.myShmHandle == nullptr &&
              view.registeredBarrierShmCount == 0);
        check(backendCleanups == 2);
        _exit(0);
      },
      ::testing::ExitedWithCode(0), "");
  ::testing::FLAGS_gtest_death_test_style = previous;
}
