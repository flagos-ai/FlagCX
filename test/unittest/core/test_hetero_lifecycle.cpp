#include "p2p.h"
#include "reg_pool.h"
#include "rma_init_rollback.hpp"
#include <string>

namespace {
class HeteroLifecycleTest : public RmaInitRollbackTest {
protected:
  inline static HeteroLifecycleTest *active;
  int fifoFrees = 0, importCloses = 0;
  bool failFifo = false, failImport = false;
  void SetUp() override {
    RmaInitRollbackTest::SetUp();
    active = this;
    device.deviceSynchronize = [] {
      active->drains++;
      return flagcxSuccess;
    };
    device.deviceFree = [](void *, flagcxMemType_t, flagcxStream_t) {
      active->fifoFrees++;
      return active->failFifo ? flagcxSystemError : flagcxSuccess;
    };
    device.ipcMemHandleClose = [](void *) {
      active->importCloses++;
      return active->failImport ? flagcxRemoteError : flagcxSuccess;
    };
    makeOwner();
    attachFallibleTransport();
    transport.cleanupProxyConnection = flagcxP2pCleanupProxyConnection;
    auto *connection = owner->proxyState->connectionPool->pools[0];
    connection->transport = TRANSPORT_P2P;
    connection->send = 0;
    connection->transportResources = calloc(1, sizeof(flagcxP2pResources));
    fifo()->localRecvFifo = (void *)0x44000;
  }
  flagcxP2pResources *fifo() {
    return (flagcxP2pResources *)owner->proxyState->connectionPool->pools[0]
        ->transportResources;
  }
  flagcxResult_t destroy() {
    flagcxResult_t result = flagcxHeteroCommDestroy(&owner);
    if (result == flagcxSuccess)
      owner = nullptr;
    return result;
  }
  void TearDown() override {
    failFifo = failImport = false;
    if (owner != nullptr) {
      EXPECT_EQ(destroy(), flagcxSuccess);
    }
    RmaInitRollbackTest::TearDown();
  }
};

TEST_F(HeteroLifecycleTest, ReleaseCallbackCannotInventPeerConfirmation) {
  auto *connection = owner->proxyState->connectionPool->pools[0];
  EXPECT_EQ(flagcxP2pCleanupProxyConnection(
                connection, flagcxTransportCleanupReleaseResources),
            flagcxInvalidUsage);
  EXPECT_EQ(fifoFrees, 0);
  EXPECT_FALSE(fifo()->importsClosedConfirmed);
  EXPECT_NE(connection->transportResources, nullptr);
}

TEST_F(HeteroLifecycleTest, PrepareDoesNotAuthorizeExporterRelease) {
  EXPECT_EQ(flagcxHeteroCommPrepareDestroy(owner), flagcxSuccess);
  EXPECT_FALSE(owner->finalizeState.importsClosedConfirmed);
  EXPECT_EQ(flagcxProxyDestroy(owner), flagcxInvalidUsage);
  EXPECT_EQ(fifoFrees, 0);
  EXPECT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  EXPECT_EQ(fifoFrees, 0);
  EXPECT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  EXPECT_EQ(drains, 0);
  EXPECT_EQ(destroy(), flagcxSuccess);
  EXPECT_EQ(fifoFrees, 1);
}

TEST_F(HeteroLifecycleTest, FailedReleaseRetriesLocallyAfterImplicitFinalize) {
  failFifo = true;
  EXPECT_EQ(destroy(), flagcxSystemError);
  ASSERT_NE(owner, nullptr);
  EXPECT_TRUE(owner->finalizeState.importsClosedConfirmed);
  EXPECT_EQ(owner->finalizeState.phase, flagcxHeteroFinalizeComplete);
  EXPECT_EQ(drains, 0);
  failFifo = false;
  device.deviceSynchronize = [] { return flagcxRemoteError; };
  EXPECT_EQ(destroy(), flagcxSuccess);
  EXPECT_EQ(fifoFrees, 2);
  EXPECT_EQ(drains, 0);
}

TEST_F(HeteroLifecycleTest, FailedImportKeepsOwnerAndSuccessfulDrainPrefix) {
  // Replace the exporter with an imported FIFO to exercise the real close.
  fifo()->localRecvFifo = nullptr;
  fifo()->importedRecvFifoBase = (void *)0x55000;
  owner->proxyState->connectionPool->pools[0]->send = 1;
  failImport = true;
  EXPECT_EQ(destroy(), flagcxRemoteError);
  ASSERT_NE(owner, nullptr);
  EXPECT_FALSE(owner->finalizeState.importsClosedConfirmed);
  EXPECT_NE(fifo()->importedRecvFifoBase, nullptr);
  EXPECT_EQ(fifoFrees, 0);
  failImport = false;
  EXPECT_EQ(destroy(), flagcxSuccess);
  EXPECT_EQ(importCloses, 2);
  EXPECT_EQ(drains, 0);
}

TEST_F(HeteroLifecycleTest, FinalizeRejectsFurtherTwoSidedSubmission) {
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  EXPECT_EQ(flagcxHeteroSend(nullptr, 0, flagcxChar, 0, owner, nullptr),
            flagcxInvalidUsage);
  EXPECT_EQ(flagcxHeteroRecv(nullptr, 0, flagcxChar, 0, owner, nullptr),
            flagcxInvalidUsage);
}
TEST_F(HeteroLifecycleTest, FailedDrainKeepsRmaAdmissionClosedUntilRetry) {
  failDevice = true;
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxRemoteError);
  EXPECT_EQ(owner->finalizeState.started, 1);
  EXPECT_EQ(owner->finalizeState.phase, flagcxHeteroFinalizeDrain);
  EXPECT_NE(fifo()->localRecvFifo, nullptr);
  EXPECT_EQ(fifoFrees, 0);
  EXPECT_EQ(flagcxHeteroPut(owner, 0, 0, 0, 8, 0, 0), flagcxInvalidUsage);
  EXPECT_EQ(flagcxHeteroGet(owner, 0, 0, 0, 8, 0, 0), flagcxInvalidUsage);
  failDevice = false;
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  EXPECT_EQ(owner->finalizeState.started, 1);
  EXPECT_EQ(flagcxHeteroPut(owner, 0, 0, 0, 8, 0, 0), flagcxInvalidUsage);
  EXPECT_EQ(destroy(), flagcxSuccess);
  EXPECT_EQ(fifoFrees, 1);
}

TEST_F(HeteroLifecycleTest, FailedCompletionWaitKeepsRmaAdmissionClosed) {
  static bool failWait;
  failWait = true;
  device.eventCreate = [](flagcxEvent_t *event, flagcxEventType_t) {
    *event = reinterpret_cast<flagcxEvent_t>(1);
    return flagcxSuccess;
  };
  device.eventRecord = [](flagcxEvent_t, flagcxStream_t) {
    return flagcxSuccess;
  };
  device.eventQuery = [](flagcxEvent_t) { return flagcxSuccess; };
  device.eventSynchronize = [](flagcxEvent_t) {
    return failWait ? flagcxUnhandledDeviceError : flagcxSuccess;
  };
  device.eventDestroy = [](flagcxEvent_t) { return flagcxSuccess; };
  flagcxWorkRecord *record = nullptr;
  auto stream = reinterpret_cast<flagcxStream_t>(1);
  ASSERT_EQ(flagcxWorkPrepare(&owner->submittedWork, stream, &record),
            flagcxSuccess);
  ASSERT_EQ(
      flagcxWorkFinish(record, stream, flagcxSuccess, flagcxWorkHeteroGroup, 0),
      flagcxSuccess);
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxUnhandledDeviceError);
  EXPECT_EQ(owner->finalizeState.phase, flagcxHeteroFinalizeDrain);
  EXPECT_EQ(flagcxHeteroPut(owner, 0, 0, 0, 8, 0, 0), flagcxInvalidUsage);
  EXPECT_EQ(fifoFrees, 0);
  failWait = false;
  ASSERT_EQ(flagcxHeteroCommFinalize(owner), flagcxSuccess);
  EXPECT_EQ(flagcxHeteroPut(owner, 0, 0, 0, 8, 0, 0), flagcxInvalidUsage);
  EXPECT_EQ(destroy(), flagcxSuccess);
  EXPECT_EQ(fifoFrees, 1);
}

TEST_F(HeteroLifecycleTest, DirectDestroyReleasesRegistrationOwnerReferences) {
  void *buffer = (void *)0x880000;
  ASSERT_EQ(globalRegPool.registerBuffer(owner, buffer, 64), flagcxSuccess);
  ASSERT_EQ(destroy(), flagcxSuccess);
  flagcxRegItem *item = globalRegPool.acquireItem(nullptr, buffer);
  EXPECT_EQ(item, nullptr);
  if (item != nullptr)
    globalRegPool.releaseItem(item);
}

TEST_F(HeteroLifecycleTest, OuterDestroyUsesItsOwnerAfterEnvironmentChanges) {
  struct RestoreEnvironment {
    bool present = getenv("FLAGCX_USE_HETERO_COMM") != nullptr;
    std::string value = present ? getenv("FLAGCX_USE_HETERO_COMM") : "";
    ~RestoreEnvironment() {
      if (present)
        setenv("FLAGCX_USE_HETERO_COMM", value.c_str(), 1);
      else
        unsetenv("FLAGCX_USE_HETERO_COMM");
    }
  } restore;
  setenv("FLAGCX_USE_HETERO_COMM", "0", 1);
  outer = (flagcxComm_t)calloc(1, sizeof(flagcxComm));
  ASSERT_NE(outer, nullptr);
  outer->commType = flagcxCommunicatorHomo;
  outer->nranks = outer->localRanks = 1;
  outer->ipcTable = outer->ipcTableInline;
  outer->ipcTableCapacity = FLAGCX_MAX_IPC_ENTRIES;
  outer->heteroComm = owner;
  owner->ownerComm = outer;
  owner->workOwner = &outer->submittedWork;
  owner = nullptr;
  ASSERT_EQ(flagcxCommInitFinalizePolicy(outer), flagcxSuccess);
  ASSERT_EQ(flagcxCommFinalize(outer), flagcxSuccess);
  EXPECT_EQ(fifoFrees, 0);
  EXPECT_EQ(flagcxCommDestroy(outer), flagcxSuccess);
  outer = nullptr;
  EXPECT_EQ(fifoFrees, 1);
}

} // namespace
