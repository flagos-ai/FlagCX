#include "p2p.h"
#include "reg_pool.h"
#include "rma_init_rollback.hpp"

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
    flagcxResult_t result = flagcxHeteroCommDestroy(owner);
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
TEST_F(HeteroLifecycleTest, DirectDestroyReleasesRegistrationOwnerReferences) {
  void *buffer = (void *)0x880000;
  ASSERT_EQ(globalRegPool.registerBuffer(owner, buffer, 64), flagcxSuccess);
  ASSERT_EQ(destroy(), flagcxSuccess);
  flagcxRegSnapshot snapshot = {};
  ASSERT_EQ(globalRegPool.acquireItem(nullptr, buffer, &snapshot),
            flagcxSuccess);
  EXPECT_EQ(snapshot.item, nullptr);
  globalRegPool.releaseItem(&snapshot);
}

} // namespace
