#include "relay_shutdown.hpp"
#include "rma_init_rollback.hpp"

namespace {
class RelayShutdownTest : public RmaInitRollbackTest {
protected:
  inline static RelayShutdownTest *active;
  int peerFd = -1, closed = 0;
  bool failClose = false;
  flagcxProxyConnector *connector = nullptr;
  void SetUp() override {
    RmaInitRollbackTest::SetUp();
    active = this;
    makeOwner();
    connector = attachRelayShutdownEndpoint(owner, &peerFd);
    ASSERT_NE(connector, nullptr);
    owner->proxyState->initialized = 1;
    device.ipcMemHandleClose = [](void *) {
      active->closed++;
      return active->failClose ? flagcxRemoteError : flagcxSuccess;
    };
  }
  void TearDown() override {
    failClose = false;
    owner->proxyState->initialized = 1;
    if (!connector->connection->relayReleaseAcked) {
      EXPECT_TRUE(replyRelayShutdown(peerFd, connector, flagcxSuccess));
      EXPECT_EQ(flagcxProxyCloseRelayImports(owner), flagcxSuccess);
    }
    EXPECT_EQ(flagcxProxyStop(owner), flagcxSuccess);
    owner->proxyState->initialized =
        0; // This wire fixture has no service thread.
    close(peerFd);
    RmaInitRollbackTest::TearDown();
  }
};

TEST_F(RelayShutdownTest, CloseFailurePreservesServiceAndSkipsAck) {
  failClose = true;
  EXPECT_EQ(flagcxHeteroCommCloseRelayImports(owner), flagcxRemoteError);
  EXPECT_FALSE(owner->proxyState->relayImports->closed);
  EXPECT_EQ(flagcxProxyStop(owner), flagcxInvalidUsage);
  EXPECT_EQ(owner->proxyState->stop, 0);
  EXPECT_EQ(owner->proxyState->rpcStopping, 0);
  EXPECT_EQ(owner->proxyState->initialized, 1);
  failClose = false;
  ASSERT_TRUE(replyRelayShutdown(peerFd, connector, flagcxSuccess));
  EXPECT_EQ(flagcxHeteroCommCloseRelayImports(owner), flagcxSuccess);
  EXPECT_TRUE(connector->connection->relayReleaseAcked);
  EXPECT_EQ(closed, 2);
}

TEST_F(RelayShutdownTest, AckFailureRetriesWithoutClosingMappingAgain) {
  ASSERT_TRUE(replyRelayShutdown(peerFd, connector, flagcxRemoteError));
  EXPECT_EQ(flagcxHeteroCommCloseRelayImports(owner), flagcxRemoteError);
  EXPECT_TRUE(owner->proxyState->relayImports->closed);
  EXPECT_FALSE(connector->connection->relayReleaseAcked);
  EXPECT_EQ(connector->connection->relayBufferImport, nullptr);
  EXPECT_EQ(closed, 1);
  EXPECT_EQ(flagcxProxyStop(owner), flagcxInvalidUsage);
  EXPECT_EQ(owner->proxyState->rpcStopping, 0);
  ASSERT_TRUE(replyRelayShutdown(peerFd, connector, flagcxSuccess));
  EXPECT_EQ(flagcxHeteroCommCloseRelayImports(owner), flagcxSuccess);
  EXPECT_TRUE(connector->connection->relayReleaseAcked);
  EXPECT_EQ(flagcxHeteroCommCloseRelayImports(owner), flagcxSuccess);
  EXPECT_EQ(closed, 1);
}

TEST_F(RelayShutdownTest, StoppedServiceDoesNotHideUnclosedImportOrMissingAck) {
  owner->proxyState->initialized = 0;
  EXPECT_EQ(flagcxProxyCloseRelayImports(owner), flagcxInvalidUsage);
  owner->proxyState->relayImports->closed = true;
  EXPECT_EQ(flagcxProxyCloseRelayImports(owner), flagcxInvalidUsage);
  EXPECT_EQ(closed, 0);
}
} // namespace
