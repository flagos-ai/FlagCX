#include "comm.h"
#include "proxy.h"
#include <gtest/gtest.h>

TEST(RelayLifecycle, StoppedServiceCannotHideAnOpenImport) {
  flagcxHeteroComm comm = {};
  flagcxProxyState state = {};
  flagcxNetRelayImport import = {};
  comm.proxyState = &state;
  state.relayImports = &import;
  EXPECT_EQ(flagcxProxyCloseRelayImports(&comm), flagcxInvalidUsage);
  EXPECT_EQ(flagcxProxyStop(&comm), flagcxInvalidUsage);
  EXPECT_FALSE(import.closed);
  import.closed = true;
  EXPECT_EQ(flagcxProxyCloseRelayImports(&comm), flagcxSuccess);
}
TEST(RelayLifecycle, ClosedMappingStillNeedsReleaseAcknowledgement) {
  flagcxHeteroComm comm = {};
  flagcxProxyState state = {};
  flagcxNetRelayImport import = {};
  flagcxChannelPeer peer = {};
  flagcxChannelPeer *peers[] = {&peer};
  flagcxProxyConnection connection = {};
  comm.proxyState = &state;
  comm.nRanks = 1;
  comm.channels[0].peers = peers;
  peer.send[0].proxyConn.connection = &connection;
  state.relayImports = &import;
  import.closed = true;
  connection.relayImport = &import;
  EXPECT_EQ(flagcxProxyCloseRelayImports(&comm), flagcxInvalidUsage);
  connection.relayReleaseAcked = 1;
  EXPECT_EQ(flagcxProxyCloseRelayImports(&comm), flagcxSuccess);
}
