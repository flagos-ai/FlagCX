#include "p2p_control.h"

#include <gtest/gtest.h>

namespace {

TEST(P2pProtocolTest, AcceptsMatchingImplementationAndTransport) {
  const auto local = flagcxP2pControl::protocolHello(
      flagcxP2pControl::kProtocolLegacy, flagcxP2pControl::kProtocolIbrc);
  const auto remote = flagcxP2pControl::protocolHello(
      flagcxP2pControl::kProtocolLegacy, flagcxP2pControl::kProtocolIbrc);

  EXPECT_TRUE(flagcxP2pControl::protocolCompatible(local, remote));
}

TEST(P2pProtocolTest, RejectsMixedEngineImplementations) {
  const auto legacy = flagcxP2pControl::protocolHello(
      flagcxP2pControl::kProtocolLegacy, flagcxP2pControl::kProtocolIbrc);
  const auto shared = flagcxP2pControl::protocolHello(
      flagcxP2pControl::kProtocolShared, flagcxP2pControl::kProtocolIbrc);

  EXPECT_FALSE(flagcxP2pControl::protocolCompatible(legacy, shared));
  EXPECT_FALSE(flagcxP2pControl::protocolCompatible(shared, legacy));
}

TEST(P2pProtocolTest, RejectsMixedTransportsAndVersions) {
  const auto ibrc = flagcxP2pControl::protocolHello(
      flagcxP2pControl::kProtocolLegacy, flagcxP2pControl::kProtocolIbrc);
  auto barex = flagcxP2pControl::protocolHello(
      flagcxP2pControl::kProtocolLegacy, flagcxP2pControl::kProtocolBarex);

  EXPECT_FALSE(flagcxP2pControl::protocolCompatible(ibrc, barex));
  barex = ibrc;
  ++barex.version;
  EXPECT_FALSE(flagcxP2pControl::protocolCompatible(ibrc, barex));
}

} // namespace
