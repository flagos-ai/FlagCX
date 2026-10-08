#include "p2p_control.h"

#include <gtest/gtest.h>

#include <sys/socket.h>
#include <unistd.h>

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

TEST(P2pProtocolTest, FormatsIpv4DnsAndIpv6Endpoints) {
  EXPECT_EQ(flagcxP2pControl::hostPort("127.0.0.1", 1234), "127.0.0.1:1234");
  EXPECT_EQ(flagcxP2pControl::hostPort("example.test", 1234),
            "example.test:1234");
  EXPECT_EQ(flagcxP2pControl::hostPort("2001:db8::1", 1234),
            "[2001:db8::1]:1234");
  EXPECT_EQ(flagcxP2pControl::hostPort("[2001:db8::1]", 1234),
            "[2001:db8::1]:1234");
}

TEST(P2pProtocolTest, BoundedReceiveRejectsStalledPeer) {
  int sockets[2] = {-1, -1};
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  std::atomic<bool> stop{false};
  uint64_t preface = 0;

  EXPECT_FALSE(flagcxP2pControl::receive(sockets[0], &preface, sizeof(preface),
                                         stop, 10));

  close(sockets[0]);
  close(sockets[1]);
}

TEST(P2pProtocolTest, BoundedReceiveObservesShutdown) {
  int sockets[2] = {-1, -1};
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  std::atomic<bool> stop{true};
  uint64_t preface = 0;

  EXPECT_FALSE(flagcxP2pControl::receive(sockets[0], &preface, sizeof(preface),
                                         stop, 5000));

  close(sockets[0]);
  close(sockets[1]);
}

TEST(P2pProtocolTest, BoundedSendObservesShutdown) {
  int sockets[2] = {-1, -1};
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets), 0);
  std::atomic<bool> stop{true};
  const uint64_t preface = 0;

  EXPECT_FALSE(flagcxP2pControl::send(sockets[0], &preface, sizeof(preface),
                                      stop, 5000));

  close(sockets[0]);
  close(sockets[1]);
}

} // namespace
