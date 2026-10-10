#include "graph.h"
#include "net.h"
#include "topo.h"
#include "transport.h"

#include <gtest/gtest.h>
#include <initializer_list>
#include <memory>

namespace {

TEST(PxnRelayTopology, ResolvesRelayLocalNicFromGuid) {
  static uint64_t relayGuids[2] = {22, 11};
  static int failedDev = -1;
  struct flagcxNetAdaptor netAdaptor = {};
  netAdaptor.devices = [](int *count) {
    *count = 2;
    return flagcxSuccess;
  };
  netAdaptor.getProperties = [](int dev, void *output) {
    if (dev == failedDev)
      return flagcxSystemError;
    auto *props = static_cast<flagcxNetProperties_t *>(output);
    props->guid = relayGuids[dev];
    return flagcxSuccess;
  };

  // Source NET/0 is GUID 11; relay enumerates the same NIC as NET/1.
  int relayDev = -1;
  flagcxNetSendSetupRequest request = {11};
  EXPECT_EQ(flagcxNetDevFromGuid(&netAdaptor, request.netGuid, &relayDev),
            flagcxSuccess);
  EXPECT_EQ(relayDev, 1);

  relayDev = -1;
  EXPECT_EQ(flagcxNetDevFromGuid(&netAdaptor, 0, &relayDev),
            flagcxInvalidArgument);
  EXPECT_EQ(flagcxNetDevFromGuid(&netAdaptor, 99, &relayDev),
            flagcxNotSupported);
  EXPECT_EQ(relayDev, -1);

  relayGuids[0] = 11;
  EXPECT_EQ(flagcxNetDevFromGuid(&netAdaptor, 11, &relayDev),
            flagcxNotSupported);
  EXPECT_EQ(relayDev, -1);
  relayGuids[0] = 22;

  failedDev = 0;
  EXPECT_EQ(flagcxNetDevFromGuid(&netAdaptor, 11, &relayDev),
            flagcxSystemError);
  EXPECT_EQ(relayDev, -1);
  failedDev = -1;
}

TEST(PxnRelayTopology, ResolvesListenerByNicGuid) {
  uint64_t localGuids[3] = {33, 11, 22};
  auto server = std::make_unique<flagcxTopoServer>();
  server->nodes[NET].count = 3;
  for (int dev = 0; dev < 3; ++dev) {
    server->nodes[NET].nodes[dev].net.dev = dev;
    server->nodes[NET].nodes[dev].net.guid = localGuids[dev];
  }
  int resolvedDev = -1;
  EXPECT_EQ(flagcxTopoNetDevFromGuid(server.get(), 22, &resolvedDev),
            flagcxSuccess);
  EXPECT_EQ(resolvedDev, 2);
  EXPECT_EQ(flagcxTopoNetDevFromGuid(server.get(), 99, &resolvedDev),
            flagcxNotSupported);
}

class PxnRelayTest : public ::testing::Test {
protected:
  void SetUp() override {
    topo = std::make_unique<flagcxTopoServer>();
    topo->nodes[APU].count = 2;
    topo->nodes[NET].count = 1;
    auto &source = topo->nodes[APU].nodes[0];
    auto &relay = topo->nodes[APU].nodes[1];
    source.id = FLAGCX_TOPO_ID(0, 1);
    source.apu.rank = 0;
    source.apu.gdrSupport = 1;
    relay.id = FLAGCX_TOPO_ID(0, 2);
    relay.apu.rank = 1;
    relay.apu.gdrSupport = 1;
    topo->nodes[NET].nodes[0].id = FLAGCX_TOPO_ID(0, 100);
    topo->nodes[NET].nodes[0].net.gdrSupport = 1;
    source.paths[APU] = new flagcxTopoPath[2]();
    source.paths[NET] = new flagcxTopoPath[1]();
    relay.paths[NET] = new flagcxTopoPath[1]();
    source.paths[APU][1].type = PATH_CCI;
    source.paths[APU][1].bw = 24;
    source.paths[NET][0].type = PATH_PHB;
    source.paths[NET][0].bw = 12;
    relay.paths[NET][0].type = PATH_PIX;
    relay.paths[NET][0].bw = 24;
  }

  void TearDown() override {
    delete[] topo->nodes[APU].nodes[0].paths[APU];
    delete[] topo->nodes[APU].nodes[0].paths[NET];
    delete[] topo->nodes[APU].nodes[1].paths[NET];
  }

  int selectedRank() {
    int rank = -1;
    EXPECT_EQ(flagcxTopoSelectPxnRelay(topo.get(), 0, 0, &rank), flagcxSuccess);
    return rank;
  }

  std::unique_ptr<flagcxTopoServer> topo;
};

TEST_F(PxnRelayTest, ChoosesLocalGdrRelayWithBetterNicPath) {
  EXPECT_EQ(selectedRank(), 1);
}

TEST_F(PxnRelayTest, KeepsDirectPathWhenPeerLinkCannotCarryPxn) {
  topo->nodes[APU].nodes[0].paths[APU][1].type = PATH_PHB;
  EXPECT_EQ(selectedRank(), 0);
}

TEST_F(PxnRelayTest, RejectsRemoteOrNonGdrRelay) {
  topo->nodes[APU].nodes[1].id = FLAGCX_TOPO_ID(1, 2);
  EXPECT_EQ(selectedRank(), 0);
  topo->nodes[APU].nodes[1].id = FLAGCX_TOPO_ID(0, 2);
  topo->nodes[APU].nodes[1].apu.gdrSupport = 0;
  EXPECT_EQ(selectedRank(), 0);
}

TEST_F(PxnRelayTest, RejectsNicWithoutGdrSupport) {
  topo->nodes[NET].nodes[0].net.gdrSupport = 0;
  EXPECT_EQ(selectedRank(), 0);
}

TEST_F(PxnRelayTest, UsesCapableRelayWhenSourceGpuLacksGdr) {
  auto &source = topo->nodes[APU].nodes[0];
  auto &relay = topo->nodes[APU].nodes[1];
  source.apu.gdrSupport = 0;
  EXPECT_EQ(selectedRank(), 1);

  relay.apu.gdrSupport = 0;
  EXPECT_EQ(selectedRank(), 0);
}

TEST_F(PxnRelayTest, AvoidsAnInferiorRelayWhenDirectNicPathIsUsable) {
  topo->nodes[APU].nodes[0].paths[NET][0].type = PATH_PXB;
  topo->nodes[APU].nodes[1].paths[NET][0].bw = 8;
  EXPECT_EQ(selectedRank(), 0);
}

TEST_F(PxnRelayTest, RejectsInvalidTopologyIndex) {
  int rank = -1;
  EXPECT_EQ(flagcxTopoSelectPxnRelay(topo.get(), 2, 0, &rank),
            flagcxInvalidArgument);
}

TEST_F(PxnRelayTest, IntermediateRankTracksTheSelectedPath) {
  constexpr int64_t netId = FLAGCX_TOPO_ID(0, 100);
  int intermediate = -1;
  ASSERT_EQ(flagcxTopoGetIntermediateRank(topo.get(), 0, netId, &intermediate),
            flagcxSuccess);
  EXPECT_EQ(intermediate, 0);

  auto &source = topo->nodes[APU].nodes[0];
  auto &relay = topo->nodes[APU].nodes[1];
  source.paths[NET][0].type = PATH_PXN;
  source.links[0].remNode = &relay;
  source.paths[NET][0].list[0] = &source.links[0];
  source.paths[NET][0].count = 1;
  ASSERT_EQ(flagcxTopoGetIntermediateRank(topo.get(), 0, netId, &intermediate),
            flagcxSuccess);
  EXPECT_EQ(intermediate, 1);
}

TEST_F(PxnRelayTest, RejectsPxnPathWithoutLocalRelay) {
  auto &path = topo->nodes[APU].nodes[0].paths[NET][0];
  path.type = PATH_PXN;
  int intermediate = -1;
  EXPECT_EQ(flagcxTopoGetIntermediateRank(topo.get(), 0, FLAGCX_TOPO_ID(0, 100),
                                          &intermediate),
            flagcxInternalError);
}

TEST(PxnRelayTopology, EightRankTwoFabricLayoutSelectsDestinationFabric) {
  auto topo = std::make_unique<flagcxTopoServer>();
  topo->nodes[APU].count = 8;
  topo->nodes[NET].count = 2;
  auto interServer = std::make_unique<flagcxInterServerTopo>();
  for (int net = 0; net < 2; ++net) {
    auto &nic = topo->nodes[NET].nodes[net];
    nic.id = FLAGCX_TOPO_ID(0, 100 + net);
    nic.net.dev = net;
    nic.net.bw = 24;
    nic.net.gdrSupport = 1;
  }
  for (int rank = 0; rank < 8; ++rank) {
    auto &apu = topo->nodes[APU].nodes[rank];
    apu.id = FLAGCX_TOPO_ID(0, rank + 1);
    apu.apu.rank = rank;
    apu.paths[APU] = new flagcxTopoPath[8]();
    apu.paths[NET] = new flagcxTopoPath[2]();
    const int ownFabric = rank < 4 ? 0 : 1;
    apu.apu.gdrSupport = 1;
    for (int net = 0; net < 2; ++net) {
      apu.paths[NET][net].type = net == ownFabric ? PATH_PIX : PATH_PHB;
      apu.paths[NET][net].bw = net == ownFabric ? 24 : 8;
    }
  }
  for (int rank = 0; rank < 8; ++rank) {
    for (int peer = 0; peer < 8; ++peer)
      topo->nodes[APU].nodes[rank].paths[APU][peer].type = PATH_CCI;
    for (int peer = 0; peer < 8; ++peer)
      topo->nodes[APU].nodes[rank].paths[APU][peer].bw = 24;
  }
  for (int rank : {0, 1, 4, 6}) {
    const int destinationFabric = rank < 4 ? 1 : 0;
    int selected = -1;
    ASSERT_EQ(flagcxTopoSelectPxnRelay(topo.get(), rank, destinationFabric,
                                       &selected),
              flagcxSuccess);
    EXPECT_NE(selected, rank);
    EXPECT_EQ(selected < 4, destinationFabric == 0);
  }
  for (int sourceRank = 0; sourceRank < 8; ++sourceRank) {
    for (int peerRank = 0; peerRank < 8; ++peerRank) {
      if (sourceRank == peerRank)
        continue;
      const int receiverNetDev = peerRank < 4 ? 0 : 1;
      int sendNetDev = -1;
      int relayRank = -1;
      ASSERT_EQ(flagcxTopoSelectNetRoute(
                    topo.get(), topo.get(), interServer.get(), sourceRank,
                    peerRank, receiverNetDev, &sendNetDev, &relayRank),
                flagcxSuccess);
      EXPECT_EQ(sendNetDev, receiverNetDev);
      const bool relayOnReceiverFabric =
          (relayRank < 4) == (receiverNetDev == 0);
      EXPECT_TRUE(relayOnReceiverFabric)
          << "source " << sourceRank << " peer " << peerRank;
      const bool crossesFabric = (sourceRank < 4) != (receiverNetDev == 0);
      if (crossesFabric)
        EXPECT_NE(relayRank, sourceRank);
      else
        EXPECT_EQ(relayRank, sourceRank);
    }
  }
  // A direct PCI route exists, but no GPU on the receiver's fabric supports
  // GDR. The source's distant path is not a usable PXN route either.
  for (int rank : {4, 5, 6, 7})
    topo->nodes[APU].nodes[rank].apu.gdrSupport = 0;
  int unsupportedDev = -1;
  int unsupportedRelay = -1;
  EXPECT_EQ(flagcxTopoSelectNetRoute(topo.get(), topo.get(), interServer.get(),
                                     0, 4, 1, &unsupportedDev,
                                     &unsupportedRelay),
            flagcxNotSupported);
  for (int rank = 0; rank < 8; ++rank) {
    delete[] topo->nodes[APU].nodes[rank].paths[APU];
    delete[] topo->nodes[APU].nodes[rank].paths[NET];
  }
}

TEST(PxnRelayTopology, ChoosesRouteToReceiversSelectedNic) {
  auto local = std::make_unique<flagcxTopoServer>();
  auto remote = std::make_unique<flagcxTopoServer>();
  auto interServer = std::make_unique<flagcxInterServerTopo>();
  local->nodes[APU].count = 2;
  local->nodes[NET].count = 2;
  local->serverId = 0;
  remote->serverId = 1;
  local->nodes[NET].nodes[0].net.gdrSupport = 1;
  local->nodes[NET].nodes[1].net.gdrSupport = 1;
  remote->nodes[APU].count = 1;
  remote->nodes[NET].count = 1;
  auto &source = local->nodes[APU].nodes[0];
  auto &relay = local->nodes[APU].nodes[1];
  source.id = FLAGCX_TOPO_ID(0, 1);
  source.apu.rank = 0;
  relay.id = FLAGCX_TOPO_ID(0, 2);
  relay.apu.rank = 1;
  relay.apu.gdrSupport = 1;
  source.paths[APU] = new flagcxTopoPath[2]();
  source.paths[APU][1].type = PATH_CCI;
  source.paths[APU][1].bw = 24;
  source.paths[NET] = new flagcxTopoPath[2]();
  relay.paths[NET] = new flagcxTopoPath[2]();
  for (int n = 0; n < 2; ++n) {
    auto &net = local->nodes[NET].nodes[n];
    net.id = FLAGCX_TOPO_ID(0, 100 + n);
    net.net.dev = n;
    net.net.guid = 100 + n;
    net.net.bw = 24;
    source.paths[NET][n].type = n == 0 ? PATH_PIX : PATH_PHB;
    source.paths[NET][n].bw = n == 0 ? 24 : 8;
    relay.paths[NET][n].type = n == 0 ? PATH_PHB : PATH_PIX;
    relay.paths[NET][n].bw = n == 0 ? 8 : 24;
  }
  auto &receiver = remote->nodes[APU].nodes[0];
  receiver.apu.rank = 2;
  receiver.apu.dev = 0;
  receiver.paths[NET] = new flagcxTopoPath[1]();
  receiver.paths[NET][0].type = PATH_PIX;
  receiver.paths[NET][0].bw = 24;
  remote->nodes[NET].nodes[0].net.guid = 300;
  remote->nodes[NET].nodes[0].net.bw = 24;
  auto route = std::make_unique<flagcxInterServerRoute>();
  route->interBw = 24;
  interServer->routeMap[101][300] = route.get();

  int netDev = -1;
  int relayRank = -1;
  EXPECT_EQ(flagcxTopoSelectNetRoute(local.get(), remote.get(),
                                     interServer.get(), 0, 2, 0, &netDev,
                                     &relayRank),
            flagcxSuccess);
  EXPECT_EQ(netDev, 1);
  EXPECT_EQ(relayRank, 1);

  interServer->routeMap.clear();
  EXPECT_EQ(flagcxTopoSelectNetRoute(local.get(), remote.get(),
                                     interServer.get(), 0, 2, 0, &netDev,
                                     &relayRank),
            flagcxNotSupported);

  remote->serverId = 0;
  remote->nodes[NET].nodes[0].net.guid = 101;
  remote->nodes[NET].nodes[0].id = FLAGCX_TOPO_ID(0, 101);
  EXPECT_EQ(flagcxTopoSelectNetRoute(local.get(), remote.get(),
                                     interServer.get(), 0, 2, 0, &netDev,
                                     &relayRank),
            flagcxSuccess);
  EXPECT_EQ(netDev, 1);
  EXPECT_EQ(relayRank, 1);

  // The receiver can explicitly choose a NIC other than its closest one.
  // The route query must use that advertised choice.
  remote->serverId = 1;
  remote->nodes[NET].count = 2;
  delete[] receiver.paths[NET];
  receiver.paths[NET] = new flagcxTopoPath[2]();
  receiver.paths[NET][0].bw = 24;
  receiver.paths[NET][0].type = PATH_PIX;
  receiver.paths[NET][1].bw = 8;
  receiver.paths[NET][1].type = PATH_PHB;
  auto &overrideNet = remote->nodes[NET].nodes[1];
  overrideNet.id = FLAGCX_TOPO_ID(1, 301);
  overrideNet.net.dev = 1;
  overrideNet.net.guid = 301;
  overrideNet.net.bw = 24;
  interServer->routeMap[101][301] = route.get();
  EXPECT_EQ(flagcxTopoSelectNetRoute(local.get(), remote.get(),
                                     interServer.get(), 0, 2, 1, &netDev,
                                     &relayRank),
            flagcxSuccess);
  EXPECT_EQ(netDev, 1);
  EXPECT_EQ(relayRank, 1);

  delete[] source.paths[APU];
  delete[] source.paths[NET];
  delete[] relay.paths[NET];
  delete[] receiver.paths[NET];
}

TEST(PxnRelayTopology, BaseNicPathDoesNotTransitAnotherApu) {
  auto topo = std::make_unique<flagcxTopoServer>();
  topo->nodes[APU].count = 2;
  topo->nodes[NET].count = 2;
  for (int i = 0; i < 2; ++i) {
    auto &apu = topo->nodes[APU].nodes[i];
    auto &net = topo->nodes[NET].nodes[i];
    apu.type = APU;
    net.type = NET;
    apu.id = FLAGCX_TOPO_ID(0, i + 1);
    net.id = FLAGCX_TOPO_ID(0, i + 100);
    ASSERT_EQ(flagcxTopoConnectNodes(&apu, &net, LINK_PCI, 12), flagcxSuccess);
    ASSERT_EQ(flagcxTopoConnectNodes(&net, &apu, LINK_PCI, 12), flagcxSuccess);
  }
  auto &first = topo->nodes[APU].nodes[0];
  auto &second = topo->nodes[APU].nodes[1];
  ASSERT_EQ(flagcxTopoConnectNodes(&first, &second, LINK_CCI, 24),
            flagcxSuccess);
  ASSERT_EQ(flagcxTopoConnectNodes(&second, &first, LINK_CCI, 24),
            flagcxSuccess);
  ASSERT_EQ(flagcxTopoComputePaths(topo.get(), NULL), flagcxSuccess);
  ASSERT_NE(first.paths[NET], nullptr);
  ASSERT_NE(second.paths[NET], nullptr);
  EXPECT_GT(first.paths[NET][0].bw, 0);
  EXPECT_EQ(first.paths[NET][1].bw, 0);
  EXPECT_EQ(second.paths[NET][0].bw, 0);
  EXPECT_GT(second.paths[NET][1].bw, 0);
  EXPECT_EQ(first.paths[APU][1].type, PATH_CCI);
  for (int t = 0; t < FLAGCX_TOPO_NODE_TYPES; ++t) {
    for (int n = 0; n < topo->nodes[t].count; ++n) {
      for (int p = 0; p < FLAGCX_TOPO_NODE_TYPES; ++p)
        free(topo->nodes[t].nodes[n].paths[p]);
    }
  }
}

} // namespace
