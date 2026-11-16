#include <cstdlib>
#include <memory>

#include <gtest/gtest.h>

#include "graph.h"
#include "topo.h"
#include "xml.h"

namespace {

flagcxResult_t addCpuXml(flagcxXml *xml, flagcxXmlNode *system,
                         const char *hostHash, const char *numaId) {
  flagcxXmlNode *cpu = nullptr;
  FLAGCXCHECK(xmlAddNode(xml, system, "cpu", &cpu));
  FLAGCXCHECK(xmlSetAttr(cpu, "host_hash", hostHash));
  FLAGCXCHECK(xmlSetAttr(cpu, "numaid", numaId));
  // A synthetic CPU architecture keeps this test independent of the host.
  FLAGCXCHECK(xmlSetAttr(cpu, "arch", "arm64"));
  return flagcxSuccess;
}

flagcxResult_t connectBothWays(flagcxTopoNode *first, flagcxTopoNode *second) {
  FLAGCXCHECK(flagcxTopoConnectNodes(first, second, LINK_PCI, PCI_BW));
  FLAGCXCHECK(flagcxTopoConnectNodes(second, first, LINK_PCI, PCI_BW));
  return flagcxSuccess;
}

TEST(TopoUnknownNuma, SameHostCanReachNicWithoutConnectingOtherHost) {
  flagcxXml *rawXml = nullptr;
  ASSERT_EQ(xmlAlloc(&rawXml, 5), flagcxSuccess);
  std::unique_ptr<flagcxXml, decltype(&std::free)> xml(rawXml, &std::free);

  flagcxXmlNode *system = nullptr;
  ASSERT_EQ(xmlAddNode(xml.get(), nullptr, "system", &system), flagcxSuccess);
  ASSERT_EQ(addCpuXml(xml.get(), system, "11", "-1"), flagcxSuccess);
  ASSERT_EQ(addCpuXml(xml.get(), system, "11", "5"), flagcxSuccess);
  ASSERT_EQ(addCpuXml(xml.get(), system, "22", "-1"), flagcxSuccess);
  ASSERT_EQ(addCpuXml(xml.get(), system, "22", "6"), flagcxSuccess);

  flagcxTopoServer *rawTopo = nullptr;
  ASSERT_EQ(flagcxTopoGetServerTopoFromXml(xml.get(), &rawTopo, 0x11),
            flagcxSuccess);
  std::unique_ptr<flagcxTopoServer, decltype(&flagcxTopoFree)> topo(
      rawTopo, &flagcxTopoFree);
  ASSERT_EQ(topo->nodes[CPU].count, 4);
  flagcxTopoNode *cpus = topo->nodes[CPU].nodes;

  EXPECT_EQ(FLAGCX_TOPO_ID_SERVER_ID(cpus[0].id), 0);
  EXPECT_EQ(FLAGCX_TOPO_ID_LOCAL_ID(cpus[0].id), 0x00ffffffffffffffLL);
  EXPECT_EQ(cpus[1].id, FLAGCX_TOPO_ID(0, 5));
  EXPECT_EQ(FLAGCX_TOPO_ID_SERVER_ID(cpus[2].id), 1);
  EXPECT_EQ(FLAGCX_TOPO_ID_LOCAL_ID(cpus[2].id), 0x00ffffffffffffffLL);
  EXPECT_EQ(cpus[3].id, FLAGCX_TOPO_ID(1, 6));

  flagcxTopoNode *apu = nullptr;
  ASSERT_EQ(flagcxTopoCreateNode(topo.get(), &apu, APU, FLAGCX_TOPO_ID(0, 1)),
            flagcxSuccess);
  apu->apu.rank = 0;
  apu->apu.dev = 0;
  ASSERT_EQ(connectBothWays(apu, cpus + 0), flagcxSuccess);

  flagcxTopoNode *localNet = nullptr;
  ASSERT_EQ(
      flagcxTopoCreateNode(topo.get(), &localNet, NET, FLAGCX_TOPO_ID(0, 1)),
      flagcxSuccess);
  localNet->net.dev = 3;
  ASSERT_EQ(connectBothWays(localNet, cpus + 1), flagcxSuccess);

  flagcxTopoNode *remoteNet = nullptr;
  ASSERT_EQ(
      flagcxTopoCreateNode(topo.get(), &remoteNet, NET, FLAGCX_TOPO_ID(1, 1)),
      flagcxSuccess);
  remoteNet->net.dev = 4;
  ASSERT_EQ(connectBothWays(remoteNet, cpus + 3), flagcxSuccess);

  ASSERT_EQ(flagcxTopoComputePaths(topo.get(), nullptr), flagcxSuccess);
  ASSERT_NE(apu->paths[NET], nullptr);
  EXPECT_GT(apu->paths[NET][0].bw, 0);
  EXPECT_EQ(apu->paths[NET][1].bw, 0);
  int netDev = -1;
  EXPECT_EQ(flagcxTopoGetLocalNet(topo.get(), 0, &netDev), flagcxSuccess);
  EXPECT_EQ(netDev, 3);
}

} // namespace
