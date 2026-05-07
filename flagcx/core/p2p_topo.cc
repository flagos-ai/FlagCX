/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "p2p_topo.h"
#include "adaptor.h"
#include "flagcx_net.h"
#include "utils.h"
#include "xml.h"

// Build the XML topology for all local GPUs and NICs without a communicator.
// Mirrors the logic in flagcxTopoGetXmlTopo (topo.cc) but uses deviceAdaptor
// for GPU enumeration and the provided netAdaptor for NIC enumeration.
static flagcxResult_t flagcxP2pTopoBuildXml(struct flagcxNetAdaptor *netAdaptor,
                                            struct flagcxXml *xml,
                                            int *nGpusOut) {
  // Create root node
  struct flagcxXmlNode *top;
  FLAGCXCHECK(xmlAddNode(xml, NULL, "system", &top));
  FLAGCXCHECK(xmlSetAttrInt(top, "version", FLAGCX_TOPO_XML_VERSION));

  // Enumerate local GPUs
  int nGpus = 0;
  FLAGCXCHECK(deviceAdaptor->getDeviceCount(&nGpus));
  INFO(FLAGCX_INIT, "P2P topo: detected %d GPUs", nGpus);

  for (int i = 0; i < nGpus; i++) {
    char busId[FLAGCX_DEVICE_PCI_BUSID_BUFFER_SIZE];
    FLAGCXCHECK(deviceAdaptor->getDevicePciBusId(busId, sizeof(busId), i));

    struct flagcxXmlNode *node;
    FLAGCXCHECK(flagcxTopoFillApu(xml, busId, &node));
    if (node == NULL) {
      continue;
    }

    int devLogicalIdx = 0;
    FLAGCXCHECK(deviceAdaptor->getDeviceByPciBusId(&devLogicalIdx, busId));
    FLAGCXCHECK(xmlSetAttrInt(node, "dev", devLogicalIdx));
    FLAGCXCHECK(xmlSetAttrInt(node, "rank", i));
  }

  // Enumerate NICs
  int netDevCount = 0;
  FLAGCXCHECK(netAdaptor->devices(&netDevCount));
  INFO(FLAGCX_INIT, "P2P topo: detected %d NICs", netDevCount);

  for (int n = 0; n < netDevCount; n++) {
    flagcxNetProperties_t props;
    FLAGCXCHECK(netAdaptor->getProperties(n, (void *)&props));

    struct flagcxXmlNode *netNode;
    FLAGCXCHECK(flagcxTopoFillNet(xml, props.pciPath, props.name, &netNode));
    FLAGCXCHECK(xmlSetAttrInt(netNode, "dev", n));
    FLAGCXCHECK(xmlSetAttrInt(netNode, "speed", props.speed));
    FLAGCXCHECK(xmlSetAttrFloat(netNode, "latency", props.latency));
    FLAGCXCHECK(xmlSetAttrInt(netNode, "port", props.port));
    FLAGCXCHECK(xmlInitAttrUint64(netNode, "guid", props.guid));
    FLAGCXCHECK(xmlSetAttrInt(netNode, "maxConn", props.maxComms));
  }

  *nGpusOut = nGpus;
  return flagcxSuccess;
}

flagcxResult_t flagcxP2pTopoInit(struct flagcxNetAdaptor *netAdaptor,
                                 struct flagcxP2pTopoManager **mgr) {
  struct flagcxP2pTopoManager *m;
  FLAGCXCHECK(flagcxCalloc(&m, 1));

  // Build XML topology
  struct flagcxXml *xml;
  FLAGCXCHECK(xmlAlloc(&xml, FLAGCX_TOPO_XML_MAX_NODES));

  int nGpus = 0;
  FLAGCXCHECK(flagcxP2pTopoBuildXml(netAdaptor, xml, &nGpus));

  // Convert XML to server topology
  uint64_t localHostHash = getHostHash();
  FLAGCXCHECK(
      flagcxTopoGetServerTopoFromXml(xml, &m->topoServer, localHostHash));

  free(xml);

  // Compute shortest paths between all nodes
  FLAGCXCHECK(flagcxTopoComputePaths(m->topoServer, NULL));

  m->nGpus = nGpus;
  *mgr = m;

  INFO(FLAGCX_INIT, "P2P topo manager initialized: %d GPUs", nGpus);
  return flagcxSuccess;
}

flagcxResult_t flagcxP2pTopoGetNetDev(struct flagcxP2pTopoManager *mgr,
                                      int gpuDev, int *netDev) {
  if (mgr == NULL || netDev == NULL) {
    return flagcxInvalidArgument;
  }
  if (gpuDev < 0 || gpuDev >= mgr->nGpus) {
    WARN("P2P topo: gpuDev %d out of range [0, %d)", gpuDev, mgr->nGpus);
    return flagcxInvalidArgument;
  }
  // gpuDev is used directly as the synthetic rank
  FLAGCXCHECK(flagcxTopoGetLocalNet(mgr->topoServer, gpuDev, netDev));
  return flagcxSuccess;
}

flagcxResult_t flagcxP2pTopoDestroy(struct flagcxP2pTopoManager *mgr) {
  if (mgr == NULL) {
    return flagcxSuccess;
  }
  free(mgr->topoServer);
  free(mgr);
  return flagcxSuccess;
}
