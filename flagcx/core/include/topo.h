/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef FLAGCX_TOPO_H_
#define FLAGCX_TOPO_H_

#include "core.h"
#include "graph.h"
#include <unordered_map>
#include <unordered_set>

#define LOC_BW 5000.0
#define SM60_NVLINK_BW 18.0
#define SM70_NVLINK_BW 20.0
#define SM80_NVLINK_BW 20.0
#define SM90_NVLINK_BW 20.6
#define SM86_NVLINK_BW 12.0
#define PCI_BW 12.0 // PCI Gen3 x16
#define QPI_BW 6.0
#define AMD_BW 16.0
#define SKL_QPI_BW 10.0
#define ZPI_BW 6.0
#define YONGFENG_ZPI_BW 9.0
#define P9_BW 32.0
#define ARM_BW 6.0
#define NET_BW 12.0 // 100Gbit

// Intel CPU convert GPU P2P traffic into 64B PCI TLPs, so GPU
// to GPU traffic consumes more PCI bandwidth.
#define INTEL_P2P_OVERHEAD(bw) (bw * 6 / 5)

#define FLAGCX_TOPO_NODE_TYPES 7
#define APU 0
#define PCI 1
#define CCI 2
#define CPU 3 // Actually NUMA domains
#define NIC 4
#define NET 5
#define HBD 6
extern const char *topoNodeTypeStr[];

// We want link types and path types to match as much as possible
#define LINK_LOC 0
#define LINK_CCI 1
// Skipping 2 for PATH_NVB
#define LINK_PCI 3
// Skipping 4 for PATH_PXB
// Skipping 5 for PATH_PXN
// Skipping 6 for PATH_PHB
#define LINK_SYS 7
#define LINK_NET 8
extern const char *topoLinkTypeStr[];

// Local (myself)
#define PATH_LOC 0

// Connection traversing CCI link
#define PATH_CCI 1

// Connection through CCI link using an intermediate APU
#define PATH_CCB 2

// Connection traversing at most a single PCIe bridge
#define PATH_PIX 3

// Connection traversing multiple PCIe bridges (without traversing the PCIe Host
// Bridge)
#define PATH_PXB 4

// Connection between a GPU and a NIC using an intermediate GPU. Used to enable
// rail-local, aggregated network send/recv operations.
#define PATH_PXN 5

// Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
#define PATH_PHB 6

// Connection traversing PCIe as well as the SMP interconnect between NUMA nodes
// (e.g., QPI/UPI)
#define PATH_SYS 7

// Connection through the network
#define PATH_NET 8

// Disconnected
#define PATH_DIS 9
extern const char *topoPathTypeStr[];

struct flagcxTopoNode;
struct flagcxTopoLink {
  int type;
  float bw;
  struct flagcxTopoNode *remNode;
};
#define FLAGCX_TOPO_MAX_LINKS 128
#define FLAGCX_TOPO_MAX_HOPS (FLAGCX_TOPO_MAX_NODES * FLAGCX_TOPO_NODE_TYPES)
#define FLAGCX_MAX_INTER_SERVER_HOPS                                           \
  16 // TODO: decide on a decent number for this variable
#define FLAGCX_MAX_SERVER_NUM                                                  \
  16 // TODO: decide on a decent number for this variable

struct flagcxTopoPath {
  struct flagcxTopoLink *list[FLAGCX_TOPO_MAX_HOPS];
  int count;
  float bw;
  int type;
};

#define FLAGCX_TOPO_CPU_INTEL_BDW 1
#define FLAGCX_TOPO_CPU_INTEL_SKL 2

#define FLAGCX_TOPO_UNDEF (-1)

#define FLAGCX_TOPO_ID_SERVER_ID(id) (id >> 56)
#define FLAGCX_TOPO_ID_LOCAL_ID(id) (id & 0x00ffffffffffffff)
#define FLAGCX_TOPO_ID(serverid, localid) (((int64_t)serverid << 56) + localid)

struct flagcxTopoNode {
  int type;
  int64_t id;
  // Type specific data
  union {
    struct {
      int dev; // NVML dev number
      int rank;
      int gdrSupport;
      int vendor;
      int interServerRouteCnt;
      struct flagcxInterServerRoute *interServerRoutes[FLAGCX_TOPO_MAX_NODES];
    } apu;
    struct {
      int dev; // Plugin dev number
      uint64_t guid;
      int port;
      int ip;
      float bw;
      float latency;
      int gdrSupport;
      int maxConn;
      // int64_t guid;
    } net;
    struct {
      int arch;
      int vendor;
      int model;
      cpu_set_t affinity;
    } cpu;
    struct {
      uint64_t device;
    } pci;
  };
  int nlinks;
  struct flagcxTopoLink links[FLAGCX_TOPO_MAX_LINKS];
  // Pre-computed paths to GPUs and NICs
  struct flagcxTopoPath *paths[FLAGCX_TOPO_NODE_TYPES];
  // Used during search
  uint64_t used;
};

struct flagcxTopoNodeSet {
  int count;
  struct flagcxTopoNode nodes[FLAGCX_TOPO_MAX_NODES];
};

struct flagcxTopoServer {
  int serverId;
  uint64_t hostHashes[FLAGCX_TOPO_MAX_NODES];
  int nHosts;
  struct flagcxTopoNodeSet nodes[FLAGCX_TOPO_NODE_TYPES];
  float maxBw;
  float totalBw;
};

struct flagcxSwitch {
  float downBw;
  float upBw;
  int upLink;
  int downLink;
  bool isTop;
};

// inter-server topo sturcture might need to be changed
struct flagcxInterServerRoute {
  int switchCount;
  struct flagcxTopoNode *localNic;
  struct flagcxTopoNode *remoteNic;
  int remoteRank;
  float interBw;
  struct flagcxSwitch switchInfos[FLAGCX_MAX_INTER_SERVER_HOPS];
};

struct flagcxInterServerTopo {
  int numServers;
  struct flagcxTopoServer
      *servers; // contain topology of all servers except current server, topo
                // of current server is stored in comm->topoServer
  std::unordered_map<uint64_t, int>
      netToServerMap; // {{netGuid, serverId}, ...}
  std::unordered_map<
      uint64_t, std::unordered_map<uint64_t, struct flagcxInterServerRoute *>>
      routeMap; // {{localNetGuid, {remoteNetGuid, route}}, ...}
  char interServerTopoFile[256];
};

struct topoArgs {
  int rank;
  int nranks;
  flagcxUniqueId uniqueId;
  void *bootstrap;
};

struct flagcxDevProps {
  char name[256];
  int pciBusId;
  int pciDeviceId;
  int pciDomainId;
  // remove unused field for now
  // int gdrSupported;
};

struct flatTopoLink {
  int type;
  float bw;
  int remNodeIdx;
  int remNodeType;
};

struct flatTopoNode {
  int type;
  int64_t id;
  // Type specific data
  union {
    struct {
      int dev; // NVML dev number
      int rank;
      int vendor;
    } apu;
    struct {
      int dev; // Plugin dev number
      int port;
      float bw;
      float latency;
      int maxConn;
      uint64_t guid;
    } net;
    struct {
      int arch;
      int vendor;
      int model;
    } cpu;
    struct {
      uint64_t device;
    } pci;
  };
  int nlinks;
  struct flatTopoLink links[FLAGCX_TOPO_MAX_LINKS];
};

struct flatTopoNodeSet {
  int count;
  struct flatTopoNode nodes[FLAGCX_TOPO_MAX_NODES];
};

struct flatTopoServer {
  int serverId;
  uint64_t hostHashes[FLAGCX_TOPO_MAX_NODES];
  int nHosts;
  struct flatTopoNodeSet nodes[FLAGCX_TOPO_NODE_TYPES];
};

struct flagcxNicDistance {
  int distance;
  uint64_t netGuid;
};

flagcxResult_t flagcxTopoGetNode(struct flagcxTopoServer *topoServer,
                                 struct flagcxTopoNode **node, int type,
                                 uint64_t id);
flagcxResult_t flagcxTopoCreateNode(struct flagcxTopoServer *topoServer,
                                    struct flagcxTopoNode **node, int type,
                                    uint64_t id);
flagcxResult_t flagcxTopoRemoveNode(struct flagcxTopoServer *topoServer,
                                    int type, int id);
flagcxResult_t flagcxTopoConnectNodes(struct flagcxTopoNode *node,
                                      struct flagcxTopoNode *remNode, int type,
                                      float bw);
flagcxResult_t flagcxTopoPrintPaths(struct flagcxTopoServer *topoServer);
flagcxResult_t flagcxTopoLoadServer(const char *xmlTopoFile,
                                    struct flagcxTopoServer *topoServer);
flagcxResult_t
flagcxTopoGetIntermediateRank(struct flagcxTopoServer *topoServer, int rank,
                              int64_t netId, int *intermediateRank);

flagcxResult_t flagcxTopoPrint(struct flagcxTopoServer *topoServer);

flagcxResult_t flagcxTopoPrintPaths(struct flagcxTopoServer *topoServer);

flagcxResult_t
flagcxGetInterServerTopo(struct flagcxHeteroComm *comm,
                         struct flagcxInterServerTopo **interServerTopo,
                         struct flagcxTopoServer *topoServer);

#define FLAGCX_TOPO_XML_MAX_NODES 256
#define FLAGCX_GRAPH_XML_MAX_NODES 4096
flagcxResult_t
flagcxTopoGetServerTopoFromXml(struct flagcxXml *xml,
                               struct flagcxTopoServer **topoServer,
                               uint64_t localHostHash);
flagcxResult_t flagcxTopoGetGraphFromXml(struct flagcxXmlNode *xmlGraphs,
                                         struct flagcxTopoServer *topoServer,
                                         struct flagcxTopoGraph *graph,
                                         int *nChannels);
flagcxResult_t flagcxTopoGetXmlFromGraphs(int ngraphs,
                                          struct flagcxTopoGraph **graphs,
                                          struct flagcxTopoServer *topoServer,
                                          struct flagcxXml *xml);
flagcxResult_t flagcxTopoGetXmlTopo(struct flagcxHeteroComm *comm,
                                    struct flagcxXml *xml);
flagcxResult_t flagcxTopoGetServerTopo(struct flagcxHeteroComm *comm,
                                       struct flagcxTopoServer **topoServer);

flagcxResult_t flagcxTopoGetCompCap(struct flagcxTopoServer *topoServer,
                                    int *ccMin, int *ccMax);

flagcxResult_t flagcxGetNicDistance(struct flagcxTopoServer *topoServer,
                                    int rank,
                                    struct flagcxNicDistance *distInfo);
flagcxResult_t flagcxTopoGetLocalNetNode(struct flagcxTopoServer *topoServer,
                                         int rank,
                                         struct flagcxTopoNode **netNode);

flagcxResult_t
flagcxTopoGetServerFromRank(int rank, struct flagcxInterServerTopo *interServer,
                            struct flagcxTopoServer *currServer,
                            struct flagcxTopoServer **retServer);

// static flagcxResult_t flagcxTopoIdToIndex(struct flagcxTopoServer*
// serverTopo, int type, int64_t id, int* index) {
//   *index = -1;
//   for (int i=0; i<serverTopo->nodes[type].count; i++) {
//     if (serverTopo->nodes[type].nodes[i].id == id) {
//       *index = i;
//       return flagcxSuccess;
//     }
//   }
//   return flagcxInternalError;
// }

// static flagcxResult_t flagcxTopoRankToIndex(struct flagcxTopoServer*
// serverTopo, int rank, int* index) {
//   *index = -1;
//   for (int i=0; i<serverTopo->nodes[GPU].count; i++) {
//     if (serverTopo->nodes[GPU].nodes[i].apu.rank == rank) {
//       *index = i;
//       return flagcxSuccess;
//     }
//   }
//   return flagcxInternalError;
// }

// static flagcxResult_t flagcxTopoDevToRank(struct flagcxTopoServer*
// serverTopo, int dev, int* rank) {
//   *rank = -1;
//   for (int i=0; i<serverTopo->nodes[GPU].count; i++) {
//     if (FLAGCX_TOPO_ID_SERVER_ID(serverTopo->nodes[GPU].nodes[i].id) !=
//     serverTopo->serverId) continue; // Only consider GPUs on our node if
//     (serverTopo->nodes[GPU].nodes[i].apu.dev == dev) {
//       *rank = serverTopo->nodes[GPU].nodes[i].apu.rank;
//       return flagcxSuccess;
//     }
//   }
//   return flagcxInternalError;
// }

// static flagcxResult_t flagcxTopoIdToNetDev(struct flagcxTopoServer*
// serverTopo, int64_t id, int* netDev) {
//   *netDev = -1;
//   for (int i=0; i<serverTopo->nodes[NET].count; i++) {
//     if (serverTopo->nodes[NET].nodes[i].id == id) {
//       *netDev = serverTopo->nodes[NET].nodes[i].net.dev;
//       return flagcxSuccess;
//     }
//   }
//   WARN("Could not find NET with id %lx\n", id);
//   return flagcxInternalError;
// }

// // Returns NVLink bw in GB/s
// static float flagcxTopoNVLinkBw(int cudaCompCap) {
//   return
//     cudaCompCap >= 90 ? SM90_NVLINK_BW :
//     cudaCompCap == 86 ? SM86_NVLINK_BW :
//     cudaCompCap >= 80 ? SM80_NVLINK_BW :
//     cudaCompCap >= 70 ? SM70_NVLINK_BW :
//     cudaCompCap >= 60 ? SM60_NVLINK_BW :
//     SM80_NVLINK_BW;
// }

// Mirror bits
static bool isPow2(int val) { return (val & (val - 1)) == 0; }
static int mirrorBits(int val, int pow2) {
  int mirror = 0;
  for (int b = 1, mb = (pow2 >> 1); b < pow2; b <<= 1, mb >>= 1)
    if (val & b)
      mirror |= mb;
  return mirror;
}

#ifdef CREATE_DEVICE_TOPO_API
#define DEVICE_TOPO_API_EXTERN
#else
#define DEVICE_TOPO_API_EXTERN extern
#endif

// DEVICE_TOPO_API_EXTERN flagcxResult_t (*flagcxTopoGetLocalNet)(int gpu,
//                                                                char *name);

#endif
