/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_P2P_TOPO_H_
#define FLAGCX_P2P_TOPO_H_

#include "flagcx_net_adaptor.h"
#include "topo.h"

struct flagcxP2pTopoManager {
  struct flagcxTopoServer *topoServer;
  int nGpus; // number of visible GPUs
};

// Initialize the P2P topo manager. Enumerates all local GPUs and NICs,
// builds a node-scoped topology graph, and computes paths for NIC selection.
// netAdaptor: the network adaptor used for NIC enumeration.
flagcxResult_t flagcxP2pTopoInit(struct flagcxNetAdaptor *netAdaptor,
                                 struct flagcxP2pTopoManager **mgr);

// Query: given a local GPU device index, return the best net device index.
flagcxResult_t flagcxP2pTopoGetNetDev(struct flagcxP2pTopoManager *mgr,
                                      int gpuDev, int *netDev);

// Free all resources owned by the manager.
flagcxResult_t flagcxP2pTopoDestroy(struct flagcxP2pTopoManager *mgr);

#endif // FLAGCX_P2P_TOPO_H_
