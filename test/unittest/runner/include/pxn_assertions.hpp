#pragma once

#include "coll_proxy_transport.h"
#include "comm.h"
#include "global_comm.h"
#include "graph.h"
#include "net.h"
#include "runner_fixtures.hpp"
#include "transport.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

enum class FlagCXPxnTraffic { AllToAll, RingP2P, ConnectedCollective };

inline uint64_t flagcxRunnerRelayChunks(flagcxComm_t comm) {
  return comm == nullptr || comm->heteroComm == nullptr
             ? 0
             : __atomic_load_n(&comm->heteroComm->pxnRelayChunksCompleted,
                               __ATOMIC_ACQUIRE);
}

// Inspect only the connections used by this operation. P2P uses channel zero;
// collectives use edge-specific internal channels. The hybrid collective
// planner can choose non-neighbor ranks, so inspect its connected cross-cluster
// edges. AllToAll uses every cross-cluster edge. Sample the completion counter
// around each operation so an earlier test cannot satisfy its relay check.
// The connected collective may choose NIC-local inter-ranks and use direct
// NET; validate any relays it does choose without requiring one.
inline void flagcxRunnerAssertPxnTraffic(flagcxComm_t comm, int rank,
                                         int nranks, FlagCXPxnTraffic traffic,
                                         uint64_t relayChunksBefore,
                                         const char *operation) {
  const char *expectPxn = std::getenv("FLAGCX_CI_EXPECT_PXN");
  const int localMode = expectPxn == nullptr               ? -1
                        : std::strcmp(expectPxn, "0") == 0 ? 0
                        : std::strcmp(expectPxn, "1") == 0 ? 1
                                                           : 2;
  int minMode = 0;
  int maxMode = 0;
  MPI_Allreduce(&localMode, &minMode, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&localMode, &maxMode, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  ASSERT_EQ(minMode, maxMode)
      << operation << " received inconsistent PXN expectations across ranks";
  if (minMode == -1)
    return;
  ASSERT_TRUE(minMode == 0 || minMode == 1)
      << operation << " received an invalid FLAGCX_CI_EXPECT_PXN value";

  const bool enabled = minMode == 1;
  const int localReady = comm != nullptr && comm->clusterIds != nullptr &&
                         comm->heteroComm != nullptr &&
                         comm->heteroComm->netAdaptor != nullptr &&
                         (!enabled || comm->heteroComm->topoServer != nullptr);
  int allReady = 0;
  MPI_Allreduce(&localReady, &allReady, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  ASSERT_EQ(allReady, 1) << operation << " PXN assertion setup failed";

  auto *hcomm = comm->heteroComm;
  int localLayoutErrors = 0;
  const int ranksPerGroup = nranks / 2;
  if (enabled) {
    if (nranks < 4 || nranks % 2 != 0) {
      ++localLayoutErrors;
    } else {
      const int firstGroup = comm->clusterIds[0];
      const int secondGroup = comm->clusterIds[ranksPerGroup];
      if (firstGroup == secondGroup)
        ++localLayoutErrors;
      for (int member = 0; member < nranks; ++member) {
        if (comm->clusterIds[member] !=
            (member < ranksPerGroup ? firstGroup : secondGroup))
          ++localLayoutErrors;
      }
    }
  }

  // A relay is only a valid expectation when this rank has a NIC-local GPU
  // reachable through the same topology policy used by the route selector.
  // In particular, a GPU-to-NIC PHB path cannot provide PXN on this host.
  int localRelayEligibleByGroup[2] = {0, 0};
  if (enabled && traffic != FlagCXPxnTraffic::ConnectedCollective &&
      localLayoutErrors == 0) {
    auto *topo = hcomm->topoServer;
    int sourceIndex = -1;
    for (int i = 0; i < topo->nodes[APU].count; ++i) {
      if (topo->nodes[APU].nodes[i].apu.rank == rank) {
        sourceIndex = i;
        break;
      }
    }
    if (sourceIndex < 0 || topo->nodes[NET].count == 0) {
      ++localLayoutErrors;
    } else {
      for (int netIndex = 0; netIndex < topo->nodes[NET].count; ++netIndex) {
        int relayRank = rank;
        if (flagcxTopoSelectPxnRelay(topo, sourceIndex, netIndex, &relayRank,
                                     ranksPerGroup) != flagcxSuccess) {
          ++localLayoutErrors;
          break;
        }
        if (relayRank != rank)
          localRelayEligibleByGroup[rank < ranksPerGroup ? 0 : 1] = 1;
      }
    }
  }

  std::vector<uint64_t> listenerGuids;
  if (enabled) {
    flagcxNetProperties_t listenProps = {};
    if (hcomm->netDev < 0 ||
        hcomm->netAdaptor->getProperties(hcomm->netDev, &listenProps) !=
            flagcxSuccess ||
        listenProps.guid == 0)
      ++localLayoutErrors;
    listenerGuids.resize(nranks);
    const int guidBytes = static_cast<int>(sizeof(listenProps.guid));
    MPI_Allgather(&listenProps.guid, guidBytes, MPI_BYTE, listenerGuids.data(),
                  guidBytes, MPI_BYTE, MPI_COMM_WORLD);
  }

  int localChecked = 0;
  int localNetSends = 0;
  int localConnectionErrors = 0;
  int localRouteErrors = 0;
  int localRelayed = 0;
  int localRelayedByGroup[2] = {0, 0};
  std::map<std::pair<int, uint64_t>, flagcxNetRelayImport *> importsByPool;
  const int nextRank = enabled && traffic == FlagCXPxnTraffic::RingP2P
                           ? (rank + nranks / 2) % nranks
                           : (rank + 1) % nranks;
  const int prevRank = enabled && traffic == FlagCXPxnTraffic::RingP2P
                           ? nextRank
                           : (rank - 1 + nranks) % nranks;
  for (int peer = 0; peer < nranks; ++peer) {
    if (comm->clusterIds[rank] == comm->clusterIds[peer])
      continue;
    const bool allToAll = traffic == FlagCXPxnTraffic::AllToAll;
    const bool collective = traffic == FlagCXPxnTraffic::ConnectedCollective;
    const int sendChannel = traffic == FlagCXPxnTraffic::RingP2P
                                ? 0
                                : flagcxCollProxyChannelForEdge(rank, peer, 0);
    const int recvChannel = traffic == FlagCXPxnTraffic::RingP2P
                                ? 0
                                : flagcxCollProxyChannelForEdge(peer, rank, 0);
    auto *sendPeerState = hcomm->channels[sendChannel].peers[peer];
    auto *recvPeerState = hcomm->channels[recvChannel].peers[peer];
    bool checkSend = allToAll || peer == nextRank;
    bool checkRecv = allToAll || peer == prevRank;
    if (collective) {
      checkSend = sendPeerState != nullptr &&
                  (sendPeerState->send[0].connected ||
                   sendPeerState->send[0].proxyConn.initialized);
      checkRecv = recvPeerState != nullptr &&
                  (recvPeerState->recv[0].connected ||
                   recvPeerState->recv[0].proxyConn.initialized);
    }
    if (!checkSend && !checkRecv)
      continue;

    if (checkSend) {
      ++localChecked;
      auto *peerState = sendPeerState;
      if (peerState == nullptr) {
        ++localConnectionErrors;
      } else {
        const auto &send = peerState->send[0];
        if (!send.connected || !send.proxyConn.initialized ||
            send.proxyConn.transport != TRANSPORT_NET ||
            send.proxyConn.connection == nullptr) {
          ++localConnectionErrors;
        } else if (!enabled) {
          ++localNetSends;
          if (send.proxyConn.tpRank != rank ||
              send.proxyConn.connection != send.proxyConn.remoteConnection)
            ++localRouteErrors;
        } else {
          ++localNetSends;
          const int proxyRank = send.proxyConn.tpRank;
          if (proxyRank < 0 || proxyRank >= nranks) {
            ++localRouteErrors;
          } else if (proxyRank == rank) {
            if (!send.proxyConn.sameProcess ||
                send.proxyConn.connection != send.proxyConn.remoteConnection ||
                send.proxyConn.connection->relayImport != nullptr)
              ++localRouteErrors;
          } else {
            ++localRelayed;
            if (nranks >= 4 && nranks % 2 == 0)
              ++localRelayedByGroup[rank < ranksPerGroup ? 0 : 1];
            if (send.proxyConn.sameProcess ||
                send.proxyConn.connection == send.proxyConn.remoteConnection ||
                comm->clusterIds[proxyRank] != comm->clusterIds[rank] ||
                hcomm->peerInfo[proxyRank].hostHash !=
                    hcomm->peerInfo[rank].hostHash)
              ++localRouteErrors;
            auto *topo = hcomm->topoServer;
            int sourceIndex = -1;
            int relayIndex = -1;
            int netIndex = -1;
            for (int i = 0; i < topo->nodes[APU].count; ++i) {
              if (topo->nodes[APU].nodes[i].apu.rank == rank)
                sourceIndex = i;
              if (topo->nodes[APU].nodes[i].apu.rank == proxyRank)
                relayIndex = i;
            }
            for (int i = 0; i < topo->nodes[NET].count; ++i) {
              if (topo->nodes[NET].nodes[i].net.dev == send.proxyConn.netDev)
                netIndex = i;
            }
            if (sourceIndex < 0 || relayIndex < 0 || netIndex < 0 ||
                topo->nodes[APU].nodes[sourceIndex].paths[APU] == nullptr ||
                topo->nodes[APU].nodes[relayIndex].paths[NET] == nullptr ||
                topo->nodes[APU]
                        .nodes[sourceIndex]
                        .paths[APU][relayIndex]
                        .type > PATH_CCI ||
                topo->nodes[APU].nodes[relayIndex].paths[NET][netIndex].type >
                    PATH_PXB)
              ++localRouteErrors;
            auto *connection = send.proxyConn.connection;
            auto *import = connection->relayImport;
            if (import == nullptr || import->base == nullptr ||
                import->poolId == 0 ||
                connection->relayBufferImport != import->base ||
                connection->relayBufferCapacity != import->capacity ||
                import->capacity != static_cast<size_t>(flagcxNetBufferSize)) {
              ++localRouteErrors;
            } else {
              const auto key = std::make_pair(proxyRank, import->poolId);
              const auto inserted = importsByPool.emplace(key, import);
              if (!inserted.second && inserted.first->second != import)
                ++localRouteErrors;
            }
          }
          if (send.proxyConn.netDev < 0 || send.proxyConn.peerNetDev < 0 ||
              listenerGuids[peer] == 0) {
            ++localRouteErrors;
          } else {
            flagcxNetProperties_t sendProps = {};
            flagcxNetProperties_t recvProps = {};
            if (hcomm->netAdaptor->getProperties(send.proxyConn.netDev,
                                                 &sendProps) != flagcxSuccess ||
                hcomm->netAdaptor->getProperties(send.proxyConn.peerNetDev,
                                                 &recvProps) != flagcxSuccess ||
                sendProps.guid == 0 || recvProps.guid != listenerGuids[peer])
              ++localRouteErrors;
            // Different NICs are allowed when the single-host test simulates
            // two nodes. A successful NET transfer validates that NIC pair.
          }
        }
      }
    }
    if (checkRecv) {
      ++localChecked;
      auto *peerState = recvPeerState;
      if (peerState == nullptr) {
        ++localConnectionErrors;
      } else {
        const auto &recv = peerState->recv[0];
        if (!recv.connected || !recv.proxyConn.initialized ||
            recv.proxyConn.transport != TRANSPORT_NET ||
            recv.proxyConn.connection == nullptr) {
          ++localConnectionErrors;
        } else if (recv.proxyConn.tpRank != rank) {
          ++localRouteErrors;
        }
      }
    }
  }

  const int localCompletionErrors =
      enabled && localRelayed > 0 &&
      flagcxRunnerRelayChunks(comm) <= relayChunksBefore;
  int globalLayoutErrors = 0;
  int globalChecked = 0;
  int globalNetSends = 0;
  int globalConnectionErrors = 0;
  int globalRouteErrors = 0;
  int globalRelayed = 0;
  int globalRelayedByGroup[2] = {0, 0};
  int globalRelayEligibleByGroup[2] = {0, 0};
  int globalCompletionErrors = 0;
  MPI_Allreduce(&localLayoutErrors, &globalLayoutErrors, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(&localChecked, &globalChecked, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(&localNetSends, &globalNetSends, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(&localConnectionErrors, &globalConnectionErrors, 1, MPI_INT,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&localRouteErrors, &globalRouteErrors, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(&localRelayed, &globalRelayed, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(localRelayedByGroup, globalRelayedByGroup, 2, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(localRelayEligibleByGroup, globalRelayEligibleByGroup, 2,
                MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&localCompletionErrors, &globalCompletionErrors, 1, MPI_INT,
                MPI_SUM, MPI_COMM_WORLD);
  EXPECT_EQ(globalLayoutErrors, 0) << operation << " has an invalid PXN layout";
  EXPECT_GT(globalChecked, 0) << operation << " used no cross-cluster NET edge";
  EXPECT_GT(globalNetSends, 0)
      << operation << " established no cross-cluster NET send";
  EXPECT_EQ(globalConnectionErrors, 0)
      << operation << " did not establish the expected NET connections";
  EXPECT_EQ(globalRouteErrors, 0)
      << operation << " selected an invalid NET NIC or proxy rank";
  EXPECT_EQ(globalCompletionErrors, 0)
      << operation << " has no completed relay chunk on a relaying source";
  if (enabled && traffic != FlagCXPxnTraffic::ConnectedCollective) {
    for (int group = 0; group < 2; ++group) {
      if (globalRelayEligibleByGroup[group] > 0) {
        EXPECT_GT(globalRelayedByGroup[group], 0)
            << operation << " did not relay from logical node " << group;
      } else {
        EXPECT_EQ(globalRelayedByGroup[group], 0)
            << operation
            << " relayed without an eligible topology on logical node "
            << group;
        if (rank == 0)
          std::fprintf(stderr,
                       "%s PXN relay coverage skipped on logical node %d: "
                       "no eligible GPU-to-NIC path; direct NET fallback "
                       "and data correctness were checked\n",
                       operation, group);
      }
    }
  } else if (!enabled) {
    EXPECT_EQ(globalRelayed, 0) << operation << " unexpectedly used PXN";
  }
}
