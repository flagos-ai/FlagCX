#pragma once

#include "coll_proxy_transport.h"
#include "comm.h"
#include "global_comm.h"
#include "net.h"
#include "runner_fixtures.hpp"
#include "transport.h"

#include <cstdlib>
#include <cstring>
#include <vector>

enum class FlagCXPxnTraffic { AllToAll, RingP2P, ConnectedCollective };

inline uint64_t flagcxRunnerRelayChunks(flagcxComm_t comm) {
  return comm == nullptr || comm->heteroComm == nullptr
             ? 0
             : __atomic_load_n(&comm->heteroComm->pxnRelayChunksCompleted,
                               __ATOMIC_ACQUIRE);
}

// Inspect only the connections used by this operation. RingP2P uses channel
// zero; collectives use edge-specific internal channels. The hybrid collective
// planner can choose non-neighbor ranks, so inspect its connected cross-cluster
// edges. AllToAll uses every cross-cluster edge. Sample the completion counter
// around each operation so an earlier test cannot satisfy its relay check.
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
                         comm->heteroComm->netAdaptor != nullptr;
  int allReady = 0;
  MPI_Allreduce(&localReady, &allReady, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  ASSERT_EQ(allReady, 1) << operation << " PXN assertion setup failed";

  auto *hcomm = comm->heteroComm;
  int localLayoutErrors = 0;
  if (enabled) {
    if (nranks != 8) {
      ++localLayoutErrors;
    } else {
      const int firstGroup = comm->clusterIds[0];
      const int secondGroup = comm->clusterIds[4];
      if (firstGroup == secondGroup)
        ++localLayoutErrors;
      for (int member = 0; member < 8; ++member) {
        if (comm->clusterIds[member] != (member < 4 ? firstGroup : secondGroup))
          ++localLayoutErrors;
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
  int localConnectionErrors = 0;
  int localRouteErrors = 0;
  int localRelayed = 0;
  const int nextRank = (rank + 1) % nranks;
  const int prevRank = (rank - 1 + nranks) % nranks;
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
          if (send.proxyConn.tpRank != rank ||
              send.proxyConn.connection != send.proxyConn.remoteConnection)
            ++localRouteErrors;
        } else {
          const int proxyRank = send.proxyConn.tpRank;
          if (proxyRank < 0 || proxyRank >= nranks) {
            ++localRouteErrors;
          } else if (proxyRank != rank) {
            ++localRelayed;
            if (send.proxyConn.sameProcess ||
                send.proxyConn.connection == send.proxyConn.remoteConnection ||
                hcomm->peerInfo[proxyRank].hostHash !=
                    hcomm->peerInfo[rank].hostHash)
              ++localRouteErrors;
          }
          if (send.proxyConn.netDev < 0 || send.proxyConn.peerNetDev < 0 ||
              listenerGuids[peer] == 0) {
            ++localRouteErrors;
          } else {
            flagcxNetProperties_t sendProps = {};
            if (hcomm->netAdaptor->getProperties(send.proxyConn.netDev,
                                                 &sendProps) != flagcxSuccess ||
                sendProps.guid != listenerGuids[peer])
              ++localRouteErrors;
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
  int globalConnectionErrors = 0;
  int globalRouteErrors = 0;
  int globalRelayed = 0;
  int globalCompletionErrors = 0;
  MPI_Allreduce(&localLayoutErrors, &globalLayoutErrors, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(&localChecked, &globalChecked, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(&localConnectionErrors, &globalConnectionErrors, 1, MPI_INT,
                MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&localRouteErrors, &globalRouteErrors, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(&localRelayed, &globalRelayed, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(&localCompletionErrors, &globalCompletionErrors, 1, MPI_INT,
                MPI_SUM, MPI_COMM_WORLD);
  EXPECT_EQ(globalLayoutErrors, 0) << operation << " has an invalid PXN layout";
  EXPECT_GT(globalChecked, 0) << operation << " used no cross-cluster NET edge";
  EXPECT_EQ(globalConnectionErrors, 0)
      << operation << " did not establish the expected NET connections";
  EXPECT_EQ(globalRouteErrors, 0)
      << operation << " selected an invalid NET NIC or proxy rank";
  EXPECT_EQ(globalCompletionErrors, 0)
      << operation << " has no completed relay chunk on a relaying source";
  if (enabled)
    EXPECT_GT(globalRelayed, 0) << operation << " did not use a PXN relay";
  else
    EXPECT_EQ(globalRelayed, 0) << operation << " unexpectedly used PXN";
}
