// AlltoAll correctness test.
// Each rank fills sendbuff so that chunk i contains data identifying the
// sender. After AlltoAll, each rank verifies it received the correct data from
// each peer.

#include "coll_proxy_transport.h"
#include "net.h"
#include "pxn_assertions.hpp"
#include "runner_fixtures.hpp"
#include "test_utils.hpp"
#include "transport.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

TEST_F(FlagCXCollTest, AlltoAll) {

  size_t countPerRank = count / nranks;
  const uint64_t relayChunksBefore = flagcxRunnerRelayChunks(comm);

  // Fill sendbuff: chunk[i] = rank * nranks + i (so receiver can verify sender)
  float *hsend = static_cast<float *>(hostsendbuff);
  for (int i = 0; i < nranks; i++) {
    for (size_t j = 0; j < countPerRank; j++) {
      hsend[i * countPerRank + j] = static_cast<float>(rank * nranks + i);
    }
  }

  devHandle->deviceMemcpy(sendbuff, hostsendbuff, size,
                          flagcxMemcpyHostToDevice, stream);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxAlltoAll(sendbuff, recvbuff, countPerRank, flagcxFloat, comm, stream);

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size,
                          flagcxMemcpyDeviceToHost, stream);
  ASSERT_EQ(synchronizeAndCheckAsyncError(), flagcxSuccess);

  const char *expectMultiChannel =
      std::getenv("FLAGCX_CI_EXPECT_COLL_MULTICHANNEL");
  if (expectMultiChannel != nullptr &&
      std::strcmp(expectMultiChannel, "1") == 0) {
    ASSERT_NE(comm->heteroComm, nullptr);
    uint64_t localChannelMask = 0;
    uint64_t localLaneMask = 0;
    int localChecked = 0;
    int localMismatches = 0;
    const bool expectIb =
        comm->heteroComm->netAdaptor != nullptr &&
        std::strcmp(comm->heteroComm->netAdaptor->name, "IB") == 0;
    const bool expectStriping =
        std::getenv("FLAGCX_CI_EXPECT_COLL_QP_STRIPING") != nullptr;

    for (int peer = 0; peer < nranks; ++peer) {
      if (comm->clusterIds[rank] == comm->clusterIds[peer])
        continue;
      const int sendChannel = flagcxCollProxyChannelForEdge(rank, peer, 0);
      const int recvChannel = flagcxCollProxyChannelForEdge(peer, rank, 0);
      localChannelMask |= 1ULL << sendChannel;
      localChannelMask |= 1ULL << recvChannel;

      flagcxConnector *connectors[2] = {
          &comm->heteroComm->channels[sendChannel].peers[peer]->send[0],
          &comm->heteroComm->channels[recvChannel].peers[peer]->recv[0],
      };
      for (flagcxConnector *connector : connectors) {
        localChecked++;
        if (!connector->connected || !connector->proxyConn.initialized ||
            connector->proxyConn.transport != TRANSPORT_NET ||
            connector->proxyConn.connection == nullptr) {
          localMismatches++;
          continue;
        }
        if (expectIb) {
          auto *connection = connector->proxyConn.connection;
          const uint64_t mask =
              __atomic_load_n(&connection->collDataLaneMask, __ATOMIC_RELAXED);
          if (mask == 0) {
            localMismatches++;
          } else {
            localLaneMask |= mask;
            if (expectStriping && __builtin_popcountll(mask) < 2)
              localMismatches++;
            if (!expectStriping && __builtin_popcountll(mask) != 1)
              localMismatches++;
          }
        }
      }
    }

    uint64_t globalChannelMask = 0;
    uint64_t globalLaneMask = 0;
    int globalChecked = 0;
    int globalMismatches = 0;
    MPI_Allreduce(&localChannelMask, &globalChannelMask, 1, MPI_UINT64_T,
                  MPI_BOR, MPI_COMM_WORLD);
    MPI_Allreduce(&localLaneMask, &globalLaneMask, 1, MPI_UINT64_T, MPI_BOR,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&localChecked, &globalChecked, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&localMismatches, &globalMismatches, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    EXPECT_GT(globalChecked, 0);
    EXPECT_GE(__builtin_popcountll(globalChannelMask), 2)
        << "collective did not establish multiple internal channels";
    if (expectIb)
      EXPECT_GE(__builtin_popcountll(globalLaneMask), 2)
          << "collective did not exercise multiple physical QPs";
    EXPECT_EQ(globalMismatches, 0);
  }

  flagcxRunnerAssertPxnTraffic(comm, rank, nranks, FlagCXPxnTraffic::AllToAll,
                               relayChunksBefore, "AllToAll");

  MPI_Barrier(MPI_COMM_WORLD);

  // Verify: chunk[i] in recvbuff should be data from rank i,
  // which sent rank i's chunk destined for us = i * nranks + rank
  float *hrecv = static_cast<float *>(hostrecvbuff);
  bool success = true;
  for (int i = 0; i < nranks; i++) {
    float expected = static_cast<float>(i * nranks + rank);
    for (size_t j = 0; j < countPerRank; j++) {
      size_t idx = i * countPerRank + j;
      if (hrecv[idx] != expected) {
        if (rank == 0) {
          std::cout << "Mismatch at chunk " << i << " index " << j
                    << ": expected " << expected << ", got " << hrecv[idx]
                    << std::endl;
        }
        success = false;
        break;
      }
    }
    if (!success)
      break;
  }
  EXPECT_TRUE(success);
}
