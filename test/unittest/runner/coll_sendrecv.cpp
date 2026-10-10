// Point-to-point Send/Recv correctness test.
// Ring pattern: each rank sends to (rank+1)%nranks, receives from
// (rank-1+nranks)%nranks. Each rank fills its sendbuff with its own rank ID.
// After the exchange, each rank verifies it received the sender's rank ID.

#include "comm.h"
#include "global_comm.h"
#include "net.h"
#include "p2p.h"
#include "pxn_assertions.hpp"
#include "runner_fixtures.hpp"
#include "runner_result.h"
#include "test_utils.hpp"
#include "transport.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

TEST_F(FlagCXCollTest, SendRecv) {

  int sendPeer = (rank + 1) % nranks;
  int recvPeer = (rank - 1 + nranks) % nranks;

  // Fill sendbuff with my rank
  float *hsend = static_cast<float *>(hostsendbuff);
  for (size_t i = 0; i < count; i++) {
    hsend[i] = static_cast<float>(rank);
  }

  devHandle->deviceMemcpy(sendbuff, hostsendbuff, size,
                          flagcxMemcpyHostToDevice, stream);

  MPI_Barrier(MPI_COMM_WORLD);
  const uint64_t relayChunksBefore = flagcxRunnerRelayChunks(comm);

  // Use group API for concurrent send/recv
  flagcxResult_t groupResult = flagcxGroupStart(comm);
  if (flagcxRunnerResultAccepted(groupResult)) {
    groupResult =
        flagcxSend(sendbuff, count, flagcxFloat, sendPeer, comm, stream);
    if (flagcxRunnerResultAccepted(groupResult)) {
      groupResult =
          flagcxRecv(recvbuff, count, flagcxFloat, recvPeer, comm, stream);
    }
    flagcxResult_t groupEndResult = flagcxGroupEnd(comm);
    flagcxResult_t firstError = flagcxSuccess;
    flagcxRunnerRecordFirstError(groupResult, &firstError);
    flagcxRunnerRecordFirstError(groupEndResult, &firstError);
    groupResult = firstError;
  }
  int localGroupResult = static_cast<int>(groupResult);
  int globalGroupResult = static_cast<int>(flagcxSuccess);
  MPI_Allreduce(&localGroupResult, &globalGroupResult, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  ASSERT_EQ(globalGroupResult, static_cast<int>(flagcxSuccess));

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size,
                          flagcxMemcpyDeviceToHost, stream);
  ASSERT_EQ(synchronizeAndCheckAsyncError(), flagcxSuccess);

  flagcxRunnerAssertPxnTraffic(comm, rank, nranks, FlagCXPxnTraffic::RingP2P,
                               relayChunksBefore, "SendRecv");

  // The heterogeneous CI modes must prove which peer transport was actually
  // connected.  FLAGCX_P2P_DISABLE selects NET, but this assertion catches an
  // ignored setting or an unintended transport fallback in the runner itself.
  const char *expectedTransport =
      std::getenv("FLAGCX_CI_EXPECT_PEER_TRANSPORT");
  if (expectedTransport != nullptr && expectedTransport[0] != '\0') {
    int localChecked = 0;
    int localMismatches = 0;
    if (comm->commType == flagcxCommunicatorHybrid &&
        comm->heteroComm != nullptr) {
      const bool expectP2p = std::strcmp(expectedTransport, "P2P") == 0;
      const bool expectNet = std::strcmp(expectedTransport, "NET") == 0;
      auto checkConnector = [&](flagcxConnector *connector) {
        localChecked++;
        bool matches = connector->connected &&
                       connector->proxyConn.initialized &&
                       connector->proxyConn.connection != nullptr;
        if (matches && expectP2p) {
          matches = connector->proxyConn.transport == TRANSPORT_P2P;
        } else if (matches && expectNet) {
          matches = connector->proxyConn.transport == TRANSPORT_NET;
        } else if (!expectP2p && !expectNet) {
          matches = false;
        }
        if (!matches)
          localMismatches++;
      };

      // Only cross-cluster peers use the heterogeneous transport. Same-cluster
      // ring edges are handled by the homogeneous CCL and intentionally do not
      // establish P2P/NET connectors.
      if (comm->clusterIds[rank] != comm->clusterIds[sendPeer]) {
        checkConnector(&comm->heteroComm->channels[0].peers[sendPeer]->send[0]);
      }
      if (comm->clusterIds[rank] != comm->clusterIds[recvPeer]) {
        checkConnector(&comm->heteroComm->channels[0].peers[recvPeer]->recv[0]);
      }
    }
    int globalChecked = 0;
    int globalMismatches = 0;
    MPI_Allreduce(&localChecked, &globalChecked, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&localMismatches, &globalMismatches, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    ASSERT_GT(globalChecked, 0)
        << "SendRecv did not exercise a cross-cluster edge";
    ASSERT_EQ(globalMismatches, 0)
        << "Runner expected cross-cluster peer transport " << expectedTransport;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Verify: all received elements should equal recvPeer's rank
  float *hrecv = static_cast<float *>(hostrecvbuff);
  float expected = static_cast<float>(recvPeer);
  bool success = true;
  for (size_t i = 0; i < count; i++) {
    if (hrecv[i] != expected) {
      if (rank == 0) {
        std::cout << "Mismatch at index " << i << ": expected " << expected
                  << ", got " << hrecv[i] << std::endl;
      }
      success = false;
      break;
    }
  }
  EXPECT_TRUE(success);
}
