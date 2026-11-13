// Point-to-point Send/Recv correctness test.
// Normally uses a ring. PXN CI pairs ranks across the two logical nodes so
// sends from GPUs away from the NIC exercise sender-side relay paths.
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

  const char *expectedRunnerMode = std::getenv("FLAGCX_CI_EXPECT_RUNNER_MODE");
  const bool traceHybrid = expectedRunnerMode != nullptr &&
                           std::strcmp(expectedRunnerMode, "HYBRID") == 0;
  auto tracePhase = [&](const char *phase) {
    if (traceHybrid)
      std::cerr << "SendRecv rank=" << rank << " phase=" << phase << std::endl;
  };

  const char *expectPxn = std::getenv("FLAGCX_CI_EXPECT_PXN");
  const bool pxnCrossPairs = expectPxn != nullptr &&
                             std::strcmp(expectPxn, "1") == 0 && nranks >= 4 &&
                             nranks % 2 == 0;
  // Pair opposite logical nodes in PXN CI so both directions exercise
  // sender-side relays, regardless of the even-sized CI world.
  int sendPeer =
      pxnCrossPairs ? (rank + nranks / 2) % nranks : (rank + 1) % nranks;
  int recvPeer = pxnCrossPairs ? sendPeer : (rank - 1 + nranks) % nranks;

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
  tracePhase("before-group");
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
  tracePhase("after-group");
  int localGroupResult = static_cast<int>(groupResult);
  int globalGroupResult = static_cast<int>(flagcxSuccess);
  MPI_Allreduce(&localGroupResult, &globalGroupResult, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  ASSERT_EQ(globalGroupResult, static_cast<int>(flagcxSuccess));

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size,
                          flagcxMemcpyDeviceToHost, stream);
  tracePhase("before-stream-sync");
  ASSERT_EQ(synchronizeAndCheckAsyncError(), flagcxSuccess);
  tracePhase("after-stream-sync");

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
