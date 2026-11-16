// Isolate the group shape, cross-cluster direction, and D2H submission order
// involved in Iluvatar HYBRID SendRecv timeouts. Built only on request.

#include "comm.h"
#include "global_comm.h"
#include "runner_fixtures.hpp"
#include "runner_result.h"
#include "transport.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

TEST_F(FlagCXCollTest, SendRecvProbe) {
  const char *pattern = std::getenv("FLAGCX_CI_SENDRECV_PROBE_PATTERN");
  const char *copyOrder = std::getenv("FLAGCX_CI_SENDRECV_PROBE_COPY_ORDER");
  ASSERT_NE(pattern, nullptr);
  ASSERT_NE(copyOrder, nullptr);
  const bool earlyCopy = std::strcmp(copyOrder, "EARLY") == 0;
  ASSERT_TRUE(earlyCopy || std::strcmp(copyOrder, "LATE") == 0);
  ASSERT_EQ(nranks, 4);

  int localTopology = comm->commType == flagcxCommunicatorHybrid &&
                      comm->heteroComm != nullptr &&
                      comm->clusterIds[0] == comm->clusterIds[1] &&
                      comm->clusterIds[2] == comm->clusterIds[3] &&
                      comm->clusterIds[0] != comm->clusterIds[2];
  int allTopology = 0;
  MPI_Allreduce(&localTopology, &allTopology, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  ASSERT_EQ(allTopology, 1)
      << "Probe requires two logical clusters of two ranks";

  const bool forward = std::strcmp(pattern, "F") == 0;
  const bool reverse = std::strcmp(pattern, "R") == 0;
  const bool dual = std::strcmp(pattern, "D") == 0;
  const bool split = std::strcmp(pattern, "S") == 0;
  const bool ring = std::strcmp(pattern, "RING") == 0;
  const bool pxnDual = std::strcmp(pattern, "PXN_D") == 0;
  ASSERT_TRUE(forward || reverse || dual || split || ring || pxnDual)
      << "Unknown SendRecv probe pattern: " << pattern;

  auto trace = [&](const char *phase, int peer = -1, int result = -1) {
    std::cerr << "SendRecvProbe rank=" << rank << " pattern=" << pattern
              << " copy_order=" << copyOrder << " phase=" << phase
              << " peer=" << peer << " result=" << result
              << " time=" << MPI_Wtime() << std::endl;
  };

  float *hostSend = static_cast<float *>(hostsendbuff);
  for (size_t i = 0; i < count; ++i)
    hostSend[i] = static_cast<float>(rank);
  ASSERT_EQ(
      devHandle->deviceMemset(recvbuff, 0xff, size, flagcxMemDevice, stream),
      flagcxSuccess);
  ASSERT_EQ(
      devHandle->deviceMemset(hostrecvbuff, 0xff, size, flagcxMemHost, nullptr),
      flagcxSuccess);
  ASSERT_EQ(devHandle->deviceMemcpy(sendbuff, hostsendbuff, size,
                                    flagcxMemcpyHostToDevice, stream),
            flagcxSuccess);
  MPI_Barrier(MPI_COMM_WORLD);

  const bool forwardSender = rank == 0 || rank == 2;
  const int pairedPeer = 3 - rank;
  const int pxnPeer = (rank + 2) % 4;
  bool hasSend = true;
  bool hasRecv = true;
  if (forward) {
    hasSend = forwardSender;
    hasRecv = !forwardSender;
  } else if (reverse) {
    hasSend = !forwardSender;
    hasRecv = forwardSender;
  }
  const int sendPeer = ring ? (rank + 1) % 4 : pxnDual ? pxnPeer : pairedPeer;
  const int recvPeer = ring ? (rank + 3) % 4 : pxnDual ? pxnPeer : pairedPeer;

  auto runGroup = [&](const char *label, bool send, bool recv, int sendTo,
                      int recvFrom) {
    trace(label);
    flagcxResult_t result = flagcxRunnerRunGroup(
        [&]() { return flagcxGroupStart(comm); },
        [&]() {
          flagcxResult_t firstError = flagcxSuccess;
          if (send) {
            trace("send-enter", sendTo);
            flagcxResult_t sendResult =
                flagcxSend(sendbuff, count, flagcxFloat, sendTo, comm, stream);
            trace("send-return", sendTo, sendResult);
            flagcxRunnerRecordFirstError(sendResult, &firstError);
            if (!flagcxRunnerResultAccepted(sendResult))
              return firstError;
          }
          if (recv) {
            trace("recv-enter", recvFrom);
            flagcxResult_t recvResult = flagcxRecv(recvbuff, count, flagcxFloat,
                                                   recvFrom, comm, stream);
            trace("recv-return", recvFrom, recvResult);
            flagcxRunnerRecordFirstError(recvResult, &firstError);
          }
          return firstError;
        },
        [&]() { return flagcxGroupEnd(comm); });
    trace("group-return", -1, result);
    return result;
  };

  flagcxResult_t groupResult = flagcxSuccess;
  if (split) {
    groupResult = runGroup("forward-group-enter", forwardSender, !forwardSender,
                           pairedPeer, pairedPeer);
    if (flagcxRunnerResultAccepted(groupResult))
      groupResult = runGroup("reverse-group-enter", !forwardSender,
                             forwardSender, pairedPeer, pairedPeer);
  } else {
    groupResult = runGroup("group-enter", hasSend, hasRecv, sendPeer, recvPeer);
  }
  int localGroupResult = static_cast<int>(groupResult);
  int globalGroupResult = 0;
  trace("group-allreduce-enter", -1, localGroupResult);
  MPI_Allreduce(&localGroupResult, &globalGroupResult, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  trace("group-allreduce-return", -1, globalGroupResult);
  ASSERT_EQ(globalGroupResult, static_cast<int>(flagcxSuccess));

  auto copyToHost = [&]() {
    if (!hasRecv)
      return flagcxSuccess;
    trace("d2h-enter", recvPeer);
    flagcxResult_t result = devHandle->deviceMemcpy(
        hostrecvbuff, recvbuff, size, flagcxMemcpyDeviceToHost, stream);
    trace("d2h-return", recvPeer, result);
    return result;
  };
  if (earlyCopy)
    ASSERT_EQ(copyToHost(), flagcxSuccess);

  trace("first-sync-enter");
  flagcxResult_t syncResult = synchronizeAndCheckAsyncError();
  trace("first-sync-return", -1, syncResult);
  ASSERT_EQ(syncResult, flagcxSuccess);
  if (!earlyCopy) {
    ASSERT_EQ(copyToHost(), flagcxSuccess);
    trace("second-sync-enter");
    syncResult = synchronizeAndCheckAsyncError();
    trace("second-sync-return", -1, syncResult);
    ASSERT_EQ(syncResult, flagcxSuccess);
  }

  const char *expectedTransport =
      std::getenv("FLAGCX_CI_EXPECT_PEER_TRANSPORT");
  ASSERT_NE(expectedTransport, nullptr);
  int localChecked = 0;
  int localMismatches = 0;
  auto checkConnector = [&](int peer, bool send) {
    if (comm->clusterIds[rank] == comm->clusterIds[peer])
      return;
    ++localChecked;
    flagcxConnector *connector =
        send ? &comm->heteroComm->channels[0].peers[peer]->send[0]
             : &comm->heteroComm->channels[0].peers[peer]->recv[0];
    const bool connected = connector->connected &&
                           connector->proxyConn.initialized &&
                           connector->proxyConn.connection != nullptr;
    const bool matches =
        connected && ((std::strcmp(expectedTransport, "P2P") == 0 &&
                       connector->proxyConn.transport == TRANSPORT_P2P) ||
                      (std::strcmp(expectedTransport, "NET") == 0 &&
                       connector->proxyConn.transport == TRANSPORT_NET));
    if (!matches)
      ++localMismatches;
  };
  if (hasSend)
    checkConnector(sendPeer, true);
  if (hasRecv)
    checkConnector(recvPeer, false);
  int checked = 0;
  int mismatches = 0;
  MPI_Allreduce(&localChecked, &checked, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&localMismatches, &mismatches, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  ASSERT_GT(checked, 0);
  ASSERT_EQ(mismatches, 0) << "Unexpected cross-cluster transport";

  if (hasRecv) {
    const float *hostRecv = static_cast<const float *>(hostrecvbuff);
    for (size_t i = 0; i < count; ++i) {
      ASSERT_EQ(hostRecv[i], static_cast<float>(recvPeer))
          << "rank=" << rank << " index=" << i << " source=" << recvPeer;
    }
  }
  trace("verified");
}
