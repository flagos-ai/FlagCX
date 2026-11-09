// Exercise communicator teardown while peer ranks reach destruction at
// different times. Odd ranks retain their imported FIFO mappings briefly so
// even-rank exporters cannot rely on service-thread scheduling for lifetime.

#include "global_comm.h"
#include "runner_fixtures.hpp"

#include <cstdlib>
#include <cstring>

TEST_F(FlagCXCollTest, SkewedP2pTeardown) {
  const char *useHetero = std::getenv("FLAGCX_USE_HETERO_COMM");
  const char *p2pDisabled = std::getenv("FLAGCX_P2P_DISABLE");
  const char *expectedTransport =
      std::getenv("FLAGCX_CI_EXPECT_PEER_TRANSPORT");
  const bool expectP2p = expectedTransport != nullptr &&
                         std::strcmp(expectedTransport, "P2P") == 0;
  const bool heteroRequested =
      useHetero != nullptr && std::strcmp(useHetero, "1") == 0;
  if (comm->commType != flagcxCommunicatorHybrid ||
      (!expectP2p && !heteroRequested) ||
      (p2pDisabled != nullptr && std::strcmp(p2pDisabled, "1") == 0)) {
    GTEST_SKIP() << "requires the heterogeneous same-node P2P transport";
  }

  int sendPeer = (rank + 1) % nranks;
  int recvPeer = (rank - 1 + nranks) % nranks;
  static_cast<float *>(hostsendbuff)[0] = static_cast<float>(rank);
  ASSERT_EQ(devHandle->deviceMemcpy(sendbuff, hostsendbuff, sizeof(float),
                                    flagcxMemcpyHostToDevice, stream),
            flagcxSuccess);

  MPI_Barrier(MPI_COMM_WORLD);
  ASSERT_EQ(flagcxGroupStart(comm), flagcxSuccess);
  ASSERT_EQ(flagcxSend(sendbuff, 1, flagcxFloat, sendPeer, comm, stream),
            flagcxSuccess);
  ASSERT_EQ(flagcxRecv(recvbuff, 1, flagcxFloat, recvPeer, comm, stream),
            flagcxSuccess);
  ASSERT_EQ(flagcxGroupEnd(comm), flagcxSuccess);
  ASSERT_EQ(devHandle->deviceMemcpy(hostrecvbuff, recvbuff, sizeof(float),
                                    flagcxMemcpyDeviceToHost, stream),
            flagcxSuccess);
  ASSERT_EQ(synchronizeAndCheckAsyncError(), flagcxSuccess);
  EXPECT_EQ(static_cast<float *>(hostrecvbuff)[0],
            static_cast<float>(recvPeer));

  MPI_Barrier(MPI_COMM_WORLD);
  teardownDelayMs = rank % 2 == 1 ? 500 : 0;
}
