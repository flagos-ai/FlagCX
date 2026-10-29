// AllReduce correctness test (migrated from test/unittest/main.cpp)
#include "device_api/flagcx_device_internal.h"
#include "pxn_assertions.hpp"
#include "runner_fixtures.hpp"
#include "test_utils.hpp"
#include "uni_runner_algorithm.h"
#include <cstring>
#include <vector>

TEST_F(FlagCXCollTest, AllReduce) {

  for (size_t i = 0; i < count; i++) {
    ((float *)hostsendbuff)[i] = rank * 1000.0f + (i % 10);
  }

  devHandle->deviceMemcpy(sendbuff, hostsendbuff, size,
                          flagcxMemcpyHostToDevice, stream);

  MPI_Barrier(MPI_COMM_WORLD);
  const uint64_t relayChunksBefore = flagcxRunnerRelayChunks(comm);

  flagcxAllReduce(sendbuff, recvbuff, count, flagcxFloat, flagcxSum, comm,
                  stream);

  devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size,
                          flagcxMemcpyDeviceToHost, stream);

  ASSERT_EQ(synchronizeAndCheckAsyncError(), flagcxSuccess);

  flagcxRunnerAssertPxnTraffic(comm, rank, nranks,
                               FlagCXPxnTraffic::ConnectedCollective,
                               relayChunksBefore, "AllReduce");

  MPI_Barrier(MPI_COMM_WORLD);

  // AllReduce with Sum: result[i] = sum over all ranks of (rank * 1000.0f + (i
  // % 10))
  //   = nranks*(nranks-1)/2 * 1000.0f + nranks * (i % 10)
  std::vector<float> expected(count);
  for (size_t i = 0; i < count; i++) {
    expected[i] =
        nranks * (nranks - 1) / 2.0f * 1000.0f + nranks * (float)(i % 10);
  }
  EXPECT_TRUE(
      verifyBuffer(static_cast<float *>(hostrecvbuff), expected.data(), count));
}

TEST_F(FlagCXCollTest, GroupedARGroupSizeAgreement) {
  if (nranks < 2)
    GTEST_SKIP() << "Requires at least two ranks";

  const int64_t groupSize = rank == 0 ? 1 : nranks;
  EXPECT_EQ(agreeUniRunnerAlgorithm(comm, uniRunnerAlgorithmGroupedAR,
                                    flagcxSuccess, groupSize),
            flagcxInvalidUsage);
}

TEST_F(FlagCXCollTest, GroupedARInvalidGroupSizeAgreement) {
  if (nranks < 2)
    GTEST_SKIP() << "Requires at least two ranks";

  const flagcxResult_t localResult =
      rank == 0 ? flagcxInvalidArgument : flagcxSuccess;
  EXPECT_EQ(agreeUniRunnerAlgorithm(comm, uniRunnerAlgorithmGroupedAR,
                                    localResult, rank == 0 ? 0 : 1),
            flagcxInvalidArgument);
}

TEST_F(FlagCXCollTest, UniRunnerIpcPreparationErrorConverges) {
  if (nranks < 2)
    GTEST_SKIP() << "Requires at least two ranks";
  int bootstrapReady = comm->bootstrap != nullptr ? 1 : 0;
  int allBootstrapReady = 0;
  MPI_Allreduce(&bootstrapReady, &allBootstrapReady, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  if (!allBootstrapReady)
    GTEST_SKIP() << "Communicator has no bootstrap state";

  flagcxDevMem_t handle = nullptr;
  const flagcxResult_t result = flagcxUniRunnerIpcDevMemCreate(
      comm, rank == 0 ? nullptr : sendbuff, size, &handle);
  EXPECT_EQ(result, flagcxInvalidArgument);
  EXPECT_EQ(handle, nullptr);
}
