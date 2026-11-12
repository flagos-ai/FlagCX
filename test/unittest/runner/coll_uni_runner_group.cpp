#include "device_api/flagcx_device_internal.h"
#include "global_comm.h"
#include "group.h"
#include "runner_fixtures.hpp"
#include "test_utils.hpp"
#include "uni_runner_algorithm.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

class ScopedUniRunnerAlgorithm {
public:
  ScopedUniRunnerAlgorithm() {
    const char *value = std::getenv("FLAGCX_UNIRUNNER_ALGO");
    hadValue = value != nullptr;
    if (hadValue)
      previousValue = value;
  }

  ~ScopedUniRunnerAlgorithm() {
    if (hadValue)
      setenv("FLAGCX_UNIRUNNER_ALGO", previousValue.c_str(), 1);
    else
      unsetenv("FLAGCX_UNIRUNNER_ALGO");
  }

  void set(UniRunnerAlgorithm algorithm) {
    const std::string value = std::to_string(static_cast<int64_t>(algorithm));
    setenv("FLAGCX_UNIRUNNER_ALGO", value.c_str(), 1);
  }

private:
  bool hadValue = false;
  std::string previousValue;
};

bool usingUniRunner() {
  const char *value = std::getenv("FLAGCX_USE_HETERO_COMM");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

} // namespace

TEST_F(FlagCXCollTest, UniRunnerDagAlgorithmsRejectOuterGroup) {
  if (!usingUniRunner())
    GTEST_SKIP() << "requires UniRunner";

  ScopedUniRunnerAlgorithm selectedAlgorithm;
  for (int value = uniRunnerAlgorithmRingAR;
       value <= uniRunnerAlgorithmSlicedAR; ++value) {
    selectedAlgorithm.set(static_cast<UniRunnerAlgorithm>(value));
    ASSERT_EQ(flagcxGroupStart(comm), flagcxSuccess);
    EXPECT_EQ(flagcxGetGroupDepth(), 1);
    EXPECT_EQ(flagcxAllReduce(sendbuff, recvbuff, 1, flagcxFloat, flagcxSum,
                              comm, stream),
              flagcxInvalidUsage)
        << "AllReduce algorithm " << value;
    EXPECT_EQ(flagcxGetGroupDepth(), 1);
    EXPECT_EQ(flagcxGroupEnd(comm), flagcxSuccess);
    EXPECT_EQ(flagcxGetGroupDepth(), 0);
  }

  selectedAlgorithm.set(uniRunnerAlgorithmGroupedAG);
  ASSERT_EQ(flagcxGroupStart(comm), flagcxSuccess);
  EXPECT_EQ(flagcxAllGather(sendbuff, recvbuff, 1, flagcxFloat, comm, stream),
            flagcxInvalidUsage);
  EXPECT_EQ(flagcxGroupEnd(comm), flagcxSuccess);
  EXPECT_EQ(flagcxGetGroupDepth(), 0);

  ASSERT_EQ(flagcxGroupStart(comm), flagcxSuccess);
  EXPECT_EQ(flagcxReduce(sendbuff, recvbuff, 1, flagcxFloat, flagcxSum, 0, comm,
                         stream),
            flagcxInvalidUsage);
  EXPECT_EQ(flagcxGroupEnd(comm), flagcxSuccess);
  EXPECT_EQ(flagcxGetGroupDepth(), 0);

  ASSERT_EQ(flagcxGroupStart(comm), flagcxSuccess);
  EXPECT_EQ(flagcxReduceScatter(sendbuff, recvbuff, 1, flagcxFloat, flagcxSum,
                                comm, stream),
            flagcxInvalidUsage);
  EXPECT_EQ(flagcxGroupEnd(comm), flagcxSuccess);
  EXPECT_EQ(flagcxGetGroupDepth(), 0);

  for (int value = uniRunnerAlgorithmAllGatherGemmBuiltIn;
       value <= uniRunnerAlgorithmGemmReduceScatterDsl; ++value) {
    selectedAlgorithm.set(static_cast<UniRunnerAlgorithm>(value));
    ASSERT_EQ(flagcxGroupStart(comm), flagcxSuccess);
    const flagcxResult_t result =
        value <= uniRunnerAlgorithmAllGatherGemmDsl
            ? flagcxAllGatherGemm(sendbuff, sendbuff, recvbuff, 1, 1, 1,
                                  flagcxFloat, comm, stream)
            : flagcxGemmReduceScatter(sendbuff, sendbuff, recvbuff,
                                      static_cast<size_t>(nranks), 1, 1,
                                      flagcxFloat, flagcxSum, comm, stream);
    EXPECT_EQ(result, flagcxInvalidUsage) << "fused algorithm " << value;
    EXPECT_EQ(flagcxGroupEnd(comm), flagcxSuccess);
    EXPECT_EQ(flagcxGetGroupDepth(), 0);
  }

  // The rejected calls must leave the communicator usable without a group.
  selectedAlgorithm.set(uniRunnerAlgorithmP2pAG);
  static_cast<float *>(hostsendbuff)[0] = static_cast<float>(rank);
  ASSERT_EQ(devHandle->deviceMemcpy(sendbuff, hostsendbuff, sizeof(float),
                                    flagcxMemcpyHostToDevice, stream),
            flagcxSuccess);
  ASSERT_EQ(flagcxAllGather(sendbuff, recvbuff, 1, flagcxFloat, comm, stream),
            flagcxSuccess);
  ASSERT_EQ(devHandle->deviceMemcpy(hostrecvbuff, recvbuff,
                                    static_cast<size_t>(nranks) * sizeof(float),
                                    flagcxMemcpyDeviceToHost, stream),
            flagcxSuccess);
  ASSERT_EQ(synchronizeAndCheckAsyncError(), flagcxSuccess);
  for (int source = 0; source < nranks; ++source)
    EXPECT_EQ(static_cast<float *>(hostrecvbuff)[source],
              static_cast<float>(source));
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

TEST_F(FlagCXCollTest, AllGatherInsideGroup) {
  for (size_t i = 0; i < count / nranks; ++i)
    static_cast<float *>(hostsendbuff)[i] = rank * 1000.0f + (i % 10);
  ASSERT_EQ(devHandle->deviceMemcpy(sendbuff, hostsendbuff, size / nranks,
                                    flagcxMemcpyHostToDevice, stream),
            flagcxSuccess);

  ASSERT_EQ(flagcxGroupStart(comm), flagcxSuccess);
  const flagcxResult_t submitResult = flagcxAllGather(
      sendbuff, recvbuff, count / nranks, flagcxFloat, comm, stream);
  const flagcxResult_t groupResult = flagcxGroupEnd(comm);
  ASSERT_EQ(submitResult, flagcxSuccess);
  ASSERT_EQ(groupResult, flagcxSuccess);
  ASSERT_EQ(devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size,
                                    flagcxMemcpyDeviceToHost, stream),
            flagcxSuccess);
  ASSERT_EQ(synchronizeAndCheckAsyncError(), flagcxSuccess);

  const size_t chunkCount = count / nranks;
  std::vector<float> expected(chunkCount);
  for (int source = 0; source < nranks; ++source) {
    for (size_t i = 0; i < chunkCount; ++i)
      expected[i] = source * 1000.0f + (i % 10);
    EXPECT_TRUE(
        verifyBuffer(static_cast<float *>(hostrecvbuff) + source * chunkCount,
                     expected.data(), chunkCount));
  }
}
