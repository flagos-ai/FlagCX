#include "runner_fixtures.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

TEST_F(FlagCXCollTest, AllGatherGemm) {
  setenv("FLAGCX_UNIRUNNER_GEMM_KSLICES", "3", 0);

  const size_t mPerRank = 130;
  const size_t k = 19;
  const size_t n = 130;
  const size_t inputElements = mPerRank * k;
  const size_t weightElements = k * n;
  const size_t outputElements = nranks * mPerRank * n;

  std::vector<float> hostInput(inputElements);
  std::vector<float> hostWeight(weightElements);
  std::vector<float> hostOutput(outputElements);
  std::vector<float> gatheredInput(nranks * inputElements);
  std::vector<float> expected(outputElements);
  for (size_t i = 0; i < inputElements; ++i) {
    hostInput[i] = 0.1f * static_cast<float>(rank + 1) +
                   0.01f * static_cast<float>(i + 1);
  }
  for (size_t i = 0; i < weightElements; ++i) {
    hostWeight[i] = 0.02f * static_cast<float>(rank + 1) +
                    0.005f * static_cast<float>((i % 7) + 1);
  }
  MPI_Allgather(hostInput.data(), static_cast<int>(inputElements), MPI_FLOAT,
                gatheredInput.data(), static_cast<int>(inputElements),
                MPI_FLOAT, MPI_COMM_WORLD);
  for (size_t row = 0; row < nranks * mPerRank; ++row) {
    for (size_t col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (size_t p = 0; p < k; ++p) {
        sum += gatheredInput[row * k + p] * hostWeight[p * n + col];
      }
      expected[row * n + col] = sum;
    }
  }

  void *deviceWeight = NULL;
  ASSERT_EQ(devHandle->deviceMalloc(&deviceWeight,
                                    weightElements * sizeof(float),
                                    flagcxMemDevice, NULL),
            flagcxSuccess);
  ASSERT_EQ(devHandle->deviceMemcpy(sendbuff, hostInput.data(),
                                    inputElements * sizeof(float),
                                    flagcxMemcpyHostToDevice, stream),
            flagcxSuccess);
  ASSERT_EQ(devHandle->deviceMemcpy(deviceWeight, hostWeight.data(),
                                    weightElements * sizeof(float),
                                    flagcxMemcpyHostToDevice, stream),
            flagcxSuccess);

  EXPECT_EQ(flagcxAllGatherGemm(sendbuff, deviceWeight, recvbuff, mPerRank, n,
                               k, flagcxFloat32, comm, stream),
            flagcxSuccess);
  ASSERT_EQ(devHandle->deviceMemcpy(hostOutput.data(), recvbuff,
                                    outputElements * sizeof(float),
                                    flagcxMemcpyDeviceToHost, stream),
            flagcxSuccess);
  ASSERT_EQ(devHandle->streamSynchronize(stream), flagcxSuccess);

  for (size_t i = 0; i < outputElements; ++i) {
    float tolerance = 1e-4f + 1e-4f * std::fabs(expected[i]);
    EXPECT_NEAR(hostOutput[i], expected[i], tolerance) << "index " << i;
  }
  EXPECT_EQ(devHandle->deviceFree(deviceWeight, flagcxMemDevice, NULL),
            flagcxSuccess);
}
