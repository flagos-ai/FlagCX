#include "runner_fixtures.hpp"

#include <cmath>
#include <cstdlib>
#include <vector>

TEST_F(FlagCXCollTest, GemmReduceScatter) {
  setenv("FLAGCX_UNIRUNNER_GEMM_KSLICES", "3", 0);

  const size_t rowsPerRank = 130;
  const size_t m = rowsPerRank * nranks;
  const size_t kPerRank = 19;
  const size_t n = 130;
  const size_t inputElements = m * kPerRank;
  const size_t weightElements = kPerRank * n;
  const size_t fullOutputElements = m * n;
  const size_t outputElements = rowsPerRank * n;

  std::vector<float> hostInput(inputElements);
  std::vector<float> hostWeight(weightElements);
  std::vector<float> localProduct(fullOutputElements);
  std::vector<float> reducedProduct(fullOutputElements);
  std::vector<float> hostOutput(outputElements);
  for (size_t i = 0; i < inputElements; ++i) {
    hostInput[i] = 0.08f * static_cast<float>(rank + 1) +
                   0.01f * static_cast<float>((i % 11) + 1);
  }
  for (size_t i = 0; i < weightElements; ++i) {
    hostWeight[i] = 0.03f * static_cast<float>(rank + 1) +
                    0.004f * static_cast<float>((i % 5) + 1);
  }
  for (size_t row = 0; row < m; ++row) {
    for (size_t col = 0; col < n; ++col) {
      float sum = 0.0f;
      for (size_t p = 0; p < kPerRank; ++p) {
        sum += hostInput[row * kPerRank + p] *
               hostWeight[p * n + col];
      }
      localProduct[row * n + col] = sum;
    }
  }
  MPI_Allreduce(localProduct.data(), reducedProduct.data(),
                static_cast<int>(fullOutputElements), MPI_FLOAT, MPI_SUM,
                MPI_COMM_WORLD);

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

  EXPECT_EQ(flagcxGemmReduceScatter(sendbuff, deviceWeight, recvbuff, m, n,
                                    kPerRank, flagcxFloat32, flagcxSum, comm,
                                    stream),
            flagcxSuccess);
  ASSERT_EQ(devHandle->deviceMemcpy(hostOutput.data(), recvbuff,
                                    outputElements * sizeof(float),
                                    flagcxMemcpyDeviceToHost, stream),
            flagcxSuccess);
  ASSERT_EQ(devHandle->streamSynchronize(stream), flagcxSuccess);

  size_t expectedOffset = static_cast<size_t>(rank) * outputElements;
  for (size_t i = 0; i < outputElements; ++i) {
    float expected = reducedProduct[expectedOffset + i];
    float tolerance = 1e-4f + 1e-4f * std::fabs(expected);
    EXPECT_NEAR(hostOutput[i], expected, tolerance) << "index " << i;
  }
  EXPECT_EQ(devHandle->deviceFree(deviceWeight, flagcxMemDevice, NULL),
            flagcxSuccess);
}
