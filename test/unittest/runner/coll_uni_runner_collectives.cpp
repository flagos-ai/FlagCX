#include "comm_lifecycle_check.hpp"
#include "global_comm.h"
#include "runner_fixtures.hpp"
#include "uni_runner_algorithm.h"

#include <cstdlib>
#include <string>

namespace {
class ScopedAlgorithm {
public:
  ScopedAlgorithm() {
    const char *value = std::getenv("FLAGCX_UNIRUNNER_ALGO");
    hadValue = value != nullptr;
    if (hadValue)
      previous = value;
  }
  ~ScopedAlgorithm() {
    if (hadValue)
      setenv("FLAGCX_UNIRUNNER_ALGO", previous.c_str(), 1);
    else
      unsetenv("FLAGCX_UNIRUNNER_ALGO");
  }
  void set(UniRunnerAlgorithm value) {
    const auto text = std::to_string(static_cast<int64_t>(value));
    setenv("FLAGCX_UNIRUNNER_ALGO", text.c_str(), 1);
  }

private:
  bool hadValue = false;
  std::string previous;
};
} // namespace

TEST_F(FlagCXCollTest, UniRunnerUnsupportedReductionTypes) {
#if defined(USE_MUSA_ADAPTOR) || defined(USE_DU_ADAPTOR) ||                    \
    defined(USE_PPU_ADAPTOR)
  ASSERT_TRUE(comm->uniRunnerKernelSupported);
  ScopedAlgorithm algorithm;
  for (const auto datatype : {flagcxFloat16, flagcxBfloat16}) {
    for (int value = uniRunnerAlgorithmRingAR;
         value <= uniRunnerAlgorithmSlicedAR; ++value) {
      algorithm.set(static_cast<UniRunnerAlgorithm>(value));
      EXPECT_EQ(flagcxAllReduce(sendbuff, recvbuff, 1, datatype, flagcxSum,
                                comm, stream),
                flagcxNotSupported);
    }
    EXPECT_EQ(flagcxReduce(sendbuff, recvbuff, 1, datatype, flagcxSum, 0, comm,
                           stream),
              flagcxNotSupported);
    EXPECT_EQ(flagcxReduceScatter(sendbuff, recvbuff, 1, datatype, flagcxSum,
                                  comm, stream),
              flagcxNotSupported);
  }
#else
  GTEST_SKIP() << "This executor has half/bfloat16 reduction specializations";
#endif
}

TEST_F(FlagCXCollTest, UniRunnerAllReduceStreamsAndInPlace) {
  ASSERT_TRUE(comm->uniRunnerKernelSupported);
  ASSERT_TRUE(comm->streamValueSupported);
  ScopedAlgorithm algorithm;
  flagcxStream_t second = nullptr;
  ASSERT_EQ(devHandle->streamCreate(&second), flagcxSuccess);
  // These are sequential calls on alternating streams; the per-communicator
  // UniRunner state is shared and is not a concurrent submission interface.
  for (const auto selected :
       {uniRunnerAlgorithmRingAR, uniRunnerAlgorithmIpcRingAR,
        uniRunnerAlgorithmSlicedAR}) {
    algorithm.set(selected);
    for (int iteration = 0; iteration < 4; ++iteration) {
      const size_t elements = 1031; // Uneven across ranks and executor workers.
      auto *input = static_cast<float *>(hostsendbuff);
      for (size_t i = 0; i < elements; ++i)
        input[i] = static_cast<float>(rank + 1);
      flagcxStream_t active = iteration % 2 == 0 ? stream : second;
      ASSERT_EQ(devHandle->deviceMemcpy(sendbuff, hostsendbuff,
                                        elements * sizeof(float),
                                        flagcxMemcpyHostToDevice, active),
                flagcxSuccess);
      void *output = iteration < 2 ? recvbuff : sendbuff;
      EXPECT_EQ(flagcxAllReduce(sendbuff, output, elements, flagcxFloat32,
                                flagcxSum, comm, active),
                flagcxSuccess);
      EXPECT_EQ(devHandle->deviceMemcpy(hostrecvbuff, output,
                                        elements * sizeof(float),
                                        flagcxMemcpyDeviceToHost, active),
                flagcxSuccess);
      EXPECT_EQ(flagcxTestSynchronizeStream(devHandle, active), flagcxSuccess);
      ASSERT_EQ(synchronizeAndCheckAsyncError(), flagcxSuccess);
      for (size_t i = 0; i < elements; ++i)
        EXPECT_EQ(static_cast<float *>(hostrecvbuff)[i],
                  static_cast<float>(nranks * (nranks + 1) / 2));
    }
  }
  EXPECT_EQ(devHandle->streamDestroy(second), flagcxSuccess);
}
