#include "adaptor.h"
#include "flagcx.h"
#include <cstdlib>
#include <gtest/gtest.h>
#include <string>

#ifdef USE_NVIDIA_ADAPTOR
#include "nvidia_adaptor.h"

namespace {
struct CaptureEnvironment {
  bool hadHetero = getenv("FLAGCX_USE_HETERO_COMM") != nullptr;
  bool hadSplit = getenv("FLAGCX_CLUSTER_SPLIT_LIST") != nullptr;
  std::string hetero = hadHetero ? getenv("FLAGCX_USE_HETERO_COMM") : "";
  std::string split = hadSplit ? getenv("FLAGCX_CLUSTER_SPLIT_LIST") : "";
  CaptureEnvironment() {
    setenv("FLAGCX_USE_HETERO_COMM", "0", 1);
    setenv("FLAGCX_CLUSTER_SPLIT_LIST", "1", 1);
  }
  ~CaptureEnvironment() {
    if (hadHetero)
      setenv("FLAGCX_USE_HETERO_COMM", hetero.c_str(), 1);
    else
      unsetenv("FLAGCX_USE_HETERO_COMM");
    if (hadSplit)
      setenv("FLAGCX_CLUSTER_SPLIT_LIST", split.c_str(), 1);
    else
      unsetenv("FLAGCX_CLUSTER_SPLIT_LIST");
  }
};
struct GraphOwners {
  flagcxComm_t comm = nullptr;
  flagcxStream_t stream = nullptr;
  void *buffer = nullptr;
  cudaGraph_t graphs[2] = {};
  cudaGraphExec_t execs[2] = {};
  bool capturing = false;
  ~GraphOwners() {
    if (capturing) {
      cudaGraph_t discard = nullptr;
      cudaStreamEndCapture(stream->base, &discard);
      if (discard)
        cudaGraphDestroy(discard);
    }
    if (stream)
      EXPECT_EQ(deviceAdaptor->streamSynchronize(stream), flagcxSuccess);
    for (int i = 0; i < 2; i++) {
      if (execs[i])
        EXPECT_EQ(cudaGraphExecDestroy(execs[i]), cudaSuccess);
      if (graphs[i])
        EXPECT_EQ(cudaGraphDestroy(graphs[i]), cudaSuccess);
    }
    if (comm)
      EXPECT_EQ(flagcxCommDestroy(comm), flagcxSuccess);
    if (buffer)
      EXPECT_EQ(deviceAdaptor->deviceFree(buffer, flagcxMemDevice, nullptr),
                flagcxSuccess);
    if (stream)
      EXPECT_EQ(deviceAdaptor->streamDestroy(stream), flagcxSuccess);
  }
};
} // namespace
#endif

TEST(GraphCaptureLifecycle, NativeCollectivesCaptureReplayAndFinalize) {
#ifndef USE_NVIDIA_ADAPTOR
  GTEST_SKIP() << "Requires the CUDA SDK and a real device";
#else
  int count = 0;
  ASSERT_EQ(deviceAdaptor->getDeviceCount(&count), flagcxSuccess);
  if (count == 0)
    GTEST_SKIP() << "Requires a CUDA device";
  CaptureEnvironment environment;
  ASSERT_EQ(deviceAdaptor->setDevice(0), flagcxSuccess);
  GraphOwners owners;
  flagcxUniqueId id = {};
  ASSERT_EQ(flagcxGetUniqueId(&id), flagcxSuccess);
  ASSERT_EQ(flagcxCommInitRank(&owners.comm, 1, &id, 0), flagcxSuccess);
  ASSERT_EQ(deviceAdaptor->streamCreate(&owners.stream), flagcxSuccess);
  ASSERT_EQ(deviceAdaptor->deviceMalloc(&owners.buffer, sizeof(float),
                                        flagcxMemDevice, nullptr),
            flagcxSuccess);
  ASSERT_EQ(deviceAdaptor->deviceMemset(owners.buffer, 0, sizeof(float),
                                        flagcxMemDevice, owners.stream),
            flagcxSuccess);
  auto submit = [&] {
    return flagcxAllReduce(owners.buffer, owners.buffer, 1, flagcxFloat,
                           flagcxSum, owners.comm, owners.stream);
  };
  ASSERT_EQ(submit(), flagcxSuccess); // Initialize native SDK launch resources.
  ASSERT_EQ(deviceAdaptor->streamSynchronize(owners.stream), flagcxSuccess);
  for (int graph = 0; graph < 2; graph++) {
    ASSERT_EQ(cudaStreamBeginCapture(owners.stream->base,
                                     cudaStreamCaptureModeThreadLocal),
              cudaSuccess);
    owners.capturing = true;
    if (graph == 1)
      ASSERT_EQ(flagcxGroupStart(owners.comm), flagcxSuccess);
    EXPECT_EQ(submit(), flagcxSuccess);
    EXPECT_EQ(submit(), flagcxSuccess);
    if (graph == 1)
      EXPECT_EQ(flagcxGroupEnd(owners.comm), flagcxSuccess);
    cudaError_t ended =
        cudaStreamEndCapture(owners.stream->base, &owners.graphs[graph]);
    owners.capturing = false;
    ASSERT_EQ(ended, cudaSuccess);
    ASSERT_EQ(cudaGraphInstantiate(&owners.execs[graph], owners.graphs[graph],
                                   nullptr, nullptr, 0),
              cudaSuccess);
    ASSERT_EQ(submit(), flagcxSuccess); // Ordinary work between two captures.
    ASSERT_EQ(deviceAdaptor->streamSynchronize(owners.stream), flagcxSuccess);
  }
  for (int replay = 0; replay < 3; replay++)
    for (int graph = 0; graph < 2; graph++)
      ASSERT_EQ(cudaGraphLaunch(owners.execs[graph], owners.stream->base),
                cudaSuccess);
  ASSERT_EQ(deviceAdaptor->streamSynchronize(owners.stream), flagcxSuccess);
  // The application stops using its graphs before finalizing the communicator.
  for (int graph = 0; graph < 2; graph++) {
    ASSERT_EQ(cudaGraphExecDestroy(owners.execs[graph]), cudaSuccess);
    owners.execs[graph] = nullptr;
    ASSERT_EQ(cudaGraphDestroy(owners.graphs[graph]), cudaSuccess);
    owners.graphs[graph] = nullptr;
  }
  EXPECT_EQ(flagcxCommFinalize(owners.comm), flagcxSuccess);
#endif
}
