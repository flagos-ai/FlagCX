/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "runner_fixtures.hpp"
#include <algorithm>
#include <vector>

namespace {

size_t pairCount(int src, int dst, int nranks, int round) {
  if (dst == (src + round + 1) % nranks)
    return 0;
  return 1 + static_cast<size_t>((src * 7 + dst * 3 + round) % 5);
}

float payloadValue(int round, int src, int dst, size_t index) {
  return static_cast<float>(round * 1000000 + src * 10000 + dst * 100 +
                            static_cast<int>(index));
}

} // namespace

TEST_F(FlagCXCollTest, AlltoAllvAsymmetric) {
  constexpr float kSendSentinel = -101.0f;
  constexpr float kRecvSentinel = -303.0f;
  const size_t hostElements = size / sizeof(float);

  for (int round = 0; round < 2; ++round) {
    std::vector<size_t> sendcounts(nranks);
    std::vector<size_t> recvcounts(nranks);
    std::vector<size_t> sdispls(nranks);
    std::vector<size_t> rdispls(nranks);

    size_t checkedRecvElements = 0;
    for (int peer = 0; peer < nranks; ++peer) {
      sendcounts[peer] = pairCount(rank, peer, nranks, round);
      recvcounts[peer] = pairCount(peer, rank, nranks, round);
      sdispls[peer] = 3 + static_cast<size_t>(peer) * 11 +
                      static_cast<size_t>((rank + peer + round) % 3);
      rdispls[peer] = 5 + static_cast<size_t>(peer) * 13 +
                      static_cast<size_t>((rank + 2 * peer + round) % 4);
      checkedRecvElements =
          std::max(checkedRecvElements,
                   rdispls[peer] + std::max<size_t>(recvcounts[peer], 1));
    }
    ASSERT_LT(checkedRecvElements, hostElements);

    float *hostSend = static_cast<float *>(hostsendbuff);
    float *hostRecv = static_cast<float *>(hostrecvbuff);
    std::fill(hostSend, hostSend + hostElements, kSendSentinel);
    std::fill(hostRecv, hostRecv + hostElements, kRecvSentinel);
    for (int peer = 0; peer < nranks; ++peer) {
      for (size_t i = 0; i < sendcounts[peer]; ++i) {
        hostSend[sdispls[peer] + i] = payloadValue(round, rank, peer, i);
      }
    }

    ASSERT_EQ(devHandle->deviceMemcpy(sendbuff, hostsendbuff, size,
                                      flagcxMemcpyHostToDevice, stream),
              flagcxSuccess);
    ASSERT_EQ(devHandle->deviceMemcpy(recvbuff, hostrecvbuff, size,
                                      flagcxMemcpyHostToDevice, stream),
              flagcxSuccess);
    ASSERT_EQ(synchronizeAndCheckAsyncError(), flagcxSuccess);
    MPI_Barrier(MPI_COMM_WORLD);

    flagcxResult_t localResult = flagcxAlltoAllv(
        sendbuff, sendcounts.data(), sdispls.data(), recvbuff,
        recvcounts.data(), rdispls.data(), flagcxFloat, comm, stream);
    int localResultCode = static_cast<int>(localResult);
    int globalResultCode = static_cast<int>(flagcxSuccess);
    MPI_Allreduce(&localResultCode, &globalResultCode, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    ASSERT_EQ(globalResultCode, static_cast<int>(flagcxSuccess));
    ASSERT_EQ(devHandle->deviceMemcpy(hostrecvbuff, recvbuff, size,
                                      flagcxMemcpyDeviceToHost, stream),
              flagcxSuccess);
    ASSERT_EQ(synchronizeAndCheckAsyncError(), flagcxSuccess);

    std::vector<bool> written(checkedRecvElements, false);
    for (int peer = 0; peer < nranks; ++peer) {
      for (size_t i = 0; i < recvcounts[peer]; ++i) {
        const size_t index = rdispls[peer] + i;
        written[index] = true;
        EXPECT_EQ(hostRecv[index], payloadValue(round, peer, rank, i))
            << "round=" << round << " source=" << peer << " element=" << i;
      }
      if (recvcounts[peer] == 0) {
        EXPECT_EQ(hostRecv[rdispls[peer]], kRecvSentinel)
            << "zero-count source " << peer << " modified its destination";
      }
    }
    for (size_t i = 0; i < checkedRecvElements; ++i) {
      if (!written[i]) {
        EXPECT_EQ(hostRecv[i], kRecvSentinel)
            << "AlltoAllV modified displacement gap at index " << i;
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}
