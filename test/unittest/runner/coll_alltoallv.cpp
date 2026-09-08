// AlltoAllV correctness test with variable and zero-sized peer transfers.

#include "runner_fixtures.hpp"
#include <cstdint>
#include <vector>

TEST_F(FlagCXCollTest, AlltoAllV) {
  constexpr size_t kCountUnit = 256;

  std::vector<size_t> sendcounts(nranks);
  std::vector<size_t> recvcounts(nranks);
  std::vector<size_t> sdispls(nranks);
  std::vector<size_t> rdispls(nranks);

  size_t sendTotal = 0;
  size_t recvTotal = 0;
  for (int peer = 0; peer < nranks; ++peer) {
    // recvcounts[peer] is the transpose of the send-count matrix: it matches
    // what peer sends to this rank while keeping incoming and outgoing layouts
    // asymmetric.
    sendcounts[peer] = static_cast<size_t>((2 * rank + peer) % 3) * kCountUnit;
    recvcounts[peer] = static_cast<size_t>((2 * peer + rank) % 3) * kCountUnit;
    sdispls[peer] = sendTotal;
    rdispls[peer] = recvTotal;
    sendTotal += sendcounts[peer];
    recvTotal += recvcounts[peer];
  }

  ASSERT_LE(sendTotal, count);
  ASSERT_LE(recvTotal, count);

  float *hsend = static_cast<float *>(hostsendbuff);
  for (int peer = 0; peer < nranks; ++peer) {
    const float value = static_cast<float>(rank * nranks + peer);
    for (size_t i = 0; i < sendcounts[peer]; ++i)
      hsend[sdispls[peer] + i] = value;
  }

  ASSERT_EQ(devHandle->deviceMemcpy(sendbuff, hostsendbuff,
                                    sendTotal * sizeof(float),
                                    flagcxMemcpyHostToDevice, stream),
            flagcxSuccess);
  ASSERT_EQ(devHandle->deviceMemset(recvbuff, 0xFF, recvTotal * sizeof(float),
                                    flagcxMemDevice, stream),
            flagcxSuccess);

  MPI_Barrier(MPI_COMM_WORLD);

  ASSERT_EQ(flagcxAlltoAllv(sendbuff, sendcounts.data(), sdispls.data(),
                            recvbuff, recvcounts.data(), rdispls.data(),
                            flagcxFloat, comm, stream),
            flagcxSuccess);

  ASSERT_EQ(devHandle->deviceMemcpy(hostrecvbuff, recvbuff,
                                    recvTotal * sizeof(float),
                                    flagcxMemcpyDeviceToHost, stream),
            flagcxSuccess);
  ASSERT_EQ(devHandle->streamSynchronize(stream), flagcxSuccess);

  float *hrecv = static_cast<float *>(hostrecvbuff);
  for (int peer = 0; peer < nranks; ++peer) {
    const float expected = static_cast<float>(peer * nranks + rank);
    for (size_t i = 0; i < recvcounts[peer]; ++i) {
      EXPECT_EQ(hrecv[rdispls[peer] + i], expected)
          << "Mismatch from peer " << peer << " at element " << i;
    }
  }
}
