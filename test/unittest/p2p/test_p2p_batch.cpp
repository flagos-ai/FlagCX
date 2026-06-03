// Unit tests for the IB P2P net adaptor batch APIs (testBatch, igetBatch).
// These tests require IB hardware and use loopback connections.
// Tests skip gracefully via GTEST_SKIP() when no IB devices are available.

#include <cstring>
#include <future>
#include <gtest/gtest.h>
#include <infiniband/verbs.h>
#include <thread>

#include "flagcx_net.h"
#include "flagcx_net_adaptor.h"

extern struct flagcxNetAdaptor flagcxNetIbP2p;

// ---------------------------------------------------------------------------
// Fixture: establishes a loopback connection for batch operation testing
// ---------------------------------------------------------------------------
class P2pBatchTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    initResult_ = flagcxNetIbP2p.init();
    if (initResult_ == flagcxSuccess)
      flagcxNetIbP2p.devices(&nDevs_);
  }

  void SetUp() override {
    if (initResult_ != flagcxSuccess || nDevs_ <= 0)
      GTEST_SKIP() << "No IB devices available, skipping batch tests";

    // Establish loopback connection
    ASSERT_EQ(flagcxNetIbP2p.listen(0, handle_, &listenComm_), flagcxSuccess);

    auto acceptFut = std::async(std::launch::async, [this]() {
      void *comm = nullptr;
      flagcxNetIbP2p.accept(listenComm_, &comm);
      return comm;
    });
    auto connectFut = std::async(std::launch::async, [this]() {
      void *comm = nullptr;
      flagcxNetIbP2p.connect(0, handle_, &comm);
      return comm;
    });

    auto timeout = std::chrono::seconds(10);
    ASSERT_EQ(connectFut.wait_for(timeout), std::future_status::ready)
        << "connect() timed out";
    sendComm_ = connectFut.get();
    ASSERT_EQ(acceptFut.wait_for(timeout), std::future_status::ready)
        << "accept() timed out";
    recvComm_ = acceptFut.get();
    ASSERT_NE(sendComm_, nullptr);
    ASSERT_NE(recvComm_, nullptr);
  }

  void TearDown() override {
    if (sendComm_)
      flagcxNetIbP2p.closeSend(sendComm_);
    if (recvComm_)
      flagcxNetIbP2p.closeRecv(recvComm_);
    if (listenComm_)
      flagcxNetIbP2p.closeListen(listenComm_);
  }

  static flagcxResult_t initResult_;
  static int nDevs_;

  char handle_[FLAGCX_NET_HANDLE_MAXSIZE] = {};
  void *listenComm_ = nullptr;
  void *sendComm_ = nullptr;
  void *recvComm_ = nullptr;
};

flagcxResult_t P2pBatchTest::initResult_ = flagcxInternalError;
int P2pBatchTest::nDevs_ = 0;

// ---------------------------------------------------------------------------
// testBatch function pointer exists
// ---------------------------------------------------------------------------
TEST(P2pBatchStruct, TestBatchFunctionExists) {
  // testBatch is optional but should be non-NULL in the optimized adaptor
  EXPECT_NE(flagcxNetIbP2p.testBatch, nullptr);
}

TEST(P2pBatchStruct, IgetBatchFunctionExists) {
  // igetBatch is optional but should be non-NULL in the optimized adaptor
  EXPECT_NE(flagcxNetIbP2p.igetBatch, nullptr);
}

// ---------------------------------------------------------------------------
// testBatch with NULL requests reports all done
// ---------------------------------------------------------------------------
TEST(P2pBatchStruct, TestBatchNullRequestsAllDone) {
  if (flagcxNetIbP2p.testBatch == nullptr)
    GTEST_SKIP() << "testBatch not implemented";

  void *requests[3] = {nullptr, nullptr, nullptr};
  int doneFlags[3] = {0, 0, 0};
  int doneCount = 0;

  EXPECT_EQ(flagcxNetIbP2p.testBatch(requests, 3, doneFlags, &doneCount),
            flagcxSuccess);
  EXPECT_EQ(doneCount, 3);
  for (int i = 0; i < 3; i++)
    EXPECT_EQ(doneFlags[i], 1);
}

TEST(P2pBatchStruct, TestBatchZeroRequests) {
  if (flagcxNetIbP2p.testBatch == nullptr)
    GTEST_SKIP() << "testBatch not implemented";

  int doneCount = -1;
  EXPECT_EQ(flagcxNetIbP2p.testBatch(nullptr, 0, nullptr, &doneCount),
            flagcxSuccess);
  EXPECT_EQ(doneCount, 0);
}

// ---------------------------------------------------------------------------
// Single iput followed by testBatch (batch of 1)
// ---------------------------------------------------------------------------
TEST_F(P2pBatchTest, IputThenTestBatch) {
  if (flagcxNetIbP2p.testBatch == nullptr)
    GTEST_SKIP() << "testBatch not implemented";

  const size_t bufSize = 4096;
  void *srcBuf = malloc(bufSize);
  void *dstBuf = malloc(bufSize);
  ASSERT_NE(srcBuf, nullptr);
  ASSERT_NE(dstBuf, nullptr);
  memset(srcBuf, 0xCD, bufSize);
  memset(dstBuf, 0, bufSize);

  int mrFlags = FLAGCX_NET_MR_FLAG_NONE;
  void *srcMr = nullptr;
  void *dstMr = nullptr;
  ASSERT_EQ(flagcxNetIbP2p.regMr(sendComm_, srcBuf, bufSize, FLAGCX_PTR_HOST,
                                 mrFlags, &srcMr),
            flagcxSuccess);
  ASSERT_EQ(flagcxNetIbP2p.regMr(sendComm_, dstBuf, bufSize, FLAGCX_PTR_HOST,
                                 mrFlags, &dstMr),
            flagcxSuccess);

  // Issue iput
  void *request = nullptr;
  ASSERT_EQ(flagcxNetIbP2p.iput(sendComm_, 0, 0, bufSize, 0, 0, (void **)srcMr,
                                (void **)dstMr, &request),
            flagcxSuccess);
  ASSERT_NE(request, nullptr);

  // Poll with testBatch (batch of 1)
  void *requests[1] = {request};
  int doneFlags[1] = {0};
  int doneCount = 0;
  int polls = 0;
  while (doneFlags[0] == 0 && polls < 1000000) {
    ASSERT_EQ(flagcxNetIbP2p.testBatch(requests, 1, doneFlags, &doneCount),
              flagcxSuccess);
    polls++;
  }
  EXPECT_EQ(doneFlags[0], 1) << "iput did not complete via testBatch";
  EXPECT_EQ(doneCount, 1);

  // Verify data
  EXPECT_EQ(memcmp(srcBuf, dstBuf, bufSize), 0)
      << "RDMA write via testBatch poll did not transfer correctly";

  flagcxNetIbP2p.deregMr(sendComm_, srcMr);
  flagcxNetIbP2p.deregMr(sendComm_, dstMr);
  free(srcBuf);
  free(dstBuf);
}

// ---------------------------------------------------------------------------
// Multiple iputs followed by testBatch (batch of N)
// ---------------------------------------------------------------------------
TEST_F(P2pBatchTest, MultipleIputsThenTestBatch) {
  if (flagcxNetIbP2p.testBatch == nullptr)
    GTEST_SKIP() << "testBatch not implemented";

  const int numOps = 4;
  const size_t bufSize = 1024;
  void *srcBufs[numOps], *dstBufs[numOps];
  void *srcMrs[numOps], *dstMrs[numOps];
  void *requests[numOps];
  int mrFlags = FLAGCX_NET_MR_FLAG_NONE;

  for (int i = 0; i < numOps; i++) {
    srcBufs[i] = malloc(bufSize);
    dstBufs[i] = malloc(bufSize);
    ASSERT_NE(srcBufs[i], nullptr);
    ASSERT_NE(dstBufs[i], nullptr);
    memset(srcBufs[i], 0x10 + i, bufSize);
    memset(dstBufs[i], 0, bufSize);

    ASSERT_EQ(flagcxNetIbP2p.regMr(sendComm_, srcBufs[i], bufSize,
                                   FLAGCX_PTR_HOST, mrFlags, &srcMrs[i]),
              flagcxSuccess);
    ASSERT_EQ(flagcxNetIbP2p.regMr(sendComm_, dstBufs[i], bufSize,
                                   FLAGCX_PTR_HOST, mrFlags, &dstMrs[i]),
              flagcxSuccess);

    ASSERT_EQ(flagcxNetIbP2p.iput(sendComm_, 0, 0, bufSize, 0, 0,
                                  (void **)srcMrs[i], (void **)dstMrs[i],
                                  &requests[i]),
              flagcxSuccess);
    ASSERT_NE(requests[i], nullptr);
  }

  // Poll all with testBatch
  int doneFlags[numOps] = {};
  int doneCount = 0;
  int polls = 0;
  while (doneCount < numOps && polls < 2000000) {
    ASSERT_EQ(flagcxNetIbP2p.testBatch(requests, numOps, doneFlags, &doneCount),
              flagcxSuccess);
    polls++;
  }
  EXPECT_EQ(doneCount, numOps) << "Not all iputs completed via testBatch";

  // Verify each transfer
  for (int i = 0; i < numOps; i++) {
    EXPECT_EQ(memcmp(srcBufs[i], dstBufs[i], bufSize), 0)
        << "Transfer " << i << " data mismatch";
    flagcxNetIbP2p.deregMr(sendComm_, srcMrs[i]);
    flagcxNetIbP2p.deregMr(sendComm_, dstMrs[i]);
    free(srcBufs[i]);
    free(dstBufs[i]);
  }
}

// ---------------------------------------------------------------------------
// igetBatch: batch READ of multiple regions
// ---------------------------------------------------------------------------
TEST_F(P2pBatchTest, IgetBatchSingleRegion) {
  if (flagcxNetIbP2p.igetBatch == nullptr)
    GTEST_SKIP() << "igetBatch not implemented";

  const size_t bufSize = 4096;
  void *remoteBuf = malloc(bufSize); // "remote" side, source for READ
  void *localBuf = malloc(bufSize);  // "local" side, destination for READ
  ASSERT_NE(remoteBuf, nullptr);
  ASSERT_NE(localBuf, nullptr);
  memset(remoteBuf, 0xEF, bufSize);
  memset(localBuf, 0, bufSize);

  int mrFlags = FLAGCX_NET_MR_FLAG_NONE;
  void *remoteMr = nullptr;
  void *localMr = nullptr;
  ASSERT_EQ(flagcxNetIbP2p.regMr(sendComm_, remoteBuf, bufSize, FLAGCX_PTR_HOST,
                                 mrFlags, &remoteMr),
            flagcxSuccess);
  ASSERT_EQ(flagcxNetIbP2p.regMr(sendComm_, localBuf, bufSize, FLAGCX_PTR_HOST,
                                 mrFlags, &localMr),
            flagcxSuccess);

  // Issue batch read of 1 region
  uint64_t srcOffs[1] = {0};
  uint64_t dstOffs[1] = {0};
  size_t sizes[1] = {bufSize};
  void *srcHandles[1] = {remoteMr};
  void *dstHandles[1] = {localMr};
  void *request = nullptr;

  ASSERT_EQ(flagcxNetIbP2p.igetBatch(sendComm_, 1, srcOffs, dstOffs, sizes, 0,
                                     0, srcHandles, dstHandles, &request),
            flagcxSuccess);
  ASSERT_NE(request, nullptr);

  // Poll until done
  int done = 0;
  int polls = 0;
  while (!done && polls < 1000000) {
    ASSERT_EQ(flagcxNetIbP2p.test(request, &done, nullptr), flagcxSuccess);
    polls++;
  }
  EXPECT_TRUE(done) << "igetBatch did not complete within poll limit";

  // Verify read data
  EXPECT_EQ(memcmp(remoteBuf, localBuf, bufSize), 0)
      << "igetBatch READ did not transfer data correctly";

  flagcxNetIbP2p.deregMr(sendComm_, remoteMr);
  flagcxNetIbP2p.deregMr(sendComm_, localMr);
  free(remoteBuf);
  free(localBuf);
}

// ---------------------------------------------------------------------------
// igetBatch: batch READ of multiple regions with testBatch polling
// ---------------------------------------------------------------------------
TEST_F(P2pBatchTest, IgetBatchMultipleRegions) {
  if (flagcxNetIbP2p.igetBatch == nullptr)
    GTEST_SKIP() << "igetBatch not implemented";
  if (flagcxNetIbP2p.testBatch == nullptr)
    GTEST_SKIP() << "testBatch not implemented";

  const int count = 3;
  const size_t sizes[3] = {512, 1024, 2048};
  void *srcBufs[count], *dstBufs[count];
  void *srcMrs[count], *dstMrs[count];
  uint64_t srcOffs[count] = {0, 0, 0};
  uint64_t dstOffs[count] = {0, 0, 0};
  int mrFlags = FLAGCX_NET_MR_FLAG_NONE;

  for (int i = 0; i < count; i++) {
    srcBufs[i] = malloc(sizes[i]);
    dstBufs[i] = malloc(sizes[i]);
    ASSERT_NE(srcBufs[i], nullptr);
    ASSERT_NE(dstBufs[i], nullptr);
    memset(srcBufs[i], 0x30 + i, sizes[i]);
    memset(dstBufs[i], 0, sizes[i]);

    ASSERT_EQ(flagcxNetIbP2p.regMr(sendComm_, srcBufs[i], sizes[i],
                                   FLAGCX_PTR_HOST, mrFlags, &srcMrs[i]),
              flagcxSuccess);
    ASSERT_EQ(flagcxNetIbP2p.regMr(sendComm_, dstBufs[i], sizes[i],
                                   FLAGCX_PTR_HOST, mrFlags, &dstMrs[i]),
              flagcxSuccess);
  }

  // Issue batch read
  void *request = nullptr;
  ASSERT_EQ(flagcxNetIbP2p.igetBatch(sendComm_, count, srcOffs, dstOffs, sizes,
                                     0, 0, (void *const *)srcMrs,
                                     (void *const *)dstMrs, &request),
            flagcxSuccess);
  ASSERT_NE(request, nullptr);

  // Poll with testBatch (single request in batch)
  void *requests[1] = {request};
  int doneFlags[1] = {0};
  int doneCount = 0;
  int polls = 0;
  while (doneFlags[0] == 0 && polls < 1000000) {
    flagcxResult_t rc =
        flagcxNetIbP2p.testBatch(requests, 1, doneFlags, &doneCount);
    ASSERT_EQ(rc, flagcxSuccess);
    polls++;
  }
  EXPECT_EQ(doneFlags[0], 1) << "igetBatch multi-region did not complete";

  // Verify each region
  for (int i = 0; i < count; i++) {
    EXPECT_EQ(memcmp(srcBufs[i], dstBufs[i], sizes[i]), 0)
        << "igetBatch region " << i << " data mismatch";
    flagcxNetIbP2p.deregMr(sendComm_, srcMrs[i]);
    flagcxNetIbP2p.deregMr(sendComm_, dstMrs[i]);
    free(srcBufs[i]);
    free(dstBufs[i]);
  }
}

// ---------------------------------------------------------------------------
// igetBatch: invalid arguments return error
// ---------------------------------------------------------------------------
TEST_F(P2pBatchTest, IgetBatchInvalidCountReturnsError) {
  if (flagcxNetIbP2p.igetBatch == nullptr)
    GTEST_SKIP() << "igetBatch not implemented";

  void *request = nullptr;
  // count=0 should be handled gracefully (either success with NULL req or
  // error)
  (void)flagcxNetIbP2p.igetBatch(sendComm_, 0, nullptr, nullptr, nullptr, 0, 0,
                                 nullptr, nullptr, &request);
  // Negative count should fail
  flagcxResult_t rcNeg =
      flagcxNetIbP2p.igetBatch(sendComm_, -1, nullptr, nullptr, nullptr, 0, 0,
                               nullptr, nullptr, &request);
  EXPECT_NE(rcNeg, flagcxSuccess);
}
