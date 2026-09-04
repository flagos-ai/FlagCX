// Adaptor unit tests for the IB P2P batch APIs (iputBatch, testBatch,
// igetBatch).
// These tests require IB hardware and use loopback connections.
// Missing IB devices or required batch callbacks are test failures.

#include <cstring>
#include <future>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

#include "flagcx_net.h"
#include "flagcx_net_adaptor.h"
#include "onesided.h"

extern struct flagcxNetAdaptor flagcxP2pNetIb;

namespace {

struct P2pTestWindow {
  struct flagcxOneSideHandleInfo info = {};
  uintptr_t baseVa = 0;
  size_t regionSize = 0;
  struct flagcxNetMrInfo mrInfo = {};

  flagcxResult_t init(void *buffer, size_t size, void *mrHandle) {
    flagcxResult_t result = flagcxP2pNetIb.getMrInfo(mrHandle, &mrInfo);
    if (result != flagcxSuccess)
      return result;
    baseVa = reinterpret_cast<uintptr_t>(buffer);
    regionSize = size;
    info.baseVas = &baseVa;
    info.regionSize = size;
    info.regionSizes = &regionSize;
    info.mrInfos = &mrInfo;
    info.localMrHandle = mrHandle;
    info.nRanks = 1;
    info.signalIpcSlot = -1;
    return flagcxSuccess;
  }

  void **opaque() { return reinterpret_cast<void **>(&info); }
};

} // namespace

// ---------------------------------------------------------------------------
// Fixture: establishes a loopback connection for batch operation testing
// ---------------------------------------------------------------------------
class P2pBatchTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    initResult_ = flagcxP2pNetIb.init();
    if (initResult_ == flagcxSuccess)
      flagcxP2pNetIb.devices(&nDevs_);
  }

  void SetUp() override {
    ASSERT_EQ(initResult_, flagcxSuccess) << "P2P net adaptor init failed";
    ASSERT_GT(nDevs_, 0) << "No IB devices available";

    // Establish loopback connection
    ASSERT_EQ(flagcxP2pNetIb.listen(0, handle_, &listenComm_), flagcxSuccess);

    auto acceptFut = std::async(std::launch::async, [this]() {
      void *comm = nullptr;
      flagcxP2pNetIb.accept(listenComm_, &comm);
      return comm;
    });
    auto connectFut = std::async(std::launch::async, [this]() {
      void *comm = nullptr;
      flagcxP2pNetIb.connect(0, handle_, &comm);
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
      flagcxP2pNetIb.closeSend(sendComm_);
    if (recvComm_)
      flagcxP2pNetIb.closeRecv(recvComm_);
    if (listenComm_)
      flagcxP2pNetIb.closeListen(listenComm_);
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
  EXPECT_NE(flagcxP2pNetIb.testBatch, nullptr);
}

TEST(P2pBatchStruct, IputBatchFunctionExists) {
  EXPECT_NE(flagcxP2pNetIb.iputBatch, nullptr);
}

TEST(P2pBatchStruct, IgetBatchFunctionExists) {
  // igetBatch is optional but should be non-NULL in the optimized adaptor
  EXPECT_NE(flagcxP2pNetIb.igetBatch, nullptr);
}

// ---------------------------------------------------------------------------
// testBatch with NULL requests reports all done
// ---------------------------------------------------------------------------
TEST(P2pBatchStruct, TestBatchNullRequestsAllDone) {
  ASSERT_NE(flagcxP2pNetIb.testBatch, nullptr);

  void *requests[3] = {nullptr, nullptr, nullptr};
  int doneFlags[3] = {0, 0, 0};
  int doneCount = 0;

  EXPECT_EQ(flagcxP2pNetIb.testBatch(requests, 3, doneFlags, &doneCount),
            flagcxSuccess);
  EXPECT_EQ(doneCount, 3);
  for (int i = 0; i < 3; i++)
    EXPECT_EQ(doneFlags[i], 1);
}

TEST(P2pBatchStruct, TestBatchZeroRequests) {
  ASSERT_NE(flagcxP2pNetIb.testBatch, nullptr);

  int doneCount = -1;
  EXPECT_EQ(flagcxP2pNetIb.testBatch(nullptr, 0, nullptr, &doneCount),
            flagcxSuccess);
  EXPECT_EQ(doneCount, 0);
}

TEST(P2pBatchStruct, IputBatchZeroOperations) {
  ASSERT_NE(flagcxP2pNetIb.iputBatch, nullptr);

  void *requests[1] = {nullptr};
  int posted = -1;
  EXPECT_EQ(flagcxP2pNetIb.iputBatch(nullptr, 0, nullptr, nullptr, nullptr, 0,
                                     0, nullptr, nullptr, requests, &posted),
            flagcxSuccess);
  EXPECT_EQ(posted, 0);
}

// ---------------------------------------------------------------------------
// iputBatch: chained prefix submission with an independent request per WRITE
// ---------------------------------------------------------------------------
TEST_F(P2pBatchTest, IputBatchTransfersMultipleRegions) {
  ASSERT_NE(flagcxP2pNetIb.iputBatch, nullptr);
  ASSERT_NE(flagcxP2pNetIb.testBatch, nullptr);

  constexpr int count = 3;
  constexpr size_t totalSize = 4096;
  const uint64_t srcOffs[count] = {0, 640, 2048};
  const uint64_t dstOffs[count] = {128, 1280, 3072};
  const size_t sizes[count] = {257, 513, 777};
  std::vector<unsigned char> srcBuf(totalSize, 0);
  std::vector<unsigned char> dstBuf(totalSize, 0);
  for (int i = 0; i < count; i++)
    memset(srcBuf.data() + srcOffs[i], 0x40 + i, sizes[i]);

  int mrFlags = FLAGCX_NET_MR_FLAG_NONE;
  void *srcMr = nullptr;
  void *dstMr = nullptr;
  ASSERT_EQ(flagcxP2pNetIb.regMr(sendComm_, srcBuf.data(), totalSize,
                                 FLAGCX_PTR_HOST, mrFlags, &srcMr),
            flagcxSuccess);
  ASSERT_EQ(flagcxP2pNetIb.regMr(sendComm_, dstBuf.data(), totalSize,
                                 FLAGCX_PTR_HOST, mrFlags, &dstMr),
            flagcxSuccess);
  P2pTestWindow srcWindow, dstWindow;
  ASSERT_EQ(srcWindow.init(srcBuf.data(), totalSize, srcMr), flagcxSuccess);
  ASSERT_EQ(dstWindow.init(dstBuf.data(), totalSize, dstMr), flagcxSuccess);

  int submitted = 0;
  while (submitted < count) {
    void *requests[count] = {};
    int posted = 0;
    ASSERT_EQ(flagcxP2pNetIb.iputBatch(
                  sendComm_, count - submitted, srcOffs + submitted,
                  dstOffs + submitted, sizes + submitted, 0, 0,
                  srcWindow.opaque(), dstWindow.opaque(), requests, &posted),
              flagcxSuccess);
    ASSERT_GT(posted, 0);
    ASSERT_LE(posted, count - submitted);
    for (int i = 0; i < posted; i++)
      ASSERT_NE(requests[i], nullptr);

    int doneFlags[count] = {};
    int doneCount = 0;
    int polls = 0;
    while (doneCount < posted && polls < 1000000) {
      ASSERT_EQ(
          flagcxP2pNetIb.testBatch(requests, posted, doneFlags, &doneCount),
          flagcxSuccess);
      polls++;
    }
    ASSERT_EQ(doneCount, posted) << "iputBatch prefix did not complete";
    submitted += posted;
  }
  ASSERT_EQ(submitted, count);
  for (int i = 0; i < count; i++) {
    EXPECT_EQ(memcmp(srcBuf.data() + srcOffs[i], dstBuf.data() + dstOffs[i],
                     sizes[i]),
              0)
        << "iputBatch region " << i << " data mismatch";
  }

  flagcxP2pNetIb.deregMr(sendComm_, srcMr);
  flagcxP2pNetIb.deregMr(sendComm_, dstMr);
}

// ---------------------------------------------------------------------------
// Single iput followed by testBatch (batch of 1)
// ---------------------------------------------------------------------------
TEST_F(P2pBatchTest, IputThenTestBatch) {
  ASSERT_NE(flagcxP2pNetIb.testBatch, nullptr);

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
  ASSERT_EQ(flagcxP2pNetIb.regMr(sendComm_, srcBuf, bufSize, FLAGCX_PTR_HOST,
                                 mrFlags, &srcMr),
            flagcxSuccess);
  ASSERT_EQ(flagcxP2pNetIb.regMr(sendComm_, dstBuf, bufSize, FLAGCX_PTR_HOST,
                                 mrFlags, &dstMr),
            flagcxSuccess);
  P2pTestWindow srcWindow, dstWindow;
  ASSERT_EQ(srcWindow.init(srcBuf, bufSize, srcMr), flagcxSuccess);
  ASSERT_EQ(dstWindow.init(dstBuf, bufSize, dstMr), flagcxSuccess);

  // Issue iput
  void *request = nullptr;
  ASSERT_EQ(flagcxP2pNetIb.iput(sendComm_, 0, 0, bufSize, 0, 0,
                                srcWindow.opaque(), dstWindow.opaque(),
                                &request),
            flagcxSuccess);
  ASSERT_NE(request, nullptr);

  // Poll with testBatch (batch of 1)
  void *requests[1] = {request};
  int doneFlags[1] = {0};
  int doneCount = 0;
  int polls = 0;
  while (doneFlags[0] == 0 && polls < 1000000) {
    ASSERT_EQ(flagcxP2pNetIb.testBatch(requests, 1, doneFlags, &doneCount),
              flagcxSuccess);
    polls++;
  }
  EXPECT_EQ(doneFlags[0], 1) << "iput did not complete via testBatch";
  EXPECT_EQ(doneCount, 1);

  // Verify data
  EXPECT_EQ(memcmp(srcBuf, dstBuf, bufSize), 0)
      << "RDMA write via testBatch poll did not transfer correctly";

  flagcxP2pNetIb.deregMr(sendComm_, srcMr);
  flagcxP2pNetIb.deregMr(sendComm_, dstMr);
  free(srcBuf);
  free(dstBuf);
}

// ---------------------------------------------------------------------------
// Multiple iputs followed by testBatch (batch of N)
// ---------------------------------------------------------------------------
TEST_F(P2pBatchTest, MultipleIputsThenTestBatch) {
  ASSERT_NE(flagcxP2pNetIb.testBatch, nullptr);

  const int numOps = 4;
  const size_t bufSize = 1024;
  void *srcBufs[numOps], *dstBufs[numOps];
  void *srcMrs[numOps], *dstMrs[numOps];
  P2pTestWindow srcWindows[numOps], dstWindows[numOps];
  void *requests[numOps];
  int mrFlags = FLAGCX_NET_MR_FLAG_NONE;

  for (int i = 0; i < numOps; i++) {
    srcBufs[i] = malloc(bufSize);
    dstBufs[i] = malloc(bufSize);
    ASSERT_NE(srcBufs[i], nullptr);
    ASSERT_NE(dstBufs[i], nullptr);
    memset(srcBufs[i], 0x10 + i, bufSize);
    memset(dstBufs[i], 0, bufSize);

    ASSERT_EQ(flagcxP2pNetIb.regMr(sendComm_, srcBufs[i], bufSize,
                                   FLAGCX_PTR_HOST, mrFlags, &srcMrs[i]),
              flagcxSuccess);
    ASSERT_EQ(flagcxP2pNetIb.regMr(sendComm_, dstBufs[i], bufSize,
                                   FLAGCX_PTR_HOST, mrFlags, &dstMrs[i]),
              flagcxSuccess);
    ASSERT_EQ(srcWindows[i].init(srcBufs[i], bufSize, srcMrs[i]),
              flagcxSuccess);
    ASSERT_EQ(dstWindows[i].init(dstBufs[i], bufSize, dstMrs[i]),
              flagcxSuccess);

    ASSERT_EQ(flagcxP2pNetIb.iput(sendComm_, 0, 0, bufSize, 0, 0,
                                  srcWindows[i].opaque(),
                                  dstWindows[i].opaque(), &requests[i]),
              flagcxSuccess);
    ASSERT_NE(requests[i], nullptr);
  }

  // Poll all with testBatch
  int doneFlags[numOps] = {};
  int doneCount = 0;
  int polls = 0;
  while (doneCount < numOps && polls < 2000000) {
    ASSERT_EQ(flagcxP2pNetIb.testBatch(requests, numOps, doneFlags, &doneCount),
              flagcxSuccess);
    polls++;
  }
  EXPECT_EQ(doneCount, numOps) << "Not all iputs completed via testBatch";

  // Verify each transfer
  for (int i = 0; i < numOps; i++) {
    EXPECT_EQ(memcmp(srcBufs[i], dstBufs[i], bufSize), 0)
        << "Transfer " << i << " data mismatch";
    flagcxP2pNetIb.deregMr(sendComm_, srcMrs[i]);
    flagcxP2pNetIb.deregMr(sendComm_, dstMrs[i]);
    free(srcBufs[i]);
    free(dstBufs[i]);
  }
}

// ---------------------------------------------------------------------------
// igetBatch: batch READ of multiple regions
// ---------------------------------------------------------------------------
TEST_F(P2pBatchTest, IgetBatchSingleRegion) {
  ASSERT_NE(flagcxP2pNetIb.igetBatch, nullptr);

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
  ASSERT_EQ(flagcxP2pNetIb.regMr(sendComm_, remoteBuf, bufSize, FLAGCX_PTR_HOST,
                                 mrFlags, &remoteMr),
            flagcxSuccess);
  ASSERT_EQ(flagcxP2pNetIb.regMr(sendComm_, localBuf, bufSize, FLAGCX_PTR_HOST,
                                 mrFlags, &localMr),
            flagcxSuccess);
  P2pTestWindow remoteWindow, localWindow;
  ASSERT_EQ(remoteWindow.init(remoteBuf, bufSize, remoteMr), flagcxSuccess);
  ASSERT_EQ(localWindow.init(localBuf, bufSize, localMr), flagcxSuccess);

  // Issue batch read of 1 region
  uint64_t srcOffs[1] = {0};
  uint64_t dstOffs[1] = {0};
  size_t sizes[1] = {bufSize};
  void *request = nullptr;

  ASSERT_EQ(flagcxP2pNetIb.igetBatch(sendComm_, 1, srcOffs, dstOffs, sizes, 0,
                                     0, remoteWindow.opaque(),
                                     localWindow.opaque(), &request),
            flagcxSuccess);
  ASSERT_NE(request, nullptr);

  // Poll until done
  int done = 0;
  int polls = 0;
  while (!done && polls < 1000000) {
    ASSERT_EQ(flagcxP2pNetIb.test(request, &done, nullptr), flagcxSuccess);
    polls++;
  }
  EXPECT_TRUE(done) << "igetBatch did not complete within poll limit";

  // Verify read data
  EXPECT_EQ(memcmp(remoteBuf, localBuf, bufSize), 0)
      << "igetBatch READ did not transfer data correctly";

  flagcxP2pNetIb.deregMr(sendComm_, remoteMr);
  flagcxP2pNetIb.deregMr(sendComm_, localMr);
  free(remoteBuf);
  free(localBuf);
}

// ---------------------------------------------------------------------------
// igetBatch: batch READ of multiple regions with testBatch polling
// ---------------------------------------------------------------------------
TEST_F(P2pBatchTest, IgetBatchMultipleRegions) {
  ASSERT_NE(flagcxP2pNetIb.igetBatch, nullptr);
  ASSERT_NE(flagcxP2pNetIb.testBatch, nullptr);

  const int count = 3;
  const size_t sizes[3] = {512, 1024, 2048};
  const uint64_t srcOffs[count] = {0, 512, 1536};
  const uint64_t dstOffs[count] = {0, 512, 1536};
  const size_t totalSize = 3584;
  std::vector<unsigned char> srcBuf(totalSize);
  std::vector<unsigned char> dstBuf(totalSize, 0);
  void *srcMr = nullptr;
  void *dstMr = nullptr;
  int mrFlags = FLAGCX_NET_MR_FLAG_NONE;

  for (int i = 0; i < count; i++) {
    memset(srcBuf.data() + srcOffs[i], 0x30 + i, sizes[i]);
  }
  ASSERT_EQ(flagcxP2pNetIb.regMr(sendComm_, srcBuf.data(), totalSize,
                                 FLAGCX_PTR_HOST, mrFlags, &srcMr),
            flagcxSuccess);
  ASSERT_EQ(flagcxP2pNetIb.regMr(sendComm_, dstBuf.data(), totalSize,
                                 FLAGCX_PTR_HOST, mrFlags, &dstMr),
            flagcxSuccess);
  P2pTestWindow srcWindow, dstWindow;
  ASSERT_EQ(srcWindow.init(srcBuf.data(), totalSize, srcMr), flagcxSuccess);
  ASSERT_EQ(dstWindow.init(dstBuf.data(), totalSize, dstMr), flagcxSuccess);

  // Issue batch read
  void *request = nullptr;
  ASSERT_EQ(flagcxP2pNetIb.igetBatch(sendComm_, count, srcOffs, dstOffs, sizes,
                                     0, 0, srcWindow.opaque(),
                                     dstWindow.opaque(), &request),
            flagcxSuccess);
  ASSERT_NE(request, nullptr);

  // Poll with testBatch (single request in batch)
  void *requests[1] = {request};
  int doneFlags[1] = {0};
  int doneCount = 0;
  int polls = 0;
  while (doneFlags[0] == 0 && polls < 1000000) {
    flagcxResult_t rc =
        flagcxP2pNetIb.testBatch(requests, 1, doneFlags, &doneCount);
    ASSERT_EQ(rc, flagcxSuccess);
    polls++;
  }
  EXPECT_EQ(doneFlags[0], 1) << "igetBatch multi-region did not complete";

  // Verify each region
  for (int i = 0; i < count; i++) {
    EXPECT_EQ(memcmp(srcBuf.data() + srcOffs[i], dstBuf.data() + dstOffs[i],
                     sizes[i]),
              0)
        << "igetBatch region " << i << " data mismatch";
  }
  flagcxP2pNetIb.deregMr(sendComm_, srcMr);
  flagcxP2pNetIb.deregMr(sendComm_, dstMr);
}

// ---------------------------------------------------------------------------
// igetBatch: invalid arguments return error
// ---------------------------------------------------------------------------
TEST_F(P2pBatchTest, IgetBatchInvalidCountReturnsError) {
  ASSERT_NE(flagcxP2pNetIb.igetBatch, nullptr);

  void *request = nullptr;
  // count=0 should be handled gracefully (either success with NULL req or
  // error)
  (void)flagcxP2pNetIb.igetBatch(sendComm_, 0, nullptr, nullptr, nullptr, 0, 0,
                                 nullptr, nullptr, &request);
  // Negative count should fail
  flagcxResult_t rcNeg =
      flagcxP2pNetIb.igetBatch(sendComm_, -1, nullptr, nullptr, nullptr, 0, 0,
                               nullptr, nullptr, &request);
  EXPECT_NE(rcNeg, flagcxSuccess);
}
