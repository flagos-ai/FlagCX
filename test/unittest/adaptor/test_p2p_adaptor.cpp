// Adaptor unit tests for the IB P2P net adaptor.
// Tests that don't require IB hardware always run.
// Tests that need real IB devices fail when the devices are unavailable.
// Links against libflagcx.

#include <cstring>
#include <future>
#include <gtest/gtest.h>
#include <infiniband/verbs.h>
#include <thread>

#include "flagcx_net.h"
#include "flagcx_net_adaptor.h"
#include "onesided.h"

// The P2P adaptor struct is non-static in ibrc_p2p_adaptor.cc
extern struct flagcxNetAdaptor flagcxP2pNetIb;

// ---------------------------------------------------------------------------
// Fixture: initializes the adaptor once, caches device count
// ---------------------------------------------------------------------------
class P2pAdaptorTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    initResult = flagcxP2pNetIb.init();
    if (initResult == flagcxSuccess) {
      flagcxP2pNetIb.devices(&nDevs);
    }
  }

  void SetUp() override {
    ASSERT_EQ(initResult, flagcxSuccess) << "P2P net adaptor init failed";
    ASSERT_GT(nDevs, 0) << "No IB devices available";
  }

  static flagcxResult_t initResult;
  static int nDevs;
};

flagcxResult_t P2pAdaptorTest::initResult = flagcxInternalError;
int P2pAdaptorTest::nDevs = 0;

// ---------------------------------------------------------------------------
// 1. Adaptor struct completeness — always runs, no hardware needed
// ---------------------------------------------------------------------------
TEST(P2pAdaptorStruct, AllFunctionPointersSet) {
  EXPECT_NE(flagcxP2pNetIb.name, nullptr);
  EXPECT_STREQ(flagcxP2pNetIb.name, "IB_P2P");

  // Basic
  EXPECT_NE(flagcxP2pNetIb.init, nullptr);
  EXPECT_NE(flagcxP2pNetIb.devices, nullptr);
  EXPECT_NE(flagcxP2pNetIb.getProperties, nullptr);

  // Connection setup
  EXPECT_NE(flagcxP2pNetIb.listen, nullptr);
  EXPECT_NE(flagcxP2pNetIb.connect, nullptr);
  EXPECT_NE(flagcxP2pNetIb.accept, nullptr);
  EXPECT_NE(flagcxP2pNetIb.closeSend, nullptr);
  EXPECT_NE(flagcxP2pNetIb.closeRecv, nullptr);
  EXPECT_NE(flagcxP2pNetIb.closeListen, nullptr);

  // Memory registration
  EXPECT_NE(flagcxP2pNetIb.regMr, nullptr);
  EXPECT_NE(flagcxP2pNetIb.regMrDmaBuf, nullptr);
  EXPECT_NE(flagcxP2pNetIb.deregMr, nullptr);

  // Two-sided (stubs)
  EXPECT_NE(flagcxP2pNetIb.isend, nullptr);
  EXPECT_NE(flagcxP2pNetIb.irecv, nullptr);
  EXPECT_NE(flagcxP2pNetIb.iflush, nullptr);
  EXPECT_NE(flagcxP2pNetIb.test, nullptr);

  // One-sided
  EXPECT_NE(flagcxP2pNetIb.iput, nullptr);
  EXPECT_NE(flagcxP2pNetIb.iget, nullptr);
  EXPECT_NE(flagcxP2pNetIb.iputSignal, nullptr);
  EXPECT_NE(flagcxP2pNetIb.getMrInfo, nullptr);

  // Device lookup
  EXPECT_NE(flagcxP2pNetIb.getDevFromName, nullptr);
}

// Two-sided stubs should return errors
TEST(P2pAdaptorStruct, TwoSidedStubsReturnError) {
  void *dummy = nullptr;
  EXPECT_NE(
      flagcxP2pNetIb.isend(nullptr, nullptr, 0, 0, nullptr, nullptr, &dummy),
      flagcxSuccess);
  EXPECT_NE(flagcxP2pNetIb.irecv(nullptr, 0, nullptr, nullptr, nullptr, nullptr,
                                 nullptr, &dummy),
            flagcxSuccess);
  EXPECT_NE(
      flagcxP2pNetIb.iflush(nullptr, 0, nullptr, nullptr, nullptr, &dummy),
      flagcxSuccess);
}

// ---------------------------------------------------------------------------
// 2. Init + Devices — requires IB hardware
// ---------------------------------------------------------------------------
TEST_F(P2pAdaptorTest, InitSucceeds) { EXPECT_EQ(initResult, flagcxSuccess); }

TEST_F(P2pAdaptorTest, DevicesReturnsPositive) { EXPECT_GT(nDevs, 0); }

TEST_F(P2pAdaptorTest, InitIsIdempotent) {
  // Calling init again should succeed without side effects
  EXPECT_EQ(flagcxP2pNetIb.init(), flagcxSuccess);
  int nDevs2 = 0;
  EXPECT_EQ(flagcxP2pNetIb.devices(&nDevs2), flagcxSuccess);
  EXPECT_EQ(nDevs2, nDevs);
}

// ---------------------------------------------------------------------------
// 3. GetProperties — requires IB hardware
// ---------------------------------------------------------------------------
TEST_F(P2pAdaptorTest, GetPropertiesForEachDevice) {
  for (int d = 0; d < nDevs; d++) {
    flagcxNetProperties_t props;
    memset(&props, 0, sizeof(props));
    EXPECT_EQ(flagcxP2pNetIb.getProperties(d, &props), flagcxSuccess);
    EXPECT_GT(props.speed, 0);
    EXPECT_NE(props.name, nullptr);
  }
}

// ---------------------------------------------------------------------------
// 4. Listen + Connect + Accept loopback — requires IB hardware
// ---------------------------------------------------------------------------
class P2pLoopbackTest : public P2pAdaptorTest {
protected:
  void SetUp() override { P2pAdaptorTest::SetUp(); }
};

TEST_F(P2pLoopbackTest, ListenConnectAcceptClose) {
  // Listen
  char handle[FLAGCX_NET_HANDLE_MAXSIZE];
  void *listenComm = nullptr;
  ASSERT_EQ(flagcxP2pNetIb.listen(0, handle, &listenComm), flagcxSuccess);
  ASSERT_NE(listenComm, nullptr);

  // Connect + Accept in parallel using std::async with timeout
  auto acceptFuture = std::async(std::launch::async, [&]() {
    void *comm = nullptr;
    flagcxResult_t r = flagcxP2pNetIb.accept(listenComm, &comm);
    return std::make_pair(r, comm);
  });

  auto connectFuture = std::async(std::launch::async, [&]() {
    void *comm = nullptr;
    flagcxResult_t r = flagcxP2pNetIb.connect(0, handle, &comm);
    return std::make_pair(r, comm);
  });

  // Wait with timeout to avoid hanging forever
  auto timeout = std::chrono::seconds(10);

  ASSERT_EQ(connectFuture.wait_for(timeout), std::future_status::ready)
      << "connect() timed out after 10s";
  const std::pair<flagcxResult_t, void *> connectEndpoint = connectFuture.get();
  const flagcxResult_t connectResult = connectEndpoint.first;
  void *sendComm = connectEndpoint.second;

  ASSERT_EQ(acceptFuture.wait_for(timeout), std::future_status::ready)
      << "accept() timed out after 10s";
  const std::pair<flagcxResult_t, void *> acceptEndpoint = acceptFuture.get();
  const flagcxResult_t acceptResult = acceptEndpoint.first;
  void *recvComm = acceptEndpoint.second;

  ASSERT_EQ(connectResult, flagcxSuccess) << "connect() failed";
  ASSERT_EQ(acceptResult, flagcxSuccess) << "accept() failed";
  ASSERT_NE(sendComm, nullptr);
  ASSERT_NE(recvComm, nullptr);

  // Close
  EXPECT_EQ(flagcxP2pNetIb.closeSend(sendComm), flagcxSuccess);
  EXPECT_EQ(flagcxP2pNetIb.closeRecv(recvComm), flagcxSuccess);
  EXPECT_EQ(flagcxP2pNetIb.closeListen(listenComm), flagcxSuccess);
}

// ---------------------------------------------------------------------------
// 5. RegMr + DeregMr — requires IB hardware + a loopback connection
// ---------------------------------------------------------------------------
TEST_F(P2pLoopbackTest, RegMrDeregMr) {
  // Set up loopback connection
  char handle[FLAGCX_NET_HANDLE_MAXSIZE];
  void *listenComm = nullptr;
  ASSERT_EQ(flagcxP2pNetIb.listen(0, handle, &listenComm), flagcxSuccess);

  auto acceptFuture = std::async(std::launch::async, [&]() {
    void *comm = nullptr;
    flagcxP2pNetIb.accept(listenComm, &comm);
    return comm;
  });
  auto connectFuture = std::async(std::launch::async, [&]() {
    void *comm = nullptr;
    flagcxP2pNetIb.connect(0, handle, &comm);
    return comm;
  });

  auto timeout = std::chrono::seconds(10);
  ASSERT_EQ(connectFuture.wait_for(timeout), std::future_status::ready)
      << "connect() timed out";
  void *sendComm = connectFuture.get();
  ASSERT_EQ(acceptFuture.wait_for(timeout), std::future_status::ready)
      << "accept() timed out";
  void *recvComm = acceptFuture.get();
  ASSERT_NE(sendComm, nullptr);
  ASSERT_NE(recvComm, nullptr);

  // Register MR on send side
  const size_t bufSize = 4096;
  void *buf = malloc(bufSize);
  ASSERT_NE(buf, nullptr);
  memset(buf, 0, bufSize);

  void *mhandle = nullptr;
  int mrFlags = FLAGCX_NET_MR_FLAG_NONE;
  EXPECT_EQ(flagcxP2pNetIb.regMr(sendComm, buf, bufSize, FLAGCX_PTR_HOST,
                                 mrFlags, &mhandle),
            flagcxSuccess);
  EXPECT_NE(mhandle, nullptr);

  // Deregister
  EXPECT_EQ(flagcxP2pNetIb.deregMr(sendComm, mhandle), flagcxSuccess);

  // Register on recv side too (symmetric)
  void *mhandle2 = nullptr;
  EXPECT_EQ(flagcxP2pNetIb.regMr(recvComm, buf, bufSize, FLAGCX_PTR_HOST,
                                 mrFlags, &mhandle2),
            flagcxSuccess);
  EXPECT_NE(mhandle2, nullptr);
  EXPECT_EQ(flagcxP2pNetIb.deregMr(recvComm, mhandle2), flagcxSuccess);

  free(buf);
  flagcxP2pNetIb.closeSend(sendComm);
  flagcxP2pNetIb.closeRecv(recvComm);
  flagcxP2pNetIb.closeListen(listenComm);
}

// ---------------------------------------------------------------------------
// 6. Iput + Test — requires IB hardware + loopback
// ---------------------------------------------------------------------------
TEST_F(P2pLoopbackTest, IputAndTest) {
  // Set up loopback connection
  char handle[FLAGCX_NET_HANDLE_MAXSIZE];
  void *listenComm = nullptr;
  ASSERT_EQ(flagcxP2pNetIb.listen(0, handle, &listenComm), flagcxSuccess);

  auto acceptFuture = std::async(std::launch::async, [&]() {
    void *comm = nullptr;
    flagcxP2pNetIb.accept(listenComm, &comm);
    return comm;
  });
  auto connectFuture = std::async(std::launch::async, [&]() {
    void *comm = nullptr;
    flagcxP2pNetIb.connect(0, handle, &comm);
    return comm;
  });

  auto timeout = std::chrono::seconds(10);
  ASSERT_EQ(connectFuture.wait_for(timeout), std::future_status::ready)
      << "connect() timed out";
  void *sendComm = connectFuture.get();
  ASSERT_EQ(acceptFuture.wait_for(timeout), std::future_status::ready)
      << "accept() timed out";
  void *recvComm = acceptFuture.get();
  ASSERT_NE(sendComm, nullptr);
  ASSERT_NE(recvComm, nullptr);

  // Allocate and register src + dst buffers
  const size_t bufSize = 4096;
  void *srcBuf = malloc(bufSize);
  void *dstBuf = malloc(bufSize);
  ASSERT_NE(srcBuf, nullptr);
  ASSERT_NE(dstBuf, nullptr);

  // Fill src with pattern, dst with zeros
  memset(srcBuf, 0xAB, bufSize);
  memset(dstBuf, 0, bufSize);

  int mrFlags = FLAGCX_NET_MR_FLAG_NONE;

  void *srcMr = nullptr;
  void *dstMr = nullptr;
  ASSERT_EQ(flagcxP2pNetIb.regMr(sendComm, srcBuf, bufSize, FLAGCX_PTR_HOST,
                                 mrFlags, &srcMr),
            flagcxSuccess);
  ASSERT_EQ(flagcxP2pNetIb.regMr(sendComm, dstBuf, bufSize, FLAGCX_PTR_HOST,
                                 mrFlags, &dstMr),
            flagcxSuccess);

  uintptr_t srcBase = reinterpret_cast<uintptr_t>(srcBuf);
  uintptr_t dstBase = reinterpret_cast<uintptr_t>(dstBuf);
  size_t srcRegionSize = bufSize;
  size_t dstRegionSize = bufSize;
  flagcxNetMrInfo srcMrInfo = {};
  flagcxNetMrInfo dstMrInfo = {};
  ASSERT_EQ(flagcxP2pNetIb.getMrInfo(srcMr, &srcMrInfo), flagcxSuccess);
  ASSERT_EQ(flagcxP2pNetIb.getMrInfo(dstMr, &dstMrInfo), flagcxSuccess);

  flagcxOneSideHandleInfo srcInfo = {};
  srcInfo.baseVas = &srcBase;
  srcInfo.regionSize = bufSize;
  srcInfo.regionSizes = &srcRegionSize;
  srcInfo.mrInfos = &srcMrInfo;
  srcInfo.localMrHandle = srcMr;
  srcInfo.nRanks = 1;

  flagcxOneSideHandleInfo dstInfo = {};
  dstInfo.baseVas = &dstBase;
  dstInfo.regionSize = bufSize;
  dstInfo.regionSizes = &dstRegionSize;
  dstInfo.mrInfos = &dstMrInfo;
  dstInfo.localMrHandle = dstMr;
  dstInfo.nRanks = 1;

  // Iput: write srcBuf -> dstBuf via RDMA
  void *request = nullptr;
  ASSERT_EQ(flagcxP2pNetIb.iput(sendComm, 0, 0, bufSize, 0, 0,
                                reinterpret_cast<void **>(&srcInfo),
                                reinterpret_cast<void **>(&dstInfo), &request),
            flagcxSuccess);
  ASSERT_NE(request, nullptr);

  // Poll until done
  int done = 0;
  int sizes = 0;
  int polls = 0;
  while (!done && polls < 1000000) {
    ASSERT_EQ(flagcxP2pNetIb.test(request, &done, &sizes), flagcxSuccess);
    polls++;
  }
  EXPECT_TRUE(done) << "iput did not complete within poll limit";

  // Verify data was written
  EXPECT_EQ(memcmp(srcBuf, dstBuf, bufSize), 0)
      << "RDMA write did not transfer data correctly";

  // Cleanup
  flagcxP2pNetIb.deregMr(sendComm, srcMr);
  flagcxP2pNetIb.deregMr(sendComm, dstMr);
  free(srcBuf);
  free(dstBuf);
  flagcxP2pNetIb.closeSend(sendComm);
  flagcxP2pNetIb.closeRecv(recvComm);
  flagcxP2pNetIb.closeListen(listenComm);
}

// ---------------------------------------------------------------------------
// 7. Close with NULL is safe
// ---------------------------------------------------------------------------
TEST(P2pAdaptorStruct, CloseNullIsSafe) {
  EXPECT_EQ(flagcxP2pNetIb.closeSend(nullptr), flagcxSuccess);
  EXPECT_EQ(flagcxP2pNetIb.closeRecv(nullptr), flagcxSuccess);
  EXPECT_EQ(flagcxP2pNetIb.closeListen(nullptr), flagcxSuccess);
}

// ---------------------------------------------------------------------------
// 8. Test with NULL request returns done immediately
// ---------------------------------------------------------------------------
TEST(P2pAdaptorStruct, TestNullRequestIsDone) {
  int done = 0;
  int sizes = 0;
  EXPECT_EQ(flagcxP2pNetIb.test(nullptr, &done, &sizes), flagcxSuccess);
  EXPECT_EQ(done, 1);
}
