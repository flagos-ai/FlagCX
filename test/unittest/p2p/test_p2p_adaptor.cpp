// Unit tests for the IB P2P net adaptor.
// Tests that don't require IB hardware always run.
// Tests that need real IB devices skip gracefully via GTEST_SKIP().
// Links against libflagcx.

#include <cstring>
#include <future>
#include <gtest/gtest.h>
#include <infiniband/verbs.h>
#include <thread>

#include "flagcx_net.h"
#include "flagcx_net_adaptor.h"

// The P2P adaptor struct is non-static in ibrc_p2p_adaptor.cc
extern struct flagcxNetAdaptor flagcxNetIbP2p;

// ---------------------------------------------------------------------------
// Fixture: initializes the adaptor once, caches device count
// ---------------------------------------------------------------------------
class P2pAdaptorTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    initResult = flagcxNetIbP2p.init();
    if (initResult == flagcxSuccess) {
      flagcxNetIbP2p.devices(&nDevs);
    }
  }

  void skipIfNoIb() {
    if (initResult != flagcxSuccess || nDevs <= 0) {
      GTEST_SKIP() << "No IB devices available, skipping";
    }
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
  EXPECT_NE(flagcxNetIbP2p.name, nullptr);
  EXPECT_STREQ(flagcxNetIbP2p.name, "IB_P2P");

  // Basic
  EXPECT_NE(flagcxNetIbP2p.init, nullptr);
  EXPECT_NE(flagcxNetIbP2p.devices, nullptr);
  EXPECT_NE(flagcxNetIbP2p.getProperties, nullptr);

  // Connection setup
  EXPECT_NE(flagcxNetIbP2p.listen, nullptr);
  EXPECT_NE(flagcxNetIbP2p.connect, nullptr);
  EXPECT_NE(flagcxNetIbP2p.accept, nullptr);
  EXPECT_NE(flagcxNetIbP2p.closeSend, nullptr);
  EXPECT_NE(flagcxNetIbP2p.closeRecv, nullptr);
  EXPECT_NE(flagcxNetIbP2p.closeListen, nullptr);

  // Memory registration
  EXPECT_NE(flagcxNetIbP2p.regMr, nullptr);
  EXPECT_NE(flagcxNetIbP2p.regMrDmaBuf, nullptr);
  EXPECT_NE(flagcxNetIbP2p.deregMr, nullptr);

  // Two-sided (stubs)
  EXPECT_NE(flagcxNetIbP2p.isend, nullptr);
  EXPECT_NE(flagcxNetIbP2p.irecv, nullptr);
  EXPECT_NE(flagcxNetIbP2p.iflush, nullptr);
  EXPECT_NE(flagcxNetIbP2p.test, nullptr);

  // One-sided
  EXPECT_NE(flagcxNetIbP2p.iput, nullptr);
  EXPECT_NE(flagcxNetIbP2p.iget, nullptr);
  EXPECT_NE(flagcxNetIbP2p.iputSignal, nullptr);

  // Device lookup
  EXPECT_NE(flagcxNetIbP2p.getDevFromName, nullptr);
}

// Two-sided stubs should return errors
TEST(P2pAdaptorStruct, TwoSidedStubsReturnError) {
  void *dummy = nullptr;
  EXPECT_NE(
      flagcxNetIbP2p.isend(nullptr, nullptr, 0, 0, nullptr, nullptr, &dummy),
      flagcxSuccess);
  EXPECT_NE(flagcxNetIbP2p.irecv(nullptr, 0, nullptr, nullptr, nullptr, nullptr,
                                 nullptr, &dummy),
            flagcxSuccess);
  EXPECT_NE(
      flagcxNetIbP2p.iflush(nullptr, 0, nullptr, nullptr, nullptr, &dummy),
      flagcxSuccess);
}

// ---------------------------------------------------------------------------
// 2. Init + Devices — requires IB hardware
// ---------------------------------------------------------------------------
TEST_F(P2pAdaptorTest, InitSucceeds) {
  skipIfNoIb();
  EXPECT_EQ(initResult, flagcxSuccess);
}

TEST_F(P2pAdaptorTest, DevicesReturnsPositive) {
  skipIfNoIb();
  EXPECT_GT(nDevs, 0);
}

TEST_F(P2pAdaptorTest, InitIsIdempotent) {
  skipIfNoIb();
  // Calling init again should succeed without side effects
  EXPECT_EQ(flagcxNetIbP2p.init(), flagcxSuccess);
  int nDevs2 = 0;
  EXPECT_EQ(flagcxNetIbP2p.devices(&nDevs2), flagcxSuccess);
  EXPECT_EQ(nDevs2, nDevs);
}

// ---------------------------------------------------------------------------
// 3. GetProperties — requires IB hardware
// ---------------------------------------------------------------------------
TEST_F(P2pAdaptorTest, GetPropertiesForEachDevice) {
  skipIfNoIb();
  for (int d = 0; d < nDevs; d++) {
    flagcxNetProperties_t props;
    memset(&props, 0, sizeof(props));
    EXPECT_EQ(flagcxNetIbP2p.getProperties(d, &props), flagcxSuccess);
    EXPECT_GT(props.speed, 0);
    EXPECT_NE(props.name, nullptr);
  }
}

// ---------------------------------------------------------------------------
// 4. Listen + Connect + Accept loopback — requires IB hardware
// ---------------------------------------------------------------------------
class P2pLoopbackTest : public P2pAdaptorTest {
protected:
  void SetUp() override { skipIfNoIb(); }
};

TEST_F(P2pLoopbackTest, ListenConnectAcceptClose) {
  // Listen
  char handle[FLAGCX_NET_HANDLE_MAXSIZE];
  void *listenComm = nullptr;
  ASSERT_EQ(flagcxNetIbP2p.listen(0, handle, &listenComm), flagcxSuccess);
  ASSERT_NE(listenComm, nullptr);

  // Connect + Accept in parallel using std::async with timeout
  auto acceptFuture = std::async(std::launch::async, [&]() {
    void *comm = nullptr;
    flagcxResult_t r = flagcxNetIbP2p.accept(listenComm, &comm);
    return std::make_pair(r, comm);
  });

  auto connectFuture = std::async(std::launch::async, [&]() {
    void *comm = nullptr;
    flagcxResult_t r = flagcxNetIbP2p.connect(0, handle, &comm);
    return std::make_pair(r, comm);
  });

  // Wait with timeout to avoid hanging forever
  auto timeout = std::chrono::seconds(10);

  ASSERT_EQ(connectFuture.wait_for(timeout), std::future_status::ready)
      << "connect() timed out after 10s";
  auto [connectResult, sendComm] = connectFuture.get();

  ASSERT_EQ(acceptFuture.wait_for(timeout), std::future_status::ready)
      << "accept() timed out after 10s";
  auto [acceptResult, recvComm] = acceptFuture.get();

  ASSERT_EQ(connectResult, flagcxSuccess) << "connect() failed";
  ASSERT_EQ(acceptResult, flagcxSuccess) << "accept() failed";
  ASSERT_NE(sendComm, nullptr);
  ASSERT_NE(recvComm, nullptr);

  // Close
  EXPECT_EQ(flagcxNetIbP2p.closeSend(sendComm), flagcxSuccess);
  EXPECT_EQ(flagcxNetIbP2p.closeRecv(recvComm), flagcxSuccess);
  EXPECT_EQ(flagcxNetIbP2p.closeListen(listenComm), flagcxSuccess);
}

// ---------------------------------------------------------------------------
// 5. RegMr + DeregMr — requires IB hardware + a loopback connection
// ---------------------------------------------------------------------------
TEST_F(P2pLoopbackTest, RegMrDeregMr) {
  // Set up loopback connection
  char handle[FLAGCX_NET_HANDLE_MAXSIZE];
  void *listenComm = nullptr;
  ASSERT_EQ(flagcxNetIbP2p.listen(0, handle, &listenComm), flagcxSuccess);

  auto acceptFuture = std::async(std::launch::async, [&]() {
    void *comm = nullptr;
    flagcxNetIbP2p.accept(listenComm, &comm);
    return comm;
  });
  auto connectFuture = std::async(std::launch::async, [&]() {
    void *comm = nullptr;
    flagcxNetIbP2p.connect(0, handle, &comm);
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
  EXPECT_EQ(flagcxNetIbP2p.regMr(sendComm, buf, bufSize, FLAGCX_PTR_HOST,
                                 mrFlags, &mhandle),
            flagcxSuccess);
  EXPECT_NE(mhandle, nullptr);

  // Deregister
  EXPECT_EQ(flagcxNetIbP2p.deregMr(sendComm, mhandle), flagcxSuccess);

  // Register on recv side too (symmetric)
  void *mhandle2 = nullptr;
  EXPECT_EQ(flagcxNetIbP2p.regMr(recvComm, buf, bufSize, FLAGCX_PTR_HOST,
                                 mrFlags, &mhandle2),
            flagcxSuccess);
  EXPECT_NE(mhandle2, nullptr);
  EXPECT_EQ(flagcxNetIbP2p.deregMr(recvComm, mhandle2), flagcxSuccess);

  free(buf);
  flagcxNetIbP2p.closeSend(sendComm);
  flagcxNetIbP2p.closeRecv(recvComm);
  flagcxNetIbP2p.closeListen(listenComm);
}

// ---------------------------------------------------------------------------
// 6. Iput + Test — requires IB hardware + loopback
// ---------------------------------------------------------------------------
TEST_F(P2pLoopbackTest, IputAndTest) {
  // Set up loopback connection
  char handle[FLAGCX_NET_HANDLE_MAXSIZE];
  void *listenComm = nullptr;
  ASSERT_EQ(flagcxNetIbP2p.listen(0, handle, &listenComm), flagcxSuccess);

  auto acceptFuture = std::async(std::launch::async, [&]() {
    void *comm = nullptr;
    flagcxNetIbP2p.accept(listenComm, &comm);
    return comm;
  });
  auto connectFuture = std::async(std::launch::async, [&]() {
    void *comm = nullptr;
    flagcxNetIbP2p.connect(0, handle, &comm);
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
  ASSERT_EQ(flagcxNetIbP2p.regMr(sendComm, srcBuf, bufSize, FLAGCX_PTR_HOST,
                                 mrFlags, &srcMr),
            flagcxSuccess);
  ASSERT_EQ(flagcxNetIbP2p.regMr(sendComm, dstBuf, bufSize, FLAGCX_PTR_HOST,
                                 mrFlags, &dstMr),
            flagcxSuccess);

  // Iput: write srcBuf -> dstBuf via RDMA
  void *request = nullptr;
  ASSERT_EQ(flagcxNetIbP2p.iput(sendComm, 0, 0, bufSize, 0, 0, (void **)srcMr,
                                (void **)dstMr, &request),
            flagcxSuccess);
  ASSERT_NE(request, nullptr);

  // Poll until done
  int done = 0;
  int sizes = 0;
  int polls = 0;
  while (!done && polls < 1000000) {
    ASSERT_EQ(flagcxNetIbP2p.test(request, &done, &sizes), flagcxSuccess);
    polls++;
  }
  EXPECT_TRUE(done) << "iput did not complete within poll limit";

  // Verify data was written
  EXPECT_EQ(memcmp(srcBuf, dstBuf, bufSize), 0)
      << "RDMA write did not transfer data correctly";

  // Cleanup
  flagcxNetIbP2p.deregMr(sendComm, srcMr);
  flagcxNetIbP2p.deregMr(sendComm, dstMr);
  free(srcBuf);
  free(dstBuf);
  flagcxNetIbP2p.closeSend(sendComm);
  flagcxNetIbP2p.closeRecv(recvComm);
  flagcxNetIbP2p.closeListen(listenComm);
}

// ---------------------------------------------------------------------------
// 7. Close with NULL is safe
// ---------------------------------------------------------------------------
TEST(P2pAdaptorStruct, CloseNullIsSafe) {
  EXPECT_EQ(flagcxNetIbP2p.closeSend(nullptr), flagcxSuccess);
  EXPECT_EQ(flagcxNetIbP2p.closeRecv(nullptr), flagcxSuccess);
  EXPECT_EQ(flagcxNetIbP2p.closeListen(nullptr), flagcxSuccess);
}

// ---------------------------------------------------------------------------
// 8. Test with NULL request returns done immediately
// ---------------------------------------------------------------------------
TEST(P2pAdaptorStruct, TestNullRequestIsDone) {
  int done = 0;
  int sizes = 0;
  EXPECT_EQ(flagcxNetIbP2p.test(nullptr, &done, &sizes), flagcxSuccess);
  EXPECT_EQ(done, 1);
}
