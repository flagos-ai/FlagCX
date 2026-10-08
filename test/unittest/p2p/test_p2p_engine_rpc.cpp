// Unit tests for FlagCX P2P RPC Engine with Bootstrap P2P integration.
// Tests engine lifecycle, metadata exchange, connect/accept handshake,
// RPC server, and descriptor table exchange.
//
// Hardware-dependent tests skip gracefully via GTEST_SKIP().

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <future>
#include <memory>
#include <string>
#include <strings.h>
#include <thread>

#include <gtest/gtest.h>

#include "adaptor.h"
#include "bootstrap.h"
#include "flagcx_net_adaptor.h"
#include "flagcx_p2p.h"
#include "p2p_control.h"

extern struct flagcxNetAdaptor flagcxNetIb;

namespace {

// Helper to parse "ip:port?gpuIdx?notifPort" metadata
struct ParsedMetadata {
  std::string ip;
  int port = -1;
  int gpuIdx = -1;
  int notifPort = -1;
};

bool parseMetadata(const char *raw, ParsedMetadata *out) {
  if (raw == nullptr || out == nullptr)
    return false;
  std::string s(raw);
  size_t q1 = s.find('?');
  if (q1 == std::string::npos)
    return false;
  size_t q2 = s.find('?', q1 + 1);
  if (q2 == std::string::npos)
    return false;

  std::string endpoint = s.substr(0, q1);
  std::string gpuPart = s.substr(q1 + 1, q2 - q1 - 1);
  std::string notifPart = s.substr(q2 + 1);

  try {
    size_t colon = endpoint.rfind(':');
    if (colon == std::string::npos)
      return false;
    out->ip = endpoint.substr(0, colon);
    out->port = std::stoi(endpoint.substr(colon + 1));
    out->gpuIdx = std::stoi(gpuPart);
    out->notifPort = std::stoi(notifPart);
  } catch (...) {
    return false;
  }

  return !out->ip.empty() && out->port >= 0;
}

// ============================================================================
// Fixture: Sets up two engines (server + client)
// ============================================================================

class P2pEngineRpcTest : public ::testing::Test {
protected:
  void SetUp() override {
    serverEngine = flagcxP2pEngineCreate();
    clientEngine = flagcxP2pEngineCreate();
    if (serverEngine == nullptr || clientEngine == nullptr) {
      if (serverEngine) {
        flagcxP2pEngineDestroy(serverEngine);
        serverEngine = nullptr;
      }
      if (clientEngine) {
        flagcxP2pEngineDestroy(clientEngine);
        clientEngine = nullptr;
      }
      GTEST_SKIP() << "Unable to create P2P engines (no transport hardware)";
    }
  }

  void TearDown() override {
    if (serverConn) {
      flagcxP2pEngineConnDestroy(serverConn);
      serverConn = nullptr;
    }
    if (clientConn) {
      flagcxP2pEngineConnDestroy(clientConn);
      clientConn = nullptr;
    }
    if (serverEngine) {
      flagcxP2pEngineDestroy(serverEngine);
      serverEngine = nullptr;
    }
    if (clientEngine) {
      flagcxP2pEngineDestroy(clientEngine);
      clientEngine = nullptr;
    }
  }

  // Helper: connect client to server via bootstrap port
  ::testing::AssertionResult connectViaBsPort() {
    if (serverEngine == nullptr)
      return ::testing::AssertionFailure() << "serverEngine is null";
    if (clientEngine == nullptr)
      return ::testing::AssertionFailure() << "clientEngine is null";

    char *metaRaw = nullptr;
    if (flagcxP2pEngineGetMetadata(serverEngine, &metaRaw) != 0)
      return ::testing::AssertionFailure() << "metadata query failed";
    if (metaRaw == nullptr)
      return ::testing::AssertionFailure() << "metadata is null";
    std::unique_ptr<char[]> metadata(metaRaw);

    ParsedMetadata parsed;
    if (!parseMetadata(metadata.get(), &parsed))
      return ::testing::AssertionFailure() << "metadata=" << metadata.get();

    // Get server's RPC port (bootstrap listen port)
    const int serverRpcPort = flagcxP2pEngineGetRpcPort(serverEngine);
    if (serverRpcPort <= 0)
      return ::testing::AssertionFailure()
             << "invalid RPC port " << serverRpcPort;

    // Accept in background thread
    auto acceptFuture = std::async(std::launch::async, [this]() {
      char ipBuf[256] = {};
      int remoteGpuIdx = -1;
      FlagcxP2pConn *conn = flagcxP2pEngineAccept(serverEngine, ipBuf,
                                                  sizeof(ipBuf), &remoteGpuIdx);
      acceptedIp = ipBuf;
      acceptedRemoteGpuIdx = remoteGpuIdx;
      return conn;
    });

    // Connect from client to server's bootstrap port
    clientConn = flagcxP2pEngineConnect(clientEngine, parsed.ip.c_str(),
                                        parsed.gpuIdx, serverRpcPort, false);
    if (clientConn == nullptr) {
      flagcxP2pEngineStopAccept(serverEngine);
      if (acceptFuture.wait_for(std::chrono::seconds(1)) ==
          std::future_status::ready) {
        serverConn = acceptFuture.get();
      }
      return ::testing::AssertionFailure() << "client connection is null";
    }

    if (acceptFuture.wait_for(std::chrono::seconds(10)) !=
        std::future_status::ready) {
      flagcxP2pEngineStopAccept(serverEngine);
      if (acceptFuture.wait_for(std::chrono::seconds(1)) ==
          std::future_status::ready) {
        serverConn = acceptFuture.get();
      }
      return ::testing::AssertionFailure() << "Accept timed out";
    }
    serverConn = acceptFuture.get();
    if (serverConn == nullptr)
      return ::testing::AssertionFailure() << "server connection is null";
    return ::testing::AssertionSuccess();
  }

  FlagcxP2pEngine *serverEngine = nullptr;
  FlagcxP2pEngine *clientEngine = nullptr;
  FlagcxP2pConn *serverConn = nullptr;
  FlagcxP2pConn *clientConn = nullptr;
  std::string acceptedIp;
  int acceptedRemoteGpuIdx = -1;

  static bool hasEngineDevices() {
    struct flagcxNetAdaptor *adaptor = &flagcxNetIb;
#ifdef USE_ACCL_BAREX
    const char *transport = std::getenv("FLAGCX_P2P_TRANSPORT");
    if (transport != nullptr && (strcasecmp(transport, "accl") == 0 ||
                                 strcasecmp(transport, "barex") == 0))
      adaptor = getNetAdaptor(RDMA);
#endif
    int nDevs = 0;
    return adaptor != nullptr && adaptor->init() == flagcxSuccess &&
           adaptor->devices(&nDevs) == flagcxSuccess && nDevs > 0;
  }
};

class P2pEngineRpcTransportTest : public P2pEngineRpcTest {
protected:
  void SetUp() override {
    P2pEngineRpcTest::SetUp();
    if (!hasEngineDevices()) {
      GTEST_SKIP() << "No selected transport devices available, skipping P2P "
                      "RPC connection test";
    }
  }
};

// ============================================================================
// 1. Engine Lifecycle
// ============================================================================

TEST(P2pEngineLifecycle, CreateDestroyWithoutTransport) {
  // Does not require a transport device; this only checks null-safety.
  FlagcxP2pEngine *engine = flagcxP2pEngineCreate();
  // Creation may fail when no selected transport is available, but destruction
  // must remain safe.
  if (engine) {
    flagcxP2pEngineDestroy(engine);
  }
}

TEST(P2pEngineLifecycle, DoubleDestroyIsNoop) {
  flagcxP2pEngineDestroy(nullptr);
  flagcxP2pEngineDestroy(nullptr);
  // Should not crash
}

TEST_F(P2pEngineRpcTest, EngineCreateInitializesBootstrap) {
  ASSERT_NE(serverEngine, nullptr);
  const int port = flagcxP2pEngineGetRpcPort(serverEngine);
  EXPECT_GT(port, 0) << "Bootstrap listen port should be > 0";
}

// ============================================================================
// 2. Metadata / Port Discovery
// ============================================================================

TEST_F(P2pEngineRpcTest, GetRpcPortReturnsBootstrapPort) {
  const int port = flagcxP2pEngineGetRpcPort(serverEngine);
  EXPECT_GT(port, 0);
}

TEST_F(P2pEngineRpcTest, GetMetadataContainsIpAndPort) {
  char *metaRaw = nullptr;
  ASSERT_EQ(flagcxP2pEngineGetMetadata(serverEngine, &metaRaw), 0);
  ASSERT_NE(metaRaw, nullptr);
  std::unique_ptr<char[]> metadata(metaRaw);

  ParsedMetadata parsed;
  ASSERT_TRUE(parseMetadata(metadata.get(), &parsed))
      << "metadata=" << metadata.get();

  EXPECT_FALSE(parsed.ip.empty());
  EXPECT_GT(parsed.port, 0);
  EXPECT_GE(parsed.gpuIdx, -1);
  EXPECT_GT(parsed.notifPort, 0);
}

TEST_F(P2pEngineRpcTest, GetMetadataPortMatchesRpcPort) {
  char *metaRaw = nullptr;
  ASSERT_EQ(flagcxP2pEngineGetMetadata(serverEngine, &metaRaw), 0);
  std::unique_ptr<char[]> metadata(metaRaw);

  ParsedMetadata parsed;
  ASSERT_TRUE(parseMetadata(metadata.get(), &parsed));

  const int rpcPort = flagcxP2pEngineGetRpcPort(serverEngine);
  // After bootstrap P2P integration, metadata exposes the bootstrap listen
  // port — the same port used for RPC and initial connection handshake.
  EXPECT_EQ(parsed.port, rpcPort)
      << "metadata port should equal bootstrap RPC port";
}

// ============================================================================
// 3. Connect / Accept handshake
// ============================================================================

TEST_F(P2pEngineRpcTransportTest, ConnectAcceptBasic) {
  ASSERT_TRUE(connectViaBsPort());
  EXPECT_NE(clientConn, nullptr);
  EXPECT_NE(serverConn, nullptr);
}

TEST_F(P2pEngineRpcTransportTest, RejectsOtherEnginePrefaceAndAcceptsNextPeer) {
#if defined(USE_ACCL_BAREX) && !defined(USE_SHARED_P2P_ENGINE)
  const char *selectedTransport = std::getenv("FLAGCX_P2P_TRANSPORT");
  if (selectedTransport != nullptr &&
      (strcasecmp(selectedTransport, "accl") == 0 ||
       strcasecmp(selectedTransport, "barex") == 0))
    GTEST_SKIP() << "Legacy ACCL does not use the IBRC protocol preface";
#endif

  char *rawMetadata = nullptr;
  ASSERT_EQ(flagcxP2pEngineGetMetadata(serverEngine, &rawMetadata), 0);
  ASSERT_NE(rawMetadata, nullptr);
  std::unique_ptr<char[]> metadata(rawMetadata);
  ParsedMetadata parsed;
  ASSERT_TRUE(parseMetadata(metadata.get(), &parsed));

  struct flagcxBootstrapHandle handle = {};
  handle.magic = FLAGCX_SOCKET_MAGIC;
  const std::string endpoint =
      flagcxP2pControl::hostPort(parsed.ip, parsed.port);
  ASSERT_EQ(flagcxSocketGetAddrFromString(&handle.addr, endpoint.c_str()),
            flagcxSuccess);

  auto acceptFuture = std::async(std::launch::async, [this]() {
    char ipBuf[256] = {};
    int remoteGpuIdx = -1;
    return flagcxP2pEngineAccept(serverEngine, ipBuf, sizeof(ipBuf),
                                 &remoteGpuIdx);
  });

  struct bootstrapState *probe = nullptr;
  const flagcxResult_t connectResult =
      bootstrapP2pConnect(&handle, FLAGCX_SOCKET_MAGIC, nullptr, &probe);
  flagcxResult_t exchangeResult = flagcxInternalError;
  flagcxP2pControl::ProtocolHello reply = {};
  if (connectResult == flagcxSuccess) {
#ifdef USE_SHARED_P2P_ENGINE
    constexpr auto oppositeImplementation = flagcxP2pControl::kProtocolLegacy;
#else
    constexpr auto oppositeImplementation = flagcxP2pControl::kProtocolShared;
#endif
    auto transport = flagcxP2pControl::kProtocolIbrc;
#ifdef USE_ACCL_BAREX
    const char *selectedTransport = std::getenv("FLAGCX_P2P_TRANSPORT");
    if (selectedTransport != nullptr &&
        (strcasecmp(selectedTransport, "accl") == 0 ||
         strcasecmp(selectedTransport, "barex") == 0))
      transport = flagcxP2pControl::kProtocolBarex;
#endif
    const auto opposite =
        flagcxP2pControl::protocolHello(oppositeImplementation, transport);
    exchangeResult =
        bootstrapExchange(probe, 0, flagcxP2pControl::kProtocolTag, &opposite,
                          sizeof(opposite), &reply, sizeof(reply));
#ifdef USE_SHARED_P2P_ENGINE
    const auto expected = flagcxP2pControl::protocolHello(
        flagcxP2pControl::kProtocolShared, transport);
#else
    const auto expected = flagcxP2pControl::protocolHello(
        flagcxP2pControl::kProtocolLegacy, transport);
#endif
    EXPECT_TRUE(flagcxP2pControl::protocolCompatible(expected, reply));
    bootstrapClose(probe);
  } else {
    flagcxP2pEngineStopAccept(serverEngine);
  }

  auto acceptStatus = acceptFuture.wait_for(std::chrono::seconds(10));
  if (acceptStatus != std::future_status::ready) {
    flagcxP2pEngineStopAccept(serverEngine);
    acceptStatus = acceptFuture.wait_for(std::chrono::seconds(1));
  }
  ASSERT_EQ(acceptStatus, std::future_status::ready);
  EXPECT_EQ(acceptFuture.get(), nullptr);
  ASSERT_EQ(connectResult, flagcxSuccess);
  ASSERT_EQ(exchangeResult, flagcxSuccess);

  // A rejected peer must not consume the listener or publish a connection.
  ASSERT_TRUE(connectViaBsPort());
}

TEST_F(P2pEngineRpcTransportTest, ConnectAcceptExchangesGpuIdx) {
  ASSERT_TRUE(connectViaBsPort());
  // Check that remote GPU index was exchanged on accept side
  EXPECT_GE(acceptedRemoteGpuIdx, -1);
  // Both connections are non-null (verified in connectViaBsPort)
  EXPECT_NE(clientConn, nullptr);
  EXPECT_NE(serverConn, nullptr);
}

TEST_F(P2pEngineRpcTransportTest, ConnectAcceptIsLocalSameHost) {
  ASSERT_TRUE(connectViaBsPort());
  // Single-host test — both sides should detect local connection
  EXPECT_TRUE(flagcxP2pEngineConnIsLocal(serverConn));
  EXPECT_TRUE(flagcxP2pEngineConnIsLocal(clientConn));
}

TEST_F(P2pEngineRpcTransportTest, AllZeroBatchWriteIsNoop) {
  ASSERT_TRUE(connectViaBsPort());
  const uint64_t src[] = {0, UINT64_MAX};
  const uint64_t dst[] = {UINT64_MAX, 0};
  const uint64_t sizes[] = {0, 0};
  EXPECT_EQ(flagcxP2pRpcBatchWriteSync(clientConn, 2, src, dst, sizes), 0);
}

TEST(P2pEngineDescriptorTest, UpdateOnlyAcceptsOriginalSubrange) {
  FlagcxP2pRdmaDesc desc{};
  desc.addr = 0x1000;
  desc.size = 0x1000;

  EXPECT_NE(flagcxP2pEngineUpdateDesc(desc, 0x0fff, 1), 0);
  EXPECT_NE(flagcxP2pEngineUpdateDesc(desc, 0x1800, 0x801), 0);
  EXPECT_EQ(desc.addr, 0x1000u);
  EXPECT_EQ(desc.size, 0x1000u);

  EXPECT_EQ(flagcxP2pEngineUpdateDesc(desc, 0x1800, 0x800), 0);
  EXPECT_EQ(desc.addr, 0x1800u);
  EXPECT_EQ(desc.size, 0x800u);
}

TEST_F(P2pEngineRpcTransportTest, ConnectToInvalidHostReturnsNull) {
  // Use an invalid numeric IPv4 literal so address parsing fails fast.
  FlagcxP2pConn *conn =
      flagcxP2pEngineConnect(clientEngine, "256.256.256.256", -1, 12345, false);
  EXPECT_EQ(conn, nullptr);
}

TEST_F(P2pEngineRpcTransportTest, ConnectToInvalidPortReturnsNull) {
  // Connect to localhost:1 (privileged, nothing listening)
  FlagcxP2pConn *conn =
      flagcxP2pEngineConnect(clientEngine, "127.0.0.1", -1, 1, false);
  EXPECT_EQ(conn, nullptr);
}

TEST_F(P2pEngineRpcTest, AcceptAfterStopReturnsNull) {
  flagcxP2pEngineStopAccept(serverEngine);
  char ipBuf[256] = {};
  int remoteGpuIdx = -1;
  // After StopAccept, engine->bsListenState is NULL, should return NULL
  FlagcxP2pConn *conn =
      flagcxP2pEngineAccept(serverEngine, ipBuf, sizeof(ipBuf), &remoteGpuIdx);
  EXPECT_EQ(conn, nullptr);
}

// ============================================================================
// 4. RPC Server (thread-based accept loop)
// ============================================================================

TEST_F(P2pEngineRpcTest, StartRpcServerTwiceIsIdempotent) {
  ASSERT_EQ(flagcxP2pEngineStartRpcServer(serverEngine), 0);
  ASSERT_EQ(flagcxP2pEngineStartRpcServer(serverEngine), 0);
  // Second call should return 0 (already running)
}

TEST_F(P2pEngineRpcTransportTest, GetConnCreatesConnection) {
  ASSERT_EQ(flagcxP2pEngineStartRpcServer(serverEngine), 0);

  char *metaRaw = nullptr;
  ASSERT_EQ(flagcxP2pEngineGetMetadata(serverEngine, &metaRaw), 0);
  std::unique_ptr<char[]> metadata(metaRaw);

  ParsedMetadata parsed;
  ASSERT_TRUE(parseMetadata(metadata.get(), &parsed));

  const int serverRpcPort = flagcxP2pEngineGetRpcPort(serverEngine);
  ASSERT_GT(serverRpcPort, 0);

  char sessionKey[256];
  snprintf(sessionKey, sizeof(sessionKey), "%s:%d", parsed.ip.c_str(),
           serverRpcPort);

  // GetConn should create and cache the connection
  FlagcxP2pConn *conn = flagcxP2pEngineGetConn(clientEngine, sessionKey);
  ASSERT_NE(conn, nullptr);

  // Clean up accepted connection on server side
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

TEST_F(P2pEngineRpcTransportTest, GetConnReturnsCachedOnSecondCall) {
  ASSERT_EQ(flagcxP2pEngineStartRpcServer(serverEngine), 0);

  char *metaRaw = nullptr;
  ASSERT_EQ(flagcxP2pEngineGetMetadata(serverEngine, &metaRaw), 0);
  std::unique_ptr<char[]> metadata(metaRaw);

  ParsedMetadata parsed;
  ASSERT_TRUE(parseMetadata(metadata.get(), &parsed));

  const int serverRpcPort = flagcxP2pEngineGetRpcPort(serverEngine);
  char sessionKey[256];
  snprintf(sessionKey, sizeof(sessionKey), "%s:%d", parsed.ip.c_str(),
           serverRpcPort);

  FlagcxP2pConn *conn1 = flagcxP2pEngineGetConn(clientEngine, sessionKey);
  ASSERT_NE(conn1, nullptr);

  FlagcxP2pConn *conn2 = flagcxP2pEngineGetConn(clientEngine, sessionKey);
  EXPECT_EQ(conn1, conn2) << "Second GetConn should return cached connection";
}

TEST_F(P2pEngineRpcTest, GetConnInvalidSessionReturnsNull) {
  FlagcxP2pConn *conn = flagcxP2pEngineGetConn(clientEngine, "no_colon");
  EXPECT_EQ(conn, nullptr);
}

// ============================================================================
// 5. Descriptor Table Exchange
// ============================================================================

TEST_F(P2pEngineRpcTransportTest, DescTableExchangedOnConnect) {
  ASSERT_TRUE(connectViaBsPort());
  // After handshake with no registered memory, MakeDesc should fail
  // (no remote regions to map) — this indirectly confirms empty desc table
  FlagcxP2pRdmaDesc desc;
  int ret = flagcxP2pEngineMakeDesc(clientConn, 0x1000, 64, &desc);
  EXPECT_NE(ret, 0) << "MakeDesc should fail with no registered memory";
}

// ============================================================================
// 6. Connection Teardown
// ============================================================================

TEST(P2pEngineConnTeardown, ConnDestroyNullIsNoop) {
  flagcxP2pEngineConnDestroy(nullptr);
  // Should not crash
}

TEST_F(P2pEngineRpcTransportTest, ConnDestroyAfterHandshake) {
  ASSERT_TRUE(connectViaBsPort());
  flagcxP2pEngineConnDestroy(clientConn);
  clientConn = nullptr;
  flagcxP2pEngineConnDestroy(serverConn);
  serverConn = nullptr;
  // Should not crash or leak
}

TEST_F(P2pEngineRpcTransportTest,
       StopAcceptPreservesEstablishedControlConnection) {
  ASSERT_TRUE(connectViaBsPort());
  (void)flagcxP2pEngineGetNotifs();

  flagcxP2pEngineStopAccept(serverEngine);
  flagcxP2pEngineStopAccept(clientEngine);

  FlagcxP2pNotifyMsg msg = {};
  std::strncpy(msg.name, "stop-accept", sizeof(msg.name) - 1);
  std::strncpy(msg.msg, "established-control-remains-live",
               sizeof(msg.msg) - 1);
  ASSERT_EQ(flagcxP2pEngineSendNotif(clientConn, &msg),
            static_cast<int>(sizeof(msg)));

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline) {
    const std::vector<FlagcxP2pNotifyMsg> received = flagcxP2pEngineGetNotifs();
    for (const FlagcxP2pNotifyMsg &candidate : received) {
      if (std::strcmp(candidate.name, msg.name) == 0 &&
          std::strcmp(candidate.msg, msg.msg) == 0)
        return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  FAIL() << "notification was not delivered after StopAccept";
}

TEST_F(P2pEngineRpcTest, StopAcceptThenDestroy) {
  flagcxP2pEngineStopAccept(serverEngine);
  flagcxP2pEngineDestroy(serverEngine);
  serverEngine = nullptr;
  // Should not deadlock or crash
}

} // namespace
