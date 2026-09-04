// Unit tests for FlagCX P2P RPC Engine with Bootstrap P2P integration.
// Tests engine lifecycle, metadata exchange, connect/accept handshake,
// RPC server, and descriptor table exchange.
//
// Hardware-dependent tests fail when their required devices are unavailable.

#include <chrono>
#include <cstring>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "flagcx_p2p.h"

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
      FAIL() << "Unable to create P2P engines (no IB hardware)";
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
  ::testing::AssertionResult connectViaBsPort(bool sameProcess = false) {
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
    clientConn =
        flagcxP2pEngineConnect(clientEngine, parsed.ip.c_str(), parsed.gpuIdx,
                               serverRpcPort, sameProcess);
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
};

class P2pEngineRpcIbTest : public P2pEngineRpcTest {
protected:
  void SetUp() override { P2pEngineRpcTest::SetUp(); }
};

class ScopedRpcEngine {
public:
  ScopedRpcEngine() : engine(flagcxP2pRpcEngineCreate()) {}
  ~ScopedRpcEngine() { flagcxP2pRpcEngineDestroy(engine); }

  ScopedRpcEngine(const ScopedRpcEngine &) = delete;
  ScopedRpcEngine &operator=(const ScopedRpcEngine &) = delete;

  void *get() const { return engine; }

private:
  void *engine = nullptr;
};

// ============================================================================
// 1. Engine Lifecycle
// ============================================================================

TEST(P2pEngineLifecycle, CreateDestroyNoIb) {
  // Doesn't require IB — just checks null-safety
  FlagcxP2pEngine *engine = flagcxP2pEngineCreate();
  // May be null if no IB, but should not crash
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
  EXPECT_GE(parsed.notifPort, 0);
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

TEST_F(P2pEngineRpcIbTest, ConnectAcceptBasic) {
  ASSERT_TRUE(connectViaBsPort());
  EXPECT_NE(clientConn, nullptr);
  EXPECT_NE(serverConn, nullptr);
}

TEST_F(P2pEngineRpcIbTest, ConnectAcceptExchangesGpuIdx) {
  ASSERT_TRUE(connectViaBsPort());
  // Check that remote GPU index was exchanged on accept side
  EXPECT_GE(acceptedRemoteGpuIdx, -1);
  // Both connections are non-null (verified in connectViaBsPort)
  EXPECT_NE(clientConn, nullptr);
  EXPECT_NE(serverConn, nullptr);
}

TEST_F(P2pEngineRpcIbTest, ConnectAcceptIsLocalSameHost) {
  ASSERT_TRUE(connectViaBsPort());
  // Locality is established by the common bootstrap handshake and is
  // independent of the selected network adaptor.
  EXPECT_TRUE(flagcxP2pEngineConnIsLocal(serverConn));
  EXPECT_TRUE(flagcxP2pEngineConnIsLocal(clientConn));
}

TEST_F(P2pEngineRpcIbTest, ConnectToInvalidHostReturnsNull) {
  // Use an invalid numeric IPv4 literal so address parsing fails fast.
  FlagcxP2pConn *conn =
      flagcxP2pEngineConnect(clientEngine, "256.256.256.256", -1, 12345, false);
  EXPECT_EQ(conn, nullptr);
}

TEST_F(P2pEngineRpcIbTest, ConnectToInvalidPortReturnsNull) {
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

TEST_F(P2pEngineRpcIbTest, GetConnCreatesConnection) {
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

TEST_F(P2pEngineRpcIbTest, GetConnReturnsCachedOnSecondCall) {
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

TEST_F(P2pEngineRpcIbTest, DescTableExchangedOnConnect) {
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

TEST_F(P2pEngineRpcIbTest, ConnDestroyAfterHandshake) {
  ASSERT_TRUE(connectViaBsPort());
  flagcxP2pEngineConnDestroy(clientConn);
  clientConn = nullptr;
  flagcxP2pEngineConnDestroy(serverConn);
  serverConn = nullptr;
  // Should not crash or leak
}

TEST_F(P2pEngineRpcTest, StopAcceptThenDestroy) {
  flagcxP2pEngineStopAccept(serverEngine);
  flagcxP2pEngineDestroy(serverEngine);
  serverEngine = nullptr;
  // Should not deadlock or crash
}

TEST(P2pRdmaDesc, SerializationRoundTripPreservesEveryField) {
  FlagcxP2pRdmaDesc expected = {};
  expected.addr = 0x0123456789abcdefULL;
  expected.size = 0x10203040u;
  expected.rkey = 0x50607080u;
  expected.nmsgs = 0x11223344u;
  expected.rid = 0x55667788u;
  expected.idx = 0xfedcba9876543210ULL;
  for (size_t i = 0; i < sizeof(expected.padding); ++i)
    expected.padding[i] = static_cast<char>(i + 1);

  char serialized[FLAGCX_P2P_DESC_SIZE] = {};
  flagcxP2pSerializeRdmaDesc(expected, serialized);
  FlagcxP2pRdmaDesc actual = {};
  flagcxP2pDeserializeRdmaDesc(serialized, &actual);
  EXPECT_EQ(std::memcmp(&actual, &expected, sizeof(expected)), 0);
}

TEST(P2pGlobalConfig, ReturnsStableValidConfiguration) {
  const FlagcxP2pGlobalConfig &first = flagcxP2pGlobalConfig();
  const FlagcxP2pGlobalConfig &second = flagcxP2pGlobalConfig();
  EXPECT_EQ(&first, &second);
  EXPECT_GT(first.qpsPerConn, 0);
  EXPECT_GT(first.workersPerPool, 0);
  EXPECT_GT(first.sharedCqDepth, 0u);
  EXPECT_GT(first.maxWrPerPost, 0u);
  EXPECT_GT(first.maxRequests, 0u);
  flagcxP2pDumpGlobalConfig();
}

TEST(P2pEngineUnsupportedTwoSided, ReturnsErrorWithoutSubmittingTransfer) {
  uint64_t transferId = 123;
  EXPECT_EQ(flagcxP2pEngineSend(nullptr, 0, nullptr, 0, &transferId), -1);
  EXPECT_EQ(flagcxP2pEngineSendVector(nullptr, {}, {}, {}, 0, &transferId), -1);
  EXPECT_EQ(flagcxP2pEngineRecv(nullptr, 0, nullptr, 0), -1);
  EXPECT_EQ(transferId, 123u);
}

TEST(P2pEngineTransferStatus, InvalidConnectionCannotRemainPending) {
  EXPECT_TRUE(flagcxP2pEngineXferStatus(nullptr, 1));
}

TEST_F(P2pEngineRpcIbTest, StartListenerIsIdempotentCompatibilityCall) {
  ASSERT_TRUE(connectViaBsPort());
  EXPECT_EQ(flagcxP2pEngineStartListener(clientConn), 0);
  EXPECT_EQ(flagcxP2pEngineStartListener(clientConn), 0);
}

TEST_F(P2pEngineRpcIbTest, SameProcessNotificationIsDeliveredAndConsumed) {
  ASSERT_TRUE(connectViaBsPort(true));
  EXPECT_TRUE(flagcxP2pEngineConnIsLocal(clientConn));
  flagcxP2pEngineGetNotifs();

  FlagcxP2pNotifyMsg sent = {};
  std::strncpy(sent.name, "p2p-engine-test", sizeof(sent.name) - 1);
  std::strncpy(sent.msg, "notification-payload", sizeof(sent.msg) - 1);
  EXPECT_EQ(flagcxP2pEngineSendNotif(clientConn, &sent),
            static_cast<int>(sizeof(sent)));

  std::vector<FlagcxP2pNotifyMsg> received = flagcxP2pEngineGetNotifs();
  ASSERT_EQ(received.size(), 1u);
  EXPECT_STREQ(received[0].name, sent.name);
  EXPECT_STREQ(received[0].msg, sent.msg);
  EXPECT_TRUE(flagcxP2pEngineGetNotifs().empty());
}

TEST_F(P2pEngineRpcTest, HostRegistrationHasNoGpuIpcHandle) {
  std::vector<char> hostBuffer(4096);
  FlagcxP2pMr mr = 0;
  ASSERT_EQ(flagcxP2pRpcRegisterHost(
                serverEngine, reinterpret_cast<uint64_t>(hostBuffer.data()),
                hostBuffer.size(), &mr),
            0);

  char ipcBuf[FLAGCX_P2P_IPC_INFO_SIZE] = {};
  bool hasIpc = true;
  EXPECT_EQ(flagcxP2pEngineGetIpcInfo(
                serverEngine, reinterpret_cast<uintptr_t>(hostBuffer.data()),
                ipcBuf, &hasIpc),
            0);
  EXPECT_FALSE(hasIpc);
  EXPECT_EQ(flagcxP2pEngineGetIpcInfo(serverEngine, 0, ipcBuf, &hasIpc), -1);
  EXPECT_EQ(flagcxP2pEngineUpdateIpcInfo(nullptr, 0x2000, 0x1000, 16), -1);
  EXPECT_EQ(flagcxP2pEngineUpdateIpcInfo(ipcBuf, 0x0fff, 0x1000, 16), -1);
  EXPECT_EQ(flagcxP2pEngineUpdateIpcInfo(ipcBuf, 0x1010, 0x1000, 16), 0);

  flagcxP2pEngineMrDestroy(serverEngine, mr);
}

TEST(P2pRpcCAbi, BatchedHostWriteCoversFacadeEndToEnd) {
  ScopedRpcEngine server;
  ScopedRpcEngine client;
  ASSERT_NE(server.get(), nullptr);
  ASSERT_NE(client.get(), nullptr);

  constexpr size_t kBufferSize = 256;
  std::vector<uint8_t> source(kBufferSize);
  std::vector<uint8_t> destination(kBufferSize, 0xA5);
  std::vector<uint8_t> expected(destination);
  for (size_t i = 0; i < source.size(); ++i)
    source[i] = static_cast<uint8_t>(i ^ 0x5A);

  uint64_t sourceMr = 0;
  uint64_t destinationMr = 0;
  uint64_t autoDetectedMr = 0;
  ASSERT_EQ(flagcxP2pRpcRegisterHost(client.get(),
                                     reinterpret_cast<uint64_t>(source.data()),
                                     source.size(), &sourceMr),
            0);
  ASSERT_EQ(flagcxP2pRpcRegisterHost(
                server.get(), reinterpret_cast<uint64_t>(destination.data()),
                destination.size(), &destinationMr),
            0);
  std::vector<uint8_t> autoDetectedHostBuffer(64);
  ASSERT_EQ(flagcxP2pRpcRegister(
                client.get(),
                reinterpret_cast<uint64_t>(autoDetectedHostBuffer.data()),
                autoDetectedHostBuffer.size(), &autoDetectedMr),
            0);
  EXPECT_NE(sourceMr, 0u);
  EXPECT_NE(destinationMr, 0u);
  EXPECT_NE(autoDetectedMr, 0u);

  ASSERT_EQ(flagcxP2pRpcStartServer(server.get()), 0);
  const int port = flagcxP2pRpcGetPort(server.get());
  ASSERT_GT(port, 0);
  char *metadataRaw = nullptr;
  ASSERT_EQ(
      flagcxP2pEngineGetMetadata(
          reinterpret_cast<FlagcxP2pEngine *>(server.get()), &metadataRaw),
      0);
  std::unique_ptr<char[]> metadata(metadataRaw);
  ParsedMetadata parsed;
  ASSERT_TRUE(parseMetadata(metadata.get(), &parsed));
  const std::string session = parsed.ip + ":" + std::to_string(port);
  void *conn = flagcxP2pRpcGetConn(client.get(), session.c_str());
  ASSERT_NE(conn, nullptr);

  constexpr int kCount = 3;
  const size_t srcOffsets[kCount] = {3, 71, 151};
  const size_t dstOffsets[kCount] = {17, 93, 189};
  const size_t lengths[kCount] = {11, 23, 31};
  uint64_t srcVa[kCount];
  uint64_t dstVa[kCount];
  uint64_t sizes[kCount];
  for (int i = 0; i < kCount; ++i) {
    srcVa[i] = reinterpret_cast<uint64_t>(source.data() + srcOffsets[i]);
    dstVa[i] = reinterpret_cast<uint64_t>(destination.data() + dstOffsets[i]);
    sizes[i] = lengths[i];
    std::memcpy(expected.data() + dstOffsets[i], source.data() + srcOffsets[i],
                lengths[i]);
  }
  ASSERT_EQ(flagcxP2pRpcBatchWriteSync(conn, kCount, srcVa, dstVa, sizes), 0);
  EXPECT_EQ(destination, expected);

  EXPECT_EQ(flagcxP2pRpcBatchWriteSync(conn, 0, nullptr, nullptr, nullptr), 0);
  EXPECT_EQ(flagcxP2pRpcBatchWriteSync(conn, -1, nullptr, nullptr, nullptr),
            -1);
  EXPECT_EQ(flagcxP2pRpcRegister(client.get(), 0, 0, nullptr), -1);
}

} // namespace
