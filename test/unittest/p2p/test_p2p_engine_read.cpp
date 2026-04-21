// Unit tests for the FlagCX P2P engine one-sided READ path.
// These mirror the UCCL test flow: remote metadata exchange, connect/accept,
// remote descriptor handoff, initiator-side read, and async completion polling.

#include <chrono>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "flagcx_p2p.h"

namespace {

struct ParsedEngineMetadata {
  std::string ip;
  int rdmaPort = -1;
  int remoteGpuIdx = -1;
  int notifPort = -1;
};

struct AcceptResult {
  FlagcxP2pConn *conn = nullptr;
  std::string remoteIp;
  int remoteGpuIdx = -1;
};

class ScopedMr {
public:
  ScopedMr() = default;

  ~ScopedMr() { reset(); }

  ScopedMr(const ScopedMr &) = delete;
  ScopedMr &operator=(const ScopedMr &) = delete;

  void set(FlagcxP2pEngine *engineArg, FlagcxP2pMr mrArg) {
    reset();
    engine = engineArg;
    mr = mrArg;
    active = true;
  }

  FlagcxP2pMr get() const { return mr; }

  void reset() {
    if (active && engine != nullptr) {
      flagcxP2pEngineMrDestroy(engine, mr);
    }
    engine = nullptr;
    mr = 0;
    active = false;
  }

private:
  FlagcxP2pEngine *engine = nullptr;
  FlagcxP2pMr mr = 0;
  bool active = false;
};

bool parseEngineMetadata(const char *metadata, ParsedEngineMetadata *out) {
  if (metadata == nullptr || out == nullptr) {
    return false;
  }

  const std::string text(metadata);
  const size_t firstSep = text.find('?');
  const size_t secondSep =
      firstSep == std::string::npos ? std::string::npos : text.find('?', firstSep + 1);
  if (firstSep == std::string::npos || secondSep == std::string::npos) {
    return false;
  }

  const std::string endpoint = text.substr(0, firstSep);
  const std::string gpuPart = text.substr(firstSep + 1, secondSep - firstSep - 1);
  const std::string notifPart = text.substr(secondSep + 1);

  try {
    if (!endpoint.empty() && endpoint.front() == '[') {
      const size_t closeBracket = endpoint.find(']');
      if (closeBracket == std::string::npos || closeBracket + 1 >= endpoint.size() ||
          endpoint[closeBracket + 1] != ':') {
        return false;
      }
      out->ip = endpoint.substr(1, closeBracket - 1);
      out->rdmaPort = std::stoi(endpoint.substr(closeBracket + 2));
    } else {
      const size_t colon = endpoint.rfind(':');
      if (colon == std::string::npos) {
        return false;
      }
      out->ip = endpoint.substr(0, colon);
      out->rdmaPort = std::stoi(endpoint.substr(colon + 1));
    }
    out->remoteGpuIdx = std::stoi(gpuPart);
    out->notifPort = std::stoi(notifPart);
  } catch (...) {
    return false;
  }

  return !out->ip.empty() && out->rdmaPort >= 0;
}

bool pollTransferDone(FlagcxP2pConn *conn, uint64_t transferId,
                      std::chrono::milliseconds timeout) {
  if (transferId == 0) {
    return true;
  }

  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (flagcxP2pEngineXferStatus(conn, transferId)) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return flagcxP2pEngineXferStatus(conn, transferId);
}

class FlagcxP2pEngineReadTest : public ::testing::Test {
protected:
  void SetUp() override {
    serverEngine = flagcxP2pEngineCreate(0, false);
    clientEngine = flagcxP2pEngineCreate(0, false);
    if (serverEngine == nullptr || clientEngine == nullptr) {
      if (serverEngine != nullptr) {
        flagcxP2pEngineDestroy(serverEngine);
        serverEngine = nullptr;
      }
      if (clientEngine != nullptr) {
        flagcxP2pEngineDestroy(clientEngine);
        clientEngine = nullptr;
      }
      GTEST_SKIP() << "Unable to create FlagCX P2P engines; likely no IB-capable device";
    }
  }

  void TearDown() override {
    if (serverConn != nullptr) {
      flagcxP2pEngineConnDestroy(serverConn);
      serverConn = nullptr;
    }
    if (clientConn != nullptr) {
      flagcxP2pEngineConnDestroy(clientConn);
      clientConn = nullptr;
    }
    if (serverEngine != nullptr) {
      flagcxP2pEngineDestroy(serverEngine);
      serverEngine = nullptr;
    }
    if (clientEngine != nullptr) {
      flagcxP2pEngineDestroy(clientEngine);
      clientEngine = nullptr;
    }
  }

  void connectViaClientMetadata() {
    ASSERT_NE(serverEngine, nullptr);
    ASSERT_NE(clientEngine, nullptr);

    char *metadataRaw = nullptr;
    ASSERT_EQ(flagcxP2pEngineGetMetadata(clientEngine, &metadataRaw), 0);
    ASSERT_NE(metadataRaw, nullptr);
    std::unique_ptr<char[]> metadata(metadataRaw);

    ParsedEngineMetadata parsed;
    ASSERT_TRUE(parseEngineMetadata(metadata.get(), &parsed))
        << "metadata=" << metadata.get();

    auto acceptFuture = std::async(std::launch::async, [this]() {
      char ipBuf[256] = {};
      int remoteGpuIdx = -1;
      AcceptResult result;
      result.conn =
          flagcxP2pEngineAccept(clientEngine, ipBuf, sizeof(ipBuf), &remoteGpuIdx);
      result.remoteIp = ipBuf;
      result.remoteGpuIdx = remoteGpuIdx;
      return result;
    });

    serverConn = flagcxP2pEngineConnect(serverEngine, parsed.ip.c_str(),
                                        parsed.remoteGpuIdx, parsed.rdmaPort,
                                        false);
    ASSERT_NE(serverConn, nullptr);

    ASSERT_EQ(acceptFuture.wait_for(std::chrono::seconds(10)),
              std::future_status::ready)
        << "flagcxP2pEngineAccept timed out";
    AcceptResult accepted = acceptFuture.get();
    clientConn = accepted.conn;
    ASSERT_NE(clientConn, nullptr);
    EXPECT_FALSE(accepted.remoteIp.empty());
    EXPECT_GE(accepted.remoteGpuIdx, 0);
  }

  FlagcxP2pEngine *serverEngine = nullptr;
  FlagcxP2pEngine *clientEngine = nullptr;
  FlagcxP2pConn *serverConn = nullptr;
  FlagcxP2pConn *clientConn = nullptr;
};

TEST_F(FlagcxP2pEngineReadTest, ReadsWholeRegisteredBufferAfterMetadataHandshake) {
  connectViaClientMetadata();

  constexpr size_t kElemCount = 1024;
  const size_t bytes = kElemCount * sizeof(uint32_t);

  std::vector<uint32_t> remoteSource(kElemCount);
  std::iota(remoteSource.begin(), remoteSource.end(), 1u);
  std::vector<uint32_t> localDestination(kElemCount, 0u);

  FlagcxP2pMr remoteMr = 0;
  FlagcxP2pMr localMr = 0;
  ScopedMr remoteMrGuard;
  ScopedMr localMrGuard;

  ASSERT_EQ(flagcxP2pEngineReg(clientEngine,
                               reinterpret_cast<uintptr_t>(remoteSource.data()),
                               bytes, remoteMr),
            0);
  remoteMrGuard.set(clientEngine, remoteMr);

  ASSERT_EQ(flagcxP2pEngineReg(serverEngine,
                               reinterpret_cast<uintptr_t>(localDestination.data()),
                               bytes, localMr),
            0);
  localMrGuard.set(serverEngine, localMr);

  char descBuf[FLAGCX_P2P_DESC_SIZE] = {};
  ASSERT_EQ(flagcxP2pEnginePrepareDesc(clientEngine, remoteMr, remoteSource.data(),
                                       bytes, descBuf),
            0);

  FlagcxP2pRdmaDesc remoteDesc;
  flagcxP2pDeserializeRdmaDesc(descBuf, &remoteDesc);

  uint64_t transferId = 0;
  ASSERT_EQ(flagcxP2pEngineRead(serverConn, localMr, localDestination.data(),
                                bytes, remoteDesc, &transferId),
            0);
  ASSERT_TRUE(pollTransferDone(serverConn, transferId, std::chrono::seconds(10)))
      << "Timed out waiting for flagcxP2pEngineRead completion";

  EXPECT_EQ(localDestination, remoteSource);
}

TEST_F(FlagcxP2pEngineReadTest, ReadsRetargetedRemoteSubrangeIntoLocalWindow) {
  connectViaClientMetadata();

  constexpr size_t kSourceElems = 256;
  constexpr size_t kDestElems = 128;
  constexpr size_t kSrcOffsetElems = 37;
  constexpr size_t kDstOffsetElems = 19;
  constexpr size_t kReadElems = 48;
  const size_t sourceBytes = kSourceElems * sizeof(uint32_t);
  const size_t destBytes = kDestElems * sizeof(uint32_t);
  const size_t readBytes = kReadElems * sizeof(uint32_t);

  std::vector<uint32_t> remoteSource(kSourceElems);
  for (size_t i = 0; i < remoteSource.size(); ++i) {
    remoteSource[i] = static_cast<uint32_t>(1000 + i);
  }

  std::vector<uint32_t> localDestination(kDestElems, 0xDEADBEEFu);

  FlagcxP2pMr remoteMr = 0;
  FlagcxP2pMr localMr = 0;
  ScopedMr remoteMrGuard;
  ScopedMr localMrGuard;

  ASSERT_EQ(flagcxP2pEngineReg(clientEngine,
                               reinterpret_cast<uintptr_t>(remoteSource.data()),
                               sourceBytes, remoteMr),
            0);
  remoteMrGuard.set(clientEngine, remoteMr);

  ASSERT_EQ(flagcxP2pEngineReg(serverEngine,
                               reinterpret_cast<uintptr_t>(localDestination.data()),
                               destBytes, localMr),
            0);
  localMrGuard.set(serverEngine, localMr);

  char descBuf[FLAGCX_P2P_DESC_SIZE] = {};
  ASSERT_EQ(flagcxP2pEnginePrepareDesc(clientEngine, remoteMr, remoteSource.data(),
                                       sourceBytes, descBuf),
            0);

  FlagcxP2pRdmaDesc remoteDesc;
  flagcxP2pDeserializeRdmaDesc(descBuf, &remoteDesc);
  ASSERT_EQ(flagcxP2pEngineUpdateDesc(
                remoteDesc,
                reinterpret_cast<uint64_t>(remoteSource.data() + kSrcOffsetElems),
                static_cast<uint32_t>(readBytes)),
            0);

  uint64_t transferId = 0;
  ASSERT_EQ(flagcxP2pEngineRead(serverConn, localMr,
                                localDestination.data() + kDstOffsetElems,
                                readBytes, remoteDesc, &transferId),
            0);
  ASSERT_TRUE(pollTransferDone(serverConn, transferId, std::chrono::seconds(10)))
      << "Timed out waiting for retargeted flagcxP2pEngineRead completion";

  for (size_t i = 0; i < kDstOffsetElems; ++i) {
    EXPECT_EQ(localDestination[i], 0xDEADBEEFu);
  }
  for (size_t i = 0; i < kReadElems; ++i) {
    EXPECT_EQ(localDestination[kDstOffsetElems + i],
              remoteSource[kSrcOffsetElems + i]);
  }
  for (size_t i = kDstOffsetElems + kReadElems; i < localDestination.size(); ++i) {
    EXPECT_EQ(localDestination[i], 0xDEADBEEFu);
  }
}

}  // namespace
