// Unit tests for the FlagCX P2P engine one-sided READ path.
// These mirror the UCCL test flow: remote metadata exchange, connect/accept,
// remote descriptor handoff, initiator-side read, and async completion polling.
//
// The source and destination buffers are GPU-side allocations. The tests use
// pinned host staging buffers only to initialize device data and verify the
// final contents after the read completes.

#include <chrono>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <sched.h>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "adaptor.h"
#include "flagcx.h"
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

class ScopedAllocation {
public:
  ScopedAllocation() = default;

  ~ScopedAllocation() { reset(); }

  ScopedAllocation(const ScopedAllocation &) = delete;
  ScopedAllocation &operator=(const ScopedAllocation &) = delete;

  flagcxResult_t allocDevice(flagcxDeviceHandle_t devHandleArg, size_t sizeArg,
                             flagcxMemType_t memTypeArg,
                             flagcxStream_t streamArg) {
    reset();
    devHandle = devHandleArg;
    memType = memTypeArg;
    stream = streamArg;
    allocKind = AllocKind::DeviceMalloc;
    return devHandle->deviceMalloc(&ptr, sizeArg, memTypeArg, streamArg);
  }

  flagcxResult_t allocFlagcxMem(size_t sizeArg) {
    flagcxDeviceHandle_t savedDevHandle = devHandle;
    const int savedDeviceIdx = deviceIdx;
    flagcxStream_t savedStream = stream;
    const flagcxMemType_t savedMemType = memType;
    reset();
    devHandle = savedDevHandle;
    deviceIdx = savedDeviceIdx;
    stream = savedStream;
    memType = savedMemType;
    allocKind = AllocKind::FlagcxMemAlloc;
    if (devHandle != nullptr && deviceIdx >= 0) {
      const flagcxResult_t setRes = devHandle->setDevice(deviceIdx);
      if (setRes != flagcxSuccess) {
        allocKind = AllocKind::None;
        return setRes;
      }
    }
    return flagcxMemAlloc(&ptr, sizeArg);
  }

  void configure(flagcxDeviceHandle_t devHandleArg, int deviceIdxArg,
                 flagcxStream_t streamArg, flagcxMemType_t memTypeArg) {
    devHandle = devHandleArg;
    deviceIdx = deviceIdxArg;
    stream = streamArg;
    memType = memTypeArg;
  }

  void *get() const { return ptr; }

  template <typename T> T *as() const { return static_cast<T *>(ptr); }

  void reset() {
    if (ptr == nullptr) {
      allocKind = AllocKind::None;
      devHandle = nullptr;
      deviceIdx = -1;
      stream = nullptr;
      memType = flagcxMemDevice;
      return;
    }

    if (devHandle != nullptr && deviceIdx >= 0) {
      devHandle->setDevice(deviceIdx);
    }

    if (allocKind == AllocKind::FlagcxMemAlloc) {
      flagcxMemFree(ptr);
    } else if (allocKind == AllocKind::DeviceMalloc && devHandle != nullptr) {
      devHandle->deviceFree(ptr, memType, stream);
    }

    ptr = nullptr;
    allocKind = AllocKind::None;
    devHandle = nullptr;
    deviceIdx = -1;
    stream = nullptr;
    memType = flagcxMemDevice;
  }

private:
  enum class AllocKind {
    None,
    DeviceMalloc,
    FlagcxMemAlloc,
  };

  void *ptr = nullptr;
  AllocKind allocKind = AllocKind::None;
  flagcxDeviceHandle_t devHandle = nullptr;
  int deviceIdx = -1;
  flagcxStream_t stream = nullptr;
  flagcxMemType_t memType = flagcxMemDevice;
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
      firstSep == std::string::npos ? std::string::npos
                                    : text.find('?', firstSep + 1);
  if (firstSep == std::string::npos || secondSep == std::string::npos) {
    return false;
  }

  const std::string endpoint = text.substr(0, firstSep);
  const std::string gpuPart =
      text.substr(firstSep + 1, secondSep - firstSep - 1);
  const std::string notifPart = text.substr(secondSep + 1);

  try {
    if (!endpoint.empty() && endpoint.front() == '[') {
      const size_t closeBracket = endpoint.find(']');
      if (closeBracket == std::string::npos ||
          closeBracket + 1 >= endpoint.size() ||
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
  static constexpr int kClientGpuIdx = 0;
  static constexpr int kServerGpuIdx = 1;

  void SetUp() override {
    ASSERT_EQ(flagcxHandleInit(&handler), flagcxSuccess);
    devHandle = handler->devHandle;
    ASSERT_NE(devHandle, nullptr);

    int numDevices = 0;
    ASSERT_EQ(devHandle->getDeviceCount(&numDevices), flagcxSuccess);
    if (numDevices <= kServerGpuIdx) {
      flagcxHandleFree(handler);
      handler = nullptr;
      devHandle = nullptr;
      GTEST_SKIP() << "At least 2 GPU devices are required";
    }

    ASSERT_EQ(devHandle->setDevice(kServerGpuIdx), flagcxSuccess);
    ASSERT_EQ(devHandle->streamCreate(&serverStream), flagcxSuccess);
    serverEngine = flagcxP2pEngineCreate(0, false);
    ASSERT_EQ(devHandle->setDevice(kClientGpuIdx), flagcxSuccess);
    ASSERT_EQ(devHandle->streamCreate(&clientStream), flagcxSuccess);
    clientEngine = flagcxP2pEngineCreate(0, false);
    if (serverEngine == nullptr || clientEngine == nullptr ||
        serverStream == nullptr || clientStream == nullptr) {
      if (serverEngine != nullptr) {
        flagcxP2pEngineDestroy(serverEngine);
        serverEngine = nullptr;
      }
      if (clientEngine != nullptr) {
        flagcxP2pEngineDestroy(clientEngine);
        clientEngine = nullptr;
      }
      if (serverStream != nullptr) {
        devHandle->setDevice(kServerGpuIdx);
        devHandle->streamDestroy(serverStream);
        serverStream = nullptr;
      }
      if (clientStream != nullptr) {
        devHandle->setDevice(kClientGpuIdx);
        devHandle->streamDestroy(clientStream);
        clientStream = nullptr;
      }
      flagcxHandleFree(handler);
      handler = nullptr;
      devHandle = nullptr;
      GTEST_SKIP()
          << "Unable to create FlagCX P2P engines; likely no IB-capable device";
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
    if (serverStream != nullptr && devHandle != nullptr) {
      devHandle->setDevice(kServerGpuIdx);
      devHandle->streamDestroy(serverStream);
      serverStream = nullptr;
    }
    if (clientStream != nullptr && devHandle != nullptr) {
      devHandle->setDevice(kClientGpuIdx);
      devHandle->streamDestroy(clientStream);
      clientStream = nullptr;
    }
    if (handler != nullptr) {
      flagcxHandleFree(handler);
      handler = nullptr;
      devHandle = nullptr;
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
          flagcxP2pEngineAccept(clientEngine, ipBuf, sizeof(ipBuf),
                                &remoteGpuIdx);
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

  void allocGpuBuffer(ScopedAllocation *buffer, size_t bytes) {
    allocGpuBufferOnDevice(buffer, bytes, kClientGpuIdx, clientStream);
  }

  void allocGpuBufferOnDevice(ScopedAllocation *buffer, size_t bytes,
                              int deviceIdx, flagcxStream_t stream) {
    ASSERT_NE(buffer, nullptr);
    ASSERT_NE(devHandle, nullptr);
    buffer->configure(devHandle, deviceIdx, stream, flagcxMemDevice);
    ASSERT_EQ(buffer->allocFlagcxMem(bytes), flagcxSuccess);
    ASSERT_NE(buffer->get(), nullptr);
  }

  void allocHostBuffer(ScopedAllocation *buffer, size_t bytes, int deviceIdx,
                       flagcxStream_t stream) {
    ASSERT_NE(buffer, nullptr);
    ASSERT_NE(devHandle, nullptr);
    ASSERT_EQ(devHandle->setDevice(deviceIdx), flagcxSuccess);
    buffer->configure(devHandle, deviceIdx, stream, flagcxMemHost);
    ASSERT_EQ(buffer->allocDevice(devHandle, bytes, flagcxMemHost, stream),
              flagcxSuccess);
    ASSERT_NE(buffer->get(), nullptr);
  }

  void copyHostToDevice(int deviceIdx, flagcxStream_t stream, void *devicePtr,
                        void *hostPtr, size_t bytes) {
    ASSERT_NE(devHandle, nullptr);
    ASSERT_EQ(devHandle->setDevice(deviceIdx), flagcxSuccess);
    ASSERT_EQ(devHandle->deviceMemcpy(devicePtr, hostPtr, bytes,
                                      flagcxMemcpyHostToDevice, stream),
              flagcxSuccess);
    ASSERT_EQ(devHandle->streamSynchronize(stream), flagcxSuccess);
  }

  void copyDeviceToHost(int deviceIdx, flagcxStream_t stream, void *hostPtr,
                        void *devicePtr, size_t bytes) {
    ASSERT_NE(devHandle, nullptr);
    ASSERT_EQ(devHandle->setDevice(deviceIdx), flagcxSuccess);
    ASSERT_EQ(devHandle->deviceMemcpy(hostPtr, devicePtr, bytes,
                                      flagcxMemcpyDeviceToHost, stream),
              flagcxSuccess);
    ASSERT_EQ(devHandle->streamSynchronize(stream), flagcxSuccess);
  }

  flagcxHandlerGroup_t handler = nullptr;
  flagcxDeviceHandle_t devHandle = nullptr;
  flagcxStream_t clientStream = nullptr;
  flagcxStream_t serverStream = nullptr;
  FlagcxP2pEngine *serverEngine = nullptr;
  FlagcxP2pEngine *clientEngine = nullptr;
  FlagcxP2pConn *serverConn = nullptr;
  FlagcxP2pConn *clientConn = nullptr;
};

TEST_F(FlagcxP2pEngineReadTest,
       ReadsWholeRegisteredGpuBufferAfterMetadataHandshake) {
  connectViaClientMetadata();

  constexpr size_t kElemCount = 1024;
  const size_t bytes = kElemCount * sizeof(uint32_t);

  ScopedAllocation remoteSource;
  ScopedAllocation localDestination;
  ScopedAllocation hostExpected;
  ScopedAllocation hostActual;

  allocGpuBufferOnDevice(&remoteSource, bytes, kClientGpuIdx, clientStream);
  allocGpuBufferOnDevice(&localDestination, bytes, kServerGpuIdx, serverStream);
  allocHostBuffer(&hostExpected, bytes, kClientGpuIdx, clientStream);
  allocHostBuffer(&hostActual, bytes, kServerGpuIdx, serverStream);

  uint32_t *expected = hostExpected.as<uint32_t>();
  uint32_t *actual = hostActual.as<uint32_t>();
  for (size_t i = 0; i < kElemCount; ++i) {
    expected[i] = static_cast<uint32_t>(i + 1);
    actual[i] = 0;
  }

  copyHostToDevice(kClientGpuIdx, clientStream, remoteSource.get(),
                   hostExpected.get(), bytes);
  copyHostToDevice(kServerGpuIdx, serverStream, localDestination.get(),
                   hostActual.get(), bytes);

  FlagcxP2pMr remoteMr = 0;
  FlagcxP2pMr localMr = 0;
  ScopedMr remoteMrGuard;
  ScopedMr localMrGuard;

  ASSERT_EQ(flagcxP2pEngineReg(clientEngine,
                               reinterpret_cast<uintptr_t>(remoteSource.get()),
                               bytes, remoteMr),
            0);
  remoteMrGuard.set(clientEngine, remoteMr);

  ASSERT_EQ(flagcxP2pEngineReg(
                serverEngine,
                reinterpret_cast<uintptr_t>(localDestination.get()), bytes,
                localMr),
            0);
  localMrGuard.set(serverEngine, localMr);

  char descBuf[FLAGCX_P2P_DESC_SIZE] = {};
  ASSERT_EQ(flagcxP2pEnginePrepareDesc(clientEngine, remoteMr, remoteSource.get(),
                                       bytes, descBuf),
            0);

  FlagcxP2pRdmaDesc remoteDesc;
  flagcxP2pDeserializeRdmaDesc(descBuf, &remoteDesc);

  uint64_t transferId = 0;
  ASSERT_EQ(flagcxP2pEngineRead(serverConn, localMr, localDestination.get(),
                                bytes, remoteDesc, &transferId),
            0);
  ASSERT_TRUE(pollTransferDone(serverConn, transferId, std::chrono::seconds(10)))
      << "Timed out waiting for flagcxP2pEngineRead completion";

  copyDeviceToHost(kServerGpuIdx, serverStream, hostActual.get(),
                   localDestination.get(), bytes);
  for (size_t i = 0; i < kElemCount; ++i) {
    EXPECT_EQ(actual[i], expected[i]) << "Mismatch at index " << i;
  }
}

TEST_F(FlagcxP2pEngineReadTest, ReadsRetargetedRemoteGpuSubrangeIntoLocalWindow) {
  connectViaClientMetadata();

  constexpr size_t kSourceElems = 256;
  constexpr size_t kDestElems = 128;
  constexpr size_t kSrcOffsetElems = 37;
  constexpr size_t kDstOffsetElems = 19;
  constexpr size_t kReadElems = 48;
  const size_t sourceBytes = kSourceElems * sizeof(uint32_t);
  const size_t destBytes = kDestElems * sizeof(uint32_t);
  const size_t readBytes = kReadElems * sizeof(uint32_t);

  ScopedAllocation remoteSource;
  ScopedAllocation localDestination;
  ScopedAllocation hostExpectedSource;
  ScopedAllocation hostExpectedDestination;
  ScopedAllocation hostActualDestination;

  allocGpuBufferOnDevice(&remoteSource, sourceBytes, kClientGpuIdx,
                         clientStream);
  allocGpuBufferOnDevice(&localDestination, destBytes, kServerGpuIdx,
                         serverStream);
  allocHostBuffer(&hostExpectedSource, sourceBytes, kClientGpuIdx,
                  clientStream);
  allocHostBuffer(&hostExpectedDestination, destBytes, kServerGpuIdx,
                  serverStream);
  allocHostBuffer(&hostActualDestination, destBytes, kServerGpuIdx,
                  serverStream);

  uint32_t *expectedSource = hostExpectedSource.as<uint32_t>();
  uint32_t *expectedDestination = hostExpectedDestination.as<uint32_t>();
  uint32_t *actualDestination = hostActualDestination.as<uint32_t>();
  for (size_t i = 0; i < kSourceElems; ++i) {
    expectedSource[i] = static_cast<uint32_t>(1000 + i);
  }
  for (size_t i = 0; i < kDestElems; ++i) {
    expectedDestination[i] = 0xDEADBEEFu;
    actualDestination[i] = 0;
  }

  copyHostToDevice(kClientGpuIdx, clientStream, remoteSource.get(),
                   hostExpectedSource.get(), sourceBytes);
  copyHostToDevice(kServerGpuIdx, serverStream, localDestination.get(),
                   hostExpectedDestination.get(), destBytes);

  FlagcxP2pMr remoteMr = 0;
  FlagcxP2pMr localMr = 0;
  ScopedMr remoteMrGuard;
  ScopedMr localMrGuard;

  ASSERT_EQ(flagcxP2pEngineReg(clientEngine,
                               reinterpret_cast<uintptr_t>(remoteSource.get()),
                               sourceBytes, remoteMr),
            0);
  remoteMrGuard.set(clientEngine, remoteMr);

  ASSERT_EQ(flagcxP2pEngineReg(
                serverEngine,
                reinterpret_cast<uintptr_t>(localDestination.get()), destBytes,
                localMr),
            0);
  localMrGuard.set(serverEngine, localMr);

  char descBuf[FLAGCX_P2P_DESC_SIZE] = {};
  ASSERT_EQ(flagcxP2pEnginePrepareDesc(clientEngine, remoteMr, remoteSource.get(),
                                       sourceBytes, descBuf),
            0);

  FlagcxP2pRdmaDesc remoteDesc;
  flagcxP2pDeserializeRdmaDesc(descBuf, &remoteDesc);
  ASSERT_EQ(flagcxP2pEngineUpdateDesc(
                remoteDesc,
                reinterpret_cast<uint64_t>(remoteSource.as<uint32_t>() +
                                           kSrcOffsetElems),
                static_cast<uint32_t>(readBytes)),
            0);

  uint64_t transferId = 0;
  ASSERT_EQ(flagcxP2pEngineRead(serverConn, localMr,
                                localDestination.as<uint32_t>() +
                                    kDstOffsetElems,
                                readBytes, remoteDesc, &transferId),
            0);
  ASSERT_TRUE(pollTransferDone(serverConn, transferId, std::chrono::seconds(10)))
      << "Timed out waiting for retargeted flagcxP2pEngineRead completion";

  copyDeviceToHost(kServerGpuIdx, serverStream, hostActualDestination.get(),
                   localDestination.get(), destBytes);
  for (size_t i = 0; i < kDstOffsetElems; ++i) {
    EXPECT_EQ(actualDestination[i], expectedDestination[i]);
  }
  for (size_t i = 0; i < kReadElems; ++i) {
    EXPECT_EQ(actualDestination[kDstOffsetElems + i],
              expectedSource[kSrcOffsetElems + i]);
  }
  for (size_t i = kDstOffsetElems + kReadElems; i < kDestElems; ++i) {
    EXPECT_EQ(actualDestination[i], expectedDestination[i]);
  }
}

} // namespace
