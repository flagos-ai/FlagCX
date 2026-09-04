// Unit tests for the FlagCX P2P engine one-sided READ path.
// These mirror the UCCL test flow: remote metadata exchange, connect/accept,
// remote descriptor handoff, initiator-side read, and async completion polling.
//
// The source and destination buffers are GPU-side allocations. The tests use
// pinned host staging buffers only to initialize device data and verify the
// final contents after the read completes.

#include <chrono>
#include <cstdint>
#include <cstdlib>
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

class ScopedEnvVar {
public:
  ScopedEnvVar(const char *nameArg, const char *value) : name(nameArg) {
    const char *old = std::getenv(name.c_str());
    if (old != nullptr) {
      hadOldValue = true;
      oldValue = old;
    }
    setenv(name.c_str(), value, 1);
  }

  ~ScopedEnvVar() {
    if (hadOldValue)
      setenv(name.c_str(), oldValue.c_str(), 1);
    else
      unsetenv(name.c_str());
  }

private:
  std::string name;
  std::string oldValue;
  bool hadOldValue = false;
};

class ScopedAllocation {
public:
  ScopedAllocation() = default;

  ~ScopedAllocation() { reset(); }

  ScopedAllocation(const ScopedAllocation &) = delete;
  ScopedAllocation &operator=(const ScopedAllocation &) = delete;

  flagcxResult_t allocDevice(flagcxDeviceHandle_t devHandleArg,
                             int deviceIdxArg, size_t sizeArg,
                             flagcxMemType_t memTypeArg,
                             flagcxStream_t streamArg) {
    reset();
    if (devHandleArg == nullptr || deviceIdxArg < 0)
      return flagcxInvalidArgument;
    devHandle = devHandleArg;
    deviceIdx = deviceIdxArg;
    memType = memTypeArg;
    stream = streamArg;
    allocKind = AllocKind::DeviceMalloc;
    const flagcxResult_t setRes = devHandle->setDevice(deviceIdx);
    if (setRes != flagcxSuccess) {
      allocKind = AllocKind::None;
      return setRes;
    }
    const flagcxResult_t allocRes =
        devHandle->deviceMalloc(&ptr, sizeArg, memTypeArg, streamArg);
    if (allocRes == flagcxSuccess && ptr == nullptr) {
      allocKind = AllocKind::None;
      return flagcxUnhandledDeviceError;
    }
    return allocRes;
  }

  flagcxResult_t allocGdr(flagcxDeviceHandle_t devHandleArg, int deviceIdxArg,
                          size_t sizeArg) {
    reset();
    if (devHandleArg == nullptr || deviceIdxArg < 0)
      return flagcxInvalidArgument;
    if (deviceAdaptor == nullptr || deviceAdaptor->gdrMemAlloc == nullptr ||
        deviceAdaptor->gdrMemFree == nullptr)
      return flagcxNotSupported;

    devHandle = devHandleArg;
    deviceIdx = deviceIdxArg;
    allocKind = AllocKind::GdrMalloc;
    const flagcxResult_t setRes = devHandle->setDevice(deviceIdx);
    if (setRes != flagcxSuccess) {
      allocKind = AllocKind::None;
      return setRes;
    }

    const flagcxResult_t allocRes =
        deviceAdaptor->gdrMemAlloc(&ptr, sizeArg, nullptr);
    if (allocRes == flagcxSuccess && ptr == nullptr) {
      allocKind = AllocKind::None;
      return flagcxUnhandledDeviceError;
    }
    return allocRes;
  }

  void *get() const { return ptr; }

  template <typename T>
  T *as() const {
    return static_cast<T *>(ptr);
  }

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

    if (allocKind == AllocKind::DeviceMalloc && devHandle != nullptr) {
      devHandle->deviceFree(ptr, memType, stream);
    } else if (allocKind == AllocKind::GdrMalloc && deviceAdaptor != nullptr &&
               deviceAdaptor->gdrMemFree != nullptr) {
      deviceAdaptor->gdrMemFree(ptr, nullptr);
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
    GdrMalloc,
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
  const size_t secondSep = firstSep == std::string::npos
                               ? std::string::npos
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
    ASSERT_EQ(flagcxDeviceHandleInit(&devHandle), flagcxSuccess);
    ASSERT_NE(devHandle, nullptr);

    int numDevices = 0;
    flagcxResult_t countRes = devHandle->getDeviceCount(&numDevices);
    ASSERT_EQ(countRes, flagcxSuccess) << "GPU device enumeration failed";
    ASSERT_GT(numDevices, kServerGpuIdx)
        << "At least 2 GPU devices are required";

    ASSERT_EQ(devHandle->setDevice(kServerGpuIdx), flagcxSuccess);
    ASSERT_EQ(devHandle->streamCreate(&serverStream), flagcxSuccess);
    serverEngine = flagcxP2pEngineCreate();
    ASSERT_EQ(devHandle->setDevice(kClientGpuIdx), flagcxSuccess);
    ASSERT_EQ(devHandle->streamCreate(&clientStream), flagcxSuccess);
    clientEngine = flagcxP2pEngineCreate();
    ASSERT_NE(serverStream, nullptr) << "Unable to create server stream";
    ASSERT_NE(clientStream, nullptr) << "Unable to create client stream";
    ASSERT_NE(serverEngine, nullptr) << "Unable to create server P2P engine";
    ASSERT_NE(clientEngine, nullptr) << "Unable to create client P2P engine";
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
    if (devHandle != nullptr) {
      flagcxDeviceHandleFree(devHandle);
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
      result.conn = flagcxP2pEngineAccept(clientEngine, ipBuf, sizeof(ipBuf),
                                          &remoteGpuIdx);
      result.remoteIp = ipBuf;
      result.remoteGpuIdx = remoteGpuIdx;
      return result;
    });

    serverConn =
        flagcxP2pEngineConnect(serverEngine, parsed.ip.c_str(),
                               parsed.remoteGpuIdx, parsed.rdmaPort, false);
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

  flagcxResult_t allocGpuBufferOnDevice(ScopedAllocation *buffer, size_t bytes,
                                        int deviceIdx, flagcxStream_t stream) {
    if (buffer == nullptr || devHandle == nullptr || stream == nullptr)
      return flagcxInvalidArgument;
    // P2P Engine registers GPU buffers for direct RDMA. Use the device
    // adaptor's GDR path so each platform applies its required allocation and
    // memory-visibility semantics.
    return buffer->allocGdr(devHandle, deviceIdx, bytes);
  }

  flagcxResult_t allocHostBuffer(ScopedAllocation *buffer, size_t bytes,
                                 int deviceIdx, flagcxStream_t stream) {
    if (buffer == nullptr || devHandle == nullptr || stream == nullptr)
      return flagcxInvalidArgument;
    return buffer->allocDevice(devHandle, deviceIdx, bytes, flagcxMemHost,
                               stream);
  }

  flagcxResult_t copyHostToDevice(int deviceIdx, flagcxStream_t stream,
                                  void *devicePtr, void *hostPtr,
                                  size_t bytes) {
    if (devHandle == nullptr || stream == nullptr || devicePtr == nullptr ||
        hostPtr == nullptr)
      return flagcxInvalidArgument;
    flagcxResult_t result = devHandle->setDevice(deviceIdx);
    if (result != flagcxSuccess)
      return result;
    result = devHandle->deviceMemcpy(devicePtr, hostPtr, bytes,
                                     flagcxMemcpyHostToDevice, stream);
    if (result != flagcxSuccess)
      return result;
    return devHandle->streamSynchronize(stream);
  }

  flagcxResult_t copyDeviceToHost(int deviceIdx, flagcxStream_t stream,
                                  void *hostPtr, void *devicePtr,
                                  size_t bytes) {
    if (devHandle == nullptr || stream == nullptr || hostPtr == nullptr ||
        devicePtr == nullptr)
      return flagcxInvalidArgument;
    flagcxResult_t result = devHandle->setDevice(deviceIdx);
    if (result != flagcxSuccess)
      return result;
    result = devHandle->deviceMemcpy(hostPtr, devicePtr, bytes,
                                     flagcxMemcpyDeviceToHost, stream);
    if (result != flagcxSuccess)
      return result;
    return devHandle->streamSynchronize(stream);
  }

  flagcxDeviceHandle_t devHandle = nullptr;
  flagcxComm_t comm = nullptr;
  flagcxStream_t clientStream = nullptr;
  flagcxStream_t serverStream = nullptr;
  FlagcxP2pEngine *serverEngine = nullptr;
  FlagcxP2pEngine *clientEngine = nullptr;
  FlagcxP2pConn *serverConn = nullptr;
  FlagcxP2pConn *clientConn = nullptr;
};

TEST_F(FlagcxP2pEngineReadTest,
       EngineDestroyDeregistersOutstandingTransportMemoryChunks) {
  ScopedEnvVar chunkSize("FLAGCX_ACCL_MAX_MR_MB", "1");
  constexpr size_t bytes = 2 * 1024 * 1024 + 64 * 1024;

  ScopedAllocation buffer;
  ASSERT_EQ(allocGpuBufferOnDevice(&buffer, bytes, kServerGpuIdx, serverStream),
            flagcxSuccess);

  FlagcxP2pMr mr = 0;
  ASSERT_EQ(flagcxP2pEngineReg(serverEngine,
                               reinterpret_cast<uintptr_t>(buffer.get()), bytes,
                               mr),
            0);

  // Leave the logical MR outstanding: engine teardown owns releasing every
  // underlying adaptor MR (one for IBRC, multiple chunks for BAREX).
  flagcxP2pEngineDestroy(serverEngine);
  serverEngine = nullptr;
}

TEST_F(FlagcxP2pEngineReadTest,
       EngineDestroyDoesNotRemoveAnotherEnginesLogicalMr) {
  constexpr size_t bytes = 4096;
  ScopedAllocation sharedHostBuffer;
  ASSERT_EQ(
      allocHostBuffer(&sharedHostBuffer, bytes, kClientGpuIdx, clientStream),
      flagcxSuccess);

  FlagcxP2pMr serverMr = 0;
  FlagcxP2pMr clientMr = 0;
  ASSERT_EQ(
      flagcxP2pEngineReg(serverEngine,
                         reinterpret_cast<uintptr_t>(sharedHostBuffer.get()),
                         bytes, serverMr),
      0);
  ASSERT_EQ(
      flagcxP2pEngineReg(clientEngine,
                         reinterpret_cast<uintptr_t>(sharedHostBuffer.get()),
                         bytes, clientMr),
      0);

  flagcxP2pEngineDestroy(serverEngine);
  serverEngine = nullptr;

  char descBuf[FLAGCX_P2P_DESC_SIZE] = {};
  EXPECT_EQ(flagcxP2pEnginePrepareDesc(clientEngine, clientMr,
                                       sharedHostBuffer.get(), bytes, descBuf),
            0);
  flagcxP2pEngineMrDestroy(clientEngine, clientMr);
}

TEST_F(FlagcxP2pEngineReadTest,
       ReadsWholeRegisteredGpuBufferAfterMetadataHandshake) {
  ASSERT_NO_FATAL_FAILURE(connectViaClientMetadata());

  constexpr size_t kElemCount = 1024;
  const size_t bytes = kElemCount * sizeof(uint32_t);

  ScopedAllocation remoteSource;
  ScopedAllocation localDestination;
  ScopedAllocation hostExpected;
  ScopedAllocation hostActual;

  ASSERT_EQ(
      allocGpuBufferOnDevice(&remoteSource, bytes, kClientGpuIdx, clientStream),
      flagcxSuccess);
  ASSERT_EQ(allocGpuBufferOnDevice(&localDestination, bytes, kServerGpuIdx,
                                   serverStream),
            flagcxSuccess);
  ASSERT_EQ(allocHostBuffer(&hostExpected, bytes, kClientGpuIdx, clientStream),
            flagcxSuccess);
  ASSERT_EQ(allocHostBuffer(&hostActual, bytes, kServerGpuIdx, serverStream),
            flagcxSuccess);

  uint32_t *expected = hostExpected.as<uint32_t>();
  uint32_t *actual = hostActual.as<uint32_t>();
  for (size_t i = 0; i < kElemCount; ++i) {
    expected[i] = static_cast<uint32_t>(i + 1);
    actual[i] = 0;
  }

  ASSERT_EQ(copyHostToDevice(kClientGpuIdx, clientStream, remoteSource.get(),
                             hostExpected.get(), bytes),
            flagcxSuccess);
  ASSERT_EQ(copyHostToDevice(kServerGpuIdx, serverStream,
                             localDestination.get(), hostActual.get(), bytes),
            flagcxSuccess);

  FlagcxP2pMr remoteMr = 0;
  FlagcxP2pMr localMr = 0;
  ScopedMr remoteMrGuard;
  ScopedMr localMrGuard;

  ASSERT_EQ(flagcxP2pEngineReg(clientEngine,
                               reinterpret_cast<uintptr_t>(remoteSource.get()),
                               bytes, remoteMr),
            0);
  remoteMrGuard.set(clientEngine, remoteMr);

  ASSERT_EQ(
      flagcxP2pEngineReg(serverEngine,
                         reinterpret_cast<uintptr_t>(localDestination.get()),
                         bytes, localMr),
      0);
  localMrGuard.set(serverEngine, localMr);

  char descBuf[FLAGCX_P2P_DESC_SIZE] = {};
  ASSERT_EQ(flagcxP2pEnginePrepareDesc(clientEngine, remoteMr,
                                       remoteSource.get(), bytes, descBuf),
            0);

  FlagcxP2pRdmaDesc remoteDesc;
  flagcxP2pDeserializeRdmaDesc(descBuf, &remoteDesc);

  uint64_t transferId = 0;
  EXPECT_NE(flagcxP2pEngineRead(serverConn, UINT64_MAX, localDestination.get(),
                                bytes, remoteDesc, &transferId),
            0)
      << "Read must reject a local buffer that is not owned by the supplied "
         "MR handle";
  ASSERT_EQ(flagcxP2pEngineRead(serverConn, localMr, localDestination.get(),
                                bytes, remoteDesc, &transferId),
            0);
  ASSERT_TRUE(
      pollTransferDone(serverConn, transferId, std::chrono::seconds(10)))
      << "Timed out waiting for flagcxP2pEngineRead completion";

  ASSERT_EQ(copyDeviceToHost(kServerGpuIdx, serverStream, hostActual.get(),
                             localDestination.get(), bytes),
            flagcxSuccess);
  for (size_t i = 0; i < kElemCount; ++i) {
    EXPECT_EQ(actual[i], expected[i]) << "Mismatch at index " << i;
  }
}

TEST_F(FlagcxP2pEngineReadTest,
       ReadsRetargetedRemoteGpuSubrangeIntoLocalWindow) {
  ASSERT_NO_FATAL_FAILURE(connectViaClientMetadata());

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

  ASSERT_EQ(allocGpuBufferOnDevice(&remoteSource, sourceBytes, kClientGpuIdx,
                                   clientStream),
            flagcxSuccess);
  ASSERT_EQ(allocGpuBufferOnDevice(&localDestination, destBytes, kServerGpuIdx,
                                   serverStream),
            flagcxSuccess);
  ASSERT_EQ(allocHostBuffer(&hostExpectedSource, sourceBytes, kClientGpuIdx,
                            clientStream),
            flagcxSuccess);
  ASSERT_EQ(allocHostBuffer(&hostExpectedDestination, destBytes, kServerGpuIdx,
                            serverStream),
            flagcxSuccess);
  ASSERT_EQ(allocHostBuffer(&hostActualDestination, destBytes, kServerGpuIdx,
                            serverStream),
            flagcxSuccess);

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

  ASSERT_EQ(copyHostToDevice(kClientGpuIdx, clientStream, remoteSource.get(),
                             hostExpectedSource.get(), sourceBytes),
            flagcxSuccess);
  ASSERT_EQ(copyHostToDevice(kServerGpuIdx, serverStream,
                             localDestination.get(),
                             hostExpectedDestination.get(), destBytes),
            flagcxSuccess);

  FlagcxP2pMr remoteMr = 0;
  FlagcxP2pMr localMr = 0;
  ScopedMr remoteMrGuard;
  ScopedMr localMrGuard;

  ASSERT_EQ(flagcxP2pEngineReg(clientEngine,
                               reinterpret_cast<uintptr_t>(remoteSource.get()),
                               sourceBytes, remoteMr),
            0);
  remoteMrGuard.set(clientEngine, remoteMr);

  ASSERT_EQ(
      flagcxP2pEngineReg(serverEngine,
                         reinterpret_cast<uintptr_t>(localDestination.get()),
                         destBytes, localMr),
      0);
  localMrGuard.set(serverEngine, localMr);

  char descBuf[FLAGCX_P2P_DESC_SIZE] = {};
  ASSERT_EQ(flagcxP2pEnginePrepareDesc(clientEngine, remoteMr,
                                       remoteSource.get(), sourceBytes,
                                       descBuf),
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
  ASSERT_EQ(
      flagcxP2pEngineRead(serverConn, localMr,
                          localDestination.as<uint32_t>() + kDstOffsetElems,
                          readBytes, remoteDesc, &transferId),
      0);
  ASSERT_TRUE(
      pollTransferDone(serverConn, transferId, std::chrono::seconds(10)))
      << "Timed out waiting for retargeted flagcxP2pEngineRead completion";

  ASSERT_EQ(copyDeviceToHost(kServerGpuIdx, serverStream,
                             hostActualDestination.get(),
                             localDestination.get(), destBytes),
            flagcxSuccess);
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

TEST_F(FlagcxP2pEngineReadTest, ConnectionDestroyQuiescesScheduledRead) {
  ASSERT_NO_FATAL_FAILURE(connectViaClientMetadata());

  constexpr size_t bytes = 4 * 1024 * 1024;
  ScopedAllocation remoteSource;
  ScopedAllocation localDestination;
  ASSERT_EQ(
      allocGpuBufferOnDevice(&remoteSource, bytes, kClientGpuIdx, clientStream),
      flagcxSuccess);
  ASSERT_EQ(allocGpuBufferOnDevice(&localDestination, bytes, kServerGpuIdx,
                                   serverStream),
            flagcxSuccess);

  FlagcxP2pMr remoteMr = 0;
  FlagcxP2pMr localMr = 0;
  ASSERT_EQ(flagcxP2pEngineReg(clientEngine,
                               reinterpret_cast<uintptr_t>(remoteSource.get()),
                               bytes, remoteMr),
            0);
  ASSERT_EQ(
      flagcxP2pEngineReg(serverEngine,
                         reinterpret_cast<uintptr_t>(localDestination.get()),
                         bytes, localMr),
      0);

  char descBuf[FLAGCX_P2P_DESC_SIZE] = {};
  ASSERT_EQ(flagcxP2pEnginePrepareDesc(clientEngine, remoteMr,
                                       remoteSource.get(), bytes, descBuf),
            0);
  FlagcxP2pRdmaDesc remoteDesc;
  flagcxP2pDeserializeRdmaDesc(descBuf, &remoteDesc);

  uint64_t transferId = 0;
  ASSERT_EQ(flagcxP2pEngineRead(serverConn, localMr, localDestination.get(),
                                bytes, remoteDesc, &transferId),
            0);

  // ConnDestroy must wait until workers have stopped touching the transport
  // comm. Under ASan/TSan this regresses the former close/delete-versus-poll
  // race even when the transfer has not yet reached the first CQ poll.
  flagcxP2pEngineConnDestroy(serverConn);
  serverConn = nullptr;

  flagcxP2pEngineMrDestroy(serverEngine, localMr);
  // The peer connection has just observed a disconnect, so leave its MR to
  // clientEngine teardown rather than requiring a removal ACK.
}

TEST_F(FlagcxP2pEngineReadTest, ReadsAcrossTransportMemoryRegistrationChunks) {
  // Keep the test small while forcing BAREX to create multiple physical MRs.
  // Establish the connection first so this also verifies dynamic publication
  // of every physical chunk into the peer's remote MR table.
  ScopedEnvVar chunkSize("FLAGCX_ACCL_MAX_MR_MB", "1");
  constexpr size_t kElemCount =
      (2 * 1024 * 1024 + 64 * 1024) / sizeof(uint32_t);
  const size_t bytes = kElemCount * sizeof(uint32_t);

  ScopedAllocation remoteSource;
  ScopedAllocation localDestination;
  ScopedAllocation hostExpected;
  ScopedAllocation hostActual;
  ASSERT_EQ(
      allocGpuBufferOnDevice(&remoteSource, bytes, kClientGpuIdx, clientStream),
      flagcxSuccess);
  ASSERT_EQ(allocGpuBufferOnDevice(&localDestination, bytes, kServerGpuIdx,
                                   serverStream),
            flagcxSuccess);
  ASSERT_EQ(allocHostBuffer(&hostExpected, bytes, kClientGpuIdx, clientStream),
            flagcxSuccess);
  ASSERT_EQ(allocHostBuffer(&hostActual, bytes, kServerGpuIdx, serverStream),
            flagcxSuccess);

  uint32_t *expected = hostExpected.as<uint32_t>();
  uint32_t *actual = hostActual.as<uint32_t>();
  for (size_t i = 0; i < kElemCount; i++) {
    expected[i] = static_cast<uint32_t>(i ^ 0x5A5A1234u);
    actual[i] = 0;
  }
  ASSERT_EQ(copyHostToDevice(kClientGpuIdx, clientStream, remoteSource.get(),
                             hostExpected.get(), bytes),
            flagcxSuccess);
  ASSERT_EQ(copyHostToDevice(kServerGpuIdx, serverStream,
                             localDestination.get(), hostActual.get(), bytes),
            flagcxSuccess);

  ASSERT_NO_FATAL_FAILURE(connectViaClientMetadata());

  FlagcxP2pMr remoteMr = 0;
  FlagcxP2pMr localMr = 0;
  ScopedMr remoteMrGuard;
  ScopedMr localMrGuard;
  ASSERT_EQ(flagcxP2pEngineReg(clientEngine,
                               reinterpret_cast<uintptr_t>(remoteSource.get()),
                               bytes, remoteMr),
            0);
  remoteMrGuard.set(clientEngine, remoteMr);
  ASSERT_EQ(
      flagcxP2pEngineReg(serverEngine,
                         reinterpret_cast<uintptr_t>(localDestination.get()),
                         bytes, localMr),
      0);
  localMrGuard.set(serverEngine, localMr);

  char descBuf[FLAGCX_P2P_DESC_SIZE] = {};
  ASSERT_EQ(flagcxP2pEnginePrepareDesc(clientEngine, remoteMr,
                                       remoteSource.get(), bytes, descBuf),
            0);
  FlagcxP2pRdmaDesc remoteDesc;
  flagcxP2pDeserializeRdmaDesc(descBuf, &remoteDesc);

  uint64_t transferId = 0;
  ASSERT_EQ(flagcxP2pEngineRead(serverConn, localMr, localDestination.get(),
                                bytes, remoteDesc, &transferId),
            0);
  ASSERT_TRUE(
      pollTransferDone(serverConn, transferId, std::chrono::seconds(20)))
      << "Timed out waiting for chunked P2P read completion";
  ASSERT_EQ(copyDeviceToHost(kServerGpuIdx, serverStream, hostActual.get(),
                             localDestination.get(), bytes),
            flagcxSuccess);
  for (size_t i = 0; i < kElemCount; i++)
    ASSERT_EQ(actual[i], expected[i]) << "Mismatch at index " << i;
}

TEST_F(FlagcxP2pEngineReadTest,
       DescriptorDoesNotCrossAdjacentLogicalMemoryRegistrations) {
  ASSERT_NO_FATAL_FAILURE(connectViaClientMetadata());

  constexpr size_t halfBytes = 4096;
  constexpr size_t totalBytes = 2 * halfBytes;
  ScopedAllocation remoteBuffer;
  ASSERT_EQ(
      allocHostBuffer(&remoteBuffer, totalBytes, kClientGpuIdx, clientStream),
      flagcxSuccess);

  const uintptr_t base = reinterpret_cast<uintptr_t>(remoteBuffer.get());
  FlagcxP2pMr firstMr = 0;
  FlagcxP2pMr secondMr = 0;
  ScopedMr firstMrGuard;
  ScopedMr secondMrGuard;
  ASSERT_EQ(flagcxP2pEngineReg(clientEngine, base, halfBytes, firstMr), 0);
  firstMrGuard.set(clientEngine, firstMr);
  ASSERT_EQ(
      flagcxP2pEngineReg(clientEngine, base + halfBytes, halfBytes, secondMr),
      0);
  secondMrGuard.set(clientEngine, secondMr);

  FlagcxP2pRdmaDesc desc = {};
  EXPECT_EQ(flagcxP2pEngineMakeDesc(serverConn, base, halfBytes, &desc), 0);
  EXPECT_EQ(desc.idx, firstMr);

  ScopedAllocation localBuffer;
  ASSERT_EQ(allocHostBuffer(&localBuffer, 2, kServerGpuIdx, serverStream),
            flagcxSuccess);
  FlagcxP2pMr localMr = 0;
  ScopedMr localMrGuard;
  ASSERT_EQ(flagcxP2pEngineReg(serverEngine,
                               reinterpret_cast<uintptr_t>(localBuffer.get()),
                               2, localMr),
            0);
  localMrGuard.set(serverEngine, localMr);

  ASSERT_EQ(flagcxP2pEngineUpdateDesc(desc, base + halfBytes - 1, 2), 0);
  uint64_t transferId = 0;
  EXPECT_NE(flagcxP2pEngineRead(serverConn, localMr, localBuffer.get(), 2, desc,
                                &transferId),
            0);

  EXPECT_EQ(
      flagcxP2pEngineMakeDesc(serverConn, base + halfBytes, halfBytes, &desc),
      0);
  EXPECT_NE(flagcxP2pEngineMakeDesc(serverConn, base, totalBytes, &desc), 0);
}

TEST_F(FlagcxP2pEngineReadTest, ReadsMultipleNonContiguousVectors) {
  constexpr size_t kElemCount = 128;
  constexpr size_t kIovCount = 3;
  const size_t bytes = kElemCount * sizeof(uint32_t);
  const size_t srcOffsets[kIovCount] = {3, 31, 79};
  const size_t dstOffsets[kIovCount] = {7, 43, 91};
  const size_t elemCounts[kIovCount] = {5, 11, 17};
  constexpr uint32_t kSentinel = 0xA5A5A5A5u;

  ScopedAllocation remoteSource;
  ScopedAllocation localDestination;
  ScopedAllocation hostSource;
  ScopedAllocation hostInitialDestination;
  ScopedAllocation hostActual;
  ASSERT_EQ(
      allocGpuBufferOnDevice(&remoteSource, bytes, kClientGpuIdx, clientStream),
      flagcxSuccess);
  ASSERT_EQ(allocGpuBufferOnDevice(&localDestination, bytes, kServerGpuIdx,
                                   serverStream),
            flagcxSuccess);
  ASSERT_EQ(allocHostBuffer(&hostSource, bytes, kClientGpuIdx, clientStream),
            flagcxSuccess);
  ASSERT_EQ(allocHostBuffer(&hostInitialDestination, bytes, kServerGpuIdx,
                            serverStream),
            flagcxSuccess);
  ASSERT_EQ(allocHostBuffer(&hostActual, bytes, kServerGpuIdx, serverStream),
            flagcxSuccess);

  std::vector<uint32_t> expected(kElemCount, kSentinel);
  for (size_t i = 0; i < kElemCount; ++i) {
    hostSource.as<uint32_t>()[i] = static_cast<uint32_t>(0x1000 + i);
    hostInitialDestination.as<uint32_t>()[i] = kSentinel;
  }
  for (size_t iov = 0; iov < kIovCount; ++iov) {
    for (size_t i = 0; i < elemCounts[iov]; ++i) {
      expected[dstOffsets[iov] + i] =
          hostSource.as<uint32_t>()[srcOffsets[iov] + i];
    }
  }
  ASSERT_EQ(copyHostToDevice(kClientGpuIdx, clientStream, remoteSource.get(),
                             hostSource.get(), bytes),
            flagcxSuccess);
  ASSERT_EQ(copyHostToDevice(kServerGpuIdx, serverStream,
                             localDestination.get(),
                             hostInitialDestination.get(), bytes),
            flagcxSuccess);

  FlagcxP2pMr remoteMr = 0;
  FlagcxP2pMr localMr = 0;
  ScopedMr remoteMrGuard;
  ScopedMr localMrGuard;
  ASSERT_EQ(flagcxP2pEngineReg(clientEngine,
                               reinterpret_cast<uintptr_t>(remoteSource.get()),
                               bytes, remoteMr),
            0);
  remoteMrGuard.set(clientEngine, remoteMr);
  ASSERT_EQ(
      flagcxP2pEngineReg(serverEngine,
                         reinterpret_cast<uintptr_t>(localDestination.get()),
                         bytes, localMr),
      0);
  localMrGuard.set(serverEngine, localMr);
  ASSERT_NO_FATAL_FAILURE(connectViaClientMetadata());

  char descBuf[FLAGCX_P2P_DESC_SIZE] = {};
  ASSERT_EQ(flagcxP2pEnginePrepareDesc(clientEngine, remoteMr,
                                       remoteSource.get(), bytes, descBuf),
            0);
  FlagcxP2pRdmaDesc baseDesc;
  flagcxP2pDeserializeRdmaDesc(descBuf, &baseDesc);

  std::vector<FlagcxP2pMr> mrIds(kIovCount, localMr);
  std::vector<void *> dstVec;
  std::vector<size_t> sizeVec;
  std::vector<FlagcxP2pRdmaDesc> descs;
  for (size_t iov = 0; iov < kIovCount; ++iov) {
    dstVec.push_back(localDestination.as<uint32_t>() + dstOffsets[iov]);
    sizeVec.push_back(elemCounts[iov] * sizeof(uint32_t));
    FlagcxP2pRdmaDesc desc = baseDesc;
    ASSERT_EQ(flagcxP2pEngineUpdateDesc(
                  desc,
                  reinterpret_cast<uint64_t>(remoteSource.as<uint32_t>() +
                                             srcOffsets[iov]),
                  static_cast<uint32_t>(sizeVec.back())),
              0);
    descs.push_back(desc);
  }

  uint64_t transferId = 0;
  const std::vector<FlagcxP2pMr> invalidMrIds(kIovCount, UINT64_MAX);
  EXPECT_NE(flagcxP2pEngineReadVector(serverConn, invalidMrIds, dstVec, sizeVec,
                                      descs, static_cast<int>(kIovCount),
                                      &transferId),
            0);
  ASSERT_EQ(flagcxP2pEngineReadVector(serverConn, mrIds, dstVec, sizeVec, descs,
                                      static_cast<int>(kIovCount), &transferId),
            0);
  ASSERT_TRUE(
      pollTransferDone(serverConn, transferId, std::chrono::seconds(10)));
  ASSERT_EQ(copyDeviceToHost(kServerGpuIdx, serverStream, hostActual.get(),
                             localDestination.get(), bytes),
            flagcxSuccess);
  for (size_t i = 0; i < kElemCount; ++i)
    EXPECT_EQ(hostActual.as<uint32_t>()[i], expected[i])
        << "Mismatch at index " << i;
}

TEST_F(FlagcxP2pEngineReadTest, WritesWholeRegisteredGpuBuffer) {
  constexpr size_t kElemCount = 1024;
  const size_t bytes = kElemCount * sizeof(uint32_t);
  ScopedAllocation localSource;
  ScopedAllocation remoteDestination;
  ScopedAllocation hostExpected;
  ScopedAllocation hostActual;
  ASSERT_EQ(
      allocGpuBufferOnDevice(&localSource, bytes, kServerGpuIdx, serverStream),
      flagcxSuccess);
  ASSERT_EQ(allocGpuBufferOnDevice(&remoteDestination, bytes, kClientGpuIdx,
                                   clientStream),
            flagcxSuccess);
  ASSERT_EQ(allocHostBuffer(&hostExpected, bytes, kServerGpuIdx, serverStream),
            flagcxSuccess);
  ASSERT_EQ(allocHostBuffer(&hostActual, bytes, kClientGpuIdx, clientStream),
            flagcxSuccess);
  for (size_t i = 0; i < kElemCount; ++i) {
    hostExpected.as<uint32_t>()[i] = static_cast<uint32_t>(0x2000 + i);
    hostActual.as<uint32_t>()[i] = 0;
  }
  ASSERT_EQ(copyHostToDevice(kServerGpuIdx, serverStream, localSource.get(),
                             hostExpected.get(), bytes),
            flagcxSuccess);
  ASSERT_EQ(copyHostToDevice(kClientGpuIdx, clientStream,
                             remoteDestination.get(), hostActual.get(), bytes),
            flagcxSuccess);

  FlagcxP2pMr localMr = 0;
  FlagcxP2pMr remoteMr = 0;
  ScopedMr localMrGuard;
  ScopedMr remoteMrGuard;
  ASSERT_EQ(flagcxP2pEngineReg(serverEngine,
                               reinterpret_cast<uintptr_t>(localSource.get()),
                               bytes, localMr),
            0);
  localMrGuard.set(serverEngine, localMr);
  ASSERT_EQ(
      flagcxP2pEngineReg(clientEngine,
                         reinterpret_cast<uintptr_t>(remoteDestination.get()),
                         bytes, remoteMr),
      0);
  remoteMrGuard.set(clientEngine, remoteMr);
  ASSERT_NO_FATAL_FAILURE(connectViaClientMetadata());

  FlagcxP2pRdmaDesc remoteDesc = {};
  ASSERT_EQ(flagcxP2pEngineMakeDesc(
                serverConn, reinterpret_cast<uint64_t>(remoteDestination.get()),
                static_cast<uint32_t>(bytes), &remoteDesc),
            0);
  uint64_t transferId = 0;
  EXPECT_NE(flagcxP2pEngineWrite(serverConn, UINT64_MAX, localSource.get(),
                                 bytes, remoteDesc, &transferId),
            0)
      << "Write must reject a local buffer that is not owned by the supplied "
         "MR handle";
  ASSERT_EQ(flagcxP2pEngineWrite(serverConn, localMr, localSource.get(), bytes,
                                 remoteDesc, &transferId),
            0);
  ASSERT_TRUE(
      pollTransferDone(serverConn, transferId, std::chrono::seconds(10)));
  ASSERT_EQ(copyDeviceToHost(kClientGpuIdx, clientStream, hostActual.get(),
                             remoteDestination.get(), bytes),
            flagcxSuccess);
  for (size_t i = 0; i < kElemCount; ++i)
    EXPECT_EQ(hostActual.as<uint32_t>()[i], hostExpected.as<uint32_t>()[i])
        << "Mismatch at index " << i;
}

TEST_F(FlagcxP2pEngineReadTest, WritesMultipleNonContiguousVectors) {
  constexpr size_t kElemCount = 128;
  constexpr size_t kIovCount = 3;
  const size_t bytes = kElemCount * sizeof(uint32_t);
  const size_t srcOffsets[kIovCount] = {2, 37, 83};
  const size_t dstOffsets[kIovCount] = {11, 49, 97};
  const size_t elemCounts[kIovCount] = {7, 13, 19};
  constexpr uint32_t kSentinel = 0x5A5A5A5Au;

  ScopedAllocation localSource;
  ScopedAllocation remoteDestination;
  ScopedAllocation hostSource;
  ScopedAllocation hostInitialDestination;
  ScopedAllocation hostActual;
  ASSERT_EQ(
      allocGpuBufferOnDevice(&localSource, bytes, kServerGpuIdx, serverStream),
      flagcxSuccess);
  ASSERT_EQ(allocGpuBufferOnDevice(&remoteDestination, bytes, kClientGpuIdx,
                                   clientStream),
            flagcxSuccess);
  ASSERT_EQ(allocHostBuffer(&hostSource, bytes, kServerGpuIdx, serverStream),
            flagcxSuccess);
  ASSERT_EQ(allocHostBuffer(&hostInitialDestination, bytes, kClientGpuIdx,
                            clientStream),
            flagcxSuccess);
  ASSERT_EQ(allocHostBuffer(&hostActual, bytes, kClientGpuIdx, clientStream),
            flagcxSuccess);

  std::vector<uint32_t> expected(kElemCount, kSentinel);
  for (size_t i = 0; i < kElemCount; ++i) {
    hostSource.as<uint32_t>()[i] = static_cast<uint32_t>(0x3000 + i);
    hostInitialDestination.as<uint32_t>()[i] = kSentinel;
  }
  for (size_t iov = 0; iov < kIovCount; ++iov) {
    for (size_t i = 0; i < elemCounts[iov]; ++i) {
      expected[dstOffsets[iov] + i] =
          hostSource.as<uint32_t>()[srcOffsets[iov] + i];
    }
  }
  ASSERT_EQ(copyHostToDevice(kServerGpuIdx, serverStream, localSource.get(),
                             hostSource.get(), bytes),
            flagcxSuccess);
  ASSERT_EQ(copyHostToDevice(kClientGpuIdx, clientStream,
                             remoteDestination.get(),
                             hostInitialDestination.get(), bytes),
            flagcxSuccess);

  FlagcxP2pMr localMr = 0;
  FlagcxP2pMr remoteMr = 0;
  ScopedMr localMrGuard;
  ScopedMr remoteMrGuard;
  ASSERT_EQ(flagcxP2pEngineReg(serverEngine,
                               reinterpret_cast<uintptr_t>(localSource.get()),
                               bytes, localMr),
            0);
  localMrGuard.set(serverEngine, localMr);
  ASSERT_EQ(
      flagcxP2pEngineReg(clientEngine,
                         reinterpret_cast<uintptr_t>(remoteDestination.get()),
                         bytes, remoteMr),
      0);
  remoteMrGuard.set(clientEngine, remoteMr);
  ASSERT_NO_FATAL_FAILURE(connectViaClientMetadata());

  std::vector<FlagcxP2pMr> mrIds(kIovCount, localMr);
  std::vector<void *> srcVec;
  std::vector<size_t> sizeVec;
  std::vector<FlagcxP2pRdmaDesc> descs;
  for (size_t iov = 0; iov < kIovCount; ++iov) {
    srcVec.push_back(localSource.as<uint32_t>() + srcOffsets[iov]);
    sizeVec.push_back(elemCounts[iov] * sizeof(uint32_t));
    FlagcxP2pRdmaDesc desc = {};
    ASSERT_EQ(flagcxP2pEngineMakeDesc(
                  serverConn,
                  reinterpret_cast<uint64_t>(remoteDestination.as<uint32_t>() +
                                             dstOffsets[iov]),
                  static_cast<uint32_t>(sizeVec.back()), &desc),
              0);
    descs.push_back(desc);
  }

  uint64_t transferId = 0;
  const std::vector<FlagcxP2pMr> invalidMrIds(kIovCount, UINT64_MAX);
  EXPECT_NE(flagcxP2pEngineWriteVector(
                serverConn, invalidMrIds, srcVec, sizeVec, descs,
                static_cast<int>(kIovCount), &transferId),
            0);
  ASSERT_EQ(flagcxP2pEngineWriteVector(serverConn, mrIds, srcVec, sizeVec,
                                       descs, static_cast<int>(kIovCount),
                                       &transferId),
            0);
  ASSERT_TRUE(
      pollTransferDone(serverConn, transferId, std::chrono::seconds(10)));
  ASSERT_EQ(copyDeviceToHost(kClientGpuIdx, clientStream, hostActual.get(),
                             remoteDestination.get(), bytes),
            flagcxSuccess);
  for (size_t i = 0; i < kElemCount; ++i)
    EXPECT_EQ(hostActual.as<uint32_t>()[i], expected[i])
        << "Mismatch at index " << i;
}

TEST_F(FlagcxP2pEngineReadTest, WriteVectorSyncCompletesBeforeReturn) {
  constexpr size_t kElemCount = 64;
  const size_t bytes = kElemCount * sizeof(uint32_t);
  ScopedAllocation localSource;
  ScopedAllocation remoteDestination;
  ScopedAllocation hostExpected;
  ScopedAllocation hostActual;
  ASSERT_EQ(
      allocGpuBufferOnDevice(&localSource, bytes, kServerGpuIdx, serverStream),
      flagcxSuccess);
  ASSERT_EQ(allocGpuBufferOnDevice(&remoteDestination, bytes, kClientGpuIdx,
                                   clientStream),
            flagcxSuccess);
  ASSERT_EQ(allocHostBuffer(&hostExpected, bytes, kServerGpuIdx, serverStream),
            flagcxSuccess);
  ASSERT_EQ(allocHostBuffer(&hostActual, bytes, kClientGpuIdx, clientStream),
            flagcxSuccess);
  for (size_t i = 0; i < kElemCount; ++i) {
    hostExpected.as<uint32_t>()[i] = static_cast<uint32_t>(0x4000 + i);
    hostActual.as<uint32_t>()[i] = 0;
  }
  ASSERT_EQ(copyHostToDevice(kServerGpuIdx, serverStream, localSource.get(),
                             hostExpected.get(), bytes),
            flagcxSuccess);
  ASSERT_EQ(copyHostToDevice(kClientGpuIdx, clientStream,
                             remoteDestination.get(), hostActual.get(), bytes),
            flagcxSuccess);

  FlagcxP2pMr localMr = 0;
  FlagcxP2pMr remoteMr = 0;
  ScopedMr localMrGuard;
  ScopedMr remoteMrGuard;
  ASSERT_EQ(flagcxP2pEngineReg(serverEngine,
                               reinterpret_cast<uintptr_t>(localSource.get()),
                               bytes, localMr),
            0);
  localMrGuard.set(serverEngine, localMr);
  ASSERT_EQ(
      flagcxP2pEngineReg(clientEngine,
                         reinterpret_cast<uintptr_t>(remoteDestination.get()),
                         bytes, remoteMr),
      0);
  remoteMrGuard.set(clientEngine, remoteMr);
  ASSERT_NO_FATAL_FAILURE(connectViaClientMetadata());

  FlagcxP2pRdmaDesc desc = {};
  ASSERT_EQ(flagcxP2pEngineMakeDesc(
                serverConn, reinterpret_cast<uint64_t>(remoteDestination.get()),
                static_cast<uint32_t>(bytes), &desc),
            0);
  const std::vector<FlagcxP2pMr> mrIds(1, localMr);
  const std::vector<void *> srcVec(1, localSource.get());
  const std::vector<size_t> sizeVec(1, bytes);
  const std::vector<FlagcxP2pRdmaDesc> descs(1, desc);
  ASSERT_EQ(
      flagcxP2pEngineWriteVectorSync(serverConn, mrIds, srcVec, sizeVec, descs),
      0);

  ASSERT_EQ(copyDeviceToHost(kClientGpuIdx, clientStream, hostActual.get(),
                             remoteDestination.get(), bytes),
            flagcxSuccess);
  for (size_t i = 0; i < kElemCount; ++i)
    EXPECT_EQ(hostActual.as<uint32_t>()[i], hostExpected.as<uint32_t>()[i])
        << "Mismatch at index " << i;
}

} // namespace
