/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "adaptor.h"
#include "flagcx.h"

namespace {

#define ASSERT_FLAGCX_SUCCESS(expr)                                            \
  do {                                                                         \
    flagcxResult_t result = (expr);                                            \
    if (result != flagcxSuccess) {                                             \
      ADD_FAILURE() << #expr << " returned " << static_cast<int>(result);      \
      MPI_Abort(MPI_COMM_WORLD, static_cast<int>(result));                     \
      return;                                                                  \
    }                                                                          \
  } while (0)

#define ASSERT_MPI_SUCCESS(expr)                                               \
  do {                                                                         \
    int result = (expr);                                                       \
    if (result != MPI_SUCCESS) {                                               \
      ADD_FAILURE() << #expr << " returned " << result;                        \
      MPI_Abort(MPI_COMM_WORLD, result);                                       \
      return;                                                                  \
    }                                                                          \
  } while (0)

#define ASSERT_MPI_TRUE(condition)                                             \
  do {                                                                         \
    if (!(condition)) {                                                        \
      ADD_FAILURE() << "MPI assertion failed: " << #condition;                 \
      MPI_Abort(MPI_COMM_WORLD, 1);                                            \
      return;                                                                  \
    }                                                                          \
  } while (0)

enum class ImportAccess { Read, Write };
enum class CopyThreading { SameThread, ProxyThreads };

struct IpcCase {
  ImportAccess access;
  size_t size;
  const char *name;
  CopyThreading threading = CopyThreading::SameThread;
  bool metaXOnly = false;
};

struct ThreadOperationResult {
  flagcxResult_t result = flagcxSuccess;
  const char *operation = nullptr;
};

static bool runThreadOperation(ThreadOperationResult &threadResult,
                               const char *operation, flagcxResult_t result) {
  if (result == flagcxSuccess) {
    return true;
  }
  threadResult.result = result;
  threadResult.operation = operation;
  return false;
}

class IpcMemHandleMpiTestBase : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_MPI_SUCCESS(MPI_Comm_rank(MPI_COMM_WORLD, &rank_));
    ASSERT_MPI_SUCCESS(MPI_Comm_size(MPI_COMM_WORLD, &worldSize_));
    if (worldSize_ != 2) {
      GTEST_SKIP() << "IPC tests require exactly two MPI ranks";
    }

    ASSERT_FLAGCX_SUCCESS(flagcxDeviceHandleInit(&devHandle_));
    ASSERT_MPI_TRUE(devHandle_ != nullptr);

    int localDeviceCount = 0;
    ASSERT_FLAGCX_SUCCESS(devHandle_->getDeviceCount(&localDeviceCount));
    ASSERT_MPI_SUCCESS(MPI_Allreduce(&localDeviceCount, &deviceCount_, 1,
                                     MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    if (deviceCount_ < 2) {
      GTEST_SKIP() << "Cross-GPU IPC tests require at least two devices";
    }

    device_ = rank_ % deviceCount_;
    ASSERT_FLAGCX_SUCCESS(devHandle_->setDevice(device_));
  }

  void TearDown() override {
    if (devHandle_ != nullptr) {
      flagcxDeviceHandleFree(devHandle_);
    }
  }

  void copyDeviceToDeviceAndWait(void *dst, void *src, size_t size) {
    flagcxStream_t stream = nullptr;
    flagcxEvent_t event = nullptr;
    ASSERT_FLAGCX_SUCCESS(devHandle_->streamCreate(&stream));
    ASSERT_MPI_TRUE(stream != nullptr);
    ASSERT_FLAGCX_SUCCESS(
        devHandle_->eventCreate(&event, flagcxEventDisableTiming));
    ASSERT_MPI_TRUE(event != nullptr);

    // Match the core P2P proxy: enqueue D2D on a non-default stream, record an
    // event, and surface an asynchronous mapping failure at completion.
    ASSERT_FLAGCX_SUCCESS(devHandle_->deviceMemcpy(
        dst, src, size, flagcxMemcpyDeviceToDevice, stream));
    ASSERT_FLAGCX_SUCCESS(devHandle_->eventRecord(event, stream));
    ASSERT_FLAGCX_SUCCESS(devHandle_->eventSynchronize(event));
    ASSERT_FLAGCX_SUCCESS(devHandle_->eventQuery(event));
    ASSERT_FLAGCX_SUCCESS(devHandle_->streamSynchronize(stream));

    ASSERT_FLAGCX_SUCCESS(devHandle_->eventDestroy(event));
    ASSERT_FLAGCX_SUCCESS(devHandle_->streamDestroy(stream));
  }

  void copyDeviceToDeviceAcrossProxyThreadsAndWait(void *dst, void *src,
                                                   size_t size) {
    struct SharedCompletionResources {
      std::mutex mutex;
      std::condition_variable condition;
      bool ready = false;
      bool release = false;
      flagcxStream_t stream = nullptr;
      flagcxEvent_t event = nullptr;
      ThreadOperationResult setupResult;
      ThreadOperationResult cleanupResult;
    } shared;

    // Match the core P2P lifecycle: the caller has already imported the IPC
    // mapping, the proxy service thread creates completion objects, and a
    // separate proxy progress thread performs the D2D transfer.
    std::thread serviceThread([&]() {
      bool setupOk =
          runThreadOperation(shared.setupResult, "setDevice(service thread)",
                             devHandle_->setDevice(device_));
      if (setupOk) {
        setupOk = runThreadOperation(shared.setupResult,
                                     "streamCreate(service thread)",
                                     devHandle_->streamCreate(&shared.stream));
      }
      if (setupOk) {
        runThreadOperation(
            shared.setupResult, "eventCreate(service thread)",
            devHandle_->eventCreate(&shared.event, flagcxEventDisableTiming));
      }

      {
        std::lock_guard<std::mutex> lock(shared.mutex);
        shared.ready = true;
      }
      shared.condition.notify_one();

      {
        std::unique_lock<std::mutex> lock(shared.mutex);
        shared.condition.wait(lock, [&]() { return shared.release; });
      }

      if (shared.event != nullptr) {
        runThreadOperation(shared.cleanupResult, "eventDestroy(service thread)",
                           devHandle_->eventDestroy(shared.event));
      }
      if (shared.stream != nullptr) {
        runThreadOperation(shared.cleanupResult,
                           "streamDestroy(service thread)",
                           devHandle_->streamDestroy(shared.stream));
      }
    });

    {
      std::unique_lock<std::mutex> lock(shared.mutex);
      shared.condition.wait(lock, [&]() { return shared.ready; });
    }

    if (shared.setupResult.result != flagcxSuccess) {
      {
        std::lock_guard<std::mutex> lock(shared.mutex);
        shared.release = true;
      }
      shared.condition.notify_one();
      serviceThread.join();
      ADD_FAILURE() << shared.setupResult.operation << " returned "
                    << static_cast<int>(shared.setupResult.result);
      MPI_Abort(MPI_COMM_WORLD, static_cast<int>(shared.setupResult.result));
      return;
    }

    ThreadOperationResult progressResult;
    std::thread progressThread([&]() {
      bool copyOk =
          runThreadOperation(progressResult, "setDevice(progress thread)",
                             devHandle_->setDevice(device_));
      if (copyOk) {
        copyOk = runThreadOperation(
            progressResult, "deviceMemcpy(progress thread)",
            devHandle_->deviceMemcpy(dst, src, size, flagcxMemcpyDeviceToDevice,
                                     shared.stream));
      }
      if (copyOk) {
        copyOk = runThreadOperation(
            progressResult, "eventRecord(progress thread)",
            devHandle_->eventRecord(shared.event, shared.stream));
      }
      if (copyOk) {
        copyOk = runThreadOperation(progressResult,
                                    "eventSynchronize(progress thread)",
                                    devHandle_->eventSynchronize(shared.event));
      }
      if (copyOk) {
        copyOk =
            runThreadOperation(progressResult, "eventQuery(progress thread)",
                               devHandle_->eventQuery(shared.event));
      }
      if (copyOk) {
        runThreadOperation(progressResult, "streamSynchronize(progress thread)",
                           devHandle_->streamSynchronize(shared.stream));
      }
    });
    progressThread.join();

    {
      std::lock_guard<std::mutex> lock(shared.mutex);
      shared.release = true;
    }
    shared.condition.notify_one();
    serviceThread.join();

    if (progressResult.result != flagcxSuccess) {
      ADD_FAILURE() << progressResult.operation << " returned "
                    << static_cast<int>(progressResult.result);
      MPI_Abort(MPI_COMM_WORLD, static_cast<int>(progressResult.result));
      return;
    }
    if (shared.cleanupResult.result != flagcxSuccess) {
      ADD_FAILURE() << shared.cleanupResult.operation << " returned "
                    << static_cast<int>(shared.cleanupResult.result);
      MPI_Abort(MPI_COMM_WORLD, static_cast<int>(shared.cleanupResult.result));
    }
  }

  std::vector<uint8_t> makePattern(size_t size) const {
    std::vector<uint8_t> pattern(size);
    for (size_t i = 0; i < size; ++i) {
      pattern[i] = static_cast<uint8_t>((i * 131 + 17) & 0xff);
    }
    return pattern;
  }

  void assertBufferMatches(const std::vector<uint8_t> &actual,
                           const std::vector<uint8_t> &expected) {
    auto mismatch = std::mismatch(actual.begin(), actual.end(),
                                  expected.begin(), expected.end());
    if (mismatch.first != actual.end()) {
      const size_t offset =
          static_cast<size_t>(mismatch.first - actual.begin());
      ADD_FAILURE() << "IPC data mismatch at offset " << offset
                    << ": actual=" << static_cast<int>(*mismatch.first)
                    << ", expected=" << static_cast<int>(*mismatch.second);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  flagcxDeviceHandle_t devHandle_ = nullptr;
  int rank_ = -1;
  int worldSize_ = 0;
  int deviceCount_ = 0;
  int device_ = -1;
};

class IpcMemHandleMpiTest : public IpcMemHandleMpiTestBase,
                            public ::testing::WithParamInterface<IpcCase> {};

TEST_P(IpcMemHandleMpiTest, ImportedMappingSupportsDeviceCopy) {
  const IpcCase testCase = GetParam();

  if (testCase.metaXOnly) {
    char vendor[128] = {};
    ASSERT_FLAGCX_SUCCESS(devHandle_->getVendor(vendor));
    int localIsMetaX = std::strcmp(vendor, "METAX") == 0;
    int allAreMetaX = 0;
    ASSERT_MPI_SUCCESS(MPI_Allreduce(&localIsMetaX, &allAreMetaX, 1, MPI_INT,
                                     MPI_MIN, MPI_COMM_WORLD));
    if (!allAreMetaX) {
      GTEST_SKIP() << "Cross-thread IPC diagnosis is MetaX-specific";
    }
  }

  int localApisAvailable = devHandle_->ipcMemHandleCreate != nullptr &&
                           devHandle_->ipcMemHandleGet != nullptr &&
                           devHandle_->ipcMemHandleOpen != nullptr &&
                           devHandle_->ipcMemHandleClose != nullptr &&
                           devHandle_->ipcMemHandleFree != nullptr &&
                           devHandle_->deviceMemcpy != nullptr &&
                           devHandle_->streamCreate != nullptr &&
                           devHandle_->streamSynchronize != nullptr &&
                           devHandle_->streamDestroy != nullptr &&
                           devHandle_->eventCreate != nullptr &&
                           devHandle_->eventRecord != nullptr &&
                           devHandle_->eventSynchronize != nullptr &&
                           devHandle_->eventQuery != nullptr &&
                           devHandle_->eventDestroy != nullptr;
  int allApisAvailable = 0;
  ASSERT_MPI_SUCCESS(MPI_Allreduce(&localApisAvailable, &allApisAvailable, 1,
                                   MPI_INT, MPI_MIN, MPI_COMM_WORLD));
  if (!allApisAvailable) {
    GTEST_SKIP() << "Required IPC or allocation APIs are unavailable";
  }

  flagcxIpcMemHandle_t handle = nullptr;
  size_t localHandleSize = 0;
  flagcxResult_t createResult =
      devHandle_->ipcMemHandleCreate(&handle, &localHandleSize);
  int localSupported = createResult != flagcxNotSupported;
  int allSupported = 0;
  ASSERT_MPI_SUCCESS(MPI_Allreduce(&localSupported, &allSupported, 1, MPI_INT,
                                   MPI_MIN, MPI_COMM_WORLD));
  if (!allSupported) {
    if (createResult == flagcxSuccess && handle != nullptr) {
      ASSERT_FLAGCX_SUCCESS(devHandle_->ipcMemHandleFree(handle));
    }
    GTEST_SKIP() << "IPC memory handles are unsupported";
  }
  ASSERT_FLAGCX_SUCCESS(createResult);
  ASSERT_MPI_TRUE(handle != nullptr);
  ASSERT_MPI_TRUE(localHandleSize > 0);

  std::vector<uint8_t> expected = makePattern(testCase.size);

  if (rank_ == 0) {
    void *ownerPtr = nullptr;
    ASSERT_FLAGCX_SUCCESS(devHandle_->deviceMalloc(&ownerPtr, testCase.size,
                                                   flagcxMemDevice, nullptr));
    ASSERT_MPI_TRUE(ownerPtr != nullptr);
    std::cout << "IPC case=" << testCase.name << " rank=" << rank_
              << " device=" << device_ << " ownerPtr=" << ownerPtr
              << " size=" << testCase.size << std::endl;

    if (testCase.access == ImportAccess::Read) {
      ASSERT_FLAGCX_SUCCESS(
          devHandle_->deviceMemcpy(ownerPtr, expected.data(), testCase.size,
                                   flagcxMemcpyHostToDevice, nullptr));
    } else {
      ASSERT_FLAGCX_SUCCESS(devHandle_->deviceMemset(ownerPtr, 0, testCase.size,
                                                     flagcxMemDevice, nullptr));
    }
    ASSERT_FLAGCX_SUCCESS(devHandle_->deviceSynchronize());
    ASSERT_FLAGCX_SUCCESS(devHandle_->ipcMemHandleGet(handle, ownerPtr));

    ASSERT_MPI_SUCCESS(MPI_Send(&localHandleSize, sizeof(localHandleSize),
                                MPI_BYTE, 1, 0, MPI_COMM_WORLD));
    ASSERT_MPI_SUCCESS(MPI_Send(handle, static_cast<int>(localHandleSize),
                                MPI_BYTE, 1, 1, MPI_COMM_WORLD));

    int acknowledgement = 0;
    ASSERT_MPI_SUCCESS(MPI_Recv(&acknowledgement, 1, MPI_INT, 1, 2,
                                MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    ASSERT_MPI_TRUE(acknowledgement == 1);

    if (testCase.access == ImportAccess::Write) {
      std::vector<uint8_t> actual(testCase.size);
      ASSERT_FLAGCX_SUCCESS(
          devHandle_->deviceMemcpy(actual.data(), ownerPtr, testCase.size,
                                   flagcxMemcpyDeviceToHost, nullptr));
      ASSERT_FLAGCX_SUCCESS(devHandle_->deviceSynchronize());
      assertBufferMatches(actual, expected);
    }

    ASSERT_FLAGCX_SUCCESS(devHandle_->ipcMemHandleFree(handle));
    ASSERT_FLAGCX_SUCCESS(
        devHandle_->deviceFree(ownerPtr, flagcxMemDevice, nullptr));
  } else {
    size_t receivedHandleSize = 0;
    ASSERT_MPI_SUCCESS(MPI_Recv(&receivedHandleSize, sizeof(receivedHandleSize),
                                MPI_BYTE, 0, 0, MPI_COMM_WORLD,
                                MPI_STATUS_IGNORE));
    ASSERT_MPI_TRUE(localHandleSize == receivedHandleSize);
    ASSERT_MPI_SUCCESS(MPI_Recv(handle, static_cast<int>(receivedHandleSize),
                                MPI_BYTE, 0, 1, MPI_COMM_WORLD,
                                MPI_STATUS_IGNORE));

    void *mappedPtr = nullptr;
    ASSERT_FLAGCX_SUCCESS(devHandle_->ipcMemHandleOpen(handle, &mappedPtr));
    ASSERT_MPI_TRUE(mappedPtr != nullptr);
    std::cout << "IPC case=" << testCase.name << " rank=" << rank_
              << " device=" << device_ << " mappedPtr=" << mappedPtr
              << " size=" << testCase.size << std::endl;

    void *localPtr = nullptr;
    ASSERT_FLAGCX_SUCCESS(devHandle_->deviceMalloc(&localPtr, testCase.size,
                                                   flagcxMemDevice, nullptr));
    ASSERT_MPI_TRUE(localPtr != nullptr);

    if (testCase.access == ImportAccess::Read) {
      if (testCase.threading == CopyThreading::ProxyThreads) {
        copyDeviceToDeviceAcrossProxyThreadsAndWait(localPtr, mappedPtr,
                                                    testCase.size);
      } else {
        copyDeviceToDeviceAndWait(localPtr, mappedPtr, testCase.size);
      }

      std::vector<uint8_t> actual(testCase.size);
      ASSERT_FLAGCX_SUCCESS(
          devHandle_->deviceMemcpy(actual.data(), localPtr, testCase.size,
                                   flagcxMemcpyDeviceToHost, nullptr));
      ASSERT_FLAGCX_SUCCESS(devHandle_->deviceSynchronize());
      assertBufferMatches(actual, expected);
    } else {
      ASSERT_FLAGCX_SUCCESS(
          devHandle_->deviceMemcpy(localPtr, expected.data(), testCase.size,
                                   flagcxMemcpyHostToDevice, nullptr));
      ASSERT_FLAGCX_SUCCESS(devHandle_->deviceSynchronize());
      if (testCase.threading == CopyThreading::ProxyThreads) {
        std::cout << "MetaX cross-thread IPC write rank=" << rank_
                  << " device=" << device_ << " mappedPtr=" << mappedPtr
                  << " size=" << testCase.size << std::endl;
        copyDeviceToDeviceAcrossProxyThreadsAndWait(mappedPtr, localPtr,
                                                    testCase.size);
      } else {
        copyDeviceToDeviceAndWait(mappedPtr, localPtr, testCase.size);
      }
    }

    ASSERT_FLAGCX_SUCCESS(
        devHandle_->deviceFree(localPtr, flagcxMemDevice, nullptr));
    ASSERT_FLAGCX_SUCCESS(devHandle_->ipcMemHandleClose(mappedPtr));
    ASSERT_FLAGCX_SUCCESS(devHandle_->ipcMemHandleFree(handle));

    int acknowledgement = 1;
    ASSERT_MPI_SUCCESS(
        MPI_Send(&acknowledgement, 1, MPI_INT, 0, 2, MPI_COMM_WORLD));
  }

  ASSERT_MPI_SUCCESS(MPI_Barrier(MPI_COMM_WORLD));
}

INSTANTIATE_TEST_SUITE_P(
    CrossGpuIpc, IpcMemHandleMpiTest,
    ::testing::Values(
        IpcCase{ImportAccess::Read, 4 * 1024, "Read4KiB"},
        IpcCase{ImportAccess::Write, 4 * 1024, "Write4KiB"},
        IpcCase{ImportAccess::Read, 16 * 1024 * 1024, "Read16MiB"},
        IpcCase{ImportAccess::Write, 16 * 1024 * 1024, "Write16MiB"},
        IpcCase{ImportAccess::Write, 4 * 1024, "MetaXCrossThreadWrite4KiB",
                CopyThreading::ProxyThreads, true}),
    [](const ::testing::TestParamInfo<IpcCase> &info) {
      return std::string(info.param.name);
    });

} // namespace

int main(int argc, char **argv) {
  int mpiResult = MPI_Init(&argc, &argv);
  if (mpiResult != MPI_SUCCESS) {
    return mpiResult;
  }

  ::testing::InitGoogleTest(&argc, argv);
  int testResult = RUN_ALL_TESTS();
  MPI_Finalize();
  return testResult;
}
