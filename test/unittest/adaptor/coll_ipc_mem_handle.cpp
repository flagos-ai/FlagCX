/*************************************************************************
 * Copyright (c) 2026. All Rights Reserved.
 * Cross-process IPC memory handle test.
 *
 * Run exactly two MPI ranks on the same host. Device IPC handles are
 * host-local and cannot be transferred between different nodes.
 ************************************************************************/

#include <gtest/gtest.h>
#include <mpi.h>

#include <vector>

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

class IpcMemHandleMpiTest : public ::testing::Test {
protected:
  void SetUp() override {
    flagcxDeviceHandleInit(&devHandle);
    ASSERT_MPI_TRUE(devHandle != nullptr);

    int deviceCount = 0;
    ASSERT_FLAGCX_SUCCESS(devHandle->getDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
      ADD_FAILURE() << "No visible device";
      MPI_Abort(MPI_COMM_WORLD, 1);
      return;
    }
    ASSERT_FLAGCX_SUCCESS(devHandle->setDevice(0));
  }

  void TearDown() override {
    if (devHandle != nullptr) {
      flagcxDeviceHandleFree(devHandle);
    }
  }

  flagcxDeviceHandle_t devHandle = nullptr;
};

enum class CrossGpuIpcDirection { Read, Write };

void runCrossGpuIpcTransfer(flagcxDeviceHandle_t devHandle,
                            CrossGpuIpcDirection direction) {
  int rank = -1;
  int worldSize = 0;
  ASSERT_MPI_SUCCESS(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  ASSERT_MPI_SUCCESS(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
  if (worldSize != 2) {
    GTEST_SKIP() << "Cross-GPU IPC tests require exactly 2 MPI ranks";
  }

  int localDeviceCount = 0;
  ASSERT_FLAGCX_SUCCESS(devHandle->getDeviceCount(&localDeviceCount));
  int minDeviceCount = 0;
  ASSERT_MPI_SUCCESS(MPI_Allreduce(&localDeviceCount, &minDeviceCount, 1,
                                   MPI_INT, MPI_MIN, MPI_COMM_WORLD));
  if (minDeviceCount < 2) {
    GTEST_SKIP() << "Cross-GPU IPC tests require at least 2 visible devices";
  }
  ASSERT_FLAGCX_SUCCESS(devHandle->setDevice(rank));

  int localApisAvailable = devHandle->ipcMemHandleCreate != nullptr &&
                           devHandle->ipcMemHandleGet != nullptr &&
                           devHandle->ipcMemHandleOpen != nullptr &&
                           devHandle->ipcMemHandleClose != nullptr &&
                           devHandle->ipcMemHandleFree != nullptr;
  int allApisAvailable = 0;
  ASSERT_MPI_SUCCESS(MPI_Allreduce(&localApisAvailable, &allApisAvailable, 1,
                                   MPI_INT, MPI_MIN, MPI_COMM_WORLD));
  if (!allApisAvailable) {
    GTEST_SKIP() << "IPC memory handle APIs are not available";
  }

  flagcxIpcMemHandle_t handle = nullptr;
  size_t localHandleSize = 0;
  flagcxResult_t createResult =
      devHandle->ipcMemHandleCreate(&handle, &localHandleSize);
  if (createResult != flagcxSuccess && createResult != flagcxNotSupported) {
    ADD_FAILURE() << "ipcMemHandleCreate returned "
                  << static_cast<int>(createResult);
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(createResult));
    return;
  }
  int localSupported = createResult != flagcxNotSupported;
  int allSupported = 0;
  ASSERT_MPI_SUCCESS(MPI_Allreduce(&localSupported, &allSupported, 1, MPI_INT,
                                   MPI_MIN, MPI_COMM_WORLD));
  if (!allSupported) {
    if (createResult == flagcxSuccess && handle != nullptr) {
      ASSERT_FLAGCX_SUCCESS(devHandle->ipcMemHandleFree(handle));
    }
    GTEST_SKIP() << "IPC memory handles are not supported";
  }
  ASSERT_FLAGCX_SUCCESS(createResult);
  ASSERT_MPI_TRUE(handle != nullptr);
  ASSERT_MPI_TRUE(localHandleSize > 0);

  constexpr size_t bufferSize = 4096;
  std::vector<unsigned char> expected(bufferSize);
  for (size_t i = 0; i < bufferSize; ++i) {
    expected[i] = static_cast<unsigned char>((i * 17 + 3) & 0xff);
  }

  flagcxStream_t stream = nullptr;
  ASSERT_FLAGCX_SUCCESS(devHandle->streamCreate(&stream));
  ASSERT_MPI_TRUE(stream != nullptr);

  const int tagBase = direction == CrossGpuIpcDirection::Read ? 10 : 20;
  if (rank == 0) {
    void *exportedPtr = nullptr;
    ASSERT_FLAGCX_SUCCESS(devHandle->deviceMalloc(&exportedPtr, bufferSize,
                                                  flagcxMemDevice, nullptr));
    ASSERT_MPI_TRUE(exportedPtr != nullptr);

    if (direction == CrossGpuIpcDirection::Read) {
      ASSERT_FLAGCX_SUCCESS(
          devHandle->deviceMemcpy(exportedPtr, expected.data(), bufferSize,
                                  flagcxMemcpyHostToDevice, stream));
    } else {
      ASSERT_FLAGCX_SUCCESS(devHandle->deviceMemset(exportedPtr, 0, bufferSize,
                                                    flagcxMemDevice, stream));
    }
    ASSERT_FLAGCX_SUCCESS(devHandle->streamSynchronize(stream));
    ASSERT_FLAGCX_SUCCESS(devHandle->ipcMemHandleGet(handle, exportedPtr));

    ASSERT_MPI_SUCCESS(MPI_Send(&localHandleSize, sizeof(localHandleSize),
                                MPI_BYTE, 1, tagBase, MPI_COMM_WORLD));
    ASSERT_MPI_SUCCESS(MPI_Send(handle, static_cast<int>(localHandleSize),
                                MPI_BYTE, 1, tagBase + 1, MPI_COMM_WORLD));

    int acknowledgement = 0;
    ASSERT_MPI_SUCCESS(MPI_Recv(&acknowledgement, 1, MPI_INT, 1, tagBase + 2,
                                MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    ASSERT_MPI_TRUE(acknowledgement == 1);

    if (direction == CrossGpuIpcDirection::Write) {
      std::vector<unsigned char> actual(bufferSize, 0);
      ASSERT_FLAGCX_SUCCESS(
          devHandle->deviceMemcpy(actual.data(), exportedPtr, bufferSize,
                                  flagcxMemcpyDeviceToHost, stream));
      ASSERT_FLAGCX_SUCCESS(devHandle->streamSynchronize(stream));
      ASSERT_MPI_TRUE(actual == expected);
    }

    ASSERT_FLAGCX_SUCCESS(devHandle->ipcMemHandleFree(handle));
    ASSERT_FLAGCX_SUCCESS(
        devHandle->deviceFree(exportedPtr, flagcxMemDevice, nullptr));
  } else {
    size_t receivedHandleSize = 0;
    ASSERT_MPI_SUCCESS(MPI_Recv(&receivedHandleSize, sizeof(receivedHandleSize),
                                MPI_BYTE, 0, tagBase, MPI_COMM_WORLD,
                                MPI_STATUS_IGNORE));
    ASSERT_MPI_TRUE(localHandleSize == receivedHandleSize);
    ASSERT_MPI_SUCCESS(MPI_Recv(handle, static_cast<int>(receivedHandleSize),
                                MPI_BYTE, 0, tagBase + 1, MPI_COMM_WORLD,
                                MPI_STATUS_IGNORE));

    void *mappedPtr = nullptr;
    ASSERT_FLAGCX_SUCCESS(devHandle->ipcMemHandleOpen(handle, &mappedPtr));
    ASSERT_MPI_TRUE(mappedPtr != nullptr);

    void *localPtr = nullptr;
    ASSERT_FLAGCX_SUCCESS(devHandle->deviceMalloc(&localPtr, bufferSize,
                                                  flagcxMemDevice, nullptr));
    ASSERT_MPI_TRUE(localPtr != nullptr);

    if (direction == CrossGpuIpcDirection::Read) {
      ASSERT_FLAGCX_SUCCESS(devHandle->deviceMemset(localPtr, 0, bufferSize,
                                                    flagcxMemDevice, stream));
      ASSERT_FLAGCX_SUCCESS(devHandle->deviceMemcpy(
          localPtr, mappedPtr, bufferSize, flagcxMemcpyDeviceToDevice, stream));
      ASSERT_FLAGCX_SUCCESS(devHandle->streamSynchronize(stream));

      std::vector<unsigned char> actual(bufferSize, 0);
      ASSERT_FLAGCX_SUCCESS(
          devHandle->deviceMemcpy(actual.data(), localPtr, bufferSize,
                                  flagcxMemcpyDeviceToHost, stream));
      ASSERT_FLAGCX_SUCCESS(devHandle->streamSynchronize(stream));
      ASSERT_MPI_TRUE(actual == expected);
    } else {
      ASSERT_FLAGCX_SUCCESS(
          devHandle->deviceMemcpy(localPtr, expected.data(), bufferSize,
                                  flagcxMemcpyHostToDevice, stream));
      ASSERT_FLAGCX_SUCCESS(devHandle->deviceMemcpy(
          mappedPtr, localPtr, bufferSize, flagcxMemcpyDeviceToDevice, stream));
      ASSERT_FLAGCX_SUCCESS(devHandle->streamSynchronize(stream));
    }

    ASSERT_FLAGCX_SUCCESS(
        devHandle->deviceFree(localPtr, flagcxMemDevice, nullptr));
    ASSERT_FLAGCX_SUCCESS(devHandle->ipcMemHandleClose(mappedPtr));
    ASSERT_FLAGCX_SUCCESS(devHandle->ipcMemHandleFree(handle));

    int acknowledgement = 1;
    ASSERT_MPI_SUCCESS(
        MPI_Send(&acknowledgement, 1, MPI_INT, 0, tagBase + 2, MPI_COMM_WORLD));
  }

  ASSERT_FLAGCX_SUCCESS(devHandle->streamDestroy(stream));
  ASSERT_MPI_SUCCESS(MPI_Barrier(MPI_COMM_WORLD));
}

TEST_F(IpcMemHandleMpiTest, CrossProcessLifecycle) {
  int rank = -1;
  int worldSize = 0;
  ASSERT_MPI_SUCCESS(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  ASSERT_MPI_SUCCESS(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));

  if (worldSize != 2) {
    GTEST_SKIP() << "CrossProcessLifecycle requires exactly 2 MPI ranks; "
                 << "run with: make MPI_NP=2 run-mpi";
  }

  constexpr size_t bufferSize = 4096;
  constexpr int expectedValue = 0x12345678;

  int localApisAvailable = devHandle->ipcMemHandleCreate != nullptr &&
                           devHandle->ipcMemHandleGet != nullptr &&
                           devHandle->ipcMemHandleOpen != nullptr &&
                           devHandle->ipcMemHandleClose != nullptr &&
                           devHandle->ipcMemHandleFree != nullptr;
  int allApisAvailable = 0;
  ASSERT_MPI_SUCCESS(MPI_Allreduce(&localApisAvailable, &allApisAvailable, 1,
                                   MPI_INT, MPI_MIN, MPI_COMM_WORLD));
  if (!allApisAvailable) {
    GTEST_SKIP() << "IPC memory handle APIs are not available";
  }

  // Every rank creates its receive/export storage before communication. This
  // also provides a coordinated runtime capability check for stub backends.
  flagcxIpcMemHandle_t handle = nullptr;
  size_t localHandleSize = 0;
  flagcxResult_t createResult =
      devHandle->ipcMemHandleCreate(&handle, &localHandleSize);
  if (createResult != flagcxSuccess && createResult != flagcxNotSupported) {
    ADD_FAILURE() << "ipcMemHandleCreate returned "
                  << static_cast<int>(createResult);
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(createResult));
    return;
  }
  int localSupported = createResult != flagcxNotSupported;
  int allSupported = 0;
  ASSERT_MPI_SUCCESS(MPI_Allreduce(&localSupported, &allSupported, 1, MPI_INT,
                                   MPI_MIN, MPI_COMM_WORLD));
  if (!allSupported) {
    if (createResult == flagcxSuccess && handle != nullptr) {
      ASSERT_FLAGCX_SUCCESS(devHandle->ipcMemHandleFree(handle));
    }
    GTEST_SKIP() << "IPC memory handles are not supported";
  }
  ASSERT_FLAGCX_SUCCESS(createResult);
  ASSERT_MPI_TRUE(handle != nullptr);
  ASSERT_MPI_TRUE(localHandleSize > 0);

  if (rank == 0) {
    void *devPtr = nullptr;
    ASSERT_FLAGCX_SUCCESS(
        devHandle->deviceMalloc(&devPtr, bufferSize, flagcxMemDevice, nullptr));
    ASSERT_MPI_TRUE(devPtr != nullptr);

    int hostValue = expectedValue;
    ASSERT_FLAGCX_SUCCESS(
        devHandle->deviceMemcpy(devPtr, &hostValue, sizeof(hostValue),
                                flagcxMemcpyHostToDevice, nullptr));
    ASSERT_FLAGCX_SUCCESS(devHandle->streamSynchronize(nullptr));

    ASSERT_FLAGCX_SUCCESS(devHandle->ipcMemHandleGet(handle, devPtr));

    ASSERT_MPI_SUCCESS(MPI_Send(&localHandleSize, sizeof(localHandleSize),
                                MPI_BYTE, 1, 0, MPI_COMM_WORLD));
    ASSERT_MPI_SUCCESS(MPI_Send(handle, static_cast<int>(localHandleSize),
                                MPI_BYTE, 1, 1, MPI_COMM_WORLD));

    int acknowledgement = 0;
    ASSERT_MPI_SUCCESS(MPI_Recv(&acknowledgement, 1, MPI_INT, 1, 2,
                                MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    ASSERT_MPI_TRUE(acknowledgement == 1);

    ASSERT_FLAGCX_SUCCESS(devHandle->ipcMemHandleFree(handle));
    ASSERT_FLAGCX_SUCCESS(
        devHandle->deviceFree(devPtr, flagcxMemDevice, nullptr));
  } else {
    size_t receivedHandleSize = 0;
    ASSERT_MPI_SUCCESS(MPI_Recv(&receivedHandleSize, sizeof(receivedHandleSize),
                                MPI_BYTE, 0, 0, MPI_COMM_WORLD,
                                MPI_STATUS_IGNORE));

    if (localHandleSize != receivedHandleSize) {
      ADD_FAILURE() << "IPC handle size mismatch: local=" << localHandleSize
                    << ", remote=" << receivedHandleSize;
      MPI_Abort(MPI_COMM_WORLD, 1);
      return;
    }

    ASSERT_MPI_SUCCESS(MPI_Recv(handle, static_cast<int>(receivedHandleSize),
                                MPI_BYTE, 0, 1, MPI_COMM_WORLD,
                                MPI_STATUS_IGNORE));

    void *mappedPtr = nullptr;
    ASSERT_FLAGCX_SUCCESS(devHandle->ipcMemHandleOpen(handle, &mappedPtr));
    ASSERT_MPI_TRUE(mappedPtr != nullptr);

    int receivedValue = 0;
    ASSERT_FLAGCX_SUCCESS(devHandle->deviceMemcpy(
        &receivedValue, mappedPtr, sizeof(receivedValue),
        flagcxMemcpyDeviceToHost, nullptr));
    ASSERT_FLAGCX_SUCCESS(devHandle->streamSynchronize(nullptr));
    ASSERT_MPI_TRUE(receivedValue == expectedValue);

    ASSERT_FLAGCX_SUCCESS(devHandle->ipcMemHandleClose(mappedPtr));
    ASSERT_FLAGCX_SUCCESS(devHandle->ipcMemHandleFree(handle));

    int acknowledgement = 1;
    ASSERT_MPI_SUCCESS(
        MPI_Send(&acknowledgement, 1, MPI_INT, 0, 2, MPI_COMM_WORLD));
  }

  ASSERT_MPI_SUCCESS(MPI_Barrier(MPI_COMM_WORLD));
}

TEST_F(IpcMemHandleMpiTest, CrossGpuImportedMappingRead) {
  runCrossGpuIpcTransfer(devHandle, CrossGpuIpcDirection::Read);
}

TEST_F(IpcMemHandleMpiTest, CrossGpuImportedMappingWrite) {
  runCrossGpuIpcTransfer(devHandle, CrossGpuIpcDirection::Write);
}

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
