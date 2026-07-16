/*************************************************************************
 * Copyright (c) 2026. All Rights Reserved.
 * Cross-process IPC memory handle test (requires MPI and KunlunXin XPU).
 ************************************************************************/

#include <gtest/gtest.h>
#include <mpi.h>

#include "adaptor.h"
#include "flagcx.h"


namespace {

#define ASSERT_FLAGCX_SUCCESS(expr)                                           \
  do {                                                                        \
    flagcxResult_t result = (expr);                                            \
    if (result != flagcxSuccess) {                                             \
      ADD_FAILURE() << #expr << " returned " << static_cast<int>(result);      \
      MPI_Abort(MPI_COMM_WORLD, static_cast<int>(result));                     \
      return;                                                                  \
    }                                                                          \
  } while (0)

#define ASSERT_MPI_SUCCESS(expr)                                              \
  do {                                                                        \
    int result = (expr);                                                       \
    if (result != MPI_SUCCESS) {                                               \
      ADD_FAILURE() << #expr << " returned " << result;                        \
      MPI_Abort(MPI_COMM_WORLD, result);                                       \
      return;                                                                  \
    }                                                                          \
  } while (0)

class IpcMemHandleMpiTest : public ::testing::Test {
protected:
  void SetUp() override {
    flagcxDeviceHandleInit(&devHandle);
    ASSERT_NE(devHandle, nullptr);

    int deviceCount = 0;
    ASSERT_FLAGCX_SUCCESS(devHandle->getDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
      ADD_FAILURE() << "No visible XPU device";
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

TEST_F(IpcMemHandleMpiTest, CrossProcessLifecycle) {
  int rank = -1;
  int worldSize = 0;
  ASSERT_MPI_SUCCESS(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  ASSERT_MPI_SUCCESS(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));

  if (worldSize != 2) {
    GTEST_SKIP() << "CrossProcessLifecycle requires exactly 2 MPI ranks";
  }

  constexpr size_t bufferSize = 4096;
  constexpr int expectedValue = 0x12345678;

  if (rank == 0) {
    void *devPtr = nullptr;
    ASSERT_FLAGCX_SUCCESS(devHandle->deviceMalloc(
        &devPtr, bufferSize, flagcxMemDevice, nullptr));
    ASSERT_NE(devPtr, nullptr);

    int hostValue = expectedValue;
    ASSERT_FLAGCX_SUCCESS(devHandle->deviceMemcpy(
        devPtr, &hostValue, sizeof(hostValue), flagcxMemcpyHostToDevice,
        nullptr));
    ASSERT_FLAGCX_SUCCESS(devHandle->streamSynchronize(nullptr));

    flagcxIpcMemHandle_t handle = nullptr;
    size_t handleSize = 0;
    ASSERT_FLAGCX_SUCCESS(
        devHandle->ipcMemHandleCreate(&handle, &handleSize));
    ASSERT_NE(handle, nullptr);
    ASSERT_GT(handleSize, static_cast<size_t>(0));
    ASSERT_FLAGCX_SUCCESS(devHandle->ipcMemHandleGet(handle, devPtr));

    ASSERT_MPI_SUCCESS(MPI_Send(&handleSize, sizeof(handleSize), MPI_BYTE, 1,
                                0, MPI_COMM_WORLD));
    ASSERT_MPI_SUCCESS(MPI_Send(handle, static_cast<int>(handleSize), MPI_BYTE,
                                1, 1, MPI_COMM_WORLD));

    int acknowledgement = 0;
    ASSERT_MPI_SUCCESS(MPI_Recv(&acknowledgement, 1, MPI_INT, 1, 2,
                                MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    ASSERT_EQ(acknowledgement, 1);

    ASSERT_FLAGCX_SUCCESS(devHandle->ipcMemHandleFree(handle));
    ASSERT_FLAGCX_SUCCESS(
        devHandle->deviceFree(devPtr, flagcxMemDevice, nullptr));
  } else {
    size_t receivedHandleSize = 0;
    ASSERT_MPI_SUCCESS(MPI_Recv(&receivedHandleSize, sizeof(receivedHandleSize),
                                MPI_BYTE, 0, 0, MPI_COMM_WORLD,
                                MPI_STATUS_IGNORE));

    flagcxIpcMemHandle_t handle = nullptr;
    size_t localHandleSize = 0;
    ASSERT_FLAGCX_SUCCESS(
        devHandle->ipcMemHandleCreate(&handle, &localHandleSize));
    ASSERT_NE(handle, nullptr);
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
    ASSERT_NE(mappedPtr, nullptr);

    int receivedValue = 0;
    ASSERT_FLAGCX_SUCCESS(devHandle->deviceMemcpy(
        &receivedValue, mappedPtr, sizeof(receivedValue),
        flagcxMemcpyDeviceToHost, nullptr));
    ASSERT_FLAGCX_SUCCESS(devHandle->streamSynchronize(nullptr));
    EXPECT_EQ(receivedValue, expectedValue);

    ASSERT_FLAGCX_SUCCESS(devHandle->ipcMemHandleClose(mappedPtr));
    ASSERT_FLAGCX_SUCCESS(devHandle->ipcMemHandleFree(handle));

    int acknowledgement = 1;
    ASSERT_MPI_SUCCESS(
        MPI_Send(&acknowledgement, 1, MPI_INT, 0, 2, MPI_COMM_WORLD));
  }

  ASSERT_MPI_SUCCESS(MPI_Barrier(MPI_COMM_WORLD));
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
