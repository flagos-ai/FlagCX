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
enum class IpcAllocationKind { Device, Gdr };
enum class IpcCopyMode { Synchronous, Asynchronous };

flagcxResult_t allocateIpcTestBuffer(flagcxDeviceHandle_t devHandle,
                                     IpcAllocationKind kind, void **ptr,
                                     size_t size) {
  if (kind == IpcAllocationKind::Device) {
    return devHandle->deviceMalloc(ptr, size, flagcxMemDevice, nullptr);
  }
  if (deviceAdaptor == nullptr || deviceAdaptor->gdrMemAlloc == nullptr) {
    return flagcxNotSupported;
  }
  return deviceAdaptor->gdrMemAlloc(ptr, size, nullptr);
}

flagcxResult_t freeIpcTestBuffer(flagcxDeviceHandle_t devHandle,
                                 IpcAllocationKind kind, void *ptr) {
  if (kind == IpcAllocationKind::Device) {
    return devHandle->deviceFree(ptr, flagcxMemDevice, nullptr);
  }
  if (deviceAdaptor == nullptr || deviceAdaptor->gdrMemFree == nullptr) {
    return flagcxNotSupported;
  }
  return deviceAdaptor->gdrMemFree(ptr, nullptr);
}

void runCrossGpuIpcTransfer(
    flagcxDeviceHandle_t devHandle, CrossGpuIpcDirection direction,
    IpcAllocationKind exportedAllocation = IpcAllocationKind::Device,
    IpcAllocationKind localAllocation = IpcAllocationKind::Device,
    IpcCopyMode copyMode = IpcCopyMode::Asynchronous, int exporterRank = 0,
    size_t allocationSize = 4096, size_t transferOffset = 0,
    size_t transferSize = 0) {
  int rank = -1;
  int worldSize = 0;
  ASSERT_MPI_SUCCESS(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  ASSERT_MPI_SUCCESS(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
  if (worldSize != 2) {
    GTEST_SKIP() << "Cross-GPU IPC tests require exactly 2 MPI ranks";
  }
  ASSERT_MPI_TRUE(exporterRank == 0 || exporterRank == 1);
  if (transferSize == 0)
    transferSize = allocationSize;
  ASSERT_MPI_TRUE(transferOffset <= allocationSize);
  ASSERT_MPI_TRUE(transferSize <= allocationSize - transferOffset);
  const int importerRank = 1 - exporterRank;

  int localDeviceCount = 0;
  ASSERT_FLAGCX_SUCCESS(devHandle->getDeviceCount(&localDeviceCount));
  int minDeviceCount = 0;
  ASSERT_MPI_SUCCESS(MPI_Allreduce(&localDeviceCount, &minDeviceCount, 1,
                                   MPI_INT, MPI_MIN, MPI_COMM_WORLD));
  if (minDeviceCount < 2) {
    GTEST_SKIP() << "Cross-GPU IPC tests require at least 2 visible devices";
  }
  ASSERT_FLAGCX_SUCCESS(devHandle->setDevice(rank));

  int localAllocatorsAvailable =
      (exportedAllocation == IpcAllocationKind::Device &&
       localAllocation == IpcAllocationKind::Device) ||
      (deviceAdaptor != nullptr && deviceAdaptor->gdrMemAlloc != nullptr &&
       deviceAdaptor->gdrMemFree != nullptr);
  int allAllocatorsAvailable = 0;
  ASSERT_MPI_SUCCESS(MPI_Allreduce(&localAllocatorsAvailable,
                                   &allAllocatorsAvailable, 1, MPI_INT, MPI_MIN,
                                   MPI_COMM_WORLD));
  if (!allAllocatorsAvailable) {
    GTEST_SKIP() << "GDR memory allocation is not available";
  }

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

  std::vector<unsigned char> expected(transferSize);
  for (size_t i = 0; i < transferSize; ++i) {
    expected[i] = static_cast<unsigned char>((i * 17 + 3) & 0xff);
  }

  flagcxStream_t stream = nullptr;
  ASSERT_FLAGCX_SUCCESS(devHandle->streamCreate(&stream));
  ASSERT_MPI_TRUE(stream != nullptr);

  const int tagBase =
      10 + exporterRank * 100 + static_cast<int>(direction) * 40 +
      static_cast<int>(exportedAllocation) * 20 +
      static_cast<int>(localAllocation) * 10 + static_cast<int>(copyMode) * 3;
  if (rank == exporterRank) {
    void *exportedPtr = nullptr;
    ASSERT_FLAGCX_SUCCESS(allocateIpcTestBuffer(devHandle, exportedAllocation,
                                                &exportedPtr, allocationSize));
    ASSERT_MPI_TRUE(exportedPtr != nullptr);

    ASSERT_FLAGCX_SUCCESS(devHandle->deviceMemset(
        exportedPtr, 0, allocationSize, flagcxMemDevice, nullptr));
    if (direction == CrossGpuIpcDirection::Read) {
      void *exportedRange = static_cast<char *>(exportedPtr) + transferOffset;
      ASSERT_FLAGCX_SUCCESS(
          devHandle->deviceMemcpy(exportedRange, expected.data(), transferSize,
                                  flagcxMemcpyHostToDevice, nullptr));
    }
    ASSERT_FLAGCX_SUCCESS(devHandle->deviceSynchronize());
    ASSERT_FLAGCX_SUCCESS(devHandle->ipcMemHandleGet(handle, exportedPtr));

    ASSERT_MPI_SUCCESS(MPI_Send(&localHandleSize, sizeof(localHandleSize),
                                MPI_BYTE, importerRank, tagBase,
                                MPI_COMM_WORLD));
    ASSERT_MPI_SUCCESS(MPI_Send(handle, static_cast<int>(localHandleSize),
                                MPI_BYTE, importerRank, tagBase + 1,
                                MPI_COMM_WORLD));

    int acknowledgement = 0;
    ASSERT_MPI_SUCCESS(MPI_Recv(&acknowledgement, 1, MPI_INT, importerRank,
                                tagBase + 2, MPI_COMM_WORLD,
                                MPI_STATUS_IGNORE));
    ASSERT_MPI_TRUE(acknowledgement == 1);

    if (direction == CrossGpuIpcDirection::Write) {
      std::vector<unsigned char> actual(transferSize, 0);
      void *exportedRange = static_cast<char *>(exportedPtr) + transferOffset;
      ASSERT_FLAGCX_SUCCESS(
          devHandle->deviceMemcpy(actual.data(), exportedRange, transferSize,
                                  flagcxMemcpyDeviceToHost, nullptr));
      ASSERT_FLAGCX_SUCCESS(devHandle->deviceSynchronize());
      ASSERT_MPI_TRUE(actual == expected);
    }

    ASSERT_FLAGCX_SUCCESS(devHandle->ipcMemHandleFree(handle));
    ASSERT_FLAGCX_SUCCESS(
        freeIpcTestBuffer(devHandle, exportedAllocation, exportedPtr));
  } else {
    size_t receivedHandleSize = 0;
    ASSERT_MPI_SUCCESS(MPI_Recv(&receivedHandleSize, sizeof(receivedHandleSize),
                                MPI_BYTE, exporterRank, tagBase, MPI_COMM_WORLD,
                                MPI_STATUS_IGNORE));
    ASSERT_MPI_TRUE(localHandleSize == receivedHandleSize);
    ASSERT_MPI_SUCCESS(MPI_Recv(handle, static_cast<int>(receivedHandleSize),
                                MPI_BYTE, exporterRank, tagBase + 1,
                                MPI_COMM_WORLD, MPI_STATUS_IGNORE));

    void *mappedPtr = nullptr;
    ASSERT_FLAGCX_SUCCESS(devHandle->ipcMemHandleOpen(handle, &mappedPtr));
    ASSERT_MPI_TRUE(mappedPtr != nullptr);

    void *localPtr = nullptr;
    ASSERT_FLAGCX_SUCCESS(allocateIpcTestBuffer(devHandle, localAllocation,
                                                &localPtr, transferSize));
    ASSERT_MPI_TRUE(localPtr != nullptr);
    void *mappedRange = static_cast<char *>(mappedPtr) + transferOffset;

    flagcxStream_t copyStream =
        copyMode == IpcCopyMode::Asynchronous ? stream : nullptr;

    if (direction == CrossGpuIpcDirection::Read) {
      ASSERT_FLAGCX_SUCCESS(devHandle->deviceMemset(localPtr, 0, transferSize,
                                                    flagcxMemDevice, nullptr));
      ASSERT_FLAGCX_SUCCESS(devHandle->deviceSynchronize());
      ASSERT_FLAGCX_SUCCESS(
          devHandle->deviceMemcpy(localPtr, mappedRange, transferSize,
                                  flagcxMemcpyDeviceToDevice, copyStream));
      if (copyMode == IpcCopyMode::Asynchronous) {
        ASSERT_FLAGCX_SUCCESS(devHandle->streamSynchronize(stream));
      } else {
        ASSERT_FLAGCX_SUCCESS(devHandle->deviceSynchronize());
      }

      std::vector<unsigned char> actual(transferSize, 0);
      ASSERT_FLAGCX_SUCCESS(
          devHandle->deviceMemcpy(actual.data(), localPtr, transferSize,
                                  flagcxMemcpyDeviceToHost, nullptr));
      ASSERT_FLAGCX_SUCCESS(devHandle->deviceSynchronize());
      ASSERT_MPI_TRUE(actual == expected);
    } else {
      ASSERT_FLAGCX_SUCCESS(
          devHandle->deviceMemcpy(localPtr, expected.data(), transferSize,
                                  flagcxMemcpyHostToDevice, nullptr));
      ASSERT_FLAGCX_SUCCESS(devHandle->deviceSynchronize());
      ASSERT_FLAGCX_SUCCESS(
          devHandle->deviceMemcpy(mappedRange, localPtr, transferSize,
                                  flagcxMemcpyDeviceToDevice, copyStream));
      if (copyMode == IpcCopyMode::Asynchronous) {
        ASSERT_FLAGCX_SUCCESS(devHandle->streamSynchronize(stream));
      } else {
        ASSERT_FLAGCX_SUCCESS(devHandle->deviceSynchronize());
      }
    }

    ASSERT_FLAGCX_SUCCESS(
        freeIpcTestBuffer(devHandle, localAllocation, localPtr));
    ASSERT_FLAGCX_SUCCESS(devHandle->ipcMemHandleClose(mappedPtr));
    ASSERT_FLAGCX_SUCCESS(devHandle->ipcMemHandleFree(handle));

    int acknowledgement = 1;
    ASSERT_MPI_SUCCESS(MPI_Send(&acknowledgement, 1, MPI_INT, exporterRank,
                                tagBase + 2, MPI_COMM_WORLD));
  }

  ASSERT_FLAGCX_SUCCESS(devHandle->streamDestroy(stream));
  ASSERT_MPI_SUCCESS(MPI_Barrier(MPI_COMM_WORLD));
}

void runBidirectionalCrossGpuGdrIpcWrite(flagcxDeviceHandle_t devHandle) {
  int rank = -1;
  int worldSize = 0;
  ASSERT_MPI_SUCCESS(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  ASSERT_MPI_SUCCESS(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
  if (worldSize != 2) {
    GTEST_SKIP() << "Bidirectional cross-GPU IPC tests require exactly 2 MPI "
                    "ranks";
  }

  int localDeviceCount = 0;
  ASSERT_FLAGCX_SUCCESS(devHandle->getDeviceCount(&localDeviceCount));
  int minDeviceCount = 0;
  ASSERT_MPI_SUCCESS(MPI_Allreduce(&localDeviceCount, &minDeviceCount, 1,
                                   MPI_INT, MPI_MIN, MPI_COMM_WORLD));
  if (minDeviceCount < 2) {
    GTEST_SKIP() << "Bidirectional cross-GPU IPC tests require at least 2 "
                    "visible devices";
  }
  ASSERT_FLAGCX_SUCCESS(devHandle->setDevice(rank));

  int localCapabilitiesAvailable = deviceAdaptor != nullptr &&
                                   deviceAdaptor->gdrMemAlloc != nullptr &&
                                   deviceAdaptor->gdrMemFree != nullptr &&
                                   devHandle->ipcMemHandleCreate != nullptr &&
                                   devHandle->ipcMemHandleGet != nullptr &&
                                   devHandle->ipcMemHandleOpen != nullptr &&
                                   devHandle->ipcMemHandleClose != nullptr &&
                                   devHandle->ipcMemHandleFree != nullptr;
  int allCapabilitiesAvailable = 0;
  ASSERT_MPI_SUCCESS(MPI_Allreduce(&localCapabilitiesAvailable,
                                   &allCapabilitiesAvailable, 1, MPI_INT,
                                   MPI_MIN, MPI_COMM_WORLD));
  if (!allCapabilitiesAvailable) {
    GTEST_SKIP() << "GDR allocation or IPC memory handles are not available";
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
  void *exportedPtr = nullptr;
  void *sourcePtr = nullptr;
  ASSERT_FLAGCX_SUCCESS(allocateIpcTestBuffer(devHandle, IpcAllocationKind::Gdr,
                                              &exportedPtr, bufferSize));
  ASSERT_FLAGCX_SUCCESS(allocateIpcTestBuffer(devHandle, IpcAllocationKind::Gdr,
                                              &sourcePtr, bufferSize));
  ASSERT_MPI_TRUE(exportedPtr != nullptr);
  ASSERT_MPI_TRUE(sourcePtr != nullptr);

  std::vector<unsigned char> expected(bufferSize);
  for (size_t i = 0; i < bufferSize; ++i) {
    expected[i] = static_cast<unsigned char>((i * 29 + rank * 37 + 11) & 0xff);
  }
  ASSERT_FLAGCX_SUCCESS(
      devHandle->deviceMemcpy(sourcePtr, expected.data(), bufferSize,
                              flagcxMemcpyHostToDevice, nullptr));
  ASSERT_FLAGCX_SUCCESS(devHandle->deviceMemset(exportedPtr, 0, bufferSize,
                                                flagcxMemDevice, nullptr));
  ASSERT_FLAGCX_SUCCESS(devHandle->deviceSynchronize());
  ASSERT_FLAGCX_SUCCESS(devHandle->ipcMemHandleGet(handle, exportedPtr));

  unsigned long long handleSize = localHandleSize;
  unsigned long long minHandleSize = 0;
  unsigned long long maxHandleSize = 0;
  ASSERT_MPI_SUCCESS(MPI_Allreduce(&handleSize, &minHandleSize, 1,
                                   MPI_UNSIGNED_LONG_LONG, MPI_MIN,
                                   MPI_COMM_WORLD));
  ASSERT_MPI_SUCCESS(MPI_Allreduce(&handleSize, &maxHandleSize, 1,
                                   MPI_UNSIGNED_LONG_LONG, MPI_MAX,
                                   MPI_COMM_WORLD));
  ASSERT_MPI_TRUE(minHandleSize == maxHandleSize);

  std::vector<unsigned char> handles(worldSize * localHandleSize);
  ASSERT_MPI_SUCCESS(MPI_Allgather(
      handle, static_cast<int>(localHandleSize), MPI_BYTE, handles.data(),
      static_cast<int>(localHandleSize), MPI_BYTE, MPI_COMM_WORLD));

  const int peer = 1 - rank;
  flagcxIpcMemHandle_t peerHandle = reinterpret_cast<flagcxIpcMemHandle_t>(
      handles.data() + peer * localHandleSize);
  void *mappedPeerPtr = nullptr;
  ASSERT_FLAGCX_SUCCESS(
      devHandle->ipcMemHandleOpen(peerHandle, &mappedPeerPtr));
  ASSERT_MPI_TRUE(mappedPeerPtr != nullptr);

  flagcxStream_t stream = nullptr;
  ASSERT_FLAGCX_SUCCESS(devHandle->streamCreate(&stream));
  ASSERT_MPI_TRUE(stream != nullptr);

  // RMA initializes a full mesh of peer mappings before it submits a transfer.
  // Exercise both directions sequentially so a failure identifies the exact
  // writer/target direction without introducing concurrent-write ambiguity.
  for (int writer = 0; writer < worldSize; ++writer) {
    const int target = 1 - writer;
    if (rank == target) {
      ASSERT_FLAGCX_SUCCESS(devHandle->deviceMemset(exportedPtr, 0, bufferSize,
                                                    flagcxMemDevice, nullptr));
      ASSERT_FLAGCX_SUCCESS(devHandle->deviceSynchronize());
    }
    ASSERT_MPI_SUCCESS(MPI_Barrier(MPI_COMM_WORLD));

    if (rank == writer) {
      ASSERT_FLAGCX_SUCCESS(
          devHandle->deviceMemcpy(mappedPeerPtr, sourcePtr, bufferSize,
                                  flagcxMemcpyDeviceToDevice, stream));
      ASSERT_FLAGCX_SUCCESS(devHandle->streamSynchronize(stream));
    }
    ASSERT_MPI_SUCCESS(MPI_Barrier(MPI_COMM_WORLD));

    int localDataValid = 1;
    if (rank == target) {
      std::vector<unsigned char> actual(bufferSize, 0);
      ASSERT_FLAGCX_SUCCESS(
          devHandle->deviceMemcpy(actual.data(), exportedPtr, bufferSize,
                                  flagcxMemcpyDeviceToHost, nullptr));
      ASSERT_FLAGCX_SUCCESS(devHandle->deviceSynchronize());
      for (size_t i = 0; i < bufferSize; ++i) {
        unsigned char expectedValue =
            static_cast<unsigned char>((i * 29 + writer * 37 + 11) & 0xff);
        if (actual[i] != expectedValue) {
          localDataValid = 0;
          break;
        }
      }
    }
    int allDataValid = 0;
    ASSERT_MPI_SUCCESS(MPI_Allreduce(&localDataValid, &allDataValid, 1, MPI_INT,
                                     MPI_MIN, MPI_COMM_WORLD));
    ASSERT_MPI_TRUE(allDataValid == 1);
  }

  ASSERT_FLAGCX_SUCCESS(devHandle->streamDestroy(stream));
  ASSERT_FLAGCX_SUCCESS(devHandle->ipcMemHandleClose(mappedPeerPtr));
  ASSERT_FLAGCX_SUCCESS(devHandle->ipcMemHandleFree(handle));
  ASSERT_FLAGCX_SUCCESS(
      freeIpcTestBuffer(devHandle, IpcAllocationKind::Gdr, sourcePtr));
  ASSERT_FLAGCX_SUCCESS(
      freeIpcTestBuffer(devHandle, IpcAllocationKind::Gdr, exportedPtr));
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

TEST_F(IpcMemHandleMpiTest, CrossGpuImportedGdrMappingReadSync) {
  runCrossGpuIpcTransfer(devHandle, CrossGpuIpcDirection::Read,
                         IpcAllocationKind::Gdr, IpcAllocationKind::Device,
                         IpcCopyMode::Synchronous);
}

TEST_F(IpcMemHandleMpiTest, CrossGpuImportedGdrMappingReadAsync) {
  runCrossGpuIpcTransfer(devHandle, CrossGpuIpcDirection::Read,
                         IpcAllocationKind::Gdr, IpcAllocationKind::Device,
                         IpcCopyMode::Asynchronous);
}

TEST_F(IpcMemHandleMpiTest, CrossGpuDeviceToImportedGdrWriteSync) {
  runCrossGpuIpcTransfer(devHandle, CrossGpuIpcDirection::Write,
                         IpcAllocationKind::Gdr, IpcAllocationKind::Device,
                         IpcCopyMode::Synchronous);
}

TEST_F(IpcMemHandleMpiTest, CrossGpuDeviceToImportedGdrWriteAsync) {
  runCrossGpuIpcTransfer(devHandle, CrossGpuIpcDirection::Write,
                         IpcAllocationKind::Gdr, IpcAllocationKind::Device,
                         IpcCopyMode::Asynchronous);
}

TEST_F(IpcMemHandleMpiTest, CrossGpuGdrToImportedDeviceWriteSync) {
  runCrossGpuIpcTransfer(devHandle, CrossGpuIpcDirection::Write,
                         IpcAllocationKind::Device, IpcAllocationKind::Gdr,
                         IpcCopyMode::Synchronous);
}

TEST_F(IpcMemHandleMpiTest, CrossGpuGdrToImportedDeviceWriteAsync) {
  runCrossGpuIpcTransfer(devHandle, CrossGpuIpcDirection::Write,
                         IpcAllocationKind::Device, IpcAllocationKind::Gdr,
                         IpcCopyMode::Asynchronous);
}

TEST_F(IpcMemHandleMpiTest, CrossGpuGdrToImportedGdrWriteSync) {
  runCrossGpuIpcTransfer(devHandle, CrossGpuIpcDirection::Write,
                         IpcAllocationKind::Gdr, IpcAllocationKind::Gdr,
                         IpcCopyMode::Synchronous);
}

TEST_F(IpcMemHandleMpiTest, CrossGpuGdrToImportedGdrWriteAsync) {
  runCrossGpuIpcTransfer(devHandle, CrossGpuIpcDirection::Write,
                         IpcAllocationKind::Gdr, IpcAllocationKind::Gdr,
                         IpcCopyMode::Asynchronous);
}

TEST_F(IpcMemHandleMpiTest, CrossGpuReverseGdrToImportedGdrWriteSync) {
  runCrossGpuIpcTransfer(devHandle, CrossGpuIpcDirection::Write,
                         IpcAllocationKind::Gdr, IpcAllocationKind::Gdr,
                         IpcCopyMode::Synchronous, 1);
}

TEST_F(IpcMemHandleMpiTest, CrossGpuReverseGdrToImportedGdrWriteAsync) {
  runCrossGpuIpcTransfer(devHandle, CrossGpuIpcDirection::Write,
                         IpcAllocationKind::Gdr, IpcAllocationKind::Gdr,
                         IpcCopyMode::Asynchronous, 1);
}

TEST_F(IpcMemHandleMpiTest, CrossGpuBidirectionalGdrMappingsWriteAsync) {
  runBidirectionalCrossGpuGdrIpcWrite(devHandle);
}

TEST_F(IpcMemHandleMpiTest, CrossGpuLargeGdrSubrangesWriteAsync) {
  constexpr size_t allocationSize = 1 << 20;
  constexpr size_t transferSize = 64;
  for (size_t offset :
       {size_t{0}, size_t{0x400}, allocationSize - transferSize}) {
    runCrossGpuIpcTransfer(devHandle, CrossGpuIpcDirection::Write,
                           IpcAllocationKind::Gdr, IpcAllocationKind::Gdr,
                           IpcCopyMode::Asynchronous, 0, allocationSize, offset,
                           transferSize);
  }
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
