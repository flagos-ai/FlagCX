// Hardware conformance for the exact visibility boundary consumed by a GPU.
// The test intentionally forbids any D2H copy or device synchronization
// between RDMA completion and the consumer kernel.

#include "rma_test.hpp"

#ifdef FLAGCX_TEST_GDR_VISIBILITY_CONSUMER

#include "comm.h"
#include "flagcx_device_adaptor.h"
#include "gdr_visibility_test.h"
#include "global_comm.h"
#include "onesided_types.h"
#include "sym_heap.h"

#include <cstring>
#include <vector>

namespace {

constexpr size_t kVisibilityBytes = 256 * 1024;
constexpr int kConsumerThreads = 256;

enum class ExpectedResult { Success, NotSupported, RemoteError };

ExpectedResult expectedResult(const char *name) {
  const char *value = std::getenv(name);
  EXPECT_NE(value, nullptr) << name << " must be set by the conformance job";
  if (value != nullptr && std::strcmp(value, "unsupported") == 0)
    return ExpectedResult::NotSupported;
  if (value != nullptr && std::strcmp(value, "remote_error") == 0)
    return ExpectedResult::RemoteError;
  EXPECT_TRUE(value != nullptr && std::strcmp(value, "success") == 0)
      << name << " must be success, unsupported, or remote_error";
  return ExpectedResult::Success;
}

bool conformanceEnabled() {
  const char *value = std::getenv("FLAGCX_CI_GDR_VISIBILITY_RUN");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

int expectedRequirement(const char *name) {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0')
    return -1;
  EXPECT_TRUE(std::strcmp(value, "0") == 0 || std::strcmp(value, "1") == 0)
      << name << " must be 0 or 1";
  return std::strcmp(value, "1") == 0 ? 1 : 0;
}

flagcxVmmMrRoute_t expectedRoute() {
  const char *value = std::getenv("FLAGCX_CI_EXPECT_VMM_MR_ROUTE");
  EXPECT_NE(value, nullptr)
      << "FLAGCX_CI_EXPECT_VMM_MR_ROUTE must be set by the conformance job";
  if (value != nullptr && std::strcmp(value, "va") == 0)
    return FLAGCX_VMM_MR_ROUTE_VA;
  if (value != nullptr && std::strcmp(value, "dmabuf") == 0)
    return FLAGCX_VMM_MR_ROUTE_DMABUF;
  EXPECT_TRUE(value != nullptr && std::strcmp(value, "none") == 0)
      << "expected route must be none, va, or dmabuf";
  return FLAGCX_VMM_MR_ROUTE_NONE;
}

flagcxOneSideHandleInfo *dataHandle(flagcxComm_t comm, flagcxWindow_t window) {
  if (comm == nullptr || comm->heteroComm == nullptr || window == nullptr ||
      window->defaultBase == nullptr)
    return nullptr;
  const int mrIndex = window->defaultBase->mrIndex;
  if (mrIndex < 0 || mrIndex >= comm->heteroComm->oneSideHandleCount ||
      comm->heteroComm->oneSideHandles == nullptr)
    return nullptr;
  return comm->heteroComm->oneSideHandles[mrIndex];
}

void verifyPolicyAndRoute(flagcxComm_t comm, flagcxWindow_t window,
                          uint32_t direction) {
  flagcxOneSideHandleInfo *handle = dataHandle(comm, window);
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(handle->registrationRoute, static_cast<uint8_t>(expectedRoute()));

  const char *requirementName = direction == FLAGCX_GDR_READ_REQUIRES_FLUSH
                                    ? "FLAGCX_CI_EXPECT_GDR_READ_REQUIRED"
                                    : "FLAGCX_CI_EXPECT_GDR_WRITE_REQUIRED";
  const int expected = expectedRequirement(requirementName);
  if (expected >= 0) {
    EXPECT_EQ((handle->gdrFlushRequirements & direction) != 0, expected != 0)
        << requirementName;
  }
}

flagcxResult_t prepareStatus(flagcxDeviceHandle_t device, int **deviceStatus) {
  flagcxResult_t result =
      flagcxMemAlloc(reinterpret_cast<void **>(deviceStatus), 2 * sizeof(int));
  if (result != flagcxSuccess)
    return result;
  return device->deviceMemset(*deviceStatus, 0, 2 * sizeof(int),
                              flagcxMemDevice, nullptr);
}

bool anyRankFailed(flagcxResult_t result) {
  int failed = result != flagcxSuccess;
  MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  return failed != 0;
}

void freeStatus(int *deviceStatus) {
  if (deviceStatus != nullptr)
    EXPECT_EQ(flagcxMemFree(deviceStatus), flagcxSuccess);
}

void verifyConsumerStatus(flagcxDeviceHandle_t device, int *deviceStatus) {
  int status[2] = {-1, -1};
  ASSERT_EQ(device->deviceMemcpy(status, deviceStatus, sizeof(status),
                                 flagcxMemcpyDeviceToHost, nullptr),
            flagcxSuccess);
  EXPECT_EQ(status[0], 0) << "consumer kernel observed stale RDMA payload";
  EXPECT_EQ(status[1], kConsumerThreads)
      << "consumer kernel did not execute all participating threads";
}

} // namespace

TEST_F(RmaTest, DirectConsumerReadVisibility) {
  if (!conformanceEnabled())
    GTEST_SKIP() << "Enabled only by the GDR visibility conformance job";
  ASSERT_EQ(nranks, 2) << "Conformance requires exactly two ranks";
  ASSERT_FALSE(requireIpc) << "Conformance requires the RDMA transport";
  verifyPolicyAndRoute(comm, dataWin, FLAGCX_GDR_READ_REQUIRES_FLUSH);

  const ExpectedResult expected =
      expectedResult("FLAGCX_CI_GDR_VISIBILITY_EXPECT_READ");
  constexpr uint8_t patternByte = 0xA7;
  int *deviceStatus = nullptr;
  flagcxResult_t statusPreparation = prepareStatus(devHandle, &deviceStatus);
  if (anyRankFailed(statusPreparation)) {
    freeStatus(deviceStatus);
    ADD_FAILURE() << "status-buffer preparation failed on at least one rank";
    return;
  }

  flagcxResult_t setup = flagcxSuccess;
  if (rank == 0) {
    std::vector<uint8_t> pattern(kVisibilityBytes, patternByte);
    setup = devHandle->deviceMemcpy(dataBuff, pattern.data(), pattern.size(),
                                    flagcxMemcpyHostToDevice, nullptr);
  } else {
    setup = devHandle->deviceMemset(dataBuff, 0, kVisibilityBytes,
                                    flagcxMemDevice, nullptr);
  }
  if (setup == flagcxSuccess)
    setup = devHandle->deviceSynchronize();
  if (anyRankFailed(setup)) {
    freeStatus(deviceStatus);
    ADD_FAILURE() << "payload preparation failed on at least one rank";
    return;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxResult_t operation = flagcxSuccess;
  flagcxResult_t launch = flagcxSuccess;
  if (rank == 1) {
    uint64_t before = 0;
    operation = flagcxReadCounter(comm, &before);
    if (operation == flagcxSuccess)
      operation = flagcxGet(comm, 0, 0, 0, kVisibilityBytes, 0, 0);
    if (operation == flagcxSuccess)
      operation = flagcxWaitCounter(comm, before + 1);

    // This is the conformance boundary: no copy or device synchronization is
    // permitted between completion (including any required flush) and launch.
    if (operation == flagcxSuccess) {
      launch = flagcxTestLaunchGdrVisibilityConsumer(
          dataBuff, kVisibilityBytes, patternByte, deviceStatus, stream);
      if (launch == flagcxSuccess)
        launch = devHandle->streamSynchronize(stream);
    }
  }

  int operationCode = static_cast<int>(operation);
  int launchCode = static_cast<int>(launch);
  MPI_Bcast(&operationCode, 1, MPI_INT, 1, MPI_COMM_WORLD);
  MPI_Bcast(&launchCode, 1, MPI_INT, 1, MPI_COMM_WORLD);
  if (expected == ExpectedResult::NotSupported) {
    EXPECT_EQ(operationCode, static_cast<int>(flagcxNotSupported));
  } else if (expected == ExpectedResult::RemoteError) {
    // GET is asynchronous: an unsupported post-READ visibility flush is
    // published through the proxy error flag and observed by WaitCounter.
    EXPECT_EQ(operationCode, static_cast<int>(flagcxRemoteError));
  } else {
    EXPECT_EQ(operationCode, static_cast<int>(flagcxSuccess));
    EXPECT_EQ(launchCode, static_cast<int>(flagcxSuccess));
    if (rank == 1 && launch == flagcxSuccess)
      verifyConsumerStatus(devHandle, deviceStatus);
  }

  freeStatus(deviceStatus);
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(RmaTest, DirectConsumerWriteVisibility) {
  if (!conformanceEnabled())
    GTEST_SKIP() << "Enabled only by the GDR visibility conformance job";
  ASSERT_EQ(nranks, 2) << "Conformance requires exactly two ranks";
  ASSERT_FALSE(requireIpc) << "Conformance requires the RDMA transport";
  ASSERT_FALSE(signalRmaSetupFailed) << signalRmaSkipReason;
  verifyPolicyAndRoute(comm, dataWin, FLAGCX_GDR_WRITE_REQUIRES_FLUSH);

  const ExpectedResult expected =
      expectedResult("FLAGCX_CI_GDR_VISIBILITY_EXPECT_WRITE");
  constexpr uint8_t patternByte = 0x5C;
  int *deviceStatus = nullptr;
  flagcxResult_t statusPreparation = prepareStatus(devHandle, &deviceStatus);
  if (anyRankFailed(statusPreparation)) {
    freeStatus(deviceStatus);
    ADD_FAILURE() << "status-buffer preparation failed on at least one rank";
    return;
  }

  flagcxResult_t setup = flagcxSuccess;
  if (rank == 0) {
    std::vector<uint8_t> pattern(kVisibilityBytes, patternByte);
    setup = devHandle->deviceMemcpy(dataBuff, pattern.data(), pattern.size(),
                                    flagcxMemcpyHostToDevice, nullptr);
  } else {
    setup = devHandle->deviceMemset(dataBuff, 0, kVisibilityBytes,
                                    flagcxMemDevice, nullptr);
  }
  if (setup == flagcxSuccess)
    setup = devHandle->deviceMemset(signalBuff, 0, signalSize, flagcxMemDevice,
                                    nullptr);
  if (setup == flagcxSuccess)
    setup = devHandle->deviceSynchronize();
  if (anyRankFailed(setup)) {
    freeStatus(deviceStatus);
    ADD_FAILURE() << "payload preparation failed on at least one rank";
    return;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxResult_t putResult = flagcxSuccess;
  flagcxResult_t waitResult = flagcxSuccess;
  flagcxResult_t launch = flagcxSuccess;
  if (rank == 0) {
    putResult = flagcxPutSignal(dataBuff, kVisibilityBytes, flagcxChar, 1,
                                dataWin, 0, 0, comm, stream);
    if (putResult == flagcxSuccess)
      putResult = devHandle->streamSynchronize(stream);
  } else {
    flagcxWaitSignalDesc_t desc = {1, 0};
    waitResult = flagcxWaitSignal(1, &desc, comm, stream);

    // WaitSignal and the consumer kernel are adjacent in the same stream. No
    // host or device synchronization may create an implicit visibility edge.
    if (waitResult == flagcxSuccess) {
      launch = flagcxTestLaunchGdrVisibilityConsumer(
          dataBuff, kVisibilityBytes, patternByte, deviceStatus, stream);
      if (launch == flagcxSuccess)
        launch = devHandle->streamSynchronize(stream);
    }
  }

  int putCode = static_cast<int>(putResult);
  int waitCode = static_cast<int>(waitResult);
  int launchCode = static_cast<int>(launch);
  MPI_Bcast(&putCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&waitCode, 1, MPI_INT, 1, MPI_COMM_WORLD);
  MPI_Bcast(&launchCode, 1, MPI_INT, 1, MPI_COMM_WORLD);
  EXPECT_EQ(putCode, static_cast<int>(flagcxSuccess));
  if (expected == ExpectedResult::NotSupported) {
    EXPECT_EQ(waitCode, static_cast<int>(flagcxNotSupported));
  } else if (expected == ExpectedResult::RemoteError) {
    EXPECT_EQ(waitCode, static_cast<int>(flagcxRemoteError));
  } else {
    EXPECT_EQ(waitCode, static_cast<int>(flagcxSuccess));
    EXPECT_EQ(launchCode, static_cast<int>(flagcxSuccess));
    if (rank == 1 && launch == flagcxSuccess)
      verifyConsumerStatus(devHandle, deviceStatus);
  }

  freeStatus(deviceStatus);
  MPI_Barrier(MPI_COMM_WORLD);
}

#endif // FLAGCX_TEST_GDR_VISIBILITY_CONSUMER
