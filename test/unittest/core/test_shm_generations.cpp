#include "ipc_lifecycle_mock.hpp"
#include <cstring>
#include <unistd.h>

TEST_F(IpcLifecycleMock,
       RetiredHostShmGenerationsKeepBarrierAndSignalContents) {
  std::string previous = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.intraBarrierCount = reqs.interSignalCount = 1;
  EXPECT_EXIT(
      {
        auto check = [](bool ok) {
          if (!ok)
            _exit(91);
        };
        setenv("FLAGCX_SIGNAL_HOST_ENABLE", "1", 1);
        comm->commType = flagcxCommunicatorHybrid;
        comm->magic = (uint64_t)getpid();
        comm->localRankToRank = (int *)calloc(1, sizeof(int));
        device.hostRegister = [](void *, size_t) { return flagcxSuccess; };
        device.hostUnregister = [](void *) { return flagcxSuccess; };
        device.hostGetDevicePointer = [](void **out, void *ptr) {
          *out = ptr;
          return flagcxSuccess;
        };
        device.deviceMemcpy = [](void *dst, void *src, size_t bytes,
                                 flagcxMemcpyType_t, flagcxStream_t, void *) {
          memcpy(dst, src, bytes);
          return flagcxSuccess;
        };
        device.deviceMemset = [](void *ptr, int value, size_t bytes,
                                 flagcxMemType_t, flagcxStream_t) {
          memset(ptr, value, bytes);
          return flagcxSuccess;
        };
        flagcxDevComm_t first = nullptr;
        flagcxDevComm_t second = nullptr;
        check(flagcxDevCommCreate(comm, &reqs, &first) == flagcxSuccess);
        check(first->shmGeneration == 1);
        auto *barrier = (uint64_t *)first->localBarrierShmPtr;
        auto *signal = (uint64_t *)first->signalHostBuffer;
        check(barrier != nullptr && signal != nullptr);
        *barrier = 0x1234;
        *signal = 0x5678;
        check(flagcxDevCommDestroy(comm, first) == flagcxSuccess);
        check(comm->retiredDevComms == first && comm->devCommCount == 0);
        check(flagcxDevCommCreate(comm, &reqs, &second) == flagcxSuccess);
        check(second->shmGeneration == 2 && second->devCommId == 1);
        check(*barrier == 0x1234 && *signal == 0x5678);
        check(*(uint64_t *)second->localBarrierShmPtr == 0);
        check(*(uint64_t *)second->signalHostBuffer == 0);
        check(flagcxDevCommDestroy(comm, second) == flagcxSuccess);
        device.hostGetDevicePointer = [](void **, void *) {
          return flagcxRemoteError;
        };
        flagcxDevComm_t failed = nullptr;
        check(flagcxDevCommCreate(comm, &reqs, &failed) == flagcxRemoteError);
        check(failed == nullptr && comm->shmGeneration == 3);
        device.hostGetDevicePointer = [](void **out, void *ptr) {
          *out = ptr;
          return flagcxSuccess;
        };
        flagcxDevComm_t replacement = nullptr;
        check(flagcxDevCommCreate(comm, &reqs, &replacement) == flagcxSuccess);
        check(replacement->shmGeneration == 4);
        check(*barrier == 0x1234 && *signal == 0x5678);
        check(flagcxDevCommDestroy(comm, replacement) == flagcxSuccess);
        check(flagcxCommFinalize(comm) == flagcxSuccess);
        check(flagcxCommDestroy(comm) == flagcxSuccess);
        _exit(0);
      },
      ::testing::ExitedWithCode(0), "");
  ::testing::FLAGS_gtest_death_test_style = previous;
}

TEST_F(IpcLifecycleMock, ExhaustedHostShmGenerationNeverWrapsOrPublishesOwner) {
  std::string previous = ::testing::FLAGS_gtest_death_test_style;
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.intraBarrierCount = 1;
  EXPECT_EXIT(
      {
        setenv("FLAGCX_SIGNAL_HOST_ENABLE", "1", 1);
        comm->commType = flagcxCommunicatorHybrid;
        comm->shmGeneration = UINT64_MAX;
        flagcxDevComm_t view = nullptr;
        for (int i = 0; i < 2; ++i) {
          if (flagcxDevCommCreate(comm, &reqs, &view) != flagcxInvalidUsage ||
              view != nullptr || comm->shmGeneration != UINT64_MAX)
            _exit(92);
        }
        _exit(0);
      },
      ::testing::ExitedWithCode(0), "");
  ::testing::FLAGS_gtest_death_test_style = previous;
}
