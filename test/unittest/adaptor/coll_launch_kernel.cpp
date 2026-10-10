/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Verifies the DeviceAdaptor launchKernel slot against real kernels.
 *
 * The slot has no production caller, so it is exercised here directly against
 * kernels compiled from test/kernel/<platform>/device_api.cu and device_ir.cu,
 * the two translation units named by this task's acceptance criteria.
 *
 * Positive cases launch the same kernel twice: once through the existing
 * <<<>>> launcher and once through deviceAdaptor->launchKernel. Both
 * byte-identical results AND the absolute expected values are required --
 * equality alone would also accept two paths that are wrong in the same way.
 * The CoopGroups case deliberately uses an asymmetric 4x256 launch so that a
 * block/grid transposition cannot hide: the kernel derives its tile count from
 * blockDim.x, so a transposition trips CoopTileSpan's count<=0 trap instead of
 * silently producing wrong results.
 *
 * Negative cases require a real error rather than a silent success: invalid
 * arguments must be rejected with flagcxInvalidArgument, and a configuration the
 * runtime refuses must not be reported as flagcxSuccess.
 *
 * Usage: mpirun -np N ./adaptor_launch_kernel_tests
 *   Skips when the slot is NULL (e.g. the NVIDIA adaptor leaves it NULL).
 *   Only built where the launch bridges exist (Iluvatar/CoreX).
 ************************************************************************/

#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "adaptor.h"
#include "device_api.h"
#include "flagcx.h"
#include "flagcx_kernel.h"

// The device_ir.cu leg is disabled by default: test/kernel/<platform>/device_ir.o
// does not compile on Iluvatar yet. nvidia/device_ir.cu:163 assigns an
// address_space(1) pointer (FLAGCX_IR_GLOBAL_RETURN_PTR on CoreX) to a plain
// void *, which the ivcore front end rejects. Until that is resolved the IR
// half of "kernels from both translation units can be launched" cannot be
// verified, and linking device_ir.o is impossible. Flip this to 1 (and add
// device_ir.o back to LAUNCH_KERNEL_OBJS in the Makefile) once fixed.
#ifndef FLAGCX_TEST_LAUNCH_KERNEL_IR_LEG
#define FLAGCX_TEST_LAUNCH_KERNEL_IR_LEG 0
#endif
#if FLAGCX_TEST_LAUNCH_KERNEL_IR_LEG
#include "device_ir.h"
#endif

namespace {

#define ASSERT_FLAGCX_SUCCESS(expr)                                            \
  do {                                                                         \
    flagcxResult_t result_ = (expr);                                           \
    if (result_ != flagcxSuccess) {                                            \
      ADD_FAILURE() << #expr << " returned " << static_cast<int>(result_);     \
      return;                                                                  \
    }                                                                          \
  } while (0)

#define ASSERT_MPI_SUCCESS(expr)                                               \
  do {                                                                         \
    int mpiResult_ = (expr);                                                   \
    if (mpiResult_ != MPI_SUCCESS) {                                           \
      ADD_FAILURE() << #expr << " returned " << mpiResult_;                    \
      return;                                                                  \
    }                                                                          \
  } while (0)

constexpr size_t kWindowBytes = 4u << 20; // symmetric window, 4 MiB
constexpr int kResultsInts = 16;

class LaunchKernelTest : public ::testing::Test {
protected:
  void SetUp() override {
    if (deviceAdaptor == nullptr || deviceAdaptor->launchKernel == nullptr) {
      GTEST_SKIP() << "launchKernel is not implemented on this platform";
    }

    ASSERT_MPI_SUCCESS(MPI_Comm_rank(MPI_COMM_WORLD, &proc_));
    ASSERT_MPI_SUCCESS(MPI_Comm_size(MPI_COMM_WORLD, &totalProcs_));
    ASSERT_FLAGCX_SUCCESS(flagcxDeviceHandleInit(&devHandle_));

    int deviceCount = 0;
    ASSERT_FLAGCX_SUCCESS(devHandle_->getDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
      ADD_FAILURE() << "no visible device";
      return;
    }
    // Same rank -> device binding the Device API drivers use.
    ASSERT_FLAGCX_SUCCESS(devHandle_->setDevice(proc_ % deviceCount));

    if (proc_ == 0) {
      ASSERT_FLAGCX_SUCCESS(flagcxGetUniqueId(&uniqueId_));
    }
    ASSERT_MPI_SUCCESS(MPI_Bcast(&uniqueId_, sizeof(flagcxUniqueId), MPI_BYTE, 0,
                                 MPI_COMM_WORLD));
    ASSERT_MPI_SUCCESS(MPI_Barrier(MPI_COMM_WORLD));

    ASSERT_FLAGCX_SUCCESS(
        flagcxCommInitRank(&comm_, totalProcs_, &uniqueId_, proc_));
    ASSERT_FLAGCX_SUCCESS(devHandle_->streamCreate(&stream_));

    flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.intraBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
    reqs.interBarrierCount = 0;
    reqs.interSignalCount = 0;
    reqs.interCounterCount = 0;
    ASSERT_FLAGCX_SUCCESS(flagcxDevCommCreate(comm_, &reqs, &devComm_));

    ASSERT_FLAGCX_SUCCESS(
        flagcxMemAlloc(&regBuff_, kWindowBytes, memAllocator_));
    ASSERT_FLAGCX_SUCCESS(flagcxCommWindowRegister(
        comm_, regBuff_, kWindowBytes, &win_, FLAGCX_WIN_COLL_SYMMETRIC,
        memAllocator_));
    ASSERT_FLAGCX_SUCCESS(
        flagcxDevMemCreate(comm_, regBuff_, kWindowBytes, win_, &devMem_));

    ASSERT_FLAGCX_SUCCESS(devHandle_->deviceMalloc(
        (void **)&devResults_, kResultsInts * sizeof(int), flagcxMemDevice,
        nullptr));
  }

  void TearDown() override {
    // Reverse order, tolerating a partially completed SetUp.
    if (devHandle_ != nullptr && devResults_ != nullptr) {
      devHandle_->deviceFree(devResults_, flagcxMemDevice, nullptr);
    }
    if (comm_ != nullptr && devMem_ != nullptr) {
      flagcxDevMemDestroy(comm_, devMem_);
    }
    if (comm_ != nullptr && win_ != nullptr) {
      flagcxCommWindowDeregister(comm_, win_, memAllocator_);
    }
    if (regBuff_ != nullptr) {
      flagcxMemFree(regBuff_, memAllocator_);
    }
    if (comm_ != nullptr && devComm_ != nullptr) {
      flagcxDevCommDestroy(comm_, devComm_);
    }
    if (devHandle_ != nullptr && stream_ != nullptr) {
      devHandle_->streamDestroy(stream_);
    }
    if (comm_ != nullptr) {
      flagcxCommDestroy(comm_);
    }
    if (devHandle_ != nullptr) {
      flagcxDeviceHandleFree(devHandle_);
    }
  }

  // Reference path: the existing <<<>>> launchers (same configuration the
  // Device API drivers use), copied back into hostRef.
  void runReferenceCommQueries(int *hostRef) {
    ASSERT_FLAGCX_SUCCESS(devHandle_->deviceMemset(
        devResults_, 0, kResultsInts * sizeof(int), flagcxMemDevice, stream_));
    ASSERT_FLAGCX_SUCCESS(
        launchKernelCommQueries(devMem_, devComm_, devResults_, stream_));
    ASSERT_FLAGCX_SUCCESS(devHandle_->streamSynchronize(stream_));
    ASSERT_FLAGCX_SUCCESS(devHandle_->deviceMemcpy(
        hostRef, devResults_, kResultsInts * sizeof(int),
        flagcxMemcpyDeviceToHost, stream_));
  }

  flagcxDeviceHandle_t devHandle_ = nullptr;
  flagcxComm_t comm_ = nullptr;
  flagcxDevComm_t devComm_ = nullptr;
  flagcxDevMem_t devMem_ = nullptr;
  flagcxWindow_t win_ = nullptr;
  void *regBuff_ = nullptr;
  flagcxMemAllocator_t memAllocator_ = flagcxMemCCL;
  flagcxStream_t stream_ = nullptr;
  int *devResults_ = nullptr;
  flagcxUniqueId uniqueId_{};
  int proc_ = 0;
  int totalProcs_ = 1;
};

// K9 CommQueries through the slot. Its parameters are passed by value
// (flagcxDevMem / flagcxDevComm), so this covers the by-value aggregate form of
// the kernelParams contract.
TEST_F(LaunchKernelTest, CommQueriesMatchesLegacyLauncher) {
  int ref[kResultsInts] = {};
  int got[kResultsInts] = {};
  runReferenceCommQueries(ref);

  ASSERT_FLAGCX_SUCCESS(devHandle_->deviceMemset(
      devResults_, 0, kResultsInts * sizeof(int), flagcxMemDevice, stream_));
  ASSERT_FLAGCX_SUCCESS(flagcxTestLaunchCommQueriesViaAdaptor(
      deviceAdaptor->launchKernel, devMem_, devComm_, devResults_,
      /*blockX=*/32, /*gridX=*/1, stream_));
  ASSERT_FLAGCX_SUCCESS(devHandle_->streamSynchronize(stream_));
  ASSERT_FLAGCX_SUCCESS(devHandle_->deviceMemcpy(
      got, devResults_, kResultsInts * sizeof(int), flagcxMemcpyDeviceToHost,
      stream_));

  EXPECT_EQ(std::memcmp(ref, got, sizeof(ref)), 0)
      << "slot path differs from the <<<>>> path";
  EXPECT_EQ(got[0], 1);           // hasWindow
  EXPECT_EQ(got[1], proc_);       // intraRank
  EXPECT_EQ(got[2], totalProcs_); // intraSize
  EXPECT_EQ(got[3], proc_);       // rank
  EXPECT_EQ(got[4], totalProcs_); // size
}

// K10 CoopGroups through the slot, with the same asymmetric 4x256 launch used by
// the existing launcher. This is the only case that can expose a block/grid
// transposition, because the kernel derives its tile count from blockDim.x.
TEST_F(LaunchKernelTest, CoopGroupsMatchesLegacyLauncher) {
  int ref[kResultsInts] = {};
  int got[kResultsInts] = {};

  ASSERT_FLAGCX_SUCCESS(devHandle_->deviceMemset(
      devResults_, 0, kResultsInts * sizeof(int), flagcxMemDevice, stream_));
  ASSERT_FLAGCX_SUCCESS(launchKernelCoopGroups(devResults_, stream_));
  ASSERT_FLAGCX_SUCCESS(devHandle_->streamSynchronize(stream_));
  ASSERT_FLAGCX_SUCCESS(devHandle_->deviceMemcpy(
      ref, devResults_, kResultsInts * sizeof(int), flagcxMemcpyDeviceToHost,
      stream_));

  ASSERT_FLAGCX_SUCCESS(devHandle_->deviceMemset(
      devResults_, 0, kResultsInts * sizeof(int), flagcxMemDevice, stream_));
  int *results = devResults_; // parameters are plain pointers here, so the host
  void *args[] = {&results};  // test marshals the args itself
  ASSERT_FLAGCX_SUCCESS(deviceAdaptor->launchKernel(
      flagcxTestKernelCoopGroupsPtr(), /*blockX=*/256, 1, 1, /*gridX=*/4, 1, 1,
      args, 0, stream_, nullptr));
  ASSERT_FLAGCX_SUCCESS(devHandle_->streamSynchronize(stream_));
  ASSERT_FLAGCX_SUCCESS(devHandle_->deviceMemcpy(
      got, devResults_, kResultsInts * sizeof(int), flagcxMemcpyDeviceToHost,
      stream_));

  EXPECT_EQ(std::memcmp(ref, got, sizeof(ref)), 0)
      << "slot path differs from the <<<>>> path";
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(got[i], 1) << "coop-group self check " << i << " failed";
  }
}

// S1 CommQueries from device_ir.cu through the slot. Its parameter is a raw
// devComm device pointer.
#if FLAGCX_TEST_LAUNCH_KERNEL_IR_LEG
TEST_F(LaunchKernelTest, IrCommQueriesMatchesLegacyLauncher) {
  void *devCommPtr = nullptr;
  ASSERT_FLAGCX_SUCCESS(flagcxDevCommGetDevicePtr(devComm_, &devCommPtr));

  int ref[4] = {};
  int got[4] = {};

  ASSERT_FLAGCX_SUCCESS(devHandle_->deviceMemset(
      devResults_, 0, 4 * sizeof(int), flagcxMemDevice, stream_));
  launchKernelCommQueriesS(devCommPtr, devResults_, stream_);
  ASSERT_FLAGCX_SUCCESS(devHandle_->streamSynchronize(stream_));
  ASSERT_FLAGCX_SUCCESS(devHandle_->deviceMemcpy(
      ref, devResults_, 4 * sizeof(int), flagcxMemcpyDeviceToHost, stream_));

  ASSERT_FLAGCX_SUCCESS(devHandle_->deviceMemset(
      devResults_, 0, 4 * sizeof(int), flagcxMemDevice, stream_));
  int *results = devResults_;
  void *args[] = {&devCommPtr, &results};
  ASSERT_FLAGCX_SUCCESS(deviceAdaptor->launchKernel(
      flagcxTestKernelCommQueriesSPtr(), 1, 1, 1, 1, 1, 1, args, 0, stream_,
      nullptr));
  ASSERT_FLAGCX_SUCCESS(devHandle_->streamSynchronize(stream_));
  ASSERT_FLAGCX_SUCCESS(devHandle_->deviceMemcpy(
      got, devResults_, 4 * sizeof(int), flagcxMemcpyDeviceToHost, stream_));

  EXPECT_EQ(std::memcmp(ref, got, sizeof(ref)), 0)
      << "slot path differs from the <<<>>> path";
  EXPECT_EQ(got[0], proc_);       // rank
  EXPECT_EQ(got[1], totalProcs_); // size
  EXPECT_EQ(got[2], proc_);       // intraRank
  EXPECT_EQ(got[3], totalProcs_); // intraSize
}
#endif // FLAGCX_TEST_LAUNCH_KERNEL_IR_LEG

// Negative cases. Kept at the end of the file: a rejected launch can leave the
// sticky error state set, and getLastError may be NULL on this adapter.
TEST_F(LaunchKernelTest, RejectsInvalidArguments) {
  int *results = devResults_;
  void *args[] = {&results};
  void *func = flagcxTestKernelCoopGroupsPtr();
  ASSERT_NE(func, nullptr);

  EXPECT_EQ(deviceAdaptor->launchKernel(nullptr, 256, 1, 1, 4, 1, 1, args, 0,
                                        stream_, nullptr),
            flagcxInvalidArgument);
  EXPECT_EQ(deviceAdaptor->launchKernel(func, 256, 1, 1, 4, 1, 1, nullptr, 0,
                                        stream_, nullptr),
            flagcxInvalidArgument);
  EXPECT_EQ(deviceAdaptor->launchKernel(func, 0, 1, 1, 4, 1, 1, args, 0,
                                        stream_, nullptr),
            flagcxInvalidArgument);
  EXPECT_EQ(deviceAdaptor->launchKernel(func, 256, 1, 1, 0, 1, 1, args, 0,
                                        stream_, nullptr),
            flagcxInvalidArgument);
}

TEST_F(LaunchKernelTest, ReportsRuntimeRejectionInsteadOfSuccess) {
  int *results = devResults_;
  void *args[] = {&results};
  void *func = flagcxTestKernelCoopGroupsPtr();

  // A block of 2^20 threads exceeds any supported limit, so the launch must not
  // be reported as successful.
  flagcxResult_t res = deviceAdaptor->launchKernel(func, 1u << 20, 1, 1, 1, 1, 1,
                                                   args, 0, stream_, nullptr);
  if (res == flagcxSuccess) {
    // Some runtimes validate asynchronously; then the stream must surface it.
    EXPECT_NE(devHandle_->streamSynchronize(stream_), flagcxSuccess)
        << "oversized block was neither rejected at launch nor failed on sync";
  } else {
    EXPECT_EQ(res, flagcxUnhandledDeviceError);
  }

  // Best-effort cleanup of the sticky error so it cannot leak into later tests.
  if (deviceAdaptor->getLastError != nullptr) {
    (void)deviceAdaptor->getLastError();
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
