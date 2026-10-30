// MPI test for IntraAllReduce via Device Pointer pipeline.
// Full end-to-end test: DevComm + DevMem + DevicePtr + IntraAllReduce.
// Requires MPI + GPUs.

#include "device_api.h"
#include "deviceapi_test.hpp"
#include <cstring>

// ---------------------------------------------------------------------------
// IntraAllReduce via DevicePtr — full pipeline test
// ---------------------------------------------------------------------------

TEST_F(DeviceApiTest, IntraAllReduceViaDevicePtr) {
  // Allocate buffer with flagcxMemAlloc for symmetric window
  void *regBuff = nullptr;
  ASSERT_EQ(flagcxMemAlloc(&regBuff, size, memAllocator), flagcxSuccess);

  // Register symmetric window
  flagcxWindow_t win = nullptr;
  ASSERT_EQ(flagcxCommWindowRegister(comm, regBuff, size, &win,
                                     FLAGCX_WIN_COLL_SYMMETRIC, memAllocator),
            flagcxSuccess);

  MPI_Barrier(MPI_COMM_WORLD);

  // Create DevComm
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.intraBarrierCount = FLAGCX_DEVICE_CTA_COUNT;

  flagcxDevComm_t devComm = nullptr;
  ASSERT_EQ(flagcxDevCommCreate(comm, &reqs, &devComm), flagcxSuccess);

  // Create DevMem
  flagcxDevMem_t devMem = nullptr;
  ASSERT_EQ(flagcxDevMemCreate(comm, regBuff, size, win, &devMem),
            flagcxSuccess);

  MPI_Barrier(MPI_COMM_WORLD);

  // Get device pointers (not used in this test, but verifies they work)
  void *devCommPtr = nullptr;
  void *devMemPtr = nullptr;
  EXPECT_EQ(flagcxDevCommGetDevicePtr(devComm, &devCommPtr), flagcxSuccess);
  EXPECT_EQ(flagcxDevMemGetDevicePtr(devMem, &devMemPtr), flagcxSuccess);

  // Initialize: each rank fills buffer with (rank + 1)
  int worldRank, worldSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

  size_t floatCount = size / sizeof(float);
  float *hostInit = new float[floatCount];
  for (size_t i = 0; i < floatCount; i++) {
    hostInit[i] = (float)(worldRank + 1);
  }

  ASSERT_EQ(devHandle->deviceMemcpy(regBuff, hostInit, size,
                                    flagcxMemcpyHostToDevice, nullptr),
            flagcxSuccess);
  delete[] hostInit;

  MPI_Barrier(MPI_COMM_WORLD);

  // Run IntraAllReduce
  EXPECT_EQ(launchKernelIntraAllReduce(devMem, floatCount, flagcxFloat, devComm,
                                       stream),
            flagcxSuccess);
  ASSERT_EQ(devHandle->streamSynchronize(stream), flagcxSuccess);

  MPI_Barrier(MPI_COMM_WORLD);

  // Verify: expected value = sum(1..nRanks) = nRanks*(nRanks+1)/2
  float expected = (float)(worldSize * (worldSize + 1)) / 2.0f;
  float *hostResult = new float[floatCount];
  ASSERT_EQ(devHandle->deviceMemcpy(hostResult, regBuff, size,
                                    flagcxMemcpyDeviceToHost, nullptr),
            flagcxSuccess);

  for (size_t i = 0; i < floatCount; i++) {
    EXPECT_NEAR(hostResult[i], expected, 1e-3f)
        << "Mismatch at index " << i << " on rank " << worldRank;
  }
  delete[] hostResult;

  MPI_Barrier(MPI_COMM_WORLD);

  // Cleanup
  flagcxDevMemFreeDevicePtr(devMem);
  flagcxDevCommFreeDevicePtr(devComm);
  flagcxDevMemDestroy(comm, devMem);
  flagcxDevCommDestroy(comm, devComm);
  flagcxCommWindowDeregister(comm, win, memAllocator);
  flagcxMemFree(regBuff, memAllocator);
}
