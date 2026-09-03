// MPI tests for Device Pointer API lifecycle.
// Tests DevCommGetDevicePtr, DevMemGetDevicePtr, and their Free counterparts.
// Requires MPI + GPUs.

#include "deviceapi_test.hpp"

// ---------------------------------------------------------------------------
// DevCommGetDevicePtr — basic functionality
// ---------------------------------------------------------------------------

TEST_F(DeviceApiTest, DevCommGetDevicePtrBasic) {
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.intraBarrierCount = 1;

  flagcxDevComm_t devComm = nullptr;
  ASSERT_EQ(flagcxDevCommCreate(comm, &reqs, &devComm), flagcxSuccess);
  ASSERT_NE(devComm, nullptr);

  MPI_Barrier(MPI_COMM_WORLD);

  void *devPtr = nullptr;
  EXPECT_EQ(flagcxDevCommGetDevicePtr(devComm, &devPtr), flagcxSuccess);
  EXPECT_NE(devPtr, nullptr);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxDevCommFreeDevicePtr(devComm);
  flagcxDevCommDestroy(comm, devComm);
}

// ---------------------------------------------------------------------------
// DevCommGetDevicePtr — cached (calling twice returns same pointer)
// ---------------------------------------------------------------------------

TEST_F(DeviceApiTest, DevCommGetDevicePtrCached) {
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.intraBarrierCount = 1;

  flagcxDevComm_t devComm = nullptr;
  ASSERT_EQ(flagcxDevCommCreate(comm, &reqs, &devComm), flagcxSuccess);

  MPI_Barrier(MPI_COMM_WORLD);

  void *devPtr1 = nullptr;
  EXPECT_EQ(flagcxDevCommGetDevicePtr(devComm, &devPtr1), flagcxSuccess);
  EXPECT_NE(devPtr1, nullptr);

  void *devPtr2 = nullptr;
  EXPECT_EQ(flagcxDevCommGetDevicePtr(devComm, &devPtr2), flagcxSuccess);
  EXPECT_EQ(devPtr1, devPtr2); // Should return cached pointer

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxDevCommFreeDevicePtr(devComm);
  flagcxDevCommDestroy(comm, devComm);
}

// ---------------------------------------------------------------------------
// DevCommFreeDevicePtr + GetDevicePtr again — new pointer
// ---------------------------------------------------------------------------

TEST_F(DeviceApiTest, DevCommFreeAndReget) {
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.intraBarrierCount = 1;

  flagcxDevComm_t devComm = nullptr;
  ASSERT_EQ(flagcxDevCommCreate(comm, &reqs, &devComm), flagcxSuccess);

  MPI_Barrier(MPI_COMM_WORLD);

  void *devPtr1 = nullptr;
  EXPECT_EQ(flagcxDevCommGetDevicePtr(devComm, &devPtr1), flagcxSuccess);
  EXPECT_NE(devPtr1, nullptr);

  EXPECT_EQ(flagcxDevCommFreeDevicePtr(devComm), flagcxSuccess);

  void *devPtr2 = nullptr;
  EXPECT_EQ(flagcxDevCommGetDevicePtr(devComm, &devPtr2), flagcxSuccess);
  EXPECT_NE(devPtr2, nullptr);
  // devPtr2 may or may not equal devPtr1 (implementation detail)

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxDevCommFreeDevicePtr(devComm);
  flagcxDevCommDestroy(comm, devComm);
}

// ---------------------------------------------------------------------------
// DevMemGetDevicePtr — IPC mode (no window)
// ---------------------------------------------------------------------------

TEST_F(DeviceApiTest, DevMemGetDevicePtrIPC) {
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxDevMem_t devMem = nullptr;
  ASSERT_EQ(flagcxDevMemCreate(comm, devBuff, size, nullptr, &devMem),
            flagcxSuccess);
  ASSERT_NE(devMem, nullptr);

  MPI_Barrier(MPI_COMM_WORLD);

  void *devPtr = nullptr;
  EXPECT_EQ(flagcxDevMemGetDevicePtr(devMem, &devPtr), flagcxSuccess);
  EXPECT_NE(devPtr, nullptr);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxDevMemFreeDevicePtr(devMem);
  flagcxDevMemDestroy(comm, devMem);
}

// ---------------------------------------------------------------------------
// DevMemGetDevicePtr — Window mode (symmetric window)
// ---------------------------------------------------------------------------

TEST_F(DeviceApiTest, DevMemGetDevicePtrWindow) {
  flagcxWindow_t win = nullptr;
  ASSERT_EQ(flagcxCommWindowRegister(comm, devBuff, size, &win,
                                     FLAGCX_WIN_COLL_SYMMETRIC, memAllocator),
            flagcxSuccess);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxDevMem_t devMem = nullptr;
  ASSERT_EQ(flagcxDevMemCreate(comm, devBuff, size, win, &devMem),
            flagcxSuccess);
  ASSERT_NE(devMem, nullptr);

  MPI_Barrier(MPI_COMM_WORLD);

  void *devPtr = nullptr;
  EXPECT_EQ(flagcxDevMemGetDevicePtr(devMem, &devPtr), flagcxSuccess);
  EXPECT_NE(devPtr, nullptr);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxDevMemFreeDevicePtr(devMem);
  flagcxDevMemDestroy(comm, devMem);
  flagcxCommWindowDeregister(comm, win, memAllocator);
}

// ---------------------------------------------------------------------------
// DevMemGetDevicePtr — cached (calling twice returns same pointer)
// ---------------------------------------------------------------------------

TEST_F(DeviceApiTest, DevMemGetDevicePtrCached) {
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxDevMem_t devMem = nullptr;
  ASSERT_EQ(flagcxDevMemCreate(comm, devBuff, size, nullptr, &devMem),
            flagcxSuccess);

  MPI_Barrier(MPI_COMM_WORLD);

  void *devPtr1 = nullptr;
  EXPECT_EQ(flagcxDevMemGetDevicePtr(devMem, &devPtr1), flagcxSuccess);
  EXPECT_NE(devPtr1, nullptr);

  void *devPtr2 = nullptr;
  EXPECT_EQ(flagcxDevMemGetDevicePtr(devMem, &devPtr2), flagcxSuccess);
  EXPECT_EQ(devPtr1, devPtr2); // Should return cached pointer

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxDevMemFreeDevicePtr(devMem);
  flagcxDevMemDestroy(comm, devMem);
}

// ---------------------------------------------------------------------------
// DevMemFreeDevicePtr — basic functionality
// ---------------------------------------------------------------------------

TEST_F(DeviceApiTest, DevMemFreeDevicePtrBasic) {
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxDevMem_t devMem = nullptr;
  ASSERT_EQ(flagcxDevMemCreate(comm, devBuff, size, nullptr, &devMem),
            flagcxSuccess);

  MPI_Barrier(MPI_COMM_WORLD);

  void *devPtr = nullptr;
  EXPECT_EQ(flagcxDevMemGetDevicePtr(devMem, &devPtr), flagcxSuccess);

  EXPECT_EQ(flagcxDevMemFreeDevicePtr(devMem), flagcxSuccess);

  MPI_Barrier(MPI_COMM_WORLD);

  flagcxDevMemDestroy(comm, devMem);
}

// ---------------------------------------------------------------------------
// DevCommDestroy implicitly frees cached device pointer
// ---------------------------------------------------------------------------

TEST_F(DeviceApiTest, DevCommDestroyFreesPtr) {
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.intraBarrierCount = 1;

  flagcxDevComm_t devComm = nullptr;
  ASSERT_EQ(flagcxDevCommCreate(comm, &reqs, &devComm), flagcxSuccess);

  MPI_Barrier(MPI_COMM_WORLD);

  void *devPtr = nullptr;
  EXPECT_EQ(flagcxDevCommGetDevicePtr(devComm, &devPtr), flagcxSuccess);

  MPI_Barrier(MPI_COMM_WORLD);

  // Destroy without explicit FreeDevicePtr — should not leak or double-free
  EXPECT_EQ(flagcxDevCommDestroy(comm, devComm), flagcxSuccess);
}

// ---------------------------------------------------------------------------
// DevMemDestroy implicitly frees cached device pointer
// ---------------------------------------------------------------------------

TEST_F(DeviceApiTest, DevMemDestroyFreesPtr) {
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxDevMem_t devMem = nullptr;
  ASSERT_EQ(flagcxDevMemCreate(comm, devBuff, size, nullptr, &devMem),
            flagcxSuccess);

  MPI_Barrier(MPI_COMM_WORLD);

  void *devPtr = nullptr;
  EXPECT_EQ(flagcxDevMemGetDevicePtr(devMem, &devPtr), flagcxSuccess);

  MPI_Barrier(MPI_COMM_WORLD);

  // Destroy without explicit FreeDevicePtr — should not leak or double-free
  EXPECT_EQ(flagcxDevMemDestroy(comm, devMem), flagcxSuccess);
}
