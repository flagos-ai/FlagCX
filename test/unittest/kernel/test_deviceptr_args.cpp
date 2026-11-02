// Unit tests for Device Pointer API argument validation.
// No MPI or GPU required — runs locally.
// Links against libflagcx.

#include <gtest/gtest.h>

#include "flagcx.h"
#include "flagcx_kernel.h"

// ---------------------------------------------------------------------------
// DevCommGetDevicePtr — NULL argument tests
// ---------------------------------------------------------------------------

TEST(DevicePtrArgs, DevCommGetDevicePtrNullDevComm) {
  void *devPtr = nullptr;
  flagcxResult_t res = flagcxDevCommGetDevicePtr(nullptr, &devPtr);
  EXPECT_EQ(res, flagcxInvalidArgument);
}

TEST(DevicePtrArgs, DevCommGetDevicePtrNullOutput) {
  // We can't create a real devComm without MPI, but we can test NULL output
  // The function should check output pointer before dereferencing devComm
  flagcxDevComm_t fakeDevComm = (flagcxDevComm_t)0x1; // non-null fake pointer
  flagcxResult_t res = flagcxDevCommGetDevicePtr(fakeDevComm, nullptr);
  EXPECT_EQ(res, flagcxInvalidArgument);
}

// ---------------------------------------------------------------------------
// DevCommFreeDevicePtr — NULL safety
// ---------------------------------------------------------------------------

TEST(DevicePtrArgs, DevCommFreeDevicePtrNull) {
  flagcxResult_t res = flagcxDevCommFreeDevicePtr(nullptr);
  EXPECT_EQ(res, flagcxInvalidArgument);
}

// ---------------------------------------------------------------------------
// DevMemGetDevicePtr — NULL argument tests
// ---------------------------------------------------------------------------

TEST(DevicePtrArgs, DevMemGetDevicePtrNullDevMem) {
  void *devPtr = nullptr;
  flagcxResult_t res = flagcxDevMemGetDevicePtr(nullptr, &devPtr);
  EXPECT_EQ(res, flagcxInvalidArgument);
}

TEST(DevicePtrArgs, DevMemGetDevicePtrNullOutput) {
  flagcxDevMem_t fakeDevMem = (flagcxDevMem_t)0x1; // non-null fake pointer
  flagcxResult_t res = flagcxDevMemGetDevicePtr(fakeDevMem, nullptr);
  EXPECT_EQ(res, flagcxInvalidArgument);
}

// ---------------------------------------------------------------------------
// DevMemFreeDevicePtr — NULL safety
// ---------------------------------------------------------------------------

TEST(DevicePtrArgs, DevMemFreeDevicePtrNull) {
  flagcxResult_t res = flagcxDevMemFreeDevicePtr(nullptr);
  EXPECT_EQ(res, flagcxInvalidArgument);
}
