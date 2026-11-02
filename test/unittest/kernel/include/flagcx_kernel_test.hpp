#pragma once

#include "flagcx.h"
#include "flagcx_kernel.h"
#include "flagcx_test.hpp"

class FlagCXKernelTest : public FlagCXTest {
protected:
  FlagCXKernelTest() {}

  void SetUp();

  void TearDown();

  void Run();

  flagcxDeviceHandle_t devHandle = nullptr;
  flagcxComm_t comm = nullptr;
  flagcxStream_t stream;
  void *sendbuff;
  void *recvbuff;
  void *hostsendbuff;
  void *hostrecvbuff;
  size_t size;
  size_t count;
};
