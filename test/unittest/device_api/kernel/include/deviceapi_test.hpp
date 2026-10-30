#pragma once

#include "flagcx.h"
#include "flagcx_kernel.h"
#include "flagcx_test.hpp"

#define DEVICEAPI_TEST_SIZE (1ULL * 1024 * 1024)

class DeviceApiTest : public FlagCXTest {
protected:
  static void SetUpTestSuite();
  static void TearDownTestSuite();
  void SetUp() override;
  void TearDown() override {}

  static flagcxDeviceHandle_t devHandle;
  static flagcxComm_t comm;
  static flagcxStream_t stream;
  static void *devBuff;
  static size_t size;
  static flagcxMemAllocator_t memAllocator;
  static flagcxDevComm_t shmemLifetimeComm;
};
