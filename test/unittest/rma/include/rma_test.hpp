#pragma once

#include "flagcx.h"
#include "flagcx_kernel.h"
#include "flagcx_test.hpp"

// Buffer size for RMA tests (1 MB)
#ifndef RMA_TEST_SIZE
#define RMA_TEST_SIZE (1ULL * 1024 * 1024)
#endif

class RmaTest : public FlagCXTest {
protected:
  static void SetUpTestSuite();
  static void TearDownTestSuite();

  void SetUp() override;
  void TearDown() override {}

  bool hasHeteroComm() const;

  static flagcxDeviceHandle_t devHandle;
  static flagcxComm_t comm;
  static flagcxStream_t stream;
  // Data window buffer (registered for RMA)
  static void *dataBuff;
  // Signal buffer (registered for one-sided signals)
  static void *signalBuff;
  static flagcxWindow_t dataWin;
  static size_t size;
  static size_t signalSize;
  static bool oneSidedAvailable;
  static const char *oneSidedSkipReason;
};
