#pragma once

#include "flagcx.h"
#include "flagcx_kernel.h"
#include "flagcx_test.hpp"

// Buffer size for symmetric memory tests (1 MB)
#ifndef SYMMEM_TEST_SIZE
#define SYMMEM_TEST_SIZE (1ULL * 1024 * 1024)
#endif

class SymMemTest : public FlagCXTest {
protected:
  static void SetUpTestSuite();
  static void TearDownTestSuite();

  void SetUp() override;
  void TearDown() override {}

  bool hasHeteroComm() const;

  static flagcxHandlerGroup_t handler;
  static flagcxComm_t comm;
  static flagcxStream_t stream;
  static void *devBuff;
  static void *devBuff2;
  static void *hostBuff;
  static size_t size;
  static size_t count;
};
