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
  SymMemTest() {}

  void SetUp();
  void TearDown();

  // Returns true if heteroComm is available (inter-node / forced hetero mode)
  bool hasHeteroComm() const;

  flagcxHandlerGroup_t handler;
  flagcxStream_t stream;
  void *devBuff;  // GDR-pinned device buffer (flagcxMemAlloc)
  void *devBuff2; // second device buffer for multi-window tests
  void *hostBuff; // host buffer for verification
  size_t size;
  size_t count; // number of floats
};
