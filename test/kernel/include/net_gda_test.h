/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 * Diagnostics for the existing inter AlltoAll test's GDA execution path.
 ************************************************************************/
#ifndef FLAGCX_NET_GDA_TEST_H_
#define FLAGCX_NET_GDA_TEST_H_

#include "flagcx_kernel.h"
#include <stdint.h>

struct flagcxNetGdaVisibilityResult {
  uint64_t preReadChecksum;
  uint64_t checkedBytes;
  uint64_t mismatchCount;
  uint64_t firstMismatchOffset;
};

// Same K14 launcher and kernel, with optional Float32 payload diagnostics.
flagcxResult_t launchKernelNetOneSidedAlltoAll(
    flagcxDevMem_t send, flagcxDevMem_t recv, size_t count,
    flagcxDataType_t datatype, flagcxDevComm_t comm, flagcxStream_t stream,
    int round, size_t chunkBytes, flagcxNetGdaVisibilityResult *validation);

#endif
