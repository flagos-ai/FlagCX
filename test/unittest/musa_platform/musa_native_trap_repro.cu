/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Compiler-only check of the native trap used by MCCL device primitives.
 * A runtime argument keeps the failure path present during optimization.
 ************************************************************************/

#include <musa_runtime.h>

extern "C" __global__ void musaNativeTrapRepro(unsigned *output, int fail) {
  if (fail)
    __trap();
  output[threadIdx.x] = threadIdx.x;
}
