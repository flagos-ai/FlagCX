/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Compiler-only reproducer for MCCL's counted warp-span barrier on mp_31.
 * Compile the same kernel with FLAGCX_MUSA_NAMED_BARRIER_REPRO=0 and =1.
 * Launch contract, if executed: one-dimensional blocks of 64 threads.
 ************************************************************************/

#include <musa_runtime.h>

#ifndef FLAGCX_MUSA_NAMED_BARRIER_REPRO
#define FLAGCX_MUSA_NAMED_BARRIER_REPRO 0
#endif

extern "C" __global__ void musaBarrierRepro(unsigned *output) {
  __shared__ unsigned values[64];
  const unsigned thread = threadIdx.x;
  values[thread] = thread;
#if FLAGCX_MUSA_NAMED_BARRIER_REPRO
  // The operation used by mcclCoopWarpSpan(0, 2, 0).
  __barrier_sync_count(1, 64);
#else
  __syncthreads();
#endif
  output[thread] = values[(thread + 32) % 64];
}
