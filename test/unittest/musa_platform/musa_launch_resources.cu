/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 * SDK-only comparison of kernel resource metadata. No FlagCX headers.
 * Queries attributes only: no kernel, especially no trap, is executed.
 ************************************************************************/

#include <musa_runtime.h>
#include <cstdio>
#include <cstdlib>

constexpr int kThreads = 64;

extern "C" __global__ __launch_bounds__(kThreads)
void resourcePlain(unsigned *output, int) {
  output[threadIdx.x] = threadIdx.x;
}

extern "C" __global__ __launch_bounds__(kThreads)
void resourceWarp(unsigned *output, int) {
  __shared__ unsigned values[kThreads];
  const unsigned thread = threadIdx.x;
  values[thread] = thread;
  __syncwarp(0xffffffffu);
  output[thread] = values[thread ^ 1u];
}

extern "C" __global__ __launch_bounds__(kThreads)
void resourceBlock(unsigned *output, int) {
  __shared__ unsigned values[kThreads];
  const unsigned thread = threadIdx.x;
  values[thread] = thread;
  __syncthreads();
  output[thread] = values[(thread + 32u) % kThreads];
}

extern "C" __global__ __launch_bounds__(kThreads)
void resourceTrap(unsigned *output, int fail) {
  if (fail)
    __trap();
  output[threadIdx.x] = threadIdx.x;
}

extern "C" __global__ __launch_bounds__(kThreads)
void resourceGuardedWarp(unsigned *output, int fail) {
  if (fail)
    __trap();
  __shared__ unsigned values[kThreads];
  const unsigned thread = threadIdx.x;
  values[thread] = thread;
  __syncwarp(0xffffffffu);
  output[thread] = values[thread ^ 1u];
}

static int report(const char *name, const void *kernel) {
  musaFuncAttributes attr{};
  const musaError_t error = musaFuncGetAttributes(&attr, kernel);
  if (error != musaSuccess) {
    std::fprintf(stderr, "%s attribute query failed: %s\n", name,
                 musaGetErrorString(error));
    return 1;
  }
  std::printf("%s: kernel=%p maxThreadsPerBlock=%d numRegs=%d "
              "sharedBytes=%zu localBytes=%zu\n",
              name, const_cast<void *>(kernel), attr.maxThreadsPerBlock,
              attr.numRegs, static_cast<size_t>(attr.sharedSizeBytes),
              static_cast<size_t>(attr.localSizeBytes));
  return 0;
}

int main() {
  int failures = 0;
  failures += report("plain", reinterpret_cast<const void *>(resourcePlain));
  failures += report("warp", reinterpret_cast<const void *>(resourceWarp));
  failures += report("block", reinterpret_cast<const void *>(resourceBlock));
  failures += report("trap", reinterpret_cast<const void *>(resourceTrap));
  failures += report("guarded-warp",
                     reinterpret_cast<const void *>(resourceGuardedWarp));
  std::puts("SDK resource queries only; no kernels launched and no GPU correctness verdict.");
  return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
