#ifndef FLAGCX_ADAPTOR_DEVICE_UTILS_H_
#define FLAGCX_ADAPTOR_DEVICE_UTILS_H_

// Temporary device utils macros, to be refactored later.
#ifdef USE_NVIDIA_ADAPTOR
#define FLAGCX_HOST_DECORATOR __host__
#define FLAGCX_DEVICE_DECORATOR __device__
#define FLAGCX_GLOBAL_DECORATOR __global__
#define FLAGCX_DEVICE_INLINE_DECORATOR __forceinline__ __device__
#define FLAGCX_DEVICE_CONSTANT_DECORATOR __device__ __constant__
#define FLAGCX_DEVICE_STREAM_PTR cudaStream_t *
#define FLAGCX_DEVICE_THREAD_FENCE __threadfence_system
#define FLAGCX_DEVICE_LAUNCH_KERNEL cudaLaunchKernel
#define FLAGCX_DEVICE_SYNC_THREADS __syncthreads
#include <cuda.h>
#include <cuda_runtime.h>

#if defined(__CUDACC__)
FLAGCX_DEVICE_INLINE_DECORATOR void spinBackoff(int iter) {
  int delay = 1 << (iter < 15 ? iter : 15);
#if __CUDA_ARCH__ >= 700
  __nanosleep(delay);
#else
  uint64_t start = clock64();
  while (clock64() - start < (uint64_t)delay) { /* spin */
  }
#endif
}
#endif // __CUDACC__

#else
#define FLAGCX_HOST_DECORATOR
#define FLAGCX_DEVICE_DECORATOR
#define FLAGCX_GLOBAL_DECORATOR
#define FLAGCX_DEVICE_INLINE_DECORATOR
#define FLAGCX_DEVICE_CONSTANT_DECORATOR
#define FLAGCX_DEVICE_STREAM_PTR
#define FLAGCX_DEVICE_THREAD_FENCE
#define FLAGCX_DEVICE_LAUNCH_KERNEL
#define FLAGCX_DEVICE_NANOSLEEP
#define FLAGCX_DEVICE_SYNC_THREADS
#endif

#endif // FLAGCX_ADAPTOR_DEVICE_UTILS_H_
