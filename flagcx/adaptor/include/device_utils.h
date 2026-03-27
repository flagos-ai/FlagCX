/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_ADAPTOR_DEVICE_UTILS_H_
#define FLAGCX_ADAPTOR_DEVICE_UTILS_H_

// Device compiler detection — defined when any GPU device compiler is active.
// Extend with __ASCEND_CC__ etc. as new platforms are added.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define FLAGCX_DEVICE_COMPILE 1
#endif

// Suppress unused-variable warnings for static arrays in headers
#define FLAGCX_MAYBE_UNUSED __attribute__((unused))

#ifdef USE_NVIDIA_ADAPTOR
#include <cuda.h>
#include <cuda_runtime.h>

#if defined(__CUDACC__)
// Compiling with nvcc — full CUDA qualifiers
#define FLAGCX_HOST_DECORATOR __host__
#define FLAGCX_DEVICE_DECORATOR __device__
#define FLAGCX_GLOBAL_DECORATOR __global__
#define FLAGCX_DEVICE_INLINE_DECORATOR __forceinline__ __device__
#define FLAGCX_HOST_DEVICE_INLINE __forceinline__ __host__ __device__
#define FLAGCX_DEVICE_CONSTANT_DECORATOR __device__ __constant__
#define FLAGCX_DEVICE_THREAD_FENCE __threadfence_system
#define FLAGCX_DEVICE_SYNC_THREADS __syncthreads
#define FLAGCX_THREAD_IDX_X threadIdx.x
#define FLAGCX_BLOCK_IDX_X blockIdx.x
#define FLAGCX_BLOCK_DIM_X blockDim.x
#define FLAGCX_GRID_DIM_X gridDim.x

// SIMT lockstep width (32 lanes on NVIDIA/CUDA)
#define FLAGCX_SIMT_WIDTH 32
#define FLAGCX_SHARED __shared__
#else
// Host compiler (g++/clang++) on NVIDIA platform — no CUDA qualifiers
#define FLAGCX_HOST_DECORATOR
#define FLAGCX_DEVICE_DECORATOR
#define FLAGCX_GLOBAL_DECORATOR
#define FLAGCX_DEVICE_INLINE_DECORATOR inline
#define FLAGCX_HOST_DEVICE_INLINE inline
#define FLAGCX_DEVICE_CONSTANT_DECORATOR
#define FLAGCX_DEVICE_THREAD_FENCE() ((void)0)
#define FLAGCX_DEVICE_SYNC_THREADS() ((void)0)
#define FLAGCX_THREAD_IDX_X 0
#define FLAGCX_BLOCK_IDX_X 0
#define FLAGCX_BLOCK_DIM_X 1
#define FLAGCX_GRID_DIM_X 1

// SIMT width (same as device, for template instantiation)
#define FLAGCX_SIMT_WIDTH 32
#define FLAGCX_SHARED static
#endif // __CUDACC__

// CUDA runtime macros — available from both nvcc and host compiler
#define FLAGCX_DEVICE_STREAM_PTR cudaStream_t *

#else
// Non-NVIDIA platform
#define FLAGCX_HOST_DECORATOR
#define FLAGCX_DEVICE_DECORATOR
#define FLAGCX_GLOBAL_DECORATOR
#define FLAGCX_DEVICE_INLINE_DECORATOR
#define FLAGCX_HOST_DEVICE_INLINE inline
#define FLAGCX_DEVICE_CONSTANT_DECORATOR
#define FLAGCX_DEVICE_STREAM_PTR
#define FLAGCX_DEVICE_THREAD_FENCE() ((void)0)
#define FLAGCX_DEVICE_SYNC_THREADS() ((void)0)
#define FLAGCX_THREAD_IDX_X 0
#define FLAGCX_BLOCK_IDX_X 0
#define FLAGCX_BLOCK_DIM_X 1
#define FLAGCX_GRID_DIM_X 1
#endif

#endif // FLAGCX_ADAPTOR_DEVICE_UTILS_H_
