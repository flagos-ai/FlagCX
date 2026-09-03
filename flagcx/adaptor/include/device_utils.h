/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_ADAPTOR_DEVICE_UTILS_H_
#define FLAGCX_ADAPTOR_DEVICE_UTILS_H_

// Device compiler detection — defined when any GPU device compiler is active.
// Extend with __ASCEND_CC__ etc. as new platforms are added.
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__xpu__)
#define FLAGCX_DEVICE_COMPILE 1
#endif

// Device compiler check (for conditional compilation in headers)
#ifndef FLAGCX_CHECK_DEVICE_CC
#ifdef __CUDACC__
#define FLAGCX_CHECK_DEVICE_CC 1
#else
#define FLAGCX_CHECK_DEVICE_CC 0
#endif
#endif

// IR extern "C" linkage — active only when building LLVM bitcode with clang
#ifdef __clang_llvm_bitcode_lib__
#define FLAGCX_IR_EXTERN_C extern "C"
#else
#define FLAGCX_IR_EXTERN_C
#endif

// Suppress unused-variable warnings for static arrays in headers
#define FLAGCX_MAYBE_UNUSED __attribute__((unused))

#if defined(USE_KUNLUNXIN_ADAPTOR)

#if defined(__xpu__)
#include "xpu/kernel/xtdk.h"
#include "xpu/runtime.h"
#if defined(FLAGCX_COMM_TRAITS_SHMEM)
#include "xshmem/xshmemx.h"
#endif

#define FLAGCX_HOST_DECORATOR __host__
#define FLAGCX_DEVICE_DECORATOR __device__
#define FLAGCX_GLOBAL_DECORATOR __global__
#define FLAGCX_DEVICE_INLINE_DECORATOR __device__ inline
#define FLAGCX_HOST_DEVICE_INLINE __host__ __device__ inline
#define FLAGCX_DEVICE_CONSTANT_DECORATOR __device__
#define FLAGCX_DEVICE_THREAD_FENCE() mfence()
#if defined(FLAGCX_COMM_TRAITS_SHMEM)
#define FLAGCX_DEVICE_SYNC_THREADS()                                           \
  xshmemi_threadgroup_sync<XSHMEMI_THREADGROUP_CLUSTER>()
#else
// XTDK does not expose a backend-neutral cluster barrier through FlagCX's
// supported headers. Default-backend device synchronization is therefore an
// explicit unsupported operation, not an XSHMEM-private compile dependency.
#define FLAGCX_DEVICE_SYNC_THREADS() __builtin_trap()
#endif
#define FLAGCX_THREAD_IDX_X core_id()
#define FLAGCX_BLOCK_IDX_X cluster_id()
#define FLAGCX_BLOCK_DIM_X core_num()
#define FLAGCX_GRID_DIM_X cluster_num()

// FlagCX KLX kernels use 16 cores per cluster. Model that launch contract as
// the logical SIMT width; wider XPU launches need a wider public lane mask and
// matching cooperative-group support first.
#define FLAGCX_SIMT_WIDTH 16
#define FLAGCX_SHARED __local__
#define FLAGCX_DEVICE_STREAM_PTR XPUStream *

#else
// Host compiler on KunlunXin — mirror the CUDA host pass: retain the same
// logical platform constants while erasing XPU qualifiers and builtins.
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
#define FLAGCX_SIMT_WIDTH 16
#define FLAGCX_SHARED static
#define FLAGCX_DEVICE_STREAM_PTR void **
#endif // __xpu__

#elif defined(USE_NVIDIA_ADAPTOR) || defined(USE_DU_ADAPTOR)
#include <cuda.h>
#include <cuda_runtime.h>

#if defined(__CUDACC__)
// Compiling with nvcc or clang CUDA — full CUDA qualifiers
#define FLAGCX_HOST_DECORATOR __host__
#define FLAGCX_DEVICE_DECORATOR __device__
#define FLAGCX_GLOBAL_DECORATOR __global__
#if defined(__clang_llvm_bitcode_lib__)
// clang bitcode mode: use always_inline (clang doesn't support __forceinline__)
#define FLAGCX_DEVICE_INLINE_DECORATOR __device__ __attribute__((always_inline))
#define FLAGCX_HOST_DEVICE_INLINE                                              \
  __host__ __device__ __attribute__((always_inline))
#else
#define FLAGCX_DEVICE_INLINE_DECORATOR __forceinline__ __device__
#define FLAGCX_HOST_DEVICE_INLINE __forceinline__ __host__ __device__
#endif
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
