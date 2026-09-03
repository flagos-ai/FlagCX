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
#if defined(__CUDACC__) || defined(__xpu__)
#define FLAGCX_CHECK_DEVICE_CC 1
#else
#define FLAGCX_CHECK_DEVICE_CC 0
#endif
#endif

// Tag for pointers that address device global memory and must keep the same
// width in every compilation pass. On XPU (xpu3) a generic pointer is 4 bytes
// on the device pass but 8 bytes on the host pass, while a __global_ptr__ is 8
// bytes in both — and a generic pointer cannot even receive a __global_ptr__
// value. So every pointer that crosses passes (members of by-value kernel
// parameters, IR signatures that hand device addresses around) carries this
// tag. It expands to nothing on all other platforms.
#if defined(__xpu__)
#define FLAGCX_DEV_VALUE_PTR __global_ptr__
#else
#define FLAGCX_DEV_VALUE_PTR
#endif

// How an IR entry point gets hold of its flagcxDevNet.
//
// Everywhere a generic pointer can name device memory, the pre-built context
// array in the comm is used directly -- no per-call construction, which is what
// flagcxDevNetGetFromCommS exists for. On XPU that is impossible: the array is
// in address space 1, the *S entry points take generic pointers, and there is
// no conversion between the two. There the descriptor is rebuilt as a local
// object instead (it is a comm snapshot plus a context index, all of its
// methods are const, and every piece of mutable state lives behind the pointers
// it carries, so the local object is interchangeable with the shared original).
//
// The three macros keep that difference out of the IR sources: DECL introduces
// the variable, REF yields an object to call methods on, ARG yields the opaque
// pointer the *S entry points take.
#if defined(__xpu__)
#define FLAGCX_IR_NET_DECL(var, commOpaque, contextId)                         \
  flagcxDevNet var(*(const flagcxDevComm *)(commOpaque), (int)(contextId))
#define FLAGCX_IR_NET_REF(var) (var)
#define FLAGCX_IR_NET_ARG(var) ((const void *)&(var))
#else
#define FLAGCX_IR_NET_DECL(var, commOpaque, contextId)                         \
  const flagcxDevNet *var = (const flagcxDevNet *)flagcxDevNetGetFromCommS(    \
      (commOpaque), (int)(contextId))
#define FLAGCX_IR_NET_REF(var) (*(var))
#define FLAGCX_IR_NET_ARG(var) ((const void *)(var))
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
// xtdk.h only pulls in xtdk_io.h on the __XCN__ path, so device-side printf
// (declared in xtdk_io_xpu3.h under __arch_xpu3__) is missing on KLX. The
// xshmem device headers use printf in their __arch_xpu3__ branches.
#include "xpu/kernel/xtdk_io.h"
// Unlike xshmem.h, xshmemx.h does not include coll/barrier.h, yet
// xshmemx_coll_defines.h calls xshmemi_{barrier,sync}_threadgroup from it.
// barrier.h in turn uses xshmemi_threadfence_system (declared in
// xshmemi_common_device.h), so that must be included first.
#include "xshmem/non_abi/device/common/xshmemi_common_device.h"
#include "xshmem/non_abi/device/coll/barrier.h"
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
