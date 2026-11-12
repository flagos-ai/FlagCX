/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_ADAPTOR_DEVICE_UTILS_H_
#define FLAGCX_ADAPTOR_DEVICE_UTILS_H_

#include <stdint.h>

#if defined(__cplusplus)
// Register operand shapes for one complete value. For example, Vector4 on
// uint32_t selects four byte operands (.v4.b8).
struct Scalar {};
struct Vector2 {};
struct Vector4 {};
struct Vector8 {};

struct alignas(16) Bulk16 {
  uint32_t words[4];
};

struct alignas(32) Bulk32 {
  uint32_t words[8];
};

static_assert(sizeof(Bulk16) == 16 && alignof(Bulk16) == 16,
              "Bulk16 must occupy and align to 16 bytes");
static_assert(sizeof(Bulk32) == 32 && alignof(Bulk32) == 32,
              "Bulk32 must occupy and align to 32 bytes");
#endif

// Device compiler detection — defined when any GPU device compiler is active.
// Extend with __ASCEND_CC__ etc. as new platforms are added.
//
// Ascend takes the same route as Iluvatar, for the same reason: bisheng
// compiles one ASC translation unit in two passes and defines
// __CCE_IS_AICORE__ on the device pass only, while the types guarded here
// (flagcxDevNet) are named by BOTH halves of the helper translation unit —
// the host half has to size the struct whose bytes it hands to the kernel,
// so it cannot be blind to the type. The makefile therefore passes
// -DFLAGCX_ASCEND_DEVICE_COMPILE=1 for the whole compilation and the vendor
// condition keys on it; see makefiles/ascend.mk.
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__xpu__) ||           \
    defined(__CCE_IS_AICORE__) ||                                              \
    (defined(USE_ILUVATAR_ADAPTOR) &&                                          \
     defined(FLAGCX_ILUVATAR_DEVICE_COMPILE)) ||                               \
    (defined(USE_ASCEND_ADAPTOR) && defined(FLAGCX_ASCEND_DEVICE_COMPILE))
#define FLAGCX_DEVICE_COMPILE 1
#endif

// Unreachable device path. CUDA requires its device intrinsic; the other
// supported device compilers and host compilers accept __builtin_trap().
#if defined(__CUDA_ARCH__)
#define FLAGCX_DEVICE_TRAP() __trap()
#else
#define FLAGCX_DEVICE_TRAP() __builtin_trap()
#endif

// Device compiler check (for conditional compilation in headers)
//
// Deliberately NOT extended with the Ascend vendor condition above: this
// macro keys the NCCL backend's device-only blocks (nccl_comm_traits.h), and
// those have to stay out of the bisheng host pass, which emits host code even
// though the translation unit as a whole is a device compilation. What the
// Ascend device path needs to distinguish is keyed on __CCE_IS_AICORE__
// directly; see ascend_platform_traits.h.
#ifndef FLAGCX_CHECK_DEVICE_CC
#if defined(__CUDACC__) || defined(__HIPCC__) || defined(__xpu__) ||           \
    defined(__CCE_IS_AICORE__) ||                                              \
    (defined(USE_ILUVATAR_ADAPTOR) && defined(FLAGCX_ILUVATAR_DEVICE_COMPILE))
#define FLAGCX_CHECK_DEVICE_CC 1
#else
#define FLAGCX_CHECK_DEVICE_CC 0
#endif
#endif

// KunlunXin LLVM bitcode boundary.  Exported Device IR pointers live in global
// address space 1, and IR byte offsets must be 64-bit even though the xpu3
// device pass treats size_t as 32-bit.

// Tag for pointers that address device global memory and must keep the same
// width in every compilation pass. On XPU (xpu3) a generic pointer is 4 bytes
// on the device pass but 8 bytes on the host pass, while a __global_ptr__ is 8
// bytes in both — and a generic pointer cannot even receive a __global_ptr__
// value. So every pointer that crosses passes (members of by-value kernel
// parameters, IR signatures that hand device addresses around) carries this
// tag. It expands to nothing on all other platforms.
#if defined(USE_KUNLUNXIN_ADAPTOR) && defined(FLAGCX_DEVICE_COMPILE)
#define FLAGCX_DEVICE_GLOBAL_PTR __global_ptr__
#define FLAGCX_DEVICE_REQUIRES_LOCAL_NET_DESCRIPTOR 1
#else
#define FLAGCX_DEVICE_GLOBAL_PTR
#define FLAGCX_DEVICE_REQUIRES_LOCAL_NET_DESCRIPTOR 0
#endif

#define FLAGCX_DEVICE_GLOBAL_PTR_CAST(type, ptr)                               \
  ((FLAGCX_DEVICE_GLOBAL_PTR type *)(ptr))

// Pointer qualifier used only at the exported Device IR boundary. CoreX
// lowers persistent device objects (comm/window/net descriptors and remote
// pointers) in global address space 1. Keep this separate from
// FLAGCX_DEVICE_GLOBAL_PTR: the latter is also used by by-value C++ layouts,
// where changing the pointer type would alter the shared host/device ABI.
#if (defined(USE_ILUVATAR_ADAPTOR) &&                                          \
     defined(FLAGCX_ILUVATAR_DEVICE_COMPILE)) ||                               \
    (defined(USE_KUNLUNXIN_ADAPTOR) && defined(__clang_llvm_bitcode_lib__))
#define FLAGCX_IR_GLOBAL_PTR __attribute__((address_space(1)))
#define FLAGCX_IR_GLOBAL_RETURN_PTR __attribute__((address_space(1)))
#define FLAGCX_IR_GLOBAL_PTR_CAST(type, ptr)                                   \
  ((FLAGCX_IR_GLOBAL_PTR type *)(ptr))
#else
#define FLAGCX_IR_GLOBAL_PTR
#define FLAGCX_IR_GLOBAL_RETURN_PTR FLAGCX_DEVICE_GLOBAL_PTR
#define FLAGCX_IR_GLOBAL_PTR_CAST(type, ptr)                                   \
  FLAGCX_DEVICE_GLOBAL_PTR_CAST(type, ptr)
#endif

#if defined(USE_KUNLUNXIN_ADAPTOR) && defined(__clang_llvm_bitcode_lib__)
#define FLAGCX_IR_OFFSET_TYPE int64_t
#else
#define FLAGCX_IR_OFFSET_TYPE size_t
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
#if FLAGCX_DEVICE_REQUIRES_LOCAL_NET_DESCRIPTOR
#define FLAGCX_IR_NET_DECL(var, commOpaque, contextId)                         \
  flagcxDevNet var(*(const flagcxDevComm *)(commOpaque), (int)(contextId))
#define FLAGCX_IR_NET_REF(var) (var)
#define FLAGCX_IR_NET_ARG(var) ((const void FLAGCX_IR_GLOBAL_PTR *)&(var))
#else
#define FLAGCX_IR_NET_DECL(var, commOpaque, contextId)                         \
  const flagcxDevNet *var = (const flagcxDevNet *)flagcxDevNetGetFromCommS(    \
      (commOpaque), (int)(contextId))
#define FLAGCX_IR_NET_REF(var) (*(var))
#define FLAGCX_IR_NET_ARG(var) ((const void FLAGCX_IR_GLOBAL_PTR *)(var))
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
// clang-format off
#include "xshmem/non_abi/device/common/xshmemi_common_device.h"
#include "xshmem/non_abi/device/coll/barrier.h"
// clang-format on
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
#define FLAGCX_SHARED_DECORATOR __local__
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
#define FLAGCX_SHARED_DECORATOR static
#define FLAGCX_DEVICE_STREAM_PTR void **
#endif // __xpu__

#elif defined(USE_ILUVATAR_ADAPTOR)
#include <cuda.h>
#include <cuda_runtime.h>

#if defined(FLAGCX_ILUVATAR_DEVICE_COMPILE)
// clang -x ivcore does not guarantee __CUDACC__. Use the explicit
// device-compiler marker supplied by the native and Device IR build rules.
#define FLAGCX_HOST_DECORATOR __host__
#define FLAGCX_DEVICE_DECORATOR __device__
#define FLAGCX_GLOBAL_DECORATOR __global__
#define FLAGCX_DEVICE_INLINE_DECORATOR __device__ __attribute__((always_inline))
#if defined(__clang_llvm_bitcode_lib__)
#define FLAGCX_HOST_DEVICE_INLINE                                              \
  __host__ __device__ __attribute__((always_inline))
#else
#define FLAGCX_HOST_DEVICE_INLINE                                              \
  __host__ __device__ inline __attribute__((always_inline))
#endif
#define FLAGCX_DEVICE_CONSTANT_DECORATOR __device__ __constant__
// ivcore11 cannot select llvm.nvvm.membar.sys. The device-wide fence lowers
// to the CoreX cache writeback/invalidate operation used by peer-visible P2P
// synchronization in IXCCL and in FlagTree PR 1184.
#define FLAGCX_DEVICE_THREAD_FENCE __threadfence
#define FLAGCX_DEVICE_SYNC_THREADS __syncthreads
#define FLAGCX_THREAD_IDX_X threadIdx.x
#define FLAGCX_BLOCK_IDX_X blockIdx.x
#define FLAGCX_BLOCK_DIM_X blockDim.x
#define FLAGCX_GRID_DIM_X gridDim.x
#define FLAGCX_SIMT_WIDTH 64
#define FLAGCX_SHARED_DECORATOR __shared__
#else
// Host pass: retain layout constants and erase device-only qualifiers.
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
#define FLAGCX_SIMT_WIDTH 64
#define FLAGCX_SHARED_DECORATOR static
#endif // FLAGCX_ILUVATAR_DEVICE_COMPILE

#define FLAGCX_DEVICE_STREAM_PTR cudaStream_t *

#elif defined(USE_DU_ADAPTOR)
#include <hip/hip_runtime.h>

#if defined(__HIPCC__)
#define FLAGCX_HOST_DECORATOR __host__
#define FLAGCX_DEVICE_DECORATOR __device__
#define FLAGCX_GLOBAL_DECORATOR __global__
#if defined(__clang_llvm_bitcode_lib__)
// Export wrapper definitions from the standalone bitcode translation unit.
#define FLAGCX_DEVICE_INLINE_DECORATOR __device__ __attribute__((always_inline))
#define FLAGCX_HOST_DEVICE_INLINE                                              \
  __host__ __device__ __attribute__((always_inline))
#else
#define FLAGCX_DEVICE_INLINE_DECORATOR                                         \
  __device__ inline __attribute__((always_inline))
#define FLAGCX_HOST_DEVICE_INLINE                                              \
  __host__ __device__ inline __attribute__((always_inline))
#endif
#define FLAGCX_DEVICE_CONSTANT_DECORATOR __device__ __constant__
#define FLAGCX_DEVICE_THREAD_FENCE __threadfence_system
#define FLAGCX_DEVICE_SYNC_THREADS __syncthreads
#define FLAGCX_THREAD_IDX_X threadIdx.x
#define FLAGCX_BLOCK_IDX_X blockIdx.x
#define FLAGCX_BLOCK_DIM_X blockDim.x
#define FLAGCX_GRID_DIM_X gridDim.x
#define FLAGCX_SHARED_DECORATOR __shared__
#else
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
#define FLAGCX_SHARED_DECORATOR static
#endif
#define FLAGCX_SIMT_WIDTH 64
#define FLAGCX_DEVICE_STREAM_PTR hipStream_t *

#elif defined(USE_NVIDIA_ADAPTOR)
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
#define FLAGCX_SHARED_DECORATOR __shared__
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
#define FLAGCX_SHARED_DECORATOR static
#endif // __CUDACC__

// CUDA runtime macros — available from both nvcc and host compiler
#define FLAGCX_DEVICE_STREAM_PTR cudaStream_t *

#elif defined(USE_ASCEND_ADAPTOR)
// Ascend (CANN). bisheng compiles both passes of a CCE translation unit: it
// defines __CCE_IS_AICORE__ (plus __NPU_DEVICE__, __NPU_ARCH__ 2201 on the
// 910C soc Ascend910_9382) on the aicore pass and __NPU_HOST__ on the host
// pass, both verified with `bisheng -x asc --npu-soc=<soc> --cce-aicore-only
// -dM -E`. makefiles/ascend.mk additionally passes
// -DFLAGCX_ASCEND_DEVICE_COMPILE=1 so the branch does not depend on
// compiler-provided names alone (same trick Iluvatar uses, because clang -x
// ivcore does not guarantee __CUDACC__).
#if defined(__CCE_IS_AICORE__)
// Device pass. Ascend 910C gives FlagCX no usable SIMT lane model:
//   * `--cce-enable-puresimt` rejects this soc ("NPU soc is not supported");
//   * `__simt_vf__` and `__simd_vf__` cannot be combined with `__global__`
//     ('cce_simt_entry' / 'cce_simd_vf' and 'cce_kernel' attributes are not
//     compatible);
//   * the lane intrinsics that do exist (warpSize = 32, __ballot, __shfl in
//     __clang_cce_simt_dav3510_intrinsic.h) are named for the dav3510
//     architecture, while this soc reports __NPU_ARCH__ 2201, and `laneid()`
//     is not declared at all on this toolchain.
// So the model is one logical thread per AI core, with core-to-core and
// rank-to-rank cooperation expressed through atomics on global memory.
// FLAGCX_SIMT_WIDTH stays 1 — that is a measured statement about this
// toolchain, not a defaulted width (adaptation doc §5.1).
#define FLAGCX_HOST_DECORATOR
#define FLAGCX_DEVICE_DECORATOR __aicore__
#define FLAGCX_GLOBAL_DECORATOR __global__ __aicore__
#define FLAGCX_DEVICE_INLINE_DECORATOR __aicore__ inline
#define FLAGCX_HOST_DEVICE_INLINE __aicore__ inline
#define FLAGCX_DEVICE_CONSTANT_DECORATOR __aicore__
// Block-scope scratch lives in Unified Buffer. `__ub__` is not a keyword here
// (it fails as "unknown type name"); `__ubuf__` is, and a kernel declaring an
// `__ubuf__` array compiles with rc=0.
#define FLAGCX_SHARED __ubuf__
// No block barrier is validated for this soc yet, so these stay loud: the
// failure lands at build time (see the split below), which is a declared
// absence the adaptation doc allows (§13.3.2) rather than the silent no-op it
// forbids.
//
// __builtin_trap() is not available on this pass. bisheng lowers it to a call
// to abort and the CCE verifier refuses that outright - "Some error found in
// cce source file. (cannot call to symbol: abort)" - measured with a kernel
// whose entire body is the builtin and with one that also writes through a
// __gm__ pointer (probes T0/T1). Two further measurements then decide what
// "loud" can mean here:
//   * bisheng links the device image during -c, not in a separate device-link
//     step: a call to a declared-but-never-defined aicore function dies with
//     "ld.lld: error: undefined symbol: nf.cube / nf.vector" (probes T2/T3),
//     so a never-defined function refuses the BUILD, which is what a
//     declared-absent capability wants.
//   * that only works for paths nothing instantiates by default. The shared
//     headers end several non-void fallbacks with FLAGCX_DEVICE_TRAP() and no
//     return - default_comm_traits.h:94-100 and :170-182, reached through a
//     runtime provider switch, so the fallback is instantiated in EVERY
//     platform's device translation unit - and the CCE front end insists such
//     a function returns a value ("non return function ... should return a
//     value", gate run of ef8f74a). Those sites need a noreturn spelling that
//     names no symbol, or no library can be built at all.
// So the Ascend branch splits the two roles:
//   * FLAGCX_DEVICE_TRAP() - the shared "this cannot be reached" marker -
//     becomes __builtin_unreachable(): noreturn, no symbol, and defensible
//     here because the Default backend rejects the GDA provider long before a
//     device view is published (DefaultNetGdaProvider::isValid() returns false
//     at :86-92, supportsBarrierFence() accepts only Relaxed at :80-84), so a
//     guarded call cannot exist at runtime. Replacing even this with a real
//     trap instruction is an open measurement.
//   * every capability THIS platform declares absent - cache fence, block
//     barrier, read-modify-write, lane intrinsics - calls one of the three
//     never-defined functions below, so a kernel that reaches one fails to
//     build rather than running unordered or hanging.
extern "C" FLAGCX_DEVICE_DECORATOR void flagcxAscendNoCacheFence();
extern "C" FLAGCX_DEVICE_DECORATOR void flagcxAscendNoBlockBarrier();
extern "C" FLAGCX_DEVICE_DECORATOR void flagcxAscendDeviceOpUnsupported();
#undef FLAGCX_DEVICE_TRAP
#define FLAGCX_DEVICE_TRAP() __builtin_unreachable()
#define FLAGCX_DEVICE_THREAD_FENCE() flagcxAscendNoCacheFence()
#define FLAGCX_DEVICE_SYNC_THREADS() flagcxAscendNoBlockBarrier()
#define FLAGCX_THREAD_IDX_X 0
#define FLAGCX_BLOCK_IDX_X 0
#define FLAGCX_BLOCK_DIM_X 1
#define FLAGCX_GRID_DIM_X 1
#define FLAGCX_SIMT_WIDTH 1
// ACL typedefs aclrtStream to void*, so the stream slot needs no ACL header
// here; only vendor collective kernels (not built on Ascend) use this macro.
#define FLAGCX_DEVICE_STREAM_PTR void *
#else
// Host pass of an Ascend translation unit — keep the same logical platform
// constants and erase device-only qualifiers.
#define FLAGCX_HOST_DECORATOR
#define FLAGCX_DEVICE_DECORATOR
#define FLAGCX_GLOBAL_DECORATOR
#define FLAGCX_DEVICE_INLINE_DECORATOR inline
#define FLAGCX_HOST_DEVICE_INLINE inline
#define FLAGCX_DEVICE_CONSTANT_DECORATOR
#define FLAGCX_DEVICE_STREAM_PTR
#define FLAGCX_DEVICE_THREAD_FENCE() ((void)0)
#define FLAGCX_DEVICE_SYNC_THREADS() ((void)0)
#define FLAGCX_THREAD_IDX_X 0
#define FLAGCX_BLOCK_IDX_X 0
#define FLAGCX_BLOCK_DIM_X 1
#define FLAGCX_GRID_DIM_X 1
#define FLAGCX_SIMT_WIDTH 1
#define FLAGCX_SHARED static
#endif // __CCE_IS_AICORE__

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
