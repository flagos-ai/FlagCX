/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 *
 * Custom AllReduce kernels using FlagCX Device API.
 * Communication infrastructure (comm, mem handle, barrier, multicast pointer)
 * uses FlagCX Device API. PTX multimem asm, pack/unpack, array_t/packed_t
 * are algorithm-specific and remain local to this file.
 ************************************************************************/

#include "device_api/flagcx_device.h"
#include "dev_comm_state.h"
#include "flagcx_kernel.h"
#include "nvidia_adaptor.h" // for struct flagcxStream (stream->base)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>

#ifdef FLAGCX_DEVICE_API_VENDOR

// Type aliases
typedef __half half;
typedef __nv_bfloat16 nv_bfloat16;

// ============================================================
// Algorithm-specific helpers: array_t, packed_t, pack/unpack,
// PTX multimem asm — local to this file, not part of Device API
// ============================================================

// Aligned array for vectorized operations
template <typename T, int N>
struct __align__(alignof(T) * N) array_t {
  T data[N];
  using type = T;
  static constexpr int size = N;
};

// Storage type based on byte size (4, 8, or 16 bytes)
template <int ByteSize> struct storage_type;
template <> struct storage_type<4>  { using type = uint32_t; };
template <> struct storage_type<8>  { using type = uint2; };
template <> struct storage_type<16> { using type = uint4; };

// Packed type: N is byte size (4, 8, or 16 bytes = 32, 64, or 128 bits)
template <typename T, int ByteSize = 4>
struct packed_t {
  static_assert(ByteSize == 4 || ByteSize == 8 || ByteSize == 16,
                "ByteSize must be 4, 8, or 16");
  static_assert(ByteSize >= sizeof(T), "ByteSize must be >= sizeof(T)");

  static constexpr int num_elems = ByteSize / sizeof(T);
  using elem_t = T;
  using array_type = array_t<T, num_elems>;
  using storage_t = typename storage_type<ByteSize>::type;
};

// Pack elements into storage type
template <typename T, int ByteSize = 4>
FLAGCX_DEVICE_INLINE_DECORATOR typename packed_t<T, ByteSize>::storage_t
pack(const T* data) {
  using P = packed_t<T, ByteSize>;
  if constexpr (ByteSize == 4) {
    if constexpr (sizeof(T) == 2) {
      uint16_t lo = *reinterpret_cast<const uint16_t*>(&data[0]);
      uint16_t hi = *reinterpret_cast<const uint16_t*>(&data[1]);
      return uint32_t(lo) | (uint32_t(hi) << 16);
    } else {
      return *reinterpret_cast<const uint32_t*>(&data[0]);
    }
  } else if constexpr (ByteSize == 8) {
    uint2 ret;
    ret.x = pack<T, 4>(&data[0]);
    ret.y = pack<T, 4>(&data[P::num_elems / 2]);
    return ret;
  } else if constexpr (ByteSize == 16) {
    uint4 ret;
    constexpr int quarter = P::num_elems / 4;
    ret.x = pack<T, 4>(&data[0]);
    ret.y = pack<T, 4>(&data[quarter]);
    ret.z = pack<T, 4>(&data[quarter * 2]);
    ret.w = pack<T, 4>(&data[quarter * 3]);
    return ret;
  }
}

// Unpack storage type into elements
template <typename T, int ByteSize = 4>
FLAGCX_DEVICE_INLINE_DECORATOR void
unpack(typename packed_t<T, ByteSize>::storage_t v, T* data) {
  using P = packed_t<T, ByteSize>;
  if constexpr (ByteSize == 4) {
    if constexpr (sizeof(T) == 2) {
      uint16_t lo = v & 0xffff;
      uint16_t hi = v >> 16;
      data[0] = *reinterpret_cast<T*>(&lo);
      data[1] = *reinterpret_cast<T*>(&hi);
    } else {
      data[0] = *reinterpret_cast<T*>(&v);
    }
  } else if constexpr (ByteSize == 8) {
    unpack<T, 4>(v.x, &data[0]);
    unpack<T, 4>(v.y, &data[P::num_elems / 2]);
  } else if constexpr (ByteSize == 16) {
    constexpr int quarter = P::num_elems / 4;
    unpack<T, 4>(v.x, &data[0]);
    unpack<T, 4>(v.y, &data[quarter]);
    unpack<T, 4>(v.z, &data[quarter * 2]);
    unpack<T, 4>(v.w, &data[quarter * 3]);
  }
}

// Convenience overloads for array_t
template <typename T, int ByteSize = 4>
FLAGCX_DEVICE_INLINE_DECORATOR typename packed_t<T, ByteSize>::storage_t
pack(const typename packed_t<T, ByteSize>::array_type& arr) {
  return pack<T, ByteSize>(arr.data);
}

template <typename T, int ByteSize = 4>
FLAGCX_DEVICE_INLINE_DECORATOR typename packed_t<T, ByteSize>::array_type
unpack(typename packed_t<T, ByteSize>::storage_t v) {
  typename packed_t<T, ByteSize>::array_type ret;
  unpack<T, ByteSize>(v, ret.data);
  return ret;
}

// Default ByteSize: 4 bytes for types <= 4 bytes, 8 bytes for 8-byte types
template <typename T>
constexpr int defaultByteSize() {
  return (sizeof(T) <= 4) ? 4 : 8;
}

// Elements per pack for given ByteSize
template <typename T, int ByteSize = defaultByteSize<T>()>
FLAGCX_DEVICE_INLINE_DECORATOR constexpr size_t elemsPerPack() {
  return packed_t<T, ByteSize>::num_elems;
}

// ============================================================
// PTX multimem load-reduce operations (SM90+)
// ============================================================

// Sum reduction
template <typename T, int ByteSize = (sizeof(T) <= 4 ? 4 : 8)>
FLAGCX_DEVICE_INLINE_DECORATOR typename packed_t<T, ByteSize>::array_type
multimem_sum(T* addr) {
  using P = packed_t<T, ByteSize>;
  typename P::array_type ret;
#if __CUDA_ARCH__ >= 900
  if constexpr (std::is_same<T, nv_bfloat16>::value) {
    typename P::storage_t h;
    asm volatile(
        "multimem.ld_reduce.global.add.bf16x2 %0, [%1];"
        : "=r"(h)
        : "l"(addr)
        : "memory");
    unpack<T, ByteSize>(h, ret.data);
  } else if constexpr (std::is_same<T, half>::value) {
    typename P::storage_t h;
    asm volatile(
        "multimem.ld_reduce.global.add.f16x2 %0, [%1];"
        : "=r"(h)
        : "l"(addr)
        : "memory");
    unpack<T, ByteSize>(h, ret.data);
  } else if constexpr (std::is_same<T, float>::value) {
    asm volatile(
        "multimem.ld_reduce.global.add.f32 %0, [%1];"
        : "=f"(ret.data[0])
        : "l"(addr)
        : "memory");
  } else if constexpr (std::is_same<T, double>::value) {
    asm volatile(
        "multimem.ld_reduce.global.add.f64 %0, [%1];"
        : "=d"(ret.data[0])
        : "l"(addr)
        : "memory");
  }
#endif
  return ret;
}

// Min reduction (only supported for 16-bit types)
template <typename T, int ByteSize = 4>
FLAGCX_DEVICE_INLINE_DECORATOR typename packed_t<T, ByteSize>::array_type
multimem_min(T* addr) {
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value,
                "multimem min only supports half and bfloat16");
  using P = packed_t<T, ByteSize>;
  typename P::array_type ret;
#if __CUDA_ARCH__ >= 900
  if constexpr (std::is_same<T, nv_bfloat16>::value) {
    typename P::storage_t h;
    asm volatile(
        "multimem.ld_reduce.global.min.bf16x2 %0, [%1];"
        : "=r"(h)
        : "l"(addr)
        : "memory");
    unpack<T, ByteSize>(h, ret.data);
  } else if constexpr (std::is_same<T, half>::value) {
    typename P::storage_t h;
    asm volatile(
        "multimem.ld_reduce.global.min.f16x2 %0, [%1];"
        : "=r"(h)
        : "l"(addr)
        : "memory");
    unpack<T, ByteSize>(h, ret.data);
  }
#endif
  return ret;
}

// Max reduction (only supported for 16-bit types)
template <typename T, int ByteSize = 4>
FLAGCX_DEVICE_INLINE_DECORATOR typename packed_t<T, ByteSize>::array_type
multimem_max(T* addr) {
  static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value,
                "multimem max only supports half and bfloat16");
  using P = packed_t<T, ByteSize>;
  typename P::array_type ret;
#if __CUDA_ARCH__ >= 900
  if constexpr (std::is_same<T, nv_bfloat16>::value) {
    typename P::storage_t h;
    asm volatile(
        "multimem.ld_reduce.global.max.bf16x2 %0, [%1];"
        : "=r"(h)
        : "l"(addr)
        : "memory");
    unpack<T, ByteSize>(h, ret.data);
  } else if constexpr (std::is_same<T, half>::value) {
    typename P::storage_t h;
    asm volatile(
        "multimem.ld_reduce.global.max.f16x2 %0, [%1];"
        : "=r"(h)
        : "l"(addr)
        : "memory");
    unpack<T, ByteSize>(h, ret.data);
  }
#endif
  return ret;
}

// Generic multimem reduce dispatcher
template <typename T, int ByteSize = (sizeof(T) <= 4 ? 4 : 8)>
FLAGCX_DEVICE_INLINE_DECORATOR typename packed_t<T, ByteSize>::array_type
multimem_reduce(T* addr, flagcxRedOp_t op) {
  if constexpr (std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value) {
    switch (op) {
      case flagcxSum:
        return multimem_sum<T, ByteSize>(addr);
      case flagcxMin:
        return multimem_min<T, ByteSize>(addr);
      case flagcxMax:
        return multimem_max<T, ByteSize>(addr);
      default:
        return multimem_sum<T, ByteSize>(addr);
    }
  } else {
    return multimem_sum<T, ByteSize>(addr);
  }
}

// Multimem store: broadcasts value to all GPUs
template <typename T, int ByteSize = (sizeof(T) <= 4 ? 4 : 8)>
FLAGCX_DEVICE_INLINE_DECORATOR void
multimem_st(T* addr, typename packed_t<T, ByteSize>::array_type val) {
#if __CUDA_ARCH__ >= 900
  using P = packed_t<T, ByteSize>;
  if constexpr (std::is_same<T, nv_bfloat16>::value) {
    typename P::storage_t h = pack<T, ByteSize>(val.data);
    asm volatile(
        "multimem.st.global.bf16x2 [%0], %1;"
        :
        : "l"(addr), "r"(h)
        : "memory");
  } else if constexpr (std::is_same<T, half>::value) {
    typename P::storage_t h = pack<T, ByteSize>(val.data);
    asm volatile(
        "multimem.st.global.f16x2 [%0], %1;"
        :
        : "l"(addr), "r"(h)
        : "memory");
  } else if constexpr (std::is_same<T, float>::value) {
    asm volatile(
        "multimem.st.global.f32 [%0], %1;"
        :
        : "l"(addr), "f"(val.data[0])
        : "memory");
  } else if constexpr (std::is_same<T, double>::value) {
    asm volatile(
        "multimem.st.global.f64 [%0], %1;"
        :
        : "l"(addr), "d"(val.data[0])
        : "memory");
  }
#endif
}

// Store to local memory using vectorized store
template <typename T, int ByteSize = defaultByteSize<T>()>
FLAGCX_DEVICE_INLINE_DECORATOR void
lsa_st(T* addr, typename packed_t<T, ByteSize>::array_type val) {
  constexpr int N = packed_t<T, ByteSize>::num_elems;
  if constexpr (N == 1) {
    addr[0] = val.data[0];
  } else {
    using storage_t = typename packed_t<T, ByteSize>::storage_t;
    *reinterpret_cast<storage_t*>(addr) = pack<T, ByteSize>(val.data);
  }
}

// ============================================================
// Kernels — communication via FlagCX Device API,
// algorithm via local PTX functions above
// ============================================================

// Local AllReduce: reduce from multimem, store to local buffer
template <typename T, int ByteSize = defaultByteSize<T>()>
__global__ void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
flagcxLocalAllReduceKernel(flagcxDevMem sendmem, size_t sendoffset,
                           void* recvbuffer, size_t count,
                           flagcxRedOp_t op, flagcxDevComm devComm) {
#if __CUDA_ARCH__ >= 900
  // FlagCX Device API: barrier
  flagcxTeam_t intra = flagcxTeamIntra(devComm);
  flagcxDevBarrier<flagcxTeamTagIntra, flagcxCoopBlock> bar{
      flagcxCoopBlock(), devComm, intra, FLAGCX_BLOCK_IDX_X, true};
  bar.sync(flagcxDeviceMemoryOrderAcquire);

  const int globalTid = FLAGCX_THREAD_IDX_X + FLAGCX_BLOCK_DIM_X * FLAGCX_BLOCK_IDX_X;
  const int globalNthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X;
  constexpr size_t pSize = elemsPerPack<T, ByteSize>();
  const size_t packCount = count / pSize;

  // FlagCX Device API: multicast pointer
  T* mmSendPtr = (T*)flagcxGetMulticastPointer(sendmem, sendoffset, devComm);
  T* lsaRecvPtr = (T*)recvbuffer;

  // Algorithm: local PTX multimem reduce + store
  for (size_t offset = globalTid; offset < packCount; offset += globalNthreads) {
    auto v = multimem_reduce<T, ByteSize>(mmSendPtr + pSize * offset, op);
    lsa_st<T, ByteSize>(lsaRecvPtr + pSize * offset, v);
  }

  bar.sync(flagcxDeviceMemoryOrderRelease);
#endif
}

// Interleaved AllReduce: reduce from multimem, store to multimem
template <typename T, int ByteSize = defaultByteSize<T>()>
__global__ void __launch_bounds__(FLAGCX_DEVICE_THREADS_PER_CTA)
flagcxInterleavedAllReduceKernel(flagcxDevMem sendmem, size_t sendoffset,
                                 flagcxDevMem recvmem, size_t recvoffset,
                                 size_t count, flagcxRedOp_t op,
                                 flagcxDevComm devComm) {
#if __CUDA_ARCH__ >= 900
  // FlagCX Device API: rank, barrier
  int rank = devComm.getIntraRank();
  int nRanks = devComm.getIntraSize();
  flagcxTeam_t intra = flagcxTeamIntra(devComm);
  flagcxDevBarrier<flagcxTeamTagIntra, flagcxCoopBlock> bar{
      flagcxCoopBlock(), devComm, intra, FLAGCX_BLOCK_IDX_X, true};
  bar.sync(flagcxDeviceMemoryOrderAcquire);

  const int globalTid = FLAGCX_THREAD_IDX_X +
      FLAGCX_BLOCK_DIM_X * (rank + FLAGCX_BLOCK_IDX_X * nRanks);
  const int globalNthreads = FLAGCX_BLOCK_DIM_X * FLAGCX_GRID_DIM_X * nRanks;
  constexpr size_t pSize = elemsPerPack<T, ByteSize>();
  const size_t packCount = count / pSize;

  // FlagCX Device API: multicast pointers
  T* mmSendPtr = (T*)flagcxGetMulticastPointer(sendmem, sendoffset, devComm);
  T* mmRecvPtr = (T*)flagcxGetMulticastPointer(recvmem, recvoffset, devComm);

  // Algorithm: local PTX multimem reduce + multimem store
  for (size_t offset = globalTid; offset < packCount; offset += globalNthreads) {
    auto v = multimem_reduce<T, ByteSize>(mmSendPtr + pSize * offset, op);
    multimem_st<T, ByteSize>(mmRecvPtr + pSize * offset, v);
  }

  bar.sync(flagcxDeviceMemoryOrderRelease);
#endif
}

// ============================================================
// Kernel launchers
// ============================================================

template <typename T>
flagcxResult_t launchLocalAllReduceKernel(flagcxDevMem sendmem, void* recvbuffer,
                                          size_t count, flagcxRedOp_t op,
                                          flagcxDevComm devComm, cudaStream_t stream) {
  flagcxLocalAllReduceKernel<T><<<FLAGCX_DEVICE_CTA_COUNT,
                                  FLAGCX_DEVICE_THREADS_PER_CTA, 0, stream>>>(
      sendmem, 0, recvbuffer, count, op, devComm);
  cudaError_t err = cudaGetLastError();
  return (err == cudaSuccess) ? flagcxSuccess : flagcxUnhandledDeviceError;
}

template <typename T>
flagcxResult_t launchInterleavedAllReduceKernel(flagcxDevMem sendmem,
                                                flagcxDevMem recvmem,
                                                size_t count, flagcxRedOp_t op,
                                                flagcxDevComm devComm,
                                                cudaStream_t stream) {
  flagcxInterleavedAllReduceKernel<T><<<FLAGCX_DEVICE_CTA_COUNT,
                                        FLAGCX_DEVICE_THREADS_PER_CTA, 0, stream>>>(
      sendmem, 0, recvmem, 0, count, op, devComm);
  cudaError_t err = cudaGetLastError();
  return (err == cudaSuccess) ? flagcxSuccess : flagcxUnhandledDeviceError;
}

// ============================================================
// Host entry points
// ============================================================

static inline size_t getAlignmentRequirement(flagcxDataType_t datatype) {
  switch (datatype) {
    case flagcxFloat16:
    case flagcxBfloat16:
      return 2;
    default:
      return 1;
  }
}

static flagcxResult_t flagcxLocalAllReduceDispatch(
    flagcxDevMem sendmem, void* recvbuff, size_t count,
    flagcxDataType_t datatype, flagcxRedOp_t op,
    flagcxDevComm devComm, cudaStream_t stream) {
  if (op != flagcxSum && op != flagcxMin && op != flagcxMax)
    return flagcxInvalidArgument;
  if ((op == flagcxMin || op == flagcxMax) &&
      (datatype != flagcxFloat16 && datatype != flagcxBfloat16))
    return flagcxInvalidArgument;
  if (count % getAlignmentRequirement(datatype) != 0)
    return flagcxInvalidArgument;

  switch (datatype) {
    case flagcxFloat16:
      return launchLocalAllReduceKernel<half>(sendmem, recvbuff, count, op, devComm, stream);
    case flagcxFloat32:
      return launchLocalAllReduceKernel<float>(sendmem, recvbuff, count, op, devComm, stream);
    case flagcxFloat64:
      return launchLocalAllReduceKernel<double>(sendmem, recvbuff, count, op, devComm, stream);
    case flagcxBfloat16:
      return launchLocalAllReduceKernel<nv_bfloat16>(sendmem, recvbuff, count, op, devComm, stream);
    default:
      return flagcxInvalidArgument;
  }
}

static flagcxResult_t flagcxInterleavedAllReduceDispatch(
    flagcxDevMem sendmem, flagcxDevMem recvmem, size_t count,
    flagcxDataType_t datatype, flagcxRedOp_t op,
    flagcxDevComm devComm, cudaStream_t stream) {
  if (op != flagcxSum && op != flagcxMin && op != flagcxMax)
    return flagcxInvalidArgument;
  if ((op == flagcxMin || op == flagcxMax) &&
      (datatype != flagcxFloat16 && datatype != flagcxBfloat16))
    return flagcxInvalidArgument;
  if (count % getAlignmentRequirement(datatype) != 0)
    return flagcxInvalidArgument;

  switch (datatype) {
    case flagcxFloat16:
      return launchInterleavedAllReduceKernel<half>(sendmem, recvmem, count, op, devComm, stream);
    case flagcxFloat32:
      return launchInterleavedAllReduceKernel<float>(sendmem, recvmem, count, op, devComm, stream);
    case flagcxFloat64:
      return launchInterleavedAllReduceKernel<double>(sendmem, recvmem, count, op, devComm, stream);
    case flagcxBfloat16:
      return launchInterleavedAllReduceKernel<nv_bfloat16>(sendmem, recvmem, count, op, devComm, stream);
    default:
      return flagcxInvalidArgument;
  }
}

// ============================================================
// Custom AllReduce entry point — registered as custom op
// ============================================================

extern "C" flagcxResult_t flagcxCustomAllReduceImpl(
    const void *sendbuff, void *recvbuff, size_t count,
    flagcxDataType_t datatype, flagcxRedOp_t op,
    flagcxComm_t comm, flagcxStream_t stream) {
  if (comm->devCommState == nullptr || !comm->devCommState->initialized)
    return flagcxNotSupported;

  auto *state = comm->devCommState;

  // Compute data size
  size_t elemSize = 0;
  switch (datatype) {
    case flagcxFloat16: case flagcxBfloat16: elemSize = 2; break;
    case flagcxFloat32: elemSize = 4; break;
    case flagcxFloat64: elemSize = 8; break;
    default: return flagcxNotSupported;
  }
  size_t size = count * elemSize;

  // Copy sendbuff to staged buffer
  cudaStream_t cudaStream = stream->base;
  cudaError_t cerr = cudaMemcpyAsync(state->sendStagedBuff, sendbuff, size,
                                      cudaMemcpyDeviceToDevice, cudaStream);
  if (cerr != cudaSuccess)
    return flagcxUnhandledDeviceError;

  // Construct device-side objects from host handles
  flagcxDevComm dc(*state->devComm);
  flagcxDevMem sm(*state->sendStagedMem);
  flagcxDevMem rm(*state->recvStagedMem);

  int nranks = state->devComm->intraSize;

  flagcxResult_t res;
  if ((nranks <= 4 && size < 512 * 1024) ||
      (nranks <= 8 && size < 256 * 1024)) {
    // Local allreduce: reduce from multimem, store to local recvbuff
    res = flagcxLocalAllReduceDispatch(sm, recvbuff, count, datatype, op,
                                       dc, cudaStream);
  } else {
    // Interleaved allreduce: reduce from multimem, store to multimem
    res = flagcxInterleavedAllReduceDispatch(sm, rm, count, datatype, op,
                                              dc, cudaStream);
    if (res == flagcxSuccess) {
      cerr = cudaMemcpyAsync(recvbuff, state->recvStagedBuff, size,
                              cudaMemcpyDeviceToDevice, cudaStream);
      if (cerr != cudaSuccess)
        return flagcxUnhandledDeviceError;
    }
  }
  return res;
}

#endif // FLAGCX_DEVICE_API_VENDOR
