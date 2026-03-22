/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 *
 * Custom AllReduce kernels using NCCL's Device API.
 ************************************************************************/

#include "nvidia_adaptor.h"
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
#include "nccl_device.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <type_traits>

// Type aliases
typedef __half half;
typedef __nv_bfloat16 nv_bfloat16;

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
// Example: packed_t<half, 4> = 2 half values in uint32_t
//          packed_t<half, 16> = 8 half values in uint4
//          packed_t<float, 4> = 1 float in uint32_t
//          packed_t<float, 16> = 4 floats in uint4
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

// Pack elements into storage type (ByteSize = 4, 8, or 16 bytes)
template <typename T, int ByteSize = 4>
FLAGCX_DEVICE_INLINE_DECORATOR typename packed_t<T, ByteSize>::storage_t
pack(const T* data) {
  using P = packed_t<T, ByteSize>;
  if constexpr (ByteSize == 4) {
    if constexpr (sizeof(T) == 2) {
      // 2x 16-bit → uint32_t
      uint16_t lo = *reinterpret_cast<const uint16_t*>(&data[0]);
      uint16_t hi = *reinterpret_cast<const uint16_t*>(&data[1]);
      return uint32_t(lo) | (uint32_t(hi) << 16);
    } else {
      // 1x 32-bit → uint32_t
      return *reinterpret_cast<const uint32_t*>(&data[0]);
    }
  } else if constexpr (ByteSize == 8) {
    // Recursively pack two 4-byte chunks → uint2
    uint2 ret;
    ret.x = pack<T, 4>(&data[0]);
    ret.y = pack<T, 4>(&data[P::num_elems / 2]);
    return ret;
  } else if constexpr (ByteSize == 16) {
    // Recursively pack four 4-byte chunks → uint4
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
      // uint32_t → 2x 16-bit
      uint16_t lo = v & 0xffff;
      uint16_t hi = v >> 16;
      data[0] = *reinterpret_cast<T*>(&lo);
      data[1] = *reinterpret_cast<T*>(&hi);
    } else {
      // uint32_t → 1x 32-bit
      data[0] = *reinterpret_cast<T*>(&v);
    }
  } else if constexpr (ByteSize == 8) {
    // uint2 → recursively unpack
    unpack<T, 4>(v.x, &data[0]);
    unpack<T, 4>(v.y, &data[P::num_elems / 2]);
  } else if constexpr (ByteSize == 16) {
    // uint4 → recursively unpack
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

// Multimem load-reduce operations
// Supported types: half, nv_bfloat16, float, double
// Supported ops: sum (add), min, max
// ByteSize=4 for 16/32-bit types, ByteSize=8 for 64-bit types

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

// Min reduction (only supported for 16-bit types: half, bfloat16)
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

// Max reduction (only supported for 16-bit types: half, bfloat16)
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
// Note: min/max only supported for 16-bit types (half, bfloat16)
//       sum supported for all floating-point types
template <typename T, int ByteSize = (sizeof(T) <= 4 ? 4 : 8)>
FLAGCX_DEVICE_INLINE_DECORATOR typename packed_t<T, ByteSize>::array_type
multimem_reduce(T* addr, ncclRedOp_t op) {
  if constexpr (std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value) {
    // 16-bit types support sum, min, max
    switch (op) {
      case ncclSum:
        return multimem_sum<T, ByteSize>(addr);
      case ncclMin:
        return multimem_min<T, ByteSize>(addr);
      case ncclMax:
        return multimem_max<T, ByteSize>(addr);
      default:
        return multimem_sum<T, ByteSize>(addr);
    }
  } else {
    // 32/64-bit types only support sum
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

// Store to local/shared memory using vectorized store
template <typename T, int ByteSize = defaultByteSize<T>()>
FLAGCX_DEVICE_INLINE_DECORATOR void
lsa_st(T* addr, typename packed_t<T, ByteSize>::array_type val) {
  constexpr int N = packed_t<T, ByteSize>::num_elems;
  if constexpr (N == 1) {
    // Single element (float, double) - direct store is already optimal
    addr[0] = val.data[0];
  } else {
    // Multiple elements (half, bfloat16) - use vectorized store
    using storage_t = typename packed_t<T, ByteSize>::storage_t;
    *reinterpret_cast<storage_t*>(addr) = pack<T, ByteSize>(val.data);
  }
}

// Local AllReduce: reduce from multimem, store to local buffer
// ByteSize: 4 bytes for 16/32-bit types, 8 bytes for 64-bit types
// Note: This kernel requires sm_90+ for multicast support
template <typename T, int ByteSize = defaultByteSize<T>()>
__global__ void __launch_bounds__(NCCL_ADAPTOR_DEVICE_THREADS_PER_CTA)
localAllReduceKernel(ncclWindow_t sendwin, size_t sendoffset,
                     void* recvbuffer, size_t count,
                     ncclRedOp_t op, struct ncclDevComm devComm) {
#if __CUDA_ARCH__ >= 900
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), devComm,
                                         ncclTeamLsa(devComm),
                                         devComm.lsaBarrier, blockIdx.x, true};
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire);

  const int globalTid = threadIdx.x + blockDim.x * blockIdx.x;
  const int globalNthreads = blockDim.x * gridDim.x;
  constexpr size_t pSize = elemsPerPack<T, ByteSize>();
  const size_t packCount = count / pSize;

  T* mmSendPtr = (T*)ncclGetLsaMultimemPointer(sendwin, sendoffset, devComm);
  T* lsaRecvPtr = (T*)recvbuffer;

  for (size_t offset = globalTid; offset < packCount; offset += globalNthreads) {
    auto v = multimem_reduce<T, ByteSize>(mmSendPtr + pSize * offset, op);
    lsa_st<T, ByteSize>(lsaRecvPtr + pSize * offset, v);
  }
#endif
}

// Interleaved AllReduce: reduce from multimem, store to multimem
// Note: This kernel requires sm_90+ for multicast support
template <typename T, int ByteSize = defaultByteSize<T>()>
__global__ void __launch_bounds__(NCCL_ADAPTOR_DEVICE_THREADS_PER_CTA)
interleavedAllReduceKernel(ncclWindow_t sendwin, size_t sendoffset,
                           ncclWindow_t recvwin, size_t recvoffset,
                           size_t count, ncclRedOp_t op,
                           struct ncclDevComm devComm) {
#if __CUDA_ARCH__ >= 900
  ncclLsaBarrierSession<ncclCoopCta> bar{ncclCoopCta(), devComm,
                                         ncclTeamLsa(devComm),
                                         devComm.lsaBarrier, blockIdx.x, true};
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

  const int rank = devComm.rank, nRanks = devComm.nRanks;
  const int globalTid = threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
  const int globalNthreads = blockDim.x * gridDim.x * nRanks;
  constexpr size_t pSize = elemsPerPack<T, ByteSize>();
  const size_t packCount = count / pSize;

  T* mmSendPtr = (T*)ncclGetLsaMultimemPointer(sendwin, sendoffset, devComm);
  T* mmRecvPtr = (T*)ncclGetLsaMultimemPointer(recvwin, recvoffset, devComm);

  for (size_t offset = globalTid; offset < packCount; offset += globalNthreads) {
    auto v = multimem_reduce<T, ByteSize>(mmSendPtr + pSize * offset, op);
    multimem_st<T, ByteSize>(mmRecvPtr + pSize * offset, v);
  }
  bar.sync(ncclCoopCta(), cuda::memory_order_release);
#endif
}

// Kernel launchers with error checking
template <typename T>
ncclResult_t launchLocalAllReduceKernel(ncclWindow_t sendwin, void* recvbuffer,
                                        size_t count, ncclRedOp_t op,
                                        ncclDevComm& devComm, cudaStream_t stream) {
  localAllReduceKernel<T><<<NCCL_ADAPTOR_DEVICE_CTA_COUNT,
                            NCCL_ADAPTOR_DEVICE_THREADS_PER_CTA, 0, stream>>>(
      sendwin, 0, recvbuffer, count, op, devComm);
  cudaError_t err = cudaGetLastError();
  return (err == cudaSuccess) ? ncclSuccess : ncclUnhandledCudaError;
}

template <typename T>
ncclResult_t launchInterleavedAllReduceKernel(ncclWindow_t sendwin, ncclWindow_t recvwin,
                                              size_t count, ncclRedOp_t op,
                                              ncclDevComm& devComm, cudaStream_t stream) {
  interleavedAllReduceKernel<T><<<NCCL_ADAPTOR_DEVICE_CTA_COUNT,
                                  NCCL_ADAPTOR_DEVICE_THREADS_PER_CTA, 0, stream>>>(
      sendwin, 0, recvwin, 0, count, op, devComm);
  cudaError_t err = cudaGetLastError();
  return (err == cudaSuccess) ? ncclSuccess : ncclUnhandledCudaError;
}

// Helper to get element alignment requirement for a datatype
static inline size_t getAlignmentRequirement(ncclDataType_t datatype) {
  switch (datatype) {
    case ncclFloat16:
    case ncclBfloat16:
      return 2;  // multimem operates on 2 elements (f16x2, bf16x2)
    default:
      return 1;  // float/double operate on single elements
  }
}

// Public API
// Supported types: half, bfloat16, float, double (floating-point only)
// Supported ops:
//   - ncclSum: all floating-point types
//   - ncclMin, ncclMax: only half and bfloat16 (16-bit types)
// Integer types are NOT supported by multimem.ld_reduce
// ncclProd and ncclAvg are NOT supported by multimem hardware
// Note: For 16-bit types, count must be even (multimem operates on 2 elements)
extern "C" ncclResult_t ncclAdaptorLocalAllReduce(
    const void* sendbuff, void* recvbuff, ncclWindow_t sendwin,
    ncclWindow_t recvwin, size_t count, ncclDataType_t datatype,
    ncclRedOp_t op, ncclDevComm& devComm, cudaStream_t stream) {
  // Validate reduction operation
  if (op != ncclSum && op != ncclMin && op != ncclMax) {
    return ncclInvalidArgument;
  }
  // min/max only supported for 16-bit types
  if ((op == ncclMin || op == ncclMax) &&
      (datatype != ncclFloat16 && datatype != ncclBfloat16)) {
    return ncclInvalidArgument;
  }
  // Validate count alignment (16-bit types require even count)
  if (count % getAlignmentRequirement(datatype) != 0) {
    return ncclInvalidArgument;
  }

  switch (datatype) {
    case ncclFloat16:
      return launchLocalAllReduceKernel<half>(sendwin, recvbuff, count, op, devComm, stream);
    case ncclFloat32:
      return launchLocalAllReduceKernel<float>(sendwin, recvbuff, count, op, devComm, stream);
    case ncclFloat64:
      return launchLocalAllReduceKernel<double>(sendwin, recvbuff, count, op, devComm, stream);
    case ncclBfloat16:
      return launchLocalAllReduceKernel<nv_bfloat16>(sendwin, recvbuff, count, op, devComm, stream);
    default:
      return ncclInvalidArgument;
  }
}

extern "C" ncclResult_t ncclAdaptorInterleavedAllReduce(
    const void* sendbuff, void* recvbuff, ncclWindow_t sendwin,
    ncclWindow_t recvwin, size_t count, ncclDataType_t datatype,
    ncclRedOp_t op, ncclDevComm& devComm, cudaStream_t stream) {
  // Validate reduction operation
  if (op != ncclSum && op != ncclMin && op != ncclMax) {
    return ncclInvalidArgument;
  }
  // min/max only supported for 16-bit types
  if ((op == ncclMin || op == ncclMax) &&
      (datatype != ncclFloat16 && datatype != ncclBfloat16)) {
    return ncclInvalidArgument;
  }
  // Validate count alignment (16-bit types require even count)
  if (count % getAlignmentRequirement(datatype) != 0) {
    return ncclInvalidArgument;
  }

  switch (datatype) {
    case ncclFloat16:
      return launchInterleavedAllReduceKernel<half>(sendwin, recvwin, count, op, devComm, stream);
    case ncclFloat32:
      return launchInterleavedAllReduceKernel<float>(sendwin, recvwin, count, op, devComm, stream);
    case ncclFloat64:
      return launchInterleavedAllReduceKernel<double>(sendwin, recvwin, count, op, devComm, stream);
    case ncclBfloat16:
      return launchInterleavedAllReduceKernel<nv_bfloat16>(sendwin, recvwin, count, op, devComm, stream);
    default:
      return ncclInvalidArgument;
  }
}

#endif // NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)