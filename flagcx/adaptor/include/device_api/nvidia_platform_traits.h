/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA Platform Traits — CUDA SIMT intrinsics and cuda::atomic_ref.
 *
 * Provides PlatformTraits<NvidiaPlatform> with:
 *   - Intrin: lane(), activemask(), syncwarp(), popc(), spinBackoff(), ...
 *   - Atomic: load(), store(), fetchAdd(), compareExchange(), ...
 *   - CoopOps: per-thread and cooperative single-device memory operations
 *   - Clock: device nanosecond timer
 *
 * Both __CUDACC__ (device) and host-compiler paths are NVIDIA-specific;
 * the #ifdef __CUDACC__ split is device-vs-host, NOT vendor-vs-fallback.
 ************************************************************************/

#ifndef FLAGCX_NVIDIA_PLATFORM_TRAITS_H_
#define FLAGCX_NVIDIA_PLATFORM_TRAITS_H_

#include "nvidia_ptx_memory.h"
#include <cassert>
#include <cstdint>
#include <cuda/atomic>
#include <type_traits>
#include <utility>

struct NvidiaPlatform {};

template <>
struct PlatformTraits<NvidiaPlatform> {
  struct CoopOps;
  struct Clock;

  // ==============================================================
  // Intrin — CUDA SIMT intrinsics
  // ==============================================================
  struct Intrin {
    static constexpr int simtWidth = FLAGCX_SIMT_WIDTH;
    static FLAGCX_HOST_DEVICE_INLINE constexpr flagcxLaneMask_t fullMask() {
      return 0xffffffffull;
    }

#if defined(__CUDACC__)
    static FLAGCX_DEVICE_INLINE_DECORATOR int lane() {
      int l;
      asm("mov.u32 %0, %%laneid;" : "=r"(l));
      return l;
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR flagcxLaneMask_t lanemaskLt() {
      uint32_t m;
      asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
      return static_cast<flagcxLaneMask_t>(m);
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR flagcxLaneMask_t activemask() {
      return static_cast<flagcxLaneMask_t>(__activemask());
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR void
    validateMask(flagcxLaneMask_t mask) {
      if ((mask & ~fullMask()) != 0)
        __trap();
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR void
    syncwarp(flagcxLaneMask_t mask = fullMask()) {
      validateMask(mask);
      __syncwarp(static_cast<uint32_t>(mask));
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR int popc(flagcxLaneMask_t x) {
      validateMask(x);
      return __popc(static_cast<uint32_t>(x));
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR void namedBarrierSync(int id,
                                                                int nThreads) {
      __barrier_sync_count(id, nThreads);
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR void spinBackoff(int iter) {
      int delay = 1 << (iter < 15 ? iter : 15);
#if __CUDA_ARCH__ >= 700
      __nanosleep(delay);
#else
      uint64_t start = clock64();
      while (clock64() - start < (uint64_t)delay) { /* spin */
      }
#endif
    }

    // System-level fence: ensures stores are visible across all system agents
    // (including other processes via IPC). Maps to PTX membar.sys.
    static FLAGCX_DEVICE_INLINE_DECORATOR void threadfenceSystem() {
      __threadfence_system();
    }

    // Device-level fence: ordering within this GPU only. Maps to membar.gl.
    static FLAGCX_DEVICE_INLINE_DECORATOR void threadfenceDevice() {
      __threadfence();
    }

    // Internal strided byte-copy core shared by the legacy and new entries.
    //
    // Cascades from 16B vectors (int4) down to byte level so that aligned
    // regions move in the widest available store. Pattern adopted from NVSHMEM
    // for stronger memory ordering guarantees.
    template <typename DstPtr, typename SrcPtr>
    static FLAGCX_DEVICE_INLINE_DECORATOR void
    copyBytesStridedImpl(DstPtr dstIn, SrcPtr srcIn, size_t bytes, int rank,
                         int size) {
      void *dst = (void *)dstIn;
      const void *src = (const void *)srcIn;

      if (((uintptr_t)dst % 16 == 0) && ((uintptr_t)src % 16 == 0)) {
        int4 *d = (int4 *)dst;
        const int4 *s = (const int4 *)src;
        size_t nelems = bytes / 16;
        for (size_t i = (size_t)rank; i < nelems; i += (size_t)size)
          d[i] = s[i];
        bytes -= nelems * 16;
        if (bytes == 0)
          return;
        dst = (void *)(d + nelems);
        src = (const void *)(s + nelems);
      }

      if (((uintptr_t)dst % 8 == 0) && ((uintptr_t)src % 8 == 0)) {
        uint64_t *d = (uint64_t *)dst;
        const uint64_t *s = (const uint64_t *)src;
        size_t nelems = bytes / 8;
        for (size_t i = (size_t)rank; i < nelems; i += (size_t)size)
          d[i] = s[i];
        bytes -= nelems * 8;
        if (bytes == 0)
          return;
        dst = (void *)(d + nelems);
        src = (const void *)(s + nelems);
      }

      if (((uintptr_t)dst % 4 == 0) && ((uintptr_t)src % 4 == 0)) {
        uint32_t *d = (uint32_t *)dst;
        const uint32_t *s = (const uint32_t *)src;
        size_t nelems = bytes / 4;
        for (size_t i = (size_t)rank; i < nelems; i += (size_t)size)
          d[i] = s[i];
        bytes -= nelems * 4;
        if (bytes == 0)
          return;
        dst = (void *)(d + nelems);
        src = (const void *)(s + nelems);
      }

      if (((uintptr_t)dst % 2 == 0) && ((uintptr_t)src % 2 == 0)) {
        uint16_t *d = (uint16_t *)dst;
        const uint16_t *s = (const uint16_t *)src;
        size_t nelems = bytes / 2;
        for (size_t i = (size_t)rank; i < nelems; i += (size_t)size)
          d[i] = s[i];
        bytes -= nelems * 2;
        if (bytes == 0)
          return;
        dst = (void *)(d + nelems);
        src = (const void *)(s + nelems);
      }

      unsigned char *d = (unsigned char *)dst;
      const unsigned char *s = (const unsigned char *)src;
      for (size_t i = (size_t)rank; i < bytes; i += (size_t)size)
        d[i] = s[i];
    }

    template <typename DstPtr, typename SrcPtr>
    FLAGCX_DEVICE_API_DEPRECATED("use CoopOps::copy for cooperative copies")
    static FLAGCX_DEVICE_INLINE_DECORATOR
        void coopCopyBytes(DstPtr dst, SrcPtr src, size_t bytes, int rank,
                           int size) {
      copyBytesStridedImpl(dst, src, bytes, rank, size);
    }

    // One issuing thread copies the entire byte range.  coopCopyBytes above
    // retains its legacy rank/size partitioning and tail behavior.
    template <typename DstPtr, typename SrcPtr>
    static FLAGCX_DEVICE_INLINE_DECORATOR void copyBytes(DstPtr dst, SrcPtr src,
                                                         size_t bytes) {
      copyBytesStridedImpl(dst, src, bytes, 0, 1);
    }

#else
    // Host-compiler stubs (allow template instantiation, never called at
    // runtime)
    static inline int lane() {
      assert(false && "lane() called on host");
      return 0;
    }
    static inline flagcxLaneMask_t lanemaskLt() {
      assert(false && "lanemaskLt() called on host");
      return 0;
    }
    static inline flagcxLaneMask_t activemask() {
      assert(false && "activemask() called on host");
      return 1;
    }
    static inline void validateMask(flagcxLaneMask_t mask) {
      assert((mask & ~fullMask()) == 0 &&
             "lane mask exceeds NVIDIA SIMT width");
    }
    static inline void syncwarp(flagcxLaneMask_t mask = fullMask()) {
      (void)mask;
      assert(false && "syncwarp() called on host");
    }
    static inline int popc(flagcxLaneMask_t x) {
      (void)x;
      assert(false && "popc() called on host");
      return 0;
    }
    static inline void namedBarrierSync(int id, int nThreads) {
      (void)id;
      (void)nThreads;
      assert(false && "namedBarrierSync() called on host");
    }
    static inline void spinBackoff(int iter) {
      (void)iter;
      assert(false && "spinBackoff() called on host");
    }
    static inline void threadfenceSystem() {
      assert(false && "threadfenceSystem() called on host");
    }
    static inline void threadfenceDevice() {
      assert(false && "threadfenceDevice() called on host");
    }
    template <typename DstPtr, typename SrcPtr>
    static inline void copyBytesStridedImpl(DstPtr, SrcPtr, size_t, int, int) {
      assert(false && "copyBytesStridedImpl() called on host");
    }
    template <typename DstPtr, typename SrcPtr>
    FLAGCX_DEVICE_API_DEPRECATED("use CoopOps::copy for cooperative copies")
    static inline void coopCopyBytes(DstPtr dst, SrcPtr src, size_t bytes,
                                     int rank, int size) {
      copyBytesStridedImpl(dst, src, bytes, rank, size);
    }
    template <typename DstPtr, typename SrcPtr>
    static inline void copyBytes(DstPtr, SrcPtr, size_t) {
      assert(false && "copyBytes() called on host");
    }
#endif // __CUDACC__

    template <typename T, typename Shape, flagcxDeviceMemorySpace_t Space,
              flagcxDeviceMemoryOrder_t Order = flagcxDeviceMemoryOrderWeak,
              flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem,
              flagcxDeviceMemoryAccessKind_t Kind =
                  flagcxDeviceMemoryAccessKindNormal,
              flagcxDeviceLoadIntent_t Intent = flagcxDeviceLoadIntentOrdinary,
              flagcxDeviceCacheHint_t CacheHint = flagcxDeviceCacheHintNone>
    static FLAGCX_DEVICE_INLINE_DECORATOR T load(const T *src) {
      return flagcxNvidiaMemory::Load<T, Shape, Space, Order, Scope, Kind,
                                      Intent, CacheHint>::run(src);
    }

    template <typename T, typename Shape, flagcxDeviceMemorySpace_t Space,
              flagcxDeviceMemoryOrder_t Order = flagcxDeviceMemoryOrderWeak,
              flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem,
              flagcxDeviceMemoryAccessKind_t Kind =
                  flagcxDeviceMemoryAccessKindNormal,
              flagcxDeviceCacheHint_t CacheHint = flagcxDeviceCacheHintNone>
    static FLAGCX_DEVICE_INLINE_DECORATOR void store(T *dst, T value) {
      flagcxNvidiaMemory::Store<T, Shape, Space, Order, Scope, Kind,
                                CacheHint>::run(dst, value);
    }

    // One issuing thread moves one complete T value. No asynchronous completion
    // mechanism is involved; each endpoint chooses its own PTX load/store form.
    template <
        typename T, typename SrcShape, typename DstShape,
        flagcxDeviceMemorySpace_t SrcSpace, flagcxDeviceMemorySpace_t DstSpace,
        flagcxDeviceMemoryOrder_t SrcOrder = flagcxDeviceMemoryOrderWeak,
        flagcxDeviceMemoryOrder_t DstOrder = flagcxDeviceMemoryOrderWeak,
        flagcxDeviceScope_t SrcScope = flagcxDeviceScopeSystem,
        flagcxDeviceScope_t DstScope = flagcxDeviceScopeSystem,
        flagcxDeviceMemoryAccessKind_t SrcKind =
            flagcxDeviceMemoryAccessKindNormal,
        flagcxDeviceMemoryAccessKind_t DstKind =
            flagcxDeviceMemoryAccessKindNormal,
        flagcxDeviceLoadIntent_t SrcIntent = flagcxDeviceLoadIntentOrdinary,
        flagcxDeviceCacheHint_t SrcCacheHint = flagcxDeviceCacheHintNone,
        flagcxDeviceCacheHint_t DstCacheHint = flagcxDeviceCacheHintNone>
    static FLAGCX_DEVICE_INLINE_DECORATOR void copy(T *dst, const T *src) {
      T value = load<T, SrcShape, SrcSpace, SrcOrder, SrcScope, SrcKind,
                     SrcIntent, SrcCacheHint>(src);
      store<T, DstShape, DstSpace, DstOrder, DstScope, DstKind, DstCacheHint>(
          dst, value);
    }

    template <typename T>
    static FLAGCX_HOST_DEVICE_INLINE T byteSwap(T value) {
      static_assert(std::is_integral<T>::value && std::is_unsigned<T>::value &&
                        !std::is_same<T, bool>::value &&
                        (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 ||
                         sizeof(T) == 8),
                    "byteSwap requires an unsigned 8/16/32/64-bit integer");
      T result = 0;
      for (unsigned i = 0; i < sizeof(T); ++i) {
        result = static_cast<T>((result << 8) | (value & T{0xff}));
        value = static_cast<T>(value >> 8);
      }
      return result;
    }

    template <typename T,
              flagcxDeviceFunnelShiftMode_t Mode = flagcxDeviceFunnelShiftWrap>
    static FLAGCX_HOST_DEVICE_INLINE T funnelShiftRight(T lo, T hi,
                                                        unsigned shift) {
      static_assert(std::is_integral<T>::value && std::is_unsigned<T>::value &&
                        (sizeof(T) == 4 || sizeof(T) == 8),
                    "funnelShiftRight requires an unsigned 32/64-bit integer");
      static_assert(Mode == flagcxDeviceFunnelShiftWrap ||
                        Mode == flagcxDeviceFunnelShiftClamp,
                    "invalid funnel shift mode");
      constexpr unsigned bits = sizeof(T) * 8;
#if defined(__CUDA_ARCH__)
      if (sizeof(T) == 4) {
        const uint32_t lo32 = static_cast<uint32_t>(lo);
        const uint32_t hi32 = static_cast<uint32_t>(hi);
        return static_cast<T>(Mode == flagcxDeviceFunnelShiftWrap
                                  ? __funnelshift_r(lo32, hi32, shift)
                                  : __funnelshift_rc(lo32, hi32, shift));
      }
#endif
      const unsigned amount = Mode == flagcxDeviceFunnelShiftWrap
                                  ? shift % bits
                                  : (shift < bits ? shift : bits);
      if (amount == 0)
        return lo;
      if (amount == bits)
        return hi;
      return static_cast<T>((lo >> amount) | (hi << (bits - amount)));
    }
  };

  // ==============================================================
  // Atomic — cuda::atomic_ref scoped operations
  // ==============================================================
  struct Atomic {
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T
    load(T *ptr, flagcxDeviceMemoryOrder_t order) {
      using ref_t = typename ScopeHelper<T, Scope>::atomic_ref_t;
      return ref_t{*ptr}.load(toOrder(order));
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR void
    store(T *ptr, const T &val, flagcxDeviceMemoryOrder_t order) {
      using ref_t = typename ScopeHelper<T, Scope>::atomic_ref_t;
      ref_t{*ptr}.store(val, toOrder(order));
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T
    fetchAdd(T *ptr, const T &val, flagcxDeviceMemoryOrder_t order) {
      using ref_t = typename ScopeHelper<T, Scope>::atomic_ref_t;
      return ref_t{*ptr}.fetch_add(val, toOrder(order));
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T
    fetchSub(T *ptr, const T &val, flagcxDeviceMemoryOrder_t order) {
      using ref_t = typename ScopeHelper<T, Scope>::atomic_ref_t;
      return ref_t{*ptr}.fetch_sub(val, toOrder(order));
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T
    fetchOr(T *ptr, const T &val, flagcxDeviceMemoryOrder_t order) {
      using ref_t = typename ScopeHelper<T, Scope>::atomic_ref_t;
      return ref_t{*ptr}.fetch_or(val, toOrder(order));
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T
    fetchAnd(T *ptr, const T &val, flagcxDeviceMemoryOrder_t order) {
      using ref_t = typename ScopeHelper<T, Scope>::atomic_ref_t;
      return ref_t{*ptr}.fetch_and(val, toOrder(order));
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T
    fetchMax(T *ptr, T val, flagcxDeviceMemoryOrder_t order) {
      static_assert(std::is_integral<T>::value &&
                        !std::is_same<T, bool>::value &&
                        (sizeof(T) == 4 || sizeof(T) == 8),
                    "fetchMax requires a 32/64-bit integer");
      if (order < flagcxDeviceMemoryOrderRelaxed ||
          order > flagcxDeviceMemoryOrderSeqCst) {
#if defined(__CUDA_ARCH__)
        __trap();
#else
        assert(false && "invalid fetchMax order");
#endif
      }
      using ref_t = typename ScopeHelper<T, Scope>::atomic_ref_t;
      ref_t ref{*ptr};
      T old = ref.load(cuda::memory_order_relaxed);
      for (;;) {
        const T desired = old < val ? val : old;
        if (ref.compare_exchange_weak(old, desired, toOrder(order),
                                      cuda::memory_order_relaxed))
          return old;
      }
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T
    exchange(T *ptr, const T &val, flagcxDeviceMemoryOrder_t order) {
      using ref_t = typename ScopeHelper<T, Scope>::atomic_ref_t;
      return ref_t{*ptr}.exchange(val, toOrder(order));
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR bool
    compareExchange(T *ptr, T &expected, const T &desired,
                    flagcxDeviceMemoryOrder_t order) {
      using ref_t = typename ScopeHelper<T, Scope>::atomic_ref_t;
      return ref_t{*ptr}.compare_exchange_strong(expected, desired,
                                                 toOrder(order));
    }

  private:
    static FLAGCX_DEVICE_INLINE_DECORATOR cuda::memory_order
    toOrder(flagcxDeviceMemoryOrder_t o) {
      switch (o) {
        case flagcxDeviceMemoryOrderRelaxed:
          return cuda::memory_order_relaxed;
        case flagcxDeviceMemoryOrderAcquire:
          return cuda::memory_order_acquire;
        case flagcxDeviceMemoryOrderRelease:
          return cuda::memory_order_release;
        case flagcxDeviceMemoryOrderAcqRel:
          return cuda::memory_order_acq_rel;
        case flagcxDeviceMemoryOrderSeqCst:
          return cuda::memory_order_seq_cst;
        default:
#if defined(__CUDA_ARCH__)
          __trap();
#else
          __builtin_trap();
#endif
          return cuda::memory_order_relaxed;
      }
    }

  public:
    // Public conversion helpers for vendor code that passes native enums
    // to NCCL/CUDA functions.
    static FLAGCX_DEVICE_INLINE_DECORATOR cuda::memory_order
    toNativeOrder(flagcxDeviceMemoryOrder_t o) {
      return toOrder(o);
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR cuda::thread_scope
    toNativeScope(flagcxDeviceScope_t s) {
      FLAGCX_MAYBE_UNUSED static FLAGCX_DEVICE_CONSTANT_DECORATOR
          cuda::thread_scope map[] = {
              cuda::thread_scope_system, cuda::thread_scope_device,
              cuda::thread_scope_block, cuda::thread_scope_thread};
      return map[s];
    }

  private:
    // Scope dispatch helper
    template <typename T, flagcxDeviceScope_t Scope>
    struct ScopeHelper;

    template <typename T>
    struct ScopeHelper<T, flagcxDeviceScopeSystem> {
      using atomic_ref_t = cuda::atomic_ref<T, cuda::thread_scope_system>;
    };

    template <typename T>
    struct ScopeHelper<T, flagcxDeviceScopeDevice> {
      using atomic_ref_t = cuda::atomic_ref<T, cuda::thread_scope_device>;
    };

    template <typename T>
    struct ScopeHelper<T, flagcxDeviceScopeBlock> {
      using atomic_ref_t = cuda::atomic_ref<T, cuda::thread_scope_block>;
    };

    template <typename T>
    struct ScopeHelper<T, flagcxDeviceScopeThread> {
      using atomic_ref_t = cuda::atomic_ref<T, cuda::thread_scope_thread>;
    };
  };

  // ==============================================================
  // Coop — SIMT cooperative groups
  // ==============================================================

  struct CoopBlock {
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return FLAGCX_THREAD_IDX_X;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const {
      return FLAGCX_BLOCK_DIM_X;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() { FLAGCX_DEVICE_SYNC_THREADS(); }
  };

  template <int N>
  struct CoopTile {
    static_assert(N > 0 && (N & (N - 1)) == 0 && N <= Intrin::simtWidth,
                  "N must be power of 2 and <= simtWidth");
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return Intrin::lane() % N;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return N; }
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxLaneMask_t laneMask() const {
      return (Intrin::fullMask() >> (Intrin::simtWidth - N))
             << (Intrin::lane() & -N);
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() {
      if (N > 1)
        Intrin::syncwarp(laneMask());
    }
  };

  using CoopThread = CoopTile<1>;
  using CoopWarp = CoopTile<Intrin::simtWidth>;

  struct CoopTileSpan {
    uint32_t t0 : 8, nTiles : 8, id : 8;
    FLAGCX_DEVICE_INLINE_DECORATOR CoopTileSpan(int t0, int nTiles, int id)
        : t0(t0), nTiles(nTiles), id(id) {}
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return FLAGCX_THREAD_IDX_X - Intrin::simtWidth * t0;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const {
      return Intrin::simtWidth * nTiles;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() {
      Intrin::namedBarrierSync(1 + id, Intrin::simtWidth * nTiles);
    }
  };

  struct CoopLanes {
    flagcxLaneMask_t lmask;
    FLAGCX_DEVICE_INLINE_DECORATOR
    CoopLanes(flagcxLaneMask_t lmask = Intrin::fullMask()) : lmask(lmask) {
      Intrin::validateMask(lmask);
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return Intrin::popc(lmask & Intrin::lanemaskLt());
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const {
      return Intrin::popc(lmask);
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() { Intrin::syncwarp(lmask); }
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxLaneMask_t getLmask() const {
      return lmask;
    }
  };

  using CoopAny = PlatformCoop;
};

namespace flagcxNvidiaDetail {
template <typename...>
using FlagcxNvidiaVoidT = void;

template <typename T, typename = void>
struct FlagcxNvidiaHasLaneMask : std::false_type {};
template <typename T>
struct FlagcxNvidiaHasLaneMask<
    T, FlagcxNvidiaVoidT<decltype(std::declval<const T &>().laneMask())>>
    : std::true_type {};
template <typename T, typename = void>
struct FlagcxNvidiaHasLmask : std::false_type {};
template <typename T>
struct FlagcxNvidiaHasLmask<
    T, FlagcxNvidiaVoidT<decltype(std::declval<const T &>().getLmask())>>
    : std::true_type {};
template <typename T>
struct FlagcxNvidiaWarpCoop
    : std::integral_constant<bool, FlagcxNvidiaHasLaneMask<T>::value ||
                                       FlagcxNvidiaHasLmask<T>::value> {};

template <typename Ptr>
struct FlagcxNvidiaPointee {
  using type =
      typename std::remove_cv<typename std::remove_pointer<Ptr>::type>::type;
};

} // namespace flagcxNvidiaDetail

struct PlatformTraits<NvidiaPlatform>::Clock {
  using tick_t = uint64_t;

  static FLAGCX_DEVICE_INLINE_DECORATOR tick_t now() {
#if defined(__CUDA_ARCH__)
    // PTX defines nanoseconds, but documents this register as target-specific.
    tick_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
#else
    assert(false && "Clock::now() called on host");
    return 0;
#endif
  }

  static FLAGCX_HOST_DEVICE_INLINE tick_t elapsedNs(tick_t start, tick_t end) {
    return end - start;
  }
};

struct PlatformTraits<NvidiaPlatform>::CoopOps {
  using Platform = PlatformTraits<NvidiaPlatform>;

  // MMIO additionally requires a caller-verified mapped I/O address and
  // access permissions; no enum can establish pointer provenance by itself.

  // One batch belongs to one issuing thread. It does not represent a Coop-wide
  // completion event and cannot be used to wait for storeAsync.
  struct AsyncCopyBatch {
    bool issued = false;
    bool committed = false;
  };

  // T is the entire value; Shape selects its PTX register decomposition.
  // Each caller performs its own load/store, including for Local addresses.
  template <
      typename T, typename Shape, flagcxDeviceMemorySpace_t Space,
      flagcxDeviceMemoryOrder_t Order = flagcxDeviceMemoryOrderWeak,
      flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem,
      flagcxDeviceMemoryAccessKind_t Kind = flagcxDeviceMemoryAccessKindNormal,
      flagcxDeviceLoadIntent_t Intent = flagcxDeviceLoadIntentOrdinary,
      flagcxDeviceCacheHint_t CacheHint = flagcxDeviceCacheHintNone,
      typename Coop>
  static FLAGCX_DEVICE_INLINE_DECORATOR T load(Coop coop, const T *src) {
    (void)coop;
    return Platform::Intrin::template load<T, Shape, Space, Order, Scope, Kind,
                                           Intent, CacheHint>(src);
  }

  template <
      typename T, typename Shape, flagcxDeviceMemorySpace_t Space,
      flagcxDeviceMemoryOrder_t Order = flagcxDeviceMemoryOrderWeak,
      flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem,
      flagcxDeviceMemoryAccessKind_t Kind = flagcxDeviceMemoryAccessKindNormal,
      flagcxDeviceCacheHint_t CacheHint = flagcxDeviceCacheHintNone,
      typename Coop>
  static FLAGCX_DEVICE_INLINE_DECORATOR void store(Coop coop, T *dst, T value) {
    (void)coop;
    Platform::Intrin::template store<T, Shape, Space, Order, Scope, Kind,
                                     CacheHint>(dst, value);
  }

  // Count is a number of T values. Each rank copies its own strided subset.
  template <typename T, typename SrcShape, typename DstShape,
            flagcxDeviceMemorySpace_t SrcSpace,
            flagcxDeviceMemorySpace_t DstSpace,
            flagcxDeviceMemoryOrder_t SrcOrder = flagcxDeviceMemoryOrderWeak,
            flagcxDeviceMemoryOrder_t DstOrder = flagcxDeviceMemoryOrderWeak,
            flagcxDeviceScope_t SrcScope = flagcxDeviceScopeSystem,
            flagcxDeviceScope_t DstScope = flagcxDeviceScopeSystem,
            flagcxDeviceMemoryAccessKind_t SrcKind =
                flagcxDeviceMemoryAccessKindNormal,
            flagcxDeviceMemoryAccessKind_t DstKind =
                flagcxDeviceMemoryAccessKindNormal,
            flagcxDeviceLoadIntent_t SrcIntent = flagcxDeviceLoadIntentOrdinary,
            flagcxDeviceCacheHint_t SrcCacheHint = flagcxDeviceCacheHintNone,
            flagcxDeviceCacheHint_t DstCacheHint = flagcxDeviceCacheHintNone,
            typename Coop, typename Count>
  static FLAGCX_DEVICE_INLINE_DECORATOR void copy(Coop coop, T *dst,
                                                  const T *src, Count count) {
    static_assert(std::is_integral<Count>::value, "count must be integral");
    static_assert((SrcSpace != flagcxDeviceMemoryLocal &&
                   DstSpace != flagcxDeviceMemoryLocal) ||
                      std::is_same<Coop, Platform::CoopThread>::value,
                  "Local cooperative copy requires CoopThread");
    if (negativeCount(count, typename std::is_signed<Count>::type()))
      reject();
    const size_t n = static_cast<size_t>(count);
    if (n > SIZE_MAX / sizeof(T) || coop.size() <= 0 || coop.threadRank() < 0 ||
        coop.threadRank() >= coop.size())
      reject();
    const size_t step = static_cast<size_t>(coop.size());
    for (size_t i = static_cast<size_t>(coop.threadRank()); i < n;) {
      Platform::Intrin::template copy<T, SrcShape, DstShape, SrcSpace, DstSpace,
                                      SrcOrder, DstOrder, SrcScope, DstScope,
                                      SrcKind, DstKind, SrcIntent, SrcCacheHint,
                                      DstCacheHint>(dst + i, src + i);
      if (n - i <= step)
        break;
      i += step;
    }
  }

  // Preserve the existing arbitrary-size byte-copy form for callers that do
  // not select a PTX shape. New typed copies use Intrin::copy above.
  template <typename Coop, typename DstPtr, typename SrcPtr, typename Count>
  static FLAGCX_DEVICE_INLINE_DECORATOR void copy(Coop coop, DstPtr dst,
                                                  SrcPtr src, Count count) {
    static_assert(std::is_pointer<DstPtr>::value &&
                      std::is_pointer<SrcPtr>::value,
                  "copy requires typed source and destination pointers");
    using Dst = typename std::remove_cv<
        typename std::remove_pointer<DstPtr>::type>::type;
    using Src = typename std::remove_cv<
        typename std::remove_pointer<SrcPtr>::type>::type;
    static_assert(std::is_same<Dst, Src>::value &&
                      std::is_trivially_copyable<Dst>::value,
                  "copy requires equal, trivially copyable element types");
    static_assert(std::is_integral<Count>::value, "count must be integral");
    static_assert(
        !std::is_const<typename std::remove_pointer<DstPtr>::type>::value &&
            !std::is_volatile<
                typename std::remove_pointer<DstPtr>::type>::value &&
            !std::is_volatile<
                typename std::remove_pointer<SrcPtr>::type>::value,
        "copy requires non-const, non-volatile destination and source");
    if (negativeCount(count, typename std::is_signed<Count>::type()))
      reject();
    const size_t n = static_cast<size_t>(count);
    if (n > SIZE_MAX / sizeof(Dst) || coop.size() <= 0 ||
        coop.threadRank() < 0 || coop.threadRank() >= coop.size())
      reject();
    Platform::Intrin::copyBytesStridedImpl(dst, src, n * sizeof(Dst),
                                           coop.threadRank(), coop.size());
  }

  template <typename Coop, typename T>
  static FLAGCX_DEVICE_INLINE_DECORATOR T broadcast(Coop coop, T value,
                                                    int root) {
    static_assert(std::is_trivially_copyable<T>::value && sizeof(T) <= 8,
                  "broadcast supports trivially copyable values up to 8 bytes");
    return broadcastImpl(coop, value, root,
                         flagcxNvidiaDetail::FlagcxNvidiaWarpCoop<Coop>());
  }

  template <typename Coop, typename T, typename RedOp>
  static FLAGCX_DEVICE_INLINE_DECORATOR T reduce(Coop coop, T value, RedOp op) {
    return reduceImpl(coop, value, op,
                      flagcxNvidiaDetail::FlagcxNvidiaWarpCoop<Coop>());
  }

  template <typename Coop>
  static FLAGCX_DEVICE_INLINE_DECORATOR void
  fence(Coop coop, flagcxDeviceMemoryOrder_t order, flagcxDeviceScope_t scope) {
    (void)coop;
    if (scope < flagcxDeviceScopeSystem || scope > flagcxDeviceScopeThread)
      reject();
    if (order == flagcxDeviceMemoryOrderRelaxed)
      return;
    if (order < flagcxDeviceMemoryOrderAcquire ||
        order > flagcxDeviceMemoryOrderSeqCst)
      reject();
#if defined(__CUDA_ARCH__)
    switch (scope) {
      case flagcxDeviceScopeSystem:
        __threadfence_system();
        return;
      case flagcxDeviceScopeDevice:
        __threadfence();
        return;
      case flagcxDeviceScopeBlock:
        __threadfence_block();
        return;
      case flagcxDeviceScopeThread:
        asm volatile("" ::: "memory");
        return;
      default:
        reject();
    }
#else
    reject();
#endif
  }

  // The only first-version async copy is CTA shared -> global bulk copy.
  // Callers synchronize shared-memory producers before issuing, and synchronize
  // the Coop separately after every issuing thread has waited on its batch.
  template <typename Coop, typename DstPtr, typename SrcPtr, typename Count,
            flagcxDeviceMemorySpace_t DstSpace = flagcxDeviceMemoryGlobal,
            flagcxDeviceMemorySpace_t SrcSpace = flagcxDeviceMemoryScratchpad>
  static FLAGCX_DEVICE_INLINE_DECORATOR void
  copyAsync(Coop coop, AsyncCopyBatch &batch, DstPtr dst, SrcPtr src,
            Count count) {
    static_assert(DstSpace == flagcxDeviceMemoryGlobal &&
                      SrcSpace == flagcxDeviceMemoryScratchpad,
                  "copyAsync supports Scratchpad to Global only");
    static_assert(std::is_pointer<DstPtr>::value &&
                      std::is_pointer<SrcPtr>::value &&
                      std::is_integral<Count>::value,
                  "copyAsync requires pointers and an integral count");
    using Dst = typename std::remove_cv<
        typename std::remove_pointer<DstPtr>::type>::type;
    using Src = typename std::remove_cv<
        typename std::remove_pointer<SrcPtr>::type>::type;
    static_assert(std::is_same<Dst, Src>::value &&
                      std::is_trivially_copyable<Dst>::value,
                  "copyAsync element types must match");
    static_assert(
        !std::is_const<typename std::remove_pointer<DstPtr>::type>::value,
        "copyAsync destination must be writable");
    static_assert(
        !std::is_volatile<typename std::remove_pointer<DstPtr>::type>::value &&
            !std::is_volatile<
                typename std::remove_pointer<SrcPtr>::type>::value,
        "copyAsync does not accept volatile pointers");
#if defined(__CUDA_ARCH__)
    static_assert(__CUDA_ARCH__ >= 900 || sizeof(Coop) == 0,
                  "copyAsync bulk_group requires sm_90 or newer");
#endif
    if (negativeCount(count, typename std::is_signed<Count>::type()))
      reject();
    const size_t n = static_cast<size_t>(count);
    if (n > SIZE_MAX / sizeof(Dst) || batch.committed || coop.size() <= 0 ||
        coop.threadRank() < 0 || coop.threadRank() >= coop.size())
      reject();
    const size_t bytes = n * sizeof(Dst);
    if ((bytes & 15) != 0 || (reinterpret_cast<uintptr_t>(dst) & 15) != 0 ||
        (reinterpret_cast<uintptr_t>(src) & 15) != 0)
      reject();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const char *s = reinterpret_cast<const char *>(src);
    char *d = reinterpret_cast<char *>(dst);
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    for (size_t chunk = static_cast<size_t>(coop.threadRank());
         chunk < bytes / 16; chunk += static_cast<size_t>(coop.size())) {
      const uint32_t sharedAddr =
          static_cast<uint32_t>(__cvta_generic_to_shared(s + chunk * 16));
      asm volatile("cp.async.bulk.global.shared::cta.bulk_group "
                   "[%0], [%1], 16;"
                   :
                   : "l"(d + chunk * 16), "r"(sharedAddr)
                   : "memory");
      batch.issued = true;
    }
#else
    reject();
#endif
  }

  template <typename Coop>
  static FLAGCX_DEVICE_INLINE_DECORATOR void
  commitAsync(Coop coop, AsyncCopyBatch &batch) {
    (void)coop;
    if (batch.committed)
      reject();
#if defined(__CUDA_ARCH__)
    static_assert(__CUDA_ARCH__ >= 900 || sizeof(Coop) == 0,
                  "commitAsync bulk_group requires sm_90 or newer");
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    batch.committed = true;
#else
    reject();
#endif
  }

  template <typename Coop>
  static FLAGCX_DEVICE_INLINE_DECORATOR void waitAsync(Coop coop,
                                                       AsyncCopyBatch &batch) {
    (void)coop;
    if (!batch.committed)
      reject();
#if defined(__CUDA_ARCH__)
    static_assert(__CUDA_ARCH__ >= 900 || sizeof(Coop) == 0,
                  "waitAsync bulk_group requires sm_90 or newer");
    asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
    batch.issued = false;
    batch.committed = false;
#else
    reject();
#endif
  }

  // st.async.global has no bulk-group completion and is not waitAsync-able.
  template <
      typename Coop, typename Ptr, typename T,
      flagcxDeviceMemorySpace_t Space = flagcxDeviceMemoryGlobal,
      flagcxDeviceMemoryAccessKind_t Kind = flagcxDeviceMemoryAccessKindNormal,
      flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
  static FLAGCX_DEVICE_INLINE_DECORATOR void
  storeAsync(Coop coop, Ptr dst, T value, flagcxDeviceMemoryOrder_t order) {
    (void)coop;
    (void)dst;
    (void)value;
    static_assert(Space == flagcxDeviceMemoryGlobal &&
                      Scope == flagcxDeviceScopeSystem,
                  "storeAsync supports system-scoped Global addresses only");
    static_assert(Kind == flagcxDeviceMemoryAccessKindNormal ||
                      Kind == flagcxDeviceMemoryAccessKindMmio,
                  "invalid storeAsync access kind");
    static_assert(std::is_pointer<Ptr>::value &&
                      (std::is_same<T, uint32_t>::value ||
                       std::is_same<T, uint64_t>::value),
                  "storeAsync supports uint32_t and uint64_t values");
    static_assert(
        std::is_same<typename std::remove_pointer<Ptr>::type, T>::value,
        "storeAsync destination and value types must match");
#if defined(__CUDA_ARCH__)
    static_assert(__CUDA_ARCH__ >= 1000 || sizeof(Ptr) == 0,
                  "global storeAsync requires sm_100 or newer");
#if __CUDACC_VER_MAJOR__ < 12 ||                                               \
    (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 8)
    static_assert(sizeof(Ptr) == 0,
                  "global storeAsync requires a toolkit with PTX ISA 8.7");
#endif
#endif
    if (order != flagcxDeviceMemoryOrderRelease)
      reject();
    if ((reinterpret_cast<uintptr_t>(dst) & (sizeof(T) - 1)) != 0)
      reject();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    if (Kind == flagcxDeviceMemoryAccessKindMmio) {
      if (sizeof(T) == 4)
        asm volatile("st.async.mmio.release.sys.global.u32 [%0], %1;"
                     :
                     : "l"(dst), "r"(static_cast<uint32_t>(value))
                     : "memory");
      else
        asm volatile("st.async.mmio.release.sys.global.u64 [%0], %1;"
                     :
                     : "l"(dst), "l"(static_cast<uint64_t>(value))
                     : "memory");
    } else {
      if (sizeof(T) == 4)
        asm volatile("st.async.release.sys.global.u32 [%0], %1;"
                     :
                     : "l"(dst), "r"(static_cast<uint32_t>(value))
                     : "memory");
      else
        asm volatile("st.async.release.sys.global.u64 [%0], %1;"
                     :
                     : "l"(dst), "l"(static_cast<uint64_t>(value))
                     : "memory");
    }
#else
    reject();
#endif
  }

  template <
      typename Coop, typename Ptr, typename T,
      flagcxDeviceMemorySpace_t Space = flagcxDeviceMemoryGlobal,
      flagcxDeviceMemoryAccessKind_t Kind = flagcxDeviceMemoryAccessKindNormal,
      flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem,
      flagcxDeviceMemoryOrder_t Order>
  static FLAGCX_DEVICE_INLINE_DECORATOR void
  storeAsync(Coop coop, Ptr dst, T value,
             std::integral_constant<flagcxDeviceMemoryOrder_t, Order>) {
    static_assert(Order == flagcxDeviceMemoryOrderRelease,
                  "storeAsync requires release order");
    storeAsync<Coop, Ptr, T, Space, Kind, Scope>(coop, dst, value, Order);
  }

private:
  static FLAGCX_DEVICE_INLINE_DECORATOR void reject() {
#if defined(__CUDA_ARCH__)
    __trap();
#else
    assert(false && "unsupported NVIDIA CoopOps operation");
#endif
  }

  template <typename Count>
  static FLAGCX_HOST_DEVICE_INLINE bool negativeCount(Count count,
                                                      std::true_type) {
    return count < 0;
  }

  template <typename Count>
  static FLAGCX_HOST_DEVICE_INLINE bool negativeCount(Count, std::false_type) {
    return false;
  }

  template <typename Coop, typename T>
  static FLAGCX_DEVICE_INLINE_DECORATOR T broadcastImpl(Coop coop, T value,
                                                        int root,
                                                        std::true_type) {
    if (root < 0 || root >= coop.size())
      reject();
#if defined(__CUDA_ARCH__)
    const uint32_t mask = static_cast<uint32_t>(groupMask(coop));
    const int sourceLane = laneAtRank(mask, root);
    uint32_t words[2] = {0, 0};
    __builtin_memcpy(words, &value, sizeof(T));
    words[0] = __shfl_sync(mask, words[0], sourceLane);
    if (sizeof(T) > 4)
      words[1] = __shfl_sync(mask, words[1], sourceLane);
    T result;
    __builtin_memcpy(&result, words, sizeof(T));
    return result;
#else
    reject();
    return value;
#endif
  }

  template <typename Coop, typename T>
  static FLAGCX_DEVICE_INLINE_DECORATOR T broadcastImpl(Coop, T value, int,
                                                        std::false_type) {
    // TODO: multi-warp register exchange needs caller-provided scratch.
    // In particular, CoopAny exposes only rank, size and sync.
    reject();
    return value;
  }

  template <typename Coop, typename T, typename RedOp>
  static FLAGCX_DEVICE_INLINE_DECORATOR T reduceImpl(Coop coop, T value,
                                                     RedOp op, std::true_type) {
    T result = broadcast(coop, value, 0);
    for (int root = 1; root < coop.size(); ++root)
      result = op(result, broadcast(coop, value, root));
    return result;
  }

  template <typename Coop, typename T, typename RedOp>
  static FLAGCX_DEVICE_INLINE_DECORATOR T reduceImpl(Coop, T value, RedOp,
                                                     std::false_type) {
    // TODO: implement an explicit-scratch overload for multi-warp groups.
    reject();
    return value;
  }

  template <typename Coop>
  static FLAGCX_DEVICE_INLINE_DECORATOR flagcxLaneMask_t groupMask(Coop coop) {
    return groupMaskImpl(coop,
                         flagcxNvidiaDetail::FlagcxNvidiaHasLaneMask<Coop>());
  }

  template <typename Coop>
  static FLAGCX_DEVICE_INLINE_DECORATOR flagcxLaneMask_t
  groupMaskImpl(Coop coop, std::true_type) {
    return coop.laneMask();
  }

  template <typename Coop>
  static FLAGCX_DEVICE_INLINE_DECORATOR flagcxLaneMask_t
  groupMaskImpl(Coop coop, std::false_type) {
    return coop.getLmask();
  }

  static FLAGCX_DEVICE_INLINE_DECORATOR int laneAtRank(uint32_t mask,
                                                       int rank) {
    for (int lane = 0; lane < 32; ++lane) {
      if (mask & (uint32_t{1} << lane)) {
        if (rank-- == 0)
          return lane;
      }
    }
    reject();
    return 0;
  }
};

#endif // FLAGCX_NVIDIA_PLATFORM_TRAITS_H_
