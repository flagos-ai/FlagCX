/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 * See LICENSE-NCCL.txt for NCCL license information
 *
 * NVIDIA Platform Traits — CUDA SIMT intrinsics and cuda::atomic_ref.
 *
 * Provides PlatformTraits<NvidiaPlatform> with:
 *   - Intrin: lane(), activemask(), syncwarp(), popc(), spinBackoff(), ...
 *   - Atomic: load(), store(), fetchAdd(), compareExchange(), ...
 *
 * Both __CUDACC__ (device) and host-compiler paths are NVIDIA-specific;
 * the #ifdef __CUDACC__ split is device-vs-host, NOT vendor-vs-fallback.
 ************************************************************************/

#ifndef FLAGCX_NVIDIA_PLATFORM_TRAITS_H_
#define FLAGCX_NVIDIA_PLATFORM_TRAITS_H_

#include <cassert>
#include <cstdint>
#include <cuda/atomic>

struct NvidiaPlatform {};

template <>
struct PlatformTraits<NvidiaPlatform> {

  // ==============================================================
  // Intrin — CUDA SIMT intrinsics
  // ==============================================================
  struct Intrin {
    static constexpr int simtWidth = 32;

#if defined(__CUDACC__)
    static FLAGCX_DEVICE_INLINE_DECORATOR int lane() {
      int l;
      asm("mov.u32 %0, %%laneid;" : "=r"(l));
      return l;
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t lanemaskLt() {
      uint32_t m;
      asm("mov.u32 %0, %%lanemask_lt;" : "=r"(m));
      return m;
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t activemask() {
      return __activemask();
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR void
    syncwarp(uint32_t mask = 0xffffffffu) {
      __syncwarp(mask);
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR int popc(uint32_t x) {
      return __popc(x);
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

#else
    // Host-compiler stubs (allow template instantiation, never called at
    // runtime)
    static inline int lane() {
      assert(false && "lane() called on host");
      return 0;
    }
    static inline uint32_t lanemaskLt() {
      assert(false && "lanemaskLt() called on host");
      return 0;
    }
    static inline uint32_t activemask() {
      assert(false && "activemask() called on host");
      return 1;
    }
    static inline void syncwarp(uint32_t mask = 0xffffffffu) {
      (void)mask;
      assert(false && "syncwarp() called on host");
    }
    static inline int popc(uint32_t x) {
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
#endif // __CUDACC__
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
      FLAGCX_MAYBE_UNUSED static FLAGCX_DEVICE_CONSTANT_DECORATOR
          cuda::memory_order map[] = {
              cuda::memory_order_relaxed, cuda::memory_order_acquire,
              cuda::memory_order_release, cuda::memory_order_acq_rel,
              cuda::memory_order_seq_cst};
      return map[o];
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
    FLAGCX_DEVICE_INLINE_DECORATOR uint32_t laneMask() const {
      return (0xffffffffu >> (32 - N)) << (Intrin::lane() & -N);
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
    uint32_t lmask;
    FLAGCX_DEVICE_INLINE_DECORATOR CoopLanes(uint32_t lmask = 0xffffffffu)
        : lmask(lmask) {}
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return Intrin::popc(lmask & Intrin::lanemaskLt());
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const {
      return Intrin::popc(lmask);
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() { Intrin::syncwarp(lmask); }
    FLAGCX_DEVICE_INLINE_DECORATOR uint32_t getLmask() const { return lmask; }
  };

  using CoopAny = PlatformCoop;
};

#endif // FLAGCX_NVIDIA_PLATFORM_TRAITS_H_
