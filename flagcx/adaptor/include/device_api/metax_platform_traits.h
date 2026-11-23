/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Provides PlatformTraits<MetaxPlatform> with:
 *   - Intrin: lane(), activemask(), syncwarp(), popc(), spinBackoff(), ...
 *   - Atomic: load(), store(), fetchAdd(), compareExchange(), ...
 ************************************************************************/

#ifndef FLAGCX_METAX_PLATFORM_TRAITS_H_
#define FLAGCX_METAX_PLATFORM_TRAITS_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <mcr/mc_runtime.h>

struct MetaxPlatform {};

template <>
struct PlatformTraits<MetaxPlatform> {
  struct CoopOps {};
  struct Clock {};

  // ==============================================================
  // Intrin — MACA 64-lane SIMT intrinsics
  // ==============================================================
  struct Intrin {
    static constexpr int simtWidth = FLAGCX_SIMT_WIDTH;
    static FLAGCX_HOST_DEVICE_INLINE constexpr flagcxLaneMask_t fullMask() {
      return 0xffffffffffffffffull;
    }

#if defined(FLAGCX_METAX_DEVICE_COMPILE)
    static FLAGCX_DEVICE_INLINE_DECORATOR int lane() {
      return static_cast<int>(FLAGCX_THREAD_IDX_X & (simtWidth - 1));
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR flagcxLaneMask_t lanemaskLt() {
      const unsigned id = static_cast<unsigned>(lane());
      return id == 0 ? 0ull : ((flagcxLaneMask_t{1} << id) - 1ull);
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR flagcxLaneMask_t activemask() {
      static_assert(sizeof(decltype(__activemask())) ==
                        sizeof(flagcxLaneMask_t),
                    "MACA active masks must include lanes 32-63");
      return static_cast<flagcxLaneMask_t>(__activemask());
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR void
    validateMask(flagcxLaneMask_t mask) {
      if ((mask & ~fullMask()) != 0)
        FLAGCX_DEVICE_TRAP();
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR void
    syncwarp(flagcxLaneMask_t mask = fullMask()) {
      validateMask(mask);
      if ((mask & (flagcxLaneMask_t{1} << lane())) == 0)
        FLAGCX_DEVICE_TRAP();
      __syncwarp(static_cast<unsigned long long>(mask));
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR int popc(flagcxLaneMask_t x) {
      validateMask(x);
      return __popcll(static_cast<unsigned long long>(x));
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR void namedBarrierSync(int id,
                                                                int nThreads) {
      (void)id;
      if (nThreads == simtWidth) {
        syncwarp();
      } else if (nThreads == static_cast<int>(blockDim.x)) {
        __syncthreads();
      } else {
        // Only a wave or a whole CTA can be synchronized here. Fail explicitly
        // rather than silently skipping the synchronization.
        FLAGCX_DEVICE_TRAP();
      }
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR void spinBackoff(int iter) {
      int delay = 1 << (iter < 15 ? iter : 15);
      uint64_t start = clock64();
      while (clock64() - start < (uint64_t)delay) { /* spin */
      }
    }

    // System-level fence: ensures stores are visible across all system agents
    // (including other processes via IPC).
    static FLAGCX_DEVICE_INLINE_DECORATOR void threadfenceSystem() {
      __threadfence_system();
    }

    // Device-level fence: ordering within this GPU only.
    static FLAGCX_DEVICE_INLINE_DECORATOR void threadfenceDevice() {
      __threadfence();
    }

    // Cooperative strided copy between two device-memory buffers. Cascades
    // from 16B vectors (int4) down to byte level.
    template <typename DstPtr, typename SrcPtr>
    static FLAGCX_DEVICE_INLINE_DECORATOR void
    coopCopyBytes(DstPtr dstIn, SrcPtr srcIn, size_t bytes, int rank,
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
      assert((mask & ~fullMask()) == 0 && "lane mask exceeds MetaX SIMT width");
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
    static inline void coopCopyBytes(DstPtr, SrcPtr, size_t, int, int) {
      assert(false && "coopCopyBytes() called on host");
    }
#endif // FLAGCX_METAX_DEVICE_COMPILE
  };

  // ==============================================================
  // Atomic — unscoped LLVM atomics retain system syncscope. Use seq_cst
  // conservatively on every scope; system operations also explicitly fence
  // payloads shared through IPC or the host-mapped FIFO. Peer allocations
  // must use the MACA finegrained policy in maca_adaptor.cc.
  // ==============================================================
  struct Atomic {
  private:
    template <flagcxDeviceScope_t Scope>
    static FLAGCX_DEVICE_INLINE_DECORATOR void fence() {
#if defined(FLAGCX_METAX_DEVICE_COMPILE)
      if (Scope == flagcxDeviceScopeSystem)
        Intrin::threadfenceSystem();
      else if (Scope == flagcxDeviceScopeDevice)
        Intrin::threadfenceDevice();
      else if (Scope == flagcxDeviceScopeBlock)
        __threadfence_block();
#endif
    }

  public:
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T load(T *ptr,
                                                 flagcxDeviceMemoryOrder_t) {
      T value = __atomic_load_n(ptr, __ATOMIC_SEQ_CST);
      fence<Scope>();
      return value;
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR void
    store(T *ptr, const T &value, flagcxDeviceMemoryOrder_t) {
      fence<Scope>();
      __atomic_store_n(ptr, value, __ATOMIC_SEQ_CST);
      fence<Scope>();
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T
    fetchAdd(T *ptr, const T &value, flagcxDeviceMemoryOrder_t) {
      fence<Scope>();
      T previous = __atomic_fetch_add(ptr, value, __ATOMIC_SEQ_CST);
      fence<Scope>();
      return previous;
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T fetchOr(T *ptr, const T &value,
                                                    flagcxDeviceMemoryOrder_t) {
      fence<Scope>();
      T previous = __atomic_fetch_or(ptr, value, __ATOMIC_SEQ_CST);
      fence<Scope>();
      return previous;
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T
    fetchAnd(T *ptr, const T &value, flagcxDeviceMemoryOrder_t) {
      fence<Scope>();
      T previous = __atomic_fetch_and(ptr, value, __ATOMIC_SEQ_CST);
      fence<Scope>();
      return previous;
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T
    exchange(T *ptr, const T &value, flagcxDeviceMemoryOrder_t) {
      fence<Scope>();
      T previous = __atomic_exchange_n(ptr, value, __ATOMIC_SEQ_CST);
      fence<Scope>();
      return previous;
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T
    fetchSub(T *ptr, const T &value, flagcxDeviceMemoryOrder_t order) {
      return fetchAdd<T, Scope>(ptr, T{} - value, order);
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR bool
    compareExchange(T *ptr, T &expected, const T &desired,
                    flagcxDeviceMemoryOrder_t) {
      fence<Scope>();
      bool success = __atomic_compare_exchange_n(
          ptr, &expected, desired, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
      fence<Scope>();
      return success;
    }
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

#endif // FLAGCX_METAX_PLATFORM_TRAITS_H_
