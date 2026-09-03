/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * KunlunXin/XPU platform traits for the common FlagCX Device API.
 ************************************************************************/

#ifndef FLAGCX_KUNLUNXIN_PLATFORM_TRAITS_H_
#define FLAGCX_KUNLUNXIN_PLATFORM_TRAITS_H_

struct KunlunxinPlatform {};

template <>
struct PlatformTraits<KunlunxinPlatform> {
  struct Intrin {
    static constexpr int simtWidth = FLAGCX_SIMT_WIDTH;

    FLAGCX_DEVICE_INLINE_DECORATOR static uint32_t fullMask() {
      int n = FLAGCX_BLOCK_DIM_X;
      return n >= 32 ? 0xffffffffu : ((1u << n) - 1u);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static int lane() {
      return FLAGCX_THREAD_IDX_X;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static uint32_t lanemaskLt() {
      int id = FLAGCX_THREAD_IDX_X;
      if (id <= 0)
        return 0;
      if (id >= 32)
        return 0xffffffffu;
      return (1u << id) - 1u;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static uint32_t activemask() {
      // XTDK does not provide a reliable divergent-control-flow active mask.
      // Callers such as flagcxCoopCoalesced must fail explicitly instead of
      // manufacturing a full mask that can deadlock at a cluster barrier.
      __builtin_trap();
      return 0;
    }

    // XSHMEM exposes cluster synchronization, but no partial-core named
    // barrier. Never silently weaken a requested subgroup barrier.
    FLAGCX_DEVICE_INLINE_DECORATOR static void
    syncwarp(uint32_t mask = 0xffffffffu) {
      uint32_t active = fullMask();
      if ((mask & active) == active)
        FLAGCX_DEVICE_SYNC_THREADS();
      else if (popc(mask & active) > 1)
        __builtin_trap();
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static int popc(uint32_t x) {
      return __builtin_popcount(x);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static void namedBarrierSync(int, int n) {
      if (n >= FLAGCX_BLOCK_DIM_X)
        FLAGCX_DEVICE_SYNC_THREADS();
      else if (n > 1)
        __builtin_trap();
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static void spinBackoff(int) {}

    FLAGCX_DEVICE_INLINE_DECORATOR static void threadfenceSystem() {
      FLAGCX_DEVICE_THREAD_FENCE();
    }
  };

  struct Atomic {
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    FLAGCX_DEVICE_INLINE_DECORATOR static T load(T *ptr,
                                                 flagcxDeviceMemoryOrder_t) {
      return __atomic_load_n(ptr, __ATOMIC_ACQUIRE);
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    FLAGCX_DEVICE_INLINE_DECORATOR static void
    store(T *ptr, const T &value, flagcxDeviceMemoryOrder_t) {
      __atomic_store_n(ptr, value, __ATOMIC_RELEASE);
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    FLAGCX_DEVICE_INLINE_DECORATOR static T
    fetchAdd(T *ptr, const T &value, flagcxDeviceMemoryOrder_t) {
      return __atomic_fetch_add(ptr, value, __ATOMIC_ACQ_REL);
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    FLAGCX_DEVICE_INLINE_DECORATOR static T
    fetchSub(T *ptr, const T &value, flagcxDeviceMemoryOrder_t) {
      return __atomic_fetch_sub(ptr, value, __ATOMIC_ACQ_REL);
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    FLAGCX_DEVICE_INLINE_DECORATOR static T fetchOr(T *ptr, const T &value,
                                                    flagcxDeviceMemoryOrder_t) {
      return __atomic_fetch_or(ptr, value, __ATOMIC_ACQ_REL);
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    FLAGCX_DEVICE_INLINE_DECORATOR static T
    fetchAnd(T *ptr, const T &value, flagcxDeviceMemoryOrder_t) {
      return __atomic_fetch_and(ptr, value, __ATOMIC_ACQ_REL);
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    FLAGCX_DEVICE_INLINE_DECORATOR static T
    exchange(T *ptr, const T &value, flagcxDeviceMemoryOrder_t) {
      return __atomic_exchange_n(ptr, value, __ATOMIC_ACQ_REL);
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    FLAGCX_DEVICE_INLINE_DECORATOR static bool
    compareExchange(T *ptr, T &expected, const T &desired,
                    flagcxDeviceMemoryOrder_t) {
      return __atomic_compare_exchange_n(ptr, &expected, desired, false,
                                         __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
    }
  };

  struct CoopBlock {
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return FLAGCX_THREAD_IDX_X;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const {
      return FLAGCX_BLOCK_DIM_X;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR uint32_t laneMask() const {
      return Intrin::fullMask();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() const {
      FLAGCX_DEVICE_SYNC_THREADS();
    }
  };

  template <int N>
  struct CoopTile {
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return FLAGCX_THREAD_IDX_X % N;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return N; }
    FLAGCX_DEVICE_INLINE_DECORATOR uint32_t laneMask() const {
      int base = (FLAGCX_THREAD_IDX_X / N) * N;
      uint32_t bits = N >= 32 ? 0xffffffffu : ((1u << N) - 1u);
      return base >= 32 ? 0u : (bits << base) & Intrin::fullMask();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() const {
      if (N >= FLAGCX_BLOCK_DIM_X)
        FLAGCX_DEVICE_SYNC_THREADS();
      else if (N > 1)
        __builtin_trap();
    }
  };

  using CoopThread = CoopTile<1>;
  using CoopWarp = CoopTile<FLAGCX_SIMT_WIDTH>;

  struct CoopTileSpan {
    int first;
    int count;
    int barrierId;

    FLAGCX_DEVICE_INLINE_DECORATOR CoopTileSpan(int t0, int nTiles, int id)
        : first(t0), count(nTiles), barrierId(id) {}
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return FLAGCX_THREAD_IDX_X - first * FLAGCX_SIMT_WIDTH;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const {
      return count * FLAGCX_SIMT_WIDTH;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() const {
      if (size() >= FLAGCX_BLOCK_DIM_X)
        FLAGCX_DEVICE_SYNC_THREADS();
      else if (size() > 1)
        __builtin_trap();
    }
  };

  struct CoopLanes {
    uint32_t mask;

    FLAGCX_DEVICE_INLINE_DECORATOR explicit CoopLanes(uint32_t m = 1u)
        : mask(m & Intrin::fullMask()) {}
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return Intrin::popc(mask & Intrin::lanemaskLt());
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const {
      return Intrin::popc(mask);
    }
    FLAGCX_DEVICE_INLINE_DECORATOR uint32_t getLmask() const { return mask; }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() const { Intrin::syncwarp(mask); }
  };

  using CoopAny = PlatformCoop;
};

#endif // FLAGCX_KUNLUNXIN_PLATFORM_TRAITS_H_
