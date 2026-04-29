/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Default Platform Traits — host-only stubs for non-NVIDIA/DU backends.
 *
 * Provides PlatformTraits<DefaultPlatform> with:
 *   - Intrin: assert-false stubs (never called; IPC path is host-only)
 *   - Atomic: GCC built-in atomics (__atomic_*)
 *   - Coop*:  single-thread no-op implementations
 *
 * Used by CommTraits<Default<DefaultPlatform>> when neither
 * USE_NVIDIA_ADAPTOR nor USE_DU_ADAPTOR is defined.
 ************************************************************************/

#ifndef FLAGCX_DEFAULT_PLATFORM_TRAITS_H_
#define FLAGCX_DEFAULT_PLATFORM_TRAITS_H_

#include <cassert>
#include <cstdint>

struct DefaultPlatform {};

template <>
struct PlatformTraits<DefaultPlatform> {
  // ==============================================================
  // Intrin — assert-false stubs (IPC fallback never calls these)
  // ==============================================================
  struct Intrin {
    static constexpr int simtWidth = 1;
    static inline int lane() {
      assert(false && "lane() on DefaultPlatform");
      return 0;
    }
    static inline uint32_t lanemaskLt() {
      assert(false && "lanemaskLt() on DefaultPlatform");
      return 0;
    }
    static inline uint32_t activemask() {
      assert(false && "activemask() on DefaultPlatform");
      return 1;
    }
    static inline void syncwarp(uint32_t = 0xffffffffu) {
      assert(false && "syncwarp() on DefaultPlatform");
    }
    static inline int popc(uint32_t x) {
      (void)x;
      assert(false && "popc() on DefaultPlatform");
      return 0;
    }
    static inline void namedBarrierSync(int, int) {
      assert(false && "namedBarrierSync() on DefaultPlatform");
    }
    static inline void spinBackoff(int) {
      assert(false && "spinBackoff() on DefaultPlatform");
    }
    static inline void threadfenceSystem() {
      assert(false && "threadfenceSystem() on DefaultPlatform");
    }
  };

  // ==============================================================
  // Atomic — GCC built-in atomics
  // ==============================================================
  struct Atomic {
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline T load(T *ptr, flagcxDeviceMemoryOrder_t) {
      return __atomic_load_n(ptr, __ATOMIC_SEQ_CST);
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline void store(T *ptr, const T &val, flagcxDeviceMemoryOrder_t) {
      __atomic_store_n(ptr, val, __ATOMIC_SEQ_CST);
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline T fetchAdd(T *ptr, const T &val, flagcxDeviceMemoryOrder_t) {
      return __atomic_fetch_add(ptr, val, __ATOMIC_SEQ_CST);
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline T fetchSub(T *ptr, const T &val, flagcxDeviceMemoryOrder_t) {
      return __atomic_fetch_sub(ptr, val, __ATOMIC_SEQ_CST);
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline T fetchOr(T *ptr, const T &val, flagcxDeviceMemoryOrder_t) {
      return __atomic_fetch_or(ptr, val, __ATOMIC_SEQ_CST);
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline T fetchAnd(T *ptr, const T &val, flagcxDeviceMemoryOrder_t) {
      return __atomic_fetch_and(ptr, val, __ATOMIC_SEQ_CST);
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline T exchange(T *ptr, const T &val, flagcxDeviceMemoryOrder_t) {
      return __atomic_exchange_n(ptr, val, __ATOMIC_SEQ_CST);
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline bool compareExchange(T *ptr, T &expected, const T &desired,
                                       flagcxDeviceMemoryOrder_t) {
      return __atomic_compare_exchange_n(ptr, &expected, desired, false,
                                         __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    }
  };

  // ==============================================================
  // Coop types — single-thread no-ops
  // ==============================================================
  struct CoopBlock {
    int threadRank() const { return 0; }
    int size() const { return 1; }
    void sync() {}
  };
  template <int N>
  struct CoopTile {
    int threadRank() const { return 0; }
    int size() const { return N; }
    void sync() {}
  };
  using CoopThread = CoopTile<1>;
  using CoopWarp = CoopTile<1>;
  struct CoopTileSpan {
    CoopTileSpan(int, int, int) {}
    int threadRank() const { return 0; }
    int size() const { return 1; }
    void sync() {}
  };
  struct CoopLanes {
    CoopLanes(uint32_t = 1u) {}
    int threadRank() const { return 0; }
    int size() const { return 1; }
    void sync() {}
  };
  using CoopAny = PlatformCoop;
};

#endif // FLAGCX_DEFAULT_PLATFORM_TRAITS_H_
