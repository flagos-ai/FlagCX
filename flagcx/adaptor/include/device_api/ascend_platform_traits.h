/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Ascend (CANN) Platform Traits — stage-1 host wiring.
 *
 * Provides PlatformTraits<AscendPlatform> with:
 *   - Intrin: assert-false stubs; simtWidth = 1 asserts NO SIMT capability
 *             (per adaptation doc §5.1 a platform must not default to 32
 *             without evidence; the real width lands with the device
 *             implementation below)
 *   - Atomic: GCC built-in atomics (__atomic_*), host-side only for now
 *   - Coop*:  single-thread no-op implementations
 *   - CoopOps/Clock: intentionally empty, following the DU precedent; an
 *     empty type is not a capability claim
 *
 * Used by CommTraits<DefaultBackend<AscendPlatform>> when
 * USE_ASCEND_ADAPTOR is defined. Device-compiler (bisheng) implementations
 * of Intrin/Atomic/Coop are deferred to the device-kernel build stage; they
 * will be added under an explicit device-compile guard once the Ascend
 * toolchain's warp/SIMT capability and cross-process sharing model are
 * validated on hardware. Until then this specialization only supports
 * host-side template instantiation (the Default backend's host control
 * path), never device kernel execution.
 ************************************************************************/

#ifndef FLAGCX_ASCEND_PLATFORM_TRAITS_H_
#define FLAGCX_ASCEND_PLATFORM_TRAITS_H_

#include <cassert>
#include <cstdint>
#include <stddef.h>

struct AscendPlatform {};

template <>
struct PlatformTraits<AscendPlatform> {
  struct CoopOps {};
  struct Clock {};
  // ==============================================================
  // Intrin — host stubs (device kernels not built on Ascend yet)
  // ==============================================================
  struct Intrin {
    static constexpr int simtWidth = 1;
    static constexpr flagcxLaneMask_t fullMask() { return 1ull; }
    static inline int lane() {
      assert(false && "lane() on AscendPlatform");
      return 0;
    }
    static inline flagcxLaneMask_t lanemaskLt() {
      assert(false && "lanemaskLt() on AscendPlatform");
      return 0;
    }
    static inline flagcxLaneMask_t activemask() {
      assert(false && "activemask() on AscendPlatform");
      return 1;
    }
    static inline void syncwarp(flagcxLaneMask_t = fullMask()) {
      assert(false && "syncwarp() on AscendPlatform");
    }
    static inline int popc(flagcxLaneMask_t x) {
      (void)x;
      assert(false && "popc() on AscendPlatform");
      return 0;
    }
    static inline void namedBarrierSync(int, int) {
      assert(false && "namedBarrierSync() on AscendPlatform");
    }
    static inline void spinBackoff(int) {
      assert(false && "spinBackoff() on AscendPlatform");
    }
    static inline void threadfenceSystem() {
      assert(false && "threadfenceSystem() on AscendPlatform");
    }
    static inline void threadfenceDevice() {
      assert(false && "threadfenceDevice() on AscendPlatform");
    }
    template <typename DstPtr, typename SrcPtr>
    static inline void coopCopyBytes(DstPtr, SrcPtr, size_t, int, int) {
      assert(false && "coopCopyBytes() on AscendPlatform");
    }
  };

  // ==============================================================
  // Atomic — GCC built-in atomics (host-side control path)
  // ==============================================================
  struct Atomic {
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline T load(T *ptr, flagcxDeviceMemoryOrder_t) {
      (void)Scope;
      return __atomic_load_n(ptr, __ATOMIC_SEQ_CST);
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline void store(T *ptr, const T &val,
                             flagcxDeviceMemoryOrder_t) {
      (void)Scope;
      __atomic_store_n(ptr, val, __ATOMIC_SEQ_CST);
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline T fetchAdd(T *ptr, const T &val,
                             flagcxDeviceMemoryOrder_t) {
      (void)Scope;
      return __atomic_fetch_add(ptr, val, __ATOMIC_SEQ_CST);
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline T fetchSub(T *ptr, const T &val,
                             flagcxDeviceMemoryOrder_t) {
      (void)Scope;
      return __atomic_fetch_sub(ptr, val, __ATOMIC_SEQ_CST);
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline T fetchOr(T *ptr, const T &val,
                            flagcxDeviceMemoryOrder_t) {
      (void)Scope;
      return __atomic_fetch_or(ptr, val, __ATOMIC_SEQ_CST);
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline T fetchAnd(T *ptr, const T &val,
                             flagcxDeviceMemoryOrder_t) {
      (void)Scope;
      return __atomic_fetch_and(ptr, val, __ATOMIC_SEQ_CST);
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline T exchange(T *ptr, const T &val,
                             flagcxDeviceMemoryOrder_t) {
      (void)Scope;
      return __atomic_exchange_n(ptr, val, __ATOMIC_SEQ_CST);
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline bool compareExchange(T *ptr, T &expected, const T &desired,
                                       flagcxDeviceMemoryOrder_t) {
      (void)Scope;
      return __atomic_compare_exchange_n(ptr, &expected, desired, false,
                                         __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    }
  };

  // ==============================================================
  // Coop types — single-thread no-ops (host instantiation only)
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
    flagcxLaneMask_t laneMask() const { return Intrin::fullMask(); }
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
    CoopLanes(flagcxLaneMask_t = Intrin::fullMask()) {}
    int threadRank() const { return 0; }
    int size() const { return 1; }
    void sync() {}
    flagcxLaneMask_t getLmask() const { return Intrin::fullMask(); }
  };
  using CoopAny = PlatformCoop;
};

#endif // FLAGCX_ASCEND_PLATFORM_TRAITS_H_
