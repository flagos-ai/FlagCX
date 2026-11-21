/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 * XPU NetIO primitive contract implemented by FlagCX NVIDIA PlatformTraits.
 ************************************************************************/
#ifndef FLAGCX_NVIDIA_XPU_NETIO_PLATFORM_CUH_
#define FLAGCX_NVIDIA_XPU_NETIO_PLATFORM_CUH_

#include "device_api/nvidia_platform_traits.h"

struct flagcxXpuNetioNvidiaPlatform {
  using PT = PlatformTraits<NvidiaPlatform>;
  using Atomic = PT::Atomic;
  using Ops = PT::CoopOps;
  using Thread = PT::CoopThread;
  static constexpr uint32_t simtWidth = PT::Intrin::simtWidth;

  template <int Scope>
  static FLAGCX_HOST_DEVICE_INLINE constexpr flagcxDeviceScope_t scope() {
    static_assert(Scope >= 0 && Scope <= 3, "invalid XPU NetIO atomic scope");
    return static_cast<flagcxDeviceScope_t>(Scope);
  }
  template <int Scope, class T>
  static FLAGCX_DEVICE_INLINE_DECORATOR T loadScoped(T *p) {
    return Atomic::load<T, scope<Scope>()>(p, flagcxDeviceMemoryOrderRelaxed);
  }
  template <int Scope, class T>
  static FLAGCX_DEVICE_INLINE_DECORATOR void storeScoped(T *p, T v) {
    Atomic::store<T, scope<Scope>()>(p, v, flagcxDeviceMemoryOrderRelease);
  }
  template <int Scope, class T>
  static FLAGCX_DEVICE_INLINE_DECORATOR T addScoped(T *p, T v) {
    return Atomic::fetchAdd<T, scope<Scope>()>(p, v,
                                               flagcxDeviceMemoryOrderRelaxed);
  }
  template <int Scope, bool Acquire = false, class T>
  static FLAGCX_DEVICE_INLINE_DECORATOR T maxScoped(T *p, T v) {
    return Atomic::fetchMax<T, scope<Scope>()>(
        p, v,
        Acquire ? flagcxDeviceMemoryOrderAcquire
                : flagcxDeviceMemoryOrderRelaxed);
  }
  template <int Scope, class T>
  static FLAGCX_DEVICE_INLINE_DECORATOR T casScoped(T *p, T expected,
                                                    T desired) {
    // FlagCX returns success and updates expected on failure; SDK returns old.
    Atomic::compareExchange<T, scope<Scope>()>(p, expected, desired,
                                               flagcxDeviceMemoryOrderRelaxed);
    return expected;
  }
  template <int Scope, bool Acquire>
  static FLAGCX_DEVICE_INLINE_DECORATOR void fenceScoped() {
    Ops::fence(Thread{},
               Acquire ? flagcxDeviceMemoryOrderAcquire
                       : flagcxDeviceMemoryOrderRelease,
               scope<Scope>());
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR void compilerFence() {
    fenceScoped<3, false>();
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR void acquireSystem() {
    fenceScoped<0, true>();
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR void acquireDevice() {
    fenceScoped<1, true>();
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR void releaseSystem() {
    fenceScoped<0, false>();
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR void releaseDevice() {
    fenceScoped<1, false>();
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t activeMask() {
    return PT::Intrin::activemask();
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t popcount(uint32_t m) {
    return PT::Intrin::popc(m);
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t laneId() {
    return PT::Intrin::lane();
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR void trap() { FLAGCX_DEVICE_TRAP(); }
  static FLAGCX_DEVICE_INLINE_DECORATOR void
  warpSync(uint32_t mask = 0xffffffff) {
    PT::Intrin::syncwarp(mask);
  }
  struct Max {
    FLAGCX_DEVICE_INLINE_DECORATOR uint32_t operator()(uint32_t a,
                                                       uint32_t b) const {
      return a > b ? a : b;
    }
  };
  static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t
  warpMax(uint32_t v, uint32_t mask = 0xffffffff) {
    return Ops::reduce(PT::CoopLanes{mask}, v, Max{});
  }
  template <class T>
  static FLAGCX_DEVICE_INLINE_DECORATOR T
  warpBroadcast(T v, int source, uint32_t mask = 0xffffffff) {
    // SDK source is a physical lane, FlagCX root is a rank within the mask.
    if (source < 0 || source >= 32 || !(mask & (uint32_t{1} << source)))
      trap();
    const int root = popcount(mask & ((uint32_t{1} << source) - 1));
    return Ops::broadcast(PT::CoopLanes{mask}, v, root);
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR uint64_t globalAddress(const void *p) {
    return flagcxNvidiaMemory::Address<flagcxDeviceMemoryGlobal>::get(p);
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t sharedAddress(const void *p) {
    return flagcxNvidiaMemory::Address<flagcxDeviceMemoryScratchpad>::get(p);
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR void copyBytes(void *d, const void *s,
                                                       uint64_t n) {
    Ops::copy(Thread{}, static_cast<uint8_t *>(d),
              static_cast<const uint8_t *>(s), n);
  }
  template <class T>
  static FLAGCX_DEVICE_INLINE_DECORATOR T loadReadOnly(const T *p) {
    return Ops::load<T, Scalar, flagcxDeviceMemoryGlobal,
                     flagcxDeviceMemoryOrderWeak, flagcxDeviceScopeSystem,
                     flagcxDeviceMemoryAccessKindNormal,
                     flagcxDeviceLoadIntentImmutable>(Thread{}, p);
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t funnelRight(uint32_t a,
                                                             uint32_t b,
                                                             uint32_t s) {
    return PT::Intrin::funnelShiftRight(a, b, s);
  }
  template <class T>
  static FLAGCX_DEVICE_INLINE_DECORATOR T loadDma(const T *p) {
    return Ops::load<
        T, Scalar, flagcxDeviceMemoryGlobal, flagcxDeviceMemoryOrderRelaxed,
        flagcxDeviceScopeSystem, flagcxDeviceMemoryAccessKindNormal,
        flagcxDeviceLoadIntentOrdinary, flagcxDeviceCacheHintNoAllocate>(
        Thread{}, p);
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR uint8_t loadDma8(const uint8_t *p) {
    return loadDma(p);
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR uint64_t loadDma64(const uint64_t *p) {
    return loadDma(p);
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR uint32_t
  loadDma32(const volatile uint32_t *p) {
    return loadDma(const_cast<const uint32_t *>(p));
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR void
  storeReleaseDeviceNoAllocate(uint32_t *p, uint32_t v) {
    Ops::store<uint32_t, Scalar, flagcxDeviceMemoryGlobal,
               flagcxDeviceMemoryOrderRelease, flagcxDeviceScopeDevice,
               flagcxDeviceMemoryAccessKindNormal,
               flagcxDeviceCacheHintNoAllocate>(Thread{}, p, v);
  }
  template <class T>
  static FLAGCX_DEVICE_INLINE_DECORATOR T loadDevice(T *p) {
    // Preserve the SDK's RMW progress read, rather than a cached plain load.
    return Atomic::fetchAdd<T, flagcxDeviceScopeDevice>(
        p, T{0}, flagcxDeviceMemoryOrderRelaxed);
  }
  template <class T>
  static FLAGCX_DEVICE_INLINE_DECORATOR T compareExchangeDevice(T *p, T e,
                                                                T d) {
    return casScoped<1>(p, e, d);
  }
  template <class T>
  static FLAGCX_DEVICE_INLINE_DECORATOR T exchangeDevice(T *p, T v) {
    return Atomic::exchange<T, flagcxDeviceScopeDevice>(
        p, v, flagcxDeviceMemoryOrderRelaxed);
  }
  template <class T>
  static FLAGCX_DEVICE_INLINE_DECORATOR T loadSystem(const T *p) {
    return loadScoped<0>(const_cast<T *>(p));
  }
  template <class T>
  static FLAGCX_DEVICE_INLINE_DECORATOR T loadAcquireSystem(const T *p) {
    return Atomic::load(const_cast<T *>(p), flagcxDeviceMemoryOrderAcquire);
  }
  template <class T>
  static FLAGCX_DEVICE_INLINE_DECORATOR void storeSystem(T *p, T v) {
    storeScoped<0>(p, v);
  }
  struct alignas(16) QueueWord {
    uint64_t lo, hi;
  };
  static FLAGCX_DEVICE_INLINE_DECORATOR void storeQueue128(uint64_t *p,
                                                           const uint64_t *v) {
    Ops::store<QueueWord, Vector2, flagcxDeviceMemoryGlobal,
               flagcxDeviceMemoryOrderWeak, flagcxDeviceScopeSystem,
               flagcxDeviceMemoryAccessKindNormal,
               flagcxDeviceCacheHintEvictFirst>(
        Thread{}, reinterpret_cast<QueueWord *>(p), {v[0], v[1]});
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR void writeMmio64(void *p, uint64_t v) {
    Ops::store<uint64_t, Scalar, flagcxDeviceMemoryGlobal,
               flagcxDeviceMemoryOrderRelaxed, flagcxDeviceScopeSystem,
               flagcxDeviceMemoryAccessKindMmio>(Thread{},
                                                 static_cast<uint64_t *>(p), v);
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR void asyncStoreSystem(uint32_t *p,
                                                              uint32_t v) {
#if __CUDA_ARCH__ >= 1000
    Ops::storeAsync(Thread{}, p, v, flagcxDeviceMemoryOrderRelease);
#else
    storeSystem(p, v);
#endif
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR void asyncStoreSystem(uint64_t *p,
                                                              uint64_t v) {
#if __CUDA_ARCH__ >= 1000
    Ops::storeAsync<Thread, uint64_t *, uint64_t, flagcxDeviceMemoryGlobal,
                    flagcxDeviceMemoryAccessKindMmio>(
        Thread{}, p, v, flagcxDeviceMemoryOrderRelease);
#else
    releaseSystem();
    writeMmio64(p, v);
#endif
  }
  // SDK has one implicit bulk group per issuing thread. FlagCX batches below
  // refer to that same hardware group across the separate SDK entry points.
  template <unsigned Bytes = 64>
  static FLAGCX_DEVICE_INLINE_DECORATOR void
  copySharedToGlobal64(void *d, const void *s) {
    static_assert(Bytes == 64, "SDK bulk copy is exactly 64 bytes");
    Ops::AsyncCopyBatch batch{};
    Ops::copyAsync(Thread{}, batch, static_cast<uint8_t *>(d),
                   static_cast<const uint8_t *>(s), Bytes);
  }
  template <unsigned Arch = 900>
  static FLAGCX_DEVICE_INLINE_DECORATOR void bulkCopyCommit() {
    Ops::AsyncCopyBatch batch{};
    batch.issued = true;
    Ops::commitAsync(Thread{}, batch);
  }
  template <unsigned Arch = 900>
  static FLAGCX_DEVICE_INLINE_DECORATOR void bulkCopyWait() {
    Ops::AsyncCopyBatch batch{};
    batch.issued = batch.committed = true;
    Ops::waitAsync(Thread{}, batch);
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR void spinPause() {
    PT::Intrin::spinBackoff(6);
  }
  static FLAGCX_DEVICE_INLINE_DECORATOR uint64_t clockNs() {
    return PT::Clock::now();
  }
};

#define XPU_NETIO_DEVICE_PLATFORM_TYPE flagcxXpuNetioNvidiaPlatform
#endif
