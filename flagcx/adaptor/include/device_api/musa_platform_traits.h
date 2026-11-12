/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 * Copyright (c) 2025-2026 Moore Threads Technology Co., Ltd.
 * Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Adapted from MooreThreads/mccl commit
 * 23edec5afda741efe0119410000dd80fd03092ef:
 *   src/include/mccl_device/coop.h
 *   src/include/mccl_device/utility.h
 *   src/include/mccl_device/gin/proxy/gin_proxy.h
 *
 * Cooperative groups follow MCCL's 32-lane MUSA implementation. Single-warp
 * and whole-CTA spans use the corresponding MUSA synchronization primitives;
 * partial-CTA multi-warp spans are explicitly unsupported. Atomic
 * access and fences use the scoped SDK primitives used by MCCL. The SDK
 * must provide native musa::atomic_ref with the requested operations.
 ************************************************************************/

#ifndef FLAGCX_MUSA_PLATFORM_TRAITS_H_
#define FLAGCX_MUSA_PLATFORM_TRAITS_H_

#include <cstddef>
#include <cstdint>
#include <type_traits>

// Match MCCL's compiler boundary: native atomic headers instantiate SDK
// floating-point types whose definitions are provided by the MUSA compiler.
// Ordinary host compilation needs only the FlagCX API and object layouts.
#if defined(FLAGCX_MUSA_DEVICE_COMPILER)
#include <musa_fp16.h>
#include <musa_bf16.h>
#include <musa/atomic>
#endif

struct MusaPlatform {};

template <>
struct PlatformTraits<MusaPlatform> {
  // Extended memory/clock operations are separate from the legacy Device
  // API port. Empty types expose no extended operation capability.
  struct CoopOps {};
  struct Clock {};

  struct Intrin {
    static constexpr int simtWidth = FLAGCX_SIMT_WIDTH;

    static FLAGCX_HOST_DEVICE_INLINE constexpr flagcxLaneMask_t fullMask() {
      return 0xffffffffull;
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR void
    validateMask(flagcxLaneMask_t mask) {
      if ((mask & ~fullMask()) != 0)
        FLAGCX_DEVICE_TRAP();
    }

#if defined(FLAGCX_MUSA_DEVICE_COMPILER)
    static FLAGCX_DEVICE_INLINE_DECORATOR int lane() {
      return __builtin_popcount(__get_lanemask_lt());
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR flagcxLaneMask_t lanemaskLt() {
      return static_cast<flagcxLaneMask_t>(__get_lanemask_lt());
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR flagcxLaneMask_t activemask() {
      return static_cast<flagcxLaneMask_t>(__activemask());
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR void
    syncwarp(flagcxLaneMask_t mask = fullMask()) {
      validateMask(mask);
      __syncwarp(static_cast<uint32_t>(mask));
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR int popc(flagcxLaneMask_t mask) {
      validateMask(mask);
      return __builtin_popcount(static_cast<uint32_t>(mask));
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR void namedBarrierSync(int,
                                                                int nThreads) {
      // A whole-CTA collective has the same participants for every group id.
      // A subset must use a genuine subgroup barrier with its own state.
      const unsigned blockThreads = blockDim.x * blockDim.y * blockDim.z;
      if (nThreads <= 0 || static_cast<unsigned>(nThreads) != blockThreads)
        FLAGCX_DEVICE_TRAP();
      __syncthreads();
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR void spinBackoff(int iter) {
      const unsigned exponent = static_cast<unsigned>(iter);
      const unsigned delay = 1u << (exponent < 15 ? exponent : 15);
      const uint64_t start = clock64();
      while (clock64() - start < delay) {
      }
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR void threadfenceSystem() {
      FLAGCX_DEVICE_THREAD_FENCE();
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR void threadfenceDevice() {
      musa::atomic_thread_fence(musa::memory_order_seq_cst,
                                musa::thread_scope_device);
    }
#else
    static inline int lane() {
      FLAGCX_DEVICE_TRAP();
      return 0;
    }
    static inline flagcxLaneMask_t lanemaskLt() {
      FLAGCX_DEVICE_TRAP();
      return 0;
    }
    static inline flagcxLaneMask_t activemask() {
      FLAGCX_DEVICE_TRAP();
      return 0;
    }
    static inline void syncwarp(flagcxLaneMask_t = fullMask()) {
      FLAGCX_DEVICE_TRAP();
    }
    static inline int popc(flagcxLaneMask_t) {
      FLAGCX_DEVICE_TRAP();
      return 0;
    }
    static inline void namedBarrierSync(int, int) { FLAGCX_DEVICE_TRAP(); }
    static inline void spinBackoff(int) { FLAGCX_DEVICE_TRAP(); }
    static inline void threadfenceSystem() { FLAGCX_DEVICE_TRAP(); }
    static inline void threadfenceDevice() { FLAGCX_DEVICE_TRAP(); }
#endif

    template <typename DstPtr, typename SrcPtr>
    static FLAGCX_DEVICE_INLINE_DECORATOR void
    coopCopyBytes(DstPtr dst, SrcPtr src, size_t bytes, int rank, int size) {
      auto *out = reinterpret_cast<unsigned char *>(dst);
      const auto *in = reinterpret_cast<const unsigned char *>(src);
      for (size_t i = static_cast<size_t>(rank); i < bytes; i += size)
        out[i] = in[i];
    }

    template <typename DstPtr, typename SrcPtr>
    static FLAGCX_DEVICE_INLINE_DECORATOR void copyBytes(DstPtr dst, SrcPtr src,
                                                         size_t bytes) {
      coopCopyBytes(dst, src, bytes, 0, 1);
    }
  };

  struct Atomic {
#if defined(FLAGCX_MUSA_DEVICE_COMPILER)
    static FLAGCX_HOST_DEVICE_INLINE constexpr musa::memory_order
    toNativeOrder(flagcxDeviceMemoryOrder_t order) {
      switch (order) {
        case flagcxDeviceMemoryOrderRelaxed:
          return musa::memory_order_relaxed;
        case flagcxDeviceMemoryOrderAcquire:
          return musa::memory_order_acquire;
        case flagcxDeviceMemoryOrderRelease:
          return musa::memory_order_release;
        case flagcxDeviceMemoryOrderAcqRel:
          return musa::memory_order_acq_rel;
        case flagcxDeviceMemoryOrderSeqCst:
          return musa::memory_order_seq_cst;
        default:
          FLAGCX_DEVICE_TRAP();
          return musa::memory_order_relaxed;
      }
    }

    static FLAGCX_HOST_DEVICE_INLINE constexpr musa::thread_scope
    toNativeScope(flagcxDeviceScope_t scope) {
      switch (scope) {
        case flagcxDeviceScopeSystem:
          return musa::thread_scope_system;
        case flagcxDeviceScopeDevice:
          return musa::thread_scope_device;
        case flagcxDeviceScopeBlock:
          return musa::thread_scope_block;
        case flagcxDeviceScopeThread:
          return musa::thread_scope_thread;
        default:
          FLAGCX_DEVICE_TRAP();
          return musa::thread_scope_thread;
      }
    }

    template <typename T, flagcxDeviceScope_t Scope>
    using Ref = musa::atomic_ref<T, toNativeScope(Scope)>;

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T
    load(T *ptr, flagcxDeviceMemoryOrder_t order) {
      return Ref<T, Scope>{*ptr}.load(toNativeOrder(order));
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR void
    store(T *ptr, const T &value, flagcxDeviceMemoryOrder_t order) {
      Ref<T, Scope>{*ptr}.store(value, toNativeOrder(order));
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T
    fetchAdd(T *ptr, const T &value, flagcxDeviceMemoryOrder_t order) {
      return Ref<T, Scope>{*ptr}.fetch_add(value, toNativeOrder(order));
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T
    fetchSub(T *ptr, const T &value, flagcxDeviceMemoryOrder_t order) {
      return Ref<T, Scope>{*ptr}.fetch_sub(value, toNativeOrder(order));
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T
    fetchOr(T *ptr, const T &value, flagcxDeviceMemoryOrder_t order) {
      return Ref<T, Scope>{*ptr}.fetch_or(value, toNativeOrder(order));
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T
    fetchAnd(T *ptr, const T &value, flagcxDeviceMemoryOrder_t order) {
      return Ref<T, Scope>{*ptr}.fetch_and(value, toNativeOrder(order));
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T
    exchange(T *ptr, const T &value, flagcxDeviceMemoryOrder_t order) {
      return Ref<T, Scope>{*ptr}.exchange(value, toNativeOrder(order));
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR bool
    compareExchange(T *ptr, T &expected, const T &desired,
                    flagcxDeviceMemoryOrder_t order) {
      return Ref<T, Scope>{*ptr}.compare_exchange_strong(expected, desired,
                                                       toNativeOrder(order));
    }

    // CAS-based max matches the existing FlagCX atomic contract, including
    // release ordering when the maximum is already present.
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T
    fetchMax(T *ptr, T value, flagcxDeviceMemoryOrder_t order) {
      static_assert(std::is_integral<T>::value &&
                        !std::is_same<T, bool>::value &&
                        (sizeof(T) == 4 || sizeof(T) == 8),
                    "fetchMax requires a 32/64-bit integer");
      Ref<T, Scope> ref{*ptr};
      T old = ref.load(musa::memory_order_relaxed);
      for (;;) {
        const T desired = old < value ? value : old;
        if (ref.compare_exchange_weak(old, desired, toNativeOrder(order),
                                      musa::memory_order_relaxed))
          return old;
      }
    }
#else
    // Host compilation needs the same API surface for object layouts and
    // backend templates. Execution of a device atomic on the host is invalid.
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline T load(T *, flagcxDeviceMemoryOrder_t) {
      FLAGCX_DEVICE_TRAP();
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline void store(T *, const T &, flagcxDeviceMemoryOrder_t) {
      FLAGCX_DEVICE_TRAP();
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline T fetchAdd(T *, const T &, flagcxDeviceMemoryOrder_t) {
      FLAGCX_DEVICE_TRAP();
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline T fetchSub(T *, const T &, flagcxDeviceMemoryOrder_t) {
      FLAGCX_DEVICE_TRAP();
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline T fetchOr(T *, const T &, flagcxDeviceMemoryOrder_t) {
      FLAGCX_DEVICE_TRAP();
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline T fetchAnd(T *, const T &, flagcxDeviceMemoryOrder_t) {
      FLAGCX_DEVICE_TRAP();
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline T exchange(T *, const T &, flagcxDeviceMemoryOrder_t) {
      FLAGCX_DEVICE_TRAP();
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline bool compareExchange(T *, T &, const T &,
                                      flagcxDeviceMemoryOrder_t) {
      FLAGCX_DEVICE_TRAP();
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline T fetchMax(T *, T, flagcxDeviceMemoryOrder_t) {
      FLAGCX_DEVICE_TRAP();
    }
#endif
  };

  // Adapted from mcclCoopCta, mcclCoopTile, mcclCoopWarpSpan and
  // mcclCoopLanes. Public method names follow the FlagCX group contract.
  struct CoopBlock {
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return FLAGCX_THREAD_IDX_X;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const {
      return FLAGCX_BLOCK_DIM_X;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() {
      FLAGCX_DEVICE_SYNC_THREADS();
    }
  };

  template <int N>
  struct CoopTile {
    static_assert(N > 0 && N <= Intrin::simtWidth && (N & (N - 1)) == 0,
                  "MUSA tiles require a power of two within one warp");

    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return Intrin::lane() % N;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR constexpr int size() const { return N; }
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
  using CoopWarp = CoopTile<32>;

  struct CoopTileSpan {
    int firstTile;
    int tileCount;
    int barrierId;

    FLAGCX_HOST_DEVICE_INLINE constexpr CoopTileSpan(int first, int count,
                                                    int id)
        : firstTile(first), tileCount(count), barrierId(id) {}

    // Query the supported one-dimensional launch contract before using a
    // span. The bounds checks precede subtraction to avoid invalid ranges.
    static FLAGCX_HOST_DEVICE_INLINE constexpr bool
    supportsSync(int first, int count, int blockThreads) {
      return first >= 0 && count > 0 &&
             first < blockThreads / Intrin::simtWidth &&
             count <= blockThreads / Intrin::simtWidth - first &&
             (count == 1 ||
              (first == 0 && count == blockThreads / Intrin::simtWidth &&
               blockThreads % Intrin::simtWidth == 0));
    }

    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return static_cast<int>(FLAGCX_THREAD_IDX_X) -
             Intrin::simtWidth * firstTile;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const {
      return Intrin::simtWidth * tileCount;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR bool supportsCallerSync() const {
#if defined(FLAGCX_MUSA_DEVICE_COMPILER)
      return blockDim.y == 1 && blockDim.z == 1 &&
             supportsSync(firstTile, tileCount, FLAGCX_BLOCK_DIM_X) &&
             threadRank() >= 0 && threadRank() < size();
#else
      return false;
#endif
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void sync() {
#if defined(FLAGCX_MUSA_DEVICE_COMPILER)
      if (!supportsCallerSync())
        FLAGCX_DEVICE_TRAP();

      if (tileCount == 1) {
        Intrin::syncwarp(Intrin::fullMask());
        return;
      }
      Intrin::namedBarrierSync(barrierId, size());
#else
      FLAGCX_DEVICE_TRAP();
#endif
    }
  };

  struct CoopLanes {
    flagcxLaneMask_t mask;

    FLAGCX_HOST_DEVICE_INLINE explicit constexpr CoopLanes(
        flagcxLaneMask_t lanes = Intrin::fullMask())
        : mask(lanes) {}

    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      Intrin::validateMask(mask);
      return Intrin::popc(mask & Intrin::lanemaskLt());
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const {
      return Intrin::popc(mask);
    }
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxLaneMask_t getLmask() const {
      return mask;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() { Intrin::syncwarp(mask); }
  };

  // mcc 5.1.0 / mp_31 rejects ABS64 relocations in PlatformCoop's static
  // device function-pointer table even with -fPIC. Resolve the group facts
  // at conversion and dispatch directly to the same synchronization primitive.
  // No device function addresses or indirect calls are stored here.
  struct CoopAny {
    enum SyncKind : uint32_t {
      SyncNone,
      SyncWarp,
      SyncBlock,
      SyncUnsupported,
    };

    int _rank;
    int _size;
    flagcxLaneMask_t _mask;
    SyncKind _syncKind;

    FLAGCX_DEVICE_INLINE_DECORATOR CoopAny()
        : _rank(0), _size(1), _mask(1), _syncKind(SyncNone) {}
    CoopAny(CoopAny const &) = default;

    FLAGCX_DEVICE_INLINE_DECORATOR CoopAny(CoopBlock group)
        : _rank(group.threadRank()), _size(group.size()),
          _mask(Intrin::fullMask()), _syncKind(SyncBlock) {}

    template <int N>
    FLAGCX_DEVICE_INLINE_DECORATOR CoopAny(CoopTile<N> group)
        : _rank(group.threadRank()), _size(group.size()),
          _mask(group.laneMask()), _syncKind(N == 1 ? SyncNone : SyncWarp) {}

    FLAGCX_DEVICE_INLINE_DECORATOR CoopAny(CoopTileSpan group)
        : _rank(group.threadRank()), _size(group.size()),
          _mask(Intrin::fullMask()),
          _syncKind(!group.supportsCallerSync()
                        ? SyncUnsupported
                        : (group.tileCount == 1 ? SyncWarp : SyncBlock)) {}

    FLAGCX_DEVICE_INLINE_DECORATOR CoopAny(CoopLanes group)
        : _rank(group.threadRank()), _size(group.size()),
          _mask(group.getLmask()), _syncKind(SyncWarp) {}

    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const { return _rank; }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return _size; }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() const {
      switch (_syncKind) {
      case SyncNone:
        return;
      case SyncWarp:
        Intrin::syncwarp(_mask);
        return;
      case SyncBlock:
        FLAGCX_DEVICE_SYNC_THREADS();
        return;
      case SyncUnsupported:
      default:
        FLAGCX_DEVICE_TRAP();
      }
    }
  };
};

static_assert(sizeof(PlatformTraits<MusaPlatform>::CoopAny) == 24,
              "MUSA CoopAny opaque layout changed");
static_assert(alignof(PlatformTraits<MusaPlatform>::CoopAny) == 8,
              "MUSA CoopAny alignment changed");

#endif // FLAGCX_MUSA_PLATFORM_TRAITS_H_
