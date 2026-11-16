/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * MUSA Platform Traits — MUSA SIMT intrinsics and scoped atomics.
 *
 * Provides PlatformTraits<MusaPlatform> for
 * CommTraits<DefaultBackend<MusaPlatform>>.
 *
 * Measured on MTT S5000 (mcc 5.1.0, target mp_31):
 *   - warpSize == 32, block-wide __syncthreads, __threadfence/_block.
 *   - __activemask() returns a 32-bit mask; __syncwarp() is a whole-wave
 *     barrier and ignores its mask argument.
 *   - __threadfence_system() is an alias of __threadfence() (device scope).
 *     The backend has no system membar at all: __nvvm_membar_sys is commented
 *     out in clang/Basic/BuiltinsMTGPU.def and libLLVMMTGPUDesc only defines
 *     MTGPU_MEMBAR_CTA/GL, so System scope is served at GL scope here and
 *     cross-device ordering must come from the transport. The SDK also
 *     exposes __musa_membar_gl_release, __musa_fence_rel and
 *     __musa_membar_remote_signal; their scope is undocumented, so this
 *     adaptor does not build a memory model on them.
 *   - 32- and 64-bit integer atomics (atomicAdd/Sub/Or/And/Xor/Exch/CAS) are
 *     available. atomicSub has no 64-bit overload, so 64-bit subtraction is
 *     expressed as atomicAdd of the two's-complement negation.
 ************************************************************************/

#ifndef FLAGCX_MUSA_PLATFORM_TRAITS_H_
#define FLAGCX_MUSA_PLATFORM_TRAITS_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

struct MusaPlatform {};

// __builtin_trap() is not selectable by the MUSA backend: it lowers to an
// external `abort` symbol and mcc fails with "Cannot select: ExternalSymbol
// 'abort'". The SDK's device intrinsic __trap() is supported and is used on
// the device pass; the host pass keeps the portable builtin (never executed).
#if defined(__MUSACC__)
#define FLAGCX_MUSA_TRAP() __trap()
#else
#define FLAGCX_MUSA_TRAP() __builtin_trap()
#endif

template <>
struct PlatformTraits<MusaPlatform> {
  // Single-device cooperative copies and the device clock are not part of
  // the MUSA test surface yet; DefaultBackend only requires the types.
  struct CoopOps {};
  struct Clock {};

  // ==============================================================
  // Intrin — MUSA SIMT intrinsics
  // ==============================================================
  struct Intrin {
    static constexpr int simtWidth = FLAGCX_SIMT_WIDTH;

    static FLAGCX_HOST_DEVICE_INLINE constexpr flagcxLaneMask_t fullMask() {
      return (flagcxLaneMask_t{1} << simtWidth) - 1;
    }

#if defined(__MUSACC__)
    static FLAGCX_DEVICE_INLINE_DECORATOR int lane() {
      return static_cast<int>(FLAGCX_THREAD_IDX_X & (simtWidth - 1));
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR flagcxLaneMask_t lanemaskLt() {
      const unsigned laneId = static_cast<unsigned>(lane());
      return laneId == 0 ? flagcxLaneMask_t{0}
                         : ((flagcxLaneMask_t{1} << laneId) - 1);
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR flagcxLaneMask_t activemask() {
      static_assert(sizeof(decltype(__activemask())) == sizeof(uint32_t),
                    "MUSA __activemask() must return a 32-bit mask");
      return static_cast<flagcxLaneMask_t>(__activemask());
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR void
    validateMask(flagcxLaneMask_t mask) {
      if ((mask & ~fullMask()) != 0)
        FLAGCX_MUSA_TRAP();
    }

    // MUSA has a single whole-wave barrier: __syncwarp() ignores its mask
    // argument (measured). For the full mask that is exactly the request. A
    // sub-warp mask is served by the same whole-wave barrier, which is a
    // *stronger* sync, but it is only valid when every lane of the wave
    // reaches the call -- which is how FlagCX's CoopTile groups are used
    // (uniform participation). A single-lane group needs no barrier at all.
    static FLAGCX_DEVICE_INLINE_DECORATOR void
    syncwarp(flagcxLaneMask_t mask = fullMask()) {
      validateMask(mask);
      if (popc(mask) <= 1)
        return;
      __syncwarp();
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR int popc(flagcxLaneMask_t value) {
      return __popcll(static_cast<unsigned long long>(value));
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR void namedBarrierSync(int, int n) {
      // MUSA's __barrier_sync/__barrier_sync_count make mcc 5.1.0 loop
      // forever without a diagnostic (measured), so only the granularities
      // that map to verified hardware primitives are exposed: a whole block
      // maps to __syncthreads, a single tile (= one 32-lane wave) maps to
      // __syncwarp, and one thread needs nothing.
      if (n == FLAGCX_BLOCK_DIM_X) {
        FLAGCX_DEVICE_SYNC_THREADS();
        return;
      }
      if (n == simtWidth) {
        __syncwarp();
        return;
      }
      if (n == 1)
        return;
      FLAGCX_MUSA_TRAP();
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR void spinBackoff(int iter) {
      // No verified hardware sleep instruction on S5000. Back off in the
      // compiler only: the empty asm memory clobber keeps the polling load
      // from being hoisted, the integer loop adds delay without bus traffic.
      const int delay = 1 << (iter < 10 ? iter : 10);
      asm volatile("" ::: "memory");
      for (int i = 0; i < delay; ++i)
        asm volatile("");
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR void threadfenceSystem() {
      // __threadfence_system() is an alias of __threadfence() here: the MTGPU
      // backend has no membar.sys and no __nvvm_membar_sys, so System scope
      // can only be served at global/device scope. See the file header.
      __threadfence();
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR void threadfenceDevice() {
      __threadfence();
    }
#else
    // Host pass: allow template instantiation, never called at runtime.
    static inline int lane() {
      assert(false && "lane() called on MUSA host pass");
      return 0;
    }
    static inline flagcxLaneMask_t lanemaskLt() {
      assert(false && "lanemaskLt() called on MUSA host pass");
      return 0;
    }
    static inline flagcxLaneMask_t activemask() {
      assert(false && "activemask() called on MUSA host pass");
      return fullMask();
    }
    static inline void validateMask(flagcxLaneMask_t mask) {
      assert((mask & ~fullMask()) == 0 && "lane mask exceeds MUSA SIMT width");
    }
    static inline void syncwarp(flagcxLaneMask_t = fullMask()) {
      assert(false && "syncwarp() called on MUSA host pass");
    }
    static inline int popc(flagcxLaneMask_t value) {
      return __builtin_popcountll(value);
    }
    static inline void namedBarrierSync(int, int) {
      assert(false && "namedBarrierSync() called on MUSA host pass");
    }
    static inline void spinBackoff(int) {}
    static inline void threadfenceSystem() {
      assert(false && "threadfenceSystem() called on MUSA host pass");
    }
    static inline void threadfenceDevice() {
      assert(false && "threadfenceDevice() called on MUSA host pass");
    }
#endif // __MUSACC__

    // Width-cascading strided copy between two device-memory buffers. Same
    // contract as the CUDA path: a typed access is used only while both
    // operands satisfy that width's alignment.
    template <typename DstPtr, typename SrcPtr>
    FLAGCX_DEVICE_INLINE_DECORATOR static void
    coopCopyBytes(DstPtr dstIn, SrcPtr srcIn, size_t bytes, int rank,
                  int size) {
      void *dst = (void *)dstIn;
      const void *src = (const void *)srcIn;

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

      unsigned char *d = (unsigned char *)dst;
      const unsigned char *s = (const unsigned char *)src;
      for (size_t i = (size_t)rank; i < bytes; i += (size_t)size)
        d[i] = s[i];
    }
  };

  // ==============================================================
  // Atomic — volatile access + explicit fence, hardware RMW
  //
  // The hardware RMWs below (atomicAdd/Sub/Or/And/Exch/CAS) are relaxed and
  // take no memory order. The contract is atomic_ref<T, Scope>::fetch_*(order),
  // so an order stronger than relaxed is realised as the standard prescribes: a
  // release fence before the RMW and an acquire fence after it. Leaving those
  // out dropped even the device-scope release that a Signal or Counter publish
  // depends on.
  //
  // mp_31 cannot select a scoped atomic load on the addrspace-3 pointers used
  // by the unified IR wait paths ("Cannot select: AtomicLoadSub ...
  // syncscope(\"musa_device\")"), and the SDK has no documented system-scope
  // membar. Scope is therefore honoured by the fence only: Thread is a
  // compiler barrier, Block is __threadfence_block(), and every wider scope
  // (Device and System) is __threadfence(). A System-scope publish to host or
  // peer memory therefore relies on the transport for cross-device ordering;
  // see the file header.
  // ==============================================================
  struct Atomic {
#if defined(__MUSACC__)
    template <flagcxDeviceScope_t Scope>
    FLAGCX_DEVICE_INLINE_DECORATOR static void scopedFence() {
      if (Scope == flagcxDeviceScopeThread) {
        asm volatile("" ::: "memory");
      } else if (Scope == flagcxDeviceScopeBlock) {
        __threadfence_block();
      } else {
        __threadfence();
      }
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR bool
    hasAcquire(flagcxDeviceMemoryOrder_t order) {
      return order == flagcxDeviceMemoryOrderAcquire ||
             order == flagcxDeviceMemoryOrderAcqRel ||
             order == flagcxDeviceMemoryOrderSeqCst;
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR bool
    hasRelease(flagcxDeviceMemoryOrder_t order) {
      return order == flagcxDeviceMemoryOrderRelease ||
             order == flagcxDeviceMemoryOrderAcqRel ||
             order == flagcxDeviceMemoryOrderSeqCst;
    }

    template <flagcxDeviceScope_t Scope>
    static FLAGCX_DEVICE_INLINE_DECORATOR void
    rmwFenceBefore(flagcxDeviceMemoryOrder_t order) {
      if (hasRelease(order))
        scopedFence<Scope>();
    }

    template <flagcxDeviceScope_t Scope>
    static FLAGCX_DEVICE_INLINE_DECORATOR void
    rmwFenceAfter(flagcxDeviceMemoryOrder_t order) {
      if (hasAcquire(order))
        scopedFence<Scope>();
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    FLAGCX_DEVICE_INLINE_DECORATOR static T
    load(T *ptr, flagcxDeviceMemoryOrder_t order) {
      const T value = *((volatile T *)ptr);
      if (hasAcquire(order))
        scopedFence<Scope>();
      return value;
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    FLAGCX_DEVICE_INLINE_DECORATOR static void
    store(T *ptr, const T &value, flagcxDeviceMemoryOrder_t order) {
      if (order != flagcxDeviceMemoryOrderRelaxed)
        scopedFence<Scope>();
      *((volatile T *)ptr) = value;
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    FLAGCX_DEVICE_INLINE_DECORATOR static T
    fetchAdd(T *ptr, const T &value, flagcxDeviceMemoryOrder_t order) {
      static_assert(
          std::is_integral<T>::value && (sizeof(T) == 4 || sizeof(T) == 8),
          "MUSA atomic RMW requires a 32- or 64-bit integral operand");
      rmwFenceBefore<Scope>(order);
      T old;
      if constexpr (sizeof(T) == 8)
        old = static_cast<T>(
            atomicAdd((unsigned long long *)ptr, (unsigned long long)value));
      else
        old =
            static_cast<T>(atomicAdd((unsigned int *)ptr, (unsigned int)value));
      rmwFenceAfter<Scope>(order);
      return old;
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    FLAGCX_DEVICE_INLINE_DECORATOR static T
    fetchSub(T *ptr, const T &value, flagcxDeviceMemoryOrder_t order) {
      static_assert(
          std::is_integral<T>::value && (sizeof(T) == 4 || sizeof(T) == 8),
          "MUSA atomic RMW requires a 32- or 64-bit integral operand");
      // No 64-bit atomicSub overload: subtracting is adding the two's
      // complement negation. Negate in the unsigned domain so a signed minimum
      // value does not overflow.
      using U = typename std::make_unsigned<T>::type;
      return fetchAdd<T, Scope>(
          ptr, static_cast<T>(U(0) - static_cast<U>(value)), order);
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    FLAGCX_DEVICE_INLINE_DECORATOR static T
    fetchOr(T *ptr, const T &value, flagcxDeviceMemoryOrder_t order) {
      static_assert(
          std::is_integral<T>::value && (sizeof(T) == 4 || sizeof(T) == 8),
          "MUSA atomic RMW requires a 32- or 64-bit integral operand");
      rmwFenceBefore<Scope>(order);
      T old;
      if constexpr (sizeof(T) == 8)
        old = static_cast<T>(
            atomicOr((unsigned long long *)ptr, (unsigned long long)value));
      else
        old =
            static_cast<T>(atomicOr((unsigned int *)ptr, (unsigned int)value));
      rmwFenceAfter<Scope>(order);
      return old;
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    FLAGCX_DEVICE_INLINE_DECORATOR static T
    fetchAnd(T *ptr, const T &value, flagcxDeviceMemoryOrder_t order) {
      static_assert(
          std::is_integral<T>::value && (sizeof(T) == 4 || sizeof(T) == 8),
          "MUSA atomic RMW requires a 32- or 64-bit integral operand");
      rmwFenceBefore<Scope>(order);
      T old;
      if constexpr (sizeof(T) == 8)
        old = static_cast<T>(
            atomicAnd((unsigned long long *)ptr, (unsigned long long)value));
      else
        old =
            static_cast<T>(atomicAnd((unsigned int *)ptr, (unsigned int)value));
      rmwFenceAfter<Scope>(order);
      return old;
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    FLAGCX_DEVICE_INLINE_DECORATOR static T
    exchange(T *ptr, const T &value, flagcxDeviceMemoryOrder_t order) {
      static_assert(
          std::is_integral<T>::value && (sizeof(T) == 4 || sizeof(T) == 8),
          "MUSA atomic RMW requires a 32- or 64-bit integral operand");
      rmwFenceBefore<Scope>(order);
      T old;
      if constexpr (sizeof(T) == 8)
        old = static_cast<T>(
            atomicExch((unsigned long long *)ptr, (unsigned long long)value));
      else
        old = static_cast<T>(
            atomicExch((unsigned int *)ptr, (unsigned int)value));
      rmwFenceAfter<Scope>(order);
      return old;
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    FLAGCX_DEVICE_INLINE_DECORATOR static bool
    compareExchange(T *ptr, T &expected, const T &desired,
                    flagcxDeviceMemoryOrder_t order) {
      static_assert(
          std::is_integral<T>::value && (sizeof(T) == 4 || sizeof(T) == 8),
          "MUSA atomic RMW requires a 32- or 64-bit integral operand");
      rmwFenceBefore<Scope>(order);
      bool ok;
      if constexpr (sizeof(T) == 8) {
        const unsigned long long exp = (unsigned long long)expected;
        const unsigned long long old = atomicCAS((unsigned long long *)ptr, exp,
                                                 (unsigned long long)desired);
        ok = (old == exp);
        if (!ok)
          expected = static_cast<T>(old);
      } else {
        const unsigned int exp = (unsigned int)expected;
        const unsigned int old =
            atomicCAS((unsigned int *)ptr, exp, (unsigned int)desired);
        ok = (old == exp);
        if (!ok)
          expected = static_cast<T>(old);
      }
      rmwFenceAfter<Scope>(order);
      return ok;
    }
#else
    // Host pass. These functions only exist so the shared templates can be
    // instantiated by the host compiler; they are never executed. The GCC
    // builtins give the same signatures as the device implementation.
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline T load(T *ptr, flagcxDeviceMemoryOrder_t) {
      return __atomic_load_n(ptr, __ATOMIC_SEQ_CST);
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline void store(T *ptr, const T &value,
                             flagcxDeviceMemoryOrder_t) {
      __atomic_store_n(ptr, value, __ATOMIC_SEQ_CST);
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline T fetchAdd(T *ptr, const T &value,
                             flagcxDeviceMemoryOrder_t) {
      return __atomic_fetch_add(ptr, value, __ATOMIC_SEQ_CST);
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline T fetchSub(T *ptr, const T &value,
                             flagcxDeviceMemoryOrder_t) {
      return __atomic_fetch_sub(ptr, value, __ATOMIC_SEQ_CST);
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline T fetchOr(T *ptr, const T &value, flagcxDeviceMemoryOrder_t) {
      return __atomic_fetch_or(ptr, value, __ATOMIC_SEQ_CST);
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline T fetchAnd(T *ptr, const T &value,
                             flagcxDeviceMemoryOrder_t) {
      return __atomic_fetch_and(ptr, value, __ATOMIC_SEQ_CST);
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline T exchange(T *ptr, const T &value,
                             flagcxDeviceMemoryOrder_t) {
      return __atomic_exchange_n(ptr, value, __ATOMIC_SEQ_CST);
    }
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static inline bool compareExchange(T *ptr, T &expected, const T &desired,
                                       flagcxDeviceMemoryOrder_t) {
      return __atomic_compare_exchange_n(ptr, &expected, desired, false,
                                         __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    }
#endif // __MUSACC__
  };

  // ==============================================================
  // Coop — MUSA cooperative groups
  // ==============================================================

  struct CoopBlock {
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return static_cast<int>(FLAGCX_THREAD_IDX_X);
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const {
      return static_cast<int>(FLAGCX_BLOCK_DIM_X);
    }
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxLaneMask_t laneMask() const {
      return Intrin::fullMask();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() const {
      FLAGCX_DEVICE_SYNC_THREADS();
    }
  };

  template <int N>
  struct CoopTile {
    static_assert(N > 0 && (N & (N - 1)) == 0 && N <= Intrin::simtWidth,
                  "N must be a power of two and <= MUSA wave width");
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return Intrin::lane() % N;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return N; }
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxLaneMask_t laneMask() const {
      const int base = Intrin::lane() & -N;
      const flagcxLaneMask_t bits =
          Intrin::fullMask() >> (Intrin::simtWidth - N);
      return bits << base;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() const {
      if (N > 1)
        Intrin::syncwarp(laneMask());
    }
  };

  using CoopThread = CoopTile<1>;
  using CoopWarp = CoopTile<Intrin::simtWidth>;

  struct CoopTileSpan {
    int first;
    int count;
    int barrierId;

    FLAGCX_DEVICE_INLINE_DECORATOR CoopTileSpan(int t0, int nTiles, int id)
        : first(t0), count(nTiles), barrierId(id) {
      if (first < 0 || count <= 0)
        FLAGCX_MUSA_TRAP();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return static_cast<int>(FLAGCX_THREAD_IDX_X) - first * Intrin::simtWidth;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const {
      return count * Intrin::simtWidth;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() const {
      Intrin::namedBarrierSync(barrierId, size());
    }
  };

  struct CoopLanes {
    flagcxLaneMask_t mask;

    FLAGCX_DEVICE_INLINE_DECORATOR explicit CoopLanes(
        flagcxLaneMask_t value = Intrin::fullMask())
        : mask(value) {
      Intrin::validateMask(mask);
      if (mask == 0)
        FLAGCX_MUSA_TRAP();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return Intrin::popc(mask & Intrin::lanemaskLt());
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const {
      return Intrin::popc(mask);
    }
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxLaneMask_t getLmask() const {
      return mask;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() const { Intrin::syncwarp(mask); }
  };

  // The common vtable-based PlatformCoop relies on relocatable function
  // pointers inside device code. MUSA's clang backend is not a proven target
  // for that, so the group facts are stored directly and the sync switch only
  // calls verified primitives -- same approach as the Other CoreX backend.
  struct CoopAny {
    enum SyncKind : uint32_t {
      SyncNone = 0,
      SyncWarp = 1,
      SyncBlock = 2,
      SyncUnsupported = 3,
    };

    int _rank;
    int _size;
    flagcxLaneMask_t _mask;
    SyncKind _syncKind;

    FLAGCX_DEVICE_INLINE_DECORATOR CoopAny()
        : _rank(0), _size(1), _mask(1), _syncKind(SyncNone) {}
    CoopAny(CoopAny const &) = default;

    FLAGCX_DEVICE_INLINE_DECORATOR explicit CoopAny(CoopBlock group)
        : _rank(group.threadRank()), _size(group.size()),
          _mask(Intrin::fullMask()), _syncKind(SyncBlock) {}

    template <int N>
    FLAGCX_DEVICE_INLINE_DECORATOR explicit CoopAny(CoopTile<N> group)
        : _rank(group.threadRank()), _size(group.size()),
          _mask(group.laneMask()),
          _syncKind(
              N == 1 ? SyncNone
                     : (N == Intrin::simtWidth ? SyncWarp : SyncUnsupported)) {}

    FLAGCX_DEVICE_INLINE_DECORATOR explicit CoopAny(CoopTileSpan group)
        : _rank(group.threadRank()), _size(group.size()),
          _mask(Intrin::fullMask()),
          // A span of one tile is exactly one wave, which __syncwarp() serves
          // (the same mapping CoopTileSpan::sync() uses). A span covering the
          // whole block maps to __syncthreads. Wider spans have no verified
          // MUSA primitive, so they stay unsupported instead of silently
          // degrading to a barrier that is weaker than the request.
          _syncKind(group.size() == FLAGCX_BLOCK_DIM_X
                        ? SyncBlock
                        : (group.count == 1 ? SyncWarp : SyncUnsupported)) {}

    FLAGCX_DEVICE_INLINE_DECORATOR explicit CoopAny(CoopLanes group)
        : _rank(group.threadRank()), _size(group.size()),
          _mask(group.getLmask()),
          _syncKind(group.size() == 1 ? SyncNone
                                      : (group.getLmask() == Intrin::fullMask()
                                             ? SyncWarp
                                             : SyncUnsupported)) {}

    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const { return _rank; }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return _size; }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() const {
      switch (_syncKind) {
        case SyncNone:
          return;
        case SyncWarp:
          Intrin::syncwarp(Intrin::fullMask());
          return;
        case SyncBlock:
          FLAGCX_DEVICE_SYNC_THREADS();
          return;
        case SyncUnsupported:
        default:
          FLAGCX_MUSA_TRAP();
      }
    }
  };
};

static_assert(sizeof(PlatformTraits<MusaPlatform>::CoopAny) == 24,
              "MUSA CoopAny must fit the established opaque layout");
static_assert(alignof(PlatformTraits<MusaPlatform>::CoopAny) == 8,
              "MUSA CoopAny alignment changed");

#endif // FLAGCX_MUSA_PLATFORM_TRAITS_H_
