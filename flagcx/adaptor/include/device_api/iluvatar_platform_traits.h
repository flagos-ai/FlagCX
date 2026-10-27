/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Iluvatar/CoreX platform traits for the common FlagCX Device API.
 ************************************************************************/

#ifndef FLAGCX_ILUVATAR_PLATFORM_TRAITS_H_
#define FLAGCX_ILUVATAR_PLATFORM_TRAITS_H_

#include <cassert>
#include <cstddef>
#include <cstdint>

struct IluvatarPlatform {};

template <>
struct PlatformCompletionWord<IluvatarPlatform> {
  using type = uint32_t;
};

template <>
struct PlatformTraits<IluvatarPlatform> {
  struct Intrin {
    static constexpr int simtWidth = FLAGCX_SIMT_WIDTH;

    static FLAGCX_HOST_DEVICE_INLINE constexpr flagcxLaneMask_t fullMask() {
      return ~flagcxLaneMask_t{0};
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static void
    validateMask(flagcxLaneMask_t mask) {
      (void)mask;
    }

#if defined(FLAGCX_ILUVATAR_DEVICE_COMPILE)
    FLAGCX_DEVICE_INLINE_DECORATOR static int lane() {
      return static_cast<int>(FLAGCX_THREAD_IDX_X & (simtWidth - 1));
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static flagcxLaneMask_t lanemaskLt() {
      const unsigned laneId = static_cast<unsigned>(lane());
      return laneId == 0 ? 0ull : ((flagcxLaneMask_t{1} << laneId) - 1ull);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static flagcxLaneMask_t activemask() {
      // Do not repeat the old FlagTree adapter's uint32_t cast: that silently
      // excluded lanes 32-63. The CoreX intrinsic must expose the complete
      // 64-lane mask or the device build is rejected.
      static_assert(sizeof(decltype(__activemask())) ==
                        sizeof(flagcxLaneMask_t),
                    "CoreX __activemask() must return a 64-bit mask");
      return static_cast<flagcxLaneMask_t>(__activemask());
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static int popc(flagcxLaneMask_t value) {
      return __popcll(value);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static void
    syncwarp(flagcxLaneMask_t mask = fullMask()) {
      if (mask == fullMask()) {
        __syncwarp();
        return;
      }
      if (popc(mask) == 1)
        return;
      // CoreX exposes a full-wave barrier but no verified arbitrary-mask
      // barrier. Expanding a partial mask to the whole wave can deadlock.
      __builtin_trap();
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static void namedBarrierSync(int, int n) {
      if (n == FLAGCX_BLOCK_DIM_X) {
        FLAGCX_DEVICE_SYNC_THREADS();
        return;
      }
      if (n == 1)
        return;
      __builtin_trap();
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static void spinBackoff(int) {
      asm volatile("");
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static void threadfenceSystem() {
      FLAGCX_DEVICE_THREAD_FENCE();
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static void threadfenceDevice() {
      FLAGCX_DEVICE_THREAD_FENCE();
    }
#else
    static inline int lane() {
      assert(false && "lane() called on Iluvatar host pass");
      return 0;
    }
    static inline flagcxLaneMask_t lanemaskLt() {
      assert(false && "lanemaskLt() called on Iluvatar host pass");
      return 0;
    }
    static inline flagcxLaneMask_t activemask() {
      assert(false && "activemask() called on Iluvatar host pass");
      return fullMask();
    }
    static inline int popc(flagcxLaneMask_t value) {
      return __builtin_popcountll(value);
    }
    static inline void syncwarp(flagcxLaneMask_t = fullMask()) {
      assert(false && "syncwarp() called on Iluvatar host pass");
    }
    static inline void namedBarrierSync(int, int) {
      assert(false && "namedBarrierSync() called on Iluvatar host pass");
    }
    static inline void spinBackoff(int) {}
    static inline void threadfenceSystem() {
      assert(false && "threadfenceSystem() called on Iluvatar host pass");
    }
    static inline void threadfenceDevice() {
      assert(false && "threadfenceDevice() called on Iluvatar host pass");
    }
#endif

    template <typename DstPtr, typename SrcPtr>
    FLAGCX_DEVICE_INLINE_DECORATOR static void
    coopCopyBytes(DstPtr dst, SrcPtr src, size_t bytes, int rank, int size) {
      // Match the CUDA path's rule that typed accesses are used only when both
      // operands satisfy the access width's alignment. Device API offsets are
      // byte-granular, so the allocation's base alignment alone is not enough.
      if ((bytes & 3u) == 0 &&
          (reinterpret_cast<uintptr_t>(dst) & (alignof(uint32_t) - 1u)) == 0 &&
          (reinterpret_cast<uintptr_t>(src) & (alignof(uint32_t) - 1u)) == 0) {
        // uint32_t is the copy element width, not a lane-mask type. Widening
        // flagcxLaneMask_t to uint64_t therefore does not affect this cast.
        auto dw = FLAGCX_IR_GLOBAL_PTR_CAST(uint32_t, dst);
        auto sw = FLAGCX_IR_GLOBAL_PTR_CAST(const uint32_t, src);
        const size_t nwords = bytes / sizeof(uint32_t);
        for (size_t i = static_cast<size_t>(rank); i < nwords;
             i += static_cast<size_t>(size))
          dw[i] = sw[i];
        return;
      }
      auto db = FLAGCX_IR_GLOBAL_PTR_CAST(unsigned char, dst);
      auto sb = FLAGCX_IR_GLOBAL_PTR_CAST(const unsigned char, src);
      for (size_t i = static_cast<size_t>(rank); i < bytes;
           i += static_cast<size_t>(size))
        db[i] = sb[i];
    }
  };

  // Block-scope fence, kept backend-private on purpose: this fix must not
  // require editing shared device_utils.h. On the device pass
  // __threadfence_block lowers to a bare sl_wait (measured on ivcore11); on the
  // host pass it must vanish, mirroring the host branch of the shared macro.
#if defined(FLAGCX_ILUVATAR_DEVICE_COMPILE)
#define FLAGCX_ILUVATAR_BLOCK_FENCE() __threadfence_block()
#else
#define FLAGCX_ILUVATAR_BLOCK_FENCE() ((void)0)
#endif

  struct Atomic {
    // FlagTree PR 1184 validates aligned load/store and 32-bit CoreX RMW.
    // RMW is deliberately restricted to uint32_t. DefaultBackend selects this
    // width at compile time; no unsupported 64-bit RMW is emitted for ivcore11.
    //
    // The relaxed/release path used to call __atomic_load_n / __atomic_store_n.
    // ivcore11 has no pattern for AtomicLoad/AtomicStore in address space 1, so
    // that crashed llc with "Cannot select: AtomicLoad ... addrspace 1" for BOTH
    // i32 and i64; every polling loop in a real kernel died with it. Every
    // access is therefore a plain volatile load/store, which carries no ordering
    // of its own: volatile only keeps volatile accesses from being reordered
    // against each other, and does nothing to stop the compiler from hoisting an
    // ordinary payload read above the flag read. The ordering is therefore
    // emitted explicitly, on the side the requested order names: release before
    // the store, acquire after the load.
    //
    // Scope dispatch. A scope names how far a fence has to reach, so the
    // cheapest correct fence differs per level: threads inside one CTA share an
    // L1 (the compiler is the only remaining reordering agent), CTAs on one
    // device share L2, and nothing on this target can order across devices.
    //
    // System is rejected at compile time rather than silently served at device
    // strength. ivcore11 cannot select llvm.nvvm.membar.sys, so a System fence
    // is not a slow fence here -- it is an absent one, and a caller that
    // believes it got cross-device ordering is the exact failure mode this
    // whole file is supposed to prevent. Cross-device ordering must go through
    // the FIFO/Net signalling path instead.
    //
    // The default is Device, matching what this backend can actually deliver.
    // It used to default to System, which promised a level the implementation
    // never provided.
    template <flagcxDeviceScope_t Scope>
    FLAGCX_DEVICE_INLINE_DECORATOR static void scopedFence() {
      static_assert(Scope != flagcxDeviceScopeSystem,
                    "Iluvatar has no system-scope fence: ivcore11 cannot "
                    "select llvm.nvvm.membar.sys and __threadfence_system "
                    "aborts llc. Order cross-device accesses through the "
                    "FIFO/Net signalling path instead of a fence.");
      if (Scope == flagcxDeviceScopeThread) {
        // Same thread: no cache is involved, only instruction scheduling.
        asm volatile("" ::: "memory");
      } else if (Scope == flagcxDeviceScopeBlock) {
        // Measured on ivcore11: __threadfence_block lowers to a bare sl_wait
        // (wait for the wave's outstanding LSA operations) and never to
        // ml_lsa_wbinv, so block scope costs no L2 writeback/invalidate. Warps
        // in one CTA share an L1, so the cache hierarchy plus that wait carries
        // the ordering. A sl_wait issued directly after another one coalesces,
        // which is why an acquire load shows no block/thread split in the ISA
        // while a store does. Only the device level pays for a wbinv.
        FLAGCX_ILUVATAR_BLOCK_FENCE();
      } else {
        FLAGCX_DEVICE_THREAD_FENCE();
      }
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static bool
    hasAcquire(flagcxDeviceMemoryOrder_t order) {
      return order == flagcxDeviceMemoryOrderAcquire ||
             order == flagcxDeviceMemoryOrderAcqRel ||
             order == flagcxDeviceMemoryOrderSeqCst;
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeDevice>
    FLAGCX_DEVICE_INLINE_DECORATOR static T
    load(T *ptr, flagcxDeviceMemoryOrder_t order) {
      const T value = *FLAGCX_IR_GLOBAL_VOLATILE_CAST(T, ptr);
      // The read has to come first: a fence ahead of it would be a release on a
      // load, which orders nothing. The acquire edge only exists after the read
      // whose result the caller is about to act on.
      if (hasAcquire(order)) {
        scopedFence<Scope>();
      }
      return value;
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeDevice>
    FLAGCX_DEVICE_INLINE_DECORATOR static void
    store(T *ptr, const T &value, flagcxDeviceMemoryOrder_t order) {
      // Release publication is the fence plus the volatile store. The condition
      // stays "anything but Relaxed" instead of "Release-or-stronger": a store
      // asked for Acquire then gets a fence it does not strictly need, and
      // serving a stronger edge is allowed where serving a weaker one would not
      // be. Narrowing it would weaken a path that works today for no gain.
      if (order != flagcxDeviceMemoryOrderRelaxed) {
        scopedFence<Scope>();
      }
      *FLAGCX_IR_GLOBAL_VOLATILE_CAST(T, ptr) = value;
    }

    // The six RMW members below still ignore both Scope and order: they lower
    // to __atomic_*_fetch_*(..., __ATOMIC_SEQ_CST), which is device scope on
    // this target. SeqCst is never weaker than what the caller asked for, so
    // this is not a correctness gap in the ordering dimension -- but it cannot
    // express system scope either, and that stays tracked separately.
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeDevice>
    FLAGCX_DEVICE_INLINE_DECORATOR static T
    fetchAdd(T *ptr, const T &value, flagcxDeviceMemoryOrder_t) {
      static_assert(sizeof(T) == sizeof(uint32_t),
                    "CoreX RMW operations require a 32-bit operand");
      (void)Scope;
      return __atomic_fetch_add(ptr, value, __ATOMIC_SEQ_CST);
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeDevice>
    FLAGCX_DEVICE_INLINE_DECORATOR static T
    fetchSub(T *ptr, const T &value, flagcxDeviceMemoryOrder_t) {
      static_assert(sizeof(T) == sizeof(uint32_t),
                    "CoreX RMW operations require a 32-bit operand");
      (void)Scope;
      return __atomic_fetch_sub(ptr, value, __ATOMIC_SEQ_CST);
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeDevice>
    FLAGCX_DEVICE_INLINE_DECORATOR static T fetchOr(T *ptr, const T &value,
                                                    flagcxDeviceMemoryOrder_t) {
      static_assert(sizeof(T) == sizeof(uint32_t),
                    "CoreX RMW operations require a 32-bit operand");
      (void)Scope;
      return __atomic_fetch_or(ptr, value, __ATOMIC_SEQ_CST);
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeDevice>
    FLAGCX_DEVICE_INLINE_DECORATOR static T
    fetchAnd(T *ptr, const T &value, flagcxDeviceMemoryOrder_t) {
      static_assert(sizeof(T) == sizeof(uint32_t),
                    "CoreX RMW operations require a 32-bit operand");
      (void)Scope;
      return __atomic_fetch_and(ptr, value, __ATOMIC_SEQ_CST);
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeDevice>
    FLAGCX_DEVICE_INLINE_DECORATOR static T
    exchange(T *ptr, const T &value, flagcxDeviceMemoryOrder_t) {
      static_assert(sizeof(T) == sizeof(uint32_t),
                    "CoreX RMW operations require a 32-bit operand");
      (void)Scope;
      return __atomic_exchange_n(ptr, value, __ATOMIC_SEQ_CST);
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeDevice>
    FLAGCX_DEVICE_INLINE_DECORATOR static bool
    compareExchange(T *ptr, T &expected, const T &desired,
                    flagcxDeviceMemoryOrder_t) {
      static_assert(sizeof(T) == sizeof(uint32_t),
                    "CoreX RMW operations require a 32-bit operand");
      (void)Scope;
      return __atomic_compare_exchange_n(ptr, &expected, desired, false,
                                         __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    }
  };

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
                  "N must be a power of two and <= CoreX wave width");
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
        __builtin_trap();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return static_cast<int>(FLAGCX_THREAD_IDX_X) - first * Intrin::simtWidth;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const {
      return count * Intrin::simtWidth;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() const {
      const int n = this->size();
      if (first == 0 && n == FLAGCX_BLOCK_DIM_X)
        FLAGCX_DEVICE_SYNC_THREADS();
      else
        __builtin_trap();
    }
  };

  struct CoopLanes {
    flagcxLaneMask_t mask;

    FLAGCX_DEVICE_INLINE_DECORATOR explicit CoopLanes(
        flagcxLaneMask_t value = Intrin::fullMask())
        : mask(value) {
      Intrin::validateMask(mask);
      if (mask == 0)
        __builtin_trap();
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

  // CoreX bitcode cannot reliably relocate the PlatformCoop function-pointer
  // table used by the common implementation. Store the resolved group facts
  // directly and perform only direct intrinsic calls from the switch below.
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
          _syncKind(group.first == 0 && group.size() == FLAGCX_BLOCK_DIM_X
                        ? SyncBlock
                        : SyncUnsupported) {}

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
          __builtin_trap();
      }
    }
  };
};

static_assert(sizeof(PlatformTraits<IluvatarPlatform>::CoopAny) == 24,
              "Iluvatar CoopAny must fit the established opaque layout");
static_assert(alignof(PlatformTraits<IluvatarPlatform>::CoopAny) == 8,
              "Iluvatar CoopAny alignment changed");

#endif // FLAGCX_ILUVATAR_PLATFORM_TRAITS_H_
