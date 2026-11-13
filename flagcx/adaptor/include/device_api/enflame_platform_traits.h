/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Enflame Platform Traits — TOPS/GCU SIMT intrinsics and tops::atomic_ref.
 *
 * Provides PlatformTraits<EnflamePlatform> with:
 *   - Intrin: lane(), lanemaskLt(), activemask(), syncwarp(), popc(),
 *             spinBackoff(), fences and a strided cooperative copy
 *   - Atomic: load/store/fetchAdd/... over tops::atomic_ref with an explicit
 *             memory order and thread scope
 *   - Coop*:  sub-thread and lane cooperative groups
 *
 * Role in the vendor guide:
 *   - PlatformTraits is the P0-Core platform layer and is required on every
 *     Device API backend (Default and CCL alike).
 *   - PlatformTraits never performs communication. CommTraits builds the
 *     communication semantics on top of the primitives below; ECCL's device
 *     API (eccl_comm_traits.h) both consumes this layer and carries its own
 *     GIN primitives.
 *
 * Device pass detection: topscc defines __TOPSCC__. The host pass keeps the
 * layout constants and erases device-only qualifiers so the same traits can be
 * instantiated by the host translation units that materialize flagcxDevComm
 * and flagcxDevMem. Naming follows ECCL's own device layer
 * (src/include/eccl_device/utility.h, coop.h) so that each GCU primitive is
 * spelled once in the stack.
 ************************************************************************/

#ifndef FLAGCX_ENFLAME_PLATFORM_TRAITS_H_
#define FLAGCX_ENFLAME_PLATFORM_TRAITS_H_

#include <cassert>
#include <cstddef>
#include <cstdint>


#if FLAGCX_CHECK_DEVICE_CC

namespace tops {
  enum thread_scope {
    thread_scope_thread  = 0,
    thread_scope_block   = 1,
    thread_scope_device  = 2,
    thread_scope_system  = 3,
  };

  enum memory_order {
    memory_order_relaxed = __ATOMIC_RELAXED,
    memory_order_consume = __ATOMIC_CONSUME,
    memory_order_acquire = __ATOMIC_ACQUIRE,
    memory_order_release = __ATOMIC_RELEASE,
    memory_order_acq_rel = __ATOMIC_ACQ_REL,
    memory_order_seq_cst = __ATOMIC_SEQ_CST,
  };

  template <typename T, thread_scope Scope = thread_scope_system>
  struct atomic_ref {
    T &ref_;

    FLAGCX_DEVICE_INLINE_DECORATOR explicit atomic_ref(T &r) : ref_(r) {}

    FLAGCX_DEVICE_INLINE_DECORATOR T load(memory_order) const { return T{}; }
    FLAGCX_DEVICE_INLINE_DECORATOR void store(const T &, memory_order) {}
    FLAGCX_DEVICE_INLINE_DECORATOR T fetch_add(const T &, memory_order) { return T{}; }
    FLAGCX_DEVICE_INLINE_DECORATOR T fetch_sub(const T &, memory_order) { return T{}; }
    FLAGCX_DEVICE_INLINE_DECORATOR T fetch_or(const T &, memory_order) { return T{}; }
    FLAGCX_DEVICE_INLINE_DECORATOR T fetch_and(const T &, memory_order) { return T{}; }
    FLAGCX_DEVICE_INLINE_DECORATOR T exchange(const T &, memory_order) { return T{}; }
    FLAGCX_DEVICE_INLINE_DECORATOR bool compare_exchange_strong(T &, const T &, memory_order, memory_order) {
      return false;
    }
  };
} // namespace tops

#endif


struct EnflamePlatform {};

template <>
struct PlatformTraits<EnflamePlatform> {

  // ==============================================================
  // Intrin — TOPS/GCU SIMT intrinsics
  // ==============================================================
  struct Intrin {
    static constexpr int simtWidth = FLAGCX_SIMT_WIDTH;

    static FLAGCX_HOST_DEVICE_INLINE flagcxLaneMask_t fullMask() {
      __builtin_trap();
      return 0xffffffffull;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static void
    validateMask(flagcxLaneMask_t) {
      __builtin_trap();
    }

#if FLAGCX_CHECK_DEVICE_CC
    FLAGCX_DEVICE_INLINE_DECORATOR static int lane() {
      __builtin_trap();
      return 0;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static flagcxLaneMask_t lanemaskLt() {
      __builtin_trap();
      return 0;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static flagcxLaneMask_t activemask() {
      __builtin_trap();
      return 0;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static int popc(flagcxLaneMask_t) {
      __builtin_trap();
      return 0;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static void
    syncwarp(flagcxLaneMask_t) {
      __builtin_trap();
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static void namedBarrierSync(int, int) {
      __builtin_trap();
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static void spinBackoff(int) {
      __builtin_trap();
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static void threadfenceSystem() {
      __builtin_trap();
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static void threadfenceDevice() {
      __builtin_trap();
    }

    template <typename DstPtr, typename SrcPtr>
    static FLAGCX_DEVICE_INLINE_DECORATOR void
    coopCopyBytes(DstPtr, SrcPtr, size_t, int, int) {
      __builtin_trap();
    }
#else
    // Host pass: retain the constants, erase the device builtins.
    static inline int lane() {
      assert(false && "lane() called on Enflame host pass");
      return 0;
    }
    static inline flagcxLaneMask_t lanemaskLt() {
      assert(false && "lanemaskLt() called on Enflame host pass");
      return 0;
    }
    static inline flagcxLaneMask_t activemask() {
      assert(false && "activemask() called on Enflame host pass");
      return fullMask();
    }
    static inline int popc(flagcxLaneMask_t x) {
      return __builtin_popcountll(x);
    }
    static inline void syncwarp(flagcxLaneMask_t = fullMask()) {
      assert(false && "syncwarp() called on Enflame host pass");
    }
    static inline void namedBarrierSync(int, int) {
      assert(false && "namedBarrierSync() called on Enflame host pass");
    }
    static inline void spinBackoff(int) {}
    static inline void threadfenceSystem() {
      assert(false && "threadfenceSystem() called on Enflame host pass");
    }
    static inline void threadfenceDevice() {
      assert(false && "threadfenceDevice() called on Enflame host pass");
    }
    template <typename DstPtr, typename SrcPtr>
    static inline void coopCopyBytes(DstPtr, SrcPtr, size_t, int, int) {
      assert(false && "coopCopyBytes() called on Enflame host pass");
    }
#endif // FLAGCX_CHECK_DEVICE_CC
  };

  // ==============================================================
  // Atomic — tops::atomic_ref with explicit order and scope
  //
  // Signal, counter, FIFO and barrier correctness all rest on the acquire /
  // release halves, not merely on atomicity: an implementation that stores a
  // value atomically but drops the release ordering produces cross-rank hangs
  // that are very hard to reproduce.
  // ==============================================================
  struct Atomic {
#if FLAGCX_CHECK_DEVICE_CC
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T
    load(T *, flagcxDeviceMemoryOrder_t) {
      __builtin_trap();
      return T{};
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR void
    store(T *, const T &, flagcxDeviceMemoryOrder_t) {
      __builtin_trap();
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T
    fetchAdd(T *, const T &, flagcxDeviceMemoryOrder_t) {
      __builtin_trap();
      return T{};
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T
    fetchSub(T *, const T &, flagcxDeviceMemoryOrder_t) {
      __builtin_trap();
      return T{};
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T
    fetchOr(T *, const T &, flagcxDeviceMemoryOrder_t) {
      __builtin_trap();
      return T{};
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T
    fetchAnd(T *, const T &, flagcxDeviceMemoryOrder_t) {
      __builtin_trap();
      return T{};
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR T
    exchange(T *, const T &, flagcxDeviceMemoryOrder_t) {
      __builtin_trap();
      return T{};
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    static FLAGCX_DEVICE_INLINE_DECORATOR bool
    compareExchange(T *, T &, const T &, flagcxDeviceMemoryOrder_t) {
      __builtin_trap();
      return false;
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR tops::memory_order
    toNativeOrder(flagcxDeviceMemoryOrder_t) {
      __builtin_trap();
      return tops::memory_order_seq_cst;
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR tops::thread_scope
    toNativeScope(flagcxDeviceScope_t) {
      __builtin_trap();
      return tops::thread_scope_system;
    }
#else
    // Host pass: GCC builtins keep the same signatures so host-side code that
    // instantiates the traits still compiles.
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
#endif // FLAGCX_CHECK_DEVICE_CC
  };

#if FLAGCX_CHECK_DEVICE_CC
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
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      __builtin_trap();
      return 0;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const {
      __builtin_trap();
      return N;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxLaneMask_t laneMask() const {
      __builtin_trap();
      return 0;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() {
      __builtin_trap();
    }
  };

  using CoopThread = CoopTile<1>;
  using CoopWarp = CoopTile<8>;

  struct CoopTileSpan {
    uint32_t t0 : 8, nTiles : 8, id : 8;
    FLAGCX_DEVICE_INLINE_DECORATOR CoopTileSpan(int t0, int nTiles, int id)
        : t0(t0), nTiles(nTiles), id(id) {
      __builtin_trap();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      __builtin_trap();
      return 0;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const {
      __builtin_trap();
      return 0;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() {
      __builtin_trap();
    }
    // ECCL's ecclCoopAny requires the snake_case spellings; comm_traits.h hands
    // this type to it without unwrapping, so both spellings must exist.
    FLAGCX_DEVICE_INLINE_DECORATOR int thread_rank() const {
      return threadRank();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int num_threads() const { return size(); }
  };

  struct CoopLanes {
    flagcxLaneMask_t lmask;
    FLAGCX_DEVICE_INLINE_DECORATOR
    CoopLanes(flagcxLaneMask_t lmask = Intrin::fullMask()) : lmask(lmask) {
      __builtin_trap();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      __builtin_trap();
      return 0;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const {
      __builtin_trap();
      return 0;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() {
      __builtin_trap();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxLaneMask_t getLmask() const {
      __builtin_trap();
      return 0;
    }
  };

  using CoopAny = PlatformCoop;
#else
  // Host pass: single-threaded no-ops, as on every other platform.
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
  using CoopWarp = CoopTile<8>;
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
#endif // FLAGCX_CHECK_DEVICE_CC

  // ==============================================================
  // CoopOps — placeholder until Enflame cooperative operations are
  // implemented.  An empty struct satisfies the static_assert in
  // flagcx_device_core.h (sizeof > 0) without providing any real
  // capability.
  // ==============================================================
  struct CoopOps {};

  // ==============================================================
  // Clock — placeholder until Enflame device-side time source is
  // implemented.
  // ==============================================================
  struct Clock {};
};

#endif // FLAGCX_ENFLAME_PLATFORM_TRAITS_H_
