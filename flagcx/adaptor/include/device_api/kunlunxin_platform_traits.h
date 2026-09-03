/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * KunlunXin/XPU platform traits for the common FlagCX Device API.
 ************************************************************************/

#ifndef FLAGCX_KUNLUNXIN_PLATFORM_TRAITS_H_
#define FLAGCX_KUNLUNXIN_PLATFORM_TRAITS_H_

#if defined(__xpu__)
#include <cstddef>
// XTDK ships no usable non-allocating placement new: in both compilation passes
// `::new (p) T(...)` fails with "no matching 'operator new' function for
// non-allocating placement new expression", and #include <new> does not help --
// libstdc++'s inline definition is never a candidate under this driver. A
// __device__-qualified declaration is (an unqualified one still is not), so
// declare it here and let the shared device code keep the ordinary spelling.
__device__ inline void *operator new(std::size_t, void *place) noexcept {
  return place;
}
#endif

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

    // xpu3 has a single fence instruction; there is no weaker device-scope
    // form to map this onto, so it is the same mfence.
    FLAGCX_DEVICE_INLINE_DECORATOR static void threadfenceDevice() {
      FLAGCX_DEVICE_THREAD_FENCE();
    }

    // Cooperative strided copy between two device-memory buffers.
    //
    // A device-memory pointer cannot be converted to an integer on the xpu3
    // device pass, so the chunk width is chosen from the byte count alone
    // rather than from the address alignment. Both operands come from the
    // symmetric heap and are therefore at least 64B aligned.
    template <typename DstPtr, typename SrcPtr>
    FLAGCX_DEVICE_INLINE_DECORATOR static void
    coopCopyBytes(DstPtr dst, SrcPtr src, size_t bytes, int rank, int size) {
      if ((bytes & 3u) == 0) {
        auto dw = (FLAGCX_DEV_VALUE_PTR uint32_t *)dst;
        auto sw = (FLAGCX_DEV_VALUE_PTR const uint32_t *)src;
        size_t nwords = bytes / 4;
        for (size_t i = (size_t)rank; i < nwords; i += (size_t)size)
          dw[i] = sw[i];
        return;
      }
      auto db = (FLAGCX_DEV_VALUE_PTR char *)dst;
      auto sb = (FLAGCX_DEV_VALUE_PTR const char *)src;
      for (size_t i = (size_t)rank; i < bytes; i += (size_t)size)
        db[i] = sb[i];
    }

    // Volatile 64-bit store to device memory. Kept here because the cast needs
    // the platform's address-space qualifier, which the shared IR code cannot
    // spell.
    template <typename Ptr>
    FLAGCX_DEVICE_INLINE_DECORATOR static void storeVolatile64(Ptr ptr,
                                                               uint64_t value) {
      *(volatile FLAGCX_DEV_VALUE_PTR uint64_t *)ptr = value;
    }
  };

  // P800 has no atomic read-modify-write on global memory, and the pointers
  // reaching these entry points are device-memory pointers (address space 1 on
  // the xpu3 device pass), which the __atomic_* builtins cannot take at all.
  //
  // Aligned 64-bit loads and stores are single instructions and are what the
  // single-writer signal/counter schemes already rely on, so they are provided.
  // Every read-modify-write traps instead of emulating atomicity: a
  // load-modify-store pair would look like it worked and silently lose
  // increments. Callers must probe the capability (Comm::usesDirectP2pSignals,
  // Comm::supportsDirectCounterAccess) and take a single-writer path instead.
  struct Atomic {
    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    FLAGCX_DEVICE_INLINE_DECORATOR static T
    load(FLAGCX_DEV_VALUE_PTR T *ptr, flagcxDeviceMemoryOrder_t) {
      T value = *(volatile FLAGCX_DEV_VALUE_PTR T *)ptr;
      FLAGCX_DEVICE_THREAD_FENCE();
      return value;
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    FLAGCX_DEVICE_INLINE_DECORATOR static void
    store(FLAGCX_DEV_VALUE_PTR T *ptr, const T &value,
          flagcxDeviceMemoryOrder_t) {
      FLAGCX_DEVICE_THREAD_FENCE();
      *(volatile FLAGCX_DEV_VALUE_PTR T *)ptr = value;
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    FLAGCX_DEVICE_INLINE_DECORATOR static T
    fetchAdd(FLAGCX_DEV_VALUE_PTR T *, const T &, flagcxDeviceMemoryOrder_t) {
      __builtin_trap();
      return T();
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    FLAGCX_DEVICE_INLINE_DECORATOR static T
    fetchSub(FLAGCX_DEV_VALUE_PTR T *, const T &, flagcxDeviceMemoryOrder_t) {
      __builtin_trap();
      return T();
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    FLAGCX_DEVICE_INLINE_DECORATOR static T
    fetchOr(FLAGCX_DEV_VALUE_PTR T *, const T &, flagcxDeviceMemoryOrder_t) {
      __builtin_trap();
      return T();
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    FLAGCX_DEVICE_INLINE_DECORATOR static T
    fetchAnd(FLAGCX_DEV_VALUE_PTR T *, const T &, flagcxDeviceMemoryOrder_t) {
      __builtin_trap();
      return T();
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    FLAGCX_DEVICE_INLINE_DECORATOR static T
    exchange(FLAGCX_DEV_VALUE_PTR T *, const T &, flagcxDeviceMemoryOrder_t) {
      __builtin_trap();
      return T();
    }

    template <typename T, flagcxDeviceScope_t Scope = flagcxDeviceScopeSystem>
    FLAGCX_DEVICE_INLINE_DECORATOR static bool
    compareExchange(FLAGCX_DEV_VALUE_PTR T *, T &, const T &,
                    flagcxDeviceMemoryOrder_t) {
      __builtin_trap();
      return false;
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
      // this-> : xpu-clang drops const on member calls from const members of
      // classes nested in templates (here CoopTileSpan lives inside
      // PlatformTraits<KunlunxinPlatform>).
      if (this->size() >= FLAGCX_BLOCK_DIM_X)
        FLAGCX_DEVICE_SYNC_THREADS();
      else if (this->size() > 1)
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

  // Type-erased cooperative group.
  //
  // The shared PlatformCoop erases the type behind a vtable of function
  // pointers. That cannot work on xpu3: the vtable is constant-memory data and
  // the pointer to it is held in a generic pointer, which is 32-bit on the
  // device pass, so any call that survives inlining jumps through a truncated
  // address and the kernel faults (XPUERR_KEXCEPTION). Every XPU coop type is
  // fully described by (threadRank, size, how to synchronize), and all of them
  // synchronize the same way — the whole cluster or nothing, since P800 has no
  // partial-core barrier — so resolve those three properties at construction
  // and dispatch with a switch.
  struct CoopAny {
    enum SyncKind : int {
      SyncNone = 0,        // single core: nothing to synchronize
      SyncCluster = 1,     // group spans the cluster: sync_cluster()
      SyncUnsupported = 2, // proper subset of the cluster: no such barrier
    };

    int _rank;
    int _size;
    int _sync;

    FLAGCX_DEVICE_INLINE_DECORATOR CoopAny()
        : _rank(0), _size(1), _sync(SyncNone) {}
    CoopAny(CoopAny const &) = default;

    template <typename Impl>
    FLAGCX_DEVICE_INLINE_DECORATOR CoopAny(Impl impl)
        : _rank(impl.threadRank()), _size(impl.size()),
          _sync(impl.size() >= FLAGCX_BLOCK_DIM_X
                    ? SyncCluster
                    : (impl.size() > 1 ? SyncUnsupported : SyncNone)) {}

    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const { return _rank; }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return _size; }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() const {
      if (_sync == SyncCluster)
        FLAGCX_DEVICE_SYNC_THREADS();
      else if (_sync == SyncUnsupported)
        __builtin_trap();
    }
  };
};

#endif // FLAGCX_KUNLUNXIN_PLATFORM_TRAITS_H_
