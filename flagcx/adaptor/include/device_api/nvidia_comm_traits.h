/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA Vendor Device Traits — wraps NCCL device API types.
 *
 * CommTraits<NvidiaVendor> provides:
 *   - Intrin, Atomic: from PlatformTraits<NvidiaPlatform> via using
 *   - Window:   wraps ncclWindow_t with member functions
 *   - DevComm:  wraps ncclDevComm with member functions
 *   - Team:     wraps ncclTeam_t with member functions
 *   - Multimem: wraps ncclMultimemHandle_t
 *
 * Also defines FLAGCX_DEVICE_API_VENDOR and the DeviceAPI selection.
 ************************************************************************/

#ifndef FLAGCX_NVIDIA_DEVICE_TRAITS_H_
#define FLAGCX_NVIDIA_DEVICE_TRAITS_H_

#include "nccl.h"

// ============================================================
// NVIDIA Vendor Backend (NCCL device API)
// ============================================================
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0) &&                              \
    !defined(FLAGCX_FORCE_FALLBACK)

#include "nccl_device.h"

struct NvidiaVendor {};

template <>
struct CommTraits<NvidiaVendor> {
  // Platform capabilities (via using, not inheritance)
  using Intrin = PlatformTraits<NvidiaPlatform>::Intrin;
  using Atomic = PlatformTraits<NvidiaPlatform>::Atomic;

  // ---- Team: wraps ncclTeam_t ----
  // Exposes nRanks/rank/stride for direct field access (used by flagcxTeam)
  struct Team {
    int nRanks, rank, stride;

    FLAGCX_HOST_DEVICE_INLINE Team() : nRanks(0), rank(0), stride(0) {}
    FLAGCX_HOST_DEVICE_INLINE Team(int nr, int r, int s)
        : nRanks(nr), rank(r), stride(s) {}

    // Implicit conversion to ncclTeam_t for NCCL API calls
    FLAGCX_HOST_DEVICE_INLINE operator ncclTeam_t() const {
      ncclTeam_t t;
      t.nRanks = nRanks;
      t.rank = rank;
      t.stride = stride;
      return t;
    }
  };

  // ---- Multimem: wraps ncclMultimemHandle_t ----
  struct Multimem {
    ncclMultimemHandle_t _impl;

    FLAGCX_HOST_DEVICE_INLINE Multimem() : _impl() {}

    // Implicit conversion for NCCL API calls
    FLAGCX_HOST_DEVICE_INLINE operator ncclMultimemHandle_t() const {
      return _impl;
    }
  };

  // ---- Window: wraps ncclWindow_t ----
  struct Window {
    ncclWindow_t _impl;

    FLAGCX_HOST_DEVICE_INLINE Window() : _impl() {}

    FLAGCX_DEVICE_INLINE_DECORATOR void *
    getPeerPointer(size_t offset, const Team &team, int peer) const {
      return ncclGetPeerPointer(_impl, offset, (ncclTeam_t)team, peer);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void *getLocalPointer(size_t offset) const {
      return ncclGetLocalPointer(_impl, offset);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void *getIntraPointer(size_t offset,
                                                         int peer) const {
      return ncclGetLsaPointer(_impl, offset, peer);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void *
    getMulticastPointer(size_t offset, const Multimem &mm) const {
      return ncclGetMultimemPointer(_impl, offset, mm._impl);
    }

    FLAGCX_HOST_DEVICE_INLINE bool hasAccess() const {
      return _impl.base != 0 || _impl.size != 0;
    }
    FLAGCX_HOST_DEVICE_INLINE void *getRawPtr() const {
      return (void *)_impl.base;
    }
    FLAGCX_HOST_DEVICE_INLINE void **getDevPeerPtrs() const { return nullptr; }
    FLAGCX_HOST_DEVICE_INLINE int getMrIndex() const { return -1; }

    FLAGCX_DEVICE_INLINE_DECORATOR bool operator==(const Window &o) const {
      return _impl.base == o._impl.base && _impl.size == o._impl.size;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR bool operator!=(const Window &o) const {
      return !(*this == o);
    }
  };

  // ---- DevComm: wraps ncclDevComm ----
  struct DevComm {
    ncclDevComm _impl;

    FLAGCX_HOST_DEVICE_INLINE DevComm() : _impl() {}

    // Implicit conversion to ncclDevComm for NCCL API calls
    FLAGCX_HOST_DEVICE_INLINE operator const ncclDevComm &() const {
      return _impl;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR int getIntraRank() const {
      return _impl.lsaRank;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int getIntraSize() const {
      return _impl.lsaSize;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int getRank() const { return _impl.rank; }
    FLAGCX_DEVICE_INLINE_DECORATOR int getSize() const { return _impl.nRanks; }
    FLAGCX_DEVICE_INLINE_DECORATOR void *getFifoBuffer() const {
      return nullptr;
    }

    // No-op: vendor DevComm is populated via devComm pointer cast
    template <typename DI>
    static FLAGCX_HOST_DEVICE_INLINE void populateFromInternal(DevComm &,
                                                               const DI &) {}
  };

  // ---- CoopBlock: wraps ncclCoopCta ----
  struct CoopBlock {
    ncclCoopCta _impl;

    FLAGCX_HOST_DEVICE_INLINE CoopBlock() : _impl() {}

    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return _impl.thread_rank();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return _impl.size(); }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _impl.sync(); }
  };

  // ---- CoopTile<N>: wraps ncclCoopTile<N> ----
  template <int N>
  struct CoopTile {
    ncclCoopTile<N> _impl;

    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return _impl.thread_rank();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return N; }
    FLAGCX_DEVICE_INLINE_DECORATOR uint32_t laneMask() const {
      return _impl.laneMask();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _impl.sync(); }
  };

  using CoopThread = CoopTile<1>;
  using CoopWarp = CoopTile<32>;

  // ---- CoopTileSpan: wraps ncclCoopWarpSpan ----
  struct CoopTileSpan {
    ncclCoopWarpSpan _impl;

    FLAGCX_DEVICE_INLINE_DECORATOR CoopTileSpan(int t0, int nTiles, int id)
        : _impl(t0, nTiles, id) {}

    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return _impl.thread_rank();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return _impl.size(); }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _impl.sync(); }
  };

  // ---- CoopLanes: wraps ncclCoopLanes ----
  struct CoopLanes {
    ncclCoopLanes _impl;

    FLAGCX_DEVICE_INLINE_DECORATOR CoopLanes(uint32_t lmask = 0xffffffffu)
        : _impl{lmask} {}

    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return _impl.thread_rank();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return _impl.size(); }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _impl.sync(); }
    FLAGCX_DEVICE_INLINE_DECORATOR uint32_t getLmask() const {
      return _impl.lmask;
    }
  };

  // ---- CoopAny: wraps ncclCoopAny ----
  struct CoopAny {
    ncclCoopAny _impl;

    CoopAny() = default;
    CoopAny(CoopAny const &) = default;

    FLAGCX_DEVICE_INLINE_DECORATOR CoopAny(CoopBlock b) : _impl(b._impl) {}
    template <int N>
    FLAGCX_DEVICE_INLINE_DECORATOR CoopAny(CoopTile<N> t) : _impl(t._impl) {}
    FLAGCX_DEVICE_INLINE_DECORATOR CoopAny(CoopTileSpan s) : _impl(s._impl) {}
    FLAGCX_DEVICE_INLINE_DECORATOR CoopAny(CoopLanes l) : _impl(l._impl) {}

    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return _impl.thread_rank();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return _impl.size(); }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _impl.sync(); }
  };

  // ---- Barrier handles ----
  struct IntraBarrierHandle {
    ncclLsaBarrierHandle _impl;
  };
  struct InterBarrierHandle {
    ncclGinBarrierHandle _impl;
  };

  // ---- Barrier / GIN type aliases ----
  using Barrier = ncclLsaBarrier;
  using RemoteAction = ncclGinRemoteAction;
  using LocalAction = ncclGinLocalAction;
  using FenceLevel = ncclGinFenceLevel;

  // ---- DevBarrier alias: delegates to standalone DevBarrier<Backend, Tag>
  // ----
  template <typename Tag, typename Coop>
  using DevBarrier = ::DevBarrier<NvidiaVendor, Tag, Coop>;

  // ---- Action type conversion helpers (flagcx -> NCCL) ----
  FLAGCX_DEVICE_INLINE_DECORATOR static ncclGin_None toNccl(flagcxDevNet_None) {
    return {};
  }
  FLAGCX_DEVICE_INLINE_DECORATOR static ncclGin_SignalInc
  toNccl(flagcxDevNet_SignalInc a) {
    return {a.signal};
  }
  FLAGCX_DEVICE_INLINE_DECORATOR static ncclGin_SignalAdd
  toNccl(flagcxDevNet_SignalAdd a) {
    return {a.signal, a.value};
  }
  FLAGCX_DEVICE_INLINE_DECORATOR static ncclGin_CounterInc
  toNccl(flagcxDevNet_CounterInc a) {
    return {a.counter};
  }
  FLAGCX_DEVICE_INLINE_DECORATOR static ncclGin_DescriptorSmem
  toNccl(flagcxDevNet_DescriptorSmem a) {
    return {(ncclGinDescriptorSmem *)a.smem._impl};
  }

  // ---- Net: wraps ncclGin for one-sided operations ----
  struct Net {
    ncclGin _gin;
    int _contextId;

    FLAGCX_DEVICE_INLINE_DECORATOR
    Net(const DevComm &dc, int contextIndex)
        : _gin(dc, contextIndex), _contextId(contextIndex) {}

    // --- One-sided: put (raw Window) ---
    template <typename RA, typename LA, typename Coop, typename Desc>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    put(Team team, int peer, Window dst, size_t dstOff, Window src,
        size_t srcOff, size_t bytes, RA ra, LA la, Coop coop, Desc desc,
        flagcxDeviceScope_t ar, flagcxDeviceScope_t es) const {
      _gin.put((ncclTeam_t)team, peer, dst._impl, dstOff, src._impl, srcOff,
               bytes, toNccl(ra), toNccl(la), coop._impl, toNccl(desc),
               Atomic::toNativeScope(ar), Atomic::toNativeScope(es));
    }

    // --- One-sided: putValue ---
    template <typename T, typename RA, typename Coop, typename Desc>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    putValue(Team team, int peer, Window dst, size_t dstOff, T value, RA ra,
             Coop coop, Desc desc, flagcxDeviceScope_t ar,
             flagcxDeviceScope_t es) const {
      _gin.putValue((ncclTeam_t)team, peer, dst._impl, dstOff, value,
                    toNccl(ra), coop._impl, toNccl(desc),
                    Atomic::toNativeScope(ar), Atomic::toNativeScope(es));
    }

    // --- One-sided: signal ---
    template <typename RA, typename Coop, typename Desc>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    signal(Team team, int peer, RA ra, Coop coop, Desc desc,
           flagcxDeviceScope_t ar, flagcxDeviceScope_t es) const {
      _gin.signal((ncclTeam_t)team, peer, toNccl(ra), coop._impl, toNccl(desc),
                  Atomic::toNativeScope(ar), Atomic::toNativeScope(es));
    }

    // --- One-sided: flush ---
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    flush(Coop coop, flagcxDeviceMemoryOrder_t order) const {
      _gin.flush(coop._impl, Atomic::toNativeOrder(order));
    }

    // --- One-sided: waitSignal ---
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitSignal(Coop coop, flagcxDevNetSignal_t signal, uint64_t least, int bits,
               flagcxDeviceMemoryOrder_t order) const {
      _gin.waitSignal(coop._impl, signal, least, bits,
                      Atomic::toNativeOrder(order));
    }

    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitSignalMeetShadow(Coop coop, flagcxDevNetSignal_t signal, int bits,
                         flagcxDeviceMemoryOrder_t order) const {
      _gin.waitSignalMeetShadow(coop._impl, signal, bits,
                                Atomic::toNativeOrder(order));
    }

    template <typename Coop, typename Uint>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitSignalFollowShadow(Coop coop, flagcxDevNetSignal_t signal,
                           Uint leastDelta, Uint *before, Uint *delta, int bits,
                           flagcxDeviceMemoryOrder_t order) const {
      _gin.waitSignalFollowShadow(coop._impl, signal, leastDelta, before, delta,
                                  bits, Atomic::toNativeOrder(order));
    }

    // --- Shadow manipulation ---
    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t *
    getSignalShadowPtr(flagcxDevNetSignal_t signal) const {
      return _gin.getSignalShadowPtr(signal);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    increaseSignalShadow(flagcxDevNetSignal_t signal, uint64_t delta) const {
      _gin.increaseSignalShadow(signal, delta);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
    readSignal(flagcxDevNetSignal_t signal, int bits,
               flagcxDeviceMemoryOrder_t order) const {
      return _gin.readSignal(signal, bits, Atomic::toNativeOrder(order));
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    resetSignal(flagcxDevNetSignal_t signal) const {
      _gin.resetSignal(signal);
    }

    // --- Counter ---
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitCounter(Coop coop, flagcxDevNetCounter_t counter, uint64_t least,
                int bits, flagcxDeviceMemoryOrder_t order) const {
      _gin.waitCounter(coop._impl, counter, least, bits,
                       Atomic::toNativeOrder(order));
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
    readCounter(flagcxDevNetCounter_t counter, int bits,
                flagcxDeviceMemoryOrder_t order) const {
      return _gin.readCounter(counter, bits, Atomic::toNativeOrder(order));
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    resetCounter(flagcxDevNetCounter_t counter) const {
      _gin.resetCounter(counter);
    }

    // --- Two-sided stubs (never called on vendor, exist for compilation) ---
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t send(Coop, Window, size_t,
                                                       size_t, flagcxDataType_t,
                                                       int) const {
      return flagcxInternalError;
    }
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t recv(Coop, Window, size_t,
                                                       size_t, flagcxDataType_t,
                                                       int) const {
      return flagcxInternalError;
    }
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t term(Coop) const {
      return flagcxInternalError;
    }
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t wait(Coop) const {
      return flagcxInternalError;
    }

    // --- get stub (fallback-only, vendor has no RDMA READ) ---
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void get(Team, int, Window, size_t, Window,
                                            size_t, size_t, Coop) const {}
  };
};

// Fence level mapping (file scope for CUDA __constant__ compatibility)
#ifdef FLAGCX_DEVICE_COMPILE
FLAGCX_MAYBE_UNUSED static FLAGCX_DEVICE_CONSTANT_DECORATOR ncclGinFenceLevel
    flagcxGinFenceLevelMap[] = {ncclGinFenceLevel::Relaxed};
static_assert(
    sizeof(flagcxGinFenceLevelMap) / sizeof(flagcxGinFenceLevelMap[0]) ==
        static_cast<int>(flagcxGinFenceLevel::Relaxed) + 1,
    "flagcxGinFenceLevelMap must cover all flagcxGinFenceLevel values");
#endif

// ============================================================
// DevBarrier specializations for NvidiaVendor
// ============================================================

// ---- DevBarrier<NvidiaVendor, flagcxBarrierIntra, Coop> ----
// Wraps ncclLsaBarrierSession<ncclCoopCta>.
template <typename Coop>
struct DevBarrier<NvidiaVendor, flagcxBarrierIntra, Coop> {
  using Atomic = PlatformTraits<NvidiaPlatform>::Atomic;
  using DevComm = CommTraits<NvidiaVendor>::DevComm;
  using Team = CommTraits<NvidiaVendor>::Team;
  using Multimem = CommTraits<NvidiaVendor>::Multimem;

  Coop _coop;
  ncclLsaBarrierSession<ncclCoopCta> _impl;

  // Default ctor
  FLAGCX_DEVICE_INLINE_DECORATOR
  DevBarrier() : _coop(), _impl() {}

  // Active ctor
  FLAGCX_DEVICE_INLINE_DECORATOR
  DevBarrier(Coop coop, const DevComm &dc, Team team, uint32_t index,
             bool multimem = false, const Multimem &mm = {})
      : _coop(coop), _impl(ncclCoopCta(), dc, ncclTeamLsa(dc),
                           dc._impl.lsaBarrier, index, multimem, mm._impl) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    _impl.arrive(ncclCoopCta(), Atomic::toNativeOrder(order));
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    _impl.wait(ncclCoopCta(), Atomic::toNativeOrder(order));
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    _impl.sync(ncclCoopCta(), Atomic::toNativeOrder(order));
  }
};

// ---- DevBarrier<NvidiaVendor, flagcxBarrierInter, Coop> ----
// Wraps ncclGinBarrierSession<ncclCoopCta> via placement new.
template <typename Coop>
struct DevBarrier<NvidiaVendor, flagcxBarrierInter, Coop> {
  using Atomic = PlatformTraits<NvidiaPlatform>::Atomic;
  using DevComm = CommTraits<NvidiaVendor>::DevComm;
  using Team = CommTraits<NvidiaVendor>::Team;
  using Net = CommTraits<NvidiaVendor>::Net;

  Coop _coop;
  alignas(ncclGinBarrierSession<ncclCoopCta>) char _implStorage[sizeof(
      ncclGinBarrierSession<ncclCoopCta>)];
  int _nInterPeers;

  // Default ctor (no-op barrier)
  FLAGCX_DEVICE_INLINE_DECORATOR
  DevBarrier() : _coop(), _nInterPeers(0) {}

  // Active ctor
  FLAGCX_DEVICE_INLINE_DECORATOR
  DevBarrier(Coop coop, const Net &net, const DevComm &dc, Team team,
             uint32_t index, int nInterPeers)
      : _coop(coop), _nInterPeers(nInterPeers) {
    new (_implStorage) ncclGinBarrierSession<ncclCoopCta>(
        ncclCoopCta(), net._gin, team, net._gin.comm.railGinBarrier, index);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
         flagcxGinFenceLevel fence = flagcxGinFenceLevel::Relaxed) {
    // ncclGinBarrierSession only exposes sync; arrive is a no-op on vendor
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxGinFenceLevel fence = flagcxGinFenceLevel::Relaxed) {
    // ncclGinBarrierSession only exposes sync; wait is a no-op on vendor
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxGinFenceLevel fence = flagcxGinFenceLevel::Relaxed) {
    if (_nInterPeers > 0) {
      reinterpret_cast<ncclGinBarrierSession<ncclCoopCta> *>(_implStorage)
          ->sync(ncclCoopCta(), Atomic::toNativeOrder(order),
                 flagcxGinFenceLevelMap[static_cast<int>(fence)]);
    }
  }
};

// ---- DevBarrier<NvidiaVendor, flagcxBarrierWorld, Coop> ----
// World: wraps ncclBarrierSession. Intra: wraps ncclLsaBarrierSession.
// Uses placement new for the union of both types.
template <typename Coop>
struct DevBarrier<NvidiaVendor, flagcxBarrierWorld, Coop> {
  using Atomic = PlatformTraits<NvidiaPlatform>::Atomic;
  using DevComm = CommTraits<NvidiaVendor>::DevComm;
  using Net = CommTraits<NvidiaVendor>::Net;

  // Storage large enough for the larger of the two session types
  static constexpr size_t kWorldSize = sizeof(ncclBarrierSession<ncclCoopCta>);
  static constexpr size_t kIntraSize =
      sizeof(ncclLsaBarrierSession<ncclCoopCta>);
  static constexpr size_t kMaxSize =
      (kWorldSize > kIntraSize) ? kWorldSize : kIntraSize;
  static constexpr size_t kWorldAlign =
      alignof(ncclBarrierSession<ncclCoopCta>);
  static constexpr size_t kIntraAlign =
      alignof(ncclLsaBarrierSession<ncclCoopCta>);
  static constexpr size_t kMaxAlign =
      (kWorldAlign > kIntraAlign) ? kWorldAlign : kIntraAlign;

  Coop _coop;
  alignas(kMaxAlign) char _implStorage[kMaxSize];
  bool _intraOnly;

  // World barrier (intra + inter)
  FLAGCX_DEVICE_INLINE_DECORATOR
  DevBarrier(Coop coop, flagcxBarrierWorld::World, const Net &net,
             const DevComm &dc, uint32_t index, bool multimem, int)
      : _coop(coop), _intraOnly(false) {
    new (_implStorage) ncclBarrierSession<ncclCoopCta>(
        ncclCoopCta(), ncclTeamTagWorld(), net._gin, index, multimem);
  }

  // Intra-only barrier
  FLAGCX_DEVICE_INLINE_DECORATOR
  DevBarrier(Coop coop, flagcxBarrierWorld::Intra, const Net &,
             const DevComm &dc, uint32_t index, bool multimem, int)
      : _coop(coop), _intraOnly(true) {
    new (_implStorage) ncclLsaBarrierSession<ncclCoopCta>(
        ncclCoopCta(), dc, ncclTeamLsa(dc), dc._impl.lsaBarrier, index,
        multimem);
  }

  // Inter-only barrier (ncclTeamTagRail)
  FLAGCX_DEVICE_INLINE_DECORATOR
  DevBarrier(Coop coop, flagcxBarrierWorld::Inter, const Net &net,
             const DevComm &, uint32_t index, bool, int)
      : _coop(coop), _intraOnly(false) {
    new (_implStorage) ncclBarrierSession<ncclCoopCta>(
        ncclCoopCta(), ncclTeamTagRail(), net._gin, index);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
         flagcxGinFenceLevel fence = flagcxGinFenceLevel::Relaxed) {
    if (_intraOnly) {
      reinterpret_cast<ncclLsaBarrierSession<ncclCoopCta> *>(_implStorage)
          ->arrive(ncclCoopCta(), Atomic::toNativeOrder(order));
    }
    // ncclBarrierSession (World/Inter) only exposes sync; arrive is a no-op
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxGinFenceLevel fence = flagcxGinFenceLevel::Relaxed) {
    if (_intraOnly) {
      reinterpret_cast<ncclLsaBarrierSession<ncclCoopCta> *>(_implStorage)
          ->wait(ncclCoopCta(), Atomic::toNativeOrder(order));
    }
    // ncclBarrierSession (World/Inter) only exposes sync; wait is a no-op
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxGinFenceLevel fence = flagcxGinFenceLevel::Relaxed) {
    if (_intraOnly) {
      reinterpret_cast<ncclLsaBarrierSession<ncclCoopCta> *>(_implStorage)
          ->sync(ncclCoopCta(), Atomic::toNativeOrder(order));
    } else {
      reinterpret_cast<ncclBarrierSession<ncclCoopCta> *>(_implStorage)
          ->sync(ncclCoopCta(), Atomic::toNativeOrder(order),
                 flagcxGinFenceLevelMap[static_cast<int>(fence)]);
    }
  }
};

#define FLAGCX_DEVICE_API_VENDOR 1
using DeviceAPI = CommTraits<NvidiaVendor>;

#else
// ============================================================
// NVIDIA Fallback Backend (IPC barriers + FIFO one-sided)
// Uses common Fallback<> partial specialization with NVIDIA platform
// ============================================================
#include "fallback_comm_traits.h"
using DeviceAPI = CommTraits<Fallback<NvidiaPlatform>>;

#endif // NCCL version check

#endif // FLAGCX_NVIDIA_DEVICE_TRAITS_H_
