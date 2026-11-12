#ifndef FLAGCX_ECCL_COMM_TRAITS_H_
#define FLAGCX_ECCL_COMM_TRAITS_H_

#include <stdio.h>

#include "eccl.h"

#include "flagcx.h"

#if FLAGCX_CHECK_DEVICE_CC
// Must precede the ECCL device headers: <tops/tops_runtime.h> supplies
// vector_types.h, which the ECCL device code uses for ulong2/ulonglong2. ECCL
// itself takes these as transitively available, which holds under nvcc but not
// under topscc.
#if defined(__TOPSCC__)
#include <tops/tops_runtime.h>
#endif

#include "eccl_device.h"
// eccl_device.h keeps the GIN implementation headers commented out, but the
// GIN device layer does compile under topscc now and the Net backend below
// forwards to ecclGin, so pull in gin.h (session declarations) and
// impl/gin__funcs.h (definitions) explicitly.
// gin__funcs.h also provides the device-pass placement new/delete that the
// LSA barrier sessions below are constructed with; device_utils.h
// deliberately does not define them on Enflame -- the two definitions would
// collide in the same TU.
#include "eccl_device/gin.h"
#include "eccl_device/impl/gin__funcs.h"
#else
#include "enflame_adaptor.h"
#include "sym_heap.h"
#endif

// ============================================================
// Enflame Vendor Backend (ECCL device API)
// ============================================================
struct EcclBackend {};

template <> struct CommTraits<EcclBackend> {
  // Platform capabilities (via using, not inheritance)
  using Intrin = PlatformTraits<EnflamePlatform>::Intrin;
  using Atomic = PlatformTraits<EnflamePlatform>::Atomic;
  using CoopOps = PlatformTraits<EnflamePlatform>::CoopOps;
  using Clock = PlatformTraits<EnflamePlatform>::Clock;

  // ---- Team: wraps ecclTeam_t ----
  // Exposes nRanks/rank/stride for direct field access (used by flagcxTeam)
  struct Team {
    int nRanks, rank, stride;

    FLAGCX_HOST_DEVICE_INLINE Team() : nRanks(0), rank(0), stride(0) {}
    FLAGCX_HOST_DEVICE_INLINE Team(int nr, int r, int s)
        : nRanks(nr), rank(r), stride(s) {}

    // Implicit conversion to ecclTeam_t for ECCL API calls
    FLAGCX_HOST_DEVICE_INLINE operator ecclTeam_t() const {
      ecclTeam_t t;
      t.nRanks = nRanks;
      t.rank = rank;
      t.stride = stride;
      return t;
    }
  };

  // ---- Multimem: wraps ecclMultimemHandle_t ----
  struct Multimem {
    ecclMultimemHandle_t _impl;

    FLAGCX_HOST_DEVICE_INLINE Multimem() : _impl() {}

    // Implicit conversion for ECCL API calls
    FLAGCX_HOST_DEVICE_INLINE operator ecclMultimemHandle_t() const {
      return _impl;
    }
  };

  // ---- Window: wraps ecclWindow_t ----
  struct Window {
    ecclWindow_t _impl;

    FLAGCX_HOST_DEVICE_INLINE Window() : _impl(nullptr) {}

#if FLAGCX_CHECK_DEVICE_CC
    FLAGCX_DEVICE_INLINE_DECORATOR void *
    getPeerPointer(size_t offset, const Team &team, int peer) const {
      return ecclGetPeerPointer(_impl, offset, (ecclTeam_t)team, peer);
    }

    // ECCL window mappings carry no per-access permissions, so all declared
    // access contracts resolve to the same mapping.
    FLAGCX_DEVICE_INLINE_DECORATOR void *
    getPeerPointer(size_t offset, const Team &team, int peer,
                   flagcxDevPeerAccess_t access) const {
      switch (access) {
        case flagcxDevPeerAccessReadWrite:
        case flagcxDevPeerAccessWriteOnly:
        case flagcxDevPeerAccessReadOnly:
          return getPeerPointer(offset, team, peer);
        default:
          return nullptr;
      }
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void *getLocalPointer(size_t offset) const {
      return ecclGetLocalPointer(_impl, offset);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void *getIntraPointer(size_t offset,
                                                         int peer) const {
      return ecclGetLsaPointer(_impl, offset, peer);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void *
    getIntraPointer(size_t offset, int peer,
                    flagcxDevPeerAccess_t access) const {
      switch (access) {
        case flagcxDevPeerAccessReadWrite:
        case flagcxDevPeerAccessWriteOnly:
        case flagcxDevPeerAccessReadOnly:
          return getIntraPointer(offset, peer);
        default:
          return nullptr;
      }
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void *
    getMulticastPointer(size_t offset, const Multimem &mm) const {
      return ecclGetMultimemPointer(_impl, offset, mm._impl);
    }
#endif // FLAGCX_CHECK_DEVICE_CC

    FLAGCX_HOST_DEVICE_INLINE bool hasAccess() const {
      return _impl != nullptr;
    }
    FLAGCX_HOST_DEVICE_INLINE void *getRawPtr() const {
      return _impl ? _impl->winHost : nullptr;
    }
    FLAGCX_HOST_DEVICE_INLINE void **getDevPeerPtrs() const { return nullptr; }
    FLAGCX_HOST_DEVICE_INLINE int getMrIndex() const { return -1; }

    FLAGCX_HOST_DEVICE_INLINE bool operator==(const Window &o) const {
      return _impl == o._impl;
    }
    FLAGCX_HOST_DEVICE_INLINE bool operator!=(const Window &o) const {
      return !(*this == o);
    }

#if !ECCL_CHECK_TOPSCC
    // Host-side population from flagcxWindow_t (vendor path). The ECCL window
    // struct is a singleton handle per registration, so there is no
    // isSymmetricDefault bookkeeping as on NCCL.
    void populateFromHost(flagcxWindow_t win, void * /*rawPtr*/,
                          int /*intraRank*/, int /*mrIndex*/,
                          uintptr_t /*mrBase*/, int /*ipcIndex*/,
                          void ** /*ipcDevPeerPtrs*/) {
      if (win && win->vendorBase) {
        _impl = win->vendorBase->base;
      }
    }
#endif // !ECCL_CHECK_TOPSCC
  };

  // ---- Comm: wraps ecclDevComm ----
  struct Comm {
    ecclDevComm_t _impl;

    FLAGCX_HOST_DEVICE_INLINE Comm() : _impl() {}

    // Implicit conversion to ecclDevComm for ECCL API calls
    FLAGCX_HOST_DEVICE_INLINE operator const ecclDevComm &() const {
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
    FLAGCX_DEVICE_INLINE_DECORATOR void *
    getFifoBuffer(int /*contextId*/) const {
      return nullptr;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR Multimem getMulticastHandle() const {
      Multimem mm;
      mm._impl = _impl.lsaMultimem;
      return mm;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR bool p2pSignalSupport(int /*peer*/) const {
      return false;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t *
    getSignalPeerPtr(int /*peer*/) const {
      return nullptr;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR bool usesDirectP2pSignals() const {
      return false;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR bool isOneSidedTransportReady() const {
      return true;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR bool supportsDirectCounterAccess() const {
      return false;
    }

    // No-op: the vendor Comm is populated through the devComm pointer cast in
    // flagcx_device_core.h, mirroring the NCCL backend.
    template <typename DI>
    static FLAGCX_HOST_DEVICE_INLINE void populateFromInternal(Comm &,
                                                               const DI &) {}
  };

#if FLAGCX_CHECK_DEVICE_CC
  struct CoopBlock {
    ecclCoopCta _impl;

    FLAGCX_HOST_DEVICE_INLINE CoopBlock() : _impl() {}

    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return _impl.thread_rank();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return _impl.size(); }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _impl.sync(); }
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

    FLAGCX_DEVICE_INLINE_DECORATOR int thread_rank() const {
      return threadRank();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int num_threads() const { return size(); }
  };

  // Single-thread coop is native to the GCU thread model (1 SIP = 1 thread,
  // no warp), so CoopTile<1> gets a real implementation backed by
  // ecclCoopThread. Wider tiles stay unimplemented: ECCL's ecclCoopTile and
  // ecclCoopLanes are CUDA warp-model stubs with sync() deleted on TOPS.
  template <>
  struct CoopTile<1> {
    ecclCoopThread _impl;

    FLAGCX_HOST_DEVICE_INLINE CoopTile() : _impl() {}

    FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
      return _impl.thread_rank();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return _impl.size(); }
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxLaneMask_t laneMask() const {
      return 1;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _impl.sync(); }

    FLAGCX_DEVICE_INLINE_DECORATOR int thread_rank() const {
      return threadRank();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int num_threads() const { return size(); }
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

    FLAGCX_DEVICE_INLINE_DECORATOR int thread_rank() const {
      return threadRank();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int num_threads() const { return size(); }
  };

  struct CoopLanes {
    flagcxLaneMask_t lmask;

    FLAGCX_DEVICE_INLINE_DECORATOR
    CoopLanes(flagcxLaneMask_t lmask) : lmask(lmask) {
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

    FLAGCX_DEVICE_INLINE_DECORATOR int thread_rank() const {
      return threadRank();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int num_threads() const { return size(); }
  };

  struct CoopAny {
    ecclCoopAny _impl;

    CoopAny() = default;
    CoopAny(CoopAny const &) = default;

    FLAGCX_DEVICE_INLINE_DECORATOR CoopAny(CoopBlock b) : _impl(b._impl) {}
    template <int N>
    FLAGCX_DEVICE_INLINE_DECORATOR CoopAny(CoopTile<N> t) : _impl(t) {
      __builtin_trap();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR CoopAny(CoopTileSpan s) : _impl(s) {
      __builtin_trap();
    }
    FLAGCX_DEVICE_INLINE_DECORATOR CoopAny(CoopLanes l) : _impl(l) {
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
  };
#endif // FLAGCX_CHECK_DEVICE_CC

  // ---- Barrier handles ----
  struct IntraBarrierHandle {
    ecclLsaBarrierHandle_t _impl;
  };
  struct InterBarrierHandle {
    ecclGinBarrierHandle_t _impl;
  };

#if FLAGCX_CHECK_DEVICE_CC
  // ---- Barrier alias: delegates to standalone Barrier<Backend, Tag> ----
  template <typename Tag, typename Coop>
  using Barrier = ::Barrier<EcclBackend, Tag, Coop>;

  // ---- Action type conversion helpers (flagcx -> ECCL) ----
  // Mirrors the toNccl block in nccl_comm_traits.h. The order/scope overloads
  // target ECCL's global ::memory_order / ::thread_scope enums
  // (eccl_device/utility.h) -- the types the ecclGin entry points take;
  // FlagCX's own tops:: shim in enflame_platform_traits.h is not involved
  // here.
  FLAGCX_DEVICE_INLINE_DECORATOR static ecclGin_None
  toEccl(flagcxDevNet_None) {
    return {};
  }
  // GCU400 does not support cross-NIC RDMA atomics (AMO_SIGNAL_ADD).
  // Map SignalInc to SignalSet (write-value) for hardware compatibility.
  // The signal slot will be set to 1 instead of atomically incremented.
  // Callers must use per-sender slots and wait on each slot individually.
  FLAGCX_DEVICE_INLINE_DECORATOR static ecclGin_SignalSet
  toEccl(flagcxDevNet_SignalInc a) {
    return {a.signal, 1};  // write value 1 to the slot
  }
  FLAGCX_DEVICE_INLINE_DECORATOR static ecclGin_SignalAdd
  toEccl(flagcxDevNet_SignalAdd a) {
    return {a.signal, a.value};
  }
  FLAGCX_DEVICE_INLINE_DECORATOR static ecclGin_CounterInc
  toEccl(flagcxDevNet_CounterInc a) {
    return {a.counter};
  }
  FLAGCX_DEVICE_INLINE_DECORATOR static ecclGin_DescriptorSmem
  toEccl(flagcxDevNet_DescriptorSmem a) {
    return {(ecclGinDescriptorSmem *)a.smem._impl};
  }
  FLAGCX_DEVICE_INLINE_DECORATOR static memory_order
  toEccl(flagcxDeviceMemoryOrder_t o) {
    switch (o) {
      case flagcxDeviceMemoryOrderRelaxed:
        return memory_order_relaxed;
      case flagcxDeviceMemoryOrderAcquire:
        return memory_order_acquire;
      case flagcxDeviceMemoryOrderRelease:
        return memory_order_release;
      case flagcxDeviceMemoryOrderAcqRel:
        return memory_order_acq_rel;
      case flagcxDeviceMemoryOrderSeqCst:
        return memory_order_seq_cst;
    }
    return memory_order_seq_cst;
  }
  FLAGCX_DEVICE_INLINE_DECORATOR static thread_scope
  toEccl(flagcxDeviceScope_t s) {
    switch (s) {
      case flagcxDeviceScopeSystem:
        return thread_scope_system;
      case flagcxDeviceScopeDevice:
        return thread_scope_device;
      case flagcxDeviceScopeBlock:
        return thread_scope_block;
      case flagcxDeviceScopeThread:
        return thread_scope_thread;
    }
    return thread_scope_system;
  }

  // ---- Net: wraps ecclGin for one-sided operations ----
  // Mirrors the NCCL backend: ecclGin
  // (ecclGin_BackendMask<ECCL_GIN_BACKEND_MASK_ALL>) is to ECCL what ncclGin
  // is to NCCL. Constructing the session loads the GIN context for
  // (comm, contextIndex); every call then dispatches through ecclGinCall to
  // the PROXY or GDAKI(mori) backend according to the connection's net device
  // type. The devComm must have been created with GIN resources
  // (ginSignalCount / ginCounterCount / ginContextCount / ... in
  // ecclDevCommRequirements_t), otherwise the handles loaded here are null.
  struct Net {
    Comm _dc;
    ecclGin _gin;
    int _contextId;

    FLAGCX_DEVICE_INLINE_DECORATOR
    Net(const Comm &dc, int contextIndex)
        : _dc(dc), _gin(dc._impl, contextIndex), _contextId(contextIndex) {}

    FLAGCX_DEVICE_INLINE_DECORATOR bool isIntraPeer(int peer) const {
      int intraBase = _dc.getRank() - _dc.getIntraRank();
      return peer >= intraBase && peer < intraBase + _dc.getIntraSize();
    }

    FLAGCX_DEVICE_INLINE_DECORATOR bool isValid() const { return true; }

    FLAGCX_DEVICE_INLINE_DECORATOR int getContextId() const {
      return _contextId;
    }

    // No direct signal/counter pointers: ECCL GIN signals and counters are
    // addressed by id through the context, not exposed as raw pointers. Same
    // shape as the NCCL backend.
    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t *
    getSignalPtr(flagcxDevSignal_t /*signal*/) const {
      return nullptr;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t *
    getPeerSignalPtr(int /*localPeer*/, flagcxDevSignal_t /*signal*/) const {
      return nullptr;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t *
    getCounterPtr(flagcxDevCounter_t /*counter*/) const {
      return nullptr;
    }

    // --- One-sided: put (raw Window) ---
    template <typename RA, typename LA, typename Coop, typename Desc>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    put(Team team, int peer, Window dst, size_t dstOff, Window src,
        size_t srcOff, size_t bytes, RA ra, LA la, Coop coop, Desc desc,
        flagcxDeviceScope_t ar, flagcxDeviceScope_t es) const {
      _gin.put((ecclTeam_t)team, peer, dst._impl, dstOff, src._impl, srcOff,
               bytes, toEccl(ra), toEccl(la), coop._impl, toEccl(desc),
               toEccl(ar), toEccl(es));
    }

    // --- One-sided: putValue ---
    template <typename T, typename RA, typename Coop, typename Desc>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    putValue(Team team, int peer, Window dst, size_t dstOff, T value, RA ra,
             Coop coop, Desc desc, flagcxDeviceScope_t ar,
             flagcxDeviceScope_t es) const {
      _gin.putValue((ecclTeam_t)team, peer, dst._impl, dstOff, value,
                    toEccl(ra), coop._impl, toEccl(desc), toEccl(ar),
                    toEccl(es));
    }

    // --- One-sided: signal ---
    template <typename RA, typename Coop, typename Desc>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    signal(Team team, int peer, RA ra, Coop coop, Desc desc,
           flagcxDeviceScope_t ar, flagcxDeviceScope_t es) const {
      _gin.signal((ecclTeam_t)team, peer, toEccl(ra), coop._impl, toEccl(desc),
                  toEccl(ar), toEccl(es));
    }

    // --- One-sided: flush ---
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    flush(Coop coop, flagcxDeviceMemoryOrder_t order) const {
      _gin.flush(coop._impl, toEccl(order));
    }

    // --- One-sided: waitSignal ---
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitSignal(Coop coop, flagcxDevSignal_t signal, uint64_t least, int bits,
               flagcxDeviceMemoryOrder_t order) const {
      _gin.waitSignal(coop._impl, signal, least, bits, toEccl(order));
    }

    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitSignalMeetShadow(Coop coop, flagcxDevSignal_t signal, int bits,
                         flagcxDeviceMemoryOrder_t order) const {
      _gin.waitSignalMeetShadow(coop._impl, signal, bits, toEccl(order));
    }

    template <typename Coop, typename Uint>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitSignalFollowShadow(Coop coop, flagcxDevSignal_t signal, Uint leastDelta,
                           Uint *before, Uint *delta, int bits,
                           flagcxDeviceMemoryOrder_t order) const {
      _gin.waitSignalFollowShadow(coop._impl, signal, leastDelta, before, delta,
                                  bits, toEccl(order));
    }

    // --- Shadow manipulation ---
    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t *
    getSignalShadowPtr(flagcxDevSignal_t signal) const {
      return _gin.getSignalShadowPtr(signal);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    increaseSignalShadow(flagcxDevSignal_t signal, uint64_t delta) const {
      _gin.increaseSignalShadow(signal, delta);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
    readSignal(flagcxDevSignal_t signal, int bits,
               flagcxDeviceMemoryOrder_t order) const {
      return _gin.readSignal(signal, bits, toEccl(order));
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    resetSignal(flagcxDevSignal_t signal) const {
      _gin.resetSignal(signal);
    }

    // --- Counter ---
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitCounter(Coop coop, flagcxDevCounter_t counter, uint64_t least, int bits,
                flagcxDeviceMemoryOrder_t order) const {
      _gin.waitCounter(coop._impl, counter, least, bits, toEccl(order));
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
    readCounter(flagcxDevCounter_t counter, int bits,
                flagcxDeviceMemoryOrder_t order) const {
      return _gin.readCounter(counter, bits, toEccl(order));
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    resetCounter(flagcxDevCounter_t counter) const {
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

    // --- One-sided: get (unsupported by the ECCL backend) ---
    // ECCL GIN only pushes (put / putValue / signal); there is no one-sided
    // read. Failing loudly here is deliberate: silently returning stale data
    // would be far harder to diagnose than a missing feature.
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    get(Team team, int peer, Window src, size_t srcOff, Window dst,
        size_t dstOff, size_t bytes, Coop coop) const {
      if (coop.threadRank() == 0) {
        printf("FLAGCX WARN: ECCL Device Get is unsupported; operation is a "
               "no-op\n");
      }
      (void)team;
      (void)peer;
      (void)src;
      (void)srcOff;
      (void)dst;
      (void)dstOff;
      (void)bytes;
    }
  };
#endif // FLAGCX_CHECK_DEVICE_CC
};

// No ecclGinFenceLevel translation table here yet: the fence argument is
// unused while the inter/world Barrier specializations below remain no-ops.
// Reinstate one when the ecclGinBarrierSession wiring lands.

// ============================================================
// Barrier specializations for EcclBackend
// ============================================================
#if FLAGCX_CHECK_DEVICE_CC

// ---- Barrier<EcclBackend, flagcxTeamTagIntra, Coop> ----
// Wraps ecclLsaBarrierSession<ecclCoopCta> via placement new. The session reads
// the epoch out of the resource buffer on construction and writes it back in
// its destructor, so the object must not be copied or left dangling.
template <typename Coop>
struct Barrier<EcclBackend, flagcxTeamTagIntra, Coop> {
  using Comm = CommTraits<EcclBackend>::Comm;
  using Team = CommTraits<EcclBackend>::Team;
  using Multimem = CommTraits<EcclBackend>::Multimem;

  Coop _coop;
  alignas(ecclLsaBarrierSession<ecclCoopCta>) char _implStorage[sizeof(
      ecclLsaBarrierSession<ecclCoopCta>)];
  bool _active;

  // Default ctor — inactive
  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier() : _coop(), _active(false) {}

  // Active ctor
  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, const Comm &dc, Team team, uint32_t index,
          bool multimem = false, const Multimem &mm = {})
      : _coop(coop), _active(true) {
    new (_implStorage) ecclLsaBarrierSession<ecclCoopCta>(
        ecclCoopCta(), dc, (ecclTeam_t)team, dc._impl.lsaBarrier, index);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR
  ~Barrier() {
    if (_active)
      reinterpret_cast<ecclLsaBarrierSession<ecclCoopCta> *>(_implStorage)
          ->~ecclLsaBarrierSession();
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    reinterpret_cast<ecclLsaBarrierSession<ecclCoopCta> *>(_implStorage)
        ->arrive(ecclCoopCta(), ecclLsaBarrierFenceRelease);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    reinterpret_cast<ecclLsaBarrierSession<ecclCoopCta> *>(_implStorage)
        ->wait(ecclCoopCta());
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    reinterpret_cast<ecclLsaBarrierSession<ecclCoopCta> *>(_implStorage)
        ->sync(ecclCoopCta(), ecclLsaBarrierFenceRelease);
  }
};

// ---- Barrier<EcclBackend, flagcxTeamTagInter, Coop> ----
// ECCL's inter-node barrier is ecclGinBarrierSession, whose sync() drives
// net.signal()/net.waitSignal() (gin_barrier__funcs.h). GIN itself compiles
// under topscc now and Net forwards to it, but the barrier session is not
// wired up yet: the type is kept so the IR layer still instantiates, and
// every operation does nothing -- an Enflame run cannot yet synchronize
// across nodes from the device side.
template <typename Coop>
struct Barrier<EcclBackend, flagcxTeamTagInter, Coop> {
  using Comm = CommTraits<EcclBackend>::Comm;
  using Team = CommTraits<EcclBackend>::Team;
  using Net = CommTraits<EcclBackend>::Net;

  Coop _coop;
  int _nInterPeers;

  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier() : _coop(), _nInterPeers(0) {}

  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, const Net &net, const Comm &dc, Team team, uint32_t index,
          int nInterPeers)
      : _coop(coop), _nInterPeers(nInterPeers) {
    (void)net;
    (void)dc;
    (void)team;
    (void)index;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR
  ~Barrier() {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
         flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    (void)order;
    (void)fence;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    (void)order;
    (void)fence;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    (void)order;
    (void)fence;
  }
};

// ---- Barrier<EcclBackend, flagcxTeamTagWorld, Coop> ----
// Only the intra half exists. ECCL's ecclBarrierSession (World / Rail) is built
// on ecclGinBarrierSession, which is not wired up yet, so world and inter
// barriers do nothing. An intra barrier still uses ecclLsaBarrierSession, which
// is LSA-only and sound. The LSA session reads the epoch out of the resource
// buffer on construction and writes it back in its destructor, so the object
// must not be copied or left dangling.
template <typename Coop>
struct Barrier<EcclBackend, flagcxTeamTagWorld, Coop> {
  using Comm = CommTraits<EcclBackend>::Comm;
  using Net = CommTraits<EcclBackend>::Net;

  Coop _coop;
  alignas(ecclLsaBarrierSession<ecclCoopCta>) char _implStorage[sizeof(
      ecclLsaBarrierSession<ecclCoopCta>)];
  bool _intraOnly;

  // World barrier (intra + inter) -- the inter half is unavailable, so this is
  // inactive. The ctor signature is shared with the other backends.
  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, flagcxTeamTagWorld, const Net &net, const Comm &dc,
          uint32_t index, bool multimem, int)
      : _coop(coop), _intraOnly(false) {
    (void)net;
    (void)dc;
    (void)index;
    (void)multimem;
  }

  // Intra-only barrier
  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, flagcxTeamTagIntra, const Net &, const Comm &dc,
          uint32_t index, bool multimem, int)
      : _coop(coop), _intraOnly(true) {
    (void)multimem;
    new (_implStorage) ecclLsaBarrierSession<ecclCoopCta>(
        ecclCoopCta(), dc, ecclTeamTagLsa(), index);
  }

  // Inter-only barrier (ecclTeamTagRail) -- the session is not wired up yet.
  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, flagcxTeamTagInter, const Net &net, const Comm &,
          uint32_t index, bool, int)
      : _coop(coop), _intraOnly(false) {
    (void)net;
    (void)index;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR
  ~Barrier() {
    if (_intraOnly)
      reinterpret_cast<ecclLsaBarrierSession<ecclCoopCta> *>(_implStorage)
          ->~ecclLsaBarrierSession();
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
         flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    (void)fence;
    if (_intraOnly) {
      reinterpret_cast<ecclLsaBarrierSession<ecclCoopCta> *>(_implStorage)
          ->arrive(ecclCoopCta(), ecclLsaBarrierFenceRelease);
    }
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    (void)fence;
    if (_intraOnly) {
      reinterpret_cast<ecclLsaBarrierSession<ecclCoopCta> *>(_implStorage)
          ->wait(ecclCoopCta());
    }
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    (void)fence;
    if (_intraOnly) {
      reinterpret_cast<ecclLsaBarrierSession<ecclCoopCta> *>(_implStorage)
          ->sync(ecclCoopCta(), ecclLsaBarrierFenceRelease);
    }
  }
};
#endif // FLAGCX_CHECK_DEVICE_CC

#define FLAGCX_DEVICE_API_VENDOR 1
using DeviceAPI = CommTraits<EcclBackend>;

#endif // FLAGCX_ECCL_COMM_TRAITS_H_
