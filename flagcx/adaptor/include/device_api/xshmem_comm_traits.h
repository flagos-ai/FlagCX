#ifndef FLAGCX_XSHMEM_COMM_TRAITS_H_
#define FLAGCX_XSHMEM_COMM_TRAITS_H_

#include "flagcx_kernel_core.h"
#include "xshmem_state_layout.h"
#include <cstddef>
#include <cstdint>

#include "xshmem/xshmem.h"
#include "xshmem/xshmemx.h"

#if defined(__xpu__)
#include "xpu/kernel/xtdk.h"
// assert_device (xpu3) lives in xtdk_io.h, which xtdk.h does not pull in on
// the __xpu__ path.
#include "xpu/kernel/xtdk_io.h"
#define XSHMEM_FGP __global_ptr__
#else
#define XSHMEM_FGP
// Some xccl releases keep device comparison/signal constants out of their
// host-facing header. Their values are irrelevant to the no-op host shims.
#ifndef XSHMEM_CMP_GE
#define XSHMEM_CMP_GE 0
#endif
#ifndef XSHMEM_SIGNAL_ADD
#define XSHMEM_SIGNAL_ADD 0
#endif
#ifndef XSHMEM_SIGNAL_SET
#define XSHMEM_SIGNAL_SET 0
#endif
#endif

#define XSHMEM_DEVICE_INLINE FLAGCX_DEVICE_INLINE_DECORATOR
#define XSHMEM_HOST_DEVICE_INLINE FLAGCX_HOST_DEVICE_INLINE

// xccl exposes several XSHMEM operations only to the XPU device pass. Keep
// those names out of host template parsing, just as nvshmem_comm_traits.h does
// for NVSHMEM's device-only entry points. Host versions are compile-time shims;
// Device API methods are never executed by host code.
namespace flagcxXshmemDevice {
#if defined(__xpu__)
XSHMEM_DEVICE_INLINE float *localScratch() {
  return (float *)get_xshmemi_local_buf();
}
XSHMEM_DEVICE_INLINE int localScratchBytes() { return XSHMEMI_LOCAL_BUF_LEN; }
XSHMEM_DEVICE_INLINE int myPe() { return xshmem_my_pe(); }
// xccl exposes no xshmem_ptr, but its own put/signal paths translate a
// symmetric address the same way nvshmem_ptr does, so do it here:
//   remote = peer_heap_base_remote_p2p[pe] + (ptr - heap_base)
// The result is WRITE-ONLY. kl3 has no C2C read (xshmemi_get* traps), so the
// returned pointer must never be dereferenced for a load.
XSHMEM_DEVICE_INLINE XSHMEM_FGP void *peerWritePtr(XSHMEM_FGP void *ptr,
                                                   int pe) {
  // No null check on the state: it is a __shared_ptr__ into the kernel's own
  // shared allocation, and XSHMEM_DEVICE_INIT usually places it at shared
  // offset 0 -- i.e. a valid state compares equal to nullptr. xshmem's own
  // device code never tests it either; XSHMEM_DEVICE_INIT is the contract.
  __shared_ptr__ xshmemi_device_host_state_t *state =
      get_xshmemi_device_state();
  if (ptr == nullptr)
    return nullptr;
  XSHMEM_FGP char *heapBase = (XSHMEM_FGP char *)state->heap_base;
  if (pe == state->my_pe)
    return ptr;
  XSHMEM_FGP char *peerBase = nullptr;
  GM2LM((XSHMEM_FGP char *)state->peer_heap_base_remote_p2p + pe * 8, &peerBase,
        sizeof(peerBase));
  if (peerBase == nullptr)
    return nullptr;
  return (XSHMEM_FGP void *)(peerBase + ((XSHMEM_FGP char *)ptr - heapBase));
}
XSHMEM_DEVICE_INLINE XSHMEM_FGP void *peerPtr(XSHMEM_FGP void *ptr, int pe) {
  // Reserved for callers that intend to read through the pointer. kl3 has no
  // C2C read, so fail loudly at the offending call site instead of returning
  // nullptr and crashing later at a confusing dereference.
  assert_device(false && "CommTraits<XshmemBackend>::peerPtr: P800 C2C has no "
                         "remote read; use peerWritePtr / the put-based path");
  (void)ptr;
  (void)pe;
  return nullptr;
}
XSHMEM_DEVICE_INLINE void threadFence() { mfence(); }
// xshmemi_quiet<SCOPE> brackets its fence with xshmemi_threadgroup_sync<SCOPE>,
// so the scope must match the set of cores that actually reach the call. The
// CLUSTER form is a sync_cluster(): calling it from a single core while the
// other cores of the cluster wait on a different barrier deadlocks. Keep both
// forms and let each call site pick by its cooperative scope.
XSHMEM_DEVICE_INLINE void quietCluster(int pe) {
  xshmemi_quiet<XSHMEMI_THREADGROUP_CLUSTER>(pe);
}
XSHMEM_DEVICE_INLINE void quietCore(int pe) {
  xshmemi_quiet<XSHMEMI_THREADGROUP_CORE>(pe);
}
XSHMEM_DEVICE_INLINE void quiet(int pe) { quietCluster(pe); }
XSHMEM_DEVICE_INLINE void signalOp(XSHMEM_FGP uint64_t *addr, uint64_t value,
                                   int op, int pe) {
  xshmemx_signal_op(addr, value, op, pe);
}
XSHMEM_DEVICE_INLINE void waitUntil(XSHMEM_FGP uint64_t *addr, int cmp,
                                    uint64_t value) {
  xshmem_signal_wait_until(addr, cmp, value);
}
XSHMEM_DEVICE_INLINE void fence() { xshmem_fence(); }
XSHMEM_DEVICE_INLINE void barrierAll() { xshmem_barrier_all(); }
XSHMEM_DEVICE_INLINE void putFloatCluster(XSHMEM_FGP float *dst,
                                          XSHMEM_FGP float *src, size_t count,
                                          int pe) {
  xshmemx_float_put_nbi_cluster(dst, src, count, pe);
}
#else
XSHMEM_DEVICE_INLINE float *localScratch() { return nullptr; }
XSHMEM_DEVICE_INLINE int localScratchBytes() { return 0; }
XSHMEM_DEVICE_INLINE int myPe() { return 0; }
XSHMEM_DEVICE_INLINE void *peerWritePtr(void *, int) { return nullptr; }
XSHMEM_DEVICE_INLINE void *peerPtr(void *, int) { return nullptr; }
XSHMEM_DEVICE_INLINE void threadFence() {}
XSHMEM_DEVICE_INLINE void quietCluster(int) {}
XSHMEM_DEVICE_INLINE void quietCore(int) {}
XSHMEM_DEVICE_INLINE void quiet(int) {}
XSHMEM_DEVICE_INLINE void signalOp(uint64_t *, uint64_t, int, int) {}
XSHMEM_DEVICE_INLINE void waitUntil(uint64_t *, int, uint64_t) {}
XSHMEM_DEVICE_INLINE void fence() {}
XSHMEM_DEVICE_INLINE void barrierAll() {}
XSHMEM_DEVICE_INLINE void putFloatCluster(float *, float *, size_t, int) {}
#endif

// Drain outstanding operations to `pe` at whatever granularity the caller's
// cooperative group actually has. A cluster-wide group can use the collective
// form; anything narrower must not, or the cores that never entered the call
// will never reach the sync_cluster() inside it.
template <typename Coop>
XSHMEM_DEVICE_INLINE void quietForCoop(Coop coop, int pe) {
  if (coop.size() >= FLAGCX_BLOCK_DIM_X)
    quietCluster(pe);
  else if (coop.threadRank() == 0)
    quietCore(pe);
}
} // namespace flagcxXshmemDevice

// A signal/counter word in the symmetric heap. P800 has no atomic on global
// memory and none is emulated here: every word below is single-writer by
// construction (see xshmem_state_layout.h), so a plain volatile access is both
// correct and the only thing the hardware offers.
typedef volatile XSHMEM_FGP uint64_t XshmemGMWord;

struct XshmemBackend {};

template <>
struct CommTraits<XshmemBackend> {
  using Intrin = PlatformTraits<KunlunxinPlatform>::Intrin;
  using Atomic = PlatformTraits<KunlunxinPlatform>::Atomic;

  // ---- Multimem ----
  struct Multimem {
    void *mcBasePtr;
  };

  // ---- Team ----
  // XSHMEM has no team objects; PE ids equal global ranks.
  struct Team {
    int nRanks, rank, stride;
  };

  // ---- Local scratch (per-core LM staging buffer) ----
  static XSHMEM_DEVICE_INLINE float *localScratch() {
    return flagcxXshmemDevice::localScratch();
  }
  static XSHMEM_DEVICE_INLINE int localScratchBytes() {
    return flagcxXshmemDevice::localScratchBytes();
  }

  // ---- Window ----
  struct Window {
    XSHMEM_FGP void *symBase;
    uint64_t allocSize;
    XSHMEM_FGP void *rawPtr;
    XSHMEM_FGP int *intraPeMap;
    int intraRank;
    int intraSize;

    XSHMEM_DEVICE_INLINE int resolveWorldPeer(const Team &team, int peer) const {
      if (intraPeMap != nullptr && team.nRanks == intraSize &&
          team.rank == intraRank && team.stride == 1)
        return intraPeMap[peer];
      int base = flagcxXshmemDevice::myPe() - team.rank * team.stride;
      return base + peer * team.stride;
    }

    // P800's C2C link cannot read peer memory, so only the write-only form can
    // be translated at all: peerPtr asserts, peerWritePtr does the arithmetic.
    XSHMEM_DEVICE_INLINE XSHMEM_FGP void *
    getPeerPointer(size_t offset, const Team &team, int peer,
                   flagcxDevPeerAccess_t access =
                       flagcxDevPeerAccessReadWrite) const {
      if (access == flagcxDevPeerAccessWriteOnly)
        return flagcxXshmemDevice::peerWritePtr(
            (XSHMEM_FGP char *)symBase + offset,
            this->resolveWorldPeer(team, peer));
      return flagcxXshmemDevice::peerPtr((XSHMEM_FGP char *)symBase + offset,
                                         this->resolveWorldPeer(team, peer));
    }
    XSHMEM_DEVICE_INLINE XSHMEM_FGP void *getLocalPointer(size_t offset) const {
      return (XSHMEM_FGP char *)rawPtr + offset;
    }
    XSHMEM_DEVICE_INLINE XSHMEM_FGP void *
    getIntraPointer(size_t offset, int peer,
                    flagcxDevPeerAccess_t access =
                        flagcxDevPeerAccessReadWrite) const {
      if (access == flagcxDevPeerAccessWriteOnly)
        return flagcxXshmemDevice::peerWritePtr(
            (XSHMEM_FGP char *)symBase + offset, intraPeMap[peer]);
      return flagcxXshmemDevice::peerPtr((XSHMEM_FGP char *)symBase + offset,
                                         intraPeMap[peer]);
    }
    XSHMEM_DEVICE_INLINE XSHMEM_FGP void *
    getMulticastPointer(size_t, const Multimem &) const {
      return nullptr;
    }
    XSHMEM_HOST_DEVICE_INLINE XSHMEM_FGP void *getRawPtr() const {
      return rawPtr;
    }
    XSHMEM_HOST_DEVICE_INLINE bool hasAccess() const {
      return symBase != nullptr;
    }
    XSHMEM_HOST_DEVICE_INLINE void **getDevPeerPtrs() const { return nullptr; }
    XSHMEM_HOST_DEVICE_INLINE int getMrIndex() const { return 0; }
    XSHMEM_DEVICE_INLINE bool operator==(const Window &o) const {
      return symBase == o.symBase;
    }
    XSHMEM_DEVICE_INLINE bool operator!=(const Window &o) const {
      return !(*this == o);
    }
  };

  // ---- Comm ----
  struct Comm {
    int rank, nRanks;
    int intraRank, intraSize;
    XSHMEM_FGP int *intraPeMap;
    xshmem_team_t intraTeam;
    xshmem_team_t interTeam;
    xshmem_team_t worldTeam;

    XSHMEM_FGP uint64_t *signalBuffer;
    int signalCount;
    XSHMEM_FGP uint64_t *counterBuffer;
    int counterCount;
    XSHMEM_FGP uint64_t *shadowBuffer;

    XSHMEM_FGP uint64_t *gridSyncState;

    XSHMEM_FGP void *devStateHandle;

    XSHMEM_FGP uint64_t *scratchBuffer;
    uint64_t scratchBytes;

    XSHMEM_DEVICE_INLINE int getIntraRank() const { return intraRank; }
    XSHMEM_DEVICE_INLINE int getIntraSize() const { return intraSize; }
    XSHMEM_DEVICE_INLINE int getRank() const { return rank; }
    XSHMEM_DEVICE_INLINE int getSize() const { return nRanks; }
    XSHMEM_DEVICE_INLINE void *getFifoBuffer(int) const { return nullptr; }

    // Window over the symmetric scratch, so the regular put path can target it
    // exactly like an application-registered buffer.
    XSHMEM_DEVICE_INLINE Window getScratchWindow() const {
      Window w;
      w.symBase = scratchBuffer;
      w.allocSize = scratchBytes;
      w.rawPtr = scratchBuffer;
      w.intraPeMap = intraPeMap;
      w.intraRank = intraRank;
      w.intraSize = intraSize;
      return w;
    }
    XSHMEM_DEVICE_INLINE Multimem getMulticastHandle() const {
      Multimem mm;
      mm.mcBasePtr = nullptr;
      return mm;
    }

    XSHMEM_DEVICE_INLINE bool p2pSignalSupport(int localPeer) const {
      // this-> : xpu-clang drops the const qualifier on member calls from a
      // const member of a class nested in an explicit specialization.
      return this->getSignalPeerPtr(localPeer) != nullptr;
    }
    XSHMEM_DEVICE_INLINE XSHMEM_FGP uint64_t *
    getSignalPeerPtr(int localPeer) const {
      int worldPeer = intraPeMap[localPeer];
      return (XSHMEM_FGP uint64_t *)flagcxXshmemDevice::peerWritePtr(
          signalBuffer, worldPeer);
    }
    XSHMEM_DEVICE_INLINE bool usesDirectP2pSignals() const { return false; }
    XSHMEM_DEVICE_INLINE bool isOneSidedTransportReady() const { return true; }
    // A "direct" counter update is a read-modify-write on a global-memory word,
    // which P800 cannot do atomically. Counter increments must go through the
    // Net local-action path, where the word has exactly one writer.
    XSHMEM_DEVICE_INLINE bool supportsDirectCounterAccess() const {
      return false;
    }

    template <typename DI>
    static XSHMEM_HOST_DEVICE_INLINE void populateFromInternal(Comm &dc,
                                                               const DI &di) {
      // Only scalar baseline fields are copied here; signal/counter/shadow
      // buffers and the device state handle are filled in by the host launcher
      // (they live in address-space 1 on the xpu3 device pass and cannot be
      // assigned from plain host pointers inside this header). This keeps the
      // function parseable by both the host pass and the xpu3 device pass.
      dc.rank = di.rank;
      dc.nRanks = di.nRanks;
      dc.intraRank = di.intraRank;
      dc.intraSize = di.intraSize;
      dc.intraPeMap = nullptr;
      dc.intraTeam = XSHMEM_TEAM_INVALID;
      dc.interTeam = XSHMEM_TEAM_INVALID;
      dc.worldTeam = XSHMEM_TEAM_WORLD;
      dc.signalBuffer = nullptr;
      dc.signalCount = 0;
      dc.counterBuffer = nullptr;
      dc.counterCount = 0;
      dc.shadowBuffer = nullptr;
      dc.gridSyncState = nullptr;
      dc.devStateHandle = nullptr;
      dc.scratchBuffer = nullptr;
      dc.scratchBytes = 0;
    }
  };

  // ---- Coop types: supplied by the XPU platform layer ----
  using CoopBlock = typename PlatformTraits<KunlunxinPlatform>::CoopBlock;
  template <int N>
  using CoopTile =
      typename PlatformTraits<KunlunxinPlatform>::template CoopTile<N>;
  using CoopThread = typename PlatformTraits<KunlunxinPlatform>::CoopThread;
  using CoopWarp = typename PlatformTraits<KunlunxinPlatform>::CoopWarp;
  using CoopTileSpan = typename PlatformTraits<KunlunxinPlatform>::CoopTileSpan;
  using CoopLanes = typename PlatformTraits<KunlunxinPlatform>::CoopLanes;
  using CoopAny = typename PlatformTraits<KunlunxinPlatform>::CoopAny;

  // ---- Barrier handles ----
  struct IntraBarrierHandle {
    int nBarriers;
  };
  struct InterBarrierHandle {
    int placeholder;
  };

  // ---- DescriptorSmem: empty for XSHMEM ----
  struct DescriptorSmem {};

  // ---- Barrier alias ----
  template <typename Tag, typename Coop>
  using Barrier = ::Barrier<XshmemBackend, Tag, Coop>;

  // ---- Net ----
  struct Net {
    Comm _dc;
    int _contextId;

    XSHMEM_HOST_DEVICE_INLINE
    Net(const Comm &dc, int contextIndex) : _dc(dc), _contextId(contextIndex) {}

    XSHMEM_DEVICE_INLINE bool isValid() const {
      if (_dc.signalCount > 0 && _dc.signalBuffer == nullptr)
        return false;
      if (_dc.counterCount > 0 && _dc.counterBuffer == nullptr)
        return false;
      if (_dc.signalCount > 0 && _dc.shadowBuffer == nullptr)
        return false;
      return _dc.devStateHandle != nullptr;
    }

    XSHMEM_DEVICE_INLINE int getContextId() const { return _contextId; }

    // The "signal pointer" is this PE's own receive slot. Passing it to
    // xshmemx_signal_op makes the peer's slot at the same symmetric offset the
    // write target, which is exactly the per-source slot this PE owns there.
    XSHMEM_DEVICE_INLINE XSHMEM_FGP uint64_t *
    getSignalPtr(flagcxDevSignal_t signalId) const {
      return &_dc.signalBuffer[this->signalBase(signalId) + _dc.rank];
    }

    XSHMEM_DEVICE_INLINE XSHMEM_FGP uint64_t *
    getPeerSignalPtr(int, flagcxDevSignal_t) const {
      return nullptr;
    }

    XSHMEM_DEVICE_INLINE XSHMEM_FGP uint64_t *
    getCounterPtr(flagcxDevCounter_t counterId) const {
      return &_dc.counterBuffer[this->counterBase(counterId) + _dc.rank];
    }

    // ---- Helper: resolve PE from team + peer index ----
    static XSHMEM_DEVICE_INLINE int resolvePE(const Comm &dc, Team team,
                                              int peer) {
      if (dc.intraPeMap != nullptr && team.nRanks == dc.intraSize &&
          team.rank == dc.intraRank && team.stride == 1)
        return dc.intraPeMap[peer];
      int base = dc.rank - team.rank * team.stride;
      return base + peer * team.stride;
    }

    // ---- One-sided: put ----
    template <typename RA, typename LA, typename Coop, typename Desc>
    XSHMEM_DEVICE_INLINE void
    put(Team team, int peer, Window dst, size_t dstOff, Window src,
        size_t srcOff, size_t bytes, RA ra, LA la, Coop coop, Desc desc,
        flagcxDeviceScope_t ar, flagcxDeviceScope_t es) const {
      (void)desc;
      (void)ar;
      (void)es;
      // Two transports are available. XSHMEM's aligned put is a *cluster
      // collective*, so it needs the whole block; the write-only peer mapping
      // is just a store and works at any cooperative granularity. Pick the
      // collective only when it is actually usable.
      bool aligned = ((dstOff | srcOff | bytes) & (sizeof(uint32_t) - 1)) == 0;
      if (!aligned || coop.size() != FLAGCX_BLOCK_DIM_X) {
        XSHMEM_FGP char *remote = (XSHMEM_FGP char *)dst.getPeerPointer(
            dstOff, team, peer, flagcxDevPeerAccessWriteOnly);
        XSHMEM_FGP char *local = (XSHMEM_FGP char *)src.getLocalPointer(srcOff);
        if (remote == nullptr) {
          if (coop.threadRank() == 0)
            __builtin_trap();
          return;
        }
        if (aligned) {
          XSHMEM_FGP uint32_t *rw = (XSHMEM_FGP uint32_t *)remote;
          XSHMEM_FGP uint32_t *lw = (XSHMEM_FGP uint32_t *)local;
          for (size_t i = (size_t)coop.threadRank(); i < bytes / 4;
               i += (size_t)coop.size())
            rw[i] = lw[i];
        } else {
          for (size_t i = (size_t)coop.threadRank(); i < bytes;
               i += (size_t)coop.size())
            remote[i] = local[i];
        }
        flagcxXshmemDevice::threadFence();
        coop.sync();
        if (coop.threadRank() == 0) {
          int pe = resolvePE(_dc, team, peer);
          flagcxXshmemDevice::quietCore(pe);
          this->remoteActionImpl(pe, ra);
          this->localActionImpl(la);
        }
        coop.sync();
        return;
      }

      // The installed aligned put is a cluster collective.
      coop.sync();
      int pe = resolvePE(_dc, team, peer);
      this->putImpl(_dc,
                    (XSHMEM_FGP float *)((XSHMEM_FGP char *)dst.symBase + dstOff),
                    (XSHMEM_FGP float *)((XSHMEM_FGP char *)src.rawPtr + srcOff),
                    bytes, pe, ra, la);
      coop.sync();
    }

    // ---- One-sided: putValue ----
    template <typename T, typename RA, typename Coop, typename Desc>
    XSHMEM_DEVICE_INLINE void
    putValue(Team team, int peer, Window dst, size_t dstOff, T value, RA ra,
             Coop coop, Desc, flagcxDeviceScope_t, flagcxDeviceScope_t) const {
      coop.sync();
      XSHMEM_FGP T *remote = (XSHMEM_FGP T *)dst.getPeerPointer(
          dstOff, team, peer, flagcxDevPeerAccessWriteOnly);
      int pe = resolvePE(_dc, team, peer);
      if (remote != nullptr) {
        if (coop.threadRank() == 0) {
          *remote = value;
          flagcxXshmemDevice::threadFence();
        }
      } else if constexpr (sizeof(T) == sizeof(uint64_t)) {
        if (coop.threadRank() == 0) {
          union {
            T value;
            uint64_t bits;
          } encoded;
          encoded.value = value;
          flagcxXshmemDevice::signalOp(
              (XSHMEM_FGP uint64_t *)((XSHMEM_FGP char *)dst.symBase + dstOff),
              encoded.bits, XSHMEM_SIGNAL_SET, pe);
        }
      } else {
        if (coop.threadRank() == 0)
          __builtin_trap();
        return;
      }
      coop.sync();
      flagcxXshmemDevice::quietForCoop(coop, pe);
      if (coop.threadRank() == 0) {
        this->remoteActionImpl(pe, ra);
      }
      coop.sync();
    }

    // ---- One-sided: signal ----
    template <typename RA, typename Coop, typename Desc>
    XSHMEM_DEVICE_INLINE void signal(Team team, int peer, RA ra, Coop coop,
                                     Desc desc, flagcxDeviceScope_t ar,
                                     flagcxDeviceScope_t es) const {
      (void)desc;
      (void)ar;
      (void)es;
      coop.sync();
      if (coop.threadRank() == 0) {
        int pe = resolvePE(_dc, team, peer);
        this->remoteActionImpl(pe, ra);
      }
      coop.sync();
    }

    // ---- Ordering: flush ----
    template <typename Coop>
    XSHMEM_DEVICE_INLINE void flush(Coop coop,
                                    flagcxDeviceMemoryOrder_t order) const {
      if (order == flagcxDeviceMemoryOrderAcqRel) {
        coop.sync();
        for (int pe = 0; pe < _dc.nRanks; ++pe)
          flagcxXshmemDevice::quietForCoop(coop, pe);
        coop.sync();
      } else {
        flagcxXshmemDevice::fence();
      }
    }

    // ---- Wait: waitSignal ----
    // A signal is the sum of its per-source slots, so xshmem_signal_wait_until
    // (single address) cannot express the condition; spin on the aggregate.
    template <typename Coop>
    XSHMEM_DEVICE_INLINE void
    waitSignal(Coop coop, flagcxDevNetSignal_t signalId, uint64_t least,
               int bits, flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      (void)order;
      coop.sync();
      if (coop.threadRank() == 0)
        this->spinSignal(signalId, least);
      coop.sync();
    }

    // ---- Wait: waitSignalMeetShadow ----
    template <typename Coop>
    XSHMEM_DEVICE_INLINE void
    waitSignalMeetShadow(Coop coop, flagcxDevSignal_t signalId, int bits,
                         flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      (void)order;
      coop.sync();
      if (coop.threadRank() == 0)
        this->spinSignal(signalId,
                         *(XshmemGMWord *)this->getSignalShadowPtr(signalId));
      coop.sync();
    }

    // ---- Wait: waitSignalFollowShadow ----
    template <typename Coop, typename Uint>
    XSHMEM_DEVICE_INLINE void
    waitSignalFollowShadow(Coop coop, flagcxDevSignal_t signalId,
                           Uint leastDelta, Uint *before, Uint *delta, int bits,
                           flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      (void)order;
      coop.sync();
      if (coop.threadRank() == 0) {
        uint64_t shadow = *(XshmemGMWord *)this->getSignalShadowPtr(signalId);
        uint64_t current =
            this->spinSignal(signalId, shadow + (uint64_t)leastDelta);
        if (before)
          *before = (Uint)shadow;
        if (delta)
          *delta = (Uint)(current - shadow);
      }
      coop.sync();
    }

    // ---- Shadow access ----
    XSHMEM_DEVICE_INLINE XSHMEM_FGP uint64_t *
    getSignalShadowPtr(flagcxDevSignal_t signalId) const {
      return &_dc.shadowBuffer[this->shadowIndex(signalId)];
    }

    XSHMEM_DEVICE_INLINE void increaseSignalShadow(flagcxDevSignal_t signalId,
                                                   uint64_t delta) const {
      XshmemGMWord *shadow = (XshmemGMWord *)this->getSignalShadowPtr(signalId);
      *shadow = *shadow + delta;
    }

    XSHMEM_DEVICE_INLINE uint64_t
    readSignal(flagcxDevSignal_t signalId, int,
               flagcxDeviceMemoryOrder_t) const {
      return this->sumSignal(signalId);
    }

    XSHMEM_DEVICE_INLINE void resetSignal(flagcxDevSignal_t signalId) const {
      int base = this->signalBase(signalId);
      for (int i = 0; i < FLAGCX_XSHMEM_SIGNAL_SLOTS(_dc.nRanks); ++i)
        *(XshmemGMWord *)&_dc.signalBuffer[base + i] = (uint64_t)0;
      *(XshmemGMWord *)this->getSignalShadowPtr(signalId) = (uint64_t)0;
    }

    // ---- Local signal write ----
    // Only the local-action slot is writable: the per-source slots belong to
    // their senders, so an absolute value can only be published locally.
    XSHMEM_DEVICE_INLINE void setSignal(flagcxDevSignal_t signalId,
                                        uint64_t value) const {
      *(XshmemGMWord *)this->getSignalLocalPtr(signalId) = value;
    }

    // ---- Counter interfaces ----
    template <typename Coop>
    XSHMEM_DEVICE_INLINE void
    waitCounter(Coop coop, flagcxDevCounter_t counterId, uint64_t least, int,
                flagcxDeviceMemoryOrder_t order) const {
      (void)order;
      coop.sync();
      if (coop.threadRank() == 0) {
        int iter = 0;
        while (this->sumCounter(counterId) < least)
          Intrin::spinBackoff(iter++);
      }
      coop.sync();
    }

    XSHMEM_DEVICE_INLINE uint64_t
    readCounter(flagcxDevCounter_t counterId, int,
                flagcxDeviceMemoryOrder_t) const {
      return this->sumCounter(counterId);
    }

    XSHMEM_DEVICE_INLINE void resetCounter(flagcxDevCounter_t counterId) const {
      int base = this->counterBase(counterId);
      for (int i = 0; i < FLAGCX_XSHMEM_SIGNAL_SLOTS(_dc.nRanks); ++i)
        *(XshmemGMWord *)&_dc.counterBuffer[base + i] = (uint64_t)0;
    }

    // ---- Collective: barrierAll ----
    XSHMEM_DEVICE_INLINE void barrierAll() const {
      flagcxXshmemDevice::barrierAll();
    }

    // ---- Two-sided: send/recv/term/wait ----
    template <typename Coop>
    XSHMEM_DEVICE_INLINE flagcxResult_t send(Coop, Window, size_t, size_t,
                                             flagcxDataType_t, int) const {
      return flagcxNotSupported;
    }

    template <typename Coop>
    XSHMEM_DEVICE_INLINE flagcxResult_t recv(Coop, Window, size_t, size_t,
                                             flagcxDataType_t, int) const {
      return flagcxNotSupported;
    }

    template <typename Coop>
    XSHMEM_DEVICE_INLINE flagcxResult_t term(Coop) const {
      return flagcxNotSupported;
    }

    template <typename Coop>
    XSHMEM_DEVICE_INLINE flagcxResult_t wait(Coop) const {
      return flagcxNotSupported;
    }

    // ---- One-sided: get ----
    // XCCL's installed XSHMEM versions do not expose one stable collective-get
    // spelling. Direct peer mappings are still usable, so provide the same
    // byte-granular path as Window::getPeerPointer and fail loudly when a peer
    // is not directly addressable.
    template <typename Coop>
    XSHMEM_DEVICE_INLINE void get(Team team, int peer, Window src,
                                  size_t srcOff, Window dst, size_t dstOff,
                                  size_t bytes, Coop coop) const {
      coop.sync();
      XSHMEM_FGP char *remote =
          (XSHMEM_FGP char *)src.getPeerPointer(srcOff, team, peer);
      XSHMEM_FGP char *local = (XSHMEM_FGP char *)dst.getLocalPointer(dstOff);
      if (remote == nullptr) {
        if (coop.threadRank() == 0)
          __builtin_trap();
        return;
      }
      for (size_t i = (size_t)coop.threadRank(); i < bytes;
           i += (size_t)coop.size())
        local[i] = remote[i];
      flagcxXshmemDevice::threadFence();
      coop.sync();
    }

  private:
    // Base word index of the fan-out block owned by one (context, signal).
    // Layout is described in xshmem_state_layout.h.
    XSHMEM_DEVICE_INLINE int signalBase(flagcxDevSignal_t signalId) const {
      return (_contextId * _dc.signalCount + (int)signalId) *
             FLAGCX_XSHMEM_SIGNAL_SLOTS(_dc.nRanks);
    }

    XSHMEM_DEVICE_INLINE int counterBase(flagcxDevCounter_t counterId) const {
      return (_contextId * _dc.counterCount + (int)counterId) *
             FLAGCX_XSHMEM_SIGNAL_SLOTS(_dc.nRanks);
    }

    // The shadow buffer is local and holds one aggregate per (context, signal),
    // so it keeps the original flat indexing.
    XSHMEM_DEVICE_INLINE int shadowIndex(flagcxDevSignal_t signalId) const {
      return _contextId * _dc.signalCount + (int)signalId;
    }

    // Ticket this PE has already sent to `dst` for one signal/counter. Written
    // only by this PE, so the plain read-add-write below has a single writer.
    XSHMEM_DEVICE_INLINE XSHMEM_FGP uint64_t *
    getSignalSentPtr(flagcxDevSignal_t signalId, int dst) const {
      return &_dc.signalBuffer[this->signalBase(signalId) + _dc.nRanks + dst];
    }

    XSHMEM_DEVICE_INLINE XSHMEM_FGP uint64_t *
    getCounterSentPtr(flagcxDevCounter_t counterId, int dst) const {
      return &_dc.counterBuffer[this->counterBase(counterId) + _dc.nRanks + dst];
    }

    // Slot for increments published by this PE to itself. Kept apart from the
    // per-source receive slots so a self-directed remote action (peer == rank,
    // which happens in AlltoAll) cannot clobber a local action's count.
    XSHMEM_DEVICE_INLINE XSHMEM_FGP uint64_t *
    getSignalLocalPtr(flagcxDevSignal_t signalId) const {
      return &_dc.signalBuffer[this->signalBase(signalId) + 2 * _dc.nRanks];
    }

    XSHMEM_DEVICE_INLINE XSHMEM_FGP uint64_t *
    getCounterLocalPtr(flagcxDevCounter_t counterId) const {
      return &_dc.counterBuffer[this->counterBase(counterId) + 2 * _dc.nRanks];
    }

    // Aggregate value of a signal: every source's ticket plus local actions.
    XSHMEM_DEVICE_INLINE uint64_t
    sumSignal(flagcxDevSignal_t signalId) const {
      int base = this->signalBase(signalId);
      uint64_t total = 0;
      for (int src = 0; src < _dc.nRanks; ++src)
        total += *(XshmemGMWord *)&_dc.signalBuffer[base + src];
      return total + *(XshmemGMWord *)this->getSignalLocalPtr(signalId);
    }

    XSHMEM_DEVICE_INLINE uint64_t
    sumCounter(flagcxDevCounter_t counterId) const {
      int base = this->counterBase(counterId);
      uint64_t total = 0;
      for (int src = 0; src < _dc.nRanks; ++src)
        total += *(XshmemGMWord *)&_dc.counterBuffer[base + src];
      return total + *(XshmemGMWord *)this->getCounterLocalPtr(counterId);
    }

    XSHMEM_DEVICE_INLINE uint64_t spinSignal(flagcxDevSignal_t signalId,
                                             uint64_t least) const {
      int iter = 0;
      uint64_t current = this->sumSignal(signalId);
      while (current < least) {
        Intrin::spinBackoff(iter++);
        current = this->sumSignal(signalId);
      }
      return current;
    }

    // Publish an increment of `delta` to `pe`. SET is the only signal op P800
    // implements, so send the running total this PE owes that destination.
    XSHMEM_DEVICE_INLINE void sendSignalTicket(int pe,
                                               flagcxDevSignal_t signalId,
                                               uint64_t delta) const {
      XshmemGMWord *sent = (XshmemGMWord *)this->getSignalSentPtr(signalId, pe);
      uint64_t ticket = *sent + delta;
      *sent = ticket;
      flagcxXshmemDevice::signalOp(this->getSignalPtr(signalId), ticket,
                                   XSHMEM_SIGNAL_SET, pe);
    }

    XSHMEM_DEVICE_INLINE void sendCounterTicket(int pe,
                                                flagcxDevCounter_t counterId,
                                                uint64_t delta) const {
      XshmemGMWord *sent =
          (XshmemGMWord *)this->getCounterSentPtr(counterId, pe);
      uint64_t ticket = *sent + delta;
      *sent = ticket;
      flagcxXshmemDevice::signalOp(this->getCounterPtr(counterId), ticket,
                                   XSHMEM_SIGNAL_SET, pe);
    }

    // ---- Cooperative data put (all cores issue one non-blocking call) ----
    static XSHMEM_DEVICE_INLINE void putData(XSHMEM_FGP float *dst,
                                             XSHMEM_FGP float *src,
                                             size_t bytes, int pe) {
      flagcxXshmemDevice::putFloatCluster(dst, src, bytes / sizeof(float), pe);
    }

    // ---- put dispatch ----
    template <typename RA, typename LA>
    XSHMEM_DEVICE_INLINE void putImpl(const Comm &, XSHMEM_FGP float *dst,
                                      XSHMEM_FGP float *src, size_t bytes,
                                      int pe, RA ra, LA la) const {
      putData(dst, src, bytes, pe);
      // Only reached from the aligned path, which is a cluster collective: the
      // whole block is here, so the cluster-scoped drain is the right one.
      flagcxXshmemDevice::quietCluster(pe);
      if (FLAGCX_THREAD_IDX_X == 0) {
        this->remoteActionImpl(pe, ra);
        this->localActionImpl(la);
      }
    }

    // Remote actions target the peer's symmetric signal/counter allocation.
    // P800 has no C2C atomic, so an increment is published as a SET of this
    // PE's private slot inside the peer (see xshmem_state_layout.h).
    XSHMEM_DEVICE_INLINE void
    remoteActionImpl(int pe, flagcxDevNet_SignalInc action) const {
      this->sendSignalTicket(pe, action.signal, (uint64_t)1);
    }
    XSHMEM_DEVICE_INLINE void
    remoteActionImpl(int pe, flagcxDevNet_SignalAdd action) const {
      this->sendSignalTicket(pe, action.signal, action.value);
    }
    XSHMEM_DEVICE_INLINE void
    remoteActionImpl(int pe, flagcxDevNet_CounterInc action) const {
      this->sendCounterTicket(pe, action.counter, (uint64_t)1);
    }
    template <typename Action>
    XSHMEM_DEVICE_INLINE void remoteActionImpl(int, Action) const {}

    // Local actions are published only after the put has completed locally.
    // They land in a slot no remote PE writes, so this PE is the sole writer.
    XSHMEM_DEVICE_INLINE void
    localActionImpl(flagcxDevNet_SignalInc action) const {
      this->bumpLocal(this->getSignalLocalPtr(action.signal), (uint64_t)1);
    }
    XSHMEM_DEVICE_INLINE void
    localActionImpl(flagcxDevNet_SignalAdd action) const {
      this->bumpLocal(this->getSignalLocalPtr(action.signal), action.value);
    }
    XSHMEM_DEVICE_INLINE void
    localActionImpl(flagcxDevNet_CounterInc action) const {
      this->bumpLocal(this->getCounterLocalPtr(action.counter), (uint64_t)1);
    }
    template <typename Action>
    XSHMEM_DEVICE_INLINE void localActionImpl(Action) const {}

    static XSHMEM_DEVICE_INLINE void bumpLocal(XSHMEM_FGP uint64_t *slot,
                                               uint64_t delta) {
      XshmemGMWord *w = (XshmemGMWord *)slot;
      *w = *w + delta;
    }
  };
};

// XSHMEM's installed device API only implements a barrier over
// XSHMEM_TEAM_WORLD (non_abi/device/coll/barrier.h asserts on any other team),
// so a team-scoped barrier is built here. Every local cluster arrives first,
// then cluster 0 publishes this PE's arrival ticket into the slot it owns
// inside each team member — the same single-writer + SET scheme xshmem's own
// barrier uses, because P800 has no remote atomic to accumulate with.
template <typename Coop>
XSHMEM_DEVICE_INLINE void
xshmemGridArrive(Coop &coop, volatile XSHMEM_FGP uint64_t *state, int worldBase,
                 int teamSize, int teamStride, int teamRank,
                 XSHMEM_FGP const int *teamPeMap) {
  int nclusters = FLAGCX_GRID_DIM_X;
  volatile XSHMEM_FGP uint64_t *localArrive = state;
  volatile XSHMEM_FGP uint64_t *arriveRelease = state + FLAGCX_DEVICE_CTA_COUNT;
  // This PE's own slot; its symmetric offset selects the same slot in the peer.
  XSHMEM_FGP uint64_t *myArriveSlot =
      (XSHMEM_FGP uint64_t *)(state + 2 * FLAGCX_DEVICE_CTA_COUNT + teamRank);

  // Complete operations issued by this cluster before publishing arrival.
  for (int member = 0; member < teamSize; ++member) {
    int pe = teamPeMap ? teamPeMap[member] : worldBase + member * teamStride;
    flagcxXshmemDevice::quietForCoop(coop, pe);
  }
  coop.sync();
  if (coop.threadRank() == 0)
    localArrive[FLAGCX_BLOCK_IDX_X]++;
  flagcxXshmemDevice::threadFence();
  coop.sync();

  uint64_t expected = localArrive[FLAGCX_BLOCK_IDX_X];
  if (FLAGCX_BLOCK_IDX_X == 0) {
    for (int i = coop.threadRank(); i < nclusters; i += coop.size()) {
      while (localArrive[i] < expected) {
      }
    }
    coop.sync();
    if (coop.threadRank() == 0) {
      for (int member = 0; member < teamSize; ++member) {
        int pe =
            teamPeMap ? teamPeMap[member] : worldBase + member * teamStride;
        flagcxXshmemDevice::signalOp(myArriveSlot, expected, XSHMEM_SIGNAL_SET,
                                     pe);
      }
    }
    coop.sync();
    for (int i = coop.threadRank(); i < nclusters; i += coop.size())
      arriveRelease[i] = expected;
    flagcxXshmemDevice::threadFence();
  } else if (coop.threadRank() == 0) {
    while (arriveRelease[FLAGCX_BLOCK_IDX_X] < expected) {
    }
  }
  coop.sync();
}

template <typename Coop>
XSHMEM_DEVICE_INLINE void
xshmemGridWait(Coop &coop, volatile XSHMEM_FGP uint64_t *state, int teamSize) {
  volatile XSHMEM_FGP uint64_t *localArrive = state;
  volatile XSHMEM_FGP uint64_t *arriveSlots =
      state + 2 * FLAGCX_DEVICE_CTA_COUNT;
  volatile XSHMEM_FGP uint64_t *waitRelease =
      state + 2 * FLAGCX_DEVICE_CTA_COUNT + FLAGCX_XSHMEM_MAX_PES;
  uint64_t expected = localArrive[FLAGCX_BLOCK_IDX_X];

  if (FLAGCX_BLOCK_IDX_X == 0) {
    // Each member owns one slot, so the barrier completes when every slot has
    // caught up to this generation instead of when one counter reaches N.
    for (int member = coop.threadRank(); member < teamSize;
         member += coop.size()) {
      while (arriveSlots[member] < expected) {
      }
    }
    coop.sync();
    if (coop.threadRank() == 0)
      flagcxXshmemDevice::threadFence();
    coop.sync();
    for (int i = coop.threadRank(); i < FLAGCX_GRID_DIM_X; i += coop.size())
      waitRelease[i] = expected;
    flagcxXshmemDevice::threadFence();
  } else if (coop.threadRank() == 0) {
    while (waitRelease[FLAGCX_BLOCK_IDX_X] < expected) {
    }
  }
  coop.sync();
}

template <typename Coop>
struct XshmemBarrierBase {
  Coop coop;
  volatile XSHMEM_FGP uint64_t *state;
  int worldBase;
  int teamSize;
  int teamStride;
  int teamRank;
  XSHMEM_FGP const int *teamPeMap;

  XSHMEM_DEVICE_INLINE XshmemBarrierBase()
      : coop(), state(nullptr), worldBase(0), teamSize(0), teamStride(1),
        teamRank(0), teamPeMap(nullptr) {}
  XSHMEM_DEVICE_INLINE XshmemBarrierBase(Coop c, XSHMEM_FGP uint64_t *s,
                                         int base, int size, int stride,
                                         int rank,
                                         XSHMEM_FGP const int *peMap = nullptr)
      : coop(c), state((volatile XSHMEM_FGP uint64_t *)s), worldBase(base),
        teamSize(size), teamStride(stride), teamRank(rank), teamPeMap(peMap) {}

  XSHMEM_DEVICE_INLINE void arrive(flagcxDeviceMemoryOrder_t) {
    if (state && teamSize > 0)
      xshmemGridArrive(coop, state, worldBase, teamSize, teamStride, teamRank,
                       teamPeMap);
  }
  XSHMEM_DEVICE_INLINE void wait(flagcxDeviceMemoryOrder_t) {
    if (state && teamSize > 0)
      xshmemGridWait(coop, state, teamSize);
  }
  XSHMEM_DEVICE_INLINE void sync(flagcxDeviceMemoryOrder_t order) {
    arrive(order);
    wait(order);
  }
};

template <typename Coop>
struct Barrier<XshmemBackend, flagcxTeamTagIntra, Coop>
    : XshmemBarrierBase<Coop> {
  using Base = XshmemBarrierBase<Coop>;
  using Comm = CommTraits<XshmemBackend>::Comm;
  using Team = CommTraits<XshmemBackend>::Team;
  using Multimem = CommTraits<XshmemBackend>::Multimem;

  XSHMEM_DEVICE_INLINE Barrier() : Base() {}
  XSHMEM_DEVICE_INLINE Barrier(Coop coop, const Comm &dc, Team team, uint32_t,
                               bool = false, const Multimem & = {})
      : Base(coop, dc.gridSyncState, 0, team.nRanks, 1, team.rank,
             dc.intraPeMap) {}
};

template <typename Coop>
struct Barrier<XshmemBackend, flagcxTeamTagInter, Coop>
    : XshmemBarrierBase<Coop> {
  using Base = XshmemBarrierBase<Coop>;
  using Comm = CommTraits<XshmemBackend>::Comm;
  using Team = CommTraits<XshmemBackend>::Team;
  using Net = CommTraits<XshmemBackend>::Net;

  XSHMEM_DEVICE_INLINE Barrier() : Base() {}
  XSHMEM_DEVICE_INLINE Barrier(Coop coop, const Net &, const Comm &dc,
                               Team team, uint32_t, int = 0)
      : Base(coop,
             dc.gridSyncState
                 ? dc.gridSyncState + FLAGCX_XSHMEM_BARRIER_STATE_STRIDE
                 : nullptr,
             dc.rank - team.rank * team.stride, team.nRanks, team.stride,
             team.rank) {}

  XSHMEM_DEVICE_INLINE void
  arrive(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
         flagcxDevNetFenceLevel = flagcxDevNetFenceLevel::Relaxed) {
    Base::arrive(order);
  }
  XSHMEM_DEVICE_INLINE void
  wait(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxDevNetFenceLevel = flagcxDevNetFenceLevel::Relaxed) {
    Base::wait(order);
  }
  XSHMEM_DEVICE_INLINE void
  sync(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxDevNetFenceLevel = flagcxDevNetFenceLevel::Relaxed) {
    Base::sync(order);
  }
};

template <typename Coop>
struct Barrier<XshmemBackend, flagcxTeamTagWorld, Coop>
    : XshmemBarrierBase<Coop> {
  using Base = XshmemBarrierBase<Coop>;
  using Comm = CommTraits<XshmemBackend>::Comm;
  using Net = CommTraits<XshmemBackend>::Net;

  XSHMEM_DEVICE_INLINE Barrier() : Base() {}
  XSHMEM_DEVICE_INLINE Barrier(Coop coop, flagcxTeamTagWorld, const Net &,
                               const Comm &dc, uint32_t, bool, int)
      : Base(coop,
             dc.gridSyncState
                 ? dc.gridSyncState + 2 * FLAGCX_XSHMEM_BARRIER_STATE_STRIDE
                 : nullptr,
             0, dc.nRanks, 1, dc.rank) {}
  XSHMEM_DEVICE_INLINE Barrier(Coop coop, flagcxTeamTagIntra, const Net &,
                               const Comm &dc, uint32_t, bool, int)
      : Base(coop, dc.gridSyncState, 0, dc.intraSize, 1, dc.intraRank,
             dc.intraPeMap) {}
  XSHMEM_DEVICE_INLINE Barrier(Coop coop, flagcxTeamTagInter, const Net &,
                               const Comm &dc, uint32_t, bool, int)
      : Base(coop,
             dc.gridSyncState
                 ? dc.gridSyncState + FLAGCX_XSHMEM_BARRIER_STATE_STRIDE
                 : nullptr,
             dc.intraRank, dc.intraSize > 0 ? dc.nRanks / dc.intraSize : 1,
             dc.intraSize > 0 ? dc.intraSize : 1,
             dc.intraSize > 0 ? (dc.rank - dc.intraRank) / dc.intraSize : 0) {}

  XSHMEM_DEVICE_INLINE void
  arrive(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
         flagcxDevNetFenceLevel = flagcxDevNetFenceLevel::Relaxed) {
    Base::arrive(order);
  }
  XSHMEM_DEVICE_INLINE void
  wait(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxDevNetFenceLevel = flagcxDevNetFenceLevel::Relaxed) {
    Base::wait(order);
  }
  XSHMEM_DEVICE_INLINE void
  sync(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxDevNetFenceLevel = flagcxDevNetFenceLevel::Relaxed) {
    Base::sync(order);
  }
};

#endif
