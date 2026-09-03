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
XSHMEM_DEVICE_INLINE XSHMEM_FGP void *peerPtr(XSHMEM_FGP void *ptr, int pe) {
  return xshmem_ptr(ptr, pe);
}
XSHMEM_DEVICE_INLINE void threadFence() { mfence(); }
XSHMEM_DEVICE_INLINE void quiet(int pe) {
  xshmemi_quiet<XSHMEMI_THREADGROUP_CLUSTER>(pe);
}
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
XSHMEM_DEVICE_INLINE void *peerPtr(void *, int) { return nullptr; }
XSHMEM_DEVICE_INLINE void threadFence() {}
XSHMEM_DEVICE_INLINE void quiet(int) {}
XSHMEM_DEVICE_INLINE void signalOp(uint64_t *, uint64_t, int, int) {}
XSHMEM_DEVICE_INLINE void waitUntil(uint64_t *, int, uint64_t) {}
XSHMEM_DEVICE_INLINE void fence() {}
XSHMEM_DEVICE_INLINE void barrierAll() {}
XSHMEM_DEVICE_INLINE void putFloatCluster(float *, float *, size_t, int) {}
#endif
} // namespace flagcxXshmemDevice

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

    XSHMEM_DEVICE_INLINE XSHMEM_FGP void *
    getPeerPointer(size_t offset, const Team &team, int peer) const {
      int myPe = flagcxXshmemDevice::myPe();
      int worldPeer;
      if (intraPeMap != nullptr && team.nRanks == intraSize &&
          team.rank == intraRank && team.stride == 1) {
        worldPeer = intraPeMap[peer];
      } else {
        int base = myPe - team.rank * team.stride;
        worldPeer = base + peer * team.stride;
      }
      return flagcxXshmemDevice::peerPtr((XSHMEM_FGP char *)symBase + offset,
                                         worldPeer);
    }
    XSHMEM_DEVICE_INLINE XSHMEM_FGP void *getLocalPointer(size_t offset) const {
      return (XSHMEM_FGP char *)rawPtr + offset;
    }
    XSHMEM_DEVICE_INLINE XSHMEM_FGP void *getIntraPointer(size_t offset,
                                                          int peer) const {
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

    XSHMEM_DEVICE_INLINE int getIntraRank() const { return intraRank; }
    XSHMEM_DEVICE_INLINE int getIntraSize() const { return intraSize; }
    XSHMEM_DEVICE_INLINE int getRank() const { return rank; }
    XSHMEM_DEVICE_INLINE int getSize() const { return nRanks; }
    XSHMEM_DEVICE_INLINE void *getFifoBuffer(int) const { return nullptr; }
    XSHMEM_DEVICE_INLINE Multimem getMulticastHandle() const {
      Multimem mm;
      mm.mcBasePtr = nullptr;
      return mm;
    }

    XSHMEM_DEVICE_INLINE bool p2pSignalSupport(int localPeer) const {
      return getSignalPeerPtr(localPeer) != nullptr;
    }
    XSHMEM_DEVICE_INLINE XSHMEM_FGP uint64_t *
    getSignalPeerPtr(int localPeer) const {
      int worldPeer = intraPeMap[localPeer];
      return (XSHMEM_FGP uint64_t *)flagcxXshmemDevice::peerPtr(signalBuffer,
                                                                worldPeer);
    }
    XSHMEM_DEVICE_INLINE bool usesDirectP2pSignals() const { return false; }
    XSHMEM_DEVICE_INLINE bool isOneSidedTransportReady() const { return true; }
    XSHMEM_DEVICE_INLINE bool supportsDirectCounterAccess() const {
      return counterBuffer != nullptr;
    }

    template <typename DI>
    static XSHMEM_HOST_DEVICE_INLINE void populateFromInternal(Comm &dc,
                                                               const DI &di) {
      dc.rank = di.rank;
      dc.nRanks = di.nRanks;
      dc.intraRank = di.intraRank;
      dc.intraSize = di.intraSize;
      dc.intraPeMap = di.intraPeMap;
      dc.intraTeam = XSHMEM_TEAM_INVALID;
      dc.interTeam = XSHMEM_TEAM_INVALID;
      dc.worldTeam = XSHMEM_TEAM_WORLD;
      dc.signalBuffer = di.signalBuffer;
      dc.signalCount = di.signalCount;
      dc.counterBuffer = di.counterBuffer;
      dc.counterCount = di.counterCount;
      dc.shadowBuffer = di.shadowBuffer;
      dc.gridSyncState = nullptr;
      dc.devStateHandle = nullptr;
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

    XSHMEM_DEVICE_INLINE XSHMEM_FGP uint64_t *
    getSignalPtr(flagcxDevSignal_t signalId) const {
      return &_dc.signalBuffer[signalIndex(signalId)];
    }

    XSHMEM_DEVICE_INLINE XSHMEM_FGP uint64_t *
    getPeerSignalPtr(int, flagcxDevSignal_t) const {
      return nullptr;
    }

    XSHMEM_DEVICE_INLINE XSHMEM_FGP uint64_t *
    getCounterPtr(flagcxDevCounter_t counterId) const {
      return &_dc.counterBuffer[counterIndex(counterId)];
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
      // XSHMEM currently exposes only a cluster-wide collective/synchronizer.
      // A partial group cannot safely publish completion after its members'
      // stores, so reject it for both aligned and byte-copy paths.
      if (coop.size() != FLAGCX_BLOCK_DIM_X) {
        if (coop.threadRank() == 0)
          __builtin_trap();
        return;
      }
      if (((dstOff | srcOff | bytes) & (sizeof(float) - 1)) != 0) {
        XSHMEM_FGP char *remote =
            (XSHMEM_FGP char *)dst.getPeerPointer(dstOff, team, peer);
        XSHMEM_FGP char *local = (XSHMEM_FGP char *)src.getLocalPointer(srcOff);
        if (remote == nullptr) {
          if (coop.threadRank() == 0)
            __builtin_trap();
          return;
        }
        for (size_t i = (size_t)coop.threadRank(); i < bytes;
             i += (size_t)coop.size())
          remote[i] = local[i];
        flagcxXshmemDevice::threadFence();
        coop.sync();
        if (coop.threadRank() == 0) {
          int pe = resolvePE(_dc, team, peer);
          remoteActionImpl(pe, ra);
          localActionImpl(la);
        }
        coop.sync();
        return;
      }

      // The installed aligned put is a cluster collective.
      coop.sync();
      int pe = resolvePE(_dc, team, peer);
      putImpl(_dc,
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
      XSHMEM_FGP T *remote =
          (XSHMEM_FGP T *)dst.getPeerPointer(dstOff, team, peer);
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
      flagcxXshmemDevice::quiet(pe);
      if (coop.threadRank() == 0) {
        remoteActionImpl(pe, ra);
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
        remoteActionImpl(pe, ra);
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
          flagcxXshmemDevice::quiet(pe);
        coop.sync();
      } else {
        flagcxXshmemDevice::fence();
      }
    }

    // ---- Wait: waitSignal ----
    template <typename Coop>
    XSHMEM_DEVICE_INLINE void
    waitSignal(Coop coop, flagcxDevNetSignal_t signalId, uint64_t least,
               int bits, flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      (void)order;
      coop.sync();
      if (coop.threadRank() == 0) {
        XSHMEM_FGP uint64_t *addr = getSignalPtr(signalId);
        flagcxXshmemDevice::waitUntil(addr, XSHMEM_CMP_GE, least);
      }
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
        flagcxXshmemDevice::waitUntil(getSignalPtr(signalId), XSHMEM_CMP_GE,
                                      *getSignalShadowPtr(signalId));
      coop.sync();
    }

    // ---- Wait: waitSignalFollowShadow ----
    template <typename Coop, typename Uint>
    XSHMEM_DEVICE_INLINE void
    waitSignalFollowShadow(Coop coop, flagcxDevSignal_t signalId,
                           Uint leastDelta, Uint *before, Uint *delta, int bits,
                           flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      coop.sync();
      if (coop.threadRank() == 0) {
        uint64_t shadow = *getSignalShadowPtr(signalId);
        flagcxXshmemDevice::waitUntil(getSignalPtr(signalId), XSHMEM_CMP_GE,
                                      shadow + (uint64_t)leastDelta);
        uint64_t current = Atomic::load(getSignalPtr(signalId), order);
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
      return &_dc.shadowBuffer[signalIndex(signalId)];
    }

    XSHMEM_DEVICE_INLINE void increaseSignalShadow(flagcxDevSignal_t signalId,
                                                   uint64_t delta) const {
      Atomic::fetchAdd(getSignalShadowPtr(signalId), delta,
                       flagcxDeviceMemoryOrderRelease);
    }

    XSHMEM_DEVICE_INLINE uint64_t
    readSignal(flagcxDevSignal_t signalId, int,
               flagcxDeviceMemoryOrder_t order) const {
      return Atomic::load(getSignalPtr(signalId), order);
    }

    XSHMEM_DEVICE_INLINE void resetSignal(flagcxDevSignal_t signalId) const {
      Atomic::store(getSignalPtr(signalId), (uint64_t)0,
                    flagcxDeviceMemoryOrderRelease);
      Atomic::store(getSignalShadowPtr(signalId), (uint64_t)0,
                    flagcxDeviceMemoryOrderRelease);
    }

    // ---- Local signal write ----
    XSHMEM_DEVICE_INLINE void setSignal(flagcxDevSignal_t signalId,
                                        uint64_t value) const {
      Atomic::store(getSignalPtr(signalId), value,
                    flagcxDeviceMemoryOrderRelease);
    }

    // ---- Counter interfaces ----
    template <typename Coop>
    XSHMEM_DEVICE_INLINE void
    waitCounter(Coop coop, flagcxDevCounter_t counterId, uint64_t least, int,
                flagcxDeviceMemoryOrder_t order) const {
      coop.sync();
      if (coop.threadRank() == 0) {
        int iter = 0;
        while (Atomic::load(getCounterPtr(counterId), order) < least)
          Intrin::spinBackoff(iter++);
      }
      coop.sync();
    }

    XSHMEM_DEVICE_INLINE uint64_t
    readCounter(flagcxDevCounter_t counterId, int,
                flagcxDeviceMemoryOrder_t order) const {
      return Atomic::load(getCounterPtr(counterId), order);
    }

    XSHMEM_DEVICE_INLINE void resetCounter(flagcxDevCounter_t counterId) const {
      Atomic::store(getCounterPtr(counterId), (uint64_t)0,
                    flagcxDeviceMemoryOrderRelease);
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
    XSHMEM_DEVICE_INLINE int signalIndex(flagcxDevSignal_t signalId) const {
      return _contextId * _dc.signalCount + (int)signalId;
    }

    XSHMEM_DEVICE_INLINE int counterIndex(flagcxDevCounter_t counterId) const {
      return _contextId * _dc.counterCount + (int)counterId;
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
      flagcxXshmemDevice::quiet(pe);
      if (FLAGCX_THREAD_IDX_X == 0) {
        remoteActionImpl(pe, ra);
        localActionImpl(la);
      }
    }

    // Remote actions target the peer's symmetric signal/counter allocation.
    XSHMEM_DEVICE_INLINE void
    remoteActionImpl(int pe, flagcxDevNet_SignalInc action) const {
      flagcxXshmemDevice::signalOp(getSignalPtr(action.signal), 1,
                                   XSHMEM_SIGNAL_ADD, pe);
    }
    XSHMEM_DEVICE_INLINE void
    remoteActionImpl(int pe, flagcxDevNet_SignalAdd action) const {
      flagcxXshmemDevice::signalOp(getSignalPtr(action.signal), action.value,
                                   XSHMEM_SIGNAL_ADD, pe);
    }
    XSHMEM_DEVICE_INLINE void
    remoteActionImpl(int pe, flagcxDevNet_CounterInc action) const {
      flagcxXshmemDevice::signalOp(getCounterPtr(action.counter), 1,
                                   XSHMEM_SIGNAL_ADD, pe);
    }
    template <typename Action>
    XSHMEM_DEVICE_INLINE void remoteActionImpl(int, Action) const {}

    // Local actions are published only after the put has completed locally.
    XSHMEM_DEVICE_INLINE void
    localActionImpl(flagcxDevNet_SignalInc action) const {
      Atomic::fetchAdd(getSignalPtr(action.signal), (uint64_t)1,
                       flagcxDeviceMemoryOrderRelease);
    }
    XSHMEM_DEVICE_INLINE void
    localActionImpl(flagcxDevNet_SignalAdd action) const {
      Atomic::fetchAdd(getSignalPtr(action.signal), action.value,
                       flagcxDeviceMemoryOrderRelease);
    }
    XSHMEM_DEVICE_INLINE void
    localActionImpl(flagcxDevNet_CounterInc action) const {
      Atomic::fetchAdd(getCounterPtr(action.counter), (uint64_t)1,
                       flagcxDeviceMemoryOrderRelease);
    }
    template <typename Action>
    XSHMEM_DEVICE_INLINE void localActionImpl(Action) const {}
  };
};

// XSHMEM's installed device API does not expose a portable team collective.
// Build a scoped barrier from symmetric counters: every local cluster first
// arrives, then cluster 0 signals only the PEs in the requested FlagCX team.
template <typename Coop>
XSHMEM_DEVICE_INLINE void
xshmemGridArrive(Coop &coop, volatile XSHMEM_FGP uint64_t *state, int worldBase,
                 int teamSize, int teamStride,
                 XSHMEM_FGP const int *teamPeMap) {
  int nclusters = FLAGCX_GRID_DIM_X;
  volatile XSHMEM_FGP uint64_t *localArrive = state;
  volatile XSHMEM_FGP uint64_t *arriveRelease = state + FLAGCX_DEVICE_CTA_COUNT;
  XSHMEM_FGP uint64_t *remoteArrive =
      (XSHMEM_FGP uint64_t *)(state + 2 * FLAGCX_DEVICE_CTA_COUNT);

  // Complete operations issued by this cluster before publishing arrival.
  for (int member = 0; member < teamSize; ++member) {
    int pe = teamPeMap ? teamPeMap[member] : worldBase + member * teamStride;
    flagcxXshmemDevice::quiet(pe);
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
        flagcxXshmemDevice::signalOp(remoteArrive, 1, XSHMEM_SIGNAL_ADD, pe);
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
  XSHMEM_FGP uint64_t *remoteArrive =
      (XSHMEM_FGP uint64_t *)(state + 2 * FLAGCX_DEVICE_CTA_COUNT);
  volatile XSHMEM_FGP uint64_t *waitRelease =
      state + 2 * FLAGCX_DEVICE_CTA_COUNT + 1;
  uint64_t expected = localArrive[FLAGCX_BLOCK_IDX_X];

  if (FLAGCX_BLOCK_IDX_X == 0) {
    coop.sync();
    if (coop.threadRank() == 0) {
      flagcxXshmemDevice::waitUntil(remoteArrive, XSHMEM_CMP_GE,
                                    expected * (uint64_t)teamSize);
      flagcxXshmemDevice::threadFence();
    }
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
  XSHMEM_FGP const int *teamPeMap;

  XSHMEM_DEVICE_INLINE XshmemBarrierBase()
      : coop(), state(nullptr), worldBase(0), teamSize(0), teamStride(1),
        teamPeMap(nullptr) {}
  XSHMEM_DEVICE_INLINE XshmemBarrierBase(Coop c, XSHMEM_FGP uint64_t *s,
                                         int base, int size, int stride,
                                         XSHMEM_FGP const int *peMap = nullptr)
      : coop(c), state((volatile XSHMEM_FGP uint64_t *)s), worldBase(base),
        teamSize(size), teamStride(stride), teamPeMap(peMap) {}

  XSHMEM_DEVICE_INLINE void arrive(flagcxDeviceMemoryOrder_t) {
    if (state && teamSize > 0)
      xshmemGridArrive(coop, state, worldBase, teamSize, teamStride, teamPeMap);
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
      : Base(coop, dc.gridSyncState, 0, team.nRanks, 1, dc.intraPeMap) {}
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
             dc.rank - team.rank * team.stride, team.nRanks, team.stride) {}

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
             0, dc.nRanks, 1) {}
  XSHMEM_DEVICE_INLINE Barrier(Coop coop, flagcxTeamTagIntra, const Net &,
                               const Comm &dc, uint32_t, bool, int)
      : Base(coop, dc.gridSyncState, 0, dc.intraSize, 1, dc.intraPeMap) {}
  XSHMEM_DEVICE_INLINE Barrier(Coop coop, flagcxTeamTagInter, const Net &,
                               const Comm &dc, uint32_t, bool, int)
      : Base(coop,
             dc.gridSyncState
                 ? dc.gridSyncState + FLAGCX_XSHMEM_BARRIER_STATE_STRIDE
                 : nullptr,
             dc.intraRank, dc.intraSize > 0 ? dc.nRanks / dc.intraSize : 1,
             dc.intraSize > 0 ? dc.intraSize : 1) {}

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
