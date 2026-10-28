/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * DUSHMEM CommTraits — device-side backend using DUSHMEM PGAS APIs
 * (Sugon/HIP, DTK). Provides CommTraits<DushmemBackend> specialization:
 *   - Comm, Team, Window, Multimem (data types)
 *   - Net (one-sided: put, putValue, signal, flush, wait)
 *   - Barrier specializations (intra/inter/world via dushmemx_barrier_block)
 *
 * The Comm layout must stay identical to flagcxShmemCommInternal in
 * shmem/dushmem_adaptor.h — verified by static_asserts in the backend.
 ************************************************************************/

#ifndef FLAGCX_DUSHMEM_COMM_TRAITS_H_
#define FLAGCX_DUSHMEM_COMM_TRAITS_H_

#include "flagcx_kernel_core.h"
#include "du_platform_traits.h"
#include <dushmem.h>
#include <dushmemx.h>

// dushmem_uint64_wait_until and dushmem_putmem_signal are device-only (only
// declared under __HIPCC__ in DUSHMEM headers). Provide inline no-op stubs so
// template bodies compile in plain host .cc files (never called at runtime).
// dushmem_ptr, dushmemx_signal_op, and dushmemx_barrier_block already have
// unconditional host declarations — do NOT stub them.
#ifndef __HIPCC__
#include <cstddef>
#include <cstdint>
#ifndef DUSHMEM_CMP_GE
#define DUSHMEM_CMP_GE 3
#endif
#ifndef DUSHMEM_SIGNAL_ADD
#define DUSHMEM_SIGNAL_ADD 10
#endif
static inline void dushmem_uint64_wait_until(uint64_t *, int, uint64_t) {}
static inline void dushmem_putmem_signal(void *, const void *, size_t,
                                         uint64_t *, uint64_t, int, int) {}
#endif

struct DushmemBackend {};

template <>
struct CommTraits<DushmemBackend> {
  using Intrin = PlatformTraits<DuPlatform>::Intrin;
  using Atomic = PlatformTraits<DuPlatform>::Atomic;

  // ---- Multimem ----
  struct Multimem {
    void *mcBasePtr;
  };

  // ---- Team ----
  struct Team {
    int nRanks, rank, stride;
  };

  // ---- Window ----
  struct Window {
    void *symBase;
    size_t allocSize;
    void *rawPtr;

    FLAGCX_DEVICE_INLINE_DECORATOR void *
    getPeerPointer(size_t offset, const Team &team, int peer) const {
      int myPE = dushmem_my_pe();
      int base = myPE - team.rank * team.stride;
      int worldPeer = base + peer * team.stride;
      return dushmem_ptr((char *)symBase + offset, worldPeer);
    }

    // dushmem_ptr supports all declared access contracts.
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
      return (char *)rawPtr + offset;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void *getIntraPointer(size_t offset,
                                                         int peer) const {
      return dushmem_ptr((char *)symBase + offset, peer);
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
    getMulticastPointer(size_t, const Multimem &) const {
      return nullptr; // DUSHMEM doesn't use multicast
    }
    FLAGCX_HOST_DEVICE_INLINE void *getRawPtr() const { return rawPtr; }
    FLAGCX_HOST_DEVICE_INLINE bool hasAccess() const {
      return symBase != nullptr;
    }
    FLAGCX_HOST_DEVICE_INLINE void **getDevPeerPtrs() const { return nullptr; }
    FLAGCX_HOST_DEVICE_INLINE int getMrIndex() const { return 0; }
    FLAGCX_DEVICE_INLINE_DECORATOR bool operator==(const Window &o) const {
      return symBase == o.symBase;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR bool operator!=(const Window &o) const {
      return !(*this == o);
    }
  };

  // ---- Comm ----
  struct Comm {
    int rank, nRanks;
    int intraRank, intraSize;
    dushmem_team_t intraTeam;
    dushmem_team_t interTeam;
    dushmem_team_t worldTeam;

    uint64_t *signalBuffer;
    int signalCount;
    uint64_t *counterBuffer;
    int counterCount;
    uint64_t *shadowBuffer;

    uint64_t *gridSyncState; // per-block flags: arrive[CTA_COUNT] +
                             // release[CTA_COUNT], x3 barriers

    FLAGCX_DEVICE_INLINE_DECORATOR int getIntraRank() const {
      return intraRank;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int getIntraSize() const {
      return intraSize;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int getRank() const { return rank; }
    FLAGCX_DEVICE_INLINE_DECORATOR int getSize() const { return nRanks; }
    FLAGCX_DEVICE_INLINE_DECORATOR void *getFifoBuffer(int) const {
      return nullptr;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR Multimem getMulticastHandle() const {
      Multimem mm;
      mm.mcBasePtr = nullptr;
      return mm;
    }

    // P2P signal support — DUSHMEM always has P2P capability
    FLAGCX_DEVICE_INLINE_DECORATOR bool p2pSignalSupport(int /*peer*/) const {
      return true;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t *
    getSignalPeerPtr(int localPeer) const {
      int worldPeer = rank - intraRank + localPeer;
      return (uint64_t *)dushmem_ptr((void *)signalBuffer, worldPeer);
    }

    // Unified IR must use DUSHMEM signal operations rather than directly
    // atomically updating an implementation-owned symmetric buffer.
    FLAGCX_DEVICE_INLINE_DECORATOR bool usesDirectP2pSignals() const {
      return false;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR bool isOneSidedTransportReady() const {
      return true;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR bool supportsDirectCounterAccess() const {
      return counterBuffer != nullptr;
    }

    template <typename DI>
    static FLAGCX_HOST_DEVICE_INLINE void populateFromInternal(Comm &dc,
                                                               const DI &di) {
      // Only copy fields present in flagcxDevCommInternal (baseline).
      // DUSHMEM-specific fields (teams, barriers) are populated via devComm
      // pointer path, so this fallback only needs baseline fields.
      dc.rank = di.rank;
      dc.nRanks = di.nRanks;
      dc.intraRank = di.intraRank;
      dc.intraSize = di.intraSize;
      dc.intraTeam = DUSHMEM_TEAM_INVALID;
      dc.interTeam = DUSHMEM_TEAM_INVALID;
      dc.worldTeam = DUSHMEM_TEAM_WORLD;
      dc.signalBuffer = di.signalBuffer;
      dc.signalCount = di.signalCount;
      dc.counterBuffer = di.counterBuffer;
      dc.counterCount = di.counterCount;
      dc.shadowBuffer = di.shadowBuffer;
      dc.gridSyncState = nullptr;
    }
  };

  // ---- Coop types: aliased from PlatformTraits ----
  using CoopBlock = typename PlatformTraits<DuPlatform>::CoopBlock;
  template <int N>
  using CoopTile = typename PlatformTraits<DuPlatform>::template CoopTile<N>;
  using CoopThread = typename PlatformTraits<DuPlatform>::CoopThread;
  using CoopWarp = typename PlatformTraits<DuPlatform>::CoopWarp;
  using CoopTileSpan = typename PlatformTraits<DuPlatform>::CoopTileSpan;
  using CoopLanes = typename PlatformTraits<DuPlatform>::CoopLanes;
  using CoopAny = typename PlatformTraits<DuPlatform>::CoopAny;

  // ---- Barrier handles ----
  struct IntraBarrierHandle {
    int nBarriers;
  };
  struct InterBarrierHandle {
    int placeholder;
  };

  // ---- DescriptorSmem: empty for DUSHMEM ----
  struct DescriptorSmem {};

  // ---- Barrier alias ----
  template <typename Tag, typename Coop>
  using Barrier = ::Barrier<DushmemBackend, Tag, Coop>;

  // ---- Net ----
  struct Net {
    Comm _dc;
    int _contextId;

    FLAGCX_HOST_DEVICE_INLINE
    Net(const Comm &dc, int contextIndex) : _dc(dc), _contextId(contextIndex) {}

    FLAGCX_DEVICE_INLINE_DECORATOR bool isValid() const {
      if (_dc.signalCount > 0 && _dc.signalBuffer == nullptr)
        return false;
      if (_dc.counterCount > 0 && _dc.counterBuffer == nullptr)
        return false;
      return true;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR int getContextId() const {
      return _contextId;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t *
    getSignalPtr(flagcxDevSignal_t signalId) const {
      return &_dc.signalBuffer[signalIndex(signalId)];
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t *
    getPeerSignalPtr(int localPeer, flagcxDevSignal_t signalId) const {
      uint64_t *peerBuffer = _dc.getSignalPeerPtr(localPeer);
      return peerBuffer ? &peerBuffer[signalIndex(signalId)] : nullptr;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t *
    getCounterPtr(flagcxDevCounter_t counterId) const {
      return &_dc.counterBuffer[counterIndex(counterId)];
    }

    // ---- Helper: resolve PE from team + peer index ----
    FLAGCX_DEVICE_INLINE_DECORATOR int resolvePE(Team team, int peer) const {
      // peer is team-local rank; derive base from own world rank
      int base = _dc.rank - team.rank * team.stride;
      return base + peer * team.stride;
    }

    // ---- One-sided: put ----
    template <typename RA, typename LA, typename Coop, typename Desc>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    put(Team team, int peer, Window dst, size_t dstOff, Window src,
        size_t srcOff, size_t bytes, RA ra, LA la, Coop coop, Desc desc,
        flagcxDeviceScope_t ar, flagcxDeviceScope_t es) const {
      (void)desc;
      (void)ar;
      (void)es;
      coop.sync();
      if (coop.threadRank() == 0) {
        void *dstPtr = (char *)dst.symBase + dstOff;
        void *srcPtr = (char *)src.rawPtr + srcOff;
        int pe = resolvePE(team, peer);
        putImpl(dstPtr, srcPtr, bytes, pe, ra, la);
      }
      coop.sync();
    }

    // ---- One-sided: putValue ----
    template <typename T, typename RA, typename Coop, typename Desc>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    putValue(Team team, int peer, Window dst, size_t dstOff, T value, RA ra,
             Coop coop, Desc desc, flagcxDeviceScope_t ar,
             flagcxDeviceScope_t es) const {
      (void)desc;
      (void)ar;
      (void)es;
      coop.sync();
      if (coop.threadRank() == 0) {
        int pe = resolvePE(team, peer);
        void *dstPtr = (char *)dst.symBase + dstOff;
        dushmem_putmem(dstPtr, (const void *)&value, sizeof(T), pe);
        signalImpl(pe, ra);
      }
      coop.sync();
    }

    // ---- One-sided: signal ----
    template <typename RA, typename Coop, typename Desc>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    signal(Team team, int peer, RA ra, Coop coop, Desc desc,
           flagcxDeviceScope_t ar, flagcxDeviceScope_t es) const {
      (void)desc;
      (void)ar;
      (void)es;
      coop.sync();
      if (coop.threadRank() == 0) {
        int pe = resolvePE(team, peer);
        signalImpl(pe, ra);
      }
      coop.sync();
    }

    // ---- Ordering: flush ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    flush(Coop coop, flagcxDeviceMemoryOrder_t order) const {
      if (order == flagcxDeviceMemoryOrderAcqRel) {
        coop.sync();
        if (coop.threadRank() == 0)
          dushmem_quiet();
        coop.sync();
      } else {
        dushmem_fence();
      }
    }

    // ---- Wait: waitSignal ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitSignal(Coop coop, flagcxDevSignal_t signalId, uint64_t least, int bits,
               flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      (void)order;
      coop.sync();
      if (coop.threadRank() == 0) {
        uint64_t *addr = getSignalPtr(signalId);
        dushmem_uint64_wait_until(addr, DUSHMEM_CMP_GE, least);
      }
      coop.sync();
    }

    // ---- Wait: waitSignalMeetShadow ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitSignalMeetShadow(Coop coop, flagcxDevSignal_t signalId, int bits,
                         flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      (void)order;
      coop.sync();
      if (coop.threadRank() == 0) {
        uint64_t target = *getSignalShadowPtr(signalId);
        uint64_t *addr = getSignalPtr(signalId);
        dushmem_uint64_wait_until(addr, DUSHMEM_CMP_GE, target);
      }
      coop.sync();
    }

    // ---- Wait: waitSignalFollowShadow ----
    template <typename Coop, typename Uint>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitSignalFollowShadow(Coop coop, flagcxDevSignal_t signalId,
                           Uint leastDelta, Uint *before, Uint *delta, int bits,
                           flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      coop.sync();
      if (coop.threadRank() == 0) {
        uint64_t shadow = *getSignalShadowPtr(signalId);
        uint64_t target = shadow + (uint64_t)leastDelta;
        uint64_t *addr = getSignalPtr(signalId);
        dushmem_uint64_wait_until(addr, DUSHMEM_CMP_GE, target);
        uint64_t cur = Atomic::load(addr, order);
        if (before)
          *before = (Uint)shadow;
        if (delta)
          *delta = (Uint)(cur - shadow);
      }
      coop.sync();
    }

    // ---- Shadow access ----
    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t *
    getSignalShadowPtr(flagcxDevSignal_t signalId) const {
      return &_dc.shadowBuffer[signalIndex(signalId)];
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    increaseSignalShadow(flagcxDevSignal_t signalId, uint64_t delta) const {
      *getSignalShadowPtr(signalId) += delta;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
    readSignal(flagcxDevSignal_t signalId, int bits,
               flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      return Atomic::load(getSignalPtr(signalId), order);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    resetSignal(flagcxDevSignal_t signalId) const {
      Atomic::store(getSignalPtr(signalId), (uint64_t)0,
                    flagcxDeviceMemoryOrderRelease);
      Atomic::store(getSignalShadowPtr(signalId), (uint64_t)0,
                    flagcxDeviceMemoryOrderRelease);
    }

    // ---- Counter: waitCounter ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitCounter(Coop coop, flagcxDevCounter_t counterId, uint64_t least,
                int bits, flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      coop.sync();
      if (coop.threadRank() == 0) {
        uint64_t *counter = getCounterPtr(counterId);
        int iter = 0;
        while (Atomic::load(counter, order) < least) {
          Intrin::spinBackoff(iter++);
        }
      }
      coop.sync();
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
    readCounter(flagcxDevCounter_t counterId, int bits,
                flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      return Atomic::load(getCounterPtr(counterId), order);
    }

    // ---- Two-sided: send/recv/term/wait (DUSHMEM uses one-sided, these are
    //      no-op stubs to satisfy the DeviceAPI interface) ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t
    send(Coop coop, Window, size_t, size_t, flagcxDataType_t, int) const {
      (void)coop;
      return flagcxSuccess;
    }

    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t
    recv(Coop coop, Window, size_t, size_t, flagcxDataType_t, int) const {
      (void)coop;
      return flagcxSuccess;
    }

    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t term(Coop coop) const {
      (void)coop;
      return flagcxSuccess;
    }

    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t wait(Coop coop) const {
      (void)coop;
      return flagcxSuccess;
    }

    // ---- One-sided: get ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    get(Team team, int peer, Window src, size_t srcOff, Window dst,
        size_t dstOff, size_t bytes, Coop coop) const {
      coop.sync();
      if (coop.threadRank() == 0) {
        int pe = resolvePE(team, peer);
        void *dstPtr = (char *)dst.rawPtr + dstOff;
        void *srcPtr = (char *)src.symBase + srcOff;
        dushmem_getmem(dstPtr, srcPtr, bytes, pe);
      }
      coop.sync();
    }

  private:
    FLAGCX_DEVICE_INLINE_DECORATOR int
    signalIndex(flagcxDevSignal_t signalId) const {
      return _contextId * _dc.signalCount + (int)signalId;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR int
    counterIndex(flagcxDevCounter_t counterId) const {
      return _contextId * _dc.counterCount + (int)counterId;
    }

    // ---- put dispatch: select fused put+signal vs plain put ----
    template <typename LA>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    putImpl(void *dst, void *src, size_t bytes, int pe,
            flagcxDevNet_SignalInc ra, LA la) const {
      uint64_t *sigAddr = getSignalPtr(ra.signal);
      dushmem_putmem_signal(dst, src, bytes, sigAddr, 1, DUSHMEM_SIGNAL_ADD,
                            pe);
      counterImpl(la);
    }

    template <typename LA>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    putImpl(void *dst, void *src, size_t bytes, int pe,
            flagcxDevNet_SignalAdd ra, LA la) const {
      uint64_t *sigAddr = getSignalPtr(ra.signal);
      dushmem_putmem_signal(dst, src, bytes, sigAddr, ra.value,
                            DUSHMEM_SIGNAL_ADD, pe);
      counterImpl(la);
    }

    template <typename LA>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    putImpl(void *dst, void *src, size_t bytes, int pe,
            flagcxDevNet_CounterInc c, LA) const {
      dushmem_putmem(dst, src, bytes, pe);
      counterImpl(c);
    }

    template <typename RA, typename LA>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    putImpl(void *dst, void *src, size_t bytes, int pe, RA, LA la) const {
      dushmem_putmem(dst, src, bytes, pe);
      counterImpl(la);
    }

    // ---- signal dispatch ----
    FLAGCX_DEVICE_INLINE_DECORATOR void
    signalImpl(int pe, flagcxDevNet_SignalInc ra) const {
      uint64_t *sigAddr = getSignalPtr(ra.signal);
      dushmemx_signal_op(sigAddr, 1, DUSHMEM_SIGNAL_ADD, pe);
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void
    signalImpl(int pe, flagcxDevNet_SignalAdd ra) const {
      uint64_t *sigAddr = getSignalPtr(ra.signal);
      dushmemx_signal_op(sigAddr, ra.value, DUSHMEM_SIGNAL_ADD, pe);
    }
    template <typename RA>
    FLAGCX_DEVICE_INLINE_DECORATOR void signalImpl(int, RA) const {}

    // ---- counter helper ----
    FLAGCX_DEVICE_INLINE_DECORATOR void
    counterImpl(flagcxDevNet_CounterInc c) const {
      Atomic::fetchAdd(getCounterPtr(c.counter), (uint64_t)1,
                       flagcxDeviceMemoryOrderRelease);
    }
    template <typename LA>
    FLAGCX_DEVICE_INLINE_DECORATOR void counterImpl(LA) const {}

  public:
    // ---- reset counter ----
    FLAGCX_DEVICE_INLINE_DECORATOR void
    resetCounter(flagcxDevCounter_t counterId) const {
      Atomic::store(getCounterPtr(counterId), (uint64_t)0,
                    flagcxDeviceMemoryOrderRelease);
    }
  }; // struct Net
};   // struct CommTraits<DushmemBackend>

// ============================================================
// Barrier specializations for DushmemBackend
// ============================================================
// Grid-wide synchronization helpers for DUSHMEM barriers.
// Block 0 collects arrive flags from all blocks, performs the
// team-level (PE) barrier, then releases the other blocks.
// ============================================================

template <typename Coop>
FLAGCX_DEVICE_INLINE_DECORATOR void
dushmemGridArrive(Coop &coop, dushmem_team_t team, volatile uint64_t *arrive,
                  volatile uint64_t *release) {
#ifdef __HIPCC__
  int numBlocks = FLAGCX_GRID_DIM_X;
  coop.sync();
  if (coop.threadRank() == 0) {
    arrive[FLAGCX_BLOCK_IDX_X]++;
  }
  if (FLAGCX_BLOCK_IDX_X == 0) {
    coop.sync();
    uint64_t expected = arrive[0];
    for (int i = coop.threadRank(); i < numBlocks; i += FLAGCX_BLOCK_DIM_X) {
      if (i == 0)
        continue;
      while (arrive[i] < expected) {
      }
    }
    coop.sync();
  }
#endif
}

template <typename Coop>
FLAGCX_DEVICE_INLINE_DECORATOR void
dushmemGridWait(Coop &coop, dushmem_team_t team, volatile uint64_t *arrive,
                volatile uint64_t *release) {
#ifdef __HIPCC__
  int numBlocks = FLAGCX_GRID_DIM_X;
  if (FLAGCX_BLOCK_IDX_X == 0) {
    coop.sync();
    // PE-level barrier, executed by block 0 only.
    //
    // DUSHMEM's non-ABI block/warp-scoped collectives -- the family
    // nvshmemx_barrier_block() corresponds to -- are declared in
    // dushmemx_coll_api.h but never implemented and never reachable, so calling
    // dushmemx_barrier_block() leaves an unresolved device symbol and the GPU
    // jumps to address 0 (VMFault FaultAddress 0). Use the internal
    // threadgroup-scoped barrier the ABI wrappers build on
    // (dushmem_barrier_all() -> dushmemi_barrier_threadgroup<thread>): with
    // SCOPE = BLOCK only thread 0 of the block advances the team counter and
    // the block's threads cover the team members, which is exactly the
    // one-block-per-PE contract this grid barrier needs.
    // The device-pass guard is required because dushmemi_barrier_threadgroup is
    // only declared under __HIP_DEVICE_COMPILE__ (non-ABI headers).
#if defined(__HIP_DEVICE_COMPILE__)
    if (team != DUSHMEM_TEAM_INVALID)
      dushmemi_barrier_threadgroup<DUSHMEMI_THREADGROUP_BLOCK>(team);
#endif
    coop.sync();
    for (int i = coop.threadRank(); i < numBlocks; i += FLAGCX_BLOCK_DIM_X) {
      release[i]++;
    }
  } else {
    if (coop.threadRank() == 0) {
      uint64_t cur = release[FLAGCX_BLOCK_IDX_X];
      while (release[FLAGCX_BLOCK_IDX_X] == cur) {
      }
    }
  }
  coop.sync();
#endif
}

// ============================================================
// Signal-based split-phase barriers using dushmemx_signal_op.
// ============================================================

// ---- Barrier<DushmemBackend, flagcxTeamTagIntra, Coop> ----
template <typename Coop>
struct Barrier<DushmemBackend, flagcxTeamTagIntra, Coop> {
  using Intrin = PlatformTraits<DuPlatform>::Intrin;
  using Atomic = PlatformTraits<DuPlatform>::Atomic;
  using Comm = CommTraits<DushmemBackend>::Comm;
  using Team = CommTraits<DushmemBackend>::Team;
  using Multimem = CommTraits<DushmemBackend>::Multimem;

  Coop _coop;
  dushmem_team_t _team;
  int _teamSize, _teamRank;
  volatile uint64_t *_gridSyncState; // arrive[CTA_COUNT] + release[CTA_COUNT]

  FLAGCX_DEVICE_INLINE_DECORATOR Barrier()
      : _coop(), _team(DUSHMEM_TEAM_INVALID), _teamSize(0), _teamRank(0),
        _gridSyncState(nullptr) {}

  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, const Comm &dc, Team team, uint32_t index, bool = false,
          const Multimem & = {})
      : _coop(coop), _team(dc.intraTeam), _teamSize(dc.intraSize),
        _teamRank(dc.intraRank),
        _gridSyncState((volatile uint64_t *)dc.gridSyncState) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    dushmemGridArrive(_coop, _team, _gridSyncState,
                      _gridSyncState + FLAGCX_DEVICE_CTA_COUNT);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    dushmemGridWait(_coop, _team, _gridSyncState,
                    _gridSyncState + FLAGCX_DEVICE_CTA_COUNT);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    dushmemGridArrive(_coop, _team, _gridSyncState,
                      _gridSyncState + FLAGCX_DEVICE_CTA_COUNT);
    dushmemGridWait(_coop, _team, _gridSyncState,
                    _gridSyncState + FLAGCX_DEVICE_CTA_COUNT);
  }
};

// ---- Barrier<DushmemBackend, flagcxTeamTagInter, Coop> ----
template <typename Coop>
struct Barrier<DushmemBackend, flagcxTeamTagInter, Coop> {
  using Intrin = PlatformTraits<DuPlatform>::Intrin;
  using Atomic = PlatformTraits<DuPlatform>::Atomic;
  using Comm = CommTraits<DushmemBackend>::Comm;
  using Team = CommTraits<DushmemBackend>::Team;
  using Net = CommTraits<DushmemBackend>::Net;
  using Multimem = CommTraits<DushmemBackend>::Multimem;

  Coop _coop;
  dushmem_team_t _team;
  int _teamSize, _teamRank;
  volatile uint64_t *_gridSyncState; // arrive[CTA_COUNT] + release[CTA_COUNT]

  FLAGCX_DEVICE_INLINE_DECORATOR Barrier()
      : _coop(), _team(DUSHMEM_TEAM_INVALID), _teamSize(0), _teamRank(0),
        _gridSyncState(nullptr) {}

  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, const Net &, const Comm &dc, Team, uint32_t index, int = 0)
      : _coop(coop), _team(dc.interTeam),
        _teamSize((dc.intraSize > 0) ? dc.nRanks / dc.intraSize : 1),
        _teamRank((dc.intraSize > 0) ? dc.rank / dc.intraSize : 0),
        _gridSyncState((volatile uint64_t *)(dc.gridSyncState +
                                             2 * FLAGCX_DEVICE_CTA_COUNT)) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
         flagcxDevNetFenceLevel = flagcxDevNetFenceLevel::Relaxed) {
    dushmemGridArrive(_coop, _team, _gridSyncState,
                      _gridSyncState + FLAGCX_DEVICE_CTA_COUNT);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxDevNetFenceLevel = flagcxDevNetFenceLevel::Relaxed) {
    dushmemGridWait(_coop, _team, _gridSyncState,
                    _gridSyncState + FLAGCX_DEVICE_CTA_COUNT);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    dushmemGridArrive(_coop, _team, _gridSyncState,
                      _gridSyncState + FLAGCX_DEVICE_CTA_COUNT);
    dushmemGridWait(_coop, _team, _gridSyncState,
                    _gridSyncState + FLAGCX_DEVICE_CTA_COUNT);
  }
};

// ---- Barrier<DushmemBackend, flagcxTeamTagWorld, Coop> ----
template <typename Coop>
struct Barrier<DushmemBackend, flagcxTeamTagWorld, Coop> {
  using Intrin = PlatformTraits<DuPlatform>::Intrin;
  using Atomic = PlatformTraits<DuPlatform>::Atomic;
  using Comm = CommTraits<DushmemBackend>::Comm;
  using Team = CommTraits<DushmemBackend>::Team;
  using Net = CommTraits<DushmemBackend>::Net;
  using Multimem = CommTraits<DushmemBackend>::Multimem;

  Coop _coop;
  dushmem_team_t _team;
  volatile uint64_t *_gridSyncState; // arrive[CTA_COUNT] + release[CTA_COUNT]

  FLAGCX_DEVICE_INLINE_DECORATOR Barrier()
      : _coop(), _team(DUSHMEM_TEAM_INVALID), _gridSyncState(nullptr) {}

  // World barrier (intra + inter)
  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, flagcxTeamTagWorld, const Net &net, const Comm &dc,
          uint32_t index, bool multimem, int nInterPeers)
      : _coop(coop), _team(dc.worldTeam),
        _gridSyncState((volatile uint64_t *)(dc.gridSyncState +
                                             4 * FLAGCX_DEVICE_CTA_COUNT)) {}

  // Intra-only barrier
  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, flagcxTeamTagIntra, const Net &, const Comm &dc,
          uint32_t index, bool, int)
      : _coop(coop), _team(dc.intraTeam),
        _gridSyncState((volatile uint64_t *)dc.gridSyncState) {}

  // Inter-only barrier
  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, flagcxTeamTagInter, const Net &net, const Comm &dc,
          uint32_t index, bool, int nInterPeers)
      : _coop(coop), _team(dc.interTeam),
        _gridSyncState((volatile uint64_t *)(dc.gridSyncState +
                                             2 * FLAGCX_DEVICE_CTA_COUNT)) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
         flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    dushmemGridArrive(_coop, _team, _gridSyncState,
                      _gridSyncState + FLAGCX_DEVICE_CTA_COUNT);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    dushmemGridWait(_coop, _team, _gridSyncState,
                    _gridSyncState + FLAGCX_DEVICE_CTA_COUNT);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    dushmemGridArrive(_coop, _team, _gridSyncState,
                      _gridSyncState + FLAGCX_DEVICE_CTA_COUNT);
    dushmemGridWait(_coop, _team, _gridSyncState,
                    _gridSyncState + FLAGCX_DEVICE_CTA_COUNT);
  }
};

#endif // FLAGCX_DUSHMEM_COMM_TRAITS_H_
