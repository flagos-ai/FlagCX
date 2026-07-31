/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * NVSHMEM CommTraits — device-side backend using NVSHMEM PGAS APIs.
 * Provides CommTraits<NvshmemBackend> specialization with:
 *   - Comm, Team, Window, Multimem (data types)
 *   - Net (one-sided: put, putValue, signal, flush, wait)
 *   - Barrier specializations (intra/inter/world via nvshmemx_signal_op)
 ************************************************************************/

#ifndef FLAGCX_NVSHMEM_COMM_TRAITS_H_
#define FLAGCX_NVSHMEM_COMM_TRAITS_H_

#include "flagcx_kernel_core.h"
#include <nvshmem.h>
#include <nvshmemx.h>

// nvshmem_uint64_wait_until, nvshmem_putmem_signal, and nvshmem_ptr are
// device-only (no host declaration in NVSHMEM headers). Provide inline no-op
// stubs so template bodies compile in host .cc files (never called at runtime).
// nvshmemx_signal_op already has a host declaration — do NOT stub it.
#ifndef __CUDACC__
#include <cstddef>
#include <cstdint>
#ifndef NVSHMEM_CMP_GE
#define NVSHMEM_CMP_GE 3
#endif
#ifndef NVSHMEM_SIGNAL_ADD
#define NVSHMEM_SIGNAL_ADD 1
#endif
static inline void nvshmem_uint64_wait_until(uint64_t *, int, uint64_t) {}
static inline void nvshmem_putmem_signal(void *, const void *, size_t,
                                         uint64_t *, uint64_t, int, int) {}
static inline void *nvshmem_ptr(void *, int) { return nullptr; }
#endif

struct NvshmemBackend {};

template <>
struct CommTraits<NvshmemBackend> {
  using Intrin = PlatformTraits<NvidiaPlatform>::Intrin;
  using Atomic = PlatformTraits<NvidiaPlatform>::Atomic;

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
    getPeerPointer(size_t offset, const Team &, int peer) const {
      return nvshmem_ptr((char *)symBase + offset, peer);
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void *getLocalPointer(size_t offset) const {
      return (char *)rawPtr + offset;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void *getIntraPointer(size_t offset,
                                                         int peer) const {
      return nvshmem_ptr((char *)symBase + offset, peer);
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void *
    getMulticastPointer(size_t, const Multimem &) const {
      return nullptr; // NVSHMEM doesn't use multicast
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
    nvshmem_team_t intraTeam;
    nvshmem_team_t interTeam;
    nvshmem_team_t worldTeam;

    uint64_t *signalBuffer;
    int signalCount;
    uint64_t *counterBuffer;
    int counterCount;
    uint64_t *shadowBuffer;

    uint64_t *intraBarrierSignals;
    uint64_t *interBarrierSignals;
    uint64_t *worldBarrierSignals;
    uint64_t *barrierUsage;

    int intraBarrierCount;
    int interBarrierCount;
    int worldBarrierCount;

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

    template <typename DI>
    static FLAGCX_HOST_DEVICE_INLINE void populateFromInternal(Comm &dc,
                                                               const DI &di) {
      // Only copy fields present in flagcxDevCommInternal (baseline).
      // NVSHMEM-specific fields (teams, barriers) are populated via devComm
      // pointer path, so this fallback only needs baseline fields.
      dc.rank = di.rank;
      dc.nRanks = di.nRanks;
      dc.intraRank = di.intraRank;
      dc.intraSize = di.intraSize;
      dc.intraTeam = NVSHMEM_TEAM_INVALID;
      dc.interTeam = NVSHMEM_TEAM_INVALID;
      dc.worldTeam = NVSHMEM_TEAM_WORLD;
      dc.signalBuffer = di.signalBuffer;
      dc.signalCount = di.signalCount;
      dc.counterBuffer = di.counterBuffer;
      dc.counterCount = di.counterCount;
      dc.shadowBuffer = di.shadowBuffer;
      dc.intraBarrierSignals = nullptr;
      dc.interBarrierSignals = nullptr;
      dc.worldBarrierSignals = nullptr;
      dc.barrierUsage = nullptr;
      dc.intraBarrierCount = 0;
      dc.interBarrierCount = 0;
      dc.worldBarrierCount = 0;
    }
  };

  // ---- Coop types: aliased from PlatformTraits ----
  using CoopBlock = typename PlatformTraits<NvidiaPlatform>::CoopBlock;
  template <int N>
  using CoopTile =
      typename PlatformTraits<NvidiaPlatform>::template CoopTile<N>;
  using CoopThread = typename PlatformTraits<NvidiaPlatform>::CoopThread;
  using CoopWarp = typename PlatformTraits<NvidiaPlatform>::CoopWarp;
  using CoopTileSpan = typename PlatformTraits<NvidiaPlatform>::CoopTileSpan;
  using CoopLanes = typename PlatformTraits<NvidiaPlatform>::CoopLanes;
  using CoopAny = typename PlatformTraits<NvidiaPlatform>::CoopAny;

  // ---- Barrier handles ----
  struct IntraBarrierHandle {
    int nBarriers;
  };
  struct InterBarrierHandle {
    int placeholder;
  };

  // ---- DescriptorSmem: empty for NVSHMEM ----
  struct DescriptorSmem {};

  // ---- Barrier alias ----
  template <typename Tag, typename Coop>
  using Barrier = ::Barrier<NvshmemBackend, Tag, Coop>;

  // ---- Net ----
  struct Net {
    Comm _dc;

    FLAGCX_HOST_DEVICE_INLINE
    Net(const Comm &dc, int /*contextIndex*/) : _dc(dc) {}

    FLAGCX_DEVICE_INLINE_DECORATOR bool isValid() const {
      if (_dc.signalCount > 0 && _dc.signalBuffer == nullptr)
        return false;
      if (_dc.counterCount > 0 && _dc.counterBuffer == nullptr)
        return false;
      return true;
    }

    // ---- Helper: resolve PE from team + peer index ----
    FLAGCX_DEVICE_INLINE_DECORATOR int resolvePE(Team team, int peer) const {
      // team.rank is my rank within team; peer is absolute rank
      // For NVSHMEM, peer IS the PE number (world-scope)
      return peer;
    }

    // ---- One-sided: put ----
    template <typename RA, typename LA, typename Coop, typename Desc>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    put(Team team, int peer, Window dst, size_t dstOff, Window src,
        size_t srcOff, size_t bytes, RA ra, LA la, Coop coop, Desc desc,
        flagcxDeviceScope_t ar, flagcxDeviceScope_t es) const {
      (void)team;
      (void)desc;
      (void)ar;
      (void)es;
      coop.sync();
      if (coop.threadRank() == 0) {
        void *dstPtr = (char *)dst.symBase + dstOff;
        void *srcPtr = (char *)src.rawPtr + srcOff;
        int pe = peer;
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
      (void)team;
      (void)desc;
      (void)ar;
      (void)es;
      coop.sync();
      if (coop.threadRank() == 0) {
        void *dstPtr = (char *)dst.symBase + dstOff;
        nvshmem_putmem(dstPtr, (const void *)&value, sizeof(T), peer);
        signalImpl(peer, ra);
      }
      coop.sync();
    }

    // ---- One-sided: signal ----
    template <typename RA, typename Coop, typename Desc>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    signal(Team team, int peer, RA ra, Coop coop, Desc desc,
           flagcxDeviceScope_t ar, flagcxDeviceScope_t es) const {
      (void)team;
      (void)desc;
      (void)ar;
      (void)es;
      coop.sync();
      if (coop.threadRank() == 0) {
        signalImpl(peer, ra);
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
          nvshmem_quiet();
        coop.sync();
      } else {
        nvshmem_fence();
      }
    }

    // ---- Wait: waitSignal ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitSignal(Coop coop, flagcxDevNetSignal_t signalId, uint64_t least,
               int bits, flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      (void)order;
      coop.sync();
      if (coop.threadRank() == 0) {
        uint64_t *addr = _dc.signalBuffer + (int)signalId;
        nvshmem_uint64_wait_until(addr, NVSHMEM_CMP_GE, least);
      }
      coop.sync();
    }

    // ---- Wait: waitSignalMeetShadow ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitSignalMeetShadow(Coop coop, flagcxDevNetSignal_t signalId, int bits,
                         flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      (void)order;
      coop.sync();
      if (coop.threadRank() == 0) {
        uint64_t target = _dc.shadowBuffer[(int)signalId];
        uint64_t *addr = _dc.signalBuffer + (int)signalId;
        nvshmem_uint64_wait_until(addr, NVSHMEM_CMP_GE, target);
      }
      coop.sync();
    }

    // ---- Wait: waitSignalFollowShadow ----
    template <typename Coop, typename Uint>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitSignalFollowShadow(Coop coop, flagcxDevNetSignal_t signalId,
                           Uint leastDelta, Uint *before, Uint *delta, int bits,
                           flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      (void)order;
      coop.sync();
      if (coop.threadRank() == 0) {
        uint64_t shadow = _dc.shadowBuffer[(int)signalId];
        uint64_t target = shadow + (uint64_t)leastDelta;
        uint64_t *addr = _dc.signalBuffer + (int)signalId;
        nvshmem_uint64_wait_until(addr, NVSHMEM_CMP_GE, target);
        uint64_t cur = Atomic::load(addr, flagcxDeviceMemoryOrderAcquire);
        if (before)
          *before = (Uint)shadow;
        if (delta)
          *delta = (Uint)(cur - shadow);
      }
      coop.sync();
    }

    // ---- Shadow access ----
    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t *
    getSignalShadowPtr(flagcxDevNetSignal_t signalId) const {
      return &_dc.shadowBuffer[(int)signalId];
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    increaseSignalShadow(flagcxDevNetSignal_t signalId, uint64_t delta) const {
      _dc.shadowBuffer[(int)signalId] += delta;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
    readSignal(flagcxDevNetSignal_t signalId, int bits,
               flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      (void)order;
      return Atomic::load(&_dc.signalBuffer[(int)signalId],
                          flagcxDeviceMemoryOrderAcquire);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    resetSignal(flagcxDevNetSignal_t signalId) const {
      Atomic::store(&_dc.signalBuffer[(int)signalId], (uint64_t)0,
                    flagcxDeviceMemoryOrderRelease);
    }

    // ---- Counter: waitCounter ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitCounter(Coop coop, flagcxDevNetCounter_t counterId, uint64_t least,
                int bits, flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      (void)order;
      coop.sync();
      if (coop.threadRank() == 0) {
        int idx = (int)counterId;
        int iter = 0;
        while (Atomic::load(&_dc.counterBuffer[idx],
                            flagcxDeviceMemoryOrderAcquire) < least) {
          Intrin::spinBackoff(iter++);
        }
      }
      coop.sync();
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
    readCounter(flagcxDevNetCounter_t counterId, int bits,
                flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      (void)order;
      return Atomic::load(&_dc.counterBuffer[(int)counterId],
                          flagcxDeviceMemoryOrderAcquire);
    }

    // ---- Two-sided: send/recv/term/wait (NVSHMEM uses one-sided, these are
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
      (void)team;
      coop.sync();
      if (coop.threadRank() == 0) {
        void *dstPtr = (char *)dst.rawPtr + dstOff;
        void *srcPtr = (char *)src.symBase + srcOff;
        nvshmem_getmem(dstPtr, srcPtr, bytes, peer);
      }
      coop.sync();
    }

  private:
    // ---- put dispatch: select fused put+signal vs plain put ----
    template <typename LA>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    putImpl(void *dst, void *src, size_t bytes, int pe,
            flagcxDevNet_SignalInc ra, LA la) const {
      uint64_t *sigAddr = _dc.signalBuffer + (int)ra.signal;
      nvshmem_putmem_signal(dst, src, bytes, sigAddr, 1, NVSHMEM_SIGNAL_ADD,
                            pe);
      counterImpl(la);
    }

    template <typename LA>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    putImpl(void *dst, void *src, size_t bytes, int pe,
            flagcxDevNet_SignalAdd ra, LA la) const {
      uint64_t *sigAddr = _dc.signalBuffer + (int)ra.signal;
      nvshmem_putmem_signal(dst, src, bytes, sigAddr, ra.value,
                            NVSHMEM_SIGNAL_ADD, pe);
      counterImpl(la);
    }

    template <typename LA>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    putImpl(void *dst, void *src, size_t bytes, int pe,
            flagcxDevNet_CounterInc c, LA) const {
      nvshmem_putmem(dst, src, bytes, pe);
      counterImpl(c);
    }

    template <typename RA, typename LA>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    putImpl(void *dst, void *src, size_t bytes, int pe, RA, LA la) const {
      nvshmem_putmem(dst, src, bytes, pe);
      counterImpl(la);
    }

    // ---- signal dispatch ----
    FLAGCX_DEVICE_INLINE_DECORATOR void
    signalImpl(int pe, flagcxDevNet_SignalInc ra) const {
      uint64_t *sigAddr = _dc.signalBuffer + (int)ra.signal;
      nvshmemx_signal_op(sigAddr, 1, NVSHMEM_SIGNAL_ADD, pe);
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void
    signalImpl(int pe, flagcxDevNet_SignalAdd ra) const {
      uint64_t *sigAddr = _dc.signalBuffer + (int)ra.signal;
      nvshmemx_signal_op(sigAddr, ra.value, NVSHMEM_SIGNAL_ADD, pe);
    }
    template <typename RA>
    FLAGCX_DEVICE_INLINE_DECORATOR void signalImpl(int, RA) const {}

    // ---- counter helper ----
    FLAGCX_DEVICE_INLINE_DECORATOR void
    counterImpl(flagcxDevNet_CounterInc c) const {
      Atomic::fetchAdd(&_dc.counterBuffer[(int)c.counter], (uint64_t)1,
                       flagcxDeviceMemoryOrderRelease);
    }
    template <typename LA>
    FLAGCX_DEVICE_INLINE_DECORATOR void counterImpl(LA) const {}

  public:
    // ---- reset counter ----
    FLAGCX_DEVICE_INLINE_DECORATOR void
    resetCounter(flagcxDevNetCounter_t counterId) const {
      int idx = (int)counterId;
      Atomic::store(&_dc.counterBuffer[idx], (uint64_t)0,
                    flagcxDeviceMemoryOrderRelease);
    }
  }; // struct Net
};   // struct CommTraits<NvshmemBackend>

// ============================================================
// Barrier specializations for NvshmemBackend
// Signal-based split-phase barriers using nvshmemx_signal_op.
// ============================================================

// ---- Barrier<NvshmemBackend, flagcxTeamTagIntra, Coop> ----
template <typename Coop>
struct Barrier<NvshmemBackend, flagcxTeamTagIntra, Coop> {
  using Intrin = PlatformTraits<NvidiaPlatform>::Intrin;
  using Atomic = PlatformTraits<NvidiaPlatform>::Atomic;
  using Comm = CommTraits<NvshmemBackend>::Comm;
  using Team = CommTraits<NvshmemBackend>::Team;
  using Multimem = CommTraits<NvshmemBackend>::Multimem;

  Coop _coop;
  nvshmem_team_t _team;
  int _teamSize, _teamRank;
  uint64_t *_barrierSignals;
  uint64_t *_usageCount;

  FLAGCX_DEVICE_INLINE_DECORATOR Barrier()
      : _coop(), _team(NVSHMEM_TEAM_INVALID), _teamSize(0), _teamRank(0),
        _barrierSignals(nullptr), _usageCount(nullptr) {}

  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, const Comm &dc, Team team, uint32_t index, bool = false,
          const Multimem & = {})
      : _coop(coop), _team(dc.intraTeam), _teamSize(dc.intraSize),
        _teamRank(dc.intraRank),
        _barrierSignals(dc.intraBarrierSignals + index * dc.intraSize),
        _usageCount(&dc.barrierUsage[index]) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    _coop.sync();
    if (_coop.threadRank() == 0)
      nvshmem_fence();
    _coop.sync();
    for (int i = _coop.threadRank(); i < _teamSize - 1; i += _coop.size()) {
      int peer = 1 + _teamRank + i;
      if (peer >= _teamSize)
        peer -= _teamSize;
      int peerPE = nvshmem_team_translate_pe(_team, peer, NVSHMEM_TEAM_WORLD);
      nvshmemx_signal_op(_barrierSignals + _teamRank, 1, NVSHMEM_SIGNAL_ADD,
                         peerPE);
    }
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    uint64_t target = *_usageCount + 1;
    for (int i = _coop.threadRank(); i < _teamSize - 1; i += _coop.size()) {
      int peer = 1 + _teamRank + i;
      if (peer >= _teamSize)
        peer -= _teamSize;
      nvshmem_uint64_wait_until(_barrierSignals + peer, NVSHMEM_CMP_GE, target);
    }
    _coop.sync();
    if (_coop.threadRank() == 0)
      *_usageCount = target;
    _coop.sync();
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    arrive(order);
    wait(order);
  }
};

// ---- Barrier<NvshmemBackend, flagcxTeamTagInter, Coop> ----
template <typename Coop>
struct Barrier<NvshmemBackend, flagcxTeamTagInter, Coop> {
  using Intrin = PlatformTraits<NvidiaPlatform>::Intrin;
  using Atomic = PlatformTraits<NvidiaPlatform>::Atomic;
  using Comm = CommTraits<NvshmemBackend>::Comm;
  using Team = CommTraits<NvshmemBackend>::Team;
  using Net = CommTraits<NvshmemBackend>::Net;
  using Multimem = CommTraits<NvshmemBackend>::Multimem;

  Coop _coop;
  nvshmem_team_t _team;
  int _teamSize, _teamRank;
  uint64_t *_barrierSignals;
  uint64_t *_usageCount;

  FLAGCX_DEVICE_INLINE_DECORATOR Barrier()
      : _coop(), _team(NVSHMEM_TEAM_INVALID), _teamSize(0), _teamRank(0),
        _barrierSignals(nullptr), _usageCount(nullptr) {}

  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, const Net &, const Comm &dc, Team, uint32_t index, int = 0)
      : _coop(coop), _team(dc.interTeam),
        _teamSize((dc.intraSize > 0) ? dc.nRanks / dc.intraSize : 1),
        _teamRank((dc.intraSize > 0) ? dc.rank / dc.intraSize : 0),
        _barrierSignals(
            dc.interBarrierSignals +
            index * ((dc.intraSize > 0) ? dc.nRanks / dc.intraSize : 1)),
        _usageCount(&dc.barrierUsage[dc.intraBarrierCount + index]) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
         flagcxDevNetFenceLevel = flagcxDevNetFenceLevel::Relaxed) {
    _coop.sync();
    if (_coop.threadRank() == 0)
      nvshmem_fence();
    _coop.sync();
    for (int i = _coop.threadRank(); i < _teamSize - 1; i += _coop.size()) {
      int peer = 1 + _teamRank + i;
      if (peer >= _teamSize)
        peer -= _teamSize;
      int peerPE = nvshmem_team_translate_pe(_team, peer, NVSHMEM_TEAM_WORLD);
      nvshmemx_signal_op(_barrierSignals + _teamRank, 1, NVSHMEM_SIGNAL_ADD,
                         peerPE);
    }
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxDevNetFenceLevel = flagcxDevNetFenceLevel::Relaxed) {
    uint64_t target = *_usageCount + 1;
    for (int i = _coop.threadRank(); i < _teamSize - 1; i += _coop.size()) {
      int peer = 1 + _teamRank + i;
      if (peer >= _teamSize)
        peer -= _teamSize;
      nvshmem_uint64_wait_until(_barrierSignals + peer, NVSHMEM_CMP_GE, target);
    }
    _coop.sync();
    if (_coop.threadRank() == 0)
      *_usageCount = target;
    _coop.sync();
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    arrive(order, fence);
    wait(order, fence);
  }
};

// ---- Barrier<NvshmemBackend, flagcxTeamTagWorld, Coop> ----
template <typename Coop>
struct Barrier<NvshmemBackend, flagcxTeamTagWorld, Coop> {
  using Intrin = PlatformTraits<NvidiaPlatform>::Intrin;
  using Atomic = PlatformTraits<NvidiaPlatform>::Atomic;
  using Comm = CommTraits<NvshmemBackend>::Comm;
  using Team = CommTraits<NvshmemBackend>::Team;
  using Net = CommTraits<NvshmemBackend>::Net;
  using Multimem = CommTraits<NvshmemBackend>::Multimem;

  Coop _coop;
  Barrier<NvshmemBackend, flagcxTeamTagIntra, Coop> _intra;
  Barrier<NvshmemBackend, flagcxTeamTagInter, Coop> _inter;
  int _nInterPeers;

  FLAGCX_DEVICE_INLINE_DECORATOR Barrier()
      : _coop(), _intra(), _inter(), _nInterPeers(0) {}

  // World barrier (intra + inter)
  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, flagcxTeamTagWorld, const Net &net, const Comm &dc,
          uint32_t index, bool multimem, int nInterPeers)
      : _coop(coop), _intra(coop, dc, Team{}, index),
        _inter(coop, net, dc, Team{}, index, nInterPeers),
        _nInterPeers(nInterPeers) {}

  // Intra-only barrier
  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, flagcxTeamTagIntra, const Net &, const Comm &dc,
          uint32_t index, bool, int)
      : _coop(coop), _intra(coop, dc, Team{}, index), _inter(),
        _nInterPeers(0) {}

  // Inter-only barrier
  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, flagcxTeamTagInter, const Net &net, const Comm &dc,
          uint32_t index, bool, int nInterPeers)
      : _coop(coop), _intra(),
        _inter(coop, net, dc, Team{}, index, nInterPeers),
        _nInterPeers(nInterPeers) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
         flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    if (_nInterPeers > 0) {
      _intra.arrive(flagcxDeviceMemoryOrderRelease);
      _intra.wait(flagcxDeviceMemoryOrderRelease);
      _inter.arrive(order);
    } else {
      _intra.arrive(order);
    }
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    if (_nInterPeers > 0) {
      _inter.wait(order);
      _intra.arrive(flagcxDeviceMemoryOrderAcquire);
      _intra.wait(flagcxDeviceMemoryOrderAcquire);
    } else {
      _intra.wait(order);
    }
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    if (_nInterPeers > 0) {
      _intra.arrive(flagcxDeviceMemoryOrderRelease);
      _intra.wait(flagcxDeviceMemoryOrderRelease);
      _inter.arrive(order);
      _inter.wait(order);
      _intra.arrive(flagcxDeviceMemoryOrderAcquire);
      _intra.wait(flagcxDeviceMemoryOrderAcquire);
    } else {
      _intra.arrive(order);
      _intra.wait(order);
    }
  }
};

#endif // FLAGCX_NVSHMEM_COMM_TRAITS_H_
