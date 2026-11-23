/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * ACLSHMEM CommTraits — device-side backend over cann-shmem (910B/910C).
 *
 * The transport is the one cann-shmem validated on this hardware: MTE
 * engines move bytes into a peer's symmetric heap, and each PE polls a
 * local flag.  Direct scalar access to peer memory is not used (it freezes
 * the AI core; r285 / ascend.sh), and neither is the cann-shmem signal API
 * (int32-only).  Signals here follow the XSHMEM per-source ticket scheme:
 * every source keeps a private sent-ticket slot per destination and writes
 * that running total into its own receive slot inside the destination.
 *
 * Address spaces: bisheng refuses every plain -> __gm__ conversion inside
 * an aicore function (probe Q7/Q9, 2026-11-23), so the state pointers in
 * Window/Comm carry __gm__ end to end on the device pass.  The shared IR
 * still needs a plain local pointer (getLocalPointer), so Window keeps a
 * second plain copy of the same address.  The macro is empty on every
 * other pass, which keeps host/device layouts identical.
 *
 * Ordering: the only measured ordered channel on this soc is asc_dcci +
 * __gm__ 32-bit words (ascend_platform_traits.h).  All state words are
 * 32-bit and every load/store below spells that protocol verbatim.
 ************************************************************************/

#ifndef FLAGCX_ACLSHMEM_COMM_TRAITS_H_
#define FLAGCX_ACLSHMEM_COMM_TRAITS_H_

#include "aclshmem_state_layout.h"
#include "flagcx_kernel_core.h"
#include "platform_traits.h"

#include <shmem.h>

#if defined(__CCE_IS_AICORE__)
#define FLAGCX_ACLSHMEM_GM __gm__
#else
#define FLAGCX_ACLSHMEM_GM
#endif

struct AclshmemBackend {};

// ---- The measured 32-bit ordered channel (see CceAtomic for the record) ----
FLAGCX_DEVICE_INLINE_DECORATOR uint32_t
aclshmemGmLoad32(const FLAGCX_ACLSHMEM_GM volatile uint32_t *p) {
#if defined(__CCE_IS_AICORE__)
  asc_dcci_entire_all();
#endif
  return *p;
}

FLAGCX_DEVICE_INLINE_DECORATOR void
aclshmemGmStore32(FLAGCX_ACLSHMEM_GM volatile uint32_t *p, uint32_t v) {
  *p = v;
#if defined(__CCE_IS_AICORE__)
  asc_dcci_entire_out();
#endif
}

// ---- cann-shmem device entry points, host-pass shims ----
// The declarations are device-pass only in the vendor header, and no host
// code ever executes these bodies.
FLAGCX_DEVICE_INLINE_DECORATOR void
aclshmemPutMemNbi(FLAGCX_ACLSHMEM_GM void *dst, FLAGCX_ACLSHMEM_GM void *src,
                  uint32_t bytes, int pe) {
#if defined(__CCE_IS_AICORE__)
  // shortcut: the vendor entry takes a 32-bit length; the IR suite moves
  // at most a few MiB per call.  Widen when a caller can exceed 4 GiB.
  aclshmem_putmem_nbi(dst, src, bytes, (int32_t)pe);
#else
  (void)dst;
  (void)src;
  (void)bytes;
  (void)pe;
#endif
}

FLAGCX_DEVICE_INLINE_DECORATOR void
aclshmemGetMem(FLAGCX_ACLSHMEM_GM void *dst, FLAGCX_ACLSHMEM_GM void *src,
               uint32_t bytes, int pe) {
#if defined(__CCE_IS_AICORE__)
  aclshmem_getmem(dst, src, bytes, (int32_t)pe);
#else
  (void)dst;
  (void)src;
  (void)bytes;
  (void)pe;
#endif
}

FLAGCX_DEVICE_INLINE_DECORATOR void aclshmemQuiet() {
#if defined(__CCE_IS_AICORE__)
  aclshmem_quiet();
#endif
}

FLAGCX_DEVICE_INLINE_DECORATOR void
flagcxAclshmemRequireRelaxedFence(flagcxDevNetFenceLevel fence) {
  if (fence != flagcxDevNetFenceLevel::Relaxed)
    FLAGCX_DEVICE_TRAP();
}

template <>
struct CommTraits<AclshmemBackend> {
  using Intrin = PlatformTraits<AscendPlatform>::Intrin;
  using Atomic = PlatformTraits<AscendPlatform>::Atomic;
  using CoopOps = PlatformTraits<AscendPlatform>::CoopOps;
  using Clock = PlatformTraits<AscendPlatform>::Clock;

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
    FLAGCX_ACLSHMEM_GM void *symBase;
    size_t allocSize;
    FLAGCX_ACLSHMEM_GM void *rawPtr;
    void *localPtr; // plain copy of rawPtr for the shared IR's local path

    // Peer access is MTE-only: a peer address is never published, so the
    // unified dispatch falls through to the Net path.
    FLAGCX_DEVICE_INLINE_DECORATOR void *
    getPeerPointer(size_t, const Team &, int) const {
      return nullptr;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void *
    getPeerPointer(size_t offset, const Team &team, int peer,
                   flagcxDevPeerAccess_t) const {
      return getPeerPointer(offset, team, peer);
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void *getLocalPointer(size_t offset) const {
      return (char *)localPtr + offset;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void *getIntraPointer(size_t, int) const {
      return nullptr;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void *
    getIntraPointer(size_t offset, int peer, flagcxDevPeerAccess_t) const {
      return getIntraPointer(offset, peer);
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void *
    getMulticastPointer(size_t, const Multimem &) const {
      return nullptr;
    }
    FLAGCX_HOST_DEVICE_INLINE void *getRawPtr() const { return localPtr; }
    FLAGCX_HOST_DEVICE_INLINE bool hasAccess() const {
      return localPtr != nullptr;
    }
    FLAGCX_HOST_DEVICE_INLINE void **getDevPeerPtrs() const { return nullptr; }
    FLAGCX_HOST_DEVICE_INLINE int getMrIndex() const { return 0; }
    FLAGCX_DEVICE_INLINE_DECORATOR bool operator==(const Window &o) const {
      return localPtr == o.localPtr;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR bool operator!=(const Window &o) const {
      return !(*this == o);
    }
  };

  // ---- Comm ----
  // Field order must match flagcxShmemCommInternal exactly: the core casts
  // the adaptor handle onto this type.
  struct Comm {
    int rank, nRanks;
    int intraRank, intraSize;

    FLAGCX_ACLSHMEM_GM uint64_t *signalBuffer; // symmetric heap, 32-bit words
    int signalCount;
    FLAGCX_ACLSHMEM_GM uint64_t *counterBuffer; // local device memory
    int counterCount;
    FLAGCX_ACLSHMEM_GM uint64_t *shadowBuffer; // local device memory

    FLAGCX_ACLSHMEM_GM uint64_t *gridSyncState; // symmetric heap, 3 kinds
    FLAGCX_ACLSHMEM_GM uint64_t *scratchBuffer; // symmetric putValue source

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
    FLAGCX_DEVICE_INLINE_DECORATOR bool p2pSignalSupport(int) const {
      return true;
    }
    // Signals are delivered by writing the peer's symmetric state; no peer
    // pointer to that state is published.
    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t *getSignalPeerPtr(int) const {
      return nullptr;
    }
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
      dc.rank = di.rank;
      dc.nRanks = di.nRanks;
      dc.intraRank = di.intraRank;
      dc.intraSize = di.intraSize;
      dc.signalBuffer = di.signalBuffer;
      dc.signalCount = di.signalCount;
      dc.counterBuffer = di.counterBuffer;
      dc.counterCount = di.counterCount;
      dc.shadowBuffer = di.shadowBuffer;
      dc.gridSyncState = nullptr;
      dc.scratchBuffer = nullptr;
    }
  };

  // ---- Coop types ----
  using CoopBlock = typename PlatformTraits<AscendPlatform>::CoopBlock;
  template <int N>
  using CoopTile =
      typename PlatformTraits<AscendPlatform>::template CoopTile<N>;
  using CoopThread = typename PlatformTraits<AscendPlatform>::CoopThread;
  using CoopWarp = typename PlatformTraits<AscendPlatform>::CoopWarp;
  using CoopTileSpan =
      typename PlatformTraits<AscendPlatform>::CoopTileSpan;
  using CoopLanes = typename PlatformTraits<AscendPlatform>::CoopLanes;
  using CoopAny = typename PlatformTraits<AscendPlatform>::CoopAny;

  // ---- Barrier handles ----
  struct IntraBarrierHandle {
    int nBarriers;
  };
  struct InterBarrierHandle {
    int placeholder;
  };

  struct DescriptorSmem {};

  template <typename Tag, typename Coop>
  using Barrier = ::Barrier<AclshmemBackend, Tag, Coop>;

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
      if (_dc.signalCount > 0 && _dc.shadowBuffer == nullptr)
        return false;
      return _dc.gridSyncState != nullptr;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR int getContextId() const {
      return _contextId;
    }

    static FLAGCX_DEVICE_INLINE_DECORATOR int
    resolvePE(const Comm &dc, Team team, int peer) {
      int base = dc.rank - team.rank * team.stride;
      return base + peer * team.stride;
    }

    // ---- One-sided: put ----
    template <typename RA, typename LA, typename Coop, typename Desc>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    put(Team team, int peer, Window dst, size_t dstOff, Window src,
        size_t srcOff, size_t bytes, RA ra, LA la, Coop coop, Desc desc,
        flagcxDeviceScope_t, flagcxDeviceScope_t) const {
      (void)desc;
      coop.sync();
      if (coop.threadRank() == 0) {
        int pe = resolvePE(_dc, team, peer);
        putImpl((FLAGCX_ACLSHMEM_GM char *)dst.symBase + dstOff,
                (FLAGCX_ACLSHMEM_GM char *)src.rawPtr + srcOff, bytes, pe, ra,
                la);
      }
      coop.sync();
    }

    // ---- One-sided: putValue ----
    template <typename T, typename RA, typename Coop, typename Desc>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    putValue(Team team, int peer, Window dst, size_t dstOff, T value, RA ra,
             Coop coop, Desc, flagcxDeviceScope_t, flagcxDeviceScope_t) const {
      coop.sync();
      if (coop.threadRank() == 0) {
        int pe = resolvePE(_dc, team, peer);
        // MTE sources must be GM; the value arrives in a register.
        FLAGCX_ACLSHMEM_GM volatile uint32_t *scratch =
            (FLAGCX_ACLSHMEM_GM volatile uint32_t *)_dc.scratchBuffer +
            2 * _contextId;
        uint64_t v = (uint64_t)value;
        aclshmemGmStore32(scratch, (uint32_t)v);
        aclshmemGmStore32(scratch + 1, (uint32_t)(v >> 32));
        aclshmemPutMemNbi((FLAGCX_ACLSHMEM_GM char *)dst.symBase + dstOff,
                          (FLAGCX_ACLSHMEM_GM void *)scratch, 8u, pe);
        // The scratch word is reused by the next call: the transfer must
        // have read it before it can be overwritten.
        aclshmemQuiet();
        remoteActionImpl(pe, ra);
      }
      coop.sync();
    }

    // ---- One-sided: signal ----
    template <typename RA, typename Coop, typename Desc>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    signal(Team team, int peer, RA ra, Coop coop, Desc, flagcxDeviceScope_t,
           flagcxDeviceScope_t) const {
      coop.sync();
      if (coop.threadRank() == 0) {
        int pe = resolvePE(_dc, team, peer);
        remoteActionImpl(pe, ra);
      }
      coop.sync();
    }

    // ---- One-sided: get (T2-era surface; the r285 hang was a WRITE to a
    // peer VA, a symmetric-heap read goes through cann-shmem's translator) ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    get(Team team, int peer, Window src, size_t srcOff, Window dst,
        size_t dstOff, size_t bytes, Coop coop) const {
      coop.sync();
      if (coop.threadRank() == 0) {
        int pe = resolvePE(_dc, team, peer);
        aclshmemGetMem((FLAGCX_ACLSHMEM_GM char *)dst.rawPtr + dstOff,
                       (FLAGCX_ACLSHMEM_GM char *)src.symBase + srcOff, bytes,
                       pe);
      }
      coop.sync();
    }

    // ---- Ordering: flush ----
    // Stronger than the caller may ask for: quiet() drains this PE's MTE
    // queue, which is the only way a Relaxed flush can guarantee that the
    // data written before it has left the core.
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    flush(Coop coop, flagcxDeviceMemoryOrder_t) const {
      coop.sync();
      if (coop.threadRank() == 0)
        aclshmemQuiet();
      coop.sync();
    }

    // ---- Wait: waitSignal ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitSignal(Coop coop, flagcxDevSignal_t signalId, uint64_t least, int,
               flagcxDeviceMemoryOrder_t) const {
      coop.sync();
      if (coop.threadRank() == 0)
        spinSignal(signalId, least);
      coop.sync();
    }

    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitSignalMeetShadow(Coop coop, flagcxDevSignal_t signalId, int,
                         flagcxDeviceMemoryOrder_t) const {
      coop.sync();
      if (coop.threadRank() == 0)
        spinSignal(signalId, (uint64_t)aclshmemGmLoad32(shadowWord(signalId)));
      coop.sync();
    }

    template <typename Coop, typename Uint>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitSignalFollowShadow(Coop coop, flagcxDevSignal_t signalId,
                           Uint leastDelta, Uint *before, Uint *delta, int,
                           flagcxDeviceMemoryOrder_t) const {
      coop.sync();
      if (coop.threadRank() == 0) {
        uint64_t shadow =
            (uint64_t)aclshmemGmLoad32(shadowWord(signalId));
        uint64_t current = spinSignal(signalId, shadow + (uint64_t)leastDelta);
        if (before)
          *before = (Uint)shadow;
        if (delta)
          *delta = (Uint)(current - shadow);
      }
      coop.sync();
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    increaseSignalShadow(flagcxDevSignal_t signalId, uint64_t delta) const {
      FLAGCX_ACLSHMEM_GM volatile uint32_t *p = shadowWord(signalId);
      aclshmemGmStore32(p, aclshmemGmLoad32(p) + (uint32_t)delta);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
    readSignal(flagcxDevSignal_t signalId, int,
               flagcxDeviceMemoryOrder_t) const {
      return (uint64_t)signalValue(signalId);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    resetSignal(flagcxDevSignal_t signalId) const {
      aclshmemGmStore32(baselineWord(signalId), sumSignalRemote(signalId));
      aclshmemGmStore32(shadowWord(signalId), 0u);
    }

    // ---- Counters: one local 32-bit word per slot ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitCounter(Coop coop, flagcxDevCounter_t counterId, uint64_t least, int,
                flagcxDeviceMemoryOrder_t) const {
      coop.sync();
      if (coop.threadRank() == 0) {
        FLAGCX_ACLSHMEM_GM volatile uint32_t *p = counterWord(counterId);
        int iter = 0;
        while (aclshmemGmLoad32(p) < (uint32_t)least)
          Intrin::spinBackoff(iter++);
      }
      coop.sync();
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
    readCounter(flagcxDevCounter_t counterId, int,
                flagcxDeviceMemoryOrder_t) const {
      return (uint64_t)aclshmemGmLoad32(counterWord(counterId));
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    resetCounter(flagcxDevCounter_t counterId) const {
      aclshmemGmStore32(counterWord(counterId), 0u);
    }

    // Selected by the IR's SFINAE probe in preference to Atomic::fetchAdd,
    // which this soc declares absent.
    FLAGCX_DEVICE_INLINE_DECORATOR void
    completeCounter(flagcxDevCounter_t counterId, uint64_t delta) const {
      FLAGCX_ACLSHMEM_GM volatile uint32_t *p = counterWord(counterId);
      aclshmemGmStore32(p, aclshmemGmLoad32(p) + (uint32_t)delta);
    }

    // ---- Two-sided stubs (one-sided backend) ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t
    send(Coop, Window, size_t, size_t, flagcxDataType_t, int) const {
      return flagcxSuccess;
    }
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t
    recv(Coop, Window, size_t, size_t, flagcxDataType_t, int) const {
      return flagcxSuccess;
    }
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t term(Coop) const {
      return flagcxSuccess;
    }
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t wait(Coop) const {
      return flagcxSuccess;
    }

  private:
    // Base 32-bit word of one (context, signal) ticket block.
    FLAGCX_DEVICE_INLINE_DECORATOR FLAGCX_ACLSHMEM_GM uint32_t *
    signalWords(flagcxDevSignal_t signalId) const {
      size_t word = ((size_t)_contextId * (size_t)_dc.signalCount +
                     (size_t)signalId) *
                    (size_t)FLAGCX_ACLSHMEM_SIGNAL_SLOTS;
      return (FLAGCX_ACLSHMEM_GM uint32_t *)_dc.signalBuffer + word;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR FLAGCX_ACLSHMEM_GM volatile uint32_t *
    baselineWord(flagcxDevSignal_t signalId) const {
      return signalWords(signalId) + FLAGCX_ACLSHMEM_SIGNAL_RESET_BASELINE;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR FLAGCX_ACLSHMEM_GM volatile uint32_t *
    shadowWord(flagcxDevSignal_t signalId) const {
      size_t idx =
          (size_t)_contextId * (size_t)_dc.signalCount + (size_t)signalId;
      return (FLAGCX_ACLSHMEM_GM volatile uint32_t *)_dc.shadowBuffer + idx;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR FLAGCX_ACLSHMEM_GM volatile uint32_t *
    counterWord(flagcxDevCounter_t counterId) const {
      size_t idx =
          (size_t)_contextId * (size_t)_dc.counterCount + (size_t)counterId;
      return (FLAGCX_ACLSHMEM_GM volatile uint32_t *)_dc.counterBuffer + idx;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint32_t
    sumSignalRemote(flagcxDevSignal_t signalId) const {
      FLAGCX_ACLSHMEM_GM uint32_t *base = signalWords(signalId);
      uint32_t total = 0;
      for (int src = 0; src < _dc.nRanks; ++src)
        total += aclshmemGmLoad32(base + src);
      return total;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint32_t
    signalValue(flagcxDevSignal_t signalId) const {
      return sumSignalRemote(signalId) -
             aclshmemGmLoad32(baselineWord(signalId));
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
    spinSignal(flagcxDevSignal_t signalId, uint64_t least) const {
      int iter = 0;
      uint64_t current = (uint64_t)signalValue(signalId);
      while (current < least) {
        Intrin::spinBackoff(iter++);
        current = (uint64_t)signalValue(signalId);
      }
      return current;
    }

    // Publish `delta` more to `pe`: send this PE's running ticket for that
    // destination to the peer's slot for this source.
    FLAGCX_DEVICE_INLINE_DECORATOR void
    sendSignalTicket(int pe, flagcxDevSignal_t signalId, uint32_t delta) const {
      FLAGCX_ACLSHMEM_GM uint32_t *base = signalWords(signalId);
      FLAGCX_ACLSHMEM_GM volatile uint32_t *sent =
          base + FLAGCX_ACLSHMEM_MAX_PES + pe;
      uint32_t ticket = aclshmemGmLoad32(sent) + delta;
      aclshmemGmStore32(sent, ticket);
      // dst is this PE's slot at the same offset; cann-shmem translates it
      // to the peer's copy.
      aclshmemPutMemNbi((FLAGCX_ACLSHMEM_GM void *)(base + _dc.rank),
                        (FLAGCX_ACLSHMEM_GM void *)sent, 4u, pe);
    }

    // Data must be in the peer's GM before any ticket: flush this core's
    // dirty lines, transfer, drain.
    FLAGCX_DEVICE_INLINE_DECORATOR void
    putData(FLAGCX_ACLSHMEM_GM void *dst, FLAGCX_ACLSHMEM_GM void *src,
            size_t bytes, int pe) const {
#if defined(__CCE_IS_AICORE__)
      asc_dcci_entire_out();
#endif
      aclshmemPutMemNbi(dst, src, (uint32_t)bytes, pe);
      aclshmemQuiet();
    }

    template <typename RA, typename LA>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    putImpl(FLAGCX_ACLSHMEM_GM void *dst, FLAGCX_ACLSHMEM_GM void *src,
            size_t bytes, int pe, RA ra, LA la) const {
      putData(dst, src, bytes, pe);
      remoteActionImpl(pe, ra);
      localActionImpl(la);
    }

    // Remote actions available at a peer: MTE copies and the ticket word
    // they carry. A remote counter increment would need an RMW, which this
    // soc does not provide, so it stays a hard failure.
    FLAGCX_DEVICE_INLINE_DECORATOR void
    remoteActionImpl(int, flagcxDevNet_None) const {}
    FLAGCX_DEVICE_INLINE_DECORATOR void
    remoteActionImpl(int pe, flagcxDevNet_SignalInc ra) const {
      sendSignalTicket(pe, ra.signal, 1u);
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void
    remoteActionImpl(int pe, flagcxDevNet_SignalAdd ra) const {
      sendSignalTicket(pe, ra.signal, (uint32_t)ra.value);
    }
    FLAGCX_DEVICE_INLINE_DECORATOR void
    remoteActionImpl(int, flagcxDevNet_CounterInc) const {
      FLAGCX_DEVICE_TRAP();
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    localActionImpl(flagcxDevNet_CounterInc c) const {
      completeCounter(c.counter, 1u);
    }
    template <typename LA>
    FLAGCX_DEVICE_INLINE_DECORATOR void localActionImpl(LA) const {}
  }; // struct Net
};   // struct CommTraits<AclshmemBackend>

// ============================================================
// Barriers
//
// One state block per kind in the symmetric heap:
//   [0]                 this PE's local generation
//   [1 + teamRank]      generation this PE published for itself
// A member's slot is written only by that member, so a wait is a plain
// scan of the team's slots.  One generation counter per kind means two
// barrier *indices* of the same kind must not be in flight concurrently -
// the IR suite uses its four indices sequentially (see device_ir.asc).
// ============================================================
template <typename Coop>
struct Barrier<AclshmemBackend, flagcxTeamTagIntra, Coop> {
  using Intrin = PlatformTraits<AscendPlatform>::Intrin;
  using Comm = CommTraits<AclshmemBackend>::Comm;
  using Team = CommTraits<AclshmemBackend>::Team;
  using Multimem = CommTraits<AclshmemBackend>::Multimem;

  Coop _coop;
  FLAGCX_ACLSHMEM_GM uint32_t *_state;
  int _base;
  int _teamSize;
  int _teamRank;
  int _stride;

  FLAGCX_DEVICE_INLINE_DECORATOR Barrier()
      : _coop(), _state(nullptr), _base(0), _teamSize(0), _teamRank(0),
        _stride(1) {}

  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, const Comm &dc, Team team, uint32_t, bool = false,
          const Multimem & = {})
      : _coop(coop),
        _state((FLAGCX_ACLSHMEM_GM uint32_t *)dc.gridSyncState), _base(0),
        _teamSize(team.nRanks), _teamRank(team.rank), _stride(team.stride) {
    _base = dc.rank - team.rank * team.stride;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t = flagcxDeviceMemoryOrderAcqRel) {
    if (_teamSize <= 1)
      return;
    // Earlier writes must reach the peer before its barrier ticket.
    aclshmemQuiet();
    uint32_t gen = aclshmemGmLoad32(_state) + 1u;
    aclshmemGmStore32(_state, gen);
    aclshmemGmStore32(_state + 1 + _teamRank, gen);
    for (int m = 0; m < _teamSize; ++m) {
      if (m == _teamRank)
        continue;
      aclshmemPutMemNbi((FLAGCX_ACLSHMEM_GM void *)(_state + 1 + _teamRank),
                        (FLAGCX_ACLSHMEM_GM void *)(_state + 1 + _teamRank),
                        4u, _base + m * _stride);
    }
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t = flagcxDeviceMemoryOrderAcqRel) {
    if (_teamSize <= 1)
      return;
    uint32_t gen = aclshmemGmLoad32(_state);
    for (int m = 0; m < _teamSize; ++m) {
      int iter = 0;
      while (aclshmemGmLoad32(_state + 1 + m) < gen)
        Intrin::spinBackoff(iter++);
    }
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    arrive(order);
    wait(order);
  }
};

template <typename Coop>
struct Barrier<AclshmemBackend, flagcxTeamTagInter, Coop> {
  using Intrin = PlatformTraits<AscendPlatform>::Intrin;
  using Comm = CommTraits<AclshmemBackend>::Comm;
  using Team = CommTraits<AclshmemBackend>::Team;
  using Net = CommTraits<AclshmemBackend>::Net;
  using Multimem = CommTraits<AclshmemBackend>::Multimem;

  Coop _coop;
  FLAGCX_ACLSHMEM_GM uint32_t *_state;
  int _base;
  int _teamSize;
  int _teamRank;
  int _stride;

  FLAGCX_DEVICE_INLINE_DECORATOR Barrier()
      : _coop(), _state(nullptr), _base(0), _teamSize(0), _teamRank(0),
        _stride(1) {}

  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, const Net &, const Comm &dc, Team team, uint32_t, int = 0)
      : _coop(coop),
        _state((FLAGCX_ACLSHMEM_GM uint32_t *)dc.gridSyncState +
               FLAGCX_ACLSHMEM_BARRIER_SLOTS),
        _base(dc.rank - team.rank * team.stride), _teamSize(team.nRanks),
        _teamRank(team.rank), _stride(team.stride) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t = flagcxDeviceMemoryOrderAcqRel,
         flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    flagcxAclshmemRequireRelaxedFence(fence);
    if (_teamSize <= 1)
      return;
    aclshmemQuiet();
    uint32_t gen = aclshmemGmLoad32(_state) + 1u;
    aclshmemGmStore32(_state, gen);
    aclshmemGmStore32(_state + 1 + _teamRank, gen);
    for (int m = 0; m < _teamSize; ++m) {
      if (m == _teamRank)
        continue;
      aclshmemPutMemNbi((FLAGCX_ACLSHMEM_GM void *)(_state + 1 + _teamRank),
                        (FLAGCX_ACLSHMEM_GM void *)(_state + 1 + _teamRank),
                        4u, _base + m * _stride);
    }
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t = flagcxDeviceMemoryOrderAcqRel,
       flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    flagcxAclshmemRequireRelaxedFence(fence);
    if (_teamSize <= 1)
      return;
    uint32_t gen = aclshmemGmLoad32(_state);
    for (int m = 0; m < _teamSize; ++m) {
      int iter = 0;
      while (aclshmemGmLoad32(_state + 1 + m) < gen)
        Intrin::spinBackoff(iter++);
    }
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    arrive(order, fence);
    wait(order, fence);
  }
};

template <typename Coop>
struct Barrier<AclshmemBackend, flagcxTeamTagWorld, Coop> {
  using Intrin = PlatformTraits<AscendPlatform>::Intrin;
  using Comm = CommTraits<AclshmemBackend>::Comm;
  using Net = CommTraits<AclshmemBackend>::Net;
  using Multimem = CommTraits<AclshmemBackend>::Multimem;

  Coop _coop;
  FLAGCX_ACLSHMEM_GM uint32_t *_state;
  int _base;
  int _teamSize;
  int _teamRank;
  int _stride;

  FLAGCX_DEVICE_INLINE_DECORATOR Barrier()
      : _coop(), _state(nullptr), _base(0), _teamSize(0), _teamRank(0),
        _stride(1) {}

  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, flagcxTeamTagWorld, const Net &, const Comm &dc,
          uint32_t, bool, int)
      : _coop(coop),
        _state((FLAGCX_ACLSHMEM_GM uint32_t *)dc.gridSyncState +
               2 * FLAGCX_ACLSHMEM_BARRIER_SLOTS),
        _base(0), _teamSize(dc.nRanks), _teamRank(dc.rank), _stride(1) {}

  // Intra-only form of the world barrier.
  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, flagcxTeamTagIntra, const Net &, const Comm &dc,
          uint32_t, bool, int)
      : _coop(coop),
        _state((FLAGCX_ACLSHMEM_GM uint32_t *)dc.gridSyncState +
               2 * FLAGCX_ACLSHMEM_BARRIER_SLOTS),
        _base(dc.rank - dc.intraRank), _teamSize(dc.intraSize),
        _teamRank(dc.intraRank), _stride(1) {}

  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, flagcxTeamTagInter, const Net &, const Comm &dc,
          uint32_t, bool, int)
      : _coop(coop),
        _state((FLAGCX_ACLSHMEM_GM uint32_t *)dc.gridSyncState +
               2 * FLAGCX_ACLSHMEM_BARRIER_SLOTS),
        _base(dc.rank), _teamSize(1), _teamRank(0), _stride(1) {
    (void)dc;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t = flagcxDeviceMemoryOrderAcqRel,
         flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    flagcxAclshmemRequireRelaxedFence(fence);
    if (_teamSize <= 1)
      return;
    aclshmemQuiet();
    uint32_t gen = aclshmemGmLoad32(_state) + 1u;
    aclshmemGmStore32(_state, gen);
    aclshmemGmStore32(_state + 1 + _teamRank, gen);
    for (int m = 0; m < _teamSize; ++m) {
      if (m == _teamRank)
        continue;
      aclshmemPutMemNbi((FLAGCX_ACLSHMEM_GM void *)(_state + 1 + _teamRank),
                        (FLAGCX_ACLSHMEM_GM void *)(_state + 1 + _teamRank),
                        4u, _base + m * _stride);
    }
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t = flagcxDeviceMemoryOrderAcqRel,
       flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    flagcxAclshmemRequireRelaxedFence(fence);
    if (_teamSize <= 1)
      return;
    uint32_t gen = aclshmemGmLoad32(_state);
    for (int m = 0; m < _teamSize; ++m) {
      int iter = 0;
      while (aclshmemGmLoad32(_state + 1 + m) < gen)
        Intrin::spinBackoff(iter++);
    }
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    arrive(order, fence);
    wait(order, fence);
  }
};

#endif // FLAGCX_ACLSHMEM_COMM_TRAITS_H_
