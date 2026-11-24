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
#include "gm2gm/engine/shmem_device_mte.hpp"
#include "gm2gm/shmem_device_cc.h"
#else
#include "host/data_plane/shmem_host_p2p_sync.h"
#endif

#if defined(__CCE_IS_AICORE__)
#define FLAGCX_ACLSHMEM_GM __gm__
#else
#define FLAGCX_ACLSHMEM_GM
#endif

struct AclshmemBackend {};

// ---- FFTS base address ----
// A2/A3 kernels must install the FFTS base address before their first
// aclshmem call: the vendor docs list a missing util_set_ffts_config as the
// A2/A3 "barrier 异常" cause (docs/principles/init_finalize.md), and every
// vendor example does it at kernel entry.  The value is host state
// (rtGetC2cCtrlAddr, cached by aclshmem init), so the launcher reads it and
// the kernel installs it - the same two-step util_get/util_set the vendor's
// own on-stream barrier kernel uses.
FLAGCX_DEVICE_INLINE_DECORATOR void aclshmemSetFftsConfig(uint64_t config) {
#if defined(__CCE_IS_AICORE__)
  util_set_ffts_config(config);
  // The vendor puts the same barrier between the FFTS config write and the
  // first aclshmem call (barrier_all_kernel.cpp:19-21). Without it the first
  // transfer of a kernel is occasionally issued before the config takes
  // effect and silently does nothing: measured at np=2 as S18B combo 0
  // landing as 0 on rank 0, 4 of 15 sizes, while combos 1-5 always arrived.
  pipe_barrier(PIPE_ALL);
#else
  (void)config;
#endif
}

// Deliberately not FLAGCX_DEVICE_INLINE_DECORATOR: that expands to
// "[aicore] inline" on BOTH bisheng passes, and a host launcher cannot call
// an aicore function - the first cut of this helper failed exactly there.
inline uint64_t aclshmemHostFftsConfig() {
#if defined(__CCE_IS_AICORE__)
  return 0;
#else
  return util_get_ffts_config();
#endif
}

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

// ---- The 32-bit completion contract ----
// Every completion word this backend publishes is 32 bits
// (aclshmem_state_layout.h), which the trait advertises as completionBits /
// defaultCounterBits = 32; the shared IR passes that width back here.  A
// threshold or delta the 32-bit word cannot represent is rejected, never
// truncated (MC952 review P2).  There is no runtime trap on this pass
// (device_utils.h trap note), so a rejected value is a contract violation:
// the project's noreturn marker is the loudest spelling available.
FLAGCX_DEVICE_INLINE_DECORATOR uint32_t
aclshmemCompletionValue(uint64_t value) {
  if (value > 0xffffffffull)
    FLAGCX_DEVICE_TRAP();
  return (uint32_t)value;
}

FLAGCX_DEVICE_INLINE_DECORATOR void aclshmemRequireCompletionBits(int bits) {
  if (bits != 32)
    FLAGCX_DEVICE_TRAP();
}

// Circular comparison: wrap is well defined while the outstanding distance
// stays below 2^31, the documented range of a 32-bit sequence number.
FLAGCX_DEVICE_INLINE_DECORATOR bool aclshmemCompletionBefore(uint32_t current,
                                                             uint32_t target) {
  return (int)(current - target) < 0;
}

// ---- cann-shmem device entry points, host-pass shims ----
// The declarations are device-pass only in the vendor header, and no host
// code ever executes these bodies.
FLAGCX_DEVICE_INLINE_DECORATOR void
aclshmemPutMemNbi(FLAGCX_ACLSHMEM_GM void *dst, FLAGCX_ACLSHMEM_GM void *src,
                  uint32_t bytes, int pe) {
#if defined(__CCE_IS_AICORE__)
  // A source this core just stored to GM has to be retired before the
  // engine reads it back. The first transfer of a kernel otherwise copies
  // the previous contents: measured at np=2 as putValue combo 0 landing as
  // 0 (the scratch word was written two instructions earlier), 4 of 15
  // sizes, while every later combo arrived. Silent when it happens, so the
  // barrier is paid on every put rather than chased per call site.
  pipe_barrier(PIPE_ALL);
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

FLAGCX_DEVICE_INLINE_DECORATOR void aclshmemQuiet();

FLAGCX_DEVICE_INLINE_DECORATOR void aclshmemGetMem(FLAGCX_ACLSHMEM_GM void *dst,
                                                   FLAGCX_ACLSHMEM_GM void *src,
                                                   uint32_t bytes, int pe) {
#if defined(__CCE_IS_AICORE__)
  aclshmem_getmem(dst, src, bytes, (int32_t)pe);
  // The vendor's get RMA pairs its nbi copy with mte_quiet (see
  // shmem_device_rma.hpp); the bare getmem entry does not promise the
  // destination is complete when it returns, so drain before the caller
  // reads it.
  aclshmemQuiet();
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
  // mte_put_nbi deliberately skips the MTE3 wait on its last (only, for the
  // 4B/8B transfers here) chunk - "Last PIPE Sync Should be done outside" -
  // and the PipeBarrier inside quiet() does not retire that ub2gm. Measured
  // at np=2: a 4B barrier ticket was lost outright (S16 hang), and a batch
  // of 8B putValues surfaced ~200ms after the kernel (S18B late). Large
  // puts never failed because their loop carries an explicit MTE3_MTE2
  // wait. Close the vendor's own event pair with the same sync id so quiet
  // returns only after the outstanding ub2gm has completed.
  FLAGCX_ACLSHMEM_GM aclshmem_device_host_state_t *st = aclshmemi_get_state();
  AscendC::TEventID mteSyncId = (AscendC::TEventID)st->mte_config.sync_id;
  AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(mteSyncId);
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(mteSyncId);
#endif
}

// The peer's address of one symmetric element: reads are the lossless
// direction (see the barrier design comment below), so the barrier wait,
// the putValue read-back and the test kernels' peer scans all go through
// here. Templated only so a caller can index the element type it stored;
// the translation is the same aclshmem_ptr for every type.
template <typename T>
FLAGCX_DEVICE_INLINE_DECORATOR FLAGCX_ACLSHMEM_GM volatile T *
aclshmemPeerWord(FLAGCX_ACLSHMEM_GM T *local, int pe) {
#if defined(__CCE_IS_AICORE__)
  return (FLAGCX_ACLSHMEM_GM volatile T *)aclshmem_ptr(
      (FLAGCX_ACLSHMEM_GM void *)local, pe);
#else
  (void)pe;
  return local;
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

  // Completion words are 32 bits wide end to end (state layout).  Advertised
  // so the shared IR never assumes a 64-bit signal/counter here; the paths
  // below reject values beyond 32 bits instead of truncating them.
  static constexpr int completionBits = 32;
  static constexpr int defaultCounterBits = 32;

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
    FLAGCX_DEVICE_INLINE_DECORATOR void *getPeerPointer(size_t, const Team &,
                                                        int) const {
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
    int barrierCapacity; // granted (kind, index) channels; set at creation

    // Barrier plane for one channel; see the Barriers section comment.  The
    // planes are allocated to the capacity granted at communicator creation,
    // so an index beyond it traps instead of aliasing another channel.
    FLAGCX_DEVICE_INLINE_DECORATOR FLAGCX_ACLSHMEM_GM uint32_t *
    barrierState(int kind, uint32_t index) const {
      if (barrierCapacity <= 0 || index >= (uint32_t)barrierCapacity)
        FLAGCX_DEVICE_TRAP();
      return (FLAGCX_ACLSHMEM_GM uint32_t *)gridSyncState +
             ((size_t)kind * (size_t)barrierCapacity + (size_t)index) *
                 FLAGCX_ACLSHMEM_BARRIER_SLOTS;
    }

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
      dc.signalCount = di.signalCount;
      dc.counterCount = di.counterCount;
#if defined(__CCE_IS_AICORE__)
      // This body is a device function on the bisheng pass even though only
      // the host ever runs it (flagcx_device_core.h:99-104); a handle's
      // plain pointers have no legal conversion to __gm__ there, so the
      // aicore copy is left empty and the state comes from the prebuilt
      // value struct (di.devComm branch) instead.
      dc.signalBuffer = nullptr;
      dc.counterBuffer = nullptr;
      dc.shadowBuffer = nullptr;
#else
      dc.signalBuffer = di.signalBuffer;
      dc.counterBuffer = di.counterBuffer;
      dc.shadowBuffer = di.shadowBuffer;
#endif
      dc.gridSyncState = nullptr;
      dc.scratchBuffer = nullptr;
      dc.barrierCapacity = FLAGCX_ACLSHMEM_BARRIER_INDICES;
    }
  };

  // ---- Coop types ----
  using CoopBlock = typename PlatformTraits<AscendPlatform>::CoopBlock;
  template <int N>
  using CoopTile =
      typename PlatformTraits<AscendPlatform>::template CoopTile<N>;
  using CoopThread = typename PlatformTraits<AscendPlatform>::CoopThread;
  using CoopWarp = typename PlatformTraits<AscendPlatform>::CoopWarp;
  using CoopTileSpan = typename PlatformTraits<AscendPlatform>::CoopTileSpan;
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

    static FLAGCX_DEVICE_INLINE_DECORATOR int resolvePE(const Comm &dc,
                                                        Team team, int peer) {
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
        FLAGCX_ACLSHMEM_GM char *dstPtr =
            (FLAGCX_ACLSHMEM_GM char *)dst.symBase + dstOff;
        aclshmemPutMemNbi(dstPtr, (FLAGCX_ACLSHMEM_GM void *)scratch, 8u, pe);
        // The scratch word is reused by the next call: the transfer must
        // have read it before it can be overwritten.
        aclshmemQuiet();
        // A posted 8B put can be dropped outright and the peer's copy can
        // surface only much later; neither is visible from the sender
        // without a read-back, and putValue is the primitive both measured
        // as S18B's lost combo 0. The destination lives in the symmetric
        // heap, so poll the peer mapping (the one lossless direction) and
        // re-issue the same bytes until they are there.  The 8B value is
        // the whole payload, so observing it at the peer is this call's
        // completion proof; there is no retry budget, because a value that
        // never arrives must not fall through to the remote action (the
        // old code sent the signal anyway - MC952 review P1).
        FLAGCX_ACLSHMEM_GM volatile uint32_t *remoteWord =
            aclshmemPeerWord((FLAGCX_ACLSHMEM_GM uint32_t *)dstPtr, pe);
        for (int spins = 0;; ++spins) {
          uint32_t lo = aclshmemGmLoad32(remoteWord);
          uint32_t hi = aclshmemGmLoad32(remoteWord + 1);
          if (lo == (uint32_t)v && hi == (uint32_t)(v >> 32))
            break;
          if ((spins & 31) == 31) {
            aclshmemPutMemNbi(dstPtr, (FLAGCX_ACLSHMEM_GM void *)scratch, 8u,
                              pe);
            aclshmemQuiet();
          }
          Intrin::spinBackoff(spins < 16 ? spins : 16);
        }
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
    FLAGCX_DEVICE_INLINE_DECORATOR void flush(Coop coop,
                                              flagcxDeviceMemoryOrder_t) const {
      coop.sync();
      if (coop.threadRank() == 0)
        aclshmemQuiet();
      coop.sync();
    }

    // ---- Wait: waitSignal ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitSignal(Coop coop, flagcxDevSignal_t signalId, uint64_t least, int bits,
               flagcxDeviceMemoryOrder_t) const {
      aclshmemRequireCompletionBits(bits);
      uint32_t target = aclshmemCompletionValue(least);
      coop.sync();
      if (coop.threadRank() == 0)
        spinSignal(signalId, target);
      coop.sync();
    }

    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitSignalMeetShadow(Coop coop, flagcxDevSignal_t signalId, int bits,
                         flagcxDeviceMemoryOrder_t) const {
      aclshmemRequireCompletionBits(bits);
      coop.sync();
      if (coop.threadRank() == 0)
        spinSignal(signalId, aclshmemGmLoad32(shadowWord(signalId)));
      coop.sync();
    }

    template <typename Coop, typename Uint>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitSignalFollowShadow(Coop coop, flagcxDevSignal_t signalId,
                           Uint leastDelta, Uint *before, Uint *delta, int bits,
                           flagcxDeviceMemoryOrder_t) const {
      aclshmemRequireCompletionBits(bits);
      coop.sync();
      if (coop.threadRank() == 0) {
        uint32_t shadow = aclshmemGmLoad32(shadowWord(signalId));
        // Range-check the delta alone, then advance in the 32-bit domain:
        // shadow + delta may be a valid sequence-number rollover while the
        // outstanding distance stays below 2^31, so the widened sum must
        // not be validated (MC952 review P2; default_completion.h advance).
        uint32_t target =
            shadow + aclshmemCompletionValue((uint64_t)leastDelta);
        uint32_t current = spinSignal(signalId, target);
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
      aclshmemGmStore32(p,
                        aclshmemGmLoad32(p) + aclshmemCompletionValue(delta));
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t readSignal(
        flagcxDevSignal_t signalId, int bits, flagcxDeviceMemoryOrder_t) const {
      aclshmemRequireCompletionBits(bits);
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
    waitCounter(Coop coop, flagcxDevCounter_t counterId, uint64_t least,
                int bits, flagcxDeviceMemoryOrder_t) const {
      aclshmemRequireCompletionBits(bits);
      uint32_t target = aclshmemCompletionValue(least);
      coop.sync();
      if (coop.threadRank() == 0) {
        FLAGCX_ACLSHMEM_GM volatile uint32_t *p = counterWord(counterId);
        int iter = 0;
        while (aclshmemCompletionBefore(aclshmemGmLoad32(p), target))
          Intrin::spinBackoff(iter++);
      }
      coop.sync();
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
    readCounter(flagcxDevCounter_t counterId, int bits,
                flagcxDeviceMemoryOrder_t) const {
      aclshmemRequireCompletionBits(bits);
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
      aclshmemGmStore32(p,
                        aclshmemGmLoad32(p) + aclshmemCompletionValue(delta));
    }

    // ---- Two-sided stubs (one-sided backend) ----
    // This transport has no two-sided FIFO: the operations are declared
    // absent, as the XSHMEM backend does, so a caller that depends on them
    // sees flagcxNotSupported instead of a success that transmitted
    // nothing (MC952 review P1).
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t send(Coop, Window, size_t,
                                                       size_t, flagcxDataType_t,
                                                       int) const {
      return flagcxNotSupported;
    }
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t recv(Coop, Window, size_t,
                                                       size_t, flagcxDataType_t,
                                                       int) const {
      return flagcxNotSupported;
    }
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t term(Coop) const {
      return flagcxNotSupported;
    }
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t wait(Coop) const {
      return flagcxNotSupported;
    }

  private:
    // Base 32-bit word of one (context, signal) ticket block.
    FLAGCX_DEVICE_INLINE_DECORATOR FLAGCX_ACLSHMEM_GM uint32_t *
    signalWords(flagcxDevSignal_t signalId) const {
      size_t word =
          ((size_t)_contextId * (size_t)_dc.signalCount + (size_t)signalId) *
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

    FLAGCX_DEVICE_INLINE_DECORATOR uint32_t
    spinSignal(flagcxDevSignal_t signalId, uint32_t target) const {
      int iter = 0;
      uint32_t current = signalValue(signalId);
      while (aclshmemCompletionBefore(current, target)) {
        Intrin::spinBackoff(iter++);
        current = signalValue(signalId);
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
      FLAGCX_ACLSHMEM_GM void *dst =
          (FLAGCX_ACLSHMEM_GM void *)(base + _dc.rank);
      aclshmemPutMemNbi(dst, (FLAGCX_ACLSHMEM_GM void *)sent, 4u, pe);
      // A posted 4B put can be dropped outright (measured as the np=2 S16
      // barrier-ticket hang). The ticket is the absolute count, so a
      // re-issue carries the same meaning; poll the peer mapping - the
      // lossless direction - and re-put until this exact ticket is there.
      // The poll also serialises consecutive signals on one slot, which
      // keeps the 5-in-a-row S25 increments from landing out of order.
      //
      // Observing the ticket at the peer is the delivery proof for this
      // signal, and an unconfirmed ticket must never be reported as
      // delivered: there is no runtime trap on this pass (device_utils.h
      // trap note), so a transport that keeps dropping makes the kernel
      // not return instead of letting the caller signal success it did not
      // observe (MC952 review P1).
      FLAGCX_ACLSHMEM_GM volatile uint32_t *remote =
          aclshmemPeerWord((FLAGCX_ACLSHMEM_GM uint32_t *)dst, pe);
      for (int spins = 0;; ++spins) {
        if (aclshmemGmLoad32(remote) == ticket)
          return;
        if ((spins & 31) == 31) {
          aclshmemPutMemNbi(dst, (FLAGCX_ACLSHMEM_GM void *)sent, 4u, pe);
          aclshmemQuiet();
        }
        Intrin::spinBackoff(spins < 16 ? spins : 16);
      }
    }

    // Data must be in the peer's GM before any ticket: flush this core's
    // dirty lines, transfer, drain.  The transfer's completion evidence is
    // the ticket putImpl sends next: sendSignalTicket polls that exact
    // ticket at the peer through the pull path before it returns, and the
    // data put, the quiet and the ticket put issue in that order from this
    // core to the same PE - the vendor's own data-then-doorbell pattern
    // (barrier_all_kernel.cpp writes, quiet, then the ticket).  The probe
    // below is the retransmit trigger, not a completion proof: it reads the
    // first and last 4B chunks back, which is where the measured drop modes
    // (a lost posted write, a skipped last-chunk retire) surface; an
    // interior chunk lost on its own is invisible here and would show up as
    // a receiver-side payload mismatch, never as a boundary-consistent
    // pass.  There is no retry budget: with no runtime trap on this pass
    // (device_utils.h trap note) a put whose boundary chunks never arrive
    // keeps retrying instead of returning to signal delivery it did not
    // observe (MC952 review P1).  Sub-word and odd-tail transfers return
    // after quiet: they have no 4B probe, and their completion is the
    // ticket as well.
    FLAGCX_DEVICE_INLINE_DECORATOR void putData(FLAGCX_ACLSHMEM_GM void *dst,
                                                FLAGCX_ACLSHMEM_GM void *src,
                                                size_t bytes, int pe) const {
#if defined(__CCE_IS_AICORE__)
      asc_dcci_entire_out();
#endif
      aclshmemPutMemNbi(dst, src, (uint32_t)bytes, pe);
      aclshmemQuiet();
      if (bytes < 2u * sizeof(uint32_t) || (bytes % sizeof(uint32_t)) != 0u)
        return;
      size_t words = bytes / sizeof(uint32_t);
      const FLAGCX_ACLSHMEM_GM uint32_t *sw =
          (const FLAGCX_ACLSHMEM_GM uint32_t *)src;
      uint32_t head = sw[0];
      uint32_t tail = sw[words - 1u];
      FLAGCX_ACLSHMEM_GM uint32_t *dw = (FLAGCX_ACLSHMEM_GM uint32_t *)dst;
      FLAGCX_ACLSHMEM_GM volatile uint32_t *remoteHead =
          aclshmemPeerWord(dw, pe);
      FLAGCX_ACLSHMEM_GM volatile uint32_t *remoteTail =
          aclshmemPeerWord(dw + words - 1u, pe);
      for (int spins = 0;; ++spins) {
        if (aclshmemGmLoad32(remoteHead) == head &&
            aclshmemGmLoad32(remoteTail) == tail)
          return;
        if ((spins & 31) == 31) {
          aclshmemPutMemNbi(dst, src, (uint32_t)bytes, pe);
          aclshmemQuiet();
        }
        Intrin::spinBackoff(spins < 16 ? spins : 16);
      }
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
      sendSignalTicket(pe, ra.signal, aclshmemCompletionValue(ra.value));
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
// One state block per (kind, index) in the symmetric heap:
//   [0]                 this PE's local generation
//   [1 + teamRank]      generation this PE published for itself
// A member's slot is written only by that member, so a wait is a plain
// scan of the team members' slots, read from the owner's own memory.  The
// index is a channel dimension: concurrent Coop groups use distinct
// indices (flagcx_device_core.h, flagcxDevBarrier), so each index owns its
// own generation plane and two indices never alias.  The channel width is
// the largest count requested at communicator creation (floor:
// FLAGCX_ACLSHMEM_BARRIER_INDICES); anything beyond the granted capacity is
// rejected rather than folded onto another channel (MC952 review P1).
// ============================================================
// Tickets are pulled, never pushed.  A posted MTE put can be dropped; at
// np=2 a 4B barrier ticket was lost after its sender had already passed the
// barrier, so no retransmission by the sender could repair the wait (S16
// hung with one rank past and one rank spinning).  The vendor's own team
// barrier has the same shape this file now uses: each PE stores its
// generation in its own slot with a local store and peers poll that slot
// through aclshmem_ptr(), a read being the one direction that cannot lose.

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
  Barrier(Coop coop, const Comm &dc, Team team, uint32_t index, bool = false,
          const Multimem & = {})
      : _coop(coop), _state(dc.barrierState(0, index)),
        _base(0), _teamSize(team.nRanks), _teamRank(team.rank),
        _stride(team.stride) {
    _base = dc.rank - team.rank * team.stride;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t = flagcxDeviceMemoryOrderAcqRel) {
    if (_teamSize <= 1)
      return;
    // Earlier writes must reach the peer before its barrier ticket; the
    // published generation is a local store the peer reads back.
    aclshmemQuiet();
    uint32_t gen = aclshmemGmLoad32(_state) + 1u;
    aclshmemGmStore32(_state, gen);
    aclshmemGmStore32(_state + 1 + _teamRank, gen);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t = flagcxDeviceMemoryOrderAcqRel) {
    if (_teamSize <= 1)
      return;
    uint32_t gen = aclshmemGmLoad32(_state);
    for (int m = 0; m < _teamSize; ++m) {
      if (m == _teamRank)
        continue;
      FLAGCX_ACLSHMEM_GM volatile uint32_t *slot =
          aclshmemPeerWord(_state + 1 + m, _base + m * _stride);
      int iter = 0;
      while (aclshmemGmLoad32(slot) < gen)
        Intrin::spinBackoff(iter < 1000 ? iter : 1000), ++iter;
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
  Barrier(Coop coop, const Net &, const Comm &dc, Team team, uint32_t index,
          int = 0)
      : _coop(coop), _state(dc.barrierState(1, index)),
        _base(dc.rank - team.rank * team.stride), _teamSize(team.nRanks),
        _teamRank(team.rank), _stride(team.stride) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t = flagcxDeviceMemoryOrderAcqRel,
         flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    flagcxAclshmemRequireRelaxedFence(fence);
    if (_teamSize <= 1)
      return;
    // Earlier writes must reach the peer before its barrier ticket; the
    // published generation is a local store the peer reads back.
    aclshmemQuiet();
    uint32_t gen = aclshmemGmLoad32(_state) + 1u;
    aclshmemGmStore32(_state, gen);
    aclshmemGmStore32(_state + 1 + _teamRank, gen);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t = flagcxDeviceMemoryOrderAcqRel,
       flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    flagcxAclshmemRequireRelaxedFence(fence);
    if (_teamSize <= 1)
      return;
    uint32_t gen = aclshmemGmLoad32(_state);
    for (int m = 0; m < _teamSize; ++m) {
      if (m == _teamRank)
        continue;
      FLAGCX_ACLSHMEM_GM volatile uint32_t *slot =
          aclshmemPeerWord(_state + 1 + m, _base + m * _stride);
      int iter = 0;
      while (aclshmemGmLoad32(slot) < gen)
        Intrin::spinBackoff(iter < 1000 ? iter : 1000), ++iter;
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
          uint32_t index, bool, int)
      : _coop(coop), _state(dc.barrierState(2, index)),
        _base(0), _teamSize(dc.nRanks), _teamRank(dc.rank), _stride(1) {}

  // Intra-only form of the world barrier.
  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, flagcxTeamTagIntra, const Net &, const Comm &dc,
          uint32_t index, bool, int)
      : _coop(coop), _state(dc.barrierState(2, index)),
        _base(dc.rank - dc.intraRank), _teamSize(dc.intraSize),
        _teamRank(dc.intraRank), _stride(1) {}

  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, flagcxTeamTagInter, const Net &, const Comm &dc,
          uint32_t index, bool, int)
      : _coop(coop), _state(dc.barrierState(2, index)),
        _base(dc.rank), _teamSize(1), _teamRank(0), _stride(1) {
    (void)dc;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t = flagcxDeviceMemoryOrderAcqRel,
         flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    flagcxAclshmemRequireRelaxedFence(fence);
    if (_teamSize <= 1)
      return;
    // Earlier writes must reach the peer before its barrier ticket; the
    // published generation is a local store the peer reads back.
    aclshmemQuiet();
    uint32_t gen = aclshmemGmLoad32(_state) + 1u;
    aclshmemGmStore32(_state, gen);
    aclshmemGmStore32(_state + 1 + _teamRank, gen);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t = flagcxDeviceMemoryOrderAcqRel,
       flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    flagcxAclshmemRequireRelaxedFence(fence);
    if (_teamSize <= 1)
      return;
    uint32_t gen = aclshmemGmLoad32(_state);
    for (int m = 0; m < _teamSize; ++m) {
      if (m == _teamRank)
        continue;
      FLAGCX_ACLSHMEM_GM volatile uint32_t *slot =
          aclshmemPeerWord(_state + 1 + m, _base + m * _stride);
      int iter = 0;
      while (aclshmemGmLoad32(slot) < gen)
        Intrin::spinBackoff(iter < 1000 ? iter : 1000), ++iter;
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
