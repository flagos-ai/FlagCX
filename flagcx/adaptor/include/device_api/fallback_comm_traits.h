/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Fallback Device Traits — Common IPC-based implementation.
 *
 * CommTraits<Fallback<PlatformTag>> provides:
 *   - Intrin, Atomic: inherited from PlatformTraits<PlatformTag> via using
 *   - Window:   IPC peer pointers + raw pointer
 *   - DevComm:  rank/size + IPC barriers + signal buffers
 *   - Team:     pure arithmetic (nRanks, rank, stride)
 *   - Multimem: placeholder (no multicast)
 *
 * This partial specialization is written ONCE and works for any platform.
 * Adding a new platform requires zero changes here.
 ************************************************************************/

#ifndef FLAGCX_FALLBACK_DEVICE_TRAITS_H_
#define FLAGCX_FALLBACK_DEVICE_TRAITS_H_

#include "flagcx_kernel.h"

template <typename PlatformTag>
struct CommTraits<Fallback<PlatformTag>> {
  // Platform capabilities (resolved via PlatformTag)
  using Intrin = typename PlatformTraits<PlatformTag>::Intrin;
  using Atomic = typename PlatformTraits<PlatformTag>::Atomic;

  // ---- Team: Pure arithmetic ----
  struct Team {
    int nRanks, rank, stride;
  };

  // ---- Multimem: Placeholder ----
  struct Multimem {
    void *mcBasePtr;
  };

  // ---- Window: IPC + MR + rawPtr ----
  struct Window {
    void *rawPtr;     // Raw memory pointer (always valid)
    void **peerPtrs;  // IPC peer pointers (nullptr if no IPC)
    int intraRank;    // Local rank index
    uintptr_t mrBase; // MR base VA
    int mrIndex;      // MR table index

    FLAGCX_DEVICE_INLINE_DECORATOR void *
    getPeerPointer(size_t offset, const Team &team, int peer) const {
      if (peerPtrs) {
        int index = team.rank + (peer - team.rank) * team.stride;
        return (char *)peerPtrs[index] + offset;
      }
      return nullptr;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void *getLocalPointer(size_t offset) const {
      if (peerPtrs)
        return (char *)peerPtrs[intraRank] + offset;
      return (char *)rawPtr + offset;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void *getIntraPointer(size_t offset,
                                                         int peer) const {
      if (peerPtrs)
        return (char *)peerPtrs[peer] + offset;
      return nullptr;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void *
    getMulticastPointer(size_t offset, const Multimem &mm) const {
      (void)offset;
      (void)mm;
      return nullptr; // Multicast not available in fallback
    }

    FLAGCX_HOST_DEVICE_INLINE bool hasAccess() const {
      return rawPtr != nullptr || peerPtrs != nullptr;
    }
    FLAGCX_HOST_DEVICE_INLINE void *getRawPtr() const { return rawPtr; }
    FLAGCX_HOST_DEVICE_INLINE void **getDevPeerPtrs() const { return peerPtrs; }
    FLAGCX_HOST_DEVICE_INLINE int getMrIndex() const { return mrIndex; }

    FLAGCX_DEVICE_INLINE_DECORATOR bool operator==(const Window &o) const {
      return rawPtr == o.rawPtr;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR bool operator!=(const Window &o) const {
      return !(*this == o);
    }
  };

  // ---- DevComm: All fallback layers ----
  struct DevComm {
    // Baseline
    int rank, nRanks;
    int intraRank, intraSize;
    void *fifoBuffer;

    // IPC barriers
    uint64_t **barrierPeers;
    uint64_t intraBarrierEpoch;
    int nBarriers;

    // Inter-node signal relay
    uint64_t *interSignalFlags;
    int nInterPeers;
    bool isInterLeader;
    uint64_t interBarrierEpoch;

    // One-sided fallback
    uint64_t *signalBuffer;
    uint64_t *shadowBuffer;
    uint64_t *counterBuffer;
    int signalCount;
    int counterCount;
    int contextCount;

    FLAGCX_DEVICE_INLINE_DECORATOR int getIntraRank() const {
      return intraRank;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int getIntraSize() const {
      return intraSize;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int getRank() const { return rank; }
    FLAGCX_DEVICE_INLINE_DECORATOR int getSize() const { return nRanks; }
    FLAGCX_DEVICE_INLINE_DECORATOR void *getFifoBuffer() const {
      return fifoBuffer;
    }

    // Populate from host-side handle (deferred template avoids forward-decl)
    template <typename DI>
    static FLAGCX_HOST_DEVICE_INLINE void populateFromInternal(DevComm &dc,
                                                               const DI &di) {
      dc.rank = di.rank;
      dc.nRanks = di.nRanks;
      dc.intraRank = di.intraRank;
      dc.intraSize = di.intraSize;
      dc.fifoBuffer = di.fifoBuffer;
      dc.barrierPeers = di.barrierPeers;
      dc.intraBarrierEpoch = di.intraBarrierEpoch;
      dc.nBarriers = di.nBarriers;
      dc.interSignalFlags = di.interSignalFlags;
      dc.nInterPeers = di.nInterPeers;
      dc.isInterLeader = di.isInterLeader;
      dc.interBarrierEpoch = di.interBarrierEpoch;
      dc.signalBuffer = di.signalBuffer;
      dc.shadowBuffer = di.shadowBuffer;
      dc.counterBuffer = di.counterBuffer;
      dc.signalCount = di.signalCount;
      dc.counterCount = di.counterCount;
      dc.contextCount = di.contextCount;
    }
  };

  // ---- Coop types: aliased from PlatformTraits ----
  using CoopBlock = typename PlatformTraits<PlatformTag>::CoopBlock;
  template <int N>
  using CoopTile = typename PlatformTraits<PlatformTag>::template CoopTile<N>;
  using CoopThread = typename PlatformTraits<PlatformTag>::CoopThread;
  using CoopWarp = typename PlatformTraits<PlatformTag>::CoopWarp;
  using CoopTileSpan = typename PlatformTraits<PlatformTag>::CoopTileSpan;
  using CoopLanes = typename PlatformTraits<PlatformTag>::CoopLanes;
  using CoopAny = typename PlatformTraits<PlatformTag>::CoopAny;

  // ---- Barrier handles ----
  struct IntraBarrierHandle {
    int nBarriers;
  };
  struct InterBarrierHandle {
    int placeholder;
  };

  // ---- DescriptorSmem: empty on fallback ----
  struct DescriptorSmem {};

  // ---- DevBarrier alias: delegates to standalone DevBarrier<Backend, Tag>
  // ----
  template <typename Tag, typename Coop>
  using DevBarrier = ::DevBarrier<Fallback<PlatformTag>, Tag, Coop>;

  // ============================================================
  // Static FIFO helpers (used by Net and InterBarrierSession)
  // ============================================================

  // Build trd common header: prim(4) | peerRank(20) | primSpecific(36)
  FLAGCX_DEVICE_INLINE_DECORATOR
  static uint64_t buildTrd(uint64_t prim, uint64_t peerRank,
                           uint64_t primSpecific) {
    return ((prim & flagcxTriggerMask(flagcxDeviceTriggerBitsPrim))
            << flagcxDeviceTriggerOffPrim) |
           ((peerRank & flagcxTriggerMask(flagcxDeviceTriggerBitsPeerRank))
            << flagcxDeviceTriggerOffPeerRank) |
           primSpecific;
  }

  // Enqueue a trigger into the device FIFO buffer.
  // Atomically reserves a slot, waits for space, writes 3 words.
  FLAGCX_DEVICE_INLINE_DECORATOR
  static flagcxResult_t fifoEnqueue(void *fifoBuffer, uint64_t fstVal,
                                    uint64_t sndVal, uint64_t trdVal) {
    uint64_t *buffer = (uint64_t *)fifoBuffer;
    uint64_t capacity = Atomic::load(&buffer[flagcxFifoIdxCapacity],
                                     flagcxDeviceMemoryOrderRelaxed);

    // 1. Atomically reserve a slot
    uint64_t mySlot =
        Atomic::fetchAdd(&buffer[flagcxFifoIdxProduced], (uint64_t)1,
                         flagcxDeviceMemoryOrderAcqRel);

    // 2. Wait until there's space (mySlot - consumed < capacity)
    int iter = 0;
    while ((int64_t)(mySlot - Atomic::load(&buffer[flagcxFifoIdxConsumed],
                                           flagcxDeviceMemoryOrderAcquire)) >=
           (int64_t)capacity) {
      Intrin::spinBackoff(iter++);
    }

    // 3. Compute slot index and get pointers to slot's 3 uint64_t fields
    uint64_t idx = mySlot % capacity;
    uint64_t *slotFst = buffer + flagcxFifoIdxData +
                        idx * (sizeof(flagcxDeviceTrigger) / sizeof(uint64_t));
    uint64_t *slotSnd = slotFst + 1;
    uint64_t *slotTrd = slotFst + 2;

    // 4. Write fst, snd (payload, relaxed)
    Atomic::store(slotFst, fstVal, flagcxDeviceMemoryOrderRelaxed);
    Atomic::store(slotSnd, sndVal, flagcxDeviceMemoryOrderRelaxed);

    // 5. Write trd with valid bit (release ensures payload visible before
    // control)
    Atomic::store(slotTrd, trdVal | flagcxDeviceTriggerValidMask,
                  flagcxDeviceMemoryOrderRelease);

    return flagcxSuccess;
  }

  // Flush: snapshot produced, then spin until consumed >= snapshot.
  FLAGCX_DEVICE_INLINE_DECORATOR
  static flagcxResult_t fifoFlush(void *fifoBuffer) {
    uint64_t *buffer = (uint64_t *)fifoBuffer;
    uint64_t snapshot = Atomic::load(&buffer[flagcxFifoIdxProduced],
                                     flagcxDeviceMemoryOrderAcquire);
    int iter = 0;
    while (Atomic::load(&buffer[flagcxFifoIdxConsumed],
                        flagcxDeviceMemoryOrderAcquire) < snapshot) {
      Intrin::spinBackoff(iter++);
    }
    return flagcxSuccess;
  }

  // Wait: enqueue PrimWait + flush.
  FLAGCX_DEVICE_INLINE_DECORATOR
  static flagcxResult_t fifoWait(void *fifoBuffer) {
    fifoEnqueue(fifoBuffer, 0, 0, buildTrd(flagcxDevicePrimWait, 0, 0));
    return fifoFlush(fifoBuffer);
  }

  // ============================================================
  // Net: FIFO-based two-sided + one-sided + GPU-spin signal/counter
  // ============================================================
  struct Net {
    void *fifoBuffer;
    uint64_t *signalBuffer;
    uint64_t *shadowBuffer;
    uint64_t *counterBuffer;
    int signalCount;
    int counterCount;
    int contextId;

    FLAGCX_DEVICE_INLINE_DECORATOR
    Net(const DevComm &dc, int contextIndex)
        : fifoBuffer(dc.fifoBuffer), signalBuffer(dc.signalBuffer),
          shadowBuffer(dc.shadowBuffer), counterBuffer(dc.counterBuffer),
          signalCount(dc.signalCount), counterCount(dc.counterCount) {
      int cnt = (dc.contextCount > 0) ? dc.contextCount : 1;
      contextId = contextIndex % cnt;
    }

    // ---- Two-sided FIFO encoders ----
    FLAGCX_DEVICE_INLINE_DECORATOR void
    enqueueFifoSend(const Window &mem, size_t offset, size_t count,
                    flagcxDataType_t datatype, int peer) const {
      void *ptr = mem.getLocalPointer(offset);
      fifoEnqueue(
          fifoBuffer, (uint64_t)((uintptr_t)ptr), 0,
          buildTrd(flagcxDevicePrimSend, peer,
                   ((uint64_t)datatype << flagcxDeviceTriggerOffDatatype) |
                       ((uint64_t)count << flagcxDeviceTriggerOffCount)));
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    enqueueFifoRecv(const Window &mem, size_t offset, size_t count,
                    flagcxDataType_t datatype, int peer) const {
      void *ptr = mem.getLocalPointer(offset);
      fifoEnqueue(
          fifoBuffer, (uint64_t)((uintptr_t)ptr), 0,
          buildTrd(flagcxDevicePrimRecv, peer,
                   ((uint64_t)datatype << flagcxDeviceTriggerOffDatatype) |
                       ((uint64_t)count << flagcxDeviceTriggerOffCount)));
    }

    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t
    enqueueFifoTerm(int totalCoops) const {
      return fifoEnqueue(fifoBuffer, (uint64_t)totalCoops, 0,
                         buildTrd(flagcxDevicePrimTerm, 0, 0));
    }

    // ---- Two-sided Coop-scope operations ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t
    send(Coop coop, Window mem, size_t offset, size_t count,
         flagcxDataType_t datatype, int peer) const {
      coop.sync();
      if (coop.threadRank() == 0)
        enqueueFifoSend(mem, offset, count, datatype, peer);
      coop.sync();
      return flagcxSuccess;
    }

    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t
    recv(Coop coop, Window mem, size_t offset, size_t count,
         flagcxDataType_t datatype, int peer) const {
      coop.sync();
      if (coop.threadRank() == 0)
        enqueueFifoRecv(mem, offset, count, datatype, peer);
      coop.sync();
      return flagcxSuccess;
    }

    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t term(Coop coop) const {
      coop.sync();
      if (coop.threadRank() == 0) {
        int totalCoops = (FLAGCX_GRID_DIM_X * FLAGCX_BLOCK_DIM_X) / coop.size();
        enqueueFifoTerm(totalCoops);
      }
      coop.sync();
      return flagcxSuccess;
    }

    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t wait(Coop coop) const {
      coop.sync();
      if (coop.threadRank() == 0)
        fifoWait(fifoBuffer);
      coop.sync();
      return flagcxSuccess;
    }

    // ---- One-sided FIFO encoders ----
    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t
    enqueueFifoPut(size_t srcOffset, size_t dstOffset, size_t size, int peer,
                   int srcMrIdx, int dstMrIdx) const {
      uint64_t fstValue =
          ((uint64_t)srcOffset << flagcxDeviceTriggerOffSrcOffset) |
          ((uint64_t)dstOffset << flagcxDeviceTriggerOffDstOffset);
      uint64_t sndValue = (uint64_t)size << flagcxDeviceTriggerOffSize;
      uint64_t trdSpecific =
          ((uint64_t)srcMrIdx << flagcxDeviceTriggerOffSrcMrIdx) |
          ((uint64_t)dstMrIdx << flagcxDeviceTriggerOffDstMrIdx);
      return fifoEnqueue(fifoBuffer, fstValue, sndValue,
                         buildTrd(flagcxDevicePrimPut, peer, trdSpecific));
    }

    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t
    enqueueFifoGet(size_t srcOffset, size_t dstOffset, size_t size, int peer,
                   int srcMrIdx, int dstMrIdx) const {
      uint64_t fstValue =
          ((uint64_t)srcOffset << flagcxDeviceTriggerOffSrcOffset) |
          ((uint64_t)dstOffset << flagcxDeviceTriggerOffDstOffset);
      uint64_t sndValue = (uint64_t)size << flagcxDeviceTriggerOffSize;
      uint64_t trdSpecific =
          ((uint64_t)srcMrIdx << flagcxDeviceTriggerOffSrcMrIdx) |
          ((uint64_t)dstMrIdx << flagcxDeviceTriggerOffDstMrIdx);
      return fifoEnqueue(fifoBuffer, fstValue, sndValue,
                         buildTrd(flagcxDevicePrimGet, peer, trdSpecific));
    }

    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t
    enqueueFifoSignalRaw(int signalIdx, int peer) const {
      uint64_t trdSpecific = ((uint64_t)(contextId * signalCount + signalIdx)
                              << flagcxDeviceTriggerOffSignalIdxSig) |
                             ((uint64_t)1 << flagcxDeviceTriggerOffSignalValue);
      return fifoEnqueue(fifoBuffer, 0, 0,
                         buildTrd(flagcxDevicePrimSignal, peer, trdSpecific));
    }

    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t enqueueFifoSignal(
        int signalIdx, uint32_t value, int peer, uint64_t bufferType) const {
      int combinedIdx = (bufferType == 0)
                            ? (contextId * signalCount + signalIdx)
                            : (contextId * counterCount + signalIdx);
      uint64_t trdSpecific =
          ((uint64_t)bufferType << flagcxDeviceTriggerOffBufferType) |
          ((uint64_t)combinedIdx << flagcxDeviceTriggerOffSignalIdxSig) |
          ((uint64_t)(value & 0xFFFFu) << flagcxDeviceTriggerOffSignalValue);
      return fifoEnqueue(fifoBuffer, 0, 0,
                         buildTrd(flagcxDevicePrimSignal, peer, trdSpecific));
    }

    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t enqueueFifoPutValue(
        size_t dstOffset, uint64_t value, int peer, int dstMrIdx) const {
      uint64_t fstValue = (uint64_t)dstOffset &
                          flagcxTriggerMask(flagcxDeviceTriggerBitsDstOffset);
      uint64_t trdSpecific = (uint64_t)dstMrIdx
                             << flagcxDeviceTriggerOffDstMrIdx;
      return fifoEnqueue(fifoBuffer, fstValue, value,
                         buildTrd(flagcxDevicePrimPutValue, peer, trdSpecific));
    }

    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t enqueueFifoPutSignal(
        size_t srcOffset, size_t dstOffset, size_t size, int signalIdx,
        uint32_t signalValue, int peer, int srcMrIdx, int dstMrIdx) const {
      uint64_t fstValue =
          ((uint64_t)srcOffset << flagcxDeviceTriggerOffSrcOffset) |
          ((uint64_t)dstOffset << flagcxDeviceTriggerOffDstOffset);
      uint64_t sndValue = ((uint64_t)size << flagcxDeviceTriggerOffSize) |
                          ((uint64_t)(signalValue & 0xFFFFu)
                           << flagcxDeviceTriggerOffSignalValuePut);
      uint64_t trdSpecific =
          ((uint64_t)srcMrIdx << flagcxDeviceTriggerOffSrcMrIdx) |
          ((uint64_t)dstMrIdx << flagcxDeviceTriggerOffDstMrIdx) |
          ((uint64_t)(contextId * signalCount + signalIdx)
           << flagcxDeviceTriggerOffSignalIdx);
      return fifoEnqueue(
          fifoBuffer, fstValue, sndValue,
          buildTrd(flagcxDevicePrimPutSignal, peer, trdSpecific));
    }

    // ---- MR offset helper ----
    FLAGCX_DEVICE_INLINE_DECORATOR
    static size_t toDataOffset(const Window &win, size_t off) {
      void *ptr = win.getLocalPointer(off);
      return (uintptr_t)ptr - win.mrBase;
    }

    // ---- Action decomposition helpers ----
    template <typename T>
    FLAGCX_DEVICE_INLINE_DECORATOR constexpr bool isSignal(T) const {
      return false;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR constexpr bool
    isSignal(flagcxDevNet_SignalInc) const {
      return true;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR constexpr bool
    isSignal(flagcxDevNet_SignalAdd) const {
      return true;
    }

    template <typename T>
    FLAGCX_DEVICE_INLINE_DECORATOR constexpr int getSignalIdx(T) const {
      return 0;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR constexpr int
    getSignalIdx(flagcxDevNet_SignalInc a) const {
      return a.signal;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR constexpr int
    getSignalIdx(flagcxDevNet_SignalAdd a) const {
      return a.signal;
    }

    template <typename T>
    FLAGCX_DEVICE_INLINE_DECORATOR constexpr uint32_t getSignalValue(T) const {
      return 0;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR constexpr uint32_t
    getSignalValue(flagcxDevNet_SignalInc) const {
      return 1;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR constexpr uint32_t
    getSignalValue(flagcxDevNet_SignalAdd a) const {
      return (uint32_t)a.value;
    }

    template <typename T>
    FLAGCX_DEVICE_INLINE_DECORATOR constexpr bool canFuseSignal(T) const {
      return false;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR constexpr bool
    canFuseSignal(flagcxDevNet_SignalInc) const {
      return true;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR constexpr bool
    canFuseSignal(flagcxDevNet_SignalAdd) const {
      return true;
    }

    template <typename T>
    FLAGCX_DEVICE_INLINE_DECORATOR constexpr bool isCounter(T) const {
      return false;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR constexpr bool
    isCounter(flagcxDevNet_CounterInc) const {
      return true;
    }

    template <typename T>
    FLAGCX_DEVICE_INLINE_DECORATOR constexpr int getCounterIdx(T) const {
      return 0;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR constexpr int
    getCounterIdx(flagcxDevNet_CounterInc a) const {
      return a.counter;
    }

    // ---- One-sided: put (raw Window) ----
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
        size_t srcDataOff = toDataOffset(src, srcOff);
        size_t dstDataOff = toDataOffset(dst, dstOff);
        if (canFuseSignal(ra)) {
          enqueueFifoPutSignal(srcDataOff, dstDataOff, bytes, getSignalIdx(ra),
                               getSignalValue(ra), peer, src.getMrIndex(),
                               dst.getMrIndex());
        } else {
          enqueueFifoPut(srcDataOff, dstDataOff, bytes, peer, src.getMrIndex(),
                         dst.getMrIndex());
          if (isSignal(ra))
            enqueueFifoSignal(getSignalIdx(ra), getSignalValue(ra), peer, 0);
        }
        if (isCounter(la))
          enqueueFifoSignal(getCounterIdx(la), 1, 0, 1);
      }
      coop.sync();
    }

    // ---- One-sided: get (Coop-scope, Fallback only) ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    get(Team team, int peer, Window src, size_t srcOff, Window dst,
        size_t dstOff, size_t bytes, Coop coop) const {
      (void)team;
      coop.sync();
      if (coop.threadRank() == 0) {
        size_t srcDataOff = toDataOffset(src, srcOff);
        size_t dstDataOff = toDataOffset(dst, dstOff);
        enqueueFifoGet(srcDataOff, dstDataOff, bytes, peer, src.getMrIndex(),
                       dst.getMrIndex());
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
        size_t dstDataOff = toDataOffset(dst, dstOff);
        enqueueFifoPutValue(dstDataOff, (uint64_t)value, peer,
                            dst.getMrIndex());
        if (isSignal(ra))
          enqueueFifoSignal(getSignalIdx(ra), getSignalValue(ra), peer, 0);
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
        if (isSignal(ra))
          enqueueFifoSignal(getSignalIdx(ra), getSignalValue(ra), peer, 0);
      }
      coop.sync();
    }

    // ---- flush: drain FIFO (snapshot-spin, no PrimWait) ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    flush(Coop coop, flagcxDeviceMemoryOrder_t order) const {
      (void)order;
      coop.sync();
      if (coop.threadRank() == 0 && fifoBuffer != nullptr) {
        fifoFlush(fifoBuffer);
      }
      coop.sync();
    }

    // ---- waitSignal: GPU spin on signalBuffer[ctx*N+id] ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitSignal(Coop coop, flagcxDevNetSignal_t signalId, uint64_t least,
               int bits, flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      (void)order;
      coop.sync();
      if (coop.threadRank() == 0) {
        int idx = contextId * signalCount + (int)signalId;
        int iter = 0;
        while (Atomic::load(&signalBuffer[idx],
                            flagcxDeviceMemoryOrderAcquire) < least) {
          Intrin::spinBackoff(iter++);
        }
      }
      coop.sync();
    }

    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitSignalMeetShadow(Coop coop, flagcxDevNetSignal_t signalId, int bits,
                         flagcxDeviceMemoryOrder_t order) const {
      int idx = contextId * signalCount + (int)signalId;
      uint64_t shadow = ((volatile uint64_t *)shadowBuffer)[idx];
      waitSignal(coop, signalId, shadow, bits, order);
    }

    template <typename Coop, typename Uint>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitSignalFollowShadow(Coop coop, flagcxDevNetSignal_t signalId, Uint delta,
                           Uint *outSignalValue, Uint *outShadowValue, int bits,
                           flagcxDeviceMemoryOrder_t order) const {
      int idx = contextId * signalCount + (int)signalId;
      uint64_t shadow = ((volatile uint64_t *)shadowBuffer)[idx];
      uint64_t target = shadow + (uint64_t)delta;
      waitSignal(coop, signalId, target, bits, order);
      shadowBuffer[idx] = target;
      if (outSignalValue)
        *outSignalValue = (Uint)target;
      if (outShadowValue)
        *outShadowValue = (Uint)target;
    }

    // ---- Shadow manipulation ----
    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t *
    getSignalShadowPtr(flagcxDevNetSignal_t signalId) const {
      return &shadowBuffer[contextId * signalCount + (int)signalId];
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    increaseSignalShadow(flagcxDevNetSignal_t signalId, uint64_t delta) const {
      shadowBuffer[contextId * signalCount + (int)signalId] += delta;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
    readSignal(flagcxDevNetSignal_t signalId, int bits,
               flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      (void)order;
      int idx = contextId * signalCount + (int)signalId;
      return Atomic::load(&signalBuffer[idx], flagcxDeviceMemoryOrderAcquire);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    resetSignal(flagcxDevNetSignal_t signalId) const {
      int idx = contextId * signalCount + (int)signalId;
      Atomic::store(&signalBuffer[idx], (uint64_t)0,
                    flagcxDeviceMemoryOrderRelease);
    }

    // ---- Counter: GPU spin on counterBuffer[ctx*N+id] ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitCounter(Coop coop, flagcxDevNetCounter_t counterId, uint64_t least,
                int bits, flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      (void)order;
      coop.sync();
      if (coop.threadRank() == 0) {
        int idx = contextId * counterCount + (int)counterId;
        int iter = 0;
        while (Atomic::load(&counterBuffer[idx],
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
      int idx = contextId * counterCount + (int)counterId;
      return Atomic::load(&counterBuffer[idx], flagcxDeviceMemoryOrderAcquire);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    resetCounter(flagcxDevNetCounter_t counterId) const {
      int idx = contextId * counterCount + (int)counterId;
      Atomic::store(&counterBuffer[idx], (uint64_t)0,
                    flagcxDeviceMemoryOrderRelease);
    }
  };
};

// ============================================================
// DevBarrier specializations for Fallback<P>
//
// Standalone partial specializations (C++ forbids explicit specialization
// of member templates inside a partial class specialization).
// ============================================================

// ---- DevBarrier<Fallback<P>, flagcxBarrierIntra, Coop> ----
// Thread-striped per-peer inbox barrier using IPC-mapped atomics.
template <typename P, typename Coop>
struct DevBarrier<Fallback<P>, flagcxBarrierIntra, Coop> {
  using Atomic = typename PlatformTraits<P>::Atomic;
  using Intrin = typename PlatformTraits<P>::Intrin;
  using DevComm = typename CommTraits<Fallback<P>>::DevComm;
  using Team = typename CommTraits<Fallback<P>>::Team;
  using Multimem = typename CommTraits<Fallback<P>>::Multimem;

  Coop _coop;
  uint64_t **_peerBuffers;
  int _nRanks, _myRank;
  int _nBarriers;
  uint32_t _ctaIndex;
  uint64_t _epoch;

  // Default ctor (no-op, for barrier composition)
  FLAGCX_DEVICE_INLINE_DECORATOR
  DevBarrier()
      : _coop(), _peerBuffers(nullptr), _nRanks(0), _myRank(0), _nBarriers(0),
        _ctaIndex(0), _epoch(0) {}

  // Active ctor
  FLAGCX_DEVICE_INLINE_DECORATOR
  DevBarrier(Coop coop, const DevComm &dc, Team team, uint32_t index,
             bool = false, const Multimem & = {})
      : _coop(coop), _peerBuffers(dc.barrierPeers), _nRanks(team.nRanks),
        _myRank(team.rank), _nBarriers(dc.nBarriers), _ctaIndex(index),
        _epoch(dc.intraBarrierEpoch) {}

  // arrive: thread-striped store epoch+1 to each peer's inbox slot for me
  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    _coop.sync();
    for (int i = _coop.threadRank(); i < _nRanks - 1; i += _coop.size()) {
      int peer = 1 + _myRank + i;
      if (peer >= _nRanks)
        peer -= _nRanks;
      uint64_t *slot = &_peerBuffers[peer][_myRank * _nBarriers + _ctaIndex];
      Atomic::store(slot, _epoch + 1, flagcxDeviceMemoryOrderRelease);
    }
  }

  // wait: thread-striped spin on own inbox slots from each peer
  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    for (int i = _coop.threadRank(); i < _nRanks - 1; i += _coop.size()) {
      int peer = 1 + _myRank + i;
      if (peer >= _nRanks)
        peer -= _nRanks;
      uint64_t *slot = &_peerBuffers[_myRank][peer * _nBarriers + _ctaIndex];
      int iter = 0;
      while (Atomic::load(slot, flagcxDeviceMemoryOrderAcquire) < _epoch + 1) {
        Intrin::spinBackoff(iter++);
      }
    }
    _epoch += 1;
    _coop.sync();
  }

  // sync = arrive + wait
  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    arrive(order);
    wait(order);
  }
};

// ---- DevBarrier<Fallback<P>, flagcxBarrierInter, Coop> ----
// Inter-node barrier via FIFO BarrierSignal + host-mapped interSignalFlags.
// Only the inter leader actually sends/waits; non-leaders are no-ops.
template <typename P, typename Coop>
struct DevBarrier<Fallback<P>, flagcxBarrierInter, Coop> {
  using Atomic = typename PlatformTraits<P>::Atomic;
  using Intrin = typename PlatformTraits<P>::Intrin;
  using DevComm = typename CommTraits<Fallback<P>>::DevComm;
  using Team = typename CommTraits<Fallback<P>>::Team;
  using Net = typename CommTraits<Fallback<P>>::Net;

  Coop _coop;
  uint64_t *_interSignals;
  void *_fifoBuffer;
  int _nInterPeers;
  bool _isLeader;
  uint32_t _ctaIndex;
  uint64_t _epoch;

  // Default ctor (no-op)
  FLAGCX_DEVICE_INLINE_DECORATOR
  DevBarrier()
      : _coop(), _interSignals(nullptr), _fifoBuffer(nullptr), _nInterPeers(0),
        _isLeader(false), _ctaIndex(0), _epoch(0) {}

  // Active ctor
  FLAGCX_DEVICE_INLINE_DECORATOR
  DevBarrier(Coop coop, const Net &net, const DevComm &dc, Team, uint32_t index,
             int nInterPeers)
      : _coop(coop), _interSignals(dc.interSignalFlags),
        _fifoBuffer(dc.fifoBuffer), _nInterPeers(nInterPeers),
        _isLeader(dc.isInterLeader), _ctaIndex(index),
        _epoch(dc.interBarrierEpoch) {}

  // arrive: FIFO BarrierSignal (leader only)
  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
         flagcxGinFenceLevel fence = flagcxGinFenceLevel::Relaxed) {
    _epoch += _nInterPeers;
    _coop.sync();
    if (_coop.threadRank() == 0 && _isLeader) {
      CommTraits<Fallback<P>>::fifoEnqueue(
          _fifoBuffer, (uint64_t)_ctaIndex, 0,
          CommTraits<Fallback<P>>::buildTrd(flagcxDevicePrimBarrierSignal, 0,
                                            0));
    }
    _coop.sync();
  }

  // wait: spin on host-mapped inter signal array (leader only)
  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxGinFenceLevel fence = flagcxGinFenceLevel::Relaxed) {
    _coop.sync();
    if (_coop.threadRank() == 0 && _isLeader) {
      int iter = 0;
      while (Atomic::load(&_interSignals[_ctaIndex],
                          flagcxDeviceMemoryOrderAcquire) < _epoch) {
        Intrin::spinBackoff(iter++);
      }
    }
    _coop.sync();
  }

  // sync = arrive + wait
  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxGinFenceLevel fence = flagcxGinFenceLevel::Relaxed) {
    arrive(order);
    wait(order);
  }
};

// ---- DevBarrier<Fallback<P>, flagcxBarrierWorld, Coop> ----
// Composes intra + inter barriers.
// Three-phase pattern for multi-node: intra → inter → intra.
// Single-node: just one intra sync.
template <typename P, typename Coop>
struct DevBarrier<Fallback<P>, flagcxBarrierWorld, Coop> {
  using DevComm = typename CommTraits<Fallback<P>>::DevComm;
  using Team = typename CommTraits<Fallback<P>>::Team;
  using Net = typename CommTraits<Fallback<P>>::Net;

  Coop _coop;
  DevBarrier<Fallback<P>, flagcxBarrierIntra, Coop> _intra;
  DevBarrier<Fallback<P>, flagcxBarrierInter, Coop> _inter;
  int _nInterPeers;

  // World barrier: intra (IPC) + inter (FIFO Signal)
  FLAGCX_DEVICE_INLINE_DECORATOR
  DevBarrier(Coop coop, flagcxBarrierWorld::World, const Net &net,
             const DevComm &dc, uint32_t index, bool multimem, int nInterPeers)
      : _coop(coop),
        _intra(coop, dc, Team{dc.intraSize, dc.intraRank, 1}, index),
        _inter(coop, net, dc, Team{}, index, nInterPeers),
        _nInterPeers(nInterPeers) {}

  // Intra-only barrier: inter is default constructed (no-op)
  FLAGCX_DEVICE_INLINE_DECORATOR
  DevBarrier(Coop coop, flagcxBarrierWorld::Intra, const Net &,
             const DevComm &dc, uint32_t index, bool multimem, int)
      : _coop(coop),
        _intra(coop, dc, Team{dc.intraSize, dc.intraRank, 1}, index), _inter(),
        _nInterPeers(0) {}

  // Inter-only barrier: intra is default constructed (no-op)
  FLAGCX_DEVICE_INLINE_DECORATOR
  DevBarrier(Coop coop, flagcxBarrierWorld::Inter, const Net &net,
             const DevComm &dc, uint32_t index, bool, int nInterPeers)
      : _coop(coop), _intra(),
        _inter(coop, net, dc, Team{}, index, nInterPeers),
        _nInterPeers(nInterPeers) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
         flagcxGinFenceLevel fence = flagcxGinFenceLevel::Relaxed) {
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
       flagcxGinFenceLevel fence = flagcxGinFenceLevel::Relaxed) {
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
       flagcxGinFenceLevel fence = flagcxGinFenceLevel::Relaxed) {
    if (_nInterPeers > 0) {
      // Phase 1: intra sync
      _intra.arrive(flagcxDeviceMemoryOrderRelease);
      _intra.wait(flagcxDeviceMemoryOrderRelease);
      // Phase 2: inter signal+wait (leader only)
      _inter.arrive(order);
      _inter.wait(order);
      // Phase 3: intra sync (broadcast inter completion)
      _intra.arrive(flagcxDeviceMemoryOrderAcquire);
      _intra.wait(flagcxDeviceMemoryOrderAcquire);
    } else {
      // Single-node: one intra sync
      _intra.arrive(order);
      _intra.wait(order);
    }
  }
};

#endif // FLAGCX_FALLBACK_DEVICE_TRAITS_H_
