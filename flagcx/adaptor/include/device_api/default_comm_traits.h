/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Default Device Traits — Common IPC-based implementation.
 *
 * CommTraits<DefaultBackend<PlatformTag>> provides:
 *   - Intrin, Atomic: inherited from PlatformTraits<PlatformTag> via using
 *   - Window:   IPC peer pointers + raw pointer
 *   - Comm:  rank/size + IPC barriers + signal buffers
 *   - Team:     pure arithmetic (nRanks, rank, stride)
 *   - Multimem: placeholder (no multicast)
 *
 * This partial specialization is written ONCE and works for any platform.
 * Adding a new platform requires zero changes here.
 ************************************************************************/

#ifndef FLAGCX_FALLBACK_DEVICE_TRAITS_H_
#define FLAGCX_FALLBACK_DEVICE_TRAITS_H_

#include "flagcx_kernel_core.h"
#include <cassert>
#ifndef __CUDACC__
#include "sym_heap.h"
#endif

template <typename PlatformTag>
struct CommTraits<DefaultBackend<PlatformTag>> {
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

  // ---- Window: Symmetric (VMM) or Asymmetric (IPC) ----
  enum WindowMode { SYMMETRIC = 0, ASYMMETRIC = 1 };
  struct Window {
    WindowMode mode;    // SYMMETRIC (VMM) or ASYMMETRIC (IPC)
    void *flatBasePtr;  // Flat VA base (SYMMETRIC mode, nullable)
    size_t allocSize;   // Per-rank allocation size (SYMMETRIC mode)
    void *mcBasePtr;    // Multicast base (nullable, SYMMETRIC mode only)
    void **ipcBasePtrs; // IPC peer pointers (ASYMMETRIC mode, nullable)
    int intraRank;      // Local rank index
    int intraSize;      // Number of local ranks (bounds for ipcBasePtrs)
    uintptr_t mrBase;   // MR base VA (inter-node, orthogonal to mode)
    int mrIndex;        // MR table index (-1 if none)
    void *rawPtr;       // Raw pointer fallback (for getLocalPointer)

    FLAGCX_DEVICE_INLINE_DECORATOR void *
    getPeerPointer(size_t offset, const Team &team, int peer) const {
      if (mode == SYMMETRIC && flatBasePtr) {
        int index = team.rank + (peer - team.rank) * team.stride;
        return (char *)flatBasePtr + (size_t)index * allocSize + offset;
      } else if (ipcBasePtrs) {
        int index = intraRank + (peer - team.rank) * team.stride;
        if (index < 0 || index >= intraSize)
          return nullptr; // Not a local peer — fall through to Net
        if (ipcBasePtrs[index] == nullptr)
          return nullptr;
        return (char *)ipcBasePtrs[index] + offset;
      }
      return nullptr;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void *getLocalPointer(size_t offset) const {
      if (mode == SYMMETRIC && flatBasePtr)
        return (char *)flatBasePtr + (size_t)intraRank * allocSize + offset;
      else if (ipcBasePtrs && intraRank >= 0 && intraRank < intraSize &&
               ipcBasePtrs[intraRank])
        return (char *)ipcBasePtrs[intraRank] + offset;
      return (char *)rawPtr + offset;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void *getIntraPointer(size_t offset,
                                                         int peer) const {
      if (mode == SYMMETRIC && flatBasePtr)
        return (char *)flatBasePtr + (size_t)peer * allocSize + offset;
      else if (ipcBasePtrs && peer >= 0 && peer < intraSize &&
               ipcBasePtrs[peer])
        return (char *)ipcBasePtrs[peer] + offset;
      return nullptr;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void *
    getMulticastPointer(size_t offset, const Multimem &mm) const {
      if (mcBasePtr)
        return (char *)mcBasePtr + offset;
      (void)mm;
      return nullptr;
    }

    FLAGCX_HOST_DEVICE_INLINE bool hasAccess() const {
      return (mode == SYMMETRIC && flatBasePtr != nullptr) ||
             (mode == ASYMMETRIC && ipcBasePtrs != nullptr) || mrIndex >= 0;
    }
    FLAGCX_HOST_DEVICE_INLINE void *getRawPtr() const { return rawPtr; }
    FLAGCX_HOST_DEVICE_INLINE void **getDevPeerPtrs() const {
      return ipcBasePtrs;
    }
    FLAGCX_HOST_DEVICE_INLINE int getMrIndex() const { return mrIndex; }

    FLAGCX_DEVICE_INLINE_DECORATOR bool operator==(const Window &o) const {
      if (mode == SYMMETRIC && o.mode == SYMMETRIC)
        return flatBasePtr == o.flatBasePtr && intraRank == o.intraRank;
      return rawPtr == o.rawPtr;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR bool operator!=(const Window &o) const {
      return !(*this == o);
    }

#ifndef __CUDACC__
    // Host-side population from flagcxWindow_t (sym heap or IPC).
    void populateFromHost(flagcxWindow_t win, void *rawPtr_, int intraRank_,
                          int intraSize_, int mrIndex_, uintptr_t mrBase_,
                          int ipcIndex_, void **ipcDevPeerPtrs_) {
      rawPtr = rawPtr_;
      intraRank = intraRank_;
      intraSize = intraSize_;
      mrBase = mrBase_;
      mrIndex = mrIndex_;

      flagcxSymWindow_t d =
          (win && win->isSymmetricDefault) ? win->defaultBase : nullptr;

      if (d && d->isVMM && d->flatBase) {
        mode = SYMMETRIC;
        flatBasePtr = d->flatBase;
        allocSize = d->allocSize;
        mcBasePtr = d->mcBase;
        ipcBasePtrs = nullptr;
      } else {
        mode = ASYMMETRIC;
        flatBasePtr = nullptr;
        allocSize = 0;
        mcBasePtr = nullptr;
        ipcBasePtrs = (ipcIndex_ >= 0) ? ipcDevPeerPtrs_ : nullptr;
      }
    }
#endif // __CUDACC__
  };

  // ---- Comm: All fallback layers ----
  struct Comm {
    // Baseline
    int rank, nRanks;
    int intraRank, intraSize;
    void *fifoBuffers[FLAGCX_DEVICE_CTA_COUNT];

    // IPC barriers
    uint64_t **barrierPeers;
    uint64_t
        *epochBuffer; // Device pointer: [intra live, inter live] × CTA_COUNT
    int nBarriers;

    // Inter-node signal relay
    int nInterPeers;

    // NCCL GIN-style barrier
    int teamRank;          // this rank's position in inter-node team
    int nTeamRanks;        // total nodes in team
    int barrierSignalBase; // first signal slot for barriers

    // One-sided fallback
    uint64_t *signalBuffer;
    uint64_t *shadowBuffer;
    uint64_t *counterBuffer;
    int signalCount;
    int counterCount;
    int contextCount;

    int netOneSidedReady;
    int netSignalReady;
    int netPutValueReady;
    int useP2pSignals;

    // P2P signal delivery (IPC-mapped pointers to each peer's signal buffers).
    // signalPeerPtrs[peer] → peer's signalBuffer (nullptr if not P2P-reachable)
    uint64_t **signalPeerPtrs;

    FLAGCX_DEVICE_INLINE_DECORATOR int getIntraRank() const {
      return intraRank;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int getIntraSize() const {
      return intraSize;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR int getRank() const { return rank; }
    FLAGCX_DEVICE_INLINE_DECORATOR int getSize() const { return nRanks; }
    FLAGCX_DEVICE_INLINE_DECORATOR void *getFifoBuffer(int contextId) const {
      return fifoBuffers[contextId];
    }
    FLAGCX_DEVICE_INLINE_DECORATOR Multimem getMulticastHandle() const {
      Multimem mm;
      mm.mcBasePtr = nullptr;
      return mm;
    }

    // P2P signal support getter
    FLAGCX_DEVICE_INLINE_DECORATOR bool p2pSignalSupport(int peer) const {
      return signalPeerPtrs && signalPeerPtrs[peer] != nullptr;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t *getSignalPeerPtr(int peer) const {
      return signalPeerPtrs ? signalPeerPtrs[peer] : nullptr;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR bool usesDirectP2pSignals() const {
      return useP2pSignals != 0;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR bool isOneSidedTransportReady() const {
      return netOneSidedReady != 0;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR bool supportsDirectCounterAccess() const {
      return counterBuffer != nullptr;
    }

    // Populate from host-side handle (deferred template avoids forward-decl)
    template <typename DI>
    static FLAGCX_HOST_DEVICE_INLINE void populateFromInternal(Comm &dc,
                                                               const DI &di) {
      dc.rank = di.rank;
      dc.nRanks = di.nRanks;
      dc.intraRank = di.intraRank;
      dc.intraSize = di.intraSize;
      for (int i = 0; i < di.contextCount; i++)
        dc.fifoBuffers[i] = di.fifoBuffers[i];
      dc.barrierPeers = di.barrierPeers;
      dc.epochBuffer = di.epochBuffer;
      dc.nBarriers = di.nBarriers;
      dc.nInterPeers = di.nInterPeers;
      dc.teamRank = di.teamRank;
      dc.nTeamRanks = di.nTeamRanks;
      dc.barrierSignalBase = di.barrierSignalBase;
      dc.signalBuffer = di.signalBuffer;
      dc.shadowBuffer = di.shadowBuffer;
      dc.counterBuffer = di.counterBuffer;
      dc.signalCount = di.signalCount;
      dc.counterCount = di.counterCount;
      dc.contextCount = di.contextCount;
      dc.signalPeerPtrs = di.signalPeerPtrs;
      dc.netOneSidedReady = di.netOneSidedReady;
      dc.netSignalReady = di.netSignalReady;
      dc.netPutValueReady = di.netPutValueReady;
      dc.useP2pSignals = di.useP2pSignals;
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

  // ---- Barrier alias: delegates to standalone Barrier<Backend, Tag>
  // ----
  template <typename Tag, typename Coop>
  using Barrier = ::Barrier<DefaultBackend<PlatformTag>, Tag, Coop>;

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

  // Flush: snapshot produced, then spin until completed >= snapshot.
  // The CPU proxy advances 'completed' after IB test() confirms each op done.
  // This replaces the old PrimWait+streamSynchronize approach that caused
  // deadlocks.
  FLAGCX_DEVICE_INLINE_DECORATOR
  static flagcxResult_t fifoFlush(void *fifoBuffer) {
    uint64_t *buffer = (uint64_t *)fifoBuffer;
    uint64_t snapshot = Atomic::load(&buffer[flagcxFifoIdxProduced],
                                     flagcxDeviceMemoryOrderAcquire);
    int iter = 0;
    while (Atomic::load(&buffer[flagcxFifoIdxCompleted],
                        flagcxDeviceMemoryOrderAcquire) < snapshot) {
      Intrin::spinBackoff(iter++);
    }
    return flagcxSuccess;
  }

  // Wait: just flush (no PrimWait enqueue needed with completion-based flush).
  FLAGCX_DEVICE_INLINE_DECORATOR
  static flagcxResult_t fifoWait(void *fifoBuffer) {
    return fifoFlush(fifoBuffer);
  }

  // ============================================================
  // Net: FIFO-based two-sided + one-sided + GPU-spin signal/counter
  // ============================================================
  struct Net {
    Comm _dc;
    void *fifoBuffer;
    uint64_t *signalBuffer;
    uint64_t *shadowBuffer;
    uint64_t *counterBuffer;
    int signalCount;
    int counterCount;
    int _contextId;
    int teamRank;
    int nTeamRanks;
    int barrierSignalBase;

    FLAGCX_DEVICE_INLINE_DECORATOR
    Net(const Comm &dc, int contextIndex)
        : _dc(dc),
          fifoBuffer(
              dc.fifoBuffers[contextIndex %
                             ((dc.contextCount > 0) ? dc.contextCount : 1)]),
          signalBuffer(dc.signalBuffer), shadowBuffer(dc.shadowBuffer),
          counterBuffer(dc.counterBuffer), signalCount(dc.signalCount),
          counterCount(dc.counterCount) {
      int cnt = (dc.contextCount > 0) ? dc.contextCount : 1;
      _contextId = contextIndex % cnt;
      teamRank = dc.teamRank;
      nTeamRanks = dc.nTeamRanks;
      barrierSignalBase = dc.barrierSignalBase;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR int getContextId() const {
      return _contextId;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t *
    getSignalPtr(flagcxDevSignal_t signalId) const {
      return &signalBuffer[_contextId * signalCount + (int)signalId];
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t *
    getPeerSignalPtr(int localPeer, flagcxDevSignal_t signalId) const {
      uint64_t *peerBuffer = _dc.getSignalPeerPtr(localPeer);
      return peerBuffer ? &peerBuffer[_contextId * signalCount + (int)signalId]
                        : nullptr;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t *
    getCounterPtr(flagcxDevCounter_t counterId) const {
      return &counterBuffer[_contextId * counterCount + (int)counterId];
    }

    FLAGCX_DEVICE_INLINE_DECORATOR bool isIntraPeer(int peer) const {
      int intraBase = _dc.rank - _dc.intraRank;
      return peer >= intraBase && peer < intraBase + _dc.intraSize;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR bool isValid() const {
      return signalBuffer != nullptr && counterBuffer != nullptr &&
             fifoBuffer != nullptr;
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
      uint64_t trdSpecific = ((uint64_t)(_contextId * signalCount + signalIdx)
                              << flagcxDeviceTriggerOffSignalIdxSig) |
                             ((uint64_t)1 << flagcxDeviceTriggerOffSignalValue);
      return fifoEnqueue(fifoBuffer, 0, 0,
                         buildTrd(flagcxDevicePrimSignal, peer, trdSpecific));
    }

    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t enqueueFifoSignal(
        int signalIdx, uint32_t value, int peer, uint64_t bufferType) const {
      int combinedIdx = (bufferType == 0)
                            ? (_contextId * signalCount + signalIdx)
                            : (_contextId * counterCount + signalIdx);
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
          ((uint64_t)(_contextId * signalCount + signalIdx)
           << flagcxDeviceTriggerOffSignalIdx);
      return fifoEnqueue(
          fifoBuffer, fstValue, sndValue,
          buildTrd(flagcxDevicePrimPutSignal, peer, trdSpecific));
    }

    // ---- MR offset helper ----
    FLAGCX_DEVICE_INLINE_DECORATOR
    static size_t toDataOffset(const Window &win, size_t off) {
      // Use rawPtr (the original buffer VA used for MR registration) rather
      // than getLocalPointer() which may return a VMM flat-mapped VA that
      // differs from the MR-registered VA.
      return (uintptr_t)win.getRawPtr() + off - win.mrBase;
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

    // ---- Team-scoped peer → world rank resolution ----
    FLAGCX_DEVICE_INLINE_DECORATOR int teamRankToWorld(Team team,
                                                       int peer) const {
      return _dc.rank + (peer - team.rank) * team.stride;
    }

    // ---- One-sided: put (raw Window) ----
    template <typename RA, typename LA, typename Coop, typename Desc>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    put(Team team, int peer, Window dst, size_t dstOff, Window src,
        size_t srcOff, size_t bytes, RA ra, LA la, Coop coop, Desc desc,
        flagcxDeviceScope_t ar, flagcxDeviceScope_t es) const {
      (void)desc;
      (void)ar;
      (void)es;
      int worldPeer = teamRankToWorld(team, peer);
      coop.sync();
      if (coop.threadRank() == 0) {
        size_t srcDataOff = toDataOffset(src, srcOff);
        size_t dstDataOff = toDataOffset(dst, dstOff);
        if (canFuseSignal(ra)) {
          enqueueFifoPutSignal(srcDataOff, dstDataOff, bytes, getSignalIdx(ra),
                               getSignalValue(ra), worldPeer, src.getMrIndex(),
                               dst.getMrIndex());
        } else {
          enqueueFifoPut(srcDataOff, dstDataOff, bytes, worldPeer,
                         src.getMrIndex(), dst.getMrIndex());
          if (isSignal(ra))
            enqueueFifoSignal(getSignalIdx(ra), getSignalValue(ra), worldPeer,
                              0);
        }
        if (isCounter(la))
          enqueueFifoSignal(getCounterIdx(la), 1, 0, 1);
      }
      coop.sync();
    }

    // ---- One-sided: get (Coop-scope, Default only) ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    get(Team team, int peer, Window src, size_t srcOff, Window dst,
        size_t dstOff, size_t bytes, Coop coop) const {
      int worldPeer = teamRankToWorld(team, peer);
      coop.sync();
      if (coop.threadRank() == 0) {
        size_t srcDataOff = toDataOffset(src, srcOff);
        size_t dstDataOff = toDataOffset(dst, dstOff);
        enqueueFifoGet(srcDataOff, dstDataOff, bytes, worldPeer,
                       src.getMrIndex(), dst.getMrIndex());
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
      int worldPeer = teamRankToWorld(team, peer);
      coop.sync();
      if (coop.threadRank() == 0) {
        size_t dstDataOff = toDataOffset(dst, dstOff);
        enqueueFifoPutValue(dstDataOff, (uint64_t)value, worldPeer,
                            dst.getMrIndex());
        if (isSignal(ra))
          enqueueFifoSignal(getSignalIdx(ra), getSignalValue(ra), worldPeer, 0);
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
      int worldPeer = teamRankToWorld(team, peer);
      coop.sync();
      if (coop.threadRank() == 0) {
        if (isSignal(ra))
          enqueueFifoSignal(getSignalIdx(ra), getSignalValue(ra), worldPeer, 0);
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
    waitSignal(Coop coop, flagcxDevSignal_t signalId, uint64_t least, int bits,
               flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      coop.sync();
      if (coop.threadRank() == 0) {
        int idx = _contextId * signalCount + (int)signalId;
        int iter = 0;
        uint64_t cur;
        while ((cur = Atomic::load(&signalBuffer[idx], order)) < least) {
          Intrin::spinBackoff(iter++);
        }
      }
      coop.sync();
    }

    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitSignalMeetShadow(Coop coop, flagcxDevSignal_t signalId, int bits,
                         flagcxDeviceMemoryOrder_t order) const {
      int idx = _contextId * signalCount + (int)signalId;
      uint64_t shadow = ((volatile uint64_t *)shadowBuffer)[idx];
      waitSignal(coop, signalId, shadow, bits, order);
    }

    template <typename Coop, typename Uint>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitSignalFollowShadow(Coop coop, flagcxDevSignal_t signalId, Uint delta,
                           Uint *outSignalValue, Uint *outShadowValue, int bits,
                           flagcxDeviceMemoryOrder_t order) const {
      int idx = _contextId * signalCount + (int)signalId;
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
    getSignalShadowPtr(flagcxDevSignal_t signalId) const {
      return &shadowBuffer[_contextId * signalCount + (int)signalId];
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    increaseSignalShadow(flagcxDevSignal_t signalId, uint64_t delta) const {
      shadowBuffer[_contextId * signalCount + (int)signalId] += delta;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
    readSignal(flagcxDevSignal_t signalId, int bits,
               flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      int idx = _contextId * signalCount + (int)signalId;
      return Atomic::load(&signalBuffer[idx], order);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    resetSignal(flagcxDevSignal_t signalId) const {
      int idx = _contextId * signalCount + (int)signalId;
      Atomic::store(&signalBuffer[idx], (uint64_t)0,
                    flagcxDeviceMemoryOrderRelease);
      Atomic::store(&shadowBuffer[idx], (uint64_t)0,
                    flagcxDeviceMemoryOrderRelease);
    }

    // ---- Counter: GPU spin on counterBuffer[ctx*N+id] ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitCounter(Coop coop, flagcxDevCounter_t counterId, uint64_t least,
                int bits, flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      coop.sync();
      if (coop.threadRank() == 0) {
        int idx = _contextId * counterCount + (int)counterId;
        int iter = 0;
        while (Atomic::load(&counterBuffer[idx], order) < least) {
          Intrin::spinBackoff(iter++);
        }
      }
      coop.sync();
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
    readCounter(flagcxDevCounter_t counterId, int bits,
                flagcxDeviceMemoryOrder_t order) const {
      (void)bits;
      int idx = _contextId * counterCount + (int)counterId;
      return Atomic::load(&counterBuffer[idx], order);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    resetCounter(flagcxDevCounter_t counterId) const {
      int idx = _contextId * counterCount + (int)counterId;
      Atomic::store(&counterBuffer[idx], (uint64_t)0,
                    flagcxDeviceMemoryOrderRelease);
    }
  };
};

// ============================================================
// Barrier specializations for DefaultBackend<P>
//
// Standalone partial specializations (C++ forbids explicit specialization
// of member templates inside a partial class specialization).
// ============================================================

// ---- Barrier<DefaultBackend<P>, flagcxTeamTagIntra, Coop> ----
// Thread-striped per-peer inbox barrier using IPC-mapped atomics.
template <typename P, typename Coop>
struct Barrier<DefaultBackend<P>, flagcxTeamTagIntra, Coop> {
  using Atomic = typename PlatformTraits<P>::Atomic;
  using Intrin = typename PlatformTraits<P>::Intrin;
  using Comm = typename CommTraits<DefaultBackend<P>>::Comm;
  using Team = typename CommTraits<DefaultBackend<P>>::Team;
  using Multimem = typename CommTraits<DefaultBackend<P>>::Multimem;

  Coop _coop;
  uint64_t **_peerBuffers;
  int _nRanks, _myRank;
  int _nBarriers;
  uint32_t _ctaIndex;
  uint64_t *_epochBuffer;
  uint64_t _epoch;

  // Default ctor (no-op, for barrier composition)
  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier()
      : _coop(), _peerBuffers(nullptr), _nRanks(0), _myRank(0), _nBarriers(0),
        _ctaIndex(0), _epochBuffer(nullptr), _epoch(0) {}

  // Active ctor
  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, const Comm &dc, Team team, uint32_t index, bool = false,
          const Multimem & = {})
      : _coop(coop), _peerBuffers(dc.barrierPeers), _nRanks(team.nRanks),
        _myRank(team.rank), _nBarriers(dc.nBarriers), _ctaIndex(index),
        _epochBuffer(dc.epochBuffer),
        _epoch(Atomic::load(&dc.epochBuffer[index],
                            flagcxDeviceMemoryOrderAcquire)) {
    assert(index < FLAGCX_DEVICE_CTA_COUNT);
  }

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
      uint64_t current = Atomic::load(slot, flagcxDeviceMemoryOrderAcquire);
      while (current < _epoch + 1) {
        Intrin::spinBackoff(iter++);
        current = Atomic::load(slot, flagcxDeviceMemoryOrderAcquire);
      }
    }
    _epoch += 1;
    if (_coop.threadRank() == 0) {
      Atomic::store(&_epochBuffer[_ctaIndex], _epoch,
                    flagcxDeviceMemoryOrderRelease);
    }
    _coop.sync();
  }

  // sync = arrive + wait
  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    arrive(order);
    wait(order);
  }
};

// ---- Barrier<DefaultBackend<P>, flagcxTeamTagInter, Coop> ----
// NCCL GIN-style: all ranks participate, interleaved signal+wait per peer.
// No leader gating. GPU polls device memory signal buffer directly.
// Barrier signals use regular PrimSignal FIFO entries (async, non-blocking
// proxy).
template <typename P, typename Coop>
struct Barrier<DefaultBackend<P>, flagcxTeamTagInter, Coop> {
  using Atomic = typename PlatformTraits<P>::Atomic;
  using Intrin = typename PlatformTraits<P>::Intrin;
  using Comm = typename CommTraits<DefaultBackend<P>>::Comm;
  using Team = typename CommTraits<DefaultBackend<P>>::Team;
  using Net = typename CommTraits<DefaultBackend<P>>::Net;

  Coop _coop;
  void *_fifoBuffer;
  uint64_t *_signalBuffer;
  uint64_t *_shadowBuffer;
  int _signalCount;
  int _contextId;
  int _teamRank;
  int _nTeamRanks;
  int _barrierSignal0; // base signal slot for this barrier instance
  int _stride;         // intraSize — used to compute target global rank
  int _localRank;      // intra-node rank (for rank-to-rank peer mapping)

  // Default ctor (no-op)
  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier()
      : _coop(), _fifoBuffer(nullptr), _signalBuffer(nullptr),
        _shadowBuffer(nullptr), _signalCount(0), _contextId(0), _teamRank(0),
        _nTeamRanks(0), _barrierSignal0(0), _stride(0), _localRank(0) {}

  // Active ctor
  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, const Net &net, const Comm &dc, Team, uint32_t index,
          int nInterPeers)
      : _coop(coop), _fifoBuffer(net.fifoBuffer),
        _signalBuffer(net.signalBuffer), _shadowBuffer(net.shadowBuffer),
        _signalCount(net.signalCount), _contextId(net.getContextId()),
        _teamRank(net.teamRank), _nTeamRanks(net.nTeamRanks),
        _barrierSignal0(net.barrierSignalBase + (int)index * net.nTeamRanks),
        _stride(dc.intraSize), _localRank(dc.intraRank) {
    assert(index < FLAGCX_DEVICE_CTA_COUNT);
  }

  // arrive: signal all remote peers "I have arrived"
  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
         flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    if (_nTeamRanks <= 1)
      return;
    int nPeers = _nTeamRanks - 1;
    FLAGCX_DEVICE_SYNC_THREADS();
    // Each thread handles a subset of peers (cooperative distribution)
    for (int i = FLAGCX_THREAD_IDX_X; i < nPeers; i += FLAGCX_BLOCK_DIM_X) {
      int peerIdx = (1 + _teamRank + i) % _nTeamRanks;
      int targetRank = peerIdx * _stride + _localRank;
      // Signal remote peer: enqueue PrimSignal to FIFO
      // signalIdx = barrierSignal0 + myTeamRank (where peer expects my signal)
      int signalIdx = _barrierSignal0 + _teamRank;
      int combinedIdx = _contextId * _signalCount + signalIdx;
      uint64_t trdSpecific =
          ((uint64_t)0 << flagcxDeviceTriggerOffBufferType) |
          ((uint64_t)combinedIdx << flagcxDeviceTriggerOffSignalIdxSig) |
          ((uint64_t)1 << flagcxDeviceTriggerOffSignalValue);
      CommTraits<DefaultBackend<P>>::fifoEnqueue(
          _fifoBuffer, 0, 0,
          CommTraits<DefaultBackend<P>>::buildTrd(flagcxDevicePrimSignal,
                                                  targetRank, trdSpecific));
    }
    FLAGCX_DEVICE_SYNC_THREADS();
  }

  // wait: wait for all remote peers' signals at MY buffer
  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    if (_nTeamRanks <= 1)
      return;
    int nPeers = _nTeamRanks - 1;
    FLAGCX_DEVICE_SYNC_THREADS();
    // Each thread waits for a subset of peers (cooperative distribution)
    for (int i = FLAGCX_THREAD_IDX_X; i < nPeers; i += FLAGCX_BLOCK_DIM_X) {
      int peerIdx = (1 + _teamRank + i) % _nTeamRanks;
      // Wait for peer's signal at slot [barrierSignal0 + peerIdx]
      int signalIdx = _barrierSignal0 + peerIdx;
      int absIdx = _contextId * _signalCount + signalIdx;
      // Increment shadow (expected value) and spin on signal buffer
      _shadowBuffer[absIdx] += 1;
      uint64_t expected = _shadowBuffer[absIdx];
      uint64_t current =
          Atomic::load(&_signalBuffer[absIdx], flagcxDeviceMemoryOrderAcquire);
      int iter = 0;
      while (current < expected) {
        Intrin::spinBackoff(iter++);
        current = Atomic::load(&_signalBuffer[absIdx],
                               flagcxDeviceMemoryOrderAcquire);
      }
    }
    // Flush: ensure all signal FIFO entries from arrive() completed on wire
    if (FLAGCX_THREAD_IDX_X == 0 && _fifoBuffer != nullptr) {
      CommTraits<DefaultBackend<P>>::fifoFlush(_fifoBuffer);
    }
    FLAGCX_DEVICE_SYNC_THREADS();
  }

  // sync = arrive + wait
  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    arrive(order, fence);
    wait(order, fence);
  }
};

// ---- Barrier<DefaultBackend<P>, flagcxTeamTagWorld, Coop> ----
// Composes intra + inter barriers.
// Three-phase pattern for multi-node: intra → inter → intra.
// Single-node: just one intra sync.
template <typename P, typename Coop>
struct Barrier<DefaultBackend<P>, flagcxTeamTagWorld, Coop> {
  using Comm = typename CommTraits<DefaultBackend<P>>::Comm;
  using Team = typename CommTraits<DefaultBackend<P>>::Team;
  using Net = typename CommTraits<DefaultBackend<P>>::Net;

  Coop _coop;
  Barrier<DefaultBackend<P>, flagcxTeamTagIntra, Coop> _intra;
  Barrier<DefaultBackend<P>, flagcxTeamTagInter, Coop> _inter;
  int _nInterPeers;

  // World barrier: intra (IPC) + inter (FIFO Signal)
  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, flagcxTeamTagWorld, const Net &net, const Comm &dc,
          uint32_t index, bool multimem, int nInterPeers)
      : _coop(coop),
        _intra(coop, dc, Team{dc.intraSize, dc.intraRank, 1}, index),
        _inter(coop, net, dc, Team{}, index, nInterPeers),
        _nInterPeers(nInterPeers) {}

  // Intra-only barrier: inter is default constructed (no-op)
  FLAGCX_DEVICE_INLINE_DECORATOR
  Barrier(Coop coop, flagcxTeamTagIntra, const Net &, const Comm &dc,
          uint32_t index, bool multimem, int)
      : _coop(coop),
        _intra(coop, dc, Team{dc.intraSize, dc.intraRank, 1}, index), _inter(),
        _nInterPeers(0) {}

  // Inter-only barrier: intra is default constructed (no-op)
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
      _inter.arrive(order, fence);
    } else {
      _intra.arrive(order);
    }
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    if (_nInterPeers > 0) {
      _inter.wait(order, fence);
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
      // Phase 1: inter signal+wait (rank-to-rank across nodes)
      _inter.arrive(order, fence);
      _inter.wait(order, fence);
      // Phase 2: intra sync (broadcast inter completion to local ranks)
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
