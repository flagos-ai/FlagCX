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

#include "default_completion.h"
#include "fifo_producer_gate.h"
#include "flagcx_kernel_core.h"
#include <cassert>
#ifndef FLAGCX_DEVICE_COMPILE
#include "sym_heap.h"
#endif

// Default Net contract shared by every platform using DefaultBackend.
// A platform adapter supplies GDA resources and operation implementations.
enum DefaultNetProviderKind {
  DefaultNetProviderProxy = 0,
  DefaultNetProviderGda = 1
};

enum DefaultNetGdaCapability {
  DefaultNetGdaPut = 1u << 0,
  DefaultNetGdaGet = 1u << 1,
  DefaultNetGdaPutValue = 1u << 2,
  DefaultNetGdaSignal = 1u << 3,
  DefaultNetGdaFlush = 1u << 4,
  DefaultNetGdaSignalCompletion = 1u << 5,
  DefaultNetGdaCounterCompletion = 1u << 6,
  DefaultNetGdaBarrier = 1u << 7,
};

static constexpr unsigned DefaultNetGdaRequiredCapabilities =
    DefaultNetGdaPut | DefaultNetGdaGet | DefaultNetGdaPutValue |
    DefaultNetGdaSignal | DefaultNetGdaFlush | DefaultNetGdaSignalCompletion |
    DefaultNetGdaCounterCompletion | DefaultNetGdaBarrier;

struct DefaultNetContext {
  DefaultNetProviderKind provider;
  unsigned providerId; // Platform-defined GDA implementation; zero for Proxy.
  void *handle;
  int contextId;
};

template <DefaultNetProviderKind Provider, typename Operation>
struct DefaultNetApi;

template <typename Operation>
struct DefaultNetApi<DefaultNetProviderProxy, Operation> {
  template <typename... Args>
  FLAGCX_DEVICE_INLINE_DECORATOR static auto call(DefaultNetContext ctx,
                                                  Args &&...args)
      -> decltype(Operation::proxy(ctx, static_cast<Args &&>(args)...)) {
    return Operation::proxy(ctx, static_cast<Args &&>(args)...);
  }
};

// Each platform may specialize this for its compiled GDA implementations.
// An unavailable provider traps if a malformed context reaches device code.
template <typename PlatformTag>
struct DefaultNetGdaProvider {
  FLAGCX_HOST_DEVICE_INLINE static constexpr bool compiled(unsigned) {
    return false;
  }

  template <typename Operation, typename... Args>
  FLAGCX_DEVICE_INLINE_DECORATOR static auto call(DefaultNetContext ctx,
                                                  Args &&...args)
      -> decltype(DefaultNetApi<DefaultNetProviderProxy, Operation>::call(
          ctx, static_cast<Args &&>(args)...)) {
    FLAGCX_DEVICE_TRAP();
  }
};

// Host chooses a provider before publishing the Comm device view. Capability
// bits cover the complete one-sided and Inter Barrier operation contract.
template <typename PlatformTag>
FLAGCX_HOST_DEVICE_INLINE constexpr DefaultNetProviderKind
defaultNetSelectProvider(unsigned providerId, unsigned gdaCapabilities,
                         bool resourcesReady) {
  return DefaultNetGdaProvider<PlatformTag>::compiled(providerId) &&
                 resourcesReady &&
                 (gdaCapabilities & DefaultNetGdaRequiredCapabilities) ==
                     DefaultNetGdaRequiredCapabilities
             ? DefaultNetProviderGda
             : DefaultNetProviderProxy;
}

template <typename PlatformTag>
struct DefaultNetProviderPolicy {
  struct WindowState {};
  struct CommState {};
  struct BarrierState {};

  template <typename Window>
  FLAGCX_HOST_DEVICE_INLINE static void initWindow(Window &) {}

  template <typename Comm>
  FLAGCX_HOST_DEVICE_INLINE static void initComm(Comm &) {}

  template <typename Comm>
  FLAGCX_HOST_DEVICE_INLINE static DefaultNetContext
  makeContext(const Comm &, void *fifo, int contextId) {
    return DefaultNetContext{DefaultNetProviderProxy, 0, fifo, contextId};
  }

  template <typename Barrier>
  FLAGCX_HOST_DEVICE_INLINE static void initBarrier(Barrier &,
                                                    DefaultNetContext) {}

  template <typename Barrier>
  FLAGCX_HOST_DEVICE_INLINE static DefaultNetContext
  barrierContext(const Barrier &, void *fifo, int contextId) {
    return DefaultNetContext{DefaultNetProviderProxy, 0, fifo, contextId};
  }
};

// The same runtime dispatch applies to every DefaultBackend platform. A
// platform-specific GDA provider handles only its own native implementation.
template <typename PlatformTag>
struct DefaultNetDispatch {
  template <typename Op, typename... Args>
  FLAGCX_DEVICE_INLINE_DECORATOR static auto call(DefaultNetContext ctx,
                                                  Args &&...args)
      -> decltype(DefaultNetApi<DefaultNetProviderProxy, Op>::call(
          ctx, static_cast<Args &&>(args)...)) {
    switch (ctx.provider) {
      case DefaultNetProviderProxy:
        return DefaultNetApi<DefaultNetProviderProxy, Op>::call(
            ctx, static_cast<Args &&>(args)...);
      case DefaultNetProviderGda:
        if (!DefaultNetGdaProvider<PlatformTag>::compiled(ctx.providerId) ||
            ctx.handle == nullptr)
          FLAGCX_DEVICE_TRAP();
        return DefaultNetGdaProvider<PlatformTag>::template call<Op>(
            ctx, static_cast<Args &&>(args)...);
      default:
        FLAGCX_DEVICE_TRAP();
    }
  }
};

template <typename PlatformTag>
struct CommTraits<DefaultBackend<PlatformTag>> {
  // Platform capabilities (resolved via PlatformTag)
  using Intrin = typename PlatformTraits<PlatformTag>::Intrin;
  using Atomic = typename PlatformTraits<PlatformTag>::Atomic;
  using CoopOps = typename PlatformTraits<PlatformTag>::CoopOps;
  using Clock = typename PlatformTraits<PlatformTag>::Clock;
  using CompletionWord = typename PlatformCompletionWord<PlatformTag>::type;
  using CompletionStorage = DefaultCompletionStorage<CompletionWord>;
  static_assert(sizeof(CompletionWord) == sizeof(uint32_t) ||
                    sizeof(CompletionWord) == sizeof(uint64_t),
                "DefaultBackend completion word must be 32 or 64 bits");
  static constexpr int completionBits = sizeof(CompletionWord) * 8;
  static constexpr int defaultCounterBits =
      completionBits < 56 ? completionBits : 56;

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
  struct SymPeerLayout {
    size_t userOffset;
    size_t userSize;
  };
  struct Window : DefaultNetProviderPolicy<PlatformTag>::WindowState {
    WindowMode mode;   // SYMMETRIC (VMM) or ASYMMETRIC (IPC)
    void *flatBasePtr; // Flat VA base (SYMMETRIC mode, nullable)
    size_t allocSize;  // Per-rank allocation size (SYMMETRIC mode)
    FLAGCX_DEVICE_GLOBAL_PTR const SymPeerLayout *symPeerLayouts;
    void *mcBasePtr;    // Multicast base (nullable, SYMMETRIC mode only)
    void **ipcBasePtrs; // IPC peer pointers (ASYMMETRIC mode, nullable)
    int intraRank;      // Local rank index
    int intraSize;      // Number of local ranks (bounds for ipcBasePtrs)
    uintptr_t mrBase;   // MR base VA (inter-node, orthogonal to mode)
    int mrIndex;        // MR table index (-1 if none)
    void *rawPtr;       // Raw pointer fallback (for getLocalPointer)

    FLAGCX_DEVICE_INLINE_DECORATOR void *getSegmentedPointer(size_t offset,
                                                             int index) const {
      if (index < 0 || index >= intraSize)
        return nullptr;
      if (offset >= symPeerLayouts[index].userSize)
        return nullptr;
      return (char *)flatBasePtr + (size_t)index * allocSize +
             symPeerLayouts[index].userOffset + offset;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void *
    getPeerPointer(size_t offset, const Team &team, int peer) const {
      if (mode == SYMMETRIC && flatBasePtr) {
        int index = team.rank + (peer - team.rank) * team.stride;
        if (symPeerLayouts)
          return getSegmentedPointer(offset, index);
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

    // IPC/VMM mappings support all declared access contracts.
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
      if (mode == SYMMETRIC && flatBasePtr) {
        if (symPeerLayouts)
          return getSegmentedPointer(offset, intraRank);
        return (char *)flatBasePtr + (size_t)intraRank * allocSize + offset;
      } else if (ipcBasePtrs && intraRank >= 0 && intraRank < intraSize &&
                 ipcBasePtrs[intraRank])
        return (char *)ipcBasePtrs[intraRank] + offset;
      return (char *)rawPtr + offset;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void *getIntraPointer(size_t offset,
                                                         int peer) const {
      if (mode == SYMMETRIC && flatBasePtr) {
        if (symPeerLayouts)
          return getSegmentedPointer(offset, peer);
        return (char *)flatBasePtr + (size_t)peer * allocSize + offset;
      } else if (ipcBasePtrs && peer >= 0 && peer < intraSize &&
                 ipcBasePtrs[peer])
        return (char *)ipcBasePtrs[peer] + offset;
      return nullptr;
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

#ifndef FLAGCX_DEVICE_COMPILE
    // Host-side population from flagcxWindow_t (sym heap or IPC).
    void populateFromHost(flagcxWindow_t win, void *rawPtr_, int intraRank_,
                          int intraSize_, int mrIndex_, uintptr_t mrBase_,
                          int ipcIndex_, void **ipcDevPeerPtrs_) {
      DefaultNetProviderPolicy<PlatformTag>::initWindow(*this);
      rawPtr = rawPtr_;
      intraRank = intraRank_;
      intraSize = intraSize_;
      mrBase = mrBase_;
      mrIndex = mrIndex_;

      flagcxSymWindow_t d =
          (win && win->isSymmetricDefault) ? win->defaultBase : nullptr;

      if (d && d->hasFlatMapping && d->flatBase) {
        mode = SYMMETRIC;
        flatBasePtr = d->flatBase;
        allocSize = d->rankLayouts != nullptr ? d->slotStride : d->allocSize;
        symPeerLayouts = nullptr;
        mcBasePtr = d->mcBase;
        ipcBasePtrs = nullptr;
      } else {
        mode = ASYMMETRIC;
        flatBasePtr = nullptr;
        allocSize = 0;
        symPeerLayouts = nullptr;
        mcBasePtr = nullptr;
        ipcBasePtrs = (ipcIndex_ >= 0) ? ipcDevPeerPtrs_ : nullptr;
      }
    }
#endif // FLAGCX_DEVICE_COMPILE
  };

  // ---- Comm: All fallback layers ----
  struct Comm : CompletionStorage,
                DefaultNetProviderPolicy<PlatformTag>::CommState {
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
    uint64_t *localSignalBuffer;
    uint64_t *shadowBuffer;
    uint64_t *counterBuffer;
    int signalCount;
    int counterCount;
    int contextCount;

    int netOneSidedReady;
    int netSignalReady;
    int netPutValueReady;
    int useP2pSignals;

    // Direct local signal delivery through IPC or mapped SHM.
    // signalPeerPtrs[peer] → peer's direct signal plane when reachable
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

    FLAGCX_DEVICE_INLINE_DECORATOR CompletionWord *
    getCompletionSignalPeerPtr(int peer) const {
      return this->getDirectSignalPeerPtr(*this, peer);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR CompletionWord *
    getCompletionSignalBuffer() const {
      return this->getDirectSignalBuffer(*this);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR CompletionWord *
    getCompletionCounterBuffer() const {
      return this->getDirectCounterBuffer(*this);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR bool usesDirectP2pSignals() const {
      return useP2pSignals != 0;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR bool isOneSidedTransportReady() const {
      return netOneSidedReady != 0;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR bool supportsDirectCounterAccess() const {
      return getCompletionCounterBuffer() != nullptr;
    }

    // Populate from host-side handle (deferred template avoids forward-decl)
    template <typename DI>
    static FLAGCX_HOST_DEVICE_INLINE void populateFromInternal(Comm &dc,
                                                               const DI &di) {
      DefaultNetProviderPolicy<PlatformTag>::initComm(dc);
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
      dc.localSignalBuffer = di.localSignalBuffer;
      dc.shadowBuffer = di.shadowBuffer;
      dc.counterBuffer =
          di.counterDeviceBuffer ? di.counterDeviceBuffer : di.counterBuffer;
      dc.populateCompletion(di);
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
  // Proxy transport: FIFO encoding, producer gate, and completion domain.
  // ============================================================
  struct ProxyProvider {

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

    FLAGCX_DEVICE_INLINE_DECORATOR static bool
    fifoWordBefore(CompletionWord lhs, CompletionWord rhs) {
      return CompletionStorage::before(lhs, rhs);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static flagcxResult_t
    fifoTerminalStatus(uint64_t *buffer) {
      return static_cast<flagcxResult_t>(
          Atomic::load(&buffer[flagcxFifoIdxTerminalStatus],
                       flagcxDeviceMemoryOrderAcquire));
    }

    // Enqueue a trigger into the device FIFO buffer.
    // Atomically reserves a slot, waits for space, writes 3 words.
    FLAGCX_DEVICE_INLINE_DECORATOR
    static flagcxResult_t fifoEnqueue(void *fifoBuffer, uint64_t fstVal,
                                      uint64_t sndVal, uint64_t trdVal) {
      uint64_t *buffer = (uint64_t *)fifoBuffer;
      CompletionWord *producerState = reinterpret_cast<CompletionWord *>(
          &buffer[flagcxFifoIdxProducerState]);
      if (!flagcxFifoProducerTryEnter<Atomic>(producerState)) {
        flagcxResult_t terminal = fifoTerminalStatus(buffer);
        return terminal == flagcxSuccess ? flagcxInternalError : terminal;
      }
      uint64_t capacity = Atomic::load(&buffer[flagcxFifoIdxCapacity],
                                       flagcxDeviceMemoryOrderRelaxed);

      // 1. Atomically reserve a slot
      CompletionWord *produced =
          reinterpret_cast<CompletionWord *>(&buffer[flagcxFifoIdxProduced]);
      CompletionWord *consumed =
          reinterpret_cast<CompletionWord *>(&buffer[flagcxFifoIdxConsumed]);
      CompletionWord mySlot = Atomic::fetchAdd(produced, CompletionWord{1},
                                               flagcxDeviceMemoryOrderAcqRel);

      // 2. Wait until there's space (mySlot - consumed < capacity)
      int iter = 0;
      while ((CompletionWord)(mySlot -
                              Atomic::load(consumed,
                                           flagcxDeviceMemoryOrderAcquire)) >=
             (CompletionWord)capacity) {
        flagcxResult_t terminal = fifoTerminalStatus(buffer);
        if (terminal != flagcxSuccess) {
          flagcxFifoProducerLeave<Atomic>(producerState);
          return terminal;
        }
        Intrin::spinBackoff(iter++);
      }

      // A terminal error may be published after the slot reservation but before
      // the FIFO becomes full. Leave the sequence hole unpublished; terminal
      // cleanup waits for this producer reference and then advances all failed
      // reservations to the stable produced value.
      flagcxResult_t terminal = fifoTerminalStatus(buffer);
      if (terminal != flagcxSuccess) {
        flagcxFifoProducerLeave<Atomic>(producerState);
        return terminal;
      }

      // 3. Compute slot index and get pointers to slot's 3 uint64_t fields
      uint64_t idx = (uint64_t)mySlot % capacity;
      uint64_t *slotFst =
          buffer + flagcxFifoIdxData +
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

      flagcxFifoProducerLeave<Atomic>(producerState);
      return flagcxSuccess;
    }

    // Flush: snapshot produced, then spin until completed >= snapshot.
    // The CPU proxy advances 'completed' after IB test() confirms each op done.
    // This replaces the old PrimWait+streamSynchronize approach that caused
    // deadlocks.
    FLAGCX_DEVICE_INLINE_DECORATOR
    static flagcxResult_t fifoFlush(void *fifoBuffer) {
      uint64_t *buffer = (uint64_t *)fifoBuffer;
      flagcxResult_t terminal = fifoTerminalStatus(buffer);
      if (terminal != flagcxSuccess)
        return terminal;
      CompletionWord *produced =
          reinterpret_cast<CompletionWord *>(&buffer[flagcxFifoIdxProduced]);
      CompletionWord *completed =
          reinterpret_cast<CompletionWord *>(&buffer[flagcxFifoIdxCompleted]);
      CompletionWord snapshot =
          Atomic::load(produced, flagcxDeviceMemoryOrderAcquire);
      int iter = 0;
      while (fifoWordBefore(
          Atomic::load(completed, flagcxDeviceMemoryOrderAcquire), snapshot)) {
        terminal = fifoTerminalStatus(buffer);
        if (terminal != flagcxSuccess)
          return terminal;
        Intrin::spinBackoff(iter++);
      }
      return flagcxSuccess;
    }

    // Wait: just flush (no PrimWait enqueue needed with completion-based
    // flush).
    FLAGCX_DEVICE_INLINE_DECORATOR
    static flagcxResult_t fifoWait(void *fifoBuffer) {
      return fifoFlush(fifoBuffer);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static flagcxResult_t
    enqueueSignal(void *fifoBuffer, int contextId, int signalCount,
                  int counterCount, int signalIdx, uint32_t value, int peer,
                  uint64_t bufferType) {
      int combinedIdx = bufferType == 0 ? contextId * signalCount + signalIdx
                                        : contextId * counterCount + signalIdx;
      uint64_t trdSpecific =
          (bufferType << flagcxDeviceTriggerOffBufferType) |
          ((uint64_t)combinedIdx << flagcxDeviceTriggerOffSignalIdxSig) |
          ((uint64_t)(value & 0xFFFFu) << flagcxDeviceTriggerOffSignalValue);
      return fifoEnqueue(fifoBuffer, 0, 0,
                         buildTrd(flagcxDevicePrimSignal, peer, trdSpecific));
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static void
    waitSignalValue(void *fifoBuffer, uint64_t *signalBuffer, int index,
                    uint64_t expected) {
      uint64_t current =
          Atomic::load(&signalBuffer[index], flagcxDeviceMemoryOrderAcquire);
      int iter = 0;
      while (current < expected) {
        if (fifoBuffer != nullptr &&
            Atomic::load(&((uint64_t *)fifoBuffer)[flagcxFifoIdxTerminalStatus],
                         flagcxDeviceMemoryOrderAcquire) != flagcxSuccess)
          break;
        Intrin::spinBackoff(iter++);
        current =
            Atomic::load(&signalBuffer[index], flagcxDeviceMemoryOrderAcquire);
      }
    }
  };

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

    // Operation tags follow NCCL GIN's per-operation dispatch pattern. The
    // Proxy implementation remains the only compiled specialization for now.
    struct PutOp {
      template <typename NetT, typename... Args>
      FLAGCX_DEVICE_INLINE_DECORATOR static void
      proxy(DefaultNetContext, const NetT &net, Args &&...args) {
        net.proxyPut(static_cast<Args &&>(args)...);
      }
    };
    struct GetOp {
      template <typename NetT, typename... Args>
      FLAGCX_DEVICE_INLINE_DECORATOR static void
      proxy(DefaultNetContext, const NetT &net, Args &&...args) {
        net.proxyGet(static_cast<Args &&>(args)...);
      }
    };
    struct PutValueOp {
      template <typename NetT, typename... Args>
      FLAGCX_DEVICE_INLINE_DECORATOR static void
      proxy(DefaultNetContext, const NetT &net, Args &&...args) {
        net.proxyPutValue(static_cast<Args &&>(args)...);
      }
    };
    struct SignalOp {
      template <typename NetT, typename... Args>
      FLAGCX_DEVICE_INLINE_DECORATOR static void
      proxy(DefaultNetContext, const NetT &net, Args &&...args) {
        net.proxySignal(static_cast<Args &&>(args)...);
      }
    };
    struct FlushOp {
      template <typename NetT, typename... Args>
      FLAGCX_DEVICE_INLINE_DECORATOR static void
      proxy(DefaultNetContext, const NetT &net, Args &&...args) {
        net.proxyFlush(static_cast<Args &&>(args)...);
      }
    };
    struct WaitSignalOp {
      template <typename NetT, typename... Args>
      FLAGCX_DEVICE_INLINE_DECORATOR static void
      proxy(DefaultNetContext, const NetT &net, Args &&...args) {
        net.proxyWaitSignal(static_cast<Args &&>(args)...);
      }
    };
    struct ReadSignalOp {
      template <typename NetT, typename... Args>
      FLAGCX_DEVICE_INLINE_DECORATOR static uint64_t
      proxy(DefaultNetContext, const NetT &net, Args &&...args) {
        return net.proxyReadSignal(static_cast<Args &&>(args)...);
      }
    };
    struct ResetSignalOp {
      template <typename NetT, typename... Args>
      FLAGCX_DEVICE_INLINE_DECORATOR static void
      proxy(DefaultNetContext, const NetT &net, Args &&...args) {
        net.proxyResetSignal(static_cast<Args &&>(args)...);
      }
    };
    struct WaitCounterOp {
      template <typename NetT, typename... Args>
      FLAGCX_DEVICE_INLINE_DECORATOR static void
      proxy(DefaultNetContext, const NetT &net, Args &&...args) {
        net.proxyWaitCounter(static_cast<Args &&>(args)...);
      }
    };
    struct ReadCounterOp {
      template <typename NetT, typename... Args>
      FLAGCX_DEVICE_INLINE_DECORATOR static uint64_t
      proxy(DefaultNetContext, const NetT &net, Args &&...args) {
        return net.proxyReadCounter(static_cast<Args &&>(args)...);
      }
    };
    struct ResetCounterOp {
      template <typename NetT, typename... Args>
      FLAGCX_DEVICE_INLINE_DECORATOR static void
      proxy(DefaultNetContext, const NetT &net, Args &&...args) {
        net.proxyResetCounter(static_cast<Args &&>(args)...);
      }
    };

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

    FLAGCX_DEVICE_INLINE_DECORATOR DefaultNetContext context() const {
      return DefaultNetProviderPolicy<PlatformTag>::makeContext(_dc, fifoBuffer,
                                                                _contextId);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR CompletionWord *
    getSignalPtr(flagcxDevSignal_t signalId) const {
      return &_dc.getCompletionSignalBuffer()[_contextId * signalCount +
                                              (int)signalId];
    }

    FLAGCX_DEVICE_INLINE_DECORATOR CompletionWord *
    getPeerSignalPtr(int localPeer, flagcxDevSignal_t signalId) const {
      CompletionWord *peerBuffer = _dc.getCompletionSignalPeerPtr(localPeer);
      return peerBuffer ? &peerBuffer[_contextId * signalCount + (int)signalId]
                        : nullptr;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR CompletionWord *
    getCounterPtr(flagcxDevCounter_t counterId) const {
      return &_dc.getCompletionCounterBuffer()[_contextId * counterCount +
                                               (int)counterId];
    }

    FLAGCX_DEVICE_INLINE_DECORATOR bool isIntraPeer(int peer) const {
      int intraBase = _dc.rank - _dc.intraRank;
      return peer >= intraBase && peer < intraBase + _dc.intraSize;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR bool isValid() const {
      return context().provider == DefaultNetProviderProxy &&
             _dc.getCompletionSignalBuffer() != nullptr &&
             _dc.getCompletionCounterBuffer() != nullptr &&
             fifoBuffer != nullptr;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR CompletionWord
    loadSignalValue(int idx, flagcxDeviceMemoryOrder_t order) const {
      CompletionWord value = CompletionStorage::template loadCompletion<Atomic>(
          _dc.getCompletionSignalBuffer(), signalBuffer, idx, order);
      if (_dc.localSignalBuffer != nullptr)
        value += static_cast<CompletionWord>(
            Atomic::load(&_dc.localSignalBuffer[idx], order));
      return value;
    }

    FLAGCX_DEVICE_INLINE_DECORATOR CompletionWord
    loadCounterValue(int idx, flagcxDeviceMemoryOrder_t order) const {
      return CompletionStorage::template loadCompletion<Atomic>(
          _dc.getCompletionCounterBuffer(), counterBuffer, idx, order);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    completeCounter(flagcxDevCounter_t counterId, uint64_t delta) const {
      CompletionWord *ptr = getCounterPtr(counterId);
      Atomic::fetchAdd(ptr, CompletionStorage::completionValue(delta),
                       flagcxDeviceMemoryOrderRelease);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static void
    validateCompletionValue(uint64_t value) {
      // The public ABI stays uint64_t. A 32-bit completion specialization
      // rejects values it cannot represent instead of truncating them.
      CompletionStorage::validate(value);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR static void
    validateCompletionBits(int bits) {
      CompletionStorage::validateBits(bits);
    }

    // ---- Two-sided FIFO encoders ----
    FLAGCX_DEVICE_INLINE_DECORATOR void
    enqueueFifoSend(const Window &mem, size_t offset, size_t count,
                    flagcxDataType_t datatype, int peer) const {
      void *ptr = mem.getLocalPointer(offset);
      ProxyProvider::fifoEnqueue(
          fifoBuffer, (uint64_t)((uintptr_t)ptr), 0,
          ProxyProvider::buildTrd(
              flagcxDevicePrimSend, peer,
              ((uint64_t)datatype << flagcxDeviceTriggerOffDatatype) |
                  ((uint64_t)count << flagcxDeviceTriggerOffCount)));
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    enqueueFifoRecv(const Window &mem, size_t offset, size_t count,
                    flagcxDataType_t datatype, int peer) const {
      void *ptr = mem.getLocalPointer(offset);
      ProxyProvider::fifoEnqueue(
          fifoBuffer, (uint64_t)((uintptr_t)ptr), 0,
          ProxyProvider::buildTrd(
              flagcxDevicePrimRecv, peer,
              ((uint64_t)datatype << flagcxDeviceTriggerOffDatatype) |
                  ((uint64_t)count << flagcxDeviceTriggerOffCount)));
    }

    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t
    enqueueFifoTerm(int totalCoops) const {
      return ProxyProvider::fifoEnqueue(
          fifoBuffer, (uint64_t)totalCoops, 0,
          ProxyProvider::buildTrd(flagcxDevicePrimTerm, 0, 0));
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
        ProxyProvider::fifoWait(fifoBuffer);
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
      return ProxyProvider::fifoEnqueue(
          fifoBuffer, fstValue, sndValue,
          ProxyProvider::buildTrd(flagcxDevicePrimPut, peer, trdSpecific));
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
      return ProxyProvider::fifoEnqueue(
          fifoBuffer, fstValue, sndValue,
          ProxyProvider::buildTrd(flagcxDevicePrimGet, peer, trdSpecific));
    }

    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t
    enqueueFifoSignalRaw(int signalIdx, int peer) const {
      return ProxyProvider::enqueueSignal(fifoBuffer, _contextId, signalCount,
                                          counterCount, signalIdx, 1, peer, 0);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t enqueueFifoSignal(
        int signalIdx, uint32_t value, int peer, uint64_t bufferType) const {
      return ProxyProvider::enqueueSignal(fifoBuffer, _contextId, signalCount,
                                          counterCount, signalIdx, value, peer,
                                          bufferType);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t enqueueFifoSignalValue(
        int signalIdx, uint64_t value, int peer, uint64_t bufferType) const {
      int combinedIdx = (bufferType == 0)
                            ? (_contextId * signalCount + signalIdx)
                            : (_contextId * counterCount + signalIdx);
      uint64_t trdSpecific =
          ((uint64_t)bufferType << flagcxDeviceTriggerOffBufferType) |
          ((uint64_t)combinedIdx << flagcxDeviceTriggerOffSignalIdxSig);
      return ProxyProvider::fifoEnqueue(
          fifoBuffer, 0, value,
          ProxyProvider::buildTrd(flagcxDevicePrimSignalValue, peer,
                                  trdSpecific));
    }

    FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t enqueueFifoPutValue(
        size_t dstOffset, uint64_t value, int peer, int dstMrIdx) const {
      uint64_t fstValue = (uint64_t)dstOffset &
                          flagcxTriggerMask(flagcxDeviceTriggerBitsDstOffset);
      uint64_t trdSpecific = (uint64_t)dstMrIdx
                             << flagcxDeviceTriggerOffDstMrIdx;
      return ProxyProvider::fifoEnqueue(
          fifoBuffer, fstValue, value,
          ProxyProvider::buildTrd(flagcxDevicePrimPutValue, peer, trdSpecific));
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
      return ProxyProvider::fifoEnqueue(
          fifoBuffer, fstValue, sndValue,
          ProxyProvider::buildTrd(flagcxDevicePrimPutSignal, peer,
                                  trdSpecific));
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
    FLAGCX_DEVICE_INLINE_DECORATOR constexpr uint64_t getSignalValue(T) const {
      return 0;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR constexpr uint64_t
    getSignalValue(flagcxDevNet_SignalInc) const {
      return 1;
    }
    FLAGCX_DEVICE_INLINE_DECORATOR constexpr uint64_t
    getSignalValue(flagcxDevNet_SignalAdd a) const {
      return a.value;
    }

    template <typename T>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    enqueueFifoSignalAction(T action, int peer, uint64_t bufferType) const {
      uint64_t value = getSignalValue(action);
      if (value <= flagcxTriggerMask(flagcxDeviceTriggerBitsSignalValue))
        enqueueFifoSignal(getSignalIdx(action), (uint32_t)value, peer,
                          bufferType);
      else
        enqueueFifoSignalValue(getSignalIdx(action), value, peer, bufferType);
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
    canFuseSignal(flagcxDevNet_SignalAdd a) const {
      return a.value <=
             flagcxTriggerMask(flagcxDeviceTriggerBitsSignalValuePut);
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

    // ---- One-sided: public dispatch and Proxy implementation ----
    template <typename RA, typename LA, typename Coop, typename Desc>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    put(Team team, int peer, Window dst, size_t dstOff, Window src,
        size_t srcOff, size_t bytes, RA ra, LA la, Coop coop, Desc desc,
        flagcxDeviceScope_t ar, flagcxDeviceScope_t es) const {
      DefaultNetDispatch<PlatformTag>::template call<PutOp>(
          context(), *this, team, peer, dst, dstOff, src, srcOff, bytes, ra, la,
          coop, desc, ar, es);
    }

    template <typename RA, typename LA, typename Coop, typename Desc>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    proxyPut(Team team, int peer, Window dst, size_t dstOff, Window src,
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
                               (uint32_t)getSignalValue(ra), worldPeer,
                               src.getMrIndex(), dst.getMrIndex());
        } else {
          enqueueFifoPut(srcDataOff, dstDataOff, bytes, worldPeer,
                         src.getMrIndex(), dst.getMrIndex());
          if (isSignal(ra)) {
            validateCompletionValue(getSignalValue(ra));
            enqueueFifoSignalAction(ra, worldPeer, 0);
          }
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
      DefaultNetDispatch<PlatformTag>::template call<GetOp>(
          context(), *this, team, peer, src, srcOff, dst, dstOff, bytes, coop);
    }

    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    proxyGet(Team team, int peer, Window src, size_t srcOff, Window dst,
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
      DefaultNetDispatch<PlatformTag>::template call<PutValueOp>(
          context(), *this, team, peer, dst, dstOff, value, ra, coop, desc, ar,
          es);
    }

    template <typename T, typename RA, typename Coop, typename Desc>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    proxyPutValue(Team team, int peer, Window dst, size_t dstOff, T value,
                  RA ra, Coop coop, Desc desc, flagcxDeviceScope_t ar,
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
        if (isSignal(ra)) {
          validateCompletionValue(getSignalValue(ra));
          enqueueFifoSignalAction(ra, worldPeer, 0);
        }
      }
      coop.sync();
    }

    // ---- One-sided: signal ----
    template <typename RA, typename Coop, typename Desc>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    signal(Team team, int peer, RA ra, Coop coop, Desc desc,
           flagcxDeviceScope_t ar, flagcxDeviceScope_t es) const {
      DefaultNetDispatch<PlatformTag>::template call<SignalOp>(
          context(), *this, team, peer, ra, coop, desc, ar, es);
    }

    template <typename RA, typename Coop, typename Desc>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    proxySignal(Team team, int peer, RA ra, Coop coop, Desc desc,
                flagcxDeviceScope_t ar, flagcxDeviceScope_t es) const {
      (void)desc;
      (void)ar;
      (void)es;
      int worldPeer = teamRankToWorld(team, peer);
      coop.sync();
      if (coop.threadRank() == 0) {
        if (isSignal(ra)) {
          validateCompletionValue(getSignalValue(ra));
          int localPeer = worldPeer - (_dc.rank - _dc.intraRank);
          CompletionWord *peerSignal =
              (localPeer >= 0 && localPeer < _dc.intraSize)
                  ? getPeerSignalPtr(localPeer,
                                     (flagcxDevSignal_t)getSignalIdx(ra))
                  : nullptr;
          bool useDirect = CompletionStorage::useDirectSignal(
              peerSignal != nullptr, _dc.usesDirectP2pSignals());
          if (useDirect)
            Atomic::fetchAdd(
                peerSignal,
                CompletionStorage::completionValue(getSignalValue(ra)),
                flagcxDeviceMemoryOrderRelease);
          else
            enqueueFifoSignalAction(ra, worldPeer, 0);
        }
      }
      coop.sync();
    }

    // ---- flush: drain FIFO (snapshot-spin, no PrimWait) ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    flush(Coop coop, flagcxDeviceMemoryOrder_t order) const {
      DefaultNetDispatch<PlatformTag>::template call<FlushOp>(context(), *this,
                                                              coop, order);
    }

    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    proxyFlush(Coop coop, flagcxDeviceMemoryOrder_t order) const {
      (void)order;
      coop.sync();
      if (coop.threadRank() == 0 && fifoBuffer != nullptr) {
        ProxyProvider::fifoFlush(fifoBuffer);
      }
      coop.sync();
    }

    // ---- waitSignal: GPU spin on the combined local and network signals ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitSignal(Coop coop, flagcxDevSignal_t signalId, uint64_t least, int bits,
               flagcxDeviceMemoryOrder_t order) const {
      DefaultNetDispatch<PlatformTag>::template call<WaitSignalOp>(
          context(), *this, coop, signalId, least, bits, order);
    }

    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    proxyWaitSignal(Coop coop, flagcxDevSignal_t signalId, uint64_t least,
                    int bits, flagcxDeviceMemoryOrder_t order) const {
      validateCompletionBits(bits);
      CompletionWord target = CompletionStorage::completionValue(least);
      coop.sync();
      if (coop.threadRank() == 0) {
        int idx = _contextId * signalCount + (int)signalId;
        int iter = 0;
        while (CompletionStorage::waitBefore(loadSignalValue(idx, order),
                                             target)) {
          if (fifoBuffer != nullptr &&
              ProxyProvider::fifoTerminalStatus((uint64_t *)fifoBuffer) !=
                  flagcxSuccess)
            break;
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
      CompletionWord target =
          CompletionStorage::advance(shadow, static_cast<uint64_t>(delta));
      waitSignal(coop, signalId, static_cast<uint64_t>(target), bits, order);
      shadowBuffer[idx] = static_cast<uint64_t>(target);
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
      int idx = _contextId * signalCount + (int)signalId;
      CompletionWord next =
          CompletionStorage::advance(shadowBuffer[idx], delta);
      shadowBuffer[idx] = static_cast<uint64_t>(next);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
    readSignal(flagcxDevSignal_t signalId, int bits,
               flagcxDeviceMemoryOrder_t order) const {
      return DefaultNetDispatch<PlatformTag>::template call<ReadSignalOp>(
          context(), *this, signalId, bits, order);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
    proxyReadSignal(flagcxDevSignal_t signalId, int bits,
                    flagcxDeviceMemoryOrder_t order) const {
      validateCompletionBits(bits);
      int idx = _contextId * signalCount + (int)signalId;
      return static_cast<uint64_t>(loadSignalValue(idx, order));
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    resetSignal(flagcxDevSignal_t signalId) const {
      DefaultNetDispatch<PlatformTag>::template call<ResetSignalOp>(
          context(), *this, signalId);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    proxyResetSignal(flagcxDevSignal_t signalId) const {
      int idx = _contextId * signalCount + (int)signalId;
      Atomic::store(&_dc.getCompletionSignalBuffer()[idx], CompletionWord{0},
                    flagcxDeviceMemoryOrderRelease);
      if (_dc.localSignalBuffer != nullptr)
        Atomic::store(&_dc.localSignalBuffer[idx], uint64_t{0},
                      flagcxDeviceMemoryOrderRelease);
      CompletionStorage::template resetProxy<Atomic>(
          signalBuffer, idx, flagcxDeviceMemoryOrderRelease);
      Atomic::store(&shadowBuffer[idx], (uint64_t)0,
                    flagcxDeviceMemoryOrderRelease);
    }

    // ---- Counter: GPU spin on counterBuffer[ctx*N+id] ----
    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    waitCounter(Coop coop, flagcxDevCounter_t counterId, uint64_t least,
                int bits, flagcxDeviceMemoryOrder_t order) const {
      DefaultNetDispatch<PlatformTag>::template call<WaitCounterOp>(
          context(), *this, coop, counterId, least, bits, order);
    }

    template <typename Coop>
    FLAGCX_DEVICE_INLINE_DECORATOR void
    proxyWaitCounter(Coop coop, flagcxDevCounter_t counterId, uint64_t least,
                     int bits, flagcxDeviceMemoryOrder_t order) const {
      validateCompletionBits(bits);
      CompletionWord target = CompletionStorage::completionValue(least);
      coop.sync();
      if (coop.threadRank() == 0) {
        int idx = _contextId * counterCount + (int)counterId;
        int iter = 0;
        while (CompletionStorage::waitBefore(loadCounterValue(idx, order),
                                             target)) {
          if (fifoBuffer != nullptr &&
              ProxyProvider::fifoTerminalStatus((uint64_t *)fifoBuffer) !=
                  flagcxSuccess)
            break;
          Intrin::spinBackoff(iter++);
        }
      }
      coop.sync();
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
    readCounter(flagcxDevCounter_t counterId, int bits,
                flagcxDeviceMemoryOrder_t order) const {
      return DefaultNetDispatch<PlatformTag>::template call<ReadCounterOp>(
          context(), *this, counterId, bits, order);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
    proxyReadCounter(flagcxDevCounter_t counterId, int bits,
                     flagcxDeviceMemoryOrder_t order) const {
      validateCompletionBits(bits);
      int idx = _contextId * counterCount + (int)counterId;
      return static_cast<uint64_t>(loadCounterValue(idx, order));
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    resetCounter(flagcxDevCounter_t counterId) const {
      DefaultNetDispatch<PlatformTag>::template call<ResetCounterOp>(
          context(), *this, counterId);
    }

    FLAGCX_DEVICE_INLINE_DECORATOR void
    proxyResetCounter(flagcxDevCounter_t counterId) const {
      int idx = _contextId * counterCount + (int)counterId;
      Atomic::store(&_dc.getCompletionCounterBuffer()[idx], CompletionWord{0},
                    flagcxDeviceMemoryOrderRelease);
      CompletionStorage::template resetProxy<Atomic>(
          counterBuffer, idx, flagcxDeviceMemoryOrderRelease);
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
struct Barrier<DefaultBackend<P>, flagcxTeamTagInter, Coop>
    : DefaultNetProviderPolicy<P>::BarrierState {
  using Comm = typename CommTraits<DefaultBackend<P>>::Comm;
  using Team = typename CommTraits<DefaultBackend<P>>::Team;
  using Net = typename CommTraits<DefaultBackend<P>>::Net;
  using ProxyProvider = typename CommTraits<DefaultBackend<P>>::ProxyProvider;
  using SignalOp = typename Net::SignalOp;
  using WaitSignalOp = typename Net::WaitSignalOp;
  using FlushOp = typename Net::FlushOp;

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
        _nTeamRanks(0), _barrierSignal0(0), _stride(0), _localRank(0) {
    DefaultNetProviderPolicy<P>::initBarrier(
        *this, DefaultNetContext{DefaultNetProviderProxy, 0, nullptr, 0});
  }

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
    DefaultNetProviderPolicy<P>::initBarrier(*this, net.context());
    assert(index < FLAGCX_DEVICE_CTA_COUNT);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR DefaultNetContext context() const {
    return DefaultNetProviderPolicy<P>::barrierContext(*this, _fifoBuffer,
                                                       _contextId);
  }

  // Proxy implementations of the same operations used by Net. Each Barrier
  // lane acts as a one-thread Coop; no block-wide sync belongs inside these.
  FLAGCX_DEVICE_INLINE_DECORATOR void
  proxySignal(Team, int peer, flagcxDevNet_SignalInc action,
              typename PlatformTraits<P>::CoopThread, flagcxDevNet_None,
              flagcxDeviceScope_t, flagcxDeviceScope_t) const {
    int targetRank = peer * _stride + _localRank;
    ProxyProvider::enqueueSignal(_fifoBuffer, _contextId, _signalCount, 0,
                                 (int)action.signal, 1, targetRank, 0);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  proxyWaitSignal(typename PlatformTraits<P>::CoopThread,
                  flagcxDevSignal_t signalId, uint64_t expected, int,
                  flagcxDeviceMemoryOrder_t) const {
    int absIdx = _contextId * _signalCount + (int)signalId;
    ProxyProvider::waitSignalValue(_fifoBuffer, _signalBuffer, absIdx,
                                   expected);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  proxyFlush(typename PlatformTraits<P>::CoopThread,
             flagcxDeviceMemoryOrder_t) const {
    if (_fifoBuffer != nullptr)
      ProxyProvider::fifoFlush(_fifoBuffer);
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
      int signalIdx = _barrierSignal0 + _teamRank;
      DefaultNetDispatch<P>::template call<SignalOp>(
          context(), *this, Team{_nTeamRanks, _teamRank, _stride}, peerIdx,
          flagcxDevNet_SignalInc{(flagcxDevSignal_t)signalIdx},
          typename PlatformTraits<P>::CoopThread{}, flagcxDevNet_None{},
          flagcxDeviceScopeThread, flagcxDeviceScopeDevice);
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
      DefaultNetDispatch<P>::template call<WaitSignalOp>(
          context(), *this, typename PlatformTraits<P>::CoopThread{},
          (flagcxDevSignal_t)signalIdx, expected,
          (int)CommTraits<DefaultBackend<P>>::completionBits,
          flagcxDeviceMemoryOrderAcquire);
    }
    // Flush: ensure all signal FIFO entries from arrive() completed on wire
    if (FLAGCX_THREAD_IDX_X == 0) {
      DefaultNetDispatch<P>::template call<FlushOp>(
          context(), *this, typename PlatformTraits<P>::CoopThread{},
          flagcxDeviceMemoryOrderAcquire);
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
