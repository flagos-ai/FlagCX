/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 *
 * FlagCX Device API - Template wrappers and inline functions for
 * platform-agnostic device-side communication primitives.
 *
 * On NVIDIA (NCCL > 2.28): wraps NCCL device API types and functions.
 * On other platforms: provides fallback implementations using IPC.
 *
 * This header is safe to include from both .cu files (nvcc) and
 * .cc files (g++).  Device-only functions (Sections 5-8) are guarded
 * by FLAGCX_DEVICE_COMPILE so they are invisible to host compilers
 * on all platforms.
 ************************************************************************/

#ifndef FLAGCX_DEVICE_API_H_
#define FLAGCX_DEVICE_API_H_

#include "atomic_device.h"
#include "device_utils.h"
#include "flagcx.h"
#include "flagcx_kernel.h"

// ============================================================
// NVIDIA backend: include NCCL device headers
// ============================================================
#ifdef USE_NVIDIA_ADAPTOR
#include "nccl.h"
#if NCCL_VERSION_CODE > NCCL_VERSION(2, 28, 0)
#include "nccl_device.h"
#define FLAGCX_DEVICE_API_NCCL 1
#endif
#endif

// ============================================================
// Section 1: flagcxDevCommInternal — Host-Side Opaque Handle
//
// Backing struct for flagcxDevComm_t (declared in flagcx_kernel.h).
// Populated by flagcxDevCommCreate, freed by flagcxDevCommDestroy.
// Unified capability-based design: baseline always populated,
// IPC and NCCL layers added when available.
// ============================================================
struct flagcxDevCommInternal {
  // ---- Baseline (always set) ----
  int rank, nRanks;
  int intraRank, intraSize;
  void *fifoBuffer; // Device-accessible FIFO (from heteroComm, may be null)
  bool hasNcclDev;  // true if NCCL device comm layer is available (Tier 1)

  // ---- Grid sync counter (for multi-block two-sided kernels) ----
  unsigned int *gridDoneCounter; // device-allocated atomic counter

  // ---- IPC barrier layer (set if IPC barrier setup succeeds, else nullptr)
  // ----
  uint32_t *
      *barrierPeers; // device pointer to array of nLocalRanks device pointers
  uint32_t *localBarrierFlags; // this rank's barrier memory (CTA_COUNT entries)
  uint32_t barrierEpoch; // monotonically increasing, set by host before launch
  // Host-side cleanup bookkeeping (not passed to kernel)
  void **peerBarrierPtrs; // host array of IPC-mapped pointers (for close)
  int *localRankToRank;   // intra-node rank mapping (for IPC exchange)
  int nLocalRanks;

#ifdef FLAGCX_DEVICE_API_NCCL
  // ---- NCCL layer (set if ncclDevCommCreate succeeds) ----
  ncclDevComm ncclDev;
#endif
};

// ============================================================
// Section 2: flagcxDevMemInternal — Host-Side Memory Handle
//
// Backing struct for flagcxDevMem_t.
// Created by flagcxDevMemCreate, freed by flagcxDevMemDestroy.
// Unified capability-based design: rawPtr always populated,
// IPC and Window layers added when available.
// Capabilities detected by null-checks:
//   devPeerPtrs != nullptr → IPC available
//   hasWindow == true       → Window available (Tier 1 only)
// ============================================================
struct flagcxDevMemInternal {
  // ---- Baseline (always set) ----
  void *rawPtr;   // = buff parameter
  bool hasWindow; // true if any window layer is available (basic or symmetric)
  bool isSymmetric; // true only for FLAGCX_WIN_COLL_SYMMETRIC (enables GIN)

  // ---- IPC layer (set if IPC exchange succeeds, else nullptr) ----
  void **devPeerPtrs; // device array: [localRank] -> peer buffer ptr
  int nPeers;
  int intraRank;       // this rank's local rank index (for IPC local pointer)
  void **hostPeerPtrs; // host array: for ipcMemHandleClose cleanup
  void *basePtr;       // this rank's buffer pointer (for IPC close check)

#ifdef FLAGCX_DEVICE_API_NCCL
  // ---- Window layer (set if window registration succeeds) ----
  ncclWindow_t ncclWin;
  void *winHandle;
#endif
};
#ifndef FLAGCX_DEV_MEM_T_DEFINED
#define FLAGCX_DEV_MEM_T_DEFINED
typedef struct flagcxDevMemInternal *flagcxDevMem_t;
#endif

// ============================================================
// Section 3: flagcxDevComm — Device Communicator (kernel-facing)
//
// Value type passed to kernels by value.
// Unified capability-based design: baseline fields always valid,
// IPC barrier pointers present (may be nullptr),
// NCCL device comm present only on Tier 1 (guarded by _hasBase).
// Constructor from flagcxDevCommInternal enables
// tier-agnostic host code: flagcxDevComm dc(*devCommHandle);
// ============================================================
struct flagcxDevComm {
  // ---- Baseline (always valid) ----
  int _rank, _nRanks;
  int _intraRank, _intraSize;
  void *_fifoBuffer; // FIFO for device Send/Recv (from heteroComm, may be null)
  bool _hasBase;     // true if NCCL device comm layer is available (Tier 1)

  // ---- IPC layer (may be nullptr if IPC barrier not set up) ----
  uint32_t **_barrierPeers;
  uint32_t _barrierEpoch;

#ifdef FLAGCX_DEVICE_API_NCCL
  // ---- NCCL layer (valid only if _hasBase is true) ----
  ncclDevComm _base;
#endif

  FLAGCX_HOST_DEVICE_INLINE flagcxDevComm()
      : _rank(0), _nRanks(0), _intraRank(0), _intraSize(0),
        _fifoBuffer(nullptr), _hasBase(false), _barrierPeers(nullptr),
        _barrierEpoch(0)
#ifdef FLAGCX_DEVICE_API_NCCL
        ,
        _base()
#endif
  {
  }

  // Unified constructor from host handle
  FLAGCX_HOST_DEVICE_INLINE flagcxDevComm(const flagcxDevCommInternal &di)
      : _rank(di.rank), _nRanks(di.nRanks), _intraRank(di.intraRank),
        _intraSize(di.intraSize), _fifoBuffer(di.fifoBuffer),
        _hasBase(di.hasNcclDev), _barrierPeers(di.barrierPeers),
        _barrierEpoch(di.barrierEpoch)
#ifdef FLAGCX_DEVICE_API_NCCL
        ,
        _base(di.ncclDev)
#endif
  {
  }

  // Accessors (unified — always use baseline fields)
  FLAGCX_DEVICE_INLINE_DECORATOR int getIntraRank() const { return _intraRank; }
  FLAGCX_DEVICE_INLINE_DECORATOR int getIntraSize() const { return _intraSize; }
  FLAGCX_DEVICE_INLINE_DECORATOR int getRank() const { return _rank; }
  FLAGCX_DEVICE_INLINE_DECORATOR int getSize() const { return _nRanks; }
  FLAGCX_DEVICE_INLINE_DECORATOR void *getFifoBuffer() const {
    return _fifoBuffer;
  }
};

// ============================================================
// Section 4: flagcxDevMem — Device-Side Memory Handle
//
// Value type passed to kernels by value.
// Unified capability-based design: rawPtr always valid,
// IPC peerPtrs present (may be nullptr),
// Window _base present only on Tier 1 (guarded by _hasWindow).
// Runtime dispatch in flagcxGetPeerPointer uses priority:
//   Window > IPC > Raw.
// Constructor from flagcxDevMemInternal enables
// tier-agnostic host code: flagcxDevMem dm(*devMemHandle);
// ============================================================
struct flagcxDevMem {
  // ---- Baseline (always valid) ----
  void *rawPtr;
  bool _hasWindow; // true if any window layer is available (basic or symmetric)
  bool _isSymmetric; // true only for FLAGCX_WIN_COLL_SYMMETRIC (enables GIN)

  // ---- IPC layer (may be nullptr) ----
  void **peerPtrs; // device array [localRank] -> peer buffer
  int intraRank;   // local rank index (for flagcxGetLocalPointer in IPC mode)

#ifdef FLAGCX_DEVICE_API_NCCL
  // ---- Window layer (valid only if _hasWindow is true) ----
  ncclWindow_t _base;
#endif

  FLAGCX_HOST_DEVICE_INLINE flagcxDevMem()
      : rawPtr(nullptr), _hasWindow(false), _isSymmetric(false),
        peerPtrs(nullptr), intraRank(0)
#ifdef FLAGCX_DEVICE_API_NCCL
        ,
        _base()
#endif
  {
  }

  // Unified constructor from host handle
  FLAGCX_HOST_DEVICE_INLINE flagcxDevMem(const flagcxDevMemInternal &di)
      : rawPtr(di.rawPtr), _hasWindow(di.hasWindow),
        _isSymmetric(di.isSymmetric), peerPtrs(di.devPeerPtrs),
        intraRank(di.intraRank)
#ifdef FLAGCX_DEVICE_API_NCCL
        ,
        _base(di.ncclWin)
#endif
  {
  }
};

// ============================================================
// Section 4b: flagcxTeam_t — Team Descriptor
//
// Represents a subset of ranks (intra-node, inter-node, etc.).
// On NVIDIA: wraps ncclTeam_t.
// ============================================================
struct flagcxTeam {
  int nRanks;
  int rank;
  int stride;

#ifdef FLAGCX_DEVICE_API_NCCL
  ncclTeam_t _base;

  FLAGCX_HOST_DEVICE_INLINE flagcxTeam()
      : nRanks(0), rank(0), stride(0), _base() {}
  FLAGCX_HOST_DEVICE_INLINE flagcxTeam(ncclTeam_t base)
      : nRanks(base.nRanks), rank(base.rank), stride(base.stride), _base(base) {
  }
  FLAGCX_HOST_DEVICE_INLINE operator ncclTeam_t() const { return _base; }
#else
  FLAGCX_HOST_DEVICE_INLINE flagcxTeam() : nRanks(0), rank(0), stride(0) {}
#endif
};
typedef struct flagcxTeam flagcxTeam_t;

// Team tag types for barrier session constructors
struct flagcxTeamTagWorld {};
struct flagcxTeamTagIntra {};
struct flagcxTeamTagInter {};

// ============================================================
// Sections 5-8: Device-only functions
//
// These sections use device builtins (threadIdx, __syncthreads, atomics)
// and are only safe under a device compiler (nvcc, hipcc, etc.).
// FLAGCX_DEVICE_COMPILE is defined in device_utils.h.
// ============================================================
#ifdef FLAGCX_DEVICE_COMPILE

// ============================================================
// Section 5: Team Accessor Functions (Inline Wrappers)
//
// On Tier 1 with _hasBase: uses NCCL team functions.
// Otherwise: computes from baseline fields.
// ============================================================
FLAGCX_DEVICE_INLINE_DECORATOR
flagcxTeam_t flagcxTeamIntra(const flagcxDevComm &devComm) {
#ifdef FLAGCX_DEVICE_API_NCCL
  if (devComm._hasBase)
    return flagcxTeam_t(ncclTeamLsa(devComm._base));
#endif
  flagcxTeam_t team;
  team.nRanks = devComm.getIntraSize();
  team.rank = devComm.getIntraRank();
  team.stride = 1;
  return team;
}
FLAGCX_DEVICE_INLINE_DECORATOR
flagcxTeam_t flagcxTeamWorld(const flagcxDevComm &devComm) {
#ifdef FLAGCX_DEVICE_API_NCCL
  if (devComm._hasBase)
    return flagcxTeam_t(ncclTeamWorld(devComm._base));
#endif
  flagcxTeam_t team;
  team.nRanks = devComm.getSize();
  team.rank = devComm.getRank();
  team.stride = 1;
  return team;
}
FLAGCX_DEVICE_INLINE_DECORATOR
flagcxTeam_t flagcxTeamInter(const flagcxDevComm &devComm) {
#ifdef FLAGCX_DEVICE_API_NCCL
  if (devComm._hasBase)
    return flagcxTeam_t(ncclTeamRail(devComm._base));
#endif
  flagcxTeam_t team;
  team.nRanks = devComm.getSize() / devComm.getIntraSize();
  team.rank = devComm.getRank() / devComm.getIntraSize();
  team.stride = devComm.getIntraSize();
  return team;
}

// ============================================================
// Section 6: flagcxCoopBlock — Block-Level Cooperative Group
//
// On NVIDIA: wraps ncclCoopCta.
// ============================================================
struct flagcxCoopBlock {
#ifdef FLAGCX_DEVICE_API_NCCL
  ncclCoopCta _impl;

  FLAGCX_HOST_DEVICE_INLINE flagcxCoopBlock() : _impl() {}

  FLAGCX_DEVICE_INLINE_DECORATOR int thread_rank() const {
    return _impl.thread_rank();
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return _impl.size(); }
  FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _impl.sync(); }

  // Implicit conversion for passthrough to NCCL APIs
  FLAGCX_HOST_DEVICE_INLINE operator ncclCoopCta() const { return _impl; }
#else
  FLAGCX_DEVICE_INLINE_DECORATOR int thread_rank() const {
    return 0; // placeholder for fallback
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int size() const {
    return 0; // placeholder for fallback
  }
  FLAGCX_DEVICE_INLINE_DECORATOR void sync() { FLAGCX_DEVICE_SYNC_THREADS(); }
#endif
};

// ============================================================
// Section 7: flagcxIntraBarrierSession — Intra-Node Barrier
//
// On NVIDIA (NCCL > 2.28): wraps ncclLsaBarrierSession.
// On fallback: flag-based barrier using IPC-mapped peer memory + atomics.
// ============================================================
template <typename Coop>
struct flagcxIntraBarrierSession {
#ifdef FLAGCX_DEVICE_API_NCCL
  ncclLsaBarrierSession<ncclCoopCta> _impl;

  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxIntraBarrierSession(Coop coop, const flagcxDevComm &devComm,
                            flagcxTeam_t team, uint32_t index)
      : _impl(ncclCoopCta(), devComm._base, ncclTeamLsa(devComm._base),
              devComm._base.lsaBarrier, index, false) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(Coop coop,
         flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    _impl.arrive(ncclCoopCta(), flagcxDeviceMemoryOrderMap[order]);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    _impl.wait(ncclCoopCta(), flagcxDeviceMemoryOrderMap[order]);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    _impl.sync(ncclCoopCta(), flagcxDeviceMemoryOrderMap[order]);
  }
#else
  // Fallback: flag-based barrier using IPC-mapped peer memory + atomics
  uint32_t **_peerBarriers;
  int _nRanks, _myRank;
  uint32_t _ctaIndex;
  uint32_t _phase;

  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxIntraBarrierSession(Coop coop, const flagcxDevComm &devComm,
                            flagcxTeam_t team, uint32_t index)
      : _peerBarriers(devComm._barrierPeers), _nRanks(team.nRanks),
        _myRank(team.rank), _ctaIndex(index), _phase(devComm._barrierEpoch) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(Coop coop,
         flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    sync(coop, order);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    sync(coop, order);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    _phase++;
    coop.sync();
    if (threadIdx.x == 0) {
      // Signal: write my counter with release ordering
      flagcxDeviceAtomicStore(&_peerBarriers[_myRank][_ctaIndex], _phase,
                              flagcxDeviceMemoryOrderRelease);
      // Wait: spin until all peers reach this phase
      for (int p = 0; p < _nRanks; p++) {
        if (p == _myRank)
          continue;
        int iter = 0;
        while (flagcxDeviceAtomicLoad(&_peerBarriers[p][_ctaIndex],
                                      flagcxDeviceMemoryOrderAcquire) <
               _phase) {
          spinBackoff(iter++);
        }
      }
    }
    coop.sync();
  }
#endif
};

// ============================================================
// Section 8: Pointer Access Functions (Inline Wrappers)
//
// 3 functions total (see plan Decision 7.8 / 7.9 / 7.19):
//   flagcxGetPeerPointer(mem, off, team, peer)  — canonical unicast
//   flagcxGetLocalPointer(mem, off)              — convenience (own buffer)
//   flagcxGetMulticastPointer(mem, off, devComm) — intra-node multicast
//
// Priority dispatch: Window > IPC > Raw.
// Capabilities detected by _hasWindow flag and peerPtrs null-check.
// ============================================================
FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetPeerPointer(const flagcxDevMem &mem, size_t offset, flagcxTeam_t team,
                     int peer) {
#ifdef FLAGCX_DEVICE_API_NCCL
  if (mem._hasWindow)
    return ncclGetPeerPointer(mem._base, offset, team._base, peer);
#endif
  if (mem.peerPtrs != nullptr) {
    int index = team.rank + (peer - team.rank) * team.stride;
    return (char *)mem.peerPtrs[index] + offset;
  }
  return nullptr; // Raw mode: no peer access
}

FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetLocalPointer(const flagcxDevMem &mem, size_t offset) {
#ifdef FLAGCX_DEVICE_API_NCCL
  if (mem._hasWindow)
    return ncclGetLocalPointer(mem._base, offset);
#endif
  if (mem.peerPtrs != nullptr)
    return (char *)mem.peerPtrs[mem.intraRank] + offset;
  return (char *)mem.rawPtr + offset;
}

FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetMulticastPointer(const flagcxDevMem &mem, size_t offset,
                          const flagcxDevComm &devComm) {
#ifdef FLAGCX_DEVICE_API_NCCL
  if (mem._hasWindow)
    return ncclGetLsaMultimemPointer(mem._base, offset, devComm._base);
#endif
  return nullptr; // IPC/raw mode: multicast not available
}

// ============================================================
// Section 8b: flagcxSymPtr<T> — Typed Symmetric Pointer
//
// Value type storing {flagcxDevMem, offset}. Provides typed
// pointer methods and type-aware arithmetic.
// Mirrors NCCL's ncclSymPtr<T>.
// ============================================================
template <typename T>
struct flagcxSymPtr {
  flagcxDevMem mem;
  size_t offset;

  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr() : mem(), offset(0) {}
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr(flagcxDevMem m, size_t off)
      : mem(m), offset(off) {}

  // Typed pointer methods (delegate to existing free functions)
  FLAGCX_DEVICE_INLINE_DECORATOR T *localPtr() const {
    return (T *)flagcxGetLocalPointer(mem, offset);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR T *peerPtr(flagcxTeam_t team, int peer) const {
    return (T *)flagcxGetPeerPointer(mem, offset, team, peer);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR T *
  multicastPtr(const flagcxDevComm &devComm) const {
    return (T *)flagcxGetMulticastPointer(mem, offset, devComm);
  }

  // Type-aware pointer arithmetic
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> &operator+=(int d) {
    offset += d * sizeof(T);
    return *this;
  }
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> &operator-=(int d) {
    offset -= d * sizeof(T);
    return *this;
  }
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> operator+(int d) const {
    return {mem, offset + d * sizeof(T)};
  }
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> operator-(int d) const {
    return {mem, offset - d * sizeof(T)};
  }
};

#endif // FLAGCX_DEVICE_COMPILE

// ============================================================
// Section 9: Constants
// ============================================================
#ifndef FLAGCX_DEVICE_CTA_COUNT
#define FLAGCX_DEVICE_CTA_COUNT 36
#endif
#ifndef FLAGCX_DEVICE_THREADS_PER_CTA
#define FLAGCX_DEVICE_THREADS_PER_CTA 512
#endif

// ============================================================
// Sections 9b-12: flagcxDevNet + Barriers (device-only)
// ============================================================
#ifdef FLAGCX_DEVICE_COMPILE

// ---- Inline FIFO helpers (used by flagcxDevNet and Tier 2 barriers) ----

// Enqueue a trigger into the device FIFO buffer.
// Atomically reserves a slot, waits for space, writes trigger with valid bit.
FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t
flagcxFifoEnqueue(void *fifoBuffer, uint64_t addr, uint64_t count,
                  uint64_t peerRank, uint64_t datatype, uint64_t type) {
  uint64_t *buffer = (uint64_t *)fifoBuffer;
  uint64_t capacity = flagcxDeviceAtomicLoad(&buffer[flagcxFifoIdxCapacity],
                                             flagcxDeviceMemoryOrderRelaxed);

  // 1. Atomically reserve a slot
  uint64_t mySlot =
      flagcxDeviceAtomicFetchAdd(&buffer[flagcxFifoIdxProduced], (uint64_t)1,
                                 flagcxDeviceMemoryOrderAcqRel);

  // 2. Wait until there's space (mySlot - consumed < capacity)
  int iter = 0;
  while ((int64_t)(mySlot -
                   flagcxDeviceAtomicLoad(&buffer[flagcxFifoIdxConsumed],
                                          flagcxDeviceMemoryOrderAcquire)) >=
         (int64_t)capacity) {
    spinBackoff(iter++);
  }

  // 3. Compute slot index and get pointer to slot's raw uint64_t fields
  uint64_t idx = mySlot % capacity;
  uint64_t *slotFst = buffer + flagcxFifoIdxData +
                      idx * (sizeof(flagcxDeviceTrigger) / sizeof(uint64_t));
  uint64_t *slotSnd = slotFst + 1;

  // 4. Write address first
  flagcxDeviceAtomicStore(slotFst, addr, flagcxDeviceMemoryOrderRelaxed);

  // 5. Build snd value with valid bit set
  uint64_t sndValue =
      ((count & flagcxTriggerMask(flagcxDeviceTriggerBitsCount))
       << flagcxDeviceTriggerOffCount) |
      ((peerRank & flagcxTriggerMask(flagcxDeviceTriggerBitsPeerRank))
       << flagcxDeviceTriggerOffPeerRank) |
      ((datatype & flagcxTriggerMask(flagcxDeviceTriggerBitsDatatype))
       << flagcxDeviceTriggerOffDatatype) |
      ((type & flagcxTriggerMask(flagcxDeviceTriggerBitsPrim))
       << flagcxDeviceTriggerOffPrim) |
      flagcxDeviceTriggerValidMask;

  // 6. Write snd with valid bit (release ensures fst is visible before snd)
  flagcxDeviceAtomicStore(slotSnd, sndValue, flagcxDeviceMemoryOrderRelease);

  return flagcxSuccess;
}

// Wait until all FIFO entries are consumed by the host proxy.
// Enqueues a WAIT marker, then spins until produced == consumed.
FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t flagcxFifoWait(void *fifoBuffer) {
  flagcxFifoEnqueue(fifoBuffer, 0, 0, 0, 0, flagcxDevicePrimWait);

  uint64_t *buffer = (uint64_t *)fifoBuffer;
  int iter = 0;
  while (flagcxDeviceAtomicLoad(&buffer[flagcxFifoIdxProduced],
                                flagcxDeviceMemoryOrderAcquire) >
         flagcxDeviceAtomicLoad(&buffer[flagcxFifoIdxConsumed],
                                flagcxDeviceMemoryOrderAcquire)) {
    spinBackoff(iter++);
  }
  return flagcxSuccess;
}

// ============================================================
// Section 9b: GIN Types (Tier 1 only)
// ============================================================
// Fence level enum — available on all tiers for unified barrier API
enum class flagcxGinFenceLevel { Relaxed };

// GIN action types and typedefs — available on all tiers for API completeness.
// On Tier 2, GIN methods are stubs (compile but trap at runtime).
typedef uint32_t flagcxDevNetSignal_t;
typedef uint32_t flagcxDevNetCounter_t;

struct flagcxDevNet_None {};
struct flagcxDevNet_SignalInc {
  flagcxDevNetSignal_t signal;
};
struct flagcxDevNet_SignalAdd {
  flagcxDevNetSignal_t signal;
  uint64_t value;
};
struct flagcxDevNet_CounterInc {
  flagcxDevNetCounter_t counter;
};

#ifdef FLAGCX_DEVICE_API_NCCL
// Action type mapping helpers (flagcx -> nccl)
FLAGCX_DEVICE_INLINE_DECORATOR ncclGin_None toNccl(flagcxDevNet_None) {
  return {};
}
FLAGCX_DEVICE_INLINE_DECORATOR ncclGin_SignalInc
toNccl(flagcxDevNet_SignalInc a) {
  return {a.signal};
}
FLAGCX_DEVICE_INLINE_DECORATOR ncclGin_SignalAdd
toNccl(flagcxDevNet_SignalAdd a) {
  return {a.signal, a.value};
}
FLAGCX_DEVICE_INLINE_DECORATOR ncclGin_CounterInc
toNccl(flagcxDevNet_CounterInc a) {
  return {a.counter};
}
FLAGCX_DEVICE_INLINE_DECORATOR ncclCoopCta toNccl(flagcxCoopBlock) {
  return {};
}
#endif // FLAGCX_DEVICE_API_NCCL

// ============================================================
// Section 10: flagcxDevNet — Device Network (all tiers)
// ============================================================
struct flagcxDevNet {
  const flagcxDevComm &_devComm; // for barrier + Send/Recv on all tiers

#ifdef FLAGCX_DEVICE_API_NCCL
  ncclGin _gin; // GIN backend (Tier 1 only)

  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxDevNet(const flagcxDevComm &dc, int contextIndex = 0)
      : _devComm(dc), _gin(dc._base, contextIndex) {}
#else
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxDevNet(const flagcxDevComm &dc, int contextIndex = 0) : _devComm(dc) {}
#endif

  // ---- Two-sided operations (all tiers, via FIFO) ----
  // send/recv use flagcxDevMem for API consistency; extract rawPtr for FIFO.
  // GIN is one-sided (put + signals); two-sided send/recv always use FIFO.
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t send(const flagcxDevMem &mem,
                                                     size_t offset,
                                                     size_t count,
                                                     flagcxDataType_t datatype,
                                                     int peer) const {
    return flagcxFifoEnqueue(
        _devComm.getFifoBuffer(),
        (uint64_t)((uintptr_t)((const char *)mem.rawPtr + offset)), count, peer,
        datatype, flagcxDevicePrimSend);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t recv(const flagcxDevMem &mem,
                                                     size_t offset,
                                                     size_t count,
                                                     flagcxDataType_t datatype,
                                                     int peer) const {
    return flagcxFifoEnqueue(
        _devComm.getFifoBuffer(),
        (uint64_t)((uintptr_t)((char *)mem.rawPtr + offset)), count, peer,
        datatype, flagcxDevicePrimRecv);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t term() const {
    return flagcxFifoEnqueue(_devComm.getFifoBuffer(), 0, 0, 0, 0,
                             flagcxDevicePrimTerm);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t wait() const {
    return flagcxFifoWait(_devComm.getFifoBuffer());
  }

  // ---- One-sided FIFO operations (all tiers, via FIFO) ----
  // put/signal use FIFO for proxy-based one-sided operations.
  // These are simpler than GIN put/signal and work on all tiers.
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t put(size_t srcOffset,
                                                    size_t dstOffset,
                                                    size_t count,
                                                    flagcxDataType_t datatype,
                                                    int peer) const {
    uint64_t fstValue =
        ((uint64_t)srcOffset << flagcxDeviceTriggerOffSrcOffset) |
        ((uint64_t)dstOffset << flagcxDeviceTriggerOffDstOffset);
    return flagcxFifoEnqueue(_devComm.getFifoBuffer(), fstValue, count, peer,
                             datatype, flagcxDevicePrimPut);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t signal(size_t dstOffset,
                                                       int peer) const {
    uint64_t fstValue = (uint64_t)dstOffset << flagcxDeviceTriggerOffDstOffset;
    return flagcxFifoEnqueue(_devComm.getFifoBuffer(), fstValue, 0, peer, 0,
                             flagcxDevicePrimSignal);
  }

#ifdef FLAGCX_DEVICE_API_NCCL
  // ---- GIN one-sided operations (Tier 1 only) ----

  template <typename RemoteAction = flagcxDevNet_None,
            typename LocalAction = flagcxDevNet_None,
            typename Coop = flagcxCoopBlock>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  put(flagcxTeam_t team, int peer, const flagcxDevMem &dstMem, size_t dstOffset,
      const flagcxDevMem &srcMem, size_t srcOffset, size_t bytes,
      RemoteAction remoteAction = flagcxDevNet_None{},
      LocalAction localAction = flagcxDevNet_None{},
      Coop coop = flagcxCoopBlock{}) const {
    _gin.put(team._base, peer, dstMem._base, dstOffset, srcMem._base, srcOffset,
             bytes, toNccl(remoteAction), toNccl(localAction), toNccl(coop));
  }

  // SymPtr-based put overload — convenience wrapper, delegates to window-based
  // put with nElts * sizeof(T) bytes.
  template <typename T, typename RemoteAction = flagcxDevNet_None,
            typename LocalAction = flagcxDevNet_None,
            typename Coop = flagcxCoopBlock>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  put(flagcxTeam_t team, int peer, flagcxSymPtr<T> dst, flagcxSymPtr<T> src,
      size_t nElts, RemoteAction remoteAction = flagcxDevNet_None{},
      LocalAction localAction = flagcxDevNet_None{},
      Coop coop = flagcxCoopBlock{}) const {
    this->put(team, peer, dst.mem, dst.offset, src.mem, src.offset,
              nElts * sizeof(T), remoteAction, localAction, coop);
  }

  template <typename T, typename RemoteAction = flagcxDevNet_None,
            typename Coop = flagcxCoopBlock>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  putValue(flagcxTeam_t team, int peer, const flagcxDevMem &dstMem,
           size_t dstOffset, T value,
           RemoteAction remoteAction = flagcxDevNet_None{},
           Coop coop = flagcxCoopBlock{}) const {
    _gin.putValue(team._base, peer, dstMem._base, dstOffset, value,
                  toNccl(remoteAction), toNccl(coop));
  }

  template <typename RemoteAction, typename Coop = flagcxCoopBlock>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  signal(flagcxTeam_t team, int peer, RemoteAction remoteAction,
         Coop coop = flagcxCoopBlock{}) const {
    _gin.signal(team._base, peer, toNccl(remoteAction), toNccl(coop));
  }

  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR void flush(
      Coop coop,
      flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcquire) const {
    _gin.flush(toNccl(coop), flagcxDeviceMemoryOrderMap[order]);
  }

  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR void waitSignal(
      Coop coop, flagcxDevNetSignal_t signal, uint64_t least, int bits = 64,
      flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcquire) const {
    _gin.waitSignal(toNccl(coop), signal, least, bits,
                    flagcxDeviceMemoryOrderMap[order]);
  }

  // Wait for signal to meet or exceed its shadow value.
  // Convenience — avoids passing explicit `least`.
  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR void waitSignalMeetShadow(
      Coop coop, flagcxDevNetSignal_t signal, int bits = 64,
      flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcquire) const {
    _gin.waitSignalMeetShadow(toNccl(coop), signal, bits,
                              flagcxDeviceMemoryOrderMap[order]);
  }

  // Wait until signal exceeds shadow by leastDelta, updates shadow with latest
  // value. Returns previous shadow in *before and difference in *delta.
  // Used in pipelined patterns for incremental progress tracking.
  template <typename Coop, typename Uint>
  FLAGCX_DEVICE_INLINE_DECORATOR void waitSignalFollowShadow(
      Coop coop, flagcxDevNetSignal_t signal, Uint leastDelta, Uint *before,
      Uint *delta, int bits = 64,
      flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcquire) const {
    _gin.waitSignalFollowShadow(toNccl(coop), signal, leastDelta, before, delta,
                                bits, flagcxDeviceMemoryOrderMap[order]);
  }

  // --- Shadow manipulation ---

  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t *
  getSignalShadowPtr(flagcxDevNetSignal_t signal) const {
    return _gin.getSignalShadowPtr(signal);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  increaseSignalShadow(flagcxDevNetSignal_t signal, uint64_t delta) const {
    _gin.increaseSignalShadow(signal, delta);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t readSignal(
      flagcxDevNetSignal_t signal, int bits = 64,
      flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcquire) const {
    return _gin.readSignal(signal, bits, flagcxDeviceMemoryOrderMap[order]);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  resetSignal(flagcxDevNetSignal_t signal) const {
    _gin.resetSignal(signal);
  }

  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR void waitCounter(
      Coop coop, flagcxDevNetCounter_t counter, uint64_t least, int bits = 56,
      flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcquire) const {
    _gin.waitCounter(toNccl(coop), counter, least, bits,
                     flagcxDeviceMemoryOrderMap[order]);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t readCounter(
      flagcxDevNetCounter_t counter, int bits = 56,
      flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcquire) const {
    return _gin.readCounter(counter, bits, flagcxDeviceMemoryOrderMap[order]);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  resetCounter(flagcxDevNetCounter_t counter) const {
    _gin.resetCounter(counter);
  }
#else
  // ---- Tier 2 stubs: GIN not available ----
  // These compile but should never be called at runtime.
  // Host-side code must check tier before launching GIN kernels.

  template <typename RemoteAction = flagcxDevNet_None,
            typename LocalAction = flagcxDevNet_None,
            typename Coop = flagcxCoopBlock>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  put(flagcxTeam_t, int, const flagcxDevMem &, size_t, const flagcxDevMem &,
      size_t, size_t, RemoteAction = flagcxDevNet_None{},
      LocalAction = flagcxDevNet_None{}, Coop = flagcxCoopBlock{}) const {}

  template <typename T, typename RemoteAction = flagcxDevNet_None,
            typename LocalAction = flagcxDevNet_None,
            typename Coop = flagcxCoopBlock>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  put(flagcxTeam_t, int, flagcxSymPtr<T>, flagcxSymPtr<T>, size_t,
      RemoteAction = flagcxDevNet_None{}, LocalAction = flagcxDevNet_None{},
      Coop = flagcxCoopBlock{}) const {}

  template <typename T, typename RemoteAction = flagcxDevNet_None,
            typename Coop = flagcxCoopBlock>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  putValue(flagcxTeam_t, int, const flagcxDevMem &, size_t, T,
           RemoteAction = flagcxDevNet_None{}, Coop = flagcxCoopBlock{}) const {
  }

  template <typename RemoteAction, typename Coop = flagcxCoopBlock>
  FLAGCX_DEVICE_INLINE_DECORATOR void signal(flagcxTeam_t, int, RemoteAction,
                                             Coop = flagcxCoopBlock{}) const {}

  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  flush(Coop,
        flagcxDeviceMemoryOrder_t = flagcxDeviceMemoryOrderAcquire) const {}

  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  waitSignal(Coop, flagcxDevNetSignal_t, uint64_t, int = 64,
             flagcxDeviceMemoryOrder_t = flagcxDeviceMemoryOrderAcquire) const {
  }

  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR void waitSignalMeetShadow(
      Coop, flagcxDevNetSignal_t, int = 64,
      flagcxDeviceMemoryOrder_t = flagcxDeviceMemoryOrderAcquire) const {}

  template <typename Coop, typename Uint>
  FLAGCX_DEVICE_INLINE_DECORATOR void waitSignalFollowShadow(
      Coop, flagcxDevNetSignal_t, Uint, Uint *, Uint *, int = 64,
      flagcxDeviceMemoryOrder_t = flagcxDeviceMemoryOrderAcquire) const {}

  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t *
  getSignalShadowPtr(flagcxDevNetSignal_t) const {
    return nullptr;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void increaseSignalShadow(flagcxDevNetSignal_t,
                                                           uint64_t) const {}

  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t
  readSignal(flagcxDevNetSignal_t, int = 64,
             flagcxDeviceMemoryOrder_t = flagcxDeviceMemoryOrderAcquire) const {
    return 0;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void resetSignal(flagcxDevNetSignal_t) const {}

  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR void waitCounter(
      Coop, flagcxDevNetCounter_t, uint64_t, int = 56,
      flagcxDeviceMemoryOrder_t = flagcxDeviceMemoryOrderAcquire) const {}

  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t readCounter(
      flagcxDevNetCounter_t, int = 56,
      flagcxDeviceMemoryOrder_t = flagcxDeviceMemoryOrderAcquire) const {
    return 0;
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  resetCounter(flagcxDevNetCounter_t) const {}
#endif // FLAGCX_DEVICE_API_NCCL
};

// ============================================================
// Section 11: flagcxInterBarrierSession — GIN Barrier (Tier 1 only)
// ============================================================
#ifdef FLAGCX_DEVICE_API_NCCL
template <typename Coop>
struct flagcxInterBarrierSession {
  ncclGinBarrierSession<ncclCoopCta> _impl;

  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxInterBarrierSession(Coop coop, const flagcxDevNet &net,
                            flagcxTeam_t team, uint32_t index)
      : _impl(ncclCoopCta(), net._gin, team._base, net._gin.comm.railGinBarrier,
              index) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxGinFenceLevel fence = flagcxGinFenceLevel::Relaxed) {
    _impl.sync(ncclCoopCta(), flagcxDeviceMemoryOrderMap[order],
               ncclGinFenceLevel::Relaxed);
  }
};
#else
// Tier 2 stub — compiles but should never be instantiated at runtime.
template <typename Coop>
struct flagcxInterBarrierSession {
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxInterBarrierSession(Coop, const flagcxDevNet &, flagcxTeam_t,
                            uint32_t) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(Coop, flagcxDeviceMemoryOrder_t = flagcxDeviceMemoryOrderAcqRel,
       flagcxGinFenceLevel = flagcxGinFenceLevel::Relaxed) {}
};
#endif

// ============================================================
// Section 12: flagcxBarrierSession — Unified Barrier (both tiers)
// ============================================================
#ifdef FLAGCX_DEVICE_API_NCCL
template <typename Coop>
struct flagcxBarrierSession {
  ncclBarrierSession<ncclCoopCta> _impl;

  // World barrier (intra + inter)
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxBarrierSession(Coop coop, flagcxTeamTagWorld, const flagcxDevNet &net,
                       uint32_t index)
      : _impl(ncclCoopCta(), ncclTeamTagWorld(), net._gin, index) {}

  // Intra-only barrier
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxBarrierSession(Coop coop, flagcxTeamTagIntra,
                       const flagcxDevComm &devComm, uint32_t index)
      : _impl(ncclCoopCta(), ncclTeamTagLsa(), devComm._base, index) {}

  // Inter-only barrier
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxBarrierSession(Coop coop, flagcxTeamTagInter, const flagcxDevNet &net,
                       uint32_t index)
      : _impl(ncclCoopCta(), ncclTeamTagRail(), net._gin, index) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxGinFenceLevel fence = flagcxGinFenceLevel::Relaxed) {
    _impl.sync(ncclCoopCta(), flagcxDeviceMemoryOrderMap[order],
               ncclGinFenceLevel::Relaxed);
  }
};
#else
// Tier 2: World barrier uses Term/Wait (FIFO-based, covers both intra +
// inter)
//           Intra barrier uses flagcxIntraBarrierSession (IPC-based, multi-CTA)
//           Inter barrier not available standalone (use World instead)
template <typename Coop>
struct flagcxBarrierSession {
  bool _useTermWait;                      // true=World (Term/Wait), false=Intra
  flagcxDevComm _devCommCopy;             // copy for Term/Wait in World mode
  flagcxIntraBarrierSession<Coop> _intra; // used in Intra mode

  // World barrier: Wait via FIFO (single-CTA constraint)
  // flagcxDevNet::wait() drains FIFO — spins until all items consumed.
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxBarrierSession(Coop coop, flagcxTeamTagWorld, const flagcxDevNet &net,
                       uint32_t index)
      : _useTermWait(true), _devCommCopy(net._devComm),
        _intra(coop, net._devComm, flagcxTeamIntra(net._devComm), index) {}

  // Intra-only barrier: IPC-based (multi-CTA)
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxBarrierSession(Coop coop, flagcxTeamTagIntra,
                       const flagcxDevComm &devComm, uint32_t index)
      : _useTermWait(false), _devCommCopy(),
        _intra(coop, devComm, flagcxTeamIntra(devComm), index) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxGinFenceLevel fence = flagcxGinFenceLevel::Relaxed) {
    if (_useTermWait) {
      // World barrier: Wait drains FIFO (all threads sync before thread 0 acts)
      coop.sync();
      if (threadIdx.x == 0) {
        flagcxDevNet(_devCommCopy).wait();
      }
      coop.sync();
    } else {
      // Intra-only barrier: IPC-based
      _intra.sync(coop, order);
    }
  }
};
#endif

#endif // FLAGCX_DEVICE_COMPILE (Sections 9b-12)

#endif // FLAGCX_DEVICE_API_H_
