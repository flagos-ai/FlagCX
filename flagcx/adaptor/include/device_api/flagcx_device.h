/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX Device API - Template wrappers and inline functions for
 * platform-agnostic device-side communication primitives.
 *
 * On Vendor: wraps vendor device API types and functions.
 * On other platforms: provides fallback implementations using IPC.
 *
 * This header is safe to include from both .cu files (nvcc) and
 * .cc files (g++).  Device-only functions (Sections 5-8) are guarded
 * by FLAGCX_DEVICE_COMPILE so they are invisible to host compilers
 * on all platforms.
 ************************************************************************/

#ifndef FLAGCX_DEVICE_API_H_
#define FLAGCX_DEVICE_API_H_

#include <cstddef> // ptrdiff_t, size_t

#include "device_utils.h"
#include "flagcx.h"
#include "flagcx_kernel.h"
#include "shmutils.h" // flagcxShmHandle_t

// Device traits — provides DeviceAPI with all type/function dispatch.
// Also defines FLAGCX_DEVICE_API_VENDOR when vendor backend is active.
// Action types (flagcxDevNet_*, flagcxDescriptorSmem, etc.) are defined
// inside comm_traits.h before the backend-specific trait includes.
#include "comm_traits.h"

// Forward declaration for typed vendor device comm handle
struct flagcxInnerDevComm;
typedef struct flagcxInnerDevComm *flagcxInnerDevComm_t;

// ============================================================
// Section 1: flagcxDevCommInternal — Host-Side Opaque Handle
//
#define FLAGCX_MAX_INTER_PEERS 256

// Backing struct for flagcxDevComm_t (declared in flagcx_kernel.h).
// Populated by flagcxDevCommCreate, freed by flagcxDevCommDestroy.
// Unified capability-based design: baseline always populated,
// IPC and Vendor layers added when available.
// ============================================================
struct flagcxDevCommInternal {
  // ---- Baseline (always set) ----
  int rank, nRanks;
  int intraRank, intraSize;
  void *fifoBuffers[FLAGCX_DEVICE_CTA_COUNT]; // Device-accessible FIFOs (one
                                              // per context, from heteroComm,
                                              // may be null)
  // ---- IPC barrier layer (set if IPC barrier setup succeeds, else nullptr)
  // ----
  uint64_t *
      *barrierPeers; // device pointer to array of nLocalRanks device pointers
  uint64_t
      *localBarrierFlags; // this rank's inbox buffer (nLocalRanks × CTA_COUNT)
  uint64_t
      intraBarrierEpoch; // monotonically increasing, set by host before launch
  int nBarriers;         // = FLAGCX_DEVICE_CTA_COUNT (needed in kernel)
  // Host-side cleanup bookkeeping (not passed to kernel)
  int barrierIpcIndex;  // index into comm->ipcTable (-1 if no IPC barrier)
  int *localRankToRank; // intra-node rank mapping (for IPC exchange)
  int nLocalRanks;
  // flagcxShm barrier path (non-null when FLAGCX_SIGNAL_HOST_ENABLE=1)
  void *localBarrierShmPtr;  // CPU VA of own shm mapping (hostUnregister +
                             // flagcxShmClose)
  void **peerBarrierShmPtrs; // CPU VA array [nLocalRanks] for each peer's shm
                             // mapping (hostUnregister + flagcxShmClose)
  size_t barrierShmSize;     // size in bytes (for hostUnregister)
  uint64_t **barrierDevPeerPtrsRaw;  // standalone device array (deviceFree; shm
                                     // path only)
  flagcxShmHandle_t myShmHandle;     // own shm handle (flagcxShmClose)
  flagcxShmHandle_t *peerShmHandles; // peer shm handles [nLocalRanks]

  // ---- Inter-node signal relay (set if nInterPeers > 0, else nullptr) ----
  uint64_t *interSignalFlags;     // device pointer (from hostGetDevicePointer)
  uint64_t *interSignalFlagsHost; // host pointer (for recv thread + dealloc)
  uint64_t
      interBarrierEpoch; // inter-node epoch (separate from intraBarrierEpoch)
  int nInterPeers;       // number of inter-node peers (set on ALL ranks)
  bool isInterLeader;    // true only on localRank 0 (manages connections)
  int *interPeerRanks;   // global ranks of inter-node peers
  // netAdaptor connections for signal relay (one-sided RDMA atomic)
  void **signalSendComms;  // [nInterPeers] sendComm (for iputSignal)
  void **barrierRecvComms; // [nInterPeers] recvComm (kept alive for QP)
  void *barrierHandleInfo; // flagcxOneSideHandleInfo* with rkeys/baseVas
  // netAdaptor pointer (cached for proxy)
  void *netAdaptorPtr;

  // ---- One-sided Default layer (set if interSignalCount/interCounterCount >
  // 0)
  // ----
  uint64_t *signalBuffer; // GPU memory (flagcxMemAlloc), [signalCount] entries
  uint64_t
      *shadowBuffer; // GPU memory (local only, no MR), [signalCount] entries
  uint64_t
      *counterBuffer; // GPU memory (flagcxMemAlloc), [counterCount] entries
  int signalCount;
  int counterCount;
  int contextCount; // = reqs.interContextCount (default 4)
  // Host-only: MR handles + staging for cleanup
  void *signalBufferMr;        // MR handle for signalBuffer
  void *counterBufferMr;       // MR handle for counterBuffer
  void *putValueStagingBuffer; // 8 bytes host-pinned, MR registered
  void *putValueStagingMr;     // MR handle for staging buffer

  // ---- Vendor device comm (set if adaptor->devCommCreate succeeds, else NULL)
  // ----
  flagcxInnerDevComm_t devComm; // Typed vendor handle (per-adaptor struct)
};

// ============================================================
// Section 2: flagcxDevMemInternal — Host-Side Memory Handle
//
// Backing struct for flagcxDevMem_t.
// Created by flagcxDevMemCreate, freed by flagcxDevMemDestroy.
// Unified capability-based design: rawPtr always populated,
// IPC and Window layers added when available.
// Capabilities detected by null-checks:
//   devPeerPtrs != nullptr  → IPC available
//   window != nullptr       → Window available (Vendor or default)
// ============================================================
struct flagcxDevMemInternal {
  // ---- Baseline (always set) ----
  void *rawPtr;   // = buff parameter
  bool hasWindow; // true if any window layer is available (basic or symmetric)
  bool isSymmetric; // true only for FLAGCX_WIN_COLL_SYMMETRIC (enables
                    // one-sided)

  // ---- Per-comm MR layer (set by flagcxDevMemCreate from handle table) ----
  int mrIndex; // index into heteroComm->oneSideHandles (-1 if not registered)
  uintptr_t mrBase; // handles[mrIndex]->baseVas[myRank] (cached for device)

  // ---- IPC layer (set if IPC exchange succeeds, else nullptr) ----
  void **devPeerPtrs; // cached from comm->ipcTable[ipcIndex].devPeerPtrs
  int ipcIndex;       // index into comm->ipcTable (-1 if no IPC)
  int intraRank;      // this rank's local rank index (for IPC local pointer)

  // ---- Window layer (opaque pointer to DeviceAPI::Window) ----
  void *window;    // Points to vendor Window or defaultDeviceImpl::Window
                   // (fallback)
  void *winHandle; // Host-side handle for cleanup
};
#ifndef FLAGCX_DEV_MEM_T_DEFINED
#define FLAGCX_DEV_MEM_T_DEFINED
typedef struct flagcxDevMemInternal *flagcxDevMem_t;
#endif

// ============================================================
// Section 3: flagcxDevComm — Device Communicator (kernel-facing)
//
// Value type passed to kernels by value.
// Pure wrapper around DeviceAPI::Comm which contains all fields.
// On Vendor: Comm = vendor Comm
// On default: Comm = {rank, nRanks, fifoBuffer, barrierPeers, ...}
// ============================================================
struct flagcxDevComm {
  typename DeviceAPI::Comm _commBase;

  // Wrapper-level fields needed by FIFO encoding on all paths.
  // Populated from flagcxDevCommInternal; safe to be 0 when unused.
  int _signalCount;
  int _counterCount;
  int _contextCount;
  int _nInterPeers;

  FLAGCX_HOST_DEVICE_INLINE flagcxDevComm()
      : _commBase(), _signalCount(0), _counterCount(0), _contextCount(0),
        _nInterPeers(0) {}

  FLAGCX_HOST_DEVICE_INLINE flagcxDevComm(const flagcxDevCommInternal &di)
      : _signalCount(di.signalCount), _counterCount(di.counterCount),
        _contextCount(di.contextCount), _nInterPeers(di.nInterPeers) {
    if (di.devComm) {
      _commBase = *(typename DeviceAPI::Comm *)di.devComm;
    } else {
      // Default: populate _commBase directly from handle fields.
      // Vendor path: no-op (devComm pointer always set).
      // Dispatch resolved at compile time via DeviceAPI::Comm.
      DeviceAPI::Comm::populateFromInternal(_commBase, di);
    }
  }

  // Accessors delegate to _commBase member functions
  FLAGCX_DEVICE_INLINE_DECORATOR int getIntraRank() const {
    return _commBase.getIntraRank();
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int getIntraSize() const {
    return _commBase.getIntraSize();
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int getRank() const {
    return _commBase.getRank();
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int getSize() const {
    return _commBase.getSize();
  }
  FLAGCX_DEVICE_INLINE_DECORATOR void *getFifoBuffer(int contextId) const {
    return _commBase.getFifoBuffer(contextId);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR typename DeviceAPI::Multimem
  getMulticastHandle() const {
    return _commBase.getMulticastHandle();
  }
};

// ============================================================
// Section 4: flagcxDevMem — Device-Side Memory Handle
//
// Value type passed to kernels by value.
// Pure wrapper around DeviceAPI::Window which contains all fields.
// On Vendor: Window = vendor Window
// On default: Window = {rawPtr, peerPtrs, intraRank, mrBase, mrIndex}
// ============================================================
struct flagcxDevMem {
  typename DeviceAPI::Window _winBase;
  void *_rawPtr;

  FLAGCX_HOST_DEVICE_INLINE flagcxDevMem() : _winBase(), _rawPtr(nullptr) {}

  FLAGCX_HOST_DEVICE_INLINE flagcxDevMem(const flagcxDevMemInternal &di)
      : _rawPtr(di.rawPtr) {
    if (di.window)
      _winBase = *(typename DeviceAPI::Window *)di.window;
  }

  FLAGCX_HOST_DEVICE_INLINE bool hasWindow() const {
    return _winBase.hasAccess();
  }
  FLAGCX_HOST_DEVICE_INLINE void *getRawPtr() const { return _rawPtr; }
  FLAGCX_HOST_DEVICE_INLINE void **getDevPeerPtrs() const {
    return _winBase.getDevPeerPtrs();
  }
  FLAGCX_HOST_DEVICE_INLINE int getMrIndex() const {
    return _winBase.getMrIndex();
  }
};

// ============================================================
// Section 4b: flagcxTeam_t — Team Descriptor
//
// Represents a subset of ranks (intra-node, inter-node, etc.).
// Pure wrapper around DeviceAPI::Team.
// ============================================================
struct flagcxTeam {
  typename DeviceAPI::Team _teamBase;

  FLAGCX_HOST_DEVICE_INLINE flagcxTeam() : _teamBase() {}
  FLAGCX_HOST_DEVICE_INLINE flagcxTeam(int nr, int r, int s) {
    _teamBase.nRanks = nr;
    _teamBase.rank = r;
    _teamBase.stride = s;
  }

  FLAGCX_HOST_DEVICE_INLINE int nRanks() const { return _teamBase.nRanks; }
  FLAGCX_HOST_DEVICE_INLINE int rank() const { return _teamBase.rank; }
  FLAGCX_HOST_DEVICE_INLINE int stride() const { return _teamBase.stride; }
};
typedef struct flagcxTeam flagcxTeam_t;

// ============================================================
// Section 4c: flagcxMulticastHandle — Multicast Memory Handle
//
// Pure wrapper around DeviceAPI::Multimem.
// On Vendor: Multimem = vendor MultimemHandle
// On default: Multimem = {mcBasePtr}
// ============================================================
struct flagcxMulticastHandle {
  typename DeviceAPI::Multimem _multimemBase;

  FLAGCX_HOST_DEVICE_INLINE flagcxMulticastHandle() : _multimemBase() {}
};
typedef struct flagcxMulticastHandle flagcxMulticastHandle_t;

// ============================================================
// Section 4d: Barrier Handle Types
//
// flagcxIntraBarrierHandle → vendor intra-barrier handle (Vendor)
// flagcxInterBarrierHandle → vendor inter-barrier handle (Vendor)
// Default: placeholder structs (no resource-handle model yet).
// ============================================================
struct flagcxIntraBarrierHandle {
  typename DeviceAPI::IntraBarrierHandle _base;
};
typedef struct flagcxIntraBarrierHandle flagcxIntraBarrierHandle_t;

struct flagcxInterBarrierHandle {
  typename DeviceAPI::InterBarrierHandle _base;
};
typedef struct flagcxInterBarrierHandle flagcxInterBarrierHandle_t;

// Team tag types for barrier session constructors are defined in comm_traits.h

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
// On Vendor: forwards to vendor team functions via _commBase.
// On default: computes from baseline fields in _commBase.
// No #ifdef — DeviceAPI resolves at compile time.
// ============================================================
FLAGCX_DEVICE_INLINE_DECORATOR
flagcxTeam flagcxTeamIntra(const flagcxDevComm &devComm) {
  flagcxTeam team;
  team._teamBase.nRanks = devComm.getIntraSize();
  team._teamBase.rank = devComm.getIntraRank();
  team._teamBase.stride = 1;
  return team;
}
FLAGCX_DEVICE_INLINE_DECORATOR
flagcxTeam flagcxTeamWorld(const flagcxDevComm &devComm) {
  flagcxTeam team;
  team._teamBase.nRanks = devComm.getSize();
  team._teamBase.rank = devComm.getRank();
  team._teamBase.stride = 1;
  return team;
}
FLAGCX_DEVICE_INLINE_DECORATOR
flagcxTeam flagcxTeamInter(const flagcxDevComm &devComm) {
  flagcxTeam team;
  team._teamBase.nRanks = devComm.getSize() / devComm.getIntraSize();
  team._teamBase.rank = devComm.getRank() / devComm.getIntraSize();
  team._teamBase.stride = devComm.getIntraSize();
  return team;
}

// ---- Team Algebra (pure arithmetic on {nRanks, rank, stride}) ----
// These 5 functions are identical on all tiers — no vendor delegation needed.

// Is team b's bPeer also a member of team a?
FLAGCX_HOST_DEVICE_INLINE bool flagcxTeamRankIsMember(flagcxTeam a,
                                                      flagcxTeam b, int bPeer) {
  int wrank = (bPeer - b._teamBase.rank) * b._teamBase.stride;
  int adelta = wrank / a._teamBase.stride;
  int amod = wrank % a._teamBase.stride;
  int arank = a._teamBase.rank + adelta;
  return 0 <= arank && arank < a._teamBase.nRanks && amod == 0;
}

// Convert team b's bPeer to team a's rank.
FLAGCX_HOST_DEVICE_INLINE int flagcxTeamRankToTeam(flagcxTeam a, flagcxTeam b,
                                                   int bPeer) {
  int wrank = (bPeer - b._teamBase.rank) * b._teamBase.stride;
  int adelta = wrank / a._teamBase.stride;
  int arank = a._teamBase.rank + adelta;
  return arank;
}

// Extract inner sub-team (first innerSize ranks per stride group).
FLAGCX_HOST_DEVICE_INLINE flagcxTeam flagcxTeamInnerFactor(flagcxTeam parent,
                                                           int innerSize) {
  flagcxTeam ans;
  ans._teamBase.nRanks = innerSize;
  ans._teamBase.rank = parent._teamBase.rank % innerSize;
  ans._teamBase.stride = parent._teamBase.stride;
  return ans;
}

// Extract outer sub-team (stride groups).
FLAGCX_HOST_DEVICE_INLINE flagcxTeam flagcxTeamOuterFactor(flagcxTeam parent,
                                                           int innerSize) {
  flagcxTeam ans;
  ans._teamBase.nRanks = parent._teamBase.nRanks / innerSize;
  ans._teamBase.rank = parent._teamBase.rank / innerSize;
  ans._teamBase.stride = parent._teamBase.stride * innerSize;
  return ans;
}

// Return the index'th element of parent minus subset (set difference).
FLAGCX_HOST_DEVICE_INLINE int
flagcxTeamRankInDifference(flagcxTeam parent, flagcxTeam subset, int index) {
  int stride = subset._teamBase.stride / parent._teamBase.stride;
  int below = parent._teamBase.rank - subset._teamBase.rank * stride;
  if (stride < 0) {
    stride = -stride;
    below -= (subset._teamBase.nRanks - 1) * stride;
  }
  if (index < below) {
    return index;
  } else if (index - below < (subset._teamBase.nRanks - 1) * (stride - 1)) {
    return below + 1 + ((index - below) / (stride - 1)) * stride +
           (index - below) % (stride - 1);
  } else {
    return below + 1 + (subset._teamBase.nRanks - 1) * stride +
           (index - below - (subset._teamBase.nRanks - 1) * (stride - 1));
  }
}

// ---- Comm-dependent team conversions ----

// Convert team rank to world rank.
FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxTeamRankToWorld(const flagcxDevComm &devComm, flagcxTeam team, int rank) {
  return devComm.getRank() +
         (rank - team._teamBase.rank) * team._teamBase.stride;
}

// Convert team rank to intra-node rank.
FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxTeamRankToIntra(const flagcxDevComm &devComm, flagcxTeam team, int rank) {
  return devComm.getIntraRank() +
         (rank - team._teamBase.rank) * team._teamBase.stride;
}

// ============================================================
// Section 6: Cooperative Group Types
//
// Platform-neutral cooperative groups for device-side synchronization.
// Naming: "Tile" = N PEs cooperating (avoids vendor-specific
//         Warp/Wave/Subgroup terms).
//
// All implementations live in CommTraits; these are thin wrappers.
// ============================================================

// ---- 6a. flagcxCoopBlock — CTA-level cooperative group ----
struct flagcxCoopBlock {
  typename DeviceAPI::CoopBlock _base;

  FLAGCX_HOST_DEVICE_INLINE flagcxCoopBlock() : _base() {}

  FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
    return _base.threadRank();
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return _base.size(); }
  FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _base.sync(); }
};

// ---- 6b. flagcxCoopTile<N> — Tile of N threads within a warp ----
template <int N>
struct flagcxCoopTile {
  typename DeviceAPI::template CoopTile<N> _base;

  FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
    return _base.threadRank();
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return N; }
  FLAGCX_DEVICE_INLINE_DECORATOR uint32_t laneMask() const {
    return _base.laneMask();
  }
  FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _base.sync(); }
};

// ---- 6c. flagcxCoopThread — single-thread alias ----
typedef flagcxCoopTile<1> flagcxCoopThread;

// ---- 6d. flagcxCoopWarp — full-warp alias ----
typedef flagcxCoopTile<FLAGCX_SIMT_WIDTH> flagcxCoopWarp;

// ---- 6e. flagcxCoopTileSpan — consecutive tiles with named barrier ----
struct flagcxCoopTileSpan {
  typename DeviceAPI::CoopTileSpan _base;

  FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopTileSpan(int t0, int nTiles, int id)
      : _base(t0, nTiles, id) {}

  FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
    return _base.threadRank();
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return _base.size(); }
  FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _base.sync(); }
};

// ---- 6f. flagcxCoopLanes — arbitrary lane bitmask ----
struct flagcxCoopLanes {
  typename DeviceAPI::CoopLanes _base;

  FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopLanes(uint32_t lmask = 0xffffffffu)
      : _base(lmask) {}

  FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
    return _base.threadRank();
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return _base.size(); }
  FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _base.sync(); }
  FLAGCX_DEVICE_INLINE_DECORATOR uint32_t getLmask() const {
    return _base.getLmask();
  }
};

// ---- 6g. flagcxCoopAny — type-erased cooperative group ----
struct flagcxCoopAny {
  typename DeviceAPI::CoopAny _base;

  flagcxCoopAny() = default;
  flagcxCoopAny(flagcxCoopAny const &) = default;

  FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopAny(flagcxCoopBlock b)
      : _base(b._base) {}
  template <int N>
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopAny(flagcxCoopTile<N> t)
      : _base(t._base) {}
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopAny(flagcxCoopTileSpan s)
      : _base(s._base) {}
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopAny(flagcxCoopLanes l)
      : _base(l._base) {}

  FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
    return _base.threadRank();
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return _base.size(); }
  FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _base.sync(); }
};

// ---- 6h. Free functions ----

// flagcxCoopGetLaneMask: get the active lane bitmask for a cooperative group
template <int N>
FLAGCX_DEVICE_INLINE_DECORATOR uint32_t
flagcxCoopGetLaneMask(flagcxCoopTile<N> coop) {
  return coop.laneMask();
}
FLAGCX_DEVICE_INLINE_DECORATOR uint32_t flagcxCoopGetLaneMask(flagcxCoopBlock) {
  return 0xffffffffu;
}
FLAGCX_DEVICE_INLINE_DECORATOR uint32_t
flagcxCoopGetLaneMask(flagcxCoopLanes coop) {
  return coop.getLmask();
}
FLAGCX_DEVICE_INLINE_DECORATOR uint32_t
flagcxCoopGetLaneMask(flagcxCoopTileSpan) {
  return 0xffffffffu;
}

// flagcxCoopIsThread: compile-time check if group is a single thread
template <int N>
FLAGCX_DEVICE_INLINE_DECORATOR bool flagcxCoopIsThread(flagcxCoopTile<N>) {
  return N == 1;
}
FLAGCX_DEVICE_INLINE_DECORATOR bool flagcxCoopIsThread(flagcxCoopBlock) {
  return false;
}
FLAGCX_DEVICE_INLINE_DECORATOR bool flagcxCoopIsThread(flagcxCoopLanes) {
  return false;
}
FLAGCX_DEVICE_INLINE_DECORATOR bool flagcxCoopIsThread(flagcxCoopTileSpan) {
  return false;
}

// flagcxCoopWithinTile: compile-time check if group fits within a single tile
template <int N>
FLAGCX_DEVICE_INLINE_DECORATOR bool flagcxCoopWithinTile(flagcxCoopTile<N>) {
  return true;
}
FLAGCX_DEVICE_INLINE_DECORATOR bool flagcxCoopWithinTile(flagcxCoopBlock) {
  return false;
}
FLAGCX_DEVICE_INLINE_DECORATOR bool flagcxCoopWithinTile(flagcxCoopLanes) {
  return true;
}
FLAGCX_DEVICE_INLINE_DECORATOR bool flagcxCoopWithinTile(flagcxCoopTileSpan) {
  return false;
}

// flagcxCoopCoalesced: get a cooperative group of active/safe threads
FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopLanes flagcxCoopCoalesced() {
  return flagcxCoopLanes{DeviceAPI::Intrin::activemask()};
}
template <typename Coop>
FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopWarp flagcxCoopCoalesced(Coop) {
  return flagcxCoopWarp();
}
FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopLanes
flagcxCoopCoalesced(flagcxCoopLanes coop) {
  return coop;
}
template <int N>
FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopTile<N>
flagcxCoopCoalesced(flagcxCoopTile<N> coop) {
  return coop;
}

// ============================================================
// Section 7: flagcxDevBarrier — Barrier Session Wrappers
//
// Thin wrappers delegating to DeviceAPI::Barrier<Tag>.
// No #ifdef FLAGCX_DEVICE_API_VENDOR — dispatch resolved by CommTraits.
// ============================================================

// Primary template
template <typename Tag, typename Coop>
struct flagcxDevBarrier;

// ---- Intra ----
template <typename Coop>
struct flagcxDevBarrier<flagcxTeamTagIntra, Coop> {
  typename DeviceAPI::template Barrier<flagcxTeamTagIntra, Coop> _impl;

  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxDevBarrier() : _impl() {}

  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxDevBarrier(Coop coop, const flagcxDevComm &devComm, flagcxTeam team,
                   uint32_t index, bool multimem = false,
                   flagcxMulticastHandle mcHandle = {})
      : _impl(coop, devComm._commBase, team._teamBase, index, multimem,
              mcHandle._multimemBase) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    _impl.arrive(order);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    _impl.wait(order);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    _impl.sync(order);
  }
};

// ============================================================
// Section 8: Pointer Access Functions (Inline Wrappers)
//
// All functions delegate to _winBase member functions — no #ifdef branches.
// On Vendor: forwards to vendor pointer functions via _winBase.
// On default: uses IPC peerPtrs / rawPtr fallback.
// ============================================================
FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetPeerPointer(const flagcxDevMem &mem, size_t offset, flagcxTeam team,
                     int peer) {
  return mem._winBase.getPeerPointer(offset, team._teamBase, peer);
}

FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetLocalPointer(const flagcxDevMem &mem, size_t offset) {
  return mem._winBase.getLocalPointer(offset);
}

FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetMulticastPointer(const flagcxDevMem &mem, size_t offset,
                          const flagcxDevComm &devComm) {
  return mem._winBase.getMulticastPointer(offset, devComm.getMulticastHandle());
}

// ---- Additional pointer functions ----

// Peer pointer without team parameter.
FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetPeerPointer(const flagcxDevMem &mem, size_t offset, int peer) {
  // Without team, treat as intra-node access
  return mem._winBase.getIntraPointer(offset, peer);
}

// Intra-node rank pointer.
FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetIntraPointer(const flagcxDevMem &mem, size_t offset, int peer) {
  return mem._winBase.getIntraPointer(offset, peer);
}

// Multicast pointer with explicit MulticastHandle.
FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetMulticastPointer(const flagcxDevMem &mem, size_t offset,
                          flagcxMulticastHandle_t mmHandle) {
  return mem._winBase.getMulticastPointer(offset, mmHandle._multimemBase);
}

// Reverse lookup: raw pointer → flagcxDevMem.
// Vendor: cooperative search through vendor window table.
// Default: not supported (returns empty flagcxDevMem).
template <typename Coop>
FLAGCX_DEVICE_INLINE_DECORATOR flagcxDevMem
flagcxFindMem(Coop coop, const flagcxDevComm &devComm, void const *ptr) {
  flagcxDevMem result;
  (void)coop;
  (void)devComm;
  (void)ptr;
  return result;
}

// ============================================================
// Section 8b: flagcxSymPtr<T> — Typed Symmetric Pointer
//
// Value type storing {flagcxDevMem, offset}. Provides typed
// pointer methods and type-aware arithmetic.
// Mirrors vendor's SymPtr<T>.
// ============================================================
template <typename T>
struct flagcxSymPtr {
  flagcxDevMem mem;
  size_t offset;

  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr() : mem(), offset(0) {}
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr(flagcxDevMem m, size_t off)
      : mem(m), offset(off) {}

  // Type conversion (e.g. flagcxSymPtr<float> → flagcxSymPtr<char>)
  template <typename U>
  FLAGCX_HOST_DEVICE_INLINE operator flagcxSymPtr<U>() const {
    return {mem, offset};
  }

  // Typed pointer methods (delegate to free functions)
  FLAGCX_DEVICE_INLINE_DECORATOR T *localPtr() const {
    return (T *)flagcxGetLocalPointer(mem, offset);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR T *peerPtr(flagcxTeam team, int peer) const {
    return (T *)flagcxGetPeerPointer(mem, offset, team, peer);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR T *peerPtr(int peer) const {
    return (T *)flagcxGetPeerPointer(mem, offset, peer);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR T *intraPtr(int peer) const {
    return (T *)flagcxGetIntraPointer(mem, offset, peer);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR T *
  multicastPtr(const flagcxDevComm &devComm) const {
    return (T *)flagcxGetMulticastPointer(mem, offset, devComm);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR T *
  multicastPtr(flagcxMulticastHandle_t mmHandle) const {
    return (T *)flagcxGetMulticastPointer(mem, offset, mmHandle);
  }

  // Type-aware pointer arithmetic (integer math, no UB)
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> &operator+=(int d) {
    offset += d * sizeof(T);
    return *this;
  }
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> &operator+=(unsigned int d) {
    offset += d * sizeof(T);
    return *this;
  }
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> &operator+=(long d) {
    offset += d * sizeof(T);
    return *this;
  }
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> &operator+=(unsigned long d) {
    offset += d * sizeof(T);
    return *this;
  }
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> &operator+=(long long d) {
    offset += d * sizeof(T);
    return *this;
  }
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> &operator+=(unsigned long long d) {
    offset += d * sizeof(T);
    return *this;
  }

  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> &operator-=(int d) {
    offset -= d * sizeof(T);
    return *this;
  }
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> &operator-=(unsigned int d) {
    offset -= d * sizeof(T);
    return *this;
  }
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> &operator-=(long d) {
    offset -= d * sizeof(T);
    return *this;
  }
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> &operator-=(unsigned long d) {
    offset -= d * sizeof(T);
    return *this;
  }
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> &operator-=(long long d) {
    offset -= d * sizeof(T);
    return *this;
  }
  FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> &operator-=(unsigned long long d) {
    offset -= d * sizeof(T);
    return *this;
  }
};

// Free operators for flagcxSymPtr<T>
template <typename T, typename Int>
FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> operator+(flagcxSymPtr<T> p, Int d) {
  return p += d;
}
template <typename T, typename Int>
FLAGCX_HOST_DEVICE_INLINE flagcxSymPtr<T> operator-(flagcxSymPtr<T> p, Int d) {
  return p -= d;
}
template <typename T>
FLAGCX_HOST_DEVICE_INLINE ptrdiff_t operator-(flagcxSymPtr<T> a,
                                              flagcxSymPtr<T> b) {
  return ((ptrdiff_t)a.offset - (ptrdiff_t)b.offset) / (ptrdiff_t)sizeof(T);
}
template <typename T>
FLAGCX_HOST_DEVICE_INLINE bool operator==(flagcxSymPtr<T> a,
                                          flagcxSymPtr<T> b) {
  return a.mem._winBase == b.mem._winBase && a.offset == b.offset;
}
template <typename T>
FLAGCX_HOST_DEVICE_INLINE bool operator!=(flagcxSymPtr<T> a,
                                          flagcxSymPtr<T> b) {
  return !(a == b);
}

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

// ============================================================
// Section 10: flagcxDevNet — Device Network Transport
//
// Thin wrapper around DeviceAPI::Transport. Adds flagcxDevComm-accepting
// ctor, defined here after flagcxDevComm and flagcxDevMem are fully defined.
// ============================================================
struct flagcxDevNet : DeviceAPI::Transport {
  int _nInterPeers;

  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxDevNet(const flagcxDevComm &devComm, int idx)
      : DeviceAPI::Transport(
            devComm._commBase,
            devComm._contextCount > 0 ? idx % devComm._contextCount : 0),
        _nInterPeers(devComm._nInterPeers) {}

  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t readSignal(
      flagcxDevNetSignal_t signalId, int bits = 64,
      flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcquire) const {
    return DeviceAPI::Transport::readSignal(signalId, bits, order);
  }

  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR void flush(
      Coop coop,
      flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcquire) const {
    DeviceAPI::Transport::flush(coop._base, order);
  }

  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR void waitSignal(
      Coop coop, flagcxDevNetSignal_t signalId, uint64_t least, int bits = 64,
      flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcquire) const {
    DeviceAPI::Transport::waitSignal(coop._base, signalId, least, bits, order);
  }

  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR void waitSignalMeetShadow(
      Coop coop, flagcxDevNetSignal_t signalId, int bits = 64,
      flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcquire) const {
    DeviceAPI::Transport::waitSignalMeetShadow(coop._base, signalId, bits,
                                               order);
  }

  template <typename Coop, typename Uint>
  FLAGCX_DEVICE_INLINE_DECORATOR void waitSignalFollowShadow(
      Coop coop, flagcxDevNetSignal_t signalId, Uint leastDelta, Uint *before,
      Uint *delta, int bits = 64,
      flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcquire) const {
    DeviceAPI::Transport::waitSignalFollowShadow(
        coop._base, signalId, leastDelta, before, delta, bits, order);
  }

  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR void waitCounter(
      Coop coop, flagcxDevNetCounter_t counterId, uint64_t least, int bits = 56,
      flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcquire) const {
    DeviceAPI::Transport::waitCounter(coop._base, counterId, least, bits,
                                      order);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR uint64_t readCounter(
      flagcxDevNetCounter_t counterId, int bits = 56,
      flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcquire) const {
    return DeviceAPI::Transport::readCounter(counterId, bits, order);
  }

  // ---- Two-sided ----
  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t
  send(Coop coop, const flagcxDevMem &mem, size_t offset, size_t count,
       flagcxDataType_t datatype, int peer) const {
    return DeviceAPI::Transport::send(coop._base, mem._winBase, offset, count,
                                      datatype, peer);
  }

  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t
  recv(Coop coop, const flagcxDevMem &mem, size_t offset, size_t count,
       flagcxDataType_t datatype, int peer) const {
    return DeviceAPI::Transport::recv(coop._base, mem._winBase, offset, count,
                                      datatype, peer);
  }

  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t term(Coop coop) const {
    return DeviceAPI::Transport::term(coop._base);
  }

  template <typename Coop>
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t wait(Coop coop) const {
    return DeviceAPI::Transport::wait(coop._base);
  }

  // ---- One-sided: put ----
  template <typename RemoteAction = flagcxDevNet_None,
            typename LocalAction = flagcxDevNet_None,
            typename Coop = flagcxCoopBlock, typename Desc = flagcxDevNet_None>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  put(flagcxTeam team, int peer, const flagcxDevMem &dst, size_t dstOffset,
      const flagcxDevMem &src, size_t srcOffset, size_t bytes,
      RemoteAction ra = flagcxDevNet_None{},
      LocalAction la = flagcxDevNet_None{}, Coop coop = flagcxCoopBlock{},
      Desc desc = flagcxDevNet_None{},
      flagcxDeviceScope_t ar = flagcxDeviceScopeThread,
      flagcxDeviceScope_t es = flagcxDeviceScopeDevice) const {
    DeviceAPI::Transport::put(team._teamBase, peer, dst._winBase, dstOffset,
                              src._winBase, srcOffset, bytes, ra, la,
                              coop._base, desc, ar, es);
  }

  // ---- One-sided: signal ----
  template <typename RemoteAction, typename Coop = flagcxCoopBlock,
            typename Desc = flagcxDevNet_None>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  signal(flagcxTeam team, int peer, RemoteAction ra,
         Coop coop = flagcxCoopBlock{}, Desc desc = flagcxDevNet_None{},
         flagcxDeviceScope_t ar = flagcxDeviceScopeThread,
         flagcxDeviceScope_t es = flagcxDeviceScopeDevice) const {
    DeviceAPI::Transport::signal(team._teamBase, peer, ra, coop._base, desc, ar,
                                 es);
  }

  // ---- One-sided: putValue ----
  template <typename T, typename RemoteAction = flagcxDevNet_None,
            typename Coop = flagcxCoopBlock, typename Desc = flagcxDevNet_None>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  putValue(flagcxTeam team, int peer, const flagcxDevMem &dst, size_t dstOffset,
           T value, RemoteAction ra = flagcxDevNet_None{},
           Coop coop = flagcxCoopBlock{}, Desc desc = flagcxDevNet_None{},
           flagcxDeviceScope_t ar = flagcxDeviceScopeThread,
           flagcxDeviceScope_t es = flagcxDeviceScopeDevice) const {
    DeviceAPI::Transport::putValue(team._teamBase, peer, dst._winBase,
                                   dstOffset, value, ra, coop._base, desc, ar,
                                   es);
  }

  // ---- One-sided: get (fallback only) ----
  template <typename Coop = flagcxCoopBlock>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  get(flagcxTeam team, int peer, const flagcxDevMem &src, size_t srcOffset,
      const flagcxDevMem &dst, size_t dstOffset, size_t bytes,
      Coop coop = flagcxCoopBlock{}) const {
    DeviceAPI::Transport::get(team._teamBase, peer, src._winBase, srcOffset,
                              dst._winBase, dstOffset, bytes, coop._base);
  }
};

// ============================================================
// Section 11: flagcxDevBarrier<flagcxTeamTagInter> — Inter-Node Barrier
// ============================================================
// ---- Inter ----
template <typename Coop>
struct flagcxDevBarrier<flagcxTeamTagInter, Coop> {
  typename DeviceAPI::template Barrier<flagcxTeamTagInter, Coop> _impl;

  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxDevBarrier() : _impl() {}

  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxDevBarrier(Coop coop, const flagcxDevNet &trans, flagcxTeam team,
                   uint32_t index)
      : _impl(coop, trans, trans._dc, team._teamBase, index,
              trans._nInterPeers) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
         flagcxTransportFenceLevel fence = flagcxTransportFenceLevel::Relaxed) {
    _impl.arrive(order, fence);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxTransportFenceLevel fence = flagcxTransportFenceLevel::Relaxed) {
    _impl.wait(order, fence);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxTransportFenceLevel fence = flagcxTransportFenceLevel::Relaxed) {
    _impl.sync(order, fence);
  }
};

// ---- World ----
template <typename Coop>
struct flagcxDevBarrier<flagcxTeamTagWorld, Coop> {
  typename DeviceAPI::template Barrier<flagcxTeamTagWorld, Coop> _impl;

  // World barrier (intra + inter)
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxDevBarrier(Coop coop, flagcxTeamTagWorld, const flagcxDevNet &trans,
                   uint32_t index, bool multimem = false)
      : _impl(coop, flagcxTeamTagWorld{}, trans, trans._dc, index, multimem,
              trans._nInterPeers) {}

  // Intra-only barrier
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxDevBarrier(Coop coop, flagcxTeamTagIntra, const flagcxDevNet &trans,
                   uint32_t index, bool multimem = false)
      : _impl(coop, flagcxTeamTagIntra{}, trans, trans._dc, index, multimem,
              0) {}

  // Inter-only barrier
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxDevBarrier(Coop coop, flagcxTeamTagInter, const flagcxDevNet &trans,
                   uint32_t index, bool multimem = false)
      : _impl(coop, flagcxTeamTagInter{}, trans, trans._dc, index, multimem,
              trans._nInterPeers) {}

  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
         flagcxTransportFenceLevel fence = flagcxTransportFenceLevel::Relaxed) {
    _impl.arrive(order, fence);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxTransportFenceLevel fence = flagcxTransportFenceLevel::Relaxed) {
    _impl.wait(order, fence);
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxTransportFenceLevel fence = flagcxTransportFenceLevel::Relaxed) {
    _impl.sync(order, fence);
  }
};

// Backward-compatible aliases
template <typename Coop>
using flagcxIntraBarrierSession = flagcxDevBarrier<flagcxTeamTagIntra, Coop>;
template <typename Coop>
using flagcxInterBarrierSession = flagcxDevBarrier<flagcxTeamTagInter, Coop>;
template <typename Coop>
using flagcxBarrierSession = flagcxDevBarrier<flagcxTeamTagWorld, Coop>;

#endif // FLAGCX_DEVICE_COMPILE (Sections 9b-12)

#endif // FLAGCX_DEVICE_API_H_
