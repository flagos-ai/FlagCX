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

#include <cstddef> // ptrdiff_t, size_t

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
#define FLAGCX_MAX_INTER_PEERS 256

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

  // ---- IPC barrier layer (set if IPC barrier setup succeeds, else nullptr)
  // ----
  uint64_t *
      *barrierPeers; // device pointer to array of nLocalRanks device pointers
  uint64_t *localBarrierFlags; // this rank's signal array (CTA_COUNT entries)
  uint64_t barrierEpoch; // monotonically increasing, set by host before launch
  // Host-side cleanup bookkeeping (not passed to kernel)
  void **peerBarrierPtrs; // host array of IPC-mapped pointers (for close)
  int *localRankToRank;   // intra-node rank mapping (for IPC exchange)
  int nLocalRanks;

  // ---- Inter-node signal relay (set if nInterPeers > 0, else nullptr) ----
  uint64_t *interSignalFlags;     // device pointer (from hostGetDevicePointer)
  uint64_t *interSignalFlagsHost; // host pointer (for recv thread + dealloc)
  uint64_t interBarrierEpoch; // inter-node epoch (separate from barrierEpoch)
  int nInterPeers;            // number of inter-node peers (set on ALL ranks)
  bool isInterLeader;         // true only on localRank 0 (manages connections)
  int *interPeerRanks;        // global ranks of inter-node peers
  // netAdaptor connections for signal relay
  void **signalSendComms; // [nInterPeers] netAdaptor sendComm
  void **signalRecvComms; // [nInterPeers] netAdaptor recvComm
  // Signal recv thread
  pthread_t signalRecvThread;
  int signalRecvStop;
  // MR + staging buffers (per-peer for IB PD safety)
  void **signalSendMrs;
  void **signalRecvMrs;
  char *signalSendBufs; // [nInterPeers * 8]
  char *signalRecvBufs; // [nInterPeers * 8]
  // netAdaptor pointer (cached for recv thread)
  void *netAdaptorPtr;

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
  uint64_t **_barrierPeers;
  uint64_t _barrierEpoch;

  // ---- Inter-node signal relay (nullptr if single-node) ----
  uint64_t *_interSignalFlags; // device ptr to host-mapped inter signals
  int _nInterPeers;
  bool _isInterLeader;
  uint64_t _interBarrierEpoch;

#ifdef FLAGCX_DEVICE_API_NCCL
  // ---- NCCL layer (valid only if _hasBase is true) ----
  ncclDevComm _base;
#endif

  FLAGCX_HOST_DEVICE_INLINE flagcxDevComm()
      : _rank(0), _nRanks(0), _intraRank(0), _intraSize(0),
        _fifoBuffer(nullptr), _hasBase(false), _barrierPeers(nullptr),
        _barrierEpoch(0ULL), _interSignalFlags(nullptr), _nInterPeers(0),
        _isInterLeader(false), _interBarrierEpoch(0ULL)
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
        _barrierEpoch(di.barrierEpoch), _interSignalFlags(di.interSignalFlags),
        _nInterPeers(di.nInterPeers), _isInterLeader(di.isInterLeader),
        _interBarrierEpoch(di.interBarrierEpoch)
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

// ============================================================
// Section 4c: flagcxMulticastHandle — Multicast Memory Handle
//
// Wraps ncclMultimemHandle on Tier 1. Empty shell on Tier 2.
// Naming: NCCL "Multimem" → FlagCX "Multicast".
// ============================================================
struct flagcxMulticastHandle {
  void *mcBasePtr;

#ifdef FLAGCX_DEVICE_API_NCCL
  ncclMultimemHandle_t _base;

  FLAGCX_HOST_DEVICE_INLINE flagcxMulticastHandle()
      : mcBasePtr(nullptr), _base() {}
  FLAGCX_HOST_DEVICE_INLINE flagcxMulticastHandle(ncclMultimemHandle_t base)
      : mcBasePtr(base.mcBasePtr), _base(base) {}
  FLAGCX_HOST_DEVICE_INLINE operator ncclMultimemHandle_t() const {
    return _base;
  }
#else
  FLAGCX_HOST_DEVICE_INLINE flagcxMulticastHandle() : mcBasePtr(nullptr) {}
#endif
};
typedef struct flagcxMulticastHandle flagcxMulticastHandle_t;

// ============================================================
// Section 4d: Barrier Handle Types
//
// flagcxIntraBarrierHandle → ncclLsaBarrierHandle (Tier 1)
// flagcxInterBarrierHandle → ncclGinBarrierHandle (Tier 1)
// Tier 2: placeholder structs (no resource-handle model yet).
// ============================================================
struct flagcxIntraBarrierHandle {
#ifdef FLAGCX_DEVICE_API_NCCL
  ncclLsaBarrierHandle _impl;
#else
  int nBarriers;
#endif
};
typedef struct flagcxIntraBarrierHandle flagcxIntraBarrierHandle_t;

struct flagcxInterBarrierHandle {
#ifdef FLAGCX_DEVICE_API_NCCL
  ncclGinBarrierHandle _impl;
#else
  int placeholder;
#endif
};
typedef struct flagcxInterBarrierHandle flagcxInterBarrierHandle_t;

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

// ---- Team Algebra (pure arithmetic on {nRanks, rank, stride}) ----
// These 5 functions are identical on all tiers — no NCCL delegation needed.

// Is team b's bPeer also a member of team a?
FLAGCX_HOST_DEVICE_INLINE bool
flagcxTeamRankIsMember(flagcxTeam_t a, flagcxTeam_t b, int bPeer) {
  int wrank = (bPeer - b.rank) * b.stride;
  int adelta = wrank / a.stride;
  int amod = wrank % a.stride;
  int arank = a.rank + adelta;
  return 0 <= arank && arank < a.nRanks && amod == 0;
}

// Convert team b's bPeer to team a's rank.
FLAGCX_HOST_DEVICE_INLINE int flagcxTeamRankToTeam(flagcxTeam_t a,
                                                   flagcxTeam_t b, int bPeer) {
  int wrank = (bPeer - b.rank) * b.stride;
  int adelta = wrank / a.stride;
  int arank = a.rank + adelta;
  return arank;
}

// Extract inner sub-team (first innerSize ranks per stride group).
FLAGCX_HOST_DEVICE_INLINE flagcxTeam_t
flagcxTeamInnerFactor(flagcxTeam_t parent, int innerSize) {
  flagcxTeam_t ans;
  ans.nRanks = innerSize;
  ans.rank = parent.rank % innerSize;
  ans.stride = parent.stride;
  return ans;
}

// Extract outer sub-team (stride groups).
FLAGCX_HOST_DEVICE_INLINE flagcxTeam_t
flagcxTeamOuterFactor(flagcxTeam_t parent, int innerSize) {
  flagcxTeam_t ans;
  ans.nRanks = parent.nRanks / innerSize;
  ans.rank = parent.rank / innerSize;
  ans.stride = parent.stride * innerSize;
  return ans;
}

// Return the index'th element of parent minus subset (set difference).
FLAGCX_HOST_DEVICE_INLINE int flagcxTeamRankInDifference(flagcxTeam_t parent,
                                                         flagcxTeam_t subset,
                                                         int index) {
  int stride = subset.stride / parent.stride;
  int below = parent.rank - subset.rank * stride;
  if (stride < 0) {
    stride = -stride;
    below -= (subset.nRanks - 1) * stride;
  }
  if (index < below) {
    return index;
  } else if (index - below < (subset.nRanks - 1) * (stride - 1)) {
    return below + 1 + ((index - below) / (stride - 1)) * stride +
           (index - below) % (stride - 1);
  } else {
    return below + 1 + (subset.nRanks - 1) * stride +
           (index - below - (subset.nRanks - 1) * (stride - 1));
  }
}

// ---- DevComm-dependent team conversions ----

// Convert team rank to world rank.
FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxTeamRankToWorld(const flagcxDevComm &devComm, flagcxTeam_t team,
                      int rank) {
#ifdef FLAGCX_DEVICE_API_NCCL
  if (devComm._hasBase)
    return ncclTeamRankToWorld(devComm._base, team._base, rank);
#endif
  return devComm.getRank() + (rank - team.rank) * team.stride;
}

// Convert team rank to intra-node rank (NCCL "Lsa" → FlagCX "Intra").
FLAGCX_DEVICE_INLINE_DECORATOR int
flagcxTeamRankToIntra(const flagcxDevComm &devComm, flagcxTeam_t team,
                      int rank) {
#ifdef FLAGCX_DEVICE_API_NCCL
  if (devComm._hasBase)
    return ncclTeamRankToLsa(devComm._base, team._base, rank);
#endif
  return devComm.getIntraRank() + (rank - team.rank) * team.stride;
}

// ============================================================
// Section 6: Cooperative Group Types
//
// Platform-neutral cooperative groups for device-side synchronization.
// Naming: "Tile" = N PEs cooperating (avoids vendor-specific
//         Warp/Wave/Subgroup terms).
//
// 3-way guard: FLAGCX_DEVICE_API_NCCL → wrap NCCL types (Tier 1)
//              FLAGCX_SIMT_WIDTH       → SIMT intrinsics (Tier 2)
//              else                    → non-SIMT stubs
// ============================================================

// ---- 6a. flagcxCoopBlock — CTA-level cooperative group ----
struct flagcxCoopBlock {
#ifdef FLAGCX_DEVICE_API_NCCL
  ncclCoopCta _impl;

  FLAGCX_HOST_DEVICE_INLINE flagcxCoopBlock() : _impl() {}

  FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
    return _impl.thread_rank();
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return _impl.size(); }
  FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _impl.sync(); }

  // Implicit conversion for passthrough to NCCL APIs
  FLAGCX_HOST_DEVICE_INLINE operator ncclCoopCta() const { return _impl; }
#elif defined(FLAGCX_SIMT_WIDTH)
  FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
    return FLAGCX_THREAD_IDX_X;
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return FLAGCX_BLOCK_DIM_X; }
  FLAGCX_DEVICE_INLINE_DECORATOR void sync() { FLAGCX_DEVICE_SYNC_THREADS(); }
#else
  int threadRank() const { return FLAGCX_THREAD_IDX_X; }
  int size() const { return FLAGCX_BLOCK_DIM_X; }
  void sync() {}
#endif
};

// ---- 6b. flagcxCoopTile<N> — Tile of N threads within a warp ----
template <int N>
struct flagcxCoopTile {
#ifdef FLAGCX_DEVICE_API_NCCL
  ncclCoopTile<N> _impl;

  FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
    return _impl.thread_rank();
  }
  FLAGCX_DEVICE_INLINE_DECORATOR constexpr int size() const { return N; }
  FLAGCX_DEVICE_INLINE_DECORATOR uint32_t laneMask() const {
    return _impl.laneMask();
  }
  FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _impl.sync(); }
#elif defined(FLAGCX_SIMT_WIDTH)
  static_assert(N > 0 && (N & (N - 1)) == 0 && N <= FLAGCX_SIMT_WIDTH,
                "N must be a power of 2 and <= FLAGCX_SIMT_WIDTH");

  FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
    return flagcxLane() % N;
  }
  FLAGCX_DEVICE_INLINE_DECORATOR constexpr int size() const { return N; }
  FLAGCX_DEVICE_INLINE_DECORATOR uint32_t laneMask() const {
    return (0xffffffffu >> (32 - N)) << (flagcxLane() & -N);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR void sync() {
    if (N > 1)
      flagcxSyncwarp(laneMask());
  }
#else
  int threadRank() const { return 0; }
  constexpr int size() const { return N; }
  void sync() {}
#endif
};

// ---- 6c. flagcxCoopThread — single-thread alias ----
typedef flagcxCoopTile<1> flagcxCoopThread;

// ---- 6d. flagcxCoopWarp — full-warp alias (SIMT only) ----
#ifdef FLAGCX_SIMT_WIDTH
typedef flagcxCoopTile<FLAGCX_SIMT_WIDTH> flagcxCoopWarp;
#endif

// ---- 6e. flagcxCoopTileSpan — consecutive tiles with named barrier ----
#ifdef FLAGCX_SIMT_WIDTH
struct flagcxCoopTileSpan {
#ifdef FLAGCX_DEVICE_API_NCCL
  ncclCoopWarpSpan _impl;

  FLAGCX_DEVICE_INLINE_DECORATOR constexpr flagcxCoopTileSpan(int t0,
                                                              int nTiles,
                                                              int id)
      : _impl(t0, nTiles, id) {}

  FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
    return _impl.thread_rank();
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return _impl.size(); }
  FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _impl.sync(); }
#else
  uint32_t t0 : 8, nTiles : 8, id : 8;

  FLAGCX_DEVICE_INLINE_DECORATOR constexpr flagcxCoopTileSpan(int t0,
                                                              int nTiles,
                                                              int id)
      : t0(t0), nTiles(nTiles), id(id) {}

  FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
    return FLAGCX_THREAD_IDX_X - FLAGCX_SIMT_WIDTH * t0;
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int size() const {
    return FLAGCX_SIMT_WIDTH * nTiles;
  }
  FLAGCX_DEVICE_INLINE_DECORATOR void sync() {
    flagcxNamedBarrierSync(1 + id, FLAGCX_SIMT_WIDTH * nTiles);
  }
#endif
};
#endif // FLAGCX_SIMT_WIDTH

// ---- 6f. flagcxCoopLanes — arbitrary lane bitmask ----
#ifdef FLAGCX_SIMT_WIDTH
struct flagcxCoopLanes {
#ifdef FLAGCX_DEVICE_API_NCCL
  ncclCoopLanes _impl;

  FLAGCX_DEVICE_INLINE_DECORATOR constexpr flagcxCoopLanes(
      uint32_t lmask = 0xffffffffu)
      : _impl{lmask} {}

  FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
    return _impl.thread_rank();
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return _impl.size(); }
  FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _impl.sync(); }
  FLAGCX_DEVICE_INLINE_DECORATOR uint32_t getLmask() const {
    return _impl.lmask;
  }
#else
  uint32_t lmask;

  FLAGCX_DEVICE_INLINE_DECORATOR constexpr flagcxCoopLanes(
      uint32_t lmask = 0xffffffffu)
      : lmask(lmask) {}

  FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
    return flagcxPopc(lmask & flagcxLanemaskLt());
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return flagcxPopc(lmask); }
  FLAGCX_DEVICE_INLINE_DECORATOR void sync() { flagcxSyncwarp(lmask); }
  FLAGCX_DEVICE_INLINE_DECORATOR uint32_t getLmask() const { return lmask; }
#endif
};
#endif // FLAGCX_SIMT_WIDTH

// ---- 6g. flagcxCoopAny — type-erased cooperative group ----
struct flagcxCoopAny {
#ifdef FLAGCX_DEVICE_API_NCCL
  ncclCoopAny _impl;

  flagcxCoopAny() = default;
  flagcxCoopAny(flagcxCoopAny const &) = default;

  FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopAny(flagcxCoopBlock b)
      : _impl(b._impl) {}
  template <int N>
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopAny(flagcxCoopTile<N> t)
      : _impl(t._impl) {}
#ifdef FLAGCX_SIMT_WIDTH
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopAny(flagcxCoopTileSpan s)
      : _impl(s._impl) {}
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopAny(flagcxCoopLanes l)
      : _impl(l._impl) {}
#endif

  FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
    return _impl.thread_rank();
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int size() const { return _impl.size(); }
  FLAGCX_DEVICE_INLINE_DECORATOR void sync() { _impl.sync(); }

#elif defined(FLAGCX_SIMT_WIDTH)
  // Tier 2: own vtable-based type erasure
  struct Storage {
    alignas(alignof(void *)) char space[16];
  };
  struct VTable {
    int (*threadRank)(void const *);
    int (*size)(void const *);
    void (*sync)(void *);
  };

  template <typename Impl>
  FLAGCX_DEVICE_INLINE_DECORATOR static int threadRank_fn(void const *o) {
    return static_cast<Impl const *>(o)->threadRank();
  }
  template <typename Impl>
  FLAGCX_DEVICE_INLINE_DECORATOR static int size_fn(void const *o) {
    return static_cast<Impl const *>(o)->size();
  }
  template <typename Impl>
  FLAGCX_DEVICE_INLINE_DECORATOR static void sync_fn(void *o) {
    static_cast<Impl *>(o)->sync();
  }

  template <typename Impl>
  FLAGCX_DEVICE_INLINE_DECORATOR static VTable const *get_vtable() {
    static_assert(sizeof(Impl) <= sizeof(Storage), "Coop type too large");
    static_assert(alignof(Impl) <= alignof(Storage),
                  "Coop type alignment too large");
    static constexpr VTable v = {&threadRank_fn<Impl>, &size_fn<Impl>,
                                 &sync_fn<Impl>};
    return &v;
  }

  Storage storage;
  VTable const *vtable;

  FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopAny()
      : storage{}, vtable(get_vtable<flagcxCoopThread>()) {}
  flagcxCoopAny(flagcxCoopAny const &) = default;

  template <typename Impl>
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopAny(Impl impl) {
    char const *src = reinterpret_cast<char const *>(&impl);
    for (unsigned i = 0; i < sizeof(Impl); ++i)
      this->storage.space[i] = src[i];
    this->vtable = get_vtable<Impl>();
  }

  FLAGCX_DEVICE_INLINE_DECORATOR int threadRank() const {
    return vtable->threadRank(&storage);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR int size() const {
    return vtable->size(&storage);
  }
  FLAGCX_DEVICE_INLINE_DECORATOR void sync() { vtable->sync(&storage); }

#else
  // Non-SIMT fallback: simple capture
  int _threadRank;
  int _size;

  flagcxCoopAny() : _threadRank(0), _size(1) {}
  flagcxCoopAny(flagcxCoopBlock b)
      : _threadRank(b.threadRank()), _size(b.size()) {}
  template <int N>
  flagcxCoopAny(flagcxCoopTile<N>) : _threadRank(0), _size(N) {}

  int threadRank() const { return _threadRank; }
  int size() const { return _size; }
  void sync() {}
#endif
};

// ---- 6h. Free functions ----

// flagcxCoopGetLaneMask: get the active lane bitmask for a cooperative group
#ifdef FLAGCX_SIMT_WIDTH
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
#endif // FLAGCX_SIMT_WIDTH

// flagcxCoopIsThread: compile-time check if group is a single thread
template <int N>
FLAGCX_DEVICE_INLINE_DECORATOR constexpr bool
flagcxCoopIsThread(flagcxCoopTile<N>) {
  return N == 1;
}
FLAGCX_DEVICE_INLINE_DECORATOR constexpr bool
flagcxCoopIsThread(flagcxCoopBlock) {
  return false;
}
#ifdef FLAGCX_SIMT_WIDTH
FLAGCX_DEVICE_INLINE_DECORATOR constexpr bool
flagcxCoopIsThread(flagcxCoopLanes) {
  return false;
}
FLAGCX_DEVICE_INLINE_DECORATOR constexpr bool
flagcxCoopIsThread(flagcxCoopTileSpan) {
  return false;
}
#endif // FLAGCX_SIMT_WIDTH

// flagcxCoopWithinTile: compile-time check if group fits within a single tile
template <int N>
FLAGCX_DEVICE_INLINE_DECORATOR constexpr bool
flagcxCoopWithinTile(flagcxCoopTile<N>) {
  return true;
}
FLAGCX_DEVICE_INLINE_DECORATOR constexpr bool
flagcxCoopWithinTile(flagcxCoopBlock) {
  return false;
}
#ifdef FLAGCX_SIMT_WIDTH
FLAGCX_DEVICE_INLINE_DECORATOR constexpr bool
flagcxCoopWithinTile(flagcxCoopLanes) {
  return true;
}
FLAGCX_DEVICE_INLINE_DECORATOR constexpr bool
flagcxCoopWithinTile(flagcxCoopTileSpan) {
  return false;
}
#endif // FLAGCX_SIMT_WIDTH

// flagcxCoopCoalesced: get a cooperative group of active/safe threads
#ifdef FLAGCX_SIMT_WIDTH
FLAGCX_DEVICE_INLINE_DECORATOR flagcxCoopLanes flagcxCoopCoalesced() {
  return flagcxCoopLanes{flagcxActivemask()};
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
#endif // FLAGCX_SIMT_WIDTH

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
                            flagcxTeam_t team, uint32_t index,
                            bool multimem = false,
                            flagcxMulticastHandle mcHandle = {})
      : _impl(ncclCoopCta(), devComm._base, ncclTeamLsa(devComm._base),
              devComm._base.lsaBarrier, index, multimem, mcHandle) {}

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
  // Fallback: epoch-based barrier using IPC-mapped peer signals + atomics.
  // NCCL-style: atomicAdd fan-out to peers + wait on own signal.
  uint64_t **_peerSignals;
  int _nRanks, _myRank;
  uint32_t _ctaIndex;
  uint64_t _epoch;

  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxIntraBarrierSession(Coop coop, const flagcxDevComm &devComm,
                            flagcxTeam_t team, uint32_t index,
                            bool multimem = false,
                            flagcxMulticastHandle mcHandle = {})
      : _peerSignals(devComm._barrierPeers), _nRanks(team.nRanks),
        _myRank(team.rank), _ctaIndex(index), _epoch(devComm._barrierEpoch) {}

  // Lightweight arrive: signal all intra peers (thread-0 only, no coop.sync).
  // Used by flagcxBarrierSession to compose with inter barrier.
  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(Coop coop,
         flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    _epoch += _nRanks - 1;
    if (FLAGCX_THREAD_IDX_X == 0) {
      for (int i = 0; i < _nRanks - 1; i++) {
        int peer = 1 + _myRank + i;
        if (peer >= _nRanks)
          peer -= _nRanks;
        flagcxDeviceAtomicFetchAdd(&_peerSignals[peer][_ctaIndex], (uint64_t)1,
                                   flagcxDeviceMemoryOrderRelease);
      }
    }
  }

  // Lightweight wait: spin on own signal (thread-0 only, no coop.sync).
  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    if (FLAGCX_THREAD_IDX_X == 0) {
      int iter = 0;
      while (flagcxDeviceAtomicLoad(&_peerSignals[_myRank][_ctaIndex],
                                    flagcxDeviceMemoryOrderAcquire) < _epoch) {
        spinBackoff(iter++);
      }
    }
  }

  // Standalone sync: includes coop.sync() bookends.
  // Used directly by flagcxIntraAllReduceKernel (intra-only demo).
  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    // Advance epoch: each peer will atomicAdd(1) to our signal
    _epoch += _nRanks - 1;
    coop.sync();
    if (FLAGCX_THREAD_IDX_X == 0) {
      // Fan-out: signal all peers (release — our writes visible before signal)
      for (int i = 0; i < _nRanks - 1; i++) {
        int peer = 1 + _myRank + i;
        if (peer >= _nRanks)
          peer -= _nRanks;
        flagcxDeviceAtomicFetchAdd(&_peerSignals[peer][_ctaIndex], (uint64_t)1,
                                   flagcxDeviceMemoryOrderRelease);
      }
      // Wait: spin on own signal reaching epoch (acquire — peer writes visible)
      int iter = 0;
      while (flagcxDeviceAtomicLoad(&_peerSignals[_myRank][_ctaIndex],
                                    flagcxDeviceMemoryOrderAcquire) < _epoch) {
        spinBackoff(iter++);
      }
    }
    coop.sync();
  }
#endif
};

// ============================================================
// Section 8: Pointer Access Functions (Inline Wrappers)
//
// 7 functions total (see plan Decision 7.8 / 7.9 / 7.19):
//   flagcxGetPeerPointer(mem, off, team, peer)      — canonical unicast
//   flagcxGetLocalPointer(mem, off)                  — convenience (own buffer)
//   flagcxGetMulticastPointer(mem, off, devComm)     — intra-node multicast
//   (LSA convenience) flagcxGetPeerPointer(mem, off, peer)             —
//   world-rank unicast (no team) flagcxGetIntraPointer(mem, off, peer) —
//   intra-node rank pointer flagcxGetMulticastPointer(mem, off, mmHandle)    —
//   explicit multicast handle flagcxFindMem(coop, devComm, ptr) — reverse
//   lookup
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

// ---- Additional pointer functions (Step 2c) ----

// Peer pointer without team parameter.
// Tier 1: delegates to NCCL (peer interpreted by NCCL runtime).
// Tier 2: peer is an intra-node rank index into peerPtrs[] (same as
// flagcxGetIntraPointer). Without a team, world→local mapping is not possible.
FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetPeerPointer(const flagcxDevMem &mem, size_t offset, int peer) {
#ifdef FLAGCX_DEVICE_API_NCCL
  if (mem._hasWindow)
    return ncclGetPeerPointer(mem._base, offset, peer);
#endif
  if (mem.peerPtrs != nullptr)
    return (char *)mem.peerPtrs[peer] + offset;
  return nullptr;
}

// Intra-node rank pointer (NCCL "Lsa" → FlagCX "Intra").
// peer is a local (intra-node) rank index.
FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetIntraPointer(const flagcxDevMem &mem, size_t offset, int peer) {
#ifdef FLAGCX_DEVICE_API_NCCL
  if (mem._hasWindow)
    return ncclGetLsaPointer(mem._base, offset, peer);
#endif
  if (mem.peerPtrs != nullptr)
    return (char *)mem.peerPtrs[peer] + offset;
  return nullptr;
}

// Multicast pointer with explicit MulticastHandle.
FLAGCX_DEVICE_INLINE_DECORATOR void *
flagcxGetMulticastPointer(const flagcxDevMem &mem, size_t offset,
                          flagcxMulticastHandle_t mmHandle) {
#ifdef FLAGCX_DEVICE_API_NCCL
  if (mem._hasWindow)
    return ncclGetMultimemPointer(mem._base, offset, mmHandle._base);
#endif
  return nullptr; // Multicast not available without NCCL window
}

// Reverse lookup: raw pointer → flagcxDevMem.
// Tier 1: cooperative search through NCCL window table.
// Tier 2: not supported (returns empty flagcxDevMem).
template <typename Coop>
FLAGCX_DEVICE_INLINE_DECORATOR flagcxDevMem
flagcxFindMem(Coop coop, const flagcxDevComm &devComm, void const *ptr) {
  flagcxDevMem result;
#ifdef FLAGCX_DEVICE_API_NCCL
  if (devComm._hasBase) {
    ncclWindow_t w = ncclFindWindow(toNccl(coop), devComm._base, ptr);
    result._base = w;
    result._hasWindow = true;
    return result;
  }
#endif
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
// Mirrors NCCL's ncclSymPtr<T>.
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
  FLAGCX_DEVICE_INLINE_DECORATOR T *peerPtr(flagcxTeam_t team, int peer) const {
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
  return a.mem.rawPtr == b.mem.rawPtr && a.offset == b.offset;
}
template <typename T>
FLAGCX_HOST_DEVICE_INLINE bool operator!=(flagcxSymPtr<T> a,
                                          flagcxSymPtr<T> b) {
  return a.mem.rawPtr != b.mem.rawPtr || a.offset != b.offset;
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

// Shared memory descriptor for NIC descriptor optimization (Tier 1 / GIN only).
// On Tier 2, the struct is empty; GIN methods accepting this type are stubs.
struct flagcxDescriptorSmem {
#ifdef FLAGCX_DEVICE_API_NCCL
  ncclGinDescriptorSmem *_impl;
#endif
};

struct flagcxDevNet_DescriptorSmem {
  flagcxDescriptorSmem smem;
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
FLAGCX_DEVICE_INLINE_DECORATOR ncclGin_DescriptorSmem
toNccl(flagcxDevNet_DescriptorSmem a) {
  return {a.smem._impl};
}
FLAGCX_DEVICE_INLINE_DECORATOR ncclCoopCta toNccl(flagcxCoopBlock b) {
  return b._impl;
}
template <int N>
FLAGCX_DEVICE_INLINE_DECORATOR ncclCoopTile<N> toNccl(flagcxCoopTile<N> t) {
  return t._impl;
}
FLAGCX_DEVICE_INLINE_DECORATOR ncclCoopWarpSpan toNccl(flagcxCoopTileSpan s) {
  return s._impl;
}
FLAGCX_DEVICE_INLINE_DECORATOR ncclCoopLanes toNccl(flagcxCoopLanes l) {
  return l._impl;
}
FLAGCX_DEVICE_INLINE_DECORATOR ncclCoopAny toNccl(flagcxCoopAny a) {
  return a._impl;
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
  FLAGCX_DEVICE_INLINE_DECORATOR flagcxResult_t get(size_t srcOffset,
                                                    size_t dstOffset,
                                                    size_t count,
                                                    flagcxDataType_t datatype,
                                                    int peer) const {
    uint64_t fstValue =
        ((uint64_t)srcOffset << flagcxDeviceTriggerOffSrcOffset) |
        ((uint64_t)dstOffset << flagcxDeviceTriggerOffDstOffset);
    return flagcxFifoEnqueue(_devComm.getFifoBuffer(), fstValue, count, peer,
                             datatype, flagcxDevicePrimGet);
  }

#ifdef FLAGCX_DEVICE_API_NCCL
  // ---- GIN one-sided operations (Tier 1 only) ----

  template <typename RemoteAction = flagcxDevNet_None,
            typename LocalAction = flagcxDevNet_None,
            typename Coop = flagcxCoopBlock,
            typename DescriptorSmem = flagcxDevNet_None>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  put(flagcxTeam_t team, int peer, const flagcxDevMem &dstMem, size_t dstOffset,
      const flagcxDevMem &srcMem, size_t srcOffset, size_t bytes,
      RemoteAction remoteAction = flagcxDevNet_None{},
      LocalAction localAction = flagcxDevNet_None{},
      Coop coop = flagcxCoopBlock{},
      DescriptorSmem descriptor = flagcxDevNet_None{},
      flagcxDeviceScope_t alreadyReleased = flagcxDeviceScopeThread,
      flagcxDeviceScope_t expected_scope = flagcxDeviceScopeDevice) const {
    _gin.put(team._base, peer, dstMem._base, dstOffset, srcMem._base, srcOffset,
             bytes, toNccl(remoteAction), toNccl(localAction), toNccl(coop),
             toNccl(descriptor), flagcxDeviceScopeMap[alreadyReleased],
             flagcxDeviceScopeMap[expected_scope]);
  }

  // SymPtr-based put overload — convenience wrapper.
  template <typename T, typename RemoteAction = flagcxDevNet_None,
            typename LocalAction = flagcxDevNet_None,
            typename Coop = flagcxCoopBlock,
            typename DescriptorSmem = flagcxDevNet_None>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  put(flagcxTeam_t team, int peer, flagcxSymPtr<T> dst, flagcxSymPtr<T> src,
      size_t nElts, RemoteAction remoteAction = flagcxDevNet_None{},
      LocalAction localAction = flagcxDevNet_None{},
      Coop coop = flagcxCoopBlock{},
      DescriptorSmem descriptor = flagcxDevNet_None{},
      flagcxDeviceScope_t alreadyReleased = flagcxDeviceScopeThread,
      flagcxDeviceScope_t expected_scope = flagcxDeviceScopeDevice) const {
    this->put(team, peer, dst.mem, dst.offset, src.mem, src.offset,
              nElts * sizeof(T), remoteAction, localAction, coop, descriptor,
              alreadyReleased, expected_scope);
  }

  template <typename T, typename RemoteAction = flagcxDevNet_None,
            typename Coop = flagcxCoopBlock,
            typename DescriptorSmem = flagcxDevNet_None>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  putValue(flagcxTeam_t team, int peer, const flagcxDevMem &dstMem,
           size_t dstOffset, T value,
           RemoteAction remoteAction = flagcxDevNet_None{},
           Coop coop = flagcxCoopBlock{},
           DescriptorSmem descriptor = flagcxDevNet_None{},
           flagcxDeviceScope_t alreadyReleased = flagcxDeviceScopeThread,
           flagcxDeviceScope_t expected_scope = flagcxDeviceScopeDevice) const {
    _gin.putValue(team._base, peer, dstMem._base, dstOffset, value,
                  toNccl(remoteAction), toNccl(coop), toNccl(descriptor),
                  flagcxDeviceScopeMap[alreadyReleased],
                  flagcxDeviceScopeMap[expected_scope]);
  }

  // SymPtr-based putValue overload.
  template <typename T, typename RemoteAction = flagcxDevNet_None,
            typename Coop = flagcxCoopBlock,
            typename DescriptorSmem = flagcxDevNet_None>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  putValue(flagcxTeam_t team, int peer, flagcxSymPtr<T> dst, T value,
           RemoteAction remoteAction = flagcxDevNet_None{},
           Coop coop = flagcxCoopBlock{},
           DescriptorSmem descriptor = flagcxDevNet_None{},
           flagcxDeviceScope_t alreadyReleased = flagcxDeviceScopeThread,
           flagcxDeviceScope_t expected_scope = flagcxDeviceScopeDevice) const {
    this->putValue(team, peer, dst.mem, dst.offset, value, remoteAction, coop,
                   descriptor, alreadyReleased, expected_scope);
  }

  template <typename RemoteAction, typename Coop = flagcxCoopBlock,
            typename DescriptorSmem = flagcxDevNet_None>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  signal(flagcxTeam_t team, int peer, RemoteAction remoteAction,
         Coop coop = flagcxCoopBlock{},
         DescriptorSmem descriptor = flagcxDevNet_None{},
         flagcxDeviceScope_t alreadyReleased = flagcxDeviceScopeThread,
         flagcxDeviceScope_t expected_scope = flagcxDeviceScopeDevice) const {
    _gin.signal(team._base, peer, toNccl(remoteAction), toNccl(coop),
                toNccl(descriptor), flagcxDeviceScopeMap[alreadyReleased],
                flagcxDeviceScopeMap[expected_scope]);
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
            typename Coop = flagcxCoopBlock,
            typename DescriptorSmem = flagcxDevNet_None>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  put(flagcxTeam_t, int, const flagcxDevMem &, size_t, const flagcxDevMem &,
      size_t, size_t, RemoteAction = flagcxDevNet_None{},
      LocalAction = flagcxDevNet_None{}, Coop = flagcxCoopBlock{},
      DescriptorSmem = flagcxDevNet_None{},
      flagcxDeviceScope_t = flagcxDeviceScopeThread,
      flagcxDeviceScope_t = flagcxDeviceScopeDevice) const {}

  template <typename T, typename RemoteAction = flagcxDevNet_None,
            typename LocalAction = flagcxDevNet_None,
            typename Coop = flagcxCoopBlock,
            typename DescriptorSmem = flagcxDevNet_None>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  put(flagcxTeam_t, int, flagcxSymPtr<T>, flagcxSymPtr<T>, size_t,
      RemoteAction = flagcxDevNet_None{}, LocalAction = flagcxDevNet_None{},
      Coop = flagcxCoopBlock{}, DescriptorSmem = flagcxDevNet_None{},
      flagcxDeviceScope_t = flagcxDeviceScopeThread,
      flagcxDeviceScope_t = flagcxDeviceScopeDevice) const {}

  template <typename T, typename RemoteAction = flagcxDevNet_None,
            typename Coop = flagcxCoopBlock,
            typename DescriptorSmem = flagcxDevNet_None>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  putValue(flagcxTeam_t, int, const flagcxDevMem &, size_t, T,
           RemoteAction = flagcxDevNet_None{}, Coop = flagcxCoopBlock{},
           DescriptorSmem = flagcxDevNet_None{},
           flagcxDeviceScope_t = flagcxDeviceScopeThread,
           flagcxDeviceScope_t = flagcxDeviceScopeDevice) const {}

  template <typename T, typename RemoteAction = flagcxDevNet_None,
            typename Coop = flagcxCoopBlock,
            typename DescriptorSmem = flagcxDevNet_None>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  putValue(flagcxTeam_t, int, flagcxSymPtr<T>, T,
           RemoteAction = flagcxDevNet_None{}, Coop = flagcxCoopBlock{},
           DescriptorSmem = flagcxDevNet_None{},
           flagcxDeviceScope_t = flagcxDeviceScopeThread,
           flagcxDeviceScope_t = flagcxDeviceScopeDevice) const {}

  template <typename RemoteAction, typename Coop = flagcxCoopBlock,
            typename DescriptorSmem = flagcxDevNet_None>
  FLAGCX_DEVICE_INLINE_DECORATOR void
  signal(flagcxTeam_t, int, RemoteAction, Coop = flagcxCoopBlock{},
         DescriptorSmem = flagcxDevNet_None{},
         flagcxDeviceScope_t = flagcxDeviceScopeThread,
         flagcxDeviceScope_t = flagcxDeviceScopeDevice) const {}

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
// Tier 2: Inter-node barrier via FIFO Signal + netAdaptor isend/irecv.
// Sends signals to inter-node peers, waits on host-mapped interSignalFlags.
// Only the inter leader (localRank 0) actually sends/waits; non-leaders are
// no-ops. All ranks know nInterPeers for two-phase logic in BarrierSession.
template <typename Coop>
struct flagcxInterBarrierSession {
  uint64_t *_interSignals; // host-mapped inter signal array [CTA_COUNT]
  void *_fifoBuffer;       // for FIFO Signal entries
  int _nInterPeers;
  bool _isLeader;
  uint32_t _ctaIndex;
  uint64_t _epoch;

  // Active constructor (world barrier with inter-node peers)
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxInterBarrierSession(Coop coop, const flagcxDevComm &devComm,
                            uint32_t index)
      : _interSignals(devComm._interSignalFlags),
        _fifoBuffer(devComm.getFifoBuffer()),
        _nInterPeers(devComm._nInterPeers), _isLeader(devComm._isInterLeader),
        _ctaIndex(index), _epoch(devComm._interBarrierEpoch) {}

  // Default constructor (intra-only, all operations are no-ops)
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxInterBarrierSession()
      : _interSignals(nullptr), _fifoBuffer(nullptr), _nInterPeers(0),
        _isLeader(false), _ctaIndex(0), _epoch(0) {}

  // Arrive: write one FIFO Signal entry (proxy fans out to all inter peers)
  // Only the leader sends; non-leaders skip.
  FLAGCX_DEVICE_INLINE_DECORATOR void
  arrive(Coop coop,
         flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    _epoch += _nInterPeers;
    if (FLAGCX_THREAD_IDX_X == 0 && _isLeader) {
      flagcxFifoEnqueue(_fifoBuffer, (uint64_t)_ctaIndex, 0, 0, 0,
                        flagcxDevicePrimBarrierSignal);
    }
  }

  // Wait: spin on host-mapped inter signal array
  // Only the leader waits; non-leaders skip.
  FLAGCX_DEVICE_INLINE_DECORATOR void
  wait(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    if (FLAGCX_THREAD_IDX_X == 0 && _isLeader) {
      int iter = 0;
      while (flagcxDeviceAtomicLoad(&_interSignals[_ctaIndex],
                                    flagcxDeviceMemoryOrderAcquire) < _epoch) {
        spinBackoff(iter++);
      }
    }
  }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel) {
    arrive(coop, order);
    wait(coop, order);
  }
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
                       uint32_t index, bool multimem = false)
      : _impl(ncclCoopCta(), ncclTeamTagWorld(), net._gin, index, multimem) {}

  // Intra-only barrier
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxBarrierSession(Coop coop, flagcxTeamTagIntra,
                       const flagcxDevComm &devComm, uint32_t index,
                       bool multimem = false)
      : _impl(ncclCoopCta(), ncclTeamTagLsa(), devComm._base, index, multimem) {
  }

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
// Tier 2: Composes intra (IPC atomicAdd) + inter (FIFO Signal relay).
//         Three-phase pattern for multi-node:
//           Phase 1: intra sync (all local ranks ensure data visible)
//           Phase 2: leader inter signal+wait (non-leaders skip)
//           Phase 3: intra sync (broadcasts inter completion)
//         Single-node: just one intra sync (no phase 2/3).
template <typename Coop>
struct flagcxBarrierSession {
  flagcxIntraBarrierSession<Coop> _intra;
  flagcxInterBarrierSession<Coop> _inter;
  int _nInterPeers;

  // World barrier: intra (IPC) + inter (FIFO Signal → isend)
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxBarrierSession(Coop coop, flagcxTeamTagWorld, const flagcxDevNet &net,
                       uint32_t index, bool multimem = false)
      : _intra(coop, net._devComm, flagcxTeamIntra(net._devComm), index),
        _inter(coop, net._devComm, index),
        _nInterPeers(net._devComm._nInterPeers) {}

  // Intra-only barrier: inter is default constructed (no-op)
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxBarrierSession(Coop coop, flagcxTeamTagIntra,
                       const flagcxDevComm &devComm, uint32_t index,
                       bool multimem = false)
      : _intra(coop, devComm, flagcxTeamIntra(devComm), index), _inter(),
        _nInterPeers(0) {}

  // Accessors for sub-barrier sessions (Tier 2 only)
  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxIntraBarrierSession<Coop> &intraBarrier() { return _intra; }

  FLAGCX_DEVICE_INLINE_DECORATOR
  flagcxInterBarrierSession<Coop> &interBarrier() { return _inter; }

  FLAGCX_DEVICE_INLINE_DECORATOR void
  sync(Coop coop,
       flagcxDeviceMemoryOrder_t order = flagcxDeviceMemoryOrderAcqRel,
       flagcxGinFenceLevel fence = flagcxGinFenceLevel::Relaxed) {
    if (_nInterPeers > 0) {
      // Multi-node: two-phase intra barrier with inter in between
      // Phase 1: intra sync — all local ranks ensure data is visible
      coop.sync();
      _intra.arrive(coop, order);
      _intra.wait(coop, order);
      coop.sync();

      // Phase 2: leader does inter signal+wait (non-leaders skip)
      _inter.arrive(coop, order);
      _inter.wait(coop, order);

      // Phase 3: intra sync — broadcast inter completion to all local ranks
      coop.sync();
      _intra.arrive(coop, order);
      _intra.wait(coop, order);
      coop.sync();
    } else {
      // Single-node: just one intra sync
      coop.sync();
      _intra.arrive(coop, order);
      _intra.wait(coop, order);
      coop.sync();
    }
  }
};
#endif

#endif // FLAGCX_DEVICE_COMPILE (Sections 9b-12)

#endif // FLAGCX_DEVICE_API_H_
