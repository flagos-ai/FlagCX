/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Symmetric memory coordination for the default (non-vendor) path.
 * Implements VMM-based flat VA mapping with IPC fallback.
 ************************************************************************/

#include "sym_heap.h"
#include "adaptor.h"
#include "alloc.h"
#include "bootstrap.h"
#include "check.h"
#include "comm.h"
#include "global_comm.h"
#include "ipcsocket.h"
#include "mem_alloc_registry.h"
#include "onesided.h"
#include "param.h"
#include "transport.h"
#include "utils.h"
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <new>
#include <unistd.h>

struct flagcxSymLsaArena {
  void *base = nullptr;
  size_t rankStride = 0;
  size_t totalSize = 0;
  size_t alignment = 0;
  bool invalid = false;
  void *mcBase = nullptr;
  void *mcHandle = nullptr;
  size_t mcGranularity = 0;
  bool mcMappingOwned = false;
  bool mcVaOwned = false;
  bool mcReady = false;
  bool mcUnavailable = false;
  std::map<size_t, size_t> freeRanges; // offset -> length within one rank slot
};

namespace {

constexpr int kSymStatusGatherTag = 0x5950;
constexpr int kSymStatusBroadcastTag = 0x5951;
constexpr int kSymPendingGatherTag = 0x5952;
constexpr int kSymPendingBroadcastTag = 0x5953;
constexpr int kSymNetworkRouteGatherTag = 0x5954;
constexpr int kSymNetworkRouteBroadcastTag = 0x5955;
constexpr int kSymFallbackRouteGatherTag = 0x5956;
constexpr int kSymFallbackRouteBroadcastTag = 0x5957;

enum flagcxSymScalarReduction {
  flagcxSymFirstError,
  flagcxSymLogicalOr,
};

// Symmetric-memory rollback may be entered because a rank-local allocation
// failed. Keep the convergence primitive allocation-free so reporting that
// failure cannot itself strand peers in a collective. Rank 0 gathers one
// scalar from each peer and broadcasts the deterministic result.
static flagcxResult_t flagcxSymGlobalConvergeScalar(
    flagcxHeteroComm_t comm, int localValue, int gatherTag, int broadcastTag,
    flagcxSymScalarReduction reduction, int *commonValue) {
  if (comm == nullptr || comm->bootstrap == nullptr || commonValue == nullptr ||
      comm->rank < 0 || comm->rank >= comm->nRanks || comm->nRanks <= 0)
    return flagcxInvalidArgument;

  constexpr int root = 0;
  int common = localValue;
  if (comm->rank == root) {
    for (int peer = 1; peer < comm->nRanks; peer++) {
      int peerValue = 0;
      FLAGCXCHECK(bootstrapRecv(comm->bootstrap, peer, gatherTag, &peerValue,
                                sizeof(peerValue)));
      if (reduction == flagcxSymLogicalOr) {
        common = common != 0 || peerValue != 0;
      } else if (common == flagcxSuccess && peerValue != flagcxSuccess) {
        // Peers are visited in rank order, preserving the existing
        // lowest-failing-rank error selection.
        common = peerValue;
      }
    }
    for (int peer = 1; peer < comm->nRanks; peer++) {
      FLAGCXCHECK(bootstrapSend(comm->bootstrap, peer, broadcastTag, &common,
                                sizeof(common)));
    }
  } else {
    FLAGCXCHECK(bootstrapSend(comm->bootstrap, root, gatherTag, &localValue,
                              sizeof(localValue)));
    FLAGCXCHECK(bootstrapRecv(comm->bootstrap, root, broadcastTag, &common,
                              sizeof(common)));
  }

  *commonValue = common;
  return flagcxSuccess;
}

} // namespace

static flagcxResult_t flagcxSymLocalConverge(flagcxHeteroComm_t comm,
                                             flagcxResult_t localStatus,
                                             int tag,
                                             flagcxResult_t *commonStatus) {
  if (comm == nullptr || comm->bootstrap == nullptr ||
      commonStatus == nullptr || comm->localRanks <= 0 || comm->localRank < 0 ||
      comm->localRank >= comm->localRanks || comm->localRankToRank == nullptr)
    return flagcxInvalidArgument;
  int local = static_cast<int>(localStatus);
  int common = local;
  constexpr int localRoot = 0;
  int root = comm->localRankToRank[localRoot];
  if (comm->localRank == localRoot) {
    for (int i = 1; i < comm->localRanks; i++) {
      int peerStatus = static_cast<int>(flagcxSuccess);
      int peer = comm->localRankToRank[i];
      FLAGCXCHECK(bootstrapRecv(comm->bootstrap, peer, tag, &peerStatus,
                                sizeof(peerStatus)));
      if (common == flagcxSuccess && peerStatus != flagcxSuccess)
        common = peerStatus;
    }
    for (int i = 1; i < comm->localRanks; i++) {
      int peer = comm->localRankToRank[i];
      FLAGCXCHECK(
          bootstrapSend(comm->bootstrap, peer, tag, &common, sizeof(common)));
    }
  } else {
    FLAGCXCHECK(
        bootstrapSend(comm->bootstrap, root, tag, &local, sizeof(local)));
    FLAGCXCHECK(
        bootstrapRecv(comm->bootstrap, root, tag, &common, sizeof(common)));
  }
  // Local rank zero receives in local-rank order, so different simultaneous
  // failures still converge to one deterministic error on every peer.
  *commonStatus = static_cast<flagcxResult_t>(common);
  return flagcxSuccess;
}

static flagcxResult_t flagcxSymGlobalConverge(flagcxHeteroComm_t comm,
                                              flagcxResult_t localStatus,
                                              flagcxResult_t *commonStatus) {
  if (commonStatus == nullptr)
    return flagcxInvalidArgument;
  int common = static_cast<int>(flagcxSuccess);
  FLAGCXCHECK(flagcxSymGlobalConvergeScalar(
      comm, static_cast<int>(localStatus), kSymStatusGatherTag,
      kSymStatusBroadcastTag, flagcxSymFirstError, &common));
  *commonStatus = static_cast<flagcxResult_t>(common);
  return flagcxSuccess;
}

flagcxResult_t flagcxSymConvergeStatus(flagcxHeteroComm_t comm,
                                       flagcxResult_t localStatus,
                                       flagcxResult_t *commonStatus) {
  return flagcxSymGlobalConverge(comm, localStatus, commonStatus);
}

static flagcxResult_t flagcxSymLsaFindOffset(const flagcxSymLsaArena *arena,
                                             size_t span, size_t alignment,
                                             size_t *offset) {
  if (arena == nullptr || offset == nullptr || span == 0 || alignment == 0 ||
      span > arena->rankStride)
    return flagcxInvalidArgument;
  for (std::map<size_t, size_t>::const_iterator it = arena->freeRanges.begin();
       it != arena->freeRanges.end(); ++it) {
    const size_t start = it->first;
    const size_t length = it->second;
    if (start > arena->rankStride || length > arena->rankStride - start ||
        start > std::numeric_limits<size_t>::max() - (alignment - 1))
      return flagcxInternalError;
    const size_t aligned = ((start + alignment - 1) / alignment) * alignment;
    if (aligned >= start && aligned - start <= length &&
        span <= length - (aligned - start)) {
      *offset = aligned;
      return flagcxSuccess;
    }
  }
  return flagcxNotSupported;
}

static flagcxResult_t flagcxSymLsaMarkOffset(flagcxSymLsaArena *arena,
                                             size_t offset, size_t span) {
  if (arena == nullptr || span == 0 || offset > arena->rankStride ||
      span > arena->rankStride - offset)
    return flagcxInvalidArgument;
  try {
    std::map<size_t, size_t> next(arena->freeRanges);
    std::map<size_t, size_t>::iterator it = next.upper_bound(offset);
    if (it == next.begin())
      return flagcxInvalidUsage;
    --it;
    const size_t start = it->first;
    const size_t end = start + it->second;
    if (offset < start || offset > end || span > end - offset)
      return flagcxInvalidUsage;
    next.erase(it);
    if (start < offset)
      next.emplace(start, offset - start);
    if (offset + span < end)
      next.emplace(offset + span, end - offset - span);
    arena->freeRanges.swap(next);
  } catch (const std::bad_alloc &) {
    return flagcxSystemError;
  }
  return flagcxSuccess;
}

static flagcxResult_t flagcxSymLsaReleaseOffset(flagcxSymLsaArena *arena,
                                                size_t offset, size_t span) {
  if (arena == nullptr || span == 0 || offset > arena->rankStride ||
      span > arena->rankStride - offset)
    return flagcxInvalidArgument;
  try {
    std::map<size_t, size_t> next(arena->freeRanges);
    size_t start = offset;
    size_t end = offset + span;
    std::map<size_t, size_t>::iterator after = next.lower_bound(offset);
    if (after != next.begin()) {
      std::map<size_t, size_t>::iterator before = after;
      --before;
      const size_t beforeEnd = before->first + before->second;
      if (beforeEnd > start)
        return flagcxInvalidUsage;
      if (beforeEnd == start) {
        start = before->first;
        next.erase(before);
      }
    }
    after = next.lower_bound(offset);
    if (after != next.end()) {
      if (after->first < end)
        return flagcxInvalidUsage;
      if (after->first == end) {
        end += after->second;
        next.erase(after);
      }
    }
    next.emplace(start, end - start);
    arena->freeRanges.swap(next);
  } catch (const std::bad_alloc &) {
    return flagcxSystemError;
  }
  return flagcxSuccess;
}

static flagcxResult_t flagcxSymLsaReleaseMulticast(flagcxSymLsaArena *arena) {
  if (arena == nullptr)
    return flagcxInvalidArgument;
  if (arena->mcMappingOwned) {
    if (deviceAdaptor == nullptr ||
        deviceAdaptor->symMulticastMappingUnmap == nullptr)
      return flagcxInternalError;
    FLAGCXCHECK(deviceAdaptor->symMulticastMappingUnmap(arena->mcBase,
                                                        arena->rankStride));
    arena->mcMappingOwned = false;
  }
  if (arena->mcVaOwned) {
    if (deviceAdaptor == nullptr ||
        deviceAdaptor->symMulticastVaFree == nullptr)
      return flagcxInternalError;
    FLAGCXCHECK(
        deviceAdaptor->symMulticastVaFree(arena->mcBase, arena->rankStride));
    arena->mcVaOwned = false;
    arena->mcBase = nullptr;
  }
  if (arena->mcHandle != nullptr) {
    if (deviceAdaptor == nullptr || deviceAdaptor->symMulticastFree == nullptr)
      return flagcxInternalError;
    FLAGCXCHECK(deviceAdaptor->symMulticastFree(arena->mcHandle));
    arena->mcHandle = nullptr;
  }
  arena->mcReady = false;
  return flagcxSuccess;
}

static flagcxResult_t flagcxSymLsaEnsureArena(flagcxHeteroComm_t comm,
                                              size_t windowSpan,
                                              size_t windowAlignment) {
  if (comm == nullptr || deviceAdaptor == nullptr ||
      deviceAdaptor->symLsaGetCapacity == nullptr ||
      deviceAdaptor->symLsaVaReserve == nullptr ||
      deviceAdaptor->symLsaVaFree == nullptr)
    return flagcxNotSupported;
  flagcxSymLsaArena *arena = comm->symLsaArena;
  flagcxResult_t common = flagcxSuccess;
  if (arena != nullptr && arena->invalid) {
    flagcxResult_t local = flagcxSymLsaReleaseMulticast(arena);
    if (local == flagcxSuccess && arena->base != nullptr) {
      local = deviceAdaptor->symLsaVaFree(arena->base, arena->totalSize);
      if (local == flagcxSuccess)
        arena->base = nullptr;
    }
    FLAGCXCHECK(flagcxSymLocalConverge(comm, local, 0x5980, &common));
    if (common != flagcxSuccess)
      return common;
    delete arena;
    comm->symLsaArena = nullptr;
    arena = nullptr;
  }
  if (arena != nullptr)
    return windowAlignment <= arena->alignment &&
                   arena->alignment % windowAlignment == 0 &&
                   windowSpan <= arena->rankStride
               ? flagcxSuccess
               : flagcxNotSupported;

  struct Capacity {
    size_t totalBytes;
    size_t alignment;
  } capacity = {0, 0};
  flagcxResult_t local = deviceAdaptor->symLsaGetCapacity(&capacity.totalBytes,
                                                          &capacity.alignment);
  if (local == flagcxSuccess &&
      (capacity.totalBytes == 0 || capacity.alignment == 0 ||
       (capacity.alignment & (capacity.alignment - 1)) != 0))
    local = flagcxInvalidArgument;
  FLAGCXCHECK(flagcxSymLocalConverge(comm, local, 0x5981, &common));
  if (common != flagcxSuccess)
    return common;

  const int root = comm->localRankToRank[0];
  if (comm->localRank == 0) {
    for (int peer = 1; peer < comm->localRanks; peer++) {
      Capacity remote = {0, 0};
      FLAGCXCHECK(bootstrapRecv(comm->bootstrap, comm->localRankToRank[peer],
                                0x5982, &remote, sizeof(remote)));
      if (capacity.totalBytes < remote.totalBytes)
        capacity.totalBytes = remote.totalBytes;
      if (capacity.alignment < remote.alignment)
        capacity.alignment = remote.alignment;
    }
    for (int peer = 1; peer < comm->localRanks; peer++)
      FLAGCXCHECK(bootstrapSend(comm->bootstrap, comm->localRankToRank[peer],
                                0x5983, &capacity, sizeof(capacity)));
  } else {
    FLAGCXCHECK(bootstrapSend(comm->bootstrap, root, 0x5982, &capacity,
                              sizeof(capacity)));
    FLAGCXCHECK(bootstrapRecv(comm->bootstrap, root, 0x5983, &capacity,
                              sizeof(capacity)));
  }

  const size_t alignment = capacity.alignment > windowAlignment
                               ? capacity.alignment
                               : windowAlignment;
  const size_t required =
      capacity.totalBytes > windowSpan ? capacity.totalBytes : windowSpan;
  local = required > std::numeric_limits<size_t>::max() - (alignment - 1)
              ? flagcxInvalidArgument
              : flagcxSuccess;
  const size_t rankStride =
      local == flagcxSuccess
          ? ((required + alignment - 1) / alignment) * alignment
          : 0;
  const size_t minimumRequired =
      windowSpan > alignment ? windowSpan : alignment;
  const size_t minimumStride =
      local == flagcxSuccess
          ? ((minimumRequired + alignment - 1) / alignment) * alignment
          : 0;
  if (local == flagcxSuccess &&
      (rankStride > std::numeric_limits<size_t>::max() /
                        static_cast<size_t>(comm->localRanks)))
    local = flagcxNotSupported;
  if (local == flagcxSuccess) {
    arena = new (std::nothrow) flagcxSymLsaArena();
    if (arena == nullptr)
      local = flagcxSystemError;
    else {
      arena->rankStride = rankStride;
      arena->totalSize = rankStride * static_cast<size_t>(comm->localRanks);
      arena->alignment = alignment;
      try {
        arena->freeRanges.emplace(0, rankStride);
      } catch (const std::bad_alloc &) {
        local = flagcxSystemError;
      }
    }
  }
  FLAGCXCHECK(flagcxSymLocalConverge(comm, local, 0x5984, &common));
  if (common != flagcxSuccess) {
    delete arena;
    return common;
  }
  comm->symLsaArena = arena;
  for (;;) {
    local = deviceAdaptor->symLsaVaReserve(arena->totalSize, arena->alignment,
                                           &arena->base);
    if (local == flagcxSuccess && arena->base == nullptr)
      local = flagcxInternalError;
    // Only an address-space capacity failure may trigger a smaller retry.
    // Converge fatal errors separately so a simultaneous allocation failure
    // cannot hide a real error on another local rank.
    flagcxResult_t fatal = local == flagcxNotSupported ? flagcxSuccess : local;
    flagcxResult_t commonFatal = flagcxSuccess;
    FLAGCXCHECK(flagcxSymLocalConverge(comm, fatal, 0x598e, &commonFatal));
    FLAGCXCHECK(flagcxSymLocalConverge(comm, local, 0x5985, &common));
    if (commonFatal == flagcxSuccess && common == flagcxSuccess)
      return flagcxSuccess;

    arena->invalid = true;
    flagcxResult_t rollback =
        arena->base != nullptr
            ? deviceAdaptor->symLsaVaFree(arena->base, arena->totalSize)
            : flagcxSuccess;
    if (rollback == flagcxSuccess)
      arena->base = nullptr;
    flagcxResult_t commonRollback = flagcxSuccess;
    FLAGCXCHECK(
        flagcxSymLocalConverge(comm, rollback, 0x5986, &commonRollback));
    if (commonRollback != flagcxSuccess)
      return commonRollback;

    if (commonFatal == flagcxSuccess && common == flagcxNotSupported &&
        arena->rankStride > minimumStride) {
      size_t nextStride = (arena->rankStride / 2 / alignment) * alignment;
      if (nextStride < minimumStride)
        nextStride = minimumStride;
      if (comm->localRank == 0)
        INFO(FLAGCX_INIT, "symmem LSA VA reserve retry: stride %zu -> %zu",
             arena->rankStride, nextStride);
      arena->rankStride = nextStride;
      arena->totalSize = nextStride * static_cast<size_t>(comm->localRanks);
      arena->freeRanges.begin()->second = nextStride;
      arena->invalid = false;
      continue;
    }

    delete arena;
    comm->symLsaArena = nullptr;
    return commonFatal != flagcxSuccess ? commonFatal : common;
  }
}

static flagcxResult_t flagcxSymLsaEnsureMulticast(flagcxHeteroComm_t comm) {
  flagcxSymLsaArena *arena = comm->symLsaArena;
  if (arena == nullptr || arena->invalid)
    return flagcxInvalidUsage;
  if (arena->mcReady)
    return flagcxSuccess;
  if (arena->mcUnavailable)
    return flagcxNotSupported;

  const bool callbacks = deviceAdaptor->symMulticastSupported != nullptr &&
                         deviceAdaptor->symMulticastCreate != nullptr &&
                         deviceAdaptor->symMulticastImport != nullptr &&
                         deviceAdaptor->symMulticastFree != nullptr &&
                         deviceAdaptor->symMulticastMappingUnmap != nullptr &&
                         deviceAdaptor->symMulticastVaFree != nullptr &&
                         deviceAdaptor->symLsaMulticastGranularity != nullptr &&
                         deviceAdaptor->symLsaMulticastMap != nullptr &&
                         deviceAdaptor->symLsaMulticastBind != nullptr &&
                         deviceAdaptor->symLsaMulticastUnbind != nullptr;
  int supported = 0;
  flagcxResult_t local = callbacks
                             ? deviceAdaptor->symMulticastSupported(&supported)
                             : flagcxNotSupported;
  if (local == flagcxSuccess && !supported)
    local = flagcxNotSupported;
  flagcxResult_t common = flagcxSuccess;
  FLAGCXCHECK(flagcxSymLocalConverge(comm, local, 0x5990, &common));
  if (common != flagcxSuccess) {
    arena->mcUnavailable = true;
    return flagcxNotSupported;
  }

  size_t granularity = 0;
  local = deviceAdaptor->symLsaMulticastGranularity(
      arena->rankStride, comm->localRanks, &granularity);
  if (local == flagcxSuccess &&
      (granularity == 0 || arena->alignment % granularity != 0 ||
       arena->rankStride % granularity != 0))
    local = flagcxNotSupported;
  FLAGCXCHECK(flagcxSymLocalConverge(comm, local, 0x5991, &common));
  if (common != flagcxSuccess) {
    arena->mcUnavailable = true;
    return flagcxNotSupported;
  }
  const int root = comm->localRankToRank[0];
  size_t agreedGranularity = granularity;
  if (comm->localRank == 0) {
    for (int peer = 1; peer < comm->localRanks; peer++) {
      size_t remote = 0;
      FLAGCXCHECK(bootstrapRecv(comm->bootstrap, comm->localRankToRank[peer],
                                0x599d, &remote, sizeof(remote)));
      if (remote != granularity)
        agreedGranularity = 0;
    }
    for (int peer = 1; peer < comm->localRanks; peer++)
      FLAGCXCHECK(bootstrapSend(comm->bootstrap, comm->localRankToRank[peer],
                                0x599e, &agreedGranularity,
                                sizeof(agreedGranularity)));
  } else {
    FLAGCXCHECK(bootstrapSend(comm->bootstrap, root, 0x599d, &granularity,
                              sizeof(granularity)));
    FLAGCXCHECK(bootstrapRecv(comm->bootstrap, root, 0x599e, &agreedGranularity,
                              sizeof(agreedGranularity)));
  }
  if (agreedGranularity == 0) {
    arena->mcUnavailable = true;
    return flagcxNotSupported;
  }
  arena->mcGranularity = granularity;

  // A failed optional setup can continue through the ordinary LSA mapping
  // only after every local rank has released its partial multicast state.
  auto rollback = [&]() -> flagcxResult_t {
    flagcxResult_t release = flagcxSymLsaReleaseMulticast(arena);
    flagcxResult_t agreed = flagcxSuccess;
    FLAGCXCHECK(flagcxSymLocalConverge(comm, release, 0x5992, &agreed));
    if (agreed != flagcxSuccess) {
      arena->invalid = true;
      return agreed;
    }
    arena->mcUnavailable = true;
    arena->mcGranularity = 0;
    return flagcxNotSupported;
  };

  int fd = -1;
  int created = 0;
  if (comm->localRank == 0) {
    int *devices = static_cast<int *>(
        calloc(static_cast<size_t>(comm->localRanks), sizeof(int)));
    if (devices != nullptr) {
      for (int peer = 0; peer < comm->localRanks; peer++)
        devices[peer] = comm->peerInfo[comm->localRankToRank[peer]].cudaDev;
      local = deviceAdaptor->symMulticastCreate(
          arena->rankStride, comm->localRanks, devices, &arena->mcHandle, &fd);
      free(devices);
      created = local == flagcxSuccess && arena->mcHandle != nullptr && fd >= 0;
    }
  }
  for (int peer = 1; peer < comm->localRanks; peer++) {
    if (comm->localRank == 0) {
      FLAGCXCHECK(bootstrapSend(comm->bootstrap, comm->localRankToRank[peer],
                                0x5993, &created, sizeof(created)));
    } else if (comm->localRank == peer) {
      FLAGCXCHECK(bootstrapRecv(comm->bootstrap, comm->localRankToRank[0],
                                0x5993, &created, sizeof(created)));
    }
  }
  if (!created) {
    if (fd >= 0)
      close(fd);
    return rollback();
  }

  flagcxIpcSocket socket = {};
  const uint64_t hash = comm->commHash ^ 0x6c73616d756c7469ULL;
  local = comm->localRanks > 1
              ? flagcxIpcSocketInit(&socket, comm->rank, hash, /*block=*/1)
              : flagcxSuccess;
  FLAGCXCHECK(flagcxSymLocalConverge(comm, local, 0x5994, &common));
  if (common != flagcxSuccess) {
    if (local == flagcxSuccess && comm->localRanks > 1)
      flagcxIpcSocketClose(&socket);
    if (fd >= 0)
      close(fd);
    return rollback();
  }
  if (comm->localRanks > 1) {
    FLAGCXCHECK(bootstrapCollIntraNodeBarrier(
        comm->bootstrap, comm->localRankToRank, comm->localRank,
        comm->localRanks, 0x5995));
    int sent = 1;
    if (comm->localRank == 0) {
      for (int peer = 1; peer < comm->localRanks; peer++) {
        int tag = 0;
        if (flagcxIpcSocketSendMsg(&socket, &tag, sizeof(tag), fd,
                                   comm->localRankToRank[peer],
                                   hash) != flagcxSuccess)
          sent = 0;
      }
      for (int peer = 1; peer < comm->localRanks; peer++)
        FLAGCXCHECK(bootstrapSend(comm->bootstrap, comm->localRankToRank[peer],
                                  0x599c, &sent, sizeof(sent)));
    } else {
      FLAGCXCHECK(bootstrapRecv(comm->bootstrap, comm->localRankToRank[0],
                                0x599c, &sent, sizeof(sent)));
      if (sent) {
        int tag = -1;
        local = flagcxIpcSocketRecvMsg(&socket, &tag, sizeof(tag), &fd);
        if (local == flagcxSuccess && (tag != 0 || fd < 0))
          local = flagcxInternalError;
      } else {
        local = flagcxInternalError;
      }
    }
    if (!sent)
      local = flagcxInternalError;
    flagcxIpcSocketClose(&socket);
  }
  if (local == flagcxSuccess && comm->localRank != 0)
    local = deviceAdaptor->symMulticastImport(fd, &arena->mcHandle);
  if (fd >= 0)
    close(fd);
  if (local == flagcxSuccess && arena->mcHandle == nullptr)
    local = flagcxInternalError;
  FLAGCXCHECK(flagcxSymLocalConverge(comm, local, 0x5996, &common));
  if (common != flagcxSuccess)
    return rollback();

  local = deviceAdaptor->symLsaMulticastMap(
      arena->mcHandle, arena->rankStride, arena->alignment, &arena->mcBase,
      &arena->mcMappingOwned, &arena->mcVaOwned);
  if (local == flagcxSuccess &&
      (arena->mcBase == nullptr || !arena->mcMappingOwned || !arena->mcVaOwned))
    local = flagcxInternalError;
  FLAGCXCHECK(flagcxSymLocalConverge(comm, local, 0x5997, &common));
  if (common != flagcxSuccess)
    return rollback();
  arena->mcReady = true;
  return flagcxSuccess;
}

static bool flagcxSymLsaCanBindMulticast(const flagcxSymLsaArena *arena,
                                         const flagcxSymWindow_t window,
                                         int localRanks) {
  if (arena == nullptr || !arena->mcReady || arena->mcGranularity == 0 ||
      window == nullptr || window->rankLayouts == nullptr || localRanks <= 0)
    return false;
  const flagcxSymRankLayout &first = window->rankLayouts[0];
  if (first.mappedSize == 0 || first.mappedSize % arena->mcGranularity != 0)
    return false;
  for (int peer = 1; peer < localRanks; peer++) {
    const flagcxSymRankLayout &layout = window->rankLayouts[peer];
    if (layout.mappedSize != first.mappedSize ||
        layout.userOffset != first.userOffset)
      return false;
  }
  return true;
}

static flagcxResult_t flagcxSymLsaAllocateWindow(flagcxHeteroComm_t comm,
                                                 flagcxSymWindow_t window) {
  if (comm == nullptr || window == nullptr || comm->symLsaArena == nullptr ||
      comm->symLsaArena->invalid)
    return flagcxInvalidArgument;
  flagcxSymLsaArena *arena = comm->symLsaArena;
  const size_t span = window->slotStride;
  const bool mcCandidate =
      flagcxSymLsaCanBindMulticast(arena, window, comm->localRanks);
  const size_t alignment =
      mcCandidate && arena->mcGranularity > window->slotAlignment
          ? arena->mcGranularity
          : window->slotAlignment;
  size_t offset = 0;
  flagcxResult_t local =
      flagcxSymLsaFindOffset(arena, span, alignment, &offset);
  flagcxResult_t common = flagcxSuccess;
  FLAGCXCHECK(flagcxSymLocalConverge(comm, local, 0x5987, &common));
  if (common != flagcxSuccess)
    return common;

  const int root = comm->localRankToRank[0];
  size_t agreedOffset = offset;
  if (comm->localRank == 0) {
    for (int peer = 1; peer < comm->localRanks; peer++) {
      size_t remote = 0;
      FLAGCXCHECK(bootstrapRecv(comm->bootstrap, comm->localRankToRank[peer],
                                0x5988, &remote, sizeof(remote)));
      if (remote != offset)
        agreedOffset = std::numeric_limits<size_t>::max();
    }
    for (int peer = 1; peer < comm->localRanks; peer++)
      FLAGCXCHECK(bootstrapSend(comm->bootstrap, comm->localRankToRank[peer],
                                0x5989, &agreedOffset, sizeof(agreedOffset)));
  } else {
    FLAGCXCHECK(
        bootstrapSend(comm->bootstrap, root, 0x5988, &offset, sizeof(offset)));
    FLAGCXCHECK(bootstrapRecv(comm->bootstrap, root, 0x5989, &agreedOffset,
                              sizeof(agreedOffset)));
  }
  if (agreedOffset == std::numeric_limits<size_t>::max())
    return flagcxInvalidUsage;

  local = flagcxSymLsaMarkOffset(arena, offset, span);
  if (local == flagcxSuccess) {
    window->lsaWindowOffset = offset;
    window->lsaWindowSpan = span;
    window->lsaOffsetOwned = true;
    window->slotStride = arena->rankStride;
    window->resourcePath = flagcxSymResourceLsa;
  }
  FLAGCXCHECK(flagcxSymLocalConverge(comm, local, 0x598a, &common));
  return common;
}

flagcxResult_t flagcxSymLsaDestroy(flagcxHeteroComm_t comm) {
  if (comm == nullptr)
    return flagcxInvalidArgument;
  flagcxSymLsaArena *arena = comm->symLsaArena;
  if (arena == nullptr)
    return flagcxSuccess;
  if (!arena->invalid &&
      (arena->freeRanges.size() != 1 || arena->freeRanges.begin()->first != 0 ||
       arena->freeRanges.begin()->second != arena->rankStride))
    return flagcxInvalidUsage;
  FLAGCXCHECK(flagcxSymLsaReleaseMulticast(arena));
  if (arena->base != nullptr) {
    if (deviceAdaptor == nullptr || deviceAdaptor->symLsaVaFree == nullptr)
      return flagcxInternalError;
    flagcxResult_t result =
        deviceAdaptor->symLsaVaFree(arena->base, arena->totalSize);
    if (result != flagcxSuccess)
      return result;
  }
  delete arena;
  comm->symLsaArena = nullptr;
  return flagcxSuccess;
}

flagcxResult_t
flagcxSymValidateLayouts(int nRanks, const flagcxSymRankLayout *layouts,
                         int nSegments, const flagcxSymSegmentMeta *segments,
                         size_t slotAlignment, size_t *slotStride) {
  if (nRanks <= 0 || layouts == nullptr || nSegments <= 0 ||
      segments == nullptr || slotAlignment == 0 || slotStride == nullptr)
    return flagcxInvalidArgument;

  size_t maxMappedSize = 0;
  int expectedFirst = 0;
  for (int rank = 0; rank < nRanks; rank++) {
    const flagcxSymRankLayout &layout = layouts[rank];
    if (layout.firstSegment != expectedFirst || layout.segmentCount <= 0 ||
        layout.segmentCount > nSegments - expectedFirst ||
        layout.userSize == 0 || layout.userOffset > layout.mappedSize ||
        layout.userSize > layout.mappedSize - layout.userOffset)
      return flagcxInvalidArgument;

    size_t previousEnd = 0;
    size_t coveredTo = layout.userOffset;
    const size_t userEnd = layout.userOffset + layout.userSize;
    for (int index = layout.firstSegment;
         index < layout.firstSegment + layout.segmentCount; index++) {
      const flagcxSymSegmentMeta &segment = segments[index];
      // Slot alignment constrains the start of each rank's VA slot. Segment
      // map granularity belongs to the adaptor that owns the provider handle.
      if (segment.size == 0 || segment.offset > layout.mappedSize ||
          segment.size > layout.mappedSize - segment.offset ||
          (index != layout.firstSegment && segment.offset < previousEnd))
        return flagcxInvalidArgument;
      const size_t end = segment.offset + segment.size;
      if (segment.offset <= coveredTo && end > coveredTo)
        coveredTo = end;
      previousEnd = end;
    }
    if (previousEnd != layout.mappedSize || coveredTo < userEnd)
      return flagcxInvalidArgument;
    if (layout.mappedSize > maxMappedSize)
      maxMappedSize = layout.mappedSize;
    expectedFirst += layout.segmentCount;
  }
  if (expectedFirst != nSegments ||
      maxMappedSize > std::numeric_limits<size_t>::max() - (slotAlignment - 1))
    return flagcxInvalidArgument;
  const size_t stride =
      ((maxMappedSize + slotAlignment - 1) / slotAlignment) * slotAlignment;
  if (stride == 0 ||
      stride > std::numeric_limits<size_t>::max() / static_cast<size_t>(nRanks))
    return flagcxInvalidArgument;
  *slotStride = stride;
  return flagcxSuccess;
}

flagcxResult_t flagcxSymLsaPeerPtr(void *arenaBase, size_t rankStride,
                                   size_t windowOffset, int localRanks,
                                   int peerLocalRank,
                                   const flagcxSymRankLayout *layout,
                                   size_t offset, size_t size, void **ptr) {
  if (ptr == nullptr)
    return flagcxInvalidArgument;
  *ptr = nullptr;
  if (arenaBase == nullptr || layout == nullptr || rankStride == 0 ||
      localRanks <= 0 || peerLocalRank < 0 || peerLocalRank >= localRanks ||
      rankStride > std::numeric_limits<size_t>::max() /
                       static_cast<size_t>(localRanks) ||
      windowOffset > rankStride ||
      layout->mappedSize > rankStride - windowOffset ||
      layout->userOffset > layout->mappedSize ||
      layout->userSize > layout->mappedSize - layout->userOffset ||
      offset > layout->userSize || size > layout->userSize - offset)
    return flagcxInvalidArgument;
  const size_t displacement = static_cast<size_t>(peerLocalRank) * rankStride +
                              windowOffset + layout->userOffset + offset;
  if (displacement > std::numeric_limits<uintptr_t>::max() -
                         reinterpret_cast<uintptr_t>(arenaBase))
    return flagcxInvalidArgument;
  *ptr = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(arenaBase) +
                                  displacement);
  return flagcxSuccess;
}

flagcxResult_t
flagcxSymCountUserRangeSegments(const flagcxSymRankLayout *layout,
                                const flagcxSymSegmentMeta *rankSegments,
                                int *count) {
  if (layout == nullptr || rankSegments == nullptr || count == nullptr ||
      layout->segmentCount <= 0 || layout->userSize == 0 ||
      layout->userOffset > layout->mappedSize ||
      layout->userSize > layout->mappedSize - layout->userOffset)
    return flagcxInvalidArgument;
  *count = 0;
  const size_t userEnd = layout->userOffset + layout->userSize;
  size_t coveredTo = layout->userOffset;
  size_t previousEnd = 0;
  for (int i = 0; i < layout->segmentCount; ++i) {
    const flagcxSymSegmentMeta &segment = rankSegments[i];
    if (segment.size == 0 || segment.offset > layout->mappedSize ||
        segment.size > layout->mappedSize - segment.offset ||
        (i > 0 && segment.offset < previousEnd))
      return flagcxInvalidArgument;
    const size_t end = segment.offset + segment.size;
    if (segment.offset < userEnd && end > layout->userOffset) {
      if (segment.offset > coveredTo)
        return flagcxInvalidArgument;
      if (end > coveredTo)
        coveredTo = end;
      ++*count;
    }
    previousEnd = end;
  }
  return coveredTo >= userEnd && *count > 0 ? flagcxSuccess
                                            : flagcxInvalidArgument;
}

static bool flagcxSymWindowHasValidNetworkMr(flagcxHeteroComm_t comm,
                                             flagcxSymWindow_t window) {
  if (comm == nullptr || window == nullptr || !window->hasNetworkMrRef ||
      window->mrIndex < 0 || window->mrIndex >= comm->oneSideHandleCount ||
      comm->oneSideHandles == nullptr)
    return false;
  struct flagcxOneSideHandleInfo *handle =
      comm->oneSideHandles[window->mrIndex];
  return handle != nullptr && handle->baseVas != nullptr &&
         handle->regionSizes != nullptr && comm->rank >= 0 &&
         comm->rank < comm->nRanks &&
         handle->baseVas[comm->rank] == (uintptr_t)window->localBase &&
         handle->regionSizes[comm->rank] == window->heapSize;
}

flagcxResult_t flagcxSymWindowEnsureNetworkMr(flagcxHeteroComm_t comm,
                                              flagcxSymWindow_t window) {
  if (comm == nullptr || window == nullptr || window->localBase == nullptr ||
      window->heapSize == 0)
    return flagcxInvalidArgument;
  if (window->hasNetworkMrRef)
    return flagcxSymWindowHasValidNetworkMr(comm, window) ? flagcxSuccess
                                                          : flagcxInternalError;

  int mrRangeSegmentCount = 0;
  flagcxResult_t rangeStatus = flagcxSuccess;
  if (window->isVmmAllocation) {
    if (window->rankLayouts != nullptr && window->segments != nullptr) {
      if (comm->localRank < 0 || comm->localRank >= window->localRanks) {
        rangeStatus = flagcxInternalError;
      } else {
        const flagcxSymRankLayout &layout =
            window->rankLayouts[comm->localRank];
        if (layout.firstSegment < 0 || layout.segmentCount <= 0 ||
            layout.firstSegment > window->totalSegments ||
            layout.segmentCount > window->totalSegments - layout.firstSegment) {
          rangeStatus = flagcxInternalError;
        } else {
          rangeStatus = flagcxSymCountUserRangeSegments(
              &layout, window->segments + layout.firstSegment,
              &mrRangeSegmentCount);
        }
      }
    } else if (window->resourcePath == flagcxSymResourceLegacyV1) {
      mrRangeSegmentCount = 1;
    } else if (window->resourcePath == flagcxSymResourceSegments ||
               window->resourcePath == flagcxSymResourceLsa) {
      rangeStatus = flagcxInternalError;
    }
  }
  flagcxResult_t commonRangeStatus = flagcxSuccess;
  FLAGCXCHECK(flagcxSymGlobalConverge(comm, rangeStatus, &commonRangeStatus));
  if (commonRangeStatus != flagcxSuccess)
    return commonRangeStatus;

  int mrIndex = -1;
  bool rollbackPending = false;
  flagcxResult_t result = flagcxOneSideRegisterInternal(
      comm, window->localBase, window->heapSize, window->isVmmAllocation,
      /*acquireWindowRef=*/true, &mrIndex, &rollbackPending,
      mrRangeSegmentCount);
  if (result != flagcxSuccess) {
    window->hasPendingNetworkCleanup = rollbackPending;
    return result;
  }

  // Take ownership before validating the published metadata. If validation
  // detects an internal inconsistency, the caller's normal rollback path must
  // still release (or retain for retry) the acquired MR reference.
  window->mrIndex = mrIndex;
  window->hasNetworkMrRef = true;
  if (!flagcxSymWindowHasValidNetworkMr(comm, window))
    return flagcxInternalError;
  window->mrBase = comm->oneSideHandles[mrIndex]->baseVas[comm->rank];
  return flagcxSuccess;
}

flagcxResult_t
flagcxSymWindowValidateDataRoutesForMode(flagcxHeteroComm_t comm,
                                         flagcxSymWindow_t window,
                                         bool localPeerTransportEnabled) {
  if (comm == nullptr || window == nullptr || comm->nRanks <= 0 ||
      comm->localRanks <= 0 || comm->localRanks > comm->nRanks)
    return flagcxInvalidArgument;

  const bool hasNetworkMr = flagcxSymWindowHasValidNetworkMr(comm, window);
  const bool needsLocalPeerRoute = comm->localRanks > 1;
  const bool hasLocalPeerRoute =
      !needsLocalPeerRoute || hasNetworkMr ||
      (localPeerTransportEnabled &&
       ((window->hasFlatMapping && window->flatBase != nullptr) ||
        window->ipcSlot >= 0));
  const bool hasRemotePeerRoute =
      comm->localRanks == comm->nRanks || hasNetworkMr;
  return hasLocalPeerRoute && hasRemotePeerRoute ? flagcxSuccess
                                                 : flagcxNotSupported;
}

flagcxResult_t flagcxSymWindowValidateDataRoutes(flagcxHeteroComm_t comm,
                                                 flagcxSymWindow_t window) {
  return flagcxSymWindowValidateDataRoutesForMode(comm, window,
                                                  !flagcxParamP2pDisable());
}

static flagcxResult_t flagcxSymCleanupStepConverge(flagcxHeteroComm_t comm,
                                                   flagcxResult_t local,
                                                   int tag,
                                                   flagcxSymCleanupMode mode,
                                                   flagcxResult_t *common) {
  if (common == nullptr)
    return flagcxInvalidArgument;
  if (mode == flagcxSymCleanupCollective && comm != nullptr &&
      comm->bootstrap != nullptr && comm->localRanks > 1)
    return flagcxSymLocalConverge(comm, local, tag, common);
  *common = local;
  return flagcxSuccess;
}

static flagcxResult_t flagcxSymCleanupMappings(flagcxHeteroComm_t comm,
                                               flagcxSymWindow_t d,
                                               flagcxSymCleanupMode mode) {
  const bool hasVmmOwnership =
      d != nullptr &&
      (d->flatMappingOwned || d->flatVaOwned || d->multicastMappingOwned ||
       d->multicastVaOwned || d->mcHandle != nullptr || d->mcLsaBound ||
       d->physHandle != nullptr || d->lsaOffsetOwned);
  if (hasVmmOwnership && deviceAdaptor == nullptr)
    return flagcxInternalError;

  flagcxResult_t local = flagcxSuccess;
  flagcxResult_t common = flagcxSuccess;

  if (d != nullptr && d->mcLsaBound) {
    flagcxSymLsaArena *arena = comm->symLsaArena;
    local = arena != nullptr && arena->mcHandle != nullptr &&
                    deviceAdaptor->symLsaMulticastUnbind != nullptr
                ? deviceAdaptor->symLsaMulticastUnbind(
                      arena->mcHandle, d->lsaWindowOffset, d->mcLsaBindSize)
                : flagcxInternalError;
    if (local == flagcxSuccess) {
      d->mcLsaBound = false;
      d->mcLsaBindSize = 0;
      d->mcBase = nullptr;
    }
  }
  FLAGCXCHECK(
      flagcxSymCleanupStepConverge(comm, local, /*tag=*/0x599b, mode, &common));
  if (common != flagcxSuccess)
    return common;

  local = flagcxSuccess;

  if (hasVmmOwnership && d->multicastMappingOwned) {
    local =
        deviceAdaptor->symMulticastMappingUnmap != nullptr
            ? deviceAdaptor->symMulticastMappingUnmap(d->mcBase, d->mcMapSize)
            : flagcxInternalError;
    if (local == flagcxSuccess)
      d->multicastMappingOwned = false;
  }
  FLAGCXCHECK(
      flagcxSymCleanupStepConverge(comm, local, /*tag=*/0x5941, mode, &common));
  if (common != flagcxSuccess)
    return common;

  local = flagcxSuccess;
  if (hasVmmOwnership && d->multicastVaOwned) {
    local = deviceAdaptor->symMulticastVaFree != nullptr
                ? deviceAdaptor->symMulticastVaFree(d->mcBase, d->mcMapSize)
                : flagcxInternalError;
    if (local == flagcxSuccess) {
      d->multicastVaOwned = false;
      d->mcBase = nullptr;
      d->mcMapSize = 0;
    }
  }
  FLAGCXCHECK(
      flagcxSymCleanupStepConverge(comm, local, /*tag=*/0x5942, mode, &common));
  if (common != flagcxSuccess)
    return common;

  local = flagcxSuccess;
  if (hasVmmOwnership && d->mcHandle != nullptr) {
    local = deviceAdaptor->symMulticastFree != nullptr
                ? deviceAdaptor->symMulticastFree(d->mcHandle)
                : flagcxInternalError;
    if (local == flagcxSuccess)
      d->mcHandle = nullptr;
  }
  FLAGCXCHECK(
      flagcxSymCleanupStepConverge(comm, local, /*tag=*/0x5943, mode, &common));
  if (common != flagcxSuccess)
    return common;

  local = flagcxSuccess;
  if (hasVmmOwnership && d->flatMappingOwned) {
    if (d->resourcePath == flagcxSymResourceSegments ||
        d->resourcePath == flagcxSymResourceLsa)
      local = deviceAdaptor->symFlatMappingUnmapSegments != nullptr
                  ? deviceAdaptor->symFlatMappingUnmapSegments(d->flatBase)
                  : flagcxInternalError;
    else
      local = deviceAdaptor->symFlatMappingUnmap != nullptr
                  ? deviceAdaptor->symFlatMappingUnmap(
                        d->flatBase, d->allocSize, d->localRanks)
                  : flagcxInternalError;
    if (local == flagcxSuccess) {
      d->flatMappingOwned = false;
      d->hasFlatMapping = false;
    }
  }
  FLAGCXCHECK(
      flagcxSymCleanupStepConverge(comm, local, /*tag=*/0x5944, mode, &common));
  if (common != flagcxSuccess)
    return common;

  local = flagcxSuccess;
  if (hasVmmOwnership && d->flatVaOwned) {
    if (d->resourcePath == flagcxSymResourceSegments ||
        d->resourcePath == flagcxSymResourceLsa)
      local = deviceAdaptor->symFlatVaFreeSegments != nullptr
                  ? deviceAdaptor->symFlatVaFreeSegments(d->flatBase)
                  : flagcxInternalError;
    else
      local = deviceAdaptor->symFlatVaFree != nullptr
                  ? deviceAdaptor->symFlatVaFree(d->flatBase, d->allocSize,
                                                 d->localRanks)
                  : flagcxInternalError;
    if (local == flagcxSuccess) {
      d->flatVaOwned = false;
      d->flatBase = nullptr;
    }
  }
  FLAGCXCHECK(
      flagcxSymCleanupStepConverge(comm, local, /*tag=*/0x5945, mode, &common));
  if (common != flagcxSuccess)
    return common;

  local = flagcxSuccess;
  if (d != nullptr && d->lsaOffsetOwned) {
    local = flagcxSymLsaReleaseOffset(comm->symLsaArena, d->lsaWindowOffset,
                                      d->lsaWindowSpan);
    if (local == flagcxSuccess)
      d->lsaOffsetOwned = false;
  }
  FLAGCXCHECK(
      flagcxSymCleanupStepConverge(comm, local, /*tag=*/0x598b, mode, &common));
  if (common != flagcxSuccess)
    return common;

  local = flagcxSuccess;
  if (hasVmmOwnership && d->physHandle != nullptr) {
    if (d->resourcePath == flagcxSymResourceSegments ||
        d->resourcePath == flagcxSymResourceLsa)
      local = deviceAdaptor->symPhysFreeSegments != nullptr
                  ? deviceAdaptor->symPhysFreeSegments(d->physHandle)
                  : flagcxInternalError;
    else
      local = deviceAdaptor->symPhysFree != nullptr
                  ? deviceAdaptor->symPhysFree(d->physHandle)
                  : flagcxInternalError;
    if (local == flagcxSuccess)
      d->physHandle = nullptr;
  }
  FLAGCXCHECK(
      flagcxSymCleanupStepConverge(comm, local, /*tag=*/0x5946, mode, &common));
  if (common == flagcxSuccess && d != nullptr && !d->flatMappingOwned &&
      !d->flatVaOwned && !d->multicastMappingOwned && !d->multicastVaOwned &&
      d->mcHandle == nullptr && d->physHandle == nullptr &&
      !d->lsaOffsetOwned) {
    d->hasFlatMapping = false;
    d->allocSize = 0;
  }
  return common;
}

static void flagcxSymFreeWindowState(flagcxSymWindow_t d) {
  if (d == nullptr)
    return;
  free(d->rankLayouts);
  free(d->segments);
  free(d);
}

static flagcxResult_t flagcxSymReleaseAllocationLease(flagcxSymWindow_t d) {
  if (d == nullptr || !d->hasAllocationLease)
    return flagcxSuccess;
  flagcxResult_t result =
      globalMemAllocRegistry.releaseWindow(d->allocationBase);
  if (result == flagcxSuccess) {
    d->hasAllocationLease = false;
    d->allocationBase = nullptr;
  }
  return result;
}

// Mapping teardown is ordered among local peers because multicast and flat VA
// ownership are node-local. Public deregistration is nevertheless a
// communicator-wide collective: no rank may detach/free its window until all
// nodes report that their local teardown completed. Communicator destruction
// uses local mode and deliberately skips this rendezvous.
static flagcxResult_t
flagcxSymCleanupMappingsConverged(flagcxHeteroComm_t comm, flagcxSymWindow_t d,
                                  flagcxSymCleanupMode mode) {
  flagcxResult_t local = flagcxSymCleanupMappings(comm, d, mode);
  if (mode == flagcxSymCleanupLocal)
    return local;

  flagcxResult_t common = flagcxSuccess;
  FLAGCXCHECK(flagcxSymGlobalConverge(comm, local, &common));
  if (d != nullptr && d->resourcePath == flagcxSymResourceLsa)
    d->lsaCleanupFailed = common != flagcxSuccess;
  return common;
}

static flagcxResult_t flagcxSymCleanupUnpublished(flagcxHeteroComm_t comm,
                                                  flagcxSymWindow_t d,
                                                  flagcxSymCleanupMode mode) {
  if (comm == nullptr)
    return flagcxInvalidArgument;
  flagcxResult_t local = flagcxSuccess;
  const bool retryingPendingNetworkCleanup =
      d != nullptr && d->hasPendingNetworkCleanup;
  if (retryingPendingNetworkCleanup) {
    local = flagcxOneSideRetryPendingCleanup(comm);
  } else if (d != nullptr && d->hasNetworkMrRef && d->mrIndex >= 0) {
    local = flagcxOneSideDeregisterInternal(comm, d->mrIndex);
    if (local == flagcxSuccess) {
      d->hasNetworkMrRef = false;
      d->mrIndex = -1;
      d->mrBase = 0;
    }
  }
  flagcxResult_t common = flagcxSuccess;
  if (mode == flagcxSymCleanupCollective) {
    FLAGCXCHECK(flagcxSymGlobalConverge(comm, local, &common));
  } else {
    common = local;
  }
  if (common != flagcxSuccess)
    return common;

  // Pending cleanup is a collective participation token.  A rank with no
  // local object to retry must retain it until every rank reports success;
  // otherwise the next retry can split control flow and strand peers in the
  // status convergence.
  if (retryingPendingNetworkCleanup)
    d->hasPendingNetworkCleanup = false;

  return flagcxSymCleanupMappingsConverged(comm, d, mode);
}

flagcxResult_t flagcxSymRetryPendingCleanup(flagcxHeteroComm_t comm,
                                            flagcxSymCleanupMode mode) {
  if (comm == nullptr)
    return flagcxInvalidArgument;
  if (mode == flagcxSymCleanupLocal) {
    while (comm->pendingSymCleanup != nullptr) {
      flagcxSymWindow_t d = comm->pendingSymCleanup;
      FLAGCXCHECK(flagcxSymCleanupUnpublished(comm, d, mode));
      FLAGCXCHECK(flagcxSymReleaseAllocationLease(d));
      comm->pendingSymCleanup = d->cleanupNext;
      flagcxWindow_t owner = d->owner;
      flagcxSymFreeWindowState(d);
      free(owner);
    }
    return flagcxSuccess;
  }

  // Rollback ownership can be asymmetric: only a rank whose local teardown
  // failed needs an entry. Every rank still participates in each retry round
  // so MR and local-mapping status convergence cannot deadlock.
  while (true) {
    int anyPending = 0;
    FLAGCXCHECK(flagcxSymGlobalConvergeScalar(
        comm, comm->pendingSymCleanup != nullptr ? 1 : 0, kSymPendingGatherTag,
        kSymPendingBroadcastTag, flagcxSymLogicalOr, &anyPending));
    if (!anyPending)
      break;

    flagcxSymWindow_t d = comm->pendingSymCleanup;
    FLAGCXCHECK(flagcxSymCleanupUnpublished(comm, d, mode));
    if (d != nullptr) {
      FLAGCXCHECK(flagcxSymReleaseAllocationLease(d));
      comm->pendingSymCleanup = d->cleanupNext;
      flagcxWindow_t owner = d->owner;
      flagcxSymFreeWindowState(d);
      free(owner);
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxSymRetainPendingCleanup(flagcxHeteroComm_t comm,
                                             flagcxWindow_t win) {
  if (comm == nullptr || win == nullptr || win->defaultBase == nullptr)
    return flagcxInvalidArgument;
  flagcxSymWindow_t d = win->defaultBase;
  if (d->published)
    return flagcxInvalidUsage;
  for (flagcxSymWindow_t pending = comm->pendingSymCleanup; pending != nullptr;
       pending = pending->cleanupNext) {
    if (pending == d)
      return flagcxSuccess;
  }
  d->cleanupNext = comm->pendingSymCleanup;
  comm->pendingSymCleanup = d;
  d->state = flagcxSymWindowCleanupRequired;
  return flagcxSuccess;
}

struct flagcxSymSegmentHeader {
  size_t mappedSize;
  size_t userOffset;
  size_t userSize;
  size_t alignment;
  int segmentCount;
  int providerFamily;
};

static size_t flagcxSymGcd(size_t lhs, size_t rhs) {
  while (rhs != 0) {
    const size_t remainder = lhs % rhs;
    lhs = rhs;
    rhs = remainder;
  }
  return lhs;
}

static flagcxResult_t
flagcxSymExchangeLayouts(flagcxHeteroComm_t comm, flagcxSymWindow_t d,
                         const flagcxSymSegmentMeta *localSegments,
                         int localCount, size_t userOffset, size_t alignment) {
  const int nPeers = comm->localRanks;
  const int self = comm->localRank;
  const int root = comm->localRankToRank[0];
  flagcxSymSegmentHeader *headers = nullptr;
  flagcxResult_t local = flagcxCalloc(&headers, nPeers);
  flagcxResult_t common = flagcxSuccess;
  FLAGCXCHECK(flagcxSymLocalConverge(comm, local, 0x5960, &common));
  if (common != flagcxSuccess) {
    free(headers);
    return common;
  }
  headers[self] = {localSegments[localCount - 1].offset +
                       localSegments[localCount - 1].size,
                   userOffset,
                   d->heapSize,
                   alignment,
                   localCount,
                   deviceAdaptor->gdrDeviceFamily};
  auto exchange = [&]() -> flagcxResult_t {
    if (self == 0) {
      for (int peer = 1; peer < nPeers; peer++) {
        FLAGCXCHECK(bootstrapRecv(comm->bootstrap, comm->localRankToRank[peer],
                                  0x5961, &headers[peer], sizeof(*headers)));
      }
      for (int peer = 1; peer < nPeers; peer++) {
        FLAGCXCHECK(bootstrapSend(comm->bootstrap, comm->localRankToRank[peer],
                                  0x5962, headers,
                                  sizeof(*headers) * (size_t)nPeers));
      }
    } else {
      FLAGCXCHECK(bootstrapSend(comm->bootstrap, root, 0x5961, &headers[self],
                                sizeof(*headers)));
      FLAGCXCHECK(bootstrapRecv(comm->bootstrap, root, 0x5962, headers,
                                sizeof(*headers) * (size_t)nPeers));
    }

    int total = 0;
    size_t slotAlignment = 1;
    flagcxResult_t validation = flagcxSuccess;
    for (int peer = 0; peer < nPeers; peer++) {
      const flagcxSymSegmentHeader &header = headers[peer];
      if (header.providerFamily != headers[self].providerFamily ||
          header.providerFamily == FLAGCX_GDR_DEVICE_UNKNOWN) {
        validation = flagcxNotSupported;
        break;
      }
      if (header.segmentCount <= 0 ||
          header.segmentCount > std::numeric_limits<int>::max() - total ||
          header.alignment == 0 || header.mappedSize == 0 ||
          header.userOffset > header.mappedSize ||
          header.userSize > header.mappedSize - header.userOffset) {
        validation = flagcxInvalidArgument;
        break;
      }
      total += header.segmentCount;
      // Every rank's slot base must satisfy that rank's VA alignment. The
      // largest value alone is insufficient when alignments are not nested.
      const size_t factor =
          header.alignment / flagcxSymGcd(slotAlignment, header.alignment);
      if (slotAlignment > std::numeric_limits<size_t>::max() / factor) {
        validation = flagcxInvalidArgument;
        break;
      }
      slotAlignment *= factor;
    }
    FLAGCXCHECK(flagcxSymLocalConverge(comm, validation, 0x5963, &common));
    if (common != flagcxSuccess)
      return common;

    local = flagcxCalloc(&d->rankLayouts, nPeers);
    if (local == flagcxSuccess)
      local = flagcxCalloc(&d->segments, total);
    FLAGCXCHECK(flagcxSymLocalConverge(comm, local, 0x5964, &common));
    if (common != flagcxSuccess)
      return common;
    d->totalSegments = total;
    int first = 0;
    for (int peer = 0; peer < nPeers; peer++) {
      const flagcxSymSegmentHeader &header = headers[peer];
      d->rankLayouts[peer] = {header.mappedSize, header.userOffset,
                              header.userSize, first, header.segmentCount};
      first += header.segmentCount;
    }
    memcpy(d->segments + d->rankLayouts[self].firstSegment, localSegments,
           sizeof(*localSegments) * (size_t)localCount);
    if (self == 0) {
      for (int peer = 1; peer < nPeers; peer++) {
        const flagcxSymRankLayout &layout = d->rankLayouts[peer];
        FLAGCXCHECK(
            bootstrapRecv(comm->bootstrap, comm->localRankToRank[peer], 0x5965,
                          d->segments + layout.firstSegment,
                          sizeof(*d->segments) * (size_t)layout.segmentCount));
      }
      for (int peer = 1; peer < nPeers; peer++) {
        FLAGCXCHECK(bootstrapSend(comm->bootstrap, comm->localRankToRank[peer],
                                  0x5966, d->segments,
                                  sizeof(*d->segments) * (size_t)total));
      }
    } else {
      FLAGCXCHECK(bootstrapSend(comm->bootstrap, root, 0x5965,
                                d->segments + d->rankLayouts[self].firstSegment,
                                sizeof(*d->segments) * (size_t)localCount));
      FLAGCXCHECK(bootstrapRecv(comm->bootstrap, root, 0x5966, d->segments,
                                sizeof(*d->segments) * (size_t)total));
    }
    flagcxResult_t result =
        flagcxSymValidateLayouts(nPeers, d->rankLayouts, total, d->segments,
                                 slotAlignment, &d->slotStride);
    if (result == flagcxSuccess)
      d->slotAlignment = slotAlignment;
    return result;
  };
  flagcxResult_t result = exchange();
  free(headers);
  return result;
}

struct flagcxSymFdIdentity {
  int senderLocalRank;
  int segmentIndex;
};

static bool flagcxSymRecoverableMapError(flagcxResult_t result) {
  return result == flagcxNotSupported || result == flagcxUnhandledDeviceError;
}

static flagcxResult_t flagcxSymMapSegmentsStreaming(flagcxHeteroComm_t comm,
                                                    flagcxSymWindow_t d,
                                                    bool *mapFailure) {
  *mapFailure = false;
  void *flatBase = nullptr;
  flagcxResult_t local = flagcxSuccess;
  if (d->resourcePath == flagcxSymResourceLsa) {
    if (comm->symLsaArena == nullptr || comm->symLsaArena->base == nullptr ||
        deviceAdaptor->symLsaMapBeginAt == nullptr ||
        reinterpret_cast<uintptr_t>(comm->symLsaArena->base) >
            std::numeric_limits<uintptr_t>::max() - d->lsaWindowOffset)
      local = flagcxInternalError;
    else {
      void *windowBase = reinterpret_cast<void *>(
          reinterpret_cast<uintptr_t>(comm->symLsaArena->base) +
          d->lsaWindowOffset);
      local = deviceAdaptor->symLsaMapBeginAt(
          windowBase, comm->localRanks, comm->localRank, d->physHandle,
          d->rankLayouts, d->segments, d->totalSegments, d->slotStride,
          d->slotAlignment, &flatBase);
    }
  } else {
    local = deviceAdaptor->symFlatMapSegmentsBegin(
        comm->localRanks, comm->localRank, d->physHandle, d->rankLayouts,
        d->segments, d->totalSegments, d->slotStride, d->slotAlignment,
        &flatBase);
  }
  d->flatBase = flatBase;
  d->flatMappingOwned = flatBase != nullptr;
  d->flatVaOwned = flatBase != nullptr;
  if (local == flagcxSuccess && flatBase == nullptr)
    local = flagcxInternalError;
  flagcxResult_t common = flagcxSuccess;
  FLAGCXCHECK(flagcxSymLocalConverge(comm, local, 0x5967, &common));
  if (common != flagcxSuccess) {
    flagcxResult_t fallbackStatus = flagcxSuccess;
    FLAGCXCHECK(flagcxSymLocalConverge(
        comm,
        local == flagcxSuccess || flagcxSymRecoverableMapError(local)
            ? flagcxSuccess
            : flagcxNotSupported,
        0x5975, &fallbackStatus));
    *mapFailure = d->resourcePath != flagcxSymResourceLsa &&
                  fallbackStatus == flagcxSuccess;
    return common;
  }

  flagcxIpcSocket sock = {};
  const uint64_t ipcHash =
      comm->commHash ^ 0x7365676d656e7473ULL ^ comm->symWindowFdExchangeCount;
  local = comm->localRanks > 1
              ? flagcxIpcSocketInit(&sock, comm->rank, ipcHash, /*block=*/1)
              : flagcxSuccess;
  flagcxResult_t socketStatus =
      flagcxSymLocalConverge(comm, local, 0x5969, &common);
  if (socketStatus != flagcxSuccess) {
    if (local == flagcxSuccess && comm->localRanks > 1)
      flagcxIpcSocketClose(&sock);
    return socketStatus;
  }
  if (common != flagcxSuccess) {
    if (local == flagcxSuccess && comm->localRanks > 1)
      flagcxIpcSocketClose(&sock);
    return common;
  }

  auto transfer = [&]() -> flagcxResult_t {
    if (comm->localRanks > 1) {
      FLAGCXCHECK(bootstrapCollIntraNodeBarrier(
          comm->bootstrap, comm->localRankToRank, comm->localRank,
          comm->localRanks, 0x596a));
    }
    // Every rank follows the same owner/segment order. Only one exported FD
    // and one received FD can be live in a process in each round.
    for (int owner = 0; owner < comm->localRanks; owner++) {
      const flagcxSymRankLayout &layout = d->rankLayouts[owner];
      for (int segmentIndex = 0; segmentIndex < layout.segmentCount;
           segmentIndex++) {
        const int index = layout.firstSegment + segmentIndex;
        int fd = -1;
        local = flagcxSuccess;
        if (owner == comm->localRank && comm->localRanks > 1) {
          local = deviceAdaptor->symPhysExportSegment(d->physHandle,
                                                      segmentIndex, &fd);
          if (local == flagcxSuccess && fd < 0)
            local = flagcxInternalError;
        }
        flagcxResult_t exportStatus =
            flagcxSymLocalConverge(comm, local, 0x5968, &common);
        if (exportStatus != flagcxSuccess) {
          if (fd >= 0)
            close(fd);
          return exportStatus;
        }
        if (common != flagcxSuccess) {
          if (fd >= 0)
            close(fd);
          return common;
        }

        flagcxResult_t sendStatus = flagcxSuccess;
        if (owner == comm->localRank) {
          for (int peer = 0; peer < comm->localRanks; peer++) {
            if (peer == owner)
              continue;
            flagcxSymFdIdentity identity = {owner, segmentIndex};
            sendStatus =
                flagcxIpcSocketSendMsg(&sock, &identity, sizeof(identity), fd,
                                       comm->localRankToRank[peer], ipcHash);
            if (sendStatus != flagcxSuccess)
              break;
          }
        }
        if (fd >= 0)
          close(fd);
        FLAGCXCHECK(flagcxSymLocalConverge(comm, sendStatus, 0x596b, &common));
        if (common != flagcxSuccess)
          return common;

        flagcxResult_t mapStatus = flagcxSuccess;
        bool providerMapFailed = false;
        if (owner == comm->localRank) {
          mapStatus = deviceAdaptor->symFlatMapSegmentOne(flatBase, index, -1);
          providerMapFailed = mapStatus != flagcxSuccess;
        } else {
          flagcxSymFdIdentity identity = {-1, -1};
          int receivedFd = -1;
          mapStatus = flagcxIpcSocketRecvMsg(&sock, &identity, sizeof(identity),
                                             &receivedFd);
          if (mapStatus == flagcxSuccess &&
              (identity.senderLocalRank != owner ||
               identity.segmentIndex != segmentIndex || receivedFd < 0))
            mapStatus = flagcxInternalError;
          if (mapStatus == flagcxSuccess) {
            mapStatus = deviceAdaptor->symFlatMapSegmentOne(flatBase, index,
                                                            receivedFd);
            providerMapFailed = mapStatus != flagcxSuccess;
          }
          if (receivedFd >= 0)
            close(receivedFd);
        }
        FLAGCXCHECK(flagcxSymLocalConverge(comm, mapStatus, 0x596c, &common));
        if (common != flagcxSuccess) {
          flagcxResult_t fallbackStatus = flagcxSuccess;
          FLAGCXCHECK(flagcxSymLocalConverge(
              comm,
              mapStatus == flagcxSuccess ||
                      (providerMapFailed &&
                       flagcxSymRecoverableMapError(mapStatus))
                  ? flagcxSuccess
                  : flagcxNotSupported,
              0x5976, &fallbackStatus));
          *mapFailure = d->resourcePath != flagcxSymResourceLsa &&
                        fallbackStatus == flagcxSuccess;
          return common;
        }
      }
    }
    return flagcxSuccess;
  };
  flagcxResult_t result = transfer();
  if (comm->localRanks > 1)
    flagcxIpcSocketClose(&sock);
  if (result != flagcxSuccess)
    return result;
  comm->symWindowFdExchangeCount++;
  d->hasFlatMapping = true;
  return flagcxSuccess;
}

static flagcxResult_t flagcxSymLsaBindMulticast(flagcxHeteroComm_t comm,
                                                flagcxSymWindow_t d) {
  flagcxSymLsaArena *arena = comm->symLsaArena;
  if (arena == nullptr || !arena->mcReady)
    return flagcxSuccess;
  const flagcxSymRankLayout &self = d->rankLayouts[comm->localRank];
  const bool eligible =
      flagcxSymLsaCanBindMulticast(arena, d, comm->localRanks) &&
      d->lsaWindowOffset % arena->mcGranularity == 0;
  flagcxResult_t common = flagcxSuccess;
  FLAGCXCHECK(flagcxSymLocalConverge(
      comm, eligible ? flagcxSuccess : flagcxNotSupported, 0x5998, &common));
  if (common != flagcxSuccess)
    return flagcxSuccess;

  // The LSA alias is aligned to the arena's multicast granularity, even when
  // the caller's original allocation VA was reserved at a smaller VMM page.
  const uintptr_t source =
      reinterpret_cast<uintptr_t>(d->flatBase) +
      static_cast<size_t>(comm->localRank) * arena->rankStride;
  flagcxResult_t local = deviceAdaptor->symLsaMulticastBind(
      arena->mcHandle, d->lsaWindowOffset,
      reinterpret_cast<const void *>(source), self.mappedSize);
  if (local == flagcxSuccess) {
    d->mcLsaBound = true;
    d->mcLsaBindSize = self.mappedSize;
  }
  FLAGCXCHECK(flagcxSymLocalConverge(comm, local, 0x5999, &common));
  if (common != flagcxSuccess) {
    local = flagcxSuccess;
    if (d->mcLsaBound) {
      local = deviceAdaptor->symLsaMulticastUnbind(
          arena->mcHandle, d->lsaWindowOffset, d->mcLsaBindSize);
      if (local == flagcxSuccess) {
        d->mcLsaBound = false;
        d->mcLsaBindSize = 0;
      }
    }
    FLAGCXCHECK(flagcxSymLocalConverge(comm, local, 0x599a, &common));
    return common == flagcxSuccess ? flagcxSuccess : common;
  }
  d->mcBase =
      reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(arena->mcBase) +
                               d->lsaWindowOffset + self.userOffset);
  return flagcxSuccess;
}

static flagcxResult_t flagcxSymTrySegmentLayout(flagcxHeteroComm_t comm,
                                                flagcxSymWindow_t d, void *buff,
                                                size_t size, bool *used,
                                                bool *ordinaryIpcEligible,
                                                bool *cleanupFailed) {
  *used = false;
  *ordinaryIpcEligible = false;
  *cleanupFailed = false;
  const bool available =
      flagcxParamVmmEnable() && deviceAdaptor != nullptr &&
      deviceAdaptor->symPhysDescribe != nullptr &&
      deviceAdaptor->symPhysExportSegment != nullptr &&
      deviceAdaptor->symPhysFreeSegments != nullptr &&
      deviceAdaptor->symFlatMapSegments != nullptr &&
      deviceAdaptor->symFlatMapSegmentsBegin != nullptr &&
      deviceAdaptor->symFlatMapSegmentOne != nullptr &&
      deviceAdaptor->symFlatMappingUnmapSegments != nullptr &&
      deviceAdaptor->symFlatVaFreeSegments != nullptr;
  flagcxResult_t commonAvailability = flagcxSuccess;
  FLAGCXCHECK(flagcxSymLocalConverge(
      comm, available ? flagcxSuccess : flagcxNotSupported, 0x5974,
      &commonAvailability));
  const bool trySegmentLayout = commonAvailability == flagcxSuccess;
  bool mapFailure = false;
  void *bundle = nullptr;
  const flagcxSymSegmentMeta *localSegments = nullptr;
  int localCount = 0;
  size_t userOffset = 0;
  size_t alignment = 0;
  flagcxResult_t local =
      trySegmentLayout
          ? deviceAdaptor->symPhysDescribe(buff, size, &bundle, &localSegments,
                                           &localCount, &userOffset, &alignment)
          : flagcxNotSupported;
  if (bundle != nullptr) {
    d->physHandle = bundle;
    d->resourcePath = flagcxSymResourceSegments;
  }
  if (local == flagcxSuccess &&
      (bundle == nullptr || localSegments == nullptr || localCount <= 0 ||
       alignment == 0 ||
       userOffset > std::numeric_limits<size_t>::max() - size ||
       localSegments[localCount - 1].offset >
           std::numeric_limits<size_t>::max() -
               localSegments[localCount - 1].size))
    local = flagcxInternalError;

  const char *prepareStage = "describe";
  auto prepare = [&]() -> flagcxResult_t {
    flagcxResult_t fatal = local == flagcxNotSupported ? flagcxSuccess : local;
    flagcxResult_t common = flagcxSuccess;
    FLAGCXCHECK(flagcxSymLocalConverge(comm, fatal, 0x596d, &common));
    if (common != flagcxSuccess)
      return common;
    FLAGCXCHECK(flagcxSymLocalConverge(comm, local, 0x596e, &common));
    if (common == flagcxNotSupported) {
      // An ordinary allocation may fall back to legacy IPC. A mixed node or
      // a partially described VMM allocation must fail as a whole.
      flagcxResult_t mixed =
          local == flagcxSuccess
              ? flagcxInvalidUsage
              : (bundle != nullptr ? flagcxNotSupported : flagcxSuccess);
      FLAGCXCHECK(flagcxSymLocalConverge(comm, mixed, 0x596f, &common));
      if (common != flagcxSuccess)
        return common;
      return flagcxNotSupported;
    }
    if (common != flagcxSuccess)
      return common;
    *used = true;
    d->isVmmAllocation = true;

    prepareStage = "exchange-layouts";
    local = flagcxSymExchangeLayouts(comm, d, localSegments, localCount,
                                     userOffset, alignment);
    FLAGCXCHECK(flagcxSymLocalConverge(comm, local, 0x5970, &common));
    if (common != flagcxSuccess)
      return common;

    prepareStage = "lsa-capability";
    const bool lsaAvailable = deviceAdaptor->symLsaGetCapacity != nullptr &&
                              deviceAdaptor->symLsaVaReserve != nullptr &&
                              deviceAdaptor->symLsaVaFree != nullptr &&
                              deviceAdaptor->symLsaMapBeginAt != nullptr;
    FLAGCXCHECK(flagcxSymLocalConverge(
        comm, lsaAvailable ? flagcxSuccess : flagcxNotSupported, 0x598c,
        &common));
    if (common != flagcxSuccess)
      return common;
    prepareStage = "lsa-arena";
    local = flagcxSymLsaEnsureArena(comm, d->slotStride, d->slotAlignment);
    FLAGCXCHECK(flagcxSymLocalConverge(comm, local, 0x598d, &common));
    if (common != flagcxSuccess)
      return common;
    prepareStage = "lsa-multicast";
    local = flagcxSymLsaEnsureMulticast(comm);
    if (local != flagcxSuccess && local != flagcxNotSupported)
      return local;
    prepareStage = "lsa-window";
    local = flagcxSymLsaAllocateWindow(comm, d);
    if (local != flagcxSuccess)
      return local;
    prepareStage = "segment-map";
    local = flagcxSymMapSegmentsStreaming(comm, d, &mapFailure);
    if (local != flagcxSuccess)
      return local;
    prepareStage = "multicast-bind";
    return flagcxSymLsaBindMulticast(comm, d);
  };

  flagcxResult_t prepared = prepare();
  if (prepared != flagcxSuccess &&
      !(prepared == flagcxNotSupported && !*used && bundle == nullptr))
    WARN("symmem segment preparation failed: stage=%s rank=%d result=%d "
         "mapFailure=%d",
         prepareStage, comm->rank, static_cast<int>(prepared),
         static_cast<int>(mapFailure));
  flagcxResult_t global = flagcxSuccess;
  FLAGCXCHECK(flagcxSymGlobalConverge(comm,
                                      prepared == flagcxNotSupported &&
                                              !*used && bundle == nullptr
                                          ? flagcxSuccess
                                          : prepared,
                                      &global));
  if (global == flagcxSuccess && prepared == flagcxNotSupported) {
    *used = false;
    // A latest adaptor identified ordinary IPC memory. Do not probe it again
    // through the v1 VMM callback, which expects a physical VMM allocation.
    *ordinaryIpcEligible =
        trySegmentLayout && local == flagcxNotSupported && bundle == nullptr;
    return flagcxSuccess;
  }
  if (global == flagcxSuccess)
    return flagcxSuccess;

  // Only provider mapping failures may switch to NET. Protocol, layout, and
  // FD exchange failures remain fatal even if the partial mapping is released.
  flagcxResult_t fallbackStatus = flagcxSuccess;
  FLAGCXCHECK(flagcxSymGlobalConverge(
      comm,
      prepared == flagcxSuccess ||
              (prepared == flagcxNotSupported && !*used && bundle == nullptr) ||
              mapFailure
          ? flagcxSuccess
          : flagcxNotSupported,
      &fallbackStatus));
  flagcxResult_t netStatus = flagcxSuccess;
  FLAGCXCHECK(flagcxSymGlobalConverge(
      comm, flagcxParamIbDisable() ? flagcxNotSupported : flagcxSuccess,
      &netStatus));

  // Every rank enters the same local cleanup stages, including ranks that
  // owned no provider objects. A failed teardown stays attached to the
  // unpublished window for the existing cleanup-token retry path.
  flagcxResult_t rollback =
      flagcxSymCleanupMappings(comm, d, flagcxSymCleanupCollective);
  flagcxResult_t globalRollback = flagcxSuccess;
  FLAGCXCHECK(flagcxSymGlobalConverge(comm, rollback, &globalRollback));
  *cleanupFailed = globalRollback != flagcxSuccess;
  if (*cleanupFailed)
    return globalRollback;
  // Keep the VMM provenance and peer layouts for NET MR selection and bounds
  // checks, but never publish an incomplete flat mapping.
  return fallbackStatus == flagcxSuccess && netStatus == flagcxSuccess
             ? flagcxSuccess
             : global;
}

flagcxResult_t flagcxSymWindowRegisterInternal(flagcxHeteroComm_t comm,
                                               void *buff, size_t size,
                                               flagcxWindow_t *win,
                                               int winFlags,
                                               bool isVmmAllocation) {
  if (comm == nullptr || buff == nullptr || size == 0 || win == nullptr)
    return flagcxInvalidArgument;
  *win = nullptr;
  // A cleanup token represents an unfinished collective transaction. Starting
  // another registration could give ranks different window/MR slot numbers,
  // so all ranks reject it until the application retries deregistration.
  int anyPendingCleanup = 0;
  int localPendingCleanup = comm->pendingSymCleanup != nullptr ? 1 : 0;
  for (flagcxSymWindow_t current = comm->symWindows;
       current != nullptr && !localPendingCleanup; current = current->next)
    if (current->lsaCleanupFailed)
      localPendingCleanup = 1;
  FLAGCXCHECK(flagcxSymGlobalConvergeScalar(
      comm, localPendingCleanup, kSymPendingGatherTag, kSymPendingBroadcastTag,
      flagcxSymLogicalOr, &anyPendingCleanup));
  if (anyPendingCleanup)
    return flagcxInvalidUsage;

  // Flat VMM and IPC mappings are node-local. The runtime bypasses them when
  // P2P is disabled or the device API explicitly forces NET. Converge whether
  // any rank needs NET so every rank takes the same collective MR path even if
  // its local environment is inconsistent.
  int needsNetworkMr = 0;
  FLAGCXCHECK(flagcxSymGlobalConvergeScalar(
      comm,
      comm->localRanks < comm->nRanks ||
              (comm->localRanks > 1 && flagcxParamP2pDisable()) ||
              flagcxParamDeviceOneSidedForceNet()
          ? 1
          : 0,
      kSymNetworkRouteGatherTag, kSymNetworkRouteBroadcastTag,
      flagcxSymLogicalOr, &needsNetworkMr));

  // A required network route cannot be published while NET is disabled.
  // Converge this prerequisite before any rank enters VMM FD or MR metadata
  // exchange; this also handles inconsistent rank-local environments without
  // splitting collective control flow.
  flagcxResult_t localRouteStatus = needsNetworkMr && flagcxParamIbDisable()
                                        ? flagcxNotSupported
                                        : flagcxSuccess;
  flagcxResult_t commonRouteStatus = flagcxSuccess;
  FLAGCXCHECK(
      flagcxSymGlobalConverge(comm, localRouteStatus, &commonRouteStatus));
  if (commonRouteStatus != flagcxSuccess)
    return commonRouteStatus;

  flagcxResult_t res = flagcxSuccess;
  flagcxWindow_t w = nullptr;
  flagcxSymWindow_t d = nullptr;
  int *allFds = nullptr;
  int *localDevices = nullptr;
  void **peerHandles = nullptr;
  void *physHandle = nullptr;
  void *mcHandle = nullptr;
  int mcFd = -1;
  int shareableFd = -1;
  bool ipcSockOpen = false;
  bool mcIpcSockOpen = false;
  bool retainCleanupForRetry = false;
  bool prepareRollbackConverged = false;
  bool segmentCallbacksUsed = false;
  bool ordinaryIpcEligible = false;
  bool segmentedCleanupFailed = false;
  bool networkMrAttempted = false;
  flagcxResult_t localPrepareStatus = flagcxSuccess;
  void *allocationBase = nullptr;
  flagcxResult_t leaseResult = flagcxSuccess;
  struct flagcxIpcSocket ipcSock;
  struct flagcxIpcSocket mcIpcSock;
  memset(&ipcSock, 0, sizeof(ipcSock));
  memset(&mcIpcSock, 0, sizeof(mcIpcSock));

  // The window wrappers are rank-local preparation. Converge their outcome
  // before any rank starts VMM/IPC collectives; otherwise a single calloc
  // failure can leave its peers blocked in FD exchange.
  flagcxResult_t localInitStatus = flagcxCalloc(&w, 1);
  if (localInitStatus == flagcxSuccess)
    localInitStatus = flagcxCalloc(&d, 1);
  flagcxResult_t commonInitStatus = flagcxSuccess;
  res = flagcxSymGlobalConverge(comm, localInitStatus, &commonInitStatus);
  if (res != flagcxSuccess) {
    free(d);
    free(w);
    return res;
  }
  if (commonInitStatus != flagcxSuccess) {
    free(d);
    free(w);
    return commonInitStatus;
  }

  w->vendorBase = nullptr;
  w->defaultBase = d;
  w->isSymmetricDefault = 1;
  w->winFlags = winFlags;

  d->mrIndex = -1;
  d->mrBase = 0;
  d->hasNetworkMrRef = false;
  d->ipcSlot = -1;
  d->localBase = buff;
  d->owner = w;
  d->isVmmAllocation = isVmmAllocation;
  d->state = flagcxSymWindowPreparing;

  // Buffers owned by flagcxMemAlloc carry a lease for the full lifetime of
  // the window, including failed-registration cleanup. External/internal
  // allocations are intentionally allowed and simply have no registry lease.
  leaseResult =
      globalMemAllocRegistry.retainWindowRange(buff, size, &allocationBase);
  if (leaseResult == flagcxSuccess) {
    d->hasAllocationLease = true;
    d->allocationBase = allocationBase;
  } else if (leaseResult != flagcxInvalidUsage) {
    localPrepareStatus = leaseResult;
  }

  // Registry retention can fail on one rank only (for example, allocation of
  // registry bookkeeping). Do not let successful ranks enter VMM preparation
  // until that status is communicator-wide.
  {
    flagcxResult_t commonLeaseStatus = flagcxSuccess;
    FLAGCXCHECKGOTO(
        flagcxSymGlobalConverge(comm, localPrepareStatus, &commonLeaseStatus),
        res, fail);
    if (commonLeaseStatus != flagcxSuccess) {
      res = commonLeaseStatus;
      goto fail;
    }
  }

  int localRanks;
  localRanks = comm->localRanks;
  int localRank;
  localRank = comm->localRank;
  d->localRanks = localRanks;
  d->heapSize = size;

  res =
      flagcxSymTrySegmentLayout(comm, d, buff, size, &segmentCallbacksUsed,
                                &ordinaryIpcEligible, &segmentedCleanupFailed);
  if (res != flagcxSuccess) {
    prepareRollbackConverged = true;
    retainCleanupForRetry = segmentedCleanupFailed;
    goto fail;
  }

  // ---- Try VMM path ----
  if (!segmentCallbacksUsed && !ordinaryIpcEligible) {
    bool vmmOk = false;
    const bool localVmmAvailable =
        deviceAdaptor->symPhysAlloc != nullptr &&
        deviceAdaptor->symPhysFree != nullptr &&
        deviceAdaptor->symFlatMap != nullptr &&
        deviceAdaptor->symFlatMappingUnmap != nullptr &&
        deviceAdaptor->symFlatVaFree != nullptr && flagcxParamVmmEnable();
    flagcxResult_t commonVmmAvailability = flagcxSuccess;
    FLAGCXCHECKGOTO(flagcxSymLocalConverge(
                        comm,
                        localVmmAvailable ? flagcxSuccess : flagcxNotSupported,
                        /*tag=*/0x592f, &commonVmmAvailability),
                    res, fail);
    // VMM FD exchange is node-local, so every local rank must select the same
    // path even when environment or plugin capabilities differ by process.
    if (commonVmmAvailability == flagcxSuccess) {
      // Allocate every rank-local exchange array before the first provider or
      // bootstrap operation. Their status is converged within the node so no
      // peer can enter FD exchange while another rank exits on ENOMEM.
      flagcxResult_t localScratchStatus = flagcxCalloc(&allFds, localRanks);
      if (localScratchStatus == flagcxSuccess)
        localScratchStatus = flagcxCalloc(&peerHandles, localRanks);
      if (localScratchStatus == flagcxSuccess)
        localScratchStatus = flagcxCalloc(&localDevices, localRanks);
      if (allFds != nullptr) {
        for (int i = 0; i < localRanks; i++)
          allFds[i] = -1;
      }
      flagcxResult_t scratchStatus = flagcxSuccess;
      FLAGCXCHECKGOTO(flagcxSymLocalConverge(comm, localScratchStatus,
                                             /*tag=*/0x5930, &scratchStatus),
                      res, fail);
      if (scratchStatus != flagcxSuccess) {
        localPrepareStatus = scratchStatus;
      } else {
        size_t handleSize = sizeof(int);
        size_t allocSize = 0;
        void *rangeBase = nullptr;
        size_t rangeSize = 0;
        const flagcxResult_t rangeRes =
            deviceAdaptor->getAddressRange != nullptr
                ? deviceAdaptor->getAddressRange(buff, &rangeBase, &rangeSize)
                : flagcxNotSupported;

        d->resourcePath = flagcxSymResourceLegacyV1;
        flagcxResult_t allocRes = deviceAdaptor->symPhysAlloc(
            buff, size, &physHandle, &shareableFd, &handleSize, &allocSize);
        bool unsupportedLegacyGeometry = false;
        if (rangeRes == flagcxSuccess &&
            (allocRes == flagcxSuccess || allocRes == flagcxInvalidUsage)) {
          const uintptr_t address = (uintptr_t)buff;
          const uintptr_t base = (uintptr_t)rangeBase;
          unsupportedLegacyGeometry =
              address < base || address - base > rangeSize ||
              size > rangeSize - (address - base) || address != base;
        }
        INFO(FLAGCX_INIT,
             "[symWindowRegister] symPhysAlloc: res=%d physHandle=%p "
             "shareableFd=%d allocSize=%zu buff=%p size=%zu",
             (int)allocRes, physHandle, shareableFd, allocSize, buff, size);
        flagcxResult_t localAllocStatus =
            (allocRes == flagcxSuccess && physHandle != nullptr &&
             shareableFd >= 0 && allocSize >= size)
                ? flagcxSuccess
                : (allocRes == flagcxSuccess ? flagcxInternalError : allocRes);
        if (unsupportedLegacyGeometry)
          localAllocStatus = flagcxNotSupported;

        // A non-VMM pointer is an expected capability miss, while a provider
        // failure after recognizing the allocation is fatal. Converge fatal
        // errors separately so an earlier rank's NotSupported result cannot
        // hide a later rank's real provider error.
        flagcxResult_t localAllocFatal =
            localAllocStatus == flagcxSuccess ||
                    localAllocStatus == flagcxNotSupported
                ? flagcxSuccess
                : localAllocStatus;
        flagcxResult_t allocFatal = flagcxSuccess;
        FLAGCXCHECKGOTO(flagcxSymLocalConverge(comm, localAllocFatal,
                                               /*tag=*/0x5931, &allocFatal),
                        res, fail);
        if (allocFatal == flagcxSuccess) {
          FLAGCXCHECKGOTO(flagcxSymLocalConverge(comm,
                                                 unsupportedLegacyGeometry
                                                     ? flagcxNotSupported
                                                     : flagcxSuccess,
                                                 /*tag=*/0x5975, &allocFatal),
                          res, fail);
        }
        flagcxResult_t allocStatus = allocFatal;
        if (allocStatus == flagcxSuccess) {
          FLAGCXCHECKGOTO(flagcxSymLocalConverge(comm, localAllocStatus,
                                                 /*tag=*/0x593d, &allocStatus),
                          res, fail);
        }
        if (allocStatus == flagcxNotSupported &&
            allocFatal != flagcxNotSupported) {
          // Either every local rank has an ordinary allocation or every rank
          // must use VMM. A mixed result would make peers select incompatible
          // IPC/flat ownership models, so reject it instead of silently
          // treating the successfully-probed VMM allocations as ordinary.
          flagcxResult_t localMixedStatus = localAllocStatus == flagcxSuccess
                                                ? flagcxInvalidUsage
                                                : flagcxSuccess;
          flagcxResult_t mixedStatus = flagcxSuccess;
          FLAGCXCHECKGOTO(flagcxSymLocalConverge(comm, localMixedStatus,
                                                 /*tag=*/0x593e, &mixedStatus),
                          res, fail);
          if (mixedStatus != flagcxSuccess)
            allocStatus = mixedStatus;
        }
        bool allAllocOk = allocStatus == flagcxSuccess;

        if (allAllocOk) {
          // symPhysAlloc is the native allocation probe. The public registry
          // is intentionally not required: custom allocators may create valid
          // VMM buffers directly through the device adaptor/runtime.
          d->isVmmAllocation = true;
          comm->symWindowFdExchangeCount++;
          // Exchange shareable FDs with intra-node peers via Unix Domain Socket
          allFds[localRank] = shareableFd;

          // Hash must be identical across all ranks
          uint64_t ipcHash = comm->commHash ^ size;

          FLAGCXCHECKGOTO(
              flagcxIpcSocketInit(&ipcSock, comm->rank, ipcHash, /*block=*/1),
              res, fail);
          ipcSockOpen = true;

          // Barrier to ensure all sockets are created before sending
          struct bootstrapState *state = comm->bootstrap;
          FLAGCXCHECKGOTO(bootstrapCollIntraNodeBarrier(state,
                                                        comm->localRankToRank,
                                                        localRank, localRanks,
                                                        /*tag=*/0x5932),
                          res, fail);

          // Send our FD to each peer
          for (int i = 0; i < localRanks; i++) {
            if (i == localRank)
              continue;
            int peerGlobalRank = comm->localRankToRank[i];
            FLAGCXCHECKGOTO(
                flagcxIpcSocketSendMsg(&ipcSock, &localRank, sizeof(localRank),
                                       shareableFd, peerGlobalRank, ipcHash),
                res, fail);
          }

          // Receive FDs from each peer
          int received = 0;
          int expected = localRanks - 1;
          while (received < expected) {
            int senderLocalRank = -1;
            int fd = -1;
            FLAGCXCHECKGOTO(flagcxIpcSocketRecvMsg(&ipcSock, &senderLocalRank,
                                                   sizeof(senderLocalRank),
                                                   &fd),
                            res, fail);
            if (senderLocalRank < 0 || senderLocalRank >= localRanks ||
                senderLocalRank == localRank || fd < 0 ||
                allFds[senderLocalRank] >= 0) {
              if (fd >= 0)
                close(fd);
              res = flagcxInternalError;
              goto fail;
            }
            allFds[senderLocalRank] = fd;
            received++;
          }

          flagcxIpcSocketClose(&ipcSock);
          ipcSockOpen = false;

          // Build peer handle pointers for symFlatMap
          for (int i = 0; i < localRanks; i++) {
            peerHandles[i] = &allFds[i];
          }

          void *flatBase = nullptr;
          flagcxResult_t mapRes = deviceAdaptor->symFlatMap
                                      ? deviceAdaptor->symFlatMap(
                                            peerHandles, localRanks, localRank,
                                            physHandle, allocSize, &flatBase)
                                      : flagcxNotSupported;
          INFO(FLAGCX_INIT,
               "[symWindowRegister] symFlatMap: mapRes=%d flatBase=%p "
               "localRanks=%d allocSize=%zu",
               (int)mapRes, flatBase, localRanks, allocSize);

          // Take ownership before convergence. A peer can fail after this rank
          // has mapped successfully, and rollback must retain both the mapping
          // and physical handle if unmap/free fails.
          d->flatBase = flatBase;
          d->physHandle = physHandle;
          d->allocSize = allocSize;
          d->hasFlatMapping = false;
          d->flatMappingOwned = flatBase != nullptr;
          d->flatVaOwned = flatBase != nullptr;
          physHandle = nullptr;
          flagcxResult_t localMapStatus =
              mapRes == flagcxSuccess && flatBase != nullptr
                  ? flagcxSuccess
                  : (mapRes == flagcxSuccess ? flagcxInternalError : mapRes);
          flagcxResult_t mapStatus = flagcxSuccess;
          FLAGCXCHECKGOTO(flagcxSymLocalConverge(comm, localMapStatus,
                                                 /*tag=*/0x5935, &mapStatus),
                          res, fail);
          if (mapStatus == flagcxSuccess) {
            d->hasFlatMapping = true;
            vmmOk = true;

            // Try multicast setup
            d->mcBase = nullptr;
            int mcSupported = 0;
            flagcxResult_t mcSupportResult =
                deviceAdaptor->symMulticastSupported != nullptr
                    ? deviceAdaptor->symMulticastSupported(&mcSupported)
                    : flagcxNotSupported;
            // Rank-local communicator destruction cannot rendezvous with peers.
            // Multicast is therefore enabled only when every process can retain
            // its own provider reference until after its local mapping
            // teardown.
            if (deviceAdaptor->symMulticastCreate == nullptr ||
                deviceAdaptor->symMulticastImport == nullptr ||
                deviceAdaptor->symMulticastBind == nullptr ||
                deviceAdaptor->symMulticastFree == nullptr)
              mcSupported = 0;
            if (deviceAdaptor->symMulticastMappingUnmap == nullptr ||
                deviceAdaptor->symMulticastVaFree == nullptr)
              mcSupported = 0;
            flagcxResult_t commonMcAvailability = flagcxSuccess;
            FLAGCXCHECKGOTO(flagcxSymLocalConverge(
                                comm,
                                mcSupportResult == flagcxSuccess && mcSupported
                                    ? flagcxSuccess
                                    : flagcxNotSupported,
                                /*tag=*/0x592e, &commonMcAvailability),
                            res, fail);
            mcSupported = commonMcAvailability == flagcxSuccess ? 1 : 0;
            if (mcSupported) {
              // Build local device ordinal array from peerInfo
              for (int i = 0; i < localRanks; i++) {
                int globalRank = comm->localRankToRank[i];
                localDevices[i] = comm->peerInfo[globalRank].cudaDev;
              }

              if (localRank == 0) {
                flagcxResult_t mcRes = deviceAdaptor->symMulticastCreate
                                           ? deviceAdaptor->symMulticastCreate(
                                                 allocSize, localRanks,
                                                 localDevices, &mcHandle, &mcFd)
                                           : flagcxNotSupported;
                if (mcRes != flagcxSuccess || mcHandle == nullptr || mcFd < 0) {
                  mcSupported = 0;
                } else {
                  // Retain ownership immediately so every later failure uses
                  // the same ordered cleanup state machine.
                  d->mcHandle = mcHandle;
                  mcHandle = nullptr;
                }
              }
              // A provider may fail after publishing a partial object. Move it
              // into the retryable window state before broadcasting the
              // optional multicast decision; never discard ownership on an
              // error return.
              flagcxResult_t mcCreateCleanup = flagcxSuccess;
              if (localRank == 0 && !mcSupported) {
                if (mcHandle != nullptr) {
                  d->mcHandle = mcHandle;
                  mcHandle = nullptr;
                  mcCreateCleanup =
                      deviceAdaptor->symMulticastFree(d->mcHandle);
                  if (mcCreateCleanup == flagcxSuccess)
                    d->mcHandle = nullptr;
                }
                if (mcFd >= 0) {
                  close(mcFd);
                  mcFd = -1;
                }
              }
              // Broadcast success/failure from rank 0
              struct bootstrapState *mcState = comm->bootstrap;
              if (localRank == 0) {
                for (int i = 1; i < localRanks; i++) {
                  int peerGlobalRank = comm->localRankToRank[i];
                  FLAGCXCHECKGOTO(bootstrapSend(mcState, peerGlobalRank,
                                                /*tag=*/0x5933, &mcSupported,
                                                sizeof(mcSupported)),
                                  res, fail);
                }
              } else {
                int rank0Global = comm->localRankToRank[0];
                FLAGCXCHECKGOTO(bootstrapRecv(mcState, rank0Global,
                                              /*tag=*/0x5933, &mcSupported,
                                              sizeof(mcSupported)),
                                res, fail);
              }
              flagcxResult_t mcCreateCleanupStatus = flagcxSuccess;
              FLAGCXCHECKGOTO(flagcxSymLocalConverge(comm, mcCreateCleanup,
                                                     /*tag=*/0x593c,
                                                     &mcCreateCleanupStatus),
                              res, fail);
              if (mcCreateCleanupStatus != flagcxSuccess) {
                localPrepareStatus = mcCreateCleanupStatus;
                retainCleanupForRetry = true;
                mcSupported = 0;
              }

              if (mcSupported) {
                uint64_t mcIpcHash = ipcHash ^ 0x4D43; // "MC"

                FLAGCXCHECKGOTO(flagcxIpcSocketInit(&mcIpcSock, comm->rank,
                                                    mcIpcHash, /*block=*/1),
                                res, fail);
                mcIpcSockOpen = true;

                // Barrier: ensure all peers have created their IPC sockets
                FLAGCXCHECKGOTO(bootstrapCollIntraNodeBarrier(
                                    state, comm->localRankToRank, localRank,
                                    localRanks, /*tag=*/0x5934),
                                res, fail);

                if (localRank == 0) {
                  for (int i = 1; i < localRanks; i++) {
                    int peerGlobalRank = comm->localRankToRank[i];
                    int tag = 0;
                    FLAGCXCHECKGOTO(
                        flagcxIpcSocketSendMsg(&mcIpcSock, &tag, sizeof(tag),
                                               mcFd, peerGlobalRank, mcIpcHash),
                        res, fail);
                  }
                } else {
                  int tag = -1;
                  FLAGCXCHECKGOTO(flagcxIpcSocketRecvMsg(&mcIpcSock, &tag,
                                                         sizeof(tag), &mcFd),
                                  res, fail);
                  if (tag != 0 || mcFd < 0) {
                    res = flagcxInternalError;
                    goto fail;
                  }
                }

                flagcxIpcSocketClose(&mcIpcSock);
                mcIpcSockOpen = false;

                // Every non-owner rank imports and retains its own provider
                // reference. This makes rank-local teardown safe even when the
                // creator rank destroys its communicator first.
                flagcxResult_t localImportStatus = flagcxSuccess;
                if (localRank != 0) {
                  localImportStatus =
                      deviceAdaptor->symMulticastImport(mcFd, &d->mcHandle);
                  if (localImportStatus == flagcxSuccess &&
                      d->mcHandle == nullptr)
                    localImportStatus = flagcxInternalError;
                } else if (d->mcHandle == nullptr) {
                  localImportStatus = flagcxInternalError;
                }
                flagcxResult_t importStatus = flagcxSuccess;
                FLAGCXCHECKGOTO(flagcxSymLocalConverge(comm, localImportStatus,
                                                       /*tag=*/0x5939,
                                                       &importStatus),
                                res, fail);
                if (importStatus != flagcxSuccess) {
                  flagcxResult_t cleanupRes = flagcxSuccess;
                  if (d->mcHandle != nullptr) {
                    cleanupRes =
                        deviceAdaptor->symMulticastFree != nullptr
                            ? deviceAdaptor->symMulticastFree(d->mcHandle)
                            : flagcxInternalError;
                    if (cleanupRes == flagcxSuccess)
                      d->mcHandle = nullptr;
                  }
                  flagcxResult_t cleanupStatus = flagcxSuccess;
                  FLAGCXCHECKGOTO(flagcxSymLocalConverge(comm, cleanupRes,
                                                         /*tag=*/0x593a,
                                                         &cleanupStatus),
                                  res, fail);
                  if (cleanupStatus != flagcxSuccess) {
                    localPrepareStatus = cleanupStatus;
                    retainCleanupForRetry = true;
                  }
                  mcSupported = 0;
                }

                if (mcSupported) {
                  // All ranks bind through their retained local handle.
                  void *mcBaseVa = nullptr;
                  size_t mcMapSize = 0;
                  flagcxResult_t mcRes =
                      deviceAdaptor->symMulticastBind
                          ? deviceAdaptor->symMulticastBind(
                                d->mcHandle, /*importFd=*/-1, d->physHandle,
                                allocSize, localRank, localRanks, &mcBaseVa,
                                &mcMapSize)
                          : flagcxNotSupported;
                  flagcxResult_t localMcStatus =
                      mcRes == flagcxSuccess && mcBaseVa != nullptr
                          ? flagcxSuccess
                          : (mcRes == flagcxSuccess ? flagcxInternalError
                                                    : mcRes);
                  d->mcBase = mcBaseVa;
                  d->mcMapSize = mcMapSize;
                  d->multicastMappingOwned = mcBaseVa != nullptr;
                  d->multicastVaOwned = mcBaseVa != nullptr;
                  flagcxResult_t mcStatus = flagcxSuccess;
                  FLAGCXCHECKGOTO(flagcxSymLocalConverge(comm, localMcStatus,
                                                         /*tag=*/0x5936,
                                                         &mcStatus),
                                  res, fail);
                  if (mcStatus == flagcxSuccess) {
                  } else {
                    WARN("symMulticastBind failed: res=%d mcBaseVa=%p "
                         "(localRank=%d)",
                         mcStatus, mcBaseVa, localRank);
                    flagcxResult_t cleanupRes = flagcxSuccess;
                    if (d->multicastMappingOwned) {
                      cleanupRes = deviceAdaptor->symMulticastMappingUnmap(
                          d->mcBase, d->mcMapSize);
                      if (cleanupRes == flagcxSuccess)
                        d->multicastMappingOwned = false;
                    }
                    flagcxResult_t cleanupStatus = flagcxSuccess;
                    FLAGCXCHECKGOTO(flagcxSymLocalConverge(comm, cleanupRes,
                                                           /*tag=*/0x5937,
                                                           &cleanupStatus),
                                    res, fail);
                    if (cleanupStatus != flagcxSuccess) {
                      localPrepareStatus = cleanupStatus;
                      retainCleanupForRetry = true;
                    } else {
                      cleanupRes = flagcxSuccess;
                      if (d->multicastVaOwned) {
                        cleanupRes = deviceAdaptor->symMulticastVaFree(
                            d->mcBase, d->mcMapSize);
                        if (cleanupRes == flagcxSuccess) {
                          d->multicastVaOwned = false;
                          d->mcBase = nullptr;
                          d->mcMapSize = 0;
                        }
                      }
                      FLAGCXCHECKGOTO(flagcxSymLocalConverge(comm, cleanupRes,
                                                             /*tag=*/0x593b,
                                                             &cleanupStatus),
                                      res, fail);
                      if (cleanupStatus != flagcxSuccess) {
                        localPrepareStatus = cleanupStatus;
                        retainCleanupForRetry = true;
                      }
                    }
                    if (cleanupStatus == flagcxSuccess) {
                      cleanupRes = flagcxSuccess;
                      if (d->mcHandle != nullptr) {
                        cleanupRes =
                            deviceAdaptor->symMulticastFree != nullptr
                                ? deviceAdaptor->symMulticastFree(d->mcHandle)
                                : flagcxInternalError;
                        if (cleanupRes == flagcxSuccess)
                          d->mcHandle = nullptr;
                      }
                      FLAGCXCHECKGOTO(flagcxSymLocalConverge(comm, cleanupRes,
                                                             /*tag=*/0x5938,
                                                             &cleanupStatus),
                                      res, fail);
                      if (cleanupStatus != flagcxSuccess) {
                        localPrepareStatus = cleanupStatus;
                        retainCleanupForRetry = true;
                      }
                    }
                    mcSupported = 0;
                  }
                }

                // Close the multicast FD
                if (mcFd >= 0) {
                  close(mcFd);
                  mcFd = -1;
                }
              }
              if (mcFd >= 0) {
                close(mcFd);
                mcFd = -1;
              }
            }
          } else {
            // A peer may have mapped successfully before another rank failed.
            // All ranks participate in ordered cleanup. If any rank cannot
            // release its mapping, preserve the complete ownership record for a
            // later retry instead of falling through to IPC.
            flagcxResult_t cleanupRes =
                flagcxSymCleanupMappings(comm, d, flagcxSymCleanupCollective);
            if (cleanupRes != flagcxSuccess) {
              WARN("symFlatMap rollback retained for retry: res=%d",
                   cleanupRes);
              localPrepareStatus = cleanupRes;
              retainCleanupForRetry = true;
            } else {
              d->hasFlatMapping = false;
              d->allocSize = 0;
            }
          }

          free(peerHandles);
          peerHandles = nullptr;
          // Close all FDs
          for (int i = 0; i < localRanks; i++) {
            if (allFds[i] >= 0)
              close(allFds[i]);
          }
          free(allFds);
          allFds = nullptr;
          shareableFd = -1;
        } else {
          // Some ranks may already own a physical allocation. Make every local
          // rank enter the same cleanup phases, retaining ownership if release
          // fails, before allowing IPC fallback.
          d->allocSize = allocSize;
          if (physHandle != nullptr) {
            d->physHandle = physHandle;
            physHandle = nullptr;
          }
          if (shareableFd >= 0) {
            close(shareableFd);
            shareableFd = -1;
          }
          flagcxResult_t cleanupRes =
              flagcxSymCleanupMappings(comm, d, flagcxSymCleanupCollective);
          if (cleanupRes != flagcxSuccess) {
            WARN("symPhysAlloc rollback retained for retry: res=%d",
                 cleanupRes);
            localPrepareStatus = cleanupRes;
            retainCleanupForRetry = true;
          } else {
            d->hasFlatMapping = false;
            d->allocSize = 0;
          }
          if (allocStatus != flagcxNotSupported &&
              localPrepareStatus == flagcxSuccess)
            localPrepareStatus = allocStatus;
          if (allocFatal == flagcxNotSupported &&
              localPrepareStatus == flagcxSuccess)
            localPrepareStatus = flagcxNotSupported;
        }
        if (!allAllocOk && shareableFd >= 0) {
          close(shareableFd);
          shareableFd = -1;
        }
      }
      free(localDevices);
      localDevices = nullptr;
      // The success path normally released these after FD exchange; these
      // guards also cover scratch allocation failure and phys-alloc rollback.
      if (allFds != nullptr) {
        for (int i = 0; i < localRanks; i++) {
          if (allFds[i] >= 0)
            close(allFds[i]);
        }
        free(allFds);
        allFds = nullptr;
      }
      free(peerHandles);
      peerHandles = nullptr;
    }

    // ---- IPC fallback if VMM not available ----
    if (!vmmOk && localPrepareStatus == flagcxSuccess) {
      // A provider error may return a partial physical handle. Route it
      // through the staged cleanup state machine so a failed release remains
      // represented by the window cleanup token.
      if (physHandle != nullptr) {
        d->physHandle = physHandle;
        physHandle = nullptr;
        flagcxResult_t cleanupRes =
            flagcxSymCleanupMappings(comm, d, flagcxSymCleanupCollective);
        if (cleanupRes != flagcxSuccess) {
          localPrepareStatus = cleanupRes;
          retainCleanupForRetry = true;
        }
      }
      if (localPrepareStatus == flagcxSuccess) {
        d->flatBase = nullptr;
        d->mcBase = nullptr;
        d->physHandle = nullptr;
        d->hasFlatMapping = false;
        d->allocSize = 0;
      }
    }
  }

  // VMM preparation and its local rollback complete independently on each
  // node. Converge the outcome across the whole communicator before any rank
  // enters one-sided MR setup, whose collectives must never race the caller's
  // window-status collective on another node.
  {
    flagcxResult_t globalPrepareStatus = flagcxSuccess;
    FLAGCXCHECKGOTO(
        flagcxSymGlobalConverge(comm, localPrepareStatus, &globalPrepareStatus),
        res, fail);
    if (globalPrepareStatus != flagcxSuccess) {
      // Nodes that prepared successfully roll back now. A node whose earlier
      // rollback failed keeps its exact ownership for the next registration's
      // retry instead of immediately re-running a failed release operation.
      // Keep the preparation result separate from rollback ownership. A rank
      // whose local prepare failed may already have released every partial
      // provider object successfully; that original error must still be
      // returned, but it must not manufacture a cleanup-required token.
      flagcxResult_t localRollbackStatus =
          retainCleanupForRetry ? localPrepareStatus : flagcxSuccess;
      if (localPrepareStatus == flagcxSuccess) {
        localRollbackStatus =
            flagcxSymCleanupMappings(comm, d, flagcxSymCleanupCollective);
        if (localRollbackStatus == flagcxSuccess) {
          d->hasFlatMapping = false;
          d->allocSize = 0;
        } else {
          retainCleanupForRetry = true;
        }
      }

      flagcxResult_t globalRollbackStatus = flagcxSuccess;
      FLAGCXCHECKGOTO(flagcxSymGlobalConverge(comm, localRollbackStatus,
                                              &globalRollbackStatus),
                      res, fail);
      prepareRollbackConverged = true;
      retainCleanupForRetry = globalRollbackStatus != flagcxSuccess;
      res = retainCleanupForRetry ? globalRollbackStatus : globalPrepareStatus;
      goto fail;
    }
  }

  // ---- Network MR registration ----
  // Remote peers always require NET. A local-only communicator also requires
  // NET when P2P is disabled because runtime PUT/signal paths intentionally
  // bypass its flat/IPC mappings. A VMM allocation whose optional flat mapping
  // failed is also NET-only: it must never enter legacy IPC export.
  FLAGCXCHECKGOTO(
      flagcxSymGlobalConvergeScalar(
          comm, d->isVmmAllocation && !d->hasFlatMapping ? 1 : needsNetworkMr,
          kSymFallbackRouteGatherTag, kSymFallbackRouteBroadcastTag,
          flagcxSymLogicalOr, &needsNetworkMr),
      res, fail);
  if (needsNetworkMr) {
    INFO(FLAGCX_INIT,
         "[symWindowRegister] vmmOk=%d, registering MR for buff=%p size=%zu",
         (int)d->hasFlatMapping, buff, size);
    // needsNetworkMr is communicator-wide. If MR registration fails, every
    // rank must enter the same rollback collectives, even an ordinary-memory
    // rank with no local mapping or MR left to release.
    networkMrAttempted = true;
    FLAGCXCHECKGOTO(flagcxSymWindowEnsureNetworkMr(comm, d), res, fail);
  }

  *win = w;
  return flagcxSuccess;

fail:
  if (mcIpcSockOpen)
    flagcxIpcSocketClose(&mcIpcSock);
  if (ipcSockOpen)
    flagcxIpcSocketClose(&ipcSock);
  if (shareableFd >= 0 && (allFds == NULL || allFds[localRank] != shareableFd))
    close(shareableFd);
  if (allFds) {
    for (int i = 0; i < comm->localRanks; i++) {
      if (allFds[i] >= 0)
        close(allFds[i]);
    }
    free(allFds);
  }
  free(peerHandles);
  free(localDevices);
  // Move all provider objects into the window before rollback so a failed
  // release can be represented by the returned cleanup token.
  if (d != nullptr && physHandle != nullptr) {
    d->physHandle = physHandle;
    physHandle = nullptr;
  }
  if (d != nullptr && mcHandle != nullptr) {
    d->mcHandle = mcHandle;
    mcHandle = nullptr;
  }
  if (retainCleanupForRetry && d != nullptr) {
    d->state = flagcxSymWindowCleanupRequired;
    flagcxSymRetainPendingCleanup(comm, w);
    *win = w;
    d = nullptr;
    w = nullptr;
  } else if (!prepareRollbackConverged && d != nullptr &&
             (networkMrAttempted || d->hasNetworkMrRef ||
              d->hasPendingNetworkCleanup || d->flatMappingOwned ||
              d->flatVaOwned || d->lsaOffsetOwned || d->multicastMappingOwned ||
              d->multicastVaOwned || d->physHandle != nullptr ||
              d->mcHandle != nullptr)) {
    flagcxResult_t cleanup =
        flagcxSymCleanupUnpublished(comm, d, flagcxSymCleanupCollective);
    if (cleanup != flagcxSuccess) {
      WARN("symWindowRegister rollback retained for retry: res=%d", cleanup);
      d->state = flagcxSymWindowCleanupRequired;
      flagcxSymRetainPendingCleanup(comm, w);
      *win = w;
      d = nullptr;
      w = nullptr;
      res = cleanup;
    }
  }
  if (d != nullptr) {
    flagcxResult_t leaseCleanup = flagcxSymReleaseAllocationLease(d);
    if (leaseCleanup != flagcxSuccess) {
      d->state = flagcxSymWindowCleanupRequired;
      flagcxSymRetainPendingCleanup(comm, w);
      *win = w;
      d = nullptr;
      w = nullptr;
      res = leaseCleanup;
    }
  }
  if (physHandle != nullptr && deviceAdaptor->symPhysFree)
    deviceAdaptor->symPhysFree(physHandle);
  if (mcFd >= 0)
    close(mcFd);
  if (mcHandle != nullptr && deviceAdaptor->symMulticastFree)
    deviceAdaptor->symMulticastFree(mcHandle);
  flagcxSymFreeWindowState(d);
  free(w);
  return res;
}

flagcxResult_t flagcxSymWindowRegister(flagcxHeteroComm_t comm, void *buff,
                                       size_t size, flagcxWindow_t *win,
                                       int winFlags) {
  return flagcxSymWindowRegisterInternal(
      comm, buff, size, win, winFlags,
      flagcxOneSideRegistryRangeIsVmm(buff, size));
}

flagcxResult_t flagcxSymWindowDeregister(flagcxHeteroComm_t comm,
                                         flagcxWindow_t win,
                                         flagcxSymCleanupMode mode) {
  if (win == nullptr)
    return flagcxSuccess;

  flagcxSymWindow_t d = win->defaultBase;
  if (d != nullptr) {
    if (d->published && comm == nullptr)
      return flagcxInvalidArgument;
    d->state = flagcxSymWindowCleanupRequired;
    FLAGCXCHECK(flagcxSymCleanupMappingsConverged(comm, d, mode));
    FLAGCXCHECK(flagcxSymReleaseAllocationLease(d));

    // Only detach after all fallible cleanup has completed. A failure above
    // leaves both the public handle and list ownership intact for retry.
    if (d->published && comm != nullptr) {
      flagcxSymWindow_t *link = &comm->symWindows;
      while (*link != nullptr && *link != d)
        link = &(*link)->next;
      if (*link == d)
        *link = d->next;
      d->published = false;
    }

    if (comm != nullptr) {
      flagcxSymWindow_t *cleanupLink = &comm->pendingSymCleanup;
      while (*cleanupLink != nullptr && *cleanupLink != d)
        cleanupLink = &(*cleanupLink)->cleanupNext;
      if (*cleanupLink == d)
        *cleanupLink = d->cleanupNext;
    }

    flagcxSymFreeWindowState(d);
    win->defaultBase = nullptr;
  }

  free(win);
  return flagcxSuccess;
}

flagcxResult_t flagcxSymWindowPublish(flagcxHeteroComm_t comm,
                                      flagcxWindow_t win) {
  if (comm == nullptr || win == nullptr || win->defaultBase == nullptr)
    return flagcxInvalidArgument;
  flagcxSymWindow_t d = win->defaultBase;
  if (d->state == flagcxSymWindowCleanupRequired)
    return flagcxInvalidUsage;
  if (d->published)
    return flagcxSuccess;
  FLAGCXCHECK(flagcxSymWindowValidateDataRoutes(comm, d));
  flagcxSymWindowCommit(comm, win);
  return flagcxSuccess;
}

void flagcxSymWindowCommit(flagcxHeteroComm_t comm, flagcxWindow_t win) {
  flagcxSymWindow_t d = win->defaultBase;
  if (d->published)
    return;
  d->next = comm->symWindows;
  comm->symWindows = d;
  d->published = true;
  d->state = flagcxSymWindowPublished;
}

flagcxSymWindow_t flagcxSymWindowFind(flagcxHeteroComm_t comm, const void *ptr,
                                      size_t size, size_t *offset) {
  if (comm == nullptr || ptr == nullptr)
    return nullptr;

  uintptr_t address = (uintptr_t)ptr;
  for (flagcxSymWindow_t window = comm->symWindows; window != nullptr;
       window = window->next) {
    if (window->state != flagcxSymWindowPublished)
      continue;
    uintptr_t base = (uintptr_t)window->localBase;
    if (address < base)
      continue;
    size_t localOffset = (size_t)(address - base);
    if (localOffset > window->heapSize || size > window->heapSize - localOffset)
      continue;
    if (offset != nullptr)
      *offset = localOffset;
    return window;
  }
  return nullptr;
}

flagcxResult_t flagcxSymWindowPeerUserSize(flagcxHeteroComm_t comm,
                                           flagcxSymWindow_t window, int peer,
                                           size_t *userSize) {
  if (comm == nullptr || window == nullptr || userSize == nullptr || peer < 0 ||
      peer >= comm->nRanks)
    return flagcxInvalidArgument;
  *userSize = 0;
  if (peer == comm->rank) {
    *userSize = window->heapSize;
    return flagcxSuccess;
  }
  const bool localPeer =
      comm->rankToNode != nullptr && comm->rankToNode[peer] == comm->node;
  if (localPeer && window->rankLayouts != nullptr) {
    if (comm->rankToLocalRank == nullptr)
      return flagcxInternalError;
    const int localRank = comm->rankToLocalRank[peer];
    if (localRank < 0 || localRank >= window->localRanks)
      return flagcxInternalError;
    *userSize = window->rankLayouts[localRank].userSize;
    return flagcxSuccess;
  }
  if (flagcxSymWindowHasValidNetworkMr(comm, window)) {
    *userSize = comm->oneSideHandles[window->mrIndex]->regionSizes[peer];
    return *userSize != 0 ? flagcxSuccess : flagcxInvalidUsage;
  }
  if (localPeer) {
    *userSize = window->heapSize;
    return flagcxSuccess;
  }
  return flagcxNotSupported;
}

flagcxResult_t flagcxSymWindowResolveIpcPeerPtr(flagcxHeteroComm_t comm,
                                                flagcxSymWindow_t window,
                                                int peer, size_t offset,
                                                size_t size, void **ptr) {
  if (comm == nullptr || window == nullptr || ptr == nullptr || peer < 0 ||
      peer >= comm->nRanks)
    return flagcxInvalidArgument;
  *ptr = nullptr;
  if (comm->rankToNode == nullptr || comm->rankToLocalRank == nullptr ||
      comm->rankToNode[peer] != comm->node)
    return flagcxNotSupported;

  int peerLocalRank = comm->rankToLocalRank[peer];
  if (peerLocalRank < 0 || peerLocalRank >= window->localRanks)
    return flagcxInternalError;

  if (peer == comm->rank) {
    if (offset > window->heapSize || size > window->heapSize - offset)
      return flagcxInvalidArgument;
    *ptr = (void *)((uintptr_t)window->localBase + offset);
    return flagcxSuccess;
  }

  if (window->rankLayouts != nullptr && window->hasFlatMapping &&
      window->flatBase != nullptr) {
    if (window->resourcePath == flagcxSymResourceLsa) {
      if (comm->symLsaArena == nullptr || comm->symLsaArena->base == nullptr)
        return flagcxInternalError;
      return flagcxSymLsaPeerPtr(
          comm->symLsaArena->base, comm->symLsaArena->rankStride,
          window->lsaWindowOffset, window->localRanks, peerLocalRank,
          &window->rankLayouts[peerLocalRank], offset, size, ptr);
    }
    if (window->slotStride == 0)
      return flagcxInternalError;
    const flagcxSymRankLayout &layout = window->rankLayouts[peerLocalRank];
    if (offset > layout.userSize || size > layout.userSize - offset ||
        layout.userOffset > window->slotStride ||
        offset > window->slotStride - layout.userOffset ||
        size > window->slotStride - layout.userOffset - offset)
      return flagcxInvalidArgument;
    const uintptr_t slot = (uintptr_t)peerLocalRank * window->slotStride;
    const uintptr_t displacement = slot + layout.userOffset + offset;
    if (displacement >
        std::numeric_limits<uintptr_t>::max() - (uintptr_t)window->flatBase)
      return flagcxInvalidArgument;
    *ptr = (void *)((uintptr_t)window->flatBase + displacement);
    return flagcxSuccess;
  }

  if (offset > window->heapSize || size > window->heapSize - offset)
    return flagcxInvalidArgument;

  if (window->hasFlatMapping && window->flatBase != nullptr) {
    *ptr = (void *)((uintptr_t)window->flatBase +
                    (size_t)peerLocalRank * window->allocSize + offset);
    return flagcxSuccess;
  }

  int slot = window->ipcSlot;
  if (slot < 0 || comm->ipcTable == nullptr || slot >= comm->ipcTableSize)
    return flagcxNotSupported;
  struct flagcxIpcTableEntry *entry = &comm->ipcTable[slot];
  if (!entry->inUse || entry->hostPeerPtrs == nullptr ||
      peerLocalRank >= entry->nPeers ||
      entry->hostPeerPtrs[peerLocalRank] == nullptr)
    return flagcxNotSupported;

  *ptr = (void *)((uintptr_t)entry->hostPeerPtrs[peerLocalRank] + offset);
  return flagcxSuccess;
}
