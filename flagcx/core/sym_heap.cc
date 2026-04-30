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
#include "global_comm.h"
#include "onesided.h"
#include "param.h"
#include "utils.h"
#include <cstdlib>
#include <cstring>

// Default max heap size multiplier (4x initial size)
FLAGCX_PARAM(SymMaxHeapSize, "SYM_MAX_HEAP_SIZE", 0);

flagcxResult_t flagcxSymWindowRegister(flagcxComm_t comm, void *buff,
                                       size_t size, flagcxWindow_t *win,
                                       int winFlags) {
  if (comm == nullptr || buff == nullptr || size == 0 || win == nullptr)
    return flagcxInvalidArgument;

  flagcxWindow_t w = (flagcxWindow_t)calloc(1, sizeof(struct flagcxWindow));
  if (w == nullptr)
    return flagcxSystemError;

  flagcxSymWindow_t d =
      (flagcxSymWindow_t)calloc(1, sizeof(struct flagcxSymWindow));
  if (d == nullptr) {
    free(w);
    return flagcxSystemError;
  }

  w->vendorBase = nullptr;
  w->defaultBase = d;
  w->isSymmetricDefault = true;

  d->mrIndex = -1;
  d->mrBase = 0;
  d->growthPhysHandles = nullptr;
  d->growthCount = 0;
  d->growthCapacity = 0;

  int localRanks = comm->localRanks;
  int localRank = comm->localRank;
  d->localRanks = localRanks;

  // Determine maxHeapSize
  size_t maxHeapSize = flagcxParamSymMaxHeapSize();
  if (maxHeapSize == 0)
    maxHeapSize = size * 4; // default: 4x initial
  if (maxHeapSize < size)
    maxHeapSize = size;
  d->heapSize = size;
  d->maxHeapSize = maxHeapSize;

  // ---- Try VMM path ----
  bool vmmOk = false;
  if (deviceAdaptor->symPhysAlloc != nullptr) {
    void *physHandle = nullptr;
    int shareableFd = -1;
    size_t handleSize = sizeof(int);

    flagcxResult_t res = deviceAdaptor->symPhysAlloc(buff, size, &physHandle,
                                                     &shareableFd, &handleSize);
    if (res == flagcxSuccess && physHandle != nullptr) {
      // Exchange shareable handles with intra-node peers
      int *allFds = (int *)malloc(localRanks * sizeof(int));
      if (allFds == nullptr) {
        deviceAdaptor->symPhysFree(physHandle);
        free(d);
        free(w);
        return flagcxSystemError;
      }
      allFds[localRank] = shareableFd;

      // Use intra-node allgather via bootstrap send/recv
      struct bootstrapState *state = comm->bootstrap;
      for (int i = 0; i < localRanks; i++) {
        if (i == localRank)
          continue;
        int peerGlobalRank = comm->localRankToRank[i];
        FLAGCXCHECK(bootstrapSend(state, peerGlobalRank, /*tag=*/0x5931,
                                  &shareableFd, sizeof(int)));
      }
      for (int i = 0; i < localRanks; i++) {
        if (i == localRank)
          continue;
        int peerGlobalRank = comm->localRankToRank[i];
        FLAGCXCHECK(bootstrapRecv(state, peerGlobalRank, /*tag=*/0x5931,
                                  &allFds[i], sizeof(int)));
      }

      // Build peer handle pointers for symFlatMap
      void **peerHandles = (void **)malloc(localRanks * sizeof(void *));
      if (peerHandles == nullptr) {
        free(allFds);
        deviceAdaptor->symPhysFree(physHandle);
        free(d);
        free(w);
        return flagcxSystemError;
      }
      for (int i = 0; i < localRanks; i++) {
        peerHandles[i] = &allFds[i];
      }

      void *flatBase = nullptr;
      res = deviceAdaptor->symFlatMap(peerHandles, localRanks, localRank,
                                      physHandle, size, maxHeapSize, &flatBase);
      if (res == flagcxSuccess && flatBase != nullptr) {
        // Build devPeerPtrs on device
        void **hostPeerPtrs = (void **)malloc(localRanks * sizeof(void *));
        if (hostPeerPtrs == nullptr) {
          deviceAdaptor->symFlatUnmap(flatBase, maxHeapSize, localRanks);
          free(peerHandles);
          free(allFds);
          deviceAdaptor->symPhysFree(physHandle);
          free(d);
          free(w);
          return flagcxSystemError;
        }
        for (int i = 0; i < localRanks; i++) {
          hostPeerPtrs[i] = (char *)flatBase + (size_t)i * maxHeapSize;
        }

        void **devPeerPtrs = nullptr;
        FLAGCXCHECK(deviceAdaptor->deviceMalloc((void **)&devPeerPtrs,
                                                localRanks * sizeof(void *),
                                                flagcxMemDevice, nullptr));
        FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
            devPeerPtrs, hostPeerPtrs, localRanks * sizeof(void *),
            flagcxMemcpyHostToDevice, nullptr, nullptr));
        free(hostPeerPtrs);

        d->flatBase = flatBase;
        d->devPeerPtrs = devPeerPtrs;
        d->physHandle = physHandle;
        d->isVMM = true;
        vmmOk = true;

        // Try multicast setup
        d->mcBase = nullptr;
        if (deviceAdaptor->symMulticastSetup != nullptr) {
          void *mcBase = nullptr;
          flagcxResult_t mcRes = deviceAdaptor->symMulticastSetup(
              physHandle, size, localRanks, &mcBase);
          if (mcRes == flagcxSuccess && mcBase != nullptr) {
            d->mcBase = mcBase;
          }
        }

        INFO(FLAGCX_INIT,
             "flagcxSymWindowRegister: VMM path OK, flatBase=%p "
             "heapSize=%zu maxHeapSize=%zu localRanks=%d",
             flatBase, size, maxHeapSize, localRanks);
      } else {
        deviceAdaptor->symPhysFree(physHandle);
        physHandle = nullptr;
      }

      free(peerHandles);
      free(allFds);
    }
  }

  // ---- IPC fallback if VMM not available ----
  if (!vmmOk) {
    d->flatBase = nullptr;
    d->mcBase = nullptr;
    d->physHandle = nullptr;
    d->isVMM = false;
    d->devPeerPtrs = nullptr;
    d->maxHeapSize = size; // no growth on IPC path
    INFO(FLAGCX_INIT,
         "flagcxSymWindowRegister: VMM not available, IPC fallback "
         "(devPeerPtrs will be set by flagcxDevMemCreate)");
  }

  // ---- Inter-node MR registration ----
  if (comm->heteroComm != nullptr) {
    flagcxResult_t regRes = flagcxOneSideRegister(comm, buff, size);
    if (regRes == flagcxSuccess) {
      struct flagcxHeteroComm *hc = comm->heteroComm;
      for (int i = 0; i < hc->oneSideHandleCount; i++) {
        struct flagcxOneSideHandleInfo *info = hc->oneSideHandles[i];
        if (info != nullptr && info->baseVas != nullptr) {
          uintptr_t base = info->baseVas[comm->rank];
          if ((uintptr_t)buff == base) {
            d->mrIndex = i;
            d->mrBase = base;
            break;
          }
        }
      }
    }
  }

  *win = w;
  return flagcxSuccess;
}

flagcxResult_t flagcxSymWindowDeregister(flagcxComm_t comm,
                                         flagcxWindow_t win) {
  if (win == nullptr)
    return flagcxSuccess;

  flagcxSymWindow_t d = win->defaultBase;
  if (d != nullptr) {
    if (d->isVMM) {
      // Free growth physHandles
      for (int i = 0; i < d->growthCount; i++) {
        if (d->growthPhysHandles[i] != nullptr)
          deviceAdaptor->symPhysFree(d->growthPhysHandles[i]);
      }
      free(d->growthPhysHandles);

      // Teardown multicast
      if (d->mcBase != nullptr &&
          deviceAdaptor->symMulticastTeardown != nullptr)
        deviceAdaptor->symMulticastTeardown(d->mcBase, d->maxHeapSize);

      // Unmap flat VA
      if (d->flatBase != nullptr && deviceAdaptor->symFlatUnmap != nullptr)
        deviceAdaptor->symFlatUnmap(d->flatBase, d->maxHeapSize, d->localRanks);

      // Free physical handle
      if (d->physHandle != nullptr && deviceAdaptor->symPhysFree != nullptr)
        deviceAdaptor->symPhysFree(d->physHandle);
    }

    // Free device peer pointers
    if (d->devPeerPtrs != nullptr)
      deviceAdaptor->deviceFree(d->devPeerPtrs, flagcxMemDevice, nullptr);

    free(d);
  }

  free(win);
  return flagcxSuccess;
}

flagcxResult_t flagcxSymWindowGrow(flagcxComm_t comm, flagcxWindow_t win,
                                   void *newBuff, size_t newSize) {
  if (comm == nullptr || win == nullptr || newBuff == nullptr)
    return flagcxInvalidArgument;

  flagcxSymWindow_t d = win->defaultBase;
  if (d == nullptr)
    return flagcxNotSupported;

  if (!d->isVMM)
    return flagcxNotSupported;

  if (newSize <= d->heapSize || newSize > d->maxHeapSize)
    return flagcxInvalidArgument;

  if (deviceAdaptor->symHeapGrow == nullptr)
    return flagcxNotSupported;

  size_t deltaSize = newSize - d->heapSize;
  int localRanks = d->localRanks;
  int localRank = comm->localRank;

  // Export new physical pages
  void *newPhysHandle = nullptr;
  int newFd = -1;
  size_t handleSize = sizeof(int);
  FLAGCXCHECK(deviceAdaptor->symPhysAlloc(newBuff, deltaSize, &newPhysHandle,
                                          &newFd, &handleSize));

  // Exchange new shareable handles
  int *allNewFds = (int *)malloc(localRanks * sizeof(int));
  if (allNewFds == nullptr) {
    deviceAdaptor->symPhysFree(newPhysHandle);
    return flagcxSystemError;
  }
  allNewFds[localRank] = newFd;

  struct bootstrapState *state = comm->bootstrap;
  for (int i = 0; i < localRanks; i++) {
    if (i == localRank)
      continue;
    int peerGlobalRank = comm->localRankToRank[i];
    FLAGCXCHECK(bootstrapSend(state, peerGlobalRank, /*tag=*/0x5932, &newFd,
                              sizeof(int)));
  }
  for (int i = 0; i < localRanks; i++) {
    if (i == localRank)
      continue;
    int peerGlobalRank = comm->localRankToRank[i];
    FLAGCXCHECK(bootstrapRecv(state, peerGlobalRank, /*tag=*/0x5932,
                              &allNewFds[i], sizeof(int)));
  }

  void **peerNewHandles = (void **)malloc(localRanks * sizeof(void *));
  if (peerNewHandles == nullptr) {
    free(allNewFds);
    deviceAdaptor->symPhysFree(newPhysHandle);
    return flagcxSystemError;
  }
  for (int i = 0; i < localRanks; i++) {
    peerNewHandles[i] = &allNewFds[i];
  }

  // Map new pages into existing flat VA
  FLAGCXCHECK(deviceAdaptor->symHeapGrow(d->flatBase, peerNewHandles,
                                         localRanks, localRank, newPhysHandle,
                                         d->heapSize, newSize, d->maxHeapSize));

  free(peerNewHandles);
  free(allNewFds);

  // Grow multicast if available
  if (d->mcBase != nullptr && deviceAdaptor->symMulticastGrow != nullptr) {
    deviceAdaptor->symMulticastGrow(d->mcBase, newPhysHandle, d->heapSize,
                                    newSize);
  }

  // Track growth physHandle for cleanup
  if (d->growthCount >= d->growthCapacity) {
    int newCap = d->growthCapacity == 0 ? 4 : d->growthCapacity * 2;
    void **newArr =
        (void **)realloc(d->growthPhysHandles, newCap * sizeof(void *));
    if (newArr == nullptr) {
      deviceAdaptor->symPhysFree(newPhysHandle);
      return flagcxSystemError;
    }
    d->growthPhysHandles = newArr;
    d->growthCapacity = newCap;
  }
  d->growthPhysHandles[d->growthCount++] = newPhysHandle;

  d->heapSize = newSize;

  INFO(FLAGCX_INIT, "flagcxSymWindowGrow: grown to heapSize=%zu (max=%zu)",
       newSize, d->maxHeapSize);

  return flagcxSuccess;
}
