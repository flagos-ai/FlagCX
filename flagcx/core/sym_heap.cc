/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Symmetric memory coordination for the non-homo (fallback) path.
 * Implements VMM-based flat VA mapping with IPC fallback.
 ************************************************************************/

#include "sym_heap.h"
#include "adaptor.h"
#include "alloc.h"
#include "bootstrap.h"
#include "check.h"
#include "flagcx_sym_window_fallback.h"
#include "global_comm.h"
#include "onesided.h"
#include "param.h"
#include "utils.h"
#include <cstdlib>
#include <cstring>

// Default max heap size multiplier (4x initial size)
FLAGCX_PARAM(SymMaxHeapSize, "SYM_MAX_HEAP_SIZE", 0);

flagcxResult_t flagcxSymWindowRegister(flagcxComm_t comm, void *buff,
                                       size_t size, flagcxSymWindow_t *win,
                                       int winFlags) {
  if (comm == nullptr || buff == nullptr || size == 0 || win == nullptr)
    return flagcxInvalidArgument;

  flagcxSymWindow_t w =
      (flagcxSymWindow_t)calloc(1, sizeof(struct flagcxSymWindow));
  if (w == nullptr)
    return flagcxSystemError;

  w->winFlags = winFlags;
  w->mrIndex = -1;
  w->mrBase = 0;
  w->growthPhysHandles = nullptr;
  w->growthCount = 0;
  w->growthCapacity = 0;

  int localRanks = comm->localRanks;
  int localRank = comm->localRank;
  w->localRanks = localRanks;

  // Determine maxHeapSize
  size_t maxHeapSize = flagcxParamSymMaxHeapSize();
  if (maxHeapSize == 0)
    maxHeapSize = size * 4; // default: 4x initial
  if (maxHeapSize < size)
    maxHeapSize = size;
  w->heapSize = size;
  w->maxHeapSize = maxHeapSize;

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

        w->flatBase = flatBase;
        w->devPeerPtrs = devPeerPtrs;
        w->physHandle = physHandle;
        w->isVMM = true;
        w->isSymmetricFallback = true;
        vmmOk = true;

        // Try multicast setup
        w->mcBase = nullptr;
        if (deviceAdaptor->symMulticastSetup != nullptr) {
          void *mcBase = nullptr;
          flagcxResult_t mcRes = deviceAdaptor->symMulticastSetup(
              physHandle, size, localRanks, &mcBase);
          if (mcRes == flagcxSuccess && mcBase != nullptr) {
            w->mcBase = mcBase;
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
    w->flatBase = nullptr;
    w->mcBase = nullptr;
    w->physHandle = nullptr;
    w->isVMM = false;
    w->isSymmetricFallback = true;
    w->devPeerPtrs = nullptr;
    w->maxHeapSize = size; // no growth on IPC path
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
            w->mrIndex = i;
            w->mrBase = base;
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
                                         flagcxSymWindow_t win) {
  if (win == nullptr)
    return flagcxSuccess;

  if (win->isVMM) {
    // Free growth physHandles
    for (int i = 0; i < win->growthCount; i++) {
      if (win->growthPhysHandles[i] != nullptr)
        deviceAdaptor->symPhysFree(win->growthPhysHandles[i]);
    }
    free(win->growthPhysHandles);

    // Teardown multicast
    if (win->mcBase != nullptr &&
        deviceAdaptor->symMulticastTeardown != nullptr)
      deviceAdaptor->symMulticastTeardown(win->mcBase, win->maxHeapSize);

    // Unmap flat VA
    if (win->flatBase != nullptr && deviceAdaptor->symFlatUnmap != nullptr)
      deviceAdaptor->symFlatUnmap(win->flatBase, win->maxHeapSize,
                                  win->localRanks);

    // Free physical handle
    if (win->physHandle != nullptr && deviceAdaptor->symPhysFree != nullptr)
      deviceAdaptor->symPhysFree(win->physHandle);
  }

  // Free device peer pointers
  if (win->devPeerPtrs != nullptr)
    deviceAdaptor->deviceFree(win->devPeerPtrs, flagcxMemDevice, nullptr);

  free(win);
  return flagcxSuccess;
}

flagcxResult_t flagcxSymWindowGrow(flagcxComm_t comm, flagcxSymWindow_t win,
                                   void *newBuff, size_t newSize) {
  if (comm == nullptr || win == nullptr || newBuff == nullptr)
    return flagcxInvalidArgument;

  if (!win->isVMM)
    return flagcxNotSupported;

  if (newSize <= win->heapSize || newSize > win->maxHeapSize)
    return flagcxInvalidArgument;

  if (deviceAdaptor->symHeapGrow == nullptr)
    return flagcxNotSupported;

  size_t deltaSize = newSize - win->heapSize;
  int localRanks = win->localRanks;
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
  FLAGCXCHECK(deviceAdaptor->symHeapGrow(
      win->flatBase, peerNewHandles, localRanks, localRank, newPhysHandle,
      win->heapSize, newSize, win->maxHeapSize));

  free(peerNewHandles);
  free(allNewFds);

  // Grow multicast if available
  if (win->mcBase != nullptr && deviceAdaptor->symMulticastGrow != nullptr) {
    deviceAdaptor->symMulticastGrow(win->mcBase, newPhysHandle, win->heapSize,
                                    newSize);
  }

  // Track growth physHandle for cleanup
  if (win->growthCount >= win->growthCapacity) {
    int newCap = win->growthCapacity == 0 ? 4 : win->growthCapacity * 2;
    void **newArr =
        (void **)realloc(win->growthPhysHandles, newCap * sizeof(void *));
    if (newArr == nullptr) {
      deviceAdaptor->symPhysFree(newPhysHandle);
      return flagcxSystemError;
    }
    win->growthPhysHandles = newArr;
    win->growthCapacity = newCap;
  }
  win->growthPhysHandles[win->growthCount++] = newPhysHandle;

  win->heapSize = newSize;

  INFO(FLAGCX_INIT, "flagcxSymWindowGrow: grown to heapSize=%zu (max=%zu)",
       newSize, win->maxHeapSize);

  return flagcxSuccess;
}
