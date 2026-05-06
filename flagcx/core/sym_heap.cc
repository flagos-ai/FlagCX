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
#include "ipcsocket.h"
#include "onesided.h"
#include "utils.h"
#include <cstdlib>
#include <cstring>

flagcxResult_t flagcxSymWindowRegister(flagcxHeteroComm_t comm, void *buff,
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

  int localRanks = comm->localRanks;
  int localRank = comm->localRank;
  d->localRanks = localRanks;
  d->heapSize = size;

  // ---- Try VMM path ----
  bool vmmOk = false;
  if (deviceAdaptor->symPhysAlloc != nullptr) {
    void *physHandle = nullptr;
    int shareableFd = -1;
    size_t handleSize = sizeof(int);
    size_t allocSize = 0;

    flagcxResult_t res = deviceAdaptor->symPhysAlloc(
        buff, size, &physHandle, &shareableFd, &handleSize, &allocSize);
    if (res == flagcxSuccess && physHandle != nullptr && allocSize > 0) {
      // Exchange shareable FDs with intra-node peers via Unix Domain Socket
      // (SCM_RIGHTS). Raw FD integers are process-local and cannot be sent
      // over TCP bootstrap — the kernel must duplicate them via sendmsg.
      int *allFds = (int *)malloc(localRanks * sizeof(int));
      if (allFds == nullptr) {
        deviceAdaptor->symPhysFree(physHandle);
        free(d);
        free(w);
        return flagcxSystemError;
      }
      allFds[localRank] = shareableFd;

      // Hash must be identical across all ranks — do not include buff
      // (device VA is process-local and may differ across peers)
      uint64_t ipcHash = comm->commHash ^ size;

      struct flagcxIpcSocket ipcSock;
      memset(&ipcSock, 0, sizeof(ipcSock));
      FLAGCXCHECK(
          flagcxIpcSocketInit(&ipcSock, comm->rank, ipcHash, /*block=*/1));

      // Barrier to ensure all sockets are created before sending
      struct bootstrapState *state = comm->bootstrap;
      for (int i = 0; i < localRanks; i++) {
        if (i == localRank)
          continue;
        int peerGlobalRank = comm->localRankToRank[i];
        int dummy = 1;
        FLAGCXCHECK(bootstrapSend(state, peerGlobalRank, /*tag=*/0x5932, &dummy,
                                  sizeof(dummy)));
      }
      for (int i = 0; i < localRanks; i++) {
        if (i == localRank)
          continue;
        int peerGlobalRank = comm->localRankToRank[i];
        int dummy = 0;
        FLAGCXCHECK(bootstrapRecv(state, peerGlobalRank, /*tag=*/0x5932, &dummy,
                                  sizeof(dummy)));
      }

      // Send our FD to each peer, tagging with our localRank in the header
      for (int i = 0; i < localRanks; i++) {
        if (i == localRank)
          continue;
        int peerGlobalRank = comm->localRankToRank[i];
        FLAGCXCHECK(flagcxIpcSocketSendMsg(&ipcSock, &localRank,
                                           sizeof(localRank), shareableFd,
                                           peerGlobalRank, ipcHash));
      }

      // Receive FDs from each peer — use header to identify sender
      int received = 0;
      int expected = localRanks - 1;
      while (received < expected) {
        int senderLocalRank = -1;
        int fd = -1;
        FLAGCXCHECK(flagcxIpcSocketRecvMsg(&ipcSock, &senderLocalRank,
                                           sizeof(senderLocalRank), &fd));
        allFds[senderLocalRank] = fd;
        received++;
      }

      flagcxIpcSocketClose(&ipcSock);

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
                                      physHandle, allocSize, &flatBase);
      if (res == flagcxSuccess && flatBase != nullptr) {
        // Build devPeerPtrs on device
        void **hostPeerPtrs = (void **)malloc(localRanks * sizeof(void *));
        if (hostPeerPtrs == nullptr) {
          deviceAdaptor->symFlatUnmap(flatBase, allocSize, localRanks);
          free(peerHandles);
          free(allFds);
          deviceAdaptor->symPhysFree(physHandle);
          free(d);
          free(w);
          return flagcxSystemError;
        }
        for (int i = 0; i < localRanks; i++) {
          hostPeerPtrs[i] = (char *)flatBase + (size_t)i * allocSize;
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
        d->allocSize = allocSize;
        d->isVMM = true;
        vmmOk = true;

        // Try multicast setup
        d->mcBase = nullptr;
        if (deviceAdaptor->symMulticastSetup != nullptr) {
          void *mcBase = nullptr;
          flagcxResult_t mcRes = deviceAdaptor->symMulticastSetup(
              physHandle, allocSize, localRanks, &mcBase);
          if (mcRes == flagcxSuccess && mcBase != nullptr) {
            d->mcBase = mcBase;
          }
        }
      } else {
        deviceAdaptor->symPhysFree(physHandle);
        physHandle = nullptr;
      }

      free(peerHandles);
      // Close received peer FDs (our own shareableFd stays open for physHandle)
      for (int i = 0; i < localRanks; i++) {
        if (i != localRank && allFds[i] >= 0)
          close(allFds[i]);
      }
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
    d->allocSize = 0;
  }

  // ---- Inter-node MR registration ----
  {
    flagcxResult_t regRes = flagcxOneSideRegister(comm, buff, size);
    if (regRes == flagcxSuccess) {
      for (int i = 0; i < comm->oneSideHandleCount; i++) {
        struct flagcxOneSideHandleInfo *info = comm->oneSideHandles[i];
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

flagcxResult_t flagcxSymWindowDeregister(flagcxHeteroComm_t comm,
                                         flagcxWindow_t win) {
  if (win == nullptr)
    return flagcxSuccess;

  flagcxSymWindow_t d = win->defaultBase;
  if (d != nullptr) {
    if (d->isVMM) {
      // Teardown multicast
      if (d->mcBase != nullptr &&
          deviceAdaptor->symMulticastTeardown != nullptr)
        deviceAdaptor->symMulticastTeardown(d->mcBase, d->allocSize);

      // Unmap flat VA
      if (d->flatBase != nullptr && deviceAdaptor->symFlatUnmap != nullptr)
        deviceAdaptor->symFlatUnmap(d->flatBase, d->allocSize, d->localRanks);

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
