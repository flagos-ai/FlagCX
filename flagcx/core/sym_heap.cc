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

        // Try multicast setup: rank 0 creates, broadcasts FD to peers
        d->mcBase = nullptr;
        int mcSupported = 0;
        deviceAdaptor->symMulticastSupported(&mcSupported);
        if (mcSupported) {
          void *mcHandle = nullptr;
          int mcFd = -1;

          if (localRank == 0) {
            // Rank 0 creates the multicast object and gets the shareable FD
            flagcxResult_t mcRes = deviceAdaptor->symMulticastCreate(
                allocSize, localRanks, &mcHandle, &mcFd);
            if (mcRes != flagcxSuccess) {
              mcSupported = 0; // Fall through without multicast
            }
          }

          // Broadcast success/failure from rank 0 to all peers
          struct bootstrapState *mcState = comm->bootstrap;
          if (localRank == 0) {
            for (int i = 1; i < localRanks; i++) {
              int peerGlobalRank = comm->localRankToRank[i];
              FLAGCXCHECK(bootstrapSend(mcState, peerGlobalRank, /*tag=*/0x5933,
                                        &mcSupported, sizeof(mcSupported)));
            }
          } else {
            int rank0Global = comm->localRankToRank[0];
            FLAGCXCHECK(bootstrapRecv(mcState, rank0Global, /*tag=*/0x5933,
                                      &mcSupported, sizeof(mcSupported)));
          }

          if (mcSupported) {
            // Exchange multicast FD: rank 0 sends to all peers via IPC socket
            uint64_t mcIpcHash = ipcHash ^ 0x4D43; // "MC"
            if (localRank == 0) {
              // Send mcFd to each peer
              for (int i = 1; i < localRanks; i++) {
                int peerGlobalRank = comm->localRankToRank[i];
                int tag = 0; // rank 0 is always the sender
                FLAGCXCHECK(flagcxIpcSocketSendMsg(&ipcSock, &tag, sizeof(tag),
                                                   mcFd, peerGlobalRank,
                                                   mcIpcHash));
              }
            } else {
              // Receive mcFd from rank 0
              int tag = -1;
              FLAGCXCHECK(
                  flagcxIpcSocketRecvMsg(&ipcSock, &tag, sizeof(tag), &mcFd));
            }

            // All ranks: bind their physical allocation and map
            void *mcBaseVa = nullptr;
            flagcxResult_t mcRes = deviceAdaptor->symMulticastBind(
                (localRank == 0) ? mcHandle : nullptr, mcFd, physHandle,
                allocSize, localRank, localRanks, &mcBaseVa);
            if (mcRes == flagcxSuccess && mcBaseVa != nullptr) {
              d->mcBase = mcBaseVa;
            }

            // Close imported FD (rank 0's FD was already consumed by export)
            if (localRank != 0 && mcFd >= 0)
              close(mcFd);
          }

          // Store mcHandle for cleanup (rank 0 only)
          if (localRank == 0 && mcHandle != nullptr) {
            d->mcHandle = mcHandle;
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
      if (d->mcBase != nullptr)
        deviceAdaptor->symMulticastTeardown(d->mcBase, d->allocSize);

      // Free multicast handle (rank 0 only allocated it)
      if (d->mcHandle != nullptr)
        free(d->mcHandle);

      // Unmap flat VA
      if (d->flatBase != nullptr)
        deviceAdaptor->symFlatUnmap(d->flatBase, d->allocSize, d->localRanks);

      // Free physical handle
      if (d->physHandle != nullptr)
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
