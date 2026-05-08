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

  flagcxResult_t res = flagcxSuccess;
  flagcxWindow_t w = nullptr;
  flagcxSymWindow_t d = nullptr;
  int *allFds = nullptr;
  void **peerHandles = nullptr;
  void *physHandle = nullptr;
  int shareableFd = -1;
  bool ipcSockOpen = false;
  bool mcIpcSockOpen = false;
  struct flagcxIpcSocket ipcSock;
  struct flagcxIpcSocket mcIpcSock;
  memset(&ipcSock, 0, sizeof(ipcSock));
  memset(&mcIpcSock, 0, sizeof(mcIpcSock));

  FLAGCXCHECKGOTO(flagcxCalloc(&w, 1), res, fail);
  FLAGCXCHECKGOTO(flagcxCalloc(&d, 1), res, fail);

  w->vendorBase = nullptr;
  w->defaultBase = d;
  w->isSymmetricDefault = 1;

  d->mrIndex = -1;
  d->mrBase = 0;

  int localRanks;
  localRanks = comm->localRanks;
  int localRank;
  localRank = comm->localRank;
  d->localRanks = localRanks;
  d->heapSize = size;

  // ---- Try VMM path ----
  {
    bool vmmOk = false;
    if (deviceAdaptor->symPhysAlloc != nullptr) {
      size_t handleSize = sizeof(int);
      size_t allocSize = 0;

      flagcxResult_t allocRes = deviceAdaptor->symPhysAlloc(
          buff, size, &physHandle, &shareableFd, &handleSize, &allocSize);
      if (allocRes == flagcxSuccess && physHandle != nullptr && allocSize > 0) {
        // Exchange shareable FDs with intra-node peers via Unix Domain Socket
        FLAGCXCHECKGOTO(flagcxCalloc(&allFds, localRanks), res, fail);
        allFds[localRank] = shareableFd;

        // Hash must be identical across all ranks
        uint64_t ipcHash = comm->commHash ^ size;

        FLAGCXCHECKGOTO(
            flagcxIpcSocketInit(&ipcSock, comm->rank, ipcHash, /*block=*/1),
            res, fail);
        ipcSockOpen = true;

        // Barrier to ensure all sockets are created before sending
        struct bootstrapState *state = comm->bootstrap;
        FLAGCXCHECKGOTO(bootstrapIntraNodeBarrier(state, comm->localRankToRank,
                                                  localRank, localRanks,
                                                  /*tag=*/0x5932),
                        res, fail);

        // Send our FD to each peer
        for (int i = 0; i < localRanks; i++) {
          if (i == localRank)
            continue;
          int peerGlobalRank = comm->localRankToRank[i];
          FLAGCXCHECKGOTO(flagcxIpcSocketSendMsg(&ipcSock, &localRank,
                                                 sizeof(localRank), shareableFd,
                                                 peerGlobalRank, ipcHash),
                          res, fail);
        }

        // Receive FDs from each peer
        int received = 0;
        int expected = localRanks - 1;
        while (received < expected) {
          int senderLocalRank = -1;
          int fd = -1;
          FLAGCXCHECKGOTO(flagcxIpcSocketRecvMsg(&ipcSock, &senderLocalRank,
                                                 sizeof(senderLocalRank), &fd),
                          res, fail);
          allFds[senderLocalRank] = fd;
          received++;
        }

        flagcxIpcSocketClose(&ipcSock);
        ipcSockOpen = false;

        // Build peer handle pointers for symFlatMap
        FLAGCXCHECKGOTO(flagcxCalloc(&peerHandles, localRanks), res, fail);
        for (int i = 0; i < localRanks; i++) {
          peerHandles[i] = &allFds[i];
        }

        void *flatBase = nullptr;
        flagcxResult_t mapRes =
            deviceAdaptor->symFlatMap
                ? deviceAdaptor->symFlatMap(peerHandles, localRanks, localRank,
                                            physHandle, allocSize, &flatBase)
                : flagcxNotSupported;
        if (mapRes == flagcxSuccess && flatBase != nullptr) {
          d->flatBase = flatBase;
          d->physHandle = physHandle;
          d->allocSize = allocSize;
          d->isVMM = true;
          vmmOk = true;
          physHandle = nullptr; // ownership transferred to d

          // Try multicast setup
          d->mcBase = nullptr;
          int mcSupported = 0;
          if (deviceAdaptor->symMulticastSupported)
            deviceAdaptor->symMulticastSupported(&mcSupported);
          if (mcSupported) {
            void *mcHandle = nullptr;
            int mcFd = -1;

            if (localRank == 0) {
              flagcxResult_t mcRes =
                  deviceAdaptor->symMulticastCreate
                      ? deviceAdaptor->symMulticastCreate(allocSize, localRanks,
                                                          &mcHandle, &mcFd)
                      : flagcxNotSupported;
              if (mcRes != flagcxSuccess) {
                mcSupported = 0;
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

            if (mcSupported) {
              uint64_t mcIpcHash = ipcHash ^ 0x4D43; // "MC"

              FLAGCXCHECKGOTO(flagcxIpcSocketInit(&mcIpcSock, comm->rank,
                                                  mcIpcHash, /*block=*/1),
                              res, fail);
              mcIpcSockOpen = true;

              // Barrier: ensure all peers have created their IPC sockets
              FLAGCXCHECKGOTO(bootstrapIntraNodeBarrier(
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
              }

              flagcxIpcSocketClose(&mcIpcSock);
              mcIpcSockOpen = false;

              // All ranks: bind their physical allocation and map
              void *mcBaseVa = nullptr;
              size_t mcMapSize = 0;
              flagcxResult_t mcRes =
                  deviceAdaptor->symMulticastBind
                      ? deviceAdaptor->symMulticastBind(
                            (localRank == 0) ? mcHandle : nullptr, mcFd,
                            d->physHandle, allocSize, localRank, localRanks,
                            &mcBaseVa, &mcMapSize)
                      : flagcxNotSupported;
              if (mcRes == flagcxSuccess && mcBaseVa != nullptr) {
                d->mcBase = mcBaseVa;
                d->mcMapSize = mcMapSize;
              } else {
                WARN("symMulticastBind failed: res=%d mcBaseVa=%p "
                     "(localRank=%d)",
                     mcRes, mcBaseVa, localRank);
              }

              // Close the multicast FD
              if (mcFd >= 0)
                close(mcFd);
            }

            // Store mcHandle for cleanup (rank 0 only)
            if (localRank == 0 && mcHandle != nullptr) {
              d->mcHandle = mcHandle;
            }
          }
        } else {
          // symFlatMap failed — physHandle freed below in cleanup
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
      }
    }

    // ---- IPC fallback if VMM not available ----
    if (!vmmOk) {
      // Free physHandle if we still own it (symFlatMap failed or wasn't
      // attempted)
      if (physHandle != nullptr && deviceAdaptor->symPhysFree) {
        deviceAdaptor->symPhysFree(physHandle);
        physHandle = nullptr;
      }
      d->flatBase = nullptr;
      d->mcBase = nullptr;
      d->physHandle = nullptr;
      d->isVMM = false;
      d->allocSize = 0;
    }
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

fail:
  if (mcIpcSockOpen)
    flagcxIpcSocketClose(&mcIpcSock);
  if (ipcSockOpen)
    flagcxIpcSocketClose(&ipcSock);
  if (allFds) {
    for (int i = 0; i < comm->localRanks; i++) {
      if (allFds[i] >= 0)
        close(allFds[i]);
    }
    free(allFds);
  }
  free(peerHandles);
  if (physHandle != nullptr && deviceAdaptor->symPhysFree)
    deviceAdaptor->symPhysFree(physHandle);
  free(d);
  free(w);
  return res;
}

flagcxResult_t flagcxSymWindowDeregister(flagcxHeteroComm_t comm,
                                         flagcxWindow_t win) {
  if (win == nullptr)
    return flagcxSuccess;

  flagcxSymWindow_t d = win->defaultBase;
  if (d != nullptr) {
    if (d->isVMM) {
      // Teardown multicast
      if (d->mcBase != nullptr && deviceAdaptor->symMulticastTeardown)
        deviceAdaptor->symMulticastTeardown(d->mcBase, d->mcMapSize);

      // Release multicast handle (rank 0 only allocated it)
      if (d->mcHandle != nullptr && deviceAdaptor->symMulticastFree)
        deviceAdaptor->symMulticastFree(d->mcHandle);

      // Unmap flat VA
      if (d->flatBase != nullptr && deviceAdaptor->symFlatUnmap)
        deviceAdaptor->symFlatUnmap(d->flatBase, d->allocSize, d->localRanks);

      // Free physical handle
      if (d->physHandle != nullptr && deviceAdaptor->symPhysFree)
        deviceAdaptor->symPhysFree(d->physHandle);
    }

    free(d);
  }

  free(win);
  return flagcxSuccess;
}
