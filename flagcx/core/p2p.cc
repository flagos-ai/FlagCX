#include "p2p.h"
#include "adaptor.h"
#include "comm.h"
#include "info.h"
#include "proxy.h"
#include "reg_pool.h"
#include <algorithm>
#include <map>
#include <string.h> // for memcpy

int64_t flagcxP2PBufferSize;
int64_t flagcxP2PChunkSize;

struct p2pIpcExpInfo {
  flagcxP2pIpcDesc ipcDesc;
  bool legacyIpcCap;
  int impFd;
  size_t size;
  uintptr_t offset;
};

static std::map<int, std::pair<int, int>>
    p2pOpHashMap; // <opHash, sendCounter, recvCounter>

void setP2pSlotInfo(int rank, int peerRank, size_t size, flagcxDataType_t dtype,
                    int isRecv, int *opHash, size_t *slotIdx) {
  // TODO: try a better hash function to reduce collisions
  int key = rank * 1000 + int(size >> 12) + dtype * 10 + peerRank * 100;
  int opHashCounter;
  auto it = p2pOpHashMap.find(key);
  if (it != p2pOpHashMap.end()) {
    if (isRecv) {
      opHashCounter = ++(it->second.second);
    } else {
      opHashCounter = ++(it->second.first);
    }
  } else {
    if (isRecv) {
      p2pOpHashMap[key] = std::make_pair(0, 1);
    } else {
      p2pOpHashMap[key] = std::make_pair(1, 0);
    }
    opHashCounter = 1;
  }
  // Ensure that opHash is unique for each operation
  *opHash = key + opHashCounter;
  // First half slots for send, second half for recv
  *slotIdx = (*opHash) % (FLAGCX_P2P_MAX_OPS / 2);
  if (isRecv) {
    *slotIdx += (FLAGCX_P2P_MAX_OPS / 2);
  }
}

flagcxResult_t flagcxP2pProxySend(struct flagcxP2pResources *resources,
                                  void *data, size_t size,
                                  struct flagcxProxyArgs *args) {
  // Avoid further processing slots if done
  if (args->done == 1)
    return flagcxSuccess;
  // Make sure data is valid
  if (!args->semaphore->pollStart())
    return flagcxSuccess;

  struct flagcxP2pSyncSlot *slotPtr =
      &resources->proxyInfo.shm->slots[args->p2pSlotIdx];
  struct flagcxP2pSyncSlot *peerSlotPtr =
      &resources->proxyInfo.shm->slots[args->p2pPeerSlotIdx];
  struct p2pRegInfo *regInfoPtr =
      &resources->proxyInfo.shm->regInfos[args->p2pSlotIdx];

  // Reset slot for new operation, only if previous operation
  // is done for both sides
  if (slotPtr->opHash == -1 && slotPtr->done == 1 && slotPtr->peerDone == 1) {
    slotPtr->opHash = args->p2pOpHash;
    slotPtr->done = 0;
    slotPtr->peerDone = 0;
    slotPtr->sendHead = 0;
    slotPtr->recvTail = FLAGCX_P2P_MAX_STEPS;
    // Reset reg info for new operation
    regInfoPtr->copyStarted = 0;
    regInfoPtr->copyDone = 0;
  }

  // Retry later since the slot is still in use
  if (slotPtr->opHash != args->p2pOpHash)
    return flagcxSuccess;

  // Retry later since the peer slot is still in use
  if (peerSlotPtr->opHash != args->p2pPeerOpHash && slotPtr->peerDone == 0)
    return flagcxSuccess;

  // Zero-copy mode: sender directly copies to receiver's buffer
  if (args->regBufFlag && args->p2pRmtAddr) {
    if (args->transmitted < args->chunkSteps) {
      // Single-step copy directly to receiver's buffer
      if (args->copied == 0) {
        regInfoPtr->copyStarted = 1;
        FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
            (void *)args->p2pRmtAddr, data, size, flagcxMemcpyDeviceToDevice,
            resources->proxyInfo.stream, NULL));
        FLAGCXCHECK(deviceAdaptor->eventRecord(resources->proxyInfo.events[0],
                                               resources->proxyInfo.stream));
        args->copied = args->chunkSteps; // Mark all chunks as copied
        args->totalCopySize = size;
      }

      // Check if copy is complete
      if (args->transmitted < args->copied) {
        flagcxResult_t res =
            deviceAdaptor->eventQuery(resources->proxyInfo.events[0]);
        if (res == flagcxSuccess) {
          args->transmitted = args->chunkSteps;
          regInfoPtr->copyDone = 1; // Signal to receiver that copy is done
        }
      }
    } else {
      // Cleanup phase
      if (args->done != 1) {
        if (slotPtr->done != 1) {
          if (peerSlotPtr->peerDone != 1) {
            peerSlotPtr->peerDone = 1;
          }
          if (slotPtr->peerDone == 1) {
            __atomic_store_n(&slotPtr->opHash, -1, __ATOMIC_RELAXED);
            __atomic_store_n(&slotPtr->done, 1, __ATOMIC_RELEASE);
            args->semaphore->signalCounter(1);
            if (deviceAsyncLoad && deviceAsyncStore) {
              if (args->deviceFuncRelaxedOrdering == 1) {
                FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
                    args->dlArgs, (void *)&args->hlArgs, sizeof(bool),
                    flagcxMemcpyHostToDevice, resources->proxyInfo.stream,
                    NULL));
              }
            }
            args->done = 1;
          }
        }
      }
    }
    return flagcxSuccess;
  }

  // Non-zero-copy mode: use FIFO buffer
  if (args->transmitted < args->chunkSteps) {
    if (args->copied < args->chunkSteps &&
        args->copied - args->transmitted < FLAGCX_P2P_MAX_STEPS) {
      int step = args->copied & args->sendStepMask;

      volatile uint64_t *recvTail = &peerSlotPtr->recvTail;

      if (*recvTail > args->copied) {
        args->subs[step].stepSize =
            std::min(args->chunkSize, size - args->totalCopySize);
        args->subs[step].stepBuff =
            resources->proxyInfo.recvFifo + (flagcxP2PChunkSize * step);

        FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
            args->subs[step].stepBuff, (char *)data + args->totalCopySize,
            args->subs[step].stepSize, flagcxMemcpyDeviceToDevice,
            resources->proxyInfo.stream, args->subs[step].copyArgs));
        FLAGCXCHECK(deviceAdaptor->eventRecord(
            resources->proxyInfo.events[step], resources->proxyInfo.stream));

        args->totalCopySize += args->subs[step].stepSize;
        args->copied++;
      }
    }

    if (args->transmitted < args->copied) {
      int step = args->transmitted & args->sendStepMask;
      flagcxResult_t res =
          deviceAdaptor->eventQuery(resources->proxyInfo.events[step]);

      if (res == flagcxSuccess) {
        args->transmitted++;
        // Update sendHead in the shared slot
        volatile uint64_t *sendHead = &slotPtr->sendHead;
        *sendHead = args->transmitted;
      }
    }
  } else {
    if (args->done != 1) {
      if (slotPtr->done != 1) {
        // Inform peer that this side is done
        if (peerSlotPtr->peerDone != 1) {
          peerSlotPtr->peerDone = 1;
        }
        // Signal done only when the peer side is also done
        if (slotPtr->peerDone == 1) {
          __atomic_store_n(&slotPtr->opHash, -1, __ATOMIC_RELAXED);
          __atomic_store_n(&slotPtr->done, 1, __ATOMIC_RELEASE);
          args->semaphore->signalCounter(1);
          if (deviceAsyncLoad && deviceAsyncStore) {
            if (args->deviceFuncRelaxedOrdering == 1) {
              FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
                  args->dlArgs, (void *)&args->hlArgs, sizeof(bool),
                  flagcxMemcpyHostToDevice, resources->proxyInfo.stream, NULL));
            }
          }
          args->done = 1;
        }
      }
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxP2pProxyRecv(struct flagcxP2pResources *resources,
                                  void *data, size_t size,
                                  struct flagcxProxyArgs *args) {
  // Avoid further processing slots if done
  if (args->done == 1)
    return flagcxSuccess;
  // Make sure data is valid
  if (!args->semaphore->pollStart())
    return flagcxSuccess;

  struct flagcxP2pSyncSlot *slotPtr =
      &resources->proxyInfo.shm->slots[args->p2pSlotIdx];
  struct flagcxP2pSyncSlot *peerSlotPtr =
      &resources->proxyInfo.shm->slots[args->p2pPeerSlotIdx];
  // For zero-copy, receiver checks sender's regInfo (using peerSlotIdx)
  struct p2pRegInfo *peerRegInfoPtr =
      &resources->proxyInfo.shm->regInfos[args->p2pPeerSlotIdx];

  // Reset slot for new operation, only if previous operation
  // is done for both sides
  if (slotPtr->opHash == -1 && slotPtr->done == 1 && slotPtr->peerDone == 1) {
    slotPtr->opHash = args->p2pOpHash;
    slotPtr->done = 0;
    slotPtr->peerDone = 0;
    slotPtr->sendHead = 0;
    slotPtr->recvTail = FLAGCX_P2P_MAX_STEPS;
  }

  // Return and retry later since the slot is still in use
  if (slotPtr->opHash != args->p2pOpHash)
    return flagcxSuccess;

  // Retry later since the peer slot is still in use
  if (peerSlotPtr->opHash != args->p2pPeerOpHash && slotPtr->peerDone == 0)
    return flagcxSuccess;

  // Zero-copy mode: receiver just waits for sender to complete the copy
  if (args->regBufFlag) {
    if (args->transmitted < args->chunkSteps) {
      // Wait for sender to signal copyDone
      if (peerRegInfoPtr->copyDone == 1) {
        args->copied = args->chunkSteps;
        args->transmitted = args->chunkSteps;
        args->totalCopySize = size;
      }
    } else {
      // Cleanup phase
      if (args->done != 1) {
        if (slotPtr->done != 1) {
          if (peerSlotPtr->peerDone != 1) {
            peerSlotPtr->peerDone = 1;
          }
          if (slotPtr->peerDone == 1) {
            __atomic_store_n(&slotPtr->opHash, -1, __ATOMIC_RELAXED);
            __atomic_store_n(&slotPtr->done, 1, __ATOMIC_RELEASE);
            args->semaphore->signalCounter(1);
            if (deviceAsyncLoad && deviceAsyncStore) {
              if (args->deviceFuncRelaxedOrdering == 1) {
                FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
                    args->dlArgs, (void *)&args->hlArgs, sizeof(bool),
                    flagcxMemcpyHostToDevice, resources->proxyInfo.stream,
                    NULL));
              }
            }
            args->done = 1;
          }
        }
      }
    }
    return flagcxSuccess;
  }

  // Non-zero-copy mode: use FIFO buffer
  if (args->transmitted < args->chunkSteps) {
    if (args->copied < args->chunkSteps &&
        args->copied - args->transmitted < FLAGCX_P2P_MAX_STEPS) {
      int step = args->copied & args->sendStepMask;
      volatile uint64_t *sendHead = &peerSlotPtr->sendHead;

      if (*sendHead > args->copied) {
        args->subs[step].stepSize =
            std::min(args->chunkSize, size - args->totalCopySize);
        args->subs[step].stepBuff =
            resources->proxyInfo.recvFifo + (flagcxP2PChunkSize * step);

        FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
            (char *)data + args->totalCopySize, args->subs[step].stepBuff,
            args->subs[step].stepSize, flagcxMemcpyDeviceToDevice,
            resources->proxyInfo.stream, args->subs[step].copyArgs));
        FLAGCXCHECK(deviceAdaptor->eventRecord(
            resources->proxyInfo.events[step], resources->proxyInfo.stream));

        args->totalCopySize += args->subs[step].stepSize;
        args->copied++;
      }
    }

    if (args->transmitted < args->copied) {
      int step = args->transmitted & args->sendStepMask;
      flagcxResult_t res =
          deviceAdaptor->eventQuery(resources->proxyInfo.events[step]);

      if (res == flagcxSuccess) {
        args->transmitted++;
        // Update recvTail in the shared slot
        volatile uint64_t *recvTail = &slotPtr->recvTail;
        *recvTail = args->transmitted + FLAGCX_P2P_MAX_STEPS;
      }
    }
  } else {
    if (args->done != 1) {
      if (slotPtr->done != 1) {
        // Inform peer that this side is done
        if (peerSlotPtr->peerDone != 1) {
          peerSlotPtr->peerDone = 1;
        }

        // Signal done only when the peer side is also done
        if (slotPtr->peerDone == 1) {
          __atomic_store_n(&slotPtr->opHash, -1, __ATOMIC_RELAXED);
          __atomic_store_n(&slotPtr->done, 1, __ATOMIC_RELEASE);
          args->semaphore->signalCounter(1);
          if (deviceAsyncLoad && deviceAsyncStore) {
            if (args->deviceFuncRelaxedOrdering == 1) {
              FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
                  args->dlArgs, (void *)&args->hlArgs, sizeof(bool),
                  flagcxMemcpyHostToDevice, resources->proxyInfo.stream, NULL));
            }
          }
          args->done = 1;
        }
      }
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxP2pProxySelfCopy(struct flagcxP2pResources *resources,
                                      void *sendData, void *recvData,
                                      size_t size,
                                      struct flagcxProxyArgs *args) {
  // Make sure data is valid
  if (!args->semaphore->pollStart())
    return flagcxSuccess;

  if (args->transmitted < args->chunkSteps) {
    // Perform single copy step
    if (args->copied < args->chunkSteps) {
      FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
          recvData, sendData, size, flagcxMemcpyDeviceToDevice,
          resources->proxyInfo.stream, NULL));
      FLAGCXCHECK(
          deviceAdaptor->eventRecord(resources->proxyInfo.events[args->copied],
                                     resources->proxyInfo.stream));
      args->copied++;
    }

    // Check for completed copy step
    if (args->transmitted < args->copied) {
      flagcxResult_t res = deviceAdaptor->eventQuery(
          resources->proxyInfo.events[args->transmitted]);
      if (res == flagcxSuccess) {
        args->transmitted++;
      }
    }
  } else {
    if (args->done != 1) {
      args->semaphore->signalCounter(1);
      // Deprecated device func handling
      if (deviceAsyncLoad && deviceAsyncStore) {
        if (args->deviceFuncRelaxedOrdering == 1) {
          FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
              args->dlArgs, (void *)&args->hlArgs, sizeof(bool),
              flagcxMemcpyHostToDevice, resources->proxyInfo.stream, NULL));
        }
      }
      args->done = 1;
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxP2pSendProxySetup(struct flagcxProxyConnection *connection,
                                       struct flagcxProxyState *proxyState,
                                       void *reqBuff, int reqSize,
                                       void *respBuff, int respSize,
                                       int *done) {
  if (respSize != sizeof(struct flagcxP2pShmProxyInfo))
    return flagcxInternalError;

  // Use the resources that was already allocated by transport.cc
  struct flagcxP2pResources *resources =
      (struct flagcxP2pResources *)connection->transportResources;
  if (resources == NULL) {
    WARN("flagcxP2pSendProxySetup: transportResources is NULL");
    return flagcxInternalError;
  }

  // Allocate shared memory and store in resources->proxyInfo
  size_t shmSize = sizeof(struct flagcxP2pShm);
  INFO(FLAGCX_P2P, "flagcxP2pSendProxySetup: Allocating shared memory size=%zu",
       shmSize);
  FLAGCXCHECK(flagcxShmAllocateShareableBuffer(
      shmSize, &resources->proxyInfo.desc, (void **)&resources->proxyInfo.shm,
      NULL));

  // Initialize all synchronization slots
  for (int i = 0; i < FLAGCX_P2P_MAX_OPS; i++) {
    resources->proxyInfo.shm->slots[i].sendHead = 0;
    resources->proxyInfo.shm->slots[i].recvTail = FLAGCX_P2P_MAX_STEPS;
    resources->proxyInfo.shm->slots[i].opHash = -1;
    resources->proxyInfo.shm->slots[i].done = 1;     // 1 = slot is free
    resources->proxyInfo.shm->slots[i].peerDone = 1; // 1 = slot is free
  }

  INFO(FLAGCX_P2P, "flagcxP2pSendProxySetup: Copying response, shm=%p",
       resources->proxyInfo.shm);
  memcpy(respBuff, &resources->proxyInfo, sizeof(struct flagcxP2pShmProxyInfo));
  *done = 1;

  INFO(FLAGCX_P2P, "flagcxP2pSendProxySetup: Completed successfully");
  return flagcxSuccess;
}

flagcxResult_t flagcxP2pRecvProxySetup(struct flagcxProxyConnection *connection,
                                       struct flagcxProxyState *proxyState,
                                       void *reqBuff, int reqSize,
                                       void *respBuff, int respSize,
                                       int *done) {
  INFO(FLAGCX_P2P,
       "flagcxP2pRecvProxySetup: reqSize=%d respSize=%d expectedReqSize=%zu "
       "expectedRespSize=%zu",
       reqSize, respSize, sizeof(struct flagcxP2pRequest),
       sizeof(struct flagcxP2pBuff));

  struct flagcxP2pRequest *req = (struct flagcxP2pRequest *)reqBuff;

  if (reqSize != sizeof(struct flagcxP2pRequest)) {
    WARN("flagcxP2pRecvProxySetup: Invalid reqSize %d, expected %zu", reqSize,
         sizeof(struct flagcxP2pRequest));
    return flagcxInternalError;
  }

  int size = req->size;
  if (respSize != sizeof(struct flagcxP2pBuff))
    return flagcxInternalError;
  struct flagcxP2pBuff *p2pBuff = (struct flagcxP2pBuff *)respBuff;
  FLAGCXCHECK(flagcxP2pAllocateShareableBuffer(
      size, req->refcount, &p2pBuff->ipcDesc, &p2pBuff->directPtr));
  p2pBuff->size = size;
  *done = 1;
  return flagcxSuccess;
}

flagcxResult_t
flagcxP2pSendProxyConnect(struct flagcxProxyConnection *connection,
                          struct flagcxProxyState *proxyState, void *reqBuff,
                          int reqSize, void *respBuff, int respSize,
                          int *done) {
  // Use the resources that was already allocated by transport.cc
  struct flagcxP2pResources *resources =
      (struct flagcxP2pResources *)connection->transportResources;

  if (resources == NULL) {
    WARN("flagcxP2pSendProxyConnect: transportResources is NULL");
    return flagcxInternalError;
  }

  // Recv sends recvFifo pointer to us
  if (reqSize != sizeof(void *)) {
    WARN("flagcxP2pSendProxyConnect: Invalid reqSize %d, expected %zu", reqSize,
         sizeof(void *));
    return flagcxInternalError;
  }

  resources->proxyInfo.recvFifo = *((char **)reqBuff);

  // Create stream and events for data transfers
  FLAGCXCHECK(deviceAdaptor->streamCreate(&resources->proxyInfo.stream));
  for (int i = 0; i < FLAGCX_P2P_MAX_STEPS; i++) {
    FLAGCXCHECK(deviceAdaptor->eventCreate(&resources->proxyInfo.events[i],
                                           flagcxEventDisableTiming));
  }

  *done = 1;
  INFO(FLAGCX_P2P, "flagcxP2pSendProxyConnect: Completed, recvFifo=%p",
       resources->proxyInfo.recvFifo);
  return flagcxSuccess;
}

flagcxResult_t
flagcxP2pRecvProxyConnect(struct flagcxProxyConnection *connection,
                          struct flagcxProxyState *proxyState, void *reqBuff,
                          int reqSize, void *respBuff, int respSize,
                          int *done) {
  // Use the resources that was already allocated by transport.cc
  struct flagcxP2pResources *resources =
      (struct flagcxP2pResources *)connection->transportResources;

  if (resources == NULL) {
    WARN("flagcxP2pRecvProxyConnect: transportResources is NULL");
    return flagcxInternalError;
  }

  // Create stream and events for data transfers
  FLAGCXCHECK(deviceAdaptor->streamCreate(&resources->proxyInfo.stream));
  for (int i = 0; i < FLAGCX_P2P_MAX_STEPS; i++) {
    FLAGCXCHECK(deviceAdaptor->eventCreate(&resources->proxyInfo.events[i],
                                           flagcxEventDisableTiming));
  }

  *done = 1;
  INFO(FLAGCX_P2P, "flagcxP2pRecvProxyConnect: Completed");
  return flagcxSuccess;
}

flagcxResult_t
flagcxP2pAllocateShareableBuffer(size_t size, int directMap,
                                 struct flagcxP2pIpcDesc *ipcDesc, void **ptr) {
  // 'directMap' parameter is reserved for future cuMem (direct mapping)
  FLAGCXCHECK(deviceAdaptor->deviceMalloc(ptr, size, flagcxMemDevice, NULL));
  size_t ipcSize = 0;
  flagcxIpcMemHandle_t handlePtr = NULL;
  flagcxResult_t res = deviceAdaptor->ipcMemHandleCreate(&handlePtr, &ipcSize);
  if (res != flagcxSuccess) {
    WARN("deviceAdaptor->ipcMemHandleCreate failed");
    deviceAdaptor->deviceFree(*ptr, flagcxMemDevice, NULL);
    *ptr = NULL;
    return res;
  }

  // Get the actual IPC handle data
  res = deviceAdaptor->ipcMemHandleGet(handlePtr, *ptr);
  if (res != flagcxSuccess) {
    WARN("deviceAdaptor->ipcMemHandleGet failed for ptr %p size %zu", *ptr,
         size);
    deviceAdaptor->ipcMemHandleFree(handlePtr);
    deviceAdaptor->deviceFree(*ptr, flagcxMemDevice, NULL);
    *ptr = NULL;
    return res;
  }
  memcpy(&ipcDesc->handleData, handlePtr, sizeof(flagcxIpcHandleData));
  ipcDesc->size = size;

  // Free the temporary handle wrapper
  deviceAdaptor->ipcMemHandleFree(handlePtr);
  return flagcxSuccess;
}

flagcxResult_t flagcxP2pImportShareableBuffer(struct flagcxHeteroComm *comm,
                                              int peer, size_t size,
                                              struct flagcxP2pIpcDesc *ipcDesc,
                                              void **devMemPtr) {
  *devMemPtr = NULL;

  // CRITICAL: Set device context before opening IPC handle
  FLAGCXCHECK(deviceAdaptor->setDevice(comm->cudaDev));
  flagcxIpcMemHandle_t handlePtr = (flagcxIpcMemHandle_t)&ipcDesc->handleData;

  flagcxResult_t res = deviceAdaptor->ipcMemHandleOpen(handlePtr, devMemPtr);
  if (res != flagcxSuccess) {
    WARN("Failed to open IPC handle for peer %d: error %d", peer, res);
    return res;
  }
  if (*devMemPtr == NULL) {
    WARN("IPC handle opened but devMemPtr is NULL for peer %d", peer);
    return flagcxInternalError;
  }
  INFO(FLAGCX_P2P,
       "Imported shareable buffer from peer %d device %d size %zu ptr %p", peer,
       comm->cudaDev, size, *devMemPtr);

  return flagcxSuccess;
}

static flagcxResult_t
p2pRegisterBuffer(flagcxHeteroComm *comm, const void *userbuff, size_t buffsize,
                  struct flagcxConnector **peerConns, int *peerRanks,
                  int nPeers, flagcxReg *regRecord, flagcxP2pRegisterMode mode,
                  int *regBufFlag, uintptr_t *offsetOut,
                  uintptr_t **peerRmtAddrsOut, bool *isLegacyIpc) {
  flagcxResult_t ret = flagcxSuccess;
  *regBufFlag = 0;
  *offsetOut = 0;
  *peerRmtAddrsOut = NULL;
  int legacyIpcCap = 0;
  uintptr_t baseAddr = 0;
  uintptr_t baseSize = 0;
  const bool allowRegister = (mode == flagcxP2pRegisterModeRegister);

  if (isLegacyIpc)
    *isLegacyIpc = false;
  flagcxRegItem *regItem =
      globalRegPool.getItem(comm, const_cast<void *>(userbuff));
  if (regRecord == NULL || regItem == NULL) {
    INFO(FLAGCX_REG,
         "p2pRegisterBuffer skip: regRecord=%p regItem=%p for buff %p size %zu",
         regRecord, regItem, userbuff, buffsize);
    return flagcxSuccess;
  }
  INFO(FLAGCX_REG,
       "p2pRegisterBuffer enter: rank %d buff %p size %zu regAddr %p "
       "handles=%zu peers=%d",
       comm ? comm->rank : -1, userbuff, buffsize, (void *)regRecord->addr,
       regItem->handles.size(), nPeers);

  for (int p = 0; p < nPeers; p++) {
    int peerRank = peerRanks[p];
    struct flagcxConnector *peerConn = peerConns[p];
    struct flagcxProxyConnector *proxyConn = &peerConn->proxyConn;

    flagcxIpcRegInfo *existingInfo = NULL;
    for (auto &handlePair : regItem->handles) {
      if (handlePair.second.proxyConn == proxyConn &&
          handlePair.second.handle) {
        existingInfo = (flagcxIpcRegInfo *)handlePair.second.handle;
        break;
      }
    }

    if (existingInfo && (!allowRegister)) {
      // Sender already has a reg record but may still need to import the handle
      if (existingInfo->handleReady && existingInfo->impInfo.rmtRegAddr) {
        *regBufFlag = 1;
        if (isLegacyIpc)
          *isLegacyIpc = existingInfo->impInfo.legacyIpcCap;
        INFO(FLAGCX_REG,
             "rank %d - P2P reuse buffer %p size %zu to peer %d regAddr %p "
             "current mode %d",
             comm->rank, userbuff, buffsize, peerRank,
             existingInfo->impInfo.rmtRegAddr, mode);
        continue;
      }
      // else: existingInfo exists but handleReady==false, need to do
      // bootstrapRecv below
    }

    if (existingInfo && allowRegister) {
      // Receiver already exported for this peer, nothing else to do
      *regBufFlag = 1;
      if (isLegacyIpc)
        *isLegacyIpc = existingInfo->impInfo.legacyIpcCap;
      INFO(FLAGCX_REG,
           "rank %d - P2P reuse buffer %p size %zu to peer %d regAddr %p",
           comm->rank, userbuff, buffsize, peerRank,
           existingInfo->impInfo.rmtRegAddr);
      continue;
    }

    // Either existingInfo is NULL, or it's sender with handleReady==false
    {
      struct flagcxIpcRegInfo *newInfo = NULL;

      if (!allowRegister) {
        // Lookup mode (sender): receive ipcInfo from peer, open handle
        INFO(
            FLAGCX_REG,
            "rank %d - IPC lookup buffer %p size %zu for peer %d via bootstrap",
            comm->rank, userbuff, buffsize, peerRank);

        struct p2pIpcExpInfo recvIpcInfo;
        memset(&recvIpcInfo, 0, sizeof(p2pIpcExpInfo));
        void *rmtRegAddr = NULL;

        // Receive ipcInfo from peer (tag = 4000 + comm->rank, receiver sends
        // using our rank)
        FLAGCXCHECKGOTO(bootstrapRecv(comm->bootstrap, peerRank,
                                      4000 + comm->rank, &recvIpcInfo,
                                      sizeof(p2pIpcExpInfo)),
                        ret, fail);

        // Open handle in our device context
        deviceAdaptor->setDevice(comm->cudaDev);
        if (recvIpcInfo.legacyIpcCap) {
          flagcxIpcMemHandle_t ipcHandle =
              (flagcxIpcMemHandle_t)&recvIpcInfo.ipcDesc.handleData;
          FLAGCXCHECKGOTO(
              deviceAdaptor->ipcMemHandleOpen(ipcHandle, &rmtRegAddr), ret,
              fail);
          if (rmtRegAddr) {
            rmtRegAddr = (void *)((uintptr_t)rmtRegAddr + recvIpcInfo.offset);
          }
        } else {
          WARN("rank %d - Non-legacy IPC not fully implemented yet for peer %d",
               comm->rank, peerRank);
          goto fail;
        }

        if (!existingInfo) {
          newInfo = (flagcxIpcRegInfo *)calloc(1, sizeof(flagcxIpcRegInfo));
          if (newInfo == NULL) {
            WARN("Failed to allocate IPC registration info");
            goto fail;
          }
          newInfo->peerRank = peerRank;
          newInfo->baseAddr = NULL;
          newInfo->ipcProxyconn = NULL; // Not using proxy for P2P IPC
          FLAGCXCHECKGOTO(
              globalRegPool.addP2pHandle(comm, regItem, newInfo, proxyConn),
              ret, fail);
          existingInfo = newInfo;
        }

        if (rmtRegAddr) {
          existingInfo->impInfo.rmtRegAddr = rmtRegAddr;
          existingInfo->impInfo.offset = recvIpcInfo.offset;
          existingInfo->impInfo.legacyIpcCap = recvIpcInfo.legacyIpcCap;
          existingInfo->handleReady = true;
          regRecord->state |= IPC_REG_COMPLETE;
          *regBufFlag = 1;
          INFO(FLAGCX_REG,
               "rank %d - IPC lookup completed buffer %p size %zu for peer %d "
               "regAddr %p",
               comm->rank, userbuff, buffsize, peerRank, rmtRegAddr);
        }
        continue;
      }

      // Register mode (receiver): get IPC handle and send to peer
      struct p2pIpcExpInfo ipcInfo;
      memset(&ipcInfo, 0, sizeof(p2pIpcExpInfo));

      if (baseAddr == 0) {
        uintptr_t beginAddr = 0;
        uintptr_t endAddr = 0;
        if (regRecord->baseAddr && regRecord->baseSize) {
          beginAddr = regRecord->baseAddr;
          endAddr = regRecord->baseAddr + regRecord->baseSize;
        } else {
          globalRegPool.getPagedAddr(const_cast<void *>(userbuff), buffsize,
                                     &beginAddr, &endAddr);
        }
        baseAddr = beginAddr;
        baseSize = endAddr - beginAddr;
        legacyIpcCap = 1;
        INFO(FLAGCX_REG,
             "rank %d - computed register range base=%p size=%zu user=%p "
             "regAddr=%p",
             comm->rank, (void *)baseAddr, (size_t)baseSize, userbuff,
             (void *)regRecord->addr);
      }

      if (legacyIpcCap) {
        // Get IPC handle
        flagcxIpcMemHandle_t ipcHandle = NULL;
        size_t handleSize = 0;
        FLAGCXCHECKGOTO(
            deviceAdaptor->ipcMemHandleCreate(&ipcHandle, &handleSize), ret,
            fail);
        FLAGCXCHECKGOTO(
            deviceAdaptor->ipcMemHandleGet(ipcHandle, (void *)baseAddr), ret,
            fail);
        if (handleSize <= sizeof(flagcxIpcHandleData)) {
          memcpy(&ipcInfo.ipcDesc.handleData, ipcHandle, handleSize);
        }
        deviceAdaptor->ipcMemHandleFree(ipcHandle);

        ipcInfo.legacyIpcCap = true;
        if (isLegacyIpc)
          *isLegacyIpc = true;
      } else {
        WARN("rank %d - Non-legacy IPC not fully implemented yet for peer %d",
             comm->rank, peerRank);
        goto fail;
      }

      ipcInfo.size = (size_t)baseSize;
      ipcInfo.offset = regRecord->addr - baseAddr;

      // Send ipcInfo to sender via bootstrap
      INFO(FLAGCX_REG,
           "rank %d - IPC registering buffer %p size %zu (baseAddr %p size "
           "%zu) to peer %d via bootstrap",
           comm->rank, userbuff, buffsize, (void *)regRecord->addr,
           ipcInfo.size, peerRank);
      // Tag = 4000 + peerRank (sender will recv using its own rank)
      FLAGCXCHECKGOTO(bootstrapSend(comm->bootstrap, peerRank, 4000 + peerRank,
                                    &ipcInfo, sizeof(p2pIpcExpInfo)),
                      ret, fail);

      // Save receiver's buffer info
      if (!existingInfo) {
        newInfo = (flagcxIpcRegInfo *)calloc(1, sizeof(flagcxIpcRegInfo));
      } else {
        newInfo = existingInfo;
      }
      if (newInfo == NULL) {
        WARN("Failed to allocate IPC registration info");
        goto fail;
      }

      regRecord->state |= IPC_REG_COMPLETE;
      newInfo->peerRank = peerRank;
      newInfo->baseAddr = (void *)baseAddr;
      newInfo->impInfo.rmtRegAddr = NULL; // Receiver waits for sender import
      newInfo->impInfo.offset = ipcInfo.offset;
      newInfo->impInfo.legacyIpcCap = ipcInfo.legacyIpcCap;
      newInfo->ipcProxyconn = NULL; // Not using proxy for P2P IPC
      newInfo->handleReady = false;
      FLAGCXCHECKGOTO(
          globalRegPool.addP2pHandle(comm, regItem, newInfo, proxyConn), ret,
          fail);

      *regBufFlag = 1;

      INFO(FLAGCX_REG,
           "rank %d - IPC registered buffer %p size %zu (baseAddr %p) to peer "
           "%d",
           comm->rank, userbuff, buffsize, (void *)regRecord->addr, peerRank);
    }
  }

  if (*regBufFlag) {
    assert(nPeers == 1);
    // p2p always returns remote addr here since remote buffer addr is passed in
    // device work struct
    struct flagcxProxyConnector *targetProxyConn = &peerConns[0]->proxyConn;
    for (auto &handlePair : regItem->handles) {
      if (handlePair.second.proxyConn == targetProxyConn &&
          handlePair.second.handle) {
        flagcxIpcRegInfo *info = (flagcxIpcRegInfo *)handlePair.second.handle;
        *peerRmtAddrsOut = (uintptr_t *)info->impInfo.rmtRegAddr;
        INFO(FLAGCX_REG,
             "rank %d - returning remote addr %p offset %zu for buff %p",
             comm ? comm->rank : -1, info->impInfo.rmtRegAddr,
             (uintptr_t)userbuff - regRecord->addr, userbuff);
        break;
      }
    }
    *offsetOut = (uintptr_t)userbuff - regRecord->addr;
  }

  return flagcxSuccess;

fail:
  return ret;
}

flagcxResult_t flagcxP2pRegisterBuffer(struct flagcxHeteroComm *comm,
                                       const void *userbuff, size_t buffSize,
                                       struct flagcxConnector **peerConns,
                                       int *peerRanks, int nPeers,
                                       flagcxP2pRegisterMode mode,
                                       int *regBufFlag, uintptr_t *offsetOut,
                                       uintptr_t **peerRmtAddrsOut) {
  flagcxReg tempReg = {};
  struct flagcxReg *regRecord = NULL;
  *regBufFlag = 0;
  *offsetOut = 0;
  *peerRmtAddrsOut = NULL;
  if (comm && userbuff && buffSize > 0 && nPeers > 0) {
    INFO(FLAGCX_REG,
         "flagcxP2pRegisterBuffer enter: comm=%p rank=%d buff=%p size=%zu "
         "nPeers=%d mode=%d",
         comm, comm->rank, userbuff, buffSize, nPeers, (int)mode);
    flagcxRegItem *regItem =
        globalRegPool.getItem(comm, const_cast<void *>(userbuff));
    if (regItem != NULL) {
      tempReg.addr = regItem->beginAddr;
      tempReg.baseAddr = regItem->beginAddr;
      tempReg.baseSize = regItem->endAddr - regItem->beginAddr;
      tempReg.regSize = tempReg.baseSize;
      regRecord = &tempReg;
    } else {
      INFO(FLAGCX_REG,
           "flagcxP2pRegisterBuffer: no regItem for buff %p size %zu", userbuff,
           buffSize);
    }
    FLAGCXCHECK(p2pRegisterBuffer(
        comm, userbuff, buffSize, peerConns, peerRanks, nPeers, regRecord, mode,
        regBufFlag, offsetOut, peerRmtAddrsOut, NULL));
    INFO(FLAGCX_REG,
         "flagcxP2pRegisterBuffer exit: buff=%p regBufFlag=%d offset=%zu "
         "peerAddr=%p",
         userbuff, *regBufFlag, *offsetOut,
         peerRmtAddrsOut && *peerRmtAddrsOut ? *peerRmtAddrsOut : NULL);
  } else {
    INFO(FLAGCX_REG,
         "flagcxP2pRegisterBuffer skip: comm=%p buff=%p size=%zu nPeers=%d",
         comm, userbuff, buffSize, nPeers);
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxP2pDeregisterBuffer(struct flagcxHeteroComm *comm,
                                         flagcxIpcRegInfo *info) {
  if (comm == NULL || info == NULL) {
    return flagcxSuccess;
  }
  INFO(FLAGCX_REG,
       "P2P deregister buffer: comm=%p peerRank=%d rmtRegAddr=%p offset=%zu "
       "legacyIpcCap=%d",
       comm, info->peerRank, info->impInfo.rmtRegAddr, info->impInfo.offset,
       info->impInfo.legacyIpcCap);

  // Close IPC handle if opened (sender side)
  if (info->impInfo.rmtRegAddr && info->impInfo.legacyIpcCap) {
    // Need to close the IPC memory handle that was opened
    void *baseAddr =
        (void *)((uintptr_t)info->impInfo.rmtRegAddr - info->impInfo.offset);
    deviceAdaptor->ipcMemHandleClose(baseAddr);
    INFO(FLAGCX_REG,
         "P2P deregister: closed IPC handle for rmtRegAddr=%p baseAddr=%p",
         info->impInfo.rmtRegAddr, baseAddr);
  }
  free(info);

  return flagcxSuccess;
}

/*
  If support inter-process P2P via proxy, implement these functions
*/
// flagcxResult_t flagcxP2pProxyRegister(struct flagcxProxyConnection*
// connection,
//                                       struct flagcxProxyState* proxyState,
//                                       void* reqBuff, int reqSize,
//                                       void* respBuff, int respSize, int*
//                                       done) {
//   struct p2pIpcExpInfo* ipcExpInfo = (struct p2pIpcExpInfo*)reqBuff;
//   void* regAddr = NULL;
//   flagcxResult_t ret = flagcxSuccess;

//   if (proxyState == NULL) {
//     WARN("Proxy register missing state context");
//     *done = 1;
//     return flagcxInvalidArgument;
//   }
//   INFO(FLAGCX_REG, "Proxy rank %d register reqBuff %p size %ld offset %ld
//   legacyIpcCap %d sameProcess %d", proxyState->cudaDev, reqBuff,
//   ipcExpInfo->size, ipcExpInfo->offset, ipcExpInfo->legacyIpcCap,
//   connection->sameProcess);
//   FLAGCXCHECKGOTO(deviceAdaptor->setDevice(proxyState->cudaDev), ret, exit);

//   if (sizeof(struct p2pIpcExpInfo) != reqSize) {
//     WARN("Invalid request size for P2P proxy register: expected %zu, got %d",
//          sizeof(struct p2pIpcExpInfo), reqSize);
//     *done = 1;
//     return flagcxInvalidArgument;
//   }

//   if (sizeof(void*) != respSize) {
//     WARN("Invalid response size for P2P proxy register: expected %zu, got
//     %d",
//          sizeof(void*), respSize);
//     *done = 1;
//     return flagcxInvalidArgument;
//   }

//   // Request peer passes all necessary buffer info to import. The proxy
//   thread would register
//   // the buffer locally and return register addr back
//   if (ipcExpInfo->legacyIpcCap) {
//     if (connection->sameProcess) {
//       void *baseAddr = NULL;
//       memcpy(&baseAddr, &ipcExpInfo->ipcDesc.handleData, sizeof(void *));
//       regAddr = (void *)((uintptr_t)baseAddr + ipcExpInfo->offset);
//     } else {
//       // Legacy CUDA IPC import
//       flagcxIpcMemHandle_t ipcHandle =
//           (flagcxIpcMemHandle_t)&ipcExpInfo->ipcDesc.handleData;

//       flagcxResult_t openRes =
//           deviceAdaptor->ipcMemHandleOpen(ipcHandle, &regAddr);
//       if (openRes != flagcxSuccess) {
//         WARN("ipcMemHandleOpen failed: res=%d size=%zu offset=%zu
//         legacyIpc=%d",
//              static_cast<int>(openRes), ipcExpInfo->size, ipcExpInfo->offset,
//              ipcExpInfo->legacyIpcCap);
//         ret = openRes;
//         goto fail;
//       }
//       if (regAddr == NULL) {
//         WARN("ipcMemHandleOpen returned NULL ptr size=%zu offset=%zu "
//              "legacyIpc=%d",
//              ipcExpInfo->size, ipcExpInfo->offset, ipcExpInfo->legacyIpcCap);
//         goto fail;
//       }
//       regAddr = (void *)((uintptr_t)regAddr + ipcExpInfo->offset);
//     }
//   } else {
//     // cuMem or advanced IPC import not fully supported yet
//     WARN("Non-legacy IPC import not implemented in proxy");
//     goto fail;
//   }
//   INFO(FLAGCX_REG, "Proxy register success regAddr %p size %zu offset %zu
//   legacyIpcCap %d sameProcess %d",
//        regAddr, ipcExpInfo->size, ipcExpInfo->offset,
//        ipcExpInfo->legacyIpcCap, connection->sameProcess);

// exit:
//   memcpy(respBuff, (void*)&regAddr, sizeof(void*));
//   *done = 1;
//   return ret;

// fail:
//   regAddr = NULL;
//   goto exit;
// }

// flagcxResult_t flagcxP2pProxyDeregister(struct flagcxProxyConnection*
// connection,
//   void* reqBuff, int reqSize, int* done) {
//                                           // struct flagcxProxyState*
//                                           proxyState,
//   flagcxResult_t ret = flagcxSuccess;
//   struct flagcxIpcImpInfo* ipcInfo = (struct flagcxIpcImpInfo*)reqBuff;

//   // if (proxyState == NULL) {
//   //   WARN("Proxy deregister missing state context");
//   //   *done = 1;
//   //   return flagcxInvalidArgument;
//   // }
//   // deviceAdaptor->setDevice(proxyState->cudaDev);

//   if (sizeof(struct flagcxIpcImpInfo) != reqSize) {
//     WARN("Invalid request size for P2P proxy deregister: expected %zu, got
//     %d",
//          sizeof(struct flagcxIpcImpInfo), reqSize);
//     *done = 1;
//     return flagcxInvalidArgument;
//   }

//   void* baseAddr = (void*)((uintptr_t)ipcInfo->rmtRegAddr - ipcInfo->offset);

//   if (ipcInfo->legacyIpcCap) {
//     // Legacy CUDA IPC close
//     FLAGCXCHECKGOTO(deviceAdaptor->ipcMemHandleClose(baseAddr), ret, fail);
//   } else {
//     // cuMem or advanced IPC deallocation not fully supported yet
//     WARN("Non-legacy IPC deregister not implemented in proxy");
//     goto fail;
//   }
// exit:
//   *done = 1;
//   return ret;

// fail:
//   goto exit;
// }

flagcxResult_t flagcxP2pSendProxyFree(struct flagcxP2pResources *resources) {
  if (resources == NULL)
    return flagcxSuccess;

  for (int s = 0; s < FLAGCX_P2P_MAX_STEPS; s++) {
    if (resources->proxyInfo.events[s] != NULL) {
      FLAGCXCHECK(deviceAdaptor->eventDestroy(resources->proxyInfo.events[s]));
    }
  }

  if (resources->proxyInfo.stream != NULL) {
    FLAGCXCHECK(deviceAdaptor->streamDestroy(resources->proxyInfo.stream));
  }

  if (resources->proxyInfo.shm != NULL) {
    FLAGCXCHECK(flagcxShmIpcClose(&resources->proxyInfo.desc));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxP2pRecvProxyFree(struct flagcxP2pResources *resources) {
  if (resources == NULL)
    return flagcxSuccess;

  // Destroy events
  for (int s = 0; s < FLAGCX_P2P_MAX_STEPS; s++) {
    if (resources->proxyInfo.events[s] != NULL) {
      FLAGCXCHECK(deviceAdaptor->eventDestroy(resources->proxyInfo.events[s]));
    }
  }

  // Destroy stream
  if (resources->proxyInfo.stream != NULL) {
    FLAGCXCHECK(deviceAdaptor->streamDestroy(resources->proxyInfo.stream));
  }
  return flagcxSuccess;
}