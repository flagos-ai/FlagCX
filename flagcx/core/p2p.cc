#include "p2p.h"
#include "adaptor.h"
#include "info.h"
#include <algorithm>
#include <map>
#include <string.h> // for memcpy

int64_t flagcxP2PBufferSize;
int64_t flagcxP2PChunkSize;

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

  // Reset slot for new operation, only if previous operation
  // is done for both sides
  if (slotPtr->opHash == -1 && slotPtr->done == 1 && slotPtr->peerDone == 1) {
    slotPtr->opHash = args->p2pOpHash;
    slotPtr->done = 0;
    slotPtr->peerDone = 0;
    slotPtr->sendHead = 0;
    slotPtr->recvTail = FLAGCX_P2P_MAX_STEPS;
  }

  // Retry later since the slot is still in use
  if (slotPtr->opHash != args->p2pOpHash)
    return flagcxSuccess;

  // Retry later since the peer slot is still in use
  if (peerSlotPtr->opHash != args->p2pPeerOpHash && slotPtr->peerDone == 0)
    return flagcxSuccess;

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