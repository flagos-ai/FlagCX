#include "flagcx_hetero.h"
#include "adaptor.h"
#include "group.h"
#include "net.h"
#include "onesided.h"
#include "transport.h"
#include "type.h"

#include <climits>
#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Async RMA proxy implementation
// ---------------------------------------------------------------------------

static flagcxResult_t allocRmaDesc(struct flagcxRmaDesc **out) {
  struct flagcxRmaDesc *d =
      (struct flagcxRmaDesc *)calloc(1, sizeof(struct flagcxRmaDesc));
  if (d == NULL) {
    WARN("allocRmaDesc: out of memory");
    return flagcxSystemError;
  }
  *out = d;
  return flagcxSuccess;
}

static void enqueuePending(struct flagcxRmaProxyState *proxy,
                           struct flagcxRmaDesc *desc) {
  desc->next = NULL;
  pthread_mutex_lock(&proxy->pendingMutex);
  if (proxy->pendingTail != NULL) {
    proxy->pendingTail->next = desc;
    proxy->pendingTail = desc;
  } else {
    proxy->pendingHead = proxy->pendingTail = desc;
  }
  pthread_mutex_unlock(&proxy->pendingMutex);
}

static void rmaDescComplete(struct flagcxRmaProxyState *proxy,
                            struct flagcxRmaDesc *desc) {
  __atomic_fetch_add(&proxy->completionCount, 1ULL, __ATOMIC_RELEASE);
  __atomic_store_n(&proxy->doneSeqs[desc->peer], desc->seq, __ATOMIC_RELEASE);
  free(desc);
}

static void *flagcxRmaProgressThread(void *arg) {
  struct flagcxRmaProxyState *proxy = (struct flagcxRmaProxyState *)arg;
  struct flagcxHeteroComm *comm = proxy->comm;

  bool stopping = false;
  bool did_work = false;
  while (!stopping || proxy->inProgressHead != NULL || did_work) {
    if (__atomic_load_n(&proxy->stop, __ATOMIC_ACQUIRE))
      stopping = true;

    did_work = false;
    struct flagcxRmaDesc *desc = NULL;

    // ---- 1. Poll head of inProgress for completion ----
    struct flagcxRmaDesc *head = proxy->inProgressHead;
    if (head != NULL) {
      int done = 0;
      if (head->request != NULL) {
        flagcxResult_t testRes =
            comm->netAdaptor->test(head->request, &done, NULL);
        if (testRes != flagcxSuccess) {
          WARN("flagcxRmaProgressThread: test failed peer=%d res=%d",
               head->peer, (int)testRes);
          __atomic_store_n(&proxy->rmaError, 1, __ATOMIC_RELEASE);
          proxy->inProgressHead = head->next;
          if (proxy->inProgressHead == NULL)
            proxy->inProgressTail = NULL;
          free(head);
          did_work = true;
          goto next;
        }
      } else {
        done = 1;
      }
      if (done) {
        proxy->inProgressHead = head->next;
        if (proxy->inProgressHead == NULL)
          proxy->inProgressTail = NULL;
        rmaDescComplete(proxy, head);
        did_work = true;
      }
    }

    // ---- 2. Dequeue one desc from pending and post IB op ----
    // When stopping, skip posting new ops — just let inProgress drain.
    if (!stopping) {
      pthread_mutex_lock(&proxy->pendingMutex);
      desc = proxy->pendingHead;
      if (desc != NULL) {
        proxy->pendingHead = desc->next;
        if (proxy->pendingHead == NULL)
          proxy->pendingTail = NULL;
      }
      pthread_mutex_unlock(&proxy->pendingMutex);
    }

    if (desc != NULL) {
      int p = desc->peer;
      desc->next = NULL;
      desc->request = NULL;

      // Resolve sendComm from desc->peer.
      void *sendComm = NULL;
      if (globalOneSideHandleCount > 0 && globalOneSideHandleTable[0] != NULL &&
          globalOneSideHandleTable[0]->fullSendComms != NULL) {
        sendComm = globalOneSideHandleTable[0]->fullSendComms[p];
      }
      if (sendComm == NULL) {
        WARN("flagcxRmaProgressThread: no sendComm for peer %d", p);
        __atomic_store_n(&proxy->rmaError, 1, __ATOMIC_RELEASE);
        free(desc);
        did_work = true;
        goto next;
      }

      // Resolve data MR handles (NULL when size==0 or not applicable).
      void **srcHandles = NULL, **dstHandles = NULL;
      if (desc->size > 0 && desc->srcMrIdx >= 0) {
        srcHandles = (void **)globalOneSideHandleTable[desc->srcMrIdx];
        dstHandles = (void **)globalOneSideHandleTable[desc->dstMrIdx];
      }

      flagcxResult_t res = flagcxSuccess;
      switch (desc->type) {
        case FLAGCX_RMA_PUT:
          res = comm->netAdaptor->iput(sendComm, desc->srcOff, desc->dstOff,
                                       desc->size, comm->rank, p, srcHandles,
                                       dstHandles, &desc->request);
          break;

        case FLAGCX_RMA_PUT_SIGNAL: {
          void **sigHandles = (void **)globalOneSideSignalHandles;
          res = comm->netAdaptor->iputSignal(
              sendComm, desc->srcOff, desc->dstOff, desc->size, comm->rank, p,
              srcHandles, dstHandles, desc->signalOff, sigHandles,
              desc->signalValue, &desc->request);
          break;
        }

        case FLAGCX_RMA_GET:
          res = comm->netAdaptor->iget(
              sendComm, desc->srcOff, desc->dstOff, desc->size, p /* srcRank */,
              comm->rank /* dstRank */, srcHandles, dstHandles, &desc->request);
          break;

        case FLAGCX_RMA_PUT_VALUE: {
          struct flagcxOneSideHandleInfo *stagingH =
              globalOneSideStagingHandles;
          if (stagingH == NULL || stagingH->baseVas == NULL) {
            WARN("flagcxRmaProgressThread: staging handles not initialized");
            res = flagcxInternalError;
            break;
          }
          *(volatile uint64_t *)(stagingH->baseVas[comm->rank]) =
              desc->putValue;
          void **stagingHandles = (void **)stagingH;
          void **dstH = (void **)globalOneSideHandleTable[desc->dstMrIdx];
          res = comm->netAdaptor->iput(sendComm, 0, desc->dstOff,
                                       sizeof(uint64_t), comm->rank, p,
                                       stagingHandles, dstH, &desc->request);
          break;
        }
      }

      if (res != flagcxSuccess) {
        WARN("flagcxRmaProgressThread: op failed peer=%d type=%d res=%d", p,
             (int)desc->type, (int)res);
        __atomic_store_n(&proxy->rmaError, 1, __ATOMIC_RELEASE);
        free(desc);
      } else {
        // Enqueue to inProgress for later polling.
        if (proxy->inProgressTail != NULL) {
          proxy->inProgressTail->next = desc;
          proxy->inProgressTail = desc;
        } else {
          proxy->inProgressHead = proxy->inProgressTail = desc;
        }
      }
      did_work = true;
    }

  next:
    if (!did_work)
      sched_yield();
  }

  pthread_mutex_lock(&proxy->pendingMutex);
  struct flagcxRmaDesc *d = proxy->pendingHead;
  proxy->pendingHead = proxy->pendingTail = NULL;
  pthread_mutex_unlock(&proxy->pendingMutex);
  while (d) {
    struct flagcxRmaDesc *nxt = d->next;
    rmaDescComplete(proxy, d);
    d = nxt;
  }
  return NULL;
}

flagcxResult_t flagcxHeteroRmaProxyStart(flagcxHeteroComm_t comm) {
  int nRanks = comm->nRanks;
  struct flagcxRmaProxyState *proxy = (struct flagcxRmaProxyState *)calloc(
      1, sizeof(struct flagcxRmaProxyState));
  if (proxy == NULL) {
    WARN("flagcxHeteroRmaProxyStart: failed to allocate proxy state");
    return flagcxSystemError;
  }

  proxy->nRanks = nRanks;
  proxy->comm = comm;
  // pendingHead/Tail and inProgressHead/Tail are single pointers, already
  // zero-initialized by calloc above.
  proxy->nextSeqs = (volatile uint64_t *)calloc(nRanks, sizeof(uint64_t));
  proxy->doneSeqs = (volatile uint64_t *)calloc(nRanks, sizeof(uint64_t));

  if (proxy->nextSeqs == NULL || proxy->doneSeqs == NULL) {
    WARN("flagcxHeteroRmaProxyStart: failed to allocate seq arrays");
    free((void *)proxy->nextSeqs);
    free((void *)proxy->doneSeqs);
    free(proxy);
    return flagcxSystemError;
  }

  pthread_mutex_init(&proxy->pendingMutex, NULL);
  proxy->stop = 0;
  comm->rmaProxy = proxy;

  if (pthread_create(&proxy->thread, NULL, flagcxRmaProgressThread, proxy) !=
      0) {
    WARN("flagcxHeteroRmaProxyStart: pthread_create failed");
    free((void *)proxy->nextSeqs);
    free((void *)proxy->doneSeqs);
    pthread_mutex_destroy(&proxy->pendingMutex);
    free(proxy);
    comm->rmaProxy = NULL;
    return flagcxSystemError;
  }

  INFO(FLAGCX_INIT, "RMA progress thread started (nRanks=%d)", nRanks);
  return flagcxSuccess;
}

flagcxResult_t flagcxHeteroRmaProxyStop(flagcxHeteroComm_t comm) {
  struct flagcxRmaProxyState *proxy = comm->rmaProxy;
  if (proxy == NULL)
    return flagcxSuccess;

  __atomic_store_n(&proxy->stop, 1, __ATOMIC_RELEASE);
  pthread_join(proxy->thread, NULL);

  free((void *)proxy->nextSeqs);
  free((void *)proxy->doneSeqs);
  pthread_mutex_destroy(&proxy->pendingMutex);
  free(proxy);
  comm->rmaProxy = NULL;
  return flagcxSuccess;
}

flagcxResult_t flagcxHeteroFlushRma(flagcxHeteroComm_t comm, int peer,
                                    uint64_t seq) {
  struct flagcxRmaProxyState *proxy = comm->rmaProxy;
  if (proxy == NULL || seq == 0)
    return flagcxSuccess;
  if (peer < 0 || peer >= proxy->nRanks) {
    WARN("flagcxHeteroFlushRma: peer %d out of range (nRanks=%d)", peer,
         proxy->nRanks);
    return flagcxInvalidArgument;
  }
  while (__atomic_load_n(&proxy->doneSeqs[peer], __ATOMIC_ACQUIRE) < seq) {
    if (__atomic_load_n(&proxy->rmaError, __ATOMIC_ACQUIRE))
      return flagcxRemoteError;
    usleep(100);
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxHeteroFlushAllRma(flagcxHeteroComm_t comm) {
  struct flagcxRmaProxyState *proxy = comm->rmaProxy;
  if (proxy == NULL)
    return flagcxSuccess;
  for (int p = 0; p < proxy->nRanks; p++) {
    uint64_t target = __atomic_load_n(&proxy->nextSeqs[p], __ATOMIC_RELAXED);
    if (target == 0)
      continue;
    while (__atomic_load_n(&proxy->doneSeqs[p], __ATOMIC_ACQUIRE) < target) {
      if (__atomic_load_n(&proxy->rmaError, __ATOMIC_ACQUIRE))
        return flagcxRemoteError;
      usleep(100);
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxHeteroReadCounter(flagcxHeteroComm_t comm,
                                       uint64_t *count) {
  if (comm == NULL || comm->rmaProxy == NULL || count == NULL)
    return flagcxInvalidArgument;
  *count = __atomic_load_n(&comm->rmaProxy->completionCount, __ATOMIC_ACQUIRE);
  return flagcxSuccess;
}

flagcxResult_t flagcxHeteroWaitCounter(flagcxHeteroComm_t comm,
                                       uint64_t target) {
  if (comm == NULL || comm->rmaProxy == NULL)
    return flagcxInvalidArgument;
  while (__atomic_load_n(&comm->rmaProxy->completionCount, __ATOMIC_ACQUIRE) <
         target) {
    if (__atomic_load_n(&comm->rmaProxy->rmaError, __ATOMIC_ACQUIRE))
      return flagcxRemoteError;
    sched_yield();
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxHeteroSend(const void *sendbuff, size_t count,
                                flagcxDataType_t datatype, int peer,
                                flagcxHeteroComm_t comm, flagcxStream_t stream,
                                int opId, int step) {
  flagcxHeteroGroupStart();
  int channelId = 0;
  if (comm->channels[channelId].peers[peer]->send[0].connected == 0 &&
      comm->channels[channelId].peers[peer]->send[0].registered == 0) {
    comm->connectSend[peer] |= (1UL << channelId);
    flagcxGroupCommPreconnect(comm);
    comm->channels[channelId].peers[peer]->send[0].registered = 1;
  }
  struct flagcxTaskP2p *p2p;
  struct flagcxTasks *tasks = &comm->tasks;
  FLAGCXCHECK(flagcxCalloc(&p2p, 1));
  p2p->buff = (void *)sendbuff;
  p2p->bytes = count * getFlagcxDataTypeSize(datatype);
  p2p->chunk = 0;
  p2p->dtype = datatype;
  p2p->stream = stream;
  p2p->opId = opId;
  p2p->step = step;
  if (flagcxIntruQueueEmpty(&tasks->peers[peer].sendQueue))
    tasks->p2pOrder[tasks->p2pOrderSteps++] = peer;
  flagcxIntruQueueEnqueue(&tasks->peers[peer].sendQueue, p2p);

  flagcxGroupCommJoin(comm);
  flagcxHeteroGroupEnd();
  return flagcxSuccess;
}

flagcxResult_t flagcxHeteroRecv(void *recvbuff, size_t count,
                                flagcxDataType_t datatype, int peer,
                                flagcxHeteroComm_t comm, flagcxStream_t stream,
                                int opId, int step) {
  flagcxHeteroGroupStart();
  int channelId = 0;
  if (comm->channels[channelId].peers[peer]->recv[0].connected == 0 &&
      comm->channels[channelId].peers[peer]->recv[0].registered == 0) {
    comm->connectRecv[peer] |= (1UL << channelId);
    flagcxGroupCommPreconnect(comm);
    comm->channels[channelId].peers[peer]->recv[0].registered = 1;
  }
  struct flagcxTaskP2p *p2p;
  struct flagcxTasks *tasks = &comm->tasks;
  FLAGCXCHECK(flagcxCalloc(&p2p, 1));
  p2p->buff = (void *)recvbuff;
  p2p->bytes = count * getFlagcxDataTypeSize(datatype);
  p2p->chunk = 0;
  p2p->dtype = datatype;
  p2p->stream = stream;
  p2p->opId = opId;
  p2p->step = step;
  if (flagcxIntruQueueEmpty(&tasks->peers[peer].recvQueue))
    tasks->p2pOrder[tasks->p2pOrderSteps++] = peer;
  flagcxIntruQueueEnqueue(&tasks->peers[peer].recvQueue, p2p);

  flagcxGroupCommJoin(comm);
  flagcxHeteroGroupEnd();
  return flagcxSuccess;
}

flagcxResult_t flagcxHeteroPut(flagcxHeteroComm_t comm, int peer,
                               size_t srcOffset, size_t dstOffset, size_t size,
                               int srcMrIdx, int dstMrIdx) {
  if (comm->netAdaptor == NULL || comm->netAdaptor->iput == NULL)
    return flagcxNotSupported;
  if (peer < 0 || peer >= comm->nRanks) {
    WARN("flagcxHeteroPut: peer %d out of range (nRanks=%d)", peer,
         comm->nRanks);
    return flagcxInvalidArgument;
  }
  if (comm->rmaProxy == NULL) {
    WARN("flagcxHeteroPut: rmaProxy not initialized");
    return flagcxInternalError;
  }
  struct flagcxRmaDesc *desc;
  FLAGCXCHECK(allocRmaDesc(&desc));
  desc->peer = peer;
  desc->type = FLAGCX_RMA_PUT;
  desc->srcOff = (uint64_t)srcOffset;
  desc->dstOff = (uint64_t)dstOffset;
  desc->size = size;
  desc->srcMrIdx = srcMrIdx;
  desc->dstMrIdx = dstMrIdx;
  desc->seq =
      __atomic_add_fetch(&comm->rmaProxy->nextSeqs[peer], 1, __ATOMIC_RELAXED);
  enqueuePending(comm->rmaProxy, desc);
  return flagcxSuccess;
}

flagcxResult_t flagcxHeteroGet(flagcxHeteroComm_t comm, int peer,
                               size_t srcOffset, size_t dstOffset, size_t size,
                               int srcMrIdx, int dstMrIdx) {
  if (comm->netAdaptor == NULL || comm->netAdaptor->iget == NULL)
    return flagcxNotSupported;
  if (peer < 0 || peer >= comm->nRanks) {
    WARN("flagcxHeteroGet: peer %d out of range (nRanks=%d)", peer,
         comm->nRanks);
    return flagcxInvalidArgument;
  }
  if (comm->rmaProxy == NULL) {
    WARN("flagcxHeteroGet: rmaProxy not initialized");
    return flagcxInternalError;
  }
  struct flagcxRmaDesc *desc;
  FLAGCXCHECK(allocRmaDesc(&desc));
  desc->peer = peer;
  desc->type = FLAGCX_RMA_GET;
  desc->srcOff = (uint64_t)srcOffset;
  desc->dstOff = (uint64_t)dstOffset;
  desc->size = size;
  desc->srcMrIdx = srcMrIdx;
  desc->dstMrIdx = dstMrIdx;
  desc->seq =
      __atomic_add_fetch(&comm->rmaProxy->nextSeqs[peer], 1, __ATOMIC_RELAXED);
  enqueuePending(comm->rmaProxy, desc);
  return flagcxSuccess;
}

flagcxResult_t flagcxHeteroPutSignal(flagcxHeteroComm_t comm, int peer,
                                     size_t srcOffset, size_t dstOffset,
                                     size_t size, size_t signalOffset,
                                     int srcMrIdx, int dstMrIdx,
                                     uint64_t signalValue) {
  if (comm->netAdaptor == NULL || comm->netAdaptor->iputSignal == NULL)
    return flagcxNotSupported;
  if (peer < 0 || peer >= comm->nRanks) {
    WARN("flagcxHeteroPutSignal: peer %d out of range (nRanks=%d)", peer,
         comm->nRanks);
    return flagcxInvalidArgument;
  }
  if (comm->rmaProxy == NULL) {
    WARN("flagcxHeteroPutSignal: rmaProxy not initialized");
    return flagcxInternalError;
  }
  struct flagcxRmaDesc *desc;
  FLAGCXCHECK(allocRmaDesc(&desc));
  desc->peer = peer;
  desc->type = FLAGCX_RMA_PUT_SIGNAL;
  desc->srcOff = (uint64_t)srcOffset;
  desc->dstOff = (uint64_t)dstOffset;
  desc->size = size;
  // For signal-only (size==0) there are no data MR handles.
  desc->srcMrIdx = (size > 0) ? srcMrIdx : -1;
  desc->dstMrIdx = (size > 0) ? dstMrIdx : -1;
  desc->signalOff = (uint64_t)signalOffset;
  desc->signalValue = signalValue;
  desc->seq =
      __atomic_add_fetch(&comm->rmaProxy->nextSeqs[peer], 1, __ATOMIC_RELAXED);
  enqueuePending(comm->rmaProxy, desc);
  return flagcxSuccess;
}

flagcxResult_t flagcxHeteroFlush(flagcxHeteroComm_t comm, void *gpuAddr,
                                 size_t size, void *gHandleInfo) {
  struct flagcxOneSideHandleInfo *info =
      (struct flagcxOneSideHandleInfo *)gHandleInfo;
  if (info == NULL || info->localRecvComm == NULL ||
      info->localMrHandle == NULL)
    return flagcxNotSupported;
  if (comm->netAdaptor == NULL || comm->netAdaptor->iflush == NULL)
    return flagcxNotSupported;

  if (size > (size_t)INT_MAX) {
    WARN("flagcxHeteroFlush: size %zu exceeds int limit", size);
    return flagcxInternalError;
  }
  void *data_arr[1] = {gpuAddr};
  int sizes_arr[1] = {(int)size};
  void *mh_arr[1] = {info->localMrHandle};
  void *request = NULL;
  FLAGCXCHECK(comm->netAdaptor->iflush(info->localRecvComm, 1, data_arr,
                                       sizes_arr, mh_arr, &request));
  if (request != NULL) {
    int done = 0;
    while (!done) {
      FLAGCXCHECK(comm->netAdaptor->test(request, &done, NULL));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxHeteroWaitSignal(flagcxHeteroComm_t comm, int peer,
                                      size_t signalOffset, uint64_t expected,
                                      flagcxStream_t stream) {
  (void)peer;
  struct flagcxOneSideHandleInfo *info =
      (struct flagcxOneSideHandleInfo *)globalOneSideSignalHandles;
  if (info == NULL || info->baseVas == NULL)
    return flagcxNotSupported;

  int myRank = comm->rank;
  void *signalAddr = (void *)(info->baseVas[myRank] + signalOffset);

  // Device-side wait (streamWaitValue64) for GPU signal buffer.
  // RMA signal buffers are GPU memory (flagcxMemAlloc) — host-side volatile
  // polling would segfault. Non-CUDA platforms return flagcxNotSupported.
  // No flush needed: FORCE_SO on signal MR guarantees PCIe ordering.
  if (stream == NULL)
    return flagcxInternalError;

  return deviceAdaptor->streamWaitValue64(stream, signalAddr, expected, 0);
}

flagcxResult_t flagcxHeteroPutValue(flagcxHeteroComm_t comm, int peer,
                                    uint64_t value, size_t dstOffset,
                                    int dstMrIdx) {
  if (comm->netAdaptor == NULL || comm->netAdaptor->iput == NULL)
    return flagcxNotSupported;
  if (peer < 0 || peer >= comm->nRanks) {
    WARN("flagcxHeteroPutValue: peer %d out of range (nRanks=%d)", peer,
         comm->nRanks);
    return flagcxInvalidArgument;
  }
  if (dstMrIdx < 0 || dstMrIdx >= globalOneSideHandleCount) {
    WARN("flagcxHeteroPutValue: dstMrIdx %d out of range (count=%d)", dstMrIdx,
         globalOneSideHandleCount);
    return flagcxInvalidArgument;
  }
  if (comm->rmaProxy == NULL) {
    WARN("flagcxHeteroPutValue: rmaProxy not initialized");
    return flagcxInternalError;
  }
  struct flagcxRmaDesc *desc;
  FLAGCXCHECK(allocRmaDesc(&desc));
  desc->peer = peer;
  desc->type = FLAGCX_RMA_PUT_VALUE;
  desc->dstOff = (uint64_t)dstOffset;
  desc->dstMrIdx = dstMrIdx;
  desc->putValue = value;
  desc->seq =
      __atomic_add_fetch(&comm->rmaProxy->nextSeqs[peer], 1, __ATOMIC_RELAXED);
  enqueuePending(comm->rmaProxy, desc);
  return flagcxSuccess;
}
