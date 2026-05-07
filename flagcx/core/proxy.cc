/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE-NCCL.txt for license information
 ************************************************************************/

#include "proxy.h"
#include "adaptor.h"
#include "comm.h"
#include "device_api/flagcx_device.h" // flagcxDevCommInternal, devComm
#include "flagcx_hetero.h"
#include "flagcx_kernel.h" // FLAGCX_DEVICE_CTA_COUNT
#include "info.h"
#include "net.h"
#include "onesided.h"
#include "p2p.h"
#include "socket.h"
#include "transport.h"
#define ENABLE_TIMER 0
#include "timer.h"

#include <assert.h>
#include <string>
#include <sys/syscall.h>
#include <sys/time.h>
#include <unistd.h>
using namespace std;

enum { proxyRecv = 0, proxySend = 1 };
extern union flagcxSocketAddress bootstrapNetIfAddr;

static bool proxyMatchOpType(int type) {
  switch (type) {
    case flagcxProxyMsgInit:
    case flagcxProxyMsgSharedInit:
    case flagcxProxyMsgSetup:
    case flagcxProxyMsgConnect:
    case flagcxProxyMsgGetFd:
    case flagcxProxyMsgRegister:
    case flagcxProxyMsgDeregister:
    case flagcxProxyMsgRegMr:
    case flagcxProxyMsgDeregMr:
    case flagcxProxyMsgSendRecv:
      return true;
    default:
      return false;
  }
}

FLAGCX_TEMPLETELIST_DEFINE(ProdProgChannel, struct flagcxProxyOps,
                           prodPrevChannel, prodNextChannel);
FLAGCX_TEMPLETELIST_DEFINE(ConsProgChannel, struct flagcxProxyOps,
                           consPrevChannel, consNextChannel);
FLAGCX_TEMPLETELIST_DEFINE(ProgPeer, struct flagcxProxyOps::consPeer, prevPeer,
                           nextPeer);

flagcxResult_t
flagcxProxyProgressChannelJoin(struct flagcxProxyState *proxyState,
                               struct flagcxProxyState *) {

  return flagcxSuccess;
}

static flagcxResult_t asyncProxyOpEnqueue(flagcxProxyAsyncOp **opHead,
                                          flagcxProxyAsyncOp *newOp) {
  flagcxProxyAsyncOp *list = *opHead;
  if (list == NULL)
    *opHead = newOp;
  else {
    while (list->next)
      list = list->next;
    list->next = newOp;
    newOp->prev = list;
  }
  return flagcxSuccess;
}

static flagcxResult_t asyncProxyOpDequeue(flagcxProxyAsyncOp **opHead,
                                          flagcxProxyAsyncOp *op) {
  if (*opHead == op)
    *opHead = op->next;
  if (op->next)
    op->next->prev = op->prev;
  if (op->prev)
    op->prev->next = op->next;
  if (op->reqSize)
    free(op->reqBuff);
  if (op->respSize)
    free(op->respBuff);
  free(op);
  return flagcxSuccess;
}

static flagcxResult_t SaveProxy(struct flagcxHeteroComm *comm,
                                struct flagcxChannel *channel, int type,
                                int peer, struct flagcxProxyOp *op,
                                int connIndex, bool *justInquire) {
  if (peer < 0)
    return flagcxSuccess;

  if (justInquire)
    *justInquire = true;
  else {
    struct flagcxProxyOps *proxyOps;
    struct flagcxIntruQueue<struct flagcxProxyOp, &flagcxProxyOp::next> *queue;

    proxyOps = &comm->proxyState->proxyOps[op->channelId];
    queue = type == proxySend ? &proxyOps->prodPeers.sendQueue
                              : &proxyOps->prodPeers.recvQueue;

    pthread_mutex_lock(&comm->proxyState->mutex);
    flagcxProdProgChannelListEnList(&comm->proxyState->prodProgChannelHead,
                                    proxyOps);
    flagcxIntruQueueEnqueue(queue, op);
    pthread_cond_signal(&comm->proxyState->cond);
    pthread_mutex_unlock(&comm->proxyState->mutex);
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxProxySaveOp(struct flagcxHeteroComm *comm,
                                 struct flagcxProxyOp *op, bool *justInquire) {
  struct flagcxChannel *channel = &comm->channels[op->channelId];
  if (justInquire)
    *justInquire = false;
  switch (op->pattern) {
    case flagcxPatternSend:
      // Self-copy will be saved as a send operation
      if (op->root == comm->rank)
        op->selfCopy = 1;
      FLAGCXCHECK(
          SaveProxy(comm, channel, proxySend, op->root, op, 0, justInquire));
      break;
    case flagcxPatternRecv:
      if (op->root == comm->rank)
        return flagcxSuccess;
      FLAGCXCHECK(
          SaveProxy(comm, channel, proxyRecv, op->root, op, 0, justInquire));
      break;
  }
  return flagcxSuccess;
}

// Only for double check purpose, we can check if the progress queue is empty
// It is safe to not call this function in the progress thread.
static void flagcxProgressQueEmptyCheck(struct flagcxProxyState *proxyState) {
  bool error = 0;
  if (!flagcxProdProgChannelListEmpty(proxyState->prodProgChannelHead) ||
      !flagcxConsProgChannelListEmpty(proxyState->consProgChannelHead)) {
    error = 1;
  }
  for (int i = 0; i < MAXCHANNELS; i++) {
    if (!flagcxProgPeerListEmpty(proxyState->proxyOps[i].consProgPeerHead))
      error = 1;
    for (int r = 0; r < proxyState->nRanks; r++) {
      if (!flagcxIntruQueueEmpty(
              &proxyState->proxyOps[i].consPeers[r].sendQueue) ||
          !flagcxIntruQueueEmpty(
              &proxyState->proxyOps[i].consPeers[r].recvQueue))
        error = 1;
    }
    if (!flagcxIntruQueueEmpty(&proxyState->proxyOps[i].prodPeers.sendQueue) ||
        !flagcxIntruQueueEmpty(&proxyState->proxyOps[i].prodPeers.recvQueue))
      error = 1;
  }
  if (error)
    INFO(FLAGCX_INIT, "progress queue is not empty");
}

// process all the ProxyOps in the consumer queue
// idle is set to 1 if no operations are pending
// if idle is set to 0, it means there are pending operations
// For simplicity, if these are any pending operations in queue, we set idle to
// 0
static flagcxResult_t progressOps(struct flagcxProxyState *proxyState,
                                  int *idle) {
  *idle = 1;
  if (!flagcxConsProgChannelListEmpty(proxyState->consProgChannelHead)) {
    struct flagcxProxyOps *proxyOps = proxyState->consProgChannelHead;
    do {
      struct flagcxProxyOps *next = proxyOps->consNextChannel;

      if (!flagcxProgPeerListEmpty(proxyOps->consProgPeerHead)) {
        struct flagcxProxyOps::consPeer *peer = proxyOps->consProgPeerHead;
        do {
          struct flagcxProxyOps::consPeer *next = peer->nextPeer;
          struct flagcxIntruQueue<struct flagcxProxyOp, &flagcxProxyOp::next>
              *queue;
          queue = &peer->sendQueue;
          if (!flagcxIntruQueueEmpty(queue)) {
            *idle &= 0;
            struct flagcxProxyOp *op = flagcxIntruQueueHead(queue);
            if (op->connection->transport == TRANSPORT_NET) {
              struct sendNetResources *resources =
                  (sendNetResources *)op->connection->transportResources;
              flagcxProxySend(resources, op->recvbuff, op->nbytes, &op->args);
              if (op->args.done == 1 && op->args.semaphore->pollEnd()) {
                op->args.semaphore.reset();
                flagcxIntruQueueDelete(queue, op);
                free(op);
              }
            } else if (op->connection->transport == TRANSPORT_P2P) {
              struct flagcxP2pResources *resources =
                  (flagcxP2pResources *)op->connection->transportResources;
              if (op->selfCopy == 0) {
                flagcxP2pProxySend(resources, op->recvbuff, op->nbytes,
                                   &op->args);
              } else {
                flagcxP2pProxySelfCopy(resources, op->sendbuff, op->recvbuff,
                                       op->nbytes, &op->args);
              }
              if (op->args.done == 1 && op->args.semaphore->pollEnd()) {
                op->args.semaphore.reset();
                flagcxIntruQueueDelete(queue, op);
                free(op);
              }
            }
          }
          queue = &peer->recvQueue;
          if (!flagcxIntruQueueEmpty(queue)) {
            *idle &= 0;
            struct flagcxProxyOp *op = flagcxIntruQueueHead(queue);
            if (op->connection->transport == TRANSPORT_NET) {
              struct recvNetResources *resources =
                  (recvNetResources *)op->connection->transportResources;
              flagcxProxyRecv(resources, op->recvbuff, op->nbytes, &op->args);
              if (op->args.done == 1 && op->args.semaphore->pollEnd()) {
                // update refcount and delete semaphore when refcount = 0
                op->args.semaphore.reset();
                flagcxIntruQueueDelete(queue, op);
                free(op);
              }
            } else if (op->connection->transport == TRANSPORT_P2P) {
              struct flagcxP2pResources *resources =
                  (flagcxP2pResources *)op->connection->transportResources;
              flagcxP2pProxyRecv(resources, op->recvbuff, op->nbytes,
                                 &op->args);
              if (op->args.done == 1 && op->args.semaphore->pollEnd()) {
                // update refcount and delete semaphore when refcount = 0
                op->args.semaphore.reset();
                flagcxIntruQueueDelete(queue, op);
                free(op);
              }
            }
          }
          if (flagcxIntruQueueEmpty(&peer->sendQueue) &&
              flagcxIntruQueueEmpty(&peer->recvQueue)) {
            flagcxProgPeerListDelete(&proxyOps->consProgPeerHead, peer);
          }
          peer = next;
        } while (peer != NULL);
      }
      if (flagcxProgPeerListEmpty(proxyOps->consProgPeerHead)) {
        flagcxConsProgChannelListDelete(&proxyState->consProgChannelHead,
                                        proxyOps);
      }
      proxyOps = next;
    } while (proxyOps != NULL);
  }
  return flagcxSuccess;
}

// get proxy operations from the producer queue
// and move them to the consumer queue
// added means the number of operations fetched from producer queue and added to
// the consumer queue.
static flagcxResult_t
flagcxProxyGetPostedOps(struct flagcxProxyState *proxyState, int *added) {
  struct flagcxProxyProgressState *state = &proxyState->progressState;
  // No need to block waiting for the lock to be available. Exit, continue
  // progress, and come back later.
  if (pthread_mutex_trylock(&proxyState->mutex) != 0) {
    *added = 0;
    return flagcxSuccess;
  }

  // If we have ops to progress, no need to block waiting for something to
  // arrive
  if (flagcxConsProgChannelListEmpty(proxyState->consProgChannelHead)) {
    while (flagcxProdProgChannelListEmpty(proxyState->prodProgChannelHead) &&
           state->stop == 0) {
      pthread_cond_wait(&proxyState->cond, &proxyState->mutex);
    }
    if (state->stop != 0) {
      pthread_mutex_unlock(&proxyState->mutex);
      *added = 0;
      return flagcxSuccess;
    }
  }

  // Put anything available right now in the producer queue into the consumer
  // queue.
  while (!flagcxProdProgChannelListEmpty(proxyState->prodProgChannelHead)) {
    struct flagcxProxyOps *proxyOps =
        flagcxProdProgChannelListDeList(&proxyState->prodProgChannelHead);

    flagcxConsProgChannelListEnList(&proxyState->consProgChannelHead, proxyOps);
    struct flagcxIntruQueue<struct flagcxProxyOp, &flagcxProxyOp::next> *queue;
    queue = &proxyOps->prodPeers.sendQueue;
    while (!flagcxIntruQueueEmpty(queue)) {
      struct flagcxProxyOp *op = flagcxIntruQueueDequeue(queue);
      flagcxProgPeerListEnList(&proxyOps->consProgPeerHead,
                               &proxyOps->consPeers[op->root]);
      flagcxIntruQueueEnqueue(&proxyOps->consPeers[op->root].sendQueue, op);
      (*added)++;
    }
    queue = &proxyOps->prodPeers.recvQueue;
    while (!flagcxIntruQueueEmpty(queue)) {
      struct flagcxProxyOp *op = flagcxIntruQueueDequeue(queue);
      flagcxProgPeerListEnList(&proxyOps->consProgPeerHead,
                               &proxyOps->consPeers[op->root]);
      flagcxIntruQueueEnqueue(&proxyOps->consPeers[op->root].recvQueue, op);
      (*added)++;
    }
  }
  pthread_mutex_unlock(&proxyState->mutex);
  return flagcxSuccess;
}

FLAGCX_PARAM(ProgressAppendOpFreq, "PROGRESS_APPENDOP_FREQ", 8);
FLAGCX_PARAM(KernelProxyParallelism, "KERNEL_PROXY_PARALLELISM", 4);

inline void *flagcxProxyProgress(void *proxyState_) {
  struct flagcxProxyState *proxyState = (flagcxProxyState *)proxyState_;
  // flag indicating if there is any in-operating operation
  int idle = 1;
  /* Too frequent call of ncclProxyGetPostedOps() will result in perf regression
   * for small message communication. proxyOpAppendCounter is a counter that
   * helps us decide if we need to append proxy ops. After each progress,
   * proxyOpAppendCounter will increase by 1 and compare with environment
   * variable ncclParamProgressAppendOpFreq(). If they are equal, we will append
   * proxy ops. This will decrease the frequency of calling
   * ncclProxyGetPostedOps() and reduce the perf impact. */
  int proxyOpAppendCounter = 0;
  deviceAdaptor->setDevice(proxyState->cudaDev);
  struct flagcxProxyProgressState *state = &proxyState->progressState;

  while (state->stop == 0 || idle == 0) {
    idle = 1;
    // consume the operations in the consumer queue
    progressOps(proxyState, &idle);

    if (idle || (++proxyOpAppendCounter == flagcxParamProgressAppendOpFreq())) {
      int added = 0;
      proxyOpAppendCounter = 0;
      if (state->stop == 0) {
        // move all the operations from the producer queue to the consumer queue
        flagcxProxyGetPostedOps(proxyState, &added);
      }
      if (added == 0) {
        sched_yield(); // No request progressed. Let others run.
      }
    }
  }

  flagcxProgressQueEmptyCheck(proxyState);
  return NULL;
}

static flagcxResult_t expectedProxyResponseStore(struct flagcxProxyState *state,
                                                 void *opId, void *respBuff,
                                                 int respSize,
                                                 flagcxResult_t res) {
  struct flagcxExpectedProxyResponse *elem = state->expectedResponses;
  while (elem) {
    if (elem->opId == opId) {
      if (respSize != elem->respSize) {
        WARN("Mismatched response size for opId=%p", opId);
        return flagcxInternalError;
      }

      if (elem->done) {
        WARN("Storing response for already completed opId=%p", opId);
        return flagcxInternalError;
      }

      if (respSize > 0 && respBuff != NULL) {
        memcpy(elem->respBuff, respBuff, respSize);
        free(respBuff);
      }
      elem->done = true;
      elem->res = res;
      return flagcxSuccess;
    }
    elem = elem->next;
  }

  WARN("Proxy response for opId=%p doesn't match any expected response", opId);
  return flagcxInternalError;
}

static flagcxResult_t
expectedProxyResponseEnqueue(struct flagcxProxyState *state, void *opId,
                             int respSize) {
  struct flagcxExpectedProxyResponse *ex;
  FLAGCXCHECK(flagcxCalloc(&ex, 1));
  ex->opId = opId;

  // Pre-alloc response buffer
  ex->respBuff = malloc(respSize);
  ex->respSize = respSize;
  ex->res = flagcxInternalError;
  ex->done = false;

  // Enqueue
  struct flagcxExpectedProxyResponse *list = state->expectedResponses;
  if (list == NULL) {
    state->expectedResponses = ex;
    return flagcxSuccess;
  }
  while (list->next)
    list = list->next;
  list->next = ex;
  return flagcxSuccess;
}

static flagcxResult_t
expectedProxyResponseDequeue(struct flagcxProxyState *state, void *opId,
                             void *respBuff, int *found) {
  struct flagcxExpectedProxyResponse *elem = state->expectedResponses;
  struct flagcxExpectedProxyResponse *prev = NULL;
  *found = 0;
  while (elem) {
    if ((elem->opId == opId) && elem->done) {
      if (prev == NULL) {
        state->expectedResponses = elem->next;
      } else {
        prev->next = elem->next;
      }
      memcpy(respBuff, elem->respBuff, elem->respSize);
      flagcxResult_t res = elem->res;
      free(elem->respBuff);
      free(elem);
      *found = 1;
      return res;
    }
    prev = elem;
    elem = elem->next;
  }
  return flagcxSuccess;
}

static flagcxResult_t
expectedProxyResponseRemove(struct flagcxProxyState *state, void *opId) {
  struct flagcxExpectedProxyResponse *elem = state->expectedResponses;
  struct flagcxExpectedProxyResponse *prev = NULL;
  while (elem) {
    if (elem->opId == opId) {
      if (prev == NULL) {
        state->expectedResponses = elem->next;
      } else {
        prev->next = elem->next;
      }
      free(elem->respBuff);
      free(elem);
      return flagcxSuccess;
    }
    prev = elem;
    elem = elem->next;
  }
  WARN("Couldn't find opId=%p", opId);
  return flagcxInternalError;
}

flagcxResult_t flagcxPollProxyResponse(struct flagcxHeteroComm *comm,
                                       struct flagcxProxyConnector *proxyConn,
                                       void *respBuff, void *opId) {
  struct flagcxProxyState *sharedProxyState = comm->proxyState;
  // Check response queue
  int found = 0;
  flagcxResult_t res =
      expectedProxyResponseDequeue(sharedProxyState, opId, respBuff, &found);

  if (found == 0) {
    // Attempt to read in a new response header from the proxy thread
    if (sharedProxyState->peerSocks == NULL)
      return flagcxInternalError;
    struct flagcxSocket *sock = &sharedProxyState->peerSocks[proxyConn->tpRank];
    flagcxProxyRpcResponseHeader resp = {0};
    int offset = 0;
    if (flagcxSuccess != flagcxSocketProgress(FLAGCX_SOCKET_RECV, sock, &resp,
                                              sizeof(resp), &offset)) {
      WARN("Socket recv failed while polling for opId=%p", opId);
      return flagcxInternalError;
    }

    if (offset == 0) {
      return flagcxInProgress;
      // If we've returned a partial response, block to receive the rest of it
    } else if (offset < sizeof(resp)) {
      while (offset < sizeof(resp))
        FLAGCXCHECK(flagcxSocketProgress(FLAGCX_SOCKET_RECV, sock, &resp,
                                         sizeof(resp), &offset));
    }

    INFO(FLAGCX_PROXY, "flagcxPollProxyResponse Received new opId=%p",
         resp.opId);

    // If there's a respSize to recv
    if (resp.respSize > 0) {
      if (resp.opId != opId) {
        // Unexpected response, need to buffer the socket data
        respBuff = malloc(resp.respSize);
      }
      assert(respBuff != NULL);
      FLAGCXCHECK(flagcxSocketRecv(sock, respBuff, resp.respSize));
    }

    if (resp.opId == opId) {
      INFO(FLAGCX_PROXY, "resp.opId=%p matches expected opId=%p", resp.opId,
           opId);
      FLAGCXCHECK(expectedProxyResponseRemove(sharedProxyState, resp.opId));
      return resp.res;
    } else {
      INFO(FLAGCX_PROXY, "Queuing opId=%p respBuff=%p respSize=%d", resp.opId,
           respBuff, resp.respSize);
      // Store the result and mark response as completed
      FLAGCXCHECK(expectedProxyResponseStore(
          sharedProxyState, resp.opId, respBuff, resp.respSize, resp.res));
      return flagcxInProgress;
    }
  } else {
    INFO(FLAGCX_PROXY, "flagcxPollProxyResponse Dequeued cached opId=%p", opId);
  }
  return res;
}

static flagcxResult_t proxyProgressAsync(flagcxProxyAsyncOp **opHead,
                                         flagcxProxyAsyncOp *op,
                                         int *asyncOpCount) {
  int done = 0;
  const char *dmaBufEnable = flagcxGetEnv("FLAGCX_DMABUF_ENABLE");
  bool dmaEnabled = false; // disabled by default
  if (dmaBufEnable != NULL) {
    if (strcmp(dmaBufEnable, "1") == 0) {
      dmaEnabled = true;
    }
  }
  bool dmaBufferSupport = false;
  if (deviceAdaptor->dmaSupport != NULL) {
    deviceAdaptor->dmaSupport(&dmaBufferSupport);
  }
  dmaBufferSupport = dmaEnabled && dmaBufferSupport;
  if (op->type == flagcxProxyMsgConnect) {
    TRACE(FLAGCX_PROXY,
          "proxyProgressAsync::flagcxProxyMsgConnect opId=%p op.reqBuff=%p, "
          "op->reqSize=%d, op->respSize=%d, transport=%d",
          op->opId, op->reqBuff, op->reqSize, op->respSize,
          op->connection->transport);

    if (op->connection->transport == TRANSPORT_P2P) {
      // P2P transport
      if (op->connection->send) {
        INFO(FLAGCX_PROXY, "Calling flagcxP2pSendProxyConnect");
        flagcxP2pSendProxyConnect(op->connection, NULL, op->reqBuff,
                                  op->reqSize, op->respBuff, op->respSize,
                                  &done);
        INFO(FLAGCX_PROXY, "flagcxP2pSendProxyConnect completed, done=%d",
             done);
      } else {
        INFO(FLAGCX_PROXY, "Calling flagcxP2pRecvProxyConnect");
        flagcxP2pRecvProxyConnect(op->connection, NULL, op->reqBuff,
                                  op->reqSize, op->respBuff, op->respSize,
                                  &done);
        INFO(FLAGCX_PROXY, "flagcxP2pRecvProxyConnect completed, done=%d",
             done);
      }
    } else if (op->connection->transport == TRANSPORT_NET) {
      // NET transport (original logic)
      if (op->connection->send) {
        struct sendNetResources *resources =
            (struct sendNetResources *)op->connection->transportResources;
        if (!resources->netSendComm) {
          FLAGCXCHECK(resources->netAdaptor->connect(
              resources->netDev, (void *)op->reqBuff, &resources->netSendComm));
        } else {
          if (dmaBufferSupport &&
              resources->netAdaptor == getUnifiedNetAdaptor(IBRC)) {
            INFO(FLAGCX_PROXY,
                 "Registering memory region with DMA-BUF support");
            int dmabuf_fd;
            FLAGCXCHECK(deviceAdaptor->getHandleForAddressRange(
                (void *)&dmabuf_fd, resources->buffers[0],
                resources->buffSizes[0], 0));
            FLAGCXCHECK(resources->netAdaptor->regMrDmaBuf(
                resources->netSendComm, resources->buffers[0],
                resources->buffSizes[0], 2, 0ULL, dmabuf_fd, 0,
                &resources->mhandles[0]));
            (void)close(dmabuf_fd);
          } else {
            if (resources->netAdaptor == getUnifiedNetAdaptor(IBRC)) {
              FLAGCXCHECK(resources->netAdaptor->regMr(
                  resources->netSendComm, resources->buffers[0],
                  resources->buffSizes[0], 2, 0, &resources->mhandles[0]));
            } else if (resources->netAdaptor == getUnifiedNetAdaptor(SOCKET)) {
              FLAGCXCHECK(resources->netAdaptor->regMr(
                  resources->netSendComm, resources->buffers[0],
                  resources->buffSizes[0], 1, 0, &resources->mhandles[0]));
            } else {
              FLAGCXCHECK(resources->netAdaptor->regMr(
                  resources->netSendComm, resources->buffers[0],
                  resources->buffSizes[0],
                  (resources->ptrSupport & FLAGCX_PTR_CUDA) ? FLAGCX_PTR_CUDA
                                                            : FLAGCX_PTR_HOST,
                  0, &resources->mhandles[0]));
            }
          }
          done = 1;
        }
      } else {
        struct recvNetResources *resources =
            (struct recvNetResources *)op->connection->transportResources;
        if (!resources->netRecvComm) {
          FLAGCXCHECK(resources->netAdaptor->accept(resources->netListenComm,
                                                    &resources->netRecvComm));
        } else {
          if (dmaBufferSupport) {
            INFO(FLAGCX_PROXY,
                 "Registering memory region with DMA-BUF support");
            int dmabuf_fd;
            FLAGCXCHECK(deviceAdaptor->getHandleForAddressRange(
                (void *)&dmabuf_fd, resources->buffers[0],
                resources->buffSizes[0], 0));
            FLAGCXCHECK(resources->netAdaptor->regMrDmaBuf(
                resources->netRecvComm, resources->buffers[0],
                resources->buffSizes[0], 2, 0ULL, dmabuf_fd, 0,
                &resources->mhandles[0]));
            (void)close(dmabuf_fd);
          } else {
            if (resources->netAdaptor == getUnifiedNetAdaptor(IBRC)) {
              FLAGCXCHECK(resources->netAdaptor->regMr(
                  resources->netRecvComm, resources->buffers[0],
                  resources->buffSizes[0], 2, 0, &resources->mhandles[0]));
            } else if (resources->netAdaptor == getUnifiedNetAdaptor(SOCKET)) {
              FLAGCXCHECK(resources->netAdaptor->regMr(
                  resources->netRecvComm, resources->buffers[0],
                  resources->buffSizes[0], 1, 0, &resources->mhandles[0]));
            } else {
              FLAGCXCHECK(resources->netAdaptor->regMr(
                  resources->netRecvComm, resources->buffers[0],
                  resources->buffSizes[0],
                  (resources->ptrSupport & FLAGCX_PTR_CUDA) ? FLAGCX_PTR_CUDA
                                                            : FLAGCX_PTR_HOST,
                  0, &resources->mhandles[0]));
            }
          }
          done = 1;
        }
      }
    }
  } else if (op->type == flagcxProxyMsgRegister) {
    TRACE(FLAGCX_PROXY,
          "proxyProgressAsync::flagcxProxyMsgRegister opId=%p op.reqBuff=%p, "
          "op->reqSize=%d, op->respSize=%d",
          op->opId, op->reqBuff, op->reqSize, op->respSize);
    if (op->connection->transport == TRANSPORT_P2P) {
      FLAGCXCHECK(flagcxP2pProxyRegister(op->connection, NULL, op->reqBuff,
                                         op->reqSize, op->respBuff,
                                         op->respSize, &done));
    } else if (op->connection->transport == TRANSPORT_NET) {
      void *handle;
      struct netRegInfo *info = (struct netRegInfo *)op->reqBuff;
      assert(op->reqSize == sizeof(struct netRegInfo));
      assert(op->respSize == sizeof(void *));
      if (op->connection->send) {
        // send side
        struct sendNetResources *resources =
            (struct sendNetResources *)(op->connection->transportResources);
        if (dmaBufferSupport) {
          int dmabuf_fd;
          FLAGCXCHECK(deviceAdaptor->getHandleForAddressRange(
              (void *)&dmabuf_fd, (void *)info->buffer, info->size, 0));
          FLAGCXCHECK(resources->netAdaptor->regMrDmaBuf(
              resources->netSendComm, (void *)info->buffer, info->size, 2, 0ULL,
              dmabuf_fd, 0, &handle));
          (void)close(dmabuf_fd);
        } else {
          FLAGCXCHECK(resources->netAdaptor->regMr(resources->netSendComm,
                                                   (void *)info->buffer,
                                                   info->size, 2, 0, &handle));
        }
      } else {
        // recv side
        struct recvNetResources *resources =
            (struct recvNetResources *)(op->connection->transportResources);
        if (dmaBufferSupport) {
          int dmabuf_fd;
          FLAGCXCHECK(deviceAdaptor->getHandleForAddressRange(
              (void *)&dmabuf_fd, (void *)info->buffer, info->size, 0));
          FLAGCXCHECK(resources->netAdaptor->regMrDmaBuf(
              resources->netRecvComm, (void *)info->buffer, info->size, 2, 0ULL,
              dmabuf_fd, 0, &handle));
          (void)close(dmabuf_fd);
        } else {
          FLAGCXCHECK(resources->netAdaptor->regMr(resources->netRecvComm,
                                                   (void *)info->buffer,
                                                   info->size, 2, 0, &handle));
        }
      }
      memcpy(op->respBuff, (void *)&handle, sizeof(void *));
      done = 1;
    }
  } else if (op->type == flagcxProxyMsgDeregister) {
    TRACE(FLAGCX_PROXY,
          "proxyProgressAsync::flagcxProxyMsgDeregister opId=%p op.reqBuff=%p, "
          "op->reqSize=%d, op->respSize=%d",
          op->opId, op->reqBuff, op->reqSize, op->respSize);
    if (op->connection->transport == TRANSPORT_P2P) {
      FLAGCXCHECK(flagcxP2pProxyDeregister(op->connection, NULL, op->reqBuff,
                                           op->reqSize, &done));
    } else if (op->connection->transport == TRANSPORT_NET) {
      void *handle;
      assert(op->reqSize == sizeof(void *));
      memcpy(&handle, op->reqBuff, sizeof(void *));
      if (op->connection->send) {
        // send side
        struct sendNetResources *resources =
            (struct sendNetResources *)(op->connection->transportResources);
        FLAGCXCHECK(
            resources->netAdaptor->deregMr(resources->netSendComm, handle));
      } else {
        // recv side
        struct recvNetResources *resources =
            (struct recvNetResources *)(op->connection->transportResources);
        FLAGCXCHECK(
            resources->netAdaptor->deregMr(resources->netRecvComm, handle));
      }
      done = 1;
    }
  } else if (op->type == flagcxProxyMsgSetup &&
             op->connection->transport == TRANSPORT_P2P) {
    if (op->connection->send) {
      // P2P Send side setup
      INFO(FLAGCX_PROXY, "Calling flagcxP2pSendProxySetup");
      flagcxP2pSendProxySetup(op->connection, NULL, op->reqBuff, op->reqSize,
                              op->respBuff, op->respSize, &done);
      INFO(FLAGCX_PROXY, "flagcxP2pSendProxySetup completed, done=%d", done);
    } else {
      // P2P Recv side setup
      INFO(FLAGCX_PROXY, "Calling flagcxP2pRecvProxySetup");
      flagcxP2pRecvProxySetup(op->connection, NULL, op->reqBuff, op->reqSize,
                              op->respBuff, op->respSize, &done);
      INFO(FLAGCX_PROXY, "flagcxP2pRecvProxySetup completed, done=%d", done);
    }
  } else {
    return flagcxInternalError;
  }
  if (done) {
    INFO(FLAGCX_PROXY,
         "proxyProgressAsync opId=%p op.type=%d op.reqBuff=%p op.respSize=%d "
         "done",
         op->opId, op->type, op->reqBuff, op->respSize);
    if (op->connection->transport == TRANSPORT_NET) {
      if (op->type == flagcxProxyMsgConnect)
        __atomic_store_n(&op->connection->state, connConnected,
                         __ATOMIC_RELEASE);
    }

    /* if setup or connect is done, we should not return any error at this point
     * since flagcxSocketSend might already send the respBuff to the requester.
     * If we still choose to abort and close the connection, it can cause
     * segfault if the requester is using the respBuff. */

    flagcxProxyRpcResponseHeader resp = {op->opId, flagcxSuccess, op->respSize};

    // Send the opId for referencing async operation
    FLAGCXCHECK(flagcxSocketSend(op->connection->sock, &resp, sizeof(resp)));
    if (op->respSize) {
      // Send the response
      FLAGCXCHECK(
          flagcxSocketSend(op->connection->sock, op->respBuff, op->respSize));
    }

    asyncProxyOpDequeue(opHead, op);
    (*asyncOpCount)--;
    return flagcxSuccess;
  }

  return flagcxInProgress;
}

flagcxResult_t flagcxProxyCallAsync(struct flagcxHeteroComm *comm,
                                    struct flagcxProxyConnector *proxyConn,
                                    int type, void *reqBuff, int reqSize,
                                    int respSize, void *opId) {
  struct flagcxSocket *sock;
  flagcxResult_t ret = flagcxSuccess;
  struct flagcxProxyState *sharedProxyState = comm->proxyState;

  if (sharedProxyState->peerSocks == NULL)
    return flagcxInternalError;
  sock = &sharedProxyState->peerSocks[proxyConn->tpRank];

  FLAGCXCHECKGOTO(flagcxSocketSend(sock, &type, sizeof(int)), ret, error);
  FLAGCXCHECKGOTO(
      flagcxSocketSend(sock, &proxyConn->connection, sizeof(void *)), ret,
      error);
  FLAGCXCHECKGOTO(flagcxSocketSend(sock, &reqSize, sizeof(int)), ret, error);
  FLAGCXCHECKGOTO(flagcxSocketSend(sock, &respSize, sizeof(int)), ret, error);
  if (reqSize)
    FLAGCXCHECKGOTO(flagcxSocketSend(sock, reqBuff, reqSize), ret, error);

  // Send opId to proxy
  FLAGCXCHECKGOTO(flagcxSocketSend(sock, &opId, sizeof(opId)), ret, error);

  FLAGCXCHECK(expectedProxyResponseEnqueue(sharedProxyState, opId, respSize));
  return flagcxSuccess;
error:
  return ret;
}

static flagcxResult_t proxyServiceInitOp(int type, struct flagcxSocket *sock,
                                         struct flagcxProxyAsyncOp **opHead,
                                         flagcxHeteroComm_t comm,
                                         int *asyncOpCount) {
  struct flagcxProxyAsyncOp *asyncOp;
  FLAGCXCHECK(flagcxCalloc(&asyncOp, 1));

  asyncOp->type = type;
  FLAGCXCHECK(flagcxSocketRecv(sock, &asyncOp->connection, sizeof(void *)));

  FLAGCXCHECK(flagcxSocketRecv(sock, &asyncOp->reqSize, sizeof(int)));
  FLAGCXCHECK(flagcxSocketRecv(sock, &asyncOp->respSize, sizeof(int)));
  if (asyncOp->reqSize) {
    FLAGCXCHECK(flagcxCalloc(&asyncOp->reqBuff, asyncOp->reqSize));
    FLAGCXCHECK(flagcxSocketRecv(sock, asyncOp->reqBuff, asyncOp->reqSize));
  }

  // Store opId for completion response
  FLAGCXCHECK(flagcxSocketRecv(sock, &asyncOp->opId, sizeof(asyncOp->opId)));

  asyncOp->connection->sock = sock;
  if (asyncOp->respSize)
    FLAGCXCHECK(flagcxCalloc(&asyncOp->respBuff, asyncOp->respSize));

  FLAGCXCHECK(asyncProxyOpEnqueue(opHead, asyncOp));
  (*asyncOpCount)++;
  FLAGCXCHECK(proxyProgressAsync(opHead, asyncOp, asyncOpCount));
  return flagcxSuccess;
}

flagcxResult_t flagcxProxyCallBlocking(struct flagcxHeteroComm *comm,
                                       struct flagcxProxyConnector *proxyConn,
                                       int type, void *reqBuff, int reqSize,
                                       void *respBuff, int respSize) {
  // Alloc some memory to act as a handle
  flagcxResult_t res = flagcxSuccess;
  void *opId = malloc(1);

  FLAGCXCHECKGOTO(flagcxProxyCallAsync(comm, proxyConn, type, reqBuff, reqSize,
                                       respSize, opId),
                  res, fail);

  do {
    res = flagcxPollProxyResponse(comm, proxyConn, respBuff, opId);
  } while (res == flagcxInProgress);

exit:
  free(opId);
  return res;
fail:
  goto exit;
}

struct flagcxProxyKernelServiceArg {
  struct flagcxHeteroComm *comm;
  int contextId;
};

// Proxy init request/response for peer proxy connections
struct flagcxProxyInitReq {
  int transport;
  int send;
  int tpLocalRank;
  int tpRank;
  int sameProcess;
};

struct flagcxProxyInitResp {
  flagcxProxyConnection *connection;
};

flagcxResult_t flagcxProxyConnect(struct flagcxHeteroComm *comm, int transport,
                                  int send, int proxyRank,
                                  struct flagcxProxyConnector *proxyConn) {
  proxyConn->sameProcess = ((comm->peerInfo[proxyRank].hostHash ==
                             comm->peerInfo[comm->rank].hostHash) &&
                            (comm->peerInfo[proxyRank].pidHash ==
                             comm->peerInfo[comm->rank].pidHash))
                               ? 1
                               : 0;
  proxyConn->connection = NULL;
  proxyConn->tpRank = proxyRank;
  proxyConn->tpLocalRank = 0;

  // peerSocks must already be allocated and connected during init
  if (comm->proxyState->peerSocks == NULL)
    return flagcxInternalError;

  struct flagcxProxyInitReq req = {};
  req.transport = transport;
  req.send = send;
  req.tpLocalRank = comm->localRank;
  req.tpRank = comm->rank;
  req.sameProcess = proxyConn->sameProcess;

  // Mark initialized before the Init RPC so CallAsync uses peerSocks[tpRank]
  proxyConn->initialized = true;

  struct flagcxProxyInitResp resp = {};
  FLAGCXCHECK(flagcxProxyCallBlocking(comm, proxyConn, flagcxProxyMsgInit, &req,
                                      sizeof(req), &resp, sizeof(resp)));
  proxyConn->connection = resp.connection;
  INFO(FLAGCX_PROXY,
       "flagcxProxyConnect rank %d -> peer %d connection %p sameProcess %d",
       comm->rank, proxyRank, proxyConn->connection, proxyConn->sameProcess);
  return flagcxSuccess;
}

flagcxResult_t flagcxProxyInit(struct flagcxHeteroComm *comm) {
  INFO(FLAGCX_INIT, "rank=%d flagcxProxyInit called.", comm->rank);
  FLAGCXCHECK(flagcxSocketInit(&comm->proxyState->listenSock,
                               &bootstrapNetIfAddr, comm->magic,
                               flagcxSocketTypeProxy, NULL, 1));
  FLAGCXCHECK(flagcxSocketListen(&comm->proxyState->listenSock));

  comm->proxyState->cudaDev = comm->cudaDev;
  pthread_create(&comm->proxyState->thread, NULL, flagcxProxyService,
                 (void *)comm);
  pthread_create(&comm->proxyState->progressState.thread, NULL,
                 flagcxProxyProgress, comm->proxyState);
#ifdef COMPILE_KERNEL_HOST
  // Initialize synchronization primitives before creating threads
  pthread_mutex_init(&comm->proxyState->kernelState.initMutex, NULL);
  pthread_cond_init(&comm->proxyState->kernelState.initCond, NULL);
  comm->proxyState->kernelState.ready = 0;

  int nKernelProxies = flagcxParamKernelProxyParallelism();
  if (nKernelProxies < 1)
    nKernelProxies = 1;
  if (nKernelProxies > FLAGCX_DEVICE_CTA_COUNT)
    nKernelProxies = FLAGCX_DEVICE_CTA_COUNT;
  comm->proxyState->kernelState.contextCount = nKernelProxies;

  int nStarted = 0;
  for (int i = 0; i < nKernelProxies; i++) {
    flagcxProxyKernelServiceArg *arg = new flagcxProxyKernelServiceArg{comm, i};
    if (pthread_create(&comm->proxyState->kernelState.threads[i], NULL,
                       flagcxProxyKernelService, arg) != 0) {
      WARN("flagcxProxyInit: failed to create kernel proxy thread %d", i);
      delete arg;
      break;
    }
    nStarted++;
  }
  // Adjust contextCount to the number of threads actually started so the
  // cond-wait below and the stop/join loop use a consistent count.
  comm->proxyState->kernelState.contextCount = nStarted;

  // Wait for all started kernel proxy threads to finish initialization
  pthread_mutex_lock(&comm->proxyState->kernelState.initMutex);
  while (comm->proxyState->kernelState.ready < nStarted) {
    pthread_cond_wait(&comm->proxyState->kernelState.initCond,
                      &comm->proxyState->kernelState.initMutex);
  }
  pthread_mutex_unlock(&comm->proxyState->kernelState.initMutex);

  if (nStarted == 0) {
    WARN("flagcxProxyInit: no kernel proxy threads started");
    return flagcxSystemError;
  }
#endif

  comm->proxyState->initialized = 1;
  return flagcxSuccess;
}

void *flagcxProxyService(void *args) {
  int stop = 0;
  int asyncOpCount = 0;
  struct flagcxHeteroComm *comm = (struct flagcxHeteroComm *)args;
  struct flagcxProxyAsyncOp *opHead = NULL;
  struct flagcxProxyAsyncOp *list = NULL;
  flagcxResult_t res = flagcxSuccess;

  // Max connections: 1 listenSock + nRanks peer connections + 1 stop connection
  int maxConns = 1 + comm->nRanks + 1;
  struct pollfd *pollfds =
      (struct pollfd *)calloc(maxConns, sizeof(struct pollfd));
  struct flagcxSocket *connSocks = (struct flagcxSocket *)calloc(
      comm->nRanks + 1, sizeof(struct flagcxSocket));
  int nConns = 0;
  struct flagcxProxyConnection **allocatedConns = NULL;
  int nAllocatedConns = 0;
  FLAGCXCHECKGOTO(flagcxCalloc(&allocatedConns, comm->nRanks), res, out);

  // Set device context
  FLAGCXCHECKGOTO(deviceAdaptor->setDevice(comm->cudaDev), res, out);

  // pollfds[0] = listenSock, pollfds[1..] = accepted connections
  pollfds[0].fd = comm->proxyState->listenSock.fd;
  pollfds[0].events = POLLIN;

  while (!stop || (stop && opHead)) {
    int nfds = 1 + nConns;
    int ret;
    do {
      ret = poll(pollfds, nfds, asyncOpCount ? 0 : 500);
    } while (ret < 0 && errno == EINTR);
    if (ret < 0) {
      WARN("[Proxy Service] Poll failed: %s", strerror(errno));
      break;
    }

    // Progress all ops
    list = opHead;
    while (list) {
      struct flagcxProxyAsyncOp *opNext = list->next;
      res = proxyProgressAsync(&opHead, list, &asyncOpCount);
      if (res == flagcxSuccess || res == flagcxInProgress) {
        list = opNext;
      } else {
        WARN("[Service thread] Error encountered progressing operation with "
             "res=%d",
             res);
        break;
      }
    }

    // Helper lambda to process incoming data on a socket
    auto processSocket = [&](struct flagcxSocket *sock) -> bool {
      int type;
      int closed = 0;
      res = flagcxSocketTryRecv(sock, &type, sizeof(int), &closed,
                                false /*blocking*/);
      if (res != flagcxSuccess && res != flagcxInProgress) {
        WARN("[Service thread] Could not receive type, res=%u closed=%d", res,
             closed);
        return false;
      } else if (closed) {
        INFO(FLAGCX_PROXY, "[Service thread] Connection closed");
        return false;
      } else if (res == flagcxSuccess) {
        if (type == flagcxProxyMsgStop) {
          stop = 1;
          return true;
        } else if (type == flagcxProxyMsgInit) {
          // Peer proxy init: create a new flagcxProxyConnection
          struct flagcxProxyAsyncOp *asyncOp = NULL;
          flagcxResult_t initRes = flagcxSuccess;
          void *dummyConn = NULL;
          struct flagcxProxyConnection *newConn = NULL;
          struct flagcxProxyInitReq *req = NULL;
          flagcxProxyRpcResponseHeader resp = {};

          FLAGCXCHECKGOTO(flagcxCalloc(&asyncOp, 1), initRes, initFail);
          asyncOp->type = type;

          // For Init, connection pointer sent is NULL — read and discard it
          FLAGCXCHECKGOTO(flagcxSocketRecv(sock, &dummyConn, sizeof(void *)),
                          initRes, initFail);
          FLAGCXCHECKGOTO(
              flagcxSocketRecv(sock, &asyncOp->reqSize, sizeof(int)), initRes,
              initFail);
          FLAGCXCHECKGOTO(
              flagcxSocketRecv(sock, &asyncOp->respSize, sizeof(int)), initRes,
              initFail);

          // Validate sizes
          if (asyncOp->reqSize != (int)sizeof(struct flagcxProxyInitReq)) {
            WARN("[Service thread] Invalid reqSize %d for Init, expected %zu",
                 asyncOp->reqSize, sizeof(struct flagcxProxyInitReq));
            initRes = flagcxInvalidArgument;
            goto initFail;
          }
          if (asyncOp->respSize != (int)sizeof(struct flagcxProxyInitResp)) {
            WARN("[Service thread] Invalid respSize %d for Init, expected %zu",
                 asyncOp->respSize, sizeof(struct flagcxProxyInitResp));
            initRes = flagcxInvalidArgument;
            goto initFail;
          }

          FLAGCXCHECKGOTO(flagcxCalloc(&asyncOp->reqBuff, asyncOp->reqSize),
                          initRes, initFail);
          FLAGCXCHECKGOTO(
              flagcxSocketRecv(sock, asyncOp->reqBuff, asyncOp->reqSize),
              initRes, initFail);
          FLAGCXCHECKGOTO(
              flagcxSocketRecv(sock, &asyncOp->opId, sizeof(asyncOp->opId)),
              initRes, initFail);

          // Create a new connection for this peer
          FLAGCXCHECKGOTO(flagcxCalloc(&newConn, 1), initRes, initFail);
          req = (struct flagcxProxyInitReq *)asyncOp->reqBuff;
          newConn->transport = req->transport;
          newConn->send = req->send;
          newConn->tpLocalRank = req->tpLocalRank;
          newConn->sameProcess = req->sameProcess;
          newConn->cudaDev = comm->cudaDev;
          newConn->sock = sock;

          asyncOp->connection = newConn;
          allocatedConns[nAllocatedConns++] = newConn;
          FLAGCXCHECKGOTO(flagcxCalloc(&asyncOp->respBuff, asyncOp->respSize),
                          initRes, initFail);

          // Fill response with the connection pointer
          if (asyncOp->respSize >= (int)sizeof(void *)) {
            memcpy(asyncOp->respBuff, &newConn, sizeof(void *));
          }

          INFO(FLAGCX_PROXY,
               "[Service thread] Peer proxy init: transport=%d send=%d "
               "tpLocalRank=%d sameProcess=%d conn=%p",
               req->transport, req->send, req->tpLocalRank, req->sameProcess,
               newConn);

          // Send response immediately
          resp.opId = asyncOp->opId;
          resp.res = flagcxSuccess;
          resp.respSize = asyncOp->respSize;
          FLAGCXCHECKGOTO(flagcxSocketSend(sock, &resp, sizeof(resp)), initRes,
                          initFail);
          if (asyncOp->respSize) {
            FLAGCXCHECKGOTO(
                flagcxSocketSend(sock, asyncOp->respBuff, asyncOp->respSize),
                initRes, initFail);
          }
          free(asyncOp->reqBuff);
          free(asyncOp->respBuff);
          free(asyncOp);
          return true;

        initFail:
          if (asyncOp) {
            free(asyncOp->reqBuff);
            free(asyncOp->respBuff);
            free(asyncOp);
          }
          return false;
        } else if (proxyMatchOpType(type)) {
          res = proxyServiceInitOp(type, sock, &opHead, comm, &asyncOpCount);
          if (res != flagcxSuccess) {
            WARN("[Service thread] Error encountered initializing operation "
                 "with res=%d",
                 res);
            return false;
          }
          return true;
        } else {
          INFO(FLAGCX_PROXY, "[Service thread] Unknown command %d from rank %d",
               type, comm->rank);
          return false;
        }
      }
      return true;
    };

    // Check listenSock for new connections (nRanks peers + 1 stop connection)
    if (pollfds[0].revents & POLLIN) {
      if (nConns < comm->nRanks + 1) {
        struct flagcxSocket *newSock = &connSocks[nConns];
        FLAGCXCHECKGOTO(flagcxSocketInit(newSock), res, out);
        res = flagcxSocketAccept(newSock, &comm->proxyState->listenSock);
        if (res == flagcxSuccess) {
          pollfds[1 + nConns].fd = newSock->fd;
          pollfds[1 + nConns].events = POLLIN;
          nConns++;
          INFO(FLAGCX_PROXY, "[Service thread] Accepted connection %d", nConns);
        }
      }
    }

    // Check all connected sockets
    for (int i = 0; i < nConns && !stop; i++) {
      if (pollfds[1 + i].revents & POLLIN) {
        if (!processSocket(&connSocks[i])) {
          // Connection closed — close socket and mark fd as invalid
          flagcxSocketClose(&connSocks[i]);
          pollfds[1 + i].fd = -1;
        }
      }
    }

    if (stop && !opHead)
      break;
  }
out:
  // Stop progress thread before freeing any resource
  pthread_mutex_lock(&comm->proxyState->mutex);
  comm->proxyState->progressState.stop = 1;
  pthread_cond_signal(&comm->proxyState->cond);
  pthread_mutex_unlock(&comm->proxyState->mutex);
  pthread_join(comm->proxyState->progressState.thread, nullptr);
#ifdef COMPILE_KERNEL_HOST
  // Stop all kernel threads and cleanup
  for (int i = 0; i < comm->proxyState->kernelState.contextCount; i++) {
    pthread_join(comm->proxyState->kernelState.threads[i], nullptr);
  }
  pthread_mutex_destroy(&comm->proxyState->kernelState.initMutex);
  pthread_cond_destroy(&comm->proxyState->kernelState.initCond);
#endif

  // Free P2P resources in proxy thread (CUDA resources must be freed in the
  // same thread where they were created)
  for (int peer = 0; peer < comm->nRanks; peer++) {
    for (int c = 0; c < MAXCHANNELS; c++) {
      if (comm->channels[c].peers[peer]->recv[0].connected == 1) {
        struct flagcxConnector *conn = comm->channels[c].peers[peer]->recv;
        if (conn->proxyConn.connection->transport == TRANSPORT_P2P) {
          struct flagcxP2pResources *resources =
              (struct flagcxP2pResources *)
                  conn->proxyConn.connection->transportResources;
          flagcxP2pRecvProxyFree(resources);
        }
      }
      if (comm->channels[c].peers[peer]->send[0].connected == 1) {
        struct flagcxConnector *conn = comm->channels[c].peers[peer]->send;
        if (conn->proxyConn.connection->transport == TRANSPORT_P2P) {
          struct flagcxP2pResources *resources =
              (struct flagcxP2pResources *)
                  conn->proxyConn.connection->transportResources;
          flagcxP2pSendProxyFree(resources);
        }
      }
    }
  }

  // Free allocated proxy connections
  for (int i = 0; i < nAllocatedConns; i++)
    free(allocatedConns[i]);
  free(allocatedConns);

  // Close sockets (skip already-closed ones marked with fd = -1)
  for (int i = 0; i < nConns; i++) {
    if (pollfds[1 + i].fd != -1) {
      flagcxSocketClose(&connSocks[i]);
    }
  }
  flagcxSocketClose(&comm->proxyState->listenSock);
  free(pollfds);
  free(connSocks);

  // Dequeue unhandled ops
  list = opHead;
  while (list) {
    struct flagcxProxyAsyncOp *opNext = list->next;
    asyncProxyOpDequeue(&opHead, list);
    list = opNext;
  }

  INFO(FLAGCX_PROXY,
       "[Service thread] Wait for progress thread joined and free resources");
  return NULL;
}

void *flagcxProxyKernelService(void *args) {
  int groupCount = 0;
  int termCount = 0;
  flagcxDeviceTrigger_t ptr = NULL;
  flagcxFifo_t fifo = NULL;
  flagcxStream_t stream = NULL;
  flagcxProxyKernelServiceArg *arg = (flagcxProxyKernelServiceArg *)args;
  struct flagcxHeteroComm *comm = arg->comm;
  int contextId = arg->contextId;
  delete arg;
  flagcxResult_t res = flagcxSuccess;

  auto validateOneSidedPeer = [](struct flagcxHeteroComm *comm,
                                 int peerRank) -> flagcxResult_t {
    struct flagcxOneSideHandleInfo *meshH =
        (comm->oneSideHandleCount > 0) ? comm->oneSideHandles[0] : NULL;
    if (meshH == NULL)
      return flagcxNotSupported;
    if (peerRank < 0 || peerRank >= comm->nRanks)
      return flagcxInvalidArgument;

    // Check full-mesh connection exists for this peer (including self-loopback)
    if (meshH->fullSendComms == NULL || meshH->fullSendComms[peerRank] == NULL)
      return flagcxNotSupported;

    return flagcxSuccess;
  };

  // Set device context
  FLAGCXCHECKGOTO(deviceAdaptor->setDevice(comm->cudaDev), res, out);

  // Create FIFO for this thread
  comm->proxyState->kernelState.fifos[contextId] = new flagcxFifo();
  FLAGCXCHECKGOTO(
      comm->proxyState->kernelState.fifos[contextId]->flagcxFifoInit(), res,
      out);
  fifo = comm->proxyState->kernelState.fifos[contextId];
  FLAGCXCHECKGOTO(
      deviceAdaptor->hostGetDevicePointer(
          &comm->fifoBuffers[contextId],
          (void *)comm->proxyState->kernelState.fifos[contextId]->buffer),
      res, out);

  // Create a dedicated stream
  FLAGCXCHECKGOTO(deviceAdaptor->streamCreate(&stream), res, out);
  INFO(FLAGCX_P2P, "rank %d p2p stream %lu", comm->rank, (uintptr_t)stream);

  // Allocate trigger structure
  FLAGCXCHECKGOTO(flagcxCalloc(&ptr, sizeof(flagcxDeviceTrigger)), res, out);

  // Signal that initialization is complete
  pthread_mutex_lock(&comm->proxyState->kernelState.initMutex);
  comm->proxyState->kernelState.ready++;
  pthread_cond_broadcast(&comm->proxyState->kernelState.initCond);
  pthread_mutex_unlock(&comm->proxyState->kernelState.initMutex);

  while (true) {
    if (comm->proxyState->kernelState.stop == 1)
      break;
    dequeue(fifo->buffer, ptr);
    if ((ptr->getPrim() == flagcxDevicePrimSend ||
         ptr->getPrim() == flagcxDevicePrimRecv) &&
        ptr->getAddr() == 0) {
      sched_yield();
      continue;
    }
    switch (ptr->getPrim()) {
      case flagcxDevicePrimSend:
        if (groupCount == 0) {
          res = flagcxHeteroGroupStart();
          TRACE(FLAGCX_P2P,
                "rank=%d flagcxHeteroGroupStart called by proxyKernelService.",
                comm->rank);
          groupCount++;
        }
        TRACE(FLAGCX_P2P,
              "rank=%d flagcxDevicePrimSend called by proxyKernelService.",
              comm->rank);
        res = flagcxHeteroSend((const void *)(uintptr_t)(ptr->getAddr()),
                               ptr->getCount(),
                               (flagcxDataType_t)(ptr->getDatatype()),
                               ptr->getPeerRank(), comm, stream);
        break;
      case flagcxDevicePrimRecv:
        if (groupCount == 0) {
          res = flagcxHeteroGroupStart();
          TRACE(FLAGCX_P2P,
                "rank=%d flagcxHeteroGroupStart called by proxyKernelService.",
                comm->rank);
          groupCount++;
        }
        TRACE(FLAGCX_P2P,
              "rank=%d flagcxDevicePrimRecv called by proxyKernelService.",
              comm->rank);
        res = flagcxHeteroRecv((void *)(uintptr_t)(ptr->getAddr()),
                               ptr->getCount(),
                               (flagcxDataType_t)(ptr->getDatatype()),
                               ptr->getPeerRank(), comm, stream);
        break;
      case flagcxDevicePrimTerm: {
        termCount++;
        int totalCoops = (int)ptr->getTotalCoops();
        TRACE(FLAGCX_P2P,
              "rank=%d flagcxDevicePrimTerm called by proxyKernelService "
              "groupCount=%d termCount=%d/%d.",
              comm->rank, groupCount, termCount, totalCoops);
        if (groupCount > 0 && termCount >= totalCoops) {
          res = flagcxHeteroGroupEnd();
          TRACE(FLAGCX_P2P,
                "rank=%d flagcxHeteroGroupEnd called by proxyKernelService.",
                comm->rank);
          groupCount--;
          termCount = 0;
        }
        break;
      }
      case flagcxDevicePrimPut: {
        TRACE(FLAGCX_P2P,
              "rank=%d flagcxDevicePrimPut called by proxyKernelService.",
              comm->rank);
        int peerRank = (int)ptr->getPeerRank();
        res = validateOneSidedPeer(comm, peerRank);
        if (res != flagcxSuccess)
          break;
        int srcMrIdx = (int)ptr->getSrcMrIdx();
        int dstMrIdx = (int)ptr->getDstMrIdx();
        size_t srcOffset = (size_t)ptr->getSrcOffset();
        size_t dstOffset = (size_t)ptr->getDstOffset();
        size_t size = (size_t)ptr->getSize();
        res = flagcxHeteroPut(comm, peerRank, srcOffset, dstOffset, size,
                              srcMrIdx, dstMrIdx);
        break;
      }
      case flagcxDevicePrimSignal: {
        TRACE(FLAGCX_P2P,
              "rank=%d flagcxDevicePrimSignal called by proxyKernelService.",
              comm->rank);
        uint64_t bufType = ptr->getBufferType();
        int signalIdx = (int)ptr->getSignalIdx();
        uint64_t signalValue = ptr->getSignalValue();
        size_t signalOff = (size_t)signalIdx * sizeof(uint64_t);

        if (bufType == 0) {
          // Signal buffer: RDMA FETCH_AND_ADD to peer's signalBuffer
          int peerRank = (int)ptr->getPeerRank();
          res = validateOneSidedPeer(comm, peerRank);
          if (res != flagcxSuccess)
            break;
          if (comm->signalHandle == NULL) {
            WARN("flagcxDevicePrimSignal: signal handles not initialized "
                 "for this comm — call flagcxOneSideSignalRegister() before "
                 "use");
            res = flagcxInternalError;
            break;
          }
          res = flagcxHeteroPutSignal(comm, peerRank, 0, 0, 0, signalOff, 0, 0,
                                      signalValue);
        } else {
          // Counter buffer: local CPU atomic increment (no network operation)
          flagcxDevComm_t dc = comm->devCommHandle;
          if (dc == NULL || dc->counterBuffer == NULL) {
            WARN("flagcxDevicePrimSignal: counterBuffer not initialized");
            res = flagcxInternalError;
            break;
          }
          uint64_t *counterPtr = (uint64_t *)dc->counterBuffer + signalIdx;
          __atomic_fetch_add(counterPtr, signalValue, __ATOMIC_RELAXED);
        }
        break;
      }
      case flagcxDevicePrimWaitSignal: {
        TRACE(
            FLAGCX_P2P,
            "rank=%d flagcxDevicePrimWaitSignal called by proxyKernelService.",
            comm->rank);
        uint64_t wsBufType = ptr->getBufferType(); // 0=signal, 1=counter
        int wsSignalIdx = (int)ptr->getSignalIdx();
        uint32_t wsExpected = (uint32_t)ptr->getExpectedValue();
        size_t wsSignalOff = (size_t)wsSignalIdx * sizeof(uint64_t);
        flagcxDevComm_t dc = comm->devCommHandle;
        if (dc == NULL) {
          WARN("flagcxDevicePrimWaitSignal: devComm not initialized");
          res = flagcxInternalError;
          break;
        }
        // Select target buffer based on buffer type
        uint64_t *targetBuffer =
            (wsBufType == 0) ? dc->signalBuffer : dc->counterBuffer;
        if (targetBuffer == NULL) {
          WARN("flagcxDevicePrimWaitSignal: %s buffer not allocated",
               wsBufType == 0 ? "signal" : "counter");
          res = flagcxInternalError;
          break;
        }
        void *waitAddr = (void *)((char *)targetBuffer + wsSignalOff);
        res = deviceAdaptor->streamWaitValue64(stream, waitAddr,
                                               (uint64_t)wsExpected, 0);
        break;
      }
      case flagcxDevicePrimPutSignal: {
        TRACE(FLAGCX_P2P,
              "rank=%d flagcxDevicePrimPutSignal called by proxyKernelService.",
              comm->rank);
        int peerRank = (int)ptr->getPeerRank();
        res = validateOneSidedPeer(comm, peerRank);
        if (res != flagcxSuccess)
          break;
        int srcMrIdx = (int)ptr->getSrcMrIdx();
        int dstMrIdx = (int)ptr->getDstMrIdx();
        size_t srcOffset = (size_t)ptr->getSrcOffset();
        size_t dstOffset = (size_t)ptr->getDstOffset();
        size_t size = (size_t)ptr->getSize();
        int signalIdx = (int)ptr->getSignalIdx();
        uint64_t signalValue = ptr->getSignalValue();
        size_t signalOff = (size_t)signalIdx * sizeof(uint64_t);
        if (comm->signalHandle == NULL) {
          WARN("flagcxDevicePrimPutSignal: signal handles not initialized "
               "for this comm — call flagcxOneSideSignalRegister() before use");
          res = flagcxInternalError;
          break;
        }
        res = flagcxHeteroPutSignal(comm, peerRank, srcOffset, dstOffset, size,
                                    signalOff, srcMrIdx, dstMrIdx, signalValue);
        break;
      }
      case flagcxDevicePrimPutValue: {
        int peerRank = (int)ptr->getPeerRank();
        int dstMrIdx = (int)ptr->getDstMrIdx();
        size_t dstOffset = (size_t)ptr->getDstOffset();
        uint64_t value = ptr->getValue();
        res = flagcxHeteroPutValue(comm, peerRank, value, dstOffset, dstMrIdx);
        break;
      }
      case flagcxDevicePrimGet: {
        TRACE(FLAGCX_P2P,
              "rank=%d flagcxDevicePrimGet called by proxyKernelService.",
              comm->rank);
        int peerRank = (int)ptr->getPeerRank();
        res = validateOneSidedPeer(comm, peerRank);
        if (res != flagcxSuccess)
          break;
        int srcMrIdx = (int)ptr->getSrcMrIdx();
        int dstMrIdx = (int)ptr->getDstMrIdx();
        size_t srcOffset = (size_t)ptr->getSrcOffset();
        size_t dstOffset = (size_t)ptr->getDstOffset();
        size_t size = (size_t)ptr->getSize();
        res = flagcxHeteroGet(comm, peerRank, srcOffset, dstOffset, size,
                              srcMrIdx, dstMrIdx);
        break;
      }
      case flagcxDevicePrimWait:
        TRACE(FLAGCX_P2P,
              "rank=%d flagcxDevicePrimWait called by proxyKernelService.",
              comm->rank);
        deviceAdaptor->streamSynchronize(stream);
        break;
      case flagcxDevicePrimBarrierSignal: {
        // Inter-node barrier: RDMA ATOMIC FETCH_AND_ADD to each peer's
        // interSignalFlagsHost counter via iputSignal (signal-only, size=0).
        flagcxDevComm_t dc = comm->devCommHandle;
        if (dc && dc->nInterPeers > 0 && dc->barrierHandleInfo) {
          uint32_t ctaIdx = (uint32_t)ptr->getAddr();
          struct flagcxNetAdaptor *net =
              (struct flagcxNetAdaptor *)dc->netAdaptorPtr;
          size_t signalOff = (size_t)ctaIdx * sizeof(uint64_t);

          void *reqs[FLAGCX_MAX_INTER_PEERS];
          for (int p = 0; p < dc->nInterPeers; p++) {
            reqs[p] = nullptr;
            net->iputSignal(dc->signalSendComms[p], 0, 0, 0, comm->rank,
                            dc->interPeerRanks[p], NULL, NULL,
                            (uint64_t)signalOff, (void **)dc->barrierHandleInfo,
                            1, &reqs[p]);
          }
          for (int p = 0; p < dc->nInterPeers; p++) {
            if (reqs[p]) {
              int done = 0;
              while (!done) {
                net->test(reqs[p], &done, nullptr);
              }
            }
          }
        }
        break;
      }
      default:
        break;
    }
    // Mark item as consumed AFTER processing
    __sync_synchronize();
    ((volatile uint64_t *)fifo->buffer)[flagcxFifoIdxConsumed]++;
    if (res != flagcxSuccess)
      break;
  }
out:
  // destroy stream (only if created)
  if (stream != nullptr) {
    deviceAdaptor->streamSynchronize(stream);
    deviceAdaptor->streamDestroy(stream);
  }
  // deallocate trigger structure (only if allocated)
  free(ptr);
  // destroy fifo (only if created)
  if (comm->proxyState->kernelState.fifos[contextId] != nullptr) {
    comm->proxyState->kernelState.fifos[contextId]->flagcxFifoDestroy();
    delete comm->proxyState->kernelState.fifos[contextId];
    comm->proxyState->kernelState.fifos[contextId] = nullptr;
  }
  comm->fifoBuffers[contextId] = NULL;
  return NULL;
}

flagcxResult_t flagcxProxyFree(struct flagcxHeteroComm *comm) {
  for (int peer = 0; peer < comm->nRanks; peer++) {
    for (int c = 0; c < MAXCHANNELS; c++) {
      if (comm->channels[c].peers[peer]->recv[0].connected == 1) {
        struct flagcxConnector *conn = comm->channels[c].peers[peer]->recv;
        int transport = conn->proxyConn.connection->transport;

        if (transport == TRANSPORT_NET) {
          struct recvNetResources *resources =
              (struct recvNetResources *)
                  conn->proxyConn.connection->transportResources;
          flagcxRecvProxyFree(resources);
        }
      }
      if (comm->channels[c].peers[peer]->send[0].connected == 1) {
        struct flagcxConnector *conn = comm->channels[c].peers[peer]->send;
        int transport = conn->proxyConn.connection->transport;

        if (transport == TRANSPORT_NET) {
          struct sendNetResources *resources =
              (struct sendNetResources *)
                  conn->proxyConn.connection->transportResources;
          flagcxSendProxyFree(resources);
        }
      }
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxProxyDestroy(struct flagcxHeteroComm *comm) {
  if (comm->proxyState->initialized == 1) {
    INFO(FLAGCX_PROXY, "flagcxProxyDestroy: sending stop to service thread...");
    // Send stop via a temporary socket to own listenSock (like NCCL)
    struct flagcxSocket sock;
    int type = flagcxProxyMsgStop;
    FLAGCXCHECK(flagcxSocketInit(&sock, &comm->proxyState->listenSock.addr,
                                 comm->magic, flagcxSocketTypeProxy));
    if (flagcxSocketConnect(&sock) == flagcxSuccess) {
      int ready = 0;
      while (!ready) {
        (void)flagcxSocketReady(&sock, &ready);
      }
      (void)flagcxSocketSend(&sock, &type, sizeof(int));
    }
    (void)flagcxSocketClose(&sock);
    comm->proxyState->kernelState.stop = 1;
    INFO(FLAGCX_PROXY, "flagcxProxyDestroy: joining service thread...");
    pthread_join(comm->proxyState->thread, nullptr);
    INFO(FLAGCX_PROXY, "flagcxProxyDestroy: service thread joined, freeing...");
    flagcxProxyFree(comm);
    INFO(FLAGCX_PROXY, "flagcxProxyDestroy: done");
  }
  return flagcxSuccess;
}
