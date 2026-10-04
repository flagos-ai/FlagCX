/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE-NCCL.txt for license information
 ************************************************************************/

#include "proxy.h"
#include "adaptor.h"
#include "bootstrap.h"
#include "comm.h"
#include "device_api/completion_word.h"
#include "device_api/flagcx_device.h" // flagcxDevCommInternal, devComm
#include "flagcx_hetero.h"
#include "flagcx_kernel.h" // FLAGCX_DEVICE_CTA_COUNT
#include "info.h"
#include "kernel_proxy_transport.h"
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
#include <time.h>
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

static bool proxyCleanupOpType(int type) {
  return type == flagcxProxyMsgDeregister || type == flagcxProxyMsgDeregMr;
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

static flagcxResult_t asyncProxyOpEnqueue(struct flagcxProxyLocalPeer *peer,
                                          flagcxProxyAsyncOp *newOp) {
  flagcxProxyAsyncOp *list = peer->asyncOps;
  if (list == NULL) {
    peer->asyncOps = newOp;
  } else {
    while (list->next)
      list = list->next;
    list->next = newOp;
    newOp->prev = list;
  }
  return flagcxSuccess;
}

static flagcxResult_t asyncProxyOpDequeue(struct flagcxProxyLocalPeer *peer,
                                          flagcxProxyAsyncOp *op) {
  if (peer->asyncOps == op)
    peer->asyncOps = op->next;
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

flagcxResult_t
flagcxProxyRecordConnectionError(struct flagcxProxyConnection *connection,
                                 flagcxResult_t result) {
  if (connection == NULL || result == flagcxSuccess ||
      result == flagcxInProgress)
    return result;

  flagcxResult_t expected = flagcxSuccess;
  __atomic_compare_exchange_n(&connection->result, &expected, result, false,
                              __ATOMIC_RELEASE, __ATOMIC_RELAXED);
  __atomic_store_n(&connection->state, connFailed, __ATOMIC_RELEASE);
  return result;
}

flagcxResult_t
flagcxProxyGetConnectionError(struct flagcxProxyConnection *connection) {
  if (connection == NULL)
    return flagcxInvalidArgument;
  if (__atomic_load_n(&connection->state, __ATOMIC_ACQUIRE) != connFailed)
    return flagcxSuccess;

  flagcxResult_t result =
      __atomic_load_n(&connection->result, __ATOMIC_ACQUIRE);
  return result == flagcxSuccess ? flagcxInternalError : result;
}

// Complete every parsed control RPC exactly once. Permanent setup/connect
// failures still need a response so the caller cannot wait forever or treat a
// half-built connection as usable.
static flagcxResult_t proxyServiceCompleteOp(struct flagcxProxyLocalPeer *peer,
                                             struct flagcxProxyAsyncOp *op,
                                             int *asyncOpCount,
                                             flagcxResult_t result) {
  if (result != flagcxSuccess && op->respBuff != NULL && op->respSize > 0)
    memset(op->respBuff, 0, op->respSize);

  flagcxProxyRpcResponseHeader resp = {op->opId, result, op->respSize};
  flagcxResult_t sendResult =
      flagcxSocketSend(&peer->sock, &resp, sizeof(resp));
  if (sendResult == flagcxSuccess && op->respSize > 0)
    sendResult = flagcxSocketSend(&peer->sock, op->respBuff, op->respSize);

  asyncProxyOpDequeue(peer, op);
  (*asyncOpCount)--;
  return sendResult;
}

// ============================================================
// Proxy Init Request/Response (forward declarations for connection pool)
// ============================================================

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

static flagcxResult_t
flagcxNetProxyConnect(struct flagcxProxyConnection *connection,
                      struct flagcxProxyState *proxyState, void *reqBuff,
                      int reqSize, void *respBuff, int respSize, int *done);
static flagcxResult_t
flagcxNetProxyRegister(struct flagcxProxyConnection *connection,
                       struct flagcxProxyState *proxyState, void *reqBuff,
                       int reqSize, void *respBuff, int respSize, int *done);
static flagcxResult_t
flagcxNetProxyDeregister(struct flagcxProxyConnection *connection,
                         struct flagcxProxyState *proxyState, void *reqBuff,
                         int reqSize, int *done);

static struct flagcxTransportComm flagcxP2pSendTransportComm = {
    NULL,
    NULL,
    NULL,
    NULL,
    flagcxP2pSendProxySetup,
    flagcxP2pSendProxyConnect,
    NULL,
    NULL,
    flagcxP2pProxyRegister,
    flagcxP2pProxyDeregister,
    flagcxP2pPrepareProxyOp,
    flagcxP2pProgressProxyOp,
    flagcxP2pCleanupProxyConnection,
};

static struct flagcxTransportComm flagcxP2pRecvTransportComm = {
    NULL,
    NULL,
    NULL,
    NULL,
    flagcxP2pRecvProxySetup,
    flagcxP2pRecvProxyConnect,
    NULL,
    NULL,
    flagcxP2pProxyRegister,
    flagcxP2pProxyDeregister,
    flagcxP2pPrepareProxyOp,
    flagcxP2pProgressProxyOp,
    flagcxP2pCleanupProxyConnection,
};

static struct flagcxTransportComm flagcxNetSendTransportComm = {
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    flagcxNetProxyConnect,
    NULL,
    NULL,
    flagcxNetProxyRegister,
    flagcxNetProxyDeregister,
    flagcxNetPrepareProxyOp,
    flagcxNetProgressProxyOp,
    flagcxNetCleanupProxyConnection,
};

static struct flagcxTransportComm flagcxNetRecvTransportComm = {
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    flagcxNetProxyConnect,
    NULL,
    NULL,
    flagcxNetProxyRegister,
    flagcxNetProxyDeregister,
    flagcxNetPrepareProxyOp,
    flagcxNetProgressProxyOp,
    flagcxNetCleanupProxyConnection,
};

static struct flagcxTransportComm *flagcxProxyTransportComm(int transport,
                                                            int send) {
  if (transport == TRANSPORT_P2P)
    return send ? &flagcxP2pSendTransportComm : &flagcxP2pRecvTransportComm;
  if (transport == TRANSPORT_NET)
    return send ? &flagcxNetSendTransportComm : &flagcxNetRecvTransportComm;
  return NULL;
}

// ============================================================
// Connection Pool
// ============================================================

#define FLAGCX_PROXY_CONN_POOL_SIZE_POW2 7
#define FLAGCX_PROXY_CONN_POOL_SIZE (1 << (FLAGCX_PROXY_CONN_POOL_SIZE_POW2))
#define FLAGCX_PROXY_CONN_POOL_MASK ((FLAGCX_PROXY_CONN_POOL_SIZE)-1)

struct flagcxProxyConnectionPool {
  struct flagcxProxyConnection **pools;
  int banks;
  int offset;
};

static flagcxResult_t
flagcxProxyNewConnection(struct flagcxProxyConnectionPool *pool, int *id) {
  if (pool->offset == FLAGCX_PROXY_CONN_POOL_SIZE) {
    FLAGCXCHECK(flagcxRealloc(&pool->pools, pool->banks, pool->banks + 1));
    FLAGCXCHECK(
        flagcxCalloc(pool->pools + pool->banks, FLAGCX_PROXY_CONN_POOL_SIZE));
    pool->banks++;
    pool->offset = 0;
  }
  *id = ((pool->banks - 1) << FLAGCX_PROXY_CONN_POOL_SIZE_POW2) + pool->offset;
  pool->offset++;
  return flagcxSuccess;
}

static flagcxResult_t
flagcxProxyGetConnection(struct flagcxProxyConnectionPool *pool, int id,
                         struct flagcxProxyConnection **conn) {
  int bank = id >> FLAGCX_PROXY_CONN_POOL_SIZE_POW2;
  int offset = id & FLAGCX_PROXY_CONN_POOL_MASK;
  if ((id < 0) || (pool->pools == NULL) || (bank >= pool->banks) ||
      (pool->pools[bank] == NULL))
    return flagcxInternalError;
  *conn = pool->pools[bank] + offset;
  return flagcxSuccess;
}

static flagcxResult_t
proxyConnInit(struct flagcxProxyLocalPeer *peer,
              struct flagcxProxyConnectionPool *connectionPool,
              struct flagcxHeteroComm *comm, struct flagcxProxyInitReq *req,
              struct flagcxProxyInitResp *resp,
              struct flagcxProxyConnection **connection) {
  int id;
  FLAGCXCHECK(flagcxProxyNewConnection(connectionPool, &id));
  FLAGCXCHECK(flagcxProxyGetConnection(connectionPool, id, connection));

  (*connection)->sock = &peer->sock;
  (*connection)->transport = req->transport;
  (*connection)->send = req->send;
  (*connection)->tcomm = flagcxProxyTransportComm(req->transport, req->send);
  if ((*connection)->tcomm == NULL)
    return flagcxNotSupported;
  (*connection)->tpLocalRank = req->tpLocalRank;
  (*connection)->sameProcess = req->sameProcess;
  (*connection)->cudaDev = comm->cudaDev;
  peer->tpLocalRank = req->tpLocalRank;
  peer->tpRank = req->tpRank;

  resp->connection = *connection;

  INFO(FLAGCX_PROXY,
       "[Service thread] New proxy %s connection %d from local rank %d, "
       "transport %d",
       (*connection)->send ? "send" : "recv", id, (*connection)->tpLocalRank,
       (*connection)->transport);
  __atomic_store_n(&(*connection)->result, flagcxSuccess, __ATOMIC_RELEASE);
  __atomic_store_n(&(*connection)->state, connInitialized, __ATOMIC_RELEASE);
  return flagcxSuccess;
}

static flagcxResult_t
proxyFreeConnection(struct flagcxProxyConnection *connection,
                    struct flagcxHeteroComm *comm, bool closeP2pImports) {
  (void)comm;
  if (connection == NULL || connection->tcomm == NULL ||
      connection->tcomm->cleanupProxyConnection == NULL)
    return flagcxSuccess;
  return connection->tcomm->cleanupProxyConnection(
      connection, closeP2pImports ? flagcxTransportCleanupCloseImports
                                  : flagcxTransportCleanupReleaseResources);
}

static void flagcxProxyCaptureCleanupResult(flagcxResult_t nextResult,
                                            flagcxResult_t *firstResult) {
  if (nextResult != flagcxSuccess && nextResult != flagcxInProgress &&
      *firstResult == flagcxSuccess)
    *firstResult = nextResult;
}

static flagcxResult_t
flagcxProxyFreeConnections(struct flagcxProxyConnectionPool *pool,
                           struct flagcxHeteroComm *comm) {
  flagcxResult_t result = flagcxSuccess;

  // Phase 1 closes every imported P2P FIFO and publishes the peer-visible ACK.
  // Running this pass across the whole pool before freeing any exported FIFO
  // prevents two service threads from waiting on each other in recv cleanup.
  for (int b = 0; b < pool->banks; b++) {
    int max = b == pool->banks - 1 ? pool->offset : FLAGCX_PROXY_CONN_POOL_SIZE;
    for (int i = 0; i < max; i++) {
      struct flagcxProxyConnection *connection = pool->pools[b] + i;
      if (connection->state != connUninitialized)
        flagcxProxyCaptureCleanupResult(
            proxyFreeConnection(connection, comm, true), &result);
    }
  }

  // Phase 2 may now release exported P2P FIFOs, then cleans up NET resources.
  for (int b = 0; b < pool->banks; b++) {
    int max = b == pool->banks - 1 ? pool->offset : FLAGCX_PROXY_CONN_POOL_SIZE;
    for (int i = 0; i < max; i++) {
      struct flagcxProxyConnection *connection = pool->pools[b] + i;
      if (connection->state != connUninitialized)
        flagcxProxyCaptureCleanupResult(
            proxyFreeConnection(connection, comm, false), &result);
    }
    free(pool->pools[b]);
  }
  free(pool->pools);
  return result;
}

static flagcxResult_t SaveProxy(struct flagcxHeteroComm *comm,
                                struct flagcxChannel *channel, int type,
                                int peer, struct flagcxProxyOp *op,
                                int connIndex, bool *justInquire) {
  if (peer < 0)
    return flagcxSuccess;

  flagcxResult_t asyncResult =
      __atomic_load_n(&comm->proxyState->asyncResult, __ATOMIC_ACQUIRE);
  if (asyncResult != flagcxSuccess && asyncResult != flagcxInProgress)
    return asyncResult;

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

flagcxResult_t flagcxProxyRecordAsyncError(struct flagcxProxyState *proxyState,
                                           flagcxResult_t res) {
  if (proxyState == NULL || res == flagcxSuccess || res == flagcxInProgress)
    return res;

  flagcxResult_t expected = flagcxSuccess;
  __atomic_compare_exchange_n(&proxyState->asyncResult, &expected, res, false,
                              __ATOMIC_RELEASE, __ATOMIC_RELAXED);
  if (proxyState->abortFlag != NULL)
    __atomic_store_n(proxyState->abortFlag, 1, __ATOMIC_RELEASE);
  return res;
}

static void flagcxProxyRetireFailedOp(
    struct flagcxIntruQueue<struct flagcxProxyOp, &flagcxProxyOp::next> *queue,
    struct flagcxProxyOp *op) {
  if (op->args.done == 0) {
    if (op->args.semaphore != nullptr)
      op->args.semaphore->subCounter(op->args.opId);
    op->args.done = 1;
  }
  op->args.semaphore.reset();
  flagcxIntruQueueDelete(queue, op);
  free(op);
}

static void flagcxProxyRetireFailedQueue(
    struct flagcxIntruQueue<struct flagcxProxyOp, &flagcxProxyOp::next>
        *queue) {
  while (!flagcxIntruQueueEmpty(queue)) {
    struct flagcxProxyOp *op = flagcxIntruQueueHead(queue);
    flagcxProxyRetireFailedOp(queue, op);
  }
}

flagcxResult_t flagcxProxyFailProgressQueue(
    struct flagcxProxyState *proxyState,
    struct flagcxIntruQueue<struct flagcxProxyOp, &flagcxProxyOp::next> *queue,
    flagcxResult_t result) {
  if (queue == NULL)
    return flagcxInvalidArgument;
  flagcxProxyRecordAsyncError(proxyState, result);
  flagcxProxyRetireFailedQueue(queue);
  return result;
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
            flagcxResult_t asyncResult =
                __atomic_load_n(&proxyState->asyncResult, __ATOMIC_ACQUIRE);
            if (asyncResult != flagcxSuccess &&
                asyncResult != flagcxInProgress) {
              flagcxProxyRetireFailedQueue(queue);
              op = NULL;
            }
            if (op != NULL) {
              flagcxResult_t res =
                  op->connection->tcomm == NULL ||
                          op->connection->tcomm->progressProxyOp == NULL
                      ? flagcxNotSupported
                      : op->connection->tcomm->progressProxyOp(op->connection,
                                                               op);
              if (res != flagcxSuccess && res != flagcxInProgress) {
                flagcxProxyFailProgressQueue(proxyState, queue, res);
                op = NULL;
              }
              if (op != NULL && op->args.done == 1 &&
                  op->args.semaphore->pollEnd()) {
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
            flagcxResult_t asyncResult =
                __atomic_load_n(&proxyState->asyncResult, __ATOMIC_ACQUIRE);
            if (asyncResult != flagcxSuccess &&
                asyncResult != flagcxInProgress) {
              flagcxProxyRetireFailedQueue(queue);
              op = NULL;
            }
            if (op != NULL) {
              flagcxResult_t res =
                  op->connection->tcomm == NULL ||
                          op->connection->tcomm->progressProxyOp == NULL
                      ? flagcxNotSupported
                      : op->connection->tcomm->progressProxyOp(op->connection,
                                                               op);
              if (res != flagcxSuccess && res != flagcxInProgress) {
                flagcxProxyFailProgressQueue(proxyState, queue, res);
                op = NULL;
              }
              if (op != NULL && op->args.done == 1 &&
                  op->args.semaphore->pollEnd()) {
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
  *added = 0;
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
  flagcxResult_t asyncResult =
      __atomic_load_n(&proxyState->asyncResult, __ATOMIC_ACQUIRE);
  const bool failed =
      asyncResult != flagcxSuccess && asyncResult != flagcxInProgress;
  while (!flagcxProdProgChannelListEmpty(proxyState->prodProgChannelHead)) {
    struct flagcxProxyOps *proxyOps =
        flagcxProdProgChannelListDeList(&proxyState->prodProgChannelHead);

    struct flagcxIntruQueue<struct flagcxProxyOp, &flagcxProxyOp::next> *queue;
    queue = &proxyOps->prodPeers.sendQueue;
    if (failed) {
      flagcxProxyRetireFailedQueue(queue);
      flagcxProxyRetireFailedQueue(&proxyOps->prodPeers.recvQueue);
      continue;
    }

    flagcxConsProgChannelListEnList(&proxyState->consProgChannelHead, proxyOps);
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
  /* Too frequent call of flagcxProxyGetPostedOps() will result in perf
   * regression for small message communication. proxyOpAppendCounter is a
   * counter that helps us decide if we need to append proxy ops. After each
   * progress, proxyOpAppendCounter will increase by 1 and compare with
   * environment variable flagcxParamProgressAppendOpFreq(). If they are equal,
   * we will append proxy ops. This will decrease the frequency of calling
   * flagcxProxyGetPostedOps() and reduce the perf impact. */
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

static bool flagcxProxyDmaBufferSupport() {
  const char *dmaBufEnable = flagcxGetEnv("FLAGCX_DMABUF_ENABLE");
  bool enabled = dmaBufEnable != NULL && strcmp(dmaBufEnable, "1") == 0;
  bool supported = false;
  if (deviceAdaptor->dmaSupport != NULL)
    deviceAdaptor->dmaSupport(&supported);
  return enabled && supported;
}

static flagcxResult_t
flagcxNetProxyConnect(struct flagcxProxyConnection *connection,
                      struct flagcxProxyState *proxyState, void *reqBuff,
                      int reqSize, void *respBuff, int respSize, int *done) {
  (void)proxyState;
  (void)reqSize;
  (void)respBuff;
  (void)respSize;
  if (connection == NULL || connection->transportResources == NULL ||
      done == NULL)
    return flagcxInvalidArgument;

  bool dmaBufferSupport = flagcxProxyDmaBufferSupport();
  if (connection->send) {
    struct sendNetResources *resources =
        (struct sendNetResources *)connection->transportResources;
    resources->useDmaBuf = dmaBufferSupport;
    if (resources->netSendComm == NULL) {
      FLAGCXCHECK(resources->netAdaptor->connect(resources->netDev, reqBuff,
                                                 &resources->netSendComm));
      return flagcxSuccess;
    }

    if (dmaBufferSupport && resources->netAdaptor == getNetAdaptor(RDMA)) {
      int dmabufFd;
      FLAGCXCHECK(deviceAdaptor->getHandleForAddressRange(
          &dmabufFd, resources->buffers[0], resources->buffSizes[0], 0));
      flagcxResult_t result = resources->netAdaptor->regMrDmaBuf(
          resources->netSendComm, resources->buffers[0],
          resources->buffSizes[0], FLAGCX_PTR_CUDA, 0ULL, dmabufFd, 0,
          &resources->mhandles[0]);
      (void)close(dmabufFd);
      FLAGCXCHECK(result);
    } else {
      int type = resources->netAdaptor == getNetAdaptor(SOCKET)
                     ? FLAGCX_PTR_HOST
                     : ((resources->netAdaptor == getNetAdaptor(RDMA) ||
                         (resources->ptrSupport & FLAGCX_PTR_CUDA))
                            ? FLAGCX_PTR_CUDA
                            : FLAGCX_PTR_HOST);
      FLAGCXCHECK(resources->netAdaptor->regMr(
          resources->netSendComm, resources->buffers[0],
          resources->buffSizes[0], type, 0, &resources->mhandles[0]));
    }
  } else {
    struct recvNetResources *resources =
        (struct recvNetResources *)connection->transportResources;
    resources->useDmaBuf = dmaBufferSupport;
    if (resources->netRecvComm == NULL) {
      FLAGCXCHECK(resources->netAdaptor->accept(resources->netListenComm,
                                                &resources->netRecvComm));
      return flagcxSuccess;
    }

    if (dmaBufferSupport) {
      int dmabufFd;
      FLAGCXCHECK(deviceAdaptor->getHandleForAddressRange(
          &dmabufFd, resources->buffers[0], resources->buffSizes[0], 0));
      flagcxResult_t result = resources->netAdaptor->regMrDmaBuf(
          resources->netRecvComm, resources->buffers[0],
          resources->buffSizes[0], FLAGCX_PTR_CUDA, 0ULL, dmabufFd, 0,
          &resources->mhandles[0]);
      (void)close(dmabufFd);
      FLAGCXCHECK(result);
    } else {
      int type = resources->netAdaptor == getNetAdaptor(SOCKET)
                     ? FLAGCX_PTR_HOST
                     : ((resources->netAdaptor == getNetAdaptor(RDMA) ||
                         (resources->ptrSupport & FLAGCX_PTR_CUDA))
                            ? FLAGCX_PTR_CUDA
                            : FLAGCX_PTR_HOST);
      FLAGCXCHECK(resources->netAdaptor->regMr(
          resources->netRecvComm, resources->buffers[0],
          resources->buffSizes[0], type, 0, &resources->mhandles[0]));
    }
  }
  *done = 1;
  return flagcxSuccess;
}

static flagcxResult_t
flagcxNetProxyRegister(struct flagcxProxyConnection *connection,
                       struct flagcxProxyState *proxyState, void *reqBuff,
                       int reqSize, void *respBuff, int respSize, int *done) {
  (void)proxyState;
  if (connection == NULL || connection->transportResources == NULL ||
      reqBuff == NULL || respBuff == NULL || done == NULL ||
      reqSize != (int)sizeof(struct netRegInfo) ||
      respSize != (int)sizeof(void *))
    return flagcxInvalidArgument;

  struct netRegInfo *info = (struct netRegInfo *)reqBuff;
  void *handle = NULL;
  bool dmaBufferSupport = flagcxProxyDmaBufferSupport();
  void *netComm =
      connection->send
          ? ((struct sendNetResources *)connection->transportResources)
                ->netSendComm
          : ((struct recvNetResources *)connection->transportResources)
                ->netRecvComm;
  struct flagcxNetAdaptor *netAdaptor =
      connection->send
          ? ((struct sendNetResources *)connection->transportResources)
                ->netAdaptor
          : ((struct recvNetResources *)connection->transportResources)
                ->netAdaptor;

  if (dmaBufferSupport) {
    int dmabufFd;
    FLAGCXCHECK(deviceAdaptor->getHandleForAddressRange(
        &dmabufFd, (void *)info->buffer, info->size, 0));
    flagcxResult_t result =
        netAdaptor->regMrDmaBuf(netComm, (void *)info->buffer, info->size,
                                FLAGCX_PTR_CUDA, 0ULL, dmabufFd, 0, &handle);
    (void)close(dmabufFd);
    FLAGCXCHECK(result);
  } else {
    FLAGCXCHECK(netAdaptor->regMr(netComm, (void *)info->buffer, info->size,
                                  FLAGCX_PTR_CUDA, 0, &handle));
  }
  memcpy(respBuff, &handle, sizeof(handle));
  *done = 1;
  return flagcxSuccess;
}

static flagcxResult_t
flagcxNetProxyDeregister(struct flagcxProxyConnection *connection,
                         struct flagcxProxyState *proxyState, void *reqBuff,
                         int reqSize, int *done) {
  (void)proxyState;
  if (connection == NULL || connection->transportResources == NULL ||
      reqBuff == NULL || done == NULL || reqSize != (int)sizeof(void *))
    return flagcxInvalidArgument;
  void *handle = NULL;
  memcpy(&handle, reqBuff, sizeof(handle));
  if (connection->send) {
    struct sendNetResources *resources =
        (struct sendNetResources *)connection->transportResources;
    FLAGCXCHECK(resources->netAdaptor->deregMr(resources->netSendComm, handle));
  } else {
    struct recvNetResources *resources =
        (struct recvNetResources *)connection->transportResources;
    FLAGCXCHECK(resources->netAdaptor->deregMr(resources->netRecvComm, handle));
  }
  *done = 1;
  return flagcxSuccess;
}

static flagcxResult_t
proxyProgressAsync(struct flagcxProxyLocalPeer *peer, flagcxProxyAsyncOp *op,
                   int *asyncOpCount,
                   struct flagcxProxyConnectionPool *connectionPool,
                   struct flagcxHeteroComm *comm) {
  int done = 0;
  flagcxResult_t res = flagcxSuccess;
  if (op->type != flagcxProxyMsgInit && op->connection != NULL &&
      !proxyCleanupOpType(op->type)) {
    flagcxResult_t connectionResult =
        flagcxProxyGetConnectionError(op->connection);
    if (connectionResult != flagcxSuccess)
      return connectionResult;
  }
  if (op->type == flagcxProxyMsgInit) {
    // Allocate connection from pool
    res = proxyConnInit(
        peer, connectionPool, comm, (struct flagcxProxyInitReq *)op->reqBuff,
        (struct flagcxProxyInitResp *)op->respBuff, &op->connection);
    if (res != flagcxSuccess)
      return res;
    done = 1;
  } else {
    TRACE(FLAGCX_PROXY,
          "proxyProgressAsync opId=%p type=%d reqBuff=%p reqSize=%d "
          "respSize=%d transport=%d",
          op->opId, op->type, op->reqBuff, op->reqSize, op->respSize,
          op->connection->transport);

    struct flagcxTransportComm *tcomm = op->connection->tcomm;
    if (tcomm == NULL)
      return flagcxNotSupported;

    if (op->type == flagcxProxyMsgSetup) {
      if (tcomm->proxySetup == NULL)
        return flagcxNotSupported;
      FLAGCXCHECK(tcomm->proxySetup(op->connection, NULL, op->reqBuff,
                                    op->reqSize, op->respBuff, op->respSize,
                                    &done));
    } else if (op->type == flagcxProxyMsgConnect) {
      if (tcomm->proxyConnect == NULL)
        return flagcxNotSupported;
      FLAGCXCHECK(tcomm->proxyConnect(op->connection, NULL, op->reqBuff,
                                      op->reqSize, op->respBuff, op->respSize,
                                      &done));
    } else if (op->type == flagcxProxyMsgRegister) {
      if (tcomm->proxyRegister == NULL)
        return flagcxNotSupported;
      FLAGCXCHECK(tcomm->proxyRegister(op->connection, NULL, op->reqBuff,
                                       op->reqSize, op->respBuff, op->respSize,
                                       &done));
    } else if (op->type == flagcxProxyMsgDeregister) {
      if (tcomm->proxyDeregister == NULL)
        return flagcxNotSupported;
      FLAGCXCHECK(tcomm->proxyDeregister(op->connection, NULL, op->reqBuff,
                                         op->reqSize, &done));
    } else {
      return flagcxInternalError;
    }
  }
  if (done) {
    INFO(FLAGCX_PROXY,
         "proxyProgressAsync opId=%p op.type=%d op.reqBuff=%p op.respSize=%d "
         "done",
         op->opId, op->type, op->reqBuff, op->respSize);
    if (op->type == flagcxProxyMsgSetup)
      __atomic_store_n(&op->connection->state, connSetupDone, __ATOMIC_RELEASE);
    else if (op->type == flagcxProxyMsgConnect)
      __atomic_store_n(&op->connection->state, connConnected, __ATOMIC_RELEASE);

    /* if setup or connect is done, we should not return any error at this point
     * since flagcxSocketSend might already send the respBuff to the requester.
     * If we still choose to abort and close the connection, it can cause
     * segfault if the requester is using the respBuff. */

    flagcxProxyRpcResponseHeader resp = {op->opId, res, op->respSize};

    FLAGCXCHECK(flagcxSocketSend(op->connection->sock, &resp, sizeof(resp)));
    if (op->respSize)
      FLAGCXCHECK(
          flagcxSocketSend(op->connection->sock, op->respBuff, op->respSize));

    asyncProxyOpDequeue(peer, op);
    (*asyncOpCount)--;
    return flagcxSuccess;
  } else if (comm->abortFlag &&
             __atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE) != 0) {
    return flagcxInternalError;
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

  flagcxResult_t asyncResult =
      __atomic_load_n(&sharedProxyState->asyncResult, __ATOMIC_ACQUIRE);
  if (asyncResult != flagcxSuccess && asyncResult != flagcxInProgress &&
      !proxyCleanupOpType(type))
    return asyncResult;

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

static flagcxResult_t
proxyServiceInitOp(int type, struct flagcxProxyLocalPeer *peer,
                   struct flagcxProxyConnectionPool *connectionPool,
                   flagcxHeteroComm_t comm, int *asyncOpCount) {
  flagcxResult_t ret = flagcxSuccess;
  struct flagcxSocket *sock = &peer->sock;
  struct flagcxProxyAsyncOp *asyncOp;
  FLAGCXCHECK(flagcxCalloc(&asyncOp, 1));

  asyncOp->type = type;
  FLAGCXCHECKGOTO(flagcxSocketRecv(sock, &asyncOp->connection, sizeof(void *)),
                  ret, fail);

  FLAGCXCHECKGOTO(flagcxSocketRecv(sock, &asyncOp->reqSize, sizeof(int)), ret,
                  fail);
  FLAGCXCHECKGOTO(flagcxSocketRecv(sock, &asyncOp->respSize, sizeof(int)), ret,
                  fail);

  // Validate buffer sizes for Init to prevent heap corruption from
  // malformed/version-mismatched peers
  if (type == flagcxProxyMsgInit) {
    if (asyncOp->reqSize != (int)sizeof(struct flagcxProxyInitReq) ||
        asyncOp->respSize != (int)sizeof(struct flagcxProxyInitResp)) {
      WARN("proxyServiceInitOp: Init message size mismatch "
           "(reqSize=%d expected=%zu, respSize=%d expected=%zu)",
           asyncOp->reqSize, sizeof(struct flagcxProxyInitReq),
           asyncOp->respSize, sizeof(struct flagcxProxyInitResp));
      ret = flagcxInternalError;
      goto fail;
    }
  }

  if (asyncOp->reqSize) {
    FLAGCXCHECKGOTO(flagcxCalloc(&asyncOp->reqBuff, asyncOp->reqSize), ret,
                    fail);
    FLAGCXCHECKGOTO(flagcxSocketRecv(sock, asyncOp->reqBuff, asyncOp->reqSize),
                    ret, fail);
  }

  // Store opId for completion response
  FLAGCXCHECKGOTO(flagcxSocketRecv(sock, &asyncOp->opId, sizeof(asyncOp->opId)),
                  ret, fail);

  if (asyncOp->respSize)
    FLAGCXCHECKGOTO(flagcxCalloc(&asyncOp->respBuff, asyncOp->respSize), ret,
                    fail);

  // For Init messages, connection is NULL (will be allocated from pool in
  // proxyProgressAsync). For other messages, set the socket on the connection.
  if (type != flagcxProxyMsgInit) {
    asyncOp->connection->sock = sock;
  }

  asyncProxyOpEnqueue(peer, asyncOp);
  (*asyncOpCount)++;
  ret = proxyProgressAsync(peer, asyncOp, asyncOpCount, connectionPool, comm);
  if (ret != flagcxSuccess && ret != flagcxInProgress) {
    if (!proxyCleanupOpType(type))
      flagcxProxyRecordConnectionError(asyncOp->connection, ret);
    flagcxResult_t responseResult =
        proxyServiceCompleteOp(peer, asyncOp, asyncOpCount, ret);
    if (responseResult != flagcxSuccess)
      return responseResult;
  }
  return flagcxSuccess;
fail:
  if (asyncOp->reqBuff)
    free(asyncOp->reqBuff);
  if (asyncOp->respBuff)
    free(asyncOp->respBuff);
  free(asyncOp);
  return ret;
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
  proxyConn->transport = -1;
  proxyConn->tpRank = proxyRank;
  proxyConn->tpLocalRank = 0;

  // Lazy peerSocks allocation
  struct flagcxProxyState *sharedProxyState = comm->proxyState;
  if (sharedProxyState->peerSocks == NULL) {
    FLAGCXCHECK(flagcxCalloc(&sharedProxyState->peerSocks, comm->nRanks));
    sharedProxyState->nPeerSocks = comm->nRanks;
    for (int i = 0; i < comm->nRanks; i++)
      FLAGCXCHECK(flagcxSocketSetFd(-1, &sharedProxyState->peerSocks[i]));
  }
  // Lazy connect to peer
  {
    struct flagcxSocket *sock = &sharedProxyState->peerSocks[proxyRank];
    int ready = 0;
    FLAGCXCHECK(flagcxSocketReady(sock, &ready));
    if (!ready) {
      FLAGCXCHECK(flagcxSocketInit(sock,
                                   sharedProxyState->peerAddresses + proxyRank,
                                   comm->magic, flagcxSocketTypeProxy));
      FLAGCXCHECK(flagcxSocketConnect(sock));
    }
  }

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
  if (proxyConn->connection == NULL) {
    WARN("flagcxProxyConnect: service thread returned NULL connection for rank "
         "%d -> peer %d",
         comm->rank, proxyRank);
    return flagcxInternalError;
  }
  proxyConn->transport = transport;
  INFO(FLAGCX_PROXY,
       "flagcxProxyConnect rank %d -> peer %d connection %p sameProcess %d",
       comm->rank, proxyRank, proxyConn->connection, proxyConn->sameProcess);
  return flagcxSuccess;
}

flagcxResult_t flagcxProxyInit(struct flagcxHeteroComm *comm) {
  INFO(FLAGCX_INIT, "rank=%d flagcxProxyInit called.", comm->rank);
  FLAGCXCHECK(flagcxSocketInit(&comm->proxyState->listenSock,
                               &bootstrapNetIfAddr, comm->magic,
                               flagcxSocketTypeProxy, NULL, 0));
  FLAGCXCHECK(flagcxSocketListen(&comm->proxyState->listenSock));

  // Allgather proxy listen addresses
  FLAGCXCHECK(flagcxCalloc(&comm->proxyState->peerAddresses, comm->nRanks));
  comm->proxyState->peerAddresses[comm->rank] =
      comm->proxyState->listenSock.addr;
  FLAGCXCHECK(bootstrapCollAllGather(comm->bootstrap,
                                     comm->proxyState->peerAddresses,
                                     sizeof(union flagcxSocketAddress)));

  comm->proxyState->cudaDev = comm->cudaDev;
  comm->proxyState->nRanks = comm->nRanks;
  comm->proxyState->abortFlag = comm->abortFlag;
  comm->proxyState->asyncResult = flagcxSuccess;
  comm->proxyState->cleanupResult = flagcxSuccess;
  comm->proxyState->stop = 0;
  pthread_create(&comm->proxyState->thread, NULL, flagcxProxyService,
                 (void *)comm);
  pthread_create(&comm->proxyState->progressState.thread, NULL,
                 flagcxProxyProgress, comm->proxyState);
#ifdef COMPILE_KERNEL_HOST
  // Initialize synchronization primitives before creating threads
  pthread_mutex_init(&comm->proxyState->kernelState.initMutex, NULL);
  pthread_cond_init(&comm->proxyState->kernelState.initCond, NULL);
  comm->proxyState->kernelState.ready = 0;
  comm->proxyState->kernelState.terminalResult = flagcxSuccess;

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
  int initFailed = comm->proxyState->kernelState.initFailed;
  pthread_mutex_unlock(&comm->proxyState->kernelState.initMutex);

  if (initFailed > 0) {
    WARN("flagcxProxyInit: %d kernel proxy thread(s) failed initialization",
         initFailed);
    return flagcxSystemError;
  }

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
  flagcxResult_t res = flagcxSuccess;

  // Peer slots [0..maxConns-1], listen socket at [maxConns]
  int maxConns = comm->nRanks + 1;
  struct pollfd *pollfds =
      (struct pollfd *)calloc(maxConns + 1, sizeof(struct pollfd));
  struct flagcxProxyLocalPeer *peers = (struct flagcxProxyLocalPeer *)calloc(
      maxConns, sizeof(struct flagcxProxyLocalPeer));
  int npeers = 0;
  int maxnpeers = 0;

  // Connection pool — owns all connection structs
  struct flagcxProxyConnectionPool connectionPool;
  connectionPool.pools = NULL;
  connectionPool.banks = 0;
  connectionPool.offset = FLAGCX_PROXY_CONN_POOL_SIZE;

  // Set device context
  FLAGCXCHECKGOTO(deviceAdaptor->setDevice(comm->cudaDev), res, out);

  // All peer slots start invalid; listen socket at last index
  for (int i = 0; i < maxConns; i++) {
    pollfds[i].fd = -1;
    pollfds[i].events = POLLIN;
    peers[i].tpRank = -1;
    peers[i].tpLocalRank = -1;
  }
  pollfds[maxConns].fd = comm->proxyState->listenSock.fd;
  pollfds[maxConns].events = POLLIN;

  while (!stop || npeers > 0) {
    // Check backup atomic stop flag
    if (!stop && __atomic_load_n(&comm->proxyState->stop, __ATOMIC_ACQUIRE)) {
      stop = 1;
      INFO(FLAGCX_PROXY,
           "[Service thread] Stop flag detected via atomic, npeers=%d", npeers);
    }
    int ret;
    do {
      ret = poll(pollfds, maxConns + 1, asyncOpCount ? 0 : 500);
    } while (ret < 0 && errno == EINTR);
    if (ret < 0) {
      WARN("[Proxy Service] Poll failed: %s", strerror(errno));
      break;
    }

    // Progress async ops per-peer
    for (int i = 0; i < maxnpeers; i++) {
      if (pollfds[i].fd == -1)
        continue;
      struct flagcxProxyLocalPeer *peer = &peers[i];
      struct flagcxProxyAsyncOp *op = peer->asyncOps;
      while (op) {
        struct flagcxProxyAsyncOp *opNext = op->next;
        flagcxResult_t asyncResult =
            __atomic_load_n(&comm->proxyState->asyncResult, __ATOMIC_ACQUIRE);
        res = (asyncResult != flagcxSuccess &&
               asyncResult != flagcxInProgress && !proxyCleanupOpType(op->type))
                  ? asyncResult
                  : proxyProgressAsync(peer, op, &asyncOpCount, &connectionPool,
                                       comm);
        if (res == flagcxSuccess || res == flagcxInProgress) {
          op = opNext;
        } else {
          WARN("[Service thread] Error encountered progressing operation with "
               "res=%d",
               res);
          if (!proxyCleanupOpType(op->type))
            flagcxProxyRecordConnectionError(op->connection, res);
          flagcxResult_t responseResult =
              proxyServiceCompleteOp(peer, op, &asyncOpCount, res);
          if (responseResult != flagcxSuccess) {
            // At this point no reliable RPC response can be delivered. Poison
            // the proxy so callers terminate through the socket/abort path.
            flagcxProxyRecordAsyncError(comm->proxyState, responseResult);
          }
          op = opNext;
        }
      }
    }

    // Helper lambda to process incoming data on a socket
    auto processSocket = [&](struct flagcxProxyLocalPeer *curPeer) -> bool {
      struct flagcxSocket *sock = &curPeer->sock;
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
          return false; // close the stop socket
        } else if (type == flagcxProxyMsgClose) {
          INFO(FLAGCX_PROXY, "[Service thread] Received close from peer");
          return false; // graceful close from peer
        } else if (proxyMatchOpType(type)) {
          res = proxyServiceInitOp(type, curPeer, &connectionPool, comm,
                                   &asyncOpCount);
          if (res != flagcxSuccess) {
            WARN("[Service thread] Error encountered initializing operation "
                 "with res=%d",
                 res);
            flagcxProxyRecordAsyncError(comm->proxyState, res);
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

    // Check listenSock for new connections (at last index)
    if (pollfds[maxConns].revents & POLLIN) {
      // Find first free slot (fd == -1)
      int slot = -1;
      for (int i = 0; i < maxConns; i++) {
        if (pollfds[i].fd == -1) {
          slot = i;
          break;
        }
      }

      if (slot >= 0) {
        struct flagcxSocket *newSock = &peers[slot].sock;
        FLAGCXCHECKGOTO(flagcxSocketInit(newSock), res, out);
        if (flagcxSocketAccept(newSock, &comm->proxyState->listenSock) !=
            flagcxSuccess) {
          INFO(FLAGCX_PROXY, "[Service thread] Accept failed");
        } else {
          pollfds[slot].fd = newSock->fd;
          pollfds[slot].events = POLLIN;
          peers[slot].tpRank = -1;
          peers[slot].tpLocalRank = -1;
          peers[slot].asyncOps = NULL;
          npeers++;
          if (maxnpeers < slot + 1)
            maxnpeers = slot + 1;
          INFO(FLAGCX_PROXY,
               "[Service thread] Accepted connection at slot %d (npeers=%d)",
               slot, npeers);
        }
      } else {
        WARN("[Service thread] No free slot for new connection (npeers=%d)",
             npeers);
      }
    }

    // Check all peer slots (only up to high-water mark)
    for (int i = 0; i < maxnpeers; i++) {
      if (pollfds[i].fd == -1)
        continue;
      bool closeConn = false;
      if (pollfds[i].revents & (POLLHUP | POLLERR)) {
        closeConn = true;
      } else if (pollfds[i].revents & POLLIN) {
        if (!processSocket(&peers[i]))
          closeConn = true;
      }
      if (closeConn) {
        // Drain any remaining async ops for this peer
        while (peers[i].asyncOps) {
          asyncProxyOpDequeue(&peers[i], peers[i].asyncOps);
          asyncOpCount--;
        }
        flagcxSocketClose(&peers[i].sock);
        pollfds[i].fd = -1;
        npeers--;
        INFO(FLAGCX_PROXY,
             "[Service thread] Closed connection at slot %d tpRank %d "
             "(npeers=%d)",
             i, peers[i].tpRank, npeers);
        peers[i].tpRank = -1;
        peers[i].tpLocalRank = -1;
        peers[i].asyncOps = NULL;
      }
    }

    if (stop && npeers == 0)
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
  // FIFO memory remains GPU-visible after a worker observes a terminal error.
  // Keep every FIFO alive until all workers have exited so terminal publication
  // cannot race another worker's teardown and GPU waiters have a stable word to
  // observe.
  for (int i = 0; i < comm->proxyState->kernelState.contextCount; i++) {
    flagcxFifo_t fifo = comm->proxyState->kernelState.fifos[i];
    if (fifo != nullptr) {
      fifo->flagcxFifoDestroy();
      delete fifo;
      comm->proxyState->kernelState.fifos[i] = nullptr;
    }
    comm->fifoBuffers[i] = nullptr;
  }
  pthread_mutex_destroy(&comm->proxyState->kernelState.initMutex);
  pthread_cond_destroy(&comm->proxyState->kernelState.initCond);
#endif

  // Close sockets and drain any remaining async ops
  for (int i = 0; i < maxConns; i++) {
    if (pollfds[i].fd != -1) {
      while (peers[i].asyncOps) {
        asyncProxyOpDequeue(&peers[i], peers[i].asyncOps);
      }
      flagcxSocketClose(&peers[i].sock);
    }
  }

  // Free all connections from pool (all resource cleanup happens
  // inside the service thread before it exits)
  flagcxResult_t cleanupResult =
      flagcxProxyFreeConnections(&connectionPool, comm);
  if (cleanupResult != flagcxSuccess && cleanupResult != flagcxInProgress) {
    flagcxResult_t expected = flagcxSuccess;
    __atomic_compare_exchange_n(&comm->proxyState->cleanupResult, &expected,
                                cleanupResult, false, __ATOMIC_RELEASE,
                                __ATOMIC_RELAXED);
    WARN("[Service thread] transport cleanup failed with result %d",
         cleanupResult);
  }

  flagcxSocketClose(&comm->proxyState->listenSock);
  free(pollfds);
  free(peers);

  INFO(FLAGCX_PROXY,
       "[Service thread] Wait for progress thread joined and free resources");
  return NULL;
}

// ============================================================================
// Kernel Proxy Direct NetAdaptor Posting
// Bypasses RMA Proxy: posts IB ops directly from kernel proxy thread.
// ============================================================================

struct flagcxKernelProxyState {
  struct flagcxKernelProxyTransport transport;
  int nRanks;
  struct flagcxFifo *fifo; // owning FIFO (for completed counter advancement)
  int contextId;
};

static void flagcxKernelProxyStoreLocalTerminal(struct flagcxFifo *fifo,
                                                flagcxResult_t result) {
  if (fifo != NULL && fifo->buffer != NULL && result != flagcxSuccess &&
      result != flagcxInProgress)
    __atomic_store_n(&fifo->buffer[flagcxFifoIdxTerminalStatus],
                     (uint64_t)result, __ATOMIC_RELEASE);
}

static void
flagcxKernelProxyPublishTerminal(struct flagcxKernelProxyState *state,
                                 struct flagcxHeteroComm *comm,
                                 flagcxResult_t result) {
  if (result == flagcxSuccess || result == flagcxInProgress)
    return;
  flagcxResult_t expected = flagcxSuccess;
  bool published = __atomic_compare_exchange_n(
      &comm->proxyState->kernelState.terminalResult, &expected, result, false,
      __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
  flagcxResult_t terminal = published ? result : expected;
  // A worker owns exactly one FIFO. Other workers observe terminalResult and
  // publish the same first error to their own FIFO before exiting.
  flagcxKernelProxyStoreLocalTerminal(state == NULL ? NULL : state->fifo,
                                      terminal);
  if (comm->rmaProxy != NULL)
    __atomic_store_n(&comm->rmaProxy->rmaError, 1, __ATOMIC_RELEASE);
}

static void
flagcxKernelProxyAdvanceCompleted(struct flagcxKernelProxyState *state,
                                  uint32_t advanced) {
  if (advanced == 0 || state->fifo == NULL)
    return;
  __atomic_fetch_add(
      flagcxFifoControlPtr(state->fifo->buffer, flagcxFifoIdxCompleted),
      (flagcxCompletionWord_t)advanced, __ATOMIC_RELEASE);
}

static flagcxResult_t
flagcxKernelProxyPostGetVisibilityFlush(void *context, void *recvComm,
                                        int dstMrIdx, uint64_t dstOff,
                                        size_t size, void **request) {
  return flagcxOneSidePostGetVisibilityFlush(
      static_cast<struct flagcxHeteroComm *>(context), dstMrIdx, dstOff, size,
      recvComm, request);
}

class flagcxKernelSubmitScope {
public:
  explicit flagcxKernelSubmitScope(
      const struct flagcxNetSubmitContext *submit) {
    active_ = flagcxNetSetSubmitContext(submit) == flagcxSuccess;
  }
  ~flagcxKernelSubmitScope() {
    if (active_)
      flagcxNetClearSubmitContext();
  }

private:
  bool active_ = false;
};

// Poll all native requests, irrespective of peer or posting order. The
// transport scoreboard converts arbitrary CQE order into one contiguous FIFO
// completion prefix.
static void flagcxKernelProxyPoll(struct flagcxKernelProxyState *state,
                                  struct flagcxHeteroComm *comm) {
  if (state->transport.nativeInflight == 0)
    return;
  struct flagcxNetAdaptor *net = comm->netAdaptor;
  if (net == NULL || net->test == NULL)
    return;
  // First collect every available GET data CQE. A later pass snapshots each
  // peer's contiguous completed prefix and can cover it with one flush.
  for (uint32_t i = 0; i < state->transport.capacity; ++i) {
    struct flagcxKernelProxyRequest *entry = &state->transport.requests[i];
    if (entry->state != FLAGCX_KERNEL_PROXY_REQUEST_POSTED ||
        entry->requiresGetFlush == 0 ||
        entry->completionStage != FLAGCX_KERNEL_PROXY_COMPLETION_DATA_POSTED)
      continue;
    int ready = 0;
    flagcxResult_t completionResult = flagcxSuccess;
    flagcxResult_t progressResult = flagcxKernelProxyProgressRequest(
        &state->transport, i, net->test,
        flagcxKernelProxyPostGetVisibilityFlush, comm, &ready,
        &completionResult);
    if (progressResult != flagcxSuccess) {
      WARN("flagcxKernelProxyPoll: GET data progress failed peer=%d res=%d",
           entry->peer, (int)progressResult);
      flagcxKernelProxyPublishTerminal(state, comm, progressResult);
    }
  }
  for (uint32_t i = 0; i < state->transport.capacity; ++i) {
    struct flagcxKernelProxyRequest *entry = &state->transport.requests[i];
    if (entry->state != FLAGCX_KERNEL_PROXY_REQUEST_POSTED)
      continue;
    int ready = 0;
    flagcxResult_t completionResult = flagcxSuccess;
    flagcxResult_t progressResult = flagcxKernelProxyProgressRequest(
        &state->transport, i, net->test,
        flagcxKernelProxyPostGetVisibilityFlush, comm, &ready,
        &completionResult);
    if (progressResult != flagcxSuccess) {
      WARN("flagcxKernelProxyPoll: progress failed peer=%d res=%d", entry->peer,
           (int)progressResult);
      int stagingSlot = -1;
      flagcxResult_t abortResult =
          flagcxKernelProxyAbortRequest(&state->transport, i, &stagingSlot);
      if (abortResult == flagcxSuccess && stagingSlot >= 0)
        abortResult =
            flagcxKernelProxyReleaseStagingSlot(&state->transport, stagingSlot);
      if (abortResult != flagcxSuccess)
        WARN("flagcxKernelProxyPoll: abort failed slot=%u res=%d", i,
             (int)abortResult);
      flagcxKernelProxyPublishTerminal(state, comm, progressResult);
      continue;
    }
    if (!ready)
      continue;

    uint32_t advanced = 0;
    int stagingSlot = -1;
    flagcxResult_t result = flagcxKernelProxyCompleteRequest(
        &state->transport, i, completionResult, &advanced, &stagingSlot);
    if (result != flagcxSuccess) {
      flagcxResult_t abortResult =
          flagcxKernelProxyAbortRequest(&state->transport, i, &stagingSlot);
      if (abortResult == flagcxSuccess && stagingSlot >= 0)
        abortResult =
            flagcxKernelProxyReleaseStagingSlot(&state->transport, stagingSlot);
      if (abortResult != flagcxSuccess)
        WARN("flagcxKernelProxyPoll: completion abort failed slot=%u res=%d", i,
             (int)abortResult);
      flagcxKernelProxyPublishTerminal(state, comm, result);
      continue;
    }
    if (stagingSlot >= 0) {
      result =
          flagcxKernelProxyReleaseStagingSlot(&state->transport, stagingSlot);
      if (result != flagcxSuccess)
        flagcxKernelProxyPublishTerminal(state, comm, result);
    }
    if (completionResult != flagcxSuccess)
      flagcxKernelProxyPublishTerminal(state, comm, completionResult);
    // Publish a terminal error before advancing completed. The GPU performs an
    // acquire load of completed and may return immediately once it reaches its
    // snapshot, so the terminal word must already be visible at that point.
    flagcxKernelProxyAdvanceCompleted(state, advanced);
  }
}

// Post an IB operation directly from the kernel proxy thread.
static flagcxResult_t flagcxKernelProxyPost(
    struct flagcxKernelProxyState *state, struct flagcxHeteroComm *comm,
    const struct flagcxNetSubmitContext *submit, int peer, int type,
    int contextId, uint64_t srcOff, uint64_t dstOff, size_t size, int srcMrIdx,
    int dstMrIdx, uint64_t signalOff, uint64_t signalValue, uint64_t putValue,
    bool *posted) {
  *posted = false;
  struct flagcxRmaProxyState *proxy = comm->rmaProxy;
  struct flagcxNetAdaptor *net = comm->netAdaptor;
  if (proxy == NULL || net == NULL) {
    WARN("flagcxKernelProxyPost: rmaProxy or netAdaptor not initialized");
    return flagcxInternalError;
  }

  // Validate MR indices (mirrors flagcxHeteroPut/PutValue validation)
  if (comm->oneSideHandleCount < 1 || comm->oneSideHandles[0] == NULL) {
    WARN("flagcxKernelProxyPost: no oneSideHandles available");
    return flagcxInternalError;
  }
  if (dstMrIdx >= 0 && dstMrIdx >= comm->oneSideHandleCount) {
    WARN("flagcxKernelProxyPost: dstMrIdx %d out of range (count=%d)", dstMrIdx,
         comm->oneSideHandleCount);
    return flagcxInvalidArgument;
  }
  if (srcMrIdx >= 0 && srcMrIdx >= comm->oneSideHandleCount) {
    WARN("flagcxKernelProxyPost: srcMrIdx %d out of range (count=%d)", srcMrIdx,
         comm->oneSideHandleCount);
    return flagcxInvalidArgument;
  }

  // Use this context's own QP (context 0 = RMA proxy, contexts 1..N = kernel
  // proxy).
  int ctx = contextId + 1;
  struct flagcxOneSideHandleInfo *h0 = comm->oneSideHandles[0];
  if (h0->contextSendComms == NULL || ctx >= h0->nContexts) {
    WARN("flagcxKernelProxyPost: contextSendComms not available ctx=%d "
         "nContexts=%d",
         ctx, h0->nContexts);
    return flagcxInternalError;
  }
  void *sendComm = h0->contextSendComms[ctx][peer];
  if (sendComm == NULL) {
    WARN("flagcxKernelProxyPost: sendComm is NULL for ctx=%d peer=%d", ctx,
         peer);
    return flagcxInternalError;
  }
  void **srcHandles = NULL, **dstHandles = NULL;
  if (size > 0 && srcMrIdx >= 0 && dstMrIdx >= 0) {
    srcHandles = (void **)comm->oneSideHandles[srcMrIdx];
    dstHandles = (void **)comm->oneSideHandles[dstMrIdx];
  }
  // Reject data-bearing ops with invalid MR indices (handles would be NULL).
  if (size > 0 && (srcHandles == NULL || dstHandles == NULL)) {
    WARN("flagcxKernelProxyPost: data op size=%zu with NULL handles peer=%d",
         size, peer);
    return flagcxInvalidArgument;
  }

  // Kernel proxy threads use their own per-context QPs and do not participate
  // in the RMA proxy's opSeqs/doneSeqs tracking. Completion is signaled to the
  // GPU via the signal/counter mechanism (WaitSignal).

  int stagingSlot = -1;
  uint64_t stagingOffset = 0;
  if (type == FLAGCX_RMA_PUT_VALUE) {
    flagcxResult_t slotResult =
        flagcxKernelProxyAcquireStagingSlot(&state->transport, &stagingSlot);
    if (slotResult != flagcxSuccess)
      return slotResult;
  }

  uint32_t requestSlot = 0;
  flagcxResult_t reserveResult = flagcxKernelProxyReserveRequest(
      &state->transport, submit, peer, stagingSlot, &requestSlot);
  if (reserveResult != flagcxSuccess) {
    if (stagingSlot >= 0)
      flagcxKernelProxyReleaseStagingSlot(&state->transport, stagingSlot);
    return reserveResult;
  }

  const bool requiresGetFlush =
      type == FLAGCX_RMA_GET && size != 0 &&
      flagcxOneSideGetCompletionRequiresFlush(comm, dstMrIdx);
  if (requiresGetFlush) {
    void *flushRecvComm = h0->contextRecvComms != NULL && ctx < h0->nContexts &&
                                  h0->contextRecvComms[ctx] != NULL
                              ? h0->contextRecvComms[ctx][comm->rank]
                              : NULL;
    flagcxResult_t configureResult = flagcxKernelProxyRequireGetFlush(
        &state->transport, requestSlot, dstMrIdx, dstOff, size, flushRecvComm);
    if (configureResult != flagcxSuccess) {
      flagcxResult_t cancelResult =
          flagcxKernelProxyCancelRequest(&state->transport, requestSlot);
      if (stagingSlot >= 0)
        flagcxKernelProxyReleaseStagingSlot(&state->transport, stagingSlot);
      return cancelResult == flagcxSuccess ? configureResult : cancelResult;
    }
  }

  void *request = NULL;
  flagcxResult_t res = flagcxSuccess;
  {
    flagcxKernelSubmitScope submitScope(submit);
    switch (type) {
      case FLAGCX_RMA_PUT:
        res = net->iput == NULL
                  ? flagcxNotSupported
                  : net->iput(sendComm, srcOff, dstOff, size, comm->rank, peer,
                              srcHandles, dstHandles, &request);
        break;
      case FLAGCX_RMA_GET:
        res = net->iget == NULL
                  ? flagcxNotSupported
                  : net->iget(sendComm, srcOff, dstOff, size, peer, comm->rank,
                              srcHandles, dstHandles, &request);
        break;
      case FLAGCX_RMA_PUT_SIGNAL: {
        if (net->iputSignal == NULL) {
          res = flagcxNotSupported;
          break;
        }
        void **sigHandles = (void **)comm->signalHandle;
        res = net->iputSignal(sendComm, srcOff, dstOff, size, comm->rank, peer,
                              srcHandles, dstHandles, signalOff, sigHandles,
                              signalValue, &request);
        break;
      }
      case FLAGCX_RMA_PUT_VALUE: {
        struct flagcxOneSideHandleInfo *stagingH = comm->stagingHandle;
        flagcxDevComm_t dc = comm->devCommHandle;
        if (stagingH == NULL || stagingH->baseVas == NULL || dc == NULL ||
            dc->putValueStagingBuffer == NULL ||
            contextId >= dc->putValueStagingContextCount ||
            dc->putValueStagingSlotCount !=
                (int)state->transport.stagingSlotCount) {
          res = flagcxInternalError;
          break;
        }
        if (dstMrIdx < 0 || dstMrIdx >= comm->oneSideHandleCount) {
          res = flagcxInvalidArgument;
          break;
        }
        if (net->iput == NULL) {
          res = flagcxNotSupported;
          break;
        }
        if (!flagcxKernelPutValueStagingOffset(dc, contextId, stagingSlot,
                                               &stagingOffset)) {
          res = flagcxInternalError;
          break;
        }
        volatile uint64_t *staging =
            (volatile uint64_t *)((uint8_t *)stagingH->baseVas[comm->rank] +
                                  stagingOffset);
        *staging = putValue;
        void **stagingHandles = (void **)stagingH;
        void **dstH = (void **)comm->oneSideHandles[dstMrIdx];
        res = net->iput(sendComm, stagingOffset, dstOff, sizeof(uint64_t),
                        comm->rank, peer, stagingHandles, dstH, &request);
        break;
      }
      default:
        res = flagcxInternalError;
        break;
    }
  }

  if (request != NULL) {
    flagcxResult_t completionResult =
        res == flagcxInProgress ? flagcxSuccess : res;
    flagcxResult_t publishResult = flagcxKernelProxyPublishRequest(
        &state->transport, requestSlot, request, completionResult);
    if (publishResult != flagcxSuccess) {
      flagcxKernelProxyPublishTerminal(state, comm, publishResult);
      return publishResult;
    }
    *posted = true;
    if (completionResult != flagcxSuccess)
      flagcxKernelProxyPublishTerminal(state, comm, completionResult);
    return flagcxSuccess;
  }

  if (res == flagcxSuccess && requiresGetFlush) {
    flagcxResult_t publishResult =
        flagcxKernelProxyPublishGetFlushPending(&state->transport, requestSlot);
    if (publishResult != flagcxSuccess) {
      flagcxResult_t cancelResult =
          flagcxKernelProxyCancelRequest(&state->transport, requestSlot);
      if (cancelResult != flagcxSuccess)
        publishResult = cancelResult;
      flagcxKernelProxyPublishTerminal(state, comm, publishResult);
      return publishResult;
    }
    *posted = true;
    return flagcxSuccess;
  }

  flagcxResult_t cancelResult =
      flagcxKernelProxyCancelRequest(&state->transport, requestSlot);
  if (cancelResult != flagcxSuccess)
    res = cancelResult;
  if (stagingSlot >= 0) {
    flagcxResult_t releaseResult =
        flagcxKernelProxyReleaseStagingSlot(&state->transport, stagingSlot);
    if (releaseResult != flagcxSuccess) {
      flagcxKernelProxyPublishTerminal(state, comm, releaseResult);
      return releaseResult;
    }
  }
  if (res != flagcxSuccess && res != flagcxInProgress) {
    WARN("flagcxKernelProxyPost: post failed peer=%d type=%d res=%d", peer,
         type, (int)res);
    flagcxKernelProxyPublishTerminal(state, comm, res);
  }
  return res;
}

// Drain accepted requests at shutdown. Staging slots and scoreboard entries
// remain owned until their native request retires.
static void flagcxKernelProxyDrain(struct flagcxKernelProxyState *state,
                                   struct flagcxHeteroComm *comm) {
  while (state->transport.nativeInflight != 0) {
    uint32_t before = state->transport.nativeInflight;
    flagcxKernelProxyPoll(state, comm);
    if (state->transport.nativeInflight == before)
      sched_yield();
  }
}

// Validate that a one-sided peer is reachable via the given context's QP.
static flagcxResult_t
flagcxKernelProxyValidatePeer(struct flagcxHeteroComm *comm, int peerRank,
                              int ctx) {
  struct flagcxOneSideHandleInfo *meshH =
      (comm->oneSideHandleCount > 0) ? comm->oneSideHandles[0] : NULL;
  if (meshH == NULL)
    return flagcxNotSupported;
  if (peerRank < 0 || peerRank >= comm->nRanks)
    return flagcxInvalidArgument;

  // Check per-context full-mesh connection exists for this peer
  if (meshH->contextSendComms == NULL || ctx >= meshH->nContexts ||
      meshH->contextSendComms[ctx] == NULL ||
      meshH->contextSendComms[ctx][peerRank] == NULL)
    return flagcxNotSupported;

  return flagcxSuccess;
}

static uint32_t
flagcxKernelProxySubmitFlags(struct flagcxDeviceTrigger *trigger) {
  switch (trigger->getPrim()) {
    case flagcxDevicePrimPut:
    case flagcxDevicePrimGet:
    case flagcxDevicePrimPutValue:
      return FLAGCX_NET_SUBMIT_DATA | FLAGCX_NET_SUBMIT_INDEPENDENT;
    case flagcxDevicePrimPutSignal:
      return FLAGCX_NET_SUBMIT_DATA | FLAGCX_NET_SUBMIT_RELEASE |
             FLAGCX_NET_SUBMIT_INDEPENDENT;
    case flagcxDevicePrimSignal:
    case flagcxDevicePrimSignalValue:
      return FLAGCX_NET_SUBMIT_RELEASE | FLAGCX_NET_SUBMIT_INDEPENDENT;
    default:
      return 0;
  }
}

void *flagcxProxyKernelService(void *args) {
  int groupCount = 0;
  int termCount = 0;
  flagcxDeviceTrigger_t ptr = NULL;
  flagcxFifo_t fifo = NULL;
  flagcxStream_t stream = NULL;
  struct flagcxKernelProxyState *kproxyState = NULL;
  flagcxProxyKernelServiceArg *arg = (flagcxProxyKernelServiceArg *)args;
  struct flagcxHeteroComm *comm = arg->comm;
  int contextId = arg->contextId;
  delete arg;
  flagcxResult_t res = flagcxSuccess;
  flagcxResult_t existingTerminal = flagcxSuccess;
  uint64_t generation = 1;
  bool hasPending = false;
  bool pendingTracked = false;
  struct flagcxDeviceTrigger pending = {};
  struct flagcxNetSubmitContext submit = {};

  int ctx = contextId + 1; // kernel proxy context index

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
  existingTerminal = __atomic_load_n(
      &comm->proxyState->kernelState.terminalResult, __ATOMIC_ACQUIRE);
  if (existingTerminal != flagcxSuccess)
    __atomic_store_n(&fifo->buffer[flagcxFifoIdxTerminalStatus],
                     (uint64_t)existingTerminal, __ATOMIC_RELEASE);

  // Create a dedicated stream
  FLAGCXCHECKGOTO(deviceAdaptor->streamCreate(&stream), res, out);
  INFO(FLAGCX_P2P, "rank %d p2p stream %lu", comm->rank, (uintptr_t)stream);

  // Allocate trigger structure
  FLAGCXCHECKGOTO(flagcxCalloc(&ptr, sizeof(flagcxDeviceTrigger)), res, out);

  // Initialize direct posting state (bypasses RMA Proxy for one-sided ops)
  kproxyState =
      (struct flagcxKernelProxyState *)calloc(1, sizeof(*kproxyState));
  if (kproxyState == NULL) {
    WARN("flagcxProxyKernelService: failed to allocate kproxyState");
    res = flagcxSystemError;
    goto init_done;
  }
  kproxyState->nRanks = comm->nRanks;
  kproxyState->fifo = fifo;
  kproxyState->contextId = contextId;
  generation = comm->rmaProxy != NULL && comm->rmaProxy->generation != 0
                   ? comm->rmaProxy->generation
                   : 1;
  res = flagcxKernelProxyTransportInit(
      &kproxyState->transport, FLAGCX_KERNEL_PROXY_MAX_INFLIGHT,
      FLAGCX_KERNEL_PROXY_PUT_VALUE_SLOTS, generation, (uint64_t)contextId);
  if (res != flagcxSuccess) {
    WARN("flagcxProxyKernelService: transport init failed res=%d", (int)res);
    goto init_done;
  }

init_done:
  // Signal initialization complete (even on failure, so init thread doesn't
  // deadlock)
  pthread_mutex_lock(&comm->proxyState->kernelState.initMutex);
  comm->proxyState->kernelState.ready++;
  if (res != flagcxSuccess)
    comm->proxyState->kernelState.initFailed++;
  pthread_cond_broadcast(&comm->proxyState->kernelState.initCond);
  pthread_mutex_unlock(&comm->proxyState->kernelState.initMutex);

  if (res != flagcxSuccess)
    goto out;

  while (true) {
    if (comm->proxyState->kernelState.stop == 1)
      break;

    flagcxKernelProxyPoll(kproxyState, comm);
    flagcxResult_t terminal = __atomic_load_n(
        &comm->proxyState->kernelState.terminalResult, __ATOMIC_ACQUIRE);
    if (terminal != flagcxSuccess) {
      flagcxKernelProxyStoreLocalTerminal(fifo, terminal);
      res = terminal;
      break;
    }

    if (!hasPending) {
      res = dequeue(fifo->buffer, ptr);
      if (res == flagcxInProgress) {
        sched_yield();
        continue;
      }
      if (res != flagcxSuccess) {
        flagcxKernelProxyPublishTerminal(kproxyState, comm, res);
        break;
      }
      pending = *ptr;
      hasPending = true;
      pendingTracked = false;
    }

    if (!pendingTracked) {
      res = flagcxKernelProxyTrackNext(&kproxyState->transport,
                                       flagcxKernelProxySubmitFlags(&pending),
                                       &submit);
      if (res == flagcxInProgress) {
        sched_yield();
        continue;
      }
      if (res != flagcxSuccess) {
        flagcxKernelProxyPublishTerminal(kproxyState, comm, res);
        break;
      }
      pendingTracked = true;
    }

    bool retryPending = false;
    bool postedIB = false;
    res = flagcxSuccess;
    if ((submit.flags & FLAGCX_NET_SUBMIT_RELEASE) != 0) {
      int ready = 0;
      flagcxResult_t firstError = flagcxSuccess;
      res = flagcxKernelProxyReleaseReady(&kproxyState->transport, &submit,
                                          &ready, &firstError);
      if (res == flagcxSuccess && firstError != flagcxSuccess)
        res = firstError;
      if (res == flagcxSuccess && !ready) {
        sched_yield();
        continue;
      }
      if (res != flagcxSuccess) {
        flagcxKernelProxyPublishTerminal(kproxyState, comm, res);
      }
    }

    if (res == flagcxSuccess)
      switch (pending.getPrim()) {
        case flagcxDevicePrimSend:
          if (groupCount == 0) {
            res = flagcxHeteroGroupStart();
            TRACE(
                FLAGCX_P2P,
                "rank=%d flagcxHeteroGroupStart called by proxyKernelService.",
                comm->rank);
            groupCount++;
          }
          TRACE(FLAGCX_P2P,
                "rank=%d flagcxDevicePrimSend called by proxyKernelService.",
                comm->rank);
          res = flagcxHeteroSend((const void *)(uintptr_t)(pending.getAddr()),
                                 pending.getCount(),
                                 (flagcxDataType_t)(pending.getDatatype()),
                                 pending.getPeerRank(), comm, stream);
          break;
        case flagcxDevicePrimRecv:
          if (groupCount == 0) {
            res = flagcxHeteroGroupStart();
            TRACE(
                FLAGCX_P2P,
                "rank=%d flagcxHeteroGroupStart called by proxyKernelService.",
                comm->rank);
            groupCount++;
          }
          TRACE(FLAGCX_P2P,
                "rank=%d flagcxDevicePrimRecv called by proxyKernelService.",
                comm->rank);
          res = flagcxHeteroRecv((void *)(uintptr_t)(pending.getAddr()),
                                 pending.getCount(),
                                 (flagcxDataType_t)(pending.getDatatype()),
                                 pending.getPeerRank(), comm, stream);
          break;
        case flagcxDevicePrimTerm: {
          termCount++;
          int totalCoops = (int)pending.getTotalCoops();
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
          INFO(FLAGCX_P2P,
               "rank=%d PrimPut peer=%d srcOff=%lu dstOff=%lu size=%lu "
               "inflight=%u",
               comm->rank, (int)pending.getPeerRank(),
               (unsigned long)pending.getSrcOffset(),
               (unsigned long)pending.getDstOffset(),
               (unsigned long)pending.getSize(),
               kproxyState->transport.nativeInflight);
          int peerRank = (int)pending.getPeerRank();
          res = flagcxKernelProxyValidatePeer(comm, peerRank, ctx);
          if (res != flagcxSuccess)
            break;
          int srcMrIdx = (int)pending.getSrcMrIdx();
          int dstMrIdx = (int)pending.getDstMrIdx();
          size_t srcOffset = (size_t)pending.getSrcOffset();
          size_t dstOffset = (size_t)pending.getDstOffset();
          size_t size = (size_t)pending.getSize();
          res = flagcxKernelProxyPost(kproxyState, comm, &submit, peerRank,
                                      FLAGCX_RMA_PUT, contextId, srcOffset,
                                      dstOffset, size, srcMrIdx, dstMrIdx, 0, 0,
                                      0, &postedIB);
          retryPending = res == flagcxInProgress;
          INFO(FLAGCX_P2P, "rank=%d PrimPut posted res=%d postedIB=%d",
               comm->rank, (int)res, postedIB ? 1 : 0);
          break;
        }
        case flagcxDevicePrimSignal:
        case flagcxDevicePrimSignalValue: {
          uint64_t bufType = pending.getBufferType();
          int signalIdx = (int)pending.getSignalIdx();
          uint64_t signalValue = pending.getSignalValue();
          size_t signalOff = (size_t)signalIdx * sizeof(uint64_t);

          if (bufType == 0) {
            // Signal buffer: RDMA FETCH_AND_ADD to peer's signalBuffer
            int peerRank = (int)pending.getPeerRank();
            res = flagcxKernelProxyValidatePeer(comm, peerRank, ctx);
            if (res != flagcxSuccess) {
              break;
            }
            if (comm->signalHandle == NULL) {
              res = flagcxInternalError;
              break;
            }
            res = flagcxKernelProxyPost(kproxyState, comm, &submit, peerRank,
                                        FLAGCX_RMA_PUT_SIGNAL, contextId, 0, 0,
                                        0, -1, -1, signalOff, signalValue, 0,
                                        &postedIB);
            retryPending = res == flagcxInProgress;
          } else {
            flagcxDevComm_t dc = comm->devCommHandle;
            if (dc == NULL || dc->counterBuffer == NULL) {
              res = flagcxInternalError;
              break;
            }
            int contextCount = dc->contextCount > 0 ? dc->contextCount : 1;
            size_t totalCounterCount =
                (size_t)contextCount * (size_t)dc->counterCount;
            if (dc->counterCount <= 0 || signalIdx < 0 ||
                (size_t)signalIdx >= totalCounterCount) {
              WARN("rank=%d invalid encoded counter index=%d count=%d "
                   "contexts=%d",
                   comm->rank, signalIdx, dc->counterCount, contextCount);
              res = flagcxInvalidArgument;
              break;
            }
            int encodedContext = signalIdx / dc->counterCount;
            int counterId = signalIdx % dc->counterCount;
            if (encodedContext != contextId) {
              WARN("rank=%d counter context mismatch proxy=%d encoded=%d "
                   "counter=%d",
                   comm->rank, contextId, encodedContext, counterId);
              res = flagcxInternalError;
              break;
            }
            // enqueueFifoSignal has already flattened context and counter into
            // signalIdx. Use that offset directly; applying the context stride
            // here again would address the wrong slot.
            size_t counterOffset = (size_t)signalIdx;
            uint64_t *counterPtr =
                (uint64_t *)dc->counterBuffer + counterOffset;
            __atomic_fetch_add(counterPtr, signalValue, __ATOMIC_RELEASE);
          }
          break;
        }
        case flagcxDevicePrimWaitSignal: {
          // No-op: GPU now polls signal buffer directly (NCCL-style).
          // The proxy no longer needs to call streamWaitValue64.
          TRACE(FLAGCX_P2P,
                "rank=%d flagcxDevicePrimWaitSignal (no-op) by "
                "proxyKernelService.",
                comm->rank);
          break;
        }
        case flagcxDevicePrimPutSignal: {
          TRACE(
              FLAGCX_P2P,
              "rank=%d flagcxDevicePrimPutSignal called by proxyKernelService.",
              comm->rank);
          int peerRank = (int)pending.getPeerRank();
          res = flagcxKernelProxyValidatePeer(comm, peerRank, ctx);
          if (res != flagcxSuccess)
            break;
          int srcMrIdx = (int)pending.getSrcMrIdx();
          int dstMrIdx = (int)pending.getDstMrIdx();
          size_t srcOffset = (size_t)pending.getSrcOffset();
          size_t dstOffset = (size_t)pending.getDstOffset();
          size_t size = (size_t)pending.getSize();
          int signalIdx = (int)pending.getSignalIdx();
          uint64_t signalValue = pending.getSignalValue();
          size_t signalOff = (size_t)signalIdx * sizeof(uint64_t);
          if (comm->signalHandle == NULL) {
            res = flagcxInternalError;
            break;
          }
          res = flagcxKernelProxyPost(
              kproxyState, comm, &submit, peerRank, FLAGCX_RMA_PUT_SIGNAL,
              contextId, srcOffset, dstOffset, size, srcMrIdx, dstMrIdx,
              signalOff, signalValue, 0, &postedIB);
          retryPending = res == flagcxInProgress;
          break;
        }
        case flagcxDevicePrimPutValue: {
          int peerRank = (int)pending.getPeerRank();
          res = flagcxKernelProxyValidatePeer(comm, peerRank, ctx);
          if (res != flagcxSuccess)
            break;
          int dstMrIdx = (int)pending.getDstMrIdx();
          size_t dstOffset = (size_t)pending.getDstOffset();
          uint64_t value = pending.getValue();
          res = flagcxKernelProxyPost(
              kproxyState, comm, &submit, peerRank, FLAGCX_RMA_PUT_VALUE,
              contextId, 0, dstOffset, 0, -1, dstMrIdx, 0, 0, value, &postedIB);
          retryPending = res == flagcxInProgress;
          break;
        }
        case flagcxDevicePrimGet: {
          TRACE(FLAGCX_P2P,
                "rank=%d flagcxDevicePrimGet called by proxyKernelService.",
                comm->rank);
          int peerRank = (int)pending.getPeerRank();
          res = flagcxKernelProxyValidatePeer(comm, peerRank, ctx);
          if (res != flagcxSuccess)
            break;
          int srcMrIdx = (int)pending.getSrcMrIdx();
          int dstMrIdx = (int)pending.getDstMrIdx();
          size_t srcOffset = (size_t)pending.getSrcOffset();
          size_t dstOffset = (size_t)pending.getDstOffset();
          size_t size = (size_t)pending.getSize();
          res = flagcxKernelProxyPost(kproxyState, comm, &submit, peerRank,
                                      FLAGCX_RMA_GET, contextId, srcOffset,
                                      dstOffset, size, srcMrIdx, dstMrIdx, 0, 0,
                                      0, &postedIB);
          retryPending = res == flagcxInProgress;
          break;
        }
        case flagcxDevicePrimWait:
          // No-op: GPU now polls FIFO completed counter directly (NCCL-style).
          // The proxy no longer needs to call streamSynchronize.
          TRACE(FLAGCX_P2P,
                "rank=%d flagcxDevicePrimWait (no-op) by proxyKernelService.",
                comm->rank);
          break;
        case flagcxDevicePrimBarrierSignal: {
          // Legacy: no longer used by NCCL GIN-style barriers.
          // New barriers use per-peer PrimSignal entries (async, non-blocking).
          TRACE(FLAGCX_P2P,
                "rank=%d flagcxDevicePrimBarrierSignal (legacy no-op)",
                comm->rank);
          break;
        }
        default:
          break;
      }

    if (retryPending) {
      sched_yield();
      continue;
    }

    // Mark item as consumed AFTER processing.
    // Release ensures the GPU's fifoEnqueue space-check (acquire load of
    // consumed) observes all prior CPU writes (slot clear, etc.).
    flagcxCompletionWord_t nextCons =
        __atomic_load_n(
            flagcxFifoControlPtr(fifo->buffer, flagcxFifoIdxConsumed),
            __ATOMIC_RELAXED) +
        1;
    __atomic_store_n(flagcxFifoControlPtr(fifo->buffer, flagcxFifoIdxConsumed),
                     nextCons, __ATOMIC_RELEASE);
    if (!postedIB) {
      uint32_t advanced = 0;
      flagcxResult_t completeResult = flagcxKernelProxyCompleteImmediate(
          &kproxyState->transport, &submit, res, &advanced);
      if (completeResult != flagcxSuccess)
        res = completeResult;
      // Match the asynchronous completion path: a waiter that observes the
      // completed counter must also observe the terminal error.
      if (res != flagcxSuccess)
        flagcxKernelProxyPublishTerminal(kproxyState, comm, res);
      flagcxKernelProxyAdvanceCompleted(kproxyState, advanced);
    }
    hasPending = false;
    pendingTracked = false;
    if (res != flagcxSuccess) {
      flagcxKernelProxyPublishTerminal(kproxyState, comm, res);
      break;
    }
  }

  INFO(FLAGCX_PROXY,
       "rank=%d Proxy loop exited: stop=%d res=%d produced=%lu completed=%lu",
       comm->rank, comm->proxyState->kernelState.stop, (int)res,
       (unsigned long)__atomic_load_n(
           flagcxFifoControlPtr(fifo->buffer, flagcxFifoIdxProduced),
           __ATOMIC_ACQUIRE),
       (unsigned long)__atomic_load_n(
           flagcxFifoControlPtr(fifo->buffer, flagcxFifoIdxCompleted),
           __ATOMIC_ACQUIRE));

out:
  // Drain all in-flight direct IB requests before teardown
  if (kproxyState != NULL) {
    bool producerGateClosed = false;
    flagcxResult_t terminal = __atomic_load_n(
        &comm->proxyState->kernelState.terminalResult, __ATOMIC_ACQUIRE);
    flagcxKernelProxyStoreLocalTerminal(fifo, terminal);
    if (terminal != flagcxSuccess && fifo != NULL) {
      // terminalStatus is published before the gate is closed. A producer that
      // loses the gate CAS observes that status; one that won the CAS remains
      // counted until it publishes or abandons its reservation.
      flagcxResult_t gateResult =
          flagcxKernelProxyCloseFifoProducerGate(fifo->buffer);
      if (gateResult != flagcxSuccess) {
        WARN("flagcxProxyKernelService: failed to close producer gate res=%d",
             (int)gateResult);
        res = gateResult;
      } else {
        producerGateClosed = true;
      }
    }
    flagcxKernelProxyDrain(kproxyState, comm);
    terminal = __atomic_load_n(&comm->proxyState->kernelState.terminalResult,
                               __ATOMIC_ACQUIRE);
    flagcxKernelProxyStoreLocalTerminal(fifo, terminal);
    if (terminal != flagcxSuccess && fifo != NULL) {
      // Drain may itself discover the first transport failure. Close the gate
      // here as well so the final produced snapshot is stable in both paths.
      if (!producerGateClosed) {
        flagcxResult_t gateResult =
            flagcxKernelProxyCloseFifoProducerGate(fifo->buffer);
        if (gateResult != flagcxSuccess) {
          WARN("flagcxProxyKernelService: failed to close producer gate "
               "after drain res=%d",
               (int)gateResult);
          res = gateResult;
        } else {
          producerGateClosed = true;
        }
      }
      // Already-published and reserved-but-unpublished descriptors are failed
      // only after all producers and accepted native requests have retired.
      if (producerGateClosed) {
        flagcxResult_t finalizeResult =
            flagcxKernelProxyFinalizeTerminalFifo(fifo->buffer);
        if (finalizeResult != flagcxSuccess) {
          WARN("flagcxProxyKernelService: failed to finalize terminal FIFO "
               "res=%d",
               (int)finalizeResult);
          res = finalizeResult;
        }
      }
    }
    flagcxKernelProxyTransportDestroy(&kproxyState->transport);
    free(kproxyState);
    kproxyState = NULL;
  }
  // destroy stream (only if created)
  if (stream != nullptr) {
    deviceAdaptor->streamSynchronize(stream);
    deviceAdaptor->streamDestroy(stream);
  }
  // deallocate trigger structure (only if allocated)
  free(ptr);
  return NULL;
}

flagcxResult_t flagcxProxyFree(struct flagcxHeteroComm *comm) {
  // Connection structs and transport resources are now freed by the service
  // thread via flagcxProxyFreeConnections. We only NULL out
  // the pointers here to prevent dangling references.
  for (int peer = 0; peer < comm->nRanks; peer++) {
    for (int c = 0; c < MAXCHANNELS; c++) {
      if (comm->channels[c].peers[peer]->recv[0].connected == 1) {
        comm->channels[c].peers[peer]->recv[0].proxyConn.connection = NULL;
      }
      if (comm->channels[c].peers[peer]->send[0].connected == 1) {
        comm->channels[c].peers[peer]->send[0].proxyConn.connection = NULL;
      }
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxProxyStop(struct flagcxHeteroComm *comm) {
  if (comm->proxyState->initialized != 1) {
    return flagcxSuccess;
  }

  INFO(FLAGCX_PROXY, "flagcxProxyStop: sending stop to service thread...");
  // 1. Send MsgStop to own service thread via listen socket (best-effort)
  {
    struct flagcxSocket sock;
    int type = flagcxProxyMsgStop;
    if (flagcxSocketInit(&sock, &comm->proxyState->listenSock.addr, comm->magic,
                         flagcxSocketTypeProxy) == flagcxSuccess) {
      if (flagcxSocketConnect(&sock) == flagcxSuccess) {
        int ready = 0;
        while (!ready) {
          (void)flagcxSocketReady(&sock, &ready);
        }
        (void)flagcxSocketSend(&sock, &type, sizeof(int));
      }
      (void)flagcxSocketClose(&sock);
    }
  }

  // 2. Send MsgClose + close each peerSock (best-effort)
  if (comm->proxyState->peerSocks != NULL) {
    for (int i = 0; i < comm->proxyState->nPeerSocks; i++) {
      if (comm->proxyState->peerSocks[i].fd >= 0) {
        int closeType = flagcxProxyMsgClose;
        (void)flagcxSocketSend(&comm->proxyState->peerSocks[i], &closeType,
                               sizeof(int));
        flagcxSocketClose(&comm->proxyState->peerSocks[i]);
      }
    }
  }

  // 3. Set atomic stop flag (unconditional — authoritative shutdown signal)
  __atomic_store_n(&comm->proxyState->stop, 1, __ATOMIC_RELEASE);

  // 4. Signal kernel threads to stop (unconditional)
  comm->proxyState->kernelState.stop = 1;
  INFO(FLAGCX_PROXY, "flagcxProxyStop: done");
  return flagcxSuccess;
}

flagcxResult_t flagcxProxyDestroy(struct flagcxHeteroComm *comm) {
  flagcxResult_t result = flagcxSuccess;
  if (comm->proxyState->initialized == 1) {
    // Join service thread
    INFO(FLAGCX_PROXY, "flagcxProxyDestroy: joining service thread...");
    pthread_join(comm->proxyState->thread, nullptr);
    result =
        __atomic_load_n(&comm->proxyState->cleanupResult, __ATOMIC_ACQUIRE);
    INFO(FLAGCX_PROXY, "flagcxProxyDestroy: service thread joined, freeing...");
    // Free transport resources (must happen after thread join)
    flagcxProxyFree(comm);
    INFO(FLAGCX_PROXY, "flagcxProxyDestroy: done");
  }
  // free peerSocks
  if (comm->proxyState->peerSocks != NULL) {
    free(comm->proxyState->peerSocks);
    comm->proxyState->peerSocks = NULL;
  }
  if (comm->proxyState->peerAddresses != NULL) {
    free(comm->proxyState->peerAddresses);
    comm->proxyState->peerAddresses = NULL;
  }
  return result;
}
