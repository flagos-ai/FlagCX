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
#include <errno.h>
#include <new>
#include <poll.h>
#include <string>
#include <sys/socket.h>
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
    case flagcxProxyMsgCancelRelay:
    case flagcxProxyMsgReleaseRelay:
      return true;
    default:
      return false;
  }
}

static bool proxyCleanupOpType(int type) {
  return type == flagcxProxyMsgDeregister || type == flagcxProxyMsgDeregMr ||
         type == flagcxProxyMsgCancelRelay ||
         type == flagcxProxyMsgReleaseRelay;
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
  op->args.semaphore.reset();
  delete op;
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
  if (op->connection != NULL && op->connection->activeRelayOp == op)
    op->connection->activeRelayOp = NULL;
  if (op->type == flagcxProxyMsgSendRecv && op->respBuff != NULL &&
      op->respSize == 1) {
    // A completed NET send no longer reads this slot. A failed send must not
    // permit the source to overwrite the relay-owned buffer.
    op->respBuff[0] = result == flagcxSuccess ? 1 : 0;
  } else if (result != flagcxSuccess && op->respBuff != NULL &&
             op->respSize > 0) {
    memset(op->respBuff, 0, op->respSize);
  }

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

static flagcxResult_t flagcxNetRelaySendRpc(flagcxProxyAsyncOp *op,
                                            flagcxHeteroComm *comm,
                                            int sourceRank, int *done) {
  (void)comm;
  (void)sourceRank;
  if (op->reqSize != sizeof(flagcxNetRelaySendRequest) || op->respSize != 1 ||
      op->reqBuff == NULL || op->connection == NULL || !op->connection->send ||
      op->connection->transport != TRANSPORT_NET ||
      op->connection->state != connConnected || flagcxNetChunks <= 0 ||
      flagcxNetChunkSize <= 0)
    return flagcxInvalidArgument;
  auto *resources =
      static_cast<sendNetResources *>(op->connection->transportResources);
  if (resources == NULL || resources->netSendComm == NULL ||
      !resources->relayIpcBuffer || resources->relayExportBuffer == NULL ||
      resources->buffers[0] != resources->relayExportBuffer ||
      resources->mhandles[0] == NULL)
    return flagcxInvalidArgument;
  const auto *request =
      reinterpret_cast<const flagcxNetRelaySendRequest *>(op->reqBuff);
  if (request->bytes == 0 || request->bytes > (size_t)flagcxNetChunkSize ||
      request->sequence == UINT64_MAX || request->requestId == 0)
    return flagcxInvalidArgument;

  if (op->connection->activeRelayOp != NULL &&
      op->connection->activeRelayOp != op)
    return flagcxSuccess;
  op->connection->activeRelayOp = op;

  if (op->args.semaphore == nullptr) {
    op->args.chunkSize = flagcxNetChunkSize;
    op->args.chunkSteps = 1;
    op->args.sendStepMask = flagcxNetChunks - 1;
    op->args.opId = 0;
    op->args.step = 0;
    FLAGCXCHECK(flagcxCollProxyTransportInit(
        &op->args.collTransport, (uint32_t)flagcxNetChunks, request->generation,
        request->orderingKey, request->submitFlags));
    FLAGCXCHECK(flagcxNetCompletionScoreboardInit(
        &op->args.collTransport.scoreboard, op->args.collTransport.entries,
        (uint32_t)flagcxNetChunks, request->generation, request->sequence));
    op->args.collTransport.nextSubmit = request->sequence;
    op->args.sequenceBase = request->sequence;
    op->args.collTransport.laneMask = &op->connection->collDataLaneMask;
    op->args.regBufFlag = 1;
    op->args.regHandle = resources->mhandles[0];
    op->args.semaphore = std::make_shared<flagcxHostSemaphore>();
    op->args.semaphore->addCounter(0);
    op->args.semaphore->signalStart();
  }

  FLAGCXCHECK(flagcxProxySend(resources, resources->relayExportBuffer,
                              request->bytes, &op->args));
  *done = op->args.done;
  return flagcxSuccess;
}

static flagcxResult_t
flagcxNetSendProxySetup(struct flagcxProxyConnection *connection,
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
    flagcxNetSendProxySetup,
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
    struct flagcxProxyState *proxyState,
    struct flagcxIntruQueue<struct flagcxProxyOp, &flagcxProxyOp::next> *queue,
    struct flagcxProxyOp *op) {
  if (op->args.done == 0) {
    if (op->args.semaphore != nullptr)
      op->args.semaphore->subCounter(op->args.opId);
    op->args.done = 1;
  }
  const bool gpuDone =
      op->args.semaphore == nullptr || op->args.semaphore->pollEnd();
  flagcxIntruQueueDelete(queue, op);
  if (!gpuDone && op->relaySendState != NULL) {
    op->next = proxyState->deferredRelayOps;
    proxyState->deferredRelayOps = op;
    return;
  }
  if (gpuDone)
    flagcxNetCleanupRelaySendOp(op);
  else
    flagcxNetAbandonRelaySendOp(op);
  op->args.semaphore.reset();
  delete op;
}

static void flagcxProxyRetireFailedQueue(
    struct flagcxProxyState *proxyState,
    struct flagcxIntruQueue<struct flagcxProxyOp, &flagcxProxyOp::next>
        *queue) {
  while (!flagcxIntruQueueEmpty(queue)) {
    struct flagcxProxyOp *op = flagcxIntruQueueHead(queue);
    flagcxProxyRetireFailedOp(proxyState, queue, op);
  }
}

static void flagcxProxyDrainDeferredRelayOps(struct flagcxProxyState *state,
                                             int *idle) {
  flagcxProxyOp **link = &state->deferredRelayOps;
  while (*link != NULL) {
    flagcxProxyOp *op = *link;
    const bool gpuDone =
        op->args.semaphore == nullptr || op->args.semaphore->pollEnd();
    if (!gpuDone && state->progressState.stop == 0) {
      *idle = 0;
      link = &op->next;
      continue;
    }
    *link = op->next;
    if (gpuDone)
      flagcxNetCleanupRelaySendOp(op);
    else
      flagcxNetAbandonRelaySendOp(op);
    op->args.semaphore.reset();
    delete op;
  }
}

flagcxResult_t flagcxProxyFailProgressQueue(
    struct flagcxProxyState *proxyState,
    struct flagcxIntruQueue<struct flagcxProxyOp, &flagcxProxyOp::next> *queue,
    flagcxResult_t result) {
  if (queue == NULL)
    return flagcxInvalidArgument;
  flagcxProxyRecordAsyncError(proxyState, result);
  flagcxProxyRetireFailedQueue(proxyState, queue);
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
  flagcxProxyDrainDeferredRelayOps(proxyState, idle);
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
              flagcxProxyRetireFailedQueue(proxyState, queue);
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
                flagcxNetCleanupRelaySendOp(op);
                op->args.semaphore.reset();
                flagcxIntruQueueDelete(queue, op);
                delete op;
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
              flagcxProxyRetireFailedQueue(proxyState, queue);
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
                delete op;
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
  if (flagcxConsProgChannelListEmpty(proxyState->consProgChannelHead) &&
      proxyState->deferredRelayOps == NULL) {
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
      flagcxProxyRetireFailedQueue(proxyState, queue);
      flagcxProxyRetireFailedQueue(proxyState, &proxyOps->prodPeers.recvQueue);
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
  if (respSize < 0)
    return flagcxInvalidArgument;
  struct flagcxExpectedProxyResponse *ex;
  FLAGCXCHECK(flagcxCalloc(&ex, 1));
  ex->opId = opId;

  // Pre-alloc response buffer
  if (respSize > 0) {
    ex->respBuff = malloc(respSize);
    if (ex->respBuff == NULL) {
      free(ex);
      return flagcxSystemError;
    }
  }
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
      if (elem->respSize > 0)
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

static flagcxResult_t
flagcxProxyFailResponseRead(struct flagcxSocket *sock,
                            struct flagcxProxyRpcReadState *read,
                            flagcxResult_t result) {
  free(read->body);
  *read = {};
  (void)flagcxSocketClose(sock);
  return result;
}

static flagcxResult_t flagcxPollProxyResponseUnlocked(
    struct flagcxHeteroComm *comm, struct flagcxProxyConnector *proxyConn,
    void *respBuff, void *opId, bool *responseReceived) {
  if (responseReceived != NULL)
    *responseReceived = false;
  struct flagcxProxyState *sharedProxyState = comm->proxyState;
  // Check response queue
  int found = 0;
  flagcxResult_t res =
      expectedProxyResponseDequeue(sharedProxyState, opId, respBuff, &found);

  if (found != 0) {
    INFO(FLAGCX_PROXY, "flagcxPollProxyResponse Dequeued cached opId=%p", opId);
    if (responseReceived != NULL)
      *responseReceived = true;
    return res;
  }

  if (sharedProxyState->peerSocks == NULL || proxyConn->tpRank < 0 ||
      proxyConn->tpRank >= sharedProxyState->nPeerSocks)
    return flagcxInternalError;
  if (sharedProxyState->rpcReadStates == NULL)
    FLAGCXCHECK(flagcxCalloc(&sharedProxyState->rpcReadStates,
                             sharedProxyState->nPeerSocks));

  struct flagcxSocket *sock = &sharedProxyState->peerSocks[proxyConn->tpRank];
  auto *read = &sharedProxyState->rpcReadStates[proxyConn->tpRank];
  if (read->headerBytes < sizeof(read->header)) {
    int offset = static_cast<int>(read->headerBytes);
    res = flagcxSocketProgress(FLAGCX_SOCKET_RECV, sock, &read->header,
                               sizeof(read->header), &offset);
    if (res != flagcxSuccess)
      return flagcxProxyFailResponseRead(sock, read, res);
    read->headerBytes = offset;
    if (read->headerBytes < sizeof(read->header))
      return flagcxInProgress;

    // Validate the announced size before allocating or reading a body. Every
    // reply must correspond to an outstanding RPC on this communicator.
    struct flagcxExpectedProxyResponse *expected =
        sharedProxyState->expectedResponses;
    while (expected != NULL && expected->opId != read->header.opId)
      expected = expected->next;
    if (expected == NULL || expected->done || read->header.respSize < 0 ||
        expected->respSize != read->header.respSize)
      return flagcxProxyFailResponseRead(sock, read, flagcxInternalError);
    if (read->header.respSize > 0) {
      read->body = malloc(read->header.respSize);
      if (read->body == NULL)
        return flagcxProxyFailResponseRead(sock, read, flagcxSystemError);
    }
  }

  if (read->header.respSize > 0) {
    int offset = static_cast<int>(read->bodyBytes);
    res = flagcxSocketProgress(FLAGCX_SOCKET_RECV, sock, read->body,
                               read->header.respSize, &offset);
    if (res != flagcxSuccess)
      return flagcxProxyFailResponseRead(sock, read, res);
    read->bodyBytes = offset;
    if (read->bodyBytes < static_cast<size_t>(read->header.respSize))
      return flagcxInProgress;
  }

  const flagcxProxyRpcResponseHeader header = read->header;
  void *body = read->body;
  read->header = {};
  read->headerBytes = 0;
  read->body = NULL;
  read->bodyBytes = 0;
  res = expectedProxyResponseStore(sharedProxyState, header.opId, body,
                                   header.respSize, header.res);
  if (res != flagcxSuccess) {
    free(body);
    (void)flagcxSocketClose(sock);
    return res;
  }
  if (header.opId != opId)
    return flagcxInProgress;
  res = expectedProxyResponseDequeue(sharedProxyState, opId, respBuff, &found);
  if (responseReceived != NULL)
    *responseReceived = found != 0;
  return res;
}

flagcxResult_t flagcxPollProxyResponse(struct flagcxHeteroComm *comm,
                                       struct flagcxProxyConnector *proxyConn,
                                       void *respBuff, void *opId) {
  return flagcxPollProxyResponseWithStatus(comm, proxyConn, respBuff, opId,
                                           NULL);
}

flagcxResult_t flagcxPollProxyResponseWithStatus(
    struct flagcxHeteroComm *comm, struct flagcxProxyConnector *proxyConn,
    void *respBuff, void *opId, bool *responseReceived) {
  if (responseReceived != NULL)
    *responseReceived = false;
  if (__atomic_load_n(&comm->proxyState->rpcStopping, __ATOMIC_ACQUIRE))
    return flagcxInternalError;
  pthread_mutex_lock(&comm->proxyState->rpcMutex);
  flagcxResult_t res =
      __atomic_load_n(&comm->proxyState->rpcStopping, __ATOMIC_ACQUIRE)
          ? flagcxInternalError
          : flagcxPollProxyResponseUnlocked(comm, proxyConn, respBuff, opId,
                                            responseReceived);
  pthread_mutex_unlock(&comm->proxyState->rpcMutex);
  return res;
}

void flagcxProxyForgetResponse(struct flagcxHeteroComm *comm, void *opId) {
  if (comm == NULL || comm->proxyState == NULL)
    return;
  if (__atomic_load_n(&comm->proxyState->rpcStopping, __ATOMIC_ACQUIRE))
    return; // Stop owns the remaining expected-response list.
  pthread_mutex_lock(&comm->proxyState->rpcMutex);
  if (!__atomic_load_n(&comm->proxyState->rpcStopping, __ATOMIC_ACQUIRE))
    (void)expectedProxyResponseRemove(comm->proxyState, opId);
  pthread_mutex_unlock(&comm->proxyState->rpcMutex);
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
flagcxNetSendProxySetup(struct flagcxProxyConnection *connection,
                        struct flagcxProxyState *proxyState, void *reqBuff,
                        int reqSize, void *respBuff, int respSize, int *done) {
  (void)respBuff;
  if (connection == NULL || !connection->send || proxyState == NULL ||
      reqBuff == NULL || reqSize != sizeof(struct flagcxNetSendSetupRequest) ||
      respSize != 0 || done == NULL || connection->transportResources != NULL)
    return flagcxInvalidArgument;

  struct flagcxNetAdaptor *netAdaptor =
      __atomic_load_n(&proxyState->netAdaptor, __ATOMIC_ACQUIRE);
  if (netAdaptor == NULL)
    return flagcxInvalidArgument;

  const auto *request =
      static_cast<const struct flagcxNetSendSetupRequest *>(reqBuff);
  int relayNetDev = -1;
  FLAGCXCHECK(flagcxNetDevFromGuid(netAdaptor, request->netGuid, &relayNetDev));
  struct sendNetResources *resources = NULL;
  FLAGCXCHECK(flagcxCalloc(&resources, 1));
  // The service thread runs on the relay rank's device. Connection cleanup
  // releases the allocated stream, events, and NET buffer at proxy teardown.
  connection->transportResources = resources;
  FLAGCXCHECK(
      flagcxNetInitSendResources(netAdaptor, relayNetDev, resources, true));
  *done = 1;
  return flagcxSuccess;
}

static flagcxResult_t
flagcxNetProxyConnect(struct flagcxProxyConnection *connection,
                      struct flagcxProxyState *proxyState, void *reqBuff,
                      int reqSize, void *respBuff, int respSize, int *done) {
  (void)proxyState;
  (void)reqSize;
  if (connection == NULL || connection->transportResources == NULL ||
      done == NULL)
    return flagcxInvalidArgument;

  bool dmaBufferSupport = flagcxProxyDmaBufferSupport();
  if (connection->send) {
    struct sendNetResources *resources =
        (struct sendNetResources *)connection->transportResources;
    if (resources->relayIpcBuffer) {
      if (respBuff == NULL || respSize != sizeof(flagcxNetRelayBufferInfo))
        return flagcxInvalidArgument;
    } else if (respSize != 0) {
      return flagcxInvalidArgument;
    }
    // PXN always exports the gdrMemAlloc buffer through device IPC and
    // registers that same allocation through regMr. CI runs with VMM off.
    if (resources->relayIpcBuffer)
      dmaBufferSupport = false;
    resources->useDmaBuf = dmaBufferSupport;
    if (resources->netSendComm == NULL) {
      FLAGCXCHECK(resources->netAdaptor->connect(resources->netDev, reqBuff,
                                                 &resources->netSendComm));
      return flagcxSuccess;
    }

    auto registerSendBuffer = [&]() -> flagcxResult_t {
      if (dmaBufferSupport && resources->netAdaptor == getNetAdaptor(RDMA)) {
        int dmabufFd;
        FLAGCXCHECK(deviceAdaptor->getHandleForAddressRange(
            &dmabufFd, resources->buffers[0], resources->buffSizes[0], 0));
        flagcxResult_t result = resources->netAdaptor->regMrDmaBuf(
            resources->netSendComm, resources->buffers[0],
            resources->buffSizes[0], FLAGCX_PTR_CUDA, 0ULL, dmabufFd, 0,
            &resources->mhandles[0]);
        (void)close(dmabufFd);
        return result;
      }
      int type = resources->netAdaptor == getNetAdaptor(SOCKET)
                     ? FLAGCX_PTR_HOST
                     : ((resources->netAdaptor == getNetAdaptor(RDMA) ||
                         (resources->ptrSupport & FLAGCX_PTR_CUDA))
                            ? FLAGCX_PTR_CUDA
                            : FLAGCX_PTR_HOST);
      return resources->netAdaptor->regMr(
          resources->netSendComm, resources->buffers[0],
          resources->buffSizes[0], type, 0, &resources->mhandles[0]);
    };
    FLAGCXCHECK(registerSendBuffer());
    if (resources->relayIpcBuffer) {
      if (resources->mhandles[0] == NULL)
        return flagcxInternalError;
      auto *info = static_cast<flagcxNetRelayBufferInfo *>(respBuff);
      info->handleData = resources->relayHandleData;
      info->handleSize = resources->relayHandleSize;
      info->capacity = resources->buffSizes[0];
      // A reply may be lost after publication; require an explicit source
      // release acknowledgement before freeing the exported allocation.
      resources->relaySourceReleased = false;
      INFO(FLAGCX_NET, "PXN relay registered exported buffer %p with NET/%d",
           resources->relayExportBuffer, resources->netDev);
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
      FLAGCXCHECK(tcomm->proxySetup(op->connection, comm->proxyState,
                                    op->reqBuff, op->reqSize, op->respBuff,
                                    op->respSize, &done));
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
    } else if (op->type == flagcxProxyMsgSendRecv) {
      FLAGCXCHECK(flagcxNetRelaySendRpc(op, comm, peer->tpRank, &done));
    } else if (op->type == flagcxProxyMsgCancelRelay) {
      if (op->reqSize != sizeof(flagcxNetRelayCancelRequest) ||
          op->respSize != 1 || op->reqBuff == NULL)
        return flagcxInvalidArgument;
      const auto *cancel =
          reinterpret_cast<const flagcxNetRelayCancelRequest *>(op->reqBuff);
      if (cancel->requestId == 0)
        return flagcxInvalidArgument;
      done = 1;
      for (flagcxProxyAsyncOp *pending = peer->asyncOps; pending != NULL;
           pending = pending->next) {
        if (pending->type != flagcxProxyMsgSendRecv ||
            pending->reqSize != sizeof(flagcxNetRelaySendRequest) ||
            pending->reqBuff == NULL)
          continue;
        const auto *request =
            reinterpret_cast<const flagcxNetRelaySendRequest *>(
                pending->reqBuff);
        if (request->requestId == cancel->requestId) {
          // Let an already submitted NET send retire before reusing its slot.
          // The cancellation reply certifies that the slot is idle.
          done = 0;
          break;
        }
      }
      if (done)
        op->respBuff[0] = 1;
    } else if (op->type == flagcxProxyMsgReleaseRelay) {
      if (op->reqSize != 0 || op->respSize != 0 || op->connection == NULL ||
          !op->connection->send || op->connection->transport != TRANSPORT_NET ||
          op->connection->transportResources == NULL)
        return flagcxInvalidArgument;
      auto *resources =
          static_cast<sendNetResources *>(op->connection->transportResources);
      if (!resources->relayIpcBuffer || op->connection->activeRelayOp != NULL)
        return flagcxInvalidArgument;
      resources->relaySourceReleased = true;
      done = 1;
    } else {
      return flagcxInternalError;
    }
  }
  if (done) {
    if (op->type == flagcxProxyMsgSendRecv && op->respSize == 1)
      op->respBuff[0] = 1;
    if (op->connection != NULL && op->connection->activeRelayOp == op)
      op->connection->activeRelayOp = NULL;
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
  } else if (!proxyCleanupOpType(op->type) && comm->abortFlag &&
             __atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE) != 0) {
    return flagcxInternalError;
  }

  return flagcxInProgress;
}

// A proxy request is a sequence of small writes protected by rpcMutex. Use
// bounded, interruptible writes so Stop can break a peer that stopped reading
// without first acquiring that mutex.
static flagcxResult_t flagcxProxySendRpcBytes(struct flagcxProxyState *state,
                                              struct flagcxSocket *sock,
                                              const void *data, int size) {
  if (sock->fd < 0 || sock->state != flagcxSocketStateReady)
    return flagcxInternalError;
  const char *bytes = static_cast<const char *>(data);
  int offset = 0;
  struct timespec started = {};
  if (clock_gettime(CLOCK_MONOTONIC, &started) != 0)
    return flagcxSystemError;
  const uint64_t deadline = uint64_t(started.tv_sec) * 1000000000ULL +
                            started.tv_nsec + 30000000000ULL;
  while (offset < size) {
    if (__atomic_load_n(&state->rpcStopping, __ATOMIC_ACQUIRE))
      return flagcxInternalError;
    ssize_t sent = send(sock->fd, bytes + offset, size - offset,
                        MSG_DONTWAIT | MSG_NOSIGNAL);
    if (sent > 0) {
      offset += sent;
      continue;
    }
    if (sent < 0 && errno == EINTR)
      continue;
    if (sent < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      struct timespec now = {};
      if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return flagcxSystemError;
      if (uint64_t(now.tv_sec) * 1000000000ULL + now.tv_nsec >= deadline)
        return flagcxRemoteError;
      struct pollfd waitForWrite = {sock->fd, POLLOUT, 0};
      int ready = poll(&waitForWrite, 1, 10);
      if (ready < 0 && errno != EINTR)
        return flagcxRemoteError;
      if (ready > 0 && (waitForWrite.revents & (POLLERR | POLLHUP | POLLNVAL)))
        return flagcxRemoteError;
      continue;
    }
    return flagcxRemoteError;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxProxyCallAsync(struct flagcxHeteroComm *comm,
                                    struct flagcxProxyConnector *proxyConn,
                                    int type, void *reqBuff, int reqSize,
                                    int respSize, void *opId) {
  struct flagcxSocket *sock = NULL;
  flagcxResult_t ret = flagcxSuccess;
  struct flagcxProxyState *sharedProxyState = comm->proxyState;
  struct flagcxProxyConnection *rpcConnection =
      proxyConn->remoteConnection != NULL ? proxyConn->remoteConnection
                                          : proxyConn->connection;

  flagcxResult_t asyncResult =
      __atomic_load_n(&sharedProxyState->asyncResult, __ATOMIC_ACQUIRE);
  if (asyncResult != flagcxSuccess && asyncResult != flagcxInProgress &&
      !proxyCleanupOpType(type))
    return asyncResult;
  if (__atomic_load_n(&sharedProxyState->rpcStopping, __ATOMIC_ACQUIRE))
    return flagcxInternalError;

  pthread_mutex_lock(&sharedProxyState->rpcMutex);
  if (__atomic_load_n(&sharedProxyState->rpcStopping, __ATOMIC_ACQUIRE) ||
      sharedProxyState->peerSocks == NULL || proxyConn->tpRank < 0 ||
      proxyConn->tpRank >= sharedProxyState->nPeerSocks) {
    ret = flagcxInternalError;
    goto error;
  }
  sock = &sharedProxyState->peerSocks[proxyConn->tpRank];

  FLAGCXCHECKGOTO(
      flagcxProxySendRpcBytes(sharedProxyState, sock, &type, sizeof(type)), ret,
      error);
  FLAGCXCHECKGOTO(flagcxProxySendRpcBytes(sharedProxyState, sock,
                                          &rpcConnection, sizeof(void *)),
                  ret, error);
  FLAGCXCHECKGOTO(flagcxProxySendRpcBytes(sharedProxyState, sock, &reqSize,
                                          sizeof(reqSize)),
                  ret, error);
  FLAGCXCHECKGOTO(flagcxProxySendRpcBytes(sharedProxyState, sock, &respSize,
                                          sizeof(respSize)),
                  ret, error);
  if (reqSize)
    FLAGCXCHECKGOTO(
        flagcxProxySendRpcBytes(sharedProxyState, sock, reqBuff, reqSize), ret,
        error);

  // Send opId to proxy
  FLAGCXCHECKGOTO(
      flagcxProxySendRpcBytes(sharedProxyState, sock, &opId, sizeof(opId)), ret,
      error);

  FLAGCXCHECKGOTO(
      expectedProxyResponseEnqueue(sharedProxyState, opId, respSize), ret,
      error);
  pthread_mutex_unlock(&sharedProxyState->rpcMutex);
  return flagcxSuccess;
error:
  // A failed write can leave a partial request on the wire. It cannot be
  // reused for another RPC frame; EOF lets the service discard that peer.
  if (sock != NULL)
    (void)flagcxSocketClose(sock);
  pthread_mutex_unlock(&sharedProxyState->rpcMutex);
  return ret;
}

static flagcxResult_t
proxyServiceInitOp(int type, struct flagcxProxyLocalPeer *peer,
                   struct flagcxProxyConnectionPool *connectionPool,
                   flagcxHeteroComm_t comm, int *asyncOpCount) {
  flagcxResult_t ret = flagcxSuccess;
  struct flagcxSocket *sock = &peer->sock;
  struct flagcxProxyAsyncOp *asyncOp = new (std::nothrow) flagcxProxyAsyncOp{};
  if (asyncOp == NULL)
    return flagcxSystemError;

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
  delete asyncOp;
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

  if (!proxyConn->sameProcess && proxyConn->remoteConnection != NULL &&
      proxyConn->connection != NULL &&
      (type == flagcxProxyMsgSetup || type == flagcxProxyMsgConnect)) {
    if (res == flagcxSuccess) {
      __atomic_store_n(&proxyConn->connection->state,
                       type == flagcxProxyMsgSetup ? connSetupDone
                                                   : connConnected,
                       __ATOMIC_RELEASE);
    } else {
      flagcxProxyRecordConnectionError(proxyConn->connection, res);
    }
  }

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
  // A NET relay on another rank must execute on that rank's device even when
  // both ranks share an address space. Give the source a local shadow and use
  // IPC staging for all cross-rank NET sends.
  const bool remoteNetDevice =
      transport == TRANSPORT_NET && send && proxyRank != comm->rank;
  proxyConn->sameProcess = !remoteNetDevice &&
                                   (comm->peerInfo[proxyRank].hostHash ==
                                    comm->peerInfo[comm->rank].hostHash) &&
                                   (comm->peerInfo[proxyRank].pidHash ==
                                    comm->peerInfo[comm->rank].pidHash)
                               ? 1
                               : 0;
  proxyConn->connection = NULL;
  proxyConn->remoteConnection = NULL;
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
  proxyConn->remoteConnection = resp.connection;
  if (proxyConn->remoteConnection == NULL) {
    WARN("flagcxProxyConnect: service thread returned NULL connection for rank "
         "%d -> peer %d",
         comm->rank, proxyRank);
    return flagcxInternalError;
  }
  if (proxyConn->sameProcess) {
    proxyConn->connection = proxyConn->remoteConnection;
  } else {
    struct flagcxProxyConnection *localConnection = NULL;
    FLAGCXCHECK(flagcxCalloc(&localConnection, 1));
    localConnection->transport = transport;
    localConnection->send = send;
    localConnection->tcomm = flagcxProxyTransportComm(transport, send);
    localConnection->result = flagcxSuccess;
    localConnection->state = connInitialized;
    proxyConn->connection = localConnection;
  }
  proxyConn->transport = transport;
  INFO(FLAGCX_PROXY,
       "flagcxProxyConnect rank %d -> peer %d local connection %p remote "
       "handle %p sameProcess %d",
       comm->rank, proxyRank, proxyConn->connection,
       proxyConn->remoteConnection, proxyConn->sameProcess);
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
  __atomic_store_n(&comm->proxyState->netAdaptor, comm->netAdaptor,
                   __ATOMIC_RELEASE);
  comm->proxyState->nRanks = comm->nRanks;
  comm->proxyState->abortFlag = comm->abortFlag;
  comm->proxyState->asyncResult = flagcxSuccess;
  comm->proxyState->cleanupResult = flagcxSuccess;
  comm->proxyState->stop = 0;
  comm->proxyState->rpcStopping = 0;
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
  // The service thread owns its connections. Free only caller-side shadows;
  // the remoteConnection values are opaque handles, never local allocations.
  auto releaseLocalConnection = [](struct flagcxProxyConnector *proxyConn) {
    if (proxyConn->connection != NULL &&
        proxyConn->connection != proxyConn->remoteConnection)
      free(proxyConn->connection);
    proxyConn->connection = NULL;
    proxyConn->remoteConnection = NULL;
  };
  for (int peer = 0; peer < comm->nRanks; peer++) {
    for (int c = 0; c < MAXCHANNELS; c++) {
      if (comm->channels[c].peers == NULL ||
          comm->channels[c].peers[peer] == NULL)
        continue;
      for (int index = 0; index < FLAGCX_MAX_CONNS; ++index) {
        releaseLocalConnection(
            &comm->channels[c].peers[peer]->recv[index].proxyConn);
        releaseLocalConnection(
            &comm->channels[c].peers[peer]->send[index].proxyConn);
      }
    }
    if (comm->gproxyConn != NULL)
      releaseLocalConnection(&comm->gproxyConn[peer]);
  }
  return flagcxSuccess;
}

static uint64_t flagcxProxyMonotonicNs() {
  struct timespec now = {};
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
    return 0;
  return uint64_t(now.tv_sec) * 1000000000ULL + now.tv_nsec;
}

static flagcxResult_t
flagcxProxyAcknowledgeRelayRelease(flagcxHeteroComm *comm,
                                   flagcxProxyConnector *connector,
                                   uint64_t deadlineNs) {
  if (deadlineNs == 0 || flagcxProxyMonotonicNs() >= deadlineNs)
    return flagcxInternalError;
  FLAGCXCHECK(flagcxProxyCallAsync(comm, connector, flagcxProxyMsgReleaseRelay,
                                   NULL, 0, 0, connector));
  while (flagcxProxyMonotonicNs() < deadlineNs) {
    flagcxResult_t result =
        flagcxPollProxyResponse(comm, connector, NULL, connector);
    if (result != flagcxInProgress)
      return result;
    usleep(1000);
  }
  flagcxProxyForgetResponse(comm, connector);
  return flagcxInternalError;
}

flagcxResult_t flagcxProxyStop(struct flagcxHeteroComm *comm) {
  if (comm->proxyState->initialized != 1) {
    return flagcxSuccess;
  }

  // Importers close first, while relay service threads and their exported NET
  // buffers are still alive. The relay keeps its allocation if this ACK is
  // missing, including when an in-flight operation has uncertain ownership.
  flagcxResult_t relayCleanupResult = flagcxSuccess;
  bool relayDeviceSet = false;
  const uint64_t nowNs = flagcxProxyMonotonicNs();
  const uint64_t releaseDeadlineNs = nowNs == 0 ? 0 : nowNs + 30000000000ULL;
  for (int peer = 0; peer < comm->nRanks; ++peer) {
    for (int channel = 0; channel < MAXCHANNELS; ++channel) {
      if (comm->channels[channel].peers == NULL ||
          comm->channels[channel].peers[peer] == NULL)
        continue;
      for (int index = 0; index < FLAGCX_MAX_CONNS; ++index) {
        auto *connector =
            &comm->channels[channel].peers[peer]->send[index].proxyConn;
        auto *connection = connector->connection;
        if (connection == NULL || connection->relayBufferImport == NULL)
          continue;
        if (!relayDeviceSet) {
          flagcxResult_t setResult = deviceAdaptor->setDevice(comm->cudaDev);
          if (setResult != flagcxSuccess) {
            if (relayCleanupResult == flagcxSuccess)
              relayCleanupResult = setResult;
            continue;
          }
          relayDeviceSet = true;
        }
        while (__atomic_load_n(&connection->relayActiveOps, __ATOMIC_ACQUIRE) !=
                   0 &&
               releaseDeadlineNs != 0 &&
               flagcxProxyMonotonicNs() < releaseDeadlineNs)
          usleep(1000);
        if (__atomic_load_n(&connection->relayActiveOps, __ATOMIC_ACQUIRE) !=
            0) {
          WARN("PXN relay mapping still has active sends at shutdown; "
               "retaining it until process exit");
          if (relayCleanupResult == flagcxSuccess)
            relayCleanupResult = flagcxInternalError;
          continue;
        }
        INFO(FLAGCX_PROXY,
             "PXN closing persistent relay import: source %d peer %d "
             "channel %d",
             comm->rank, peer, channel);
        flagcxResult_t closeResult =
            deviceAdaptor->ipcMemHandleClose(connection->relayBufferImport);
        INFO(FLAGCX_PROXY, "PXN persistent relay close returned %d",
             closeResult);
        if (closeResult != flagcxSuccess) {
          if (relayCleanupResult == flagcxSuccess)
            relayCleanupResult = closeResult;
          continue;
        }
        connection->relayBufferImport = NULL;
        flagcxResult_t ackResult = flagcxProxyAcknowledgeRelayRelease(
            comm, connector, releaseDeadlineNs);
        if (ackResult != flagcxSuccess && relayCleanupResult == flagcxSuccess)
          relayCleanupResult = ackResult;
      }
    }
  }

  // Publish shutdown before waiting for rpcMutex. A writer waiting for socket
  // space checks rpcStopping and releases the mutex; response reads never wait
  // for a complete frame while holding it.
  __atomic_store_n(&comm->proxyState->rpcStopping, 1, __ATOMIC_RELEASE);
  __atomic_store_n(&comm->proxyState->stop, 1, __ATOMIC_RELEASE);

  // The service thread checks stop after at most one poll interval. Closing
  // client sockets also interrupts any incomplete request on that side.
  pthread_mutex_lock(&comm->proxyState->rpcMutex);
  if (comm->proxyState->peerSocks != NULL) {
    for (int i = 0; i < comm->proxyState->nPeerSocks; i++) {
      if (comm->proxyState->peerSocks[i].fd >= 0) {
        int closeType = flagcxProxyMsgClose;
        // Best effort only: a blocked close message must not prevent teardown.
        (void)send(comm->proxyState->peerSocks[i].fd, &closeType,
                   sizeof(closeType), MSG_DONTWAIT | MSG_NOSIGNAL);
        flagcxSocketClose(&comm->proxyState->peerSocks[i]);
      }
    }
  }
  if (comm->proxyState->rpcReadStates != NULL) {
    for (int i = 0; i < comm->proxyState->nPeerSocks; i++)
      free(comm->proxyState->rpcReadStates[i].body);
    free(comm->proxyState->rpcReadStates);
    comm->proxyState->rpcReadStates = NULL;
  }
  while (comm->proxyState->expectedResponses != NULL) {
    auto *response = comm->proxyState->expectedResponses;
    comm->proxyState->expectedResponses = response->next;
    free(response->respBuff);
    free(response);
  }
  pthread_mutex_unlock(&comm->proxyState->rpcMutex);
  comm->proxyState->kernelState.stop = 1;
  INFO(FLAGCX_PROXY, "flagcxProxyStop: done");
  return relayCleanupResult;
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
