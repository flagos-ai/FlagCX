#include "flagcx_hetero.h"
#include "adaptor.h"
#include "global_comm.h"
#include "group.h"
#include "net.h"
#include "onesided.h"
#include "param.h"
#include "sym_heap.h"
#include "transport.h"
#include "type.h"

#include <climits>
#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifndef FLAGCX_RMA_QUEUE_SIZE
#define FLAGCX_RMA_QUEUE_SIZE 256
#endif
#ifndef FLAGCX_RMA_BATCH_MAX
#define FLAGCX_RMA_BATCH_MAX 256
#endif
#define FLAGCX_RMA_BATCH_MAX_LIMIT 256

FLAGCX_PARAM(RmaQueueSize, "RMA_QUEUE_SIZE", FLAGCX_RMA_QUEUE_SIZE);
FLAGCX_PARAM(RmaBatchMax, "RMA_BATCH_MAX", FLAGCX_RMA_BATCH_MAX);
FLAGCX_PARAM(RmaStreamOps, "RMA_STREAM_OPS",
             0); // 0 = HOST_FUNC (default), 1 = STREAM_OPS

// ---- Stream→Proxy ready signal infrastructure ----

// Context for launchHostFunc callbacks (HOST_FUNC path).
// Allocated per-op, freed inside the callback.
struct flagcxRmaReadyCtx {
  volatile uint64_t *readySeqsCpu; // pointer to proxy's readySeqsCpu[peer]
  uint64_t opSeq;                  // the sequence number to signal
};

// Host-func callback: signals proxy that GPU stream work is done (source buffer
// data is committed). Called by driver after all prior stream ops complete.
static void flagcxRmaReadyHostFunc(void *arg) {
  struct flagcxRmaReadyCtx *ctx = (struct flagcxRmaReadyCtx *)arg;
  __atomic_store_n(ctx->readySeqsCpu, ctx->opSeq, __ATOMIC_RELEASE);
  free(ctx);
}

// Context for launchHostFunc done-wait callbacks (HOST_FUNC path).
struct flagcxRmaDoneWaitCtx {
  volatile uint64_t *doneSeqsCpu; // pointer to proxy's doneSeqsCpu[peer]
  uint64_t opSeq;                 // sequence to wait for
  volatile int *rmaError;         // proxy error flag
  pthread_mutex_t *doneMutex;     // proxy done condvar mutex
  pthread_cond_t *doneCond;       // proxy done condvar
};

// Host-func callback: blocks stream until proxy signals completion.
// Uses condvar to sleep instead of spinning — zero CPU while waiting.
static void flagcxRmaDoneWaitHostFunc(void *arg) {
  struct flagcxRmaDoneWaitCtx *ctx = (struct flagcxRmaDoneWaitCtx *)arg;
  pthread_mutex_lock(ctx->doneMutex);
  while (__atomic_load_n(ctx->doneSeqsCpu, __ATOMIC_ACQUIRE) < ctx->opSeq) {
    if (__atomic_load_n(ctx->rmaError, __ATOMIC_ACQUIRE)) {
      break;
    }
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_nsec += 1000000; // 1ms timeout as safety net
    if (ts.tv_nsec >= 1000000000) {
      ts.tv_sec++;
      ts.tv_nsec -= 1000000000;
    }
    pthread_cond_timedwait(ctx->doneCond, ctx->doneMutex, &ts);
  }
  pthread_mutex_unlock(ctx->doneMutex);
  free(ctx);
}

// Signal proxy that source data is ready (stream-ordered).
// Uses streamWriteValue64 (STREAM_OPS) or launchHostFunc (HOST_FUNC).
static flagcxResult_t flagcxRmaSignalReady(struct flagcxRmaProxyState *proxy,
                                           int peer, uint64_t opSeq,
                                           flagcxStream_t stream) {
  if (proxy->useStreamOps) {
    // STREAM_OPS: GPU writes readySeqsDev directly (zero-overhead hardware op)
    return deviceAdaptor->streamWriteValue64(stream, &proxy->readySeqsDev[peer],
                                             opSeq, 0);
  } else {
    // HOST_FUNC: launch callback on stream that writes readySeqsCpu
    struct flagcxRmaReadyCtx *ctx =
        (struct flagcxRmaReadyCtx *)malloc(sizeof(*ctx));
    if (ctx == NULL)
      return flagcxSystemError;
    ctx->readySeqsCpu = &proxy->readySeqsCpu[peer];
    ctx->opSeq = opSeq;
    return deviceAdaptor->launchHostFunc(stream, flagcxRmaReadyHostFunc, ctx);
  }
}

// Wait for proxy completion (stream-ordered).
// Uses streamWaitValue64 (STREAM_OPS) or launchHostFunc (HOST_FUNC).
static flagcxResult_t flagcxRmaWaitDone(struct flagcxRmaProxyState *proxy,
                                        int peer, uint64_t opSeq,
                                        flagcxStream_t stream) {
  if (proxy->useStreamOps) {
    // STREAM_OPS: GPU waits on doneSeqsDev (hardware poll, no CPU involvement)
    return deviceAdaptor->streamWaitValue64(stream, &proxy->doneSeqsDev[peer],
                                            opSeq, 0);
  } else {
    // HOST_FUNC: launch callback that waits on doneCond until done
    struct flagcxRmaDoneWaitCtx *ctx =
        (struct flagcxRmaDoneWaitCtx *)malloc(sizeof(*ctx));
    if (ctx == NULL)
      return flagcxSystemError;
    ctx->doneSeqsCpu = &proxy->doneSeqsCpu[peer];
    ctx->opSeq = opSeq;
    ctx->rmaError = &proxy->rmaError;
    ctx->doneMutex = &proxy->doneMutex;
    ctx->doneCond = &proxy->doneCond;
    return deviceAdaptor->launchHostFunc(stream, flagcxRmaDoneWaitHostFunc,
                                         ctx);
  }
}

// ---- Circular buffer helpers ----

static inline bool
flagcxRmaProxyCircularBufFull(struct flagcxRmaProxyState *proxy, int peer) {
  uint32_t pi = __atomic_load_n(&proxy->pis[peer], __ATOMIC_RELAXED);
  uint32_t ci = __atomic_load_n(&proxy->cis[peer], __ATOMIC_ACQUIRE);
  return (pi - ci) >= proxy->queueSize;
}

static inline bool
flagcxRmaProxyCircularBufEmpty(struct flagcxRmaProxyState *proxy, int peer) {
  uint32_t ci = __atomic_load_n(&proxy->cis[peer], __ATOMIC_RELAXED);
  uint32_t pi = __atomic_load_n(&proxy->pis[peer], __ATOMIC_ACQUIRE);
  return ci >= pi;
}

static flagcxResult_t flagcxRmaProxyEnqueueDesc(
    struct flagcxRmaProxyState *proxy, int peer, struct flagcxRmaDesc *desc,
    bool streamSyncReady = false, uint64_t *assignedSeq = NULL) {
  pthread_mutex_lock(&proxy->peerProducerMutexes[peer]);
  while (flagcxRmaProxyCircularBufFull(proxy, peer)) {
    if (__atomic_load_n(&proxy->rmaError, __ATOMIC_ACQUIRE)) {
      pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
      return flagcxRemoteError;
    }
    pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
    sched_yield();
    pthread_mutex_lock(&proxy->peerProducerMutexes[peer]);
  }
  uint32_t pi = __atomic_load_n(&proxy->pis[peer], __ATOMIC_RELAXED);
  uint32_t idx = pi & proxy->queueMask;
  desc->peer = peer;
  desc->next = NULL;
  desc->request = NULL;
  desc->opSeq = __atomic_add_fetch(&proxy->opSeqs[peer], 1, __ATOMIC_RELAXED);
  proxy->circularBuffers[(size_t)peer * proxy->queueSize + idx] = desc;
  // For non-stream callers: CPU data is already committed, auto-advance
  // readySeq so the proxy won't wait on it. Stream-path callers signal via
  // stream ops.
  if (!streamSyncReady && proxy->readySeqsCpu != NULL) {
    __atomic_store_n(&proxy->readySeqsCpu[peer], desc->opSeq, __ATOMIC_RELEASE);
  }
  // RELEASE so the progress thread sees desc contents before the pi bump.
  __atomic_store_n(&proxy->pis[peer], pi + 1, __ATOMIC_RELEASE);
  if (assignedSeq != NULL)
    *assignedSeq = desc->opSeq;
  pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
  return flagcxSuccess;
}

static flagcxResult_t
flagcxRmaProxyEnqueueDescBatch(struct flagcxRmaProxyState *proxy, int peer,
                               struct flagcxRmaDesc **descs, size_t count,
                               size_t *enqueued) {
  *enqueued = 0;
  if (count == 0)
    return flagcxSuccess;

  pthread_mutex_lock(&proxy->peerProducerMutexes[peer]);
  for (size_t i = 0; i < count; i++) {
    while (flagcxRmaProxyCircularBufFull(proxy, peer)) {
      if (__atomic_load_n(&proxy->rmaError, __ATOMIC_ACQUIRE)) {
        pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
        return flagcxRemoteError;
      }
      pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
      sched_yield();
      pthread_mutex_lock(&proxy->peerProducerMutexes[peer]);
    }

    uint32_t pi = __atomic_load_n(&proxy->pis[peer], __ATOMIC_RELAXED);
    uint32_t idx = pi & proxy->queueMask;
    struct flagcxRmaDesc *desc = descs[i];
    desc->peer = peer;
    desc->next = NULL;
    desc->request = NULL;
    desc->opSeq = __atomic_add_fetch(&proxy->opSeqs[peer], 1, __ATOMIC_RELAXED);
    proxy->circularBuffers[(size_t)peer * proxy->queueSize + idx] = desc;
    // Batch path is always non-stream; auto-advance readySeq
    if (proxy->readySeqsCpu != NULL) {
      __atomic_store_n(&proxy->readySeqsCpu[peer], desc->opSeq,
                       __ATOMIC_RELEASE);
    }
    __atomic_store_n(&proxy->pis[peer], pi + 1, __ATOMIC_RELEASE);
    (*enqueued)++;
  }
  pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
  return flagcxSuccess;
}

// Post a single desc via the net adaptor. desc->request is populated on
// success. Returns the adaptor's result.
static flagcxResult_t flagcxRmaProxyPostOp(struct flagcxHeteroComm *comm,
                                           struct flagcxRmaDesc *desc,
                                           void *sendComm) {
  int p = desc->peer;
  void **srcHandles = NULL, **dstHandles = NULL;
  if (desc->size > 0 && desc->srcMrIdx >= 0) {
    srcHandles = (void **)comm->oneSideHandles[desc->srcMrIdx];
    dstHandles = (void **)comm->oneSideHandles[desc->dstMrIdx];
  }
  switch (desc->type) {
    case FLAGCX_RMA_PUT:
      return comm->netAdaptor->iput(sendComm, desc->srcOff, desc->dstOff,
                                    desc->size, comm->rank, p, srcHandles,
                                    dstHandles, &desc->request);
    case FLAGCX_RMA_PUT_SIGNAL: {
      void **sigHandles = (void **)comm->signalHandle;
      return comm->netAdaptor->iputSignal(
          sendComm, desc->srcOff, desc->dstOff, desc->size, comm->rank, p,
          srcHandles, dstHandles, desc->signalOff, sigHandles,
          desc->signalValue, &desc->request);
    }
    case FLAGCX_RMA_GET:
      return comm->netAdaptor->iget(
          sendComm, desc->srcOff, desc->dstOff, desc->size, p /* srcRank */,
          comm->rank /* dstRank */, srcHandles, dstHandles, &desc->request);
    case FLAGCX_RMA_PUT_VALUE: {
      struct flagcxOneSideHandleInfo *stagingH = comm->stagingHandle;
      if (stagingH == NULL || stagingH->baseVas == NULL) {
        WARN("flagcxRmaProxyPostOp: staging handles not initialized");
        return flagcxInternalError;
      }
      // Use per-peer slot to avoid staging buffer races: the NIC may still be
      // reading a previous peer's slot when we post the next putValue.
      size_t slot = (size_t)p * sizeof(uint64_t);
      volatile uint64_t *staging =
          (volatile uint64_t *)(stagingH->baseVas[comm->rank] + slot);
      *staging = desc->putValue;
      void **stagingHandles = (void **)stagingH;
      void **dstH = (void **)comm->oneSideHandles[desc->dstMrIdx];
      return comm->netAdaptor->iput(sendComm, slot, desc->dstOff,
                                    sizeof(uint64_t), comm->rank, p,
                                    stagingHandles, dstH, &desc->request);
    }
  }
  return flagcxInternalError;
}

static flagcxResult_t flagcxRmaProxyPostPutBatch(struct flagcxHeteroComm *comm,
                                                 struct flagcxRmaDesc **descs,
                                                 int count, void *sendComm,
                                                 void **requests, int *posted) {
  *posted = 0;
  if (count <= 0)
    return flagcxSuccess;
  if (comm->netAdaptor == NULL || comm->netAdaptor->iputBatch == NULL)
    return flagcxNotSupported;

  int p = descs[0]->peer;
  uint64_t srcOffs[FLAGCX_RMA_BATCH_MAX_LIMIT];
  uint64_t dstOffs[FLAGCX_RMA_BATCH_MAX_LIMIT];
  size_t sizes[FLAGCX_RMA_BATCH_MAX_LIMIT];
  for (int i = 0; i < count; i++) {
    srcOffs[i] = descs[i]->srcOff;
    dstOffs[i] = descs[i]->dstOff;
    sizes[i] = descs[i]->size;
    requests[i] = NULL;
  }
  assert(descs[0]->srcMrIdx >= 0 && descs[0]->dstMrIdx >= 0);
  void **srcHandles = (void **)comm->oneSideHandles[descs[0]->srcMrIdx];
  void **dstHandles = (void **)comm->oneSideHandles[descs[0]->dstMrIdx];
  return comm->netAdaptor->iputBatch(sendComm, count, srcOffs, dstOffs, sizes,
                                     comm->rank, p, srcHandles, dstHandles,
                                     requests, posted);
}

// Poll and retire completed descs at the head of inProgressQueues[peer].
// Returns after the head desc is not yet complete (enforces per-peer FIFO).
static bool
flagcxRmaProxyPollNonPersistCompletion(struct flagcxRmaProxyState *proxy,
                                       int peer) {
  struct flagcxHeteroComm *comm = proxy->comm;
  bool did = false;
  while (!flagcxIntruQueueEmpty(&proxy->inProgressQueues[peer])) {
    struct flagcxRmaDesc *desc =
        flagcxIntruQueueHead(&proxy->inProgressQueues[peer]);
    int done = 0;
    bool failed = false;
    if (desc->request != NULL) {
      flagcxResult_t res = comm->netAdaptor->test(desc->request, &done, NULL);
      if (res != flagcxSuccess) {
        WARN("flagcxRmaProxyPollNonPersistCompletion: test failed peer=%d "
             "res=%d",
             peer, (int)res);
        __atomic_store_n(&proxy->rmaError, 1, __ATOMIC_RELEASE);
        done = 1;
        failed = true; // retire without advancing counters
      }
    } else {
      // Issuance already failed; drain without advancing counters.
      done = 1;
      failed = true;
    }
    if (!done)
      break;
    flagcxIntruQueueDequeue(&proxy->inProgressQueues[peer]);
    __atomic_fetch_sub(&proxy->inFlights[peer], 1, __ATOMIC_RELAXED);
    if (!failed) {
      // Publish completion: doneSeqs with RELEASE so waiters acquire-see it.
      __atomic_store_n(&proxy->doneSeqs[peer], desc->opSeq, __ATOMIC_RELEASE);
      // Also write doneSeqsCpu for stream/host-func waiters.
      // In STREAM_OPS mode: this is a CPU mapping of doneSeqsDev (GPU-visible).
      // In HOST_FUNC mode: this is host memory polled by launchHostFunc
      // callback.
      if (proxy->doneSeqsCpu != NULL)
        __atomic_store_n(&proxy->doneSeqsCpu[peer], desc->opSeq,
                         __ATOMIC_RELEASE);
      __atomic_fetch_add(&proxy->completionCount, 1ULL, __ATOMIC_RELEASE);
      // Wake HOST_FUNC waiters sleeping on doneCond
      if (!proxy->useStreamOps) {
        pthread_mutex_lock(&proxy->doneMutex);
        pthread_cond_broadcast(&proxy->doneCond);
        pthread_mutex_unlock(&proxy->doneMutex);
      }
    }
    free(desc);
    did = true;
  }
  return did;
}

// Poll pending descs from the ring and issue them. On success advance
// cis[peer] and move the desc to inProgressQueues[peer]. On
// request-pool-full (flagcxInternalError) leave cis untouched and retry
// next round. On other errors mark rmaError and push to inProgress with
// NULL request so PollNonPersistCompletion retires it.
static bool flagcxRmaProxyPollNonPersistDesc(struct flagcxRmaProxyState *proxy,
                                             int peer, void *sendComm) {
  struct flagcxHeteroComm *comm = proxy->comm;
  bool did = false;
  while (!flagcxRmaProxyCircularBufEmpty(proxy, peer)) {
    uint32_t inFlight =
        __atomic_load_n(&proxy->inFlights[peer], __ATOMIC_RELAXED);
    if (inFlight >= proxy->queueSize)
      break;
    uint32_t ci = __atomic_load_n(&proxy->cis[peer], __ATOMIC_RELAXED);
    uint32_t idx = ci & proxy->queueMask;
    struct flagcxRmaDesc *desc =
        proxy->circularBuffers[(size_t)peer * proxy->queueSize + idx];

    // Poll readySeq: wait for GPU stream to signal source data is committed.
    // Both STREAM_OPS (streamWriteValue64) and HOST_FUNC (callback) write here.
    if (proxy->readySeqsCpu != NULL) {
      uint64_t ready =
          __atomic_load_n(&proxy->readySeqsCpu[peer], __ATOMIC_ACQUIRE);
      if (ready < desc->opSeq) {
        // GPU hasn't signaled ready yet; skip this peer for now
        break;
      }
    }

    bool canBatch = desc->type == FLAGCX_RMA_PUT && comm->netAdaptor != NULL &&
                    comm->netAdaptor->name != NULL &&
                    strcmp(comm->netAdaptor->name, "IB") == 0 &&
                    comm->netAdaptor->iputBatch != NULL;
    if (canBatch) {
      int64_t paramBatchMax = flagcxParamRmaBatchMax();
      if (paramBatchMax <= 0)
        paramBatchMax = 1;
      if (paramBatchMax > FLAGCX_RMA_BATCH_MAX_LIMIT)
        paramBatchMax = FLAGCX_RMA_BATCH_MAX_LIMIT;

      uint32_t pi = __atomic_load_n(&proxy->pis[peer], __ATOMIC_ACQUIRE);
      uint32_t ringAvailable = pi - ci;
      uint32_t flightAvailable = proxy->queueSize - inFlight;
      uint32_t batchLimit =
          ringAvailable < flightAvailable ? ringAvailable : flightAvailable;
      if (batchLimit > (uint32_t)paramBatchMax)
        batchLimit = (uint32_t)paramBatchMax;

      struct flagcxRmaDesc *descs[FLAGCX_RMA_BATCH_MAX_LIMIT];
      int batchCount = 0;
      for (; batchCount < (int)batchLimit; batchCount++) {
        uint32_t curIdx = (ci + (uint32_t)batchCount) & proxy->queueMask;
        struct flagcxRmaDesc *cur =
            proxy->circularBuffers[(size_t)peer * proxy->queueSize + curIdx];
        if (cur->type != FLAGCX_RMA_PUT || cur->srcMrIdx != desc->srcMrIdx ||
            cur->dstMrIdx != desc->dstMrIdx) {
          break;
        }
        descs[batchCount] = cur;
      }

      if (batchCount > 1) {
        void *requests[FLAGCX_RMA_BATCH_MAX_LIMIT];
        int posted = 0;
        flagcxResult_t res = flagcxRmaProxyPostPutBatch(
            comm, descs, batchCount, sendComm, requests, &posted);
        if (posted == 0) {
          if (res != flagcxSuccess && res != flagcxSystemError &&
              res != flagcxInternalError) {
            WARN("flagcxRmaProxyPollNonPersistDesc: batch op failed peer=%d "
                 "res=%d",
                 peer, (int)res);
            __atomic_store_n(&proxy->rmaError, 1, __ATOMIC_RELEASE);
          }
          break;
        }

        for (int i = 0; i < posted; i++) {
          descs[i]->request = requests[i];
          descs[i]->next = NULL;
          flagcxIntruQueueEnqueue(&proxy->inProgressQueues[peer], descs[i]);
        }
        __atomic_store_n(&proxy->cis[peer], ci + (uint32_t)posted,
                         __ATOMIC_RELEASE);
        __atomic_fetch_add(&proxy->inFlights[peer], (uint32_t)posted,
                           __ATOMIC_RELAXED);
        did = true;
        continue;
      }
    }

    desc->request = NULL;
    flagcxResult_t res = flagcxRmaProxyPostOp(comm, desc, sendComm);
    if (res == flagcxInternalError) {
      // Request pool exhausted; retry this slot next round (cis unchanged).
      break;
    }
    if (res != flagcxSuccess) {
      WARN("flagcxRmaProxyPollNonPersistDesc: op failed peer=%d type=%d "
           "res=%d",
           peer, (int)desc->type, (int)res);
      __atomic_store_n(&proxy->rmaError, 1, __ATOMIC_RELEASE);
      desc->request = NULL; // completion path will treat as done.
    }
    // RELEASE so the producer sees the slot freed.
    __atomic_store_n(&proxy->cis[peer], ci + 1, __ATOMIC_RELEASE);
    // Enqueue to inProgressQueues[peer] (progress-thread private).
    desc->next = NULL;
    flagcxIntruQueueEnqueue(&proxy->inProgressQueues[peer], desc);
    __atomic_fetch_add(&proxy->inFlights[peer], 1, __ATOMIC_RELAXED);
    did = true;
  }
  return did;
}

// Drain the peer's ring without posting: dequeue, advance cis, free each
// desc without bumping doneSeqs/completionCount. Called at shutdown when
// no sendComm is available and we cannot actually issue the ops.
static void flagcxRmaProxyDrainRing(struct flagcxRmaProxyState *proxy,
                                    int peer) {
  while (!flagcxRmaProxyCircularBufEmpty(proxy, peer)) {
    uint32_t ci = __atomic_load_n(&proxy->cis[peer], __ATOMIC_RELAXED);
    uint32_t idx = ci & proxy->queueMask;
    struct flagcxRmaDesc *desc =
        proxy->circularBuffers[(size_t)peer * proxy->queueSize + idx];
    __atomic_store_n(&proxy->cis[peer], ci + 1, __ATOMIC_RELEASE);
    free(desc);
  }
}

// One pass over all peers: poll completions and issue pending descs.
// Returns true if any progress was made.
static bool flagcxRmaProxyProgress(struct flagcxRmaProxyState *proxy,
                                   bool stopping, bool *anyOutstanding) {
  bool did = false;
  *anyOutstanding = false;
  // Read the cached fullSendComms once per pass (published exactly once
  // from the registration path; NULL until then). See
  // flagcxHeteroRmaProxyPublishSendComms() for why this is safe.
  void *const *fullSendComms =
      __atomic_load_n(&proxy->fullSendComms, __ATOMIC_ACQUIRE);
  for (int p = 0; p < proxy->nRanks; p++) {
    if (flagcxRmaProxyPollNonPersistCompletion(proxy, p))
      did = true;

    void *sendComm = (fullSendComms != NULL) ? fullSendComms[p] : NULL;
    if (sendComm != NULL) {
      if (flagcxRmaProxyPollNonPersistDesc(proxy, p, sendComm))
        did = true;
    } else if (!flagcxRmaProxyCircularBufEmpty(proxy, p)) {
      if (stopping) {
        // Shutdown with queued-but-unissued descs and no transport.
        // Drain to let the thread exit; flag the error so waiters fail.
        WARN("flagcxRmaProxyProgress: stop with queued descs but no "
             "sendComm peer=%d; draining",
             p);
        __atomic_store_n(&proxy->rmaError, 1, __ATOMIC_RELEASE);
        flagcxRmaProxyDrainRing(proxy, p);
        did = true;
      } else {
        // Pre-registration: caller enqueued an op before the full mesh
        // is ready. Surface as an error rather than spin forever.
        WARN("flagcxRmaProxyProgress: no sendComm for peer %d", p);
        __atomic_store_n(&proxy->rmaError, 1, __ATOMIC_RELEASE);
      }
    }

    if (!flagcxRmaProxyCircularBufEmpty(proxy, p) ||
        !flagcxIntruQueueEmpty(&proxy->inProgressQueues[p]))
      *anyOutstanding = true;
  }
  return did;
}

static void *flagcxRmaProxyProgressThread(void *arg) {
  struct flagcxRmaProxyState *proxy = (struct flagcxRmaProxyState *)arg;
  bool stopping = false;
  while (true) {
    if (__atomic_load_n(&proxy->stop, __ATOMIC_ACQUIRE))
      stopping = true;
    bool anyOutstanding = false;
    bool did = flagcxRmaProxyProgress(proxy, stopping, &anyOutstanding);
    if (stopping && !anyOutstanding && !did)
      break;
    if (!did)
      sched_yield();
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

  uint32_t qs = (uint32_t)flagcxParamRmaQueueSize();
  if (qs < 2 || (qs & (qs - 1)) != 0) {
    WARN("flagcxHeteroRmaProxyStart: invalid RMA queue size %u;"
         "FLAGCX_RMA_QUEUE_SIZE must be a power of two and >= 2",
         qs);
    free(proxy);
    return flagcxInvalidArgument;
  }
  proxy->queueSize = qs;
  proxy->queueMask = qs - 1;

  size_t ringBytes = (size_t)nRanks * qs * sizeof(struct flagcxRmaDesc *);
  proxy->circularBuffers = (struct flagcxRmaDesc **)calloc(1, ringBytes);
  proxy->pis = (volatile uint32_t *)calloc(nRanks, sizeof(uint32_t));
  proxy->cis = (volatile uint32_t *)calloc(nRanks, sizeof(uint32_t));
  proxy->inProgressQueues =
      (flagcxIntruQueue<struct flagcxRmaDesc, &flagcxRmaDesc::next> *)calloc(
          nRanks,
          sizeof(flagcxIntruQueue<struct flagcxRmaDesc, &flagcxRmaDesc::next>));
  proxy->peerProducerMutexes =
      (pthread_mutex_t *)calloc(nRanks, sizeof(pthread_mutex_t));
  proxy->opSeqs = (volatile uint64_t *)calloc(nRanks, sizeof(uint64_t));
  proxy->doneSeqs = (volatile uint64_t *)calloc(nRanks, sizeof(uint64_t));
  proxy->inFlights = (volatile uint32_t *)calloc(nRanks, sizeof(uint32_t));

  if (proxy->circularBuffers == NULL || proxy->pis == NULL ||
      proxy->cis == NULL || proxy->inProgressQueues == NULL ||
      proxy->peerProducerMutexes == NULL || proxy->opSeqs == NULL ||
      proxy->doneSeqs == NULL || proxy->inFlights == NULL) {
    WARN("flagcxHeteroRmaProxyStart: failed to allocate ring buffers");
    free(proxy->circularBuffers);
    free((void *)proxy->pis);
    free((void *)proxy->cis);
    free(proxy->inProgressQueues);
    free(proxy->peerProducerMutexes);
    free((void *)proxy->opSeqs);
    free((void *)proxy->doneSeqs);
    free((void *)proxy->inFlights);
    free(proxy);
    return flagcxSystemError;
  }

  for (int p = 0; p < nRanks; p++) {
    pthread_mutex_init(&proxy->peerProducerMutexes[p], NULL);
    flagcxIntruQueueConstruct(&proxy->inProgressQueues[p]);
  }

  pthread_mutex_init(&proxy->doneMutex, NULL);
  pthread_cond_init(&proxy->doneCond, NULL);

  // Allocate device memory for stream-based synchronization.
  proxy->doneSeqsDev = NULL;
  proxy->doneSeqsCpu = NULL;
  proxy->readySeqsDev = NULL;
  proxy->readySeqsCpu = NULL;
  if (deviceAdaptor->gdrMemAlloc != NULL) {
    flagcxResult_t memRes = deviceAdaptor->gdrMemAlloc(
        (void **)&proxy->doneSeqsDev, nRanks * sizeof(uint64_t), NULL);
    if (memRes != flagcxSuccess)
      proxy->doneSeqsDev = NULL;
    memRes = deviceAdaptor->gdrMemAlloc((void **)&proxy->readySeqsDev,
                                        nRanks * sizeof(uint64_t), NULL);
    if (memRes != flagcxSuccess)
      proxy->readySeqsDev = NULL;
  }

  // Map device memory to CPU if adaptor supports gdrPtrMmap (e.g. kunlunxin).
  // This gives the proxy thread direct CPU access to the device buffers.
  bool mmapDone = false, mmapReady = false;
  if (deviceAdaptor->gdrPtrMmap != NULL) {
    if (proxy->doneSeqsDev != NULL) {
      if (deviceAdaptor->gdrPtrMmap((void **)&proxy->doneSeqsCpu,
                                    proxy->doneSeqsDev,
                                    nRanks * sizeof(uint64_t)) == flagcxSuccess)
        mmapDone = true;
      else
        proxy->doneSeqsCpu = NULL;
    }
    if (proxy->readySeqsDev != NULL) {
      if (deviceAdaptor->gdrPtrMmap((void **)&proxy->readySeqsCpu,
                                    proxy->readySeqsDev,
                                    nRanks * sizeof(uint64_t)) == flagcxSuccess)
        mmapReady = true;
      else
        proxy->readySeqsCpu = NULL;
    }
  }

  // If no CPU mapping available, allocate separate host memory for HOST_FUNC.
  if (proxy->doneSeqsCpu == NULL)
    proxy->doneSeqsCpu = (volatile uint64_t *)calloc(nRanks, sizeof(uint64_t));
  if (proxy->readySeqsCpu == NULL)
    proxy->readySeqsCpu = (volatile uint64_t *)calloc(nRanks, sizeof(uint64_t));

  // Zero-init device buffers
  if (proxy->doneSeqsDev != NULL)
    deviceAdaptor->deviceMemset(proxy->doneSeqsDev, 0,
                                nRanks * sizeof(uint64_t), flagcxMemDevice,
                                NULL);
  if (proxy->readySeqsDev != NULL)
    deviceAdaptor->deviceMemset(proxy->readySeqsDev, 0,
                                nRanks * sizeof(uint64_t), flagcxMemDevice,
                                NULL);

  // STREAM_OPS requires: device buffers AND successful CPU mmap of both
  // (so proxy thread can write doneSeqsDev from CPU via the mmap'd pointer).
  // If mmap failed, doneSeqsCpu/readySeqsCpu are separate host allocations
  // and STREAM_OPS would hang (GPU waits on device memory proxy never updates).
  proxy->useStreamOps =
      (flagcxParamRmaStreamOps() == 1 && proxy->doneSeqsDev != NULL &&
       proxy->readySeqsDev != NULL && mmapDone && mmapReady)
          ? 1
          : 0;
  INFO(FLAGCX_INIT, "RMA proxy sync method: %s",
       proxy->useStreamOps ? "STREAM_OPS" : "HOST_FUNC");

  proxy->stop = 0;
  comm->rmaProxy = proxy;

  if (pthread_create(&proxy->thread, NULL, flagcxRmaProxyProgressThread,
                     proxy) != 0) {
    WARN("flagcxHeteroRmaProxyStart: pthread_create failed");
    if (proxy->useStreamOps && deviceAdaptor->gdrPtrMunmap != NULL) {
      if (proxy->doneSeqsCpu != NULL)
        deviceAdaptor->gdrPtrMunmap((void *)proxy->doneSeqsCpu,
                                    nRanks * sizeof(uint64_t));
      if (proxy->readySeqsCpu != NULL)
        deviceAdaptor->gdrPtrMunmap((void *)proxy->readySeqsCpu,
                                    nRanks * sizeof(uint64_t));
    } else {
      free((void *)proxy->doneSeqsCpu);
      free((void *)proxy->readySeqsCpu);
    }
    if (proxy->doneSeqsDev != NULL)
      deviceAdaptor->gdrMemFree(proxy->doneSeqsDev, NULL);
    if (proxy->readySeqsDev != NULL)
      deviceAdaptor->gdrMemFree(proxy->readySeqsDev, NULL);
    for (int p = 0; p < nRanks; p++)
      pthread_mutex_destroy(&proxy->peerProducerMutexes[p]);
    pthread_cond_destroy(&proxy->doneCond);
    pthread_mutex_destroy(&proxy->doneMutex);
    free(proxy->circularBuffers);
    free((void *)proxy->pis);
    free((void *)proxy->cis);
    free(proxy->inProgressQueues);
    free(proxy->peerProducerMutexes);
    free((void *)proxy->opSeqs);
    free((void *)proxy->doneSeqs);
    free((void *)proxy->inFlights);
    free(proxy);
    comm->rmaProxy = NULL;
    return flagcxSystemError;
  }

  INFO(FLAGCX_INIT, "RMA progress thread started (nRanks=%d queueSize=%u)",
       nRanks, qs);
  return flagcxSuccess;
}

flagcxResult_t flagcxHeteroRmaProxyStop(flagcxHeteroComm_t comm) {
  struct flagcxRmaProxyState *proxy = comm->rmaProxy;
  if (proxy == NULL)
    return flagcxSuccess;

  __atomic_store_n(&proxy->stop, 1, __ATOMIC_RELEASE);
  pthread_join(proxy->thread, NULL);

  // Free IPC state (D2D bypass peer pointers)
  flagcxHeteroRmaIpcDestroy(comm);

  for (int p = 0; p < proxy->nRanks; p++)
    pthread_mutex_destroy(&proxy->peerProducerMutexes[p]);

  pthread_cond_destroy(&proxy->doneCond);
  pthread_mutex_destroy(&proxy->doneMutex);

  // Free CPU mappings / host memory
  if (proxy->useStreamOps && deviceAdaptor->gdrPtrMunmap != NULL) {
    if (proxy->doneSeqsCpu != NULL)
      deviceAdaptor->gdrPtrMunmap((void *)proxy->doneSeqsCpu,
                                  proxy->nRanks * sizeof(uint64_t));
    if (proxy->readySeqsCpu != NULL)
      deviceAdaptor->gdrPtrMunmap((void *)proxy->readySeqsCpu,
                                  proxy->nRanks * sizeof(uint64_t));
  } else {
    free((void *)proxy->doneSeqsCpu);
    free((void *)proxy->readySeqsCpu);
  }

  // Free device memory
  if (proxy->doneSeqsDev != NULL)
    deviceAdaptor->gdrMemFree(proxy->doneSeqsDev, NULL);
  if (proxy->readySeqsDev != NULL)
    deviceAdaptor->gdrMemFree(proxy->readySeqsDev, NULL);

  free(proxy->circularBuffers);
  free((void *)proxy->pis);
  free((void *)proxy->cis);
  free(proxy->inProgressQueues);
  free(proxy->peerProducerMutexes);
  free((void *)proxy->opSeqs);
  free((void *)proxy->doneSeqs);
  free((void *)proxy->inFlights);
  free(proxy);
  comm->rmaProxy = NULL;
  return flagcxSuccess;
}

flagcxResult_t
flagcxHeteroRmaProxyPublishSendComms(flagcxHeteroComm_t comm,
                                     void *const *fullSendComms) {
  if (comm == NULL || comm->rmaProxy == NULL)
    return flagcxInvalidArgument;
  if (fullSendComms == NULL)
    return flagcxSuccess;
  // Publish only if unset; later registrations reuse the same array.
  void *const *cur =
      __atomic_load_n(&comm->rmaProxy->fullSendComms, __ATOMIC_ACQUIRE);
  if (cur == NULL) {
    __atomic_store_n(&comm->rmaProxy->fullSendComms, fullSendComms,
                     __ATOMIC_RELEASE);
  }
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
  // Final rmaError check: kernel proxy or network failures set rmaError;
  // catch errors that occurred after doneSeqs reached the target.
  if (__atomic_load_n(&proxy->rmaError, __ATOMIC_ACQUIRE))
    return flagcxRemoteError;
  return flagcxSuccess;
}

flagcxResult_t flagcxHeteroFlushRmaStream(flagcxHeteroComm_t comm, int peer,
                                          uint64_t seq, flagcxStream_t stream) {
  struct flagcxRmaProxyState *proxy = comm->rmaProxy;
  if (proxy == NULL || seq == 0)
    return flagcxSuccess;
  if (peer < 0 || peer >= proxy->nRanks) {
    WARN("flagcxHeteroFlushRmaStream: peer %d out of range (nRanks=%d)", peer,
         proxy->nRanks);
    return flagcxInvalidArgument;
  }
  if (stream == NULL || !proxy->useStreamOps || proxy->doneSeqsDev == NULL) {
    // Fallback to host-side spin if stream or STREAM_OPS not available.
    // In HOST_FUNC mode, proxy writes doneSeqsCpu (host memory), so GPU-side
    // streamWaitValue64 on doneSeqsDev would stall forever.
    return flagcxHeteroFlushRma(comm, peer, seq);
  }
  // GPU-side wait: stream stalls until doneSeqsDev[peer] >= seq
  return deviceAdaptor->streamWaitValue64(stream, &proxy->doneSeqsDev[peer],
                                          seq, 0 /*GEQ*/);
}

flagcxResult_t flagcxHeteroFlushAllRma(flagcxHeteroComm_t comm) {
  struct flagcxRmaProxyState *proxy = comm->rmaProxy;
  if (proxy == NULL)
    return flagcxSuccess;
  for (int p = 0; p < proxy->nRanks; p++) {
    uint64_t target = __atomic_load_n(&proxy->opSeqs[p], __ATOMIC_RELAXED);
    if (target == 0)
      continue;
    while (__atomic_load_n(&proxy->doneSeqs[p], __ATOMIC_ACQUIRE) < target) {
      if (__atomic_load_n(&proxy->rmaError, __ATOMIC_ACQUIRE))
        return flagcxRemoteError;
      usleep(100);
    }
  }
  // Final rmaError check: kernel proxy or network failures set rmaError;
  // catch errors that occurred after doneSeqs reached the target.
  if (__atomic_load_n(&proxy->rmaError, __ATOMIC_ACQUIRE))
    return flagcxRemoteError;
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
                               int srcMrIdx, int dstMrIdx, bool streamSyncReady,
                               uint64_t *assignedSeq) {
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
  struct flagcxRmaDesc *desc = (struct flagcxRmaDesc *)calloc(1, sizeof(*desc));
  if (desc == NULL)
    return flagcxSystemError;
  desc->type = FLAGCX_RMA_PUT;
  desc->srcOff = (uint64_t)srcOffset;
  desc->dstOff = (uint64_t)dstOffset;
  desc->size = size;
  desc->srcMrIdx = srcMrIdx;
  desc->dstMrIdx = dstMrIdx;
  flagcxResult_t res = flagcxRmaProxyEnqueueDesc(comm->rmaProxy, peer, desc,
                                                 streamSyncReady, assignedSeq);
  if (res != flagcxSuccess)
    free(desc);
  return res;
}

flagcxResult_t flagcxHeteroBatchPut(flagcxHeteroComm_t comm, int peer,
                                    const size_t *srcOffsets,
                                    const size_t *dstOffsets,
                                    const size_t *sizes, const int *srcMrIdxs,
                                    const int *dstMrIdxs, size_t count) {
  if (count == 0)
    return flagcxSuccess;
  if (comm == NULL || srcOffsets == NULL || dstOffsets == NULL ||
      sizes == NULL || srcMrIdxs == NULL || dstMrIdxs == NULL)
    return flagcxInvalidArgument;
  if (comm->netAdaptor == NULL || comm->netAdaptor->iput == NULL)
    return flagcxNotSupported;
  if (peer < 0 || peer >= comm->nRanks) {
    WARN("flagcxHeteroBatchPut: peer %d out of range (nRanks=%d)", peer,
         comm->nRanks);
    return flagcxInvalidArgument;
  }
  if (comm->rmaProxy == NULL) {
    WARN("flagcxHeteroBatchPut: rmaProxy not initialized");
    return flagcxInternalError;
  }

  struct flagcxRmaDesc **descs =
      (struct flagcxRmaDesc **)calloc(count, sizeof(struct flagcxRmaDesc *));
  if (descs == NULL)
    return flagcxSystemError;

  for (size_t i = 0; i < count; i++) {
    struct flagcxRmaDesc *desc =
        (struct flagcxRmaDesc *)calloc(1, sizeof(*desc));
    if (desc == NULL) {
      for (size_t j = 0; j < i; j++)
        free(descs[j]);
      free(descs);
      return flagcxSystemError;
    }
    desc->type = FLAGCX_RMA_PUT;
    desc->srcOff = (uint64_t)srcOffsets[i];
    desc->dstOff = (uint64_t)dstOffsets[i];
    desc->size = sizes[i];
    desc->srcMrIdx = srcMrIdxs[i];
    desc->dstMrIdx = dstMrIdxs[i];
    descs[i] = desc;
  }

  size_t enqueued = 0;
  flagcxResult_t res = flagcxRmaProxyEnqueueDescBatch(comm->rmaProxy, peer,
                                                      descs, count, &enqueued);
  if (res != flagcxSuccess) {
    for (size_t i = enqueued; i < count; i++)
      free(descs[i]);
  }
  free(descs);
  return res;
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
  struct flagcxRmaDesc *desc = (struct flagcxRmaDesc *)calloc(1, sizeof(*desc));
  if (desc == NULL)
    return flagcxSystemError;
  desc->type = FLAGCX_RMA_GET;
  desc->srcOff = (uint64_t)srcOffset;
  desc->dstOff = (uint64_t)dstOffset;
  desc->size = size;
  desc->srcMrIdx = srcMrIdx;
  desc->dstMrIdx = dstMrIdx;
  flagcxResult_t res = flagcxRmaProxyEnqueueDesc(comm->rmaProxy, peer, desc);
  if (res != flagcxSuccess)
    free(desc);
  return res;
}

flagcxResult_t flagcxHeteroPutSignal(flagcxHeteroComm_t comm, int peer,
                                     size_t srcOffset, size_t dstOffset,
                                     size_t size, size_t signalOffset,
                                     int srcMrIdx, int dstMrIdx,
                                     uint64_t signalValue, bool streamSyncReady,
                                     uint64_t *assignedSeq) {
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
  struct flagcxRmaDesc *desc = (struct flagcxRmaDesc *)calloc(1, sizeof(*desc));
  if (desc == NULL)
    return flagcxSystemError;
  desc->type = FLAGCX_RMA_PUT_SIGNAL;
  desc->srcOff = (uint64_t)srcOffset;
  desc->dstOff = (uint64_t)dstOffset;
  desc->size = size;
  // For signal-only (size==0) there are no data MR handles.
  desc->srcMrIdx = (size > 0) ? srcMrIdx : -1;
  desc->dstMrIdx = (size > 0) ? dstMrIdx : -1;
  desc->signalOff = (uint64_t)signalOffset;
  desc->signalValue = signalValue;
  flagcxResult_t res = flagcxRmaProxyEnqueueDesc(comm->rmaProxy, peer, desc,
                                                 streamSyncReady, assignedSeq);
  if (res != flagcxSuccess)
    free(desc);
  return res;
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
  struct flagcxOneSideHandleInfo *info = comm->signalHandle;
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
  if (dstMrIdx < 0 || dstMrIdx >= comm->oneSideHandleCount) {
    WARN("flagcxHeteroPutValue: dstMrIdx %d out of range (count=%d)", dstMrIdx,
         comm->oneSideHandleCount);
    return flagcxInvalidArgument;
  }
  if (comm->rmaProxy == NULL) {
    WARN("flagcxHeteroPutValue: rmaProxy not initialized");
    return flagcxInternalError;
  }
  struct flagcxRmaDesc *desc = (struct flagcxRmaDesc *)calloc(1, sizeof(*desc));
  if (desc == NULL)
    return flagcxSystemError;
  desc->type = FLAGCX_RMA_PUT_VALUE;
  desc->dstOff = (uint64_t)dstOffset;
  desc->dstMrIdx = dstMrIdx;
  desc->size = 0;
  desc->srcMrIdx = -1;
  desc->putValue = value;
  flagcxResult_t res = flagcxRmaProxyEnqueueDesc(comm->rmaProxy, peer, desc);
  if (res != flagcxSuccess)
    free(desc);
  return res;
}

// ---- Intra-node topology helper ----

// Get pointer to peer's buffer in the flat-mapped symmetric VA range.
// Equivalent to NCCL's ncclDevrGetLsaRankPtr — computes local VA for peer's
// memory. Returns NULL if symmetric memory not available for this handle.
static inline void *flagcxGetIntraRankPtr(flagcxSymWindow_t symWin,
                                          int peerLocalRank, size_t offset) {
  if (symWin == NULL || symWin->flatBase == NULL || !symWin->isVMM)
    return NULL;

  return (void *)((uintptr_t)symWin->flatBase +
                  (size_t)peerLocalRank * symWin->allocSize + offset);
}

static inline bool flagcxIsIntraNode(flagcxHeteroComm_t comm, int peer) {
  if (comm->rankToNode == NULL)
    return false;
  return comm->rankToNode[peer] == comm->node;
}

// ---- IPC state initialization ----

flagcxResult_t flagcxHeteroRmaIpcInit(flagcxHeteroComm_t comm) {
  struct flagcxRmaProxyState *proxy = comm->rmaProxy;
  if (proxy == NULL)
    return flagcxInternalError;

  // Check if D2D memory is available (VMM OR IPC)
  bool hasD2dMemory = false;

  // Check VMM (symmetric memory flat VA)
  for (int h = 0; h < comm->oneSideHandleCount; h++) {
    struct flagcxOneSideHandleInfo *info = comm->oneSideHandles[h];
    if (info != NULL && info->symWin != NULL &&
        info->symWin->flatBase != NULL && info->symWin->isVMM) {
      hasD2dMemory = true;
      break;
    }
  }
  if (comm->signalHandle != NULL && comm->signalHandle->symWin != NULL &&
      comm->signalHandle->symWin->flatBase != NULL &&
      comm->signalHandle->symWin->isVMM) {
    hasD2dMemory = true;
  }

  // Check IPC table for matching entries
  if (!hasD2dMemory && comm->ipcTable != NULL) {
    for (int h = 0; h < comm->oneSideHandleCount; h++) {
      struct flagcxOneSideHandleInfo *info = comm->oneSideHandles[h];
      if (info == NULL || info->baseVas == NULL)
        continue;
      uintptr_t myBase = info->baseVas[comm->rank];
      for (int k = 0; k < comm->ipcTableSize; k++) {
        if (comm->ipcTable[k].inUse &&
            (uintptr_t)comm->ipcTable[k].basePtr == myBase) {
          hasD2dMemory = true;
          break;
        }
      }
      if (hasD2dMemory)
        break;
    }
  }

  if (!hasD2dMemory) {
    INFO(FLAGCX_REG, "D2D bypass disabled: no VMM or IPC memory available");
    return flagcxSuccess; // Not an error — proxy path works fine
  }

  int nRanks = comm->nRanks;
  struct flagcxRmaIpcState *ipc =
      (struct flagcxRmaIpcState *)calloc(1, sizeof(*ipc));
  if (ipc == NULL)
    return flagcxSystemError;
  ipc->nRanks = nRanks;
  ipc->dataHandleCount = comm->oneSideHandleCount;

  // Allocate per-peer pointer arrays
  ipc->peerDataBufs = (void ***)calloc(nRanks, sizeof(void **));
  ipc->peerSignalBufs = (void **)calloc(nRanks, sizeof(void *));
  ipc->signalSeqs = (uint64_t *)calloc(nRanks, sizeof(uint64_t));
  if (ipc->peerDataBufs == NULL || ipc->peerSignalBufs == NULL ||
      ipc->signalSeqs == NULL) {
    free(ipc->peerDataBufs);
    free(ipc->peerSignalBufs);
    free(ipc->signalSeqs);
    free(ipc);
    return flagcxSystemError;
  }

  // For each intra-node peer, resolve D2D pointers
  for (int p = 0; p < nRanks; p++) {
    if (p == comm->rank || !flagcxIsIntraNode(comm, p))
      continue;

    int peerLocalRank = comm->rankToLocalRank[p];

    // Map signal buffer via symmetric flat mapping (VMM only)
    if (comm->signalHandle != NULL && comm->signalHandle->symWin != NULL) {
      ipc->peerSignalBufs[p] =
          flagcxGetIntraRankPtr(comm->signalHandle->symWin, peerLocalRank, 0);
    }
    // IPC fallback for signal buffer: lookup in ipcTable by signal basePtr
    if (ipc->peerSignalBufs[p] == NULL && comm->signalHandle != NULL &&
        comm->signalHandle->baseVas != NULL && comm->ipcTable != NULL) {
      uintptr_t sigBase = comm->signalHandle->baseVas[comm->rank];
      for (int k = 0; k < comm->ipcTableSize; k++) {
        if (comm->ipcTable[k].inUse &&
            (uintptr_t)comm->ipcTable[k].basePtr == sigBase &&
            comm->ipcTable[k].hostPeerPtrs != NULL) {
          ipc->peerSignalBufs[p] =
              comm->ipcTable[k].hostPeerPtrs[peerLocalRank];
          break;
        }
      }
    }

    // Map data buffers for each registered handle
    if (comm->oneSideHandleCount > 0) {
      ipc->peerDataBufs[p] =
          (void **)calloc(comm->oneSideHandleCount, sizeof(void *));
      if (ipc->peerDataBufs[p] == NULL)
        continue;
      for (int h = 0; h < comm->oneSideHandleCount; h++) {
        struct flagcxOneSideHandleInfo *info = comm->oneSideHandles[h];
        if (info == NULL)
          continue;

        // VMM path (takes precedence)
        if (info->symWin != NULL && info->symWin->flatBase != NULL &&
            info->symWin->isVMM) {
          ipc->peerDataBufs[p][h] =
              flagcxGetIntraRankPtr(info->symWin, peerLocalRank, 0);
          continue;
        }

        // IPC fallback: lookup in ipcTable by basePtr
        if (comm->ipcTable != NULL && info->baseVas != NULL) {
          uintptr_t myBase = info->baseVas[comm->rank];
          for (int k = 0; k < comm->ipcTableSize; k++) {
            if (comm->ipcTable[k].inUse &&
                (uintptr_t)comm->ipcTable[k].basePtr == myBase &&
                comm->ipcTable[k].hostPeerPtrs != NULL) {
              ipc->peerDataBufs[p][h] =
                  comm->ipcTable[k].hostPeerPtrs[peerLocalRank];
              break;
            }
          }
        }
      }
    }
  }

  proxy->ipcState = ipc;
  INFO(FLAGCX_REG, "D2D bypass enabled for %d intra-node peers",
       comm->localRanks - 1);
  return flagcxSuccess;
}

flagcxResult_t flagcxHeteroRmaIpcDestroy(flagcxHeteroComm_t comm) {
  struct flagcxRmaProxyState *proxy = comm->rmaProxy;
  if (proxy == NULL || proxy->ipcState == NULL)
    return flagcxSuccess;

  struct flagcxRmaIpcState *ipc = proxy->ipcState;
  for (int p = 0; p < ipc->nRanks; p++) {
    if (ipc->peerDataBufs != NULL)
      free(ipc->peerDataBufs[p]);
  }
  free(ipc->peerDataBufs);
  free(ipc->peerSignalBufs);
  free(ipc->signalSeqs);
  free(ipc);
  proxy->ipcState = NULL;
  return flagcxSuccess;
}

// ---- Stream-based Put with intra-node D2D bypass ----

flagcxResult_t flagcxHeteroPutStream(flagcxHeteroComm_t comm, int peer,
                                     size_t srcOffset, size_t dstOffset,
                                     size_t size, int srcMrIdx, int dstMrIdx,
                                     flagcxStream_t stream, uint64_t *opSeq) {
  struct flagcxRmaProxyState *proxy = comm->rmaProxy;
  if (proxy == NULL)
    return flagcxInternalError;

  // Try intra-node D2D path (lazy init if not yet built)
  if (stream != NULL && flagcxIsIntraNode(comm, peer)) {
    if (proxy->ipcState == NULL && !proxy->ipcInitFailed) {
      if (flagcxHeteroRmaIpcInit(comm) != flagcxSuccess)
        proxy->ipcInitFailed = true;
    }
    if (proxy->ipcState != NULL) {
      struct flagcxRmaIpcState *ipc = proxy->ipcState;
      void *srcBuf = NULL;
      void *dstBuf = NULL;

      // Resolve source and destination buffers
      if (srcMrIdx >= 0 && srcMrIdx < comm->oneSideHandleCount &&
          comm->oneSideHandles[srcMrIdx] != NULL) {
        srcBuf = (void *)(comm->oneSideHandles[srcMrIdx]->baseVas[comm->rank] +
                          srcOffset);
      }
      if (dstMrIdx >= 0 && dstMrIdx < ipc->dataHandleCount &&
          ipc->peerDataBufs[peer] != NULL &&
          ipc->peerDataBufs[peer][dstMrIdx] != NULL) {
        dstBuf =
            (void *)((uintptr_t)ipc->peerDataBufs[peer][dstMrIdx] + dstOffset);
      }

      if (srcBuf != NULL && dstBuf != NULL) {
        flagcxResult_t res = deviceAdaptor->deviceMemcpy(
            dstBuf, srcBuf, size, flagcxMemcpyDeviceToDevice, stream, NULL);
        if (opSeq != NULL)
          *opSeq = 0; // D2D path has no opSeq (no proxy involvement)
        return res;
      }
      // Fall through to proxy path if buffer resolution fails
    }
  }

  // Fallback: enqueue to proxy thread (inter-node or no IPC)
  // Stream sync: enqueue (get real opSeq) → signal ready → wait done
  if (stream != NULL && proxy->readySeqsCpu != NULL) {
    // Step 1: Enqueue to proxy (gets real opSeq under mutex, race-free)
    uint64_t assignedSeq = 0;
    flagcxResult_t res = flagcxHeteroPut(
        comm, peer, srcOffset, dstOffset, size, srcMrIdx, dstMrIdx,
        /*streamSyncReady=*/true, &assignedSeq);
    if (res != flagcxSuccess)
      return res;

    // Step 2: Signal proxy that source data is ready (GPU stream → proxy)
    flagcxResult_t syncRes =
        flagcxRmaSignalReady(proxy, peer, assignedSeq, stream);
    if (syncRes != flagcxSuccess)
      return syncRes;

    // Step 3: Wait for proxy completion (proxy → GPU stream)
    syncRes = flagcxRmaWaitDone(proxy, peer, assignedSeq, stream);
    if (syncRes != flagcxSuccess)
      return syncRes;

    if (opSeq != NULL)
      *opSeq = assignedSeq;
    return flagcxSuccess;
  } else {
    // No stream or no GDR buffers: skip sync (legacy non-stream path)
    uint64_t assignedSeq = 0;
    flagcxResult_t res =
        flagcxHeteroPut(comm, peer, srcOffset, dstOffset, size, srcMrIdx,
                        dstMrIdx, false, &assignedSeq);
    if (opSeq != NULL && res == flagcxSuccess)
      *opSeq = assignedSeq;
    return res;
  }
}

flagcxResult_t
flagcxHeteroPutSignalStream(flagcxHeteroComm_t comm, int peer, size_t srcOffset,
                            size_t dstOffset, size_t size, size_t signalOffset,
                            int srcMrIdx, int dstMrIdx, uint64_t signalValue,
                            flagcxStream_t stream, uint64_t *opSeq) {
  struct flagcxRmaProxyState *proxy = comm->rmaProxy;
  if (proxy == NULL)
    return flagcxInternalError;

  // Try intra-node D2D path (lazy init if not yet built)
  if (stream != NULL && flagcxIsIntraNode(comm, peer)) {
    if (proxy->ipcState == NULL && !proxy->ipcInitFailed) {
      if (flagcxHeteroRmaIpcInit(comm) != flagcxSuccess)
        proxy->ipcInitFailed = true;
    }
    if (proxy->ipcState != NULL) {
      struct flagcxRmaIpcState *ipc = proxy->ipcState;
      void *srcBuf = NULL;
      void *dstBuf = NULL;
      void *signalAddr = NULL;

      // Resolve source buffer (local)
      if (srcMrIdx >= 0 && srcMrIdx < comm->oneSideHandleCount &&
          comm->oneSideHandles[srcMrIdx] != NULL) {
        srcBuf = (void *)(comm->oneSideHandles[srcMrIdx]->baseVas[comm->rank] +
                          srcOffset);
      }
      // Resolve destination buffer (peer)
      if (dstMrIdx >= 0 && dstMrIdx < ipc->dataHandleCount &&
          ipc->peerDataBufs[peer] != NULL &&
          ipc->peerDataBufs[peer][dstMrIdx] != NULL) {
        dstBuf =
            (void *)((uintptr_t)ipc->peerDataBufs[peer][dstMrIdx] + dstOffset);
      }
      // Resolve signal address (peer's signal buffer)
      if (ipc->peerSignalBufs[peer] != NULL) {
        signalAddr =
            (void *)((uintptr_t)ipc->peerSignalBufs[peer] + signalOffset);
      }

      if ((size == 0 || (srcBuf != NULL && dstBuf != NULL)) &&
          signalAddr != NULL) {
        flagcxResult_t res = flagcxSuccess;
        // Data transfer (if any)
        if (size > 0) {
          res = deviceAdaptor->deviceMemcpy(
              dstBuf, srcBuf, size, flagcxMemcpyDeviceToDevice, stream, NULL);
          if (res != flagcxSuccess)
            return res;
        }
        // Signal write via D2D: accumulate monotonic counter so that
        // streamWaitValue64(GEQ) on the receiver side works correctly
        // across multiple iterations. Use atomic to prevent races if
        // multiple threads enqueue D2D PutSignal to the same peer.
        uint64_t newSeq = __atomic_add_fetch(&ipc->signalSeqs[peer],
                                             signalValue, __ATOMIC_RELAXED);
        res = deviceAdaptor->streamWriteValue64(stream, signalAddr, newSeq, 0);
        if (opSeq != NULL)
          *opSeq = 0; // D2D path
        return res;
      }
      // Fall through to proxy path
    }
  }

  // Fallback: enqueue to proxy thread
  // Stream sync: enqueue (get real opSeq) → signal ready → wait done
  if (stream != NULL && proxy->readySeqsCpu != NULL) {
    // Step 1: Enqueue to proxy (gets real opSeq under mutex, race-free)
    uint64_t assignedSeq = 0;
    flagcxResult_t res =
        flagcxHeteroPutSignal(comm, peer, srcOffset, dstOffset, size,
                              signalOffset, srcMrIdx, dstMrIdx, signalValue,
                              /*streamSyncReady=*/true, &assignedSeq);
    if (res != flagcxSuccess)
      return res;

    // Step 2: Signal proxy that source data is ready
    flagcxResult_t syncRes =
        flagcxRmaSignalReady(proxy, peer, assignedSeq, stream);
    if (syncRes != flagcxSuccess)
      return syncRes;

    // Step 3: Wait for proxy completion
    syncRes = flagcxRmaWaitDone(proxy, peer, assignedSeq, stream);
    if (syncRes != flagcxSuccess)
      return syncRes;

    if (opSeq != NULL)
      *opSeq = assignedSeq;
    return flagcxSuccess;
  } else {
    // No stream or no GDR buffers: skip sync (legacy non-stream path)
    uint64_t assignedSeq = 0;
    flagcxResult_t res = flagcxHeteroPutSignal(
        comm, peer, srcOffset, dstOffset, size, signalOffset, srcMrIdx,
        dstMrIdx, signalValue, false, &assignedSeq);
    if (opSeq != NULL && res == flagcxSuccess)
      *opSeq = assignedSeq;
    return res;
  }
}
