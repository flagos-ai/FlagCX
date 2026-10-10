#include "flagcx_hetero.h"
#include "adaptor.h"
#include "gdr_visibility.h"
#include "global_comm.h"
#include "group.h"
#include "net.h"
#include "net_transport.h"
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

static inline bool flagcxIsIntraNode(flagcxHeteroComm_t comm, int peer);

static_assert(static_cast<uint32_t>(FLAGCX_RMA_SUBMIT_DATA) ==
                  static_cast<uint32_t>(FLAGCX_NET_SUBMIT_DATA),
              "RMA and transport data flags must match");
static_assert(static_cast<uint32_t>(FLAGCX_RMA_SUBMIT_RELEASE) ==
                  static_cast<uint32_t>(FLAGCX_NET_SUBMIT_RELEASE),
              "RMA and transport release flags must match");
static_assert(static_cast<uint32_t>(FLAGCX_RMA_SUBMIT_INDEPENDENT) ==
                  static_cast<uint32_t>(FLAGCX_NET_SUBMIT_INDEPENDENT),
              "RMA and transport independence flags must match");

FLAGCX_PARAM(RmaQueueSize, "RMA_QUEUE_SIZE", FLAGCX_RMA_QUEUE_SIZE);
FLAGCX_PARAM(RmaBatchMax, "RMA_BATCH_MAX", FLAGCX_RMA_BATCH_MAX);
FLAGCX_PARAM(RmaStreamOps, "RMA_STREAM_OPS",
             0); // 0 = HOST_FUNC (default), 1 = STREAM_OPS

// RMA owns the allocation and transport owns only the embedded completion
// gate. Each descriptor that can outlive another member holds one reference.
struct flagcxRmaReleaseGroup {
  struct flagcxNetReleaseGroup transport;
  volatile uint32_t refs;
};

static struct flagcxRmaReleaseGroup *flagcxRmaReleaseGroupCreate() {
  struct flagcxRmaReleaseGroup *group =
      (struct flagcxRmaReleaseGroup *)calloc(1, sizeof(*group));
  if (group != NULL)
    group->refs = 1; // creator reference
  return group;
}

static void flagcxRmaReleaseGroupRetain(struct flagcxRmaReleaseGroup *group) {
  if (group != NULL)
    __atomic_add_fetch(&group->refs, 1, __ATOMIC_RELAXED);
}

static void flagcxRmaReleaseGroupRelease(struct flagcxRmaReleaseGroup *group) {
  if (group != NULL &&
      __atomic_sub_fetch(&group->refs, 1, __ATOMIC_ACQ_REL) == 0)
    free(group);
}

static struct flagcxNetReleaseGroup *
flagcxRmaReleaseGroupTransport(struct flagcxRmaReleaseGroup *group) {
  return group == NULL ? NULL : &group->transport;
}

static void flagcxRmaDescSetReleaseGroup(struct flagcxRmaDesc *desc,
                                         struct flagcxRmaReleaseGroup *group) {
  if (desc == NULL)
    return;
  desc->releaseGroup = group;
  flagcxRmaReleaseGroupRetain(group);
}

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
    // STREAM_OPS: the host proxy publishes completion in doneSeqsDev. This is
    // a local completion counter, not a remote payload-publishing signal.
    return deviceAdaptor->streamWaitValue64(stream, &proxy->doneSeqsDev[peer],
                                            opSeq,
                                            FLAGCX_STREAM_WAIT_VALUE_DEFAULT);
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

static struct flagcxNetSubmitContext
flagcxRmaSubmitContext(const struct flagcxRmaDesc *desc) {
  struct flagcxNetSubmitContext context = {};
  context.orderingKey = desc->orderingKey;
  // A release observes a group but is not a data member of that group.  Keep
  // it out of the group's pending count or it would wait on itself forever.
  context.groupId =
      (desc->submitFlags & FLAGCX_RMA_SUBMIT_DATA) ? desc->groupId : 0;
  context.generation = desc->generation;
  context.sequence = desc->sequence;
  context.flags = desc->submitFlags;
  return context;
}

static flagcxResult_t flagcxRmaProxyTrackDesc(struct flagcxRmaProxyState *proxy,
                                              int peer,
                                              struct flagcxRmaDesc *desc) {
  if (proxy->completionScoreboards == NULL)
    return flagcxSuccess;
  struct flagcxNetSubmitContext context = flagcxRmaSubmitContext(desc);
  struct flagcxNetReleaseGroup *group =
      (desc->submitFlags & FLAGCX_RMA_SUBMIT_DATA)
          ? flagcxRmaReleaseGroupTransport(desc->releaseGroup)
          : NULL;
  return flagcxNetTrackSubmit(&proxy->completionScoreboards[peer], &context,
                              group);
}

static void flagcxRmaDescDestroy(struct flagcxRmaDesc *desc) {
  if (desc == NULL)
    return;
  if (desc->streamEvent != NULL)
    deviceAdaptor->eventDestroy(desc->streamEvent);
  flagcxRmaReleaseGroupRelease(desc->releaseGroup);
  free(desc);
}

static flagcxResult_t
flagcxRmaProxyCancelDesc(struct flagcxRmaProxyState *proxy, int peer,
                         struct flagcxRmaDesc *desc) {
  if (proxy->completionScoreboards == NULL)
    return flagcxSuccess;
  struct flagcxNetSubmitContext context = flagcxRmaSubmitContext(desc);
  return flagcxNetTrackCancel(&proxy->completionScoreboards[peer], &context);
}

static flagcxResult_t
flagcxRmaProxyPrepareDesc(struct flagcxRmaProxyState *proxy, int peer,
                          struct flagcxRmaDesc *desc) {
  desc->peer = peer;
  desc->next = NULL;
  desc->request = NULL;
  desc->completionResult = flagcxInProgress;
  desc->completionStage = FLAGCX_RMA_COMPLETION_DATA_POSTED;
  desc->getSequence = 0;
  desc->getVisibilityDomain = UINT32_MAX;
  desc->getDataComplete = 0;
  desc->opSeq = __atomic_add_fetch(&proxy->opSeqs[peer], 1, __ATOMIC_RELAXED);
  desc->generation = proxy->generation == 0 ? 1 : proxy->generation;
  desc->sequence = desc->opSeq;
  flagcxResult_t result = flagcxRmaProxyTrackDesc(proxy, peer, desc);
  if (result != flagcxSuccess)
    __atomic_fetch_sub(&proxy->opSeqs[peer], 1, __ATOMIC_RELAXED);
  return result;
}

static flagcxResult_t flagcxRmaProxyEnqueueDesc(
    struct flagcxRmaProxyState *proxy, int peer, struct flagcxRmaDesc *desc,
    bool streamSyncReady = false, uint64_t *assignedSeq = NULL) {
  pthread_mutex_lock(&proxy->peerProducerMutexes[peer]);
  while (true) {
    if (__atomic_load_n(&proxy->quiesced, __ATOMIC_ACQUIRE)) {
      pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
      return flagcxInvalidUsage;
    }
    if (__atomic_load_n(&proxy->pendingError, __ATOMIC_ACQUIRE) ||
        __atomic_load_n(&proxy->rmaError, __ATOMIC_ACQUIRE)) {
      pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
      return flagcxRemoteError;
    }
    if (flagcxRmaProxyCircularBufFull(proxy, peer)) {
      pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
      sched_yield();
      pthread_mutex_lock(&proxy->peerProducerMutexes[peer]);
      continue;
    }

    uint32_t pi = __atomic_load_n(&proxy->pis[peer], __ATOMIC_RELAXED);
    uint32_t idx = pi & proxy->queueMask;
    flagcxResult_t prepareResult = flagcxRmaProxyPrepareDesc(proxy, peer, desc);
    if (prepareResult == flagcxInProgress) {
      pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
      sched_yield();
      pthread_mutex_lock(&proxy->peerProducerMutexes[peer]);
      continue;
    }
    if (prepareResult != flagcxSuccess) {
      pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
      return prepareResult;
    }
    proxy->circularBuffers[(size_t)peer * proxy->queueSize + idx] = desc;
    // For non-stream callers: CPU data is already committed, auto-advance
    // readySeq so the proxy won't wait on it. Stream-path callers signal via
    // stream ops.
    if (!streamSyncReady && proxy->readySeqsCpu != NULL) {
      __atomic_store_n(&proxy->readySeqsCpu[peer], desc->opSeq,
                       __ATOMIC_RELEASE);
    }
    // RELEASE so the progress thread sees desc contents before the pi bump.
    __atomic_store_n(&proxy->pis[peer], pi + 1, __ATOMIC_RELEASE);
    if (assignedSeq != NULL)
      *assignedSeq = desc->opSeq;
    pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
    return flagcxSuccess;
  }
}

static flagcxResult_t
flagcxRmaProxyEnqueueDescBatch(struct flagcxRmaProxyState *proxy, int peer,
                               struct flagcxRmaDesc **descs, size_t count,
                               size_t *enqueued) {
  *enqueued = 0;
  if (count == 0)
    return flagcxSuccess;
  if (count > proxy->queueSize)
    return flagcxInvalidArgument;

  pthread_mutex_lock(&proxy->peerProducerMutexes[peer]);
  while (true) {
    if (__atomic_load_n(&proxy->quiesced, __ATOMIC_ACQUIRE)) {
      pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
      return flagcxInvalidUsage;
    }
    if (__atomic_load_n(&proxy->pendingError, __ATOMIC_ACQUIRE) ||
        __atomic_load_n(&proxy->rmaError, __ATOMIC_ACQUIRE)) {
      pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
      return flagcxRemoteError;
    }

    uint32_t pi = __atomic_load_n(&proxy->pis[peer], __ATOMIC_RELAXED);
    uint32_t ci = __atomic_load_n(&proxy->cis[peer], __ATOMIC_ACQUIRE);
    if (proxy->queueSize - (pi - ci) < count) {
      pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
      sched_yield();
      pthread_mutex_lock(&proxy->peerProducerMutexes[peer]);
      continue;
    }

    // Reserve every scoreboard entry before making any ring slot visible.
    // If an out-of-order completion is holding scoreboard capacity, roll back
    // this unpublished attempt and retry as a whole instead of returning an
    // undocumented accepted prefix to flagcxBatchPut.
    size_t prepared = 0;
    flagcxResult_t prepareResult = flagcxSuccess;
    for (; prepared < count; prepared++) {
      prepareResult = flagcxRmaProxyPrepareDesc(proxy, peer, descs[prepared]);
      if (prepareResult != flagcxSuccess)
        break;
    }
    if (prepared != count) {
      flagcxResult_t rollbackResult = flagcxSuccess;
      while (prepared != 0) {
        prepared--;
        flagcxResult_t cancelResult =
            flagcxRmaProxyCancelDesc(proxy, peer, descs[prepared]);
        if (rollbackResult == flagcxSuccess && cancelResult != flagcxSuccess)
          rollbackResult = cancelResult;
        __atomic_fetch_sub(&proxy->opSeqs[peer], 1, __ATOMIC_RELAXED);
      }
      if (rollbackResult != flagcxSuccess ||
          prepareResult != flagcxInProgress) {
        pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
        return rollbackResult != flagcxSuccess ? rollbackResult : prepareResult;
      }
      pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
      sched_yield();
      pthread_mutex_lock(&proxy->peerProducerMutexes[peer]);
      continue;
    }

    for (size_t i = 0; i < count; i++) {
      uint32_t idx = (pi + (uint32_t)i) & proxy->queueMask;
      proxy->circularBuffers[(size_t)peer * proxy->queueSize + idx] = descs[i];
    }
    // Batch path is always non-stream. A single monotonic ready publication
    // covers every sequence in the atomically published batch.
    if (proxy->readySeqsCpu != NULL) {
      __atomic_store_n(&proxy->readySeqsCpu[peer], descs[count - 1]->opSeq,
                       __ATOMIC_RELEASE);
    }
    __atomic_store_n(&proxy->pis[peer], pi + (uint32_t)count, __ATOMIC_RELEASE);
    *enqueued = count;
    pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
    return flagcxSuccess;
  }
}

// Atomically publishes a data descriptor and its release descriptor.  Group
// membership is tracked and sealed before the producer index becomes visible
// to the progress thread.
static flagcxResult_t flagcxRmaProxyEnqueueReleaseGroup(
    struct flagcxRmaProxyState *proxy, int peer, struct flagcxRmaDesc **data,
    uint32_t dataCount, struct flagcxRmaDesc *release, bool streamSyncReady,
    uint64_t *assignedSeq) {
  if (release == NULL || dataCount >= proxy->queueSize)
    return flagcxInvalidArgument;
  const uint32_t count = dataCount + 1;
  pthread_mutex_lock(&proxy->peerProducerMutexes[peer]);
  while (true) {
    if (__atomic_load_n(&proxy->quiesced, __ATOMIC_ACQUIRE)) {
      pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
      return flagcxInvalidUsage;
    }
    if (__atomic_load_n(&proxy->pendingError, __ATOMIC_ACQUIRE) ||
        __atomic_load_n(&proxy->rmaError, __ATOMIC_ACQUIRE)) {
      pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
      return flagcxRemoteError;
    }
    uint32_t pi = __atomic_load_n(&proxy->pis[peer], __ATOMIC_RELAXED);
    uint32_t ci = __atomic_load_n(&proxy->cis[peer], __ATOMIC_ACQUIRE);
    if (proxy->queueSize - (pi - ci) < count) {
      pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
      sched_yield();
      pthread_mutex_lock(&proxy->peerProducerMutexes[peer]);
      continue;
    }
    uint32_t prepared = 0;
    flagcxResult_t result = flagcxSuccess;
    for (; prepared < dataCount; prepared++) {
      result = flagcxRmaProxyPrepareDesc(proxy, peer, data[prepared]);
      if (result != flagcxSuccess)
        break;
    }
    if (prepared != dataCount) {
      flagcxResult_t rollbackResult = flagcxSuccess;
      while (prepared != 0) {
        prepared--;
        flagcxResult_t cancelResult =
            flagcxRmaProxyCancelDesc(proxy, peer, data[prepared]);
        if (rollbackResult == flagcxSuccess && cancelResult != flagcxSuccess)
          rollbackResult = cancelResult;
        __atomic_fetch_sub(&proxy->opSeqs[peer], 1, __ATOMIC_RELAXED);
      }
      if (rollbackResult == flagcxSuccess && result == flagcxInProgress) {
        pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
        sched_yield();
        pthread_mutex_lock(&proxy->peerProducerMutexes[peer]);
        continue;
      }
      pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
      return rollbackResult != flagcxSuccess ? rollbackResult : result;
    }

    result = flagcxRmaProxyPrepareDesc(proxy, peer, release);
    if (result != flagcxSuccess) {
      flagcxResult_t rollbackResult = flagcxSuccess;
      while (prepared != 0) {
        prepared--;
        flagcxResult_t cancelResult =
            flagcxRmaProxyCancelDesc(proxy, peer, data[prepared]);
        if (rollbackResult == flagcxSuccess && cancelResult != flagcxSuccess)
          rollbackResult = cancelResult;
        __atomic_fetch_sub(&proxy->opSeqs[peer], 1, __ATOMIC_RELAXED);
      }
      if (rollbackResult == flagcxSuccess && result == flagcxInProgress) {
        pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
        sched_yield();
        pthread_mutex_lock(&proxy->peerProducerMutexes[peer]);
        continue;
      }
      pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
      return rollbackResult != flagcxSuccess ? rollbackResult : result;
    }

    result = flagcxNetReleaseGroupSeal(
        flagcxRmaReleaseGroupTransport(release->releaseGroup));
    if (result != flagcxSuccess) {
      flagcxRmaProxyCancelDesc(proxy, peer, release);
      __atomic_fetch_sub(&proxy->opSeqs[peer], 1, __ATOMIC_RELAXED);
      while (prepared != 0) {
        prepared--;
        flagcxRmaProxyCancelDesc(proxy, peer, data[prepared]);
        __atomic_fetch_sub(&proxy->opSeqs[peer], 1, __ATOMIC_RELAXED);
      }
      pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
      return result;
    }

    for (uint32_t i = 0; i < dataCount; i++) {
      proxy->circularBuffers[(size_t)peer * proxy->queueSize +
                             (pi & proxy->queueMask)] = data[i];
      pi++;
    }
    proxy->circularBuffers[(size_t)peer * proxy->queueSize +
                           (pi & proxy->queueMask)] = release;
    pi++;
    if (!streamSyncReady && proxy->readySeqsCpu != NULL)
      __atomic_store_n(&proxy->readySeqsCpu[peer], release->opSeq,
                       __ATOMIC_RELEASE);
    __atomic_store_n(&proxy->pis[peer], pi, __ATOMIC_RELEASE);
    if (assignedSeq != NULL)
      *assignedSeq = release->opSeq;
    pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
    return flagcxSuccess;
  }
}

// Post a single desc via the net adaptor. desc->request is populated on
// success. Returns the adaptor's result.
class flagcxRmaSubmitScope {
public:
  explicit flagcxRmaSubmitScope(const struct flagcxRmaDesc *desc) {
    struct flagcxNetSubmitContext context = flagcxRmaSubmitContext(desc);
    active_ = flagcxNetSetSubmitContext(&context) == flagcxSuccess;
  }
  ~flagcxRmaSubmitScope() {
    if (active_)
      flagcxNetClearSubmitContext();
  }

private:
  bool active_ = false;
};

static flagcxResult_t flagcxRmaProxyPostOp(struct flagcxHeteroComm *comm,
                                           struct flagcxRmaDesc *desc,
                                           void *sendComm) {
  flagcxRmaSubmitScope submitScope(desc);
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
    case FLAGCX_RMA_RELEASE: {
      void **sigHandles = (void **)comm->signalHandle;
      return comm->netAdaptor->iputSignal(
          sendComm, 0, 0, 0, comm->rank, p, NULL, NULL, desc->signalOff,
          sigHandles, desc->signalValue, &desc->request);
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

  flagcxRmaSubmitScope submitScope(descs[0]);

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

static void
flagcxRmaProxyRecordPendingError(struct flagcxRmaProxyState *proxy) {
  __atomic_store_n(&proxy->pendingError, 1, __ATOMIC_RELEASE);
}

static void flagcxRmaProxyPublishError(struct flagcxRmaProxyState *proxy) {
  flagcxRmaProxyRecordPendingError(proxy);
  int previous = __atomic_exchange_n(&proxy->rmaError, 1, __ATOMIC_ACQ_REL);
  if (previous == 0) {
    pthread_mutex_lock(&proxy->doneMutex);
    pthread_cond_broadcast(&proxy->doneCond);
    pthread_mutex_unlock(&proxy->doneMutex);
  }
}

static bool
flagcxRmaProxyAllNativeRequestsRetired(struct flagcxRmaProxyState *proxy) {
  for (int peer = 0; peer < proxy->nRanks; peer++) {
    if (__atomic_load_n(&proxy->inFlights[peer], __ATOMIC_ACQUIRE) != 0)
      return false;
  }
  return true;
}

static bool flagcxRmaProxyDrainRing(struct flagcxRmaProxyState *proxy,
                                    int peer);

static void flagcxRmaProxyPublishScoreboard(struct flagcxRmaProxyState *proxy,
                                            int peer, uint32_t completed) {
  if (proxy->completionScoreboards == NULL)
    return;
  uint64_t nextSequence = 0;
  uint32_t inFlight = 0;
  flagcxResult_t firstError = flagcxSuccess;
  if (flagcxNetCompletionScoreboardQuery(&proxy->completionScoreboards[peer],
                                         &nextSequence, &inFlight,
                                         &firstError) != flagcxSuccess)
    return;
  // This is a retired prefix, not a successful prefix. Stream waiters must be
  // released after every contiguous descriptor has retired even when one of
  // them failed; rmaError reports that failure independently.
  if (nextSequence > 0) {
    const uint64_t doneSequence = nextSequence - 1;
    const uint64_t current =
        __atomic_load_n(&proxy->doneSeqs[peer], __ATOMIC_RELAXED);
    if (doneSequence > current)
      __atomic_store_n(&proxy->doneSeqs[peer], doneSequence, __ATOMIC_RELEASE);
    if (proxy->doneSeqsCpu != NULL &&
        doneSequence >
            __atomic_load_n(&proxy->doneSeqsCpu[peer], __ATOMIC_RELAXED))
      __atomic_store_n(&proxy->doneSeqsCpu[peer], doneSequence,
                       __ATOMIC_RELEASE);
  }
  if (completed != 0)
    __atomic_fetch_add(&proxy->completionCount, (uint64_t)completed,
                       __ATOMIC_RELEASE);
  if (firstError != flagcxSuccess)
    flagcxRmaProxyRecordPendingError(proxy);
}

static void flagcxRmaProxyCompleteDesc(struct flagcxRmaProxyState *proxy,
                                       int peer, struct flagcxRmaDesc *desc,
                                       flagcxResult_t result) {
  // A data-bearing PutSignal is represented by one or more data descriptors
  // followed by a release descriptor.  The public completion counter counts
  // the API operation, not its internal transport submissions, so only the
  // release retires that operation.  Standalone data descriptors still count
  // normally.
  const bool completesApiOperation =
      desc->releaseGroup == NULL ||
      (desc->submitFlags & FLAGCX_RMA_SUBMIT_RELEASE) != 0;
  uint32_t advanced = 0;
  if (proxy->completionScoreboards != NULL) {
    struct flagcxNetSubmitContext context = flagcxRmaSubmitContext(desc);
    flagcxResult_t trackResult = flagcxNetTrackCompletion(
        &proxy->completionScoreboards[peer], &context, result, &advanced);
    if (trackResult != flagcxSuccess) {
      WARN("flagcxRmaProxyCompleteDesc: invalid completion peer=%d seq=%lu "
           "res=%d",
           peer, (unsigned long)desc->sequence, (int)trackResult);
      flagcxRmaProxyRecordPendingError(proxy);
    }
    flagcxRmaProxyPublishScoreboard(
        proxy, peer,
        result == flagcxSuccess && completesApiOperation ? 1u : 0u);
  } else if (result == flagcxSuccess) {
    __atomic_store_n(&proxy->doneSeqs[peer], desc->opSeq, __ATOMIC_RELEASE);
    if (proxy->doneSeqsCpu != NULL)
      __atomic_store_n(&proxy->doneSeqsCpu[peer], desc->opSeq,
                       __ATOMIC_RELEASE);
    if (completesApiOperation)
      __atomic_fetch_add(&proxy->completionCount, 1ULL, __ATOMIC_RELEASE);
  }
  if (result != flagcxSuccess) {
    if (proxy->completionScoreboards != NULL)
      flagcxRmaProxyRecordPendingError(proxy);
    else
      flagcxRmaProxyPublishError(proxy);
  }
}

// Poll every native request, not just the queue head. Native CQEs may arrive
// out of order; the scoreboard publishes only the contiguous completed prefix.
bool flagcxOneSideGetCompletionRequiresFlush(
    const struct flagcxHeteroComm *comm, int dstMrIdx) {
  if (comm == NULL || dstMrIdx < 0 || dstMrIdx >= comm->oneSideHandleCount ||
      comm->oneSideHandles == NULL)
    return false;
  const struct flagcxOneSideHandleInfo *info = comm->oneSideHandles[dstMrIdx];
  return info != NULL &&
         (info->gdrFlushRequirements & FLAGCX_GDR_READ_REQUIRES_FLUSH) != 0;
}

static int flagcxRmaVisibilityFlushSize(size_t size) {
  return size > static_cast<size_t>(INT_MAX) ? INT_MAX : static_cast<int>(size);
}

flagcxResult_t
flagcxOneSidePostGetVisibilityFlush(struct flagcxHeteroComm *comm, int dstMrIdx,
                                    uint64_t dstOff, size_t size,
                                    void *recvComm, void **request) {
  if (request == NULL)
    return flagcxInvalidArgument;
  *request = NULL;
  if (comm == NULL || dstMrIdx < 0 || dstMrIdx >= comm->oneSideHandleCount ||
      comm->oneSideHandles == NULL)
    return flagcxInternalError;
  struct flagcxOneSideHandleInfo *info = comm->oneSideHandles[dstMrIdx];
  if (info == NULL || info->baseVas == NULL || info->regionSizes == NULL ||
      comm->rank < 0 || comm->rank >= info->nRanks)
    return flagcxNotSupported;
  if (dstOff > info->regionSizes[comm->rank] ||
      size > info->regionSizes[comm->rank] - dstOff)
    return flagcxInvalidArgument;
  // A zero-byte GET is complete once its range has been validated. It needs no
  // provider capability, MR handle, or visibility operation.
  if (size == 0)
    return flagcxSuccess;
  if (info->localMrHandle == NULL || comm->netAdaptor == NULL ||
      comm->netAdaptor->iflush == NULL)
    return flagcxNotSupported;
  // A compatibility callback such as BAREX's no-op is not a visibility
  // capability. Forced READ requirements must fail rather than silently pass
  // through such a callback.
  FLAGCXCHECK(flagcxValidateGdrFlushCapability(info->gdrFlushRequirements,
                                               comm->netAdaptor->gdrFlushCaps,
                                               FLAGCX_GDR_READ_REQUIRES_FLUSH));
  if (recvComm == NULL)
    recvComm = info->localRecvComm;
  if (recvComm == NULL)
    return flagcxNotSupported;

  void *data[1] = {(void *)(info->baseVas[comm->rank] + (uintptr_t)dstOff)};
  // The legacy iflush ABI uses int sizes only to identify non-empty ranges;
  // providers issue their own fixed-size visibility operation. Saturate the
  // transferred size instead of rejecting a valid UINT32-sized RMA request.
  int sizes[1] = {flagcxRmaVisibilityFlushSize(size)};
  void *mhandles[1] = {info->localMrHandle};
  return comm->netAdaptor->iflush(recvComm, 1, data, sizes, mhandles, request);
}

static struct flagcxNetGetVisibilityDomain *
flagcxRmaProxyGetVisibilityDomain(struct flagcxRmaProxyState *proxy, int peer,
                                  uint64_t orderingKey, uint32_t *domainIndex) {
  if (proxy->getVisibilityDomains == NULL ||
      proxy->activeGetVisibilityDomains == NULL ||
      proxy->activeGetVisibilityCounts == NULL || peer < 0 ||
      peer >= proxy->nRanks || domainIndex == NULL)
    return NULL;
  const size_t base = (size_t)peer * proxy->queueSize;
  uint32_t activeCount = proxy->activeGetVisibilityCounts[peer];
  for (uint32_t i = 0; i < activeCount; ++i) {
    const uint32_t index = proxy->activeGetVisibilityDomains[base + i];
    struct flagcxNetGetVisibilityDomain *domain =
        &proxy->getVisibilityDomains[index];
    if (domain->peer == peer && domain->orderingKey == orderingKey) {
      *domainIndex = index;
      return domain;
    }
  }
  if (activeCount >= proxy->queueSize)
    return NULL;

  uint32_t freeIndex = UINT32_MAX;
  for (uint32_t i = 0; i < proxy->queueSize; ++i) {
    const uint32_t index = (uint32_t)(base + i);
    struct flagcxNetGetVisibilityDomain *domain =
        &proxy->getVisibilityDomains[base + i];
    if (domain->inUse == 0) {
      freeIndex = index;
      break;
    }
  }
  if (freeIndex == UINT32_MAX ||
      flagcxNetGetVisibilityDomainInit(&proxy->getVisibilityDomains[freeIndex],
                                       peer, orderingKey) != flagcxSuccess)
    return NULL;
  proxy->activeGetVisibilityDomains[base + activeCount] = freeIndex;
  proxy->activeGetVisibilityCounts[peer] = activeCount + 1;
  *domainIndex = freeIndex;
  return &proxy->getVisibilityDomains[freeIndex];
}

static void
flagcxRmaProxyPruneGetVisibilityDomains(struct flagcxRmaProxyState *proxy,
                                        int peer) {
  if (proxy->activeGetVisibilityDomains == NULL ||
      proxy->activeGetVisibilityCounts == NULL)
    return;
  const size_t base = (size_t)peer * proxy->queueSize;
  uint32_t write = 0;
  const uint32_t count = proxy->activeGetVisibilityCounts[peer];
  for (uint32_t read = 0; read < count; ++read) {
    const uint32_t index = proxy->activeGetVisibilityDomains[base + read];
    struct flagcxNetGetVisibilityDomain *domain =
        &proxy->getVisibilityDomains[index];
    bool referenced = false;
    struct flagcxRmaDesc *desc =
        flagcxIntruQueueHead(&proxy->inProgressQueues[peer]);
    while (desc != NULL) {
      if (desc->getVisibilityDomain == index) {
        referenced = true;
        break;
      }
      desc = desc->next;
    }
    if (!referenced && domain->flushRequest == NULL &&
        domain->issuedGetSequence == domain->visibleGetSequence) {
      memset(domain, 0, sizeof(*domain));
      domain->peer = -1;
      continue;
    }
    proxy->activeGetVisibilityDomains[base + write++] = index;
  }
  proxy->activeGetVisibilityCounts[peer] = write;
}

static flagcxResult_t
flagcxRmaProxyTrackGetVisibility(struct flagcxRmaProxyState *proxy, int peer,
                                 struct flagcxRmaDesc *desc) {
  if (desc->type != FLAGCX_RMA_GET || desc->size == 0 ||
      !flagcxOneSideGetCompletionRequiresFlush(proxy->comm, desc->dstMrIdx))
    return flagcxSuccess;
  uint32_t domainIndex = UINT32_MAX;
  struct flagcxNetGetVisibilityDomain *domain =
      flagcxRmaProxyGetVisibilityDomain(proxy, peer, desc->orderingKey,
                                        &domainIndex);
  if (domain == NULL)
    return flagcxInternalError;
  FLAGCXCHECK(flagcxNetGetVisibilityIssue(domain, &desc->getSequence));
  desc->getVisibilityDomain = domainIndex;
  return flagcxSuccess;
}

static struct flagcxRmaDesc *
flagcxRmaProxyFindGet(struct flagcxRmaProxyState *proxy, int peer,
                      uint32_t domainIndex, uint64_t getSequence) {
  struct flagcxRmaDesc *desc =
      flagcxIntruQueueHead(&proxy->inProgressQueues[peer]);
  while (desc != NULL) {
    if (desc->getVisibilityDomain == domainIndex &&
        desc->getSequence == getSequence)
      return desc;
    desc = desc->next;
  }
  return NULL;
}

static bool
flagcxRmaProxyProgressGetVisibilityDomain(struct flagcxRmaProxyState *proxy,
                                          int peer, uint32_t domainIndex) {
  struct flagcxNetGetVisibilityDomain *domain =
      &proxy->getVisibilityDomains[domainIndex];
  if (domain->inUse == 0 || domain->peer != peer)
    return false;
  bool did = false;

  if (domain->flushRequest != NULL) {
    int done = 0;
    flagcxResult_t result =
        proxy->comm->netAdaptor->test(domain->flushRequest, &done, NULL);
    if (result != flagcxSuccess) {
      done = 1;
      WARN("flagcxRmaProxyProgressGetVisibilityDomain: flush test failed "
           "peer=%d key=%lu res=%d",
           peer, (unsigned long)domain->orderingKey, (int)result);
    }
    if (done) {
      const uint64_t previousVisible = domain->visibleGetSequence;
      flagcxResult_t completeResult =
          flagcxNetGetVisibilityCompleteFlush(domain, result);
      if (completeResult != flagcxSuccess) {
        result = completeResult;
        flagcxRmaProxyRecordPendingError(proxy);
      }
      if (result != flagcxSuccess) {
        for (uint64_t sequence = previousVisible + 1;
             sequence <= domain->visibleGetSequence; ++sequence) {
          struct flagcxRmaDesc *covered =
              flagcxRmaProxyFindGet(proxy, peer, domainIndex, sequence);
          if (covered != NULL && covered->completionResult == flagcxSuccess)
            covered->completionResult = result;
        }
        flagcxRmaProxyRecordPendingError(proxy);
      }
      did = true;
    }
  }

  uint64_t completed = domain->dataCompletedGetSequence;
  while (completed < domain->issuedGetSequence) {
    struct flagcxRmaDesc *next =
        flagcxRmaProxyFindGet(proxy, peer, domainIndex, completed + 1);
    if (next == NULL || next->getDataComplete == 0)
      break;
    ++completed;
  }
  if (completed != domain->dataCompletedGetSequence) {
    if (flagcxNetGetVisibilityAdvanceData(domain, completed) != flagcxSuccess)
      flagcxRmaProxyRecordPendingError(proxy);
    did = true;
  }

  if (domain->flushRequest != NULL ||
      domain->dataCompletedGetSequence <= domain->visibleGetSequence)
    return did;

  const bool retryFlush =
      domain->flushTargetGetSequence > domain->visibleGetSequence;
  const uint64_t targetLimit = retryFlush ? domain->flushTargetGetSequence
                                          : domain->dataCompletedGetSequence;
  struct flagcxRmaDesc *flushRange = NULL;
  for (uint64_t sequence = domain->visibleGetSequence + 1;
       sequence <= targetLimit; ++sequence) {
    struct flagcxRmaDesc *candidate =
        flagcxRmaProxyFindGet(proxy, peer, domainIndex, sequence);
    if (candidate != NULL && candidate->completionResult == flagcxSuccess)
      flushRange = candidate;
  }
  if (flushRange == NULL) {
    if (flagcxNetGetVisibilityAdvanceVisible(
            domain, domain->dataCompletedGetSequence) != flagcxSuccess)
      flagcxRmaProxyRecordPendingError(proxy);
    return true;
  }

  uint64_t flushTarget = domain->flushTargetGetSequence;
  if (!retryFlush &&
      flagcxNetGetVisibilityBeginFlush(domain, &flushTarget) != flagcxSuccess) {
    flagcxRmaProxyRecordPendingError(proxy);
    return true;
  }
  flagcxResult_t result = flagcxOneSidePostGetVisibilityFlush(
      proxy->comm, flushRange->dstMrIdx, flushRange->dstOff, flushRange->size,
      NULL, &domain->flushRequest);
  if (flagcxRmaPostResultIsRetryable(result)) {
    // Keep the immutable target snapshot and retry posting on the next pass.
    domain->flushRequest = NULL;
    return did;
  }
  did = true;
  if (result == flagcxSuccess && domain->flushRequest != NULL)
    return did;

  const uint64_t previousVisible = domain->visibleGetSequence;
  flagcxResult_t completeResult =
      flagcxNetGetVisibilityCompleteFlush(domain, result);
  if (completeResult != flagcxSuccess)
    result = completeResult;
  if (result != flagcxSuccess) {
    WARN("flagcxRmaProxyProgressGetVisibilityDomain: GET flush failed "
         "peer=%d key=%lu target=%lu res=%d",
         peer, (unsigned long)domain->orderingKey, (unsigned long)flushTarget,
         (int)result);
    for (uint64_t sequence = previousVisible + 1;
         sequence <= domain->visibleGetSequence; ++sequence) {
      struct flagcxRmaDesc *covered =
          flagcxRmaProxyFindGet(proxy, peer, domainIndex, sequence);
      if (covered != NULL && covered->completionResult == flagcxSuccess)
        covered->completionResult = result;
    }
    flagcxRmaProxyRecordPendingError(proxy);
  }
  return did;
}

static bool
flagcxRmaProxyPollNonPersistCompletion(struct flagcxRmaProxyState *proxy,
                                       int peer) {
  struct flagcxHeteroComm *comm = proxy->comm;
  bool did = false;
  struct flagcxRmaDesc *desc =
      flagcxIntruQueueHead(&proxy->inProgressQueues[peer]);
  while (desc != NULL) {
    struct flagcxRmaDesc *next = desc->next;
    int done = 0;
    flagcxResult_t completionResult = desc->completionResult;
    if (desc->completionStage == FLAGCX_RMA_COMPLETION_FLUSH_PENDING) {
      desc = next;
      continue;
    }
    if (desc->request != NULL) {
      flagcxResult_t res = comm->netAdaptor->test(desc->request, &done, NULL);
      if (res != flagcxSuccess) {
        WARN("flagcxRmaProxyPollNonPersistCompletion: test failed peer=%d "
             "res=%d",
             peer, (int)res);
        done = 1;
        completionResult = res;
      }
    } else {
      // A successful post without a native request is an immediate
      // completion. Permanent post errors are carried in completionResult.
      done = 1;
    }
    if (!done) {
      desc = next;
      continue;
    }
    if (desc->getSequence != 0) {
      // Keep the descriptor until its ordering domain's contiguous completed
      // GET prefix has been made visible by one shared flush.
      desc->request = NULL;
      desc->completionResult = completionResult;
      desc->getDataComplete = 1;
      desc->completionStage = FLAGCX_RMA_COMPLETION_FLUSH_PENDING;
      if (completionResult != flagcxSuccess)
        flagcxRmaProxyRecordPendingError(proxy);
      did = true;
      desc = next;
      continue;
    }
    flagcxIntruQueueDelete(&proxy->inProgressQueues[peer], desc);
    __atomic_fetch_sub(&proxy->inFlights[peer], 1, __ATOMIC_RELAXED);
    flagcxRmaProxyCompleteDesc(proxy, peer, desc, completionResult);
    if (!proxy->useStreamOps) {
      pthread_mutex_lock(&proxy->doneMutex);
      pthread_cond_broadcast(&proxy->doneCond);
      pthread_mutex_unlock(&proxy->doneMutex);
    }
    flagcxRmaDescDestroy(desc);
    did = true;
    desc = next;
  }

  if (proxy->activeGetVisibilityDomains != NULL &&
      proxy->activeGetVisibilityCounts != NULL) {
    const size_t base = (size_t)peer * proxy->queueSize;
    const uint32_t activeCount = proxy->activeGetVisibilityCounts[peer];
    for (uint32_t i = 0; i < activeCount; ++i) {
      if (flagcxRmaProxyProgressGetVisibilityDomain(
              proxy, peer, proxy->activeGetVisibilityDomains[base + i]))
        did = true;
    }
  }

  desc = flagcxIntruQueueHead(&proxy->inProgressQueues[peer]);
  while (desc != NULL) {
    struct flagcxRmaDesc *next = desc->next;
    if (desc->completionStage != FLAGCX_RMA_COMPLETION_FLUSH_PENDING ||
        desc->getVisibilityDomain == UINT32_MAX) {
      desc = next;
      continue;
    }
    struct flagcxNetGetVisibilityDomain *domain =
        &proxy->getVisibilityDomains[desc->getVisibilityDomain];
    if (desc->getSequence > domain->visibleGetSequence) {
      desc = next;
      continue;
    }
    flagcxIntruQueueDelete(&proxy->inProgressQueues[peer], desc);
    __atomic_fetch_sub(&proxy->inFlights[peer], 1, __ATOMIC_RELAXED);
    flagcxRmaProxyCompleteDesc(proxy, peer, desc, desc->completionResult);
    if (!proxy->useStreamOps) {
      pthread_mutex_lock(&proxy->doneMutex);
      pthread_cond_broadcast(&proxy->doneCond);
      pthread_mutex_unlock(&proxy->doneMutex);
    }
    flagcxRmaDescDestroy(desc);
    did = true;
    desc = next;
  }
  flagcxRmaProxyPruneGetVisibilityDomains(proxy, peer);
  return did;
}

// Poll pending descs from the ring and issue them. On success advance
// cis[peer] and move the desc to inProgressQueues[peer]. On
// transient adaptor backpressure (flagcxInProgress) leaves cis untouched and
// retries next round. Permanent errors poison the proxy; already-posted work is
// still polled while unsubmitted descriptors are drained without being marked
// successful.
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

    if (desc->type == FLAGCX_RMA_RELEASE) {
      int groupDone = 0;
      int releaseAllowed = 0;
      flagcxResult_t groupResult = flagcxNetReleaseGroupTest(
          flagcxRmaReleaseGroupTransport(desc->releaseGroup), &groupDone,
          &releaseAllowed);
      if (!groupDone)
        break;
      if (groupResult != flagcxSuccess || !releaseAllowed) {
        desc->completionResult =
            groupResult == flagcxSuccess ? flagcxInternalError : groupResult;
        __atomic_store_n(&proxy->cis[peer], ci + 1, __ATOMIC_RELEASE);
        desc->next = NULL;
        flagcxIntruQueueEnqueue(&proxy->inProgressQueues[peer], desc);
        __atomic_fetch_add(&proxy->inFlights[peer], 1, __ATOMIC_RELAXED);
        did = true;
        continue;
      }
    }

    // A network Get event marks its input stream as ready. An IPC Get event
    // marks the D2D copy as complete. The IPC descriptor is a per-peer barrier
    // between preceding and following network submissions.
    if (desc->streamEvent != NULL) {
      if (!__atomic_load_n(&desc->eventArmed, __ATOMIC_ACQUIRE))
        break;
      flagcxResult_t eventResult =
          desc->ipcStatus == flagcxSuccess
              ? deviceAdaptor->eventQuery(desc->streamEvent)
              : desc->ipcStatus;
      if (eventResult == flagcxInProgress)
        break;
      if (desc->type == FLAGCX_RMA_IPC_GET) {
        if (__atomic_load_n(&proxy->doneSeqs[peer], __ATOMIC_ACQUIRE) <
            desc->opSeq - 1)
          break;
        __atomic_store_n(&proxy->cis[peer], ci + 1, __ATOMIC_RELEASE);
        flagcxRmaProxyCompleteDesc(proxy, peer, desc, eventResult);
        if (!proxy->useStreamOps) {
          pthread_mutex_lock(&proxy->doneMutex);
          pthread_cond_broadcast(&proxy->doneCond);
          pthread_mutex_unlock(&proxy->doneMutex);
        }
        flagcxRmaDescDestroy(desc);
        did = true;
        if (eventResult != flagcxSuccess)
          break;
        continue;
      }
      if (eventResult != flagcxSuccess) {
        __atomic_store_n(&proxy->cis[peer], ci + 1, __ATOMIC_RELEASE);
        flagcxRmaProxyCompleteDesc(proxy, peer, desc, eventResult);
        flagcxRmaDescDestroy(desc);
        did = true;
        break;
      }
    }

    // Poll readySeq: wait for GPU stream to signal source data is committed.
    // Both STREAM_OPS (streamWriteValue64) and HOST_FUNC (callback) write here.
    uint64_t readySeq = UINT64_MAX;
    if (proxy->readySeqsCpu != NULL && desc->streamEvent == NULL) {
      readySeq = __atomic_load_n(&proxy->readySeqsCpu[peer], __ATOMIC_ACQUIRE);
      if (readySeq < desc->opSeq) {
        // GPU hasn't signaled ready yet; skip this peer for now
        break;
      }
    }

    // Batch submission is an adaptor capability, not an IB-specific property.
    // This also lets BAREX use its native WriteBatch implementation.
    if (sendComm == NULL)
      break;
    bool canBatch = desc->type == FLAGCX_RMA_PUT && comm->netAdaptor != NULL &&
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
        // Each stream-ordered descriptor publishes readiness independently.
        // Do not let a ready head pull a later, not-yet-produced source buffer
        // into the same native adaptor batch.
        if (cur->opSeq > readySeq)
          break;
        if (cur->type != FLAGCX_RMA_PUT || cur->srcMrIdx != desc->srcMrIdx ||
            cur->dstMrIdx != desc->dstMrIdx ||
            cur->orderingKey != desc->orderingKey ||
            cur->submitFlags != desc->submitFlags ||
            cur->groupId != desc->groupId) {
          break;
        }
        descs[batchCount] = cur;
      }

      if (batchCount > 1) {
        void *requests[FLAGCX_RMA_BATCH_MAX_LIMIT];
        int posted = 0;
        flagcxResult_t res = flagcxRmaProxyPostPutBatch(
            comm, descs, batchCount, sendComm, requests, &posted);
        if (posted < 0 || posted > batchCount) {
          WARN("flagcxRmaProxyPollNonPersistDesc: adaptor returned invalid "
               "posted count peer=%d posted=%d count=%d",
               peer, posted, batchCount);
          posted = 0;
          res = flagcxInternalError;
        }
        bool fatal = flagcxRmaBatchPostResultIsFatal(res, posted, batchCount);
        if (posted == 0) {
          if (fatal) {
            WARN("flagcxRmaProxyPollNonPersistDesc: batch op failed peer=%d "
                 "res=%d",
                 peer, (int)res);
            flagcxRmaProxyRecordPendingError(proxy);
            did = true;
          }
          break;
        }

        for (int i = 0; i < posted; i++) {
          descs[i]->request = requests[i];
          descs[i]->completionResult = flagcxSuccess;
          descs[i]->next = NULL;
          flagcxIntruQueueEnqueue(&proxy->inProgressQueues[peer], descs[i]);
        }
        __atomic_store_n(&proxy->cis[peer], ci + (uint32_t)posted,
                         __ATOMIC_RELEASE);
        __atomic_fetch_add(&proxy->inFlights[peer], (uint32_t)posted,
                           __ATOMIC_RELAXED);
        did = true;
        if (fatal) {
          WARN("flagcxRmaProxyPollNonPersistDesc: partial batch op failed "
               "peer=%d posted=%d count=%d res=%d",
               peer, posted, batchCount, (int)res);
          flagcxRmaProxyRecordPendingError(proxy);
          break;
        }
        continue;
      }
    }

    desc->request = NULL;
    flagcxResult_t res = flagcxRmaProxyPostOp(comm, desc, sendComm);
    if (flagcxRmaPostResultIsRetryable(res)) {
      // Transient backpressure; retry this slot next round (cis unchanged).
      break;
    }
    bool fatal = res != flagcxSuccess;
    desc->completionResult = res;
    if (!fatal) {
      flagcxResult_t visibilityResult =
          flagcxRmaProxyTrackGetVisibility(proxy, peer, desc);
      if (visibilityResult != flagcxSuccess) {
        WARN("flagcxRmaProxyPollNonPersistDesc: failed to track GET "
             "visibility peer=%d res=%d",
             peer, (int)visibilityResult);
        desc->completionResult = visibilityResult;
        flagcxRmaProxyRecordPendingError(proxy);
      }
    }
    if (fatal) {
      WARN("flagcxRmaProxyPollNonPersistDesc: op failed peer=%d type=%d "
           "res=%d",
           peer, (int)desc->type, (int)res);
      flagcxRmaProxyRecordPendingError(proxy);
      desc->request = NULL;
    }
    // RELEASE so the producer sees the slot freed.
    __atomic_store_n(&proxy->cis[peer], ci + 1, __ATOMIC_RELEASE);
    // Enqueue to inProgressQueues[peer] (progress-thread private).
    desc->next = NULL;
    flagcxIntruQueueEnqueue(&proxy->inProgressQueues[peer], desc);
    __atomic_fetch_add(&proxy->inFlights[peer], 1, __ATOMIC_RELAXED);
    did = true;
    if (fatal)
      break;
  }
  return did;
}

// Drain the peer's ring without posting. Every descriptor was registered with
// the scoreboard at enqueue time, so complete it with an error before freeing
// it to avoid leaving a sequence/group permanently pending.
static bool flagcxRmaProxyDrainRing(struct flagcxRmaProxyState *proxy,
                                    int peer) {
  bool drained = false;
  while (!flagcxRmaProxyCircularBufEmpty(proxy, peer)) {
    uint32_t ci = __atomic_load_n(&proxy->cis[peer], __ATOMIC_RELAXED);
    uint32_t idx = ci & proxy->queueMask;
    struct flagcxRmaDesc *desc =
        proxy->circularBuffers[(size_t)peer * proxy->queueSize + idx];
    __atomic_store_n(&proxy->cis[peer], ci + 1, __ATOMIC_RELEASE);
    flagcxRmaProxyCompleteDesc(proxy, peer, desc, flagcxRemoteError);
    flagcxRmaDescDestroy(desc);
    drained = true;
  }
  return drained;
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

    // A permanent transport error poisons the proxy globally. Prevent any
    // additional submissions, but continue polling already posted requests so
    // their CQ resources are reclaimed. The producer mutex closes the race
    // with a caller that was publishing a descriptor when the error occurred.
    if (__atomic_load_n(&proxy->pendingError, __ATOMIC_ACQUIRE) ||
        __atomic_load_n(&proxy->rmaError, __ATOMIC_ACQUIRE)) {
      pthread_mutex_lock(&proxy->peerProducerMutexes[p]);
      if (flagcxRmaProxyDrainRing(proxy, p))
        did = true;
      pthread_mutex_unlock(&proxy->peerProducerMutexes[p]);
      if (!flagcxIntruQueueEmpty(&proxy->inProgressQueues[p]))
        *anyOutstanding = true;
      continue;
    }

    void *sendComm = (fullSendComms != NULL) ? fullSendComms[p] : NULL;
    const uint32_t ci = __atomic_load_n(&proxy->cis[p], __ATOMIC_RELAXED);
    const bool ipcHead = !flagcxRmaProxyCircularBufEmpty(proxy, p) &&
                         proxy->circularBuffers[(size_t)p * proxy->queueSize +
                                                (ci & proxy->queueMask)]
                                 ->type == FLAGCX_RMA_IPC_GET;
    if (sendComm != NULL || ipcHead) {
      if (flagcxRmaProxyPollNonPersistDesc(proxy, p, sendComm))
        did = true;
    } else if (!flagcxRmaProxyCircularBufEmpty(proxy, p)) {
      if (stopping) {
        // Shutdown with queued-but-unissued descs and no transport.
        // Drain to let the thread exit; flag the error so waiters fail.
        WARN("flagcxRmaProxyProgress: stop with queued descs but no "
             "sendComm peer=%d; draining",
             p);
        flagcxRmaProxyRecordPendingError(proxy);
        pthread_mutex_lock(&proxy->peerProducerMutexes[p]);
        flagcxRmaProxyDrainRing(proxy, p);
        pthread_mutex_unlock(&proxy->peerProducerMutexes[p]);
        did = true;
      } else {
        // Pre-registration: caller enqueued an op before the full mesh
        // is ready. Surface as an error rather than spin forever.
        WARN("flagcxRmaProxyProgress: no sendComm for peer %d", p);
        flagcxRmaProxyRecordPendingError(proxy);
      }
    }

    if (!flagcxRmaProxyCircularBufEmpty(proxy, p) ||
        !flagcxIntruQueueEmpty(&proxy->inProgressQueues[p]))
      *anyOutstanding = true;
  }
  if (__atomic_load_n(&proxy->pendingError, __ATOMIC_ACQUIRE) &&
      flagcxRmaProxyAllNativeRequestsRetired(proxy))
    flagcxRmaProxyPublishError(proxy);
  return did;
}

flagcxResult_t
flagcxHeteroRmaProxyProgressOnce(struct flagcxRmaProxyState *proxy,
                                 bool stopping, int *madeProgress,
                                 int *hasOutstanding) {
  if (proxy == NULL || madeProgress == NULL || hasOutstanding == NULL)
    return flagcxInvalidArgument;
  bool outstanding = false;
  bool progressed = flagcxRmaProxyProgress(proxy, stopping, &outstanding);
  *madeProgress = progressed ? 1 : 0;
  *hasOutstanding = outstanding ? 1 : 0;
  return flagcxSuccess;
}

static void *flagcxRmaProxyProgressThread(void *arg) {
  struct flagcxRmaProxyState *proxy = (struct flagcxRmaProxyState *)arg;
  if (deviceAdaptor != NULL && deviceAdaptor->setDevice != NULL &&
      deviceAdaptor->setDevice(proxy->comm->cudaDev) != flagcxSuccess)
    flagcxRmaProxyPublishError(proxy);
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
  if (qs < 2 || (qs & (qs - 1)) != 0 || qs > UINT32_MAX / 2) {
    WARN("flagcxHeteroRmaProxyStart: invalid RMA queue size %u;"
         "FLAGCX_RMA_QUEUE_SIZE must be a power of two and >= 2",
         qs);
    free(proxy);
    return flagcxInvalidArgument;
  }
  if (nRanks < 0 || (uint64_t)nRanks * qs > UINT32_MAX) {
    WARN("flagcxHeteroRmaProxyStart: visibility domain index overflow "
         "nRanks=%d queueSize=%u",
         nRanks, qs);
    free(proxy);
    return flagcxInvalidArgument;
  }
  proxy->queueSize = qs;
  proxy->queueMask = qs - 1;
  // Descriptors remain tracked after leaving the producer ring while their
  // native requests are in flight.  Each side can hold queueSize entries, so
  // the completion scoreboard must cover their combined upper bound.
  const uint32_t scoreboardCapacity = qs * 2;

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
  proxy->completionScoreboards = (struct flagcxNetCompletionScoreboard *)calloc(
      nRanks, sizeof(struct flagcxNetCompletionScoreboard));
  proxy->completionEntries = (struct flagcxNetCompletionEntry *)calloc(
      (size_t)nRanks * scoreboardCapacity,
      sizeof(struct flagcxNetCompletionEntry));
  proxy->getVisibilityDomains = (struct flagcxNetGetVisibilityDomain *)calloc(
      (size_t)nRanks * qs, sizeof(struct flagcxNetGetVisibilityDomain));
  proxy->activeGetVisibilityDomains =
      (uint32_t *)calloc((size_t)nRanks * qs, sizeof(uint32_t));
  proxy->activeGetVisibilityCounts =
      (uint32_t *)calloc(nRanks, sizeof(uint32_t));
  proxy->groupSeqs = (volatile uint64_t *)calloc(nRanks, sizeof(uint64_t));
  proxy->generation = 1;

  if (proxy->circularBuffers == NULL || proxy->pis == NULL ||
      proxy->cis == NULL || proxy->inProgressQueues == NULL ||
      proxy->peerProducerMutexes == NULL || proxy->opSeqs == NULL ||
      proxy->doneSeqs == NULL || proxy->inFlights == NULL ||
      proxy->completionScoreboards == NULL ||
      proxy->completionEntries == NULL || proxy->getVisibilityDomains == NULL ||
      proxy->activeGetVisibilityDomains == NULL ||
      proxy->activeGetVisibilityCounts == NULL || proxy->groupSeqs == NULL) {
    WARN("flagcxHeteroRmaProxyStart: failed to allocate ring buffers");
    free(proxy->circularBuffers);
    free((void *)proxy->pis);
    free((void *)proxy->cis);
    free(proxy->inProgressQueues);
    free(proxy->peerProducerMutexes);
    free((void *)proxy->opSeqs);
    free((void *)proxy->doneSeqs);
    free((void *)proxy->inFlights);
    free(proxy->completionScoreboards);
    free(proxy->completionEntries);
    free(proxy->getVisibilityDomains);
    free(proxy->activeGetVisibilityDomains);
    free(proxy->activeGetVisibilityCounts);
    free((void *)proxy->groupSeqs);
    free(proxy);
    return flagcxSystemError;
  }

  for (int p = 0; p < nRanks; p++) {
    pthread_mutex_init(&proxy->peerProducerMutexes[p], NULL);
    flagcxIntruQueueConstruct(&proxy->inProgressQueues[p]);
    flagcxResult_t scoreboardResult = flagcxNetCompletionScoreboardInit(
        &proxy->completionScoreboards[p],
        &proxy->completionEntries[(size_t)p * scoreboardCapacity],
        scoreboardCapacity, proxy->generation, 1);
    if (scoreboardResult != flagcxSuccess) {
      WARN("flagcxHeteroRmaProxyStart: scoreboard init failed peer=%d res=%d",
           p, (int)scoreboardResult);
      for (int initialized = 0; initialized <= p; initialized++)
        pthread_mutex_destroy(&proxy->peerProducerMutexes[initialized]);
      free(proxy->circularBuffers);
      free((void *)proxy->pis);
      free((void *)proxy->cis);
      free(proxy->inProgressQueues);
      free(proxy->peerProducerMutexes);
      free((void *)proxy->opSeqs);
      free((void *)proxy->doneSeqs);
      free((void *)proxy->inFlights);
      free(proxy->completionScoreboards);
      free(proxy->completionEntries);
      free(proxy->getVisibilityDomains);
      free(proxy->activeGetVisibilityDomains);
      free(proxy->activeGetVisibilityCounts);
      free((void *)proxy->groupSeqs);
      free(proxy);
      return scoreboardResult;
    }
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
    free(proxy->completionScoreboards);
    free(proxy->completionEntries);
    free(proxy->getVisibilityDomains);
    free(proxy->activeGetVisibilityDomains);
    free(proxy->activeGetVisibilityCounts);
    free((void *)proxy->groupSeqs);
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
  free(proxy->completionScoreboards);
  free(proxy->completionEntries);
  free(proxy->getVisibilityDomains);
  free(proxy->activeGetVisibilityDomains);
  free(proxy->activeGetVisibilityCounts);
  free((void *)proxy->groupSeqs);
  free(proxy);
  comm->rmaProxy = NULL;
  return flagcxSuccess;
}

flagcxResult_t flagcxHeteroRmaProxyQuiesce(flagcxHeteroComm_t comm) {
  if (comm == NULL || comm->rmaProxy == NULL)
    return flagcxSuccess;

  struct flagcxRmaProxyState *proxy = comm->rmaProxy;
  __atomic_store_n(&proxy->quiesced, 1, __ATOMIC_RELEASE);

  // Close the producer race: a producer that acquired a lock before the store
  // may finish publishing, while every later producer observes quiesced and
  // fails fast. The progress thread remains alive to retire the accepted work.
  for (int p = 0; p < proxy->nRanks; p++) {
    pthread_mutex_lock(&proxy->peerProducerMutexes[p]);
    pthread_mutex_unlock(&proxy->peerProducerMutexes[p]);
  }

  while (true) {
    bool outstanding = false;
    for (int p = 0; p < proxy->nRanks; p++) {
      const uint32_t pi = __atomic_load_n(&proxy->pis[p], __ATOMIC_ACQUIRE);
      const uint32_t ci = __atomic_load_n(&proxy->cis[p], __ATOMIC_ACQUIRE);
      const uint32_t inFlight =
          __atomic_load_n(&proxy->inFlights[p], __ATOMIC_ACQUIRE);
      if (ci < pi || inFlight != 0) {
        outstanding = true;
        break;
      }
    }
    if (!outstanding)
      return flagcxSuccess;
    sched_yield();
  }
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
  // GPU-side wait on the local proxy completion counter.
  return deviceAdaptor->streamWaitValue64(
      stream, &proxy->doneSeqsDev[peer], seq, FLAGCX_STREAM_WAIT_VALUE_DEFAULT);
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
  // The counter tracks successful operations, but a concurrent transport
  // error may become terminal as the target is reached. Never hide it.
  return __atomic_load_n(&comm->rmaProxy->rmaError, __ATOMIC_ACQUIRE)
             ? flagcxRemoteError
             : flagcxSuccess;
}

flagcxResult_t flagcxHeteroSend(const void *sendbuff, size_t count,
                                flagcxDataType_t datatype, int peer,
                                flagcxHeteroComm_t comm, flagcxStream_t stream,
                                int opId, int step) {
  return flagcxHeteroSendOnChannel(sendbuff, count, datatype, peer, comm,
                                   stream, 0, opId, step);
}

flagcxResult_t flagcxHeteroSendOnChannel(const void *sendbuff, size_t count,
                                         flagcxDataType_t datatype, int peer,
                                         flagcxHeteroComm_t comm,
                                         flagcxStream_t stream, int channelId,
                                         int opId, int step) {
  if (comm == NULL || peer < 0 || peer >= comm->nRanks || channelId < 0 ||
      channelId >= MAXCHANNELS)
    return flagcxInvalidArgument;
  flagcxHeteroGroupStart();
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
  p2p->channelId = channelId;
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
  return flagcxHeteroRecvOnChannel(recvbuff, count, datatype, peer, comm,
                                   stream, 0, opId, step);
}

flagcxResult_t flagcxHeteroRecvOnChannel(void *recvbuff, size_t count,
                                         flagcxDataType_t datatype, int peer,
                                         flagcxHeteroComm_t comm,
                                         flagcxStream_t stream, int channelId,
                                         int opId, int step) {
  if (comm == NULL || peer < 0 || peer >= comm->nRanks || channelId < 0 ||
      channelId >= MAXCHANNELS)
    return flagcxInvalidArgument;
  flagcxHeteroGroupStart();
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
  p2p->channelId = channelId;
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

static inline bool flagcxRmaMrIndexIsValid(flagcxHeteroComm_t comm,
                                           int mrIndex) {
  return comm != NULL && mrIndex >= 0 && mrIndex < comm->oneSideHandleCount &&
         comm->oneSideHandles != NULL && comm->oneSideHandles[mrIndex] != NULL;
}

flagcxResult_t flagcxHeteroPut(flagcxHeteroComm_t comm, int peer,
                               size_t srcOffset, size_t dstOffset, size_t size,
                               int srcMrIdx, int dstMrIdx, bool streamSyncReady,
                               uint64_t *assignedSeq, uint64_t orderingKey,
                               bool independent) {
  if (comm == NULL)
    return flagcxInvalidArgument;
  if (comm->netAdaptor == NULL || comm->netAdaptor->iput == NULL)
    return flagcxNotSupported;
  if (peer < 0 || peer >= comm->nRanks) {
    WARN("flagcxHeteroPut: peer %d out of range (nRanks=%d)", peer,
         comm->nRanks);
    return flagcxInvalidArgument;
  }
  if (!flagcxRmaMrIndexIsValid(comm, srcMrIdx) ||
      !flagcxRmaMrIndexIsValid(comm, dstMrIdx))
    return flagcxNotSupported;
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
  desc->orderingKey = independent ? orderingKey : 0;
  desc->submitFlags = FLAGCX_RMA_SUBMIT_DATA |
                      (independent ? FLAGCX_RMA_SUBMIT_INDEPENDENT : 0);
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
  for (size_t i = 0; i < count; i++) {
    if (!flagcxRmaMrIndexIsValid(comm, srcMrIdxs[i]) ||
        !flagcxRmaMrIndexIsValid(comm, dstMrIdxs[i]))
      return flagcxNotSupported;
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
    desc->orderingKey = 0;
    desc->submitFlags = FLAGCX_RMA_SUBMIT_DATA;
    descs[i] = desc;
  }

  size_t enqueued = 0;
  flagcxResult_t res = flagcxSuccess;
  // The ring is an internal bounded transport detail, not a public batch-size
  // limit.  Publish each ring-sized chunk transactionally so a batch can make
  // progress while preserving the no-partial-publication rule for each chunk.
  while (enqueued < count) {
    size_t chunkCount = count - enqueued;
    if (chunkCount > comm->rmaProxy->queueSize)
      chunkCount = comm->rmaProxy->queueSize;
    size_t chunkEnqueued = 0;
    res = flagcxRmaProxyEnqueueDescBatch(comm->rmaProxy, peer, descs + enqueued,
                                         chunkCount, &chunkEnqueued);
    enqueued += chunkEnqueued;
    if (res != flagcxSuccess)
      break;
  }
  if (res != flagcxSuccess) {
    for (size_t i = enqueued; i < count; i++)
      free(descs[i]);
  }
  free(descs);
  return res;
}

flagcxResult_t flagcxHeteroBatchPutSignal(
    flagcxHeteroComm_t comm, int peer, const size_t *srcOffsets,
    const size_t *dstOffsets, const size_t *sizes, const int *srcMrIdxs,
    const int *dstMrIdxs, const uint64_t *orderingKeys,
    const uint8_t *independent, size_t count, size_t signalOffset,
    uint64_t signalValue, uint64_t releaseOrderingKey, bool releaseIndependent,
    uint64_t *assignedSeq) {
  if (comm == NULL || count == 0 || srcOffsets == NULL || dstOffsets == NULL ||
      sizes == NULL || srcMrIdxs == NULL || dstMrIdxs == NULL ||
      orderingKeys == NULL || independent == NULL)
    return flagcxInvalidArgument;
  if (peer < 0 || peer >= comm->nRanks)
    return flagcxInvalidArgument;
  if (comm->netAdaptor == NULL || comm->netAdaptor->iput == NULL ||
      comm->netAdaptor->iputSignal == NULL || comm->signalHandle == NULL)
    return flagcxNotSupported;
  if (comm->rmaProxy == NULL || comm->rmaProxy->groupSeqs == NULL)
    return flagcxInternalError;
  if (count >= comm->rmaProxy->queueSize)
    return flagcxInvalidArgument;
  for (size_t i = 0; i < count; i++) {
    if (!flagcxRmaMrIndexIsValid(comm, srcMrIdxs[i]) ||
        !flagcxRmaMrIndexIsValid(comm, dstMrIdxs[i]))
      return flagcxNotSupported;
  }

  struct flagcxRmaDesc **data =
      (struct flagcxRmaDesc **)calloc(count, sizeof(*data));
  struct flagcxRmaReleaseGroup *group = flagcxRmaReleaseGroupCreate();
  struct flagcxRmaDesc *release =
      (struct flagcxRmaDesc *)calloc(1, sizeof(*release));
  if (data == NULL || group == NULL || release == NULL) {
    free(data);
    free(release);
    flagcxRmaReleaseGroupRelease(group);
    return flagcxSystemError;
  }
  flagcxRmaDescSetReleaseGroup(release, group);

  flagcxResult_t result = flagcxSuccess;
  const uint64_t groupId =
      __atomic_add_fetch(&comm->rmaProxy->groupSeqs[peer], 1, __ATOMIC_RELAXED);
  result = flagcxNetReleaseGroupInit(flagcxRmaReleaseGroupTransport(group),
                                     groupId, comm->rmaProxy->generation);
  for (size_t i = 0; result == flagcxSuccess && i < count; i++) {
    data[i] = (struct flagcxRmaDesc *)calloc(1, sizeof(*data[i]));
    if (data[i] == NULL) {
      result = flagcxSystemError;
      break;
    }
    data[i]->type = FLAGCX_RMA_PUT;
    data[i]->srcOff = srcOffsets[i];
    data[i]->dstOff = dstOffsets[i];
    data[i]->size = sizes[i];
    data[i]->srcMrIdx = srcMrIdxs[i];
    data[i]->dstMrIdx = dstMrIdxs[i];
    data[i]->orderingKey = independent[i] ? orderingKeys[i] : 0;
    data[i]->groupId = groupId;
    data[i]->submitFlags = FLAGCX_RMA_SUBMIT_DATA |
                           (independent[i] ? FLAGCX_RMA_SUBMIT_INDEPENDENT : 0);
    flagcxRmaDescSetReleaseGroup(data[i], group);
  }

  if (result == flagcxSuccess) {
    release->type = FLAGCX_RMA_RELEASE;
    release->srcMrIdx = -1;
    release->dstMrIdx = -1;
    release->signalOff = signalOffset;
    release->signalValue = signalValue;
    release->orderingKey = releaseIndependent ? releaseOrderingKey : 0;
    release->groupId = groupId;
    release->submitFlags =
        FLAGCX_RMA_SUBMIT_RELEASE |
        (releaseIndependent ? FLAGCX_RMA_SUBMIT_INDEPENDENT : 0);
    result = flagcxRmaProxyEnqueueReleaseGroup(comm->rmaProxy, peer, data,
                                               (uint32_t)count, release, false,
                                               assignedSeq);
  }

  if (result != flagcxSuccess) {
    for (size_t i = 0; i < count; i++)
      flagcxRmaDescDestroy(data[i]);
    flagcxRmaDescDestroy(release);
  }
  flagcxRmaReleaseGroupRelease(group); // creator reference
  free(data);
  return result;
}

flagcxResult_t flagcxHeteroGet(flagcxHeteroComm_t comm, int peer,
                               size_t srcOffset, size_t dstOffset, size_t size,
                               int srcMrIdx, int dstMrIdx, uint64_t orderingKey,
                               bool independent, flagcxEvent_t streamEvent,
                               uint64_t *assignedSeq) {
  // Ownership of a non-null event transfers here, including on failure.
  auto reject = [streamEvent](flagcxResult_t result) {
    if (streamEvent != NULL)
      deviceAdaptor->eventDestroy(streamEvent);
    return result;
  };
  if (comm == NULL)
    return reject(flagcxInvalidArgument);
  if (comm->netAdaptor == NULL || comm->netAdaptor->iget == NULL)
    return reject(flagcxNotSupported);
  if (peer < 0 || peer >= comm->nRanks) {
    WARN("flagcxHeteroGet: peer %d out of range (nRanks=%d)", peer,
         comm->nRanks);
    return reject(flagcxInvalidArgument);
  }
  if (!flagcxRmaMrIndexIsValid(comm, srcMrIdx) ||
      !flagcxRmaMrIndexIsValid(comm, dstMrIdx))
    return reject(flagcxNotSupported);
  if (comm->rmaProxy == NULL) {
    WARN("flagcxHeteroGet: rmaProxy not initialized");
    return reject(flagcxInternalError);
  }
  struct flagcxRmaDesc *desc = (struct flagcxRmaDesc *)calloc(1, sizeof(*desc));
  if (desc == NULL)
    return reject(flagcxSystemError);
  desc->type = FLAGCX_RMA_GET;
  desc->streamEvent = streamEvent;
  desc->eventArmed = streamEvent != NULL;
  desc->srcOff = (uint64_t)srcOffset;
  desc->dstOff = (uint64_t)dstOffset;
  desc->size = size;
  desc->srcMrIdx = srcMrIdx;
  desc->dstMrIdx = dstMrIdx;
  desc->orderingKey = independent ? orderingKey : 0;
  desc->submitFlags = FLAGCX_RMA_SUBMIT_DATA |
                      (independent ? FLAGCX_RMA_SUBMIT_INDEPENDENT : 0);
  flagcxResult_t res = flagcxRmaProxyEnqueueDesc(
      comm->rmaProxy, peer, desc, streamEvent != NULL, assignedSeq);
  if (res != flagcxSuccess)
    flagcxRmaDescDestroy(desc);
  return res;
}

flagcxResult_t flagcxHeteroGetStream(flagcxHeteroComm_t comm, int peer,
                                     size_t srcOffset, size_t dstOffset,
                                     size_t size, int srcMrIdx, int dstMrIdx,
                                     flagcxSymWindow_t srcWindow,
                                     flagcxSymWindow_t dstWindow,
                                     flagcxStream_t stream) {
  if (comm == NULL || stream == NULL || peer < 0 || peer >= comm->nRanks)
    return flagcxInvalidArgument;
  struct flagcxRmaProxyState *proxy = comm->rmaProxy;
  if (proxy == NULL)
    return flagcxInternalError;

  if (!flagcxParamP2pDisable() && flagcxIsIntraNode(comm, peer) &&
      srcWindow != NULL && dstWindow != NULL && dstWindow->localBase != NULL &&
      dstOffset <= dstWindow->heapSize &&
      size <= dstWindow->heapSize - dstOffset && deviceAdaptor != NULL &&
      deviceAdaptor->deviceMemcpy != NULL &&
      deviceAdaptor->eventCreate != NULL &&
      deviceAdaptor->eventRecord != NULL && deviceAdaptor->eventQuery != NULL) {
    void *srcBuf = NULL;
    flagcxResult_t mapResult = flagcxSymWindowResolveIpcPeerPtr(
        comm, srcWindow, peer, srcOffset, size, &srcBuf);
    if (mapResult == flagcxSuccess && srcBuf != NULL) {
      void *dstBuf = (void *)((uintptr_t)dstWindow->localBase + dstOffset);
      flagcxEvent_t doneEvent = NULL;
      flagcxResult_t result =
          deviceAdaptor->eventCreate(&doneEvent, flagcxEventDisableTiming);
      if (result != flagcxSuccess)
        return result;
      struct flagcxRmaDesc *desc =
          (struct flagcxRmaDesc *)calloc(1, sizeof(*desc));
      if (desc == NULL) {
        deviceAdaptor->eventDestroy(doneEvent);
        return flagcxSystemError;
      }
      desc->type = FLAGCX_RMA_IPC_GET;
      desc->streamEvent = doneEvent;
      desc->ipcStatus = flagcxSuccess;
      desc->submitFlags = FLAGCX_RMA_SUBMIT_DATA;
      desc->srcMrIdx = -1;
      desc->dstMrIdx = -1;

      pthread_mutex_lock(&proxy->peerProducerMutexes[peer]);
      while (true) {
        if (__atomic_load_n(&proxy->quiesced, __ATOMIC_ACQUIRE) ||
            __atomic_load_n(&proxy->pendingError, __ATOMIC_ACQUIRE) ||
            __atomic_load_n(&proxy->rmaError, __ATOMIC_ACQUIRE)) {
          pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
          flagcxRmaDescDestroy(desc);
          return flagcxInvalidUsage;
        }
        result = flagcxInProgress;
        if (!flagcxRmaProxyCircularBufFull(proxy, peer))
          result = flagcxRmaProxyPrepareDesc(proxy, peer, desc);
        if (result == flagcxSuccess)
          break;
        if (result != flagcxInProgress) {
          pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
          flagcxRmaDescDestroy(desc);
          return result;
        }
        pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
        sched_yield();
        pthread_mutex_lock(&proxy->peerProducerMutexes[peer]);
      }
      uint32_t pi = __atomic_load_n(&proxy->pis[peer], __ATOMIC_RELAXED);
      proxy->circularBuffers[(size_t)peer * proxy->queueSize +
                             (pi & proxy->queueMask)] = desc;
      __atomic_store_n(&proxy->pis[peer], pi + 1, __ATOMIC_RELEASE);

      // Submission remains under the producer lock until eventArmed is set.
      // The error drain takes this lock, so it cannot free desc mid-submit.
      if (desc->opSeq > 1)
        result = flagcxRmaWaitDone(proxy, peer, desc->opSeq - 1, stream);
      if (result == flagcxSuccess && size > 0)
        result = deviceAdaptor->deviceMemcpy(
            dstBuf, srcBuf, size, flagcxMemcpyDeviceToDevice, stream, NULL);
      if (result == flagcxSuccess)
        result = deviceAdaptor->eventRecord(doneEvent, stream);
      desc->ipcStatus = result;
      __atomic_store_n(&desc->eventArmed, 1, __ATOMIC_RELEASE);
      pthread_mutex_unlock(&proxy->peerProducerMutexes[peer]);
      return result;
    }
  }

  uint64_t assignedSeq = 0;
  if (deviceAdaptor == NULL || deviceAdaptor->eventCreate == NULL ||
      deviceAdaptor->eventRecord == NULL || deviceAdaptor->eventQuery == NULL)
    return flagcxNotSupported;
  flagcxEvent_t readyEvent = NULL;
  flagcxResult_t res =
      deviceAdaptor->eventCreate(&readyEvent, flagcxEventDisableTiming);
  if (res != flagcxSuccess)
    return res;
  res = deviceAdaptor->eventRecord(readyEvent, stream);
  if (res != flagcxSuccess) {
    deviceAdaptor->eventDestroy(readyEvent);
    return res;
  }
  res = flagcxHeteroGet(comm, peer, srcOffset, dstOffset, size, srcMrIdx,
                        dstMrIdx, 0, false, readyEvent, &assignedSeq);
  if (res != flagcxSuccess)
    return res;
  return flagcxRmaWaitDone(proxy, peer, assignedSeq, stream);
}

flagcxResult_t flagcxHeteroPutSignal(flagcxHeteroComm_t comm, int peer,
                                     size_t srcOffset, size_t dstOffset,
                                     size_t size, size_t signalOffset,
                                     int srcMrIdx, int dstMrIdx,
                                     uint64_t signalValue, bool streamSyncReady,
                                     uint64_t *assignedSeq,
                                     uint64_t orderingKey, bool independent) {
  if (comm == NULL)
    return flagcxInvalidArgument;
  if (comm->netAdaptor == NULL || comm->netAdaptor->iputSignal == NULL ||
      (size > 0 && comm->netAdaptor->iput == NULL))
    return flagcxNotSupported;
  if (peer < 0 || peer >= comm->nRanks) {
    WARN("flagcxHeteroPutSignal: peer %d out of range (nRanks=%d)", peer,
         comm->nRanks);
    return flagcxInvalidArgument;
  }
  if ((size > 0 && (!flagcxRmaMrIndexIsValid(comm, srcMrIdx) ||
                    !flagcxRmaMrIndexIsValid(comm, dstMrIdx))) ||
      comm->signalHandle == NULL)
    return flagcxNotSupported;
  if (comm->rmaProxy == NULL || comm->rmaProxy->groupSeqs == NULL) {
    WARN("flagcxHeteroPutSignal: rmaProxy not initialized");
    return flagcxInternalError;
  }
  struct flagcxRmaReleaseGroup *group = flagcxRmaReleaseGroupCreate();
  struct flagcxRmaDesc *data =
      size > 0 ? (struct flagcxRmaDesc *)calloc(1, sizeof(*data)) : NULL;
  struct flagcxRmaDesc *release =
      (struct flagcxRmaDesc *)calloc(1, sizeof(*release));
  if (group == NULL || release == NULL || (size > 0 && data == NULL)) {
    free(data);
    free(release);
    flagcxRmaReleaseGroupRelease(group);
    return flagcxSystemError;
  }
  flagcxRmaDescSetReleaseGroup(release, group);

  const uint64_t groupId =
      __atomic_add_fetch(&comm->rmaProxy->groupSeqs[peer], 1, __ATOMIC_RELAXED);
  flagcxResult_t res =
      flagcxNetReleaseGroupInit(flagcxRmaReleaseGroupTransport(group), groupId,
                                comm->rmaProxy->generation);
  if (res != flagcxSuccess) {
    free(data);
    flagcxRmaDescDestroy(release);
    flagcxRmaReleaseGroupRelease(group);
    return res;
  }

  if (data != NULL) {
    data->type = FLAGCX_RMA_PUT;
    data->srcOff = (uint64_t)srcOffset;
    data->dstOff = (uint64_t)dstOffset;
    data->size = size;
    data->srcMrIdx = srcMrIdx;
    data->dstMrIdx = dstMrIdx;
    data->orderingKey = independent ? orderingKey : 0;
    data->groupId = groupId;
    data->submitFlags = FLAGCX_RMA_SUBMIT_DATA |
                        (independent ? FLAGCX_RMA_SUBMIT_INDEPENDENT : 0);
    flagcxRmaDescSetReleaseGroup(data, group);
  }

  release->type = FLAGCX_RMA_RELEASE;
  release->srcMrIdx = -1;
  release->dstMrIdx = -1;
  release->signalOff = (uint64_t)signalOffset;
  release->signalValue = signalValue;
  release->orderingKey = independent ? orderingKey : 0;
  release->groupId = groupId;
  release->submitFlags = FLAGCX_RMA_SUBMIT_RELEASE |
                         (independent ? FLAGCX_RMA_SUBMIT_INDEPENDENT : 0);
  struct flagcxRmaDesc *dataMembers[1] = {data};
  res = flagcxRmaProxyEnqueueReleaseGroup(comm->rmaProxy, peer, dataMembers,
                                          data == NULL ? 0u : 1u, release,
                                          streamSyncReady, assignedSeq);
  if (res != flagcxSuccess) {
    flagcxRmaDescDestroy(data);
    flagcxRmaDescDestroy(release);
  }
  flagcxRmaReleaseGroupRelease(group); // creator reference
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

  // Preserve the explicit flush API's legacy behavior, including BAREX's
  // transitional no-op. Correctness-sensitive automatic paths separately
  // validate gdrFlushCaps before treating a callback as a visibility boundary.

  void *data_arr[1] = {gpuAddr};
  int sizes_arr[1] = {flagcxRmaVisibilityFlushSize(size)};
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

static inline bool flagcxIsIntraNode(flagcxHeteroComm_t comm, int peer);

flagcxResult_t flagcxHeteroWaitSignal(flagcxHeteroComm_t comm, int peer,
                                      size_t signalOffset, uint64_t expected,
                                      flagcxStream_t stream) {
  (void)peer;
  if (comm == NULL || comm->rmaSignalBase == NULL)
    return flagcxNotSupported;
  if (signalOffset > comm->rmaSignalSize ||
      sizeof(uint64_t) > comm->rmaSignalSize - signalOffset)
    return flagcxInvalidArgument;

  void *signalAddr = (void *)((uintptr_t)comm->rmaSignalBase + signalOffset);

  // Device-side wait (streamWaitValue64) for GPU signal buffer.
  // RMA signal buffers are GPU memory (flagcxMemAlloc) — host-side volatile
  // polling would segfault. Non-CUDA platforms return flagcxNotSupported.
  // The signal is published by a peer GPU or NIC after its payload writes.
  // WRITE requirements request an acquire-style remote-write flush; adaptors
  // that cannot provide it must return flagcxNotSupported. PPU currently uses
  // a documented transitional NONE policy while BAREX has no real flush API,
  // so its auto mode deliberately uses a plain wait for CI compatibility.
  if (stream == NULL)
    return flagcxInternalError;
  if (deviceAdaptor == NULL || deviceAdaptor->streamWaitValue64 == NULL)
    return flagcxNotSupported;

  const uint8_t registrationRoute =
      comm->signalHandle == nullptr
          ? static_cast<uint8_t>(FLAGCX_VMM_MR_ROUTE_NONE)
          : comm->signalHandle->registrationRoute;
  // An intra-node sender may publish this signal through the IPC/D2D fast
  // path. Keep the peer-GPU acquire even when Hopper/NIC topology would permit
  // omitting a network WRITE flush. If the fast path later falls back to the
  // NIC, the extra acquire is conservative.
  const int peerGpuMayPublish =
      !flagcxParamP2pDisable() && flagcxIsIntraNode(comm, peer);
  flagcxGdrVisibilityDecision visibility = {};
  FLAGCXCHECK(flagcxResolveGdrVisibilityForConnection(
      comm->topoServer, comm->rank, comm->compCap, comm->netAdaptor,
      comm->netDev, 1, peerGpuMayPublish, registrationRoute, &visibility));
  const int waitFlags =
      (visibility.requirements & FLAGCX_GDR_WRITE_REQUIRES_FLUSH)
          ? FLAGCX_STREAM_WAIT_VALUE_FLUSH_REMOTE_WRITES
          : FLAGCX_STREAM_WAIT_VALUE_DEFAULT;
  return deviceAdaptor->streamWaitValue64(stream, signalAddr, expected,
                                          waitFlags);
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
  desc->orderingKey = 0;
  desc->submitFlags = FLAGCX_RMA_SUBMIT_DATA;
  flagcxResult_t res = flagcxRmaProxyEnqueueDesc(comm->rmaProxy, peer, desc);
  if (res != flagcxSuccess)
    free(desc);
  return res;
}

// ---- Intra-node topology helper ----
static inline bool flagcxIsIntraNode(flagcxHeteroComm_t comm, int peer) {
  if (comm->rankToNode == NULL)
    return false;
  return comm->rankToNode[peer] == comm->node;
}

// ---- IPC state initialization ----

flagcxResult_t flagcxHeteroRmaIpcInit(flagcxHeteroComm_t comm) {
  if (comm == NULL)
    return flagcxInvalidArgument;
  struct flagcxRmaProxyState *proxy = comm->rmaProxy;
  if (proxy == NULL)
    return flagcxInternalError;
  if (proxy->ipcState != NULL)
    return flagcxSuccess;

  // Symmetric windows own their IPC locators independently of network MRs.
  bool hasD2dMemory = false;
  for (flagcxSymWindow_t window = comm->symWindows; window != NULL;
       window = window->next) {
    if ((window->hasFlatMapping && window->flatBase != NULL) ||
        window->ipcSlot >= 0) {
      hasD2dMemory = true;
      break;
    }
  }
  if (comm->rmaSignalIpcSlot >= 0)
    hasD2dMemory = true;

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

  ipc->peerSignalBufs = (void **)calloc(nRanks, sizeof(void *));
  ipc->signalSeqs = (uint64_t *)calloc(nRanks, sizeof(uint64_t));
  if (ipc->peerSignalBufs == NULL || ipc->signalSeqs == NULL) {
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

    int slot = comm->rmaSignalIpcSlot;
    if (slot >= 0 && comm->ipcTable != NULL && slot < comm->ipcTableSize) {
      struct flagcxIpcTableEntry *entry = &comm->ipcTable[slot];
      if (entry->inUse && entry->hostPeerPtrs != NULL && peerLocalRank >= 0 &&
          peerLocalRank < entry->nPeers) {
        ipc->peerSignalBufs[p] = entry->hostPeerPtrs[peerLocalRank];
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
                                     flagcxSymWindow_t srcWindow,
                                     flagcxSymWindow_t dstWindow,
                                     flagcxStream_t stream, uint64_t *opSeq) {
  if (comm == NULL || peer < 0 || peer >= comm->nRanks)
    return flagcxInvalidArgument;
  struct flagcxRmaProxyState *proxy = comm->rmaProxy;
  if (proxy == NULL)
    return flagcxInternalError;
  if (__atomic_load_n(&proxy->quiesced, __ATOMIC_ACQUIRE))
    return flagcxInvalidUsage;

  // Try intra-node D2D path (lazy init if not yet built)
  if (stream != NULL && !flagcxParamP2pDisable() &&
      flagcxIsIntraNode(comm, peer)) {
    if (srcWindow != NULL && srcWindow->localBase != NULL &&
        dstWindow != NULL && srcOffset <= srcWindow->heapSize &&
        size <= srcWindow->heapSize - srcOffset) {
      void *srcBuf = NULL;
      void *dstBuf = NULL;
      srcBuf = (void *)((uintptr_t)srcWindow->localBase + srcOffset);
      flagcxResult_t ipcRes = flagcxSymWindowResolveIpcPeerPtr(
          comm, dstWindow, peer, dstOffset, size, &dstBuf);

      if (ipcRes == flagcxSuccess && srcBuf != NULL && dstBuf != NULL) {
        flagcxResult_t res = deviceAdaptor->deviceMemcpy(
            dstBuf, srcBuf, size, flagcxMemcpyDeviceToDevice, stream, NULL);
        if (opSeq != NULL)
          *opSeq = 0; // D2D path has no opSeq (no proxy involvement)
        return res;
      }
      // Fall through to proxy path if buffer resolution fails
    }
  }

  if (!flagcxRmaMrIndexIsValid(comm, srcMrIdx) ||
      !flagcxRmaMrIndexIsValid(comm, dstMrIdx))
    return flagcxNotSupported;

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

flagcxResult_t flagcxHeteroPutSignalStream(
    flagcxHeteroComm_t comm, int peer, size_t srcOffset, size_t dstOffset,
    size_t size, size_t signalOffset, int srcMrIdx, int dstMrIdx,
    uint64_t signalValue, flagcxSymWindow_t srcWindow,
    flagcxSymWindow_t dstWindow, flagcxStream_t stream, uint64_t *opSeq) {
  if (comm == NULL || peer < 0 || peer >= comm->nRanks)
    return flagcxInvalidArgument;
  if (comm->rmaSignalBase == NULL)
    return flagcxNotSupported;
  if (signalOffset > comm->rmaSignalSize ||
      sizeof(uint64_t) > comm->rmaSignalSize - signalOffset)
    return flagcxInvalidArgument;
  struct flagcxRmaProxyState *proxy = comm->rmaProxy;
  if (proxy == NULL)
    return flagcxInternalError;
  if (__atomic_load_n(&proxy->quiesced, __ATOMIC_ACQUIRE))
    return flagcxInvalidUsage;

  // Try intra-node D2D path (lazy init if not yet built)
  if (stream != NULL && !flagcxParamP2pDisable() &&
      flagcxIsIntraNode(comm, peer)) {
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
      if (size > 0 && srcWindow != NULL && srcWindow->localBase != NULL &&
          srcOffset <= srcWindow->heapSize &&
          size <= srcWindow->heapSize - srcOffset)
        srcBuf = (void *)((uintptr_t)srcWindow->localBase + srcOffset);
      // Resolve destination buffer (peer)
      if (size > 0 && dstWindow != NULL &&
          flagcxSymWindowResolveIpcPeerPtr(comm, dstWindow, peer, dstOffset,
                                           size, &dstBuf) != flagcxSuccess)
        dstBuf = NULL;
      // Resolve signal address (peer's signal buffer)
      if (ipc->peerSignalBufs[peer] != NULL) {
        signalAddr =
            (void *)((uintptr_t)ipc->peerSignalBufs[peer] + signalOffset);
      }

      if ((size == 0 || (srcBuf != NULL && dstBuf != NULL)) &&
          signalAddr != NULL) {
        if (deviceAdaptor == NULL || deviceAdaptor->streamWriteValue64 == NULL)
          return flagcxNotSupported;
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

  // The network fallback requires registered data and signal MRs. Keep this
  // check after the IPC attempt so an IPC-capable window does not depend on
  // network registration state.
  if ((size > 0 && (!flagcxRmaMrIndexIsValid(comm, srcMrIdx) ||
                    !flagcxRmaMrIndexIsValid(comm, dstMrIdx))) ||
      comm->signalHandle == NULL)
    return flagcxNotSupported;

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
