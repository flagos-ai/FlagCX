#include "work.h"
#include "adaptor.h"
#include "comm.h"
#include "device_api/completion_word.h"
#include "flagcx_hetero.h"
#include "global_comm.h"
#include "proxy.h"
#include <sched.h>
#include <stdlib.h>

static __thread struct flagcxWorkRecord *pendingWork[2];
static __thread int workGroupDepth[2];
static __thread struct flagcxWorkRecord *currentWork;
static __thread int nativeSubmissionPending;

flagcxResult_t flagcxWorkNativeGroupResult(flagcxResult_t result) {
  if (result == flagcxInProgress)
    nativeSubmissionPending = 1;
  else if (result == flagcxSuccess)
    nativeSubmissionPending = 0;
  return result;
}

struct flagcxWorkRecord *flagcxWorkSetCurrent(struct flagcxWorkRecord *record) {
  struct flagcxWorkRecord *previous = currentWork;
  if (record != NULL && previous == NULL)
    nativeSubmissionPending = 0;
  currentWork = record;
  return previous;
}

static flagcxResult_t workInit(struct flagcxWorkTracker *tracker) {
  for (;;) {
    int state = __atomic_load_n(&tracker->initialized, __ATOMIC_ACQUIRE);
    if (state == 2)
      return flagcxSuccess;
    int expected = 0;
    if (state == 0 &&
        __atomic_compare_exchange_n(&tracker->initialized, &expected, 1, false,
                                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
      int result = pthread_mutex_init(&tracker->mutex, NULL);
      __atomic_store_n(&tracker->initialized, result == 0 ? 2 : 0,
                       __ATOMIC_RELEASE);
      return result == 0 ? flagcxSuccess : flagcxSystemError;
    }
    sched_yield();
  }
}

flagcxResult_t flagcxWorkReleaseBuffer(struct flagcxWorkRecord *record) {
  if (record->buffer == NULL)
    return flagcxSuccess;
  flagcxResult_t result =
      deviceAdaptor->deviceFree(record->buffer, flagcxMemDevice, NULL);
  if (result == flagcxSuccess)
    record->buffer = NULL;
  return result;
}

static flagcxResult_t workCheckCompletion(struct flagcxWorkRecord *record,
                                          int wait, bool *complete) {
  *complete = false;
  if (record->nativeRequest != NULL) {
    int done = 0;
    flagcxResult_t result =
        record->nativeAdaptor->test(record->nativeRequest, &done, NULL);
    if (done) {
      record->nativeRequest = NULL;
      *complete = true;
      // Physical retirement is independent of the already-reported API error.
      return flagcxSuccess;
    }
    return result == flagcxSuccess ? flagcxInProgress : result;
  }
  flagcxResult_t result = deviceAdaptor->eventQuery(record->event);
  if (wait && result == flagcxSuccess)
    result = deviceAdaptor->eventSynchronize(record->event);
  *complete = result == flagcxSuccess;
  return result;
}

static flagcxResult_t workCollect(struct flagcxWorkTracker *tracker) {
  uint64_t after = 0;
  for (int budget = 0; budget < 8; ++budget) {
    pthread_mutex_lock(&tracker->mutex);
    struct flagcxWorkRecord *record = tracker->head;
    while (record != NULL && (record->id <= after || record->pins != 0 ||
                              (record->state != flagcxWorkRecorded &&
                               record->state != flagcxWorkNativePending &&
                               record->state != flagcxWorkComplete)))
      record = record->next;
    if (tracker->closed || record == NULL) {
      pthread_mutex_unlock(&tracker->mutex);
      break;
    }
    after = record->id;
    int complete = record->state == flagcxWorkComplete;
    record->state = flagcxWorkChecking;
    pthread_mutex_unlock(&tracker->mutex);
    bool physicallyComplete = complete;
    flagcxResult_t result =
        complete ? flagcxSuccess
                 : workCheckCompletion(record, 0, &physicallyComplete);
    if (physicallyComplete)
      result = flagcxWorkReleaseBuffer(record);
    pthread_mutex_lock(&tracker->mutex);
    record->state = physicallyComplete ? flagcxWorkComplete
                                       : (record->nativeRequest != NULL
                                              ? flagcxWorkNativePending
                                              : flagcxWorkRecorded);
    if (result == flagcxSuccess && record->pins == 0) {
      struct flagcxWorkRecord **link = &tracker->head, *previous = NULL;
      while (*link != record) {
        previous = *link;
        link = &(*link)->next;
      }
      *link = record->next;
      if (tracker->tail == record)
        tracker->tail = previous;
      struct flagcxWorkRecord **list = &tracker->retired;
      if (record->reusable && tracker->spareCount < 32) {
        list = &tracker->spare;
        ++tracker->spareCount;
      }
      record->next = *list;
      *list = record;
    }
    pthread_mutex_unlock(&tracker->mutex);
    if (result != flagcxSuccess && result != flagcxInProgress)
      return result;
  }
  // Failed destruction remains attached to its owner for a later retry.
  for (int budget = 0; budget < 8; ++budget) {
    pthread_mutex_lock(&tracker->mutex);
    struct flagcxWorkRecord *record = tracker->retired;
    if (tracker->closed || record == NULL ||
        record->state == flagcxWorkChecking) {
      pthread_mutex_unlock(&tracker->mutex);
      break;
    }
    record->state = flagcxWorkChecking;
    pthread_mutex_unlock(&tracker->mutex);
    flagcxResult_t result = record->event != NULL
                                ? deviceAdaptor->eventDestroy(record->event)
                                : flagcxSuccess;
    pthread_mutex_lock(&tracker->mutex);
    record->state = flagcxWorkComplete;
    if (result == flagcxSuccess) {
      struct flagcxWorkRecord **link = &tracker->retired;
      while (*link != record)
        link = &(*link)->next;
      *link = record->next;
      free(record);
    }
    pthread_mutex_unlock(&tracker->mutex);
    if (result != flagcxSuccess)
      return result;
  }
  return flagcxSuccess;
}

static flagcxResult_t workPrepare(struct flagcxWorkTracker *tracker,
                                  flagcxStream_t stream,
                                  struct flagcxWorkRecord **output,
                                  int accepted, int needEvent = 1) {
  if (tracker == NULL || output == NULL)
    return flagcxInvalidArgument;
  *output = NULL;
  if (!accepted && __atomic_load_n(&tracker->closed, __ATOMIC_ACQUIRE))
    return flagcxInvalidUsage;
  flagcxResult_t result = workInit(tracker);
  if (result != flagcxSuccess)
    return result;
  result = workCollect(tracker);
  if (result != flagcxSuccess)
    return result;
  pthread_mutex_lock(&tracker->mutex);
  if ((!accepted && tracker->closed) ||
      (accepted && (currentWork == NULL || currentWork->owner != tracker ||
                    currentWork->state != flagcxWorkPreparing))) {
    pthread_mutex_unlock(&tracker->mutex);
    return flagcxInvalidUsage;
  }
  struct flagcxWorkRecord *record = tracker->spare;
  if (record != NULL) {
    tracker->spare = record->next;
    --tracker->spareCount;
  } else {
    record = (struct flagcxWorkRecord *)calloc(1, sizeof(*record));
  }
  if (record == NULL) {
    pthread_mutex_unlock(&tracker->mutex);
    return flagcxSystemError;
  }
  record->next = record->groupNext = NULL;
  record->parent = currentWork;
  record->owner = tracker;
  record->stream = stream;
  record->state = flagcxWorkPreparing;
  record->reusable = 0;
  record->operationResult = record->completionResult = flagcxSuccess;
  record->id = ++tracker->nextId;
  ++tracker->submitting;
  if (tracker->tail != NULL)
    tracker->tail->next = record;
  else
    tracker->head = record;
  tracker->tail = record;
  pthread_mutex_unlock(&tracker->mutex);

  if (!needEvent) {
    record->reusable = 1;
    *output = record;
    return flagcxSuccess;
  }
  if (deviceAdaptor->eventCreate == NULL ||
      deviceAdaptor->eventRecord == NULL ||
      deviceAdaptor->eventSynchronize == NULL ||
      deviceAdaptor->eventQuery == NULL || deviceAdaptor->eventDestroy == NULL)
    result = flagcxNotSupported;
  else if (record->event == NULL)
    result =
        deviceAdaptor->eventCreate(&record->event, flagcxEventDisableTiming);
  if (result != flagcxSuccess || record->event == NULL) {
    flagcxWorkCancel(record);
    return result != flagcxSuccess ? result : flagcxInternalError;
  }
  record->reusable = 1;
  *output = record;
  return flagcxSuccess;
}

flagcxResult_t flagcxWorkPrepare(struct flagcxWorkTracker *tracker,
                                 flagcxStream_t stream,
                                 struct flagcxWorkRecord **output) {
  return workPrepare(tracker, stream, output, 0);
}
flagcxResult_t flagcxWorkPrepareAccepted(struct flagcxWorkTracker *tracker,
                                         flagcxStream_t stream,
                                         struct flagcxWorkRecord **output) {
  return workPrepare(tracker, stream, output,
                     currentWork != NULL && currentWork->owner == tracker);
}

flagcxResult_t flagcxWorkFinishNative(struct flagcxWorkRecord *record,
                                      flagcxResult_t result) {
  pthread_mutex_lock(&record->owner->mutex);
  record->operationResult = result;
  record->state = flagcxWorkNativePending;
  record->stream = NULL;
  --record->owner->submitting;
  pthread_mutex_unlock(&record->owner->mutex);
  return result;
}

flagcxResult_t flagcxWorkCancel(struct flagcxWorkRecord *record) {
  if (record == NULL)
    return flagcxSuccess;
  if (record->owner == NULL)
    return flagcxInvalidArgument;
  pthread_mutex_lock(&record->owner->mutex);
  if (record->state != flagcxWorkPreparing) {
    pthread_mutex_unlock(&record->owner->mutex);
    return flagcxInvalidUsage;
  }
  record->stream = NULL;
  record->state = flagcxWorkComplete;
  --record->owner->submitting;
  pthread_mutex_unlock(&record->owner->mutex);
  return flagcxSuccess;
}

flagcxResult_t flagcxWorkPrepareHost(struct flagcxWorkTracker *tracker,
                                     struct flagcxWorkRecord **record) {
  return workPrepare(tracker, NULL, record,
                     currentWork != NULL && currentWork->owner == tracker, 0);
}

static flagcxResult_t workRecord(struct flagcxWorkRecord *record,
                                 flagcxResult_t submission,
                                 const flagcxWorkGroupBoundary *boundary,
                                 int finishSubmission, int submissionKnown) {
  flagcxResult_t result = submission == flagcxInProgress && !submissionKnown
                              ? flagcxInProgress
                              : flagcxSuccess;
  flagcxStream_t stream = record->stream;
  if (boundary != NULL && boundary->submitted) {
    if (!boundary->ready)
      result = submission != flagcxSuccess ? submission : flagcxInternalError;
    else
      stream = boundary->stream;
  }
  if (boundary != NULL && boundary->ready)
    result = flagcxSuccess;
  if (result == flagcxSuccess)
    result = deviceAdaptor->eventRecord(record->event, stream);
  pthread_mutex_lock(&record->owner->mutex);
  record->state =
      result == flagcxSuccess ? flagcxWorkRecorded : flagcxWorkUncertain;
  record->completionResult = result;
  record->stream = NULL;
  if (finishSubmission)
    --record->owner->submitting;
  pthread_mutex_unlock(&record->owner->mutex);
  return result;
}

flagcxResult_t flagcxWorkFinish(struct flagcxWorkRecord *record,
                                flagcxStream_t stream, flagcxResult_t result,
                                enum flagcxWorkGroupDomain domain, int deferred,
                                int submissionKnown) {
  if (record == NULL)
    return flagcxInvalidArgument;
  record->stream = stream;
  record->operationResult = result;
  if (deferred) {
    pthread_mutex_lock(&record->owner->mutex);
    record->state = flagcxWorkGroupPending;
    --record->owner->submitting;
    pthread_mutex_unlock(&record->owner->mutex);
    record->groupNext = pendingWork[domain];
    pendingWork[domain] = record;
    return result;
  }
  flagcxResult_t completion =
      workRecord(record,
                 domain == flagcxWorkNativeGroup && nativeSubmissionPending
                     ? flagcxInProgress
                     : result,
                 NULL, 1, submissionKnown && !nativeSubmissionPending);
  return result != flagcxSuccess ? result : completion;
}

flagcxResult_t flagcxWorkGroupCompleteAtBoundary(
    enum flagcxWorkGroupDomain domain, flagcxResult_t result,
    const struct flagcxWorkGroupBoundary *boundary) {
  struct flagcxWorkRecord **link = &pendingWork[domain];
  flagcxResult_t first = result;
  while (*link != NULL) {
    struct flagcxWorkRecord *record = *link;
    if (boundary != NULL && boundary->owner != NULL &&
        boundary->owner != record->owner) {
      link = &record->groupNext;
      continue;
    }
    *link = record->groupNext;
    record->groupNext = NULL;
    flagcxResult_t completion = workRecord(record, result, boundary, 0, 0);
    if (first == flagcxSuccess)
      first = completion;
  }
  return first;
}
flagcxResult_t flagcxWorkGroupComplete(enum flagcxWorkGroupDomain domain,
                                       flagcxResult_t result) {
  struct flagcxWorkGroupBoundary boundary = {};
  boundary.ready = domain != flagcxWorkNativeGroup || !nativeSubmissionPending;
  flagcxResult_t completion = flagcxWorkGroupCompleteAtBoundary(
      domain, boundary.ready ? result : flagcxInProgress, &boundary);
  return result != flagcxSuccess ? result : completion;
}
void flagcxWorkGroupStart(enum flagcxWorkGroupDomain domain) {
  if (domain == flagcxWorkNativeGroup && workGroupDepth[domain] == 0)
    nativeSubmissionPending = 0;
  ++workGroupDepth[domain];
}
int flagcxWorkGroupDepth(enum flagcxWorkGroupDomain domain) {
  return workGroupDepth[domain];
}
flagcxResult_t flagcxWorkGroupEnd(enum flagcxWorkGroupDomain domain,
                                  flagcxResult_t result) {
  if (workGroupDepth[domain] == 0)
    return flagcxInvalidUsage;
  if (--workGroupDepth[domain] == 0)
    return flagcxWorkGroupComplete(domain, result);
  return result;
}
void flagcxWorkRetain(struct flagcxWorkRecord *record) {
  pthread_mutex_lock(&record->owner->mutex);
  ++record->pins;
  pthread_mutex_unlock(&record->owner->mutex);
}
void flagcxWorkDrop(struct flagcxWorkRecord *record) {
  pthread_mutex_lock(&record->owner->mutex);
  --record->pins;
  pthread_mutex_unlock(&record->owner->mutex);
}
static flagcxResult_t workWaitRecord(struct flagcxWorkRecord *record,
                                     int wait) {
  if (record == NULL)
    return flagcxInvalidArgument;
  struct flagcxWorkTracker *tracker = record->owner;
  pthread_mutex_lock(&tracker->mutex);
  if (record->state == flagcxWorkUncertain ||
      record->state == flagcxWorkComplete) {
    flagcxResult_t result = record->state == flagcxWorkComplete
                                ? flagcxSuccess
                                : record->completionResult;
    pthread_mutex_unlock(&tracker->mutex);
    return result;
  }
  if (record->state != flagcxWorkRecorded &&
      record->state != flagcxWorkNativePending) {
    pthread_mutex_unlock(&tracker->mutex);
    return flagcxInProgress;
  }
  record->state = flagcxWorkChecking;
  pthread_mutex_unlock(&tracker->mutex);
  bool complete = false;
  // Query pending events rather than entering an uninterruptible runtime
  // synchronization while their proxy may have failed.
  flagcxResult_t result = workCheckCompletion(record, wait, &complete);
  pthread_mutex_lock(&tracker->mutex);
  record->state = complete
                      ? flagcxWorkComplete
                      : (record->nativeRequest != NULL ? flagcxWorkNativePending
                                                       : flagcxWorkRecorded);
  pthread_mutex_unlock(&tracker->mutex);
  return result;
}

flagcxResult_t flagcxWorkWaitRecord(struct flagcxWorkRecord *record) {
  return workWaitRecord(record, 1);
}
flagcxResult_t flagcxWorkPollRecord(struct flagcxWorkRecord *record) {
  return workWaitRecord(record, 0);
}

static flagcxResult_t workWait(struct flagcxWorkTracker *tracker, int close,
                               struct flagcxWorkSnapshot *snapshot = NULL) {
  if (tracker == NULL)
    return flagcxInvalidArgument;
  flagcxResult_t result = workInit(tracker);
  if (result != flagcxSuccess)
    return result;
  pthread_mutex_lock(&tracker->mutex);
  if (close && !tracker->closed) {
    for (struct flagcxWorkRecord *r = tracker->head; r != NULL; r = r->next) {
      if (r->state == flagcxWorkGroupPending) {
        pthread_mutex_unlock(&tracker->mutex);
        return flagcxInvalidUsage;
      }
    }
    __atomic_store_n(&tracker->closed, 1, __ATOMIC_RELEASE);
  }
  uint64_t target = tracker->nextId;
  if (snapshot != NULL) {
    if (!snapshot->initialized) {
      snapshot->target = target;
      snapshot->initialized = 1;
    }
    target = snapshot->target;
  }
  if (close && tracker->submitting != 0) {
    pthread_mutex_unlock(&tracker->mutex);
    return flagcxInProgress;
  }
  for (;;) {
    struct flagcxWorkRecord *record = tracker->head;
    while (record != NULL) {
      bool caller = false;
      if (!close && record->state == flagcxWorkPreparing)
        for (auto *parent = currentWork; parent != NULL;
             parent = parent->parent)
          if (record == parent)
            caller = true;
      if (record->id <= target && record->state != flagcxWorkComplete &&
          !caller)
        break;
      record = record->next;
    }
    if (record == NULL)
      break;
    if (record->state == flagcxWorkGroupPending) {
      pthread_mutex_unlock(&tracker->mutex);
      return flagcxInvalidUsage;
    }
    ++record->pins;
    pthread_mutex_unlock(&tracker->mutex);
    result = flagcxWorkWaitRecord(record);
    pthread_mutex_lock(&tracker->mutex);
    --record->pins;
    if (result != flagcxSuccess) {
      pthread_mutex_unlock(&tracker->mutex);
      return result;
    }
  }
  pthread_mutex_unlock(&tracker->mutex);
  return flagcxSuccess;
}
flagcxResult_t flagcxWorkWaitSnapshot(struct flagcxWorkTracker *tracker,
                                      struct flagcxWorkSnapshot *snapshot) {
  if (tracker != NULL &&
      __atomic_load_n(&tracker->initialized, __ATOMIC_ACQUIRE) == 0) {
    if (snapshot != NULL && !snapshot->initialized) {
      snapshot->target = 0;
      snapshot->initialized = 1;
    }
    return flagcxSuccess;
  }
  return workWait(tracker, 0, snapshot);
}
flagcxResult_t flagcxWorkWait(struct flagcxWorkTracker *tracker) {
  return workWait(tracker, 1);
}
static flagcxResult_t workProgressError(struct flagcxHeteroComm *comm) {
  if (comm == NULL)
    return flagcxSuccess;
  if (comm->proxyState != NULL) {
    flagcxResult_t result =
        __atomic_load_n(&comm->proxyState->asyncResult, __ATOMIC_ACQUIRE);
    if (result != flagcxSuccess && result != flagcxInProgress)
      return result;
    result = __atomic_load_n(&comm->proxyState->kernelState.terminalResult,
                             __ATOMIC_ACQUIRE);
    if (result != flagcxSuccess && result != flagcxInProgress)
      return result;
    result = __atomic_load_n(&comm->proxyState->workerCleanupResult,
                             __ATOMIC_ACQUIRE);
    if (result != flagcxSuccess && result != flagcxInProgress)
      return result;
  }
  if (comm->rmaProxy != NULL &&
      (__atomic_load_n(&comm->rmaProxy->pendingError, __ATOMIC_ACQUIRE) ||
       __atomic_load_n(&comm->rmaProxy->rmaError, __ATOMIC_ACQUIRE)))
    return flagcxRemoteError;
  return flagcxSuccess;
}
flagcxResult_t flagcxWorkDrainRecord(struct flagcxWorkRecord *record,
                                     struct flagcxHeteroComm *comm) {
  for (;;) {
    flagcxResult_t result = flagcxWorkWaitRecord(record);
    if (result != flagcxInProgress)
      return result;
    pthread_mutex_lock(&record->owner->mutex);
    bool uncertain = record->state == flagcxWorkUncertain;
    pthread_mutex_unlock(&record->owner->mutex);
    if (uncertain)
      return result;
    result = workProgressError(comm);
    if (result != flagcxSuccess)
      return result;
    sched_yield();
  }
}
flagcxResult_t flagcxWorkDrain(struct flagcxWorkTracker *tracker,
                               struct flagcxHeteroComm *comm) {
  for (;;) {
    flagcxResult_t result = flagcxWorkWait(tracker);
    if (result != flagcxInProgress)
      return result;
    pthread_mutex_lock(&tracker->mutex);
    bool uncertain = false;
    for (auto *record = tracker->head; record != NULL; record = record->next)
      uncertain |= record->state == flagcxWorkUncertain;
    pthread_mutex_unlock(&tracker->mutex);
    if (uncertain)
      return result;
    result = workProgressError(comm);
    if (result != flagcxSuccess)
      return result;
    sched_yield();
  }
}
flagcxResult_t flagcxCommDrainSnapshot(flagcxComm_t comm,
                                       struct flagcxWorkSnapshot *snapshot) {
  if (comm == NULL || snapshot == NULL)
    return flagcxInvalidArgument;
  auto *hetero = comm->heteroComm;
  auto *proxy = hetero != NULL ? hetero->rmaProxy : NULL;
  if (!snapshot->rmaCaptured) {
    if (proxy != NULL && proxy->nRanks > 0 && proxy->opSeqs != NULL) {
      snapshot->rmaTargets =
          (uint64_t *)calloc(proxy->nRanks, sizeof(uint64_t));
      if (snapshot->rmaTargets == NULL)
        return flagcxSystemError;
      snapshot->rmaCount = proxy->nRanks;
      for (int p = 0; p < proxy->nRanks; ++p)
        snapshot->rmaTargets[p] =
            __atomic_load_n(&proxy->opSeqs[p], __ATOMIC_ACQUIRE);
    }
    snapshot->rmaCaptured = 1;
  }
  if (!snapshot->fifoCaptured) {
    auto *state = hetero != NULL ? hetero->proxyState : NULL;
    int count = state != NULL ? state->kernelState.contextCount : 0;
    if (count != 0) {
      snapshot->fifoTargets =
          (flagcxWorkFifoTarget *)calloc(count, sizeof(flagcxWorkFifoTarget));
      if (snapshot->fifoTargets == NULL)
        return flagcxSystemError;
      snapshot->fifoCount = count;
      for (int i = 0; i < count; ++i) {
        auto *fifo = state->kernelState.fifos[i];
        if (fifo == NULL || fifo->buffer == NULL) {
          snapshot->fifoTargets[i].complete = 1;
          continue;
        }
        snapshot->fifoTargets[i].target = __atomic_load_n(
            flagcxFifoControlPtr(fifo->buffer, flagcxFifoIdxProduced),
            __ATOMIC_ACQUIRE);
      }
    }
    snapshot->fifoCaptured = 1;
  }
  for (;;) {
    flagcxResult_t result =
        flagcxWorkWaitSnapshot(&comm->submittedWork, snapshot);
    if (result == flagcxSuccess)
      break;
    if (result != flagcxInProgress)
      return result;
    result = workProgressError(hetero);
    if (result != flagcxSuccess)
      return result;
    // A native InProgress with no established submission boundary cannot be
    // repaired by recording the current user-stream tail.
    pthread_mutex_lock(&comm->submittedWork.mutex);
    bool uncertain = false;
    for (auto *record = comm->submittedWork.head; record != NULL;
         record = record->next)
      uncertain |= record->id <= snapshot->target &&
                   record->state == flagcxWorkUncertain;
    pthread_mutex_unlock(&comm->submittedWork.mutex);
    if (uncertain)
      return flagcxInProgress;
    sched_yield();
  }
  // Device producers were completed by the caller. Retire the fixed accepted
  // FIFO prefix before changing the active DevComm or releasing any view.
  // Keep the shared FIFO open so independent surviving resources can be used.
  for (int i = 0; i < snapshot->fifoCount; ++i) {
    auto *target = &snapshot->fifoTargets[i];
    if (target->complete)
      continue;
    auto *state = hetero != NULL ? hetero->proxyState : NULL;
    auto *fifo = state != NULL ? state->kernelState.fifos[i] : NULL;
    if (fifo == NULL || fifo->buffer == NULL)
      return flagcxInternalError;
    for (;;) {
      flagcxCompletionWord_t completed = __atomic_load_n(
          flagcxFifoControlPtr(fifo->buffer, flagcxFifoIdxCompleted),
          __ATOMIC_ACQUIRE);
      flagcxCompletionWord_t distance =
          completed - (flagcxCompletionWord_t)target->target;
      // At most FIFO capacity operations are outstanding, below half the word
      // range. This accepts an advanced prefix and handles 32/64-bit wrap.
      if (distance <=
          (flagcxCompletionWord_t)(~(flagcxCompletionWord_t)0 >> 1)) {
        target->complete = 1;
        break;
      }
      flagcxResult_t result = workProgressError(hetero);
      if (result != flagcxSuccess)
        return result;
      sched_yield();
    }
  }
  if (proxy != NULL) {
    for (int p = 0; p < snapshot->rmaCount; ++p) {
      flagcxResult_t result =
          flagcxHeteroFlushRma(hetero, p, snapshot->rmaTargets[p]);
      if (result == flagcxRemoteError)
        result = flagcxHeteroRmaProxyQuiesce(hetero);
      if (result != flagcxSuccess)
        return result;
    }
  }
  return flagcxSuccess;
}
void flagcxWorkSnapshotRelease(struct flagcxWorkSnapshot *snapshot) {
  free(snapshot->rmaTargets);
  free(snapshot->fifoTargets);
  *snapshot = {};
}

flagcxResult_t flagcxWorkRelease(struct flagcxWorkTracker *tracker) {
  if (tracker == NULL)
    return flagcxInvalidArgument;
  int initialized = __atomic_load_n(&tracker->initialized, __ATOMIC_ACQUIRE);
  if (initialized == 0)
    return flagcxSuccess;
  if (initialized != 2)
    return flagcxInProgress;
  if (!__atomic_load_n(&tracker->closed, __ATOMIC_ACQUIRE))
    return flagcxInvalidUsage;
  struct flagcxWorkRecord **lists[] = {&tracker->head, &tracker->spare,
                                       &tracker->retired};
  for (unsigned int i = 0; i < sizeof(lists) / sizeof(lists[0]); ++i) {
    while (*lists[i] != NULL) {
      struct flagcxWorkRecord *record = *lists[i];
      if (record->state != flagcxWorkComplete || record->pins != 0)
        return flagcxInvalidUsage;
      flagcxResult_t bufferResult = flagcxWorkReleaseBuffer(record);
      if (bufferResult != flagcxSuccess)
        return bufferResult;
      if (record->event != NULL) {
        flagcxResult_t result = deviceAdaptor->eventDestroy(record->event);
        if (result != flagcxSuccess)
          return result;
        record->event = NULL;
      }
      *lists[i] = record->next;
      free(record);
    }
  }
  tracker->tail = NULL;
  tracker->spareCount = 0;
  if (pthread_mutex_destroy(&tracker->mutex) != 0)
    return flagcxSystemError;
  __atomic_store_n(&tracker->initialized, 0, __ATOMIC_RELEASE);
  return flagcxSuccess;
}
