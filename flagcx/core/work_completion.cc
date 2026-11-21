#include "work_completion.h"
#include "adaptor.h"
#include <sched.h>
#include <stdlib.h>

static __thread struct flagcxWorkRecord *pendingWork[2];

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

// Reuse completed events without waiting for device work on the submit path.
// Keep the checking record linked so concurrent Finalize cannot overlook it.
static void workCollect(struct flagcxWorkTracker *tracker, int allowQuery) {
  for (int budget = 0; budget < 8; budget++) {
    pthread_mutex_lock(&tracker->mutex);
    struct flagcxWorkRecord *record = tracker->head;
    if (tracker->closed || record == NULL || record->pins != 0 ||
        (record->state != flagcxWorkRecorded &&
         record->state != flagcxWorkComplete &&
         record->state != flagcxWorkExternalGraph) ||
        (!allowQuery && record->state == flagcxWorkRecorded)) {
      pthread_mutex_unlock(&tracker->mutex);
      return;
    }
    int complete = record->state == flagcxWorkComplete ||
                   record->state == flagcxWorkExternalGraph;
    record->state = flagcxWorkChecking;
    pthread_mutex_unlock(&tracker->mutex);
    flagcxResult_t result =
        complete ? flagcxSuccess : deviceAdaptor->eventQuery(record->event);
    pthread_mutex_lock(&tracker->mutex);
    if (result == flagcxSuccess) {
      tracker->head = record->next;
      if (tracker->head == NULL)
        tracker->tail = NULL;
      record->state = flagcxWorkComplete;
      struct flagcxWorkRecord **list =
          record->reusable ? &tracker->spare : &tracker->retired;
      record->next = *list;
      *list = record;
    } else {
      record->state = flagcxWorkRecorded;
    }
    pthread_mutex_unlock(&tracker->mutex);
    if (result != flagcxSuccess)
      return;
  }
}

flagcxResult_t flagcxWorkPrepare(struct flagcxWorkTracker *tracker,
                                 flagcxStream_t stream, int allowCapture,
                                 struct flagcxWorkRecord **output) {
  if (tracker == NULL || output == NULL)
    return flagcxInvalidArgument;
  *output = NULL;
  if (__atomic_load_n(&tracker->closed, __ATOMIC_ACQUIRE))
    return flagcxInvalidUsage;
  if (deviceAdaptor->streamGetCaptureStatus == NULL)
    return flagcxNotSupported;
  flagcxStreamCaptureStatus_t captureStatus = flagcxStreamCaptureInvalidated;
  flagcxResult_t captureResult =
      deviceAdaptor->streamGetCaptureStatus(stream, &captureStatus);
  if (captureResult != flagcxSuccess)
    return captureResult;
  if (captureStatus == flagcxStreamCaptureInvalidated)
    return flagcxInvalidUsage;
  if (captureStatus != flagcxStreamCaptureNone &&
      captureStatus != flagcxStreamCaptureActive)
    return flagcxInternalError;
  if (captureStatus == flagcxStreamCaptureActive && !allowCapture)
    return flagcxNotSupported;
  FLAGCXCHECK(workInit(tracker));
  workCollect(tracker, captureStatus == flagcxStreamCaptureNone);
  pthread_mutex_lock(&tracker->mutex);
  if (tracker->closed) {
    pthread_mutex_unlock(&tracker->mutex);
    return flagcxInvalidUsage;
  }
  struct flagcxWorkRecord *record = tracker->spare;
  if (record != NULL)
    tracker->spare = record->next;
  else
    record = (struct flagcxWorkRecord *)calloc(1, sizeof(*record));
  if (record == NULL) {
    pthread_mutex_unlock(&tracker->mutex);
    return flagcxSystemError;
  }
  record->next = NULL;
  record->groupNext = NULL;
  record->owner = tracker;
  record->state = flagcxWorkPreparing;
  record->error = flagcxSuccess;
  record->captureStatus = captureStatus;
  if (tracker->tail != NULL)
    tracker->tail->next = record;
  else
    tracker->head = record;
  tracker->tail = record;
  pthread_mutex_unlock(&tracker->mutex);

  flagcxResult_t result = flagcxSuccess;
  if (captureStatus == flagcxStreamCaptureNone && record->event == NULL) {
    if (deviceAdaptor->eventCreate == NULL ||
        deviceAdaptor->eventRecord == NULL ||
        deviceAdaptor->eventSynchronize == NULL ||
        deviceAdaptor->eventQuery == NULL ||
        deviceAdaptor->eventDestroy == NULL)
      result = flagcxNotSupported;
    else
      result =
          deviceAdaptor->eventCreate(&record->event, flagcxEventDisableTiming);
  }
  pthread_mutex_lock(&tracker->mutex);
  if (result != flagcxSuccess ||
      (captureStatus == flagcxStreamCaptureNone && record->event == NULL)) {
    record->state = flagcxWorkComplete; // No submission took place.
    record->reusable = 0;
    pthread_mutex_unlock(&tracker->mutex);
    return result != flagcxSuccess ? result : flagcxInternalError;
  }
  record->reusable = 1;
  pthread_mutex_unlock(&tracker->mutex);
  *output = record;
  return flagcxSuccess;
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
  record->error = flagcxSuccess;
  record->reusable = 0;
  record->state = flagcxWorkComplete;
  pthread_mutex_unlock(&record->owner->mutex);
  return flagcxSuccess;
}

static flagcxResult_t
workRecord(struct flagcxWorkRecord *record, flagcxResult_t submission,
           const struct flagcxWorkGroupBoundary *boundary) {
  // A terminal operation error does not invalidate the submitted prefix's
  // completion fence. Pending submission can still enqueue work after this
  // call, so it cannot establish a completion boundary yet.
  flagcxResult_t result =
      submission == flagcxInProgress ? flagcxInProgress : flagcxSuccess;
  flagcxStream_t completionStream = record->stream;
  if (boundary != NULL && boundary->submitted) {
    if (!boundary->ready)
      result = submission != flagcxSuccess ? submission : flagcxInternalError;
    else
      completionStream = boundary->stream;
  }
  flagcxStreamCaptureStatus_t captureStatus = flagcxStreamCaptureInvalidated;
  if (result == flagcxSuccess) {
    result =
        deviceAdaptor->streamGetCaptureStatus(record->stream, &captureStatus);
    if (result == flagcxSuccess && captureStatus != record->captureStatus)
      result = flagcxInvalidUsage;
  }
  if (result == flagcxSuccess && completionStream != record->stream) {
    flagcxStreamCaptureStatus_t completionCapture =
        flagcxStreamCaptureInvalidated;
    result = deviceAdaptor->streamGetCaptureStatus(completionStream,
                                                   &completionCapture);
    if (result == flagcxSuccess &&
        (captureStatus != flagcxStreamCaptureNone ||
         completionCapture != flagcxStreamCaptureNone))
      result = flagcxInvalidUsage;
  }
  if (result == flagcxSuccess && captureStatus == flagcxStreamCaptureNone)
    result = deviceAdaptor->eventRecord(record->event, completionStream);
  pthread_mutex_lock(&record->owner->mutex);
  record->state = result != flagcxSuccess ? flagcxWorkUncertain
                  : captureStatus == flagcxStreamCaptureActive
                      ? flagcxWorkExternalGraph
                      : flagcxWorkRecorded;
  record->error = result;
  record->stream = NULL;
  pthread_mutex_unlock(&record->owner->mutex);
  return result;
}

flagcxResult_t flagcxWorkFinish(struct flagcxWorkRecord *record,
                                flagcxStream_t stream, flagcxResult_t result,
                                enum flagcxWorkGroupDomain domain,
                                int deferred) {
  if (record == NULL)
    return flagcxInvalidArgument;
  record->stream = stream;
  if (deferred) {
    pthread_mutex_lock(&record->owner->mutex);
    record->state = flagcxWorkGroupPending;
    pthread_mutex_unlock(&record->owner->mutex);
    record->groupNext = pendingWork[domain];
    pendingWork[domain] = record;
    return result;
  }
  // A failed call may still have queued a successful prefix. Fence it at the
  // original boundary rather than discarding its ownership record.
  flagcxResult_t recorded = workRecord(record, result, NULL);
  return result != flagcxSuccess ? result : recorded;
}

flagcxResult_t flagcxWorkGroupComplete(enum flagcxWorkGroupDomain domain,
                                       flagcxResult_t result) {
  return flagcxWorkGroupCompleteAtBoundary(domain, result, NULL);
}

flagcxResult_t flagcxWorkGroupCompleteAtBoundary(
    enum flagcxWorkGroupDomain domain, flagcxResult_t result,
    const struct flagcxWorkGroupBoundary *boundary) {
  struct flagcxWorkRecord *records = pendingWork[domain];
  pendingWork[domain] = NULL;
  flagcxResult_t first = result;
  while (records != NULL) {
    struct flagcxWorkRecord *record = records;
    records = record->groupNext;
    record->groupNext = NULL;
    flagcxResult_t recorded = workRecord(record, result, boundary);
    if (first == flagcxSuccess)
      first = recorded;
  }
  return first;
}

void flagcxWorkRetain(struct flagcxWorkRecord *record) {
  pthread_mutex_lock(&record->owner->mutex);
  record->pins++;
  pthread_mutex_unlock(&record->owner->mutex);
}
void flagcxWorkDrop(struct flagcxWorkRecord *record) {
  pthread_mutex_lock(&record->owner->mutex);
  record->pins--;
  pthread_mutex_unlock(&record->owner->mutex);
}
// Callers outside closed admission retain the record across the wait so it
// cannot be recycled by another submission after its event completes.
flagcxResult_t flagcxWorkWaitRecord(struct flagcxWorkRecord *record) {
  if (record == NULL)
    return flagcxInvalidArgument;
  struct flagcxWorkTracker *tracker = record->owner;
  pthread_mutex_lock(&tracker->mutex);
  if (record->state == flagcxWorkUncertain ||
      record->state == flagcxWorkComplete ||
      record->state == flagcxWorkExternalGraph) {
    flagcxResult_t result =
        record->state == flagcxWorkUncertain ? record->error : flagcxSuccess;
    pthread_mutex_unlock(&tracker->mutex);
    return result;
  }
  if (record->state != flagcxWorkRecorded) {
    pthread_mutex_unlock(&tracker->mutex);
    return flagcxInProgress;
  }
  record->state = flagcxWorkChecking;
  pthread_mutex_unlock(&tracker->mutex);
  flagcxResult_t result = deviceAdaptor->eventSynchronize(record->event);
  pthread_mutex_lock(&tracker->mutex);
  record->state =
      result == flagcxSuccess ? flagcxWorkComplete : flagcxWorkRecorded;
  pthread_mutex_unlock(&tracker->mutex);
  return result;
}

flagcxResult_t flagcxWorkWait(struct flagcxWorkTracker *tracker) {
  if (tracker == NULL)
    return flagcxInvalidArgument;
  if (__atomic_load_n(&tracker->initialized, __ATOMIC_ACQUIRE) == 0 &&
      __atomic_load_n(&tracker->closed, __ATOMIC_ACQUIRE))
    return flagcxSuccess;
  FLAGCXCHECK(workInit(tracker));
  pthread_mutex_lock(&tracker->mutex);
  __atomic_store_n(&tracker->closed, 1, __ATOMIC_RELEASE);
  struct flagcxWorkRecord *record = tracker->head;
  while (record != NULL) {
    if (record->state == flagcxWorkUncertain) {
      flagcxResult_t result = record->error;
      pthread_mutex_unlock(&tracker->mutex);
      return result;
    }
    if (record->state == flagcxWorkComplete ||
        record->state == flagcxWorkExternalGraph) {
      // Graph execution is caller-owned: Finish captured a definition, not a
      // completed execution. Finalize requires the caller to quiesce replays.
      record = record->next;
      continue;
    }
    if (record->state != flagcxWorkRecorded) {
      pthread_mutex_unlock(&tracker->mutex);
      return flagcxInProgress;
    }
    record->state = flagcxWorkChecking;
    pthread_mutex_unlock(&tracker->mutex);
    flagcxResult_t result = deviceAdaptor->eventSynchronize(record->event);
    pthread_mutex_lock(&tracker->mutex);
    record->state =
        result == flagcxSuccess ? flagcxWorkComplete : flagcxWorkRecorded;
    if (result != flagcxSuccess) {
      pthread_mutex_unlock(&tracker->mutex);
      return result;
    }
    record = record->next;
  }
  pthread_mutex_unlock(&tracker->mutex);
  return flagcxSuccess;
}

flagcxResult_t flagcxWorkRelease(struct flagcxWorkTracker *tracker) {
  if (tracker == NULL)
    return flagcxInvalidArgument;
  int initialized = __atomic_load_n(&tracker->initialized, __ATOMIC_ACQUIRE);
  if (initialized == 0)
    return flagcxSuccess;
  if (initialized != 2)
    return flagcxInProgress;
  // Finalize closed admission and completed all records; only its local
  // Destroy owner can release this list. Failed releases keep their fields.
  struct flagcxWorkRecord **lists[] = {&tracker->head, &tracker->spare,
                                       &tracker->retired};
  for (unsigned i = 0; i < sizeof(lists) / sizeof(lists[0]); i++) {
    while (*lists[i] != NULL) {
      struct flagcxWorkRecord *record = *lists[i];
      if ((record->state != flagcxWorkComplete &&
           record->state != flagcxWorkExternalGraph) ||
          record->pins != 0)
        return flagcxInvalidUsage;
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
  if (pthread_mutex_destroy(&tracker->mutex) != 0)
    return flagcxSystemError;
  __atomic_store_n(&tracker->initialized, 0, __ATOMIC_RELEASE);
  return flagcxSuccess;
}
