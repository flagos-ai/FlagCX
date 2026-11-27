#ifndef FLAGCX_NCCL_COMM_COMPLETION_H_
#define FLAGCX_NCCL_COMM_COMPLETION_H_

#include "flagcx.h"
#include <limits.h>
#include <pthread.h>
#include <stdlib.h>

// This CUDA plugin implements NCCL's asynchronous API contract independently
// of the core caller-completion contract. Wrappers and events live with the
// plugin Comm, including across failed lifecycle attempts.
struct flagcxNcclStream {
  struct flagcxNcclStream *next;
  void *native;
  flagcxStream_t wrapper;
  flagcxEvent_t done;
  bool pending, complete;
};

struct flagcxNcclCompletion {
  pthread_mutex_t mutex;
  struct flagcxNcclStream *streams;
  flagcxComm_t comm;
  flagcxDeviceHandle_t device;
  int ordinal;
  bool ordinalValid, closing, nativePending;
};

struct flagcxNcclGroupedComm {
  struct flagcxNcclGroupedComm *next;
  struct flagcxNcclCompletion *completion;
};

static __thread int flagcxNcclGroupDepth;
static __thread struct flagcxNcclGroupedComm *flagcxNcclGroupedComms;

static flagcxResult_t
flagcxNcclSelectDevice(struct flagcxNcclCompletion *state) {
  flagcxResult_t result;
  if (!state->ordinalValid) {
    result = flagcxCommGetDeviceNumber(state->comm, &state->ordinal);
    if (result != flagcxSuccess)
      return result;
    state->ordinalValid = true;
  }
  return state->device->setDevice(state->ordinal);
}

// Caller holds the Comm mutex. Record only after outer group/native submission.
static flagcxResult_t
flagcxNcclRecordPending(struct flagcxNcclCompletion *state) {
  if (state->streams == NULL)
    return flagcxSuccess;
  flagcxResult_t result = flagcxNcclSelectDevice(state);
  if (result != flagcxSuccess)
    return result;
  if (state->nativePending) {
    flagcxResult_t async = flagcxInProgress;
    result = flagcxCommGetAsyncError(state->comm, &async);
    if (result != flagcxSuccess)
      return result;
    if (async != flagcxSuccess)
      return async;
    state->nativePending = false;
  }
  for (struct flagcxNcclStream *entry = state->streams; entry != NULL;
       entry = entry->next) {
    if (!entry->pending)
      continue;
    result = state->device->eventRecord(entry->done, entry->wrapper);
    if (result != flagcxSuccess)
      return result;
    entry->pending = false;
  }
  return flagcxSuccess;
}

// Success leaves the mutex locked across submission and completion recording.
static flagcxResult_t flagcxNcclBegin(struct flagcxNcclCompletion *state,
                                      flagcxComm_t comm,
                                      flagcxDeviceHandle_t device, void *native,
                                      struct flagcxNcclStream **output) {
  flagcxResult_t result = flagcxSuccess;
  struct flagcxNcclStream *entry = NULL;
  struct flagcxNcclGroupedComm *group = NULL;
  if (pthread_mutex_lock(&state->mutex) != 0)
    return flagcxSystemError;
  state->comm = comm;
  state->device = device;
  if (state->closing) {
    result = flagcxInvalidUsage;
    goto fail;
  }
  result = flagcxNcclSelectDevice(state);
  if (result != flagcxSuccess)
    goto fail;
  if (state->nativePending) {
    result = flagcxNcclRecordPending(state);
    if (result != flagcxSuccess)
      goto fail;
  }
  if (flagcxNcclGroupDepth != 0) {
    for (group = flagcxNcclGroupedComms; group != NULL; group = group->next)
      if (group->completion == state)
        break;
    if (group == NULL) {
      group = (struct flagcxNcclGroupedComm *)calloc(1, sizeof(*group));
      if (group == NULL) {
        result = flagcxSystemError;
        goto fail;
      }
      group->completion = state;
      group->next = flagcxNcclGroupedComms;
      flagcxNcclGroupedComms = group;
    }
  }
  for (entry = state->streams; entry != NULL; entry = entry->next)
    if (entry->native == native)
      break;
  if (entry == NULL) {
    entry = (struct flagcxNcclStream *)calloc(1, sizeof(*entry));
    if (entry == NULL) {
      result = flagcxSystemError;
      goto fail;
    }
    entry->native = native;
    entry->complete = true;
    entry->next = state->streams;
    state->streams = entry;
  }
  if (entry->wrapper == NULL) {
    result = device->streamCopy(&entry->wrapper, native);
    if (result != flagcxSuccess)
      goto fail;
  }
  if (entry->done == NULL) {
    result = device->eventCreate(&entry->done, flagcxEventDisableTiming);
    if (result != flagcxSuccess)
      goto fail;
  }
  entry->pending = true;
  entry->complete = false;
  *output = entry;
  return flagcxSuccess;
fail:
  pthread_mutex_unlock(&state->mutex);
  return result;
}

static flagcxResult_t flagcxNcclEnd(struct flagcxNcclCompletion *state,
                                    flagcxResult_t submitted) {
  flagcxResult_t result = submitted;
  if (flagcxNcclGroupDepth == 0) {
    state->nativePending = submitted == flagcxInProgress;
    if (submitted == flagcxSuccess)
      result = flagcxNcclRecordPending(state);
  }
  pthread_mutex_unlock(&state->mutex);
  return result;
}

static flagcxResult_t flagcxNcclGroupComplete(flagcxResult_t submitted) {
  flagcxResult_t result = submitted;
  while (flagcxNcclGroupedComms != NULL) {
    struct flagcxNcclGroupedComm *group = flagcxNcclGroupedComms;
    struct flagcxNcclCompletion *state = group->completion;
    if (pthread_mutex_lock(&state->mutex) != 0)
      return flagcxSystemError;
    state->nativePending = submitted == flagcxInProgress;
    if (submitted == flagcxSuccess) {
      flagcxResult_t recorded = flagcxNcclRecordPending(state);
      if ((result == flagcxSuccess || result == flagcxInProgress) &&
          recorded != flagcxSuccess)
        result = recorded;
    }
    pthread_mutex_unlock(&state->mutex);
    flagcxNcclGroupedComms = group->next;
    free(group);
  }
  return result;
}

static flagcxResult_t flagcxNcclDrain(struct flagcxNcclCompletion *state) {
  flagcxResult_t result = flagcxSuccess;
  if (flagcxNcclGroupDepth != 0)
    return flagcxInvalidUsage;
  if (pthread_mutex_lock(&state->mutex) != 0)
    return flagcxSystemError;
  state->closing = true;
  result = flagcxNcclRecordPending(state);
  if (result != flagcxSuccess)
    goto exit;
  for (struct flagcxNcclStream *entry = state->streams; entry != NULL;
       entry = entry->next) {
    if (entry->complete)
      continue;
    result = state->device->eventSynchronize(entry->done);
    if (result != flagcxSuccess)
      goto exit;
    entry->complete = true;
  }
exit:
  pthread_mutex_unlock(&state->mutex);
  return result;
}

// Call only after core Destroy/Abort succeeds; preserve release progress.
static flagcxResult_t flagcxNcclRelease(struct flagcxNcclCompletion *state) {
  if (state->streams != NULL) {
    flagcxResult_t result = flagcxNcclSelectDevice(state);
    if (result != flagcxSuccess)
      return result;
  }
  while (state->streams != NULL) {
    struct flagcxNcclStream *entry = state->streams;
    if (entry->done != NULL) {
      flagcxResult_t result = state->device->eventDestroy(entry->done);
      if (result != flagcxSuccess)
        return result;
      entry->done = NULL;
    }
    if (entry->wrapper != NULL) {
      flagcxResult_t result = state->device->streamFree(entry->wrapper);
      if (result != flagcxSuccess)
        return result;
      entry->wrapper = NULL;
    }
    state->streams = entry->next;
    free(entry);
  }
  return flagcxSuccess;
}

#endif
