/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "net_transport.h"

#include "onesided_types.h"

#include <limits.h>

flagcxResult_t flagcxNetSelectLane(struct flagcxNetLaneSet *lanes,
                                   enum flagcxNetLaneMode mode,
                                   uint64_t orderingKey, uint32_t *laneIndex) {
  if (lanes == NULL || laneIndex == NULL || lanes->count == 0)
    return flagcxInvalidArgument;

  switch (mode) {
    case FLAGCX_NET_LANE_ORDERED:
      *laneIndex = (uint32_t)(orderingKey % lanes->count);
      return flagcxSuccess;
    case FLAGCX_NET_LANE_UNORDERED:
      *laneIndex = lanes->unorderedCursor % lanes->count;
      return flagcxSuccess;
    default:
      return flagcxInvalidArgument;
  }
}

flagcxResult_t flagcxNetCommitLane(struct flagcxNetLaneSet *lanes,
                                   enum flagcxNetLaneMode mode,
                                   uint32_t laneIndex) {
  if (lanes == NULL || lanes->count == 0 || laneIndex >= lanes->count)
    return flagcxInvalidArgument;
  if (mode == FLAGCX_NET_LANE_ORDERED)
    return flagcxSuccess;
  if (mode != FLAGCX_NET_LANE_UNORDERED)
    return flagcxInvalidArgument;

  const uint32_t current = lanes->unorderedCursor % lanes->count;
  if (laneIndex != current)
    return flagcxInvalidArgument;
  lanes->unorderedCursor = (current + 1) % lanes->count;
  return flagcxSuccess;
}

flagcxResult_t flagcxNetCreditInit(struct flagcxNetCredit *credit,
                                   uint32_t capacity) {
  if (credit == NULL || capacity == 0)
    return flagcxInvalidArgument;
  credit->capacity = capacity;
  __atomic_store_n(&credit->inUse, 0, __ATOMIC_RELEASE);
  return flagcxSuccess;
}

flagcxResult_t flagcxNetCreditAcquire(struct flagcxNetCredit *credit,
                                      uint32_t count) {
  if (credit == NULL || count == 0 || credit->capacity == 0 ||
      count > credit->capacity)
    return flagcxInvalidArgument;

  uint32_t inUse = __atomic_load_n(&credit->inUse, __ATOMIC_ACQUIRE);
  while (true) {
    if (inUse > credit->capacity || count > credit->capacity - inUse)
      return flagcxInProgress;
    if (__atomic_compare_exchange_n(&credit->inUse, &inUse, inUse + count,
                                    false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
      return flagcxSuccess;
  }
}

flagcxResult_t flagcxNetCreditRelease(struct flagcxNetCredit *credit,
                                      uint32_t count) {
  if (credit == NULL || count == 0 || credit->capacity == 0)
    return flagcxInvalidArgument;

  uint32_t inUse = __atomic_load_n(&credit->inUse, __ATOMIC_ACQUIRE);
  while (true) {
    if (count > inUse)
      return flagcxInvalidArgument;
    if (__atomic_compare_exchange_n(&credit->inUse, &inUse, inUse - count,
                                    false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
      return flagcxSuccess;
  }
}

flagcxResult_t flagcxNetCreditAvailable(const struct flagcxNetCredit *credit,
                                        uint32_t *available) {
  if (credit == NULL || available == NULL || credit->capacity == 0)
    return flagcxInvalidArgument;
  const uint32_t inUse = __atomic_load_n(&credit->inUse, __ATOMIC_ACQUIRE);
  if (inUse > credit->capacity)
    return flagcxInternalError;
  *available = credit->capacity - inUse;
  return flagcxSuccess;
}

void flagcxNetRequestCoreInit(struct flagcxNetRequestCore *core) {
  if (core == NULL)
    return;
  __atomic_store_n(&core->pending, 0, __ATOMIC_RELAXED);
  __atomic_store_n(&core->result, flagcxSuccess, __ATOMIC_RELAXED);
  __atomic_store_n(&core->state, FLAGCX_NET_REQUEST_FREE, __ATOMIC_RELEASE);
}

flagcxResult_t flagcxNetRequestCoreAcquire(struct flagcxNetRequestCore *core) {
  if (core == NULL)
    return flagcxInvalidArgument;
  uint32_t expected = FLAGCX_NET_REQUEST_FREE;
  if (!__atomic_compare_exchange_n(&core->state, &expected,
                                   FLAGCX_NET_REQUEST_PENDING, false,
                                   __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
    return flagcxInProgress;
  __atomic_store_n(&core->pending, 0, __ATOMIC_RELAXED);
  __atomic_store_n(&core->result, flagcxSuccess, __ATOMIC_RELEASE);
  return flagcxSuccess;
}

flagcxResult_t flagcxNetRequestCoreAddPending(struct flagcxNetRequestCore *core,
                                              uint32_t count) {
  if (core == NULL || count == 0 ||
      __atomic_load_n(&core->state, __ATOMIC_ACQUIRE) !=
          FLAGCX_NET_REQUEST_PENDING)
    return flagcxInvalidArgument;
  uint32_t old = __atomic_fetch_add(&core->pending, count, __ATOMIC_ACQ_REL);
  if (old > UINT32_MAX - count) {
    __atomic_fetch_sub(&core->pending, count, __ATOMIC_ACQ_REL);
    return flagcxInvalidArgument;
  }
  return flagcxSuccess;
}

static void flagcxNetRequestCoreRecordResult(struct flagcxNetRequestCore *core,
                                             flagcxResult_t result) {
  if (result == flagcxSuccess || result == flagcxInProgress)
    return;
  flagcxResult_t expected = flagcxSuccess;
  __atomic_compare_exchange_n(&core->result, &expected, result, false,
                              __ATOMIC_RELEASE, __ATOMIC_RELAXED);
}

flagcxResult_t flagcxNetRequestCoreComplete(struct flagcxNetRequestCore *core,
                                            uint32_t count,
                                            flagcxResult_t result) {
  if (core == NULL || count == 0 ||
      __atomic_load_n(&core->state, __ATOMIC_ACQUIRE) !=
          FLAGCX_NET_REQUEST_PENDING)
    return flagcxInvalidArgument;
  flagcxNetRequestCoreRecordResult(core, result);

  uint32_t pending = __atomic_load_n(&core->pending, __ATOMIC_ACQUIRE);
  while (true) {
    if (pending < count)
      return flagcxInvalidArgument;
    const uint32_t next = pending - count;
    if (__atomic_compare_exchange_n(&core->pending, &pending, next, false,
                                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
      if (next == 0)
        __atomic_store_n(&core->state, FLAGCX_NET_REQUEST_COMPLETE,
                         __ATOMIC_RELEASE);
      return flagcxSuccess;
    }
  }
}

flagcxResult_t flagcxNetRequestCoreFinish(struct flagcxNetRequestCore *core,
                                          flagcxResult_t result) {
  if (core == NULL ||
      __atomic_load_n(&core->state, __ATOMIC_ACQUIRE) !=
          FLAGCX_NET_REQUEST_PENDING ||
      __atomic_load_n(&core->pending, __ATOMIC_ACQUIRE) != 0)
    return flagcxInvalidArgument;
  flagcxNetRequestCoreRecordResult(core, result);
  __atomic_store_n(&core->state, FLAGCX_NET_REQUEST_COMPLETE, __ATOMIC_RELEASE);
  return flagcxSuccess;
}

flagcxResult_t flagcxNetRequestCoreTest(const struct flagcxNetRequestCore *core,
                                        int *done) {
  if (core == NULL || done == NULL)
    return flagcxInvalidArgument;
  const uint32_t state = __atomic_load_n(&core->state, __ATOMIC_ACQUIRE);
  if (state == FLAGCX_NET_REQUEST_FREE)
    return flagcxInvalidArgument;
  *done = state == FLAGCX_NET_REQUEST_COMPLETE;
  return *done ? __atomic_load_n(&core->result, __ATOMIC_ACQUIRE)
               : flagcxSuccess;
}

flagcxResult_t flagcxNetRequestCoreRelease(struct flagcxNetRequestCore *core) {
  if (core == NULL || __atomic_load_n(&core->state, __ATOMIC_ACQUIRE) !=
                          FLAGCX_NET_REQUEST_COMPLETE)
    return flagcxInvalidArgument;
  flagcxNetRequestCoreInit(core);
  return flagcxSuccess;
}

flagcxResult_t flagcxNetPostResultInit(struct flagcxNetPostResult *post,
                                       int requested, int accepted,
                                       flagcxResult_t result) {
  if (post == NULL || requested < 0 || accepted < 0 || accepted > requested)
    return flagcxInvalidArgument;
  post->result = result;
  post->requested = requested;
  post->accepted = accepted;
  return flagcxSuccess;
}

flagcxResult_t
flagcxNetResolveOneSideRange(const struct flagcxOneSideHandleInfo *info,
                             int rank, uint64_t offset, size_t size,
                             struct flagcxNetResolvedRange *range) {
  if (info == NULL || range == NULL || info->baseVas == NULL ||
      info->regionSizes == NULL || info->mrInfos == NULL || rank < 0 ||
      rank >= info->nRanks)
    return flagcxInvalidArgument;

  const uintptr_t base = info->baseVas[rank];
  const size_t regionSize = info->regionSizes[rank];
  if (offset > regionSize || size > regionSize - offset ||
      offset > UINTPTR_MAX - base || size > UINTPTR_MAX - base - offset)
    return flagcxInvalidArgument;

  range->address = base + (uintptr_t)offset;
  range->size = size;
  range->mrInfo = &info->mrInfos[rank];
  return flagcxSuccess;
}

flagcxResult_t flagcxNetTestBatchCommon(void **requests, int nRequests,
                                        int *doneFlags, int *doneCount,
                                        flagcxNetTestRequestFn testFn) {
  if (doneCount == NULL || nRequests < 0 ||
      (nRequests > 0 &&
       (requests == NULL || doneFlags == NULL || testFn == NULL)))
    return flagcxInvalidArgument;

  *doneCount = 0;
  flagcxResult_t firstError = flagcxSuccess;
  for (int i = 0; i < nRequests; ++i) {
    doneFlags[i] = 0;
    if (requests[i] == NULL) {
      doneFlags[i] = 1;
    } else {
      flagcxResult_t result = testFn(requests[i], &doneFlags[i], NULL);
      if (result != flagcxSuccess && firstError == flagcxSuccess)
        firstError = result;
    }
    if (doneFlags[i]) {
      requests[i] = NULL;
      (*doneCount)++;
    }
  }
  return firstError;
}
