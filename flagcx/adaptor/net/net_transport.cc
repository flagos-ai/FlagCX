/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "net_transport.h"

#include "onesided_types.h"

#include <limits.h>

flagcxResult_t flagcxTransportResolveRegion(const flagcxTransportRegion *region,
                                            uint64_t offset, size_t size,
                                            uint32_t keyIndex, int localKey,
                                            uintptr_t *address, uint32_t *key) {
  if (region == NULL || region->mrInfo == NULL || address == NULL ||
      key == NULL || (localKey != 0 && localKey != 1) ||
      region->base > UINTPTR_MAX - region->size || offset > region->size ||
      size > region->size - offset)
    return flagcxInvalidArgument;

  const struct flagcxNetMrInfo *mrInfo = region->mrInfo;
  if (mrInfo->nKeys == 0 || mrInfo->nKeys > FLAGCX_NET_MAX_MR_KEYS)
    return flagcxInvalidArgument;
  const uint32_t resolvedIndex = mrInfo->nKeys == 1 ? 0 : keyIndex;
  if (resolvedIndex >= mrInfo->nKeys)
    return flagcxNotSupported;

  *address = region->base + offset;
  *key = localKey ? mrInfo->lkeys[resolvedIndex] : mrInfo->rkeys[resolvedIndex];
  return flagcxSuccess;
}

static thread_local struct flagcxNetSubmitContext flagcxNetThreadSubmitContext =
    {};
static thread_local bool flagcxNetThreadSubmitContextValid = false;

flagcxResult_t
flagcxNetSetSubmitContext(const struct flagcxNetSubmitContext *context) {
  if (context == NULL)
    return flagcxInvalidArgument;
  flagcxNetThreadSubmitContext = *context;
  flagcxNetThreadSubmitContextValid = true;
  return flagcxSuccess;
}

flagcxResult_t
flagcxNetGetSubmitContext(struct flagcxNetSubmitContext *context) {
  if (context == NULL)
    return flagcxInvalidArgument;
  if (!flagcxNetThreadSubmitContextValid)
    return flagcxNotSupported;
  *context = flagcxNetThreadSubmitContext;
  return flagcxSuccess;
}

void flagcxNetClearSubmitContext(void) {
  flagcxNetThreadSubmitContext = {};
  flagcxNetThreadSubmitContextValid = false;
}

static void flagcxNetTransportLock(uint32_t *lock) {
  while (__atomic_exchange_n(lock, 1, __ATOMIC_ACQUIRE) != 0) {
    while (__atomic_load_n(lock, __ATOMIC_RELAXED) != 0) {
    }
  }
}

static void flagcxNetTransportUnlock(uint32_t *lock) {
  __atomic_store_n(lock, 0, __ATOMIC_RELEASE);
}

static bool
flagcxNetSubmitContextMatches(const struct flagcxNetSubmitContext *lhs,
                              const struct flagcxNetSubmitContext *rhs) {
  return lhs->orderingKey == rhs->orderingKey && lhs->groupId == rhs->groupId &&
         lhs->generation == rhs->generation && lhs->sequence == rhs->sequence &&
         lhs->flags == rhs->flags;
}

static void
flagcxNetCompletionEntryReset(struct flagcxNetCompletionEntry *entry) {
  entry->context = {};
  entry->releaseGroup = NULL;
  entry->result = flagcxSuccess;
  entry->state = FLAGCX_NET_COMPLETION_ENTRY_FREE;
}

static flagcxResult_t
flagcxNetReleaseGroupTrackLocked(struct flagcxNetReleaseGroup *group,
                                 const struct flagcxNetSubmitContext *context) {
  if (group->state != FLAGCX_NET_RELEASE_GROUP_OPEN ||
      group->groupId != context->groupId ||
      group->generation != context->generation ||
      group->pending == UINT32_MAX || group->members == UINT32_MAX)
    return flagcxInvalidArgument;
  group->pending++;
  group->members++;
  return flagcxSuccess;
}

static flagcxResult_t
flagcxNetReleaseGroupCompleteLocked(struct flagcxNetReleaseGroup *group,
                                    flagcxResult_t result) {
  if (group->state == FLAGCX_NET_RELEASE_GROUP_COMPLETE || group->pending == 0)
    return flagcxInvalidArgument;
  if (result != flagcxSuccess && group->firstError == flagcxSuccess)
    group->firstError = result;
  group->pending--;
  if (group->pending == 0 && group->state == FLAGCX_NET_RELEASE_GROUP_SEALED) {
    group->releaseAllowed = group->firstError == flagcxSuccess;
    __atomic_store_n(&group->state, FLAGCX_NET_RELEASE_GROUP_COMPLETE,
                     __ATOMIC_RELEASE);
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxTransportSelectLane(flagcxTransportLaneSet *lanes,
                                         flagcxTransportLaneMode mode,
                                         uint64_t orderingKey,
                                         uint32_t *laneIndex) {
  return flagcxNetSelectLane(lanes, mode, orderingKey, laneIndex);
}

flagcxResult_t flagcxTransportCommitLane(flagcxTransportLaneSet *lanes,
                                         flagcxTransportLaneMode mode,
                                         uint32_t laneIndex) {
  return flagcxNetCommitLane(lanes, mode, laneIndex);
}

flagcxResult_t flagcxTransportCreditInit(flagcxTransportCredit *credit,
                                         uint32_t capacity) {
  return flagcxNetCreditInit(credit, capacity);
}

flagcxResult_t flagcxTransportCreditAcquire(flagcxTransportCredit *credit,
                                            uint32_t count) {
  return flagcxNetCreditAcquire(credit, count);
}

flagcxResult_t flagcxTransportCreditRelease(flagcxTransportCredit *credit,
                                            uint32_t count) {
  return flagcxNetCreditRelease(credit, count);
}

flagcxResult_t
flagcxTransportCreditAvailable(const flagcxTransportCredit *credit,
                               uint32_t *available) {
  return flagcxNetCreditAvailable(credit, available);
}

void flagcxTransportRequestInit(flagcxTransportRequest *request) {
  flagcxNetRequestCoreInit(request);
}

flagcxResult_t flagcxTransportRequestAcquire(flagcxTransportRequest *request) {
  return flagcxNetRequestCoreAcquire(request);
}

flagcxResult_t flagcxTransportRequestAddPending(flagcxTransportRequest *request,
                                                uint32_t count) {
  return flagcxNetRequestCoreAddPending(request, count);
}

flagcxResult_t flagcxTransportRequestComplete(flagcxTransportRequest *request,
                                              uint32_t count,
                                              flagcxResult_t result) {
  return flagcxNetRequestCoreComplete(request, count, result);
}

flagcxResult_t flagcxTransportRequestFinish(flagcxTransportRequest *request,
                                            flagcxResult_t result) {
  return flagcxNetRequestCoreFinish(request, result);
}

flagcxResult_t flagcxTransportRequestTest(const flagcxTransportRequest *request,
                                          int *done) {
  return flagcxNetRequestCoreTest(request, done);
}

flagcxResult_t flagcxTransportRequestRelease(flagcxTransportRequest *request) {
  return flagcxNetRequestCoreRelease(request);
}

flagcxResult_t flagcxTransportClassifyCompletion(flagcxResult_t result,
                                                 int *completed) {
  if (completed == NULL)
    return flagcxInvalidArgument;
  *completed = 0;
  if (result == flagcxSuccess) {
    *completed = 1;
    return flagcxSuccess;
  }
  if (result == flagcxInProgress)
    return flagcxSuccess;
  return result;
}

flagcxResult_t flagcxTransportPostResultInit(flagcxTransportPostResult *post,
                                             int requested, int accepted,
                                             flagcxResult_t result) {
  return flagcxNetPostResultInit(post, requested, accepted, result);
}

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

flagcxResult_t flagcxNetReleaseGroupInit(struct flagcxNetReleaseGroup *group,
                                         uint64_t groupId,
                                         uint64_t generation) {
  if (group == NULL || groupId == 0)
    return flagcxInvalidArgument;
  group->groupId = groupId;
  group->generation = generation;
  group->pending = 0;
  group->members = 0;
  group->releaseAllowed = 0;
  group->firstError = flagcxSuccess;
  group->lock = 0;
  __atomic_store_n(&group->state, FLAGCX_NET_RELEASE_GROUP_OPEN,
                   __ATOMIC_RELEASE);
  return flagcxSuccess;
}

flagcxResult_t flagcxNetReleaseGroupSeal(struct flagcxNetReleaseGroup *group) {
  if (group == NULL || group->groupId == 0)
    return flagcxInvalidArgument;
  flagcxNetTransportLock(&group->lock);
  flagcxResult_t result = flagcxSuccess;
  if (group->state != FLAGCX_NET_RELEASE_GROUP_OPEN) {
    result = flagcxInvalidArgument;
  } else if (group->pending == 0) {
    group->releaseAllowed = group->firstError == flagcxSuccess;
    __atomic_store_n(&group->state, FLAGCX_NET_RELEASE_GROUP_COMPLETE,
                     __ATOMIC_RELEASE);
  } else {
    __atomic_store_n(&group->state, FLAGCX_NET_RELEASE_GROUP_SEALED,
                     __ATOMIC_RELEASE);
  }
  flagcxNetTransportUnlock(&group->lock);
  return result;
}

flagcxResult_t flagcxNetReleaseGroupTest(struct flagcxNetReleaseGroup *group,
                                         int *done, int *releaseAllowed) {
  if (group == NULL || group->groupId == 0 || done == NULL ||
      releaseAllowed == NULL)
    return flagcxInvalidArgument;
  flagcxNetTransportLock(&group->lock);
  *done = group->state == FLAGCX_NET_RELEASE_GROUP_COMPLETE;
  *releaseAllowed = *done ? (int)group->releaseAllowed : 0;
  const flagcxResult_t result = *done ? group->firstError : flagcxSuccess;
  flagcxNetTransportUnlock(&group->lock);
  return result;
}

flagcxResult_t flagcxNetReleaseGroupReset(struct flagcxNetReleaseGroup *group,
                                          uint64_t groupId,
                                          uint64_t generation) {
  if (group == NULL || group->groupId == 0 || groupId == 0)
    return flagcxInvalidArgument;
  flagcxNetTransportLock(&group->lock);
  if (group->state != FLAGCX_NET_RELEASE_GROUP_COMPLETE ||
      generation <= group->generation) {
    flagcxNetTransportUnlock(&group->lock);
    return flagcxInvalidArgument;
  }
  group->groupId = groupId;
  group->generation = generation;
  group->pending = 0;
  group->members = 0;
  group->releaseAllowed = 0;
  group->firstError = flagcxSuccess;
  __atomic_store_n(&group->state, FLAGCX_NET_RELEASE_GROUP_OPEN,
                   __ATOMIC_RELEASE);
  flagcxNetTransportUnlock(&group->lock);
  return flagcxSuccess;
}

flagcxResult_t flagcxNetCompletionScoreboardInit(
    struct flagcxNetCompletionScoreboard *scoreboard,
    struct flagcxNetCompletionEntry *entries, uint32_t capacity,
    uint64_t generation, uint64_t initialSequence) {
  if (scoreboard == NULL || entries == NULL || capacity == 0 ||
      initialSequence == UINT64_MAX)
    return flagcxInvalidArgument;
  scoreboard->entries = entries;
  scoreboard->capacity = capacity;
  scoreboard->inFlight = 0;
  scoreboard->generation = generation;
  scoreboard->nextSequence = initialSequence;
  scoreboard->firstError = flagcxSuccess;
  scoreboard->lock = 0;
  for (uint32_t i = 0; i < capacity; ++i)
    flagcxNetCompletionEntryReset(&entries[i]);
  return flagcxSuccess;
}

flagcxResult_t flagcxNetCompletionScoreboardReset(
    struct flagcxNetCompletionScoreboard *scoreboard, uint64_t generation,
    uint64_t initialSequence) {
  if (scoreboard == NULL || scoreboard->entries == NULL ||
      scoreboard->capacity == 0 || initialSequence == UINT64_MAX)
    return flagcxInvalidArgument;
  flagcxNetTransportLock(&scoreboard->lock);
  if (scoreboard->inFlight != 0 || generation <= scoreboard->generation) {
    flagcxNetTransportUnlock(&scoreboard->lock);
    return flagcxInvalidArgument;
  }
  for (uint32_t i = 0; i < scoreboard->capacity; ++i) {
    if (scoreboard->entries[i].state != FLAGCX_NET_COMPLETION_ENTRY_FREE) {
      flagcxNetTransportUnlock(&scoreboard->lock);
      return flagcxInternalError;
    }
  }
  scoreboard->generation = generation;
  scoreboard->nextSequence = initialSequence;
  scoreboard->firstError = flagcxSuccess;
  flagcxNetTransportUnlock(&scoreboard->lock);
  return flagcxSuccess;
}

flagcxResult_t
flagcxNetGetVisibilityDomainInit(struct flagcxNetGetVisibilityDomain *domain,
                                 int peer, uint64_t orderingKey) {
  if (domain == NULL || peer < 0)
    return flagcxInvalidArgument;
  *domain = {};
  domain->orderingKey = orderingKey;
  domain->peer = peer;
  domain->flushResult = flagcxSuccess;
  domain->inUse = 1;
  return flagcxSuccess;
}

flagcxResult_t
flagcxNetGetVisibilityIssue(struct flagcxNetGetVisibilityDomain *domain,
                            uint64_t *getSequence) {
  if (domain == NULL || getSequence == NULL || domain->inUse == 0 ||
      domain->issuedGetSequence >= UINT64_MAX - 1)
    return flagcxInvalidArgument;
  *getSequence = ++domain->issuedGetSequence;
  return flagcxSuccess;
}

flagcxResult_t
flagcxNetGetVisibilityCancelIssue(struct flagcxNetGetVisibilityDomain *domain,
                                  uint64_t getSequence) {
  if (domain == NULL || domain->inUse == 0 || getSequence == 0 ||
      getSequence != domain->issuedGetSequence ||
      getSequence <= domain->dataCompletedGetSequence)
    return flagcxInvalidArgument;
  domain->issuedGetSequence--;
  return flagcxSuccess;
}

flagcxResult_t
flagcxNetGetVisibilityRemoveIssue(struct flagcxNetGetVisibilityDomain *domain,
                                  uint64_t getSequence) {
  if (domain == NULL || domain->inUse == 0 || getSequence == 0 ||
      getSequence > domain->issuedGetSequence)
    return flagcxInvalidArgument;
  domain->issuedGetSequence--;
  if (domain->dataCompletedGetSequence >= getSequence)
    domain->dataCompletedGetSequence--;
  if (domain->flushTargetGetSequence >= getSequence)
    domain->flushTargetGetSequence--;
  if (domain->visibleGetSequence >= getSequence)
    domain->visibleGetSequence--;
  return flagcxSuccess;
}

flagcxResult_t
flagcxNetGetVisibilityAdvanceData(struct flagcxNetGetVisibilityDomain *domain,
                                  uint64_t dataCompletedGetSequence) {
  if (domain == NULL || domain->inUse == 0 ||
      dataCompletedGetSequence < domain->dataCompletedGetSequence ||
      dataCompletedGetSequence > domain->issuedGetSequence)
    return flagcxInvalidArgument;
  domain->dataCompletedGetSequence = dataCompletedGetSequence;
  return flagcxSuccess;
}

flagcxResult_t
flagcxNetGetVisibilityBeginFlush(struct flagcxNetGetVisibilityDomain *domain,
                                 uint64_t *flushTarget) {
  if (domain == NULL || flushTarget == NULL || domain->inUse == 0 ||
      domain->flushRequest != NULL ||
      domain->flushTargetGetSequence != domain->visibleGetSequence ||
      domain->dataCompletedGetSequence <= domain->visibleGetSequence)
    return flagcxInvalidArgument;
  domain->flushTargetGetSequence = domain->dataCompletedGetSequence;
  *flushTarget = domain->flushTargetGetSequence;
  return flagcxSuccess;
}

flagcxResult_t flagcxNetGetVisibilityAdvanceVisible(
    struct flagcxNetGetVisibilityDomain *domain, uint64_t visibleGetSequence) {
  if (domain == NULL || domain->inUse == 0 || domain->flushRequest != NULL ||
      domain->flushTargetGetSequence != domain->visibleGetSequence ||
      visibleGetSequence < domain->visibleGetSequence ||
      visibleGetSequence > domain->dataCompletedGetSequence)
    return flagcxInvalidArgument;
  domain->visibleGetSequence = visibleGetSequence;
  domain->flushTargetGetSequence = visibleGetSequence;
  domain->flushResult = flagcxSuccess;
  return flagcxSuccess;
}

flagcxResult_t
flagcxNetGetVisibilityCompleteFlush(struct flagcxNetGetVisibilityDomain *domain,
                                    flagcxResult_t result) {
  if (domain == NULL || domain->inUse == 0 ||
      domain->flushTargetGetSequence < domain->visibleGetSequence ||
      domain->flushTargetGetSequence > domain->dataCompletedGetSequence ||
      (domain->flushTargetGetSequence == domain->visibleGetSequence &&
       domain->flushRequest == NULL))
    return flagcxInvalidArgument;
  domain->flushRequest = NULL;
  domain->flushResult = result;
  // A failed flush still retires the covered logical range with an error. It
  // must not leave later shutdown/drain progress permanently blocked.
  domain->visibleGetSequence = domain->flushTargetGetSequence;
  return flagcxSuccess;
}

flagcxResult_t
flagcxNetTrackSubmit(struct flagcxNetCompletionScoreboard *scoreboard,
                     const struct flagcxNetSubmitContext *context,
                     struct flagcxNetReleaseGroup *releaseGroup) {
  if (scoreboard == NULL || context == NULL || scoreboard->entries == NULL ||
      scoreboard->capacity == 0 || context->sequence == UINT64_MAX)
    return flagcxInvalidArgument;

  flagcxNetTransportLock(&scoreboard->lock);
  flagcxResult_t result = flagcxSuccess;
  struct flagcxNetCompletionEntry *entry = NULL;
  if (context->generation != scoreboard->generation ||
      context->sequence < scoreboard->nextSequence) {
    result = flagcxInvalidArgument;
    goto exit;
  }
  if (context->sequence - scoreboard->nextSequence >= scoreboard->capacity) {
    result = flagcxInProgress;
    goto exit;
  }

  entry = &scoreboard->entries[context->sequence % scoreboard->capacity];
  if (entry->state != FLAGCX_NET_COMPLETION_ENTRY_FREE) {
    result = flagcxInvalidArgument;
    goto exit;
  }
  if ((releaseGroup == NULL && context->groupId != 0) ||
      (releaseGroup != NULL && context->groupId == 0)) {
    result = flagcxInvalidArgument;
    goto exit;
  }

  if (releaseGroup != NULL) {
    flagcxNetTransportLock(&releaseGroup->lock);
    result = flagcxNetReleaseGroupTrackLocked(releaseGroup, context);
    if (result == flagcxSuccess) {
      entry->context = *context;
      entry->releaseGroup = releaseGroup;
      entry->result = flagcxSuccess;
      entry->state = FLAGCX_NET_COMPLETION_ENTRY_PENDING;
      scoreboard->inFlight++;
    }
    flagcxNetTransportUnlock(&releaseGroup->lock);
  } else {
    entry->context = *context;
    entry->releaseGroup = NULL;
    entry->result = flagcxSuccess;
    entry->state = FLAGCX_NET_COMPLETION_ENTRY_PENDING;
    scoreboard->inFlight++;
  }

exit:
  flagcxNetTransportUnlock(&scoreboard->lock);
  return result;
}

flagcxResult_t
flagcxNetTrackCancel(struct flagcxNetCompletionScoreboard *scoreboard,
                     const struct flagcxNetSubmitContext *context) {
  if (scoreboard == NULL || context == NULL || scoreboard->entries == NULL ||
      scoreboard->capacity == 0)
    return flagcxInvalidArgument;

  flagcxNetTransportLock(&scoreboard->lock);
  if (context->generation != scoreboard->generation ||
      context->sequence < scoreboard->nextSequence ||
      context->sequence - scoreboard->nextSequence >= scoreboard->capacity) {
    flagcxNetTransportUnlock(&scoreboard->lock);
    return flagcxInvalidArgument;
  }

  struct flagcxNetCompletionEntry *entry =
      &scoreboard->entries[context->sequence % scoreboard->capacity];
  if (entry->state != FLAGCX_NET_COMPLETION_ENTRY_PENDING ||
      !flagcxNetSubmitContextMatches(&entry->context, context)) {
    flagcxNetTransportUnlock(&scoreboard->lock);
    return flagcxInvalidArgument;
  }

  if (entry->releaseGroup != NULL) {
    flagcxNetTransportLock(&entry->releaseGroup->lock);
    if (entry->releaseGroup->state != FLAGCX_NET_RELEASE_GROUP_OPEN ||
        entry->releaseGroup->pending == 0 ||
        entry->releaseGroup->members == 0) {
      flagcxNetTransportUnlock(&entry->releaseGroup->lock);
      flagcxNetTransportUnlock(&scoreboard->lock);
      return flagcxInvalidArgument;
    }
    entry->releaseGroup->pending--;
    entry->releaseGroup->members--;
    flagcxNetTransportUnlock(&entry->releaseGroup->lock);
  }

  flagcxNetCompletionEntryReset(entry);
  scoreboard->inFlight--;
  flagcxNetTransportUnlock(&scoreboard->lock);
  return flagcxSuccess;
}

flagcxResult_t
flagcxNetTrackCompletion(struct flagcxNetCompletionScoreboard *scoreboard,
                         const struct flagcxNetSubmitContext *context,
                         flagcxResult_t result, uint32_t *advanced) {
  if (scoreboard == NULL || context == NULL || advanced == NULL ||
      scoreboard->entries == NULL || scoreboard->capacity == 0)
    return flagcxInvalidArgument;
  *advanced = 0;

  flagcxNetTransportLock(&scoreboard->lock);
  if (context->generation != scoreboard->generation ||
      context->sequence < scoreboard->nextSequence ||
      context->sequence - scoreboard->nextSequence >= scoreboard->capacity) {
    flagcxNetTransportUnlock(&scoreboard->lock);
    return flagcxInvalidArgument;
  }
  struct flagcxNetCompletionEntry *entry =
      &scoreboard->entries[context->sequence % scoreboard->capacity];
  if (entry->state != FLAGCX_NET_COMPLETION_ENTRY_PENDING ||
      !flagcxNetSubmitContextMatches(&entry->context, context)) {
    flagcxNetTransportUnlock(&scoreboard->lock);
    return flagcxInvalidArgument;
  }
  if (result == flagcxInProgress) {
    flagcxNetTransportUnlock(&scoreboard->lock);
    return flagcxSuccess;
  }

  if (entry->releaseGroup != NULL) {
    flagcxNetTransportLock(&entry->releaseGroup->lock);
    const flagcxResult_t groupResult =
        flagcxNetReleaseGroupCompleteLocked(entry->releaseGroup, result);
    flagcxNetTransportUnlock(&entry->releaseGroup->lock);
    if (groupResult != flagcxSuccess) {
      flagcxNetTransportUnlock(&scoreboard->lock);
      return groupResult;
    }
  }
  if (result != flagcxSuccess && scoreboard->firstError == flagcxSuccess)
    scoreboard->firstError = result;
  entry->result = result;
  entry->state = FLAGCX_NET_COMPLETION_ENTRY_COMPLETE;

  while (scoreboard->inFlight != 0) {
    struct flagcxNetCompletionEntry *next =
        &scoreboard->entries[scoreboard->nextSequence % scoreboard->capacity];
    if (next->state != FLAGCX_NET_COMPLETION_ENTRY_COMPLETE ||
        next->context.generation != scoreboard->generation ||
        next->context.sequence != scoreboard->nextSequence)
      break;
    flagcxNetCompletionEntryReset(next);
    scoreboard->inFlight--;
    scoreboard->nextSequence++;
    (*advanced)++;
  }
  flagcxNetTransportUnlock(&scoreboard->lock);
  return flagcxSuccess;
}

flagcxResult_t flagcxNetCompletionScoreboardQuery(
    struct flagcxNetCompletionScoreboard *scoreboard, uint64_t *nextSequence,
    uint32_t *inFlight, flagcxResult_t *firstError) {
  if (scoreboard == NULL || nextSequence == NULL || inFlight == NULL ||
      firstError == NULL || scoreboard->entries == NULL ||
      scoreboard->capacity == 0)
    return flagcxInvalidArgument;
  flagcxNetTransportLock(&scoreboard->lock);
  *nextSequence = scoreboard->nextSequence;
  *inFlight = scoreboard->inFlight;
  *firstError = scoreboard->firstError;
  flagcxNetTransportUnlock(&scoreboard->lock);
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
