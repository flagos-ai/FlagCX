/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "kernel_proxy_transport.h"

#include "device_api/completion_word.h"
#include "device_api/fifo_producer_gate.h"
#include "flagcx_kernel_core.h"

#include <sched.h>
#include <stdlib.h>
#include <string.h>

static void
flagcxKernelProxyRequestReset(struct flagcxKernelProxyRequest *request) {
  memset(request, 0, sizeof(*request));
  request->completionResult = flagcxSuccess;
  request->peer = -1;
  request->dstMrIdx = -1;
  request->stagingSlot = -1;
  request->getVisibilityDomain = UINT32_MAX;
  request->state = FLAGCX_KERNEL_PROXY_REQUEST_FREE;
  request->completionStage = FLAGCX_KERNEL_PROXY_COMPLETION_DATA_POSTED;
}

flagcxResult_t
flagcxKernelProxyTransportInit(struct flagcxKernelProxyTransport *transport,
                               uint32_t capacity, uint32_t stagingSlotCount,
                               uint64_t generation, uint64_t orderingKey) {
  if (transport == NULL || capacity == 0 || generation == 0)
    return flagcxInvalidArgument;

  memset(transport, 0, sizeof(*transport));
  transport->completionEntries = (struct flagcxNetCompletionEntry *)calloc(
      capacity, sizeof(struct flagcxNetCompletionEntry));
  transport->requests = (struct flagcxKernelProxyRequest *)calloc(
      capacity, sizeof(struct flagcxKernelProxyRequest));
  transport->getVisibilityDomains =
      (struct flagcxNetGetVisibilityDomain *)calloc(
          capacity, sizeof(struct flagcxNetGetVisibilityDomain));
  transport->getVisibilityDomainReferenced =
      (uint8_t *)calloc(capacity, sizeof(uint8_t));
  if (stagingSlotCount != 0)
    transport->stagingInUse =
        (uint8_t *)calloc(stagingSlotCount, sizeof(uint8_t));
  if (transport->completionEntries == NULL || transport->requests == NULL ||
      transport->getVisibilityDomains == NULL ||
      transport->getVisibilityDomainReferenced == NULL ||
      (stagingSlotCount != 0 && transport->stagingInUse == NULL)) {
    flagcxKernelProxyTransportDestroy(transport);
    return flagcxSystemError;
  }

  for (uint32_t i = 0; i < capacity; ++i)
    flagcxKernelProxyRequestReset(&transport->requests[i]);
  flagcxResult_t result = flagcxNetCompletionScoreboardInit(
      &transport->scoreboard, transport->completionEntries, capacity,
      generation, 1);
  if (result != flagcxSuccess) {
    flagcxKernelProxyTransportDestroy(transport);
    return result;
  }

  transport->capacity = capacity;
  transport->stagingSlotCount = stagingSlotCount;
  transport->generation = generation;
  transport->orderingKey = orderingKey;
  transport->nextSequence = 1;
  return flagcxSuccess;
}

void flagcxKernelProxyTransportDestroy(
    struct flagcxKernelProxyTransport *transport) {
  if (transport == NULL)
    return;
  free(transport->stagingInUse);
  free(transport->getVisibilityDomainReferenced);
  free(transport->getVisibilityDomains);
  free(transport->requests);
  free(transport->completionEntries);
  memset(transport, 0, sizeof(*transport));
}

flagcxResult_t flagcxKernelProxyCloseFifoProducerGate(uint64_t *fifoBuffer) {
  if (fifoBuffer == NULL)
    return flagcxInvalidArgument;
  flagcxCompletionWord_t *producerState =
      flagcxFifoControlPtr(fifoBuffer, flagcxFifoIdxProducerState);
  const flagcxCompletionWord_t closedMask =
      flagcxFifoProducerClosedMask<flagcxCompletionWord_t>();
  const flagcxCompletionWord_t activeMask =
      flagcxFifoProducerActiveMask<flagcxCompletionWord_t>();
  __atomic_fetch_or(producerState, closedMask, __ATOMIC_ACQ_REL);
  while ((__atomic_load_n(producerState, __ATOMIC_ACQUIRE) & activeMask) != 0)
    sched_yield();
  return flagcxSuccess;
}

flagcxResult_t flagcxKernelProxyFinalizeTerminalFifo(uint64_t *fifoBuffer) {
  if (fifoBuffer == NULL)
    return flagcxInvalidArgument;
  flagcxCompletionWord_t *producerState =
      flagcxFifoControlPtr(fifoBuffer, flagcxFifoIdxProducerState);
  const flagcxCompletionWord_t state =
      __atomic_load_n(producerState, __ATOMIC_ACQUIRE);
  const flagcxCompletionWord_t closedMask =
      flagcxFifoProducerClosedMask<flagcxCompletionWord_t>();
  const flagcxCompletionWord_t activeMask =
      flagcxFifoProducerActiveMask<flagcxCompletionWord_t>();
  if ((state & closedMask) == 0 || (state & activeMask) != 0)
    return flagcxInvalidUsage;

  flagcxCompletionWord_t produced =
      __atomic_load_n(flagcxFifoControlPtr(fifoBuffer, flagcxFifoIdxProduced),
                      __ATOMIC_ACQUIRE);
  __atomic_store_n(flagcxFifoControlPtr(fifoBuffer, flagcxFifoIdxConsumed),
                   produced, __ATOMIC_RELEASE);
  __atomic_store_n(flagcxFifoControlPtr(fifoBuffer, flagcxFifoIdxCompleted),
                   produced, __ATOMIC_RELEASE);
  return flagcxSuccess;
}

flagcxResult_t
flagcxKernelProxyTrackNext(struct flagcxKernelProxyTransport *transport,
                           uint32_t flags,
                           struct flagcxNetSubmitContext *submit) {
  if (transport == NULL || submit == NULL || transport->capacity == 0)
    return flagcxInvalidArgument;
  struct flagcxNetSubmitContext context = {};
  context.orderingKey = transport->orderingKey;
  context.generation = transport->generation;
  context.sequence = transport->nextSequence;
  context.flags = flags;
  flagcxResult_t result =
      flagcxNetTrackSubmit(&transport->scoreboard, &context, NULL);
  if (result != flagcxSuccess)
    return result;
  transport->nextSequence++;
  *submit = context;
  return flagcxSuccess;
}

flagcxResult_t
flagcxKernelProxyReserveRequest(struct flagcxKernelProxyTransport *transport,
                                const struct flagcxNetSubmitContext *submit,
                                int peer, int stagingSlot, uint32_t *slot) {
  if (transport == NULL || submit == NULL || slot == NULL || peer < 0)
    return flagcxInvalidArgument;
  for (uint32_t i = 0; i < transport->capacity; ++i) {
    struct flagcxKernelProxyRequest *request = &transport->requests[i];
    if (request->state != FLAGCX_KERNEL_PROXY_REQUEST_FREE)
      continue;
    request->submit = *submit;
    request->peer = peer;
    request->stagingSlot = stagingSlot;
    request->state = FLAGCX_KERNEL_PROXY_REQUEST_RESERVED;
    *slot = i;
    return flagcxSuccess;
  }
  return flagcxInProgress;
}

flagcxResult_t
flagcxKernelProxyPublishRequest(struct flagcxKernelProxyTransport *transport,
                                uint32_t slot, void *request,
                                flagcxResult_t completionResult) {
  if (transport == NULL || request == NULL || slot >= transport->capacity)
    return flagcxInvalidArgument;
  struct flagcxKernelProxyRequest *entry = &transport->requests[slot];
  if (entry->state != FLAGCX_KERNEL_PROXY_REQUEST_RESERVED)
    return flagcxInvalidArgument;
  entry->request = request;
  entry->completionResult = completionResult;
  entry->state = FLAGCX_KERNEL_PROXY_REQUEST_POSTED;
  transport->nativeInflight++;
  return flagcxSuccess;
}

flagcxResult_t
flagcxKernelProxyRequireGetFlush(struct flagcxKernelProxyTransport *transport,
                                 uint32_t slot, int dstMrIdx, uint64_t dstOff,
                                 size_t size, void *flushRecvComm) {
  if (transport == NULL || slot >= transport->capacity || dstMrIdx < 0 ||
      size == 0 || flushRecvComm == NULL)
    return flagcxInvalidArgument;
  struct flagcxKernelProxyRequest *entry = &transport->requests[slot];
  if (entry->state != FLAGCX_KERNEL_PROXY_REQUEST_RESERVED)
    return flagcxInvalidArgument;
  struct flagcxNetGetVisibilityDomain *domain = NULL;
  uint32_t domainIndex = UINT32_MAX;
  // Matching and unused-domain lookup are independent linear passes. Do not
  // scan requests inside this loop: that made every GET O(capacity^2).
  for (uint32_t i = 0; i < transport->capacity; ++i) {
    struct flagcxNetGetVisibilityDomain *candidate =
        &transport->getVisibilityDomains[i];
    if (candidate->inUse != 0 && candidate->peer == entry->peer &&
        candidate->orderingKey == transport->orderingKey) {
      domain = candidate;
      domainIndex = i;
      break;
    }
  }
  if (domain == NULL) {
    for (uint32_t i = 0; i < transport->capacity; ++i) {
      if (transport->getVisibilityDomains[i].inUse == 0) {
        domain = &transport->getVisibilityDomains[i];
        domainIndex = i;
        break;
      }
    }
  }
  if (domain == NULL) {
    memset(transport->getVisibilityDomainReferenced, 0, transport->capacity);
    for (uint32_t requestIndex = 0; requestIndex < transport->capacity;
         ++requestIndex) {
      const struct flagcxKernelProxyRequest *request =
          &transport->requests[requestIndex];
      if (request->state != FLAGCX_KERNEL_PROXY_REQUEST_FREE &&
          request->getVisibilityDomain < transport->capacity)
        transport->getVisibilityDomainReferenced[request->getVisibilityDomain] =
            1;
    }
    for (uint32_t i = 0; i < transport->capacity; ++i) {
      struct flagcxNetGetVisibilityDomain *candidate =
          &transport->getVisibilityDomains[i];
      if (candidate->issuedGetSequence == candidate->visibleGetSequence &&
          candidate->flushRequest == NULL &&
          transport->getVisibilityDomainReferenced[i] == 0) {
        domain = candidate;
        domainIndex = i;
        break;
      }
    }
  }
  if (domain == NULL)
    return flagcxInProgress;
  if (domain->inUse == 0 || domain->peer != entry->peer ||
      domain->orderingKey != transport->orderingKey) {
    FLAGCXCHECK(flagcxNetGetVisibilityDomainInit(domain, entry->peer,
                                                 transport->orderingKey));
  }
  FLAGCXCHECK(flagcxNetGetVisibilityIssue(domain, &entry->getSequence));
  entry->getVisibilityDomain = domainIndex;
  entry->dstMrIdx = dstMrIdx;
  entry->dstOff = dstOff;
  entry->size = size;
  entry->flushRecvComm = flushRecvComm;
  entry->requiresGetFlush = 1;
  return flagcxSuccess;
}

flagcxResult_t flagcxKernelProxyPublishGetFlushPending(
    struct flagcxKernelProxyTransport *transport, uint32_t slot) {
  if (transport == NULL || slot >= transport->capacity)
    return flagcxInvalidArgument;
  struct flagcxKernelProxyRequest *entry = &transport->requests[slot];
  if (entry->state != FLAGCX_KERNEL_PROXY_REQUEST_RESERVED ||
      entry->requiresGetFlush == 0)
    return flagcxInvalidArgument;
  entry->request = NULL;
  entry->completionResult = flagcxSuccess;
  entry->completionStage = FLAGCX_KERNEL_PROXY_COMPLETION_FLUSH_PENDING;
  entry->getDataComplete = 1;
  entry->state = FLAGCX_KERNEL_PROXY_REQUEST_POSTED;
  transport->nativeInflight++;
  return flagcxSuccess;
}

flagcxResult_t flagcxKernelProxyProgressRequest(
    struct flagcxKernelProxyTransport *transport, uint32_t slot,
    flagcxKernelProxyTestRequestFn testRequest,
    flagcxKernelProxyPostFlushFn postFlush, void *flushContext, int *ready,
    flagcxResult_t *completionResult) {
  if (transport == NULL || slot >= transport->capacity || ready == NULL ||
      completionResult == NULL)
    return flagcxInvalidArgument;
  struct flagcxKernelProxyRequest *entry = &transport->requests[slot];
  if (entry->state != FLAGCX_KERNEL_PROXY_REQUEST_POSTED)
    return flagcxInvalidArgument;

  *ready = 0;
  *completionResult = entry->completionResult;
  if (entry->completionStage == FLAGCX_KERNEL_PROXY_COMPLETION_FLUSH_PENDING) {
    // Progress for the whole peer domain is handled below after every data
    // CQE has had a chance to update the contiguous GET prefix.
  } else {
    if (entry->request == NULL || testRequest == NULL)
      return flagcxInvalidArgument;
    int done = 0;
    flagcxResult_t result = testRequest(entry->request, &done, NULL);
    if (result != flagcxSuccess) {
      entry->completionResult = result;
      *completionResult = result;
      done = 1;
    }
    if (!done)
      return flagcxSuccess;
    if (entry->requiresGetFlush == 0) {
      *ready = 1;
      return flagcxSuccess;
    }
    entry->request = NULL;
    entry->getDataComplete = 1;
    entry->completionStage = FLAGCX_KERNEL_PROXY_COMPLETION_FLUSH_PENDING;
    // Defer posting until a later progress call. The production poll first
    // visits all data requests, matching NCCL's snapshot-after-issued model.
    if (entry->completionResult == flagcxSuccess)
      return flagcxSuccess;
  }

  if (entry->getVisibilityDomain == UINT32_MAX ||
      entry->getVisibilityDomain >= transport->capacity)
    return flagcxInternalError;
  struct flagcxNetGetVisibilityDomain *domain =
      &transport->getVisibilityDomains[entry->getVisibilityDomain];

  if (domain->flushRequest != NULL) {
    if (testRequest == NULL)
      return flagcxInvalidArgument;
    int done = 0;
    flagcxResult_t result = testRequest(domain->flushRequest, &done, NULL);
    if (result != flagcxSuccess)
      done = 1;
    if (done) {
      const uint64_t previousVisible = domain->visibleGetSequence;
      FLAGCXCHECK(flagcxNetGetVisibilityCompleteFlush(domain, result));
      if (result != flagcxSuccess) {
        for (uint32_t i = 0; i < transport->capacity; ++i) {
          struct flagcxKernelProxyRequest *covered = &transport->requests[i];
          if (covered->state == FLAGCX_KERNEL_PROXY_REQUEST_POSTED &&
              covered->getVisibilityDomain == entry->getVisibilityDomain &&
              covered->getSequence > previousVisible &&
              covered->getSequence <= domain->visibleGetSequence &&
              covered->completionResult == flagcxSuccess)
            covered->completionResult = result;
        }
      }
    }
  }

  uint64_t completed = domain->dataCompletedGetSequence;
  while (completed < domain->issuedGetSequence) {
    struct flagcxKernelProxyRequest *next = NULL;
    for (uint32_t i = 0; i < transport->capacity; ++i) {
      struct flagcxKernelProxyRequest *candidate = &transport->requests[i];
      if (candidate->state == FLAGCX_KERNEL_PROXY_REQUEST_POSTED &&
          candidate->getVisibilityDomain == entry->getVisibilityDomain &&
          candidate->getSequence == completed + 1) {
        next = candidate;
        break;
      }
    }
    if (next == NULL || next->getDataComplete == 0)
      break;
    ++completed;
  }
  FLAGCXCHECK(flagcxNetGetVisibilityAdvanceData(domain, completed));

  if (domain->flushRequest == NULL &&
      domain->dataCompletedGetSequence > domain->visibleGetSequence) {
    const bool retryFlush =
        domain->flushTargetGetSequence > domain->visibleGetSequence;
    const uint64_t target = retryFlush ? domain->flushTargetGetSequence
                                       : domain->dataCompletedGetSequence;
    struct flagcxKernelProxyRequest *flushRange = NULL;
    for (uint32_t i = 0; i < transport->capacity; ++i) {
      struct flagcxKernelProxyRequest *candidate = &transport->requests[i];
      if (candidate->state == FLAGCX_KERNEL_PROXY_REQUEST_POSTED &&
          candidate->getVisibilityDomain == entry->getVisibilityDomain &&
          candidate->getSequence > domain->visibleGetSequence &&
          candidate->getSequence <= target &&
          candidate->completionResult == flagcxSuccess)
        flushRange = candidate;
    }
    if (flushRange == NULL) {
      FLAGCXCHECK(flagcxNetGetVisibilityAdvanceVisible(
          domain, domain->dataCompletedGetSequence));
    } else {
      uint64_t flushTarget = domain->flushTargetGetSequence;
      if (!retryFlush)
        FLAGCXCHECK(flagcxNetGetVisibilityBeginFlush(domain, &flushTarget));
      if (postFlush == NULL)
        return flagcxInvalidArgument;
      flagcxResult_t result = postFlush(
          flushContext, flushRange->flushRecvComm, flushRange->dstMrIdx,
          flushRange->dstOff, flushRange->size, &domain->flushRequest);
      if (result != flagcxInProgress &&
          (result != flagcxSuccess || domain->flushRequest == NULL)) {
        const uint64_t previousVisible = domain->visibleGetSequence;
        FLAGCXCHECK(flagcxNetGetVisibilityCompleteFlush(domain, result));
        if (result != flagcxSuccess) {
          for (uint32_t i = 0; i < transport->capacity; ++i) {
            struct flagcxKernelProxyRequest *covered = &transport->requests[i];
            if (covered->state == FLAGCX_KERNEL_PROXY_REQUEST_POSTED &&
                covered->getVisibilityDomain == entry->getVisibilityDomain &&
                covered->getSequence > previousVisible &&
                covered->getSequence <= domain->visibleGetSequence &&
                covered->completionResult == flagcxSuccess)
              covered->completionResult = result;
          }
        }
      }
    }
  }

  *completionResult = entry->completionResult;
  *ready = entry->getSequence <= domain->visibleGetSequence;
  return flagcxSuccess;
}

static flagcxResult_t flagcxKernelProxyRemoveGetVisibilitySequence(
    struct flagcxKernelProxyTransport *transport,
    struct flagcxKernelProxyRequest *entry) {
  if (entry->getSequence != 0) {
    if (entry->getVisibilityDomain >= transport->capacity)
      return flagcxInternalError;
    struct flagcxNetGetVisibilityDomain *domain =
        &transport->getVisibilityDomains[entry->getVisibilityDomain];
    if (domain->inUse == 0 || entry->getSequence > domain->issuedGetSequence)
      return flagcxInternalError;

    // Cancellation or terminal abort can remove a non-tail request. Compact
    // every later live request before adjusting the domain watermarks; leaving
    // a gap would prevent the contiguous data-completed prefix from advancing.
    for (uint32_t i = 0; i < transport->capacity; ++i) {
      struct flagcxKernelProxyRequest *later = &transport->requests[i];
      if (later->state != FLAGCX_KERNEL_PROXY_REQUEST_FREE &&
          later->getVisibilityDomain == entry->getVisibilityDomain &&
          later->getSequence > entry->getSequence)
        later->getSequence--;
    }
    FLAGCXCHECK(flagcxNetGetVisibilityRemoveIssue(domain, entry->getSequence));
  }
  return flagcxSuccess;
}

flagcxResult_t
flagcxKernelProxyCancelRequest(struct flagcxKernelProxyTransport *transport,
                               uint32_t slot) {
  if (transport == NULL || slot >= transport->capacity)
    return flagcxInvalidArgument;
  struct flagcxKernelProxyRequest *entry = &transport->requests[slot];
  if (entry->state != FLAGCX_KERNEL_PROXY_REQUEST_RESERVED)
    return flagcxInvalidArgument;
  FLAGCXCHECK(flagcxKernelProxyRemoveGetVisibilitySequence(transport, entry));
  flagcxKernelProxyRequestReset(entry);
  return flagcxSuccess;
}

flagcxResult_t
flagcxKernelProxyAbortRequest(struct flagcxKernelProxyTransport *transport,
                              uint32_t slot, int *releasedStagingSlot) {
  if (transport == NULL || releasedStagingSlot == NULL ||
      slot >= transport->capacity)
    return flagcxInvalidArgument;
  struct flagcxKernelProxyRequest *entry = &transport->requests[slot];
  if (entry->state != FLAGCX_KERNEL_PROXY_REQUEST_POSTED ||
      transport->nativeInflight == 0)
    return flagcxInvalidArgument;
  FLAGCXCHECK(flagcxKernelProxyRemoveGetVisibilitySequence(transport, entry));
  *releasedStagingSlot = entry->stagingSlot;
  transport->nativeInflight--;
  flagcxKernelProxyRequestReset(entry);
  return flagcxSuccess;
}

flagcxResult_t
flagcxKernelProxyCompleteRequest(struct flagcxKernelProxyTransport *transport,
                                 uint32_t slot, flagcxResult_t result,
                                 uint32_t *advanced, int *releasedStagingSlot) {
  if (transport == NULL || advanced == NULL || releasedStagingSlot == NULL ||
      slot >= transport->capacity)
    return flagcxInvalidArgument;
  struct flagcxKernelProxyRequest *entry = &transport->requests[slot];
  if (entry->state != FLAGCX_KERNEL_PROXY_REQUEST_POSTED ||
      transport->nativeInflight == 0)
    return flagcxInvalidArgument;

  flagcxResult_t trackResult = flagcxNetTrackCompletion(
      &transport->scoreboard, &entry->submit, result, advanced);
  if (trackResult != flagcxSuccess)
    return trackResult;
  *releasedStagingSlot = entry->stagingSlot;
  transport->nativeInflight--;
  flagcxKernelProxyRequestReset(entry);
  return flagcxSuccess;
}

flagcxResult_t
flagcxKernelProxyCompleteImmediate(struct flagcxKernelProxyTransport *transport,
                                   const struct flagcxNetSubmitContext *submit,
                                   flagcxResult_t result, uint32_t *advanced) {
  if (transport == NULL || submit == NULL || advanced == NULL)
    return flagcxInvalidArgument;
  return flagcxNetTrackCompletion(&transport->scoreboard, submit, result,
                                  advanced);
}

flagcxResult_t flagcxKernelProxyAcquireStagingSlot(
    struct flagcxKernelProxyTransport *transport, int *slot) {
  if (transport == NULL || slot == NULL || transport->stagingSlotCount == 0 ||
      transport->stagingInUse == NULL)
    return flagcxInvalidArgument;
  for (uint32_t i = 0; i < transport->stagingSlotCount; ++i) {
    uint32_t candidate =
        (transport->stagingCursor + i) % transport->stagingSlotCount;
    if (transport->stagingInUse[candidate] != 0)
      continue;
    transport->stagingInUse[candidate] = 1;
    transport->stagingCursor = (candidate + 1) % transport->stagingSlotCount;
    *slot = (int)candidate;
    return flagcxSuccess;
  }
  return flagcxInProgress;
}

flagcxResult_t flagcxKernelProxyReleaseStagingSlot(
    struct flagcxKernelProxyTransport *transport, int slot) {
  if (transport == NULL || slot < 0 ||
      (uint32_t)slot >= transport->stagingSlotCount ||
      transport->stagingInUse == NULL || transport->stagingInUse[slot] == 0)
    return flagcxInvalidArgument;
  transport->stagingInUse[slot] = 0;
  return flagcxSuccess;
}

flagcxResult_t
flagcxKernelProxyQuery(struct flagcxKernelProxyTransport *transport,
                       uint64_t *nextSequence, uint32_t *inFlight,
                       flagcxResult_t *firstError) {
  if (transport == NULL)
    return flagcxInvalidArgument;
  return flagcxNetCompletionScoreboardQuery(&transport->scoreboard,
                                            nextSequence, inFlight, firstError);
}

flagcxResult_t
flagcxKernelProxyReleaseReady(struct flagcxKernelProxyTransport *transport,
                              const struct flagcxNetSubmitContext *submit,
                              int *ready, flagcxResult_t *firstError) {
  if (transport == NULL || submit == NULL || ready == NULL ||
      firstError == NULL)
    return flagcxInvalidArgument;
  uint64_t nextSequence = 0;
  uint32_t inFlight = 0;
  flagcxResult_t result =
      flagcxKernelProxyQuery(transport, &nextSequence, &inFlight, firstError);
  if (result != flagcxSuccess)
    return result;
  *ready = nextSequence == submit->sequence;
  return flagcxSuccess;
}
