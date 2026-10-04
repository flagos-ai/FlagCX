/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_KERNEL_PROXY_TRANSPORT_H_
#define FLAGCX_KERNEL_PROXY_TRANSPORT_H_

#include "net_transport.h"

#include <stddef.h>
#include <stdint.h>

enum flagcxKernelProxyRequestState {
  FLAGCX_KERNEL_PROXY_REQUEST_FREE = 0,
  FLAGCX_KERNEL_PROXY_REQUEST_RESERVED = 1,
  FLAGCX_KERNEL_PROXY_REQUEST_POSTED = 2,
};

enum flagcxKernelProxyCompletionStage {
  FLAGCX_KERNEL_PROXY_COMPLETION_DATA_POSTED = 0,
  FLAGCX_KERNEL_PROXY_COMPLETION_FLUSH_PENDING = 1,
};

typedef flagcxResult_t (*flagcxKernelProxyTestRequestFn)(void *request,
                                                         int *done, int *sizes);
typedef flagcxResult_t (*flagcxKernelProxyPostFlushFn)(
    void *context, void *recvComm, int dstMrIdx, uint64_t dstOff, size_t size,
    void **request);

struct flagcxKernelProxyRequest {
  void *request;
  void *flushRecvComm;
  struct flagcxNetSubmitContext submit;
  flagcxResult_t completionResult;
  uint64_t dstOff;
  uint64_t getSequence;
  size_t size;
  int peer;
  int dstMrIdx;
  int stagingSlot;
  uint32_t state;
  uint32_t getVisibilityDomain;
  uint8_t requiresGetFlush;
  uint8_t getDataComplete;
  uint8_t completionStage;
};

// Per-kernel-context ordering state. The completion scoreboard orders the
// whole FIFO, while request slots may retire in any peer/CQE order.
struct flagcxKernelProxyTransport {
  struct flagcxNetCompletionScoreboard scoreboard;
  struct flagcxNetCompletionEntry *completionEntries;
  struct flagcxKernelProxyRequest *requests;
  struct flagcxNetGetVisibilityDomain *getVisibilityDomains;
  // Progress-thread-private scratch storage used to find recyclable domains
  // in O(capacity) without allocating on the GET posting path.
  uint8_t *getVisibilityDomainReferenced;
  uint8_t *stagingInUse;
  uint32_t capacity;
  uint32_t nativeInflight;
  uint32_t stagingSlotCount;
  uint32_t stagingCursor;
  uint64_t nextSequence;
  uint64_t generation;
  uint64_t orderingKey;
};

flagcxResult_t
flagcxKernelProxyTransportInit(struct flagcxKernelProxyTransport *transport,
                               uint32_t capacity, uint32_t stagingSlotCount,
                               uint64_t generation, uint64_t orderingKey);
void flagcxKernelProxyTransportDestroy(
    struct flagcxKernelProxyTransport *transport);

// Permanently closes a FIFO producer gate and waits until every producer that
// entered before closure has either published or abandoned its reservation.
// Once this returns, the FIFO produced counter cannot advance again.
flagcxResult_t flagcxKernelProxyCloseFifoProducerGate(uint64_t *fifoBuffer);

// Advances failed/unserviceable FIFO entries after the producer gate is closed
// and accepted native requests have retired.
flagcxResult_t flagcxKernelProxyFinalizeTerminalFifo(uint64_t *fifoBuffer);

// Registers the next FIFO entry. A full sequence window is transient
// backpressure; nextSequence is not consumed until registration succeeds.
flagcxResult_t
flagcxKernelProxyTrackNext(struct flagcxKernelProxyTransport *transport,
                           uint32_t flags,
                           struct flagcxNetSubmitContext *submit);

flagcxResult_t
flagcxKernelProxyReserveRequest(struct flagcxKernelProxyTransport *transport,
                                const struct flagcxNetSubmitContext *submit,
                                int peer, int stagingSlot, uint32_t *slot);
flagcxResult_t
flagcxKernelProxyPublishRequest(struct flagcxKernelProxyTransport *transport,
                                uint32_t slot, void *request,
                                flagcxResult_t completionResult);
flagcxResult_t
flagcxKernelProxyRequireGetFlush(struct flagcxKernelProxyTransport *transport,
                                 uint32_t slot, int dstMrIdx, uint64_t dstOff,
                                 size_t size, void *flushRecvComm);
flagcxResult_t flagcxKernelProxyPublishGetFlushPending(
    struct flagcxKernelProxyTransport *transport, uint32_t slot);
flagcxResult_t flagcxKernelProxyProgressRequest(
    struct flagcxKernelProxyTransport *transport, uint32_t slot,
    flagcxKernelProxyTestRequestFn testRequest,
    flagcxKernelProxyPostFlushFn postFlush, void *flushContext, int *ready,
    flagcxResult_t *completionResult);
flagcxResult_t
flagcxKernelProxyCancelRequest(struct flagcxKernelProxyTransport *transport,
                               uint32_t slot);
flagcxResult_t
flagcxKernelProxyAbortRequest(struct flagcxKernelProxyTransport *transport,
                              uint32_t slot, int *releasedStagingSlot);
flagcxResult_t
flagcxKernelProxyCompleteRequest(struct flagcxKernelProxyTransport *transport,
                                 uint32_t slot, flagcxResult_t result,
                                 uint32_t *advanced, int *releasedStagingSlot);
flagcxResult_t
flagcxKernelProxyCompleteImmediate(struct flagcxKernelProxyTransport *transport,
                                   const struct flagcxNetSubmitContext *submit,
                                   flagcxResult_t result, uint32_t *advanced);

flagcxResult_t flagcxKernelProxyAcquireStagingSlot(
    struct flagcxKernelProxyTransport *transport, int *slot);
flagcxResult_t flagcxKernelProxyReleaseStagingSlot(
    struct flagcxKernelProxyTransport *transport, int slot);

flagcxResult_t
flagcxKernelProxyQuery(struct flagcxKernelProxyTransport *transport,
                       uint64_t *nextSequence, uint32_t *inFlight,
                       flagcxResult_t *firstError);
flagcxResult_t
flagcxKernelProxyReleaseReady(struct flagcxKernelProxyTransport *transport,
                              const struct flagcxNetSubmitContext *submit,
                              int *ready, flagcxResult_t *firstError);

#endif // FLAGCX_KERNEL_PROXY_TRANSPORT_H_
