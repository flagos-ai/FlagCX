/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_P2P_ENGINE_CORE_H_
#define FLAGCX_P2P_ENGINE_CORE_H_

#include "net_transport.h"

#include <stdint.h>
#include <vector>

// One provider-neutral data operation. Provider backends translate the region
// handles and offsets into their native request format; the Engine core owns
// retry, aggregation, ordering, and completion state.
struct flagcxP2pTransferOp {
  uint64_t srcOffset = 0;
  uint64_t dstOffset = 0;
  size_t size = 0;
  void *srcMr = NULL;
  void *dstMr = NULL;
  uint64_t orderingKey = 0;
  uint32_t submitFlags = FLAGCX_NET_SUBMIT_DATA;
  uint64_t *laneMask = NULL;
};

typedef flagcxResult_t (*flagcxP2pTransferPostFn)(
    void *context, const struct flagcxP2pTransferOp *ops, uint32_t count,
    void **requests, flagcxTransportPostResult *post);
typedef flagcxResult_t (*flagcxP2pTransferTestFn)(void *context, void *request,
                                                  int *done);

struct flagcxP2pTransferBackend {
  void *context = NULL;
  flagcxP2pTransferPostFn post = NULL;
  flagcxP2pTransferTestFn test = NULL;
};

enum flagcxP2pTransferItemState {
  FLAGCX_P2P_TRANSFER_QUEUED = 0,
  FLAGCX_P2P_TRANSFER_INFLIGHT = 1,
  FLAGCX_P2P_TRANSFER_COMPLETE = 2,
};

struct flagcxP2pTransferItem {
  struct flagcxP2pTransferOp op;
  struct flagcxNetSubmitContext submit;
  void *request = NULL;
  uint32_t state = FLAGCX_P2P_TRANSFER_QUEUED;
};

struct flagcxP2pTransferStatus {
  uint32_t requested = 0;
  uint32_t submitted = 0;
  uint32_t completed = 0;
  uint32_t retired = 0;
  uint32_t pending = 0;
  uint32_t inFlight = 0;
  int done = 0;
  int releaseAllowed = 0;
  flagcxResult_t result = flagcxSuccess;
};

struct flagcxP2pTransfer {
  flagcxP2pTransfer() = default;
  flagcxP2pTransfer(const flagcxP2pTransfer &) = delete;
  flagcxP2pTransfer &operator=(const flagcxP2pTransfer &) = delete;
  flagcxP2pTransfer(flagcxP2pTransfer &&) = delete;
  flagcxP2pTransfer &operator=(flagcxP2pTransfer &&) = delete;

  struct flagcxP2pTransferBackend backend;
  std::vector<struct flagcxP2pTransferItem> items;
  std::vector<struct flagcxNetCompletionEntry> completionEntries;
  struct flagcxNetCompletionScoreboard scoreboard = {};
  struct flagcxNetReleaseGroup releaseGroup = {};
  flagcxTransportCredit credit = {};
  flagcxTransportRequest request = {};
  uint32_t maxPostBatch = 0;
  uint32_t submitted = 0;
  uint32_t completed = 0;
  flagcxResult_t terminalResult = flagcxSuccess;
  int initialized = 0;
};

flagcxResult_t
flagcxP2pTransferInit(struct flagcxP2pTransfer *transfer,
                      const struct flagcxP2pTransferBackend *backend,
                      const struct flagcxP2pTransferOp *ops, uint32_t count,
                      uint32_t maxInFlight, uint32_t maxPostBatch,
                      uint64_t groupId, uint64_t generation);

// Poll accepted requests and submit another prefix when credits are available.
// Transport backpressure is retained as queued work, not reported as failure.
flagcxResult_t
flagcxP2pTransferProgress(struct flagcxP2pTransfer *transfer,
                          struct flagcxP2pTransferStatus *status);

flagcxResult_t flagcxP2pTransferQuery(struct flagcxP2pTransfer *transfer,
                                      struct flagcxP2pTransferStatus *status);

// Reset is legal only after every accepted backend request has completed.
flagcxResult_t flagcxP2pTransferReset(struct flagcxP2pTransfer *transfer);

#endif // FLAGCX_P2P_ENGINE_CORE_H_
