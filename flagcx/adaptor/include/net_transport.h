/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_NET_TRANSPORT_H_
#define FLAGCX_NET_TRANSPORT_H_

#include "flagcx.h"
#include "flagcx_net_adaptor.h"

#include <stddef.h>
#include <stdint.h>

struct flagcxOneSideHandleInfo;

// Transport-neutral lane identity. Backend-specific lanes embed this object
// and retain their native endpoint (QP, XChannel, socket, ...) separately.
struct flagcxNetLane {
  uint32_t index;
  int localDevIndex;
  int remoteDevIndex;
};

struct flagcxNetLaneSet {
  uint32_t count;
  uint32_t unorderedCursor;
};

enum flagcxNetLaneMode {
  FLAGCX_NET_LANE_ORDERED = 0,
  FLAGCX_NET_LANE_UNORDERED = 1,
};

// orderingKey is reserved for a future order-domain identifier. Existing
// compatibility paths pass zero, which maps to lane zero exactly as today.
flagcxResult_t flagcxNetSelectLane(struct flagcxNetLaneSet *lanes,
                                   enum flagcxNetLaneMode mode,
                                   uint64_t orderingKey, uint32_t *laneIndex);
flagcxResult_t flagcxNetCommitLane(struct flagcxNetLaneSet *lanes,
                                   enum flagcxNetLaneMode mode,
                                   uint32_t laneIndex);

// A bounded credit counter for SQ entries, callback slots, or backend request
// objects. Exhaustion is transient backpressure and is reported as
// flagcxInProgress; invalid release is a permanent caller error.
struct flagcxNetCredit {
  uint32_t capacity;
  uint32_t inUse;
};

flagcxResult_t flagcxNetCreditInit(struct flagcxNetCredit *credit,
                                   uint32_t capacity);
flagcxResult_t flagcxNetCreditAcquire(struct flagcxNetCredit *credit,
                                      uint32_t count);
flagcxResult_t flagcxNetCreditRelease(struct flagcxNetCredit *credit,
                                      uint32_t count);
flagcxResult_t flagcxNetCreditAvailable(const struct flagcxNetCredit *credit,
                                        uint32_t *available);

enum flagcxNetRequestState {
  FLAGCX_NET_REQUEST_FREE = 0,
  FLAGCX_NET_REQUEST_PENDING = 1,
  FLAGCX_NET_REQUEST_COMPLETE = 2,
};

// Common request state only. Backends keep their native progress metadata,
// such as per-CQ event counts, UCP workers, and callback ownership.
struct flagcxNetRequestCore {
  uint32_t state;
  uint32_t pending;
  flagcxResult_t result;
};

void flagcxNetRequestCoreInit(struct flagcxNetRequestCore *core);
flagcxResult_t flagcxNetRequestCoreAcquire(struct flagcxNetRequestCore *core);
flagcxResult_t flagcxNetRequestCoreAddPending(struct flagcxNetRequestCore *core,
                                              uint32_t count);
flagcxResult_t flagcxNetRequestCoreComplete(struct flagcxNetRequestCore *core,
                                            uint32_t count,
                                            flagcxResult_t result);
flagcxResult_t flagcxNetRequestCoreFinish(struct flagcxNetRequestCore *core,
                                          flagcxResult_t result);
flagcxResult_t flagcxNetRequestCoreTest(const struct flagcxNetRequestCore *core,
                                        int *done);
flagcxResult_t flagcxNetRequestCoreRelease(struct flagcxNetRequestCore *core);

struct flagcxNetPostResult {
  flagcxResult_t result;
  int requested;
  int accepted;
};

flagcxResult_t flagcxNetPostResultInit(struct flagcxNetPostResult *post,
                                       int requested, int accepted,
                                       flagcxResult_t result);

struct flagcxNetResolvedRange {
  uintptr_t address;
  size_t size;
  const struct flagcxNetMrInfo *mrInfo;
};

flagcxResult_t
flagcxNetResolveOneSideRange(const struct flagcxOneSideHandleInfo *info,
                             int rank, uint64_t offset, size_t size,
                             struct flagcxNetResolvedRange *range);

typedef flagcxResult_t (*flagcxNetTestRequestFn)(void *request, int *done,
                                                 int *sizes);
flagcxResult_t flagcxNetTestBatchCommon(void **requests, int nRequests,
                                        int *doneFlags, int *doneCount,
                                        flagcxNetTestRequestFn testFn);

#endif // FLAGCX_NET_TRANSPORT_H_
