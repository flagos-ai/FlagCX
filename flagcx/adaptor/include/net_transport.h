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

// Canonical transport-neutral names.  The flagcxNet* spellings below remain
// compatibility interfaces for existing adaptor implementations; new core
// transports should use these aliases so SHM/IPC, verbs, BAREX, and Socket can
// share request and scheduling machinery without pretending every backend is
// a network device.
typedef struct flagcxNetLane flagcxTransportLane;
typedef struct flagcxNetLaneSet flagcxTransportLaneSet;

struct flagcxNetLaneSet {
  uint32_t count;
  uint32_t unorderedCursor;
};

enum flagcxNetLaneMode {
  FLAGCX_NET_LANE_ORDERED = 0,
  FLAGCX_NET_LANE_UNORDERED = 1,
};
typedef enum flagcxNetLaneMode flagcxTransportLaneMode;

#define FLAGCX_TRANSPORT_LANE_ORDERED FLAGCX_NET_LANE_ORDERED
#define FLAGCX_TRANSPORT_LANE_UNORDERED FLAGCX_NET_LANE_UNORDERED

flagcxResult_t flagcxTransportSelectLane(flagcxTransportLaneSet *lanes,
                                         flagcxTransportLaneMode mode,
                                         uint64_t orderingKey,
                                         uint32_t *laneIndex);
flagcxResult_t flagcxTransportCommitLane(flagcxTransportLaneSet *lanes,
                                         flagcxTransportLaneMode mode,
                                         uint32_t laneIndex);

// Ordered submissions deterministically map orderingKey to one lane. Existing
// compatibility paths pass zero, which maps to lane zero exactly as today.
flagcxResult_t flagcxNetSelectLane(struct flagcxNetLaneSet *lanes,
                                   enum flagcxNetLaneMode mode,
                                   uint64_t orderingKey, uint32_t *laneIndex);
flagcxResult_t flagcxNetCommitLane(struct flagcxNetLaneSet *lanes,
                                   enum flagcxNetLaneMode mode,
                                   uint32_t laneIndex);

// Submission identity used by transport consumers that need deterministic
// lane selection and completion ordering.  The structure is internal to
// FlagCX and deliberately does not extend the net adaptor ABI.
struct flagcxNetSubmitContext {
  uint64_t orderingKey;
  uint64_t groupId;
  uint64_t generation;
  uint64_t sequence;
  uint32_t flags;
  // Optional transport-neutral diagnostics sink. Backends OR the physical
  // lane index used for an accepted data post; logical-lane transports may
  // leave it zero.
  uint64_t *laneMask;
};

enum flagcxNetSubmitFlags {
  FLAGCX_NET_SUBMIT_DATA = 1u << 0,
  FLAGCX_NET_SUBMIT_RELEASE = 1u << 1,
  FLAGCX_NET_SUBMIT_INDEPENDENT = 1u << 2,
};

// The adaptor ABI stays unchanged. Core transport consumers bracket an
// adaptor submission with this thread-local context, allowing in-tree
// backends to select an ordering domain without adding callback parameters.
flagcxResult_t
flagcxNetSetSubmitContext(const struct flagcxNetSubmitContext *context);
flagcxResult_t
flagcxNetGetSubmitContext(struct flagcxNetSubmitContext *context);
void flagcxNetClearSubmitContext(void);

enum flagcxNetReleaseGroupState {
  FLAGCX_NET_RELEASE_GROUP_OPEN = 0,
  FLAGCX_NET_RELEASE_GROUP_SEALED = 1,
  FLAGCX_NET_RELEASE_GROUP_COMPLETE = 2,
};

// A release group is a completion gate, not an API submission group.  It can
// outlive flagcxGroupEnd() and is updated by transport progress threads.  A
// release is allowed only after the group is sealed and every tracked member
// completes successfully.
struct flagcxNetReleaseGroup {
  uint64_t groupId;
  uint64_t generation;
  uint32_t state;
  uint32_t pending;
  uint32_t members;
  uint32_t releaseAllowed;
  flagcxResult_t firstError;
  uint32_t lock;
};

enum flagcxNetCompletionEntryState {
  FLAGCX_NET_COMPLETION_ENTRY_FREE = 0,
  FLAGCX_NET_COMPLETION_ENTRY_PENDING = 1,
  FLAGCX_NET_COMPLETION_ENTRY_COMPLETE = 2,
};

struct flagcxNetCompletionEntry {
  struct flagcxNetSubmitContext context;
  struct flagcxNetReleaseGroup *releaseGroup;
  flagcxResult_t result;
  uint32_t state;
};

// Fixed-capacity completion scoreboard.  Callers own the entry storage, so
// progress paths never allocate.  Entries are indexed by sequence modulo
// capacity; completions may arrive out of order, while nextSequence advances
// only over a contiguous completed prefix.
struct flagcxNetCompletionScoreboard {
  struct flagcxNetCompletionEntry *entries;
  uint32_t capacity;
  uint32_t inFlight;
  uint64_t generation;
  uint64_t nextSequence;
  flagcxResult_t firstError;
  uint32_t lock;
};

flagcxResult_t flagcxNetReleaseGroupInit(struct flagcxNetReleaseGroup *group,
                                         uint64_t groupId, uint64_t generation);
flagcxResult_t flagcxNetReleaseGroupSeal(struct flagcxNetReleaseGroup *group);
flagcxResult_t flagcxNetReleaseGroupTest(struct flagcxNetReleaseGroup *group,
                                         int *done, int *releaseAllowed);
flagcxResult_t flagcxNetReleaseGroupReset(struct flagcxNetReleaseGroup *group,
                                          uint64_t groupId,
                                          uint64_t generation);

flagcxResult_t flagcxNetCompletionScoreboardInit(
    struct flagcxNetCompletionScoreboard *scoreboard,
    struct flagcxNetCompletionEntry *entries, uint32_t capacity,
    uint64_t generation, uint64_t initialSequence);
flagcxResult_t flagcxNetCompletionScoreboardReset(
    struct flagcxNetCompletionScoreboard *scoreboard, uint64_t generation,
    uint64_t initialSequence);
flagcxResult_t
flagcxNetTrackSubmit(struct flagcxNetCompletionScoreboard *scoreboard,
                     const struct flagcxNetSubmitContext *context,
                     struct flagcxNetReleaseGroup *releaseGroup);
// Cancel a submission that has not been published to a transport. This is
// used only to roll back a tail of descriptors prepared transactionally by an
// upper layer; completed entries cannot be cancelled.
flagcxResult_t
flagcxNetTrackCancel(struct flagcxNetCompletionScoreboard *scoreboard,
                     const struct flagcxNetSubmitContext *context);
flagcxResult_t
flagcxNetTrackCompletion(struct flagcxNetCompletionScoreboard *scoreboard,
                         const struct flagcxNetSubmitContext *context,
                         flagcxResult_t result, uint32_t *advanced);
flagcxResult_t flagcxNetCompletionScoreboardQuery(
    struct flagcxNetCompletionScoreboard *scoreboard, uint64_t *nextSequence,
    uint32_t *inFlight, flagcxResult_t *firstError);

// Progress-thread-private visibility state for one GET ordering domain.  GET
// sequence numbers are independent of the transport completion sequence, so
// intervening PUTs do not prevent one flush from covering multiple GETs.
// Callers advance dataCompletedGetSequence only over a contiguous prefix.
struct flagcxNetGetVisibilityDomain {
  uint64_t orderingKey;
  uint64_t issuedGetSequence;
  uint64_t dataCompletedGetSequence;
  uint64_t flushTargetGetSequence;
  uint64_t visibleGetSequence;
  void *flushRequest;
  flagcxResult_t flushResult;
  int peer;
  uint8_t inUse;
};

flagcxResult_t
flagcxNetGetVisibilityDomainInit(struct flagcxNetGetVisibilityDomain *domain,
                                 int peer, uint64_t orderingKey);
flagcxResult_t
flagcxNetGetVisibilityIssue(struct flagcxNetGetVisibilityDomain *domain,
                            uint64_t *getSequence);
flagcxResult_t
flagcxNetGetVisibilityCancelIssue(struct flagcxNetGetVisibilityDomain *domain,
                                  uint64_t getSequence);
flagcxResult_t
flagcxNetGetVisibilityRemoveIssue(struct flagcxNetGetVisibilityDomain *domain,
                                  uint64_t getSequence);
flagcxResult_t
flagcxNetGetVisibilityAdvanceData(struct flagcxNetGetVisibilityDomain *domain,
                                  uint64_t dataCompletedGetSequence);
flagcxResult_t
flagcxNetGetVisibilityBeginFlush(struct flagcxNetGetVisibilityDomain *domain,
                                 uint64_t *flushTarget);
flagcxResult_t flagcxNetGetVisibilityAdvanceVisible(
    struct flagcxNetGetVisibilityDomain *domain, uint64_t visibleGetSequence);
flagcxResult_t
flagcxNetGetVisibilityCompleteFlush(struct flagcxNetGetVisibilityDomain *domain,
                                    flagcxResult_t result);

// A bounded credit counter for SQ entries, callback slots, or backend request
// objects. Exhaustion is transient backpressure and is reported as
// flagcxInProgress; invalid release is a permanent caller error.
struct flagcxNetCredit {
  uint32_t capacity;
  uint32_t inUse;
};
typedef struct flagcxNetCredit flagcxTransportCredit;

flagcxResult_t flagcxTransportCreditInit(flagcxTransportCredit *credit,
                                         uint32_t capacity);
flagcxResult_t flagcxTransportCreditAcquire(flagcxTransportCredit *credit,
                                            uint32_t count);
flagcxResult_t flagcxTransportCreditRelease(flagcxTransportCredit *credit,
                                            uint32_t count);
flagcxResult_t
flagcxTransportCreditAvailable(const flagcxTransportCredit *credit,
                               uint32_t *available);

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
typedef enum flagcxNetRequestState flagcxTransportRequestState;

#define FLAGCX_TRANSPORT_REQUEST_FREE FLAGCX_NET_REQUEST_FREE
#define FLAGCX_TRANSPORT_REQUEST_PENDING FLAGCX_NET_REQUEST_PENDING
#define FLAGCX_TRANSPORT_REQUEST_COMPLETE FLAGCX_NET_REQUEST_COMPLETE

// Common request state only. Backends keep their native progress metadata,
// such as per-CQ event counts, UCP workers, and callback ownership.
struct flagcxNetRequestCore {
  uint32_t state;
  uint32_t pending;
  flagcxResult_t result;
};
typedef struct flagcxNetRequestCore flagcxTransportRequest;

void flagcxTransportRequestInit(flagcxTransportRequest *request);
flagcxResult_t flagcxTransportRequestAcquire(flagcxTransportRequest *request);
flagcxResult_t flagcxTransportRequestAddPending(flagcxTransportRequest *request,
                                                uint32_t count);
flagcxResult_t flagcxTransportRequestComplete(flagcxTransportRequest *request,
                                              uint32_t count,
                                              flagcxResult_t result);
flagcxResult_t flagcxTransportRequestFinish(flagcxTransportRequest *request,
                                            flagcxResult_t result);
flagcxResult_t flagcxTransportRequestTest(const flagcxTransportRequest *request,
                                          int *done);
flagcxResult_t flagcxTransportRequestRelease(flagcxTransportRequest *request);

// Normalize an asynchronous backend completion without losing terminal
// errors. Pending is progress, not success; all other non-success results are
// permanent and must be propagated to the proxy's async result.
flagcxResult_t flagcxTransportClassifyCompletion(flagcxResult_t result,
                                                 int *completed);

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
typedef struct flagcxNetPostResult flagcxTransportPostResult;

flagcxResult_t flagcxTransportPostResultInit(flagcxTransportPostResult *post,
                                             int requested, int accepted,
                                             flagcxResult_t result);

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
