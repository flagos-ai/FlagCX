#ifndef FLAGCX_WORK_COMPLETION_H_
#define FLAGCX_WORK_COMPLETION_H_

#include "flagcx.h"
#include "flagcx_device_adaptor.h"
#include <pthread.h>

enum flagcxWorkState {
  flagcxWorkPreparing = 0,
  flagcxWorkGroupPending,
  flagcxWorkRecorded,
  flagcxWorkChecking,
  flagcxWorkComplete,
  flagcxWorkUncertain,
  // No completion event was inserted into the graph. The caller must finish
  // and stop replaying external graphs before Finalize, like Device API work.
  flagcxWorkExternalGraph
};
enum flagcxWorkGroupDomain { flagcxWorkNativeGroup = 0, flagcxWorkHeteroGroup };

// Borrowed only until the outermost heterogeneous GroupEnd returns.
struct flagcxWorkGroupBoundary {
  flagcxStream_t stream;
  int submitted;
  int ready;
};

struct flagcxWorkRecord {
  struct flagcxWorkRecord *next;
  struct flagcxWorkRecord *groupNext;
  struct flagcxWorkTracker *owner;
  flagcxEvent_t event;
  flagcxStream_t stream; // Only used at the submission/group boundary.
  enum flagcxWorkState state;
  flagcxResult_t error;
  int reusable;
  int pins;
  flagcxStreamCaptureStatus_t captureStatus;
};
struct flagcxWorkTracker {
  pthread_mutex_t mutex;
  int initialized;
  int closed;
  struct flagcxWorkRecord *head, *tail, *spare, *retired;
};

// Internal ownership helpers. Prepare before submitting; Finish captures the
// operation boundary, even after a partially successful submission. A failed
// event record is never repaired by recording at Finalize's stream tail.
flagcxResult_t flagcxWorkPrepare(struct flagcxWorkTracker *tracker,
                                 flagcxStream_t stream, int allowCapture,
                                 struct flagcxWorkRecord **record);
// Cancel only a prepared record for which no work has been submitted. Event
// ownership stays in the tracker for retryable local release; NULL is a no-op.
flagcxResult_t flagcxWorkCancel(struct flagcxWorkRecord *record);
flagcxResult_t flagcxWorkFinish(struct flagcxWorkRecord *record,
                                flagcxStream_t stream, flagcxResult_t result,
                                enum flagcxWorkGroupDomain domain,
                                int deferred);
flagcxResult_t flagcxWorkGroupComplete(enum flagcxWorkGroupDomain domain,
                                       flagcxResult_t result);
flagcxResult_t flagcxWorkGroupCompleteAtBoundary(
    enum flagcxWorkGroupDomain domain, flagcxResult_t result,
    const struct flagcxWorkGroupBoundary *boundary);
void flagcxWorkRetain(struct flagcxWorkRecord *record);
void flagcxWorkDrop(struct flagcxWorkRecord *record);
flagcxResult_t flagcxWorkWaitRecord(struct flagcxWorkRecord *record);
flagcxResult_t flagcxWorkWait(struct flagcxWorkTracker *tracker);
flagcxResult_t flagcxWorkRelease(struct flagcxWorkTracker *tracker);

#endif
