#ifndef FLAGCX_WORK_H_
#define FLAGCX_WORK_H_

#include "flagcx.h"
#include <pthread.h>
#include <stdint.h>

enum flagcxWorkState {
  flagcxWorkPreparing = 0,
  flagcxWorkGroupPending,
  flagcxWorkRecorded,
  flagcxWorkNativePending,
  flagcxWorkChecking,
  flagcxWorkComplete,
  flagcxWorkUncertain
};
enum flagcxWorkGroupDomain { flagcxWorkNativeGroup = 0, flagcxWorkHeteroGroup };

struct flagcxHeteroComm;
struct flagcxNetAdaptor_latest;
struct flagcxWorkTracker;
struct flagcxWorkRecord {
  struct flagcxWorkRecord *next, *groupNext, *parent;
  struct flagcxWorkTracker *owner;
  flagcxEvent_t event;
  void *buffer;
  void *nativeRequest;
  struct flagcxNetAdaptor_latest *nativeAdaptor;
  flagcxStream_t stream; // Borrowed only until the submission boundary.
  enum flagcxWorkState state;
  flagcxResult_t operationResult, completionResult;
  uint64_t id;
  int pins, reusable;
};
struct flagcxWorkTracker {
  pthread_mutex_t mutex;
  int initialized, closed, submitting;
  uint64_t nextId;
  struct flagcxWorkRecord *head, *tail, *spare, *retired;
  unsigned int spareCount;
};
struct flagcxWorkFifoTarget {
  uint64_t target;
  int complete;
};
struct flagcxWorkSnapshot {
  uint64_t target;
  int initialized, rmaCaptured, rmaCount;
  uint64_t *rmaTargets;
  int fifoCaptured, fifoCount;
  struct flagcxWorkFifoTarget *fifoTargets;
};
struct flagcxWorkGroupBoundary {
  struct flagcxWorkTracker *owner; // NULL applies to every pending owner.
  flagcxStream_t stream;
  int submitted, ready;
};

// Prepare ownership before the first asynchronous side effect. Finish records
// the last-use event at the original submission boundary, including failures
// after a prefix was accepted. No event is appended during Drain.
flagcxResult_t flagcxWorkPrepare(struct flagcxWorkTracker *tracker,
                                 flagcxStream_t stream,
                                 struct flagcxWorkRecord **record);
// Internal children of a live host submission keep progressing after admission
// closes. The calling thread must still own the parent's Preparing record.
flagcxResult_t flagcxWorkPrepareAccepted(struct flagcxWorkTracker *tracker,
                                         flagcxStream_t stream,
                                         struct flagcxWorkRecord **record);
flagcxResult_t flagcxWorkPrepareHost(struct flagcxWorkTracker *tracker,
                                     struct flagcxWorkRecord **record);
struct flagcxWorkRecord *flagcxWorkSetCurrent(struct flagcxWorkRecord *record);
flagcxResult_t flagcxWorkReleaseBuffer(struct flagcxWorkRecord *record);
flagcxResult_t flagcxWorkFinishNative(struct flagcxWorkRecord *record,
                                      flagcxResult_t result);
flagcxResult_t flagcxWorkCancel(struct flagcxWorkRecord *record);
flagcxResult_t flagcxWorkFinish(struct flagcxWorkRecord *record,
                                flagcxStream_t stream, flagcxResult_t result,
                                enum flagcxWorkGroupDomain domain, int deferred,
                                int submissionKnown = 0);
flagcxResult_t flagcxWorkNativeGroupResult(flagcxResult_t result);
flagcxResult_t flagcxWorkGroupCompleteAtBoundary(
    enum flagcxWorkGroupDomain domain, flagcxResult_t result,
    const struct flagcxWorkGroupBoundary *boundary);
flagcxResult_t flagcxWorkGroupComplete(enum flagcxWorkGroupDomain domain,
                                       flagcxResult_t result);
void flagcxWorkGroupStart(enum flagcxWorkGroupDomain domain);
int flagcxWorkGroupDepth(enum flagcxWorkGroupDomain domain);
flagcxResult_t flagcxWorkGroupEnd(enum flagcxWorkGroupDomain domain,
                                  flagcxResult_t result);
void flagcxWorkRetain(struct flagcxWorkRecord *record);
void flagcxWorkDrop(struct flagcxWorkRecord *record);
flagcxResult_t flagcxWorkWaitRecord(struct flagcxWorkRecord *record);
flagcxResult_t flagcxWorkPollRecord(struct flagcxWorkRecord *record);
// Wait snapshots a finite prefix. Close additionally stops future admission.
flagcxResult_t
flagcxWorkWaitSnapshot(struct flagcxWorkTracker *tracker,
                       struct flagcxWorkSnapshot *snapshot = NULL);
flagcxResult_t flagcxWorkWait(struct flagcxWorkTracker *tracker);
flagcxResult_t flagcxWorkDrain(struct flagcxWorkTracker *tracker,
                               struct flagcxHeteroComm *comm);
flagcxResult_t flagcxWorkDrainRecord(struct flagcxWorkRecord *record,
                                     struct flagcxHeteroComm *comm);
flagcxResult_t flagcxCommDrainSnapshot(flagcxComm_t comm,
                                       struct flagcxWorkSnapshot *snapshot);
void flagcxWorkSnapshotRelease(struct flagcxWorkSnapshot *snapshot);
flagcxResult_t flagcxWorkRelease(struct flagcxWorkTracker *tracker);

int flagcxWorkHasCurrent(struct flagcxWorkTracker *tracker);

#endif
