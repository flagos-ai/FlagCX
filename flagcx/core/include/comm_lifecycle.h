#ifndef FLAGCX_COMM_LIFECYCLE_H_
#define FLAGCX_COMM_LIFECYCLE_H_

#include "flagcx.h"
#include <pthread.h>

enum flagcxCommLifecycle {
  flagcxCommActive = 0,
  flagcxCommFinalizing,
  flagcxCommFinalized,
  flagcxCommDestroying
};

enum flagcxFinalizePhase {
  flagcxFinalizeDrain = 0,
  flagcxFinalizeTransport,
  flagcxFinalizeOwners,
  flagcxFinalizeProxy,
  flagcxFinalizeImports,
  flagcxFinalizeComplete
};

enum flagcxDestroyPhase {
  flagcxDestroyBuffers = 0,
  flagcxDestroyVendors,
  flagcxDestroyHetero,
  flagcxDestroyHost,
  flagcxDestroyBootstrap,
  flagcxDestroyTuner,
  flagcxDestroyNetPlugin,
  flagcxDestroyCclPlugin,
  flagcxDestroyDevicePlugin,
  flagcxDestroyComplete
};

struct flagcxCommFinalizeState {
  enum flagcxCommLifecycle lifecycle;
  enum flagcxFinalizePhase commonPhase;
  unsigned int localDoneMask;
  flagcxResult_t lastResult;
  int running;
  int vendorFinalizeStarted;
  int vendorFinalizeComplete;
  unsigned int vendorIndex;
};

// One communicator clique's ranks in this process; independent cliques never
// share a destruction counter. The retry owner is kept alive until the end.
struct flagcxIntraProcLifecycle {
  pthread_mutex_t mutex;
  unsigned int memberCount;
  unsigned int destroyRequested;
  int reclaiming;
  flagcxComm_t members;
  flagcxComm_t retryOwner;
};

typedef flagcxResult_t (*flagcxLifecycleFn)(flagcxComm_t);
typedef int (*flagcxLifecycleThreadCreate)(pthread_t *, const pthread_attr_t *,
                                           void *(*)(void *), void *);

flagcxResult_t flagcxCommMakeIntraProcGroup(flagcxComm_t *members, int count);
flagcxResult_t flagcxCommInitIntraProcGroup(flagcxComm_t comm);
flagcxResult_t flagcxCommReclaim(flagcxComm_t comm);
flagcxResult_t flagcxCommRunLifecycleJobs(flagcxComm_t *members, int count,
                                          flagcxLifecycleFn fn,
                                          flagcxLifecycleThreadCreate create);
void flagcxCommLifecycleGroupStart();
flagcxResult_t flagcxCommLifecycleGroupEnd(flagcxResult_t result);
int flagcxCommLifecycleGroupDepth();
flagcxResult_t flagcxCommFinalizeEnqueue(flagcxComm_t comm);

// Admission is separate from handle validation: cleanup/status APIs remain
// usable after admission has stopped, including failed finalization retries.
flagcxResult_t flagcxCommRequireActive(flagcxComm_t comm);
flagcxResult_t flagcxCommFinalizeInternal(flagcxComm_t comm);
flagcxResult_t flagcxCommDestroyLocal(flagcxComm_t comm);

#endif
