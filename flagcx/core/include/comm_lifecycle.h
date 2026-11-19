#ifndef FLAGCX_COMM_LIFECYCLE_H_
#define FLAGCX_COMM_LIFECYCLE_H_

#include "flagcx.h"

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

// Admission is separate from handle validation: cleanup/status APIs remain
// usable after admission has stopped, including failed finalization retries.
flagcxResult_t flagcxCommRequireActive(flagcxComm_t comm);
flagcxResult_t flagcxCommFinalizeInternal(flagcxComm_t comm);
flagcxResult_t flagcxCommDestroyLocal(flagcxComm_t comm);

#endif
