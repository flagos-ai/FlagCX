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
  flagcxFinalizeRelayImports,
  flagcxFinalizeProxy,
  flagcxFinalizeImports,
  flagcxFinalizeVendors,
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

flagcxResult_t flagcxCommInitFinalizePolicy(flagcxComm_t comm);
// Collective in the node's communicator group, called before IPC creation.
// The declaration stays set until Destroy, including failed/partial creates.
flagcxResult_t flagcxCommRequireIpcClose(flagcxComm_t comm, int localRequired);
flagcxResult_t flagcxCommReclaim(flagcxComm_t comm);
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
