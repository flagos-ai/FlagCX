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
enum flagcxNativeFinalizeScope {
  flagcxNativeFinalizeLocal = 0,
  flagcxNativeFinalizeComm
};
struct flagcxCommFinalizeState {
  enum flagcxCommLifecycle lifecycle;
  enum flagcxFinalizePhase commonPhase;
  unsigned int localDoneMask;
  flagcxResult_t lastResult;
  int running;
  unsigned int vendorIndex;
  int vendorFinalizeStarted;
  int vendorDestroyStarted;
  void **vendorOwners;
  unsigned int vendorCount;
  int vendorCatalogReady;
  int hostFinalized;
  int abortStarted, abortComplete;
  unsigned int abortIndex;
};
enum flagcxHeteroFinalizePhase {
  flagcxHeteroFinalizeDrain = 0,
  flagcxHeteroFinalizeImports,
  flagcxHeteroFinalizeProxy,
  flagcxHeteroFinalizeComplete
};
struct flagcxHeteroFinalizeState {
  int started, running, rollbackPending, importsClosedConfirmed;
  enum flagcxHeteroFinalizePhase phase;
  unsigned int localDoneMask;
};

flagcxResult_t flagcxCommRequireActive(flagcxComm_t comm);
flagcxResult_t flagcxCommInitFinalizePolicy(flagcxComm_t comm);
flagcxResult_t flagcxCommRequireIpcClose(flagcxComm_t comm, int required);
#endif
