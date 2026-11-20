#include "comm_lifecycle.h"
#include "adaptor.h"
#include "flagcx_kernel_internal.h"
#include "global_comm.h"
#include <stdlib.h>

flagcxResult_t flagcxCommRequireActive(flagcxComm_t comm) {
  if (comm == NULL)
    return flagcxInvalidArgument;
  return __atomic_load_n(&comm->finalizeState.lifecycle, __ATOMIC_ACQUIRE) ==
                 flagcxCommActive
             ? flagcxSuccess
             : flagcxInvalidUsage;
}

// Agree SDK shutdown scope and the lifetime of the lazy heterogeneous IPC
// transport domain at creation, before any rank can export a FIFO/PXN buffer.
flagcxResult_t flagcxCommInitFinalizePolicy(flagcxComm_t comm) {
  struct Policy {
    int nativeScope;
    int ipcRequired;
  };
  struct Policy *policies =
      (struct Policy *)calloc(comm->nranks, sizeof(*policies));
  flagcxResult_t common;
  flagcxResult_t result = flagcxConvergeBootstrapStatus(
      comm, policies != NULL ? flagcxSuccess : flagcxSystemError, &common);
  if (result == flagcxSuccess)
    result = common;
  if (result == flagcxSuccess) {
    policies[comm->rank].nativeScope =
        cclAdaptors[flagcxCCLAdaptorDevice]->finalizeScope;
    policies[comm->rank].ipcRequired =
        comm->internalBuffersNeedPeerClose || comm->heteroComm != NULL;
    result =
        bootstrapCollAllGather(comm->bootstrap, policies, sizeof(*policies));
    if (result == flagcxSuccess) {
      for (int rank = 0; rank < comm->nranks; rank++)
        if (policies[rank].nativeScope > comm->nativeFinalizeScope)
          comm->nativeFinalizeScope = policies[rank].nativeScope;
      for (int local = 0; local < comm->localRanks; local++)
        if (policies[comm->localRankToRank[local]].ipcRequired)
          comm->internalBuffersNeedPeerClose = true;
      comm->ipcClosePolicyConfirmed = true;
    }
  }
  free(policies);
  return result;
}

flagcxResult_t flagcxCommRequireIpcClose(flagcxComm_t comm, int localRequired) {
  if (comm == NULL)
    return flagcxInvalidArgument;
  // Creation has already agreed this sticky domain on every local rank. Do
  // not put another bootstrap socket exchange on cached UniRunner execution.
  // The required bit alone is insufficient after an uncertain declaration.
  if (comm->ipcClosePolicyConfirmed && comm->internalBuffersNeedPeerClose)
    return flagcxSuccess;
  localRequired |= comm->internalBuffersNeedPeerClose;
  // Single-rank/standalone contexts have no remote bootstrap owner. Production
  // multi-rank communicators always carry bootstrap and local-rank metadata.
  if (comm->bootstrap == NULL || comm->localRanks <= 1) {
    comm->internalBuffersNeedPeerClose = localRequired != 0;
    comm->ipcClosePolicyConfirmed = comm->localRanks <= 1;
    return flagcxSuccess;
  }
  // Status reduction selects a non-success if any participant supplies one.
  // InProgress is only a wire encoding of the declaration bit in this exchange.
  comm->ipcClosePolicyConfirmed = false;
  flagcxResult_t anyRequired = flagcxSuccess;
  flagcxResult_t result = bootstrapCollIntraNodeStatus(
      comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks,
      0x6e40, localRequired ? flagcxInProgress : flagcxSuccess, &anyRequired);
  if (result != flagcxSuccess) {
    // An uncertain declaration must never enable an unconfirmed local free.
    comm->internalBuffersNeedPeerClose = true;
    return result;
  }
  comm->internalBuffersNeedPeerClose = anyRequired == flagcxInProgress;
  comm->ipcClosePolicyConfirmed = true;
  return flagcxSuccess;
}

flagcxResult_t flagcxCommReclaim(flagcxComm_t comm) {
  if (flagcxCommLifecycleGroupDepth() != 0)
    return flagcxInvalidUsage;
  if (comm->finalizeState.commonPhase != flagcxFinalizeComplete) {
    flagcxResult_t result = flagcxCommFinalizeInternal(comm);
    if (result != flagcxSuccess)
      return result;
  }
  return flagcxCommDestroyLocal(comm);
}

struct flagcxFinalizeRequest {
  flagcxComm_t comm;
  struct flagcxFinalizeRequest *next;
};
static __thread int lifecycleGroupDepth;
static __thread flagcxResult_t lifecycleGroupError;
static __thread struct flagcxFinalizeRequest *finalizeRequests;

int flagcxCommLifecycleGroupDepth() { return lifecycleGroupDepth; }
void flagcxCommLifecycleGroupStart() { lifecycleGroupDepth++; }

flagcxResult_t flagcxCommFinalizeEnqueue(flagcxComm_t comm) {
  if (lifecycleGroupDepth == 0)
    return flagcxCommFinalizeInternal(comm);
  for (struct flagcxFinalizeRequest *p = finalizeRequests; p != NULL;
       p = p->next)
    if (p->comm == comm)
      return flagcxSuccess;
  struct flagcxFinalizeRequest *request =
      (struct flagcxFinalizeRequest *)calloc(1, sizeof(*request));
  if (request == NULL)
    return lifecycleGroupError = flagcxSystemError;
  request->comm = comm;
  request->next = NULL;
  struct flagcxFinalizeRequest **tail = &finalizeRequests;
  while (*tail != NULL)
    tail = &(*tail)->next;
  *tail = request;
  if (comm->finalizeState.commonPhase != flagcxFinalizeComplete) {
    __atomic_store_n(&comm->finalizeState.lifecycle, flagcxCommFinalizing,
                     __ATOMIC_RELEASE);
    __atomic_store_n(&comm->finalizeState.lastResult, flagcxInProgress,
                     __ATOMIC_RELEASE);
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxCommLifecycleGroupEnd(flagcxResult_t incoming) {
  if (lifecycleGroupDepth <= 0)
    return flagcxInvalidUsage;
  if (incoming != flagcxSuccess && lifecycleGroupError == flagcxSuccess)
    lifecycleGroupError = incoming;
  if (--lifecycleGroupDepth != 0)
    return lifecycleGroupError;
  flagcxResult_t completed =
      flagcxWorkGroupComplete(flagcxWorkNativeGroup, incoming);
  flagcxResult_t result =
      lifecycleGroupError != flagcxSuccess ? lifecycleGroupError : completed;
  lifecycleGroupError = flagcxSuccess;
  int execute = result == flagcxSuccess;
  // All callbacks stay on the application thread (including native wrapper
  // TLS). Callers must submit shared communicators in a compatible order.
  while (finalizeRequests != NULL) {
    struct flagcxFinalizeRequest *request = finalizeRequests;
    finalizeRequests = request->next;
    if (execute) {
      flagcxResult_t local = flagcxCommFinalizeInternal(request->comm);
      if (result == flagcxSuccess)
        result = local;
    } else {
      __atomic_store_n(&request->comm->finalizeState.lastResult, result,
                       __ATOMIC_RELEASE);
    }
    free(request);
  }
  return result;
}
