#include "comm_lifecycle.h"
#include "adaptor.h"
#include "flagcx_kernel_internal.h"
#include "global_comm.h"

flagcxResult_t flagcxCommRequireActive(flagcxComm_t comm) {
  if (comm == NULL)
    return flagcxInvalidArgument;
  if (comm->initRollbackPending || comm->pluginRollbackOnly)
    return flagcxInvalidUsage;
  return __atomic_load_n(&comm->finalizeState.lifecycle, __ATOMIC_ACQUIRE) ==
                 flagcxCommActive
             ? flagcxSuccess
             : flagcxInvalidUsage;
}

// Capabilities belong to the selected built-in adaptor, not to a plugin's
// name or a speculative SDK symbol. Unknown plugins never get this fallback.
static bool builtinCompatFinalize(flagcxCCLAdaptor *adaptor) {
#if defined(USE_ASCEND_ADAPTOR)
  return adaptor == &hcclAdaptor;
#elif defined(USE_CAMBRICON_ADAPTOR)
  return adaptor == &cnclAdaptor;
#elif defined(USE_ENFLAME_ADAPTOR)
  return adaptor == &ecclAdaptor;
#elif defined(USE_MUSA_ADAPTOR)
  return adaptor == &musa_mcclAdaptor;
#elif defined(USE_SUNRISE_ADAPTOR)
  return adaptor == &pcclAdaptor;
#elif defined(USE_KUNLUNXIN_ADAPTOR)
  return adaptor == &xcclAdaptor;
#else
  (void)adaptor;
  return false;
#endif
}

flagcxResult_t flagcxCommInitFinalizePolicy(flagcxComm_t comm) {
  if (comm == NULL)
    return flagcxInvalidArgument;
  comm->allowNativeDestroyFinalize =
      builtinCompatFinalize(cclAdaptors[flagcxCCLAdaptorDevice]);
  struct Policy {
    int scope, required;
  };
  Policy local = {
      comm->nativeFinalizeScope == flagcxNativeFinalizeComm ||
          comm->allowNativeDestroyFinalize || comm->hostComm != NULL,
      comm->internalBuffersNeedPeerClose || comm->heteroComm != NULL};
  if (comm->nranks <= 1) {
    comm->nativeFinalizeScope = (flagcxNativeFinalizeScope)local.scope;
    comm->internalBuffersNeedPeerClose = local.required;
    comm->ipcClosePolicyConfirmed = true;
    return flagcxSuccess;
  }
  if (comm->bootstrap == NULL || comm->localRankToRank == NULL ||
      comm->localRanks <= 0)
    return flagcxInvalidUsage;
  Policy *policies = (Policy *)calloc(comm->nranks, sizeof(Policy));
  flagcxResult_t common,
      result = flagcxConvergeBootstrapStatus(
          comm, policies != NULL ? flagcxSuccess : flagcxSystemError, &common);
  if (result == flagcxSuccess)
    result = common;
  if (result == flagcxSuccess) {
    policies[comm->rank] = local;
    result = bootstrapCollAllGather(comm->bootstrap, policies, sizeof(Policy));
  }
  if (result == flagcxSuccess) {
    for (int rank = 0; rank < comm->nranks; ++rank)
      if (policies[rank].scope)
        comm->nativeFinalizeScope = flagcxNativeFinalizeComm;
    for (int rank = 0; rank < comm->localRanks; ++rank)
      comm->internalBuffersNeedPeerClose |=
          policies[comm->localRankToRank[rank]].required != 0;
    comm->ipcClosePolicyConfirmed = true;
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
  if (comm->localRanks <= 1 && comm->nranks <= 1) {
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
