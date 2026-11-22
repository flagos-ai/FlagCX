// Instantiated with the real PCCL function table in ppu_pccl_adaptor.cc and
// native shims in the adaptor suite. Only successful native Destroy releases
// the base; the wrapper remains owned by FlagCX until local Destroy.
enum ppupcclClosePhase {
  ppupcclCloseOpen = 0,
  ppupcclCloseFinalizing,
  ppupcclCloseFinalized,
  ppupcclCloseComplete
};

static flagcxResult_t ppupcclProgressClose(flagcxInnerComm_t comm,
                                           const ppupcclSymbols *api) {
  flagcxResult_t result;
  if (comm->closePhase == ppupcclCloseOpen) {
    INFO(FLAGCX_INIT, "Lifecycle PPU comm=%p native-finalize begin", comm);
    result = ppupcclResultToFlagcx(api->pcclCommFinalize(comm->base));
    INFO(FLAGCX_INIT, "Lifecycle PPU comm=%p native-finalize end result=%d",
         comm, (int)result);
    if (result == flagcxInProgress)
      comm->closePhase = ppupcclCloseFinalizing;
    else if (result == flagcxSuccess)
      comm->closePhase = ppupcclCloseFinalized;
    else
      return result;
    if (result == flagcxInProgress)
      return result;
  }
  if (comm->closePhase == ppupcclCloseFinalizing) {
    pcclResult_t status = pcclSuccess;
    INFO(FLAGCX_INIT, "Lifecycle PPU comm=%p native-async-query begin", comm);
    result =
        ppupcclResultToFlagcx(api->pcclCommGetAsyncError(comm->base, &status));
    INFO(FLAGCX_INIT, "Lifecycle PPU comm=%p native-async-query end result=%d",
         comm, (int)result);
    if (result != flagcxSuccess)
      return result;
    result = ppupcclResultToFlagcx(status);
    if (result != flagcxSuccess)
      return result;
    comm->closePhase = ppupcclCloseFinalized;
  }
  if (comm->closePhase == ppupcclCloseFinalized) {
    INFO(FLAGCX_INIT, "Lifecycle PPU comm=%p native-destroy begin", comm);
    result = ppupcclResultToFlagcx(api->pcclCommDestroy(comm->base));
    INFO(FLAGCX_INIT, "Lifecycle PPU comm=%p native-destroy end result=%d",
         comm, (int)result);
    // Retain the native owner on any non-success, just as direct Destroy
    // does. Resume this close instead of restarting native Finalize. Never
    // query a native handle after a successful destructive call.
    if (result != flagcxSuccess)
      return result;
    comm->base = NULL;
    comm->closePhase = ppupcclCloseComplete;
  }
  return flagcxSuccess;
}

flagcxResult_t ppupcclAdaptorCommFinalize(flagcxInnerComm_t comm) {
  if (comm == NULL)
    return flagcxInvalidArgument;
  if (comm->base == NULL)
    return flagcxSuccess;
  const ppupcclSymbols *api = NULL;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  return ppupcclProgressClose(comm, api);
}

flagcxResult_t ppupcclAdaptorCommDestroy(flagcxInnerComm_t comm) {
  if (comm == NULL)
    return flagcxInvalidArgument;
  if (comm->base != NULL) {
    const ppupcclSymbols *api = NULL;
    FLAGCXCHECK(ppupcclLoadApi(&api));
    // Preserve direct adaptor Destroy for initialization rollback. A normally
    // finalized wrapper already has no native base and requires no peer.
    flagcxResult_t result =
        ppupcclResultToFlagcx(api->pcclCommDestroy(comm->base));
    if (result != flagcxSuccess)
      return result;
    comm->base = NULL;
  }
  free(comm);
  return flagcxSuccess;
}

flagcxResult_t ppupcclAdaptorCommAbort(flagcxInnerComm_t comm) {
  if (comm == NULL)
    return flagcxInvalidArgument;
  if (comm->base == NULL)
    return flagcxSuccess;
  const ppupcclSymbols *api = NULL;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  return ppupcclResultToFlagcx(api->pcclCommAbort(comm->base));
}

flagcxResult_t ppupcclAdaptorCommGetAsyncError(flagcxInnerComm_t comm,
                                               flagcxResult_t *asyncError) {
  if (comm == NULL || asyncError == NULL)
    return flagcxInvalidArgument;
  if (comm->base == NULL) {
    *asyncError = flagcxSuccess;
    return flagcxSuccess;
  }
  const ppupcclSymbols *api = NULL;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  if (comm->closePhase != ppupcclCloseOpen) {
    // The outer finalizer polls here after InProgress. Native Finalize alone
    // is insufficient: only report completion after native Destroy succeeds.
    *asyncError = ppupcclProgressClose(comm, api);
    return flagcxSuccess;
  }
  pcclResult_t status = pcclSuccess;
  flagcxResult_t result =
      ppupcclResultToFlagcx(api->pcclCommGetAsyncError(comm->base, &status));
  if (result == flagcxSuccess)
    *asyncError = ppupcclResultToFlagcx(status);
  return result;
}
