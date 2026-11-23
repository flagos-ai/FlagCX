// Like NCCL, Finalize retains the native communicator and its wrapper.
// Only Destroy closes the native context and releases the wrapper on success.
flagcxResult_t ppupcclAdaptorCommFinalize(flagcxInnerComm_t comm) {
  if (comm == NULL)
    return flagcxInvalidArgument;
  if (comm->base == NULL)
    return flagcxSuccess;
  const ppupcclSymbols *api = NULL;
  FLAGCXCHECK(ppupcclLoadApi(&api));
  INFO(FLAGCX_INIT, "Lifecycle PPU comm=%p native-finalize begin", comm);
  flagcxResult_t result =
      ppupcclResultToFlagcx(api->pcclCommFinalize(comm->base));
  INFO(FLAGCX_INIT, "Lifecycle PPU comm=%p native-finalize end result=%d", comm,
       (int)result);
  return result;
}

flagcxResult_t ppupcclAdaptorCommDestroy(flagcxInnerComm_t comm) {
  if (comm == NULL)
    return flagcxInvalidArgument;
  if (comm->base != NULL) {
    const ppupcclSymbols *api = NULL;
    FLAGCXCHECK(ppupcclLoadApi(&api));
    INFO(FLAGCX_INIT, "Lifecycle PPU comm=%p native-destroy begin", comm);
    flagcxResult_t result =
        ppupcclResultToFlagcx(api->pcclCommDestroy(comm->base));
    INFO(FLAGCX_INIT, "Lifecycle PPU comm=%p native-destroy end result=%d",
         comm, (int)result);
    // Errors and InProgress both retain the owner for a local retry.
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
  pcclResult_t status = pcclSuccess;
  flagcxResult_t result =
      ppupcclResultToFlagcx(api->pcclCommGetAsyncError(comm->base, &status));
  if (result == flagcxSuccess)
    *asyncError = ppupcclResultToFlagcx(status);
  return result;
}
