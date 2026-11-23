// Instantiated by NCCL-compatible adaptors and the host lifecycle shims.
// Finalize retains the wrapper; only a completed native Destroy releases it.
flagcxResult_t FLAGCX_NCCL_FINALIZE(flagcxInnerComm_t comm) {
  if (comm == NULL)
    return flagcxInvalidArgument;
  return (flagcxResult_t)ncclCommFinalize(comm->base);
}

flagcxResult_t FLAGCX_NCCL_DESTROY(flagcxInnerComm_t comm) {
  if (comm == NULL)
    return flagcxInvalidArgument;
  flagcxResult_t result = (flagcxResult_t)ncclCommDestroy(comm->base);
  if (result != flagcxSuccess)
    return result;
  free(comm);
  return flagcxSuccess;
}
#undef FLAGCX_NCCL_FINALIZE
#undef FLAGCX_NCCL_DESTROY
