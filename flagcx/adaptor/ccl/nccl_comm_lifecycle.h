// Instantiated by NCCL-compatible adaptors and the host lifecycle shims.
// Finalize retains the wrapper; only a completed native Destroy releases it.
#ifndef FLAGCX_CCL_NATIVE_FINALIZE
#define FLAGCX_CCL_NATIVE_FINALIZE ncclCommFinalize
#endif
#ifndef FLAGCX_CCL_NATIVE_DESTROY
#define FLAGCX_CCL_NATIVE_DESTROY ncclCommDestroy
#endif

flagcxResult_t FLAGCX_CCL_FINALIZE(flagcxInnerComm_t comm) {
  if (comm == NULL)
    return flagcxInvalidArgument;
  return (flagcxResult_t)FLAGCX_CCL_NATIVE_FINALIZE(comm->base);
}

flagcxResult_t FLAGCX_CCL_DESTROY(flagcxInnerComm_t comm) {
  if (comm == NULL)
    return flagcxInvalidArgument;
  flagcxResult_t result = (flagcxResult_t)FLAGCX_CCL_NATIVE_DESTROY(comm->base);
  if (result != flagcxSuccess)
    return result;
  free(comm);
  return flagcxSuccess;
}
#undef FLAGCX_CCL_FINALIZE
#undef FLAGCX_CCL_DESTROY
#undef FLAGCX_CCL_NATIVE_FINALIZE
#undef FLAGCX_CCL_NATIVE_DESTROY
