// Included once per host adaptor, and by SDK-independent contract tests.
flagcxResult_t FLAGCX_HOST_FINALIZE(flagcxInnerComm_t comm) {
  flagcxResult_t result = FLAGCX_HOST_CLEANUP(comm);
  if (result != flagcxSuccess)
    return result;
  comm->base.reset();
  return flagcxSuccess;
}

flagcxResult_t FLAGCX_HOST_DESTROY(flagcxInnerComm_t comm) {
  flagcxResult_t result = FLAGCX_HOST_FINALIZE(comm);
  if (result != flagcxSuccess)
    return result;
  free(comm);
  return flagcxSuccess;
}
#undef FLAGCX_HOST_FINALIZE
#undef FLAGCX_HOST_DESTROY
#undef FLAGCX_HOST_CLEANUP
