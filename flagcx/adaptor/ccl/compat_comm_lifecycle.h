// Intentionally instantiated by each builtin without a native Finalize, and by
// the host lifecycle shims. The opaque native base is the closed-state marker.
// Owners/registrations must already be closed before this callback is entered.
flagcxResult_t FLAGCX_COMPAT_FINALIZE(flagcxInnerComm_t comm) {
  if (comm == NULL)
    return flagcxInvalidArgument;
  if (comm->base == NULL)
    return flagcxSuccess;
  flagcxResult_t result = FLAGCX_COMPAT_CLOSE_NATIVE(comm);
  if (result == flagcxSuccess)
    comm->base = NULL;
  return result;
}

flagcxResult_t FLAGCX_COMPAT_DESTROY(flagcxInnerComm_t comm) {
  FLAGCXCHECK(FLAGCX_COMPAT_FINALIZE(comm));
  free(comm);
  return flagcxSuccess;
}
#undef FLAGCX_COMPAT_FINALIZE
#undef FLAGCX_COMPAT_DESTROY
#undef FLAGCX_COMPAT_CLOSE_NATIVE
