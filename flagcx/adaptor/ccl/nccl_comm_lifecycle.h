#ifndef FLAGCX_NCCL_COMM_LIFECYCLE_H_
#define FLAGCX_NCCL_COMM_LIFECYCLE_H_

// Kept separate so host tests can exercise these exact adaptor functions with
// native API shims. Finalize must retain the wrapper for deregistration, async
// queries and the eventual Destroy, including native in-progress completion.
flagcxResult_t ncclAdaptorCommFinalize(flagcxInnerComm_t comm) {
  if (comm == NULL)
    return flagcxInvalidArgument;
  return (flagcxResult_t)ncclCommFinalize(comm->base);
}

flagcxResult_t ncclAdaptorCommDestroy(flagcxInnerComm_t comm) {
  if (comm == NULL)
    return flagcxInvalidArgument;
  FLAGCXCHECK((flagcxResult_t)ncclCommDestroy(comm->base));
  free(comm);
  return flagcxSuccess;
}

#endif
