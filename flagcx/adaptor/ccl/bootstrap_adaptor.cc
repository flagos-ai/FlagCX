#include "bootstrap_adaptor.h"
#include "bootstrap.h"

#ifdef USE_BOOTSTRAP_ADAPTOR

// TODO: unsupported
flagcxResult_t bootstrapAdaptorGetVersion(int *version) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t bootstrapAdaptorGetUniqueId(flagcxUniqueId_t *uniqueId) {
  return flagcxNotSupported;
}

// TODO: unsupported
const char *bootstrapAdaptorGetErrorString(flagcxResult_t result) {
  return "Not Implemented";
}

// TODO: unsupported
const char *bootstrapAdaptorGetLastError(flagcxInnerComm_t comm) {
  return "Not Implemented";
}

flagcxResult_t bootstrapAdaptorCommWindowRegister(flagcxInnerComm_t comm,
                                                  void *buff, size_t size,
                                                  void **win, int flags) {
  return flagcxNotSupported;
}

flagcxResult_t bootstrapAdaptorCommWindowDeregister(flagcxInnerComm_t comm,
                                                    void *win) {
  return flagcxNotSupported;
}

flagcxResult_t bootstrapAdaptorGetStagedBuffer(const flagcxInnerComm_t comm,
                                                void **buff, size_t size,
                                                int isRecv) {
  return flagcxNotSupported;
}

flagcxResult_t bootstrapAdaptorCommInitRank(flagcxInnerComm_t *comm, int nranks,
                                            flagcxUniqueId_t /*commId*/,
                                            int rank,
                                            bootstrapState *bootstrap) {
  if (*comm == NULL) {
    FLAGCXCHECK(flagcxCalloc(comm, 1));
  }
  (*comm)->base = bootstrap;

  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorCommFinalize(flagcxInnerComm_t comm) {
  // Note that the bootstrap member is destroyed in the flagcxCommDestroy
  // function
  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorCommDestroy(flagcxInnerComm_t comm) {
  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorCommAbort(flagcxInnerComm_t comm) {
  // We don't need to do anything here
  return flagcxSuccess;
}

// TODO: unsupported
flagcxResult_t bootstrapAdaptorCommResume(flagcxInnerComm_t comm) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t bootstrapAdaptorCommSuspend(flagcxInnerComm_t comm) {
  return flagcxNotSupported;
}

flagcxResult_t bootstrapAdaptorCommCount(const flagcxInnerComm_t comm,
                                         int *count) {
  *count = comm->base->nranks;
  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorCommCuDevice(const flagcxInnerComm_t comm,
                                            int *device) {
  device = NULL;
  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorCommUserRank(const flagcxInnerComm_t comm,
                                            int *rank) {
  *rank = comm->base->rank;
  return flagcxSuccess;
}

// TODO: unsupported
flagcxResult_t bootstrapAdaptorCommGetAsyncError(flagcxInnerComm_t comm,
                                                 flagcxResult_t *asyncError) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t bootstrapAdaptorMemAlloc(void **ptr, size_t size) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t bootstrapAdaptorMemFree(void *ptr) { return flagcxNotSupported; }

// TODO: unsupported
flagcxResult_t bootstrapAdaptorCommRegister(flagcxInnerComm_t comm, void *buff,
                                            size_t size, void **handle) {
  return flagcxNotSupported;
}

// TODO: unsupported
flagcxResult_t bootstrapAdaptorCommDeregister(flagcxInnerComm_t comm,
                                              void *handle) {
  return flagcxNotSupported;
}

flagcxResult_t bootstrapAdaptorGather(const void *sendbuff, void *recvbuff,
                                      size_t count, flagcxDataType_t datatype,
                                      int root, flagcxInnerComm_t comm,
                                      flagcxStream_t /*stream*/) {
  FLAGCXCHECK(
      GatherBootstrap(comm->base, sendbuff, recvbuff, count, datatype, root));
  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorScatter(const void *sendbuff, void *recvbuff,
                                       size_t count, flagcxDataType_t datatype,
                                       int root, flagcxInnerComm_t comm,
                                       flagcxStream_t /*stream*/) {
  FLAGCXCHECK(
      ScatterBootstrap(comm->base, sendbuff, recvbuff, count, datatype, root));
  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                         size_t count,
                                         flagcxDataType_t datatype, int root,
                                         flagcxInnerComm_t comm,
                                         flagcxStream_t /*stream*/) {
  FLAGCXCHECK(BroadcastBootstrap(comm->base, sendbuff, recvbuff, count,
                                 datatype, root));
  return flagcxSuccess;
}

flagcxResult_t
bootstrapAdaptorAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                          flagcxDataType_t datatype, flagcxRedOp_t op,
                          flagcxInnerComm_t comm, flagcxStream_t /*stream*/) {
  FLAGCXCHECK(
      AllReduceBootstrap(comm->base, sendbuff, recvbuff, count, datatype, op));
  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorReduce(const void *sendbuff, void *recvbuff,
                                      size_t count, flagcxDataType_t datatype,
                                      flagcxRedOp_t op, int root,
                                      flagcxInnerComm_t comm,
                                      flagcxStream_t /*stream*/) {
  FLAGCXCHECK(ReduceBootstrap(comm->base, sendbuff, recvbuff, count, datatype,
                              op, root));
  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorReduceScatter(const void *sendbuff,
                                             void *recvbuff, size_t recvcount,
                                             flagcxDataType_t datatype,
                                             flagcxRedOp_t op,
                                             flagcxInnerComm_t comm,
                                             flagcxStream_t /*stream*/) {
  FLAGCXCHECK(ReduceScatterBootstrap(comm->base, sendbuff, recvbuff, recvcount,
                                     datatype, op));
  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                         size_t sendcount,
                                         flagcxDataType_t datatype,
                                         flagcxInnerComm_t comm,
                                         flagcxStream_t /*stream*/) {
  FLAGCXCHECK(
      AllGatherBootstrap(comm->base, sendbuff, recvbuff, sendcount, datatype));
  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                        size_t count, flagcxDataType_t datatype,
                                        flagcxInnerComm_t comm,
                                        flagcxStream_t /*stream*/) {
  FLAGCXCHECK(
      AlltoAllBootstrap(comm->base, sendbuff, recvbuff, count, datatype));
  return flagcxSuccess;
}

flagcxResult_t
bootstrapAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                          size_t *sdispls, void *recvbuff, size_t *recvcounts,
                          size_t *rdispls, flagcxDataType_t datatype,
                          flagcxInnerComm_t comm, flagcxStream_t /*stream*/) {
  FLAGCXCHECK(AlltoAllvBootstrap(comm->base, sendbuff, sendcounts, sdispls,
                                 recvbuff, recvcounts, rdispls, datatype));
  return flagcxSuccess;
}

#define BOOTSTRAP_SEND_RECV_TAG -6767
flagcxResult_t bootstrapAdaptorSend(const void *sendbuff, size_t count,
                                    flagcxDataType_t datatype, int peer,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t /*stream*/) {
  FLAGCXCHECK(bootstrapSend(comm->base, peer, BOOTSTRAP_SEND_RECV_TAG,
                            (void *)sendbuff,
                            count * getFlagcxDataTypeSize(datatype)));
  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorRecv(void *recvbuff, size_t count,
                                    flagcxDataType_t datatype, int peer,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t /*stream*/) {
  FLAGCXCHECK(bootstrapRecv(comm->base, peer, BOOTSTRAP_SEND_RECV_TAG, recvbuff,
                            count * getFlagcxDataTypeSize(datatype)));
  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorGroupStart() {
  // We don't need to do anything here
  return flagcxSuccess;
}

flagcxResult_t bootstrapAdaptorGroupEnd() {
  // We don't need to do anything here
  return flagcxSuccess;
}

struct flagcxCCLAdaptor bootstrapAdaptor = {
    "BOOTSTRAP",
    // Basic functions
    bootstrapAdaptorGetVersion, bootstrapAdaptorGetUniqueId,
    bootstrapAdaptorGetErrorString, bootstrapAdaptorGetLastError,
    // Symmetric operations
    bootstrapAdaptorCommWindowRegister, bootstrapAdaptorCommWindowDeregister,
    bootstrapAdaptorGetStagedBuffer,
    // Communicator functions
    bootstrapAdaptorCommInitRank, bootstrapAdaptorCommFinalize,
    bootstrapAdaptorCommDestroy, bootstrapAdaptorCommAbort,
    bootstrapAdaptorCommResume, bootstrapAdaptorCommSuspend,
    bootstrapAdaptorCommCount, bootstrapAdaptorCommCuDevice,
    bootstrapAdaptorCommUserRank, bootstrapAdaptorCommGetAsyncError,
    bootstrapAdaptorMemAlloc, bootstrapAdaptorMemFree,
    bootstrapAdaptorCommRegister, bootstrapAdaptorCommDeregister,
    // Communication functions
    bootstrapAdaptorReduce, bootstrapAdaptorGather, bootstrapAdaptorScatter,
    bootstrapAdaptorBroadcast, bootstrapAdaptorAllReduce,
    bootstrapAdaptorReduceScatter, bootstrapAdaptorAllGather,
    bootstrapAdaptorAlltoAll, bootstrapAdaptorAlltoAllv, bootstrapAdaptorSend,
    bootstrapAdaptorRecv,
    // Group semantics
    bootstrapAdaptorGroupStart, bootstrapAdaptorGroupEnd};

#endif // USE_BOOTSTRAP_ADAPTOR