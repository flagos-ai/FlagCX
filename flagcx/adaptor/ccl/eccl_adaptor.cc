/*************************************************************************
 * Copyright (c) 2025, ENFLAME CORPORATION. All rights reserved.
 ************************************************************************/

#include "enflame_adaptor.h"

#ifdef USE_ENFLAME_ADAPTOR

#include "adaptor.h"
#include "alloc.h"
#include "comm.h"

static_assert(sizeof(ecclUniqueId) <= sizeof(flagcxInnerUniqueId),
              "ecclUniqueId does not fit inside flagcxInnerUniqueId");

flagcxResult_t ecclAdaptorGetVersion(int *version) {
  return (flagcxResult_t)ecclGetVersion(version);
}

flagcxResult_t ecclAdaptorGetUniqueId(flagcxInnerUniqueId_t *uniqueId) {
  if (*uniqueId == NULL) {
    flagcxCalloc(uniqueId, 1);
  }
  return (flagcxResult_t)ecclGetUniqueId((ecclUniqueId *)(*uniqueId));
}

const char *ecclAdaptorGetErrorString(flagcxResult_t result) {
  return ecclGetErrorString((ecclResult_t)result);
}

const char *ecclAdaptorGetLastError(flagcxInnerComm_t comm) {
  return ecclGetLastError(comm->base);
}

flagcxResult_t ecclAdaptorGetStagedBuffer(const flagcxInnerComm_t comm,
                                          void **buff, size_t size,
                                          int isRecv) {
  return flagcxNotSupported;
}

flagcxResult_t ecclAdaptorCommInitRank(flagcxInnerComm_t *comm, int nranks,
                                       flagcxInnerUniqueId_t commId, int rank,
                                       struct bootstrapState * /*bootstrap*/) {
  if (*comm == NULL) {
    flagcxCalloc(comm, 1);
  }
  return (flagcxResult_t)ecclCommInitRank(&(*comm)->base, nranks,
                                          *(ecclUniqueId *)commId, rank);
}

flagcxResult_t ecclAdaptorCommFinalize(flagcxInnerComm_t comm) {
  // ECCL does not have a separate finalize function, use destroy
  return flagcxSuccess;
}

flagcxResult_t ecclAdaptorCommDestroy(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ecclCommDestroy(comm->base);
}

flagcxResult_t ecclAdaptorCommAbort(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ecclCommAbort(comm->base);
}

flagcxResult_t ecclAdaptorCommResume(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ecclInvalidUsage;
}

flagcxResult_t ecclAdaptorCommSuspend(flagcxInnerComm_t comm) {
  return (flagcxResult_t)ecclInvalidUsage;
}

flagcxResult_t ecclAdaptorCommCount(const flagcxInnerComm_t comm, int *count) {
  return (flagcxResult_t)ecclCommCount(comm->base, count);
}

flagcxResult_t ecclAdaptorCommCuDevice(const flagcxInnerComm_t comm,
                                       int *device) {
  return (flagcxResult_t)ecclCommDevice(comm->base, device);
}

flagcxResult_t ecclAdaptorCommUserRank(const flagcxInnerComm_t comm,
                                       int *rank) {
  return (flagcxResult_t)ecclCommUserRank(comm->base, rank);
}

flagcxResult_t ecclAdaptorCommGetAsyncError(flagcxInnerComm_t comm,
                                            flagcxResult_t *asyncError) {
  return (flagcxResult_t)ecclCommGetAsyncError(comm->base,
                                               (ecclResult_t *)asyncError);
}

flagcxResult_t ecclAdaptorMemAlloc(void **ptr, size_t size) {
  return (flagcxResult_t)ecclMemAlloc(ptr, 2*size);
}

flagcxResult_t ecclAdaptorMemFree(void *ptr) {
  return (flagcxResult_t)ecclMemFree(ptr);
}

flagcxResult_t ecclAdaptorCommRegister(const flagcxInnerComm_t comm, void *buff,
                                       size_t size, void **handle) {
  return flagcxNotSupported;
}

flagcxResult_t ecclAdaptorCommDeregister(const flagcxInnerComm_t comm,
                                         void *handle) {
  return flagcxNotSupported;
}

flagcxResult_t ecclAdaptorCommWindowRegister(flagcxInnerComm_t comm, void *buff,
                                             size_t size,
                                             flagcxInnerWindow_t *win,
                                             int winFlags) {
  if (*win == NULL) {
    FLAGCXCHECK(flagcxCalloc(win, 1));
  }
  ecclWindow_t ecclWin = NULL;
  flagcxResult_t res = (flagcxResult_t)ecclCommWindowRegister(
      comm->base, buff, 2*size, &ecclWin, winFlags);
  if (res == flagcxSuccess) {
    (*win)->base = ecclWin;
    (*win)->winFlags = winFlags;
  } else {
    free(*win);
    *win = NULL;
  }
  return res;
}

flagcxResult_t ecclAdaptorCommWindowDeregister(flagcxInnerComm_t comm,
                                               flagcxInnerWindow_t win) {
  flagcxResult_t res =
      (flagcxResult_t)ecclCommWindowDeregister(comm->base, win->base);
  free(win);
  return res;
}

flagcxResult_t ecclAdaptorReduce(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 flagcxRedOp_t op, int root,
                                 flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
  return (flagcxResult_t)ecclReduce(sendbuff, recvbuff, count,
                                    (ecclDataType_t)datatype, (ecclRedOp_t)op,
                                    root, comm->base, stream->base);
}

flagcxResult_t ecclAdaptorGather(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 int root, flagcxInnerComm_t comm,
                                 flagcxStream_t stream) {
  return (flagcxResult_t)ecclGather(sendbuff, recvbuff, count,
                                    (ecclDataType_t)datatype, root, comm->base,
                                    stream->base);
}

flagcxResult_t ecclAdaptorScatter(const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  int root, flagcxInnerComm_t comm,
                                  flagcxStream_t stream) {
  return (flagcxResult_t)ecclScatter(sendbuff, recvbuff, count,
                                     (ecclDataType_t)datatype, root, comm->base,
                                     stream->base);
}

flagcxResult_t ecclAdaptorBroadcast(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    int root, flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return (flagcxResult_t)ecclBroadcast(sendbuff, recvbuff, count,
                                       (ecclDataType_t)datatype, root,
                                       comm->base, stream->base);
}

flagcxResult_t ecclAdaptorAllReduce(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    flagcxRedOp_t op, flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return (flagcxResult_t)ecclAllReduce(
      sendbuff, recvbuff, count, (ecclDataType_t)datatype, (ecclRedOp_t)op,
      comm->base, stream->base);
}

flagcxResult_t
ecclAdaptorReduceScatter(const void *sendbuff, void *recvbuff, size_t recvcount,
                         flagcxDataType_t datatype, flagcxRedOp_t op,
                         flagcxInnerComm_t comm, flagcxStream_t stream) {
  return (flagcxResult_t)ecclReduceScatter(
      sendbuff, recvbuff, recvcount, (ecclDataType_t)datatype, (ecclRedOp_t)op,
      comm->base, stream->base);
}

flagcxResult_t ecclAdaptorAllGather(const void *sendbuff, void *recvbuff,
                                    size_t sendcount, flagcxDataType_t datatype,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return (flagcxResult_t)ecclAllGather(sendbuff, recvbuff, sendcount,
                                       (ecclDataType_t)datatype, comm->base,
                                       stream->base);
}

flagcxResult_t ecclAdaptorAlltoAll(const void *sendbuff, void *recvbuff,
                                   size_t count, flagcxDataType_t datatype,
                                   flagcxInnerComm_t comm,
                                   flagcxStream_t stream) {
  return (flagcxResult_t)ecclAlltoall(sendbuff, count, (ecclDataType_t)datatype,
                                      recvbuff, count, (ecclDataType_t)datatype,
                                      comm->base, stream->base);
}

flagcxResult_t ecclAdaptorAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                    size_t *sdispls, void *recvbuff,
                                    size_t *recvcounts, size_t *rdispls,
                                    flagcxDataType_t datatype,
                                    flagcxInnerComm_t comm,
                                    flagcxStream_t stream) {
  return (flagcxResult_t)ecclAlltoAllv(
      (void *)sendbuff, sendcounts, sdispls, (ecclDataType_t)datatype, recvbuff,
      recvcounts, rdispls, (ecclDataType_t)datatype, comm->base, stream->base);
}

flagcxResult_t ecclAdaptorSend(const void *sendbuff, size_t count,
                               flagcxDataType_t datatype, int peer,
                               flagcxInnerComm_t comm, flagcxStream_t stream) {
  return (flagcxResult_t)ecclSend(sendbuff, count, (ecclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

flagcxResult_t ecclAdaptorRecv(void *recvbuff, size_t count,
                               flagcxDataType_t datatype, int peer,
                               flagcxInnerComm_t comm, flagcxStream_t stream) {
  return (flagcxResult_t)ecclRecv(recvbuff, count, (ecclDataType_t)datatype,
                                  peer, comm->base, stream->base);
}

flagcxResult_t ecclAdaptorGroupStart() {
  return (flagcxResult_t)ecclGroupStart();
}

flagcxResult_t ecclAdaptorGroupEnd() { return (flagcxResult_t)ecclGroupEnd(); }

flagcxResult_t
ecclAdaptorDevCommReqsInit(flagcxInnerComm_t comm,
                           flagcxDevCommRequirements *reqs) {
  ecclCommProperties_t props = ECCL_COMM_PROPERTIES_INITIALIZER;
  flagcxResult_t ret =
      (flagcxResult_t)ecclCommQueryProperties(comm->base, &props);
  if (ret != flagcxSuccess) {
    return ret;
  }
  if (!props.deviceApiSupport) {
    return flagcxNotSupported;
  }

  *reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs->intraBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
  reqs->intraMulticast = props.multimemSupport;
  return flagcxSuccess;
}

flagcxResult_t
ecclAdaptorDevCommCreate(flagcxInnerComm_t comm,
                         const flagcxDevCommRequirements *reqs,
                         flagcxInnerDevComm_t *devComm) {
  flagcxInnerDevComm_t inner =
      (flagcxInnerDevComm_t)malloc(sizeof(struct flagcxInnerDevComm));
  if (!inner)
    return flagcxSystemError;

  ecclDevCommRequirements_t ecclReqs = ECCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
//  ecclReqs.barrierCount = reqs->barrierCount;
  ecclReqs.lsaBarrierCount = reqs->intraBarrierCount;
  ecclReqs.lsaLLA2ABlockCount = reqs->intraLLA2ABlockCount;
  ecclReqs.lsaLLA2ASlotCount = reqs->intraLLA2ASlotCount;
//  ecclReqs.railGinBarrierCount = reqs->interBarrierCount;
//  ecclReqs.ginSignalCount = reqs->interSignalCount;
//  ecclReqs.ginContextCount = reqs->interContextCount;
//  ecclReqs.ginCounterCount = reqs->interCounterCount;

  int nRanks = -1;
  (flagcxResult_t)ecclCommCount(comm->base, &nRanks);

  printf("\necclAdaptorDevCommCreate nRanks %d\n", nRanks);

  ecclReqs.barrierCount = 1;         // one combined LSA + GIN world barrier (index 0)
  ecclReqs.lareContextCount = 1;     // one LARE context (intra-node put + signal)
  ecclReqs.lareSignalCount = 1;      // one LARE signal slot (intra-node arrival counter)
  ecclReqs.ginSignalCount = 2*nRanks;  // per-peer GIN signal slot (slot == sender world rank)
  ecclReqs.ginContextCount = 1;      // single GIN context per rank
  ecclReqs.ginConnectionType = ECCL_GIN_CONNECTION_FULL;  // cross-node ibgda, non-railed


//  // Connection-type decision mirrors NCCL >= 2.29.7. ginForceEnable stays
//  // false: ECCL deprecates it in favor of an explicit ginConnectionType.
//  const bool needsFullGin = reqs->interForceEnable ||
//                            reqs->interSignalCount > 0 ||
//                            reqs->interCounterCount > 0;
//  const bool needsRailGin =
//      reqs->barrierCount > 0 || reqs->interBarrierCount > 0;
//  ecclReqs.ginConnectionType = needsFullGin   ? ECCL_GIN_CONNECTION_FULL
//                               : needsRailGin ? ECCL_GIN_CONNECTION_RAIL
//                                              : ECCL_GIN_CONNECTION_NONE;

  flagcxResult_t ret =
      (flagcxResult_t)ecclDevCommCreate(comm->base, &ecclReqs, &inner->base);
  if (ret != flagcxSuccess) {
    free(inner);
    return ret;
  }

  *devComm = inner;
  return flagcxSuccess;
}

flagcxResult_t ecclAdaptorDevCommDestroy(flagcxInnerComm_t comm,
                                         flagcxInnerDevComm_t devComm) {
  if (!devComm)
    return flagcxSuccess;
  flagcxResult_t ret =
      (flagcxResult_t)ecclDevCommDestroy(comm->base, &devComm->base);
  free(devComm);
  return ret;
}

struct flagcxCCLAdaptor ecclAdaptor = {
    "ECCL",
    // Basic functions
    ecclAdaptorGetVersion, ecclAdaptorGetUniqueId, ecclAdaptorGetErrorString,
    ecclAdaptorGetLastError, ecclAdaptorGetStagedBuffer,
    // Communicator functions
    ecclAdaptorCommInitRank, ecclAdaptorCommFinalize, ecclAdaptorCommDestroy,
    ecclAdaptorCommAbort, ecclAdaptorCommResume, ecclAdaptorCommSuspend,
    ecclAdaptorCommCount, ecclAdaptorCommCuDevice, ecclAdaptorCommUserRank,
    ecclAdaptorCommGetAsyncError, ecclAdaptorMemAlloc, ecclAdaptorMemFree,
    ecclAdaptorCommRegister, ecclAdaptorCommDeregister,
    // Symmetric functions
    ecclAdaptorCommWindowRegister, ecclAdaptorCommWindowDeregister,
    // Communication functions
    ecclAdaptorReduce, ecclAdaptorGather, ecclAdaptorScatter,
    ecclAdaptorBroadcast, ecclAdaptorAllReduce, ecclAdaptorReduceScatter,
    ecclAdaptorAllGather, ecclAdaptorAlltoAll, ecclAdaptorAlltoAllv,
    ecclAdaptorSend, ecclAdaptorRecv,
    // Group semantics
    ecclAdaptorGroupStart, ecclAdaptorGroupEnd,
    // Device API
    ecclAdaptorDevCommReqsInit, ecclAdaptorDevCommCreate,
    ecclAdaptorDevCommDestroy};

#endif // USE_ENFLAME_ADAPTOR
