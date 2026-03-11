/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "flagcx_hetero.h"
#include "proxy.h"
#include "runner.h"
#include "uni_runner_impl.h"

FLAGCX_PARAM(UniRunnerUseLocRed, "UNIRUNNER_USE_LOCRED", 0);
FLAGCX_PARAM(UniRunnerUseRingAG, "UNIRUNNER_USE_RINGAG", 0);
FLAGCX_PARAM(UniRunnerUseSlicedAR, "UNIRUNNER_USE_SLICEDAR", 0);

flagcxResult_t uniRunnerReduce(const void *sendbuff, void *recvbuff,
                               size_t count, flagcxDataType_t datatype,
                               flagcxRedOp_t op, int root, flagcxComm_t comm,
                               flagcxStream_t stream) {
  flagcxResult_t res = flagcxSuccess;
  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  void *scratchbuff = nullptr;
  FLAGCXCHECK(deviceAdaptor->deviceMalloc(
      &scratchbuff, 2 * count * getFlagcxDataTypeSize(datatype),
      flagcxMemDevice, stream));
  FLAGCXCHECKGOTO(initUniRunner(hcomm), res, out);
  FLAGCXCHECKGOTO(initUniRunnerStateTreeRed(runnerState, sendbuff, recvbuff,
                                            scratchbuff, count, datatype, op,
                                            root, hcomm),
                  res, out);
  FLAGCXCHECKGOTO(runUniRunner(hcomm, stream), res, out);
out:
  FLAGCXCHECK(deviceAdaptor->deviceFree(scratchbuff, flagcxMemDevice, stream));
  FLAGCXCHECK(cleanupUniRunner(hcomm));
  return res;
}

flagcxResult_t uniRunnerGather(const void *sendbuff, void *recvbuff,
                               size_t count, flagcxDataType_t datatype,
                               int root, flagcxComm_t comm,
                               flagcxStream_t stream) {
  size_t size = count * getFlagcxDataTypeSize(datatype);
  char *buffer = static_cast<char *>(recvbuff);

  FLAGCXCHECK(flagcxHeteroGroupStart());
  if (comm->rank == root) {
    for (int r = 0; r < comm->nranks; r++) {
      FLAGCXCHECK(flagcxHeteroRecv(static_cast<void *>(buffer + r * size),
                                   count, datatype, r, comm->heteroComm,
                                   stream));
    }
  }
  FLAGCXCHECK(flagcxHeteroSend(sendbuff, count, datatype, root,
                               comm->heteroComm, stream));
  FLAGCXCHECK(flagcxHeteroGroupEnd());
  return flagcxSuccess;
}

flagcxResult_t uniRunnerScatter(const void *sendbuff, void *recvbuff,
                                size_t count, flagcxDataType_t datatype,
                                int root, flagcxComm_t comm,
                                flagcxStream_t stream) {
  size_t size = count * getFlagcxDataTypeSize(datatype);
  const char *buffer = static_cast<const char *>(sendbuff);

  FLAGCXCHECK(flagcxHeteroGroupStart());
  if (comm->rank == root) {
    for (int r = 0; r < comm->nranks; r++) {
      FLAGCXCHECK(flagcxHeteroSend(static_cast<const void *>(buffer + r * size),
                                   count, datatype, r, comm->heteroComm,
                                   stream));
    }
  }
  FLAGCXCHECK(flagcxHeteroRecv(recvbuff, count, datatype, root,
                               comm->heteroComm, stream));
  FLAGCXCHECK(flagcxHeteroGroupEnd());
  return flagcxSuccess;
}

flagcxResult_t uniRunnerBroadcast(const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  int root, flagcxComm_t comm,
                                  flagcxStream_t stream) {
  FLAGCXCHECK(flagcxHeteroGroupStart());
  if (comm->rank == root) {
    for (int r = 0; r < comm->nranks; r++) {
      FLAGCXCHECK(flagcxHeteroSend(sendbuff, count, datatype, r,
                                   comm->heteroComm, stream));
    }
  }
  FLAGCXCHECK(flagcxHeteroRecv(recvbuff, count, datatype, root,
                               comm->heteroComm, stream));
  FLAGCXCHECK(flagcxHeteroGroupEnd());
  return flagcxSuccess;
}

flagcxResult_t uniRunnerAllReduce(const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  flagcxRedOp_t op, flagcxComm_t comm,
                                  flagcxStream_t stream) {
  flagcxResult_t res = flagcxSuccess;
  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  FLAGCXCHECK(initUniRunner(hcomm));
  if (flagcxParamUniRunnerUseLocRed()) {
    /* initialize uniRunnerState for reduce test */
    FLAGCXCHECKGOTO(initUniRunnerStateLocRed(runnerState, sendbuff, recvbuff,
                                             count, datatype, op, hcomm),
                    res, out);
  } else if (flagcxParamUniRunnerUseRingAG()) {
    /* initialize uniRunnerState for p2p test */
    FLAGCXCHECKGOTO(initUniRunnerStateRingAG(runnerState, sendbuff, recvbuff,
                                             count, datatype, op, hcomm),
                    res, out);
  } else if (flagcxParamUniRunnerUseSlicedAR()) {
    /* initialize uniRunnerState for sliced AllReduce */
    FLAGCXCHECKGOTO(initUniRunnerStateSlicedAR(runnerState, sendbuff, recvbuff,
                                               count, datatype, op, hcomm),
                    res, out);
  } else {
    /* initialize uniRunnerState for ring AllReduce */
    FLAGCXCHECKGOTO(initUniRunnerStateRingAR(runnerState, sendbuff, recvbuff,
                                             count, datatype, op, hcomm),
                    res, out);
  }
  FLAGCXCHECK(runUniRunner(hcomm, stream));
out:
  FLAGCXCHECK(cleanupUniRunner(hcomm));
  return res;
}

flagcxResult_t uniRunnerReduceScatter(const void *sendbuff, void *recvbuff,
                                      size_t recvcount,
                                      flagcxDataType_t datatype,
                                      flagcxRedOp_t op, flagcxComm_t comm,
                                      flagcxStream_t stream) {
  flagcxResult_t res = flagcxSuccess;
  flagcxHeteroComm_t hcomm = comm->heteroComm;
  flagcxUniRunnerState *runnerState = &hcomm->proxyState->uniRunnerState;
  void *scratchbuff = nullptr;
  FLAGCXCHECK(deviceAdaptor->deviceMalloc(
      &scratchbuff, recvcount * comm->nranks * getFlagcxDataTypeSize(datatype),
      flagcxMemDevice, stream));
  FLAGCXCHECKGOTO(initUniRunner(hcomm), res, out);
  FLAGCXCHECKGOTO(initUniRunnerStateRingRS(runnerState, sendbuff, recvbuff,
                                           scratchbuff, recvcount, datatype, op,
                                           hcomm),
                  res, out);
  FLAGCXCHECKGOTO(runUniRunner(hcomm, stream), res, out);
out:
  FLAGCXCHECK(deviceAdaptor->deviceFree(scratchbuff, flagcxMemDevice, stream));
  FLAGCXCHECK(cleanupUniRunner(hcomm));
  return res;
}

flagcxResult_t uniRunnerAllGather(const void *sendbuff, void *recvbuff,
                                  size_t sendcount, flagcxDataType_t datatype,
                                  flagcxComm_t comm, flagcxStream_t stream) {
  size_t size = sendcount * getFlagcxDataTypeSize(datatype);
  char *bufferOut = static_cast<char *>(recvbuff);
  FLAGCXCHECK(flagcxHeteroGroupStart());
  for (int r = 0; r < comm->nranks; r++) {
    FLAGCXCHECK(flagcxHeteroSend(sendbuff, sendcount, datatype, r,
                                 comm->heteroComm, stream));
    FLAGCXCHECK(flagcxHeteroRecv(static_cast<void *>(bufferOut + r * size),
                                 sendcount, datatype, r, comm->heteroComm,
                                 stream));
  }
  FLAGCXCHECK(flagcxHeteroGroupEnd());
  return flagcxSuccess;
}

flagcxResult_t uniRunnerAlltoAll(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 flagcxComm_t comm, flagcxStream_t stream) {
  size_t size = count * getFlagcxDataTypeSize(datatype);
  const char *bufferIn = static_cast<const char *>(sendbuff);
  char *bufferOut = static_cast<char *>(recvbuff);
  FLAGCXCHECK(flagcxHeteroGroupStart());
  for (int r = 0; r < comm->nranks; r++) {
    FLAGCXCHECK(flagcxHeteroSend(static_cast<const void *>(bufferIn + r * size),
                                 count, datatype, r, comm->heteroComm, stream));
    FLAGCXCHECK(flagcxHeteroRecv(static_cast<void *>(bufferOut + r * size),
                                 count, datatype, r, comm->heteroComm, stream));
  }
  FLAGCXCHECK(flagcxHeteroGroupEnd());
  return flagcxSuccess;
}

flagcxResult_t uniRunnerAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                  size_t *sdispls, void *recvbuff,
                                  size_t *recvcounts, size_t *rdispls,
                                  flagcxDataType_t datatype, flagcxComm_t comm,
                                  flagcxStream_t stream) {
  size_t size = getFlagcxDataTypeSize(datatype);
  const char *bufferIn = static_cast<const char *>(sendbuff);
  char *bufferOut = static_cast<char *>(recvbuff);
  FLAGCXCHECK(flagcxHeteroGroupStart());
  for (int r = 0; r < comm->nranks; r++) {
    if (flagcxCCLAdaptorNeedSendrecv(sendcounts[r])) {
      FLAGCXCHECK(flagcxHeteroSend(
          static_cast<const void *>(bufferIn + sdispls[r] * size),
          sendcounts[r], datatype, r, comm->heteroComm, stream));
    }
    if (flagcxCCLAdaptorNeedSendrecv(recvcounts[r])) {
      FLAGCXCHECK(flagcxHeteroRecv(
          static_cast<void *>(bufferOut + rdispls[r] * size), recvcounts[r],
          datatype, r, comm->heteroComm, stream));
    }
  }
  FLAGCXCHECK(flagcxHeteroGroupEnd());
  return flagcxSuccess;
}

flagcxResult_t uniRunnerSend(const void *sendbuff, size_t count,
                             flagcxDataType_t datatype, int peer,
                             flagcxComm_t comm, flagcxStream_t stream) {
  FLAGCXCHECK(flagcxHeteroSend(sendbuff, count, datatype, peer,
                               comm->heteroComm, stream));
  return flagcxSuccess;
}

flagcxResult_t uniRunnerRecv(void *recvbuff, size_t count,
                             flagcxDataType_t datatype, int peer,
                             flagcxComm_t comm, flagcxStream_t stream) {
  FLAGCXCHECK(flagcxHeteroRecv(recvbuff, count, datatype, peer,
                               comm->heteroComm, stream));
  return flagcxSuccess;
}

flagcxResult_t uniRunnerGroupStart() {
  FLAGCXCHECK(flagcxHeteroGroupStart());
  return flagcxSuccess;
}

flagcxResult_t uniRunnerGroupEnd() {
  FLAGCXCHECK(flagcxHeteroGroupEnd());
  return flagcxSuccess;
}

struct flagcxRunner uniRunner = {
    // Communication functions
    uniRunnerReduce, uniRunnerGather, uniRunnerScatter, uniRunnerBroadcast,
    uniRunnerAllReduce, uniRunnerReduceScatter, uniRunnerAllGather,
    uniRunnerAlltoAll, uniRunnerAlltoAllv, uniRunnerSend, uniRunnerRecv,
    // Group semantics
    uniRunnerGroupStart, uniRunnerGroupEnd};