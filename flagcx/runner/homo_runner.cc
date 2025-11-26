/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "flagcx_tuner.h"
#include "runner.h"

flagcxResult_t homoRunnerReduce(const void *sendbuff, void *recvbuff,
                                size_t count, flagcxDataType_t datatype,
                                flagcxRedOp_t op, int root, flagcxComm_t comm,
                                flagcxStream_t stream) {
  if (comm->tuner == NULL) {
    FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->reduce(
        sendbuff, recvbuff, count, datatype, op, root, comm->homo_comm,
        stream));
  } else {
    FLAGCXCALLWITHTUNER(cclAdaptors[flagcxCCLAdaptorDevice]->reduce(
                            sendbuff, recvbuff, count, datatype, op, root,
                            comm->tunerInnerComm, stream),
                        comm, flagcxCommOpReduce, count, datatype, stream);
  }
  return flagcxSuccess;
}

flagcxResult_t homoRunnerGather(const void *sendbuff, void *recvbuff,
                                size_t count, flagcxDataType_t datatype,
                                int root, flagcxComm_t comm,
                                flagcxStream_t stream) {
  if (comm->tuner == NULL) {
    FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->gather(
        sendbuff, recvbuff, count, datatype, root, comm->homo_comm, stream));
  } else {
    FLAGCXCALLWITHTUNER(cclAdaptors[flagcxCCLAdaptorDevice]->gather(
                            sendbuff, recvbuff, count, datatype, root,
                            comm->tunerInnerComm, stream),
                        comm, flagcxCommOpGather, count, datatype, stream);
  }
  return flagcxSuccess;
}

flagcxResult_t homoRunnerScatter(const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype,
                                 int root, flagcxComm_t comm,
                                 flagcxStream_t stream) {
  if (comm->tuner == NULL) {
    FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->scatter(
        sendbuff, recvbuff, count, datatype, root, comm->homo_comm, stream));
  } else {
    FLAGCXCALLWITHTUNER(cclAdaptors[flagcxCCLAdaptorDevice]->scatter(
                            sendbuff, recvbuff, count, datatype, root,
                            comm->tunerInnerComm, stream),
                        comm, flagcxCommOpScatter, count, datatype, stream);
  }
  return flagcxSuccess;
}

flagcxResult_t homoRunnerBroadcast(const void *sendbuff, void *recvbuff,
                                   size_t count, flagcxDataType_t datatype,
                                   int root, flagcxComm_t comm,
                                   flagcxStream_t stream) {
  if (comm->tuner == NULL) {
    FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->broadcast(
        sendbuff, recvbuff, count, datatype, root, comm->homo_comm, stream));
  } else {
    FLAGCXCALLWITHTUNER(cclAdaptors[flagcxCCLAdaptorDevice]->broadcast(
                            sendbuff, recvbuff, count, datatype, root,
                            comm->tunerInnerComm, stream),
                        comm, flagcxCommOpBroadcast, count, datatype, stream);
  }
  return flagcxSuccess;
}

flagcxResult_t homoRunnerAllReduce(const void *sendbuff, void *recvbuff,
                                   size_t count, flagcxDataType_t datatype,
                                   flagcxRedOp_t op, flagcxComm_t comm,
                                   flagcxStream_t stream) {
  if (comm->tuner == NULL) {
    FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->allReduce(
        sendbuff, recvbuff, count, datatype, op, comm->homo_comm, stream));
  } else {
    FLAGCXCALLWITHTUNER(cclAdaptors[flagcxCCLAdaptorDevice]->allReduce(
                            sendbuff, recvbuff, count, datatype, op,
                            comm->tunerInnerComm, stream),
                        comm, flagcxCommOpAllReduce, count, datatype, stream);
  }
  return flagcxSuccess;
}

flagcxResult_t homoRunnerReduceScatter(const void *sendbuff, void *recvbuff,
                                       size_t recvcount,
                                       flagcxDataType_t datatype,
                                       flagcxRedOp_t op, flagcxComm_t comm,
                                       flagcxStream_t stream) {
  if (comm->tuner == NULL) {
    FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->reduceScatter(
        sendbuff, recvbuff, recvcount, datatype, op, comm->homo_comm, stream));
  } else {
    FLAGCXCALLWITHTUNER(cclAdaptors[flagcxCCLAdaptorDevice]->reduceScatter(
                            sendbuff, recvbuff, recvcount, datatype, op,
                            comm->tunerInnerComm, stream),
                        comm, flagcxCommOpReduceScatter, recvcount, datatype,
                        stream);
  }
  return flagcxSuccess;
}

flagcxResult_t homoRunnerAllGather(const void *sendbuff, void *recvbuff,
                                   size_t sendcount, flagcxDataType_t datatype,
                                   flagcxComm_t comm, flagcxStream_t stream) {
  if (comm->tuner == NULL) {
    FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->allGather(
        sendbuff, recvbuff, sendcount, datatype, comm->homo_comm, stream));
  } else {
    FLAGCXCALLWITHTUNER(cclAdaptors[flagcxCCLAdaptorDevice]->allGather(
                            sendbuff, recvbuff, sendcount, datatype,
                            comm->tunerInnerComm, stream),
                        comm, flagcxCommOpAllGather, sendcount, datatype,
                        stream);
  }
  return flagcxSuccess;
}

flagcxResult_t homoRunnerAlltoAll(const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  flagcxComm_t comm, flagcxStream_t stream) {
  if (comm->tuner == NULL) {
    FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->alltoAll(
        sendbuff, recvbuff, count, datatype, comm->homo_comm, stream));
  } else {
    FLAGCXCALLWITHTUNER(
        cclAdaptors[flagcxCCLAdaptorDevice]->alltoAll(
            sendbuff, recvbuff, count, datatype, comm->tunerInnerComm, stream),
        comm, flagcxCommOpAlltoAll, count, datatype, stream);
  }
  return flagcxSuccess;
}

flagcxResult_t homoRunnerAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                   size_t *sdispls, void *recvbuff,
                                   size_t *recvcounts, size_t *rdispls,
                                   flagcxDataType_t datatype, flagcxComm_t comm,
                                   flagcxStream_t stream) {
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->alltoAllv(
      sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls, datatype,
      comm->homo_comm, stream));
  return flagcxSuccess;
}

flagcxResult_t homoRunnerSend(const void *sendbuff, size_t count,
                              flagcxDataType_t datatype, int peer,
                              flagcxComm_t comm, flagcxStream_t stream) {
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->send(
      sendbuff, count, datatype, peer, comm->homo_comm, stream));
  return flagcxSuccess;
}

flagcxResult_t homoRunnerRecv(void *recvbuff, size_t count,
                              flagcxDataType_t datatype, int peer,
                              flagcxComm_t comm, flagcxStream_t stream) {
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->recv(
      recvbuff, count, datatype, peer, comm->homo_comm, stream));
  return flagcxSuccess;
}

flagcxResult_t homoRunnerGroupStart() {
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->groupStart());
  return flagcxSuccess;
}

flagcxResult_t homoRunnerGroupEnd() {
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->groupEnd());
  return flagcxSuccess;
}

struct flagcxRunner homoRunner = {
    // Communication functions
    homoRunnerReduce, homoRunnerGather, homoRunnerScatter, homoRunnerBroadcast,
    homoRunnerAllReduce, homoRunnerReduceScatter, homoRunnerAllGather,
    homoRunnerAlltoAll, homoRunnerAlltoAllv, homoRunnerSend, homoRunnerRecv,
    // Group semantics
    homoRunnerGroupStart, homoRunnerGroupEnd};