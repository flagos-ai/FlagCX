/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "c2c_algo.h"
#include "runner.h"

#define FLAGCX_CACHE_CAPACITY 16
static flagcxLRUCache<size_t, flagcxC2cPlanner>
    planCache(FLAGCX_CACHE_CAPACITY);

flagcxResult_t hybridRunnerReduce(const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  flagcxRedOp_t op, int root, flagcxComm_t comm,
                                  flagcxStream_t stream) {
  // Construct flagcxC2cPlanner and find corresponding strategy
  flagcxC2cPlanner planner;
  auto hashValue = getC2cCommPatternHash(count, comm->cluster_ids[root],
                                         flagcxCommOpReduce, op, comm);
  if (!planCache.get(hashValue, planner)) {
    INFO(FLAGCX_COLL,
         "No available plan is found, create a new one with "
         "communication pattern "
         "(count, rootClsuterId, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         count, comm->cluster_ids[root], flagcxCommOpReduce, op,
         (size_t)((uintptr_t)comm), hashValue);
    planner =
        flagcxC2cPlanner(count, count, root, comm, flagcxCommOpReduce, op);
    planCache.put(hashValue, planner);
  } else {
    INFO(FLAGCX_COLL,
         "Found available plan with communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         count, comm->cluster_ids[root], flagcxCommOpReduce, op,
         (size_t)((uintptr_t)comm), hashValue);
  }
  FLAGCXCHECK(planner.execute(sendbuff, recvbuff, datatype, root, stream));
  return flagcxSuccess;
}

flagcxResult_t hybridRunnerGather(const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  int root, flagcxComm_t comm,
                                  flagcxStream_t stream) {
  // Construct flagcxC2cPlanner and find corresponding strategy
  flagcxC2cPlanner planner;
  auto hashValue = getC2cCommPatternHash(count, root, flagcxCommOpGather,
                                         flagcxRedNoOp, comm);
  if (!planCache.get(hashValue, planner)) {
    INFO(FLAGCX_COLL,
         "No available plan is found, create a new one with "
         "communication pattern "
         "(count, rootRank, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         count, root, flagcxCommOpGather, flagcxRedNoOp,
         (size_t)((uintptr_t)comm), hashValue);
    planner = flagcxC2cPlanner(count, count * comm->nranks, root, comm,
                               flagcxCommOpGather, flagcxRedNoOp);
    planCache.put(hashValue, planner);
  } else {
    INFO(FLAGCX_COLL,
         "Found available plan with communication pattern "
         "(count, rootRank, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         count, root, flagcxCommOpGather, flagcxRedNoOp,
         (size_t)((uintptr_t)comm), hashValue);
  }
  FLAGCXCHECK(planner.execute(sendbuff, recvbuff, datatype, root, stream));
  return flagcxSuccess;
}

flagcxResult_t hybridRunnerScatter(const void *sendbuff, void *recvbuff,
                                   size_t count, flagcxDataType_t datatype,
                                   int root, flagcxComm_t comm,
                                   flagcxStream_t stream) {
  // Construct flagcxC2cPlanner and find corresponding strategy
  flagcxC2cPlanner planner;
  auto hashValue = getC2cCommPatternHash(count, root, flagcxCommOpScatter,
                                         flagcxRedNoOp, comm);
  if (!planCache.get(hashValue, planner)) {
    INFO(FLAGCX_COLL,
         "No available plan is found, create a new one with "
         "communication pattern "
         "(count, rootRank, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         count, root, flagcxCommOpScatter, flagcxRedNoOp,
         (size_t)((uintptr_t)comm), hashValue);
    planner = flagcxC2cPlanner(count * comm->nranks, count, root, comm,
                               flagcxCommOpScatter, flagcxRedNoOp);
    planCache.put(hashValue, planner);
  } else {
    INFO(FLAGCX_COLL,
         "Found available plan with communication pattern "
         "(count, rootRank, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         count, root, flagcxCommOpScatter, flagcxRedNoOp,
         (size_t)((uintptr_t)comm), hashValue);
  }
  FLAGCXCHECK(planner.execute(sendbuff, recvbuff, datatype, root, stream));
  return flagcxSuccess;
}

flagcxResult_t hybridRunnerBroadcast(const void *sendbuff, void *recvbuff,
                                     size_t count, flagcxDataType_t datatype,
                                     int root, flagcxComm_t comm,
                                     flagcxStream_t stream) {
  // Construct flagcxC2cPlanner and find corresponding strategy
  flagcxC2cPlanner planner;
  auto hashValue =
      getC2cCommPatternHash(count, comm->cluster_ids[root],
                            flagcxCommOpBroadcast, flagcxRedNoOp, comm);
  if (!planCache.get(hashValue, planner)) {
    INFO(FLAGCX_COLL,
         "No available plan is found, create a new one with "
         "communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         count, comm->cluster_ids[root], flagcxCommOpBroadcast, flagcxRedNoOp,
         (size_t)((uintptr_t)comm), hashValue);
    planner = flagcxC2cPlanner(count, count, root, comm, flagcxCommOpBroadcast,
                               flagcxRedNoOp);
    planCache.put(hashValue, planner);
  } else {
    INFO(FLAGCX_COLL,
         "Found available plan with communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         count, comm->cluster_ids[root], flagcxCommOpBroadcast, flagcxRedNoOp,
         (size_t)((uintptr_t)comm), hashValue);
  }
  FLAGCXCHECK(planner.execute(sendbuff, recvbuff, datatype, root, stream));
  return flagcxSuccess;
}

flagcxResult_t hybridRunnerAllReduce(const void *sendbuff, void *recvbuff,
                                     size_t count, flagcxDataType_t datatype,
                                     flagcxRedOp_t op, flagcxComm_t comm,
                                     flagcxStream_t stream) {
  // Construct flagcxC2cPlanner and find corresponding strategy
  flagcxC2cPlanner planner;
  auto hashValue =
      getC2cCommPatternHash(count, comm->nclusters, flagcxCommOpAllReduce, op,
                            comm); // use nclusters as rootClusterId for hash
  if (!planCache.get(hashValue, planner)) {
    INFO(FLAGCX_COLL,
         "No available plan is found, create a new one with "
         "communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         count, comm->nclusters, flagcxCommOpAllReduce, op,
         (size_t)((uintptr_t)comm), hashValue);
    planner =
        flagcxC2cPlanner(count, count, -1, comm, flagcxCommOpAllReduce, op);
    planCache.put(hashValue, planner);
    // TODO: add estimator part
    // flagcxAlgoTimeEstimator estimator(planner, datatype);
    // float time = 0.0;
    // FLAGCXCHECK(estimator.getAlgoTime(&time));
  } else {
    INFO(FLAGCX_COLL,
         "Found available plan with communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         count, comm->nclusters, flagcxCommOpAllReduce, op,
         (size_t)((uintptr_t)comm), hashValue);
  }
  FLAGCXCHECK(planner.execute(sendbuff, recvbuff, datatype, -1, stream));
  return flagcxSuccess;
}

flagcxResult_t hybridRunnerReduceScatter(const void *sendbuff, void *recvbuff,
                                         size_t recvcount,
                                         flagcxDataType_t datatype,
                                         flagcxRedOp_t op, flagcxComm_t comm,
                                         flagcxStream_t stream) {
  // Construct flagcxC2cPlanner and find corresponding strategy
  flagcxC2cPlanner planner;
  auto hashValue = getC2cCommPatternHash(
      recvcount, comm->nclusters, flagcxCommOpReduceScatter, op,
      comm); // use nclusters as rootClusterId for hash
  if (!planCache.get(hashValue, planner)) {
    INFO(FLAGCX_COLL,
         "No available plan is found, create a new one with "
         "communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         recvcount, comm->nclusters, flagcxCommOpReduceScatter, op,
         (size_t)((uintptr_t)comm), hashValue);
    planner = flagcxC2cPlanner(comm->nranks * recvcount, recvcount, -1, comm,
                               flagcxCommOpReduceScatter, op);
    planCache.put(hashValue, planner);
  } else {
    INFO(FLAGCX_COLL,
         "Found available plan with communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         recvcount, comm->nclusters, flagcxCommOpReduceScatter, op,
         (size_t)((uintptr_t)comm), hashValue);
  }
  FLAGCXCHECK(planner.execute(sendbuff, recvbuff, datatype, -1, stream));
  return flagcxSuccess;
}

flagcxResult_t hybridRunnerAllGather(const void *sendbuff, void *recvbuff,
                                     size_t sendcount,
                                     flagcxDataType_t datatype,
                                     flagcxComm_t comm, flagcxStream_t stream) {
  // Construct flagcxC2cPlanner and find corresponding strategy
  flagcxC2cPlanner planner;
  auto hashValue = getC2cCommPatternHash(
      sendcount, comm->nclusters,
      flagcxCommOpAllGather, // use nclusters as rootClusterId for hash
      flagcxRedNoOp, comm);
  if (!planCache.get(hashValue, planner)) {
    INFO(FLAGCX_COLL,
         "No available plan is found, create a new one with "
         "communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         sendcount, comm->nclusters, flagcxCommOpAllGather, flagcxRedNoOp,
         (size_t)((uintptr_t)comm), hashValue);
    planner = flagcxC2cPlanner(sendcount, sendcount * comm->nranks, -1, comm,
                               flagcxCommOpAllGather, flagcxRedNoOp);
    planCache.put(hashValue, planner);
  } else {
    INFO(FLAGCX_COLL,
         "Found available plan with communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         sendcount, comm->nclusters, flagcxCommOpAllGather, flagcxRedNoOp,
         (size_t)((uintptr_t)comm), hashValue);
  }
  FLAGCXCHECK(planner.execute(sendbuff, recvbuff, datatype, -1, stream));
  return flagcxSuccess;
}

flagcxResult_t hybridRunnerAlltoAll(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    flagcxComm_t comm, flagcxStream_t stream) {
  // Construct flagcxC2cPlanner and find corresponding strategy
  flagcxC2cPlanner planner;
  auto hashValue =
      getC2cCommPatternHash(count, 1, // use 1 as rootClusterId for hash
                            flagcxCommOpAlltoAll, flagcxRedNoOp, comm);
  if (!planCache.get(hashValue, planner)) {
    INFO(FLAGCX_COLL,
         "No available plan is found, create a new one with "
         "communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         count, 1, flagcxCommOpAlltoAll, flagcxRedNoOp,
         (size_t)((uintptr_t)comm), hashValue);
    planner = flagcxC2cPlanner(count, count, -1, comm, flagcxCommOpAlltoAll,
                               flagcxRedNoOp);
    planCache.put(hashValue, planner);
  } else {
    INFO(FLAGCX_COLL,
         "Found available plan with communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         count, 1, flagcxCommOpAlltoAll, flagcxRedNoOp,
         (size_t)((uintptr_t)comm), hashValue);
  }
  FLAGCXCHECK(planner.execute(sendbuff, recvbuff, datatype, -1, stream));
  return flagcxSuccess;
}

flagcxResult_t hybridRunnerAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                     size_t *sdispls, void *recvbuff,
                                     size_t *recvcounts, size_t *rdispls,
                                     flagcxDataType_t datatype,
                                     flagcxComm_t comm, flagcxStream_t stream) {
  flagcxC2cPlanner planner;
  auto hashValue = getC2cCommPatternHash(
      1, 1, // use 1 both as count and rootClusterId for hash
      flagcxCommOpAlltoAllv, flagcxRedNoOp, comm);
  if (!planCache.get(hashValue, planner)) {
    INFO(FLAGCX_COLL,
         "No available plan is found, create a new one with "
         "communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%d, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         1, 1, flagcxCommOpAlltoAllv, flagcxRedNoOp, (size_t)((uintptr_t)comm),
         hashValue);
    planner =
        flagcxC2cPlanner(1, 1, -1, comm, flagcxCommOpAlltoAllv, flagcxRedNoOp);
    planCache.put(hashValue, planner);
  } else {
    INFO(FLAGCX_COLL,
         "Found available plan with communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%d, %d, %d, %d, "
         "%ld), hashValue = "
         "%ld",
         1, 1, flagcxCommOpAlltoAllv, flagcxRedNoOp, (size_t)((uintptr_t)comm),
         hashValue);
  }
  FLAGCXCHECK(planner.execute(sendbuff, recvbuff, datatype, -1, stream,
                              sendcounts, sdispls, recvcounts, rdispls));
  return flagcxSuccess;
}

flagcxResult_t hybridRunnerSend(const void *sendbuff, size_t count,
                                flagcxDataType_t datatype, int peer,
                                flagcxComm_t comm, flagcxStream_t stream) {
  if (comm->cluster_ids[comm->rank] == comm->cluster_ids[peer]) {
    FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->send(
        sendbuff, count, datatype, comm->globalrank2homorank[peer],
        comm->homo_comm, stream));
  } else {
    FLAGCXCHECK(flagcxHeteroSend(sendbuff, count, datatype, peer,
                                 comm->hetero_comm, stream));
  }
  return flagcxSuccess;
}

flagcxResult_t hybridRunnerRecv(void *recvbuff, size_t count,
                                flagcxDataType_t datatype, int peer,
                                flagcxComm_t comm, flagcxStream_t stream) {
  if (comm->cluster_ids[comm->rank] == comm->cluster_ids[peer]) {
    FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->recv(
        recvbuff, count, datatype, comm->globalrank2homorank[peer],
        comm->homo_comm, stream));
  } else {
    FLAGCXCHECK(flagcxHeteroRecv(recvbuff, count, datatype, peer,
                                 comm->hetero_comm, stream));
  }
  return flagcxSuccess;
}

flagcxResult_t hybridRunnerGroupStart() {
  FLAGCXCHECK(flagcxHeteroGroupStart());
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->groupStart());
  return flagcxSuccess;
}

flagcxResult_t hybridRunnerGroupEnd() {
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->groupEnd());
  FLAGCXCHECK(flagcxHeteroGroupEnd());
  return flagcxSuccess;
}

struct flagcxRunner hybridRunner = {
    // Communication functions
    hybridRunnerReduce, hybridRunnerGather, hybridRunnerScatter,
    hybridRunnerBroadcast, hybridRunnerAllReduce, hybridRunnerReduceScatter,
    hybridRunnerAllGather, hybridRunnerAlltoAll, hybridRunnerAlltoAllv,
    hybridRunnerSend, hybridRunnerRecv,
    // Group semantics
    hybridRunnerGroupStart, hybridRunnerGroupEnd};