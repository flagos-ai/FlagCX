/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "c2c_algo.h"
#include "runner.h"
#include "runner_result.h"

#define FLAGCX_CACHE_CAPACITY 16
static flagcxLRUCache<C2cPatternKey, flagcxC2cPlanner, C2cPatternKeyHash>
    planCache(FLAGCX_CACHE_CAPACITY);

flagcxResult_t hybridRunnerReduce(const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  flagcxRedOp_t op, int root, flagcxComm_t comm,
                                  flagcxStream_t stream) {
  // Construct flagcxC2cPlanner and find corresponding strategy
  flagcxC2cPlanner planner;
  C2cPatternKey key = {count, (size_t)comm->clusterIds[root],
                       flagcxCommOpReduce, op, (uintptr_t)comm};
  if (!planCache.get(key, planner)) {
    INFO(FLAGCX_COLL,
         "No available plan is found, create a new one with "
         "communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %ld, %d, %d, "
         "%ld)",
         key.count, key.rootClusterId, key.commOp, key.redOp, key.comm);
    planner =
        flagcxC2cPlanner(count, count, root, comm, flagcxCommOpReduce, op);
    planCache.put(key, planner);
  } else {
    INFO(FLAGCX_COLL,
         "Found available plan with communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %ld, %d, %d, "
         "%ld)",
         key.count, key.rootClusterId, key.commOp, key.redOp, key.comm);
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
  C2cPatternKey key = {count, (size_t)root, flagcxCommOpGather, flagcxRedNoOp,
                       (uintptr_t)comm};
  if (!planCache.get(key, planner)) {
    INFO(FLAGCX_COLL,
         "No available plan is found, create a new one with "
         "communication pattern "
         "(count, rootRank, commOp, redOp, comm) = (%ld, %ld, %d, %d, %ld)",
         key.count, key.rootClusterId, key.commOp, key.redOp, key.comm);
    planner = flagcxC2cPlanner(count, count * comm->nranks, root, comm,
                               flagcxCommOpGather, flagcxRedNoOp);
    planCache.put(key, planner);
  } else {
    INFO(FLAGCX_COLL,
         "Found available plan with communication pattern "
         "(count, rootRank, commOp, redOp, comm) = (%ld, %ld, %d, %d, %ld)",
         key.count, key.rootClusterId, key.commOp, key.redOp, key.comm);
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
  C2cPatternKey key = {count, (size_t)root, flagcxCommOpScatter, flagcxRedNoOp,
                       (uintptr_t)comm};
  if (!planCache.get(key, planner)) {
    INFO(FLAGCX_COLL,
         "No available plan is found, create a new one with "
         "communication pattern "
         "(count, rootRank, commOp, redOp, comm) = (%ld, %ld, %d, %d, %ld)",
         key.count, key.rootClusterId, key.commOp, key.redOp, key.comm);
    planner = flagcxC2cPlanner(count * comm->nranks, count, root, comm,
                               flagcxCommOpScatter, flagcxRedNoOp);
    planCache.put(key, planner);
  } else {
    INFO(FLAGCX_COLL,
         "Found available plan with communication pattern "
         "(count, rootRank, commOp, redOp, comm) = (%ld, %ld, %d, %d, %ld)",
         key.count, key.rootClusterId, key.commOp, key.redOp, key.comm);
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
  C2cPatternKey key = {count, (size_t)comm->clusterIds[root],
                       flagcxCommOpBroadcast, flagcxRedNoOp, (uintptr_t)comm};
  if (!planCache.get(key, planner)) {
    INFO(FLAGCX_COLL,
         "No available plan is found, create a new one with "
         "communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %ld, %d, %d, "
         "%ld)",
         key.count, key.rootClusterId, key.commOp, key.redOp, key.comm);
    planner = flagcxC2cPlanner(count, count, root, comm, flagcxCommOpBroadcast,
                               flagcxRedNoOp);
    planCache.put(key, planner);
  } else {
    INFO(FLAGCX_COLL,
         "Found available plan with communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %ld, %d, %d, "
         "%ld)",
         key.count, key.rootClusterId, key.commOp, key.redOp, key.comm);
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
  C2cPatternKey key = {count, (size_t)comm->nclusters, flagcxCommOpAllReduce,
                       op, (uintptr_t)comm};
  if (!planCache.get(key, planner)) {
    INFO(FLAGCX_COLL,
         "No available plan is found, create a new one with "
         "communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %ld, %d, %d, "
         "%ld)",
         key.count, key.rootClusterId, key.commOp, key.redOp, key.comm);
    planner =
        flagcxC2cPlanner(count, count, -1, comm, flagcxCommOpAllReduce, op);
    planCache.put(key, planner);
  } else {
    INFO(FLAGCX_COLL,
         "Found available plan with communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %ld, %d, %d, "
         "%ld)",
         key.count, key.rootClusterId, key.commOp, key.redOp, key.comm);
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
  C2cPatternKey key = {recvcount, (size_t)comm->nclusters,
                       flagcxCommOpReduceScatter, op, (uintptr_t)comm};
  if (!planCache.get(key, planner)) {
    INFO(FLAGCX_COLL,
         "No available plan is found, create a new one with "
         "communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %ld, %d, %d, "
         "%ld)",
         key.count, key.rootClusterId, key.commOp, key.redOp, key.comm);
    planner = flagcxC2cPlanner(comm->nranks * recvcount, recvcount, -1, comm,
                               flagcxCommOpReduceScatter, op);
    planCache.put(key, planner);
  } else {
    INFO(FLAGCX_COLL,
         "Found available plan with communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %ld, %d, %d, "
         "%ld)",
         key.count, key.rootClusterId, key.commOp, key.redOp, key.comm);
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
  C2cPatternKey key = {sendcount, (size_t)comm->nclusters,
                       flagcxCommOpAllGather, flagcxRedNoOp, (uintptr_t)comm};
  if (!planCache.get(key, planner)) {
    INFO(FLAGCX_COLL,
         "No available plan is found, create a new one with "
         "communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %ld, %d, %d, "
         "%ld)",
         key.count, key.rootClusterId, key.commOp, key.redOp, key.comm);
    planner = flagcxC2cPlanner(sendcount, sendcount * comm->nranks, -1, comm,
                               flagcxCommOpAllGather, flagcxRedNoOp);
    planCache.put(key, planner);
  } else {
    INFO(FLAGCX_COLL,
         "Found available plan with communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %ld, %d, %d, "
         "%ld)",
         key.count, key.rootClusterId, key.commOp, key.redOp, key.comm);
  }
  FLAGCXCHECK(planner.execute(sendbuff, recvbuff, datatype, -1, stream));
  return flagcxSuccess;
}

flagcxResult_t hybridRunnerAlltoAll(const void *sendbuff, void *recvbuff,
                                    size_t count, flagcxDataType_t datatype,
                                    flagcxComm_t comm, flagcxStream_t stream) {
  // Construct flagcxC2cPlanner and find corresponding strategy
  flagcxC2cPlanner planner;
  C2cPatternKey key = {count, 1, flagcxCommOpAlltoAll, flagcxRedNoOp,
                       (uintptr_t)comm};
  if (!planCache.get(key, planner)) {
    INFO(FLAGCX_COLL,
         "No available plan is found, create a new one with "
         "communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %ld, %d, %d, "
         "%ld)",
         key.count, key.rootClusterId, key.commOp, key.redOp, key.comm);
    planner = flagcxC2cPlanner(count, count, -1, comm, flagcxCommOpAlltoAll,
                               flagcxRedNoOp);
    planCache.put(key, planner);
  } else {
    INFO(FLAGCX_COLL,
         "Found available plan with communication pattern "
         "(count, rootClusterId, commOp, redOp, comm) = (%ld, %ld, %d, %d, "
         "%ld)",
         key.count, key.rootClusterId, key.commOp, key.redOp, key.comm);
  }
  FLAGCXCHECK(planner.execute(sendbuff, recvbuff, datatype, -1, stream));
  return flagcxSuccess;
}

flagcxResult_t hybridRunnerAlltoAllv(const void *sendbuff, size_t *sendcounts,
                                     size_t *sdispls, void *recvbuff,
                                     size_t *recvcounts, size_t *rdispls,
                                     flagcxDataType_t datatype,
                                     flagcxComm_t comm, flagcxStream_t stream) {
  // AlltoAllv plans contain the per-peer counts and displacements. The
  // existing C2cPatternKey only identifies the communicator and operation, so
  // caching here would replay stale offsets when a later call uses a different
  // layout. Rebuild until the cache key includes the complete variable-count
  // pattern.
  flagcxC2cPlanner planner(1, 1, -1, comm, flagcxCommOpAlltoAllv,
                           flagcxRedNoOp);
  FLAGCXCHECK(planner.execute(sendbuff, recvbuff, datatype, -1, stream,
                              sendcounts, sdispls, recvcounts, rdispls));
  return flagcxSuccess;
}

flagcxResult_t hybridRunnerSend(const void *sendbuff, size_t count,
                                flagcxDataType_t datatype, int peer,
                                flagcxComm_t comm, flagcxStream_t stream) {
  if (comm->clusterIds[comm->rank] == comm->clusterIds[peer]) {
    FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->send(
        sendbuff, count, datatype, comm->globalRank2HomoRank[peer],
        comm->homoComm, stream));
  } else {
    FLAGCXCHECK(flagcxHeteroSend(sendbuff, count, datatype, peer,
                                 comm->heteroComm, stream));
  }
  return flagcxSuccess;
}

flagcxResult_t hybridRunnerRecv(void *recvbuff, size_t count,
                                flagcxDataType_t datatype, int peer,
                                flagcxComm_t comm, flagcxStream_t stream) {
  if (comm->clusterIds[comm->rank] == comm->clusterIds[peer]) {
    FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->recv(
        recvbuff, count, datatype, comm->globalRank2HomoRank[peer],
        comm->homoComm, stream));
  } else {
    FLAGCXCHECK(flagcxHeteroRecv(recvbuff, count, datatype, peer,
                                 comm->heteroComm, stream));
  }
  return flagcxSuccess;
}

flagcxResult_t hybridRunnerGroupStart() {
  flagcxResult_t result = flagcxHeteroGroupStart();
  if (result != flagcxSuccess)
    return result;
  result = cclAdaptors[flagcxCCLAdaptorDevice]->groupStart();
  if (result != flagcxSuccess) {
    // Roll back the group that was already opened, but preserve the CCL
    // start failure returned to the caller.
    flagcxHeteroGroupEnd();
    return result;
  }
  return flagcxSuccess;
}

flagcxResult_t hybridRunnerGroupEnd() {
  flagcxResult_t result = cclAdaptors[flagcxCCLAdaptorDevice]->groupEnd();
  flagcxRunnerRecordFirstError(flagcxHeteroGroupEnd(), &result);
  return result;
}

struct flagcxRunner hybridRunner = {
    // Communication functions
    hybridRunnerReduce, hybridRunnerGather, hybridRunnerScatter,
    hybridRunnerBroadcast, hybridRunnerAllReduce, hybridRunnerReduceScatter,
    hybridRunnerAllGather, hybridRunnerAlltoAll, hybridRunnerAlltoAllv,
    hybridRunnerSend, hybridRunnerRecv,
    // Group semantics
    hybridRunnerGroupStart, hybridRunnerGroupEnd};
