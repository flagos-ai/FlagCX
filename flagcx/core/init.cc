/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "adaptor.h"
#include "bootstrap.h"
#include "check.h"
#include "collectives.h"
#include "flagcx.h"
#include "group.h"
#include "net.h"
#include "p2p.h"
#include "topo.h"
#include "transport.h"
#include "type.h"
#include "utils.h"
#include <algorithm>
#include <string.h>

static bool initialized = false;
pthread_mutex_t initLock = PTHREAD_MUTEX_INITIALIZER;

struct flagcxCommInitRankAsyncJob {
  struct flagcxAsyncJob base;
  struct flagcxHeteroComm *comm;
  struct flagcxHeteroComm **newcomm;
  int cudaDev;
  // For flagcxCommInitRank
  int nranks, myrank;
  flagcxUniqueId commId;
  // for flagcxCommSplit
  struct flagcxHeteroComm *parent;
  int color, key;
};

flagcxResult_t flagcxHeteroGetVersion(int *version) {
  if (version == NULL)
    return flagcxInvalidArgument;
  *version = FLAGCX_VERSION(1, 0, 0);
  return flagcxSuccess;
}

static flagcxResult_t flagcxInit() {
  if (__atomic_load_n(&initialized, __ATOMIC_ACQUIRE))
    return flagcxSuccess;
  pthread_mutex_lock(&initLock);
  if (!initialized) {
    // FLAGCXCHECK(loadDeviceSymbol());
    FLAGCXCHECK(bootstrapNetInit());
    // FLAGCXCHECK(flagcxNetPluginInit());
    __atomic_store_n(&initialized, true, __ATOMIC_RELEASE);
  }
  pthread_mutex_unlock(&initLock);
  return flagcxSuccess;
}

flagcxResult_t flagcxHeteroGetUniqueId(flagcxUniqueId *out) {
  FLAGCXCHECK(flagcxInit());
  flagcxResult_t res =
      bootstrapGetUniqueId((struct flagcxBootstrapHandle *)out);
  return res;
}

static uint64_t hashUniqueId(flagcxUniqueId const &id) {
  char const *bytes = (char const *)&id;
  uint64_t h = 0xdeadbeef;
  for (int i = 0; i < (int)sizeof(flagcxUniqueId); i++) {
    h ^= h >> 32;
    h *= 0x8db3db47fa2994ad;
    h += bytes[i];
  }
  return h;
}

static flagcxResult_t fillPeerInfo(flagcxHeteroComm_t comm,
                                   struct flagcxPeerInfo *info,
                                   uint64_t commHash) {
  info->rank = comm->rank;
  info->cudaDev = comm->cudaDev;
  info->hostHash = getHostHash() + commHash;
  info->pidHash = getPidHash() + commHash;
  info->busId = comm->busId;
  info->comm = comm;

  return flagcxSuccess;
}

static flagcxResult_t initTransportsRank(flagcxHeteroComm_t comm,
                                         flagcxHeteroComm_t parent) {
  INFO(FLAGCX_INIT, "inside initTransportsRank");
  flagcxResult_t ret = flagcxSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  int nNodes = 1;

  // fill peer info
  FLAGCXCHECKGOTO(flagcxCalloc(&comm->peerInfo, nranks), ret, fail);
  INFO(FLAGCX_INIT, "start fillPeerInfo");
  FLAGCXCHECKGOTO(fillPeerInfo(comm, comm->peerInfo + rank, comm->commHash),
                  ret, fail);
  // Question: where did we initialize comm->bootstrap?
  INFO(FLAGCX_INIT, "start bootstrapAllGather for peerInfo");
  FLAGCXCHECKGOTO(bootstrapAllGather(comm->bootstrap, (void *)comm->peerInfo,
                                     sizeof(struct flagcxPeerInfo)),
                  ret, fail);
  FLAGCXCHECKGOTO(bootstrapBarrier(comm->bootstrap, rank, nranks, 0), ret,
                  fail);

  // check for duplicate GPUs
  INFO(FLAGCX_INIT, "start check for duplicate GPUs");
  for (int i = 0; i < nranks; i++) {
    if (comm->peerInfo[i].hostHash != comm->peerInfo[rank].hostHash)
      nNodes++;
    if ((i != rank) &&
        (comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash) &&
        (comm->peerInfo[i].busId == comm->peerInfo[rank].busId)) {
      WARN("Duplicate GPU detected : rank %d and rank %d both on CUDA device "
           "%lx",
           rank, i, comm->peerInfo[rank].busId);
      ret = flagcxInvalidUsage;
      goto fail;
    }
  }

  {
    FLAGCXCHECKGOTO(flagcxCalloc(&comm->rankToNode, nranks), ret, fail);
    int *nodesFirstRank = NULL;
    FLAGCXCHECKGOTO(flagcxCalloc(&nodesFirstRank, nranks), ret, fail);
    comm->nNodes = 0;
    for (int r = 0; r < nranks; r++) {
      int node;
      for (node = 0; node < comm->nNodes; node++) {
        if (comm->peerInfo[nodesFirstRank[node]].hostHash ==
            comm->peerInfo[r].hostHash)
          break;
      }
      if (node == comm->nNodes) {
        nodesFirstRank[comm->nNodes] = r;
        comm->nNodes++;
      }
      comm->rankToNode[r] = node;
    }

    // Allocate nodeRanks and count localRanks per node
    FLAGCXCHECKGOTO(flagcxCalloc(&comm->nodeRanks, comm->nNodes), ret, fail);
    FLAGCXCHECKGOTO(flagcxCalloc(&comm->rankToLocalRank, nranks), ret, fail);
    for (int r = 0; r < nranks; r++) {
      int node = comm->rankToNode[r];
      comm->rankToLocalRank[r] = comm->nodeRanks[node].localRanks;
      comm->nodeRanks[node].localRanks++;
    }

    // Allocate localRankToRank arrays and find maxLocalRanks
    comm->maxLocalRanks = 0;
    for (int n = 0; n < comm->nNodes; n++) {
      FLAGCXCHECKGOTO(flagcxCalloc(&comm->nodeRanks[n].localRankToRank,
                                   comm->nodeRanks[n].localRanks),
                      ret, fail);
      comm->maxLocalRanks =
          std::max(comm->maxLocalRanks, comm->nodeRanks[n].localRanks);
      comm->nodeRanks[n].localRanks = 0; // Reset for filling
    }

    // Fill localRankToRank arrays
    for (int r = 0; r < nranks; r++) {
      int node = comm->rankToNode[r];
      comm->nodeRanks[node]
          .localRankToRank[comm->nodeRanks[node].localRanks++] = r;
    }

    // Set local info for this rank
    comm->node = comm->rankToNode[rank];
    comm->localRank = comm->rankToLocalRank[rank];
    comm->localRanks = comm->nodeRanks[comm->node].localRanks;
    comm->localRankToRank = comm->nodeRanks[comm->node].localRankToRank;

    // Build p2pSchedule with two-level scheduling (like NCCL)
    int node = comm->node;
    int local = comm->localRank;
    int nLocals = comm->maxLocalRanks;
    struct flagcxNodeRanks *nodeRanks = comm->nodeRanks;
    bool flat = false;
    for (int n = 0; n < comm->nNodes; n++) {
      if (comm->nodeRanks[n].localRanks != nLocals) {
        flat = true;
        comm->nNodes = 1;
        node = 0;
        nLocals = nranks;
        local = rank;
        break;
      }
    }
    int nNodesPow2 = pow2Up(comm->nNodes);
    int nLocalsPow2 = pow2Up(nLocals);
    uint32_t nodeRound = 0;
    uint32_t nodeDelta = 0;
    int round = 0;
    do {
      if ((int)nodeDelta < comm->nNodes) { // Filter nonsensical node deltas
        int sendNode = (node + nodeDelta) % comm->nNodes;
        int recvNode = (node - nodeDelta + comm->nNodes) % comm->nNodes;
        uint32_t localRound = 0;
        uint32_t localDelta = 0;
        do {
          if ((int)localDelta < nLocals) { // Filter nonsensical local deltas
            int sendLocal = (local + localDelta) % nLocals;
            int recvLocal = (local - localDelta + nLocals) % nLocals;
            comm->p2pSchedule[round].sendRank =
                flat ? sendLocal
                     : nodeRanks[sendNode].localRankToRank[sendLocal];
            comm->p2pSchedule[round].recvRank =
                flat ? recvLocal
                     : nodeRanks[recvNode].localRankToRank[recvLocal];
            round += 1;
          }
          localRound += 1;
          localDelta =
              (localDelta + localRound) & (nLocalsPow2 - 1); // Quadratic update
        } while (localRound != (uint32_t)nLocalsPow2);
      }
      nodeRound += 1;
      nodeDelta = (nodeDelta + nodeRound) & (nNodesPow2 - 1);
    } while (nodeRound != (uint32_t)nNodesPow2);

    if (round != nranks) {
      WARN("P2p schedule creation has bugs: round=%d nranks=%d", round, nranks);
      ret = flagcxInternalError;
      free(nodesFirstRank);
      goto fail;
    }
    free(nodesFirstRank);
  }

  INFO(FLAGCX_INIT, "start flagcxTopoGetServerTopo");
  FLAGCXCHECKGOTO(flagcxTopoGetServerTopo(comm, &comm->topoServer), ret, fail);
  FLAGCXCHECKGOTO(flagcxTopoComputePaths(comm->topoServer, comm), ret, fail);
  if (comm->rank == 0) {
    FLAGCXCHECK(flagcxTopoPrint(comm->topoServer));
  }
  INFO(FLAGCX_INIT, "start getting local net from gpu");
  FLAGCXCHECKGOTO(flagcxGetLocalNetFromGpu(comm->cudaDev, &comm->netDev, comm),
                  ret, fail);

  INFO(FLAGCX_INIT, "start getting topoServer from other servers");
  FLAGCXCHECKGOTO(
      flagcxGetInterServerTopo(comm, &comm->interServerTopo, comm->topoServer),
      ret, fail);

  return ret;
fail:
  return flagcxInternalError;
}

FLAGCX_PARAM(P2PBufferSize, "P2P_BUFFER_SIZE", 64L * 1024 * 1024); // default value to 64MB
FLAGCX_PARAM(P2PChunkSize, "P2P_CHUNK_SIZE", 4L * 1024 * 1024); // default value to 4MB
FLAGCX_PARAM(NetBufferSize, "NET_BUFFER_SIZE", 64L * 1024 * 1024); // default value to 64MB
FLAGCX_PARAM(NetChunkSize, "NET_CHUNK_SIZE", 4L * 1024 * 1024); // default value to 4MB

static flagcxResult_t flagcxCommInitRankFunc(struct flagcxAsyncJob *job_) {
  struct flagcxCommInitRankAsyncJob *job =
      (struct flagcxCommInitRankAsyncJob *)job_;
  flagcxHeteroComm_t comm = job->comm;
  flagcxResult_t res = flagcxSuccess;
  const char *env = flagcxGetEnv("FLAGCX_ENABLE_TOPO_DETECT");

  if (!job->parent) {
    // New version of calling bootstrapInit
    struct bootstrapState *state;
    FLAGCXCHECK(flagcxCalloc(&state, 1));
    state->rank = comm->rank;
    state->nranks = comm->nRanks;
    state->abortFlag = comm->abortFlag;
    comm->bootstrap = state;
    state->magic = ((struct flagcxBootstrapHandle *)&job->commId)->magic;
    comm->magic = ((struct flagcxBootstrapHandle *)&job->commId)->magic;
    FLAGCXCHECKGOTO(
        bootstrapInit((struct flagcxBootstrapHandle *)&job->commId, state), res,
        fail);
  }

  if (!job->parent) {
    // Setting up proxy network
    int nranks = comm->nRanks;
    for (int i = 0; i < MAXCHANNELS; i++) {
      FLAGCXCHECK(flagcxCalloc(&comm->channels[i].peers, nranks));
      for (int r = 0; r < nranks; r++)
        FLAGCXCHECK(flagcxCalloc(&comm->channels[i].peers[r], nranks));
    }
    FLAGCXCHECK(flagcxCalloc(&comm->connectSend, nranks));
    FLAGCXCHECK(flagcxCalloc(&comm->connectRecv, nranks));
    FLAGCXCHECK(flagcxCalloc(&comm->proxyState, 1));
    FLAGCXCHECK(flagcxCalloc(&comm->tasks.peers, nranks));
    FLAGCXCHECK(flagcxCalloc(&comm->tasks.p2pOrder, 2 * nranks));
    FLAGCXCHECK(flagcxCalloc(&comm->p2pSchedule, nranks));
    // Setup mutex/cond to work inter-process
    pthread_mutexattr_t mutexAttr;
    pthread_mutexattr_init(&mutexAttr);
    pthread_mutexattr_setpshared(&mutexAttr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&comm->proxyState->mutex, &mutexAttr);
    pthread_condattr_t condAttr;
    pthread_condattr_init(&condAttr);
    pthread_condattr_setpshared(&condAttr, PTHREAD_PROCESS_SHARED);
    pthread_cond_init(&comm->proxyState->cond, &condAttr);

    for (int i = 0; i < MAXCHANNELS; i++) {
      FLAGCXCHECK(
          flagcxCalloc(&comm->proxyState->proxyOps[i].consPeers, nranks));
      comm->proxyState->proxyOps[i].consNextChannel =
          reinterpret_cast<struct flagcxProxyOps *>(0x1);
      comm->proxyState->proxyOps[i].prodNextChannel =
          reinterpret_cast<struct flagcxProxyOps *>(0x1);
      pthread_mutex_init(&comm->proxyState->proxyOps[i].mutex, 0);
      for (int peer = 0; peer < nranks; peer++) {
        comm->proxyState->proxyOps[i].consPeers[peer].nextPeer =
            reinterpret_cast<struct flagcxProxyOps::consPeer *>(0x1);
      }
    }

    comm->groupNext = reinterpret_cast<struct flagcxHeteroComm *>(0x1);
    comm->preconnectNext = reinterpret_cast<struct flagcxHeteroComm *>(0x1);
    comm->proxyState->nRanks = comm->nRanks;

    bool runtimeProxy = false;
    const char *runtimeEnv = flagcxGetEnv("FLAGCX_RUNTIME_PROXY");
    if (runtimeEnv) {
      runtimeProxy = (std::stoi(runtimeEnv) == 1) ? true : false;
    }
    INFO(FLAGCX_INIT, "Flagcx RuntimeProxy flag set to %d", runtimeProxy);
    if (!runtimeProxy) {
      FLAGCXCHECK(flagcxProxyInit(comm));
    }
  }

  flagcxNetBufferSize = flagcxParamNetBufferSize();
  flagcxNetChunkSize = flagcxParamNetChunkSize();
  flagcxP2PBufferSize = flagcxParamP2PBufferSize();
  flagcxP2PChunkSize = flagcxParamP2PChunkSize();
  assert((flagcxNetBufferSize + flagcxNetChunkSize - 1) / flagcxNetChunkSize <= FLAGCX_NET_MAX_STEPS);
  assert((flagcxP2PBufferSize + flagcxP2PChunkSize - 1) / flagcxP2PChunkSize <= FLAGCX_P2P_MAX_STEPS);

  FLAGCXCHECK(flagcxNetInit(comm));
  INFO(FLAGCX_INIT, "Using network %s", comm->netAdaptor->name);
  if (env && (strcmp(env, "TRUE") == 0 || strcmp(env, "True") == 0)) {
    INFO(FLAGCX_INIT, "getting busId for cudaDev %d", comm->cudaDev);
    FLAGCXCHECK(getBusId(comm->cudaDev, &comm->busId));
    INFO(FLAGCX_INIT, "getting commHash for rank %d", comm->rank);
    comm->commHash = getHash(job->commId.internal, FLAGCX_UNIQUE_ID_BYTES);
    INFO(FLAGCX_INIT, "commHash for rank %d is %lu", comm->rank,
         comm->commHash);
    // TODO: put net init into a separate function

    INFO(FLAGCX_INIT, "start initTransportsRank");
    FLAGCXCHECKGOTO(initTransportsRank(comm, NULL), res, fail);
  } else {
    flagcxGetLocalNetFromGpu(comm->cudaDev, &comm->netDev, comm);
  }

exit:
  return res;
fail:
  comm->initState = res;
  goto exit;
}

static flagcxResult_t flagcxCommInitRankDev(flagcxHeteroComm_t *newcomm,
                                            int nranks, flagcxUniqueId commId,
                                            int myrank, int cudaDev,
                                            flagcxConfig_t *config) {
  flagcxResult_t res = flagcxSuccess;
  flagcxHeteroComm_t comm = NULL;
  struct flagcxCommInitRankAsyncJob *job = NULL;
  const char *env = flagcxGetEnv("FLAGCX_COMM_ID");

  if (env && myrank == 0) {
    INFO(FLAGCX_ENV, "FLAGCX_COMM_ID set by environment to %s", env);
    FLAGCXCHECKGOTO(
        bootstrapCreateRoot((struct flagcxBootstrapHandle *)&commId, true), res,
        fail);
  }

  if (nranks < 1 || myrank < 0 || myrank >= nranks) {
    WARN("Invalid rank requested : %d/%d", myrank, nranks);
    res = flagcxInvalidArgument;
    goto fail;
  }

  FLAGCXCHECKGOTO(flagcxCalloc(&comm, 1), res, fail);
  comm->startMagic = comm->endMagic =
      FLAGCX_MAGIC; // Used to detect comm corruption.
  FLAGCXCHECKGOTO(flagcxCalloc((uint32_t **)&comm->abortFlagRefCount, 1), res,
                  fail);
  *comm->abortFlagRefCount = 1;
  /* start with flagcxInternalError and will be changed to flagcxSuccess if init
   * succeeds. */
  comm->initState = flagcxInternalError;
  comm->nRanks = nranks;
  comm->rank = myrank;
  comm->cudaDev = cudaDev;
  *newcomm = comm;

  FLAGCXCHECKGOTO(flagcxCalloc(&job, 1), res, fail);
  job->comm = comm;
  job->nranks = nranks;
  job->commId = commId; // C++ struct assignment
  job->myrank = myrank;
  job->cudaDev = cudaDev;
  FLAGCXCHECKGOTO(flagcxCommInitRankFunc(&job->base), res, fail);
  free(job);
exit:
  return flagcxGroupErrCheck(res);
fail:
  if (comm) {
    if (comm->abortFlagRefCount)
      free(comm->abortFlagRefCount);
    free(comm);
  }
  if (newcomm)
    *newcomm = NULL;
  goto exit;
}

flagcxResult_t flagcxHeteroCommInitRank(flagcxHeteroComm_t *newcomm, int nranks,
                                        flagcxUniqueId commId, int myrank) {
  FLAGCXCHECK(flagcxInit());
  int cudaDev = 0;
  flagcxConfig_t config;
  // flagcxGetDevice(&cudaDev);
  deviceAdaptor->getDevice(&cudaDev);
  FLAGCXCHECK(
      flagcxCommInitRankDev(newcomm, nranks, commId, myrank, cudaDev, &config));
  return flagcxSuccess;
}

flagcxResult_t flagcxHeteroCommCount(const flagcxHeteroComm_t comm,
                                     int *count) {
  *count = comm->nRanks;
  return flagcxSuccess;
}

flagcxResult_t flagcxHeteroCommUserRank(const flagcxHeteroComm_t comm,
                                        int *rank) {
  *rank = comm->rank;
  return flagcxSuccess;
}

flagcxResult_t flagcxHeteroCommDestroy(flagcxHeteroComm_t comm) {
  flagcxProxyDestroy(comm);
  for (int i = 0; i < MAXCHANNELS; i++) {
    for (int r = 0; r < comm->nRanks; r++) {
      free(comm->channels[i].peers[r]);
    }
    free(comm->channels[i].peers);
  }
  for (int i = 0; i < MAXCHANNELS; i++) {
    free(comm->proxyState->proxyOps[i].consPeers);
  }

  free(comm->connectSend);
  free(comm->connectRecv);
  free(comm->proxyState);
  free(comm->tasks.peers);
  free(comm->tasks.p2pOrder);
  free(comm->abortFlagRefCount);
  if (comm->topoServer) {
    flagcxTopoFree(comm->topoServer);
  }
  if (comm->interServerTopo) {
    flagcxInterServerTopoFree(comm->interServerTopo);
  }
  free(comm->peerInfo);
  free(comm);

  return flagcxSuccess;
}
