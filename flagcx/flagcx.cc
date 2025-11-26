#include "flagcx.h"
#include "adaptor.h"
#include "alloc.h"
#include "bootstrap.h"
#include "check.h"
#include "cluster.h"
#include "comm.h"
#include "cost_model.h"
#include "flagcx_hetero.h"
#include "launch_kernel.h"
#include "param.h"
#include "proxy.h"
#include "reg_pool.h"
#include "runner.h"
#include "timer.h"
#include "utils.h"
#include <cassert>
#include <stdio.h>
#include <string.h>
#include <unordered_map>

flagcxRegPool globalRegPool;

size_t getFlagcxDataTypeSize(flagcxDataType_t dtype) {
  switch (dtype) {
    // case flagcxInt8:
    case flagcxChar:
      return sizeof(char); // 1 byte
    case flagcxUint8:
      return sizeof(unsigned char); // 1 byte
    // case flagcxInt32:
    case flagcxInt:
      return sizeof(int); // 4 bytes
    case flagcxUint32:
      return sizeof(unsigned int); // 4 bytes
    case flagcxInt64:
      return sizeof(long long); // 8 bytes
    case flagcxUint64:
      return sizeof(unsigned long long); // 8 bytes
    // case flagcxFloat16:
    case flagcxHalf:
      return 2; // Half precision float is 2 bytes
    // case flagcxFloat32:
    case flagcxFloat:
      return sizeof(float); // 4 bytes
    // case flagcxFloat64:
    case flagcxDouble:
      return sizeof(double); // 8 bytes
    case flagcxBfloat16:
      return 2; // BFloat16 is typically 2 bytes
    default:
      fprintf(stderr, "Unknown flagcx data type\n");
      return 0;
  }
}

// Wrapper function for deviceMemcpy without the usage of invalid args
flagcxResult_t wrapper_deviceMemcpy(void *dst, void *src, size_t size,
                                    flagcxMemcpyType_t type,
                                    flagcxStream_t stream) {
  return deviceAdaptor->deviceMemcpy(dst, src, size, type, stream, NULL);
}

static struct flagcxDeviceHandle globalDeviceHandle {
  // Basic functions
  deviceAdaptor->deviceSynchronize, wrapper_deviceMemcpy,
      deviceAdaptor->deviceMemset, deviceAdaptor->deviceMalloc,
      deviceAdaptor->deviceFree, deviceAdaptor->setDevice,
      deviceAdaptor->getDevice, deviceAdaptor->getDeviceCount,
      deviceAdaptor->getVendor, deviceAdaptor->hostGetDevicePointer,
      // Stream functions
      deviceAdaptor->streamCreate, deviceAdaptor->streamDestroy,
      deviceAdaptor->streamCopy, deviceAdaptor->streamFree,
      deviceAdaptor->streamSynchronize, deviceAdaptor->streamQuery,
      deviceAdaptor->streamWaitEvent,
      // Event functions
      deviceAdaptor->eventCreate, deviceAdaptor->eventDestroy,
      deviceAdaptor->eventRecord, deviceAdaptor->eventSynchronize,
      deviceAdaptor->eventQuery,
      // IpcMemHandle functions
      deviceAdaptor->ipcMemHandleCreate, deviceAdaptor->ipcMemHandleGet,
      deviceAdaptor->ipcMemHandleOpen, deviceAdaptor->ipcMemHandleClose,
      deviceAdaptor->ipcMemHandleFree,
};

flagcxResult_t flagcxEnsureCommReady(flagcxComm_t comm) {
  if (comm == NULL) {
    return flagcxInternalError;
  }
  if (comm->comm_type != flagcxCommunicatorHybrid &&
      comm->comm_type != flagcxCommunicatorHomo) {
    return flagcxInternalError;
  }
  return flagcxSuccess;
}

bool useHomoComm(flagcxComm_t comm) {
  return comm->comm_type == flagcxCommunicatorHomo;
}

bool useHostComm() {
  const char *useHostComm = flagcxGetEnv("FLAGCX_USE_HOST_COMM");
  if (useHostComm) {
    return std::stoi(useHostComm) == 1;
  }
  return false;
}

bool useHeteroComm() {
  const char *useHeteroComm = flagcxGetEnv("FLAGCX_USE_HETERO_COMM");
  if (useHeteroComm) {
    return std::stoi(useHeteroComm) == 1;
  }
  return false;
}

flagcxResult_t flagcxHandleInit(flagcxHandlerGroup_t *handler) {
  (*handler) = NULL;
  flagcxCalloc(handler, 1);
  flagcxCalloc(&(*handler)->uniqueId, 1);
  flagcxCalloc(&(*handler)->comm, 1);
  flagcxCalloc(&(*handler)->devHandle, 1);
  *(*handler)->devHandle = globalDeviceHandle;
  return flagcxSuccess;
}

flagcxResult_t flagcxHandleFree(flagcxHandlerGroup_t handler) {
  if (handler != NULL) {
    free(handler->uniqueId);
    free(handler->comm);
    free(handler->devHandle);
    handler->uniqueId = NULL;
    handler->comm = NULL;
    handler->devHandle = NULL;
    free(handler);
    handler = NULL;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxMemAlloc(void **ptr, size_t size, flagcxComm_t comm) {
  if (*ptr != NULL || size == 0) {
    WARN("Invalid pointer(!=NULL) or size(0) for allocation.");
    return flagcxSuccess;
  }
  if (comm != NULL && useHomoComm(comm)) {
    FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->memAlloc(ptr, size));
    return flagcxSuccess;
  }
  FLAGCXCHECK(deviceAdaptor->gdrMemAlloc(ptr, size, NULL));
  if (*ptr != NULL) {
    INFO(FLAGCX_REG, "User buffer memory allocated with [%p, %ld]", *ptr, size);
  } else {
    WARN("User buffer allocation failed");
    return flagcxUnhandledDeviceError;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxMemFree(void *ptr, flagcxComm_t comm) {
  if (ptr == NULL) {
    WARN("Invalid pointer(=NULL)for de-allocation.");
    return flagcxSuccess;
  }
  if (comm != NULL && useHomoComm(comm)) {
    FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->memFree(ptr));
    return flagcxSuccess;
  }
  FLAGCXCHECK(deviceAdaptor->gdrMemFree(ptr, NULL));
  INFO(FLAGCX_REG, "User buffer memory deallocated");
  return flagcxSuccess;
}

flagcxResult_t flagcxCommRegister(const flagcxComm_t comm, void *buff,
                                  size_t size, void **handle) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (buff == NULL || size == 0) {
    WARN("Invalid buffer or size for buffer registration.");
    return flagcxInvalidArgument;
  }
  if (useHomoComm(comm)) {
    cclAdaptors[flagcxCCLAdaptorDevice]->commRegister(comm->homo_comm, buff,
                                                      size, handle);
  } else {
    globalRegPool.registerBuffer((void *)comm->hetero_comm, buff, size);
    *handle = reinterpret_cast<void *>(
        globalRegPool.getItem((void *)comm->hetero_comm, buff));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxCommDeregister(const flagcxComm_t comm, void *handle) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHomoComm(comm)) {
    cclAdaptors[flagcxCCLAdaptorDevice]->commDeregister(comm->homo_comm,
                                                        handle);
  } else {
    globalRegPool.deregisterBuffer((void *)comm->hetero_comm, handle);
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxIsHomoComm(flagcxComm_t comm, int *isHomo) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHomoComm(comm)) {
    *isHomo = 1;
  } else {
    *isHomo = 0;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxGetVersion(int *version) {
  // TODO: implement a method to retrieve global verison
  return flagcxHeteroGetVersion(version);
}

flagcxResult_t flagcxGetUniqueId(flagcxUniqueId_t *uniqueId) {
  (*uniqueId) = NULL;
  flagcxCalloc(uniqueId, 1);

  // Init bootstrap net
  FLAGCXCHECK(bootstrapNetInit());

  // Init uniqueId using bootstrap
  struct flagcxBootstrapHandle handle;
  FLAGCXCHECK(bootstrapGetUniqueId(&handle));
  // flagcxUniqueId and bootstrapHandle don't have the same size and alignment
  // reset to 0 to avoid undefined data
  memset((void *)*uniqueId, 0, sizeof(**uniqueId));
  // copy to avoid alignment mismatch
  memcpy((void *)*uniqueId, &handle, sizeof(handle));
  return flagcxSuccess;
}

const char *flagcxGetErrorString(flagcxResult_t result) {
  // TODO: implement a method to retrieve error string
  return "Not implemented.";
}

const char *flagcxGetLastError(flagcxComm_t comm) {
  // TODO: implement a method to retrieve last error string
  if (comm == NULL) {
    return "Undefined: flagcxComm is not fully initialized.";
  }
  if (useHomoComm(comm)) {
    return cclAdaptors[flagcxCCLAdaptorDevice]->getLastError(comm->homo_comm);
  }
  return "Not implemented.";
}

flagcxResult_t flagcxCommInitRank(flagcxComm_t *comm, int nranks,
                                  flagcxUniqueId_t commId, int rank) {
  if (nranks < 1 || rank < 0 || rank >= nranks) {
    WARN("Invalid rank requested : %d/%d", rank, nranks);
    return flagcxInvalidArgument;
  }

  (*comm) = NULL;
  flagcxCalloc(comm, 1);
  (*comm)->rank = rank;
  (*comm)->nranks = nranks;
  (*comm)->nclusters = -1;
  (*comm)->homo_rank = -1;
  (*comm)->homo_root_rank = -1;
  (*comm)->homo_ranks = -1;
  (*comm)->has_single_rank_homo_comm = -1;
  (*comm)->magic = 0;
  (*comm)->abortFlag = 0;
  (*comm)->bootstrap = NULL;
  (*comm)->host_comm = NULL;
  (*comm)->homo_comm = NULL;
  (*comm)->hetero_comm = NULL;
  (*comm)->cluster_ids = NULL;
  (*comm)->cluster_sizes = NULL;
  (*comm)->cluster_inter_ranks = NULL;
  (*comm)->globalrank2homorank = NULL;
  (*comm)->comm_type = flagcxCommunicatorUnknown;
  (*comm)->homoInterRootRank = -1;
  (*comm)->homoInterMyRank = -1;
  (*comm)->homoInterRanks = -1;
  (*comm)->homoInterComm = NULL;

  struct bootstrapState *state = NULL;
  FLAGCXCHECK(flagcxCalloc(&state, 1));
  state->rank = rank;
  state->nranks = nranks;
  state->abortFlag = (*comm)->abortFlag;
  (*comm)->bootstrap = state;
  state->magic = ((struct flagcxBootstrapHandle *)commId)->magic;
  (*comm)->magic = ((struct flagcxBootstrapHandle *)commId)->magic;

  // Init bootstrap net
  FLAGCXCHECK(bootstrapNetInit());

  // Init bootstrap state
  FLAGCXCHECK(bootstrapInit((struct flagcxBootstrapHandle *)commId, state));

  // Ready to detect heterogeneous/homogeneous communicator
  // Use bootstrap allgather to exchange Device info
  flagcxVendor *vendorData =
      NULL; // temp data used for device vendor gather operation.

  // Get current gpu vendor
  flagcxVendor vendor;
  deviceAdaptor->getVendor(vendor.internal);
  FLAGCXCHECK(flagcxCalloc(&vendorData, nranks));
  memcpy(vendorData + rank, &vendor, sizeof(flagcxVendor));
  FLAGCXCHECK(
      bootstrapAllGather(state, (void *)vendorData, sizeof(flagcxVendor)));
  FLAGCXCHECK(bootstrapBarrier(state, rank, nranks, 0));

  // Init cluster info
  int *globalRankToHomoRankData;
  int *clusterIdData;
  int *clusterInterRankData;
  FLAGCXCHECK(flagcxCalloc(&globalRankToHomoRankData, nranks));
  FLAGCXCHECK(flagcxCalloc(&clusterIdData, nranks));
  FLAGCXCHECK(flagcxCalloc(&clusterInterRankData, nranks));
  FLAGCXCHECK(flagcxCollectClusterInfos(
      vendorData, &(*comm)->comm_type, globalRankToHomoRankData + rank,
      &(*comm)->homo_root_rank, &(*comm)->homo_ranks, clusterIdData + rank,
      clusterInterRankData + rank, &(*comm)->nclusters, rank, nranks));
  FLAGCXCHECK(
      bootstrapAllGather(state, (void *)globalRankToHomoRankData, sizeof(int)));
  FLAGCXCHECK(bootstrapAllGather(state, (void *)clusterIdData, sizeof(int)));
  FLAGCXCHECK(
      bootstrapAllGather(state, (void *)clusterInterRankData, sizeof(int)));
  FLAGCXCHECK(bootstrapBarrier(state, rank, nranks, 0));
  (*comm)->homo_rank = globalRankToHomoRankData[rank];
  (*comm)->cluster_ids = clusterIdData;
  (*comm)->globalrank2homorank = globalRankToHomoRankData;

  // fill clusterVendorMap
  FLAGCXCHECK(flagcxFillClusterVendorInfo(vendorData, (*comm), clusterIdData,
                                          nranks, (*comm)->nclusters));

  int *clusterSizes;
  int *clusterInterRanks;
  FLAGCXCHECK(flagcxCalloc(&clusterSizes, (*comm)->nclusters));
  FLAGCXCHECK(flagcxCalloc(&clusterInterRanks, (*comm)->nclusters));
  for (int i = 0; i < (*comm)->nclusters; ++i) {
    clusterInterRanks[i] = -1;
  }

  int cid = 0;
  int sum = 0;
  for (int i = 0; i < nranks; ++i) {
    if (clusterIdData[i] == cid + 1) {
      clusterSizes[cid] = i - sum;
      cid += 1;
      sum = i;
    }
  }
  clusterSizes[cid] = nranks - sum;
  (*comm)->cluster_sizes = clusterSizes;

  for (int i = 0; i < nranks; ++i) {
    if (clusterInterRankData[i] != -1) {
      clusterInterRanks[clusterIdData[i]] = clusterInterRankData[i];
    }
  }
  (*comm)->cluster_inter_ranks = clusterInterRanks;

  int start = 0;
  if (clusterIdData[rank] >= 1) {
    for (int i = 0; i < clusterIdData[rank]; ++i) {
      start += clusterSizes[i];
    }
  }
  (*comm)->homo_inter_rank = clusterInterRanks[clusterIdData[rank]] - start;

  // Update comm has_single_rank_homo_comm
  for (int i = 0; i < (*comm)->nclusters; ++i) {
    if ((*comm)->cluster_sizes[i] == 1) {
      (*comm)->has_single_rank_homo_comm = 1;
    }
  }
  if ((*comm)->has_single_rank_homo_comm == -1) {
    (*comm)->has_single_rank_homo_comm = 0;
  }
  if ((*comm)->has_single_rank_homo_comm == 1 && useHomoComm(*comm)) {
    // no need to record it for homo comm
    (*comm)->has_single_rank_homo_comm = 0;
  }

  flagcxUniqueId *uniqueIdData;
  FLAGCXCHECK(flagcxCalloc(&uniqueIdData, nranks));

  // Tuner init
  bool useTuner = false;
  const char *useTunerEnv = flagcxGetEnv("FLAGCX_USE_TUNER");
  if (useTunerEnv) {
    useTuner = (std::stoi(useTunerEnv) == 1) ? true : false;
  }
  INFO(FLAGCX_INIT, "Flagcx USE_TUNER flag set to %d", useTuner);
  if (useTuner) {
    (*comm)->tuner = &internalTuner;
    (*comm)->tuner->bootstrap = state;
    (*comm)->tuner->rank = rank;
    (*comm)->tuner->nranks = nranks;
    (*comm)->commId = commId;
    (*comm)->uniqueIdData = uniqueIdData;
    (*comm)->tunerInnerComm = NULL;
    FLAGCXCHECK((*comm)->tuner->init((*comm)->nranks, 0, flagcxDebugLog,
                                     &((*comm)->tunerContext)));
    uint32_t nConfigs = 0;
    FLAGCXCHECK(
        (*comm)->tuner->getCandidateNumber((*comm)->tunerContext, &nConfigs));
    if (nConfigs < 1) {
      WARN("Tuner returned 0 candidates, at least 1 is required.");
      return flagcxInternalError;
    }
    (*comm)->homoCommMap.clear();
    (*comm)->homoBestCommMap.clear();
  } else {
    (*comm)->tuner = NULL;
    FLAGCXCHECK(flagcxHomoCommInit(commId, uniqueIdData, state, *comm,
                                   &((*comm)->homo_comm)));
  }

  if (!useHomoComm(*comm) || useHeteroComm()) {
    // Reset commId and hetero root rank calls flagcxHeteroGetUniqueId
    memset((void *)commId, 0, sizeof(flagcxUniqueId));
    memset((void *)uniqueIdData, 0, nranks * sizeof(flagcxUniqueId));
    if (rank == 0) {
      flagcxHeteroGetUniqueId(commId);
      memcpy((void *)&uniqueIdData[0], (void *)commId, sizeof(flagcxUniqueId));
    }
    FLAGCXCHECK(bootstrapAllGather(state, (void *)uniqueIdData,
                                   sizeof(flagcxUniqueId)));
    FLAGCXCHECK(bootstrapBarrier(state, rank, nranks, 0));

    memcpy((void *)commId, (void *)&uniqueIdData[0], sizeof(flagcxUniqueId));
    // call flagcxHeteroCommInitRank
    FLAGCXCHECK(
        flagcxHeteroCommInitRank(&(*comm)->hetero_comm, nranks, *commId, rank));

    // Init host cclAdaptor
    if (useHostComm() || (*comm)->has_single_rank_homo_comm) {
      FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->commInitRank(
          &(*comm)->host_comm, nranks, commId, rank, state));
    }
  }

  if (!useHomoComm(*comm) || useHeteroComm()) {
    // Experimental for multi-nic support
    // Collect nic distance to ranks
    (*comm)->clusterInterRankList.resize((*comm)->nclusters);
    struct flagcxNicDistance *nicDistanceData;
    FLAGCXCHECK(flagcxCalloc(&nicDistanceData, nranks));
    const char *enableTopoDetect = flagcxGetEnv("FLAGCX_ENABLE_TOPO_DETECT");
    if (enableTopoDetect && (strcmp(enableTopoDetect, "TRUE") == 0 ||
                             strcmp(enableTopoDetect, "True") ==
                                 0)) { // safety check nic distance is only
                                       // available after topo detection
      FLAGCXCHECK(flagcxGetNicDistance((*comm)->hetero_comm->topoServer, rank,
                                       nicDistanceData + rank));
    } else {
      nicDistanceData[rank].distance = rank % 2 + 1;
      nicDistanceData[rank].netGuid = rank; // give a dummy value
    }
    FLAGCXCHECK(bootstrapAllGather(state, (void *)nicDistanceData,
                                   sizeof(flagcxNicDistance)));
    FLAGCXCHECK(bootstrapBarrier(state, rank, nranks, 0));
    for (int i = 0; i < (*comm)->nclusters; ++i) {
      int minDistance = INT_MAX;
      std::unordered_map<int, std::vector<int>> nicDistanceToRanks;
      std::unordered_map<int, std::unordered_set<uint64_t>> nicDistanceToNic;
      for (int j = 0; j < nranks; ++j) {
        if (clusterIdData[j] != i) {
          continue;
        }
        int val = nicDistanceData[j].distance;
        uint64_t netGuid = nicDistanceData[j].netGuid;
        if (nicDistanceToNic[val].find(netGuid) ==
            nicDistanceToNic[val].end()) {
          nicDistanceToRanks[val].push_back(j);
          nicDistanceToNic[val].insert(netGuid);
        }
        minDistance = std::min(minDistance, val);
      }
      (*comm)->clusterInterRankList[i] =
          std::move(nicDistanceToRanks[minDistance]);
    }
    // Set homoInterMyRank, homoInterRootRank and homoInterRanks
    auto &myClusterInterRanks =
        (*comm)->clusterInterRankList[clusterIdData[rank]];
    for (size_t i = 0; i < myClusterInterRanks.size(); ++i) {
      if (rank == myClusterInterRanks[i]) {
        (*comm)->homoInterMyRank = i;
      }
    }
    if ((*comm)->homoInterMyRank != -1) {
      (*comm)->homoInterRootRank = myClusterInterRanks[0];
      (*comm)->homoInterRanks = myClusterInterRanks.size();
    }

    INFO(
        FLAGCX_INIT,
        "rank = %d, nranks = %d, nclusters = %d, cluster_id = %d, cluster_size "
        "= %d, "
        "cluster_inter_rank = %d, homo_rank = %d, homo_root_rank = %d, "
        "homo_inter_rank = %d, homo_ranks = %d, "
        "has_single_rank_homo_comm = %d, "
        "homoInterRootRank = %d, homoInterMyRank = %d, homoInterRanks = %d",
        rank, nranks, (*comm)->nclusters, (*comm)->cluster_ids[rank],
        (*comm)->cluster_sizes[(*comm)->cluster_ids[rank]],
        (*comm)->cluster_inter_ranks[(*comm)->cluster_ids[rank]],
        (*comm)->homo_rank, (*comm)->homo_root_rank, (*comm)->homo_inter_rank,
        (*comm)->homo_ranks, (*comm)->has_single_rank_homo_comm,
        (*comm)->homoInterRootRank, (*comm)->homoInterMyRank,
        (*comm)->homoInterRanks);

    // Experimental for multi-nic support
    // Reset commId and homo inter root rank calls underlying GetUniqueId
    // function for initialization of homo inter communicator
    memset((void *)commId, 0, sizeof(flagcxUniqueId));
    memset((void *)uniqueIdData, 0, nranks * sizeof(flagcxUniqueId));
    // Let homoInterRootRank call underlying GetUniqueId function
    // for initialization of homo inter communicator
    if (rank == (*comm)->homoInterRootRank) {
      cclAdaptors[flagcxCCLAdaptorDevice]->getUniqueId(&commId);
      memcpy((void *)&uniqueIdData[rank], (void *)commId,
             sizeof(flagcxUniqueId));
    }
    // Collect uniqueIdData globally
    FLAGCXCHECK(bootstrapAllGather(state, (void *)uniqueIdData,
                                   sizeof(flagcxUniqueId)));
    FLAGCXCHECK(bootstrapBarrier(state, rank, nranks, 0));
    // Call cclAdaptor->commInitRank
    if ((*comm)->homoInterRootRank != -1) {
      memcpy((void *)commId, (void *)&uniqueIdData[(*comm)->homoInterRootRank],
             sizeof(flagcxUniqueId));
      FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->commInitRank(
          &(*comm)->homoInterComm, (*comm)->homoInterRanks, commId,
          (*comm)->homoInterMyRank, NULL));
    }
    free(nicDistanceData);
    const char *deviceFuncPathEnv = flagcxGetEnv("FLAGCX_DEVICE_FUNC_PATH");
    if (deviceFuncPathEnv) {
      FLAGCXCHECK(loadKernelSymbol(deviceFuncPathEnv, "deviceAsyncLoad",
                                   &deviceAsyncLoad));
      FLAGCXCHECK(loadKernelSymbol(deviceFuncPathEnv, "deviceAsyncStore",
                                   &deviceAsyncStore));
      if (deviceAsyncLoad == NULL || deviceAsyncStore == NULL) {
        printf("Failed to load async kernel from %s\n", deviceFuncPathEnv);
        exit(1);
      }
    }
  }

  free(clusterInterRankData);
  free(vendorData);
  if (!useTuner) {
    free(uniqueIdData);
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxCommFinalize(flagcxComm_t comm) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  FLAGCXCHECK(
      cclAdaptors[flagcxCCLAdaptorDevice]->commFinalize(comm->homo_comm));
  if (!useHomoComm(comm)) {
    // TODO: to be implemented
    return flagcxNotSupported;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxCommDestroy(flagcxComm_t comm) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));

  // Destroy cluster info
  free(comm->cluster_ids);
  free(comm->cluster_sizes);
  free(comm->globalrank2homorank);

  // Destroy bootstrap state and net
  bootstrapClose(comm->bootstrap);

  if (!useHomoComm(comm)) {
    // Destroy hetero comm
    FLAGCXCHECK(flagcxHeteroCommDestroy(comm->hetero_comm));
    // Destroy host comm
    if (useHostComm()) {
      FLAGCXCHECK(
          cclAdaptors[flagcxCCLAdaptorHost]->commDestroy(comm->host_comm));
    }
  }
  // Destroy homo comms
  if (comm->tuner) {
    for (const auto &item : comm->homoCommMap) {
      if (item.second != nullptr) {
        FLAGCXCHECK(
            cclAdaptors[flagcxCCLAdaptorDevice]->commDestroy(item.second));
      }
    }
  } else {
    cclAdaptors[flagcxCCLAdaptorDevice]->commDestroy(comm->homo_comm);
  }

  // Destroy tuner
  if (comm->tuner) {
    comm->tuner->destroy(comm->tunerContext);
    // Free uniqueIdData
    free(comm->uniqueIdData);
  }

  return flagcxSuccess;
}

flagcxResult_t flagcxCommAbort(flagcxComm_t comm) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->commAbort(comm->homo_comm));
  if (!useHomoComm(comm)) {
    // TODO: to be implemented.
    return flagcxNotSupported;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxCommResume(flagcxComm_t comm) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->commResume(comm->homo_comm));
  if (!useHomoComm(comm)) {
    // TODO: to be implemented.
    return flagcxNotSupported;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxCommSuspend(flagcxComm_t comm) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  FLAGCXCHECK(
      cclAdaptors[flagcxCCLAdaptorDevice]->commSuspend(comm->homo_comm));
  if (!useHomoComm(comm)) {
    // TODO: to be implemented.
    return flagcxNotSupported;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxCommCount(const flagcxComm_t comm, int *count) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHomoComm(comm)) {
    return cclAdaptors[flagcxCCLAdaptorDevice]->commCount(comm->homo_comm,
                                                          count);
  }
  return flagcxHeteroCommCount(comm->hetero_comm, count);
}

flagcxResult_t flagcxCommGetDeviceNumber(const flagcxComm_t comm, int *device) {
  return cclAdaptors[flagcxCCLAdaptorDevice]->commGetDeviceNumber(
      comm->homo_comm, device);
}

flagcxResult_t flagcxCommUserRank(const flagcxComm_t comm, int *rank) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHomoComm(comm)) {
    return cclAdaptors[flagcxCCLAdaptorDevice]->commUserRank(comm->homo_comm,
                                                             rank);
  }
  return flagcxHeteroCommUserRank(comm->hetero_comm, rank);
}

flagcxResult_t flagcxCommFifoBuffer(const flagcxComm_t comm, void **buffer) {
  if (comm->hetero_comm->fifoBuffer == NULL) {
    return flagcxInvalidUsage;
  }
  *buffer = comm->hetero_comm->fifoBuffer;
  return flagcxSuccess;
}

flagcxResult_t flagcxCommGetAsyncError(flagcxComm_t comm,
                                       flagcxResult_t *asyncError) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHomoComm(comm)) {
    return cclAdaptors[flagcxCCLAdaptorDevice]->commGetAsyncError(
        comm->homo_comm, asyncError);
  }
  // TODO: to be implemented.
  return flagcxNotSupported;
}

flagcxResult_t flagcxBarrier(flagcxComm_t comm, flagcxStream_t stream) {
  void *barrierBuff;
  deviceAdaptor->deviceMalloc(&barrierBuff, comm->nranks, flagcxMemDevice,
                              stream);
  deviceAdaptor->deviceMemset(barrierBuff, 0, comm->nranks, flagcxMemDevice,
                              stream);
  flagcxAllReduce(barrierBuff, barrierBuff, comm->nranks, flagcxChar, flagcxMax,
                  comm, stream);
  deviceAdaptor->deviceFree(barrierBuff, flagcxMemDevice, stream);
  deviceAdaptor->streamSynchronize(stream);
  return flagcxSuccess;
}

flagcxResult_t flagcxReduce(const void *sendbuff, void *recvbuff, size_t count,
                            flagcxDataType_t datatype, flagcxRedOp_t op,
                            int root, flagcxComm_t comm,
                            flagcxStream_t stream) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHomoComm(comm)) {
    FLAGCXCHECK(flagcxRunners[flagcxHomoRunner]->reduce(
        sendbuff, recvbuff, count, datatype, op, root, comm, stream));
  } else if (useHostComm() || comm->has_single_rank_homo_comm) {
    // c2c validation
    if (comm->has_single_rank_homo_comm) {
      WARN("Host comm is required to perform C2C reduce op when "
           "comm->has_single_rank_homo_comm is True");
    }
    FLAGCXCHECK(flagcxRunners[flagcxHostRunner]->reduce(
        sendbuff, recvbuff, count, datatype, op, root, comm, stream));
  } else {
    FLAGCXCHECK(flagcxRunners[flagcxHybridRunner]->reduce(
        sendbuff, recvbuff, count, datatype, op, root, comm, stream));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxGather(const void *sendbuff, void *recvbuff, size_t count,
                            flagcxDataType_t datatype, int root,
                            flagcxComm_t comm, flagcxStream_t stream) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHeteroComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxUniRunner]->gather(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  } else if (useHomoComm(comm)) {
    FLAGCXCHECK(flagcxRunners[flagcxHomoRunner]->gather(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  } else if (useHostComm() || comm->has_single_rank_homo_comm) {
    // c2c validation
    if (comm->has_single_rank_homo_comm) {
      WARN("Host comm is required to perform C2C gather op when "
           "comm->has_single_rank_homo_comm is True");
    }
    FLAGCXCHECK(flagcxRunners[flagcxHostRunner]->gather(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  } else {
    FLAGCXCHECK(flagcxRunners[flagcxHybridRunner]->gather(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxScatter(const void *sendbuff, void *recvbuff, size_t count,
                             flagcxDataType_t datatype, int root,
                             flagcxComm_t comm, flagcxStream_t stream) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHeteroComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxUniRunner]->scatter(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  } else if (useHomoComm(comm)) {
    FLAGCXCHECK(flagcxRunners[flagcxHomoRunner]->scatter(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  } else if (useHostComm() || comm->has_single_rank_homo_comm) {
    // c2c validation
    if (comm->has_single_rank_homo_comm) {
      WARN("Host comm is required to perform C2C scatter op when "
           "comm->has_single_rank_homo_comm is True");
    }
    FLAGCXCHECK(flagcxRunners[flagcxHostRunner]->scatter(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  } else {
    FLAGCXCHECK(flagcxRunners[flagcxHybridRunner]->scatter(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxBroadcast(const void *sendbuff, void *recvbuff,
                               size_t count, flagcxDataType_t datatype,
                               int root, flagcxComm_t comm,
                               flagcxStream_t stream) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHeteroComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxUniRunner]->broadcast(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  } else if (useHomoComm(comm)) {
    FLAGCXCHECK(flagcxRunners[flagcxHomoRunner]->broadcast(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  } else if (useHostComm() || comm->has_single_rank_homo_comm) {
    // c2c validation
    if (comm->has_single_rank_homo_comm) {
      WARN("Host comm is required to perform C2C broadcast op when "
           "comm->has_single_rank_homo_comm is True");
    }
    FLAGCXCHECK(flagcxRunners[flagcxHostRunner]->broadcast(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  } else {
    FLAGCXCHECK(flagcxRunners[flagcxHybridRunner]->broadcast(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxAllReduce(const void *sendbuff, void *recvbuff,
                               size_t count, flagcxDataType_t datatype,
                               flagcxRedOp_t op, flagcxComm_t comm,
                               flagcxStream_t stream) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHomoComm(comm)) {
    FLAGCXCHECK(flagcxRunners[flagcxHomoRunner]->allReduce(
        sendbuff, recvbuff, count, datatype, op, comm, stream));
  } else if (useHostComm() || comm->has_single_rank_homo_comm) {
    // c2c validation
    if (comm->has_single_rank_homo_comm) {
      WARN("Host comm is required to perform C2C allreduce op when "
           "comm->has_single_rank_homo_comm is True");
    }
    FLAGCXCHECK(flagcxRunners[flagcxHostRunner]->allReduce(
        sendbuff, recvbuff, count, datatype, op, comm, stream));
  } else {
    FLAGCXCHECK(flagcxRunners[flagcxHybridRunner]->allReduce(
        sendbuff, recvbuff, count, datatype, op, comm, stream));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxReduceScatter(const void *sendbuff, void *recvbuff,
                                   size_t recvcount, flagcxDataType_t datatype,
                                   flagcxRedOp_t op, flagcxComm_t comm,
                                   flagcxStream_t stream) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHomoComm(comm)) {
    FLAGCXCHECK(flagcxRunners[flagcxHomoRunner]->reduceScatter(
        sendbuff, recvbuff, recvcount, datatype, op, comm, stream));
  } else if (useHostComm() || comm->has_single_rank_homo_comm) {
    // c2c validation
    if (comm->has_single_rank_homo_comm) {
      WARN("Host comm is required to perform C2C reducescatter op when "
           "comm->has_single_rank_homo_comm is True");
    }
    FLAGCXCHECK(flagcxRunners[flagcxHostRunner]->reduceScatter(
        sendbuff, recvbuff, recvcount, datatype, op, comm, stream));
  } else {
    FLAGCXCHECK(flagcxRunners[flagcxHybridRunner]->reduceScatter(
        sendbuff, recvbuff, recvcount, datatype, op, comm, stream));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxAllGather(const void *sendbuff, void *recvbuff,
                               size_t sendcount, flagcxDataType_t datatype,
                               flagcxComm_t comm, flagcxStream_t stream) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHeteroComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxUniRunner]->allGather(
        sendbuff, recvbuff, sendcount, datatype, comm, stream));
  } else if (useHomoComm(comm)) {
    FLAGCXCHECK(flagcxRunners[flagcxHomoRunner]->allGather(
        sendbuff, recvbuff, sendcount, datatype, comm, stream));
  } else if (useHostComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxHostRunner]->allGather(
        sendbuff, recvbuff, sendcount, datatype, comm, stream));
  } else {
    FLAGCXCHECK(flagcxRunners[flagcxHybridRunner]->allGather(
        sendbuff, recvbuff, sendcount, datatype, comm, stream));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxAlltoAll(const void *sendbuff, void *recvbuff,
                              size_t count, flagcxDataType_t datatype,
                              flagcxComm_t comm, flagcxStream_t stream) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHeteroComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxUniRunner]->alltoAll(
        sendbuff, recvbuff, count, datatype, comm, stream));
  } else if (useHomoComm(comm)) {
    FLAGCXCHECK(flagcxRunners[flagcxHomoRunner]->alltoAll(
        sendbuff, recvbuff, count, datatype, comm, stream));
  } else if (useHostComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxHostRunner]->alltoAll(
        sendbuff, recvbuff, count, datatype, comm, stream));
  } else {
    FLAGCXCHECK(flagcxRunners[flagcxHybridRunner]->alltoAll(
        sendbuff, recvbuff, count, datatype, comm, stream));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxAlltoAllv(const void *sendbuff, size_t *sendcounts,
                               size_t *sdispls, void *recvbuff,
                               size_t *recvcounts, size_t *rdispls,
                               flagcxDataType_t datatype, flagcxComm_t comm,
                               flagcxStream_t stream) {

  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHeteroComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxUniRunner]->alltoAllv(
        sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls, datatype,
        comm, stream));
  } else if (useHomoComm(comm)) {
    FLAGCXCHECK(flagcxRunners[flagcxHomoRunner]->alltoAllv(
        sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls, datatype,
        comm, stream));
  } else if (useHostComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxHostRunner]->alltoAllv(
        sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls, datatype,
        comm, stream));
  } else {
    FLAGCXCHECK(flagcxRunners[flagcxHybridRunner]->alltoAllv(
        sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls, datatype,
        comm, stream));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxSend(const void *sendbuff, size_t count,
                          flagcxDataType_t datatype, int peer,
                          flagcxComm_t comm, flagcxStream_t stream) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHeteroComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxUniRunner]->send(sendbuff, count, datatype,
                                                     peer, comm, stream));
  } else if (useHomoComm(comm)) {
    FLAGCXCHECK(flagcxRunners[flagcxHomoRunner]->send(sendbuff, count, datatype,
                                                      peer, comm, stream));
  } else if (useHostComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxHostRunner]->send(sendbuff, count, datatype,
                                                      peer, comm, stream));
  } else {
    FLAGCXCHECK(flagcxRunners[flagcxHybridRunner]->send(
        sendbuff, count, datatype, peer, comm, stream));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxRecv(void *recvbuff, size_t count,
                          flagcxDataType_t datatype, int peer,
                          flagcxComm_t comm, flagcxStream_t stream) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHeteroComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxUniRunner]->recv(recvbuff, count, datatype,
                                                     peer, comm, stream));
  } else if (useHomoComm(comm)) {
    FLAGCXCHECK(flagcxRunners[flagcxHomoRunner]->recv(recvbuff, count, datatype,
                                                      peer, comm, stream));
  } else if (useHostComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxHostRunner]->recv(recvbuff, count, datatype,
                                                      peer, comm, stream));
  } else {
    FLAGCXCHECK(flagcxRunners[flagcxHybridRunner]->recv(
        recvbuff, count, datatype, peer, comm, stream));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxGroupStart(flagcxComm_t comm) {
  if (useHeteroComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxUniRunner]->groupStart());
  } else if (useHomoComm(comm)) {
    FLAGCXCHECK(flagcxRunners[flagcxHomoRunner]->groupStart());
  } else if (useHostComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxHostRunner]->groupStart());
  } else {
    FLAGCXCHECK(flagcxRunners[flagcxHybridRunner]->groupStart());
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxGroupEnd(flagcxComm_t comm) {
  if (useHeteroComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxUniRunner]->groupStart());
  } else if (useHomoComm(comm)) {
    FLAGCXCHECK(flagcxRunners[flagcxHomoRunner]->groupEnd());
  } else if (useHostComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxHostRunner]->groupEnd());
  } else {
    FLAGCXCHECK(flagcxRunners[flagcxHybridRunner]->groupEnd());
  }
  return flagcxSuccess;
}
