#ifndef FLAGCX_GLOBAL_COMM_H_
#define FLAGCX_GLOBAL_COMM_H_

#include "bootstrap.h"
#include "flagcx.h"
#include "flagcx_tuner.h"

#include <map>
#include <vector>

/* Opaque handle to flagcxInnerComm */
typedef struct flagcxInnerComm *flagcxInnerComm_t;

// IPC peer pointer table entry — owned by comm, referenced by devMem.
// Cleanup deferred to flagcxCommDestroy.
// to avoid cudaFree implicit device synchronization deadlock.
#define FLAGCX_MAX_IPC_ENTRIES 16

struct flagcxIpcTableEntry {
  void **hostPeerPtrs; // host array: peer buffer ptrs (for ipcMemHandleClose)
  void **devPeerPtrs;  // device array: peer buffer ptrs (for cudaFree)
  int nPeers;          // number of local peers
  void *basePtr;       // own buffer ptr (skip in ipcMemHandleClose loop)
  bool inUse;          // true while a devMem references this entry
};

// Deferred device/host-pinned memory free — collected during cleanup,
// drained in flagcxCommDestroy.
#define FLAGCX_MAX_DEFERRED_FREES 32

struct flagcxDeferredFree {
  void *ptr;
  int memType; // flagcxMemDevice, flagcxMemHost, etc.
};

/* Opaque handle to flagcxHeteroComm */
typedef struct flagcxHeteroComm *flagcxHeteroComm_t;

typedef enum {
  flagcxCommunicatorUnknown = 0,
  flagcxCommunicatorHomo = 1,  // Homogeneous Communicator
  flagcxCommunicatorHybrid = 2 // Hybrid Communicator
} flagcxCommunicatorType_t;

struct flagcxComm {
  // TODO: adjust code format
  int rank;
  int nranks;
  int nclusters;
  int homoRank;
  int homoRootRank;
  int homoRanks;
  int hasSingleRankHomoComm;
  flagcxCommunicatorType_t commType;
  uint64_t magic;
  volatile uint32_t *abortFlag;
  int *clusterSizes;
  int *clusterIds;
  int *globalRank2HomoRank;
  int *clusterInterRanks;
  bootstrapState *bootstrap;
  int localRank;        // intra-node rank index (computed from hostHash)
  int localRanks;       // number of ranks on this node
  int *localRankToRank; // mapping: local index -> global rank
  flagcxInnerComm_t hostComm;
  flagcxInnerComm_t homoComm;
  flagcxHeteroComm_t heteroComm;
  flagcxInnerComm_t homoInterComm;
  int homoInterRootRank;
  int homoInterMyRank;
  int homoInterRanks;
  std::vector<std::vector<int>> clusterInterRankList;
  std::vector<flagcxVendorType> clusterVendorMap;
  struct flagcxTuner *tuner;
  void *tunerContext;
  std::map<struct TunerCollCategory, flagcxInnerComm_t>
      homoCommMap; // key: commTag returned by tuner
  std::map<struct flagcxCommTag, flagcxInnerComm_t> commMap;
  std::map<struct TunerCollCategory, flagcxInnerComm_t>
      homoBestCommMap;              // key: commTag returned by tuner
  flagcxInnerComm_t tunerInnerComm; // innerComm selected by tuner
  flagcxUniqueId_t commId;
  flagcxUniqueId *uniqueIdData;
  bool isTuningWithFlagscale; // whether tuning with flagscale
  bool isTunningComm;         // whether tuning the communicator
  bool isUseSingleTunerComm;  // whether tuning with one communicator
  struct C2cSchedulePair {
    int sendCluster;
    int recvCluster;
  } * c2cSchedule; // C2C schedule for pairing send/recv operations

  // IPC peer pointer table — deferred cleanup
  struct flagcxIpcTableEntry ipcTable[FLAGCX_MAX_IPC_ENTRIES];

  // Deferred device/host-pinned memory free list
  struct flagcxDeferredFree deferredFrees[FLAGCX_MAX_DEFERRED_FREES];
  int deferredFreeCount;
};

#endif // end include guard
