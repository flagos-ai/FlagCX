#ifndef FLAGCX_GLOBAL_COMM_H_
#define FLAGCX_GLOBAL_COMM_H_

#include "bootstrap.h"
#include "comm_lifecycle.h"
#include "dev_comm_state.h"
#include "flagcx.h"
#include "flagcx_tuner.h"
#include "utils.h"
#include "work.h"

#include <map>
#include <vector>

/* Opaque handle to flagcxInnerComm */
typedef struct flagcxInnerComm *flagcxInnerComm_t;

// IPC peer pointer table entry — owned by a local registration or a DevMem
// reference. Communicator destroy closes any leftover entry.
#define FLAGCX_MAX_IPC_ENTRIES 16

struct flagcxIpcTableEntry {
  void **hostPeerPtrs;     // host array: peer user-buffer pointers
  void **hostPeerBasePtrs; // raw mappings returned by ipcMemHandleOpen
  void **devPeerPtrs;      // device array: peer buffer ptrs (for cudaFree)
  int nPeers;              // number of local peers
  void *basePtr;           // own user-buffer pointer
  size_t size;
  size_t refCount;
  bool inUse;     // true only for a fully prepared mapping; partial resources
                  // reserve the slot
  bool ownerHeld; // local Window, DevComm, or signal registration owns the slot
};

/* Opaque handle to flagcxHeteroComm */
typedef struct flagcxHeteroComm *flagcxHeteroComm_t;

typedef enum {
  flagcxCommunicatorUnknown = 0,
  flagcxCommunicatorHomo = 1,  // Homogeneous Communicator
  flagcxCommunicatorHybrid = 2 // Hybrid Communicator
} flagcxCommunicatorType_t;

// Whether the user selected the hetero execution path, even on homo hardware.
bool useHeteroComm();

typedef enum flagcxInternalBufferKind {
  flagcxInternalBufferDevice,
  flagcxInternalBufferHost,
  flagcxInternalBufferGdr,
  flagcxInternalBufferCcl
} flagcxInternalBufferKind_t;

// One record per allocation, including buffers whose creation rolled back.
// Device views borrow these pointers; only CommDestroy frees the allocation.
struct flagcxInternalBuffer {
  void *pointer;
  flagcxInternalBufferKind_t kind;
  flagcxInternalBuffer *next;
};

flagcxResult_t flagcxCommAllocInternalBuffer(flagcxComm_t comm, void **pointer,
                                             size_t size,
                                             flagcxInternalBufferKind_t kind);
flagcxResult_t flagcxCommReleaseBuffer(flagcxComm_t comm, void *pointer,
                                       flagcxInternalBufferKind_t kind);
flagcxResult_t flagcxCommFreeInternalBuffers(flagcxComm_t comm);

struct flagcxComm {
  struct flagcxCommFinalizeState finalizeState;
  enum flagcxNativeFinalizeScope nativeFinalizeScope;
  bool allowNativeDestroyFinalize, ipcClosePolicyConfirmed;
  bool initRollbackPending, initRollbackNeedsAgreement, deviceOrdinalValid;
  int deviceOrdinal;
  struct flagcxWorkTracker submittedWork;
  struct flagcxC2cPendingResources *pendingC2cResources;
  bool devicePluginHeld;
  bool cclPluginHeld;
  bool netPluginHeld;
  bool pluginRollbackOnly;
  // 0: local SDK cleanup, 1: only plugin references remain.
  int destroyPhase;
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
  struct bootstrapState *bootstrap;
  bool streamValueSupported;
  bool uniRunnerKernelSupported;
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
  bool isTuningWithFlagscale;       // whether tuning with flagscale
  bool isTunningComm;               // whether tuning the communicator
  bool isUseSingleTunerComm;        // whether tuning with one communicator
  struct C2cSchedulePair {
    int sendCluster;
    int recvCluster;
  } * c2cSchedule; // C2C schedule for pairing send/recv operations

  // IPC peer pointer table — entries are released by their local owners
  struct flagcxIpcTableEntry ipcTableInline[FLAGCX_MAX_IPC_ENTRIES];
  struct flagcxIpcTableEntry *ipcTable = ipcTableInline;
  int ipcTableCapacity = FLAGCX_MAX_IPC_ENTRIES;

  flagcxInternalBuffer *internalBuffers = nullptr;
  // Set by collective creation, including ranks where allocation later fails.
  bool internalBuffersNeedPeerClose = false;

  flagcxWindow_t publicWindows; // includes empty reusable public handles
  flagcxDevMem_t activeDevMems;
  int devCommCount; // active device communicators, also the newest ID
  flagcxDevComm_t activeDevComms;
#if defined(FLAGCX_USE_GDA)
  // Public Window handles, including vendor Windows. GDA activation replays
  // registrations when these Windows precede DevComm creation.
  struct flagcxWindow *devNetWindows;
  flagcxDevComm_t devNetCommHandle;
#endif

  // Custom op state (NULL = not enabled)
  struct flagcxDevCommState *devCommState;

  // A partially-created DevComm whose backend rollback could not release all
  // owned resources. The communicator retains it until cleanup can be retried.
  flagcxDevComm_t pendingDevCommCleanup;
};

// Function helps init single homo cluster.
// return homoComm via homoComm parameter.
flagcxResult_t flagcxHomoCommInit(struct bootstrapState *state,
                                  flagcxComm_t comm,
                                  flagcxInnerComm_t *homoComm /*out*/);

#endif // end include guard
