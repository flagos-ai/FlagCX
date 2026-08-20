#ifndef FLAGCX_UNIRUNNER_IMPL_H_
#define FLAGCX_UNIRUNNER_IMPL_H_

#include "device.h"
#include "flagcx.h"
#include "flagcx_kernel.h"
#include "flagcx_kernel_internal.h"
#include "flagcx_net.h"
#include "group.h"
#include "info.h"
#include "ipcsocket.h"
#include "launch_kernel.h"
#include "net.h"
#include "reg_pool.h"
#include "socket.h"
#include "utils.h"
#include <memory>
#include <pthread.h>

// DAG node types
typedef enum {
  uniRunnerDagNodeTypeP2p = 0,
  uniRunnerDagNodeTypeRed = 1,
  uniRunnerDagNodeTypeCpy = 2,
  uniRunnerDagNodeTypeIpc = 3
} uniRunnerDagNodeType;

// Static DAG template algorithm identifiers used by the uniRunner cache.
typedef enum {
  uniRunnerDagAlgoDummy = 0,
  uniRunnerDagAlgoLocRed = 1,
  uniRunnerDagAlgoGroupedAG = 2,
  uniRunnerDagAlgoRingAG = 3,
  uniRunnerDagAlgoRingAR = 4,
  uniRunnerDagAlgoSlicedAR = 5,
  uniRunnerDagAlgoRingRS = 6,
  uniRunnerDagAlgoTreeRed = 7,
  uniRunnerDagAlgoIpcAR = 8
} uniRunnerDagAlgoType;

// Normalized algorithm parameters that affect DAG construction. Runtime-only
// executor settings (for example NTHREADS/NREDBLOCKS/NIPCBLOCKS) deliberately
// do not belong here because they do not change the cached DAG template.
typedef enum {
  uniRunnerDagBufferModeOutOfPlace = 0,
  uniRunnerDagBufferModeInPlace = 1
} uniRunnerDagBufferMode;

struct uniRunnerDagAlgorithmConfig {
  uint64_t numSlices = 0;
  uint64_t numRedSlices = 0;
  uint64_t groupSize = 0;
  uint64_t topologyHash = 0;
  uniRunnerDagBufferMode bufferMode = uniRunnerDagBufferModeOutOfPlace;
};

// Cache key describing a reusable uniRunner DAG template.
struct uniRunnerDagCacheKey {
  // algoHash covers both the algorithm type and its normalized builder
  // configuration.
  uint64_t algoHash;
  flagcxCommOp_t commOp;
  size_t count;
  flagcxDataType_t datatype;
  flagcxRedOp_t redOp;
  int rank;
  int nranks;
  int root;
};

// Single P2P operation data
struct uniRunnerP2pOpData {
  void *addr;                // Buffer address
  size_t count;              // Element count
  int peerRank;              // Peer rank
  flagcxDataType_t datatype; // Data type
  flagcxDevicePrim type;     // Primitive type (send/recv/term/wait)
};

// P2P node data (supports multiple operations in a group)
struct uniRunnerP2pNodeData {
  struct uniRunnerP2pOpData *ops; // Array of P2P operations
  int numOps;                     // Number of operations
};

// Reduce node data (operation-specific fields only)
struct uniRunnerRedNodeData {
  void *input1;
  void *input2;
  void *output;
  size_t count;
  size_t nthreads;
  flagcxDataType_t datatype;
  flagcxRedOp_t redOp;

  // Trigger and state tracking
  int triggerIdx; // Trigger index in FIFO
};

// Copy node data (operation-specific fields only)
struct uniRunnerCpyNodeData {
  void *src;
  void *dst;
  size_t count;
  flagcxDataType_t datatype;
};

struct uniRunnerIpcNodeData {
  size_t srcOffsetBytes;
  size_t dstOffsetBytes;
  size_t bytes;
  flagcxIpcBufferType srcBufferType;
  int peerLocalRank;
  uint32_t readySlot;
  uint32_t parentFlagsOffset;
  int triggerIdx;
};

// Unified DAG node with common DAG structure fields
struct uniRunnerDagNode {
  uniRunnerDagNodeType nodeType; // Discriminator for union

  // Common DAG structure fields (shared by all node types)
  int nodeIdx;                   // Unique index of the node in the DAG
  int numParents;                // Number of parent dependencies
  int *parents;                  // Array of parent node indices
  int pendingParents;            // Remaining parents before host submission
  int numChildren;               // Number of children
  int *children;                 // Array of child node indices
  struct uniRunnerDagNode *next; // Queue linkage

  // Union for type-specific operation data
  union {
    struct uniRunnerP2pNodeData p2p;
    struct uniRunnerRedNodeData red;
    struct uniRunnerCpyNodeData cpy;
    struct uniRunnerIpcNodeData ipc;
  } nodeData;
};

// Stable, host-compiled execution order for a validated DAG. Later executor
// stages derive per-node-type static queues and block assignments from this
// single order, avoiding duplicate persistent arrays.
struct uniRunnerDagExecutionPlan {
  const int *topoOrder = NULL;
  size_t numNodes = 0;
  size_t numHostNodes = 0;
  size_t numRedNodes = 0;
  size_t numIpcNodes = 0;
  size_t maxParallelRedNodes = 0;
  size_t maxRedNodeCount = 0;
  size_t maxIpcNodeBytes = 0;
  // Cached plans borrow immutable storage owned by the process cache.
  bool ownsTopoOrder = false;
  // Set only after static topology and node payload validation succeeds.
  // This lets runUniRunner keep per-invocation validation O(1).
  bool staticValidated = false;
};

// Per-invocation executor counts derived from the DAG and the runtime
// residency budget. RED trigger i is owned by block i % numRedBlocks. IPC
// triggers remain flat and in topological order; all IPC blocks advance them
// in lockstep, with chunk c owned by block c % numIpcBlocks.
struct uniRunnerStaticExecutorSchedule {
  size_t numRedTasks = 0;
  size_t numIpcTasks = 0;
  size_t numRedBlocks = 0;
  size_t numIpcBlocks = 0;
};

// Runtime bounds needed to materialize immutable IPC triggers. All fields are
// invocation-local and deliberately excluded from the DAG cache key.
struct uniRunnerStaticIpcTriggerConfig {
  size_t chunkSize = 0;
  uint64_t epoch = 0;
  size_t dataBytes = 0;
  size_t readySlots = 0;
  size_t parentFlagsCount = 0;
  int localRanks = 0;
};

typedef struct {
  pthread_t thread;
  flagcxFifo_t fifo;
  flagcxFifo_t ipcFifo;
  void *ipcFifoDevicePtr;
  flagcxStream_t commStream;
  flagcxStream_t redStream;
  flagcxStream_t cpyStream;

  // new: DAG and scheduling queues
  struct uniRunnerDagNode *dagNodes; // Array of all DAG nodes
  int numDagNodes;
  int numPendingNodes;
  struct uniRunnerDagExecutionPlan dagPlan;
  flagcxIntruQueue<struct uniRunnerDagNode, &uniRunnerDagNode::next>
      p2pReadyQueue;
  flagcxIntruQueue<struct uniRunnerDagNode, &uniRunnerDagNode::next>
      redReadyQueue;
  flagcxIntruQueue<struct uniRunnerDagNode, &uniRunnerDagNode::next>
      ipcReadyQueue;

  uint64_t uniRunnerNSlices;
  uint64_t uniRunnerMaxThreads;
  uint64_t uniRunnerMaxBlocks;
  uint64_t uniRunnerNThreads;
  uint64_t uniRunnerNRedBlocks;
  uint64_t uniRunnerNIpcBlocks;
  uint64_t uniRunnerIpcChunkSize;
  uint64_t uniRunnerNRedSlices;
  uint64_t uniRunnerRedSliceSize;
  // One launch-wide divisor used by terminal Avg RED nodes. Intermediate
  // Avg nodes are materialized as Sum nodes, so no trigger needs to carry it.
  uint64_t avgDivisor;

  // Stream completion flags backed by a reusable contiguous device pool. The
  // host-side queue stores per-node addresses within that pool.
  void *streamFlagsPool;
  void **streamFlags;
  size_t streamFlagsSize;
  size_t streamFlagsCapacity;

  // IPC/LSA AllReduce runtime resources. User-buffer DevMem views are reused
  // while their base/size bindings remain unchanged. ipcReadySlots counts only
  // logical DAG-ready entries; ipcReadyCapacity includes the static executor's
  // abort/done control prefix and is the number of allocated uint64_t slots.
  flagcxComm_t ipcOwner;
  flagcxDevMem_t ipcInputMem;
  flagcxDevMem_t ipcOutputMem;
  flagcxDevMem_t ipcReadyMem;
  const void *ipcInputBase;
  void *ipcOutputBase;
  size_t ipcDataBytes;
  void *ipcReadyBuffer;
  size_t ipcReadyCapacity;
  size_t ipcReadySlots;
  uint64_t ipcEpoch;
  uint64_t *ipcParentFlagsDevice;
  size_t ipcParentFlagsCount;
} flagcxUniRunnerState;

flagcxResult_t initUniRunnerStateDummy(flagcxUniRunnerState *runnerState);
flagcxResult_t initUniRunnerStateLocRed(flagcxUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        size_t count, flagcxDataType_t datatype,
                                        flagcxRedOp_t op, flagcxComm_t comm);
flagcxResult_t initUniRunnerStateGroupedAG(flagcxUniRunnerState *runnerState,
                                           const void *sendbuff, void *recvbuff,
                                           size_t count,
                                           flagcxDataType_t datatype,
                                           flagcxComm_t comm, int groupSize);
flagcxResult_t initUniRunnerStateRingAG(flagcxUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        size_t count, flagcxDataType_t datatype,
                                        flagcxRedOp_t op, flagcxComm_t comm);
flagcxResult_t initUniRunnerStateRingAR(flagcxUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        size_t count, flagcxDataType_t datatype,
                                        flagcxRedOp_t op, flagcxComm_t comm);
flagcxResult_t initUniRunnerStateSlicedAR(flagcxUniRunnerState *runnerState,
                                          const void *sendbuff, void *recvbuff,
                                          size_t count,
                                          flagcxDataType_t datatype,
                                          flagcxRedOp_t op, flagcxComm_t comm);
flagcxResult_t initUniRunnerStateIpcAR(flagcxUniRunnerState *runnerState,
                                       const void *sendbuff, void *recvbuff,
                                       size_t count,
                                       flagcxDataType_t datatype,
                                       flagcxRedOp_t op, flagcxComm_t comm);
flagcxResult_t initUniRunnerStateRingRS(flagcxUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        void *scratchbuff, size_t count,
                                        flagcxDataType_t datatype,
                                        flagcxRedOp_t op, flagcxComm_t comm);
flagcxResult_t initUniRunnerStateTreeRed(flagcxUniRunnerState *runnerState,
                                         const void *sendbuff, void *recvbuff,
                                         void *scratchbuff, size_t count,
                                         flagcxDataType_t datatype,
                                         flagcxRedOp_t op, int root,
                                         flagcxComm_t comm);
size_t getUniRunnerDagPatternHash(const uniRunnerDagCacheKey &key);
uint64_t getUniRunnerDagAlgorithmHash(
    uniRunnerDagAlgoType algoType,
    const uniRunnerDagAlgorithmConfig &algorithmConfig);
flagcxResult_t initUniRunner(flagcxComm_t comm, flagcxStream_t stream);

// Validate executor launch parameters before allocating per-invocation
// resources. Reduction collectives require at least one RED executor block.
flagcxResult_t validateUniRunnerLaunchConfig(flagcxCommOp_t commOp);

// Validate public reduction arguments before allocating buffers or publishing
// any RED node to the FIFO.
flagcxResult_t validateUniRunnerReduceArgs(size_t count,
                                           flagcxDataType_t datatype,
                                           flagcxRedOp_t op);
flagcxResult_t checkedUniRunnerTypeBytes(size_t count, size_t multiplier,
                                         flagcxDataType_t datatype,
                                         size_t *bytes);
flagcxResult_t checkedUniRunnerDagNodeCount(size_t outerCount,
                                            size_t nodesPerOuter,
                                            size_t extraNodes,
                                            int *nodeCount);

// Compile a structurally validated DAG into the stable HOST -> RED -> IPC
// phase-drain order used by uniRunner when FIFO capacity is not a constraint.
// The compiler is pure with respect to DAG nodes and rejects cycles instead of
// allowing the runtime scheduler to spin indefinitely.
flagcxResult_t compileUniRunnerDagExecutionPlan(
    const uniRunnerDagNode *nodes, size_t numNodes,
    uniRunnerDagExecutionPlan *plan);
void destroyUniRunnerDagExecutionPlan(uniRunnerDagExecutionPlan *plan);
flagcxResult_t resolveUniRunnerStaticExecutorSchedule(
    const uniRunnerDagExecutionPlan *plan, size_t requestedRedBlocks,
    size_t requestedIpcBlocks, size_t maxIpcParallelism,
    size_t maxExecutorBlocks,
    uniRunnerStaticExecutorSchedule *schedule);
flagcxResult_t resolveUniRunnerDagLaunchShape(
    const uniRunnerDagExecutionPlan *plan, size_t ipcChunkSize,
    size_t maxThreads, size_t maxBlocks, size_t *nthreads,
    size_t *requestedRedBlocks, size_t *requestedIpcBlocks);
flagcxResult_t getUniRunnerStaticTaskAssignment(
    size_t taskOrdinal, size_t numTasks, size_t numBlocks, size_t *blockIdx,
    size_t *blockTaskOrdinal);
// plan must be the immutable result of compileUniRunnerDagExecutionPlan for
// this exact nodes array. Runtime addresses and trigger indices may differ,
// but node indices, types, and topology must not change after compilation.
flagcxResult_t populateUniRunnerStaticRedTriggers(
    uniRunnerDagNode *nodes, size_t numNodes,
    const uniRunnerDagExecutionPlan *plan, void *const *nodeFlags,
    size_t numNodeFlags, flagcxReduceTrigger *triggers,
    size_t triggerCapacity, size_t *numTriggers);
flagcxResult_t normalizeUniRunnerIpcChunkSize(int64_t configuredChunkSize,
                                              size_t *chunkSize);
flagcxResult_t checkedUniRunnerIpcChunkCount(size_t bytes, size_t chunkSize,
                                             uint32_t *numChunks);
flagcxResult_t populateUniRunnerStaticIpcTriggers(
    uniRunnerDagNode *nodes, size_t numNodes,
    const uniRunnerDagExecutionPlan *plan, void *const *nodeFlags,
    size_t numNodeFlags, const uniRunnerStaticIpcTriggerConfig *config,
    flagcxIpcTrigger *triggers, size_t triggerCapacity, size_t *numTriggers,
    size_t *maxChunksPerTrigger);

flagcxResult_t cleanupUniRunner(flagcxComm_t comm);
flagcxResult_t
cleanupUniRunnerPersistentState(flagcxUniRunnerState *runnerState);
flagcxResult_t runUniRunner(flagcxComm_t comm);
#endif // FLAGCX_UNIRUNNER_IMPL_H_
