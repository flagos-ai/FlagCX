#ifndef FLAGCX_UNIRUNNER_IMPL_H_
#define FLAGCX_UNIRUNNER_IMPL_H_

#include "device.h"
#include "flagcx.h"
#include "flagcx_kernel.h"
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
  uniRunnerDagNodeTypeCpy = 2
} uniRunnerDagNodeType;

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
  } nodeData;
};

typedef struct {
  pthread_t thread;
  flagcxFifo_t fifo;
  flagcxStream_t commStream;
  flagcxStream_t redStream;
  flagcxStream_t cpyStream;

  // new: DAG and scheduling queues
  struct uniRunnerDagNode *dagNodes; // Array of all DAG nodes
  int numDagNodes;
  int numPendingNodes;
  flagcxIntruQueue<struct uniRunnerDagNode, &uniRunnerDagNode::next>
      p2pReadyQueue;
  flagcxIntruQueue<struct uniRunnerDagNode, &uniRunnerDagNode::next>
      redReadyQueue;

  uint64_t uniRunnerNSlices;
  uint64_t uniRunnerNThreads;
  uint64_t uniRunnerNBlocks;
  uint64_t uniRunnerNRedSlices;
  uint64_t uniRunnerRedSliceSize;

  // Stream completion flags, one uint64_t per DAG node.
  void *streamFlags;
} flagcxUniRunnerState;

flagcxResult_t initUniRunnerStateDummy(flagcxUniRunnerState *runnerState);
flagcxResult_t initUniRunnerStateLocRed(flagcxUniRunnerState *runnerState,
                                        const void *sendbuff, void *recvbuff,
                                        size_t count, flagcxDataType_t datatype,
                                        flagcxRedOp_t op, flagcxComm_t comm);
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
flagcxResult_t initUniRunner(flagcxComm_t comm, flagcxStream_t stream);
flagcxResult_t cleanupUniRunner(flagcxComm_t comm);
flagcxResult_t runUniRunner(flagcxComm_t comm);
#endif // FLAGCX_UNIRUNNER_IMPL_H_
