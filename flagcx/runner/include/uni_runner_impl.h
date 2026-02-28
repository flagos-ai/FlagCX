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
  // Operation information for P2P trigger
  struct uniRunnerP2pOpData *ops; // Array of P2P operations
  int numOps;                     // Number of operations

  // Event for completion tracking
  int eventIdx; // Index of the event in the pool
};

// Reduce node data (operation-specific fields only)
struct uniRunnerRedNodeData {
  // Operation information for reduce trigger
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
  // Operation information for local memcpy
  void *src;
  void *dst;
  size_t count;
  flagcxDataType_t datatype;

  int eventIdx;
};

// Unified DAG node with common DAG structure fields
struct uniRunnerDagNode {
  uniRunnerDagNodeType nodeType; // Discriminator for union

  // Common DAG structure fields (shared by all node types)
  int numParents;                // Number of parent dependencies
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

// Bitmap for p2pEvent availability
// 1 means in use, 0 means available
typedef struct {
  uint64_t *bits;
  size_t nextIdx;

  // Check if event at index is available
  bool isAvailable(int index);
  // Get first available event index, or -1 if none
  int getAvailable();
  // Mark event at index as in use
  void markInUse(int index);
  // Mark event at index as available
  void markAvailable(int index);
} uniRunnerP2pEventBitmap;

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
  flagcxIntruQueue<struct uniRunnerDagNode, &uniRunnerDagNode::next>
      p2pInflightQueue;
  flagcxIntruQueue<struct uniRunnerDagNode, &uniRunnerDagNode::next>
      redInflightQueue;

  // P2P event pool
  flagcxEvent_t *p2pEvents;
  uniRunnerP2pEventBitmap p2pEventMap;

  // get an available event
  int getEvent();
  void resetEvent(int idx);
} flagcxUniRunnerState;

flagcxResult_t runUniRunner(const void *sendbuff, void *recvbuff,
                            void *scratchbuff, size_t count,
                            flagcxDataType_t datatype, flagcxRedOp_t op,
                            flagcxComm_t comm, flagcxStream_t stream,
                            flagcxCommOp_t commOp);
#endif // FLAGCX_UNIRUNNER_IMPL_H_
