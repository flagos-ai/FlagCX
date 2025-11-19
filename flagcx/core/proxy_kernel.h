#ifndef FLAGCX_PROXY_KERNEL_H_
#define FLAGCX_PROXY_KERNEL_H_

#include "device.h"
#include "flagcx_kernel.h"
#include "flagcx_net.h"
#include "group.h"
#include "info.h"
#include "ipcsocket.h"
#include "launch_kernel.h"
#include "net.h"
#include "reg_pool.h"
#include "socket.h"
#include <memory>
#include <pthread.h>


// DAG node types
typedef enum {
  flagcxDagNodeTypeP2p = 0,
  flagcxDagNodeTypeRed = 1
} flagcxDagNodeType;

// P2P node data (operation-specific fields only)
struct flagcxP2pNodeData {
  // Operation information for P2P trigger
  void *addr;              // Buffer address
  size_t count;            // Element count
  int peerRank;            // Peer rank
  flagcxDataType_t datatype;  // Data type
  flagcxDevicePrim type;   // Primitive type (send/recv/term/wait)
  
  // Trigger and state tracking
  flagcxDeviceTrigger *trigger;  // Pointer to trigger in FIFO
};

// Reduce node data (operation-specific fields only)
struct flagcxRedNodeData {
  // Operation information for reduce trigger
  void *input1;
  void *input2;
  void *output;
  size_t count;
  size_t nthreads;
  flagcxDataType_t datatype;
  flagcxRedOp_t redOp;
  
  // Trigger and state tracking
  flagcxReduceTrigger *trigger;  // Pointer to trigger in FIFO
};

// Unified DAG node with common DAG structure fields
struct flagcxDagNode {
  flagcxDagNodeType nodeType;  // Discriminator for union
  
  // Common DAG structure fields (shared by all node types)
  int numParents;              // Number of parent dependencies
  int numChildren;             // Number of children
  struct flagcxDagNode **children;  // Array of child node pointers
  struct flagcxDagNode *next;  // Queue linkage
  
  // Union for type-specific operation data
  union {
    struct flagcxP2pNodeData p2p;
    struct flagcxRedNodeData red;
  };
};

// Simple queue for DAG nodes
struct flagcxDagQueue {
  struct flagcxDagNode *head;
  struct flagcxDagNode *tail;
  int size;
};

struct flagcxDagProxyKernelState {
  pthread_t thread;
  flagcxFifo_t fifo;
  flagcxStream_t stream;
  int stop = 0;
  
  // new: DAG and scheduling queues
  struct flagcxDagNode *dagNodes;  // Array of all DAG nodes
  int numDagNodes;
  struct flagcxDagQueue readyQueue;
  struct flagcxDagQueue inflightQueue;
  struct flagcxDagQueue pendingQueue;
};



struct flagcxDagProxyState {
  int refCount;
  int tpRank;
  int tpnRanks;
  int tpLocalnRanks;
  int cudaDev;
  int p2pnChannels;
  int p2pChunkSize;
  int nChannels;
  int buffSizes[FLAGCX_NUM_PROTOCOLS];
  bool allocP2pNetLLBuffers;
  bool dmaBufSupport;
  struct flagcxNetAdaptor *netAdaptor;
  volatile uint32_t *abortFlag;
  // Service threads
  pthread_t thread;
  pthread_t threadUDS;
  struct flagcxSocket listenSock;
  struct flagcxSocket ipcSock;
  int stop;
  flagcxResult_t asyncResult;
  int nRanks;

  // Used by main thread
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  union flagcxSocketAddress *peerAddresses;
  struct flagcxSocket peerSock;
  struct flagcxProxyOps proxyOps[MAXCHANNELS];

  struct flagcxProxyOps *prodProgChannelHead; /*producer*/
  struct flagcxProxyOps *consProgChannelHead; /*consumer*/

  void **sharedDevMems;
  struct flagcxIpcSocket peerIpcSock; // cuMEM API support (UDS)
  uint64_t *peerAddressesUDS;         // cuMem API support (UDS)

  // Progress thread
  struct flagcxProxyProgressState progressState;

  // Kernel thread
  bool enableProxyKernel = false;
  struct flagcxDagProxyKernelState kernelState; // new: DAG-based kernel state

  // Queue of expected responses from the proxy
  struct flagcxExpectedProxyResponse *expectedResponses;

  // flag indicating if the proxy is initialized.
  // This flag is used for lazy initialization of the proxy.
  // Cooperate with FLAGCX_RUNTIME_PROXY environment variable.
  int initialized;
};


#endif // FLAGCX_PROXY_KERNEL_H_