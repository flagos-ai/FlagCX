#include "adaptor.h"
#include "collectives.h"
#include "comm.h"
#include "info.h"
#include "net.h"
#include "p2p.h"
#include "proxy.h"
#include "proxy_kernel.h"
#include "socket.h"
#include "transport.h"
#define ENABLE_TIMER 0
#include "timer.h"

#include <assert.h>
#include <string>
#include <sys/syscall.h>
#include <sys/time.h>
#include <unistd.h>
using namespace std;

// DAG queue operations
static void dagQueueInit(struct flagcxDagQueue *queue) {
  queue->head = NULL;
  queue->tail = NULL;
  queue->size = 0;
}

static void dagQueueEnqueue(struct flagcxDagQueue *queue,
                            struct flagcxDagNode *node) {
  node->next = NULL;

  if (queue->tail == NULL) { // empty queue
    queue->head = node;
    queue->tail = node;
  } else {
    queue->tail->next = node;
    queue->tail = node;
  }
  queue->size++;
}

static struct flagcxDagNode *dagQueueDequeue(struct flagcxDagQueue *queue) {
  if (queue->head == NULL) {
    return NULL;
  }
  struct flagcxDagNode *node = queue->head;
  queue->head = node->next;
  node->next = NULL;

  if (queue->head == NULL) {
    queue->tail = NULL;
  }
  queue->size--;
  return node;
}

static bool dagQueueIsEmpty(struct flagcxDagQueue *queue) {
  return queue->head == NULL;
}

// Initialize DAG scheduler with fixed topology for testing
// Creates a simple 4-node DAG:
//     Node0
//    /     \
//  Node1   Node2
//    \     /
//     Node3
static flagcxResult_t
initDagScheduler(struct flagcxDagProxyKernelState *kernelState) {
  // Initialize queues
  dagQueueInit(&kernelState->readyQueue);
  dagQueueInit(&kernelState->inflightQueue);
  dagQueueInit(&kernelState->pendingQueue);

  // Create a fixed 4-node DAG for testing
  const int numNodes = 4;
  kernelState->numDagNodes = numNodes;
  kernelState->dagNodes =
      (struct flagcxDagNode *)malloc(numNodes * sizeof(struct flagcxDagNode));
  if (kernelState->dagNodes == NULL) {
    return flagcxSystemError;
  }

  // Initialize all nodes (using Red type for testing)
  for (int i = 0; i < numNodes; i++) {
    struct flagcxDagNode *node = &kernelState->dagNodes[i];
    memset(node, 0, sizeof(struct flagcxDagNode));

    // Set node type to Red for this test
    node->nodeType = flagcxDagNodeTypeRed;

    // Initialize common DAG structure fields
    node->numParents = 0;
    node->numChildren = 0;
    node->children = NULL;
    node->next = NULL;

    // Set dummy operation parameters (can be customized later)
    node->red.input1 = NULL;
    node->red.input2 = NULL;
    node->red.output = NULL;
    node->red.count = 1024;
    node->red.nthreads = 256;
    node->red.datatype = flagcxFloat;
    node->red.redOp = flagcxSum;

    node->red.trigger = NULL;
  }

  // Node 0: Root node (no parents, 2 children)
  kernelState->dagNodes[0].numParents = 0;
  kernelState->dagNodes[0].numChildren = 2;
  kernelState->dagNodes[0].children =
      (struct flagcxDagNode **)malloc(2 * sizeof(struct flagcxDagNode *));
  if (kernelState->dagNodes[0].children == NULL) {
    free(kernelState->dagNodes);
    return flagcxSystemError;
  }
  kernelState->dagNodes[0].children[0] = &kernelState->dagNodes[1];
  kernelState->dagNodes[0].children[1] = &kernelState->dagNodes[2];

  // Node 1: Left child (1 parent, 1 child)
  kernelState->dagNodes[1].numParents = 1;
  kernelState->dagNodes[1].numChildren = 1;
  kernelState->dagNodes[1].children =
      (struct flagcxDagNode **)malloc(1 * sizeof(struct flagcxDagNode *));
  if (kernelState->dagNodes[1].children == NULL) {
    free(kernelState->dagNodes[0].children);
    free(kernelState->dagNodes);
    return flagcxSystemError;
  }
  kernelState->dagNodes[1].children[0] = &kernelState->dagNodes[3];

  // Node 2: Right child (1 parent, 1 child)
  kernelState->dagNodes[2].numParents = 1;
  kernelState->dagNodes[2].numChildren = 1;
  kernelState->dagNodes[2].children =
      (struct flagcxDagNode **)malloc(1 * sizeof(struct flagcxDagNode *));
  if (kernelState->dagNodes[2].children == NULL) {
    free(kernelState->dagNodes[1].children);
    free(kernelState->dagNodes[0].children);
    free(kernelState->dagNodes);
    return flagcxSystemError;
  }
  kernelState->dagNodes[2].children[0] = &kernelState->dagNodes[3];

  // Node 3: Leaf node (2 parents, no children)
  kernelState->dagNodes[3].numParents = 2;
  kernelState->dagNodes[3].numChildren = 0;
  kernelState->dagNodes[3].children = NULL;

  // Add root node (Node 0) to ready queue since it has no parents
  dagQueueEnqueue(&kernelState->readyQueue, &kernelState->dagNodes[0]);

  // Add Node 1 and Node 2 to pending queue (they wait for Node 0)
  dagQueueEnqueue(&kernelState->pendingQueue, &kernelState->dagNodes[1]);
  dagQueueEnqueue(&kernelState->pendingQueue, &kernelState->dagNodes[2]);

  // Add Node 3 to pending queue (it waits for Node 1 and Node 2)
  dagQueueEnqueue(&kernelState->pendingQueue, &kernelState->dagNodes[3]);

  INFO(
      FLAGCX_INIT,
      "DAG scheduler initialized with fixed 4-node topology (0->1,2 | 1,2->3)");

  return flagcxSuccess;
}

// Clean up DAG nodes and queues
static void cleanupDagScheduler(struct flagcxProxyKernelState *kernelState) {
  if (kernelState->dagNodes != NULL) {
    for (int i = 0; i < kernelState->numDagNodes; i++) {
      if (kernelState->dagNodes[i].children != NULL) {
        free(kernelState->dagNodes[i].children);
      }
    }
    free(kernelState->dagNodes);
    kernelState->dagNodes = NULL;
  }
  kernelState->numDagNodes = 0;
}

// Check trigger state and return true if complete (state == 3)
static bool checkTriggerComplete(struct flagcxDagNode *node) {
  // Get trigger pointer based on node type
  if (node->nodeType == flagcxDagNodeTypeP2p) {
    return false;
  }

  if (node->red.trigger == NULL) {
    return false;
  }

  // Use trigger's pollState method
  uint64_t state = node->red.trigger->pollState();
  return (state == flagcxReduceTriggerComplete);
}

// Mark trigger as available (state = 0)
static void markTriggerAvailable(struct flagcxDagNode *node) {
  // Get trigger pointer based on node type
  if (node->nodeType == flagcxDagNodeTypeP2p) {
    return;
  }

  if (node->red.trigger == NULL) {
    return;
  }

  // Use trigger's setState method to mark as available
  node->red.trigger->setState(flagcxReduceTriggerAvailable);
}

// Process ready queue: write triggers to FIFO and move to inflight
static flagcxResult_t
processReadyQueue(struct flagcxProxyKernelState *kernelState) {
  while (!dagQueueIsEmpty(&kernelState->readyQueue)) {
    struct flagcxDagNode *node = dagQueueDequeue(&kernelState->readyQueue);
    flagcxResult_t res = flagcxSuccess;

    if (node->nodeType == flagcxDagNodeTypeP2p) {
      // TODO: Handle P2P node enqueue

    } else {
      // Handle Red node
      // Use enqueue function from flagcx_reduce_kernel_host.cc
      res =
          enqueue((void *)kernelState->fifo->buffer, (uint64_t)node->red.input1,
                  (uint64_t)node->red.input2, (uint64_t)node->red.output,
                  node->red.count, node->red.nthreads, node->red.datatype,
                  node->red.redOp, &node->red.trigger);

      if (res != flagcxSuccess)
        return res;
    }

    // Move to inflight queue
    dagQueueEnqueue(&kernelState->inflightQueue, node);
  }

  return flagcxSuccess;
}

// Process inflight queue: check completion and update pending nodes
static flagcxResult_t
processInflightQueue(struct flagcxProxyKernelState *kernelState) {
  struct flagcxDagNode *prev = NULL;
  struct flagcxDagNode *current = kernelState->inflightQueue.head;

  while (current != NULL) {
    struct flagcxDagNode *next = current->next;

    if (checkTriggerComplete(current)) {
      // Mark trigger as available
      markTriggerAvailable(current);

      // Remove from inflight queue
      if (prev == NULL) {
        kernelState->inflightQueue.head = next;
      } else {
        prev->next = next;
      }
      if (next == NULL) {
        kernelState->inflightQueue.tail = prev;
      }
      kernelState->inflightQueue.size--;

      // Update children: decrement parent count
      for (int i = 0; i < current->numChildren; i++) {
        struct flagcxDagNode *child = current->children[i];
        child->numParents--;

        // If child has no more parents, move from pending to ready
        if (child->numParents == 0) {
          // Remove from pending queue
          struct flagcxDagNode *pendingPrev = NULL;
          struct flagcxDagNode *pendingCur = kernelState->pendingQueue.head;
          while (pendingCur != NULL) {
            struct flagcxDagNode *pendingNext = pendingCur->next;

            if (pendingCur == child) {
              if (pendingPrev == NULL) { // child is head
                kernelState->pendingQueue.head = pendingNext;
              } else {
                pendingPrev->next = pendingNext;
              }
              if (pendingNext == NULL) {
                kernelState->pendingQueue.tail = pendingPrev;
              }
              kernelState->pendingQueue.size--;
              break;
            }
            pendingPrev = pendingCur;
            pendingCur = pendingNext;
          }

          // Add to ready queue
          dagQueueEnqueue(&kernelState->readyQueue, child);
        }
      }

      current = next;
    } else {
      prev = current;
      current = next;
    }
  }

  return flagcxSuccess;
}

flagcxResult_t runUniRunner(flagcxHeteroComm_t comm) {
  int groupCount = 0;
  flagcxDeviceTrigger_t ptr = NULL;
  flagcxFifo_t fifo = NULL;
  flagcxResult_t res = flagcxSuccess;

  // Set device context
  FLAGCXCHECKGOTO(deviceAdaptor->setDevice(comm->cudaDev), res, out);

  // Create FIFO
  comm->proxyState->kernelState.fifo = new flagcxFifo();
  FLAGCXCHECKGOTO(comm->proxyState->kernelState.fifo->flagcxRedFifoInit(), res,
                  out);
  fifo = comm->proxyState->kernelState.fifo;
  // comm->fifoBuffer = (void *)comm->proxyState->kernelState.fifo->buffer;
  FLAGCXCHECKGOTO(deviceAdaptor->hostGetDevicePointer(
                      &comm->fifoBuffer,
                      (void *)comm->proxyState->kernelState.fifo->buffer),
                  res, out);

  // Create a dedicated stream
  flagcxStream_t stream;
  FLAGCXCHECKGOTO(deviceAdaptor->streamCreate(&stream), res, out);
  INFO(FLAGCX_P2P, "rank %d p2p stream %lu", comm->rank, (uintptr_t)stream);

  // Allocate trigger structure
  FLAGCXCHECKGOTO(flagcxCalloc(&ptr, sizeof(flagcxDeviceTrigger)), res, out);

  // Initialize DAG scheduler
  FLAGCXCHECKGOTO(initDagScheduler(&comm->proxyState->kernelState), res, out);

  INFO(FLAGCX_PROXY, "rank %d DAG scheduler initialized", comm->rank);

  // Main scheduling loop using DAG-based three-queue scheduling
  while (true) {
    // Check stop flag and all queues empty condition
    if (comm->proxyState->kernelState.stop == 1 &&
        dagQueueIsEmpty(&comm->proxyState->kernelState.readyQueue) &&
        dagQueueIsEmpty(&comm->proxyState->kernelState.inflightQueue) &&
        dagQueueIsEmpty(&comm->proxyState->kernelState.pendingQueue)) {
      break;
    }

    // Step 1: Process ready queue - write triggers to FIFO
    FLAGCXCHECK(processReadyQueue(&comm->proxyState->kernelState));

    // Step 2: Process inflight queue - check completion and update dependencies
    FLAGCXCHECK(processInflightQueue(&comm->proxyState->kernelState));
  }

  // Clean up DAG scheduler
  cleanupDagScheduler(&comm->proxyState->kernelState);

  // destroy stream
  FLAGCXCHECKGOTO(deviceAdaptor->streamSynchronize(stream), res, out);
  FLAGCXCHECKGOTO(deviceAdaptor->streamDestroy(stream), res, out);
  // deallocate trigger structure
  free(ptr);

out:
  // destroy fifo
  FLAGCXCHECK(comm->proxyState->kernelState.fifo->flagcxRedFifoDestroy());
  delete comm->proxyState->kernelState.fifo;
  comm->fifoBuffer = NULL;
  return res;
}
