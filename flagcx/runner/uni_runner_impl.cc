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


// Initialize DAG scheduler with 2-rank Ring AllReduce topology
// 2 P2P (Group) -> 2 Reduce -> 2 P2P (Group)
static flagcxResult_t
initDagScheduler(struct flagcxUniRunnerState *runnerState, flagcxHeteroComm_t comm) {
  // Initialize queues
  runnerState->readyQueue = {0};
  runnerState->inflightQueue = {0};
  runnerState->pendingQueue = {0};

  int rank = comm->rank;
  int peer = (rank == 0) ? 1 : 0;

  // Create 4-node DAG
  const int numNodes = 4;
  runnerState->numDagNodes = numNodes;
  runnerState->dagNodes =
      (struct flagcxDagNode *)calloc(numNodes, sizeof(struct flagcxDagNode));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }

  // Node 0: P2P Group (Scatter-Reduce phase)
  runnerState->dagNodes[0].nodeType = flagcxDagNodeTypeP2p;
  runnerState->dagNodes[0].p2p.numOps = 2;
  runnerState->dagNodes[0].p2p.ops = (struct flagcxP2pOpData *)calloc(2, sizeof(struct flagcxP2pOpData));
  
  // Op 0
  runnerState->dagNodes[0].p2p.ops[0].type = (rank == 0) ? flagcxDevicePrimSend : flagcxDevicePrimRecv;
  runnerState->dagNodes[0].p2p.ops[0].peerRank = peer;
  runnerState->dagNodes[0].p2p.ops[0].count = 1024; 
  runnerState->dagNodes[0].p2p.ops[0].datatype = flagcxFloat;
  runnerState->dagNodes[0].p2p.ops[0].addr = NULL; 
  
  // Op 1
  runnerState->dagNodes[0].p2p.ops[1].type = (rank == 0) ? flagcxDevicePrimRecv : flagcxDevicePrimSend;
  runnerState->dagNodes[0].p2p.ops[1].peerRank = peer;
  runnerState->dagNodes[0].p2p.ops[1].count = 1024;
  runnerState->dagNodes[0].p2p.ops[1].datatype = flagcxFloat;
  runnerState->dagNodes[0].p2p.ops[1].addr = NULL;

  // Node 1: Reduce
  runnerState->dagNodes[1].nodeType = flagcxDagNodeTypeRed;
  runnerState->dagNodes[1].red.count = 1024;
  runnerState->dagNodes[1].red.nthreads = 256;
  runnerState->dagNodes[1].red.datatype = flagcxFloat;
  runnerState->dagNodes[1].red.redOp = flagcxSum;
  
  // Node 2: Reduce
  runnerState->dagNodes[2].nodeType = flagcxDagNodeTypeRed;
  runnerState->dagNodes[2].red.count = 1024;
  runnerState->dagNodes[2].red.nthreads = 256;
  runnerState->dagNodes[2].red.datatype = flagcxFloat;
  runnerState->dagNodes[2].red.redOp = flagcxSum;
  
  // Node 3: P2P Group (All-Gather phase)
  runnerState->dagNodes[3].nodeType = flagcxDagNodeTypeP2p;
  runnerState->dagNodes[3].p2p.numOps = 2;
  runnerState->dagNodes[3].p2p.ops = (struct flagcxP2pOpData *)calloc(2, sizeof(struct flagcxP2pOpData));
  
  // Op 0
  runnerState->dagNodes[3].p2p.ops[0].type = (rank == 0) ? flagcxDevicePrimSend : flagcxDevicePrimRecv;
  runnerState->dagNodes[3].p2p.ops[0].peerRank = peer;
  runnerState->dagNodes[3].p2p.ops[0].count = 1024;
  runnerState->dagNodes[3].p2p.ops[0].datatype = flagcxFloat;
  runnerState->dagNodes[3].p2p.ops[0].addr = NULL;
  
  // Op 1
  runnerState->dagNodes[3].p2p.ops[1].type = (rank == 0) ? flagcxDevicePrimRecv : flagcxDevicePrimSend;
  runnerState->dagNodes[3].p2p.ops[1].peerRank = peer;
  runnerState->dagNodes[3].p2p.ops[1].count = 1024;
  runnerState->dagNodes[3].p2p.ops[1].datatype = flagcxFloat;
  runnerState->dagNodes[3].p2p.ops[1].addr = NULL;

  // Dependencies: 
  //      0
  //     / \
  //    1   2
  //     \ /
  //      3
  
  // Node 0: No parents, 2 children (Node 1, Node 2)
  runnerState->dagNodes[0].numParents = 0;
  runnerState->dagNodes[0].numChildren = 2;
  runnerState->dagNodes[0].children = (struct flagcxDagNode **)malloc(2 * sizeof(void*));
  runnerState->dagNodes[0].children[0] = &runnerState->dagNodes[1];
  runnerState->dagNodes[0].children[1] = &runnerState->dagNodes[2];
  
  // Node 1: 1 parent (Node 0), 1 child (Node 3)
  runnerState->dagNodes[1].numParents = 1;
  runnerState->dagNodes[1].numChildren = 1;
  runnerState->dagNodes[1].children = (struct flagcxDagNode **)malloc(sizeof(void*));
  runnerState->dagNodes[1].children[0] = &runnerState->dagNodes[3];
  
  // Node 2: 1 parent (Node 0), 1 child (Node 3)
  runnerState->dagNodes[2].numParents = 1;
  runnerState->dagNodes[2].numChildren = 1;
  runnerState->dagNodes[2].children = (struct flagcxDagNode **)malloc(sizeof(void*));
  runnerState->dagNodes[2].children[0] = &runnerState->dagNodes[3];
  
  // Node 3: 2 parents (Node 1, Node 2), 0 children
  runnerState->dagNodes[3].numParents = 2;
  runnerState->dagNodes[3].numChildren = 0;
  runnerState->dagNodes[3].children = NULL;
  
  // Enqueue
  dagQueueEnqueue(&runnerState->readyQueue, &runnerState->dagNodes[0]);
  dagQueueEnqueue(&runnerState->pendingQueue, &runnerState->dagNodes[1]);
  dagQueueEnqueue(&runnerState->pendingQueue, &runnerState->dagNodes[2]);
  dagQueueEnqueue(&runnerState->pendingQueue, &runnerState->dagNodes[3]);

  // Initialize P2P event pool
  runnerState->p2pEventPoolSize = 32;
  
  // Allocate event array
  runnerState->p2pEvents = (flagcxEvent_t *)malloc(runnerState->p2pEventPoolSize * sizeof(flagcxEvent_t));
  if (runnerState->p2pEvents == NULL) {
    return flagcxSystemError;
  }
  
  // Allocate free stack
  runnerState->p2pFreeStack = (int *)malloc(runnerState->p2pEventPoolSize * sizeof(int));
  if (runnerState->p2pFreeStack == NULL) {
    free(runnerState->p2pEvents);
    return flagcxSystemError;
  }
  runnerState->p2pFreeTop = 0;

  for (int i = 0; i < runnerState->p2pEventPoolSize; i++) {
    flagcxResult_t res = deviceAdaptor->eventCreate(&runnerState->p2pEvents[i], flagcxEventDisableTiming);
    if (res != flagcxSuccess) {
      return res;
    }
    // Push all indices to free stack
    runnerState->p2pFreeStack[runnerState->p2pFreeTop++] = i;
  }
  
  INFO(
      FLAGCX_INIT,
      "DAG scheduler initialized with 2-rank Ring AllReduce topology");

  return flagcxSuccess;
}

// Clean up DAG nodes and queues
static void cleanupDagScheduler(struct flagcxUniRunnerState *runnerState) {
  // Cleanup P2P events
  if (runnerState->p2pEvents != NULL) {
    for (int i = 0; i < runnerState->p2pEventPoolSize; i++) {
      deviceAdaptor->eventDestroy(runnerState->p2pEvents[i]);
    }
    free(runnerState->p2pEvents);
    runnerState->p2pEvents = NULL;
  }
  
  if (runnerState->p2pFreeStack != NULL) {
    free(runnerState->p2pFreeStack);
    runnerState->p2pFreeStack = NULL;
  }

  if (runnerState->dagNodes != NULL) {
    for (int i = 0; i < runnerState->numDagNodes; i++) {
      if (runnerState->dagNodes[i].children != NULL) {
        free(runnerState->dagNodes[i].children);
      }
      if (runnerState->dagNodes[i].nodeType == flagcxDagNodeTypeP2p && 
          runnerState->dagNodes[i].p2p.ops != NULL) {
        free(runnerState->dagNodes[i].p2p.ops);
      }
    }
    free(runnerState->dagNodes);
    runnerState->dagNodes = NULL;
  }
  runnerState->numDagNodes = 0;
}

// Process ready queue: write triggers to FIFO and move to inflight
static flagcxResult_t
processReadyQueue(struct flagcxUniRunnerState *runnerState, flagcxHeteroComm_t comm) {
  while (runnerState->readyQueue.head != NULL) {
    // Dequeue
    struct flagcxDagNode *node = runnerState->readyQueue.head;
    runnerState->readyQueue.head = node->next;
    node->next = NULL;
    if (runnerState->readyQueue.head == NULL) {
      runnerState->readyQueue.tail = NULL;
    }
    runnerState->readyQueue.size--;

    flagcxResult_t res = flagcxSuccess;

    if (node->nodeType == flagcxDagNodeTypeP2p) {
      // Check P2P inflight limit (check if free stack is empty)
      if (runnerState->p2pFreeTop == 0) {
        // Put back to head and stop processing
        node->next = runnerState->readyQueue.head;
        runnerState->readyQueue.head = node;
        if (runnerState->readyQueue.tail == NULL) {
          runnerState->readyQueue.tail = node;
        }
        runnerState->readyQueue.size++;
        break;
      }

      // Get event from pool (pop from stack)
      int eventIdx = runnerState->p2pFreeStack[--runnerState->p2pFreeTop];
      flagcxEvent_t event = runnerState->p2pEvents[eventIdx];
      
      // Prepare ops list
      struct flagcxP2pOpData *ops = node->p2p.ops;

      // Start Group
      flagcxHeteroGroupStart();

      for (int i = 0; i < p2p.numOps; i++) {
        struct flagcxP2pOpData *op = &ops[i];
        if (op->type == flagcxDevicePrimSend) {
           res = flagcxHeteroSend(op->addr, op->count, op->datatype, 
                                  op->peerRank, comm, runnerState->stream);
        } else if (op->type == flagcxDevicePrimRecv) {
           res = flagcxHeteroRecv(op->addr, op->count, op->datatype, 
                                  op->peerRank, comm, runnerState->stream);
        }
        if (res != flagcxSuccess) break;
      }
      
      if (res == flagcxSuccess) {
        flagcxHeteroGroupEnd();
        
        // Record event
        FLAGCXCHECK(deviceAdaptor->eventRecord(event, runnerState->stream));
        
        node->p2p.event = event;
        node->p2p.eventIdx = eventIdx;
      } else {
        // Handle error: Push back event
        runnerState->p2pFreeStack[runnerState->p2pFreeTop++] = eventIdx;
        return res;
      }

    } else {
      // Handle Red node
      // Use enqueue function from flagcx_reduce_kernel_host.cc
      res =
          enqueue((void *)runnerState->fifo->buffer, (uint64_t)node->red.input1,
                  (uint64_t)node->red.input2, (uint64_t)node->red.output,
                  node->red.count, node->red.nthreads, node->red.datatype,
                  node->red.redOp, &node->red.trigger);

      if (res != flagcxSuccess)
        return res;
    }

    // Move to inflight queue
    dagQueueEnqueue(&runnerState->inflightQueue, node);
  }

  return flagcxSuccess;
}

// Process inflight queue: check completion and update pending nodes
static flagcxResult_t
processInflightQueue(struct flagcxUniRunnerState *runnerState) {
  struct flagcxDagNode *prev = NULL;
  struct flagcxDagNode *current = runnerState->inflightQueue.head;

  while (current != NULL) {
    struct flagcxDagNode *next = current->next;

    bool isComplete = false;
    if (current->nodeType == flagcxDagNodeTypeP2p) {
      if (current->p2p.event != NULL) {
        flagcxResult_t status = deviceAdaptor->eventQuery(current->p2p.event);
        isComplete = (status == flagcxSuccess);
      }
    } else if (current->red.trigger != NULL) {
      uint64_t state = current->red.trigger->pollState();
      isComplete = (state == flagcxReduceTriggerComplete);
    }

    if (isComplete) {
      // Mark trigger as available
      if (current->nodeType == flagcxDagNodeTypeP2p) {
        current->p2p.event = NULL;
      } else if (current->red.trigger != NULL) {
        current->red.trigger->setState(flagcxReduceTriggerAvailable);
      }

      // Return event to pool if P2P node
      if (current->nodeType == flagcxDagNodeTypeP2p) {
        // Push back to free stack
        runnerState->p2pFreeStack[runnerState->p2pFreeTop++] = current->p2p.eventIdx;
        current->p2p.event = NULL;
      }
      
      // Remove from inflight queue
      if (prev == NULL) {
        runnerState->inflightQueue.head = next;
      } else {
        prev->next = next;
      }
      if (next == NULL) {
        runnerState->inflightQueue.tail = prev;
      }
      runnerState->inflightQueue.size--;

      // Update children: decrement parent count
      for (int i = 0; i < current->numChildren; i++) {
        struct flagcxDagNode *child = current->children[i];
        child->numParents--;

        // If child has no more parents, move from pending to ready
        if (child->numParents == 0) {
          // Remove from pending queue
          struct flagcxDagNode *pendingPrev = NULL;
          struct flagcxDagNode *pendingCur = runnerState->pendingQueue.head;
          while (pendingCur != NULL) {
            struct flagcxDagNode *pendingNext = pendingCur->next;

            if (pendingCur == child) {
              if (pendingPrev == NULL) { // child is head
                runnerState->pendingQueue.head = pendingNext;
              } else {
                pendingPrev->next = pendingNext;
              }
              if (pendingNext == NULL) {
                runnerState->pendingQueue.tail = pendingPrev;
              }
              runnerState->pendingQueue.size--;
              break;
            }
            pendingPrev = pendingCur;
            pendingCur = pendingNext;
          }

          // Add to ready queue
          dagQueueEnqueue(&runnerState->readyQueue, child);
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
  flagcxReduceTrigger_t ptr = NULL;
  flagcxFifo_t fifo = NULL;
  flagcxResult_t res = flagcxSuccess;

  // Set device context
  FLAGCXCHECKGOTO(deviceAdaptor->setDevice(comm->cudaDev), res, out);

  // Create FIFO
  comm->proxyState->uniRunnerState.fifo = new flagcxFifo();
  FLAGCXCHECKGOTO(comm->proxyState->uniRunnerState.fifo->flagcxRedFifoInit(),
                  res, out);
  fifo = comm->proxyState->uniRunnerState.fifo;
  // comm->fifoBuffer = (void *)comm->proxyState->runnerState.fifo->buffer;
  FLAGCXCHECKGOTO(deviceAdaptor->hostGetDevicePointer(
                      &comm->uniRunnerFifoBuffer,
                      (void *)comm->proxyState->uniRunnerState.fifo->buffer),
                  res, out);

  // Create a dedicated stream
  flagcxStream_t stream;
  FLAGCXCHECKGOTO(deviceAdaptor->streamCreate(&stream), res, out);
  comm->proxyState->uniRunnerState.stream = stream;
  // Allocate trigger structure
  FLAGCXCHECKGOTO(flagcxCalloc(&ptr, sizeof(flagcxReduceTrigger)), res, out);

  // Initialize DAG scheduler
  FLAGCXCHECKGOTO(initDagScheduler(&comm->proxyState->uniRunnerState, comm), res,
                  out);

  // Main scheduling loop using DAG-based three-queue scheduling
  while (true) {
    // Check stop flag and all queues empty condition
    if (comm->proxyState->uniRunnerState.readyQueue.head == NULL &&
        comm->proxyState->uniRunnerState.inflightQueue.head == NULL &&
        comm->proxyState->uniRunnerState.pendingQueue.head == NULL) {
      break;
    }

    // Step 1: Process ready queue - write triggers to FIFO
    FLAGCXCHECK(processReadyQueue(&comm->proxyState->uniRunnerState, comm));

    // Step 2: Process inflight queue - check completion and update dependencies
    FLAGCXCHECK(processInflightQueue(&comm->proxyState->uniRunnerState));
  }

  // Clean up DAG scheduler
  cleanupDagScheduler(&comm->proxyState->uniRunnerState);

  // destroy stream
  FLAGCXCHECKGOTO(deviceAdaptor->streamSynchronize(stream), res, out);
  FLAGCXCHECKGOTO(deviceAdaptor->streamDestroy(stream), res, out);
  // deallocate trigger structure
  free(ptr);

out:
  // destroy fifo
  FLAGCXCHECK(comm->proxyState->uniRunnerState.fifo->flagcxRedFifoDestroy());
  delete comm->proxyState->uniRunnerState.fifo;
  comm->uniRunnerFifoBuffer = NULL;
  return res;
}
