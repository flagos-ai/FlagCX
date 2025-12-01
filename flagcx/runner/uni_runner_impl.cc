#include "uni_runner_impl.h"
#include "adaptor.h"
#include "collectives.h"
#include "comm.h"
#include "info.h"
#include "net.h"
#include "p2p.h"
#include "proxy.h"
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

#define UNIRUNNER_NTHREADS 32

// DAG queue operations

static void dagQueueEnqueue(struct uniRunnerDagQueue *queue,
                            struct uniRunnerDagNode *node) {
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
static flagcxResult_t initUniRunnerState(flagcxUniRunnerState runnerState,
                                         const void *sendbuff, void *recvbuff,
                                         size_t count,
                                         flagcxDataType_t datatype,
                                         flagcxRedOp_t op, flagcxComm_t comm) {
  // Initialize queues
  runnerState->readyQueue = {0};
  runnerState->inflightQueue = {0};
  runnerState->pendingQueue = {0};

  // Todo: currently only support 2-rank AllReduce
  int rank = comm->rank;
  int peer = (rank == 0) ? 1 : 0;

  // Create 4-node DAG
  const int numNodes = 4;
  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                           numNodes * sizeof(struct uniRunnerDagNode)));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }

  // Node 0: P2P Group (Scatter-Reduce phase)
  runnerState->dagNodes[0].nodeType = uniRunnerDagNodeTypeP2p;
  runnerState->dagNodes[0].p2p.numOps = 2;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes[0].p2p.ops,
                           runnerState->dagNodes[0].p2p.numOps *
                               sizeof(struct uniRunnerP2pOpData)));

  // Op 0
  runnerState->dagNodes[0].p2p.ops[0].type = flagcxDevicePrimSend;
  runnerState->dagNodes[0].p2p.ops[0].peerRank = peer;
  runnerState->dagNodes[0].p2p.ops[0].count = count / 2;
  runnerState->dagNodes[0].p2p.ops[0].datatype = datatype;
  runnerState->dagNodes[0].p2p.ops[0].addr =
      sendbuff + (peer * (count / 2) * getFlagcxDataTypeSize(datatype));

  // Op 1
  runnerState->dagNodes[0].p2p.ops[1].type = flagcxDevicePrimRecv;
  runnerState->dagNodes[0].p2p.ops[1].peerRank = peer;
  runnerState->dagNodes[0].p2p.ops[1].count = count;
  runnerState->dagNodes[0].p2p.ops[1].datatype = datatype;
  runnerState->dagNodes[0].p2p.ops[1].addr =
      recvbuff + (rank * (count / 2) * getFlagcxDataTypeSize(datatype));

  // Node 1: Reduce
  runnerState->dagNodes[1].nodeType = uniRunnerDagNodeTypeRed;
  runnerState->dagNodes[1].red.input1 =
      recvbuff + (rank * (count / 2) * getFlagcxDataTypeSize(datatype));
  runnerState->dagNodes[1].red.input2 =
      sendbuff + (rank * (count / 2) * getFlagcxDataTypeSize(datatype));
  runnerState->dagNodes[1].red.output =
      recvbuff + (rank * (count / 2) * getFlagcxDataTypeSize(datatype));
  runnerState->dagNodes[1].red.count = count / 2;
  runnerState->dagNodes[1].red.nthreads = UNIRUNNER_NTHREADS;
  runnerState->dagNodes[1].red.datatype = datatype;
  runnerState->dagNodes[1].red.redOp = op;

  // Node 2: P2P Group (All-Gather phase)
  runnerState->dagNodes[2].nodeType = uniRunnerDagNodeTypeP2p;
  runnerState->dagNodes[2].p2p.numOps = 2;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes[3].p2p.ops,
                           runnerState->dagNodes[2].p2p.numOps *
                               sizeof(struct uniRunnerP2pOpData)));

  // Op 0
  runnerState->dagNodes[2].p2p.ops[0].type = flagcxDevicePrimSend;
  runnerState->dagNodes[2].p2p.ops[0].peerRank = peer;
  runnerState->dagNodes[2].p2p.ops[0].count = count / 2;
  runnerState->dagNodes[2].p2p.ops[0].datatype = datatype;
  runnerState->dagNodes[2].p2p.ops[0].addr =
      recvbuff + (rank * (count / 2) * getFlagcxDataTypeSize(datatype));

  // Op 1
  runnerState->dagNodes[2].p2p.ops[1].type = flagcxDevicePrimRecv;
  runnerState->dagNodes[2].p2p.ops[1].peerRank = peer;
  runnerState->dagNodes[2].p2p.ops[1].count = count / 2;
  runnerState->dagNodes[2].p2p.ops[1].datatype = datatype;
  runnerState->dagNodes[2].p2p.ops[1].addr =
      recvbuff + (peer * (count / 2) * getFlagcxDataTypeSize(datatype));

  // Dependencies:
  //      0
  //      |
  //      1
  //      |
  //      2

  // Node 0: No parents, 1 child (Node 1)
  runnerState->dagNodes[0].numParents = 0;
  runnerState->dagNodes[0].numChildren = 1;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes[0].children,
                           runnerState->dagNodes[0].numChildren *
                               sizeof(struct uniRunnerDagNode *)));
  runnerState->dagNodes[0].children[0] = &runnerState->dagNodes[1];

  // Node 1: 1 parent (Node 0), 1 child (Node 2)
  runnerState->dagNodes[1].numParents = 1;
  runnerState->dagNodes[1].numChildren = 1;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes[1].children,
                           runnerState->dagNodes[1].numChildren *
                               sizeof(struct uniRunnerDagNode *)));
  runnerState->dagNodes[1].children[0] = &runnerState->dagNodes[2];

  // Node 2: 1 parent (Node 1), 0 child
  runnerState->dagNodes[2].numParents = 1;
  runnerState->dagNodes[2].numChildren = 0;
  runnerState->dagNodes[2].children[0] = NULL;

  // Enqueue
  dagQueueEnqueue(&runnerState->readyQueue, &runnerState->dagNodes[0]);
  dagQueueEnqueue(&runnerState->pendingQueue, &runnerState->dagNodes[1]);
  dagQueueEnqueue(&runnerState->pendingQueue, &runnerState->dagNodes[2]);

  // Initialize P2P event pool
  for (int i = 0; i < P2P_EVENT_POOL_SIZE; i++) {
    FLAGCXCHECK(deviceAdaptor->eventCreate(&runnerState->p2pEvents[i],
                                           flagcxEventDisableTiming));
  }
  runnerState->p2pEventMap.bits = {0};

  INFO(FLAGCX_INIT,
       "DAG scheduler initialized with 2-rank Ring AllReduce topology");

  return flagcxSuccess;
}

// Clean up DAG nodes and queues
static flagcxResult_t cleanupDagScheduler(flagcxUniRunnerState *runnerState) {
  // Cleanup P2P events
  for (int i = 0; i < P2P_EVENT_POOL_SIZE; i++) {
    FLAGCXCHECK(deviceAdaptor->eventDestroy(runnerState->p2pEvents[i]));
  }

  if (runnerState->dagNodes != NULL) {
    for (int i = 0; i < runnerState->numDagNodes; i++) {
      if (runnerState->dagNodes[i].nodeType == uniRunnerDagNodeTypeP2p &&
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
static flagcxResult_t processReadyQueue(flagcxUniRunnerState *runnerState,
                                        flagcxHeteroComm_t comm) {
  while (runnerState->readyQueue.head != NULL) {
    // Dequeue
    struct uniRunnerDagNode *node = runnerState->readyQueue.head;

    if (node->nodeType == uniRunnerDagNodeTypeP2p) {
      // Check P2P inflight limit (check if free stack is empty)
      int eventIdx = runnerState->getEvent();
      if (eventIdx == -1) {
        sched_yield();
        continue; // No available event, skip for now
      }
      // Dequeue
      runnerState->readyQueue.head = node->next;
      node->next = NULL;
      if (runnerState->readyQueue.head == NULL) {
        runnerState->readyQueue.tail = NULL;
      }
      runnerState->readyQueue.size--;

      // Get event from pool (pop from stack)
      flagcxEvent_t event = runnerState->p2pEvents[eventIdx];

      // Prepare ops list
      struct uniRunnerP2pOpData *ops = node->p2p.ops;

      // Start Group
      FLAGCXCHECK(flagcxHeteroGroupStart());

      for (int i = 0; i < p2p.numOps; i++) {
        struct uniRunnerP2pOpData *op = &ops[i];
        if (op->type == flagcxDevicePrimSend) {
          FLAGCXCHECK(flagcxHeteroSend(op->addr, op->count, op->datatype,
                                       op->peerRank, comm,
                                       runnerState->stream));
        } else if (op->type == flagcxDevicePrimRecv) {
          FLAGCXCHECK(flagcxHeteroRecv(op->addr, op->count, op->datatype,
                                       op->peerRank, comm,
                                       runnerState->stream));
        }
      }

      FLAGCXCHECK(flagcxHeteroGroupEnd());

      // Record event
      FLAGCXCHECK(deviceAdaptor->eventRecord(event, runnerState->stream));

      node->p2p.event = event;
      node->p2p.eventIdx = eventIdx;

    } else {
      // Handle Red node
      // Use enqueue function from flagcx_reduce_kernel_host.cc
      FLAGCXCHECK(
          enqueue((void *)runnerState->fifo->buffer, (uint64_t)node->red.input1,
                  (uint64_t)node->red.input2, (uint64_t)node->red.output,
                  node->red.count, node->red.nthreads, node->red.datatype,
                  node->red.redOp, &node->red.trigger));
    }

    // Move to inflight queue
    dagQueueEnqueue(&runnerState->inflightQueue, node);
  }

  return flagcxSuccess;
}

// Process inflight queue: check completion and update pending nodes
static flagcxResult_t processInflightQueue(flagcxUniRunnerState *runnerState) {
  struct uniRunnerDagNode *prev = NULL;
  struct uniRunnerDagNode *current = runnerState->inflightQueue.head;

  while (current != NULL) {
    struct uniRunnerDagNode *next = current->next;

    bool isComplete = false;
    if (current->nodeType == uniRunnerDagNodeTypeP2p) {
      if (current->p2p.event != NULL) {
        isComplete =
            (deviceAdaptor->eventQuery(current->p2p.event) == flagcxSuccess);
      }
    } else if (current->red.trigger != NULL) {
      isComplete =
          (current->red.trigger->pollState() == flagcxReduceTriggerComplete);
    }

    if (isComplete) {
      // Mark trigger as available
      if (current->nodeType == uniRunnerDagNodeTypeP2p) {
        runnerState->setAvail(current->p2p.eventIdx);
        current->p2p.eventIdx = -1;
        current->p2p.event = NULL;
      } else if (current->red.trigger != NULL) {
        current->red.trigger->setState(flagcxReduceTriggerAvailable);
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
        struct uniRunnerDagNode *child = current->children[i];
        child->numParents--;

        // If child has no more parents, move from pending to ready
        if (child->numParents == 0) {
          // Remove from pending queue
          struct uniRunnerDagNode *pendingPrev = NULL;
          struct uniRunnerDagNode *pendingCur = runnerState->pendingQueue.head;
          while (pendingCur != NULL) {
            struct uniRunnerDagNode *pendingNext = pendingCur->next;

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

int flagcxUniRunnerState::getEvent() {
  int idx = p2pEventMap.getAvailable();
  if (idx != -1) {
    p2pEventMap.markInUse(idx);
  }
  return idx;
}

void flagcxUniRunnerState::setAvail(int idx) { p2pEventMap.markAvailable(idx); }

flagcxResult_t runUniRunner(const void *sendbuff, void *recvbuff, size_t count,
                            flagcxDataType_t datatype, flagcxRedOp_t op,
                            flagcxComm_t comm, flagcxStream_t stream) {
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
  FLAGCXCHECKGOTO(initUniRunnerState(&comm->proxyState->uniRunnerState,
                                     sendbuff, recvbuff, count, datatype, op,
                                     comm),
                  res, out);

  // Main scheduling loop using DAG-based three-queue scheduling
  while (true) {
    // Check stop flag and all queues empty condition
    if (comm->proxyState->uniRunnerState.readyQueue.head == NULL &&
        comm->proxyState->uniRunnerState.inflightQueue.head == NULL &&
        comm->proxyState->uniRunnerState.pendingQueue.head == NULL) {
      fifo->buffer[3] = 1; // set terminate flag
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
