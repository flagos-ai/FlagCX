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

#define UNIRUNNER_NTHREADS 32
#define UNIRUNNER_NBLOCKS 1

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
static flagcxResult_t initUniRunnerState(flagcxUniRunnerState *runnerState,
                                         const void *sendbuff, void *recvbuff,
                                         size_t count,
                                         flagcxDataType_t datatype,
                                         flagcxRedOp_t op, flagcxComm_t comm) {
  TRACE(FLAGCX_INIT, "rank %d initUniRunnerState called", comm->rank);

  // Initialize queues
  runnerState->readyQueue = {0};
  runnerState->inflightQueue = {0};
  runnerState->pendingQueue = {0};
  // TRACE(FLAGCX_KERNEL, "initUniRunnerState bp1 (queues initialized)");

  // Todo: currently only support 2-rank AllReduce
  int rank = comm->rank;
  int peer = (rank == 0) ? 1 : 0;

  // Create 3-node DAG
  const int numNodes = 3;
  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                           numNodes * sizeof(struct uniRunnerDagNode)));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }
  // TRACE(FLAGCX_KERNEL, "initUniRunnerState bp2 (DAG nodes allocated)");

  // Node 0: P2P Group (Scatter-Reduce phase)
  runnerState->dagNodes[0].nodeType = uniRunnerDagNodeTypeP2p;
  runnerState->dagNodes[0].nodeData.p2p.numOps = 2;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes[0].nodeData.p2p.ops,
                           runnerState->dagNodes[0].nodeData.p2p.numOps *
                               sizeof(struct uniRunnerP2pOpData)));
  // TRACE(FLAGCX_KERNEL, "initUniRunnerState bp3 (DAG node 0 ops allocated)");

  // Op 0
  runnerState->dagNodes[0].nodeData.p2p.ops[0].type = flagcxDevicePrimSend;
  runnerState->dagNodes[0].nodeData.p2p.ops[0].peerRank = peer;
  runnerState->dagNodes[0].nodeData.p2p.ops[0].count = count / 2;
  runnerState->dagNodes[0].nodeData.p2p.ops[0].datatype = datatype;
  runnerState->dagNodes[0].nodeData.p2p.ops[0].addr = static_cast<void *>(
      static_cast<char *>(const_cast<void *>(sendbuff)) +
      (peer * (count / 2) * getFlagcxDataTypeSize(datatype)));
  // TRACE(FLAGCX_KERNEL, "initUniRunnerState bp4 (DAG node 0 op 0 set)");

  // Op 1
  runnerState->dagNodes[0].nodeData.p2p.ops[1].type = flagcxDevicePrimRecv;
  runnerState->dagNodes[0].nodeData.p2p.ops[1].peerRank = peer;
  runnerState->dagNodes[0].nodeData.p2p.ops[1].count = count / 2;
  runnerState->dagNodes[0].nodeData.p2p.ops[1].datatype = datatype;
  runnerState->dagNodes[0].nodeData.p2p.ops[1].addr = static_cast<void *>(
      static_cast<char *>(recvbuff) +
      (rank * (count / 2) * getFlagcxDataTypeSize(datatype)));
  // TRACE(FLAGCX_KERNEL, "initUniRunnerState bp5 (DAG node 0 op 1 set)");

  // Node 1: Reduce
  runnerState->dagNodes[1].nodeType = uniRunnerDagNodeTypeRed;
  runnerState->dagNodes[1].nodeData.red.input1 = static_cast<void *>(
      static_cast<char *>(recvbuff) +
      (rank * (count / 2) * getFlagcxDataTypeSize(datatype)));
  runnerState->dagNodes[1].nodeData.red.input2 = static_cast<void *>(
      static_cast<char *>(const_cast<void *>(sendbuff)) +
      (rank * (count / 2) * getFlagcxDataTypeSize(datatype)));
  runnerState->dagNodes[1].nodeData.red.output = static_cast<void *>(
      static_cast<char *>(recvbuff) +
      (rank * (count / 2) * getFlagcxDataTypeSize(datatype)));
  runnerState->dagNodes[1].nodeData.red.count = count / 2;
  runnerState->dagNodes[1].nodeData.red.nthreads = UNIRUNNER_NTHREADS;
  runnerState->dagNodes[1].nodeData.red.datatype = datatype;
  runnerState->dagNodes[1].nodeData.red.redOp = op;
  // TRACE(FLAGCX_KERNEL, "initUniRunnerState bp6 (DAG node 1 set)");

  // Node 2: P2P Group (All-Gather phase)
  runnerState->dagNodes[2].nodeType = uniRunnerDagNodeTypeP2p;
  runnerState->dagNodes[2].nodeData.p2p.numOps = 2;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes[2].nodeData.p2p.ops,
                           runnerState->dagNodes[2].nodeData.p2p.numOps *
                               sizeof(struct uniRunnerP2pOpData)));
  // TRACE(FLAGCX_KERNEL, "initUniRunnerState bp7 (DAG node 2 ops allocated)");

  // Op 0
  runnerState->dagNodes[2].nodeData.p2p.ops[0].type = flagcxDevicePrimSend;
  runnerState->dagNodes[2].nodeData.p2p.ops[0].peerRank = peer;
  runnerState->dagNodes[2].nodeData.p2p.ops[0].count = count / 2;
  runnerState->dagNodes[2].nodeData.p2p.ops[0].datatype = datatype;
  runnerState->dagNodes[2].nodeData.p2p.ops[0].addr = static_cast<void *>(
      static_cast<char *>(recvbuff) +
      (rank * (count / 2) * getFlagcxDataTypeSize(datatype)));
  // TRACE(FLAGCX_KERNEL, "initUniRunnerState bp8 (DAG node 2 op 0 set)");

  // Op 1
  runnerState->dagNodes[2].nodeData.p2p.ops[1].type = flagcxDevicePrimRecv;
  runnerState->dagNodes[2].nodeData.p2p.ops[1].peerRank = peer;
  runnerState->dagNodes[2].nodeData.p2p.ops[1].count = count / 2;
  runnerState->dagNodes[2].nodeData.p2p.ops[1].datatype = datatype;
  runnerState->dagNodes[2].nodeData.p2p.ops[1].addr = static_cast<void *>(
      static_cast<char *>(recvbuff) +
      (peer * (count / 2) * getFlagcxDataTypeSize(datatype)));
  // TRACE(FLAGCX_KERNEL, "initUniRunnerState bp9 (DAG node 2 op 1 set)");

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
  // TRACE(FLAGCX_KERNEL, "initUniRunnerState bp10 (DAG node 0 deps set)");

  // Node 1: 1 parent (Node 0), 1 child (Node 2)
  runnerState->dagNodes[1].numParents = 1;
  runnerState->dagNodes[1].numChildren = 1;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes[1].children,
                           runnerState->dagNodes[1].numChildren *
                               sizeof(struct uniRunnerDagNode *)));
  runnerState->dagNodes[1].children[0] = &runnerState->dagNodes[2];
  // TRACE(FLAGCX_KERNEL, "initUniRunnerState bp11 (DAG node 1 deps set)");

  // Node 2: 1 parent (Node 1), 0 child
  runnerState->dagNodes[2].numParents = 1;
  runnerState->dagNodes[2].numChildren = 0;
  // TRACE(FLAGCX_KERNEL, "initUniRunnerState bp12 (DAG node 2 deps set)");

  // Enqueue
  dagQueueEnqueue(&runnerState->readyQueue, &runnerState->dagNodes[0]);
  dagQueueEnqueue(&runnerState->pendingQueue, &runnerState->dagNodes[1]);
  dagQueueEnqueue(&runnerState->pendingQueue, &runnerState->dagNodes[2]);
  // TRACE(FLAGCX_KERNEL, "initUniRunnerState bp13 (DAG nodes enqueued)");

  // Initialize P2P event pool
  for (int i = 0; i < P2P_EVENT_POOL_SIZE; i++) {
    FLAGCXCHECK(deviceAdaptor->eventCreate(&runnerState->p2pEvents[i],
                                           flagcxEventDisableTiming));
  }
  // TRACE(FLAGCX_KERNEL, "initUniRunnerState bp14 (P2P events created)");
  memset(runnerState->p2pEventMap.bits, 0,
         (P2P_EVENT_POOL_SIZE + 63) / 64 * sizeof(uint64_t));
  // TRACE(FLAGCX_KERNEL, "initUniRunnerState bp15 (P2P event map
  // initialized)");

  INFO(FLAGCX_INIT,
       "DAG scheduler initialized with 2-rank Ring AllReduce topology");

  return flagcxSuccess;
}

// Clean up DAG nodes and queues
static flagcxResult_t cleanupDagScheduler(flagcxUniRunnerState *runnerState) {
  TRACE(FLAGCX_KERNEL, "cleanupDagScheduler called");
  // Cleanup P2P events
  for (int i = 0; i < P2P_EVENT_POOL_SIZE; i++) {
    FLAGCXCHECK(deviceAdaptor->eventDestroy(runnerState->p2pEvents[i]));
  }

  if (runnerState->dagNodes != NULL) {
    for (int i = 0; i < runnerState->numDagNodes; i++) {
      if (runnerState->dagNodes[i].nodeType == uniRunnerDagNodeTypeP2p &&
          runnerState->dagNodes[i].nodeData.p2p.ops != NULL) {
        free(runnerState->dagNodes[i].nodeData.p2p.ops);
      } else {
        if (runnerState->dagNodes[i].nodeData.red.trigger != NULL) {
          // log value[3] for debug
          TRACE(FLAGCX_KERNEL, "value[3] at cleanup: 0x%016lx",
                runnerState->dagNodes[i].nodeData.red.trigger->value[3]);
        }
      }
    }
    free(runnerState->dagNodes);
    runnerState->dagNodes = NULL;
  }
  runnerState->numDagNodes = 0;
  return flagcxSuccess;
}

// Process ready queue: write triggers to FIFO and move to inflight
static flagcxResult_t processReadyQueue(flagcxUniRunnerState *runnerState,
                                        flagcxHeteroComm_t comm) {
  TRACE(FLAGCX_KERNEL, "rank %d processReadyQueue called", comm->rank);

  while (runnerState->readyQueue.head != NULL) {
    // Dequeue
    struct uniRunnerDagNode *node = runnerState->readyQueue.head;
    // TRACE(FLAGCX_KERNEL, "rank %d processReadyQueue bp1 (dequeue head)",
    //       comm->rank);

    if (node->nodeType == uniRunnerDagNodeTypeP2p) {
      // Check P2P inflight limit (check if free stack is empty)
      int eventIdx = runnerState->getEvent();
      // TRACE(FLAGCX_KERNEL, "rank %d processReadyQueue bp2 (get event)",
      //       comm->rank);

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
      // TRACE(FLAGCX_KERNEL, "rank %d processReadyQueue bp3 (dequeue
      // confirmed)",
      //       comm->rank);

      // Prepare ops list
      struct uniRunnerP2pOpData *ops = node->nodeData.p2p.ops;

      // Start Group
      FLAGCXCHECK(flagcxHeteroGroupStart());

      for (int i = 0; i < node->nodeData.p2p.numOps; i++) {
        struct uniRunnerP2pOpData *op = &ops[i];
        if (op->type == flagcxDevicePrimSend) {
          FLAGCXCHECK(flagcxHeteroSend(op->addr, op->count, op->datatype,
                                       op->peerRank, comm,
                                       runnerState->comm_stream));
        } else if (op->type == flagcxDevicePrimRecv) {
          FLAGCXCHECK(flagcxHeteroRecv(op->addr, op->count, op->datatype,
                                       op->peerRank, comm,
                                       runnerState->comm_stream));
        }
      }

      FLAGCXCHECK(flagcxHeteroGroupEnd());

      // Record event
      FLAGCXCHECK(deviceAdaptor->eventRecord(event, runnerState->comm_stream));
      // TRACE(FLAGCX_KERNEL, "rank %d processReadyQueue bp4 (p2p event
      // recorded)",
      //       comm->rank);

      node->nodeData.p2p.event = event;
      node->nodeData.p2p.eventIdx = eventIdx;

    } else {
      // Handle Red node
      // Use enqueue function from flagcx_reduce_kernel_host.cc
      // Dequeue node
      runnerState->readyQueue.head = node->next;
      node->next = NULL;
      if (runnerState->readyQueue.head == NULL) {
        runnerState->readyQueue.tail = NULL;
      }
      runnerState->readyQueue.size--;
      // TRACE(FLAGCX_KERNEL, "rank %d processReadyQueue bp5 (enqueue reduce)",
      //       comm->rank);
      int idx = -1;
      FLAGCXCHECK(enqueue((void *)comm->proxyState->uniRunnerState.fifo->buffer,
                          (uintptr_t)node->nodeData.red.input1,
                          (uintptr_t)node->nodeData.red.input2,
                          (uintptr_t)node->nodeData.red.output,
                          node->nodeData.red.count, node->nodeData.red.nthreads,
                          node->nodeData.red.datatype, node->nodeData.red.redOp,
                          &idx));
      // &node->nodeData.red.trigger));
      node->nodeData.red.trigger =
          (flagcxReduceTrigger
               *)(comm->proxyState->uniRunnerState.fifo->buffer + 4) +
          idx;
      // TRACE(FLAGCX_KERNEL, "rank %d processReadyQueue bp6 (enq red
      // confirmed)",
      //       comm->rank);
    }

    // Move to inflight queue
    dagQueueEnqueue(&runnerState->inflightQueue, node);
  }

  return flagcxSuccess;
}

// Process inflight queue: check completion and update pending nodes
static flagcxResult_t processInflightQueue(flagcxUniRunnerState *runnerState) {
  TRACE(FLAGCX_KERNEL, "processInflightQueue called");

  struct uniRunnerDagNode *prev = NULL;
  struct uniRunnerDagNode *current = runnerState->inflightQueue.head;

  int flag = 0;
  int counter = 0;
  while (flag == 0) {
    struct uniRunnerDagNode *next = current->next;

    bool isComplete = false;
    if (current->nodeType == uniRunnerDagNodeTypeP2p) {
      if (current->nodeData.p2p.event != NULL) {
        isComplete = (deviceAdaptor->eventQuery(current->nodeData.p2p.event) ==
                      flagcxSuccess);
        // TRACE(FLAGCX_KERNEL, "processInflightQueue bp1 (p2p query returned
        // %d)",
        //       isComplete);
      }
    } else if (current->nodeData.red.trigger != NULL) {
      uint64_t curr_state = current->nodeData.red.trigger->pollState();
      isComplete = (curr_state == flagcxReduceTriggerComplete);
      // debug
      // uint64_t curr_c = *(runnerState->fifo->buffer + 1);
      // uint64_t curr_p = *(runnerState->fifo->buffer + 2);
      // if (counter > 1e5) {
      //   TRACE(FLAGCX_KERNEL, "processInflightQueue: timeout c=%lu, p=%lu",
      //         curr_c, curr_p);
      //   TRACE(FLAGCX_KERNEL, "value[3]: 0x%016lx",
      //         current->nodeData.red.trigger->value[3]);
      //   isComplete = 1;
      // }
    }

    if (isComplete) {
      // Mark trigger as available
      // TRACE(FLAGCX_KERNEL, "processInflightQueue bp (node complete)");
      if (current->nodeType == uniRunnerDagNodeTypeP2p) {
        runnerState->setAvail(current->nodeData.p2p.eventIdx);
        current->nodeData.p2p.eventIdx = -1;
        current->nodeData.p2p.event = NULL;
        // TRACE(FLAGCX_KERNEL, "processInflightQueue bp3 (p2p marked
        // available)");
      } else if (current->nodeData.red.trigger != NULL) {
        current->nodeData.red.trigger->setState(flagcxReduceTriggerAvailable);
        // TRACE(FLAGCX_KERNEL, "processInflightQueue bp4 (red marked
        // available)");
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
      // TRACE(FLAGCX_KERNEL, "processInflightQueue bp5 (dequeue confirmed)");

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
          // TRACE(FLAGCX_KERNEL, "processInflightQueue bp6 (pending
          // dequeued)");

          // Add to ready queue
          dagQueueEnqueue(&runnerState->readyQueue, child);
          flag = 1;
        }
      }

      current = next;
    } else {
      counter++;
      prev = current;
      current = next;
      if (current == NULL && flag == 0) {
        prev = NULL;
        current = runnerState->inflightQueue.head;
        if (current == NULL) {
          break;
        }
      }
      sched_yield();
    }
    if (current == NULL) {
      break;
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
  flagcxFifo_t fifo = NULL;
  flagcxResult_t res = flagcxSuccess;
  flagcxHeteroComm_t hcomm = comm->hetero_comm;
  size_t loop_counter = 0;

  // Set device context
  FLAGCXCHECKGOTO(deviceAdaptor->setDevice(hcomm->cudaDev), res, out);

  // Create FIFO
  hcomm->proxyState->uniRunnerState.fifo = new flagcxFifo();
  FLAGCXCHECKGOTO(hcomm->proxyState->uniRunnerState.fifo->flagcxRedFifoInit(),
                  res, out);
  fifo = hcomm->proxyState->uniRunnerState.fifo;
  // comm->fifoBuffer = (void *)comm->proxyState->runnerState.fifo->buffer;
  FLAGCXCHECKGOTO(deviceAdaptor->hostGetDevicePointer(
                      &hcomm->uniRunnerFifoBuffer,
                      (void *)hcomm->proxyState->uniRunnerState.fifo->buffer),
                  res, out);

  // Create a dedicated stream
  flagcxStream_t red_stream;
  FLAGCXCHECKGOTO(deviceAdaptor->streamCreate(&red_stream), res, out);
  hcomm->proxyState->uniRunnerState.comm_stream = stream;
  hcomm->proxyState->uniRunnerState.red_stream = red_stream;
  // Launch collective kernel
  flagcxLaunchCollectiveKernel(hcomm->uniRunnerFifoBuffer, UNIRUNNER_NTHREADS,
                               UNIRUNNER_NBLOCKS, red_stream);

  // Initialize DAG scheduler
  FLAGCXCHECKGOTO(initUniRunnerState(&hcomm->proxyState->uniRunnerState,
                                     sendbuff, recvbuff, count, datatype, op,
                                     comm),
                  res, out);

  // Main scheduling loop using DAG-based three-queue scheduling
  while (true) {
    // if (loop_counter > 1e5) {
    //   res = flagcxSystemError;
    //   TRACE(FLAGCX_KERNEL, "runUniRunner error: loop counter exceeded
    //   limit"); break;
    // }

    // Check stop flag and all queues empty condition
    if (hcomm->proxyState->uniRunnerState.readyQueue.head == NULL &&
        hcomm->proxyState->uniRunnerState.inflightQueue.head == NULL &&
        hcomm->proxyState->uniRunnerState.pendingQueue.head == NULL) {
      TRACE(FLAGCX_KERNEL,
            "runUniRunner: all queues empty, terminating runner loop");
      // set terminate flag
      __atomic_store_n(fifo->buffer + 3, 1, __ATOMIC_RELEASE);
      __sync_synchronize();
      break;
    }

    // Step 1: Process ready queue - write triggers to FIFO
    FLAGCXCHECK(processReadyQueue(&hcomm->proxyState->uniRunnerState, hcomm));

    // Step 2: Process inflight queue - check completion and update dependencies
    FLAGCXCHECK(processInflightQueue(&hcomm->proxyState->uniRunnerState));
    loop_counter++;
  }
  TRACE(FLAGCX_KERNEL, "rank %d runUniRunner bp (before sync stream)",
        hcomm->rank);
  deviceAdaptor->streamSynchronize(red_stream);
  TRACE(FLAGCX_KERNEL, "rank %d runUniRunner bp (after sync stream)",
        hcomm->rank);

  // Clean up DAG scheduler
  cleanupDagScheduler(&hcomm->proxyState->uniRunnerState);
  // TRACE(FLAGCX_KERNEL, "rank %d runUniRunner bp (DAG scheduler cleaned up)",
  //       hcomm->rank);

  // destroy stream
  FLAGCXCHECKGOTO(deviceAdaptor->streamSynchronize(red_stream), res, out);
  FLAGCXCHECKGOTO(deviceAdaptor->streamDestroy(red_stream), res, out);

out:
  // destroy fifo
  FLAGCXCHECK(hcomm->proxyState->uniRunnerState.fifo->flagcxRedFifoDestroy());
  delete hcomm->proxyState->uniRunnerState.fifo;
  hcomm->uniRunnerFifoBuffer = NULL;
  return res;
}
