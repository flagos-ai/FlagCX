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

static flagcxResult_t
initUniRunnerStateDummy(flagcxUniRunnerState *runnerState) {
  // Initialize queues
  runnerState->readyQueue = {0};
  runnerState->inflightQueue = {0};
  runnerState->pendingQueue = {0};
  return flagcxNotSupported;
}

static flagcxResult_t
initUniRunnerStateRingAR(flagcxUniRunnerState *runnerState,
                         const void *sendbuff, void *recvbuff, size_t count,
                         flagcxDataType_t datatype, flagcxRedOp_t op,
                         flagcxComm_t comm, int numSlices = 1) {
  TRACE(FLAGCX_INIT, "rank %d initUniRunnerState called", comm->rank);

  // Initialize queues
  runnerState->readyQueue = {0};
  runnerState->inflightQueue = {0};
  runnerState->pendingQueue = {0};
  // TRACE(FLAGCX_KERNEL, "initUniRunnerState bp1 (queues initialized)");

  int rank = comm->rank;
  int nranks = comm->nranks;

  if (nranks < 2) {
    return flagcxSystemError;
  }

  int next_rank = (rank + 1) % nranks;
  int prev_rank = (rank - 1 + nranks) % nranks;
  size_t typeSize = getFlagcxDataTypeSize(datatype);

  // Pipeline configuration
  size_t rankChunkCount = count / nranks;
  size_t sliceCount = rankChunkCount / numSlices;

  // Nodes per slice chain:
  // Scatter-Reduce: (P2P + Reduce) * (nranks - 1)
  // All-Gather: P2P * (nranks - 1)
  const int nodesPerSlice = 3 * (nranks - 1);
  const int numNodes = numSlices * nodesPerSlice;

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                           numNodes * sizeof(struct uniRunnerDagNode)));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }
  // TRACE(FLAGCX_KERNEL, "initUniRunnerState bp2 (DAG nodes allocated)");

  int globalNodeIdx = 0;

  /* reduce-scatter phase (nranks - 1 steps)
   * slice = s, step = i
   * p2pNodeIdx = s * nodesPerSlice + i * 2
   * redNodeIdx = s * nodesPerSlice + i * 2 + 1
   * all-gather phase (nranks - 1 steps)
   * slice = s, step = i
   * p2pNodeIdx = s * nodesPerSlice + (nranks - 1) * 2 + i
   */
  for (int s = 0; s < numSlices; s++) {
    int sliceNodeBaseIdx = globalNodeIdx;
    size_t sliceOffsetInChunk = s * sliceCount * typeSize;

    // Phase 1: Scatter-Reduce
    for (int i = 0; i < nranks - 1; i++) {
      // P2P Node
      int p2pNodeIdx = globalNodeIdx++;
      runnerState->dagNodes[p2pNodeIdx].nodeType = uniRunnerDagNodeTypeP2p;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.numOps = 2;
      FLAGCXCHECK(
          flagcxCalloc(&runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops,
                       2 * sizeof(struct uniRunnerP2pOpData)));

      int tx_chunk = (rank - i + nranks) % nranks;
      int rx_chunk = (rank - i - 1 + nranks) % nranks;

      size_t tx_offset =
          (tx_chunk * rankChunkCount * typeSize) + sliceOffsetInChunk;
      size_t rx_offset =
          (rx_chunk * rankChunkCount * typeSize) + sliceOffsetInChunk;

      // Op 0: Send
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].type =
          flagcxDevicePrimSend;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].peerRank =
          next_rank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].count = sliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].datatype = datatype;
      // First step sends from sendbuff, others from recvbuff
      const void *srcBase = (i == 0) ? sendbuff : recvbuff;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].addr =
          static_cast<void *>(static_cast<char *>(const_cast<void *>(srcBase)) +
                              tx_offset);

      // Op 1: Recv
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].type =
          flagcxDevicePrimRecv;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].peerRank =
          prev_rank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].count = sliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].datatype = datatype;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].addr =
          static_cast<void *>(static_cast<char *>(recvbuff) + rx_offset);

      // Reduce Node
      int redNodeIdx = globalNodeIdx++;
      runnerState->dagNodes[redNodeIdx].nodeType = uniRunnerDagNodeTypeRed;
      runnerState->dagNodes[redNodeIdx].nodeData.red.input1 =
          static_cast<void *>(static_cast<char *>(recvbuff) + rx_offset);
      runnerState->dagNodes[redNodeIdx].nodeData.red.input2 =
          static_cast<void *>(
              static_cast<char *>(const_cast<void *>(sendbuff)) + rx_offset);
      runnerState->dagNodes[redNodeIdx].nodeData.red.output =
          static_cast<void *>(static_cast<char *>(recvbuff) + rx_offset);
      runnerState->dagNodes[redNodeIdx].nodeData.red.count = sliceCount;
      runnerState->dagNodes[redNodeIdx].nodeData.red.nthreads =
          UNIRUNNER_NTHREADS;
      runnerState->dagNodes[redNodeIdx].nodeData.red.datatype = datatype;
      runnerState->dagNodes[redNodeIdx].nodeData.red.redOp = op;
    }

    // Phase 2: All-Gather
    for (int i = 0; i < nranks - 1; i++) {
      int p2pNodeIdx = globalNodeIdx++;
      runnerState->dagNodes[p2pNodeIdx].nodeType = uniRunnerDagNodeTypeP2p;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.numOps = 2;
      FLAGCXCHECK(
          flagcxCalloc(&runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops,
                       2 * sizeof(struct uniRunnerP2pOpData)));

      int tx_chunk = (rank - i + 1 + nranks) % nranks;
      int rx_chunk = (rank - i + nranks) % nranks;

      size_t tx_offset =
          (tx_chunk * rankChunkCount * typeSize) + sliceOffsetInChunk;
      size_t rx_offset =
          (rx_chunk * rankChunkCount * typeSize) + sliceOffsetInChunk;

      // Op 0: Send
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].type =
          flagcxDevicePrimSend;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].peerRank =
          next_rank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].count = sliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].datatype = datatype;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].addr =
          static_cast<void *>(static_cast<char *>(recvbuff) + tx_offset);

      // Op 1: Recv
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].type =
          flagcxDevicePrimRecv;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].peerRank =
          prev_rank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].count = sliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].datatype = datatype;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].addr =
          static_cast<void *>(static_cast<char *>(recvbuff) + rx_offset);
    }

    // Setup dependencies linearly within the slice chain
    for (int i = 0; i < nodesPerSlice; i++) {
      int currIdx = sliceNodeBaseIdx + i;

      if (i == 0) {
        runnerState->dagNodes[currIdx].numParents = 0;
      } else {
        runnerState->dagNodes[currIdx].numParents = 1;
      }

      if (i == nodesPerSlice - 1) {
        runnerState->dagNodes[currIdx].numChildren = 0;
      } else {
        runnerState->dagNodes[currIdx].numChildren = 1;
        FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes[currIdx].children,
                                 sizeof(struct uniRunnerDagNode *)));
        runnerState->dagNodes[currIdx].children[0] =
            &runnerState->dagNodes[currIdx + 1];
      }
    }

    // Enqueue the head of this slice chain to Ready Queue
    dagQueueEnqueue(&runnerState->readyQueue,
                    &runnerState->dagNodes[sliceNodeBaseIdx]);

    // Enqueue the rest to Pending Queue
    for (int i = 1; i < nodesPerSlice; i++) {
      dagQueueEnqueue(&runnerState->pendingQueue,
                      &runnerState->dagNodes[sliceNodeBaseIdx + i]);
    }
  }

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

  TRACE(FLAGCX_INIT,
        "DAG scheduler initialized with %d-rank Ring AllReduce topology (%d "
        "slices)",
        nranks, numSlices);

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
        runnerState->resetEvent(current->nodeData.p2p.eventIdx);
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

void flagcxUniRunnerState::resetEvent(int idx) {
  p2pEventMap.markAvailable(idx);
}

flagcxResult_t runUniRunner(const void *sendbuff, void *recvbuff, size_t count,
                            flagcxDataType_t datatype, flagcxRedOp_t op,
                            flagcxComm_t comm, flagcxStream_t stream,
                            flagcxCommOp_t commOp) {
  flagcxFifo_t fifo = NULL;
  flagcxResult_t res = flagcxSuccess;
  flagcxHeteroComm_t hcomm = comm->hetero_comm;

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

  // Initialize DAG scheduler
  if (commOp == flagcxCommOpAllReduce) {
    FLAGCXCHECKGOTO(initUniRunnerStateRingAR(&hcomm->proxyState->uniRunnerState,
                                             sendbuff, recvbuff, count,
                                             datatype, op, comm),
                    res, out);
  } else {
    FLAGCXCHECKGOTO(initUniRunnerStateDummy(&hcomm->proxyState->uniRunnerState),
                    res, out);
  }

  // Create a dedicated stream
  flagcxStream_t red_stream;
  FLAGCXCHECKGOTO(deviceAdaptor->streamCreate(&red_stream), res, out);
  hcomm->proxyState->uniRunnerState.comm_stream = stream;
  hcomm->proxyState->uniRunnerState.red_stream = red_stream;
  // Launch collective kernel
  flagcxLaunchCollectiveKernel(hcomm->uniRunnerFifoBuffer, UNIRUNNER_NTHREADS,
                               UNIRUNNER_NBLOCKS, red_stream);

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
      break;
    }

    // Step 1: Process ready queue - write triggers to FIFO
    FLAGCXCHECK(processReadyQueue(&hcomm->proxyState->uniRunnerState, hcomm));

    // Step 2: Process inflight queue - check completion and update dependencies
    FLAGCXCHECK(processInflightQueue(&hcomm->proxyState->uniRunnerState));
  }
  deviceAdaptor->streamSynchronize(red_stream);

  // Clean up DAG scheduler
  cleanupDagScheduler(&hcomm->proxyState->uniRunnerState);

  // destroy stream
  FLAGCXCHECK(deviceAdaptor->streamSynchronize(red_stream));
  FLAGCXCHECK(deviceAdaptor->streamDestroy(red_stream));

out:
  // destroy fifo
  FLAGCXCHECK(hcomm->proxyState->uniRunnerState.fifo->flagcxRedFifoDestroy());
  delete hcomm->proxyState->uniRunnerState.fifo;
  hcomm->uniRunnerFifoBuffer = NULL;
  return res;
}
