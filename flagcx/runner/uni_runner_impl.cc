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

FLAGCX_PARAM(P2pEventPoolSize, "P2P_EVENT_POOL_SIZE", 1024);
FLAGCX_PARAM(UniRunnerNSlices, "UNIRUNNER_NSLICES", 1);
FLAGCX_PARAM(UniRunnerNThreads, "UNIRUNNER_NTHREADS", 32);
FLAGCX_PARAM(UniRunnerNBlocks, "UNIRUNNER_NBLOCKS", 1);

static uint64_t p2pEventPoolSize;
static uint64_t uniRunnerNSlices;
static uint64_t uniRunnerNThreads;
static uint64_t uniRunnerNBlocks;

// Check if event at index is available
bool uniRunnerP2pEventBitmap::isAvailable(int index) {
  int wordIdx = index / 64;
  int bitIdx = index % 64;
  return (bits[wordIdx] & (1ULL << bitIdx)) == 0;
}

// Get first available event index, or -1 if none
int uniRunnerP2pEventBitmap::getAvailable() {
  int ret = -1;
  for (int i = 0; i < p2pEventPoolSize; i++) {
    if (isAvailable(nextIdx)) {
      ret = nextIdx;
      nextIdx = (nextIdx + 1) % p2pEventPoolSize;
      break;
    }
    nextIdx = (nextIdx + 1) % p2pEventPoolSize;
  }
  return ret;
}

// Mark event at index as in use
void uniRunnerP2pEventBitmap::markInUse(int index) {
  int wordIdx = index / 64;
  int bitIdx = index % 64;
  bits[wordIdx] |= (1ULL << bitIdx);
}

// Mark event at index as available
void uniRunnerP2pEventBitmap::markAvailable(int index) {
  int wordIdx = index / 64;
  int bitIdx = index % 64;
  bits[wordIdx] &= ~(1ULL << bitIdx);
}

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
initUniRunnerStateLocRed(flagcxUniRunnerState *runnerState,
                         const void *sendbuff, void *recvbuff, size_t count,
                         flagcxDataType_t datatype, flagcxRedOp_t op,
                         flagcxComm_t comm, int numSlices = 1) {
  TRACE(FLAGCX_INIT,
        "rank %d initUniRunnerStateLocRed called, count=%lu, numSlices=%d",
        comm->rank, count, numSlices);

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

  size_t typeSize = getFlagcxDataTypeSize(datatype);

  // Pipeline configuration
  size_t rankChunkCount = count / nranks;
  size_t sliceCount = rankChunkCount / numSlices;

  const int numNodes = numSlices;

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                           numNodes * sizeof(struct uniRunnerDagNode)));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }

  for (int s = 0; s < numSlices; s++) {
    size_t sliceOffsetInChunk = s * sliceCount * typeSize;
    size_t rx_offset = (rank * rankChunkCount * typeSize) + sliceOffsetInChunk;

    // Reduce Node
    int redNodeIdx = s;
    runnerState->dagNodes[redNodeIdx].nodeType = uniRunnerDagNodeTypeRed;
    runnerState->dagNodes[redNodeIdx].nodeData.red.input1 =
        static_cast<void *>(static_cast<char *>(recvbuff) + rx_offset);
    runnerState->dagNodes[redNodeIdx].nodeData.red.input2 = static_cast<void *>(
        static_cast<char *>(const_cast<void *>(sendbuff)) + rx_offset);
    runnerState->dagNodes[redNodeIdx].nodeData.red.output =
        static_cast<void *>(static_cast<char *>(recvbuff) + rx_offset);
    runnerState->dagNodes[redNodeIdx].nodeData.red.count = sliceCount;
    runnerState->dagNodes[redNodeIdx].nodeData.red.nthreads = uniRunnerNThreads;
    runnerState->dagNodes[redNodeIdx].nodeData.red.datatype = datatype;
    runnerState->dagNodes[redNodeIdx].nodeData.red.redOp = op;

    // Setup dependencies linearly within the slice chain
    runnerState->dagNodes[redNodeIdx].numParents = 0;
    runnerState->dagNodes[redNodeIdx].numChildren = 0;
    // Enqueue the head of this slice chain to Ready Queue
    dagQueueEnqueue(&runnerState->readyQueue, &runnerState->dagNodes[s]);
  }

  return flagcxSuccess;
}

static flagcxResult_t
initUniRunnerStateRingAG(flagcxUniRunnerState *runnerState,
                         const void *sendbuff, void *recvbuff, size_t count,
                         flagcxDataType_t datatype, flagcxRedOp_t op,
                         flagcxComm_t comm, int numSlices = 1) {
  TRACE(FLAGCX_INIT,
        "rank %d initUniRunnerStateP2p called, count=%lu, numSlices=%d",
        comm->rank, count, numSlices);

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
  // All-Gather: P2P * (nranks - 1)
  const int nodesPerSlice = nranks - 1;
  const int numNodes = numSlices * nodesPerSlice;

  runnerState->numDagNodes = numNodes;
  FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes,
                           numNodes * sizeof(struct uniRunnerDagNode)));
  if (runnerState->dagNodes == NULL) {
    return flagcxSystemError;
  }

  int globalNodeIdx = 0;

  /* all-gather phase (nranks - 1 steps)
   * slice = s, step = i
   * p2pNodeIdx = i
   */
  for (int s = 0; s < numSlices; s++) {
    int sliceNodeBaseIdx = globalNodeIdx;
    size_t sliceOffsetInChunk = s * sliceCount * typeSize;
    TRACE(FLAGCX_INIT,
          "Initializing rank %d slice %d, baseIdx %d, rankCount %lu, "
          "sliceCount %lu",
          rank, s, sliceNodeBaseIdx, rankChunkCount, sliceCount);

    // All-Gather
    for (int i = 0; i < nranks - 1; i++) {
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
      TRACE(
          FLAGCX_INIT,
          "rank %d slice %d step %d, tx chunk %d off %lu, rx chunk %d off %lu",
          rank, s, i, tx_chunk, tx_offset, rx_chunk, rx_offset);

      // Op 0: Send
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].type =
          flagcxDevicePrimSend;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].peerRank =
          next_rank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].count = sliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].datatype = datatype;
      // First step sends from sendbuff, others from recvbuff
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

  return flagcxSuccess;
}

static flagcxResult_t
initUniRunnerStateRingAR(flagcxUniRunnerState *runnerState,
                         const void *sendbuff, void *recvbuff, size_t count,
                         flagcxDataType_t datatype, flagcxRedOp_t op,
                         flagcxComm_t comm, int numSlices = 1) {
  TRACE(FLAGCX_INIT,
        "rank %d initUniRunnerStateRingAR called, count=%lu, numSlices=%d",
        comm->rank, count, numSlices);

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
      void *srcBase = (i == 0) ? const_cast<void *>(sendbuff) : recvbuff;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].addr =
          static_cast<void *>(static_cast<char *>(srcBase) + tx_offset);

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
          uniRunnerNThreads;
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

  TRACE(FLAGCX_INIT,
        "DAG scheduler initialized with %d-rank Ring AllReduce topology (%d "
        "slices)",
        nranks, numSlices);

  return flagcxSuccess;
}

// Clean up DAG nodes
static flagcxResult_t cleanupDagScheduler(flagcxUniRunnerState *runnerState) {
  TRACE(FLAGCX_KERNEL, "cleanupDagScheduler called");

  if (runnerState->dagNodes != NULL) {
    for (int i = 0; i < runnerState->numDagNodes; i++) {
      if (runnerState->dagNodes[i].nodeType == uniRunnerDagNodeTypeP2p &&
          runnerState->dagNodes[i].nodeData.p2p.ops != NULL) {
        free(runnerState->dagNodes[i].nodeData.p2p.ops);
      }
    }
    free(runnerState->dagNodes);
    runnerState->dagNodes = NULL;
  }
  runnerState->numDagNodes = 0;
  return flagcxSuccess;
}

// Initialize P2P event pool
static flagcxResult_t initP2pEvents(flagcxUniRunnerState *runnerState) {
  FLAGCXCHECK(flagcxCalloc(&runnerState->p2pEvents,
                           p2pEventPoolSize * sizeof(flagcxEvent_t)));
  for (int i = 0; i < p2pEventPoolSize; i++) {
    FLAGCXCHECK(deviceAdaptor->eventCreate(&runnerState->p2pEvents[i],
                                           flagcxEventDisableTiming));
  }
  runnerState->p2pEventMap.nextIdx = 0;
  FLAGCXCHECK(flagcxCalloc(&runnerState->p2pEventMap.bits,
                           ((p2pEventPoolSize + 63) / 64) * sizeof(uint64_t)));
  memset(runnerState->p2pEventMap.bits, 0,
         ((p2pEventPoolSize + 63) / 64) * sizeof(uint64_t));
  return flagcxSuccess;
}

// Clean up P2P events
static flagcxResult_t cleanupP2pEvents(flagcxUniRunnerState *runnerState) {
  for (int i = 0; i < p2pEventPoolSize; i++) {
    FLAGCXCHECK(deviceAdaptor->eventDestroy(runnerState->p2pEvents[i]));
  }
  free(runnerState->p2pEvents);
  free(runnerState->p2pEventMap.bits);
  return flagcxSuccess;
}

// Process ready queue: write triggers to FIFO and move to inflight
static flagcxResult_t processReadyQueue(flagcxUniRunnerState *runnerState,
                                        flagcxHeteroComm_t comm) {
  TRACE(FLAGCX_KERNEL, "rank %d processReadyQueue called", comm->rank);
  struct uniRunnerDagNode *prev = NULL;
  struct uniRunnerDagNode *current = runnerState->readyQueue.head;

  while (current != NULL) {
    // TRACE(FLAGCX_KERNEL, "rank %d processReadyQueue bp1 (dequeue head)",
    //       comm->rank);
    struct uniRunnerDagNode *next = current->next;

    if (current->nodeType == uniRunnerDagNodeTypeP2p) {
      // Check P2P inflight limit (check if free stack is empty)
      int eventIdx = runnerState->getEvent();
      // TRACE(FLAGCX_KERNEL, "rank %d processReadyQueue bp2 (get event %d)",
      //       comm->rank, eventIdx);

      if (eventIdx == -1) {
        // prev = current;
        // current = next;
        sched_yield();
        continue; // No available event, skip for now
      }
      // Dequeue
      if (prev == NULL) {
        runnerState->readyQueue.head = next;
      } else {
        prev->next = next;
      }
      if (next == NULL) {
        runnerState->readyQueue.tail = prev;
      }
      runnerState->readyQueue.size--;
      current->next = NULL;

      // Get event from pool (pop from stack)
      flagcxEvent_t event = runnerState->p2pEvents[eventIdx];
      TRACE(FLAGCX_KERNEL,
            "rank %d processReadyQueue bp3 (dequeue %d confirmed)", comm->rank,
            eventIdx);

      // Prepare ops list
      struct uniRunnerP2pOpData *ops = current->nodeData.p2p.ops;
      // Start Group

      // deviceAdaptor->streamSynchronize(runnerState->comm_stream);
      FLAGCXCHECK(flagcxHeteroGroupStart());

      for (int i = 0; i < current->nodeData.p2p.numOps; i++) {
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
      // deviceAdaptor->streamSynchronize(runnerState->comm_stream);

      // Record event
      FLAGCXCHECK(deviceAdaptor->eventRecord(event, runnerState->comm_stream));
      TRACE(FLAGCX_KERNEL, "rank %d p2p event %d recorded on stream 0x%016lx",
            comm->rank, eventIdx, (uintptr_t)runnerState->comm_stream);

      current->nodeData.p2p.event = event;
      current->nodeData.p2p.eventIdx = eventIdx;
      dagQueueEnqueue(&runnerState->inflightQueue, current);
    } else {
      // Handle Red node
      // Use enqueue function from flagcx_reduce_kernel_host.cc
      // Dequeue node
      // runnerState->readyQueue.head = node->next;
      // node->next = NULL;
      // if (runnerState->readyQueue.head == NULL) {
      //   runnerState->readyQueue.tail = NULL;
      // }
      // runnerState->readyQueue.size--;
      // TRACE(FLAGCX_KERNEL, "rank %d processReadyQueue bp5 (enqueue reduce)",
      //       comm->rank);
      int idx = -1;
      FLAGCXCHECK(enqueue(
          (void *)comm->proxyState->uniRunnerState.fifo->buffer,
          (uintptr_t)current->nodeData.red.input1,
          (uintptr_t)current->nodeData.red.input2,
          (uintptr_t)current->nodeData.red.output, current->nodeData.red.count,
          current->nodeData.red.nthreads, current->nodeData.red.datatype,
          current->nodeData.red.redOp, &idx));
      if (idx == -1) {
        // prev = current;
        // current = next;
        sched_yield();
        continue;
      }
      // Dequeue
      if (prev == NULL) {
        runnerState->readyQueue.head = next;
      } else {
        prev->next = next;
      }
      if (next == NULL) {
        runnerState->readyQueue.tail = prev;
      }
      runnerState->readyQueue.size--;
      current->next = NULL;
      current->nodeData.red.trigger =
          (flagcxReduceTrigger
               *)(comm->proxyState->uniRunnerState.fifo->buffer + 4) +
          idx;
      dagQueueEnqueue(&runnerState->inflightQueue, current);
      // TRACE(FLAGCX_KERNEL, "rank %d processReadyQueue bp6 (enq red
      // confirmed)",
      //       comm->rank);
    }
    current = next;
  }

  return flagcxSuccess;
}

// Process inflight queue: check completion and update pending nodes
static flagcxResult_t processInflightQueue(flagcxUniRunnerState *runnerState) {
  TRACE(FLAGCX_KERNEL, "processInflightQueue called");

  struct uniRunnerDagNode *prev = NULL;
  struct uniRunnerDagNode *current = runnerState->inflightQueue.head;

  int flag = 0;
  // int counter = 0;
  while (flag == 0) {
    struct uniRunnerDagNode *next = current->next;

    bool isComplete = false;
    if (current->nodeType == uniRunnerDagNodeTypeP2p) {
      if (current->nodeData.p2p.event != NULL) {
        deviceAdaptor->streamSynchronize(runnerState->comm_stream);
        isComplete = (deviceAdaptor->eventQuery(current->nodeData.p2p.event) ==
                      flagcxSuccess);
      }
    } else if (current->nodeData.red.trigger != NULL) {
      uint64_t curr_state = current->nodeData.red.trigger->pollState();
      isComplete = (curr_state == flagcxReduceTriggerComplete);
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
        if (child->numParents == 0) {
          flag = 1;
        }
      }
      current = next;
    } else {
      // counter++;
      prev = current;
      current = next;
      if (current == NULL && flag == 0) {
        prev = NULL;
        current = runnerState->inflightQueue.head;
      }
      sched_yield();
    }
    if (current == NULL) {
      break;
    }
  }

  // Process pending queue
  // If child has no more parents, move from pending to ready
  struct uniRunnerDagNode *pendingPrev = NULL;
  struct uniRunnerDagNode *pendingCur = runnerState->pendingQueue.head;
  while (pendingCur != NULL) {
    struct uniRunnerDagNode *pendingNext = pendingCur->next;

    if (pendingCur->numParents == 0) {
      // Dequeue from pending queue
      if (pendingPrev == NULL) { // child is head
        runnerState->pendingQueue.head = pendingNext;
      } else {
        pendingPrev->next = pendingNext;
      }
      if (pendingNext == NULL) {
        runnerState->pendingQueue.tail = pendingPrev;
      }
      runnerState->pendingQueue.size--;
      // Add to ready queue
      dagQueueEnqueue(&runnerState->readyQueue, pendingCur);
    }
    pendingPrev = pendingCur;
    pendingCur = pendingNext;
  }
  // TRACE(FLAGCX_KERNEL, "processInflightQueue bp6 (pending
  // dequeued)");

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
  TRACE(FLAGCX_KERNEL,
        "resetEvent: event %d marked available, event map = 0x%016lx", idx,
        p2pEventMap.bits[0]);
}

flagcxResult_t runUniRunner(const void *sendbuff, void *recvbuff, size_t count,
                            flagcxDataType_t datatype, flagcxRedOp_t op,
                            flagcxComm_t comm, flagcxStream_t stream,
                            flagcxCommOp_t commOp) {
  flagcxFifo_t fifo = NULL;
  flagcxResult_t res = flagcxSuccess;
  flagcxHeteroComm_t hcomm = comm->hetero_comm;

  p2pEventPoolSize = flagcxParamP2pEventPoolSize();
  uniRunnerNSlices = flagcxParamUniRunnerNSlices();
  uniRunnerNThreads = flagcxParamUniRunnerNThreads();
  uniRunnerNBlocks = flagcxParamUniRunnerNBlocks();

  // Set device context
  FLAGCXCHECKGOTO(deviceAdaptor->setDevice(hcomm->cudaDev), res, out);

  // Create FIFO
  hcomm->proxyState->uniRunnerState.fifo = new flagcxFifo();
  FLAGCXCHECKGOTO(hcomm->proxyState->uniRunnerState.fifo->flagcxRedFifoInit(),
                  res, out);
  fifo = hcomm->proxyState->uniRunnerState.fifo;
  // hcomm->proxyState->uniRunnerState.fifo->buffer is the host pointer
  // hcomm->uniRunnerFifoBuffer stores the device pointer to fifo buffer
  FLAGCXCHECKGOTO(deviceAdaptor->hostGetDevicePointer(
                      &hcomm->uniRunnerFifoBuffer,
                      (void *)hcomm->proxyState->uniRunnerState.fifo->buffer),
                  res, out);

  // Initialize DAG scheduler
  if (commOp == flagcxCommOpAllReduce) {
    /* initialize uniRunnerState for ring AllReduce
    FLAGCXCHECKGOTO(initUniRunnerStateRingAR(
                        &hcomm->proxyState->uniRunnerState, sendbuff, recvbuff,
                        count, datatype, op, comm, uniRunnerNSlices),
                    res, out); */

    /* initialize uniRunnerState for reduce test
    FLAGCXCHECKGOTO(initUniRunnerStateLocRed(
                        &hcomm->proxyState->uniRunnerState, sendbuff, recvbuff,
                        count, datatype, op, comm, uniRunnerNSlices),
                    res, out); */

    /* initialize uniRunnerState for p2p test */
    FLAGCXCHECKGOTO(initUniRunnerStateRingAG(
                        &hcomm->proxyState->uniRunnerState, sendbuff, recvbuff,
                        count, datatype, op, comm, uniRunnerNSlices),
                    res, out);
  } else {
    FLAGCXCHECKGOTO(initUniRunnerStateDummy(&hcomm->proxyState->uniRunnerState),
                    res, out);
  }
  FLAGCXCHECKGOTO(initP2pEvents(&hcomm->proxyState->uniRunnerState), res, out);

  // Create a dedicated stream
  flagcxStream_t red_stream;
  FLAGCXCHECKGOTO(deviceAdaptor->streamCreate(&red_stream), res, out);
  hcomm->proxyState->uniRunnerState.comm_stream = stream;
  hcomm->proxyState->uniRunnerState.red_stream = red_stream;
  TRACE(FLAGCX_INIT, "comm stream: 0x%016lx, red stream: 0x%016lx",
        (uintptr_t)stream, (uintptr_t)red_stream);
  // Launch collective kernel
  flagcxLaunchCollectiveKernel(hcomm->uniRunnerFifoBuffer, uniRunnerNThreads,
                               uniRunnerNBlocks, red_stream);

  // Main scheduling loop using DAG-based three-queue scheduling
  while (true) {
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
  deviceAdaptor->streamSynchronize(stream);

  // Clean up DAG scheduler
  cleanupDagScheduler(&hcomm->proxyState->uniRunnerState);
  // Clean up P2P events
  cleanupP2pEvents(&hcomm->proxyState->uniRunnerState);

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
