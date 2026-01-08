#include "uni_runner_impl.h"
#include "adaptor.h"
#include "comm.h"
#include "flagcx_hetero.h"
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
FLAGCX_PARAM(UniRunnerUseLocRed, "UNIRUNNER_USE_LOCRED", 0);
FLAGCX_PARAM(UniRunnerUseRingAG, "UNIRUNNER_USE_RINGAG", 0);

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

static flagcxResult_t
initUniRunnerStateDummy(flagcxUniRunnerState *runnerState) {
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
  flagcxIntruQueueConstruct(&runnerState->p2pReadyQueue);
  flagcxIntruQueueConstruct(&runnerState->redReadyQueue);
  flagcxIntruQueueConstruct(&runnerState->p2pInflightQueue);
  flagcxIntruQueueConstruct(&runnerState->redInflightQueue);
  runnerState->numPendingNodes = 0;
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
    size_t rxOffset = (rank * rankChunkCount * typeSize) + sliceOffsetInChunk;

    // Reduce Node
    int redNodeIdx = s;
    runnerState->dagNodes[redNodeIdx].nodeType = uniRunnerDagNodeTypeRed;
    runnerState->dagNodes[redNodeIdx].nodeData.red.input1 =
        static_cast<void *>(static_cast<char *>(recvbuff) + rxOffset);
    runnerState->dagNodes[redNodeIdx].nodeData.red.input2 = static_cast<void *>(
        static_cast<char *>(const_cast<void *>(sendbuff)) + rxOffset);
    runnerState->dagNodes[redNodeIdx].nodeData.red.output =
        static_cast<void *>(static_cast<char *>(recvbuff) + rxOffset);
    runnerState->dagNodes[redNodeIdx].nodeData.red.count = sliceCount;
    runnerState->dagNodes[redNodeIdx].nodeData.red.nthreads = uniRunnerNThreads;
    runnerState->dagNodes[redNodeIdx].nodeData.red.datatype = datatype;
    runnerState->dagNodes[redNodeIdx].nodeData.red.redOp = op;

    // Setup dependencies linearly within the slice chain
    runnerState->dagNodes[redNodeIdx].numParents = 0;
    runnerState->dagNodes[redNodeIdx].numChildren = 0;
    // Enqueue the head of this slice chain to Ready Queue
    flagcxIntruQueueEnqueue(&runnerState->redReadyQueue,
                            &runnerState->dagNodes[redNodeIdx]);
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
  flagcxIntruQueueConstruct(&runnerState->p2pReadyQueue);
  flagcxIntruQueueConstruct(&runnerState->redReadyQueue);
  flagcxIntruQueueConstruct(&runnerState->p2pInflightQueue);
  flagcxIntruQueueConstruct(&runnerState->redInflightQueue);
  runnerState->numPendingNodes = 0;
  // TRACE(FLAGCX_KERNEL, "initUniRunnerState bp1 (queues initialized)");

  int rank = comm->rank;
  int nranks = comm->nranks;

  if (nranks < 2) {
    return flagcxSystemError;
  }

  int nextRank = (rank + 1) % nranks;
  int prevRank = (rank - 1 + nranks) % nranks;
  size_t typeSize = getFlagcxDataTypeSize(datatype);

  // Pipeline configuration
  size_t rankChunkCount = count / nranks;
  size_t sliceCount = rankChunkCount / numSlices;

  // Nodes per slice chain:
  // All-Gather: P2P * (nranks - 1)
  const int nodesPerSlice = nranks - 1;
  const int numNodes = numSlices * nodesPerSlice;

  runnerState->numDagNodes = numNodes + 1;
  FLAGCXCHECK(
      flagcxCalloc(&runnerState->dagNodes,
                   runnerState->numDagNodes * sizeof(struct uniRunnerDagNode)));
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

      int txChunk = (rank - i + nranks) % nranks;
      int rxChunk = (rank - i - 1 + nranks) % nranks;

      size_t txOffset =
          (txChunk * rankChunkCount * typeSize) + sliceOffsetInChunk;
      size_t rxOffset =
          (rxChunk * rankChunkCount * typeSize) + sliceOffsetInChunk;
      TRACE(
          FLAGCX_INIT,
          "rank %d slice %d step %d, tx chunk %d off %lu, rx chunk %d off %lu",
          rank, s, i, txChunk, txOffset, rxChunk, rxOffset);

      // Op 0: Send
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].type =
          flagcxDevicePrimSend;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].peerRank = nextRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].count = sliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].datatype = datatype;
      // First step sends from sendbuff, others from recvbuff
      void *srcBase = (i == 0) ? const_cast<void *>(sendbuff) : recvbuff;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].addr =
          static_cast<void *>(static_cast<char *>(srcBase) + txOffset);

      // Op 1: Recv
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].type =
          flagcxDevicePrimRecv;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].peerRank = prevRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].count = sliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].datatype = datatype;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].addr =
          static_cast<void *>(static_cast<char *>(recvbuff) + rxOffset);
    }

    // Setup dependencies linearly within the slice chain
    for (int i = 0; i < nodesPerSlice; i++) {
      int currIdx = sliceNodeBaseIdx + i;

      if (currIdx == 0) {
        runnerState->dagNodes[currIdx].numParents = 0;
      } else {
        runnerState->dagNodes[currIdx].numParents = 1;
      }
      if (currIdx == numNodes - 1) {
        runnerState->dagNodes[currIdx].numChildren = 0;
      } else {
        runnerState->dagNodes[currIdx].numChildren = 1;
      }
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes[currIdx].children,
                               runnerState->dagNodes[currIdx].numChildren *
                                   sizeof(int)));
      if (s == numSlices - 1) {
        if (currIdx != numNodes - 1) {
          runnerState->dagNodes[currIdx].children[0] = i + 1;
        }
      } else {
        runnerState->dagNodes[currIdx].children[0] = currIdx + nodesPerSlice;
      }
    }
  }
  // Copy local chunk from sendbuff to recvbuff before starting AG
  size_t localChunkOffset = rank * rankChunkCount * typeSize;
  int cpyNodeIdx = globalNodeIdx++;
  runnerState->dagNodes[cpyNodeIdx].nodeType = uniRunnerDagNodeTypeCpy;
  runnerState->dagNodes[cpyNodeIdx].nodeData.cpy.src = static_cast<void *>(
      static_cast<char *>(const_cast<void *>(sendbuff)) + localChunkOffset);
  runnerState->dagNodes[cpyNodeIdx].nodeData.cpy.dst =
      static_cast<void *>(static_cast<char *>(recvbuff) + localChunkOffset);
  runnerState->dagNodes[cpyNodeIdx].nodeData.cpy.count = rankChunkCount;
  runnerState->dagNodes[cpyNodeIdx].nodeData.cpy.datatype = datatype;
  runnerState->dagNodes[cpyNodeIdx].numParents = 0;
  runnerState->dagNodes[cpyNodeIdx].numChildren = 0;
  flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                          &runnerState->dagNodes[cpyNodeIdx]);
  flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                          &runnerState->dagNodes[0]);
  runnerState->numPendingNodes = numNodes - 1;

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
  flagcxIntruQueueConstruct(&runnerState->p2pReadyQueue);
  flagcxIntruQueueConstruct(&runnerState->redReadyQueue);
  flagcxIntruQueueConstruct(&runnerState->p2pInflightQueue);
  flagcxIntruQueueConstruct(&runnerState->redInflightQueue);
  runnerState->numPendingNodes = 0;
  // TRACE(FLAGCX_KERNEL, "initUniRunnerState bp1 (queues initialized)");

  int rank = comm->rank;
  int nranks = comm->nranks;

  if (nranks < 2) {
    return flagcxSystemError;
  }

  int nextRank = (rank + 1) % nranks;
  int prevRank = (rank - 1 + nranks) % nranks;
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
    // int sliceNodeBaseIdx = globalNodeIdx;
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

      int txChunk = (rank - i + nranks) % nranks;
      int rxChunk = (rank - i - 1 + nranks) % nranks;

      size_t txOffset =
          (txChunk * rankChunkCount * typeSize) + sliceOffsetInChunk;
      size_t rxOffset =
          (rxChunk * rankChunkCount * typeSize) + sliceOffsetInChunk;

      // Op 0: Send
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].type =
          flagcxDevicePrimSend;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].peerRank = nextRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].count = sliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].datatype = datatype;
      // First step sends from sendbuff, others from recvbuff
      void *srcBase = (i == 0) ? const_cast<void *>(sendbuff) : recvbuff;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].addr =
          static_cast<void *>(static_cast<char *>(srcBase) + txOffset);

      // Op 1: Recv
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].type =
          flagcxDevicePrimRecv;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].peerRank = prevRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].count = sliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].datatype = datatype;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].addr =
          static_cast<void *>(static_cast<char *>(recvbuff) + rxOffset);

      // Set up p2p node dependency
      if (p2pNodeIdx == 0) {
        runnerState->dagNodes[p2pNodeIdx].numParents = 0;
        flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                                &runnerState->dagNodes[p2pNodeIdx]);
      } else {
        if (i == 0) {
          runnerState->dagNodes[p2pNodeIdx].numParents = 1;
        } else {
          runnerState->dagNodes[p2pNodeIdx].numParents = 2;
        }
        runnerState->numPendingNodes++;
      }
      runnerState->dagNodes[p2pNodeIdx].numChildren = 2;
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes[p2pNodeIdx].children,
                               runnerState->dagNodes[p2pNodeIdx].numChildren *
                                   sizeof(int)));
      if (s == numSlices - 1) {
        runnerState->dagNodes[p2pNodeIdx].children[0] = 2 * (i + 1);
        TRACE(FLAGCX_INIT, "rank %d p2pNode %d child 0: %d", rank, p2pNodeIdx,
              2 * (i + 1));
      } else {
        runnerState->dagNodes[p2pNodeIdx].children[0] =
            p2pNodeIdx + nodesPerSlice;
        TRACE(FLAGCX_INIT, "rank %d p2pNode %d child 0: %d", rank, p2pNodeIdx,
              p2pNodeIdx + nodesPerSlice);
      }
      runnerState->dagNodes[p2pNodeIdx].children[1] = p2pNodeIdx + 1;
      TRACE(FLAGCX_INIT, "rank %d p2pNode %d child 1: %d", rank, p2pNodeIdx,
            p2pNodeIdx + 1);

      // Reduce Node
      int redNodeIdx = globalNodeIdx++;
      runnerState->dagNodes[redNodeIdx].nodeType = uniRunnerDagNodeTypeRed;
      runnerState->dagNodes[redNodeIdx].nodeData.red.input1 =
          static_cast<void *>(static_cast<char *>(recvbuff) + rxOffset);
      runnerState->dagNodes[redNodeIdx].nodeData.red.input2 =
          static_cast<void *>(
              static_cast<char *>(const_cast<void *>(sendbuff)) + rxOffset);
      runnerState->dagNodes[redNodeIdx].nodeData.red.output =
          static_cast<void *>(static_cast<char *>(recvbuff) + rxOffset);
      runnerState->dagNodes[redNodeIdx].nodeData.red.count = sliceCount;
      runnerState->dagNodes[redNodeIdx].nodeData.red.nthreads =
          uniRunnerNThreads;
      runnerState->dagNodes[redNodeIdx].nodeData.red.datatype = datatype;
      runnerState->dagNodes[redNodeIdx].nodeData.red.redOp = op;

      // Set up red node dependency
      runnerState->numPendingNodes++;
      runnerState->dagNodes[redNodeIdx].numParents = 1;
      runnerState->dagNodes[redNodeIdx].numChildren = 1;
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes[redNodeIdx].children,
                               runnerState->dagNodes[redNodeIdx].numChildren *
                                   sizeof(int)));
      runnerState->dagNodes[redNodeIdx].children[0] = redNodeIdx + 1;
      TRACE(FLAGCX_INIT, "rank %d redNode %d child 0: %d", rank, redNodeIdx,
            redNodeIdx + 1);
    }

    // Phase 2: All-Gather
    for (int i = 0; i < nranks - 1; i++) {
      int p2pNodeIdx = globalNodeIdx++;
      runnerState->dagNodes[p2pNodeIdx].nodeType = uniRunnerDagNodeTypeP2p;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.numOps = 2;
      FLAGCXCHECK(
          flagcxCalloc(&runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops,
                       2 * sizeof(struct uniRunnerP2pOpData)));

      int txChunk = (rank - i + 1 + nranks) % nranks;
      int rxChunk = (rank - i + nranks) % nranks;

      size_t txOffset =
          (txChunk * rankChunkCount * typeSize) + sliceOffsetInChunk;
      size_t rxOffset =
          (rxChunk * rankChunkCount * typeSize) + sliceOffsetInChunk;

      // Op 0: Send
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].type =
          flagcxDevicePrimSend;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].peerRank = nextRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].count = sliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].datatype = datatype;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[0].addr =
          static_cast<void *>(static_cast<char *>(recvbuff) + txOffset);

      // Op 1: Recv
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].type =
          flagcxDevicePrimRecv;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].peerRank = prevRank;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].count = sliceCount;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].datatype = datatype;
      runnerState->dagNodes[p2pNodeIdx].nodeData.p2p.ops[1].addr =
          static_cast<void *>(static_cast<char *>(recvbuff) + rxOffset);

      // Set up all-gather phase p2p node dependency
      runnerState->numPendingNodes++;
      if (i == 0) {
        runnerState->dagNodes[p2pNodeIdx].numParents = 2;
      } else {
        runnerState->dagNodes[p2pNodeIdx].numParents = 1;
      }
      if (p2pNodeIdx == numNodes - 1) {
        runnerState->dagNodes[p2pNodeIdx].numChildren = 0;
      } else {
        runnerState->dagNodes[p2pNodeIdx].numChildren = 1;
      }
      FLAGCXCHECK(flagcxCalloc(&runnerState->dagNodes[p2pNodeIdx].children,
                               runnerState->dagNodes[p2pNodeIdx].numChildren *
                                   sizeof(int)));
      if (s == numSlices - 1) {
        if (p2pNodeIdx != numNodes - 1) {
          runnerState->dagNodes[p2pNodeIdx].children[0] = 2 * nranks + i - 1;
          TRACE(FLAGCX_INIT, "rank %d p2pNode %d child 1: %d", rank, p2pNodeIdx,
                2 * nranks + i - 1);
        }
      } else {
        runnerState->dagNodes[p2pNodeIdx].children[0] =
            p2pNodeIdx + nodesPerSlice;
        TRACE(FLAGCX_INIT, "rank %d p2pNode %d child 1: %d", rank, p2pNodeIdx,
              p2pNodeIdx + nodesPerSlice);
      }
    }
  }

  TRACE(FLAGCX_INIT,
        "DAG scheduler initialized with %d-rank Ring AllReduce topology (%d "
        "slices)",
        nranks, numSlices);
  // print dependency graph
  for (int i = 0; i < runnerState->numDagNodes; i++) {
    TRACE(FLAGCX_INIT, "Node %d: type=%s, numParents=%d, numChildren=%d", i,
          (runnerState->dagNodes[i].nodeType == uniRunnerDagNodeTypeP2p)
              ? "P2P"
              : "RED",
          runnerState->dagNodes[i].numParents,
          runnerState->dagNodes[i].numChildren);
    if (runnerState->dagNodes[i].numChildren > 0) {
      std::string childStr = "  Children: ";
      for (int c = 0; c < runnerState->dagNodes[i].numChildren; c++) {
        childStr += std::to_string(runnerState->dagNodes[i].children[c]) + " ";
      }
      TRACE(FLAGCX_INIT, "%s", childStr.c_str());
    }
  }

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

static flagcxResult_t launchP2pOps(flagcxUniRunnerState *runnerState,
                                   flagcxHeteroComm_t comm, int eventIdx) {
  // Dequeue
  uniRunnerDagNode *current =
      flagcxIntruQueueDequeue(&runnerState->p2pReadyQueue);

  // Get event from pool (pop from stack)
  flagcxEvent_t event = runnerState->p2pEvents[eventIdx];
  TRACE(FLAGCX_KERNEL, "rank %d processReadyQueue bp3 (dequeue %d confirmed)",
        comm->rank, eventIdx);

  if (current->nodeType == uniRunnerDagNodeTypeP2p) {
    // Prepare ops list
    struct uniRunnerP2pOpData *ops = current->nodeData.p2p.ops;

    // Start Group P2P
    FLAGCXCHECK(flagcxHeteroGroupStart());
    for (int i = 0; i < current->nodeData.p2p.numOps; i++) {
      struct uniRunnerP2pOpData *op = &ops[i];
      if (op->type == flagcxDevicePrimSend) {
        FLAGCXCHECK(flagcxHeteroSend(op->addr, op->count, op->datatype,
                                     op->peerRank, comm,
                                     runnerState->commStream));
      } else if (op->type == flagcxDevicePrimRecv) {
        FLAGCXCHECK(flagcxHeteroRecv(op->addr, op->count, op->datatype,
                                     op->peerRank, comm,
                                     runnerState->commStream));
      }
    }
    FLAGCXCHECK(flagcxHeteroGroupEnd());

    // Record event
    FLAGCXCHECK(deviceAdaptor->eventRecord(event, runnerState->commStream));
    TRACE(FLAGCX_KERNEL, "rank %d p2p event %d recorded on stream 0x%016lx",
          comm->rank, eventIdx, (uintptr_t)runnerState->commStream);

    current->nodeData.p2p.eventIdx = eventIdx;
  } else if (current->nodeType == uniRunnerDagNodeTypeCpy) {
    // Launch copy
    FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
        current->nodeData.cpy.dst, current->nodeData.cpy.src,
        current->nodeData.cpy.count *
            getFlagcxDataTypeSize(current->nodeData.cpy.datatype),
        flagcxMemcpyDeviceToDevice, runnerState->cpyStream, NULL));

    // Record event
    FLAGCXCHECK(deviceAdaptor->eventRecord(event, runnerState->cpyStream));
    TRACE(FLAGCX_KERNEL, "rank %d cpy event %d recorded on stream 0x%016lx",
          comm->rank, eventIdx, (uintptr_t)runnerState->cpyStream);

    current->nodeData.cpy.eventIdx = eventIdx;
  } else {
    return flagcxSystemError;
  }
  flagcxIntruQueueEnqueue(&runnerState->p2pInflightQueue, current);

  return flagcxSuccess;
}

static flagcxResult_t enqueueReadyQueue(flagcxUniRunnerState *runnerState,
                                        int nodeIdx) {
  if (runnerState->dagNodes[nodeIdx].nodeType == uniRunnerDagNodeTypeP2p ||
      runnerState->dagNodes[nodeIdx].nodeType == uniRunnerDagNodeTypeCpy) {
    flagcxIntruQueueEnqueue(&runnerState->p2pReadyQueue,
                            &runnerState->dagNodes[nodeIdx]);
  } else if (runnerState->dagNodes[nodeIdx].nodeType ==
             uniRunnerDagNodeTypeRed) {
    flagcxIntruQueueEnqueue(&runnerState->redReadyQueue,
                            &runnerState->dagNodes[nodeIdx]);
  } else {
    return flagcxNotSupported;
  }
  runnerState->numPendingNodes--;
  return flagcxSuccess;
}

// Process ready queue: write triggers to FIFO and move to inflight
static flagcxResult_t processReadyQueue(flagcxUniRunnerState *runnerState,
                                        flagcxHeteroComm_t comm) {
  // TRACE(FLAGCX_KERNEL, "rank %d processReadyQueue called", comm->rank);

  // process p2pReadyQueue
  while (!flagcxIntruQueueEmpty(&runnerState->p2pReadyQueue)) {
    int eventIdx = runnerState->getEvent();
    if (eventIdx == -1) {
      sched_yield();
      break; // No available event, skip for now
    }
    FLAGCXCHECK(launchP2pOps(runnerState, comm, eventIdx));
  }

  // process redReadyQueue
  while (!flagcxIntruQueueEmpty(&runnerState->redReadyQueue)) {
    struct uniRunnerDagNode *current =
        flagcxIntruQueueHead(&runnerState->redReadyQueue);
    int idx = -1;
    FLAGCXCHECK(enqueue(
        (void *)runnerState->fifo->buffer,
        (uintptr_t)current->nodeData.red.input1,
        (uintptr_t)current->nodeData.red.input2,
        (uintptr_t)current->nodeData.red.output, current->nodeData.red.count,
        current->nodeData.red.nthreads, current->nodeData.red.datatype,
        current->nodeData.red.redOp, &idx));
    if (idx == -1) {
      sched_yield();
      break; // FIFO full, skip for now
    }
    // Dequeue
    flagcxIntruQueueDequeue(&runnerState->redReadyQueue);
    current->nodeData.red.triggerIdx = idx;
    flagcxIntruQueueEnqueue(&runnerState->redInflightQueue, current);
  }

  return flagcxSuccess;
}

// Process inflight queue: check completion and update pending nodes
static flagcxResult_t processInflightQueue(flagcxUniRunnerState *runnerState) {
  // TRACE(FLAGCX_KERNEL, "processInflightQueue called");

  // process p2pInflightQueue
  uniRunnerDagNode *prev = nullptr;
  uniRunnerDagNode *curr = flagcxIntruQueueHead(&runnerState->p2pInflightQueue);
  while (curr) {
    if (curr->nodeType == uniRunnerDagNodeTypeP2p) {
      if (deviceAdaptor->eventQuery(
              runnerState->p2pEvents[curr->nodeData.p2p.eventIdx]) ==
          flagcxSuccess) {
        runnerState->resetEvent(curr->nodeData.p2p.eventIdx);
        curr->nodeData.p2p.eventIdx = -1;
        for (int i = 0; i < curr->numChildren; i++) {
          runnerState->dagNodes[curr->children[i]].numParents--;
          if (runnerState->dagNodes[curr->children[i]].numParents == 0) {
            FLAGCXCHECK(enqueueReadyQueue(runnerState, curr->children[i]));
          }
        }
        curr = flagcxIntruQueueRemove(&runnerState->p2pInflightQueue, prev);
      } else {
        prev = curr;
        curr = curr->next;
      }
    } else if (curr->nodeType == uniRunnerDagNodeTypeCpy) {
      if (deviceAdaptor->eventQuery(
              runnerState->p2pEvents[curr->nodeData.cpy.eventIdx]) ==
          flagcxSuccess) {
        runnerState->resetEvent(curr->nodeData.cpy.eventIdx);
        curr->nodeData.cpy.eventIdx = -1;
        for (int i = 0; i < curr->numChildren; i++) {
          runnerState->dagNodes[curr->children[i]].numParents--;
          if (runnerState->dagNodes[curr->children[i]].numParents == 0) {
            FLAGCXCHECK(enqueueReadyQueue(runnerState, curr->children[i]));
          }
        }
        curr = flagcxIntruQueueRemove(&runnerState->p2pInflightQueue, prev);
      } else {
        prev = curr;
        curr = curr->next;
      }
    } else {
      return flagcxSystemError;
    }
  }

  // process redInflightQueue
  prev = nullptr;
  curr = flagcxIntruQueueHead(&runnerState->redInflightQueue);
  while (curr) {
    flagcxReduceTrigger_t trigger =
        (flagcxReduceTrigger *)(runnerState->fifo->buffer + 4) +
        curr->nodeData.red.triggerIdx;
    if (trigger->pollState() == flagcxReduceTriggerComplete) {
      trigger->setState(flagcxReduceTriggerAvailable);
      for (int i = 0; i < curr->numChildren; i++) {
        runnerState->dagNodes[curr->children[i]].numParents--;
        if (runnerState->dagNodes[curr->children[i]].numParents == 0) {
          FLAGCXCHECK(enqueueReadyQueue(runnerState, curr->children[i]));
        }
      }
      curr = flagcxIntruQueueRemove(&runnerState->redInflightQueue, prev);
    } else {
      prev = curr;
      curr = curr->next;
    }
  }

  return flagcxSuccess;
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
    if (flagcxParamUniRunnerUseLocRed()) {
      /* initialize uniRunnerState for reduce test */
      FLAGCXCHECKGOTO(
          initUniRunnerStateLocRed(&hcomm->proxyState->uniRunnerState, sendbuff,
                                   recvbuff, count, datatype, op, comm,
                                   uniRunnerNSlices),
          res, out);
    } else if (flagcxParamUniRunnerUseRingAG()) {
      /* initialize uniRunnerState for p2p test */
      FLAGCXCHECKGOTO(
          initUniRunnerStateRingAG(&hcomm->proxyState->uniRunnerState, sendbuff,
                                   recvbuff, count, datatype, op, comm,
                                   uniRunnerNSlices),
          res, out);
    } else {
      /* initialize uniRunnerState for ring AllReduce */
      FLAGCXCHECKGOTO(
          initUniRunnerStateRingAR(&hcomm->proxyState->uniRunnerState, sendbuff,
                                   recvbuff, count, datatype, op, comm,
                                   uniRunnerNSlices),
          res, out);
    }
  } else {
    FLAGCXCHECKGOTO(initUniRunnerStateDummy(&hcomm->proxyState->uniRunnerState),
                    res, out);
  }
  FLAGCXCHECKGOTO(initP2pEvents(&hcomm->proxyState->uniRunnerState), res, out);

  // Create dedicated reduce and copy streams
  flagcxStream_t redStream;
  FLAGCXCHECKGOTO(deviceAdaptor->streamCreate(&redStream), res, out);
  flagcxStream_t cpyStream;
  FLAGCXCHECKGOTO(deviceAdaptor->streamCreate(&cpyStream), res, out);
  hcomm->proxyState->uniRunnerState.commStream = stream;
  hcomm->proxyState->uniRunnerState.redStream = redStream;
  hcomm->proxyState->uniRunnerState.cpyStream = cpyStream;
#ifdef COMPILE_KERNEL_HOST
  // Launch collective kernel
  flagcxLaunchCollectiveKernel(hcomm->uniRunnerFifoBuffer, uniRunnerNThreads,
                               uniRunnerNBlocks, redStream);
#endif

  // Main scheduling loop using DAG-based three-queue scheduling
  while (true) {
    // Check stop flag and all queues empty condition
    if (flagcxIntruQueueEmpty(
            &hcomm->proxyState->uniRunnerState.p2pReadyQueue) &&
        flagcxIntruQueueEmpty(
            &hcomm->proxyState->uniRunnerState.redReadyQueue) &&
        flagcxIntruQueueEmpty(
            &hcomm->proxyState->uniRunnerState.p2pInflightQueue) &&
        flagcxIntruQueueEmpty(
            &hcomm->proxyState->uniRunnerState.redInflightQueue) &&
        hcomm->proxyState->uniRunnerState.numPendingNodes == 0) {
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
  deviceAdaptor->streamSynchronize(redStream);
  deviceAdaptor->streamSynchronize(stream);

  // Clean up DAG scheduler
  cleanupDagScheduler(&hcomm->proxyState->uniRunnerState);
  // Clean up P2P events
  cleanupP2pEvents(&hcomm->proxyState->uniRunnerState);

  // destroy stream
  FLAGCXCHECK(deviceAdaptor->streamSynchronize(redStream));
  FLAGCXCHECK(deviceAdaptor->streamDestroy(redStream));

out:
  // destroy fifo
  FLAGCXCHECK(hcomm->proxyState->uniRunnerState.fifo->flagcxRedFifoDestroy());
  delete hcomm->proxyState->uniRunnerState.fifo;
  hcomm->uniRunnerFifoBuffer = NULL;
  return res;
}
