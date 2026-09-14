#ifndef FLAGCX_HETERO_H_
#define FLAGCX_HETERO_H_

#include "flagcx.h"
#include "type.h"
#include <climits>
#include <pthread.h>
#include <stdint.h>

template <typename T, T *T::*next>
struct flagcxIntruQueue;

enum flagcxRmaDescType {
  FLAGCX_RMA_PUT = 0,
  FLAGCX_RMA_PUT_SIGNAL = 1,
  FLAGCX_RMA_GET = 2,
  FLAGCX_RMA_PUT_VALUE = 3,
};

// One-sided adaptors report request-pool or send-queue pressure exclusively as
// flagcxInProgress. All other errors are permanent for the current RMA epoch.
static inline bool flagcxRmaPostResultIsRetryable(flagcxResult_t result) {
  return result == flagcxInProgress;
}

static inline bool flagcxRmaBatchPostResultIsFatal(flagcxResult_t result,
                                                   int posted, int count) {
  if (posted < 0 || posted > count)
    return true;
  if (result == flagcxSuccess)
    return posted != count;
  return result != flagcxInProgress;
}

struct flagcxRmaDesc {
  int peer;
  enum flagcxRmaDescType type;
  uint64_t srcOff;
  uint64_t dstOff;
  size_t size;
  int srcMrIdx; // -1 when not used (e.g. signal-only PutSignal)
  int dstMrIdx;
  uint64_t signalOff;         // PUT_SIGNAL only
  uint64_t signalValue;       // PUT_SIGNAL only
  uint64_t putValue;          // PUT_VALUE only (value embedded in desc)
  void *request;              // filled by progress thread after posting IB op
  uint64_t opSeq;             // per-peer monotonic sequence number
  struct flagcxRmaDesc *next; // intrusive link for inProgressQueues
};

// Intra-node IPC signal state for direct D2D bypass. Symmetric data windows
// resolve peer pointers directly from their own IPC locators.
struct flagcxRmaIpcState {
  int nRanks;
  void **peerSignalBufs; // [nRanks] IPC-mapped signal buffers
  uint64_t *signalSeqs; // [nRanks] per-peer accumulated signal counter (for D2D
                        // signal writes)
};

// Per-comm async RMA proxy state.
// pending queues: producer = caller (proxy kernel thread), consumer = progress
// thread. inProgress queues: progress thread only (no locking needed).
struct flagcxRmaProxyState {
  uint32_t queueSize;                     // power of two
  uint32_t queueMask;                     // queueSize - 1
  struct flagcxRmaDesc **circularBuffers; // [nRanks * queueSize]
  volatile uint32_t *pis;                 // [nRanks] producer index
  volatile uint32_t *cis;                 // [nRanks] consumer index

  pthread_mutex_t *peerProducerMutexes; // [nRanks]
  struct flagcxIntruQueue<struct flagcxRmaDesc, &flagcxRmaDesc::next>
      *inProgressQueues;        // [nRanks]
  volatile uint64_t *opSeqs;    // [nRanks]
  volatile uint64_t *doneSeqs;  // [nRanks]
  volatile uint32_t *inFlights; // [nRanks]

  // GPU-visible done sequence counters for STREAM_OPS mode.
  // GPU stream waits on doneSeqsDev via streamWaitValue64.
  uint64_t *doneSeqsDev; // [nRanks] device pointer (GPU-visible)
  // CPU-side done sequence counters for HOST_FUNC mode.
  // Written by proxy thread, polled by host-func callback.
  volatile uint64_t *doneSeqsCpu; // [nRanks] host memory

  // GPU-visible ready sequence counters for STREAM_OPS mode.
  // GPU stream writes readySeqsDev via streamWriteValue64 to signal data ready.
  uint64_t *readySeqsDev; // [nRanks] device pointer (GPU-visible)
  // CPU-side ready sequence counters for HOST_FUNC mode.
  // Written by host-func callback, polled by proxy thread.
  volatile uint64_t *readySeqsCpu; // [nRanks] host memory

  // Synchronization method: HOST_FUNC (default) or STREAM_OPS (opt-in via env)
  int useStreamOps; // 0 = HOST_FUNC (default), 1 = STREAM_OPS

  // Global completion counter: incremented once for every op that completes.
  // Callers record the value before issuing ops, then poll until it advances.
  volatile uint64_t completionCount;

  // Set to 1 by the progress thread when an IB op fails (test error, post
  // error, or missing sendComm). Wait functions check this and return an error.
  volatile int rmaError;

  void *const *fullSendComms; // [nRanks] or NULL until published
  int nRanks;
  struct flagcxHeteroComm *comm; // back-pointer

  // Intra-node IPC state: per-peer device pointers for D2D bypass
  // NULL if not initialized or if no intra-node peers exist
  struct flagcxRmaIpcState *ipcState;
  bool ipcInitFailed; // true if IpcInit was attempted and failed (prevents
                      // retries)

  // Condition variable for HOST_FUNC done-wait: proxy thread broadcasts
  // after updating doneSeqsCpu so host-func callbacks wake without spinning.
  pthread_mutex_t doneMutex;
  pthread_cond_t doneCond;

  pthread_t thread;
  volatile int stop;
};

typedef struct flagcxHeteroComm *flagcxHeteroComm_t;

flagcxResult_t flagcxHeteroGetVersion(int *version);

/* C++ style */
flagcxResult_t flagcxHeteroSend(const void *sendbuff, size_t count,
                                flagcxDataType_t datatype, int peer,
                                flagcxHeteroComm_t comm, flagcxStream_t stream,
                                int opId = INT_MAX, int step = -1);

/* C++ style */
flagcxResult_t flagcxHeteroRecv(void *recvbuff, size_t count,
                                flagcxDataType_t datatype, int peer,
                                flagcxHeteroComm_t comm, flagcxStream_t stream,
                                int opId = INT_MAX, int step = -1);

flagcxResult_t flagcxHeteroGroupStart();

flagcxResult_t flagcxHeteroGroupEnd();

flagcxResult_t flagcxHeteroGetUniqueId(flagcxUniqueId *out);

flagcxResult_t flagcxHeteroCommInitRank(flagcxHeteroComm_t *newcomm, int nranks,
                                        flagcxUniqueId commId, int myrank);

flagcxResult_t flagcxHeteroCommCount(const flagcxHeteroComm_t comm, int *count);

flagcxResult_t flagcxHeteroCommUserRank(const flagcxHeteroComm_t comm,
                                        int *rank);

flagcxResult_t flagcxHeteroCommDestroy(flagcxHeteroComm_t comm);

flagcxResult_t flagcxHeteroPut(flagcxHeteroComm_t comm, int peer,
                               size_t srcOffset, size_t dstOffset, size_t size,
                               int srcMrIdx, int dstMrIdx,
                               bool streamSyncReady = false,
                               uint64_t *assignedSeq = nullptr);

flagcxResult_t flagcxHeteroBatchPut(flagcxHeteroComm_t comm, int peer,
                                    const size_t *srcOffsets,
                                    const size_t *dstOffsets,
                                    const size_t *sizes, const int *srcMrIdxs,
                                    const int *dstMrIdxs, size_t count);

// RDMA READ: pull data from remote peer's srcMrIdx buffer into local dstMrIdx
// buffer
flagcxResult_t flagcxHeteroGet(flagcxHeteroComm_t comm, int peer,
                               size_t srcOffset, size_t dstOffset, size_t size,
                               int srcMrIdx, int dstMrIdx);

// Data + signal combined (chained WRITE + ATOMIC in IB backend)
// When size == 0, only signal ATOMIC is posted (signal-only mode)
flagcxResult_t flagcxHeteroPutSignal(flagcxHeteroComm_t comm, int peer,
                                     size_t srcOffset, size_t dstOffset,
                                     size_t size, size_t signalOffset,
                                     int srcMrIdx, int dstMrIdx,
                                     uint64_t signalValue,
                                     bool streamSyncReady = false,
                                     uint64_t *assignedSeq = nullptr);

flagcxResult_t flagcxHeteroFlush(flagcxHeteroComm_t comm, void *gpuAddr,
                                 size_t size, void *gHandleInfo);

// Async RMA proxy lifecycle.
flagcxResult_t flagcxHeteroRmaProxyStart(flagcxHeteroComm_t comm);
flagcxResult_t flagcxHeteroRmaProxyStop(flagcxHeteroComm_t comm);

// Publish the stable fullSendComms pointer to the proxy.
flagcxResult_t flagcxHeteroRmaProxyPublishSendComms(flagcxHeteroComm_t comm,
                                                    void *const *fullSendComms);

// Wait until all ops for a specific peer up to seq are complete.
flagcxResult_t flagcxHeteroFlushRma(flagcxHeteroComm_t comm, int peer,
                                    uint64_t seq);

// Stream-based flush: enqueue a GPU-side wait on doneSeqsDev[peer] >= seq.
// Returns immediately; the stream will stall until the condition is met.
flagcxResult_t flagcxHeteroFlushRmaStream(flagcxHeteroComm_t comm, int peer,
                                          uint64_t seq, flagcxStream_t stream);

// Wait until all pending RMA ops for all peers are complete.
flagcxResult_t flagcxHeteroFlushAllRma(flagcxHeteroComm_t comm);

flagcxResult_t flagcxHeteroWaitSignal(flagcxHeteroComm_t comm, int peer,
                                      size_t signalOffset, uint64_t expected,
                                      flagcxStream_t stream);

// Put a 64-bit value to remote peer's buffer at dstOffset.
// Writes value to local staging buffer then does iput from staging MR.
flagcxResult_t flagcxHeteroPutValue(flagcxHeteroComm_t comm, int peer,
                                    uint64_t value, size_t dstOffset,
                                    int dstMrIdx);

// Read the current global completion counter (snapshot before issuing ops).
flagcxResult_t flagcxHeteroReadCounter(flagcxHeteroComm_t comm,
                                       uint64_t *count);

// Wait until the global completion counter reaches target.
// Typical use: before = snapshot, issue N ops, flagcxHeteroWaitCounter(comm,
// before + N).
flagcxResult_t flagcxHeteroWaitCounter(flagcxHeteroComm_t comm,
                                       uint64_t target);

// Stream-based Put (with intra-node D2D bypass).
// If peer is intra-node and IPC state is available, performs direct D2D memcpy
// on the given stream. Otherwise enqueues to the proxy thread (same as
// flagcxHeteroPut). Returns the opSeq for use with FlushRmaStream.
flagcxResult_t flagcxHeteroPutStream(flagcxHeteroComm_t comm, int peer,
                                     size_t srcOffset, size_t dstOffset,
                                     size_t size, int srcMrIdx, int dstMrIdx,
                                     flagcxSymWindow_t srcWindow,
                                     flagcxSymWindow_t dstWindow,
                                     flagcxStream_t stream, uint64_t *opSeq);

// Stream-based PutSignal (with intra-node D2D bypass).
flagcxResult_t flagcxHeteroPutSignalStream(
    flagcxHeteroComm_t comm, int peer, size_t srcOffset, size_t dstOffset,
    size_t size, size_t signalOffset, int srcMrIdx, int dstMrIdx,
    uint64_t signalValue, flagcxSymWindow_t srcWindow,
    flagcxSymWindow_t dstWindow, flagcxStream_t stream, uint64_t *opSeq);

// Initialize IPC state for intra-node D2D bypass.
// Must be called after symmetric windows and signal IPC mappings are created.
// It does not require network one-sided handles. Collective: ALL intra-node
// ranks must call.
flagcxResult_t flagcxHeteroRmaIpcInit(flagcxHeteroComm_t comm);

// Cleanup IPC state.
flagcxResult_t flagcxHeteroRmaIpcDestroy(flagcxHeteroComm_t comm);

#endif
