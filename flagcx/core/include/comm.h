/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef FLAGCX_COMM_H_
#define FLAGCX_COMM_H_

#include "bootstrap.h"
#include "device.h"
#include "flagcx_kernel.h"
#include "flagcx_net.h"
#include "flagcx_tuner.h"
#include "info.h"
#include "register.h"
#include "type.h"
#include <stdint.h>

#define CACHE_LINE_SIZE 128
#define MEM_ALIGN 4096
#define CUDA_IPC_MIN 2097152UL

// Channels / LL tuning
#define FLAGCX_LL_THREAD_THRESHOLD 8
#define FLAGCX_LL128_THREAD_THRESHOLD 8
#define FLAGCX_SIMPLE_THREAD_THRESHOLD 64

struct flagcxSendMem {
  union {
    struct {
      uint64_t head;
      char pad1[CACHE_LINE_SIZE - sizeof(uint64_t)];
      void *ptrExchange;
      uint64_t redOpArgExchange[2];
      char pad2[CACHE_LINE_SIZE - sizeof(void *) - 2 * sizeof(uint64_t)];
      int offsFifo[FLAGCX_STEPS];
    };
    char pad3[MEM_ALIGN];
  };
};

struct flagcxRecvMem {
  union {
    struct {
      uint64_t tail;
      char pad1[CACHE_LINE_SIZE - sizeof(uint64_t)];
      struct flagcxConnFifo connFifo[FLAGCX_STEPS];
      int flush; // For GDRCopy-based flush
    };
    char pad4[MEM_ALIGN];
  };
};

enum helperThreadState { ThreadStart, ThreadStop };

#define FLAGCX_IPC_POOL_SIZE (2 * FLAGCX_MAX_LOCAL_RANKS * FLAGCX_MAX_OPS)

struct flagcxGraphHelperResources {
  flagcxHeteroComm *comm;
  pthread_mutex_t threadLock;
  pthread_cond_t threadCond;
  enum helperThreadState threadState;
  void *ipcBases[FLAGCX_IPC_POOL_SIZE];
  int ipcTail;
  int ipcHead;
};

struct flagcxUserRedOp {
  int freeNext; // -1=allocated, otherwise index of next free entry in array
  flagcxDataType_t datatype;
  flagcxDevRedOpFull opFull;
};

struct flagcxNodeRanks {
  int localRanks;
  int *localRankToRank;
};

struct cliqueInfo {
  int id;
  int size;
  int *ranks;
};

struct flagcxDestructor {
  struct flagcxDestructor *next;
  void *obj;
  flagcxResult_t (*fn)(struct flagcxDestructor *me);
};

struct flagcxCommCallback {
  struct flagcxCommCallback *next;
  flagcxResult_t (*fn)(struct flagcxHeteroComm *comm,
                       struct flagcxCommCallback *cb);
};

struct flagcxSharedResources {
  int refCount;
  struct flagcxHeteroComm *owner; /* comm which creates this shared res. */
  struct flagcxChannelPeer *peers[MAXCHANNELS];
  struct flagcxDevChannelPeer *devPeers[MAXCHANNELS];
  /* P2P operation counter, one per channel */
  uint64_t p2pOpCount[MAXCHANNELS];
  /* Collective operation counter */
  uint64_t collOpCount;
  int tpNRanks;
  int tpNLocalRanks;
  int tpNChannels;
  int tpP2pNChannels;
  int tpP2pChunkSize;
  uint64_t magic;

  // top parent rank to localRank translation table
  int *tpRankToLocalRank;

  /* proxy related shared res */
  struct flagcxProxyState *proxyState;
};

struct flagcxChannel {
  struct flagcxChannelPeer **peers;
  struct flagcxDevChannelPeer **devPeers;
  /* devPeer pointer array used for host side access */
  struct flagcxDevChannelPeer **devPeersHostPtr;
  struct flagcxRing ring;
  int *devRingUserRanks;
  struct flagcxTree tree;

  struct flagcxTree collnetChain;
  struct flagcxDirect collnetDirect;

  struct flagcxNvls nvls;

  int id;                // index of this channel
  uint32_t workFifoSent; // last used work index+1

  /* comm split sharable resources */
  struct flagcxChannelPeer *collnetPeers;
  struct flagcxDevChannelPeer *collnetDevPeers;
  struct flagcxChannelPeer *nvlsPeers;
  struct flagcxDevChannelPeer *nvlsDevPeers;
};

struct flagcxWorkList {
  struct flagcxWorkList *next;
  struct flagcxWork work;
};

struct flagcxPointerList {
  struct flagcxPointerList *next;
  void *ptr;
};

struct flagcxCollnetHandleList {
  struct flagcxCollnetHandleList *next;
  void *collnetHandle;
  size_t size;
  const void *buffer;
  struct flagcxProxyConnector *proxyconn;
};

#define FLAGCX_MAGIC 0x0280028002800280 // Nickel atomic number is 28.

struct flagcxHeteroComm {
  uint64_t startMagic;
  struct flagcxMemoryStack memPermanent, memScoped;
  // List of destructors to run when comm is destructed
  struct flagcxDestructor *destructorHead;

  struct flagcxSharedResources *sharedRes;
  /* map to top parent ranks. */
  int *topParentRanks;
  int *topParentLocalRanks;
  struct flagcxChannel channels[MAXCHANNELS];
  struct flagcxPeerInfo *peerInfo;
  struct flagcxTopoServer *topoServer;
  struct flagcxInterServerTopo *interServerTopo;

  struct flagcxNetAdaptor *netAdaptor;
  struct bootstrapState *bootstrap;
  // Bitmasks for flagcxTransportP2pSetup
  uint64_t *connectSend;
  uint64_t *connectRecv;

  uint64_t magic; // Magic number for all network communication. Not a security
                  // key -- only goal is to detect mismatches.

  uint64_t commHash;
  int rank;                   // my rank in the communicator
  int nRanks;                 // number of GPUs in communicator
  int cudaDev;                // my cuda device index
  int netDev;                 // my net  device index
  int nvmlDev;                // my nvml device index
  int compCap;                // compute capability of the GPU
  int minCompCap, maxCompCap; // min/max compute capability in the communicator
  int64_t busId;              // my PCI bus ID in int format
  cpu_set_t cpuAffinity;      // CPU affinity of the GPU
  int cudaArch;               // matches __CUDA_ARCH__ of device

  int node;
  int nNodes;
  int localRank;
  int localRanks;
  int maxLocalRanks;
  int *rankToNode;
  int *rankToLocalRank;
  int *localRankToRank;
  // localRanks and localRanktoRank for all nodes
  struct flagcxNodeRanks *nodeRanks;
  // MNNVL: Multi-Node NVLink
  int MNNVL;                // true when MNNVL is available
  struct cliqueInfo clique; // Our MNNVL clique information
  int cliqueRank;           // Our rank within the MNNVL clique

  bool checkPointers;
  bool dmaBufSupport;

  // Counter for tracking CUDA launches (P2P and collectives included)
  uint64_t opCount;

  // Channels for collectives
  int nChannels;    // connection nChannels
  int collChannels; // enqueue nChannels
  int nvlsChannels; // enqueue nChannels
  // Channels (per peer) for p2p
  int p2pnChannels;
  int p2pnChannelsPerPeer;
  int p2pChannels[MAXCHANNELS];

  // Should this comm allocate LL buffers for network P2P connections?
  bool allocP2pNetLLBuffers;

  // Buffer sizes
  int buffSizes[FLAGCX_NUM_PROTOCOLS];
  int p2pChunkSize;
  int nvlsChunkSize;

  // Algorithm/Protocols thresholds
  ssize_t threadThresholds[FLAGCX_NUM_ALGORITHMS][FLAGCX_NUM_PROTOCOLS];
  float latencies[FLAGCX_NUM_FUNCTIONS][FLAGCX_NUM_ALGORITHMS]
                 [FLAGCX_NUM_PROTOCOLS];
  float bandwidths[FLAGCX_NUM_FUNCTIONS][FLAGCX_NUM_ALGORITHMS]
                  [FLAGCX_NUM_PROTOCOLS];
  float ringbdw[FLAGCX_NUM_FUNCTIONS][FLAGCX_NUM_PROTOCOLS];
  int maxThreads[FLAGCX_NUM_ALGORITHMS][FLAGCX_NUM_PROTOCOLS];

  /* This attribute can indicate the states of communicators and return code of
   * asynchronous FLAGCX operations. */
  flagcxResult_t asyncResult;

  // Flag to ask FLAGCX kernels to abort
  volatile uint32_t *abortFlag;
  volatile uint32_t *childAbortFlag;
  uint32_t *abortFlagRefCount;

  // Device side of the communicator (for cudaFree's)
  struct flagcxDevComm *devComm; // actually = &flagcxDevCommAndChannels::comm

  // Operation pool.
  int workFifoDepth; // size of workFifoHeap[], power of 2
  struct flagcxWork *workFifoHeap;
  struct flagcxWork *devWorkFifoHeap;
  void *workFifoHeapGdrHandle;

  // Work completion notificaion
  uint32_t *workFifoDone /*[MAXCHANNELS]*/; // in cudaHost memory
  uint32_t
      workFifoSent; // Monotonic (mod 1<<32) index of next unused fifo slot.
  uint32_t workFifoAckdMin; // Monotonic index of least unprocessed fifo slot
                            // over all channels.

  // Intra-process sync
  struct flagcxHeteroComm
      *intraComm0; // leader of intra-process comms (self possible)
  struct flagcxHeteroComm
      *intraNext; // next of intra-process comms, intraComm0 is head
  int intraRank;
  int intraRanks;
  uint32_t intraBarrierPhase;
  char intraPad1[64 - sizeof(uint64_t)];
  uint64_t intraBarrierCounter; // only used if this is intraComm0
  char intraPad2[64 - sizeof(uint64_t)];
  uint64_t intraBarrierGate; // only used if this is intraComm0

  struct flagcxProxyState *proxyState;
  int proxyRefCountOld; /* store proxy post-atomic-sub refcount */
  // Whether this communicator uses collNet
  int collNetSupport;
  bool collNetRegSupport;
  uint8_t collNetSupportMatrix[4 /*sum,prod,min,max*/][flagcxNumTypes];
  int intraHighestTransportType;
  int *collNetHeads;
  int collNetHeadsNum;
  int *collNetDenseToUserRank;
  int *collNetUserToDenseRank;
  /* sharable collNet proxy progress resource. */
  struct flagcxCollNetSharedRes *collNetSharedRes;

  // NVLink SHARP (NVLS) support
  int nvlsSupport;
  int nvlsRegSupport;
  /* sharable NVLS resource. */
  struct flagcxNvlsSharedRes *nvlsResources;

  // pools backed by comm->memPermanent
  struct flagcxMemoryPool memPool_flagcxProxyOp;
  struct flagcxMemoryPool memPool_flagcxKernelPlan;
  struct flagcxMemoryPool memPool_flagcxPointerList;
  struct flagcxMemoryPool memPool_flagcxNvlsHandleList;
  struct flagcxMemoryPool memPool_flagcxCollnetHandleList;
  // Next comm in this thread's active flagcxGroup[Start|End](). Holds "0x1"
  // when this comm is not yet in a group.
  struct flagcxHeteroComm *groupNext;
  // Subset of those in groupNext list. Holds 0x1 if not needing preconnect.
  struct flagcxHeteroComm *preconnectNext;
  int persistentRefs; // number of persistent plan-lists capturing this comm
  struct flagcxTasks tasks;

  // user-created reduction ops
  int userRedOpCapacity, userRedOpFreeHead;
  flagcxUserRedOp *userRedOps;

  // Queue of things for the main thread to do
  struct flagcxIntruQueueMpsc<struct flagcxCommCallback,
                              &flagcxCommCallback::next>
      callbackQueue;

  flagcxConfig_t config;
  // initState is to more conveniently reclaim resources when errors happen.
  flagcxResult_t initState;
  // flag to indicate if flagcxCommFinalize() is called
  bool finalizeCalled;
  // shared structures for finalization
  int finalizeRankCnt;
  // group job to support multi-thread FT
  struct flagcxGroupJob *groupJob;

  // Tuning plugin
  flagcxTuner_t *tuner;
  void *tunerContext;
  // buffer registration cache
  struct flagcxRegCache regCache;
  uint64_t groupHash;
  uint64_t endMagic;
  // Kernel FIFO buffer for device side communication
  void *fifoBuffer;
};

typedef struct flagcxHeteroComm *flagcxHeteroComm_t;

enum flagcxLaunchMode {
  flagcxLaunchModeInvalid = 0,
  flagcxLaunchModeParallel,
  flagcxLaunchModeGroup
};
extern enum flagcxLaunchMode flagcxParamLaunchMode;

void flagcxCommPushFree(struct flagcxHeteroComm *comm, void *buf);
void flagcxCommPushCudaFree(struct flagcxHeteroComm *comm, void *buf);
void flagcxCommPushCudaHostFree(struct flagcxHeteroComm *comm, void *buf);
void flagcxCommPushCudaGdrFree(struct flagcxHeteroComm *comm, void *handle);

inline flagcxResult_t flagcxCommPollCallbacks(struct flagcxHeteroComm *comm,
                                              bool waitSome) {
  flagcxResult_t result = flagcxSuccess;
  struct flagcxCommCallback *cb =
      flagcxIntruQueueMpscDequeueAll(&comm->callbackQueue, waitSome);
  while (cb != nullptr) {
    struct flagcxCommCallback *next = cb->next;
    flagcxResult_t res1 = cb->fn(comm, cb); // may reclaim memory of cb
    if (res1 != flagcxSuccess)
      result = res1;
    cb = next;
  }
  FLAGCXCHECK(result);
  return flagcxSuccess;
}

inline void flagcxCommIntraBarrierIn(struct flagcxHeteroComm *comm,
                                     uint32_t x) {
  int phase = comm->intraBarrierPhase;
  if (comm->intraRanks == 1) {
    // Release everyone (just me).
    comm->intraBarrierGate = (uint64_t(x) << 32) | (phase ^ 1);
  } else {
    struct flagcxHeteroComm *comm0 = comm->intraComm0;
    uint64_t count = __atomic_add_fetch(
        &comm0->intraBarrierCounter, (uint64_t(x) << 32) + 1, __ATOMIC_RELEASE);
    if (uint32_t(count) == uint32_t(comm->intraRanks)) {
      // Reset.
      __atomic_store_n(&comm0->intraBarrierCounter, 0, __ATOMIC_RELAXED);
      // Release everyone.
      __atomic_store_n(&comm0->intraBarrierGate,
                       (count >> 32 << 32) | (phase ^ 1), __ATOMIC_RELEASE);
    }
  }
}

// returns sum of x values contributed to flagcxCommIntraBarrierIn(comm, x)
inline uint32_t flagcxCommIntraBarrierOut(struct flagcxHeteroComm *comm) {
  struct flagcxHeteroComm *comm0 = comm->intraComm0;
  comm->intraBarrierPhase ^= 1;
  uint32_t phase = comm->intraBarrierPhase;
  uint64_t gate = __atomic_load_n(&comm0->intraBarrierGate, __ATOMIC_RELAXED);
  if ((gate & 1) != phase) {
    uint64_t t0 = clockNano();
    do {
      // Spin vigorously for first 5us.
      if (clockNano() - t0 >= 5 * 1000)
        sched_yield();
      gate = __atomic_load_n(&comm0->intraBarrierGate, __ATOMIC_RELAXED);
    } while ((gate & 1) != phase);
  }
  if (comm->intraRanks != 1)
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
  return gate >> 32;
}

// Scrambles the bits of non-builtin values of flagcxRedOp_t according to the
// communicator memory address. Used to catch bugs so that integer handles
// associated with this communicator won't collide with handles of other
// communicatrs. This function is its own inverse.
static inline flagcxRedOp_t flagcxUserRedOpMangle(flagcxHeteroComm *comm,
                                                  flagcxRedOp_t op) {
  // Preserve the built-in values.
  if (int(op) < int(flagcxNumRedOps))
    return op;
  uint64_t h = reinterpret_cast<uint64_t>(comm);
  h ^= h >> 32;
  h *= 0x9e3779b97f4a7c13u; // Knuth's 64-bit magical hash constant
  h >>= 32; // h is now an excellent 32-bit hash of the comm pointer
  h &= int(flagcxMaxRedOp); // flagcxMaxRedOp is a power of 2 minus 1
  int op1 = int(h) ^ int(op);
  // Since builtin values are preserved, we also have to preserve their
  // preimage.
  return op1 < int(flagcxNumRedOps) ? op : flagcxRedOp_t(op1);
}

flagcxResult_t flagcxCommEnsureReady(flagcxHeteroComm_t comm);
flagcxResult_t flagcxCommSetAsyncError(flagcxHeteroComm_t comm,
                                       flagcxResult_t nextState);

#endif
