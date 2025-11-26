/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef FLAGCX_DEVICE_H_
#define FLAGCX_DEVICE_H_

#include "align.h"
#include "flagcx_common.h"
#include "net_device.h"
#include "type.h"
#include <stdint.h>

extern const char *flagcxFuncStr[FLAGCX_NUM_FUNCTIONS];

extern const char *flagcxAlgoStr[FLAGCX_NUM_ALGORITHMS];

extern const char *flagcxProtoStr[FLAGCX_NUM_PROTOCOLS];

#define FLAGCX_MAX_OPS 2048
#define FLAGCX_STEPS 8

enum flagcxDevRedOp_t {
  flagcxDevSum,
  flagcxDevProd,
  flagcxDevMinMax,
  flagcxDevPreMulSum,
  flagcxDevSumPostDiv,
  flagcxNumDevRedOps
};
struct flagcxDevRedOpFull {
  flagcxDevRedOp_t op;
  flagcxRedOp_t proxyOp;
  bool scalarArgIsPtr;
  uint64_t scalarArg;
};

union flagcxLLFifoLine {
  /* Flags have to be *after* data, because otherwise, an incomplete receive
     from the network may receive the flag but not the data.
     Note this is assuming that either we receive contiguous chunks of data
     (sockets) or data is written with an atomicity of 8 bytes (IB/RDMA). */
  struct {
    uint32_t data1;
    uint32_t flag1;
    uint32_t data2;
    uint32_t flag2;
  };
  uint64_t v[2];
};

#define WARP_SIZE 32
#define MAXCHANNELS 32
#define FLAGCX_MAX_NTHREADS 640
#define FLAGCX_SIMPLE_MAX_NTHREADS 512
#define FLAGCX_LL_MAX_NTHREADS 512
#define FLAGCX_LL_LINES_PER_THREAD 8
#ifdef TEST_LL_CLEANUP
#define FLAGCX_LL_CLEAN_MASK 0x078 // Set to 0x100 to disable cleanup
#define FLAGCX_LL_FLAG_MAX 0x100
#define FLAGCX_LL_FLAG(a) ((uint32_t)((a) % FLAGCX_LL_FLAG_MAX))
#else
#define FLAGCX_LL_CLEAN_MASK 0x7ffffff8
#define FLAGCX_LL_FLAG(a) ((uint32_t)(a))
#endif
// Make sure the clean mask will last for at least FLAGCX_NSTEPS
static_assert(FLAGCX_LL_CLEAN_MASK % FLAGCX_STEPS == 0,
              "Invalid FLAGCX_LL_CLEAN_MASK value");

#define FLAGCX_LL128_LINESIZE 128
#define FLAGCX_LL128_LINEELEMS (FLAGCX_LL128_LINESIZE / sizeof(uint64_t))
#define FLAGCX_LL128_DATAELEMS (FLAGCX_LL128_LINEELEMS - 1)

#define FLAGCX_LL128_MAX_NTHREADS 640
#define FLAGCX_LL128_ELEMS_PER_THREAD 120

#define FLAGCX_LL128_SHMEM_ELEMS_PER_THREAD 8
#define FLAGCX_LL128_SHMEM_SIZE                                                \
  (FLAGCX_LL128_SHMEM_ELEMS_PER_THREAD * FLAGCX_LL128_MAX_NTHREADS)

#define FLAGCX_DIRECT_WRITE 0x01
#define FLAGCX_DIRECT_READ 0x02
#define FLAGCX_DIRECT_NIC 0x04
#define FLAGCX_IPC_WRITE 0x08
#define FLAGCX_IPC_READ 0x10
#define FLAGCX_NVLS_MIN_POLL 0x20

#define FLAGCX_MAX_COLLNET_SIZE (1L << 29)

enum flagcxRegBufferType {
  FLAGCX_REGULAR_BUFFER = 0,
  FLAGCX_IPC_REG_BUFFER = 1,
  FLAGCX_NVLS_REG_BUFFER = 2,
  FLAGCX_COLLNET_REG_BUFFER = 3
};

struct flagcxConnInfo {
  // Regular comm mechanism
  char *buffs[FLAGCX_NUM_PROTOCOLS]; // Local for recv, remote for send
  void *mhandles[FLAGCX_NUM_PROTOCOLS];
  uint64_t *tail; // Local for recv, remote for send
  uint64_t *head; // Local for send, remote for recv

  int flags;                  // Direct communication / other flags
  int shared;                 // Buffers are shared
  int stepSize;               // Step size for the SIMPLE buffer
  void **ptrExchange;         // Pointer exchange for direct communication
  uint64_t *redOpArgExchange; // PreOp scaler exchange for direct pull case

  struct flagcxConnFifo *connFifo; // Used for GPU - Proxy communication

  uint64_t step; // Keep where we are
  uint64_t llLastCleaning;
  flagcxNetDeviceHandle_t netDeviceHandle;
};

struct flagcxProxyConnector {
  int tpRank;
  int tpLocalRank;
  int sameProcess;
  struct flagcxProxyConnection *connection;
  flagcxResult_t (*proxyProgress)(
      struct flagcxProxyState *proxyState,
      struct flagcxProxyArgs *); // Copied from transport if necessary
};

struct flagcxConnector {
  int connected;
  struct flagcxProxyConnector proxyConn;
  struct flagcxTransportComm *transportComm;
  void *transportResources;
  struct flagcxConnInfo conn;
};

struct flagcxRing {
  // Shortcuts for userRanks[1] and userRanks[n-1]
  int prev;
  int next;

  // Maps an internal flagcx index to user-specified rank order. This is
  // necessary since we need to know how the user expects data to be ordered
  // across devices. Ordered from current device.
  int *userRanks;

  int index; // This rank's index in the ring
};

// The root of each tree only has one node down (+1 intra-node).
#define FLAGCX_MAX_TREE_ARITY_TOP 2
// Nodes inside the binary tree can have to two nodes down (+1 intra-node).
#define FLAGCX_MAX_TREE_ARITY 3
struct flagcxTree {
  int depth;
  int up;
  int down[FLAGCX_MAX_TREE_ARITY];
};

#define FLAGCX_MAX_DIRECT_ARITY 7
struct flagcxDirect {
  int depth;
  int out;
  int nHeads; // Number of parallel N<->1<->net operations we'll do in parallel;
              // size of up/down
  int headRank; // Index in 0..nHeads-1 I am the head rank of. -1 if I'm not a
                // head rank (no local NIC)
  int shift; // Shuffling of send/recv for scatter/gather operations, basically
             // localRank%nHeads
  // The heads[...] are guaranteed to be in rotated order start with self:
  //   headRank, (headRank+1)%nHeads, (headRank+2)%nHeads, ...
  int heads[FLAGCX_MAX_DIRECT_ARITY + 1];
  int up[FLAGCX_MAX_DIRECT_ARITY];
  int down[FLAGCX_MAX_DIRECT_ARITY];
};

#define FLAGCX_MAX_NVLS_ARITY 32
#define FLAGCX_MAX_NVLS_TREE_ARITY 3
struct flagcxNvls {
  int out;
  int nHeads; // Number of parallel N<->1<->net operations we'll do in parallel;
              // size of up/down
  int headRank; // Index in 0..nHeads-1 I am the head rank of. -1 if I'm not a
                // head rank (no local NIC)
  int up[FLAGCX_MAX_NVLS_ARITY];
  int down;
  int treeUp;
  int treeDown[FLAGCX_MAX_NVLS_TREE_ARITY];
  int node;
  int nNodes;
};

#if __CUDA_ARCH__ >= 900
#define FLAGCX_MAX_ARITY FLAGCX_MAX_NVLS_ARITY
#else
#define FLAGCX_MAX_ARITY FLAGCX_MAX_DIRECT_ARITY
#endif

#define FLAGCX_MAX_CONNS 4
struct flagcxChannelPeer {
  struct flagcxConnector send[FLAGCX_MAX_CONNS];
  struct flagcxConnector recv[FLAGCX_MAX_CONNS];
  int refCount;
};

struct flagcxDevComm;

/* flagcxWork is to be a power of two, currently 8x64 bytes, */
/* to make sure reads to host from the CUDA kernel are aligned. */
/* Make sure to adjust padding at the end of flagcxWorkElem. */
#define FLAGCX_WORK_SIZE 512

enum flagcxWorkType : uint8_t {
  flagcxWorkTypeUnused = 0,
  flagcxWorkTypeColl = 1,
  flagcxWorkTypeP2p = 2,
  flagcxWorkTypeRegColl = 3
};
enum flagcxWorkP2PType : uint8_t {
  flagcxWorkP2pTypeUnused = 0,
  flagcxWorkP2pTypeSend,
  flagcxWorkP2pTypeRecv
};

struct flagcxWorkHeader {
  union {
    int32_t workNext;  // when isLast=0: Offset from kernel argument workHead
    uint32_t doneAcks; // when isLast=1: Monotonic (mod 1<<32) ack value to send
                       // back.
  };
  uint16_t funcIndex;
  uint8_t isLast : 1; // last work for this kernel
  uint8_t inFifo : 1; // is this work in the fifo
  enum flagcxWorkType type;
};

struct flagcxWorkElem {
  union {
    uint8_t flagBits;
    struct {
      uint8_t isUsed : 1, redOpArgIsPtr : 1, oneNode : 1;
    };
  };
  uint8_t regUsed;
  uint8_t nWarps;
  uint8_t direct;
  uint32_t root;
  const void *sendbuff;
  void *recvbuff;

  size_t count;
  uint64_t redOpArg;
  uint64_t chunkCount : 25, workCount : 39;
  union {
    struct {
      uint64_t lastChunkCount : 25;
      uint64_t workOffset : 39;
    };
    struct {
      uint64_t bid : 32;
      uint64_t nChannels : 32;
    };
  };
};

#define FLAGCX_MAX_WORK_ELEMENTS                                               \
  ((FLAGCX_WORK_SIZE -                                                         \
    alignUp(sizeof(flagcxWorkHeader), alignof(flagcxWorkElem))) /              \
   sizeof(flagcxWorkElem))
static_assert(FLAGCX_MAX_WORK_ELEMENTS == 9,
              "Sanity check: FLAGCX_MAX_WORK_ELEMENTS == 9");

struct flagcxWorkElemP2p {
  int peer : 30;
  int proto : 2;

  enum flagcxWorkP2PType p2pType;
  uint8_t reg : 1;
  uint8_t nWarps : 5;
  uint8_t warpStart;
  uint8_t ngroups;
  // Important not to use any fields with greater than 4-byte alignment since
  // we need sizeof(flagcxWorkElemP2p)==28, but that would be padded up to 32 if
  // there were 8-byte fields.
  // void* buff;
  uint32_t buffHi32, buffLo32; // buff = buffHi32<<32 | buffLo32;
  // size_t count;
  uint32_t countHi32, countLo32; // count = countHi32<<32 | countLo32;
  int chunkSize;
};

static_assert(((FLAGCX_WORK_SIZE -
                alignUp(sizeof(flagcxWorkHeader), alignof(flagcxWorkElemP2p))) /
               sizeof(flagcxWorkElemP2p)) >= 16,
              "Sanity check: FLAGCX_MAX_WORK_ELEMENTS_P2P == 16");
#define FLAGCX_MAX_WORK_ELEMENTS_P2P 16

struct flagcxWorkElemReg {
  struct flagcxWorkElem elem;
  void *dnInputs[FLAGCX_MAX_DIRECT_ARITY + 1];
  void *dnOutputs[FLAGCX_MAX_DIRECT_ARITY + 1];
  void *upOutputs[FLAGCX_MAX_DIRECT_ARITY + 1];
};

#define FLAGCX_MAX_WORK_ELEMENTS_REG                                           \
  ((FLAGCX_WORK_SIZE -                                                         \
    alignUp(sizeof(flagcxWorkHeader), alignof(flagcxWorkElemReg))) /           \
   sizeof(flagcxWorkElemReg))
static_assert(FLAGCX_MAX_WORK_ELEMENTS_REG == 2,
              "Sanity check: FLAGCX_MAX_WORK_ELEMENTS_REG == 2");

// Number of named barriers supported by CUDA
#define FLAGCX_MAX_GROUPS 16

struct flagcxWork {
  struct flagcxWorkHeader header;
  union {
    char pad[FLAGCX_WORK_SIZE - sizeof(struct flagcxWorkHeader)];
    struct flagcxWorkElem elems[FLAGCX_MAX_WORK_ELEMENTS];
    struct flagcxWorkElemP2p p2pElems[FLAGCX_MAX_WORK_ELEMENTS_P2P];
    struct flagcxWorkElemReg regElems[FLAGCX_MAX_WORK_ELEMENTS_REG];
  };
};
static_assert(sizeof(struct flagcxWork) == FLAGCX_WORK_SIZE,
              "Sanity check: sizeof(struct flagcxWork) == FLAGCX_WORK_SIZE");
static_assert(sizeof(struct flagcxWork) % 16 == 0,
              "Sanity check: sizeof(struct flagcxWork)%16 == 0");

struct flagcxDevChannelPeer {
  // Stripped version of flagcxChannelPeer where we only keep the flagcxConnInfo
  // instead of the full flagcxConnector.
  struct flagcxConnInfo send[FLAGCX_MAX_CONNS];
  struct flagcxConnInfo recv[FLAGCX_MAX_CONNS];
};

struct alignas(16) flagcxDevChannel {
  struct flagcxDevChannelPeer **peers;
  struct flagcxRing ring;
  struct flagcxTree tree;
  struct flagcxTree collnetChain;
  struct flagcxDirect collnetDirect;
  struct flagcxNvls nvls;
  uint32_t *workFifoDone; // Location of done counter, device writes index+1 of
                          // last work processed
};

struct flagcxDevComm {
  int rank;
  int nRanks;
  int node;
  int nNodes;
  int buffSizes[FLAGCX_NUM_PROTOCOLS];
  int p2pChunkSize;

  // Operation list for aggregation
  int workFifoDepth;
  struct flagcxWork *workFifoHeap; // may be cudaHost or GDR memory

  int *collNetDenseToUserRank;

  // Flag to ask FLAGCX kernels to abort
  volatile uint32_t *abortFlag;

  // Channels, device side
  struct flagcxDevChannel *channels /*[MAXCHANNELS]*/;
};

struct alignas(16) flagcxDevCommAndChannels {
  struct flagcxDevComm comm;
  struct flagcxDevChannel channels[MAXCHANNELS];
};

#ifdef __CUDA_ARCH__
#define FLAGCX_CUDA_ARCH __CUDA_ARCH__
#else
#define FLAGCX_CUDA_ARCH 0
#endif

#endif
