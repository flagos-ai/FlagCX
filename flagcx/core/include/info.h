/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef FLAGCX_INFO_H_
#define FLAGCX_INFO_H_

#include "device.h"
#include "utils.h"

typedef struct flagcxHeteroComm *flagcxHeteroComm_t;

#define FLAGCX_MAX_LOCAL_RANKS 64

typedef enum : uint8_t {
  flagcxPatternRing,
  flagcxPatternRingTwice,
  flagcxPatternPipelineFrom,
  flagcxPatternPipelineTo,
  flagcxPatternTreeUp,
  flagcxPatternTreeDown,
  flagcxPatternTreeUpDown,
  flagcxPatternCollnetChain,
  flagcxPatternCollnetDirect,
  flagcxPatternNvls,
  flagcxPatternNvlsTree,
  flagcxPatternSend,
  flagcxPatternRecv
} flagcxPattern_t;

// Used to pass FLAGCX call information between functions
struct flagcxInfo {
  flagcxFunc_t coll;
  const char *opName;
  // FLAGCX Coll Args
  const void *sendbuff;
  void *recvbuff;
  size_t count;
  flagcxDataType_t datatype;
  flagcxRedOp_t op;
  int root; // peer for p2p operations
  flagcxHeteroComm_t comm;
  flagcxStream_t stream;
  // Algorithm details
  int chunkSteps;
  int sliceSteps;
  // Computed later
  flagcxDevRedOpFull opFull;
  flagcxPattern_t pattern;
  size_t nBytes;
  size_t aggnBytes;
  size_t workBytes;
  size_t sendbuffSize;
  size_t recvbuffSize;
  int stepSize;
  int chunkCount;
  int chunkSize;
  int channelId;
  int workFuncIndex;
  flagcxRegBufferType regBufType;
  void *regBufSend[FLAGCX_MAX_LOCAL_RANKS];
  void *regBufRecv[FLAGCX_MAX_LOCAL_RANKS];
  // collnet buffer reg handles
  void *sendMhandle;
  void *recvMhandle;
  // Need to initialize
  int nThreads;
  int nChannels;
  int algorithm;
  int protocol;
  bool userTuned;
  struct flagcxInfo *next;
  unsigned long long flagcxFuncTimes;
  uint64_t groupHash;
};

inline flagcxResult_t flagcxInfoSetDerived(struct flagcxInfo *info,
                                           int nRanks) {
  info->nBytes = info->workBytes =
      info->count * getFlagcxDataTypeSize(info->datatype);
  if (info->coll == flagcxFuncAllGather || info->coll == flagcxFuncBroadcast) {
    info->count = info->workBytes;
    info->datatype = flagcxInt8;
  }
  if (info->coll == flagcxFuncAllGather ||
      info->coll == flagcxFuncReduceScatter)
    info->nBytes *= nRanks; // count is per rank

  /* compute buffer size for NVLS buffer registration */
  if (info->coll == flagcxFuncAllGather) {
    info->sendbuffSize = info->workBytes;
    info->recvbuffSize = info->sendbuffSize * nRanks;
  } else if (info->coll == flagcxFuncReduceScatter) {
    info->recvbuffSize = info->workBytes;
    info->sendbuffSize = info->recvbuffSize * nRanks;
  } else {
    info->sendbuffSize = info->recvbuffSize = info->workBytes;
  }
  return flagcxSuccess;
}

struct flagcxTaskColl {
  struct flagcxTaskColl *next;
  flagcxFunc_t func;
  void const *sendbuff;
  void *recvbuff;
  size_t count;
  int root;
  flagcxDataType_t datatype;
  flagcxDevRedOpFull op;
  int chunkSteps, sliceSteps;
  struct flagcxInfo info;
};
struct flagcxTaskP2p {
  flagcxTaskP2p *next;
  void *buff;
  size_t bytes;
  // Stateful chunk index. If a p2p gets "cut" over two plans this keeps track
  // of where it left off.
  int chunk;
  flagcxDataType_t dtype;
  flagcxStream_t stream;
};

struct flagcxCudaStreamList {
  struct flagcxCudaStreamList *next;
  flagcxStream_t stream;
};
struct flagcxTasks {
  struct Peer {
    bool sendSeen, recvSeen;
    struct flagcxIntruQueue<struct flagcxTaskP2p, &flagcxTaskP2p::next>
        sendQueue;
    struct flagcxIntruQueue<struct flagcxTaskP2p, &flagcxTaskP2p::next>
        recvQueue;
  };
  struct flagcxIntruQueue<struct flagcxInfo, &flagcxInfo::next> collQueue;
  // Queue for user-tuned executed collectives
  struct flagcxIntruQueue<struct flagcxInfo, &flagcxInfo::next> collTunedQueue;
  // Queue for continuous bytes distribution (CBD) collectives
  struct flagcxIntruQueue<struct flagcxInfo, &flagcxInfo::next> collCBDQueue;
  // Queue for collnet
  struct flagcxIntruQueue<struct flagcxInfo, &flagcxInfo::next> collnetQueue;
  size_t workBytesTotal;
  int usableChannels;
  bool sorted;
  struct Peer *peers /*[nRanks]*/;
  int *p2pOrder;
  int p2pOrderSteps;
  int nTasksColl, nTasksP2p;

  // The list of user streams aggregated over all tasks present.
  struct flagcxCudaStreamList *streams;
  // The most recent user stream. Ignored if streams==nullptr
  flagcxStream_t streamRecent;
  // The graph capturing all user streams or invalid if none. Thus we restrict
  // the user that all streams must be captured in the same graph or not
  // captured at all. Technically we could probably relax this, but that would
  // mean collecting a different `flagcxTasks` per graph and one for non-graph.
};

#endif
