/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef FLAGCX_BOOTSTRAP_H_
#define FLAGCX_BOOTSTRAP_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "flagcx.h"
#include "socket.h"

struct flagcxBootstrapHandle {
  uint64_t magic;
  union flagcxSocketAddress addr;
};
static_assert(sizeof(struct flagcxBootstrapHandle) <= sizeof(flagcxUniqueId),
              "Bootstrap handle is too large to fit inside FLAGCX unique ID");

struct bootstrapState {
  struct flagcxSocket listenSock;
  struct flagcxSocket ringRecvSocket;
  struct flagcxSocket ringSendSocket;
  union flagcxSocketAddress *peerCommAddresses;
  union flagcxSocketAddress *peerProxyAddresses;
  struct unexConn *unexpectedConnections;
  int rank;
  int nranks;
  uint64_t magic;
  volatile uint32_t *abortFlag;
};

flagcxResult_t bootstrapNetInit();
flagcxResult_t bootstrapCreateRoot(struct flagcxBootstrapHandle *handle,
                                   bool idFromEnv);
flagcxResult_t bootstrapGetUniqueId(struct flagcxBootstrapHandle *handle);
flagcxResult_t bootstrapInit(struct flagcxBootstrapHandle *handle,
                             void *commState);
flagcxResult_t bootstrapAllGather(void *commState, void *allData, int size);

flagcxResult_t bootstrapSend(void *commState, int peer, int tag, void *data,
                             int size);
flagcxResult_t bootstrapRecv(void *commState, int peer, int tag, void *data,
                             int size);
flagcxResult_t bootstrapBarrier(void *commState, int rank, int nranks, int tag);
flagcxResult_t bootstrapBroadcast(void *commState, int rank, int nranks,
                                  int root, void *bcastData, int size);
flagcxResult_t bootstrapIntraNodeBarrier(void *commState, int *ranks, int rank,
                                         int nranks, int tag);
flagcxResult_t bootstrapIntraNodeBroadcast(void *commState, int *ranks,
                                           int rank, int nranks, int root,
                                           void *bcastData, int size);
flagcxResult_t bootstrapClose(void *commState);
flagcxResult_t bootstrapAbort(void *commState);

/* A bunch of collective communication operators */
/*
 * Broadcast
 *
 * Root device send sendcount values from other GPUs into recvbuff,
 * receiving data from rank i at offset i*sendcount.
 * Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
 * should have a size of at least nranks*sendcount elements.
 *
 * In-place operations will happen if sendbuff == recvbuff.
 */
flagcxResult_t BroadcastBootstrap(void *commState, const void *sendbuff,
                                  void *recvbuff, size_t sendcount,
                                  flagcxDataType_t datatype, int root);

/* A bunch of collective communication operators */
/*
 * Gather
 *
 * Each rank sends sendcount values from its sendbuff to the root rank.
 * Root rank receives data from rank i at offset i*sendcount in recvbuff.
 * Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
 * should have a size of at least nranks*sendcount elements.
 *
 * In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
 */
flagcxResult_t GatherBootstrap(void *commState, const void *sendbuff,
                               void *recvbuff, size_t count,
                               flagcxDataType_t datatype, int root);

/* A bunch of collective communication operators */
/*
 * Scatter
 *
 * Root rank sends sendcount values to each rank, with data for rank i
 * starting at offset i*sendcount in sendbuff.
 * Each rank receives sendcount values into its recvbuff.
 * Assumes sendcount is equal to recvcount for each rank.
 *
 * In-place operations will happen if recvbuff = sendbuff + rank * sendcount.
 */
flagcxResult_t ScatterBootstrap(void *commState, const void *sendbuff,
                                void *recvbuff, size_t count,
                                flagcxDataType_t datatype, int root);
/* A bunch of collective communication operators */
/*
 * All-Gather
 *
 * Each device gathers sendcount values from other GPUs into recvbuff,
 * receiving data from rank i at offset i*sendcount.
 * Assumes recvcount is equal to nranks*sendcount, which means that recvbuff
 * should have a size of at least nranks*sendcount elements.
 *
 * In-place operations will happen if sendbuff == recvbuff + rank * sendcount.
 */
flagcxResult_t AllGatherBootstrap(void *commState, const void *sendbuff,
                                  void *recvbuff, size_t sendcount,
                                  flagcxDataType_t datatype);
/*
 * All-Reduce
 *
 * Reduces data arrays of length count(NOT bytes size) in sendbuff using op
 * operation, and leaves identical copies of result on each recvbuff.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
flagcxResult_t AllReduceBootstrap(void *commState, const void *sendbuff,
                                  void *recvbuff, size_t count,
                                  flagcxDataType_t datatype, flagcxRedOp_t op);
/*
 * Reduce
 *
 * Reduces data arrays of length count(NOT bytes size) in sendbuff using op
 * operation, and leaves identical copies of result on root recvbuff.
 *
 * In-place operation will happen if sendbuff == recvbuff.
 */
flagcxResult_t ReduceBootstrap(void *commState, const void *sendbuff,
                               void *recvbuff, size_t count,
                               flagcxDataType_t datatype, flagcxRedOp_t op,
                               int root);
/*
 * Reduce-Scatter
 *
 * Reduces data in sendbuff using op operation and leaves reduced result
 * scattered over the devices so that recvbuff on rank i will contain the i-th
 * block of the result.
 * Assumes sendcount is equal to nranks*recvcount, which means that sendbuff
 * should have a size of at least nranks*recvcount elements.
 *
 * In-place operations will happen if recvbuff == sendbuff + rank * recvcount.
 */
flagcxResult_t ReduceScatterBootstrap(void *commState, const void *sendbuff,
                                      void *recvbuff, size_t recvcount,
                                      flagcxDataType_t datatype,
                                      flagcxRedOp_t op);

/*
 * All-to-all
 *
 * Every rank sends j-th block of its own sendbuff to the j-th rank of the
 * communicator. Meanwhile, every rank receives j-th block of its own recvbuff
 * from j-th rank.
 *
 * Every block has the size of count elements.
 *
 * In-place operations will happen if sendbuff == recvbuff.
 */
flagcxResult_t AlltoAllBootstrap(void *commState, const void *sendbuff,
                                 void *recvbuff, size_t count,
                                 flagcxDataType_t datatype);

/*
 * All-to-all with variable block sizes
 *
 * Every rank sends j-th block of its own sendbuff to the j-th rank of the
 * communicator. Meanwhile, every rank receives j-th block of its own recvbuff
 * from j-th rank.
 *
 * Each block can have different sizes:
 * - sendcounts[j] specifies the number of elements to send to rank j
 * - sdispls[j] specifies the offset in sendbuff for the j-th block
 * - recvcounts[j] specifies the number of elements to receive from rank j
 * - rdispls[j] specifies the offset in recvbuff for the j-th block
 *
 * In-place operations will happen if sendbuff == recvbuff.
 */
flagcxResult_t AlltoAllvBootstrap(void *commState, const void *sendbuff,
                                  size_t *sendcounts, size_t *sdispls,
                                  void *recvbuff, size_t *recvcounts,
                                  size_t *rdispls, flagcxDataType_t datatype);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif
