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

// ============================================================================
// Bootstrap State
// ============================================================================

enum flagcxBootstrapMode {
  FLAGCX_BOOTSTRAP_COLL = 0, // Collective mode (ring topology, N ranks)
  FLAGCX_BOOTSTRAP_P2P = 1   // P2P mode (direct socket, 1:1 RPC)
};

struct bootstrapCollState {
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
  char *bootstrapNetIfName;
  void *properties;
};

// Keep old name as typedef for source compatibility during transition
typedef struct bootstrapCollState bootstrapState;

struct bootstrapP2pState {
  bool isListener;          // true = listen mode, false = connected mode
  struct flagcxSocket sock; // listen socket OR connected peer socket
  union flagcxSocketAddress localAddr;
  uint64_t magic;
  volatile uint32_t *abortFlag;
};

struct flagcxBootstrapState {
  enum flagcxBootstrapMode mode;
  union {
    struct bootstrapCollState *coll;
    struct bootstrapP2pState *p2p;
  };
};

// ============================================================================
// Unified API (dispatches based on state->mode)
// ============================================================================

// Send data to peer. In coll mode, uses ring relay. In P2P mode, sends
// directly over the connected socket (peer param ignored).
flagcxResult_t bootstrapSend(struct flagcxBootstrapState *state, int peer,
                             int tag, void *data, int size);

// Receive data from peer. In coll mode, accepts from ring. In P2P mode,
// receives directly from the connected socket (peer param ignored).
flagcxResult_t bootstrapRecv(struct flagcxBootstrapState *state, int peer,
                             int tag, void *data, int size);

// Deadlock-free bidirectional exchange. In coll mode, lower rank sends first.
// In P2P mode, uses the same ordering based on a "who initiated" convention.
flagcxResult_t bootstrapExchange(struct flagcxBootstrapState *state, int peer,
                                 int tag, const void *sendData, int sendSize,
                                 void *recvData, int recvSize);

// Unified close. Dispatches on state->mode to free the appropriate resources.
flagcxResult_t bootstrapClose(struct flagcxBootstrapState *state);

// ============================================================================
// Collective Mode API
// ============================================================================

// Shared network init — discovers local NIC for socket binding.
// Idempotent, safe to call multiple times. Both Coll and P2P depend on this.
flagcxResult_t bootstrapNetInit();

flagcxResult_t bootstrapCollCreateRoot(struct flagcxBootstrapHandle *handle,
                                       bool idFromEnv);
flagcxResult_t bootstrapGetUniqueId(struct flagcxBootstrapHandle *handle);

// Initialize collective bootstrap (ring topology, N ranks)
flagcxResult_t bootstrapCollInit(struct flagcxBootstrapHandle *handle, int rank,
                                 int nranks, uint64_t magic,
                                 volatile uint32_t *abortFlag,
                                 struct flagcxBootstrapState **state);

// Collective operations (coll mode only)
flagcxResult_t bootstrapCollAllGather(struct flagcxBootstrapState *state,
                                      void *allData, int size);
flagcxResult_t bootstrapCollBarrier(struct flagcxBootstrapState *state,
                                    int rank, int nranks, int tag);
flagcxResult_t bootstrapCollBroadcast(struct flagcxBootstrapState *state,
                                      int rank, int nranks, int root,
                                      void *bcastData, int size);
flagcxResult_t bootstrapCollIntraNodeBarrier(struct flagcxBootstrapState *state,
                                             int *ranks, int rank, int nranks,
                                             int tag);
flagcxResult_t
bootstrapCollIntraNodeBroadcast(struct flagcxBootstrapState *state, int *ranks,
                                int rank, int nranks, int root, void *bcastData,
                                int size);
flagcxResult_t bootstrapCollAbort(struct flagcxBootstrapState *state);

// ============================================================================
// P2P Mode API (RPC-style listen/connect/accept)
// ============================================================================

// Open a listen socket. Ensures bootstrapNetInit() has been called.
// Returns handle containing address for peer to connect.
flagcxResult_t bootstrapP2pListen(uint64_t magic, volatile uint32_t *abortFlag,
                                  void *listenHandle,
                                  struct flagcxBootstrapState **state);

// Connect to a peer's listen socket using handle received out-of-band.
flagcxResult_t bootstrapP2pConnect(void *peerHandle, uint64_t magic,
                                   volatile uint32_t *abortFlag,
                                   struct flagcxBootstrapState **state);

// Accept an incoming connection on a listen state. Returns a new connected
// state.
flagcxResult_t bootstrapP2pAccept(struct flagcxBootstrapState *listenState,
                                  struct flagcxBootstrapState **connState);

// ============================================================================
// Typed collective communication operators (coll mode only)
// ============================================================================

/*
 * Broadcast
 */
flagcxResult_t BroadcastBootstrap(struct flagcxBootstrapState *state,
                                  const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  int root);

/*
 * Gather
 */
flagcxResult_t GatherBootstrap(struct flagcxBootstrapState *state,
                               const void *sendbuff, void *recvbuff,
                               size_t count, flagcxDataType_t datatype,
                               int root);

/*
 * Scatter
 */
flagcxResult_t ScatterBootstrap(struct flagcxBootstrapState *state,
                                const void *sendbuff, void *recvbuff,
                                size_t count, flagcxDataType_t datatype,
                                int root);

/*
 * Reduce
 */
flagcxResult_t ReduceBootstrap(struct flagcxBootstrapState *state,
                               const void *sendbuff, void *recvbuff,
                               size_t count, flagcxDataType_t datatype,
                               flagcxRedOp_t op, int root);

/*
 * All-reduce
 */
flagcxResult_t AllReduceBootstrap(struct flagcxBootstrapState *state,
                                  const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  flagcxRedOp_t op);

/*
 * All-gather
 */
flagcxResult_t AllGatherBootstrap(struct flagcxBootstrapState *state,
                                  const void *sendbuff, void *recvbuff,
                                  size_t sendcount, flagcxDataType_t datatype);

/*
 * Reduce-scatter
 */
flagcxResult_t ReduceScatterBootstrap(struct flagcxBootstrapState *state,
                                      const void *sendbuff, void *recvbuff,
                                      size_t recvcount,
                                      flagcxDataType_t datatype,
                                      flagcxRedOp_t op);

/*
 * All-to-all
 */
flagcxResult_t AlltoAllBootstrap(struct flagcxBootstrapState *state,
                                 const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype);

/*
 * All-to-all with variable block sizes
 */
flagcxResult_t AlltoAllvBootstrap(struct flagcxBootstrapState *state,
                                  const void *sendbuff, size_t *sendcounts,
                                  size_t *sdispls, void *recvbuff,
                                  size_t *recvcounts, size_t *rdispls,
                                  flagcxDataType_t datatype);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif
