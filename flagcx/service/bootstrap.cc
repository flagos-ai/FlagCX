/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE-NCCL.txt for license information
 ************************************************************************/

#include "bootstrap.h"
#include "alloc.h"
#include "check.h"
#include "comm.h"
#include "debug.h"
#include "param.h"
#include "utils.h"
#include <sys/types.h>
#include <unistd.h>
#include <vector>

// Internal tags for typed collective operations (must be unique per operation)
#define BOOTSTRAP_TAG_REDUCE (-9993)
#define BOOTSTRAP_TAG_BROADCAST (-9994)
#define BOOTSTRAP_TAG_ALLTOALL (-9995)
#define BOOTSTRAP_TAG_GATHER (-9996)
#define BOOTSTRAP_TAG_SCATTER (-9997)
#define BOOTSTRAP_TAG_ALLTOALLV (-9998)

struct bootstrapRootArgs {
  struct flagcxSocket *listenSock;
  uint64_t magic;
};

/* Init functions */
static char bootstrapNetIfName[MAX_IF_NAME_SIZE + 1];
union flagcxSocketAddress bootstrapNetIfAddr;
static flagcxNetProperties_t bootstrapNetProperties;
static int bootstrapNetInitDone = 0;
pthread_mutex_t bootstrapNetLock = PTHREAD_MUTEX_INITIALIZER;

flagcxResult_t bootstrapNetInit() {
  if (bootstrapNetInitDone == 0) {
    pthread_mutex_lock(&bootstrapNetLock);
    if (bootstrapNetInitDone == 0) {
      const char *env = flagcxGetEnv("FLAGCX_COMM_ID");
      if (env) {
        union flagcxSocketAddress remoteAddr;
        if (flagcxSocketGetAddrFromString(&remoteAddr, env) != flagcxSuccess) {
          WARN("Invalid FLAGCX_COMM_ID, please use format: <ipv4>:<port> or "
               "[<ipv6>]:<port> or <hostname>:<port>");
          pthread_mutex_unlock(&bootstrapNetLock);
          return flagcxInvalidArgument;
        }
        if (flagcxFindInterfaceMatchSubnet(bootstrapNetIfName,
                                           &bootstrapNetIfAddr, &remoteAddr,
                                           MAX_IF_NAME_SIZE, 1) <= 0) {
          WARN("NET/Socket : No usable listening interface found");
          pthread_mutex_unlock(&bootstrapNetLock);
          return flagcxSystemError;
        }
      } else {
        int nIfs = flagcxFindInterfaces(bootstrapNetIfName, &bootstrapNetIfAddr,
                                        MAX_IF_NAME_SIZE, 1);
        if (nIfs <= 0) {
          WARN("Bootstrap : no socket interface found");
          pthread_mutex_unlock(&bootstrapNetLock);
          return flagcxInternalError;
        }
      }
      char line[SOCKET_NAME_MAXLEN + MAX_IF_NAME_SIZE + 2];
      sprintf(line, " %s:", bootstrapNetIfName);
      flagcxSocketToString(&bootstrapNetIfAddr, line + strlen(line));
      INFO(FLAGCX_NET, "Bootstrap : Using%s", line);
      bootstrapNetInitDone = 1;
    }
    pthread_mutex_unlock(&bootstrapNetLock);
  }
  return flagcxSuccess;
}

// ============================================================================
// Accessor APIs
// ============================================================================

int bootstrapGetRank(struct bootstrapState *state) {
  if (state == NULL || state->mode != FLAGCX_BOOTSTRAP_COLL ||
      state->coll == NULL) {
    return -1;
  }
  return state->coll->rank;
}

int bootstrapGetNranks(struct bootstrapState *state) {
  if (state == NULL || state->mode != FLAGCX_BOOTSTRAP_COLL ||
      state->coll == NULL) {
    return -1;
  }
  return state->coll->nranks;
}

flagcxNetProperties_t *bootstrapGetNetProperties() {
  return &bootstrapNetProperties;
}

const char *bootstrapGetNetIfName() { return bootstrapNetIfName; }

union flagcxSocketAddress *bootstrapGetNetIfAddr() {
  return &bootstrapNetIfAddr;
}

/* Socket Interface Selection type */
enum bootstrapInterface_t { findSubnetIf = -1, dontCareIf = -2 };

// Additional sync functions
static flagcxResult_t bootstrapNetSend(struct flagcxSocket *sock, void *data,
                                       int size) {
  FLAGCXCHECK(flagcxSocketSend(sock, &size, sizeof(int)));
  FLAGCXCHECK(flagcxSocketSend(sock, data, size));
  return flagcxSuccess;
}
static flagcxResult_t bootstrapNetRecv(struct flagcxSocket *sock, void *data,
                                       int size) {
  int recvSize;
  FLAGCXCHECK(flagcxSocketRecv(sock, &recvSize, sizeof(int)));
  if (recvSize > size) {
    WARN(
        "bootstrapNetRecv: message truncated : received %d bytes instead of %d",
        recvSize, size);
    return flagcxInternalError;
  }
  FLAGCXCHECK(flagcxSocketRecv(sock, data, std::min(recvSize, size)));
  return flagcxSuccess;
}
// Sequential send-then-recv is safe here because bootstrapNetSendRecv is only
// called on ring topology with separate ringSendSocket and ringRecvSocket.
// One rank's send-to-next and recv-from-prev use independent sockets, so there
// is no risk of deadlock.
static flagcxResult_t bootstrapNetSendRecv(struct flagcxSocket *sendSocket,
                                           void *sendData, int sendSize,
                                           struct flagcxSocket *recvSocket,
                                           void *recvData, int recvSize) {
  FLAGCXCHECK(bootstrapNetSend(sendSocket, sendData, sendSize));
  FLAGCXCHECK(bootstrapNetRecv(recvSocket, recvData, recvSize));
  return flagcxSuccess;
}

struct extInfo {
  int rank;
  int nranks;
  union flagcxSocketAddress extAddressListen;
  union flagcxSocketAddress extAddressListenRoot;
};

#include <cstdint>
#include <sys/resource.h>

static flagcxResult_t setFilesLimit() {
  struct rlimit filesLimit;
  SYSCHECK(getrlimit(RLIMIT_NOFILE, &filesLimit), "getrlimit");
  filesLimit.rlim_cur = filesLimit.rlim_max;
  SYSCHECK(setrlimit(RLIMIT_NOFILE, &filesLimit), "setrlimit");
  return flagcxSuccess;
}

// Root thread for collective bootstrap ring setup
static void *bootstrapRoot(void *rargs) {
  struct bootstrapRootArgs *args = (struct bootstrapRootArgs *)rargs;
  struct flagcxSocket *listenSock = args->listenSock;
  uint64_t magic = args->magic;
  free(args);

  setFilesLimit();

  struct extInfo info;
  struct extInfo *rankInfo = NULL;
  int nranks = 0;
  flagcxResult_t res = flagcxSuccess;

  /* Receive addresses from all ranks */
  int checkedIn = 0;
  do {
    struct flagcxSocket sock;
    FLAGCXCHECKGOTO(flagcxSocketInit(&sock), res, fail);
    FLAGCXCHECKGOTO(flagcxSocketAccept(&sock, listenSock), res, fail);
    FLAGCXCHECKGOTO(bootstrapNetRecv(&sock, &info, sizeof(info)), res, fail);
    FLAGCXCHECKGOTO(flagcxSocketClose(&sock), res, fail);

    if (nranks == 0) {
      nranks = info.nranks;
      FLAGCXCHECKGOTO(flagcxCalloc(&rankInfo, nranks), res, fail);
    } else if (info.nranks != nranks) {
      WARN("Bootstrap Root: rank %d nranks mismatch: expected %d, got %d",
           info.rank, nranks, info.nranks);
      res = flagcxInvalidArgument;
      goto fail;
    }
    if (info.rank < 0 || info.rank >= nranks) {
      res = flagcxInvalidArgument;
      goto fail;
    }
    if (rankInfo[info.rank].nranks == 0)
      checkedIn++;
    rankInfo[info.rank] = info;
  } while (checkedIn < nranks);

  /* Send everyone info about their "next" rank in the ring */
  for (int i = 0; i < nranks; ++i) {
    int next = (i + 1) % nranks;
    struct flagcxSocket sock;
    FLAGCXCHECKGOTO(flagcxSocketInit(&sock, &rankInfo[i].extAddressListenRoot,
                                     magic, flagcxSocketTypeBootstrap),
                    res, fail);
    FLAGCXCHECKGOTO(flagcxSocketConnect(&sock), res, fail);
    FLAGCXCHECKGOTO(bootstrapNetSend(&sock, &rankInfo[next].extAddressListen,
                                     sizeof(union flagcxSocketAddress)),
                    res, fail);
    FLAGCXCHECKGOTO(flagcxSocketClose(&sock), res, fail);
  }

  TRACE(FLAGCX_INIT, "DONE magic %lx", magic);
fail:
  if (res != flagcxSuccess) {
    WARN("bootstrapRoot thread failed with error %d", res);
  }
  if (rankInfo)
    free(rankInfo);
  flagcxSocketClose(listenSock);
  free(listenSock);
  return NULL;
}

flagcxResult_t bootstrapCollCreateRoot(struct flagcxBootstrapHandle *handle,
                                       bool idFromEnv) {
  struct flagcxSocket *listenSock;
  struct bootstrapRootArgs *args;
  pthread_t thread;

  FLAGCXCHECK(flagcxCalloc(&listenSock, 1));
  FLAGCXCHECK(flagcxSocketInit(listenSock, &handle->addr, handle->magic,
                               flagcxSocketTypeBootstrap));
  FLAGCXCHECK(flagcxSocketListen(listenSock));
  FLAGCXCHECK(flagcxSocketGetAddr(listenSock, &handle->addr));

  FLAGCXCHECK(flagcxCalloc(&args, 1));
  args->listenSock = listenSock;
  args->magic = handle->magic;
  if (pthread_create(&thread, NULL, bootstrapRoot, (void *)args) != 0) {
    WARN("bootstrapCollCreateRoot: pthread_create failed");
    free(args);
    flagcxSocketClose(listenSock);
    free(listenSock);
    return flagcxSystemError;
  }
  flagcxSetThreadName(thread, "FlagCX BootRoot");
  (void)pthread_detach(thread);
  return flagcxSuccess;
}

flagcxResult_t bootstrapGetUniqueId(struct flagcxBootstrapHandle *handle) {
  memset(handle, 0, sizeof(*handle));
  const char *env = flagcxGetEnv("FLAGCX_COMM_ID");
  if (env) {
    INFO(FLAGCX_ENV, "FLAGCX_COMM_ID set by environment to %s", env);
    if (flagcxSocketGetAddrFromString(&handle->addr, env) != flagcxSuccess) {
      WARN("Invalid FLAGCX_COMM_ID, please use format: <ipv4>:<port> or "
           "[<ipv6>]:<port> or <hostname>:<port>");
      return flagcxInvalidArgument;
    }
    handle->magic = FLAGCX_MAGIC;
  } else {
    FLAGCXCHECK(getRandomData(&handle->magic, sizeof(handle->magic)));
    memcpy(&handle->addr, &bootstrapNetIfAddr,
           sizeof(union flagcxSocketAddress));
    FLAGCXCHECK(bootstrapCollCreateRoot(handle, false));
  }

  return flagcxSuccess;
}

// ============================================================================
// Internal helpers
// ============================================================================

struct unexConn {
  int peer;
  int tag;
  struct flagcxSocket sock;
  struct unexConn *next;
};

// Helper to unwrap bootstrapState -> bootstrapCollState
// The coll state is always valid regardless of current mode.
static inline struct bootstrapCollState *
unwrapCollState(struct bootstrapState *state) {
  if (state == NULL || state->coll == NULL) {
    return NULL;
  }
  return state->coll;
}

// ============================================================================
// Collective Mode Init
// ============================================================================

flagcxResult_t bootstrapCollInit(struct flagcxBootstrapHandle *handle, int rank,
                                 int nranks, uint64_t magic,
                                 volatile uint32_t *abortFlag,
                                 struct bootstrapState **stateOut) {
  // Allocate wrapper
  struct bootstrapState *wrapper;
  FLAGCXCHECK(flagcxCalloc(&wrapper, 1));
  wrapper->mode = FLAGCX_BOOTSTRAP_COLL;

  // Allocate coll state
  struct bootstrapCollState *state;
  FLAGCXCHECK(flagcxCalloc(&state, 1));
  wrapper->coll = state;

  // Fill in parameters
  state->rank = rank;
  state->nranks = nranks;
  state->magic = magic;
  state->abortFlag = abortFlag;

  flagcxSocketAddress nextAddr;
  struct flagcxSocket sock, listenSockRoot;
  struct extInfo info = {0};

  TRACE(FLAGCX_INIT, "rank %d nranks %d", rank, nranks);

  info.rank = rank;
  info.nranks = nranks;
  // Create socket for other ranks to contact me
  FLAGCXCHECK(flagcxSocketInit(&state->listenSock, &bootstrapNetIfAddr,
                               state->magic, flagcxSocketTypeBootstrap,
                               state->abortFlag));
  FLAGCXCHECK(flagcxSocketListen(&state->listenSock));
  FLAGCXCHECK(flagcxSocketGetAddr(&state->listenSock, &info.extAddressListen));

  // Create socket for root to contact me
  FLAGCXCHECK(flagcxSocketInit(&listenSockRoot, &bootstrapNetIfAddr,
                               state->magic, flagcxSocketTypeBootstrap,
                               state->abortFlag));
  FLAGCXCHECK(flagcxSocketListen(&listenSockRoot));
  FLAGCXCHECK(flagcxSocketGetAddr(&listenSockRoot, &info.extAddressListenRoot));

  // stagger connection times to avoid an overload of the root
  if (nranks > 128) {
    long msec = rank;
    struct timespec tv;
    tv.tv_sec = msec / 1000;
    tv.tv_nsec = 1000000 * (msec % 1000);
    TRACE(FLAGCX_INIT, "rank %d delaying connection to root by %ld msec", rank,
          msec);
    (void)nanosleep(&tv, NULL);
  }

  // send info on my listening socket to root
  FLAGCXCHECK(flagcxSocketInit(&sock, &handle->addr, state->magic,
                               flagcxSocketTypeBootstrap, state->abortFlag));
  FLAGCXCHECK(flagcxSocketConnect(&sock));
  FLAGCXCHECK(bootstrapNetSend(&sock, &info, sizeof(info)));
  FLAGCXCHECK(flagcxSocketClose(&sock));

  // get info on my "next" rank in the bootstrap ring from root
  FLAGCXCHECK(flagcxSocketInit(&sock));
  FLAGCXCHECK(flagcxSocketAccept(&sock, &listenSockRoot));
  FLAGCXCHECK(
      bootstrapNetRecv(&sock, &nextAddr, sizeof(union flagcxSocketAddress)));
  FLAGCXCHECK(flagcxSocketClose(&sock));
  FLAGCXCHECK(flagcxSocketClose(&listenSockRoot));

  FLAGCXCHECK(flagcxSocketInit(&state->ringSendSocket, &nextAddr, state->magic,
                               flagcxSocketTypeBootstrap, state->abortFlag));
  FLAGCXCHECK(flagcxSocketConnect(&state->ringSendSocket));
  // Accept the connect request from the previous rank in the AllGather ring
  FLAGCXCHECK(flagcxSocketInit(&state->ringRecvSocket));
  FLAGCXCHECK(flagcxSocketAccept(&state->ringRecvSocket, &state->listenSock));

  // AllGather all listen handlers
  FLAGCXCHECK(flagcxCalloc(&state->peerCommAddresses, nranks));
  FLAGCXCHECK(
      flagcxSocketGetAddr(&state->listenSock, state->peerCommAddresses + rank));
  FLAGCXCHECK(bootstrapCollAllGather(wrapper, state->peerCommAddresses,
                                     sizeof(union flagcxSocketAddress)));

  // Set bootstrap net info
  INFO(FLAGCX_INIT, "rank %d nranks %d - DONE", rank, nranks);

  *stateOut = wrapper;
  return flagcxSuccess;
}

// ============================================================================
// Collective Mode: Send/Recv internals (ring relay)
// ============================================================================

static flagcxResult_t bootstrapCollConnect(struct bootstrapState *state,
                                           int peer, int tag,
                                           struct flagcxSocket *sock) {
  flagcxResult_t ret = flagcxSuccess;
  struct bootstrapCollState *coll = unwrapCollState(state);
  if (coll == NULL)
    return flagcxInvalidArgument;

  FLAGCXCHECKGOTO(flagcxSocketInit(sock, coll->peerCommAddresses + peer,
                                   coll->magic, flagcxSocketTypeBootstrap),
                  ret, fail);
  FLAGCXCHECKGOTO(flagcxSocketConnect(sock), ret, fail);
  FLAGCXCHECKGOTO(bootstrapNetSend(sock, &coll->rank, sizeof(int)), ret, fail);
  FLAGCXCHECKGOTO(bootstrapNetSend(sock, &tag, sizeof(int)), ret, fail);
  return flagcxSuccess;
fail:
  FLAGCXCHECK(flagcxSocketClose(sock));
  return ret;
}

static flagcxResult_t unexpectedEnqueue(struct bootstrapCollState *state,
                                        int peer, int tag,
                                        struct flagcxSocket *sock) {
  struct unexConn *unex;
  FLAGCXCHECK(flagcxCalloc(&unex, 1));
  unex->peer = peer;
  unex->tag = tag;
  memcpy(&unex->sock, sock, sizeof(struct flagcxSocket));

  struct unexConn *list = state->unexpectedConnections;
  if (list == NULL) {
    state->unexpectedConnections = unex;
    return flagcxSuccess;
  }
  while (list->next)
    list = list->next;
  list->next = unex;
  return flagcxSuccess;
}

static flagcxResult_t unexpectedDequeue(struct bootstrapCollState *state,
                                        int peer, int tag,
                                        struct flagcxSocket *sock, int *found) {
  struct unexConn *elem = state->unexpectedConnections;
  struct unexConn *prev = NULL;
  *found = 0;
  while (elem) {
    if (elem->peer == peer && elem->tag == tag) {
      if (prev == NULL) {
        state->unexpectedConnections = elem->next;
      } else {
        prev->next = elem->next;
      }
      memcpy(sock, &elem->sock, sizeof(struct flagcxSocket));
      free(elem);
      *found = 1;
      return flagcxSuccess;
    }
    prev = elem;
    elem = elem->next;
  }
  return flagcxSuccess;
}

static void unexpectedFree(struct bootstrapCollState *state) {
  struct unexConn *elem = state->unexpectedConnections;
  struct unexConn *prev = NULL;
  while (elem) {
    prev = elem;
    elem = elem->next;
    flagcxSocketClose(&prev->sock);
    free(prev);
  }
}

static flagcxResult_t bootstrapCollAccept(struct bootstrapState *state,
                                          int peer, int tag,
                                          struct flagcxSocket *sock) {
  flagcxResult_t ret = flagcxSuccess;
  struct bootstrapCollState *coll = unwrapCollState(state);
  if (coll == NULL)
    return flagcxInvalidArgument;
  int newPeer, newTag;

  // Search unexpected connections first
  int found;
  FLAGCXCHECK(unexpectedDequeue(coll, peer, tag, sock, &found));
  if (found)
    return flagcxSuccess;

  // Then look for new connections
  while (1) {
    FLAGCXCHECKGOTO(flagcxSocketInit(sock), ret, fail);
    FLAGCXCHECKGOTO(flagcxSocketAccept(sock, &coll->listenSock), ret, fail);
    FLAGCXCHECKGOTO(bootstrapNetRecv(sock, &newPeer, sizeof(int)), ret, fail);
    FLAGCXCHECKGOTO(bootstrapNetRecv(sock, &newTag, sizeof(int)), ret, fail);
    if (newPeer == peer && newTag == tag)
      return flagcxSuccess;
    FLAGCXCHECKGOTO(unexpectedEnqueue(coll, newPeer, newTag, sock), ret, fail);
  }
  return flagcxSuccess;
fail:
  FLAGCXCHECK(flagcxSocketClose(sock));
  return ret;
}

static flagcxResult_t bootstrapCollSendInternal(struct bootstrapState *state,
                                                int peer, int tag, void *data,
                                                int size) {
  flagcxResult_t ret = flagcxSuccess;
  struct flagcxSocket sock;

  TRACE(FLAGCX_BOOTSTRAP, "Sending to peer=%d tag=%d size=%d", peer, tag, size);
  FLAGCXCHECK(bootstrapCollConnect(state, peer, tag, &sock));
  FLAGCXCHECKGOTO(bootstrapNetSend(&sock, data, size), ret, exit);
  TRACE(FLAGCX_BOOTSTRAP, "Sent to peer=%d tag=%d size=%d", peer, tag, size);

exit:
  FLAGCXCHECK(flagcxSocketClose(&sock));
  return ret;
}

static flagcxResult_t bootstrapCollRecvInternal(struct bootstrapState *state,
                                                int peer, int tag, void *data,
                                                int size) {
  flagcxResult_t ret;
  struct flagcxSocket sock;
  FLAGCXCHECK(bootstrapCollAccept(state, peer, tag, &sock));
  TRACE(FLAGCX_BOOTSTRAP, "Receiving tag=%d peer=%d size=%d", tag, peer, size);
  FLAGCXCHECKGOTO(bootstrapNetRecv(&sock, ((char *)data), size), ret, exit);
exit:
  FLAGCXCHECK(flagcxSocketClose(&sock));
  return ret;
}

// ============================================================================
// Unified Send / Recv / Exchange / Close (dispatch on mode)
// ============================================================================

flagcxResult_t bootstrapSend(struct bootstrapState *state, int peer, int tag,
                             void *data, int size) {
  if (state == NULL)
    return flagcxInvalidArgument;

  if (state->mode == FLAGCX_BOOTSTRAP_P2P) {
    // P2P mode: peer param ignored, send directly to connected socket
    struct bootstrapP2pState *p2p = state->p2p;
    if (p2p == NULL || p2p->isListener)
      return flagcxInvalidArgument;
    FLAGCXCHECK(flagcxSocketSend(&p2p->sock, &tag, sizeof(int)));
    FLAGCXCHECK(flagcxSocketSend(&p2p->sock, &size, sizeof(int)));
    FLAGCXCHECK(flagcxSocketSend(&p2p->sock, data, size));
    return flagcxSuccess;
  }

  // Coll mode: connect-per-send via ring topology
  return bootstrapCollSendInternal(state, peer, tag, data, size);
}

flagcxResult_t bootstrapRecv(struct bootstrapState *state, int peer, int tag,
                             void *data, int size) {
  if (state == NULL)
    return flagcxInvalidArgument;

  if (state->mode == FLAGCX_BOOTSTRAP_P2P) {
    // P2P mode: peer param ignored, recv directly from connected socket
    struct bootstrapP2pState *p2p = state->p2p;
    if (p2p == NULL || p2p->isListener)
      return flagcxInvalidArgument;
    int recvTag, recvSize;
    FLAGCXCHECK(flagcxSocketRecv(&p2p->sock, &recvTag, sizeof(int)));
    FLAGCXCHECK(flagcxSocketRecv(&p2p->sock, &recvSize, sizeof(int)));
    if (recvTag != tag) {
      WARN("P2P recv: tag mismatch expected %d got %d", tag, recvTag);
      return flagcxInternalError;
    }
    if (recvSize > size) {
      WARN("P2P recv: message truncated %d > buffer %d", recvSize, size);
      return flagcxInternalError;
    }
    FLAGCXCHECK(flagcxSocketRecv(&p2p->sock, data, recvSize));
    return flagcxSuccess;
  }

  // Coll mode: accept from listen socket
  return bootstrapCollRecvInternal(state, peer, tag, data, size);
}

flagcxResult_t bootstrapExchange(struct bootstrapState *state, int peer,
                                 int tag, const void *sendData, int sendSize,
                                 void *recvData, int recvSize) {
  if (state == NULL)
    return flagcxInvalidArgument;

  if (state->mode == FLAGCX_BOOTSTRAP_P2P) {
    // P2P mode: order by role to avoid deadlock when payload exceeds
    // socket buffer (connector sends first, acceptor recvs first).
    struct bootstrapP2pState *p2p = state->p2p;
    if (p2p->isConnector) {
      FLAGCXCHECK(bootstrapSend(state, peer, tag, (void *)sendData, sendSize));
      FLAGCXCHECK(bootstrapRecv(state, peer, tag, recvData, recvSize));
    } else {
      FLAGCXCHECK(bootstrapRecv(state, peer, tag, recvData, recvSize));
      FLAGCXCHECK(bootstrapSend(state, peer, tag, (void *)sendData, sendSize));
    }
    return flagcxSuccess;
  }

  // Coll mode: deadlock-free ordering by rank
  struct bootstrapCollState *coll = unwrapCollState(state);
  if (coll == NULL)
    return flagcxInvalidArgument;
  int myRank = coll->rank;

  if (myRank < peer) {
    FLAGCXCHECK(bootstrapSend(state, peer, tag, (void *)sendData, sendSize));
    FLAGCXCHECK(bootstrapRecv(state, peer, tag, recvData, recvSize));
  } else {
    FLAGCXCHECK(bootstrapRecv(state, peer, tag, recvData, recvSize));
    FLAGCXCHECK(bootstrapSend(state, peer, tag, (void *)sendData, sendSize));
  }
  return flagcxSuccess;
}

flagcxResult_t bootstrapClose(struct bootstrapState *state) {
  if (state == NULL)
    return flagcxSuccess;

  if (state->mode == FLAGCX_BOOTSTRAP_P2P) {
    struct bootstrapP2pState *p2p = state->p2p;
    if (p2p != NULL) {
      flagcxSocketClose(&p2p->sock);
      free(p2p);
    }
    free(state);
    return flagcxSuccess;
  }

  // Coll mode
  struct bootstrapCollState *coll = state->coll;
  if (coll == NULL) {
    free(state);
    return flagcxSuccess;
  }
  if (coll->unexpectedConnections != NULL) {
    unexpectedFree(coll);
    if (__atomic_load_n(coll->abortFlag, __ATOMIC_RELAXED) == 0) {
      WARN("Unexpected connections are not empty");
    }
  }

  FLAGCXCHECK(flagcxSocketClose(&coll->listenSock));
  FLAGCXCHECK(flagcxSocketClose(&coll->ringSendSocket));
  FLAGCXCHECK(flagcxSocketClose(&coll->ringRecvSocket));

  free(coll->peerCommAddresses);
  free(coll);
  free(state);
  return flagcxSuccess;
}

flagcxResult_t bootstrapCollAbort(struct bootstrapState *state) {
  if (state == NULL)
    return flagcxSuccess;
  struct bootstrapCollState *coll = state->coll;
  if (coll == NULL) {
    free(state);
    return flagcxSuccess;
  }
  FLAGCXCHECK(flagcxSocketClose(&coll->listenSock));
  FLAGCXCHECK(flagcxSocketClose(&coll->ringSendSocket));
  FLAGCXCHECK(flagcxSocketClose(&coll->ringRecvSocket));
  free(coll->peerCommAddresses);
  free(coll->peerProxyAddresses);
  free(coll);
  free(state);
  return flagcxSuccess;
}

// ============================================================================
// Collective Mode: Collective Operations
// ============================================================================

static flagcxResult_t bootstrapRingAllGather(struct flagcxSocket *prevSocket,
                                             struct flagcxSocket *nextSocket,
                                             int rank, int nranks, char *data,
                                             int size) {
  for (int i = 0; i < nranks - 1; i++) {
    size_t rslice = (rank - i - 1 + nranks) % nranks;
    size_t sslice = (rank - i + nranks) % nranks;
    FLAGCXCHECK(bootstrapNetSendRecv(nextSocket, data + sslice * size, size,
                                     prevSocket, data + rslice * size, size));
  }
  return flagcxSuccess;
}

flagcxResult_t bootstrapCollAllGather(struct bootstrapState *state,
                                      void *allData, int size) {
  struct bootstrapCollState *coll = unwrapCollState(state);
  if (coll == NULL)
    return flagcxInvalidArgument;
  int rank = coll->rank;
  int nranks = coll->nranks;
  FLAGCXCHECK(bootstrapRingAllGather(&coll->ringRecvSocket,
                                     &coll->ringSendSocket, rank, nranks,
                                     (char *)allData, size));
  return flagcxSuccess;
}

flagcxResult_t bootstrapCollIntraNodeBarrier(struct bootstrapState *state,
                                             int *ranks, int rank, int nranks,
                                             int tag) {
  if (nranks == 1)
    return flagcxSuccess;
  TRACE(FLAGCX_INIT, "rank %d nranks %d tag %x - ENTER", rank, nranks, tag);

  int data[1];
  for (int mask = 1; mask < nranks; mask <<= 1) {
    int src = (rank - mask + nranks) % nranks;
    int dst = (rank + mask) % nranks;
    FLAGCXCHECK(bootstrapSend(state, ranks ? ranks[dst] : dst, tag, data,
                              sizeof(data)));
    FLAGCXCHECK(bootstrapRecv(state, ranks ? ranks[src] : src, tag, data,
                              sizeof(data)));
  }

  TRACE(FLAGCX_INIT, "rank %d nranks %d tag %x - DONE", rank, nranks, tag);
  return flagcxSuccess;
}

flagcxResult_t bootstrapCollBarrier(struct bootstrapState *state, int rank,
                                    int nranks, int tag) {
  return bootstrapCollIntraNodeBarrier(state, NULL, rank, nranks, tag);
}

flagcxResult_t bootstrapCollIntraNodeBroadcast(struct bootstrapState *state,
                                               int *ranks, int rank, int nranks,
                                               int root, void *bcastData,
                                               int size) {
  if (nranks == 1)
    return flagcxSuccess;
  TRACE(FLAGCX_INIT, "rank %d nranks %d root %d size %d - ENTER", rank, nranks,
        root, size);

  if (rank == root) {
    for (int i = 0; i < nranks; i++) {
      if (i != root)
        FLAGCXCHECK(bootstrapSend(state, ranks ? ranks[i] : i,
                                  /*tag=*/ranks ? ranks[i] : i, bcastData,
                                  size));
    }
  } else {
    FLAGCXCHECK(bootstrapRecv(state, ranks ? ranks[root] : root,
                              /*tag=*/ranks ? ranks[rank] : rank, bcastData,
                              size));
  }

  TRACE(FLAGCX_INIT, "rank %d nranks %d root %d size %d - DONE", rank, nranks,
        root, size);
  return flagcxSuccess;
}

flagcxResult_t bootstrapCollBroadcast(struct bootstrapState *state, int rank,
                                      int nranks, int root, void *bcastData,
                                      int size) {
  return bootstrapCollIntraNodeBroadcast(state, NULL, rank, nranks, root,
                                         bcastData, size);
}

// ============================================================================
// Typed Collective Operations (Reduce, AllReduce, etc.)
// ============================================================================

static flagcxResult_t bootstrapRingReduceScatter(
    struct flagcxSocket *prevSocket, struct flagcxSocket *nextSocket, int rank,
    int nranks, const char *sendbuff, char *recvbuff, size_t *offset,
    size_t *length, flagcxDataType_t datatype, flagcxRedOp_t op) {
  uint64_t timers[TIMERS_COLL_COUNT] = {0};
  timers[TIMER_COLL_TOTAL] = clockNano();

  timers[TIMER_COLL_ALLOC] = clockNano();
  size_t subSize = 0;
  for (int i = 0; i < nranks; ++i) {
    subSize = std::max(length[i], subSize);
  }
  char *data_for_send = nullptr;
  FLAGCXCHECK(flagcxCalloc(&data_for_send, subSize));
  char *data_for_recv = nullptr;
  FLAGCXCHECK(flagcxCalloc(&data_for_recv, subSize));
  timers[TIMER_COLL_ALLOC] = clockNano() - timers[TIMER_COLL_ALLOC];

  uint64_t start = 0;
  uint64_t end = 0;
  for (int iter = 0; iter < nranks - 1; ++iter) {
    int send_chunk_no = (rank + 2 * nranks - iter - 1) % nranks;
    int recv_chunk_no = (rank + 2 * nranks - iter - 2) % nranks;
    bool needSend = length[send_chunk_no] != 0;
    bool needRecv = length[recv_chunk_no] != 0;

    INFO(FLAGCX_COLL,
         "rank %d nranks %d; iter=%d; send_chunk_no=%d; send_chunk_size=%lu; "
         "needSend=%d; recv_chunk_no=%d; recv_chunk_size=%lu; needRecv=%d",
         rank, nranks, iter, send_chunk_no, length[send_chunk_no], needSend,
         recv_chunk_no, length[recv_chunk_no], needRecv);
    if (!needSend && !needRecv) {
      continue;
    }

    if (needSend) {
      start = clockNano();
      if (iter == 0) {
        memcpy(data_for_send, sendbuff + offset[send_chunk_no],
               length[send_chunk_no]);
      } else {
        std::swap(data_for_send, data_for_recv);
      }
      end = clockNano();
      timers[TIMER_COLL_MEM] += end - start;
    }

    start = clockNano();
    if (needSend && needRecv) {
      FLAGCXCHECK(bootstrapNetSendRecv(
          nextSocket, (void *)data_for_send, length[send_chunk_no], prevSocket,
          (void *)data_for_recv, length[recv_chunk_no]));
    } else if (needSend) {
      FLAGCXCHECK(bootstrapNetSend(nextSocket, (void *)data_for_send,
                                   length[send_chunk_no]));
    } else if (needRecv) {
      FLAGCXCHECK(bootstrapNetRecv(prevSocket, (void *)data_for_recv,
                                   length[recv_chunk_no]));
    }
    end = clockNano();
    timers[TIMER_COLL_COMM] += end - start;

    if (!needRecv) {
      continue;
    }
    start = clockNano();
    switch (op) {
      case flagcxSum:
        GENERATE_ALL_TYPES(datatype, sum, data_for_recv,
                           sendbuff + offset[recv_chunk_no], data_for_recv,
                           length[recv_chunk_no] /
                               getFlagcxDataTypeSize(datatype));
        break;
      case flagcxMax:
        GENERATE_ALL_TYPES(datatype, max, data_for_recv,
                           sendbuff + offset[recv_chunk_no], data_for_recv,
                           length[recv_chunk_no] /
                               getFlagcxDataTypeSize(datatype));
        break;
      case flagcxMin:
        GENERATE_ALL_TYPES(datatype, min, data_for_recv,
                           sendbuff + offset[recv_chunk_no], data_for_recv,
                           length[recv_chunk_no] /
                               getFlagcxDataTypeSize(datatype));
        break;
      default:
        WARN("Unsupported reduction operation %d", op);
        return flagcxInvalidArgument;
    }
    end = clockNano();
    timers[TIMER_COLL_CALC] += end - start;
  }

  memcpy(recvbuff, data_for_recv, length[rank]);
  free(data_for_send);
  free(data_for_recv);

  timers[TIMER_COLL_TOTAL] = clockNano() - timers[TIMER_COLL_TOTAL];
  INFO(FLAGCX_COLL,
       "COLL timings - %s: rank %d nranks %d total %.2fms (calc %.2fms, "
       "mem_alloc %.2fms, memory %.2fms, comm %.2fms)",
       "BootstrapRingReduceScatter", rank, nranks,
       timers[TIMER_COLL_TOTAL] / 1e6, timers[TIMER_COLL_CALC] / 1e6,
       timers[TIMER_COLL_ALLOC] / 1e6, timers[TIMER_COLL_MEM] / 1e6,
       timers[TIMER_COLL_COMM] / 1e6);
  return flagcxSuccess;
}

static flagcxResult_t
bootstrapRingAllReduce(struct flagcxSocket *prevSocket,
                       struct flagcxSocket *nextSocket, int rank, int nranks,
                       const char *sendbuff, char *recvbuff, size_t count,
                       flagcxDataType_t datatype, flagcxRedOp_t op) {
  size_t size = count * getFlagcxDataTypeSize(datatype);
  // ChunkBytes = ceil(size / nranks)
  size_t ChunkBytes = (size + nranks - 1) / nranks;
  // Round up to type size
  ChunkBytes = (ChunkBytes + getFlagcxDataTypeSize(datatype) - 1) /
               getFlagcxDataTypeSize(datatype) *
               getFlagcxDataTypeSize(datatype);

  INFO(FLAGCX_COLL, "rank %d nranks %d; size=%lu; typeSize=%lu; ChunkBytes=%lu",
       rank, nranks, size, getFlagcxDataTypeSize(datatype), ChunkBytes);

  std::vector<size_t> offset(nranks, 0);
  std::vector<size_t> length(nranks, 0);
  for (size_t i = 0; i < (size_t)nranks; ++i) {
    if (ChunkBytes * i >= size) {
      offset[i] = size;
      length[i] = 0;
      continue;
    }
    offset[i] = ChunkBytes * i;
    length[i] =
        ChunkBytes * (i + 1) >= size ? size - ChunkBytes * i : ChunkBytes;
  }

  // ReduceScatter
  FLAGCXCHECK(bootstrapRingReduceScatter(prevSocket, nextSocket, rank, nranks,
                                         sendbuff, recvbuff, offset.data(),
                                         length.data(), datatype, op));

  // AllGather the results with variable chunk sizes
  // Copy my chunk into correct position
  memmove(recvbuff + offset[rank], recvbuff, length[rank]);

  // Ring AllGather with per-slice variable sizes
  for (int i = 0; i < nranks - 1; i++) {
    size_t sslice = (rank - i + nranks) % nranks;
    size_t rslice = (rank - i - 1 + nranks) % nranks;
    FLAGCXCHECK(bootstrapNetSendRecv(
        nextSocket, recvbuff + offset[sslice], (int)length[sslice], prevSocket,
        recvbuff + offset[rslice], (int)length[rslice]));
  }
  return flagcxSuccess;
}

static flagcxResult_t
bootstrapRingReduce(struct bootstrapState *commState,
                    struct flagcxSocket *prevSocket,
                    struct flagcxSocket *nextSocket, int rank, int nranks,
                    const char *sendbuff, char *recvbuff, size_t count,
                    flagcxDataType_t datatype, flagcxRedOp_t op, int root) {
  size_t size = count * getFlagcxDataTypeSize(datatype);
  size_t ChunkBytes = (size + nranks - 1) / nranks;
  ChunkBytes = (ChunkBytes + getFlagcxDataTypeSize(datatype) - 1) /
               getFlagcxDataTypeSize(datatype) *
               getFlagcxDataTypeSize(datatype);

  INFO(FLAGCX_COLL, "rank %d nranks %d; size=%lu; typeSize=%lu; ChunkBytes=%lu",
       rank, nranks, size, getFlagcxDataTypeSize(datatype), ChunkBytes);

  std::vector<size_t> offset(nranks, 0);
  std::vector<size_t> length(nranks, 0);
  for (size_t i = 0; i < (size_t)nranks; ++i) {
    if (ChunkBytes * i >= size) {
      offset[i] = size;
      length[i] = 0;
      continue;
    }
    offset[i] = ChunkBytes * i;
    length[i] =
        ChunkBytes * (i + 1) >= size ? size - ChunkBytes * i : ChunkBytes;
  }

  // reduce scatter
  FLAGCXCHECK(bootstrapRingReduceScatter(prevSocket, nextSocket, rank, nranks,
                                         sendbuff, recvbuff, offset.data(),
                                         length.data(), datatype, op));

  // Move my reduced chunk to its final position before gather to root
  memmove(recvbuff + offset[rank], recvbuff, length[rank]);

  // gather to root
  const int bootstrapTag = BOOTSTRAP_TAG_REDUCE;
  if (rank == root) {
    for (int i = 0; i < nranks; i++) {
      if (i == rank)
        continue;
      FLAGCXCHECK(bootstrapRecv(commState, i, bootstrapTag,
                                recvbuff + offset[i], length[i]));
    }
  } else {
    FLAGCXCHECK(bootstrapSend(commState, root, bootstrapTag,
                              recvbuff + offset[rank], length[rank]));
  }

  return flagcxSuccess;
}

flagcxResult_t AllReduceBootstrap(struct bootstrapState *state,
                                  const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  flagcxRedOp_t op) {
  struct bootstrapCollState *coll = unwrapCollState(state);
  if (coll == NULL)
    return flagcxInvalidArgument;
  int rank = coll->rank;
  int nranks = coll->nranks;
  if (nranks == 1) {
    if (sendbuff != recvbuff) {
      memcpy(recvbuff, sendbuff, count * getFlagcxDataTypeSize(datatype));
    }
    return flagcxSuccess;
  }
  FLAGCXCHECK(bootstrapRingAllReduce(
      &coll->ringRecvSocket, &coll->ringSendSocket, rank, nranks,
      (char *)sendbuff, (char *)recvbuff, count, datatype, op));
  return flagcxSuccess;
}

flagcxResult_t ReduceBootstrap(struct bootstrapState *state,
                               const void *sendbuff, void *recvbuff,
                               size_t count, flagcxDataType_t datatype,
                               flagcxRedOp_t op, int root) {
  struct bootstrapCollState *coll = unwrapCollState(state);
  if (coll == NULL)
    return flagcxInvalidArgument;
  int rank = coll->rank;
  int nranks = coll->nranks;
  if (nranks == 1) {
    if (sendbuff != recvbuff) {
      memcpy(recvbuff, sendbuff, count * getFlagcxDataTypeSize(datatype));
    }
    return flagcxSuccess;
  }
  FLAGCXCHECK(bootstrapRingReduce(
      state, &coll->ringRecvSocket, &coll->ringSendSocket, rank, nranks,
      (char *)sendbuff, (char *)recvbuff, count, datatype, op, root));
  return flagcxSuccess;
}

flagcxResult_t ReduceScatterBootstrap(struct bootstrapState *state,
                                      const void *sendbuff, void *recvbuff,
                                      size_t recvcount,
                                      flagcxDataType_t datatype,
                                      flagcxRedOp_t op) {
  struct bootstrapCollState *coll = unwrapCollState(state);
  if (coll == NULL)
    return flagcxInvalidArgument;
  int rank = coll->rank;
  int nranks = coll->nranks;
  if (nranks == 1) {
    if (sendbuff != recvbuff) {
      memcpy(recvbuff, sendbuff, recvcount * getFlagcxDataTypeSize(datatype));
    }
    return flagcxSuccess;
  }
  std::vector<size_t> offset(nranks, 0);
  std::vector<size_t> length(nranks, 0);
  for (size_t i = 0; i < (size_t)nranks; ++i) {
    offset[i] = i * recvcount * getFlagcxDataTypeSize(datatype);
    length[i] = recvcount * getFlagcxDataTypeSize(datatype);
  }
  FLAGCXCHECK(bootstrapRingReduceScatter(
      &coll->ringRecvSocket, &coll->ringSendSocket, rank, nranks,
      (char *)sendbuff, (char *)recvbuff, offset.data(), length.data(),
      datatype, op));
  return flagcxSuccess;
}

flagcxResult_t AlltoAllBootstrap(struct bootstrapState *state,
                                 const void *sendbuff, void *recvbuff,
                                 size_t count, flagcxDataType_t datatype) {
  struct bootstrapCollState *coll = unwrapCollState(state);
  if (coll == NULL)
    return flagcxInvalidArgument;
  int rank = coll->rank;
  int nranks = coll->nranks;
  size_t size = count * getFlagcxDataTypeSize(datatype);

  bool inPlace = (sendbuff == recvbuff);
  char *tmpBuff = nullptr;
  if (inPlace) {
    FLAGCXCHECK(flagcxCalloc(&tmpBuff, nranks * size));
    memcpy(tmpBuff, sendbuff, nranks * size);
    sendbuff = tmpBuff;
  }

  const int bootstrapTag = BOOTSTRAP_TAG_ALLTOALL;
  flagcxResult_t res = flagcxSuccess;
  for (int i = 0; i < nranks; i++) {
    if (i == rank) {
      memcpy((char *)recvbuff + rank * size, (char *)sendbuff + rank * size,
             size);
      continue;
    }
    if (rank > i) {
      FLAGCXCHECKGOTO(bootstrapSend(state, i, bootstrapTag,
                                    (char *)sendbuff + i * size, size),
                      res, cleanup);
      FLAGCXCHECKGOTO(bootstrapRecv(state, i, bootstrapTag,
                                    (char *)recvbuff + i * size, size),
                      res, cleanup);
    } else {
      FLAGCXCHECKGOTO(bootstrapRecv(state, i, bootstrapTag,
                                    (char *)recvbuff + i * size, size),
                      res, cleanup);
      FLAGCXCHECKGOTO(bootstrapSend(state, i, bootstrapTag,
                                    (char *)sendbuff + i * size, size),
                      res, cleanup);
    }
  }

cleanup:
  if (tmpBuff)
    free(tmpBuff);
  return res;
}

flagcxResult_t BroadcastBootstrap(struct bootstrapState *state,
                                  const void *sendbuff, void *recvbuff,
                                  size_t count, flagcxDataType_t datatype,
                                  int root) {
  struct bootstrapCollState *coll = unwrapCollState(state);
  if (coll == NULL)
    return flagcxInvalidArgument;
  int rank = coll->rank;
  int nranks = coll->nranks;
  const int bootstrapTag = BOOTSTRAP_TAG_BROADCAST;
  if (nranks == 1) {
    if (sendbuff != recvbuff) {
      memcpy(recvbuff, sendbuff, getFlagcxDataTypeSize(datatype) * count);
    }
    return flagcxSuccess;
  }

  if (rank == root) {
    size_t rootOffset = root * count * getFlagcxDataTypeSize(datatype);
    if ((char *)sendbuff + rootOffset != recvbuff) {
      memcpy(recvbuff, (const char *)sendbuff + rootOffset,
             getFlagcxDataTypeSize(datatype) * count);
    }
    for (int i = 0; i < nranks; ++i) {
      if (i != root) {
        size_t offset = i * count * getFlagcxDataTypeSize(datatype);
        FLAGCXCHECK(bootstrapSend(state, i, bootstrapTag,
                                  (char *)sendbuff + offset,
                                  count * getFlagcxDataTypeSize(datatype)));
      }
    }
  } else {
    FLAGCXCHECK(bootstrapRecv(state, root, bootstrapTag, recvbuff,
                              count * getFlagcxDataTypeSize(datatype)));
  }
  return flagcxSuccess;
}

flagcxResult_t GatherBootstrap(struct bootstrapState *state,
                               const void *sendbuff, void *recvbuff,
                               size_t count, flagcxDataType_t datatype,
                               int root) {
  struct bootstrapCollState *coll = unwrapCollState(state);
  if (coll == NULL)
    return flagcxInvalidArgument;
  int rank = coll->rank;
  int nranks = coll->nranks;
  const int bootstrapTag = BOOTSTRAP_TAG_GATHER;

  if (nranks == 1) {
    if (sendbuff != recvbuff) {
      memcpy(recvbuff, sendbuff, getFlagcxDataTypeSize(datatype) * count);
    }
    return flagcxSuccess;
  }

  if (rank == root) {
    size_t rootOffset = root * count * getFlagcxDataTypeSize(datatype);
    if (sendbuff != (char *)recvbuff + rootOffset) {
      memcpy((char *)recvbuff + rootOffset, sendbuff,
             getFlagcxDataTypeSize(datatype) * count);
    }
    for (int i = 0; i < nranks; ++i) {
      if (i != root) {
        int offset = i * count * getFlagcxDataTypeSize(datatype);
        FLAGCXCHECK(bootstrapRecv(state, i, bootstrapTag,
                                  (char *)recvbuff + offset,
                                  count * getFlagcxDataTypeSize(datatype)));
      }
    }
  } else {
    FLAGCXCHECK(bootstrapSend(state, root, bootstrapTag, (void *)sendbuff,
                              count * getFlagcxDataTypeSize(datatype)));
  }
  return flagcxSuccess;
}

flagcxResult_t ScatterBootstrap(struct bootstrapState *state,
                                const void *sendbuff, void *recvbuff,
                                size_t count, flagcxDataType_t datatype,
                                int root) {
  struct bootstrapCollState *coll = unwrapCollState(state);
  if (coll == NULL)
    return flagcxInvalidArgument;
  int rank = coll->rank;
  int nranks = coll->nranks;
  size_t size = count * getFlagcxDataTypeSize(datatype);
  const int bootstrapTag = BOOTSTRAP_TAG_SCATTER;

  if (rank == root) {
    // Root sends to all non-root ranks
    memcpy((void *)recvbuff, (const char *)sendbuff + rank * size, size);
    for (int i = 0; i < nranks; i++) {
      if (i != root) {
        FLAGCXCHECK(bootstrapSend(state, i, bootstrapTag,
                                  (void *)((const char *)sendbuff + i * size),
                                  size));
      }
    }
  } else {
    // Non-root receives from root
    FLAGCXCHECK(bootstrapRecv(state, root, bootstrapTag, recvbuff, size));
  }
  return flagcxSuccess;
}

flagcxResult_t AllGatherBootstrap(struct bootstrapState *state,
                                  const void *sendbuff, void *recvbuff,
                                  size_t sendcount, flagcxDataType_t datatype) {
  struct bootstrapCollState *coll = unwrapCollState(state);
  if (coll == NULL)
    return flagcxInvalidArgument;
  int rank = coll->rank;
  size_t size = sendcount * getFlagcxDataTypeSize(datatype);

  // Copy own data into the correct slot (memmove handles in-place case
  // where sendbuff == recvbuff + rank * size)
  memmove((char *)recvbuff + rank * size, sendbuff, size);

  // Use the existing bootstrapCollAllGather on the receive buffer
  FLAGCXCHECK(bootstrapCollAllGather(state, recvbuff, size));
  return flagcxSuccess;
}

flagcxResult_t AlltoAllvBootstrap(struct bootstrapState *state,
                                  const void *sendbuff, size_t *sendcounts,
                                  size_t *sdispls, void *recvbuff,
                                  size_t *recvcounts, size_t *rdispls,
                                  flagcxDataType_t datatype) {
  struct bootstrapCollState *coll = unwrapCollState(state);
  if (coll == NULL)
    return flagcxInvalidArgument;
  int rank = coll->rank;
  int nranks = coll->nranks;
  size_t typeSize = getFlagcxDataTypeSize(datatype);

  for (int i = 0; i < nranks; ++i) {
    if (i == rank) {
      memcpy((void *)((char *)recvbuff + rdispls[i] * typeSize),
             (void *)((char *)sendbuff + sdispls[i] * typeSize),
             sendcounts[i] * typeSize);
    }
    const int bootstrapTag = BOOTSTRAP_TAG_ALLTOALLV;
    if (rank > i) {
      FLAGCXCHECK(
          bootstrapSend(state, i, bootstrapTag,
                        (void *)((char *)sendbuff + sdispls[i] * typeSize),
                        sendcounts[i] * typeSize));
      FLAGCXCHECK(
          bootstrapRecv(state, i, bootstrapTag,
                        (void *)((char *)recvbuff + rdispls[i] * typeSize),
                        recvcounts[i] * typeSize));
    } else if (rank < i) {
      FLAGCXCHECK(
          bootstrapRecv(state, i, bootstrapTag,
                        (void *)((char *)recvbuff + rdispls[i] * typeSize),
                        recvcounts[i] * typeSize));
      FLAGCXCHECK(
          bootstrapSend(state, i, bootstrapTag,
                        (void *)((char *)sendbuff + sdispls[i] * typeSize),
                        sendcounts[i] * typeSize));
    }
  }
  return flagcxSuccess;
}

// ============================================================================
// P2P Mode: RPC-style Listen / Connect / Accept
// ============================================================================

flagcxResult_t bootstrapP2pListen(uint64_t magic, volatile uint32_t *abortFlag,
                                  void *listenHandle,
                                  struct bootstrapState **stateOut) {
  // Ensure network interface is discovered
  FLAGCXCHECK(bootstrapNetInit());

  // Allocate P2P state
  struct bootstrapP2pState *p2p;
  FLAGCXCHECK(flagcxCalloc(&p2p, 1));
  p2p->isListener = true;
  p2p->magic = magic;
  p2p->abortFlag = abortFlag;

  // Bind listen socket on discovered NIC
  FLAGCXCHECK(flagcxSocketInit(&p2p->sock, &bootstrapNetIfAddr, magic,
                               flagcxSocketTypeBootstrap, abortFlag));
  FLAGCXCHECK(flagcxSocketListen(&p2p->sock));
  FLAGCXCHECK(flagcxSocketGetAddr(&p2p->sock, &p2p->localAddr));

  // Fill handle for peer to use in bootstrapP2pConnect
  struct flagcxBootstrapHandle *handle =
      (struct flagcxBootstrapHandle *)listenHandle;
  handle->magic = magic;
  memcpy(&handle->addr, &p2p->localAddr, sizeof(union flagcxSocketAddress));

  // Wrap in state
  struct bootstrapState *wrapper;
  FLAGCXCHECK(flagcxCalloc(&wrapper, 1));
  wrapper->mode = FLAGCX_BOOTSTRAP_P2P;
  wrapper->p2p = p2p;

  *stateOut = wrapper;
  return flagcxSuccess;
}

flagcxResult_t bootstrapP2pConnect(void *peerHandle, uint64_t magic,
                                   volatile uint32_t *abortFlag,
                                   struct bootstrapState **stateOut) {
  // Ensure network interface is discovered
  FLAGCXCHECK(bootstrapNetInit());

  struct flagcxBootstrapHandle *handle =
      (struct flagcxBootstrapHandle *)peerHandle;

  // Allocate P2P state
  struct bootstrapP2pState *p2p;
  FLAGCXCHECK(flagcxCalloc(&p2p, 1));
  p2p->isListener = false;
  p2p->isConnector = true;
  p2p->magic = magic;
  p2p->abortFlag = abortFlag;

  // Connect to peer's listen socket
  FLAGCXCHECK(flagcxSocketInit(&p2p->sock, &handle->addr, magic,
                               flagcxSocketTypeBootstrap, abortFlag));
  FLAGCXCHECK(flagcxSocketConnect(&p2p->sock));

  // Wrap in state
  struct bootstrapState *wrapper;
  FLAGCXCHECK(flagcxCalloc(&wrapper, 1));
  wrapper->mode = FLAGCX_BOOTSTRAP_P2P;
  wrapper->p2p = p2p;

  *stateOut = wrapper;
  return flagcxSuccess;
}

flagcxResult_t bootstrapP2pAccept(struct bootstrapState *listenState,
                                  struct bootstrapState **connStateOut) {
  if (listenState == NULL || listenState->mode != FLAGCX_BOOTSTRAP_P2P) {
    WARN("bootstrapP2pAccept: not a P2P listen state");
    return flagcxInvalidArgument;
  }
  struct bootstrapP2pState *listenP2p = listenState->p2p;
  if (listenP2p == NULL || !listenP2p->isListener) {
    WARN("bootstrapP2pAccept: state is not in listen mode");
    return flagcxInvalidArgument;
  }

  // Allocate new connected P2P state
  struct bootstrapP2pState *connP2p;
  FLAGCXCHECK(flagcxCalloc(&connP2p, 1));
  connP2p->isListener = false;
  connP2p->isConnector = false;
  connP2p->magic = listenP2p->magic;
  connP2p->abortFlag = listenP2p->abortFlag;

  // Accept incoming connection
  FLAGCXCHECK(flagcxSocketInit(&connP2p->sock, NULL, listenP2p->magic,
                               flagcxSocketTypeBootstrap,
                               listenP2p->abortFlag));
  flagcxResult_t res = flagcxSocketAccept(&connP2p->sock, &listenP2p->sock);
  if (res != flagcxSuccess) {
    flagcxSocketClose(&connP2p->sock);
    free(connP2p);
    return res;
  }

  // Wrap in state
  struct bootstrapState *wrapper;
  FLAGCXCHECK(flagcxCalloc(&wrapper, 1));
  wrapper->mode = FLAGCX_BOOTSTRAP_P2P;
  wrapper->p2p = connP2p;

  *connStateOut = wrapper;
  return flagcxSuccess;
}
