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

struct bootstrapRootArgs {
  struct flagcxSocket *listenSock;
  uint64_t magic;
};

/* Init functions */
static char bootstrapNetIfName[MAX_IF_NAME_SIZE + 1];
union flagcxSocketAddress bootstrapNetIfAddr;
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
    WARN("Message truncated : received %d bytes instead of %d", recvSize, size);
    return flagcxInternalError;
  }
  FLAGCXCHECK(flagcxSocketRecv(sock, data, std::min(recvSize, size)));
  return flagcxSuccess;
}
static flagcxResult_t bootstrapNetSendRecv(struct flagcxSocket *sendSock,
                                           void *sendData, int sendSize,
                                           struct flagcxSocket *recvSock,
                                           void *recvData, int recvSize) {
  int senderRecvSize;
  FLAGCXCHECK(flagcxSocketSendRecv(sendSock, &sendSize, sizeof(int), recvSock,
                                   &senderRecvSize, sizeof(int)));
  if (senderRecvSize > recvSize) {
    WARN("Message truncated : received %d bytes instead of %d", senderRecvSize,
         recvSize);
    return flagcxInternalError;
  }
  FLAGCXCHECK(flagcxSocketSendRecv(sendSock, sendData, sendSize, recvSock,
                                   recvData, recvSize));
  return flagcxSuccess;
}

struct extInfo {
  int rank;
  int nranks;
  union flagcxSocketAddress extAddressListenRoot;
  union flagcxSocketAddress extAddressListen;
};

#include <sys/resource.h>

static flagcxResult_t setFilesLimit() {
  struct rlimit filesLimit;
  SYSCHECK(getrlimit(RLIMIT_NOFILE, &filesLimit), "getrlimit");
  filesLimit.rlim_cur = filesLimit.rlim_max;
  SYSCHECK(setrlimit(RLIMIT_NOFILE, &filesLimit), "setrlimit");
  return flagcxSuccess;
}

static void *bootstrapRoot(void *rargs) {
  struct bootstrapRootArgs *args = (struct bootstrapRootArgs *)rargs;
  struct flagcxSocket *listenSock = args->listenSock;
  uint64_t magic = args->magic;
  flagcxResult_t res = flagcxSuccess;
  int nranks = 0, c = 0;
  struct extInfo info;
  union flagcxSocketAddress *rankAddresses = NULL;
  union flagcxSocketAddress *rankAddressesRoot =
      NULL; // for initial rank <-> root information exchange
  union flagcxSocketAddress *zero = NULL;
  FLAGCXCHECKGOTO(flagcxCalloc(&zero, 1), res, out);
  setFilesLimit();

  TRACE(FLAGCX_INIT, "BEGIN");
  /* Receive addresses from all ranks */
  do {
    struct flagcxSocket sock;
    FLAGCXCHECKGOTO(flagcxSocketInit(&sock), res, out);
    FLAGCXCHECKGOTO(flagcxSocketAccept(&sock, listenSock), res, out);
    FLAGCXCHECKGOTO(bootstrapNetRecv(&sock, &info, sizeof(info)), res, out);
    FLAGCXCHECKGOTO(flagcxSocketClose(&sock), res, out);

    if (c == 0) {
      nranks = info.nranks;
      FLAGCXCHECKGOTO(flagcxCalloc(&rankAddresses, nranks), res, out);
      FLAGCXCHECKGOTO(flagcxCalloc(&rankAddressesRoot, nranks), res, out);
    }

    if (nranks != info.nranks) {
      WARN("Bootstrap Root : mismatch in rank count from procs %d : %d", nranks,
           info.nranks);
      goto out;
    }

    if (memcmp(zero, &rankAddressesRoot[info.rank],
               sizeof(union flagcxSocketAddress)) != 0) {
      WARN("Bootstrap Root : rank %d of %d ranks has already checked in",
           info.rank, nranks);
      goto out;
    }

    INFO(FLAGCX_INIT, "Bootstrap Root : rank %d of %d ranks checked in",
         info.rank, nranks);

    // Save the connection handle for that rank
    memcpy(rankAddressesRoot + info.rank, &info.extAddressListenRoot,
           sizeof(union flagcxSocketAddress));
    memcpy(rankAddresses + info.rank, &info.extAddressListen,
           sizeof(union flagcxSocketAddress));

    ++c;
    TRACE(FLAGCX_INIT, "Received connect from rank %d total %d/%d", info.rank,
          c, nranks);
  } while (c < nranks);
  TRACE(FLAGCX_INIT, "COLLECTED ALL %d HANDLES", nranks);

  // Send the connect handle for the next rank in the AllGather ring
  for (int r = 0; r < nranks; ++r) {
    int next = (r + 1) % nranks;
    struct flagcxSocket sock;
    FLAGCXCHECKGOTO(flagcxSocketInit(&sock, rankAddressesRoot + r, magic,
                                     flagcxSocketTypeBootstrap),
                    res, out);
    FLAGCXCHECKGOTO(flagcxSocketConnect(&sock), res, out);
    FLAGCXCHECKGOTO(bootstrapNetSend(&sock, rankAddresses + next,
                                     sizeof(union flagcxSocketAddress)),
                    res, out);
    FLAGCXCHECKGOTO(flagcxSocketClose(&sock), res, out);
  }
  INFO(FLAGCX_INIT, "SENT OUT ALL %d HANDLES", nranks);

out:
  if (listenSock != NULL) {
    flagcxSocketClose(listenSock);
    free(listenSock);
  }
  if (rankAddresses)
    free(rankAddresses);
  if (rankAddressesRoot)
    free(rankAddressesRoot);
  if (zero)
    free(zero);
  free(rargs);

  TRACE(FLAGCX_INIT, "DONE");
  return NULL;
}

flagcxResult_t bootstrapCreateRoot(struct flagcxBootstrapHandle *handle,
                                   bool idFromEnv) {
  struct flagcxSocket *listenSock;
  struct bootstrapRootArgs *args;
  pthread_t thread;

  FLAGCXCHECK(flagcxCalloc(&listenSock, 1));
  FLAGCXCHECK(flagcxSocketInit(listenSock, &handle->addr, handle->magic,
                               flagcxSocketTypeBootstrap, NULL, 0));
  FLAGCXCHECK(flagcxSocketListen(listenSock));
  FLAGCXCHECK(flagcxSocketGetAddr(listenSock, &handle->addr));

  FLAGCXCHECK(flagcxCalloc(&args, 1));
  args->listenSock = listenSock;
  args->magic = handle->magic;
  NEQCHECK(pthread_create(&thread, NULL, bootstrapRoot, (void *)args), 0);
  flagcxSetThreadName(thread, "FLAGCX bootstrapRoot");
  NEQCHECK(pthread_detach(thread), 0); // will not be pthread_join()'d
  return flagcxSuccess;
}

flagcxResult_t bootstrapGetUniqueId(struct flagcxBootstrapHandle *handle) {
  memset(handle, 0, sizeof(flagcxBootstrapHandle));
  FLAGCXCHECK(getRandomData(&handle->magic, sizeof(handle->magic)));

  const char *env = flagcxGetEnv("FLAGCX_COMM_ID");
  if (env) {
    INFO(FLAGCX_ENV, "FLAGCX_COMM_ID set by environment to %s", env);
    if (flagcxSocketGetAddrFromString(&handle->addr, env) != flagcxSuccess) {
      WARN("Invalid FLAGCX_COMM_ID, please use format: <ipv4>:<port> or "
           "[<ipv6>]:<port> or <hostname>:<port>");
      return flagcxInvalidArgument;
    }
  } else {
    memcpy(&handle->addr, &bootstrapNetIfAddr,
           sizeof(union flagcxSocketAddress));
    FLAGCXCHECK(bootstrapCreateRoot(handle, false));
  }

  return flagcxSuccess;
}

struct unexConn {
  int peer;
  int tag;
  struct flagcxSocket sock;
  struct unexConn *next;
};

flagcxResult_t bootstrapInit(struct flagcxBootstrapHandle *handle,
                             void *commState) {
  struct bootstrapState *state = (struct bootstrapState *)commState;
  int rank = state->rank;
  int nranks = state->nranks;
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
  FLAGCXCHECK(bootstrapAllGather(state, state->peerCommAddresses,
                                 sizeof(union flagcxSocketAddress)));

  INFO(FLAGCX_INIT, "rank %d nranks %d - DONE", rank, nranks);

  return flagcxSuccess;
}

// Bootstrap send/receive functions
//
// We do not keep connections opened with all ranks at all times, and we have no
// guarantee that connections to our unique listen socket will arrive in the
// same order as we need them. Therefore, when establishing a connection, the
// sender sends a (peer, tag) tuple to allow the receiver to identify the flow,
// and keep it in an unexpected queue if needed.

flagcxResult_t bootstrapConnect(void *commState, int peer, int tag,
                                struct flagcxSocket *sock) {
  flagcxResult_t ret = flagcxSuccess;
  struct bootstrapState *state = (struct bootstrapState *)commState;

  FLAGCXCHECKGOTO(flagcxSocketInit(sock, state->peerCommAddresses + peer,
                                   state->magic, flagcxSocketTypeBootstrap),
                  ret, fail);
  FLAGCXCHECKGOTO(flagcxSocketConnect(sock), ret, fail);
  FLAGCXCHECKGOTO(bootstrapNetSend(sock, &state->rank, sizeof(int)), ret, fail);
  FLAGCXCHECKGOTO(bootstrapNetSend(sock, &tag, sizeof(int)), ret, fail);
  return flagcxSuccess;
fail:
  FLAGCXCHECK(flagcxSocketClose(sock));
  return ret;
}

flagcxResult_t bootstrapSend(void *commState, int peer, int tag, void *data,
                             int size) {
  flagcxResult_t ret = flagcxSuccess;
  struct flagcxSocket sock;

  TRACE(FLAGCX_BOOTSTRAP, "Sending to peer=%d tag=%d size=%d", peer, tag, size);
  FLAGCXCHECK(bootstrapConnect(commState, peer, tag, &sock));
  FLAGCXCHECKGOTO(bootstrapNetSend(&sock, data, size), ret, exit);

  TRACE(FLAGCX_BOOTSTRAP, "Sent to peer=%d tag=%d size=%d", peer, tag, size);

exit:
  FLAGCXCHECK(flagcxSocketClose(&sock));
  return ret;
}

flagcxResult_t unexpectedEnqueue(struct bootstrapState *state, int peer,
                                 int tag, struct flagcxSocket *sock) {
  // New unex
  struct unexConn *unex;
  FLAGCXCHECK(flagcxCalloc(&unex, 1));
  unex->peer = peer;
  unex->tag = tag;
  memcpy(&unex->sock, sock, sizeof(struct flagcxSocket));

  // Enqueue
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

flagcxResult_t unexpectedDequeue(struct bootstrapState *state, int peer,
                                 int tag, struct flagcxSocket *sock,
                                 int *found) {
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

static void unexpectedFree(struct bootstrapState *state) {
  struct unexConn *elem = state->unexpectedConnections;
  struct unexConn *prev = NULL;

  while (elem) {
    prev = elem;
    elem = elem->next;
    free(prev);
  }
  return;
}

// We can't know who we'll receive from, so we need to receive everything at
// once
flagcxResult_t bootstrapAccept(void *commState, int peer, int tag,
                               struct flagcxSocket *sock) {
  flagcxResult_t ret = flagcxSuccess;
  struct bootstrapState *state = (struct bootstrapState *)commState;
  int newPeer, newTag;

  // Search unexpected connections first
  int found;
  FLAGCXCHECK(unexpectedDequeue(state, peer, tag, sock, &found));
  if (found)
    return flagcxSuccess;

  // Then look for new connections
  while (1) {
    FLAGCXCHECKGOTO(flagcxSocketInit(sock), ret, fail);
    FLAGCXCHECKGOTO(flagcxSocketAccept(sock, &state->listenSock), ret, fail);
    FLAGCXCHECKGOTO(bootstrapNetRecv(sock, &newPeer, sizeof(int)), ret, fail);
    FLAGCXCHECKGOTO(bootstrapNetRecv(sock, &newTag, sizeof(int)), ret, fail);
    if (newPeer == peer && newTag == tag)
      return flagcxSuccess;
    FLAGCXCHECKGOTO(unexpectedEnqueue(state, newPeer, newTag, sock), ret, fail);
  }
  return flagcxSuccess;
fail:
  FLAGCXCHECK(flagcxSocketClose(sock));
  return ret;
}

// We can't know who we'll receive from, so we need to receive everything at
// once
flagcxResult_t bootstrapRecv(void *commState, int peer, int tag, void *data,
                             int size) {
  flagcxResult_t ret;
  struct flagcxSocket sock;
  FLAGCXCHECK(bootstrapAccept(commState, peer, tag, &sock));
  TRACE(FLAGCX_BOOTSTRAP, "Receiving tag=%d peer=%d size=%d", tag, peer, size);
  FLAGCXCHECKGOTO(bootstrapNetRecv(&sock, ((char *)data), size), ret, exit);
exit:
  FLAGCXCHECK(flagcxSocketClose(&sock));
  return ret;
}

// Collective algorithms, based on bootstrapSend/Recv, and sometimes
// bootstrapConnect/Accept

flagcxResult_t bootstrapRingAllGather(struct flagcxSocket *prevSocket,
                                      struct flagcxSocket *nextSocket, int rank,
                                      int nranks, char *data, int size) {
  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from prev
   * and send previous step's data from (rank-i) to next
   */
  for (int i = 0; i < nranks - 1; i++) {
    size_t rslice = (rank - i - 1 + nranks) % nranks;
    size_t sslice = (rank - i + nranks) % nranks;

    // Send slice to the right, recv slice from the left
    FLAGCXCHECK(bootstrapNetSendRecv(nextSocket, data + sslice * size, size,
                                     prevSocket, data + rslice * size, size));
  }
  return flagcxSuccess;
}

// Another Version of RingAllGather
// The data bytes gather from multiple ranks are uneven.
flagcxResult_t bootstrapRingAllGatherV2(struct flagcxSocket *prevSocket,
                                        struct flagcxSocket *nextSocket,
                                        int rank, int nranks, char *data,
                                        size_t *offset, size_t *length) {
  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from prev
   * and send previous step's data from (rank-i) to next
   */
  uint64_t timers[TIMERS_COLL_COUNT] = {0};
  timers[TIMER_COLL_TOTAL] = clockNano();

  for (int i = 0; i < nranks - 1; i++) {
    size_t rslice = (rank - i - 1 + nranks) % nranks;
    size_t sslice = (rank - i + nranks) % nranks;

    // Send slice to the right, recv slice from the left
    FLAGCXCHECK(bootstrapNetSendRecv(
        nextSocket, (void *)(data + offset[sslice]), length[sslice], prevSocket,
        (void *)(data + offset[rslice]), length[rslice]));
  }
  timers[TIMER_COLL_TOTAL] = clockNano() - timers[TIMER_COLL_TOTAL];
  INFO(FLAGCX_COLL, "COLL timings - %s: rank %d nranks %d total %.2fms.",
       "BootstrapRingAllGatherV2", rank, nranks,
       timers[TIMER_COLL_TOTAL] / 1e6);
  return flagcxSuccess;
}
flagcxResult_t bootstrapAllGather(void *commState, void *allData, int size) {
  struct bootstrapState *state = (struct bootstrapState *)commState;
  int rank = state->rank;
  int nranks = state->nranks;

  TRACE(FLAGCX_INIT, "rank %d nranks %d size %d", rank, nranks, size);

  FLAGCXCHECK(bootstrapRingAllGather(&state->ringRecvSocket,
                                     &state->ringSendSocket, rank, nranks,
                                     (char *)allData, size));

  TRACE(FLAGCX_INIT, "rank %d nranks %d size %d - DONE", rank, nranks, size);
  return flagcxSuccess;
}

flagcxResult_t AllGatherBootstrap(void *commState, const void *sendbuff,
                                  void *recvbuff, size_t sendcount,
                                  flagcxDataType_t datatype) {
  struct bootstrapState *state = (struct bootstrapState *)commState;
  int rank = state->rank;
  // if not in-place
  if (sendbuff !=
      (void *)((char *)recvbuff +
               rank * getFlagcxDataTypeSize(datatype) * sendcount)) {
    memcpy((void *)((char *)recvbuff +
                    rank * getFlagcxDataTypeSize(datatype) * sendcount),
           sendbuff, getFlagcxDataTypeSize(datatype) * sendcount);
  }
  return bootstrapAllGather(commState, recvbuff,
                            getFlagcxDataTypeSize(datatype) * sendcount);
}
/*
 * Reduce-Scatter
 *
 * Reduces data in sendbuff using op operation and leaves reduced result
 * scattered over the devices so that recvbuff on rank i will contain the i-th
 * block of the result.
 *
 * Block size among all ranks are not necessary equal.
 * The i-th block begins at offset[i] and has the length of length[i].
 * The recvbuff of rank i should has the length of at least length[i].
 *
 * In-place operations will happen if recvbuff == sendbuff + offset[rank].
 */
flagcxResult_t bootstrapRingReduceScatter(
    struct flagcxSocket *prevSocket, struct flagcxSocket *nextSocket, int rank,
    int nranks, const char *sendbuff, char *recvbuff, size_t *offset,
    size_t *length, flagcxDataType_t datatype, flagcxRedOp_t op) {
  uint64_t timers[TIMERS_COLL_COUNT] = {0};
  timers[TIMER_COLL_TOTAL] = clockNano();

  // Allocate two temporary buffer with size length[0] to ensure that it can
  // fill any chunk size.
  timers[TIMER_COLL_ALLOC] = clockNano();
  // found the largest chunk
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
  // for iteration 0 -> n-1
  for (int iter = 0; iter < nranks - 1; ++iter) {
    // for each iteration ${iter}
    // 1. rank ${rank} should send data of chunk $(send_chunk_no) to next rank
    // 2. rank ${rank} should recv data of chunk $(recv_chunk_no) from prev rank
    int send_chunk_no = (rank + 2 * nranks - iter - 1) % nranks;
    int recv_chunk_no = (rank + 2 * nranks - iter - 2) % nranks;
    bool needSend = length[send_chunk_no] != 0;
    bool needRecv = length[recv_chunk_no] != 0;

    INFO(FLAGCX_COLL,
         "rank %d nranks %d; iter=%d; send_chunk_no=%d; send_chunk_size=%lu; "
         "needSend=%d; "
         "recv_chunk_no=%d; recv_chunk_size=%lu; needRecv=%d",
         rank, nranks, iter, send_chunk_no, length[send_chunk_no], needSend,
         recv_chunk_no, length[recv_chunk_no], needRecv);
    if (!needSend && !needRecv) {
      continue;
    }

    // step 1: prepare send data if needed
    if (needSend) {
      start = clockNano();
      if (iter == 0) {
        // initial iteration
        memcpy(data_for_send, sendbuff + offset[send_chunk_no],
               length[send_chunk_no]);
      } else {
        std::swap(data_for_send, data_for_recv);
      }
      end = clockNano();
      timers[TIMER_COLL_MEM] += end - start;
    }

    // step 2: exchange data using Send/Recv
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

    // step3 : local reduction for data_for_send & chunk ${recv_chunk_no} if
    // recv something
    //         save result in data_for_recv
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

  // copy the final reduction to recvbuff
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

const size_t MIN_CHUNK_SIZE = 1024 * 1024 * 4; // 4MB

size_t roundUp(size_t value, size_t multiple) {
  size_t remainder = value % multiple;
  if (remainder == 0) {
    return value;
  }
  return value + multiple - remainder;
}

flagcxResult_t bootstrapRingAllReduce(struct flagcxSocket *prevSocket,
                                      struct flagcxSocket *nextSocket, int rank,
                                      int nranks, const char *sendbuff,
                                      char *recvbuff, size_t count,
                                      flagcxDataType_t datatype,
                                      flagcxRedOp_t op) {

  // The ring algorithm works as follows.
  //
  // The given input is split into a number of chunks equal to the
  // number of processes. Once the reducescatter has finished, every
  // process hosts one chunk of reduced output, in sequential order
  // (rank 0 has chunk 0, rank 1 has chunk 1, etc.). As the input may
  // not be divisible by the number of processes, the chunk on the
  // final ranks may have partial output or may be empty.
  //

  size_t size = count * getFlagcxDataTypeSize(datatype);
  size_t ChunkBytes = std::max((size + nranks - 1) / nranks, MIN_CHUNK_SIZE);
  // Ensure that min chunk size is a multiple of the element size.
  ChunkBytes = roundUp(ChunkBytes, getFlagcxDataTypeSize(datatype));
  INFO(FLAGCX_COLL, "rank %d nranks %d; size=%lu; typesize=%lu; ChunkBytes=%lu",
       rank, nranks, size, getFlagcxDataTypeSize(datatype), ChunkBytes);

  // step 1: split the data and prepare offset and length array
  std::vector<size_t> offset(nranks, 0);
  std::vector<size_t> length(nranks, 0);
  for (size_t i = 0; i < nranks; ++i) {
    if (ChunkBytes * i >= size) {
      offset[i] = size;
      length[i] = 0;
      continue;
    }
    offset[i] = ChunkBytes * i;
    length[i] =
        ChunkBytes * (i + 1) >= size ? size - ChunkBytes * i : ChunkBytes;
  }

  // step 2: reduce scatter
  FLAGCXCHECK(bootstrapRingReduceScatter(
      prevSocket, nextSocket, rank, nranks, sendbuff, recvbuff + offset[rank],
      offset.data(), length.data(), datatype, op));

  // step 3: all gather
  FLAGCXCHECK(bootstrapRingAllGatherV2(prevSocket, nextSocket, rank, nranks,
                                       recvbuff, offset.data(), length.data()));
  return flagcxSuccess;
}

flagcxResult_t
bootstrapRingReduce(void *commState, struct flagcxSocket *prevSocket,
                    struct flagcxSocket *nextSocket, int rank, int nranks,
                    const char *sendbuff, char *recvbuff, size_t count,
                    flagcxDataType_t datatype, flagcxRedOp_t op, int root) {

  // The ring algorithm works as follows.
  //
  // The given input is split into a number of chunks equal to the
  // number of processes. Once the reducescatter has finished, every
  // process hosts one chunk of reduced output, in sequential order
  // (rank 0 has chunk 0, rank 1 has chunk 1, etc.). As the input may
  // not be divisible by the number of processes, the chunk on the
  // final ranks may have partial output or may be empty.
  //

  size_t size = count * getFlagcxDataTypeSize(datatype);
  size_t ChunkBytes = std::max((size + nranks - 1) / nranks, MIN_CHUNK_SIZE);
  // Ensure that min chunk size is a multiple of the element size.
  ChunkBytes = roundUp(ChunkBytes, getFlagcxDataTypeSize(datatype));
  INFO(FLAGCX_COLL, "rank %d nranks %d; size=%lu; typesize=%lu; ChunkBytes=%lu",
       rank, nranks, size, getFlagcxDataTypeSize(datatype), ChunkBytes);

  // step 1: split the data and prepare offset and length array
  std::vector<size_t> offset(nranks, 0);
  std::vector<size_t> length(nranks, 0);
  for (size_t i = 0; i < nranks; ++i) {
    if (ChunkBytes * i >= size) {
      offset[i] = size;
      length[i] = 0;
      continue;
    }
    offset[i] = ChunkBytes * i;
    length[i] =
        ChunkBytes * (i + 1) >= size ? size - ChunkBytes * i : ChunkBytes;
  }

  // step 2: reduce scatter
  FLAGCXCHECK(bootstrapRingReduceScatter(
      prevSocket, nextSocket, rank, nranks, sendbuff, recvbuff + offset[rank],
      offset.data(), length.data(), datatype, op));

  // step 3: gather
  const int bootstrapTag = -9993;
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

flagcxResult_t AllReduceBootstrap(void *commState, const void *sendbuff,
                                  void *recvbuff, size_t count,
                                  flagcxDataType_t datatype, flagcxRedOp_t op) {
  struct bootstrapState *state = (struct bootstrapState *)commState;
  int rank = state->rank;
  int nranks = state->nranks;
  if (nranks == 1) {
    if (sendbuff != recvbuff) {
      memcpy(recvbuff, sendbuff, count * getFlagcxDataTypeSize(datatype));
    }
    return flagcxSuccess;
  }
  FLAGCXCHECK(bootstrapRingAllReduce(
      &state->ringRecvSocket, &state->ringSendSocket, rank, nranks,
      (char *)sendbuff, (char *)recvbuff, count, datatype, op));

  return flagcxSuccess;
}

flagcxResult_t ReduceBootstrap(void *commState, const void *sendbuff,
                               void *recvbuff, size_t count,
                               flagcxDataType_t datatype, flagcxRedOp_t op,
                               int root) {
  struct bootstrapState *state = (struct bootstrapState *)commState;
  int rank = state->rank;
  int nranks = state->nranks;

  if (nranks == 1) {
    if (sendbuff != recvbuff) {
      memcpy(recvbuff, sendbuff, count * getFlagcxDataTypeSize(datatype));
    }
    return flagcxSuccess;
  }
  FLAGCXCHECK(bootstrapRingReduce(
      commState, &state->ringRecvSocket, &state->ringSendSocket, rank, nranks,
      (char *)sendbuff, (char *)recvbuff, count, datatype, op, root));
  return flagcxSuccess;
}

flagcxResult_t ReduceScatterBootstrap(void *commState, const void *sendbuff,
                                      void *recvbuff, size_t recvcount,
                                      flagcxDataType_t datatype,
                                      flagcxRedOp_t op) {
  struct bootstrapState *state = (struct bootstrapState *)commState;
  int rank = state->rank;
  int nranks = state->nranks;
  if (nranks == 1) {
    if (sendbuff != recvbuff) {
      memcpy(recvbuff, sendbuff, recvcount * getFlagcxDataTypeSize(datatype));
    }
    return flagcxSuccess;
  }
  // prepare offset, length vector
  std::vector<size_t> offset(nranks, 0);
  std::vector<size_t> length(nranks, 0);
  for (size_t i = 0; i < nranks; ++i) {
    offset[i] = i * recvcount * getFlagcxDataTypeSize(datatype);
    length[i] = recvcount * getFlagcxDataTypeSize(datatype);
  }
  FLAGCXCHECK(bootstrapRingReduceScatter(
      &state->ringRecvSocket, &state->ringSendSocket, rank, nranks,
      (char *)sendbuff, (char *)recvbuff, offset.data(), length.data(),
      datatype, op));
  return flagcxSuccess;
}

flagcxResult_t AlltoAllBootstrap(void *commState, const void *sendbuff,
                                 void *recvbuff, size_t count,
                                 flagcxDataType_t datatype) {
  struct bootstrapState *state = (struct bootstrapState *)commState;
  int rank = state->rank;
  int nranks = state->nranks;
  size_t size = count * getFlagcxDataTypeSize(datatype);

  bool inPlace = (sendbuff == recvbuff);
  char *tmpBuff = nullptr;
  if (inPlace) {
    FLAGCXCHECK(flagcxCalloc(&tmpBuff, size));
  }

  for (int i = 0; i < nranks; ++i) {
    if (i == rank) {
      if (!inPlace) {
        memcpy((void *)((char *)recvbuff + size * i),
               (void *)((char *)sendbuff + size * i), size);
      }
    }
    const int bootstrapTag = -9991;
    if (rank > i) {
      FLAGCXCHECK(bootstrapSend(commState, i, bootstrapTag,
                                (void *)((char *)sendbuff + size * i), size));
      if (inPlace) {
        FLAGCXCHECK(
            bootstrapRecv(commState, i, bootstrapTag, (void *)tmpBuff, size));
        memcpy((void *)((char *)recvbuff + size * i), (void *)tmpBuff, size);
      } else {
        FLAGCXCHECK(bootstrapRecv(commState, i, bootstrapTag,
                                  (void *)((char *)recvbuff + size * i), size));
      }
    } else if (rank < i) {
      if (inPlace) {
        FLAGCXCHECK(
            bootstrapRecv(commState, i, bootstrapTag, (void *)tmpBuff, size));
      } else {
        FLAGCXCHECK(bootstrapRecv(commState, i, bootstrapTag,
                                  (void *)((char *)recvbuff + size * i), size));
      }
      FLAGCXCHECK(bootstrapSend(commState, i, bootstrapTag,
                                (void *)((char *)sendbuff + size * i), size));
      if (inPlace) {
        memcpy((void *)((char *)recvbuff + size * i), (void *)tmpBuff, size);
      }
    }
  }
  free(tmpBuff);
  return flagcxSuccess;
}

flagcxResult_t BroadcastBootstrap(void *commState, const void *sendbuff,
                                  void *recvbuff, size_t sendcount,
                                  flagcxDataType_t datatype, int root) {
  struct bootstrapState *state = (struct bootstrapState *)commState;
  int rank = state->rank;
  int nranks = state->nranks;
  const int bootstrapTag = -9992;
  if (nranks == 1) {
    if (sendbuff != recvbuff) {
      memcpy(recvbuff, sendbuff, getFlagcxDataTypeSize(datatype) * sendcount);
    }
    return flagcxSuccess;
  }
  if (rank == root) {
    if (sendbuff != recvbuff) {
      memcpy(recvbuff, sendbuff, getFlagcxDataTypeSize(datatype) * sendcount);
    }
    // root sends data to all other ranks
    for (int i = 0; i < nranks; ++i) {
      if (i != root) {
        FLAGCXCHECK(bootstrapSend(commState, i, bootstrapTag,
                                  (void *)(sendbuff),
                                  sendcount * getFlagcxDataTypeSize(datatype)));
      }
    }
  } else {
    // all other ranks receive data from root
    FLAGCXCHECK(bootstrapRecv(commState, root, bootstrapTag, recvbuff,
                              sendcount * getFlagcxDataTypeSize(datatype)));
  }
  return flagcxSuccess;
}

flagcxResult_t ScatterBootstrap(void *commState, const void *sendbuff,
                                void *recvbuff, size_t count,
                                flagcxDataType_t datatype, int root) {
  struct bootstrapState *state = (struct bootstrapState *)commState;
  int rank = state->rank;
  int nranks = state->nranks;
  const int bootstrapTag = -9993;
  if (nranks == 1) {
    if (sendbuff != recvbuff) {
      memcpy(recvbuff, sendbuff, getFlagcxDataTypeSize(datatype) * count);
    }
    return flagcxSuccess;
  }

  if (rank == root) {
    // For root process, only copy its own portion of data
    size_t rootOffset = root * count * getFlagcxDataTypeSize(datatype);
    if ((char *)sendbuff + rootOffset != recvbuff) {
      memcpy(recvbuff, (const char *)sendbuff + rootOffset,
             getFlagcxDataTypeSize(datatype) * count);
    }
    // root sends data to all other ranks
    for (int i = 0; i < nranks; ++i) {
      if (i != root) {
        size_t offset = i * count * getFlagcxDataTypeSize(datatype);
        FLAGCXCHECK(bootstrapSend(commState, i, bootstrapTag,
                                  (char *)sendbuff + offset,
                                  count * getFlagcxDataTypeSize(datatype)));
      }
    }
  } else {
    // all other ranks receive data from root
    FLAGCXCHECK(bootstrapRecv(commState, root, bootstrapTag, recvbuff,
                              count * getFlagcxDataTypeSize(datatype)));
  }
  return flagcxSuccess;
}

flagcxResult_t GatherBootstrap(void *commState, const void *sendbuff,
                               void *recvbuff, size_t count,
                               flagcxDataType_t datatype, int root) {
  struct bootstrapState *state = (struct bootstrapState *)commState;
  int rank = state->rank;
  int nranks = state->nranks;
  const int bootstrapTag = -9994;

  if (nranks == 1) {
    if (sendbuff != recvbuff) {
      memcpy(recvbuff, sendbuff, getFlagcxDataTypeSize(datatype) * count);
    }
    return flagcxSuccess;
  }

  if (rank == root) {
    // Handle root's own data
    size_t rootOffset = root * count * getFlagcxDataTypeSize(datatype);
    if (sendbuff != (char *)recvbuff + rootOffset) {
      memcpy((char *)recvbuff + rootOffset, sendbuff,
             getFlagcxDataTypeSize(datatype) * count);
    }

    // Receive data from other ranks
    for (int i = 0; i < nranks; ++i) {
      if (i != root) {
        int offset = i * count * getFlagcxDataTypeSize(datatype);
        FLAGCXCHECK(bootstrapRecv(commState, i, bootstrapTag,
                                  (char *)recvbuff + offset,
                                  count * getFlagcxDataTypeSize(datatype)));
      }
    }
  } else {
    // Non-root ranks send data to root
    FLAGCXCHECK(bootstrapSend(commState, root, bootstrapTag, (void *)sendbuff,
                              count * getFlagcxDataTypeSize(datatype)));
  }
  return flagcxSuccess;
}

flagcxResult_t bootstrapIntraNodeBarrier(void *commState, int *ranks, int rank,
                                         int nranks, int tag) {
  if (nranks == 1)
    return flagcxSuccess;
  TRACE(FLAGCX_INIT, "rank %d nranks %d tag %x - ENTER", rank, nranks, tag);

  /* Simple [intra] process barrier
   *
   * Based on the dissemination algorithm by Debra Hensgen, Raphael Finkel, and
   * Udi Manbet, "Two Algorithms for Barrier Synchronization," International
   * Journal of Parallel Programming, 17(1):1-17, 1988"
   */
  int data[1];
  for (int mask = 1; mask < nranks; mask <<= 1) {
    int src = (rank - mask + nranks) % nranks;
    int dst = (rank + mask) % nranks;
    FLAGCXCHECK(bootstrapSend(commState, ranks ? ranks[dst] : dst, tag, data,
                              sizeof(data)));
    FLAGCXCHECK(bootstrapRecv(commState, ranks ? ranks[src] : src, tag, data,
                              sizeof(data)));
  }

  TRACE(FLAGCX_INIT, "rank %d nranks %d tag %x - DONE", rank, nranks, tag);
  return flagcxSuccess;
}

flagcxResult_t bootstrapBarrier(void *commState, int rank, int nranks,
                                int tag) {
  return bootstrapIntraNodeBarrier(commState, NULL, rank, nranks, tag);
}

// [IntraNode] in-place Broadcast
flagcxResult_t bootstrapIntraNodeBroadcast(void *commState, int *ranks,
                                           int rank, int nranks, int root,
                                           void *bcastData, int size) {
  if (nranks == 1)
    return flagcxSuccess;
  TRACE(FLAGCX_INIT, "rank %d nranks %d root %d size %d - ENTER", rank, nranks,
        root, size);

  if (rank == root) {
    for (int i = 0; i < nranks; i++) {
      if (i != root)
        FLAGCXCHECK(bootstrapSend(commState, ranks ? ranks[i] : i,
                                  /*tag=*/ranks ? ranks[i] : i, bcastData,
                                  size));
    }
  } else {
    FLAGCXCHECK(bootstrapRecv(commState, ranks ? ranks[root] : root,
                              /*tag=*/ranks ? ranks[rank] : rank, bcastData,
                              size));
  }

  TRACE(FLAGCX_INIT, "rank %d nranks %d root %d size %d - DONE", rank, nranks,
        root, size);
  return flagcxSuccess;
}

flagcxResult_t bootstrapBroadcast(void *commState, int rank, int nranks,
                                  int root, void *bcastData, int size) {
  return bootstrapIntraNodeBroadcast(commState, NULL, rank, nranks, root,
                                     bcastData, size);
}

flagcxResult_t bootstrapClose(void *commState) {
  struct bootstrapState *state = (struct bootstrapState *)commState;
  if (state->unexpectedConnections != NULL) {
    unexpectedFree(state);
    if (__atomic_load_n(state->abortFlag, __ATOMIC_RELAXED) == 0) {
      WARN("Unexpected connections are not empty");
      return flagcxInternalError;
    }
  }

  FLAGCXCHECK(flagcxSocketClose(&state->listenSock));
  FLAGCXCHECK(flagcxSocketClose(&state->ringSendSocket));
  FLAGCXCHECK(flagcxSocketClose(&state->ringRecvSocket));

  free(state->peerCommAddresses);
  free(state);

  return flagcxSuccess;
}

flagcxResult_t bootstrapAbort(void *commState) {
  struct bootstrapState *state = (struct bootstrapState *)commState;
  if (commState == NULL)
    return flagcxSuccess;
  FLAGCXCHECK(flagcxSocketClose(&state->listenSock));
  FLAGCXCHECK(flagcxSocketClose(&state->ringSendSocket));
  FLAGCXCHECK(flagcxSocketClose(&state->ringRecvSocket));
  free(state->peerCommAddresses);
  free(state->peerProxyAddresses);
  free(state);
  return flagcxSuccess;
}
// AlltoALlv require sendbuff and recvbuff not overlap
flagcxResult_t AlltoAllvBootstrap(void *commState, const void *sendbuff,
                                  size_t *sendcounts, size_t *sdispls,
                                  void *recvbuff, size_t *recvcounts,
                                  size_t *rdispls, flagcxDataType_t datatype) {
  struct bootstrapState *state = (struct bootstrapState *)commState;
  int rank = state->rank;
  int nranks = state->nranks;
  size_t typeSize = getFlagcxDataTypeSize(datatype);

  for (int i = 0; i < nranks; ++i) {
    if (i == rank) {
      memcpy((void *)((char *)recvbuff + rdispls[i] * typeSize),
             (void *)((char *)sendbuff + sdispls[i] * typeSize),
             sendcounts[i] * typeSize);
    }
    const int bootstrapTag = -9995; // Suggest making this unique if possible
    if (rank > i) {
      // Send to rank i
      FLAGCXCHECK(
          bootstrapSend(commState, i, bootstrapTag,
                        (void *)((char *)sendbuff + sdispls[i] * typeSize),
                        sendcounts[i] * typeSize));
      // Recv from rank i
      FLAGCXCHECK(
          bootstrapRecv(commState, i, bootstrapTag,
                        (void *)((char *)recvbuff + rdispls[i] * typeSize),
                        recvcounts[i] * typeSize));
    } else if (rank < i) {
      // Receive from rank i
      FLAGCXCHECK(
          bootstrapRecv(commState, i, bootstrapTag,
                        (void *)((char *)recvbuff + rdispls[i] * typeSize),
                        recvcounts[i] * typeSize));
      // Send to rank i
      FLAGCXCHECK(
          bootstrapSend(commState, i, bootstrapTag,
                        (void *)((char *)sendbuff + sdispls[i] * typeSize),
                        sendcounts[i] * typeSize));
    }
  }
  return flagcxSuccess;
}
