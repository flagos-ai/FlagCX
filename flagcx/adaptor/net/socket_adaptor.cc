/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "adaptor.h"
#include "comm.h"
#include "core.h"
#include "net.h"
#include "param.h"
#include "socket.h"
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <pthread.h>
#include <stdlib.h>

static int flagcxNetIfs = -1;
struct flagcxNetSocketDev {
  union flagcxSocketAddress addr;
  char devName[MAX_IF_NAME_SIZE];
  char *pciPath;
};
static struct flagcxNetSocketDev flagcxNetSocketDevs[MAX_IFS];

pthread_mutex_t flagcxNetSocketLock = PTHREAD_MUTEX_INITIALIZER;

static flagcxResult_t flagcxNetSocketGetPciPath(char *devName, char **pciPath) {
  char devicePath[PATH_MAX];
  snprintf(devicePath, PATH_MAX, "/sys/class/net/%s/device", devName);
  // May return NULL if the file doesn't exist.
  *pciPath = realpath(devicePath, NULL);
  return flagcxSuccess;
}

flagcxResult_t flagcxNetSocketInit() {
  if (flagcxNetIfs == -1) {
    pthread_mutex_lock(&flagcxNetSocketLock);
    if (flagcxNetIfs == -1) {
      char names[MAX_IF_NAME_SIZE * MAX_IFS];
      union flagcxSocketAddress addrs[MAX_IFS];
      flagcxNetIfs =
          flagcxFindInterfaces(names, addrs, MAX_IF_NAME_SIZE, MAX_IFS);
      if (flagcxNetIfs <= 0) {
        WARN("NET/Socket : no interface found");
        return flagcxInternalError;
      } else {
#define MAX_LINE_LEN (2047)
        char line[MAX_LINE_LEN + 1];
        char addrline[SOCKET_NAME_MAXLEN + 1];
        line[0] = '\0';
        addrline[SOCKET_NAME_MAXLEN] = '\0';
        for (int i = 0; i < flagcxNetIfs; i++) {
          strcpy(flagcxNetSocketDevs[i].devName, names + i * MAX_IF_NAME_SIZE);
          memcpy(&flagcxNetSocketDevs[i].addr, addrs + i,
                 sizeof(union flagcxSocketAddress));
          FLAGCXCHECK(flagcxNetSocketGetPciPath(
              flagcxNetSocketDevs[i].devName, &flagcxNetSocketDevs[i].pciPath));
          snprintf(line + strlen(line), MAX_LINE_LEN - strlen(line),
                   " [%d]%s:%s", i, names + i * MAX_IF_NAME_SIZE,
                   flagcxSocketToString(&addrs[i], addrline));
        }
        line[MAX_LINE_LEN] = '\0';
        INFO(FLAGCX_INIT | FLAGCX_NET, "NET/Socket : Using%s", line);
      }
    }
    pthread_mutex_unlock(&flagcxNetSocketLock);
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxNetSocketDevices(int *ndev) {
  *ndev = flagcxNetIfs;
  return flagcxSuccess;
}

static flagcxResult_t flagcxNetSocketGetSpeed(char *devName, int *speed) {
  *speed = 0;
  char speedPath[PATH_MAX];
  sprintf(speedPath, "/sys/class/net/%s/speed", devName);
  int fd = open(speedPath, O_RDONLY);
  if (fd != -1) {
    char speedStr[] = "        ";
    if (read(fd, speedStr, sizeof(speedStr) - 1) > 0) {
      *speed = strtol(speedStr, NULL, 0);
    }
    close(fd);
  }
  if (*speed <= 0) {
    INFO(FLAGCX_NET, "Could not get speed from %s. Defaulting to 10 Gbps.",
         speedPath);
    *speed = 10000;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxNetSocketGetProperties(int dev, void *props) {
  flagcxNetProperties_t *netProps = (flagcxNetProperties_t *)props;
  netProps->name = flagcxNetSocketDevs[dev].devName;
  netProps->pciPath = flagcxNetSocketDevs[dev].pciPath;
  netProps->guid = dev;
  netProps->ptrSupport = FLAGCX_PTR_HOST;
  netProps->regIsGlobal = 0;
  FLAGCXCHECK(flagcxNetSocketGetSpeed(netProps->name, &netProps->speed));
  netProps->latency = 0; // Not set
  netProps->port = 0;
  netProps->maxComms = 65536;
  netProps->maxRecvs = 1;
  netProps->netDeviceType = FLAGCX_NET_DEVICE_HOST;
  netProps->netDeviceVersion = FLAGCX_NET_DEVICE_INVALID_VERSION;
  return flagcxSuccess;
}

/* Communication functions */

#define MAX_SOCKETS 64
#define MAX_THREADS 16
#define MAX_REQUESTS FLAGCX_NET_MAX_REQUESTS
#define MIN_CHUNKSIZE (64 * 1024)

FLAGCX_PARAM(SocketNsocksPerThread, "NSOCKS_PERTHREAD", -2);
FLAGCX_PARAM(SocketNthreads, "SOCKET_NTHREADS", -2);

enum flagcxNetSocketCommState {
  flagcxNetSocketCommStateStart = 0,
  flagcxNetSocketCommStateConnect = 1,
  flagcxNetSocketCommStateAccept = 3,
  flagcxNetSocketCommStateSend = 4,
  flagcxNetSocketCommStateRecv = 5,
};

struct flagcxNetSocketCommStage {
  enum flagcxNetSocketCommState state;
  uint8_t iteration;
  struct flagcxSocket *sock;
  struct flagcxNetSocketComm *comm;
};

struct flagcxNetSocketHandle {
  union flagcxSocketAddress connectAddr;
  uint64_t magic; // random number to help debugging
  int nSocks;
  int nThreads;
  struct flagcxNetSocketCommStage stage;
};

struct flagcxNetSocketTask {
  int op;
  void *data;
  int size;
  struct flagcxSocket *sock;
  int offset;
  int used;
  flagcxResult_t result;
};

struct flagcxNetSocketRequest {
  int op;
  void *data;
  int size;
  struct flagcxSocket *ctrlSock;
  int offset;
  int used;
  struct flagcxNetSocketComm *comm;
  struct flagcxNetSocketTask *tasks[MAX_SOCKETS];
  int nSubs;
};

struct flagcxNetSocketTaskQueue {
  int next;
  int len;
  struct flagcxNetSocketTask *tasks;
};

struct flagcxNetSocketThreadResources {
  struct flagcxNetSocketTaskQueue threadTaskQueue;
  int stop;
  struct flagcxNetSocketComm *comm;
  pthread_mutex_t threadLock;
  pthread_cond_t threadCond;
};

struct flagcxNetSocketListenComm {
  struct flagcxSocket sock;
  struct flagcxNetSocketCommStage stage;
  int nSocks;
  int nThreads;
  int dev;
};

struct flagcxNetSocketComm {
  struct flagcxSocket ctrlSock;
  struct flagcxSocket socks[MAX_SOCKETS];
  int dev;
  int cudaDev;
  int nSocks;
  int nThreads;
  int nextSock;
  struct flagcxNetSocketRequest requests[MAX_REQUESTS];
  pthread_t helperThread[MAX_THREADS];
  struct flagcxNetSocketThreadResources threadResources[MAX_THREADS];
};

void *persistentSocketThread(void *args_) {
  struct flagcxNetSocketThreadResources *resource =
      (struct flagcxNetSocketThreadResources *)args_;
  struct flagcxNetSocketComm *comm = resource->comm;
  struct flagcxNetSocketTaskQueue *myQueue = &resource->threadTaskQueue;
  int nSocksPerThread = comm->nSocks / comm->nThreads;
  while (1) {
    int idle = 1;
    int mark = myQueue->next; // mark newest task seen
    for (int i = 0; i < myQueue->len; i += nSocksPerThread) {
      int repeat;
      do {
        repeat = 0;
        for (int j = 0; j < nSocksPerThread; j++) {
          struct flagcxNetSocketTask *r = myQueue->tasks + i + j;
          if (r != NULL && r->used == 1 && r->offset < r->size) {
            r->result = flagcxSocketProgress(r->op, r->sock, r->data, r->size,
                                             &r->offset);
            if (r->result != flagcxSuccess) {
              WARN("NET/Socket : socket progress error");
              return NULL;
            }
            idle = 0;
            if (r->offset < r->size)
              repeat = 1;
          }
        }
      } while (repeat);
    }
    if (idle) {
      pthread_mutex_lock(&resource->threadLock);
      while (mark == myQueue->next &&
             resource->stop == 0) { // no new tasks, wait
        pthread_cond_wait(&resource->threadCond, &resource->threadLock);
      }
      pthread_mutex_unlock(&resource->threadLock);
    }
    if (resource->stop)
      return NULL;
  }
}

flagcxResult_t flagcxNetSocketGetNsockNthread(int dev, int *ns, int *nt) {
  int nSocksPerThread = flagcxParamSocketNsocksPerThread();
  int nThreads = flagcxParamSocketNthreads();
  if (nThreads > MAX_THREADS) {
    WARN("NET/Socket : FLAGCX_SOCKET_NTHREADS is greater than the maximum "
         "allowed, setting to %d",
         MAX_THREADS);
    nThreads = MAX_THREADS;
  }
  if (nThreads == -2 || nSocksPerThread == -2) {
    // Auto-detection
    int autoNt = 0, autoNs = 1; // By default, we only use the main thread and
                                // do not spawn extra threads
    char vendorPath[PATH_MAX];
    snprintf(vendorPath, PATH_MAX, "/sys/class/net/%s/device/vendor",
             flagcxNetSocketDevs[dev].devName);
    char *rPath = realpath(vendorPath, NULL);
    int fd = open(rPath, O_RDONLY);
    free(rPath);
    if (fd == -1) {
      // Could not find device vendor. This is handled silently so
      // we don't want to print an INFO error.
      TRACE(FLAGCX_NET, "Open of %s failed : %s", vendorPath, strerror(errno));
      goto end;
    }
    char vendor[7];
    strncpy(vendor, "0x0000", 7);
    int len;
    SYSCHECKVAL(read(fd, vendor, 6), "read", len);
    SYSCHECK(close(fd), "close");
    if (strcmp(vendor, "0x1d0f") == 0) { // AWS
      autoNt = 2;
      autoNs = 8;
    } else if (strcmp(vendor, "0x1ae0") == 0) { // GCP
      autoNt = 4;
      autoNs = 1;
    }
  end:
    if (nThreads == -2)
      nThreads = autoNt;
    if (nSocksPerThread == -2)
      nSocksPerThread = autoNs;
  }
  int nSocks = nSocksPerThread * nThreads;
  if (nSocks > MAX_SOCKETS) {
    nSocksPerThread = MAX_SOCKETS / nThreads;
    WARN("NET/Socket : the total number of sockets is greater than the maximum "
         "allowed, setting FLAGCX_NSOCKS_PERTHREAD to %d",
         nSocksPerThread);
    nSocks = nSocksPerThread * nThreads;
  }
  *ns = nSocks;
  *nt = nThreads;
  if (nSocks > 0)
    INFO(FLAGCX_INIT, "NET/Socket: Using %d threads and %d sockets per thread",
         nThreads, nSocksPerThread);
  return flagcxSuccess;
}

flagcxResult_t flagcxNetSocketListen(int dev, void *opaqueHandle,
                                     void **listenComm) {
  if (dev < 0 ||
      dev >= flagcxNetIfs) { // data transfer socket is based on specified dev
    return flagcxInternalError;
  }
  struct flagcxNetSocketHandle *handle =
      (struct flagcxNetSocketHandle *)opaqueHandle;
  memset(handle, 0, sizeof(struct flagcxNetSocketHandle));
  static_assert(sizeof(struct flagcxNetSocketHandle) <=
                    FLAGCX_NET_HANDLE_MAXSIZE,
                "flagcxNetSocketHandle size too large");
  struct flagcxNetSocketListenComm *comm;
  FLAGCXCHECK(flagcxCalloc(&comm, 1));
  handle->magic = FLAGCX_SOCKET_MAGIC;
  FLAGCXCHECK(flagcxSocketInit(&comm->sock, &flagcxNetSocketDevs[dev].addr,
                               handle->magic, flagcxSocketTypeNetSocket, NULL,
                               1));
  FLAGCXCHECK(flagcxSocketListen(&comm->sock));
  FLAGCXCHECK(flagcxSocketGetAddr(&comm->sock, &handle->connectAddr));
  FLAGCXCHECK(
      flagcxNetSocketGetNsockNthread(dev, &comm->nSocks, &comm->nThreads));
  handle->nSocks = comm->nSocks;
  handle->nThreads = comm->nThreads;
  comm->dev = dev;
  *listenComm = comm;
  return flagcxSuccess;
}

flagcxResult_t flagcxNetSocketConnect(int dev, void *opaqueHandle,
                                      void **sendComm) {
  if (dev < 0 ||
      dev >= flagcxNetIfs) { // data transfer socket is based on specified dev
    return flagcxInternalError;
  }

  int ready;
  struct flagcxNetSocketHandle *handle =
      (struct flagcxNetSocketHandle *)opaqueHandle;
  struct flagcxNetSocketCommStage *stage = &handle->stage;
  struct flagcxNetSocketComm *comm = stage->comm;
  uint8_t i = stage->iteration;
  struct flagcxSocket *sock = stage->sock;
  *sendComm = NULL;

  if (stage->state == flagcxNetSocketCommStateConnect)
    goto socket_connect_check;
  if (stage->state == flagcxNetSocketCommStateSend)
    goto socket_send;

  FLAGCXCHECK(flagcxCalloc(&comm, 1));
  stage->comm = comm;
  comm->nSocks = handle->nSocks;
  comm->nThreads = handle->nThreads;
  comm->dev = dev;
  for (; i < comm->nSocks + 1; i++) {
    sock = (i == comm->nSocks) ? &comm->ctrlSock : comm->socks + i;
    FLAGCXCHECK(flagcxSocketInit(sock, &handle->connectAddr, handle->magic,
                                 flagcxSocketTypeNetSocket, NULL, 1));

    stage->sock = sock;
    stage->state = flagcxNetSocketCommStateConnect;
    stage->iteration = i;
    FLAGCXCHECK(flagcxSocketConnect(sock));

  socket_connect_check:
    FLAGCXCHECK(flagcxSocketReady(sock, &ready));
    if (!ready)
      return flagcxSuccess;
    stage->state = flagcxNetSocketCommStateSend;

  socket_send:
    int done = 0;
    FLAGCXCHECK(flagcxSocketProgress(FLAGCX_SOCKET_SEND, sock, &i,
                                     sizeof(uint8_t), &done));
    if (done == 0)
      return flagcxSuccess;
  }
  *sendComm = comm;
  return flagcxSuccess;
}

flagcxResult_t flagcxNetSocketAccept(void *listenComm, void **recvComm) {
  struct flagcxNetSocketListenComm *lComm =
      (struct flagcxNetSocketListenComm *)listenComm;
  struct flagcxNetSocketCommStage *stage = &lComm->stage;
  struct flagcxNetSocketComm *rComm = stage->comm;
  uint8_t i = stage->iteration;
  struct flagcxSocket *sock = stage->sock;
  int ready;

  *recvComm = NULL;
  if (stage->state == flagcxNetSocketCommStateAccept)
    goto socket_accept_check;
  if (stage->state == flagcxNetSocketCommStateRecv)
    goto socket_recv;

  FLAGCXCHECK(flagcxCalloc(&rComm, 1));
  stage->comm = rComm;
  rComm->nSocks = lComm->nSocks;
  rComm->nThreads = lComm->nThreads;
  rComm->dev = lComm->dev;
  for (; i < rComm->nSocks + 1; i++) {
    uint8_t sendSockIdx;

    FLAGCXCHECK(flagcxCalloc(&sock, 1));
    FLAGCXCHECK(flagcxSocketInit(sock));
    stage->sock = sock;
    stage->state = flagcxNetSocketCommStateAccept;
    stage->iteration = i;
    FLAGCXCHECK(flagcxSocketAccept(sock, &lComm->sock));

  socket_accept_check:
    FLAGCXCHECK(flagcxSocketReady(sock, &ready));
    if (!ready)
      return flagcxSuccess;

    stage->state = flagcxNetSocketCommStateRecv;
  socket_recv:
    int done = 0;
    FLAGCXCHECK(flagcxSocketProgress(FLAGCX_SOCKET_RECV, sock, &sendSockIdx,
                                     sizeof(uint8_t), &done));
    if (done == 0)
      return flagcxSuccess;

    if (sendSockIdx == rComm->nSocks)
      memcpy(&rComm->ctrlSock, sock, sizeof(struct flagcxSocket));
    else
      memcpy(rComm->socks + sendSockIdx, sock, sizeof(struct flagcxSocket));
    free(sock);
  }
  *recvComm = rComm;

  /* reset lComm state */
  stage->state = flagcxNetSocketCommStateStart;
  stage->iteration = 0;
  stage->sock = NULL;
  stage->comm = NULL;
  return flagcxSuccess;
}

flagcxResult_t flagcxNetSocketGetRequest(struct flagcxNetSocketComm *comm,
                                         int op, void *data, size_t size,
                                         struct flagcxNetSocketRequest **req) {
  for (int i = 0; i < MAX_REQUESTS; i++) {
    struct flagcxNetSocketRequest *r = comm->requests + i;
    if (r->used == 0) {
      r->op = op;
      r->data = data;
      r->size = size;
      r->ctrlSock = &comm->ctrlSock;
      r->used = 1;
      r->comm = comm;
      r->nSubs = 0;
      *req = r;
      return flagcxSuccess;
    }
  }
  WARN("NET/Socket : unable to allocate requests");
  return flagcxInternalError;
}

flagcxResult_t flagcxNetSocketGetTask(struct flagcxNetSocketComm *comm, int op,
                                      void *data, int size,
                                      struct flagcxNetSocketTask **req) {
  int tid = comm->nextSock % comm->nThreads;
  struct flagcxNetSocketThreadResources *res = comm->threadResources + tid;
  struct flagcxNetSocketTaskQueue *queue = &res->threadTaskQueue;
  // create helper threads and prepare per-thread task queue
  if (queue->tasks == NULL) {
    // each request can be divided up to nSocks tasks, and
    // these tasks are distributed to nThreads threads,
    // we need to make sure each thread queue has enough slots for MAX_REQUESTS
    queue->len = MAX_REQUESTS * DIVUP(comm->nSocks, comm->nThreads);
    FLAGCXCHECK(flagcxCalloc(&queue->tasks, queue->len));
    queue->next = 0;
    res->comm = comm;
    pthread_mutex_init(&res->threadLock, NULL);
    pthread_cond_init(&res->threadCond, NULL);
    pthread_create(comm->helperThread + tid, NULL, persistentSocketThread, res);
    flagcxSetThreadName(comm->helperThread[tid], "FLAGCX Sock%c%1u%2u%2u",
                        op == FLAGCX_SOCKET_SEND ? 'S' : 'R', comm->dev, tid,
                        comm->cudaDev);
  }
  struct flagcxNetSocketTask *r = queue->tasks + queue->next;
  if (r->used == 0) {
    r->op = op;
    r->data = data;
    r->size = size;
    r->sock = comm->socks + comm->nextSock;
    r->offset = 0;
    r->result = flagcxSuccess;
    comm->nextSock = (comm->nextSock + 1) % comm->nSocks;
    r->used = 1;
    *req = r;
    pthread_mutex_lock(&res->threadLock);
    queue->next = (queue->next + 1) % queue->len;
    pthread_cond_signal(&res->threadCond);
    pthread_mutex_unlock(&res->threadLock);
    return flagcxSuccess;
  }
  WARN("NET/Socket : unable to allocate subtasks");
  return flagcxInternalError;
}

flagcxResult_t flagcxNetSocketTest(void *request, int *done, int *size) {
  *done = 0;
  struct flagcxNetSocketRequest *r = (struct flagcxNetSocketRequest *)request;
  if (r == NULL) {
    WARN("NET/Socket : test called with NULL request");
    return flagcxInternalError;
  }
  if (r->used == 1) { /* try to send/recv size */
    int data = r->size;
    int offset = 0;
    FLAGCXCHECK(
        flagcxSocketProgress(r->op, r->ctrlSock, &data, sizeof(int), &offset));

    if (offset == 0)
      return flagcxSuccess; /* Not ready -- retry later */

    // Not sure we could ever receive less than 4 bytes, but just in case ...
    if (offset < sizeof(int))
      FLAGCXCHECK(
          flagcxSocketWait(r->op, r->ctrlSock, &data, sizeof(int), &offset));

    // Check size is less or equal to the size provided by the user
    if (r->op == FLAGCX_SOCKET_RECV && data > r->size) {
      char line[SOCKET_NAME_MAXLEN + 1];
      union flagcxSocketAddress addr;
      flagcxSocketGetAddr(r->ctrlSock, &addr);
      WARN(
          "NET/Socket : peer %s message truncated : receiving %d bytes instead of %d. If you believe your socket network is in healthy state, \
          there may be a mismatch in collective sizes or environment settings (e.g. FLAGCX_PROTO, FLAGCX_ALGO) between ranks",
          flagcxSocketToString(&addr, line), data, r->size);
      return flagcxInvalidUsage;
    }
    r->size = data;
    r->offset = 0;
    r->used = 2; // done exchanging size
    // divide into subtasks
    int chunkOffset = 0, i = 0;
    if (r->comm->nSocks > 0) {
      // each request can be divided up to nSocks tasks
      int taskSize = std::max(MIN_CHUNKSIZE, DIVUP(r->size, r->comm->nSocks));
      while (chunkOffset < r->size) {
        int chunkSize = std::min(taskSize, r->size - chunkOffset);
        FLAGCXCHECK(flagcxNetSocketGetTask(r->comm, r->op,
                                           (char *)(r->data) + chunkOffset,
                                           chunkSize, r->tasks + i++));
        chunkOffset += chunkSize;
      }
    }
    r->nSubs = i;
  }
  if (r->used == 2) { // already exchanged size
    if (r->nSubs > 0) {
      int nCompleted = 0;
      for (int i = 0; i < r->nSubs; i++) {
        struct flagcxNetSocketTask *sub = r->tasks[i];
        if (sub->result != flagcxSuccess)
          return sub->result;
        if (sub->offset == sub->size)
          nCompleted++;
      }
      if (nCompleted == r->nSubs) {
        if (size)
          *size = r->size;
        *done = 1;
        r->used = 0;
        for (int i = 0; i < r->nSubs; i++) {
          struct flagcxNetSocketTask *sub = r->tasks[i];
          sub->used = 0;
        }
      }
    } else { // progress request using main thread
      if (r->offset < r->size) {
        FLAGCXCHECK(flagcxSocketProgress(r->op, r->ctrlSock, r->data, r->size,
                                         &r->offset));
      }
      if (r->offset == r->size) {
        if (size)
          *size = r->size;
        *done = 1;
        r->used = 0;
      }
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxNetSocketRegMr(void *comm, void *data, size_t size,
                                    int type, int mrFlags, void **mhandle) {
  (void)mrFlags;
  return (type != FLAGCX_PTR_HOST) ? flagcxInternalError : flagcxSuccess;
}

flagcxResult_t flagcxNetSocketDeregMr(void *comm, void *mhandle) {
  return flagcxSuccess;
}

flagcxResult_t flagcxNetSocketIsend(void *sendComm, void *data, size_t size,
                                    int tag, void *mhandle, void *phandle,
                                    void **request) {
  struct flagcxNetSocketComm *comm = (struct flagcxNetSocketComm *)sendComm;
  FLAGCXCHECK(
      flagcxNetSocketGetRequest(comm, FLAGCX_SOCKET_SEND, data, size,
                                (struct flagcxNetSocketRequest **)request));
  return flagcxSuccess;
}

flagcxResult_t flagcxNetSocketIrecv(void *recvComm, int n, void **data,
                                    size_t *sizes, int *tags, void **mhandles,
                                    void **phandles, void **request) {
  struct flagcxNetSocketComm *comm = (struct flagcxNetSocketComm *)recvComm;
  if (n != 1)
    return flagcxInternalError;
  FLAGCXCHECK(
      flagcxNetSocketGetRequest(comm, FLAGCX_SOCKET_RECV, data[0], sizes[0],
                                (struct flagcxNetSocketRequest **)request));
  return flagcxSuccess;
}

flagcxResult_t flagcxNetSocketIflush(void *recvComm, int n, void **data,
                                     int *sizes, void **mhandles,
                                     void **request) {
  // We don't support CUDA pointers, so we don't need a flush operation
  return flagcxInternalError;
}

flagcxResult_t flagcxNetSocketCloseListen(void *opaqueComm) {
  struct flagcxNetSocketListenComm *comm =
      (struct flagcxNetSocketListenComm *)opaqueComm;
  if (comm) {
    int ready;
    FLAGCXCHECK(flagcxSocketReady(&comm->sock, &ready));
    if (ready)
      FLAGCXCHECK(flagcxSocketClose(&comm->sock));
    free(comm);
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxNetSocketClose(void *opaqueComm) {
  struct flagcxNetSocketComm *comm = (struct flagcxNetSocketComm *)opaqueComm;
  if (comm) {
    for (int i = 0; i < comm->nThreads; i++) {
      struct flagcxNetSocketThreadResources *res = comm->threadResources + i;
      if (comm->helperThread[i]) {
        pthread_mutex_lock(&res->threadLock);
        res->stop = 1;
        pthread_cond_signal(&res->threadCond);
        pthread_mutex_unlock(&res->threadLock);
        pthread_join(comm->helperThread[i], NULL);
      }
      free(res->threadTaskQueue.tasks);
    }
    int ready;
    FLAGCXCHECK(flagcxSocketReady(&comm->ctrlSock, &ready));
    if (ready)
      FLAGCXCHECK(flagcxSocketClose(&comm->ctrlSock));
    for (int i = 0; i < comm->nSocks; i++) {
      FLAGCXCHECK(flagcxSocketReady(&comm->socks[i], &ready));
      if (ready)
        FLAGCXCHECK(flagcxSocketClose(&comm->socks[i]));
    }
    free(comm);
  }
  return flagcxSuccess;
}

flagcxNetAdaptor flagcxNetSocket = {
    // Basic functions
    "Socket", flagcxNetSocketInit, flagcxNetSocketDevices,
    flagcxNetSocketGetProperties,
    NULL, // reduceSupport - not implemented
    NULL, // getDeviceMr - not implemented
    NULL, // irecvConsumed - not implemented

    // Setup functions
    flagcxNetSocketListen, flagcxNetSocketConnect, flagcxNetSocketAccept,
    flagcxNetSocketClose, // closeSend
    flagcxNetSocketClose, // closeRecv (same as closeSend for socket)
    flagcxNetSocketCloseListen,

    // Memory region functions
    flagcxNetSocketRegMr,
    NULL, // regMrDmaBuf - No DMA-BUF support
    flagcxNetSocketDeregMr,

    // Two-sided functions
    flagcxNetSocketIsend, flagcxNetSocketIrecv, flagcxNetSocketIflush,
    flagcxNetSocketTest,

    // One-sided functions
    NULL, // iput - not supported on socket
    NULL, // iget - not supported on socket
    NULL, // iputSignal - not supported on socket

    // Device name lookup
    NULL, // getDevFromName
};
