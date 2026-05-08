/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX P2P Engine — implements the flagcx_p2p.h API.
 *
 * Architecture: thin C-shim over the IBRC P2P net adaptor
 * (flagcxNetIbP2p) + P2P topo manager. Mirrors the structure of UCCL's
 * uccl_engine.cc so that a NIXL FlagCX backend plugin can wrap it in
 * exactly the same way the NIXL UCCL plugin wraps uccl_engine.
 ************************************************************************/

#include "flagcx_p2p.h"

#include "adaptor.h"
#include "flagcx_net.h"
#include "flagcx_net_adaptor.h"
#include "ib_common.h"
#include "p2p_topo.h"
#include "socket.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <poll.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#if defined(__linux__)
#include <sys/epoll.h>
#endif
#include <unistd.h>

extern struct flagcxNetAdaptor flagcxNetIbP2p;

struct FlagcxP2pMrHandleView {
  uintptr_t baseVa;
  uint32_t lkey;
  uint32_t rkey;
  void *mr;
  int ibDevN;
};

struct FlagcxP2pListenHandleView {
  union flagcxSocketAddress connectAddr;
  uint64_t magic;
};
static_assert(sizeof(FlagcxP2pListenHandleView) <= FLAGCX_NET_HANDLE_MAXSIZE,
              "listen handle must fit in FLAGCX_NET_HANDLE_MAXSIZE");

struct FlagcxP2pCommView {
  int ibDevN;
  struct flagcxIbNetCommDevBase base;
  struct flagcxIbQp qp;
  struct flagcxSocket sock;
};

enum {
  FLAGCX_P2P_MAX_NOTIF_PEERS = 64,
  FLAGCX_P2P_IPC_HANDLE_BYTES = 64,
  FLAGCX_P2P_NOTIF_MAGIC = 0xDEADDEADu,
  FLAGCX_P2P_CTRL_FLAG_LOCAL = 1u << 0,
  FLAGCX_P2P_CTRL_FLAG_SAME_PROCESS = 1u << 1,
  FLAGCX_P2P_IPC_FLAG_CUDA = 1u << 0,
};

struct FlagcxP2pCtrlMeta {
  int32_t gpuIdx;
  int32_t notifPort;
  uint32_t flags;
  uint32_t reserved;
};
static_assert(sizeof(FlagcxP2pCtrlMeta) == 16,
              "FlagcxP2pCtrlMeta size must be stable");

struct FlagcxP2pIpcInfo {
  alignas(8) char handleData[FLAGCX_P2P_IPC_HANDLE_BYTES];
  uint64_t baseAddr;
  uint64_t offset;
  uint64_t size;
  uint32_t flags;
  uint32_t handleSize;
  char padding[32];
};
static_assert(sizeof(FlagcxP2pIpcInfo) == FLAGCX_P2P_IPC_INFO_SIZE,
              "FlagcxP2pIpcInfo size must match FLAGCX_P2P_IPC_INFO_SIZE");

struct FlagcxP2pNotifWireMsg {
  uint32_t magic;
  uint32_t reserved;
  FlagcxP2pNotifyMsg payload;
};

struct FlagcxP2pNotifConn {
  int fd;
  union flagcxSocketAddress addr;
  std::vector<char> inBuf;
};

struct FlagcxP2pListener {
  void *listenComm;
  char handle[FLAGCX_NET_HANDLE_MAXSIZE];
};

struct FlagcxP2pEngine {
  struct flagcxNetAdaptor *adaptor;
  struct flagcxP2pTopoManager *topoMgr;
  int nDevs;
  int localGpuIdx;
  FlagcxP2pListener listeners[MAX_IB_DEVS];

  struct flagcxSocket notifListenSock;
  bool notifListenActive;
  int notifListenPort;
#if defined(__linux__)
  int notifEpollFd;
#endif
  std::thread notifThread;
  std::atomic<bool> stopNotif;
  std::unordered_map<int, FlagcxP2pNotifConn> notifPeers;
  std::mutex notifPeerMutex;
};

struct FlagcxP2pConn {
  FlagcxP2pEngine *engine;
  void *sendComm;
  void *recvComm;
  int netDev;
  int remoteGpuIdx;
  int remoteNotifPort;
  bool isLocal;
  bool sameProcess;
  struct flagcxSocket notifSock;
  bool notifSockConnected;
};

struct FlagcxP2pMemRegEntry {
  FlagcxP2pMr mrId;
  void *mhandle;
  uintptr_t baseAddr;
  size_t size;
  int ibDevN;
  int ptrType;
  bool hasIpc;
  uint32_t ipcHandleSize;
  alignas(8) char ipcHandle[FLAGCX_P2P_IPC_HANDLE_BYTES];
  char descBuf[FLAGCX_P2P_DESC_SIZE];
};

enum FlagcxP2pXferKind {
  FLAGCX_P2P_XFER_NET = 0,
  FLAGCX_P2P_XFER_IPC = 1,
};

struct FlagcxP2pXfer {
  FlagcxP2pXferKind kind;
  std::vector<void *> requests;
  FlagcxP2pConn *conn;
  int total;
  int completed;
  flagcxStream_t stream;
  flagcxEvent_t event;
  std::vector<void *> openedIpcPtrs;
};

static std::vector<FlagcxP2pNotifyMsg> gNotifyList;
static std::mutex gNotifyMutex;

static std::unordered_map<uintptr_t, FlagcxP2pMemRegEntry> gMemRegInfo;
static std::mutex gMemMutex;
static uint64_t gNextMrId = 1;

static std::unordered_map<uint64_t, FlagcxP2pXfer> gXferMap;
static std::mutex gXferMutex;
static uint64_t gNextXferId = 1;

static bool findMemReg(uintptr_t addr, FlagcxP2pMemRegEntry *out) {
  for (std::unordered_map<uintptr_t, FlagcxP2pMemRegEntry>::const_iterator it =
           gMemRegInfo.begin();
       it != gMemRegInfo.end(); ++it) {
    const uintptr_t base = it->first;
    const FlagcxP2pMemRegEntry &entry = it->second;
    if (addr >= base && addr < base + entry.size) {
      if (out)
        *out = entry;
      return true;
    }
  }
  return false;
}

static FlagcxP2pMemRegEntry *findMemRegByMr(FlagcxP2pMr mr) {
  for (std::unordered_map<uintptr_t, FlagcxP2pMemRegEntry>::iterator it =
           gMemRegInfo.begin();
       it != gMemRegInfo.end(); ++it) {
    if (it->second.mrId == mr)
      return &it->second;
  }
  return NULL;
}

static int resolveIbDevN(int netDev) {
  if (netDev < 0 || netDev >= flagcxNMergedIbDevs)
    return 0;
  return flagcxIbMergedDevs[netDev].devs[0];
}

static uint16_t socketAddrPort(const union flagcxSocketAddress *addr) {
  if (addr == NULL)
    return 0;
  return ntohs(addr->sa.sa_family == AF_INET ? addr->sin.sin_port
                                             : addr->sin6.sin6_port);
}

static void socketAddrSetPort(union flagcxSocketAddress *addr, int port) {
  if (addr == NULL)
    return;
  if (addr->sa.sa_family == AF_INET) {
    addr->sin.sin_port = htons(port);
  } else if (addr->sa.sa_family == AF_INET6) {
    addr->sin6.sin6_port = htons(port);
  }
}

static bool socketAddrSameHost(const union flagcxSocketAddress *a,
                               const union flagcxSocketAddress *b) {
  if (a == NULL || b == NULL || a->sa.sa_family != b->sa.sa_family)
    return false;
  if (a->sa.sa_family == AF_INET) {
    return a->sin.sin_addr.s_addr == b->sin.sin_addr.s_addr;
  }
  if (a->sa.sa_family == AF_INET6) {
    return memcmp(&a->sin6.sin6_addr, &b->sin6.sin6_addr,
                  sizeof(a->sin6.sin6_addr)) == 0 &&
           a->sin6.sin6_scope_id == b->sin6.sin6_scope_id;
  }
  return false;
}

static std::string socketAddrToHostString(const union flagcxSocketAddress *addr) {
  if (addr == NULL)
    return std::string();

  char host[NI_MAXHOST] = {};
  socklen_t salen = addr->sa.sa_family == AF_INET ? sizeof(struct sockaddr_in)
                                                  : sizeof(struct sockaddr_in6);
  if (getnameinfo(&addr->sa, salen, host, sizeof(host), NULL, 0,
                  NI_NUMERICHOST) != 0) {
    return std::string();
  }
  return std::string(host);
}

static std::string
socketAddrToHostPortString(const union flagcxSocketAddress *addr) {
  const std::string host = socketAddrToHostString(addr);
  if (host.empty())
    return std::string();

  const uint16_t port = socketAddrPort(addr);
  if (addr->sa.sa_family == AF_INET6) {
    return "[" + host + "]:" + std::to_string(port);
  }
  return host + ":" + std::to_string(port);
}

static void copyStringToBuf(const std::string &value, char *buf, size_t len) {
  if (buf == NULL || len == 0)
    return;
  snprintf(buf, len, "%s", value.c_str());
}

static int inferLocalGpuIdx() {
  int gpuIdx = 0;
  if (deviceAdaptor && deviceAdaptor->getDevice &&
      deviceAdaptor->getDevice(&gpuIdx) == flagcxSuccess) {
    return gpuIdx;
  }
  return 0;
}

static int chooseEngineNetDev(FlagcxP2pEngine *engine) {
  if (engine == NULL || engine->nDevs <= 0)
    return 0;

  int netDev = 0;
  if (engine->topoMgr) {
    if (flagcxP2pTopoGetNetDev(engine->topoMgr, engine->localGpuIdx, &netDev) !=
        flagcxSuccess) {
      netDev = 0;
    }
  }

  if (netDev >= 0 && netDev < engine->nDevs &&
      engine->listeners[netDev].listenComm != NULL) {
    return netDev;
  }

  for (int d = 0; d < engine->nDevs; d++) {
    if (engine->listeners[d].listenComm != NULL)
      return d;
  }
  return 0;
}

static flagcxResult_t setEngineDevice(FlagcxP2pEngine *engine) {
  if (engine && deviceAdaptor && deviceAdaptor->setDevice) {
    return deviceAdaptor->setDevice(engine->localGpuIdx);
  }
  return flagcxSuccess;
}

static int detectPtrTypeAndMaybeCacheIpc(void *ptr, char *ipcHandleBuf,
                                         uint32_t *ipcHandleSize) {
  if (ipcHandleBuf)
    memset(ipcHandleBuf, 0, FLAGCX_P2P_IPC_HANDLE_BYTES);
  if (ipcHandleSize)
    *ipcHandleSize = 0;

  if (deviceAdaptor == NULL || deviceAdaptor->ipcMemHandleCreate == NULL ||
      deviceAdaptor->ipcMemHandleGet == NULL ||
      deviceAdaptor->ipcMemHandleFree == NULL) {
    return FLAGCX_PTR_HOST;
  }

  flagcxIpcMemHandle_t handle = NULL;
  size_t handleSize = 0;
  if (deviceAdaptor->ipcMemHandleCreate(&handle, &handleSize) != flagcxSuccess) {
    return FLAGCX_PTR_HOST;
  }

  const flagcxResult_t getRes = deviceAdaptor->ipcMemHandleGet(handle, ptr);
  if (getRes == flagcxSuccess && handleSize <= FLAGCX_P2P_IPC_HANDLE_BYTES) {
    if (ipcHandleBuf)
      memcpy(ipcHandleBuf, handle, handleSize);
    if (ipcHandleSize)
      *ipcHandleSize = (uint32_t)handleSize;
    deviceAdaptor->ipcMemHandleFree(handle);
    return FLAGCX_PTR_CUDA;
  }

  deviceAdaptor->ipcMemHandleFree(handle);
  return FLAGCX_PTR_HOST;
}

static void serializeIpcInfo(const FlagcxP2pIpcInfo &info, char *buf) {
  memcpy(buf, &info, sizeof(info));
}

static void deserializeIpcInfo(const char *buf, FlagcxP2pIpcInfo *info) {
  memset(info, 0, sizeof(*info));
  memcpy(info, buf, sizeof(*info));
}

static FlagcxP2pCommView *getCommView(void *comm) {
  return reinterpret_cast<FlagcxP2pCommView *>(comm);
}

static void cleanupIpcXfer(FlagcxP2pXfer *xfer) {
  if (xfer == NULL)
    return;

  if (deviceAdaptor && deviceAdaptor->ipcMemHandleClose) {
    for (size_t i = 0; i < xfer->openedIpcPtrs.size(); i++) {
      if (xfer->openedIpcPtrs[i] != NULL) {
        deviceAdaptor->ipcMemHandleClose(xfer->openedIpcPtrs[i]);
      }
    }
  }
  xfer->openedIpcPtrs.clear();

  if (deviceAdaptor && deviceAdaptor->eventDestroy && xfer->event) {
    deviceAdaptor->eventDestroy(xfer->event);
  }
  if (deviceAdaptor && deviceAdaptor->streamDestroy && xfer->stream) {
    deviceAdaptor->streamDestroy(xfer->stream);
  }
  xfer->event = NULL;
  xfer->stream = NULL;
}

static flagcxResult_t ensureIpcAsyncResources(FlagcxP2pXfer *xfer) {
  if (xfer->stream && xfer->event)
    return flagcxSuccess;
  if (deviceAdaptor == NULL || deviceAdaptor->streamCreate == NULL ||
      deviceAdaptor->eventCreate == NULL) {
    return flagcxInternalError;
  }
  if (deviceAdaptor->streamCreate(&xfer->stream) != flagcxSuccess)
    return flagcxInternalError;
  if (deviceAdaptor->eventCreate(&xfer->event, flagcxEventDisableTiming) !=
      flagcxSuccess) {
    deviceAdaptor->streamDestroy(xfer->stream);
    xfer->stream = NULL;
    return flagcxInternalError;
  }
  return flagcxSuccess;
}

static flagcxMemcpyType_t chooseMemcpyType(bool srcIsCuda, bool dstIsCuda) {
  if (srcIsCuda) {
    return dstIsCuda ? flagcxMemcpyDeviceToDevice : flagcxMemcpyDeviceToHost;
  }
  return dstIsCuda ? flagcxMemcpyHostToDevice : flagcxMemcpyDeviceToHost;
}

static int setFdNonblocking(int fd) {
  const int flags = fcntl(fd, F_GETFL, 0);
  if (flags < 0)
    return -1;
  return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

static int recvAllFd(int fd, void *buf, size_t size) {
  size_t offset = 0;
  char *bytes = reinterpret_cast<char *>(buf);
  while (offset < size) {
    const ssize_t ret = recv(fd, bytes + offset, size - offset, 0);
    if (ret == 0)
      return -1;
    if (ret < 0) {
      if (errno == EINTR)
        continue;
      return -1;
    }
    offset += static_cast<size_t>(ret);
  }
  return 0;
}

static void queueNotifMsg(const FlagcxP2pNotifyMsg &msg) {
  std::lock_guard<std::mutex> notifLock(gNotifyMutex);
  gNotifyList.push_back(msg);
}

static void notifRemoveConnLocked(FlagcxP2pEngine *engine, int fd) {
  std::unordered_map<int, FlagcxP2pNotifConn>::iterator it =
      engine->notifPeers.find(fd);
  if (it == engine->notifPeers.end())
    return;
#if defined(__linux__)
  if (engine->notifEpollFd >= 0) {
    epoll_ctl(engine->notifEpollFd, EPOLL_CTL_DEL, fd, NULL);
  }
#endif
  ::close(fd);
  engine->notifPeers.erase(it);
}

static int notifParseMessages(FlagcxP2pNotifConn *conn) {
  while (conn->inBuf.size() >= sizeof(FlagcxP2pNotifWireMsg)) {
    FlagcxP2pNotifWireMsg wireMsg;
    memcpy(&wireMsg, conn->inBuf.data(), sizeof(wireMsg));
    conn->inBuf.erase(conn->inBuf.begin(),
                      conn->inBuf.begin() + sizeof(wireMsg));
    if (wireMsg.magic != FLAGCX_P2P_NOTIF_MAGIC) {
      return -1;
    }
    queueNotifMsg(wireMsg.payload);
  }
  return 0;
}

static int notifRegisterConn(FlagcxP2pEngine *engine, int fd,
                             const union flagcxSocketAddress *addr) {
#if defined(__linux__)
  if (engine->notifEpollFd >= 0) {
    struct epoll_event event;
    memset(&event, 0, sizeof(event));
    event.data.fd = fd;
    event.events = EPOLLIN | EPOLLET;
#ifdef EPOLLRDHUP
    event.events |= EPOLLRDHUP;
#endif
    if (epoll_ctl(engine->notifEpollFd, EPOLL_CTL_ADD, fd, &event) != 0) {
      return -1;
    }
  }
#endif

  std::lock_guard<std::mutex> lock(engine->notifPeerMutex);
  FlagcxP2pNotifConn conn;
  memset(&conn.addr, 0, sizeof(conn.addr));
  conn.fd = fd;
  if (addr != NULL)
    conn.addr = *addr;
  engine->notifPeers[fd] = std::move(conn);
  return 0;
}

static void notifAcceptLoop(FlagcxP2pEngine *engine) {
  while (!engine->stopNotif.load(std::memory_order_relaxed)) {
    union flagcxSocketAddress remoteAddr;
    socklen_t sockLen = sizeof(remoteAddr);
    const int fd = accept(engine->notifListenSock.fd, &remoteAddr.sa, &sockLen);
    if (fd < 0) {
      if (errno == EINTR)
        continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK)
        break;
      return;
    }

    const int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (char *)&one, sizeof(one));

    uint64_t magic = 0;
    enum flagcxSocketType type = flagcxSocketTypeUnknown;
    if (recvAllFd(fd, &magic, sizeof(magic)) != 0 ||
        recvAllFd(fd, &type, sizeof(type)) != 0 ||
        magic != FLAGCX_SOCKET_MAGIC || type != flagcxSocketTypeProxy ||
        setFdNonblocking(fd) != 0 || notifRegisterConn(engine, fd, &remoteAddr) != 0) {
      ::close(fd);
      continue;
    }
  }
}

static void notifHandleRead(FlagcxP2pEngine *engine, int fd) {
  std::lock_guard<std::mutex> lock(engine->notifPeerMutex);
  std::unordered_map<int, FlagcxP2pNotifConn>::iterator it =
      engine->notifPeers.find(fd);
  if (it == engine->notifPeers.end())
    return;

  char buf[4096];
  while (true) {
    const ssize_t ret = recv(fd, buf, sizeof(buf), 0);
    if (ret == 0) {
      notifRemoveConnLocked(engine, fd);
      return;
    }
    if (ret < 0) {
      if (errno == EINTR)
        continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK)
        break;
      notifRemoveConnLocked(engine, fd);
      return;
    }

    it->second.inBuf.insert(it->second.inBuf.end(), buf, buf + ret);
    if (notifParseMessages(&it->second) != 0) {
      notifRemoveConnLocked(engine, fd);
      return;
    }
  }
}

#if defined(__linux__)
static void notifPollThreadFunc(FlagcxP2pEngine *engine) {
  if (engine == NULL || engine->notifEpollFd < 0)
    return;

  struct epoll_event events[1 + FLAGCX_P2P_MAX_NOTIF_PEERS];
  while (!engine->stopNotif.load(std::memory_order_relaxed)) {
    const int n = epoll_wait(engine->notifEpollFd, events,
                             1 + FLAGCX_P2P_MAX_NOTIF_PEERS, 100);
    if (n < 0) {
      if (errno == EINTR)
        continue;
      break;
    }

    for (int i = 0; i < n; ++i) {
      const int fd = events[i].data.fd;
      if (fd == engine->notifListenSock.fd) {
        notifAcceptLoop(engine);
        continue;
      }

      if (events[i].events & (EPOLLERR | EPOLLHUP
#ifdef EPOLLRDHUP
                              | EPOLLRDHUP
#endif
                                  )) {
        std::lock_guard<std::mutex> lock(engine->notifPeerMutex);
        notifRemoveConnLocked(engine, fd);
        continue;
      }

      if (events[i].events & EPOLLIN) {
        notifHandleRead(engine, fd);
      }
    }
  }
}
#else
static void notifPollThreadFunc(FlagcxP2pEngine *engine) {
  while (!engine->stopNotif.load(std::memory_order_relaxed)) {
    std::vector<struct pollfd> pfds;
    if (engine->notifListenActive) {
      struct pollfd pfd;
      memset(&pfd, 0, sizeof(pfd));
      pfd.fd = engine->notifListenSock.fd;
      pfd.events = POLLIN;
      pfds.push_back(pfd);
    }

    {
      std::lock_guard<std::mutex> lock(engine->notifPeerMutex);
      for (std::unordered_map<int, FlagcxP2pNotifConn>::const_iterator it =
               engine->notifPeers.begin();
           it != engine->notifPeers.end(); ++it) {
        struct pollfd pfd;
        memset(&pfd, 0, sizeof(pfd));
        pfd.fd = it->first;
        pfd.events = POLLIN;
        pfds.push_back(pfd);
      }
    }

    if (pfds.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      continue;
    }

    int ret;
    do {
      ret = poll(pfds.data(), pfds.size(), 100);
    } while (ret < 0 && errno == EINTR);

    if (ret <= 0)
      continue;

    for (size_t i = 0; i < pfds.size(); ++i) {
      if ((pfds[i].revents & (POLLERR | POLLHUP)) != 0) {
        std::lock_guard<std::mutex> lock(engine->notifPeerMutex);
        notifRemoveConnLocked(engine, pfds[i].fd);
        continue;
      }
      if ((pfds[i].revents & POLLIN) == 0)
        continue;
      if (engine->notifListenActive && pfds[i].fd == engine->notifListenSock.fd) {
        notifAcceptLoop(engine);
      } else {
        notifHandleRead(engine, pfds[i].fd);
      }
    }
  }
}
#endif

static int connectNotifSocket(FlagcxP2pConn *conn,
                              const union flagcxSocketAddress *remoteAddr,
                              int notifPort) {
  if (conn == NULL || remoteAddr == NULL || notifPort <= 0)
    return -1;
  if (conn->notifSockConnected)
    return 0;

  union flagcxSocketAddress notifAddr = *remoteAddr;
  socketAddrSetPort(&notifAddr, notifPort);

  if (flagcxSocketInit(&conn->notifSock, &notifAddr, FLAGCX_SOCKET_MAGIC,
                       flagcxSocketTypeProxy, NULL, 0) != flagcxSuccess) {
    return -1;
  }
  if (flagcxSocketConnect(&conn->notifSock) != flagcxSuccess) {
    flagcxSocketClose(&conn->notifSock);
    return -1;
  }

  int ready = 0;
  for (int i = 0; i < 30000 && !ready; i++) {
    if (flagcxSocketReady(&conn->notifSock, &ready) != flagcxSuccess) {
      flagcxSocketClose(&conn->notifSock);
      return -1;
    }
    if (!ready) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  if (!ready) {
    flagcxSocketClose(&conn->notifSock);
    return -1;
  }

  conn->notifSockConnected = true;
  return 0;
}

static int startLocalTransfer(FlagcxP2pConn *conn,
                              const std::vector<void *> &localVec,
                              const std::vector<size_t> &sizeVec,
                              const std::vector<FlagcxP2pRdmaDesc> &descs,
                              int numIovs, uint64_t *transferId,
                              const std::vector<char *> &ipcBufs,
                              bool isWrite) {
  if (conn == NULL || transferId == NULL || numIovs <= 0)
    return -1;

  std::vector<FlagcxP2pMemRegEntry> localEntries(numIovs);
  std::vector<FlagcxP2pMemRegEntry> remoteEntries(numIovs);
  std::vector<bool> haveRemoteEntry(numIovs, false);

  {
    std::lock_guard<std::mutex> lock(gMemMutex);
    for (int i = 0; i < numIovs; i++) {
      if (!findMemReg((uintptr_t)localVec[i], &localEntries[i]))
        return -1;
      if (conn->sameProcess &&
          findMemReg((uintptr_t)descs[i].addr, &remoteEntries[i])) {
        haveRemoteEntry[i] = true;
      }
    }
  }

  if (setEngineDevice(conn->engine) != flagcxSuccess)
    return -1;

  FlagcxP2pXfer xfer;
  xfer.kind = FLAGCX_P2P_XFER_IPC;
  xfer.conn = conn;
  xfer.total = numIovs;
  xfer.completed = 0;
  xfer.stream = NULL;
  xfer.event = NULL;

  bool usedAsync = false;
  for (int i = 0; i < numIovs; i++) {
    void *remotePtr = NULL;
    bool remoteIsCuda = false;

    if (conn->sameProcess) {
      remotePtr = reinterpret_cast<void *>((uintptr_t)descs[i].addr);
      remoteIsCuda =
          haveRemoteEntry[i] && remoteEntries[i].ptrType == FLAGCX_PTR_CUDA;
    } else {
      if (ipcBufs.empty() || i >= (int)ipcBufs.size() || ipcBufs[i] == NULL)
        return -1;

      FlagcxP2pIpcInfo ipcInfo;
      deserializeIpcInfo(ipcBufs[i], &ipcInfo);
      if ((ipcInfo.flags & FLAGCX_P2P_IPC_FLAG_CUDA) == 0)
        return -1;

      flagcxIpcMemHandle_t handle =
          reinterpret_cast<flagcxIpcMemHandle_t>(ipcInfo.handleData);
      void *mappedBase = NULL;
      if (deviceAdaptor == NULL || deviceAdaptor->ipcMemHandleOpen == NULL ||
          deviceAdaptor->ipcMemHandleOpen(handle, &mappedBase) !=
              flagcxSuccess) {
        cleanupIpcXfer(&xfer);
        return -1;
      }

      xfer.openedIpcPtrs.push_back(mappedBase);
      remotePtr = reinterpret_cast<char *>(mappedBase) + ipcInfo.offset;
      remoteIsCuda = true;
    }

    void *dst = isWrite ? remotePtr : localVec[i];
    void *src = isWrite ? localVec[i] : remotePtr;
    const bool dstIsCuda =
        isWrite ? remoteIsCuda : localEntries[i].ptrType == FLAGCX_PTR_CUDA;
    const bool srcIsCuda =
        isWrite ? localEntries[i].ptrType == FLAGCX_PTR_CUDA : remoteIsCuda;

    if (!srcIsCuda && !dstIsCuda) {
      memcpy(dst, src, sizeVec[i]);
      continue;
    }

    if (ensureIpcAsyncResources(&xfer) != flagcxSuccess) {
      cleanupIpcXfer(&xfer);
      return -1;
    }

    const flagcxMemcpyType_t copyType = chooseMemcpyType(srcIsCuda, dstIsCuda);
    if (deviceAdaptor == NULL || deviceAdaptor->deviceMemcpy == NULL ||
        deviceAdaptor->deviceMemcpy(dst, src, sizeVec[i], copyType, xfer.stream,
                                    NULL) != flagcxSuccess) {
      cleanupIpcXfer(&xfer);
      return -1;
    }
    usedAsync = true;
  }

  if (!usedAsync) {
    cleanupIpcXfer(&xfer);
    *transferId = 0;
    return 0;
  }

  if (deviceAdaptor == NULL || deviceAdaptor->eventRecord == NULL ||
      deviceAdaptor->eventRecord(xfer.event, xfer.stream) != flagcxSuccess) {
    cleanupIpcXfer(&xfer);
    return -1;
  }

  std::lock_guard<std::mutex> xferLock(gXferMutex);
  const uint64_t xferId = gNextXferId++;
  gXferMap[xferId] = std::move(xfer);
  *transferId = xferId;
  return 0;
}

FlagcxP2pEngine *flagcxP2pEngineCreate() {
  FlagcxP2pEngine *engine = new FlagcxP2pEngine;
  engine->adaptor = &flagcxNetIbP2p;
  engine->topoMgr = NULL;
  engine->nDevs = 0;
  engine->localGpuIdx = inferLocalGpuIdx();
  engine->notifListenActive = false;
  engine->notifListenPort = 0;
#if defined(__linux__)
  engine->notifEpollFd = -1;
#endif
  engine->stopNotif = false;
  memset(engine->listeners, 0, sizeof(engine->listeners));
  memset(&engine->notifListenSock, 0, sizeof(engine->notifListenSock));

  if (engine->adaptor->init() != flagcxSuccess) {
    delete engine;
    return NULL;
  }

  engine->adaptor->devices(&engine->nDevs);
  if (flagcxP2pTopoInit(engine->adaptor, &engine->topoMgr) != flagcxSuccess) {
    engine->topoMgr = NULL;
  }

  for (int d = 0; d < engine->nDevs; d++) {
    if (engine->adaptor->listen(d, engine->listeners[d].handle,
                                &engine->listeners[d].listenComm) !=
        flagcxSuccess) {
      engine->listeners[d].listenComm = NULL;
    }
  }

  flagcxResult_t notifRes =
      flagcxSocketInit(&engine->notifListenSock, &flagcxIbIfAddr,
                       FLAGCX_SOCKET_MAGIC, flagcxSocketTypeProxy, NULL, 1);
  if (notifRes == flagcxSuccess) {
    notifRes = flagcxSocketListen(&engine->notifListenSock);
  }
  if (notifRes == flagcxSuccess) {
    union flagcxSocketAddress boundAddr;
    engine->notifListenActive = true;
    flagcxSocketGetAddr(&engine->notifListenSock, &boundAddr);
    engine->notifListenPort = socketAddrPort(&boundAddr);
#if defined(__linux__)
    engine->notifEpollFd = epoll_create1(0);
    if (engine->notifEpollFd < 0) {
      flagcxSocketClose(&engine->notifListenSock);
      engine->notifListenActive = false;
      engine->notifListenPort = 0;
    } else {
      struct epoll_event event;
      memset(&event, 0, sizeof(event));
      event.data.fd = engine->notifListenSock.fd;
      event.events = EPOLLIN | EPOLLET;
      if (epoll_ctl(engine->notifEpollFd, EPOLL_CTL_ADD,
                    engine->notifListenSock.fd, &event) != 0) {
        ::close(engine->notifEpollFd);
        engine->notifEpollFd = -1;
        flagcxSocketClose(&engine->notifListenSock);
        engine->notifListenActive = false;
        engine->notifListenPort = 0;
      }
    }
#endif
  }

  if (engine->notifListenActive) {
    engine->notifThread = std::thread(notifPollThreadFunc, engine);
  }
  return engine;
}

void flagcxP2pEngineDestroy(FlagcxP2pEngine *engine) {
  if (engine == NULL)
    return;

  engine->stopNotif = true;
  if (engine->notifListenActive) {
    flagcxSocketClose(&engine->notifListenSock);
    engine->notifListenActive = false;
  }
  if (engine->notifThread.joinable()) {
    engine->notifThread.join();
  }

  {
    std::lock_guard<std::mutex> lock(engine->notifPeerMutex);
    for (std::unordered_map<int, FlagcxP2pNotifConn>::iterator it =
             engine->notifPeers.begin();
         it != engine->notifPeers.end(); ++it) {
      ::close(it->second.fd);
    }
    engine->notifPeers.clear();
  }
#if defined(__linux__)
  if (engine->notifEpollFd >= 0) {
    ::close(engine->notifEpollFd);
    engine->notifEpollFd = -1;
  }
#endif

  for (int d = 0; d < engine->nDevs; d++) {
    if (engine->listeners[d].listenComm) {
      engine->adaptor->closeListen(engine->listeners[d].listenComm);
      engine->listeners[d].listenComm = NULL;
    }
  }

  {
    std::lock_guard<std::mutex> lock(gXferMutex);
    for (std::unordered_map<uint64_t, FlagcxP2pXfer>::iterator it =
             gXferMap.begin();
         it != gXferMap.end(); ++it) {
      cleanupIpcXfer(&it->second);
    }
    gXferMap.clear();
  }

  {
    std::lock_guard<std::mutex> lock(gMemMutex);
    for (std::unordered_map<uintptr_t, FlagcxP2pMemRegEntry>::iterator it =
             gMemRegInfo.begin();
         it != gMemRegInfo.end(); ++it) {
      struct {
        int ibDevN;
      } devCtx = {it->second.ibDevN};
      engine->adaptor->deregMr(&devCtx, it->second.mhandle);
    }
    gMemRegInfo.clear();
  }

  if (engine->topoMgr) {
    flagcxP2pTopoDestroy(engine->topoMgr);
  }

  delete engine;
}

void flagcxP2pEngineStopAccept(FlagcxP2pEngine *engine) {
  if (engine == NULL)
    return;

  engine->stopNotif = true;
  if (engine->notifListenActive) {
    flagcxSocketClose(&engine->notifListenSock);
    engine->notifListenActive = false;
  }

  for (int d = 0; d < engine->nDevs; d++) {
    if (engine->listeners[d].listenComm) {
      engine->adaptor->closeListen(engine->listeners[d].listenComm);
      engine->listeners[d].listenComm = NULL;
    }
  }
}

FlagcxP2pConn *flagcxP2pEngineConnect(FlagcxP2pEngine *engine,
                                      const char *ipAddr, int remoteGpuIdx,
                                      int remotePort, bool sameProcess) {
  if (engine == NULL || ipAddr == NULL)
    return NULL;

  const int netDev = chooseEngineNetDev(engine);

  char remoteHandleBuf[FLAGCX_NET_HANDLE_MAXSIZE];
  memset(remoteHandleBuf, 0, sizeof(remoteHandleBuf));
  FlagcxP2pListenHandleView *remoteHandle =
      reinterpret_cast<FlagcxP2pListenHandleView *>(remoteHandleBuf);

  char ipPortStr[256];
  snprintf(ipPortStr, sizeof(ipPortStr), "%s:%d", ipAddr, remotePort);
  if (flagcxSocketGetAddrFromString(&remoteHandle->connectAddr, ipPortStr) !=
      flagcxSuccess) {
    return NULL;
  }
  remoteHandle->magic = FLAGCX_SOCKET_MAGIC;

  void *sendComm = NULL;
  if (engine->adaptor->connect(netDev, remoteHandleBuf, &sendComm) !=
      flagcxSuccess) {
    return NULL;
  }

  const bool sameHost =
      socketAddrSameHost(&remoteHandle->connectAddr, &flagcxIbIfAddr);
  const bool isLocal = sameHost;
  const bool isSameProcess = sameHost && sameProcess;

  FlagcxP2pCtrlMeta localMeta;
  memset(&localMeta, 0, sizeof(localMeta));
  localMeta.gpuIdx = engine->localGpuIdx;
  localMeta.notifPort = engine->notifListenPort;
  localMeta.flags = 0;
  if (isLocal)
    localMeta.flags |= FLAGCX_P2P_CTRL_FLAG_LOCAL;
  if (isSameProcess)
    localMeta.flags |= FLAGCX_P2P_CTRL_FLAG_SAME_PROCESS;

  FlagcxP2pCtrlMeta remoteMeta;
  memset(&remoteMeta, 0, sizeof(remoteMeta));
  FlagcxP2pCommView *sendView = getCommView(sendComm);
  if (flagcxSocketSendRecv(&sendView->sock, &localMeta, sizeof(localMeta),
                           &sendView->sock, &remoteMeta,
                           sizeof(remoteMeta)) != flagcxSuccess) {
    engine->adaptor->closeSend(sendComm);
    return NULL;
  }

  FlagcxP2pConn *conn = new FlagcxP2pConn;
  conn->engine = engine;
  conn->sendComm = sendComm;
  conn->recvComm = NULL;
  conn->netDev = netDev;
  conn->remoteGpuIdx = remoteMeta.gpuIdx >= 0 ? remoteMeta.gpuIdx : remoteGpuIdx;
  conn->remoteNotifPort = remoteMeta.notifPort;
  conn->isLocal = isLocal || ((remoteMeta.flags & FLAGCX_P2P_CTRL_FLAG_LOCAL) != 0);
  conn->sameProcess =
      isSameProcess || ((remoteMeta.flags & FLAGCX_P2P_CTRL_FLAG_SAME_PROCESS) != 0);
  conn->notifSockConnected = false;
  memset(&conn->notifSock, 0, sizeof(conn->notifSock));

  if (!conn->sameProcess && remoteMeta.notifPort > 0) {
    connectNotifSocket(conn, &remoteHandle->connectAddr, remoteMeta.notifPort);
  }

  return conn;
}

FlagcxP2pConn *flagcxP2pEngineAccept(FlagcxP2pEngine *engine, char *ipAddrBuf,
                                     size_t ipAddrBufLen,
                                     int *remoteGpuIdx) {
  if (engine == NULL || ipAddrBuf == NULL || remoteGpuIdx == NULL)
    return NULL;

  const int dev = chooseEngineNetDev(engine);
  if (engine->listeners[dev].listenComm == NULL)
    return NULL;

  void *recvComm = NULL;
  if (engine->adaptor->accept(engine->listeners[dev].listenComm, &recvComm) !=
      flagcxSuccess) {
    return NULL;
  }

  FlagcxP2pCtrlMeta localMeta;
  memset(&localMeta, 0, sizeof(localMeta));
  localMeta.gpuIdx = engine->localGpuIdx;
  localMeta.notifPort = engine->notifListenPort;
  FlagcxP2pCommView *recvView = getCommView(recvComm);
  if (socketAddrSameHost(&recvView->sock.addr, &flagcxIbIfAddr)) {
    localMeta.flags |= FLAGCX_P2P_CTRL_FLAG_LOCAL;
  }

  FlagcxP2pCtrlMeta remoteMeta;
  memset(&remoteMeta, 0, sizeof(remoteMeta));
  if (flagcxSocketSendRecv(&recvView->sock, &localMeta, sizeof(localMeta),
                           &recvView->sock, &remoteMeta,
                           sizeof(remoteMeta)) != flagcxSuccess) {
    engine->adaptor->closeRecv(recvComm);
    return NULL;
  }

  FlagcxP2pConn *conn = new FlagcxP2pConn;
  conn->engine = engine;
  conn->sendComm = recvComm;
  conn->recvComm = recvComm;
  conn->netDev = dev;
  conn->remoteGpuIdx = remoteMeta.gpuIdx;
  conn->remoteNotifPort = remoteMeta.notifPort;
  conn->isLocal = (remoteMeta.flags & FLAGCX_P2P_CTRL_FLAG_LOCAL) != 0;
  conn->sameProcess =
      (remoteMeta.flags & FLAGCX_P2P_CTRL_FLAG_SAME_PROCESS) != 0;
  conn->notifSockConnected = false;
  memset(&conn->notifSock, 0, sizeof(conn->notifSock));

  copyStringToBuf(socketAddrToHostString(&recvView->sock.addr), ipAddrBuf,
                  ipAddrBufLen);
  *remoteGpuIdx = remoteMeta.gpuIdx;

  if (!conn->sameProcess && remoteMeta.notifPort > 0) {
    connectNotifSocket(conn, &recvView->sock.addr, remoteMeta.notifPort);
  }

  return conn;
}

int flagcxP2pEngineStartListener(FlagcxP2pConn *conn) {
  (void)conn;
  return 0;
}

void flagcxP2pEngineConnDestroy(FlagcxP2pConn *conn) {
  if (conn == NULL)
    return;

  if (conn->sendComm && conn->sendComm != conn->recvComm) {
    conn->engine->adaptor->closeSend(conn->sendComm);
  }
  if (conn->recvComm) {
    conn->engine->adaptor->closeRecv(conn->recvComm);
  }
  if (conn->notifSockConnected) {
    flagcxSocketClose(&conn->notifSock);
  }
  delete conn;
}

bool flagcxP2pEngineConnIsLocal(FlagcxP2pConn *conn) {
  return conn != NULL && conn->isLocal;
}

int flagcxP2pEngineReg(FlagcxP2pEngine *engine, uintptr_t data, size_t size,
                       FlagcxP2pMr &mrId) {
  if (engine == NULL || data == 0)
    return -1;

  std::lock_guard<std::mutex> lock(gMemMutex);

  std::unordered_map<uintptr_t, FlagcxP2pMemRegEntry>::iterator existing =
      gMemRegInfo.find(data);
  if (existing != gMemRegInfo.end()) {
    mrId = existing->second.mrId;
    return 0;
  }

  const int netDev = chooseEngineNetDev(engine);
  const int ibDevN = resolveIbDevN(netDev);
  struct {
    int ibDevN;
  } devCtx = {ibDevN};

  FlagcxP2pMemRegEntry entry;
  memset(&entry, 0, sizeof(entry));
  entry.mrId = gNextMrId++;
  entry.baseAddr = data;
  entry.size = size;
  entry.ibDevN = ibDevN;

  setEngineDevice(engine);
  entry.ptrType = detectPtrTypeAndMaybeCacheIpc(reinterpret_cast<void *>(data),
                                                entry.ipcHandle,
                                                &entry.ipcHandleSize);
  entry.hasIpc = entry.ptrType == FLAGCX_PTR_CUDA && entry.ipcHandleSize > 0;

  if (engine->adaptor->regMr(&devCtx, reinterpret_cast<void *>(data), size,
                             entry.ptrType, FLAGCX_NET_MR_FLAG_NONE,
                             &entry.mhandle) != flagcxSuccess ||
      entry.mhandle == NULL) {
    return -1;
  }

  gMemRegInfo[data] = entry;
  mrId = entry.mrId;
  return 0;
}

void flagcxP2pEngineMrDestroy(FlagcxP2pEngine *engine, FlagcxP2pMr mr) {
  if (engine == NULL)
    return;

  std::lock_guard<std::mutex> lock(gMemMutex);
  for (std::unordered_map<uintptr_t, FlagcxP2pMemRegEntry>::iterator it =
           gMemRegInfo.begin();
       it != gMemRegInfo.end(); ++it) {
    if (it->second.mrId == mr) {
      struct {
        int ibDevN;
      } devCtx = {it->second.ibDevN};
      engine->adaptor->deregMr(&devCtx, it->second.mhandle);
      gMemRegInfo.erase(it);
      return;
    }
  }
}

int flagcxP2pEnginePrepareDesc(FlagcxP2pEngine *engine, FlagcxP2pMr mr,
                               const void *data, size_t size, char *descBuf) {
  if (engine == NULL || data == NULL || descBuf == NULL)
    return -1;

  std::lock_guard<std::mutex> lock(gMemMutex);
  FlagcxP2pMemRegEntry *entry = findMemRegByMr(mr);
  if (entry == NULL)
    return -1;

  FlagcxP2pMrHandleView *mrView =
      reinterpret_cast<FlagcxP2pMrHandleView *>(entry->mhandle);

  FlagcxP2pRdmaDesc desc;
  memset(&desc, 0, sizeof(desc));
  desc.addr = (uint64_t)(uintptr_t)data;
  desc.size = (uint32_t)size;
  desc.rkey = mrView->rkey;

  flagcxP2pSerializeRdmaDesc(desc, descBuf);
  memcpy(entry->descBuf, descBuf, FLAGCX_P2P_DESC_SIZE);
  return 0;
}

int flagcxP2pEngineUpdateDesc(FlagcxP2pRdmaDesc &desc, uint64_t remoteAddr,
                              uint32_t size) {
  desc.addr = remoteAddr;
  desc.size = size;
  return 0;
}

int flagcxP2pEngineRead(FlagcxP2pConn *conn, FlagcxP2pMr mr, const void *data,
                        size_t size, FlagcxP2pRdmaDesc desc,
                        uint64_t *transferId) {
  (void)mr;
  if (conn == NULL || data == NULL || transferId == NULL)
    return -1;

  if (conn->sameProcess && conn->isLocal) {
    std::vector<void *> localVec(1, const_cast<void *>(data));
    std::vector<size_t> sizeVec(1, size);
    std::vector<FlagcxP2pRdmaDesc> descs(1, desc);
    std::vector<char *> ipcBufs;
    return startLocalTransfer(conn, localVec, sizeVec, descs, 1, transferId,
                              ipcBufs, false);
  }

  FlagcxP2pMemRegEntry localEntry;
  {
    std::lock_guard<std::mutex> memLock(gMemMutex);
    if (!findMemReg((uintptr_t)data, &localEntry))
      return -1;
  }

  if (getCommView(conn->sendComm)->ibDevN != localEntry.ibDevN)
    return -1;

  FlagcxP2pMrHandleView *localMr =
      reinterpret_cast<FlagcxP2pMrHandleView *>(localEntry.mhandle);

  FlagcxP2pMrHandleView remoteMr;
  memset(&remoteMr, 0, sizeof(remoteMr));
  remoteMr.baseVa = desc.addr;
  remoteMr.rkey = desc.rkey;

  const uint64_t srcOff = 0;
  const uint64_t dstOff = (uintptr_t)data - localMr->baseVa;

  void *request = NULL;
  if (conn->engine->adaptor->iget(conn->sendComm, srcOff, dstOff, size, 0, 0,
                                  (void **)&remoteMr,
                                  (void **)localEntry.mhandle,
                                  &request) != flagcxSuccess) {
    return -1;
  }

  std::lock_guard<std::mutex> xferLock(gXferMutex);
  const uint64_t xferId = gNextXferId++;
  FlagcxP2pXfer xfer;
  xfer.kind = FLAGCX_P2P_XFER_NET;
  xfer.conn = conn;
  xfer.total = 1;
  xfer.completed = 0;
  xfer.stream = NULL;
  xfer.event = NULL;
  xfer.requests.push_back(request);
  gXferMap[xferId] = xfer;
  *transferId = xferId;
  return 0;
}

int flagcxP2pEngineReadVector(FlagcxP2pConn *conn,
                              std::vector<FlagcxP2pMr> mrIds,
                              std::vector<void *> dstVec,
                              std::vector<size_t> sizeVec,
                              std::vector<FlagcxP2pRdmaDesc> descs, int numIovs,
                              uint64_t *transferId,
                              std::vector<char *> ipcBufs) {
  (void)mrIds;
  if (conn == NULL || numIovs <= 0 || transferId == NULL)
    return -1;

  if (conn->isLocal && (conn->sameProcess || !ipcBufs.empty())) {
    return startLocalTransfer(conn, dstVec, sizeVec, descs, numIovs,
                              transferId, ipcBufs, false);
  }

  std::vector<FlagcxP2pMemRegEntry> localEntries(numIovs);
  {
    std::lock_guard<std::mutex> memLock(gMemMutex);
    for (int i = 0; i < numIovs; i++) {
      if (!findMemReg((uintptr_t)dstVec[i], &localEntries[i]))
        return -1;
    }
  }

  const int connIbDevN = getCommView(conn->sendComm)->ibDevN;
  FlagcxP2pXfer xfer;
  xfer.kind = FLAGCX_P2P_XFER_NET;
  xfer.conn = conn;
  xfer.total = numIovs;
  xfer.completed = 0;
  xfer.stream = NULL;
  xfer.event = NULL;

  for (int i = 0; i < numIovs; i++) {
    if (localEntries[i].ibDevN != connIbDevN)
      return -1;

    FlagcxP2pMrHandleView *localMr =
        reinterpret_cast<FlagcxP2pMrHandleView *>(localEntries[i].mhandle);

    FlagcxP2pMrHandleView remoteMr;
    memset(&remoteMr, 0, sizeof(remoteMr));
    remoteMr.baseVa = descs[i].addr;
    remoteMr.rkey = descs[i].rkey;

    const uint64_t srcOff = 0;
    const uint64_t dstOff = (uintptr_t)dstVec[i] - localMr->baseVa;

    void *request = NULL;
    if (conn->engine->adaptor->iget(conn->sendComm, srcOff, dstOff, sizeVec[i],
                                    0, 0, (void **)&remoteMr,
                                    (void **)localEntries[i].mhandle,
                                    &request) != flagcxSuccess) {
      return -1;
    }
    xfer.requests.push_back(request);
  }

  std::lock_guard<std::mutex> xferLock(gXferMutex);
  const uint64_t xferId = gNextXferId++;
  gXferMap[xferId] = std::move(xfer);
  *transferId = xferId;
  return 0;
}

int flagcxP2pEngineWrite(FlagcxP2pConn *conn, FlagcxP2pMr mr, const void *data,
                         size_t size, FlagcxP2pRdmaDesc desc,
                         uint64_t *transferId) {
  (void)mr;
  if (conn == NULL || data == NULL || transferId == NULL)
    return -1;

  if (conn->sameProcess && conn->isLocal) {
    std::vector<void *> localVec(1, const_cast<void *>(data));
    std::vector<size_t> sizeVec(1, size);
    std::vector<FlagcxP2pRdmaDesc> descs(1, desc);
    std::vector<char *> ipcBufs;
    return startLocalTransfer(conn, localVec, sizeVec, descs, 1, transferId,
                              ipcBufs, true);
  }

  FlagcxP2pMemRegEntry localEntry;
  {
    std::lock_guard<std::mutex> memLock(gMemMutex);
    if (!findMemReg((uintptr_t)data, &localEntry))
      return -1;
  }

  if (getCommView(conn->sendComm)->ibDevN != localEntry.ibDevN)
    return -1;

  FlagcxP2pMrHandleView *localMr =
      reinterpret_cast<FlagcxP2pMrHandleView *>(localEntry.mhandle);

  FlagcxP2pMrHandleView remoteMr;
  memset(&remoteMr, 0, sizeof(remoteMr));
  remoteMr.baseVa = desc.addr;
  remoteMr.rkey = desc.rkey;

  const uint64_t srcOff = (uintptr_t)data - localMr->baseVa;
  const uint64_t dstOff = 0;

  void *request = NULL;
  if (conn->engine->adaptor->iput(conn->sendComm, srcOff, dstOff, size, 0, 0,
                                  (void **)localEntry.mhandle,
                                  (void **)&remoteMr,
                                  &request) != flagcxSuccess) {
    return -1;
  }

  std::lock_guard<std::mutex> xferLock(gXferMutex);
  const uint64_t xferId = gNextXferId++;
  FlagcxP2pXfer xfer;
  xfer.kind = FLAGCX_P2P_XFER_NET;
  xfer.conn = conn;
  xfer.total = 1;
  xfer.completed = 0;
  xfer.stream = NULL;
  xfer.event = NULL;
  xfer.requests.push_back(request);
  gXferMap[xferId] = xfer;
  *transferId = xferId;
  return 0;
}

int flagcxP2pEngineWriteVector(FlagcxP2pConn *conn,
                               std::vector<FlagcxP2pMr> mrIds,
                               std::vector<void *> dstVec,
                               std::vector<size_t> sizeVec,
                               std::vector<FlagcxP2pRdmaDesc> descs,
                               int numIovs, uint64_t *transferId,
                               std::vector<char *> ipcBufs) {
  (void)mrIds;
  if (conn == NULL || numIovs <= 0 || transferId == NULL)
    return -1;

  if (conn->isLocal && (conn->sameProcess || !ipcBufs.empty())) {
    return startLocalTransfer(conn, dstVec, sizeVec, descs, numIovs,
                              transferId, ipcBufs, true);
  }

  std::vector<FlagcxP2pMemRegEntry> localEntries(numIovs);
  {
    std::lock_guard<std::mutex> memLock(gMemMutex);
    for (int i = 0; i < numIovs; i++) {
      if (!findMemReg((uintptr_t)dstVec[i], &localEntries[i]))
        return -1;
    }
  }

  const int connIbDevN = getCommView(conn->sendComm)->ibDevN;
  FlagcxP2pXfer xfer;
  xfer.kind = FLAGCX_P2P_XFER_NET;
  xfer.conn = conn;
  xfer.total = numIovs;
  xfer.completed = 0;
  xfer.stream = NULL;
  xfer.event = NULL;

  for (int i = 0; i < numIovs; i++) {
    if (localEntries[i].ibDevN != connIbDevN)
      return -1;

    FlagcxP2pMrHandleView *localMr =
        reinterpret_cast<FlagcxP2pMrHandleView *>(localEntries[i].mhandle);

    FlagcxP2pMrHandleView remoteMr;
    memset(&remoteMr, 0, sizeof(remoteMr));
    remoteMr.baseVa = descs[i].addr;
    remoteMr.rkey = descs[i].rkey;

    const uint64_t srcOff = (uintptr_t)dstVec[i] - localMr->baseVa;
    const uint64_t dstOff = 0;

    void *request = NULL;
    if (conn->engine->adaptor->iput(conn->sendComm, srcOff, dstOff, sizeVec[i],
                                    0, 0, (void **)localEntries[i].mhandle,
                                    (void **)&remoteMr,
                                    &request) != flagcxSuccess) {
      return -1;
    }
    xfer.requests.push_back(request);
  }

  std::lock_guard<std::mutex> xferLock(gXferMutex);
  const uint64_t xferId = gNextXferId++;
  gXferMap[xferId] = std::move(xfer);
  *transferId = xferId;
  return 0;
}

int flagcxP2pEngineSend(FlagcxP2pConn *conn, FlagcxP2pMr mr, const void *data,
                        size_t size, uint64_t *transferId) {
  (void)conn;
  (void)mr;
  (void)data;
  (void)size;
  (void)transferId;
  return -1;
}

int flagcxP2pEngineSendVector(FlagcxP2pConn *conn,
                              std::vector<FlagcxP2pMr> mrIds,
                              std::vector<const void *> srcVec,
                              std::vector<size_t> sizeVec, int numIovs,
                              uint64_t *transferId) {
  (void)conn;
  (void)mrIds;
  (void)srcVec;
  (void)sizeVec;
  (void)numIovs;
  (void)transferId;
  return -1;
}

int flagcxP2pEngineRecv(FlagcxP2pConn *conn, FlagcxP2pMr mr, void *data,
                        size_t maxSize) {
  (void)conn;
  (void)mr;
  (void)data;
  (void)maxSize;
  return -1;
}

bool flagcxP2pEngineXferStatus(FlagcxP2pConn *conn, uint64_t transferId) {
  if (conn == NULL)
    return true;

  std::lock_guard<std::mutex> lock(gXferMutex);
  std::unordered_map<uint64_t, FlagcxP2pXfer>::iterator it =
      gXferMap.find(transferId);
  if (it == gXferMap.end())
    return true;

  FlagcxP2pXfer &xfer = it->second;
  if (xfer.kind == FLAGCX_P2P_XFER_IPC) {
    if (deviceAdaptor == NULL || deviceAdaptor->eventQuery == NULL) {
      cleanupIpcXfer(&xfer);
      gXferMap.erase(it);
      return true;
    }

    const flagcxResult_t queryRes = deviceAdaptor->eventQuery(xfer.event);
    if (queryRes == flagcxSuccess) {
      cleanupIpcXfer(&xfer);
      gXferMap.erase(it);
      return true;
    }
    if (queryRes != flagcxInProgress) {
      cleanupIpcXfer(&xfer);
      gXferMap.erase(it);
      return true;
    }
    return false;
  }

  for (int i = xfer.completed; i < xfer.total; i++) {
    int done = 0;
    int sizes = 0;
    const flagcxResult_t testRes =
        conn->engine->adaptor->test(xfer.requests[i], &done, &sizes);
    if (testRes != flagcxSuccess)
      return true;
    if (done) {
      xfer.completed++;
    } else {
      break;
    }
  }

  if (xfer.completed >= xfer.total) {
    gXferMap.erase(it);
    return true;
  }
  return false;
}

int flagcxP2pEngineGetMetadata(FlagcxP2pEngine *engine, char **metadataStr) {
  if (engine == NULL || metadataStr == NULL)
    return -1;

  const int netDev = chooseEngineNetDev(engine);
  if (engine->listeners[netDev].listenComm == NULL)
    return -1;

  FlagcxP2pListenHandleView *listenHandle =
      reinterpret_cast<FlagcxP2pListenHandleView *>(
          engine->listeners[netDev].handle);
  const std::string rdmaAddr =
      socketAddrToHostPortString(&listenHandle->connectAddr);
  if (rdmaAddr.empty())
    return -1;

  const std::string result = rdmaAddr + "?" +
                             std::to_string(engine->localGpuIdx) + "?" +
                             std::to_string(engine->notifListenPort);
  *metadataStr = new char[result.length() + 1];
  std::strcpy(*metadataStr, result.c_str());
  return 0;
}

std::vector<FlagcxP2pNotifyMsg> flagcxP2pEngineGetNotifs() {
  std::lock_guard<std::mutex> lock(gNotifyMutex);
  std::vector<FlagcxP2pNotifyMsg> result;
  result.swap(gNotifyList);
  return result;
}

int flagcxP2pEngineSendNotif(FlagcxP2pConn *conn,
                             FlagcxP2pNotifyMsg *notifyMsg) {
  if (conn == NULL || notifyMsg == NULL)
    return -1;

  if (conn->sameProcess) {
    std::lock_guard<std::mutex> lock(gNotifyMutex);
    gNotifyList.push_back(*notifyMsg);
    return sizeof(FlagcxP2pNotifyMsg);
  }

  if (!conn->notifSockConnected) {
    return -1;
  }

  FlagcxP2pNotifWireMsg wireMsg;
  memset(&wireMsg, 0, sizeof(wireMsg));
  wireMsg.magic = FLAGCX_P2P_NOTIF_MAGIC;
  wireMsg.payload = *notifyMsg;
  if (flagcxSocketSend(&conn->notifSock, &wireMsg,
                       sizeof(wireMsg)) != flagcxSuccess) {
    return -1;
  }
  return sizeof(FlagcxP2pNotifyMsg);
}

int flagcxP2pEngineGetIpcInfo(FlagcxP2pEngine *engine, uintptr_t addr,
                              char *ipcBuf, bool *hasIpc) {
  (void)engine;
  if (ipcBuf == NULL || hasIpc == NULL)
    return -1;

  *hasIpc = false;
  FlagcxP2pMemRegEntry entry;
  {
    std::lock_guard<std::mutex> lock(gMemMutex);
    if (!findMemReg(addr, &entry))
      return -1;
  }

  if (!entry.hasIpc)
    return 0;

  FlagcxP2pIpcInfo info;
  memset(&info, 0, sizeof(info));
  memcpy(info.handleData, entry.ipcHandle, entry.ipcHandleSize);
  info.baseAddr = entry.baseAddr;
  info.offset = addr - entry.baseAddr;
  info.size = entry.size - info.offset;
  info.flags = FLAGCX_P2P_IPC_FLAG_CUDA;
  info.handleSize = entry.ipcHandleSize;

  serializeIpcInfo(info, ipcBuf);
  *hasIpc = true;
  return 0;
}

int flagcxP2pEngineUpdateIpcInfo(char *ipcBuf, uintptr_t addr,
                                 uintptr_t baseAddr, size_t size) {
  if (ipcBuf == NULL || addr < baseAddr)
    return -1;

  FlagcxP2pIpcInfo info;
  deserializeIpcInfo(ipcBuf, &info);
  info.offset += (addr - baseAddr);
  info.size = size;
  serializeIpcInfo(info, ipcBuf);
  return 0;
}
