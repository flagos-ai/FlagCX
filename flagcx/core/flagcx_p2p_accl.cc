/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX P2P engine: ACCL (accl::barex) transport for PPU + vsolar
 * hosts, where GPU memory can only be registered and moved through
 * ACCL (no peer-mem/DMA-BUF; VMM unpinnable — run FLAGCX_VMM_ENABLE=0).
 *
 * Shape mirrors Mooncake's barex_transport: one XSimpleMempool over the
 * selected NICs (RegUserMr returns one MR/rkey per NIC); one server +
 * client XContext per NIC; XListener/XConnector own setup (no QP here);
 * transfers post via XChannel::WriteBatch/ReadBatch with callback
 * completion (no CQ poll); the per-slice remote key comes from the
 * region's per-NIC rkey vector via channel->GetPeerNicId().
 *
 * Rendezvous reuses FlagCX bootstrap with an ACCL hello (magic rejects
 * ibrc peers). The 64-byte FlagcxP2pRdmaDesc is kept: rkeys[0] at ibrc's
 * .rkey offset, count in .nmsgs, rkeys[1..7] in .padding. v1: transfers
 * initiated by the connecting side only; no IPC path; no two-sided.
 ************************************************************************/

#ifdef USE_ACCL_BAREX

#include "flagcx_p2p_accl.h"

#include "adaptor.h"
#include "bootstrap.h"
#include "debug.h"
#include "param.h"
#include "socket.h"

#undef CPU
#undef GPU
#undef NIC
#undef NET
#undef PCI

#include <accl/barex/barex_types.h>
#include <accl/barex/xchannel.h>
#include <accl/barex/xconfig_util.h>
#include <accl/barex/xconnector.h>
#include <accl/barex/xcontext.h>
#include <accl/barex/xdevice_manager.h>
#include <accl/barex/xlistener.h>
#include <accl/barex/xsimple_mempool.h>
#include <accl/barex/xthreadpool.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

using namespace accl::barex;

namespace {

constexpr uint64_t kAcclHelloMagic = 0xACC1F1A6C0DE0001ull;
constexpr uint32_t kAcclNotifMagic = 0xDEADDEADu;   /* same wire as ibrc */
constexpr int kMaxNics = kFlagcxP2pMaxQpsPerEngine; /* 8, matches desc */

struct AcclHelloWire {
  uint64_t magic;
  int32_t barexPort;
  int32_t gpuIdx;
  int32_t notifPort;
  uint32_t flags;
  char pad[128 - 24];
};
static_assert(sizeof(AcclHelloWire) == FLAGCX_NET_HANDLE_MAXSIZE,
              "hello must match the ibrc listen-handle exchange size");

struct AcclMemRegWire {
  uint64_t baseAddr;
  uint64_t size;
  uint32_t nKeys;
  uint32_t rkeys[kMaxNics];
  uint32_t reserved;
};
static_assert(sizeof(AcclMemRegWire) == 56, "stable wire layout");

struct AcclNotifWireMsg {
  uint32_t magic;
  uint32_t reserved;
  FlagcxP2pNotifyMsg payload;
};

struct AcclRemoteRegion {
  uint64_t baseAddr;
  uint64_t size;
  uint32_t nKeys;
  uint32_t rkeys[kMaxNics];
};

struct AcclMrEntry {
  uint64_t mrId;
  uintptr_t baseAddr; /* this chunk */
  size_t size;
  uintptr_t regBase; /* whole logical registration this chunk belongs to */
  size_t regSize;
  device_type dtype;
  int deviceId;
  uint32_t nKeys;
  uint32_t lkeys[kMaxNics]; /* indexed by ACCL device id */
  uint32_t rkeys[kMaxNics];
};

struct AcclXfer {
  std::atomic<int> pending{0};
  std::atomic<int> failed{0};
};

struct NotifPeerFd {
  int fd;
  std::vector<char> inBuf;
};

} // namespace

struct FlagcxAcclConn;

struct FlagcxAcclEngine {
  uint32_t kind = FLAGCX_P2P_KIND_ACCL;
  int localGpuIdx = 0;
  int nDevs = 0;

  std::vector<XDevice *> devs;
  XSimpleMempool *mempool = nullptr;
  XThreadpool *tpServer = nullptr;
  XThreadpool *tpClient = nullptr;
  std::vector<XContext *> serverCtxs;
  std::vector<XContext *> clientCtxs;
  XListener *listener = nullptr;
  XConnector *connector = nullptr;
  int barexPort = 0;

  struct bootstrapState *bsListenState = nullptr;
  int bsListenPort = 0;
  std::atomic<bool> stopAccept{false};
  volatile uint32_t acceptAbortFlag = 0;

  struct flagcxSocket notifListenSock;
  bool notifActive = false;
  int notifPort = 0;
  std::thread notifThread;
  std::atomic<bool> stopNotif{false};

  std::mutex mrMu;
  uint64_t nextMrId = 1;
  std::map<uintptr_t, AcclMrEntry> mrByBase; /* keyed by chunk base */
  /* vsolar caps a single GPU MR at 64MB, so registrations are split into
     chunks of at most this many bytes (FLAGCX_ACCL_MAX_MR_MB, 0 = off). */
  size_t mrChunkBytes = 64ull << 20;

  std::mutex xferMu;
  uint64_t nextXferId = 1;
  std::unordered_map<uint64_t, std::shared_ptr<AcclXfer>> xfers;

  std::thread rpcThread;
  std::atomic<bool> rpcActive{false};
  std::atomic<bool> stopRpc{false};
  std::unordered_map<std::string, FlagcxP2pConn *> sessions;
  std::mutex sessMu;
  std::vector<FlagcxP2pConn *> accepted;
  std::mutex accMu;
};

struct FlagcxAcclConn {
  uint32_t kind = FLAGCX_P2P_KIND_ACCL;
  FlagcxAcclEngine *engine = nullptr;
  bool initiator = false;
  int remoteGpuIdx = -1;
  int remoteNotifPort = 0;
  bool isLocal = false;
  bool sameProcess = false;

  union flagcxSocketAddress peerAddr; /* host part; for notif connect */
  struct flagcxSocket notifSock;
  bool notifConnected = false;

  std::vector<AcclRemoteRegion> remoteRegions; /* one per remote chunk */
  /* merged contiguous extents of remoteRegions, for range validation
     (chunks of one logical registration are contiguous by construction) */
  std::vector<std::pair<uint64_t, uint64_t>> remoteSpans; /* base, size */
  std::vector<XChannel *> channels;                       /* initiator side */
  std::atomic<uint64_t> rr{0};
};

namespace {

inline FlagcxAcclEngine *E(FlagcxP2pEngine *e) {
  return reinterpret_cast<FlagcxAcclEngine *>(e);
}
inline FlagcxP2pEngine *EOut(FlagcxAcclEngine *e) {
  return reinterpret_cast<FlagcxP2pEngine *>(e);
}
inline FlagcxAcclConn *C(FlagcxP2pConn *c) {
  return reinterpret_cast<FlagcxAcclConn *>(c);
}
inline FlagcxP2pConn *COut(FlagcxAcclConn *c) {
  return reinterpret_cast<FlagcxP2pConn *>(c);
}

const char *bxstr(BarexResult r) {
  auto it = BarexResultStrings.find(r);
  return it == BarexResultStrings.end() ? "UNKNOWN" : it->second;
}

uint16_t addrPort(const union flagcxSocketAddress *addr) {
  if (addr == NULL)
    return 0;
  return ntohs(addr->sa.sa_family == AF_INET ? addr->sin.sin_port
                                             : addr->sin6.sin6_port);
}

void addrSetPort(union flagcxSocketAddress *addr, int port) {
  if (addr == NULL)
    return;
  if (addr->sa.sa_family == AF_INET)
    addr->sin.sin_port = htons(port);
  else if (addr->sa.sa_family == AF_INET6)
    addr->sin6.sin6_port = htons(port);
}

std::string addrHostString(const union flagcxSocketAddress *addr) {
  if (addr == NULL)
    return std::string();
  char host[NI_MAXHOST] = {};
  socklen_t salen = addr->sa.sa_family == AF_INET ? sizeof(struct sockaddr_in)
                                                  : sizeof(struct sockaddr_in6);
  if (getnameinfo(&addr->sa, salen, host, sizeof(host), NULL, 0,
                  NI_NUMERICHOST) != 0)
    return std::string();
  return std::string(host);
}

int inferLocalGpuIdxAccl() {
  int gpuIdx = 0;
  if (deviceAdaptor && deviceAdaptor->getDevice &&
      deviceAdaptor->getDevice(&gpuIdx) == flagcxSuccess)
    return gpuIdx;
  return 0;
}

/* Classify a user pointer for RegUserMr. This transport is inherently
   CUDA-runtime based (libaccl_barex itself links the PPU CUDA stack). */
bool classifyPtr(const void *ptr, device_type *dt, int *devId) {
  cudaPointerAttributes attrs;
  memset(&attrs, 0, sizeof(attrs));
  cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
  if (err != cudaSuccess) {
    cudaGetLastError(); /* clear sticky error for host pointers */
    *dt = CPU;
    *devId = 0;
    return true;
  }
  if (attrs.type == cudaMemoryTypeDevice ||
      attrs.type == cudaMemoryTypeManaged) {
    *dt = GPU;
    *devId = attrs.device;
    return true;
  }
  *dt = CPU;
  *devId = 0;
  return true;
}

class AcclNullCb : public XChannelCallback {
public:
  void OnRecvCall(XChannel *, char *, size_t, x_msg_header) override {}
};

/* one-shot waiter for connect latches — heap-allocated and shared with
   every callback so a late ACCL callback after our timeout can never
   touch destroyed stack state. */
struct AcclConnectCtl {
  std::mutex mu;
  std::condition_variable cv;
  int remaining;
  bool abandoned = false;
  bool anyFailed = false;
  std::vector<XChannel *> channels;
  explicit AcclConnectCtl(int n) : remaining(n) {}
};

/* Notification plane: listen socket + poll thread feeding the shared notify
 * list in flagcx_p2p.cc (same wire format as the ibrc engine). */

int recvAllFdAccl(int fd, void *buf, size_t len) {
  char *p = static_cast<char *>(buf);
  size_t got = 0;
  while (got < len) {
    ssize_t r = recv(fd, p + got, len - got, 0);
    if (r == 0)
      return -1;
    if (r < 0) {
      if (errno == EINTR)
        continue;
      return -1;
    }
    got += static_cast<size_t>(r);
  }
  return 0;
}

void notifThreadFunc(FlagcxAcclEngine *engine) {
  std::vector<NotifPeerFd> peers;
  while (!engine->stopNotif.load(std::memory_order_relaxed)) {
    std::vector<struct pollfd> fds;
    fds.push_back({engine->notifListenSock.fd, POLLIN, 0});
    for (auto &p : peers)
      fds.push_back({p.fd, POLLIN, 0});

    int n = poll(fds.data(), fds.size(), 100);
    if (n < 0) {
      if (errno == EINTR)
        continue;
      break;
    }
    if (n == 0)
      continue;

    if (fds[0].revents & POLLIN) {
      union flagcxSocketAddress remoteAddr;
      socklen_t sockLen = sizeof(remoteAddr);
      int fd = accept(engine->notifListenSock.fd, &remoteAddr.sa, &sockLen);
      if (fd >= 0) {
        const int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (char *)&one, sizeof(one));
        /* accepted fds don't inherit O_NONBLOCK; bound the magic read so
           a stalled peer can't wedge this single poll loop past shutdown */
        struct timeval tv = {2, 0};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
        uint64_t magic = 0;
        int type = 0;
        if (recvAllFdAccl(fd, &magic, sizeof(magic)) != 0 ||
            recvAllFdAccl(fd, &type, sizeof(type)) != 0 ||
            magic != FLAGCX_SOCKET_MAGIC) {
          ::close(fd);
        } else {
          tv.tv_sec = 0; /* back to non-timeout; poll() gates reads below */
          setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
          peers.push_back(NotifPeerFd{fd, {}});
        }
      }
    }

    for (size_t i = 1; i < fds.size(); i++) {
      NotifPeerFd &peer = peers[i - 1];
      if (fds[i].revents & (POLLERR | POLLHUP)) {
        ::close(peer.fd);
        peer.fd = -1;
        continue;
      }
      if (!(fds[i].revents & POLLIN))
        continue;
      char buf[4096];
      ssize_t r = recv(peer.fd, buf, sizeof(buf), 0);
      if (r <= 0) {
        if (r < 0 && (errno == EINTR || errno == EAGAIN))
          continue;
        ::close(peer.fd);
        peer.fd = -1;
        continue;
      }
      peer.inBuf.insert(peer.inBuf.end(), buf, buf + r);
      while (peer.inBuf.size() >= sizeof(AcclNotifWireMsg)) {
        AcclNotifWireMsg msg;
        memcpy(&msg, peer.inBuf.data(), sizeof(msg));
        peer.inBuf.erase(peer.inBuf.begin(), peer.inBuf.begin() + sizeof(msg));
        if (msg.magic != kAcclNotifMagic)
          continue;
        flagcxP2pNotifyAppend(msg.payload);
      }
    }
    peers.erase(std::remove_if(peers.begin(), peers.end(),
                               [](const NotifPeerFd &p) { return p.fd < 0; }),
                peers.end());
  }
  for (auto &p : peers)
    if (p.fd >= 0)
      ::close(p.fd);
}

/* Desc helpers: rkey vector folded into the 64-byte desc */

void fillDescKeys(FlagcxP2pRdmaDesc *desc, const uint32_t *rkeys,
                  uint32_t nKeys) {
  desc->rkey = nKeys > 0 ? rkeys[0] : 0;
  desc->nmsgs = nKeys;
  memset(desc->padding, 0, sizeof(desc->padding));
  for (uint32_t k = 1; k < nKeys && k < kMaxNics; k++)
    memcpy(desc->padding + (k - 1) * sizeof(uint32_t), &rkeys[k],
           sizeof(uint32_t));
}

uint32_t descKeyForNic(const FlagcxP2pRdmaDesc &desc, int nic) {
  const uint32_t nKeys = desc.nmsgs;
  if (nKeys <= 1 || nic <= 0 || nic >= kMaxNics)
    return desc.rkey; /* single-key or unknown nic: rkey as-is */
  if ((uint32_t)nic >= nKeys)
    return desc.rkey;
  uint32_t k = 0;
  memcpy(&k, desc.padding + (nic - 1) * sizeof(uint32_t), sizeof(uint32_t));
  return k;
}

bool findMrContaining(FlagcxAcclEngine *engine, uintptr_t addr, size_t size,
                      AcclMrEntry *out) {
  std::lock_guard<std::mutex> lk(engine->mrMu);
  auto it = engine->mrByBase.upper_bound(addr);
  if (it == engine->mrByBase.begin())
    return false;
  --it;
  const AcclMrEntry &e = it->second;
  if (addr >= e.baseAddr && addr + size <= e.baseAddr + e.size) {
    if (out)
      *out = e;
    return true;
  }
  return false;
}

/* Chunked registrations: a remote VA resolves to the chunk that contains
   it (regions are sorted by base). Returns nullptr when the conn has no
   region table (legacy peers exchanging single-region descs). */
const AcclRemoteRegion *findRemoteRegion(const FlagcxAcclConn *conn,
                                         uint64_t va) {
  const auto &regions = conn->remoteRegions;
  if (regions.empty())
    return nullptr;
  size_t lo = 0, hi = regions.size();
  while (lo < hi) { /* first region with base > va, then step back */
    size_t mid = lo + (hi - lo) / 2;
    if (regions[mid].baseAddr <= va)
      lo = mid + 1;
    else
      hi = mid;
  }
  if (lo == 0)
    return nullptr;
  const AcclRemoteRegion &r = regions[lo - 1];
  return (va >= r.baseAddr && va < r.baseAddr + r.size) ? &r : nullptr;
}

uint32_t regionKeyForNic(const AcclRemoteRegion &r, int nic) {
  if (nic < 0 || (uint32_t)nic >= r.nKeys || nic >= kMaxNics)
    return r.nKeys > 0 ? r.rkeys[0] : 0;
  return r.rkeys[nic];
}

XChannel *pickChannel(FlagcxAcclConn *conn) {
  const size_t n = conn->channels.size();
  if (n == 0)
    return nullptr;
  uint64_t start = conn->rr.fetch_add(1, std::memory_order_relaxed);
  for (size_t i = 0; i < n; i++) {
    XChannel *ch = conn->channels[(start + i) % n];
    if (ch != nullptr && ch->IsActive())
      return ch;
  }
  return nullptr;
}

/* Shared submit for single+vector read/write. */
int acclSubmit(FlagcxAcclConn *conn, const std::vector<void *> &localVec,
               const std::vector<size_t> &sizeVec,
               const std::vector<FlagcxP2pRdmaDesc> &descs, int numIovs,
               bool isRead, uint64_t *transferId) {
  FlagcxAcclEngine *engine = conn->engine;
  if (!conn->initiator) {
    WARN("NET/ACCL_P2P : v1 supports initiator-side transfers only");
    return -1;
  }
  XChannel *ch = pickChannel(conn);
  if (ch == nullptr) {
    WARN("NET/ACCL_P2P : no active channel");
    return -1;
  }
  const int localNic = ch->GetContext()->GetXDevice()->GetId();
  const int peerNic = ch->GetPeerNicId();

  auto batch = std::make_shared<std::vector<rw_memp_t>>();
  batch->reserve(numIovs);
  for (int i = 0; i < numIovs; i++) {
    if (sizeVec[i] == 0)
      continue;
    if (sizeVec[i] > UINT32_MAX) {
      WARN("NET/ACCL_P2P : iov %d size %zu exceeds 4GiB desc limit", i,
           sizeVec[i]);
      return -1;
    }
    /* Registrations are chunked (vsolar 64MB per-MR cap), so one iov may
       span several local MRs and several remote regions. Split at every
       chunk boundary on either side; each slice carries the lkey/rkey of
       the chunks it lands in. */
    uintptr_t lcur = (uintptr_t)localVec[i];
    uint64_t rcur = descs[i].addr;
    size_t remaining = sizeVec[i];
    while (remaining > 0) {
      AcclMrEntry entry;
      if (!findMrContaining(engine, lcur, 1, &entry)) {
        WARN("NET/ACCL_P2P : local buffer %p not registered", (void *)lcur);
        return -1;
      }
      if (localNic < 0 || (uint32_t)localNic >= entry.nKeys) {
        WARN("NET/ACCL_P2P : lkey for nic %d missing (nKeys=%u)", localNic,
             entry.nKeys);
        return -1;
      }
      size_t slice = std::min(remaining, entry.baseAddr + entry.size - lcur);
      uint32_t rkey;
      const AcclRemoteRegion *rr = findRemoteRegion(conn, rcur);
      if (rr != nullptr) {
        slice = std::min(slice, (size_t)(rr->baseAddr + rr->size - rcur));
        rkey = regionKeyForNic(*rr, peerNic);
      } else {
        /* no region table entry — trust the caller's desc keys wholesale */
        rkey = descKeyForNic(descs[i], peerNic);
      }

      rw_memp_t w{};
      w.sg.addr = (uint64_t)lcur;
      w.sg.length = (uint32_t)slice;
      w.sg.lkey = entry.lkeys[localNic];
      w.data.d_type = entry.dtype;
      w.data.device_id = entry.deviceId;
      w.r_addr = rcur;
      w.r_key = rkey;
      w.r_ttl_ms = UINT64_MAX;
      batch->push_back(w);

      lcur += slice;
      rcur += slice;
      remaining -= slice;
    }
  }
  if (batch->empty()) {
    *transferId = 0; /* nothing to do; XferStatus(0) reports done */
    return 0;
  }

  auto xfer = std::make_shared<AcclXfer>();
  xfer->pending.store(1, std::memory_order_release);

  uint64_t id;
  {
    std::lock_guard<std::mutex> lk(engine->xferMu);
    id = engine->nextXferId++;
    engine->xfers[id] = xfer;
  }

  DoneCallback done = [xfer, batch](Status s) {
    if (!s.IsOk()) {
      WARN("NET/ACCL_P2P : batch failed: %s", s.ErrMsg().c_str());
      xfer->failed.fetch_add(1, std::memory_order_release);
    }
    xfer->pending.fetch_sub(1, std::memory_order_release);
  };

  BarexResult r = isRead ? ch->ReadBatch(batch, done, true)
                         : ch->WriteBatch(batch, done, true);
  if (r != BAREX_SUCCESS) {
    WARN("NET/ACCL_P2P : %s sync error: %s",
         isRead ? "ReadBatch" : "WriteBatch", bxstr(r));
    std::lock_guard<std::mutex> lk(engine->xferMu);
    engine->xfers.erase(id);
    return -1;
  }
  *transferId = id;
  return 0;
}

/* Exchange hello + desc table over an established bootstrap conn.
   Both sides call with the same tag sequence. */
int acclHandshake(FlagcxAcclEngine *engine, struct bootstrapState *bsConn,
                  FlagcxAcclConn *conn) {
  AcclHelloWire localHello;
  memset(&localHello, 0, sizeof(localHello));
  localHello.magic = kAcclHelloMagic;
  localHello.barexPort = engine->barexPort;
  localHello.gpuIdx = engine->localGpuIdx;
  localHello.notifPort = engine->notifPort;

  AcclHelloWire remoteHello;
  memset(&remoteHello, 0, sizeof(remoteHello));
  if (bootstrapExchange(bsConn, 0, 4, &localHello, sizeof(localHello),
                        &remoteHello, sizeof(remoteHello)) != flagcxSuccess)
    return -1;
  if (remoteHello.magic != kAcclHelloMagic) {
    WARN("NET/ACCL_P2P : peer is not running the ACCL transport "
         "(magic 0x%llx) — both ends must set FLAGCX_P2P_TRANSPORT=accl",
         (unsigned long long)remoteHello.magic);
    return -1;
  }
  conn->remoteGpuIdx = remoteHello.gpuIdx;
  conn->remoteNotifPort = remoteHello.notifPort;

  /* desc table with per-NIC rkey vectors */
  std::vector<AcclMemRegWire> localTable;
  {
    std::lock_guard<std::mutex> lk(engine->mrMu);
    localTable.reserve(engine->mrByBase.size());
    for (auto &kv : engine->mrByBase) {
      const AcclMrEntry &e = kv.second;
      AcclMemRegWire w;
      memset(&w, 0, sizeof(w));
      w.baseAddr = e.baseAddr;
      w.size = e.size;
      w.nKeys = e.nKeys;
      memcpy(w.rkeys, e.rkeys, sizeof(w.rkeys));
      localTable.push_back(w);
    }
  }
  uint32_t localCount = (uint32_t)localTable.size();
  uint32_t remoteCount = 0;
  if (bootstrapExchange(bsConn, 0, 2, &localCount, sizeof(localCount),
                        &remoteCount, sizeof(remoteCount)) != flagcxSuccess)
    return -1;
  if (remoteCount > 65536)
    return -1;
  std::vector<AcclMemRegWire> remoteTable(remoteCount);
  if (bootstrapExchange(
          bsConn, 0, 3, localTable.data(),
          (int)(localCount * sizeof(AcclMemRegWire)), remoteTable.data(),
          (int)(remoteCount * sizeof(AcclMemRegWire))) != flagcxSuccess)
    return -1;
  conn->remoteRegions.clear();
  conn->remoteRegions.reserve(remoteCount);
  for (uint32_t i = 0; i < remoteCount; i++) {
    AcclRemoteRegion r;
    r.baseAddr = remoteTable[i].baseAddr;
    r.size = remoteTable[i].size;
    r.nKeys = remoteTable[i].nKeys;
    memcpy(r.rkeys, remoteTable[i].rkeys, sizeof(r.rkeys));
    conn->remoteRegions.push_back(r);
  }
  std::sort(conn->remoteRegions.begin(), conn->remoteRegions.end(),
            [](const AcclRemoteRegion &a, const AcclRemoteRegion &b) {
              return a.baseAddr < b.baseAddr;
            });
  /* merge contiguous chunks into spans for range validation */
  conn->remoteSpans.clear();
  for (const AcclRemoteRegion &r : conn->remoteRegions) {
    if (!conn->remoteSpans.empty() &&
        conn->remoteSpans.back().first + conn->remoteSpans.back().second ==
            r.baseAddr)
      conn->remoteSpans.back().second += r.size;
    else
      conn->remoteSpans.emplace_back(r.baseAddr, r.size);
  }
  return remoteHello.barexPort;
}

int connectNotif(FlagcxAcclConn *conn) {
  if (conn->notifConnected)
    return 0;
  if (conn->remoteNotifPort <= 0)
    return -1;
  union flagcxSocketAddress notifAddr = conn->peerAddr;
  addrSetPort(&notifAddr, conn->remoteNotifPort);
  if (flagcxSocketInit(&conn->notifSock, &notifAddr, FLAGCX_SOCKET_MAGIC,
                       flagcxSocketTypeProxy, NULL, 0) != flagcxSuccess)
    return -1;
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
    if (!ready)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (!ready) {
    flagcxSocketClose(&conn->notifSock);
    return -1;
  }
  conn->notifConnected = true;
  return 0;
}

} // namespace

FlagcxP2pEngine *flagcxAcclEngineCreate() {
  /* ACCL orders devices by ACCL_USE_NICS; seed it from FLAGCX_IB_HCA so
     both peers see identical NIC indexing (rkey vectors align). */
  const char *hca = flagcxGetEnv("FLAGCX_IB_HCA");
  if (hca != nullptr && flagcxGetEnv("ACCL_USE_NICS") == nullptr) {
    setenv("ACCL_USE_NICS", hca, 0);
    INFO(FLAGCX_INIT, "NET/ACCL_P2P : ACCL_USE_NICS=%s (from FLAGCX_IB_HCA)",
         hca);
  }

  auto *engine = new FlagcxAcclEngine();
  engine->localGpuIdx = inferLocalGpuIdxAccl();
  memset(&engine->notifListenSock, 0, sizeof(engine->notifListenSock));

  const char *mrMb = flagcxGetEnv("FLAGCX_ACCL_MAX_MR_MB");
  if (mrMb != nullptr) {
    engine->mrChunkBytes = (size_t)strtoull(mrMb, nullptr, 10) << 20;
    INFO(FLAGCX_INIT, "NET/ACCL_P2P : MR chunk size %zu MB%s",
         engine->mrChunkBytes >> 20,
         engine->mrChunkBytes == 0 ? " (chunking off)" : "");
  }

  XDeviceManager *mgr = nullptr;
  if (XDeviceManager::Singleton(mgr) != BAREX_SUCCESS || mgr == nullptr) {
    WARN("NET/ACCL_P2P : XDeviceManager::Singleton failed");
    delete engine;
    return nullptr;
  }
  engine->devs = mgr->AllDevices();
  engine->nDevs = (int)engine->devs.size();
  if (engine->nDevs == 0 || engine->nDevs > kMaxNics) {
    WARN("NET/ACCL_P2P : %d RDMA devices (supported 1..%d)", engine->nDevs,
         kMaxNics);
    delete engine;
    return nullptr;
  }
  for (auto *d : engine->devs)
    INFO(FLAGCX_INIT, "NET/ACCL_P2P : dev id=%d name=%s", d->GetId(),
         d->GetName().c_str());

  BarexResult r = XSimpleMempool::NewInstance(engine->mempool,
                                              "flagcx-p2p-accl", engine->devs);
  if (r != BAREX_SUCCESS) {
    WARN("NET/ACCL_P2P : mempool: %s", bxstr(r));
    delete engine;
    return nullptr;
  }
  XThreadpool::NewInstance(engine->tpServer, 4, "flagcx-accl-server");
  XThreadpool::NewInstance(engine->tpClient, 4, "flagcx-accl-client");

  ContextConfig cfg = XConfigUtil::DefaultContextConfig();
  for (auto *dev : engine->devs) {
    XContext *sctx = nullptr, *cctx = nullptr;
    if (XContext::NewInstance(sctx, cfg, new AcclNullCb(), dev, engine->mempool,
                              engine->tpServer) != BAREX_SUCCESS ||
        XContext::NewInstance(cctx, cfg, new AcclNullCb(), dev, engine->mempool,
                              engine->tpClient) != BAREX_SUCCESS) {
      WARN("NET/ACCL_P2P : XContext create failed on %s",
           dev->GetName().c_str());
      flagcxAcclEngineDestroy(EOut(engine));
      return nullptr;
    }
    sctx->Start();
    cctx->Start();
    engine->serverCtxs.push_back(sctx);
    engine->clientCtxs.push_back(cctx);
  }

  /* barex data-plane listener: probe for a free port */
  const int base = 18000 + (int)(getpid() % 4096);
  for (int attempt = 0; attempt < 32; attempt++) {
    const int port = base + attempt * 3;
    XListener *lis = nullptr;
    if (XListener::NewInstance(lis, 2, port, TIMER_3S, engine->serverCtxs) ==
            BAREX_SUCCESS &&
        lis->Listen() == BAREX_SUCCESS) {
      engine->listener = lis;
      engine->barexPort = port;
      break;
    }
    if (lis != nullptr) {
      lis->Shutdown();
      lis->WaitStop();
      delete lis;
    }
  }
  if (engine->listener == nullptr) {
    WARN("NET/ACCL_P2P : no free barex port");
    flagcxAcclEngineDestroy(EOut(engine));
    return nullptr;
  }
  if (XConnector::NewInstance(engine->connector, 2, TIMER_3S,
                              engine->clientCtxs) != BAREX_SUCCESS) {
    WARN("NET/ACCL_P2P : XConnector create failed");
    flagcxAcclEngineDestroy(EOut(engine));
    return nullptr;
  }

  /* bootstrap rendezvous listener (shared FlagCX service code) */
  bootstrapNetInit();
  char bsListenHandle[FLAGCX_NET_HANDLE_MAXSIZE];
  memset(bsListenHandle, 0, sizeof(bsListenHandle));
  struct bootstrapState *bsState = nullptr;
  if (bootstrapP2pListen(FLAGCX_SOCKET_MAGIC, &engine->acceptAbortFlag,
                         bsListenHandle, &bsState) != flagcxSuccess) {
    WARN("NET/ACCL_P2P : bootstrap listen failed");
    flagcxAcclEngineDestroy(EOut(engine));
    return nullptr;
  }
  engine->bsListenState = bsState;
  union flagcxSocketAddress bsAddr;
  flagcxSocketGetAddr(&bsState->p2p->sock, &bsAddr);
  engine->bsListenPort = addrPort(&bsAddr);

  /* notif listener on the same interface, ephemeral port */
  union flagcxSocketAddress notifAddr = bsAddr;
  addrSetPort(&notifAddr, 0);
  if (flagcxSocketInit(&engine->notifListenSock, &notifAddr,
                       FLAGCX_SOCKET_MAGIC, flagcxSocketTypeProxy, NULL,
                       1) == flagcxSuccess &&
      flagcxSocketListen(&engine->notifListenSock) == flagcxSuccess) {
    union flagcxSocketAddress bound;
    flagcxSocketGetAddr(&engine->notifListenSock, &bound);
    engine->notifPort = addrPort(&bound);
    engine->notifActive = true;
    engine->notifThread = std::thread(notifThreadFunc, engine);
  }

  INFO(FLAGCX_INIT,
       "NET/ACCL_P2P : engine up (gpu=%d nics=%d barex=%d bootstrap=%d "
       "notif=%d)",
       engine->localGpuIdx, engine->nDevs, engine->barexPort,
       engine->bsListenPort, engine->notifPort);
  return EOut(engine);
}

void flagcxAcclEngineStopAccept(FlagcxP2pEngine *e) {
  FlagcxAcclEngine *engine = E(e);
  if (engine == nullptr)
    return;
  engine->stopAccept.store(true, std::memory_order_release);
  engine->stopRpc.store(true, std::memory_order_release);
  engine->acceptAbortFlag = 1;
  /* unblock the rpc thread parked in bootstrapP2pAccept (same trick as
     the ibrc engine: closing the listen socket fails the accept) */
  if (engine->bsListenState != nullptr && engine->bsListenState->p2p != nullptr)
    flagcxSocketClose(&engine->bsListenState->p2p->sock);
}

void flagcxAcclEngineDestroy(FlagcxP2pEngine *e) {
  FlagcxAcclEngine *engine = E(e);
  if (engine == nullptr)
    return;

  flagcxAcclEngineStopAccept(e);
  if (engine->rpcThread.joinable() &&
      engine->rpcThread.get_id() != std::this_thread::get_id())
    engine->rpcThread.join();

  engine->stopNotif.store(true, std::memory_order_release);
  if (engine->notifThread.joinable())
    engine->notifThread.join();
  if (engine->notifActive) {
    flagcxSocketClose(&engine->notifListenSock);
    engine->notifActive = false;
  }

  {
    std::lock_guard<std::mutex> lk(engine->sessMu);
    for (auto &kv : engine->sessions)
      flagcxAcclEngineConnDestroy(kv.second);
    engine->sessions.clear();
  }
  {
    std::lock_guard<std::mutex> lk(engine->accMu);
    for (auto *c : engine->accepted)
      flagcxAcclEngineConnDestroy(c);
    engine->accepted.clear();
  }

  if (engine->bsListenState != nullptr) {
    bootstrapClose(engine->bsListenState);
    engine->bsListenState = nullptr;
  }

  {
    std::lock_guard<std::mutex> lk(engine->mrMu);
    for (auto &kv : engine->mrByBase) {
      engine->mempool->DeregUserMr(reinterpret_cast<void *>(kv.second.baseAddr),
                                   kv.second.dtype);
    }
    engine->mrByBase.clear();
  }

  /* teardown order per vendor contract: contexts, connector/listener,
     threadpools, mempool last */
  for (auto *ctx : engine->serverCtxs) {
    ctx->Shutdown();
    ctx->WaitStop();
    delete ctx;
  }
  for (auto *ctx : engine->clientCtxs) {
    ctx->Shutdown();
    ctx->WaitStop();
    delete ctx;
  }
  engine->serverCtxs.clear();
  engine->clientCtxs.clear();
  if (engine->connector != nullptr) {
    engine->connector->Shutdown();
    engine->connector->WaitStop();
    delete engine->connector;
  }
  if (engine->listener != nullptr) {
    engine->listener->Shutdown();
    engine->listener->WaitStop();
    delete engine->listener;
  }
  if (engine->tpServer != nullptr) {
    engine->tpServer->Shutdown();
    engine->tpServer->WaitStop();
    delete engine->tpServer;
  }
  if (engine->tpClient != nullptr) {
    engine->tpClient->Shutdown();
    engine->tpClient->WaitStop();
    delete engine->tpClient;
  }
  if (engine->mempool != nullptr) {
    engine->mempool->Shutdown();
    engine->mempool->WaitStop();
    delete engine->mempool;
  }
  delete engine;
}

FlagcxP2pConn *flagcxAcclEngineConnect(FlagcxP2pEngine *e, const char *ipAddr,
                                       int remoteGpuIdx, int remotePort,
                                       bool sameProcess) {
  FlagcxAcclEngine *engine = E(e);
  if (engine == nullptr || ipAddr == nullptr)
    return nullptr;

  struct flagcxBootstrapHandle bsHandle;
  memset(&bsHandle, 0, sizeof(bsHandle));
  bsHandle.magic = FLAGCX_SOCKET_MAGIC;
  char ipPortStr[256];
  snprintf(ipPortStr, sizeof(ipPortStr), "%s:%d", ipAddr, remotePort);
  if (flagcxSocketGetAddrFromString(&bsHandle.addr, ipPortStr) != flagcxSuccess)
    return nullptr;

  struct bootstrapState *bsConn = nullptr;
  if (bootstrapP2pConnect(&bsHandle, FLAGCX_SOCKET_MAGIC, NULL, &bsConn) !=
      flagcxSuccess)
    return nullptr;

  auto *conn = new FlagcxAcclConn();
  conn->engine = engine;
  conn->initiator = true;
  conn->peerAddr = bsHandle.addr;
  memset(&conn->notifSock, 0, sizeof(conn->notifSock));

  const int peerBarexPort = acclHandshake(engine, bsConn, conn);
  bootstrapClose(bsConn);
  if (peerBarexPort <= 0) {
    delete conn;
    return nullptr;
  }
  (void)remoteGpuIdx;
  (void)sameProcess; /* v1: no IPC fast path */

  /* data-plane channels: qpsPerCtx per client ctx. Control block shared
     with callbacks; on timeout a late callback sees `abandoned` and
     destroys its own channel instead of touching freed state. */
  const auto &config = flagcxP2pGlobalConfig();
  const int qps = config.qpsPerConn;
  const int total = qps * (int)engine->clientCtxs.size();
  auto ctl = std::make_shared<AcclConnectCtl>(total);
  for (int i = 0; i < total; i++) {
    BarexResult r = engine->connector->Connect(
        std::string(ipAddr), peerBarexPort, [ctl](XChannel *ch, Status s) {
          std::lock_guard<std::mutex> lk(ctl->mu);
          if (s.IsOk() && ch != nullptr) {
            if (ctl->abandoned)
              ch->Destroy();
            else
              ctl->channels.push_back(ch);
          } else {
            ctl->anyFailed = true;
          }
          if (--ctl->remaining <= 0)
            ctl->cv.notify_all();
        });
    if (r != BAREX_SUCCESS) {
      std::lock_guard<std::mutex> lk(ctl->mu);
      ctl->anyFailed = true;
      if (--ctl->remaining <= 0)
        ctl->cv.notify_all();
    }
  }
  bool allUp = false;
  {
    std::unique_lock<std::mutex> lk(ctl->mu);
    ctl->cv.wait_for(lk, std::chrono::seconds(480),
                     [&] { return ctl->remaining <= 0; });
    ctl->abandoned = true; /* late callbacks self-clean from here on */
    allUp = ctl->remaining <= 0 && !ctl->anyFailed;
    conn->channels = std::move(ctl->channels);
  }
  if (conn->channels.empty()) {
    WARN("NET/ACCL_P2P : connect to %s:%d produced no channels", ipAddr,
         peerBarexPort);
    flagcxAcclEngineConnDestroy(COut(conn));
    return nullptr;
  }
  if (!allUp)
    WARN("NET/ACCL_P2P : %zu/%d channels up (continuing)",
         conn->channels.size(), total);

  connectNotif(conn);
  INFO(FLAGCX_INIT, "NET/ACCL_P2P : connected %s:%d (%zu channels)", ipAddr,
       peerBarexPort, conn->channels.size());
  return COut(conn);
}

FlagcxP2pConn *flagcxAcclEngineAccept(FlagcxP2pEngine *e, char *ipAddrBuf,
                                      size_t ipAddrBufLen, int *remoteGpuIdx) {
  FlagcxAcclEngine *engine = E(e);
  if (engine == nullptr || ipAddrBuf == nullptr || remoteGpuIdx == nullptr)
    return nullptr;
  if (engine->stopAccept.load(std::memory_order_acquire))
    return nullptr;
  if (engine->bsListenState == nullptr)
    return nullptr;

  struct bootstrapState *bsConn = nullptr;
  if (bootstrapP2pAccept(engine->bsListenState, &bsConn) != flagcxSuccess)
    return nullptr;
  if (engine->stopAccept.load(std::memory_order_acquire)) {
    bootstrapClose(bsConn);
    return nullptr;
  }

  auto *conn = new FlagcxAcclConn();
  conn->engine = engine;
  conn->initiator = false;
  memset(&conn->notifSock, 0, sizeof(conn->notifSock));
  flagcxSocketGetAddr(&bsConn->p2p->sock, &conn->peerAddr);

  const int peerBarexPort = acclHandshake(engine, bsConn, conn);
  bootstrapClose(bsConn);
  if (peerBarexPort <= 0) {
    delete conn;
    return nullptr;
  }

  const std::string host = addrHostString(&conn->peerAddr);
  snprintf(ipAddrBuf, ipAddrBufLen, "%s", host.c_str());
  *remoteGpuIdx = conn->remoteGpuIdx;

  /* Data channels were accepted passively by our XListener into the
     server contexts; the target side posts nothing in v1. */
  connectNotif(conn);
  return COut(conn);
}

void flagcxAcclEngineConnDestroy(FlagcxP2pConn *c) {
  FlagcxAcclConn *conn = C(c);
  if (conn == nullptr)
    return;
  FlagcxAcclEngine *engine = conn->engine;
  if (engine != nullptr && engine->connector != nullptr) {
    for (auto *ch : conn->channels) {
      if (ch == nullptr)
        continue;
      engine->connector->CloseChannel(ch, [ch](Status) { ch->Destroy(); });
    }
  }
  conn->channels.clear();
  if (conn->notifConnected) {
    flagcxSocketClose(&conn->notifSock);
    conn->notifConnected = false;
  }
  delete conn;
}

bool flagcxAcclEngineConnIsLocal(FlagcxP2pConn *c) {
  FlagcxAcclConn *conn = C(c);
  return conn != nullptr && conn->isLocal;
}

int flagcxAcclEngineReg(FlagcxP2pEngine *e, uintptr_t data, size_t size,
                        FlagcxP2pMr &mrId) {
  FlagcxAcclEngine *engine = E(e);
  if (engine == nullptr || data == 0 || size == 0)
    return -1;

  {
    std::lock_guard<std::mutex> lk(engine->mrMu);
    auto it = engine->mrByBase.find(data);
    if (it != engine->mrByBase.end()) {
      if (it->second.regBase != data || it->second.regSize != size) {
        WARN("NET/ACCL_P2P : re-register 0x%lx with different size",
             (unsigned long)data);
        return -1;
      }
      mrId = it->second.mrId;
      return 0;
    }
  }

  device_type dtype;
  int devId;
  classifyPtr(reinterpret_cast<void *>(data), &dtype, &devId);

  /* vsolar rejects GPU MRs above ~64MB (ibv_reg_mr ENOMEM), so register in
     chunks like Mooncake's barex transport does (eic_max_block_size). */
  const size_t chunkBytes =
      engine->mrChunkBytes > 0 ? engine->mrChunkBytes : size;
  std::vector<AcclMrEntry> chunks;
  for (size_t off = 0; off < size; off += chunkBytes) {
    const uintptr_t cbase = data + off;
    const size_t csize = std::min(chunkBytes, size - off);
    memp_t mem;
    BarexResult r = engine->mempool->RegUserMr(
        mem, reinterpret_cast<void *>(cbase), csize, dtype, devId);
    if (r != BAREX_SUCCESS) {
      WARN("NET/ACCL_P2P : RegUserMr(%p,%zu,%s,dev%d) failed: %s "
           "(chunk %zu/%zu of %p+%zu; VMM memory cannot be registered — "
           "run with FLAGCX_VMM_ENABLE=0)",
           reinterpret_cast<void *>(cbase), csize, dtype == GPU ? "GPU" : "CPU",
           devId, bxstr(r), off / chunkBytes + 1,
           (size + chunkBytes - 1) / chunkBytes, reinterpret_cast<void *>(data),
           size);
      for (const AcclMrEntry &c : chunks)
        engine->mempool->DeregUserMr(reinterpret_cast<void *>(c.baseAddr),
                                     dtype);
      return -1;
    }

    AcclMrEntry entry;
    memset(&entry, 0, sizeof(entry));
    entry.baseAddr = cbase;
    entry.size = csize;
    entry.regBase = data;
    entry.regSize = size;
    entry.dtype = dtype;
    entry.deviceId = devId;
    entry.nKeys = 0;
    for (auto &kv : mem.mrs) {
      const int nic = kv.first;
      if (nic < 0 || nic >= kMaxNics || kv.second == nullptr) {
        WARN("NET/ACCL_P2P : unexpected mr map entry nic=%d", nic);
        continue;
      }
      entry.lkeys[nic] = kv.second->lkey;
      entry.rkeys[nic] = kv.second->rkey;
      if ((uint32_t)(nic + 1) > entry.nKeys)
        entry.nKeys = nic + 1;
    }
    if (entry.nKeys == 0) {
      engine->mempool->DeregUserMr(reinterpret_cast<void *>(cbase), dtype);
      for (const AcclMrEntry &c : chunks)
        engine->mempool->DeregUserMr(reinterpret_cast<void *>(c.baseAddr),
                                     dtype);
      return -1;
    }
    chunks.push_back(entry);
  }

  std::lock_guard<std::mutex> lk(engine->mrMu);
  auto raced = engine->mrByBase.find(data);
  if (raced != engine->mrByBase.end()) {
    /* concurrent Reg of the same base won the race between our dedup
       check and this insert; keep theirs, drop our duplicate MRs */
    for (const AcclMrEntry &c : chunks)
      engine->mempool->DeregUserMr(reinterpret_cast<void *>(c.baseAddr), dtype);
    mrId = raced->second.mrId;
    return 0;
  }
  const uint64_t id = engine->nextMrId++;
  for (AcclMrEntry &c : chunks) {
    c.mrId = id;
    engine->mrByBase[c.baseAddr] = c;
  }
  mrId = id;
  return 0;
}

void flagcxAcclEngineMrDestroy(FlagcxP2pEngine *e, FlagcxP2pMr mr) {
  FlagcxAcclEngine *engine = E(e);
  if (engine == nullptr)
    return;
  std::lock_guard<std::mutex> lk(engine->mrMu);
  for (auto it = engine->mrByBase.begin(); it != engine->mrByBase.end();) {
    if (it->second.mrId == mr) {
      engine->mempool->DeregUserMr(reinterpret_cast<void *>(it->first),
                                   it->second.dtype);
      it = engine->mrByBase.erase(it);
    } else {
      ++it;
    }
  }
}

int flagcxAcclEnginePrepareDesc(FlagcxP2pEngine *e, FlagcxP2pMr mr,
                                const void *data, size_t size, char *descBuf) {
  FlagcxAcclEngine *engine = E(e);
  if (engine == nullptr || data == nullptr || descBuf == nullptr)
    return -1;
  std::lock_guard<std::mutex> lk(engine->mrMu);
  const uintptr_t addr = (uintptr_t)data;
  for (auto &kv : engine->mrByBase) {
    const AcclMrEntry &entry = kv.second;
    if (entry.mrId != mr)
      continue;
    /* chunked mrId: pick the chunk containing data. The 64B desc can only
       carry one chunk's rkeys; a peer writing across chunk boundaries must
       resolve per-chunk keys from its handshake region table. */
    if (addr < entry.baseAddr || addr >= entry.baseAddr + entry.size)
      continue;
    FlagcxP2pRdmaDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.addr = (uint64_t)addr;
    desc.size = (uint32_t)size;
    fillDescKeys(&desc, entry.rkeys, entry.nKeys);
    flagcxP2pSerializeRdmaDesc(desc, descBuf);
    return 0;
  }
  return -1;
}

int flagcxAcclEngineMakeDesc(FlagcxP2pConn *c, uint64_t remoteVa, uint32_t size,
                             FlagcxP2pRdmaDesc *desc) {
  FlagcxAcclConn *conn = C(c);
  if (conn == nullptr || desc == nullptr)
    return -1;
  /* the range may cross chunk boundaries — validate against merged spans;
     acclSubmit re-resolves per-chunk rkeys, the desc carries the first
     chunk's keys for legacy/single-chunk consumers. */
  for (const auto &span : conn->remoteSpans) {
    if (remoteVa >= span.first && remoteVa + size <= span.first + span.second) {
      const AcclRemoteRegion *r = findRemoteRegion(conn, remoteVa);
      if (r == nullptr)
        return -1;
      memset(desc, 0, sizeof(*desc));
      desc->addr = remoteVa;
      desc->size = size;
      fillDescKeys(desc, r->rkeys, r->nKeys);
      return 0;
    }
  }
  return -1;
}

int flagcxAcclEngineRead(FlagcxP2pConn *c, FlagcxP2pMr mr, const void *data,
                         size_t size, FlagcxP2pRdmaDesc desc,
                         uint64_t *transferId) {
  (void)mr;
  FlagcxAcclConn *conn = C(c);
  if (conn == nullptr || data == nullptr || transferId == nullptr)
    return -1;
  std::vector<void *> localVec(1, const_cast<void *>(data));
  std::vector<size_t> sizeVec(1, size);
  std::vector<FlagcxP2pRdmaDesc> descs(1, desc);
  return acclSubmit(conn, localVec, sizeVec, descs, 1, true, transferId);
}

int flagcxAcclEngineWrite(FlagcxP2pConn *c, FlagcxP2pMr mr, const void *data,
                          size_t size, FlagcxP2pRdmaDesc desc,
                          uint64_t *transferId) {
  (void)mr;
  FlagcxAcclConn *conn = C(c);
  if (conn == nullptr || data == nullptr || transferId == nullptr)
    return -1;
  std::vector<void *> localVec(1, const_cast<void *>(data));
  std::vector<size_t> sizeVec(1, size);
  std::vector<FlagcxP2pRdmaDesc> descs(1, desc);
  return acclSubmit(conn, localVec, sizeVec, descs, 1, false, transferId);
}

int flagcxAcclEngineReadVector(FlagcxP2pConn *c,
                               const std::vector<FlagcxP2pMr> &mrIds,
                               const std::vector<void *> &dstVec,
                               const std::vector<size_t> &sizeVec,
                               const std::vector<FlagcxP2pRdmaDesc> &descs,
                               int numIovs, uint64_t *transferId) {
  (void)mrIds;
  FlagcxAcclConn *conn = C(c);
  if (conn == nullptr || numIovs <= 0 || transferId == nullptr)
    return -1;
  if (dstVec.size() < (size_t)numIovs || sizeVec.size() < (size_t)numIovs ||
      descs.size() < (size_t)numIovs)
    return -1;
  return acclSubmit(conn, dstVec, sizeVec, descs, numIovs, true, transferId);
}

int flagcxAcclEngineWriteVector(FlagcxP2pConn *c,
                                const std::vector<FlagcxP2pMr> &mrIds,
                                const std::vector<void *> &srcVec,
                                const std::vector<size_t> &sizeVec,
                                const std::vector<FlagcxP2pRdmaDesc> &descs,
                                int numIovs, uint64_t *transferId) {
  (void)mrIds;
  FlagcxAcclConn *conn = C(c);
  if (conn == nullptr || numIovs <= 0 || transferId == nullptr)
    return -1;
  if (srcVec.size() < (size_t)numIovs || sizeVec.size() < (size_t)numIovs ||
      descs.size() < (size_t)numIovs)
    return -1;
  return acclSubmit(conn, srcVec, sizeVec, descs, numIovs, false, transferId);
}

bool flagcxAcclEngineXferStatus(FlagcxP2pConn *c, uint64_t transferId) {
  FlagcxAcclConn *conn = C(c);
  if (conn == nullptr || transferId == 0)
    return true;
  FlagcxAcclEngine *engine = conn->engine;
  std::shared_ptr<AcclXfer> xfer;
  {
    std::lock_guard<std::mutex> lk(engine->xferMu);
    auto it = engine->xfers.find(transferId);
    if (it == engine->xfers.end())
      return true;
    xfer = it->second;
  }
  if (xfer->pending.load(std::memory_order_acquire) > 0)
    return false;
  if (xfer->failed.load(std::memory_order_acquire) > 0)
    WARN("NET/ACCL_P2P : transfer %llu completed with failures",
         (unsigned long long)transferId);
  std::lock_guard<std::mutex> lk(engine->xferMu);
  engine->xfers.erase(transferId);
  return true;
}

int flagcxAcclEngineWriteVectorSync(
    FlagcxP2pConn *c, const std::vector<FlagcxP2pMr> &mrIds,
    const std::vector<void *> &srcVec, const std::vector<size_t> &sizeVec,
    const std::vector<FlagcxP2pRdmaDesc> &descs) {
  const int numIovs = (int)srcVec.size();
  if (numIovs <= 0)
    return 0;
  uint64_t transferId = 0;
  const int rc = flagcxAcclEngineWriteVector(c, mrIds, srcVec, sizeVec, descs,
                                             numIovs, &transferId);
  if (rc != 0)
    return rc;
  while (!flagcxAcclEngineXferStatus(c, transferId))
    std::this_thread::yield();
  return 0;
}

int flagcxAcclEngineGetMetadata(FlagcxP2pEngine *e, char **metadataStr) {
  FlagcxAcclEngine *engine = E(e);
  if (engine == nullptr || metadataStr == nullptr)
    return -1;
  if (engine->bsListenState == nullptr || engine->bsListenPort <= 0)
    return -1;
  union flagcxSocketAddress bsAddr;
  flagcxSocketGetAddr(&engine->bsListenState->p2p->sock, &bsAddr);
  const std::string host = addrHostString(&bsAddr);
  if (host.empty())
    return -1;
  const std::string result = host + ":" + std::to_string(engine->bsListenPort) +
                             "?" + std::to_string(engine->localGpuIdx) + "?" +
                             std::to_string(engine->notifPort);
  *metadataStr = new char[result.length() + 1];
  std::strcpy(*metadataStr, result.c_str());
  return 0;
}

int flagcxAcclEngineGetRpcPort(FlagcxP2pEngine *e) {
  FlagcxAcclEngine *engine = E(e);
  if (engine == nullptr || engine->bsListenPort <= 0)
    return -1;
  return engine->bsListenPort;
}

int flagcxAcclEngineStartRpcServer(FlagcxP2pEngine *e) {
  FlagcxAcclEngine *engine = E(e);
  if (engine == nullptr)
    return -1;
  bool expected = false;
  if (!engine->rpcActive.compare_exchange_strong(expected, true))
    return 0;
  engine->rpcThread = std::thread([engine]() {
    char ipBuf[256];
    while (!engine->stopRpc.load(std::memory_order_acquire)) {
      int remoteGpu = -1;
      FlagcxP2pConn *conn = flagcxAcclEngineAccept(EOut(engine), ipBuf,
                                                   sizeof(ipBuf), &remoteGpu);
      if (engine->stopRpc.load(std::memory_order_acquire)) {
        if (conn != nullptr)
          flagcxAcclEngineConnDestroy(conn);
        break;
      }
      if (conn == nullptr)
        continue;
      std::lock_guard<std::mutex> lk(engine->accMu);
      engine->accepted.push_back(conn);
    }
    engine->rpcActive.store(false, std::memory_order_release);
  });
  INFO(FLAGCX_INIT, "NET/ACCL_P2P : RPC server started (port=%d)",
       engine->bsListenPort);
  return 0;
}

FlagcxP2pConn *flagcxAcclEngineGetConn(FlagcxP2pEngine *e,
                                       const char *session) {
  FlagcxAcclEngine *engine = E(e);
  if (engine == nullptr || session == nullptr)
    return nullptr;
  const std::string key(session);
  {
    std::lock_guard<std::mutex> lk(engine->sessMu);
    auto it = engine->sessions.find(key);
    if (it != engine->sessions.end())
      return it->second;
  }
  const size_t pos = key.rfind(':');
  if (pos == std::string::npos)
    return nullptr;
  std::string host = key.substr(0, pos);
  const int port = atoi(key.substr(pos + 1).c_str());
  if (host.size() >= 2 && host.front() == '[' && host.back() == ']')
    host = host.substr(1, host.size() - 2);

  FlagcxP2pConn *conn =
      flagcxAcclEngineConnect(EOut(engine), host.c_str(), -1, port, false);
  if (conn == nullptr)
    return nullptr;
  std::lock_guard<std::mutex> lk(engine->sessMu);
  auto it = engine->sessions.find(key);
  if (it != engine->sessions.end()) {
    flagcxAcclEngineConnDestroy(conn);
    return it->second;
  }
  engine->sessions[key] = conn;
  return conn;
}

int flagcxAcclEngineSendNotif(FlagcxP2pConn *c, FlagcxP2pNotifyMsg *notifyMsg) {
  FlagcxAcclConn *conn = C(c);
  if (conn == nullptr || notifyMsg == nullptr)
    return -1;
  if (!conn->notifConnected && connectNotif(conn) != 0)
    return -1;
  AcclNotifWireMsg wire;
  memset(&wire, 0, sizeof(wire));
  wire.magic = kAcclNotifMagic;
  wire.payload = *notifyMsg;
  const char *p = reinterpret_cast<const char *>(&wire);
  size_t left = sizeof(wire);
  while (left > 0) {
    ssize_t s = send(conn->notifSock.fd, p, left, MSG_NOSIGNAL);
    if (s <= 0) {
      if (s < 0 && errno == EINTR)
        continue;
      return -1;
    }
    p += s;
    left -= (size_t)s;
  }
  return (int)sizeof(wire);
}

int flagcxAcclEngineGetIpcInfo(FlagcxP2pEngine *e, uintptr_t addr, char *ipcBuf,
                               bool *hasIpc) {
  (void)e;
  (void)addr;
  (void)ipcBuf;
  if (hasIpc)
    *hasIpc = false; /* v1: RDMA even intra-node */
  return 0;
}

#endif /* USE_ACCL_BAREX */
