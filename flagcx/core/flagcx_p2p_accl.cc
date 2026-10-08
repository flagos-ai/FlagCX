/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX P2P engine: ACCL (accl::barex) transport for PPU + vsolar
 * hosts, where GPU memory is registered and moved through ACCL. The public
 * ACCL interface registers a mapped VA through RegUserMr and does not expose
 * a DMA-BUF fd/offset registration entry point.
 *
 * Shape mirrors Mooncake's barex_transport: one XSimpleMempool over all
 * selected NICs (RegUserMr returns one MR/rkey per NIC); one server + client
 * XContext pair on the NIC closest to the local GPU; XListener/XConnector own
 * setup (no QP here);
 * transfers post via XChannel::WriteBatch/ReadBatch with callback
 * completion (no CQ poll); the per-slice remote key comes from the
 * region's per-NIC rkey vector via channel->GetPeerNicId().
 *
 * Rendezvous reuses FlagCX bootstrap with an ACCL hello (magic rejects
 * ibrc peers). The 64-byte FlagcxP2pRdmaDesc is kept: rkeys[0] at ibrc's
 * .rkey offset, count in .nmsgs, rkeys[1..7] in .padding. Both bootstrap
 * peers create outbound data channels, so either returned connection can
 * initiate one-sided transfers. There is no IPC path and no two-sided path.
 ************************************************************************/

#ifdef USE_ACCL_BAREX

#include "flagcx_p2p_accl.h"

#include "adaptor.h"
#include "bootstrap.h"
#include "debug.h"
#include "p2p_control.h"
#include "p2p_engine_transport.h"
#include "p2p_scheduler.h"
#include "p2p_topo.h"
#include "p2p_visibility.h"
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
#include <cerrno>
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
#include <unordered_set>
#include <utility>
#include <vector>

using namespace accl::barex;

namespace {

constexpr uint64_t kAcclHelloMagic = 0xACC1F1A6C0DE0001ull;
constexpr uint32_t kAcclNotifMagic = 0xDEADDEADu; /* same wire as ibrc */
constexpr uint32_t kAcclMrQueryMagic = 0xACC1A001u;
constexpr uint32_t kAcclMrReplyMagic = 0xACC1A002u;
constexpr uint32_t kAcclHelloDynamicMr = 1u << 0;
constexpr uint32_t kAcclHelloLocal = 1u << 1;
constexpr uint32_t kAcclHelloSameProcess = 1u << 2;
constexpr int kMaxNics = kFlagcxP2pMaxQpsPerEngine; /* 8, matches desc */
constexpr uint32_t kEmbeddedRkeysRid = UINT32_C(0xacc10001);
constexpr uint32_t kCachedRkeysRid = UINT32_C(0xacc10002);

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
  uint32_t mrId;
};
static_assert(sizeof(AcclMemRegWire) == 56, "stable wire layout");

struct AcclNotifWireMsg {
  uint32_t magic;
  uint32_t reserved;
  FlagcxP2pNotifyMsg payload;
};

struct AcclMrQueryWire {
  uint32_t magic;
  uint32_t reserved;
  uint64_t addr;
  uint64_t size;
  uint64_t mrId;
};

struct AcclMrReplyWire {
  uint32_t magic;
  int32_t status;
  uint32_t count;
  uint32_t reserved;
};

struct AcclRemoteRegion {
  uint64_t baseAddr;
  uint64_t size;
  uint32_t mrId;
  uint32_t nKeys;
  uint32_t rkeys[kMaxNics];
};

struct AcclRemoteSpan {
  uint64_t baseAddr;
  uint64_t size;
  uint32_t mrId;
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

using AcclXfer = flagcxP2pScheduling::CompletionTracker;

struct AcclChannelOwner {
  std::mutex mu;
  XChannel *channel;
  bool fallbackActive = false;
  bool callbackArrived = false;

  explicit AcclChannelOwner(XChannel *value) : channel(value) {}
};

enum AcclConnLifecycle : int {
  ACCL_CONN_ACTIVE = 0,
  ACCL_CONN_FAILED = 1,
  ACCL_CONN_CLOSING = 2,
  ACCL_CONN_CLOSED = 3,
};

struct AcclConnState {
  std::atomic<int> lifecycle{ACCL_CONN_ACTIVE};
  std::atomic<int> firstError{0};

  void fail(int result) {
    if (result == 0)
      result = -1;
    int expectedError = 0;
    firstError.compare_exchange_strong(expectedError, result,
                                       std::memory_order_acq_rel);
    int expectedState = ACCL_CONN_ACTIVE;
    lifecycle.compare_exchange_strong(expectedState, ACCL_CONN_FAILED,
                                      std::memory_order_acq_rel);
  }
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
  int selectedNetDev = -1;
  struct flagcxP2pTopoManager *topoMgr = nullptr;

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
  uint32_t nextMrId = 1;
  std::map<uintptr_t, AcclMrEntry> mrByBase; /* keyed by chunk base */
  std::unordered_map<uint64_t, std::vector<AcclMrEntry>> pendingMrDeregs;
  std::vector<AcclMrEntry> pendingOrphanDeregs;
  /* vsolar caps a single GPU MR at 64MB, so registrations are split into
     chunks of at most this many bytes (FLAGCX_ACCL_MAX_MR_MB, 0 = off). */
  size_t mrChunkBytes = 64ull << 20;
  std::atomic<uint64_t> runtimeSliceConfig{0};

  std::mutex xferMu;
  uint64_t nextXferId = 1;
  std::unordered_map<uint64_t, std::shared_ptr<AcclXfer>> xfers;
  std::mutex deferredChannelMu;
  std::vector<std::shared_ptr<AcclChannelOwner>> deferredChannels;

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
  int remoteGpuIdx = -1;
  int remoteNotifPort = 0;
  bool isLocal = false;
  bool sameProcess = false;
  std::shared_ptr<AcclConnState> state = std::make_shared<AcclConnState>();

  union flagcxSocketAddress peerAddr; /* host part; for notif connect */
  struct flagcxSocket notifSock;
  bool notifConnected = false;
  bool peerSupportsDynamicMr = false;
  std::mutex notifMu;
  std::mutex remoteMrMu;

  std::vector<AcclRemoteRegion> remoteRegions; /* one per remote chunk */
  /* merged contiguous extents of remoteRegions, for range validation
     (chunks of one logical registration are contiguous by construction) */
  std::vector<AcclRemoteSpan> remoteSpans;
  std::vector<XChannel *> channels; /* locally-created outbound channels */
  std::atomic<uint64_t> rr{0};
  std::mutex xferMu;
  std::unordered_set<uint64_t> xferIds;
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

const char *bxstr(BarexResult r);

bool acclMrRangesOverlap(uintptr_t firstBase, size_t firstSize,
                         uintptr_t secondBase, size_t secondSize) {
  if (firstSize == 0 || secondSize == 0)
    return firstBase == secondBase;
  if (firstBase <= secondBase)
    return secondBase - firstBase < firstSize;
  return firstBase - secondBase < secondSize;
}

/* RegUserMr/DeregUserMr use (base, dtype) as the provider identity. Before a
   new registration reuses a range whose previous deregistration failed, retry
   and remove the old provider registration. Publishing a replacement first
   would let a later retry invalidate the replacement MR. mrMu must be held. */
bool acclRetryConflictingDeregs(FlagcxAcclEngine *engine, uintptr_t base,
                                size_t size, device_type dtype) {
  bool conflict = false;
  for (auto pending = engine->pendingMrDeregs.begin();
       pending != engine->pendingMrDeregs.end();) {
    auto &chunks = pending->second;
    for (auto chunk = chunks.begin(); chunk != chunks.end();) {
      if (chunk->dtype != dtype ||
          !acclMrRangesOverlap(base, size, chunk->baseAddr, chunk->size)) {
        ++chunk;
        continue;
      }
      const BarexResult result = engine->mempool->DeregUserMr(
          reinterpret_cast<void *>(chunk->baseAddr), chunk->dtype);
      if (result == BAREX_SUCCESS) {
        chunk = chunks.erase(chunk);
      } else {
        WARN("NET/ACCL_P2P : cannot register [%p,+%zu) while overlapping MR "
             "deregistration is pending: %s",
             reinterpret_cast<void *>(base), size, bxstr(result));
        conflict = true;
        ++chunk;
      }
    }
    if (chunks.empty())
      pending = engine->pendingMrDeregs.erase(pending);
    else
      ++pending;
  }

  for (auto chunk = engine->pendingOrphanDeregs.begin();
       chunk != engine->pendingOrphanDeregs.end();) {
    if (chunk->dtype != dtype ||
        !acclMrRangesOverlap(base, size, chunk->baseAddr, chunk->size)) {
      ++chunk;
      continue;
    }
    const BarexResult result = engine->mempool->DeregUserMr(
        reinterpret_cast<void *>(chunk->baseAddr), chunk->dtype);
    if (result == BAREX_SUCCESS) {
      chunk = engine->pendingOrphanDeregs.erase(chunk);
    } else {
      WARN("NET/ACCL_P2P : cannot register [%p,+%zu) while overlapping "
           "orphan MR deregistration is pending: %s",
           reinterpret_cast<void *>(base), size, bxstr(result));
      conflict = true;
      ++chunk;
    }
  }
  return !conflict;
}

bool closeAndDeleteAcclChannel(XConnector *connector,
                               const std::shared_ptr<AcclChannelOwner> &owner) {
  if (connector == nullptr || owner == nullptr)
    return false;
  uint64_t retry = 0;
  while (true) {
    XChannel *channel = nullptr;
    {
      std::lock_guard<std::mutex> lock(owner->mu);
      if (owner->channel == nullptr)
        return true;
      owner->fallbackActive = true;
      channel = owner->channel;
    }
    BarexResult result;
    result = connector->CloseAndDeleteChannel(channel);
    XChannel *callbackOwned = nullptr;
    {
      std::lock_guard<std::mutex> lock(owner->mu);
      if (result == BAREX_SUCCESS) {
        owner->channel = nullptr;
      } else if (owner->callbackArrived) {
        callbackOwned = owner->channel;
        owner->channel = nullptr;
      }
      owner->fallbackActive = false;
      owner->callbackArrived = false;
    }
    if (callbackOwned != nullptr)
      callbackOwned->Destroy();
    if (result == BAREX_SUCCESS || callbackOwned != nullptr)
      return true;
    if (result != BAREX_ERR_CHANNEL_STAT) {
      WARN("NET/ACCL_P2P : CloseAndDeleteChannel failed: %s", bxstr(result));
      return false;
    }
    if (retry++ == 0 || retry % 1000 == 0)
      INFO(FLAGCX_NET,
           "NET/ACCL_P2P : channel cleanup is still in progress; retrying "
           "(attempt %llu)",
           (unsigned long long)retry);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

const char *bxstr(BarexResult r) {
  auto it = BarexResultStrings.find(r);
  return it == BarexResultStrings.end() ? "UNKNOWN" : it->second;
}

bool barexRetryable(BarexResult result) {
  return result == BAREX_ERR_QUEUE_FULL || result == BAREX_ERR_RATE_LIMITED;
}

int acclWorkerCount(const FlagcxP2pGlobalConfig &config) {
  return std::max(1, config.workersPerPool);
}

int acclQpsPerConn(const FlagcxP2pGlobalConfig &config) {
  return std::max(1, config.qpsPerConn);
}

uint32_t acclSliceSize(const FlagcxP2pGlobalConfig &config) {
  return static_cast<uint32_t>(config.sliceSize);
}

uint32_t acclFragmentLimit(const FlagcxP2pGlobalConfig &config,
                           uint32_t slice) {
  return std::min<uint32_t>(static_cast<uint32_t>(config.fragmentLimit), slice);
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

bool addrSameHost(const union flagcxSocketAddress *a,
                  const union flagcxSocketAddress *b) {
  if (a == nullptr || b == nullptr || a->sa.sa_family != b->sa.sa_family)
    return false;
  if (a->sa.sa_family == AF_INET)
    return a->sin.sin_addr.s_addr == b->sin.sin_addr.s_addr;
  if (a->sa.sa_family == AF_INET6) {
    return memcmp(&a->sin6.sin6_addr, &b->sin6.sin6_addr,
                  sizeof(a->sin6.sin6_addr)) == 0 &&
           a->sin6.sin6_scope_id == b->sin6.sin6_scope_id;
  }
  return false;
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

std::string hostPortString(const char *host, int port) {
  if (host == nullptr || host[0] == '\0')
    return std::string();
  const bool ipv6Literal = strchr(host, ':') != nullptr;
  const bool alreadyBracketed = host[0] == '[';
  return ipv6Literal && !alreadyBracketed
             ? "[" + std::string(host) + "]:" + std::to_string(port)
             : std::string(host) + ":" + std::to_string(port);
}

std::string addrHostPortString(const union flagcxSocketAddress *addr,
                               int port) {
  const std::string host = addrHostString(addr);
  return hostPortString(host.c_str(), port);
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

int sendAllFdAccl(int fd, const void *buf, size_t len) {
  constexpr int kWriteTimeoutMs = 5000;
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(kWriteTimeoutMs);
  const char *p = static_cast<const char *>(buf);
  size_t sent = 0;
  while (sent < len) {
    if (std::chrono::steady_clock::now() >= deadline)
      return -1;
    const ssize_t result =
        send(fd, p + sent, len - sent, MSG_NOSIGNAL | MSG_DONTWAIT);
    if (result < 0 && errno == EINTR)
      continue;
    if (result < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
      pollfd pfd{fd, POLLOUT, 0};
      int ready = -1;
      while (ready < 0) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline)
          return -1;
        const int remaining = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline -
                                                                  now)
                .count());
        ready = poll(&pfd, 1, std::max(remaining, 1));
        if (ready < 0 && errno != EINTR)
          return -1;
      }
      if (ready > 0 && !(pfd.revents & (POLLERR | POLLHUP | POLLNVAL)))
        continue;
    }
    if (result <= 0)
      return -1;
    sent += static_cast<size_t>(result);
  }
  return 0;
}

bool replyMrQuery(FlagcxAcclEngine *engine, int fd,
                  const AcclMrQueryWire &query) {
  AcclMrReplyWire reply{};
  reply.magic = kAcclMrReplyMagic;
  reply.status = -1;
  std::vector<AcclMemRegWire> regions;
  if (query.size <= UINT32_MAX && query.addr <= UINT64_MAX - query.size &&
      query.mrId != 0) {
    std::lock_guard<std::mutex> lk(engine->mrMu);
    const uint64_t end = query.addr + query.size;
    for (const auto &kv : engine->mrByBase) {
      const AcclMrEntry &entry = kv.second;
      if (entry.mrId != query.mrId || entry.baseAddr >= end ||
          entry.baseAddr + entry.size <= query.addr)
        continue;
      AcclMemRegWire region{};
      region.baseAddr = entry.baseAddr;
      region.size = entry.size;
      region.nKeys = entry.nKeys;
      region.mrId = static_cast<uint32_t>(entry.mrId);
      memcpy(region.rkeys, entry.rkeys, sizeof(region.rkeys));
      regions.push_back(region);
    }
    if (!regions.empty() && regions.size() <= 65536) {
      std::sort(regions.begin(), regions.end(),
                [](const AcclMemRegWire &a, const AcclMemRegWire &b) {
                  return a.baseAddr < b.baseAddr;
                });
      uint64_t cursor = query.addr;
      for (const AcclMemRegWire &region : regions) {
        if (cursor < region.baseAddr || cursor >= region.baseAddr + region.size)
          break;
        cursor = region.baseAddr + region.size;
        if (cursor >= end) {
          reply.status = 0;
          break;
        }
      }
    }
  }
  if (reply.status != 0)
    regions.clear();
  reply.count = static_cast<uint32_t>(regions.size());
  if (sendAllFdAccl(fd, &reply, sizeof(reply)) != 0)
    return false;
  if (regions.empty())
    return true;
  return sendAllFdAccl(fd, regions.data(),
                       regions.size() * sizeof(regions[0])) == 0;
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
      while (peer.inBuf.size() >= sizeof(uint32_t)) {
        uint32_t magic = 0;
        memcpy(&magic, peer.inBuf.data(), sizeof(magic));
        if (magic == kAcclNotifMagic) {
          if (peer.inBuf.size() < sizeof(AcclNotifWireMsg))
            break;
          AcclNotifWireMsg msg;
          memcpy(&msg, peer.inBuf.data(), sizeof(msg));
          peer.inBuf.erase(peer.inBuf.begin(),
                           peer.inBuf.begin() + sizeof(msg));
          flagcxP2pNotifyAppend(msg.payload);
          continue;
        }
        if (magic == kAcclMrQueryMagic) {
          if (peer.inBuf.size() < sizeof(AcclMrQueryWire))
            break;
          AcclMrQueryWire query;
          memcpy(&query, peer.inBuf.data(), sizeof(query));
          peer.inBuf.erase(peer.inBuf.begin(),
                           peer.inBuf.begin() + sizeof(query));
          if (!replyMrQuery(engine, peer.fd, query)) {
            ::close(peer.fd);
            peer.fd = -1;
            peer.inBuf.clear();
            break;
          }
          continue;
        }
        WARN("NET/ACCL_P2P : invalid notification/control magic 0x%x", magic);
        ::close(peer.fd);
        peer.fd = -1;
        peer.inBuf.clear();
        break;
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

bool descKeyForNic(const FlagcxP2pRdmaDesc &desc, int nic, uint32_t *rkey) {
  if (nic < 0 || rkey == nullptr)
    return false;
  return flagcxP2pDescGetKey(&desc, static_cast<uint32_t>(nic), rkey) ==
             flagcxSuccess &&
         *rkey != 0;
}

bool descContains(const FlagcxP2pRdmaDesc &desc, size_t size) {
  if (size > desc.size)
    return false;
  if (size != 0 && desc.addr == 0)
    return false;
  return desc.addr <= UINT64_MAX - size;
}

bool findMrContaining(FlagcxAcclEngine *engine, uintptr_t addr, size_t size,
                      AcclMrEntry *out) {
  std::lock_guard<std::mutex> lk(engine->mrMu);
  auto it = engine->mrByBase.upper_bound(addr);
  if (it == engine->mrByBase.begin())
    return false;
  --it;
  const AcclMrEntry &e = it->second;
  if (addr >= e.baseAddr) {
    const size_t offset = addr - e.baseAddr;
    if (offset > e.size || size > e.size - offset)
      return false;
    if (out)
      *out = e;
    return true;
  }
  return false;
}

bool registeredRangeMatchesMr(FlagcxAcclEngine *engine, FlagcxP2pMr mr,
                              uintptr_t addr, size_t size) {
  if (addr > UINTPTR_MAX - size)
    return false;
  std::lock_guard<std::mutex> lk(engine->mrMu);
  auto it = engine->mrByBase.upper_bound(addr);
  if (it == engine->mrByBase.begin())
    return false;
  --it;
  const AcclMrEntry &entry = it->second;
  if (entry.mrId != mr || addr < entry.regBase)
    return false;
  const size_t offset = addr - entry.regBase;
  return offset <= entry.regSize && size <= entry.regSize - offset;
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

bool regionKeyForNic(const AcclRemoteRegion &r, int nic, uint32_t *rkey) {
  if (rkey == nullptr || nic < 0 || nic >= kMaxNics ||
      static_cast<uint32_t>(nic) >= r.nKeys || r.rkeys[nic] == 0)
    return false;
  *rkey = r.rkeys[nic];
  return true;
}

bool hasAnyRemoteKey(const uint32_t *rkeys, uint32_t nKeys) {
  if (rkeys == nullptr)
    return false;
  for (uint32_t key = 0; key < nKeys; ++key) {
    if (rkeys[key] != 0)
      return true;
  }
  return false;
}

bool remoteRangeAvailableLocked(const FlagcxAcclConn *conn, uint64_t address,
                                uint64_t size, uint32_t mrId) {
  if (address > UINT64_MAX - size)
    return false;
  const uint64_t end = address + size;
  uint64_t cursor = address;
  while (cursor < end) {
    const AcclRemoteRegion *region = findRemoteRegion(conn, cursor);
    if (region == nullptr || region->mrId != mrId ||
        region->baseAddr > cursor || region->baseAddr + region->size <= cursor)
      return false;
    cursor = std::min(end, region->baseAddr + region->size);
  }
  return true;
}

void rebuildRemoteSpansLocked(FlagcxAcclConn *conn) {
  conn->remoteSpans.clear();
  for (const AcclRemoteRegion &region : conn->remoteRegions) {
    if (!conn->remoteSpans.empty() &&
        conn->remoteSpans.back().mrId == region.mrId &&
        conn->remoteSpans.back().baseAddr + conn->remoteSpans.back().size ==
            region.baseAddr) {
      conn->remoteSpans.back().size += region.size;
    } else {
      conn->remoteSpans.push_back({region.baseAddr, region.size, region.mrId});
    }
  }
}

int connectNotif(FlagcxAcclConn *conn);

int recvAllFdAcclDeadline(int fd, void *buffer, size_t size, int timeoutMs) {
  char *p = static_cast<char *>(buffer);
  size_t received = 0;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  while (received < size) {
    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline)
      return -1;
    const int remainingMs = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)
            .count());
    pollfd pfd{fd, POLLIN, 0};
    const int ready = poll(&pfd, 1, std::max(1, remainingMs));
    if (ready < 0 && errno == EINTR)
      continue;
    if (ready <= 0 || (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)))
      return -1;
    const ssize_t result = recv(fd, p + received, size - received, 0);
    if (result < 0 &&
        (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK))
      continue;
    if (result <= 0)
      return -1;
    received += static_cast<size_t>(result);
  }
  return 0;
}

bool refreshRemoteRange(FlagcxAcclConn *conn, const FlagcxP2pRdmaDesc &desc) {
  if (desc.rid != kCachedRkeysRid || desc.idx == 0 || desc.idx > UINT32_MAX)
    return false;
  {
    std::lock_guard<std::mutex> lk(conn->remoteMrMu);
    if (remoteRangeAvailableLocked(conn, desc.addr, desc.size,
                                   static_cast<uint32_t>(desc.idx)))
      return true;
  }
  if (!conn->peerSupportsDynamicMr)
    return false;

  std::lock_guard<std::mutex> controlLock(conn->notifMu);
  {
    std::lock_guard<std::mutex> lk(conn->remoteMrMu);
    if (remoteRangeAvailableLocked(conn, desc.addr, desc.size,
                                   static_cast<uint32_t>(desc.idx)))
      return true;
  }
  if (!conn->notifConnected && connectNotif(conn) != 0)
    return false;

  auto failControlExchange = [&]() {
    if (conn->notifConnected) {
      flagcxSocketClose(&conn->notifSock);
      conn->notifConnected = false;
    }
    return false;
  };

  AcclMrQueryWire query{};
  query.magic = kAcclMrQueryMagic;
  query.addr = desc.addr;
  query.size = desc.size;
  query.mrId = desc.idx;
  if (sendAllFdAccl(conn->notifSock.fd, &query, sizeof(query)) != 0)
    return failControlExchange();

  AcclMrReplyWire reply{};
  if (recvAllFdAcclDeadline(conn->notifSock.fd, &reply, sizeof(reply), 5000) !=
          0 ||
      reply.magic != kAcclMrReplyMagic || reply.status != 0 ||
      reply.count == 0 || reply.count > 65536)
    return failControlExchange();
  std::vector<AcclMemRegWire> wireRegions(reply.count);
  if (recvAllFdAcclDeadline(conn->notifSock.fd, wireRegions.data(),
                            wireRegions.size() * sizeof(wireRegions[0]),
                            5000) != 0)
    return failControlExchange();

  std::vector<AcclRemoteRegion> regions;
  regions.reserve(wireRegions.size());
  for (const AcclMemRegWire &wire : wireRegions) {
    if (wire.mrId != static_cast<uint32_t>(desc.idx) || wire.size == 0 ||
        wire.baseAddr > UINT64_MAX - wire.size || wire.nKeys == 0 ||
        wire.nKeys > kMaxNics)
      return failControlExchange();
    // Keys use physical NIC ids, so filtered device sets can leave holes.
    // acclSubmit validates the key for the selected peer NIC before posting.
    if (!hasAnyRemoteKey(wire.rkeys, wire.nKeys))
      return failControlExchange();
    AcclRemoteRegion region{};
    region.baseAddr = wire.baseAddr;
    region.size = wire.size;
    region.mrId = wire.mrId;
    region.nKeys = wire.nKeys;
    memcpy(region.rkeys, wire.rkeys, sizeof(region.rkeys));
    regions.push_back(region);
  }
  std::sort(regions.begin(), regions.end(),
            [](const AcclRemoteRegion &a, const AcclRemoteRegion &b) {
              return a.baseAddr < b.baseAddr;
            });

  std::lock_guard<std::mutex> lk(conn->remoteMrMu);
  const uint64_t first = regions.front().baseAddr;
  const uint64_t last = regions.back().baseAddr + regions.back().size;
  conn->remoteRegions.erase(
      std::remove_if(conn->remoteRegions.begin(), conn->remoteRegions.end(),
                     [first, last](const AcclRemoteRegion &region) {
                       return region.baseAddr < last &&
                              first < region.baseAddr + region.size;
                     }),
      conn->remoteRegions.end());
  conn->remoteRegions.insert(conn->remoteRegions.end(), regions.begin(),
                             regions.end());
  std::sort(conn->remoteRegions.begin(), conn->remoteRegions.end(),
            [](const AcclRemoteRegion &a, const AcclRemoteRegion &b) {
              return a.baseAddr < b.baseAddr;
            });
  rebuildRemoteSpansLocked(conn);
  return remoteRangeAvailableLocked(conn, desc.addr, desc.size,
                                    static_cast<uint32_t>(desc.idx));
}

bool peekBootstrapFrame(int fd, void *buffer, size_t size,
                        const std::atomic<bool> &stop, int timeoutMs = 5000) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  while (!stop.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) {
    pollfd pfd{fd, POLLIN, 0};
    const int ready = poll(&pfd, 1, 50);
    if (ready < 0) {
      if (errno == EINTR)
        continue;
      return false;
    }
    if (ready == 0)
      continue;
    const ssize_t n = recv(fd, buffer, size, MSG_PEEK | MSG_DONTWAIT);
    if (n == (ssize_t)size)
      return true;
    if (n == 0)
      return false;
    if (n < 0 && errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)
      return false;
  }
  return false;
}

/* Shared submit for single+vector read/write. */
int acclSubmit(FlagcxAcclConn *conn, const std::vector<void *> &localVec,
               const std::vector<size_t> &sizeVec,
               const std::vector<FlagcxP2pRdmaDesc> &descs, int numIovs,
               bool isRead, uint64_t *transferId) {
  if (conn == nullptr || transferId == nullptr || numIovs <= 0 ||
      localVec.size() < static_cast<size_t>(numIovs) ||
      sizeVec.size() < static_cast<size_t>(numIovs) ||
      descs.size() < static_cast<size_t>(numIovs))
    return -1;
  for (int i = 0; i < numIovs; ++i) {
    if (!descContains(descs[i], sizeVec[i]))
      return -1;
    if (descs[i].rid == kCachedRkeysRid &&
        !refreshRemoteRange(conn, descs[i])) {
      WARN("NET/ACCL_P2P : failed to resolve dynamic remote MR %llu for "
           "0x%llx+%u",
           (unsigned long long)descs[i].idx, (unsigned long long)descs[i].addr,
           descs[i].size);
      return -1;
    }
  }
  FlagcxAcclEngine *engine = conn->engine;
  if (conn->state->lifecycle.load(std::memory_order_acquire) !=
      ACCL_CONN_ACTIVE) {
    const int firstError =
        conn->state->firstError.load(std::memory_order_acquire);
    return firstError != 0 ? firstError : -1;
  }
  if (isRead) {
    const uint32_t requirements = flagcxResolveGdrFlushRequirements(
        deviceAdaptor == nullptr ? FLAGCX_GDR_FLUSH_NONE
                                 : deviceAdaptor->gdrFlushRequirements);
    for (int i = 0; i < numIovs; i++) {
      if (sizeVec[i] == 0)
        continue;
      AcclMrEntry entry;
      if (!findMrContaining(engine, reinterpret_cast<uintptr_t>(localVec[i]), 1,
                            &entry)) {
        WARN("NET/ACCL_P2P : local READ buffer %p is not registered",
             localVec[i]);
        return -1;
      }
      const int ptrType =
          entry.dtype == GPU ? FLAGCX_PTR_CUDA : FLAGCX_PTR_HOST;
      if (flagcxP2pValidateReadVisibility(requirements,
                                          FLAGCX_NET_GDR_FLUSH_NONE, ptrType,
                                          sizeVec[i], 0) != flagcxSuccess) {
        WARN("NET/ACCL_P2P : GPU READ requires a visibility flush, but BAREX "
             "has no completion flush stage");
        return -1;
      }
    }
  }
  struct ChannelGroup {
    int localNic = -1;
    int peerNic = -1;
    std::vector<XChannel *> channels;
    std::shared_ptr<std::vector<rw_memp_t>> entries =
        std::make_shared<std::vector<rw_memp_t>>();
  };
  std::vector<ChannelGroup> groups;
  const int expectedNic = engine->devs[engine->selectedNetDev]->GetId();
  for (XChannel *channel : conn->channels) {
    if (channel == nullptr || !channel->IsActive())
      continue;
    const int localNic = channel->GetContext()->GetXDevice()->GetId();
    const int peerNic = channel->GetPeerNicId();
    if (localNic != expectedNic) {
      WARN("NET/ACCL_P2P : active channel on nic %d, expected GPU-local "
           "nic %d",
           localNic, expectedNic);
      conn->state->fail(-1);
      return -1;
    }
    size_t group = groups.size();
    for (size_t i = 0; i < groups.size(); i++) {
      if (groups[i].localNic == localNic && groups[i].peerNic == peerNic) {
        group = i;
        break;
      }
    }
    if (group == groups.size()) {
      groups.push_back(ChannelGroup{});
      groups.back().localNic = localNic;
      groups.back().peerNic = peerNic;
    }
    groups[group].channels.push_back(channel);
  }
  if (groups.empty()) {
    WARN("NET/ACCL_P2P : no active channel");
    conn->state->fail(-1);
    return -1;
  }

  const uint64_t sliceConfig =
      engine->runtimeSliceConfig.load(std::memory_order_acquire);
  const size_t sliceLimit = static_cast<size_t>(sliceConfig >> 32);
  const size_t fragmentLimit = static_cast<size_t>(uint32_t(sliceConfig));
  const uint64_t submissionTicket =
      conn->rr.fetch_add(1, std::memory_order_relaxed);
  uint64_t groupCursor = submissionTicket;
  for (int i = 0; i < numIovs; i++) {
    if (sizeVec[i] == 0)
      continue;
    if (sizeVec[i] > UINT32_MAX) {
      WARN("NET/ACCL_P2P : iov %d size %zu exceeds 4GiB desc limit", i,
           sizeVec[i]);
      return -1;
    }
    uintptr_t lcur = (uintptr_t)localVec[i];
    uint64_t rcur = descs[i].addr;
    size_t remaining = sizeVec[i];
    while (remaining > 0) {
      AcclMrEntry entry;
      ChannelGroup *selected = nullptr;
      size_t selectedGroup = 0;
      for (size_t attempt = 0; attempt < groups.size(); attempt++) {
        const size_t group =
            (groupCursor + attempt) % static_cast<uint64_t>(groups.size());
        AcclMrEntry candidate;
        if (!findMrContaining(engine, lcur, 1, &candidate)) {
          WARN("NET/ACCL_P2P : local buffer %p not registered", (void *)lcur);
          return -1;
        }
        if (groups[group].localNic < 0 ||
            (uint32_t)groups[group].localNic >= candidate.nKeys ||
            candidate.lkeys[groups[group].localNic] == 0)
          continue;
        entry = candidate;
        selected = &groups[group];
        selectedGroup = group;
        break;
      }
      if (selected == nullptr) {
        WARN("NET/ACCL_P2P : no local lkey for buffer %p on active NICs",
             (void *)lcur);
        return -1;
      }
      groupCursor = selectedGroup + 1;

      size_t slice = std::min(remaining, entry.baseAddr + entry.size - lcur);
      uint32_t rkey;
      const bool useEmbeddedKeys = descs[i].rid == kEmbeddedRkeysRid;
      const bool requireCachedKeys = descs[i].rid == kCachedRkeysRid;
      AcclRemoteRegion remoteRegion{};
      bool hasRemoteRegion = false;
      if (!useEmbeddedKeys) {
        std::lock_guard<std::mutex> lk(conn->remoteMrMu);
        const AcclRemoteRegion *found = findRemoteRegion(conn, rcur);
        if (found != nullptr) {
          remoteRegion = *found;
          hasRemoteRegion = true;
        }
      }
      if (hasRemoteRegion) {
        if (requireCachedKeys && remoteRegion.mrId != descs[i].idx)
          return -1;
        slice = std::min(slice, static_cast<size_t>(remoteRegion.baseAddr +
                                                    remoteRegion.size - rcur));
        if (!regionKeyForNic(remoteRegion, selected->peerNic, &rkey))
          return -1;
      } else {
        if (requireCachedKeys ||
            !descKeyForNic(descs[i], selected->peerNic, &rkey))
          return -1;
      }
      if (sliceLimit > 0 && slice > sliceLimit &&
          slice - sliceLimit > fragmentLimit)
        slice = sliceLimit;
      if (slice == 0)
        return -1;

      rw_memp_t w{};
      w.sg.addr = (uint64_t)lcur;
      w.sg.length = (uint32_t)slice;
      w.sg.lkey = entry.lkeys[selected->localNic];
      w.data.d_type = entry.dtype;
      w.data.device_id = entry.deviceId;
      w.r_addr = rcur;
      w.r_key = rkey;
      w.r_ttl_ms = UINT64_MAX;
      selected->entries->push_back(w);
      lcur += slice;
      rcur += slice;
      remaining -= slice;
    }
  }

  struct BatchWork {
    XChannel *channel;
    std::shared_ptr<std::vector<rw_memp_t>> entries;
  };
  std::vector<BatchWork> works;
  for (auto &group : groups) {
    if (group.entries->empty())
      continue;
    const size_t qpCount =
        std::min(group.channels.size(), group.entries->size());
    const size_t batchSize = group.entries->size() / qpCount;
    const size_t remainder = group.entries->size() % qpCount;
    size_t begin = 0;
    for (size_t i = 0; i < qpCount; i++) {
      const size_t count = batchSize + (i < remainder ? 1 : 0);
      auto entries = std::make_shared<std::vector<rw_memp_t>>(
          group.entries->begin() + begin,
          group.entries->begin() + begin + count);
      const size_t channelIndex = flagcxP2pScheduling::channelForTicket(
          submissionTicket, i, group.channels.size());
      works.push_back({group.channels[channelIndex], std::move(entries)});
      begin += count;
    }
  }
  if (works.empty()) {
    *transferId = 0;
    return 0;
  }

  auto xfer = std::make_shared<AcclXfer>();
  xfer->pending.store((int)works.size(), std::memory_order_release);
  uint64_t id;
  {
    std::lock_guard<std::mutex> connLock(conn->xferMu);
    if (conn->state->lifecycle.load(std::memory_order_acquire) !=
        ACCL_CONN_ACTIVE)
      return -1;
    std::lock_guard<std::mutex> engineLock(engine->xferMu);
    id = engine->nextXferId++;
    engine->xfers[id] = xfer;
    conn->xferIds.insert(id);
  }

  std::shared_ptr<AcclConnState> connState = conn->state;
  for (size_t workIndex = 0; workIndex < works.size(); workIndex++) {
    auto &work = works[workIndex];
    const int localNic = work.channel->GetContext()->GetXDevice()->GetId();
    const int peerNic = work.channel->GetPeerNicId();
    const size_t entryCount = work.entries->size();
    DoneCallback done = [connState, xfer, entries = work.entries, localNic,
                         peerNic, entryCount](Status s) {
      (void)entries;
      int failed = 0;
      if (!s.IsOk()) {
        WARN("NET/ACCL_P2P : batch failed localNic=%d peerNic=%d entries=%zu: "
             "%s",
             localNic, peerNic, entryCount, s.ErrMsg().c_str());
        failed = 1;
        if (!barexRetryable(s.ErrCode())) {
          xfer->hardFailed.fetch_add(1, std::memory_order_release);
          connState->fail(-1);
        }
      }
      xfer->complete(1, failed);
    };
    BarexResult r = isRead ? work.channel->ReadBatch(work.entries, done, true)
                           : work.channel->WriteBatch(work.entries, done, true);
    if (r != BAREX_SUCCESS) {
      WARN("NET/ACCL_P2P : %s sync error: %s",
           isRead ? "ReadBatch" : "WriteBatch", bxstr(r));
      if (!barexRetryable(r))
        conn->state->fail(-1);
      /* The failed work and every work after it were not accepted by ACCL.
         Complete those slots locally, then drain callbacks for earlier
         accepted batches before returning.  The caller receives no transfer
         ID on an error, so returning sooner would allow its buffers to be
         reused while an accepted batch still references them. */
      const int unsubmitted = static_cast<int>(works.size() - workIndex);
      xfer->complete(unsubmitted, unsubmitted);
      xfer->wait();
      std::lock_guard<std::mutex> connLock(conn->xferMu);
      std::lock_guard<std::mutex> engineLock(engine->xferMu);
      engine->xfers.erase(id);
      conn->xferIds.erase(id);
      return -1;
    }
  }
  *transferId = id;
  return 0;
}

/* Exchange hello + desc table over an established bootstrap conn.
   Both sides call with the same tag sequence. */
int acclHandshake(FlagcxAcclEngine *engine, struct bootstrapState *bsConn,
                  FlagcxAcclConn *conn, bool sameProcessHint) {
  const bool localSameHost =
      addrSameHost(&conn->peerAddr, bootstrapGetNetIfAddr());
  AcclHelloWire localHello;
  memset(&localHello, 0, sizeof(localHello));
  localHello.magic = kAcclHelloMagic;
  localHello.barexPort = engine->barexPort;
  localHello.gpuIdx = engine->localGpuIdx;
  localHello.notifPort = engine->notifPort;
  localHello.flags = kAcclHelloDynamicMr;
  if (localSameHost)
    localHello.flags |= kAcclHelloLocal;
  if (localSameHost && sameProcessHint)
    localHello.flags |= kAcclHelloSameProcess;

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
  conn->peerSupportsDynamicMr = (remoteHello.flags & kAcclHelloDynamicMr) != 0;
  conn->isLocal = localSameHost || (remoteHello.flags & kAcclHelloLocal) != 0;
  conn->sameProcess = (localSameHost && sameProcessHint) ||
                      (remoteHello.flags & kAcclHelloSameProcess) != 0;

  /* desc table with per-NIC rkey vectors */
  std::vector<AcclMemRegWire> localTable;
  {
    std::lock_guard<std::mutex> lk(engine->mrMu);
    if (engine->mrByBase.size() > 65536)
      return -1;
    localTable.reserve(engine->mrByBase.size());
    for (auto &kv : engine->mrByBase) {
      const AcclMrEntry &e = kv.second;
      AcclMemRegWire w;
      memset(&w, 0, sizeof(w));
      w.baseAddr = e.baseAddr;
      w.size = e.size;
      w.mrId = static_cast<uint32_t>(e.mrId);
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
    if (remoteTable[i].size == 0 ||
        remoteTable[i].baseAddr > UINT64_MAX - remoteTable[i].size ||
        remoteTable[i].nKeys == 0 || remoteTable[i].nKeys > kMaxNics)
      return -1;
    if (!hasAnyRemoteKey(remoteTable[i].rkeys, remoteTable[i].nKeys))
      return -1;
    AcclRemoteRegion r;
    r.baseAddr = remoteTable[i].baseAddr;
    r.size = remoteTable[i].size;
    r.mrId = remoteTable[i].mrId;
    r.nKeys = remoteTable[i].nKeys;
    memcpy(r.rkeys, remoteTable[i].rkeys, sizeof(r.rkeys));
    conn->remoteRegions.push_back(r);
  }
  std::sort(conn->remoteRegions.begin(), conn->remoteRegions.end(),
            [](const AcclRemoteRegion &a, const AcclRemoteRegion &b) {
              return a.baseAddr < b.baseAddr;
            });
  for (size_t i = 1; i < conn->remoteRegions.size(); ++i) {
    const AcclRemoteRegion &previous = conn->remoteRegions[i - 1];
    if (previous.baseAddr + previous.size > conn->remoteRegions[i].baseAddr)
      return -1;
  }
  rebuildRemoteSpansLocked(conn);
  return remoteHello.barexPort;
}

bool exchangeAcclProtocol(struct bootstrapState *bsConn) {
  const flagcxP2pControl::ProtocolHello local = flagcxP2pControl::protocolHello(
      flagcxP2pControl::kProtocolLegacy, flagcxP2pControl::kProtocolBarex);
  flagcxP2pControl::ProtocolHello remote = {};
  if (bootstrapExchange(bsConn, 0, flagcxP2pControl::kProtocolTag, &local,
                        sizeof(local), &remote,
                        sizeof(remote)) != flagcxSuccess)
    return false;
  if (!flagcxP2pControl::protocolCompatible(local, remote)) {
    WARN("NET/ACCL_P2P : incompatible peer protocol impl=%u transport=%u "
         "version=%u",
         unsigned(remote.implementation), unsigned(remote.transport),
         unsigned(remote.version));
    return false;
  }
  return true;
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

int connectDataChannels(FlagcxAcclEngine *engine, FlagcxAcclConn *conn,
                        const char *peerHost, int peerBarexPort) {
  if (engine == nullptr || conn == nullptr || peerHost == nullptr ||
      peerHost[0] == '\0' || peerBarexPort <= 0 || engine->connector == nullptr)
    return -1;

  /* Every public connection is bidirectional. Create a local outbound set on
     both the bootstrap connect and accept paths instead of borrowing the
     passive channels owned by the peer's XListener. This keeps channel close
     ownership unambiguous and lets either returned connection issue RDMA. */
  const auto &config = flagcxP2pGlobalConfig();
  const int qps = acclQpsPerConn(config);
  auto ctl = std::make_shared<AcclConnectCtl>(qps);
  for (int i = 0; i < qps; i++) {
    BarexResult result =
        engine->connector->Connect(std::string(peerHost), peerBarexPort,
                                   [ctl](XChannel *channel, Status status) {
                                     std::lock_guard<std::mutex> lk(ctl->mu);
                                     if (status.IsOk() && channel != nullptr) {
                                       if (ctl->abandoned)
                                         channel->Destroy();
                                       else
                                         ctl->channels.push_back(channel);
                                     } else {
                                       ctl->anyFailed = true;
                                     }
                                     if (--ctl->remaining <= 0)
                                       ctl->cv.notify_all();
                                   });
    if (result != BAREX_SUCCESS) {
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
    ctl->abandoned = true;
    allUp = ctl->remaining <= 0 && !ctl->anyFailed;
    conn->channels = std::move(ctl->channels);
  }
  if (conn->channels.empty()) {
    WARN("NET/ACCL_P2P : connect to %s:%d produced no channels", peerHost,
         peerBarexPort);
    return -1;
  }

  const int expectedNic = engine->devs[engine->selectedNetDev]->GetId();
  for (XChannel *channel : conn->channels) {
    XContext *context = channel == nullptr ? nullptr : channel->GetContext();
    XDevice *device = context == nullptr ? nullptr : context->GetXDevice();
    const int actualNic = device == nullptr ? -1 : device->GetId();
    if (actualNic != expectedNic) {
      WARN("NET/ACCL_P2P : channel on nic %d, expected GPU-local nic %d",
           actualNic, expectedNic);
      return -1;
    }
  }
  if (!allUp)
    WARN("NET/ACCL_P2P : %zu/%d channels up (continuing)",
         conn->channels.size(), qps);

  INFO(FLAGCX_INIT, "NET/ACCL_P2P : connected %s:%d (%zu channels)", peerHost,
       peerBarexPort, conn->channels.size());
  return 0;
}

} // namespace

FlagcxP2pEngine *flagcxAcclEngineCreate() {
  if (flagcxParamIbDisable()) {
    INFO(FLAGCX_INIT, "NET/ACCL_P2P : disabled by FLAGCX_IB_DISABLE");
    return nullptr;
  }

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

  const auto &p2pConfig = flagcxP2pGlobalConfig();
  const uint32_t defaultSlice = acclSliceSize(p2pConfig);
  const uint32_t defaultFragment = acclFragmentLimit(p2pConfig, defaultSlice);
  engine->runtimeSliceConfig.store(
      flagcxP2pControl::pack(defaultSlice, defaultFragment),
      std::memory_order_relaxed);

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

  struct flagcxNetAdaptor *netAdaptor = getNetAdaptor(RDMA);
  if (netAdaptor == nullptr ||
      flagcxP2pTopoInit(netAdaptor, &engine->topoMgr) != flagcxSuccess ||
      flagcxP2pTopoGetNetDev(engine->topoMgr, engine->localGpuIdx,
                             &engine->selectedNetDev) != flagcxSuccess ||
      engine->selectedNetDev < 0 || engine->selectedNetDev >= engine->nDevs) {
    WARN("NET/ACCL_P2P : failed to select a NIC for GPU %d",
         engine->localGpuIdx);
    flagcxAcclEngineDestroy(EOut(engine));
    return nullptr;
  }
  INFO(FLAGCX_INIT, "NET/ACCL_P2P : selected GPU %d -> netDev %d (%s)",
       engine->localGpuIdx, engine->selectedNetDev,
       engine->devs[engine->selectedNetDev]->GetName().c_str());

  BarexResult r = XSimpleMempool::NewInstance(engine->mempool,
                                              "flagcx-p2p-accl", engine->devs);
  if (r != BAREX_SUCCESS) {
    WARN("NET/ACCL_P2P : mempool: %s", bxstr(r));
    flagcxAcclEngineDestroy(EOut(engine));
    return nullptr;
  }
  const auto &workerConfig = flagcxP2pGlobalConfig();
  const int workerCount = acclWorkerCount(workerConfig);
  r = XThreadpool::NewInstance(engine->tpServer, workerCount,
                               "flagcx-accl-server");
  if (r == BAREX_SUCCESS)
    r = XThreadpool::NewInstance(engine->tpClient, workerCount,
                                 "flagcx-accl-client");
  if (r != BAREX_SUCCESS) {
    WARN("NET/ACCL_P2P : threadpool create failed: %s", bxstr(r));
    flagcxAcclEngineDestroy(EOut(engine));
    return nullptr;
  }

  ContextConfig cfg = XConfigUtil::DefaultContextConfig();
  /* A P2P engine is scoped to one local GPU. Bind its data endpoint to the
     topology-selected NIC; exposing every RNIC here bypasses that decision
     and can create channels on non-local HCAs. */
  XDevice *selectedDev = engine->devs[engine->selectedNetDev];
  XContext *sctx = nullptr;
  XContext *cctx = nullptr;
  BarexResult contextResult =
      XContext::NewInstance(sctx, cfg, new AcclNullCb(), selectedDev,
                            engine->mempool, engine->tpServer);
  if (contextResult == BAREX_SUCCESS)
    contextResult =
        XContext::NewInstance(cctx, cfg, new AcclNullCb(), selectedDev,
                              engine->mempool, engine->tpClient);
  if (contextResult != BAREX_SUCCESS) {
    WARN("NET/ACCL_P2P : XContext create failed on %s",
         selectedDev->GetName().c_str());
    if (sctx != nullptr) {
      sctx->Shutdown();
      sctx->WaitStop();
      delete sctx;
    }
    if (cctx != nullptr) {
      cctx->Shutdown();
      cctx->WaitStop();
      delete cctx;
    }
    flagcxAcclEngineDestroy(EOut(engine));
    return nullptr;
  }
  sctx->Start();
  cctx->Start();
  engine->serverCtxs.push_back(sctx);
  engine->clientCtxs.push_back(cctx);

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
  flagcxResult_t notifResult =
      flagcxSocketInit(&engine->notifListenSock, &notifAddr,
                       FLAGCX_SOCKET_MAGIC, flagcxSocketTypeProxy, NULL, 1);
  if (notifResult != flagcxSuccess) {
    WARN("NET/ACCL_P2P : dynamic-MR notification socket init failed");
    flagcxAcclEngineDestroy(EOut(engine));
    return nullptr;
  }
  notifResult = flagcxSocketListen(&engine->notifListenSock);
  if (notifResult != flagcxSuccess) {
    WARN("NET/ACCL_P2P : dynamic-MR notification listen failed");
    flagcxSocketClose(&engine->notifListenSock);
    flagcxAcclEngineDestroy(EOut(engine));
    return nullptr;
  }
  union flagcxSocketAddress bound;
  flagcxSocketGetAddr(&engine->notifListenSock, &bound);
  engine->notifPort = addrPort(&bound);
  engine->notifActive = true;
  engine->notifThread = std::thread(notifThreadFunc, engine);

  INFO(FLAGCX_INIT,
       "NET/ACCL_P2P : engine up (gpu=%d registered_nics=%d data_nic=%d "
       "barex=%d bootstrap=%d notif=%d workers=%d qps=%d slice=%u "
       "fragment=%u)",
       engine->localGpuIdx, engine->nDevs, selectedDev->GetId(),
       engine->barexPort, engine->bsListenPort, engine->notifPort, workerCount,
       acclQpsPerConn(p2pConfig), defaultSlice, defaultFragment);
  return EOut(engine);
}

void flagcxAcclEngineStopAccept(FlagcxP2pEngine *e) {
  FlagcxAcclEngine *engine = E(e);
  if (engine == nullptr)
    return;
  engine->stopAccept.store(true, std::memory_order_release);
  engine->stopRpc.store(true, std::memory_order_release);
  __atomic_store_n(&engine->acceptAbortFlag, 1, __ATOMIC_RELEASE);
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

  /* Stop channel owners before their contexts.  This drains CloseChannel and
     completion callbacks while the XContext/XDevice objects are still alive. */
  if (engine->connector != nullptr) {
    engine->connector->Shutdown();
    engine->connector->WaitStop();
    delete engine->connector;
    engine->connector = nullptr;
  }
  if (engine->listener != nullptr) {
    engine->listener->Shutdown();
    engine->listener->WaitStop();
    delete engine->listener;
    engine->listener = nullptr;
  }

  /* Listener shutdown does not close channels already accepted by its server
     contexts.  Stop both context sets as well, which closes those channels
     and drains their completion callbacks.  Keep the context objects alive
     until DeregUserMr has finished. */
  for (auto *ctx : engine->serverCtxs) {
    ctx->Shutdown();
    ctx->WaitStop();
  }
  for (auto *ctx : engine->clientCtxs) {
    ctx->Shutdown();
    ctx->WaitStop();
  }

  /* A synchronous CloseChannel failure retains the channel here because the
     provider may still be cleaning it.  Connector/context shutdown above has
     now drained callbacks and provider work, so the remaining local objects
     can be released without racing asynchronous cleanup. */
  {
    std::lock_guard<std::mutex> lock(engine->deferredChannelMu);
    for (const auto &owner : engine->deferredChannels) {
      std::lock_guard<std::mutex> ownerLock(owner->mu);
      if (owner->channel != nullptr) {
        owner->channel->Destroy();
        owner->channel = nullptr;
      }
    }
    engine->deferredChannels.clear();
  }

  if (engine->mempool != nullptr) {
    std::unique_lock<std::mutex> lk(engine->mrMu);
    for (const auto &kv : engine->mrByBase)
      engine->pendingMrDeregs[kv.second.mrId].push_back(kv.second);
    engine->mrByBase.clear();
    uint64_t retry = 0;
    while (!engine->pendingMrDeregs.empty() ||
           !engine->pendingOrphanDeregs.empty()) {
      for (auto it = engine->pendingMrDeregs.begin();
           it != engine->pendingMrDeregs.end();) {
        auto &chunks = it->second;
        for (auto chunk = chunks.begin(); chunk != chunks.end();) {
          if (engine->mempool->DeregUserMr(
                  reinterpret_cast<void *>(chunk->baseAddr), chunk->dtype) ==
              BAREX_SUCCESS)
            chunk = chunks.erase(chunk);
          else
            ++chunk;
        }
        if (chunks.empty())
          it = engine->pendingMrDeregs.erase(it);
        else
          ++it;
      }
      for (auto chunk = engine->pendingOrphanDeregs.begin();
           chunk != engine->pendingOrphanDeregs.end();) {
        if (engine->mempool->DeregUserMr(
                reinterpret_cast<void *>(chunk->baseAddr), chunk->dtype) ==
            BAREX_SUCCESS)
          chunk = engine->pendingOrphanDeregs.erase(chunk);
        else
          ++chunk;
      }
      if (engine->pendingMrDeregs.empty() &&
          engine->pendingOrphanDeregs.empty())
        break;

      /* Destroy has no error or retry return path. Keep provider ownership and
         retry internally instead of returning a leaked, unreachable Engine.
         Release the registry mutex while backing off so diagnostics and late
         status readers cannot deadlock behind a provider retry. */
      if (retry++ == 0 || retry % 1000 == 0)
        WARN("NET/ACCL_P2P : MR deregistration incomplete; retrying teardown "
             "(attempt %llu)",
             (unsigned long long)retry);
      lk.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      lk.lock();
    }
  }
  {
    std::lock_guard<std::mutex> lk(engine->xferMu);
    engine->xfers.clear();
  }
  for (auto *ctx : engine->serverCtxs)
    delete ctx;
  for (auto *ctx : engine->clientCtxs)
    delete ctx;
  engine->serverCtxs.clear();
  engine->clientCtxs.clear();
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
  flagcxP2pTopoDestroy(engine->topoMgr);
  engine->topoMgr = nullptr;
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
  const std::string ipPort = hostPortString(ipAddr, remotePort);
  if (ipPort.empty() || flagcxSocketGetAddrFromString(
                            &bsHandle.addr, ipPort.c_str()) != flagcxSuccess)
    return nullptr;

  struct bootstrapState *bsConn = nullptr;
  if (bootstrapP2pConnect(&bsHandle, FLAGCX_SOCKET_MAGIC, NULL, &bsConn) !=
      flagcxSuccess)
    return nullptr;

  if (!exchangeAcclProtocol(bsConn)) {
    bootstrapClose(bsConn);
    return nullptr;
  }

  auto *conn = new FlagcxAcclConn();
  conn->engine = engine;
  conn->peerAddr = bsHandle.addr;
  memset(&conn->notifSock, 0, sizeof(conn->notifSock));

  const int peerBarexPort = acclHandshake(engine, bsConn, conn, sameProcess);
  bootstrapClose(bsConn);
  if (peerBarexPort <= 0) {
    delete conn;
    return nullptr;
  }
  (void)remoteGpuIdx;
  if (connectDataChannels(engine, conn, ipAddr, peerBarexPort) != 0) {
    flagcxAcclEngineConnDestroy(COut(conn));
    return nullptr;
  }

  connectNotif(conn);
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

  /* Share the same bootstrap RPC control frame as IBRC. Peek the first
     frame so a normal ACCL hello (tag 4) remains untouched for
     acclHandshake(). */
  int controlHeader[2] = {};
  const int controlFd = bsConn->p2p->sock.fd;
  if (peekBootstrapFrame(controlFd, controlHeader, sizeof(controlHeader),
                         engine->stopAccept) &&
      controlHeader[0] == flagcxP2pControl::kTag) {
    if (controlHeader[1] <= 0 ||
        controlHeader[1] > flagcxP2pControl::kMaxRequest ||
        !flagcxP2pControl::receive(controlFd, controlHeader,
                                   sizeof(controlHeader), engine->stopAccept)) {
      bootstrapClose(bsConn);
      return nullptr;
    }
    std::string request(controlHeader[1], '\0');
    if (flagcxP2pControl::receive(controlFd, &request[0], request.size(),
                                  engine->stopAccept)) {
      const char *error =
          flagcxP2pControl::update(engine->runtimeSliceConfig, request);
      const uint64_t config =
          engine->runtimeSliceConfig.load(std::memory_order_acquire);
      char reply[256];
      const int length =
          snprintf(reply, sizeof(reply),
                   "{\"status\":%d,\"slice_size\":%u,\"fragment_limit\":%u,"
                   "\"error\":\"%s\"}",
                   error == nullptr ? 0 : -1, unsigned(config >> 32),
                   unsigned(uint32_t(config)), error == nullptr ? "" : error);
      bootstrapSend(bsConn, 0, flagcxP2pControl::kTag, reply, length);
    }
    bootstrapClose(bsConn);
    return nullptr;
  }

  if (!exchangeAcclProtocol(bsConn)) {
    WARN("NET/ACCL_P2P : peer protocol is incompatible with legacy BAREX");
    bootstrapClose(bsConn);
    return nullptr;
  }

  auto *conn = new FlagcxAcclConn();
  conn->engine = engine;
  memset(&conn->notifSock, 0, sizeof(conn->notifSock));
  flagcxSocketGetAddr(&bsConn->p2p->sock, &conn->peerAddr);

  const int peerBarexPort = acclHandshake(engine, bsConn, conn, false);
  bootstrapClose(bsConn);
  if (peerBarexPort <= 0) {
    delete conn;
    return nullptr;
  }

  const std::string host = addrHostString(&conn->peerAddr);
  snprintf(ipAddrBuf, ipAddrBufLen, "%s", host.c_str());
  *remoteGpuIdx = conn->remoteGpuIdx;

  if (connectDataChannels(engine, conn, host.c_str(), peerBarexPort) != 0) {
    flagcxAcclEngineConnDestroy(COut(conn));
    return nullptr;
  }
  connectNotif(conn);
  return COut(conn);
}

void flagcxAcclEngineConnDestroy(FlagcxP2pConn *c) {
  FlagcxAcclConn *conn = C(c);
  if (conn == nullptr)
    return;
  FlagcxAcclEngine *engine = conn->engine;
  std::vector<std::pair<uint64_t, std::shared_ptr<AcclXfer>>> pending;
  {
    std::lock_guard<std::mutex> connLock(conn->xferMu);
    conn->state->lifecycle.store(ACCL_CONN_CLOSING, std::memory_order_release);
    if (engine != nullptr) {
      std::lock_guard<std::mutex> engineLock(engine->xferMu);
      pending.reserve(conn->xferIds.size());
      for (uint64_t id : conn->xferIds) {
        auto it = engine->xfers.find(id);
        if (it != engine->xfers.end())
          pending.emplace_back(id, it->second);
      }
    }
  }
  for (const auto &entry : pending)
    entry.second->wait();
  {
    std::lock_guard<std::mutex> connLock(conn->xferMu);
    if (engine != nullptr) {
      std::lock_guard<std::mutex> engineLock(engine->xferMu);
      for (uint64_t id : conn->xferIds)
        engine->xfers.erase(id);
    }
    conn->xferIds.clear();
  }

  if (engine != nullptr && engine->connector != nullptr) {
    auto closeTracker = std::make_shared<AcclXfer>();
    closeTracker->pending.store(static_cast<int>(conn->channels.size()),
                                std::memory_order_release);
    for (auto *ch : conn->channels) {
      if (ch == nullptr) {
        closeTracker->complete();
        continue;
      }
      auto owner = std::make_shared<AcclChannelOwner>(ch);
      auto completed = std::make_shared<std::atomic<bool>>(false);
      BarexResult result = engine->connector->CloseChannel(
          ch, [owner, closeTracker, completed](Status status) {
            if (!status.IsOk())
              WARN("NET/ACCL_P2P : CloseChannel failed: %s",
                   status.ErrMsg().c_str());
            XChannel *channel = nullptr;
            {
              std::lock_guard<std::mutex> lock(owner->mu);
              if (owner->fallbackActive) {
                owner->callbackArrived = true;
              } else {
                channel = owner->channel;
                owner->channel = nullptr;
              }
            }
            if (channel != nullptr)
              channel->Destroy();
            if (!completed->exchange(true, std::memory_order_acq_rel))
              closeTracker->complete();
          });
      if (result != BAREX_SUCCESS) {
        WARN("NET/ACCL_P2P : CloseChannel sync error: %s", bxstr(result));
        if (!closeAndDeleteAcclChannel(engine->connector, owner)) {
          std::lock_guard<std::mutex> lock(engine->deferredChannelMu);
          engine->deferredChannels.push_back(owner);
        }
        if (!completed->exchange(true, std::memory_order_acq_rel))
          closeTracker->complete();
      }
    }
    closeTracker->wait();
  }
  conn->channels.clear();
  if (conn->notifConnected) {
    flagcxSocketClose(&conn->notifSock);
    conn->notifConnected = false;
  }
  conn->state->lifecycle.store(ACCL_CONN_CLOSED, std::memory_order_release);
  delete conn;
}

bool flagcxAcclEngineConnIsLocal(FlagcxP2pConn *c) {
  FlagcxAcclConn *conn = C(c);
  return conn != nullptr && conn->isLocal;
}

int flagcxAcclEngineReg(FlagcxP2pEngine *e, uintptr_t data, size_t size,
                        int hintType, FlagcxP2pMr &mrId) {
  FlagcxAcclEngine *engine = E(e);
  if (engine == nullptr || data == 0 || size == 0 || data > UINTPTR_MAX - size)
    return -1;

  device_type dtype;
  int devId;
  if (hintType == FLAGCX_PTR_HOST) {
    dtype = CPU;
    devId = 0;
  } else if (hintType == FLAGCX_PTR_CUDA) {
    dtype = GPU;
    devId = inferLocalGpuIdxAccl();
  } else if (hintType == 0) {
    classifyPtr(reinterpret_cast<void *>(data), &dtype, &devId);
  } else {
    return -1;
  }

  /* RegUserMr/DeregUserMr identify registrations by (base, dtype). Serialize
     duplicate detection, physical registration, rollback, and publication so
     a losing duplicate cannot deregister the winner's live registration. */
  std::lock_guard<std::mutex> lk(engine->mrMu);
  auto existing = engine->mrByBase.find(data);
  if (existing != engine->mrByBase.end()) {
    if (existing->second.regBase != data || existing->second.regSize != size ||
        existing->second.dtype != dtype || existing->second.deviceId != devId) {
      WARN("NET/ACCL_P2P : re-register 0x%lx with incompatible attributes",
           (unsigned long)data);
      return -1;
    }
    mrId = existing->second.mrId;
    return 0;
  }
  if (!acclRetryConflictingDeregs(engine, data, size, dtype))
    return -1;
  if (engine->nextMrId == 0) {
    WARN("NET/ACCL_P2P : MR id space exhausted");
    return -1;
  }
  const uint32_t reservedMrId = engine->nextMrId++;

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
           "(chunk %zu/%zu of %p+%zu)",
           reinterpret_cast<void *>(cbase), csize, dtype == GPU ? "GPU" : "CPU",
           devId, bxstr(r), off / chunkBytes + 1,
           (size + chunkBytes - 1) / chunkBytes, reinterpret_cast<void *>(data),
           size);
      std::vector<AcclMrEntry> failed;
      for (const AcclMrEntry &c : chunks) {
        if (engine->mempool->DeregUserMr(reinterpret_cast<void *>(c.baseAddr),
                                         dtype) != BAREX_SUCCESS)
          failed.push_back(c);
      }
      if (!failed.empty()) {
        engine->pendingOrphanDeregs.insert(engine->pendingOrphanDeregs.end(),
                                           failed.begin(), failed.end());
      }
      return -1;
    }

    AcclMrEntry entry{};
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
      std::vector<AcclMrEntry> failed;
      AcclMrEntry current{};
      current.baseAddr = cbase;
      current.size = csize;
      current.regBase = data;
      current.regSize = size;
      current.dtype = dtype;
      current.deviceId = devId;
      if (engine->mempool->DeregUserMr(reinterpret_cast<void *>(cbase),
                                       dtype) != BAREX_SUCCESS)
        failed.push_back(current);
      for (const AcclMrEntry &c : chunks) {
        if (engine->mempool->DeregUserMr(reinterpret_cast<void *>(c.baseAddr),
                                         dtype) != BAREX_SUCCESS)
          failed.push_back(c);
      }
      if (!failed.empty()) {
        engine->pendingOrphanDeregs.insert(engine->pendingOrphanDeregs.end(),
                                           failed.begin(), failed.end());
      }
      return -1;
    }
    chunks.push_back(entry);
  }

  const uint64_t id = reservedMrId;
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
  auto pending = engine->pendingMrDeregs.find(mr);
  if (pending == engine->pendingMrDeregs.end()) {
    std::vector<AcclMrEntry> chunks;
    for (auto it = engine->mrByBase.begin(); it != engine->mrByBase.end();) {
      if (it->second.mrId == mr) {
        chunks.push_back(it->second);
        it = engine->mrByBase.erase(it);
      } else {
        ++it;
      }
    }
    if (chunks.empty())
      return;
    pending = engine->pendingMrDeregs.emplace(mr, std::move(chunks)).first;
  }

  auto &chunks = pending->second;
  for (auto it = chunks.begin(); it != chunks.end();) {
    const BarexResult result = engine->mempool->DeregUserMr(
        reinterpret_cast<void *>(it->baseAddr), it->dtype);
    if (result == BAREX_SUCCESS) {
      it = chunks.erase(it);
    } else {
      WARN("NET/ACCL_P2P : DeregUserMr(%p) failed: %s; retaining ownership "
           "for retry",
           reinterpret_cast<void *>(it->baseAddr), bxstr(result));
      ++it;
    }
  }
  if (chunks.empty())
    engine->pendingMrDeregs.erase(pending);
}

int flagcxAcclEnginePrepareDesc(FlagcxP2pEngine *e, FlagcxP2pMr mr,
                                const void *data, size_t size, char *descBuf) {
  FlagcxAcclEngine *engine = E(e);
  if (engine == nullptr || data == nullptr || descBuf == nullptr)
    return -1;
  if (size > UINT32_MAX)
    return -1;
  std::lock_guard<std::mutex> lk(engine->mrMu);
  const uintptr_t addr = (uintptr_t)data;
  if (addr > UINTPTR_MAX - size)
    return -1;
  for (auto &kv : engine->mrByBase) {
    const AcclMrEntry &entry = kv.second;
    if (entry.mrId != mr)
      continue;
    if (addr < entry.regBase)
      continue;
    const size_t registrationOffset = addr - entry.regBase;
    if (registrationOffset > entry.regSize ||
        size > entry.regSize - registrationOffset)
      continue;
    if (addr < entry.baseAddr || addr >= entry.baseAddr + entry.size)
      continue;
    const size_t chunkOffset = addr - entry.baseAddr;
    FlagcxP2pRdmaDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.addr = (uint64_t)addr;
    desc.size = (uint32_t)size;
    fillDescKeys(&desc, entry.rkeys, entry.nKeys);
    if (size <= entry.size - chunkOffset) {
      /* A single-chunk descriptor owns its current keys, so re-registering a
         VA cannot accidentally select stale handshake metadata. */
      desc.rid = kEmbeddedRkeysRid;
    } else {
      /* A logical multi-chunk descriptor is resolved through the per-chunk
         handshake table. The MR id prevents reuse after re-registration. */
      desc.rid = kCachedRkeysRid;
      desc.idx = entry.mrId;
    }
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
  std::lock_guard<std::mutex> lk(conn->remoteMrMu);
  /* the range may cross chunk boundaries — validate against merged spans;
     acclSubmit re-resolves per-chunk rkeys, the desc carries the first
     chunk's keys for legacy/single-chunk consumers. */
  for (const auto &span : conn->remoteSpans) {
    if (remoteVa >= span.baseAddr) {
      const uint64_t offset = remoteVa - span.baseAddr;
      if (offset > span.size || size > span.size - offset)
        continue;
      const AcclRemoteRegion *r = findRemoteRegion(conn, remoteVa);
      if (r == nullptr || r->mrId != span.mrId)
        return -1;
      memset(desc, 0, sizeof(*desc));
      desc->addr = remoteVa;
      desc->size = size;
      fillDescKeys(desc, r->rkeys, r->nKeys);
      if (!conn->peerSupportsDynamicMr || span.mrId == 0) {
        /* Pre-dynamic-MR peers populate the former reserved field with zero.
           Their handshake keys are stable for the connection lifetime and
           must be carried directly in the descriptor. Embedded keys describe
           one physical MR, so a request crossing a chunk boundary cannot be
           represented by the legacy format. */
        const uint64_t regionOffset = remoteVa - r->baseAddr;
        if (regionOffset > r->size || size > r->size - regionOffset)
          return -1;
        desc->rid = kEmbeddedRkeysRid;
      } else {
        desc->rid = kCachedRkeysRid;
        desc->idx = span.mrId;
      }
      return 0;
    }
  }
  return -1;
}

int flagcxAcclEngineResolveLocalMr(FlagcxP2pConn *c, uintptr_t address,
                                   size_t size, FlagcxP2pMr *mr) {
  FlagcxAcclConn *conn = C(c);
  if (conn == nullptr || mr == nullptr || address > UINTPTR_MAX - size)
    return -1;
  std::lock_guard<std::mutex> lk(conn->engine->mrMu);
  auto it = conn->engine->mrByBase.upper_bound(address);
  if (it == conn->engine->mrByBase.begin())
    return -1;
  --it;
  const AcclMrEntry &entry = it->second;
  if (address < entry.regBase)
    return -1;
  const size_t offset = address - entry.regBase;
  if (offset > entry.regSize || size > entry.regSize - offset)
    return -1;
  *mr = entry.mrId;
  return 0;
}

int flagcxAcclEngineRead(FlagcxP2pConn *c, FlagcxP2pMr mr, const void *data,
                         size_t size, FlagcxP2pRdmaDesc desc,
                         uint64_t *transferId) {
  FlagcxAcclConn *conn = C(c);
  if (conn == nullptr || data == nullptr || transferId == nullptr)
    return -1;
  *transferId = 0;
  if (!registeredRangeMatchesMr(conn->engine, mr, (uintptr_t)data, size))
    return -1;
  std::vector<void *> localVec(1, const_cast<void *>(data));
  std::vector<size_t> sizeVec(1, size);
  std::vector<FlagcxP2pRdmaDesc> descs(1, desc);
  return acclSubmit(conn, localVec, sizeVec, descs, 1, true, transferId);
}

int flagcxAcclEngineWrite(FlagcxP2pConn *c, FlagcxP2pMr mr, const void *data,
                          size_t size, FlagcxP2pRdmaDesc desc,
                          uint64_t *transferId) {
  FlagcxAcclConn *conn = C(c);
  if (conn == nullptr || data == nullptr || transferId == nullptr)
    return -1;
  *transferId = 0;
  if (!registeredRangeMatchesMr(conn->engine, mr, (uintptr_t)data, size))
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
  FlagcxAcclConn *conn = C(c);
  if (conn == nullptr || numIovs <= 0 || transferId == nullptr)
    return -1;
  *transferId = 0;
  if (dstVec.size() < (size_t)numIovs || sizeVec.size() < (size_t)numIovs ||
      descs.size() < (size_t)numIovs || mrIds.size() < (size_t)numIovs)
    return -1;
  for (int i = 0; i < numIovs; ++i) {
    if (!registeredRangeMatchesMr(conn->engine, mrIds[i], (uintptr_t)dstVec[i],
                                  sizeVec[i]))
      return -1;
  }
  return acclSubmit(conn, dstVec, sizeVec, descs, numIovs, true, transferId);
}

int flagcxAcclEngineWriteVector(FlagcxP2pConn *c,
                                const std::vector<FlagcxP2pMr> &mrIds,
                                const std::vector<void *> &srcVec,
                                const std::vector<size_t> &sizeVec,
                                const std::vector<FlagcxP2pRdmaDesc> &descs,
                                int numIovs, uint64_t *transferId) {
  FlagcxAcclConn *conn = C(c);
  if (conn == nullptr || numIovs <= 0 || transferId == nullptr)
    return -1;
  *transferId = 0;
  if (srcVec.size() < (size_t)numIovs || sizeVec.size() < (size_t)numIovs ||
      descs.size() < (size_t)numIovs || mrIds.size() < (size_t)numIovs)
    return -1;
  for (int i = 0; i < numIovs; ++i) {
    if (!registeredRangeMatchesMr(conn->engine, mrIds[i], (uintptr_t)srcVec[i],
                                  sizeVec[i]))
      return -1;
  }
  return acclSubmit(conn, srcVec, sizeVec, descs, numIovs, false, transferId);
}

int acclXferPoll(FlagcxAcclEngine *engine, FlagcxAcclConn *conn,
                 uint64_t transferId) {
  if (conn == nullptr || transferId == 0)
    return 1;
  std::shared_ptr<AcclXfer> xfer;
  {
    std::lock_guard<std::mutex> connLock(conn->xferMu);
    std::lock_guard<std::mutex> engineLock(engine->xferMu);
    auto it = engine->xfers.find(transferId);
    if (it == engine->xfers.end()) {
      conn->xferIds.erase(transferId);
      return 1;
    }
    xfer = it->second;
    if (xfer->pending.load(std::memory_order_acquire) > 0)
      return 0;
    engine->xfers.erase(it);
    conn->xferIds.erase(transferId);
  }
  const bool failed = xfer->failed.load(std::memory_order_acquire) > 0;
  if (failed) {
    WARN("NET/ACCL_P2P : transfer %llu completed with failures",
         (unsigned long long)transferId);
    if (xfer->hardFailed.load(std::memory_order_acquire) > 0)
      conn->state->fail(-1);
  }
  return failed ? -1 : 1;
}

bool flagcxAcclEngineXferStatus(FlagcxP2pConn *c, uint64_t transferId) {
  FlagcxAcclConn *conn = C(c);
  if (conn == nullptr)
    return true;
  return acclXferPoll(conn->engine, conn, transferId) != 0;
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
  FlagcxAcclConn *conn = C(c);
  if (conn == nullptr)
    return -1;
  int status;
  while ((status = acclXferPoll(conn->engine, conn, transferId)) == 0)
    std::this_thread::yield();
  return status < 0 ? -1 : 0;
}

int flagcxAcclEngineGetMetadata(FlagcxP2pEngine *e, char **metadataStr) {
  FlagcxAcclEngine *engine = E(e);
  if (engine == nullptr || metadataStr == nullptr)
    return -1;
  if (engine->bsListenState == nullptr || engine->bsListenPort <= 0)
    return -1;
  union flagcxSocketAddress bsAddr;
  flagcxSocketGetAddr(&engine->bsListenState->p2p->sock, &bsAddr);
  const std::string endpoint =
      addrHostPortString(&bsAddr, engine->bsListenPort);
  if (endpoint.empty())
    return -1;
  const std::string result = endpoint + "?" +
                             std::to_string(engine->localGpuIdx) + "?" +
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
  std::lock_guard<std::mutex> lk(conn->notifMu);
  if (!conn->notifConnected && connectNotif(conn) != 0)
    return -1;
  AcclNotifWireMsg wire;
  memset(&wire, 0, sizeof(wire));
  wire.magic = kAcclNotifMagic;
  wire.payload = *notifyMsg;
  if (sendAllFdAccl(conn->notifSock.fd, &wire, sizeof(wire)) != 0) {
    flagcxSocketClose(&conn->notifSock);
    conn->notifConnected = false;
    return -1;
  }
  return (int)sizeof(FlagcxP2pNotifyMsg);
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
