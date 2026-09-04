/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX net adaptor "barex": collective/C2C transport over the vendor
 * ACCL library (accl::barex) for PPU + vsolar hosts, where GPU memory
 * cannot be registered via peer-mem or DMA-BUF and must go through
 * ACCL's RegUserMr / XChannel. Requires FLAGCX_VMM_ENABLE=0 (VMM memory
 * is unpinnable) so staging buffers come from cudaMalloc.
 *
 * Rendezvous (mirrors ibrc's CTS design): connect sends HELLO{commId}
 * over an XChannel; irecv posts CTS{slot,addr,size,rkeys,seq}; isend
 * answers WriteSingle(imm=slot) and OnImmRecvCall completes the recv
 * (write-with-imm orders payload before the imm). Shared state is mutex-
 * or atomic-guarded (callbacks run on ACCL IO threads).
 *
 * Built with USE_ACCL_BAREX=1 (RDMA registry slot, like USE_UCX) or
 * loaded as a plugin .so (preferred; see the export note below).
 * FLAGCX_BAREX_DISABLE=1 opts out at runtime.
 ************************************************************************/

#ifdef USE_ACCL_BAREX

#include "adaptor.h"
#include "bootstrap.h"
#include "debug.h"
#include "flagcx_net.h"
#include "flagcx_net_adaptor.h"
#include "net.h"
#include "onesided.h"
#include "p2p_transport.h"
#include "param.h"
#include "socket.h"

/* FlagCX's topo.h (pulled in via net.h/comm.h) defines node-type macros
   that collide with accl::barex's device_type enumerators. */
#ifdef CPU
#undef CPU
#endif
#ifdef GPU
#undef GPU
#endif
#ifdef NIC
#undef NIC
#endif
#ifdef NET
#undef NET
#endif
#ifdef PCI
#undef PCI
#endif

#include <accl/barex/barex_types.h>
#include <accl/barex/xchannel.h>
#include <accl/barex/xconfig_util.h>
#include <accl/barex/xconnector.h>
#include <accl/barex/xcontext.h>
#include <accl/barex/xdevice_manager.h>
#include <accl/barex/xlistener.h>
#include <accl/barex/xsimple_mempool.h>
#include <accl/barex/xthreadpool.h>

#include <arpa/inet.h>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

using accl::barex::BarexResult;
using accl::barex::BarexResultStrings;
using accl::barex::ContextConfig;
using accl::barex::DoneCallback;
using accl::barex::memp_t;
using accl::barex::rw_memp_t;
using accl::barex::Status;
using accl::barex::TimerTick;
using accl::barex::x_msg_header;
using accl::barex::XChannel;
using accl::barex::XChannelCallback;
using accl::barex::XConfigUtil;
using accl::barex::XConnector;
using accl::barex::XContext;
using accl::barex::XDevice;
using accl::barex::XDeviceManager;
using accl::barex::XListener;
using accl::barex::XSimpleMempool;
using accl::barex::XThreadpool;

FLAGCX_PARAM(BarexDisable, "BAREX_DISABLE", 0);
FLAGCX_PARAM(BarexSpeed, "BAREX_SPEED", 100000); /* Mbps, for topo costing */

constexpr int kMaxNics = 8;       /* matches ACCL per-NIC rkey fan-out */
constexpr int kMaxRequests = 256; /* proxy keeps <=16 in flight; roomy */
constexpr uint32_t kImmSlotMask = 0x00FFFFFFu; /* imm_data carries 24 bits */
constexpr uintptr_t kCompletedRequest = 1;

static const char *bxstr(BarexResult r) {
  auto it = BarexResultStrings.find(r);
  return it == BarexResultStrings.end() ? "UNKNOWN" : it->second;
}

static flagcxResult_t barexResult(BarexResult r) {
  if (r == accl::barex::BAREX_SUCCESS)
    return flagcxSuccess;
  if (r == accl::barex::BAREX_ERR_ARG || r == accl::barex::BAREX_ERR_NPE)
    return flagcxInvalidArgument;
  if (r == accl::barex::BAREX_ERR_QUEUE_FULL ||
      r == accl::barex::BAREX_ERR_RATE_LIMITED)
    return flagcxInProgress;
  return flagcxInternalError;
}

static void addrSetPort(union flagcxSocketAddress *addr, int port) {
  if (addr->sa.sa_family == AF_INET)
    addr->sin.sin_port = htons(port);
  else if (addr->sa.sa_family == AF_INET6)
    addr->sin6.sin6_port = htons(port);
}

enum BarexMsgType : uint32_t {
  BAREX_MSG_HELLO = 0xB0E10001u,
  BAREX_MSG_CTS = 0xB0E10002u,
};

struct BarexHelloMsg {
  uint32_t type;
  uint32_t pad;
  uint64_t commId;
};

struct BarexCtsMsg {
  uint32_t type;
  uint32_t slot;
  uint64_t addr;
  uint64_t size;
  uint32_t nKeys;
  uint32_t rkeys[kMaxNics];
  uint32_t seq; /* receiver's post-order index; sender consumes CTS in
                   this order (callbacks may reorder). Former padding. */
};
static_assert(sizeof(BarexCtsMsg) == 64, "CTS wire layout must be stable");

/* Listen handle — must fit the 64-byte flagcxIbHandle buffer the collective
 * path allocates (transport.cc uses sizeof(flagcxIbHandle)) */

struct BarexNetHandle {
  union flagcxSocketAddress connectAddr; /* OOB IP + barex data port */
  uint64_t commId;                       /* demux key for accept side */
  uint32_t state;                        /* connect-side stage */
  uint32_t pad;
  void *connectState; /* connect-side heap state across retries */
  /* Keep <= 56 bytes: transport.cc writes stage.comm at offset 56 after
     bootstrapRecv, landing in this buffer's tail padding. */
};
static_assert(sizeof(BarexNetHandle) <= 56,
              "must not overlap flagcxIbHandle::stage.comm at offset 56");
static_assert(sizeof(BarexNetHandle) <= sizeof(struct flagcxIbHandle),
              "barex listen handle must fit the collective-path buffer");

enum BarexConnectStage : uint32_t {
  BAREX_CONN_INIT = 0,
  BAREX_CONN_CONNECTING = 1,
  BAREX_CONN_HELLO = 2,
};

enum BarexReqState : int {
  BAREX_REQ_FREE = 0,
  BAREX_REQ_PENDING = 1,
  BAREX_REQ_DONE = 2,
  BAREX_REQ_ERROR = 3,
};

struct BarexRequest {
  std::atomic<int> state{BAREX_REQ_FREE};
  size_t size = 0;
  struct BarexComm *comm = nullptr;
  uint32_t slot = 0;
};

/* MR registry is engine-wide and refcounted (regIsGlobal semantics: the
   user-buffer path may register one buffer through several proxy
   connections). */

struct BarexMr {
  memp_t mem; /* as returned by RegUserMr: per-NIC ibv_mr map */
  uintptr_t base = 0;
  size_t size = 0;
  accl::barex::device_type dtype = accl::barex::CPU;
  int devId = 0;
  uint32_t nKeys = 0;
  uint32_t lkeys[kMaxNics] = {0};
  uint32_t rkeys[kMaxNics] = {0};
  int refCount = 0;
};

struct BarexMrKey {
  uintptr_t base;
  accl::barex::device_type dtype;

  bool operator<(const BarexMrKey &other) const {
    if (base != other.base)
      return base < other.base;
    return dtype < other.dtype;
  }
};

struct BarexComm {
  struct BarexEngine *engine = nullptr;
  XChannel *channel = nullptr;
  uint64_t commId = 0;
  bool isSend = false;
  std::atomic<bool> dead{false};

  std::mutex mu; /* guards ctsPending + slot alloc + seq counters */
  /* Sender: CTS keyed by the receiver's post-order seq, consumed strictly
     in order (sendExpectedSeq) so chunk k lands in the buffer posted for
     it regardless of callback delivery order. Receiver: recvSeq stamps. */
  std::map<uint64_t, BarexCtsMsg> ctsPending;
  uint64_t recvSeq = 0;         /* receiver: next CTS seq to stamp */
  uint64_t sendExpectedSeq = 0; /* sender: next CTS seq to consume */
  BarexRequest requests[kMaxRequests];

  BarexRequest *allocRequest() {
    for (int i = 0; i < kMaxRequests; i++) {
      int expected = BAREX_REQ_FREE;
      if (requests[i].state.compare_exchange_strong(expected,
                                                    BAREX_REQ_PENDING)) {
        requests[i].comm = this;
        requests[i].slot = (uint32_t)i;
        requests[i].size = 0;
        return &requests[i];
      }
    }
    return nullptr;
  }
};

struct BarexListenComm {
  struct BarexEngine *engine = nullptr;
  uint64_t commId = 0;
  int dev = 0;
};

struct BarexConnectState {
  std::atomic<XChannel *> channel{nullptr};
  std::atomic<bool> connectFailed{false};
  std::atomic<bool> helloDone{false};
  std::atomic<bool> helloFailed{false};
  BarexComm *comm = nullptr;
};

struct PendingAccept {
  /* A FlagCX listen handle is reusable. Multiple connectors can therefore
     complete their HELLO before the engine drains accept(), and must retain
     FIFO association with that handle. */
  std::deque<XChannel *> channels;
};

class BarexNetCallback; /* fwd */

struct BarexEngine {
  bool started = false;
  std::vector<XDevice *> devs;
  XSimpleMempool *mempool = nullptr;
  XThreadpool *tpServer = nullptr;
  XThreadpool *tpClient = nullptr;
  std::vector<XContext *> serverCtxs;
  std::vector<XContext *> clientCtxs;
  std::vector<XConnector *> connectors; /* one per client ctx: honors dev */
  XListener *listener = nullptr;
  int barexPort = 0;
  union flagcxSocketAddress ifAddr; /* OOB IP for listen handles */

  std::mutex mu; /* guards the three maps below */
  std::unordered_map<uint64_t, PendingAccept> pendingAccepts;
  std::unordered_map<XChannel *, BarexComm *> channelComm;
  std::map<BarexMrKey, BarexMr *> mrs;

  std::mt19937_64 rng{std::random_device{}()};
};

static BarexEngine *gEngine = nullptr;
static std::mutex gEngineMu;

class BarexNetCallback : public XChannelCallback {
public:
  explicit BarexNetCallback(BarexEngine *engine) : engine_(engine) {}

  void OnRecvCall(XChannel *channel, char *buf, size_t len,
                  x_msg_header header) override {
    (void)header;
    if (buf == nullptr || len < sizeof(uint32_t))
      return;
    uint32_t type = 0;
    memcpy(&type, buf, sizeof(type));

    if (type == BAREX_MSG_HELLO && len >= sizeof(BarexHelloMsg)) {
      BarexHelloMsg hello;
      memcpy(&hello, buf, sizeof(hello));
      bool rejectChannel = false;
      {
        std::lock_guard<std::mutex> lk(engine_->mu);
        auto it = engine_->pendingAccepts.find(hello.commId);
        if (it == engine_->pendingAccepts.end())
          rejectChannel = true;
        else
          it->second.channels.push_back(channel);
      }
      if (rejectChannel) {
        WARN("NET/BAREX : HELLO for unknown commId 0x%llx",
             (unsigned long long)hello.commId);
        channel->Destroy();
      }
      return;
    }

    if (type == BAREX_MSG_CTS && len >= sizeof(BarexCtsMsg)) {
      BarexCtsMsg cts;
      memcpy(&cts, buf, sizeof(cts));
      BarexComm *comm = nullptr;
      {
        std::lock_guard<std::mutex> lk(engine_->mu);
        auto it = engine_->channelComm.find(channel);
        if (it != engine_->channelComm.end())
          comm = it->second;
      }
      if (comm == nullptr) {
        WARN("NET/BAREX : CTS on unbound channel %p", (void *)channel);
        return;
      }
      std::lock_guard<std::mutex> lk(comm->mu);
      comm->ctsPending[cts.seq] = cts;
      return;
    }

    WARN("NET/BAREX : unknown ctrl message type 0x%x len %zu", type, len);
  }

  void OnImmRecvCall(XChannel *channel, uint32_t imm) override {
    BarexComm *comm = nullptr;
    {
      std::lock_guard<std::mutex> lk(engine_->mu);
      auto it = engine_->channelComm.find(channel);
      if (it != engine_->channelComm.end())
        comm = it->second;
    }
    if (comm == nullptr)
      return;
    const uint32_t slot = imm & kImmSlotMask;
    if (slot >= kMaxRequests)
      return;
    BarexRequest *req = &comm->requests[slot];
    int expected = BAREX_REQ_PENDING;
    /* data is already placed: write-with-imm orders payload first */
    req->state.compare_exchange_strong(expected, BAREX_REQ_DONE);
  }

private:
  BarexEngine *engine_;
};

/* Cheap probe used by init()/devices(): device list only, no contexts. */
static flagcxResult_t barexProbeDevices(int *ndev) {
  /* Align ACCL's NIC indexing with FLAGCX_IB_HCA so rkey vectors agree
     between peers (same seeding as the P2P-engine ACCL transport). */
  const char *hca = flagcxGetEnv("FLAGCX_IB_HCA");
  if (hca != nullptr && flagcxGetEnv("ACCL_USE_NICS") == nullptr) {
    setenv("ACCL_USE_NICS", hca, 0);
    INFO(FLAGCX_INIT | FLAGCX_NET,
         "NET/BAREX : ACCL_USE_NICS=%s (from FLAGCX_IB_HCA)", hca);
  }
  XDeviceManager *mgr = nullptr;
  if (XDeviceManager::Singleton(mgr) != accl::barex::BAREX_SUCCESS ||
      mgr == nullptr)
    return flagcxInternalError;
  std::vector<XDevice *> devs = mgr->AllDevices();
  if (devs.empty() || (int)devs.size() > kMaxNics)
    return flagcxInternalError;
  if (ndev != nullptr)
    *ndev = (int)devs.size();
  return flagcxSuccess;
}

/* Full bring-up; called lazily from the first listen()/connect(). */
static flagcxResult_t barexEngineStart(BarexEngine **out) {
  std::lock_guard<std::mutex> lk(gEngineMu);
  if (gEngine != nullptr && gEngine->started) {
    *out = gEngine;
    return flagcxSuccess;
  }

  auto *e = gEngine != nullptr ? gEngine : new BarexEngine();
  gEngine = e;

  XDeviceManager *mgr = nullptr;
  if (XDeviceManager::Singleton(mgr) != accl::barex::BAREX_SUCCESS ||
      mgr == nullptr) {
    WARN("NET/BAREX : XDeviceManager unavailable");
    return flagcxInternalError;
  }
  e->devs = mgr->AllDevices();
  if (e->devs.empty()) {
    WARN("NET/BAREX : no ACCL devices");
    return flagcxInternalError;
  }

  BarexResult r =
      XSimpleMempool::NewInstance(e->mempool, "flagcx-net-barex", e->devs);
  if (r != accl::barex::BAREX_SUCCESS) {
    WARN("NET/BAREX : mempool: %s", bxstr(r));
    return flagcxInternalError;
  }
  XThreadpool::NewInstance(e->tpServer, 4, "flagcx-barex-server");
  XThreadpool::NewInstance(e->tpClient, 4, "flagcx-barex-client");

  ContextConfig cfg = XConfigUtil::DefaultContextConfig();
  for (XDevice *dev : e->devs) {
    XContext *sctx = nullptr, *cctx = nullptr;
    if (XContext::NewInstance(sctx, cfg, new BarexNetCallback(e), dev,
                              e->mempool,
                              e->tpServer) != accl::barex::BAREX_SUCCESS ||
        XContext::NewInstance(cctx, cfg, new BarexNetCallback(e), dev,
                              e->mempool,
                              e->tpClient) != accl::barex::BAREX_SUCCESS) {
      WARN("NET/BAREX : XContext create failed on %s", dev->GetName().c_str());
      return flagcxInternalError;
    }
    sctx->Start();
    cctx->Start();
    e->serverCtxs.push_back(sctx);
    e->clientCtxs.push_back(cctx);
  }

  /* one data-plane listener over all server contexts; probed free port */
  const int base = 19000 + (int)(getpid() % 4096);
  for (int attempt = 0; attempt < 32 && e->listener == nullptr; attempt++) {
    const int port = base + attempt * 3;
    XListener *lis = nullptr;
    if (XListener::NewInstance(lis, 2, port, accl::barex::TIMER_3S,
                               e->serverCtxs) == accl::barex::BAREX_SUCCESS &&
        lis->Listen() == accl::barex::BAREX_SUCCESS) {
      e->listener = lis;
      e->barexPort = port;
      break;
    }
    if (lis != nullptr) {
      lis->Shutdown();
      lis->WaitStop();
      delete lis;
    }
  }
  if (e->listener == nullptr) {
    WARN("NET/BAREX : no free data port");
    return flagcxInternalError;
  }

  /* one connector per client context so connect() can honor `dev` */
  for (XContext *ctx : e->clientCtxs) {
    XConnector *con = nullptr;
    std::vector<XContext *> one = {ctx};
    if (XConnector::NewInstance(con, 2, accl::barex::TIMER_3S, one) !=
        accl::barex::BAREX_SUCCESS) {
      WARN("NET/BAREX : XConnector create failed");
      return flagcxInternalError;
    }
    e->connectors.push_back(con);
  }

  /* OOB IP for handles: same interface bootstrap uses */
  bootstrapNetInit();
  union flagcxSocketAddress *ifAddr = bootstrapGetNetIfAddr();
  if (ifAddr == nullptr) {
    WARN("NET/BAREX : no OOB interface");
    return flagcxInternalError;
  }
  memcpy(&e->ifAddr, ifAddr, sizeof(e->ifAddr));

  e->started = true;
  INFO(FLAGCX_INIT | FLAGCX_NET, "NET/BAREX : engine up (nics=%zu port=%d)",
       e->devs.size(), e->barexPort);
  *out = e;
  return flagcxSuccess;
}

static flagcxResult_t flagcxBarexInit() {
  if (flagcxParamBarexDisable()) {
    INFO(FLAGCX_INIT | FLAGCX_NET,
         "NET/BAREX : disabled by FLAGCX_BAREX_DISABLE");
    return flagcxInternalError; /* registry falls through to next slot */
  }
  return barexProbeDevices(nullptr);
}

static flagcxResult_t flagcxBarexDevices(int *ndev) {
  return barexProbeDevices(ndev);
}

static flagcxResult_t flagcxBarexGetProperties(int dev, void *props) {
  if (props == nullptr)
    return flagcxInvalidArgument;
  int ndev = 0;
  FLAGCXCHECK(barexProbeDevices(&ndev));
  if (dev < 0 || dev >= ndev)
    return flagcxInvalidArgument;
  auto *p = static_cast<flagcxNetProperties_v1_t *>(props);
  memset(p, 0, sizeof(*p));

  static char devName[kMaxNics][64];
  const char *name = "barex";
  XDeviceManager *mgr = nullptr;
  if (XDeviceManager::Singleton(mgr) == accl::barex::BAREX_SUCCESS &&
      mgr != nullptr) {
    std::vector<XDevice *> devs = mgr->AllDevices();
    if (dev >= 0 && dev < (int)devs.size() && dev < kMaxNics) {
      snprintf(devName[dev], sizeof(devName[dev]), "%s",
               devs[dev]->GetName().c_str());
      name = devName[dev];
    }
  }
  p->name = const_cast<char *>(name);
  p->pciPath = nullptr;
  p->guid = (uint64_t)dev;
  /* RegUserMr pins cudaMalloc'd PPU memory (VMM off). PPU has no dmabuf. */
  p->ptrSupport = FLAGCX_PTR_HOST | FLAGCX_PTR_CUDA;
  p->regIsGlobal = 1; /* MRs live in the engine-wide mempool */
  p->speed = (int)flagcxParamBarexSpeed();
  p->port = 1;
  p->latency = 0;
  p->maxComms = 65536;
  p->maxRecvs = 1; /* proxy path always posts irecv(n=1) */
  p->netDeviceType = FLAGCX_NET_DEVICE_HOST;
  p->netDeviceVersion = FLAGCX_NET_DEVICE_INVALID_VERSION;
  return flagcxSuccess;
}

static flagcxResult_t flagcxBarexListen(int dev, void *opaqueHandle,
                                        void **listenComm) {
  BarexEngine *e = nullptr;
  FLAGCXCHECK(barexEngineStart(&e));

  auto *handle = static_cast<BarexNetHandle *>(opaqueHandle);
  memset(handle, 0, sizeof(*handle));
  memcpy(&handle->connectAddr, &e->ifAddr, sizeof(handle->connectAddr));
  addrSetPort(&handle->connectAddr, e->barexPort);

  auto *lc = new BarexListenComm();
  lc->engine = e;
  lc->dev = dev;
  {
    std::lock_guard<std::mutex> lk(e->mu);
    do {
      lc->commId = e->rng();
    } while (lc->commId == 0 || e->pendingAccepts.count(lc->commId) != 0);
    e->pendingAccepts[lc->commId] = PendingAccept{};
  }
  handle->commId = lc->commId;
  handle->state = BAREX_CONN_INIT;
  handle->connectState = nullptr;

  *listenComm = lc;
  return flagcxSuccess;
}

/* Non-blocking, resumable: *sendComm stays NULL until the channel is up
   and HELLO delivered; state lives in the handle across retries. */
static flagcxResult_t flagcxBarexConnect(int dev, void *opaqueHandle,
                                         void **sendComm) {
  *sendComm = nullptr;
  BarexEngine *e = nullptr;
  FLAGCXCHECK(barexEngineStart(&e));
  auto *handle = static_cast<BarexNetHandle *>(opaqueHandle);

  if (handle->state == BAREX_CONN_INIT) {
    auto *st = new BarexConnectState();
    handle->connectState = st;

    char ip[64] = {0};
    if (handle->connectAddr.sa.sa_family == AF_INET) {
      inet_ntop(AF_INET, &handle->connectAddr.sin.sin_addr, ip, sizeof(ip));
    } else {
      inet_ntop(AF_INET6, &handle->connectAddr.sin6.sin6_addr, ip, sizeof(ip));
    }
    const int port = ntohs(handle->connectAddr.sa.sa_family == AF_INET
                               ? handle->connectAddr.sin.sin_port
                               : handle->connectAddr.sin6.sin6_port);

    XConnector *con =
        e->connectors[(dev >= 0 && dev < (int)e->connectors.size()) ? dev : 0];
    BarexResult r =
        con->Connect(std::string(ip), port, [st](XChannel *ch, Status s) {
          if (s.IsOk() && ch != nullptr)
            st->channel.store(ch, std::memory_order_release);
          else
            st->connectFailed.store(true, std::memory_order_release);
        });
    if (r != accl::barex::BAREX_SUCCESS) {
      WARN("NET/BAREX : Connect(%s:%d) sync error: %s", ip, port, bxstr(r));
      delete st;
      handle->connectState = nullptr;
      return flagcxInternalError;
    }
    handle->state = BAREX_CONN_CONNECTING;
    return flagcxSuccess; /* in progress */
  }

  auto *st = static_cast<BarexConnectState *>(handle->connectState);
  if (st == nullptr)
    return flagcxInternalError;

  if (handle->state == BAREX_CONN_CONNECTING) {
    if (st->connectFailed.load(std::memory_order_acquire)) {
      WARN("NET/BAREX : channel connect failed (commId 0x%llx)",
           (unsigned long long)handle->commId);
      delete st;
      handle->connectState = nullptr;
      return flagcxInternalError;
    }
    XChannel *ch = st->channel.load(std::memory_order_acquire);
    if (ch == nullptr)
      return flagcxSuccess; /* still connecting */

    auto *comm = new BarexComm();
    comm->engine = e;
    comm->channel = ch;
    comm->commId = handle->commId;
    comm->isSend = true;
    st->comm = comm;
    {
      std::lock_guard<std::mutex> lk(e->mu);
      e->channelComm[ch] = comm;
    }

    /* HELLO rides a small pooled host buffer; auto_release returns it */
    memp_t msg;
    if (e->mempool->AllocBuffer(msg, sizeof(BarexHelloMsg), accl::barex::CPU,
                                ch->GetLocalNicId(),
                                0) != accl::barex::BAREX_SUCCESS) {
      WARN("NET/BAREX : HELLO buffer alloc failed");
      {
        std::lock_guard<std::mutex> lk(e->mu);
        e->channelComm.erase(ch);
      }
      delete comm;
      delete st;
      handle->connectState = nullptr;
      return flagcxInternalError;
    }
    BarexHelloMsg hello;
    hello.type = BAREX_MSG_HELLO;
    hello.pad = 0;
    hello.commId = handle->commId;
    memcpy(msg.buf, &hello, sizeof(hello));
    msg.buf_len = sizeof(hello);
    x_msg_header hdr;
    memset(&hdr, 0, sizeof(hdr));
    BarexResult r = ch->Send(
        msg, /*auto_release=*/true, hdr,
        [st](Status s) {
          if (s.IsOk())
            st->helloDone.store(true, std::memory_order_release);
          else
            st->helloFailed.store(true, std::memory_order_release);
        },
        true);
    if (r != accl::barex::BAREX_SUCCESS) {
      WARN("NET/BAREX : HELLO send sync error: %s", bxstr(r));
      /* per xchannel.h: on send failure the buffer is NOT auto-released */
      e->mempool->ReleaseBuffer(msg.buf, accl::barex::CPU);
      {
        std::lock_guard<std::mutex> lk(e->mu);
        e->channelComm.erase(ch);
      }
      delete comm;
      delete st;
      handle->connectState = nullptr;
      return flagcxInternalError;
    }
    handle->state = BAREX_CONN_HELLO;
    return flagcxSuccess; /* in progress */
  }

  if (st->helloFailed.load(std::memory_order_acquire)) {
    WARN("NET/BAREX : HELLO delivery failed");
    if (st->comm != nullptr) {
      std::lock_guard<std::mutex> lk(e->mu);
      e->channelComm.erase(st->comm->channel);
    }
    delete st->comm;
    delete st;
    handle->connectState = nullptr;
    return flagcxInternalError;
  }
  if (!st->helloDone.load(std::memory_order_acquire))
    return flagcxSuccess; /* still in flight */

  *sendComm = st->comm;
  delete st;
  handle->connectState = nullptr;
  INFO(FLAGCX_NET, "NET/BAREX : sendComm up (commId 0x%llx)",
       (unsigned long long)handle->commId);
  return flagcxSuccess;
}

/* Non-blocking: ready once the connector's HELLO bound a channel. */
static flagcxResult_t flagcxBarexAccept(void *listenComm, void **recvComm) {
  if (recvComm == nullptr)
    return flagcxInvalidArgument;
  *recvComm = nullptr;
  auto *lc = static_cast<BarexListenComm *>(listenComm);
  if (lc == nullptr)
    return flagcxInternalError;
  BarexEngine *e = lc->engine;

  XChannel *ch = nullptr;
  {
    std::lock_guard<std::mutex> lk(e->mu);
    auto it = e->pendingAccepts.find(lc->commId);
    if (it == e->pendingAccepts.end())
      return flagcxInternalError;
    if (!it->second.channels.empty()) {
      ch = it->second.channels.front();
      it->second.channels.pop_front();
    }
  }
  if (ch == nullptr)
    return flagcxSuccess; /* no HELLO yet — call again */

  auto *comm = new BarexComm();
  comm->engine = e;
  comm->channel = ch;
  comm->commId = lc->commId;
  comm->isSend = false;
  {
    std::lock_guard<std::mutex> lk(e->mu);
    e->channelComm[ch] = comm;
  }
  *recvComm = comm;
  INFO(FLAGCX_NET, "NET/BAREX : recvComm up (commId 0x%llx)",
       (unsigned long long)lc->commId);
  return flagcxSuccess;
}

static flagcxResult_t barexCloseComm(BarexComm *comm) {
  if (comm == nullptr)
    return flagcxSuccess;
  BarexEngine *e = comm->engine;
  comm->dead.store(true, std::memory_order_release);
  if (e != nullptr && comm->channel != nullptr) {
    {
      std::lock_guard<std::mutex> lk(e->mu);
      e->channelComm.erase(comm->channel);
    }
    if (comm->isSend && !e->connectors.empty()) {
      /* connector owns close notification; server side lets the peer's
         close + heartbeat reap its end (Mooncake does the same) */
      XChannel *ch = comm->channel;
      e->connectors[0]->CloseChannel(ch, [ch](Status) { ch->Destroy(); });
    }
  }
  delete comm;
  return flagcxSuccess;
}

static flagcxResult_t flagcxBarexCloseSend(void *sendComm) {
  return barexCloseComm(static_cast<BarexComm *>(sendComm));
}

static flagcxResult_t flagcxBarexCloseRecv(void *recvComm) {
  return barexCloseComm(static_cast<BarexComm *>(recvComm));
}

static flagcxResult_t flagcxBarexCloseListen(void *listenComm) {
  auto *lc = static_cast<BarexListenComm *>(listenComm);
  if (lc == nullptr)
    return flagcxSuccess;
  std::deque<XChannel *> pendingChannels;
  {
    std::lock_guard<std::mutex> lk(lc->engine->mu);
    auto it = lc->engine->pendingAccepts.find(lc->commId);
    if (it != lc->engine->pendingAccepts.end()) {
      pendingChannels.swap(it->second.channels);
      lc->engine->pendingAccepts.erase(it);
    }
  }
  for (XChannel *channel : pendingChannels)
    if (channel != nullptr)
      channel->Destroy();
  delete lc;
  return flagcxSuccess;
}

static flagcxResult_t flagcxBarexRegMr(void *comm, void *data, size_t size,
                                       int type, int mrFlags, void **mhandle) {
  (void)comm;
  (void)mrFlags;
  if (mhandle == nullptr)
    return flagcxInvalidArgument;
  *mhandle = nullptr;
  if (data == nullptr || size == 0 ||
      (type != FLAGCX_PTR_HOST && type != FLAGCX_PTR_CUDA))
    return flagcxInvalidArgument;
  BarexEngine *e = nullptr;
  FLAGCXCHECK(barexEngineStart(&e));

  const uintptr_t base = (uintptr_t)data;
  const accl::barex::device_type dtype =
      (type == FLAGCX_PTR_CUDA) ? accl::barex::GPU : accl::barex::CPU;
  int devId = 0;
  if (dtype == accl::barex::GPU && deviceAdaptor != nullptr &&
      deviceAdaptor->getDevice != nullptr) {
    deviceAdaptor->getDevice(&devId);
  }
  /* DeregUserMr only identifies an MR by (base, dtype), so never publish two
     physical registrations with the same key. */
  const BarexMrKey key = {base, dtype};
  std::unique_lock<std::mutex> lk(e->mu);
  auto exact = e->mrs.find(key);
  if (exact != e->mrs.end()) {
    BarexMr *existing = exact->second;
    if (existing->devId != devId || size > existing->size) {
      WARN("NET/BAREX : incompatible duplicate MR for %p (%zu bytes, dev%d); "
           "existing registration is %zu bytes on dev%d",
           data, size, devId, existing->size, existing->devId);
      return flagcxInvalidArgument;
    }
    existing->refCount++;
    *mhandle = existing;
    return flagcxSuccess;
  }
  for (const auto &entry : e->mrs) {
    BarexMr *existing = entry.second;
    if (existing->dtype != dtype || existing->devId != devId ||
        base < existing->base)
      continue;
    const size_t offset = (size_t)(base - existing->base);
    if (offset <= existing->size && size <= existing->size - offset) {
      existing->refCount++;
      *mhandle = existing;
      return flagcxSuccess;
    }
  }

  auto *mr = new BarexMr();
  BarexResult r = e->mempool->RegUserMr(mr->mem, data, size, dtype, devId);
  if (r != accl::barex::BAREX_SUCCESS) {
    WARN("NET/BAREX : RegUserMr(%p,%zu,%s,dev%d) failed: %s (CUDA-VMM memory "
         "cannot be pinned — run with FLAGCX_VMM_ENABLE=0)",
         data, size, dtype == accl::barex::GPU ? "GPU" : "CPU", devId,
         bxstr(r));
    delete mr;
    return flagcxInternalError;
  }
  mr->base = base;
  mr->size = size;
  mr->dtype = dtype;
  mr->devId = devId;
  mr->refCount = 1;
  for (auto &kv : mr->mem.mrs) {
    const int nic = kv.first;
    if (nic < 0 || nic >= kMaxNics || kv.second == nullptr)
      continue;
    mr->lkeys[nic] = kv.second->lkey;
    mr->rkeys[nic] = kv.second->rkey;
    if ((uint32_t)(nic + 1) > mr->nKeys)
      mr->nKeys = nic + 1;
  }
  if (mr->nKeys == 0) {
    e->mempool->DeregUserMr(data, dtype);
    delete mr;
    return flagcxInternalError;
  }

  e->mrs[key] = mr;
  *mhandle = mr;
  return flagcxSuccess;
}

static flagcxResult_t flagcxBarexDeregMr(void *comm, void *mhandle) {
  (void)comm;
  auto *mr = static_cast<BarexMr *>(mhandle);
  if (mr == nullptr)
    return flagcxSuccess;
  BarexEngine *e = gEngine;
  if (e == nullptr)
    return flagcxSuccess;
  void *base = nullptr;
  accl::barex::device_type dtype = accl::barex::CPU;
  {
    std::lock_guard<std::mutex> lk(e->mu);
    if (--mr->refCount > 0)
      return flagcxSuccess;
    e->mrs.erase(BarexMrKey{mr->base, mr->dtype});
    base = (void *)mr->base;
    dtype = mr->dtype;
    /* Keep registration and deregistration serialized. Otherwise a new
       RegUserMr for this key could race before the old physical MR is gone. */
    e->mempool->DeregUserMr(base, dtype);
  }
  delete mr;
  return flagcxSuccess;
}

static flagcxResult_t flagcxBarexGetMrInfo(void *mhandle,
                                           struct flagcxNetMrInfo *info) {
  if (mhandle == nullptr || info == nullptr)
    return flagcxInvalidArgument;
  auto *mr = static_cast<BarexMr *>(mhandle);
  if (mr->nKeys == 0 || mr->nKeys > FLAGCX_NET_MAX_MR_KEYS)
    return flagcxInternalError;
  memset(info, 0, sizeof(*info));
  info->nKeys = mr->nKeys;
  memcpy(info->lkeys, mr->lkeys, mr->nKeys * sizeof(uint32_t));
  memcpy(info->rkeys, mr->rkeys, mr->nKeys * sizeof(uint32_t));
  return flagcxSuccess;
}

static flagcxResult_t flagcxBarexIsend(void *sendComm, void *data, size_t size,
                                       int tag, void *mhandle, void *phandle,
                                       void **request) {
  (void)tag; /* always 0 on the proxy path */
  (void)phandle;
  if (request == nullptr)
    return flagcxInvalidArgument;
  *request = nullptr;
  auto *comm = static_cast<BarexComm *>(sendComm);
  auto *mr = static_cast<BarexMr *>(mhandle);
  if (comm == nullptr || (data == nullptr && size != 0) || mr == nullptr)
    return flagcxInvalidArgument;
  if (comm->dead.load(std::memory_order_acquire))
    return flagcxInternalError;

  BarexCtsMsg cts;
  BarexRequest *req = nullptr;
  {
    std::lock_guard<std::mutex> lk(comm->mu);
    /* Consume CTS in receiver post order (seq == chunk index). If the
       expected seq hasn't arrived, leave sendExpectedSeq and let the proxy
       retry — never pair a chunk with a different CTS. */
    auto it = comm->ctsPending.find(comm->sendExpectedSeq);
    if (it == comm->ctsPending.end())
      return flagcxSuccess; /* CTS for this chunk not here yet — retry */
    req = comm->allocRequest();
    if (req == nullptr)
      return flagcxSuccess; /* request pool exhausted — retry (keep CTS) */
    cts = it->second;
    comm->ctsPending.erase(it);
    comm->sendExpectedSeq++;
  }

  /* Clamp to the posted recv size (ibrc semantics: send truncates). */
  const size_t wsize = size < cts.size ? size : (size_t)cts.size;
  req->size = wsize;

  const int peerNic = comm->channel->GetPeerNicId();
  const uint32_t rkey =
      (peerNic >= 0 && (uint32_t)peerNic < cts.nKeys && peerNic < kMaxNics)
          ? cts.rkeys[peerNic]
          : cts.rkeys[0];

  /* interior pointer inside the registered region: memp_t.buf may sit
     past .base; WriteSingle resolves the lkey from .mrs per NIC */
  memp_t payload = mr->mem;
  payload.buf = static_cast<char *>(data);
  payload.buf_len = wsize;

  BarexRequest *reqCapture = req;
  BarexResult r = comm->channel->WriteSingle(
      payload, cts.addr, rkey, /*signal_peer=*/true,
      /*imm_data=*/cts.slot & kImmSlotMask,
      [reqCapture](Status s) {
        reqCapture->state.store(s.IsOk() ? BAREX_REQ_DONE : BAREX_REQ_ERROR,
                                std::memory_order_release);
      },
      /*done_inline=*/true, UINT64_MAX);
  if (r != accl::barex::BAREX_SUCCESS) {
    WARN("NET/BAREX : WriteSingle sync error: %s", bxstr(r));
    req->state.store(BAREX_REQ_FREE, std::memory_order_release);
    return flagcxInternalError;
  }
  *request = req;
  return flagcxSuccess;
}

static flagcxResult_t flagcxBarexIrecv(void *recvComm, int n, void **data,
                                       size_t *sizes, int *tags,
                                       void **mhandles, void **phandles,
                                       void **request) {
  (void)tags;
  (void)phandles;
  if (request == nullptr)
    return flagcxInvalidArgument;
  *request = nullptr;
  auto *comm = static_cast<BarexComm *>(recvComm);
  if (comm == nullptr || n != 1 || data == nullptr || sizes == nullptr ||
      mhandles == nullptr || (data[0] == nullptr && sizes[0] != 0))
    return flagcxInvalidArgument;
  auto *mr = static_cast<BarexMr *>(mhandles[0]);
  if (mr == nullptr)
    return flagcxInvalidArgument;
  if (comm->dead.load(std::memory_order_acquire))
    return flagcxInternalError;
  BarexEngine *e = comm->engine;

  BarexRequest *req = nullptr;
  uint64_t seq = 0;
  {
    std::lock_guard<std::mutex> lk(comm->mu);
    req = comm->allocRequest();
    if (req != nullptr)
      seq = comm->recvSeq++; /* stamp in post order (same lock as alloc) */
  }
  if (req == nullptr)
    return flagcxSuccess; /* pool exhausted — proxy re-posts */
  req->size = sizes[0];

  BarexCtsMsg cts;
  memset(&cts, 0, sizeof(cts));
  cts.type = BAREX_MSG_CTS;
  cts.slot = req->slot;
  cts.addr = (uint64_t)(uintptr_t)data[0];
  cts.size = sizes[0];
  cts.nKeys = mr->nKeys;
  cts.seq = (uint32_t)seq;
  memcpy(cts.rkeys, mr->rkeys, sizeof(cts.rkeys));

  memp_t msg;
  if (e->mempool->AllocBuffer(msg, sizeof(BarexCtsMsg), accl::barex::CPU,
                              comm->channel->GetLocalNicId(),
                              0) != accl::barex::BAREX_SUCCESS) {
    req->state.store(BAREX_REQ_FREE, std::memory_order_release);
    WARN("NET/BAREX : CTS buffer alloc failed");
    return flagcxInternalError;
  }
  memcpy(msg.buf, &cts, sizeof(cts));
  msg.buf_len = sizeof(cts);
  x_msg_header hdr;
  memset(&hdr, 0, sizeof(hdr));

  BarexRequest *reqCapture = req;
  BarexResult r = comm->channel->Send(
      msg, /*auto_release=*/true, hdr,
      [reqCapture](Status s) {
        if (!s.IsOk()) /* CTS lost: fail the request; sender never writes */
          reqCapture->state.store(BAREX_REQ_ERROR, std::memory_order_release);
      },
      true);
  if (r != accl::barex::BAREX_SUCCESS) {
    WARN("NET/BAREX : CTS send sync error: %s", bxstr(r));
    /* per xchannel.h: on send failure the buffer is NOT auto-released */
    e->mempool->ReleaseBuffer(msg.buf, accl::barex::CPU);
    req->state.store(BAREX_REQ_FREE, std::memory_order_release);
    return flagcxInternalError;
  }
  /* completion arrives via OnImmRecvCall(imm == req->slot) */
  *request = req;
  return flagcxSuccess;
}

/* Nothing to flush: write-with-imm orders payload before the completing
   imm. Return the (void*)0x1 sentinel because the RDMA-slot consumer
   (net.cc flagcxProxyRecv) only advances its flush stage on a request. */
static flagcxResult_t flagcxBarexIflush(void *recvComm, int n, void **data,
                                        int *sizes, void **mhandles,
                                        void **request) {
  (void)recvComm;
  (void)n;
  (void)data;
  (void)sizes;
  (void)mhandles;
  if (request == nullptr)
    return flagcxInvalidArgument;
  *request = reinterpret_cast<void *>(kCompletedRequest);
  return flagcxSuccess;
}

static flagcxResult_t flagcxBarexTest(void *request, int *done, int *sizes) {
  if (done == nullptr)
    return flagcxInvalidArgument;
  *done = 0;
  if (request == nullptr ||
      reinterpret_cast<uintptr_t>(request) == kCompletedRequest) {
    if (sizes != nullptr)
      sizes[0] = 0;
    *done = 1;
    return flagcxSuccess;
  }
  auto *req = static_cast<BarexRequest *>(request);
  const int st = req->state.load(std::memory_order_acquire);
  if (st == BAREX_REQ_PENDING)
    return flagcxSuccess;
  if (st == BAREX_REQ_ERROR) {
    /* Sticky: proxy re-tests the same pointer, so don't recycle the slot;
       surface the error every call. */
    return flagcxInternalError;
  }
  *done = 1;
  if (sizes != nullptr)
    sizes[0] = (int)req->size;
  req->state.store(BAREX_REQ_FREE, std::memory_order_release);
  return flagcxSuccess;
}

static flagcxResult_t barexPrepareOneSided(
    BarexComm *comm, const struct flagcxOneSideHandleInfo *localInfo,
    int localRank, uint64_t localOff,
    const struct flagcxOneSideHandleInfo *remoteInfo, int remoteRank,
    uint64_t remoteOff, size_t size, memp_t *localMem, uint64_t *remoteAddr,
    uint32_t *remoteRkey) {
  if (comm == nullptr || localInfo == nullptr || remoteInfo == nullptr ||
      localMem == nullptr || remoteAddr == nullptr || remoteRkey == nullptr ||
      localInfo->baseVas == nullptr || remoteInfo->baseVas == nullptr ||
      localInfo->regionSizes == nullptr || remoteInfo->regionSizes == nullptr ||
      localInfo->mrInfos == nullptr || remoteInfo->mrInfos == nullptr ||
      localInfo->localMrHandle == nullptr || localRank < 0 || remoteRank < 0 ||
      localRank >= localInfo->nRanks || remoteRank >= remoteInfo->nRanks)
    return flagcxInvalidArgument;
  const size_t localSize = localInfo->regionSizes[localRank];
  const size_t remoteSize = remoteInfo->regionSizes[remoteRank];
  if (localOff > localSize || size > localSize - localOff ||
      remoteOff > remoteSize || size > remoteSize - remoteOff)
    return flagcxInvalidArgument;
  if (comm->dead.load(std::memory_order_acquire) || comm->channel == nullptr)
    return flagcxInternalError;

  const int localNic = comm->channel->GetLocalNicId();
  const int peerNic = comm->channel->GetPeerNicId();
  const struct flagcxNetMrInfo &peerMr = remoteInfo->mrInfos[remoteRank];
  if (localNic < 0 || localNic >= kMaxNics || peerNic < 0 ||
      peerNic >= kMaxNics || (uint32_t)peerNic >= peerMr.nKeys)
    return flagcxInvalidArgument;

  auto *mr = static_cast<BarexMr *>(localInfo->localMrHandle);
  auto mrIt = mr->mem.mrs.find(localNic);
  if (mrIt == mr->mem.mrs.end() || mrIt->second == nullptr)
    return flagcxInvalidArgument;

  *localMem = mr->mem;
  localMem->buf =
      reinterpret_cast<char *>(localInfo->baseVas[localRank] + localOff);
  localMem->buf_len = size;
  localMem->mr = mrIt->second;
  *remoteAddr = remoteInfo->baseVas[remoteRank] + remoteOff;
  *remoteRkey = peerMr.rkeys[peerNic];
  return flagcxSuccess;
}

static flagcxResult_t flagcxBarexIput(void *sendComm, uint64_t srcOff,
                                      uint64_t dstOff, size_t size, int srcRank,
                                      int dstRank, void **srcHandles,
                                      void **dstHandles, void **request) {
  if (request == nullptr)
    return flagcxInvalidArgument;
  *request = nullptr;
  auto *comm = static_cast<BarexComm *>(sendComm);
  auto *srcInfo =
      reinterpret_cast<const struct flagcxOneSideHandleInfo *>(srcHandles);
  auto *dstInfo =
      reinterpret_cast<const struct flagcxOneSideHandleInfo *>(dstHandles);
  memp_t localMem;
  uint64_t remoteAddr = 0;
  uint32_t remoteRkey = 0;
  FLAGCXCHECK(barexPrepareOneSided(comm, srcInfo, srcRank, srcOff, dstInfo,
                                   dstRank, dstOff, size, &localMem,
                                   &remoteAddr, &remoteRkey));

  BarexRequest *req = comm->allocRequest();
  if (req == nullptr)
    return flagcxInternalError;
  req->size = size;
  if (size == 0) {
    req->state.store(BAREX_REQ_DONE, std::memory_order_release);
    *request = req;
    return flagcxSuccess;
  }

  BarexResult r = comm->channel->WriteSingle(
      localMem, remoteAddr, remoteRkey, /*signal_peer=*/false, 0,
      [req](Status s) {
        req->state.store(s.IsOk() ? BAREX_REQ_DONE : BAREX_REQ_ERROR,
                         std::memory_order_release);
      },
      /*done_inline=*/true, UINT64_MAX);
  if (r != accl::barex::BAREX_SUCCESS) {
    req->state.store(BAREX_REQ_FREE, std::memory_order_release);
    return barexResult(r);
  }
  *request = req;
  return flagcxSuccess;
}

static flagcxResult_t flagcxBarexIget(void *sendComm, uint64_t srcOff,
                                      uint64_t dstOff, size_t size, int srcRank,
                                      int dstRank, void **srcHandles,
                                      void **dstHandles, void **request) {
  if (request == nullptr)
    return flagcxInvalidArgument;
  *request = nullptr;
  auto *comm = static_cast<BarexComm *>(sendComm);
  auto *srcInfo =
      reinterpret_cast<const struct flagcxOneSideHandleInfo *>(srcHandles);
  auto *dstInfo =
      reinterpret_cast<const struct flagcxOneSideHandleInfo *>(dstHandles);
  memp_t localMem;
  uint64_t remoteAddr = 0;
  uint32_t remoteRkey = 0;
  FLAGCXCHECK(barexPrepareOneSided(comm, dstInfo, dstRank, dstOff, srcInfo,
                                   srcRank, srcOff, size, &localMem,
                                   &remoteAddr, &remoteRkey));

  BarexRequest *req = comm->allocRequest();
  if (req == nullptr)
    return flagcxInternalError;
  req->size = size;
  if (size == 0) {
    req->state.store(BAREX_REQ_DONE, std::memory_order_release);
    *request = req;
    return flagcxSuccess;
  }

  BarexResult r = comm->channel->ReadSingle(
      localMem, remoteAddr, remoteRkey,
      [req](Status s) {
        req->state.store(s.IsOk() ? BAREX_REQ_DONE : BAREX_REQ_ERROR,
                         std::memory_order_release);
      },
      /*done_inline=*/true, UINT64_MAX);
  if (r != accl::barex::BAREX_SUCCESS) {
    req->state.store(BAREX_REQ_FREE, std::memory_order_release);
    return barexResult(r);
  }
  *request = req;
  return flagcxSuccess;
}

static flagcxResult_t
flagcxBarexIputBatch(void *sendComm, int count, const uint64_t *srcOffs,
                     const uint64_t *dstOffs, const size_t *sizes, int srcRank,
                     int dstRank, void **srcHandles, void **dstHandles,
                     void **requests, int *posted) {
  if (posted == nullptr || requests == nullptr)
    return flagcxInvalidArgument;
  *posted = 0;
  if (count < 0 || count > kMaxRequests)
    return flagcxInvalidArgument;
  for (int i = 0; i < count; i++)
    requests[i] = nullptr;
  if (count == 0)
    return flagcxSuccess;
  if (srcOffs == nullptr || dstOffs == nullptr || sizes == nullptr)
    return flagcxInvalidArgument;

  auto *comm = static_cast<BarexComm *>(sendComm);
  auto *srcInfo =
      reinterpret_cast<const struct flagcxOneSideHandleInfo *>(srcHandles);
  auto *dstInfo =
      reinterpret_cast<const struct flagcxOneSideHandleInfo *>(dstHandles);
  auto data = std::make_shared<std::vector<rw_memp_t>>();
  auto reqs = std::make_shared<std::vector<BarexRequest *>>();
  data->reserve(count);
  reqs->reserve(count);

  for (int i = 0; i < count; i++) {
    memp_t localMem;
    uint64_t remoteAddr = 0;
    uint32_t remoteRkey = 0;
    flagcxResult_t prep = barexPrepareOneSided(
        comm, srcInfo, srcRank, srcOffs[i], dstInfo, dstRank, dstOffs[i],
        sizes[i], &localMem, &remoteAddr, &remoteRkey);
    if (prep != flagcxSuccess) {
      for (BarexRequest *req : *reqs)
        req->state.store(BAREX_REQ_FREE, std::memory_order_release);
      return prep;
    }
    BarexRequest *req = comm->allocRequest();
    if (req == nullptr) {
      for (BarexRequest *allocated : *reqs)
        allocated->state.store(BAREX_REQ_FREE, std::memory_order_release);
      return flagcxSuccess;
    }
    req->size = sizes[i];
    reqs->push_back(req);

    rw_memp_t rw;
    rw.data = localMem;
    rw.r_addr = remoteAddr;
    rw.r_key = remoteRkey;
    rw.r_ttl_ms = UINT64_MAX;
    rw.sg.addr = reinterpret_cast<uint64_t>(localMem.buf);
    rw.sg.length = static_cast<uint32_t>(sizes[i]);
    if (static_cast<size_t>(rw.sg.length) != sizes[i]) {
      for (BarexRequest *allocated : *reqs)
        allocated->state.store(BAREX_REQ_FREE, std::memory_order_release);
      return flagcxInvalidArgument;
    }
    rw.sg.lkey = localMem.mr->lkey;
    data->push_back(rw);
  }

  BarexResult r = comm->channel->WriteBatch(
      data,
      [reqs](Status s) {
        const int state = s.IsOk() ? BAREX_REQ_DONE : BAREX_REQ_ERROR;
        for (BarexRequest *req : *reqs)
          req->state.store(state, std::memory_order_release);
      },
      /*done_inline=*/true);
  if (r != accl::barex::BAREX_SUCCESS) {
    for (BarexRequest *req : *reqs)
      req->state.store(BAREX_REQ_FREE, std::memory_order_release);
    return barexResult(r);
  }
  for (int i = 0; i < count; i++)
    requests[i] = (*reqs)[i];
  *posted = count;
  return flagcxSuccess;
}

static flagcxResult_t flagcxBarexTestBatch(void **requests, int nRequests,
                                           int *doneFlags, int *doneCount) {
  if (nRequests < 0 || doneCount == nullptr ||
      (nRequests > 0 && (requests == nullptr || doneFlags == nullptr)))
    return flagcxInvalidArgument;
  *doneCount = 0;
  for (int i = 0; i < nRequests; i++) {
    int done = 0;
    FLAGCXCHECK(flagcxBarexTest(requests[i], &done, nullptr));
    doneFlags[i] = done;
    *doneCount += done;
  }
  return flagcxSuccess;
}

static flagcxResult_t
flagcxBarexIgetBatch(void *sendComm, int count, const uint64_t *srcOffs,
                     const uint64_t *dstOffs, const size_t *sizes, int srcRank,
                     int dstRank, void *const *srcHandles,
                     void *const *dstHandles, void **request) {
  if (request == nullptr)
    return flagcxInvalidArgument;
  *request = nullptr;
  if (count < 0 || count > kMaxRequests ||
      (count > 0 &&
       (srcOffs == nullptr || dstOffs == nullptr || sizes == nullptr)))
    return flagcxInvalidArgument;
  auto *comm = static_cast<BarexComm *>(sendComm);
  if (count == 0) {
    if (comm == nullptr)
      return flagcxInvalidArgument;
    BarexRequest *req = comm->allocRequest();
    if (req == nullptr)
      return flagcxInternalError;
    req->state.store(BAREX_REQ_DONE, std::memory_order_release);
    *request = req;
    return flagcxSuccess;
  }

  auto *srcInfo =
      reinterpret_cast<const struct flagcxOneSideHandleInfo *>(srcHandles);
  auto *dstInfo =
      reinterpret_cast<const struct flagcxOneSideHandleInfo *>(dstHandles);
  auto data = std::make_shared<std::vector<rw_memp_t>>();
  data->reserve(count);
  size_t totalSize = 0;
  for (int i = 0; i < count; i++) {
    memp_t localMem;
    uint64_t remoteAddr = 0;
    uint32_t remoteRkey = 0;
    FLAGCXCHECK(barexPrepareOneSided(comm, dstInfo, dstRank, dstOffs[i],
                                     srcInfo, srcRank, srcOffs[i], sizes[i],
                                     &localMem, &remoteAddr, &remoteRkey));
    rw_memp_t rw;
    rw.data = localMem;
    rw.r_addr = remoteAddr;
    rw.r_key = remoteRkey;
    rw.r_ttl_ms = UINT64_MAX;
    rw.sg.addr = reinterpret_cast<uint64_t>(localMem.buf);
    rw.sg.length = static_cast<uint32_t>(sizes[i]);
    if (static_cast<size_t>(rw.sg.length) != sizes[i])
      return flagcxInvalidArgument;
    rw.sg.lkey = localMem.mr->lkey;
    data->push_back(rw);
    totalSize += sizes[i];
  }

  BarexRequest *req = comm->allocRequest();
  if (req == nullptr)
    return flagcxInternalError;
  req->size = totalSize;
  BarexResult r = comm->channel->ReadBatch(
      data,
      [req](Status s) {
        req->state.store(s.IsOk() ? BAREX_REQ_DONE : BAREX_REQ_ERROR,
                         std::memory_order_release);
      },
      /*done_inline=*/true);
  if (r != accl::barex::BAREX_SUCCESS) {
    req->state.store(BAREX_REQ_FREE, std::memory_order_release);
    return barexResult(r);
  }
  *request = req;
  return flagcxSuccess;
}

static flagcxResult_t flagcxBarexGetDevFromName(char *name, int *dev) {
  if (name == nullptr || dev == nullptr)
    return flagcxInvalidArgument;
  XDeviceManager *mgr = nullptr;
  if (XDeviceManager::Singleton(mgr) != accl::barex::BAREX_SUCCESS ||
      mgr == nullptr)
    return flagcxSystemError;
  std::vector<XDevice *> devs = mgr->AllDevices();
  for (size_t i = 0; i < devs.size(); i++) {
    if (devs[i]->GetName() == name) {
      *dev = (int)i;
      return flagcxSuccess;
    }
  }
  return flagcxSystemError;
}

/* ------------------------------------------------------------------ */
/*  P2P Engine transport submission interface                         */
/* ------------------------------------------------------------------ */

static flagcxResult_t flagcxBarexGetRegistrationDevice(int netDev,
                                                       int *registrationDev) {
  if (registrationDev == nullptr || netDev < 0)
    return flagcxInvalidArgument;
  // BAREX registration is process-global and ignores the comm/device context.
  *registrationDev = 0;
  return flagcxSuccess;
}

static flagcxResult_t
flagcxBarexGetTransportCaps(void *, struct flagcxP2pTransportCaps *caps) {
  if (caps == nullptr)
    return flagcxInvalidArgument;
  caps->maxBatchSize = kMaxRequests;
  caps->maxInflightBatches = kMaxRequests;
  return flagcxSuccess;
}

static flagcxResult_t
flagcxBarexSubmitTransportBatch(void *sendComm,
                                const struct flagcxP2pTransportSlice *slices,
                                int count, void **request) {
  auto *comm = static_cast<BarexComm *>(sendComm);
  if (comm == nullptr || slices == nullptr || request == nullptr ||
      count <= 0 || count > kMaxRequests || comm->channel == nullptr ||
      comm->dead.load(std::memory_order_acquire))
    return flagcxInvalidArgument;
  *request = nullptr;

  const uint8_t opcode = slices[0].opcode;
  if (opcode != FLAGCX_P2P_TRANSPORT_WRITE &&
      opcode != FLAGCX_P2P_TRANSPORT_READ)
    return flagcxInvalidArgument;

  const int localNic = comm->channel->GetLocalNicId();
  const int peerNic = comm->channel->GetPeerNicId();
  if (localNic < 0 || localNic >= kMaxNics || peerNic < 0 ||
      peerNic >= kMaxNics)
    return flagcxInvalidArgument;

  auto data = std::make_shared<std::vector<rw_memp_t>>();
  data->reserve(count);
  size_t totalSize = 0;
  for (int i = 0; i < count; i++) {
    const struct flagcxP2pTransportSlice &planned = slices[i];
    if (planned.opcode != opcode || planned.length == 0 ||
        planned.localMrHandle == nullptr ||
        (uint32_t)peerNic >= planned.remoteMrInfo.nKeys) {
      return flagcxInvalidArgument;
    }

    auto *mr = static_cast<BarexMr *>(planned.localMrHandle);
    auto mrIt = mr->mem.mrs.find(localNic);
    if (mrIt == mr->mem.mrs.end() || mrIt->second == nullptr)
      return flagcxInvalidArgument;

    memp_t localMem = mr->mem;
    localMem.buf = reinterpret_cast<char *>(planned.localVa);
    localMem.buf_len = planned.length;
    localMem.mr = mrIt->second;

    rw_memp_t rw;
    rw.data = localMem;
    rw.r_addr = planned.remoteVa;
    rw.r_key = planned.remoteMrInfo.rkeys[peerNic];
    rw.r_ttl_ms = UINT64_MAX;
    rw.sg.addr = planned.localVa;
    rw.sg.length = planned.length;
    rw.sg.lkey = localMem.mr->lkey;
    data->push_back(rw);
    totalSize += planned.length;
  }

  BarexRequest *req = comm->allocRequest();
  if (req == nullptr)
    return flagcxInProgress;
  req->size = totalSize;

  auto completion = [req, data](Status status) {
    (void)data;
    req->state.store(status.IsOk() ? BAREX_REQ_DONE : BAREX_REQ_ERROR,
                     std::memory_order_release);
  };
  BarexResult result =
      opcode == FLAGCX_P2P_TRANSPORT_WRITE
          ? comm->channel->WriteBatch(data, completion, /*done_inline=*/true)
          : comm->channel->ReadBatch(data, completion, /*done_inline=*/true);
  if (result != accl::barex::BAREX_SUCCESS) {
    req->state.store(BAREX_REQ_FREE, std::memory_order_release);
    return barexResult(result);
  }

  *request = req;
  return flagcxSuccess;
}

static flagcxResult_t flagcxBarexProgressTransport(void *) {
  // ACCL drives completions from its own callback threads.
  return flagcxSuccess;
}

static flagcxResult_t flagcxBarexTestTransport(void *request, int *done,
                                               int *failed) {
  if (done == nullptr || failed == nullptr)
    return flagcxInvalidArgument;
  *done = 0;
  *failed = 0;
  if (request == nullptr) {
    *done = 1;
    return flagcxSuccess;
  }

  auto *req = static_cast<BarexRequest *>(request);
  const int state = req->state.load(std::memory_order_acquire);
  if (state == BAREX_REQ_PENDING)
    return flagcxSuccess;
  if (state != BAREX_REQ_DONE && state != BAREX_REQ_ERROR)
    return flagcxInternalError;

  *done = 1;
  *failed = state == BAREX_REQ_ERROR ? 1 : 0;
  req->state.store(BAREX_REQ_FREE, std::memory_order_release);
  return flagcxSuccess;
}

const struct flagcxP2pTransportOps flagcxP2pBarexTransportOps = {
    "BAREX",
    flagcxBarexGetRegistrationDevice,
    flagcxBarexGetTransportCaps,
    flagcxBarexSubmitTransportBatch,
    flagcxBarexProgressTransport,
    flagcxBarexTestTransport};

struct flagcxNetAdaptor flagcxNetBarex = {
    // Basic functions
    "BAREX",
    flagcxBarexInit,
    flagcxBarexDevices,
    flagcxBarexGetProperties,

    // Setup functions
    flagcxBarexListen,
    flagcxBarexConnect,
    flagcxBarexAccept,
    flagcxBarexCloseSend,
    flagcxBarexCloseRecv,
    flagcxBarexCloseListen,

    // Memory region functions
    flagcxBarexRegMr,
    nullptr, // regMrDmaBuf: ACCL/BAREX does not support DMA-BUF registration
    flagcxBarexDeregMr,

    // Two-sided functions
    flagcxBarexIsend,
    flagcxBarexIrecv,
    flagcxBarexIflush,
    flagcxBarexTest,

    // One-sided functions
    flagcxBarexIput,
    flagcxBarexIget,
    nullptr, // iputSignal: ACCL/BAREX does not expose remote atomic add

    // Device name lookup
    flagcxBarexGetDevFromName,

    // Optional batch helpers
    flagcxBarexIputBatch,
    flagcxBarexTestBatch,
    flagcxBarexIgetBatch,

    // MR metadata
    flagcxBarexGetMrInfo,
};

struct flagcxNetAdaptor flagcxP2pNetBarex = {
    // Basic functions
    "BAREX_P2P",
    flagcxBarexInit,
    flagcxBarexDevices,
    flagcxBarexGetProperties,

    // Setup functions
    flagcxBarexListen,
    flagcxBarexConnect,
    flagcxBarexAccept,
    flagcxBarexCloseSend,
    flagcxBarexCloseRecv,
    flagcxBarexCloseListen,

    // Memory region functions
    flagcxBarexRegMr,
    nullptr, // regMrDmaBuf: ACCL/BAREX does not support DMA-BUF registration
    flagcxBarexDeregMr,

    // Two-sided functions
    flagcxBarexIsend,
    flagcxBarexIrecv,
    flagcxBarexIflush,
    flagcxBarexTest,

    // One-sided functions
    flagcxBarexIput,
    flagcxBarexIget,
    nullptr, // iputSignal: ACCL/BAREX does not expose remote atomic add

    // Device name lookup
    flagcxBarexGetDevFromName,

    // Optional batch helpers
    flagcxBarexIputBatch,
    flagcxBarexTestBatch,
    flagcxBarexIgetBatch,

    // MR metadata
    flagcxBarexGetMrInfo,
};

/* Plugin export (FLAGCX_NET_ADAPTOR_PLUGIN, v1 vtable). Prefer this over
   linking libaccl_barex into libflagcx: the loader uses RTLD_LOCAL, keeping
   libu2mm.so out of the global symbol table — otherwise libpccl's own u2mm
   crashes in wrap_u2mm_symbols during pcclCommInitRank. */
extern "C" __attribute__((visibility(
    "default"))) struct flagcxNetAdaptor_v1 flagcxNetAdaptorPlugin_v1 = {
    "BAREX",
    flagcxBarexInit,
    flagcxBarexDevices,
    flagcxBarexGetProperties,
    flagcxBarexListen,
    flagcxBarexConnect,
    flagcxBarexAccept,
    flagcxBarexCloseSend,
    flagcxBarexCloseRecv,
    flagcxBarexCloseListen,
    flagcxBarexRegMr,
    NULL, // regMrDmaBuf
    flagcxBarexDeregMr,
    flagcxBarexIsend,
    flagcxBarexIrecv,
    flagcxBarexIflush,
    flagcxBarexTest,
    NULL, // iput
    NULL, // iget
    NULL, // iputSignal
    flagcxBarexGetDevFromName,
};

#endif // USE_ACCL_BAREX
