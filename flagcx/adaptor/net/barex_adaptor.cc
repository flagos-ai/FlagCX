/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX net adaptor "barex": collective/C2C transport over the vendor
 * ACCL library (accl::barex) for PPU + vsolar hosts. GPU memory registration
 * goes through ACCL's RegUserMr / XChannel. The currently exposed ACCL API
 * registers a mapped VA and has no DMA-BUF fd/offset entry point. A future
 * ACCL DMA-BUF API can be advertised without changing the common route order.
 *
 * Rendezvous (mirrors ibrc's CTS design): connect sends HELLO{commId}
 * over an XChannel; irecv posts CTS{slot,addr,size,rkeys,seq}; isend
 * answers WriteSingle(imm=slot) and OnImmRecvCall completes the recv
 * (write-with-imm orders payload before the imm). Shared state is mutex-
 * or atomic-guarded (callbacks run on ACCL IO threads).
 *
 * Built with USE_ACCL_BAREX=1 (RDMA registry slot, like USE_UCX) or
 * loaded as a plugin .so (preferred; see the export note below).
 * FLAGCX_IB_DISABLE=1 disables every RDMA-class adaptor, including BAREX.
 ************************************************************************/

#ifdef USE_ACCL_BAREX

#include "adaptor.h"
#include "barex_runtime.h"
#include "bootstrap.h"
#include "debug.h"
#include "flagcx_net.h"
#include "flagcx_net_adaptor.h"
#include "flagcx_p2p.h"
#include "net.h"
#include "net_transport.h"
#include "onesided.h"
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

#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <limits.h>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace barexnet {

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

FLAGCX_PARAM(BarexSpeed, "BAREX_SPEED", 100000); /* Mbps, for topo costing */
FLAGCX_PARAM(BarexMaxMrBytes, "BAREX_MAX_MR_BYTES", 64LL << 20);

constexpr int kMaxNics = 8;       /* matches ACCL per-NIC rkey fan-out */
constexpr int kMaxRequests = 256; /* proxy keeps <=16 in flight; roomy */
constexpr uint32_t kImmSlotMask = 0x00FFFFFFu; /* imm_data carries 24 bits */
constexpr uintptr_t kCompletedRequest = 1;

static thread_local bool gRuntimeConnectionConfigActive = false;
static thread_local flagcxBarexRuntimeConnectionConfig
    gRuntimeConnectionConfig = {};

static uint32_t barexConnectionChannelCount() {
  return gRuntimeConnectionConfigActive ? gRuntimeConnectionConfig.channels : 1;
}

static const char *bxstr(BarexResult r) {
  auto it = BarexResultStrings.find(r);
  return it == BarexResultStrings.end() ? "UNKNOWN" : it->second;
}

static flagcxResult_t barexResult(BarexResult result) {
  switch (result) {
    case accl::barex::BAREX_SUCCESS:
      return flagcxSuccess;
    case accl::barex::BAREX_ERR_ARG:
    case accl::barex::BAREX_ERR_NPE:
    case accl::barex::BAREX_ERR_ADDRESS:
    case accl::barex::BAREX_ERR_TYPE:
      return flagcxInvalidArgument;
    case accl::barex::BAREX_ERR_QUEUE_FULL:
    case accl::barex::BAREX_ERR_RATE_LIMITED:
      return flagcxInProgress;
    case accl::barex::BAREX_ERR_TCP:
    case accl::barex::BAREX_ERR_TIMEOUT:
    case accl::barex::BAREX_ERR_CHANNEL_STAT:
      return flagcxRemoteError;
    default:
      return flagcxInternalError;
  }
}

static flagcxResult_t barexStatus(Status status) {
  return status.IsOk() ? flagcxSuccess : barexResult(status.ErrCode());
}

static int barexWorkerCount(const FlagcxP2pGlobalConfig &config) {
  return std::max(1, config.workersPerPool);
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
static_assert(sizeof(BarexHelloMsg) == 16,
              "HELLO wire layout must remain compatible");

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
  uint32_t listenGeometry; /* low 16: device, high 16: channel count */
  void *connectState;      /* connect-side heap state across retries */
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
  BAREX_REQ_COMPLETING = 4,
};

enum BarexCommState : int {
  BAREX_COMM_ACTIVE = 0,
  BAREX_COMM_FAILED = 1,
  BAREX_COMM_CLOSING = 2,
  BAREX_COMM_CLOSED = 3,
};

struct BarexRequest {
  std::atomic<int> state{BAREX_REQ_FREE};
  std::atomic<int> result{flagcxSuccess};
  size_t size = 0;
  bool twoSided = false;
  struct BarexComm *comm = nullptr;
  uint32_t slot = 0;
};

/* MR registry is runtime-wide and refcounted (regIsGlobal semantics: the
   user-buffer path may register one buffer through several proxy
   connections). */

struct BarexRuntime;

struct BarexMr {
  memp_t mem; /* as returned by RegUserMr: per-NIC ibv_mr map */
  uintptr_t base = 0;
  size_t size = 0;
  accl::barex::device_type dtype = accl::barex::CPU;
  int devId = 0;
  int mrFlags = FLAGCX_NET_MR_FLAG_NONE;
  uint32_t nKeys = 0;
  uint32_t lkeys[kMaxNics] = {0};
  uint32_t rkeys[kMaxNics] = {0};
  int refCount = 0;
  bool reusable = true;
  bool deferred = false;
};

struct BarexComm {
  BarexRuntime *runtime = nullptr;
  XChannel *channel = nullptr;
  std::vector<XChannel *> channels;
  uint64_t commId = 0;
  bool isSend = false;
  int connectorDev = 0;
  std::atomic<int> state{BAREX_COMM_ACTIVE};
  std::atomic<int> firstError{flagcxSuccess};

  std::mutex callbackMu;
  std::condition_variable callbackCv;
  size_t activeCallbacks = 0;

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
        requests[i].twoSided = false;
        requests[i].result.store(flagcxSuccess, std::memory_order_relaxed);
        return &requests[i];
      }
    }
    return nullptr;
  }

  bool beginCallback() {
    std::lock_guard<std::mutex> lk(callbackMu);
    if (state.load(std::memory_order_acquire) != BAREX_COMM_ACTIVE)
      return false;
    activeCallbacks++;
    return true;
  }

  void endCallback() {
    std::lock_guard<std::mutex> lk(callbackMu);
    if (--activeCallbacks == 0)
      callbackCv.notify_all();
  }

  void waitCallbacks() {
    std::unique_lock<std::mutex> lk(callbackMu);
    callbackCv.wait(lk, [this] { return activeCallbacks == 0; });
  }

  void fail(flagcxResult_t result) {
    if (result == flagcxSuccess || result == flagcxInProgress)
      return;
    int expectedError = flagcxSuccess;
    firstError.compare_exchange_strong(expectedError, result,
                                       std::memory_order_acq_rel);
    int expectedState = BAREX_COMM_ACTIVE;
    state.compare_exchange_strong(expectedState, BAREX_COMM_FAILED,
                                  std::memory_order_acq_rel);
  }

  flagcxResult_t submissionStatus() const {
    if (state.load(std::memory_order_acquire) == BAREX_COMM_ACTIVE)
      return flagcxSuccess;
    const int result = firstError.load(std::memory_order_acquire);
    return result == flagcxSuccess ? flagcxInternalError
                                   : static_cast<flagcxResult_t>(result);
  }
};

struct BarexListenComm {
  BarexRuntime *runtime = nullptr;
  uint64_t commId = 0;
  int dev = 0;
  uint32_t channels = 1;
};

struct BarexConnectState {
  BarexRuntime *runtime = nullptr;
  int connectorDev = -1;
  std::vector<XChannel *> channels;
  uint32_t channelCount = 1;
  uint32_t connectStarted = 0;
  std::atomic<uint32_t> connectDone{0};
  std::atomic<bool> connectFailed{false};
  uint32_t helloStarted = 0;
  std::atomic<uint32_t> helloDone{0};
  std::atomic<bool> helloFailed{false};
  BarexComm *comm = nullptr;
  std::atomic<bool> abandoned{false};
  std::atomic<bool> cleanupStarted{false};
};

struct PendingAccept {
  std::vector<XChannel *> channels;
  uint32_t expectedChannels = 1;
  flagcxResult_t firstError = flagcxSuccess;
  int dev = -1;
  int nicId = -1;
};

class BarexNetCallback; /* fwd */
static void
barexCleanupAbandonedConnect(const std::shared_ptr<BarexConnectState> &state);

// Process-wide provider state owned by the BAREX net adaptor. This is not a
// second P2P Engine: scheduling, slicing, aggregation, and transfer progress
// remain in FlagcxP2pEngine.
struct BarexRuntime {
  bool started = false;
  std::vector<XDevice *> devs;
  XSimpleMempool *mempool = nullptr;
  XThreadpool *tpServer = nullptr;
  XThreadpool *tpClient = nullptr;
  std::vector<XContext *> serverCtxs;
  std::vector<XContext *> clientCtxs;
  std::vector<XConnector *> connectors; /* one per client ctx: honors dev */
  std::vector<XListener *> listeners;   /* one per server ctx: honors dev */
  std::vector<int> barexPorts;
  union flagcxSocketAddress ifAddr; /* OOB IP for listen handles */

  std::mutex mu; /* guards the three maps below */
  std::unordered_map<uint64_t, PendingAccept> pendingAccepts;
  std::unordered_map<XChannel *, BarexComm *> channelComm;
  std::map<uintptr_t, BarexMr *> mrByBase;

  std::mt19937_64 rng{std::random_device{}()};
};

static BarexRuntime *gBarexRuntime = nullptr;
static std::mutex gBarexRuntimeMu;

static void barexCompleteRequest(BarexRequest *request, flagcxResult_t result) {
  request->result.store(result, std::memory_order_relaxed);
  request->state.store(result == flagcxSuccess ? BAREX_REQ_DONE
                                               : BAREX_REQ_ERROR,
                       std::memory_order_release);
}

static void barexCompleteCallback(BarexRequest *request, Status status) {
  BarexComm *comm = request->comm;
  flagcxResult_t result = barexStatus(status);
  // Queue-full/rate-limit are retryable only when a submission is rejected
  // synchronously. Once the provider invokes the completion callback the
  // request slot is terminal and cannot be returned as InProgress.
  if (result == flagcxInProgress)
    result = flagcxInternalError;
  if (result != flagcxSuccess) {
    WARN("NET/BAREX : asynchronous transfer failed: %s",
         status.ErrMsg().c_str());
    comm->fail(result);
  }
  barexCompleteRequest(request, result);
  comm->endCallback();
}

static void barexReleaseRequest(BarexRequest *request) {
  request->size = 0;
  request->result.store(flagcxSuccess, std::memory_order_relaxed);
  request->state.store(BAREX_REQ_FREE, std::memory_order_release);
}

class BarexNetCallback : public XChannelCallback {
public:
  explicit BarexNetCallback(BarexRuntime *runtime) : runtime_(runtime) {}

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
      uint32_t channelCount = 0;
      uint32_t lane = 0;
      const flagcxResult_t geometryResult =
          flagcxBarexRuntimeDecodeHelloGeometry(hello.pad, &channelCount,
                                                &lane);
      bool accepted = false;
      {
        std::lock_guard<std::mutex> lk(runtime_->mu);
        auto it = runtime_->pendingAccepts.find(hello.commId);
        if (it != runtime_->pendingAccepts.end() &&
            channel->GetLocalNicId() == it->second.nicId &&
            geometryResult == flagcxSuccess &&
            channelCount == it->second.expectedChannels &&
            lane < it->second.channels.size() &&
            it->second.channels[lane] == nullptr) {
          it->second.channels[lane] = channel;
          accepted = true;
        } else if (it != runtime_->pendingAccepts.end()) {
          it->second.firstError = geometryResult == flagcxSuccess
                                      ? flagcxRemoteError
                                      : geometryResult;
        }
      }
      if (!accepted) {
        WARN("NET/BAREX : rejected HELLO commId 0x%llx geometry=0x%x",
             (unsigned long long)hello.commId, hello.pad);
        channel->Destroy();
      }
      return;
    }

    if (type == BAREX_MSG_CTS && len >= sizeof(BarexCtsMsg)) {
      BarexCtsMsg cts;
      memcpy(&cts, buf, sizeof(cts));
      BarexComm *comm = nullptr;
      {
        std::lock_guard<std::mutex> lk(runtime_->mu);
        auto it = runtime_->channelComm.find(channel);
        if (it != runtime_->channelComm.end() && it->second->beginCallback())
          comm = it->second;
      }
      if (comm == nullptr) {
        WARN("NET/BAREX : CTS on unbound channel %p", (void *)channel);
        return;
      }
      std::lock_guard<std::mutex> lk(comm->mu);
      comm->ctsPending[cts.seq] = cts;
      comm->endCallback();
      return;
    }

    WARN("NET/BAREX : unknown ctrl message type 0x%x len %zu", type, len);
  }

  void OnImmRecvCall(XChannel *channel, uint32_t imm) override {
    BarexComm *comm = nullptr;
    {
      std::lock_guard<std::mutex> lk(runtime_->mu);
      auto it = runtime_->channelComm.find(channel);
      if (it != runtime_->channelComm.end() && it->second->beginCallback())
        comm = it->second;
    }
    if (comm == nullptr)
      return;
    const uint32_t slot = imm & kImmSlotMask;
    if (slot >= kMaxRequests) {
      comm->endCallback();
      return;
    }
    BarexRequest *req = &comm->requests[slot];
    int expected = BAREX_REQ_PENDING;
    /* data is already placed: write-with-imm orders payload first */
    if (req->state.compare_exchange_strong(expected, BAREX_REQ_COMPLETING)) {
      req->result.store(flagcxSuccess, std::memory_order_relaxed);
      req->state.store(BAREX_REQ_DONE, std::memory_order_release);
    }
    comm->endCallback();
  }

private:
  BarexRuntime *runtime_;
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
static flagcxResult_t barexRuntimeStart(BarexRuntime **out) {
  std::lock_guard<std::mutex> lk(gBarexRuntimeMu);
  if (gBarexRuntime != nullptr && gBarexRuntime->started) {
    *out = gBarexRuntime;
    return flagcxSuccess;
  }

  auto *runtime = gBarexRuntime != nullptr ? gBarexRuntime : new BarexRuntime();
  gBarexRuntime = runtime;

  XDeviceManager *mgr = nullptr;
  if (XDeviceManager::Singleton(mgr) != accl::barex::BAREX_SUCCESS ||
      mgr == nullptr) {
    WARN("NET/BAREX : XDeviceManager unavailable");
    return flagcxInternalError;
  }
  runtime->devs = mgr->AllDevices();
  if (runtime->devs.empty()) {
    WARN("NET/BAREX : no ACCL devices");
    return flagcxInternalError;
  }

  BarexResult r = XSimpleMempool::NewInstance(
      runtime->mempool, "flagcx-net-barex", runtime->devs);
  if (r != accl::barex::BAREX_SUCCESS) {
    WARN("NET/BAREX : mempool: %s", bxstr(r));
    return flagcxInternalError;
  }
  const auto &workerConfig = flagcxP2pGlobalConfig();
  const int workerCount = barexWorkerCount(workerConfig);
  XThreadpool::NewInstance(runtime->tpServer, workerCount,
                           "flagcx-barex-server");
  XThreadpool::NewInstance(runtime->tpClient, workerCount,
                           "flagcx-barex-client");

  ContextConfig cfg = XConfigUtil::DefaultContextConfig();
  for (XDevice *dev : runtime->devs) {
    XContext *sctx = nullptr, *cctx = nullptr;
    if (XContext::NewInstance(sctx, cfg, new BarexNetCallback(runtime), dev,
                              runtime->mempool, runtime->tpServer) !=
            accl::barex::BAREX_SUCCESS ||
        XContext::NewInstance(cctx, cfg, new BarexNetCallback(runtime), dev,
                              runtime->mempool, runtime->tpClient) !=
            accl::barex::BAREX_SUCCESS) {
      WARN("NET/BAREX : XContext create failed on %s", dev->GetName().c_str());
      return flagcxInternalError;
    }
    sctx->Start();
    cctx->Start();
    runtime->serverCtxs.push_back(sctx);
    runtime->clientCtxs.push_back(cctx);
  }

  /* Bind one listener and one connector to each device.  listen(dev) and
     connect(dev) must not silently create a channel on another HCA after the
     topology layer has selected the NIC closest to the GPU. */
  const int base = 19000 + (int)(getpid() % 4096);
  runtime->listeners.resize(runtime->devs.size(), nullptr);
  runtime->barexPorts.resize(runtime->devs.size(), 0);
  for (size_t d = 0; d < runtime->devs.size(); d++) {
    std::vector<XContext *> oneServer = {runtime->serverCtxs[d]};
    for (int attempt = 0; attempt < 32 && runtime->listeners[d] == nullptr;
         attempt++) {
      const int port = base + (int)d * 96 + attempt * 3;
      XListener *lis = nullptr;
      if (XListener::NewInstance(lis, 2, port, accl::barex::TIMER_3S,
                                 oneServer) == accl::barex::BAREX_SUCCESS &&
          lis->Listen() == accl::barex::BAREX_SUCCESS) {
        runtime->listeners[d] = lis;
        runtime->barexPorts[d] = port;
        break;
      }
      if (lis != nullptr) {
        lis->Shutdown();
        lis->WaitStop();
        delete lis;
      }
    }
    if (runtime->listeners[d] == nullptr) {
      WARN("NET/BAREX : no free data port for dev %zu", d);
      return flagcxInternalError;
    }
  }

  /* one connector per client context so connect() can honor `dev` */
  for (XContext *ctx : runtime->clientCtxs) {
    XConnector *con = nullptr;
    std::vector<XContext *> one = {ctx};
    if (XConnector::NewInstance(con, 2, accl::barex::TIMER_3S, one) !=
        accl::barex::BAREX_SUCCESS) {
      WARN("NET/BAREX : XConnector create failed");
      return flagcxInternalError;
    }
    runtime->connectors.push_back(con);
  }

  /* OOB IP for handles: same interface bootstrap uses */
  bootstrapNetInit();
  union flagcxSocketAddress *ifAddr = bootstrapGetNetIfAddr();
  if (ifAddr == nullptr) {
    WARN("NET/BAREX : no OOB interface");
    return flagcxInternalError;
  }
  memcpy(&runtime->ifAddr, ifAddr, sizeof(runtime->ifAddr));

  runtime->started = true;
  INFO(FLAGCX_INIT | FLAGCX_NET,
       "NET/BAREX : runtime up with %zu device-bound endpoints",
       runtime->devs.size());
  *out = runtime;
  return flagcxSuccess;
}

static flagcxResult_t barexInit() {
  if (flagcxParamIbDisable()) {
    INFO(FLAGCX_INIT | FLAGCX_NET, "NET/BAREX : disabled by FLAGCX_IB_DISABLE");
    return flagcxInternalError; /* registry falls through to next slot */
  }
  return barexProbeDevices(nullptr);
}

static flagcxResult_t barexDevices(int *ndev) {
  return barexProbeDevices(ndev);
}

static flagcxResult_t barexGetProperties(int dev, void *props) {
  if (props == nullptr || dev < 0 || dev >= kMaxNics)
    return flagcxInvalidArgument;
  auto *p = static_cast<flagcxNetProperties_v1_t *>(props);
  memset(p, 0, sizeof(*p));

  static char devName[kMaxNics][64];
  static char pciPath[kMaxNics][PATH_MAX];
  static bool propertiesReady[kMaxNics] = {};
  static std::mutex propertiesMu;
  XDeviceManager *mgr = nullptr;
  if (XDeviceManager::Singleton(mgr) != accl::barex::BAREX_SUCCESS ||
      mgr == nullptr)
    return flagcxInternalError;
  std::vector<XDevice *> devs = mgr->AllDevices();
  if (dev >= (int)devs.size())
    return flagcxInvalidArgument;

  std::lock_guard<std::mutex> lk(propertiesMu);
  if (!propertiesReady[dev]) {
    snprintf(devName[dev], sizeof(devName[dev]), "%s",
             devs[dev]->GetName().c_str());
    char sysfsPath[PATH_MAX];
    snprintf(sysfsPath, sizeof(sysfsPath), "/sys/class/infiniband/%s/device",
             devName[dev]);
    char *resolved = realpath(sysfsPath, nullptr);
    if (resolved == nullptr) {
      WARN("NET/BAREX : cannot resolve PCI path for %s (%s)", devName[dev],
           sysfsPath);
      return flagcxSystemError;
    }
    snprintf(pciPath[dev], sizeof(pciPath[dev]), "%s", resolved);
    free(resolved);
    propertiesReady[dev] = true;
  }
  p->name = devName[dev];
  p->pciPath = pciPath[dev];
  p->guid = (uint64_t)dev;
  p->ptrSupport = FLAGCX_PTR_HOST | FLAGCX_PTR_CUDA;
  p->regIsGlobal = 1; /* MRs live in the runtime-wide mempool */
  p->speed = (int)flagcxParamBarexSpeed();
  p->port = 1;
  p->latency = 0;
  p->maxComms = 65536;
  p->maxRecvs = 1; /* proxy path always posts irecv(n=1) */
  p->netDeviceType = FLAGCX_NET_DEVICE_HOST;
  p->netDeviceVersion = FLAGCX_NET_DEVICE_INVALID_VERSION;
  return flagcxSuccess;
}

static void
barexCloseConnectorChannels(BarexRuntime *runtime, int connectorDev,
                            const std::vector<XChannel *> &channels) {
  if (runtime == nullptr || connectorDev < 0 ||
      connectorDev >= static_cast<int>(runtime->connectors.size()))
    return;
  for (XChannel *channel : channels) {
    if (channel == nullptr)
      continue;
    BarexResult result = runtime->connectors[connectorDev]->CloseChannel(
        channel, [channel](Status status) {
          if (!status.IsOk())
            WARN("NET/BAREX : CloseChannel failed: %s",
                 status.ErrMsg().c_str());
          channel->Destroy();
        });
    // Match the existing BAREX lifetime contract: a synchronous failure does
    // not authorize Destroy(), because CloseChannel may still be cleaning the
    // channel and no completion callback is guaranteed.
    if (result != accl::barex::BAREX_SUCCESS)
      WARN("NET/BAREX : CloseChannel sync error: %s", bxstr(result));
  }
}

static void
barexCleanupAbandonedConnect(const std::shared_ptr<BarexConnectState> &state) {
  if (state == nullptr || !state->abandoned.load(std::memory_order_acquire) ||
      state->connectDone.load(std::memory_order_acquire) <
          state->connectStarted ||
      state->helloDone.load(std::memory_order_acquire) < state->helloStarted)
    return;

  bool expected = false;
  if (!state->cleanupStarted.compare_exchange_strong(expected, true,
                                                     std::memory_order_acq_rel))
    return;

  if (state->comm != nullptr) {
    state->comm->state.store(BAREX_COMM_CLOSING, std::memory_order_release);
    if (state->runtime != nullptr) {
      std::lock_guard<std::mutex> lk(state->runtime->mu);
      for (XChannel *channel : state->comm->channels)
        state->runtime->channelComm.erase(channel);
    }
    state->comm->waitCallbacks();
    barexCloseConnectorChannels(state->runtime, state->connectorDev,
                                state->comm->channels);
    delete state->comm;
    state->comm = nullptr;
  } else {
    barexCloseConnectorChannels(state->runtime, state->connectorDev,
                                state->channels);
  }
  state->channels.clear();
}

static flagcxResult_t barexListen(int dev, void *opaqueHandle,
                                  void **listenComm) {
  if (opaqueHandle == nullptr || listenComm == nullptr)
    return flagcxInvalidArgument;
  *listenComm = nullptr;
  BarexRuntime *e = nullptr;
  FLAGCXCHECK(barexRuntimeStart(&e));
  if (dev < 0 || dev >= (int)e->listeners.size() ||
      e->listeners[dev] == nullptr || e->barexPorts[dev] <= 0)
    return flagcxInvalidArgument;

  auto *handle = static_cast<BarexNetHandle *>(opaqueHandle);
  memset(handle, 0, sizeof(*handle));
  memcpy(&handle->connectAddr, &e->ifAddr, sizeof(handle->connectAddr));
  addrSetPort(&handle->connectAddr, e->barexPorts[dev]);

  auto *lc = new BarexListenComm();
  lc->runtime = e;
  lc->dev = dev;
  lc->channels = barexConnectionChannelCount();
  {
    std::lock_guard<std::mutex> lk(e->mu);
    do {
      lc->commId = e->rng();
    } while (lc->commId == 0 || e->pendingAccepts.count(lc->commId) != 0);
    PendingAccept pending;
    pending.dev = dev;
    pending.nicId = e->devs[dev]->GetId();
    pending.expectedChannels = lc->channels;
    pending.channels.resize(pending.expectedChannels, nullptr);
    e->pendingAccepts[lc->commId] = std::move(pending);
  }
  handle->commId = lc->commId;
  handle->state = BAREX_CONN_INIT;
  FLAGCXCHECK(flagcxBarexRuntimeEncodeListenGeometry(
      static_cast<uint32_t>(dev), lc->channels, &handle->listenGeometry));
  handle->connectState = nullptr;

  *listenComm = lc;
  return flagcxSuccess;
}

/* Non-blocking, resumable: *sendComm stays NULL until the channel is up
   and HELLO delivered; state lives in the handle across retries. */
static flagcxResult_t barexConnect(int dev, void *opaqueHandle,
                                   void **sendComm) {
  if (opaqueHandle == nullptr || sendComm == nullptr)
    return flagcxInvalidArgument;
  *sendComm = nullptr;
  BarexRuntime *e = nullptr;
  FLAGCXCHECK(barexRuntimeStart(&e));
  if (dev < 0 || dev >= (int)e->connectors.size())
    return flagcxInvalidArgument;
  auto *handle = static_cast<BarexNetHandle *>(opaqueHandle);
  uint32_t remoteChannelCount = 0;
  uint32_t remoteDev = 0;
  FLAGCXCHECK(flagcxBarexRuntimeDecodeListenGeometry(
      handle->listenGeometry, &remoteDev, &remoteChannelCount));
  const uint32_t localChannelCount = barexConnectionChannelCount();
  if (remoteDev >= kMaxNics ||
      flagcxBarexRuntimeValidateChannelCount(remoteChannelCount) !=
          flagcxSuccess ||
      localChannelCount != remoteChannelCount) {
    WARN("NET/BAREX : incompatible Engine geometry local channels=%u remote "
         "channels=%u remote dev=%u",
         localChannelCount, remoteChannelCount, remoteDev);
    return flagcxRemoteError;
  }

  if (handle->state == BAREX_CONN_INIT) {
    auto st = std::make_shared<BarexConnectState>();
    st->runtime = e;
    st->connectorDev = dev;
    st->channelCount = localChannelCount;
    st->channels.resize(localChannelCount, nullptr);
    handle->connectState = new std::shared_ptr<BarexConnectState>(st);

    char ip[64] = {0};
    if (handle->connectAddr.sa.sa_family == AF_INET) {
      inet_ntop(AF_INET, &handle->connectAddr.sin.sin_addr, ip, sizeof(ip));
    } else {
      inet_ntop(AF_INET6, &handle->connectAddr.sin6.sin6_addr, ip, sizeof(ip));
    }
    const int port = ntohs(handle->connectAddr.sa.sa_family == AF_INET
                               ? handle->connectAddr.sin.sin_port
                               : handle->connectAddr.sin6.sin6_port);

    XConnector *con = e->connectors[dev];
    for (uint32_t lane = 0; lane < localChannelCount; ++lane) {
      st->connectStarted++;
      BarexResult r = con->Connect(
          std::string(ip), port, [st, lane](XChannel *ch, Status s) {
            if (s.IsOk() && ch != nullptr)
              st->channels[lane] = ch;
            else
              st->connectFailed.store(true, std::memory_order_release);
            st->connectDone.fetch_add(1, std::memory_order_release);
            barexCleanupAbandonedConnect(st);
          });
      if (r != accl::barex::BAREX_SUCCESS) {
        WARN("NET/BAREX : Connect(%s:%d) lane %u sync error: %s", ip, port,
             lane, bxstr(r));
        st->connectStarted--;
        st->connectFailed.store(true, std::memory_order_release);
        break;
      }
    }
    handle->state = BAREX_CONN_CONNECTING;
    return flagcxSuccess; /* in progress */
  }

  auto *holder =
      static_cast<std::shared_ptr<BarexConnectState> *>(handle->connectState);
  if (holder == nullptr || *holder == nullptr)
    return flagcxInternalError;
  std::shared_ptr<BarexConnectState> st = *holder;

  if (handle->state == BAREX_CONN_CONNECTING) {
    if (st->connectDone.load(std::memory_order_acquire) < st->connectStarted)
      return flagcxSuccess;
    if (st->connectFailed.load(std::memory_order_acquire) ||
        st->connectStarted != st->channelCount) {
      WARN("NET/BAREX : channel connect failed (commId 0x%llx)",
           (unsigned long long)handle->commId);
      st->abandoned.store(true, std::memory_order_release);
      barexCleanupAbandonedConnect(st);
      delete holder;
      handle->connectState = nullptr;
      return flagcxInternalError;
    }

    auto *comm = new BarexComm();
    comm->runtime = e;
    comm->channels = st->channels;
    comm->channel = comm->channels.front();
    comm->commId = handle->commId;
    comm->isSend = true;
    comm->connectorDev = dev;
    st->comm = comm;
    {
      std::lock_guard<std::mutex> lk(e->mu);
      for (XChannel *channel : comm->channels)
        e->channelComm[channel] = comm;
    }

    for (uint32_t lane = 0; lane < st->channelCount; ++lane) {
      XChannel *ch = comm->channels[lane];
      memp_t msg;
      if (e->mempool->AllocBuffer(msg, sizeof(BarexHelloMsg), accl::barex::CPU,
                                  ch->GetLocalNicId(),
                                  0) != accl::barex::BAREX_SUCCESS) {
        WARN("NET/BAREX : HELLO buffer alloc failed for lane %u", lane);
        st->helloFailed.store(true, std::memory_order_release);
        break;
      }
      BarexHelloMsg hello = {};
      hello.type = BAREX_MSG_HELLO;
      hello.commId = handle->commId;
      if (flagcxBarexRuntimeEncodeHelloGeometry(st->channelCount, lane,
                                                &hello.pad) != flagcxSuccess) {
        e->mempool->ReleaseBuffer(msg.buf, accl::barex::CPU);
        st->helloFailed.store(true, std::memory_order_release);
        break;
      }
      memcpy(msg.buf, &hello, sizeof(hello));
      msg.buf_len = sizeof(hello);
      x_msg_header hdr = {};
      st->helloStarted++;
      BarexResult r = ch->Send(
          msg, /*auto_release=*/true, hdr,
          [st](Status s) {
            if (!s.IsOk())
              st->helloFailed.store(true, std::memory_order_release);
            st->helloDone.fetch_add(1, std::memory_order_release);
            barexCleanupAbandonedConnect(st);
          },
          true);
      if (r != accl::barex::BAREX_SUCCESS) {
        WARN("NET/BAREX : HELLO lane %u send sync error: %s", lane, bxstr(r));
        e->mempool->ReleaseBuffer(msg.buf, accl::barex::CPU);
        st->helloStarted--;
        st->helloFailed.store(true, std::memory_order_release);
        break;
      }
    }
    handle->state = BAREX_CONN_HELLO;
    return flagcxSuccess; /* in progress */
  }

  if (st->helloDone.load(std::memory_order_acquire) < st->helloStarted)
    return flagcxSuccess;
  if (st->helloFailed.load(std::memory_order_acquire)) {
    WARN("NET/BAREX : HELLO delivery failed");
    st->abandoned.store(true, std::memory_order_release);
    barexCleanupAbandonedConnect(st);
    delete holder;
    handle->connectState = nullptr;
    return flagcxInternalError;
  }
  *sendComm = st->comm;
  st->comm = nullptr;
  delete holder;
  handle->connectState = nullptr;
  INFO(FLAGCX_NET, "NET/BAREX : sendComm up (commId 0x%llx)",
       (unsigned long long)handle->commId);
  return flagcxSuccess;
}

static flagcxResult_t barexResetConnect(void *opaqueHandle) {
  if (opaqueHandle == nullptr)
    return flagcxInvalidArgument;
  auto *handle = static_cast<BarexNetHandle *>(opaqueHandle);
  auto *holder =
      static_cast<std::shared_ptr<BarexConnectState> *>(handle->connectState);
  if (holder == nullptr) {
    handle->state = BAREX_CONN_INIT;
    return flagcxSuccess;
  }

  std::shared_ptr<BarexConnectState> state = *holder;
  handle->connectState = nullptr;
  handle->state = BAREX_CONN_INIT;
  delete holder;
  if (state != nullptr) {
    state->abandoned.store(true, std::memory_order_release);
    barexCleanupAbandonedConnect(state);
  }
  return flagcxSuccess;
}

/* Non-blocking: ready once the connector's HELLO bound a channel. */
static flagcxResult_t barexAccept(void *listenComm, void **recvComm) {
  if (recvComm == nullptr)
    return flagcxInvalidArgument;
  *recvComm = nullptr;
  auto *lc = static_cast<BarexListenComm *>(listenComm);
  if (lc == nullptr)
    return flagcxInternalError;
  BarexRuntime *e = lc->runtime;

  BarexComm *comm = nullptr;
  {
    std::lock_guard<std::mutex> lk(e->mu);
    auto it = e->pendingAccepts.find(lc->commId);
    if (it == e->pendingAccepts.end())
      return flagcxInternalError;
    if (it->second.firstError != flagcxSuccess)
      return it->second.firstError;
    if (!std::all_of(it->second.channels.begin(), it->second.channels.end(),
                     [](XChannel *channel) { return channel != nullptr; }))
      return flagcxSuccess; /* complete lane set not here yet */

    comm = new BarexComm();
    comm->runtime = e;
    comm->channels = it->second.channels;
    comm->channel = comm->channels.front();
    comm->commId = lc->commId;
    comm->isSend = false;
    for (XChannel *channel : comm->channels)
      e->channelComm[channel] = comm;
    // A listener can accept more than one connection. Keep the demux entry
    // registered while returning this complete lane set to the caller.
    std::fill(it->second.channels.begin(), it->second.channels.end(), nullptr);
  }
  *recvComm = comm;
  INFO(FLAGCX_NET, "NET/BAREX : recvComm up (commId 0x%llx)",
       (unsigned long long)lc->commId);
  return flagcxSuccess;
}

static flagcxResult_t barexCloseComm(BarexComm *comm) {
  if (comm == nullptr)
    return flagcxSuccess;
  BarexRuntime *e = comm->runtime;
  comm->state.store(BAREX_COMM_CLOSING, std::memory_order_release);
  if (e != nullptr && !comm->channels.empty()) {
    {
      std::lock_guard<std::mutex> lk(e->mu);
      for (XChannel *channel : comm->channels)
        e->channelComm.erase(channel);
    }
    if (comm->isSend && comm->connectorDev >= 0 &&
        comm->connectorDev < (int)e->connectors.size())
      barexCloseConnectorChannels(e, comm->connectorDev, comm->channels);
  }
  /* Completion/receive callbacks retain the comm explicitly.  The channel
     close callback does not reference it, so a failed asynchronous close
     cannot keep closeSend() blocked or cause a use-after-free. */
  comm->waitCallbacks();
  comm->channel = nullptr;
  comm->channels.clear();
  comm->state.store(BAREX_COMM_CLOSED, std::memory_order_release);
  delete comm;
  return flagcxSuccess;
}

static flagcxResult_t barexCloseSend(void *sendComm) {
  return barexCloseComm(static_cast<BarexComm *>(sendComm));
}

static flagcxResult_t barexCloseRecv(void *recvComm) {
  return barexCloseComm(static_cast<BarexComm *>(recvComm));
}

static flagcxResult_t barexCloseListen(void *listenComm) {
  auto *lc = static_cast<BarexListenComm *>(listenComm);
  if (lc == nullptr)
    return flagcxSuccess;
  std::vector<XChannel *> pendingChannels;
  {
    std::lock_guard<std::mutex> lk(lc->runtime->mu);
    auto it = lc->runtime->pendingAccepts.find(lc->commId);
    if (it != lc->runtime->pendingAccepts.end()) {
      pendingChannels.swap(it->second.channels);
      lc->runtime->pendingAccepts.erase(it);
    }
  }
  for (XChannel *channel : pendingChannels)
    if (channel != nullptr)
      channel->Destroy();
  delete lc;
  return flagcxSuccess;
}

static flagcxResult_t barexRegMr(void *comm, void *data, size_t size, int type,
                                 int mrFlags, void **mhandle) {
  (void)comm;
  if (mhandle == nullptr)
    return flagcxInvalidArgument;
  *mhandle = nullptr;
  if (data == nullptr || size == 0 ||
      (type != FLAGCX_PTR_HOST && type != FLAGCX_PTR_CUDA))
    return flagcxInvalidArgument;
  const int64_t maxMrBytes = flagcxParamBarexMaxMrBytes();
  if (type == FLAGCX_PTR_CUDA && maxMrBytes > 0 &&
      size > static_cast<uint64_t>(maxMrBytes)) {
    // flagcxNetMrInfo carries one key per NIC, not one key per address chunk.
    // Until the common MR contract can describe chunked registrations, reject
    // windows larger than the vsolar single-MR limit instead of advertising
    // keys that cover only a prefix of the allocation.
    WARN("NET/BAREX : GPU MR %p size %zu exceeds single-registration limit "
         "%lld (FLAGCX_BAREX_MAX_MR_BYTES)",
         data, size, (long long)maxMrBytes);
    return flagcxNotSupported;
  }
  BarexRuntime *e = nullptr;
  FLAGCXCHECK(barexRuntimeStart(&e));

  // Make forward progress on handles explicitly relinquished by the generic
  // Engine. A direct adaptor caller that received a deregistration error still
  // owns its handle, so it is never eligible for this automatic retry.
  (void)flagcxBarexRuntimeDrainDeferredMrs();

  const uintptr_t base = (uintptr_t)data;
  if (size > std::numeric_limits<uintptr_t>::max() - base)
    return flagcxInvalidArgument;
  const uintptr_t end = base + size;
  const accl::barex::device_type dtype =
      (type == FLAGCX_PTR_CUDA) ? accl::barex::GPU : accl::barex::CPU;
  int devId = 0;
  if (dtype == accl::barex::GPU && deviceAdaptor != nullptr &&
      deviceAdaptor->getDevice != nullptr) {
    FLAGCXCHECK(deviceAdaptor->getDevice(&devId));
  }

  /* RegUserMr/DeregUserMr use (base, dtype) as identity. Keep lookup and
     physical registration serialized so duplicate registrations cannot race. */
  std::unique_lock<std::mutex> lk(e->mu);
  for (const auto &entry : e->mrByBase) {
    BarexMr *existing = entry.second;
    if (existing == nullptr)
      continue;
    const uintptr_t existingEnd = existing->base + existing->size;
    if (!existing->reusable) {
      if (base < existingEnd && existing->base < end)
        return flagcxInternalError;
      continue;
    }
    const bool compatible = existing->dtype == dtype &&
                            existing->devId == devId &&
                            existing->mrFlags == mrFlags;
    if (base >= existing->base && end <= existingEnd) {
      if (!compatible)
        return flagcxInvalidArgument;
      existing->refCount++;
      *mhandle = existing;
      return flagcxSuccess;
    }
    if (base < existingEnd && existing->base < end) {
      WARN("NET/BAREX : partially overlapping MR [%p,%p) conflicts with "
           "existing [%p,%p)",
           data, (void *)end, (void *)existing->base, (void *)existingEnd);
      return flagcxInvalidArgument;
    }
  }

  auto *mr = new BarexMr();
  BarexResult r = e->mempool->RegUserMr(mr->mem, data, size, dtype, devId);
  if (r != accl::barex::BAREX_SUCCESS) {
    WARN("NET/BAREX : RegUserMr(%p,%zu,%s,dev%d) failed: %s", data, size,
         dtype == accl::barex::GPU ? "GPU" : "CPU", devId, bxstr(r));
    delete mr;
    return barexResult(r);
  }
  mr->base = base;
  mr->size = size;
  mr->dtype = dtype;
  mr->devId = devId;
  mr->mrFlags = mrFlags;
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
    BarexResult deregResult = e->mempool->DeregUserMr(data, dtype);
    if (deregResult == accl::barex::BAREX_SUCCESS) {
      delete mr;
    } else {
      // No handle was published to the caller. Retain provider ownership in
      // the adaptor so a later registration or Engine teardown can retry.
      mr->reusable = false;
      mr->deferred = true;
      e->mrByBase[base] = mr;
    }
    return flagcxInternalError;
  }
  e->mrByBase[base] = mr;
  *mhandle = mr;
  return flagcxSuccess;
}

static flagcxResult_t barexDeregMr(void *comm, void *mhandle) {
  (void)comm;
  auto *mr = static_cast<BarexMr *>(mhandle);
  if (mr == nullptr)
    return flagcxSuccess;
  BarexRuntime *e = gBarexRuntime;
  if (e == nullptr)
    return flagcxInternalError;
  std::lock_guard<std::mutex> lk(e->mu);
  auto it = e->mrByBase.find(mr->base);
  if (it == e->mrByBase.end() || it->second != mr || mr->refCount <= 0)
    return flagcxInvalidArgument;
  if (mr->refCount > 1) {
    mr->refCount--;
    return flagcxSuccess;
  }
  BarexResult result = e->mempool->DeregUserMr((void *)mr->base, mr->dtype);
  if (result != accl::barex::BAREX_SUCCESS) {
    /* A failed deregistration does not consume the handle, so a caller that
       retains it can retry. Mark the entry non-reusable immediately: common
       teardown paths may discard the handle, and future allocations must
       never inherit keys from that stale physical registration. */
    mr->reusable = false;
    WARN("NET/BAREX : DeregUserMr(%p,%s) failed: %s", (void *)mr->base,
         mr->dtype == accl::barex::GPU ? "GPU" : "CPU", bxstr(result));
    return barexResult(result);
  }
  e->mrByBase.erase(it);
  delete mr;
  return flagcxSuccess;
}

static flagcxResult_t barexDrainDeferredMrs(BarexRuntime *e) {
  if (e == nullptr)
    return flagcxSuccess;

  std::lock_guard<std::mutex> lk(e->mu);
  flagcxResult_t firstError = flagcxSuccess;
  for (auto it = e->mrByBase.begin(); it != e->mrByBase.end();) {
    BarexMr *mr = it->second;
    if (mr == nullptr || !mr->deferred) {
      ++it;
      continue;
    }
    if (mr->refCount != 1) {
      if (firstError == flagcxSuccess)
        firstError = flagcxInternalError;
      ++it;
      continue;
    }

    BarexResult result =
        e->mempool->DeregUserMr(reinterpret_cast<void *>(mr->base), mr->dtype);
    if (result != accl::barex::BAREX_SUCCESS) {
      if (firstError == flagcxSuccess)
        firstError = barexResult(result);
      ++it;
      continue;
    }
    delete mr;
    it = e->mrByBase.erase(it);
  }
  return firstError;
}

static flagcxResult_t barexDeferMr(BarexRuntime *e, void *mhandle) {
  auto *mr = static_cast<BarexMr *>(mhandle);
  if (e == nullptr || mr == nullptr)
    return flagcxInvalidArgument;

  std::lock_guard<std::mutex> lk(e->mu);
  auto it = e->mrByBase.find(mr->base);
  if (it == e->mrByBase.end() || it->second != mr || mr->refCount != 1 ||
      mr->reusable)
    return flagcxInvalidArgument;
  mr->deferred = true;
  return flagcxSuccess;
}

static flagcxResult_t barexGetMrInfo(void *mhandle,
                                     struct flagcxNetMrInfo *info) {
  if (mhandle == nullptr || info == nullptr)
    return flagcxInvalidArgument;
  auto *mr = static_cast<BarexMr *>(mhandle);
  if (!mr->reusable || mr->nKeys == 0 || mr->nKeys > FLAGCX_NET_MAX_MR_KEYS)
    return flagcxInternalError;
  memset(info, 0, sizeof(*info));
  info->nKeys = mr->nKeys;
  memcpy(info->lkeys, mr->lkeys, mr->nKeys * sizeof(uint32_t));
  memcpy(info->rkeys, mr->rkeys, mr->nKeys * sizeof(uint32_t));
  return flagcxSuccess;
}

static flagcxResult_t barexIsend(void *sendComm, void *data, size_t size,
                                 int tag, void *mhandle, void *phandle,
                                 void **request) {
  (void)tag; /* always 0 on the proxy path */
  (void)phandle;
  *request = nullptr;
  auto *comm = static_cast<BarexComm *>(sendComm);
  auto *mr = static_cast<BarexMr *>(mhandle);
  if (comm == nullptr || mr == nullptr)
    return flagcxInternalError;
  flagcxResult_t submissionStatus = comm->submissionStatus();
  if (submissionStatus != flagcxSuccess)
    return submissionStatus;

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
  req->twoSided = true;

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

  if (!comm->beginCallback()) {
    barexReleaseRequest(req);
    return comm->submissionStatus();
  }
  BarexRequest *reqCapture = req;
  BarexResult r = comm->channel->WriteSingle(
      payload, cts.addr, rkey, /*signal_peer=*/true,
      /*imm_data=*/cts.slot & kImmSlotMask,
      [reqCapture](Status s) { barexCompleteCallback(reqCapture, s); },
      /*done_inline=*/true, UINT64_MAX);
  if (r != accl::barex::BAREX_SUCCESS) {
    WARN("NET/BAREX : WriteSingle sync error: %s", bxstr(r));
    const flagcxResult_t mapped = barexResult(r);
    comm->fail(mapped);
    comm->endCallback();
    barexReleaseRequest(req);
    return mapped;
  }
  *request = req;
  return flagcxSuccess;
}

static flagcxResult_t barexIrecv(void *recvComm, int n, void **data,
                                 size_t *sizes, int *tags, void **mhandles,
                                 void **phandles, void **request) {
  (void)tags;
  (void)phandles;
  *request = nullptr;
  auto *comm = static_cast<BarexComm *>(recvComm);
  if (comm == nullptr || n != 1 || data == nullptr || sizes == nullptr ||
      mhandles == nullptr)
    return flagcxInternalError;
  auto *mr = static_cast<BarexMr *>(mhandles[0]);
  if (mr == nullptr)
    return flagcxInternalError;
  flagcxResult_t submissionStatus = comm->submissionStatus();
  if (submissionStatus != flagcxSuccess)
    return submissionStatus;
  BarexRuntime *e = comm->runtime;

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
  req->twoSided = true;

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

  if (!comm->beginCallback()) {
    e->mempool->ReleaseBuffer(msg.buf, accl::barex::CPU);
    barexReleaseRequest(req);
    return comm->submissionStatus();
  }
  BarexRequest *reqCapture = req;
  BarexResult r = comm->channel->Send(
      msg, /*auto_release=*/true, hdr,
      [reqCapture](Status s) {
        if (!s.IsOk()) { /* CTS lost: fail the request; sender never writes */
          const flagcxResult_t result = barexStatus(s);
          WARN("NET/BAREX : CTS delivery failed: %s", s.ErrMsg().c_str());
          reqCapture->comm->fail(result);
          barexCompleteRequest(reqCapture, result);
        }
        reqCapture->comm->endCallback();
      },
      true);
  if (r != accl::barex::BAREX_SUCCESS) {
    WARN("NET/BAREX : CTS send sync error: %s", bxstr(r));
    const flagcxResult_t mapped = barexResult(r);
    comm->fail(mapped);
    /* per xchannel.h: on send failure the buffer is NOT auto-released */
    e->mempool->ReleaseBuffer(msg.buf, accl::barex::CPU);
    comm->endCallback();
    barexReleaseRequest(req);
    return mapped;
  }
  /* completion arrives via OnImmRecvCall(imm == req->slot) */
  *request = req;
  return flagcxSuccess;
}

/* Transitional PPU compatibility behavior.

   BAREX write-with-imm orders the payload before the completing immediate, so
   this sentinel preserves the existing transport-completion pipeline and PPU
   CI behavior. That ordering does NOT by itself prove that NIC-written memory
   is visible to a subsequent PPU kernel. Consequently flagcxNetBarex advertises
   no GDR flush capability, and a forced visibility requirement must be rejected
   before reaching this callback.

   TODO: replace this no-op with an ACCL/PPU visibility primitive or an ACCL-
   owned self-loopback RDMA READ, then advertise capability only after an
   RDMA-completion -> immediate-kernel-consumer hardware test passes. */
static flagcxResult_t barexIflush(void *recvComm, int n, void **data,
                                  int *sizes, void **mhandles, void **request) {
  (void)recvComm;
  (void)n;
  (void)data;
  (void)sizes;
  (void)mhandles;
  *request = (void *)0x1;
  return flagcxSuccess;
}

static flagcxResult_t barexTest(void *request, int *done, int *sizes) {
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
  if (st == BAREX_REQ_PENDING || st == BAREX_REQ_COMPLETING)
    return flagcxSuccess;
  if (st != BAREX_REQ_DONE && st != BAREX_REQ_ERROR)
    return flagcxInternalError;
  const flagcxResult_t result =
      static_cast<flagcxResult_t>(req->result.load(std::memory_order_relaxed));
  // The legacy collective proxy ignores test()'s return value and advances
  // only on done. Keep a failed two-sided request incomplete so it cannot be
  // mistaken for successfully transferred data. Generic proxy async-error
  // propagation belongs to the dedicated proxy-error refactor.
  if (st == BAREX_REQ_ERROR && req->twoSided)
    return result;
  *done = 1;
  if (sizes != nullptr)
    sizes[0] = (int)req->size;
  barexReleaseRequest(req);
  return result;
}

static flagcxResult_t barexPrepareOneSided(
    BarexComm *comm, XChannel *channel,
    const struct flagcxOneSideHandleInfo *localInfo, int localRank,
    uint64_t localOffset, const struct flagcxOneSideHandleInfo *remoteInfo,
    int remoteRank, uint64_t remoteOffset, size_t size, memp_t *localMem,
    uint64_t *remoteAddress, uint32_t *remoteRkey) {
  if (comm == nullptr || channel == nullptr || localInfo == nullptr ||
      remoteInfo == nullptr || localMem == nullptr ||
      remoteAddress == nullptr || remoteRkey == nullptr ||
      localInfo->localMrHandle == nullptr)
    return flagcxInvalidArgument;
  struct flagcxNetResolvedRange localRange = {};
  struct flagcxNetResolvedRange remoteRange = {};
  FLAGCXCHECK(flagcxNetResolveOneSideRange(localInfo, localRank, localOffset,
                                           size, &localRange));
  FLAGCXCHECK(flagcxNetResolveOneSideRange(remoteInfo, remoteRank, remoteOffset,
                                           size, &remoteRange));
  if (size > std::numeric_limits<uint32_t>::max())
    return flagcxInvalidArgument;
  flagcxResult_t submissionStatus = comm->submissionStatus();
  if (submissionStatus != flagcxSuccess)
    return submissionStatus;
  const int localNic = channel->GetLocalNicId();
  const int peerNic = channel->GetPeerNicId();
  const struct flagcxNetMrInfo &remoteMrInfo = *remoteRange.mrInfo;
  if (localNic < 0 || localNic >= kMaxNics || peerNic < 0 ||
      peerNic >= kMaxNics || (uint32_t)peerNic >= remoteMrInfo.nKeys ||
      remoteMrInfo.rkeys[peerNic] == 0)
    return flagcxInvalidArgument;

  auto *mr = static_cast<BarexMr *>(localInfo->localMrHandle);
  const uintptr_t localAddress = localRange.address;
  if (localAddress < mr->base || localAddress - mr->base > mr->size ||
      size > mr->size - (localAddress - mr->base))
    return flagcxInvalidArgument;
  auto mrIt = mr->mem.mrs.find(localNic);
  if (mrIt == mr->mem.mrs.end() || mrIt->second == nullptr)
    return flagcxInvalidArgument;
  *localMem = mr->mem;
  localMem->buf = reinterpret_cast<char *>(localAddress);
  localMem->buf_len = size;
  localMem->mr = mrIt->second;
  *remoteAddress = remoteRange.address;
  *remoteRkey = remoteMrInfo.rkeys[peerNic];
  return flagcxSuccess;
}

static flagcxResult_t barexSelectOneSidedChannel(BarexComm *comm,
                                                 XChannel **channel,
                                                 uint32_t *lane) {
  if (comm == nullptr || channel == nullptr || lane == nullptr ||
      comm->channels.empty())
    return flagcxInvalidArgument;
  struct flagcxNetSubmitContext context = {};
  uint64_t orderingKey = 0;
  if (flagcxNetGetSubmitContext(&context) == flagcxSuccess &&
      (context.flags & FLAGCX_NET_SUBMIT_INDEPENDENT) != 0)
    orderingKey = context.orderingKey;
  FLAGCXCHECK(flagcxBarexRuntimeSelectLane(
      static_cast<uint32_t>(comm->channels.size()), orderingKey, lane));
  *channel = comm->channels[*lane];
  return *channel == nullptr ? flagcxInternalError : flagcxSuccess;
}

static void barexRecordOneSidedLane(uint32_t lane) {
  if (lane >= 64)
    return;
  struct flagcxNetSubmitContext context = {};
  if (flagcxNetGetSubmitContext(&context) == flagcxSuccess &&
      context.laneMask != nullptr)
    __atomic_fetch_or(context.laneMask, 1ULL << lane, __ATOMIC_RELAXED);
}

static rw_memp_t barexMakeRw(const memp_t &localMem, uint64_t remoteAddress,
                             uint32_t remoteRkey, size_t size) {
  rw_memp_t rw;
  rw.data = localMem;
  rw.r_addr = remoteAddress;
  rw.r_key = remoteRkey;
  rw.r_ttl_ms = UINT64_MAX;
  rw.sg.addr = reinterpret_cast<uint64_t>(localMem.buf);
  rw.sg.length = static_cast<uint32_t>(size);
  rw.sg.lkey = localMem.mr->lkey;
  return rw;
}

static flagcxResult_t barexIput(void *sendComm, uint64_t srcOff,
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
  uint64_t remoteAddress = 0;
  uint32_t remoteRkey = 0;
  XChannel *channel = nullptr;
  uint32_t lane = 0;
  FLAGCXCHECK(barexSelectOneSidedChannel(comm, &channel, &lane));
  FLAGCXCHECK(barexPrepareOneSided(comm, channel, srcInfo, srcRank, srcOff,
                                   dstInfo, dstRank, dstOff, size, &localMem,
                                   &remoteAddress, &remoteRkey));

  BarexRequest *req = comm->allocRequest();
  if (req == nullptr)
    return flagcxInProgress;
  req->size = size;
  if (size == 0) {
    barexCompleteRequest(req, flagcxSuccess);
    *request = req;
    return flagcxSuccess;
  }
  if (!comm->beginCallback()) {
    barexReleaseRequest(req);
    return comm->submissionStatus();
  }
  BarexResult result = channel->WriteSingle(
      localMem, remoteAddress, remoteRkey, /*signal_peer=*/false, 0,
      [req](Status status) { barexCompleteCallback(req, status); },
      /*done_inline=*/true, UINT64_MAX);
  if (result != accl::barex::BAREX_SUCCESS) {
    const flagcxResult_t mapped = barexResult(result);
    comm->fail(mapped);
    comm->endCallback();
    barexReleaseRequest(req);
    return mapped;
  }
  barexRecordOneSidedLane(lane);
  *request = req;
  return flagcxSuccess;
}

static flagcxResult_t barexIget(void *sendComm, uint64_t srcOff,
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
  uint64_t remoteAddress = 0;
  uint32_t remoteRkey = 0;
  XChannel *channel = nullptr;
  uint32_t lane = 0;
  FLAGCXCHECK(barexSelectOneSidedChannel(comm, &channel, &lane));
  FLAGCXCHECK(barexPrepareOneSided(comm, channel, dstInfo, dstRank, dstOff,
                                   srcInfo, srcRank, srcOff, size, &localMem,
                                   &remoteAddress, &remoteRkey));

  BarexRequest *req = comm->allocRequest();
  if (req == nullptr)
    return flagcxInProgress;
  req->size = size;
  if (size == 0) {
    barexCompleteRequest(req, flagcxSuccess);
    *request = req;
    return flagcxSuccess;
  }
  if (!comm->beginCallback()) {
    barexReleaseRequest(req);
    return comm->submissionStatus();
  }
  BarexResult result = channel->ReadSingle(
      localMem, remoteAddress, remoteRkey,
      [req](Status status) { barexCompleteCallback(req, status); },
      /*done_inline=*/true, UINT64_MAX);
  if (result != accl::barex::BAREX_SUCCESS) {
    const flagcxResult_t mapped = barexResult(result);
    comm->fail(mapped);
    comm->endCallback();
    barexReleaseRequest(req);
    return mapped;
  }
  barexRecordOneSidedLane(lane);
  *request = req;
  return flagcxSuccess;
}

static flagcxResult_t
barexIputBatch(void *sendComm, int count, const uint64_t *srcOffs,
               const uint64_t *dstOffs, const size_t *sizes, int srcRank,
               int dstRank, void **srcHandles, void **dstHandles,
               void **requests, int *posted) {
  if (posted == nullptr)
    return flagcxInvalidArgument;
  *posted = 0;
  if (count < 0 || count > kMaxRequests || (count > 0 && requests == nullptr))
    return flagcxInvalidArgument;
  for (int i = 0; i < count; ++i)
    requests[i] = nullptr;
  if (count == 0)
    return flagcxSuccess;
  if (sendComm == nullptr || srcOffs == nullptr || dstOffs == nullptr ||
      sizes == nullptr || srcHandles == nullptr || dstHandles == nullptr)
    return flagcxInvalidArgument;

  auto *comm = static_cast<BarexComm *>(sendComm);
  auto *srcInfo =
      reinterpret_cast<const struct flagcxOneSideHandleInfo *>(srcHandles);
  auto *dstInfo =
      reinterpret_cast<const struct flagcxOneSideHandleInfo *>(dstHandles);
  XChannel *channel = nullptr;
  uint32_t lane = 0;
  FLAGCXCHECK(barexSelectOneSidedChannel(comm, &channel, &lane));
  auto allData = std::make_shared<std::vector<rw_memp_t>>();
  allData->reserve(count);
  for (int i = 0; i < count; ++i) {
    memp_t localMem;
    uint64_t remoteAddress = 0;
    uint32_t remoteRkey = 0;
    FLAGCXCHECK(barexPrepareOneSided(
        comm, channel, srcInfo, srcRank, srcOffs[i], dstInfo, dstRank,
        dstOffs[i], sizes[i], &localMem, &remoteAddress, &remoteRkey));
    allData->push_back(
        barexMakeRw(localMem, remoteAddress, remoteRkey, sizes[i]));
  }

  auto acceptedRequests = std::make_shared<std::vector<BarexRequest *>>();
  acceptedRequests->reserve(count);
  int accepted = 0;
  for (; accepted < count; ++accepted) {
    BarexRequest *req = comm->allocRequest();
    if (req == nullptr)
      break;
    req->size = sizes[accepted];
    acceptedRequests->push_back(req);
  }
  if (accepted == 0)
    return flagcxInProgress;

  auto postedData = std::make_shared<std::vector<rw_memp_t>>(
      allData->begin(), allData->begin() + accepted);
  if (!comm->beginCallback()) {
    for (BarexRequest *req : *acceptedRequests)
      barexReleaseRequest(req);
    return comm->submissionStatus();
  }
  BarexResult result = channel->WriteBatch(
      postedData,
      [comm, postedData, acceptedRequests](Status status) {
        const flagcxResult_t completion = barexStatus(status);
        if (completion != flagcxSuccess) {
          WARN("NET/BAREX : asynchronous batch failed: %s",
               status.ErrMsg().c_str());
          comm->fail(completion);
        }
        for (BarexRequest *req : *acceptedRequests)
          barexCompleteRequest(req, completion);
        comm->endCallback();
      },
      /*done_inline=*/true);
  if (result != accl::barex::BAREX_SUCCESS) {
    const flagcxResult_t mapped = barexResult(result);
    comm->fail(mapped);
    comm->endCallback();
    for (BarexRequest *req : *acceptedRequests)
      barexReleaseRequest(req);
    /* BAREX does not report a rejected element, so a synchronous failure is
       treated as accepting no work. */
    struct flagcxNetPostResult post = {};
    FLAGCXCHECK(flagcxNetPostResultInit(&post, count, 0, mapped));
    return post.result;
  }

  struct flagcxNetPostResult post = {};
  FLAGCXCHECK(flagcxNetPostResultInit(&post, count, accepted,
                                      accepted == count ? flagcxSuccess
                                                        : flagcxInProgress));
  for (int i = 0; i < post.accepted; ++i)
    requests[i] = (*acceptedRequests)[i];
  barexRecordOneSidedLane(lane);
  *posted = post.accepted;
  return post.result;
}

static flagcxResult_t barexTestBatch(void **requests, int nRequests,
                                     int *doneFlags, int *doneCount) {
  if (nRequests > kMaxRequests)
    return flagcxInvalidArgument;
  return flagcxNetTestBatchCommon(requests, nRequests, doneFlags, doneCount,
                                  barexTest);
}

static flagcxResult_t barexIgetBatch(void *sendComm, int count,
                                     const uint64_t *srcOffs,
                                     const uint64_t *dstOffs,
                                     const size_t *sizes, int srcRank,
                                     int dstRank, void *const *srcHandles,
                                     void *const *dstHandles, void **request) {
  if (request == nullptr)
    return flagcxInvalidArgument;
  *request = nullptr;
  if (count <= 0 || count > kMaxRequests || sendComm == nullptr ||
      srcOffs == nullptr || dstOffs == nullptr || sizes == nullptr ||
      srcHandles == nullptr || dstHandles == nullptr)
    return flagcxInvalidArgument;

  auto *comm = static_cast<BarexComm *>(sendComm);
  auto *srcInfo =
      reinterpret_cast<const struct flagcxOneSideHandleInfo *>(srcHandles);
  auto *dstInfo =
      reinterpret_cast<const struct flagcxOneSideHandleInfo *>(dstHandles);
  XChannel *channel = nullptr;
  uint32_t lane = 0;
  FLAGCXCHECK(barexSelectOneSidedChannel(comm, &channel, &lane));
  auto data = std::make_shared<std::vector<rw_memp_t>>();
  data->reserve(count);
  size_t totalSize = 0;
  for (int i = 0; i < count; ++i) {
    if (sizes[i] > std::numeric_limits<size_t>::max() - totalSize)
      return flagcxInvalidArgument;
    memp_t localMem;
    uint64_t remoteAddress = 0;
    uint32_t remoteRkey = 0;
    FLAGCXCHECK(barexPrepareOneSided(
        comm, channel, dstInfo, dstRank, dstOffs[i], srcInfo, srcRank,
        srcOffs[i], sizes[i], &localMem, &remoteAddress, &remoteRkey));
    data->push_back(barexMakeRw(localMem, remoteAddress, remoteRkey, sizes[i]));
    totalSize += sizes[i];
  }

  BarexRequest *req = comm->allocRequest();
  if (req == nullptr)
    return flagcxInProgress;
  req->size = totalSize;
  if (totalSize == 0) {
    barexCompleteRequest(req, flagcxSuccess);
    *request = req;
    return flagcxSuccess;
  }
  if (!comm->beginCallback()) {
    barexReleaseRequest(req);
    return comm->submissionStatus();
  }
  BarexResult result = channel->ReadBatch(
      data, [req, data](Status status) { barexCompleteCallback(req, status); },
      /*done_inline=*/true);
  if (result != accl::barex::BAREX_SUCCESS) {
    const flagcxResult_t mapped = barexResult(result);
    comm->fail(mapped);
    comm->endCallback();
    barexReleaseRequest(req);
    return mapped;
  }
  barexRecordOneSidedLane(lane);
  *request = req;
  return flagcxSuccess;
}

static flagcxResult_t barexGetDevFromName(char *name, int *dev) {
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

} // namespace barexnet

flagcxResult_t flagcxBarexRuntimeSetConnectionConfig(
    const struct flagcxBarexRuntimeConnectionConfig *config) {
  if (config == nullptr ||
      flagcxBarexRuntimeValidateChannelCount(config->channels) != flagcxSuccess)
    return flagcxInvalidArgument;
  barexnet::gRuntimeConnectionConfig = *config;
  barexnet::gRuntimeConnectionConfigActive = true;
  return flagcxSuccess;
}

void flagcxBarexRuntimeClearConnectionConfig(void) {
  barexnet::gRuntimeConnectionConfig = {};
  barexnet::gRuntimeConnectionConfigActive = false;
}

flagcxResult_t flagcxBarexRuntimeResetConnect(void *opaqueHandle) {
  return barexnet::barexResetConnect(opaqueHandle);
}

flagcxResult_t flagcxBarexRuntimeDeferMr(void *mhandle) {
  return barexnet::barexDeferMr(barexnet::gBarexRuntime, mhandle);
}

flagcxResult_t flagcxBarexRuntimeDrainDeferredMrs(void) {
  return barexnet::barexDrainDeferredMrs(barexnet::gBarexRuntime);
}

size_t flagcxBarexRuntimeMrSegmentSize(int type) {
  if (type != FLAGCX_PTR_CUDA)
    return std::numeric_limits<size_t>::max();
  const int64_t limit = barexnet::flagcxParamBarexMaxMrBytes();
  return limit > 0 ? static_cast<size_t>(limit)
                   : std::numeric_limits<size_t>::max();
}

flagcxResult_t flagcxBarexRuntimeGetCommChannels(void *opaqueComm,
                                                 uint32_t *channels) {
  if (opaqueComm == nullptr || channels == nullptr)
    return flagcxInvalidArgument;
  auto *comm = static_cast<barexnet::BarexComm *>(opaqueComm);
  if (comm->channels.empty() ||
      comm->channels.size() > FLAGCX_BAREX_RUNTIME_MAX_CHANNELS)
    return flagcxInternalError;
  *channels = static_cast<uint32_t>(comm->channels.size());
  return flagcxSuccess;
}

/* BAREX has no remote atomic primitive, so iputSignal remains an optional
   unsupported capability. */
struct flagcxNetAdaptor flagcxNetBarex = {
    // Basic functions
    "BAREX", barexnet::barexInit, barexnet::barexDevices,
    barexnet::barexGetProperties,

    // Setup functions
    barexnet::barexListen, barexnet::barexConnect, barexnet::barexAccept,
    barexnet::barexCloseSend, barexnet::barexCloseRecv,
    barexnet::barexCloseListen,

    // Memory region functions
    barexnet::barexRegMr,
    NULL, // ACCL currently has no API that consumes a DMA-BUF fd
    barexnet::barexDeregMr,

    // Two-sided functions
    barexnet::barexIsend, barexnet::barexIrecv, barexnet::barexIflush,
    barexnet::barexTest,

    // One-sided functions
    barexnet::barexIput, barexnet::barexIget,
    NULL, // iputSignal

    // Device name lookup
    barexnet::barexGetDevFromName,

    // Optional batch helpers and MR metadata
    barexnet::barexIputBatch, barexnet::barexTestBatch,
    barexnet::barexIgetBatch, barexnet::barexGetMrInfo,

    // ACCL RegUserMr cannot currently pin VMM allocations, and ACCL exposes
    // no API that consumes a DMA-BUF fd. Ordinary GPU MR remains supported.
    FLAGCX_VMM_MR_CAP_NONE, FLAGCX_NET_ADAPTOR_INTERNAL_NONE,
    FLAGCX_NET_GDR_FLUSH_NONE,
    0, // No provider-forced visibility boundary.
    barexnet::barexResetConnect,
    nullptr, // closeListen discards incomplete accepted channels
};

/* Keep the external plugin ABI at v1. The complete one-sided and batch
   interface above is available through the build-selected flagcxNetBarex
   adaptor; v1 cannot describe getMrInfo or the batch callbacks. */
extern "C" {
struct flagcxNetAdaptor_v1 flagcxNetAdaptorPlugin_v1
    __attribute__((visibility("default"))) = {
        "BAREX",
        barexnet::barexInit,
        barexnet::barexDevices,
        barexnet::barexGetProperties,
        barexnet::barexListen,
        barexnet::barexConnect,
        barexnet::barexAccept,
        barexnet::barexCloseSend,
        barexnet::barexCloseRecv,
        barexnet::barexCloseListen,
        barexnet::barexRegMr,
        NULL,
        barexnet::barexDeregMr,
        barexnet::barexIsend,
        barexnet::barexIrecv,
        barexnet::barexIflush,
        barexnet::barexTest,
        NULL, // iput
        NULL, // iget
        NULL, // iputSignal
        barexnet::barexGetDevFromName,
};
}

#endif // USE_ACCL_BAREX
