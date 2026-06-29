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
#include "bootstrap.h"
#include "cpuset.h"
#include "debug.h"
#include "flagcx_net.h"
#include "flagcx_net_adaptor.h"
#include "ib_common.h"
#include "ibvwrap.h"
#include "p2p_topo.h"
#include "param.h"
#include "socket.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#if defined(__linux__)
#include <sys/epoll.h>
#endif
#include <unistd.h>

extern struct flagcxNetAdaptor flagcxNetIbP2p;
extern flagcxResult_t flagcxNetIbP2pAbortListen(void *listenComm);

extern "C" flagcxResult_t flagcxP2pSliceBatch(void *sendComm, struct ibv_qp *qp,
                                              int count, FlagcxSlice **slices,
                                              int *failedCount);

namespace {

FLAGCX_PARAM(P2pQpsPerConn, "P2P_QPS_PER_CONN", 4);
FLAGCX_PARAM(P2pWorkersPerPool, "P2P_WORKERS_PER_POOL", 4);
FLAGCX_PARAM(P2pShardCount, "P2P_SHARD_COUNT", 8);
FLAGCX_PARAM(P2pCqDepth, "P2P_CQ_DEPTH", 4096);
FLAGCX_PARAM(P2pMaxWrPerPost, "P2P_MAX_WR_PER_POST", 256);
FLAGCX_PARAM(P2pMaxRequests, "P2P_MAX_REQUESTS", 256);
FLAGCX_PARAM(P2pBatchPollSize, "P2P_BATCH_POLL_SIZE", 64);
FLAGCX_PARAM(P2pSliceSize, "P2P_SLICE_SIZE", 1LL << 30);
FLAGCX_PARAM(P2pFragmentLimit, "P2P_FRAGMENT_LIMIT", 4096);
FLAGCX_PARAM(P2pMaxSge, "P2P_MAX_SGE", 4);
FLAGCX_PARAM(P2pMaxInline, "P2P_MAX_INLINE", 64);
FLAGCX_PARAM(P2pIbPort, "P2P_IB_PORT", 1);
FLAGCX_PARAM(P2pGidIndex, "P2P_GID_INDEX", -1);
FLAGCX_PARAM(P2pMtu, "P2P_MTU", 4096);
FLAGCX_PARAM(P2pIbTc, "P2P_IB_TC", -1);
FLAGCX_PARAM(P2pRetryCnt, "P2P_RETRY_CNT", 7);
FLAGCX_PARAM(P2pNotifMaxPeers, "P2P_NOTIF_MAX_PEERS", 64);
FLAGCX_PARAM(P2pDestDevAffinity, "P2P_DEST_DEV_AFFINITY", 0);

template <typename T>
inline T clampParam(int64_t v, T lo, T hi, T deft, const char *name) {
  if (v < (int64_t)lo || v > (int64_t)hi) {
    INFO(FLAGCX_INIT,
         "Ignore FLAGCX_%s=%lld (out of [%lld,%lld]); using default %lld", name,
         (long long)v, (long long)lo, (long long)hi, (long long)deft);
    return deft;
  }
  return (T)v;
}

void loadGlobalConfig(FlagcxP2pGlobalConfig &c) {
  c.qpsPerConn =
      clampParam<int>(flagcxParamP2pQpsPerConn(), 1, kFlagcxP2pMaxQpsPerEngine,
                      4, "P2P_QPS_PER_CONN");
  c.workersPerPool = clampParam<int>(flagcxParamP2pWorkersPerPool(), 1, 8, 4,
                                     "P2P_WORKERS_PER_POOL");
  c.shardCount =
      clampParam<int>(flagcxParamP2pShardCount(), 1, 64, 8, "P2P_SHARD_COUNT");
  c.sharedCqDepth = clampParam<size_t>(flagcxParamP2pCqDepth(), 1, 1u << 20,
                                       4096, "P2P_CQ_DEPTH");
  c.maxWrPerPost = clampParam<size_t>(flagcxParamP2pMaxWrPerPost(), 1, 1024,
                                      256, "P2P_MAX_WR_PER_POST");
  c.maxRequests = clampParam<size_t>(flagcxParamP2pMaxRequests(), 1, 1u << 16,
                                     256, "P2P_MAX_REQUESTS");
  c.batchPollSize = clampParam<size_t>(flagcxParamP2pBatchPollSize(), 1, 256,
                                       64, "P2P_BATCH_POLL_SIZE");
  c.sliceSize = clampParam<size_t>(flagcxParamP2pSliceSize(), 0, 1u << 30,
                                   1u << 30, "P2P_SLICE_SIZE");
  c.fragmentLimit = clampParam<size_t>(flagcxParamP2pFragmentLimit(), 0,
                                       c.sliceSize, 4096, "P2P_FRAGMENT_LIMIT");
  c.maxSge =
      clampParam<size_t>(flagcxParamP2pMaxSge(), 1, 32, 4, "P2P_MAX_SGE");
  c.maxInline = clampParam<size_t>(flagcxParamP2pMaxInline(), 0, 1024, 64,
                                   "P2P_MAX_INLINE");
  c.ibPort =
      clampParam<uint8_t>(flagcxParamP2pIbPort(), 1, 255, 1, "P2P_IB_PORT");
  c.gidIndex =
      clampParam<int>(flagcxParamP2pGidIndex(), -1, 255, -1, "P2P_GID_INDEX");
  {
    int64_t mv = flagcxParamP2pMtu();
    if (mv == 512 || mv == 1024 || mv == 2048 || mv == 4096) {
      c.mtuLength = (int)mv;
    } else {
      WARN(
          "Ignore FLAGCX_P2P_MTU=%lld (must be 512/1024/2048/4096); using 4096",
          (long long)mv);
      c.mtuLength = 4096;
    }
  }
  c.ibTrafficClass =
      clampParam<int>(flagcxParamP2pIbTc(), -1, 255, -1, "P2P_IB_TC");
  c.retryCnt =
      clampParam<int>(flagcxParamP2pRetryCnt(), 0, 7, 7, "P2P_RETRY_CNT");
  c.notifMaxPeers = clampParam<int>(flagcxParamP2pNotifMaxPeers(), 1, 1024, 64,
                                    "P2P_NOTIF_MAX_PEERS");
  c.enableDestDeviceAffinity = (flagcxParamP2pDestDevAffinity() != 0);
}

void dumpGlobalConfigImpl(const FlagcxP2pGlobalConfig &c);

FlagcxP2pGlobalConfig &mutableGlobalConfig() {
  static FlagcxP2pGlobalConfig cfg;
  static std::once_flag once;
  std::call_once(once, [] {
    loadGlobalConfig(cfg);
    dumpGlobalConfigImpl(cfg);
  });
  return cfg;
}

void dumpGlobalConfigImpl(const FlagcxP2pGlobalConfig &c) {
  INFO(FLAGCX_INIT, "=== FlagCX P2P GlobalConfig ===");
  INFO(FLAGCX_INIT, "qpsPerConn=%d workersPerPool=%d shardCount=%d",
       c.qpsPerConn, c.workersPerPool, c.shardCount);
  INFO(FLAGCX_INIT,
       "sharedCqDepth=%zu maxWrPerPost=%zu maxRequests=%zu batchPollSize=%zu",
       c.sharedCqDepth, c.maxWrPerPost, c.maxRequests, c.batchPollSize);
  INFO(FLAGCX_INIT, "sliceSize=%zu fragmentLimit=%zu", c.sliceSize,
       c.fragmentLimit);
  INFO(FLAGCX_INIT,
       "ibPort=%u gidIndex=%d mtu=%d tc=%d retry=%d "
       "maxSge=%zu maxInline=%zu",
       (unsigned)c.ibPort, c.gidIndex, c.mtuLength, c.ibTrafficClass,
       c.retryCnt, c.maxSge, c.maxInline);
  INFO(FLAGCX_INIT, "notifMaxPeers=%d destDevAffinity=%d", c.notifMaxPeers,
       (int)c.enableDestDeviceAffinity);
}

} // namespace

const FlagcxP2pGlobalConfig &flagcxP2pGlobalConfig() {
  return mutableGlobalConfig();
}

void flagcxP2pDumpGlobalConfig() {
  dumpGlobalConfigImpl(flagcxP2pGlobalConfig());
}

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
  struct flagcxIbQp qp_list_[kFlagcxP2pMaxQpsPerEngine];
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

struct FlagcxP2pRemoteRegion {
  uint64_t baseAddr;
  uint64_t size;
  uint32_t rkey;
};

struct FlagcxP2pMemRegWire {
  uint64_t baseAddr;
  uint64_t size;
  uint32_t rkey;
  uint32_t reserved;
};
static_assert(sizeof(FlagcxP2pMemRegWire) == 24,
              "FlagcxP2pMemRegWire size must be stable");

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
  std::atomic<bool> stopNotif;
  std::unordered_map<int, FlagcxP2pNotifConn> notifPeers;
  std::mutex notifPeerMutex;

  /* Bootstrap P2P listen state — used for ctrl meta + desc table exchange
     during connect/accept handshake. */
  struct bootstrapState *bsListenState;
  int bsListenPort;
  std::atomic<bool> stopAccept;
  volatile uint32_t acceptAbortFlag;

  /* Control-plane RPC service: accept daemon + per-session connection
     cache (initiator side) + kept-alive accepted connections (server
     side). See flagcxP2pEngineStartRpcServer / GetConn. */
  std::thread rpcServerThread;
  std::atomic<bool> rpcServerActive;
  std::atomic<bool> stopRpcServer;
  std::unordered_map<std::string, FlagcxP2pConn *> sessionConns;
  std::mutex sessionMutex;
  std::vector<FlagcxP2pConn *> acceptedConns;
  std::mutex acceptedMutex;
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
  std::vector<FlagcxP2pRemoteRegion> remoteRegions;
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
static std::unordered_map<FlagcxP2pMr, uintptr_t> gMrToBaseAddr;
static std::mutex gMemMutex;
static uint64_t gNextMrId = 1;

static std::unordered_map<uint64_t, FlagcxP2pXfer> gXferMap;
static std::mutex gXferMutex;
static uint64_t gNextXferId = 1;

struct FlagcxSliceCache {
  static constexpr size_t kCap = 4096;
  std::vector<FlagcxSlice *> ring;
  uint64_t head = 0, tail = 0;
  FlagcxSliceCache() { ring.resize(kCap, nullptr); }
  ~FlagcxSliceCache() {
    for (uint64_t i = tail; i != head; i++)
      delete ring[i % kCap];
  }
  FlagcxSlice *allocate() {
    if (head == tail)
      return new FlagcxSlice();
    FlagcxSlice *s = ring[tail % kCap];
    tail++;
    return s;
  }
  void deallocate(FlagcxSlice *s) {
    if (s == nullptr)
      return;
    if (head - tail == kCap) { // ring full: fall back to free
      delete s;
      return;
    }
    ring[head % kCap] = s;
    head++;
  }
};

static FlagcxSliceCache &sliceCache() {
  static thread_local FlagcxSliceCache cache;
  return cache;
}

static inline FlagcxSlice *acquireSlice(uint64_t srcVa, uint64_t dstVa,
                                        uint32_t length, uint32_t lkey,
                                        uint32_t rkey, uint8_t opcode,
                                        FlagcxTransferTask *task) {
  FlagcxSlice *s = sliceCache().allocate();
  s->srcVa = srcVa;
  s->dstVa = dstVa;
  s->length = length;
  s->lkey = lkey;
  s->rkey = rkey;
  s->opcode = opcode;
  s->task = task;
  s->qpDepth = nullptr;
  return s;
}

inline void flagcxBuildSlicesRuntime(FlagcxTransferTask *task, uint64_t srcVa,
                                     uint64_t dstVa, size_t totalLen,
                                     uint32_t lkey, uint32_t rkey,
                                     uint8_t opcode, size_t blockSize,
                                     size_t fragmentSize) {
  if (blockSize == 0 || totalLen <= blockSize) {
    FlagcxSlice *s = acquireSlice(srcVa, dstVa, (uint32_t)totalLen, lkey, rkey,
                                  opcode, task);
    task->sliceList.push_back(s);
    task->sliceCount.fetch_add(1, std::memory_order_release);
    return;
  }

  size_t off = 0;
  while (off < totalLen) {
    bool merge = (totalLen - off) <= blockSize + fragmentSize;
    size_t len = merge ? (totalLen - off) : blockSize;
    FlagcxSlice *s = acquireSlice(srcVa + off, dstVa + off, (uint32_t)len, lkey,
                                  rkey, opcode, task);
    task->sliceList.push_back(s);
    task->sliceCount.fetch_add(1, std::memory_order_release);
    off += len;
    if (merge)
      break;
  }
}

static void notifPollThreadFunc(FlagcxP2pEngine *engine);

namespace {

struct PoolQpEntry {
  struct ibv_qp *qp;
  void *sendComm; // owning conn (flagcxP2pSendComm/RecvComm)
  volatile int wrDepth;

  PoolQpEntry(struct ibv_qp *q, void *sc) : qp(q), sendComm(sc), wrDepth(0) {}
  PoolQpEntry(const PoolQpEntry &) = delete;
  PoolQpEntry &operator=(const PoolQpEntry &) = delete;
};

struct PendingSliceQueue {
  std::vector<FlagcxSlice *> slices;
  size_t head = 0;

  bool empty() const { return head >= slices.size(); }
  size_t size() const { return slices.size() - head; }

  void append(const std::vector<FlagcxSlice *> &src) {
    if (src.empty())
      return;
    if (empty()) {
      slices.clear();
      head = 0;
    } else if (head >= 4096 && head * 2 >= slices.size()) {
      slices.erase(slices.begin(), slices.begin() + head);
      head = 0;
    }
    slices.insert(slices.end(), src.begin(), src.end());
  }

  void clear() {
    slices.clear();
    head = 0;
  }
};

class FlagcxWorkerPool {
public:
  FlagcxWorkerPool(int ibDevN, struct ibv_context *ctx);
  ~FlagcxWorkerPool();
  FlagcxWorkerPool(const FlagcxWorkerPool &) = delete;
  FlagcxWorkerPool &operator=(const FlagcxWorkerPool &) = delete;

  struct ibv_cq *getSharedCq() const {
    return shared_cq_;
  }
  int workerCount() const { return numWorkers_; }
  void registerQp(void *sendComm, struct ibv_qp *qp);
  void unregisterQp(struct ibv_qp *qp);

  flagcxResult_t submitPostSend(void *sendComm, FlagcxSlice **slices,
                                int count);

  void startNotif(FlagcxP2pEngine *engine);
  void stopNotif();

private:
  void transferWorkerLoop(int tid);
  void performPostSend(int tid);
  void performPollCq();
  void notifWorkerLoop();

  void pinToNicCpus(const char *role, int id);
  static uint64_t nowNs() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               clk::now().time_since_epoch())
        .count();
  }

  int ibDevN_;
  struct ibv_cq *shared_cq_ = nullptr;
  cpu_set_t cpuAffinity_;
  bool hasCpuAffinity_ = false;

  int numWorkers_;
  int numShards_;
  size_t maxWrPerPost_;
  size_t batchPollSize_;
  int maxWrDepth_ = 0;

  std::mutex qp_mu_;
  std::vector<std::unique_ptr<PoolQpEntry>> qpEntries_;
  std::unordered_map<uint32_t, int> qpNumToIdx_;
  std::vector<std::vector<int>> workerQpIdx_;
  std::vector<size_t> workerQpCursor_;
  std::unordered_map<void *, int> connQpRegCount_;
  std::vector<std::unordered_map<void *, std::vector<FlagcxSlice *>>>
      slice_queues_;
  std::unique_ptr<std::mutex[]> slice_locks_;
  std::unique_ptr<std::atomic<int>[]> slice_queue_count_;
  std::atomic<uint64_t> shardRoundRobin_{0};
  std::vector<std::unordered_map<void *, PendingSliceQueue>>
      collective_slice_queue_;

  std::atomic<uint64_t> submitted_{0};
  std::atomic<uint64_t> processed_{0};
  std::atomic<int> suspended_flag_{0};
  std::condition_variable cv_;
  std::mutex cv_mu_;

  std::atomic<bool> running_{true};
  std::vector<std::thread> transferThreads_;

  FlagcxP2pEngine *engine_ = nullptr;
  std::thread notifThread_;
  std::atomic<bool> notifSpawned_{false};
};

static bool flagcxP2pGetNicCpuset(struct ibv_context *ctx, cpu_set_t *mask) {
  if (ctx == nullptr || ctx->device == nullptr || mask == nullptr)
    return false;
  const char *devName = flagcxWrapIbvGetDeviceName(ctx->device);
  if (devName == nullptr)
    return false;
  char path[256];
  snprintf(path, sizeof(path), "/sys/class/infiniband/%s/device/local_cpus",
           devName);
  FILE *fp = fopen(path, "r");
  if (fp == nullptr)
    return false;
  char buf[1024];
  char *line = fgets(buf, sizeof(buf), fp);
  fclose(fp);
  if (line == nullptr)
    return false;
  // Strip trailing newline/CR before parsing.
  size_t n = strlen(buf);
  while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r'))
    buf[--n] = '\0';
  CPU_ZERO(mask);
  if (flagcxStrToCpuset(buf, mask) != flagcxSuccess)
    return false;
  int set = CPU_COUNT(mask);
  if (set == 0)
    return false;
  long online = sysconf(_SC_NPROCESSORS_ONLN);
  if (online > 0 && set >= online)
    return false;
  return true;
}

FlagcxWorkerPool::FlagcxWorkerPool(int ibDevN, struct ibv_context *ctx)
    : ibDevN_(ibDevN) {
  const auto &C = flagcxP2pGlobalConfig();
  numWorkers_ = C.workersPerPool;
  numShards_ = C.shardCount;
  maxWrPerPost_ = C.maxWrPerPost;
  batchPollSize_ = C.batchPollSize;

  if (numWorkers_ > 0 && C.qpsPerConn % numWorkers_ != 0) {
    WARN("NET/IB_P2P : pool[%d] qpsPerEngine=%d not divisible by "
         "workersPerPool=%d — QPs spread per connection but unevenly; some "
         "workers will own more QPs than others for each conn",
         ibDevN_, C.qpsPerConn, numWorkers_);
  }

  flagcxResult_t res = flagcxWrapIbvCreateCq(
      &shared_cq_, ctx, (int)C.sharedCqDepth, NULL, NULL, 0);
  if (res != flagcxSuccess) {
    WARN("NET/IB_P2P : pool[%d] failed to create shared CQ", ibDevN_);
    shared_cq_ = nullptr;
    return;
  }
  INFO(FLAGCX_INIT,
       "NET/IB_P2P : pool[%d] shared CQ created (depth=%zu, workers=%d, "
       "shards=%d, qpsPerConn=%d)",
       ibDevN_, C.sharedCqDepth, numWorkers_, numShards_, C.qpsPerConn);

  slice_queues_.resize(numShards_);
  slice_locks_.reset(new std::mutex[numShards_]);
  slice_queue_count_.reset(new std::atomic<int>[numShards_]);
  for (int s = 0; s < numShards_; s++)
    slice_queue_count_[s].store(0, std::memory_order_relaxed);

  workerQpIdx_.resize(numWorkers_);
  workerQpCursor_.assign(numWorkers_, 0);
  collective_slice_queue_.resize(numWorkers_);

  hasCpuAffinity_ = flagcxP2pGetNicCpuset(ctx, &cpuAffinity_);

  transferThreads_.reserve(numWorkers_);
  for (int t = 0; t < numWorkers_; t++) {
    transferThreads_.emplace_back([this, t] { this->transferWorkerLoop(t); });
  }
}

FlagcxWorkerPool::~FlagcxWorkerPool() {
  running_.store(false, std::memory_order_release);
  cv_.notify_all();
  for (auto &t : transferThreads_) {
    if (t.joinable())
      t.join();
  }
  // notifThread_ is joined explicitly via stopNotif() in EngineDestroy;
  // by the time ~pool runs at process exit it should already be joined.
  if (notifThread_.joinable()) {
    notifThread_.join();
  }
  // CQ destruction skipped: pool is process-lived; OS reclaims.
}

void FlagcxWorkerPool::startNotif(FlagcxP2pEngine *engine) {
  if (engine == nullptr)
    return;
  // Compare-exchange ensures only the first attach spawns the thread.
  bool expected = false;
  if (!notifSpawned_.compare_exchange_strong(expected, true)) {
    // Already attached — keep existing engine pointer (or update if NULL).
    if (engine_ == nullptr)
      engine_ = engine;
    return;
  }
  engine_ = engine;
  notifThread_ = std::thread([this] { this->notifWorkerLoop(); });
  INFO(FLAGCX_INIT, "NET/IB_P2P : pool[%d] notifWorker spawned", ibDevN_);
}

void FlagcxWorkerPool::stopNotif() {
  // Caller must have set engine_->stopNotif before calling — that breaks
  // the epoll loop in notifPollThreadFunc.
  if (notifThread_.joinable()) {
    notifThread_.join();
  }
  engine_ = nullptr;
  notifSpawned_.store(false, std::memory_order_release);
}

void FlagcxWorkerPool::pinToNicCpus(const char *role, int id) {
  if (!hasCpuAffinity_)
    return;
  cpu_set_t inherited, target;
  if (sched_getaffinity(0, sizeof(cpu_set_t), &inherited) != 0) {
    WARN("NET/IB_P2P : pool[%d] %s %d sched_getaffinity failed (errno=%d); "
         "leaving thread unpinned",
         ibDevN_, role, id, errno);
    return;
  }
  CPU_AND(&target, &cpuAffinity_, &inherited);
  if (CPU_COUNT(&target) == 0) {
    INFO(FLAGCX_INIT,
         "NET/IB_P2P : pool[%d] %s %d NIC-local set disjoint from inherited "
         "affinity (%d cpus); leaving thread on inherited affinity",
         ibDevN_, role, id, CPU_COUNT(&inherited));
    return;
  }
  if (sched_setaffinity(0, sizeof(cpu_set_t), &target) != 0) {
    WARN("NET/IB_P2P : pool[%d] %s %d sched_setaffinity failed (errno=%d)",
         ibDevN_, role, id, errno);
  } else {
    INFO(FLAGCX_INIT,
         "NET/IB_P2P : pool[%d] %s %d pinned to %d CPUs (NIC-local ∩ "
         "inherited)",
         ibDevN_, role, id, CPU_COUNT(&target));
  }
}

void FlagcxWorkerPool::notifWorkerLoop() {
  if (engine_ == nullptr)
    return;
  pinToNicCpus("notifWorker", 0);
  // Reuse the original engine-side body — same behavior, just owned by
  // the pool's thread.
  notifPollThreadFunc(engine_);
}

void FlagcxWorkerPool::registerQp(void *sendComm, struct ibv_qp *qp) {
  if (!qp || numWorkers_ <= 0)
    return;

  std::lock_guard<std::mutex> lk(qp_mu_);

  if (maxWrDepth_ == 0) {
    struct ibv_qp_attr attr;
    struct ibv_qp_init_attr initAttr;
    memset(&attr, 0, sizeof(attr));
    memset(&initAttr, 0, sizeof(initAttr));
    if (flagcxWrapIbvQueryQp(qp, &attr, IBV_QP_CAP, &initAttr) ==
        flagcxSuccess) {
      int cap = (int)initAttr.cap.max_send_wr;
      if (cap > 0) {
        maxWrDepth_ = cap;
        INFO(FLAGCX_INIT,
             "NET/IB_P2P : pool[%d] resolved max_wr_depth=%d from first QP",
             ibDevN_, cap);
      }
    } else {
      WARN("NET/IB_P2P : pool[%d] ibv_query_qp failed; max_wr_depth "
           "stays unresolved (slice posts will fall back to no gate)",
           ibDevN_);
    }
  }

  int idx = (int)qpEntries_.size();
  qpEntries_.emplace_back(new PoolQpEntry(qp, sendComm));
  qpNumToIdx_[qp->qp_num] = idx;

  int connIdx = connQpRegCount_[sendComm]++;
  int slot = connIdx % numWorkers_;
  workerQpIdx_[slot].push_back(idx);
}

void FlagcxWorkerPool::unregisterQp(struct ibv_qp *qp) {
  if (!qp)
    return;
  std::lock_guard<std::mutex> lk(qp_mu_);
  auto it = qpNumToIdx_.find(qp->qp_num);
  if (it == qpNumToIdx_.end())
    return;
  int idx = it->second;
  qpNumToIdx_.erase(it);
  void *sc = qpEntries_[idx]->sendComm;
  for (auto &shard : workerQpIdx_) {
    auto vit = std::find(shard.begin(), shard.end(), idx);
    if (vit != shard.end()) {
      shard.erase(vit);
      break;
    }
  }
  auto cit = connQpRegCount_.find(sc);
  if (cit != connQpRegCount_.end() && --cit->second <= 0)
    connQpRegCount_.erase(cit);
  // Slot kept alive (NULL'd) so any in-flight slice's qpDepth pointer stays
  // valid.
  qpEntries_[idx]->qp = nullptr;
  qpEntries_[idx]->sendComm = nullptr;
}

flagcxResult_t FlagcxWorkerPool::submitPostSend(void *sendComm,
                                                FlagcxSlice **slices,
                                                int count) {
  if (count <= 0 || slices == nullptr)
    return flagcxSuccess;

  // Backpressure: spin-yield until in-flight count drops below threshold.
  // Prevents unbounded queue growth under sustained submission bursts.
  const size_t maxPending = flagcxP2pGlobalConfig().maxRequests * 4;
  while (submitted_.load(std::memory_order_acquire) -
             processed_.load(std::memory_order_acquire) >
         maxPending) {
    std::this_thread::yield();
  }

  const int fanout = std::min(numShards_, count);
  int shard = 0;
  if (fanout != numShards_) {
    shard = static_cast<int>(
        shardRoundRobin_.fetch_add(fanout, std::memory_order_relaxed) %
        static_cast<uint64_t>(numShards_));
  }
  int offset = 0;
  for (int i = 0; i < fanout; i++) {
    const int remaining = count - offset;
    const int shardsLeft = fanout - i;
    const int take = (remaining + shardsLeft - 1) / shardsLeft;
    std::lock_guard<std::mutex> lk(slice_locks_[shard]);
    auto &vec = slice_queues_[shard][sendComm];
    vec.insert(vec.end(), slices + offset, slices + offset + take);
    slice_queue_count_[shard].fetch_add(take, std::memory_order_relaxed);
    offset += take;
    if (++shard == numShards_)
      shard = 0;
  }
  submitted_.fetch_add(count, std::memory_order_release);

  if (suspended_flag_.load(std::memory_order_acquire) > 0) {
    std::lock_guard<std::mutex> lk(cv_mu_);
    cv_.notify_all();
  }
  return flagcxSuccess;
}

void FlagcxWorkerPool::transferWorkerLoop(int tid) {
  pinToNicCpus("transferWorker", tid);
  const static uint64_t kWaitPeriodInNano = 100ull * 1000 * 1000; // 100ms
  uint64_t last_wait_ts = nowNs();

  while (running_.load(std::memory_order_relaxed)) {
    auto processed_slice_count = processed_.load(std::memory_order_relaxed);
    auto submitted_slice_count = submitted_.load(std::memory_order_relaxed);

    if (processed_slice_count == submitted_slice_count) {
      uint64_t curr_wait_ts = nowNs();
      if (curr_wait_ts - last_wait_ts > kWaitPeriodInNano) {
        std::unique_lock<std::mutex> lock(cv_mu_);
        suspended_flag_.fetch_add(1);
        if (processed_.load(std::memory_order_relaxed) ==
            submitted_.load(std::memory_order_relaxed)) {
          cv_.wait_for(lock, std::chrono::seconds(1));
        }
        suspended_flag_.fetch_sub(1);
        last_wait_ts = curr_wait_ts;
      }
      continue;
    }

    performPostSend(tid);
    performPollCq();
  }
}

void FlagcxWorkerPool::performPostSend(int tid) {
  if (numWorkers_ <= 0)
    return;

  auto &local = collective_slice_queue_[tid];
  for (int s = tid; s < numShards_; s += numWorkers_) {
    if (slice_queue_count_[s].load(std::memory_order_relaxed) == 0)
      continue;
    std::lock_guard<std::mutex> lk(slice_locks_[s]);
    for (auto &entry : slice_queues_[s]) {
      if (entry.second.empty())
        continue;
      auto &dst = local[entry.first];
      dst.append(entry.second);
      entry.second.clear();
    }
    slice_queue_count_[s].store(0, std::memory_order_relaxed);
  }

  std::vector<PoolQpEntry *> myQpEntries;
  int curMaxDepth;
  {
    std::lock_guard<std::mutex> lk(qp_mu_);
    myQpEntries.reserve(workerQpIdx_[tid].size());
    for (int idx : workerQpIdx_[tid])
      myQpEntries.push_back(qpEntries_[idx].get());
    curMaxDepth = maxWrDepth_;
  }

  size_t &cursor = workerQpCursor_[tid];
  for (auto &entry : local) {
    void *sc = entry.first;
    auto &pending = entry.second;
    if (pending.empty())
      continue;

    static thread_local std::vector<PoolQpEntry *> myQpOnComm;
    myQpOnComm.clear();
    myQpOnComm.reserve(myQpEntries.size());
    for (PoolQpEntry *e : myQpEntries) {
      if (e && e->qp != nullptr && e->sendComm == sc)
        myQpOnComm.push_back(e);
    }
    if (myQpOnComm.empty()) {
      WARN("NET/IB_P2P : pool[%d] worker %d has no QP for Engine %p; "
           "failing %zu slices",
           ibDevN_, tid, sc, pending.size());
      for (size_t idx = pending.head; idx < pending.slices.size(); idx++)
        pending.slices[idx]->markFailed();
      processed_.fetch_add(pending.size(), std::memory_order_release);
      pending.clear();
      continue;
    }

    const size_t ringSz = myQpOnComm.size();
    size_t i = pending.head;
    while (i < pending.slices.size()) {
      PoolQpEntry *chosen = nullptr;
      size_t take = 0;
      for (size_t k = 0; k < ringSz; k++) {
        PoolQpEntry *e = myQpOnComm[(cursor + k) % ringSz];
        int cur = e->wrDepth;
        size_t room;
        if (curMaxDepth == 0) {
          room = pending.slices.size() - i; // depth unknown: no gate
        } else if (cur >= curMaxDepth) {
          continue; // this QP is full, try the next one
        } else {
          room = (size_t)(curMaxDepth - cur);
        }
        take =
            std::min<size_t>({room, maxWrPerPost_, pending.slices.size() - i});
        chosen = e;
        cursor = (cursor + k + 1) % ringSz;
        break;
      }

      if (chosen == nullptr)
        break; // all of this worker's QPs for the engine are full; retry later

      volatile int *depthPtr = &chosen->wrDepth;
      __sync_fetch_and_add(depthPtr, (int)take);

      for (size_t k = 0; k < take; k++) {
        FlagcxSlice *sl = pending.slices[i + k];
        sl->qpDepth = depthPtr;
      }

      int failedCount = 0;
      flagcxResult_t rc = flagcxP2pSliceBatch(
          sc, chosen->qp, (int)take, pending.slices.data() + i, &failedCount);

      if (rc != flagcxSuccess && failedCount > 0)
        processed_.fetch_add(failedCount, std::memory_order_release);
      i += take;
    }
    pending.head = i;
    if (pending.empty())
      pending.clear();
  }
}

void FlagcxWorkerPool::performPollCq() {
  if (shared_cq_ == nullptr)
    return;

  constexpr int kMaxPollBatch = 256;
  struct ibv_wc wcs[kMaxPollBatch];
  int batch = (int)std::min<size_t>(batchPollSize_, kMaxPollBatch);
  int n = 0;
  if (flagcxWrapIbvPollCq(shared_cq_, batch, wcs, &n) != flagcxSuccess) {
    WARN("NET/IB_P2P : ibv_poll_cq failed on shared CQ %p", shared_cq_);
    return;
  }
  if (n == 0)
    return;

  uint64_t sliceProgressed = 0;
  std::unordered_map<volatile int *, int> qpDepthSet;
  for (int i = 0; i < n; i++) {
    uintptr_t raw = (uintptr_t)wcs[i].wr_id;
    if (raw == 0 || (raw & 1ull) == 0)
      continue;

    FlagcxSlice *slice =
        reinterpret_cast<FlagcxSlice *>(raw & ~(uintptr_t)1ull);
    if (slice->qpDepth != NULL)
      qpDepthSet[slice->qpDepth]++;
    if (wcs[i].status != IBV_WC_SUCCESS) {
      WARN("NET/IB_P2P : pool poll error status %d for slice %p", wcs[i].status,
           slice);
      slice->markFailed();
    } else {
      slice->markSuccess();
    }
    sliceProgressed++;
  }
  for (auto &entry : qpDepthSet)
    __sync_fetch_and_sub(entry.first, entry.second);
  if (sliceProgressed > 0)
    processed_.fetch_add(sliceProgressed, std::memory_order_release);
}

// ---- Per-ibDev singleton plumbing -----------------------------------

static std::unique_ptr<FlagcxWorkerPool> gPools[MAX_IB_DEVS];
static std::mutex gPoolMu;

static FlagcxWorkerPool *getOrCreatePool(int ibDevN, struct ibv_context *ctx) {
  if (ibDevN < 0 || ibDevN >= MAX_IB_DEVS || ctx == NULL)
    return NULL;
  std::lock_guard<std::mutex> lk(gPoolMu);
  if (!gPools[ibDevN])
    gPools[ibDevN].reset(new FlagcxWorkerPool(ibDevN, ctx));
  return gPools[ibDevN].get();
}

static FlagcxWorkerPool *lookupPool(int ibDevN) {
  if (ibDevN < 0 || ibDevN >= MAX_IB_DEVS)
    return nullptr;
  std::lock_guard<std::mutex> lk(gPoolMu);
  return gPools[ibDevN].get();
}

} // namespace

// ---- Hooks consumed by ibrc_p2p_adaptor.cc (forward-declared there). ----
struct ibv_cq *flagcxP2pPoolGetSharedCq(int ibDevN, struct ibv_context *ctx) {
  FlagcxWorkerPool *pool = getOrCreatePool(ibDevN, ctx);
  return pool ? pool->getSharedCq() : NULL;
}

void flagcxP2pPoolRegisterQp(int ibDevN, void *sendComm, struct ibv_qp *qp) {
  if (qp == nullptr)
    return;
  FlagcxWorkerPool *pool = lookupPool(ibDevN);
  if (pool)
    pool->registerQp(sendComm, qp);
}

void flagcxP2pPoolUnregisterQp(int ibDevN, struct ibv_qp *qp) {
  if (qp == nullptr)
    return;
  FlagcxWorkerPool *pool = lookupPool(ibDevN);
  if (pool)
    pool->unregisterQp(qp);
}

flagcxResult_t flagcxP2pPoolSubmit(int ibDevN, void *sendComm,
                                   FlagcxSlice **slices, int count) {
  FlagcxWorkerPool *pool = lookupPool(ibDevN);
  if (pool == nullptr) {
    WARN("NET/IB_P2P : flagcxP2pPoolSubmit on uninitialized pool[%d]", ibDevN);
    return flagcxInternalError;
  }
  return pool->submitPostSend(sendComm, slices, count);
}

void flagcxP2pPoolStartNotif(int ibDevN, struct ibv_context *ctx,
                             FlagcxP2pEngine *engine) {
  FlagcxWorkerPool *pool = getOrCreatePool(ibDevN, ctx);
  if (pool == nullptr) {
    WARN("NET/IB_P2P : pool[%d] cannot be created for notif", ibDevN);
    return;
  }
  pool->startNotif(engine);
}

void flagcxP2pPoolStopNotif() {
  // Stop notif on whichever pool currently owns it (only one does — the
  // first that StartNotif touched).
  std::lock_guard<std::mutex> lk(gPoolMu);
  for (int i = 0; i < MAX_IB_DEVS; i++) {
    if (gPools[i])
      gPools[i]->stopNotif();
  }
}

struct PoolTransferTask {
  FlagcxTransferTask fx;
  PoolTransferTask *poolNext = nullptr;

  void reset() {
    fx.sliceCount.store(0, std::memory_order_relaxed);
    fx.doneSliceCount.store(0, std::memory_order_relaxed);
    fx.failedCount.store(0, std::memory_order_relaxed);
    fx.sliceList.clear();
    poolNext = nullptr;
  }
};

static FlagcxP2pCommView *getCommView(void *comm) {
  return reinterpret_cast<FlagcxP2pCommView *>(comm);
}

static bool
buildAndSubmitToPool(PoolTransferTask *task, const std::vector<void *> &dataVec,
                     const std::vector<size_t> &sizeVec,
                     const std::vector<FlagcxP2pRdmaDesc> &descs,
                     const std::vector<FlagcxP2pMemRegEntry> &localEntries,
                     int numIovs, void *sendComm, int connIbDevN,
                     uint8_t opcode) {
  const auto &sliceCfg = flagcxP2pGlobalConfig();
  for (int i = 0; i < numIovs; i++) {
    if (localEntries[i].ibDevN != connIbDevN) {
      WARN("NET/IB_P2P : iov[%d] ibDevN mismatch (%d vs conn %d)", i,
           localEntries[i].ibDevN, connIbDevN);
      for (auto *s : task->fx.sliceList)
        s->markFailed();
      return false;
    }
    auto *localMr =
        reinterpret_cast<FlagcxP2pMrHandleView *>(localEntries[i].mhandle);
    uint64_t localVa = (uintptr_t)dataVec[i];
    uint64_t remoteVa = descs[i].addr;
    flagcxBuildSlicesRuntime(&task->fx, localVa, remoteVa, sizeVec[i],
                             localMr->lkey, descs[i].rkey, opcode,
                             sliceCfg.sliceSize, sliceCfg.fragmentLimit);
  }

  if (task->fx.sliceList.empty()) {
    return false;
  }

  flagcxResult_t rc =
      flagcxP2pPoolSubmit(connIbDevN, sendComm, task->fx.sliceList.data(),
                          (int)task->fx.sliceList.size());
  if (rc != flagcxSuccess) {
    for (auto *s : task->fx.sliceList)
      s->markFailed();
    return false;
  }
  return true;
}

static constexpr uint64_t kPoolXferTag = 1ull << 63;

static std::mutex gPoolTaskFreeMu;
static PoolTransferTask *gPoolTaskFreeHead = nullptr;

static PoolTransferTask *acquirePoolTask() {
  {
    std::lock_guard<std::mutex> lk(gPoolTaskFreeMu);
    if (gPoolTaskFreeHead != nullptr) {
      PoolTransferTask *t = gPoolTaskFreeHead;
      gPoolTaskFreeHead = t->poolNext;
      t->reset();
      return t;
    }
  }
  PoolTransferTask *t = new PoolTransferTask();
  t->reset();
  return t;
}

static void releasePoolTask(PoolTransferTask *t) {
  if (t == nullptr)
    return;
  std::lock_guard<std::mutex> lk(gPoolTaskFreeMu);
  t->poolNext = gPoolTaskFreeHead;
  gPoolTaskFreeHead = t;
}

static inline uint64_t encodePoolXfer(PoolTransferTask *t) {
  return (uint64_t)(uintptr_t)t | kPoolXferTag;
}

static inline PoolTransferTask *decodePoolXfer(uint64_t transferId) {
  if ((transferId & kPoolXferTag) == 0)
    return nullptr;
  return reinterpret_cast<PoolTransferTask *>(
      (uintptr_t)(transferId & ~kPoolXferTag));
}

static void finalizePoolTask(PoolTransferTask *task) {
  for (auto *s : task->fx.sliceList)
    sliceCache().deallocate(s);
  task->fx.sliceList.clear();
}

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
  std::unordered_map<FlagcxP2pMr, uintptr_t>::const_iterator mrIt =
      gMrToBaseAddr.find(mr);
  if (mrIt == gMrToBaseAddr.end())
    return NULL;

  std::unordered_map<uintptr_t, FlagcxP2pMemRegEntry>::iterator entryIt =
      gMemRegInfo.find(mrIt->second);
  if (entryIt != gMemRegInfo.end())
    return &entryIt->second;

  return NULL;
}

static bool memRegContains(const FlagcxP2pMemRegEntry &entry, uintptr_t addr,
                           size_t size) {
  if (addr < entry.baseAddr)
    return false;

  const uintptr_t offset = addr - entry.baseAddr;
  return offset <= entry.size && size <= entry.size - offset;
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

static std::string
socketAddrToHostString(const union flagcxSocketAddress *addr) {
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
  if (deviceAdaptor->ipcMemHandleCreate(&handle, &handleSize) !=
      flagcxSuccess) {
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
        setFdNonblocking(fd) != 0 ||
        notifRegisterConn(engine, fd, &remoteAddr) != 0) {
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
      if (engine->notifListenActive &&
          pfds[i].fd == engine->notifListenSock.fd) {
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

// ============================================================================
// Bootstrap P2P helpers for ctrl meta + desc table exchange
// ============================================================================

static flagcxResult_t bootstrapExchangeCtrlMeta(struct bootstrapState *bsState,
                                                FlagcxP2pCtrlMeta *localMeta,
                                                FlagcxP2pCtrlMeta *remoteMeta) {
  FLAGCXCHECK(bootstrapExchange(bsState, 0, 1, localMeta, sizeof(*localMeta),
                                remoteMeta, sizeof(*remoteMeta)));
  return flagcxSuccess;
}

static int bootstrapExchangeDescTable(struct bootstrapState *bsState,
                                      FlagcxP2pConn *conn) {
  if (bsState == NULL || conn == NULL || conn->sendComm == NULL)
    return -1;

  std::vector<FlagcxP2pMemRegWire> localTable;
  {
    std::lock_guard<std::mutex> lock(gMemMutex);
    localTable.reserve(gMemRegInfo.size());
    for (std::unordered_map<uintptr_t, FlagcxP2pMemRegEntry>::iterator it =
             gMemRegInfo.begin();
         it != gMemRegInfo.end(); ++it) {
      FlagcxP2pMrHandleView *mrView =
          reinterpret_cast<FlagcxP2pMrHandleView *>(it->second.mhandle);
      if (mrView == NULL)
        continue;
      FlagcxP2pMemRegWire w;
      w.baseAddr = it->second.baseAddr;
      w.size = it->second.size;
      w.rkey = mrView->rkey;
      w.reserved = 0;
      localTable.push_back(w);
    }
  }

  uint32_t localCount = static_cast<uint32_t>(localTable.size());
  uint32_t remoteCount = 0;
  if (bootstrapExchange(bsState, 0, 2, &localCount, sizeof(localCount),
                        &remoteCount, sizeof(remoteCount)) != flagcxSuccess)
    return -1;

  // Sanity check: reject absurdly large counts to prevent OOM or overflow
  const uint32_t MAX_REMOTE_REGIONS = 65536;
  if (remoteCount > MAX_REMOTE_REGIONS) {
    WARN("bootstrapExchangeDescTable: remote count %u exceeds limit %u",
         remoteCount, MAX_REMOTE_REGIONS);
    return -1;
  }

  std::vector<FlagcxP2pMemRegWire> remoteTable(remoteCount);
  if (bootstrapExchange(
          bsState, 0, 3, localTable.data(),
          static_cast<int>(localCount * sizeof(FlagcxP2pMemRegWire)),
          remoteTable.data(),
          static_cast<int>(remoteCount * sizeof(FlagcxP2pMemRegWire))) !=
      flagcxSuccess)
    return -1;

  conn->remoteRegions.clear();
  conn->remoteRegions.reserve(remoteCount);
  for (uint32_t i = 0; i < remoteCount; i++) {
    FlagcxP2pRemoteRegion r;
    r.baseAddr = remoteTable[i].baseAddr;
    r.size = remoteTable[i].size;
    r.rkey = remoteTable[i].rkey;
    conn->remoteRegions.push_back(r);
  }
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
  engine->rpcServerActive = false;
  engine->stopRpcServer = false;
  engine->bsListenState = NULL;
  engine->bsListenPort = 0;
  engine->stopAccept = false;
  engine->acceptAbortFlag = 0;
  memset(engine->listeners, 0, sizeof(engine->listeners));
  memset(&engine->notifListenSock, 0, sizeof(engine->notifListenSock));

  if (engine->adaptor->init() != flagcxSuccess) {
    delete engine;
    return NULL;
  }

  // Initialize bootstrap network context (discovers local NIC)
  bootstrapNetInit();

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

  if (engine->notifListenActive && flagcxNIbDevs > 0) {
    flagcxP2pPoolStartNotif(0, flagcxIbDevs[0].context, engine);
  }

  // Set up bootstrap P2P listen for ctrl meta + desc table exchange
  struct bootstrapState *bsState = NULL;
  char bsListenHandle[FLAGCX_NET_HANDLE_MAXSIZE];
  memset(bsListenHandle, 0, sizeof(bsListenHandle));
  if (bootstrapP2pListen(FLAGCX_SOCKET_MAGIC, &engine->acceptAbortFlag,
                         bsListenHandle, &bsState) == flagcxSuccess) {
    engine->bsListenState = bsState;
    union flagcxSocketAddress bsAddr;
    flagcxSocketGetAddr(&bsState->p2p->sock, &bsAddr);
    engine->bsListenPort = socketAddrPort(&bsAddr);
    INFO(FLAGCX_INIT, "NET/IB_P2P : bootstrap P2P listen on port %d",
         engine->bsListenPort);
  }

  return engine;
}

void flagcxP2pEngineDestroy(FlagcxP2pEngine *engine) {
  if (engine == NULL)
    return;

  flagcxP2pEngineStopAccept(engine);
  if (engine->notifListenActive) {
    flagcxSocketClose(&engine->notifListenSock);
    engine->notifListenActive = false;
  }
  flagcxP2pPoolStopNotif();

  if (engine->bsListenState) {
    bootstrapClose(engine->bsListenState);
    engine->bsListenState = NULL;
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

  if (engine->rpcServerThread.joinable() &&
      engine->rpcServerThread.get_id() != std::this_thread::get_id()) {
    engine->rpcServerThread.join();
  }
  {
    std::lock_guard<std::mutex> lock(engine->sessionMutex);
    for (std::unordered_map<std::string, FlagcxP2pConn *>::iterator it =
             engine->sessionConns.begin();
         it != engine->sessionConns.end(); ++it) {
      flagcxP2pEngineConnDestroy(it->second);
    }
    engine->sessionConns.clear();
  }
  {
    std::lock_guard<std::mutex> lock(engine->acceptedMutex);
    for (size_t i = 0; i < engine->acceptedConns.size(); i++) {
      flagcxP2pEngineConnDestroy(engine->acceptedConns[i]);
    }
    engine->acceptedConns.clear();
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
    gMrToBaseAddr.clear();
  }

  if (engine->topoMgr) {
    flagcxP2pTopoDestroy(engine->topoMgr);
  }

  delete engine;
}

void flagcxP2pEngineStopAccept(FlagcxP2pEngine *engine) {
  if (engine == NULL)
    return;

  engine->stopAccept.store(true, std::memory_order_release);
  engine->stopNotif = true;
  engine->stopRpcServer.store(true, std::memory_order_release);
  __atomic_store_n(&engine->acceptAbortFlag, 1, __ATOMIC_RELEASE);

  if (engine->notifListenActive) {
    flagcxSocketClose(&engine->notifListenSock);
    engine->notifListenActive = false;
  }

  if (engine->bsListenState && engine->bsListenState->p2p) {
    flagcxSocketClose(&engine->bsListenState->p2p->sock);
  }

  for (int d = 0; d < engine->nDevs; d++) {
    if (engine->listeners[d].listenComm) {
      flagcxNetIbP2pAbortListen(engine->listeners[d].listenComm);
    }
  }

  if (engine->rpcServerThread.joinable() &&
      engine->rpcServerThread.get_id() != std::this_thread::get_id()) {
    engine->rpcServerThread.join();
    engine->rpcServerActive.store(false, std::memory_order_release);
  }
}

static int exchangeMemRegTable(FlagcxP2pConn *conn) {
  if (conn == NULL || conn->sendComm == NULL)
    return -1;
  FlagcxP2pCommView *view = getCommView(conn->sendComm);

  std::vector<FlagcxP2pMemRegWire> localTable;
  {
    std::lock_guard<std::mutex> lock(gMemMutex);
    localTable.reserve(gMemRegInfo.size());
    for (std::unordered_map<uintptr_t, FlagcxP2pMemRegEntry>::iterator it =
             gMemRegInfo.begin();
         it != gMemRegInfo.end(); ++it) {
      FlagcxP2pMrHandleView *mrView =
          reinterpret_cast<FlagcxP2pMrHandleView *>(it->second.mhandle);
      if (mrView == NULL)
        continue;
      FlagcxP2pMemRegWire w;
      w.baseAddr = it->second.baseAddr;
      w.size = it->second.size;
      w.rkey = mrView->rkey;
      w.reserved = 0;
      localTable.push_back(w);
    }
  }

  uint32_t localCount = static_cast<uint32_t>(localTable.size());
  uint32_t remoteCount = 0;
  if (flagcxSocketSendRecv(&view->sock, &localCount, sizeof(localCount),
                           &view->sock, &remoteCount,
                           sizeof(remoteCount)) != flagcxSuccess)
    return -1;

  std::vector<FlagcxP2pMemRegWire> remoteTable(remoteCount);
  if (flagcxSocketSendRecv(
          &view->sock, localTable.data(),
          static_cast<int>(localCount * sizeof(FlagcxP2pMemRegWire)),
          &view->sock, remoteTable.data(),
          static_cast<int>(remoteCount * sizeof(FlagcxP2pMemRegWire))) !=
      flagcxSuccess)
    return -1;

  conn->remoteRegions.clear();
  conn->remoteRegions.reserve(remoteCount);
  for (uint32_t i = 0; i < remoteCount; i++) {
    FlagcxP2pRemoteRegion r;
    r.baseAddr = remoteTable[i].baseAddr;
    r.size = remoteTable[i].size;
    r.rkey = remoteTable[i].rkey;
    conn->remoteRegions.push_back(r);
  }
  return 0;
}

FlagcxP2pConn *flagcxP2pEngineConnect(FlagcxP2pEngine *engine,
                                      const char *ipAddr, int remoteGpuIdx,
                                      int remotePort, bool sameProcess) {
  if (engine == NULL || ipAddr == NULL)
    return NULL;

  const int netDev = chooseEngineNetDev(engine);

  // Step 1: Establish bootstrap P2P connection to remote's bootstrap listen
  // port
  struct flagcxBootstrapHandle bsHandle;
  memset(&bsHandle, 0, sizeof(bsHandle));
  bsHandle.magic = FLAGCX_SOCKET_MAGIC;

  char ipPortStr[256];
  snprintf(ipPortStr, sizeof(ipPortStr), "%s:%d", ipAddr, remotePort);
  if (flagcxSocketGetAddrFromString(&bsHandle.addr, ipPortStr) !=
      flagcxSuccess) {
    return NULL;
  }

  struct bootstrapState *bsConn = NULL;
  if (bootstrapP2pConnect(&bsHandle, FLAGCX_SOCKET_MAGIC, NULL, &bsConn) !=
      flagcxSuccess) {
    return NULL;
  }

  // Step 2: Exchange IB listen handles over bootstrap
  char localIbHandle[FLAGCX_NET_HANDLE_MAXSIZE];
  memcpy(localIbHandle, engine->listeners[netDev].handle,
         FLAGCX_NET_HANDLE_MAXSIZE);

  char remoteIbHandle[FLAGCX_NET_HANDLE_MAXSIZE];
  memset(remoteIbHandle, 0, sizeof(remoteIbHandle));
  if (bootstrapExchange(bsConn, 0, 4, localIbHandle, FLAGCX_NET_HANDLE_MAXSIZE,
                        remoteIbHandle,
                        FLAGCX_NET_HANDLE_MAXSIZE) != flagcxSuccess) {
    bootstrapClose(bsConn);
    return NULL;
  }

  // Step 3: Connect IB adaptor using remote's handle
  void *sendComm = NULL;
  if (engine->adaptor->connect(netDev, remoteIbHandle, &sendComm) !=
      flagcxSuccess) {
    bootstrapClose(bsConn);
    return NULL;
  }

  FlagcxP2pListenHandleView *remoteHandle =
      reinterpret_cast<FlagcxP2pListenHandleView *>(remoteIbHandle);
  const bool sameHost =
      socketAddrSameHost(&remoteHandle->connectAddr, &flagcxIbIfAddr);
  const bool isLocal = sameHost;
  const bool isSameProcess = sameHost && sameProcess;

  // Step 4: Exchange ctrl meta over bootstrap
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
  if (bootstrapExchangeCtrlMeta(bsConn, &localMeta, &remoteMeta) !=
      flagcxSuccess) {
    engine->adaptor->closeSend(sendComm);
    bootstrapClose(bsConn);
    return NULL;
  }

  FlagcxP2pConn *conn = new FlagcxP2pConn;
  conn->engine = engine;
  conn->sendComm = sendComm;
  conn->recvComm = NULL;
  conn->netDev = netDev;
  conn->remoteGpuIdx =
      remoteMeta.gpuIdx >= 0 ? remoteMeta.gpuIdx : remoteGpuIdx;
  conn->remoteNotifPort = remoteMeta.notifPort;
  conn->isLocal =
      isLocal || ((remoteMeta.flags & FLAGCX_P2P_CTRL_FLAG_LOCAL) != 0);
  conn->sameProcess =
      isSameProcess ||
      ((remoteMeta.flags & FLAGCX_P2P_CTRL_FLAG_SAME_PROCESS) != 0);
  conn->notifSockConnected = false;
  memset(&conn->notifSock, 0, sizeof(conn->notifSock));

  if (!conn->sameProcess && remoteMeta.notifPort > 0) {
    connectNotifSocket(conn, &remoteHandle->connectAddr, remoteMeta.notifPort);
  }

  // Step 5: Exchange desc table over bootstrap
  if (bootstrapExchangeDescTable(bsConn, conn) != 0) {
    WARN("NET/IB_P2P : connect desc-table exchange failed");
    flagcxP2pEngineConnDestroy(conn);
    bootstrapClose(bsConn);
    return NULL;
  }

  // Step 6: Close transient bootstrap connection
  bootstrapClose(bsConn);
  return conn;
}

FlagcxP2pConn *flagcxP2pEngineAccept(FlagcxP2pEngine *engine, char *ipAddrBuf,
                                     size_t ipAddrBufLen, int *remoteGpuIdx) {
  if (engine == NULL || ipAddrBuf == NULL || remoteGpuIdx == NULL)
    return NULL;
  if (engine->stopAccept.load(std::memory_order_acquire))
    return NULL;

  const int dev = chooseEngineNetDev(engine);
  if (engine->bsListenState == NULL)
    return NULL;
  if (dev < 0 || dev >= engine->nDevs ||
      engine->listeners[dev].listenComm == NULL)
    return NULL;

  // Step 1: Accept bootstrap P2P connection from connector
  struct bootstrapState *bsConn = NULL;
  if (bootstrapP2pAccept(engine->bsListenState, &bsConn) != flagcxSuccess) {
    return NULL;
  }
  if (engine->stopAccept.load(std::memory_order_acquire)) {
    bootstrapClose(bsConn);
    return NULL;
  }

  // Step 2: Exchange IB listen handles over bootstrap
  char localIbHandle[FLAGCX_NET_HANDLE_MAXSIZE];
  memcpy(localIbHandle, engine->listeners[dev].handle,
         FLAGCX_NET_HANDLE_MAXSIZE);

  char remoteIbHandle[FLAGCX_NET_HANDLE_MAXSIZE];
  memset(remoteIbHandle, 0, sizeof(remoteIbHandle));
  if (bootstrapExchange(bsConn, 0, 4, localIbHandle, FLAGCX_NET_HANDLE_MAXSIZE,
                        remoteIbHandle,
                        FLAGCX_NET_HANDLE_MAXSIZE) != flagcxSuccess) {
    bootstrapClose(bsConn);
    return NULL;
  }

  // Step 3: Accept IB connection using the adaptor
  void *recvComm = NULL;
  if (engine->stopAccept.load(std::memory_order_acquire)) {
    bootstrapClose(bsConn);
    return NULL;
  }
  if (engine->adaptor->accept(engine->listeners[dev].listenComm, &recvComm) !=
      flagcxSuccess) {
    bootstrapClose(bsConn);
    return NULL;
  }

  // Step 4: Exchange ctrl meta over bootstrap
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
  if (bootstrapExchangeCtrlMeta(bsConn, &localMeta, &remoteMeta) !=
      flagcxSuccess) {
    engine->adaptor->closeRecv(recvComm);
    bootstrapClose(bsConn);
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

  // Step 5: Exchange desc table over bootstrap
  if (bootstrapExchangeDescTable(bsConn, conn) != 0) {
    WARN("NET/IB_P2P : accept desc-table exchange failed");
    flagcxP2pEngineConnDestroy(conn);
    bootstrapClose(bsConn);
    return NULL;
  }

  // Step 6: Close transient bootstrap connection
  bootstrapClose(bsConn);
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
    gMrToBaseAddr[mrId] = existing->first;
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
  entry.ptrType = detectPtrTypeAndMaybeCacheIpc(
      reinterpret_cast<void *>(data), entry.ipcHandle, &entry.ipcHandleSize);
  entry.hasIpc = entry.ptrType == FLAGCX_PTR_CUDA && entry.ipcHandleSize > 0;

  if (engine->adaptor->regMr(&devCtx, reinterpret_cast<void *>(data), size,
                             entry.ptrType, FLAGCX_NET_MR_FLAG_NONE,
                             &entry.mhandle) != flagcxSuccess ||
      entry.mhandle == NULL) {
    return -1;
  }

  gMemRegInfo[data] = entry;
  gMrToBaseAddr[entry.mrId] = data;
  mrId = entry.mrId;
  return 0;
}

void flagcxP2pEngineMrDestroy(FlagcxP2pEngine *engine, FlagcxP2pMr mr) {
  if (engine == NULL)
    return;

  std::lock_guard<std::mutex> lock(gMemMutex);
  std::unordered_map<FlagcxP2pMr, uintptr_t>::iterator mrIt =
      gMrToBaseAddr.find(mr);
  if (mrIt == gMrToBaseAddr.end())
    return;

  std::unordered_map<uintptr_t, FlagcxP2pMemRegEntry>::iterator entryIt =
      gMemRegInfo.find(mrIt->second);
  if (entryIt == gMemRegInfo.end()) {
    gMrToBaseAddr.erase(mrIt);
    return;
  }

  struct {
    int ibDevN;
  } devCtx = {entryIt->second.ibDevN};
  engine->adaptor->deregMr(&devCtx, entryIt->second.mhandle);
  gMemRegInfo.erase(entryIt);
  gMrToBaseAddr.erase(mrIt);
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
  if (conn->engine->adaptor->iget(
          conn->sendComm, srcOff, dstOff, size, 0, 0, (void **)&remoteMr,
          (void **)localEntry.mhandle, &request) != flagcxSuccess) {
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
  if (conn == NULL || numIovs <= 0 || transferId == NULL) {
    fprintf(stderr,
            "[FlagCX P2P] ReadVector early exit: invalid args (conn=%p, "
            "numIovs=%d, transferId=%p)\n",
            conn, numIovs, (void *)transferId);
    return -1;
  }

  if (dstVec.size() < static_cast<size_t>(numIovs) ||
      sizeVec.size() < static_cast<size_t>(numIovs) ||
      descs.size() < static_cast<size_t>(numIovs)) {
    fprintf(stderr,
            "[FlagCX P2P] ReadVector early exit: vector length mismatch "
            "(numIovs=%d)\n",
            numIovs);
    return -1;
  }

  if (conn->isLocal && (conn->sameProcess || !ipcBufs.empty())) {
    fprintf(stderr,
            "[FlagCX P2P] ReadVector taking local transfer path: numIovs=%d\n",
            numIovs);
    int rc = startLocalTransfer(conn, dstVec, sizeVec, descs, numIovs,
                                transferId, ipcBufs, false);
    fprintf(stderr, "[FlagCX P2P] ReadVector local transfer returned: rc=%d\n",
            rc);
    return rc;
  }

  if (mrIds.size() < static_cast<size_t>(numIovs)) {
    fprintf(stderr,
            "[FlagCX P2P] ReadVector early exit: mrIds length mismatch "
            "(numIovs=%d)\n",
            numIovs);
    return -1;
  }

  std::vector<FlagcxP2pMemRegEntry> localEntries(numIovs);
  {
    std::lock_guard<std::mutex> memLock(gMemMutex);
    for (int i = 0; i < numIovs; i++) {
      FlagcxP2pMemRegEntry *entry = findMemRegByMr(mrIds[i]);
      if (entry == NULL) {
        fprintf(
            stderr,
            "[FlagCX P2P] ReadVector memReg lookup failed: iov=%d, mr=%lu\n", i,
            (unsigned long)mrIds[i]);
        return -1;
      }

      if (!memRegContains(*entry, reinterpret_cast<uintptr_t>(dstVec[i]),
                          sizeVec[i])) {
        fprintf(stderr,
                "[FlagCX P2P] ReadVector memReg bounds check failed: iov=%d, "
                "mr=%lu, addr=%p, size=%zu\n",
                i, (unsigned long)mrIds[i], dstVec[i], sizeVec[i]);
        return -1;
      }

      localEntries[i] = *entry;
    }
  }

  const int connIbDevN = getCommView(conn->sendComm)->ibDevN;
  PoolTransferTask *task = acquirePoolTask();

  if (!buildAndSubmitToPool(task, dstVec, sizeVec, descs, localEntries, numIovs,
                            conn->sendComm, connIbDevN, FLAGCX_SLICE_OP_READ)) {
    // sentinel so isAllDone() converges (needs total>0)
    auto *sentinel = new FlagcxSlice{
        0, 0, 0, 0, 0, FLAGCX_SLICE_OP_READ, &task->fx, nullptr};
    task->fx.sliceList.push_back(sentinel);
    task->fx.sliceCount.fetch_add(1, std::memory_order_release);
    sentinel->markFailed();
  }

  *transferId = encodePoolXfer(task);
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
                               const std::vector<FlagcxP2pMr> &mrIds,
                               const std::vector<void *> &dstVec,
                               const std::vector<size_t> &sizeVec,
                               const std::vector<FlagcxP2pRdmaDesc> &descs,
                               int numIovs, uint64_t *transferId,
                               const std::vector<char *> &ipcBufs) {
  if (conn == NULL || numIovs <= 0 || transferId == NULL)
    return -1;

  if (dstVec.size() < static_cast<size_t>(numIovs) ||
      sizeVec.size() < static_cast<size_t>(numIovs) ||
      descs.size() < static_cast<size_t>(numIovs))
    return -1;

  if (conn->isLocal && (conn->sameProcess || !ipcBufs.empty())) {
    return startLocalTransfer(conn, dstVec, sizeVec, descs, numIovs, transferId,
                              ipcBufs, true);
  }

  if (mrIds.size() < static_cast<size_t>(numIovs))
    return -1;

  std::vector<FlagcxP2pMemRegEntry> localEntries(numIovs);
  {
    std::lock_guard<std::mutex> memLock(gMemMutex);
    for (int i = 0; i < numIovs; i++) {
      FlagcxP2pMemRegEntry *entry = findMemRegByMr(mrIds[i]);
      if (entry == NULL)
        return -1;

      if (!memRegContains(*entry, reinterpret_cast<uintptr_t>(dstVec[i]),
                          sizeVec[i]))
        return -1;

      localEntries[i] = *entry;
    }
  }

  const int connIbDevN = getCommView(conn->sendComm)->ibDevN;
  PoolTransferTask *task = acquirePoolTask();

  if (!buildAndSubmitToPool(task, dstVec, sizeVec, descs, localEntries, numIovs,
                            conn->sendComm, connIbDevN,
                            FLAGCX_SLICE_OP_WRITE)) {
    auto *sentinel = new FlagcxSlice{
        0, 0, 0, 0, 0, FLAGCX_SLICE_OP_WRITE, &task->fx, nullptr};
    task->fx.sliceList.push_back(sentinel);
    task->fx.sliceCount.fetch_add(1, std::memory_order_release);
    sentinel->markFailed();
  }

  *transferId = encodePoolXfer(task);
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

  if (PoolTransferTask *task = decodePoolXfer(transferId)) {
    if (!task->fx.isAllDone())
      return false;
    if (task->fx.hasErrors()) {
      WARN("NET/IB_P2P : transfer completed with %lu failed slices",
           (unsigned long)task->fx.failedCount.load(std::memory_order_relaxed));
    }
    finalizePoolTask(task);
    releasePoolTask(task);
    return true;
  }

  // Fall through to legacy synchronous xfer map (for single Read/Write)
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

  // After bootstrap P2P integration, metadata must expose the bootstrap listen
  // port (used by flagcxP2pEngineConnect for the initial handshake), not the
  // RDMA listen port (which is now exchanged during the bootstrap handshake).
  if (engine->bsListenState == NULL || engine->bsListenPort <= 0)
    return -1;

  union flagcxSocketAddress bsAddr;
  flagcxSocketGetAddr(&engine->bsListenState->p2p->sock, &bsAddr);
  const std::string rdmaAddr = socketAddrToHostPortString(&bsAddr);
  if (rdmaAddr.empty())
    return -1;

  const std::string result = rdmaAddr + "?" +
                             std::to_string(engine->localGpuIdx) + "?" +
                             std::to_string(engine->notifListenPort);
  *metadataStr = new char[result.length() + 1];
  std::strcpy(*metadataStr, result.c_str());
  return 0;
}

/* ================================================================== */
/*  RPC control-plane service                                         */
/* ================================================================== */

int flagcxP2pEngineGetRpcPort(FlagcxP2pEngine *engine) {
  if (engine == NULL)
    return -1;
  // Return bootstrap P2P listen port for RPC metadata exchange
  if (engine->bsListenState != NULL && engine->bsListenPort > 0)
    return engine->bsListenPort;
  // Fallback to IB listen port if bootstrap not available
  const int netDev = chooseEngineNetDev(engine);
  if (engine->listeners[netDev].listenComm == NULL)
    return -1;
  FlagcxP2pListenHandleView *listenHandle =
      reinterpret_cast<FlagcxP2pListenHandleView *>(
          engine->listeners[netDev].handle);
  return static_cast<int>(socketAddrPort(&listenHandle->connectAddr));
}

int flagcxP2pEngineStartRpcServer(FlagcxP2pEngine *engine) {
  if (engine == NULL)
    return -1;
  bool expected = false;
  if (!engine->rpcServerActive.compare_exchange_strong(expected, true))
    return 0; // already running

  engine->rpcServerThread = std::thread([engine]() {
    char ipBuf[256];
    while (!engine->stopRpcServer.load(std::memory_order_acquire)) {
      int remoteGpu = -1;
      FlagcxP2pConn *conn =
          flagcxP2pEngineAccept(engine, ipBuf, sizeof(ipBuf), &remoteGpu);
      if (engine->stopRpcServer.load(std::memory_order_acquire)) {
        if (conn != NULL)
          flagcxP2pEngineConnDestroy(conn);
        break;
      }
      if (conn == NULL)
        continue;
      std::lock_guard<std::mutex> lock(engine->acceptedMutex);
      engine->acceptedConns.push_back(conn);
    }
    engine->rpcServerActive.store(false, std::memory_order_release);
  });
  INFO(FLAGCX_INIT, "NET/IB_P2P : RPC server thread started (port=%d)",
       flagcxP2pEngineGetRpcPort(engine));
  return 0;
}

FlagcxP2pConn *flagcxP2pEngineGetConn(FlagcxP2pEngine *engine,
                                      const char *session) {
  if (engine == NULL || session == NULL)
    return NULL;

  const std::string key(session);
  {
    std::lock_guard<std::mutex> lock(engine->sessionMutex);
    std::unordered_map<std::string, FlagcxP2pConn *>::iterator it =
        engine->sessionConns.find(key);
    if (it != engine->sessionConns.end())
      return it->second;
  }

  // Parse "host:port" (split on the last ':' to tolerate IPv6 forms).
  const size_t pos = key.rfind(':');
  if (pos == std::string::npos)
    return NULL;
  std::string host = key.substr(0, pos);
  const int port = atoi(key.substr(pos + 1).c_str());
  if (host.size() >= 2 && host.front() == '[' && host.back() == ']')
    host = host.substr(1, host.size() - 2);

  FlagcxP2pConn *conn =
      flagcxP2pEngineConnect(engine, host.c_str(), -1, port, false);
  if (conn == NULL)
    return NULL;

  std::lock_guard<std::mutex> lock(engine->sessionMutex);
  std::unordered_map<std::string, FlagcxP2pConn *>::iterator it =
      engine->sessionConns.find(key);
  if (it != engine->sessionConns.end()) {
    // Lost a race; keep the existing one.
    flagcxP2pEngineConnDestroy(conn);
    return it->second;
  }
  engine->sessionConns[key] = conn;
  return conn;
}

int flagcxP2pEngineMakeDesc(FlagcxP2pConn *conn, uint64_t remoteVa,
                            uint32_t size, FlagcxP2pRdmaDesc *desc) {
  if (conn == NULL || desc == NULL)
    return -1;
  for (size_t i = 0; i < conn->remoteRegions.size(); i++) {
    const FlagcxP2pRemoteRegion &r = conn->remoteRegions[i];
    if (remoteVa >= r.baseAddr && remoteVa + size <= r.baseAddr + r.size) {
      memset(desc, 0, sizeof(*desc));
      desc->addr = remoteVa;
      desc->size = size;
      desc->rkey = r.rkey;
      return 0;
    }
  }
  return -1;
}

int flagcxP2pEngineWriteVectorSync(
    FlagcxP2pConn *conn, const std::vector<FlagcxP2pMr> &mrIds,
    const std::vector<void *> &srcVec, const std::vector<size_t> &sizeVec,
    const std::vector<FlagcxP2pRdmaDesc> &descs) {
  if (conn == NULL)
    return -1;
  const int numIovs = static_cast<int>(srcVec.size());
  if (numIovs <= 0)
    return 0;

  uint64_t transferId = 0;
  const int rc = flagcxP2pEngineWriteVector(conn, mrIds, srcVec, sizeVec, descs,
                                            numIovs, &transferId);
  if (rc != 0)
    return rc;

  PoolTransferTask *task = decodePoolXfer(transferId);
  if (task) {
    uint64_t spins = 0;
    while (!task->fx.isAllDone()) {
      if ((++spins & 0xFFF) == 0) {
        std::this_thread::yield();
        continue;
      }
#if defined(__x86_64__) || defined(__i386__)
      __builtin_ia32_pause();
#endif
    }
    flagcxP2pEngineXferStatus(conn, transferId);
    return 0;
  }

  // Fallback (e.g. the local/IPC path stores the transfer in the legacy
  // xfer map): poll status, but yield instead of a fixed sleep.
  while (!flagcxP2pEngineXferStatus(conn, transferId)) {
    std::this_thread::yield();
  }
  return 0;
}

/* ================================================================== */
/*  C-ABI facade for ctypes(experimental)                             */
/* ================================================================== */
extern "C" {

void *flagcxP2pRpcEngineCreate(void) {
  return reinterpret_cast<void *>(flagcxP2pEngineCreate());
}

void flagcxP2pRpcEngineDestroy(void *engine) {
  flagcxP2pEngineDestroy(reinterpret_cast<FlagcxP2pEngine *>(engine));
}

int flagcxP2pRpcGetPort(void *engine) {
  return flagcxP2pEngineGetRpcPort(reinterpret_cast<FlagcxP2pEngine *>(engine));
}

int flagcxP2pRpcStartServer(void *engine) {
  return flagcxP2pEngineStartRpcServer(
      reinterpret_cast<FlagcxP2pEngine *>(engine));
}

int flagcxP2pRpcRegister(void *engine, uint64_t addr, uint64_t size,
                         uint64_t *mrIdOut) {
  if (mrIdOut == NULL)
    return -1;
  FlagcxP2pMr mrId = 0;
  const int rc = flagcxP2pEngineReg(reinterpret_cast<FlagcxP2pEngine *>(engine),
                                    static_cast<uintptr_t>(addr),
                                    static_cast<size_t>(size), mrId);
  if (rc != 0)
    return rc;
  *mrIdOut = mrId;
  return 0;
}

void *flagcxP2pRpcGetConn(void *engine, const char *session) {
  return reinterpret_cast<void *>(flagcxP2pEngineGetConn(
      reinterpret_cast<FlagcxP2pEngine *>(engine), session));
}

int flagcxP2pRpcBatchWriteSync(void *connPtr, int count, const uint64_t *srcVa,
                               const uint64_t *dstVa, const uint64_t *sizes) {
  FlagcxP2pConn *conn = reinterpret_cast<FlagcxP2pConn *>(connPtr);
  if (conn == NULL || count < 0)
    return -1;
  if (count == 0)
    return 0;
  if (srcVa == NULL || dstVa == NULL || sizes == NULL)
    return -1;

  std::vector<void *> srcVec(count);
  std::vector<size_t> sizeVec(count);
  std::vector<FlagcxP2pRdmaDesc> descs(count);

  // Resolve every remote rkey/desc up front. No global lock here: MakeDesc
  // scans the per-conn remoteRegions table, not gMemRegInfo.
  for (int i = 0; i < count; i++) {
    srcVec[i] = reinterpret_cast<void *>(static_cast<uintptr_t>(srcVa[i]));
    sizeVec[i] = static_cast<size_t>(sizes[i]);
    if (flagcxP2pEngineMakeDesc(conn, dstVa[i], static_cast<uint32_t>(sizes[i]),
                                &descs[i]) != 0) {
      WARN("NET/IB_P2P : BatchWriteSync MakeDesc failed for remote VA "
           "0x%llx size %llu",
           (unsigned long long)dstVa[i], (unsigned long long)sizes[i]);
      return -1;
    }
  }

  if (conn->isLocal && conn->sameProcess) {
    std::vector<FlagcxP2pMr> mrVec(count);
    {
      std::lock_guard<std::mutex> memLock(gMemMutex);
      for (int i = 0; i < count; i++) {
        FlagcxP2pMemRegEntry localEntry;
        if (!findMemReg(static_cast<uintptr_t>(srcVa[i]), &localEntry)) {
          WARN("NET/IB_P2P : BatchWriteSync no local MR for source VA 0x%llx",
               (unsigned long long)srcVa[i]);
          return -1;
        }
        mrVec[i] = localEntry.mrId;
      }
    }
    return flagcxP2pEngineWriteVectorSync(conn, mrVec, srcVec, sizeVec, descs);
  }

  std::vector<FlagcxP2pMemRegEntry> localEntries(count);
  {
    std::lock_guard<std::mutex> memLock(gMemMutex);
    for (int i = 0; i < count; i++) {
      if (!findMemReg(static_cast<uintptr_t>(srcVa[i]), &localEntries[i])) {
        WARN("NET/IB_P2P : BatchWriteSync no local MR for source VA 0x%llx",
             (unsigned long long)srcVa[i]);
        return -1;
      }
      if (!memRegContains(localEntries[i], static_cast<uintptr_t>(srcVa[i]),
                          static_cast<size_t>(sizes[i]))) {
        WARN("NET/IB_P2P : BatchWriteSync source VA 0x%llx size %llu out of MR "
             "bounds",
             (unsigned long long)srcVa[i], (unsigned long long)sizes[i]);
        return -1;
      }
    }
  }

  const int connIbDevN = getCommView(conn->sendComm)->ibDevN;
  PoolTransferTask *task = acquirePoolTask();
  if (!buildAndSubmitToPool(task, srcVec, sizeVec, descs, localEntries, count,
                            conn->sendComm, connIbDevN,
                            FLAGCX_SLICE_OP_WRITE)) {
    auto *sentinel = new FlagcxSlice{
        0, 0, 0, 0, 0, FLAGCX_SLICE_OP_WRITE, &task->fx, nullptr};
    task->fx.sliceList.push_back(sentinel);
    task->fx.sliceCount.fetch_add(1, std::memory_order_release);
    sentinel->markFailed();
  }

  const uint64_t xferId = encodePoolXfer(task);

  // Synchronously wait for completion (mirror flagcxP2pEngineWriteVectorSync).
  uint64_t spins = 0;
  while (!task->fx.isAllDone()) {
    if ((++spins & 0xFFF) == 0) {
      std::this_thread::yield();
      continue;
    }
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
  }
  flagcxP2pEngineXferStatus(conn, xferId);
  return 0;
}

} // extern "C"

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
  if (flagcxSocketSend(&conn->notifSock, &wireMsg, sizeof(wireMsg)) !=
      flagcxSuccess) {
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
