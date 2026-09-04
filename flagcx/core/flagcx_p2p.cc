/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX P2P Engine — implements the flagcx_p2p.h API.
 *
 * Architecture: the common engine owns Bootstrap control exchange, locality
 * detection, IPC selection and transfer completion. IBRC P2P and ACCL/BAREX
 * are selected only as network data-plane adaptors.
 ************************************************************************/

#include "flagcx_p2p.h"

#include "adaptor.h"
#include "bootstrap.h"
#include "debug.h"
#include "flagcx_mr_registry.h"
#include "flagcx_net.h"
#include "flagcx_net_adaptor.h"
#include "onesided.h"
#include "p2p_topo.h"
#include "p2p_transport.h"
#include "param.h"
#include "socket.h"
#include "utils.h"

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
#include <strings.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#if defined(__linux__)
#include <sys/epoll.h>
#endif
#include <unistd.h>

extern struct flagcxNetAdaptor flagcxP2pNetIb;
extern flagcxResult_t flagcxP2pNetIbAbortListen(void *listenComm);
#ifdef USE_ACCL_BAREX
extern struct flagcxNetAdaptor flagcxP2pNetBarex;
#endif

namespace {

/* Keep the engine compatible with the library's C++11 host build while
 * preserving concurrent readers on the transfer and MR lookup paths. */
class FlagcxP2pSharedMutex {
public:
  FlagcxP2pSharedMutex() {
    if (pthread_rwlock_init(&rwlock_, nullptr) != 0)
      std::abort();
  }

  ~FlagcxP2pSharedMutex() { pthread_rwlock_destroy(&rwlock_); }

  FlagcxP2pSharedMutex(const FlagcxP2pSharedMutex &) = delete;
  FlagcxP2pSharedMutex &operator=(const FlagcxP2pSharedMutex &) = delete;

  void lock() {
    if (pthread_rwlock_wrlock(&rwlock_) != 0)
      std::abort();
  }

  void unlock() {
    if (pthread_rwlock_unlock(&rwlock_) != 0)
      std::abort();
  }

  void lockShared() {
    if (pthread_rwlock_rdlock(&rwlock_) != 0)
      std::abort();
  }

  void unlockShared() {
    if (pthread_rwlock_unlock(&rwlock_) != 0)
      std::abort();
  }

private:
  pthread_rwlock_t rwlock_;
};

class FlagcxP2pSharedLock {
public:
  explicit FlagcxP2pSharedLock(FlagcxP2pSharedMutex &mutex)
      : mutex_(&mutex), owns_(false) {
    lock();
  }

  FlagcxP2pSharedLock(FlagcxP2pSharedMutex &mutex, std::defer_lock_t)
      : mutex_(&mutex), owns_(false) {}

  ~FlagcxP2pSharedLock() {
    if (owns_)
      mutex_->unlockShared();
  }

  FlagcxP2pSharedLock(const FlagcxP2pSharedLock &) = delete;
  FlagcxP2pSharedLock &operator=(const FlagcxP2pSharedLock &) = delete;

  void lock() {
    if (owns_)
      std::abort();
    mutex_->lockShared();
    owns_ = true;
  }

private:
  FlagcxP2pSharedMutex *mutex_;
  bool owns_;
};

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
FLAGCX_PARAM(P2pQuiesceTimeoutMs, "P2P_QUIESCE_TIMEOUT_MS", 5000);
FLAGCX_PARAM(MrSortedLookup, "MR_SORTED_LOOKUP", 0);

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
  c.workersPerPool = std::min(c.workersPerPool, c.qpsPerConn);
  c.shardCount =
      clampParam<int>(flagcxParamP2pShardCount(), 1, 64, 8, "P2P_SHARD_COUNT");
  c.shardCount = std::max(c.shardCount, c.workersPerPool);
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

enum {
  FLAGCX_P2P_CTRL_VERSION = 4,
  FLAGCX_P2P_MAX_NOTIF_PEERS = 64,
  FLAGCX_P2P_IPC_HANDLE_BYTES = 64,
  FLAGCX_P2P_NOTIF_MAGIC = 0xDEADDEADu,
  FLAGCX_P2P_CTRL_FLAG_SAME_PROCESS = 1u << 1,
  FLAGCX_P2P_IPC_FLAG_CUDA = 1u << 0,
  FLAGCX_P2P_MAX_NET_DEVS = 128,
};

enum FlagcxP2pNotifType : uint32_t {
  FLAGCX_P2P_NOTIF_USER = 0,
  FLAGCX_P2P_NOTIF_MR_ADD = 1,
  FLAGCX_P2P_NOTIF_MR_REMOVE = 2,
  FLAGCX_P2P_NOTIF_MR_ACK = 3,
};

static struct flagcxNetAdaptor *getP2pNetAdaptor() {
#ifdef USE_ACCL_BAREX
  return &flagcxP2pNetBarex;
#else
  return &flagcxP2pNetIb;
#endif
}

static std::atomic<uint64_t> gNextP2pEndpointId{1};

static uint64_t allocateP2pEndpointId() {
  const uint64_t sequence =
      gNextP2pEndpointId.fetch_add(1, std::memory_order_relaxed);
  return getPidHash() ^ (sequence * 0x9E3779B97F4A7C15ull);
}

struct FlagcxP2pMrChunk {
  uintptr_t baseAddr;
  size_t size;
  void *adaptorMrHandle;
  struct flagcxNetMrInfo mrInfo;
};

/* One public FlagcxP2pMr may consist of several transport registrations.
 * This wrapper is Engine-private and is the value stored in the MR registry;
 * adaptors only ever receive a chunk's adaptorMrHandle. */
struct FlagcxP2pRegisteredMemory {
  uintptr_t baseAddr;
  size_t size;
  int ibDevN;
  int ptrType;
  std::vector<FlagcxP2pMrChunk> chunks;
};

static flagcxResult_t getP2pMrInfo(void *mhandle,
                                   struct flagcxNetMrInfo *info) {
  auto *registration = static_cast<FlagcxP2pRegisteredMemory *>(mhandle);
  if (registration == NULL || info == NULL || registration->chunks.empty())
    return flagcxInvalidArgument;
  *info = registration->chunks.front().mrInfo;
  return flagcxSuccess;
}

static void fillDescKeys(FlagcxP2pRdmaDesc *desc,
                         const struct flagcxNetMrInfo &mrInfo) {
  if (desc == NULL)
    return;
  const uint32_t nKeys =
      std::min<uint32_t>(mrInfo.nKeys, FLAGCX_NET_MAX_MR_KEYS);
  desc->rkey = nKeys > 0 ? mrInfo.rkeys[0] : 0;
  desc->nmsgs = nKeys;
  memset(desc->padding, 0, sizeof(desc->padding));
  for (uint32_t i = 1; i < nKeys; ++i) {
    memcpy(desc->padding + (i - 1) * sizeof(uint32_t), &mrInfo.rkeys[i],
           sizeof(uint32_t));
  }
}

static void extractDescMrInfo(const FlagcxP2pRdmaDesc &desc,
                              struct flagcxNetMrInfo *mrInfo) {
  memset(mrInfo, 0, sizeof(*mrInfo));
  mrInfo->nKeys = desc.nmsgs == 0
                      ? 1
                      : std::min<uint32_t>(desc.nmsgs, FLAGCX_NET_MAX_MR_KEYS);
  mrInfo->rkeys[0] = desc.rkey;
  for (uint32_t i = 1; i < mrInfo->nKeys; ++i) {
    memcpy(&mrInfo->rkeys[i], desc.padding + (i - 1) * sizeof(uint32_t),
           sizeof(uint32_t));
  }
}

static_assert(FLAGCX_P2P_IPC_HANDLE_BYTES == FLAGCX_MR_IPC_HANDLE_BYTES,
              "IPC handle size mismatch between P2P and MR registry");

struct FlagcxP2pCtrlMeta {
  uint32_t version;
  int32_t gpuIdx;
  int32_t notifPort;
  uint32_t flags;
  uint64_t hostHash;
  uint64_t pidHash;
  uint64_t endpointId;
};
static_assert(sizeof(FlagcxP2pCtrlMeta) == 40,
              "FlagcxP2pCtrlMeta size must be stable");

struct FlagcxP2pRemoteRegion {
  uint64_t baseAddr;
  uint64_t size;
  uint64_t ownerEndpointId;
  uint64_t mrId;
  int32_t ptrType;
  struct flagcxNetMrInfo mrInfo;
};

struct FlagcxP2pMemRegWire {
  uint64_t baseAddr;
  uint64_t size;
  uint64_t mrId;
  uint32_t nKeys;
  uint32_t rkeys[FLAGCX_NET_MAX_MR_KEYS];
  int32_t ptrType;
};
static_assert(sizeof(FlagcxP2pMemRegWire) == 64,
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
  uint32_t type;
  FlagcxP2pNotifyMsg payload;
};

struct FlagcxP2pMrCtrlPayload {
  uint64_t senderEndpointId;
  uint64_t sequence;
  int32_t status;
  uint32_t reserved;
  FlagcxP2pMemRegWire region;
};
static_assert(sizeof(FlagcxP2pMrCtrlPayload) <= sizeof(FlagcxP2pNotifyMsg),
              "MR control payload must fit notification wire payload");

struct FlagcxP2pPendingMrUpdate {
  FlagcxP2pNotifType type;
  FlagcxP2pMrCtrlPayload payload;
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

class FlagcxP2pScheduler;
struct FlagcxP2pConn;
struct FlagcxP2pEngineState;

struct FlagcxP2pCtrlAckState {
  bool done = false;
  int status = -1;
};

struct FlagcxP2pEngine {
  struct flagcxNetAdaptor *adaptor;
  const struct flagcxP2pTransportOps *transportOps;
  FlagcxP2pScheduler *scheduler;
  struct flagcxP2pTopoManager *topoMgr;
  int nDevs;
  int localGpuIdx;
  uint64_t endpointId;
  std::atomic<uint64_t> nextCtrlSequence{1};
  FlagcxP2pEngineState *state;
  std::atomic<bool> quarantineRequired{false};
  FlagcxP2pListener listeners[FLAGCX_P2P_MAX_NET_DEVS];

  std::unordered_set<FlagcxP2pConn *> connections;
  FlagcxP2pSharedMutex connectionLifetimeMutex;
  std::mutex connectionMutex;
  std::mutex mrPublishMutex;
  std::unordered_map<uint64_t, FlagcxP2pCtrlAckState> ctrlAcks;
  std::mutex ctrlAckMutex;
  std::condition_variable ctrlAckCv;

  struct flagcxSocket notifListenSock;
  bool notifListenActive;
  int notifListenPort;
  std::thread notifThread;
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

static std::mutex gP2pEngineDirectoryMutex;
static std::unordered_map<uint64_t, FlagcxP2pEngine *> gP2pEngineDirectory;

static bool usesIbP2pAdaptor(const FlagcxP2pEngine *engine) {
  return engine != NULL && engine->adaptor == &flagcxP2pNetIb;
}

static const struct flagcxP2pTransportOps *
getP2pTransportOps(const struct flagcxNetAdaptor *adaptor) {
#ifdef USE_ACCL_BAREX
  if (adaptor == &flagcxP2pNetBarex)
    return &flagcxP2pBarexTransportOps;
#endif
  return adaptor == &flagcxP2pNetIb ? &flagcxP2pIbrcTransportOps : NULL;
}

struct FlagcxP2pConn {
  FlagcxP2pEngine *engine;
  void *sendComm;
  void *recvComm;
  std::atomic<bool> closing{false};
  int netDev;
  int remoteGpuIdx;
  int remoteNotifPort;
  bool isLocal;
  bool sameProcess;
  uint64_t remoteEndpointId;
  union flagcxSocketAddress peerBootstrapAddr;
  struct flagcxSocket notifSock;
  bool notifSockConnected;
  bool mrPublishReady;
  bool mrHandshakeComplete;
  std::vector<FlagcxP2pRemoteRegion> remoteRegions;
  std::vector<FlagcxP2pPendingMrUpdate> pendingMrUpdates;
  mutable FlagcxP2pSharedMutex remoteRegionsMutex;
  std::mutex notifSendMutex;
};

struct FlagcxP2pMemRegEntry {
  FlagcxP2pMr mrId;
  void *mhandle;
  uintptr_t baseAddr;
  size_t size;
  int ibDevN;
  int ptrType;
  struct flagcxNetMrInfo mrInfo;
  bool hasIpc;
  uint32_t ipcHandleSize;
  alignas(8) char ipcHandle[FLAGCX_P2P_IPC_HANDLE_BYTES];
  char descBuf[FLAGCX_P2P_DESC_SIZE];
};

static size_t p2pMrChunkBytes(const FlagcxP2pEngine *engine, size_t size) {
#ifdef USE_ACCL_BAREX
  if (engine != NULL && engine->adaptor == &flagcxP2pNetBarex) {
    size_t chunkBytes = 64ull << 20;
    const char *value = flagcxGetEnv("FLAGCX_ACCL_MAX_MR_MB");
    if (value != NULL) {
      const unsigned long long mb = strtoull(value, NULL, 10);
      chunkBytes = mb == 0 ? size : (size_t)mb << 20;
    }
    return chunkBytes == 0 ? size : chunkBytes;
  }
#endif
  return size;
}

static flagcxResult_t
registerP2pMemory(FlagcxP2pEngine *engine, int ibDevN, uintptr_t data,
                  size_t size, int ptrType,
                  FlagcxP2pRegisteredMemory **registrationOut) {
  if (engine == NULL || engine->adaptor == NULL || registrationOut == NULL ||
      data == 0 || size == 0 || engine->adaptor->regMr == NULL ||
      engine->adaptor->deregMr == NULL || engine->adaptor->getMrInfo == NULL)
    return flagcxInvalidArgument;
  *registrationOut = NULL;

  auto *registration = new FlagcxP2pRegisteredMemory;
  registration->baseAddr = data;
  registration->size = size;
  registration->ibDevN = ibDevN;
  registration->ptrType = ptrType;

  struct {
    int ibDevN;
  } devCtx = {ibDevN};
  const size_t chunkBytes = p2pMrChunkBytes(engine, size);
  for (size_t offset = 0; offset < size;) {
    const size_t chunkSize = std::min(chunkBytes, size - offset);
    FlagcxP2pMrChunk chunk;
    memset(&chunk, 0, sizeof(chunk));
    chunk.baseAddr = data + offset;
    chunk.size = chunkSize;
    flagcxResult_t result = engine->adaptor->regMr(
        &devCtx, reinterpret_cast<void *>(chunk.baseAddr), chunk.size, ptrType,
        FLAGCX_NET_MR_FLAG_NONE, &chunk.adaptorMrHandle);
    if (result != flagcxSuccess || chunk.adaptorMrHandle == NULL ||
        engine->adaptor->getMrInfo(chunk.adaptorMrHandle, &chunk.mrInfo) !=
            flagcxSuccess ||
        chunk.mrInfo.nKeys == 0) {
      if (chunk.adaptorMrHandle != NULL)
        engine->adaptor->deregMr(&devCtx, chunk.adaptorMrHandle);
      for (auto it = registration->chunks.rbegin();
           it != registration->chunks.rend(); ++it)
        engine->adaptor->deregMr(&devCtx, it->adaptorMrHandle);
      delete registration;
      return result == flagcxSuccess ? flagcxInternalError : result;
    }
    registration->chunks.push_back(chunk);
    offset += chunkSize;
  }

  *registrationOut = registration;
  return flagcxSuccess;
}

static void deregisterP2pMemory(FlagcxP2pEngine *engine,
                                FlagcxP2pRegisteredMemory *registration) {
  if (engine == NULL || registration == NULL)
    return;
  struct {
    int ibDevN;
  } devCtx = {registration->ibDevN};
  for (auto it = registration->chunks.rbegin();
       it != registration->chunks.rend(); ++it) {
    if (it->adaptorMrHandle != NULL)
      engine->adaptor->deregMr(&devCtx, it->adaptorMrHandle);
  }
  delete registration;
}

enum FlagcxP2pXferKind {
  FLAGCX_P2P_XFER_NET = 0,
  FLAGCX_P2P_XFER_IPC = 1,
};

enum FlagcxP2pDataPath {
  FLAGCX_P2P_PATH_NET = 0,
  FLAGCX_P2P_PATH_LOCAL_DIRECT = 1,
  FLAGCX_P2P_PATH_IPC = 2,
};

static FlagcxP2pDataPath selectP2pDataPath(const FlagcxP2pConn *conn,
                                           bool hasIpcMetadata) {
  if (conn == NULL || !conn->isLocal)
    return FLAGCX_P2P_PATH_NET;
  if (conn->sameProcess)
    return FLAGCX_P2P_PATH_LOCAL_DIRECT;
  if (hasIpcMetadata)
    return FLAGCX_P2P_PATH_IPC;
  return FLAGCX_P2P_PATH_NET;
}

struct FlagcxP2pNetTask {
  FlagcxP2pConn *conn;
  std::vector<struct flagcxP2pTransportSlice> slices;
  size_t nextSlice;
  std::vector<void *> inflight;
  std::atomic<bool> quiescing;
  std::atomic<bool> done;
  std::atomic<bool> failed;

  FlagcxP2pNetTask()
      : conn(NULL), nextSlice(0), quiescing(false), done(false), failed(false) {
  }
};

struct FlagcxP2pXfer {
  FlagcxP2pXferKind kind;
  std::vector<void *> requests;
  std::shared_ptr<FlagcxP2pNetTask> netTask;
  FlagcxP2pConn *conn;
  int total;
  int completed;
  flagcxStream_t stream;
  flagcxEvent_t event;
  std::vector<void *> openedIpcPtrs;
};

struct FlagcxP2pEngineState {
  std::unordered_map<uintptr_t, FlagcxP2pMemRegEntry> memRegInfo;
  std::unordered_map<FlagcxP2pMr, uintptr_t> mrToBaseAddr;
  std::mutex memMutex;
  uint64_t nextMrId = 1;
  std::mutex mrLifecycleMutex;
  struct flagcxMrRegistry *mrRegistry = NULL;

  std::unordered_map<uint64_t, FlagcxP2pXfer> xfers;
  std::mutex xferMutex;
  uint64_t nextXferId = 1;
};

static std::vector<FlagcxP2pNotifyMsg> &notifyList() {
  static std::vector<FlagcxP2pNotifyMsg> list;
  return list;
}

static std::mutex &notifyMutex() {
  static std::mutex mu;
  return mu;
}

#define gNotifyList notifyList()
#define gNotifyMutex notifyMutex()

class FlagcxP2pScheduler {
public:
  explicit FlagcxP2pScheduler(FlagcxP2pEngine *engine) : engine_(engine) {
    const auto &config = flagcxP2pGlobalConfig();
    numShards_ = std::max(1, config.shardCount);
    numWorkers_ = std::max(1, std::min(config.workersPerPool, numShards_));
    queues_.resize(numShards_);
    for (int worker = 0; worker < numWorkers_; ++worker)
      workers_.emplace_back(&FlagcxP2pScheduler::workerLoop, this, worker);
  }

  ~FlagcxP2pScheduler() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stopping_ = true;
      for (auto &queue : queues_) {
        for (const auto &task : queue) {
          task->failed.store(true, std::memory_order_release);
          task->done.store(true, std::memory_order_release);
          retireTaskLocked(task);
        }
        queue.clear();
      }
    }
    cv_.notify_all();
    for (std::thread &worker : workers_) {
      if (worker.joinable())
        worker.join();
    }
  }

  flagcxResult_t
  submit(FlagcxP2pConn *conn,
         const std::vector<struct flagcxP2pTransportSlice> &slices,
         std::shared_ptr<FlagcxP2pNetTask> *taskOut) {
    if (conn == NULL || slices.empty() || taskOut == NULL)
      return flagcxInvalidArgument;
    *taskOut = std::shared_ptr<FlagcxP2pNetTask>();
    auto task = std::make_shared<FlagcxP2pNetTask>();
    task->conn = conn;
    task->slices = slices;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (stopping_ || conn->closing.load(std::memory_order_acquire) ||
          closingConnections_.count(conn) != 0)
        return flagcxInternalError;
      connectionTaskCounts_[conn]++;
      queues_[shardFor(*task)].push_back(task);
    }
    cv_.notify_all();
    *taskOut = task;
    return flagcxSuccess;
  }

  bool quiesceConnection(FlagcxP2pConn *conn) {
    if (conn == NULL)
      return true;
    conn->closing.store(true, std::memory_order_release);
    std::unique_lock<std::mutex> lock(mu_);
    closingConnections_.insert(conn);
    for (auto &queue : queues_) {
      for (const auto &task : queue) {
        if (task->conn == conn)
          task->quiescing.store(true, std::memory_order_release);
      }
    }
    cv_.notify_all();
    const auto timeout = std::chrono::milliseconds(
        std::max<int64_t>(1, flagcxParamP2pQuiesceTimeoutMs()));
    if (drainCv_.wait_for(lock, timeout, [&] {
          return connectionTaskCounts_.find(conn) ==
                 connectionTaskCounts_.end();
        })) {
      closingConnections_.erase(conn);
      return true;
    }

    /* A native request may never complete after a link or callback failure.
       Stop polling it and quarantine its connection instead of blocking
       teardown forever. The owning Engine deliberately retains the task,
       transport comm and MRs so late native completion cannot touch freed
       state. */
    abandoningConnections_.insert(conn);
    for (auto &queue : queues_) {
      for (auto it = queue.begin(); it != queue.end();) {
        const auto &task = *it;
        if (task->conn != conn) {
          ++it;
          continue;
        }
        task->failed.store(true, std::memory_order_release);
        task->done.store(true, std::memory_order_release);
        retireTaskLocked(task);
        it = queue.erase(it);
      }
    }
    WARN("P2P/ENGINE : connection quiesce timed out after %lld ms; "
         "quarantining transport resources",
         (long long)timeout.count());
    return false;
  }

private:
  size_t shardFor(const FlagcxP2pNetTask &task) const {
    return (reinterpret_cast<uintptr_t>(task.conn->sendComm) >> 6) %
           (size_t)numShards_;
  }

  void enqueue(const std::shared_ptr<FlagcxP2pNetTask> &task) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (stopping_) {
        task->failed.store(true, std::memory_order_release);
        task->done.store(true, std::memory_order_release);
        retireTaskLocked(task);
        return;
      }
      if (abandoningConnections_.count(task->conn) != 0) {
        task->failed.store(true, std::memory_order_release);
        task->done.store(true, std::memory_order_release);
        retireTaskLocked(task);
        return;
      }
      if (closingConnections_.count(task->conn) != 0 ||
          task->conn->closing.load(std::memory_order_acquire))
        task->quiescing.store(true, std::memory_order_release);
      queues_[shardFor(*task)].push_back(task);
    }
    cv_.notify_all();
  }

  void retireTaskLocked(const std::shared_ptr<FlagcxP2pNetTask> &task) {
    auto it = connectionTaskCounts_.find(task->conn);
    if (it == connectionTaskCounts_.end())
      return;
    if (--it->second == 0) {
      connectionTaskCounts_.erase(it);
      drainCv_.notify_all();
    }
  }

  void retireTask(const std::shared_ptr<FlagcxP2pNetTask> &task) {
    std::lock_guard<std::mutex> lock(mu_);
    retireTaskLocked(task);
  }

  bool hasWorkFor(int worker) const {
    for (int shard = worker; shard < numShards_; shard += numWorkers_) {
      if (!queues_[shard].empty())
        return true;
    }
    return false;
  }

  std::shared_ptr<FlagcxP2pNetTask> popFor(int worker) {
    for (int shard = worker; shard < numShards_; shard += numWorkers_) {
      if (!queues_[shard].empty()) {
        auto task = queues_[shard].front();
        queues_[shard].pop_front();
        return task;
      }
    }
    return std::shared_ptr<FlagcxP2pNetTask>();
  }

  void progressTask(const std::shared_ptr<FlagcxP2pNetTask> &task) {
    const struct flagcxP2pTransportOps *ops = engine_->transportOps;
    if (ops == NULL || ops->getCaps == NULL || ops->submitBatch == NULL ||
        ops->test == NULL) {
      task->failed.store(true, std::memory_order_release);
      task->done.store(true, std::memory_order_release);
      return;
    }

    if (task->quiescing.load(std::memory_order_acquire) ||
        task->conn->closing.load(std::memory_order_acquire))
      task->nextSlice = task->slices.size();

    if (ops->progress != NULL &&
        ops->progress(task->conn->sendComm) != flagcxSuccess) {
      task->failed.store(true, std::memory_order_release);
      task->nextSlice = task->slices.size();
    }

    std::vector<void *> pending;
    pending.reserve(task->inflight.size());
    for (void *request : task->inflight) {
      int done = 0;
      int failed = 0;
      const flagcxResult_t result = ops->test(request, &done, &failed);
      if (result != flagcxSuccess) {
        task->failed.store(true, std::memory_order_release);
        task->nextSlice = task->slices.size();
        continue;
      }
      if (!done)
        pending.push_back(request);
      else if (failed)
        task->failed.store(true, std::memory_order_release);
    }
    task->inflight.swap(pending);

    struct flagcxP2pTransportCaps caps;
    memset(&caps, 0, sizeof(caps));
    if (ops->getCaps(task->conn->sendComm, &caps) != flagcxSuccess ||
        caps.maxBatchSize == 0) {
      task->failed.store(true, std::memory_order_release);
      task->nextSlice = task->slices.size();
    }
    const size_t maxInflight = std::max<size_t>(1, caps.maxInflightBatches);
    while (!task->failed.load(std::memory_order_acquire) &&
           task->nextSlice < task->slices.size() &&
           task->inflight.size() < maxInflight) {
      const int count = (int)std::min<size_t>(
          caps.maxBatchSize, task->slices.size() - task->nextSlice);
      void *request = NULL;
      const flagcxResult_t result = ops->submitBatch(
          task->conn->sendComm, task->slices.data() + task->nextSlice, count,
          &request);
      if (result == flagcxInProgress)
        break;
      if (result != flagcxSuccess || request == NULL) {
        task->failed.store(true, std::memory_order_release);
        task->nextSlice = task->slices.size();
        break;
      }
      task->inflight.push_back(request);
      task->nextSlice += count;
    }

    if (task->nextSlice == task->slices.size() && task->inflight.empty())
      task->done.store(true, std::memory_order_release);
  }

  void workerLoop(int worker) {
    while (true) {
      std::shared_ptr<FlagcxP2pNetTask> task;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return stopping_ || hasWorkFor(worker); });
        if (stopping_)
          return;
        task = popFor(worker);
      }
      if (!task)
        continue;
      progressTask(task);
      if (task->done.load(std::memory_order_acquire)) {
        retireTask(task);
      } else {
        enqueue(task);
        std::this_thread::yield();
      }
    }
  }

  FlagcxP2pEngine *engine_;
  int numShards_ = 1;
  int numWorkers_ = 1;
  std::vector<std::deque<std::shared_ptr<FlagcxP2pNetTask>>> queues_;
  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable drainCv_;
  std::unordered_map<FlagcxP2pConn *, size_t> connectionTaskCounts_;
  std::unordered_set<FlagcxP2pConn *> closingConnections_;
  std::unordered_set<FlagcxP2pConn *> abandoningConnections_;
  bool stopping_ = false;
};

static bool findMemReg(FlagcxP2pEngine *engine, uintptr_t addr,
                       FlagcxP2pMemRegEntry *out) {
  if (engine == NULL || engine->state == NULL)
    return false;
  if (!flagcxParamMrSortedLookup()) {
    /* Legacy: O(n) linear scan over hash map */
    std::lock_guard<std::mutex> lock(engine->state->memMutex);
    for (auto it = engine->state->memRegInfo.begin();
         it != engine->state->memRegInfo.end(); ++it) {
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

  /* New: O(log n) sorted-array registry lookup */
  struct flagcxMrEntry entry;
  struct flagcxMrExtension p2pExt;
  struct flagcxMrExtension *exts[FLAGCX_MR_OWNER_COUNT] = {&p2pExt, NULL, NULL};

  if (flagcxMrRegistryLookup(engine->state->mrRegistry, addr, &entry, exts) !=
      flagcxSuccess)
    return false;

  if (!(entry.ownerMask & FLAGCX_MR_OWNER_P2P) ||
      p2pExt.type != FLAGCX_MR_OWNER_P2P)
    return false;

  if (out) {
    out->mrId = p2pExt.p2p.mrId;
    out->mhandle = entry.mhandles[FLAGCX_MR_OWNER_IDX_P2P];
    out->baseAddr = entry.baseAddr;
    out->size = entry.size;
    out->ibDevN = entry.ibDevN;
    out->ptrType = entry.ptrType;
    memset(&out->mrInfo, 0, sizeof(out->mrInfo));
    getP2pMrInfo(out->mhandle, &out->mrInfo);
    out->hasIpc = p2pExt.p2p.hasIpc;
    out->ipcHandleSize = p2pExt.p2p.ipcHandleSize;
    memcpy(out->ipcHandle, p2pExt.p2p.ipcHandle, FLAGCX_P2P_IPC_HANDLE_BYTES);
  }
  return true;
}

/*
 * Batch containment lookup — acquires gMemMutex once in legacy mode.
 * Returns false (and stops) if any addr is not found.
 */
static bool findMemRegBatch(FlagcxP2pEngine *engine, const uintptr_t *addrs,
                            int count, FlagcxP2pMemRegEntry *out) {
  if (engine == NULL || engine->state == NULL)
    return false;
  if (!flagcxParamMrSortedLookup()) {
    std::lock_guard<std::mutex> lock(engine->state->memMutex);
    for (int i = 0; i < count; i++) {
      bool found = false;
      for (auto it = engine->state->memRegInfo.begin();
           it != engine->state->memRegInfo.end(); ++it) {
        if (addrs[i] >= it->first && addrs[i] < it->first + it->second.size) {
          out[i] = it->second;
          found = true;
          break;
        }
      }
      if (!found)
        return false;
    }
    return true;
  }
  /* New path: per-element registry lookup (rdlock is cheap) */
  for (int i = 0; i < count; i++) {
    if (!findMemReg(engine, addrs[i], &out[i]))
      return false;
  }
  return true;
}

static bool findMemRegByMr(FlagcxP2pEngine *engine, FlagcxP2pMr mr,
                           FlagcxP2pMemRegEntry *out) {
  if (engine == NULL || engine->state == NULL)
    return false;
  if (!flagcxParamMrSortedLookup()) {
    /* Legacy: O(1) hash lookup */
    std::lock_guard<std::mutex> lock(engine->state->memMutex);
    auto mrIt = engine->state->mrToBaseAddr.find(mr);
    if (mrIt == engine->state->mrToBaseAddr.end())
      return false;
    auto entryIt = engine->state->memRegInfo.find(mrIt->second);
    if (entryIt == engine->state->memRegInfo.end())
      return false;
    if (out)
      *out = entryIt->second;
    return true;
  }

  /* New: O(log n) sorted-array registry lookup */
  struct flagcxMrEntry found;
  struct flagcxMrExtension p2pExt;
  struct flagcxMrExtension *exts[FLAGCX_MR_OWNER_COUNT] = {&p2pExt, NULL, NULL};
  if (flagcxMrRegistryLookupById(engine->state->mrRegistry, mr, &found, exts) !=
      flagcxSuccess)
    return false;
  if (p2pExt.type != FLAGCX_MR_OWNER_P2P)
    return false;
  if (out) {
    out->mrId = p2pExt.p2p.mrId;
    out->mhandle = found.mhandles[FLAGCX_MR_OWNER_IDX_P2P];
    out->baseAddr = found.baseAddr;
    out->size = found.size;
    out->ibDevN = found.ibDevN;
    out->ptrType = found.ptrType;
    memset(&out->mrInfo, 0, sizeof(out->mrInfo));
    getP2pMrInfo(out->mhandle, &out->mrInfo);
    out->hasIpc = p2pExt.p2p.hasIpc;
    out->ipcHandleSize = p2pExt.p2p.ipcHandleSize;
    memcpy(out->ipcHandle, p2pExt.p2p.ipcHandle, FLAGCX_P2P_IPC_HANDLE_BYTES);
  }
  return true;
}

static bool memRegContains(const FlagcxP2pMemRegEntry &entry, uintptr_t addr,
                           size_t size) {
  if (addr < entry.baseAddr)
    return false;

  const uintptr_t offset = addr - entry.baseAddr;
  return offset <= entry.size && size <= entry.size - offset;
}

static const FlagcxP2pMrChunk *
findLocalMrChunk(const FlagcxP2pRegisteredMemory *registration,
                 uintptr_t addr) {
  if (registration == NULL)
    return NULL;
  for (const FlagcxP2pMrChunk &chunk : registration->chunks) {
    if (addr >= chunk.baseAddr && addr - chunk.baseAddr < chunk.size)
      return &chunk;
  }
  return NULL;
}

static uint32_t
countLocalMrChunks(const FlagcxP2pRegisteredMemory *registration,
                   uintptr_t addr, size_t size) {
  if (registration == NULL || size == 0)
    return 0;
  uint32_t count = 0;
  uintptr_t current = addr;
  size_t remaining = size;
  while (remaining > 0) {
    const FlagcxP2pMrChunk *chunk = findLocalMrChunk(registration, current);
    if (chunk == NULL)
      return 0;
    const size_t available = chunk->size - (size_t)(current - chunk->baseAddr);
    if (available == 0)
      return 0;
    const size_t consumed = std::min(remaining, available);
    if (consumed > UINTPTR_MAX - current)
      return 0;
    current += consumed;
    remaining -= consumed;
    count++;
  }
  return count;
}

static bool findRemoteMrChunk(const FlagcxP2pConn *conn, uint64_t addr,
                              FlagcxP2pRemoteRegion *out) {
  if (conn == NULL || out == NULL)
    return false;
  FlagcxP2pSharedLock lock(conn->remoteRegionsMutex);
  if (conn->remoteRegions.empty())
    return false;
  size_t lo = 0;
  size_t hi = conn->remoteRegions.size();
  while (lo < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    if (conn->remoteRegions[mid].baseAddr <= addr)
      lo = mid + 1;
    else
      hi = mid;
  }
  if (lo == 0)
    return false;
  const FlagcxP2pRemoteRegion &region = conn->remoteRegions[lo - 1];
  if (addr < region.baseAddr || addr - region.baseAddr >= region.size)
    return false;
  *out = region;
  return true;
}

/* A logical MR can be split into several transport registrations. Local
 * direct copies do not consume transport keys, so validate the complete
 * range against consecutive chunks belonging to the same logical MR rather
 * than requiring it to fit in one chunk. */
static bool validateRemoteMrRange(const FlagcxP2pConn *conn, uint64_t addr,
                                  size_t size,
                                  FlagcxP2pRemoteRegion *firstOut) {
  if (conn == NULL || firstOut == NULL)
    return false;
  FlagcxP2pSharedLock lock(conn->remoteRegionsMutex);
  if (conn->remoteRegions.empty())
    return false;

  size_t lo = 0;
  size_t hi = conn->remoteRegions.size();
  while (lo < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    if (conn->remoteRegions[mid].baseAddr <= addr)
      lo = mid + 1;
    else
      hi = mid;
  }
  if (lo == 0)
    return false;

  size_t index = lo - 1;
  const FlagcxP2pRemoteRegion first = conn->remoteRegions[index];
  if (addr < first.baseAddr || addr - first.baseAddr >= first.size)
    return false;
  *firstOut = first;

  uint64_t current = addr;
  size_t remaining = size;
  while (remaining > 0) {
    if (index >= conn->remoteRegions.size())
      return false;
    const FlagcxP2pRemoteRegion &region = conn->remoteRegions[index];
    if (region.ownerEndpointId != first.ownerEndpointId ||
        region.mrId != first.mrId || region.ptrType != first.ptrType ||
        current < region.baseAddr || current - region.baseAddr >= region.size)
      return false;
    const size_t available =
        region.size - static_cast<size_t>(current - region.baseAddr);
    const size_t consumed = std::min(remaining, available);
    if (consumed > UINT64_MAX - current)
      return false;
    current += consumed;
    remaining -= consumed;
    if (remaining == 0)
      break;
    ++index;
    if (index >= conn->remoteRegions.size() ||
        conn->remoteRegions[index].baseAddr != current)
      return false;
  }
  return true;
}

static int resolveRegistrationDevice(FlagcxP2pEngine *engine, int netDev) {
  int registrationDev = 0;
  if (engine != NULL && engine->transportOps != NULL &&
      engine->transportOps->getRegistrationDevice != NULL &&
      engine->transportOps->getRegistrationDevice(netDev, &registrationDev) ==
          flagcxSuccess)
    return registrationDev;
  return 0;
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

static int detectPtrTypeAndMaybeCacheIpc(void *ptr, int hintType,
                                         char *ipcHandleBuf,
                                         uint32_t *ipcHandleSize) {
  if (ipcHandleBuf)
    memset(ipcHandleBuf, 0, FLAGCX_P2P_IPC_HANDLE_BYTES);
  if (ipcHandleSize)
    *ipcHandleSize = 0;

  int ptrType = FLAGCX_PTR_HOST;
  bool ptrTypeKnown = false;
  if (hintType == FLAGCX_PTR_HOST || hintType == FLAGCX_PTR_CUDA) {
    ptrType = hintType;
    ptrTypeKnown = true;
  } else if (deviceAdaptor != NULL && deviceAdaptor->getPointerType != NULL &&
             deviceAdaptor->getPointerType(ptr, &ptrType) == flagcxSuccess &&
             (ptrType == FLAGCX_PTR_HOST || ptrType == FLAGCX_PTR_CUDA)) {
    ptrTypeKnown = true;
  }

  if (ptrTypeKnown && ptrType == FLAGCX_PTR_HOST)
    return FLAGCX_PTR_HOST;

  if (deviceAdaptor == NULL || deviceAdaptor->ipcMemHandleCreate == NULL ||
      deviceAdaptor->ipcMemHandleGet == NULL ||
      deviceAdaptor->ipcMemHandleFree == NULL) {
    return ptrTypeKnown ? ptrType : FLAGCX_PTR_HOST;
  }

  flagcxIpcMemHandle_t handle = NULL;
  size_t handleSize = 0;
  if (deviceAdaptor->ipcMemHandleCreate(&handle, &handleSize) !=
      flagcxSuccess) {
    return ptrTypeKnown ? ptrType : FLAGCX_PTR_HOST;
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
  if (deviceAdaptor->getLastError)
    deviceAdaptor->getLastError();
  deviceAdaptor->ipcMemHandleFree(handle);
  // IPC exportability is a data-path capability, not an address type. A GPU
  // allocation remains a GPU MR when the runtime cannot export an IPC handle.
  return ptrTypeKnown ? ptrType : FLAGCX_PTR_HOST;
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

static int sendNotifWire(FlagcxP2pConn *conn,
                         const FlagcxP2pNotifWireMsg &wireMsg) {
  if (conn == NULL || !conn->notifSockConnected)
    return -1;
  std::lock_guard<std::mutex> sendLock(conn->notifSendMutex);
  FlagcxP2pNotifWireMsg mutableWireMsg = wireMsg;
  return flagcxSocketSend(&conn->notifSock, &mutableWireMsg, sizeof(wireMsg)) ==
                 flagcxSuccess
             ? 0
             : -1;
}

static FlagcxP2pNotifWireMsg
makeMrControlWire(FlagcxP2pNotifType type,
                  const FlagcxP2pMrCtrlPayload &payload) {
  FlagcxP2pNotifWireMsg wireMsg;
  memset(&wireMsg, 0, sizeof(wireMsg));
  wireMsg.magic = FLAGCX_P2P_NOTIF_MAGIC;
  wireMsg.type = type;
  memcpy(&wireMsg.payload, &payload, sizeof(payload));
  return wireMsg;
}

static void handleMrAck(FlagcxP2pEngine *engine,
                        const FlagcxP2pMrCtrlPayload &payload) {
  std::lock_guard<std::mutex> ackLock(engine->ctrlAckMutex);
  auto it = engine->ctrlAcks.find(payload.sequence);
  if (it == engine->ctrlAcks.end())
    return;
  it->second.done = true;
  it->second.status = payload.status;
  engine->ctrlAckCv.notify_all();
}

static void
updateRemoteRegionLocked(std::vector<FlagcxP2pRemoteRegion> &regions,
                         const FlagcxP2pMrCtrlPayload &payload, bool remove) {
  regions.erase(std::remove_if(regions.begin(), regions.end(),
                               [&](const FlagcxP2pRemoteRegion &region) {
                                 if (region.ownerEndpointId !=
                                         payload.senderEndpointId ||
                                     region.mrId != payload.region.mrId)
                                   return false;
                                 return remove || region.baseAddr ==
                                                      payload.region.baseAddr;
                               }),
                regions.end());
  if (!remove) {
    FlagcxP2pRemoteRegion region;
    memset(&region, 0, sizeof(region));
    region.baseAddr = payload.region.baseAddr;
    region.size = payload.region.size;
    region.ownerEndpointId = payload.senderEndpointId;
    region.mrId = payload.region.mrId;
    region.ptrType = payload.region.ptrType;
    region.mrInfo.nKeys =
        std::min<uint32_t>(payload.region.nKeys, FLAGCX_NET_MAX_MR_KEYS);
    memcpy(region.mrInfo.rkeys, payload.region.rkeys,
           region.mrInfo.nKeys * sizeof(uint32_t));
    regions.push_back(region);
    std::sort(
        regions.begin(), regions.end(),
        [](const FlagcxP2pRemoteRegion &a, const FlagcxP2pRemoteRegion &b) {
          return a.baseAddr < b.baseAddr;
        });
  }
}

static void updateRemoteRegion(FlagcxP2pConn *conn, FlagcxP2pNotifType type,
                               const FlagcxP2pMrCtrlPayload &payload) {
  std::unique_lock<FlagcxP2pSharedMutex> lock(conn->remoteRegionsMutex);
  if (!conn->mrHandshakeComplete) {
    conn->pendingMrUpdates.push_back({type, payload});
    return;
  }
  updateRemoteRegionLocked(conn->remoteRegions, payload,
                           type == FLAGCX_P2P_NOTIF_MR_REMOVE);
}

static void handleMrUpdate(FlagcxP2pEngine *engine, FlagcxP2pNotifType type,
                           const FlagcxP2pMrCtrlPayload &payload) {
  int status = 0;
  FlagcxP2pSharedLock lifetimeLock(engine->connectionLifetimeMutex);
  std::lock_guard<std::mutex> connectionLock(engine->connectionMutex);
  FlagcxP2pConn *ackConn = NULL;
  for (FlagcxP2pConn *conn : engine->connections) {
    if (conn->remoteEndpointId != payload.senderEndpointId)
      continue;
    if (ackConn == NULL && conn->notifSockConnected)
      ackConn = conn;
    if (type == FLAGCX_P2P_NOTIF_MR_ADD) {
      if (payload.region.size == 0 || payload.region.nKeys == 0 ||
          payload.region.nKeys > FLAGCX_NET_MAX_MR_KEYS ||
          (payload.region.ptrType != FLAGCX_PTR_HOST &&
           payload.region.ptrType != FLAGCX_PTR_CUDA)) {
        status = -1;
        continue;
      }
      updateRemoteRegion(conn, type, payload);
    } else {
      updateRemoteRegion(conn, type, payload);
    }
  }
  if (ackConn == NULL)
    return;
  FlagcxP2pMrCtrlPayload ack;
  memset(&ack, 0, sizeof(ack));
  ack.senderEndpointId = engine->endpointId;
  ack.sequence = payload.sequence;
  ack.status = status;
  sendNotifWire(ackConn, makeMrControlWire(FLAGCX_P2P_NOTIF_MR_ACK, ack));
}

static bool sendMrControlAndWait(FlagcxP2pEngine *engine, FlagcxP2pConn *conn,
                                 FlagcxP2pNotifType type,
                                 FlagcxP2pMrCtrlPayload *payload) {
  payload->senderEndpointId = engine->endpointId;
  payload->sequence =
      engine->nextCtrlSequence.fetch_add(1, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> ackLock(engine->ctrlAckMutex);
    engine->ctrlAcks[payload->sequence] = FlagcxP2pCtrlAckState();
  }
  if (sendNotifWire(conn, makeMrControlWire(type, *payload)) != 0) {
    std::lock_guard<std::mutex> ackLock(engine->ctrlAckMutex);
    engine->ctrlAcks.erase(payload->sequence);
    return false;
  }

  std::unique_lock<std::mutex> ackLock(engine->ctrlAckMutex);
  const bool received =
      engine->ctrlAckCv.wait_for(ackLock, std::chrono::seconds(5), [&] {
        auto it = engine->ctrlAcks.find(payload->sequence);
        return it != engine->ctrlAcks.end() && it->second.done;
      });
  auto it = engine->ctrlAcks.find(payload->sequence);
  const bool success =
      received && it != engine->ctrlAcks.end() && it->second.status == 0;
  if (it != engine->ctrlAcks.end())
    engine->ctrlAcks.erase(it);
  return success;
}

static bool publishSameProcessMrUpdate(FlagcxP2pEngine *engine,
                                       FlagcxP2pConn *conn,
                                       FlagcxP2pNotifType type,
                                       FlagcxP2pMrCtrlPayload *payload) {
  std::lock_guard<std::mutex> directoryLock(gP2pEngineDirectoryMutex);
  auto peerIt = gP2pEngineDirectory.find(conn->remoteEndpointId);
  if (peerIt == gP2pEngineDirectory.end())
    return false;
  FlagcxP2pEngine *peerEngine = peerIt->second;
  payload->senderEndpointId = engine->endpointId;
  FlagcxP2pSharedLock lifetimeLock(peerEngine->connectionLifetimeMutex,
                                   std::defer_lock);
  /* publishMrAdd/Remove already hold the source Engine's lifetime lock. A
     self-connection must not recursively acquire the same shared lock. */
  if (peerEngine != engine)
    lifetimeLock.lock();
  std::lock_guard<std::mutex> connectionLock(peerEngine->connectionMutex);
  bool updated = false;
  for (FlagcxP2pConn *peerConn : peerEngine->connections) {
    if (!peerConn->sameProcess ||
        peerConn->remoteEndpointId != engine->endpointId)
      continue;
    updateRemoteRegion(peerConn, type, *payload);
    updated = true;
  }
  return updated;
}

static bool publishMrAdd(FlagcxP2pEngine *engine,
                         const FlagcxP2pRegisteredMemory *registration,
                         FlagcxP2pMr mrId) {
  bool success = true;
  FlagcxP2pSharedLock lifetimeLock(engine->connectionLifetimeMutex);
  std::vector<FlagcxP2pConn *> connections;
  {
    std::lock_guard<std::mutex> connectionLock(engine->connectionMutex);
    for (FlagcxP2pConn *conn : engine->connections) {
      if (conn->mrPublishReady)
        connections.push_back(conn);
    }
  }
  for (FlagcxP2pConn *conn : connections) {
    if (conn->sameProcess) {
      for (const FlagcxP2pMrChunk &chunk : registration->chunks) {
        FlagcxP2pMrCtrlPayload payload;
        memset(&payload, 0, sizeof(payload));
        payload.region.baseAddr = chunk.baseAddr;
        payload.region.size = chunk.size;
        payload.region.mrId = mrId;
        payload.region.ptrType = registration->ptrType;
        payload.region.nKeys =
            std::min<uint32_t>(chunk.mrInfo.nKeys, FLAGCX_NET_MAX_MR_KEYS);
        memcpy(payload.region.rkeys, chunk.mrInfo.rkeys,
               payload.region.nKeys * sizeof(uint32_t));
        if (!publishSameProcessMrUpdate(engine, conn, FLAGCX_P2P_NOTIF_MR_ADD,
                                        &payload))
          success = false;
      }
      continue;
    }
    if (!conn->notifSockConnected) {
      success = false;
      continue;
    }
    for (const FlagcxP2pMrChunk &chunk : registration->chunks) {
      FlagcxP2pMrCtrlPayload payload;
      memset(&payload, 0, sizeof(payload));
      payload.region.baseAddr = chunk.baseAddr;
      payload.region.size = chunk.size;
      payload.region.mrId = mrId;
      payload.region.ptrType = registration->ptrType;
      payload.region.nKeys =
          std::min<uint32_t>(chunk.mrInfo.nKeys, FLAGCX_NET_MAX_MR_KEYS);
      memcpy(payload.region.rkeys, chunk.mrInfo.rkeys,
             payload.region.nKeys * sizeof(uint32_t));
      if (!sendMrControlAndWait(engine, conn, FLAGCX_P2P_NOTIF_MR_ADD,
                                &payload))
        success = false;
    }
  }
  return success;
}

static bool publishMrRemove(FlagcxP2pEngine *engine, FlagcxP2pMr mrId) {
  bool success = true;
  FlagcxP2pSharedLock lifetimeLock(engine->connectionLifetimeMutex);
  std::vector<FlagcxP2pConn *> connections;
  {
    std::lock_guard<std::mutex> connectionLock(engine->connectionMutex);
    for (FlagcxP2pConn *conn : engine->connections) {
      if (conn->mrPublishReady)
        connections.push_back(conn);
    }
  }
  for (FlagcxP2pConn *conn : connections) {
    if (conn->sameProcess) {
      FlagcxP2pMrCtrlPayload payload;
      memset(&payload, 0, sizeof(payload));
      payload.region.mrId = mrId;
      if (!publishSameProcessMrUpdate(engine, conn, FLAGCX_P2P_NOTIF_MR_REMOVE,
                                      &payload))
        success = false;
      continue;
    }
    FlagcxP2pMrCtrlPayload payload;
    memset(&payload, 0, sizeof(payload));
    payload.region.mrId = mrId;
    if (!sendMrControlAndWait(engine, conn, FLAGCX_P2P_NOTIF_MR_REMOVE,
                              &payload))
      success = false;
  }
  return success;
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

static int notifParseMessages(FlagcxP2pEngine *engine,
                              FlagcxP2pNotifConn *conn) {
  while (conn->inBuf.size() >= sizeof(FlagcxP2pNotifWireMsg)) {
    FlagcxP2pNotifWireMsg wireMsg;
    memcpy(&wireMsg, conn->inBuf.data(), sizeof(wireMsg));
    conn->inBuf.erase(conn->inBuf.begin(),
                      conn->inBuf.begin() + sizeof(wireMsg));
    if (wireMsg.magic != FLAGCX_P2P_NOTIF_MAGIC) {
      return -1;
    }
    if (wireMsg.type == FLAGCX_P2P_NOTIF_USER) {
      queueNotifMsg(wireMsg.payload);
      continue;
    }
    if (wireMsg.type != FLAGCX_P2P_NOTIF_MR_ADD &&
        wireMsg.type != FLAGCX_P2P_NOTIF_MR_REMOVE &&
        wireMsg.type != FLAGCX_P2P_NOTIF_MR_ACK)
      return -1;
    FlagcxP2pMrCtrlPayload payload;
    memcpy(&payload, &wireMsg.payload, sizeof(payload));
    if (wireMsg.type == FLAGCX_P2P_NOTIF_MR_ACK)
      handleMrAck(engine, payload);
    else
      handleMrUpdate(engine, (FlagcxP2pNotifType)wireMsg.type, payload);
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
    if (notifParseMessages(engine, &it->second) != 0) {
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

  /* Batch local lookups (single lock acquisition in legacy mode) */
  std::vector<uintptr_t> localAddrs(numIovs);
  for (int i = 0; i < numIovs; i++)
    localAddrs[i] = (uintptr_t)localVec[i];
  if (!findMemRegBatch(conn->engine, localAddrs.data(), numIovs,
                       localEntries.data()))
    return -1;

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
      FlagcxP2pRemoteRegion remoteRegion;
      if (!validateRemoteMrRange(conn, descs[i].addr, sizeVec[i],
                                 &remoteRegion) ||
          (descs[i].idx != 0 && descs[i].idx != remoteRegion.mrId)) {
        cleanupIpcXfer(&xfer);
        return -1;
      }
      remoteIsCuda = remoteRegion.ptrType == FLAGCX_PTR_CUDA;
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

  std::lock_guard<std::mutex> xferLock(conn->engine->state->xferMutex);
  const uint64_t xferId = conn->engine->state->nextXferId++;
  conn->engine->state->xfers[xferId] = std::move(xfer);
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
  if (!flagcxParamMrSortedLookup()) {
    /* Legacy: iterate hash map */
    std::lock_guard<std::mutex> lock(conn->engine->state->memMutex);
    localTable.reserve(conn->engine->state->memRegInfo.size());
    for (auto it = conn->engine->state->memRegInfo.begin();
         it != conn->engine->state->memRegInfo.end(); ++it) {
      auto *registration =
          static_cast<FlagcxP2pRegisteredMemory *>(it->second.mhandle);
      if (registration == NULL)
        continue;
      for (const FlagcxP2pMrChunk &chunk : registration->chunks) {
        FlagcxP2pMemRegWire w;
        memset(&w, 0, sizeof(w));
        w.baseAddr = chunk.baseAddr;
        w.size = chunk.size;
        w.mrId = it->second.mrId;
        w.ptrType = registration->ptrType;
        w.nKeys =
            std::min<uint32_t>(chunk.mrInfo.nKeys, FLAGCX_NET_MAX_MR_KEYS);
        memcpy(w.rkeys, chunk.mrInfo.rkeys, w.nKeys * sizeof(uint32_t));
        localTable.push_back(w);
      }
    }
  } else {
    /* New: iterate sorted registry */
    if (flagcxMrRegistryRdLock(conn->engine->state->mrRegistry) ==
        flagcxSuccess) {
      int count = flagcxMrRegistryCount(conn->engine->state->mrRegistry);
      if (count > 0) {
        struct flagcxMrEntry *entries =
            flagcxMrRegistryEntries(conn->engine->state->mrRegistry);
        localTable.reserve(count);
        for (int i = 0; i < count; i++) {
          if (!(entries[i].ownerMask & FLAGCX_MR_OWNER_P2P))
            continue;
          auto *registration = static_cast<FlagcxP2pRegisteredMemory *>(
              entries[i].mhandles[FLAGCX_MR_OWNER_IDX_P2P]);
          if (registration == NULL)
            continue;
          for (const FlagcxP2pMrChunk &chunk : registration->chunks) {
            FlagcxP2pMemRegWire w;
            memset(&w, 0, sizeof(w));
            w.baseAddr = chunk.baseAddr;
            w.size = chunk.size;
            w.mrId = entries[i].p2p->mrId;
            w.ptrType = registration->ptrType;
            w.nKeys =
                std::min<uint32_t>(chunk.mrInfo.nKeys, FLAGCX_NET_MAX_MR_KEYS);
            memcpy(w.rkeys, chunk.mrInfo.rkeys, w.nKeys * sizeof(uint32_t));
            localTable.push_back(w);
          }
        }
      }
      flagcxMrRegistryRdUnlock(conn->engine->state->mrRegistry);
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

  std::unique_lock<FlagcxP2pSharedMutex> regionsLock(conn->remoteRegionsMutex);
  conn->remoteRegions.clear();
  conn->remoteRegions.reserve(remoteCount);
  for (uint32_t i = 0; i < remoteCount; i++) {
    FlagcxP2pRemoteRegion r;
    memset(&r, 0, sizeof(r));
    r.baseAddr = remoteTable[i].baseAddr;
    r.size = remoteTable[i].size;
    r.ownerEndpointId = conn->remoteEndpointId;
    r.mrId = remoteTable[i].mrId;
    r.ptrType = remoteTable[i].ptrType;
    r.mrInfo.nKeys =
        std::min<uint32_t>(remoteTable[i].nKeys, FLAGCX_NET_MAX_MR_KEYS);
    memcpy(r.mrInfo.rkeys, remoteTable[i].rkeys,
           r.mrInfo.nKeys * sizeof(uint32_t));
    conn->remoteRegions.push_back(r);
  }
  std::sort(conn->remoteRegions.begin(), conn->remoteRegions.end(),
            [](const FlagcxP2pRemoteRegion &a, const FlagcxP2pRemoteRegion &b) {
              return a.baseAddr < b.baseAddr;
            });
  for (const FlagcxP2pPendingMrUpdate &update : conn->pendingMrUpdates) {
    updateRemoteRegionLocked(conn->remoteRegions, update.payload,
                             update.type == FLAGCX_P2P_NOTIF_MR_REMOVE);
  }
  conn->pendingMrUpdates.clear();
  conn->mrHandshakeComplete = true;
  return 0;
}

/* Register the connection before exchanging the initial MR table.  Dynamic
 * updates that race with the snapshot are queued on the receiver and replayed
 * after the snapshot is installed, so connect/accept may safely share one
 * Engine without holding an Engine mutex across a blocking exchange. */
static int bootstrapFinalizeConnection(struct bootstrapState *bsState,
                                       FlagcxP2pConn *conn) {
  if (bsState == NULL || conn == NULL || conn->engine == NULL)
    return -1;

  FlagcxP2pEngine *engine = conn->engine;
  {
    std::lock_guard<std::mutex> connectionLock(engine->connectionMutex);
    engine->connections.insert(conn);
  }

  /* Both sides must be discoverable before either side starts publishing. */
  uint32_t localReady = 1;
  uint32_t remoteReady = 0;
  if (bootstrapExchange(bsState, 0, 5, &localReady, sizeof(localReady),
                        &remoteReady, sizeof(remoteReady)) != flagcxSuccess ||
      remoteReady != 1) {
    std::lock_guard<std::mutex> connectionLock(engine->connectionMutex);
    engine->connections.erase(conn);
    return -1;
  }

  /* Reg holds mrPublishMutex while updating the registry and enumerating
   * publish-ready connections.  Therefore a registration is either included
   * in the following snapshot or emitted as a queued incremental update. */
  {
    std::unique_lock<std::mutex> publishLock(engine->mrPublishMutex);
    std::lock_guard<std::mutex> connectionLock(engine->connectionMutex);
    conn->mrPublishReady = true;
  }

  if (bootstrapExchangeDescTable(bsState, conn) != 0) {
    std::lock_guard<std::mutex> connectionLock(engine->connectionMutex);
    conn->mrPublishReady = false;
    engine->connections.erase(conn);
    return -1;
  }

  localReady = 1;
  remoteReady = 0;
  if (bootstrapExchange(bsState, 0, 6, &localReady, sizeof(localReady),
                        &remoteReady, sizeof(remoteReady)) != flagcxSuccess ||
      remoteReady != 1) {
    std::lock_guard<std::mutex> connectionLock(engine->connectionMutex);
    conn->mrPublishReady = false;
    engine->connections.erase(conn);
    return -1;
  }
  return 0;
}

FlagcxP2pEngine *flagcxP2pEngineCreate() {
  /* Transport selection only chooses a network adaptor. Bootstrap, locality,
     IPC and completion remain owned by this common engine. */
  const char *transport = flagcxGetEnv("FLAGCX_P2P_TRANSPORT");
  if (transport != NULL && strcasecmp(transport, "accl") == 0) {
#ifndef USE_ACCL_BAREX
    WARN("FLAGCX_P2P_TRANSPORT=accl but FlagCX was built without "
         "USE_ACCL_BAREX=1");
    return NULL;
#endif
  }

  FlagcxP2pEngine *engine = new FlagcxP2pEngine;
  engine->state = new FlagcxP2pEngineState;
  if (flagcxParamMrSortedLookup() &&
      flagcxMrRegistryCreate(&engine->state->mrRegistry) != flagcxSuccess) {
    delete engine->state;
    delete engine;
    return NULL;
  }
  engine->adaptor = getP2pNetAdaptor();
  engine->transportOps = getP2pTransportOps(engine->adaptor);
  engine->scheduler = NULL;
  engine->topoMgr = NULL;
  engine->nDevs = 0;
  engine->localGpuIdx = inferLocalGpuIdx();
  engine->endpointId = allocateP2pEndpointId();
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

  if (engine->transportOps == NULL) {
    if (engine->state->mrRegistry != NULL)
      flagcxMrRegistryDestroy(engine->state->mrRegistry);
    delete engine->state;
    delete engine;
    return NULL;
  }

  if (engine->adaptor->init() != flagcxSuccess) {
    if (engine->state->mrRegistry != NULL)
      flagcxMrRegistryDestroy(engine->state->mrRegistry);
    delete engine->state;
    delete engine;
    return NULL;
  }
  engine->scheduler = new FlagcxP2pScheduler(engine);

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
      flagcxSocketInit(&engine->notifListenSock, bootstrapGetNetIfAddr(),
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

  if (engine->notifListenActive)
    engine->notifThread = std::thread(notifPollThreadFunc, engine);

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
    INFO(FLAGCX_INIT, "P2P/ENGINE : bootstrap listen on port %d",
         engine->bsListenPort);
  }

  {
    std::lock_guard<std::mutex> directoryLock(gP2pEngineDirectoryMutex);
    gP2pEngineDirectory[engine->endpointId] = engine;
  }

  return engine;
}

void flagcxP2pEngineDestroy(FlagcxP2pEngine *engine) {
  if (engine == NULL)
    return;

  {
    std::lock_guard<std::mutex> directoryLock(gP2pEngineDirectoryMutex);
    auto it = gP2pEngineDirectory.find(engine->endpointId);
    if (it != gP2pEngineDirectory.end() && it->second == engine)
      gP2pEngineDirectory.erase(it);
  }

  flagcxP2pEngineStopAccept(engine);
  if (engine->notifListenActive) {
    flagcxSocketClose(&engine->notifListenSock);
    engine->notifListenActive = false;
  }
  if (engine->notifThread.joinable())
    engine->notifThread.join();

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
  std::unordered_set<FlagcxP2pConn *> connsToDestroy;
  {
    std::lock_guard<std::mutex> lock(engine->sessionMutex);
    for (const auto &entry : engine->sessionConns)
      connsToDestroy.insert(entry.second);
    engine->sessionConns.clear();
  }
  {
    std::lock_guard<std::mutex> lock(engine->acceptedMutex);
    connsToDestroy.insert(engine->acceptedConns.begin(),
                          engine->acceptedConns.end());
    engine->acceptedConns.clear();
  }
  {
    std::lock_guard<std::mutex> lock(engine->connectionMutex);
    connsToDestroy.insert(engine->connections.begin(),
                          engine->connections.end());
  }
  for (FlagcxP2pConn *conn : connsToDestroy)
    flagcxP2pEngineConnDestroy(conn);

  // Connections quiesce their queued and in-flight work while the scheduler
  // is still alive. Only then is it safe to stop the worker threads.
  delete engine->scheduler;
  engine->scheduler = NULL;

  if (engine->quarantineRequired.load(std::memory_order_acquire)) {
    /* At least one transport request did not complete within the bounded
       quiesce interval. Keep the Engine's connection, transfer and MR state
       alive so a late native completion cannot reference freed memory. */
    WARN("P2P/ENGINE : retaining quarantined Engine resources after a "
         "connection teardown timeout");
    return;
  }

  {
    std::lock_guard<std::mutex> lock(engine->state->xferMutex);
    for (std::unordered_map<uint64_t, FlagcxP2pXfer>::iterator it =
             engine->state->xfers.begin();
         it != engine->state->xfers.end(); ++it) {
      cleanupIpcXfer(&it->second);
    }
    engine->state->xfers.clear();
  }

  {
    if (!flagcxParamMrSortedLookup()) {
      /* Legacy: deregister all from hash maps */
      std::lock_guard<std::mutex> lock(engine->state->memMutex);
      for (auto it = engine->state->memRegInfo.begin();
           it != engine->state->memRegInfo.end(); ++it) {
        deregisterP2pMemory(engine, static_cast<FlagcxP2pRegisteredMemory *>(
                                        it->second.mhandle));
      }
      engine->state->memRegInfo.clear();
      engine->state->mrToBaseAddr.clear();
    } else {
      /* Sorted lookup is private to this engine's logical MR namespace. */
      std::lock_guard<std::mutex> lifecycleLock(
          engine->state->mrLifecycleMutex);

      /* Phase 1: collect P2P mhandle info under read lock */
      struct P2pDeregInfo {
        void *mhandle;
        uintptr_t baseAddr;
      };
      std::vector<P2pDeregInfo> deregList;

      if (flagcxMrRegistryRdLock(engine->state->mrRegistry) == flagcxSuccess) {
        int count = flagcxMrRegistryCount(engine->state->mrRegistry);
        if (count > 0) {
          struct flagcxMrEntry *entries =
              flagcxMrRegistryEntries(engine->state->mrRegistry);
          for (int i = 0; i < count; i++) {
            if (!(entries[i].ownerMask & FLAGCX_MR_OWNER_P2P))
              continue;
            P2pDeregInfo info;
            info.mhandle = entries[i].mhandles[FLAGCX_MR_OWNER_IDX_P2P];
            info.baseAddr = entries[i].baseAddr;
            deregList.push_back(info);
          }
        }
        flagcxMrRegistryRdUnlock(engine->state->mrRegistry);
      }

      /* Phase 2: deregister from registry */
      for (P2pDeregInfo &info : deregList) {
        if (flagcxMrRegistryDeregister(engine->state->mrRegistry, info.baseAddr,
                                       FLAGCX_MR_OWNER_P2P, NULL,
                                       NULL) != flagcxSuccess) {
          info.mhandle = NULL;
        }
      }

      /* Phase 3: release every physical transport MR in the logical MR. */
      for (const P2pDeregInfo &info : deregList) {
        if (info.mhandle == NULL)
          continue;
        deregisterP2pMemory(
            engine, static_cast<FlagcxP2pRegisteredMemory *>(info.mhandle));
      }
    }
  }

  if (engine->state->mrRegistry != NULL)
    flagcxMrRegistryDestroy(engine->state->mrRegistry);

  if (engine->topoMgr) {
    flagcxP2pTopoDestroy(engine->topoMgr);
  }

  delete engine->state;
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

  if (usesIbP2pAdaptor(engine)) {
    for (int d = 0; d < engine->nDevs; d++) {
      if (engine->listeners[d].listenComm) {
        flagcxP2pNetIbAbortListen(engine->listeners[d].listenComm);
      }
    }
  }

  if (engine->rpcServerThread.joinable() &&
      engine->rpcServerThread.get_id() != std::this_thread::get_id()) {
    engine->rpcServerThread.join();
    engine->rpcServerActive.store(false, std::memory_order_release);
  }
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

  // Step 2: Exchange opaque network-adaptor listen handles over bootstrap.
  char localNetHandle[FLAGCX_NET_HANDLE_MAXSIZE];
  memcpy(localNetHandle, engine->listeners[netDev].handle,
         FLAGCX_NET_HANDLE_MAXSIZE);

  char remoteNetHandle[FLAGCX_NET_HANDLE_MAXSIZE];
  memset(remoteNetHandle, 0, sizeof(remoteNetHandle));
  if (bootstrapExchange(bsConn, 0, 4, localNetHandle, FLAGCX_NET_HANDLE_MAXSIZE,
                        remoteNetHandle,
                        FLAGCX_NET_HANDLE_MAXSIZE) != flagcxSuccess) {
    bootstrapClose(bsConn);
    return NULL;
  }

  // Step 3: Let the selected network adaptor establish its data channel.
  void *sendComm = NULL;
  while (sendComm == NULL &&
         !engine->stopAccept.load(std::memory_order_acquire)) {
    if (engine->adaptor->connect(netDev, remoteNetHandle, &sendComm) !=
        flagcxSuccess) {
      bootstrapClose(bsConn);
      return NULL;
    }
    if (sendComm == NULL)
      std::this_thread::yield();
  }
  if (sendComm == NULL) {
    bootstrapClose(bsConn);
    return NULL;
  }

  // Step 4: Exchange ctrl meta over bootstrap
  FlagcxP2pCtrlMeta localMeta;
  memset(&localMeta, 0, sizeof(localMeta));
  localMeta.version = FLAGCX_P2P_CTRL_VERSION;
  localMeta.gpuIdx = engine->localGpuIdx;
  localMeta.notifPort = engine->notifListenPort;
  localMeta.hostHash = getHostHash();
  localMeta.pidHash = getPidHash();
  localMeta.endpointId = engine->endpointId;
  if (sameProcess)
    localMeta.flags |= FLAGCX_P2P_CTRL_FLAG_SAME_PROCESS;

  FlagcxP2pCtrlMeta remoteMeta;
  memset(&remoteMeta, 0, sizeof(remoteMeta));
  if (bootstrapExchangeCtrlMeta(bsConn, &localMeta, &remoteMeta) !=
      flagcxSuccess) {
    engine->adaptor->closeSend(sendComm);
    bootstrapClose(bsConn);
    return NULL;
  }
  if (remoteMeta.version != FLAGCX_P2P_CTRL_VERSION) {
    WARN("P2P bootstrap protocol mismatch: local %u remote %u",
         FLAGCX_P2P_CTRL_VERSION, remoteMeta.version);
    engine->adaptor->closeSend(sendComm);
    bootstrapClose(bsConn);
    return NULL;
  }

  const bool isLocal = localMeta.hostHash == remoteMeta.hostHash;
  const bool isSameProcess =
      isLocal && sameProcess && localMeta.pidHash == remoteMeta.pidHash;

  FlagcxP2pConn *conn = new FlagcxP2pConn;
  conn->engine = engine;
  conn->sendComm = sendComm;
  conn->recvComm = NULL;
  conn->netDev = netDev;
  conn->remoteGpuIdx =
      remoteMeta.gpuIdx >= 0 ? remoteMeta.gpuIdx : remoteGpuIdx;
  conn->remoteNotifPort = remoteMeta.notifPort;
  conn->isLocal = isLocal;
  conn->sameProcess = isSameProcess;
  conn->remoteEndpointId = remoteMeta.endpointId;
  conn->peerBootstrapAddr = bsHandle.addr;
  conn->notifSockConnected = false;
  conn->mrPublishReady = false;
  conn->mrHandshakeComplete = false;
  memset(&conn->notifSock, 0, sizeof(conn->notifSock));

  if (!conn->sameProcess && (remoteMeta.notifPort <= 0 ||
                             connectNotifSocket(conn, &conn->peerBootstrapAddr,
                                                remoteMeta.notifPort) != 0)) {
    WARN("P2P/ENGINE : connect notification channel setup failed");
    flagcxP2pEngineConnDestroy(conn);
    bootstrapClose(bsConn);
    return NULL;
  }

  // Step 5: Exchange the initial MR table and activate dynamic publication.
  if (bootstrapFinalizeConnection(bsConn, conn) != 0) {
    WARN("P2P/ENGINE : connect MR handshake failed");
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

  // Step 2: Exchange opaque network-adaptor listen handles over bootstrap.
  char localNetHandle[FLAGCX_NET_HANDLE_MAXSIZE];
  memcpy(localNetHandle, engine->listeners[dev].handle,
         FLAGCX_NET_HANDLE_MAXSIZE);

  char remoteNetHandle[FLAGCX_NET_HANDLE_MAXSIZE];
  memset(remoteNetHandle, 0, sizeof(remoteNetHandle));
  if (bootstrapExchange(bsConn, 0, 4, localNetHandle, FLAGCX_NET_HANDLE_MAXSIZE,
                        remoteNetHandle,
                        FLAGCX_NET_HANDLE_MAXSIZE) != flagcxSuccess) {
    bootstrapClose(bsConn);
    return NULL;
  }

  // Step 3: Accept the selected network adaptor's data channel.
  void *recvComm = NULL;
  if (engine->stopAccept.load(std::memory_order_acquire)) {
    bootstrapClose(bsConn);
    return NULL;
  }
  while (recvComm == NULL &&
         !engine->stopAccept.load(std::memory_order_acquire)) {
    if (engine->adaptor->accept(engine->listeners[dev].listenComm, &recvComm) !=
        flagcxSuccess) {
      bootstrapClose(bsConn);
      return NULL;
    }
    if (recvComm == NULL)
      std::this_thread::yield();
  }
  if (recvComm == NULL) {
    bootstrapClose(bsConn);
    return NULL;
  }

  // Step 4: Exchange ctrl meta over bootstrap
  FlagcxP2pCtrlMeta localMeta;
  memset(&localMeta, 0, sizeof(localMeta));
  localMeta.version = FLAGCX_P2P_CTRL_VERSION;
  localMeta.gpuIdx = engine->localGpuIdx;
  localMeta.notifPort = engine->notifListenPort;
  localMeta.hostHash = getHostHash();
  localMeta.pidHash = getPidHash();
  localMeta.endpointId = engine->endpointId;

  FlagcxP2pCtrlMeta remoteMeta;
  memset(&remoteMeta, 0, sizeof(remoteMeta));
  if (bootstrapExchangeCtrlMeta(bsConn, &localMeta, &remoteMeta) !=
      flagcxSuccess) {
    engine->adaptor->closeRecv(recvComm);
    bootstrapClose(bsConn);
    return NULL;
  }
  if (remoteMeta.version != FLAGCX_P2P_CTRL_VERSION) {
    WARN("P2P bootstrap protocol mismatch: local %u remote %u",
         FLAGCX_P2P_CTRL_VERSION, remoteMeta.version);
    engine->adaptor->closeRecv(recvComm);
    bootstrapClose(bsConn);
    return NULL;
  }

  const bool isLocal = localMeta.hostHash == remoteMeta.hostHash;
  const bool isSameProcess =
      isLocal && (remoteMeta.flags & FLAGCX_P2P_CTRL_FLAG_SAME_PROCESS) != 0 &&
      localMeta.pidHash == remoteMeta.pidHash;

  FlagcxP2pConn *conn = new FlagcxP2pConn;
  conn->engine = engine;
  conn->sendComm = recvComm;
  conn->recvComm = recvComm;
  conn->netDev = dev;
  conn->remoteGpuIdx = remoteMeta.gpuIdx;
  conn->remoteNotifPort = remoteMeta.notifPort;
  conn->isLocal = isLocal;
  conn->sameProcess = isSameProcess;
  conn->remoteEndpointId = remoteMeta.endpointId;
  memset(&conn->peerBootstrapAddr, 0, sizeof(conn->peerBootstrapAddr));
  flagcxSocketGetAddr(&bsConn->p2p->sock, &conn->peerBootstrapAddr);
  conn->notifSockConnected = false;
  conn->mrPublishReady = false;
  conn->mrHandshakeComplete = false;
  memset(&conn->notifSock, 0, sizeof(conn->notifSock));

  copyStringToBuf(socketAddrToHostString(&conn->peerBootstrapAddr), ipAddrBuf,
                  ipAddrBufLen);
  *remoteGpuIdx = remoteMeta.gpuIdx;

  if (!conn->sameProcess && (remoteMeta.notifPort <= 0 ||
                             connectNotifSocket(conn, &conn->peerBootstrapAddr,
                                                remoteMeta.notifPort) != 0)) {
    WARN("P2P/ENGINE : accept notification channel setup failed");
    flagcxP2pEngineConnDestroy(conn);
    bootstrapClose(bsConn);
    return NULL;
  }

  // Step 5: Exchange the initial MR table and activate dynamic publication.
  if (bootstrapFinalizeConnection(bsConn, conn) != 0) {
    WARN("P2P/ENGINE : accept MR handshake failed");
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

  if (conn->engine != NULL) {
    FlagcxP2pEngine *engine = conn->engine;
    std::unique_lock<FlagcxP2pSharedMutex> lifetimeLock(
        engine->connectionLifetimeMutex);
    {
      std::lock_guard<std::mutex> connectionLock(engine->connectionMutex);
      conn->mrPublishReady = false;
      engine->connections.erase(conn);
    }
    const bool quiesced =
        engine->scheduler == NULL || engine->scheduler->quiesceConnection(conn);
    if (!quiesced) {
      /* The scheduler has detached this connection from further polling. Do
         not close the comm or free transfers/MRs that a late native completion
         may still reference. EngineDestroy will retain the owning state. */
      engine->quarantineRequired.store(true, std::memory_order_release);
      return;
    }
    {
      std::lock_guard<std::mutex> xferLock(engine->state->xferMutex);
      for (auto it = engine->state->xfers.begin();
           it != engine->state->xfers.end();) {
        if (it->second.conn != conn) {
          ++it;
          continue;
        }
        if (it->second.kind == FLAGCX_P2P_XFER_IPC) {
          if (deviceAdaptor != NULL &&
              deviceAdaptor->eventSynchronize != NULL &&
              it->second.event != NULL)
            deviceAdaptor->eventSynchronize(it->second.event);
          cleanupIpcXfer(&it->second);
        }
        it = engine->state->xfers.erase(it);
      }
    }
    if (conn->sendComm && conn->sendComm != conn->recvComm) {
      engine->adaptor->closeSend(conn->sendComm);
    }
    if (conn->recvComm) {
      engine->adaptor->closeRecv(conn->recvComm);
    }
    if (conn->notifSockConnected) {
      flagcxSocketClose(&conn->notifSock);
    }
  }
  delete conn;
}

bool flagcxP2pEngineConnIsLocal(FlagcxP2pConn *conn) {
  return conn != NULL && conn->isLocal;
}

int flagcxP2pEngineRegEx(FlagcxP2pEngine *engine, uintptr_t data, size_t size,
                         int hintType, FlagcxP2pMr &mrId) {
  if (engine == NULL || data == 0 || size == 0)
    return -1;

  std::unique_lock<std::mutex> publishLock(engine->mrPublishMutex);

  auto resolvePtrType = [&](char *ipcHandleBuf,
                            uint32_t *ipcHandleSize) -> int {
    return detectPtrTypeAndMaybeCacheIpc(reinterpret_cast<void *>(data),
                                         hintType, ipcHandleBuf, ipcHandleSize);
  };

  if (!flagcxParamMrSortedLookup()) {
    /* Legacy: mutex + hash maps */
    std::lock_guard<std::mutex> lock(engine->state->memMutex);
    auto existing = engine->state->memRegInfo.find(data);
    if (existing != engine->state->memRegInfo.end()) {
      if (existing->second.size != size) {
        WARN("P2P Reg: addr 0x%lx size mismatch: existing %zu vs requested "
             "%zu",
             (unsigned long)data, existing->second.size, size);
        return -1;
      }
      mrId = existing->second.mrId;
      return 0;
    }

    const int netDev = chooseEngineNetDev(engine);
    const int ibDevN = resolveRegistrationDevice(engine, netDev);

    FlagcxP2pMemRegEntry entry;
    memset(&entry, 0, sizeof(entry));
    entry.mrId = engine->state->nextMrId++;
    entry.baseAddr = data;
    entry.size = size;
    entry.ibDevN = ibDevN;

    if (setEngineDevice(engine) != flagcxSuccess)
      return -1;
    entry.ptrType = resolvePtrType(entry.ipcHandle, &entry.ipcHandleSize);
    entry.hasIpc = entry.ptrType == FLAGCX_PTR_CUDA && entry.ipcHandleSize > 0;

    FlagcxP2pRegisteredMemory *registration = NULL;
    if (registerP2pMemory(engine, ibDevN, data, size, entry.ptrType,
                          &registration) != flagcxSuccess) {
      return -1;
    }
    entry.mhandle = registration;
    entry.mrInfo = registration->chunks.front().mrInfo;

    engine->state->memRegInfo[data] = entry;
    engine->state->mrToBaseAddr[entry.mrId] = data;
    if (!publishMrAdd(engine, registration, entry.mrId)) {
      WARN("P2P/ENGINE : failed to publish MR %llu to a connected peer",
           (unsigned long long)entry.mrId);
      const bool removalConfirmed = publishMrRemove(engine, entry.mrId);
      engine->state->memRegInfo.erase(data);
      engine->state->mrToBaseAddr.erase(entry.mrId);
      /* The caller retains ownership of data when registration fails and may
       * free it immediately.  Always revoke the physical keys before
       * returning; an unacknowledged peer can then only observe a transport
       * error through stale metadata, never access reused caller memory. */
      deregisterP2pMemory(engine, registration);
      if (!removalConfirmed)
        WARN("P2P/ENGINE : MR %llu rollback was not acknowledged; physical "
             "keys were revoked",
             (unsigned long long)entry.mrId);
      return -1;
    }
    mrId = entry.mrId;
    return 0;
  }

  /* Sorted registry is local to this engine's logical MR namespace. */
  std::lock_guard<std::mutex> lifecycleLock(engine->state->mrLifecycleMutex);

  /* Check for existing exact-match registration (dedup) */
  {
    struct flagcxMrEntry existing;
    struct flagcxMrExtension p2pExt;
    struct flagcxMrExtension *exts[FLAGCX_MR_OWNER_COUNT] = {&p2pExt, NULL,
                                                             NULL};
    if (flagcxMrRegistryFindExact(engine->state->mrRegistry, data, &existing,
                                  exts) == flagcxSuccess) {
      if (existing.size != size) {
        WARN("P2P Reg: addr 0x%lx size mismatch: existing %zu vs requested "
             "%zu",
             (unsigned long)data, existing.size, size);
        return -1;
      }
      if (p2pExt.type == FLAGCX_MR_OWNER_P2P) {
        mrId = p2pExt.p2p.mrId;
        return 0;
      }
    }
  }

  const int netDev = chooseEngineNetDev(engine);
  const int ibDevN = resolveRegistrationDevice(engine, netDev);

  /* Detect pointer type and IPC handle */
  char ipcHandle[FLAGCX_P2P_IPC_HANDLE_BYTES];
  uint32_t ipcHandleSize = 0;
  memset(ipcHandle, 0, sizeof(ipcHandle));

  if (setEngineDevice(engine) != flagcxSuccess)
    return -1;
  int ptrType = resolvePtrType(ipcHandle, &ipcHandleSize);
  bool hasIpc = ptrType == FLAGCX_PTR_CUDA && ipcHandleSize > 0;

  /* Register one logical MR as one or more transport MR chunks. */
  FlagcxP2pRegisteredMemory *registration = NULL;
  if (registerP2pMemory(engine, ibDevN, data, size, ptrType, &registration) !=
      flagcxSuccess) {
    return -1;
  }

  /* Build P2P extension */
  struct flagcxMrP2pExt *p2pExt =
      (struct flagcxMrP2pExt *)calloc(1, sizeof(struct flagcxMrP2pExt));
  if (p2pExt == NULL) {
    deregisterP2pMemory(engine, registration);
    return -1;
  }
  /* mrId=0 signals registry to assign from its monotonic counter */
  p2pExt->mrId = 0;
  p2pExt->hasIpc = hasIpc;
  p2pExt->ipcHandleSize = ipcHandleSize;
  memcpy(p2pExt->ipcHandle, ipcHandle, FLAGCX_P2P_IPC_HANDLE_BYTES);

  /* Register into unified registry */
  uint64_t assignedId = 0;
  flagcxResult_t res = flagcxMrRegistryRegister(
      engine->state->mrRegistry, data, size, ibDevN, ptrType,
      FLAGCX_MR_OWNER_P2P, registration, p2pExt, &assignedId);
  if (res != flagcxSuccess) {
    deregisterP2pMemory(engine, registration);
    free(p2pExt);
    return -1;
  }

  if (!publishMrAdd(engine, registration, assignedId)) {
    WARN("P2P/ENGINE : failed to publish MR %llu to a connected peer",
         (unsigned long long)assignedId);
    const bool removalConfirmed = publishMrRemove(engine, assignedId);
    void *removedExt = NULL;
    flagcxMrRegistryDeregister(engine->state->mrRegistry, data,
                               FLAGCX_MR_OWNER_P2P, NULL, &removedExt);
    free(removedExt);
    /* Registration failure returns ownership of data to the caller.  Revoke
     * every physical key even if the REMOVE acknowledgement was lost. */
    deregisterP2pMemory(engine, registration);
    if (!removalConfirmed)
      WARN("P2P/ENGINE : MR %llu rollback was not acknowledged; physical "
           "keys were revoked",
           (unsigned long long)assignedId);
    return -1;
  }

  mrId = assignedId;
  return 0;
}

int flagcxP2pEngineReg(FlagcxP2pEngine *engine, uintptr_t data, size_t size,
                       FlagcxP2pMr &mrId) {
  return flagcxP2pEngineRegEx(engine, data, size, 0, mrId);
}

void flagcxP2pEngineMrDestroy(FlagcxP2pEngine *engine, FlagcxP2pMr mr) {
  if (engine == NULL)
    return;

  std::unique_lock<std::mutex> publishLock(engine->mrPublishMutex);

  if (!flagcxParamMrSortedLookup()) {
    /* Legacy: mutex + hash maps */
    std::lock_guard<std::mutex> lock(engine->state->memMutex);
    auto mrIt = engine->state->mrToBaseAddr.find(mr);
    if (mrIt == engine->state->mrToBaseAddr.end())
      return;
    auto entryIt = engine->state->memRegInfo.find(mrIt->second);
    if (entryIt == engine->state->memRegInfo.end()) {
      engine->state->mrToBaseAddr.erase(mrIt);
      return;
    }
    if (!publishMrRemove(engine, mr)) {
      publishMrAdd(
          engine,
          static_cast<FlagcxP2pRegisteredMemory *>(entryIt->second.mhandle),
          mr);
      WARN("P2P/ENGINE : keeping MR %llu registered because remote removal "
           "was not acknowledged",
           (unsigned long long)mr);
      return;
    }
    deregisterP2pMemory(engine, static_cast<FlagcxP2pRegisteredMemory *>(
                                    entryIt->second.mhandle));
    engine->state->memRegInfo.erase(entryIt);
    engine->state->mrToBaseAddr.erase(mrIt);
    return;
  }

  std::lock_guard<std::mutex> lifecycleLock(engine->state->mrLifecycleMutex);

  /* Find entry by mrId to get baseAddr for deregister */
  struct flagcxMrEntry mrEntry;
  if (flagcxMrRegistryLookupById(engine->state->mrRegistry, mr, &mrEntry,
                                 NULL) != flagcxSuccess) {
    return;
  }

  if (!publishMrRemove(engine, mr)) {
    publishMrAdd(engine,
                 static_cast<FlagcxP2pRegisteredMemory *>(
                     mrEntry.mhandles[FLAGCX_MR_OWNER_IDX_P2P]),
                 mr);
    WARN("P2P/ENGINE : keeping MR %llu registered because remote removal "
         "was not acknowledged",
         (unsigned long long)mr);
    return;
  }

  /* Remove from registry first — prevents concurrent readers from finding it */
  void *removedExt = NULL;
  flagcxResult_t res;
  FLAGCXCHECKGOTO(
      flagcxMrRegistryDeregister(engine->state->mrRegistry, mrEntry.baseAddr,
                                 FLAGCX_MR_OWNER_P2P, NULL, &removedExt),
      res, fail);
  free(removedExt);

  /* Now safe to deregister every physical chunk in the logical MR. */
  deregisterP2pMemory(engine, static_cast<FlagcxP2pRegisteredMemory *>(
                                  mrEntry.mhandles[FLAGCX_MR_OWNER_IDX_P2P]));
  return;

fail:
  return;
}

int flagcxP2pEnginePrepareDesc(FlagcxP2pEngine *engine, FlagcxP2pMr mr,
                               const void *data, size_t size, char *descBuf) {
  if (engine == NULL || data == NULL || descBuf == NULL)
    return -1;

  if (!flagcxParamMrSortedLookup()) {
    /* Legacy: mutex + hash lookup */
    std::lock_guard<std::mutex> lock(engine->state->memMutex);
    auto mrIt = engine->state->mrToBaseAddr.find(mr);
    if (mrIt == engine->state->mrToBaseAddr.end())
      return -1;
    auto entryIt = engine->state->memRegInfo.find(mrIt->second);
    if (entryIt == engine->state->memRegInfo.end())
      return -1;
    FlagcxP2pMemRegEntry *entry = &entryIt->second;
    const FlagcxP2pMrChunk *chunk = findLocalMrChunk(
        static_cast<FlagcxP2pRegisteredMemory *>(entry->mhandle),
        reinterpret_cast<uintptr_t>(data));
    if (!memRegContains(*entry, reinterpret_cast<uintptr_t>(data), size) ||
        size > UINT32_MAX || chunk == NULL || chunk->mrInfo.nKeys == 0)
      return -1;
    FlagcxP2pRdmaDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.addr = (uint64_t)(uintptr_t)data;
    desc.size = (uint32_t)size;
    /* Preserve the logical MR identity across UpdateDesc retargeting. */
    desc.idx = (uint64_t)mr;
    desc.rid = countLocalMrChunks(
        static_cast<FlagcxP2pRegisteredMemory *>(entry->mhandle),
        reinterpret_cast<uintptr_t>(data), size);
    if (desc.rid == 0)
      return -1;
    fillDescKeys(&desc, chunk->mrInfo);
    flagcxP2pSerializeRdmaDesc(desc, descBuf);
    memcpy(entry->descBuf, descBuf, FLAGCX_P2P_DESC_SIZE);
    return 0;
  }

  /* New: single read-lock + containment search */
  uintptr_t dataAddr = (uintptr_t)data;

  if (flagcxMrRegistryRdLock(engine->state->mrRegistry) != flagcxSuccess)
    return -1;

  int count = flagcxMrRegistryCount(engine->state->mrRegistry);
  if (count == 0) {
    flagcxMrRegistryRdUnlock(engine->state->mrRegistry);
    return -1;
  }
  struct flagcxMrEntry *entries =
      flagcxMrRegistryEntries(engine->state->mrRegistry);

  /* Fast path: single-entry registry (common case with 1-4 MRs) */
  int idx = -1;
  if (count == 1) {
    if (entries[0].baseAddr <= dataAddr &&
        (dataAddr - entries[0].baseAddr) < entries[0].size)
      idx = 0;
  } else {
    /* O(log n) containment: find rightmost entry with baseAddr <= dataAddr */
    int lo = 0, hi = count - 1;
    while (lo <= hi) {
      int mid = lo + (hi - lo) / 2;
      if (entries[mid].baseAddr <= dataAddr) {
        idx = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    /* Verify containment */
    if (idx >= 0 && (dataAddr - entries[idx].baseAddr) >= entries[idx].size)
      idx = -1;
  }

  if (idx < 0 || !(entries[idx].ownerMask & FLAGCX_MR_OWNER_P2P) ||
      !entries[idx].p2p || entries[idx].p2p->mrId != (uint64_t)mr) {
    flagcxMrRegistryRdUnlock(engine->state->mrRegistry);
    return -1;
  }

  auto *registration = static_cast<FlagcxP2pRegisteredMemory *>(
      entries[idx].mhandles[FLAGCX_MR_OWNER_IDX_P2P]);
  const FlagcxP2pMrChunk *chunk = findLocalMrChunk(registration, dataAddr);
  if (chunk == NULL || chunk->mrInfo.nKeys == 0) {
    flagcxMrRegistryRdUnlock(engine->state->mrRegistry);
    return -1;
  }

  /* Verify (data, size) fits within the MR region (overflow-safe) */
  size_t offset = (size_t)(dataAddr - entries[idx].baseAddr);
  if (size > entries[idx].size - offset) {
    flagcxMrRegistryRdUnlock(engine->state->mrRegistry);
    return -1;
  }

  if (size > UINT32_MAX) {
    flagcxMrRegistryRdUnlock(engine->state->mrRegistry);
    return -1;
  }

  FlagcxP2pRdmaDesc desc;
  memset(&desc, 0, sizeof(desc));
  desc.addr = (uint64_t)dataAddr;
  desc.size = (uint32_t)size;
  /* Preserve the logical MR identity across UpdateDesc retargeting. */
  desc.idx = (uint64_t)mr;
  desc.rid = countLocalMrChunks(registration, dataAddr, size);
  if (desc.rid == 0) {
    flagcxMrRegistryRdUnlock(engine->state->mrRegistry);
    return -1;
  }
  fillDescKeys(&desc, chunk->mrInfo);

  flagcxP2pSerializeRdmaDesc(desc, descBuf);
  flagcxMrRegistryRdUnlock(engine->state->mrRegistry);
  return 0;
}

int flagcxP2pEngineUpdateDesc(FlagcxP2pRdmaDesc &desc, uint64_t remoteAddr,
                              uint32_t size) {
  desc.addr = remoteAddr;
  desc.size = size;
  return 0;
}

static int addNetTransfer(FlagcxP2pConn *conn,
                          const std::shared_ptr<FlagcxP2pNetTask> &task,
                          uint64_t *transferId) {
  if (conn == NULL || conn->engine == NULL || conn->engine->state == NULL ||
      transferId == NULL)
    return -1;
  if (!task) {
    *transferId = 0;
    return 0;
  }
  std::lock_guard<std::mutex> xferLock(conn->engine->state->xferMutex);
  const uint64_t xferId = conn->engine->state->nextXferId++;
  FlagcxP2pXfer xfer;
  xfer.kind = FLAGCX_P2P_XFER_NET;
  xfer.conn = conn;
  xfer.total = 1;
  xfer.completed = 0;
  xfer.stream = NULL;
  xfer.event = NULL;
  xfer.netTask = task;
  conn->engine->state->xfers[xferId] = std::move(xfer);
  *transferId = xferId;
  return 0;
}

static flagcxResult_t
planNetTransfer(FlagcxP2pConn *conn, const FlagcxP2pMemRegEntry &localEntry,
                void *localData, size_t size, const FlagcxP2pRdmaDesc &desc,
                bool isWrite,
                std::vector<struct flagcxP2pTransportSlice> *planned) {
  if (conn == NULL || planned == NULL ||
      !memRegContains(localEntry, reinterpret_cast<uintptr_t>(localData),
                      size) ||
      size > desc.size)
    return flagcxInvalidArgument;
  if (size == 0)
    return flagcxSuccess;

  auto *registration =
      static_cast<FlagcxP2pRegisteredMemory *>(localEntry.mhandle);
  if (registration == NULL)
    return flagcxInvalidArgument;

  struct flagcxNetMrInfo fallbackRemoteInfo;
  extractDescMrInfo(desc, &fallbackRemoteInfo);
  if (fallbackRemoteInfo.nKeys == 0)
    return flagcxInvalidArgument;

  uintptr_t localVa = reinterpret_cast<uintptr_t>(localData);
  uint64_t remoteVa = desc.addr;
  size_t remaining = size;
  const auto &config = flagcxP2pGlobalConfig();
  while (remaining > 0) {
    const FlagcxP2pMrChunk *localChunk =
        findLocalMrChunk(registration, localVa);
    if (localChunk == NULL)
      return flagcxInvalidArgument;

    const size_t localRemaining =
        localChunk->size - (size_t)(localVa - localChunk->baseAddr);
    size_t remoteRemaining = remaining;
    FlagcxP2pRemoteRegion remoteChunk;
    const bool hasRemoteChunk = findRemoteMrChunk(conn, remoteVa, &remoteChunk);
    if (!hasRemoteChunk && desc.rid > 1)
      return flagcxInvalidArgument;
    const struct flagcxNetMrInfo *remoteMrInfo = &fallbackRemoteInfo;
    if (hasRemoteChunk) {
      /* UpdateDesc may retarget addr/size, but it must not escape the logical
       * MR from which the descriptor was created. Descriptors produced by
       * older peers have idx == 0 and retain the legacy key-only behavior. */
      if (desc.idx != 0 && remoteChunk.mrId != desc.idx)
        return flagcxInvalidArgument;
      remoteRemaining =
          remoteChunk.size - (size_t)(remoteVa - remoteChunk.baseAddr);
      remoteMrInfo = &remoteChunk.mrInfo;
    }
    const uint32_t sliceLength =
        flagcxP2pPlanSliceLength(remaining, localRemaining, remoteRemaining,
                                 config.sliceSize, config.fragmentLimit);
    if (sliceLength == 0 || remoteMrInfo->nKeys == 0)
      return flagcxInvalidArgument;

    struct flagcxP2pTransportSlice slice;
    memset(&slice, 0, sizeof(slice));
    slice.localVa = localVa;
    slice.remoteVa = remoteVa;
    slice.length = (uint32_t)sliceLength;
    slice.opcode =
        isWrite ? FLAGCX_P2P_TRANSPORT_WRITE : FLAGCX_P2P_TRANSPORT_READ;
    slice.localMrHandle = localChunk->adaptorMrHandle;
    slice.localMrInfo = localChunk->mrInfo;
    slice.remoteMrInfo = *remoteMrInfo;
    planned->push_back(slice);

    localVa += sliceLength;
    remoteVa += sliceLength;
    remaining -= sliceLength;
  }
  return flagcxSuccess;
}

static int
submitNetTransfer(FlagcxP2pConn *conn,
                  const std::vector<struct flagcxP2pTransportSlice> &planned,
                  uint64_t *transferId) {
  if (conn == NULL || transferId == NULL || conn->engine == NULL ||
      conn->engine->transportOps == NULL)
    return -1;
  if (planned.empty()) {
    *transferId = 0;
    return 0;
  }

  if (conn->engine->scheduler == NULL)
    return -1;
  std::shared_ptr<FlagcxP2pNetTask> task;
  if (conn->engine->scheduler->submit(conn, planned, &task) != flagcxSuccess)
    return -1;
  return addNetTransfer(conn, task, transferId);
}

int flagcxP2pEngineRead(FlagcxP2pConn *conn, FlagcxP2pMr mr, const void *data,
                        size_t size, FlagcxP2pRdmaDesc desc,
                        uint64_t *transferId) {
  if (conn == NULL || data == NULL || transferId == NULL ||
      conn->engine == NULL)
    return -1;
  FlagcxP2pEngine *engine = conn->engine;
  FlagcxP2pSharedLock lifetimeLock(engine->connectionLifetimeMutex);
  if (conn->closing.load(std::memory_order_acquire))
    return -1;

  FlagcxP2pMemRegEntry localEntry;
  if (!findMemRegByMr(engine, mr, &localEntry) ||
      !memRegContains(localEntry, reinterpret_cast<uintptr_t>(data), size))
    return -1;

  if (selectP2pDataPath(conn, false) == FLAGCX_P2P_PATH_LOCAL_DIRECT) {
    std::vector<void *> localVec(1, const_cast<void *>(data));
    std::vector<size_t> sizeVec(1, size);
    std::vector<FlagcxP2pRdmaDesc> descs(1, desc);
    std::vector<char *> ipcBufs;
    return startLocalTransfer(conn, localVec, sizeVec, descs, 1, transferId,
                              ipcBufs, false);
  }

  std::vector<struct flagcxP2pTransportSlice> planned;
  if (planNetTransfer(conn, localEntry, const_cast<void *>(data), size, desc,
                      false, &planned) != flagcxSuccess) {
    return -1;
  }
  return submitNetTransfer(conn, planned, transferId);
}

int flagcxP2pEngineReadVector(FlagcxP2pConn *conn,
                              std::vector<FlagcxP2pMr> mrIds,
                              std::vector<void *> dstVec,
                              std::vector<size_t> sizeVec,
                              std::vector<FlagcxP2pRdmaDesc> descs, int numIovs,
                              uint64_t *transferId,
                              std::vector<char *> ipcBufs) {
  if (conn == NULL || numIovs <= 0 || transferId == NULL ||
      conn->engine == NULL) {
    fprintf(stderr,
            "[FlagCX P2P] ReadVector early exit: invalid args (conn=%p, "
            "numIovs=%d, transferId=%p)\n",
            conn, numIovs, (void *)transferId);
    return -1;
  }
  FlagcxP2pEngine *engine = conn->engine;
  FlagcxP2pSharedLock lifetimeLock(engine->connectionLifetimeMutex);
  if (conn->closing.load(std::memory_order_acquire))
    return -1;

  if (dstVec.size() < static_cast<size_t>(numIovs) ||
      sizeVec.size() < static_cast<size_t>(numIovs) ||
      descs.size() < static_cast<size_t>(numIovs)) {
    fprintf(stderr,
            "[FlagCX P2P] ReadVector early exit: vector length mismatch "
            "(numIovs=%d)\n",
            numIovs);
    return -1;
  }

  if (mrIds.size() < static_cast<size_t>(numIovs)) {
    fprintf(stderr,
            "[FlagCX P2P] ReadVector early exit: mrIds length mismatch "
            "(numIovs=%d)\n",
            numIovs);
    return -1;
  }

  std::vector<FlagcxP2pMemRegEntry> localEntries(numIovs);
  for (int i = 0; i < numIovs; i++) {
    if (!findMemRegByMr(conn->engine, mrIds[i], &localEntries[i])) {
      fprintf(stderr,
              "[FlagCX P2P] ReadVector memReg lookup failed: iov=%d, mr=%lu\n",
              i, (unsigned long)mrIds[i]);
      return -1;
    }

    if (!memRegContains(localEntries[i], reinterpret_cast<uintptr_t>(dstVec[i]),
                        sizeVec[i])) {
      fprintf(stderr,
              "[FlagCX P2P] ReadVector memReg bounds check failed: iov=%d, "
              "mr=%lu, addr=%p, size=%zu\n",
              i, (unsigned long)mrIds[i], dstVec[i], sizeVec[i]);
      return -1;
    }
  }

  if (selectP2pDataPath(conn, !ipcBufs.empty()) != FLAGCX_P2P_PATH_NET) {
    fprintf(stderr,
            "[FlagCX P2P] ReadVector taking local transfer path: numIovs=%d\n",
            numIovs);
    int rc = startLocalTransfer(conn, dstVec, sizeVec, descs, numIovs,
                                transferId, ipcBufs, false);
    fprintf(stderr, "[FlagCX P2P] ReadVector local transfer returned: rc=%d\n",
            rc);
    return rc;
  }

  std::vector<struct flagcxP2pTransportSlice> planned;
  for (int i = 0; i < numIovs; ++i) {
    if (planNetTransfer(conn, localEntries[i], dstVec[i], sizeVec[i], descs[i],
                        false, &planned) != flagcxSuccess) {
      return -1;
    }
  }
  return submitNetTransfer(conn, planned, transferId);
}

int flagcxP2pEngineWrite(FlagcxP2pConn *conn, FlagcxP2pMr mr, const void *data,
                         size_t size, FlagcxP2pRdmaDesc desc,
                         uint64_t *transferId) {
  if (conn == NULL || data == NULL || transferId == NULL ||
      conn->engine == NULL)
    return -1;
  FlagcxP2pEngine *engine = conn->engine;
  FlagcxP2pSharedLock lifetimeLock(engine->connectionLifetimeMutex);
  if (conn->closing.load(std::memory_order_acquire))
    return -1;

  FlagcxP2pMemRegEntry localEntry;
  if (!findMemRegByMr(engine, mr, &localEntry) ||
      !memRegContains(localEntry, reinterpret_cast<uintptr_t>(data), size))
    return -1;

  if (selectP2pDataPath(conn, false) == FLAGCX_P2P_PATH_LOCAL_DIRECT) {
    std::vector<void *> localVec(1, const_cast<void *>(data));
    std::vector<size_t> sizeVec(1, size);
    std::vector<FlagcxP2pRdmaDesc> descs(1, desc);
    std::vector<char *> ipcBufs;
    return startLocalTransfer(conn, localVec, sizeVec, descs, 1, transferId,
                              ipcBufs, true);
  }

  std::vector<struct flagcxP2pTransportSlice> planned;
  if (planNetTransfer(conn, localEntry, const_cast<void *>(data), size, desc,
                      true, &planned) != flagcxSuccess) {
    return -1;
  }
  return submitNetTransfer(conn, planned, transferId);
}

int flagcxP2pEngineWriteVector(FlagcxP2pConn *conn,
                               const std::vector<FlagcxP2pMr> &mrIds,
                               const std::vector<void *> &srcVec,
                               const std::vector<size_t> &sizeVec,
                               const std::vector<FlagcxP2pRdmaDesc> &descs,
                               int numIovs, uint64_t *transferId,
                               const std::vector<char *> &ipcBufs) {
  if (conn == NULL || numIovs <= 0 || transferId == NULL ||
      conn->engine == NULL)
    return -1;
  FlagcxP2pEngine *engine = conn->engine;
  FlagcxP2pSharedLock lifetimeLock(engine->connectionLifetimeMutex);
  if (conn->closing.load(std::memory_order_acquire))
    return -1;

  if (srcVec.size() < static_cast<size_t>(numIovs) ||
      sizeVec.size() < static_cast<size_t>(numIovs) ||
      descs.size() < static_cast<size_t>(numIovs))
    return -1;

  if (mrIds.size() < static_cast<size_t>(numIovs))
    return -1;

  std::vector<FlagcxP2pMemRegEntry> localEntries(numIovs);
  for (int i = 0; i < numIovs; i++) {
    if (!findMemRegByMr(conn->engine, mrIds[i], &localEntries[i]))
      return -1;

    if (!memRegContains(localEntries[i], reinterpret_cast<uintptr_t>(srcVec[i]),
                        sizeVec[i]))
      return -1;
  }

  if (selectP2pDataPath(conn, !ipcBufs.empty()) != FLAGCX_P2P_PATH_NET) {
    return startLocalTransfer(conn, srcVec, sizeVec, descs, numIovs, transferId,
                              ipcBufs, true);
  }

  std::vector<struct flagcxP2pTransportSlice> planned;
  for (int i = 0; i < numIovs; ++i) {
    if (planNetTransfer(conn, localEntries[i], srcVec[i], sizeVec[i], descs[i],
                        true, &planned) != flagcxSuccess) {
      return -1;
    }
  }
  return submitNetTransfer(conn, planned, transferId);
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
  if (conn == NULL || conn->engine == NULL || conn->engine->state == NULL)
    return true;
  FlagcxP2pEngine *engine = conn->engine;
  FlagcxP2pSharedLock lifetimeLock(engine->connectionLifetimeMutex);
  if (conn->closing.load(std::memory_order_acquire))
    return true;
  std::lock_guard<std::mutex> lock(engine->state->xferMutex);
  std::unordered_map<uint64_t, FlagcxP2pXfer>::iterator it =
      engine->state->xfers.find(transferId);
  if (it == engine->state->xfers.end())
    return true;

  FlagcxP2pXfer &xfer = it->second;
  if (xfer.kind == FLAGCX_P2P_XFER_IPC) {
    if (deviceAdaptor == NULL || deviceAdaptor->eventQuery == NULL) {
      cleanupIpcXfer(&xfer);
      engine->state->xfers.erase(it);
      return true;
    }

    const flagcxResult_t queryRes = deviceAdaptor->eventQuery(xfer.event);
    if (queryRes == flagcxSuccess) {
      cleanupIpcXfer(&xfer);
      engine->state->xfers.erase(it);
      return true;
    }
    if (queryRes != flagcxInProgress) {
      cleanupIpcXfer(&xfer);
      engine->state->xfers.erase(it);
      return true;
    }
    return false;
  }

  if (!xfer.netTask) {
    engine->state->xfers.erase(it);
    return true;
  }
  if (xfer.netTask->done.load(std::memory_order_acquire)) {
    if (xfer.netTask->failed.load(std::memory_order_acquire))
      WARN("P2P network transport reported a failed transfer");
    engine->state->xfers.erase(it);
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
  return -1;
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
  INFO(FLAGCX_INIT, "P2P/ENGINE : RPC server thread started (port=%d)",
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
  FlagcxP2pRemoteRegion first;
  if (!findRemoteMrChunk(conn, remoteVa, &first) || first.mrInfo.nKeys == 0)
    return -1;

  uint64_t current = remoteVa;
  size_t remaining = size;
  uint32_t chunkCount = 0;
  while (remaining > 0) {
    FlagcxP2pRemoteRegion region;
    if (!findRemoteMrChunk(conn, current, &region) || region.mrInfo.nKeys == 0)
      return -1;
    if (region.ownerEndpointId != first.ownerEndpointId ||
        region.mrId != first.mrId || region.ptrType != first.ptrType)
      return -1;
    const size_t available = region.size - (size_t)(current - region.baseAddr);
    if (available == 0)
      return -1;
    const size_t consumed = std::min(remaining, available);
    if (consumed > UINT64_MAX - current)
      return -1;
    current += consumed;
    remaining -= consumed;
    chunkCount++;
  }

  memset(desc, 0, sizeof(*desc));
  desc->addr = remoteVa;
  desc->size = size;
  desc->idx = first.mrId;
  desc->rid = chunkCount;
  fillDescKeys(desc, first.mrInfo);
  return 0;
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

int flagcxP2pRpcRegisterHost(void *engine, uint64_t addr, uint64_t size,
                             uint64_t *mrIdOut) {
  if (mrIdOut == NULL)
    return -1;
  FlagcxP2pMr mrId = 0;
  const int rc = flagcxP2pEngineRegEx(
      reinterpret_cast<FlagcxP2pEngine *>(engine), static_cast<uintptr_t>(addr),
      static_cast<size_t>(size), FLAGCX_PTR_HOST, mrId);
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
  // scans the per-conn remoteRegions table, not the global MR registry.
  for (int i = 0; i < count; i++) {
    srcVec[i] = reinterpret_cast<void *>(static_cast<uintptr_t>(srcVa[i]));
    sizeVec[i] = static_cast<size_t>(sizes[i]);
    if (flagcxP2pEngineMakeDesc(conn, dstVa[i], static_cast<uint32_t>(sizes[i]),
                                &descs[i]) != 0) {
      WARN("P2P/ENGINE : BatchWriteSync MakeDesc failed for remote VA "
           "0x%llx size %llu",
           (unsigned long long)dstVa[i], (unsigned long long)sizes[i]);
      return -1;
    }
  }

  std::vector<FlagcxP2pMemRegEntry> localEntries(count);
  {
    std::vector<uintptr_t> srcAddrs(count);
    for (int i = 0; i < count; i++)
      srcAddrs[i] = static_cast<uintptr_t>(srcVa[i]);
    if (!findMemRegBatch(conn->engine, srcAddrs.data(), count,
                         localEntries.data())) {
      WARN("P2P/ENGINE : BatchWriteSync no local MR for source VA");
      return -1;
    }
  }
  for (int i = 0; i < count; i++) {
    if (!memRegContains(localEntries[i], static_cast<uintptr_t>(srcVa[i]),
                        static_cast<size_t>(sizes[i]))) {
      WARN("P2P/ENGINE : BatchWriteSync source VA 0x%llx size %llu out of MR "
           "bounds",
           (unsigned long long)srcVa[i], (unsigned long long)sizes[i]);
      return -1;
    }
  }
  std::vector<FlagcxP2pMr> mrVec(count);
  for (int i = 0; i < count; ++i)
    mrVec[i] = localEntries[i].mrId;
  return flagcxP2pEngineWriteVectorSync(conn, mrVec, srcVec, sizeVec, descs);
}

} // extern "C"

std::vector<FlagcxP2pNotifyMsg> flagcxP2pEngineGetNotifs() {
  std::lock_guard<std::mutex> lock(gNotifyMutex);
  std::vector<FlagcxP2pNotifyMsg> result;
  result.swap(gNotifyList);
  return result;
}

void flagcxP2pNotifyAppend(const FlagcxP2pNotifyMsg &msg) {
  std::lock_guard<std::mutex> lock(gNotifyMutex);
  gNotifyList.push_back(msg);
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
  wireMsg.type = FLAGCX_P2P_NOTIF_USER;
  wireMsg.payload = *notifyMsg;
  if (sendNotifWire(conn, wireMsg) != 0) {
    return -1;
  }
  return sizeof(FlagcxP2pNotifyMsg);
}

int flagcxP2pEngineGetIpcInfo(FlagcxP2pEngine *engine, uintptr_t addr,
                              char *ipcBuf, bool *hasIpc) {
  if (engine == NULL || ipcBuf == NULL || hasIpc == NULL)
    return -1;

  *hasIpc = false;
  FlagcxP2pMemRegEntry entry;
  if (!findMemReg(engine, addr, &entry))
    return -1;

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
