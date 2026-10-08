/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * FlagCX shared-transport P2P Engine — implements the flagcx_p2p.h API when
 * built with USE_SHARED_P2P_ENGINE=1.
 *
 * Architecture: transport-neutral Engine over the shared net adaptors and
 * P2P topo manager. Mirrors the structure of UCCL's
 * uccl_engine.cc so that a NIXL FlagCX backend plugin can wrap it in
 * exactly the same way the NIXL UCCL plugin wraps uccl_engine.
 ************************************************************************/

#include "flagcx_p2p.h"

#include "adaptor.h"
#ifdef USE_ACCL_BAREX
#include "barex_runtime.h"
#endif
#include "bootstrap.h"
#include "cpuset.h"
#include "debug.h"
#include "flagcx_mr_registry.h"
#include "flagcx_net.h"
#include "flagcx_net_adaptor.h"
#include "ib_common.h"
#include "onesided_types.h"
#include "p2p.h"
#include "p2p_control.h"
#include "p2p_engine_backend.h"
#include "p2p_engine_transport.h"
#include "p2p_pointer.h"
#include "p2p_scheduler.h"
#include "p2p_topo.h"
#include "p2p_visibility.h"
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
#include <functional>
#include <memory>
#include <mutex>
#include <poll.h>
#include <pthread.h>
#include <string>
#include <strings.h>
#include <thread>
#include <unordered_map>
#include <vector>
#if defined(__linux__)
#include <sys/epoll.h>
#endif
#include <unistd.h>

extern struct flagcxNetAdaptor flagcxNetIb;
#ifdef USE_ACCL_BAREX
extern struct flagcxNetAdaptor flagcxNetBarex;
#endif

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

struct FlagcxP2pListenHandleView {
  union flagcxSocketAddress connectAddr;
  uint64_t magic;
};
static_assert(sizeof(FlagcxP2pListenHandleView) <= FLAGCX_NET_HANDLE_MAXSIZE,
              "listen handle must fit in FLAGCX_NET_HANDLE_MAXSIZE");

enum {
  FLAGCX_P2P_MAX_NOTIF_PEERS = 64,
  FLAGCX_P2P_NOTIF_MAGIC = 0xDEADDEADu,
  FLAGCX_P2P_CTRL_FLAG_LOCAL = 1u << 0,
  FLAGCX_P2P_CTRL_FLAG_SAME_PROCESS = 1u << 1,
  FLAGCX_P2P_IPC_FLAG_CUDA = 1u << 0,
};

static_assert(FLAGCX_P2P_IPC_HANDLE_BYTES == FLAGCX_MR_IPC_HANDLE_BYTES,
              "IPC handle size mismatch between P2P and MR registry");

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
  struct flagcxNetMrInfo info;
};

struct FlagcxP2pMemRegWire {
  uint64_t baseAddr;
  uint64_t size;
  uint32_t nKeys;
  uint32_t rkeys[FLAGCX_NET_MAX_MR_KEYS];
  uint32_t reserved;
};
static_assert(sizeof(FlagcxP2pMemRegWire) == 56,
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
  bool isBarex;
  struct flagcxP2pTopoManager *topoMgr;
  int nDevs;
  int localGpuIdx;
  FlagcxP2pListener listeners[MAX_IB_VDEVS];

  struct flagcxSocket notifListenSock;
  bool notifListenActive;
  int notifListenPort;
#if defined(__linux__)
  int notifEpollFd;
#endif
  std::atomic<bool> stopNotif;
  std::thread notifThread;
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
  std::atomic<uint64_t> runtimeSliceConfig;
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
  // The main IB adaptor owns a non-atomic request table and CQ progress state.
  // All post/test operations using this connection's sendComm share this lock.
  std::mutex progressMutex;
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
  std::shared_ptr<struct flagcxP2pMrRecord> record;
};

struct FlagcxP2pTransferMr {
  uintptr_t base = 0;
  size_t size = 0;
  struct flagcxNetMrInfo info = {};
  struct flagcxOneSideHandleInfo handle = {};

  void init(uintptr_t regionBase, size_t regionSize,
            const struct flagcxNetMrInfo &regionInfo, void *localHandle) {
    base = regionBase;
    size = regionSize;
    info = regionInfo;
    handle.baseVas = &base;
    handle.regionSizes = &size;
    handle.mrInfos = &info;
    handle.localMrHandle = localHandle;
    handle.nRanks = 1;
  }
};

struct FlagcxP2pTransferStorage {
  struct FlagcxP2pTransferMr src;
  struct FlagcxP2pTransferMr dst;
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
  std::unique_ptr<struct flagcxP2pNetBackendContext> netBackend;
  std::unique_ptr<struct flagcxP2pTransfer> transfer;
  std::vector<struct FlagcxP2pTransferStorage> transferStorage;
  std::vector<uint64_t> laneMasks;
};

static std::vector<FlagcxP2pNotifyMsg> &notifyList() {
  static std::vector<FlagcxP2pNotifyMsg> list;
  return list;
}

static std::mutex &notifyMutex() {
  static std::mutex mu;
  return mu;
}

static std::unordered_map<uint64_t, FlagcxP2pXfer> &xferMap() {
  static std::unordered_map<uint64_t, FlagcxP2pXfer> map;
  return map;
}

static std::mutex &xferMutex() {
  static std::mutex mu;
  return mu;
}

static uint64_t &nextXferId() {
  static uint64_t id = 1;
  return id;
}

#define gNotifyList notifyList()
#define gNotifyMutex notifyMutex()
#define gXferMap xferMap()
#define gXferMutex xferMutex()
#define gNextXferId nextXferId()

static pthread_mutex_t gMrLifecycleMutex = PTHREAD_MUTEX_INITIALIZER;

/* Legacy hash-map MR storage (used when FLAGCX_MR_SORTED_LOOKUP=0) */
static std::unordered_map<uintptr_t, FlagcxP2pMemRegEntry> &memRegInfo() {
  static std::unordered_map<uintptr_t, FlagcxP2pMemRegEntry> info;
  return info;
}
static std::unordered_map<FlagcxP2pMr, uintptr_t> &mrToBaseAddr() {
  static std::unordered_map<FlagcxP2pMr, uintptr_t> map;
  return map;
}
static std::mutex &memMutex() {
  static std::mutex mu;
  return mu;
}
static uint64_t &nextMrId() {
  static uint64_t id = 1;
  return id;
}
static std::unordered_map<FlagcxP2pMr,
                          std::shared_ptr<struct flagcxP2pMrRecord>> &
mrRecords() {
  static std::unordered_map<FlagcxP2pMr,
                            std::shared_ptr<struct flagcxP2pMrRecord>>
      records;
  return records;
}
static std::mutex &mrRecordMutex() {
  static std::mutex mu;
  return mu;
}
#define gMemRegInfo memRegInfo()
#define gMrToBaseAddr mrToBaseAddr()
#define gMemMutex memMutex()
#define gNextMrId nextMrId()
#define gMrRecords mrRecords()
#define gMrRecordMutex mrRecordMutex()

static std::shared_ptr<struct flagcxP2pMrRecord> findMrRecord(FlagcxP2pMr mr) {
  std::lock_guard<std::mutex> lock(gMrRecordMutex);
  auto it = gMrRecords.find(mr);
  return it == gMrRecords.end() ? nullptr : it->second;
}

static void
storeMrRecord(FlagcxP2pMr mr,
              const std::shared_ptr<struct flagcxP2pMrRecord> &record) {
  std::lock_guard<std::mutex> lock(gMrRecordMutex);
  gMrRecords[mr] = record;
}

static void eraseMrRecord(FlagcxP2pMr mr) {
  std::lock_guard<std::mutex> lock(gMrRecordMutex);
  gMrRecords.erase(mr);
}

static bool findMemReg(uintptr_t addr, FlagcxP2pMemRegEntry *out) {
  if (!flagcxParamMrSortedLookup()) {
    /* Legacy: O(n) linear scan over hash map */
    std::lock_guard<std::mutex> lock(gMemMutex);
    for (auto it = gMemRegInfo.begin(); it != gMemRegInfo.end(); ++it) {
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

  if (flagcxMrRegistryLookup(flagcxGlobalMrRegistry, addr, &entry, exts) !=
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
    out->hasIpc = p2pExt.p2p.hasIpc;
    out->ipcHandleSize = p2pExt.p2p.ipcHandleSize;
    memcpy(out->ipcHandle, p2pExt.p2p.ipcHandle, FLAGCX_P2P_IPC_HANDLE_BYTES);
    out->record = findMrRecord(out->mrId);
  }
  return true;
}

/*
 * Batch containment lookup — acquires gMemMutex once in legacy mode.
 * Returns false (and stops) if any addr is not found.
 */
static bool findMemRegBatch(const uintptr_t *addrs, int count,
                            FlagcxP2pMemRegEntry *out) {
  if (!flagcxParamMrSortedLookup()) {
    std::lock_guard<std::mutex> lock(gMemMutex);
    for (int i = 0; i < count; i++) {
      bool found = false;
      for (auto it = gMemRegInfo.begin(); it != gMemRegInfo.end(); ++it) {
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
    if (!findMemReg(addrs[i], &out[i]))
      return false;
  }
  return true;
}

static bool findMemRegByMr(FlagcxP2pMr mr, FlagcxP2pMemRegEntry *out) {
  if (!flagcxParamMrSortedLookup()) {
    /* Legacy: O(1) hash lookup */
    std::lock_guard<std::mutex> lock(gMemMutex);
    auto mrIt = gMrToBaseAddr.find(mr);
    if (mrIt == gMrToBaseAddr.end())
      return false;
    auto entryIt = gMemRegInfo.find(mrIt->second);
    if (entryIt == gMemRegInfo.end())
      return false;
    if (out)
      *out = entryIt->second;
    return true;
  }

  /* New: O(log n) sorted-array registry lookup */
  struct flagcxMrEntry found;
  struct flagcxMrExtension p2pExt;
  struct flagcxMrExtension *exts[FLAGCX_MR_OWNER_COUNT] = {&p2pExt, NULL, NULL};
  if (flagcxMrRegistryLookupById(flagcxGlobalMrRegistry, mr, &found, exts) !=
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
    out->hasIpc = p2pExt.p2p.hasIpc;
    out->ipcHandleSize = p2pExt.p2p.ipcHandleSize;
    memcpy(out->ipcHandle, p2pExt.p2p.ipcHandle, FLAGCX_P2P_IPC_HANDLE_BYTES);
    out->record = findMrRecord(out->mrId);
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

static const flagcxP2pMrSegment *
findMrSegmentForRange(const flagcxP2pMrRecord &record, uintptr_t address,
                      size_t size) {
  if (size == 0 && address == record.base + record.size)
    return record.segments.empty() ? nullptr : &record.segments.back();
  for (const auto &segment : record.segments) {
    if (address >= segment.base && address - segment.base < segment.size)
      return &segment;
  }
  return nullptr;
}

static bool remoteDescContains(const FlagcxP2pRdmaDesc &desc, size_t size);
static int resolveIbDevN(int netDev);

static bool descMrInfo(const FlagcxP2pRdmaDesc &desc,
                       struct flagcxNetMrInfo *info) {
  if (info == NULL)
    return false;
  memset(info, 0, sizeof(*info));
  const uint32_t nKeys = desc.nmsgs == 0 ? 1 : desc.nmsgs;
  if (nKeys == 0 || nKeys > FLAGCX_NET_MAX_MR_KEYS)
    return false;
  info->nKeys = nKeys;
  for (uint32_t i = 0; i < nKeys; ++i) {
    if (flagcxP2pDescGetKey(&desc, i, &info->rkeys[i]) != flagcxSuccess)
      return false;
  }
  return true;
}

struct FlagcxP2pSliceSpec {
  int iov;
  size_t offset;
  size_t size;
  size_t localSegment;
  uint64_t remoteBase;
  size_t remoteSize;
  struct flagcxNetMrInfo remoteInfo;
};

static size_t findRemoteRegion(const FlagcxP2pConn *conn, uint64_t address) {
  size_t lo = 0;
  size_t hi = conn->remoteRegions.size();
  while (lo < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    if (conn->remoteRegions[mid].baseAddr <= address)
      lo = mid + 1;
    else
      hi = mid;
  }
  if (lo == 0)
    return SIZE_MAX;
  const size_t index = lo - 1;
  const FlagcxP2pRemoteRegion &region = conn->remoteRegions[index];
  return address >= region.baseAddr && address - region.baseAddr < region.size
             ? index
             : SIZE_MAX;
}

// Main-IB request slots and CQs belong to the connection, not to one Engine
// transfer. Drive sibling transfers before posting/polling the selected one so
// an unpolled transfer cannot retain every native request slot indefinitely.
static void progressConnectionTransfersLocked(FlagcxP2pConn *conn,
                                              uint64_t skipTransferId) {
  std::vector<struct flagcxP2pTransfer *> transfers;
  transfers.reserve(gXferMap.size());
  for (auto &entry : gXferMap) {
    FlagcxP2pXfer &xfer = entry.second;
    if (entry.first == skipTransferId || xfer.conn != conn || !xfer.transfer ||
        !xfer.transfer->initialized)
      continue;
    transfers.push_back(xfer.transfer.get());
  }
  if (!transfers.empty())
    (void)flagcxP2pTransferProgressMany(transfers.data(), transfers.size());
}

static int
startNetTransfer(FlagcxP2pConn *conn, const std::vector<void *> &dataVec,
                 const std::vector<size_t> &sizeVec,
                 const std::vector<FlagcxP2pRdmaDesc> &descs,
                 const std::vector<FlagcxP2pMemRegEntry> &localEntries,
                 int numIovs, bool write, uint64_t *transferId) {
  if (conn == NULL || conn->engine == NULL || conn->sendComm == NULL ||
      transferId == NULL || numIovs <= 0)
    return -1;

  const int connIbDevN =
      conn->engine->isBarex ? -1 : resolveIbDevN(conn->netDev);
  if (!conn->engine->isBarex && connIbDevN < 0)
    return -1;

  for (int i = 0; i < numIovs; ++i) {
    if (!conn->engine->isBarex && localEntries[i].ibDevN != connIbDevN) {
      WARN("NET/P2P_ENGINE : iov[%d] ibDevN mismatch (%d vs conn %d)", i,
           localEntries[i].ibDevN, connIbDevN);
      return -1;
    }
    if (!remoteDescContains(descs[i], sizeVec[i]) ||
        !memRegContains(localEntries[i],
                        reinterpret_cast<uintptr_t>(dataVec[i]), sizeVec[i]) ||
        localEntries[i].record == nullptr ||
        flagcxP2pMrRecordValidate(localEntries[i].record.get()) !=
            flagcxSuccess)
      return -1;
  }

  const uint64_t sliceConfig =
      conn->engine->runtimeSliceConfig.load(std::memory_order_acquire);
  const size_t sliceSize = sliceConfig >> 32;
  const size_t fragmentSize = uint32_t(sliceConfig);
  std::vector<FlagcxP2pSliceSpec> specs;
  for (int i = 0; i < numIovs; ++i) {
    if (sizeVec[i] == 0)
      continue;
    const auto &record = *localEntries[i].record;
    const uintptr_t localStart = reinterpret_cast<uintptr_t>(dataVec[i]);
    flagcxP2pMrRecord remoteRecord;
    if (conn->remoteRegions.empty()) {
      flagcxP2pMrSegment segment;
      segment.base = descs[i].addr;
      segment.size = descs[i].size;
      if (!descMrInfo(descs[i], &segment.keys))
        return -1;
      remoteRecord.base = segment.base;
      remoteRecord.size = segment.size;
      remoteRecord.segments.push_back(segment);
    } else {
      size_t regionIndex = findRemoteRegion(conn, descs[i].addr);
      if (regionIndex == SIZE_MAX)
        return -1;
      const uint64_t requestEnd = descs[i].addr + sizeVec[i];
      while (regionIndex < conn->remoteRegions.size()) {
        const auto &region = conn->remoteRegions[regionIndex];
        if (!remoteRecord.segments.empty() &&
            remoteRecord.segments.back().base +
                    remoteRecord.segments.back().size !=
                region.baseAddr)
          break;
        flagcxP2pMrSegment segment;
        segment.base = region.baseAddr;
        segment.size = region.size;
        segment.keys = region.info;
        remoteRecord.segments.push_back(segment);
        if (region.baseAddr + region.size >= requestEnd)
          break;
        ++regionIndex;
      }
      remoteRecord.base = remoteRecord.segments.front().base;
      remoteRecord.size = 0;
      for (const auto &segment : remoteRecord.segments)
        remoteRecord.size += segment.size;
    }

    std::vector<flagcxP2pMrPairSlice> pairSlices;
    if (flagcxP2pMrSplitPair(&record, localStart, &remoteRecord, descs[i].addr,
                             sizeVec[i], sliceSize, fragmentSize,
                             &pairSlices) != flagcxSuccess)
      return -1;
    for (const auto &pair : pairSlices) {
      const auto &remoteSegment =
          remoteRecord.segments[pair.remoteSegmentIndex];
      FlagcxP2pSliceSpec spec = {};
      spec.iov = i;
      spec.offset = pair.offset;
      spec.size = pair.size;
      spec.localSegment = pair.localSegmentIndex;
      spec.remoteBase = remoteSegment.base;
      spec.remoteSize = remoteSegment.size;
      spec.remoteInfo = remoteSegment.keys;
      specs.push_back(spec);
    }
  }
  if (specs.empty()) {
    *transferId = 0;
    return 0;
  }

  uint64_t xferId = 0;
  {
    std::lock_guard<std::mutex> lock(gXferMutex);
    xferId = gNextXferId++;
  }

  FlagcxP2pXfer xfer;
  xfer.kind = FLAGCX_P2P_XFER_NET;
  xfer.conn = conn;
  xfer.total = static_cast<int>(specs.size());
  xfer.completed = 0;
  xfer.stream = NULL;
  xfer.event = NULL;
  xfer.netBackend.reset(new flagcxP2pNetBackendContext);
  xfer.transfer.reset(new flagcxP2pTransfer);
  xfer.transferStorage.resize(specs.size());
  xfer.laneMasks.assign(specs.size(), 0);
  std::vector<flagcxP2pTransferOp> ops(specs.size());

  for (size_t i = 0; i < specs.size(); ++i) {
    const FlagcxP2pSliceSpec &spec = specs[i];
    const FlagcxP2pMemRegEntry &local = localEntries[spec.iov];
    const flagcxP2pMrSegment &localSegment =
        local.record->segments[spec.localSegment];
    const uintptr_t localAddress =
        reinterpret_cast<uintptr_t>(dataVec[spec.iov]);
    const uint64_t remoteAddress = descs[spec.iov].addr + spec.offset;
    FlagcxP2pTransferStorage &storage = xfer.transferStorage[i];
    if (write) {
      storage.src.init(localSegment.base, localSegment.size, localSegment.keys,
                       localSegment.adaptorMr);
      storage.dst.init(spec.remoteBase, spec.remoteSize, spec.remoteInfo, NULL);
      ops[i].srcOffset = localAddress + spec.offset - localSegment.base;
      ops[i].dstOffset = remoteAddress - spec.remoteBase;
      ops[i].srcMr = &storage.src.handle;
      ops[i].dstMr = &storage.dst.handle;
    } else {
      storage.src.init(spec.remoteBase, spec.remoteSize, spec.remoteInfo, NULL);
      storage.dst.init(localSegment.base, localSegment.size, localSegment.keys,
                       localSegment.adaptorMr);
      ops[i].srcOffset = remoteAddress - spec.remoteBase;
      ops[i].dstOffset = localAddress + spec.offset - localSegment.base;
      ops[i].srcMr = &storage.src.handle;
      ops[i].dstMr = &storage.dst.handle;
    }
    ops[i].size = spec.size;
    // Engine slicing proves these ranges independent. Stable per-slice keys
    // permit deterministic multi-QP use without inferring independence from
    // user addresses in the public RMA API.
    ops[i].orderingKey = flagcxP2pEngineOrderingKey(xferId, i);
    ops[i].submitFlags = FLAGCX_NET_SUBMIT_DATA | FLAGCX_NET_SUBMIT_INDEPENDENT;
    ops[i].laneMask = &xfer.laneMasks[i];
  }

  struct flagcxP2pTransferBackend backend = {};
  if (flagcxP2pNetBackendInit(xfer.netBackend.get(), conn->engine->adaptor,
                              conn->sendComm, &conn->progressMutex,
                              write ? 1 : 0, &backend) != flagcxSuccess)
    return -1;
  const FlagcxP2pGlobalConfig &config = flagcxP2pGlobalConfig();
  const uint32_t maxInFlight = static_cast<uint32_t>(
      std::max<size_t>(1, std::min<size_t>(specs.size(), config.maxRequests)));
  const uint32_t maxPostBatch = static_cast<uint32_t>(
      std::max<size_t>(1, std::min<size_t>(specs.size(), config.maxWrPerPost)));
  if (flagcxP2pTransferInit(xfer.transfer.get(), &backend, ops.data(),
                            ops.size(), maxInFlight, maxPostBatch, xferId,
                            1) != flagcxSuccess)
    return -1;

  {
    std::lock_guard<std::mutex> lock(gXferMutex);
    progressConnectionTransfersLocked(conn, 0);
  }
  struct flagcxP2pTransferStatus status = {};
  const flagcxResult_t progressResult =
      flagcxP2pTransferProgress(xfer.transfer.get(), &status);
  // A zero-accept fatal post completes the whole group synchronously. Do not
  // publish a transfer id for work that never reached the transport. A
  // partially accepted fatal post must remain tracked until its accepted
  // prefix drains; the synchronous API observes the terminal result below.
  if ((progressResult != flagcxSuccess && status.inFlight == 0) ||
      (status.done && status.result != flagcxSuccess)) {
    if (status.done)
      (void)flagcxP2pTransferReset(xfer.transfer.get());
    return -1;
  }

  {
    std::lock_guard<std::mutex> lock(gXferMutex);
    gXferMap.emplace(xferId, std::move(xfer));
  }
  *transferId = xferId;
  return 0;
}

static bool remoteDescContains(const FlagcxP2pRdmaDesc &desc, size_t size) {
  if (size > desc.size)
    return false;
  if (size != 0 && desc.addr == 0)
    return false;
  return desc.addr <= UINT64_MAX - size;
}

static int resolveIbDevN(int netDev) {
  if (netDev < 0 || netDev >= flagcxNMergedIbDevs)
    return -1;
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
    return -1;

  const bool portWasConfigured = getenv("FLAGCX_P2P_IB_PORT") != NULL;
  const int configuredPort = flagcxP2pGlobalConfig().ibPort;
  auto usesConfiguredPort = [&](int candidate) {
    if (engine->isBarex)
      return true;
    if (!portWasConfigured)
      return true;
    if (candidate < 0 || candidate >= flagcxNMergedIbDevs ||
        flagcxIbMergedDevs[candidate].ndevs <= 0)
      return false;
    for (int i = 0; i < flagcxIbMergedDevs[candidate].ndevs; ++i) {
      const int ibDevN = flagcxIbMergedDevs[candidate].devs[i];
      if (ibDevN < 0 || ibDevN >= flagcxNIbDevs ||
          flagcxIbDevs[ibDevN].portNum != configuredPort)
        return false;
    }
    return true;
  };

  int netDev = 0;
  if (engine->topoMgr) {
    if (flagcxP2pTopoGetNetDev(engine->topoMgr, engine->localGpuIdx, &netDev) !=
        flagcxSuccess) {
      netDev = 0;
    }
  }

  if (netDev >= 0 && netDev < engine->nDevs &&
      engine->listeners[netDev].listenComm != NULL &&
      usesConfiguredPort(netDev)) {
    return netDev;
  }

  for (int d = 0; d < engine->nDevs; d++) {
    if (engine->listeners[d].listenComm != NULL && usesConfiguredPort(d))
      return d;
  }
  if (portWasConfigured)
    WARN("P2P Engine: no main-IB device uses configured port %d",
         configuredPort);
  return -1;
}

static flagcxResult_t setEngineDevice(FlagcxP2pEngine *engine) {
  if (engine && deviceAdaptor && deviceAdaptor->setDevice) {
    return deviceAdaptor->setDevice(engine->localGpuIdx);
  }
  return flagcxSuccess;
}

static void releaseEngineMrHandle(FlagcxP2pEngine *engine, void *mhandle) {
  if (engine == NULL || mhandle == NULL)
    return;
  if (engine->isBarex) {
    flagcxResult_t result = engine->adaptor->deregMr(NULL, mhandle);
#ifdef USE_ACCL_BAREX
    // Engine cleanup consumes the provider handle even though the public API
    // cannot return a failure. Transfer retry ownership before dropping it.
    if (result != flagcxSuccess)
      (void)flagcxBarexRuntimeDeferMr(mhandle);
#else
    (void)result;
#endif
  } else {
    (void)flagcxIbEngineDeregMr(mhandle);
  }
}

static flagcxResult_t
registerEngineMr(FlagcxP2pEngine *engine, int netDev, uintptr_t data,
                 size_t size, int ptrType,
                 std::shared_ptr<struct flagcxP2pMrRecord> *recordOut) {
  if (engine == NULL || engine->adaptor == NULL || data == 0 || size == 0 ||
      recordOut == NULL)
    return flagcxInvalidArgument;
  if (data > UINTPTR_MAX - size)
    return flagcxInvalidArgument;

  auto record = std::make_shared<struct flagcxP2pMrRecord>();
  record->base = data;
  record->size = size;
  size_t segmentSize = size;
#ifdef USE_ACCL_BAREX
  if (engine->isBarex)
    segmentSize = flagcxBarexRuntimeMrSegmentSize(ptrType);
#endif
  if (segmentSize == 0)
    return flagcxInvalidArgument;

  for (size_t offset = 0; offset < size;) {
    const size_t bytes = std::min(segmentSize, size - offset);
    struct flagcxP2pMrSegment segment;
    segment.base = data + offset;
    segment.size = bytes;
    flagcxResult_t result;
    if (engine->isBarex) {
      result = engine->adaptor->regMr(
          NULL, reinterpret_cast<void *>(segment.base), bytes, ptrType,
          FLAGCX_NET_MR_FLAG_NONE, &segment.adaptorMr);
    } else {
      result = flagcxIbEngineRegMr(
          netDev, reinterpret_cast<void *>(segment.base), bytes, ptrType,
          FLAGCX_NET_MR_FLAG_NONE, &segment.adaptorMr);
    }
    if (result != flagcxSuccess || segment.adaptorMr == NULL ||
        engine->adaptor->getMrInfo == NULL ||
        engine->adaptor->getMrInfo(segment.adaptorMr, &segment.keys) !=
            flagcxSuccess) {
      releaseEngineMrHandle(engine, segment.adaptorMr);
      for (auto it = record->segments.rbegin(); it != record->segments.rend();
           ++it)
        releaseEngineMrHandle(engine, it->adaptorMr);
      return result == flagcxSuccess ? flagcxInternalError : result;
    }
    record->segments.push_back(segment);
    offset += bytes;
  }
  if (flagcxP2pMrRecordValidate(record.get()) != flagcxSuccess) {
    for (auto &segment : record->segments)
      releaseEngineMrHandle(engine, segment.adaptorMr);
    return flagcxInternalError;
  }
  *recordOut = record;
  return flagcxSuccess;
}

static void
deregisterEngineMr(FlagcxP2pEngine *engine,
                   const std::shared_ptr<struct flagcxP2pMrRecord> &record) {
  if (engine == NULL || record == nullptr)
    return;
  for (auto it = record->segments.rbegin(); it != record->segments.rend(); ++it)
    releaseEngineMrHandle(engine, it->adaptorMr);
}

static void traceP2pAddressRange(const char *stage, FlagcxP2pEngine *engine,
                                 uintptr_t addr, size_t size, FlagcxP2pMr mrId,
                                 int netDev, int ibDevN, int ptrType,
                                 void *mhandle) {
  if (flagcxDebugLevel < FLAGCX_LOG_TRACE ||
      (flagcxDebugMask & FLAGCX_P2P) == 0) {
    return;
  }
  void *allocationBase = NULL;
  size_t allocationSize = 0;
  flagcxResult_t rangeResult = flagcxNotSupported;
  if (deviceAdaptor != NULL && deviceAdaptor->getAddressRange != NULL) {
    rangeResult = deviceAdaptor->getAddressRange(
        reinterpret_cast<const void *>(addr), &allocationBase, &allocationSize);
  }

  uintptr_t allocationOffset = 0;
  if (rangeResult == flagcxSuccess &&
      addr >= reinterpret_cast<uintptr_t>(allocationBase)) {
    allocationOffset = addr - reinterpret_cast<uintptr_t>(allocationBase);
  }

  TRACE(FLAGCX_P2P,
        "P2P address trace stage=%s engine=%p gpu=%d mr=%llu addr=%p "
        "size=%zu allocationBase=%p allocationSize=%zu "
        "allocationOffset=%zu rangeResult=%d netDev=%d ibDev=%d ptrType=%d "
        "mhandle=%p",
        stage, engine, engine != NULL ? engine->localGpuIdx : -1,
        (unsigned long long)mrId, reinterpret_cast<void *>(addr), size,
        allocationBase, allocationSize, (size_t)allocationOffset,
        (int)rangeResult, netDev, ibDevN, ptrType, mhandle);
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

  TRACE(FLAGCX_P2P,
        "P2P local transfer begin conn=%p engine=%p gpu=%d sameProcess=%d "
        "isLocal=%d isWrite=%d numIovs=%d",
        conn, conn->engine,
        conn->engine != NULL ? conn->engine->localGpuIdx : -1,
        (int)conn->sameProcess, (int)conn->isLocal, (int)isWrite, numIovs);

  std::vector<FlagcxP2pMemRegEntry> localEntries(numIovs);
  std::vector<FlagcxP2pMemRegEntry> remoteEntries(numIovs);
  std::vector<bool> haveRemoteEntry(numIovs, false);

  /* Batch local lookups (single lock acquisition in legacy mode) */
  std::vector<uintptr_t> localAddrs(numIovs);
  for (int i = 0; i < numIovs; i++)
    localAddrs[i] = (uintptr_t)localVec[i];
  if (!findMemRegBatch(localAddrs.data(), numIovs, localEntries.data()))
    return -1;

  if (conn->sameProcess) {
    for (int i = 0; i < numIovs; i++) {
      if (findMemReg((uintptr_t)descs[i].addr, &remoteEntries[i]))
        haveRemoteEntry[i] = true;
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

    const long long localOffset =
        (long long)((intptr_t)(uintptr_t)localVec[i] -
                    (intptr_t)localEntries[i].baseAddr);
    const long long remoteOffset =
        haveRemoteEntry[i] ? (long long)((intptr_t)(uintptr_t)remotePtr -
                                         (intptr_t)remoteEntries[i].baseAddr)
                           : 0;
    TRACE(FLAGCX_P2P,
          "P2P local transfer iov=%d local=%p localMr=%llu "
          "localBase=%p localSize=%zu localOffset=%lld descAddr=%p "
          "descSize=%u remote=%p remoteMr=%llu remoteBase=%p "
          "remoteSize=%zu remoteOffset=%lld src=%p dst=%p bytes=%zu "
          "srcCuda=%d dstCuda=%d",
          i, localVec[i], (unsigned long long)localEntries[i].mrId,
          reinterpret_cast<void *>(localEntries[i].baseAddr),
          localEntries[i].size, localOffset,
          reinterpret_cast<void *>((uintptr_t)descs[i].addr), descs[i].size,
          remotePtr,
          (unsigned long long)(haveRemoteEntry[i] ? remoteEntries[i].mrId : 0),
          reinterpret_cast<void *>(
              haveRemoteEntry[i] ? remoteEntries[i].baseAddr : 0),
          haveRemoteEntry[i] ? remoteEntries[i].size : 0, remoteOffset, src,
          dst, sizeVec[i], (int)srcIsCuda, (int)dstIsCuda);

    if (!srcIsCuda && !dstIsCuda) {
      memcpy(dst, src, sizeVec[i]);
      continue;
    }

    if (ensureIpcAsyncResources(&xfer) != flagcxSuccess) {
      cleanupIpcXfer(&xfer);
      return -1;
    }

    const flagcxMemcpyType_t copyType = chooseMemcpyType(srcIsCuda, dstIsCuda);
    TRACE(FLAGCX_P2P,
          "P2P local memcpy iov=%d gpu=%d src=%p dst=%p bytes=%zu "
          "copyType=%d stream=%p",
          i, conn->engine != NULL ? conn->engine->localGpuIdx : -1, src, dst,
          sizeVec[i], (int)copyType, xfer.stream);
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
  TRACE(FLAGCX_P2P,
        "P2P local transfer submitted conn=%p transferId=%llu stream=%p "
        "event=%p",
        conn, (unsigned long long)xferId, gXferMap[xferId].stream,
        gXferMap[xferId].event);
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

static flagcxP2pControl::ProtocolTransport
p2pProtocolTransport(const FlagcxP2pEngine *engine) {
  return engine != NULL && engine->isBarex ? flagcxP2pControl::kProtocolBarex
                                           : flagcxP2pControl::kProtocolIbrc;
}

static bool exchangeP2pProtocol(struct bootstrapState *bsState,
                                const FlagcxP2pEngine *engine) {
  flagcxP2pControl::ProtocolHello local = flagcxP2pControl::protocolHello(
      flagcxP2pControl::kProtocolShared, p2pProtocolTransport(engine));
  flagcxP2pControl::ProtocolHello remote = {};
  if (bootstrapExchange(bsState, 0, flagcxP2pControl::kProtocolTag, &local,
                        sizeof(local), &remote,
                        sizeof(remote)) != flagcxSuccess)
    return false;
  if (!flagcxP2pControl::protocolCompatible(local, remote)) {
    WARN("NET/P2P_ENGINE : incompatible peer protocol impl=%u transport=%u "
         "version=%u",
         unsigned(remote.implementation), unsigned(remote.transport),
         unsigned(remote.version));
    return false;
  }
  return true;
}

static bool acceptP2pProtocol(struct bootstrapState *bsState,
                              const int header[2],
                              const std::atomic<bool> &stop,
                              const FlagcxP2pEngine *engine) {
  if (header[0] != flagcxP2pControl::kProtocolTag ||
      header[1] != sizeof(flagcxP2pControl::ProtocolHello))
    return false;
  flagcxP2pControl::ProtocolHello remote = {};
  if (!flagcxP2pControl::receive(bsState->p2p->sock.fd, &remote, sizeof(remote),
                                 stop))
    return false;
  flagcxP2pControl::ProtocolHello local = flagcxP2pControl::protocolHello(
      flagcxP2pControl::kProtocolShared, p2pProtocolTransport(engine));
  if (bootstrapSend(bsState, 0, flagcxP2pControl::kProtocolTag, &local,
                    sizeof(local)) != flagcxSuccess)
    return false;
  return flagcxP2pControl::protocolCompatible(local, remote);
}

static bool makeMemRegWire(const struct flagcxP2pMrSegment &segment,
                           FlagcxP2pMemRegWire *wire) {
  if (wire == NULL || segment.keys.nKeys == 0 ||
      segment.keys.nKeys > FLAGCX_NET_MAX_MR_KEYS)
    return false;
  memset(wire, 0, sizeof(*wire));
  wire->baseAddr = segment.base;
  wire->size = segment.size;
  wire->nKeys = segment.keys.nKeys;
  memcpy(wire->rkeys, segment.keys.rkeys,
         segment.keys.nKeys * sizeof(uint32_t));
  return true;
}

static bool decodeRemoteRegion(const FlagcxP2pMemRegWire &wire,
                               FlagcxP2pRemoteRegion *region) {
  if (region == NULL || wire.nKeys == 0 ||
      wire.nKeys > FLAGCX_NET_MAX_MR_KEYS || wire.size == 0 ||
      wire.baseAddr > UINT64_MAX - wire.size)
    return false;
  memset(region, 0, sizeof(*region));
  region->baseAddr = wire.baseAddr;
  region->size = wire.size;
  region->info.nKeys = wire.nKeys;
  memcpy(region->info.rkeys, wire.rkeys, wire.nKeys * sizeof(uint32_t));
  return true;
}

static int bootstrapExchangeDescTable(struct bootstrapState *bsState,
                                      FlagcxP2pConn *conn) {
  if (bsState == NULL || conn == NULL || conn->sendComm == NULL)
    return -1;

  std::vector<FlagcxP2pMemRegWire> localTable;
  if (!flagcxParamMrSortedLookup()) {
    /* Legacy: iterate hash map */
    std::lock_guard<std::mutex> lock(gMemMutex);
    localTable.reserve(gMemRegInfo.size());
    for (auto it = gMemRegInfo.begin(); it != gMemRegInfo.end(); ++it) {
      if (it->second.record == nullptr)
        continue;
      for (const auto &segment : it->second.record->segments) {
        FlagcxP2pMemRegWire w;
        if (makeMemRegWire(segment, &w))
          localTable.push_back(w);
      }
    }
  } else {
    /* New: iterate sorted registry */
    if (flagcxMrRegistryRdLock(flagcxGlobalMrRegistry) == flagcxSuccess) {
      int count = flagcxMrRegistryCount(flagcxGlobalMrRegistry);
      if (count > 0) {
        struct flagcxMrEntry *entries =
            flagcxMrRegistryEntries(flagcxGlobalMrRegistry);
        localTable.reserve(count);
        for (int i = 0; i < count; i++) {
          if (!(entries[i].ownerMask & FLAGCX_MR_OWNER_P2P))
            continue;
          std::shared_ptr<struct flagcxP2pMrRecord> record =
              entries[i].p2p ? findMrRecord(entries[i].p2p->mrId) : nullptr;
          if (record == nullptr)
            continue;
          for (const auto &segment : record->segments) {
            FlagcxP2pMemRegWire w;
            if (makeMemRegWire(segment, &w))
              localTable.push_back(w);
          }
        }
      }
      flagcxMrRegistryRdUnlock(flagcxGlobalMrRegistry);
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
    if (!decodeRemoteRegion(remoteTable[i], &r))
      return -1;
    conn->remoteRegions.push_back(r);
  }
  std::sort(conn->remoteRegions.begin(), conn->remoteRegions.end(),
            [](const FlagcxP2pRemoteRegion &a, const FlagcxP2pRemoteRegion &b) {
              return a.baseAddr < b.baseAddr;
            });
  return 0;
}

FlagcxP2pEngine *flagcxP2pEngineCreate() {
  /* The public Engine remains transport-neutral. "accl" now selects the
     in-tree BAREX adaptor instead of constructing the retired parallel
     Engine implementation. */
  const char *transport = flagcxGetEnv("FLAGCX_P2P_TRANSPORT");
  const bool useBarex =
      transport != NULL && (strcasecmp(transport, "accl") == 0 ||
                            strcasecmp(transport, "barex") == 0);
  if (useBarex) {
#ifdef USE_ACCL_BAREX
#else
    WARN("FLAGCX_P2P_TRANSPORT=accl but FlagCX was built without "
         "USE_ACCL_BAREX=1");
    return NULL;
#endif
  }

  /* Ensure MR registry is ready (only needed for sorted-lookup mode) */
  if (flagcxParamMrSortedLookup()) {
    if (flagcxMrRegistryGlobalInit() != flagcxSuccess)
      return NULL;
  }

  FlagcxP2pEngine *engine = new FlagcxP2pEngine;
  const auto &config = flagcxP2pGlobalConfig();
  engine->runtimeSliceConfig.store(
      flagcxP2pControl::pack(config.sliceSize, config.fragmentLimit),
      std::memory_order_relaxed);
  engine->isBarex = useBarex;
#ifdef USE_ACCL_BAREX
  engine->adaptor = useBarex ? &flagcxNetBarex : &flagcxNetIb;
#else
  engine->adaptor = &flagcxNetIb;
#endif
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
    if (flagcxParamMrSortedLookup()) {
      flagcxMrRegistryGlobalRelease();
    }
    delete engine;
    return NULL;
  }

  // Initialize bootstrap network context (discovers local NIC)
  bootstrapNetInit();

  engine->adaptor->devices(&engine->nDevs);
  if (engine->nDevs < 0 || engine->nDevs > MAX_IB_VDEVS) {
    if (flagcxParamMrSortedLookup())
      flagcxMrRegistryGlobalRelease();
    delete engine;
    return NULL;
  }
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

  union flagcxSocketAddress notifAddr =
      engine->isBarex ? *bootstrapGetNetIfAddr() : flagcxIbIfAddr;
  socketAddrSetPort(&notifAddr, 0);
  flagcxResult_t notifRes =
      flagcxSocketInit(&engine->notifListenSock, &notifAddr,
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
    INFO(FLAGCX_INIT, "NET/%s_P2P : bootstrap P2P listen on port %d",
         engine->adaptor->name, engine->bsListenPort);
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
  if (engine->notifThread.joinable() &&
      engine->notifThread.get_id() != std::this_thread::get_id())
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
    if (!flagcxParamMrSortedLookup()) {
      /* Legacy: deregister all from hash maps */
      std::lock_guard<std::mutex> lock(gMemMutex);
      for (auto it = gMemRegInfo.begin(); it != gMemRegInfo.end(); ++it)
        deregisterEngineMr(engine, it->second.record);
      gMemRegInfo.clear();
      gMrToBaseAddr.clear();
    } else {
      /* New: deregister from unified registry */
      pthread_mutex_lock(&gMrLifecycleMutex);

      /* Phase 1: collect P2P mhandle info under read lock */
      struct P2pDeregInfo {
        FlagcxP2pMr mrId;
        uintptr_t baseAddr;
      };
      std::vector<P2pDeregInfo> deregList;

      if (flagcxMrRegistryRdLock(flagcxGlobalMrRegistry) == flagcxSuccess) {
        int count = flagcxMrRegistryCount(flagcxGlobalMrRegistry);
        if (count > 0) {
          struct flagcxMrEntry *entries =
              flagcxMrRegistryEntries(flagcxGlobalMrRegistry);
          for (int i = 0; i < count; i++) {
            if (!(entries[i].ownerMask & FLAGCX_MR_OWNER_P2P))
              continue;
            P2pDeregInfo info;
            info.mrId = entries[i].p2p ? entries[i].p2p->mrId : 0;
            info.baseAddr = entries[i].baseAddr;
            deregList.push_back(info);
          }
        }
        flagcxMrRegistryRdUnlock(flagcxGlobalMrRegistry);
      }

      /* Phase 2: deregister from registry */
      for (P2pDeregInfo &info : deregList) {
        if (flagcxMrRegistryDeregister(flagcxGlobalMrRegistry, info.baseAddr,
                                       FLAGCX_MR_OWNER_P2P, NULL,
                                       NULL) != flagcxSuccess) {
          info.mrId = 0;
        }
      }

      /* Phase 3: call adaptor deregMr */
      for (const P2pDeregInfo &info : deregList) {
        if (info.mrId == 0)
          continue;
        deregisterEngineMr(engine, findMrRecord(info.mrId));
        eraseMrRecord(info.mrId);
      }

      pthread_mutex_unlock(&gMrLifecycleMutex);
    }
  }

  /* Release P2P engine's refcount on the global MR registry */
  if (flagcxParamMrSortedLookup()) {
    flagcxMrRegistryGlobalRelease();
  }

  if (engine->topoMgr) {
    flagcxP2pTopoDestroy(engine->topoMgr);
  }

  if (engine->isBarex) {
#ifdef USE_ACCL_BAREX
    (void)flagcxBarexRuntimeDrainDeferredMrs();
#endif
  } else {
    (void)flagcxIbEngineDrainDeferredMrs();
  }

  {
    std::lock_guard<std::mutex> lock(gMrRecordMutex);
    gMrRecords.clear();
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
      if (engine->isBarex) {
        engine->adaptor->closeListen(engine->listeners[d].listenComm);
        engine->listeners[d].listenComm = NULL;
      } else {
        flagcxIbEngineAbortListen(engine->listeners[d].listenComm);
      }
    }
  }

  if (engine->rpcServerThread.joinable() &&
      engine->rpcServerThread.get_id() != std::this_thread::get_id()) {
    engine->rpcServerThread.join();
    engine->rpcServerActive.store(false, std::memory_order_release);
  }
}

static flagcxResult_t
establishDuplexConnection(FlagcxP2pEngine *engine, int netDev, void *listenComm,
                          void *remoteHandle, void **sendComm, void **recvComm,
                          bool stopWithAccept) {
  if (engine == NULL || listenComm == NULL || remoteHandle == NULL ||
      sendComm == NULL || recvComm == NULL || netDev < 0 ||
      netDev >= engine->nDevs || engine->adaptor == NULL)
    return flagcxInvalidArgument;
  *sendComm = NULL;
  *recvComm = NULL;
  flagcxResult_t result = flagcxSuccess;
  while (*sendComm == NULL || *recvComm == NULL) {
    if (stopWithAccept && engine->stopAccept.load(std::memory_order_acquire)) {
      result = flagcxSystemError;
      break;
    }
    if (*sendComm == NULL) {
      result = engine->adaptor->connect(netDev, remoteHandle, sendComm);
      if (result != flagcxSuccess)
        break;
    }
    if (*recvComm == NULL) {
      result = engine->adaptor->accept(listenComm, recvComm);
      if (result != flagcxSuccess)
        break;
    }
    if (*sendComm == NULL || *recvComm == NULL)
      std::this_thread::yield();
  }
  if (result == flagcxSuccess && engine->isBarex) {
#ifdef USE_ACCL_BAREX
    uint32_t sendChannels = 0;
    uint32_t recvChannels = 0;
    if (flagcxBarexRuntimeGetCommChannels(*sendComm, &sendChannels) !=
            flagcxSuccess ||
        flagcxBarexRuntimeGetCommChannels(*recvComm, &recvChannels) !=
            flagcxSuccess ||
        sendChannels != recvChannels ||
        sendChannels !=
            static_cast<uint32_t>(flagcxP2pGlobalConfig().qpsPerConn)) {
      result = flagcxInternalError;
    }
#endif
  }
  if (result == flagcxSuccess)
    return flagcxSuccess;

  // The main IB connect/accept entry points keep asynchronous progress in the
  // caller-owned handle and listener. A failed handshake must discard both
  // incomplete stages before another bootstrap exchange may advertise this
  // listener. Completed comms are returned to the caller for normal close.
  flagcxResult_t cleanupResult = flagcxSuccess;
  if (*sendComm == NULL) {
    if (engine->isBarex) {
#ifdef USE_ACCL_BAREX
      cleanupResult = flagcxBarexRuntimeResetConnect(remoteHandle);
#endif
    } else {
      cleanupResult = flagcxIbEngineResetConnect(remoteHandle);
    }
  }
  if (!engine->isBarex && *recvComm == NULL) {
    flagcxResult_t recvCleanup = flagcxIbEngineResetListenAccept(listenComm);
    if (cleanupResult == flagcxSuccess)
      cleanupResult = recvCleanup;
  }
  return cleanupResult == flagcxSuccess ? result : cleanupResult;
}

class FlagcxP2pConnectionConfigScope {
public:
  explicit FlagcxP2pConnectionConfigScope(FlagcxP2pEngine *engine)
      : engine_(engine), result_(flagcxInvalidArgument), active_(false) {
    if (engine_ == NULL)
      return;
    const FlagcxP2pGlobalConfig &config = flagcxP2pGlobalConfig();
    if (engine_->isBarex) {
#ifdef USE_ACCL_BAREX
      const struct flagcxBarexRuntimeConnectionConfig barexConfig = {
          static_cast<uint32_t>(config.qpsPerConn)};
      result_ = flagcxBarexRuntimeSetConnectionConfig(&barexConfig);
#else
      result_ = flagcxNotSupported;
#endif
    } else {
      const struct flagcxIbEngineConnectionConfig ibConfig = {
          getenv("FLAGCX_IB_QPS_PER_CONNECTION") == NULL
              ? config.qpsPerConn
              : FLAGCX_IB_ENGINE_CONFIG_INHERIT,
          getenv("FLAGCX_IB_GID_INDEX") == NULL
              ? config.gidIndex
              : FLAGCX_IB_ENGINE_CONFIG_INHERIT,
          config.mtuLength,
          getenv("FLAGCX_IB_TC") == NULL ? config.ibTrafficClass
                                         : FLAGCX_IB_ENGINE_CONFIG_INHERIT,
          getenv("FLAGCX_IB_RETRY_CNT") == NULL
              ? config.retryCnt
              : FLAGCX_IB_ENGINE_CONFIG_INHERIT};
      result_ = flagcxIbEngineSetConnectionConfig(&ibConfig);
    }
    active_ = result_ == flagcxSuccess;
  }

  ~FlagcxP2pConnectionConfigScope() {
    if (!active_)
      return;
    if (engine_->isBarex) {
#ifdef USE_ACCL_BAREX
      flagcxBarexRuntimeClearConnectionConfig();
#endif
    } else {
      flagcxIbEngineClearConnectionConfig();
    }
  }

  flagcxResult_t result() const { return result_; }

private:
  FlagcxP2pEngine *engine_;
  flagcxResult_t result_;
  bool active_;
};

static void closeDuplexConnection(FlagcxP2pEngine *engine, void *sendComm,
                                  void *recvComm) {
  if (engine == NULL)
    return;
  if (sendComm != NULL)
    engine->adaptor->closeSend(sendComm);
  if (recvComm != NULL)
    engine->adaptor->closeRecv(recvComm);
}

FlagcxP2pConn *flagcxP2pEngineConnect(FlagcxP2pEngine *engine,
                                      const char *ipAddr, int remoteGpuIdx,
                                      int remotePort, bool sameProcess) {
  if (engine == NULL || ipAddr == NULL)
    return NULL;

  const int netDev = chooseEngineNetDev(engine);
  if (netDev < 0)
    return NULL;

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

  if (!exchangeP2pProtocol(bsConn, engine)) {
    bootstrapClose(bsConn);
    return NULL;
  }

  FlagcxP2pConnectionConfigScope connectionConfig(engine);
  if (connectionConfig.result() != flagcxSuccess) {
    bootstrapClose(bsConn);
    return NULL;
  }

  // Step 2: Exchange per-handshake IB listen handles over bootstrap. Main IB
  // keeps asynchronous accept state in the listener object, so a distinct
  // listener prevents concurrent bootstrap sessions from consuming each
  // other's inbound connection.
  void *listenComm = NULL;
  char localIbHandle[FLAGCX_NET_HANDLE_MAXSIZE];
  memset(localIbHandle, 0, sizeof(localIbHandle));
  if (engine->adaptor->listen(netDev, localIbHandle, &listenComm) !=
      flagcxSuccess) {
    bootstrapClose(bsConn);
    return NULL;
  }

  char remoteIbHandle[FLAGCX_NET_HANDLE_MAXSIZE];
  memset(remoteIbHandle, 0, sizeof(remoteIbHandle));
  if (bootstrapExchange(bsConn, 0, 4, localIbHandle, FLAGCX_NET_HANDLE_MAXSIZE,
                        remoteIbHandle,
                        FLAGCX_NET_HANDLE_MAXSIZE) != flagcxSuccess) {
    engine->adaptor->closeListen(listenComm);
    bootstrapClose(bsConn);
    return NULL;
  }

  // Step 3: establish a true main-adaptor send/recv pair. Both peers initiate
  // one outbound connection and accept one inbound connection; unlike the
  // retired P2P adaptor, main IB send and recv objects are not interchangeable.
  void *sendComm = NULL;
  void *recvComm = NULL;
  flagcxResult_t connectResult = establishDuplexConnection(
      engine, netDev, listenComm, remoteIbHandle, &sendComm, &recvComm, false);
  engine->adaptor->closeListen(listenComm);
  if (connectResult != flagcxSuccess) {
    closeDuplexConnection(engine, sendComm, recvComm);
    bootstrapClose(bsConn);
    return NULL;
  }

  FlagcxP2pListenHandleView *remoteHandle =
      reinterpret_cast<FlagcxP2pListenHandleView *>(remoteIbHandle);
  const union flagcxSocketAddress *peerAddress =
      engine->isBarex ? &bsHandle.addr : &remoteHandle->connectAddr;
  const union flagcxSocketAddress *localAddress =
      engine->isBarex ? bootstrapGetNetIfAddr() : &flagcxIbIfAddr;
  const bool sameHost = socketAddrSameHost(peerAddress, localAddress);
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
    closeDuplexConnection(engine, sendComm, recvComm);
    bootstrapClose(bsConn);
    return NULL;
  }

  FlagcxP2pConn *conn = new FlagcxP2pConn;
  conn->engine = engine;
  conn->sendComm = sendComm;
  conn->recvComm = recvComm;
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
    connectNotifSocket(conn, peerAddress, remoteMeta.notifPort);
  }

  // Step 5: Exchange desc table over bootstrap
  if (bootstrapExchangeDescTable(bsConn, conn) != 0) {
    WARN("NET/P2P_ENGINE : connect desc-table exchange failed");
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

  FlagcxP2pConnectionConfigScope connectionConfig(engine);
  if (connectionConfig.result() != flagcxSuccess) {
    bootstrapClose(bsConn);
    return NULL;
  }

  // The first bootstrap frame distinguishes legacy connect (tag 4) from
  // runtime control. Do not establish any QP/MR state for a control request.
  int header[2] = {};
  const int fd = bsConn->p2p->sock.fd;
  if (!flagcxP2pControl::receive(fd, header, sizeof(header),
                                 engine->stopAccept)) {
    bootstrapClose(bsConn);
    return NULL;
  }
  if (header[0] == flagcxP2pControl::kTag) {
    if (header[1] <= 0 || header[1] > flagcxP2pControl::kMaxRequest) {
      bootstrapClose(bsConn);
      return NULL;
    }
    std::string request(header[1], '\0');
    if (flagcxP2pControl::receive(fd, &request[0], request.size(),
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
                   error ? -1 : 0, unsigned(config >> 32),
                   unsigned(uint32_t(config)), error ? error : "");
      // Failure to deliver the ACK does not undo an applied update. GET lets
      // the client resolve an uncertain result without retransmitting a SET.
      bootstrapSend(bsConn, 0, flagcxP2pControl::kTag, reply, length);
      if (!error && request != "GET")
        INFO(FLAGCX_INIT,
             "P2P runtime config engine=%p sliceSize=%u fragmentLimit=%u",
             engine, unsigned(config >> 32), unsigned(uint32_t(config)));
    }
    bootstrapClose(bsConn);
    return NULL; // RPC accept loop continues; no data connection was created.
  }

  if (!acceptP2pProtocol(bsConn, header, engine->stopAccept, engine)) {
    WARN("NET/P2P_ENGINE : peer protocol is incompatible with shared Engine");
    bootstrapClose(bsConn);
    return NULL;
  }

  // Protocol is compatible; exchange the transport-specific listen handles.
  // Use a listener dedicated to this bootstrap connection so concurrent
  // sessions cannot consume each other's inbound main-IB connection.
  void *listenComm = NULL;
  char localIbHandle[FLAGCX_NET_HANDLE_MAXSIZE];
  memset(localIbHandle, 0, sizeof(localIbHandle));
  if (engine->adaptor->listen(dev, localIbHandle, &listenComm) !=
      flagcxSuccess) {
    bootstrapClose(bsConn);
    return NULL;
  }
  char remoteIbHandle[FLAGCX_NET_HANDLE_MAXSIZE];
  if (bootstrapRecv(bsConn, 0, 4, remoteIbHandle, sizeof(remoteIbHandle)) !=
          flagcxSuccess ||
      bootstrapSend(bsConn, 0, 4, localIbHandle, sizeof(localIbHandle)) !=
          flagcxSuccess) {
    engine->adaptor->closeListen(listenComm);
    bootstrapClose(bsConn);
    return NULL;
  }

  // Step 3: establish the same duplex main-adaptor pair as the connector.
  // Progressing connect and accept together avoids relying on endpoint role.
  void *sendComm = NULL;
  void *recvComm = NULL;
  if (engine->stopAccept.load(std::memory_order_acquire)) {
    engine->adaptor->closeListen(listenComm);
    bootstrapClose(bsConn);
    return NULL;
  }
  flagcxResult_t connectResult = establishDuplexConnection(
      engine, dev, listenComm, remoteIbHandle, &sendComm, &recvComm, true);
  engine->adaptor->closeListen(listenComm);
  if (connectResult != flagcxSuccess) {
    closeDuplexConnection(engine, sendComm, recvComm);
    bootstrapClose(bsConn);
    return NULL;
  }

  // Step 4: Exchange ctrl meta over bootstrap
  FlagcxP2pCtrlMeta localMeta;
  memset(&localMeta, 0, sizeof(localMeta));
  localMeta.gpuIdx = engine->localGpuIdx;
  localMeta.notifPort = engine->notifListenPort;
  union flagcxSocketAddress peerAddress;
  memset(&peerAddress, 0, sizeof(peerAddress));
  flagcxResult_t peerAddressResult =
      engine->isBarex ? flagcxSocketGetAddr(&bsConn->p2p->sock, &peerAddress)
                      : flagcxIbEngineGetCommAddress(recvComm, &peerAddress);
  if (peerAddressResult != flagcxSuccess) {
    closeDuplexConnection(engine, sendComm, recvComm);
    bootstrapClose(bsConn);
    return NULL;
  }
  const union flagcxSocketAddress *localAddress =
      engine->isBarex ? bootstrapGetNetIfAddr() : &flagcxIbIfAddr;
  if (socketAddrSameHost(&peerAddress, localAddress)) {
    localMeta.flags |= FLAGCX_P2P_CTRL_FLAG_LOCAL;
  }

  FlagcxP2pCtrlMeta remoteMeta;
  memset(&remoteMeta, 0, sizeof(remoteMeta));
  if (bootstrapExchangeCtrlMeta(bsConn, &localMeta, &remoteMeta) !=
      flagcxSuccess) {
    closeDuplexConnection(engine, sendComm, recvComm);
    bootstrapClose(bsConn);
    return NULL;
  }

  FlagcxP2pConn *conn = new FlagcxP2pConn;
  conn->engine = engine;
  conn->sendComm = sendComm;
  conn->recvComm = recvComm;
  conn->netDev = dev;
  conn->remoteGpuIdx = remoteMeta.gpuIdx;
  conn->remoteNotifPort = remoteMeta.notifPort;
  conn->isLocal = (remoteMeta.flags & FLAGCX_P2P_CTRL_FLAG_LOCAL) != 0;
  conn->sameProcess =
      (remoteMeta.flags & FLAGCX_P2P_CTRL_FLAG_SAME_PROCESS) != 0;
  conn->notifSockConnected = false;
  memset(&conn->notifSock, 0, sizeof(conn->notifSock));

  copyStringToBuf(socketAddrToHostString(&peerAddress), ipAddrBuf,
                  ipAddrBufLen);
  *remoteGpuIdx = remoteMeta.gpuIdx;

  if (!conn->sameProcess && remoteMeta.notifPort > 0) {
    connectNotifSocket(conn, &peerAddress, remoteMeta.notifPort);
  }

  // Step 5: Exchange desc table over bootstrap
  if (bootstrapExchangeDescTable(bsConn, conn) != 0) {
    WARN("NET/P2P_ENGINE : accept desc-table exchange failed");
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

int flagcxP2pEngineRegEx(FlagcxP2pEngine *engine, uintptr_t data, size_t size,
                         int hintType, FlagcxP2pMr &mrId) {
  if (engine == NULL || data == 0)
    return -1;

  auto resolvePtrType = [&](int *ptrType, char *ipcHandleBuf,
                            uint32_t *ipcHandleSize) -> flagcxResult_t {
    if (hintType == FLAGCX_PTR_HOST || hintType == FLAGCX_PTR_CUDA) {
      if (ipcHandleSize)
        *ipcHandleSize = 0;
      *ptrType = hintType;
      return flagcxSuccess;
    }
    return flagcxP2pDetectPointerType(reinterpret_cast<void *>(data), ptrType,
                                      ipcHandleBuf, ipcHandleSize);
  };

  if (!flagcxParamMrSortedLookup()) {
    /* Legacy: mutex + hash maps */
    std::lock_guard<std::mutex> lock(gMemMutex);
    auto existing = gMemRegInfo.find(data);
    if (existing != gMemRegInfo.end()) {
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
    const int ibDevN = engine->isBarex ? -1 : resolveIbDevN(netDev);
    if (netDev < 0 || (!engine->isBarex && ibDevN < 0))
      return -1;
    FlagcxP2pMemRegEntry entry = {};
    entry.mrId = gNextMrId++;
    entry.baseAddr = data;
    entry.size = size;
    entry.ibDevN = ibDevN;

    setEngineDevice(engine);
    const flagcxResult_t ptrTypeResult =
        resolvePtrType(&entry.ptrType, entry.ipcHandle, &entry.ipcHandleSize);
    if (ptrTypeResult != flagcxSuccess) {
      WARN("P2P Reg: failed to classify addr 0x%lx: %d", (unsigned long)data,
           (int)ptrTypeResult);
      return -1;
    }
    entry.hasIpc = entry.ptrType == FLAGCX_PTR_CUDA && entry.ipcHandleSize > 0;

    if (registerEngineMr(engine, netDev, data, size, entry.ptrType,
                         &entry.record) != flagcxSuccess ||
        entry.record == nullptr || entry.record->segments.empty()) {
      return -1;
    }
    entry.record->id = entry.mrId;
    entry.mhandle = entry.record->segments[0].adaptorMr;

    traceP2pAddressRange("register-legacy", engine, data, size, entry.mrId,
                         netDev, ibDevN, entry.ptrType, entry.mhandle);

    gMemRegInfo[data] = entry;
    gMrToBaseAddr[entry.mrId] = data;
    storeMrRecord(entry.mrId, entry.record);
    mrId = entry.mrId;
    return 0;
  }

  /* New: gMrLifecycleMutex + unified registry */
  pthread_mutex_lock(&gMrLifecycleMutex);

  /* Check for existing exact-match registration (dedup) */
  {
    struct flagcxMrEntry existing;
    struct flagcxMrExtension p2pExt;
    struct flagcxMrExtension *exts[FLAGCX_MR_OWNER_COUNT] = {&p2pExt, NULL,
                                                             NULL};
    if (flagcxMrRegistryFindExact(flagcxGlobalMrRegistry, data, &existing,
                                  exts) == flagcxSuccess) {
      if (existing.size != size) {
        WARN("P2P Reg: addr 0x%lx size mismatch: existing %zu vs requested "
             "%zu",
             (unsigned long)data, existing.size, size);
        pthread_mutex_unlock(&gMrLifecycleMutex);
        return -1;
      }
      if (p2pExt.type == FLAGCX_MR_OWNER_P2P) {
        mrId = p2pExt.p2p.mrId;
        pthread_mutex_unlock(&gMrLifecycleMutex);
        return 0;
      }
    }
  }

  const int netDev = chooseEngineNetDev(engine);
  const int ibDevN = engine->isBarex ? -1 : resolveIbDevN(netDev);
  if (netDev < 0 || (!engine->isBarex && ibDevN < 0)) {
    pthread_mutex_unlock(&gMrLifecycleMutex);
    return -1;
  }
  /* Detect pointer type and IPC handle */
  char ipcHandle[FLAGCX_P2P_IPC_HANDLE_BYTES];
  uint32_t ipcHandleSize = 0;
  memset(ipcHandle, 0, sizeof(ipcHandle));

  setEngineDevice(engine);
  int ptrType = FLAGCX_PTR_HOST;
  const flagcxResult_t ptrTypeResult =
      resolvePtrType(&ptrType, ipcHandle, &ipcHandleSize);
  if (ptrTypeResult != flagcxSuccess) {
    WARN("P2P Reg: failed to classify addr 0x%lx: %d", (unsigned long)data,
         (int)ptrTypeResult);
    pthread_mutex_unlock(&gMrLifecycleMutex);
    return -1;
  }
  bool hasIpc = ptrType == FLAGCX_PTR_CUDA && ipcHandleSize > 0;

  /* Register with adaptor */
  std::shared_ptr<struct flagcxP2pMrRecord> record;
  if (registerEngineMr(engine, netDev, data, size, ptrType, &record) !=
          flagcxSuccess ||
      record == nullptr || record->segments.empty()) {
    pthread_mutex_unlock(&gMrLifecycleMutex);
    return -1;
  }
  void *mhandle = record->segments[0].adaptorMr;

  /* Build P2P extension */
  struct flagcxMrP2pExt *p2pExt =
      (struct flagcxMrP2pExt *)calloc(1, sizeof(struct flagcxMrP2pExt));
  if (p2pExt == NULL) {
    deregisterEngineMr(engine, record);
    pthread_mutex_unlock(&gMrLifecycleMutex);
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
      flagcxGlobalMrRegistry, data, size, ibDevN, ptrType, FLAGCX_MR_OWNER_P2P,
      mhandle, p2pExt, &assignedId);
  if (res != flagcxSuccess) {
    deregisterEngineMr(engine, record);
    free(p2pExt);
    pthread_mutex_unlock(&gMrLifecycleMutex);
    return -1;
  }

  mrId = assignedId;
  record->id = assignedId;
  storeMrRecord(mrId, record);
  traceP2pAddressRange("register", engine, data, size, mrId, netDev, ibDevN,
                       ptrType, mhandle);
  pthread_mutex_unlock(&gMrLifecycleMutex);
  return 0;
}

int flagcxP2pEngineReg(FlagcxP2pEngine *engine, uintptr_t data, size_t size,
                       FlagcxP2pMr &mrId) {
  return flagcxP2pEngineRegEx(engine, data, size, 0, mrId);
}

void flagcxP2pEngineMrDestroy(FlagcxP2pEngine *engine, FlagcxP2pMr mr) {
  if (engine == NULL)
    return;

  if (!flagcxParamMrSortedLookup()) {
    /* Legacy: mutex + hash maps */
    std::lock_guard<std::mutex> lock(gMemMutex);
    auto mrIt = gMrToBaseAddr.find(mr);
    if (mrIt == gMrToBaseAddr.end())
      return;
    auto entryIt = gMemRegInfo.find(mrIt->second);
    if (entryIt == gMemRegInfo.end()) {
      gMrToBaseAddr.erase(mrIt);
      return;
    }
    deregisterEngineMr(engine, entryIt->second.record);
    gMemRegInfo.erase(entryIt);
    gMrToBaseAddr.erase(mrIt);
    eraseMrRecord(mr);
    return;
  }

  /* New: gMrLifecycleMutex + unified registry */
  pthread_mutex_lock(&gMrLifecycleMutex);

  /* Find entry by mrId to get baseAddr for deregister */
  struct flagcxMrEntry mrEntry;
  if (flagcxMrRegistryLookupById(flagcxGlobalMrRegistry, mr, &mrEntry, NULL) !=
      flagcxSuccess) {
    pthread_mutex_unlock(&gMrLifecycleMutex);
    return;
  }

  /* Remove from registry first — prevents concurrent readers from finding it */
  void *removedExt = NULL;
  flagcxResult_t res;
  FLAGCXCHECKGOTO(
      flagcxMrRegistryDeregister(flagcxGlobalMrRegistry, mrEntry.baseAddr,
                                 FLAGCX_MR_OWNER_P2P, NULL, &removedExt),
      res, fail);
  free(removedExt);

  /* Now safe to deregister every provider segment. */
  deregisterEngineMr(engine, findMrRecord(mr));
  eraseMrRecord(mr);
  pthread_mutex_unlock(&gMrLifecycleMutex);
  return;

fail:
  pthread_mutex_unlock(&gMrLifecycleMutex);
}

int flagcxP2pEnginePrepareDesc(FlagcxP2pEngine *engine, FlagcxP2pMr mr,
                               const void *data, size_t size, char *descBuf) {
  if (engine == NULL || data == NULL || descBuf == NULL)
    return -1;

  if (!flagcxParamMrSortedLookup()) {
    /* Legacy: mutex + hash lookup */
    std::lock_guard<std::mutex> lock(gMemMutex);
    auto mrIt = gMrToBaseAddr.find(mr);
    if (mrIt == gMrToBaseAddr.end())
      return -1;
    auto entryIt = gMemRegInfo.find(mrIt->second);
    if (entryIt == gMemRegInfo.end())
      return -1;
    FlagcxP2pMemRegEntry *entry = &entryIt->second;
    if (!memRegContains(*entry, reinterpret_cast<uintptr_t>(data), size) ||
        size > UINT32_MAX || entry->record == nullptr)
      return -1;
    const uintptr_t dataAddr = reinterpret_cast<uintptr_t>(data);
    const flagcxP2pMrSegment *segment =
        findMrSegmentForRange(*entry->record, dataAddr, size);
    if (segment == nullptr)
      return -1;
    FlagcxP2pRdmaDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.addr = (uint64_t)(uintptr_t)data;
    desc.size = (uint32_t)size;
    if (flagcxP2pDescSetKeys(&desc, segment->keys.rkeys, segment->keys.nKeys) !=
        flagcxSuccess)
      return -1;
    TRACE(FLAGCX_P2P,
          "P2P descriptor trace path=legacy engine=%p gpu=%d mr=%llu "
          "registryBase=%p registrySize=%zu data=%p dataOffset=%zu size=%zu "
          "descAddr=%p descSize=%u rkey=0x%x nkeys=%u",
          engine, engine->localGpuIdx, (unsigned long long)mr,
          reinterpret_cast<void *>(entry->baseAddr), entry->size, data,
          (size_t)((uintptr_t)data - entry->baseAddr), size,
          reinterpret_cast<void *>((uintptr_t)desc.addr), desc.size, desc.rkey,
          desc.nmsgs);
    flagcxP2pSerializeRdmaDesc(desc, descBuf);
    memcpy(entry->descBuf, descBuf, FLAGCX_P2P_DESC_SIZE);
    return 0;
  }

  /* New: single read-lock + containment search */
  uintptr_t dataAddr = (uintptr_t)data;

  if (flagcxMrRegistryRdLock(flagcxGlobalMrRegistry) != flagcxSuccess)
    return -1;

  int count = flagcxMrRegistryCount(flagcxGlobalMrRegistry);
  if (count == 0) {
    flagcxMrRegistryRdUnlock(flagcxGlobalMrRegistry);
    return -1;
  }
  struct flagcxMrEntry *entries =
      flagcxMrRegistryEntries(flagcxGlobalMrRegistry);

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
    flagcxMrRegistryRdUnlock(flagcxGlobalMrRegistry);
    return -1;
  }

  std::shared_ptr<struct flagcxP2pMrRecord> record =
      findMrRecord(entries[idx].p2p->mrId);
  if (record == nullptr) {
    flagcxMrRegistryRdUnlock(flagcxGlobalMrRegistry);
    return -1;
  }

  /* Verify (data, size) fits within the MR region (overflow-safe) */
  size_t offset = (size_t)(dataAddr - entries[idx].baseAddr);
  if (size > entries[idx].size - offset) {
    flagcxMrRegistryRdUnlock(flagcxGlobalMrRegistry);
    return -1;
  }

  if (size > UINT32_MAX) {
    flagcxMrRegistryRdUnlock(flagcxGlobalMrRegistry);
    return -1;
  }

  FlagcxP2pRdmaDesc desc;
  memset(&desc, 0, sizeof(desc));
  desc.addr = (uint64_t)dataAddr;
  desc.size = (uint32_t)size;
  const flagcxP2pMrSegment *segment =
      findMrSegmentForRange(*record, dataAddr, size);
  if (segment == nullptr ||
      flagcxP2pDescSetKeys(&desc, segment->keys.rkeys, segment->keys.nKeys) !=
          flagcxSuccess) {
    flagcxMrRegistryRdUnlock(flagcxGlobalMrRegistry);
    return -1;
  }

  TRACE(FLAGCX_P2P,
        "P2P descriptor trace path=registry engine=%p gpu=%d mr=%llu "
        "registryBase=%p registrySize=%zu data=%p dataOffset=%zu size=%zu "
        "descAddr=%p descSize=%u rkey=0x%x nkeys=%u",
        engine, engine->localGpuIdx, (unsigned long long)mr,
        reinterpret_cast<void *>(entries[idx].baseAddr), entries[idx].size,
        data, offset, size, reinterpret_cast<void *>((uintptr_t)desc.addr),
        desc.size, desc.rkey, desc.nmsgs);

  flagcxP2pSerializeRdmaDesc(desc, descBuf);
  flagcxMrRegistryRdUnlock(flagcxGlobalMrRegistry);
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
  if (conn == NULL || data == NULL || transferId == NULL)
    return -1;
  *transferId = 0;
  if (!remoteDescContains(desc, size)) {
    WARN("P2P read remote descriptor bounds check failed: addr=%p "
         "descSize=%u requestSize=%zu",
         reinterpret_cast<void *>((uintptr_t)desc.addr), desc.size, size);
    return -1;
  }
  TRACE(FLAGCX_P2P,
        "P2P read trace conn=%p engine=%p gpu=%d mr=%llu local=%p size=%zu "
        "descAddr=%p descSize=%u rkey=0x%x sameProcess=%d isLocal=%d path=%s",
        conn, conn->engine,
        conn->engine != NULL ? conn->engine->localGpuIdx : -1,
        (unsigned long long)mr, data, size,
        reinterpret_cast<void *>((uintptr_t)desc.addr), desc.size, desc.rkey,
        (int)conn->sameProcess, (int)conn->isLocal,
        conn->sameProcess && conn->isLocal ? "same-process-d2d" : "rdma");

  if (conn->sameProcess && conn->isLocal) {
    std::vector<void *> localVec(1, const_cast<void *>(data));
    std::vector<size_t> sizeVec(1, size);
    std::vector<FlagcxP2pRdmaDesc> descs(1, desc);
    std::vector<char *> ipcBufs;
    return startLocalTransfer(conn, localVec, sizeVec, descs, 1, transferId,
                              ipcBufs, false);
  }

  FlagcxP2pMemRegEntry localEntry;
  if (!findMemRegByMr(mr, &localEntry) ||
      !memRegContains(localEntry, reinterpret_cast<uintptr_t>(data), size)) {
    WARN("P2P read local MR bounds check failed: mr=%llu addr=%p size=%zu",
         (unsigned long long)mr, data, size);
    return -1;
  }

  const uint32_t requirements = flagcxResolveGdrFlushRequirements(
      deviceAdaptor == NULL ? FLAGCX_GDR_FLUSH_NONE
                            : deviceAdaptor->gdrFlushRequirements);
  const flagcxResult_t visibilityResult = flagcxP2pValidateReadVisibility(
      requirements, conn->engine->adaptor->gdrFlushCaps, localEntry.ptrType,
      size, 0);
  if (visibilityResult != flagcxSuccess) {
    WARN("P2P read rejected: GPU destination requires a post-READ visibility "
         "flush, but the shared Engine completion path has none");
    return -1;
  }

  std::vector<void *> localVec(1, const_cast<void *>(data));
  std::vector<size_t> sizes(1, size);
  std::vector<FlagcxP2pRdmaDesc> remoteDescs(1, desc);
  std::vector<FlagcxP2pMemRegEntry> localEntries(1, localEntry);
  return startNetTransfer(conn, localVec, sizes, remoteDescs, localEntries, 1,
                          false, transferId);
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
  *transferId = 0;

  if (dstVec.size() < static_cast<size_t>(numIovs) ||
      sizeVec.size() < static_cast<size_t>(numIovs) ||
      descs.size() < static_cast<size_t>(numIovs)) {
    fprintf(stderr,
            "[FlagCX P2P] ReadVector early exit: vector length mismatch "
            "(numIovs=%d)\n",
            numIovs);
    return -1;
  }
  for (int i = 0; i < numIovs; i++) {
    if (!remoteDescContains(descs[i], sizeVec[i])) {
      WARN("P2P ReadVector remote descriptor bounds check failed: iov=%d "
           "addr=%p descSize=%u requestSize=%zu",
           i, reinterpret_cast<void *>((uintptr_t)descs[i].addr), descs[i].size,
           sizeVec[i]);
      return -1;
    }
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
  for (int i = 0; i < numIovs; i++) {
    if (!findMemRegByMr(mrIds[i], &localEntries[i])) {
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

  const uint32_t requirements = flagcxResolveGdrFlushRequirements(
      deviceAdaptor == NULL ? FLAGCX_GDR_FLUSH_NONE
                            : deviceAdaptor->gdrFlushRequirements);
  for (int i = 0; i < numIovs; i++) {
    const flagcxResult_t visibilityResult = flagcxP2pValidateReadVisibility(
        requirements, conn->engine->adaptor->gdrFlushCaps,
        localEntries[i].ptrType, sizeVec[i], 0);
    if (visibilityResult != flagcxSuccess) {
      WARN("P2P ReadVector rejected: GPU destination iov=%d requires a "
           "post-READ visibility flush, but the shared Engine completion "
           "path has none",
           i);
      return -1;
    }
  }

  return startNetTransfer(conn, dstVec, sizeVec, descs, localEntries, numIovs,
                          false, transferId);
}

int flagcxP2pEngineWrite(FlagcxP2pConn *conn, FlagcxP2pMr mr, const void *data,
                         size_t size, FlagcxP2pRdmaDesc desc,
                         uint64_t *transferId) {
  if (conn == NULL || data == NULL || transferId == NULL)
    return -1;
  *transferId = 0;
  if (!remoteDescContains(desc, size))
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
  if (!findMemRegByMr(mr, &localEntry) ||
      !memRegContains(localEntry, reinterpret_cast<uintptr_t>(data), size))
    return -1;

  std::vector<void *> localVec(1, const_cast<void *>(data));
  std::vector<size_t> sizes(1, size);
  std::vector<FlagcxP2pRdmaDesc> remoteDescs(1, desc);
  std::vector<FlagcxP2pMemRegEntry> localEntries(1, localEntry);
  return startNetTransfer(conn, localVec, sizes, remoteDescs, localEntries, 1,
                          true, transferId);
}

int flagcxP2pEngineWriteVector(FlagcxP2pConn *conn,
                               const std::vector<FlagcxP2pMr> &mrIds,
                               const std::vector<void *> &dstVec,
                               const std::vector<size_t> &sizeVec,
                               const std::vector<FlagcxP2pRdmaDesc> &descs,
                               int numIovs, uint64_t *transferId,
                               const std::vector<char *> &ipcBufs) {
  if (transferId == NULL)
    return -1;
  *transferId = 0;
  if (conn == NULL || numIovs <= 0)
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
  for (int i = 0; i < numIovs; i++) {
    if (!findMemRegByMr(mrIds[i], &localEntries[i]))
      return -1;

    if (!memRegContains(localEntries[i], reinterpret_cast<uintptr_t>(dstVec[i]),
                        sizeVec[i]))
      return -1;
  }

  return startNetTransfer(conn, dstVec, sizeVec, descs, localEntries, numIovs,
                          true, transferId);
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

static bool progressEngineTransfer(FlagcxP2pConn *conn, uint64_t transferId,
                                   flagcxResult_t *transferResult) {
  if (transferResult == NULL)
    return true;
  *transferResult = flagcxSuccess;
  std::lock_guard<std::mutex> lock(gXferMutex);
  std::unordered_map<uint64_t, FlagcxP2pXfer>::iterator it =
      gXferMap.find(transferId);
  if (it == gXferMap.end())
    return true;

  progressConnectionTransfersLocked(conn, transferId);

  FlagcxP2pXfer &xfer = it->second;
  if (xfer.kind == FLAGCX_P2P_XFER_IPC) {
    if (deviceAdaptor == NULL || deviceAdaptor->eventQuery == NULL) {
      *transferResult = flagcxInternalError;
      cleanupIpcXfer(&xfer);
      gXferMap.erase(it);
      return true;
    }

    const flagcxResult_t queryRes = deviceAdaptor->eventQuery(xfer.event);
    if (queryRes == flagcxSuccess) {
      TRACE(FLAGCX_P2P,
            "P2P local transfer completed conn=%p transferId=%llu result=%d",
            conn, (unsigned long long)transferId, (int)queryRes);
      cleanupIpcXfer(&xfer);
      gXferMap.erase(it);
      return true;
    }
    if (queryRes != flagcxInProgress) {
      *transferResult = queryRes;
      TRACE(FLAGCX_P2P,
            "P2P local transfer failed conn=%p transferId=%llu result=%d", conn,
            (unsigned long long)transferId, (int)queryRes);
      cleanupIpcXfer(&xfer);
      gXferMap.erase(it);
      return true;
    }
    return false;
  }

  if (xfer.transfer) {
    struct flagcxP2pTransferStatus status = {};
    const flagcxResult_t progress =
        flagcxP2pTransferProgress(xfer.transfer.get(), &status);
    if (progress != flagcxSuccess) {
      WARN("P2P shared transfer progress failed transferId=%llu result=%d",
           (unsigned long long)transferId, (int)progress);
      if (status.done) {
        *transferResult =
            status.result != flagcxSuccess ? status.result : progress;
        (void)flagcxP2pTransferReset(xfer.transfer.get());
        gXferMap.erase(it);
        return true;
      }
      return false;
    }
    if (!status.done)
      return false;
    *transferResult = status.result;
    if (status.result != flagcxSuccess) {
      WARN("P2P shared transfer completed with error transferId=%llu result=%d",
           (unsigned long long)transferId, (int)status.result);
    }
    uint64_t usedLanes = 0;
    for (uint64_t laneMask : xfer.laneMasks)
      usedLanes |= laneMask;
    TRACE(FLAGCX_P2P,
          "P2P shared transfer completed transferId=%llu ops=%u lanes=0x%llx",
          (unsigned long long)transferId, status.requested,
          (unsigned long long)usedLanes);
    if (flagcxP2pTransferReset(xfer.transfer.get()) != flagcxSuccess)
      WARN("P2P shared transfer reset failed transferId=%llu",
           (unsigned long long)transferId);
    gXferMap.erase(it);
    return true;
  }

  for (int i = xfer.completed; i < xfer.total; i++) {
    int done = 0;
    int sizes = 0;
    const flagcxResult_t testRes =
        conn->engine->adaptor->test(xfer.requests[i], &done, &sizes);
    if (testRes != flagcxSuccess) {
      *transferResult = testRes;
      return true;
    }
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

bool flagcxP2pEngineXferStatus(FlagcxP2pConn *conn, uint64_t transferId) {
  if (conn == NULL)
    return true;

  flagcxResult_t result = flagcxSuccess;
  const bool done = progressEngineTransfer(conn, transferId, &result);
  if (done && result != flagcxSuccess)
    WARN("P2P transfer %llu completed with result=%d",
         (unsigned long long)transferId, (int)result);
  return done;
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
  if (engine->isBarex)
    return -1;
  const int netDev = chooseEngineNetDev(engine);
  if (netDev < 0)
    return -1;
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
  INFO(FLAGCX_INIT, "NET/%s_P2P : RPC server thread started (port=%d)",
       engine->adaptor->name, flagcxP2pEngineGetRpcPort(engine));
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
  size_t first = findRemoteRegion(conn, remoteVa);
  if (first == SIZE_MAX && size == 0 && !conn->remoteRegions.empty()) {
    const size_t last = conn->remoteRegions.size() - 1;
    if (conn->remoteRegions[last].baseAddr + conn->remoteRegions[last].size ==
        remoteVa)
      first = last;
  }
  if (first == SIZE_MAX)
    return -1;
  const FlagcxP2pRemoteRegion &initial = conn->remoteRegions[first];
  uint64_t cursor = remoteVa;
  size_t remaining = size;
  size_t index = first;
  while (remaining > 0) {
    if (index >= conn->remoteRegions.size())
      return -1;
    const FlagcxP2pRemoteRegion &region = conn->remoteRegions[index];
    if (cursor < region.baseAddr || cursor - region.baseAddr >= region.size)
      return -1;
    const size_t available =
        region.size - static_cast<size_t>(cursor - region.baseAddr);
    const size_t bytes = std::min(remaining, available);
    cursor += bytes;
    remaining -= bytes;
    if (remaining > 0) {
      ++index;
      if (index >= conn->remoteRegions.size() ||
          conn->remoteRegions[index].baseAddr != cursor)
        return -1;
    }
  }
  memset(desc, 0, sizeof(*desc));
  desc->addr = remoteVa;
  desc->size = size;
  return flagcxP2pDescSetKeys(desc, initial.info.rkeys, initial.info.nKeys) ==
                 flagcxSuccess
             ? 0
             : -1;
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

  flagcxResult_t transferResult = flagcxSuccess;
  while (!progressEngineTransfer(conn, transferId, &transferResult)) {
    std::this_thread::yield();
  }
  return transferResult == flagcxSuccess ? 0 : -1;
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
      WARN("NET/P2P_ENGINE : BatchWriteSync MakeDesc failed for remote VA "
           "0x%llx size %llu",
           (unsigned long long)dstVa[i], (unsigned long long)sizes[i]);
      return -1;
    }
  }

  if (conn->isLocal && conn->sameProcess) {
    std::vector<FlagcxP2pMemRegEntry> batchEntries(count);
    std::vector<uintptr_t> srcAddrs(count);
    for (int i = 0; i < count; i++)
      srcAddrs[i] = static_cast<uintptr_t>(srcVa[i]);
    if (!findMemRegBatch(srcAddrs.data(), count, batchEntries.data())) {
      WARN("NET/P2P_ENGINE : BatchWriteSync no local MR for source VA");
      return -1;
    }
    std::vector<FlagcxP2pMr> mrVec(count);
    for (int i = 0; i < count; i++)
      mrVec[i] = batchEntries[i].mrId;
    return flagcxP2pEngineWriteVectorSync(conn, mrVec, srcVec, sizeVec, descs);
  }

  std::vector<FlagcxP2pMemRegEntry> localEntries(count);
  {
    std::vector<uintptr_t> srcAddrs(count);
    for (int i = 0; i < count; i++)
      srcAddrs[i] = static_cast<uintptr_t>(srcVa[i]);
    if (!findMemRegBatch(srcAddrs.data(), count, localEntries.data())) {
      WARN("NET/P2P_ENGINE : BatchWriteSync no local MR for source VA");
      return -1;
    }
  }
  for (int i = 0; i < count; i++) {
    if (!memRegContains(localEntries[i], static_cast<uintptr_t>(srcVa[i]),
                        static_cast<size_t>(sizes[i]))) {
      WARN("NET/P2P_ENGINE : BatchWriteSync source VA 0x%llx size %llu out of "
           "MR "
           "bounds",
           (unsigned long long)srcVa[i], (unsigned long long)sizes[i]);
      return -1;
    }
  }

  uint64_t xferId = 0;
  if (startNetTransfer(conn, srcVec, sizeVec, descs, localEntries, count, true,
                       &xferId) != 0)
    return -1;
  while (!flagcxP2pEngineXferStatus(conn, xferId))
    std::this_thread::yield();
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
  if (!findMemReg(addr, &entry))
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
