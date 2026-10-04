/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * This file contains common InfiniBand structures and constants
 * shared between IBRC and UCX adaptors.
 ************************************************************************/

#ifndef FLAGCX_IB_COMMON_H_
#define FLAGCX_IB_COMMON_H_

#include "flagcx_net.h"
#include "ibv_compat.h"
#include "ibvwrap.h"
#include "net.h"
#include "onesided.h"
#include <pthread.h>
#include <stdint.h>

// Backward-compat alias so adaptor code can keep using the old name.
typedef struct flagcxOneSideHandleInfo flagcxIbGlobalHandleInfo;

#define MAXNAMESIZE 64
#define MAX_IB_DEVS 32
#define FLAGCX_IB_MAX_DEVS_PER_NIC 2
#define FLAGCX_NET_MAX_DEVS_PER_NIC 4
#define MAX_MERGED_DEV_NAME                                                    \
  (MAXNAMESIZE * FLAGCX_IB_MAX_DEVS_PER_NIC) + FLAGCX_IB_MAX_DEVS_PER_NIC
#define MAX_IB_VDEVS MAX_IB_DEVS * 8

#define ENABLE_TIMER 0
#define FLAGCX_IB_MAX_QPS 128
#define FLAGCX_NET_IB_MAX_RECVS 8
#define MAX_REQUESTS (FLAGCX_NET_MAX_REQUESTS * FLAGCX_NET_IB_MAX_RECVS)

enum flagcxIbProvider {
  IB_PROVIDER_NONE = 0,
  IB_PROVIDER_MLX5 = 1,
  IB_PROVIDER_MLX4 = 2
};

static const char *ibProviderName[]
    __attribute__((unused)) = {"NONE", "MLX5", "MLX4"};

extern int64_t flagcxParamIbMergeVfs(void);
extern int64_t flagcxParamIbAdaptiveRouting(void);
extern int64_t flagcxParamIbMergeNics(void);

static inline uint32_t flagcxIbPortLid(const struct ibv_port_attr *portAttr) {
#ifdef USE_SHCA
  return u17_to_32(portAttr->lid);
#else
  return portAttr->lid;
#endif
}

static inline flagcxResult_t flagcxIbSetAhDlid(struct ibv_ah_attr *ahAttr,
                                               uint32_t lid) {
#ifdef USE_SHCA
  ahAttr->dlid = u32_to_17(lid);
#else
  if (lid > UINT16_MAX)
    return flagcxInvalidArgument;
  ahAttr->dlid = (uint16_t)lid;
#endif
  return flagcxSuccess;
}

static inline bool flagcxIbUseGlobalRoute(uint8_t linkLayer) {
#ifdef USE_SHCA
  // SHCA programs a GID/GRH route together with its extended 17-bit DLID.
  (void)linkLayer;
  return true;
#else
  return linkLayer == IBV_LINK_LAYER_ETHERNET;
#endif
}

struct flagcxIbMr {
  uintptr_t addr;
  size_t pages;
  int refs;
  struct ibv_mr *mr;
};

struct flagcxIbMrCache {
  struct flagcxIbMr *slots;
  int capacity, population;
};

struct flagcxIbStats {
  int fatalErrorCount;
};

struct flagcxIbDev {
  pthread_mutex_t lock;
  int device;
  int ibProvider;
  uint64_t guid;
  struct ibv_port_attr portAttr;
  uint32_t lid;
  int portNum;
  int link;
  int speed;
  struct ibv_context *context;
  int pdRefs;
  struct ibv_pd *pd;
  char devName[MAXNAMESIZE];
  char *pciPath;
  int realPort;
  int maxQp;
  // Per-QP responder and initiator RDMA Read/Atomic limits reported by
  // ibv_query_device().
  int maxQpRdAtomic;
  int maxQpInitRdAtomic;
  struct flagcxIbMrCache mrCache;
  struct flagcxIbStats stats;
  int ar; // ADAPTIVE_ROUTING
  int isSharpDev;
  struct {
    struct {
      int dataDirect;
    } mlx5;
  } capsProvider;
  int dmaBufSupported;
} __attribute__((aligned(64)));

struct flagcxIbMergedDev {
  int ndevs;
  int devs[FLAGCX_IB_MAX_DEVS_PER_NIC];
  flagcxNetVDeviceProps_t vProps;
  int speed;
  char devName[MAX_MERGED_DEV_NAME];
} __attribute__((aligned(64)));

struct flagcxIbQpInfo {
  uint32_t qpn;
  struct ibv_ece ece;
  int eceSupported;
  int devIndex;
};

struct flagcxIbDevInfo {
  uint32_t lid;
  uint8_t ibPort;
  enum ibv_mtu mtu;
  uint8_t linkLayer;
  uint64_t spn;
  uint64_t iid;
  uint32_t fifoRkey;
  // Responder credits selected by this peer for QPs on this physical HCA.
  uint8_t maxDestRdAtomic;
  union ibv_gid remoteGid;
};

static inline uint8_t flagcxIbResponderAtomicDepth(int64_t requestedDepth,
                                                   int localCap) {
  if (requestedDepth <= 0 || localCap <= 0)
    return 0;
  uint64_t depth = (uint64_t)requestedDepth;
  if (depth > (uint64_t)localCap)
    depth = (uint64_t)localCap;
  if (depth > UINT8_MAX)
    depth = UINT8_MAX;
  return (uint8_t)depth;
}

static inline uint8_t flagcxIbInitiatorAtomicDepth(int64_t requestedDepth,
                                                   int localCap,
                                                   int remoteResponderDepth) {
  uint8_t depth = flagcxIbResponderAtomicDepth(requestedDepth, localCap);
  if (remoteResponderDepth <= 0)
    return 0;
  uint8_t remoteDepth = remoteResponderDepth > UINT8_MAX
                            ? UINT8_MAX
                            : (uint8_t)remoteResponderDepth;
  if (depth > remoteDepth)
    depth = remoteDepth;
  return depth;
}

struct flagcxIbGidInfo {
  uint8_t linkLayer;
  union ibv_gid localGid;
  int32_t localGidIndex;
};

struct flagcxIbMrHandle {
  ibv_mr *mrs[FLAGCX_IB_MAX_DEVS_PER_NIC];
  struct flagcxIbMrHandle *nextDeferred;
#ifdef USE_IBUC
  int type;
#endif
};

#define FLAGCX_NET_IB_REQ_UNUSED 0
#define FLAGCX_IB_UNSIGNALED_WR_ID_PREFIX 0xffffffffffffff00ULL

static inline uint64_t flagcxIbUnsignaledWrId(uint8_t reqIndex) {
  return FLAGCX_IB_UNSIGNALED_WR_ID_PREFIX | reqIndex;
}

static inline bool flagcxIbIsUnsignaledWrId(uint64_t wrId) {
  return (wrId & FLAGCX_IB_UNSIGNALED_WR_ID_PREFIX) ==
         FLAGCX_IB_UNSIGNALED_WR_ID_PREFIX;
}
#define FLAGCX_NET_IB_REQ_SEND 1
#define FLAGCX_NET_IB_REQ_RECV 2
#define FLAGCX_NET_IB_REQ_FLUSH 3
#define FLAGCX_NET_IB_REQ_IPUT 4
#define FLAGCX_NET_IB_REQ_IGET 5

extern const char *reqTypeStr[];

#define FLAGCX_IB_RETRANS_MAX_INFLIGHT 2048
#define FLAGCX_IB_RETRANS_BUFFER_SIZE 1024
#define FLAGCX_IB_RETRANS_MAX_CHUNK_SIZE (8 * 1024 * 1024)
#define FLAGCX_IB_SRQ_SIZE 1024
#define FLAGCX_IBUC_RETRANS_RECV_DEPTH 16

static inline uint32_t flagcxIbucRetransChunkCount(uint32_t size) {
  return size == 0 ? 1 : 1 + (size - 1) / FLAGCX_IB_RETRANS_MAX_CHUNK_SIZE;
}

#define FLAGCX_IB_ACK_BUF_PADDING 40
#define FLAGCX_IB_ACK_BUF_COUNT 64

struct flagcxIbRetransHdr {
  uint32_t magic;
  uint32_t seq;
  uint32_t size;
  uint32_t rkey;
  uint64_t remoteAddr;
  uint32_t immData;
  uint32_t padding;
} __attribute__((packed));

struct flagcxIbAckMsg {
  uint16_t peerId;
  uint16_t flowId;
  uint16_t path;
  uint16_t ackSeq;
  uint16_t sackBitmapCount;
  uint16_t padding;
  uint64_t timestampUs;
  uint64_t sackBitmap;
} __attribute__((packed));

struct flagcxIbCtrlQp {
  struct ibv_qp *qp;
  struct ibv_cq *cq;
  struct ibv_ah *ah;
  uint32_t remoteQpn;
  uint32_t remoteQkey;
};

struct flagcxIbRetransRecvBuf {
  void *buffer;
  struct ibv_mr *mr;
  size_t size;
  int inUse;
};

struct flagcxIbSrqMgr {
  void *srq;
  struct ibv_cq *cq;
  struct flagcxIbRetransRecvBuf bufs[FLAGCX_IB_SRQ_SIZE];
  int bufCount;
  // Buffer management for SRQ (similar to UCCL)
  int freeBufIndices[FLAGCX_IB_SRQ_SIZE]; // Stack of free buffer indices
  int freeBufCount;                       // Number of free buffers available
  int postSrqCount; // Number of recv WRs that need to be posted to SRQ
};

struct flagcxIbRetransEntry {
  uint32_t seq;
  uint32_t size;
  uint64_t sendTimeUs;
  uint64_t remoteAddr;
  void *data;
  uint32_t lkeys[FLAGCX_IB_MAX_DEVS_PER_NIC];
  uint32_t rkeys[FLAGCX_IB_MAX_DEVS_PER_NIC];
  int retryCount;
  int valid;

#ifdef USE_IBUC
  // IBUC retransmits a whole logical receive over a dedicated RC QP. Keep the
  // segment and request association until the receiver acknowledges the
  // sequence so the source buffers cannot be recycled prematurely.
  int nreqs;
  uint8_t remoteRequestSlot;
  uint16_t remoteGeneration;
  uint8_t ackDevIndex;
  struct flagcxIbRequest *requests[FLAGCX_NET_IB_MAX_RECVS];
  struct {
    uint32_t size;
    void *data;
    uint32_t lkeys[FLAGCX_IB_MAX_DEVS_PER_NIC];
  } segments[FLAGCX_NET_IB_MAX_RECVS];
  uint8_t retransSegment;
  uint32_t retransOffset;
  bool retransAllPosted;
#endif
};

struct flagcxIbRetransState {
  uint32_t sendSeq;
  uint32_t sendUna;
  uint32_t recvSeq;

  struct flagcxIbRetransEntry buffer[FLAGCX_IB_RETRANS_MAX_INFLIGHT];
  int bufferHead;
  int bufferTail;
  int bufferCount;

  uint64_t lastAckTimeUs;
  uint64_t rtoUs;
  uint64_t srttUs;
  uint64_t rttvarUs;

  uint64_t totalSent;
  uint64_t totalRetrans;
  uint64_t totalAcked;
  uint64_t totalTimeout;

  int enabled;
  int maxRetry;
  int ackInterval;
  uint32_t minRtoUs;
  uint32_t maxRtoUs;
  int retransQPIndex;
  uint32_t lastAckSeq;
  uint64_t lastAckSendTimeUs;
};

struct flagcxIbQp {
  struct ibv_qp *qp;
  int devIndex;
  int remDevIdx;
};

struct flagcxIbSendFifo {
  uint64_t addr;
  size_t size;
  uint32_t rkeys[FLAGCX_IB_MAX_DEVS_PER_NIC];
  uint32_t nreqs;
  uint32_t tag;
  uint32_t requestSlot;
  uint16_t generation;
  char padding[18];
  // The sender treats idx as the publication marker for the complete FIFO
  // record. Keep it after every field consumed by the sender so observing a
  // new idx cannot expose request metadata from a previous slot generation.
  uint64_t idx;
};

struct flagcxIbRequest {
  struct flagcxIbNetCommBase *base;
  int type;
  // Completion errors are recorded on the request identified by wr_id. Data
  // CQs are shared, so the request being polled is not necessarily the one
  // whose completion was returned.
  flagcxResult_t result;
  struct flagcxSocket *sock;
  int events[FLAGCX_IB_MAX_DEVS_PER_NIC];
#ifdef USE_IBUC
  uint32_t retransSeq;
  uint16_t retransGeneration;
  uint8_t retransSegmentMask;
  uint32_t retransSegmentBytes[FLAGCX_NET_IB_MAX_RECVS];
  // Receive requests also own a FIFO-write completion. Track UC data
  // notifications separately so ACK generation is independent of that CQE's
  // arrival order.
  int dataEvents[FLAGCX_IB_MAX_DEVS_PER_NIC];
#endif
  struct flagcxIbNetCommDevBase *devBases[FLAGCX_IB_MAX_DEVS_PER_NIC];
  int nreqs;
  union {
    struct {
      int size;
      void *data;
      uint32_t lkeys[FLAGCX_IB_MAX_DEVS_PER_NIC];
      int offset;
    } send;
    struct {
      int *sizes;
#ifdef USE_IBUC
      void *data[FLAGCX_NET_IB_MAX_RECVS];
      int types[FLAGCX_NET_IB_MAX_RECVS];
      size_t capacities[FLAGCX_NET_IB_MAX_RECVS];
#endif
    } recv;
  };
};

struct flagcxIbListenComm {
  int dev;
  struct flagcxSocket sock;
  struct flagcxIbCommStage stage;
};

struct flagcxIbConnectionMetadata {
  struct flagcxIbQpInfo qpInfo[FLAGCX_IB_MAX_QPS];
  struct flagcxIbDevInfo devs[FLAGCX_IB_MAX_DEVS_PER_NIC];
  char devName[MAX_MERGED_DEV_NAME];
  uint64_t fifoAddr;
  int ndevs;
  // Data-lane geometry is negotiated explicitly. Both endpoints must use the
  // same QP count and striping width for deterministic lane mapping.
  int nqps;
  int splitDataOnQps;

  uint32_t ctrlQpn[FLAGCX_IB_MAX_DEVS_PER_NIC];
  uint32_t retransQpn[FLAGCX_IB_MAX_DEVS_PER_NIC];
  union ibv_gid ctrlGid[FLAGCX_IB_MAX_DEVS_PER_NIC];
  uint32_t ctrlLid[FLAGCX_IB_MAX_DEVS_PER_NIC];
  int retransEnabled;
};

struct flagcxIbNetCommDevBase {
  int ibDevN;
  struct ibv_pd *pd;
  struct ibv_cq *cq;
  uint64_t pad[2];
  struct flagcxIbGidInfo gidInfo;
};

struct flagcxIbRemSizesFifo {
  int elems[MAX_REQUESTS][FLAGCX_NET_IB_MAX_RECVS];
  uint64_t fifoTail;
  uint64_t addr;
  uint32_t rkeys[FLAGCX_IB_MAX_DEVS_PER_NIC];
  uint32_t flags;
  struct ibv_mr *mrs[FLAGCX_IB_MAX_DEVS_PER_NIC];
  struct ibv_sge sge;
};

struct flagcxIbSendCommDev {
  struct flagcxIbNetCommDevBase base;
  struct ibv_mr *fifoMr;
  struct ibv_mr *putSignalScratchpadMr;

  struct flagcxIbCtrlQp ctrlQp;
  struct flagcxIbQp retransQp;
  // Keep retransmission SEND completions and reliable ACK receives off the
  // data CQ. Data-QP SQ recovery is allowed to poll the data CQ directly.
  struct ibv_cq *retransCq;
  struct ibv_mr *retransHdrMr;
  struct ibv_mr *ackMr;
  void *ackBuffer;
};

struct alignas(32) flagcxIbNetCommBase {
  int ndevs;
  bool isSend;
  struct flagcxIbRequest reqs[MAX_REQUESTS];
  struct flagcxIbQp qps[FLAGCX_IB_MAX_QPS];
  int nqps;
  int qpIndex;
  int devIndex;
  struct flagcxSocket sock;
  int ready;
  // First permanent data-plane error observed on this communicator. Once a
  // CQE fails, no later request may be posted or wait indefinitely for other
  // completions from the failed QP.
  flagcxResult_t asyncResult;
  // Track necessary remDevInfo here
  int nRemDevs;
  struct flagcxIbDevInfo remDevs[FLAGCX_IB_MAX_DEVS_PER_NIC];
  // A registration rollback can itself fail. Retain partially cleaned
  // wrappers until close retries them before destroying their QPs and PDs.
  struct flagcxIbMrHandle *deferredMrHandles;
  // IBUC setup or close can fail after partially releasing a communicator.
  // closeSend/closeRecv consume their handles, so retain any unreleased
  // resources on an adaptor-owned retry list.
  struct flagcxIbNetCommBase *nextDeferredCleanup;
  bool cleanupDeferred;
};

struct flagcxIbSendComm {
  struct flagcxIbNetCommBase base;
  struct flagcxIbSendFifo fifo[MAX_REQUESTS][FLAGCX_NET_IB_MAX_RECVS];
  // Each dev correlates to a mergedIbDev
  struct flagcxIbSendCommDev devs[FLAGCX_IB_MAX_DEVS_PER_NIC];
  struct flagcxIbRequest *fifoReqs[MAX_REQUESTS][FLAGCX_NET_IB_MAX_RECVS];
  alignas(32) struct ibv_sge sges[FLAGCX_NET_IB_MAX_RECVS];
  alignas(32) struct ibv_send_wr wrs[FLAGCX_NET_IB_MAX_RECVS + 1];
  struct flagcxIbRemSizesFifo remSizesFifo;
  uint64_t fifoHead;
  uint64_t putSignalScratchpad;
  int ar;

  struct flagcxIbRetransState retrans;
  uint64_t lastTimeoutCheckUs;

  int outstandingSends;
  int outstandingRetrans;
  int maxOutstanding;
  bool retransUsesRc;
  int retransWindowNreqs;
  uint8_t retransWindowDevIndex;
  struct flagcxIbRequest *retransWindowRequests[FLAGCX_NET_IB_MAX_RECVS];

  struct flagcxIbRetransHdr retransHdrPool[32];
  struct ibv_mr *retransHdrMr;
};

struct flagcxIbGpuFlush {
  struct ibv_mr *hostMr;
  struct ibv_sge sge;
  struct flagcxIbQp qp;
};

enum flagcxIbLazyInitState {
  FLAGCX_IB_LAZY_INIT_UNINITIALIZED = 0,
  FLAGCX_IB_LAZY_INIT_IN_PROGRESS = 1,
  FLAGCX_IB_LAZY_INIT_READY = 2,
  FLAGCX_IB_LAZY_INIT_FAILED = 3,
};

// Receive communicators are zero-allocated, so this control block starts in
// UNINITIALIZED without a separate constructor.  The implementation uses
// acquire/release atomics to let concurrent first users share one initializer.
struct flagcxIbLazyInit {
  int state;
  flagcxResult_t result;
};

struct alignas(32) flagcxIbRemFifo {
  struct flagcxIbSendFifo elems[MAX_REQUESTS][FLAGCX_NET_IB_MAX_RECVS];
  uint64_t fifoTail;
  uint64_t addr;
  uint32_t flags;
};

struct alignas(16) flagcxIbRecvCommDev {
  struct flagcxIbNetCommDevBase base;
  struct flagcxIbGpuFlush gpuFlush;
  uint32_t fifoRkey;
  struct ibv_mr *fifoMr;
  struct ibv_sge fifoSge;
  struct ibv_mr *sizesFifoMr;
  struct flagcxIbCtrlQp ctrlQp;
  struct flagcxIbQp retransQp;
  struct ibv_mr *ackMr;
  void *ackBuffer;

  void *retransRecvBufs[FLAGCX_IBUC_RETRANS_RECV_DEPTH];
  struct ibv_mr *retransRecvMr;
  int retransRecvBufCount;
};

struct alignas(32) flagcxIbRecvComm {
  struct flagcxIbNetCommBase base;
  struct flagcxIbRecvCommDev devs[FLAGCX_IB_MAX_DEVS_PER_NIC];
  struct flagcxIbRemFifo remFifo;
  int sizesFifo[MAX_REQUESTS][FLAGCX_NET_IB_MAX_RECVS];
  int gpuFlushHostMem;
  int flushEnabled;
  struct flagcxIbLazyInit flushInit;

  struct flagcxIbRetransState retrans;
  struct flagcxIbSrqMgr srqMgr;
};

// Global arrays (declared as extern, defined in adaptor files)
extern struct flagcxIbDev flagcxIbDevs[MAX_IB_DEVS];
extern struct flagcxIbMergedDev flagcxIbMergedDevs[MAX_IB_VDEVS];

// Global variables (declared as extern, defined in adaptor files)
extern char flagcxIbIfName[MAX_IF_NAME_SIZE + 1];
extern union flagcxSocketAddress flagcxIbIfAddr;
extern int flagcxNMergedIbDevs;
extern int flagcxNIbDevs;
extern pthread_mutex_t flagcxIbLock;
extern int flagcxIbRelaxedOrderingEnabled;
extern pthread_t flagcxIbAsyncThread;

// Parameter functions
extern int64_t flagcxParamIbGidIndex(void);
extern int64_t flagcxParamIbRoceVersionNum(void);
extern int64_t flagcxParamIbTimeout(void);
extern int64_t flagcxParamIbRetryCnt(void);
extern int64_t flagcxParamIbPkey(void);
extern int64_t flagcxParamIbUseInline(void);
extern int64_t flagcxParamIbSl(void);
extern int64_t flagcxParamIbTc(void);
extern int64_t flagcxParamIbArThreshold(void);
extern int64_t flagcxParamIbPciRelaxedOrdering(void);
extern int64_t flagcxParamIbAdaptiveRouting(void);
extern int64_t flagcxParamIbMergeVfs(void);
extern int64_t flagcxParamIbMergeNics(void);
extern int64_t flagcxParamIbQpsPerConn(void);
extern int64_t flagcxParamIbRdAtomicDepth(void);

extern sa_family_t envIbAddrFamily(void);
extern void *envIbAddrRange(sa_family_t af, int *mask);
extern sa_family_t getGidAddrFamily(union ibv_gid *gid);
extern bool matchGidAddrPrefix(sa_family_t af, void *prefix, int prefixlen,
                               union ibv_gid *gid);
extern bool configuredGid(union ibv_gid *gid);
extern bool linkLocalGid(union ibv_gid *gid);
extern bool validGid(union ibv_gid *gid);
extern flagcxResult_t flagcxIbRoceGetVersionNum(const char *deviceName,
                                                int portNum, int gidIndex,
                                                int *version);
extern flagcxResult_t flagcxUpdateGidIndex(struct ibv_context *context,
                                           uint8_t portNum, sa_family_t af,
                                           void *prefix, int prefixlen,
                                           int roceVer, int gidIndexCandidate,
                                           int *gidIndex);
extern flagcxResult_t flagcxIbGetGidIndex(struct ibv_context *context,
                                          uint8_t portNum, int gidTblLen,
                                          int *gidIndex);
extern flagcxResult_t flagcxIbGetPciPath(char *devName, char **path,
                                         int *realPort);
extern int flagcxIbWidth(int width);
extern int flagcxIbSpeed(int speed);
extern int flagcxIbRelaxedOrderingCapable(void);
extern int flagcxIbFindMatchingDev(int dev);
extern void *flagcxIbAsyncThreadMain(void *args);

extern int ibvWidths[];
extern int ibvSpeeds[];

extern int firstBitSet(int val, int max);

extern flagcxResult_t flagcxIbDevices(int *ndev);
extern flagcxResult_t flagcxIbGdrSupport(void);
extern flagcxResult_t flagcxIbProbeGpuMrSupport(int dev, int access,
                                                bool *supported);
extern flagcxResult_t flagcxIbDmaBufSupport(int dev);
extern flagcxResult_t flagcxIbFreeRequest(struct flagcxIbRequest *r);

struct flagcxIbCommonTestOps {
  const char *component;
  flagcxResult_t (*pre_check)(struct flagcxIbRequest *req);
  flagcxResult_t (*process_wc)(struct flagcxIbRequest *req, struct ibv_wc *wc,
                               int devIndex, bool *handled);
  // Some transports place control completions on the data CQ. Poll every CQ
  // associated with the request even when that device has no remaining data
  // events; wr_id still routes each completion to its owning request.
  bool pollAllCqs;
};

flagcxResult_t
flagcxIbCommonPostFifo(struct flagcxIbRecvComm *comm, int n, void **data,
                       size_t *sizes, int *tags, void **mhandles,
                       struct flagcxIbRequest *req,
                       void (*addEventFunc)(struct flagcxIbRequest *, int,
                                            struct flagcxIbNetCommDevBase *));

flagcxResult_t
flagcxIbCommonTestDataQp(struct flagcxIbRequest *r, int *done, int *sizes,
                         const struct flagcxIbCommonTestOps *ops);
flagcxResult_t
flagcxIbCommonRecordDataCompletion(struct flagcxIbNetCommBase *base,
                                   uint64_t wrId, int devIndex,
                                   flagcxResult_t result);
flagcxResult_t
flagcxIbCommonRecordUnsignaledCompletion(struct flagcxIbNetCommBase *base,
                                         uint64_t wrId, flagcxResult_t result);
flagcxResult_t flagcxIbCommonRecordCommError(struct flagcxIbNetCommBase *base,
                                             flagcxResult_t result);
flagcxResult_t
flagcxIbCommonGetCommError(const struct flagcxIbNetCommBase *base);
flagcxResult_t flagcxIbRunLazyInit(struct flagcxIbLazyInit *control,
                                   flagcxResult_t (*initialize)(void *),
                                   flagcxResult_t (*rollback)(void *),
                                   void *opaque);

static_assert((sizeof(struct flagcxIbNetCommBase) % 32) == 0,
              "flagcxIbNetCommBase size must be 32-byte multiple to ensure "
              "fifo is at proper offset");
static_assert((offsetof(struct flagcxIbSendComm, fifo) % 32) == 0,
              "flagcxIbSendComm fifo must be 32-byte aligned");
static_assert((sizeof(struct flagcxIbSendFifo) % 32) == 0,
              "flagcxIbSendFifo element size must be 32-byte multiples");
static_assert(sizeof(struct flagcxIbSendFifo) == 64,
              "flagcxIbSendFifo wire record must remain 64 bytes");
static_assert(offsetof(struct flagcxIbSendFifo, idx) == 56,
              "flagcxIbSendFifo publication marker must remain last");
static_assert((offsetof(struct flagcxIbSendComm, sges) % 32) == 0,
              "sges must be 32-byte aligned");
static_assert((offsetof(struct flagcxIbSendComm, wrs) % 32) == 0,
              "wrs must be 32-byte aligned");
static_assert((offsetof(struct flagcxIbRecvComm, remFifo) % 32) == 0,
              "flagcxIbRecvComm fifo must be 32-byte aligned");
static_assert(
    sizeof(struct flagcxIbHandle) < FLAGCX_NET_HANDLE_MAXSIZE,
    "flagcxIbHandle size must be smaller than FLAGCX_NET_HANDLE_MAXSIZE");

static_assert(MAX_REQUESTS <= 256, "request id are encoded in wr_id and we "
                                   "need up to 8 requests ids per completion");

// Shared IBRC helpers (defined in ibrc_adaptor.cc, used by ibrc p2p adaptor)
flagcxResult_t flagcxIbInit();
flagcxResult_t flagcxIbGetProperties(int dev, void *props);
flagcxResult_t flagcxIbCreateQp(uint8_t ib_port,
                                struct flagcxIbNetCommDevBase *base,
                                int accessFlags, struct flagcxIbQp *qp);
flagcxResult_t flagcxIbRtrQp(struct ibv_qp *qp,
                             const struct flagcxIbDev *localDev,
                             const struct flagcxIbGidInfo *localGidInfo,
                             uint32_t dest_qp_num,
                             const struct flagcxIbDevInfo *info);
flagcxResult_t flagcxIbRtsQp(struct ibv_qp *qp,
                             const struct flagcxIbDev *localDev,
                             const struct flagcxIbDevInfo *remoteInfo);
flagcxResult_t flagcxIbRegMrDmaBufInternal(flagcxIbNetCommDevBase *base,
                                           void *data, size_t size, int type,
                                           uint64_t offset, int fd, int mrFlags,
                                           ibv_mr **mhandle);
flagcxResult_t flagcxIbDeregMrInternal(flagcxIbNetCommDevBase *base,
                                       ibv_mr *mhandle);
typedef flagcxResult_t (*flagcxIbDeregMrCallback)(flagcxIbNetCommDevBase *base,
                                                  ibv_mr *mhandle);
flagcxResult_t flagcxIbDeregMrWithCallback(void *comm, void *mhandle,
                                           flagcxIbDeregMrCallback callback);
flagcxResult_t
flagcxIbDeregMrOrDeferWithCallback(struct flagcxIbNetCommBase *base,
                                   void *mhandle,
                                   flagcxIbDeregMrCallback callback);
flagcxResult_t
flagcxIbDrainDeferredMrsWithCallback(struct flagcxIbNetCommBase *base,
                                     flagcxIbDeregMrCallback callback);

#ifdef USE_IBUC
// Internal IBUC lifetime helpers exposed for transport-level ownership tests.
flagcxResult_t flagcxIbucInitCommDevBase(int ibDevN,
                                         struct flagcxIbNetCommDevBase *base);
flagcxResult_t flagcxIbucDestroyBase(struct flagcxIbNetCommDevBase *base);
flagcxResult_t flagcxIbucCloseSend(void *sendComm);
flagcxResult_t flagcxIbucCloseRecv(void *recvComm);
#endif

#endif // FLAGCX_IB_COMMON_H_
