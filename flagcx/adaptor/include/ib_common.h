/*************************************************************************
 * Copyright (c) 2023 BAAI. All rights reserved.
 *
 * This file contains common InfiniBand structures and constants
 * shared between IBRC and UCX adaptors.
 ************************************************************************/

#ifndef FLAGCX_IB_COMMON_H_
#define FLAGCX_IB_COMMON_H_

#include "flagcx_net.h"
#include "ibvcore.h"
#include "ibvwrap.h"
#include "net.h"
#include <pthread.h>
#include <stdint.h>

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
  int ece_supported;
  int devIndex;
};

struct flagcxIbDevInfo {
  uint32_t lid;
  uint8_t ib_port;
  enum ibv_mtu mtu;
  uint8_t link_layer;
  uint64_t spn;
  uint64_t iid;
  uint32_t fifoRkey;
  union ibv_gid remoteGid;
};

struct flagcxIbGidInfo {
  uint8_t link_layer;
  union ibv_gid localGid;
  int32_t localGidIndex;
};

struct flagcxIbMrHandle {
  ibv_mr *mrs[FLAGCX_IB_MAX_DEVS_PER_NIC];
};

// Structure to store handle info for allgather
struct flagcxIbGlobalHandleInfo {
  uintptr_t *base_vas;
  uint32_t *rkeys;
  uint32_t *lkeys;
};

// Global variable for one-sided operation handles
extern struct flagcxIbGlobalHandleInfo *globalOneSideHandles;

#define FLAGCX_NET_IB_REQ_UNUSED 0
#define FLAGCX_NET_IB_REQ_SEND 1
#define FLAGCX_NET_IB_REQ_RECV 2
#define FLAGCX_NET_IB_REQ_FLUSH 3
#define FLAGCX_NET_IB_REQ_IPUT 4

extern const char *reqTypeStr[];

#define FLAGCX_IB_RETRANS_MAX_INFLIGHT 2048
#define FLAGCX_IB_RETRANS_BUFFER_SIZE 1024
#define FLAGCX_IB_RETRANS_MAX_CHUNK_SIZE (8 * 1024 * 1024)
#define FLAGCX_IB_SRQ_SIZE 1024

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
  int retry_count;
  int valid;
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
  uint64_t idx;
  char padding[24];
};

struct flagcxIbRequest {
  struct flagcxIbNetCommBase *base;
  int type;
  struct flagcxSocket *sock;
  int events[FLAGCX_IB_MAX_DEVS_PER_NIC];
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

  uint32_t ctrlQpn[FLAGCX_IB_MAX_DEVS_PER_NIC];
  union ibv_gid ctrlGid[FLAGCX_IB_MAX_DEVS_PER_NIC];
  uint16_t ctrlLid[FLAGCX_IB_MAX_DEVS_PER_NIC];
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

  struct flagcxIbCtrlQp ctrlQp;
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
  // Track necessary remDevInfo here
  int nRemDevs;
  struct flagcxIbDevInfo remDevs[FLAGCX_IB_MAX_DEVS_PER_NIC];
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
  int ar;

  struct flagcxIbRetransState retrans;
  uint64_t last_timeout_check_us;

  int outstanding_sends;
  int outstanding_retrans;
  int max_outstanding;

  struct flagcxIbRetransHdr retrans_hdr_pool[32];
  struct ibv_mr *retrans_hdr_mr;
};

struct flagcxIbGpuFlush {
  struct ibv_mr *hostMr;
  struct ibv_sge sge;
  struct flagcxIbQp qp;
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
  struct ibv_mr *ackMr;
  void *ackBuffer;

  void *retransRecvBufs[32];
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
extern int64_t flagcxParamIbDisable(void);
extern int64_t flagcxParamIbMergeVfs(void);
extern int64_t flagcxParamIbMergeNics(void);
extern int64_t flagcxParamIbQpsPerConn(void);

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
extern flagcxResult_t flagcxIbDmaBufSupport(int dev);
extern flagcxResult_t flagcxIbFreeRequest(struct flagcxIbRequest *r);

struct flagcxIbCommonTestOps {
  const char *component;
  flagcxResult_t (*pre_check)(struct flagcxIbRequest *req);
  flagcxResult_t (*process_wc)(struct flagcxIbRequest *req, struct ibv_wc *wc,
                               int devIndex, bool *handled);
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

static_assert((sizeof(struct flagcxIbNetCommBase) % 32) == 0,
              "flagcxIbNetCommBase size must be 32-byte multiple to ensure "
              "fifo is at proper offset");
static_assert((offsetof(struct flagcxIbSendComm, fifo) % 32) == 0,
              "flagcxIbSendComm fifo must be 32-byte aligned");
static_assert((sizeof(struct flagcxIbSendFifo) % 32) == 0,
              "flagcxIbSendFifo element size must be 32-byte multiples");
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

#endif // FLAGCX_IB_COMMON_H_
