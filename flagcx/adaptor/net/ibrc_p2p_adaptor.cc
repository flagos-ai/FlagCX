/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * IBRC P2P Net Adaptor — implements flagcxNetAdaptor for one-sided RDMA
 * (P2P) use cases. Shares IB device discovery and utility code with the
 * existing IBRC adaptor but uses P2P-native handle formats, eager PD
 * allocation, and simplified (single-QP, no-FIFO) connection setup.
 ************************************************************************/

#include "flagcx_common.h"
#include "flagcx_net_adaptor.h"
#include "ib_common.h"
#include "ibvwrap.h"
#include "socket.h"

#include <algorithm>
#include <assert.h>
#include <chrono>
#include <pthread.h>
#include <string.h>
#include <thread>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/*  Internal structs                                                   */
/* ------------------------------------------------------------------ */

// Per-device context — created at init, holds eagerly allocated PD.
// Passed as the `comm` parameter to regMr/deregMr when no connection exists.
// ibDevN MUST be the first field so regMr can cast any comm pointer to extract
// it.
struct flagcxP2pDevCtx {
  int ibDevN;
  struct ibv_pd *pd;
};

// P2P MR handle — replaces rank-indexed flagcxOneSideHandleInfo
struct flagcxP2pMrHandle {
  uintptr_t baseVa;
  uint32_t lkey;
  uint32_t rkey;
  ibv_mr *mr;
  int ibDevN; // for cache lookup during deregMr
};

// P2P listen handle — stable wire metadata only, no mutable stage
struct flagcxP2pListenHandle {
  union flagcxSocketAddress connectAddr;
  uint64_t magic;
};
static_assert(sizeof(struct flagcxP2pListenHandle) <= FLAGCX_NET_HANDLE_MAXSIZE,
              "P2P listen handle must fit in FLAGCX_NET_HANDLE_MAXSIZE");

// P2P listen comm
struct flagcxP2pListenComm {
  int dev;
  struct flagcxSocket sock;
};

// Connection metadata exchanged over TCP during connect/accept
struct flagcxP2pConnMeta {
  uint32_t qpn;
  union ibv_gid gid;
  uint8_t ibPort;
  uint8_t linkLayer;
  uint32_t lid;
  enum ibv_mtu mtu;
};

// P2P request — simplified from flagcxIbRequest
#define FLAGCX_P2P_MAX_REQUESTS 32
#define FLAGCX_P2P_REQ_UNUSED 0
#define FLAGCX_P2P_REQ_IPUT 1
#define FLAGCX_P2P_REQ_IGET 2

struct flagcxP2pRequest {
  int type;
  int events;                    // outstanding CQEs expected
  struct ibv_cq *cq;             // CQ to poll for this request
  struct flagcxP2pRequest *reqs; // back-pointer to owning reqs[] array
};

// P2P send comm — one QP, one CQ, blocking connect
struct flagcxP2pSendComm {
  int ibDevN; // MUST be first field
  struct flagcxIbNetCommDevBase base;
  struct flagcxIbQp qp;
  struct flagcxSocket sock;
  struct flagcxP2pRequest reqs[FLAGCX_P2P_MAX_REQUESTS];
  uint64_t putSignalScratchpad;
  struct ibv_mr *putSignalScratchpadMr;
};

// P2P recv comm — symmetric with send comm so both sides can initiate transfers
struct flagcxP2pRecvComm {
  int ibDevN; // MUST be first field
  struct flagcxIbNetCommDevBase base;
  struct flagcxIbQp qp;
  struct flagcxSocket sock;
  struct flagcxP2pRequest reqs[FLAGCX_P2P_MAX_REQUESTS];
  uint64_t putSignalScratchpad;
  struct ibv_mr *putSignalScratchpadMr;
};

/* ------------------------------------------------------------------ */
/*  Globals                                                            */
/* ------------------------------------------------------------------ */

static struct flagcxP2pDevCtx flagcxP2pDevCtxs[MAX_IB_DEVS];
static int flagcxP2pInitialized = 0;
static pthread_mutex_t flagcxP2pInitLock = PTHREAD_MUTEX_INITIALIZER;

/* ------------------------------------------------------------------ */
/*  Request helpers                                                    */
/* ------------------------------------------------------------------ */

static flagcxResult_t flagcxP2pGetRequest(struct flagcxP2pRequest *reqs,
                                          struct ibv_cq *cq, int type,
                                          struct flagcxP2pRequest **req) {
  for (int i = 0; i < FLAGCX_P2P_MAX_REQUESTS; i++) {
    if (reqs[i].type == FLAGCX_P2P_REQ_UNUSED) {
      reqs[i].type = type;
      reqs[i].events = 0;
      reqs[i].cq = cq;
      reqs[i].reqs = reqs;
      *req = &reqs[i];
      return flagcxSuccess;
    }
  }
  WARN("NET/IB_P2P : unable to allocate request");
  *req = NULL;
  return flagcxInternalError;
}

static inline void flagcxP2pFreeRequest(struct flagcxP2pRequest *req) {
  req->type = FLAGCX_P2P_REQ_UNUSED;
}

/* ------------------------------------------------------------------ */
/*  Init / Devices / Properties                                        */
/* ------------------------------------------------------------------ */

static flagcxResult_t flagcxP2pInit() {
  pthread_mutex_lock(&flagcxP2pInitLock);
  if (flagcxP2pInitialized) {
    pthread_mutex_unlock(&flagcxP2pInitLock);
    return flagcxSuccess;
  }

  // Reuse IBRC device discovery (idempotent)
  FLAGCXCHECK(flagcxIbInit());

  // Eagerly allocate PD for each physical IB device
  for (int i = 0; i < flagcxNIbDevs; i++) {
    flagcxP2pDevCtxs[i].ibDevN = i;
    struct flagcxIbDev *ibDev = flagcxIbDevs + i;
    pthread_mutex_lock(&ibDev->lock);
    if (0 == ibDev->pdRefs++) {
      flagcxResult_t res;
      FLAGCXCHECKGOTO(flagcxWrapIbvAllocPd(&ibDev->pd, ibDev->context), res,
                      pd_fail);
      if (0) {
      pd_fail:
        ibDev->pdRefs--;
        pthread_mutex_unlock(&ibDev->lock);
        pthread_mutex_unlock(&flagcxP2pInitLock);
        return res;
      }
    }
    flagcxP2pDevCtxs[i].pd = ibDev->pd;
    pthread_mutex_unlock(&ibDev->lock);
  }

  flagcxP2pInitialized = 1;
  INFO(FLAGCX_INIT | FLAGCX_NET,
       "NET/IB_P2P : P2P adaptor initialized, %d devices, eager PD allocated",
       flagcxNIbDevs);
  pthread_mutex_unlock(&flagcxP2pInitLock);
  return flagcxSuccess;
}

static flagcxResult_t flagcxP2pDevices(int *ndev) {
  *ndev = flagcxNMergedIbDevs;
  return flagcxSuccess;
}

static flagcxResult_t flagcxP2pGetProperties(int dev, void *props) {
  return flagcxIbGetProperties(dev, props);
}

/* ------------------------------------------------------------------ */
/*  Memory Registration                                                */
/* ------------------------------------------------------------------ */

// Resolve ibDevN from a comm pointer. The comm may be:
//   - flagcxP2pDevCtx*  (from P2P engine, before any connection)
//   - flagcxP2pSendComm* or flagcxP2pRecvComm* (after connection)
// All have ibDevN as their first field.
static inline int flagcxP2pGetIbDevN(void *comm) { return *(int *)comm; }

static flagcxResult_t flagcxP2pRegMrDmaBuf(void *comm, void *data, size_t size,
                                           int type, uint64_t offset, int fd,
                                           int mrFlags, void **mhandle) {
  assert(size > 0);
  assert(comm != NULL);

  int ibDevN = flagcxP2pGetIbDevN(comm);
  struct flagcxIbDev *ibDev = flagcxIbDevs + ibDevN;

  // Build a temporary flagcxIbNetCommDevBase for the internal registration call
  struct flagcxIbNetCommDevBase devBase;
  memset(&devBase, 0, sizeof(devBase));
  devBase.ibDevN = ibDevN;
  devBase.pd = ibDev->pd;

  struct flagcxP2pMrHandle *handle =
      (struct flagcxP2pMrHandle *)malloc(sizeof(struct flagcxP2pMrHandle));
  if (!handle) {
    WARN("NET/IB_P2P : failed to allocate MR handle");
    return flagcxInternalError;
  }

  ibv_mr *mr = NULL;
  FLAGCXCHECK(flagcxIbRegMrDmaBufInternal(&devBase, data, size, type, offset,
                                          fd, mrFlags, &mr));

  handle->baseVa = (uintptr_t)data;
  handle->lkey = mr->lkey;
  handle->rkey = mr->rkey;
  handle->mr = mr;
  handle->ibDevN = ibDevN;

  *mhandle = (void *)handle;
  return flagcxSuccess;
}

static flagcxResult_t flagcxP2pRegMr(void *comm, void *data, size_t size,
                                     int type, int mrFlags, void **mhandle) {
  return flagcxP2pRegMrDmaBuf(comm, data, size, type, 0ULL, -1, mrFlags,
                              mhandle);
}

static flagcxResult_t flagcxP2pDeregMr(void *comm, void *mhandle) {
  struct flagcxP2pMrHandle *handle = (struct flagcxP2pMrHandle *)mhandle;

  // Build a temporary devBase for the internal deregistration call
  struct flagcxIbNetCommDevBase devBase;
  memset(&devBase, 0, sizeof(devBase));
  devBase.ibDevN = handle->ibDevN;
  devBase.pd = flagcxIbDevs[handle->ibDevN].pd;

  FLAGCXCHECK(flagcxIbDeregMrInternal(&devBase, handle->mr));
  free(handle);
  return flagcxSuccess;
}

/* ------------------------------------------------------------------ */
/*  Listen / Connect / Accept                                          */
/* ------------------------------------------------------------------ */

static flagcxResult_t flagcxP2pListen(int dev, void *opaqueHandle,
                                      void **listenComm) {
  struct flagcxP2pListenComm *comm;
  FLAGCXCHECK(flagcxCalloc(&comm, 1));
  struct flagcxP2pListenHandle *handle =
      (struct flagcxP2pListenHandle *)opaqueHandle;
  memset(handle, 0, sizeof(struct flagcxP2pListenHandle));
  comm->dev = dev;
  handle->magic = FLAGCX_SOCKET_MAGIC;
  FLAGCXCHECK(flagcxSocketInit(&comm->sock, &flagcxIbIfAddr, handle->magic,
                               flagcxSocketTypeNetIb, NULL, 1));
  FLAGCXCHECK(flagcxSocketListen(&comm->sock));
  FLAGCXCHECK(flagcxSocketGetAddr(&comm->sock, &handle->connectAddr));
  *listenComm = comm;
  return flagcxSuccess;
}

// Helper: set up PD (from eager init), CQ, QP, and GID for a connection
static flagcxResult_t flagcxP2pSetupConn(int dev,
                                         struct flagcxIbNetCommDevBase *base,
                                         struct flagcxIbQp *qp,
                                         int *outIbDevN) {
  struct flagcxIbMergedDev *mergedDev = flagcxIbMergedDevs + dev;
  int ibDevN = mergedDev->devs[0]; // v1: single physical NIC
  *outIbDevN = ibDevN;

  struct flagcxIbDev *ibDev = flagcxIbDevs + ibDevN;
  base->ibDevN = ibDevN;

  // Reuse PD from eager init, increment refcount
  pthread_mutex_lock(&ibDev->lock);
  ibDev->pdRefs++;
  base->pd = ibDev->pd;
  pthread_mutex_unlock(&ibDev->lock);

  // Create CQ for this connection
  FLAGCXCHECK(flagcxWrapIbvCreateCq(
      &base->cq, ibDev->context, 2 * FLAGCX_P2P_MAX_REQUESTS, NULL, NULL, 0));

  // Get GID info
  FLAGCXCHECK(flagcxIbGetGidIndex(ibDev->context, ibDev->portNum,
                                  ibDev->portAttr.gid_tbl_len,
                                  &base->gidInfo.localGidIndex));
  FLAGCXCHECK(flagcxWrapIbvQueryGid(ibDev->context, ibDev->portNum,
                                    base->gidInfo.localGidIndex,
                                    &base->gidInfo.localGid));
  base->gidInfo.linkLayer = ibDev->link;

  // Create RC QP with remote write, read, and atomic access
  int accessFlags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                    IBV_ACCESS_REMOTE_ATOMIC;
  FLAGCXCHECK(flagcxIbCreateQp(ibDev->portNum, base, accessFlags, qp));
  qp->devIndex = 0;

  return flagcxSuccess;
}

// Helper: build local connection metadata
static void flagcxP2pBuildConnMeta(struct flagcxP2pConnMeta *meta,
                                   struct flagcxIbNetCommDevBase *base,
                                   struct flagcxIbQp *qp, int ibDevN) {
  struct flagcxIbDev *ibDev = flagcxIbDevs + ibDevN;
  memset(meta, 0, sizeof(*meta));
  meta->qpn = qp->qp->qp_num;
  meta->gid = base->gidInfo.localGid;
  meta->ibPort = ibDev->portNum;
  meta->linkLayer = ibDev->link;
  meta->lid = ibDev->portAttr.lid;
  meta->mtu = ibDev->portAttr.active_mtu;
}

// Helper: transition QP to RTR+RTS using remote metadata
static flagcxResult_t
flagcxP2pTransitionQp(struct flagcxIbQp *qp,
                      struct flagcxIbNetCommDevBase *base,
                      struct flagcxP2pConnMeta *remoteMeta, int ibDevN) {
  struct flagcxIbDev *ibDev = flagcxIbDevs + ibDevN;

  // Clamp MTU to min(remote, local) — same as IBRC accept path
  enum ibv_mtu mtu = (enum ibv_mtu)std::min((int)remoteMeta->mtu,
                                            (int)ibDev->portAttr.active_mtu);

  struct flagcxIbDevInfo remoteInfo;
  memset(&remoteInfo, 0, sizeof(remoteInfo));
  remoteInfo.lid = remoteMeta->lid;
  remoteInfo.ibPort = remoteMeta->ibPort;
  remoteInfo.linkLayer = remoteMeta->linkLayer;
  remoteInfo.mtu = mtu;
  remoteInfo.spn = remoteMeta->gid.global.subnet_prefix;
  remoteInfo.iid = remoteMeta->gid.global.interface_id;

  FLAGCXCHECK(flagcxIbRtrQp(qp->qp, base->gidInfo.localGidIndex,
                            remoteMeta->qpn, &remoteInfo));
  FLAGCXCHECK(flagcxIbRtsQp(qp->qp));
  return flagcxSuccess;
}

static flagcxResult_t flagcxP2pConnect(int dev, void *opaqueHandle,
                                       void **sendComm) {
  struct flagcxP2pListenHandle *handle =
      (struct flagcxP2pListenHandle *)opaqueHandle;
  *sendComm = NULL;

  // Allocate send comm
  struct flagcxP2pSendComm *comm;
  FLAGCXCHECK(flagcxCalloc(&comm, 1));

  // TCP connect (blocking with timeout)
  FLAGCXCHECK(flagcxSocketInit(&comm->sock, &handle->connectAddr, handle->magic,
                               flagcxSocketTypeNetIb, NULL, 1));
  FLAGCXCHECK(flagcxSocketConnect(&comm->sock));
  int ready = 0;
  auto connectStart = std::chrono::steady_clock::now();
  while (!ready) {
    FLAGCXCHECK(flagcxSocketReady(&comm->sock, &ready));
    if (!ready) {
      if (std::chrono::steady_clock::now() - connectStart >
          std::chrono::seconds(30)) {
        WARN("NET/IB_P2P : connect socket ready timed out after 30s");
        free(comm);
        return flagcxSystemError;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  // Set up PD, CQ, QP
  FLAGCXCHECK(flagcxP2pSetupConn(dev, &comm->base, &comm->qp, &comm->ibDevN));

  // Exchange connection metadata
  struct flagcxP2pConnMeta localMeta, remoteMeta;
  flagcxP2pBuildConnMeta(&localMeta, &comm->base, &comm->qp, comm->ibDevN);
  FLAGCXCHECK(flagcxSocketSend(&comm->sock, &localMeta, sizeof(localMeta)));
  FLAGCXCHECK(flagcxSocketRecv(&comm->sock, &remoteMeta, sizeof(remoteMeta)));

  // Transition QP to RTR then RTS
  FLAGCXCHECK(
      flagcxP2pTransitionQp(&comm->qp, &comm->base, &remoteMeta, comm->ibDevN));

  // Register putSignal scratchpad MR
  comm->putSignalScratchpad = 0;
  FLAGCXCHECK(flagcxWrapIbvRegMr(
      &comm->putSignalScratchpadMr, comm->base.pd, &comm->putSignalScratchpad,
      sizeof(comm->putSignalScratchpad),
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
          IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC));

  // Exchange ready
  int localReady = 1, remoteReady = 0;
  FLAGCXCHECK(flagcxSocketSend(&comm->sock, &localReady, sizeof(localReady)));
  FLAGCXCHECK(flagcxSocketRecv(&comm->sock, &remoteReady, sizeof(remoteReady)));

  *sendComm = comm;
  return flagcxSuccess;
}

static flagcxResult_t flagcxP2pAccept(void *listenComm, void **recvComm) {
  struct flagcxP2pListenComm *lComm = (struct flagcxP2pListenComm *)listenComm;
  *recvComm = NULL;

  // Allocate recv comm
  struct flagcxP2pRecvComm *comm;
  FLAGCXCHECK(flagcxCalloc(&comm, 1));

  // TCP accept (blocking with timeout)
  FLAGCXCHECK(flagcxSocketInit(&comm->sock));
  FLAGCXCHECK(flagcxSocketAccept(&comm->sock, &lComm->sock));
  int ready = 0;
  auto acceptStart = std::chrono::steady_clock::now();
  while (!ready) {
    FLAGCXCHECK(flagcxSocketReady(&comm->sock, &ready));
    if (!ready) {
      if (std::chrono::steady_clock::now() - acceptStart >
          std::chrono::seconds(30)) {
        WARN("NET/IB_P2P : accept socket ready timed out after 30s");
        free(comm);
        return flagcxSystemError;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  // Set up PD, CQ, QP
  FLAGCXCHECK(
      flagcxP2pSetupConn(lComm->dev, &comm->base, &comm->qp, &comm->ibDevN));

  // Exchange connection metadata (accept receives first, then sends)
  struct flagcxP2pConnMeta localMeta, remoteMeta;
  flagcxP2pBuildConnMeta(&localMeta, &comm->base, &comm->qp, comm->ibDevN);
  FLAGCXCHECK(flagcxSocketRecv(&comm->sock, &remoteMeta, sizeof(remoteMeta)));
  FLAGCXCHECK(flagcxSocketSend(&comm->sock, &localMeta, sizeof(localMeta)));

  // Transition QP to RTR then RTS
  FLAGCXCHECK(
      flagcxP2pTransitionQp(&comm->qp, &comm->base, &remoteMeta, comm->ibDevN));

  // Register putSignal scratchpad MR (symmetric with connect)
  comm->putSignalScratchpad = 0;
  FLAGCXCHECK(flagcxWrapIbvRegMr(
      &comm->putSignalScratchpadMr, comm->base.pd, &comm->putSignalScratchpad,
      sizeof(comm->putSignalScratchpad),
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
          IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC));

  // Exchange ready
  int localReady = 1, remoteReady = 0;
  FLAGCXCHECK(flagcxSocketRecv(&comm->sock, &remoteReady, sizeof(remoteReady)));
  FLAGCXCHECK(flagcxSocketSend(&comm->sock, &localReady, sizeof(localReady)));

  *recvComm = comm;
  return flagcxSuccess;
}

/* ------------------------------------------------------------------ */
/*  One-sided transfers: iput / iget / iputSignal                      */
/* ------------------------------------------------------------------ */

static flagcxResult_t flagcxP2pIput(void *sendComm, uint64_t srcOff,
                                    uint64_t dstOff, size_t size, int srcRank,
                                    int dstRank, void **srcHandles,
                                    void **dstHandles, void **request) {
  struct flagcxP2pSendComm *comm = (struct flagcxP2pSendComm *)sendComm;
  struct flagcxP2pMrHandle *src = (struct flagcxP2pMrHandle *)srcHandles;
  struct flagcxP2pMrHandle *dst = (struct flagcxP2pMrHandle *)dstHandles;

  struct flagcxP2pRequest *req;
  FLAGCXCHECK(flagcxP2pGetRequest(comm->reqs, comm->base.cq,
                                  FLAGCX_P2P_REQ_IPUT, &req));

  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = src->baseVa + srcOff;
  sge.length = (uint32_t)size;
  if ((size_t)sge.length != size) {
    WARN("NET/IB_P2P : iput size %zu exceeds 32-bit limit", size);
    flagcxP2pFreeRequest(req);
    return flagcxInternalError;
  }
  sge.lkey = src->lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr_id = req - comm->reqs;
  wr.wr.rdma.remote_addr = dst->baseVa + dstOff;
  wr.wr.rdma.rkey = dst->rkey;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  struct ibv_send_wr *bad_wr;
  FLAGCXCHECK(flagcxWrapIbvPostSend(comm->qp.qp, &wr, &bad_wr));
  req->events = 1;

  *request = req;
  return flagcxSuccess;
}

static flagcxResult_t flagcxP2pIget(void *sendComm, uint64_t srcOff,
                                    uint64_t dstOff, size_t size, int srcRank,
                                    int dstRank, void **srcHandles,
                                    void **dstHandles, void **request) {
  struct flagcxP2pSendComm *comm = (struct flagcxP2pSendComm *)sendComm;
  struct flagcxP2pMrHandle *src = (struct flagcxP2pMrHandle *)srcHandles;
  struct flagcxP2pMrHandle *dst = (struct flagcxP2pMrHandle *)dstHandles;

  struct flagcxP2pRequest *req;
  FLAGCXCHECK(flagcxP2pGetRequest(comm->reqs, comm->base.cq,
                                  FLAGCX_P2P_REQ_IGET, &req));

  struct ibv_sge sge;
  memset(&sge, 0, sizeof(sge));
  sge.addr = dst->baseVa + dstOff;
  sge.length = (uint32_t)size;
  if ((size_t)sge.length != size) {
    WARN("NET/IB_P2P : iget size %zu exceeds 32-bit limit", size);
    flagcxP2pFreeRequest(req);
    return flagcxInternalError;
  }
  sge.lkey = dst->lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.opcode = IBV_WR_RDMA_READ;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.wr_id = req - comm->reqs;
  wr.wr.rdma.remote_addr = src->baseVa + srcOff;
  wr.wr.rdma.rkey = src->rkey;
  wr.sg_list = &sge;
  wr.num_sge = 1;

  struct ibv_send_wr *bad_wr;
  FLAGCXCHECK(flagcxWrapIbvPostSend(comm->qp.qp, &wr, &bad_wr));
  req->events = 1;

  *request = req;
  return flagcxSuccess;
}

static flagcxResult_t
flagcxP2pIputSignal(void *sendComm, uint64_t srcOff, uint64_t dstOff,
                    size_t size, int srcRank, int dstRank, void **srcHandles,
                    void **dstHandles, uint64_t signalOff, void **signalHandles,
                    uint64_t signalValue, void **request) {
  struct flagcxP2pSendComm *comm = (struct flagcxP2pSendComm *)sendComm;
  struct flagcxP2pMrHandle *signalInfo =
      (struct flagcxP2pMrHandle *)signalHandles;

  struct flagcxP2pRequest *req;
  FLAGCXCHECK(flagcxP2pGetRequest(comm->reqs, comm->base.cq,
                                  FLAGCX_P2P_REQ_IPUT, &req));

  bool chainData = (size > 0 && srcHandles != NULL && dstHandles != NULL);

  struct ibv_sge sge[2];
  struct ibv_send_wr wr[2];
  memset(sge, 0, sizeof(sge));
  memset(wr, 0, sizeof(wr));

  // wr[0]: RDMA WRITE for data (unsignaled, chained to wr[1])
  if (chainData) {
    struct flagcxP2pMrHandle *src = (struct flagcxP2pMrHandle *)srcHandles;
    struct flagcxP2pMrHandle *dst = (struct flagcxP2pMrHandle *)dstHandles;

    sge[0].addr = src->baseVa + srcOff;
    sge[0].length = (uint32_t)size;
    if ((size_t)sge[0].length != size) {
      WARN("NET/IB_P2P : iputSignal size %zu exceeds 32-bit limit", size);
      flagcxP2pFreeRequest(req);
      return flagcxInternalError;
    }
    sge[0].lkey = src->lkey;

    wr[0].opcode = IBV_WR_RDMA_WRITE;
    wr[0].send_flags = 0; // unsignaled
    wr[0].wr.rdma.remote_addr = dst->baseVa + dstOff;
    wr[0].wr.rdma.rkey = dst->rkey;
    wr[0].sg_list = &sge[0];
    wr[0].num_sge = 1;
    wr[0].next = &wr[1]; // chain to atomic
  }

  // wr[1]: ATOMIC FETCH_AND_ADD for signal (signaled)
  sge[1].addr = (uintptr_t)&comm->putSignalScratchpad;
  sge[1].length = sizeof(comm->putSignalScratchpad);
  sge[1].lkey = comm->putSignalScratchpadMr->lkey;

  wr[1].opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
  wr[1].send_flags = IBV_SEND_SIGNALED;
  wr[1].wr_id = req - comm->reqs;
  wr[1].wr.atomic.remote_addr = signalInfo->baseVa + signalOff;
  wr[1].wr.atomic.rkey = signalInfo->rkey;
  wr[1].wr.atomic.compare_add = signalValue;
  wr[1].sg_list = &sge[1];
  wr[1].num_sge = 1;
  wr[1].next = NULL;

  struct ibv_send_wr *bad_wr;
  FLAGCXCHECK(
      flagcxWrapIbvPostSend(comm->qp.qp, chainData ? &wr[0] : &wr[1], &bad_wr));
  req->events = 1;

  *request = req;
  return flagcxSuccess;
}

/* ------------------------------------------------------------------ */
/*  Test                                                               */
/* ------------------------------------------------------------------ */

static flagcxResult_t flagcxP2pTest(void *request, int *done, int *sizes) {
  *done = 0;
  struct flagcxP2pRequest *req = (struct flagcxP2pRequest *)request;
  if (req == NULL || req->type == FLAGCX_P2P_REQ_UNUSED) {
    *done = 1;
    return flagcxSuccess;
  }

  int nCqe = 0;
  struct ibv_wc wc;
  FLAGCXCHECK(flagcxWrapIbvPollCq(req->cq, 1, &wc, &nCqe));

  if (nCqe == 0)
    return flagcxSuccess;

  if (wc.status != IBV_WC_SUCCESS) {
    WARN("NET/IB_P2P : CQ error: status=%d opcode=%d wr_id=%lu", wc.status,
         wc.opcode, wc.wr_id);
    return flagcxRemoteError;
  }

  // Map CQE back to the correct request via wr_id
  uint32_t reqIdx = wc.wr_id;
  if (reqIdx >= FLAGCX_P2P_MAX_REQUESTS) {
    WARN("NET/IB_P2P : invalid wr_id %u in CQE", reqIdx);
    return flagcxInternalError;
  }
  struct flagcxP2pRequest *completedReq = &req->reqs[reqIdx];

  completedReq->events--;
  if (completedReq->events == 0) {
    completedReq->type = FLAGCX_P2P_REQ_UNUSED;
  }

  // Check if the originally requested op is done
  if (req->type == FLAGCX_P2P_REQ_UNUSED) {
    *done = 1;
    if (sizes)
      *sizes = 0;
  }
  return flagcxSuccess;
}

/* ------------------------------------------------------------------ */
/*  Close                                                              */
/* ------------------------------------------------------------------ */

// Helper: decrement PD refcount, dealloc if last ref
static flagcxResult_t flagcxP2pReleasePd(int ibDevN) {
  struct flagcxIbDev *ibDev = flagcxIbDevs + ibDevN;
  pthread_mutex_lock(&ibDev->lock);
  if (0 == --ibDev->pdRefs) {
    flagcxResult_t res = flagcxWrapIbvDeallocPd(ibDev->pd);
    pthread_mutex_unlock(&ibDev->lock);
    if (res != flagcxSuccess) {
      INFO(FLAGCX_ALL,
           "NET/IB_P2P : Failed to deallocate PD (non-fatal, may have "
           "remaining resources)");
    }
    return flagcxSuccess;
  }
  pthread_mutex_unlock(&ibDev->lock);
  return flagcxSuccess;
}

// Helper: drain CQ before destroying resources
static void flagcxP2pDrainCq(struct ibv_cq *cq) {
  if (!cq)
    return;
  struct ibv_wc wcs[64];
  int nCqe = 0;
  for (int i = 0; i < 16; i++) {
    flagcxWrapIbvPollCq(cq, 64, wcs, &nCqe);
    if (nCqe == 0)
      break;
  }
}

static flagcxResult_t flagcxP2pCloseSend(void *sendComm) {
  struct flagcxP2pSendComm *comm = (struct flagcxP2pSendComm *)sendComm;
  if (comm) {
    flagcxP2pDrainCq(comm->base.cq);
    if (comm->qp.qp)
      FLAGCXCHECK(flagcxWrapIbvDestroyQp(comm->qp.qp));
    if (comm->putSignalScratchpadMr)
      FLAGCXCHECK(flagcxWrapIbvDeregMr(comm->putSignalScratchpadMr));
    if (comm->base.cq)
      FLAGCXCHECK(flagcxWrapIbvDestroyCq(comm->base.cq));
    FLAGCXCHECK(flagcxP2pReleasePd(comm->ibDevN));
    FLAGCXCHECK(flagcxSocketClose(&comm->sock));
    free(comm);
  }
  return flagcxSuccess;
}

static flagcxResult_t flagcxP2pCloseRecv(void *recvComm) {
  struct flagcxP2pRecvComm *comm = (struct flagcxP2pRecvComm *)recvComm;
  if (comm) {
    flagcxP2pDrainCq(comm->base.cq);
    if (comm->qp.qp)
      FLAGCXCHECK(flagcxWrapIbvDestroyQp(comm->qp.qp));
    if (comm->putSignalScratchpadMr)
      FLAGCXCHECK(flagcxWrapIbvDeregMr(comm->putSignalScratchpadMr));
    if (comm->base.cq)
      FLAGCXCHECK(flagcxWrapIbvDestroyCq(comm->base.cq));
    FLAGCXCHECK(flagcxP2pReleasePd(comm->ibDevN));
    FLAGCXCHECK(flagcxSocketClose(&comm->sock));
    free(comm);
  }
  return flagcxSuccess;
}

static flagcxResult_t flagcxP2pCloseListen(void *listenComm) {
  struct flagcxP2pListenComm *comm = (struct flagcxP2pListenComm *)listenComm;
  if (comm) {
    FLAGCXCHECK(flagcxSocketClose(&comm->sock));
    free(comm);
  }
  return flagcxSuccess;
}

/* ------------------------------------------------------------------ */
/*  Two-sided stubs (not supported by P2P adaptor)                     */
/* ------------------------------------------------------------------ */

static flagcxResult_t flagcxP2pIsend(void *, void *, size_t, int, void *,
                                     void *, void **) {
  WARN("NET/IB_P2P : isend not supported");
  return flagcxInternalError;
}

static flagcxResult_t flagcxP2pIrecv(void *, int, void **, size_t *, int *,
                                     void **, void **, void **) {
  WARN("NET/IB_P2P : irecv not supported");
  return flagcxInternalError;
}

static flagcxResult_t flagcxP2pIflush(void *, int, void **, int *, void **,
                                      void **) {
  WARN("NET/IB_P2P : iflush not supported");
  return flagcxInternalError;
}

/* ------------------------------------------------------------------ */
/*  Device name lookup                                                 */
/* ------------------------------------------------------------------ */

static flagcxResult_t flagcxP2pGetDevFromName(char *name, int *dev) {
  for (int i = 0; i < flagcxNMergedIbDevs; i++) {
    if (strcmp(flagcxIbMergedDevs[i].devName, name) == 0) {
      *dev = i;
      return flagcxSuccess;
    }
  }
  WARN("NET/IB_P2P : device %s not found", name);
  return flagcxInternalError;
}

/* ------------------------------------------------------------------ */
/*  Adaptor struct                                                     */
/* ------------------------------------------------------------------ */

struct flagcxNetAdaptor flagcxNetIbP2p = {
    // Basic functions
    "IB_P2P", flagcxP2pInit, flagcxP2pDevices, flagcxP2pGetProperties,

    // Setup functions
    flagcxP2pListen, flagcxP2pConnect, flagcxP2pAccept, flagcxP2pCloseSend,
    flagcxP2pCloseRecv, flagcxP2pCloseListen,

    // Memory region functions
    flagcxP2pRegMr, flagcxP2pRegMrDmaBuf, flagcxP2pDeregMr,

    // Two-sided functions (stubs)
    flagcxP2pIsend, flagcxP2pIrecv, flagcxP2pIflush, flagcxP2pTest,

    // One-sided functions
    flagcxP2pIput, flagcxP2pIget, flagcxP2pIputSignal,

    // Device name lookup
    flagcxP2pGetDevFromName};
