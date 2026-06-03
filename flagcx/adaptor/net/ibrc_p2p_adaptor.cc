/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * IBRC P2P Net Adaptor — implements flagcxNetAdaptor for one-sided RDMA
 * (P2P) use cases. Shares IB device discovery and utility code with the
 * existing IBRC adaptor but uses P2P-native handle formats, eager PD
 * allocation, and simplified (no-FIFO) connection setup.
 ************************************************************************/

#include "flagcx_common.h"
#include "flagcx_net_adaptor.h"
#include "flagcx_p2p.h"
#include "ib_common.h"
#include "ibvwrap.h"
#include "socket.h"

#include <algorithm>
#include <assert.h>
#include <atomic>
#include <chrono>
#include <mutex>
#include <pthread.h>
#include <stdint.h>
#include <string.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

extern struct ibv_cq *flagcxP2pPoolGetSharedCq(int ibDevN,
                                               struct ibv_context *ctx);
extern void flagcxP2pPoolRegisterQp(int ibDevN, void *sendComm,
                                    struct ibv_qp *qp);
extern void flagcxP2pPoolUnregisterQp(int ibDevN, struct ibv_qp *qp);
extern flagcxResult_t flagcxP2pPoolSubmit(int ibDevN, void *sendComm,
                                          FlagcxSlice **slices, int count);

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

struct flagcxP2pSliceReq {
  FlagcxTransferTask task;
  FlagcxSlice slice;
};

// Field order through `sock` mirrors core's FlagcxP2pCommView — do not reorder.
struct flagcxP2pSendComm {
  int ibDevN;
  struct flagcxIbNetCommDevBase base;
  struct flagcxIbQp qp_list_[kFlagcxP2pMaxQpsPerEngine];
  struct flagcxSocket sock;
  std::atomic<uint32_t> nextChannel{0};
  int numQps{
      0}; // resolved from flagcxP2pGlobalConfig().qpsPerConn at connect/accept
};

struct flagcxP2pRecvComm {
  int ibDevN;
  struct flagcxIbNetCommDevBase base;
  struct flagcxIbQp qp_list_[kFlagcxP2pMaxQpsPerEngine];
  struct flagcxSocket sock;
  std::atomic<uint32_t> nextChannel{0};
  int numQps{
      0}; // resolved from flagcxP2pGlobalConfig().qpsPerConn at connect/accept
};

/* ------------------------------------------------------------------ */
/*  Globals                                                            */
/* ------------------------------------------------------------------ */

static struct flagcxP2pDevCtx flagcxP2pDevCtxs[MAX_IB_DEVS];
static int flagcxP2pInitialized = 0;
static pthread_mutex_t flagcxP2pInitLock = PTHREAD_MUTEX_INITIALIZER;

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

static flagcxResult_t flagcxP2pReleasePd(int ibDevN);

// Helper: set up PD (from eager init), CQs, QPs, and GID for a connection
static flagcxResult_t flagcxP2pSetupConn(int dev, void *outerComm,
                                         struct flagcxIbNetCommDevBase *base,
                                         struct flagcxIbQp *qp_list,
                                         int *outIbDevN, int numQps) {
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

  // Step 0: pull the shared CQ from the per-ibDev WorkerPool. The pool is
  // lazily created on first call (and lives for the process lifetime).
  struct ibv_cq *sharedCq = flagcxP2pPoolGetSharedCq(ibDevN, ibDev->context);
  if (sharedCq == NULL) {
    WARN("NET/IB_P2P : pool[%d] returned NULL shared CQ", ibDevN);
    flagcxP2pReleasePd(ibDevN);
    base->pd = NULL;
    return flagcxInternalError;
  }
  base->cq = sharedCq;

  int accessFlags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                    IBV_ACCESS_REMOTE_ATOMIC;

  // Get GID info
  flagcxResult_t res;
  FLAGCXCHECKGOTO(flagcxIbGetGidIndex(ibDev->context, ibDev->portNum,
                                      ibDev->portAttr.gid_tbl_len,
                                      &base->gidInfo.localGidIndex),
                  res, setup_fail);
  FLAGCXCHECKGOTO(flagcxWrapIbvQueryGid(ibDev->context, ibDev->portNum,
                                        base->gidInfo.localGidIndex,
                                        &base->gidInfo.localGid),
                  res, setup_fail);
  base->gidInfo.linkLayer = ibDev->link;

  for (int i = 0; i < numQps; i++) {
    FLAGCXCHECKGOTO(
        flagcxIbCreateQp(ibDev->portNum, base, accessFlags, &qp_list[i]), res,
        setup_fail);
    qp_list[i].devIndex = 0;
    flagcxP2pPoolRegisterQp(ibDevN, outerComm, qp_list[i].qp);
  }

  return flagcxSuccess;

setup_fail:
  for (int i = 0; i < numQps; i++) {
    if (qp_list[i].qp) {
      flagcxP2pPoolUnregisterQp(ibDevN, qp_list[i].qp);
      flagcxWrapIbvDestroyQp(qp_list[i].qp);
      qp_list[i].qp = NULL;
    }
  }
  // Do not destroy sharedCq — owned by the pool.
  base->cq = NULL;
  flagcxP2pReleasePd(ibDevN);
  base->pd = NULL;
  return res;
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

static flagcxResult_t
flagcxP2pDestroyQps(int ibDevN, struct flagcxIbQp *qp_list, int numQps) {
  for (int i = 0; i < numQps; i++) {
    if (qp_list[i].qp) {
      flagcxP2pPoolUnregisterQp(ibDevN, qp_list[i].qp);
      FLAGCXCHECK(flagcxWrapIbvDestroyQp(qp_list[i].qp));
      qp_list[i].qp = NULL;
    }
  }
  return flagcxSuccess;
}

static inline struct flagcxIbQp *
flagcxP2pNextQp(struct flagcxIbQp *qp_list, std::atomic<uint32_t> *nextChannel,
                int qpCount) {
  uint32_t mod = (qpCount > 0) ? (uint32_t)qpCount : 1u;
  uint32_t idx = nextChannel->fetch_add(1, std::memory_order_relaxed);
  return qp_list + (idx % mod);
}

static flagcxResult_t flagcxP2pConnect(int dev, void *opaqueHandle,
                                       void **sendComm) {
  struct flagcxP2pListenHandle *handle =
      (struct flagcxP2pListenHandle *)opaqueHandle;
  flagcxResult_t res;
  *sendComm = NULL;

  // Allocate send comm
  struct flagcxP2pSendComm *comm;
  FLAGCXCHECK(flagcxCalloc(&comm, 1));
  int ready = 0;
  auto connectStart = std::chrono::steady_clock::time_point();
  struct flagcxP2pConnMeta localMeta[kFlagcxP2pMaxQpsPerEngine];
  struct flagcxP2pConnMeta remoteMeta[kFlagcxP2pMaxQpsPerEngine];
  int localReady = 1, remoteReady = 0;
  uint32_t localNumQps = 0, remoteNumQps = 0, agreedNumQps = 0;

  // TCP connect (blocking with timeout)
  FLAGCXCHECKGOTO(flagcxSocketInit(&comm->sock, &handle->connectAddr,
                                   handle->magic, flagcxSocketTypeNetIb, NULL,
                                   1),
                  res, connect_fail);
  FLAGCXCHECKGOTO(flagcxSocketConnect(&comm->sock), res, connect_fail);
  connectStart = std::chrono::steady_clock::now();
  while (!ready) {
    FLAGCXCHECKGOTO(flagcxSocketReady(&comm->sock, &ready), res, connect_fail);
    if (!ready) {
      if (std::chrono::steady_clock::now() - connectStart >
          std::chrono::seconds(30)) {
        WARN("NET/IB_P2P : connect socket ready timed out after 30s");
        res = flagcxSystemError;
        goto connect_fail;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  // numQps negotiation must happen before setup so we only create the
  // QPs we'll actually use; both peers agree on min().
  localNumQps = (uint32_t)flagcxP2pGlobalConfig().qpsPerConn;
  if (localNumQps == 0 || localNumQps > (uint32_t)kFlagcxP2pMaxQpsPerEngine)
    localNumQps = (uint32_t)kFlagcxP2pMaxQpsPerEngine;
  FLAGCXCHECKGOTO(
      flagcxSocketSend(&comm->sock, &localNumQps, sizeof(localNumQps)), res,
      connect_fail);
  FLAGCXCHECKGOTO(
      flagcxSocketRecv(&comm->sock, &remoteNumQps, sizeof(remoteNumQps)), res,
      connect_fail);
  if (remoteNumQps == 0 || remoteNumQps > (uint32_t)kFlagcxP2pMaxQpsPerEngine) {
    WARN("NET/IB_P2P : peer advertised invalid numQps=%u (max=%d)",
         remoteNumQps, kFlagcxP2pMaxQpsPerEngine);
    res = flagcxInternalError;
    goto connect_fail;
  }
  agreedNumQps = std::min(localNumQps, remoteNumQps);
  if (localNumQps != remoteNumQps) {
    INFO(FLAGCX_NET,
         "NET/IB_P2P : numQps mismatch (local=%u remote=%u) — using min=%u",
         localNumQps, remoteNumQps, agreedNumQps);
  }
  comm->numQps = (int)agreedNumQps;

  FLAGCXCHECKGOTO(flagcxP2pSetupConn(dev, comm, &comm->base, comm->qp_list_,
                                     &comm->ibDevN, comm->numQps),
                  res, connect_fail);

  for (int i = 0; i < comm->numQps; i++)
    flagcxP2pBuildConnMeta(&localMeta[i], &comm->base, &comm->qp_list_[i],
                           comm->ibDevN);
  FLAGCXCHECKGOTO(flagcxSocketSend(&comm->sock, localMeta,
                                   comm->numQps * sizeof(localMeta[0])),
                  res, connect_fail);
  FLAGCXCHECKGOTO(flagcxSocketRecv(&comm->sock, remoteMeta,
                                   comm->numQps * sizeof(remoteMeta[0])),
                  res, connect_fail);

  // Transition each matched QP to RTR then RTS.
  for (int i = 0; i < comm->numQps; i++)
    FLAGCXCHECKGOTO(flagcxP2pTransitionQp(&comm->qp_list_[i], &comm->base,
                                          &remoteMeta[i], comm->ibDevN),
                    res, connect_fail);

  // Exchange ready
  FLAGCXCHECKGOTO(
      flagcxSocketSend(&comm->sock, &localReady, sizeof(localReady)), res,
      connect_fail);
  FLAGCXCHECKGOTO(
      flagcxSocketRecv(&comm->sock, &remoteReady, sizeof(remoteReady)), res,
      connect_fail);

  *sendComm = comm;
  return flagcxSuccess;

connect_fail:
  flagcxP2pDestroyQps(comm->ibDevN, comm->qp_list_, comm->numQps);
  if (comm->base.pd)
    flagcxP2pReleasePd(comm->ibDevN);
  flagcxSocketClose(&comm->sock);
  free(comm);
  return res;
}

static flagcxResult_t flagcxP2pAccept(void *listenComm, void **recvComm) {
  struct flagcxP2pListenComm *lComm = (struct flagcxP2pListenComm *)listenComm;
  *recvComm = NULL;

  // Allocate recv comm
  struct flagcxP2pRecvComm *comm;
  FLAGCXCHECK(flagcxCalloc(&comm, 1));

  // TCP accept (blocking, no timeout)
  flagcxResult_t res;
  int ready;
  struct flagcxP2pConnMeta localMeta[kFlagcxP2pMaxQpsPerEngine];
  struct flagcxP2pConnMeta remoteMeta[kFlagcxP2pMaxQpsPerEngine];
  int localReady = 1, remoteReady = 0;
  uint32_t localNumQps = 0, remoteNumQps = 0, agreedNumQps = 0;
  FLAGCXCHECKGOTO(flagcxSocketInit(&comm->sock), res, accept_fail);
  FLAGCXCHECKGOTO(flagcxSocketAccept(&comm->sock, &lComm->sock), res,
                  accept_fail);
  ready = 0;
  while (!ready) {
    FLAGCXCHECKGOTO(flagcxSocketReady(&comm->sock, &ready), res, accept_fail);
    if (!ready) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  if (0) {
  accept_fail:
    free(comm);
    return res;
  }

  // accept side mirrors connect: recv numQps first, then send.
  FLAGCXCHECKGOTO(
      flagcxSocketRecv(&comm->sock, &remoteNumQps, sizeof(remoteNumQps)), res,
      accept_cleanup);
  localNumQps = (uint32_t)flagcxP2pGlobalConfig().qpsPerConn;
  if (localNumQps == 0 || localNumQps > (uint32_t)kFlagcxP2pMaxQpsPerEngine)
    localNumQps = (uint32_t)kFlagcxP2pMaxQpsPerEngine;
  FLAGCXCHECKGOTO(
      flagcxSocketSend(&comm->sock, &localNumQps, sizeof(localNumQps)), res,
      accept_cleanup);
  if (remoteNumQps == 0 || remoteNumQps > (uint32_t)kFlagcxP2pMaxQpsPerEngine) {
    WARN("NET/IB_P2P : peer advertised invalid numQps=%u (max=%d)",
         remoteNumQps, kFlagcxP2pMaxQpsPerEngine);
    res = flagcxInternalError;
    goto accept_cleanup;
  }
  agreedNumQps = std::min(localNumQps, remoteNumQps);
  if (localNumQps != remoteNumQps) {
    INFO(FLAGCX_NET,
         "NET/IB_P2P : numQps mismatch (local=%u remote=%u) — using min=%u",
         localNumQps, remoteNumQps, agreedNumQps);
  }
  comm->numQps = (int)agreedNumQps;

  FLAGCXCHECKGOTO(flagcxP2pSetupConn(lComm->dev, comm, &comm->base,
                                     comm->qp_list_, &comm->ibDevN,
                                     comm->numQps),
                  res, accept_cleanup);

  for (int i = 0; i < comm->numQps; i++)
    flagcxP2pBuildConnMeta(&localMeta[i], &comm->base, &comm->qp_list_[i],
                           comm->ibDevN);
  FLAGCXCHECKGOTO(flagcxSocketRecv(&comm->sock, remoteMeta,
                                   comm->numQps * sizeof(remoteMeta[0])),
                  res, accept_cleanup);
  FLAGCXCHECKGOTO(flagcxSocketSend(&comm->sock, localMeta,
                                   comm->numQps * sizeof(localMeta[0])),
                  res, accept_cleanup);

  // Transition each matched QP to RTR then RTS.
  for (int i = 0; i < comm->numQps; i++)
    FLAGCXCHECKGOTO(flagcxP2pTransitionQp(&comm->qp_list_[i], &comm->base,
                                          &remoteMeta[i], comm->ibDevN),
                    res, accept_cleanup);

  // Exchange ready
  FLAGCXCHECKGOTO(
      flagcxSocketRecv(&comm->sock, &remoteReady, sizeof(remoteReady)), res,
      accept_cleanup);
  FLAGCXCHECKGOTO(
      flagcxSocketSend(&comm->sock, &localReady, sizeof(localReady)), res,
      accept_cleanup);

  *recvComm = comm;
  return flagcxSuccess;

accept_cleanup:
  flagcxP2pDestroyQps(comm->ibDevN, comm->qp_list_, comm->numQps);
  if (comm->base.pd)
    flagcxP2pReleasePd(comm->ibDevN);
  flagcxSocketClose(&comm->sock);
  free(comm);
  return res;
}

/* ------------------------------------------------------------------ */
/*  One-sided transfers: iput / iget / iputSignal                      */
/* ------------------------------------------------------------------ */

// Slice request ownership model:
//   Allocation:  iput/iget/igetBatch allocate a flagcxP2pSliceReq (and, for
//                batch paths, additional FlagcxSlice objects).
//   Submission:  The req's slices are submitted to the worker pool via
//                flagcxP2pPoolSubmit(). The pool posts WRs and marks slices
//                done (markSuccess/markFailed) when CQEs arrive.
//   Polling:     The caller polls via test() or testBatch(). Once
//                task.isAllDone() returns true, the request is complete.
//   Deallocation: test()/testBatch() call flagcxP2pFreeSliceReq() which
//                deletes any heap-allocated slices and the req itself.

static flagcxResult_t
flagcxP2pBuildSingleSliceReq(struct flagcxP2pSendComm *comm, uint64_t localVa,
                             uint64_t remoteVa, size_t size, uint32_t lkey,
                             uint32_t rkey, uint8_t opcode, void **request) {
  if ((uint32_t)size != size) {
    WARN("NET/IB_P2P : single-op size %zu exceeds 32-bit limit", size);
    return flagcxInternalError;
  }

  auto *req = new struct flagcxP2pSliceReq;
  req->slice.srcVa = localVa;
  req->slice.dstVa = remoteVa;
  req->slice.length = (uint32_t)size;
  req->slice.lkey = lkey;
  req->slice.rkey = rkey;
  req->slice.opcode = opcode;
  req->slice.peerNicPath = std::string();
  req->slice.task = &req->task;
  req->slice.qpDepth = NULL;
  req->task.sliceList.push_back(&req->slice);
  req->task.sliceCount.fetch_add(1, std::memory_order_release);

  FlagcxSlice *slicePtr = &req->slice;
  flagcxResult_t rc = flagcxP2pPoolSubmit(comm->ibDevN, comm, &slicePtr, 1);
  if (rc != flagcxSuccess) {
    delete req;
    return rc;
  }

  *request = req;
  return flagcxSuccess;
}

static flagcxResult_t flagcxP2pIput(void *sendComm, uint64_t srcOff,
                                    uint64_t dstOff, size_t size, int srcRank,
                                    int dstRank, void **srcHandles,
                                    void **dstHandles, void **request) {
  (void)srcRank;
  (void)dstRank;
  struct flagcxP2pSendComm *comm = (struct flagcxP2pSendComm *)sendComm;
  struct flagcxP2pMrHandle *src = (struct flagcxP2pMrHandle *)srcHandles;
  struct flagcxP2pMrHandle *dst = (struct flagcxP2pMrHandle *)dstHandles;
  return flagcxP2pBuildSingleSliceReq(
      comm, src->baseVa + srcOff, dst->baseVa + dstOff, size, src->lkey,
      dst->rkey, FLAGCX_SLICE_OP_WRITE, request);
}

static flagcxResult_t flagcxP2pIget(void *sendComm, uint64_t srcOff,
                                    uint64_t dstOff, size_t size, int srcRank,
                                    int dstRank, void **srcHandles,
                                    void **dstHandles, void **request) {
  (void)srcRank;
  (void)dstRank;
  struct flagcxP2pSendComm *comm = (struct flagcxP2pSendComm *)sendComm;
  struct flagcxP2pMrHandle *src = (struct flagcxP2pMrHandle *)srcHandles;
  struct flagcxP2pMrHandle *dst = (struct flagcxP2pMrHandle *)dstHandles;
  return flagcxP2pBuildSingleSliceReq(comm, dst->baseVa + dstOff,
                                      src->baseVa + srcOff, size, dst->lkey,
                                      src->rkey, FLAGCX_SLICE_OP_READ, request);
}

static flagcxResult_t
flagcxP2pIgetBatch(void *sendComm, int count, const uint64_t *srcOffs,
                   const uint64_t *dstOffs, const size_t *sizes, int srcRank,
                   int dstRank, void *const *srcHandles,
                   void *const *dstHandles, void **request) {
  (void)srcRank;
  (void)dstRank;
  struct flagcxP2pSendComm *comm = (struct flagcxP2pSendComm *)sendComm;
  const int maxWrPerPost = (int)flagcxP2pGlobalConfig().maxWrPerPost;
  if (count <= 0 || count > maxWrPerPost || srcOffs == NULL ||
      dstOffs == NULL || sizes == NULL || srcHandles == NULL ||
      dstHandles == NULL || request == NULL) {
    WARN("NET/IB_P2P : invalid igetBatch arguments, count %d (max %d)", count,
         maxWrPerPost);
    return flagcxInternalError;
  }

  auto *req = new struct flagcxP2pSliceReq;
  req->task.sliceList.reserve(count);
  for (int i = 0; i < count; i++) {
    if (srcHandles[i] == NULL || dstHandles[i] == NULL ||
        (uint32_t)sizes[i] != sizes[i]) {
      WARN("NET/IB_P2P : igetBatch slice %d invalid", i);
      for (auto *s : req->task.sliceList)
        delete s;
      delete req;
      return flagcxInternalError;
    }
    auto *src = (struct flagcxP2pMrHandle *)srcHandles[i];
    auto *dst = (struct flagcxP2pMrHandle *)dstHandles[i];
    auto *s = new FlagcxSlice{dst->baseVa + dstOffs[i],
                              src->baseVa + srcOffs[i],
                              (uint32_t)sizes[i],
                              dst->lkey,
                              src->rkey,
                              FLAGCX_SLICE_OP_READ,
                              std::string(),
                              &req->task,
                              NULL};
    req->task.sliceList.push_back(s);
    req->task.sliceCount.fetch_add(1, std::memory_order_release);
  }

  flagcxResult_t rc = flagcxP2pPoolSubmit(comm->ibDevN, comm,
                                          req->task.sliceList.data(), count);
  if (rc != flagcxSuccess) {
    for (auto *s : req->task.sliceList)
      delete s;
    delete req;
    return rc;
  }
  *request = req;
  return flagcxSuccess;
}

static flagcxResult_t flagcxP2pIputSignal(void *, uint64_t, uint64_t, size_t,
                                          int, int, void **, void **, uint64_t,
                                          void **, uint64_t, void **) {
  WARN("NET/IB_P2P : iputSignal not supported");
  return flagcxInternalError;
}

/* ------------------------------------------------------------------ */
/*  Slice batch: pool worker passes the chosen QP. wr_id = ptr|1.      */
/* ------------------------------------------------------------------ */

static inline enum ibv_wr_opcode flagcxSliceOpcodeToVerbs(uint8_t op) {
  return op == FLAGCX_SLICE_OP_READ ? IBV_WR_RDMA_READ : IBV_WR_RDMA_WRITE;
}

extern "C" flagcxResult_t flagcxP2pSliceBatch(void *sendComm, struct ibv_qp *qp,
                                              int count, FlagcxSlice **slices) {
  struct flagcxP2pSendComm *comm = (struct flagcxP2pSendComm *)sendComm;
  const char *opLabel = (slices != NULL && count > 0 && slices[0] != NULL &&
                         slices[0]->opcode == FLAGCX_SLICE_OP_READ)
                            ? "READ"
                            : "WRITE";
  const int maxWrPerPost = (int)flagcxP2pGlobalConfig().maxWrPerPost;
  if (count <= 0 || count > maxWrPerPost || slices == NULL || qp == NULL ||
      comm == NULL) {
    WARN("NET/IB_P2P : invalid sliceBatch arguments (op=%s, count=%d, qp=%p, "
         "max=%d)",
         opLabel, count, (void *)qp, maxWrPerPost);
    return flagcxInternalError;
  }

  // count can be up to flagcxP2pGlobalConfig().maxWrPerPost (default 256,
  // bounded at 1024). Heap-allocate to keep the stack small.
  std::vector<struct ibv_send_wr> wrs(count);
  std::vector<struct ibv_sge> sges(count);

  for (int i = 0; i < count; i++) {
    FlagcxSlice *s = slices[i];
    if (s == NULL) {
      WARN("NET/IB_P2P : sliceBatch slice[%d] is NULL", i);
      for (int k = 0; k < i; k++)
        slices[k]->markFailed();
      for (int k = i; k < count; k++)
        if (slices[k])
          slices[k]->markFailed();
      return flagcxInternalError;
    }

    sges[i].addr = s->srcVa;
    sges[i].length = s->length;
    sges[i].lkey = s->lkey;

    wrs[i].opcode = flagcxSliceOpcodeToVerbs(s->opcode);
    wrs[i].send_flags = IBV_SEND_SIGNALED;
    wrs[i].wr_id = ((uintptr_t)s) | 1ull;
    wrs[i].wr.rdma.remote_addr = s->dstVa;
    wrs[i].wr.rdma.rkey = s->rkey;
    wrs[i].sg_list = &sges[i];
    wrs[i].num_sge = 1;
    wrs[i].next = (i + 1 < count) ? &wrs[i + 1] : NULL;
  }

  struct ibv_send_wr *bad_wr = NULL;
  flagcxResult_t res = flagcxWrapIbvPostSend(qp, &wrs[0], &bad_wr);
  if (res != flagcxSuccess) {
    int failedFrom = 0;
    if (bad_wr != NULL) {
      ptrdiff_t off = bad_wr - &wrs[0];
      if (off >= 0 && off < count)
        failedFrom = (int)off;
    }
    // Slices in [failedFrom..count) never went on the wire — roll back
    // their share of the pool's qpDepth pre-bump so the gate doesn't leak.
    for (int k = failedFrom; k < count; k++) {
      if (slices[k]->qpDepth != NULL)
        __sync_fetch_and_sub(slices[k]->qpDepth, 1);
      slices[k]->markFailed();
    }
    WARN("NET/IB_P2P : sliceBatch ibv_post_send failed (op=%s, count=%d, "
         "failedFrom=%d)",
         opLabel, count, failedFrom);
    return res;
  }

  return flagcxSuccess;
}

/* ------------------------------------------------------------------ */
/*  Test                                                               */
/* ------------------------------------------------------------------ */

// Single-slice path uses the wrapper's embedded `slice`; batch path
// heap-allocates each — distinguish by address.
static inline void flagcxP2pFreeSliceReq(struct flagcxP2pSliceReq *req) {
  if (!req)
    return;
  for (auto *s : req->task.sliceList) {
    if (s != &req->slice)
      delete s;
  }
  delete req;
}

static flagcxResult_t flagcxP2pTest(void *request, int *done, int *sizes) {
  *done = 0;
  if (sizes)
    *sizes = 0;
  if (request == NULL) {
    *done = 1;
    return flagcxSuccess;
  }
  auto *req = static_cast<struct flagcxP2pSliceReq *>(request);
  if (req->task.isAllDone()) {
    *done = 1;
    bool failed = req->task.hasErrors();
    if (sizes && !failed) {
      uint64_t total = 0;
      for (auto *s : req->task.sliceList)
        total += s->length;
      *sizes = (int)std::min<uint64_t>(total, (uint64_t)INT32_MAX);
    }
    flagcxP2pFreeSliceReq(req);
    if (failed)
      return flagcxInternalError;
  }
  return flagcxSuccess;
}

static flagcxResult_t flagcxP2pTestBatch(void **requests, int nRequests,
                                         int *doneFlags, int *doneCount) {
  int completed = 0;
  bool anyFailed = false;
  for (int i = 0; i < nRequests; i++) {
    doneFlags[i] = 0;
    auto *req = static_cast<struct flagcxP2pSliceReq *>(requests[i]);
    if (req == NULL) {
      doneFlags[i] = 1;
      completed++;
      continue;
    }
    if (req->task.isAllDone()) {
      doneFlags[i] = 1;
      completed++;
      if (req->task.hasErrors())
        anyFailed = true;
      flagcxP2pFreeSliceReq(req);
      requests[i] = NULL;
    }
  }
  *doneCount = completed;
  return anyFailed ? flagcxInternalError : flagcxSuccess;
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

static flagcxResult_t flagcxP2pCloseSend(void *sendComm) {
  struct flagcxP2pSendComm *comm = (struct flagcxP2pSendComm *)sendComm;
  if (comm) {
    FLAGCXCHECK(
        flagcxP2pDestroyQps(comm->ibDevN, comm->qp_list_, comm->numQps));
    FLAGCXCHECK(flagcxP2pReleasePd(comm->ibDevN));
    FLAGCXCHECK(flagcxSocketClose(&comm->sock));
    free(comm);
  }
  return flagcxSuccess;
}

static flagcxResult_t flagcxP2pCloseRecv(void *recvComm) {
  struct flagcxP2pRecvComm *comm = (struct flagcxP2pRecvComm *)recvComm;
  if (comm) {
    FLAGCXCHECK(
        flagcxP2pDestroyQps(comm->ibDevN, comm->qp_list_, comm->numQps));
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
    flagcxP2pGetDevFromName,

    // Optional batch operations
    nullptr,            // iputBatch
    flagcxP2pTestBatch, // testBatch
    flagcxP2pIgetBatch, // igetBatch
};
