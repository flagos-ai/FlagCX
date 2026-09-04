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
#include "onesided.h"
#include "p2p_transport.h"
#include "socket.h"

#include <algorithm>
#include <assert.h>
#include <atomic>
#include <chrono>
#include <mutex>
#include <new>
#include <pthread.h>
#include <stdint.h>
#include <string.h>
#include <thread>
#include <unistd.h>
#include <vector>

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
  volatile uint32_t abortFlag;
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

struct flagcxP2pComm;

struct flagcxP2pSliceReq {
  FlagcxTransferTask task;
  FlagcxSlice slice;
  struct flagcxP2pComm *comm = NULL;
};

// The socket remains part of the adaptor-private connection state.
struct flagcxP2pComm {
  int ibDevN{0};
  struct flagcxIbNetCommDevBase base {};
  struct flagcxIbQp qp_list_[kFlagcxP2pMaxQpsPerEngine] {};
  struct flagcxSocket sock {};
  std::atomic<uint32_t> nextChannel{0};
  volatile int qpDepth[kFlagcxP2pMaxQpsPerEngine]{};
  int qpDepthLimit{0};
  int numQps{
      0}; // resolved from flagcxP2pGlobalConfig().qpsPerConn at connect/accept
};

static flagcxResult_t flagcxP2pSliceBatch(void *sendComm, struct ibv_qp *qp,
                                          int count, FlagcxSlice **slices,
                                          int *failedCount);

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
//   - flagcxP2pComm*  (after connection)
// All have ibDevN as their first field.
static inline int flagcxP2pGetIbDevN(void *comm) { return *(int *)comm; }

static flagcxResult_t flagcxP2pRegMrDmaBuf(void *comm, void *data, size_t size,
                                           int type, uint64_t offset, int fd,
                                           int mrFlags, void **mhandle) {
  if (mhandle == NULL)
    return flagcxInvalidArgument;
  *mhandle = NULL;
  if (comm == NULL || data == NULL || size == 0 ||
      (type != FLAGCX_PTR_HOST && type != FLAGCX_PTR_CUDA &&
       type != FLAGCX_PTR_DMABUF))
    return flagcxInvalidArgument;

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
  flagcxResult_t regResult = flagcxIbRegMrDmaBufInternal(
      &devBase, data, size, type, offset, fd, mrFlags, &mr);
  if (regResult != flagcxSuccess) {
    free(handle);
    return regResult;
  }

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
  (void)comm;
  struct flagcxP2pMrHandle *handle = (struct flagcxP2pMrHandle *)mhandle;
  if (handle == NULL)
    return flagcxSuccess;

  // Build a temporary devBase for the internal deregistration call
  struct flagcxIbNetCommDevBase devBase;
  memset(&devBase, 0, sizeof(devBase));
  devBase.ibDevN = handle->ibDevN;
  devBase.pd = flagcxIbDevs[handle->ibDevN].pd;

  FLAGCXCHECK(flagcxIbDeregMrInternal(&devBase, handle->mr));
  free(handle);
  return flagcxSuccess;
}

static flagcxResult_t flagcxP2pGetMrInfo(void *mhandle,
                                         struct flagcxNetMrInfo *info) {
  if (mhandle == NULL || info == NULL)
    return flagcxInvalidArgument;
  struct flagcxP2pMrHandle *handle =
      static_cast<struct flagcxP2pMrHandle *>(mhandle);
  memset(info, 0, sizeof(*info));
  info->nKeys = 1;
  info->lkeys[0] = handle->lkey;
  info->rkeys[0] = handle->rkey;
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
  comm->abortFlag = 0;
  handle->magic = FLAGCX_SOCKET_MAGIC;
  FLAGCXCHECK(flagcxSocketInit(&comm->sock, &flagcxIbIfAddr, handle->magic,
                               flagcxSocketTypeNetIb, &comm->abortFlag, 1));
  FLAGCXCHECK(flagcxSocketListen(&comm->sock));
  FLAGCXCHECK(flagcxSocketGetAddr(&comm->sock, &handle->connectAddr));
  *listenComm = comm;
  return flagcxSuccess;
}

static flagcxResult_t flagcxP2pReleasePd(int ibDevN);

// Helper: set up PD (from eager init), CQs, QPs, and GID for a connection
static flagcxResult_t flagcxP2pSetupConn(int dev,
                                         struct flagcxIbNetCommDevBase *base,
                                         struct flagcxIbQp *qp_list,
                                         int *outIbDevN, int numQps,
                                         int *qpDepthLimit) {
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

  const int accessFlags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                          IBV_ACCESS_REMOTE_ATOMIC;
  const size_t cqDepth =
      std::max<size_t>(flagcxP2pGlobalConfig().sharedCqDepth, (size_t)numQps);
  flagcxResult_t res = flagcxWrapIbvCreateCq(&base->cq, ibDev->context,
                                             (int)cqDepth, NULL, NULL, 0);
  if (res != flagcxSuccess)
    goto setup_fail;
  *qpDepthLimit = (int)std::max<size_t>(1, cqDepth / (size_t)numQps);
  *qpDepthLimit = std::min(*qpDepthLimit, 2 * MAX_REQUESTS);

  // Get GID info
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
  }

  return flagcxSuccess;

setup_fail:
  for (int i = 0; i < numQps; i++) {
    if (qp_list[i].qp) {
      flagcxWrapIbvDestroyQp(qp_list[i].qp);
      qp_list[i].qp = NULL;
    }
  }
  if (base->cq)
    flagcxWrapIbvDestroyCq(base->cq);
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

static flagcxResult_t flagcxP2pDestroyQps(struct flagcxIbQp *qp_list,
                                          int numQps) {
  for (int i = 0; i < numQps; i++) {
    if (qp_list[i].qp) {
      FLAGCXCHECK(flagcxWrapIbvDestroyQp(qp_list[i].qp));
      qp_list[i].qp = NULL;
    }
  }
  return flagcxSuccess;
}

static flagcxResult_t flagcxP2pDestroyCq(struct flagcxIbNetCommDevBase *base) {
  if (base->cq == NULL)
    return flagcxSuccess;
  FLAGCXCHECK(flagcxWrapIbvDestroyCq(base->cq));
  base->cq = NULL;
  return flagcxSuccess;
}

static bool flagcxP2pReserveQpDepth(volatile int *depth, int count, int limit) {
  int current = __atomic_load_n(depth, __ATOMIC_ACQUIRE);
  while (current <= limit - count) {
    if (__sync_bool_compare_and_swap(depth, current, current + count))
      return true;
    current = __atomic_load_n(depth, __ATOMIC_ACQUIRE);
  }
  return false;
}

static flagcxResult_t flagcxP2pPostSlices(struct flagcxP2pComm *comm,
                                          FlagcxSlice **slices, int count) {
  if (comm == NULL || slices == NULL || count <= 0 ||
      count > comm->qpDepthLimit)
    return flagcxInvalidArgument;
  for (int i = 0; i < count; i++) {
    if (slices[i] == NULL)
      return flagcxInvalidArgument;
  }

  const uint32_t start =
      comm->nextChannel.fetch_add(1, std::memory_order_relaxed);
  for (int attempt = 0; attempt < comm->numQps; attempt++) {
    const int qpIndex =
        (int)((start + (uint32_t)attempt) % (uint32_t)comm->numQps);
    if (!flagcxP2pReserveQpDepth(&comm->qpDepth[qpIndex], count,
                                 comm->qpDepthLimit))
      continue;
    for (int i = 0; i < count; i++)
      slices[i]->qpDepth = &comm->qpDepth[qpIndex];

    int failedCount = 0;
    flagcxResult_t result = flagcxP2pSliceBatch(
        comm, comm->qp_list_[qpIndex].qp, count, slices, &failedCount);
    // flagcxP2pSliceBatch marks every unposted WR failed and keeps posted WRs
    // live for CQ completion, so the asynchronous request owns the result even
    // when ibv_post_send reports a partial failure.
    (void)result;
    return flagcxSuccess;
  }
  return flagcxInProgress;
}

static flagcxResult_t flagcxP2pConnect(int dev, void *opaqueHandle,
                                       void **sendComm) {
  struct flagcxP2pListenHandle *handle =
      (struct flagcxP2pListenHandle *)opaqueHandle;
  flagcxResult_t res;
  *sendComm = NULL;

  // Allocate send comm
  auto *comm = new (std::nothrow) flagcxP2pComm();
  if (comm == NULL)
    return flagcxInternalError;
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

  FLAGCXCHECKGOTO(flagcxP2pSetupConn(dev, &comm->base, comm->qp_list_,
                                     &comm->ibDevN, comm->numQps,
                                     &comm->qpDepthLimit),
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
  flagcxP2pDestroyQps(comm->qp_list_, comm->numQps);
  flagcxP2pDestroyCq(&comm->base);
  if (comm->base.pd)
    flagcxP2pReleasePd(comm->ibDevN);
  flagcxSocketClose(&comm->sock);
  delete comm;
  return res;
}

static flagcxResult_t flagcxP2pAccept(void *listenComm, void **recvComm) {
  struct flagcxP2pListenComm *lComm = (struct flagcxP2pListenComm *)listenComm;
  *recvComm = NULL;
  if (lComm == NULL ||
      __atomic_load_n(&lComm->abortFlag, __ATOMIC_RELAXED) != 0)
    return flagcxInternalError;

  // Allocate recv comm
  auto *comm = new (std::nothrow) flagcxP2pComm();
  if (comm == NULL)
    return flagcxInternalError;

  // TCP accept (blocking, no timeout)
  flagcxResult_t res;
  int ready;
  struct flagcxP2pConnMeta localMeta[kFlagcxP2pMaxQpsPerEngine];
  struct flagcxP2pConnMeta remoteMeta[kFlagcxP2pMaxQpsPerEngine];
  int localReady = 1, remoteReady = 0;
  uint32_t localNumQps = 0, remoteNumQps = 0, agreedNumQps = 0;
  FLAGCXCHECKGOTO(flagcxSocketInit(&comm->sock, NULL, FLAGCX_SOCKET_MAGIC,
                                   flagcxSocketTypeNetIb, &lComm->abortFlag),
                  res, accept_fail);
  res = flagcxSocketAccept(&comm->sock, &lComm->sock);
  if (res != flagcxSuccess)
    goto accept_fail;
  ready = 0;
  while (!ready) {
    if (__atomic_load_n(&lComm->abortFlag, __ATOMIC_RELAXED) != 0) {
      res = flagcxInternalError;
      goto accept_fail;
    }
    res = flagcxSocketReady(&comm->sock, &ready);
    if (res != flagcxSuccess)
      goto accept_fail;
    if (!ready) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
  if (0) {
  accept_fail:
    flagcxSocketClose(&comm->sock);
    delete comm;
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

  FLAGCXCHECKGOTO(flagcxP2pSetupConn(lComm->dev, &comm->base, comm->qp_list_,
                                     &comm->ibDevN, comm->numQps,
                                     &comm->qpDepthLimit),
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
  flagcxP2pDestroyQps(comm->qp_list_, comm->numQps);
  flagcxP2pDestroyCq(&comm->base);
  if (comm->base.pd)
    flagcxP2pReleasePd(comm->ibDevN);
  flagcxSocketClose(&comm->sock);
  delete comm;
  return res;
}

/* ------------------------------------------------------------------ */
/*  One-sided transfers: iput / iget / iputSignal                      */
/* ------------------------------------------------------------------ */

// Slice request ownership model:
//   Allocation:  iput/iget/iputBatch/igetBatch allocate one or more
//                flagcxP2pSliceReq objects (and, for aggregate batch paths,
//                additional FlagcxSlice objects).
//   Submission:  The adaptor selects a QP, applies its depth gate, and posts
//                the request. The common P2P Engine owns worker scheduling.
//   Polling:     The caller polls via test() or testBatch(). Once
//                task.isAllDone() returns true, the request is complete.
//   Deallocation: test()/testBatch() call flagcxP2pFreeSliceReq() which
//                deletes any heap-allocated slices and the req itself.

static flagcxResult_t
flagcxP2pBuildSingleSliceReq(struct flagcxP2pComm *comm, uint64_t localVa,
                             uint64_t remoteVa, size_t size, uint32_t lkey,
                             uint32_t rkey, uint8_t opcode, void **request) {
  if ((uint32_t)size != size) {
    WARN("NET/IB_P2P : single-op size %zu exceeds 32-bit limit", size);
    return flagcxInternalError;
  }

  auto *req = new struct flagcxP2pSliceReq;
  req->comm = comm;
  req->slice.srcVa = localVa;
  req->slice.dstVa = remoteVa;
  req->slice.length = (uint32_t)size;
  req->slice.lkey = lkey;
  req->slice.rkey = rkey;
  req->slice.opcode = opcode;
  req->slice.task = &req->task;
  req->slice.qpDepth = NULL;
  req->task.sliceList.push_back(&req->slice);
  req->task.sliceCount.fetch_add(1, std::memory_order_release);

  FlagcxSlice *slicePtr = &req->slice;
  flagcxResult_t rc = flagcxP2pPostSlices(comm, &slicePtr, 1);
  if (rc != flagcxSuccess) {
    delete req;
    return rc;
  }

  *request = req;
  return flagcxSuccess;
}

static flagcxResult_t flagcxP2pPrepareOneSided(
    struct flagcxP2pComm *comm, const struct flagcxOneSideHandleInfo *localInfo,
    int localRank, uint64_t localOff,
    const struct flagcxOneSideHandleInfo *remoteInfo, int remoteRank,
    uint64_t remoteOff, size_t size, uint64_t *localVa, uint64_t *remoteVa,
    uint32_t *lkey, uint32_t *rkey) {
  if (comm == NULL || localInfo == NULL || remoteInfo == NULL ||
      localVa == NULL || remoteVa == NULL || lkey == NULL || rkey == NULL ||
      localInfo->baseVas == NULL || remoteInfo->baseVas == NULL ||
      localInfo->regionSizes == NULL || remoteInfo->regionSizes == NULL ||
      localInfo->mrInfos == NULL || remoteInfo->mrInfos == NULL ||
      localInfo->localMrHandle == NULL || localRank < 0 || remoteRank < 0 ||
      localRank >= localInfo->nRanks || remoteRank >= remoteInfo->nRanks)
    return flagcxInvalidArgument;

  const size_t localSize = localInfo->regionSizes[localRank];
  const size_t remoteSize = remoteInfo->regionSizes[remoteRank];
  if (localOff > localSize || size > localSize - localOff ||
      remoteOff > remoteSize || size > remoteSize - remoteOff)
    return flagcxInvalidArgument;

  const struct flagcxNetMrInfo &localMrInfo = localInfo->mrInfos[localRank];
  const struct flagcxNetMrInfo &remoteMrInfo = remoteInfo->mrInfos[remoteRank];
  if (localMrInfo.nKeys == 0 || remoteMrInfo.nKeys == 0)
    return flagcxInvalidArgument;

  struct flagcxP2pMrHandle *localHandle =
      static_cast<struct flagcxP2pMrHandle *>(localInfo->localMrHandle);
  if (localHandle->ibDevN != comm->ibDevN)
    return flagcxInvalidArgument;

  *localVa = localInfo->baseVas[localRank] + localOff;
  *remoteVa = remoteInfo->baseVas[remoteRank] + remoteOff;
  *lkey = localMrInfo.lkeys[0];
  *rkey = remoteMrInfo.rkeys[0];
  return flagcxSuccess;
}

static flagcxResult_t flagcxP2pIput(void *sendComm, uint64_t srcOff,
                                    uint64_t dstOff, size_t size, int srcRank,
                                    int dstRank, void **srcHandles,
                                    void **dstHandles, void **request) {
  struct flagcxP2pComm *comm = (struct flagcxP2pComm *)sendComm;
  const struct flagcxOneSideHandleInfo *src =
      reinterpret_cast<const struct flagcxOneSideHandleInfo *>(srcHandles);
  const struct flagcxOneSideHandleInfo *dst =
      reinterpret_cast<const struct flagcxOneSideHandleInfo *>(dstHandles);
  uint64_t localVa = 0, remoteVa = 0;
  uint32_t lkey = 0, rkey = 0;
  FLAGCXCHECK(flagcxP2pPrepareOneSided(comm, src, srcRank, srcOff, dst, dstRank,
                                       dstOff, size, &localVa, &remoteVa, &lkey,
                                       &rkey));
  return flagcxP2pBuildSingleSliceReq(comm, localVa, remoteVa, size, lkey, rkey,
                                      FLAGCX_SLICE_OP_WRITE, request);
}

static flagcxResult_t
flagcxP2pIputBatch(void *sendComm, int count, const uint64_t *srcOffs,
                   const uint64_t *dstOffs, const size_t *sizes, int srcRank,
                   int dstRank, void **srcHandles, void **dstHandles,
                   void **requests, int *posted) {
  if (requests == NULL || posted == NULL)
    return flagcxInvalidArgument;
  *posted = 0;

  const int maxWrPerPost = (int)flagcxP2pGlobalConfig().maxWrPerPost;
  if (count < 0)
    return flagcxInvalidArgument;
  for (int i = 0; i < count; i++)
    requests[i] = NULL;
  if (count == 0)
    return flagcxSuccess;

  auto *comm = static_cast<struct flagcxP2pComm *>(sendComm);
  const auto *src =
      reinterpret_cast<const struct flagcxOneSideHandleInfo *>(srcHandles);
  const auto *dst =
      reinterpret_cast<const struct flagcxOneSideHandleInfo *>(dstHandles);
  if (comm == NULL || srcOffs == NULL || dstOffs == NULL || sizes == NULL ||
      src == NULL || dst == NULL)
    return flagcxInvalidArgument;

  // iputBatch may consume a prefix of the caller's batch. Limit this post to
  // what one verbs chain and one QP can accept; the caller advances by
  // `posted` and retries the remaining descriptors on a later progress pass.
  const int submitCount =
      std::min(count, std::min(maxWrPerPost, comm->qpDepthLimit));

  struct PreparedPut {
    uint64_t localVa;
    uint64_t remoteVa;
    uint32_t lkey;
    uint32_t rkey;
  };
  std::vector<PreparedPut> prepared(submitCount);
  for (int i = 0; i < submitCount; i++) {
    if (sizes[i] > UINT32_MAX)
      return flagcxInvalidArgument;
    FLAGCXCHECK(flagcxP2pPrepareOneSided(
        comm, src, srcRank, srcOffs[i], dst, dstRank, dstOffs[i], sizes[i],
        &prepared[i].localVa, &prepared[i].remoteVa, &prepared[i].lkey,
        &prepared[i].rkey));
  }

  std::vector<struct flagcxP2pSliceReq *> reqs(submitCount, NULL);
  std::vector<FlagcxSlice *> slices(submitCount, NULL);
  for (int i = 0; i < submitCount; i++) {
    auto *req = new (std::nothrow) flagcxP2pSliceReq;
    if (req == NULL) {
      for (auto *allocated : reqs)
        delete allocated;
      return flagcxSystemError;
    }
    reqs[i] = req;
    req->comm = comm;
    req->slice.srcVa = prepared[i].localVa;
    req->slice.dstVa = prepared[i].remoteVa;
    req->slice.length = (uint32_t)sizes[i];
    req->slice.lkey = prepared[i].lkey;
    req->slice.rkey = prepared[i].rkey;
    req->slice.opcode = FLAGCX_SLICE_OP_WRITE;
    req->slice.task = &req->task;
    req->slice.qpDepth = NULL;
    req->task.sliceList.push_back(&req->slice);
    req->task.sliceCount.fetch_add(1, std::memory_order_release);
    slices[i] = &req->slice;
  }

  const uint32_t start =
      comm->nextChannel.fetch_add(1, std::memory_order_relaxed);
  int qpIndex = -1;
  for (int attempt = 0; attempt < comm->numQps; attempt++) {
    const int candidate =
        (int)((start + (uint32_t)attempt) % (uint32_t)comm->numQps);
    if (flagcxP2pReserveQpDepth(&comm->qpDepth[candidate], submitCount,
                                comm->qpDepthLimit)) {
      qpIndex = candidate;
      break;
    }
  }
  if (qpIndex < 0) {
    for (auto *req : reqs)
      delete req;
    return flagcxInProgress;
  }

  for (auto *slice : slices)
    slice->qpDepth = &comm->qpDepth[qpIndex];
  int failedCount = 0;
  flagcxResult_t result =
      flagcxP2pSliceBatch(comm, comm->qp_list_[qpIndex].qp, submitCount,
                          slices.data(), &failedCount);
  const int postedCount =
      result == flagcxSuccess ? submitCount : submitCount - failedCount;
  for (int i = 0; i < postedCount; i++)
    requests[i] = reqs[i];
  for (int i = postedCount; i < submitCount; i++)
    delete reqs[i];
  *posted = postedCount;
  return result;
}

static flagcxResult_t flagcxP2pIget(void *sendComm, uint64_t srcOff,
                                    uint64_t dstOff, size_t size, int srcRank,
                                    int dstRank, void **srcHandles,
                                    void **dstHandles, void **request) {
  struct flagcxP2pComm *comm = (struct flagcxP2pComm *)sendComm;
  const struct flagcxOneSideHandleInfo *src =
      reinterpret_cast<const struct flagcxOneSideHandleInfo *>(srcHandles);
  const struct flagcxOneSideHandleInfo *dst =
      reinterpret_cast<const struct flagcxOneSideHandleInfo *>(dstHandles);
  uint64_t localVa = 0, remoteVa = 0;
  uint32_t lkey = 0, rkey = 0;
  FLAGCXCHECK(flagcxP2pPrepareOneSided(comm, dst, dstRank, dstOff, src, srcRank,
                                       srcOff, size, &localVa, &remoteVa, &lkey,
                                       &rkey));
  return flagcxP2pBuildSingleSliceReq(comm, localVa, remoteVa, size, lkey, rkey,
                                      FLAGCX_SLICE_OP_READ, request);
}

static flagcxResult_t
flagcxP2pIgetBatch(void *sendComm, int count, const uint64_t *srcOffs,
                   const uint64_t *dstOffs, const size_t *sizes, int srcRank,
                   int dstRank, void *const *srcHandles,
                   void *const *dstHandles, void **request) {
  struct flagcxP2pComm *comm = (struct flagcxP2pComm *)sendComm;
  const int maxWrPerPost = (int)flagcxP2pGlobalConfig().maxWrPerPost;
  if (count <= 0 || count > maxWrPerPost || srcOffs == NULL ||
      dstOffs == NULL || sizes == NULL || srcHandles == NULL ||
      dstHandles == NULL || request == NULL) {
    WARN("NET/IB_P2P : invalid igetBatch arguments, count %d (max %d)", count,
         maxWrPerPost);
    return flagcxInternalError;
  }
  const struct flagcxOneSideHandleInfo *src =
      reinterpret_cast<const struct flagcxOneSideHandleInfo *>(srcHandles);
  const struct flagcxOneSideHandleInfo *dst =
      reinterpret_cast<const struct flagcxOneSideHandleInfo *>(dstHandles);

  auto *req = new struct flagcxP2pSliceReq;
  req->comm = comm;
  req->task.sliceList.reserve(count);
  for (int i = 0; i < count; i++) {
    if ((uint32_t)sizes[i] != sizes[i]) {
      WARN("NET/IB_P2P : igetBatch slice %d invalid", i);
      for (auto *s : req->task.sliceList)
        delete s;
      delete req;
      return flagcxInternalError;
    }
    uint64_t localVa = 0, remoteVa = 0;
    uint32_t lkey = 0, rkey = 0;
    flagcxResult_t prep = flagcxP2pPrepareOneSided(
        comm, dst, dstRank, dstOffs[i], src, srcRank, srcOffs[i], sizes[i],
        &localVa, &remoteVa, &lkey, &rkey);
    if (prep != flagcxSuccess) {
      for (auto *s : req->task.sliceList)
        delete s;
      delete req;
      return prep;
    }
    auto *s = new FlagcxSlice{localVa,    remoteVa, (uint32_t)sizes[i],
                              lkey,       rkey,     FLAGCX_SLICE_OP_READ,
                              &req->task, NULL};
    req->task.sliceList.push_back(s);
    req->task.sliceCount.fetch_add(1, std::memory_order_release);
  }

  flagcxResult_t rc =
      flagcxP2pPostSlices(comm, req->task.sliceList.data(), count);
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
/*  Slice batch: IBRC backend passes the chosen QP. wr_id = ptr|1.     */
/* ------------------------------------------------------------------ */

static inline enum ibv_wr_opcode flagcxSliceOpcodeToVerbs(uint8_t op) {
  return op == FLAGCX_SLICE_OP_READ ? IBV_WR_RDMA_READ : IBV_WR_RDMA_WRITE;
}

static flagcxResult_t flagcxP2pSliceBatch(void *sendComm, struct ibv_qp *qp,
                                          int count, FlagcxSlice **slices,
                                          int *failedCount) {
  struct flagcxP2pComm *comm = (struct flagcxP2pComm *)sendComm;
  if (failedCount != NULL)
    *failedCount = 0;
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
    int failed = 0;
    if (slices != NULL && count > 0) {
      for (int i = 0; i < count; i++) {
        if (slices[i] != NULL) {
          if (slices[i]->qpDepth != NULL)
            __sync_fetch_and_sub(slices[i]->qpDepth, 1);
          slices[i]->markFailed();
          failed++;
        }
      }
    }
    if (failedCount != NULL)
      *failedCount = failed;
    return flagcxInternalError;
  }

  static thread_local std::vector<struct ibv_send_wr> wrScratch;
  static thread_local std::vector<struct ibv_sge> sgeScratch;
  if ((int)wrScratch.size() < maxWrPerPost) {
    wrScratch.resize(maxWrPerPost);
    sgeScratch.resize(maxWrPerPost);
  }
  struct ibv_send_wr *wrs = wrScratch.data();
  struct ibv_sge *sges = sgeScratch.data();
  memset(wrs, 0, sizeof(*wrs) * count);

  for (int i = 0; i < count; i++) {
    FlagcxSlice *s = slices[i];
    if (s == NULL) {
      WARN("NET/IB_P2P : sliceBatch slice[%d] is NULL", i);
      for (int k = 0; k < i; k++) {
        if (slices[k]->qpDepth != NULL)
          __sync_fetch_and_sub(slices[k]->qpDepth, 1);
        slices[k]->markFailed();
      }
      for (int k = i; k < count; k++) {
        if (slices[k]) {
          if (slices[k]->qpDepth != NULL)
            __sync_fetch_and_sub(slices[k]->qpDepth, 1);
          slices[k]->markFailed();
        }
      }
      if (failedCount != NULL)
        *failedCount = count;
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
  flagcxResult_t res = flagcxWrapIbvPostSend(qp, wrs, &bad_wr);
  if (res != flagcxSuccess) {
    int failedFrom = 0;
    if (bad_wr != NULL) {
      ptrdiff_t off = bad_wr - wrs;
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
    if (failedCount != NULL)
      *failedCount = count - failedFrom;
    WARN("NET/IB_P2P : sliceBatch ibv_post_send failed (op=%s, count=%d, "
         "failedFrom=%d)",
         opLabel, count, failedFrom);
    return res;
  }

  return flagcxSuccess;
}

static flagcxResult_t flagcxP2pProgressCq(struct flagcxP2pComm *comm) {
  if (comm == NULL || comm->base.cq == NULL)
    return flagcxInvalidArgument;
  struct ibv_wc completions[256];
  const int maxCompletions =
      (int)std::min<size_t>(flagcxP2pGlobalConfig().batchPollSize,
                            sizeof(completions) / sizeof(completions[0]));
  int count = 0;
  FLAGCXCHECK(
      flagcxWrapIbvPollCq(comm->base.cq, maxCompletions, completions, &count));
  for (int i = 0; i < count; i++) {
    const uintptr_t wrId = (uintptr_t)completions[i].wr_id;
    if ((wrId & 1u) == 0) {
      WARN("NET/IB_P2P : unexpected completion wr_id=%llu",
           (unsigned long long)completions[i].wr_id);
      continue;
    }
    auto *slice = reinterpret_cast<FlagcxSlice *>(wrId & ~(uintptr_t)1u);
    if (slice->qpDepth != NULL)
      __sync_fetch_and_sub(slice->qpDepth, 1);
    if (completions[i].status == IBV_WC_SUCCESS) {
      slice->markSuccess();
    } else {
      WARN("NET/IB_P2P : RDMA completion failed, status=%d",
           (int)completions[i].status);
      slice->markFailed();
    }
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
  FLAGCXCHECK(flagcxP2pProgressCq(req->comm));
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
    FLAGCXCHECK(flagcxP2pProgressCq(req->comm));
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
/*  P2P Engine transport submission interface                         */
/* ------------------------------------------------------------------ */

static flagcxResult_t flagcxP2pIbrcGetRegistrationDevice(int netDev,
                                                         int *registrationDev) {
  if (registrationDev == NULL || netDev < 0 || netDev >= flagcxNMergedIbDevs)
    return flagcxInvalidArgument;
  *registrationDev = flagcxIbMergedDevs[netDev].devs[0];
  return flagcxSuccess;
}

static flagcxResult_t
flagcxP2pIbrcGetTransportCaps(void *sendComm,
                              struct flagcxP2pTransportCaps *caps) {
  auto *comm = static_cast<struct flagcxP2pComm *>(sendComm);
  if (comm == NULL || caps == NULL)
    return flagcxInvalidArgument;
  const auto &config = flagcxP2pGlobalConfig();
  caps->maxBatchSize =
      (uint32_t)std::min<size_t>(config.maxWrPerPost, comm->qpDepthLimit);
  caps->maxInflightBatches = (uint32_t)config.maxRequests;
  return flagcxSuccess;
}

static flagcxResult_t
flagcxP2pIbrcSubmitTransportBatch(void *sendComm,
                                  const struct flagcxP2pTransportSlice *slices,
                                  int count, void **request) {
  auto *comm = static_cast<struct flagcxP2pComm *>(sendComm);
  if (comm == NULL || slices == NULL || request == NULL || count <= 0 ||
      count > (int)flagcxP2pGlobalConfig().maxWrPerPost)
    return flagcxInvalidArgument;
  *request = NULL;

  auto *req = new struct flagcxP2pSliceReq;
  req->comm = comm;
  req->task.sliceList.reserve(count);
  for (int i = 0; i < count; i++) {
    const struct flagcxP2pTransportSlice &planned = slices[i];
    if (planned.length == 0 || planned.localMrHandle == NULL ||
        planned.localMrInfo.nKeys == 0 || planned.remoteMrInfo.nKeys == 0 ||
        (planned.opcode != FLAGCX_P2P_TRANSPORT_WRITE &&
         planned.opcode != FLAGCX_P2P_TRANSPORT_READ)) {
      flagcxP2pFreeSliceReq(req);
      return flagcxInvalidArgument;
    }
    auto *localMr =
        static_cast<struct flagcxP2pMrHandle *>(planned.localMrHandle);
    if (localMr->ibDevN != comm->ibDevN) {
      flagcxP2pFreeSliceReq(req);
      return flagcxInvalidArgument;
    }

    auto *slice = new FlagcxSlice{planned.localVa,
                                  planned.remoteVa,
                                  planned.length,
                                  planned.localMrInfo.lkeys[0],
                                  planned.remoteMrInfo.rkeys[0],
                                  planned.opcode == FLAGCX_P2P_TRANSPORT_READ
                                      ? FLAGCX_SLICE_OP_READ
                                      : FLAGCX_SLICE_OP_WRITE,
                                  &req->task,
                                  NULL};
    req->task.sliceList.push_back(slice);
    req->task.sliceCount.fetch_add(1, std::memory_order_release);
  }

  flagcxResult_t result =
      flagcxP2pPostSlices(comm, req->task.sliceList.data(), count);
  if (result != flagcxSuccess) {
    flagcxP2pFreeSliceReq(req);
    return result;
  }
  *request = req;
  return flagcxSuccess;
}

static flagcxResult_t flagcxP2pIbrcProgressTransport(void *sendComm) {
  return flagcxP2pProgressCq(static_cast<struct flagcxP2pComm *>(sendComm));
}

static flagcxResult_t flagcxP2pIbrcTestTransport(void *request, int *done,
                                                 int *failed) {
  if (done == NULL || failed == NULL)
    return flagcxInvalidArgument;
  *done = 0;
  *failed = 0;
  if (request == NULL) {
    *done = 1;
    return flagcxSuccess;
  }

  auto *req = static_cast<struct flagcxP2pSliceReq *>(request);
  if (!req->task.isAllDone())
    return flagcxSuccess;
  *done = 1;
  *failed = req->task.hasErrors() ? 1 : 0;
  flagcxP2pFreeSliceReq(req);
  return flagcxSuccess;
}

const struct flagcxP2pTransportOps flagcxP2pIbrcTransportOps = {
    "IBRC",
    flagcxP2pIbrcGetRegistrationDevice,
    flagcxP2pIbrcGetTransportCaps,
    flagcxP2pIbrcSubmitTransportBatch,
    flagcxP2pIbrcProgressTransport,
    flagcxP2pIbrcTestTransport};

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
  struct flagcxP2pComm *comm = (struct flagcxP2pComm *)sendComm;
  if (comm) {
    FLAGCXCHECK(flagcxP2pDestroyQps(comm->qp_list_, comm->numQps));
    FLAGCXCHECK(flagcxP2pDestroyCq(&comm->base));
    FLAGCXCHECK(flagcxP2pReleasePd(comm->ibDevN));
    FLAGCXCHECK(flagcxSocketClose(&comm->sock));
    delete comm;
  }
  return flagcxSuccess;
}

static flagcxResult_t flagcxP2pCloseRecv(void *recvComm) {
  struct flagcxP2pComm *comm = (struct flagcxP2pComm *)recvComm;
  if (comm) {
    FLAGCXCHECK(flagcxP2pDestroyQps(comm->qp_list_, comm->numQps));
    FLAGCXCHECK(flagcxP2pDestroyCq(&comm->base));
    FLAGCXCHECK(flagcxP2pReleasePd(comm->ibDevN));
    FLAGCXCHECK(flagcxSocketClose(&comm->sock));
    delete comm;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxP2pNetIbAbortListen(void *listenComm) {
  struct flagcxP2pListenComm *comm = (struct flagcxP2pListenComm *)listenComm;
  if (comm) {
    __atomic_store_n(&comm->abortFlag, 1, __ATOMIC_RELEASE);
    FLAGCXCHECK(flagcxSocketClose(&comm->sock));
  }
  return flagcxSuccess;
}

static flagcxResult_t flagcxP2pCloseListen(void *listenComm) {
  struct flagcxP2pListenComm *comm = (struct flagcxP2pListenComm *)listenComm;
  if (comm) {
    FLAGCXCHECK(flagcxP2pNetIbAbortListen(comm));
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

struct flagcxNetAdaptor flagcxP2pNetIb = {
    // Basic functions
    "IB_P2P",
    flagcxP2pInit,
    flagcxP2pDevices,
    flagcxP2pGetProperties,

    // Setup functions
    flagcxP2pListen,
    flagcxP2pConnect,
    flagcxP2pAccept,
    flagcxP2pCloseSend,
    flagcxP2pCloseRecv,
    flagcxP2pCloseListen,

    // Memory region functions
    flagcxP2pRegMr,
    flagcxP2pRegMrDmaBuf,
    flagcxP2pDeregMr,

    // Two-sided functions (stubs)
    flagcxP2pIsend,
    flagcxP2pIrecv,
    flagcxP2pIflush,
    flagcxP2pTest,

    // One-sided functions
    flagcxP2pIput,
    flagcxP2pIget,
    flagcxP2pIputSignal,

    // Device name lookup
    flagcxP2pGetDevFromName,

    // Optional batch operations
    flagcxP2pIputBatch, // iputBatch
    flagcxP2pTestBatch, // testBatch
    flagcxP2pIgetBatch, // igetBatch

    // MR metadata
    flagcxP2pGetMrInfo,
};
