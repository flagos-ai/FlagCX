#include "ib_common.h"
#include "flagcx_common.h"
#include "ib_retrans.h"
#include "ibvwrap.h"
#include "socket.h"
#include "timer.h"

#include <arpa/inet.h>
#include <sched.h>

flagcxResult_t
flagcxIbCommonPostFifo(struct flagcxIbRecvComm *comm, int n, void **data,
                       size_t *sizes, int *tags, void **mhandles,
                       struct flagcxIbRequest *req,
                       void (*addEventFunc)(struct flagcxIbRequest *, int,
                                            struct flagcxIbNetCommDevBase *)) {
  if (!comm || !req || !addEventFunc)
    return flagcxInternalError;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));

  int slot = comm->remFifo.fifoTail % MAX_REQUESTS;
  req->recv.sizes = comm->sizesFifo[slot];
  for (int i = 0; i < n; i++)
    req->recv.sizes[i] = 0;
  struct flagcxIbSendFifo *localElem = comm->remFifo.elems[slot];

  flagcxIbQp *ctsQp = comm->base.qps + comm->base.devIndex;
  comm->base.devIndex = (comm->base.devIndex + 1) % comm->base.ndevs;

  for (int i = 0; i < n; i++) {
    localElem[i].addr = (uint64_t)data[i];
    struct flagcxIbMrHandle *mhandleWrapper =
        (struct flagcxIbMrHandle *)mhandles[i];

    for (int j = 0; j < comm->base.ndevs; j++)
      localElem[i].rkeys[j] = mhandleWrapper->mrs[j]->rkey;
    localElem[i].nreqs = n;
    localElem[i].size = sizes[i];
    localElem[i].tag = tags[i];
    localElem[i].idx = comm->remFifo.fifoTail + 1;
  }
  wr.wr.rdma.remote_addr =
      comm->remFifo.addr +
      slot * FLAGCX_NET_IB_MAX_RECVS * sizeof(struct flagcxIbSendFifo);

  wr.wr.rdma.rkey = comm->base.remDevs[ctsQp->remDevIdx].fifoRkey;

  comm->devs[ctsQp->devIndex].fifoSge.addr = (uint64_t)localElem;
  comm->devs[ctsQp->devIndex].fifoSge.length =
      n * sizeof(struct flagcxIbSendFifo);
  wr.sg_list = &comm->devs[ctsQp->devIndex].fifoSge;
  wr.num_sge = 1;

  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = comm->remFifo.flags;

  if (slot == ctsQp->devIndex) {
    wr.send_flags |= IBV_SEND_SIGNALED;
    wr.wr_id = req - comm->base.reqs;
    addEventFunc(req, ctsQp->devIndex, &comm->devs[ctsQp->devIndex].base);
  }

  struct ibv_send_wr *bad_wr;
  flagcxResult_t status = flagcxWrapIbvPostSend(ctsQp->qp, &wr, &bad_wr);
  if (status != flagcxSuccess)
    return status;

  comm->remFifo.fifoTail++;
  return flagcxSuccess;
}

static inline const char *
flagcxIbCommonComponent(const struct flagcxIbCommonTestOps *ops) {
  return (ops && ops->component) ? ops->component : "NET/IB";
}

// A terminal completion consumes the request just like a successful one.
// Callers must not retry a request after test() reports a completion error;
// retaining its slot would permanently reduce the fixed request pool.
static flagcxResult_t flagcxIbCommonReleaseRequest(struct flagcxIbRequest *r) {
  if (r->type == FLAGCX_NET_IB_REQ_SEND && r->base->isSend) {
    struct flagcxIbSendComm *sComm = (struct flagcxIbSendComm *)r->base;
    if (sComm->outstandingSends > 0)
      sComm->outstandingSends--;
  }
  return flagcxIbFreeRequest(r);
}

static flagcxResult_t
flagcxIbCommonRecordRequestEvent(struct flagcxIbRequest *req, int devIndex,
                                 flagcxResult_t result) {
  if (req == NULL || devIndex < 0 || devIndex >= FLAGCX_IB_MAX_DEVS_PER_NIC ||
      req->events[devIndex] <= 0) {
    return flagcxInternalError;
  }
  req->events[devIndex]--;
  if (req->result == flagcxSuccess && result != flagcxSuccess)
    req->result = result;
  return flagcxSuccess;
}

// Record a CQE against the request encoded in wr_id, not the request whose
// test() call happened to poll the shared CQ. A batched read reuses one wr_id
// for several WRs, so every CQE consumes exactly one event and the request is
// retired only after the complete accepted prefix has been drained.
flagcxResult_t
flagcxIbCommonRecordDataCompletion(struct flagcxIbNetCommBase *base,
                                   uint64_t wrId, int devIndex,
                                   flagcxResult_t result) {
  if (base == NULL)
    return flagcxInvalidArgument;
  uint8_t reqIndex = wrId & 0xff;
  if (reqIndex >= MAX_REQUESTS)
    return flagcxInternalError;

  struct flagcxIbRequest *req = base->reqs + reqIndex;
  if (req->type == FLAGCX_NET_IB_REQ_UNUSED)
    return flagcxInternalError;

  if (req->type == FLAGCX_NET_IB_REQ_SEND) {
    if (req->nreqs <= 0 || req->nreqs > FLAGCX_NET_IB_MAX_RECVS)
      return flagcxInternalError;
    for (int j = 0; j < req->nreqs; j++) {
      uint8_t sendReqIndex = (wrId >> (j * 8)) & 0xff;
      if (sendReqIndex >= MAX_REQUESTS)
        return flagcxInternalError;
      struct flagcxIbRequest *sendReq = base->reqs + sendReqIndex;
      FLAGCXCHECK(flagcxIbCommonRecordRequestEvent(sendReq, devIndex, result));
    }
  } else {
    FLAGCXCHECK(flagcxIbCommonRecordRequestEvent(req, devIndex, result));
  }
  return flagcxSuccess;
}

// Error CQEs may be generated for unsignaled WRs. Record the error on the
// owning request, but leave its event count for the signaled tail WR that
// retires the chain.
flagcxResult_t
flagcxIbCommonRecordUnsignaledCompletion(struct flagcxIbNetCommBase *base,
                                         uint64_t wrId, flagcxResult_t result) {
  if (base == NULL || !flagcxIbIsUnsignaledWrId(wrId))
    return flagcxInvalidArgument;
  uint8_t reqIndex = wrId & 0xff;
  struct flagcxIbRequest *req = base->reqs + reqIndex;
  if (req->type == FLAGCX_NET_IB_REQ_UNUSED)
    return flagcxInternalError;
  if (req->result == flagcxSuccess && result != flagcxSuccess)
    req->result = result;
  return flagcxSuccess;
}

flagcxResult_t
flagcxIbCommonTestDataQp(struct flagcxIbRequest *r, int *done, int *sizes,
                         const struct flagcxIbCommonTestOps *ops) {
  if (!r || !done)
    return flagcxInternalError;

  if (ops && ops->pre_check) {
    FLAGCXCHECK(ops->pre_check(r));
  }

  *done = 0;
  while (1) {
    if (r->events[0] == 0 && r->events[1] == 0) {
      TRACE(FLAGCX_NET, "r=%p done", r);
      *done = 1;
      flagcxResult_t result = r->result;
      if (sizes && r->type == FLAGCX_NET_IB_REQ_RECV) {
        for (int i = 0; i < r->nreqs; i++)
          sizes[i] = r->recv.sizes[i];
      }
      if (sizes && r->type == FLAGCX_NET_IB_REQ_SEND) {
        sizes[0] = r->send.size;
      }
      FLAGCXCHECK(flagcxIbCommonReleaseRequest(r));
      return result;
    }

    int totalWrDone = 0;
    struct ibv_wc wcs[4];
    static __thread int poll_spin_count = 0;

    for (int i = 0; i < FLAGCX_IB_MAX_DEVS_PER_NIC; i++) {
      TIME_START(3);
      if (r->events[i]) {
        int wrDone = 0;
        FLAGCXCHECK(flagcxWrapIbvPollCq(r->devBases[i]->cq, 4, wcs, &wrDone));
        totalWrDone += wrDone;
        if (wrDone == 0) {
          TIME_CANCEL(3);
        } else {
          TIME_STOP(3);
          poll_spin_count = 0;
        }
        if (wrDone == 0) {
          if (++poll_spin_count > 100) {
            sched_yield();
            poll_spin_count = 0;
          }
          continue;
        }
        for (int w = 0; w < wrDone; w++) {
          struct ibv_wc *wc = wcs + w;

          if (flagcxIbIsUnsignaledWrId(wc->wr_id)) {
            if (wc->status != IBV_WC_SUCCESS) {
              FLAGCXCHECK(flagcxIbCommonRecordUnsignaledCompletion(
                  r->base, wc->wr_id, flagcxRemoteError));
              WARN("%s: unsignaled WR failed with status=%d opcode=%d "
                   "vendor_err=%d wr_id=%llu",
                   flagcxIbCommonComponent(ops), wc->status, wc->opcode,
                   wc->vendor_err, (unsigned long long)wc->wr_id);
            }
            continue;
          }

          bool handled = false;
          // Let the transport consume special completion identifiers before
          // the common path interprets wr_id as a request index. In
          // particular, IBUC SRQ receives encode a shared buffer index rather
          // than a flagcxIbRequest index, including on failed completions.
          if (ops && ops->process_wc) {
            FLAGCXCHECK(ops->process_wc(r, wc, i, &handled));
            if (handled)
              continue;
          }

          if (wc->status != IBV_WC_SUCCESS) {
            union flagcxSocketAddress addr;
            flagcxSocketGetAddr(r->sock, &addr);
            char localGidString[INET6_ADDRSTRLEN] = "";
            char remoteGidString[INET6_ADDRSTRLEN] = "";
            const char *localGidStr = NULL, *remoteGidStr = NULL;
            if (r->devBases[i]->gidInfo.linkLayer == IBV_LINK_LAYER_ETHERNET) {
              localGidStr =
                  inet_ntop(AF_INET6, &r->devBases[i]->gidInfo.localGid,
                            localGidString, sizeof(localGidString));
              remoteGidStr =
                  inet_ntop(AF_INET6, &r->base->remDevs[i].remoteGid,
                            remoteGidString, sizeof(remoteGidString));
            }
            char line[SOCKET_NAME_MAXLEN + 1];
            WARN("%s : Got completion from peer %s with status=%d opcode=%d "
                 "len=%d vendor err %d (%s)%s%s%s%s",
                 flagcxIbCommonComponent(ops),
                 flagcxSocketToString(&addr, line), wc->status, wc->opcode,
                 wc->byte_len, wc->vendor_err, reqTypeStr[r->type],
                 localGidStr ? " localGid " : "", localGidString,
                 remoteGidStr ? " remoteGid " : "", remoteGidString);
            FLAGCXCHECK(flagcxIbCommonRecordDataCompletion(
                r->base, wc->wr_id, i, flagcxRemoteError));
            continue;
          }

          uint8_t req_idx = wc->wr_id & 0xff;
          if (req_idx >= MAX_REQUESTS)
            continue;

          struct flagcxIbRequest *req = r->base->reqs + req_idx;

#ifdef ENABLE_TRACE
          union flagcxSocketAddress addr;
          flagcxSocketGetAddr(r->sock, &addr);
          char line[SOCKET_NAME_MAXLEN + 1];
          TRACE(FLAGCX_NET,
                "Got completion from peer %s with status=%d opcode=%d len=%d "
                "wr_id=%ld r=%p type=%d events={%d,%d}, i=%d",
                flagcxSocketToString(&addr, line), wc->status, wc->opcode,
                wc->byte_len, wc->wr_id, req, req->type, req->events[0],
                req->events[1], i);
#endif

          if (req->type != FLAGCX_NET_IB_REQ_SEND) {
            if (req && wc->opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
              if (req->type != FLAGCX_NET_IB_REQ_RECV) {
                static __thread int type_mismatch_count = 0;
                if ((type_mismatch_count++ % 100) == 0) {
                  WARN("%s: Type mismatch for RECV_RDMA_WITH_IMM: req->type=%d "
                       "(expected RECV=%d), count=%d",
                       flagcxIbCommonComponent(ops), req->type,
                       FLAGCX_NET_IB_REQ_RECV, type_mismatch_count);
                }
                continue;
              }
              if (req->nreqs == 1) {
                req->recv.sizes[0] = wc->imm_data;
              }
            }
          }
          FLAGCXCHECK(flagcxIbCommonRecordDataCompletion(r->base, wc->wr_id, i,
                                                         flagcxSuccess));
        }
      } else {
        TIME_CANCEL(3);
      }
    }

    if (totalWrDone == 0)
      return flagcxSuccess;
  }
}
