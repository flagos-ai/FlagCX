/*************************************************************************
 * Copyright (c) 2024, UCCL Project. All rights reserved.
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * IB Retransmission UD control-channel implementation
 ************************************************************************/

#include "flagcx_common.h"
#include "ib_retrans.h"
#include "ibvwrap.h"
#include <stdlib.h>
#include <string.h>

#ifndef USE_SHCA

bool flagcxIbRetransUdSupported(void) { return true; }

flagcxResult_t flagcxIbCreateCtrlQp(struct ibv_context *context,
                                    struct ibv_pd *pd, uint8_t port_num,
                                    struct flagcxIbCtrlQp *ctrlQp) {
  if (!context || !pd || !ctrlQp)
    return flagcxInternalError;

  memset(ctrlQp, 0, sizeof(struct flagcxIbCtrlQp));

  flagcxResult_t result =
      flagcxWrapIbvCreateCq(&ctrlQp->cq, context, 1024, NULL, NULL, 0);
  if (result != flagcxSuccess)
    return result;

  struct ibv_qp_init_attr qpInitAttr;
  memset(&qpInitAttr, 0, sizeof(qpInitAttr));
  qpInitAttr.qp_type = IBV_QPT_UD;
  qpInitAttr.send_cq = ctrlQp->cq;
  qpInitAttr.recv_cq = ctrlQp->cq;
  qpInitAttr.cap.max_send_wr =
      2048; // Increased from 512 to handle high ACK traffic
  qpInitAttr.cap.max_recv_wr = 128;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.cap.max_inline_data = 64;

  result = flagcxWrapIbvCreateQp(&ctrlQp->qp, pd, &qpInitAttr);
  if (result != flagcxSuccess) {
    WARN("Failed to create control UD QP");
    flagcxResult_t cleanupResult = flagcxIbDestroyCtrlQp(ctrlQp);
    if (cleanupResult != flagcxSuccess)
      WARN("Failed to roll back control UD CQ: %d", cleanupResult);
    return result;
  }

  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(qpAttr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = 0;
  qpAttr.port_num = port_num;
  qpAttr.qkey = 0x11111111;

  result = flagcxWrapIbvModifyQp(ctrlQp->qp, &qpAttr,
                                 IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                                     IBV_QP_PORT | IBV_QP_QKEY);
  if (result != flagcxSuccess) {
    flagcxResult_t cleanupResult = flagcxIbDestroyCtrlQp(ctrlQp);
    if (cleanupResult != flagcxSuccess)
      WARN("Failed to roll back control UD QP: %d", cleanupResult);
    return result;
  }

  TRACE(FLAGCX_NET, "Created control UD QP: qpn=%u", ctrlQp->qp->qp_num);
  return flagcxSuccess;
}

flagcxResult_t flagcxIbDestroyCtrlQp(struct flagcxIbCtrlQp *ctrlQp) {
  if (!ctrlQp)
    return flagcxSuccess;
  flagcxResult_t result = flagcxSuccess;

  // Poll any remaining completions before destroying QP/CQ.
  if (ctrlQp->cq) {
    struct ibv_wc wcs[64];
    int nCqe = 0;
    for (int i = 0; i < 16; i++) {
      flagcxWrapIbvPollCq(ctrlQp->cq, 64, wcs, &nCqe);
      if (nCqe == 0)
        break;
    }
  }

  if (ctrlQp->qp) {
    flagcxResult_t qpResult = flagcxWrapIbvDestroyQp(ctrlQp->qp);
    if (result == flagcxSuccess && qpResult != flagcxSuccess)
      result = qpResult;
    if (qpResult == flagcxSuccess)
      ctrlQp->qp = NULL;
  }

  // Posted UD work can still reference the AH while the QP is alive.
  if (ctrlQp->qp != NULL)
    return result;

  if (ctrlQp->ah) {
    flagcxResult_t ahResult = flagcxWrapIbvDestroyAh(ctrlQp->ah);
    if (result == flagcxSuccess && ahResult != flagcxSuccess)
      result = ahResult;
    if (ahResult == flagcxSuccess)
      ctrlQp->ah = NULL;
  }

  // A live QP still references its CQ. Preserve both for a later retry.
  if (ctrlQp->qp == NULL && ctrlQp->cq) {
    flagcxResult_t cqResult = flagcxWrapIbvDestroyCq(ctrlQp->cq);
    if (result == flagcxSuccess && cqResult != flagcxSuccess)
      result = cqResult;
    if (cqResult == flagcxSuccess)
      ctrlQp->cq = NULL;
  }

  return result;
}

flagcxResult_t
flagcxIbSetupCtrlQpConnection(struct ibv_context *context, struct ibv_pd *pd,
                              struct flagcxIbCtrlQp *ctrlQp,
                              uint32_t remote_qpn, union ibv_gid *remote_gid,
                              uint32_t remote_lid, uint8_t port_num,
                              uint8_t link_layer, uint8_t local_gid_index) {
  if (!ctrlQp || !ctrlQp->qp)
    return flagcxInternalError;

  ctrlQp->remoteQpn = remote_qpn;
  ctrlQp->remoteQkey = 0x11111111;

  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(qpAttr));
  qpAttr.qp_state = IBV_QPS_RTR;
  FLAGCXCHECK(flagcxWrapIbvModifyQp(ctrlQp->qp, &qpAttr, IBV_QP_STATE));

  memset(&qpAttr, 0, sizeof(qpAttr));
  qpAttr.qp_state = IBV_QPS_RTS;
  qpAttr.sq_psn = 0;
  FLAGCXCHECK(
      flagcxWrapIbvModifyQp(ctrlQp->qp, &qpAttr, IBV_QP_STATE | IBV_QP_SQ_PSN));

  struct ibv_ah_attr ahAttr;
  memset(&ahAttr, 0, sizeof(ahAttr));
  ahAttr.port_num = port_num;

  if (link_layer == IBV_LINK_LAYER_ETHERNET) {
    if (!remote_gid) {
      WARN("remote_gid is NULL for RoCE");
      return flagcxInternalError;
    }

    TRACE(FLAGCX_NET,
          "Creating AH for RoCE: remote_gid=%lx:%lx, local_gid_idx=%u, port=%u",
          (unsigned long)remote_gid->global.subnet_prefix,
          (unsigned long)remote_gid->global.interface_id, local_gid_index,
          port_num);
    ahAttr.is_global = 1;
    ahAttr.grh.dgid = *remote_gid;
    ahAttr.grh.sgid_index = local_gid_index;
    ahAttr.grh.hop_limit = 255;
    ahAttr.grh.traffic_class = 0;
    ahAttr.grh.flow_label = 0;
  } else {
    TRACE(FLAGCX_NET, "Creating AH for IB: remote_lid=%u, port=%u", remote_lid,
          port_num);
    ahAttr.is_global = 0;
    ahAttr.dlid = remote_lid;
  }

  ahAttr.sl = 0;
  ahAttr.src_path_bits = 0;

  flagcxResult_t ahResult = flagcxWrapIbvCreateAh(&ctrlQp->ah, pd, &ahAttr);
  if (ahResult != flagcxSuccess) {
    WARN("  link_layer=%d (%s)", link_layer,
         link_layer == IBV_LINK_LAYER_ETHERNET ? "RoCE" : "IB");
    WARN("  remote_lid=%u", remote_lid);
    if (link_layer == IBV_LINK_LAYER_ETHERNET && remote_gid) {
      WARN("  remote_gid=%lx:%lx",
           (unsigned long)remote_gid->global.subnet_prefix,
           (unsigned long)remote_gid->global.interface_id);
      WARN("  local_gid_index=%u", local_gid_index);
    }
    return ahResult;
  }

  INFO(FLAGCX_NET, "Control QP setup: local_qpn=%u, remote_qpn=%u",
       ctrlQp->qp->qp_num, remote_qpn);
  return flagcxSuccess;
}

flagcxResult_t flagcxIbCreateSrq(struct ibv_context *context, struct ibv_pd *pd,
                                 struct flagcxIbSrqMgr *srqMgr) {
  if (!context || !pd || !srqMgr)
    return flagcxInternalError;

  memset(srqMgr, 0, sizeof(struct flagcxIbSrqMgr));
  FLAGCXCHECK(flagcxWrapIbvCreateCq(&srqMgr->cq, context,
                                    FLAGCX_IB_SRQ_SIZE * 2, NULL, NULL, 0));

  struct ibv_srq_init_attr srqAttr;
  memset(&srqAttr, 0, sizeof(srqAttr));
  srqAttr.attr.max_wr = FLAGCX_IB_SRQ_SIZE;
  srqAttr.attr.max_sge = 1;

  struct ibv_srq *srq;
  flagcxResult_t result = flagcxWrapIbvCreateSrq(&srq, pd, &srqAttr);
  if (result != flagcxSuccess) {
    WARN("Failed to create SRQ (likely SRQ not supported or symbols not "
         "loaded)");
    flagcxResult_t cqResult = flagcxWrapIbvDestroyCq(srqMgr->cq);
    if (cqResult == flagcxSuccess)
      srqMgr->cq = NULL;
    return result;
  }
  srqMgr->srq = (void *)srq;

  TRACE(FLAGCX_NET, "SRQ created successfully: %p", srq);

  size_t bufSize =
      FLAGCX_IB_RETRANS_MAX_CHUNK_SIZE + sizeof(struct flagcxIbRetransHdr);
  for (int i = 0; i < FLAGCX_IB_SRQ_SIZE; i++) {
    srqMgr->bufs[i].buffer = malloc(bufSize);
    if (!srqMgr->bufs[i].buffer) {
      WARN("Failed to allocate SRQ buffer %d", i);
      result = flagcxInternalError;
      goto fail;
    }

    srqMgr->bufs[i].size = bufSize;
    srqMgr->bufs[i].inUse = 0;
    result = flagcxWrapIbvRegMr(&srqMgr->bufs[i].mr, pd, srqMgr->bufs[i].buffer,
                                bufSize, IBV_ACCESS_LOCAL_WRITE);
    if (result != flagcxSuccess)
      goto fail;
    srqMgr->bufCount = i + 1;
    TRACE(FLAGCX_NET, "SRQ buffer[%d]: addr=%p, size=%lu, lkey=0x%x", i,
          srqMgr->bufs[i].buffer, bufSize, srqMgr->bufs[i].mr->lkey);
  }

  srqMgr->bufCount = FLAGCX_IB_SRQ_SIZE;
  for (int i = 0; i < FLAGCX_IB_SRQ_SIZE; i++)
    srqMgr->freeBufIndices[i] = i;
  srqMgr->freeBufCount = FLAGCX_IB_SRQ_SIZE;
  srqMgr->postSrqCount = 0;

  TRACE(FLAGCX_NET,
        "Created SRQ: max_wr=%d, buf_size=%lu, srq=%p, free_buffers=%d",
        FLAGCX_IB_SRQ_SIZE, (unsigned long)bufSize, srqMgr->srq,
        srqMgr->freeBufCount);
  return flagcxSuccess;

fail:
  // Include the current buffer in rollback even when its MR registration
  // failed after allocation.
  if (srqMgr->bufCount < FLAGCX_IB_SRQ_SIZE &&
      srqMgr->bufs[srqMgr->bufCount].buffer != NULL)
    srqMgr->bufCount++;
  flagcxIbDestroySrq(srqMgr);
  return result;
}

flagcxResult_t flagcxIbDestroySrq(struct flagcxIbSrqMgr *srqMgr) {
  if (!srqMgr)
    return flagcxSuccess;
  flagcxResult_t result = flagcxSuccess;

  // Posted receive WRs retain the registered buffers. Destroy the SRQ before
  // deregistering those MRs; on failure, preserve the whole dependent set for
  // a later retry.
  if (srqMgr->srq) {
    flagcxResult_t srqResult =
        flagcxWrapIbvDestroySrq((struct ibv_srq *)srqMgr->srq);
    if (result == flagcxSuccess && srqResult != flagcxSuccess)
      result = srqResult;
    if (srqResult == flagcxSuccess)
      srqMgr->srq = NULL;
  }

  if (srqMgr->srq != NULL)
    return result;

  bool buffersReleased = true;
  for (int i = 0; i < srqMgr->bufCount; i++) {
    if (srqMgr->bufs[i].mr) {
      flagcxResult_t mrResult = flagcxWrapIbvDeregMr(srqMgr->bufs[i].mr);
      if (result == flagcxSuccess && mrResult != flagcxSuccess)
        result = mrResult;
      if (mrResult == flagcxSuccess)
        srqMgr->bufs[i].mr = NULL;
    }
    if (srqMgr->bufs[i].mr == NULL && srqMgr->bufs[i].buffer) {
      free(srqMgr->bufs[i].buffer);
      srqMgr->bufs[i].buffer = NULL;
    }
    buffersReleased &=
        srqMgr->bufs[i].mr == NULL && srqMgr->bufs[i].buffer == NULL;
  }

  if (srqMgr->cq) {
    flagcxResult_t cqResult = flagcxWrapIbvDestroyCq(srqMgr->cq);
    if (result == flagcxSuccess && cqResult != flagcxSuccess)
      result = cqResult;
    if (cqResult == flagcxSuccess)
      srqMgr->cq = NULL;
  }
  if (buffersReleased) {
    srqMgr->bufCount = 0;
    srqMgr->freeBufCount = 0;
    srqMgr->postSrqCount = 0;
  }
  return result;
}

flagcxResult_t flagcxIbSrqPostRecv(struct flagcxIbSrqMgr *srqMgr, int count) {
  if (!srqMgr || !srqMgr->srq)
    return flagcxInternalError;
  if (srqMgr->freeBufCount == 0 || srqMgr->postSrqCount == 0)
    return flagcxSuccess;

  struct ibv_srq *srq = (struct ibv_srq *)srqMgr->srq;
  int postBatch = count;
  if (postBatch > srqMgr->postSrqCount)
    postBatch = srqMgr->postSrqCount;
  if (postBatch > srqMgr->freeBufCount)
    postBatch = srqMgr->freeBufCount;
  if (postBatch == 0)
    return flagcxSuccess;

  static __thread struct ibv_recv_wr recvWrs[64];
  static __thread struct ibv_sge recvSges[64];
  if (postBatch > 64)
    postBatch = 64;

  for (int i = 0; i < postBatch; i++) {
    srqMgr->freeBufCount--;
    int bufIdx = srqMgr->freeBufIndices[srqMgr->freeBufCount];

    recvSges[i].addr = (uint64_t)srqMgr->bufs[bufIdx].buffer;
    recvSges[i].length = srqMgr->bufs[bufIdx].size;
    recvSges[i].lkey = srqMgr->bufs[bufIdx].mr->lkey;

    memset(&recvWrs[i], 0, sizeof(recvWrs[i]));
    recvWrs[i].wr_id = bufIdx;
    recvWrs[i].sg_list = &recvSges[i];
    recvWrs[i].num_sge = 1;
    recvWrs[i].next = (i == postBatch - 1) ? NULL : &recvWrs[i + 1];
    srqMgr->bufs[bufIdx].inUse = 1;

    TRACE(FLAGCX_NET, "Posting SRQ recv[%d]: buf_idx=%d, buffer=%p, wr_id=%d",
          i, bufIdx, srqMgr->bufs[bufIdx].buffer, bufIdx);
  }

  struct ibv_recv_wr *badWr;
  flagcxResult_t ret = flagcxWrapIbvPostSrqRecv(srq, &recvWrs[0], &badWr);
  if (ret != flagcxSuccess) {
    WARN("Failed to batch post %d recv WRs to SRQ", postBatch);
    return flagcxRemoteError;
  }

  srqMgr->postSrqCount -= postBatch;
  TRACE(FLAGCX_NET,
        "Posted %d recv WRs to SRQ (free_buf_count=%d, post_srq_count=%d)",
        postBatch, srqMgr->freeBufCount, srqMgr->postSrqCount);
  return flagcxSuccess;
}

#else

bool flagcxIbRetransUdSupported(void) { return false; }

flagcxResult_t flagcxIbCreateCtrlQp(struct ibv_context *, struct ibv_pd *,
                                    uint8_t, struct flagcxIbCtrlQp *ctrlQp) {
  if (ctrlQp)
    memset(ctrlQp, 0, sizeof(*ctrlQp));
  return flagcxNotSupported;
}

flagcxResult_t flagcxIbDestroyCtrlQp(struct flagcxIbCtrlQp *ctrlQp) {
  if (ctrlQp)
    memset(ctrlQp, 0, sizeof(*ctrlQp));
  return flagcxSuccess;
}

flagcxResult_t flagcxIbSetupCtrlQpConnection(struct ibv_context *,
                                             struct ibv_pd *,
                                             struct flagcxIbCtrlQp *, uint32_t,
                                             union ibv_gid *, uint32_t, uint8_t,
                                             uint8_t, uint8_t) {
  return flagcxNotSupported;
}

flagcxResult_t flagcxIbCreateSrq(struct ibv_context *, struct ibv_pd *,
                                 struct flagcxIbSrqMgr *srqMgr) {
  if (srqMgr)
    memset(srqMgr, 0, sizeof(*srqMgr));
  return flagcxNotSupported;
}

flagcxResult_t flagcxIbDestroySrq(struct flagcxIbSrqMgr *srqMgr) {
  if (srqMgr)
    memset(srqMgr, 0, sizeof(*srqMgr));
  return flagcxSuccess;
}

flagcxResult_t flagcxIbSrqPostRecv(struct flagcxIbSrqMgr *, int) {
  return flagcxNotSupported;
}

#endif
