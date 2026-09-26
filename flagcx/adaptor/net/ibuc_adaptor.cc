/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifdef USE_IBUC

#include "adaptor.h"
#include "core.h"
#include "flagcx_common.h"
#include "flagcx_net.h"
#include "ib_common.h"
#include "ib_retrans.h"
#include "ib_transport.h"
#include "ibvwrap.h"
#include "net.h"
#include "param.h"
#include "socket.h"
#include "timer.h"
#include "utils.h"
#include <algorithm>
#include <assert.h>
#include <errno.h>
#include <inttypes.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

static flagcxResult_t
flagcxIbucPostRetransRecv(struct flagcxIbRecvCommDev *commDev,
                          uint32_t creditIndex);
static flagcxResult_t flagcxIbucPostDataRecv(struct flagcxIbRecvComm *comm,
                                             uint32_t qpIndex);
static flagcxResult_t flagcxIbucPostAckRecv(struct flagcxIbSendCommDev *commDev,
                                            uint32_t creditIndex);

static constexpr size_t flagcxIbucRetransRecvEntrySize =
    sizeof(struct flagcxIbRetransHdr) + FLAGCX_IB_RETRANS_MAX_CHUNK_SIZE;

static constexpr uint64_t flagcxIbucAckRecvWrIdPrefix = 0x4000000000000000ULL;
static constexpr uint64_t flagcxIbucAckSendWrId = 0x2000000000000000ULL;

static flagcxResult_t
flagcxIbucCompleteRetransWindow(struct flagcxIbSendComm *comm,
                                const struct ibv_wc *wc) {
  if (comm == NULL || wc == NULL)
    return flagcxInvalidArgument;
  if (comm->outstandingRetrans <= 0)
    return flagcxIbCommonRecordCommError(&comm->base, flagcxInternalError);
  const bool failed = wc->status != IBV_WC_SUCCESS;
  // A failed RC completion places the QP in error, so none of its remaining
  // WRs can continue reading the source buffers. Retire the whole prefix; the
  // communicator error prevents any subsequent retransmission submission.
  if (failed)
    comm->outstandingRetrans = 0;
  else
    comm->outstandingRetrans--;
  // Every accepted retransmission chunk is signaled. Keep the shared source
  // request hold until the last CQE for this accepted prefix arrives.
  if (comm->outstandingRetrans > 0)
    return flagcxSuccess;

  const int windowDev = comm->retransWindowDevIndex;
  for (int i = 0; i < comm->retransWindowNreqs; i++) {
    struct flagcxIbRequest *windowReq = comm->retransWindowRequests[i];
    if (windowReq != NULL && windowDev >= 0 &&
        windowDev < FLAGCX_IB_MAX_DEVS_PER_NIC &&
        windowReq->events[windowDev] > 0)
      windowReq->events[windowDev]--;
    comm->retransWindowRequests[i] = NULL;
  }
  comm->retransWindowNreqs = 0;
  return failed ? flagcxIbCommonRecordCommError(&comm->base, flagcxRemoteError)
                : flagcxSuccess;
}

static flagcxResult_t flagcxIbucPollRetransCq(struct flagcxIbSendComm *comm,
                                              int devIndex) {
  if (comm == NULL || devIndex < 0 || devIndex >= comm->base.ndevs)
    return flagcxInvalidArgument;
  struct flagcxIbSendCommDev *commDev = &comm->devs[devIndex];
  if (commDev->retransCq == NULL)
    return flagcxInternalError;

  struct ibv_wc wcs[8];
  int nCqe = 0;
  FLAGCXCHECK(flagcxWrapIbvPollCq(commDev->retransCq, 8, wcs, &nCqe));
  for (int i = 0; i < nCqe; i++) {
    struct ibv_wc *wc = &wcs[i];
    if (wc->wr_id == FLAGCX_RETRANS_WR_ID) {
      FLAGCXCHECK(flagcxIbucCompleteRetransWindow(comm, wc));
      continue;
    }
    if ((wc->wr_id & flagcxIbucAckRecvWrIdPrefix) == 0 ||
        wc->status != IBV_WC_SUCCESS || wc->opcode != IBV_WC_RECV)
      return flagcxIbCommonRecordCommError(&comm->base, flagcxRemoteError);

    const uint32_t creditIndex = wc->wr_id & ~flagcxIbucAckRecvWrIdPrefix;
    if (creditIndex >= FLAGCX_IB_ACK_BUF_COUNT)
      return flagcxIbCommonRecordCommError(&comm->base, flagcxInternalError);
    const size_t entrySize =
        sizeof(struct flagcxIbAckMsg) + FLAGCX_IB_ACK_BUF_PADDING;
    struct flagcxIbAckMsg *ack =
        (struct flagcxIbAckMsg *)((char *)commDev->ackBuffer +
                                  creditIndex * entrySize +
                                  FLAGCX_IB_ACK_BUF_PADDING);
    FLAGCXCHECK(flagcxIbRetransProcessAck(&comm->retrans, ack));
    FLAGCXCHECK(flagcxIbucPostAckRecv(commDev, creditIndex));
  }
  return flagcxSuccess;
}

static flagcxResult_t flagcxIbucTestPreCheck(struct flagcxIbRequest *r) {
  if (!r)
    return flagcxInternalError;

  if (r->type == FLAGCX_NET_IB_REQ_SEND && r->base->isSend) {
    struct flagcxIbSendComm *sComm = (struct flagcxIbSendComm *)r->base;

    static __thread uint64_t lastLogTime = 0;
    uint64_t nowUs = flagcxIbGetTimeUs();
    if (nowUs - lastLogTime > 100000) {
      lastLogTime = nowUs;
    }

    if (sComm->retrans.enabled) {
      for (int i = 0; i < sComm->base.ndevs; i++) {
        FLAGCXCHECK(flagcxIbucPollRetransCq(sComm, i));
      }

      uint64_t nowUs2 = flagcxIbGetTimeUs();
      const uint64_t CHECK_INTERVAL_US = 1000;
      if (nowUs2 - sComm->lastTimeoutCheckUs >= CHECK_INTERVAL_US) {
        flagcxResult_t retransResult =
            flagcxIbRetransCheckTimeout(&sComm->retrans, sComm);
        if (retransResult != flagcxSuccess &&
            retransResult != flagcxInProgress) {
          return flagcxIbCommonRecordCommError(&sComm->base, retransResult);
        }
        sComm->lastTimeoutCheckUs = nowUs2;
      }
    }
  }

  return flagcxSuccess;
}

static flagcxResult_t flagcxIbucSendAckRc(struct flagcxIbRecvComm *comm,
                                          const struct flagcxIbAckMsg *ack,
                                          int devIndex) {
  if (comm == NULL || ack == NULL || devIndex < 0 ||
      devIndex >= comm->base.ndevs)
    return flagcxInvalidArgument;
  struct flagcxIbRecvCommDev *commDev = &comm->devs[devIndex];
  if (commDev->retransQp.qp == NULL)
    return flagcxInternalError;

  struct ibv_sge sge = {};
  sge.addr = (uint64_t)ack;
  sge.length = sizeof(*ack);
  struct ibv_send_wr wr = {};
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  // The retransmission QP has no other periodically signaled send traffic.
  // Signal every ACK so the provider can retire SQ entries indefinitely; its
  // CQE is consumed by flagcxIbucProcessWc() on the shared receive CQ.
  wr.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
  wr.wr_id = flagcxIbucAckSendWrId;
  struct ibv_send_wr *badWr = NULL;
  flagcxResult_t result =
      flagcxWrapIbvPostSendRetryable(commDev->retransQp.qp, &wr, &badWr);
  // The completion that caused this ACK has already been consumed. Failing
  // closed is safer than returning retryable pressure with no progress token.
  return result == flagcxInProgress ? flagcxSystemError : result;
}

static flagcxResult_t flagcxIbucAckSequence(struct flagcxIbRecvComm *comm,
                                            uint32_t seq, int devIndex) {
  struct flagcxIbAckMsg ack = {};
  int shouldAck = 0;
  FLAGCXCHECK(flagcxIbRetransRecvPacket(&comm->retrans, seq, &ack, &shouldAck));
  if (shouldAck)
    FLAGCXCHECK(flagcxIbucSendAckRc(comm, &ack, devIndex));
  return flagcxSuccess;
}

static flagcxResult_t flagcxIbucProcessWc(struct flagcxIbRequest *r,
                                          struct ibv_wc *wc, int devIndex,
                                          bool *handled) {
  if (!r || !wc || !handled)
    return flagcxInternalError;
  *handled = false;

  // Reliable ACK SEND completions use a reserved id and do not belong to a
  // logical receive request.
  if (wc->wr_id == flagcxIbucAckSendWrId) {
    *handled = true;
    return wc->status == IBV_WC_SUCCESS
               ? flagcxSuccess
               : flagcxIbCommonRecordCommError(r->base, flagcxRemoteError);
  }

  if (wc->wr_id == FLAGCX_RETRANS_WR_ID) {
    if (r->base->isSend) {
      struct flagcxIbSendComm *sComm = (struct flagcxIbSendComm *)r->base;
      FLAGCXCHECK(flagcxIbucCompleteRetransWindow(sComm, wc));
      TRACE(FLAGCX_NET, "RC retrans completed, outstanding_retrans=%d",
            sComm->outstandingRetrans);
    }
    *handled = true;
    return flagcxSuccess;
  }

  if (r->base->isSend)
    return flagcxSuccess;

  struct flagcxIbRecvComm *rComm = (struct flagcxIbRecvComm *)r->base;
  if ((wc->wr_id & FLAGCX_IBUC_DATA_RECV_WR_ID_PREFIX) != 0 &&
      (wc->wr_id & FLAGCX_IBUC_RETRANS_RECV_WR_ID_PREFIX) == 0) {
    *handled = true;
    const uint32_t qpIndex = wc->wr_id & ~FLAGCX_IBUC_DATA_RECV_WR_ID_PREFIX;
    if (wc->status != IBV_WC_SUCCESS)
      return flagcxIbCommonRecordCommError(r->base, flagcxRemoteError);
    if (wc->opcode != IBV_WC_RECV_RDMA_WITH_IMM ||
        qpIndex >= (uint32_t)rComm->base.nqps ||
        rComm->base.qps[qpIndex].devIndex != devIndex)
      return flagcxIbCommonRecordCommError(r->base, flagcxInternalError);

    uint32_t seq = 0;
    uint8_t requestSlot = 0;
    uint16_t generation = 0;
    flagcxIbucDecodeImmData(wc->imm_data, &seq, &requestSlot, &generation);
    struct flagcxIbRequest *target = r->base->reqs + requestSlot;
    const bool currentRequest =
        flagcxIbucRequestMatchesGeneration(target, generation);

    // Replenish the connection-level notification credit before completing or
    // acknowledging the logical request. A delayed UC notification is never
    // tied to the lifetime of the request that happened to post the WQE.
    FLAGCXCHECK(flagcxIbucPostDataRecv(rComm, qpIndex));

    if (!currentRequest || flagcxIbSeqLess(seq, rComm->retrans.recvSeq)) {
      FLAGCXCHECK(flagcxIbucAckSequence(rComm, seq, devIndex));
      return flagcxSuccess;
    }
    const uint8_t completeMask = (uint8_t)((1u << target->nreqs) - 1);
    if (target->retransSeq == seq &&
        target->retransSegmentMask == completeMask) {
      FLAGCXCHECK(flagcxIbucAckSequence(rComm, seq, devIndex));
      return flagcxSuccess;
    }
    if (target->events[devIndex] <= 0)
      return flagcxIbCommonRecordCommError(r->base, flagcxInternalError);

    target->retransSeq = seq;
    int pendingDataEvents = 0;
    for (int i = 0; i < FLAGCX_IB_MAX_DEVS_PER_NIC; i++)
      pendingDataEvents += target->dataEvents[i];
    // A transfer may be striped over several UC QPs. The final notification
    // acknowledges the whole logical batch and releases the source requests.
    if (pendingDataEvents == 1)
      FLAGCXCHECK(flagcxIbucAckSequence(rComm, seq, devIndex));
    if (target->dataEvents[devIndex] <= 0)
      return flagcxIbCommonRecordCommError(r->base, flagcxInternalError);
    target->dataEvents[devIndex]--;
    target->events[devIndex]--;
    return flagcxSuccess;
  }

  if ((wc->wr_id & FLAGCX_IBUC_RETRANS_RECV_WR_ID_PREFIX) != 0) {
    *handled = true;
    const uint32_t creditIndex =
        wc->wr_id & ~FLAGCX_IBUC_RETRANS_RECV_WR_ID_PREFIX;
    if (wc->status != IBV_WC_SUCCESS)
      return flagcxIbCommonRecordCommError(r->base, flagcxRemoteError);
    if (wc->opcode != IBV_WC_RECV || devIndex < 0 ||
        devIndex >= rComm->base.ndevs ||
        creditIndex >= (uint32_t)rComm->devs[devIndex].retransRecvBufCount)
      return flagcxIbCommonRecordCommError(r->base, flagcxInternalError);

    struct flagcxIbRecvCommDev *commDev = &rComm->devs[devIndex];
    char *buffer = (char *)commDev->retransRecvBufs[creditIndex];
    struct flagcxIbRetransHdr *hdr = (struct flagcxIbRetransHdr *)buffer;
    const uint8_t requestSlot = hdr->immData & 0xff;
    const uint8_t segment = (hdr->immData >> 8) & 0xff;
    const uint8_t nreqs = (hdr->immData >> 16) & 0xff;
    const uint32_t retransSeq = hdr->seq;
    const uint16_t generation = hdr->remoteAddr & FLAGCX_IBUC_GENERATION_MASK;
    const uint32_t chunkOffset = hdr->remoteAddr >> 16;
    const uint32_t totalSize = hdr->rkey;
    if (hdr->magic != FLAGCX_RETRANS_MAGIC || nreqs == 0 ||
        nreqs > FLAGCX_NET_IB_MAX_RECVS || segment >= nreqs ||
        hdr->size > FLAGCX_IB_RETRANS_MAX_CHUNK_SIZE ||
        chunkOffset > totalSize || hdr->size > totalSize - chunkOffset ||
        wc->byte_len != sizeof(*hdr) + hdr->size)
      return flagcxIbCommonRecordCommError(r->base, flagcxRemoteError);

    struct flagcxIbRequest *target = r->base->reqs + requestSlot;
    const bool currentRequest =
        flagcxIbucRequestMatchesGeneration(target, generation) &&
        target->nreqs == nreqs &&
        (target->retransSeq == UINT32_MAX || target->retransSeq == retransSeq);
    if (currentRequest) {
      if (totalSize > target->recv.capacities[segment])
        return flagcxIbCommonRecordCommError(r->base, flagcxRemoteError);
      const uint8_t segmentBit = (uint8_t)(1u << segment);
      const uint32_t received = target->retransSegmentBytes[segment];
      if (chunkOffset > received)
        return flagcxIbCommonRecordCommError(r->base, flagcxRemoteError);
      if (chunkOffset == received &&
          (target->retransSegmentMask & segmentBit) == 0) {
        void *payload = buffer + sizeof(*hdr);
        flagcxResult_t copyResult = flagcxSuccess;
        if (target->recv.types[segment] == FLAGCX_PTR_HOST) {
          memcpy((char *)target->recv.data[segment] + chunkOffset, payload,
                 hdr->size);
        } else {
          copyResult = deviceAdaptor->deviceMemcpy(
              (char *)target->recv.data[segment] + chunkOffset, payload,
              hdr->size, flagcxMemcpyHostToDevice, NULL, NULL);
        }
        if (copyResult != flagcxSuccess)
          return flagcxIbCommonRecordCommError(r->base, copyResult);
        target->retransSeq = retransSeq;
        target->retransSegmentBytes[segment] += hdr->size;
        if (target->retransSegmentBytes[segment] == totalSize) {
          target->recv.sizes[segment] = totalSize;
          target->retransSegmentMask |= segmentBit;
        }
      }
    }

    // Release the receive credit before sending the ACK. If the ACK path is
    // temporarily backpressured, the reliable RC channel can still progress.
    FLAGCXCHECK(flagcxIbucPostRetransRecv(commDev, creditIndex));

    const uint8_t completeMask = (uint8_t)((1u << nreqs) - 1);
    if (!currentRequest || target->retransSegmentMask == completeMask) {
      FLAGCXCHECK(flagcxIbucAckSequence(rComm, retransSeq, devIndex));
    }
    if (currentRequest && target->retransSegmentMask == completeMask) {
      for (int i = 0; i < FLAGCX_IB_MAX_DEVS_PER_NIC; i++) {
        // RC retransmission replaces only the missing UC data notifications.
        // Preserve an independently pending FIFO-write completion so the
        // request slot cannot be recycled before that CQE arrives.
        if (target->events[i] < target->dataEvents[i])
          return flagcxIbCommonRecordCommError(r->base, flagcxInternalError);
        target->events[i] -= target->dataEvents[i];
        target->dataEvents[i] = 0;
      }
    }
    return flagcxSuccess;
  }

  return flagcxSuccess;
}

flagcxResult_t flagcxIbucInit() {
  flagcxResult_t ret;
  if (flagcxParamIbDisable()) {
    return flagcxInternalError;
  }
  // UC does not provide reliable delivery. IBUC therefore requires its
  // reliable RC acknowledgement and retransmission channels; running without
  // them would silently expose lossy transport semantics to collective callers.
  if (!flagcxIbRetransUdSupported()) {
    WARN("NET/IBUC : retransmission control-channel support is unavailable");
    return flagcxNotSupported;
  }
  static int shownIbucHcaEnv = 0;
  if (flagcxWrapIbvSymbols() != flagcxSuccess) {
    return flagcxInternalError;
  }

  if (flagcxNIbDevs == -1) {
    pthread_mutex_lock(&flagcxIbLock);
    flagcxWrapIbvForkInit();
    if (flagcxNIbDevs == -1) {
      flagcxNIbDevs = 0;
      flagcxNMergedIbDevs = 0;
      if (flagcxFindInterfaces(flagcxIbIfName, &flagcxIbIfAddr,
                               MAX_IF_NAME_SIZE, 1) != 1) {
        WARN("NET/IBUC : No IP interface found.");
        ret = flagcxInternalError;
        goto fail;
      }

      // Detect IB cards
      int nIbucDevs;
      struct ibv_device **devices;

      // Check if user defined which IBUC device:port to use
      char *userIbucEnv = getenv("FLAGCX_IB_HCA");
      if (userIbucEnv != NULL && shownIbucHcaEnv++ == 0)
        INFO(FLAGCX_NET | FLAGCX_ENV, "FLAGCX_IB_HCA set to %s", userIbucEnv);
      struct netIf userIfs[MAX_IB_DEVS];
      bool searchNot = userIbucEnv && userIbucEnv[0] == '^';
      if (searchNot)
        userIbucEnv++;
      bool searchExact = userIbucEnv && userIbucEnv[0] == '=';
      if (searchExact)
        userIbucEnv++;
      int nUserIfs = parseStringList(userIbucEnv, userIfs, MAX_IB_DEVS);

      if (flagcxSuccess != flagcxWrapIbvGetDeviceList(&devices, &nIbucDevs)) {
        ret = flagcxInternalError;
        goto fail;
      }

      for (int d = 0; d < nIbucDevs && flagcxNIbDevs < MAX_IB_DEVS; d++) {
        struct ibv_context *context;
        if (flagcxSuccess != flagcxWrapIbvOpenDevice(&context, devices[d]) ||
            context == NULL) {
          WARN("NET/IBUC : Unable to open device %s", devices[d]->name);
          continue;
        }
        int nPorts = 0;
        struct ibv_device_attr devAttr;
        memset(&devAttr, 0, sizeof(devAttr));
        if (flagcxSuccess != flagcxWrapIbvQueryDevice(context, &devAttr)) {
          WARN("NET/IBUC : Unable to query device %s", devices[d]->name);
          if (flagcxSuccess != flagcxWrapIbvCloseDevice(context)) {
            ret = flagcxInternalError;
            goto fail;
          }
          continue;
        }
        for (int port_num = 1; port_num <= devAttr.phys_port_cnt; port_num++) {
          struct ibv_port_attr portAttr;
          if (flagcxSuccess !=
              flagcxWrapIbvQueryPort(context, port_num, &portAttr)) {
            WARN("NET/IBUC : Unable to query port_num %d", port_num);
            continue;
          }
          if (portAttr.state != IBV_PORT_ACTIVE)
            continue;
          if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND &&
              portAttr.link_layer != IBV_LINK_LAYER_ETHERNET)
            continue;

          // check against user specified HCAs/ports
          if (!(matchIfList(devices[d]->name, port_num, userIfs, nUserIfs,
                            searchExact) ^
                searchNot)) {
            continue;
          }
          pthread_mutex_init(&flagcxIbDevs[flagcxNIbDevs].lock, NULL);
          flagcxIbDevs[flagcxNIbDevs].device = d;
          flagcxIbDevs[flagcxNIbDevs].guid = devAttr.sys_image_guid;
          flagcxIbDevs[flagcxNIbDevs].portAttr = portAttr;
          flagcxIbDevs[flagcxNIbDevs].lid = portAttr.lid;
          flagcxIbDevs[flagcxNIbDevs].portNum = port_num;
          flagcxIbDevs[flagcxNIbDevs].link = portAttr.link_layer;
          flagcxIbDevs[flagcxNIbDevs].speed =
              flagcxIbSpeed(portAttr.active_speed) *
              flagcxIbWidth(portAttr.active_width);
          flagcxIbDevs[flagcxNIbDevs].context = context;
          flagcxIbDevs[flagcxNIbDevs].pdRefs = 0;
          flagcxIbDevs[flagcxNIbDevs].pd = NULL;
          strncpy(flagcxIbDevs[flagcxNIbDevs].devName, devices[d]->name,
                  MAXNAMESIZE);
          FLAGCXCHECK(
              flagcxIbGetPciPath(flagcxIbDevs[flagcxNIbDevs].devName,
                                 &flagcxIbDevs[flagcxNIbDevs].pciPath,
                                 &flagcxIbDevs[flagcxNIbDevs].realPort));
          flagcxIbDevs[flagcxNIbDevs].maxQp = devAttr.max_qp;
          flagcxIbDevs[flagcxNIbDevs].maxQpRdAtomic = devAttr.max_qp_rd_atom;
          flagcxIbDevs[flagcxNIbDevs].maxQpInitRdAtomic =
              devAttr.max_qp_init_rd_atom;
          flagcxIbDevs[flagcxNIbDevs].mrCache.capacity = 0;
          flagcxIbDevs[flagcxNIbDevs].mrCache.population = 0;
          flagcxIbDevs[flagcxNIbDevs].mrCache.slots = NULL;

          // Enable ADAPTIVE_ROUTING by default on IBUC networks
          // But allow it to be overloaded by an env parameter
          flagcxIbDevs[flagcxNIbDevs].ar =
              (portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND) ? 1 : 0;
          if (flagcxParamIbAdaptiveRouting() != -2)
            flagcxIbDevs[flagcxNIbDevs].ar = flagcxParamIbAdaptiveRouting();

          TRACE(
              FLAGCX_NET,
              "NET/IBUC: [%d] %s:%s:%d/%s speed=%d context=%p pciPath=%s ar=%d",
              d, devices[d]->name, devices[d]->dev_name,
              flagcxIbDevs[flagcxNIbDevs].portNum,
              portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND ? "IB" : "RoCE",
              flagcxIbDevs[flagcxNIbDevs].speed, context,
              flagcxIbDevs[flagcxNIbDevs].pciPath,
              flagcxIbDevs[flagcxNIbDevs].ar);

          pthread_create(&flagcxIbAsyncThread, NULL, flagcxIbAsyncThreadMain,
                         flagcxIbDevs + flagcxNIbDevs);
          flagcxSetThreadName(flagcxIbAsyncThread, "FLAGCX IbucAsync %2d",
                              flagcxNIbDevs);
          pthread_detach(flagcxIbAsyncThread); // will not be pthread_join()'d

          int mergedDev = flagcxNMergedIbDevs;
          if (flagcxParamIbMergeNics()) {
            mergedDev = flagcxIbFindMatchingDev(flagcxNIbDevs);
          }

          // No matching dev found, create new mergedDev entry (it's okay if
          // there's only one dev inside)
          if (mergedDev == flagcxNMergedIbDevs) {
            // Set ndevs to 1, assign first ibDevN to the current IBUC device
            flagcxIbMergedDevs[mergedDev].ndevs = 1;
            flagcxIbMergedDevs[mergedDev].devs[0] = flagcxNIbDevs;
            flagcxNMergedIbDevs++;
            strncpy(flagcxIbMergedDevs[mergedDev].devName,
                    flagcxIbDevs[flagcxNIbDevs].devName, MAXNAMESIZE);
            // Matching dev found, edit name
          } else {
            // Set next device in this array to the current IBUC device
            int ndevs = flagcxIbMergedDevs[mergedDev].ndevs;
            flagcxIbMergedDevs[mergedDev].devs[ndevs] = flagcxNIbDevs;
            flagcxIbMergedDevs[mergedDev].ndevs++;
            snprintf(flagcxIbMergedDevs[mergedDev].devName +
                         strlen(flagcxIbMergedDevs[mergedDev].devName),
                     MAXNAMESIZE + 1, "+%s",
                     flagcxIbDevs[flagcxNIbDevs].devName);
          }

          // Aggregate speed
          flagcxIbMergedDevs[mergedDev].speed +=
              flagcxIbDevs[flagcxNIbDevs].speed;
          flagcxNIbDevs++;
          nPorts++;
        }
        if (nPorts == 0 && flagcxSuccess != flagcxWrapIbvCloseDevice(context)) {
          ret = flagcxInternalError;
          goto fail;
        }
      }
      if (nIbucDevs &&
          (flagcxSuccess != flagcxWrapIbvFreeDeviceList(devices))) {
        ret = flagcxInternalError;
        goto fail;
      };
    }
    if (flagcxNIbDevs == 0) {
      INFO(FLAGCX_INIT | FLAGCX_NET, "NET/IBUC : No device found.");
    } else {
      char line[2048];
      line[0] = '\0';
      // Determine whether RELAXED_ORDERING is enabled and possible
      flagcxIbRelaxedOrderingEnabled = flagcxIbRelaxedOrderingCapable();
      for (int d = 0; d < flagcxNMergedIbDevs; d++) {
        struct flagcxIbMergedDev *mergedDev = flagcxIbMergedDevs + d;
        if (mergedDev->ndevs > 1) {
          // Print out merged dev info
          snprintf(line + strlen(line), 2047 - strlen(line), " [%d]={", d);
          for (int i = 0; i < mergedDev->ndevs; i++) {
            int ibucDev = mergedDev->devs[i];
            snprintf(line + strlen(line), 2047 - strlen(line),
                     "[%d] %s:%d/%s%s", ibucDev, flagcxIbDevs[ibucDev].devName,
                     flagcxIbDevs[ibucDev].portNum,
                     flagcxIbDevs[ibucDev].link == IBV_LINK_LAYER_INFINIBAND
                         ? "IB"
                         : "RoCE",
                     // Insert comma to delineate
                     i == (mergedDev->ndevs - 1) ? "" : ", ");
          }
          snprintf(line + strlen(line), 2047 - strlen(line), "}");
        } else {
          int ibucDev = mergedDev->devs[0];
          snprintf(line + strlen(line), 2047 - strlen(line), " [%d]%s:%d/%s",
                   ibucDev, flagcxIbDevs[ibucDev].devName,
                   flagcxIbDevs[ibucDev].portNum,
                   flagcxIbDevs[ibucDev].link == IBV_LINK_LAYER_INFINIBAND
                       ? "IB"
                       : "RoCE");
        }
      }
      line[2047] = '\0';
      char addrline[SOCKET_NAME_MAXLEN + 1];
      INFO(FLAGCX_NET, "NET/IBUC : Using%s %s; OOB %s:%s", line,
           flagcxIbRelaxedOrderingEnabled ? "[RO]" : "", flagcxIbIfName,
           flagcxSocketToString(&flagcxIbIfAddr, addrline));
    }
    pthread_mutex_unlock(&flagcxIbLock);
  }
  return flagcxSuccess;
fail:
  pthread_mutex_unlock(&flagcxIbLock);
  return ret;
}

flagcxResult_t flagcxIbucMalloc(void **ptr, size_t size);
flagcxResult_t flagcxIbucCloseSend(void *sendComm);
flagcxResult_t flagcxIbucCloseRecv(void *recvComm);
static flagcxResult_t flagcxIbucCleanupSend(struct flagcxIbSendComm *comm,
                                            bool *released);
static flagcxResult_t flagcxIbucCleanupRecv(struct flagcxIbRecvComm *comm,
                                            bool *released);
static void flagcxIbucRetainDeferredCleanup(struct flagcxIbNetCommBase *base);
static void flagcxIbucDrainDeferredCleanup(void);
static void
flagcxIbucRetainAndRetryDeferredCleanup(struct flagcxIbNetCommBase *base);
flagcxResult_t flagcxIbucCreateQpWithType(uint8_t ib_port,
                                          struct flagcxIbNetCommDevBase *base,
                                          int access_flags,
                                          enum ibv_qp_type qp_type,
                                          struct flagcxIbQp *qp);
flagcxResult_t flagcxIbucRtrQpWithType(struct ibv_qp *qp, uint8_t sGidIndex,
                                       uint32_t dest_qp_num,
                                       struct flagcxIbDevInfo *info,
                                       enum ibv_qp_type qp_type);
flagcxResult_t flagcxIbucRtsQpWithType(struct ibv_qp *qp,
                                       enum ibv_qp_type qp_type);

flagcxResult_t flagcxIbucMalloc(void **ptr, size_t size) {
  *ptr = malloc(size);
  if (*ptr == NULL)
    return flagcxInternalError;
  memset(*ptr, 0, size);
  return flagcxSuccess;
}

static flagcxResult_t
flagcxIbucProgressAbortAccept(struct flagcxIbListenComm *lComm) {
  struct flagcxIbCommStage *stage = &lComm->stage;
  struct flagcxIbRecvComm *rComm = (struct flagcxIbRecvComm *)stage->comm;
  if (rComm == NULL)
    return flagcxInternalError;

  flagcxResult_t failure = rComm->base.asyncResult;
  bool released = false;
  flagcxResult_t cleanup = flagcxIbucCleanupRecv(rComm, &released);
  if (!released)
    flagcxIbucRetainAndRetryDeferredCleanup(&rComm->base);
  stage->comm = NULL;
  stage->state = flagcxIbCommStateStart;
  if (cleanup != flagcxSuccess)
    WARN("NET/IBUC : receive setup failed with result %d and deferred cleanup "
         "with result %d",
         failure, cleanup);
  return failure != flagcxSuccess ? failure : cleanup;
}

static flagcxResult_t flagcxIbucAbortAccept(struct flagcxIbListenComm *lComm,
                                            struct flagcxIbRecvComm *rComm,
                                            flagcxResult_t failure) {
  struct flagcxIbCommStage *stage = &lComm->stage;
  free(stage->buffer);
  stage->buffer = NULL;
  stage->offset = 0;
  rComm->base.asyncResult = failure;
  stage->comm = rComm;
  return flagcxIbucProgressAbortAccept(lComm);
}

static flagcxResult_t
flagcxIbucProgressAbortConnect(struct flagcxIbHandle *handle) {
  struct flagcxIbCommStage *stage = &handle->stage;
  struct flagcxIbSendComm *comm = (struct flagcxIbSendComm *)stage->comm;
  if (comm == NULL)
    return flagcxInternalError;

  flagcxResult_t failure = comm->base.asyncResult;
  bool released = false;
  flagcxResult_t cleanup = flagcxIbucCleanupSend(comm, &released);
  if (!released)
    flagcxIbucRetainAndRetryDeferredCleanup(&comm->base);
  stage->comm = NULL;
  stage->state = flagcxIbCommStateStart;
  if (cleanup != flagcxSuccess)
    WARN("NET/IBUC : send setup failed with result %d and deferred cleanup "
         "with result %d",
         failure, cleanup);
  return failure != flagcxSuccess ? failure : cleanup;
}

static flagcxResult_t flagcxIbucAbortConnect(struct flagcxIbHandle *handle,
                                             struct flagcxIbSendComm *comm,
                                             flagcxResult_t failure) {
  struct flagcxIbCommStage *stage = &handle->stage;
  free(stage->buffer);
  stage->buffer = NULL;
  stage->offset = 0;
  comm->base.asyncResult = failure;
  stage->comm = comm;
  return flagcxIbucProgressAbortConnect(handle);
}

static void flagcxIbucAddEvent(struct flagcxIbRequest *req, int devIndex,
                               struct flagcxIbNetCommDevBase *base) {
  req->events[devIndex]++;
  req->devBases[devIndex] = base;
}

static void flagcxIbucAddDataEvent(struct flagcxIbRequest *req, int devIndex,
                                   struct flagcxIbNetCommDevBase *base) {
  flagcxIbucAddEvent(req, devIndex, base);
  req->dataEvents[devIndex]++;
}

flagcxResult_t flagcxIbucInitCommDevBase(int ibDevN,
                                         struct flagcxIbNetCommDevBase *base) {
  if (base == NULL || ibDevN < 0 || ibDevN >= flagcxNIbDevs)
    return flagcxInvalidArgument;

  base->ibDevN = ibDevN;
  flagcxIbDev *ibucDev = flagcxIbDevs + ibDevN;
  pthread_mutex_lock(&ibucDev->lock);
  if (ibucDev->pdRefs == 0) {
    flagcxResult_t result =
        flagcxWrapIbvAllocPd(&ibucDev->pd, ibucDev->context);
    if (result != flagcxSuccess) {
      pthread_mutex_unlock(&ibucDev->lock);
      return result;
    }
  }
  ibucDev->pdRefs++;
  base->pd = ibucDev->pd;
  pthread_mutex_unlock(&ibucDev->lock);

  // Recv requests can generate 2 completions (one for the post FIFO, one for
  // the Recv).
  flagcxResult_t result = flagcxWrapIbvCreateCq(
      &base->cq, ibucDev->context, 2 * MAX_REQUESTS * flagcxParamIbQpsPerConn(),
      NULL, NULL, 0);
  if (result != flagcxSuccess) {
    bool retainPdForRetry = false;
    pthread_mutex_lock(&ibucDev->lock);
    ibucDev->pdRefs--;
    if (ibucDev->pdRefs == 0) {
      flagcxResult_t deallocResult = flagcxWrapIbvDeallocPd(ibucDev->pd);
      if (deallocResult != flagcxSuccess)
        WARN("NET/IBUC : failed to deallocate PD while rolling back CQ "
             "creation");
      if (deallocResult != flagcxSuccess) {
        // Preserve ownership so staged-comm cleanup can retry deallocation.
        ibucDev->pdRefs = 1;
        retainPdForRetry = true;
      } else {
        ibucDev->pd = NULL;
      }
    }
    // The CQ failure rolled back this base's reference even when the PD is
    // still shared by another communicator. Keep the pointer only when PD
    // deallocation itself failed and the reference was restored for retry.
    if (!retainPdForRetry)
      base->pd = NULL;
    pthread_mutex_unlock(&ibucDev->lock);
    return result;
  }

  return flagcxSuccess;
}

flagcxResult_t flagcxIbucDestroyBase(struct flagcxIbNetCommDevBase *base) {
  if (base == NULL)
    return flagcxSuccess;
  flagcxResult_t result = flagcxSuccess;

  // Poll any remaining completions before destroying CQ
  if (base->cq) {
    struct ibv_wc wcs[64];
    int nCqe = 0;
    // Poll multiple times to drain all pending completions
    for (int i = 0; i < 16; i++) {
      flagcxWrapIbvPollCq(base->cq, 64, wcs, &nCqe);
      if (nCqe == 0)
        break;
    }
    flagcxResult_t cqResult = flagcxWrapIbvDestroyCq(base->cq);
    if (result == flagcxSuccess && cqResult != flagcxSuccess)
      result = cqResult;
    if (cqResult == flagcxSuccess)
      base->cq = NULL;
  }

  // A live CQ still references the context/PD. Preserve the remaining base
  // ownership so close can retry in dependency order.
  if (base->cq != NULL)
    return result;
  if (base->pd == NULL)
    return result;
  if (base->ibDevN < 0 || base->ibDevN >= flagcxNIbDevs)
    return result == flagcxSuccess ? flagcxInternalError : result;

  flagcxIbDev *ibucDev = flagcxIbDevs + base->ibDevN;
  pthread_mutex_lock(&ibucDev->lock);
  if (ibucDev->pdRefs <= 0) {
    if (result == flagcxSuccess)
      result = flagcxInternalError;
  } else if (--ibucDev->pdRefs == 0) {
    flagcxResult_t pdResult = flagcxWrapIbvDeallocPd(base->pd);
    if (result == flagcxSuccess && pdResult != flagcxSuccess)
      result = pdResult;
    if (pdResult == flagcxSuccess) {
      ibucDev->pd = NULL;
    } else {
      // Keep the reference and pointer live so a later close can retry.
      ibucDev->pdRefs = 1;
    }
  }
  pthread_mutex_unlock(&ibucDev->lock);
  if (result == flagcxSuccess)
    base->pd = NULL;
  return result;
}

flagcxResult_t flagcxIbucCreateQp(uint8_t ib_port,
                                  struct flagcxIbNetCommDevBase *base,
                                  int access_flags, struct flagcxIbQp *qp) {
  return flagcxIbucCreateQpWithType(ib_port, base, access_flags, IBV_QPT_UC,
                                    qp);
}

static flagcxResult_t flagcxIbucCreateQpWithTypeCq(
    uint8_t ib_port, struct flagcxIbNetCommDevBase *base, int access_flags,
    enum ibv_qp_type qp_type, struct ibv_cq *sendCq, struct ibv_cq *recvCq,
    uint32_t requiredInline, struct flagcxIbQp *qp) {
  struct ibv_qp_init_attr qpInitAttr;
  memset(&qpInitAttr, 0, sizeof(struct ibv_qp_init_attr));
  qpInitAttr.send_cq = sendCq;
  qpInitAttr.recv_cq = recvCq;
  qpInitAttr.qp_type = qp_type;
  // One zero-SGE receive credit is consumed per live logical request on each
  // UC data QP. The request pool bounds that number, so asking providers for
  // the larger legacy SRQ depth only wastes QP resources and is not portable.
  qpInitAttr.cap.max_recv_wr = MAX_REQUESTS;

  // We might send 2 messages per send (RDMA and RDMA_WITH_IMM)
  qpInitAttr.cap.max_send_wr = 2 * MAX_REQUESTS;
  // Retransmission SENDs carry a small protocol header plus one payload SGE.
  // UC data and the auxiliary RC flush QP continue to use a single SGE.
  qpInitAttr.cap.max_send_sge = qp_type == IBV_QPT_RC ? 2 : 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.cap.max_inline_data = std::max<uint32_t>(
      requiredInline,
      flagcxParamIbUseInline() ? sizeof(struct flagcxIbSendFifo) : 0);
  FLAGCXCHECK(flagcxWrapIbvCreateQp(&qp->qp, base->pd, &qpInitAttr));

  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = flagcxParamIbPkey();
  qpAttr.port_num = ib_port;
  qpAttr.qp_access_flags = access_flags;
  FLAGCXCHECK(flagcxWrapIbvModifyQp(qp->qp, &qpAttr,
                                    IBV_QP_STATE | IBV_QP_PKEY_INDEX |
                                        IBV_QP_PORT | IBV_QP_ACCESS_FLAGS));
  return flagcxSuccess;
}

flagcxResult_t flagcxIbucCreateQpWithType(uint8_t ib_port,
                                          struct flagcxIbNetCommDevBase *base,
                                          int access_flags,
                                          enum ibv_qp_type qp_type,
                                          struct flagcxIbQp *qp) {
  return flagcxIbucCreateQpWithTypeCq(ib_port, base, access_flags, qp_type,
                                      base->cq, base->cq, 0, qp);
}

flagcxResult_t flagcxIbucRtrQp(struct ibv_qp *qp, uint8_t sGidIndex,
                               uint32_t dest_qp_num,
                               struct flagcxIbDevInfo *info) {
  return flagcxIbucRtrQpWithType(qp, sGidIndex, dest_qp_num, info, IBV_QPT_UC);
}

flagcxResult_t flagcxIbucRtrQpWithType(struct ibv_qp *qp, uint8_t sGidIndex,
                                       uint32_t dest_qp_num,
                                       struct flagcxIbDevInfo *info,
                                       enum ibv_qp_type qp_type) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = info->mtu;
  qpAttr.dest_qp_num = dest_qp_num;
  qpAttr.rq_psn = 0;

  // For RC mode, we need additional parameters
  if (qp_type == IBV_QPT_RC) {
    qpAttr.max_dest_rd_atomic = 1;
    qpAttr.min_rnr_timer = 12;
  }
  if (info->linkLayer == IBV_LINK_LAYER_ETHERNET) {
    qpAttr.ah_attr.is_global = 1;
    qpAttr.ah_attr.grh.dgid.global.subnet_prefix = info->spn;
    qpAttr.ah_attr.grh.dgid.global.interface_id = info->iid;
    qpAttr.ah_attr.grh.flow_label = 0;
    qpAttr.ah_attr.grh.sgid_index = sGidIndex;
    qpAttr.ah_attr.grh.hop_limit = 255;
    qpAttr.ah_attr.grh.traffic_class = flagcxParamIbTc();
  } else {
    qpAttr.ah_attr.is_global = 0;
    qpAttr.ah_attr.dlid = info->lid;
  }
  qpAttr.ah_attr.sl = flagcxParamIbSl();
  qpAttr.ah_attr.src_path_bits = 0;
  qpAttr.ah_attr.port_num = info->ibPort;
  int modifyFlags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                    IBV_QP_DEST_QPN | IBV_QP_RQ_PSN;
  if (qp_type == IBV_QPT_RC) {
    modifyFlags |= IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
  }
  FLAGCXCHECK(flagcxWrapIbvModifyQp(qp, &qpAttr, modifyFlags));
  return flagcxSuccess;
}

flagcxResult_t flagcxIbucRtsQp(struct ibv_qp *qp) {
  return flagcxIbucRtsQpWithType(qp, IBV_QPT_UC);
}

flagcxResult_t flagcxIbucRtsQpWithType(struct ibv_qp *qp,
                                       enum ibv_qp_type qp_type) {
  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(struct ibv_qp_attr));
  qpAttr.qp_state = IBV_QPS_RTS;
  qpAttr.sq_psn = 0;

  // For RC mode, we need additional parameters
  if (qp_type == IBV_QPT_RC) {
    qpAttr.timeout = flagcxParamIbTimeout();
    qpAttr.retry_cnt = flagcxParamIbRetryCnt();
    qpAttr.rnr_retry = 7;
    qpAttr.max_rd_atomic = 1;
  }

  int modifyFlags = IBV_QP_STATE | IBV_QP_SQ_PSN;
  if (qp_type == IBV_QPT_RC) {
    modifyFlags |= IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                   IBV_QP_MAX_QP_RD_ATOMIC;
  }
  FLAGCXCHECK(flagcxWrapIbvModifyQp(qp, &qpAttr, modifyFlags));
  return flagcxSuccess;
}

static flagcxResult_t
flagcxIbucPostRetransRecv(struct flagcxIbRecvCommDev *commDev,
                          uint32_t creditIndex) {
  if (commDev == NULL || commDev->retransQp.qp == NULL ||
      commDev->retransRecvMr == NULL ||
      creditIndex >= (uint32_t)commDev->retransRecvBufCount ||
      commDev->retransRecvBufs[creditIndex] == NULL)
    return flagcxInvalidArgument;
  struct ibv_sge sge = {};
  sge.addr = (uint64_t)commDev->retransRecvBufs[creditIndex];
  sge.length = flagcxIbucRetransRecvEntrySize;
  sge.lkey = commDev->retransRecvMr->lkey;
  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = FLAGCX_IBUC_RETRANS_RECV_WR_ID_PREFIX | creditIndex;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  struct ibv_recv_wr *badWr = NULL;
  return flagcxWrapIbvPostRecv(commDev->retransQp.qp, &wr, &badWr);
}

static flagcxResult_t flagcxIbucPostAckRecv(struct flagcxIbSendCommDev *commDev,
                                            uint32_t creditIndex) {
  if (commDev == NULL || commDev->retransQp.qp == NULL ||
      commDev->ackMr == NULL || commDev->ackBuffer == NULL ||
      creditIndex >= FLAGCX_IB_ACK_BUF_COUNT)
    return flagcxInvalidArgument;
  const size_t entrySize =
      sizeof(struct flagcxIbAckMsg) + FLAGCX_IB_ACK_BUF_PADDING;
  struct ibv_sge sge = {};
  sge.addr = (uint64_t)((char *)commDev->ackBuffer + creditIndex * entrySize +
                        FLAGCX_IB_ACK_BUF_PADDING);
  sge.length = sizeof(struct flagcxIbAckMsg);
  sge.lkey = commDev->ackMr->lkey;
  struct ibv_recv_wr wr = {};
  wr.wr_id = flagcxIbucAckRecvWrIdPrefix | creditIndex;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  struct ibv_recv_wr *badWr = NULL;
  return flagcxWrapIbvPostRecv(commDev->retransQp.qp, &wr, &badWr);
}

static flagcxResult_t flagcxIbucPostDataRecv(struct flagcxIbRecvComm *comm,
                                             uint32_t qpIndex) {
  if (comm == NULL || qpIndex >= (uint32_t)comm->base.nqps ||
      comm->base.qps[qpIndex].qp == NULL)
    return flagcxInvalidArgument;
  struct ibv_recv_wr wr = {};
  wr.wr_id = FLAGCX_IBUC_DATA_RECV_WR_ID_PREFIX | qpIndex;
  wr.sg_list = NULL;
  wr.num_sge = 0;
  struct ibv_recv_wr *badWr = NULL;
  return flagcxWrapIbvPostRecv(comm->base.qps[qpIndex].qp, &wr, &badWr);
}

flagcxResult_t flagcxIbucListen(int dev, void *opaqueHandle,
                                void **listenComm) {
  struct flagcxIbListenComm *comm;
  FLAGCXCHECK(flagcxCalloc(&comm, 1));
  struct flagcxIbHandle *handle = (struct flagcxIbHandle *)opaqueHandle;
  memset(handle, 0, sizeof(struct flagcxIbHandle));
  comm->dev = dev;
  handle->magic = FLAGCX_SOCKET_MAGIC;
  FLAGCXCHECK(flagcxSocketInit(&comm->sock, &flagcxIbIfAddr, handle->magic,
                               flagcxSocketTypeNetIb, NULL, 1));
  FLAGCXCHECK(flagcxSocketListen(&comm->sock));
  FLAGCXCHECK(flagcxSocketGetAddr(&comm->sock, &handle->connectAddr));
  *listenComm = comm;
  return flagcxSuccess;
}

flagcxResult_t flagcxIbucConnect(int dev, void *opaqueHandle, void **sendComm) {
  struct flagcxIbHandle *handle = (struct flagcxIbHandle *)opaqueHandle;
  struct flagcxIbCommStage *stage = &handle->stage;
  struct flagcxIbSendComm *comm = (struct flagcxIbSendComm *)stage->comm;
  int ready;
  flagcxResult_t retransResult;
  *sendComm = NULL;

  if (stage->state == flagcxIbCommStateConnect)
    goto ibuc_connect_check;
  if (stage->state == flagcxIbCommStateSend)
    goto ibuc_send;
  if (stage->state == flagcxIbCommStateConnecting)
    goto ibuc_connect;
  if (stage->state == flagcxIbCommStateConnected)
    goto ibuc_send_ready;
  if (stage->state != flagcxIbCommStateStart) {
    WARN("Error: trying to connect already connected sendComm");
    return flagcxInternalError;
  }

  FLAGCXCHECK(
      flagcxIbucMalloc((void **)&comm, sizeof(struct flagcxIbSendComm)));
  FLAGCXCHECK(flagcxSocketInit(&comm->base.sock, &handle->connectAddr,
                               handle->magic, flagcxSocketTypeNetIb, NULL, 1));
  stage->comm = comm;
  stage->state = flagcxIbCommStateConnect;
  FLAGCXCHECK(flagcxSocketConnect(&comm->base.sock));

ibuc_connect_check:
  /* since flagcxSocketConnect is async, we must check if connection is complete
   */
  FLAGCXCHECK(flagcxSocketReady(&comm->base.sock, &ready));
  if (!ready)
    return flagcxSuccess;

  // IBUC Setup
  struct flagcxIbMergedDev *mergedDev;
  mergedDev = flagcxIbMergedDevs + dev;
  comm->base.ndevs = mergedDev->ndevs;
  comm->base.nqps = flagcxParamIbQpsPerConn() *
                    comm->base.ndevs; // We must have at least 1 qp per-device
  comm->base.isSend = true;

  // Init PD, Ctx for each IB device
  comm->ar = 1; // Set to 1 for logic
  for (int i = 0; i < mergedDev->ndevs; i++) {
    int ibDevN = mergedDev->devs[i];
    FLAGCXCHECK(flagcxIbucInitCommDevBase(ibDevN, &comm->devs[i].base));
    comm->ar = comm->ar &&
               flagcxIbDevs[dev]
                   .ar; // ADAPTIVE_ROUTING - if all merged devs have it enabled
  }

  struct flagcxIbConnectionMetadata meta;
  memset(&meta, 0, sizeof(meta));
  meta.ndevs = comm->base.ndevs;

  // Alternate QPs between devices
  int devIndex;
  devIndex = 0;
  for (int q = 0; q < comm->base.nqps; q++) {
    flagcxIbSendCommDev *commDev = comm->devs + devIndex;
    flagcxIbDev *ibucDev = flagcxIbDevs + commDev->base.ibDevN;
    FLAGCXCHECK(flagcxIbucCreateQp(ibucDev->portNum, &commDev->base,
                                   IBV_ACCESS_REMOTE_WRITE,
                                   comm->base.qps + q));
    comm->base.qps[q].devIndex = devIndex;
    meta.qpInfo[q].qpn = comm->base.qps[q].qp->qp_num;
    meta.qpInfo[q].devIndex = comm->base.qps[q].devIndex;

    // Query ece capabilities (enhanced connection establishment)
    FLAGCXCHECK(flagcxWrapIbvQueryEce(comm->base.qps[q].qp, &meta.qpInfo[q].ece,
                                      &meta.qpInfo[q].eceSupported));
    devIndex = (devIndex + 1) % comm->base.ndevs;
  }

  // IBUC always enables retransmission, ignore environment variable
  meta.retransEnabled = 1;

  for (int i = 0; i < comm->base.ndevs; i++) {
    flagcxIbSendCommDev *commDev = comm->devs + i;
    flagcxIbDev *ibucDev = flagcxIbDevs + commDev->base.ibDevN;

    // Write to the metadata struct via this pointer
    flagcxIbDevInfo *devInfo = meta.devs + i;
    devInfo->ibPort = ibucDev->portNum;
    devInfo->mtu = ibucDev->portAttr.active_mtu;
    devInfo->lid = ibucDev->lid;

    // Prepare my fifo
    FLAGCXCHECK(
        flagcxWrapIbvRegMr(&commDev->fifoMr, commDev->base.pd, comm->fifo,
                           sizeof(struct flagcxIbSendFifo) * MAX_REQUESTS *
                               FLAGCX_NET_IB_MAX_RECVS,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                               IBV_ACCESS_REMOTE_READ));
    devInfo->fifoRkey = commDev->fifoMr->rkey;

    // RoCE uses a GID/GRH route; native InfiniBand uses the LID route above.
    devInfo->linkLayer = commDev->base.gidInfo.linkLayer =
        ibucDev->portAttr.link_layer;
    if (devInfo->linkLayer == IBV_LINK_LAYER_ETHERNET) {
      FLAGCXCHECK(flagcxIbGetGidIndex(ibucDev->context, ibucDev->portNum,
                                      ibucDev->portAttr.gid_tbl_len,
                                      &commDev->base.gidInfo.localGidIndex));
      FLAGCXCHECK(flagcxWrapIbvQueryGid(ibucDev->context, ibucDev->portNum,
                                        commDev->base.gidInfo.localGidIndex,
                                        &commDev->base.gidInfo.localGid));
      devInfo->spn = commDev->base.gidInfo.localGid.global.subnet_prefix;
      devInfo->iid = commDev->base.gidInfo.localGid.global.interface_id;
    } else {
      commDev->base.gidInfo.localGidIndex = 0;
      memset(&commDev->base.gidInfo.localGid, 0, sizeof(union ibv_gid));
    }

    if (meta.retransEnabled) {
      flagcxResult_t retransSetupResult = flagcxWrapIbvCreateCq(
          &commDev->retransCq, ibucDev->context,
          2 * MAX_REQUESTS + FLAGCX_IB_ACK_BUF_COUNT, NULL, NULL, 0);
      if (retransSetupResult != flagcxSuccess)
        return flagcxIbucAbortConnect(handle, comm, retransSetupResult);
      retransSetupResult = flagcxIbucCreateQpWithTypeCq(
          ibucDev->portNum, &commDev->base, IBV_ACCESS_REMOTE_WRITE, IBV_QPT_RC,
          commDev->retransCq, commDev->retransCq, sizeof(struct flagcxIbAckMsg),
          &commDev->retransQp);
      if (retransSetupResult != flagcxSuccess)
        return flagcxIbucAbortConnect(handle, comm, retransSetupResult);
      commDev->retransQp.devIndex = i;
      commDev->retransQp.remDevIdx = i;
      meta.retransQpn[i] = commDev->retransQp.qp->qp_num;
      flagcxResult_t retransHdrResult = flagcxWrapIbvRegMr(
          &commDev->retransHdrMr, commDev->base.pd, comm->retransHdrPool,
          sizeof(comm->retransHdrPool), IBV_ACCESS_LOCAL_WRITE);
      if (retransHdrResult != flagcxSuccess)
        return flagcxIbucAbortConnect(handle, comm, retransHdrResult);

      flagcxResult_t ctrlResult =
          flagcxIbCreateCtrlQp(ibucDev->context, commDev->base.pd,
                               ibucDev->portNum, &commDev->ctrlQp);
      if (ctrlResult != flagcxSuccess)
        return flagcxIbucAbortConnect(handle, comm, ctrlResult);
      meta.ctrlQpn[i] = commDev->ctrlQp.qp->qp_num;
      meta.ctrlLid[i] = ibucDev->lid;
      meta.ctrlGid[i] = commDev->base.gidInfo.localGid;

      size_t ack_buf_size =
          (sizeof(struct flagcxIbAckMsg) + FLAGCX_IB_ACK_BUF_PADDING) *
          FLAGCX_IB_ACK_BUF_COUNT;
      commDev->ackBuffer = malloc(ack_buf_size);
      if (commDev->ackBuffer == NULL)
        return flagcxIbucAbortConnect(handle, comm, flagcxInternalError);
      flagcxResult_t ackMrResult = flagcxWrapIbvRegMr(
          &commDev->ackMr, commDev->base.pd, commDev->ackBuffer, ack_buf_size,
          IBV_ACCESS_LOCAL_WRITE);
      if (ackMrResult != flagcxSuccess)
        return flagcxIbucAbortConnect(handle, comm, ackMrResult);

      TRACE(FLAGCX_NET,
            "Send: Created control QP for dev %d: qpn=%u, link_layer=%d, "
            "lid=%u, gid=%lx:%lx",
            i, commDev->ctrlQp.qp->qp_num, devInfo->linkLayer, meta.ctrlLid[i],
            (unsigned long)meta.ctrlGid[i].global.subnet_prefix,
            (unsigned long)meta.ctrlGid[i].global.interface_id);
    }

    if (devInfo->linkLayer == IBV_LINK_LAYER_INFINIBAND) { // IB
      for (int q = 0; q < comm->base.nqps; q++) {
        // Print just the QPs for this dev
        if (comm->base.qps[q].devIndex == i)
          INFO(FLAGCX_NET,
               "NET/IBUC: %s %d IbucDev %d Port %d qpn %d mtu %d LID %d "
               "fifoRkey=0x%x fifoLkey=0x%x",
               comm->base.ndevs > 2 ? "FLAGCX MergedDev" : "FLAGCX Dev", dev,
               commDev->base.ibDevN, ibucDev->portNum, meta.qpInfo[q].qpn,
               devInfo->mtu, devInfo->lid, devInfo->fifoRkey,
               commDev->fifoMr->lkey);
      }
    } else { // RoCE
      for (int q = 0; q < comm->base.nqps; q++) {
        // Print just the QPs for this dev
        if (comm->base.qps[q].devIndex == i)
          INFO(FLAGCX_NET,
               "NET/IBUC: %s %d IbucDev %d Port %d qpn %d mtu %d "
               "query_ece={supported=%d, vendor_id=0x%x, options=0x%x, "
               "comp_mask=0x%x} GID %" PRId64 " (%" PRIX64 "/%" PRIX64
               ") fifoRkey=0x%x fifoLkey=0x%x",
               comm->base.ndevs > 2 ? "FLAGCX MergedDev" : "FLAGCX Dev", dev,
               commDev->base.ibDevN, ibucDev->portNum, meta.qpInfo[q].qpn,
               devInfo->mtu, meta.qpInfo[q].eceSupported,
               meta.qpInfo[q].ece.vendor_id, meta.qpInfo[q].ece.options,
               meta.qpInfo[q].ece.comp_mask,
               (int64_t)commDev->base.gidInfo.localGidIndex, devInfo->spn,
               devInfo->iid, devInfo->fifoRkey, commDev->fifoMr->lkey);
      }
    }
  }
  meta.fifoAddr = (uint64_t)comm->fifo;
  strncpy(meta.devName, mergedDev->devName, MAX_MERGED_DEV_NAME);

  stage->state = flagcxIbCommStateSend;
  stage->offset = 0;
  FLAGCXCHECK(flagcxIbucMalloc((void **)&stage->buffer, sizeof(meta)));

  memcpy(stage->buffer, &meta, sizeof(meta));

ibuc_send:
  FLAGCXCHECK(flagcxSocketProgress(FLAGCX_SOCKET_SEND, &comm->base.sock,
                                   stage->buffer, sizeof(meta),
                                   &stage->offset));
  if (stage->offset != sizeof(meta))
    return flagcxSuccess;

  stage->state = flagcxIbCommStateConnecting;
  stage->offset = 0;
  // Clear the staging buffer for re-use
  memset(stage->buffer, 0, sizeof(meta));

ibuc_connect:
  struct flagcxIbConnectionMetadata remMeta;
  FLAGCXCHECK(
      flagcxSocketProgress(FLAGCX_SOCKET_RECV, &comm->base.sock, stage->buffer,
                           sizeof(flagcxIbConnectionMetadata), &stage->offset));
  if (stage->offset != sizeof(remMeta))
    return flagcxSuccess;

  memcpy(&remMeta, stage->buffer, sizeof(flagcxIbConnectionMetadata));

  comm->base.nRemDevs = remMeta.ndevs;
  if (comm->base.nRemDevs != comm->base.ndevs) {
    mergedDev = flagcxIbMergedDevs + dev;
    WARN(
        "NET/IBUC : Local mergedDev=%s has a different number of devices=%d as "
        "remoteDev=%s nRemDevs=%d",
        mergedDev->devName, comm->base.ndevs, remMeta.devName,
        comm->base.nRemDevs);
  }

  int linkLayer;
  linkLayer = remMeta.devs[0].linkLayer;
  for (int i = 1; i < remMeta.ndevs; i++) {
    if (remMeta.devs[i].linkLayer != linkLayer) {
      WARN("NET/IBUC : Can't merge net devices with different linkLayer. i=%d "
           "remMeta.ndevs=%d linkLayer=%d rem_linkLayer=%d",
           i, remMeta.ndevs, linkLayer, remMeta.devs[i].linkLayer);
      return flagcxInternalError;
    }
  }

  // Copy remDevInfo for things like remGidInfo, remFifoAddr, etc.
  for (int i = 0; i < remMeta.ndevs; i++) {
    comm->base.remDevs[i] = remMeta.devs[i];
    comm->base.remDevs[i].remoteGid.global.interface_id =
        comm->base.remDevs[i].iid;
    comm->base.remDevs[i].remoteGid.global.subnet_prefix =
        comm->base.remDevs[i].spn;

    // Retain remote sizes fifo info and prepare RDMA ops
    comm->remSizesFifo.rkeys[i] = remMeta.devs[i].fifoRkey;
    comm->remSizesFifo.addr = remMeta.fifoAddr;
  }

  for (int i = 0; i < comm->base.ndevs; i++) {
    FLAGCXCHECK(
        flagcxWrapIbvRegMr(comm->remSizesFifo.mrs + i, comm->devs[i].base.pd,
                           &comm->remSizesFifo.elems,
                           sizeof(int) * MAX_REQUESTS * FLAGCX_NET_IB_MAX_RECVS,
                           IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
                               IBV_ACCESS_REMOTE_READ));

    struct flagcxIbSendCommDev *commDev = &comm->devs[i];
    const int remDevIdx = remMeta.qpInfo[i].devIndex;
    if (remDevIdx < 0 || remDevIdx >= remMeta.ndevs ||
        remMeta.retransQpn[remDevIdx] == 0)
      return flagcxIbucAbortConnect(handle, comm, flagcxInternalError);
    commDev->retransQp.remDevIdx = remDevIdx;
    flagcxResult_t retransSetupResult = flagcxIbucRtrQpWithType(
        commDev->retransQp.qp, commDev->base.gidInfo.localGidIndex,
        remMeta.retransQpn[remDevIdx], &remMeta.devs[remDevIdx], IBV_QPT_RC);
    if (retransSetupResult != flagcxSuccess)
      return flagcxIbucAbortConnect(handle, comm, retransSetupResult);
    retransSetupResult =
        flagcxIbucRtsQpWithType(commDev->retransQp.qp, IBV_QPT_RC);
    if (retransSetupResult != flagcxSuccess)
      return flagcxIbucAbortConnect(handle, comm, retransSetupResult);
  }
  comm->base.nRemDevs = remMeta.ndevs;

  for (int q = 0; q < comm->base.nqps; q++) {
    struct flagcxIbQpInfo *remQpInfo = remMeta.qpInfo + q;
    struct flagcxIbDevInfo *remDevInfo = remMeta.devs + remQpInfo->devIndex;

    // Assign per-QP remDev
    comm->base.qps[q].remDevIdx = remQpInfo->devIndex;
    int devIndex = comm->base.qps[q].devIndex;
    flagcxIbSendCommDev *commDev = comm->devs + devIndex;
    uint8_t gidIndex = commDev->base.gidInfo.localGidIndex;

    struct ibv_qp *qp = comm->base.qps[q].qp;
    if (remQpInfo->eceSupported)
      FLAGCXCHECK(
          flagcxWrapIbvSetEce(qp, &remQpInfo->ece, &remQpInfo->eceSupported));

    FLAGCXCHECK(flagcxIbucRtrQp(qp, gidIndex, remQpInfo->qpn, remDevInfo));
    FLAGCXCHECK(flagcxIbucRtsQp(qp));
  }

  if (linkLayer == IBV_LINK_LAYER_ETHERNET) { // RoCE
    for (int q = 0; q < comm->base.nqps; q++) {
      struct flagcxIbQp *qp = comm->base.qps + q;
      int ibDevN = comm->devs[qp->devIndex].base.ibDevN;
      struct flagcxIbDev *ibucDev = flagcxIbDevs + ibDevN;
      INFO(FLAGCX_NET,
           "NET/IBUC: IbucDev %d Port %d qpn %d set_ece={supported=%d, "
           "vendor_id=0x%x, options=0x%x, comp_mask=0x%x}",
           ibDevN, ibucDev->portNum, remMeta.qpInfo[q].qpn,
           remMeta.qpInfo[q].eceSupported, remMeta.qpInfo[q].ece.vendor_id,
           remMeta.qpInfo[q].ece.options, remMeta.qpInfo[q].ece.comp_mask);
    }
  }

  retransResult = flagcxIbRetransInit(&comm->retrans);
  if (retransResult != flagcxSuccess)
    return flagcxIbucAbortConnect(handle, comm, retransResult);
  comm->lastTimeoutCheckUs = 0;

  // IBUC always enables retransmission, force it on
  comm->retrans.enabled = 1;
  // Public requests retain their source buffers until this ACK arrives, so
  // delaying it for a packet-count threshold would stall low-volume traffic.
  comm->retrans.ackInterval = 1;
  if (!remMeta.retransEnabled) {
    WARN("NET/IBUC : receiver did not establish the required retransmission "
         "channel");
    return flagcxIbucAbortConnect(handle, comm, flagcxNotSupported);
  }

  if (comm->retrans.enabled) {
    INFO(FLAGCX_NET,
         "NET/IBUC Sender: Retransmission ENABLED (RTO=%uus, MaxRetry=%d, "
         "AckInterval=%d)",
         comm->retrans.minRtoUs, comm->retrans.maxRetry,
         comm->retrans.ackInterval);
  } else {
    INFO(FLAGCX_NET, "NET/IBUC Sender: Retransmission DISABLED");
  }

  if (comm->retrans.enabled && remMeta.retransEnabled) {
    bool all_ah_success = true;

    for (int i = 0; i < comm->base.ndevs; i++) {
      flagcxIbSendCommDev *commDev = &comm->devs[i];
      flagcxIbDev *ibucDev = flagcxIbDevs + commDev->base.ibDevN;

      TRACE(FLAGCX_NET,
            "Send: Setting up control QP conn for dev %d: remote_qpn=%u, "
            "remote_lid=%u, remote_gid=%lx:%lx, link_layer=%d",
            i, remMeta.ctrlQpn[i], remMeta.ctrlLid[i],
            (unsigned long)remMeta.ctrlGid[i].global.subnet_prefix,
            (unsigned long)remMeta.ctrlGid[i].global.interface_id,
            ibucDev->portAttr.link_layer);

      flagcxResult_t ah_result = flagcxIbSetupCtrlQpConnection(
          ibucDev->context, commDev->base.pd, &commDev->ctrlQp,
          remMeta.ctrlQpn[i], &remMeta.ctrlGid[i], remMeta.ctrlLid[i],
          ibucDev->portNum, ibucDev->portAttr.link_layer,
          commDev->base.gidInfo.localGidIndex);

      if (ah_result != flagcxSuccess || !commDev->ctrlQp.ah) {
        all_ah_success = false;
        break;
      }

      for (uint32_t r = 0; r < FLAGCX_IB_ACK_BUF_COUNT; r++) {
        flagcxResult_t postResult = flagcxIbucPostAckRecv(commDev, r);
        if (postResult != flagcxSuccess)
          return flagcxIbucAbortConnect(handle, comm, postResult);
      }

      TRACE(FLAGCX_NET,
            "Reliable ACK receives ready for dev %d on retrans_qpn=%u: "
            "posted %d recv WRs",
            i, commDev->retransQp.qp->qp_num, FLAGCX_IB_ACK_BUF_COUNT);
    }

    if (!all_ah_success) {
      return flagcxIbucAbortConnect(handle, comm, flagcxSystemError);
    }
  }

  comm->outstandingSends = 0;
  comm->outstandingRetrans = 0;
  comm->maxOutstanding = std::max<int>(1, flagcxParamIbMaxOutstanding());
  comm->retransUsesRc = true;

  comm->base.ready = 1;
  stage->state = flagcxIbCommStateConnected;
  stage->offset = 0;

ibuc_send_ready:
  FLAGCXCHECK(flagcxSocketProgress(FLAGCX_SOCKET_SEND, &comm->base.sock,
                                   &comm->base.ready, sizeof(int),
                                   &stage->offset));
  if (stage->offset != sizeof(int))
    return flagcxSuccess;

  free(stage->buffer);
  stage->state = flagcxIbCommStateStart;
  *sendComm = comm;
  return flagcxSuccess;
}

FLAGCX_PARAM(IbucGdrFlushDisable, "GDR_FLUSH_DISABLE", 0);

flagcxResult_t flagcxIbucAccept(void *listenComm, void **recvComm) {
  struct flagcxIbListenComm *lComm = (struct flagcxIbListenComm *)listenComm;
  struct flagcxIbCommStage *stage = &lComm->stage;
  struct flagcxIbRecvComm *rComm = (struct flagcxIbRecvComm *)stage->comm;
  int ready;
  *recvComm = NULL;

  // Pre-declare ALL variables before any goto to avoid crossing initialization
  struct flagcxIbMergedDev *mergedDev;
  struct flagcxIbDev *ibucDev;
  int ibDevN;
  struct flagcxIbRecvCommDev *rCommDev;
  struct flagcxIbDevInfo *remDevInfo;
  struct flagcxIbQp *qp;
  bool retransReady;
  flagcxResult_t retransResult;
  struct flagcxIbConnectionMetadata remMeta;
  struct flagcxIbConnectionMetadata meta;
  memset(&meta, 0,
         sizeof(meta)); // Initialize meta, including meta.retransEnabled = 0

  if (stage->state == flagcxIbCommStateAccept)
    goto ib_accept_check;
  if (stage->state == flagcxIbCommStateRecv)
    goto ib_recv;
  if (stage->state == flagcxIbCommStateSend)
    goto ibuc_send;
  if (stage->state == flagcxIbCommStatePendingReady)
    goto ib_recv_ready;
  if (stage->state != flagcxIbCommStateStart) {
    WARN("Listencomm in unknown state %d", stage->state);
    return flagcxInternalError;
  }

  FLAGCXCHECK(
      flagcxIbucMalloc((void **)&rComm, sizeof(struct flagcxIbRecvComm)));
  stage->comm = rComm;
  stage->state = flagcxIbCommStateAccept;
  FLAGCXCHECK(flagcxSocketInit(&rComm->base.sock));
  FLAGCXCHECK(flagcxSocketAccept(&rComm->base.sock, &lComm->sock));

ib_accept_check:
  FLAGCXCHECK(flagcxSocketReady(&rComm->base.sock, &ready));
  if (!ready)
    return flagcxSuccess;

  // remMeta already declared at function start
  stage->state = flagcxIbCommStateRecv;
  stage->offset = 0;
  FLAGCXCHECK(flagcxIbucMalloc((void **)&stage->buffer, sizeof(remMeta)));

ib_recv:
  FLAGCXCHECK(flagcxSocketProgress(FLAGCX_SOCKET_RECV, &rComm->base.sock,
                                   stage->buffer, sizeof(remMeta),
                                   &stage->offset));
  if (stage->offset != sizeof(remMeta))
    return flagcxSuccess;

  /* copy back the received info */
  memcpy(&remMeta, stage->buffer, sizeof(struct flagcxIbConnectionMetadata));

  mergedDev = flagcxIbMergedDevs + lComm->dev;
  rComm->base.ndevs = mergedDev->ndevs;
  rComm->base.nqps = flagcxParamIbQpsPerConn() *
                     rComm->base.ndevs; // We must have at least 1 qp per-device
  rComm->base.isSend = false;

  rComm->base.nRemDevs = remMeta.ndevs;
  if (rComm->base.nRemDevs != rComm->base.ndevs) {
    WARN(
        "NET/IBUC : Local mergedDev %s has a different number of devices=%d as "
        "remote %s %d",
        mergedDev->devName, rComm->base.ndevs, remMeta.devName,
        rComm->base.nRemDevs);
  }

  for (int i = 0; i < rComm->base.ndevs; i++) {
    rCommDev = rComm->devs + i;
    ibDevN = mergedDev->devs[i];
    FLAGCXCHECK(flagcxIbucInitCommDevBase(ibDevN, &rCommDev->base));
    ibucDev = flagcxIbDevs + ibDevN;
    FLAGCXCHECK(flagcxIbGetGidIndex(ibucDev->context, ibucDev->portNum,
                                    ibucDev->portAttr.gid_tbl_len,
                                    &rCommDev->base.gidInfo.localGidIndex));
    FLAGCXCHECK(flagcxWrapIbvQueryGid(ibucDev->context, ibucDev->portNum,
                                      rCommDev->base.gidInfo.localGidIndex,
                                      &rCommDev->base.gidInfo.localGid));
  }

  // Copy remDevInfo for things like remGidInfo, remFifoAddr, etc.
  for (int i = 0; i < remMeta.ndevs; i++) {
    rComm->base.remDevs[i] = remMeta.devs[i];
    rComm->base.remDevs[i].remoteGid.global.interface_id =
        rComm->base.remDevs[i].iid;
    rComm->base.remDevs[i].remoteGid.global.subnet_prefix =
        rComm->base.remDevs[i].spn;
  }

  if (!remMeta.retransEnabled) {
    WARN("NET/IBUC : sender did not request the required retransmission "
         "channel");
    return flagcxIbucAbortAccept(lComm, rComm, flagcxNotSupported);
  }
  meta.retransEnabled = 1;

  // Stripe QP creation across merged devs
  // Make sure to get correct remote peer dev and QP info
  int remDevIdx;
  int devIndex;
  devIndex = 0;
  for (int q = 0; q < rComm->base.nqps; q++) {
    remDevIdx = remMeta.qpInfo[q].devIndex;
    remDevInfo = remMeta.devs + remDevIdx;
    qp = rComm->base.qps + q;
    rCommDev = rComm->devs + devIndex;
    qp->remDevIdx = remDevIdx;

    // Local ibDevN
    ibDevN = rComm->devs[devIndex].base.ibDevN;
    ibucDev = flagcxIbDevs + ibDevN;

    FLAGCXCHECK(flagcxIbucCreateQpWithType(ibucDev->portNum, &rCommDev->base,
                                           IBV_ACCESS_REMOTE_WRITE, IBV_QPT_UC,
                                           qp));
    qp->devIndex = devIndex;
    devIndex = (devIndex + 1) % rComm->base.ndevs;

    // Set the ece (enhanced connection establishment) on this QP before RTR
    if (remMeta.qpInfo[q].eceSupported) {
      FLAGCXCHECK(flagcxWrapIbvSetEce(qp->qp, &remMeta.qpInfo[q].ece,
                                      &meta.qpInfo[q].eceSupported));

      if (meta.qpInfo[q].eceSupported)
        FLAGCXCHECK(flagcxWrapIbvQueryEce(qp->qp, &meta.qpInfo[q].ece,
                                          &meta.qpInfo[q].eceSupported));
    }

    FLAGCXCHECK(flagcxIbucRtrQp(qp->qp, rCommDev->base.gidInfo.localGidIndex,
                                remMeta.qpInfo[q].qpn, remDevInfo));
    FLAGCXCHECK(flagcxIbucRtsQp(qp->qp));
  }

  // UC RDMA_WRITE_WITH_IMM consumes a receive WQE even though it carries no
  // receive payload. Keep a connection-level pool of zero-SGE credits on each
  // data QP; completions are routed by the request slot and generation encoded
  // in immediate data, not by the lifetime of the WQE.
  for (int q = 0; q < rComm->base.nqps; q++) {
    for (int credit = 0; credit < MAX_REQUESTS; credit++) {
      flagcxResult_t postResult = flagcxIbucPostDataRecv(rComm, q);
      if (postResult != flagcxSuccess)
        return flagcxIbucAbortAccept(lComm, rComm, postResult);
    }
  }

  rComm->flushEnabled = 1;

  for (int i = 0; i < mergedDev->ndevs; i++) {
    rCommDev = rComm->devs + i;
    ibDevN = rCommDev->base.ibDevN;
    ibucDev = flagcxIbDevs + ibDevN;

    // Retain remote fifo info and prepare my RDMA ops
    rCommDev->fifoRkey = remMeta.devs[i].fifoRkey;
    rComm->remFifo.addr = remMeta.fifoAddr;
    FLAGCXCHECK(flagcxWrapIbvRegMr(
        &rCommDev->fifoMr, rCommDev->base.pd, &rComm->remFifo.elems,
        sizeof(struct flagcxIbSendFifo) * MAX_REQUESTS *
            FLAGCX_NET_IB_MAX_RECVS,
        IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
            IBV_ACCESS_REMOTE_READ));
    rCommDev->fifoSge.lkey = rCommDev->fifoMr->lkey;
    if (flagcxParamIbUseInline())
      rComm->remFifo.flags = IBV_SEND_INLINE;

    // Allocate Flush dummy buffer for GPU Direct RDMA
    if (rComm->flushEnabled) {
      FLAGCXCHECK(flagcxWrapIbvRegMr(&rCommDev->gpuFlush.hostMr,
                                     rCommDev->base.pd, &rComm->gpuFlushHostMem,
                                     sizeof(int), IBV_ACCESS_LOCAL_WRITE));
      rCommDev->gpuFlush.sge.addr = (uint64_t)&rComm->gpuFlushHostMem;
      rCommDev->gpuFlush.sge.length = 1;
      rCommDev->gpuFlush.sge.lkey = rCommDev->gpuFlush.hostMr->lkey;
      FLAGCXCHECK(flagcxIbucCreateQpWithType(
          ibucDev->portNum, &rCommDev->base,
          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ, IBV_QPT_RC,
          &rCommDev->gpuFlush.qp));
      struct flagcxIbDevInfo devInfo;
      devInfo.lid = ibucDev->lid;
      devInfo.linkLayer = ibucDev->portAttr.link_layer;
      devInfo.ibPort = ibucDev->portNum;
      devInfo.spn = rCommDev->base.gidInfo.localGid.global.subnet_prefix;
      devInfo.iid = rCommDev->base.gidInfo.localGid.global.interface_id;
      devInfo.mtu = ibucDev->portAttr.active_mtu;
      FLAGCXCHECK(flagcxIbucRtrQpWithType(
          rCommDev->gpuFlush.qp.qp, rCommDev->base.gidInfo.localGidIndex,
          rCommDev->gpuFlush.qp.qp->qp_num, &devInfo, IBV_QPT_RC));
      FLAGCXCHECK(
          flagcxIbucRtsQpWithType(rCommDev->gpuFlush.qp.qp, IBV_QPT_RC));
    }

    if (remMeta.retransEnabled && meta.retransEnabled) {
      const int remDevIdx = remMeta.qpInfo[i].devIndex;
      if (remDevIdx < 0 || remDevIdx >= remMeta.ndevs ||
          remMeta.retransQpn[remDevIdx] == 0)
        return flagcxIbucAbortAccept(lComm, rComm, flagcxInternalError);
      flagcxResult_t retransSetupResult = flagcxIbucCreateQpWithTypeCq(
          ibucDev->portNum, &rCommDev->base, IBV_ACCESS_REMOTE_WRITE,
          IBV_QPT_RC, rCommDev->base.cq, rCommDev->base.cq,
          sizeof(struct flagcxIbAckMsg), &rCommDev->retransQp);
      if (retransSetupResult != flagcxSuccess)
        return flagcxIbucAbortAccept(lComm, rComm, retransSetupResult);
      rCommDev->retransQp.devIndex = i;
      rCommDev->retransQp.remDevIdx = remDevIdx;
      meta.retransQpn[i] = rCommDev->retransQp.qp->qp_num;
      retransSetupResult = flagcxIbucRtrQpWithType(
          rCommDev->retransQp.qp, rCommDev->base.gidInfo.localGidIndex,
          remMeta.retransQpn[remDevIdx], &remMeta.devs[remDevIdx], IBV_QPT_RC);
      if (retransSetupResult != flagcxSuccess)
        return flagcxIbucAbortAccept(lComm, rComm, retransSetupResult);
      retransSetupResult =
          flagcxIbucRtsQpWithType(rCommDev->retransQp.qp, IBV_QPT_RC);
      if (retransSetupResult != flagcxSuccess)
        return flagcxIbucAbortAccept(lComm, rComm, retransSetupResult);

      const size_t retransBytes =
          FLAGCX_IBUC_RETRANS_RECV_DEPTH * flagcxIbucRetransRecvEntrySize;
      void *retransBuffer = malloc(retransBytes);
      if (retransBuffer == NULL)
        return flagcxIbucAbortAccept(lComm, rComm, flagcxSystemError);
      rCommDev->retransRecvBufCount = FLAGCX_IBUC_RETRANS_RECV_DEPTH;
      for (int credit = 0; credit < rCommDev->retransRecvBufCount; credit++)
        rCommDev->retransRecvBufs[credit] =
            (char *)retransBuffer + credit * flagcxIbucRetransRecvEntrySize;
      flagcxResult_t retransMrResult = flagcxWrapIbvRegMr(
          &rCommDev->retransRecvMr, rCommDev->base.pd, retransBuffer,
          retransBytes, IBV_ACCESS_LOCAL_WRITE);
      if (retransMrResult != flagcxSuccess)
        return flagcxIbucAbortAccept(lComm, rComm, retransMrResult);
      for (int credit = 0; credit < rCommDev->retransRecvBufCount; credit++) {
        flagcxResult_t postResult = flagcxIbucPostRetransRecv(rCommDev, credit);
        if (postResult != flagcxSuccess)
          return flagcxIbucAbortAccept(lComm, rComm, postResult);
      }

      flagcxResult_t ctrlResult =
          flagcxIbCreateCtrlQp(ibucDev->context, rCommDev->base.pd,
                               ibucDev->portNum, &rCommDev->ctrlQp);
      if (ctrlResult != flagcxSuccess)
        return flagcxIbucAbortAccept(lComm, rComm, ctrlResult);
      meta.ctrlQpn[i] = rCommDev->ctrlQp.qp->qp_num;
      meta.ctrlLid[i] = ibucDev->lid;
      meta.ctrlGid[i] = rCommDev->base.gidInfo.localGid;

      TRACE(FLAGCX_NET,
            "Receiver: Control QP created for dev %d, qpn=%u, lid=%u", i,
            meta.ctrlQpn[i], meta.ctrlLid[i]);

      size_t ack_buf_size =
          (sizeof(struct flagcxIbAckMsg) + FLAGCX_IB_ACK_BUF_PADDING) *
          FLAGCX_IB_ACK_BUF_COUNT;
      rCommDev->ackBuffer = malloc(ack_buf_size);
      if (rCommDev->ackBuffer == NULL)
        return flagcxIbucAbortAccept(lComm, rComm, flagcxInternalError);
      flagcxResult_t ackMrResult = flagcxWrapIbvRegMr(
          &rCommDev->ackMr, rCommDev->base.pd, rCommDev->ackBuffer,
          ack_buf_size, IBV_ACCESS_LOCAL_WRITE);
      if (ackMrResult != flagcxSuccess)
        return flagcxIbucAbortAccept(lComm, rComm, ackMrResult);

      TRACE(FLAGCX_NET,
            "Recv: Setting up control QP conn for dev %d: remote_qpn=%u, "
            "remote_lid=%u, remote_gid=%lx:%lx, link_layer=%d",
            i, remMeta.ctrlQpn[i], remMeta.ctrlLid[i],
            (unsigned long)remMeta.ctrlGid[i].global.subnet_prefix,
            (unsigned long)remMeta.ctrlGid[i].global.interface_id,
            ibucDev->portAttr.link_layer);

      flagcxResult_t ah_result = flagcxIbSetupCtrlQpConnection(
          ibucDev->context, rCommDev->base.pd, &rCommDev->ctrlQp,
          remMeta.ctrlQpn[i], &remMeta.ctrlGid[i], remMeta.ctrlLid[i],
          ibucDev->portNum, ibucDev->portAttr.link_layer,
          rCommDev->base.gidInfo.localGidIndex);

      if (ah_result != flagcxSuccess || !rCommDev->ctrlQp.ah) {
        WARN("Receiver control QP setup failed for dev %d", i);
        return flagcxIbucAbortAccept(
            lComm, rComm,
            ah_result == flagcxSuccess ? flagcxSystemError : ah_result);
      } else {
        TRACE(FLAGCX_NET,
              "Receiver Control QP successfully initialized for dev %d (ah=%p)",
              i, rCommDev->ctrlQp.ah);
      }
    }

    // Fill Handle
    meta.devs[i].lid = ibucDev->lid;
    meta.devs[i].linkLayer = rCommDev->base.gidInfo.linkLayer =
        ibucDev->portAttr.link_layer;
    meta.devs[i].ibPort = ibucDev->portNum;
    meta.devs[i].spn = rCommDev->base.gidInfo.localGid.global.subnet_prefix;
    meta.devs[i].iid = rCommDev->base.gidInfo.localGid.global.interface_id;

    // Adjust the MTU
    remMeta.devs[i].mtu = (enum ibv_mtu)std::min(remMeta.devs[i].mtu,
                                                 ibucDev->portAttr.active_mtu);
    meta.devs[i].mtu = remMeta.devs[i].mtu;

    // Prepare sizes fifo
    FLAGCXCHECK(flagcxWrapIbvRegMr(
        &rComm->devs[i].sizesFifoMr, rComm->devs[i].base.pd, rComm->sizesFifo,
        sizeof(int) * MAX_REQUESTS * FLAGCX_NET_IB_MAX_RECVS,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
            IBV_ACCESS_REMOTE_READ));
    meta.devs[i].fifoRkey = rComm->devs[i].sizesFifoMr->rkey;
  }
  meta.fifoAddr = (uint64_t)rComm->sizesFifo;

  for (int q = 0; q < rComm->base.nqps; q++) {
    meta.qpInfo[q].qpn = rComm->base.qps[q].qp->qp_num;
    meta.qpInfo[q].devIndex = rComm->base.qps[q].devIndex;
  }

  meta.ndevs = rComm->base.ndevs;
  // IBUC always enables retransmission, ignore remote value
  meta.retransEnabled = 1;
  strncpy(meta.devName, mergedDev->devName, MAX_MERGED_DEV_NAME);

  stage->state = flagcxIbCommStateSend;
  stage->offset = 0;
  if (stage->buffer)
    free(stage->buffer);
  FLAGCXCHECK(flagcxIbucMalloc((void **)&stage->buffer,
                               sizeof(struct flagcxIbConnectionMetadata)));
  memcpy(stage->buffer, &meta, sizeof(struct flagcxIbConnectionMetadata));

ibuc_send:
  FLAGCXCHECK(flagcxSocketProgress(
      FLAGCX_SOCKET_SEND, &rComm->base.sock, stage->buffer,
      sizeof(struct flagcxIbConnectionMetadata), &stage->offset));
  if (stage->offset < sizeof(struct flagcxIbConnectionMetadata))
    return flagcxSuccess;

  stage->offset = 0;
  stage->state = flagcxIbCommStatePendingReady;

ib_recv_ready:
  FLAGCXCHECK(flagcxSocketProgress(FLAGCX_SOCKET_RECV, &rComm->base.sock,
                                   &rComm->base.ready, sizeof(int),
                                   &stage->offset));
  if (stage->offset != sizeof(int))
    return flagcxSuccess;

  retransResult = flagcxIbRetransInit(&rComm->retrans);
  if (retransResult != flagcxSuccess)
    return flagcxIbucAbortAccept(lComm, rComm, retransResult);

  // IBUC always enables retransmission, force it on
  rComm->retrans.enabled = 1;
  rComm->retrans.ackInterval = 1;
  retransReady = true;
  for (int i = 0; i < rComm->base.ndevs && retransReady; i++) {
    struct flagcxIbRecvCommDev *commDev = rComm->devs + i;
    retransReady =
        commDev->ctrlQp.qp != NULL && commDev->ctrlQp.cq != NULL &&
        commDev->ctrlQp.ah != NULL && commDev->ackMr != NULL &&
        commDev->ackBuffer != NULL && commDev->retransQp.qp != NULL &&
        commDev->retransRecvMr != NULL && commDev->retransRecvBufCount > 0;
  }
  if (!retransReady) {
    WARN("NET/IBUC : retransmission setup did not complete on every peer");
    return flagcxIbucAbortAccept(lComm, rComm, flagcxNotSupported);
  }

  INFO(FLAGCX_NET,
       "NET/IBUC Receiver: retransmission enabled with dedicated RC QPs "
       "and %d bounded payload buffers per rail",
       rComm->devs[0].retransRecvBufCount);

  free(stage->buffer);
  *recvComm = rComm;

  /* reset lComm stage */
  stage->state = flagcxIbCommStateStart;
  stage->offset = 0;
  stage->comm = NULL;
  stage->buffer = NULL;
  return flagcxSuccess;
}

flagcxResult_t flagcxIbucGetRequest(struct flagcxIbNetCommBase *base,
                                    struct flagcxIbRequest **req) {
  for (int i = 0; i < MAX_REQUESTS; i++) {
    struct flagcxIbRequest *r = base->reqs + i;
    if (r->type == FLAGCX_NET_IB_REQ_UNUSED) {
      r->base = base;
      r->result = flagcxSuccess;
      r->sock = NULL;
      r->devBases[0] = NULL;
      r->devBases[1] = NULL;
      r->events[0] = r->events[1] = 0;
      r->dataEvents[0] = r->dataEvents[1] = 0;
      r->retransSeq = UINT32_MAX;
      r->nreqs = 0;
      *req = r;
      return flagcxSuccess;
    }
  }
  WARN("NET/IBUC : unable to allocate requests");
  *req = NULL;
  return flagcxInternalError;
}

flagcxResult_t flagcxIbucRegMrDmaBufInternal(flagcxIbNetCommDevBase *base,
                                             void *data, size_t size, int type,
                                             uint64_t offset, int fd,
                                             int mrFlags, ibv_mr **mhandle) {
  static __thread uintptr_t pageSize = 0;
  if (pageSize == 0)
    pageSize = sysconf(_SC_PAGESIZE);
  struct flagcxIbMrCache *cache = &flagcxIbDevs[base->ibDevN].mrCache;
  uintptr_t addr = (uintptr_t)data & -pageSize;
  size_t pages = ((uintptr_t)data + size - addr + pageSize - 1) / pageSize;
  flagcxResult_t res;
  pthread_mutex_lock(&flagcxIbDevs[base->ibDevN].lock);
  for (int slot = 0; /*true*/; slot++) {
    if (slot == cache->population || addr < cache->slots[slot].addr) {
      if (cache->population == cache->capacity) {
        cache->capacity = cache->capacity < 32 ? 32 : 2 * cache->capacity;
        FLAGCXCHECKGOTO(
            flagcxRealloc(&cache->slots, cache->population, cache->capacity),
            res, returning);
      }
      // Deregister / register
      struct ibv_mr *mr;
      unsigned int flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                           IBV_ACCESS_REMOTE_READ;
      if (flagcxIbRelaxedOrderingEnabled &&
          !(mrFlags & FLAGCX_NET_MR_FLAG_FORCE_SO))
        flags |= IBV_ACCESS_RELAXED_ORDERING;
      if (fd != -1) {
        /* DMA-BUF support */
        FLAGCXCHECKGOTO(flagcxWrapIbvRegDmabufMr(&mr, base->pd, offset,
                                                 pages * pageSize, addr, fd,
                                                 flags),
                        res, returning);
      } else {
        void *cpuptr = NULL;
        if (deviceAdaptor->gdrPtrMmap && deviceAdaptor->gdrPtrMunmap) {
          deviceAdaptor->gdrPtrMmap(&cpuptr, (void *)addr, pages * pageSize);
        }
        if (flagcxIbRelaxedOrderingEnabled &&
            !(mrFlags & FLAGCX_NET_MR_FLAG_FORCE_SO)) {
          // Use IBVERBS_1.8 API - needed for IBV_ACCESS_RELAXED_ORDERING
          // support
          FLAGCXCHECKGOTO(
              flagcxWrapIbvRegMrIova2(&mr, base->pd,
                                      cpuptr == NULL ? (void *)addr : cpuptr,
                                      pages * pageSize, addr, flags),
              res, returning);
        } else {
          FLAGCXCHECKGOTO(
              flagcxWrapIbvRegMr(&mr, base->pd,
                                 cpuptr == NULL ? (void *)addr : cpuptr,
                                 pages * pageSize, flags),
              res, returning);
        }
        if (deviceAdaptor->gdrPtrMmap && deviceAdaptor->gdrPtrMunmap) {
          deviceAdaptor->gdrPtrMunmap(cpuptr, pages * pageSize);
        }
      }
      TRACE(FLAGCX_INIT | FLAGCX_NET,
            "regAddr=0x%lx size=%lld rkey=0x%x lkey=0x%x fd=%d",
            (unsigned long)addr, (long long)pages * pageSize, mr->rkey,
            mr->lkey, fd);
      if (slot != cache->population)
        memmove(cache->slots + slot + 1, cache->slots + slot,
                (cache->population - slot) * sizeof(struct flagcxIbMr));
      cache->slots[slot].addr = addr;
      cache->slots[slot].pages = pages;
      cache->slots[slot].refs = 1;
      cache->slots[slot].mr = mr;
      cache->population += 1;
      *mhandle = mr;
      res = flagcxSuccess;
      goto returning;
    } else if ((addr >= cache->slots[slot].addr) &&
               ((addr - cache->slots[slot].addr) / pageSize + pages) <=
                   cache->slots[slot].pages) {
      cache->slots[slot].refs += 1;
      *mhandle = cache->slots[slot].mr;
      res = flagcxSuccess;
      goto returning;
    }
  }
returning:
  pthread_mutex_unlock(&flagcxIbDevs[base->ibDevN].lock);
  return res;
}

struct flagcxIbNetCommDevBase *
flagcxIbucGetNetCommDevBase(flagcxIbNetCommBase *base, int devIndex) {
  if (base->isSend) {
    struct flagcxIbSendComm *sComm = (struct flagcxIbSendComm *)base;
    return &sComm->devs[devIndex].base;
  } else {
    struct flagcxIbRecvComm *rComm = (struct flagcxIbRecvComm *)base;
    return &rComm->devs[devIndex].base;
  }
}

flagcxResult_t flagcxIbucDeregMrInternal(flagcxIbNetCommDevBase *base,
                                         ibv_mr *mhandle);

/* DMA-BUF support */
flagcxResult_t flagcxIbucRegMrDmaBuf(void *comm, void *data, size_t size,
                                     int type, uint64_t offset, int fd,
                                     int mrFlags, void **mhandle) {
  if (mhandle == NULL)
    return flagcxInvalidArgument;
  *mhandle = NULL;
  if (comm == NULL || data == NULL || size == 0 ||
      size > UINTPTR_MAX - (uintptr_t)data)
    return flagcxInvalidArgument;

  struct flagcxIbNetCommBase *base = (struct flagcxIbNetCommBase *)comm;
  if (base->ndevs <= 0 || base->ndevs > FLAGCX_IB_MAX_DEVS_PER_NIC)
    return flagcxInternalError;
  struct flagcxIbMrHandle *mhandleWrapper =
      (struct flagcxIbMrHandle *)calloc(1, sizeof(struct flagcxIbMrHandle));
  if (mhandleWrapper == NULL)
    return flagcxSystemError;
  mhandleWrapper->type = type;

  for (int i = 0; i < base->ndevs; i++) {
    struct flagcxIbNetCommDevBase *devComm =
        flagcxIbucGetNetCommDevBase(base, i);
    flagcxResult_t result =
        flagcxIbucRegMrDmaBufInternal(devComm, data, size, type, offset, fd,
                                      mrFlags, mhandleWrapper->mrs + i);
    if (result != flagcxSuccess) {
      for (int j = i - 1; j >= 0; j--) {
        struct flagcxIbNetCommDevBase *registeredDev =
            flagcxIbucGetNetCommDevBase(base, j);
        flagcxResult_t cleanupResult =
            flagcxIbucDeregMrInternal(registeredDev, mhandleWrapper->mrs[j]);
        if (cleanupResult == flagcxSuccess) {
          mhandleWrapper->mrs[j] = NULL;
        } else {
          WARN("NET/IBUC: failed to roll back MR registration on device %d: "
               "%d",
               j, cleanupResult);
        }
      }

      bool cleanupDeferred = false;
      for (int j = 0; j < i; j++)
        cleanupDeferred |= mhandleWrapper->mrs[j] != NULL;
      if (cleanupDeferred) {
        mhandleWrapper->nextDeferred = base->deferredMrHandles;
        base->deferredMrHandles = mhandleWrapper;
      } else {
        free(mhandleWrapper);
      }
      return result;
    }
  }
  *mhandle = (void *)mhandleWrapper;
  return flagcxSuccess;
}

flagcxResult_t flagcxIbucRegMr(void *comm, void *data, size_t size, int type,
                               int mrFlags, void **mhandle) {
  return flagcxIbucRegMrDmaBuf(comm, data, size, type, 0ULL, -1, mrFlags,
                               mhandle);
}

flagcxResult_t flagcxIbucDeregMrInternal(flagcxIbNetCommDevBase *base,
                                         ibv_mr *mhandle) {
  struct flagcxIbMrCache *cache = &flagcxIbDevs[base->ibDevN].mrCache;
  flagcxResult_t res;
  pthread_mutex_lock(&flagcxIbDevs[base->ibDevN].lock);
  for (int i = 0; i < cache->population; i++) {
    if (mhandle == cache->slots[i].mr) {
      if (cache->slots[i].refs > 1) {
        cache->slots[i].refs--;
        res = flagcxSuccess;
        goto returning;
      }

      // Keep the cache entry intact if deregistration fails so the owner can
      // retry and the live MR cannot become detached from its wrapper.
      FLAGCXCHECKGOTO(flagcxWrapIbvDeregMr(mhandle), res, returning);
      cache->population--;
      if (i < cache->population) {
        memmove(&cache->slots[i], &cache->slots[i + 1],
                (cache->population - i) * sizeof(struct flagcxIbMr));
      }
      if (cache->population == 0) {
        free(cache->slots);
        cache->slots = NULL;
        cache->capacity = 0;
      }
      res = flagcxSuccess;
      goto returning;
    }
  }
  WARN("NET/IBUC: could not find mr %p inside cache of %d entries", mhandle,
       cache->population);
  res = flagcxInternalError;
returning:
  pthread_mutex_unlock(&flagcxIbDevs[base->ibDevN].lock);
  return res;
}

flagcxResult_t flagcxIbucDeregMr(void *comm, void *mhandle) {
  return flagcxIbDeregMrOrDeferWithCallback((struct flagcxIbNetCommBase *)comm,
                                            mhandle, flagcxIbucDeregMrInternal);
}

FLAGCX_PARAM(IbucSplitDataOnQps, "IBUC_SPLIT_DATA_ON_QPS", 0);

flagcxResult_t flagcxIbucMultiSend(struct flagcxIbSendComm *comm, int slot) {
  struct flagcxIbRequest **reqs = comm->fifoReqs[slot];
  volatile struct flagcxIbSendFifo *slots = comm->fifo[slot];
  int nreqs = slots[0].nreqs;
  if (nreqs > FLAGCX_NET_IB_MAX_RECVS)
    return flagcxInternalError;
  for (int r = 0; r < nreqs; r++) {
    if (reqs[r] == NULL || reqs[r]->send.size < 0)
      return flagcxInternalError;
  }

  uint64_t wr_id = 0ULL;
  for (int r = 0; r < nreqs; r++) {
    struct ibv_send_wr *wr = comm->wrs + r;
    memset(wr, 0, sizeof(struct ibv_send_wr));

    struct ibv_sge *sge = comm->sges + r;
    sge->addr = (uintptr_t)reqs[r]->send.data;
    wr->opcode = IBV_WR_RDMA_WRITE;
    wr->send_flags = 0;
    wr->wr_id = flagcxIbUnsignaledWrId(reqs[r] - comm->base.reqs);
    wr->wr.rdma.remote_addr = slots[r].addr;
    wr->next = wr + 1;
    wr_id += (reqs[r] - comm->base.reqs) << (r * 8);
  }

  // Retransmission immediate data identifies the logical receive. Full sizes
  // are always written to the receiver's sizes FIFO so messages larger than
  // 64 KiB are not truncated by immediate-data encoding.
  uint32_t immData = 0;
  uint32_t seq = 0;

  if (comm->retrans.enabled) {
    seq = comm->retrans.sendSeq;
    comm->retrans.sendSeq =
        (comm->retrans.sendSeq + 1) & FLAGCX_IB_RETRANS_SEQ_MASK;
    immData =
        flagcxIbucEncodeImmData(seq, slots[0].requestSlot, slots[0].generation);
  } else if (nreqs == 1) {
    immData = reqs[0]->send.size;
  }

  if (comm->retrans.enabled || nreqs > 1) {
    int *sizes = comm->remSizesFifo.elems[slot];
    for (int r = 0; r < nreqs; r++)
      sizes[r] = reqs[r]->send.size;
    comm->remSizesFifo.sge.addr = (uint64_t)sizes;
    comm->remSizesFifo.sge.length = nreqs * sizeof(int);
  }

  struct ibv_send_wr *lastWr = comm->wrs + nreqs - 1;
  if (comm->retrans.enabled || nreqs > 1 ||
      (comm->ar && reqs[0]->send.size > flagcxParamIbArThreshold())) {
    // When using ADAPTIVE_ROUTING, send the bulk of the data first as an
    // RDMA_WRITE, then a 0-byte RDMA_WRITE_WITH_IMM to trigger a remote
    // completion.
    lastWr++;
    memset(lastWr, 0, sizeof(struct ibv_send_wr));
    if (comm->retrans.enabled || nreqs > 1) {
      // Write remote sizes Fifo
      lastWr->wr.rdma.remote_addr =
          comm->remSizesFifo.addr +
          slot * FLAGCX_NET_IB_MAX_RECVS * sizeof(int);
      lastWr->num_sge = 1;
      lastWr->sg_list = &comm->remSizesFifo.sge;
    }
  }
  lastWr->wr_id = wr_id;
  lastWr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  lastWr->imm_data = immData;
  lastWr->next = NULL;
  lastWr->send_flags = IBV_SEND_SIGNALED;

  // Multi-QP: make sure IB writes are multiples of 128B so that LL and LL128
  // protocols still work
  const int align = 128;
  int nqps =
      flagcxParamIbucSplitDataOnQps() ? comm->base.nqps : comm->base.ndevs;
  for (int i = 0; i < nqps; i++) {
    struct flagcxIbLane lane = {};
    FLAGCXCHECK(
        flagcxIbSelectLane(&comm->base, FLAGCX_NET_LANE_UNORDERED, 0, &lane));
    flagcxIbQp *qp = lane.ibQp;
    int devIndex = qp->devIndex;
    for (int r = 0; r < nreqs; r++) {
      comm->wrs[r].wr.rdma.rkey = slots[r].rkeys[qp->remDevIdx];

      int chunkSize = DIVUP(DIVUP(reqs[r]->send.size, nqps), align) * align;
      int length =
          std::min(reqs[r]->send.size - reqs[r]->send.offset, chunkSize);
      if (length <= 0) {
        comm->wrs[r].sg_list = NULL;
        comm->wrs[r].num_sge = 0;
      } else {
        // Select proper lkey
        comm->sges[r].lkey = reqs[r]->send.lkeys[devIndex];
        comm->sges[r].length = length;
        comm->wrs[r].sg_list = comm->sges + r;
        comm->wrs[r].num_sge = 1;
      }
    }

    if (comm->retrans.enabled || nreqs > 1) {
      // Also make sure lastWr writes remote sizes using the right lkey
      comm->remSizesFifo.sge.lkey = comm->remSizesFifo.mrs[devIndex]->lkey;
      lastWr->wr.rdma.rkey = comm->remSizesFifo.rkeys[devIndex];
    }

    // Ensure lastWr has IBV_SEND_SIGNALED set for each QP
    // (it was set before the loop, but we need to ensure it's set for each QP)
    lastWr->send_flags = IBV_SEND_SIGNALED;

    struct ibv_send_wr *bad_wr;
    // Call ibv_post_send directly to handle ENOMEM (send queue full) gracefully
    int ret = flagcxWrapIbvPostSendRaw(qp->qp, comm->wrs, &bad_wr);
    if (ret != IBV_SUCCESS) {
      // If send queue is full (ENOMEM), poll completions from all devices and
      // retry
      if (ret == ENOMEM) {
        struct ibv_wc wcs[64];
        // Poll all devices' CQs to free up send queue space
        for (int dev_i = 0; dev_i < comm->base.ndevs; dev_i++) {
          struct flagcxIbNetCommDevBase *devBase = &comm->devs[dev_i].base;
          if (!devBase || !devBase->cq)
            continue;
          for (int poll_round = 0; poll_round < 16; poll_round++) {
            int n_cqe = 0;
            flagcxWrapIbvPollCq(devBase->cq, 64, wcs, &n_cqe);
            if (n_cqe == 0)
              break;
          }
        }
        // Retry sending after polling
        ret = flagcxWrapIbvPostSendRaw(qp->qp, comm->wrs, &bad_wr);
        // If still failing after polling, continue retrying with more
        // aggressive polling
        int retry_count = 0;
        while (ret == ENOMEM && retry_count < 3) {
          // More aggressive polling
          for (int dev_i = 0; dev_i < comm->base.ndevs; dev_i++) {
            struct flagcxIbNetCommDevBase *devBase = &comm->devs[dev_i].base;
            if (!devBase || !devBase->cq)
              continue;
            for (int poll_round = 0; poll_round < 32; poll_round++) {
              int n_cqe = 0;
              flagcxWrapIbvPollCq(devBase->cq, 64, wcs, &n_cqe);
              if (n_cqe == 0)
                break;
            }
          }
          sched_yield(); // Yield CPU to allow other threads/processes to make
                         // progress
          ret = flagcxWrapIbvPostSendRaw(qp->qp, comm->wrs, &bad_wr);
          retry_count++;
        }
      }
      // If still failing, check if it's ENOMEM (don't warn) or other error
      if (ret != IBV_SUCCESS) {
        if (ret != ENOMEM) {
          WARN("ibv_post_send() failed with error %s, Bad WR %p, First WR %p",
               strerror(ret), comm->wrs, bad_wr);
        }
        FLAGCXCHECK(flagcxSystemError);
      }
    }

    for (int r = 0; r < nreqs; r++) {
      int chunkSize = DIVUP(DIVUP(reqs[r]->send.size, nqps), align) * align;
      reqs[r]->send.offset += chunkSize;
      comm->sges[r].addr += chunkSize;
      comm->wrs[r].wr.rdma.remote_addr += chunkSize;
    }

    FLAGCXCHECK(
        flagcxIbCommitLane(&comm->base, FLAGCX_NET_LANE_UNORDERED, &lane));
  }

  if (comm->retrans.enabled)
    FLAGCXCHECK(flagcxIbRetransAddBatch(&comm->retrans, seq,
                                        slots[0].requestSlot,
                                        slots[0].generation, 0, nreqs, reqs));

  comm->outstandingSends++;

  return flagcxSuccess;
}

flagcxResult_t flagcxIbucIsend(void *sendComm, void *data, size_t size, int tag,
                               void *mhandle, void *phandle, void **request) {
  if (sendComm == NULL || data == NULL || mhandle == NULL || request == NULL)
    return flagcxInvalidArgument;
  *request = NULL;

  struct flagcxIbSendComm *comm = (struct flagcxIbSendComm *)sendComm;
  if (comm->base.ready == 0) {
    WARN("NET/IBUC: flagcxIbucIsend() called when comm->base.ready == 0");
    return flagcxInternalError;
  }
  // Removed flagcxIbucPollCompletions call to match ibrc behavior
  // Completions are handled in Test function, not in Isend
  // This prevents potential hang issues

  struct flagcxIbMrHandle *mhandleWrapper = (struct flagcxIbMrHandle *)mhandle;

  // Wait for the receiver to have posted the corresponding receive
  int nreqs = 0;
  volatile struct flagcxIbSendFifo *slots;

  int slot = (comm->fifoHead) % MAX_REQUESTS;
  struct flagcxIbRequest **reqs = comm->fifoReqs[slot];
  slots = comm->fifo[slot];
  uint64_t idx = comm->fifoHead + 1;
  if (slots[0].idx != idx) {
    return flagcxSuccess;
  }
  nreqs = slots[0].nreqs;
  // Wait until all data has arrived
  for (int r = 1; r < nreqs; r++) {
    int spin_count = 0;
    while (slots[r].idx != idx) {
      if (++spin_count > 1000) {
        sched_yield(); // Yield CPU to prevent busy-wait hang
        spin_count = 0;
      }
    }
  }
  __sync_synchronize();
  for (int r = 0; r < nreqs; r++) {
    if (reqs[r] != NULL || slots[r].tag != tag)
      continue;

    if (size > slots[r].size)
      size = slots[r].size;
    // Sanity checks
    if (slots[r].size < 0 || slots[r].addr == 0 || slots[r].rkeys[0] == 0) {
      char line[SOCKET_NAME_MAXLEN + 1];
      union flagcxSocketAddress addr;
      flagcxSocketGetAddr(&comm->base.sock, &addr);
      WARN("NET/IBUC : req %d/%d tag %x peer %s posted incorrect receive info: "
           "size %zu addr %" PRIx64 " rkeys[0]=%x",
           r, nreqs, tag, flagcxSocketToString(&addr, line), slots[r].size,
           slots[r].addr, slots[r].rkeys[0]);
      return flagcxInternalError;
    }

    struct flagcxIbRequest *req;
    FLAGCXCHECK(flagcxIbucGetRequest(&comm->base, &req));
    req->type = FLAGCX_NET_IB_REQ_SEND;
    req->sock = &comm->base.sock;
    req->base = &comm->base;
    req->nreqs = nreqs;
    req->send.size = size;
    req->send.data = data;
    req->send.offset = 0;

    // Populate events
    int nEvents =
        flagcxParamIbucSplitDataOnQps() ? comm->base.nqps : comm->base.ndevs;
    struct flagcxNetLaneSet previewLanes = {
        (uint32_t)comm->base.nqps,
        (uint32_t)comm->base.qpIndex,
    };
    // Count down
    while (nEvents > 0) {
      uint32_t laneIndex = 0;
      FLAGCXCHECK(flagcxNetSelectLane(&previewLanes, FLAGCX_NET_LANE_UNORDERED,
                                      0, &laneIndex));
      flagcxIbQp *qp = comm->base.qps + laneIndex;
      int devIndex = qp->devIndex;
      flagcxIbucAddEvent(req, devIndex, &comm->devs[devIndex].base);
      // Track the valid lkey for this RDMA_Write
      req->send.lkeys[devIndex] = mhandleWrapper->mrs[devIndex]->lkey;
      nEvents--;
      // Don't update comm->base.qpIndex yet, we need to run through this same
      // set of QPs inside flagcxIbucMultiSend()
      FLAGCXCHECK(flagcxNetCommitLane(&previewLanes, FLAGCX_NET_LANE_UNORDERED,
                                      laneIndex));
    }

    // Store all lkeys
    for (int i = 0; i < comm->base.ndevs; i++) {
      req->send.lkeys[i] = mhandleWrapper->mrs[i]->lkey;
    }

    *request = reqs[r] = req;

    // If this is a multi-recv, send only when all requests have matched.
    for (int r = 0; r < nreqs; r++) {
      if (reqs[r] == NULL)
        return flagcxSuccess;
    }

    TIME_START(0);
    FLAGCXCHECK(flagcxIbucMultiSend(comm, slot));

    // Clear slots[0]->nreqs, as well as other fields to help debugging and
    // sanity checks
    memset((void *)slots, 0, sizeof(struct flagcxIbSendFifo));
    memset(reqs, 0, FLAGCX_NET_IB_MAX_RECVS * sizeof(struct flagcxIbRequest *));
    comm->fifoHead++;
    TIME_STOP(0);
    return flagcxSuccess;
  }

  *request = NULL;
  return flagcxSuccess;
}

flagcxResult_t flagcxIbucPostFifo(struct flagcxIbRecvComm *comm, int n,
                                  void **data, size_t *sizes, int *tags,
                                  void **mhandles,
                                  struct flagcxIbRequest *req) {
  if (comm == NULL || req == NULL)
    return flagcxInternalError;
  const int slot = comm->remFifo.fifoTail % MAX_REQUESTS;
  const uint32_t requestSlot = req - comm->base.reqs;
  struct flagcxIbSendFifo *localElem = comm->remFifo.elems[slot];
  for (int i = 0; i < n; i++) {
    localElem[i].requestSlot = requestSlot;
    localElem[i].generation = req->retransGeneration;
  }
  return flagcxIbCommonPostFifo(comm, n, data, sizes, tags, mhandles, req,
                                flagcxIbucAddEvent);
}

flagcxResult_t flagcxIbucIrecv(void *recvComm, int n, void **data,
                               size_t *sizes, int *tags, void **mhandles,
                               void **phandles, void **request) {
  if (recvComm == NULL || n <= 0 || n > FLAGCX_NET_IB_MAX_RECVS ||
      data == NULL || sizes == NULL || tags == NULL || mhandles == NULL ||
      request == NULL)
    return flagcxInvalidArgument;
  *request = NULL;
  for (int i = 0; i < n; i++) {
    if (data[i] == NULL || mhandles[i] == NULL)
      return flagcxInvalidArgument;
  }

  struct flagcxIbRecvComm *comm = (struct flagcxIbRecvComm *)recvComm;
  if (comm->base.ready == 0) {
    WARN("NET/IBUC: flagcxIbucIrecv() called when comm->base.ready == 0");
    return flagcxInternalError;
  }

  struct flagcxIbRequest *req;
  FLAGCXCHECK(flagcxIbucGetRequest(&comm->base, &req));
  req->type = FLAGCX_NET_IB_REQ_RECV;
  req->sock = &comm->base.sock;
  req->nreqs = n;
  req->retransGeneration =
      (req->retransGeneration + 1) & FLAGCX_IBUC_GENERATION_MASK;
  if (req->retransGeneration == 0)
    req->retransGeneration = 1;
  req->retransSegmentMask = 0;
  memset(req->retransSegmentBytes, 0, sizeof(req->retransSegmentBytes));

  for (int i = 0; i < n; i++) {
    struct flagcxIbMrHandle *mhandle = (struct flagcxIbMrHandle *)mhandles[i];
    req->recv.data[i] = data[i];
    req->recv.types[i] = mhandle->type;
    req->recv.capacities[i] = sizes[i];
  }

  for (int i = 0; i < comm->base.ndevs; i++) {
    req->devBases[i] = &comm->devs[i].base;
  }

  TIME_START(1);
  // Select either all QPs, or one qp per-device
  const int nqps =
      flagcxParamIbucSplitDataOnQps() ? comm->base.nqps : comm->base.ndevs;

  // Notification WQEs are connection-level credits posted during accept.
  // Associate this logical request with the same QPs the sender will stripe
  // over, without tying receive-WQE ownership to the request slot.
  for (int i = 0; i < nqps; i++) {
    struct flagcxIbLane lane = {};
    FLAGCXCHECK(
        flagcxIbSelectLane(&comm->base, FLAGCX_NET_LANE_UNORDERED, 0, &lane));
    struct flagcxIbQp *qp = lane.ibQp;
    flagcxIbucAddDataEvent(req, qp->devIndex, &comm->devs[qp->devIndex].base);
    FLAGCXCHECK(
        flagcxIbCommitLane(&comm->base, FLAGCX_NET_LANE_UNORDERED, &lane));
  }

  TIME_STOP(1);

  // Post to FIFO to notify sender
  TIME_START(2);
  FLAGCXCHECK(flagcxIbucPostFifo(comm, n, data, sizes, tags, mhandles, req));
  TIME_STOP(2);

  *request = req;
  return flagcxSuccess;
}

flagcxResult_t flagcxIbucIflush(void *recvComm, int n, void **data, int *sizes,
                                void **mhandles, void **request) {
  struct flagcxIbRecvComm *comm = (struct flagcxIbRecvComm *)recvComm;
  int last = -1;
  for (int i = 0; i < n; i++)
    if (sizes[i])
      last = i;
  if (comm->flushEnabled == 0 || last == -1)
    return flagcxSuccess;

  // Only flush once using the last non-zero receive
  struct flagcxIbRequest *req;
  FLAGCXCHECK(flagcxIbucGetRequest(&comm->base, &req));
  req->type = FLAGCX_NET_IB_REQ_FLUSH;
  req->sock = &comm->base.sock;
  // struct flagcxIbMrHandle *mhandle = (struct flagcxIbMrHandle
  // *)mhandles[last];

  // We don't know which devIndex the recv was on, so we flush on all devices
  // For flush operations, we use RC QP which supports RDMA_READ
  for (int i = 0; i < comm->base.ndevs; i++) {
    struct ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = req - comm->base.reqs;

    // Use RDMA_READ for flush operations
    wr.wr.rdma.remote_addr = (uint64_t)data[last];
    wr.wr.rdma.rkey = ((struct flagcxIbMrHandle *)mhandles[last])->mrs[i]->rkey;
    wr.sg_list = &comm->devs[i].gpuFlush.sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_READ;
    wr.send_flags = IBV_SEND_SIGNALED;

    TIME_START(4);
    struct ibv_send_wr *bad_wr;
    FLAGCXCHECK(
        flagcxWrapIbvPostSend(comm->devs[i].gpuFlush.qp.qp, &wr, &bad_wr));
    TIME_STOP(4);

    flagcxIbucAddEvent(req, i, &comm->devs[i].base);
  }

  *request = req;
  return flagcxSuccess;
}

flagcxResult_t flagcxIbucTest(void *request, int *done, int *sizes) {
  static const struct flagcxIbCommonTestOps kIbucTestOps = {
      .component = "NET/IBUC",
      .pre_check = flagcxIbucTestPreCheck,
      .process_wc = flagcxIbucProcessWc,
      .pollAllCqs = true,
  };
  struct flagcxIbRequest *r = (struct flagcxIbRequest *)request;
  flagcxResult_t result =
      flagcxIbCommonTestDataQp(r, done, sizes, &kIbucTestOps);
  return result;
}

static void flagcxIbucRecordCleanupError(flagcxResult_t current,
                                         flagcxResult_t *first) {
  if (*first == flagcxSuccess && current != flagcxSuccess)
    *first = current;
}

static flagcxResult_t flagcxIbucDestroyMr(struct ibv_mr **mr) {
  if (mr == NULL || *mr == NULL)
    return flagcxSuccess;
  flagcxResult_t result = flagcxWrapIbvDeregMr(*mr);
  if (result == flagcxSuccess)
    *mr = NULL;
  return result;
}

static flagcxResult_t flagcxIbucDestroyQp(struct flagcxIbQp *qp) {
  if (qp == NULL || qp->qp == NULL)
    return flagcxSuccess;
  flagcxResult_t result = flagcxWrapIbvDestroyQp(qp->qp);
  if (result == flagcxSuccess)
    qp->qp = NULL;
  return result;
}

static flagcxResult_t flagcxIbucDestroyCq(struct ibv_cq **cq) {
  if (cq == NULL || *cq == NULL)
    return flagcxSuccess;
  struct ibv_wc wcs[64];
  int nCqe = 0;
  for (int i = 0; i < 16; i++) {
    flagcxResult_t pollResult = flagcxWrapIbvPollCq(*cq, 64, wcs, &nCqe);
    if (pollResult != flagcxSuccess)
      return pollResult;
    if (nCqe == 0)
      break;
  }
  flagcxResult_t result = flagcxWrapIbvDestroyCq(*cq);
  if (result == flagcxSuccess)
    *cq = NULL;
  return result;
}

static bool flagcxIbucCtrlQpReleased(const struct flagcxIbCtrlQp *ctrlQp) {
  return ctrlQp->ah == NULL && ctrlQp->qp == NULL && ctrlQp->cq == NULL;
}

static bool flagcxIbucDataQpsReleased(const struct flagcxIbNetCommBase *base,
                                      int devIndex) {
  for (int q = 0; q < base->nqps; q++) {
    if (base->qps[q].devIndex == devIndex && base->qps[q].qp != NULL)
      return false;
  }
  return true;
}

static flagcxResult_t flagcxIbucCleanupSend(struct flagcxIbSendComm *comm,
                                            bool *released) {
  if (released == NULL)
    return flagcxInvalidArgument;
  *released = comm == NULL;
  if (comm == NULL)
    return flagcxSuccess;

  flagcxResult_t result = flagcxSuccess;
  flagcxResult_t current = flagcxIbDrainDeferredMrsWithCallback(
      &comm->base, flagcxIbucDeregMrInternal);
  flagcxIbucRecordCleanupError(current, &result);

  if (comm->retrans.enabled) {
    current = flagcxIbRetransDestroy(&comm->retrans);
    flagcxIbucRecordCleanupError(current, &result);
    if (current == flagcxSuccess)
      comm->retrans.enabled = 0;
  }
  flagcxIbucRecordCleanupError(flagcxSocketClose(&comm->base.sock), &result);

  // Drain the shared data CQs before destroying their QPs.
  for (int i = 0; i < comm->base.ndevs; i++) {
    struct flagcxIbNetCommDevBase *base = &comm->devs[i].base;
    if (base->cq != NULL) {
      struct ibv_wc wcs[64];
      int nCqe = 0;
      for (int j = 0; j < 16; j++) {
        if (flagcxWrapIbvPollCq(base->cq, 64, wcs, &nCqe) != flagcxSuccess ||
            nCqe == 0)
          break;
      }
    }
  }

  for (int i = 0; i < comm->base.ndevs; i++) {
    current = flagcxIbDestroyCtrlQp(&comm->devs[i].ctrlQp);
    flagcxIbucRecordCleanupError(current, &result);
    current = flagcxIbucDestroyQp(&comm->devs[i].retransQp);
    flagcxIbucRecordCleanupError(current, &result);
    if (comm->devs[i].retransQp.qp == NULL) {
      current = flagcxIbucDestroyCq(&comm->devs[i].retransCq);
      flagcxIbucRecordCleanupError(current, &result);
    }
  }
  for (int q = 0; q < comm->base.nqps; q++) {
    current = flagcxIbucDestroyQp(&comm->base.qps[q]);
    flagcxIbucRecordCleanupError(current, &result);
  }

  bool allDataQpsReleased = true;
  for (int q = 0; q < comm->base.nqps; q++)
    allDataQpsReleased &= comm->base.qps[q].qp == NULL;

  for (int i = 0; i < comm->base.ndevs; i++) {
    struct flagcxIbSendCommDev *commDev = &comm->devs[i];
    if (commDev->ctrlQp.qp == NULL && commDev->retransQp.qp == NULL) {
      current = flagcxIbucDestroyMr(&commDev->ackMr);
      flagcxIbucRecordCleanupError(current, &result);
      if (commDev->ackMr == NULL && commDev->ackBuffer != NULL) {
        free(commDev->ackBuffer);
        commDev->ackBuffer = NULL;
      }
    }
    if (flagcxIbucDataQpsReleased(&comm->base, i)) {
      current = flagcxIbucDestroyMr(&commDev->fifoMr);
      flagcxIbucRecordCleanupError(current, &result);
      current = flagcxIbucDestroyMr(&comm->remSizesFifo.mrs[i]);
      flagcxIbucRecordCleanupError(current, &result);
    }
    if (commDev->retransQp.qp == NULL) {
      current = flagcxIbucDestroyMr(&commDev->retransHdrMr);
      flagcxIbucRecordCleanupError(current, &result);
    }
  }
  if (allDataQpsReleased) {
    current = flagcxIbucDestroyMr(&comm->retransHdrMr);
    flagcxIbucRecordCleanupError(current, &result);
  }

  for (int i = 0; i < comm->base.ndevs; i++) {
    struct flagcxIbSendCommDev *commDev = &comm->devs[i];
    const bool hasDependencies =
        comm->base.deferredMrHandles != NULL ||
        !flagcxIbucCtrlQpReleased(&commDev->ctrlQp) ||
        commDev->retransQp.qp != NULL || commDev->retransCq != NULL ||
        !flagcxIbucDataQpsReleased(&comm->base, i) || commDev->ackMr != NULL ||
        commDev->ackBuffer != NULL || commDev->fifoMr != NULL ||
        comm->remSizesFifo.mrs[i] != NULL || commDev->retransHdrMr != NULL ||
        (i == 0 && comm->retransHdrMr != NULL);
    if (!hasDependencies) {
      current = flagcxIbucDestroyBase(&commDev->base);
      flagcxIbucRecordCleanupError(current, &result);
    }
  }

  bool allReleased =
      comm->base.deferredMrHandles == NULL && comm->retransHdrMr == NULL;
  for (int q = 0; q < comm->base.nqps; q++)
    allReleased &= comm->base.qps[q].qp == NULL;
  for (int i = 0; i < comm->base.ndevs; i++) {
    const struct flagcxIbSendCommDev *commDev = &comm->devs[i];
    allReleased &= flagcxIbucCtrlQpReleased(&commDev->ctrlQp) &&
                   commDev->retransQp.qp == NULL &&
                   commDev->retransCq == NULL && commDev->ackMr == NULL &&
                   commDev->ackBuffer == NULL && commDev->fifoMr == NULL &&
                   commDev->retransHdrMr == NULL &&
                   comm->remSizesFifo.mrs[i] == NULL &&
                   commDev->base.cq == NULL && commDev->base.pd == NULL;
  }
  if (allReleased) {
    free(comm);
    *released = true;
  } else if (result == flagcxSuccess) {
    result = flagcxInternalError;
  }
  return result;
}

static flagcxResult_t flagcxIbucCleanupRecv(struct flagcxIbRecvComm *comm,
                                            bool *released) {
  if (released == NULL)
    return flagcxInvalidArgument;
  *released = comm == NULL;
  if (comm == NULL)
    return flagcxSuccess;

  flagcxResult_t result = flagcxSuccess;
  flagcxResult_t current = flagcxIbDrainDeferredMrsWithCallback(
      &comm->base, flagcxIbucDeregMrInternal);
  flagcxIbucRecordCleanupError(current, &result);

  if (comm->retrans.enabled) {
    current = flagcxIbRetransDestroy(&comm->retrans);
    flagcxIbucRecordCleanupError(current, &result);
    if (current == flagcxSuccess)
      comm->retrans.enabled = 0;
  }
  flagcxIbucRecordCleanupError(flagcxSocketClose(&comm->base.sock), &result);

  for (int i = 0; i < comm->base.ndevs; i++) {
    struct flagcxIbNetCommDevBase *base = &comm->devs[i].base;
    if (base->cq != NULL) {
      struct ibv_wc wcs[64];
      int nCqe = 0;
      for (int j = 0; j < 16; j++) {
        if (flagcxWrapIbvPollCq(base->cq, 64, wcs, &nCqe) != flagcxSuccess ||
            nCqe == 0)
          break;
      }
    }
  }

  for (int i = 0; i < comm->base.ndevs; i++) {
    current = flagcxIbDestroyCtrlQp(&comm->devs[i].ctrlQp);
    flagcxIbucRecordCleanupError(current, &result);
    current = flagcxIbucDestroyQp(&comm->devs[i].retransQp);
    flagcxIbucRecordCleanupError(current, &result);
  }
  for (int q = 0; q < comm->base.nqps; q++) {
    current = flagcxIbucDestroyQp(&comm->base.qps[q]);
    flagcxIbucRecordCleanupError(current, &result);
  }

  for (int i = 0; i < comm->base.ndevs; i++) {
    struct flagcxIbRecvCommDev *commDev = &comm->devs[i];
    if (commDev->ctrlQp.qp == NULL) {
      current = flagcxIbucDestroyMr(&commDev->ackMr);
      flagcxIbucRecordCleanupError(current, &result);
      if (commDev->ackMr == NULL && commDev->ackBuffer != NULL) {
        free(commDev->ackBuffer);
        commDev->ackBuffer = NULL;
      }
    }
    if (commDev->gpuFlush.qp.qp != NULL) {
      current = flagcxIbucDestroyQp(&commDev->gpuFlush.qp);
      flagcxIbucRecordCleanupError(current, &result);
    }
    if (commDev->gpuFlush.qp.qp == NULL) {
      current = flagcxIbucDestroyMr(&commDev->gpuFlush.hostMr);
      flagcxIbucRecordCleanupError(current, &result);
    }
    if (commDev->retransQp.qp == NULL) {
      current = flagcxIbucDestroyMr(&commDev->retransRecvMr);
      flagcxIbucRecordCleanupError(current, &result);
      if (commDev->retransRecvMr == NULL && commDev->retransRecvBufCount > 0) {
        free(commDev->retransRecvBufs[0]);
        memset(commDev->retransRecvBufs, 0, sizeof(commDev->retransRecvBufs));
        commDev->retransRecvBufCount = 0;
      }
    }
    if (flagcxIbucDataQpsReleased(&comm->base, i)) {
      current = flagcxIbucDestroyMr(&commDev->fifoMr);
      flagcxIbucRecordCleanupError(current, &result);
      current = flagcxIbucDestroyMr(&commDev->sizesFifoMr);
      flagcxIbucRecordCleanupError(current, &result);
    }
  }

  for (int i = 0; i < comm->base.ndevs; i++) {
    struct flagcxIbRecvCommDev *commDev = &comm->devs[i];
    const bool hasDependencies =
        comm->base.deferredMrHandles != NULL ||
        !flagcxIbucCtrlQpReleased(&commDev->ctrlQp) ||
        commDev->retransQp.qp != NULL ||
        !flagcxIbucDataQpsReleased(&comm->base, i) || commDev->ackMr != NULL ||
        commDev->ackBuffer != NULL || commDev->gpuFlush.qp.qp != NULL ||
        commDev->gpuFlush.hostMr != NULL || commDev->fifoMr != NULL ||
        commDev->sizesFifoMr != NULL || commDev->retransRecvMr != NULL ||
        commDev->retransRecvBufCount != 0;
    if (!hasDependencies) {
      current = flagcxIbucDestroyBase(&commDev->base);
      flagcxIbucRecordCleanupError(current, &result);
    }
  }

  bool allReleased = comm->base.deferredMrHandles == NULL;
  for (int q = 0; q < comm->base.nqps; q++)
    allReleased &= comm->base.qps[q].qp == NULL;
  for (int i = 0; i < comm->base.ndevs; i++) {
    const struct flagcxIbRecvCommDev *commDev = &comm->devs[i];
    allReleased &=
        flagcxIbucCtrlQpReleased(&commDev->ctrlQp) &&
        commDev->retransQp.qp == NULL && commDev->ackMr == NULL &&
        commDev->ackBuffer == NULL && commDev->gpuFlush.qp.qp == NULL &&
        commDev->gpuFlush.hostMr == NULL && commDev->fifoMr == NULL &&
        commDev->sizesFifoMr == NULL && commDev->retransRecvMr == NULL &&
        commDev->retransRecvBufCount == 0 && commDev->base.cq == NULL &&
        commDev->base.pd == NULL;
  }
  if (allReleased) {
    free(comm);
    *released = true;
  } else if (result == flagcxSuccess) {
    result = flagcxInternalError;
  }
  return result;
}

static pthread_mutex_t flagcxIbucDeferredCleanupLock =
    PTHREAD_MUTEX_INITIALIZER;
static struct flagcxIbNetCommBase *flagcxIbucDeferredCleanupHead = NULL;

static void flagcxIbucRetainDeferredCleanup(struct flagcxIbNetCommBase *base) {
  if (base == NULL)
    return;
  pthread_mutex_lock(&flagcxIbucDeferredCleanupLock);
  if (!base->cleanupDeferred) {
    base->nextDeferredCleanup = flagcxIbucDeferredCleanupHead;
    flagcxIbucDeferredCleanupHead = base;
    base->cleanupDeferred = true;
  }
  pthread_mutex_unlock(&flagcxIbucDeferredCleanupLock);
}

static void flagcxIbucDrainDeferredCleanup(void) {
  pthread_mutex_lock(&flagcxIbucDeferredCleanupLock);
  struct flagcxIbNetCommBase *pending = flagcxIbucDeferredCleanupHead;
  flagcxIbucDeferredCleanupHead = NULL;
  for (struct flagcxIbNetCommBase *base = pending; base != NULL;
       base = base->nextDeferredCleanup) {
    base->cleanupDeferred = false;
  }
  pthread_mutex_unlock(&flagcxIbucDeferredCleanupLock);

  while (pending != NULL) {
    struct flagcxIbNetCommBase *base = pending;
    pending = base->nextDeferredCleanup;
    base->nextDeferredCleanup = NULL;

    bool released = false;
    flagcxResult_t result =
        base->isSend
            ? flagcxIbucCleanupSend((struct flagcxIbSendComm *)base, &released)
            : flagcxIbucCleanupRecv((struct flagcxIbRecvComm *)base, &released);
    if (!released) {
      flagcxIbucRetainDeferredCleanup(base);
      TRACE(FLAGCX_NET,
            "NET/IBUC : deferred communicator cleanup still pending with "
            "result %d",
            result);
    }
  }
}

static void
flagcxIbucRetainAndRetryDeferredCleanup(struct flagcxIbNetCommBase *base) {
  // A close/setup-failure path consumes its public handle. Queue the retained
  // ownership first, then make one final cleanup pass so a one-shot teardown
  // failure cannot leave resources waiting for an unrelated future close.
  // Persistent failures remain on the adaptor-owned list without spinning.
  flagcxIbucRetainDeferredCleanup(base);
  flagcxIbucDrainDeferredCleanup();
}

flagcxResult_t flagcxIbucCloseSend(void *sendComm) {
  struct flagcxIbSendComm *comm = (struct flagcxIbSendComm *)sendComm;
  flagcxIbucDrainDeferredCleanup();
  bool released = false;
  flagcxResult_t result = flagcxIbucCleanupSend(comm, &released);
  // Net adaptor close consumes the public handle even when teardown reports
  // an error. Keep any remaining dependencies reachable for a later internal
  // cleanup pass; existing callers discard the handle after close returns.
  if (!released && comm != NULL)
    flagcxIbucRetainAndRetryDeferredCleanup(&comm->base);
  TIME_PRINT("IBUC");
  return result;
}

flagcxResult_t flagcxIbucCloseRecv(void *recvComm) {
  struct flagcxIbRecvComm *comm = (struct flagcxIbRecvComm *)recvComm;
  flagcxIbucDrainDeferredCleanup();
  bool released = false;
  flagcxResult_t result = flagcxIbucCleanupRecv(comm, &released);
  if (!released && comm != NULL)
    flagcxIbucRetainAndRetryDeferredCleanup(&comm->base);
  return result;
}

flagcxResult_t flagcxIbucCloseListen(void *listenComm) {
  struct flagcxIbListenComm *comm = (struct flagcxIbListenComm *)listenComm;
  if (comm) {
    flagcxIbucDrainDeferredCleanup();
    flagcxResult_t result = flagcxSuccess;
    if (comm->stage.comm != NULL) {
      bool released = false;
      flagcxResult_t cleanupResult = flagcxIbucCleanupRecv(
          (struct flagcxIbRecvComm *)comm->stage.comm, &released);
      if (!released) {
        struct flagcxIbRecvComm *staged =
            (struct flagcxIbRecvComm *)comm->stage.comm;
        flagcxIbucRetainAndRetryDeferredCleanup(&staged->base);
      }
      comm->stage.comm = NULL;
      flagcxIbucRecordCleanupError(cleanupResult, &result);
    }
    free(comm->stage.buffer);
    comm->stage.buffer = NULL;
    flagcxIbucRecordCleanupError(flagcxSocketClose(&comm->sock), &result);
    free(comm);
    return result;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxIbucGetDevFromName(char *name, int *dev) {
  for (int i = 0; i < flagcxNMergedIbDevs; i++) {
    if (strcmp(flagcxIbMergedDevs[i].devName, name) == 0) {
      *dev = i;
      return flagcxSuccess;
    }
  }
  return flagcxSystemError;
}

flagcxResult_t flagcxIbucGetProperties(int dev, void *props) {
  struct flagcxIbMergedDev *mergedDev = flagcxIbMergedDevs + dev;
  flagcxNetProperties_t *properties = (flagcxNetProperties_t *)props;

  properties->name = mergedDev->devName;
  properties->speed = mergedDev->speed;

  // Take the rest of the properties from an arbitrary sub-device
  struct flagcxIbDev *ibucDev = flagcxIbDevs + mergedDev->devs[0];
  properties->pciPath = ibucDev->pciPath;
  properties->guid = ibucDev->guid;
  properties->ptrSupport = FLAGCX_PTR_HOST;

  bool gpuMrSupported = false;
  const int gpuMrAccess =
      IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
  FLAGCXCHECK(flagcxIbProbeGpuMrSupport(dev, gpuMrAccess, &gpuMrSupported));
  if (gpuMrSupported)
    properties->ptrSupport |= FLAGCX_PTR_CUDA;
  properties->regIsGlobal = 1;
  if (flagcxIbDmaBufSupport(dev) == flagcxSuccess) {
    properties->ptrSupport |= FLAGCX_PTR_DMABUF;
  }
  properties->latency = 0; // Not set
  properties->port = ibucDev->portNum + ibucDev->realPort;
  properties->maxComms = ibucDev->maxQp;
  properties->maxRecvs = FLAGCX_NET_IB_MAX_RECVS;
  properties->netDeviceType = FLAGCX_NET_DEVICE_HOST;
  properties->netDeviceVersion = FLAGCX_NET_DEVICE_INVALID_VERSION;
  return flagcxSuccess;
}

// One-sided stubs (not supported by IBUC adaptor)
// Adapter wrapper functions

struct flagcxNetAdaptor flagcxNetIbuc = {
    // Basic functions
    "IBUC", flagcxIbucInit, flagcxIbDevices, flagcxIbucGetProperties,

    // Setup functions
    flagcxIbucListen, flagcxIbucConnect, flagcxIbucAccept, flagcxIbucCloseSend,
    flagcxIbucCloseRecv, flagcxIbucCloseListen,

    // Memory region functions
    flagcxIbucRegMr, flagcxIbucRegMrDmaBuf, flagcxIbucDeregMr,

    // Two-sided functions
    flagcxIbucIsend, flagcxIbucIrecv, flagcxIbucIflush, flagcxIbucTest,

    // One-sided functions
    NULL, // iput - not supported on IBUC
    NULL, // iget - not supported on IBUC
    NULL, // iputSignal - not supported on IBUC

    // Device name lookup
    flagcxIbucGetDevFromName,

    // Optional one-sided batch helpers and MR metadata
    NULL, // iputBatch
    NULL, // testBatch
    NULL, // igetBatch
    NULL, // getMrInfo
};

#endif // USE_IBUC
