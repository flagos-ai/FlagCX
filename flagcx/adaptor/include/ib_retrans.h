/*************************************************************************
 * Copyright (c) 2023 BAAI. All rights reserved.
 * All rights reserved.
 *
 * IBUC Retransmission Support - Header
 ************************************************************************/

#ifndef FLAGCX_IBUC_RETRANS_H_
#define FLAGCX_IBUC_RETRANS_H_

#include "flagcx_common.h"
#include "ib_common.h"
#include <stdint.h>
#include <time.h>

// Retransmission constants
#define FLAGCX_RETRANS_MAGIC                                                   \
  0xDEADBEEF // Magic number for retransmission header
#define FLAGCX_RETRANS_WR_ID                                                   \
  0xFFFFFFFEULL // WR ID for retransmission completions

extern int64_t flagcxParamIbRetransEnable(void);
extern int64_t flagcxParamIbRetransTimeout(void);
extern int64_t flagcxParamIbRetransMaxRetry(void);
extern int64_t flagcxParamIbRetransAckInterval(void);
extern int64_t flagcxParamIbMaxOutstanding(void);

static inline uint64_t flagcxIbGetTimeUs(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

static inline int flagcxIbSeqLess(uint32_t a, uint32_t b) {
  uint16_t a16 = a & 0xFFFF;
  uint16_t b16 = b & 0xFFFF;
  return (int16_t)(a16 - b16) < 0;
}

static inline int flagcxIbSeqLeq(uint32_t a, uint32_t b) {
  uint16_t a16 = a & 0xFFFF;
  uint16_t b16 = b & 0xFFFF;
  return (int16_t)(a16 - b16) <= 0;
}

flagcxResult_t flagcxIbRetransInit(struct flagcxIbRetransState *state);

flagcxResult_t flagcxIbRetransDestroy(struct flagcxIbRetransState *state);

flagcxResult_t flagcxIbRetransAddPacket(struct flagcxIbRetransState *state,
                                        uint32_t seq, uint32_t size, void *data,
                                        uint64_t remote_addr, uint32_t *lkeys,
                                        uint32_t *rkeys);

flagcxResult_t flagcxIbRetransProcessAck(struct flagcxIbRetransState *state,
                                         struct flagcxIbAckMsg *ack_msg);

flagcxResult_t flagcxIbRetransCheckTimeout(struct flagcxIbRetransState *state,
                                           struct flagcxIbSendComm *comm);

flagcxResult_t flagcxIbRetransRecvPacket(struct flagcxIbRetransState *state,
                                         uint32_t seq,
                                         struct flagcxIbAckMsg *ack_msg,
                                         int *should_ack);

flagcxResult_t flagcxIbRetransPiggybackAck(struct flagcxIbSendFifo *fifo_elem,
                                           struct flagcxIbAckMsg *ack_msg);

flagcxResult_t flagcxIbRetransExtractAck(struct flagcxIbSendFifo *fifo_elem,
                                         struct flagcxIbAckMsg *ack_msg);

static inline uint32_t flagcxIbEncodeImmData(uint32_t seq, uint32_t size) {
  return ((seq & 0xFFFF) << 16) | (size & 0xFFFF);
}

static inline void flagcxIbDecodeImmData(uint32_t imm_data, uint32_t *seq,
                                         uint32_t *size) {
  *seq = (imm_data >> 16) & 0xFFFF;
  *size = imm_data & 0xFFFF;
}

void flagcxIbRetransPrintStats(struct flagcxIbRetransState *state,
                               const char *prefix);

flagcxResult_t flagcxIbCreateCtrlQp(struct ibv_context *context,
                                    struct ibv_pd *pd, uint8_t port_num,
                                    struct flagcxIbCtrlQp *ctrlQp);

flagcxResult_t flagcxIbDestroyCtrlQp(struct flagcxIbCtrlQp *ctrlQp);

flagcxResult_t
flagcxIbSetupCtrlQpConnection(struct ibv_context *context, struct ibv_pd *pd,
                              struct flagcxIbCtrlQp *ctrlQp,
                              uint32_t remote_qpn, union ibv_gid *remote_gid,
                              uint16_t remote_lid, uint8_t port_num,
                              uint8_t link_layer, uint8_t local_gid_index);

flagcxResult_t flagcxIbRetransSendAckViaUd(struct flagcxIbRecvComm *comm,
                                           struct flagcxIbAckMsg *ack_msg,
                                           int devIndex);

flagcxResult_t flagcxIbRetransRecvAckViaUd(struct flagcxIbSendComm *comm,
                                           int devIndex);

flagcxResult_t flagcxIbRetransResendViaSend(struct flagcxIbSendComm *comm,
                                            uint32_t seq);

flagcxResult_t flagcxIbCreateSrq(struct ibv_context *context, struct ibv_pd *pd,
                                 struct flagcxIbSrqMgr *srqMgr);

flagcxResult_t flagcxIbDestroySrq(struct flagcxIbSrqMgr *srqMgr);

flagcxResult_t flagcxIbSrqPostRecv(struct flagcxIbSrqMgr *srqMgr, int count);

#endif // FLAGCX_IBUC_RETRANS_H_
