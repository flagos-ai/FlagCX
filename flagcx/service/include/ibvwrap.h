/*************************************************************************
 * Copyright (c) 2004, 2005 Topspin Communications.  All rights reserved.
 * Copyright (c) 2004, 2011-2012 Intel Corporation.  All rights reserved.
 * Copyright (c) 2005, 2006, 2007 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2005 PathScale, Inc.  All rights reserved.
 *
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef FLAGCX_IBVWRAP_H_
#define FLAGCX_IBVWRAP_H_

#ifdef FLAGCX_BUILD_RDMA_CORE
#include <infiniband/verbs.h>
#else
#include "ibvcore.h"
#endif

#include "core.h"
#include <errno.h>
#include <sys/types.h>
#include <unistd.h>

typedef enum ibv_return_enum {
  IBV_SUCCESS = 0, //!< The operation was successful
} ibv_return_t;

flagcxResult_t flagcxWrapIbvSymbols(void);
/* FLAGCX wrappers of IB verbs functions */
flagcxResult_t flagcxWrapIbvForkInit(void);
flagcxResult_t flagcxWrapIbvGetDeviceList(struct ibv_device ***ret,
                                          int *num_devices);
flagcxResult_t flagcxWrapIbvFreeDeviceList(struct ibv_device **list);
const char *flagcxWrapIbvGetDeviceName(struct ibv_device *device);
flagcxResult_t flagcxWrapIbvOpenDevice(struct ibv_context **ret,
                                       struct ibv_device *device);
flagcxResult_t flagcxWrapIbvCloseDevice(struct ibv_context *context);
flagcxResult_t flagcxWrapIbvGetAsyncEvent(struct ibv_context *context,
                                          struct ibv_async_event *event);
flagcxResult_t flagcxWrapIbvAckAsyncEvent(struct ibv_async_event *event);
flagcxResult_t flagcxWrapIbvQueryDevice(struct ibv_context *context,
                                        struct ibv_device_attr *device_attr);
flagcxResult_t flagcxWrapIbvQueryPort(struct ibv_context *context,
                                      uint8_t port_num,
                                      struct ibv_port_attr *port_attr);
flagcxResult_t flagcxWrapIbvQueryGid(struct ibv_context *context,
                                     uint8_t port_num, int index,
                                     union ibv_gid *gid);
flagcxResult_t flagcxWrapIbvQueryQp(struct ibv_qp *qp, struct ibv_qp_attr *attr,
                                    int attr_mask,
                                    struct ibv_qp_init_attr *init_attr);
flagcxResult_t flagcxWrapIbvAllocPd(struct ibv_pd **ret,
                                    struct ibv_context *context);
flagcxResult_t flagcxWrapIbvDeallocPd(struct ibv_pd *pd);
flagcxResult_t flagcxWrapIbvRegMr(struct ibv_mr **ret, struct ibv_pd *pd,
                                  void *addr, size_t length, int access);
struct ibv_mr *flagcxWrapDirectIbvRegMr(struct ibv_pd *pd, void *addr,
                                        size_t length, int access);
flagcxResult_t flagcxWrapIbvRegMrIova2(struct ibv_mr **ret, struct ibv_pd *pd,
                                       void *addr, size_t length, uint64_t iova,
                                       int access);
/* DMA-BUF support */
flagcxResult_t flagcxWrapIbvRegDmabufMr(struct ibv_mr **ret, struct ibv_pd *pd,
                                        uint64_t offset, size_t length,
                                        uint64_t iova, int fd, int access);
struct ibv_mr *flagcxWrapDirectIbvRegDmabufMr(struct ibv_pd *pd,
                                              uint64_t offset, size_t length,
                                              uint64_t iova, int fd,
                                              int access);
flagcxResult_t flagcxWrapIbvDeregMr(struct ibv_mr *mr);
flagcxResult_t flagcxWrapIbvCreateCompChannel(struct ibv_comp_channel **ret,
                                              struct ibv_context *context);
flagcxResult_t
flagcxWrapIbvDestroyCompChannel(struct ibv_comp_channel *channel);
flagcxResult_t flagcxWrapIbvCreateCq(struct ibv_cq **ret,
                                     struct ibv_context *context, int cqe,
                                     void *cq_context,
                                     struct ibv_comp_channel *channel,
                                     int comp_vector);
flagcxResult_t flagcxWrapIbvDestroyCq(struct ibv_cq *cq);
static inline flagcxResult_t flagcxWrapIbvPollCq(struct ibv_cq *cq,
                                                 int num_entries,
                                                 struct ibv_wc *wc,
                                                 int *num_done) {
  int done = cq->context->ops.poll_cq(
      cq, num_entries, wc); /*returns the number of wcs or 0 on success, a
                               negative number otherwise*/
  if (done < 0) {
    WARN("Call to ibv_poll_cq() returned %d", done);
    return flagcxSystemError;
  }
  *num_done = done;
  return flagcxSuccess;
}
flagcxResult_t flagcxWrapIbvCreateQp(struct ibv_qp **ret, struct ibv_pd *pd,
                                     struct ibv_qp_init_attr *qp_init_attr);
flagcxResult_t flagcxWrapIbvModifyQp(struct ibv_qp *qp,
                                     struct ibv_qp_attr *attr, int attr_mask);
flagcxResult_t flagcxWrapIbvDestroyQp(struct ibv_qp *qp);
flagcxResult_t flagcxWrapIbvQueryEce(struct ibv_qp *qp, struct ibv_ece *ece,
                                     int *supported);
flagcxResult_t flagcxWrapIbvSetEce(struct ibv_qp *qp, struct ibv_ece *ece,
                                   int *supported);
/* SRQ support */
flagcxResult_t flagcxWrapIbvCreateSrq(struct ibv_srq **ret, struct ibv_pd *pd,
                                      struct ibv_srq_init_attr *srq_init_attr);
flagcxResult_t flagcxWrapIbvDestroySrq(struct ibv_srq *srq);

static inline flagcxResult_t
flagcxWrapIbvPostSrqRecv(struct ibv_srq *srq, struct ibv_recv_wr *wr,
                         struct ibv_recv_wr **bad_wr) {
  int ret = srq->context->ops.post_srq_recv(
      srq, wr, bad_wr); /*returns 0 on success, or the value of errno on failure
                          (which indicates the failure reason)*/
  if (ret != IBV_SUCCESS) {
    WARN("ibv_post_srq_recv() failed with error %s", strerror(ret));
    return flagcxSystemError;
  }
  return flagcxSuccess;
}

static inline flagcxResult_t
flagcxWrapIbvPostSend(struct ibv_qp *qp, struct ibv_send_wr *wr,
                      struct ibv_send_wr **bad_wr) {
  int ret = qp->context->ops.post_send(
      qp, wr, bad_wr); /*returns 0 on success, or the value of errno on failure
                          (which indicates the failure reason)*/
  if (ret != IBV_SUCCESS) {
    // Don't warn on ENOMEM (Cannot allocate memory) as it's expected when send queue is full
    if (ret != ENOMEM) {
      WARN("ibv_post_send() failed with error %s, Bad WR %p, First WR %p",
           strerror(ret), wr, *bad_wr);
    }
    return flagcxSystemError;
  }
  return flagcxSuccess;
}

static inline flagcxResult_t
flagcxWrapIbvPostRecv(struct ibv_qp *qp, struct ibv_recv_wr *wr,
                      struct ibv_recv_wr **bad_wr) {
  int ret = qp->context->ops.post_recv(
      qp, wr, bad_wr); /*returns 0 on success, or the value of errno on failure
                          (which indicates the failure reason)*/
  if (ret != IBV_SUCCESS) {
    WARN("ibv_post_recv() failed with error %s", strerror(ret));
    return flagcxSystemError;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxWrapIbvEventTypeStr(char **ret, enum ibv_event_type event);

#endif // End include guard
