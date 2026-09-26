/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_IB_TRANSPORT_H_
#define FLAGCX_IB_TRANSPORT_H_

#include "ib_common.h"
#include "net_transport.h"

struct flagcxIbLane {
  struct flagcxNetLane base;
  struct flagcxIbQp *ibQp;
};

flagcxResult_t flagcxIbSelectLane(struct flagcxIbNetCommBase *base,
                                  enum flagcxNetLaneMode mode,
                                  uint64_t orderingKey,
                                  struct flagcxIbLane *lane);
flagcxResult_t flagcxIbCommitLane(struct flagcxIbNetCommBase *base,
                                  enum flagcxNetLaneMode mode,
                                  const struct flagcxIbLane *lane);

flagcxResult_t flagcxIbPostSendList(const struct flagcxIbLane *lane,
                                    struct ibv_send_wr *head, int count,
                                    bool retryable,
                                    struct flagcxNetPostResult *post);

flagcxResult_t flagcxIbResolveOneSideRange(
    const struct flagcxOneSideHandleInfo *info, int rank, uint64_t offset,
    size_t size, const struct flagcxIbLane *lane, bool local,
    struct flagcxNetResolvedRange *range, uint32_t *key);

#endif // FLAGCX_IB_TRANSPORT_H_
