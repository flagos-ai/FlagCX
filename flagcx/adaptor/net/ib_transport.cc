/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "ib_transport.h"

#include "ibvwrap.h"
#include "onesided_types.h"

flagcxResult_t flagcxIbSelectLane(struct flagcxIbNetCommBase *base,
                                  enum flagcxNetLaneMode mode,
                                  uint64_t orderingKey,
                                  struct flagcxIbLane *lane) {
  if (base == NULL || lane == NULL || base->ready == 0 || base->nqps <= 0)
    return flagcxInvalidArgument;

  struct flagcxNetLaneSet lanes = {
      (uint32_t)base->nqps,
      (uint32_t)base->qpIndex,
  };
  uint32_t laneIndex = 0;
  FLAGCXCHECK(flagcxNetSelectLane(&lanes, mode, orderingKey, &laneIndex));
  if (laneIndex >= (uint32_t)base->nqps)
    return flagcxInternalError;

  struct flagcxIbQp *qp = &base->qps[laneIndex];
  lane->base.index = laneIndex;
  lane->base.localDevIndex = qp->devIndex;
  lane->base.remoteDevIndex = qp->remDevIdx;
  lane->ibQp = qp;
  return qp->qp == NULL ? flagcxInvalidArgument : flagcxSuccess;
}

flagcxResult_t flagcxIbCommitLane(struct flagcxIbNetCommBase *base,
                                  enum flagcxNetLaneMode mode,
                                  const struct flagcxIbLane *lane) {
  if (base == NULL || lane == NULL || base->nqps <= 0)
    return flagcxInvalidArgument;
  struct flagcxNetLaneSet lanes = {
      (uint32_t)base->nqps,
      (uint32_t)base->qpIndex,
  };
  FLAGCXCHECK(flagcxNetCommitLane(&lanes, mode, lane->base.index));
  base->qpIndex = (int)lanes.unorderedCursor;
  return flagcxSuccess;
}

static flagcxResult_t flagcxIbAcceptedPrefix(struct ibv_send_wr *head,
                                             struct ibv_send_wr *badWr,
                                             int count, int *accepted) {
  if (head == NULL || accepted == NULL || count <= 0)
    return flagcxInvalidArgument;
  if (badWr == NULL) {
    *accepted = 0;
    return flagcxSuccess;
  }

  int prefix = 0;
  for (struct ibv_send_wr *wr = head; wr != NULL && prefix < count;
       wr = wr->next, ++prefix) {
    if (wr == badWr) {
      *accepted = prefix;
      return flagcxSuccess;
    }
  }
  return flagcxInvalidArgument;
}

flagcxResult_t flagcxIbPostSendList(const struct flagcxIbLane *lane,
                                    struct ibv_send_wr *head, int count,
                                    bool retryable,
                                    struct flagcxNetPostResult *post) {
  if (lane == NULL || lane->ibQp == NULL || lane->ibQp->qp == NULL ||
      head == NULL || count <= 0 || post == NULL)
    return flagcxInvalidArgument;

  struct ibv_send_wr *badWr = NULL;
  flagcxResult_t result =
      retryable ? flagcxWrapIbvPostSendRetryable(lane->ibQp->qp, head, &badWr)
                : flagcxWrapIbvPostSend(lane->ibQp->qp, head, &badWr);
  int accepted = count;
  if (result != flagcxSuccess) {
    flagcxResult_t prefixResult =
        flagcxIbAcceptedPrefix(head, badWr, count, &accepted);
    if (prefixResult != flagcxSuccess) {
      // The post call itself has already happened. Report malformed provider
      // bookkeeping as a terminal post result so compatibility callers can
      // release requests just like any other failed submission.
      return flagcxNetPostResultInit(post, count, 0, flagcxInternalError);
    }
  }
  return flagcxNetPostResultInit(post, count, accepted, result);
}

flagcxResult_t flagcxIbResolveOneSideRange(
    const struct flagcxOneSideHandleInfo *info, int rank, uint64_t offset,
    size_t size, const struct flagcxIbLane *lane, bool local,
    struct flagcxNetResolvedRange *range, uint32_t *key) {
  if (lane == NULL || range == NULL || key == NULL || size > UINT32_MAX)
    return flagcxInvalidArgument;
  FLAGCXCHECK(flagcxNetResolveOneSideRange(info, rank, offset, size, range));

  const int keyIndex =
      local ? lane->base.localDevIndex : lane->base.remoteDevIndex;
  if (keyIndex < 0 || range->mrInfo == NULL ||
      range->mrInfo->nKeys > FLAGCX_NET_MAX_MR_KEYS ||
      (uint32_t)keyIndex >= range->mrInfo->nKeys)
    return flagcxInvalidArgument;
  *key =
      local ? range->mrInfo->lkeys[keyIndex] : range->mrInfo->rkeys[keyIndex];
  return flagcxSuccess;
}
