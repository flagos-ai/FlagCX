/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "p2p_engine_core.h"

#include <algorithm>

namespace {

void recordTerminal(struct flagcxP2pTransfer *transfer, flagcxResult_t result) {
  if (result != flagcxSuccess && result != flagcxInProgress &&
      transfer->terminalResult == flagcxSuccess)
    transfer->terminalResult = result;
}

flagcxResult_t completeItem(struct flagcxP2pTransfer *transfer, uint32_t index,
                            flagcxResult_t result) {
  if (index >= transfer->items.size())
    return flagcxInvalidArgument;
  struct flagcxP2pTransferItem &item = transfer->items[index];
  if (item.state == FLAGCX_P2P_TRANSFER_COMPLETE)
    return flagcxInvalidArgument;

  if (item.state == FLAGCX_P2P_TRANSFER_INFLIGHT) {
    flagcxResult_t creditResult =
        flagcxTransportCreditRelease(&transfer->credit, 1);
    if (creditResult != flagcxSuccess)
      return creditResult;
  }

  uint32_t advanced = 0;
  flagcxResult_t trackResult = flagcxNetTrackCompletion(
      &transfer->scoreboard, &item.submit, result, &advanced);
  if (trackResult != flagcxSuccess)
    return trackResult;
  flagcxResult_t requestResult =
      flagcxTransportRequestComplete(&transfer->request, 1, result);
  if (requestResult != flagcxSuccess)
    return requestResult;

  item.request = NULL;
  item.state = FLAGCX_P2P_TRANSFER_COMPLETE;
  transfer->completed++;
  recordTerminal(transfer, result);
  return flagcxSuccess;
}

flagcxResult_t failQueued(struct flagcxP2pTransfer *transfer,
                          flagcxResult_t result) {
  recordTerminal(transfer, result);
  for (uint32_t i = 0; i < transfer->items.size(); ++i) {
    if (transfer->items[i].state != FLAGCX_P2P_TRANSFER_QUEUED)
      continue;
    flagcxResult_t completeResult = completeItem(transfer, i, result);
    if (completeResult != flagcxSuccess)
      return completeResult;
  }
  return flagcxSuccess;
}

flagcxResult_t pollInflight(struct flagcxP2pTransfer *transfer) {
  for (uint32_t i = 0; i < transfer->items.size(); ++i) {
    struct flagcxP2pTransferItem &item = transfer->items[i];
    if (item.state != FLAGCX_P2P_TRANSFER_INFLIGHT)
      continue;
    int done = 0;
    flagcxResult_t testResult =
        transfer->backend.test(transfer->backend.context, item.request, &done);
    if (!done) {
      // A polling failure does not cancel an accepted native request. Keep
      // its credit and buffers until the backend confirms completion.
      if (testResult != flagcxSuccess && testResult != flagcxInProgress) {
        recordTerminal(transfer, testResult);
        flagcxResult_t failResult = failQueued(transfer, testResult);
        if (failResult != flagcxSuccess)
          return failResult;
      }
      continue;
    }
    // Once a backend reports done it has relinquished the native request
    // object, so the Engine must retire the item before that slot can be
    // reused. InProgress is only meaningful while done is false; normalize a
    // completed request carrying that value to a permanent contract error.
    if (done && testResult == flagcxInProgress)
      testResult = flagcxInternalError;
    if (testResult != flagcxSuccess) {
      flagcxResult_t completeResult = completeItem(transfer, i, testResult);
      if (completeResult != flagcxSuccess)
        return completeResult;
      flagcxResult_t failResult = failQueued(transfer, testResult);
      if (failResult != flagcxSuccess)
        return failResult;
      continue;
    }
    if (done) {
      flagcxResult_t completeResult = completeItem(transfer, i, flagcxSuccess);
      if (completeResult != flagcxSuccess)
        return completeResult;
    }
  }
  return flagcxSuccess;
}

flagcxResult_t postQueued(struct flagcxP2pTransfer *transfer) {
  if (transfer->terminalResult != flagcxSuccess)
    return flagcxSuccess;

  uint32_t available = 0;
  flagcxResult_t availableResult =
      flagcxTransportCreditAvailable(&transfer->credit, &available);
  if (availableResult != flagcxSuccess || available == 0)
    return availableResult;

  std::vector<uint32_t> indices;
  const uint32_t limit = std::min(available, transfer->maxPostBatch);
  indices.reserve(limit);
  for (uint32_t i = 0; i < transfer->items.size() && indices.size() < limit;
       ++i) {
    if (transfer->items[i].state == FLAGCX_P2P_TRANSFER_QUEUED)
      indices.push_back(i);
  }
  if (indices.empty())
    return flagcxSuccess;

  const uint32_t requested = static_cast<uint32_t>(indices.size());
  flagcxResult_t creditResult =
      flagcxTransportCreditAcquire(&transfer->credit, requested);
  if (creditResult != flagcxSuccess)
    return creditResult == flagcxInProgress ? flagcxSuccess : creditResult;

  std::vector<struct flagcxP2pTransferOp> ops(requested);
  std::vector<void *> requests(requested, NULL);
  for (uint32_t i = 0; i < requested; ++i)
    ops[i] = transfer->items[indices[i]].op;

  flagcxTransportPostResult post = {};
  flagcxResult_t callResult = transfer->backend.post(
      transfer->backend.context, ops.data(), requested, requests.data(), &post);
  if (callResult != flagcxSuccess) {
    flagcxTransportCreditRelease(&transfer->credit, requested);
    if (callResult == flagcxInProgress)
      return flagcxSuccess;
    return failQueued(transfer, callResult);
  }
  if (post.requested != static_cast<int>(requested) || post.accepted < 0 ||
      post.accepted > static_cast<int>(requested)) {
    flagcxTransportCreditRelease(&transfer->credit, requested);
    return failQueued(transfer, flagcxInternalError);
  }

  const uint32_t accepted = static_cast<uint32_t>(post.accepted);
  if (accepted < requested) {
    flagcxResult_t releaseResult =
        flagcxTransportCreditRelease(&transfer->credit, requested - accepted);
    if (releaseResult != flagcxSuccess)
      return releaseResult;
  }
  for (uint32_t i = 0; i < accepted; ++i) {
    struct flagcxP2pTransferItem &item = transfer->items[indices[i]];
    item.request = requests[i];
    item.state = FLAGCX_P2P_TRANSFER_INFLIGHT;
    transfer->submitted++;
    // A null native request is the shared transport convention for an
    // accepted operation that completed synchronously.
    if (item.request == NULL) {
      flagcxResult_t completeResult =
          completeItem(transfer, indices[i], flagcxSuccess);
      if (completeResult != flagcxSuccess)
        return completeResult;
    }
  }

  if (post.result != flagcxSuccess && post.result != flagcxInProgress)
    return failQueued(transfer, post.result);
  return flagcxSuccess;
}

} // namespace

flagcxResult_t
flagcxP2pTransferInit(struct flagcxP2pTransfer *transfer,
                      const struct flagcxP2pTransferBackend *backend,
                      const struct flagcxP2pTransferOp *ops, uint32_t count,
                      uint32_t maxInFlight, uint32_t maxPostBatch,
                      uint64_t groupId, uint64_t generation) {
  if (transfer == NULL || backend == NULL || backend->post == NULL ||
      backend->test == NULL || ops == NULL || count == 0 || maxInFlight == 0 ||
      maxPostBatch == 0 || groupId == 0 || transfer->initialized)
    return flagcxInvalidArgument;

  transfer->backend = *backend;
  transfer->items.assign(count, flagcxP2pTransferItem());
  transfer->completionEntries.assign(count, flagcxNetCompletionEntry());
  transfer->maxPostBatch = maxPostBatch;
  transfer->submitted = 0;
  transfer->completed = 0;
  transfer->terminalResult = flagcxSuccess;

  flagcxResult_t result =
      flagcxTransportCreditInit(&transfer->credit, maxInFlight);
  if (result != flagcxSuccess)
    return result;
  flagcxTransportRequestInit(&transfer->request);
  result = flagcxTransportRequestAcquire(&transfer->request);
  if (result != flagcxSuccess)
    return result;
  result = flagcxTransportRequestAddPending(&transfer->request, count);
  if (result != flagcxSuccess)
    return result;
  result =
      flagcxNetReleaseGroupInit(&transfer->releaseGroup, groupId, generation);
  if (result != flagcxSuccess)
    return result;
  result = flagcxNetCompletionScoreboardInit(&transfer->scoreboard,
                                             transfer->completionEntries.data(),
                                             count, generation, 0);
  if (result != flagcxSuccess)
    return result;

  for (uint32_t i = 0; i < count; ++i) {
    struct flagcxP2pTransferItem &item = transfer->items[i];
    item.op = ops[i];
    item.op.groupId = groupId;
    item.op.generation = generation;
    item.op.sequence = i;
    item.submit.orderingKey = ops[i].orderingKey;
    item.submit.groupId = groupId;
    item.submit.generation = generation;
    item.submit.sequence = i;
    item.submit.flags = ops[i].submitFlags;
    item.submit.laneMask = ops[i].laneMask;
    result = flagcxNetTrackSubmit(&transfer->scoreboard, &item.submit,
                                  &transfer->releaseGroup);
    if (result != flagcxSuccess)
      return result;
  }
  result = flagcxNetReleaseGroupSeal(&transfer->releaseGroup);
  if (result != flagcxSuccess)
    return result;
  transfer->initialized = 1;

  for (uint32_t i = 0; i < count; ++i) {
    if (transfer->items[i].op.size == 0) {
      result = completeItem(transfer, i, flagcxSuccess);
      if (result != flagcxSuccess)
        return result;
    }
  }
  return flagcxSuccess;
}

flagcxResult_t
flagcxP2pTransferProgress(struct flagcxP2pTransfer *transfer,
                          struct flagcxP2pTransferStatus *status) {
  if (transfer == NULL || status == NULL || !transfer->initialized)
    return flagcxInvalidArgument;
  flagcxResult_t result = pollInflight(transfer);
  if (result == flagcxSuccess)
    result = postQueued(transfer);
  if (result != flagcxSuccess)
    recordTerminal(transfer, result);

  flagcxResult_t queryResult = flagcxP2pTransferQuery(transfer, status);
  return result != flagcxSuccess ? result : queryResult;
}

flagcxResult_t
flagcxP2pTransferProgressMany(struct flagcxP2pTransfer *const *transfers,
                              uint32_t count) {
  if (transfers == NULL && count != 0)
    return flagcxInvalidArgument;

  flagcxResult_t firstError = flagcxSuccess;
  for (uint32_t i = 0; i < count; ++i) {
    if (transfers[i] == NULL) {
      if (firstError == flagcxSuccess)
        firstError = flagcxInvalidArgument;
      continue;
    }
    struct flagcxP2pTransferStatus status = {};
    const flagcxResult_t result =
        flagcxP2pTransferProgress(transfers[i], &status);
    if (result != flagcxSuccess && firstError == flagcxSuccess)
      firstError = result;
  }
  return firstError;
}

flagcxResult_t flagcxP2pTransferQuery(struct flagcxP2pTransfer *transfer,
                                      struct flagcxP2pTransferStatus *status) {
  if (transfer == NULL || status == NULL || !transfer->initialized)
    return flagcxInvalidArgument;

  uint64_t nextSequence = 0;
  uint32_t scoreboardInFlight = 0;
  flagcxResult_t scoreboardError = flagcxSuccess;
  flagcxResult_t result =
      flagcxNetCompletionScoreboardQuery(&transfer->scoreboard, &nextSequence,
                                         &scoreboardInFlight, &scoreboardError);
  if (result != flagcxSuccess)
    return result;

  int requestDone = 0;
  const flagcxResult_t requestResult =
      flagcxTransportRequestTest(&transfer->request, &requestDone);
  int groupDone = 0;
  int releaseAllowed = 0;
  const flagcxResult_t groupResult = flagcxNetReleaseGroupTest(
      &transfer->releaseGroup, &groupDone, &releaseAllowed);

  status->requested = static_cast<uint32_t>(transfer->items.size());
  status->submitted = transfer->submitted;
  status->completed = transfer->completed;
  status->retired = static_cast<uint32_t>(nextSequence);
  status->pending = scoreboardInFlight;
  uint32_t available = 0;
  result = flagcxTransportCreditAvailable(&transfer->credit, &available);
  if (result != flagcxSuccess)
    return result;
  status->inFlight = transfer->credit.capacity - available;
  status->done = requestDone && groupDone && scoreboardInFlight == 0;
  status->result = transfer->terminalResult;
  if (status->result == flagcxSuccess && requestDone &&
      requestResult != flagcxSuccess)
    status->result = requestResult;
  if (status->result == flagcxSuccess && groupDone &&
      groupResult != flagcxSuccess)
    status->result = groupResult;
  if (status->result == flagcxSuccess && scoreboardError != flagcxSuccess)
    status->result = scoreboardError;
  status->releaseAllowed =
      status->done && status->result == flagcxSuccess ? releaseAllowed : 0;
  return flagcxSuccess;
}

flagcxResult_t flagcxP2pTransferReset(struct flagcxP2pTransfer *transfer) {
  if (transfer == NULL || !transfer->initialized)
    return flagcxInvalidArgument;
  struct flagcxP2pTransferStatus status;
  flagcxResult_t result = flagcxP2pTransferQuery(transfer, &status);
  if (result != flagcxSuccess)
    return result;
  if (!status.done)
    return flagcxInProgress;
  result = flagcxTransportRequestRelease(&transfer->request);
  if (result != flagcxSuccess)
    return result;
  transfer->items.clear();
  transfer->completionEntries.clear();
  transfer->scoreboard = {};
  transfer->releaseGroup = {};
  transfer->credit = {};
  transfer->backend = {};
  transfer->maxPostBatch = 0;
  transfer->submitted = 0;
  transfer->completed = 0;
  transfer->terminalResult = flagcxSuccess;
  transfer->initialized = 0;
  return flagcxSuccess;
}
