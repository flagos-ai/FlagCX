/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "p2p_engine_backend.h"

#include "flagcx_net_adaptor.h"

namespace {

class SubmitScope {
public:
  explicit SubmitScope(const struct flagcxP2pTransferOp &op) {
    struct flagcxNetSubmitContext submit = {};
    submit.orderingKey = op.orderingKey;
    submit.groupId = op.groupId;
    submit.generation = op.generation;
    submit.sequence = op.sequence;
    submit.flags = op.submitFlags;
    submit.laneMask = op.laneMask;
    active_ = flagcxNetSetSubmitContext(&submit) == flagcxSuccess;
  }

  ~SubmitScope() {
    if (active_)
      flagcxNetClearSubmitContext();
  }

  bool active() const { return active_; }

private:
  bool active_ = false;
};

flagcxResult_t post(void *opaque, const struct flagcxP2pTransferOp *ops,
                    uint32_t count, void **requests,
                    flagcxTransportPostResult *postResult) {
  struct flagcxP2pNetBackendContext *context =
      static_cast<struct flagcxP2pNetBackendContext *>(opaque);
  if (context == NULL || context->adaptor == NULL ||
      context->sendComm == NULL || context->progressMutex == NULL ||
      ops == NULL || requests == NULL || postResult == NULL || count == 0)
    return flagcxInvalidArgument;

  std::lock_guard<std::mutex> lock(*context->progressMutex);
  uint32_t accepted = 0;
  flagcxResult_t result = flagcxSuccess;
  for (; accepted < count; ++accepted) {
    const struct flagcxP2pTransferOp &op = ops[accepted];
    SubmitScope scope(op);
    if (!scope.active()) {
      result = flagcxInternalError;
      break;
    }
    void *request = NULL;
    if (context->write) {
      result =
          context->adaptor->iput(context->sendComm, op.srcOffset, op.dstOffset,
                                 op.size, 0, 0, static_cast<void **>(op.srcMr),
                                 static_cast<void **>(op.dstMr), &request);
    } else {
      result =
          context->adaptor->iget(context->sendComm, op.srcOffset, op.dstOffset,
                                 op.size, 0, 0, static_cast<void **>(op.srcMr),
                                 static_cast<void **>(op.dstMr), &request);
    }
    if (result != flagcxSuccess)
      break;
    requests[accepted] = request;
  }

  return flagcxTransportPostResultInit(
      postResult, static_cast<int>(count), static_cast<int>(accepted),
      accepted == count ? flagcxSuccess : result);
}

flagcxResult_t test(void *opaque, void *request, int *done) {
  struct flagcxP2pNetBackendContext *context =
      static_cast<struct flagcxP2pNetBackendContext *>(opaque);
  if (context == NULL || context->adaptor == NULL ||
      context->progressMutex == NULL || done == NULL)
    return flagcxInvalidArgument;
  std::lock_guard<std::mutex> lock(*context->progressMutex);
  const flagcxResult_t result = context->adaptor->test(request, done, NULL);
  if (result != flagcxSuccess && result != flagcxInProgress &&
      context->quiesce != NULL) {
    // A communicator error can make IBRC report done before sibling QPs have
    // stopped accessing their MRs. Retire only after provider quiescence.
    if (context->quiesce(context->sendComm) != flagcxSuccess) {
      *done = 0;
      return result;
    }
    *done = 1;
  }
  return result;
}

} // namespace

flagcxResult_t
flagcxP2pNetBackendInit(struct flagcxP2pNetBackendContext *context,
                        struct flagcxNetAdaptor *adaptor, void *sendComm,
                        std::mutex *progressMutex, int write,
                        struct flagcxP2pTransferBackend *backend,
                        flagcxResult_t (*quiesce)(void *sendComm)) {
  if (context == NULL || adaptor == NULL || sendComm == NULL ||
      progressMutex == NULL || backend == NULL || (write != 0 && write != 1) ||
      adaptor->test == NULL ||
      (write ? adaptor->iput == NULL : adaptor->iget == NULL))
    return flagcxInvalidArgument;
  context->adaptor = adaptor;
  context->sendComm = sendComm;
  context->progressMutex = progressMutex;
  context->write = write;
  context->quiesce = quiesce;
  backend->context = context;
  backend->post = post;
  backend->test = test;
  return flagcxSuccess;
}
