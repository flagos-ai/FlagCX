#include "nccl_wrapper.cc"
#include <cassert>
#include <vector>

struct flagcxComm {
  int device;
};
struct flagcxStream {
  void *native;
};
struct flagcxEvent {
  std::vector<size_t> work;
};
struct Operation {
  flagcxComm_t comm;
  void *stream;
  bool submitted, complete;
};
static std::vector<Operation> operations;
static int selectedDevice, depth, records, waits, finalized, destroyed;
static int eventFrees, streamFrees;
static flagcxResult_t submitResult = flagcxSuccess;
static flagcxResult_t groupResult = flagcxSuccess;
static flagcxResult_t asyncResult = flagcxSuccess;
static flagcxResult_t waitResult = flagcxSuccess;
static flagcxResult_t recordResult = flagcxSuccess;
static flagcxResult_t destroyResult = flagcxSuccess;
static flagcxResult_t releaseResult = flagcxSuccess;
static flagcxDeviceHandle device;

flagcxResult_t flagcxDeviceHandleInit(flagcxDeviceHandle_t *out) {
  *out = &device;
  return flagcxSuccess;
}
flagcxResult_t flagcxDeviceHandleFree(flagcxDeviceHandle_t) {
  return flagcxSuccess;
}
flagcxResult_t flagcxCommInitRank(flagcxComm_t *out, int, flagcxUniqueId_t,
                                  int rank) {
  *out = new flagcxComm{rank};
  return flagcxSuccess;
}
flagcxResult_t flagcxCommGetDeviceNumber(const flagcxComm_t comm, int *out) {
  *out = comm->device;
  return flagcxSuccess;
}
flagcxResult_t flagcxCommGetAsyncError(flagcxComm_t comm, flagcxResult_t *out) {
  *out = asyncResult;
  if (*out == flagcxSuccess)
    for (auto &op : operations)
      if (op.comm == comm)
        op.submitted = true;
  return flagcxSuccess;
}
static void checkCompleted(flagcxComm_t comm) {
  for (const auto &op : operations)
    if (op.comm == comm)
      assert(op.complete);
}
flagcxResult_t flagcxCommFinalize(flagcxComm_t comm) {
  checkCompleted(comm);
  ++finalized;
  return flagcxSuccess;
}
flagcxResult_t flagcxCommDestroy(flagcxComm_t comm) {
  checkCompleted(comm);
  if (destroyResult != flagcxSuccess)
    return destroyResult;
  ++destroyed;
  delete comm;
  return flagcxSuccess;
}
flagcxResult_t flagcxGroupStart(flagcxComm_t) {
  ++depth;
  return flagcxSuccess;
}
flagcxResult_t flagcxGroupEnd(flagcxComm_t) {
  if (--depth == 0 && groupResult == flagcxSuccess)
    for (auto &op : operations)
      op.submitted = true;
  return groupResult;
}
static flagcxResult_t submit(flagcxComm_t comm, flagcxStream_t stream) {
  assert(stream);
  operations.push_back({comm, stream->native,
                        depth == 0 && submitResult == flagcxSuccess, false});
  return submitResult;
}
flagcxResult_t flagcxAllReduce(const void *, void *, size_t, flagcxDataType_t,
                               flagcxRedOp_t, flagcxComm_t comm,
                               flagcxStream_t stream) {
  return submit(comm, stream);
}
flagcxResult_t flagcxBroadcast(const void *, void *, size_t, flagcxDataType_t,
                               int, flagcxComm_t comm, flagcxStream_t stream) {
  return submit(comm, stream);
}
flagcxResult_t flagcxReduce(const void *, void *, size_t, flagcxDataType_t,
                            flagcxRedOp_t, int, flagcxComm_t comm,
                            flagcxStream_t stream) {
  return submit(comm, stream);
}
flagcxResult_t flagcxAllGather(const void *, void *, size_t, flagcxDataType_t,
                               flagcxComm_t comm, flagcxStream_t stream) {
  return submit(comm, stream);
}
flagcxResult_t flagcxReduceScatter(const void *, void *, size_t,
                                   flagcxDataType_t, flagcxRedOp_t,
                                   flagcxComm_t comm, flagcxStream_t stream) {
  return submit(comm, stream);
}
flagcxResult_t flagcxSend(const void *, size_t, flagcxDataType_t, int,
                          flagcxComm_t comm, flagcxStream_t stream) {
  return submit(comm, stream);
}
flagcxResult_t flagcxRecv(void *, size_t, flagcxDataType_t, int,
                          flagcxComm_t comm, flagcxStream_t stream) {
  return submit(comm, stream);
}

static ncclComm_t create(int rank = 0) {
  ncclComm_t comm = nullptr;
  ncclUniqueId id = {};
  assert(ncclCommInitRank(&comm, 1, id, rank) == ncclSuccess);
  return comm;
}

int main() {
  device.setDevice = [](int d) {
    selectedDevice = d;
    return flagcxSuccess;
  };
  device.streamCopy = [](flagcxStream_t *out, void *native) {
    *out = new flagcxStream{native};
    return flagcxSuccess;
  };
  device.streamFree = [](flagcxStream_t stream) {
    ++streamFrees;
    delete stream;
    return flagcxSuccess;
  };
  device.eventCreate = [](flagcxEvent_t *out, flagcxEventType_t) {
    *out = new flagcxEvent;
    return flagcxSuccess;
  };
  device.eventRecord = [](flagcxEvent_t event, flagcxStream_t stream) {
    if (recordResult != flagcxSuccess)
      return recordResult;
    ++records;
    event->work.clear();
    for (size_t i = 0; i < operations.size(); ++i) {
      auto &op = operations[i];
      if (!op.complete && op.comm->device == selectedDevice &&
          op.stream == stream->native) {
        assert(op.submitted); // No event before outer/native submission.
        event->work.push_back(i);
      }
    }
    return flagcxSuccess;
  };
  device.eventSynchronize = [](flagcxEvent_t event) {
    ++waits;
    if (waitResult != flagcxSuccess)
      return waitResult;
    for (size_t i : event->work)
      operations[i].complete = true;
    return flagcxSuccess;
  };
  device.eventDestroy = [](flagcxEvent_t event) {
    if (releaseResult != flagcxSuccess)
      return releaseResult;
    ++eventFrees;
    delete event;
    return flagcxSuccess;
  };
  device.deviceSynchronize = [] {
    assert(false);
    return flagcxInternalError;
  };
  device.streamSynchronize = [](flagcxStream_t) {
    assert(false);
    return flagcxInternalError;
  };

  // Every supported submission retains a stable wrapper, and stays
  // asynchronous.
  auto comm = create();
  assert(ncclAllReduce(nullptr, nullptr, 1, ncclFloat32, ncclSum, comm,
                       nullptr) == ncclSuccess);
  assert(ncclBroadcast(nullptr, nullptr, 1, ncclFloat32, 0, comm, nullptr) ==
         ncclSuccess);
  assert(ncclReduce(nullptr, nullptr, 1, ncclFloat32, ncclSum, 0, comm,
                    nullptr) == ncclSuccess);
  assert(ncclAllGather(nullptr, nullptr, 1, ncclFloat32, comm, nullptr) ==
         ncclSuccess);
  assert(ncclReduceScatter(nullptr, nullptr, 1, ncclFloat32, ncclSum, comm,
                           nullptr) == ncclSuccess);
  assert(ncclSend(nullptr, 1, ncclFloat32, 0, comm, nullptr) == ncclSuccess);
  assert(ncclRecv(nullptr, 1, ncclFloat32, 0, comm, nullptr) == ncclSuccess);
  assert(records == 7 && waits == 0 && streamFrees == 0);
  assert(comm->completion.streams->wrapper->native == nullptr);
  waitResult = flagcxInProgress;
  assert(ncclCommFinalize(comm) == ncclInProgress && finalized == 0);
  assert(ncclAllReduce(nullptr, nullptr, 1, ncclFloat32, ncclSum, comm,
                       nullptr) == ncclInvalidUsage);
  waitResult = flagcxSuccess;
  assert(ncclCommFinalize(comm) == ncclSuccess && finalized == 1);
  auto completedWaits = waits;
  destroyResult = flagcxRemoteError;
  assert(ncclCommDestroy(comm) == ncclRemoteError);
  assert(waits == completedWaits && eventFrees == 0);
  destroyResult = flagcxSuccess;
  releaseResult = flagcxRemoteError;
  assert(ncclCommDestroy(comm) == ncclRemoteError && comm->comm == nullptr);
  releaseResult = flagcxSuccess;
  assert(ncclCommDestroy(comm) == ncclSuccess);
  assert(eventFrees == 1 && streamFrees == 1);

  // Submit all Comms in nested groups before recording any completion event.
  auto first = create(), second = create(1);
  auto before = records;
  assert(ncclGroupStart() == ncclSuccess);
  assert(ncclGroupStart() == ncclSuccess);
  assert(ncclAllReduce(nullptr, nullptr, 1, ncclFloat32, ncclSum, first,
                       (void *)1) == ncclSuccess);
  assert(ncclAllReduce(nullptr, nullptr, 1, ncclFloat32, ncclSum, second,
                       (void *)2) == ncclSuccess);
  assert(records == before);
  assert(ncclGroupEnd() == ncclSuccess && records == before);
  assert(ncclGroupEnd() == ncclSuccess && records == before + 2);
  assert(ncclCommDestroy(first) == ncclSuccess);
  assert(ncclCommDestroy(second) == ncclSuccess);

  first = create();
  second = create(1);
  before = records;
  assert(ncclGroupStart() == ncclSuccess);
  assert(ncclAllReduce(nullptr, nullptr, 1, ncclFloat32, ncclSum, first,
                       (void *)1) == ncclSuccess);
  assert(ncclAllReduce(nullptr, nullptr, 1, ncclFloat32, ncclSum, second,
                       (void *)2) == ncclSuccess);
  groupResult = flagcxInProgress;
  assert(ncclGroupEnd() == ncclInProgress && records == before);
  asyncResult = flagcxInProgress;
  assert(ncclCommFinalize(first) == ncclInProgress && records == before);
  asyncResult = flagcxSuccess;
  ncclResult_t async;
  assert(ncclCommGetAsyncError(second, &async) == ncclSuccess &&
         async == ncclSuccess);
  assert(records == before + 1);
  assert(ncclCommDestroy(first) == ncclSuccess);
  assert(ncclCommDestroy(second) == ncclSuccess && records == before + 2);
  groupResult = flagcxSuccess;

  // Nonblocking native submission must finish before recording its event.
  comm = create();
  submitResult = flagcxInProgress;
  before = records;
  assert(ncclAllReduce(nullptr, nullptr, 1, ncclFloat32, ncclSum, comm,
                       nullptr) == ncclInProgress);
  assert(records == before);
  asyncResult = flagcxInProgress;
  assert(ncclCommFinalize(comm) == ncclInProgress);
  assert(records == before);
  asyncResult = flagcxSuccess;
  assert(ncclCommDestroy(comm) == ncclSuccess && records == before + 1);
  submitResult = flagcxSuccess;

  // Record failure keeps its owner and retries before core cleanup.
  comm = create();
  recordResult = flagcxRemoteError;
  assert(ncclAllReduce(nullptr, nullptr, 1, ncclFloat32, ncclSum, comm,
                       nullptr) == ncclRemoteError);
  auto beforeDestroy = destroyed;
  assert(ncclCommDestroy(comm) == ncclRemoteError &&
         destroyed == beforeDestroy);
  recordResult = flagcxSuccess;
  assert(ncclCommDestroy(comm) == ncclSuccess);
  assert(eventFrees == streamFrees);
}
