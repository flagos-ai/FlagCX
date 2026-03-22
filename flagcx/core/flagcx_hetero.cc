#include "flagcx_hetero.h"
#include "adaptor.h"
#include "group.h"
#include "net.h"
#include "onesided.h"
#include "transport.h"
#include "type.h"

#include <climits>
#include <sched.h>

flagcxResult_t flagcxHeteroSend(const void *sendbuff, size_t count,
                                flagcxDataType_t datatype, int peer,
                                flagcxHeteroComm_t comm, flagcxStream_t stream,
                                int opId, int step) {
  flagcxHeteroGroupStart();
  int channelId = 0;
  if (comm->channels[channelId].peers[peer]->send[0].connected == 0 &&
      comm->channels[channelId].peers[peer]->send[0].registered == 0) {
    comm->connectSend[peer] |= (1UL << channelId);
    flagcxGroupCommPreconnect(comm);
    comm->channels[channelId].peers[peer]->send[0].registered = 1;
  }
  struct flagcxTaskP2p *p2p;
  struct flagcxTasks *tasks = &comm->tasks;
  FLAGCXCHECK(flagcxCalloc(&p2p, 1));
  p2p->buff = (void *)sendbuff;
  p2p->bytes = count * getFlagcxDataTypeSize(datatype);
  p2p->chunk = 0;
  p2p->dtype = datatype;
  p2p->stream = stream;
  p2p->opId = opId;
  p2p->step = step;
  if (flagcxIntruQueueEmpty(&tasks->peers[peer].sendQueue))
    tasks->p2pOrder[tasks->p2pOrderSteps++] = peer;
  flagcxIntruQueueEnqueue(&tasks->peers[peer].sendQueue, p2p);

  flagcxGroupCommJoin(comm);
  flagcxHeteroGroupEnd();
  return flagcxSuccess;
}

flagcxResult_t flagcxHeteroRecv(void *recvbuff, size_t count,
                                flagcxDataType_t datatype, int peer,
                                flagcxHeteroComm_t comm, flagcxStream_t stream,
                                int opId, int step) {
  flagcxHeteroGroupStart();
  int channelId = 0;
  if (comm->channels[channelId].peers[peer]->recv[0].connected == 0 &&
      comm->channels[channelId].peers[peer]->recv[0].registered == 0) {
    comm->connectRecv[peer] |= (1UL << channelId);
    flagcxGroupCommPreconnect(comm);
    comm->channels[channelId].peers[peer]->recv[0].registered = 1;
  }
  struct flagcxTaskP2p *p2p;
  struct flagcxTasks *tasks = &comm->tasks;
  FLAGCXCHECK(flagcxCalloc(&p2p, 1));
  p2p->buff = (void *)recvbuff;
  p2p->bytes = count * getFlagcxDataTypeSize(datatype);
  p2p->chunk = 0;
  p2p->dtype = datatype;
  p2p->stream = stream;
  p2p->opId = opId;
  p2p->step = step;
  if (flagcxIntruQueueEmpty(&tasks->peers[peer].recvQueue))
    tasks->p2pOrder[tasks->p2pOrderSteps++] = peer;
  flagcxIntruQueueEnqueue(&tasks->peers[peer].recvQueue, p2p);

  flagcxGroupCommJoin(comm);
  flagcxHeteroGroupEnd();
  return flagcxSuccess;
}

flagcxResult_t flagcxHeteroPut(flagcxHeteroComm_t comm, int peer,
                               size_t srcOffset, size_t dstOffset, size_t size,
                               int srcMrIdx, int dstMrIdx) {
  // Check if netAdaptor->iput is available
  if (comm->netAdaptor == NULL || comm->netAdaptor->iput == NULL)
    return flagcxNotSupported;

  // Get sendComm from full-mesh connections (handle table slot 0 owns them)
  if (globalOneSideHandleCount == 0 ||
      globalOneSideHandleTable[0]->fullSendComms == NULL) {
    WARN("flagcxHeteroPut: no full-mesh connections");
    return flagcxInternalError;
  }
  void *sendComm = globalOneSideHandleTable[0]->fullSendComms[peer];
  if (sendComm == NULL) {
    WARN("flagcxHeteroPut: no sendComm for peer %d", peer);
    return flagcxInternalError;
  }

  // Get per-window MR handles from handle table
  if (srcMrIdx < 0 || srcMrIdx >= globalOneSideHandleCount || dstMrIdx < 0 ||
      dstMrIdx >= globalOneSideHandleCount) {
    WARN("flagcxHeteroPut: invalid MR index src=%d dst=%d (count=%d)", srcMrIdx,
         dstMrIdx, globalOneSideHandleCount);
    return flagcxInternalError;
  }
  void **srcHandles = (void **)globalOneSideHandleTable[srcMrIdx];
  void **dstHandles = (void **)globalOneSideHandleTable[dstMrIdx];

  int srcRank = comm->rank;
  int dstRank = peer;
  void *request = NULL;
  FLAGCXCHECK(comm->netAdaptor->iput(
      sendComm, (uint64_t)srcOffset, (uint64_t)dstOffset, size, srcRank,
      dstRank, srcHandles, dstHandles, &request));
  // Poll completion to free the IB request
  if (request != NULL) {
    int done = 0;
    while (!done) {
      FLAGCXCHECK(comm->netAdaptor->test(request, &done, NULL));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxHeteroPutSignal(flagcxHeteroComm_t comm, int peer,
                                     size_t srcOffset, size_t dstOffset,
                                     size_t size, size_t signalOffset,
                                     int srcMrIdx, int dstMrIdx,
                                     uint64_t signalValue) {
  // Check if netAdaptor->iputSignal is available
  if (comm->netAdaptor == NULL || comm->netAdaptor->iputSignal == NULL)
    return flagcxNotSupported;

  // Get sendComm from full-mesh connections
  if (globalOneSideHandleCount == 0 ||
      globalOneSideHandleTable[0]->fullSendComms == NULL) {
    WARN("flagcxHeteroPutSignal: no full-mesh connections");
    return flagcxInternalError;
  }
  void *sendComm = globalOneSideHandleTable[0]->fullSendComms[peer];
  if (sendComm == NULL) {
    WARN("flagcxHeteroPutSignal: no sendComm for peer %d", peer);
    return flagcxInternalError;
  }

  int srcRank = comm->rank;
  int dstRank = peer;

  // Data handles from per-window MR table
  void **srcHandles = NULL;
  void **dstHandles = NULL;
  if (size > 0) {
    if (srcMrIdx < 0 || srcMrIdx >= globalOneSideHandleCount || dstMrIdx < 0 ||
        dstMrIdx >= globalOneSideHandleCount) {
      WARN("flagcxHeteroPutSignal: invalid MR index src=%d dst=%d", srcMrIdx,
           dstMrIdx);
      return flagcxInternalError;
    }
    srcHandles = (void **)globalOneSideHandleTable[srcMrIdx];
    dstHandles = (void **)globalOneSideHandleTable[dstMrIdx];
  }
  void **signalHandles = (void **)globalOneSideSignalHandles;
  if (signalHandles == NULL) {
    WARN("flagcxHeteroPutSignal: globalOneSideSignalHandles not initialized");
    return flagcxInternalError;
  }
  void *request = NULL;
  FLAGCXCHECK(comm->netAdaptor->iputSignal(
      sendComm, (uint64_t)srcOffset, (uint64_t)dstOffset, size, srcRank,
      dstRank, srcHandles, dstHandles, (uint64_t)signalOffset, signalHandles,
      signalValue, &request));
  // Poll completion (single CQE for chained WRITE + ATOMIC)
  if (request != NULL) {
    int done = 0;
    while (!done) {
      FLAGCXCHECK(comm->netAdaptor->test(request, &done, NULL));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxHeteroFlush(flagcxHeteroComm_t comm, void *gpuAddr,
                                 size_t size, void *gHandleInfo) {
  struct flagcxOneSideHandleInfo *info =
      (struct flagcxOneSideHandleInfo *)gHandleInfo;
  if (info == NULL || info->localRecvComm == NULL ||
      info->localMrHandle == NULL)
    return flagcxNotSupported;
  if (comm->netAdaptor == NULL || comm->netAdaptor->iflush == NULL)
    return flagcxNotSupported;

  if (size > (size_t)INT_MAX) {
    WARN("flagcxHeteroFlush: size %zu exceeds int limit", size);
    return flagcxInternalError;
  }
  void *data_arr[1] = {gpuAddr};
  int sizes_arr[1] = {(int)size};
  void *mh_arr[1] = {info->localMrHandle};
  void *request = NULL;
  FLAGCXCHECK(comm->netAdaptor->iflush(info->localRecvComm, 1, data_arr,
                                       sizes_arr, mh_arr, &request));
  if (request != NULL) {
    int done = 0;
    while (!done) {
      FLAGCXCHECK(comm->netAdaptor->test(request, &done, NULL));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxHeteroWaitSignal(flagcxHeteroComm_t comm, int peer,
                                      size_t signalOffset, uint64_t expected,
                                      flagcxStream_t stream) {
  (void)peer;
  struct flagcxOneSideHandleInfo *info =
      (struct flagcxOneSideHandleInfo *)globalOneSideSignalHandles;
  if (info == NULL || info->baseVas == NULL)
    return flagcxNotSupported;

  int myRank = comm->rank;
  void *signalAddr = (void *)(info->baseVas[myRank] + signalOffset);

  // Device-side wait (streamWaitValue64) for GPU signal buffer.
  // RMA signal buffers are GPU memory (flagcxMemAlloc) — host-side volatile
  // polling would segfault. Non-CUDA platforms return flagcxNotSupported.
  // No flush needed: FORCE_SO on signal MR guarantees PCIe ordering.
  if (stream == NULL)
    return flagcxInternalError;

  return deviceAdaptor->streamWaitValue64(stream, signalAddr, expected, 0);
}

flagcxResult_t flagcxHeteroPutValue(flagcxHeteroComm_t comm, int peer,
                                    uint64_t value, size_t dstOffset,
                                    int dstMrIdx) {
  if (comm->netAdaptor == NULL || comm->netAdaptor->iput == NULL)
    return flagcxNotSupported;

  // 1. Validate staging handles
  struct flagcxOneSideHandleInfo *stagingH = globalOneSideStagingHandles;
  if (stagingH == NULL || stagingH->baseVas == NULL) {
    WARN("flagcxHeteroPutValue: staging handles not initialized");
    return flagcxInternalError;
  }

  // 2. Write value to local staging buffer
  int myRank = comm->rank;
  *(volatile uint64_t *)(stagingH->baseVas[myRank]) = value;

  // 3. Get sendComm from full-mesh connections (data handle[0] owns them)
  if (globalOneSideHandleCount == 0 ||
      globalOneSideHandleTable[0]->fullSendComms == NULL) {
    WARN("flagcxHeteroPutValue: no full-mesh connections");
    return flagcxInternalError;
  }
  void *sendComm = globalOneSideHandleTable[0]->fullSendComms[peer];
  if (sendComm == NULL) {
    WARN("flagcxHeteroPutValue: no sendComm for peer %d", peer);
    return flagcxInternalError;
  }

  // 4. Validate dst MR index
  if (dstMrIdx < 0 || dstMrIdx >= globalOneSideHandleCount) {
    WARN("flagcxHeteroPutValue: invalid dstMrIdx=%d (count=%d)", dstMrIdx,
         globalOneSideHandleCount);
    return flagcxInternalError;
  }
  void **srcHandles = (void **)stagingH;
  void **dstHandles = (void **)globalOneSideHandleTable[dstMrIdx];

  // 5. iput: srcOffset=0 (staging buffer start), size=8 bytes
  int dstRank = peer;
  void *request = NULL;
  FLAGCXCHECK(comm->netAdaptor->iput(sendComm, 0, (uint64_t)dstOffset,
                                     sizeof(uint64_t), myRank, dstRank,
                                     srcHandles, dstHandles, &request));

  // 6. Poll completion
  if (request != NULL) {
    int done = 0;
    while (!done) {
      FLAGCXCHECK(comm->netAdaptor->test(request, &done, NULL));
    }
  }
  return flagcxSuccess;
}
