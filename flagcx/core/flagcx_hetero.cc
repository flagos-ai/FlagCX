#include "flagcx_hetero.h"
#include "adaptor.h"
#include "group.h"
#include "ib_common.h"
#include "net.h"
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
                               size_t srcOffset, size_t dstOffset,
                               size_t size) {
  // Check if netAdaptor->iput is available
  if (comm->netAdaptor != NULL && comm->netAdaptor->iput != NULL) {
    int channelId = 0;
    int connIndex = 0;
    // Get sendNetResources from connector
    struct flagcxConnector *conn =
        &comm->channels[channelId].peers[peer]->send[connIndex];
    // Check connection
    if (conn->connected == 0 ||
        conn->proxyConn.connection->transport != TRANSPORT_NET) {
      return flagcxNotSupported;
    }
    struct sendNetResources *resources =
        (struct sendNetResources *)
            conn->proxyConn.connection->transportResources;
    void *sendComm = resources->netSendComm;
    int srcRank = comm->rank;
    int dstRank = peer;

    uint64_t srcOff = srcOffset;
    uint64_t dstOff = dstOffset;
    void **gHandles = (void **)globalOneSideHandles;
    if (gHandles == NULL) {
      WARN("flagcxHeteroPut: globalOneSideHandles not initialized");
      return flagcxInternalError;
    }
    void *request = NULL;
    FLAGCXCHECK(comm->netAdaptor->iput(sendComm, srcOff, dstOff, size, srcRank,
                                       dstRank, gHandles, &request));
    // Poll completion to free the IB request
    if (request != NULL) {
      int done = 0;
      while (!done) {
        FLAGCXCHECK(comm->netAdaptor->test(request, &done, NULL));
      }
    }
    return flagcxSuccess;
  }
  return flagcxNotSupported;
}

flagcxResult_t flagcxHeteroGet(flagcxHeteroComm_t comm, int peer,
                               size_t srcOffset, size_t dstOffset,
                               size_t size) {
  if (comm->netAdaptor != NULL && comm->netAdaptor->iget != NULL) {
    int channelId = 0;
    int connIndex = 0;
    struct flagcxConnector *conn =
        &comm->channels[channelId].peers[peer]->send[connIndex];
    if (conn->connected == 0 ||
        conn->proxyConn.connection->transport != TRANSPORT_NET) {
      return flagcxNotSupported;
    }
    struct sendNetResources *resources =
        (struct sendNetResources *)
            conn->proxyConn.connection->transportResources;
    void *sendComm = resources->netSendComm;
    // srcRank is the remote peer (data source), dstRank is the local rank
    int srcRank = peer;
    int dstRank = comm->rank;

    void **gHandles = (void **)globalOneSideHandles;
    if (gHandles == NULL) {
      WARN("flagcxHeteroGet: globalOneSideHandles not initialized");
      return flagcxInternalError;
    }
    void *request = NULL;
    FLAGCXCHECK(comm->netAdaptor->iget(sendComm, (uint64_t)srcOffset,
                                       (uint64_t)dstOffset, size, srcRank,
                                       dstRank, gHandles, &request));
    if (request != NULL) {
      int done = 0;
      while (!done) {
        FLAGCXCHECK(comm->netAdaptor->test(request, &done, NULL));
      }
    }
    return flagcxSuccess;
  }
  return flagcxNotSupported;
}

flagcxResult_t flagcxHeteroPutSignal(flagcxHeteroComm_t comm, int peer,
                                     size_t srcOffset, size_t dstOffset,
                                     size_t size, size_t signalOffset) {
  // Check if netAdaptor->iputSignal is available
  if (comm->netAdaptor != NULL && comm->netAdaptor->iputSignal != NULL) {
    int channelId = 0;
    int connIndex = 0;
    // Get sendNetResources from connector
    struct flagcxConnector *conn =
        &comm->channels[channelId].peers[peer]->send[connIndex];
    // Check connection
    if (conn->connected == 0 ||
        conn->proxyConn.connection->transport != TRANSPORT_NET) {
      return flagcxNotSupported;
    }
    struct sendNetResources *resources =
        (struct sendNetResources *)
            conn->proxyConn.connection->transportResources;
    void *sendComm = resources->netSendComm;
    int srcRank = comm->rank;
    int dstRank = peer;

    void **dataHandles = (void **)globalOneSideHandles;
    void **signalHandles = (void **)globalOneSideSignalHandles;
    if (signalHandles == NULL) {
      WARN("flagcxHeteroPutSignal: globalOneSideSignalHandles not initialized");
      return flagcxInternalError;
    }
    if (size > 0 && dataHandles == NULL) {
      WARN("flagcxHeteroPutSignal: globalOneSideHandles not initialized for "
           "data transfer");
      return flagcxInternalError;
    }
    void *request = NULL;
    FLAGCXCHECK(comm->netAdaptor->iputSignal(
        sendComm, (uint64_t)srcOffset, (uint64_t)dstOffset, size, srcRank,
        dstRank, dataHandles, (uint64_t)signalOffset, signalHandles, &request));
    // Poll completion (single CQE for chained WRITE + ATOMIC)
    if (request != NULL) {
      int done = 0;
      while (!done) {
        FLAGCXCHECK(comm->netAdaptor->test(request, &done, NULL));
      }
    }
    return flagcxSuccess;
  }
  return flagcxNotSupported;
}

flagcxResult_t flagcxHeteroFlush(flagcxHeteroComm_t comm, void *gpuAddr,
                                 size_t size, void *gHandleInfo) {
  struct flagcxIbGlobalHandleInfo *info =
      (struct flagcxIbGlobalHandleInfo *)gHandleInfo;
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
  struct flagcxIbGlobalHandleInfo *info =
      (struct flagcxIbGlobalHandleInfo *)globalOneSideSignalHandles;
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