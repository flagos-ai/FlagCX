#include "flagcx_hetero.h"
#include "group.h"
#include "net.h"
#include "transport.h"
#include "type.h"
#include "ib_common.h"

flagcxResult_t flagcxHeteroSend(const void *sendbuff, size_t count,
                                flagcxDataType_t datatype, int peer,
                                flagcxHeteroComm_t comm, flagcxStream_t stream,
                                int groupIdx) {
  assert(groupIdx >= 0 && groupIdx < FLAGCX_MAX_SUBGROUPS);
  flagcxHeteroGroupStart();
  int channelId = 0;
  if (comm->channels[channelId].peers[peer]->send[0].connected == 0) {
    comm->connectSend[peer] |= (1UL << channelId);
    flagcxGroupCommPreconnect(comm);
  }
  struct flagcxTaskP2p *p2p;
  struct flagcxTasks *tasks = &comm->tasks;
  FLAGCXCHECK(flagcxCalloc(&p2p, 1));
  p2p->buff = (void *)sendbuff;
  p2p->bytes = count * getFlagcxDataTypeSize(datatype);
  p2p->chunk = 0;
  p2p->dtype = datatype;
  p2p->stream = stream;
  p2p->groupIdx = groupIdx;
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
                                int groupIdx) {
  assert(groupIdx >= 0 && groupIdx < FLAGCX_MAX_SUBGROUPS);
  flagcxHeteroGroupStart();
  int channelId = 0;
  if (comm->channels[channelId].peers[peer]->recv[0].connected == 0) {
    comm->connectRecv[peer] |= (1UL << channelId);
    flagcxGroupCommPreconnect(comm);
  }
  struct flagcxTaskP2p *p2p;
  struct flagcxTasks *tasks = &comm->tasks;
  FLAGCXCHECK(flagcxCalloc(&p2p, 1));
  p2p->buff = (void *)recvbuff;
  p2p->bytes = count * getFlagcxDataTypeSize(datatype);
  p2p->chunk = 0;
  p2p->dtype = datatype;
  p2p->stream = stream;
  p2p->groupIdx = groupIdx;
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
  // Check if netAdaptor->put is available
  if (comm->netAdaptor != NULL && comm->netAdaptor->put != NULL) {
    int channelId = 0;
    int connIndex = 0;
    // Get sendNetResources from connector
    struct flagcxConnector *conn =
        &comm->channels[channelId].peers[peer]->send[connIndex];
    struct sendNetResources *resources =
        (struct sendNetResources *)conn->proxyConn.connection->transportResources;
    void *sendComm = resources->netSendComm;
    int srcRank = comm->rank;
    int dstRank = peer;

    uint64_t srcOff = srcOffset;
    uint64_t dstOff = dstOffset;
    void **gHandles = (void **)&globalOneSideHandles;
    void *request = NULL;
    return comm->netAdaptor->put(sendComm, srcOff, dstOff, size, srcRank, dstRank,
                                 gHandles, &request);
    return flagcxSuccess;
  }
  return flagcxNotSupported;
}

flagcxResult_t flagcxHeteroPutSignal(flagcxHeteroComm_t comm, int peer,
                                     size_t dstOffset) {
  // Check if netAdaptor->putSignal is available
  if (comm->netAdaptor != NULL && comm->netAdaptor->putSignal != NULL) {
    int channelId = 0;
    int connIndex = 0;
    // Get sendNetResources from connector
    struct flagcxConnector *conn =
        &comm->channels[channelId].peers[peer]->send[connIndex];
    struct sendNetResources *resources =
        (struct sendNetResources *)conn->proxyConn.connection->transportResources;
    void *sendComm = resources->netSendComm;
    int srcRank = comm->rank;
    int dstRank = peer;

    uint64_t dstOff = dstOffset;
    void **gHandles = (void **)&globalOneSideHandles;
    void *request = NULL;
    return comm->netAdaptor->putSignal(sendComm, dstOff, srcRank, dstRank,
                                       gHandles, &request);
    return flagcxSuccess;
  }
  return flagcxNotSupported;
}