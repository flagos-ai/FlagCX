#include "adaptor.h"
#include "bootstrap.h"
#include "comm.h"
#include "info.h"
#include "net.h"
#include "p2p.h"
#include "proxy.h"
#include "shmutils.h"
#include "topo.h"
#define ENABLE_TIMER 0
#include "timer.h"

FLAGCX_PARAM(RunUniRunnerAllReduce, "RUN_UNIRUNNER_ALLREDUCE", 0);

static inline bool isSameNode(struct flagcxHeteroComm *comm, int peer) {
  // force use network transport for unirunner allreduce
  int runUniRunnerAllReduce = flagcxParamRunUniRunnerAllReduce();
  if (runUniRunnerAllReduce) {
    return false;
  }
  if (comm->peerInfo == NULL) {
    // peerInfo not initialized - assume different nodes (use network transport)
    return false;
  }
  return comm->peerInfo[peer].hostHash == comm->peerInfo[comm->rank].hostHash;
}

flagcxResult_t flagcxTransportP2pSetup(struct flagcxHeteroComm *comm,
                                       struct flagcxTopoGraph *graph,
                                       int connIndex,
                                       int *highestTransportType /*=NULL*/) {
  for (int peer = 0; peer < comm->nRanks; peer++) {
    bool sameNode = isSameNode(comm, peer);
    for (int c = 0; c < MAXCHANNELS; c++) {
      if (comm->connectRecv[peer] & (1UL << c)) {
        struct flagcxConnector *conn =
            comm->channels[c].peers[peer]->recv + connIndex;
        if (sameNode) {
          INFO(FLAGCX_P2P,
               "P2P Recv setup: rank %d <- peer %d channel %d (same node)",
               comm->rank, peer, c);
          FLAGCXCHECK(flagcxCalloc(&conn->proxyConn.connection, 1));
          struct flagcxP2pResources *resources;
          FLAGCXCHECK(flagcxCalloc(&resources, 1));
          conn->proxyConn.connection->transport = TRANSPORT_P2P;
          conn->proxyConn.connection->send = 0;
          conn->proxyConn.connection->transportResources = (void *)resources;
          if (peer != comm->rank) {
            struct flagcxP2pRequest req = {(size_t(flagcxP2pBufferSize)), 0};
            struct flagcxP2pConnectInfo connectInfo = {0};
            connectInfo.rank = comm->rank;
            connectInfo.read = 0;
            FLAGCXCHECK(flagcxProxyCallBlocking(
                comm, &conn->proxyConn, flagcxProxyMsgSetup, &req, sizeof(req),
                &connectInfo.p2pBuff, sizeof(connectInfo.p2pBuff)));
            // Use the buffer directly without offset， it's equal to nccl
            // p2pMap function
            char *recvBuffer = (char *)connectInfo.p2pBuff.directPtr;
            conn->conn.buffs[FLAGCX_PROTO_SIMPLE] = recvBuffer;
            FLAGCXCHECK(bootstrapSend(comm->bootstrap, peer, 2000 + c,
                                      &connectInfo, sizeof(connectInfo)));
          }
        } else {
          INFO(FLAGCX_NET,
               "NET Recv setup: rank %d <- peer %d channel %d (different node)",
               comm->rank, peer, c);
          FLAGCXCHECK(flagcxCalloc(&conn->proxyConn.connection, 1));
          struct recvNetResources *resources;
          FLAGCXCHECK(flagcxCalloc(&resources, 1));
          conn->proxyConn.connection->transport = TRANSPORT_NET;
          conn->proxyConn.connection->send = 0;
          conn->proxyConn.connection->transportResources = (void *)resources;
          resources->netDev = comm->netDev;
          resources->netAdaptor = comm->netAdaptor;
          deviceAdaptor->streamCreate(&resources->cpStream);
          for (int s = 0; s < flagcxNetChunks; s++) {
            deviceAdaptor->eventCreate(&resources->cpEvents[s],
                                       flagcxEventDisableTiming);
          }
          resources->buffSizes[0] = flagcxNetBufferSize;
          if (comm->netAdaptor == getUnifiedNetAdaptor(SOCKET)) {
            resources->buffers[0] = (char *)malloc(resources->buffSizes[0]);
            if (!resources->buffers[0]) {
              return flagcxSystemError;
            }
          } else if (comm->netAdaptor == getUnifiedNetAdaptor(IBRC)) {
            deviceAdaptor->gdrMemAlloc((void **)&resources->buffers[0],
                                       resources->buffSizes[0], NULL);
          }
          struct flagcxIbHandle *handle = NULL;
          FLAGCXCHECK(flagcxCalloc(&handle, 1));
          comm->netAdaptor->listen(resources->netDev, (void *)handle,
                                   &resources->netListenComm);
          bootstrapSend(comm->bootstrap, peer, 1001 + c, (void *)handle,
                        sizeof(flagcxIbHandle));
          FLAGCXCHECK(flagcxProxyCallAsync(
              comm, &conn->proxyConn, flagcxProxyMsgConnect, (void *)handle,
              sizeof(flagcxIbHandle), 0, conn));
          free(handle);
        }
      }
      if (comm->connectSend[peer] & (1UL << c)) {
        struct flagcxConnector *conn =
            comm->channels[c].peers[peer]->send + connIndex;
        if (sameNode) {
          INFO(FLAGCX_P2P,
               "P2P Send setup: rank %d -> peer %d channel %d (same node)",
               comm->rank, peer, c);
          FLAGCXCHECK(flagcxCalloc(&conn->proxyConn.connection, 1));
          struct flagcxP2pResources *resources;
          FLAGCXCHECK(flagcxCalloc(&resources, 1));
          conn->proxyConn.connection->transport = TRANSPORT_P2P;
          conn->proxyConn.connection->send = 1;
          conn->proxyConn.connection->transportResources = (void *)resources;
          if (peer != comm->rank) {
            struct flagcxP2pConnectInfo connectInfo = {0};
            FLAGCXCHECK(flagcxProxyCallBlocking(
                comm, &conn->proxyConn, flagcxProxyMsgSetup, NULL, 0,
                &resources->proxyInfo, sizeof(struct flagcxP2pShmProxyInfo)));
            memcpy(&connectInfo.desc, &resources->proxyInfo.desc,
                   sizeof(flagcxShmIpcDesc_t));
            INFO(FLAGCX_P2P,
                 "Send: Sending shmDesc to peer %d, shmSuffix=%s shmSize=%zu",
                 peer, connectInfo.desc.shmSuffix, connectInfo.desc.shmSize);
            FLAGCXCHECK(bootstrapSend(comm->bootstrap, peer, 3000 + c,
                                      &connectInfo.desc,
                                      sizeof(flagcxShmIpcDesc_t)));
          }
        } else {
          INFO(FLAGCX_NET,
               "NET Send setup: rank %d -> peer %d channel %d (different node)",
               comm->rank, peer, c);
          FLAGCXCHECK(flagcxCalloc(&conn->proxyConn.connection, 1));
          struct sendNetResources *resources;
          FLAGCXCHECK(flagcxCalloc(&resources, 1));
          conn->proxyConn.connection->send = 1;
          conn->proxyConn.connection->transport = TRANSPORT_NET;
          conn->proxyConn.connection->transportResources = (void *)resources;
          resources->netDev = comm->netDev;
          resources->netAdaptor = comm->netAdaptor;
          deviceAdaptor->streamCreate(&resources->cpStream);
          for (int s = 0; s < flagcxNetChunks; s++) {
            deviceAdaptor->eventCreate(&resources->cpEvents[s],
                                       flagcxEventDisableTiming);
          }
          resources->buffSizes[0] = flagcxNetBufferSize;
          if (comm->netAdaptor == getUnifiedNetAdaptor(SOCKET)) {
            resources->buffers[0] = (char *)malloc(resources->buffSizes[0]);
            if (!resources->buffers[0]) {
              return flagcxSystemError;
            }
          } else if (comm->netAdaptor == getUnifiedNetAdaptor(IBRC)) {
            deviceAdaptor->gdrMemAlloc((void **)&resources->buffers[0],
                                       resources->buffSizes[0], NULL);
          }
          struct flagcxIbHandle *handle = NULL;
          FLAGCXCHECK(flagcxCalloc(&handle, 1));
          bootstrapRecv(comm->bootstrap, peer, 1001 + c, (void *)handle,
                        sizeof(flagcxIbHandle));
          handle->stage.comm = comm;
          FLAGCXCHECK(flagcxProxyCallAsync(
              comm, &conn->proxyConn, flagcxProxyMsgConnect, (void *)handle,
              sizeof(flagcxIbHandle), 0, conn));
          free(handle);
        }
      }
    }
  }

  for (int peer = 0; peer < comm->nRanks; peer++) {
    bool sameNode = isSameNode(comm, peer);
    for (int c = 0; c < MAXCHANNELS; c++) {
      if (comm->connectRecv[peer] & (1UL << c)) {
        struct flagcxConnector *conn =
            comm->channels[c].peers[peer]->recv + connIndex;
        if (sameNode) {
          INFO(FLAGCX_P2P,
               "P2P Recv connect: rank %d <- peer %d channel %d (same node)",
               comm->rank, peer, c);
          struct flagcxP2pResources *resources =
              (struct flagcxP2pResources *)
                  conn->proxyConn.connection->transportResources;
          if (peer != comm->rank) {
            flagcxShmIpcDesc_t shmDesc = {0};
            FLAGCXCHECK(bootstrapRecv(comm->bootstrap, peer, 3000 + c, &shmDesc,
                                      sizeof(flagcxShmIpcDesc_t)));
            FLAGCXCHECK(flagcxShmImportShareableBuffer(
                &shmDesc, (void **)&resources->shm, NULL, &resources->desc));
            resources->proxyInfo.shm = resources->shm;
            memcpy(&resources->proxyInfo.desc, &resources->desc,
                   sizeof(flagcxShmIpcDesc_t));
            // Set recvFifo in proxyInfo so proxy can copy data to it
            resources->proxyInfo.recvFifo =
                conn->conn.buffs[FLAGCX_PROTO_SIMPLE];
          }
          FLAGCXCHECK(flagcxProxyCallBlocking(
              comm, &conn->proxyConn, flagcxProxyMsgConnect, NULL, 0, NULL, 0));
        } else {
          INFO(FLAGCX_NET,
               "NET Recv connect: rank %d <- peer %d channel %d (different "
               "node)",
               comm->rank, peer, c);
          while (flagcxPollProxyResponse(comm, NULL, NULL, conn) ==
                 flagcxInProgress)
            ;
        }
        comm->channels[c].peers[peer]->recv[0].connected = 1;
        comm->connectRecv[peer] ^= (1UL << c);
      }
      if (comm->connectSend[peer] & (1UL << c)) {
        struct flagcxConnector *conn =
            comm->channels[c].peers[peer]->send + connIndex;
        if (sameNode) {
          INFO(FLAGCX_P2P,
               "P2P Send connect: rank %d -> peer %d channel %d (same node)",
               comm->rank, peer, c);
          struct flagcxP2pResources *resources =
              (struct flagcxP2pResources *)
                  conn->proxyConn.connection->transportResources;
          char *remoteBuffer = NULL;
          if (peer != comm->rank) {
            struct flagcxP2pConnectInfo connectInfo = {0};
            FLAGCXCHECK(bootstrapRecv(comm->bootstrap, peer, 2000 + c,
                                      &connectInfo, sizeof(connectInfo)));
            FLAGCXCHECK(flagcxP2pImportShareableBuffer(
                comm, peer, connectInfo.p2pBuff.size,
                &connectInfo.p2pBuff.ipcDesc, (void **)&remoteBuffer));
            if (remoteBuffer == NULL) {
              WARN("P2P Send: remoteBuffer is NULL after import for peer %d "
                   "channel %d",
                   peer, c);
              return flagcxInternalError;
            }
            conn->conn.buffs[FLAGCX_PROTO_SIMPLE] = remoteBuffer;
            resources->proxyInfo.recvFifo = remoteBuffer;
          }
          char *recvFifo = remoteBuffer;
          FLAGCXCHECK(flagcxProxyCallBlocking(comm, &conn->proxyConn,
                                              flagcxProxyMsgConnect, &recvFifo,
                                              sizeof(recvFifo), NULL, 0));
        } else {
          INFO(FLAGCX_NET,
               "NET Send connect: rank %d -> peer %d channel %d (different "
               "node)",
               comm->rank, peer, c);
          while (flagcxPollProxyResponse(comm, NULL, NULL, conn) ==
                 flagcxInProgress)
            ;
        }
        comm->channels[c].peers[peer]->send[0].connected = 1;
        comm->connectSend[peer] ^= (1UL << c);
      }
    }
  }
  return flagcxSuccess;
}