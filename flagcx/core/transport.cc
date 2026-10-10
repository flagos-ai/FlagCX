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

FLAGCX_PARAM(P2pDisable, "P2P_DISABLE", 0);

struct flagcxNetListenInfo {
  struct flagcxIbHandle handle;
  int netDev;
  uint64_t netGuid;
};

flagcxResult_t flagcxTransportPrepareProxyOp(struct flagcxHeteroComm *comm,
                                             struct flagcxProxyOp *op,
                                             void *buffer, size_t size,
                                             int peer, flagcxDataType_t dtype) {
  if (comm == NULL || op == NULL || op->connection == NULL ||
      op->connection->tcomm == NULL ||
      op->connection->tcomm->prepareProxyOp == NULL)
    return flagcxInvalidArgument;
  return op->connection->tcomm->prepareProxyOp(comm, op, buffer, size, peer,
                                               dtype);
}

static inline bool isSameNode(struct flagcxHeteroComm *comm, int peer) {
  // Self is always same node (self-copy uses P2P memcpy, not NET)
  if (peer == comm->rank)
    return true;
  // force use net transport for unirunner allreduce
  if (flagcxParamP2pDisable()) {
    return false;
  }
  if (comm->peerInfo == NULL) {
    // peerInfo not initialized - assume different nodes (use network transport)
    return false;
  }
  return comm->peerInfo[peer].hostHash == comm->peerInfo[comm->rank].hostHash;
}

static flagcxResult_t waitForProxyConnect(struct flagcxHeteroComm *comm,
                                          struct flagcxConnector *connector,
                                          void *response = NULL) {
  flagcxResult_t result;
  do {
    result = flagcxPollProxyResponse(comm, &connector->proxyConn, response,
                                     connector);
  } while (result == flagcxInProgress);
  if (result == flagcxSuccess && !connector->proxyConn.sameProcess &&
      connector->proxyConn.connection != NULL)
    __atomic_store_n(&connector->proxyConn.connection->state, connConnected,
                     __ATOMIC_RELEASE);
  else if (result != flagcxSuccess && !connector->proxyConn.sameProcess &&
           connector->proxyConn.connection != NULL)
    flagcxProxyRecordConnectionError(connector->proxyConn.connection, result);
  return result;
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
          FLAGCXCHECK(flagcxProxyConnect(comm, TRANSPORT_P2P, 0, comm->rank,
                                         &conn->proxyConn));
          struct flagcxP2pResources *resources;
          FLAGCXCHECK(flagcxCalloc(&resources, 1));
          conn->proxyConn.connection->transportResources = (void *)resources;
          if (peer != comm->rank) {
            struct flagcxP2pRequest req = {(size_t(flagcxP2pBufferSize)), 0};
            struct flagcxP2pConnectInfo connectInfo = {0};
            connectInfo.rank = comm->rank;
            connectInfo.read = 0;
            FLAGCXCHECK(flagcxProxyCallBlocking(
                comm, &conn->proxyConn, flagcxProxyMsgSetup, &req, sizeof(req),
                &connectInfo.p2pBuff, sizeof(connectInfo.p2pBuff)));
            // Use the buffer directly without offset
            char *recvBuffer = (char *)connectInfo.p2pBuff.directPtr;
            conn->conn.buffs[FLAGCX_PROTO_SIMPLE] = recvBuffer;
            FLAGCXCHECK(bootstrapSend(comm->bootstrap, peer, 2000 + c,
                                      &connectInfo, sizeof(connectInfo)));
          }
        } else {
          INFO(FLAGCX_NET,
               "NET Recv setup: rank %d <- peer %d channel %d (different node)",
               comm->rank, peer, c);
          FLAGCXCHECK(flagcxProxyConnect(comm, TRANSPORT_NET, 0, comm->rank,
                                         &conn->proxyConn));
          struct recvNetResources *resources;
          FLAGCXCHECK(flagcxCalloc(&resources, 1));
          conn->proxyConn.connection->transportResources = (void *)resources;
          resources->commPtr = comm;
          resources->netDev = comm->netDev;
          resources->netAdaptor = comm->netAdaptor;
          FLAGCXCHECK(deviceAdaptor->streamCreate(&resources->cpStream));
          for (int s = 0; s < flagcxNetChunks; s++) {
            FLAGCXCHECK(deviceAdaptor->eventCreate(&resources->cpEvents[s],
                                                   flagcxEventDisableTiming));
          }
          resources->buffSizes[0] = flagcxNetBufferSize;
          if (comm->netAdaptor == getNetAdaptor(SOCKET)) {
            resources->buffers[0] = (char *)malloc(resources->buffSizes[0]);
            if (!resources->buffers[0]) {
              return flagcxSystemError;
            }
          } else if (comm->netAdaptor == getNetAdaptor(RDMA)) {
            FLAGCXCHECK(
                deviceAdaptor->gdrMemAlloc((void **)&resources->buffers[0],
                                           resources->buffSizes[0], NULL));
          } else {
            flagcxNetProperties_t props;
            FLAGCXCHECK(
                comm->netAdaptor->getProperties(resources->netDev, &props));
            resources->ptrSupport = props.ptrSupport;
            if (resources->ptrSupport & FLAGCX_PTR_CUDA) {
              FLAGCXCHECK(
                  deviceAdaptor->gdrMemAlloc((void **)&resources->buffers[0],
                                             resources->buffSizes[0], NULL));
            } else {
              resources->buffers[0] = (char *)malloc(resources->buffSizes[0]);
              if (!resources->buffers[0])
                return flagcxSystemError;
            }
          }
          resources->useGdr = comm->netAdaptor != getNetAdaptor(SOCKET) &&
                              (comm->netAdaptor == getNetAdaptor(RDMA) ||
                               (resources->ptrSupport & FLAGCX_PTR_CUDA) != 0);
          struct flagcxNetListenInfo *listenInfo = NULL;
          FLAGCXCHECK(flagcxCalloc(&listenInfo, 1));
          listenInfo->netDev = resources->netDev;
          flagcxNetProperties_t listenProps = {};
          FLAGCXCHECK(
              comm->netAdaptor->getProperties(resources->netDev, &listenProps));
          listenInfo->netGuid = listenProps.guid;
          FLAGCXCHECK(comm->netAdaptor->listen(resources->netDev,
                                               (void *)&listenInfo->handle,
                                               &resources->netListenComm));
          FLAGCXCHECK(bootstrapSend(comm->bootstrap, peer, 1001 + c, listenInfo,
                                    sizeof(*listenInfo)));
          FLAGCXCHECK(flagcxProxyCallAsync(
              comm, &conn->proxyConn, flagcxProxyMsgConnect,
              &listenInfo->handle, sizeof(flagcxIbHandle), 0, conn));
          free(listenInfo);
        }
      }
      if (comm->connectSend[peer] & (1UL << c)) {
        struct flagcxConnector *conn =
            comm->channels[c].peers[peer]->send + connIndex;
        if (sameNode) {
          INFO(FLAGCX_P2P,
               "P2P Send setup: rank %d -> peer %d channel %d (same node)",
               comm->rank, peer, c);
          FLAGCXCHECK(flagcxProxyConnect(comm, TRANSPORT_P2P, 1, comm->rank,
                                         &conn->proxyConn));
          struct flagcxP2pResources *resources;
          FLAGCXCHECK(flagcxCalloc(&resources, 1));
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
          } else {
            // Self-copy: initialize proxyInfo stream/events for deviceMemcpy
            FLAGCXCHECK(
                deviceAdaptor->streamCreate(&resources->proxyInfo.stream));
            for (int s = 0; s < FLAGCX_P2P_MAX_STEPS; s++) {
              FLAGCXCHECK(deviceAdaptor->eventCreate(
                  &resources->proxyInfo.events[s], flagcxEventDisableTiming));
            }
          }
        } else {
          INFO(FLAGCX_NET,
               "NET Send setup: rank %d -> peer %d channel %d (different node)",
               comm->rank, peer, c);
          struct flagcxNetListenInfo *listenInfo = NULL;
          FLAGCXCHECK(flagcxCalloc(&listenInfo, 1));
          FLAGCXCHECK(bootstrapRecv(comm->bootstrap, peer, 1001 + c, listenInfo,
                                    sizeof(*listenInfo)));
          // The NET adaptor owns stage.comm during asynchronous connect.
          // The bootstrap handle may also be sent to another process for PXN,
          // so it must not contain a pointer to this communicator.
          int sendNetDev = comm->netDev;
          int proxyRank = comm->rank;
          int peerNetDev = listenInfo->netDev;
          if (flagcxPxnDisable(comm) == 0 && comm->topoServer != NULL &&
              comm->interServerTopo != NULL &&
              deviceAdaptor->ipcMemHandleCreate != NULL &&
              deviceAdaptor->ipcMemHandleGet != NULL &&
              deviceAdaptor->ipcMemHandleFree != NULL &&
              deviceAdaptor->ipcMemHandleOpen != NULL &&
              deviceAdaptor->ipcMemHandleClose != NULL) {
            struct flagcxTopoServer *remote = NULL;
            if (flagcxTopoGetServerFromRank(peer, comm->interServerTopo,
                                            comm->topoServer,
                                            &remote) == flagcxSuccess) {
              peerNetDev = -1;
              if (flagcxTopoNetDevFromGuid(remote, listenInfo->netGuid,
                                           &peerNetDev) == flagcxSuccess &&
                  flagcxTopoSelectNetRoute(comm->topoServer, remote,
                                           comm->interServerTopo, comm->rank,
                                           peer, peerNetDev, &sendNetDev,
                                           &proxyRank) == flagcxSuccess) {
                INFO(FLAGCX_NET,
                     "PXN route: %d -> %d channel %d recv NET/%d send "
                     "NET/%d proxy rank %d",
                     comm->rank, peer, c, listenInfo->netDev, sendNetDev,
                     proxyRank);
              } else {
                sendNetDev = comm->netDev;
                proxyRank = comm->rank;
              }
            }
          }
          uint64_t relayNetGuid = 0;
          if (proxyRank != comm->rank) {
            flagcxNetProperties_t sendProps = {};
            flagcxResult_t propsResult =
                comm->netAdaptor->getProperties(sendNetDev, &sendProps);
            if (propsResult != flagcxSuccess || sendProps.guid == 0) {
              free(listenInfo);
              return propsResult != flagcxSuccess ? propsResult
                                                  : flagcxNotSupported;
            }
            relayNetGuid = sendProps.guid;
          }
          FLAGCXCHECK(flagcxProxyConnect(comm, TRANSPORT_NET, 1, proxyRank,
                                         &conn->proxyConn));
          conn->proxyConn.netDev = sendNetDev;
          conn->proxyConn.peerNetDev = peerNetDev;
          if (proxyRank == comm->rank) {
            struct sendNetResources *resources;
            FLAGCXCHECK(flagcxCalloc(&resources, 1));
            conn->proxyConn.connection->transportResources = resources;
            resources->commPtr = comm;
            FLAGCXCHECK(flagcxNetInitSendResources(comm->netAdaptor, sendNetDev,
                                                   resources));
          } else {
            struct flagcxNetSendSetupRequest setup = {relayNetGuid};
            FLAGCXCHECK(flagcxProxyCallBlocking(comm, &conn->proxyConn,
                                                flagcxProxyMsgSetup, &setup,
                                                sizeof(setup), NULL, 0));
          }
          FLAGCXCHECK(flagcxProxyCallAsync(
              comm, &conn->proxyConn, flagcxProxyMsgConnect,
              &listenInfo->handle, sizeof(flagcxIbHandle),
              proxyRank == comm->rank ? 0 : sizeof(flagcxNetRelayBufferInfo),
              conn));
          free(listenInfo);
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
          FLAGCXCHECK(waitForProxyConnect(comm, conn));
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
          if (conn->proxyConn.sameProcess) {
            FLAGCXCHECK(waitForProxyConnect(comm, conn));
          } else {
            flagcxNetRelayBufferInfo relayBuffer = {};
            FLAGCXCHECK(waitForProxyConnect(comm, conn, &relayBuffer));
            if (relayBuffer.handleSize == 0 ||
                relayBuffer.handleSize > sizeof(relayBuffer.handleData) ||
                relayBuffer.capacity < size_t(flagcxNetChunkSize))
              return flagcxInternalError;
            flagcxP2pIpcDesc desc = {};
            desc.handleData = relayBuffer.handleData;
            desc.handleSize = relayBuffer.handleSize;
            desc.size = relayBuffer.capacity;
            void *mapped = NULL;
            FLAGCXCHECK(flagcxP2pImportShareableBuffer(
                comm, conn->proxyConn.tpRank, desc.size, &desc, &mapped));
            conn->proxyConn.connection->relayBufferImport = mapped;
            conn->proxyConn.connection->relayBufferCapacity = desc.size;
            INFO(FLAGCX_NET,
                 "PXN imported registered relay buffer: source %d relay %d "
                 "channel %d capacity %zu",
                 comm->rank, conn->proxyConn.tpRank, c, desc.size);
          }
        }
        comm->channels[c].peers[peer]->send[0].connected = 1;
        comm->connectSend[peer] ^= (1UL << c);
      }
    }
  }
  return flagcxSuccess;
}
