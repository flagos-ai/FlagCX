#include "adaptor.h"
#include "bootstrap.h"
#include "comm.h"
#include "info.h"
#include "net.h"
#include "p2p.h"
#include "proxy.h"
#include "shmutils.h"
#include "topo.h"
#include <string.h>
#define ENABLE_TIMER 0
#include "timer.h"

FLAGCX_PARAM(P2pDisable, "P2P_DISABLE", 0);

struct flagcxNetListenInfo {
  struct flagcxIbHandle handle;
  int netDev;
  uint64_t netGuid;
};

flagcxResult_t
flagcxNetFinalizePxnSendRoute(struct flagcxNetAdaptor *netAdaptor,
                              int sourceRank, int sourceNetDev, int *sendNetDev,
                              int *proxyRank, uint64_t *relayNetGuid) {
  if (netAdaptor == NULL || sendNetDev == NULL || proxyRank == NULL ||
      relayNetGuid == NULL)
    return flagcxInvalidArgument;
  *relayNetGuid = 0;
  if (*proxyRank == sourceRank)
    return flagcxSuccess;

  flagcxNetProperties_t sendProps = {};
  if (netAdaptor->getProperties == NULL)
    return flagcxNotSupported;
  FLAGCXCHECK(netAdaptor->getProperties(*sendNetDev, &sendProps));
  if ((sendProps.ptrSupport & FLAGCX_PTR_CUDA) == 0 ||
      netAdaptor->regMr == NULL) {
    // The relay exports an IPC allocation and registers it with regMr, even
    // when direct NET would use regMrDmaBuf for this NIC.
    *sendNetDev = sourceNetDev;
    *proxyRank = sourceRank;
    return flagcxSuccess;
  }
  if (sendProps.guid == 0)
    return flagcxNotSupported;
  *relayNetGuid = sendProps.guid;
  return flagcxSuccess;
}

static flagcxResult_t
flagcxGetRelayImport(struct flagcxHeteroComm *comm, int proxyRank,
                     const flagcxNetRelayBufferInfo *buffer,
                     flagcxNetRelayImport **out) {
  if (comm == NULL || buffer == NULL || out == NULL || buffer->poolId == 0)
    return flagcxInvalidArgument;
  auto *state = comm->proxyState;
  pthread_mutex_lock(&state->relayImportMutex);
  for (auto *entry = state->relayImports; entry != NULL; entry = entry->next) {
    if (entry->proxyRank != proxyRank || entry->poolId != buffer->poolId)
      continue;
    const bool matches = !entry->closed &&
                         entry->capacity == buffer->capacity &&
                         entry->handleSize == buffer->handleSize &&
                         memcmp(&entry->handleData, &buffer->handleData,
                                sizeof(buffer->handleData)) == 0;
    pthread_mutex_unlock(&state->relayImportMutex);
    if (!matches)
      return flagcxInternalError;
    *out = entry;
    return flagcxSuccess;
  }

  auto *entry = static_cast<flagcxNetRelayImport *>(
      calloc(1, sizeof(flagcxNetRelayImport)));
  if (entry == NULL) {
    pthread_mutex_unlock(&state->relayImportMutex);
    return flagcxSystemError;
  }
  flagcxP2pIpcDesc desc = {};
  desc.handleData = buffer->handleData;
  desc.handleSize = buffer->handleSize;
  desc.size = buffer->capacity;
  flagcxResult_t result = flagcxP2pImportShareableBuffer(
      comm, proxyRank, desc.size, &desc, &entry->base);
  if (result == flagcxSuccess) {
    entry->proxyRank = proxyRank;
    entry->poolId = buffer->poolId;
    entry->capacity = buffer->capacity;
    entry->handleData = buffer->handleData;
    entry->handleSize = buffer->handleSize;
    entry->next = state->relayImports;
    state->relayImports = entry;
    *out = entry;
  } else {
    free(entry);
  }
  pthread_mutex_unlock(&state->relayImportMutex);
  return result;
}

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
          resources->tpRemoteRank = peer;
          resources->channelId = c;
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
          int logicalRanksPerNode = 0;
          const char *logicalRanksEnv =
              flagcxGetEnv("FLAGCX_CI_PXN_RANKS_PER_NODE");
          if (logicalRanksEnv != NULL) {
            char *end = NULL;
            const long parsed = strtol(logicalRanksEnv, &end, 10);
            if (end == logicalRanksEnv || *end != '\0' || parsed <= 0 ||
                parsed > comm->nRanks || comm->nRanks % parsed != 0) {
              free(listenInfo);
              return flagcxInvalidArgument;
            }
            logicalRanksPerNode = static_cast<int>(parsed);
          }
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
                  flagcxTopoSelectNetRoute(
                      comm->topoServer, remote, comm->interServerTopo,
                      comm->rank, peer, peerNetDev, &sendNetDev, &proxyRank,
                      logicalRanksPerNode) != flagcxSuccess) {
                sendNetDev = comm->netDev;
                proxyRank = comm->rank;
              }
            }
          }
          uint64_t relayNetGuid = 0;
          const int selectedProxyRank = proxyRank;
          const int selectedSendNetDev = sendNetDev;
          flagcxResult_t routeResult = flagcxNetFinalizePxnSendRoute(
              comm->netAdaptor, comm->rank, comm->netDev, &sendNetDev,
              &proxyRank, &relayNetGuid);
          if (routeResult != flagcxSuccess) {
            free(listenInfo);
            return routeResult;
          }
          if (selectedProxyRank != comm->rank && proxyRank == comm->rank)
            INFO(FLAGCX_NET,
                 "PXN fallback to direct NET: rank %d peer %d channel %d "
                 "selected relay %d NET/%d lacks ordinary device MR",
                 comm->rank, peer, c, selectedProxyRank, selectedSendNetDev);
          if (proxyRank != comm->rank)
            INFO(FLAGCX_NET,
                 "PXN route: %d -> %d channel %d recv NET/%d send "
                 "NET/%d proxy rank %d",
                 comm->rank, peer, c, listenInfo->netDev, sendNetDev,
                 proxyRank);
          FLAGCXCHECK(flagcxProxyConnect(comm, TRANSPORT_NET, 1, proxyRank,
                                         &conn->proxyConn));
          conn->proxyConn.netDev = sendNetDev;
          conn->proxyConn.peerNetDev = peerNetDev;
          if (proxyRank == comm->rank) {
            struct sendNetResources *resources;
            FLAGCXCHECK(flagcxCalloc(&resources, 1));
            conn->proxyConn.connection->transportResources = resources;
            resources->commPtr = comm;
            resources->tpRemoteRank = peer;
            resources->channelId = c;
            FLAGCXCHECK(flagcxNetInitSendResources(comm->netAdaptor, sendNetDev,
                                                   resources));
          } else {
            struct flagcxNetSendSetupRequest setup = {relayNetGuid};
            INFO(FLAGCX_NET,
                 "PXN relay setup request: source %d peer %d relay %d "
                 "channel %d NET/%d",
                 comm->rank, peer, proxyRank, c, sendNetDev);
            FLAGCXCHECK(flagcxProxyCallBlocking(comm, &conn->proxyConn,
                                                flagcxProxyMsgSetup, &setup,
                                                sizeof(setup), NULL, 0));
            INFO(FLAGCX_NET,
                 "PXN relay setup complete: source %d peer %d relay %d "
                 "channel %d",
                 comm->rank, peer, proxyRank, c);
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
                relayBuffer.capacity < size_t(flagcxNetChunkSize) ||
                relayBuffer.chunkSize != size_t(flagcxNetChunkSize) ||
                relayBuffer.poolId == 0)
              return flagcxInternalError;
            flagcxNetRelayImport *import = NULL;
            FLAGCXCHECK(flagcxGetRelayImport(comm, conn->proxyConn.tpRank,
                                             &relayBuffer, &import));
            conn->proxyConn.connection->relayImport = import;
            conn->proxyConn.connection->relayBufferImport = import->base;
            conn->proxyConn.connection->relayBufferCapacity = import->capacity;
            INFO(FLAGCX_NET,
                 "PXN imported shared relay pool %llu: source %d relay %d "
                 "channel %d capacity %zu",
                 static_cast<unsigned long long>(import->poolId), comm->rank,
                 conn->proxyConn.tpRank, c, import->capacity);
          }
        }
        comm->channels[c].peers[peer]->send[0].connected = 1;
        comm->connectSend[peer] ^= (1UL << c);
      }
    }
  }
  return flagcxSuccess;
}
