#include "net.h"
#include "adaptor.h"
#include "adaptor_plugin_load.h"
#include "device.h"
#include "gdr_visibility.h"
#include "ib_common.h"
#include "net_transport.h"
#include "p2p.h"
#include "proxy.h"
#include "reg_pool.h"
#include "transport.h"

#include <errno.h>
#include <string.h>
#include <string>
#include <time.h>
#include <unistd.h>

namespace {

flagcxResult_t
flagcxGetCollectiveWriteVisibilityPolicy(struct recvNetResources *resources,
                                         bool *required) {
  if (resources == NULL || resources->netAdaptor == NULL || required == NULL)
    return flagcxNotSupported;
  const uint8_t registrationRoute =
      resources->useDmaBuf ? static_cast<uint8_t>(FLAGCX_VMM_MR_ROUTE_DMABUF)
                           : static_cast<uint8_t>(FLAGCX_VMM_MR_ROUTE_NONE);
  // Derive from the concrete buffer path as well as setup metadata. This keeps
  // upgraded plugins and lightweight proxy fixtures fail-closed if they do not
  // explicitly populate useGdr.
  const int useGdr = resources->useGdr ||
                     resources->netAdaptor == getNetAdaptor(RDMA) ||
                     (resources->ptrSupport & FLAGCX_PTR_CUDA) != 0;
  flagcxGdrVisibilityDecision visibility = {};
  FLAGCXCHECK(flagcxResolveGdrVisibilityForConnection(
      resources->commPtr == NULL ? NULL : resources->commPtr->topoServer,
      resources->commPtr == NULL ? -1 : resources->commPtr->rank,
      resources->commPtr == NULL ? 0 : resources->commPtr->compCap,
      resources->netAdaptor, resources->netDev, useGdr,
      /*peerGpuMayPublish=*/0, registrationRoute, &visibility));
  *required = (visibility.requirements & FLAGCX_GDR_WRITE_REQUIRES_FLUSH) != 0;
  if (!*required)
    return flagcxSuccess;
  const flagcxResult_t capabilityResult = flagcxValidateGdrFlushCapability(
      visibility.requirements, resources->netAdaptor->gdrFlushCaps,
      FLAGCX_GDR_WRITE_REQUIRES_FLUSH);
  if (capabilityResult != flagcxSuccess) {
    // BAREX intentionally keeps a legacy no-op callback for automatic PPU CI
    // compatibility. An explicitly forced WRITE requirement cannot use that
    // callback as proof of device visibility.
    return capabilityResult;
  }
  return flagcxSuccess;
}

class flagcxCollSubmitScope {
public:
  explicit flagcxCollSubmitScope(const struct flagcxNetSubmitContext *context) {
    active_ = flagcxNetSetSubmitContext(context) == flagcxSuccess;
  }
  ~flagcxCollSubmitScope() {
    if (active_)
      flagcxNetClearSubmitContext();
  }

private:
  bool active_ = false;
};

} // namespace

int64_t flagcxNetBufferSize;
int64_t flagcxNetChunkSize;
int64_t flagcxNetChunks;

struct flagcxNetRelaySendState {
  bool counted;
  flagcxStream_t stream;
  flagcxEvent_t event;
  size_t offset;
  size_t chunkBytes;
  uint64_t requestId;
  uint64_t cancelDeadlineNs;
  flagcxResult_t terminalResult;
  uint8_t sendReleased;
  uint8_t cancelReleased;
  int phase; // 0: ready, 1: copying, 2: send reply, 3/4: cancel send/reply
};

static uint64_t flagcxNextRelayRequestId = 0;

static uint64_t flagcxRelayMonotonicNs() {
  struct timespec now = {};
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
    return 0;
  return uint64_t(now.tv_sec) * 1000000000ULL + now.tv_nsec;
}

void flagcxNetCleanupRelaySendOp(struct flagcxProxyOp *op) {
  if (op == NULL || op->relaySendState == NULL)
    return;
  auto *state = static_cast<flagcxNetRelaySendState *>(op->relaySendState);
  // An uncertain RPC keeps the connection mapping alive until process exit.
  // The relay owns the allocation and can still be reading from it.
  if (state->phase >= 2) {
    WARN("PXN relay reply lost; retaining connection IPC mapping until process "
         "exit");
    op->relaySendState = NULL;
    return;
  }
  if (state->phase == 1 && state->stream != NULL &&
      deviceAdaptor->streamSynchronize(state->stream) != flagcxSuccess) {
    WARN("PXN relay copy completion is unknown; retaining connection IPC "
         "mapping until process exit");
    op->relaySendState = NULL;
    return;
  }
  if (state->event != NULL)
    (void)deviceAdaptor->eventDestroy(state->event);
  if (state->stream != NULL)
    (void)deviceAdaptor->streamDestroy(state->stream);
  if (state->counted && op->connection != NULL)
    __atomic_sub_fetch(&op->connection->relayActiveOps, 1, __ATOMIC_ACQ_REL);
  free(state);
  op->relaySendState = NULL;
}

void flagcxNetAbandonRelaySendOp(struct flagcxProxyOp *op) {
  if (op != NULL && op->relaySendState != NULL) {
    WARN("PXN relay abort before group completion; retaining connection IPC "
         "mapping until process exit");
    op->relaySendState = NULL;
  }
}

static flagcxResult_t flagcxNetProgressRelaySend(struct flagcxProxyOp *op) {
  if (op->comm == NULL || op->nbytes < 0 || flagcxNetChunkSize <= 0)
    return flagcxInvalidArgument;
  if (op->args.done)
    return flagcxSuccess;
  if (!op->args.semaphore->pollStart(op->args.opId, op->args.step))
    return flagcxSuccess;

  auto *state = static_cast<flagcxNetRelaySendState *>(op->relaySendState);
  if (op->nbytes == 0) {
    op->args.semaphore->subCounter(op->args.opId);
    op->args.done = 1;
    return flagcxSuccess;
  }
  if (state == NULL)
    return flagcxInternalError;

  if (state->phase == 0) {
    state->chunkBytes =
        std::min(static_cast<size_t>(flagcxNetChunkSize),
                 static_cast<size_t>(op->nbytes) - state->offset);
    FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
        op->connection->relayBufferImport,
        reinterpret_cast<char *>(op->recvbuff) + state->offset,
        state->chunkBytes, flagcxMemcpyDeviceToDevice, state->stream, NULL));
    state->phase = 1;
    FLAGCXCHECK(deviceAdaptor->eventRecord(state->event, state->stream));
  }

  if (state->phase == 1) {
    int completed = 0;
    FLAGCXCHECK(flagcxTransportClassifyCompletion(
        deviceAdaptor->eventQuery(state->event), &completed));
    if (!completed)
      return flagcxSuccess;
    flagcxNetRelaySendRequest request = {};
    request.bytes = state->chunkBytes;
    state->requestId =
        __atomic_add_fetch(&flagcxNextRelayRequestId, 1, __ATOMIC_RELAXED);
    request.requestId = state->requestId;
    request.generation = op->args.collTransport.generation;
    request.orderingKey = op->args.collTransport.orderingKey;
    request.sequence = state->offset / flagcxNetChunkSize;
    request.submitFlags = op->args.collTransport.submitFlags;
    auto *connector =
        &op->comm->channels[op->channelId].peers[op->root]->send[0].proxyConn;
    state->sendReleased = 0;
    // A failed socket write may already have delivered a complete request.
    // Treat ownership as uncertain before publishing any part of the frame.
    state->phase = 2;
    FLAGCXCHECK(flagcxProxyCallAsync(op->comm, connector,
                                     flagcxProxyMsgSendRecv, &request,
                                     sizeof(request), 1, op));
  }

  if (state->phase == 2) {
    auto *connector =
        &op->comm->channels[op->channelId].peers[op->root]->send[0].proxyConn;
    bool responseReceived = false;
    flagcxResult_t result = flagcxPollProxyResponseWithStatus(
        op->comm, connector, &state->sendReleased, op, &responseReceived);
    if (result == flagcxInProgress)
      return flagcxSuccess;
    if (!responseReceived) {
      state->terminalResult = result;
      state->phase = 3;
    } else if (state->sendReleased == 0) {
      // The relay replied, but did not certify that this slot is idle.
      return result == flagcxSuccess ? flagcxInternalError : result;
    } else if (result != flagcxSuccess) {
      state->phase = 0;
      return result;
    } else {
      __atomic_add_fetch(&op->comm->pxnRelayChunksCompleted, 1,
                         __ATOMIC_RELEASE);
      state->phase = 0;
      state->offset += state->chunkBytes;
      if (state->offset == static_cast<size_t>(op->nbytes)) {
        op->args.semaphore->subCounter(op->args.opId);
        op->args.done = 1;
      }
    }
  }
  if (state->phase == 3) {
    auto *connector =
        &op->comm->channels[op->channelId].peers[op->root]->send[0].proxyConn;
    flagcxNetRelayCancelRequest cancel = {state->requestId};
    state->cancelReleased = 0;
    if (flagcxProxyCallAsync(op->comm, connector, flagcxProxyMsgCancelRelay,
                             &cancel, sizeof(cancel), 1,
                             state) != flagcxSuccess) {
      // No reliable reply is possible. Retain the connection IPC mapping.
      state->phase = 2;
      return state->terminalResult;
    }
    const uint64_t now = flagcxRelayMonotonicNs();
    state->cancelDeadlineNs = now == 0 ? 0 : now + 30000000000ULL;
    state->phase = 4;
  }
  if (state->phase == 4) {
    auto *connector =
        &op->comm->channels[op->channelId].peers[op->root]->send[0].proxyConn;
    bool responseReceived = false;
    flagcxResult_t cancelResult = flagcxPollProxyResponseWithStatus(
        op->comm, connector, &state->cancelReleased, state, &responseReceived);
    if (cancelResult == flagcxInProgress && state->cancelDeadlineNs != 0 &&
        flagcxRelayMonotonicNs() < state->cancelDeadlineNs)
      return flagcxSuccess;
    if (responseReceived && cancelResult == flagcxSuccess &&
        state->cancelReleased != 0) {
      flagcxProxyForgetResponse(op->comm, op);
      state->phase = 0;
    } else {
      state->phase = 2;
    }
    return state->terminalResult;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxNetDevFromGuid(struct flagcxNetAdaptor *netAdaptor,
                                    uint64_t netGuid, int *netDev) {
  if (netAdaptor == NULL || netAdaptor->devices == NULL ||
      netAdaptor->getProperties == NULL || netDev == NULL || netGuid == 0)
    return flagcxInvalidArgument;

  int ndev = 0;
  FLAGCXCHECK(netAdaptor->devices(&ndev));
  if (ndev <= 0)
    return flagcxNotSupported;

  int found = -1;
  for (int dev = 0; dev < ndev; ++dev) {
    flagcxNetProperties_t props = {};
    FLAGCXCHECK(netAdaptor->getProperties(dev, &props));
    if (props.guid != netGuid)
      continue;
    if (found >= 0)
      return flagcxNotSupported;
    found = dev;
  }
  if (found < 0)
    return flagcxNotSupported;
  *netDev = found;
  return flagcxSuccess;
}

flagcxResult_t flagcxNetInitSendResources(struct flagcxNetAdaptor *netAdaptor,
                                          int netDev,
                                          struct sendNetResources *resources,
                                          bool relay) {
  if (netAdaptor == NULL || resources == NULL || netDev < 0 ||
      flagcxNetChunks > FLAGCX_NET_MAX_STEPS)
    return flagcxInvalidArgument;

  resources->netDev = netDev;
  resources->netAdaptor = netAdaptor;
  FLAGCXCHECK(deviceAdaptor->streamCreate(&resources->cpStream));
  for (int s = 0; s < flagcxNetChunks; ++s)
    FLAGCXCHECK(deviceAdaptor->eventCreate(&resources->cpEvents[s],
                                           flagcxEventDisableTiming));

  resources->buffSizes[0] = flagcxNetBufferSize;
  if (relay) {
    if (resources->netAdaptor != getNetAdaptor(RDMA)) {
      flagcxNetProperties_t props = {};
      FLAGCXCHECK(resources->netAdaptor->getProperties(netDev, &props));
      resources->ptrSupport = props.ptrSupport;
      if ((resources->ptrSupport & FLAGCX_PTR_CUDA) == 0)
        return flagcxNotSupported;
    }
    FLAGCXCHECK(deviceAdaptor->gdrMemAlloc(
        reinterpret_cast<void **>(&resources->buffers[0]),
        resources->buffSizes[0], NULL));
    resources->relayExportBuffer = resources->buffers[0];
    resources->relayIpcBuffer = true;
    // No descriptor has been published yet, so the source cannot own a map.
    resources->relaySourceReleased = true;
    flagcxP2pIpcDesc desc = {};
    FLAGCXCHECK(flagcxP2pExportShareableBuffer(resources->relayExportBuffer,
                                               resources->buffSizes[0], &desc));
    resources->relayHandleData = desc.handleData;
    resources->relayHandleSize = desc.handleSize;
  } else if (resources->netAdaptor == getNetAdaptor(SOCKET)) {
    resources->buffers[0] = (char *)malloc(resources->buffSizes[0]);
    if (resources->buffers[0] == NULL)
      return flagcxSystemError;
  } else if (resources->netAdaptor == getNetAdaptor(RDMA)) {
    FLAGCXCHECK(deviceAdaptor->gdrMemAlloc((void **)&resources->buffers[0],
                                           resources->buffSizes[0], NULL));
  } else {
    flagcxNetProperties_t props = {};
    FLAGCXCHECK(resources->netAdaptor->getProperties(netDev, &props));
    resources->ptrSupport = props.ptrSupport;
    if (resources->ptrSupport & FLAGCX_PTR_CUDA) {
      FLAGCXCHECK(deviceAdaptor->gdrMemAlloc((void **)&resources->buffers[0],
                                             resources->buffSizes[0], NULL));
    } else {
      resources->buffers[0] = (char *)malloc(resources->buffSizes[0]);
      if (resources->buffers[0] == NULL)
        return flagcxSystemError;
    }
  }
  resources->useGdr = resources->netAdaptor != getNetAdaptor(SOCKET) &&
                      (resources->netAdaptor == getNetAdaptor(RDMA) ||
                       (resources->ptrSupport & FLAGCX_PTR_CUDA) != 0);
  return flagcxSuccess;
}

flagcxResult_t flagcxNetPrepareProxyOp(struct flagcxHeteroComm *comm,
                                       struct flagcxProxyOp *op, void *buffer,
                                       size_t size, int peer,
                                       flagcxDataType_t dtype) {
  (void)dtype;
  if (comm == NULL || op == NULL || op->connection == NULL)
    return flagcxInvalidArgument;

  op->args.chunkSize = flagcxNetChunkSize;
  op->args.chunkSteps = (size + flagcxNetChunkSize - 1) / flagcxNetChunkSize;
  op->args.sendStepMask = flagcxNetChunks - 1;
  const int srcRank = op->pattern == flagcxPatternRecv ? peer : comm->rank;
  const int dstRank = op->pattern == flagcxPatternRecv ? comm->rank : peer;
  const uint64_t orderingKey =
      flagcxCollProxyOrderingKey(op->channelId, srcRank, dstRank);
  const uint32_t submitFlags =
      op->channelId == 0 ? 0 : FLAGCX_NET_SUBMIT_INDEPENDENT;
  const uint64_t generation =
      __atomic_add_fetch(&op->connection->collGeneration, 1, __ATOMIC_RELAXED);
  FLAGCXCHECK(flagcxCollProxyTransportInit(
      &op->args.collTransport, (uint32_t)flagcxNetChunks, generation,
      orderingKey, submitFlags));
  op->args.collTransport.laneMask = &op->connection->collDataLaneMask;
  flagcxConnector *peerConns[] = {
      op->pattern == flagcxPatternRecv
          ? comm->channels[op->channelId].peers[peer]->recv
          : comm->channels[op->channelId].peers[peer]->send};
  if (op->pattern == flagcxPatternSend &&
      !peerConns[0]->proxyConn.sameProcess) {
    // The NET registration belongs to the relay. The source writes into the
    // relay-owned persistent buffer and asks the relay to send that slot.
    op->args.regBufFlag = 0;
    op->args.regHandle = NULL;
    if (size != 0) {
      if (flagcxNetChunkSize <= 0)
        return flagcxInvalidArgument;
      if (op->connection->relayBufferImport == NULL ||
          op->connection->relayBufferCapacity <
              static_cast<size_t>(flagcxNetChunkSize))
        return flagcxInternalError;
      flagcxNetRelaySendState *state = NULL;
      FLAGCXCHECK(flagcxCalloc(&state, 1));
      op->relaySendState = state;
      flagcxResult_t result = deviceAdaptor->streamCreate(&state->stream);
      if (result == flagcxSuccess)
        result =
            deviceAdaptor->eventCreate(&state->event, flagcxEventDisableTiming);
      if (result != flagcxSuccess) {
        flagcxNetCleanupRelaySendOp(op);
        return result;
      }
      __atomic_add_fetch(&op->connection->relayActiveOps, 1, __ATOMIC_ACQ_REL);
      state->counted = true;
    }
    return flagcxSuccess;
  }
  return flagcxNetRegisterBuffer(comm, buffer, size, peerConns, 1,
                                 &op->args.regBufFlag, &op->args.regHandle);
}

flagcxResult_t
flagcxNetProgressProxyOp(struct flagcxProxyConnection *connection,
                         struct flagcxProxyOp *op) {
  if (connection == NULL || op == NULL)
    return flagcxInvalidArgument;
  if (connection->send && connection->transportResources == NULL &&
      op->comm != NULL)
    return flagcxNetProgressRelaySend(op);
  if (connection->transportResources == NULL)
    return flagcxInvalidArgument;
  return connection->send
             ? flagcxProxySend(
                   (struct sendNetResources *)connection->transportResources,
                   op->recvbuff, op->nbytes, &op->args)
             : flagcxProxyRecv(
                   (struct recvNetResources *)connection->transportResources,
                   op->recvbuff, op->nbytes, &op->args);
}

flagcxResult_t
flagcxNetCleanupProxyConnection(struct flagcxProxyConnection *connection,
                                int cleanupPhase) {
  if (connection == NULL)
    return flagcxInvalidArgument;
  if (connection->transportResources == NULL ||
      cleanupPhase == flagcxTransportCleanupCloseImports)
    return flagcxSuccess;
  if (cleanupPhase != flagcxTransportCleanupReleaseResources)
    return flagcxInvalidArgument;
  flagcxResult_t result =
      connection->send
          ? flagcxSendProxyFree(
                (struct sendNetResources *)connection->transportResources)
          : flagcxRecvProxyFree(
                (struct recvNetResources *)connection->transportResources);
  free(connection->transportResources);
  connection->transportResources = NULL;
  return result;
}

static pthread_mutex_t netLock = PTHREAD_MUTEX_INITIALIZER;
// Use adaptor system for all network types
struct flagcxNetAdaptor *flagcxNetAdaptors[3] = {nullptr, getNetAdaptor(RDMA),
                                                 getNetAdaptor(SOCKET)};
enum flagcxNetState flagcxNetStates[3] = {
    flagcxNetStateInit, flagcxNetStateInit, flagcxNetStateInit};

static bool flagcxNetCanUseDmaBuf(struct flagcxNetAdaptor *netAdaptor,
                                  const flagcxNetProperties_t *properties) {
  if (netAdaptor == NULL || properties == NULL)
    return false;
  // DMA-BUF is an alternate GDR route only when this process will actually
  // export device buffers through it. A property bit by itself is insufficient.
  const char *enabled = flagcxGetEnv("FLAGCX_DMABUF_ENABLE");
  if (netAdaptor != getNetAdaptor(RDMA) || enabled == NULL ||
      strcmp(enabled, "1") != 0 ||
      (properties->ptrSupport & FLAGCX_PTR_DMABUF) == 0 ||
      deviceAdaptor == NULL || deviceAdaptor->dmaSupport == NULL ||
      deviceAdaptor->getHandleForAddressRange == NULL)
    return false;
  bool supported = false;
  return deviceAdaptor->dmaSupport(&supported) == flagcxSuccess && supported;
}

bool flagcxNetCanUseDeviceMemory(struct flagcxNetAdaptor *netAdaptor,
                                 const flagcxNetProperties_t *properties) {
  return netAdaptor != NULL && properties != NULL &&
         ((properties->ptrSupport & FLAGCX_PTR_CUDA) != 0 ||
          flagcxNetCanUseDmaBuf(netAdaptor, properties));
}

flagcxResult_t flagcxGpuGdrSupport(struct flagcxHeteroComm *comm,
                                   int *gdrSupport) {
  if (comm == NULL || comm->netAdaptor == NULL || gdrSupport == NULL)
    return flagcxInvalidArgument;
  *gdrSupport = 0;
  if (deviceAdaptor == NULL || deviceAdaptor->gdrMemAlloc == NULL ||
      deviceAdaptor->gdrMemFree == NULL)
    return flagcxSuccess;

  // As in NCCL's registration fallback, one successful local registration
  // establishes a GPU-level capability. It does not authorize every GPU/NIC
  // pair; the selected connection still performs its own MR registration.
  void *gpuPtr = NULL;
  const flagcxResult_t allocResult =
      deviceAdaptor->gdrMemAlloc(&gpuPtr, 64 * 1024, NULL);
  if (allocResult != flagcxSuccess || gpuPtr == NULL) {
    if (gpuPtr != NULL)
      (void)deviceAdaptor->gdrMemFree(gpuPtr, NULL);
    return flagcxSuccess;
  }
  int ndev = 0;
  bool retainAllocation = false;
  if (comm->netAdaptor->devices(&ndev) == flagcxSuccess) {
    for (int dev = 0; dev < ndev; ++dev) {
      flagcxNetProperties_t props = {};
      if (comm->netAdaptor->getProperties(dev, &props) != flagcxSuccess ||
          !flagcxNetCanUseDeviceMemory(comm->netAdaptor, &props))
        continue;

      if (props.ptrSupport & FLAGCX_PTR_CUDA) {
        bool builtInIb = comm->netAdaptor == &flagcxNetIb;
#ifdef USE_IBUC
        builtInIb |= comm->netAdaptor == &flagcxNetIbuc;
#endif
        if (builtInIb) {
          bool registered = false;
          const int access =
              IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
              IBV_ACCESS_REMOTE_READ |
              (comm->netAdaptor == &flagcxNetIb ? IBV_ACCESS_REMOTE_ATOMIC : 0);
          if (flagcxIbProbeGpuMrSupport(dev, access, &registered) ==
                  flagcxSuccess &&
              registered) {
            *gdrSupport = 1;
            break;
          }
        }
#ifdef USE_ACCL_BAREX
        else if (comm->netAdaptor == &flagcxNetBarex &&
                 comm->netAdaptor->regMr != NULL &&
                 comm->netAdaptor->deregMr != NULL) {
          // BAREX owns a runtime-wide memory pool; its MR API needs no
          // connection. A failed deregistration can leave a provider reference,
          // so keep the small allocation alive rather than freeing it early.
          void *handle = NULL;
          const flagcxResult_t regResult = comm->netAdaptor->regMr(
              NULL, gpuPtr, 64 * 1024, FLAGCX_PTR_CUDA, 0, &handle);
          if (regResult != flagcxSuccess || handle == NULL) {
            retainAllocation = true;
            break;
          }
          if (comm->netAdaptor->deregMr(NULL, handle) != flagcxSuccess) {
            retainAllocation = true;
            break;
          }
          *gdrSupport = 1;
          break;
        }
#endif
      }
      if (flagcxNetCanUseDmaBuf(comm->netAdaptor, &props)) {
        int dmaBufFd = -1;
        if (deviceAdaptor->getHandleForAddressRange(
                &dmaBufFd, gpuPtr, 64 * 1024, 0) == flagcxSuccess &&
            dmaBufFd >= 0)
          *gdrSupport = 1;
        if (dmaBufFd >= 0)
          (void)close(dmaBufFd);
        if (*gdrSupport)
          break;
      }
    }
  }
  if (retainAllocation) {
    WARN("GPU GDR probe could not confirm MR cleanup; retaining probe buffer");
    *gdrSupport = 0;
  } else if (deviceAdaptor->gdrMemFree(gpuPtr, NULL) != flagcxSuccess) {
    *gdrSupport = 0;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxNetCheckDeviceVersion(struct flagcxHeteroComm *comm,
                                           struct flagcxNetAdaptor *net,
                                           int dev) {
  flagcxNetProperties_v1_t props;

  FLAGCXCHECK(net->getProperties(dev, (void *)&props));
  flagcxNetDeviceType type = props.netDeviceType;
  if (type)
    switch (type) {
      case FLAGCX_NET_DEVICE_UNPACK:
        if (props.netDeviceVersion == FLAGCX_NET_DEVICE_UNPACK_VERSION) {
          INFO(FLAGCX_INIT,
               "Using FLAGCX_NET_DEVICE_UNPACK net plugin version %d",
               props.netDeviceVersion);
          return flagcxSuccess;
        } else {
          WARN("FLAGCX_DEVICE_UNPACK plugin has incompatible version %d, this "
               "flagcx build is compatible with %d, not using it",
               props.netDeviceVersion, FLAGCX_NET_DEVICE_UNPACK_VERSION);
          return flagcxInternalError;
        }
      default:
        WARN("Unknown device code index");
        return flagcxInternalError;
    }

  INFO(FLAGCX_INIT, "Using non-device net plugin version %d",
       props.netDeviceVersion);
  return flagcxSuccess;
}

static flagcxResult_t netGetState(int i, enum flagcxNetState *state) {
  pthread_mutex_lock(&netLock);
  if (flagcxNetStates[i] == flagcxNetStateInit) {
    int ndev;
    if (flagcxNetAdaptors[i] == nullptr) {
      flagcxNetStates[i] = flagcxNetStateDisabled;
    } else if (flagcxNetAdaptors[i]->init() != flagcxSuccess) {
      flagcxNetStates[i] = flagcxNetStateDisabled;
    } else if (flagcxNetAdaptors[i]->devices(&ndev) != flagcxSuccess ||
               ndev <= 0) {
      flagcxNetStates[i] = flagcxNetStateDisabled;
    } else {
      flagcxNetStates[i] = flagcxNetStateEnabled;
    }
  }
  *state = flagcxNetStates[i];
  pthread_mutex_unlock(&netLock);
  return flagcxSuccess;
}

flagcxResult_t flagcxNetInit(struct flagcxHeteroComm *comm) {
  // Initialize main communication network
  const char *netName;
  bool ok = false;

  const char *forceSocketEnv = getenv("FLAGCX_FORCE_NET_SOCKET");
  bool forceSocket = (forceSocketEnv && atoi(forceSocketEnv) == 1);

  netName = comm->config.netName;

  if (!forceSocket) {
    // Load net plugin if FLAGCX_NET_ADAPTOR_PLUGIN is set.
    // This populates flagcxNetAdaptors[0] with the plugin.
    // Must be called before the selection loop below.
    FLAGCXCHECK(flagcxNetAdaptorPluginInit());
  }

  if (forceSocket) {
    // Force socket network usage
    for (int i = 2; i >= 0; i--) {
      if (flagcxNetAdaptors[i] == nullptr)
        continue;
      if (flagcxNetAdaptors[i] != getNetAdaptor(SOCKET))
        continue;
      enum flagcxNetState state;
      FLAGCXCHECK(netGetState(i, &state));
      if (state != flagcxNetStateEnabled)
        continue;
      if (netName && strcasecmp(netName, flagcxNetAdaptors[i]->name) != 0)
        continue;
      if (flagcxSuccess !=
          flagcxNetCheckDeviceVersion(comm, flagcxNetAdaptors[i], 0)) {
        continue;
      }

      comm->netAdaptor = flagcxNetAdaptors[i];
      ok = true;

      break;
    }
  } else {
    // Normal network selection order: plugin, build-selected RDMA adaptor,
    // then socket.
    for (int i = 0; i < 3; i++) {
      if (flagcxNetAdaptors[i] == nullptr)
        continue;
      enum flagcxNetState state;
      FLAGCXCHECK(netGetState(i, &state));
      if (state != flagcxNetStateEnabled)
        continue;
      if (netName && strcasecmp(netName, flagcxNetAdaptors[i]->name) != 0)
        continue;
      if (flagcxSuccess !=
          flagcxNetCheckDeviceVersion(comm, flagcxNetAdaptors[i], 0)) {
        continue;
      }

      comm->netAdaptor = flagcxNetAdaptors[i];
      ok = true;

      break;
    }
  }

  if (!ok) {
    WARN("Error: network %s not found.", netName ? netName : "");
    return flagcxInvalidUsage;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxProxySend(sendNetResources *resources, void *data,
                               size_t size, flagcxProxyArgs *args) {
  if (args->done) {
    return flagcxSuccess;
  }
  if (!args->semaphore->pollStart(args->opId, args->step)) {
    return flagcxSuccess;
  }
  if (args->transmitted < args->chunkSteps) {
    int stepMask = args->sendStepMask;

    if (args->waitCopy < args->chunkSteps &&
        args->waitCopy - args->transmitted < flagcxNetChunks) {
      int step = args->waitCopy & stepMask;
      args->subs[step].stepSize =
          std::min(args->chunkSize, size - args->totalCopySize);
      if (!args->regBufFlag) {
        args->subs[step].stepBuff =
            resources->buffers[0] + (flagcxNetChunkSize * step);
        if (resources->netAdaptor == getNetAdaptor(RDMA)) {
          FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
              args->subs[step].stepBuff, (char *)data + args->totalCopySize,
              args->subs[step].stepSize, flagcxMemcpyDeviceToDevice,
              resources->cpStream, args->subs[step].copyArgs));
        } else if (resources->netAdaptor == getNetAdaptor(SOCKET)) {
          FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
              args->subs[step].stepBuff, (char *)data + args->totalCopySize,
              args->subs[step].stepSize, flagcxMemcpyDeviceToHost,
              resources->cpStream, args->subs[step].copyArgs));
        } else {
          FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
              args->subs[step].stepBuff, (char *)data + args->totalCopySize,
              args->subs[step].stepSize,
              (resources->ptrSupport & FLAGCX_PTR_CUDA)
                  ? flagcxMemcpyDeviceToDevice
                  : flagcxMemcpyDeviceToHost,
              resources->cpStream, args->subs[step].copyArgs));
        }
        FLAGCXCHECK(deviceAdaptor->eventRecord(resources->cpEvents[step],
                                               resources->cpStream));
      } else {
        args->subs[step].stepBuff =
            (void *)((char *)data + (flagcxNetChunkSize * args->waitCopy));
      }
      args->totalCopySize += args->subs[step].stepSize;
      args->waitCopy++;
    }

    if (args->posted < args->waitCopy) {
      int step = args->posted & stepMask;
      int done = 0;
      if (!args->regBufFlag) {
        int completed = 0;
        FLAGCXCHECK(flagcxTransportClassifyCompletion(
            deviceAdaptor->eventQuery(resources->cpEvents[step]), &completed));
        if (completed) {
          args->copied++;
          done = 1;
        }
      } else {
        done = 1;
      }
      if (done) {
        void *req = NULL;
        struct flagcxNetSubmitContext *submit = NULL;
        flagcxResult_t trackRes =
            flagcxCollProxyTrackNext(&args->collTransport, &submit);
        if (trackRes == flagcxInProgress)
          return flagcxSuccess;
        FLAGCXCHECK(trackRes);
        flagcxResult_t sendRes;
        {
          flagcxCollSubmitScope submitScope(submit);
          sendRes = resources->netAdaptor->isend(
              resources->netSendComm,
              args->subs[args->posted & stepMask].stepBuff,
              args->subs[args->posted & stepMask].stepSize, 0,
              args->regBufFlag ? args->regHandle : resources->mhandles[0], NULL,
              &req);
        }
        if (sendRes != flagcxSuccess && sendRes != flagcxInProgress) {
          FLAGCXCHECK(flagcxCollProxyCancel(&args->collTransport, submit));
          return sendRes;
        }
        if (req) {
          args->subs[args->posted++ & stepMask].requests[0] = req;
        } else {
          FLAGCXCHECK(flagcxCollProxyCancel(&args->collTransport, submit));
        }
      }
    }

    if (args->transmitted < args->posted) {
      // Poll every accepted request. CQEs from different physical QPs may be
      // observed in any order; the scoreboard retires only a contiguous
      // sequence prefix.
      for (int sequence = args->transmitted; sequence < args->posted;
           ++sequence) {
        int slot = sequence & stepMask;
        void *req = args->subs[slot].requests[0];
        if (req == NULL)
          continue;
        int done = 0, sizes = 0;
        flagcxResult_t testRes =
            resources->netAdaptor->test(req, &done, &sizes);
        if (testRes != flagcxSuccess && testRes != flagcxInProgress)
          return testRes;
        if (done) {
          args->subs[slot].requests[0] = NULL;
          uint32_t advanced = 0;
          FLAGCXCHECK(flagcxCollProxyComplete(
              &args->collTransport,
              &args->collTransport.contexts[(args->sequenceBase + sequence) %
                                            args->collTransport.capacity],
              flagcxSuccess, &advanced));
          args->transmitted += advanced;
        }
      }
    }
  } else {
    if (args->done != 1) {
      args->semaphore->subCounter(args->opId);
      args->done = 1;
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxProxyRecv(recvNetResources *resources, void *data,
                               size_t size, flagcxProxyArgs *args) {
  if (args->done) {
    return flagcxSuccess;
  }
  if (!args->semaphore->pollStart(args->opId, args->step)) {
    return flagcxSuccess;
  }
  if (args->copied < args->chunkSteps) {
    int stepMask = args->sendStepMask;
    if (args->posted < args->chunkSteps &&
        args->posted - args->copied < flagcxNetChunks) {
      int tags[8] = {0};
      void *req = NULL;
      const int sequence = args->posted;
      const int slot = sequence & stepMask;
      args->subs[slot].stepSize =
          std::min(args->chunkSize, size - args->totalPostSize);
      if (!args->regBufFlag) {
        args->subs[slot].stepBuff =
            resources->buffers[0] + flagcxNetChunkSize * slot;
      } else {
        args->subs[slot].stepBuff =
            (void *)((char *)data + flagcxNetChunkSize * args->posted);
      }
      struct flagcxNetSubmitContext *submit = NULL;
      flagcxResult_t trackRes =
          flagcxCollProxyTrackNext(&args->collTransport, &submit);
      if (trackRes == flagcxInProgress)
        return flagcxSuccess;
      FLAGCXCHECK(trackRes);
      flagcxResult_t recvRes;
      {
        flagcxCollSubmitScope submitScope(submit);
        recvRes = resources->netAdaptor->irecv(
            resources->netRecvComm, 1, &args->subs[slot].stepBuff,
            (size_t *)&args->subs[slot].stepSize, tags,
            args->regBufFlag ? &args->regHandle : resources->mhandles, NULL,
            &req);
      }
      if (recvRes != flagcxSuccess && recvRes != flagcxInProgress) {
        FLAGCXCHECK(flagcxCollProxyCancel(&args->collTransport, submit));
        return recvRes;
      }
      if (req) {
        args->netCompleted[slot] = 0;
        args->subs[slot].requests[0] = req;
        args->totalPostSize += args->subs[slot].stepSize;
        args->posted++;
        return flagcxSuccess;
      }
      FLAGCXCHECK(flagcxCollProxyCancel(&args->collTransport, submit));
    }

    // Receive CQEs are independent. Record every observed completion, then
    // let the flush/copy pipeline consume the contiguous prefix.
    for (int sequence = args->postFlush; sequence < args->posted; ++sequence) {
      int slot = sequence & stepMask;
      if (args->netCompleted[slot])
        continue;
      void *req = args->subs[slot].requests[0];
      int done = 0, sizes = 0;
      flagcxResult_t testRes = resources->netAdaptor->test(req, &done, &sizes);
      if (testRes != flagcxSuccess && testRes != flagcxInProgress)
        return testRes;
      if (done) {
        args->netCompleted[slot] = 1;
        args->subs[slot].requests[0] = NULL;
      }
    }

    if (args->postFlush < args->posted) {
      int step = args->postFlush & stepMask;
      if (args->netCompleted[step]) {
        if (resources->netAdaptor == getNetAdaptor(RDMA)) {
          bool flushRequired = false;
          FLAGCXCHECK(flagcxGetCollectiveWriteVisibilityPolicy(resources,
                                                               &flushRequired));
          if (!flushRequired) {
            args->subs[args->postFlush++ & stepMask].requests[0] = (void *)0x1;
            return flagcxSuccess;
          }
          void *req = NULL;
          flagcxResult_t flushRes = resources->netAdaptor->iflush(
              resources->netRecvComm, 1,
              &args->subs[args->postFlush & stepMask].stepBuff,
              &args->subs[args->postFlush & stepMask].stepSize,
              args->regBufFlag ? &args->regHandle : resources->mhandles, &req);
          if (flushRes != flagcxSuccess && flushRes != flagcxInProgress)
            return flushRes;
          if (req) {
            args->subs[args->postFlush++ & stepMask].requests[0] = req;
          } else if (flushRes == flagcxSuccess) {
            // Providers may complete a real flush synchronously. Advance the
            // pipeline instead of retrying the same completed step forever.
            args->subs[args->postFlush++ & stepMask].requests[0] = (void *)0x1;
          }
        } else if (resources->netAdaptor == getNetAdaptor(SOCKET)) {
          args->subs[args->postFlush++ & stepMask].requests[0] = (void *)0x1;
        } else {
          if (resources->ptrSupport & FLAGCX_PTR_CUDA) {
            // RDMA-style: flush
            bool flushRequired = false;
            FLAGCXCHECK(flagcxGetCollectiveWriteVisibilityPolicy(
                resources, &flushRequired));
            if (!flushRequired) {
              args->subs[args->postFlush++ & stepMask].requests[0] =
                  (void *)0x1;
              return flagcxSuccess;
            }
            void *req = NULL;
            flagcxResult_t flushRes = resources->netAdaptor->iflush(
                resources->netRecvComm, 1,
                &args->subs[args->postFlush & stepMask].stepBuff,
                &args->subs[args->postFlush & stepMask].stepSize,
                args->regBufFlag ? &args->regHandle : resources->mhandles,
                &req);
            if (flushRes != flagcxSuccess && flushRes != flagcxInProgress)
              return flushRes;
            if (req) {
              args->subs[args->postFlush++ & stepMask].requests[0] = req;
            } else if (flushRes == flagcxSuccess) {
              args->subs[args->postFlush++ & stepMask].requests[0] =
                  (void *)0x1;
            }
          } else {
            // Host-only: skip flush
            args->subs[args->postFlush++ & stepMask].requests[0] = (void *)0x1;
          }
        }
        return flagcxSuccess;
      }
    }

    if (args->waitCopy < args->postFlush) {
      int step = args->waitCopy & stepMask;
      void *req = args->subs[step].requests[0];
      int done = 0, sizes;
      if (req == (void *)0x1) {
        done = 1;
        sizes = 0;
      } else {
        flagcxResult_t testRes =
            resources->netAdaptor->test(req, &done, &sizes);
        if (testRes != flagcxSuccess && testRes != flagcxInProgress)
          return testRes;
      }
      if (done) {
        if (!args->regBufFlag) {
          if (resources->netAdaptor == getNetAdaptor(RDMA)) {
            FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
                (char *)data + args->totalCopySize, args->subs[step].stepBuff,
                args->subs[step].stepSize, flagcxMemcpyDeviceToDevice,
                resources->cpStream, args->subs[step].copyArgs));
          } else if (resources->netAdaptor == getNetAdaptor(SOCKET)) {
            FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
                (char *)data + args->totalCopySize, args->subs[step].stepBuff,
                args->subs[step].stepSize, flagcxMemcpyHostToDevice,
                resources->cpStream, args->subs[step].copyArgs));
          } else {
            FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
                (char *)data + args->totalCopySize, args->subs[step].stepBuff,
                args->subs[step].stepSize,
                (resources->ptrSupport & FLAGCX_PTR_CUDA)
                    ? flagcxMemcpyDeviceToDevice
                    : flagcxMemcpyHostToDevice,
                resources->cpStream, args->subs[step].copyArgs));
          }
          FLAGCXCHECK(deviceAdaptor->eventRecord(resources->cpEvents[step],
                                                 resources->cpStream));
        }
        args->totalCopySize += args->subs[step].stepSize;
        args->waitCopy++;
        return flagcxSuccess;
      }
    }

    if (args->copied < args->waitCopy) {
      int step = args->copied & stepMask;
      if (!args->regBufFlag) {
        int completed = 0;
        FLAGCXCHECK(flagcxTransportClassifyCompletion(
            deviceAdaptor->eventQuery(resources->cpEvents[step]), &completed));
        if (completed) {
          uint32_t advanced = 0;
          FLAGCXCHECK(flagcxCollProxyComplete(
              &args->collTransport,
              &args->collTransport
                   .contexts[args->copied % args->collTransport.capacity],
              flagcxSuccess, &advanced));
          args->copied += advanced;
        }
      } else {
        uint32_t advanced = 0;
        FLAGCXCHECK(flagcxCollProxyComplete(
            &args->collTransport,
            &args->collTransport
                 .contexts[args->copied % args->collTransport.capacity],
            flagcxSuccess, &advanced));
        args->copied += advanced;
      }
    }
  } else {
    if (args->done != 1) {
      args->semaphore->subCounter(args->opId);
      args->done = 1;
    }
  }
  return flagcxSuccess;
}

static void flagcxProxyCleanupResult(flagcxResult_t cleanupResult,
                                     flagcxResult_t *firstResult) {
  if (cleanupResult != flagcxSuccess && cleanupResult != flagcxInProgress &&
      *firstResult == flagcxSuccess)
    *firstResult = cleanupResult;
}

flagcxResult_t flagcxSendProxyFree(sendNetResources *resources) {
  if (resources == NULL)
    return flagcxSuccess;

  flagcxResult_t result = flagcxSuccess;
  bool relayNetReleased = true;
  for (int s = 0; s < flagcxNetChunks; s++) {
    if (resources->cpEvents[s] != NULL) {
      flagcxProxyCleanupResult(
          deviceAdaptor->eventDestroy(resources->cpEvents[s]), &result);
      resources->cpEvents[s] = NULL;
    }
  }
  if (resources->cpStream != NULL) {
    flagcxProxyCleanupResult(deviceAdaptor->streamDestroy(resources->cpStream),
                             &result);
    resources->cpStream = NULL;
  }
  if (resources->netSendComm != NULL && resources->mhandles[0] != NULL) {
    flagcxResult_t deregResult = resources->netAdaptor->deregMr(
        resources->netSendComm, resources->mhandles[0]);
    if (deregResult != flagcxSuccess)
      relayNetReleased = false;
    flagcxProxyCleanupResult(deregResult, &result);
    resources->mhandles[0] = NULL;
  }
  if (resources->netSendComm != NULL) {
    flagcxResult_t closeResult =
        resources->netAdaptor->closeSend(resources->netSendComm);
    if (closeResult != flagcxSuccess)
      relayNetReleased = false;
    flagcxProxyCleanupResult(closeResult, &result);
    resources->netSendComm = NULL;
  }
  if (resources->relayIpcBuffer) {
    if (!relayNetReleased) {
      WARN("PXN relay NET resources were not fully released; retaining GPU "
           "buffers until process exit");
    } else {
      if (!resources->relaySourceReleased) {
        WARN("PXN relay source did not confirm IPC close; retaining exported "
             "NET buffer until process exit");
      } else if (resources->relayExportBuffer != NULL) {
        flagcxProxyCleanupResult(
            deviceAdaptor->gdrMemFree(resources->relayExportBuffer, NULL),
            &result);
      }
    }
    resources->relayExportBuffer = NULL;
    resources->buffers[0] = NULL;
  } else if (resources->buffers[0] != NULL) {
    if (resources->netAdaptor == getNetAdaptor(SOCKET)) {
      free(resources->buffers[0]);
    } else if (resources->netAdaptor == getNetAdaptor(RDMA)) {
      flagcxProxyCleanupResult(
          deviceAdaptor->gdrMemFree(resources->buffers[0], NULL), &result);
    } else {
      if (resources->ptrSupport & FLAGCX_PTR_CUDA) {
        flagcxProxyCleanupResult(
            deviceAdaptor->gdrMemFree(resources->buffers[0], NULL), &result);
      } else {
        free(resources->buffers[0]);
      }
    }
    resources->buffers[0] = NULL;
  }
  return result;
}

flagcxResult_t flagcxRecvProxyFree(recvNetResources *resources) {
  if (resources == NULL)
    return flagcxSuccess;

  flagcxResult_t result = flagcxSuccess;
  for (int s = 0; s < flagcxNetChunks; s++) {
    if (resources->cpEvents[s] != NULL) {
      flagcxProxyCleanupResult(
          deviceAdaptor->eventDestroy(resources->cpEvents[s]), &result);
      resources->cpEvents[s] = NULL;
    }
  }
  if (resources->cpStream != NULL) {
    flagcxProxyCleanupResult(deviceAdaptor->streamDestroy(resources->cpStream),
                             &result);
    resources->cpStream = NULL;
  }
  if (resources->netRecvComm != NULL && resources->mhandles[0] != NULL) {
    flagcxProxyCleanupResult(
        resources->netAdaptor->deregMr(resources->netRecvComm,
                                       resources->mhandles[0]),
        &result);
    resources->mhandles[0] = NULL;
  }
  if (resources->netRecvComm != NULL) {
    flagcxProxyCleanupResult(
        resources->netAdaptor->closeRecv(resources->netRecvComm), &result);
    resources->netRecvComm = NULL;
  }
  if (resources->netListenComm != NULL) {
    flagcxProxyCleanupResult(
        resources->netAdaptor->closeListen(resources->netListenComm), &result);
    resources->netListenComm = NULL;
  }
  if (resources->buffers[0] != NULL) {
    if (resources->netAdaptor == getNetAdaptor(SOCKET)) {
      free(resources->buffers[0]);
    } else if (resources->netAdaptor == getNetAdaptor(RDMA)) {
      flagcxProxyCleanupResult(
          deviceAdaptor->gdrMemFree(resources->buffers[0], NULL), &result);
    } else {
      if (resources->ptrSupport & FLAGCX_PTR_CUDA) {
        flagcxProxyCleanupResult(
            deviceAdaptor->gdrMemFree(resources->buffers[0], NULL), &result);
      } else {
        free(resources->buffers[0]);
      }
    }
    resources->buffers[0] = NULL;
  }
  return result;
}

static flagcxResult_t netRegisterBuffer(flagcxHeteroComm *comm,
                                        const void *userbuff, size_t buffSize,
                                        struct flagcxConnector **peerConns,
                                        int nPeers, flagcxRegItem *regRecord,
                                        int *outRegBufFlag, void **outHandle) {
  *outRegBufFlag = 0;
  if (regRecord) {
    for (int p = 0; p < nPeers; ++p) {
      struct flagcxConnector *peerConn = peerConns[p];
      struct flagcxProxyConnector *peerProxyConn = NULL;
      bool found = false;
      if (peerConn == NULL)
        continue;
      peerProxyConn = &peerConn->proxyConn;
      for (auto it = regRecord->handles.begin(); it != regRecord->handles.end();
           it++) {
        if (it->first.proxyConn == peerProxyConn && it->first.handle &&
            it->first.ownerComm == comm) {
          found = true;
          outHandle[p] = it->first.handle;
          *outRegBufFlag = 1;
          INFO(FLAGCX_REG,
               "rank %d - NET reuse buffer %p size %ld (baseAddr %p size %ld) "
               "handle %p",
               comm->rank, userbuff, buffSize, (void *)regRecord->beginAddr,
               regRecord->endAddr - regRecord->beginAddr, it->first.handle);
          break;
        }
      }
      if (!found) {
        struct netRegInfo info = {regRecord->beginAddr,
                                  regRecord->endAddr - regRecord->beginAddr};
        void *handle = NULL;
        FLAGCXCHECK(flagcxProxyCallBlocking(
            (flagcxHeteroComm *)comm, peerProxyConn, flagcxProxyMsgRegister,
            &info, sizeof(struct netRegInfo), &handle, sizeof(void *)));
        if (handle) {
          FLAGCXCHECK(globalRegPool.addNetHandle(comm, regRecord, handle,
                                                 peerProxyConn));
          outHandle[p] = handle;
          *outRegBufFlag = 1;
          INFO(FLAGCX_REG,
               "rank %d - NET register userbuff %p (handle %p), buffSize %ld",
               comm->rank, userbuff, handle, buffSize);
        } else {
          INFO(FLAGCX_REG,
               "rank %d failed to NET register userbuff %p buffSize %ld",
               comm->rank, userbuff, buffSize);
        }
      }
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxNetRegisterBuffer(flagcxHeteroComm *comm,
                                       const void *userbuff, size_t buffSize,
                                       struct flagcxConnector **peerConns,
                                       int nPeers, int *outRegBufFlag,
                                       void **outHandle) {
  INFO(FLAGCX_REG, "comm = %p, userbuff = %p, buffSize = %ld, nPeers = %d",
       comm, userbuff, buffSize, nPeers);
  *outRegBufFlag = 0;
  if (comm && userbuff && buffSize > 0 && nPeers > 0) {
    flagcxRegItem *reg = globalRegPool.getItem(reinterpret_cast<void *>(comm),
                                               const_cast<void *>(userbuff));
    if (reg != NULL && reg->refCount > 0) {
      FLAGCXCHECK(netRegisterBuffer(comm, userbuff, buffSize, peerConns, nPeers,
                                    reg, outRegBufFlag, outHandle));
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxNetDeregisterBuffer(void *comm,
                                         struct flagcxProxyConnector *proxyConn,
                                         void *handle) {
  INFO(FLAGCX_REG, "rank %d - deregister net buffer handle %p",
       reinterpret_cast<flagcxHeteroComm *>(comm)->rank, handle);
  FLAGCXCHECK(flagcxProxyCallBlocking(
      reinterpret_cast<flagcxHeteroComm *>(comm), proxyConn,
      flagcxProxyMsgDeregister, &handle, sizeof(void *), NULL, 0));
  return flagcxSuccess;
}
