#include "net.h"
#include "adaptor.h"
#include "adaptor_plugin_load.h"
#include "device.h"
#include "net_transport.h"
#include "proxy.h"
#include "reg_pool.h"
#include "transport.h"

#include <errno.h>
#include <string.h>
#include <string>

namespace {

flagcxResult_t
flagcxGetCollectiveWriteVisibilityPolicy(struct flagcxNetAdaptor *net,
                                         bool *required) {
  if (net == NULL || deviceAdaptor == NULL || required == NULL)
    return flagcxNotSupported;
  const uint32_t requirements =
      flagcxResolveGdrFlushRequirements(deviceAdaptor->gdrFlushRequirements);
  *required = (requirements & FLAGCX_GDR_WRITE_REQUIRES_FLUSH) != 0;
  if (!*required)
    return flagcxSuccess;
  const flagcxResult_t capabilityResult = flagcxValidateGdrFlushCapability(
      requirements, net->gdrFlushCaps, FLAGCX_GDR_WRITE_REQUIRES_FLUSH);
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
  return flagcxNetRegisterBuffer(comm, buffer, size, peerConns, 1,
                                 &op->args.regBufFlag, &op->args.regHandle);
}

flagcxResult_t
flagcxNetProgressProxyOp(struct flagcxProxyConnection *connection,
                         struct flagcxProxyOp *op) {
  if (connection == NULL || op == NULL ||
      connection->transportResources == NULL)
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
  return connection->send
             ? flagcxSendProxyFree(
                   (struct sendNetResources *)connection->transportResources)
             : flagcxRecvProxyFree(
                   (struct recvNetResources *)connection->transportResources);
}

static pthread_mutex_t netLock = PTHREAD_MUTEX_INITIALIZER;
// Use adaptor system for all network types
struct flagcxNetAdaptor *flagcxNetAdaptors[3] = {nullptr, getNetAdaptor(RDMA),
                                                 getNetAdaptor(SOCKET)};
enum flagcxNetState flagcxNetStates[3] = {
    flagcxNetStateInit, flagcxNetStateInit, flagcxNetStateInit};

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
              &args->collTransport
                   .contexts[sequence % args->collTransport.capacity],
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
          FLAGCXCHECK(flagcxGetCollectiveWriteVisibilityPolicy(
              resources->netAdaptor, &flushRequired));
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
                resources->netAdaptor, &flushRequired));
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
    flagcxProxyCleanupResult(
        resources->netAdaptor->deregMr(resources->netSendComm,
                                       resources->mhandles[0]),
        &result);
    resources->mhandles[0] = NULL;
  }
  if (resources->netSendComm != NULL) {
    flagcxProxyCleanupResult(
        resources->netAdaptor->closeSend(resources->netSendComm), &result);
    resources->netSendComm = NULL;
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
