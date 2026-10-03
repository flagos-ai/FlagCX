#include "flagcx.h"
#include "adaptor.h"
#include "adaptor_plugin_load.h"
#include "alloc.h"
#include "bootstrap.h"
#include "check.h"
#include "cluster.h"
#include "comm.h"
#include "cost_model.h"
#include "dev_api_backend.h"
#include "flagcx_hetero.h"
#include "flagcx_kernel_internal.h"
#include "flagcx_net.h"
#include "launch_kernel.h"
#include "mem_alloc_registry.h"
#include "net.h"
#include "onesided.h"
#include "param.h"
#include "proxy.h"
#include "reg_pool.h"
#include "runner.h"
#include "shmem_adaptor.h"
#include "sym_heap.h"
#include "timer.h"
#include "transport.h"
#include "utils.h"
#include <cassert>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

flagcxRegPool globalRegPool;

// -1 keeps the adaptor default, 0 is an expert disable, and 1 forces the
// requirement. These knobs change policy only; they cannot manufacture a
// missing transport or device visibility capability.
FLAGCX_PARAM(GdrReadRequiresFlush, "GDR_READ_REQUIRES_FLUSH", -1);
FLAGCX_PARAM(GdrWriteRequiresFlush, "GDR_WRITE_REQUIRES_FLUSH", -1);

uint32_t flagcxApplyGdrFlushRequirementOverrides(uint32_t defaults,
                                                 int64_t readOverride,
                                                 int64_t writeOverride) {
  uint32_t requirements = defaults;
  if (readOverride == 0)
    requirements &= ~FLAGCX_GDR_READ_REQUIRES_FLUSH;
  else if (readOverride == 1)
    requirements |= FLAGCX_GDR_READ_REQUIRES_FLUSH;

  if (writeOverride == 0)
    requirements &= ~FLAGCX_GDR_WRITE_REQUIRES_FLUSH;
  else if (writeOverride == 1)
    requirements |= FLAGCX_GDR_WRITE_REQUIRES_FLUSH;
  return requirements;
}

uint32_t flagcxResolveGdrFlushRequirements(uint32_t defaults) {
  return flagcxApplyGdrFlushRequirementOverrides(
      defaults, flagcxParamGdrReadRequiresFlush(),
      flagcxParamGdrWriteRequiresFlush());
}

flagcxResult_t flagcxValidateGdrFlushCapability(uint32_t requirements,
                                                uint32_t capabilities,
                                                uint32_t direction) {
  // Requirement and provider-capability bits intentionally use the same READ
  // and WRITE positions, but keep the two enums separate: needing a visibility
  // boundary and being able to implement it are different contracts.
  static_assert(static_cast<uint32_t>(FLAGCX_GDR_READ_REQUIRES_FLUSH) ==
                    static_cast<uint32_t>(FLAGCX_NET_GDR_FLUSH_READ),
                "READ requirement and capability bits must stay aligned");
  static_assert(static_cast<uint32_t>(FLAGCX_GDR_WRITE_REQUIRES_FLUSH) ==
                    static_cast<uint32_t>(FLAGCX_NET_GDR_FLUSH_WRITE),
                "WRITE requirement and capability bits must stay aligned");
  const uint32_t knownDirections =
      FLAGCX_GDR_READ_REQUIRES_FLUSH | FLAGCX_GDR_WRITE_REQUIRES_FLUSH;
  if (direction == 0 || (direction & ~knownDirections) != 0)
    return flagcxInvalidArgument;
  return (requirements & direction & ~capabilities) == 0 ? flagcxSuccess
                                                         : flagcxNotSupported;
}

size_t getFlagcxDataTypeSize(flagcxDataType_t dtype) {
  switch (dtype) {
    // case flagcxInt8:
    case flagcxChar:
      return sizeof(char); // 1 byte
    case flagcxUint8:
      return sizeof(unsigned char); // 1 byte
    // case flagcxInt32:
    case flagcxInt:
      return sizeof(int); // 4 bytes
    case flagcxUint32:
      return sizeof(unsigned int); // 4 bytes
    case flagcxInt64:
      return sizeof(long long); // 8 bytes
    case flagcxUint64:
      return sizeof(unsigned long long); // 8 bytes
    // case flagcxFloat16:
    case flagcxHalf:
      return 2; // Half precision float is 2 bytes
    // case flagcxFloat32:
    case flagcxFloat:
      return sizeof(float); // 4 bytes
    // case flagcxFloat64:
    case flagcxDouble:
      return sizeof(double); // 8 bytes
    case flagcxBfloat16:
      return 2; // BFloat16 is typically 2 bytes
    default:
      fprintf(stderr, "Unknown flagcx data type\n");
      return 0;
  }
}

// Wrapper function for deviceMemcpy without the usage of invalid args
flagcxResult_t wrapper_deviceMemcpy(void *dst, void *src, size_t size,
                                    flagcxMemcpyType_t type,
                                    flagcxStream_t stream) {
  return deviceAdaptor->deviceMemcpy(dst, src, size, type, stream, NULL);
}

static struct flagcxDeviceHandle globalDeviceHandle {
  // Basic functions
  deviceAdaptor->deviceSynchronize, wrapper_deviceMemcpy,
      deviceAdaptor->deviceMemset, deviceAdaptor->deviceMalloc,
      deviceAdaptor->deviceFree, deviceAdaptor->setDevice,
      deviceAdaptor->getDevice, deviceAdaptor->getDeviceCount,
      deviceAdaptor->getVendor, deviceAdaptor->hostGetDevicePointer,
      // Stream functions
      deviceAdaptor->streamCreate, deviceAdaptor->streamDestroy,
      deviceAdaptor->streamCopy, deviceAdaptor->streamFree,
      deviceAdaptor->streamSynchronize, deviceAdaptor->streamQuery,
      deviceAdaptor->streamWaitEvent,
      // Event functions
      deviceAdaptor->eventCreate, deviceAdaptor->eventDestroy,
      deviceAdaptor->eventRecord, deviceAdaptor->eventSynchronize,
      deviceAdaptor->eventQuery, deviceAdaptor->eventElapsedTime,
      // IpcMemHandle functions
      deviceAdaptor->ipcMemHandleCreate, deviceAdaptor->ipcMemHandleGet,
      deviceAdaptor->ipcMemHandleOpen, deviceAdaptor->ipcMemHandleClose,
      deviceAdaptor->ipcMemHandleFree,
};

void flagcxRebuildGlobalDeviceHandle() {
  // Basic functions
  globalDeviceHandle.deviceSynchronize = deviceAdaptor->deviceSynchronize;
  globalDeviceHandle.deviceMemcpy = wrapper_deviceMemcpy;
  globalDeviceHandle.deviceMemset = deviceAdaptor->deviceMemset;
  globalDeviceHandle.deviceMalloc = deviceAdaptor->deviceMalloc;
  globalDeviceHandle.deviceFree = deviceAdaptor->deviceFree;
  globalDeviceHandle.setDevice = deviceAdaptor->setDevice;
  globalDeviceHandle.getDevice = deviceAdaptor->getDevice;
  globalDeviceHandle.getDeviceCount = deviceAdaptor->getDeviceCount;
  globalDeviceHandle.getVendor = deviceAdaptor->getVendor;
  globalDeviceHandle.hostGetDevicePointer = deviceAdaptor->hostGetDevicePointer;
  // Stream functions
  globalDeviceHandle.streamCreate = deviceAdaptor->streamCreate;
  globalDeviceHandle.streamDestroy = deviceAdaptor->streamDestroy;
  globalDeviceHandle.streamCopy = deviceAdaptor->streamCopy;
  globalDeviceHandle.streamFree = deviceAdaptor->streamFree;
  globalDeviceHandle.streamSynchronize = deviceAdaptor->streamSynchronize;
  globalDeviceHandle.streamQuery = deviceAdaptor->streamQuery;
  globalDeviceHandle.streamWaitEvent = deviceAdaptor->streamWaitEvent;
  // Event functions
  globalDeviceHandle.eventCreate = deviceAdaptor->eventCreate;
  globalDeviceHandle.eventDestroy = deviceAdaptor->eventDestroy;
  globalDeviceHandle.eventRecord = deviceAdaptor->eventRecord;
  globalDeviceHandle.eventSynchronize = deviceAdaptor->eventSynchronize;
  globalDeviceHandle.eventQuery = deviceAdaptor->eventQuery;
  globalDeviceHandle.eventElapsedTime = deviceAdaptor->eventElapsedTime;
  // IpcMemHandle functions
  globalDeviceHandle.ipcMemHandleCreate = deviceAdaptor->ipcMemHandleCreate;
  globalDeviceHandle.ipcMemHandleGet = deviceAdaptor->ipcMemHandleGet;
  globalDeviceHandle.ipcMemHandleOpen = deviceAdaptor->ipcMemHandleOpen;
  globalDeviceHandle.ipcMemHandleClose = deviceAdaptor->ipcMemHandleClose;
  globalDeviceHandle.ipcMemHandleFree = deviceAdaptor->ipcMemHandleFree;
}

flagcxResult_t flagcxEnsureCommReady(flagcxComm_t comm) {
  if (comm == NULL) {
    return flagcxInternalError;
  }
  if (comm->commType != flagcxCommunicatorHybrid &&
      comm->commType != flagcxCommunicatorHomo) {
    return flagcxInternalError;
  }
  return flagcxSuccess;
}

bool useHomoComm(flagcxComm_t comm) {
  return comm->commType == flagcxCommunicatorHomo;
}

bool useHostComm() {
  const char *useHostComm = flagcxGetEnv("FLAGCX_USE_HOST_COMM");
  if (useHostComm) {
    return std::stoi(useHostComm) == 1;
  }
  return false;
}

bool useHeteroComm() {
  const char *useHeteroComm = flagcxGetEnv("FLAGCX_USE_HETERO_COMM");
  if (useHeteroComm) {
    return std::stoi(useHeteroComm) == 1;
  }
  return false;
}

flagcxResult_t flagcxDeviceHandleInit(flagcxDeviceHandle_t *devHandle) {
  if (devHandle == NULL) {
    WARN("flagcxDeviceHandleInit: devHandle is NULL");
    return flagcxInvalidArgument;
  }
  flagcxResult_t res = flagcxSuccess;
  flagcxDeviceAdaptorPluginInit();
  flagcxCCLAdaptorPluginInit();
  (*devHandle) = NULL;
  FLAGCXCHECKGOTO(flagcxCalloc(devHandle, 1), res, fail);
  **devHandle = globalDeviceHandle;
  return flagcxSuccess;

fail:
  if (*devHandle) {
    free(*devHandle);
    *devHandle = NULL;
  }
  flagcxCCLAdaptorPluginFinalize();
  flagcxDeviceAdaptorPluginFinalize();
  return res;
}

flagcxResult_t flagcxDeviceHandleFree(flagcxDeviceHandle_t devHandle) {
  if (devHandle == NULL)
    return flagcxSuccess;
  free(devHandle);
  flagcxCCLAdaptorPluginFinalize();
  flagcxDeviceAdaptorPluginFinalize();
  return flagcxSuccess;
}

flagcxResult_t flagcxHandleInit(flagcxHandlerGroup_t *handler) {
  flagcxResult_t res = flagcxSuccess;
  (*handler) = NULL;
  FLAGCXCHECKGOTO(flagcxCalloc(handler, 1), res, fail);
  FLAGCXCHECKGOTO(flagcxDeviceHandleInit(&(*handler)->devHandle), res, fail);
  return flagcxSuccess;

fail:
  if (*handler) {
    free(*handler);
    *handler = NULL;
  }
  return res;
}

flagcxResult_t flagcxHandleFree(flagcxHandlerGroup_t handler) {
  if (handler != NULL) {
    flagcxDeviceHandleFree(handler->devHandle);
    handler->devHandle = NULL;
    handler->uniqueId = NULL;
    handler->comm = NULL;
    free(handler);
  }
  return flagcxSuccess;
}

static flagcxResult_t flagcxMemFreeByBackend(void *ptr,
                                             flagcxMemAllocBackend_t backend) {
  switch (backend) {
    case flagcxMemAllocBackendNative:
      if (deviceAdaptor == nullptr || deviceAdaptor->gdrMemFree == nullptr) {
        WARN("flagcxMemFree: native allocator is not available");
        return flagcxInternalError;
      }
      return deviceAdaptor->gdrMemFree(ptr, nullptr);
    case flagcxMemAllocBackendCcl:
      if (cclAdaptors[flagcxCCLAdaptorDevice] == nullptr ||
          cclAdaptors[flagcxCCLAdaptorDevice]->memFree == nullptr) {
        WARN("flagcxMemFree: CCL allocator is not available");
        return flagcxInternalError;
      }
      return cclAdaptors[flagcxCCLAdaptorDevice]->memFree(ptr);
    case flagcxMemAllocBackendShmem:
      if (shmemAdaptor == nullptr || shmemAdaptor->free == nullptr) {
        WARN("flagcxMemFree: SHMEM allocator is not available");
        return flagcxInternalError;
      }
      return shmemAdaptor->free(ptr);
    default:
      WARN("flagcxMemFree: unknown allocation backend %d", (int)backend);
      return flagcxInvalidArgument;
  }
}

flagcxResult_t flagcxMemAlloc(void **ptr, size_t size,
                              flagcxMemAllocator_t allocator) {
  if (ptr == NULL || size == 0) {
    WARN("Invalid ptr(NULL) or size(0) for allocation.");
    return flagcxInvalidArgument;
  }
  *ptr = nullptr;

  flagcxResult_t res = flagcxSuccess;
  flagcxMemAllocBackend_t backend = flagcxMemAllocBackendNative;
  switch (allocator) {
    case flagcxMemCCL:
      if (useHeteroComm()) {
        backend = flagcxMemAllocBackendNative;
        if (deviceAdaptor == nullptr || deviceAdaptor->gdrMemAlloc == nullptr) {
          WARN("flagcxMemAlloc: native allocator is not available");
          return flagcxInternalError;
        }
        res = deviceAdaptor->gdrMemAlloc(ptr, size, nullptr);
      } else {
        backend = flagcxMemAllocBackendCcl;
        if (cclAdaptors[flagcxCCLAdaptorDevice] == nullptr ||
            cclAdaptors[flagcxCCLAdaptorDevice]->memAlloc == nullptr) {
          WARN("flagcxMemAlloc: CCL allocator is not available");
          return flagcxInternalError;
        }
        res = cclAdaptors[flagcxCCLAdaptorDevice]->memAlloc(ptr, size);
      }
      break;
    case flagcxMemSHMEM:
      backend = flagcxMemAllocBackendShmem;
      if (shmemAdaptor == nullptr || shmemAdaptor->malloc == nullptr) {
        WARN("flagcxMemAlloc: SHMEM allocator is not available");
        return flagcxInternalError;
      }
      res = shmemAdaptor->malloc(ptr, size);
      break;
    default:
      WARN("flagcxMemAlloc: unknown allocator %d", (int)allocator);
      return flagcxInvalidArgument;
  }

  if (res != flagcxSuccess) {
    if (*ptr != nullptr) {
      flagcxResult_t freeRes = flagcxMemFreeByBackend(*ptr, backend);
      if (freeRes != flagcxSuccess)
        WARN("flagcxMemAlloc: failed to roll back partial allocation");
      *ptr = nullptr;
    }
    return res;
  }
  if (*ptr == nullptr) {
    WARN("flagcxMemAlloc: backend %d returned a null pointer", (int)backend);
    return flagcxUnhandledDeviceError;
  }

  // Latest adaptors explicitly advertise whether their native allocator uses
  // VMM. A frozen v1 plugin cannot expose that field, so preserve its previous
  // environment-driven routing through latest-only loader metadata. Local
  // symmetric flat-map support is a separate capability and is not evidence
  // about the allocation itself.
  bool isVmm = backend == flagcxMemAllocBackendNative &&
               flagcxDeviceAdaptorNativeAllocIsVmm(deviceAdaptor,
                                                   flagcxParamVmmEnable());
  flagcxMemAllocationInfo info{*ptr, size, allocator, backend, isVmm};
  res = globalMemAllocRegistry.insert(info);
  if (res != flagcxSuccess) {
    flagcxResult_t freeRes = flagcxMemFreeByBackend(*ptr, backend);
    if (freeRes != flagcxSuccess)
      WARN("flagcxMemAlloc: failed to roll back untracked allocation");
    *ptr = nullptr;
    return res;
  }

  return flagcxSuccess;
}

flagcxResult_t flagcxMemFree(void *ptr, flagcxMemAllocator_t allocator) {
  if (ptr == NULL) {
    WARN("Invalid pointer(=NULL) for de-allocation.");
    return flagcxSuccess;
  }

  flagcxMemAllocationInfo info;
  flagcxResult_t res = globalMemAllocRegistry.findExact(ptr, &info);
  if (res != flagcxSuccess) {
    WARN("flagcxMemFree: pointer was not allocated by flagcxMemAlloc");
    return flagcxInvalidUsage;
  }
  if (allocator != info.allocator) {
    WARN("flagcxMemFree: allocator mismatch (requested %d, recorded %d)",
         (int)allocator, (int)info.allocator);
    return flagcxInvalidUsage;
  }

  // Remove ownership before calling the backend so concurrent frees cannot
  // both release the same allocation. Restore it if the backend rejects the
  // free and still owns the memory.
  FLAGCXCHECK(globalMemAllocRegistry.erase(ptr));
  res = flagcxMemFreeByBackend(ptr, info.backend);
  if (res != flagcxSuccess) {
    flagcxResult_t restoreRes = globalMemAllocRegistry.insert(info);
    if (restoreRes != flagcxSuccess)
      WARN("flagcxMemFree: failed to restore allocation provenance");
    return res;
  }
  INFO(FLAGCX_REG, "flagcxMemFree: backend %d memory deallocated",
       (int)info.backend);
  return flagcxSuccess;
}

// Build full-mesh IB connections (including self-loopback) for one-sided ops.
// Called once on the first flagcxOneSideRegister invocation; stored in
// handle[0]. Pattern aligned with NCCL GIN gin.cc:146-158.
// Build one full-mesh of IB connections for a single context.
// On success, stores sendComms[peer] and recvComms[peer] arrays into
// *outSend/*outRecv.
static flagcxResult_t
flagcxOneSideConvergeStatus(struct bootstrapState *bootstrap, int rank,
                            int nranks, flagcxResult_t localStatus) {
  if (bootstrap == NULL || rank < 0 || rank >= nranks || nranks <= 0)
    return flagcxInvalidArgument;
  constexpr int root = 0;
  constexpr int gatherTag = 0x5970;
  constexpr int broadcastTag = 0x5971;
  int local = static_cast<int>(localStatus);
  int common = local;
  if (rank == root) {
    for (int peer = 1; peer < nranks; peer++) {
      int peerStatus = static_cast<int>(flagcxSuccess);
      FLAGCXCHECK(bootstrapRecv(bootstrap, peer, gatherTag, &peerStatus,
                                sizeof(peerStatus)));
      if (common == flagcxSuccess && peerStatus != flagcxSuccess)
        common = peerStatus;
    }
    for (int peer = 1; peer < nranks; peer++)
      FLAGCXCHECK(bootstrapSend(bootstrap, peer, broadcastTag, &common,
                                sizeof(common)));
  } else {
    FLAGCXCHECK(
        bootstrapSend(bootstrap, root, gatherTag, &local, sizeof(local)));
    FLAGCXCHECK(
        bootstrapRecv(bootstrap, root, broadcastTag, &common, sizeof(common)));
  }
  return static_cast<flagcxResult_t>(common);
}

static flagcxResult_t
flagcxOneSideBuildOneContext(struct flagcxHeteroComm *heteroComm, int nranks,
                             int rank, int contextIdx, void ***outSend,
                             void ***outRecv) {
  flagcxResult_t res = flagcxSuccess;
  void *listenComm = NULL;
  flagcxNetHandle_t *allHandles = NULL;
  void **sendComms = NULL;
  void **recvComms = NULL;

  res = flagcxCalloc(&sendComms, nranks);
  if (res == flagcxSuccess)
    res = flagcxCalloc(&recvComms, nranks);

  flagcxNetHandle_t myListenHandle = {};
  if (res == flagcxSuccess)
    res = heteroComm->netAdaptor->listen(heteroComm->netDev,
                                         (void *)myListenHandle, &listenComm);
  if (res == flagcxSuccess)
    res = flagcxCalloc(&allHandles, nranks);

  // No rank enters the listen-handle collective until every rank completed
  // local allocation/listen preparation successfully.
  {
    flagcxResult_t common =
        flagcxOneSideConvergeStatus(heteroComm->bootstrap, rank, nranks, res);
    if (common != flagcxSuccess) {
      res = common;
      goto fail_handles;
    }
  }

  {
    memcpy(&allHandles[rank], &myListenHandle, sizeof(flagcxNetHandle_t));
    res = bootstrapCollAllGather(heteroComm->bootstrap, (void *)allHandles,
                                 sizeof(flagcxNetHandle_t));
    res = flagcxOneSideConvergeStatus(heteroComm->bootstrap, rank, nranks, res);
    if (res != flagcxSuccess)
      goto fail_handles;

    // Deadlock-free full-mesh connection (NCCL GIN pattern)
    for (int i = 0; i < nranks; i++) {
      int connectPeer = (rank + i) % nranks;
      int acceptPeer = (rank - i + nranks) % nranks;

      void *sc = NULL, *rc = NULL;
      while (sc == NULL || rc == NULL) {
        if (sc == NULL) {
          res = heteroComm->netAdaptor->connect(
              heteroComm->netDev, (void *)&allHandles[connectPeer], &sc);
          if (res != flagcxSuccess && res != flagcxInProgress) {
            INFO(
                FLAGCX_REG,
                "flagcxOneSideBuildFullMesh: ctx %d connect to peer %d failed, "
                "res=%d",
                contextIdx, connectPeer, res);
            goto fail_handles;
          }
        }
        if (rc == NULL) {
          res = heteroComm->netAdaptor->accept(listenComm, &rc);
          if (res != flagcxSuccess && res != flagcxInProgress) {
            INFO(FLAGCX_REG,
                 "flagcxOneSideBuildFullMesh: ctx %d accept from peer %d "
                 "failed, "
                 "res=%d",
                 contextIdx, acceptPeer, res);
            goto fail_handles;
          }
        }
        if (sc == NULL || rc == NULL)
          sched_yield();
      }
      sendComms[connectPeer] = sc;
      recvComms[acceptPeer] = rc;
    }

    free(allHandles);
    heteroComm->netAdaptor->closeListen(listenComm);
  }

  INFO(FLAGCX_REG, "flagcxOneSideBuildFullMesh: rank %d ctx %d, %d connections",
       rank, contextIdx, nranks);
  *outSend = sendComms;
  *outRecv = recvComms;
  return flagcxSuccess;

fail_handles:
  for (int i = 0; i < nranks; i++) {
    if (sendComms != NULL && sendComms[i])
      heteroComm->netAdaptor->closeSend(sendComms[i]);
    if (recvComms != NULL && recvComms[i])
      heteroComm->netAdaptor->closeRecv(recvComms[i]);
  }
  free(allHandles);
  if (listenComm != NULL)
    heteroComm->netAdaptor->closeListen(listenComm);
  free(sendComms);
  free(recvComms);
  *outSend = NULL;
  *outRecv = NULL;
  return res;
}

static flagcxResult_t
flagcxOneSideBuildFullMesh(struct flagcxHeteroComm *heteroComm,
                           struct flagcxOneSideHandleInfo *info) {
  int nranks = heteroComm->nRanks;
  int rank = heteroComm->rank;
  flagcxResult_t res = flagcxSuccess;

  // Determine number of contexts: 1 (RMA proxy) + N (kernel proxy threads).
  // contextCount defaults to 0; only set by flagcxProxyInit when
  // COMPILE_KERNEL_HOST. Ensure proxy is initialized so contextCount is
  // reliable.
  int nContexts = 1;
  if (heteroComm->proxyState != NULL &&
      heteroComm->proxyState->initialized == 0) {
    res = flagcxProxyInit(heteroComm);
  }
  if (res == flagcxSuccess && heteroComm->proxyState != NULL)
    nContexts = 1 + heteroComm->proxyState->kernelState.contextCount;
  info->nContexts = nContexts;
  info->nRanks = nranks;

  if (res == flagcxSuccess)
    res = flagcxCalloc(&info->contextSendComms, nContexts);
  if (res == flagcxSuccess)
    res = flagcxCalloc(&info->contextRecvComms, nContexts);
  res = flagcxOneSideConvergeStatus(heteroComm->bootstrap, rank, nranks, res);
  if (res != flagcxSuccess)
    goto fail_partial;

  // Build one full-mesh per context (collective: all ranks participate per
  // round)
  for (int ctx = 0; ctx < nContexts; ctx++) {
    flagcxResult_t localResult = flagcxOneSideBuildOneContext(
        heteroComm, nranks, rank, ctx, &info->contextSendComms[ctx],
        &info->contextRecvComms[ctx]);
    // Converge connection outcomes before any rank advances to the next
    // context or to MR metadata exchange.
    res = flagcxOneSideConvergeStatus(heteroComm->bootstrap, rank, nranks,
                                      localResult);
    if (res != flagcxSuccess)
      goto fail_partial;
  }

  // Legacy aliases: context 0 = RMA proxy
  info->fullSendComms = info->contextSendComms[0];
  info->fullRecvComms = info->contextRecvComms[0];
  info->ownsConnections = 1;

  INFO(FLAGCX_REG,
       "flagcxOneSideBuildFullMesh: rank %d, %d contexts × %d peers "
       "(including self-loopback)",
       rank, nContexts, nranks);
  return flagcxSuccess;

fail_partial:
  // Cleanup contexts that were successfully created
  for (int ctx = 0; ctx < nContexts; ctx++) {
    if (info->contextSendComms != NULL && info->contextSendComms[ctx] != NULL) {
      for (int i = 0; i < nranks; i++) {
        if (info->contextSendComms[ctx][i])
          heteroComm->netAdaptor->closeSend(info->contextSendComms[ctx][i]);
        if (info->contextRecvComms != NULL &&
            info->contextRecvComms[ctx] != NULL &&
            info->contextRecvComms[ctx][i])
          heteroComm->netAdaptor->closeRecv(info->contextRecvComms[ctx][i]);
      }
      free(info->contextSendComms[ctx]);
      if (info->contextRecvComms != NULL)
        free(info->contextRecvComms[ctx]);
    }
  }
  free(info->contextSendComms);
  free(info->contextRecvComms);
  info->contextSendComms = NULL;
  info->contextRecvComms = NULL;
  info->fullSendComms = NULL;
  info->fullRecvComms = NULL;
  info->nRanks = 0;
  info->nContexts = 0;
  return res;
}

// Ensure full-mesh one-sided connections exist for this heteroComm.
// If no data handle has been registered yet, lazily build a connection-only
// handle at slot 0 so that signal/staging registration can proceed without
// requiring a prior flagcxCommRegister call.
// NOTE: This is a collective operation — all ranks must call it together.
static flagcxResult_t
flagcxOneSideEnsureFullMesh(struct flagcxHeteroComm *heteroComm) {
  if (heteroComm == NULL || heteroComm->netAdaptor == NULL ||
      heteroComm->netAdaptor->regMr == NULL)
    return flagcxNotSupported;

  if (heteroComm->bootstrap == NULL)
    return flagcxNotSupported;

  flagcxResult_t res = flagcxSuccess;
  struct flagcxOneSideHandleInfo *info = NULL;
  struct flagcxOneSideHandleInfo **publishHandles = heteroComm->oneSideHandles;
  int publishCapacity = heteroComm->oneSideHandleCapacity;
  bool replaceHandleArray = false;

  res = flagcxOneSideRetryPendingCleanup(heteroComm);
  res = flagcxOneSideConvergeStatus(heteroComm->bootstrap, heteroComm->rank,
                                    heteroComm->nRanks, res);
  if (res != flagcxSuccess)
    return res;

  // Only take the existing-mesh fast path after every rank has retried and
  // converged cleanup retained by an earlier failed registration rollback.
  if (heteroComm->oneSideHandleCount > 0 &&
      heteroComm->oneSideHandles[0] != NULL &&
      heteroComm->oneSideHandles[0]->fullRecvComms != NULL) {
    return flagcxSuccess;
  }

  // Prepare capacity without publishing it until the full mesh converges.
  if (heteroComm->oneSideHandleCount >= heteroComm->oneSideHandleCapacity) {
    int newCap = heteroComm->oneSideHandleCapacity == 0
                     ? 4
                     : heteroComm->oneSideHandleCapacity * 2;
    publishHandles = (struct flagcxOneSideHandleInfo **)calloc(
        newCap, sizeof(struct flagcxOneSideHandleInfo *));
    if (publishHandles == NULL)
      res = flagcxSystemError;
    else {
      for (int i = 0; i < heteroComm->oneSideHandleCount; i++)
        publishHandles[i] = heteroComm->oneSideHandles[i];
      publishCapacity = newCap;
      replaceHandleArray = true;
    }
  }

  res = flagcxOneSideConvergeStatus(heteroComm->bootstrap, heteroComm->rank,
                                    heteroComm->nRanks, res);
  if (res != flagcxSuccess)
    goto fail_info;
  res = flagcxCalloc(&info, 1);
  res = flagcxOneSideConvergeStatus(heteroComm->bootstrap, heteroComm->rank,
                                    heteroComm->nRanks, res);
  if (res != flagcxSuccess)
    goto fail_info;
  FLAGCXCHECKGOTO(flagcxOneSideBuildFullMesh(heteroComm, info), res, fail_info);

  // Store at current slot (should be slot 0 if this is truly the first)
  {
    int slot = heteroComm->oneSideHandleCount;
    if (replaceHandleArray) {
      free(heteroComm->oneSideHandles);
      heteroComm->oneSideHandles = publishHandles;
      heteroComm->oneSideHandleCapacity = publishCapacity;
      replaceHandleArray = false;
    }
    heteroComm->oneSideHandles[slot] = info;
    heteroComm->oneSideHandleCount = slot + 1;

    // Publish sendComms to RMA proxy so its progress thread can use them
    if (slot == 0 && info->fullSendComms != NULL) {
      flagcxHeteroRmaProxyPublishSendComms(heteroComm, info->fullSendComms);
    }
  }

  INFO(FLAGCX_REG,
       "flagcxOneSideEnsureFullMesh: lazily built full-mesh connections "
       "(slot %d)",
       heteroComm->oneSideHandleCount - 1);
  return flagcxSuccess;

fail_info:
  free(info);
  if (replaceHandleArray)
    free(publishHandles);
  return res;
}

static flagcxResult_t flagcxOneSideGetMrInfo(struct flagcxNetAdaptor *net,
                                             void *mrHandle,
                                             struct flagcxNetMrInfo *mrInfo) {
  if (net == NULL || mrHandle == NULL || mrInfo == NULL)
    return flagcxInvalidArgument;
  if (net->getMrInfo == NULL)
    return flagcxNotSupported;

  memset(mrInfo, 0, sizeof(*mrInfo));
  FLAGCXCHECK(net->getMrInfo(mrHandle, mrInfo));
  if (mrInfo->nKeys == 0 || mrInfo->nKeys > FLAGCX_NET_MAX_MR_KEYS)
    return flagcxInternalError;
  return flagcxSuccess;
}

static flagcxResult_t
flagcxOneSideExchangeMrInfo(struct bootstrapState *bootstrap, int rank,
                            int nranks, void *buffer, size_t size,
                            const struct flagcxNetMrInfo *localMrInfo,
                            struct flagcxOneSideHandleInfo *info) {
  if (bootstrap == NULL || info == NULL || localMrInfo == NULL || rank < 0 ||
      rank >= nranks)
    return flagcxInvalidArgument;

  flagcxResult_t res = flagcxSuccess;
  res = flagcxCalloc(&info->baseVas, nranks);
  if (res == flagcxSuccess)
    res = flagcxCalloc(&info->regionSizes, nranks);
  if (res == flagcxSuccess)
    res = flagcxCalloc(&info->mrInfos, nranks);
  res = flagcxOneSideConvergeStatus(bootstrap, rank, nranks, res);
  if (res != flagcxSuccess)
    goto fail;

  info->baseVas[rank] = (uintptr_t)buffer;
  info->regionSizes[rank] = size;
  info->mrInfos[rank] = *localMrInfo;
  info->nRanks = nranks;

  res = bootstrapCollAllGather(bootstrap, info->baseVas, sizeof(uintptr_t));
  res = flagcxOneSideConvergeStatus(bootstrap, rank, nranks, res);
  if (res != flagcxSuccess)
    goto fail;
  res = bootstrapCollAllGather(bootstrap, info->regionSizes, sizeof(size_t));
  res = flagcxOneSideConvergeStatus(bootstrap, rank, nranks, res);
  if (res != flagcxSuccess)
    goto fail;
  res = bootstrapCollAllGather(bootstrap, info->mrInfos,
                               sizeof(struct flagcxNetMrInfo));
  res = flagcxOneSideConvergeStatus(bootstrap, rank, nranks, res);
  if (res != flagcxSuccess)
    goto fail;
  return flagcxSuccess;

fail:
  free(info->mrInfos);
  free(info->regionSizes);
  free(info->baseVas);
  info->mrInfos = NULL;
  info->regionSizes = NULL;
  info->baseVas = NULL;
  return res;
}

static void flagcxOneSideFreeMrInfo(struct flagcxOneSideHandleInfo *info) {
  if (info == NULL)
    return;
  free(info->mrInfos);
  free(info->regionSizes);
  free(info->baseVas);
  info->mrInfos = NULL;
  info->regionSizes = NULL;
  info->baseVas = NULL;
}

static void
flagcxOneSideCloseConnections(struct flagcxHeteroComm *heteroComm,
                              struct flagcxOneSideHandleInfo *info) {
  if (heteroComm == NULL || info == NULL || !info->ownsConnections)
    return;
  if (heteroComm->netAdaptor != NULL && info->contextSendComms != NULL) {
    for (int ctx = 0; ctx < info->nContexts; ctx++) {
      for (int peer = 0; peer < info->nRanks; peer++) {
        if (info->contextSendComms[ctx] != NULL &&
            info->contextSendComms[ctx][peer] != NULL)
          heteroComm->netAdaptor->closeSend(info->contextSendComms[ctx][peer]);
        if (info->contextRecvComms != NULL &&
            info->contextRecvComms[ctx] != NULL &&
            info->contextRecvComms[ctx][peer] != NULL)
          heteroComm->netAdaptor->closeRecv(info->contextRecvComms[ctx][peer]);
      }
      free(info->contextSendComms[ctx]);
      if (info->contextRecvComms != NULL)
        free(info->contextRecvComms[ctx]);
    }
  }
  free(info->contextSendComms);
  free(info->contextRecvComms);
  info->contextSendComms = NULL;
  info->contextRecvComms = NULL;
  info->fullSendComms = NULL;
  info->fullRecvComms = NULL;
  info->ownsConnections = 0;
}

// Release one handle in dependency order. On deregMr failure absolutely no
// upper-level ownership is discarded, so the same object can be retried.
static flagcxResult_t
flagcxOneSideCleanupHandle(struct flagcxHeteroComm *heteroComm,
                           struct flagcxOneSideHandleInfo *info,
                           bool closeConnections) {
  if (info == NULL)
    return flagcxSuccess;
  if (info->localMrHandle != NULL) {
    if (heteroComm == NULL || heteroComm->netAdaptor == NULL ||
        heteroComm->netAdaptor->deregMr == NULL || info->localRecvComm == NULL)
      return flagcxInternalError;
    flagcxResult_t result = heteroComm->netAdaptor->deregMr(
        info->localRecvComm, info->localMrHandle);
    if (result != flagcxSuccess)
      return result;
    info->localMrHandle = NULL;
    info->ownsLocalMr = 0;
  }
  flagcxOneSideFreeMrInfo(info);
  if (closeConnections)
    flagcxOneSideCloseConnections(heteroComm, info);
  return flagcxSuccess;
}

static void flagcxOneSideRetainCleanup(struct flagcxHeteroComm *heteroComm,
                                       struct flagcxOneSideHandleInfo *info) {
  info->cleanupNext = heteroComm->pendingOneSideCleanup;
  heteroComm->pendingOneSideCleanup = info;
}

flagcxResult_t
flagcxOneSideRetryPendingCleanup(struct flagcxHeteroComm *heteroComm) {
  while (heteroComm->pendingOneSideCleanup != NULL) {
    struct flagcxOneSideHandleInfo *info = heteroComm->pendingOneSideCleanup;
    flagcxResult_t result = flagcxOneSideCleanupHandle(heteroComm, info, true);
    if (result != flagcxSuccess)
      return result;
    heteroComm->pendingOneSideCleanup = info->cleanupNext;
    free(info);
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxOneSideParseVmmMrMode(const char *value,
                                           flagcxVmmMrMode_t *mode) {
  if (mode == nullptr)
    return flagcxInvalidArgument;
  if (value == nullptr || value[0] == '\0' || strcasecmp(value, "auto") == 0) {
    *mode = flagcxVmmMrModeAuto;
    return flagcxSuccess;
  }
  if (strcasecmp(value, "dmabuf") == 0) {
    *mode = flagcxVmmMrModeDmaBuf;
    return flagcxSuccess;
  }
  if (strcasecmp(value, "va") == 0) {
    *mode = flagcxVmmMrModeVa;
    return flagcxSuccess;
  }
  WARN("Invalid FLAGCX_VMM_MR_MODE=%s; expected auto, dmabuf, or va", value);
  return flagcxInvalidArgument;
}

flagcxVmmMrRoute_t flagcxOneSideSelectVmmMrRoute(uint32_t deviceCaps,
                                                 int netPtrSupport,
                                                 bool dmaBufExportSupported,
                                                 bool hasDmaBufRegistration,
                                                 bool hasVaRegistration,
                                                 flagcxVmmMrMode_t mode) {
  const bool allowDmaBuf = mode != flagcxVmmMrModeVa;
  const bool allowVa = mode != flagcxVmmMrModeDmaBuf;
  if (allowDmaBuf && (deviceCaps & FLAGCX_VMM_MR_CAP_DMABUF) &&
      dmaBufExportSupported && (netPtrSupport & FLAGCX_PTR_DMABUF) &&
      hasDmaBufRegistration)
    return FLAGCX_VMM_MR_ROUTE_DMABUF;
  if (allowVa && (deviceCaps & FLAGCX_VMM_MR_CAP_VA) &&
      (netPtrSupport & FLAGCX_PTR_CUDA) && hasVaRegistration)
    return FLAGCX_VMM_MR_ROUTE_VA;
  return FLAGCX_VMM_MR_ROUTE_NONE;
}

flagcxResult_t flagcxOneSideSelectCommonPublishSlot(const uint8_t *occupancy,
                                                    int nRanks, int nSlots,
                                                    int *slot) {
  if (nRanks <= 0 || nSlots < 0 || slot == nullptr ||
      (nSlots > 0 && occupancy == nullptr))
    return flagcxInvalidArgument;
  if (nSlots == 0) {
    *slot = 0;
    return flagcxSuccess;
  }

  const bool slotZeroOccupied = occupancy[0] != 0;
  for (int rank = 1; rank < nRanks; rank++) {
    if ((occupancy[rank * nSlots] != 0) != slotZeroOccupied)
      return flagcxInvalidUsage;
  }
  if (!slotZeroOccupied) {
    for (int rank = 0; rank < nRanks; rank++) {
      for (int i = 1; i < nSlots; i++) {
        if (occupancy[rank * nSlots + i] != 0)
          return flagcxInvalidUsage;
      }
    }
    *slot = 0;
    return flagcxSuccess;
  }

  for (int i = 1; i < nSlots; i++) {
    bool freeEverywhere = true;
    for (int rank = 0; rank < nRanks; rank++)
      freeEverywhere = freeEverywhere && occupancy[rank * nSlots + i] == 0;
    if (freeEverywhere) {
      *slot = i;
      return flagcxSuccess;
    }
  }
  *slot = nSlots;
  return flagcxSuccess;
}

bool flagcxOneSideRegistryRangeIsVmm(const void *buff, size_t size) {
  flagcxMemAllocationInfo allocation = {};
  return globalMemAllocRegistry.findRange(buff, size, &allocation) ==
             flagcxSuccess &&
         allocation.isVmm;
}

static flagcxResult_t flagcxOneSideValidateRegisteredMr(flagcxResult_t result,
                                                        const void *mrHandle) {
  // A provider that reports success without publishing an MR handle violates
  // the registration contract. Treat that as an internal error rather than a
  // capability miss so callers cannot silently fall back or skip the failure.
  return result == flagcxSuccess && mrHandle == NULL ? flagcxInternalError
                                                     : result;
}

static flagcxResult_t
flagcxOneSideGetDmaBufExportRange(void *buff, size_t size, void **exportBase,
                                  size_t *exportSize, uint64_t *dmaBufOffset) {
  if (buff == nullptr || size == 0 || exportBase == nullptr ||
      exportSize == nullptr || dmaBufOffset == nullptr)
    return flagcxInvalidArgument;

  flagcxMemAllocationInfo allocation = {};
  flagcxResult_t provenanceResult =
      globalMemAllocRegistry.findRange(buff, size, &allocation);
  if (provenanceResult != flagcxSuccess &&
      provenanceResult != flagcxInvalidUsage)
    return provenanceResult;

  // The allocation registry deliberately records the user-visible size, not
  // the allocator's granularity-rounded mapping extent. DMA-BUF export APIs
  // require the native extent, so always query it from the latest adaptor.
  // The registry remains the source of truth for ownership and user bounds.
  if (deviceAdaptor == nullptr || deviceAdaptor->getAddressRange == nullptr)
    return flagcxNotSupported;
  void *base = nullptr;
  size_t allocationSize = 0;
  flagcxResult_t rangeResult =
      deviceAdaptor->getAddressRange(buff, &base, &allocationSize);
  if (rangeResult != flagcxSuccess)
    return rangeResult;

  uintptr_t address = reinterpret_cast<uintptr_t>(buff);
  uintptr_t baseAddress = reinterpret_cast<uintptr_t>(base);
  if (base == nullptr || allocationSize == 0 || address < baseAddress)
    return flagcxInvalidUsage;
  size_t userOffset = address - baseAddress;
  if (userOffset > allocationSize || size > allocationSize - userOffset)
    return flagcxInvalidUsage;
  if (provenanceResult == flagcxSuccess) {
    uintptr_t trackedBase = reinterpret_cast<uintptr_t>(allocation.base);
    if (trackedBase < baseAddress || allocation.size > allocationSize ||
        trackedBase - baseAddress > allocationSize - allocation.size)
      return flagcxInvalidUsage;
  }

  long pageSizeResult = sysconf(_SC_PAGESIZE);
  if (pageSizeResult <= 0)
    return flagcxSystemError;
  uintptr_t pageSize = static_cast<uintptr_t>(pageSizeResult);
  uintptr_t registrationAddress = address - address % pageSize;
  if (registrationAddress < baseAddress)
    return flagcxInvalidUsage;

  *exportBase = base;
  *exportSize = allocationSize;
  *dmaBufOffset = static_cast<uint64_t>(registrationAddress - baseAddress);
  return flagcxSuccess;
}

static flagcxResult_t
flagcxOneSideChoosePublishSlot(flagcxHeteroComm_t heteroComm,
                               int *publishSlot) {
  if (heteroComm == nullptr || heteroComm->bootstrap == nullptr ||
      heteroComm->nRanks <= 0 || publishSlot == nullptr)
    return flagcxInvalidArgument;

  int *counts = nullptr;
  flagcxResult_t prepareResult = flagcxCalloc(&counts, heteroComm->nRanks);
  flagcxResult_t commonPrepare =
      flagcxOneSideConvergeStatus(heteroComm->bootstrap, heteroComm->rank,
                                  heteroComm->nRanks, prepareResult);
  if (commonPrepare != flagcxSuccess) {
    free(counts);
    return commonPrepare;
  }
  counts[heteroComm->rank] = heteroComm->oneSideHandleCount;
  flagcxResult_t result =
      bootstrapCollAllGather(heteroComm->bootstrap, counts, sizeof(int));
  if (result != flagcxSuccess) {
    free(counts);
    return result;
  }
  int maxCount = 0;
  for (int rank = 0; rank < heteroComm->nRanks; rank++) {
    int count = counts[rank];
    if (count < 0) {
      free(counts);
      return flagcxInternalError;
    }
    if (count > maxCount)
      maxCount = count;
  }
  free(counts);
  if (maxCount == 0) {
    *publishSlot = 0;
    return flagcxSuccess;
  }

  size_t occupancyCount = 0;
  if (static_cast<size_t>(maxCount) >
      SIZE_MAX / static_cast<size_t>(heteroComm->nRanks)) {
    prepareResult = flagcxSystemError;
  } else {
    occupancyCount = static_cast<size_t>(heteroComm->nRanks) * maxCount;
  }
  uint8_t *occupancy = nullptr;
  if (prepareResult == flagcxSuccess)
    prepareResult = flagcxCalloc(&occupancy, occupancyCount);
  commonPrepare =
      flagcxOneSideConvergeStatus(heteroComm->bootstrap, heteroComm->rank,
                                  heteroComm->nRanks, prepareResult);
  if (commonPrepare != flagcxSuccess) {
    free(occupancy);
    return commonPrepare;
  }
  uint8_t *local = occupancy + heteroComm->rank * maxCount;
  for (int i = 0; i < heteroComm->oneSideHandleCount; i++)
    local[i] = heteroComm->oneSideHandles[i] != nullptr ? 1 : 0;
  result = bootstrapCollAllGather(heteroComm->bootstrap, occupancy,
                                  maxCount * sizeof(uint8_t));
  if (result == flagcxSuccess)
    result = flagcxOneSideSelectCommonPublishSlot(occupancy, heteroComm->nRanks,
                                                  maxCount, publishSlot);
  free(occupancy);
  return result;
}

flagcxResult_t flagcxOneSideRegisterMr(struct flagcxHeteroComm *heteroComm,
                                       void *regComm, void *buff, size_t size,
                                       int ptrType, bool isVmm, int mrFlags,
                                       void **mrHandle,
                                       flagcxVmmMrRoute_t *selectedRoute) {
  if (heteroComm == NULL || heteroComm->netAdaptor == NULL || regComm == NULL ||
      buff == NULL || size == 0 || mrHandle == NULL || selectedRoute == NULL)
    return flagcxInvalidArgument;
  *mrHandle = NULL;
  *selectedRoute = FLAGCX_VMM_MR_ROUTE_NONE;

  struct flagcxNetAdaptor *net = heteroComm->netAdaptor;
  if (!isVmm) {
    if (net->regMr == NULL)
      return flagcxNotSupported;
    flagcxResult_t result =
        net->regMr(regComm, buff, size, ptrType, mrFlags, mrHandle);
    return flagcxOneSideValidateRegisteredMr(result, *mrHandle);
  }

  const bool legacyDeviceV1 =
      deviceAdaptor != NULL && (deviceAdaptor->internalFlags &
                                FLAGCX_DEVICE_ADAPTOR_INTERNAL_LEGACY_V1) != 0;
  const bool legacyNetV1 =
      (net->internalFlags & FLAGCX_NET_ADAPTOR_INTERNAL_LEGACY_V1) != 0;
  const bool legacyV1 = legacyDeviceV1 || legacyNetV1;
  if (legacyV1) {
    // v1 had no explicit per-allocation or MR-route capability. Preserve its
    // historical behavior exactly: try DMA-BUF when both callbacks exist, but
    // treat any export failure as a request to use ordinary VA registration.
    // A failure after a successful export remains a real transport error.
    if (deviceAdaptor->getHandleForAddressRange != NULL &&
        net->regMrDmaBuf != NULL) {
      int dmaBufFd = -1;
      flagcxResult_t exportResult = deviceAdaptor->getHandleForAddressRange(
          (void *)&dmaBufFd, buff, size, 0);
      if (exportResult == flagcxSuccess && dmaBufFd >= 0) {
        flagcxResult_t result = net->regMrDmaBuf(
            regComm, buff, size, ptrType, 0ULL, dmaBufFd, mrFlags, mrHandle);
        close(dmaBufFd);
        result = flagcxOneSideValidateRegisteredMr(result, *mrHandle);
        if (result == flagcxSuccess) {
          *selectedRoute = FLAGCX_VMM_MR_ROUTE_DMABUF;
          INFO(FLAGCX_REG,
               "Legacy v1 MR registered through DMA-BUF: buff=%p size=%zu",
               buff, size);
        }
        return result;
      }
      if (dmaBufFd >= 0)
        close(dmaBufFd);
    }

    if (net->regMr == NULL)
      return flagcxNotSupported;
    flagcxResult_t result =
        net->regMr(regComm, buff, size, ptrType, mrFlags, mrHandle);
    result = flagcxOneSideValidateRegisteredMr(result, *mrHandle);
    if (result == flagcxSuccess) {
      *selectedRoute = FLAGCX_VMM_MR_ROUTE_VA;
      INFO(FLAGCX_REG,
           "Legacy v1 MR registered through VA fallback: buff=%p size=%zu",
           buff, size);
    }
    return result;
  }

  flagcxVmmMrMode_t mode = flagcxVmmMrModeAuto;
  flagcxResult_t modeResult =
      flagcxOneSideParseVmmMrMode(flagcxGetEnv("FLAGCX_VMM_MR_MODE"), &mode);
  if (modeResult != flagcxSuccess)
    return modeResult;

  flagcxNetProperties_t properties = {};
  if (net->getProperties == NULL)
    return flagcxNotSupported;
  flagcxResult_t propertiesResult =
      net->getProperties(heteroComm->netDev, &properties);
  if (propertiesResult != flagcxSuccess)
    return propertiesResult;

  // Match NCCL's operation-driven route selection: adaptor and provider
  // capabilities describe candidates, while exporting/registering this exact
  // allocation is the authoritative probe. Do not require the allocation to
  // appear in a private allocator map; externally-created VMM allocations are
  // valid inputs once symPhysAlloc has identified them as VMM-backed.
  const uint32_t deviceCaps = deviceAdaptor->vmmMrCaps;
  const uint32_t providerCaps = net->vmmMrCaps;
  const uint32_t effectiveCaps = deviceCaps & providerCaps;

  bool dmaBufExportSupported = false;
  if (mode != flagcxVmmMrModeVa && (effectiveCaps & FLAGCX_VMM_MR_CAP_DMABUF) &&
      deviceAdaptor->dmaSupport != NULL) {
    flagcxResult_t supportResult =
        deviceAdaptor->dmaSupport(&dmaBufExportSupported);
    if (supportResult == flagcxNotSupported) {
      dmaBufExportSupported = false;
    } else if (supportResult != flagcxSuccess) {
      return supportResult;
    }
  }

  flagcxVmmMrRoute_t route = flagcxOneSideSelectVmmMrRoute(
      effectiveCaps, properties.ptrSupport, dmaBufExportSupported,
      deviceAdaptor->getHandleForAddressRange != NULL &&
          net->regMrDmaBuf != NULL,
      net->regMr != NULL, mode);
  INFO(FLAGCX_REG,
       "VMM MR route selection: provider=%s mode=%s deviceCaps=0x%x "
       "providerCaps=0x%x effectiveCaps=0x%x ptrSupport=0x%x "
       "dmaBufExport=%d regMrDmaBuf=%d regMr=%d selected=%d",
       net->name != NULL ? net->name : "unknown",
       mode == flagcxVmmMrModeDmaBuf
           ? "dmabuf"
           : (mode == flagcxVmmMrModeVa ? "va" : "auto"),
       deviceCaps, providerCaps, effectiveCaps, properties.ptrSupport,
       dmaBufExportSupported ? 1 : 0,
       deviceAdaptor->getHandleForAddressRange != NULL &&
               net->regMrDmaBuf != NULL
           ? 1
           : 0,
       net->regMr != NULL ? 1 : 0, static_cast<int>(route));
  if (route == FLAGCX_VMM_MR_ROUTE_NONE)
    return flagcxNotSupported;

  flagcxResult_t result = flagcxSuccess;
  if (route == FLAGCX_VMM_MR_ROUTE_DMABUF) {
    int dmaBufFd = -1;
    void *exportBase = nullptr;
    size_t exportSize = 0;
    uint64_t dmaBufOffset = 0;
    result = flagcxOneSideGetDmaBufExportRange(buff, size, &exportBase,
                                               &exportSize, &dmaBufOffset);
    if (result == flagcxSuccess)
      result = deviceAdaptor->getHandleForAddressRange(
          (void *)&dmaBufFd, exportBase, exportSize, 0);
    if (result == flagcxSuccess && dmaBufFd < 0)
      result = flagcxNotSupported;
    INFO(FLAGCX_REG,
         "VMM MR DMA-BUF export: provider=%s base=%p size=%zu offset=%llu "
         "result=%d fdValid=%d",
         net->name != NULL ? net->name : "unknown", exportBase, exportSize,
         (unsigned long long)dmaBufOffset, static_cast<int>(result),
         dmaBufFd >= 0 ? 1 : 0);
    if (result == flagcxSuccess) {
      result = net->regMrDmaBuf(regComm, buff, size, FLAGCX_PTR_CUDA,
                                dmaBufOffset, dmaBufFd, mrFlags, mrHandle);
      INFO(FLAGCX_REG,
           "VMM MR NET registration: provider=%s route=dmabuf buff=%p "
           "size=%zu result=%d handleValid=%d",
           net->name != NULL ? net->name : "unknown", buff, size,
           static_cast<int>(result), *mrHandle != NULL ? 1 : 0);
    }
    if (dmaBufFd >= 0)
      close(dmaBufFd);

    // A component may advertise DMA-BUF generally but reject this allocation.
    // Only an explicit NotSupported result selects the validated VA fallback;
    // transport/device failures remain visible and are never masked.
    if (result == flagcxNotSupported && mode == flagcxVmmMrModeAuto) {
      // A provider may return NotSupported after partially creating an MR.
      // Do not overwrite that ownership with the VA handle. Complete its
      // rollback first; if rollback fails, leave the handle visible to the
      // caller so the normal pending-cleanup state machine can retry it.
      if (*mrHandle != NULL) {
        if (net->deregMr == NULL)
          return flagcxInternalError;
        flagcxResult_t cleanupResult = net->deregMr(regComm, *mrHandle);
        if (cleanupResult != flagcxSuccess)
          return cleanupResult;
        *mrHandle = NULL;
      }
      route = flagcxOneSideSelectVmmMrRoute(
          effectiveCaps & ~FLAGCX_VMM_MR_CAP_DMABUF, properties.ptrSupport,
          false, false, net->regMr != NULL, mode);
      if (route == FLAGCX_VMM_MR_ROUTE_VA) {
        *mrHandle = NULL;
        result =
            net->regMr(regComm, buff, size, FLAGCX_PTR_CUDA, mrFlags, mrHandle);
        INFO(FLAGCX_REG,
             "VMM MR NET registration: provider=%s route=va-fallback "
             "buff=%p size=%zu result=%d handleValid=%d",
             net->name != NULL ? net->name : "unknown", buff, size,
             static_cast<int>(result), *mrHandle != NULL ? 1 : 0);
      }
    }
    if (result != flagcxSuccess) {
      return result;
    }
  } else {
    result =
        net->regMr(regComm, buff, size, FLAGCX_PTR_CUDA, mrFlags, mrHandle);
    INFO(FLAGCX_REG,
         "VMM MR NET registration: provider=%s route=va buff=%p size=%zu "
         "result=%d handleValid=%d",
         net->name != NULL ? net->name : "unknown", buff, size,
         static_cast<int>(result), *mrHandle != NULL ? 1 : 0);
  }
  result = flagcxOneSideValidateRegisteredMr(result, *mrHandle);
  if (result == flagcxSuccess) {
    *selectedRoute = route;
    INFO(FLAGCX_REG, "VMM MR registered through %s (mode=%s): buff=%p size=%zu",
         route == FLAGCX_VMM_MR_ROUTE_DMABUF ? "DMA-BUF" : "VA",
         mode == flagcxVmmMrModeDmaBuf
             ? "dmabuf"
             : (mode == flagcxVmmMrModeVa ? "va" : "auto"),
         buff, size);
  }
  return result;
}

flagcxResult_t flagcxOneSideRegisterInternal(flagcxHeteroComm_t heteroComm,
                                             void *buff, size_t size,
                                             bool isVmm, bool acquireWindowRef,
                                             int *mrIndex,
                                             bool *rollbackPending) {
  if (mrIndex != NULL)
    *mrIndex = -1;
  if (rollbackPending != NULL)
    *rollbackPending = false;
  if (heteroComm == NULL || heteroComm->netAdaptor == NULL ||
      heteroComm->netAdaptor->iput == NULL ||
      heteroComm->netAdaptor->regMr == NULL ||
      heteroComm->netAdaptor->getMrInfo == NULL) {
    return flagcxNotSupported;
  }

  if (heteroComm->bootstrap == NULL) {
    INFO(FLAGCX_REG, "flagcxOneSideRegister: bootstrap is NULL");
    return flagcxNotSupported;
  }

  // Check for duplicate registration of the same buffer within this comm.
  // Every registration entry point is collective: all ranks must agree on
  // the exact reused slot before any rank returns or begins a new metadata
  // transaction. This also covers public registrations after an asymmetric
  // deregMr failure left different local handle tables.
  int existingIndex = -1;
  flagcxResult_t lookupResult = flagcxSuccess;
  for (int i = 0; i < heteroComm->oneSideHandleCount; i++) {
    struct flagcxOneSideHandleInfo *h = heteroComm->oneSideHandles[i];
    if (h != NULL && h->baseVas != NULL && h->regionSizes != NULL &&
        h->baseVas[heteroComm->rank] == (uintptr_t)buff) {
      if (h->regionSizes[heteroComm->rank] != size)
        lookupResult = flagcxInvalidUsage;
      else if (isVmm && h->registrationRoute == FLAGCX_VMM_MR_ROUTE_NONE)
        // Never attach a VMM window to an MR that was registered through the
        // ordinary-memory path before the native allocation was identified.
        lookupResult = flagcxInvalidUsage;
      else
        existingIndex = i;
      break;
    }
  }
  lookupResult =
      flagcxOneSideConvergeStatus(heteroComm->bootstrap, heteroComm->rank,
                                  heteroComm->nRanks, lookupResult);
  if (lookupResult != flagcxSuccess)
    return lookupResult;
  // Encode index+1 so ranks agree on both reuse-vs-create and the public MR
  // index. Agreeing only on a boolean could silently pair different slots
  // after asymmetric cleanup histories.
  int *reuseDecisions = nullptr;
  flagcxResult_t allocationResult =
      flagcxCalloc(&reuseDecisions, heteroComm->nRanks);
  allocationResult =
      flagcxOneSideConvergeStatus(heteroComm->bootstrap, heteroComm->rank,
                                  heteroComm->nRanks, allocationResult);
  if (allocationResult != flagcxSuccess) {
    free(reuseDecisions);
    return allocationResult;
  }
  reuseDecisions[heteroComm->rank] = existingIndex + 1;
  flagcxResult_t gatherResult = bootstrapCollAllGather(
      heteroComm->bootstrap, reuseDecisions, sizeof(int));
  if (gatherResult != flagcxSuccess) {
    free(reuseDecisions);
    return gatherResult;
  }
  for (int peer = 1; peer < heteroComm->nRanks; peer++) {
    if (reuseDecisions[peer] != reuseDecisions[0]) {
      free(reuseDecisions);
      return flagcxInvalidUsage;
    }
  }
  free(reuseDecisions);
  if (existingIndex >= 0) {
    struct flagcxOneSideHandleInfo *h =
        heteroComm->oneSideHandles[existingIndex];
    INFO(FLAGCX_REG,
         "flagcxOneSideRegister: buffer %p already registered at index %d",
         buff, existingIndex);
    if (acquireWindowRef) {
      flagcxResult_t refResult = flagcxOneSideConvergeStatus(
          heteroComm->bootstrap, heteroComm->rank, heteroComm->nRanks,
          h->windowRefs == UINT32_MAX ? flagcxSystemError : flagcxSuccess);
      if (refResult != flagcxSuccess)
        return refResult;
      h->windowRefs++;
    } else {
      h->commOwned = 1;
    }
    if (mrIndex != NULL)
      *mrIndex = existingIndex;
    return flagcxSuccess;
  }

  // A prior rollback may still own an MR whose deregistration failed. Retry
  // it before creating another connection/MR transaction.
  flagcxResult_t pendingResult = flagcxOneSideRetryPendingCleanup(heteroComm);
  pendingResult =
      flagcxOneSideConvergeStatus(heteroComm->bootstrap, heteroComm->rank,
                                  heteroComm->nRanks, pendingResult);
  if (pendingResult != flagcxSuccess)
    return pendingResult;

  flagcxResult_t res = flagcxSuccess;
  void *mrHandle = NULL;
  struct flagcxNetMrInfo localMrInfo = {};
  void *regComm = NULL;
  struct flagcxOneSideHandleInfo *info = NULL;
  flagcxVmmMrRoute_t selectedRoute = FLAGCX_VMM_MR_ROUTE_NONE;
  struct flagcxOneSideHandleInfo **publishHandles = heteroComm->oneSideHandles;
  int publishCapacity = heteroComm->oneSideHandleCapacity;
  bool replaceHandleArray = false;
  int publishSlot = -1;
  res = flagcxOneSideChoosePublishSlot(heteroComm, &publishSlot);
  if (res != flagcxSuccess)
    return res;

  // Prepare a replacement array locally, but do not attach it until the
  // collective transaction succeeds on every rank.
  if (publishSlot >= heteroComm->oneSideHandleCapacity) {
    int newCap = heteroComm->oneSideHandleCapacity == 0
                     ? 4
                     : heteroComm->oneSideHandleCapacity;
    while (newCap <= publishSlot)
      newCap *= 2;
    publishHandles = (struct flagcxOneSideHandleInfo **)calloc(
        newCap, sizeof(struct flagcxOneSideHandleInfo *));
    if (publishHandles == NULL) {
      res = flagcxSystemError;
    } else {
      for (int i = 0; i < heteroComm->oneSideHandleCount; i++)
        publishHandles[i] = heteroComm->oneSideHandles[i];
      publishCapacity = newCap;
      replaceHandleArray = true;
    }
  }

  res = flagcxOneSideConvergeStatus(heteroComm->bootstrap, heteroComm->rank,
                                    heteroComm->nRanks, res);
  if (res != flagcxSuccess) {
    if (replaceHandleArray)
      free(publishHandles);
    return res;
  }

  bool isFirstHandle = (publishSlot == 0);

  res = flagcxCalloc(&info, 1);
  res = flagcxOneSideConvergeStatus(heteroComm->bootstrap, heteroComm->rank,
                                    heteroComm->nRanks, res);
  if (res != flagcxSuccess)
    goto fail_info;

  // First handle for this heteroComm: build a new full-mesh IB connection set
  if (isFirstHandle) {
    FLAGCXCHECKGOTO(flagcxOneSideBuildFullMesh(heteroComm, info), res,
                    fail_info);
  }

  // Use self recvComm for MR registration (PD match)
  {
    void *selfRecvComm =
        isFirstHandle
            ? info->fullRecvComms[heteroComm->rank]
            : heteroComm->oneSideHandles[0]->fullRecvComms[heteroComm->rank];
    info->localRecvComm = selfRecvComm;
    regComm = selfRecvComm;
  }

  res = flagcxOneSideRegisterMr(heteroComm, regComm, buff, size,
                                FLAGCX_PTR_CUDA, isVmm, FLAGCX_NET_MR_FLAG_NONE,
                                &mrHandle, &selectedRoute);
  info->registrationRoute = (uint8_t)selectedRoute;
  // Visibility is a property of the device/NIC path, not of how this MR was
  // registered. In particular, an ordinary cudaMalloc-style allocation needs
  // the same post-READ acquire as a VMM allocation on Hygon.
  info->gdrFlushRequirements =
      flagcxResolveGdrFlushRequirements(deviceAdaptor->gdrFlushRequirements);
  if (mrHandle != NULL) {
    info->localMrHandle = mrHandle;
    info->ownsLocalMr = 1;
  }
  if (res == flagcxSuccess && mrHandle == NULL)
    res = flagcxInternalError;
  if (res != flagcxSuccess) {
    INFO(FLAGCX_REG, "flagcxOneSideRegister: regMr failed, res=%d", res);
  } else {
    res =
        flagcxOneSideGetMrInfo(heteroComm->netAdaptor, mrHandle, &localMrInfo);
  }

  // Every rank reports local connection/MR preparation before any rank enters
  // the MR base/size/key metadata collectives.
  res = flagcxOneSideConvergeStatus(heteroComm->bootstrap, heteroComm->rank,
                                    heteroComm->nRanks, res);
  if (res != flagcxSuccess)
    goto fail_mr;

  // Allgather MR info
  {
    int nranks = heteroComm->nRanks;
    heteroComm->oneSideDataMetadataExchangeCount++;
    flagcxResult_t exchangeResult =
        flagcxOneSideExchangeMrInfo(heteroComm->bootstrap, heteroComm->rank,
                                    nranks, buff, size, &localMrInfo, info);
    res = flagcxOneSideConvergeStatus(heteroComm->bootstrap, heteroComm->rank,
                                      nranks, exchangeResult);
    if (res != flagcxSuccess)
      goto fail_mr;

    int slot = publishSlot;
    if (replaceHandleArray) {
      free(heteroComm->oneSideHandles);
      heteroComm->oneSideHandles = publishHandles;
      heteroComm->oneSideHandleCapacity = publishCapacity;
      replaceHandleArray = false;
    }
    heteroComm->oneSideHandles[slot] = info;
    info->windowRefs = acquireWindowRef ? 1 : 0;
    info->commOwned = acquireWindowRef ? 0 : 1;
    if (slot >= heteroComm->oneSideHandleCount)
      heteroComm->oneSideHandleCount = slot + 1;
    if (mrIndex != NULL)
      *mrIndex = slot;

    // Publish fullSendComms to the RMA proxy on the first registration so
    // its progress thread can look up per-peer sendComms without racing
    // on the realloc-resized oneSideHandles array.
    if (slot == 0 && info->fullSendComms != NULL) {
      flagcxHeteroRmaProxyPublishSendComms(heteroComm, info->fullSendComms);
    }

    INFO(FLAGCX_REG,
         "One-sided register index %d allgather results (rank %d, nranks %d):",
         slot, heteroComm->rank, nranks);
    for (int i = 0; i < nranks; i++) {
      INFO(FLAGCX_REG, "  Rank %d: base_va=0x%lx, size=%zu, keys=%u", i,
           info->baseVas[i], info->regionSizes[i], info->mrInfos[i].nKeys);
    }
  }

  return flagcxSuccess;

fail_mr : {
  flagcxResult_t localCleanupResult = flagcxSuccess;
  if (info != NULL) {
    localCleanupResult =
        flagcxOneSideCleanupHandle(heteroComm, info, isFirstHandle);
    if (localCleanupResult != flagcxSuccess) {
      WARN("flagcxOneSideRegister: rollback retained for retry, res=%d",
           (int)localCleanupResult);
      flagcxOneSideRetainCleanup(heteroComm, info);
      info = NULL;
    }
  }
  flagcxResult_t commonCleanupResult =
      flagcxOneSideConvergeStatus(heteroComm->bootstrap, heteroComm->rank,
                                  heteroComm->nRanks, localCleanupResult);
  if (commonCleanupResult != flagcxSuccess) {
    // Every symmetric-window caller keeps a cleanup token, including ranks
    // whose local MR rollback already completed, so its later collective
    // deregistration cannot leave the failing rank alone in convergence.
    if (rollbackPending != NULL)
      *rollbackPending = true;
    res = commonCleanupResult;
  }
}
fail_info:
  free(info);
  if (replaceHandleArray)
    free(publishHandles);
  return res;
}

flagcxResult_t flagcxOneSideRegister(flagcxComm_t comm, void *buff,
                                     size_t size) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (comm->heteroComm == nullptr)
    return flagcxNotSupported;
  return flagcxOneSideRegisterInternal(
      comm->heteroComm, buff, size,
      flagcxOneSideRegistryRangeIsVmm(buff, size));
}

flagcxResult_t
flagcxOneSideDeregisterInternal(struct flagcxHeteroComm *heteroComm,
                                int index) {
  if (heteroComm == NULL || index < 0 ||
      index >= heteroComm->oneSideHandleCount ||
      heteroComm->oneSideHandles == NULL)
    return flagcxInvalidArgument;

  struct flagcxOneSideHandleInfo *info = heteroComm->oneSideHandles[index];
  if (info == NULL || info->windowRefs == 0)
    return flagcxInvalidUsage;
  if (info->windowRefs > 1) {
    info->windowRefs--;
    return flagcxSuccess;
  }
  if (info->commOwned) {
    info->windowRefs = 0;
    return flagcxSuccess;
  }
  flagcxResult_t localResult = flagcxSuccess;
  if (info != NULL && info->localMrHandle != NULL) {
    if (heteroComm->netAdaptor == NULL ||
        heteroComm->netAdaptor->deregMr == NULL ||
        info->localRecvComm == NULL) {
      localResult = flagcxInternalError;
    } else {
      localResult = heteroComm->netAdaptor->deregMr(info->localRecvComm,
                                                    info->localMrHandle);
      if (localResult == flagcxSuccess) {
        info->localMrHandle = NULL;
        info->ownsLocalMr = 0;
      }
    }
  }

  // Deregistration is deliberately rank-local. Communicator teardown can be
  // called serially by ranks in one process, and a collective here would
  // deadlock that valid lifecycle. A failed rank keeps the exact MR object and
  // its window so that rank can retry; successful ranks may release theirs.
  if (localResult != flagcxSuccess)
    return localResult;

  info->windowRefs = 0;
  flagcxOneSideFreeMrInfo(info);
  if (info->ownsConnections) {
    // Slot zero remains as a connection-only owner. Later registrations and
    // signal/staging MRs still need its recvComm/PD and full-mesh QPs.
    return flagcxSuccess;
  }
  free(info);
  heteroComm->oneSideHandles[index] = NULL;
  return flagcxSuccess;
}

flagcxResult_t flagcxOneSideDeregister(struct flagcxHeteroComm *heteroComm) {
  if (heteroComm == NULL)
    return flagcxInternalError;

  FLAGCXCHECK(flagcxOneSideRetryPendingCleanup(heteroComm));

  // Deregister all data handles in reverse order
  for (int i = heteroComm->oneSideHandleCount - 1; i >= 0; i--) {
    struct flagcxOneSideHandleInfo *info = heteroComm->oneSideHandles[i];
    if (info == NULL)
      continue;

    const bool closeConnections = info->ownsConnections != 0;
    // Signal/staging MRs use the first data handle's recvComm/PD. They must be
    // released before that connection owner can be destroyed.
    if (closeConnections &&
        (heteroComm->signalHandle != NULL || heteroComm->stagingHandle != NULL))
      return flagcxInProgress;
    flagcxResult_t result =
        flagcxOneSideCleanupHandle(heteroComm, info, closeConnections);
    if (result != flagcxSuccess)
      return result;
    if (closeConnections && heteroComm->rmaProxy != NULL)
      __atomic_store_n(&heteroComm->rmaProxy->fullSendComms, NULL,
                       __ATOMIC_RELEASE);
    free(info);
    heteroComm->oneSideHandles[i] = NULL;
  }

  free(heteroComm->oneSideHandles);
  heteroComm->oneSideHandles = NULL;
  heteroComm->oneSideHandleCount = 0;
  heteroComm->oneSideHandleCapacity = 0;
  return flagcxSuccess;
}

flagcxResult_t flagcxOneSideSignalRegisterInternal(const flagcxComm_t comm,
                                                   void *buff, size_t size,
                                                   int ptrType,
                                                   bool isVmmAllocation) {
  if (comm == NULL || buff == NULL || size == 0)
    return flagcxInvalidArgument;
  if (useHomoComm(comm) && !useHeteroComm()) {
    return flagcxSuccess;
  }

  struct flagcxHeteroComm *heteroComm = comm->heteroComm;
  if (heteroComm == NULL)
    return flagcxNotSupported;

  // Per-heteroComm dedup: signal IPC and network state share this local base,
  // but either transport may be absent.
  if (heteroComm->rmaSignalBase != NULL) {
    if (heteroComm->rmaSignalBase != buff) {
      WARN("flagcxOneSideSignalRegister: comm %p already registered with a "
           "different buffer",
           (void *)comm);
      return flagcxInvalidUsage;
    }
    return flagcxSuccess;
  }

  struct flagcxOneSideHandleInfo *existing = heteroComm->signalHandle;
  if (existing != NULL) {
    if (existing->baseVas != NULL &&
        existing->baseVas[comm->rank] != (uintptr_t)buff) {
      WARN("flagcxOneSideSignalRegister: comm %p already registered with a "
           "different buffer",
           (void *)comm);
      return flagcxInvalidUsage;
    }
    return flagcxSuccess;
  }

  // Validate ptrType — only known pointer types are accepted.
  if (ptrType != FLAGCX_PTR_HOST && ptrType != FLAGCX_PTR_CUDA &&
      ptrType != FLAGCX_PTR_DMABUF) {
    WARN("flagcxOneSideSignalRegister: invalid ptrType %d", ptrType);
    return flagcxInvalidArgument;
  }

  // Build the local IPC mapping before attempting any network setup. This is
  // collective and remains valid when the RDMA provider is unavailable.
  const bool isVmm = ptrType == FLAGCX_PTR_CUDA && isVmmAllocation;
  int ipcSlot = -1;
  if (ptrType == FLAGCX_PTR_CUDA && !isVmm)
    ipcSlot = buildIpcPeerPointers(comm, buff, size);
  heteroComm->rmaSignalBase = buff;
  heteroComm->rmaSignalSize = size;
  heteroComm->rmaSignalIpcSlot = ipcSlot;
  if (heteroComm->rmaProxy != NULL && heteroComm->rmaProxy->ipcState != NULL)
    flagcxHeteroRmaIpcDestroy(heteroComm);

  if (heteroComm->netAdaptor == NULL ||
      heteroComm->netAdaptor->iputSignal == NULL ||
      heteroComm->netAdaptor->regMr == NULL ||
      heteroComm->netAdaptor->getMrInfo == NULL) {
    if (ipcSlot >= 0)
      return flagcxSuccess;
    heteroComm->rmaSignalBase = NULL;
    heteroComm->rmaSignalSize = 0;
    return flagcxNotSupported;
  }

  if (heteroComm->bootstrap == NULL) {
    INFO(FLAGCX_REG, "flagcxOneSideSignalRegister: bootstrap is NULL");
    if (ipcSlot >= 0)
      return flagcxSuccess;
    heteroComm->rmaSignalBase = NULL;
    heteroComm->rmaSignalSize = 0;
    return flagcxNotSupported;
  }

  // Signal registration reuses full-mesh connections from this heteroComm's
  // first data handle.  Lazily build them if not yet established.
  {
    flagcxResult_t meshRes = flagcxOneSideEnsureFullMesh(heteroComm);
    if (meshRes != flagcxSuccess) {
      INFO(FLAGCX_REG,
           "flagcxOneSideSignalRegister: failed to ensure full-mesh (%d)",
           (int)meshRes);
      if (ipcSlot >= 0)
        return flagcxSuccess;
      heteroComm->rmaSignalBase = NULL;
      heteroComm->rmaSignalSize = 0;
      return meshRes;
    }
  }
  struct flagcxOneSideHandleInfo *firstDataHandle =
      heteroComm->oneSideHandles[0];

  flagcxResult_t res = flagcxSuccess;
  void *mrHandle = NULL;
  struct flagcxNetMrInfo localMrInfo = {};
  void *regComm = NULL;
  struct flagcxOneSideHandleInfo *info = NULL;
  flagcxVmmMrRoute_t selectedRoute = FLAGCX_VMM_MR_ROUTE_NONE;
  void *selfRecvComm = firstDataHandle->fullRecvComms[heteroComm->rank];
  regComm = selfRecvComm;

  res = flagcxCalloc(&info, 1);
  res = flagcxOneSideConvergeStatus(heteroComm->bootstrap, heteroComm->rank,
                                    heteroComm->nRanks, res);
  if (res != flagcxSuccess)
    goto fail_mr;

  res = flagcxOneSideRegisterMr(heteroComm, regComm, buff, size, ptrType, isVmm,
                                FLAGCX_NET_MR_FLAG_FORCE_SO, &mrHandle,
                                &selectedRoute);
  info->registrationRoute = (uint8_t)selectedRoute;
  if (mrHandle != NULL) {
    info->localMrHandle = mrHandle;
    info->localRecvComm = selfRecvComm;
    info->ownsLocalMr = 1;
  }
  if (res == flagcxSuccess && mrHandle == NULL)
    res = flagcxInternalError;
  if (res != flagcxSuccess) {
    INFO(FLAGCX_REG, "flagcxOneSideSignalRegister: regMr failed, res=%d", res);
  } else {
    res =
        flagcxOneSideGetMrInfo(heteroComm->netAdaptor, mrHandle, &localMrInfo);
  }

  res = flagcxOneSideConvergeStatus(heteroComm->bootstrap, heteroComm->rank,
                                    heteroComm->nRanks, res);
  if (res != flagcxSuccess)
    goto fail_mr;

  {
    int nranks = heteroComm->nRanks;
    flagcxResult_t exchangeResult =
        flagcxOneSideExchangeMrInfo(heteroComm->bootstrap, heteroComm->rank,
                                    nranks, buff, size, &localMrInfo, info);
    res = flagcxOneSideConvergeStatus(heteroComm->bootstrap, heteroComm->rank,
                                      nranks, exchangeResult);
    if (res != flagcxSuccess)
      goto fail_mr;
    heteroComm->signalHandle = info;
    INFO(FLAGCX_REG, "Signal register allgather results (rank %d, nranks %d):",
         heteroComm->rank, nranks);
    for (int i = 0; i < nranks; i++) {
      INFO(FLAGCX_REG, "  Rank %d: base_va=0x%lx, size=%zu, keys=%u", i,
           info->baseVas[i], info->regionSizes[i], info->mrInfos[i].nKeys);
    }
  }

  if (ipcSlot >= 0)
    INFO(FLAGCX_REG, "Signal buffer IPC registered (slot %d) for D2D bypass",
         ipcSlot);

  return flagcxSuccess;

fail_mr:
  if (info) {
    flagcxResult_t cleanupResult =
        flagcxOneSideCleanupHandle(heteroComm, info, false);
    if (cleanupResult != flagcxSuccess) {
      flagcxOneSideRetainCleanup(heteroComm, info);
      info = NULL;
    }
    free(info);
  }
  if (ipcSlot >= 0)
    return flagcxSuccess;
  heteroComm->rmaSignalBase = NULL;
  heteroComm->rmaSignalSize = 0;
  heteroComm->rmaSignalIpcSlot = -1;
  return res;
}

flagcxResult_t flagcxOneSideSignalRegister(const flagcxComm_t comm, void *buff,
                                           size_t size, int ptrType) {
  const bool isVmmAllocation =
      ptrType == FLAGCX_PTR_CUDA && flagcxOneSideRegistryRangeIsVmm(buff, size);
  return flagcxOneSideSignalRegisterInternal(comm, buff, size, ptrType,
                                             isVmmAllocation);
}

flagcxResult_t flagcxOneSideSignalDeregister(flagcxComm_t comm) {
  if (comm == NULL || comm->heteroComm == NULL) {
    return flagcxInternalError;
  }
  struct flagcxHeteroComm *heteroComm = comm->heteroComm;
  struct flagcxOneSideHandleInfo *info = heteroComm->signalHandle;
  if (info == NULL && heteroComm->rmaSignalBase == NULL) {
    return flagcxSuccess;
  }

  if (info != NULL) {
    flagcxResult_t result = flagcxOneSideCleanupHandle(heteroComm, info, false);
    if (result != flagcxSuccess)
      return result;
  }

  if (heteroComm->rmaProxy != NULL && heteroComm->rmaProxy->ipcState != NULL)
    flagcxHeteroRmaIpcDestroy(heteroComm);

  // Release IPC and network state independently.
  if (heteroComm->rmaSignalIpcSlot >= 0)
    releaseIpcTableSlot(comm, heteroComm->rmaSignalIpcSlot);
  heteroComm->rmaSignalIpcSlot = -1;
  heteroComm->rmaSignalBase = NULL;
  heteroComm->rmaSignalSize = 0;

  if (info != NULL) {
    free(info);
  }
  heteroComm->signalHandle = NULL;
  return flagcxSuccess;
}

flagcxResult_t flagcxOneSideStagingRegister(const flagcxComm_t comm, void *buff,
                                            size_t size) {
  if (useHomoComm(comm) && !useHeteroComm()) {
    return flagcxSuccess;
  }

  // Per-heteroComm dedup
  struct flagcxOneSideHandleInfo *existingStg = comm->heteroComm->stagingHandle;
  if (existingStg != NULL) {
    if (existingStg->baseVas != NULL &&
        existingStg->baseVas[comm->rank] != (uintptr_t)buff) {
      WARN("flagcxOneSideStagingRegister: comm %p already registered with a "
           "different buffer",
           (void *)comm);
    }
    return flagcxSuccess;
  }

  struct flagcxHeteroComm *heteroComm = comm->heteroComm;
  if (heteroComm == NULL || heteroComm->netAdaptor == NULL ||
      heteroComm->netAdaptor->iput == NULL ||
      heteroComm->netAdaptor->regMr == NULL ||
      heteroComm->netAdaptor->getMrInfo == NULL) {
    INFO(FLAGCX_REG, "flagcxOneSideStagingRegister: heteroComm is NULL");
    return flagcxSuccess;
  }

  if (heteroComm->bootstrap == NULL) {
    INFO(FLAGCX_REG, "flagcxOneSideStagingRegister: bootstrap is NULL");
    return flagcxNotSupported;
  }

  // Staging registration reuses full-mesh connections from this heteroComm's
  // first data handle.  Lazily build them if not yet established.
  {
    flagcxResult_t meshRes = flagcxOneSideEnsureFullMesh(heteroComm);
    if (meshRes != flagcxSuccess) {
      INFO(FLAGCX_REG,
           "flagcxOneSideStagingRegister: failed to ensure full-mesh (%d)",
           (int)meshRes);
      return meshRes;
    }
  }
  struct flagcxOneSideHandleInfo *firstDataHandleStg =
      heteroComm->oneSideHandles[0];

  flagcxResult_t res = flagcxSuccess;
  void *mrHandle = NULL;
  struct flagcxNetMrInfo localMrInfo = {};
  void *regComm = NULL;
  struct flagcxOneSideHandleInfo *info = NULL;
  void *selfRecvComm = firstDataHandleStg->fullRecvComms[heteroComm->rank];
  regComm = selfRecvComm;

  res = flagcxCalloc(&info, 1);
  res = flagcxOneSideConvergeStatus(heteroComm->bootstrap, heteroComm->rank,
                                    heteroComm->nRanks, res);
  if (res != flagcxSuccess)
    goto fail_mr;

  // Use self recvComm from this comm's first data handle for MR registration
  // (PD match)
  {
    int type = FLAGCX_PTR_HOST;
    res =
        heteroComm->netAdaptor->regMr(regComm, buff, size, type, 0, &mrHandle);
  }
  if (mrHandle != NULL) {
    info->localMrHandle = mrHandle;
    info->localRecvComm = selfRecvComm;
    info->ownsLocalMr = 1;
  }
  if (res != flagcxSuccess || mrHandle == NULL) {
    INFO(FLAGCX_REG, "flagcxOneSideStagingRegister: regMr failed, res=%d", res);
    res = flagcxNotSupported;
  } else {
    res =
        flagcxOneSideGetMrInfo(heteroComm->netAdaptor, mrHandle, &localMrInfo);
  }

  res = flagcxOneSideConvergeStatus(heteroComm->bootstrap, heteroComm->rank,
                                    heteroComm->nRanks, res);
  if (res != flagcxSuccess)
    goto fail_mr;

  {
    int nranks = heteroComm->nRanks;
    flagcxResult_t exchangeResult =
        flagcxOneSideExchangeMrInfo(heteroComm->bootstrap, heteroComm->rank,
                                    nranks, buff, size, &localMrInfo, info);
    res = flagcxOneSideConvergeStatus(heteroComm->bootstrap, heteroComm->rank,
                                      nranks, exchangeResult);
    if (res != flagcxSuccess)
      goto fail_mr;
    heteroComm->stagingHandle = info;
    INFO(FLAGCX_REG, "Staging register allgather results (rank %d, nranks %d):",
         heteroComm->rank, nranks);
    for (int i = 0; i < nranks; i++) {
      INFO(FLAGCX_REG, "  Rank %d: base_va=0x%lx, size=%zu, keys=%u", i,
           info->baseVas[i], info->regionSizes[i], info->mrInfos[i].nKeys);
    }
  }

  return flagcxSuccess;

fail_mr:
  if (info) {
    flagcxResult_t cleanupResult =
        flagcxOneSideCleanupHandle(heteroComm, info, false);
    if (cleanupResult != flagcxSuccess) {
      flagcxOneSideRetainCleanup(heteroComm, info);
      info = NULL;
    }
    free(info);
  }
  return res;
}

flagcxResult_t flagcxOneSideStagingDeregister(const flagcxComm_t comm) {
  if (comm == NULL || comm->heteroComm == NULL)
    return flagcxInternalError;
  struct flagcxHeteroComm *heteroComm = comm->heteroComm;
  struct flagcxOneSideHandleInfo *info = heteroComm->stagingHandle;
  if (info == NULL)
    return flagcxSuccess;

  flagcxResult_t result = flagcxOneSideCleanupHandle(heteroComm, info, false);
  if (result != flagcxSuccess)
    return result;
  free(info);
  heteroComm->stagingHandle = NULL;
  return flagcxSuccess;
}

flagcxResult_t
flagcxOneSideBarrierRegister(const flagcxComm_t comm, void *recvComm,
                             void *buff, size_t size,
                             struct flagcxOneSideHandleInfo **outInfo) {
  if (comm == NULL || outInfo == NULL)
    return flagcxInvalidArgument;
  *outInfo = NULL;

  struct flagcxHeteroComm *heteroComm = comm->heteroComm;
  if (heteroComm == NULL || heteroComm->netAdaptor == NULL ||
      heteroComm->netAdaptor->regMr == NULL ||
      heteroComm->netAdaptor->getMrInfo == NULL)
    return flagcxNotSupported;

  if (comm->bootstrap == NULL)
    return flagcxNotSupported;

  struct flagcxNetAdaptor *net = heteroComm->netAdaptor;
  flagcxResult_t res = flagcxSuccess;
  void *mrHandle = NULL;
  struct flagcxNetMrInfo localMrInfo = {};
  struct flagcxOneSideHandleInfo *info = NULL;

  res = flagcxCalloc(&info, 1);
  res = flagcxOneSideConvergeStatus(comm->bootstrap, comm->rank, comm->nranks,
                                    res);
  if (res != flagcxSuccess)
    goto fail_mr;

  // Leaders (recvComm != NULL): register MR and extract keys
  if (recvComm != NULL && buff != NULL && size > 0) {
    void *regComm = recvComm;
    res = net->regMr(regComm, buff, size, FLAGCX_PTR_HOST,
                     FLAGCX_NET_MR_FLAG_FORCE_SO, &mrHandle);
    if (mrHandle != NULL) {
      info->localMrHandle = mrHandle;
      info->localRecvComm = recvComm;
      info->ownsLocalMr = 1;
    }
    if (res != flagcxSuccess || mrHandle == NULL) {
      INFO(FLAGCX_REG, "flagcxOneSideBarrierRegister: regMr failed, res=%d",
           res);
      res = flagcxNotSupported;
    } else {
      res = flagcxOneSideGetMrInfo(net, mrHandle, &localMrInfo);
    }
  }

  res = flagcxOneSideConvergeStatus(comm->bootstrap, comm->rank, comm->nranks,
                                    res);
  if (res != flagcxSuccess)
    goto fail_mr;

  // ALL ranks: allocate info, populate own entry, AllGather
  {
    int nranks = comm->nranks;
    int myRank = comm->rank;
    void *localBuffer = mrHandle == NULL ? NULL : buff;
    size_t localSize = mrHandle == NULL ? 0 : size;
    flagcxResult_t exchangeResult =
        flagcxOneSideExchangeMrInfo(comm->bootstrap, myRank, nranks,
                                    localBuffer, localSize, &localMrInfo, info);
    res = flagcxOneSideConvergeStatus(comm->bootstrap, myRank, nranks,
                                      exchangeResult);
    if (res != flagcxSuccess)
      goto fail_mr;

    INFO(FLAGCX_REG,
         "Barrier register allgather results (rank %d, nranks %d):", myRank,
         nranks);
    for (int i = 0; i < nranks; i++) {
      INFO(FLAGCX_REG, "  Rank %d: base_va=0x%lx, size=%zu, keys=%u", i,
           info->baseVas[i], info->regionSizes[i], info->mrInfos[i].nKeys);
    }
  }

  *outInfo = info;
  return flagcxSuccess;

fail_mr:
  if (info) {
    flagcxResult_t cleanupResult =
        flagcxOneSideCleanupHandle(heteroComm, info, false);
    if (cleanupResult != flagcxSuccess && heteroComm != NULL) {
      flagcxOneSideRetainCleanup(heteroComm, info);
      info = NULL;
    }
    free(info);
  }
  return res;
}

flagcxResult_t
flagcxOneSideBarrierDeregister(const flagcxComm_t comm,
                               struct flagcxOneSideHandleInfo *info) {
  if (info == NULL)
    return flagcxSuccess;
  if (comm == NULL)
    return flagcxInternalError;

  struct flagcxHeteroComm *heteroComm = comm->heteroComm;
  flagcxResult_t result = flagcxOneSideCleanupHandle(heteroComm, info, false);
  if (result != flagcxSuccess)
    return result;
  free(info);
  return flagcxSuccess;
}

static flagcxResult_t
flagcxValidateMemoryRange(void *buff, size_t size,
                          flagcxMemAllocator_t allocator) {
  if (allocator != flagcxMemCCL && allocator != flagcxMemSHMEM) {
    WARN("Invalid allocator %d for buffer registration.", (int)allocator);
    return flagcxInvalidArgument;
  }

  flagcxMemAllocationInfo info;
  flagcxResult_t res = globalMemAllocRegistry.findRange(buff, 1, &info);
  if (res == flagcxSuccess) {
    if (globalMemAllocRegistry.findRange(buff, size, &info) != flagcxSuccess) {
      WARN("Registration range exceeds its flagcxMemAlloc allocation.");
      return flagcxInvalidUsage;
    }
    if (info.allocator != allocator) {
      WARN("Registration allocator mismatch (requested %d, recorded %d).",
           (int)allocator, (int)info.allocator);
      return flagcxInvalidUsage;
    }
    if (allocator == flagcxMemSHMEM &&
        info.backend != flagcxMemAllocBackendShmem) {
      WARN("SHMEM registration requires a SHMEM-backed allocation.");
      return flagcxInvalidUsage;
    }
    return flagcxSuccess;
  }
  if (res != flagcxInvalidUsage)
    return res;

  // CCL registration also accepts external user buffers. SHMEM Device API
  // memory must come from flagcxMemAlloc so symmetric-heap provenance and
  // exact bounds are known.
  if (allocator == flagcxMemSHMEM) {
    WARN("SHMEM registration requires flagcxMemAlloc(..., flagcxMemSHMEM).");
    return flagcxInvalidUsage;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxCommRegister(const flagcxComm_t comm, void *buff,
                                  size_t size, void **handle,
                                  flagcxMemAllocator_t allocator) {
  if (comm != nullptr) {
    FLAGCXCHECK(flagcxEnsureCommReady(comm));
  }

  if (buff == NULL || size == 0 || handle == nullptr) {
    WARN("Invalid buffer, size, or handle for buffer registration.");
    return flagcxInvalidArgument;
  }
  *handle = nullptr;
  FLAGCXCHECK(flagcxValidateMemoryRange(buff, size, allocator));

  // Step 1: Register in globalRegPool (both paths)
  // Key: heteroComm if available (p2p/net downstream use it), else homoComm
  // If comm is NULL, register in global pool only (GLOBAL_POOL_KEY)
  void *regKey = nullptr;
  if (comm != nullptr) {
    regKey =
        comm->heteroComm ? (void *)comm->heteroComm : (void *)comm->homoComm;
  }
  FLAGCXCHECK(globalRegPool.registerBuffer(regKey, buff, size));
  flagcxRegItem *regItem = globalRegPool.getItem(regKey, buff);
  if (regItem == nullptr) {
    WARN("flagcxCommRegister: globalRegPool did not return a registration");
    return flagcxInternalError;
  }

  *handle = reinterpret_cast<void *>(regItem);

  // Null comm: pool-only registration, skip backend steps
  if (comm == nullptr) {
    return flagcxSuccess;
  }

  // SHMEM path: buffer is in the SHMEM symmetric heap,
  // no IPC handles or MR registration needed.
  if (allocator == flagcxMemSHMEM) {
    return flagcxSuccess;
  }

  uintptr_t thisCommKey = reinterpret_cast<uintptr_t>(regKey);

  flagcxResult_t res = flagcxSuccess;

  // Step 2: Homo path — backend CCL registration. NCCL handles IPC/VMM
  // internally via ncclCommRegister, so skip the hetero one-sided MR path.
  if (useHomoComm(comm) && !useHeteroComm()) {
    // Re-registration: this comm already completed homo backend init
    if (regItem->homoRegHandles.count(thisCommKey)) {
      return flagcxSuccess;
    }
    void *homoHandle = nullptr;
    res = cclAdaptors[flagcxCCLAdaptorDevice]->commRegister(
        comm->homoComm, buff, size, &homoHandle);
    if (res != flagcxSuccess)
      goto fail;
    regItem->homoRegHandles[thisCommKey] = homoHandle;
    return flagcxSuccess;
  }

  // Step 3: One-sided MR registration (hetero path only). IPC handles are
  // exported lazily by each IPC consumer from the allocation base; a page
  // registration item cannot safely own a single allocation handle.
  {
    flagcxResult_t regRes = flagcxOneSideRegisterInternal(
        comm->heteroComm, buff, size,
        flagcxOneSideRegistryRangeIsVmm(buff, size));
    if (regRes != flagcxSuccess) {
      INFO(FLAGCX_REG, "flagcxCommRegister: one-sided register skipped (%d)",
           regRes);
    }
  }

  return flagcxSuccess;

fail:
  // Undo Step 2a
  if (useHomoComm(comm) && !useHeteroComm()) {
    auto it = regItem->homoRegHandles.find(thisCommKey);
    if (it != regItem->homoRegHandles.end()) {
      cclAdaptors[flagcxCCLAdaptorDevice]->commDeregister(comm->homoComm,
                                                          it->second);
      regItem->homoRegHandles.erase(it);
    }
  }
  // Undo Step 1
  globalRegPool.deregisterBuffer(regKey, regItem);
  *handle = nullptr;
  return res;
}

flagcxResult_t flagcxCommDeregister(const flagcxComm_t comm, void *handle,
                                    flagcxMemAllocator_t allocator) {
  if (comm != nullptr) {
    FLAGCXCHECK(flagcxEnsureCommReady(comm));
  }
  if (handle == nullptr)
    return flagcxSuccess;
  flagcxRegItem *regItem = reinterpret_cast<flagcxRegItem *>(handle);

  // Null comm: only valid if no backend handles exist on this item
  // AND the item is not mapped under any comm-specific key
  if (comm == nullptr) {
    if (!regItem->homoRegHandles.empty() || !regItem->handles.empty()) {
      WARN("flagcxCommDeregister: comm is nullptr but handle has backend "
           "registrations that require a valid comm to clean up");
      return flagcxInvalidArgument;
    }
    // Check if item is mapped under any non-global commKey
    auto &globalMap = globalRegPool.getGlobalMap();
    for (auto &entry : globalMap) {
      if (entry.first == flagcxRegPool::GLOBAL_POOL_KEY)
        continue;
      if (entry.second.find(regItem->beginAddr) != entry.second.end()) {
        WARN("flagcxCommDeregister: comm is nullptr but handle has "
             "comm-specific regMap entries that require a valid comm");
        return flagcxInvalidArgument;
      }
    }
    globalRegPool.deregisterBuffer(nullptr, handle);
    return flagcxSuccess;
  }

  void *regKey =
      comm->heteroComm ? (void *)comm->heteroComm : (void *)comm->homoComm;

  // Backend-specific deregistration (homo path)
  uintptr_t thisCommKey = reinterpret_cast<uintptr_t>(regKey);
  if (useHomoComm(comm) && !useHeteroComm()) {
    auto it = regItem->homoRegHandles.find(thisCommKey);
    if (it != regItem->homoRegHandles.end()) {
      cclAdaptors[flagcxCCLAdaptorDevice]->commDeregister(comm->homoComm,
                                                          it->second);
      regItem->homoRegHandles.erase(it);
    }
  }

  // Remove this comm's net/p2p handles from the regItem
  globalRegPool.removeRegItemNetHandles(regKey, regItem);
  globalRegPool.removeRegItemP2pHandles(regKey, regItem);

  // Clean up globalRegPool (refCount--, page mappings, item removal at 0)
  globalRegPool.deregisterBuffer(regKey, handle);
  return flagcxSuccess;
}

static flagcxResult_t
flagcxCommWindowDeregisterInternal(flagcxComm_t comm, flagcxWindow_t win,
                                   flagcxMemAllocator_t allocator,
                                   flagcxSymCleanupMode cleanupMode);

static flagcxResult_t flagcxRollbackWindowRegistration(
    flagcxComm_t comm, flagcxWindow_t *win, flagcxMemAllocator_t allocator,
    flagcxSymCleanupMode cleanupMode, flagcxResult_t originalResult,
    const char *phase) {
  if (win == nullptr || *win == nullptr)
    return originalResult;

  flagcxWindow_t failedWin = *win;
  // The lower-level registration path already attempted rollback and retained
  // this object when provider teardown failed. Do not immediately retry that
  // teardown here: the non-null output is the caller's cleanup token and must
  // remain valid until an explicit deregistration retry.
  if (failedWin->defaultBase != nullptr &&
      failedWin->defaultBase->state == flagcxSymWindowCleanupRequired) {
    flagcxResult_t retainResult = flagcxSymRetainPendingCleanup(
        comm != nullptr ? comm->heteroComm : nullptr, failedWin);
    return retainResult == flagcxSuccess ? originalResult : retainResult;
  }

  flagcxResult_t cleanupResult = flagcxCommWindowDeregisterInternal(
      comm, failedWin, allocator, cleanupMode);
  if (cleanupResult == flagcxSuccess) {
    *win = nullptr;
    return originalResult;
  }

  // A non-null output on failure is an explicit cleanup token. Keep it out of
  // the published list, retain it for communicator-destroy fallback, and let
  // the caller retry flagcxCommWindowDeregister with the same handle.
  WARN("flagcxCommWindowRegister: %s rollback returned %d; returning cleanup "
       "token %p",
       phase, (int)cleanupResult, failedWin);
  if (comm != nullptr && comm->heteroComm != nullptr &&
      failedWin->defaultBase != nullptr) {
    FLAGCXCHECK(flagcxSymRetainPendingCleanup(comm->heteroComm, failedWin));
  }
  return cleanupResult;
}

flagcxResult_t flagcxCommWindowRegister(flagcxComm_t comm, void *buff,
                                        size_t size, flagcxWindow_t *win,
                                        int winFlags,
                                        flagcxMemAllocator_t allocator) {
  if (buff == nullptr || size == 0 || win == nullptr || *win != nullptr) {
    return flagcxInvalidArgument;
  }
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  FLAGCXCHECK(flagcxValidateMemoryRange(buff, size, allocator));
  // SHMEM path: buffer is already in the SHMEM symmetric heap, so no
  // additional communicator window registration is needed.
  if (allocator == flagcxMemSHMEM) {
    *win = nullptr;
    return flagcxSuccess;
  }
  if (useHomoComm(comm) && !useHeteroComm()) {
    FLAGCXCHECK(flagcxCalloc(win, 1));
    flagcxResult_t res =
        cclAdaptors[flagcxCCLAdaptorDevice]->commWindowRegister(
            comm->homoComm, buff, size, &(*win)->vendorBase, winFlags);
    if (res == flagcxSuccess) {
      (*win)->winFlags = winFlags;
      return flagcxSuccess;
    }
    if (res != flagcxNotSupported) {
      free(*win);
      *win = nullptr;
      return res;
    }
    WARN("flagcxCommWindowRegister: backend returned %d, window not available, "
         "falling back",
         res);
    // Free any vendorBase the backend may have partially allocated
    if ((*win)->vendorBase != nullptr) {
      cclAdaptors[flagcxCCLAdaptorDevice]->commWindowDeregister(
          comm->homoComm, (*win)->vendorBase);
      (*win)->vendorBase = nullptr;
    }
    free(*win);
    *win = nullptr;
  }
  // Non-homo or homo-fallback: use symmetric heap path
  if ((winFlags & FLAGCX_WIN_COLL_SYMMETRIC) && comm->heteroComm != nullptr) {
    flagcxResult_t res =
        flagcxSymWindowRegister(comm->heteroComm, buff, size, win, winFlags);

    // Window construction is collective. Use the allocation-free convergence
    // path so even a rank-local ENOMEM cannot strand peers before IPC setup.
    flagcxResult_t commonWindowRes = flagcxSuccess;
    flagcxResult_t statusGatherRes =
        flagcxSymConvergeStatus(comm->heteroComm, res, &commonWindowRes);
    if (statusGatherRes != flagcxSuccess)
      commonWindowRes = statusGatherRes;
    if (commonWindowRes != flagcxSuccess) {
      return flagcxRollbackWindowRegistration(
          comm, win, allocator,
          statusGatherRes == flagcxSuccess ? flagcxSymCleanupCollective
                                           : flagcxSymCleanupLocal,
          commonWindowRes, "collective registration failure");
    }

    // Only ordinary device allocations may use legacy IPC. A VMM allocation
    // with no flat mapping is NET-only; exporting it through the legacy IPC
    // path is unsupported and can split the local collective.
    if (res == flagcxSuccess && *win != NULL && (*win)->defaultBase != NULL &&
        flagcxSymWindowCanUseLegacyIpc((*win)->defaultBase)) {
      int ipcSlot = buildIpcPeerPointers(comm, buff, size);
      flagcxResult_t commonIpcStatus = flagcxSuccess;
      flagcxResult_t gatherRes = flagcxSymConvergeStatus(
          comm->heteroComm, ipcSlot >= 0 ? flagcxSuccess : flagcxNotSupported,
          &commonIpcStatus);
      bool allIpcReady =
          gatherRes == flagcxSuccess && commonIpcStatus == flagcxSuccess;

      if (allIpcReady) {
        (*win)->defaultBase->ipcSlot = ipcSlot;
      } else {
        if (ipcSlot >= 0)
          releaseIpcTableSlot(comm, ipcSlot);
        if (gatherRes != flagcxSuccess) {
          return flagcxRollbackWindowRegistration(
              comm, win, allocator, flagcxSymCleanupLocal, gatherRes,
              "IPC collective failure");
        }

        // No local window is published without a real data path. Remote
        // communicators already acquired an MR in sym_heap; local-only
        // communicators acquire one lazily here so the normal IPC path does not
        // pay full-mesh/MR setup cost.
        flagcxResult_t fallbackCapability =
            flagcxParamIbDisable() ? flagcxNotSupported : flagcxSuccess;
        flagcxResult_t commonFallbackCapability = flagcxSuccess;
        flagcxResult_t fallbackConvergeResult = flagcxSymConvergeStatus(
            comm->heteroComm, fallbackCapability, &commonFallbackCapability);
        if (fallbackConvergeResult != flagcxSuccess) {
          return flagcxRollbackWindowRegistration(
              comm, win, allocator, flagcxSymCleanupLocal,
              fallbackConvergeResult, "NET fallback capability convergence");
        }
        flagcxResult_t fallbackResult = commonFallbackCapability;
        if (fallbackResult == flagcxSuccess &&
            !(*win)->defaultBase->hasNetworkMrRef) {
          fallbackResult = flagcxSymWindowEnsureNetworkMr(comm->heteroComm,
                                                          (*win)->defaultBase);
        }
        flagcxResult_t commonFallbackResult = flagcxSuccess;
        fallbackConvergeResult = flagcxSymConvergeStatus(
            comm->heteroComm, fallbackResult, &commonFallbackResult);
        if (fallbackConvergeResult != flagcxSuccess) {
          return flagcxRollbackWindowRegistration(
              comm, win, allocator, flagcxSymCleanupLocal,
              fallbackConvergeResult, "NET fallback result convergence");
        }
        if (commonFallbackResult != flagcxSuccess) {
          INFO(FLAGCX_REG,
               "flagcxCommWindowRegister: IPC mapping unavailable for %p and "
               "no NET MR fallback exists (res=%d)",
               buff, (int)commonFallbackResult);
          return flagcxRollbackWindowRegistration(
              comm, win, allocator, flagcxSymCleanupCollective,
              commonFallbackResult, "NET MR fallback failure");
        }
        INFO(FLAGCX_REG,
             "flagcxCommWindowRegister: IPC mapping unavailable for %p; using "
             "registered NET MR fallback at index %d",
             buff, (*win)->defaultBase->mrIndex);
      }
    }

    // Route readiness is a publish invariant, not an implication of an enabled
    // environment variable. Converge the actual window state before any rank
    // makes it visible through comm->symWindows.
    flagcxResult_t routeResult = flagcxSymWindowValidateDataRoutes(
        comm->heteroComm, (*win)->defaultBase);
    flagcxResult_t commonRouteResult = flagcxSuccess;
    flagcxResult_t routeConvergeResult = flagcxSymConvergeStatus(
        comm->heteroComm, routeResult, &commonRouteResult);
    if (routeConvergeResult != flagcxSuccess) {
      return flagcxRollbackWindowRegistration(
          comm, win, allocator, flagcxSymCleanupLocal, routeConvergeResult,
          "route validation convergence");
    }
    if (commonRouteResult != flagcxSuccess) {
      return flagcxRollbackWindowRegistration(
          comm, win, allocator, flagcxSymCleanupCollective, commonRouteResult,
          "route validation failure");
    }

    // Nothing becomes visible through comm->symWindows until VMM/IPC and the
    // required network MR have all converged successfully.
    flagcxResult_t publishRes = flagcxSymWindowPublish(comm->heteroComm, *win);
    if (publishRes != flagcxSuccess) {
      return flagcxRollbackWindowRegistration(comm, win, allocator,
                                              flagcxSymCleanupCollective,
                                              publishRes, "publish failure");
    }
    return res;
  }
  *win = nullptr;
  return flagcxSuccess;
}

static flagcxResult_t
flagcxCommWindowDeregisterInternal(flagcxComm_t comm, flagcxWindow_t win,
                                   flagcxMemAllocator_t allocator,
                                   flagcxSymCleanupMode cleanupMode) {
  if (win == nullptr) {
    return flagcxSuccess;
  }
  if (allocator == flagcxMemSHMEM) {
    return flagcxSuccess;
  }
  FLAGCXCHECK(flagcxEnsureCommReady(comm));

  // Use isSymmetricDefault flag to determine ownership:
  // - If backend owns it (vendorBase != nullptr && !isSymmetricDefault),
  //   deregister via backend only
  // - Otherwise (hetero path or homo fallback), deregister via sym_heap
  if (useHomoComm(comm) && !useHeteroComm() && win->vendorBase != nullptr &&
      !win->isSymmetricDefault) {
    // Backend owns this window — deregister via backend only
    flagcxResult_t res =
        cclAdaptors[flagcxCCLAdaptorDevice]->commWindowDeregister(
            comm->homoComm, win->vendorBase);
    free(win);
    return res; // propagate real errors, don't fall through
  }
  // Sym-heap owns this window (hetero path, or homo fallback)
  flagcxHeteroComm_t hetero = comm->heteroComm;
  flagcxSymWindow_t sym = win->defaultBase;
  int ipcSlot = sym != NULL ? sym->ipcSlot : -1;
  int mrIndex = sym != NULL ? sym->mrIndex : -1;
  bool hasNetworkMrRef = sym != NULL && sym->hasNetworkMrRef;
  if (sym != NULL)
    sym->state = flagcxSymWindowCleanupRequired;

  // A registration rollback may have retained an unpublished MR object rather
  // than a published mrIndex. Retry that exact object before releasing VMM
  // mappings or the allocation lease. All ranks carry the flag when the
  // rollback result converged, even though only the rank whose deregMr failed
  // has an entry in pendingOneSideCleanup.
  if (sym != NULL && sym->hasPendingNetworkCleanup) {
    flagcxResult_t pendingResult = flagcxOneSideRetryPendingCleanup(hetero);
    flagcxResult_t commonPendingResult = pendingResult;
    if (cleanupMode == flagcxSymCleanupCollective) {
      FLAGCXCHECK(
          flagcxSymConvergeStatus(hetero, pendingResult, &commonPendingResult));
    }
    if (commonPendingResult != flagcxSuccess)
      return commonPendingResult;
    sym->hasPendingNetworkCleanup = false;
  }

  // Keep a registered VA mapped until deregMr succeeds. The handle cleanup
  // helper preserves its MR and connection ownership when deregistration
  // fails, allowing this window operation to be retried unchanged. Every rank
  // participates in status convergence on every attempt: after a partial
  // failure, ranks that already released their local MR must still rendezvous
  // with the rank retrying deregMr.
  flagcxResult_t mrResult = flagcxSuccess;
  if (hasNetworkMrRef) {
    mrResult = flagcxOneSideDeregisterInternal(hetero, mrIndex);
    if (mrResult == flagcxSuccess) {
      sym->hasNetworkMrRef = false;
      sym->mrIndex = -1;
      sym->mrBase = 0;
    }
  }
  flagcxResult_t commonMrResult = flagcxSuccess;
  if (cleanupMode == flagcxSymCleanupCollective) {
    FLAGCXCHECK(flagcxSymConvergeStatus(hetero, mrResult, &commonMrResult));
  } else {
    commonMrResult = mrResult;
  }
  if (commonMrResult != flagcxSuccess)
    return commonMrResult;

  flagcxResult_t result = flagcxSymWindowDeregister(hetero, win, cleanupMode);
  if (result != flagcxSuccess)
    return result;

  // Drop derived references only after fallible VMM cleanup succeeds. This
  // preserves a complete, retryable window when teardown reports an error.
  if (hetero != NULL && sym != NULL) {
    if (hetero->rmaProxy != NULL && hetero->rmaProxy->ipcState != NULL)
      flagcxHeteroRmaIpcDestroy(hetero);
  }
  if (ipcSlot >= 0)
    releaseIpcTableSlot(comm, ipcSlot);
  return flagcxSuccess;
}

flagcxResult_t flagcxCommWindowDeregister(flagcxComm_t comm, flagcxWindow_t win,
                                          flagcxMemAllocator_t allocator) {
  return flagcxCommWindowDeregisterInternal(comm, win, allocator,
                                            flagcxSymCleanupCollective);
}

flagcxResult_t flagcxIsHomoComm(flagcxComm_t comm, int *isHomo) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHomoComm(comm)) {
    *isHomo = 1;
  } else {
    *isHomo = 0;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxGetVersion(int *version) {
  // TODO: implement a method to retrieve global verison
  return flagcxHeteroGetVersion(version);
}

flagcxResult_t flagcxGetUniqueId(flagcxUniqueId_t uniqueId) {
  if (uniqueId == NULL) {
    WARN("flagcxGetUniqueId: uniqueId is NULL");
    return flagcxInvalidArgument;
  }

  // Init bootstrap net
  FLAGCXCHECK(bootstrapNetInit());

  // Init uniqueId using bootstrap
  struct flagcxBootstrapHandle handle;
  FLAGCXCHECK(bootstrapGetUniqueId(&handle));
  // flagcxUniqueId and bootstrapHandle don't have the same size and alignment
  // reset to 0 to avoid undefined data
  memset((void *)uniqueId, 0, sizeof(*uniqueId));
  // copy to avoid alignment mismatch
  memcpy((void *)uniqueId, &handle, sizeof(handle));
  return flagcxSuccess;
}

const char *flagcxGetErrorString(flagcxResult_t result) {
  // TODO: implement a method to retrieve error string
  return "Not implemented.";
}

const char *flagcxGetLastError(flagcxComm_t comm) {
  // TODO: implement a method to retrieve last error string
  if (comm == NULL) {
    return "Undefined: flagcxComm is not fully initialized.";
  }
  if (useHomoComm(comm)) {
    return cclAdaptors[flagcxCCLAdaptorDevice]->getLastError(comm->homoComm);
  }
  return "Not implemented.";
}

// ---- Custom op DevComm state init/destroy ----

FLAGCX_PARAM(CustomOpEnable, "CUSTOM_OP_ENABLE", 0);

#define FLAGCX_CUSTOM_OP_STAGED_BUFFER_SIZE (8 * 1024 * 1024)

// Forward declaration of custom allreduce implementation
// Defined in kernels/custom_allreduce.cu (NVIDIA only).
// Weak symbol: resolves to NULL when not linked (non-NVIDIA or no kernel
// build).
extern "C" __attribute__((weak)) flagcxResult_t
flagcxCustomAllReduceImpl(const void *sendbuff, void *recvbuff, size_t count,
                          flagcxDataType_t datatype, flagcxRedOp_t op,
                          flagcxComm_t comm, flagcxStream_t stream);

static flagcxResult_t flagcxDevCommStateDeregisterWindow(
    flagcxComm_t comm, flagcxWindow_t *window,
    flagcxSymCleanupMode cleanupMode = flagcxSymCleanupCollective) {
  if (window == nullptr || *window == nullptr)
    return flagcxSuccess;

  flagcxWindow_t current = *window;
  flagcxResult_t result = flagcxSuccess;
  if (current->vendorBase != nullptr) {
    result = cclAdaptors[flagcxCCLAdaptorDevice]->commWindowDeregister(
        comm->homoComm, current->vendorBase);
    if (result == flagcxSuccess)
      free(current);
  } else if (current->isSymmetricDefault) {
    // Use the public ownership path so a staged window's network MR is
    // released before its VMM/IPC mapping. Failures leave *window intact.
    result = flagcxCommWindowDeregisterInternal(comm, current, flagcxMemCCL,
                                                cleanupMode);
  } else {
    free(current);
  }
  if (result == flagcxSuccess)
    *window = nullptr;
  return result;
}

static flagcxResult_t flagcxDevCommStateFreeBuffer(void **buffer,
                                                   bool usesGdrAllocator) {
  if (buffer == nullptr || *buffer == nullptr)
    return flagcxSuccess;
  flagcxResult_t result =
      usesGdrAllocator ? deviceAdaptor->gdrMemFree(*buffer, nullptr)
                       : cclAdaptors[flagcxCCLAdaptorDevice]->memFree(*buffer);
  if (result == flagcxSuccess)
    *buffer = nullptr;
  return result;
}

static flagcxResult_t flagcxDevCommStatePublishWindow(flagcxComm_t comm,
                                                      flagcxWindow_t window,
                                                      flagcxDevMem_t devMem) {
  if (window == nullptr || !window->isSymmetricDefault)
    return flagcxSuccess;
  if (comm == nullptr || comm->heteroComm == nullptr ||
      window->defaultBase == nullptr || devMem == nullptr)
    return flagcxInvalidArgument;

  flagcxSymWindow_t d = window->defaultBase;
  // DevMem creates the IPC mapping for a non-VMM staged allocation. Transfer
  // its ownership before route validation, but do not publish the window until
  // every rank reports a usable route.
  if (flagcxSymWindowCanUseLegacyIpc(d) && d->ipcSlot < 0 &&
      devMem->ipcIndex >= 0) {
    d->ipcSlot = devMem->ipcIndex;
    devMem->ipcIndex = -1;
  }

  flagcxResult_t localResult =
      flagcxSymWindowValidateDataRoutes(comm->heteroComm, d);
  flagcxResult_t commonResult = flagcxSuccess;
  flagcxResult_t convergeResult =
      flagcxSymConvergeStatus(comm->heteroComm, localResult, &commonResult);
  if (convergeResult != flagcxSuccess)
    return convergeResult;
  if (commonResult != flagcxSuccess)
    return commonResult;

  localResult = flagcxSymWindowPublish(comm->heteroComm, window);
  convergeResult =
      flagcxSymConvergeStatus(comm->heteroComm, localResult, &commonResult);
  return convergeResult == flagcxSuccess ? commonResult : convergeResult;
}

static flagcxResult_t flagcxDevCommStateInit(flagcxComm_t comm) {
  if (!flagcxParamCustomOpEnable() || flagcxCustomAllReduceImpl == nullptr) {
    comm->devCommState = nullptr;
    return flagcxSuccess;
  }

  flagcxDevCommState *state;
  FLAGCXCHECK(flagcxCalloc(&state, 1));

  // 1. Auto-detect requirements via adaptor; fall back to Default path if
  //    adaptor doesn't support DevComm (e.g. NCCL < 2.28).
  flagcxDevCommRequirements reqs = FLAGCX_DEV_COMM_REQUIREMENTS_INITIALIZER;
  bool vendorReqs = false;
  flagcxResult_t res = flagcxSuccess;
  if (cclAdaptors[flagcxCCLAdaptorDevice]->devCommReqsInit != NULL) {
    res = cclAdaptors[flagcxCCLAdaptorDevice]->devCommReqsInit(comm->homoComm,
                                                               &reqs);
    if (res == flagcxSuccess) {
      vendorReqs = true;
    } else if (res != flagcxNotSupported) {
      free(state);
      comm->devCommState = nullptr;
      INFO(FLAGCX_INIT, "Custom allreduce: DevComm requirements init failed, "
                        "disabled");
      return flagcxSuccess; // non-fatal
    }
  }
  if (!vendorReqs) {
    // Default path: IPC barriers are sufficient for LSA allreduce
    reqs.intraBarrierCount = FLAGCX_DEVICE_CTA_COUNT;
    // Check if multicast (NVLS) is available via adaptor
    int mcSupported = 0;
    if (deviceAdaptor->symMulticastSupported)
      deviceAdaptor->symMulticastSupported(&mcSupported);
    if (mcSupported)
      reqs.intraMulticast = true;
    INFO(FLAGCX_INIT, "Custom allreduce: using Default path (%s)",
         reqs.intraMulticast ? "multicast + LSA" : "LSA only");
  }

  // Record capability flags
  state->hasMulticast = reqs.intraMulticast;

  // 2. Create DevComm
  res = flagcxDevCommCreate(comm, &reqs, &state->devComm);
  if (res != flagcxSuccess) {
    free(state);
    comm->devCommState = nullptr;
    INFO(FLAGCX_INIT, "Custom allreduce: DevComm creation failed, disabled");
    return flagcxSuccess; // non-fatal
  }

  // 3. Allocate staged buffers
  state->stagedBuffSize = FLAGCX_CUSTOM_OP_STAGED_BUFFER_SIZE;

  // On the default path with multicast, allocate through the native GDR
  // owner. Whether that allocation is actually VMM-backed is captured
  // separately because gdrMemAlloc uses ordinary device memory when VMM is
  // disabled.
  bool useGdrAllocator = false;
#ifndef FLAGCX_DEVICE_API_VENDOR
  {
    int mcSupported = 0;
    if (deviceAdaptor->symMulticastSupported)
      deviceAdaptor->symMulticastSupported(&mcSupported);
    useGdrAllocator = mcSupported && deviceAdaptor->gdrMemAlloc != nullptr;
  }
#endif

  if (useGdrAllocator) {
    // Record ownership before the first fallible allocation so rollback uses
    // the matching deallocator even if only one staged buffer is created.
    state->stagedUsesGdrAllocator = true;
    state->stagedAllocationIsVmm = flagcxDeviceAdaptorNativeAllocIsVmm(
        deviceAdaptor, flagcxParamVmmEnable());
    FLAGCXCHECKGOTO(deviceAdaptor->gdrMemAlloc(&state->sendStagedBuff,
                                               state->stagedBuffSize, nullptr),
                    res, fail);
    FLAGCXCHECKGOTO(deviceAdaptor->gdrMemAlloc(&state->recvStagedBuff,
                                               state->stagedBuffSize, nullptr),
                    res, fail);
  } else {
    FLAGCXCHECKGOTO(cclAdaptors[flagcxCCLAdaptorDevice]->memAlloc(
                        &state->sendStagedBuff, state->stagedBuffSize),
                    res, fail);
    FLAGCXCHECKGOTO(cclAdaptors[flagcxCCLAdaptorDevice]->memAlloc(
                        &state->recvStagedBuff, state->stagedBuffSize),
                    res, fail);
  }

  // 4. Register windows (symmetric) — skip if adaptor doesn't support it
  if (cclAdaptors[flagcxCCLAdaptorDevice]->commWindowRegister != NULL) {
    FLAGCXCHECKGOTO(flagcxCalloc(&state->sendStagedWin, 1), res, fail);
    res = cclAdaptors[flagcxCCLAdaptorDevice]->commWindowRegister(
        comm->homoComm, state->sendStagedBuff, state->stagedBuffSize,
        &state->sendStagedWin->vendorBase, FLAGCX_WIN_COLL_SYMMETRIC);
    if (res != flagcxSuccess && res != flagcxNotSupported)
      goto fail;
    if (res == flagcxSuccess) {
      state->sendStagedWin->winFlags = FLAGCX_WIN_COLL_SYMMETRIC;
      FLAGCXCHECKGOTO(flagcxCalloc(&state->recvStagedWin, 1), res, fail);
      res = cclAdaptors[flagcxCCLAdaptorDevice]->commWindowRegister(
          comm->homoComm, state->recvStagedBuff, state->stagedBuffSize,
          &state->recvStagedWin->vendorBase, FLAGCX_WIN_COLL_SYMMETRIC);
      if (res != flagcxSuccess && res != flagcxNotSupported)
        goto fail;
      if (res == flagcxSuccess) {
        state->recvStagedWin->winFlags = FLAGCX_WIN_COLL_SYMMETRIC;
      } else {
        free(state->recvStagedWin);
        state->recvStagedWin = nullptr;
      }
    } else {
      free(state->sendStagedWin);
      state->sendStagedWin = nullptr;
    }
  }

  // Default path: if vendor didn't provide windows and multicast is supported,
  // register staged buffers via sym heap path
  if (state->sendStagedWin == nullptr && state->recvStagedWin == nullptr) {
    int mcSupported = 0;
    if (deviceAdaptor->symMulticastSupported)
      deviceAdaptor->symMulticastSupported(&mcSupported);
    if (mcSupported && comm->heteroComm != nullptr) {
      // Register send staged buffer
      FLAGCXCHECKGOTO(flagcxSymWindowRegisterInternal(
                          comm->heteroComm, state->sendStagedBuff,
                          state->stagedBuffSize, &state->sendStagedWin,
                          FLAGCX_WIN_COLL_SYMMETRIC,
                          state->stagedAllocationIsVmm),
                      res, fail);
      // Register recv staged buffer
      FLAGCXCHECKGOTO(flagcxSymWindowRegisterInternal(
                          comm->heteroComm, state->recvStagedBuff,
                          state->stagedBuffSize, &state->recvStagedWin,
                          FLAGCX_WIN_COLL_SYMMETRIC,
                          state->stagedAllocationIsVmm),
                      res, fail);
    }
  }

  // 5. Create DevMem (for kernel parameters)
  res = flagcxDevMemCreate(comm, state->sendStagedBuff, state->stagedBuffSize,
                           state->sendStagedWin, &state->sendStagedMem);
  if (res != flagcxSuccess)
    goto fail;
  res = flagcxDevMemCreate(comm, state->recvStagedBuff, state->stagedBuffSize,
                           state->recvStagedWin, &state->recvStagedMem);
  if (res != flagcxSuccess)
    goto fail;

  // The internal staged-window path deliberately bypasses the public window
  // registration wrapper. For a non-VMM allocation, DevMem creation above is
  // therefore the operation that establishes the IPC route. Transfer that
  // slot's ownership to the window before publishing it: the window remains a
  // valid route owner for its complete lifetime and DevMem teardown cannot
  // release the slot early (or release it a second time).
  if (state->sendStagedWin != nullptr &&
      state->sendStagedWin->isSymmetricDefault) {
    FLAGCXCHECKGOTO(flagcxDevCommStatePublishWindow(comm, state->sendStagedWin,
                                                    state->sendStagedMem),
                    res, fail);
  }
  if (state->recvStagedWin != nullptr &&
      state->recvStagedWin->isSymmetricDefault) {
    FLAGCXCHECKGOTO(flagcxDevCommStatePublishWindow(comm, state->recvStagedWin,
                                                    state->recvStagedMem),
                    res, fail);
  }

  // Verify multicast is actually available on the staged buffers
  if (state->hasMulticast && state->sendStagedWin != nullptr &&
      state->sendStagedWin->isSymmetricDefault) {
    flagcxSymWindow_t d = state->sendStagedWin->defaultBase;
    if (d == nullptr || d->mcBase == nullptr) {
      INFO(FLAGCX_INIT,
           "Custom allreduce: multicast bind failed, falling back to LSA");
      state->hasMulticast = false;
    }
  }

  // 6. Register custom op
  state->customAllReduce = flagcxCustomAllReduceImpl;
  state->initialized = true;
  comm->devCommState = state;

  INFO(FLAGCX_INIT, "Custom allreduce: enabled, staged buffer %zuMB",
       state->stagedBuffSize / (1024 * 1024));
  return flagcxSuccess;

fail:
  if (state->devComm) {
    flagcxResult_t result = flagcxDevCommDestroy(comm, state->devComm);
    if (result != flagcxSuccess) {
      // The backend still owns resources reachable through state->devComm.
      // Publish the partially initialized state so communicator teardown can
      // retry instead of leaking the entire ownership graph.
      comm->devCommState = state;
      return result;
    }
    state->devComm = nullptr;
  }
  if (state->recvStagedMem) {
    flagcxResult_t cleanupResult =
        flagcxDevMemDestroy(comm, state->recvStagedMem);
    if (cleanupResult != flagcxSuccess) {
      comm->devCommState = state;
      return cleanupResult;
    }
    state->recvStagedMem = nullptr;
  }
  if (state->sendStagedMem) {
    flagcxResult_t cleanupResult =
        flagcxDevMemDestroy(comm, state->sendStagedMem);
    if (cleanupResult != flagcxSuccess) {
      comm->devCommState = state;
      return cleanupResult;
    }
    state->sendStagedMem = nullptr;
  }
  {
    flagcxResult_t cleanupResult =
        flagcxDevCommStateDeregisterWindow(comm, &state->recvStagedWin);
    if (cleanupResult == flagcxSuccess)
      cleanupResult =
          flagcxDevCommStateDeregisterWindow(comm, &state->sendStagedWin);
    if (cleanupResult != flagcxSuccess) {
      // Do not free a staged buffer while a failed MR/VMM teardown still owns
      // it. Communicator teardown can retry through the published state.
      comm->devCommState = state;
      return cleanupResult;
    }
  }
  {
    flagcxResult_t cleanupResult = flagcxDevCommStateFreeBuffer(
        &state->recvStagedBuff, state->stagedUsesGdrAllocator);
    if (cleanupResult == flagcxSuccess)
      cleanupResult = flagcxDevCommStateFreeBuffer(
          &state->sendStagedBuff, state->stagedUsesGdrAllocator);
    if (cleanupResult != flagcxSuccess) {
      comm->devCommState = state;
      return cleanupResult;
    }
  }
  free(state);
  comm->devCommState = nullptr;
  INFO(FLAGCX_INIT, "Custom allreduce: init failed, disabled");
  return flagcxSuccess; // non-fatal
}

static flagcxResult_t flagcxDevCommStateDestroy(flagcxComm_t comm) {
  if (comm->devCommState == nullptr)
    return flagcxSuccess;

  auto *state = comm->devCommState;

  // Destroy DevComm first — vendor may reference windows/buffers internally
  if (state->devComm) {
    flagcxResult_t result = flagcxDevCommDestroy(comm, state->devComm);
    if (result != flagcxSuccess)
      return result;
    state->devComm = nullptr;
  }
  if (state->sendStagedMem) {
    flagcxResult_t result = flagcxDevMemDestroy(comm, state->sendStagedMem);
    if (result != flagcxSuccess)
      return result;
    state->sendStagedMem = nullptr;
  }
  if (state->recvStagedMem) {
    flagcxResult_t result = flagcxDevMemDestroy(comm, state->recvStagedMem);
    if (result != flagcxSuccess)
      return result;
    state->recvStagedMem = nullptr;
  }
  flagcxResult_t result = flagcxDevCommStateDeregisterWindow(
      comm, &state->sendStagedWin, flagcxSymCleanupLocal);
  if (result != flagcxSuccess)
    return result;
  result = flagcxDevCommStateDeregisterWindow(comm, &state->recvStagedWin,
                                              flagcxSymCleanupLocal);
  if (result != flagcxSuccess)
    return result;
  result = flagcxDevCommStateFreeBuffer(&state->sendStagedBuff,
                                        state->stagedUsesGdrAllocator);
  if (result != flagcxSuccess)
    return result;
  result = flagcxDevCommStateFreeBuffer(&state->recvStagedBuff,
                                        state->stagedUsesGdrAllocator);
  if (result != flagcxSuccess)
    return result;
  free(state);
  comm->devCommState = nullptr;
  return flagcxSuccess;
}

static flagcxResult_t flagcxCollectUniqueIdResult(struct bootstrapState *state,
                                                  int rank, int nranks,
                                                  flagcxResult_t localResult) {
  std::vector<flagcxResult_t> resultData(nranks, flagcxSuccess);
  resultData[rank] = localResult;
  FLAGCXCHECK(bootstrapCollAllGather(state, (void *)resultData.data(),
                                     sizeof(flagcxResult_t)));
  FLAGCXCHECK(bootstrapCollBarrier(state, rank, nranks, 0));

  for (int peer = 0; peer < nranks; peer++) {
    if (resultData[peer] != flagcxSuccess) {
      return resultData[peer];
    }
  }
  return flagcxSuccess;
}

static flagcxResult_t flagcxBuildHomoRankList(flagcxComm_t comm,
                                              std::vector<int> &globalRanks) {
  globalRanks.assign(comm->homoRanks, -1);
  int clusterId = comm->clusterIds[comm->rank];
  for (int globalRank = 0; globalRank < comm->nranks; globalRank++) {
    if (comm->clusterIds[globalRank] != clusterId) {
      continue;
    }
    int homoRank = comm->globalRank2HomoRank[globalRank];
    if (homoRank < 0 || homoRank >= comm->homoRanks ||
        globalRanks[homoRank] != -1) {
      return flagcxInternalError;
    }
    globalRanks[homoRank] = globalRank;
  }
  for (int globalRank : globalRanks) {
    if (globalRank == -1) {
      return flagcxInternalError;
    }
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxHomoCommInit(struct bootstrapState *state,
                                  flagcxComm_t comm,
                                  flagcxInnerComm_t *homoComm /*out*/) {
  int rank = comm->rank;
  int nranks = comm->nranks;
  flagcxInnerUniqueId commIdStorage = {};
  flagcxInnerUniqueId_t commId = &commIdStorage;
  std::vector<int> homoGlobalRanks;
  flagcxResult_t uniqueIdResult =
      flagcxBuildHomoRankList(comm, homoGlobalRanks);
  if (uniqueIdResult == flagcxSuccess && comm->homoRank == 0) {
    uniqueIdResult = cclAdaptors[flagcxCCLAdaptorDevice]->getUniqueId(&commId);
  }
  FLAGCXCHECK(flagcxCollectUniqueIdResult(state, rank, nranks, uniqueIdResult));

  FLAGCXCHECK(bootstrapCollSubgroupBroadcast(state, homoGlobalRanks.data(),
                                             comm->homoRank, comm->homoRanks, 0,
                                             (void *)commId, sizeof(*commId)));
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->commInitRank(
      homoComm, comm->homoRanks, commId, comm->homoRank, NULL));
  return flagcxSuccess;
}

flagcxResult_t flagcxCommInitRank(flagcxComm_t *comm, int nranks,
                                  flagcxUniqueId_t commId, int rank) {
  if (nranks < 1 || rank < 0 || rank >= nranks) {
    WARN("Invalid rank requested : %d/%d", rank, nranks);
    return flagcxInvalidArgument;
  }
  if (commId == NULL || comm == NULL) {
    WARN("flagcxCommInitRank: commId or comm is NULL");
    return flagcxInvalidArgument;
  }

  // Ensure device/CCL plugins are loaded (idempotent, ref-counted)
  flagcxDeviceAdaptorPluginInit();
  flagcxCCLAdaptorPluginInit();

  (*comm) = NULL;
  flagcxCalloc(comm, 1);
  (*comm)->rank = rank;
  (*comm)->nranks = nranks;
  (*comm)->nclusters = -1;
  (*comm)->homoRank = -1;
  (*comm)->homoRootRank = -1;
  (*comm)->homoRanks = -1;
  (*comm)->hasSingleRankHomoComm = -1;
  (*comm)->magic = 0;
  (*comm)->abortFlag = 0;
  (*comm)->bootstrap = NULL;
  (*comm)->localRank = 0;
  (*comm)->localRanks = 1;
  (*comm)->localRankToRank = NULL;
  (*comm)->hostComm = NULL;
  (*comm)->homoComm = NULL;
  (*comm)->heteroComm = NULL;
  (*comm)->clusterIds = NULL;
  (*comm)->clusterSizes = NULL;
  (*comm)->clusterInterRanks = NULL;
  (*comm)->globalRank2HomoRank = NULL;
  (*comm)->commType = flagcxCommunicatorUnknown;
  (*comm)->homoInterRootRank = -1;
  (*comm)->homoInterMyRank = -1;
  (*comm)->homoInterRanks = -1;
  (*comm)->homoInterComm = NULL;
  (*comm)->c2cSchedule = NULL;
  (*comm)->devCommState = NULL;
  (*comm)->pendingDevCommCleanup = NULL;
  flagcxIntruQueueConstruct(&(*comm)->deferredBufferQueue);
  (*comm)->deferredBufferCount = 0;

  uint64_t magic = ((struct flagcxBootstrapHandle *)commId)->magic;
  (*comm)->magic = magic;

  // Init bootstrap net
  FLAGCXCHECK(bootstrapNetInit());

  // Init bootstrap state (creates wrapped state)
  FLAGCXCHECK(bootstrapCollInit((struct flagcxBootstrapHandle *)commId, rank,
                                nranks, magic, (*comm)->abortFlag,
                                &(*comm)->bootstrap));
  struct bootstrapState *state = (*comm)->bootstrap;

  // Ready to detect heterogeneous/homogeneous communicator
  // Use bootstrap allgather to exchange Device info
  flagcxVendor *vendorData =
      NULL; // temp data used for device vendor gather operation.

  // Get current gpu vendor
  flagcxVendor vendor;
  deviceAdaptor->getVendor(vendor.internal);
  FLAGCXCHECK(flagcxCalloc(&vendorData, nranks));
  memcpy(vendorData + rank, &vendor, sizeof(flagcxVendor));
  FLAGCXCHECK(
      bootstrapCollAllGather(state, (void *)vendorData, sizeof(flagcxVendor)));
  FLAGCXCHECK(bootstrapCollBarrier(state, rank, nranks, 0));

  // Compute intra-node topology using hostHash
  {
    uint64_t myHash = getHostHash();
    uint64_t *hostHashes = nullptr;
    FLAGCXCHECK(flagcxCalloc(&hostHashes, nranks));
    hostHashes[rank] = myHash;
    FLAGCXCHECK(bootstrapCollAllGather(state, hostHashes, sizeof(uint64_t)));
    FLAGCXCHECK(bootstrapCollBarrier(state, rank, nranks, 0));

    int localCount = 0;
    for (int r = 0; r < nranks; r++) {
      if (hostHashes[r] == myHash)
        localCount++;
    }
    (*comm)->localRanks = localCount;

    FLAGCXCHECK(flagcxCalloc(&(*comm)->localRankToRank, localCount));
    int lr = 0;
    for (int r = 0; r < nranks; r++) {
      if (hostHashes[r] == myHash) {
        (*comm)->localRankToRank[lr] = r;
        if (r == rank)
          (*comm)->localRank = lr;
        lr++;
      }
    }
    free(hostHashes);
    INFO(FLAGCX_INIT, "Intra-node topology: localRank=%d localRanks=%d",
         (*comm)->localRank, (*comm)->localRanks);
  }

  // Init cluster info
  int *globalRankToHomoRankData;
  int *clusterIdData;
  int *clusterInterRankData;
  FLAGCXCHECK(flagcxCalloc(&globalRankToHomoRankData, nranks));
  FLAGCXCHECK(flagcxCalloc(&clusterIdData, nranks));
  FLAGCXCHECK(flagcxCalloc(&clusterInterRankData, nranks));
  FLAGCXCHECK(flagcxCollectClusterInfos(
      vendorData, &(*comm)->commType, globalRankToHomoRankData + rank,
      &(*comm)->homoRootRank, &(*comm)->homoRanks, clusterIdData + rank,
      clusterInterRankData + rank, &(*comm)->nclusters, rank, nranks));
  FLAGCXCHECK(bootstrapCollAllGather(state, (void *)globalRankToHomoRankData,
                                     sizeof(int)));
  FLAGCXCHECK(
      bootstrapCollAllGather(state, (void *)clusterIdData, sizeof(int)));
  FLAGCXCHECK(
      bootstrapCollAllGather(state, (void *)clusterInterRankData, sizeof(int)));
  FLAGCXCHECK(bootstrapCollBarrier(state, rank, nranks, 0));
  (*comm)->homoRank = globalRankToHomoRankData[rank];
  (*comm)->clusterIds = clusterIdData;
  (*comm)->globalRank2HomoRank = globalRankToHomoRankData;

  // fill clusterVendorMap
  FLAGCXCHECK(flagcxFillClusterVendorInfo(vendorData, (*comm), clusterIdData,
                                          nranks, (*comm)->nclusters));

  int *clusterSizes;
  int *clusterInterRanks;
  FLAGCXCHECK(flagcxCalloc(&clusterSizes, (*comm)->nclusters));
  FLAGCXCHECK(flagcxCalloc(&clusterInterRanks, (*comm)->nclusters));
  for (int i = 0; i < (*comm)->nclusters; ++i) {
    clusterInterRanks[i] = -1;
  }

  int cid = 0;
  int sum = 0;
  for (int i = 0; i < nranks; ++i) {
    if (clusterIdData[i] == cid + 1) {
      clusterSizes[cid] = i - sum;
      cid += 1;
      sum = i;
    }
  }
  clusterSizes[cid] = nranks - sum;
  (*comm)->clusterSizes = clusterSizes;

  for (int i = 0; i < nranks; ++i) {
    if (clusterInterRankData[i] != -1) {
      clusterInterRanks[clusterIdData[i]] = clusterInterRankData[i];
    }
  }
  (*comm)->clusterInterRanks = clusterInterRanks;

  int start = 0;
  if (clusterIdData[rank] >= 1) {
    for (int i = 0; i < clusterIdData[rank]; ++i) {
      start += clusterSizes[i];
    }
  }

  // Build c2cSchedule
  FLAGCXCHECK(flagcxCalloc(&(*comm)->c2cSchedule, (*comm)->nclusters));
  int nLocals = (*comm)->nclusters;
  int local = (*comm)->clusterIds[rank];

  int nLocalsPow2 = pow2Up(nLocals);
  uint32_t localRound = 0;
  uint32_t localDelta = 0;
  int round = 0;
  do {
    if ((int)localDelta < nLocals) { // Filter nonsensical local deltas
      int sendLocal = (local + localDelta) % nLocals;
      int recvLocal = (local - localDelta + nLocals) % nLocals;
      (*comm)->c2cSchedule[round].sendCluster = sendLocal;
      (*comm)->c2cSchedule[round].recvCluster = recvLocal;
      round += 1;
    }
    localRound += 1;
    // Quadratic update
    localDelta = (localDelta + localRound) & (nLocalsPow2 - 1);
  } while (localRound != (uint32_t)nLocalsPow2);
  for (int i = 0; i < round; ++i) {
    INFO(FLAGCX_INIT,
         "cluster %d c2cSchedule[%d] sendCluster %d recvCluster %d", local, i,
         (*comm)->c2cSchedule[i].sendCluster,
         (*comm)->c2cSchedule[i].recvCluster);
  }

  // Update comm hasSingleRankHomoComm
  for (int i = 0; i < (*comm)->nclusters; ++i) {
    if ((*comm)->clusterSizes[i] == 1) {
      (*comm)->hasSingleRankHomoComm = 1;
    }
  }
  if ((*comm)->hasSingleRankHomoComm == -1) {
    (*comm)->hasSingleRankHomoComm = 0;
  }
  if ((*comm)->hasSingleRankHomoComm == 1 && useHomoComm(*comm)) {
    // no need to record it for homo comm
    (*comm)->hasSingleRankHomoComm = 0;
  }

  // Tuner init
  bool useTuner = false;
  const char *useTunerEnv = flagcxGetEnv("FLAGCX_USE_TUNER");
  if (useTunerEnv) {
    useTuner = (std::stoi(useTunerEnv) == 1) ? true : false;
  }
  INFO(FLAGCX_INIT, "Flagcx USE_TUNER flag set to %d", useTuner);
  if (useTuner) {
    (*comm)->tuner = &internalTuner;
    (*comm)->tunerInnerComm = NULL;
    (*comm)->isTunningComm = false;
    (*comm)->isTuningWithFlagscale = false;
    (*comm)->isUseSingleTunerComm = false;
    bool isTuningWithFlagscale = false;
    const char *isTuningWithFlagscaleEnv =
        flagcxGetEnv("FLAGCX_TUNING_WITH_FLAGSCALE");
    if (isTuningWithFlagscaleEnv) {
      isTuningWithFlagscale =
          (std::stoi(isTuningWithFlagscaleEnv) == 1) ? true : false;
    }
    (*comm)->isTuningWithFlagscale = isTuningWithFlagscale;

    bool isUseSingleTunerComm = false;
    const char *isUseSingleTunerCommEnv =
        flagcxGetEnv("TUNNING_WITH_SINGLE_COMM");

    if (isUseSingleTunerCommEnv) {
      isUseSingleTunerComm =
          (std::stoi(isUseSingleTunerCommEnv) == 1) ? true : false;
    }
    (*comm)->isUseSingleTunerComm = isUseSingleTunerComm;

    FLAGCXCHECK((*comm)->tuner->init((*comm)->nranks, (*comm)->rank,
                                     flagcxDebugLog, &((*comm)->tunerContext),
                                     state));
    uint32_t nConfigs = 0;
    FLAGCXCHECK(
        (*comm)->tuner->getCandidateNumber((*comm)->tunerContext, &nConfigs));
    if (nConfigs < 1) {
      WARN("Tuner returned 0 candidates, at least 1 is required.");
      return flagcxInternalError;
    }
    (*comm)->homoCommMap.clear();
    (*comm)->homoBestCommMap.clear();
    (*comm)->commMap.clear();

    if (!isUseSingleTunerComm) {
      // Note: The tuner only support homo comm optimization for now
      for (uint32_t i = 0; i < nConfigs; ++i) {
        struct flagcxCommTag tag = {""};
        FLAGCXCHECK(
            (*comm)->tuner->setCandidate((*comm)->tunerContext, i, &tag));
        INFO(FLAGCX_INIT | FLAGCX_TUNING,
             "start to prepare communicator tag=%s(%u/%u)", tag.tag, i,
             nConfigs);

        flagcxInnerComm_t innerComm = NULL;
        FLAGCXCHECK(flagcxHomoCommInit(state, *comm, &innerComm));
        // Insert item into commMap
        (*comm)->commMap[tag] = innerComm;
        // For backward compatible, also assign homo_comm field.
        (*comm)->homoComm = innerComm;
      }
    }

    if (isTuningWithFlagscale) {
      // Create a default communicator based on the default config
      flagcxInnerComm_t innerComm = NULL;
      FLAGCXCHECK(flagcxHomoCommInit(state, *comm, &innerComm));
      // Insert item into homoCommMap
      (*comm)->tunerInnerComm = innerComm;
      // For backward compatible, also assign homoComm field.
      (*comm)->homoComm = innerComm;
    }
  } else {
    (*comm)->tuner = NULL;
    FLAGCXCHECK(flagcxHomoCommInit(state, *comm, &((*comm)->homoComm)));
  }

  if (!useHomoComm(*comm) || useHeteroComm()) {
    flagcxUniqueId heteroCommId = {};
    flagcxResult_t uniqueIdResult = flagcxSuccess;
    if (rank == 0) {
      uniqueIdResult = flagcxHeteroGetUniqueId(&heteroCommId);
    }
    FLAGCXCHECK(
        flagcxCollectUniqueIdResult(state, rank, nranks, uniqueIdResult));
    FLAGCXCHECK(bootstrapCollBroadcast(
        state, rank, nranks, 0, (void *)&heteroCommId, sizeof(heteroCommId)));

    // call flagcxHeteroCommInitRank
    FLAGCXCHECK(flagcxHeteroCommInitRank(&(*comm)->heteroComm, nranks,
                                         heteroCommId, rank));

    // Share ipcTable with heteroComm for intra-node D2D bypass
    (*comm)->heteroComm->ipcTable = (*comm)->ipcTable;
    (*comm)->heteroComm->ipcTableSize = FLAGCX_MAX_IPC_ENTRIES;

    // Init host cclAdaptor
    if (useHostComm() || (*comm)->hasSingleRankHomoComm) {
      if (!flagcxParamTopoDetectionDisable()) {
        FLAGCXCHECK((*comm)->heteroComm->netAdaptor->getProperties(
            (*comm)->heteroComm->netDev, bootstrapGetNetProperties()));
      }
      flagcxInnerUniqueId hostCommId = {};
      FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->commInitRank(
          &(*comm)->hostComm, nranks, &hostCommId, rank, state));
    }
  }

  if ((!useHomoComm(*comm) || useHeteroComm()) && !useHostComm()) {
    // Experimental for multi-nic support
    // Collect nic distance to ranks
    (*comm)->clusterInterRankList.resize((*comm)->nclusters);
    struct flagcxNicDistance *nicDistanceData;
    FLAGCXCHECK(flagcxCalloc(&nicDistanceData, nranks));
    FLAGCXCHECK(flagcxGetNicDistance((*comm)->heteroComm->topoServer, rank,
                                     nicDistanceData + rank));
    FLAGCXCHECK(bootstrapCollAllGather(state, (void *)nicDistanceData,
                                       sizeof(flagcxNicDistance)));
    FLAGCXCHECK(bootstrapCollBarrier(state, rank, nranks, 0));
    for (int i = 0; i < (*comm)->nclusters; ++i) {
      int minDistance = INT_MAX;
      std::unordered_map<int, std::vector<int>> nicDistanceToRanks;
      std::unordered_map<int, std::unordered_set<uint64_t>> nicDistanceToNic;
      for (int j = 0; j < nranks; ++j) {
        if (clusterIdData[j] != i) {
          continue;
        }
        int val = nicDistanceData[j].distance;
        uint64_t netGuid = nicDistanceData[j].netGuid;
        if (nicDistanceToNic[val].find(netGuid) ==
            nicDistanceToNic[val].end()) {
          nicDistanceToRanks[val].push_back(j);
          nicDistanceToNic[val].insert(netGuid);
        }
        minDistance = std::min(minDistance, val);
      }
      (*comm)->clusterInterRankList[i] =
          std::move(nicDistanceToRanks[minDistance]);
    }
    // Set homoInterMyRank, homoInterRootRank and homoInterRanks
    auto &myClusterInterRanks =
        (*comm)->clusterInterRankList[clusterIdData[rank]];
    for (size_t i = 0; i < myClusterInterRanks.size(); ++i) {
      if (rank == myClusterInterRanks[i]) {
        (*comm)->homoInterMyRank = i;
      }
    }
    if ((*comm)->homoInterMyRank != -1) {
      (*comm)->homoInterRootRank = myClusterInterRanks[0];
      (*comm)->homoInterRanks = myClusterInterRanks.size();
    }

    INFO(FLAGCX_INIT,
         "rank = %d, nranks = %d, nclusters = %d, "
         "clusterId = %d, clusterSize = %d, "
         "clusterInterRank = %d, homoRank = %d, "
         "homoRootRank = %d, homoRanks = %d, "
         "homoInterRootRank = %d, homoInterMyRank = %d, "
         "homoInterRanks = %d, hasSingleRankHomoComm = %d, ",
         rank, nranks, (*comm)->nclusters, (*comm)->clusterIds[rank],
         (*comm)->clusterSizes[(*comm)->clusterIds[rank]],
         (*comm)->clusterInterRanks[(*comm)->clusterIds[rank]],
         (*comm)->homoRank, (*comm)->homoRootRank, (*comm)->homoRanks,
         (*comm)->homoInterRootRank, (*comm)->homoInterMyRank,
         (*comm)->homoInterRanks, (*comm)->hasSingleRankHomoComm);

    // Experimental for multi-nic support
    flagcxInnerUniqueId homoInterCommIdStorage = {};
    flagcxInnerUniqueId_t homoInterCommId = &homoInterCommIdStorage;
    // Let homoInterRootRank call underlying GetUniqueId function
    // for initialization of homo inter communicator
    flagcxResult_t uniqueIdResult = flagcxSuccess;
    if (rank == (*comm)->homoInterRootRank) {
      uniqueIdResult =
          cclAdaptors[flagcxCCLAdaptorDevice]->getUniqueId(&homoInterCommId);
    }
    FLAGCXCHECK(
        flagcxCollectUniqueIdResult(state, rank, nranks, uniqueIdResult));

    // Call cclAdaptor->commInitRank
    if ((*comm)->homoInterRootRank != -1) {
      FLAGCXCHECK(bootstrapCollSubgroupBroadcast(
          state, myClusterInterRanks.data(), (*comm)->homoInterMyRank,
          (*comm)->homoInterRanks, 0, (void *)homoInterCommId,
          sizeof(*homoInterCommId)));
      FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->commInitRank(
          &(*comm)->homoInterComm, (*comm)->homoInterRanks, homoInterCommId,
          (*comm)->homoInterMyRank, NULL));
    }
    free(nicDistanceData);
    const char *deviceFuncPathEnv = flagcxGetEnv("FLAGCX_DEVICE_FUNC_PATH");
    if (deviceFuncPathEnv) {
      FLAGCXCHECK(loadKernelSymbol(deviceFuncPathEnv, "deviceAsyncKernel",
                                   &deviceAsyncKernel));
      if (deviceAsyncKernel == NULL) {
        WARN("Failed to load async kernel from %s", deviceFuncPathEnv);
        return flagcxInvalidArgument;
      }
    }
  }

  free(clusterInterRankData);
  free(vendorData);
  // Initialize custom op state (non-fatal if fails)
  FLAGCXCHECK(flagcxDevCommStateInit(*comm));

  return flagcxSuccess;
}

flagcxResult_t flagcxCommFinalize(flagcxComm_t comm) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  FLAGCXCHECK(
      cclAdaptors[flagcxCCLAdaptorDevice]->commFinalize(comm->homoComm));

  if (!useHomoComm(comm) || useHeteroComm()) {
    // Finalize initiates local RMA quiescence without adding a blocking rank
    // collective. This preserves serial multi-rank lifecycle calls in one
    // process; transport-specific peer shutdown remains in commCleanup.
    FLAGCXCHECK(flagcxCommQuiesce(comm));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxCommDestroy(flagcxComm_t comm) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  flagcxResult_t destroyResult = flagcxSuccess;
  TRACE(FLAGCX_INIT,
        "flagcxCommDestroy begin comm=%p rank=%d tuner=%p homoComm=%p "
        "homoCommMapSize=%zu commMapSize=%zu",
        comm, comm->rank, comm->tuner, comm->homoComm, comm->homoCommMap.size(),
        comm->commMap.size());

  // Stop new producers and drain device FIFOs plus the RMA proxy before any MR
  // deregistration can close a connection. Quiesce is idempotent so a failed
  // deregistration leaves this communicator in a retryable, owned state.
  if (!useHomoComm(comm) || useHeteroComm())
    FLAGCXCHECK(flagcxCommQuiesce(comm));

  // Preserve transport-specific shutdown coordination before any DevComm or
  // one-sided registration can release remotely accessible memory.
  if (!useHomoComm(comm) || useHeteroComm())
    FLAGCXCHECK(flagcxCommCleanup(comm));

  // A failed create rollback can exist for any Device API backend. Retry it
  // while the outer communicator and its vendor communicator are still live.
  flagcxResult_t cleanupResult = flagcxDevCommRetryPendingCleanup(comm);
  if (cleanupResult != flagcxSuccess)
    return cleanupResult;

  // Custom-op staged windows are also present in symWindows. Remove their
  // device objects and explicit owners first so the generic live-window drain
  // below cannot free a handle still referenced by devCommState.
  TRACE(FLAGCX_INIT, "flagcxCommDestroy custom-op cleanup begin comm=%p", comm);
  cleanupResult = flagcxDevCommStateDestroy(comm);
  if (cleanupResult != flagcxSuccess)
    return cleanupResult;
  TRACE(FLAGCX_INIT, "flagcxCommDestroy custom-op cleanup end comm=%p", comm);

  if (!useHomoComm(comm) || useHeteroComm()) {
    // Each stage retains ownership on failure so the same rank can retry the
    // call safely after transport shutdown coordination has completed.
    cleanupResult = flagcxOneSideStagingDeregister(comm);
    if (cleanupResult != flagcxSuccess)
      return cleanupResult;
    cleanupResult = flagcxOneSideSignalDeregister(comm);
    if (cleanupResult != flagcxSuccess)
      return cleanupResult;
    cleanupResult =
        flagcxSymRetryPendingCleanup(comm->heteroComm, flagcxSymCleanupLocal);
    if (cleanupResult != flagcxSuccess)
      return cleanupResult;
    // Applications are allowed to destroy a communicator without explicitly
    // deregistering every window. Release local mappings before the shared MR
    // table and its connections disappear. Each window remains linked if a
    // fallible teardown step fails, so flagcxCommDestroy itself is retryable.
    while (comm->heteroComm != NULL && comm->heteroComm->symWindows != NULL) {
      flagcxWindow_t liveWindow = comm->heteroComm->symWindows->owner;
      if (liveWindow == NULL)
        return flagcxInternalError;
      cleanupResult = flagcxCommWindowDeregisterInternal(
          comm, liveWindow, flagcxMemCCL, flagcxSymCleanupLocal);
      if (cleanupResult != flagcxSuccess)
        return cleanupResult;
    }
    cleanupResult = flagcxOneSideDeregister(comm->heteroComm);
    if (cleanupResult != flagcxSuccess)
      return cleanupResult;
  }

  // Destroy cluster info
  free(comm->clusterIds);
  free(comm->clusterSizes);
  free(comm->globalRank2HomoRank);
  free(comm->localRankToRank);
  free(comm->c2cSchedule);
  free(comm->clusterInterRanks);

  // Destroy homo comms
  TRACE(FLAGCX_INIT, "flagcxCommDestroy homo cleanup begin comm=%p", comm);
  if (comm->tuner) {
    size_t homoCommIndex = 0;
    for (const auto &item : comm->homoCommMap) {
      if (item.second != nullptr) {
        TRACE(FLAGCX_INIT,
              "flagcxCommDestroy homo comm begin comm=%p index=%zu inner=%p",
              comm, homoCommIndex, item.second);
        FLAGCXCHECK(
            cclAdaptors[flagcxCCLAdaptorDevice]->commDestroy(item.second));
        TRACE(FLAGCX_INIT,
              "flagcxCommDestroy homo comm end comm=%p index=%zu inner=%p",
              comm, homoCommIndex, item.second);
      }
      homoCommIndex++;
    }
  } else {
    FLAGCXCHECK(
        cclAdaptors[flagcxCCLAdaptorDevice]->commDestroy(comm->homoComm));
  }
  TRACE(FLAGCX_INIT, "flagcxCommDestroy homo cleanup end comm=%p", comm);

  if (!useHomoComm(comm) || useHeteroComm()) {
    // Backend-level comm cleanup: relay teardown, IPC table cleanup.
    // Must run before flagcxHeteroCommDestroy, which frees proxyState and
    // heteroComm.
    TRACE(FLAGCX_INIT,
          "flagcxCommDestroy backend cleanup begin comm=%p heteroComm=%p", comm,
          comm->heteroComm);
    // Destroy hetero comm
    TRACE(FLAGCX_INIT,
          "flagcxCommDestroy hetero cleanup begin comm=%p heteroComm=%p", comm,
          comm->heteroComm);
    destroyResult = flagcxHeteroCommDestroy(comm->heteroComm);
    TRACE(FLAGCX_INIT, "flagcxCommDestroy hetero cleanup end comm=%p result=%d",
          comm, destroyResult);
    // Destroy host comm
    if (useHostComm()) {
      FLAGCXCHECK(
          cclAdaptors[flagcxCCLAdaptorHost]->commDestroy(comm->hostComm));
    }
  }

  // Clean up IPC peer pointer table — deferred to here.
  TRACE(FLAGCX_INIT, "flagcxCommDestroy IPC table cleanup begin comm=%p", comm);
  FLAGCXCHECK(flagcxCommCleanupIpcTable(comm));

  // Drain deferred IPC entries (slots released at runtime).
  FLAGCXCHECK(flagcxCommDrainDeferredIpc(comm));
  TRACE(FLAGCX_INIT, "flagcxCommDestroy IPC table cleanup end comm=%p", comm);

  // Drain deferred DevComm buffer queue.
  FLAGCXCHECK(flagcxCommDrainDeferredBuffers(comm));

  // Drain deferred device/host-pinned memory frees,
  // collected during DevComm/DevMem cleanup.
  FLAGCXCHECK(flagcxCommDrainDeferredFrees(comm));

  // Destroy bootstrap state and net
  bootstrapClose(comm->bootstrap);

  // Destroy tuner
  if (comm->tuner) {
    comm->tuner->destroy(comm->tunerContext);
  }

  // Finalize net adaptor plugin (dlclose)
  FLAGCXCHECK(flagcxNetAdaptorPluginFinalize());

  // Finalize device/CCL adaptor plugins (ref-counted)
  flagcxCCLAdaptorPluginFinalize();
  flagcxDeviceAdaptorPluginFinalize();

  free(comm);
  return destroyResult;
}

flagcxResult_t flagcxCommAbort(flagcxComm_t comm) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->commAbort(comm->homoComm));
  if (!useHomoComm(comm)) {
    // TODO: to be implemented.
    return flagcxNotSupported;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxCommResume(flagcxComm_t comm) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->commResume(comm->homoComm));
  if (!useHomoComm(comm)) {
    // TODO: to be implemented.
    return flagcxNotSupported;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxCommSuspend(flagcxComm_t comm) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->commSuspend(comm->homoComm));
  if (!useHomoComm(comm)) {
    // TODO: to be implemented.
    return flagcxNotSupported;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxCommCount(const flagcxComm_t comm, int *count) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHomoComm(comm)) {
    return cclAdaptors[flagcxCCLAdaptorDevice]->commCount(comm->homoComm,
                                                          count);
  }
  return flagcxHeteroCommCount(comm->heteroComm, count);
}

flagcxResult_t flagcxCommGetDeviceNumber(const flagcxComm_t comm, int *device) {
  return cclAdaptors[flagcxCCLAdaptorDevice]->commGetDeviceNumber(
      comm->homoComm, device);
}

flagcxResult_t flagcxCommUserRank(const flagcxComm_t comm, int *rank) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHomoComm(comm)) {
    return cclAdaptors[flagcxCCLAdaptorDevice]->commUserRank(comm->homoComm,
                                                             rank);
  }
  return flagcxHeteroCommUserRank(comm->heteroComm, rank);
}

flagcxResult_t flagcxCommFifoBuffer(const flagcxComm_t comm, int contextId,
                                    void **buffer) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));

  if (buffer == nullptr) {
    return flagcxInvalidArgument;
  }

  if (contextId < 0 || contextId >= FLAGCX_DEVICE_CTA_COUNT) {
    return flagcxInvalidArgument;
  }

  // FIFO buffers are only available on hetero communicators
  if (useHomoComm(comm) && !useHeteroComm()) {
    return flagcxNotSupported;
  }

  if (comm->heteroComm == nullptr) {
    return flagcxNotSupported;
  }

  if (comm->heteroComm->fifoBuffers[contextId] == nullptr) {
    return flagcxInvalidUsage;
  }

  *buffer = comm->heteroComm->fifoBuffers[contextId];
  return flagcxSuccess;
}

flagcxResult_t flagcxCommGetAsyncError(flagcxComm_t comm,
                                       flagcxResult_t *asyncError) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (asyncError == nullptr)
    return flagcxInvalidArgument;

  if (comm->heteroComm != nullptr && (!useHomoComm(comm) || useHeteroComm())) {
    flagcxResult_t result = flagcxSuccess;

    // Hybrid collectives also execute intra-cluster work on the device CCL.
    // A healthy heterogeneous proxy must not hide an asynchronous CCL error.
    flagcxInnerComm_t cclComms[] = {
        comm->homoComm,
        comm->homoInterComm != comm->homoComm ? comm->homoInterComm : nullptr};
    for (flagcxInnerComm_t cclComm : cclComms) {
      if (result != flagcxSuccess || cclComm == nullptr)
        continue;
      flagcxResult_t cclAsync = flagcxSuccess;
      FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->commGetAsyncError(
          cclComm, &cclAsync));
      result = cclAsync;
    }
    if (result == flagcxSuccess && comm->heteroComm->proxyState != nullptr) {
      result = __atomic_load_n(&comm->heteroComm->proxyState->asyncResult,
                               __ATOMIC_ACQUIRE);
      if (result == flagcxSuccess) {
        result = __atomic_load_n(
            &comm->heteroComm->proxyState->kernelState.terminalResult,
            __ATOMIC_ACQUIRE);
      }
    }
    if (result == flagcxSuccess && comm->heteroComm->rmaProxy != nullptr &&
        __atomic_load_n(&comm->heteroComm->rmaProxy->rmaError,
                        __ATOMIC_ACQUIRE) != 0) {
      result = flagcxInternalError;
    }
    *asyncError = result;
    return flagcxSuccess;
  }

  if (useHomoComm(comm)) {
    return cclAdaptors[flagcxCCLAdaptorDevice]->commGetAsyncError(
        comm->homoComm, asyncError);
  }

  return flagcxNotSupported;
}

flagcxResult_t flagcxBarrier(flagcxComm_t comm, flagcxStream_t stream) {
  void *barrierBuff;
  deviceAdaptor->deviceMalloc(&barrierBuff, comm->nranks, flagcxMemDevice,
                              stream);
  deviceAdaptor->deviceMemset(barrierBuff, 0, comm->nranks, flagcxMemDevice,
                              stream);
  flagcxAllReduce(barrierBuff, barrierBuff, comm->nranks, flagcxChar, flagcxMax,
                  comm, stream);
  deviceAdaptor->deviceFree(barrierBuff, flagcxMemDevice, stream);
  deviceAdaptor->streamSynchronize(stream);
  return flagcxSuccess;
}

flagcxResult_t flagcxReduce(const void *sendbuff, void *recvbuff, size_t count,
                            flagcxDataType_t datatype, flagcxRedOp_t op,
                            int root, flagcxComm_t comm,
                            flagcxStream_t stream) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHeteroComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxUniRunner]->reduce(
        sendbuff, recvbuff, count, datatype, op, root, comm, stream));
  } else if (useHomoComm(comm)) {
    FLAGCXCHECK(flagcxRunners[flagcxHomoRunner]->reduce(
        sendbuff, recvbuff, count, datatype, op, root, comm, stream));
  } else if (useHostComm() || comm->hasSingleRankHomoComm) {
    // c2c validation
    if (comm->hasSingleRankHomoComm) {
      WARN("Host comm is required to perform C2C reduce op when "
           "comm->hasSingleRankHomoComm is True");
    }
    FLAGCXCHECK(flagcxRunners[flagcxHostRunner]->reduce(
        sendbuff, recvbuff, count, datatype, op, root, comm, stream));
  } else {
    FLAGCXCHECK(flagcxRunners[flagcxHybridRunner]->reduce(
        sendbuff, recvbuff, count, datatype, op, root, comm, stream));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxGather(const void *sendbuff, void *recvbuff, size_t count,
                            flagcxDataType_t datatype, int root,
                            flagcxComm_t comm, flagcxStream_t stream) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHeteroComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxUniRunner]->gather(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  } else if (useHomoComm(comm)) {
    FLAGCXCHECK(flagcxRunners[flagcxHomoRunner]->gather(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  } else if (useHostComm() || comm->hasSingleRankHomoComm) {
    // c2c validation
    if (comm->hasSingleRankHomoComm) {
      WARN("Host comm is required to perform C2C gather op when "
           "comm->hasSingleRankHomoComm is True");
    }
    FLAGCXCHECK(flagcxRunners[flagcxHostRunner]->gather(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  } else {
    FLAGCXCHECK(flagcxRunners[flagcxHybridRunner]->gather(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxScatter(const void *sendbuff, void *recvbuff, size_t count,
                             flagcxDataType_t datatype, int root,
                             flagcxComm_t comm, flagcxStream_t stream) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHeteroComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxUniRunner]->scatter(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  } else if (useHomoComm(comm)) {
    FLAGCXCHECK(flagcxRunners[flagcxHomoRunner]->scatter(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  } else if (useHostComm() || comm->hasSingleRankHomoComm) {
    // c2c validation
    if (comm->hasSingleRankHomoComm) {
      WARN("Host comm is required to perform C2C scatter op when "
           "comm->hasSingleRankHomoComm is True");
    }
    FLAGCXCHECK(flagcxRunners[flagcxHostRunner]->scatter(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  } else {
    FLAGCXCHECK(flagcxRunners[flagcxHybridRunner]->scatter(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxBroadcast(const void *sendbuff, void *recvbuff,
                               size_t count, flagcxDataType_t datatype,
                               int root, flagcxComm_t comm,
                               flagcxStream_t stream) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHeteroComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxUniRunner]->broadcast(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  } else if (useHomoComm(comm)) {
    FLAGCXCHECK(flagcxRunners[flagcxHomoRunner]->broadcast(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  } else if (useHostComm() || comm->hasSingleRankHomoComm) {
    // c2c validation
    if (comm->hasSingleRankHomoComm) {
      WARN("Host comm is required to perform C2C broadcast op when "
           "comm->hasSingleRankHomoComm is True");
    }
    FLAGCXCHECK(flagcxRunners[flagcxHostRunner]->broadcast(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  } else {
    FLAGCXCHECK(flagcxRunners[flagcxHybridRunner]->broadcast(
        sendbuff, recvbuff, count, datatype, root, comm, stream));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxAllReduce(const void *sendbuff, void *recvbuff,
                               size_t count, flagcxDataType_t datatype,
                               flagcxRedOp_t op, flagcxComm_t comm,
                               flagcxStream_t stream) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));

  // Try custom allreduce if registered
  if (comm->devCommState != NULL &&
      comm->devCommState->customAllReduce != NULL &&
      comm->localRanks == comm->nranks) {
    auto *state = comm->devCommState;
    size_t size = count * getFlagcxDataTypeSize(datatype);
    if (size <= state->stagedBuffSize) {
      flagcxResult_t res = state->customAllReduce(sendbuff, recvbuff, count,
                                                  datatype, op, comm, stream);
      if (res == flagcxSuccess) {
        return flagcxSuccess;
      }
      if (res != flagcxNotSupported) {
        return res;
      }
    }
    // size >= stagedBuffSize or flagcxNotSupported: fallback to standard path
  }

  if (useHeteroComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxUniRunner]->allReduce(
        sendbuff, recvbuff, count, datatype, op, comm, stream));
  } else if (useHomoComm(comm)) {
    FLAGCXCHECK(flagcxRunners[flagcxHomoRunner]->allReduce(
        sendbuff, recvbuff, count, datatype, op, comm, stream));
  } else if (useHostComm() || comm->hasSingleRankHomoComm) {
    // c2c validation
    if (comm->hasSingleRankHomoComm) {
      WARN("Host comm is required to perform C2C allreduce op when "
           "comm->hasSingleRankHomoComm is True");
    }
    FLAGCXCHECK(flagcxRunners[flagcxHostRunner]->allReduce(
        sendbuff, recvbuff, count, datatype, op, comm, stream));
  } else {
    FLAGCXCHECK(flagcxRunners[flagcxHybridRunner]->allReduce(
        sendbuff, recvbuff, count, datatype, op, comm, stream));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxReduceScatter(const void *sendbuff, void *recvbuff,
                                   size_t recvcount, flagcxDataType_t datatype,
                                   flagcxRedOp_t op, flagcxComm_t comm,
                                   flagcxStream_t stream) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHeteroComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxUniRunner]->reduceScatter(
        sendbuff, recvbuff, recvcount, datatype, op, comm, stream));
  } else if (useHomoComm(comm)) {
    FLAGCXCHECK(flagcxRunners[flagcxHomoRunner]->reduceScatter(
        sendbuff, recvbuff, recvcount, datatype, op, comm, stream));
  } else if (useHostComm() || comm->hasSingleRankHomoComm) {
    // c2c validation
    if (comm->hasSingleRankHomoComm) {
      WARN("Host comm is required to perform C2C reducescatter op when "
           "comm->hasSingleRankHomoComm is True");
    }
    FLAGCXCHECK(flagcxRunners[flagcxHostRunner]->reduceScatter(
        sendbuff, recvbuff, recvcount, datatype, op, comm, stream));
  } else {
    FLAGCXCHECK(flagcxRunners[flagcxHybridRunner]->reduceScatter(
        sendbuff, recvbuff, recvcount, datatype, op, comm, stream));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxAllGather(const void *sendbuff, void *recvbuff,
                               size_t sendcount, flagcxDataType_t datatype,
                               flagcxComm_t comm, flagcxStream_t stream) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHeteroComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxUniRunner]->allGather(
        sendbuff, recvbuff, sendcount, datatype, comm, stream));
  } else if (useHomoComm(comm)) {
    FLAGCXCHECK(flagcxRunners[flagcxHomoRunner]->allGather(
        sendbuff, recvbuff, sendcount, datatype, comm, stream));
  } else if (useHostComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxHostRunner]->allGather(
        sendbuff, recvbuff, sendcount, datatype, comm, stream));
  } else {
    FLAGCXCHECK(flagcxRunners[flagcxHybridRunner]->allGather(
        sendbuff, recvbuff, sendcount, datatype, comm, stream));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxAlltoAll(const void *sendbuff, void *recvbuff,
                              size_t count, flagcxDataType_t datatype,
                              flagcxComm_t comm, flagcxStream_t stream) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHeteroComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxUniRunner]->alltoAll(
        sendbuff, recvbuff, count, datatype, comm, stream));
  } else if (useHomoComm(comm)) {
    FLAGCXCHECK(flagcxRunners[flagcxHomoRunner]->alltoAll(
        sendbuff, recvbuff, count, datatype, comm, stream));
  } else if (useHostComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxHostRunner]->alltoAll(
        sendbuff, recvbuff, count, datatype, comm, stream));
  } else {
    FLAGCXCHECK(flagcxRunners[flagcxHybridRunner]->alltoAll(
        sendbuff, recvbuff, count, datatype, comm, stream));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxAlltoAllv(const void *sendbuff, size_t *sendcounts,
                               size_t *sdispls, void *recvbuff,
                               size_t *recvcounts, size_t *rdispls,
                               flagcxDataType_t datatype, flagcxComm_t comm,
                               flagcxStream_t stream) {

  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHeteroComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxUniRunner]->alltoAllv(
        sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls, datatype,
        comm, stream));
  } else if (useHomoComm(comm)) {
    FLAGCXCHECK(flagcxRunners[flagcxHomoRunner]->alltoAllv(
        sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls, datatype,
        comm, stream));
  } else if (useHostComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxHostRunner]->alltoAllv(
        sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls, datatype,
        comm, stream));
  } else {
    FLAGCXCHECK(flagcxRunners[flagcxHybridRunner]->alltoAllv(
        sendbuff, sendcounts, sdispls, recvbuff, recvcounts, rdispls, datatype,
        comm, stream));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxSend(const void *sendbuff, size_t count,
                          flagcxDataType_t datatype, int peer,
                          flagcxComm_t comm, flagcxStream_t stream) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHeteroComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxUniRunner]->send(sendbuff, count, datatype,
                                                     peer, comm, stream));
  } else if (useHomoComm(comm)) {
    FLAGCXCHECK(flagcxRunners[flagcxHomoRunner]->send(sendbuff, count, datatype,
                                                      peer, comm, stream));
  } else if (useHostComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxHostRunner]->send(sendbuff, count, datatype,
                                                      peer, comm, stream));
  } else {
    FLAGCXCHECK(flagcxRunners[flagcxHybridRunner]->send(
        sendbuff, count, datatype, peer, comm, stream));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxRecv(void *recvbuff, size_t count,
                          flagcxDataType_t datatype, int peer,
                          flagcxComm_t comm, flagcxStream_t stream) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (useHeteroComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxUniRunner]->recv(recvbuff, count, datatype,
                                                     peer, comm, stream));
  } else if (useHomoComm(comm)) {
    FLAGCXCHECK(flagcxRunners[flagcxHomoRunner]->recv(recvbuff, count, datatype,
                                                      peer, comm, stream));
  } else if (useHostComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxHostRunner]->recv(recvbuff, count, datatype,
                                                      peer, comm, stream));
  } else {
    FLAGCXCHECK(flagcxRunners[flagcxHybridRunner]->recv(
        recvbuff, count, datatype, peer, comm, stream));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxGet(flagcxComm_t comm, int peer, size_t srcOffset,
                         size_t dstOffset, size_t size, int srcMrIdx,
                         int dstMrIdx) {
  if (comm == NULL || comm->heteroComm == NULL)
    return flagcxInvalidArgument;
  return flagcxHeteroGet(comm->heteroComm, peer, srcOffset, dstOffset, size,
                         srcMrIdx, dstMrIdx);
}

flagcxResult_t flagcxPut(flagcxComm_t comm, int peer, size_t srcOffset,
                         size_t dstOffset, size_t size, int srcMrIdx,
                         int dstMrIdx) {
  if (comm == NULL || comm->heteroComm == NULL)
    return flagcxInvalidArgument;
  return flagcxHeteroPut(comm->heteroComm, peer, srcOffset, dstOffset, size,
                         srcMrIdx, dstMrIdx);
}

flagcxResult_t flagcxBatchPut(flagcxComm_t comm, int peer,
                              const size_t *srcOffsets,
                              const size_t *dstOffsets, const size_t *sizes,
                              const int *srcMrIdxs, const int *dstMrIdxs,
                              size_t count) {
  if (comm == NULL || comm->heteroComm == NULL)
    return flagcxInvalidArgument;
  return flagcxHeteroBatchPut(comm->heteroComm, peer, srcOffsets, dstOffsets,
                              sizes, srcMrIdxs, dstMrIdxs, count);
}

// ---- High-level NCCL-aligned one-sided APIs (window-based, stream-integrated)
// ----

flagcxResult_t flagcxPutSignal(const void *localbuff, size_t count,
                               flagcxDataType_t datatype, int peer,
                               flagcxWindow_t peerWin, size_t peerWinOffset,
                               unsigned int flags, flagcxComm_t comm,
                               flagcxStream_t stream) {
  (void)flags;
  if (comm == NULL || comm->heteroComm == NULL)
    return flagcxInvalidArgument;
  if (peerWin == NULL)
    return flagcxInvalidArgument;
  if (peer < 0 || peer >= comm->heteroComm->nRanks) {
    WARN("flagcxPutSignal: peer %d out of range (nRanks=%d)", peer,
         comm->heteroComm->nRanks);
    return flagcxInvalidArgument;
  }

  flagcxHeteroComm_t hetero = comm->heteroComm;
  if (!peerWin->isSymmetricDefault || peerWin->defaultBase == NULL)
    return flagcxInvalidArgument;

  size_t elementSize = getFlagcxDataTypeSize(datatype);
  if (elementSize == 0 || count > SIZE_MAX / elementSize)
    return flagcxInvalidArgument;
  size_t byteSize = count * elementSize;
  if (byteSize > 0 && localbuff == NULL)
    return flagcxInvalidArgument;
  if (peerWinOffset > peerWin->defaultBase->heapSize ||
      byteSize > peerWin->defaultBase->heapSize - peerWinOffset)
    return flagcxInvalidArgument;

  // Resolve the source through the transport-neutral symmetric-window
  // registry. A window may be IPC-capable even if network MR registration
  // failed and mrIndex remains -1.
  size_t srcOffset = 0;
  flagcxSymWindow_t srcWindow =
      byteSize > 0
          ? flagcxSymWindowFind(hetero, localbuff, byteSize, &srcOffset)
          : NULL;
  if (byteSize > 0 && srcWindow == NULL) {
    WARN("flagcxPutSignal: localbuff %p is not in an active symmetric window",
         localbuff);
    return flagcxInvalidArgument;
  }
  flagcxSymWindow_t dstWindow = peerWin->defaultBase;
  int srcMrIdx = srcWindow != NULL ? srcWindow->mrIndex : -1;
  int dstMrIdx = dstWindow->mrIndex;

  // Signal offset: sender writes to its own slot in receiver's signal buffer,
  // so receiver can identify which peer sent the signal.
  size_t signalOffset = (size_t)hetero->rank * sizeof(uint64_t);

  uint64_t opSeq = 0;
  return flagcxHeteroPutSignalStream(
      hetero, peer, srcOffset, peerWinOffset, byteSize, signalOffset, srcMrIdx,
      dstMrIdx, 1 /*signalValue*/, srcWindow, dstWindow, stream, &opSeq);
}

flagcxResult_t flagcxSignal(int peer, unsigned int flags, flagcxComm_t comm,
                            flagcxStream_t stream) {
  (void)flags;
  if (comm == NULL || comm->heteroComm == NULL)
    return flagcxInvalidArgument;
  if (peer < 0 || peer >= comm->heteroComm->nRanks) {
    WARN("flagcxSignal: peer %d out of range (nRanks=%d)", peer,
         comm->heteroComm->nRanks);
    return flagcxInvalidArgument;
  }

  // Sender writes to its own slot in receiver's signal buffer
  size_t signalOffset = (size_t)comm->heteroComm->rank * sizeof(uint64_t);
  uint64_t opSeq = 0;
  return flagcxHeteroPutSignalStream(comm->heteroComm, peer, 0, 0, 0,
                                     signalOffset, -1, -1, 1 /*signalValue*/,
                                     NULL, NULL, stream, &opSeq);
}

flagcxResult_t flagcxWaitSignal(int nDesc,
                                const flagcxWaitSignalDesc_t *signalDescs,
                                flagcxComm_t comm, flagcxStream_t stream) {
  if (nDesc == 0)
    return flagcxSuccess;
  if (comm == NULL || comm->heteroComm == NULL)
    return flagcxInvalidArgument;
  if (stream == NULL || signalDescs == NULL)
    return flagcxInvalidArgument;

  flagcxHeteroComm_t hetero = comm->heteroComm;
  if (hetero->rmaProxy == NULL)
    return flagcxInvalidArgument;

  for (int i = 0; i < nDesc; i++) {
    int peer = signalDescs[i].peer;
    if (peer < 0 || peer >= hetero->nRanks) {
      WARN("flagcxWaitSignal: peer %d out of range (nRanks=%d)", peer,
           hetero->nRanks);
      return flagcxInvalidArgument;
    }
    uint64_t opCnt = (uint64_t)signalDescs[i].opCnt;

    // Signal offset: peer's slot in my local signal buffer
    // (sender writes to sender's rank slot on receiver)
    size_t signalOffset = (size_t)peer * sizeof(uint64_t);

    // Wait for signal value >= opCnt using GPU-side streamWaitValue64
    flagcxResult_t res =
        flagcxHeteroWaitSignal(hetero, peer, signalOffset, opCnt, stream);
    if (res != flagcxSuccess)
      return res;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxReadCounter(flagcxComm_t comm, uint64_t *count) {
  if (comm == NULL || comm->heteroComm == NULL)
    return flagcxInvalidArgument;
  return flagcxHeteroReadCounter(comm->heteroComm, count);
}

flagcxResult_t flagcxWaitCounter(flagcxComm_t comm, uint64_t target) {
  if (comm == NULL || comm->heteroComm == NULL)
    return flagcxInvalidArgument;
  return flagcxHeteroWaitCounter(comm->heteroComm, target);
}

flagcxResult_t flagcxGroupStart(flagcxComm_t comm) {
  if (useHeteroComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxUniRunner]->groupStart());
  } else if (comm == NULL || useHomoComm(comm)) {
    if (comm == NULL) {
      INFO(
          FLAGCX_COLL,
          "flagcxGroupStart: comm is NULL, delegating to homo runner directly");
    }
    FLAGCXCHECK(flagcxRunners[flagcxHomoRunner]->groupStart());
  } else if (useHostComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxHostRunner]->groupStart());
  } else {
    FLAGCXCHECK(flagcxRunners[flagcxHybridRunner]->groupStart());
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxGroupEnd(flagcxComm_t comm) {
  if (useHeteroComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxUniRunner]->groupEnd());
  } else if (comm == NULL || useHomoComm(comm)) {
    if (comm == NULL) {
      INFO(FLAGCX_COLL,
           "flagcxGroupEnd: comm is NULL, delegating to homo runner directly");
    }
    FLAGCXCHECK(flagcxRunners[flagcxHomoRunner]->groupEnd());
  } else if (useHostComm()) {
    FLAGCXCHECK(flagcxRunners[flagcxHostRunner]->groupEnd());
  } else {
    FLAGCXCHECK(flagcxRunners[flagcxHybridRunner]->groupEnd());
  }
  return flagcxSuccess;
}
