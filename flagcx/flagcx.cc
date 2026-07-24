#include "flagcx.h"
#include "adaptor.h"
#include "adaptor_plugin_load.h"
#include "alloc.h"
#include "bootstrap.h"
#include "check.h"
#include "cluster.h"
#include "comm.h"
#include "cost_model.h"
#include "flagcx_hetero.h"
#include "flagcx_kernel.h"
#include "flagcx_net.h"
#include "ib_common.h"
#include "launch_kernel.h"
#include "net.h"
#include "onesided.h"
#include "param.h"
#include "proxy.h"
#include "reg_pool.h"
#include "runner.h"
#include "sym_heap.h"
#include "timer.h"
#include "transport.h"
#include "utils.h"
#include <cassert>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <unordered_map>

flagcxRegPool globalRegPool;

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

FLAGCX_PARAM(MemEnable, "MEM_ENABLE", 0);

flagcxResult_t flagcxMemAlloc(void **ptr, size_t size) {
  if (ptr == NULL || size == 0) {
    WARN("Invalid ptr(NULL) or size(0) for allocation.");
    return flagcxInvalidArgument;
  }
  if (flagcxParamMemEnable()) {
    FLAGCXCHECK(deviceAdaptor->gdrMemAlloc(ptr, size, NULL));
    if (*ptr != NULL) {
      INFO(FLAGCX_REG, "flagcxMemAlloc: GDR allocated [%p, %ld]", *ptr, size);
    } else {
      WARN("flagcxMemAlloc: GDR allocation failed");
      return flagcxUnhandledDeviceError;
    }
  } else {
    FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->memAlloc(ptr, size));
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxMemFree(void *ptr) {
  if (ptr == NULL) {
    WARN("Invalid pointer(=NULL) for de-allocation.");
    return flagcxSuccess;
  }
  if (flagcxParamMemEnable()) {
    FLAGCXCHECK(deviceAdaptor->gdrMemFree(ptr, NULL));
    INFO(FLAGCX_REG, "flagcxMemFree: GDR memory deallocated");
  } else {
    FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->memFree(ptr));
  }
  return flagcxSuccess;
}

// Build full-mesh IB connections (including self-loopback) for one-sided ops.
// Called once on the first flagcxOneSideRegister invocation; stored in
// handle[0]. Pattern aligned with NCCL GIN gin.cc:146-158.
// Build one full-mesh of IB connections for a single context.
// On success, stores sendComms[peer] and recvComms[peer] arrays into
// *outSend/*outRecv.
static flagcxResult_t
flagcxOneSideBuildOneContext(struct flagcxHeteroComm *heteroComm, int nranks,
                             int rank, int contextIdx, void ***outSend,
                             void ***outRecv) {
  flagcxResult_t res = flagcxSuccess;
  void *listenComm = NULL;
  flagcxNetHandle_t *allHandles = NULL;
  void **sendComms = NULL;
  void **recvComms = NULL;

  FLAGCXCHECKGOTO(flagcxCalloc(&sendComms, nranks), res, fail);
  FLAGCXCHECKGOTO(flagcxCalloc(&recvComms, nranks), res, fail);

  {
    flagcxNetHandle_t myListenHandle = {};
    FLAGCXCHECKGOTO(heteroComm->netAdaptor->listen(heteroComm->netDev,
                                                   (void *)myListenHandle,
                                                   &listenComm),
                    res, fail);

    FLAGCXCHECKGOTO(flagcxCalloc(&allHandles, nranks), res, fail_listen);
    memcpy(&allHandles[rank], &myListenHandle, sizeof(flagcxNetHandle_t));
    FLAGCXCHECKGOTO(bootstrapCollAllGather(heteroComm->bootstrap,
                                           (void *)allHandles,
                                           sizeof(flagcxNetHandle_t)),
                    res, fail_handles);

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
    if (sendComms[i])
      heteroComm->netAdaptor->closeSend(sendComms[i]);
    if (recvComms[i])
      heteroComm->netAdaptor->closeRecv(recvComms[i]);
  }
  free(allHandles);
fail_listen:
  heteroComm->netAdaptor->closeListen(listenComm);
fail:
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
    FLAGCXCHECKGOTO(flagcxProxyInit(heteroComm), res, fail);
  }
  if (heteroComm->proxyState != NULL)
    nContexts = 1 + heteroComm->proxyState->kernelState.contextCount;
  info->nContexts = nContexts;
  info->nRanks = nranks;

  FLAGCXCHECKGOTO(flagcxCalloc(&info->contextSendComms, nContexts), res, fail);
  FLAGCXCHECKGOTO(flagcxCalloc(&info->contextRecvComms, nContexts), res, fail);

  // Build one full-mesh per context (collective: all ranks participate per
  // round)
  for (int ctx = 0; ctx < nContexts; ctx++) {
    FLAGCXCHECKGOTO(flagcxOneSideBuildOneContext(heteroComm, nranks, rank, ctx,
                                                 &info->contextSendComms[ctx],
                                                 &info->contextRecvComms[ctx]),
                    res, fail_partial);
  }

  // Legacy aliases: context 0 = RMA proxy
  info->fullSendComms = info->contextSendComms[0];
  info->fullRecvComms = info->contextRecvComms[0];

  INFO(FLAGCX_REG,
       "flagcxOneSideBuildFullMesh: rank %d, %d contexts × %d peers "
       "(including self-loopback)",
       rank, nContexts, nranks);
  return flagcxSuccess;

fail_partial:
  // Cleanup contexts that were successfully created
  for (int ctx = 0; ctx < nContexts; ctx++) {
    if (info->contextSendComms[ctx] != NULL) {
      for (int i = 0; i < nranks; i++) {
        if (info->contextSendComms[ctx][i])
          heteroComm->netAdaptor->closeSend(info->contextSendComms[ctx][i]);
        if (info->contextRecvComms[ctx][i])
          heteroComm->netAdaptor->closeRecv(info->contextRecvComms[ctx][i]);
      }
      free(info->contextSendComms[ctx]);
      free(info->contextRecvComms[ctx]);
    }
  }
  free(info->contextSendComms);
  free(info->contextRecvComms);
fail:
  info->contextSendComms = NULL;
  info->contextRecvComms = NULL;
  info->fullSendComms = NULL;
  info->fullRecvComms = NULL;
  info->nRanks = 0;
  info->nContexts = 0;
  return res;
}

flagcxResult_t flagcxOneSideRegisterInternal(flagcxHeteroComm_t heteroComm,
                                             void *buff, size_t size) {
  if (heteroComm == NULL || heteroComm->netAdaptor == NULL ||
      heteroComm->netAdaptor->iput == NULL ||
      heteroComm->netAdaptor->regMr == NULL) {
    return flagcxNotSupported;
  }

  // Check for duplicate registration of the same buffer within this comm
  for (int i = 0; i < heteroComm->oneSideHandleCount; i++) {
    struct flagcxOneSideHandleInfo *h = heteroComm->oneSideHandles[i];
    if (h != NULL && h->baseVas != NULL &&
        h->baseVas[heteroComm->rank] == (uintptr_t)buff) {
      INFO(FLAGCX_REG,
           "flagcxOneSideRegister: buffer %p already registered at index %d",
           buff, i);
      return flagcxSuccess;
    }
  }

  if (heteroComm->bootstrap == NULL) {
    INFO(FLAGCX_REG, "flagcxOneSideRegister: bootstrap is NULL");
    return flagcxNotSupported;
  }

  flagcxResult_t res = flagcxSuccess;
  void *mrHandle = NULL;
  struct ibv_mr *mr = NULL;
  void *regComm = NULL;
  struct flagcxOneSideHandleInfo *info = NULL;

  // Grow dynamic array if needed (doubling strategy, initial capacity 4)
  if (heteroComm->oneSideHandleCount >= heteroComm->oneSideHandleCapacity) {
    int newCap = heteroComm->oneSideHandleCapacity == 0
                     ? 4
                     : heteroComm->oneSideHandleCapacity * 2;
    struct flagcxOneSideHandleInfo **newArr =
        (struct flagcxOneSideHandleInfo **)realloc(
            heteroComm->oneSideHandles,
            newCap * sizeof(struct flagcxOneSideHandleInfo *));
    if (newArr == NULL) {
      WARN("flagcxOneSideRegister: realloc failed");
      return flagcxSystemError;
    }
    // Zero-init new slots
    for (int i = heteroComm->oneSideHandleCapacity; i < newCap; i++)
      newArr[i] = NULL;
    heteroComm->oneSideHandles = newArr;
    heteroComm->oneSideHandleCapacity = newCap;
  }

  bool isFirstHandle = (heteroComm->oneSideHandleCount == 0);

  FLAGCXCHECKGOTO(flagcxCalloc(&info, 1), res, fail);

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
    if (heteroComm->netAdaptor->name &&
        strcmp(heteroComm->netAdaptor->name, "IB") == 0) {
      struct flagcxIbRecvComm *ibRecvComm = (struct flagcxIbRecvComm *)regComm;
      regComm = (void *)&ibRecvComm->base;
    }
  }

  // Register MR for this buffer
  {
    int type = FLAGCX_PTR_CUDA;
    int dmaBufFd = -1;

    // For VMM-allocated memory, use DMA-BUF registration (ibv_reg_dmabuf_mr)
    // instead of nvidia-peermem (ibv_reg_mr). nvidia-peermem cannot correctly
    // map VMM memory for RDMA, causing IBV_WC_REM_ACCESS_ERR at runtime.
    if (flagcxParamVmmEnable() &&
        deviceAdaptor->getHandleForAddressRange != NULL &&
        heteroComm->netAdaptor->regMrDmaBuf != NULL) {
      flagcxResult_t fdRes = deviceAdaptor->getHandleForAddressRange(
          (void *)&dmaBufFd, buff, size, 0);
      if (fdRes == flagcxSuccess && dmaBufFd >= 0) {
        type = FLAGCX_PTR_DMABUF;
        INFO(FLAGCX_REG,
             "[OneSideRegister] using DMA-BUF: buff=%p size=%zu fd=%d", buff,
             size, dmaBufFd);
      } else {
        INFO(FLAGCX_REG,
             "[OneSideRegister] getHandleForAddressRange failed (res=%d), "
             "falling back to nvidia-peermem",
             (int)fdRes);
        dmaBufFd = -1;
      }
    }

    if (dmaBufFd >= 0) {
      res = heteroComm->netAdaptor->regMrDmaBuf(
          regComm, buff, size, type, 0ULL, dmaBufFd, FLAGCX_NET_MR_FLAG_NONE,
          &mrHandle);
      close(dmaBufFd);
    } else {
      res = heteroComm->netAdaptor->regMr(regComm, buff, size, type,
                                          FLAGCX_NET_MR_FLAG_NONE, &mrHandle);
    }
  }
  if (res != flagcxSuccess || mrHandle == NULL) {
    INFO(FLAGCX_REG, "flagcxOneSideRegister: regMr failed, res=%d", res);
    res = flagcxNotSupported;
    goto fail_mesh;
  }

  {
    struct flagcxIbMrHandle *localMrHandle =
        (struct flagcxIbMrHandle *)mrHandle;
    mr = localMrHandle->mrs[0];
  }

  // Allgather MR info
  {
    int nranks = heteroComm->nRanks;
    FLAGCXCHECKGOTO(flagcxCalloc(&info->baseVas, nranks), res, fail_mr);
    FLAGCXCHECKGOTO(flagcxCalloc(&info->rkeys, nranks), res, fail_mr);
    FLAGCXCHECKGOTO(flagcxCalloc(&info->lkeys, nranks), res, fail_mr);

    info->baseVas[heteroComm->rank] = (uintptr_t)buff;
    info->regionSize = size;
    info->rkeys[heteroComm->rank] = mr->rkey;
    info->lkeys[heteroComm->rank] = mr->lkey;
    info->localMrHandle = mrHandle;

    FLAGCXCHECKGOTO(bootstrapCollAllGather(heteroComm->bootstrap,
                                           (void *)info->baseVas,
                                           sizeof(uintptr_t)),
                    res, fail_mr);
    FLAGCXCHECKGOTO(bootstrapCollAllGather(heteroComm->bootstrap,
                                           (void *)info->rkeys,
                                           sizeof(uint32_t)),
                    res, fail_mr);
    FLAGCXCHECKGOTO(bootstrapCollAllGather(heteroComm->bootstrap,
                                           (void *)info->lkeys,
                                           sizeof(uint32_t)),
                    res, fail_mr);

    int slot = heteroComm->oneSideHandleCount;
    heteroComm->oneSideHandles[slot] = info;
    heteroComm->oneSideHandleCount = slot + 1;

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
      INFO(FLAGCX_REG, "  Rank %d: base_va=0x%lx, rkey=0x%x, lkey=0x%x", i,
           info->baseVas[i], info->rkeys[i], info->lkeys[i]);
    }
  }

  return flagcxSuccess;

fail_mr:
  if (info) {
    free(info->lkeys);
    free(info->rkeys);
    free(info->baseVas);
  }
  if (regComm && mrHandle)
    heteroComm->netAdaptor->deregMr(regComm, mrHandle);
fail_mesh:
  if (isFirstHandle) {
    // Clean up per-context full-mesh connections on first-handle failure
    for (int ctx = 0; ctx < info->nContexts; ctx++) {
      if (info->contextSendComms && info->contextSendComms[ctx]) {
        for (int i = 0; i < heteroComm->nRanks; i++) {
          if (info->contextSendComms[ctx][i])
            heteroComm->netAdaptor->closeSend(info->contextSendComms[ctx][i]);
          if (info->contextRecvComms[ctx][i])
            heteroComm->netAdaptor->closeRecv(info->contextRecvComms[ctx][i]);
        }
        free(info->contextSendComms[ctx]);
        free(info->contextRecvComms[ctx]);
      }
    }
    free(info->contextSendComms);
    free(info->contextRecvComms);
  }
fail_info:
  free(info);
fail:
  return res;
}

flagcxResult_t flagcxOneSideRegister(flagcxComm_t comm, void *buff,
                                     size_t size) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  if (comm->heteroComm == nullptr)
    return flagcxNotSupported;
  return flagcxOneSideRegisterInternal(comm->heteroComm, buff, size);
}

flagcxResult_t flagcxOneSideDeregister(struct flagcxHeteroComm *heteroComm) {
  if (heteroComm == NULL)
    return flagcxInternalError;

  // Deregister all data handles in reverse order
  for (int i = heteroComm->oneSideHandleCount - 1; i >= 0; i--) {
    struct flagcxOneSideHandleInfo *info = heteroComm->oneSideHandles[i];
    if (info == NULL)
      continue;

    if (heteroComm->netAdaptor != NULL) {
      // Deregister MR
      if (info->localMrHandle != NULL && info->localRecvComm != NULL) {
        void *regComm = info->localRecvComm;
        if (heteroComm->netAdaptor->name &&
            strcmp(heteroComm->netAdaptor->name, "IB") == 0) {
          struct flagcxIbRecvComm *ibRecvComm =
              (struct flagcxIbRecvComm *)regComm;
          regComm = (void *)&ibRecvComm->base;
        }
        heteroComm->netAdaptor->deregMr(regComm, info->localMrHandle);
      }

      // Close per-context full-mesh connections (only in the first handle)
      if (info->contextSendComms != NULL) {
        for (int ctx = 0; ctx < info->nContexts; ctx++) {
          if (info->contextSendComms[ctx] != NULL) {
            for (int j = 0; j < info->nRanks; j++) {
              if (info->contextSendComms[ctx][j])
                heteroComm->netAdaptor->closeSend(
                    info->contextSendComms[ctx][j]);
              if (info->contextRecvComms[ctx][j])
                heteroComm->netAdaptor->closeRecv(
                    info->contextRecvComms[ctx][j]);
            }
            free(info->contextSendComms[ctx]);
            free(info->contextRecvComms[ctx]);
          }
        }
        free(info->contextSendComms);
        free(info->contextRecvComms);
      }
    }

    free(info->baseVas);
    free(info->rkeys);
    free(info->lkeys);
    free(info);
    heteroComm->oneSideHandles[i] = NULL;
  }

  free(heteroComm->oneSideHandles);
  heteroComm->oneSideHandles = NULL;
  heteroComm->oneSideHandleCount = 0;
  heteroComm->oneSideHandleCapacity = 0;
  return flagcxSuccess;
}

flagcxResult_t flagcxOneSideSignalRegister(const flagcxComm_t comm, void *buff,
                                           size_t size, int ptrType) {
  if (useHomoComm(comm) && !useHeteroComm()) {
    return flagcxSuccess;
  }

  // Per-heteroComm dedup: skip if already registered
  struct flagcxHeteroComm *heteroComm = comm->heteroComm;
  struct flagcxOneSideHandleInfo *existing = heteroComm->signalHandle;
  if (existing != NULL) {
    if (existing->baseVas != NULL &&
        existing->baseVas[comm->rank] != (uintptr_t)buff) {
      WARN("flagcxOneSideSignalRegister: comm %p already registered with a "
           "different buffer",
           (void *)comm);
    }
    return flagcxSuccess;
  }

  // Validate ptrType — only known pointer types are accepted.
  if (ptrType != FLAGCX_PTR_HOST && ptrType != FLAGCX_PTR_CUDA &&
      ptrType != FLAGCX_PTR_DMABUF) {
    WARN("flagcxOneSideSignalRegister: invalid ptrType %d", ptrType);
    return flagcxInvalidArgument;
  }

  if (heteroComm == NULL || heteroComm->netAdaptor == NULL ||
      heteroComm->netAdaptor->iputSignal == NULL ||
      heteroComm->netAdaptor->regMr == NULL) {
    return flagcxSuccess;
  }

  if (heteroComm->bootstrap == NULL) {
    INFO(FLAGCX_REG, "flagcxOneSideSignalRegister: bootstrap is NULL");
    return flagcxNotSupported;
  }

  // Signal registration reuses full-mesh connections from this heteroComm's
  // first data handle.  Requires at least one data handle first.
  struct flagcxOneSideHandleInfo *firstDataHandle = NULL;
  if (heteroComm->oneSideHandleCount > 0 &&
      heteroComm->oneSideHandles[0] != NULL &&
      heteroComm->oneSideHandles[0]->fullRecvComms != NULL) {
    firstDataHandle = heteroComm->oneSideHandles[0];
  }
  if (firstDataHandle == NULL) {
    INFO(FLAGCX_REG,
         "flagcxOneSideSignalRegister: no full-mesh connections for "
         "this heteroComm, register a data buffer first");
    return flagcxNotSupported;
  }

  flagcxResult_t res = flagcxSuccess;
  void *mrHandle = NULL;
  struct ibv_mr *mr = NULL;
  void *regComm = NULL;
  struct flagcxOneSideHandleInfo *info = NULL;

  // Use self recvComm from this comm's first data handle for MR registration
  // (PD match)
  void *selfRecvComm = firstDataHandle->fullRecvComms[heteroComm->rank];
  regComm = selfRecvComm;
  if (heteroComm->netAdaptor->name &&
      strcmp(heteroComm->netAdaptor->name, "IB") == 0) {
    struct flagcxIbRecvComm *ibRecvComm = (struct flagcxIbRecvComm *)regComm;
    regComm = (void *)&ibRecvComm->base;
  }

  {
    int dmaBufFd = -1;

    // For VMM-allocated signal buffers (FLAGCX_PTR_CUDA + VMM), use DMA-BUF
    if (ptrType == FLAGCX_PTR_CUDA && flagcxParamVmmEnable() &&
        deviceAdaptor->getHandleForAddressRange != NULL &&
        heteroComm->netAdaptor->regMrDmaBuf != NULL) {
      flagcxResult_t fdRes = deviceAdaptor->getHandleForAddressRange(
          (void *)&dmaBufFd, buff, size, 0);
      if (fdRes == flagcxSuccess && dmaBufFd >= 0) {
        ptrType = FLAGCX_PTR_DMABUF;
        INFO(FLAGCX_REG,
             "[OneSideSignalRegister] using DMA-BUF: buff=%p size=%zu fd=%d",
             buff, size, dmaBufFd);
      } else {
        INFO(FLAGCX_REG,
             "[OneSideSignalRegister] getHandleForAddressRange failed (res=%d),"
             " falling back to nvidia-peermem",
             (int)fdRes);
        dmaBufFd = -1;
      }
    }

    if (dmaBufFd >= 0) {
      res = heteroComm->netAdaptor->regMrDmaBuf(
          regComm, buff, size, ptrType, 0ULL, dmaBufFd,
          FLAGCX_NET_MR_FLAG_FORCE_SO, &mrHandle);
      close(dmaBufFd);
    } else {
      res = heteroComm->netAdaptor->regMr(
          regComm, buff, size, ptrType, FLAGCX_NET_MR_FLAG_FORCE_SO, &mrHandle);
    }
  }
  if (res != flagcxSuccess || mrHandle == NULL) {
    INFO(FLAGCX_REG, "flagcxOneSideSignalRegister: regMr failed, res=%d", res);
    return flagcxNotSupported;
  }

  {
    struct flagcxIbMrHandle *localMrHandle =
        (struct flagcxIbMrHandle *)mrHandle;
    mr = localMrHandle->mrs[0];
  }

  {
    int nranks = heteroComm->nRanks;
    FLAGCXCHECKGOTO(flagcxCalloc(&info, 1), res, fail_mr);
    FLAGCXCHECKGOTO(flagcxCalloc(&info->baseVas, nranks), res, fail_mr);
    FLAGCXCHECKGOTO(flagcxCalloc(&info->rkeys, nranks), res, fail_mr);
    FLAGCXCHECKGOTO(flagcxCalloc(&info->lkeys, nranks), res, fail_mr);

    info->baseVas[heteroComm->rank] = (uintptr_t)buff;
    info->regionSize = size;
    info->rkeys[heteroComm->rank] = mr->rkey;
    info->lkeys[heteroComm->rank] = mr->lkey;
    info->localMrHandle = mrHandle;
    info->localRecvComm = selfRecvComm;
    info->signalIpcSlot = -1;
    FLAGCXCHECKGOTO(bootstrapCollAllGather(heteroComm->bootstrap,
                                           (void *)info->baseVas,
                                           sizeof(uintptr_t)),
                    res, fail_mr);
    FLAGCXCHECKGOTO(bootstrapCollAllGather(heteroComm->bootstrap,
                                           (void *)info->rkeys,
                                           sizeof(uint32_t)),
                    res, fail_mr);
    FLAGCXCHECKGOTO(bootstrapCollAllGather(heteroComm->bootstrap,
                                           (void *)info->lkeys,
                                           sizeof(uint32_t)),
                    res, fail_mr);
    heteroComm->signalHandle = info;
    INFO(FLAGCX_REG, "Signal register allgather results (rank %d, nranks %d):",
         heteroComm->rank, nranks);
    for (int i = 0; i < nranks; i++) {
      INFO(FLAGCX_REG, "  Rank %d: base_va=0x%lx, rkey=0x%x, lkey=0x%x", i,
           info->baseVas[i], info->rkeys[i], info->lkeys[i]);
    }
  }

  // Register signal buffer in IPC table for intra-node D2D bypass.
  // Skip when VMM is enabled: VMM buffers don't support cudaIpcGetMemHandle,
  // and flagcxDevMemCreate uses flat VA peer access (Priority 1) instead.
  if (ptrType == FLAGCX_PTR_CUDA && !flagcxParamVmmEnable()) {
    int idx = buildIpcPeerPointers(comm, buff, size);
    if (idx >= 0) {
      info->signalIpcSlot = idx;
      INFO(FLAGCX_REG, "Signal buffer IPC registered (slot %d) for D2D bypass",
           idx);
    }
  }

  return flagcxSuccess;

fail_mr:
  if (info) {
    free(info->lkeys);
    free(info->rkeys);
    free(info->baseVas);
    free(info);
  }
  heteroComm->netAdaptor->deregMr(regComm, mrHandle);
  return res;
}

flagcxResult_t flagcxOneSideSignalDeregister(flagcxComm_t comm) {
  if (comm == NULL || comm->heteroComm == NULL)
    return flagcxInternalError;
  struct flagcxHeteroComm *heteroComm = comm->heteroComm;
  struct flagcxOneSideHandleInfo *info = heteroComm->signalHandle;
  if (info == NULL)
    return flagcxSuccess;

  if (heteroComm->netAdaptor != NULL) {
    if (info->localMrHandle != NULL && info->localRecvComm != NULL) {
      void *regComm = info->localRecvComm;
      if (heteroComm->netAdaptor->name &&
          strcmp(heteroComm->netAdaptor->name, "IB") == 0) {
        struct flagcxIbRecvComm *ibRecvComm =
            (struct flagcxIbRecvComm *)regComm;
        regComm = (void *)&ibRecvComm->base;
      }
      heteroComm->netAdaptor->deregMr(regComm, info->localMrHandle);
    }
  }

  // Release IPC table slot (resources deferred to comm destroy).
  if (info->signalIpcSlot >= 0) {
    releaseIpcTableSlot(comm, info->signalIpcSlot);
  }

  free(info->baseVas);
  free(info->rkeys);
  free(info->lkeys);
  free(info);
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
      heteroComm->netAdaptor->regMr == NULL) {
    INFO(FLAGCX_REG, "flagcxOneSideStagingRegister: heteroComm is NULL");
    return flagcxSuccess;
  }

  if (heteroComm->bootstrap == NULL) {
    INFO(FLAGCX_REG, "flagcxOneSideStagingRegister: bootstrap is NULL");
    return flagcxNotSupported;
  }

  // Staging registration reuses full-mesh connections from this heteroComm's
  // first data handle.  Requires at least one data handle first.
  struct flagcxOneSideHandleInfo *firstDataHandleStg = NULL;
  if (heteroComm->oneSideHandleCount > 0 &&
      heteroComm->oneSideHandles[0] != NULL &&
      heteroComm->oneSideHandles[0]->fullRecvComms != NULL) {
    firstDataHandleStg = heteroComm->oneSideHandles[0];
  }
  if (firstDataHandleStg == NULL) {
    INFO(FLAGCX_REG,
         "flagcxOneSideStagingRegister: no full-mesh connections for "
         "this heteroComm, register a data buffer first");
    return flagcxNotSupported;
  }

  flagcxResult_t res = flagcxSuccess;
  void *mrHandle = NULL;
  struct ibv_mr *mr = NULL;
  void *regComm = NULL;
  struct flagcxOneSideHandleInfo *info = NULL;

  // Use self recvComm from this comm's first data handle for MR registration
  // (PD match)
  void *selfRecvComm = firstDataHandleStg->fullRecvComms[heteroComm->rank];
  regComm = selfRecvComm;
  if (heteroComm->netAdaptor->name &&
      strcmp(heteroComm->netAdaptor->name, "IB") == 0) {
    struct flagcxIbRecvComm *ibRecvComm = (struct flagcxIbRecvComm *)regComm;
    regComm = (void *)&ibRecvComm->base;
  }

  {
    int type = FLAGCX_PTR_HOST;
    res =
        heteroComm->netAdaptor->regMr(regComm, buff, size, type, 0, &mrHandle);
  }
  if (res != flagcxSuccess || mrHandle == NULL) {
    INFO(FLAGCX_REG, "flagcxOneSideStagingRegister: regMr failed, res=%d", res);
    return flagcxNotSupported;
  }

  {
    struct flagcxIbMrHandle *localMrHandle =
        (struct flagcxIbMrHandle *)mrHandle;
    mr = localMrHandle->mrs[0];
  }

  {
    int nranks = heteroComm->nRanks;
    FLAGCXCHECKGOTO(flagcxCalloc(&info, 1), res, fail_mr);
    FLAGCXCHECKGOTO(flagcxCalloc(&info->baseVas, nranks), res, fail_mr);
    FLAGCXCHECKGOTO(flagcxCalloc(&info->rkeys, nranks), res, fail_mr);
    FLAGCXCHECKGOTO(flagcxCalloc(&info->lkeys, nranks), res, fail_mr);

    info->baseVas[heteroComm->rank] = (uintptr_t)buff;
    info->regionSize = size;
    info->rkeys[heteroComm->rank] = mr->rkey;
    info->lkeys[heteroComm->rank] = mr->lkey;
    info->localMrHandle = mrHandle;
    info->localRecvComm = selfRecvComm;

    FLAGCXCHECKGOTO(bootstrapCollAllGather(heteroComm->bootstrap,
                                           (void *)info->baseVas,
                                           sizeof(uintptr_t)),
                    res, fail_mr);
    FLAGCXCHECKGOTO(bootstrapCollAllGather(heteroComm->bootstrap,
                                           (void *)info->rkeys,
                                           sizeof(uint32_t)),
                    res, fail_mr);
    FLAGCXCHECKGOTO(bootstrapCollAllGather(heteroComm->bootstrap,
                                           (void *)info->lkeys,
                                           sizeof(uint32_t)),
                    res, fail_mr);
    heteroComm->stagingHandle = info;
    INFO(FLAGCX_REG, "Staging register allgather results (rank %d, nranks %d):",
         heteroComm->rank, nranks);
    for (int i = 0; i < nranks; i++) {
      INFO(FLAGCX_REG, "  Rank %d: base_va=0x%lx, rkey=0x%x, lkey=0x%x", i,
           info->baseVas[i], info->rkeys[i], info->lkeys[i]);
    }
  }

  return flagcxSuccess;

fail_mr:
  if (info) {
    free(info->lkeys);
    free(info->rkeys);
    free(info->baseVas);
    free(info);
  }
  heteroComm->netAdaptor->deregMr(regComm, mrHandle);
  return res;
}

flagcxResult_t flagcxOneSideStagingDeregister(const flagcxComm_t comm) {
  if (comm == NULL || comm->heteroComm == NULL)
    return flagcxInternalError;
  struct flagcxHeteroComm *heteroComm = comm->heteroComm;
  struct flagcxOneSideHandleInfo *info = heteroComm->stagingHandle;
  if (info == NULL)
    return flagcxSuccess;

  if (heteroComm->netAdaptor != NULL) {
    if (info->localMrHandle != NULL && info->localRecvComm != NULL) {
      void *regComm = info->localRecvComm;
      if (heteroComm->netAdaptor->name &&
          strcmp(heteroComm->netAdaptor->name, "IB") == 0) {
        struct flagcxIbRecvComm *ibRecvComm =
            (struct flagcxIbRecvComm *)regComm;
        regComm = (void *)&ibRecvComm->base;
      }
      heteroComm->netAdaptor->deregMr(regComm, info->localMrHandle);
    }
  }

  free(info->baseVas);
  free(info->rkeys);
  free(info->lkeys);
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
      heteroComm->netAdaptor->regMr == NULL)
    return flagcxNotSupported;

  if (comm->bootstrap == NULL)
    return flagcxNotSupported;

  struct flagcxNetAdaptor *net = heteroComm->netAdaptor;
  flagcxResult_t res = flagcxSuccess;
  void *mrHandle = NULL;
  uint32_t rkey = 0, lkey = 0;
  uintptr_t baseVa = 0;
  struct flagcxOneSideHandleInfo *info = NULL;

  // Leaders (recvComm != NULL): register MR and extract keys
  if (recvComm != NULL && buff != NULL && size > 0) {
    void *regComm = recvComm;
    if (net->name && strcmp(net->name, "IB") == 0) {
      struct flagcxIbRecvComm *ibRecvComm = (struct flagcxIbRecvComm *)regComm;
      regComm = (void *)&ibRecvComm->base;
    }
    res = net->regMr(regComm, buff, size, FLAGCX_PTR_HOST,
                     FLAGCX_NET_MR_FLAG_FORCE_SO, &mrHandle);
    if (res != flagcxSuccess || mrHandle == NULL) {
      INFO(FLAGCX_REG, "flagcxOneSideBarrierRegister: regMr failed, res=%d",
           res);
      return flagcxNotSupported;
    }
    struct flagcxIbMrHandle *ibMrHandle = (struct flagcxIbMrHandle *)mrHandle;
    struct ibv_mr *mr = ibMrHandle->mrs[0];
    rkey = mr->rkey;
    lkey = mr->lkey;
    baseVa = (uintptr_t)buff;
  }

  // ALL ranks: allocate info, populate own entry, AllGather
  {
    int nranks = comm->nranks;
    int myRank = comm->rank;
    FLAGCXCHECKGOTO(flagcxCalloc(&info, 1), res, fail_mr);
    FLAGCXCHECKGOTO(flagcxCalloc(&info->baseVas, nranks), res, fail_mr);
    FLAGCXCHECKGOTO(flagcxCalloc(&info->rkeys, nranks), res, fail_mr);
    FLAGCXCHECKGOTO(flagcxCalloc(&info->lkeys, nranks), res, fail_mr);

    info->baseVas[myRank] = baseVa;
    info->rkeys[myRank] = rkey;
    info->lkeys[myRank] = lkey;
    info->localMrHandle = mrHandle;
    info->localRecvComm = recvComm;

    FLAGCXCHECKGOTO(bootstrapCollAllGather(comm->bootstrap,
                                           (void *)info->baseVas,
                                           sizeof(uintptr_t)),
                    res, fail_mr);
    FLAGCXCHECKGOTO(bootstrapCollAllGather(comm->bootstrap, (void *)info->rkeys,
                                           sizeof(uint32_t)),
                    res, fail_mr);
    FLAGCXCHECKGOTO(bootstrapCollAllGather(comm->bootstrap, (void *)info->lkeys,
                                           sizeof(uint32_t)),
                    res, fail_mr);

    INFO(FLAGCX_REG,
         "Barrier register allgather results (rank %d, nranks %d):", myRank,
         nranks);
    for (int i = 0; i < nranks; i++) {
      INFO(FLAGCX_REG, "  Rank %d: base_va=0x%lx, rkey=0x%x, lkey=0x%x", i,
           info->baseVas[i], info->rkeys[i], info->lkeys[i]);
    }
  }

  *outInfo = info;
  return flagcxSuccess;

fail_mr:
  if (info) {
    free(info->lkeys);
    free(info->rkeys);
    free(info->baseVas);
    free(info);
  }
  if (mrHandle != NULL) {
    void *regComm = recvComm;
    if (net->name && strcmp(net->name, "IB") == 0) {
      struct flagcxIbRecvComm *ibRecvComm = (struct flagcxIbRecvComm *)regComm;
      regComm = (void *)&ibRecvComm->base;
    }
    net->deregMr(regComm, mrHandle);
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
  if (heteroComm != NULL && heteroComm->netAdaptor != NULL) {
    if (info->localMrHandle != NULL && info->localRecvComm != NULL) {
      void *regComm = info->localRecvComm;
      if (heteroComm->netAdaptor->name &&
          strcmp(heteroComm->netAdaptor->name, "IB") == 0) {
        struct flagcxIbRecvComm *ibRecvComm =
            (struct flagcxIbRecvComm *)regComm;
        regComm = (void *)&ibRecvComm->base;
      }
      heteroComm->netAdaptor->deregMr(regComm, info->localMrHandle);
    }
  }

  free(info->baseVas);
  free(info->rkeys);
  free(info->lkeys);
  free(info);
  return flagcxSuccess;
}

flagcxResult_t flagcxCommRegister(const flagcxComm_t comm, void *buff,
                                  size_t size, void **handle) {
  if (comm != nullptr) {
    FLAGCXCHECK(flagcxEnsureCommReady(comm));
  }

  if (buff == NULL || size == 0) {
    WARN("Invalid buffer or size for buffer registration.");
    return flagcxInvalidArgument;
  }

  // Step 1: Register in globalRegPool (both paths)
  // Key: heteroComm if available (p2p/net downstream use it), else homoComm
  // If comm is NULL, register in global pool only (GLOBAL_POOL_KEY)
  void *regKey = nullptr;
  if (comm != nullptr) {
    regKey =
        comm->heteroComm ? (void *)comm->heteroComm : (void *)comm->homoComm;
  }
  globalRegPool.registerBuffer(regKey, buff, size);
  flagcxRegItem *regItem = globalRegPool.getItem(regKey, buff);

  *handle = reinterpret_cast<void *>(regItem);

  // Null comm: pool-only registration, skip backend steps
  if (comm == nullptr) {
    return flagcxSuccess;
  }

  uintptr_t thisCommKey = reinterpret_cast<uintptr_t>(regKey);

  flagcxResult_t res = flagcxSuccess;

  // Step 2a: Homo path — backend CCL registration
  // NCCL handles IPC/VMM internally via ncclCommRegister, so skip Step 2b
  // (cudaIpcGetMemHandle is incompatible with ncclMemAlloc VMM buffers)
  // and Step 3 (one-sided MR registration, hetero-only).
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

  // Step 2b: Create IPC handle for the buffer (hetero path only)
  // Write-once: if localIpcHandleData is already populated, skip.
  // Note: cudaIpcGetMemHandle is incompatible with VMM buffers (cuMemCreate/
  // cuMemMap). When it fails, we skip IPC handle creation and still proceed
  // to Step 3 (one-sided MR registration) which handles VMM buffers correctly.
  {
    char zeros[sizeof(flagcxIpcHandleData)] = {};
    if (memcmp(&regItem->localIpcHandleData, zeros,
               sizeof(flagcxIpcHandleData)) == 0) {
      flagcxIpcMemHandle_t handlePtr = nullptr;
      size_t ipcSize = 0;
      res = deviceAdaptor->ipcMemHandleCreate(&handlePtr, &ipcSize);
      if (res != flagcxSuccess) {
        res = flagcxSuccess;
        goto skip_ipc;
      }
      res = deviceAdaptor->ipcMemHandleGet(handlePtr, buff);
      if (res != flagcxSuccess) {
        deviceAdaptor->ipcMemHandleFree(handlePtr);
        INFO(FLAGCX_REG,
             "flagcxCommRegister: ipcMemHandleGet failed (%d) for buff %p "
             "(likely VMM memory), skipping IPC handle",
             (int)res, buff);
        res = flagcxSuccess;
        goto skip_ipc;
      }
      if (ipcSize > sizeof(flagcxIpcHandleData)) {
        deviceAdaptor->ipcMemHandleFree(handlePtr);
        INFO(FLAGCX_REG,
             "flagcxCommRegister: ipcSize %zu exceeds storage, skipping IPC",
             ipcSize);
        goto skip_ipc;
      }
      memcpy(&regItem->localIpcHandleData, handlePtr, ipcSize);
      deviceAdaptor->ipcMemHandleFree(handlePtr);
    }
  }
skip_ipc:

  // Step 3: One-sided MR registration (hetero path only)
  {
    flagcxResult_t regRes =
        flagcxOneSideRegisterInternal(comm->heteroComm, buff, size);
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

flagcxResult_t flagcxCommDeregister(const flagcxComm_t comm, void *handle) {
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

flagcxResult_t flagcxCommWindowRegister(flagcxComm_t comm, void *buff,
                                        size_t size, flagcxWindow_t *win,
                                        int winFlags) {
  if (win == nullptr || *win != nullptr) {
    return flagcxInvalidArgument;
  }
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
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

    // Initialize D2D bypass if this is the first successful symmetric window
    // and RMA proxy is running but IPC not yet initialized.
    if (res == flagcxSuccess && *win != NULL && (*win)->defaultBase != NULL &&
        (*win)->defaultBase->flatBase != NULL) {
      // D2D IPC init deferred to first stream-path use (when both data
      // windows and signal buffer are registered).
    }

    return res;
  }
  *win = nullptr;
  return flagcxSuccess;
}

flagcxResult_t flagcxCommWindowDeregister(flagcxComm_t comm,
                                          flagcxWindow_t win) {
  if (win == nullptr) {
    return flagcxSuccess;
  }
  FLAGCXCHECK(flagcxEnsureCommReady(comm));

  flagcxHeteroComm_t hetero = comm->heteroComm;

  // Clear symWin pointer in oneSideHandles to prevent use-after-free
  if (hetero != NULL && win->defaultBase != NULL) {
    for (int h = 0; h < hetero->oneSideHandleCount; h++) {
      struct flagcxOneSideHandleInfo *info = hetero->oneSideHandles[h];
      if (info != NULL && info->symWin == win->defaultBase) {
        info->symWin = NULL;
        break;
      }
    }
    // Rebuild IPC state since D2D pointers derived from symWin are now invalid
    if (hetero->rmaProxy != NULL && hetero->rmaProxy->ipcState != NULL) {
      flagcxHeteroRmaIpcDestroy(hetero);
      // Will be lazily re-initialized on next D2D attempt
    }
  }

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
  return flagcxSymWindowDeregister(comm->heteroComm, win);
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

  // On the default path with multicast, use VMM-backed allocation so that
  // symPhysAlloc can export the physical handle for flat VA + multicast.
  bool needVmmAlloc = false;
#ifndef FLAGCX_DEVICE_API_VENDOR
  {
    int mcSupported = 0;
    if (deviceAdaptor->symMulticastSupported)
      deviceAdaptor->symMulticastSupported(&mcSupported);
    needVmmAlloc = (mcSupported && deviceAdaptor->gdrMemAlloc != nullptr);
  }
#endif

  if (needVmmAlloc) {
    FLAGCXCHECKGOTO(deviceAdaptor->gdrMemAlloc(&state->sendStagedBuff,
                                               state->stagedBuffSize, nullptr),
                    res, fail);
    FLAGCXCHECKGOTO(deviceAdaptor->gdrMemAlloc(&state->recvStagedBuff,
                                               state->stagedBuffSize, nullptr),
                    res, fail);
    state->stagedVmmAlloc = true;
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
      FLAGCXCHECKGOTO(
          flagcxSymWindowRegister(comm->heteroComm, state->sendStagedBuff,
                                  state->stagedBuffSize, &state->sendStagedWin,
                                  FLAGCX_WIN_COLL_SYMMETRIC),
          res, fail);
      // Register recv staged buffer
      FLAGCXCHECKGOTO(
          flagcxSymWindowRegister(comm->heteroComm, state->recvStagedBuff,
                                  state->stagedBuffSize, &state->recvStagedWin,
                                  FLAGCX_WIN_COLL_SYMMETRIC),
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
  if (state->devComm)
    flagcxDevCommDestroy(comm, state->devComm);
  if (state->recvStagedMem)
    flagcxDevMemDestroy(comm, state->recvStagedMem);
  if (state->sendStagedMem)
    flagcxDevMemDestroy(comm, state->sendStagedMem);
  if (state->recvStagedWin) {
    if (state->recvStagedWin->vendorBase != nullptr) {
      cclAdaptors[flagcxCCLAdaptorDevice]->commWindowDeregister(
          comm->homoComm, state->recvStagedWin->vendorBase);
      free(state->recvStagedWin);
    } else if (state->recvStagedWin->isSymmetricDefault) {
      flagcxSymWindowDeregister(comm->heteroComm, state->recvStagedWin);
    } else {
      free(state->recvStagedWin);
    }
  }
  if (state->sendStagedWin) {
    if (state->sendStagedWin->vendorBase != nullptr) {
      cclAdaptors[flagcxCCLAdaptorDevice]->commWindowDeregister(
          comm->homoComm, state->sendStagedWin->vendorBase);
      free(state->sendStagedWin);
    } else if (state->sendStagedWin->isSymmetricDefault) {
      flagcxSymWindowDeregister(comm->heteroComm, state->sendStagedWin);
    } else {
      free(state->sendStagedWin);
    }
  }
  if (state->recvStagedBuff) {
    if (state->stagedVmmAlloc)
      deviceAdaptor->gdrMemFree(state->recvStagedBuff, nullptr);
    else
      cclAdaptors[flagcxCCLAdaptorDevice]->memFree(state->recvStagedBuff);
  }
  if (state->sendStagedBuff) {
    if (state->stagedVmmAlloc)
      deviceAdaptor->gdrMemFree(state->sendStagedBuff, nullptr);
    else
      cclAdaptors[flagcxCCLAdaptorDevice]->memFree(state->sendStagedBuff);
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
  if (state->devComm)
    flagcxDevCommDestroy(comm, state->devComm);
  if (state->sendStagedMem)
    flagcxDevMemDestroy(comm, state->sendStagedMem);
  if (state->recvStagedMem)
    flagcxDevMemDestroy(comm, state->recvStagedMem);
  if (state->sendStagedWin) {
    if (state->sendStagedWin->vendorBase != nullptr) {
      cclAdaptors[flagcxCCLAdaptorDevice]->commWindowDeregister(
          comm->homoComm, state->sendStagedWin->vendorBase);
      free(state->sendStagedWin);
    } else if (state->sendStagedWin->isSymmetricDefault) {
      flagcxSymWindowDeregister(comm->heteroComm, state->sendStagedWin);
    } else {
      free(state->sendStagedWin);
    }
  }
  if (state->recvStagedWin) {
    if (state->recvStagedWin->vendorBase != nullptr) {
      cclAdaptors[flagcxCCLAdaptorDevice]->commWindowDeregister(
          comm->homoComm, state->recvStagedWin->vendorBase);
      free(state->recvStagedWin);
    } else if (state->recvStagedWin->isSymmetricDefault) {
      flagcxSymWindowDeregister(comm->heteroComm, state->recvStagedWin);
    } else {
      free(state->recvStagedWin);
    }
  }
  if (state->sendStagedBuff) {
    if (state->stagedVmmAlloc)
      deviceAdaptor->gdrMemFree(state->sendStagedBuff, nullptr);
    else
      cclAdaptors[flagcxCCLAdaptorDevice]->memFree(state->sendStagedBuff);
  }
  if (state->recvStagedBuff) {
    if (state->stagedVmmAlloc)
      deviceAdaptor->gdrMemFree(state->recvStagedBuff, nullptr);
    else
      cclAdaptors[flagcxCCLAdaptorDevice]->memFree(state->recvStagedBuff);
  }
  free(state);
  comm->devCommState = nullptr;
  return flagcxSuccess;
}

flagcxResult_t flagcxHomoCommInit(flagcxUniqueId_t commId,
                                  flagcxUniqueId *uniqueIdData,
                                  struct bootstrapState *state,
                                  flagcxComm_t comm,
                                  flagcxInnerComm_t *homoComm /*out*/) {
  int rank = comm->rank;
  int nranks = comm->nranks;
  memset((void *)commId, 0, sizeof(*commId));
  memset((void *)uniqueIdData, 0, nranks * sizeof(flagcxUniqueId));
  if (comm->homoRank == 0) {
    cclAdaptors[flagcxCCLAdaptorDevice]->getUniqueId(&commId);
  }
  if (comm->homoRank == 0) {
    memcpy((void *)&uniqueIdData[rank], (void *)commId, sizeof(flagcxUniqueId));
  }
  FLAGCXCHECK(bootstrapCollAllGather(state, (void *)uniqueIdData,
                                     sizeof(flagcxUniqueId)));
  FLAGCXCHECK(bootstrapCollBarrier(state, rank, nranks, 0));

  memcpy((void *)commId, (void *)&uniqueIdData[comm->homoRootRank],
         sizeof(flagcxUniqueId));
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

  flagcxUniqueId *uniqueIdData;
  FLAGCXCHECK(flagcxCalloc(&uniqueIdData, nranks));

  // Tuner init
  bool useTuner = false;
  const char *useTunerEnv = flagcxGetEnv("FLAGCX_USE_TUNER");
  if (useTunerEnv) {
    useTuner = (std::stoi(useTunerEnv) == 1) ? true : false;
  }
  INFO(FLAGCX_INIT, "Flagcx USE_TUNER flag set to %d", useTuner);
  if (useTuner) {
    (*comm)->tuner = &internalTuner;
    FLAGCXCHECK(flagcxCalloc(&(*comm)->commId, 1));
    memcpy((*comm)->commId, commId, sizeof(flagcxUniqueId));
    (*comm)->uniqueIdData = uniqueIdData;
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
        FLAGCXCHECK(
            flagcxHomoCommInit(commId, uniqueIdData, state, *comm, &innerComm));
        // Insert item into commMap
        (*comm)->commMap[tag] = innerComm;
        // For backward compatible, also assign homo_comm field.
        (*comm)->homoComm = innerComm;
      }
    }

    if (isTuningWithFlagscale) {
      // Create a default communicator based on the default config
      flagcxInnerComm_t innerComm = NULL;
      FLAGCXCHECK(
          flagcxHomoCommInit(commId, uniqueIdData, state, *comm, &innerComm));
      // Insert item into homoCommMap
      (*comm)->tunerInnerComm = innerComm;
      // For backward compatible, also assign homoComm field.
      (*comm)->homoComm = innerComm;
    }
  } else {
    (*comm)->tuner = NULL;
    FLAGCXCHECK(flagcxHomoCommInit(commId, uniqueIdData, state, *comm,
                                   &((*comm)->homoComm)));
  }

  if (!useHomoComm(*comm) || useHeteroComm()) {
    // Reset commId and hetero root rank calls flagcxHeteroGetUniqueId
    memset((void *)commId, 0, sizeof(flagcxUniqueId));
    memset((void *)uniqueIdData, 0, nranks * sizeof(flagcxUniqueId));
    if (rank == 0) {
      flagcxHeteroGetUniqueId(commId);
      memcpy((void *)&uniqueIdData[0], (void *)commId, sizeof(flagcxUniqueId));
    }
    FLAGCXCHECK(bootstrapCollAllGather(state, (void *)uniqueIdData,
                                       sizeof(flagcxUniqueId)));
    FLAGCXCHECK(bootstrapCollBarrier(state, rank, nranks, 0));

    memcpy((void *)commId, (void *)&uniqueIdData[0], sizeof(flagcxUniqueId));
    // call flagcxHeteroCommInitRank
    FLAGCXCHECK(
        flagcxHeteroCommInitRank(&(*comm)->heteroComm, nranks, *commId, rank));

    // Share ipcTable with heteroComm for intra-node D2D bypass
    (*comm)->heteroComm->ipcTable = (*comm)->ipcTable;
    (*comm)->heteroComm->ipcTableSize = FLAGCX_MAX_IPC_ENTRIES;

    // Init host cclAdaptor
    if (useHostComm() || (*comm)->hasSingleRankHomoComm) {
      if (!flagcxParamTopoDetectionDisable()) {
        FLAGCXCHECK((*comm)->heteroComm->netAdaptor->getProperties(
            (*comm)->heteroComm->netDev, bootstrapGetNetProperties()));
      }
      FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorHost]->commInitRank(
          &(*comm)->hostComm, nranks, commId, rank, state));
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
    // Reset commId and homo inter root rank calls underlying GetUniqueId
    // function for initialization of homo inter communicator
    memset((void *)commId, 0, sizeof(flagcxUniqueId));
    memset((void *)uniqueIdData, 0, nranks * sizeof(flagcxUniqueId));
    // Let homoInterRootRank call underlying GetUniqueId function
    // for initialization of homo inter communicator
    if (rank == (*comm)->homoInterRootRank) {
      cclAdaptors[flagcxCCLAdaptorDevice]->getUniqueId(&commId);
      memcpy((void *)&uniqueIdData[rank], (void *)commId,
             sizeof(flagcxUniqueId));
    }
    // Collect uniqueIdData globally
    FLAGCXCHECK(bootstrapCollAllGather(state, (void *)uniqueIdData,
                                       sizeof(flagcxUniqueId)));
    FLAGCXCHECK(bootstrapCollBarrier(state, rank, nranks, 0));
    // Call cclAdaptor->commInitRank
    if ((*comm)->homoInterRootRank != -1) {
      memcpy((void *)commId, (void *)&uniqueIdData[(*comm)->homoInterRootRank],
             sizeof(flagcxUniqueId));
      FLAGCXCHECK(cclAdaptors[flagcxCCLAdaptorDevice]->commInitRank(
          &(*comm)->homoInterComm, (*comm)->homoInterRanks, commId,
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
  if (!useTuner) {
    free(uniqueIdData);
  }

  // Initialize custom op state (non-fatal if fails)
  FLAGCXCHECK(flagcxDevCommStateInit(*comm));

  return flagcxSuccess;
}

flagcxResult_t flagcxCommFinalize(flagcxComm_t comm) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));
  FLAGCXCHECK(
      cclAdaptors[flagcxCCLAdaptorDevice]->commFinalize(comm->homoComm));
  if (!useHomoComm(comm)) {
    // TODO: to be implemented
    return flagcxNotSupported;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxCommDestroy(flagcxComm_t comm) {
  FLAGCXCHECK(flagcxEnsureCommReady(comm));

  // Destroy cluster info
  free(comm->clusterIds);
  free(comm->clusterSizes);
  free(comm->globalRank2HomoRank);
  free(comm->localRankToRank);
  free(comm->c2cSchedule);
  free(comm->clusterInterRanks);

  // Destroy custom op state before homo comm — vendor DevCommDestroy
  // needs the NCCL comm to still be alive.
  FLAGCXCHECK(flagcxDevCommStateDestroy(comm));

  // Destroy homo comms
  if (comm->tuner) {
    for (const auto &item : comm->homoCommMap) {
      if (item.second != nullptr) {
        FLAGCXCHECK(
            cclAdaptors[flagcxCCLAdaptorDevice]->commDestroy(item.second));
      }
    }
  } else {
    FLAGCXCHECK(
        cclAdaptors[flagcxCCLAdaptorDevice]->commDestroy(comm->homoComm));
  }

  if (!useHomoComm(comm)) {
    // Tear down inter-node signal relay first: drains FIFOs and closes RDMA
    // connections. Must run before flagcxHeteroCommDestroy, which frees
    // proxyState and heteroComm. Proxy threads are stopped inside
    // flagcxCommRelayDestroy via the bootstrap barrier before any teardown.
    FLAGCXCHECK(flagcxCommRelayDestroy(comm));
    // Destroy hetero comm (stops/joins proxy threads, frees proxyState)
    flagcxOneSideStagingDeregister(comm);
    flagcxOneSideSignalDeregister(comm);
    flagcxOneSideDeregister(comm->heteroComm);

    // Destroy hetero comm
    FLAGCXCHECK(flagcxHeteroCommDestroy(comm->heteroComm));
    // Destroy host comm
    if (useHostComm()) {
      FLAGCXCHECK(
          cclAdaptors[flagcxCCLAdaptorHost]->commDestroy(comm->hostComm));
    }
  }

  // Clean up IPC peer pointer table — deferred to here.
  FLAGCXCHECK(flagcxCommCleanupIpcTable(comm));

  // Drain deferred IPC entries (slots released at runtime).
  FLAGCXCHECK(flagcxCommDrainDeferredIpc(comm));

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
    // Free uniqueIdData and commId
    free(comm->uniqueIdData);
    free(comm->commId);
  }

  // Finalize net adaptor plugin (dlclose)
  FLAGCXCHECK(flagcxNetAdaptorPluginFinalize());

  // Finalize device/CCL adaptor plugins (ref-counted)
  flagcxCCLAdaptorPluginFinalize();
  flagcxDeviceAdaptorPluginFinalize();

  free(comm);
  return flagcxSuccess;
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
  if (useHomoComm(comm)) {
    return cclAdaptors[flagcxCCLAdaptorDevice]->commGetAsyncError(
        comm->homoComm, asyncError);
  }
  // TODO: to be implemented.
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

  // Resolve window to MR index and compute byte size
  size_t byteSize = count * getFlagcxDataTypeSize(datatype);
  int dstMrIdx = -1;
  if (peerWin->isSymmetricDefault && peerWin->defaultBase != NULL) {
    dstMrIdx = peerWin->defaultBase->mrIndex;
  }
  if (dstMrIdx < 0)
    return flagcxInvalidArgument;

  // Source is localbuff — find which MR it belongs to by VA range lookup
  flagcxHeteroComm_t hetero = comm->heteroComm;
  int srcMrIdx = -1;
  size_t srcOffset = 0;
  for (int h = 0; h < hetero->oneSideHandleCount; h++) {
    struct flagcxOneSideHandleInfo *info = hetero->oneSideHandles[h];
    if (info == NULL)
      continue;
    uintptr_t base = info->baseVas[hetero->rank];
    if ((uintptr_t)localbuff >= base &&
        (uintptr_t)localbuff < base + info->regionSize) {
      srcMrIdx = h;
      srcOffset = (uintptr_t)localbuff - base;
      break;
    }
  }
  if (srcMrIdx < 0) {
    WARN("flagcxPutSignal: localbuff %p not in any registered MR", localbuff);
    return flagcxInvalidArgument;
  }

  // Signal offset: sender writes to its own slot in receiver's signal buffer,
  // so receiver can identify which peer sent the signal.
  size_t signalOffset = (size_t)hetero->rank * sizeof(uint64_t);

  uint64_t opSeq = 0;
  return flagcxHeteroPutSignalStream(hetero, peer, srcOffset, peerWinOffset,
                                     byteSize, signalOffset, srcMrIdx, dstMrIdx,
                                     1 /*signalValue*/, stream, &opSeq);
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
                                     signalOffset, -1, 0, 1 /*signalValue*/,
                                     stream, &opSeq);
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
