/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "p2p_pointer.h"

#include "debug.h"
#include "flagcx_device_adaptor.h"

#include <atomic>
#include <string.h>

extern struct flagcxDeviceAdaptor *deviceAdaptor;

flagcxResult_t flagcxP2pDetectPointerType(void *ptr, size_t size, int *ptrType,
                                          char *ipcHandleBuf,
                                          uint32_t *ipcHandleSize,
                                          uintptr_t *ipcAllocationBase) {
  if (ptr == NULL || size == 0 || ptrType == NULL)
    return flagcxInvalidArgument;

  if (ipcHandleBuf)
    memset(ipcHandleBuf, 0, FLAGCX_P2P_IPC_HANDLE_BYTES);
  if (ipcHandleSize)
    *ipcHandleSize = 0;
  if (ipcAllocationBase)
    *ipcAllocationBase = 0;

  // Pointer classification and IPC export answer different questions. In
  // particular, DU can export an IPC handle for mapped host memory. Latest
  // adaptors must classify through their runtime; IPC is only optional sharing
  // metadata after a GPU result. Loader-upgraded v1 plugins and explicitly
  // marked built-ins retain the old IPC inference until they gain a runtime
  // pointer query.
  const bool legacyV1 =
      deviceAdaptor != NULL && (deviceAdaptor->internalFlags &
                                FLAGCX_DEVICE_ADAPTOR_INTERNAL_LEGACY_V1) != 0;
  const bool transitionalInference =
      deviceAdaptor != NULL &&
      (deviceAdaptor->internalFlags &
       FLAGCX_DEVICE_ADAPTOR_INTERNAL_IPC_POINTER_INFERENCE) != 0;
  bool typeKnown = false;
  if (deviceAdaptor != NULL && deviceAdaptor->getPointerType != NULL) {
    const flagcxResult_t typeResult =
        deviceAdaptor->getPointerType(ptr, ptrType);
    if (typeResult == flagcxSuccess) {
      if (*ptrType != FLAGCX_PTR_HOST && *ptrType != FLAGCX_PTR_CUDA)
        return flagcxInternalError;
      typeKnown = true;
    } else if (typeResult != flagcxNotSupported) {
      return typeResult;
    }
  }

  if (!typeKnown && !legacyV1 && !transitionalInference)
    return flagcxNotSupported;

  if (!typeKnown) {
    static std::atomic<bool> warnedLegacyInference{false};
    if (!warnedLegacyInference.exchange(true, std::memory_order_relaxed)) {
      WARN("P2P Reg: device adaptor '%s' has no authoritative pointer query; "
           "temporarily inferring GPU memory from IPC export",
           deviceAdaptor != NULL ? deviceAdaptor->name : "unknown");
    }
  }

  if (typeKnown && *ptrType == FLAGCX_PTR_HOST)
    return flagcxSuccess;

  if (deviceAdaptor == NULL || deviceAdaptor->ipcMemHandleCreate == NULL ||
      deviceAdaptor->ipcMemHandleGet == NULL ||
      deviceAdaptor->ipcMemHandleFree == NULL) {
    if (typeKnown)
      return flagcxSuccess;
    return flagcxNotSupported;
  }

  void *exportBase = ptr;
  if (deviceAdaptor->getAddressRange != NULL) {
    void *allocationBase = NULL;
    size_t allocationSize = 0;
    const flagcxResult_t rangeResult =
        deviceAdaptor->getAddressRange(ptr, &allocationBase, &allocationSize);
    if (rangeResult == flagcxSuccess) {
      const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
      const uintptr_t base = reinterpret_cast<uintptr_t>(allocationBase);
      if (allocationBase == NULL || address < base ||
          address - base > allocationSize ||
          size > allocationSize - (address - base)) {
        WARN("P2P Reg: IPC registration exceeds allocation range");
        if (!typeKnown)
          *ptrType = FLAGCX_PTR_HOST;
        return flagcxSuccess;
      }
      exportBase = allocationBase;
    } else if (rangeResult != flagcxNotSupported) {
      WARN("P2P Reg: IPC allocation range query failed: %d", rangeResult);
      if (!typeKnown)
        *ptrType = FLAGCX_PTR_HOST;
      return flagcxSuccess;
    }
  }

  flagcxIpcMemHandle_t handle = NULL;
  size_t handleSize = 0;
  if (deviceAdaptor->ipcMemHandleCreate(&handle, &handleSize) !=
      flagcxSuccess) {
    *ptrType = typeKnown ? *ptrType : FLAGCX_PTR_HOST;
    return flagcxSuccess;
  }

  const flagcxResult_t getRes =
      deviceAdaptor->ipcMemHandleGet(handle, exportBase);
  if (getRes == flagcxSuccess) {
    if (!typeKnown)
      *ptrType = FLAGCX_PTR_CUDA;
    if (handleSize <= FLAGCX_P2P_IPC_HANDLE_BYTES) {
      if (ipcHandleBuf)
        memcpy(ipcHandleBuf, handle, handleSize);
      if (ipcHandleSize)
        *ipcHandleSize = (uint32_t)handleSize;
      if (ipcAllocationBase)
        *ipcAllocationBase = reinterpret_cast<uintptr_t>(exportBase);
    }
  } else {
    if (deviceAdaptor->getLastError)
      deviceAdaptor->getLastError();
    if (!typeKnown)
      *ptrType = FLAGCX_PTR_HOST;
  }
  deviceAdaptor->ipcMemHandleFree(handle);
  return flagcxSuccess;
}
