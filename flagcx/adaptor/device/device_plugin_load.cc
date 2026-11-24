/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "adaptor.h"
#include "adaptor_plugin_load.h"
#include "alloc.h"
#include "core.h"
#include "flagcx_device_adaptor.h"
#include "launch_kernel.h"

#include <dlfcn.h>
#include <mutex>
#include <string.h>

static void *devicePluginDlHandle = NULL;
static int devicePluginRefCount = 0;
static std::mutex devicePluginMutex;
static struct flagcxDeviceAdaptor *defaultDeviceAdaptor = NULL;
static struct flagcxDeviceAdaptor *upgradedPluginAdaptor = NULL;
extern struct flagcxDeviceAdaptor *deviceAdaptor;

// Defined in flagcx.cc — rebuilds globalDeviceHandle from current deviceAdaptor
extern void flagcxRebuildGlobalDeviceHandle();

flagcxResult_t flagcxDeviceAdaptorPluginLoad() {
  // Already loaded — nothing to do.
  if (devicePluginDlHandle != NULL) {
    return flagcxSuccess;
  }

  const char *envValue = getenv("FLAGCX_DEVICE_ADAPTOR_PLUGIN");
  if (envValue == NULL || strcmp(envValue, "none") == 0) {
    return flagcxSuccess;
  }

  devicePluginDlHandle = flagcxAdaptorOpenPluginLib(envValue);
  if (devicePluginDlHandle == NULL) {
    WARN("ADAPTOR/Plugin: Failed to open device adaptor plugin '%s'", envValue);
    return flagcxSuccess;
  }

  // Try latest symbol first; fall back to v1 and upgrade.
  struct flagcxDeviceAdaptor *plugin = (struct flagcxDeviceAdaptor *)dlsym(
      devicePluginDlHandle, "flagcxDeviceAdaptorPlugin_v1");
  if (plugin == NULL) {
    WARN("ADAPTOR/Plugin: Failed to find symbol 'flagcxDeviceAdaptorPlugin_v1' "
         "in '%s': %s",
         envValue, dlerror());
    FLAGCXCHECK(flagcxAdaptorClosePluginLib(devicePluginDlHandle));
    devicePluginDlHandle = NULL;
    return flagcxSuccess;
  }
  // Upgrade v1 plugin to latest: copy v1 fields and zero optional fields added
  // beyond v1. Always required since plugins only export versioned symbols and
  // _latest is an internal-only struct.
  if (flagcxCalloc(&upgradedPluginAdaptor, 1) != flagcxSuccess) {
    WARN("ADAPTOR/Plugin: Failed to allocate upgraded adaptor struct");
    FLAGCXCHECK(flagcxAdaptorClosePluginLib(devicePluginDlHandle));
    devicePluginDlHandle = NULL;
    return flagcxSystemError;
  }
  flagcxDeviceAdaptorUpgradeV1((const struct flagcxDeviceAdaptor_v1 *)plugin,
                               upgradedPluginAdaptor);
  plugin = upgradedPluginAdaptor;

  // Validate function pointers that all built-in adaptors implement.
  // Fields that some adaptors leave NULL (hostGetDevicePointer, memHandleInit,
  // memHandleDestroy, hostShareMemAlloc, hostShareMemFree, gdrPtrMmap,
  // gdrPtrMunmap, launchKernel, copyArgsInit, copyArgsFree, launchDeviceFunc,
  // getDeviceProperties, getDevicePciBusId, getDeviceByPciBusId, dmaSupport,
  // getHandleForAddressRange) are intentionally not checked here.
  if (plugin->name[0] == '\0' || plugin->deviceSynchronize == NULL ||
      plugin->deviceMemcpy == NULL || plugin->deviceMemset == NULL ||
      plugin->deviceMalloc == NULL || plugin->deviceFree == NULL ||
      plugin->setDevice == NULL || plugin->getDevice == NULL ||
      plugin->getDeviceCount == NULL || plugin->getVendor == NULL ||
      plugin->gdrMemAlloc == NULL || plugin->gdrMemFree == NULL ||
      plugin->streamCreate == NULL || plugin->streamDestroy == NULL ||
      plugin->streamCopy == NULL || plugin->streamFree == NULL ||
      plugin->streamSynchronize == NULL || plugin->streamQuery == NULL ||
      plugin->streamWaitEvent == NULL || plugin->streamWaitValue64 == NULL ||
      plugin->streamWriteValue64 == NULL || plugin->eventCreate == NULL ||
      plugin->eventDestroy == NULL || plugin->eventRecord == NULL ||
      plugin->eventSynchronize == NULL || plugin->eventQuery == NULL ||
      plugin->eventElapsedTime == NULL || plugin->ipcMemHandleCreate == NULL ||
      plugin->ipcMemHandleGet == NULL || plugin->ipcMemHandleOpen == NULL ||
      plugin->ipcMemHandleClose == NULL || plugin->ipcMemHandleFree == NULL ||
      plugin->launchHostFunc == NULL) {
    WARN("ADAPTOR/Plugin: Device adaptor plugin '%s' is missing required "
         "function pointers",
         envValue);
    free(upgradedPluginAdaptor);
    upgradedPluginAdaptor = NULL;
    FLAGCXCHECK(flagcxAdaptorClosePluginLib(devicePluginDlHandle));
    devicePluginDlHandle = NULL;
    return flagcxSuccess;
  }

  // A built-in adaptor may have been used before this valid plugin was
  // requested. Release its cached signals before switching adaptors.
  flagcxResult_t poolResult = flagcxStreamValueProbeCleanup();
  if (poolResult == flagcxSuccess)
    poolResult = streamValueBufferPool.clear();
  if (poolResult == flagcxSuccess)
    poolResult = deviceSemaphoreBufferPool.clear();
  if (poolResult != flagcxSuccess) {
    free(upgradedPluginAdaptor);
    upgradedPluginAdaptor = NULL;
    FLAGCXCHECK(flagcxAdaptorClosePluginLib(devicePluginDlHandle));
    devicePluginDlHandle = NULL;
    return poolResult;
  }
  flagcxClearStreamValueProbeCache();

  defaultDeviceAdaptor = deviceAdaptor;
  deviceAdaptor = plugin;
  flagcxRebuildGlobalDeviceHandle();
  INFO(FLAGCX_INIT, "ADAPTOR/Plugin: Loaded device adaptor plugin '%s'",
       plugin->name);
  return flagcxSuccess;
}

flagcxResult_t flagcxDeviceAdaptorPluginUnload() {
  flagcxResult_t result = flagcxStreamValueProbeCleanup();
  if (result == flagcxSuccess)
    result = streamValueBufferPool.clear();
  if (result == flagcxSuccess)
    result = deviceSemaphoreBufferPool.clear();
  if (result != flagcxSuccess)
    return result;
  FLAGCXCHECK(flagcxAdaptorClosePluginLib(devicePluginDlHandle));
  flagcxClearStreamValueProbeCache();
  if (defaultDeviceAdaptor != NULL) {
    deviceAdaptor = defaultDeviceAdaptor;
    defaultDeviceAdaptor = NULL;
    flagcxRebuildGlobalDeviceHandle();
  }
  free(upgradedPluginAdaptor);
  upgradedPluginAdaptor = NULL;
  devicePluginDlHandle = NULL;
  return flagcxSuccess;
}

flagcxResult_t flagcxDeviceAdaptorPluginInit(bool *acquired) {
  std::lock_guard<std::mutex> lock(devicePluginMutex);
  if (acquired != NULL)
    *acquired = false;
  flagcxResult_t result = flagcxDeviceAdaptorPluginLoad();
  // A failed load may retain a library whose rollback failed. Give that
  // cleanup reference to the same owner instead of losing it at Init failure.
  if (result == flagcxSuccess || devicePluginDlHandle != NULL) {
    ++devicePluginRefCount;
    if (acquired != NULL)
      *acquired = true;
  }
  return result;
}

flagcxResult_t flagcxDeviceAdaptorPluginFinalize() {
  std::lock_guard<std::mutex> lock(devicePluginMutex);
  if (devicePluginRefCount == 0)
    return flagcxSuccess;
  if (devicePluginRefCount == 1) {
    flagcxResult_t result = flagcxDeviceAdaptorPluginUnload();
    if (result != flagcxSuccess)
      return result;
  }
  --devicePluginRefCount;
  return flagcxSuccess;
}
