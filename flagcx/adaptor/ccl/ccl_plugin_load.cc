/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "adaptor.h"
#include "adaptor_plugin_load.h"
#include "core.h"
#include "flagcx_ccl_adaptor.h"

#include <dlfcn.h>
#include <mutex>
#include <stdlib.h>
#include <string.h>

static void *cclPluginDlHandle = NULL;
static int cclPluginRefCount = 0;
static std::mutex cclPluginMutex;
static struct flagcxCCLAdaptor *cclDefaultDeviceAdaptor = NULL;
// Heap-allocated latest struct used when upgrading a v1 plugin.
static struct flagcxCCLAdaptor *cclUpgradedPlugin = NULL;
extern struct flagcxCCLAdaptor *cclAdaptors[];

flagcxResult_t flagcxCCLAdaptorPluginLoad() {
  // Already loaded — nothing to do.
  if (cclPluginDlHandle != NULL) {
    return flagcxSuccess;
  }

  const char *envValue = getenv("FLAGCX_CCL_ADAPTOR_PLUGIN");
  if (envValue == NULL || strcmp(envValue, "none") == 0) {
    return flagcxSuccess;
  }

  cclPluginDlHandle = flagcxAdaptorOpenPluginLib(envValue);
  if (cclPluginDlHandle == NULL) {
    WARN("ADAPTOR/Plugin: Failed to open CCL adaptor plugin '%s'", envValue);
    return flagcxSuccess;
  }

  // Try the highest known version first, then fall back to older versions.
  // Currently only v1 exists. When v2 is added, try v2 first, then v1.
  struct flagcxCCLAdaptor_v1 *v1 = (struct flagcxCCLAdaptor_v1 *)dlsym(
      cclPluginDlHandle, "flagcxCCLAdaptorPlugin_v1");
  if (v1 == NULL) {
    WARN("ADAPTOR/Plugin: Failed to find symbol "
         "'flagcxCCLAdaptorPlugin_v1' in '%s': %s",
         envValue, dlerror());
    FLAGCXCHECK(flagcxAdaptorClosePluginLib(cclPluginDlHandle));
    cclPluginDlHandle = NULL;
    return flagcxSuccess;
  }
  // Upgrade v1 to latest — all fields added after v1 are zero-initialized.
  cclUpgradedPlugin =
      (struct flagcxCCLAdaptor *)malloc(sizeof(struct flagcxCCLAdaptor));
  if (cclUpgradedPlugin == NULL) {
    FLAGCXCHECK(flagcxAdaptorClosePluginLib(cclPluginDlHandle));
    cclPluginDlHandle = NULL;
    return flagcxSystemError;
  }
  flagcxCCLAdaptorUpgradeV1(v1, cclUpgradedPlugin);
  struct flagcxCCLAdaptor *plugin = cclUpgradedPlugin;

  // Validate required adaptor metadata and function pointers.
  if (plugin->name == NULL || plugin->getVersion == NULL ||
      plugin->getUniqueId == NULL || plugin->getErrorString == NULL ||
      plugin->getLastError == NULL || plugin->getStagedBuffer == NULL ||
      plugin->commInitRank == NULL || plugin->commFinalize == NULL ||
      plugin->commDestroy == NULL || plugin->commAbort == NULL ||
      plugin->commResume == NULL || plugin->commSuspend == NULL ||
      plugin->commCount == NULL || plugin->commGetDeviceNumber == NULL ||
      plugin->commUserRank == NULL || plugin->commGetAsyncError == NULL ||
      plugin->memAlloc == NULL || plugin->memFree == NULL ||
      plugin->commRegister == NULL || plugin->commDeregister == NULL ||
      plugin->commWindowRegister == NULL ||
      plugin->commWindowDeregister == NULL || plugin->reduce == NULL ||
      plugin->gather == NULL || plugin->scatter == NULL ||
      plugin->broadcast == NULL || plugin->allReduce == NULL ||
      plugin->reduceScatter == NULL || plugin->allGather == NULL ||
      plugin->alltoAll == NULL || plugin->alltoAllv == NULL ||
      plugin->send == NULL || plugin->recv == NULL ||
      plugin->groupStart == NULL || plugin->groupEnd == NULL) {
    WARN("ADAPTOR/Plugin: CCL adaptor plugin '%s' is missing required function "
         "pointers",
         envValue);
    free(cclUpgradedPlugin);
    cclUpgradedPlugin = NULL;
    FLAGCXCHECK(flagcxAdaptorClosePluginLib(cclPluginDlHandle));
    cclPluginDlHandle = NULL;
    return flagcxSuccess;
  }

  cclDefaultDeviceAdaptor = cclAdaptors[flagcxCCLAdaptorDevice];
  cclAdaptors[flagcxCCLAdaptorDevice] = plugin;
  INFO(FLAGCX_INIT, "ADAPTOR/Plugin: Loaded CCL adaptor plugin '%s'",
       plugin->name);
  return flagcxSuccess;
}

flagcxResult_t flagcxCCLAdaptorPluginUnload() {
  FLAGCXCHECK(flagcxAdaptorClosePluginLib(cclPluginDlHandle));
  if (cclDefaultDeviceAdaptor != NULL) {
    cclAdaptors[flagcxCCLAdaptorDevice] = cclDefaultDeviceAdaptor;
    cclDefaultDeviceAdaptor = NULL;
  }
  free(cclUpgradedPlugin);
  cclUpgradedPlugin = NULL;
  cclPluginDlHandle = NULL;
  return flagcxSuccess;
}

flagcxResult_t flagcxCCLAdaptorPluginInit(bool *acquired) {
  std::lock_guard<std::mutex> lock(cclPluginMutex);
  if (acquired != NULL)
    *acquired = false;
  flagcxResult_t result = flagcxCCLAdaptorPluginLoad();
  // A failed load may retain a library whose rollback failed. Give that
  // cleanup reference to the same owner instead of losing it at Init failure.
  if (result == flagcxSuccess || cclPluginDlHandle != NULL) {
    ++cclPluginRefCount;
    if (acquired != NULL)
      *acquired = true;
  }
  return result;
}

flagcxResult_t flagcxCCLAdaptorPluginFinalize() {
  std::lock_guard<std::mutex> lock(cclPluginMutex);
  if (cclPluginRefCount == 0)
    return flagcxSuccess;
  if (cclPluginRefCount == 1) {
    FLAGCXCHECK(flagcxCCLAdaptorPluginUnload());
  }
  --cclPluginRefCount;
  return flagcxSuccess;
}
