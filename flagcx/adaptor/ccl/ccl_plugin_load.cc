/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
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

  // Future: When v2 is introduced, try dlsym("flagcxCCLAdaptorPlugin_v2")
  // first, then fall back to "flagcxCCLAdaptorPlugin_v1" and wrap in a v1→v2
  // shim.
  struct flagcxCCLAdaptor *plugin = (struct flagcxCCLAdaptor *)dlsym(
      cclPluginDlHandle, "flagcxCCLAdaptorPlugin_v1");
  if (plugin == NULL) {
    WARN("ADAPTOR/Plugin: Failed to find symbol 'flagcxCCLAdaptorPlugin_v1' in "
         "'%s': %s",
         envValue, dlerror());
    flagcxAdaptorClosePluginLib(cclPluginDlHandle);
    cclPluginDlHandle = NULL;
    return flagcxSuccess;
  }

  // Validate all 34 function pointers
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
    flagcxAdaptorClosePluginLib(cclPluginDlHandle);
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
  if (cclDefaultDeviceAdaptor != NULL) {
    cclAdaptors[flagcxCCLAdaptorDevice] = cclDefaultDeviceAdaptor;
    cclDefaultDeviceAdaptor = NULL;
  }
  flagcxAdaptorClosePluginLib(cclPluginDlHandle);
  cclPluginDlHandle = NULL;
  return flagcxSuccess;
}

flagcxResult_t flagcxCCLAdaptorPluginInit() {
  std::lock_guard<std::mutex> lock(cclPluginMutex);
  flagcxCCLAdaptorPluginLoad();
  if (cclPluginDlHandle != NULL) {
    cclPluginRefCount++;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxCCLAdaptorPluginFinalize() {
  std::lock_guard<std::mutex> lock(cclPluginMutex);
  if (cclPluginRefCount > 0 && --cclPluginRefCount == 0) {
    INFO(FLAGCX_INIT, "Unloading CCL adaptor plugin");
    flagcxCCLAdaptorPluginUnload();
  }
  return flagcxSuccess;
}
