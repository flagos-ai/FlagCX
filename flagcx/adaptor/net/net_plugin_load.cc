/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "adaptor_plugin_load.h"
#include "core.h"
#include "flagcx_net_adaptor.h"
#include "net.h"

#include <dlfcn.h>
#include <mutex>
#include <stdlib.h>
#include <string.h>

static void *netPluginDlHandle = NULL;
static int netPluginRefCount = 0;
static std::mutex netPluginMutex;

extern struct flagcxNetAdaptor *flagcxNetAdaptors[3];

static flagcxResult_t flagcxNetAdaptorPluginLoad() {
  // Already loaded — nothing to do.
  if (netPluginDlHandle != NULL) {
    return flagcxSuccess;
  }

  const char *envValue = getenv("FLAGCX_NET_ADAPTOR_PLUGIN");
  if (envValue == NULL || strcmp(envValue, "none") == 0) {
    return flagcxSuccess;
  }

  netPluginDlHandle = flagcxAdaptorOpenPluginLib(envValue);
  if (netPluginDlHandle == NULL) {
    WARN("ADAPTOR/Plugin: Failed to open net adaptor plugin '%s'", envValue);
    return flagcxSuccess;
  }

  // Future: When v2 is introduced, try dlsym("flagcxNetAdaptorPlugin_v2")
  // first, then fall back to "flagcxNetAdaptorPlugin_v1" and wrap in a v1→v2
  // shim.
  struct flagcxNetAdaptor *plugin = (struct flagcxNetAdaptor *)dlsym(
      netPluginDlHandle, "flagcxNetAdaptorPlugin_v1");
  if (plugin == NULL) {
    WARN("ADAPTOR/Plugin: Failed to find symbol 'flagcxNetAdaptorPlugin_v1' in "
         "'%s': %s",
         envValue, dlerror());
    flagcxAdaptorClosePluginLib(netPluginDlHandle);
    netPluginDlHandle = NULL;
    return flagcxSuccess;
  }

  // Validate function pointers that all built-in net adaptors implement.
  // Fields left NULL in some adaptors (regMrDmaBuf, iput, iget, iputSignal,
  // getDevFromName) are intentionally not checked here.
  if (plugin->name == NULL || plugin->init == NULL || plugin->devices == NULL ||
      plugin->getProperties == NULL || plugin->listen == NULL ||
      plugin->connect == NULL || plugin->accept == NULL ||
      plugin->closeSend == NULL || plugin->closeRecv == NULL ||
      plugin->closeListen == NULL || plugin->regMr == NULL ||
      plugin->deregMr == NULL || plugin->isend == NULL ||
      plugin->irecv == NULL || plugin->iflush == NULL || plugin->test == NULL) {
    WARN("ADAPTOR/Plugin: Net adaptor plugin '%s' is missing required function "
         "pointers",
         envValue);
    flagcxAdaptorClosePluginLib(netPluginDlHandle);
    netPluginDlHandle = NULL;
    return flagcxSuccess;
  }

  flagcxNetAdaptors[0] = plugin;
  INFO(FLAGCX_INIT, "ADAPTOR/Plugin: Loaded net adaptor plugin '%s'",
       plugin->name);
  return flagcxSuccess;
}

static flagcxResult_t flagcxNetAdaptorPluginUnload() {
  flagcxNetAdaptors[0] = nullptr;
  flagcxNetStates[0] = flagcxNetStateInit;
  flagcxAdaptorClosePluginLib(netPluginDlHandle);
  netPluginDlHandle = NULL;
  return flagcxSuccess;
}

flagcxResult_t flagcxNetAdaptorPluginInit() {
  std::lock_guard<std::mutex> lock(netPluginMutex);
  FLAGCXCHECK(flagcxNetAdaptorPluginLoad());
  if (netPluginDlHandle != NULL) {
    netPluginRefCount++;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxNetAdaptorPluginFinalize() {
  std::lock_guard<std::mutex> lock(netPluginMutex);
  if (netPluginRefCount > 0 && --netPluginRefCount == 0) {
    INFO(FLAGCX_NET, "Unloading net adaptor plugin");
    FLAGCXCHECK(flagcxNetAdaptorPluginUnload());
  }
  return flagcxSuccess;
}
