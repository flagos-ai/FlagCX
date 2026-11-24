/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "adaptor_plugin_load.h"
#include "alloc.h"
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
static struct flagcxNetAdaptor *upgradedNetPluginAdaptor = NULL;

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

  struct flagcxNetAdaptor_v1 *pluginV1 = (struct flagcxNetAdaptor_v1 *)dlsym(
      netPluginDlHandle, "flagcxNetAdaptorPlugin_v1");
  if (pluginV1 == NULL) {
    WARN("ADAPTOR/Plugin: Failed to find symbol 'flagcxNetAdaptorPlugin_v1' in "
         "'%s': %s",
         envValue, dlerror());
    flagcxResult_t closeResult = flagcxAdaptorClosePluginLib(netPluginDlHandle);
    if (closeResult != flagcxSuccess)
      return closeResult;
    netPluginDlHandle = NULL;
    return flagcxSuccess;
  }

  if (flagcxCalloc(&upgradedNetPluginAdaptor, 1) != flagcxSuccess) {
    WARN("ADAPTOR/Plugin: Failed to allocate upgraded net adaptor struct");
    flagcxResult_t closeResult = flagcxAdaptorClosePluginLib(netPluginDlHandle);
    if (closeResult != flagcxSuccess)
      return closeResult;
    netPluginDlHandle = NULL;
    return flagcxSystemError;
  }
  flagcxNetAdaptorUpgrade(pluginV1, upgradedNetPluginAdaptor);
  struct flagcxNetAdaptor *plugin = upgradedNetPluginAdaptor;

  // Validate function pointers that all built-in net adaptors implement.
  // Fields left NULL in some adaptors (regMrDmaBuf, iput, iget, iputSignal,
  // getDevFromName, batch helpers, and getMrInfo) are intentionally not
  // checked here.
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
    free(upgradedNetPluginAdaptor);
    upgradedNetPluginAdaptor = NULL;
    flagcxResult_t closeResult = flagcxAdaptorClosePluginLib(netPluginDlHandle);
    if (closeResult != flagcxSuccess)
      return closeResult;
    netPluginDlHandle = NULL;
    return flagcxSuccess;
  }

  flagcxNetAdaptors[0] = plugin;
  INFO(FLAGCX_INIT, "ADAPTOR/Plugin: Loaded net adaptor plugin '%s'",
       plugin->name);
  return flagcxSuccess;
}

static flagcxResult_t flagcxNetAdaptorPluginUnload() {
  flagcxResult_t result = flagcxAdaptorClosePluginLib(netPluginDlHandle);
  if (result != flagcxSuccess)
    return result;
  flagcxNetAdaptors[0] = NULL;
  flagcxNetStates[0] = flagcxNetStateInit;
  free(upgradedNetPluginAdaptor);
  upgradedNetPluginAdaptor = NULL;
  netPluginDlHandle = NULL;
  return flagcxSuccess;
}

flagcxResult_t flagcxNetAdaptorPluginInit(bool *acquired) {
  std::lock_guard<std::mutex> lock(netPluginMutex);
  if (acquired != NULL)
    *acquired = false;
  flagcxResult_t result = flagcxNetAdaptorPluginLoad();
  // A failed load may retain a library whose rollback failed. Give that
  // cleanup reference to the same owner instead of losing it at Init failure.
  if (result == flagcxSuccess || netPluginDlHandle != NULL) {
    ++netPluginRefCount;
    if (acquired != NULL)
      *acquired = true;
  }
  return result;
}

flagcxResult_t flagcxNetAdaptorPluginFinalize() {
  std::lock_guard<std::mutex> lock(netPluginMutex);
  if (netPluginRefCount == 0)
    return flagcxSuccess;
  if (netPluginRefCount == 1) {
    flagcxResult_t result = flagcxNetAdaptorPluginUnload();
    if (result != flagcxSuccess)
      return result;
  }
  --netPluginRefCount;
  return flagcxSuccess;
}
