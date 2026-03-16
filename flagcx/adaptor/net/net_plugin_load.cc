/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "adaptor_plugin_load.h"
#include "core.h"
#include "flagcx_net_adaptor.h"

#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

static void *netPluginDlHandle = NULL;
static int netPluginRefCount = 0;

extern struct flagcxNetAdaptor *flagcxNetAdaptors[3];

flagcxResult_t flagcxNetAdaptorPluginLoad() {
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

  // Validate critical function pointers
  if (plugin->name == NULL || plugin->init == NULL || plugin->devices == NULL ||
      plugin->listen == NULL || plugin->connect == NULL ||
      plugin->accept == NULL || plugin->isend == NULL ||
      plugin->irecv == NULL) {
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

flagcxResult_t flagcxNetAdaptorPluginUnload() {
  flagcxNetAdaptors[0] = nullptr;
  flagcxAdaptorClosePluginLib(netPluginDlHandle);
  netPluginDlHandle = NULL;
  return flagcxSuccess;
}

flagcxResult_t flagcxNetAdaptorPluginInit() {
  flagcxNetAdaptorPluginLoad();
  if (netPluginDlHandle != NULL) {
    netPluginRefCount++;
  }
  return flagcxSuccess;
}

flagcxResult_t flagcxNetAdaptorPluginFinalize() {
  if (netPluginRefCount > 0 && --netPluginRefCount == 0) {
    INFO(FLAGCX_NET, "Unloading net adaptor plugin");
    flagcxNetAdaptorPluginUnload();
  }
  return flagcxSuccess;
}
