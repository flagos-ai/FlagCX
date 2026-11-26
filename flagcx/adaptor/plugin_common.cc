/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "adaptor_plugin_load.h"

#include <dlfcn.h>
#include <stdio.h>

#include "core.h"

void *flagcxAdaptorOpenPluginLib(const char *path) {
  if (path == NULL || path[0] == '\0') {
    return NULL;
  }
  void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (handle == NULL) {
    INFO(FLAGCX_INIT, "ADAPTOR/Plugin: dlopen(%s) failed: %s", path, dlerror());
  }
  return handle;
}

flagcxResult_t flagcxAdaptorClosePluginLib(void *handle) {
  if (handle != NULL) {
    dlclose(handle);
  }
  return flagcxSuccess;
}
