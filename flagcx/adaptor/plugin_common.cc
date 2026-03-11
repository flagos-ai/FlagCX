/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "adaptor_plugin_load.h"

#include <dlfcn.h>
#include <stdio.h>

#include "core.h"

void *flagcxAdaptorTryOpenLib(const char *name) {
  if (name == NULL || name[0] == '\0') {
    return NULL;
  }
  void *handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
  if (handle == NULL) {
    INFO(FLAGCX_INIT, "ADAPTOR/Plugin: dlopen(%s) failed: %s", name, dlerror());
  }
  return handle;
}

void *flagcxAdaptorOpenPluginLib(const char *path) {
  return flagcxAdaptorTryOpenLib(path);
}

flagcxResult_t flagcxAdaptorClosePluginLib(void *handle) {
  if (handle != NULL) {
    dlclose(handle);
  }
  return flagcxSuccess;
}
