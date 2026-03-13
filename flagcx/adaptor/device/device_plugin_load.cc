/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "adaptor_plugin_load.h"

// TODO (Phase 4): Implement Device adaptor plugin loading.
//
// This file should:
// 1. Read env var FLAGCX_DEVICE_ADAPTOR_PLUGIN
// 2. If NULL or "none": return flagcxSuccess (keep default)
// 3. dlopen the path via flagcxAdaptorOpenPluginLib()
// 4. dlsym for "flagcxDeviceAdaptorPlugin_v1" (struct flagcxDeviceAdaptor)
// 5. Validate: name non-NULL, critical function pointers non-NULL
// 6. Backup default: save deviceAdaptor
// 7. Override: deviceAdaptor = loaded plugin
// 8. Log success/failure
//
// NOTE: Must also handle globalDeviceHandle in flagcx/flagcx.cc which
// caches deviceAdaptor function pointers at static init time. Either
// rebuild globalDeviceHandle after plugin loading, or change it to
// dispatch through deviceAdaptor indirectly.
//
// Static state needed:
//   static void *devicePluginDlHandle = NULL;
//   static struct flagcxDeviceAdaptor *defaultDeviceAdaptor = NULL;

static int devicePluginRefCount = 0;

flagcxResult_t flagcxDeviceAdaptorPluginLoad() { return flagcxSuccess; }

flagcxResult_t flagcxDeviceAdaptorPluginUnload() { return flagcxSuccess; }

flagcxResult_t flagcxDeviceAdaptorPluginInit() {
  flagcxResult_t ret = flagcxDeviceAdaptorPluginLoad();
  if (ret != flagcxSuccess) {
    // TODO (Phase 4): fallback to compile-time default
    return ret;
  }
  // TODO (Phase 4): only increment when Load actually opened a library
  // devicePluginRefCount++;
  return flagcxSuccess;
}

flagcxResult_t flagcxDeviceAdaptorPluginFinalize() {
  // TODO (Phase 4): decrement and unload only when refcount reaches zero
  // if (devicePluginRefCount > 0 && --devicePluginRefCount == 0) {
  //   flagcxDeviceAdaptorPluginUnload();
  // }
  return flagcxSuccess;
}
