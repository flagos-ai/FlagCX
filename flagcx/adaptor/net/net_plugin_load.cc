/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "adaptor_plugin_load.h"

// TODO (Phase 2): Implement Net adaptor plugin loading.
//
// This file should:
// 1. Read env var FLAGCX_NET_ADAPTOR_PLUGIN
// 2. If NULL or "none": return flagcxSuccess (slot [0] stays nullptr)
// 3. dlopen the path via flagcxAdaptorOpenPluginLib()
// 4. dlsym for "flagcxNetAdaptorPlugin_v1" (struct flagcxNetAdaptor)
// 5. Validate: name non-NULL, critical function pointers non-NULL
//    (init, listen, connect, accept, isend, irecv)
// 6. Populate flagcxNetAdaptors[0] with the loaded plugin
//    (do NOT touch slots [1] and [2] — built-in IBRC/Socket remain)
// 7. The existing flagcxNetInit() priority loop handles selection:
//    it tries [0] first, falls back to [1]/[2] on failure
// 8. Log success/failure
//
// Static state needed:
//   static void *netPluginDlHandle = NULL;

static int netPluginRefCount = 0;

flagcxResult_t flagcxNetAdaptorPluginLoad() { return flagcxSuccess; }

flagcxResult_t flagcxNetAdaptorPluginUnload() { return flagcxSuccess; }

flagcxResult_t flagcxNetAdaptorPluginInit() {
  flagcxResult_t ret = flagcxNetAdaptorPluginLoad();
  if (ret != flagcxSuccess) {
    // TODO (Phase 2): fallback to compile-time default
    return ret;
  }
  // TODO (Phase 2): only increment when Load actually opened a library
  // netPluginRefCount++;
  return flagcxSuccess;
}

flagcxResult_t flagcxNetAdaptorPluginFinalize() {
  // TODO (Phase 2): decrement and unload only when refcount reaches zero
  // if (netPluginRefCount > 0 && --netPluginRefCount == 0) {
  //   flagcxNetAdaptorPluginUnload();
  // }
  return flagcxSuccess;
}
