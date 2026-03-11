/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "adaptor_plugin_load.h"

// TODO (Phase 3): Implement CCL adaptor plugin loading.
//
// This file should:
// 1. Read env var FLAGCX_CCL_ADAPTOR_PLUGIN
// 2. If NULL or "none": return flagcxSuccess (keep default)
// 3. dlopen the path via flagcxAdaptorOpenPluginLib()
// 4. dlsym for "flagcxCCLAdaptorPlugin_v1" (struct flagcxCCLAdaptor)
// 5. Validate: name non-NULL, critical function pointers non-NULL
// 6. Backup default: save cclAdaptors[flagcxCCLAdaptorDevice]
// 7. Override: cclAdaptors[flagcxCCLAdaptorDevice] = loaded plugin
// 8. Log success/failure
//
// Static state needed:
//   static void *cclPluginDlHandle = NULL;
//   static struct flagcxCCLAdaptor *defaultCCLAdaptorDevice = NULL;

flagcxResult_t flagcxCCLAdaptorPluginLoad() { return flagcxSuccess; }

flagcxResult_t flagcxCCLAdaptorPluginUnload() { return flagcxSuccess; }
