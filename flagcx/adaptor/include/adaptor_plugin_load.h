/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_ADAPTOR_PLUGIN_LOAD_H_
#define FLAGCX_ADAPTOR_PLUGIN_LOAD_H_

#include "flagcx.h"

// ---- Shared utility functions (used by per-type plugin loaders) ----

// Open a plugin library by path (e.g., the value of an env var).
// Calls dlopen on the given path. Returns handle or NULL.
void *flagcxAdaptorOpenPluginLib(const char *path);

// Close a previously opened plugin library.
flagcxResult_t flagcxAdaptorClosePluginLib(void *handle);

// ---- Per-type plugin load/unload (implemented in ccl/, device/, net/) ----

// CCL adaptor plugin loading (ccl/ccl_plugin_load.cc)
// Reads FLAGCX_CCL_ADAPTOR_PLUGIN, overrides
// cclAdaptors[flagcxCCLAdaptorDevice].
flagcxResult_t flagcxCCLAdaptorPluginLoad();
flagcxResult_t flagcxCCLAdaptorPluginUnload();

// Device adaptor plugin loading (device/device_plugin_load.cc)
// Reads FLAGCX_DEVICE_ADAPTOR_PLUGIN, overrides deviceAdaptor.
flagcxResult_t flagcxDeviceAdaptorPluginLoad();
flagcxResult_t flagcxDeviceAdaptorPluginUnload();

// Net adaptor plugin loading (net/net_plugin_load.cc)
// Reads FLAGCX_NET_ADAPTOR_PLUGIN, populates flagcxNetAdaptors[0].
flagcxResult_t flagcxNetAdaptorPluginLoad();
flagcxResult_t flagcxNetAdaptorPluginUnload();

// ---- Per-type plugin init/finalize (wrap Load/Unload with fallback) ----

// CCL adaptor plugin init/finalize
// Init calls Load, with fallback logic on failure.
// Finalize calls Unload, with best-effort cleanup on failure.
flagcxResult_t flagcxCCLAdaptorPluginInit();
flagcxResult_t flagcxCCLAdaptorPluginFinalize();

// Device adaptor plugin init/finalize
flagcxResult_t flagcxDeviceAdaptorPluginInit();
flagcxResult_t flagcxDeviceAdaptorPluginFinalize();

// Net adaptor plugin init/finalize
flagcxResult_t flagcxNetAdaptorPluginInit();
flagcxResult_t flagcxNetAdaptorPluginFinalize();

// Top-level orchestrators removed: each plugin type (device, ccl, net)
// has different lifecycle requirements and will be initialized/finalized
// at the appropriate stage in later phases.

#endif // FLAGCX_ADAPTOR_PLUGIN_LOAD_H_
