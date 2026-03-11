/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_ADAPTOR_PLUGIN_LOAD_H_
#define FLAGCX_ADAPTOR_PLUGIN_LOAD_H_

#include "flagcx.h"

// ---- Shared utility functions (used by per-type plugin loaders) ----

// Try to dlopen a library by name. Returns handle or NULL.
void *flagcxAdaptorTryOpenLib(const char *name);

// Open a plugin library by path (the value of the env var).
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

// ---- Top-level init/finalize (called from FlagCX core init) ----

// Initialize and load all adaptor plugins based on environment variables.
// Called once during FlagCX initialization, AFTER compile-time adaptors are
// set. Delegates to per-type loaders: flagcxCCLAdaptorPluginLoad(), etc.
flagcxResult_t flagcxAdaptorPluginInit();

// Finalize and unload all adaptor plugins.
// Delegates to per-type finalizers.
flagcxResult_t flagcxAdaptorPluginFinalize();

#endif // FLAGCX_ADAPTOR_PLUGIN_LOAD_H_
