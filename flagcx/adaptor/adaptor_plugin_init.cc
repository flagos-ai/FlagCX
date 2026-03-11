/*************************************************************************
 * Copyright (c) 2025 BAAI. All rights reserved.
 ************************************************************************/

#include "adaptor_plugin_load.h"

#include "core.h"

// Top-level orchestrator for adaptor plugin loading.
// Per-type load/unload functions are implemented in:
//   ccl/ccl_plugin_load.cc
//   device/device_plugin_load.cc
//   net/net_plugin_load.cc

flagcxResult_t flagcxAdaptorPluginInit() {
  FLAGCXCHECK(flagcxCCLAdaptorPluginLoad());
  FLAGCXCHECK(flagcxDeviceAdaptorPluginLoad());
  FLAGCXCHECK(flagcxNetAdaptorPluginLoad());
  return flagcxSuccess;
}

flagcxResult_t flagcxAdaptorPluginFinalize() {
  FLAGCXCHECK(flagcxCCLAdaptorPluginUnload());
  FLAGCXCHECK(flagcxDeviceAdaptorPluginUnload());
  FLAGCXCHECK(flagcxNetAdaptorPluginUnload());
  return flagcxSuccess;
}
