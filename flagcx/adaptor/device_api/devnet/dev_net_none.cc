/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Default DevNet binding: Proxy resources and device operations are supplied
 * by DefaultBackend. There are no additional GDA registrations to manage.
 ************************************************************************/
#include "device_api/devnet/flagcx_dev_net_adaptor.h"

namespace {
flagcxResult_t createComm(flagcxComm_t, flagcxDevComm_t,
                          const flagcxDevCommRequirements *) {
  return flagcxSuccess;
}
flagcxResult_t commNoop(flagcxDevComm_t) { return flagcxSuccess; }
flagcxResult_t registerWindow(flagcxComm_t, flagcxDevComm_t, flagcxWindow_t) {
  return flagcxSuccess;
}
flagcxResult_t windowNoop(flagcxWindow_t) { return flagcxSuccess; }
flagcxResult_t bindDevMem(flagcxWindow_t, flagcxDevMem_t, size_t, void *) {
  return flagcxSuccess;
}
flagcxResult_t unbindDevMem(flagcxDevMem_t) { return flagcxSuccess; }
} // namespace

const flagcxDevNetAdaptor devNetAdaptor = {
    "None (Proxy)", false,      createComm, commNoop,
    registerWindow, windowNoop, windowNoop, bindDevMem,
    unbindDevMem,   commNoop,   commNoop,   commNoop};
