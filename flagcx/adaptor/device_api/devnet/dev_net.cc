/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Common DevNet configuration for the build-selected host/device binding.
 ************************************************************************/
#include <cstddef>

#include "adaptor.h"
#include "bootstrap.h"
#include "device_api/devnet/flagcx_dev_net_adaptor.h"
#include "device_api/flagcx_device_internal.h"
#include "flagcx_kernel_internal.h"
#include "global_comm.h"
#include "utils.h"
#include <cstring>
#include <new>
#include <vector>

struct flagcxDevNetConfig {
  int enabled;
  int gda; // Whether every rank has the same compiled GDA candidate.
  char binding[32];
  int wantsNet;
  int forceOneSidedNet;
  int valid;
};

static flagcxResult_t readFlagcxDevNetConfig(flagcxComm_t comm, bool wantsNet,
                                             bool forceOneSidedNet,
                                             flagcxDevNetConfig *config) {
  if (comm == nullptr || config == nullptr || comm->nranks <= 0 ||
      comm->rank < 0 || comm->rank >= comm->nranks)
    return flagcxInvalidArgument;
  flagcxDevNetConfig local{};
  local.enabled = 1;
  local.gda = devNetAdaptor.isGda ? 1 : 0;
  local.wantsNet = wantsNet ? 1 : 0;
  local.forceOneSidedNet = forceOneSidedNet ? 1 : 0;
  std::strncpy(local.binding, devNetAdaptor.name, sizeof(local.binding) - 1);
  local.valid = flagcxReadDevNetConfig(&local.enabled) == flagcxSuccess;
  // A single rank has no peer configuration to compare and needs no
  // bootstrap transport (also used by host-only cleanup tests).
  if (comm->nranks == 1) {
    if (!local.valid)
      return flagcxInvalidArgument;
    *config = local;
    return flagcxSuccess;
  }
  std::vector<flagcxDevNetConfig> all;
  flagcxResult_t allocation = flagcxSuccess;
  try {
    all.resize(comm->nranks);
  } catch (const std::bad_alloc &) {
    allocation = flagcxSystemError;
  }
  flagcxResult_t common = flagcxSuccess;
  FLAGCXCHECK(flagcxConvergeBootstrapStatus(comm, allocation, &common));
  if (common != flagcxSuccess)
    return common;
  all[comm->rank] = local;
  FLAGCXCHECK(
      bootstrapCollAllGather(comm->bootstrap, all.data(), sizeof(local)));
  bool compatible = local.gda != 0;
  for (const flagcxDevNetConfig &peer : all) {
    if (!peer.valid)
      return flagcxInvalidArgument;
    if (peer.enabled != local.enabled || peer.wantsNet != local.wantsNet ||
        peer.forceOneSidedNet != local.forceOneSidedNet)
      return flagcxInvalidUsage;
    compatible =
        compatible && peer.gda != 0 &&
        std::memcmp(peer.binding, local.binding, sizeof(local.binding)) == 0;
  }
  local.gda = compatible ? 1 : 0;
  *config = local;
  return flagcxSuccess;
}

flagcxResult_t flagcxDevNetConfigure(flagcxComm_t comm, bool wantsNet,
                                     bool forceOneSidedNet, bool *attemptGda) {
  if (attemptGda == nullptr)
    return flagcxInvalidArgument;
  flagcxDevNetConfig config{};
  FLAGCXCHECK(
      readFlagcxDevNetConfig(comm, wantsNet, forceOneSidedNet, &config));
  *attemptGda = wantsNet && config.enabled != 0 && config.gda != 0;
  return flagcxSuccess;
}

flagcxResult_t
flagcxDevNetCreateComm(flagcxComm_t comm, flagcxDevComm_t devComm,
                       const flagcxDevCommRequirements *requirements) {
  if (comm == nullptr || devComm == nullptr || requirements == nullptr)
    return flagcxInvalidArgument;
  flagcxResult_t local = devNetAdaptor.createComm(comm, devComm, requirements);
  flagcxResult_t common = flagcxSuccess;
  FLAGCXCHECK(flagcxConvergeBootstrapStatus(comm, local, &common));
  if (common == flagcxSuccess) {
    local =
        devComm->gdaDeviceContext != nullptr && devComm->gdaHostOwner != nullptr
            ? flagcxSuccess
            : flagcxInternalError;
    FLAGCXCHECK(flagcxConvergeBootstrapStatus(comm, local, &common));
    if (common == flagcxSuccess) {
      devComm->netUseGda = 1;
      return flagcxSuccess;
    }
    // A provider must not report success before every resource is published.
    // Roll back locally, then agree before considering any other path.
    local = devNetAdaptor.destroyComm(devComm);
    flagcxResult_t cleanup = flagcxSuccess;
    FLAGCXCHECK(flagcxConvergeBootstrapStatus(comm, local, &cleanup));
    return cleanup == flagcxSuccess ? common : cleanup;
  }
  // createComm retains cleanup-only ownership on a failed release. Never
  // start Proxy while any rank still owns candidate GDA resources.
  local =
      devComm->gdaHostOwner == nullptr && devComm->gdaDeviceContext == nullptr
          ? flagcxSuccess
          : common;
  flagcxResult_t cleanup = flagcxSuccess;
  FLAGCXCHECK(flagcxConvergeBootstrapStatus(comm, local, &cleanup));
  if (cleanup != flagcxSuccess)
    return cleanup;
  if (common != flagcxNotSupported && common != flagcxSystemError &&
      common != flagcxRemoteError && common != flagcxUnhandledDeviceError)
    return common;
  INFO(FLAGCX_INIT, "DevNet GDA candidate %s unavailable (%d); using Proxy",
       devNetAdaptor.name, static_cast<int>(common));
  return flagcxSuccess;
}
