/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "gdr_visibility.h"

#include "flagcx_net_adaptor.h"
#include "param.h"

extern "C" {
extern struct flagcxDeviceAdaptor *deviceAdaptor;
}

namespace {

constexpr uint32_t kKnownRequirements =
    FLAGCX_GDR_READ_REQUIRES_FLUSH | FLAGCX_GDR_WRITE_REQUIRES_FLUSH;

FLAGCX_PARAM(GdrReadRequiresFlush, "GDR_READ_REQUIRES_FLUSH", -1);
FLAGCX_PARAM(GdrWriteRequiresFlush, "GDR_WRITE_REQUIRES_FLUSH", -1);

bool isKnownDeviceFamily(flagcxGdrDeviceFamily_t family) {
  return family >= FLAGCX_GDR_DEVICE_UNKNOWN && family <= FLAGCX_GDR_DEVICE_PPU;
}

bool isKnownGpuNetPath(flagcxGdrGpuNetPath_t path) {
  return path >= FLAGCX_GDR_GPU_NET_PATH_UNKNOWN &&
         path <= FLAGCX_GDR_GPU_NET_PATH_PCIE_FAR;
}

bool isKnownGpuCpuPath(flagcxGdrGpuCpuPath_t path) {
  return path >= FLAGCX_GDR_GPU_CPU_PATH_UNKNOWN &&
         path <= FLAGCX_GDR_GPU_CPU_PATH_C2C;
}

void applyOverride(uint32_t bit, int64_t value, uint32_t *requirements,
                   uint32_t *reasons) {
  if (value == 0) {
    *requirements &= ~bit;
    *reasons |= FLAGCX_GDR_VISIBILITY_REASON_ENV_DISABLE;
  } else if (value == 1) {
    *requirements |= bit;
    *reasons |= FLAGCX_GDR_VISIBILITY_REASON_ENV_FORCE;
  }
}

} // namespace

flagcxResult_t flagcxResolveGdrVisibilityPolicy(
    const struct flagcxGdrVisibilityContext *context, int64_t readOverride,
    int64_t writeOverride, struct flagcxGdrVisibilityDecision *decision) {
  if (context == nullptr || decision == nullptr)
    return flagcxInvalidArgument;
  if ((context->defaultRequirements & ~kKnownRequirements) != 0 ||
      (context->providerForceRequirements & ~kKnownRequirements) != 0 ||
      context->useGdr < 0 || context->useGdr > 1 ||
      context->topologyKnown < 0 || context->topologyKnown > 1 ||
      context->peerGpuMayPublish < 0 || context->peerGpuMayPublish > 1 ||
      !isKnownDeviceFamily(context->deviceFamily) ||
      !isKnownGpuNetPath(context->gpuNetPath) ||
      !isKnownGpuCpuPath(context->gpuCpuPath))
    return flagcxInvalidArgument;

  decision->requirements = FLAGCX_GDR_FLUSH_NONE;
  decision->reasons = FLAGCX_GDR_VISIBILITY_REASON_NONE;
  if (!context->useGdr) {
    decision->reasons |= FLAGCX_GDR_VISIBILITY_REASON_NON_GDR;
    return flagcxSuccess;
  }

  uint32_t requirements = context->defaultRequirements;
  uint32_t reasons = requirements == FLAGCX_GDR_FLUSH_NONE
                         ? FLAGCX_GDR_VISIBILITY_REASON_NONE
                         : FLAGCX_GDR_VISIBILITY_REASON_DEVICE_DEFAULT;

  // NCCL's compute-capability exemption applies to the incoming receive
  // (remote WRITE) path only. GET/READ remains unchanged. Missing topology
  // evidence is conservative: it cannot establish a coherent Hopper path.
  if (context->peerGpuMayPublish)
    reasons |= FLAGCX_GDR_VISIBILITY_REASON_PEER_GPU_PUBLISH;

  if (!context->peerGpuMayPublish &&
      context->deviceFamily == FLAGCX_GDR_DEVICE_CUDA &&
      context->deviceArchitecture >= 90) {
    const bool topologyComplete =
        context->topologyKnown &&
        context->gpuNetPath != FLAGCX_GDR_GPU_NET_PATH_UNKNOWN &&
        context->gpuCpuPath != FLAGCX_GDR_GPU_CPU_PATH_UNKNOWN;
    if (!topologyComplete) {
      reasons |= FLAGCX_GDR_VISIBILITY_REASON_TOPOLOGY_UNKNOWN;
    } else {
      requirements &= ~FLAGCX_GDR_WRITE_REQUIRES_FLUSH;
      reasons |= FLAGCX_GDR_VISIBILITY_REASON_CUDA_HOPPER;
      // NCCL's DataDirect exception: the NIC data path uses nearby PCIe while
      // the CPU control/sync path uses C2C, so the two paths need an explicit
      // ordering boundary even on Hopper.
      if (context->gpuNetPath == FLAGCX_GDR_GPU_NET_PATH_PCIE_NEAR &&
          context->gpuCpuPath == FLAGCX_GDR_GPU_CPU_PATH_C2C) {
        requirements |= FLAGCX_GDR_WRITE_REQUIRES_FLUSH;
        reasons |= FLAGCX_GDR_VISIBILITY_REASON_CUDA_DATADIRECT_C2C;
      }
    }
  }

  if (context->providerForceRequirements != FLAGCX_GDR_FLUSH_NONE) {
    requirements |= context->providerForceRequirements;
    reasons |= FLAGCX_GDR_VISIBILITY_REASON_PROVIDER_FORCE;
  }

  // Keep the existing expert override contract. In particular, zero may
  // deliberately clear a provider-forced bit for diagnostics; call sites will
  // log that unsafe choice when the resolver is connected to production paths.
  applyOverride(FLAGCX_GDR_READ_REQUIRES_FLUSH, readOverride, &requirements,
                &reasons);
  applyOverride(FLAGCX_GDR_WRITE_REQUIRES_FLUSH, writeOverride, &requirements,
                &reasons);

  decision->requirements = requirements;
  decision->reasons = reasons;
  return flagcxSuccess;
}

extern "C" uint32_t
flagcxApplyGdrFlushRequirementOverrides(uint32_t defaults, int64_t readOverride,
                                        int64_t writeOverride) {
  const flagcxGdrVisibilityContext context = {
      defaults,
      FLAGCX_GDR_FLUSH_NONE,
      FLAGCX_GDR_DEVICE_UNKNOWN,
      0,
      1,
      0,
      0,
      FLAGCX_GDR_GPU_NET_PATH_UNKNOWN,
      FLAGCX_GDR_GPU_CPU_PATH_UNKNOWN,
      static_cast<uint8_t>(FLAGCX_VMM_MR_ROUTE_NONE),
  };
  flagcxGdrVisibilityDecision decision = {};
  if (flagcxResolveGdrVisibilityPolicy(&context, readOverride, writeOverride,
                                       &decision) != flagcxSuccess)
    return defaults;
  return decision.requirements;
}

extern "C" uint32_t flagcxResolveGdrFlushRequirements(uint32_t defaults) {
  return flagcxApplyGdrFlushRequirementOverrides(
      defaults, flagcxParamGdrReadRequiresFlush(),
      flagcxParamGdrWriteRequiresFlush());
}

flagcxResult_t flagcxResolveGdrVisibilityForConnection(
    const struct flagcxTopoServer *topology, int rank, int deviceArchitecture,
    const struct flagcxNetAdaptor_latest *netAdaptor, int netDev, int useGdr,
    int peerGpuMayPublish, uint8_t registrationRoute,
    struct flagcxGdrVisibilityDecision *decision) {
  if (deviceAdaptor == nullptr || netAdaptor == nullptr || decision == nullptr)
    return flagcxInvalidArgument;

  flagcxGdrVisibilityContext context = {};
  context.defaultRequirements = deviceAdaptor->gdrFlushRequirements;
  context.providerForceRequirements = netAdaptor->gdrFlushForceRequirements;
  context.deviceFamily = deviceAdaptor->gdrDeviceFamily;
  context.deviceArchitecture = deviceArchitecture;
  context.useGdr = useGdr;
  context.peerGpuMayPublish = peerGpuMayPublish;
  context.registrationRoute = registrationRoute;
  const flagcxResult_t topologyResult = flagcxClassifyGdrTopology(
      topology, rank, netDev, &context.gpuNetPath, &context.gpuCpuPath);
  if (topologyResult != flagcxSuccess)
    return topologyResult;
  context.topologyKnown =
      context.gpuNetPath != FLAGCX_GDR_GPU_NET_PATH_UNKNOWN &&
      context.gpuCpuPath != FLAGCX_GDR_GPU_CPU_PATH_UNKNOWN;

  return flagcxResolveGdrVisibilityPolicy(
      &context, flagcxParamGdrReadRequiresFlush(),
      flagcxParamGdrWriteRequiresFlush(), decision);
}
