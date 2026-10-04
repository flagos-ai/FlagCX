/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "gdr_visibility.h"
#include "topo.h"

#include <gtest/gtest.h>
#include <initializer_list>
#include <memory>

namespace {

constexpr uint32_t kRead = FLAGCX_GDR_READ_REQUIRES_FLUSH;
constexpr uint32_t kWrite = FLAGCX_GDR_WRITE_REQUIRES_FLUSH;

flagcxGdrVisibilityContext cudaContext(int architecture) {
  flagcxGdrVisibilityContext context = {};
  context.defaultRequirements = kRead | kWrite;
  context.deviceFamily = FLAGCX_GDR_DEVICE_CUDA;
  context.deviceArchitecture = architecture;
  context.useGdr = 1;
  context.topologyKnown = 1;
  context.gpuNetPath = FLAGCX_GDR_GPU_NET_PATH_PCIE_FAR;
  context.gpuCpuPath = FLAGCX_GDR_GPU_CPU_PATH_PCIE;
  return context;
}

flagcxGdrVisibilityDecision resolve(const flagcxGdrVisibilityContext &context,
                                    int64_t readOverride = -1,
                                    int64_t writeOverride = -1) {
  flagcxGdrVisibilityDecision decision = {};
  EXPECT_EQ(flagcxResolveGdrVisibilityPolicy(&context, readOverride,
                                             writeOverride, &decision),
            flagcxSuccess);
  return decision;
}

TEST(GdrVisibilityPolicy, RejectsInvalidArgumentsAndMasks) {
  flagcxGdrVisibilityContext context = cudaContext(80);
  flagcxGdrVisibilityDecision decision = {};
  EXPECT_EQ(flagcxResolveGdrVisibilityPolicy(nullptr, -1, -1, &decision),
            flagcxInvalidArgument);
  EXPECT_EQ(flagcxResolveGdrVisibilityPolicy(&context, -1, -1, nullptr),
            flagcxInvalidArgument);

  context.defaultRequirements = 1u << 31;
  EXPECT_EQ(flagcxResolveGdrVisibilityPolicy(&context, -1, -1, &decision),
            flagcxInvalidArgument);

  context = cudaContext(80);
  context.gpuCpuPath = static_cast<flagcxGdrGpuCpuPath_t>(99);
  EXPECT_EQ(flagcxResolveGdrVisibilityPolicy(&context, -1, -1, &decision),
            flagcxInvalidArgument);
}

TEST(GdrVisibilityPolicy, NonGdrConnectionNeedsNoVisibilityBoundary) {
  flagcxGdrVisibilityContext context = cudaContext(80);
  context.useGdr = 0;
  context.providerForceRequirements = kRead | kWrite;
  const flagcxGdrVisibilityDecision decision = resolve(context, 1, 1);
  EXPECT_EQ(decision.requirements, FLAGCX_GDR_FLUSH_NONE);
  EXPECT_NE(decision.reasons & FLAGCX_GDR_VISIBILITY_REASON_NON_GDR, 0u);
}

TEST(GdrVisibilityPolicy, PreHopperKeepsReadAndWriteRequirements) {
  const flagcxGdrVisibilityDecision decision = resolve(cudaContext(80));
  EXPECT_EQ(decision.requirements, kRead | kWrite);
  EXPECT_NE(decision.reasons & FLAGCX_GDR_VISIBILITY_REASON_DEVICE_DEFAULT, 0u);
  EXPECT_EQ(decision.reasons & FLAGCX_GDR_VISIBILITY_REASON_CUDA_HOPPER, 0u);
}

TEST(GdrVisibilityPolicy, HopperClearsWriteButNeverRead) {
  const flagcxGdrVisibilityDecision decision = resolve(cudaContext(90));
  EXPECT_EQ(decision.requirements, kRead);
  EXPECT_NE(decision.reasons & FLAGCX_GDR_VISIBILITY_REASON_CUDA_HOPPER, 0u);
}

TEST(GdrVisibilityPolicy, PeerGpuSignalPreservesHopperWriteAcquire) {
  flagcxGdrVisibilityContext context = cudaContext(90);
  context.peerGpuMayPublish = 1;
  const flagcxGdrVisibilityDecision decision = resolve(context);
  EXPECT_EQ(decision.requirements, kRead | kWrite);
  EXPECT_NE(decision.reasons & FLAGCX_GDR_VISIBILITY_REASON_PEER_GPU_PUBLISH,
            0u);
  EXPECT_EQ(decision.reasons & FLAGCX_GDR_VISIBILITY_REASON_CUDA_HOPPER, 0u);
}

TEST(GdrVisibilityPolicy, HopperWithoutTopologyStaysConservative) {
  flagcxGdrVisibilityContext context = cudaContext(90);
  context.topologyKnown = 0;
  context.gpuNetPath = FLAGCX_GDR_GPU_NET_PATH_UNKNOWN;
  context.gpuCpuPath = FLAGCX_GDR_GPU_CPU_PATH_UNKNOWN;
  const flagcxGdrVisibilityDecision decision = resolve(context);
  EXPECT_EQ(decision.requirements, kRead | kWrite);
  EXPECT_NE(decision.reasons & FLAGCX_GDR_VISIBILITY_REASON_TOPOLOGY_UNKNOWN,
            0u);
}

TEST(GdrVisibilityPolicy,
     HopperWithPartiallyClassifiedTopologyStaysConservative) {
  flagcxGdrVisibilityContext context = cudaContext(90);
  context.gpuCpuPath = FLAGCX_GDR_GPU_CPU_PATH_UNKNOWN;
  flagcxGdrVisibilityDecision decision = resolve(context);
  EXPECT_EQ(decision.requirements, kRead | kWrite);
  EXPECT_NE(decision.reasons & FLAGCX_GDR_VISIBILITY_REASON_TOPOLOGY_UNKNOWN,
            0u);

  context = cudaContext(90);
  context.gpuNetPath = FLAGCX_GDR_GPU_NET_PATH_UNKNOWN;
  decision = resolve(context);
  EXPECT_EQ(decision.requirements, kRead | kWrite);
  EXPECT_NE(decision.reasons & FLAGCX_GDR_VISIBILITY_REASON_TOPOLOGY_UNKNOWN,
            0u);
}

TEST(GdrVisibilityPolicy, HopperDataDirectC2cRestoresWriteRequirement) {
  flagcxGdrVisibilityContext context = cudaContext(90);
  context.gpuNetPath = FLAGCX_GDR_GPU_NET_PATH_PCIE_NEAR;
  context.gpuCpuPath = FLAGCX_GDR_GPU_CPU_PATH_C2C;
  const flagcxGdrVisibilityDecision decision = resolve(context);
  EXPECT_EQ(decision.requirements, kRead | kWrite);
  EXPECT_NE(decision.reasons & FLAGCX_GDR_VISIBILITY_REASON_CUDA_DATADIRECT_C2C,
            0u);
}

TEST(GdrVisibilityPolicy, CudaCompatibleBackendsDoNotUseCudaArchitecture) {
  for (flagcxGdrDeviceFamily_t family :
       {FLAGCX_GDR_DEVICE_METAX, FLAGCX_GDR_DEVICE_DU, FLAGCX_GDR_DEVICE_PPU,
        FLAGCX_GDR_DEVICE_UNKNOWN}) {
    flagcxGdrVisibilityContext context = cudaContext(90);
    context.deviceFamily = family;
    EXPECT_EQ(resolve(context).requirements, kRead | kWrite);
  }
}

TEST(GdrVisibilityPolicy, ProviderCanForceEitherDirection) {
  flagcxGdrVisibilityContext context = cudaContext(90);
  context.defaultRequirements = FLAGCX_GDR_FLUSH_NONE;
  context.providerForceRequirements = kRead | kWrite;
  const flagcxGdrVisibilityDecision decision = resolve(context);
  EXPECT_EQ(decision.requirements, kRead | kWrite);
  EXPECT_NE(decision.reasons & FLAGCX_GDR_VISIBILITY_REASON_PROVIDER_FORCE, 0u);
}

TEST(GdrVisibilityPolicy, ExpertOverridesRemainDirectional) {
  flagcxGdrVisibilityContext context = cudaContext(80);
  flagcxGdrVisibilityDecision decision = resolve(context, 0, -1);
  EXPECT_EQ(decision.requirements, kWrite);
  EXPECT_NE(decision.reasons & FLAGCX_GDR_VISIBILITY_REASON_ENV_DISABLE, 0u);

  context.defaultRequirements = FLAGCX_GDR_FLUSH_NONE;
  decision = resolve(context, -1, 1);
  EXPECT_EQ(decision.requirements, kWrite);
  EXPECT_NE(decision.reasons & FLAGCX_GDR_VISIBILITY_REASON_ENV_FORCE, 0u);

  // Preserve the old resolver's behavior for invalid tri-state values.
  decision = resolve(context, 2, -2);
  EXPECT_EQ(decision.requirements, FLAGCX_GDR_FLUSH_NONE);
}

TEST(GdrVisibilityPolicy, RegistrationRouteDoesNotChangePolicy) {
  flagcxGdrVisibilityContext context = cudaContext(80);
  uint32_t expected = 0;
  for (uint8_t route : {static_cast<uint8_t>(FLAGCX_VMM_MR_ROUTE_NONE),
                        static_cast<uint8_t>(FLAGCX_VMM_MR_ROUTE_VA),
                        static_cast<uint8_t>(FLAGCX_VMM_MR_ROUTE_DMABUF)}) {
    context.registrationRoute = route;
    const uint32_t requirements = resolve(context).requirements;
    if (expected == 0)
      expected = requirements;
    EXPECT_EQ(requirements, expected);
  }
}

TEST(GdrVisibilityTopology, ClassifiesOnlyProvablePciePaths) {
  auto topology = std::make_unique<flagcxTopoServer>();
  topology->nodes[APU].count = 1;
  topology->nodes[APU].nodes[0].apu.rank = 7;
  topology->nodes[NET].count = 1;
  topology->nodes[NET].nodes[0].net.dev = 3;
  topology->nodes[CPU].count = 1;
  topology->nodes[CPU].nodes[0].cpu.arch = FLAGCX_TOPO_CPU_ARCH_X86;

  flagcxTopoPath netPaths[1] = {};
  netPaths[0].type = PATH_PXB;
  netPaths[0].bw = 32.0f;
  flagcxTopoPath cpuPaths[1] = {};
  cpuPaths[0].type = PATH_PHB;
  cpuPaths[0].bw = 32.0f;
  topology->nodes[APU].nodes[0].paths[NET] = netPaths;
  topology->nodes[APU].nodes[0].paths[CPU] = cpuPaths;

  flagcxGdrGpuNetPath_t netPath = FLAGCX_GDR_GPU_NET_PATH_UNKNOWN;
  flagcxGdrGpuCpuPath_t cpuPath = FLAGCX_GDR_GPU_CPU_PATH_UNKNOWN;
  EXPECT_EQ(flagcxClassifyGdrTopology(topology.get(), 7, 3, &netPath, &cpuPath),
            flagcxSuccess);
  EXPECT_EQ(netPath, FLAGCX_GDR_GPU_NET_PATH_PCIE_NEAR);
  EXPECT_EQ(cpuPath, FLAGCX_GDR_GPU_CPU_PATH_PCIE);

  topology->nodes[CPU].nodes[0].cpu.arch = FLAGCX_TOPO_CPU_ARCH_ARM;
  EXPECT_EQ(flagcxClassifyGdrTopology(topology.get(), 7, 3, &netPath, &cpuPath),
            flagcxSuccess);
  EXPECT_EQ(netPath, FLAGCX_GDR_GPU_NET_PATH_PCIE_NEAR);
  EXPECT_EQ(cpuPath, FLAGCX_GDR_GPU_CPU_PATH_UNKNOWN);

  // Numeric ordering of topology enums is not coherence evidence. In
  // particular, CCI/CCB values sort below PCIe paths but must remain unknown.
  topology->nodes[CPU].nodes[0].cpu.arch = FLAGCX_TOPO_CPU_ARCH_X86;
  netPaths[0].type = PATH_CCI;
  cpuPaths[0].type = PATH_CCB;
  EXPECT_EQ(flagcxClassifyGdrTopology(topology.get(), 7, 3, &netPath, &cpuPath),
            flagcxSuccess);
  EXPECT_EQ(netPath, FLAGCX_GDR_GPU_NET_PATH_UNKNOWN);
  EXPECT_EQ(cpuPath, FLAGCX_GDR_GPU_CPU_PATH_UNKNOWN);
}

TEST(GdrVisibilityTopology, MissingTopologyRemainsUnknown) {
  flagcxGdrGpuNetPath_t netPath = FLAGCX_GDR_GPU_NET_PATH_PCIE_NEAR;
  flagcxGdrGpuCpuPath_t cpuPath = FLAGCX_GDR_GPU_CPU_PATH_PCIE;
  EXPECT_EQ(flagcxClassifyGdrTopology(nullptr, 0, 0, &netPath, &cpuPath),
            flagcxSuccess);
  EXPECT_EQ(netPath, FLAGCX_GDR_GPU_NET_PATH_UNKNOWN);
  EXPECT_EQ(cpuPath, FLAGCX_GDR_GPU_CPU_PATH_UNKNOWN);
  EXPECT_EQ(flagcxClassifyGdrTopology(nullptr, 0, 0, nullptr, &cpuPath),
            flagcxInvalidArgument);
}

} // namespace
