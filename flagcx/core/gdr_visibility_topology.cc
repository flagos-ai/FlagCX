/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#include "gdr_visibility.h"
#include "topo.h"

namespace {

flagcxGdrGpuNetPath_t classifyGpuNetPath(int pathType) {
  switch (pathType) {
    case PATH_PIX:
    case PATH_PXB:
      return FLAGCX_GDR_GPU_NET_PATH_PCIE_NEAR;
    case PATH_PXN:
    case PATH_PHB:
    case PATH_SYS:
      return FLAGCX_GDR_GPU_NET_PATH_PCIE_FAR;
    default:
      // LOC, CCI, CCB, NET, and future path types do not prove a PCIe route.
      return FLAGCX_GDR_GPU_NET_PATH_UNKNOWN;
  }
}

bool isPciePath(int pathType) {
  return classifyGpuNetPath(pathType) != FLAGCX_GDR_GPU_NET_PATH_UNKNOWN;
}

} // namespace

flagcxResult_t
flagcxClassifyGdrTopology(const struct flagcxTopoServer *topology, int rank,
                          int netDev, flagcxGdrGpuNetPath_t *gpuNetPath,
                          flagcxGdrGpuCpuPath_t *gpuCpuPath) {
  if (gpuNetPath == nullptr || gpuCpuPath == nullptr)
    return flagcxInvalidArgument;
  *gpuNetPath = FLAGCX_GDR_GPU_NET_PATH_UNKNOWN;
  *gpuCpuPath = FLAGCX_GDR_GPU_CPU_PATH_UNKNOWN;
  if (topology == nullptr)
    return flagcxSuccess;

  const struct flagcxTopoNode *apu = nullptr;
  for (int i = 0; i < topology->nodes[APU].count; ++i) {
    if (topology->nodes[APU].nodes[i].apu.rank == rank) {
      apu = &topology->nodes[APU].nodes[i];
      break;
    }
  }
  if (apu == nullptr)
    return flagcxSuccess;

  const struct flagcxTopoPath *netPaths = apu->paths[NET];
  if (netPaths != nullptr) {
    for (int i = 0; i < topology->nodes[NET].count; ++i) {
      if (topology->nodes[NET].nodes[i].net.dev != netDev)
        continue;
      *gpuNetPath = classifyGpuNetPath(netPaths[i].type);
      break;
    }
  }

  // FlagCX topology does not yet have an NVIDIA-specific C2C path type. An
  // x86 CPU is positive evidence for PCIe; ARM is deliberately UNKNOWN
  // because it could be a Grace Hopper C2C control path. This prevents the
  // Hopper exemption from becoming a false coherence claim.
  const struct flagcxTopoPath *cpuPaths = apu->paths[CPU];
  int bestCpu = -1;
  float bestBw = 0.0f;
  int bestType = PATH_DIS;
  if (cpuPaths != nullptr) {
    for (int i = 0; i < topology->nodes[CPU].count; ++i) {
      if (cpuPaths[i].type == PATH_DIS)
        continue;
      if (bestCpu == -1 || cpuPaths[i].bw > bestBw ||
          (cpuPaths[i].bw == bestBw && cpuPaths[i].type < bestType)) {
        bestCpu = i;
        bestBw = cpuPaths[i].bw;
        bestType = cpuPaths[i].type;
      }
    }
  }
  if (bestCpu >= 0 && isPciePath(bestType) &&
      topology->nodes[CPU].nodes[bestCpu].cpu.arch == FLAGCX_TOPO_CPU_ARCH_X86)
    *gpuCpuPath = FLAGCX_GDR_GPU_CPU_PATH_PCIE;
  return flagcxSuccess;
}
