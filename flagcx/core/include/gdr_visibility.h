/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/

#ifndef FLAGCX_GDR_VISIBILITY_H_
#define FLAGCX_GDR_VISIBILITY_H_

#include "flagcx_device_adaptor.h"

#include <stdint.h>

// Semantic path classes used by visibility policy. They deliberately do not
// reuse the topology path ordering: a generic high-speed interconnect is not
// proof of NVIDIA C2C ordering semantics.
typedef enum {
  FLAGCX_GDR_GPU_NET_PATH_UNKNOWN = 0,
  FLAGCX_GDR_GPU_NET_PATH_PCIE_NEAR = 1, // NCCL PATH_PXB or closer
  FLAGCX_GDR_GPU_NET_PATH_PCIE_FAR = 2,
} flagcxGdrGpuNetPath_t;

typedef enum {
  FLAGCX_GDR_GPU_CPU_PATH_UNKNOWN = 0,
  FLAGCX_GDR_GPU_CPU_PATH_PCIE = 1,
  FLAGCX_GDR_GPU_CPU_PATH_C2C = 2,
} flagcxGdrGpuCpuPath_t;

typedef enum {
  FLAGCX_GDR_VISIBILITY_REASON_NONE = 0,
  FLAGCX_GDR_VISIBILITY_REASON_DEVICE_DEFAULT = 1u << 0,
  FLAGCX_GDR_VISIBILITY_REASON_NON_GDR = 1u << 1,
  FLAGCX_GDR_VISIBILITY_REASON_TOPOLOGY_UNKNOWN = 1u << 2,
  FLAGCX_GDR_VISIBILITY_REASON_CUDA_HOPPER = 1u << 3,
  FLAGCX_GDR_VISIBILITY_REASON_CUDA_DATADIRECT_C2C = 1u << 4,
  FLAGCX_GDR_VISIBILITY_REASON_PROVIDER_FORCE = 1u << 5,
  FLAGCX_GDR_VISIBILITY_REASON_ENV_FORCE = 1u << 6,
  FLAGCX_GDR_VISIBILITY_REASON_ENV_DISABLE = 1u << 7,
  FLAGCX_GDR_VISIBILITY_REASON_PEER_GPU_PUBLISH = 1u << 8,
} flagcxGdrVisibilityReason_t;

// All fields describe one concrete device/NIC connection. registrationRoute
// is retained for diagnostics and conformance-test reporting only; the
// resolver must not derive cache-coherence semantics from an MR API choice.
struct flagcxGdrVisibilityContext {
  uint32_t defaultRequirements;
  uint32_t providerForceRequirements;
  flagcxGdrDeviceFamily_t deviceFamily;
  int deviceArchitecture; // CUDA compute capability (e.g. 80, 90).
  int useGdr;
  int topologyKnown;
  // A signal on this path may be published by a peer GPU through IPC/D2D.
  // NIC-specific coherence exemptions must not remove that peer-GPU acquire.
  int peerGpuMayPublish;
  flagcxGdrGpuNetPath_t gpuNetPath;
  flagcxGdrGpuCpuPath_t gpuCpuPath;
  uint8_t registrationRoute;
};

struct flagcxGdrVisibilityDecision {
  uint32_t requirements;
  uint32_t reasons;
};

struct flagcxNetAdaptor_latest;
struct flagcxTopoServer;

// Resolve policy without reading process-global environment state so topology
// and override behavior can be exhaustively unit tested. Override values match
// FLAGCX_GDR_{READ,WRITE}_REQUIRES_FLUSH: -1 keeps automatic policy, 0 clears
// it as an expert override, and 1 forces it. Other values preserve policy for
// backward compatibility with the existing resolver.
flagcxResult_t flagcxResolveGdrVisibilityPolicy(
    const struct flagcxGdrVisibilityContext *context, int64_t readOverride,
    int64_t writeOverride, struct flagcxGdrVisibilityDecision *decision);

// Convert FlagCX's generic topology into only the semantic facts that policy
// can prove. Unsupported path kinds are returned as UNKNOWN, never guessed.
flagcxResult_t
flagcxClassifyGdrTopology(const struct flagcxTopoServer *topology, int rank,
                          int netDev, flagcxGdrGpuNetPath_t *gpuNetPath,
                          flagcxGdrGpuCpuPath_t *gpuCpuPath);

// Build policy input from one live device/NIC connection. Unknown or partially
// modelled topology is fail-closed and retains the device default. This helper
// intentionally does not serve the standalone IB_P2P engine, whose GPU READ
// path remains unsupported until it has a real visibility stage.
flagcxResult_t flagcxResolveGdrVisibilityForConnection(
    const struct flagcxTopoServer *topology, int rank, int deviceArchitecture,
    const struct flagcxNetAdaptor_latest *netAdaptor, int netDev, int useGdr,
    int peerGpuMayPublish, uint8_t registrationRoute,
    struct flagcxGdrVisibilityDecision *decision);

#endif // FLAGCX_GDR_VISIBILITY_H_
