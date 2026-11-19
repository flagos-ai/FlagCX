/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * XPU NetIO host ownership for the build-selected GPU platform.
 ************************************************************************/

#include "device_api/devnet/xpu_netio_gda_host.h"

#include "adaptor.h"
#include "bootstrap.h"
#include "device_api/devnet/flagcx_dev_net_adaptor.h"
#include "device_api/devnet/xpu_netio_gda_runtime.h"
#include "device_api/devnet/xpu_netio_gda_view.h"
#include "device_api/flagcx_device.h"
#include "flagcx_kernel_internal.h"
#include "sym_heap.h"
#include "utils.h"

#include <xpu_netio/provider.h>
#include <xpu_netio/runtime.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

namespace XpuNetioGda {
namespace {

static constexpr unsigned kProviderId = 1;
static constexpr uint64_t kTimeoutNs = UINT64_C(30000000000);
static constexpr unsigned kAllToAllCapabilities =
    DefaultNetGdaPut | DefaultNetGdaPutValue | DefaultNetGdaFlush |
    DefaultNetGdaBarrier;

flagcxResult_t result(xpu_netio_status_t status) {
  switch (status) {
    case XPU_NETIO_SUCCESS:
      return flagcxSuccess;
    case XPU_NETIO_INVALID_ARGUMENT:
      return flagcxInvalidArgument;
    case XPU_NETIO_UNSUPPORTED:
    case XPU_NETIO_ABI_MISMATCH:
      return flagcxNotSupported;
    case XPU_NETIO_OUT_OF_MEMORY:
      return flagcxSystemError;
    default:
      return flagcxRemoteError;
  }
}

flagcxResult_t stage(flagcxComm_t comm, flagcxResult_t local) {
  flagcxResult_t common = flagcxSuccess;
  flagcxResult_t transport =
      flagcxConvergeBootstrapStatus(comm, local, &common);
  return transport == flagcxSuccess ? common : transport;
}

template <typename T>
flagcxResult_t gather(flagcxComm_t comm, const T &local, std::vector<T> &all) {
  if (comm->nranks <= 0 || sizeof(T) > static_cast<size_t>(INT_MAX))
    return flagcxInvalidArgument;
  try {
    all.resize(comm->nranks);
  } catch (const std::bad_alloc &) {
    return stage(comm, flagcxSystemError);
  }
  FLAGCXCHECK(stage(comm, flagcxSuccess));
  all[comm->rank] = local;
  return bootstrapCollAllGather(comm->bootstrap, all.data(), sizeof(T));
}

struct CommOwner;

struct WindowOwner {
  CommOwner *comm;
  xpu_netio_mr_t *mr = nullptr;
  std::vector<xpu_netio_remote_mr_t *> remote;
  xpu_netio_local_region_t local{};
  xpu_netio_remote_region_t *deviceRegions = nullptr;
  flagcxXpuNetioDeviceWindow *deviceView = nullptr;
  size_t bytes = 0;
  WindowOwner *orphanNext = nullptr;
};

struct CommOwner {
  flagcxComm_t comm = nullptr;
  const xpu_netio_provider_ops_t *ops = nullptr;
  xpu_netio_device_t *device = nullptr;
  xpu_netio_context_t *context = nullptr;
  std::vector<xpu_netio_qp_t *> qps;
  xpu_netio_device_handle_t published{};
  bool publishAttempted = false;
  WindowOwner *signals = nullptr;
  WindowOwner *orphanWindows = nullptr;
  flagcxXpuNetioDeviceContext *deviceView = nullptr;
  uint32_t *errorFlag = nullptr;
  unsigned capabilities = 0;
  unsigned liveWindows = 0;
  xpu_netio_caps_t caps{};
};

flagcxResult_t releaseWindow(WindowOwner *owner) {
  if (owner == nullptr)
    return flagcxSuccess;
  if (owner->deviceView != nullptr) {
    FLAGCXCHECK(
        deviceAdaptor->deviceFree(owner->deviceView, flagcxMemDevice, nullptr));
    owner->deviceView = nullptr;
  }
  if (owner->deviceRegions != nullptr) {
    FLAGCXCHECK(deviceAdaptor->deviceFree(owner->deviceRegions, flagcxMemDevice,
                                          nullptr));
    owner->deviceRegions = nullptr;
  }
  for (size_t i = 0; i < owner->remote.size(); ++i) {
    if (owner->remote[i] != nullptr)
      FLAGCXCHECK(
          result(xpu_netio_destroy_remote_memory_retryable(&owner->remote[i])));
  }
  if (owner->mr != nullptr)
    FLAGCXCHECK(result(xpu_netio_deregister_memory_retryable(&owner->mr)));
  delete owner;
  return flagcxSuccess;
}

flagcxResult_t releaseComm(CommOwner *owner) {
  if (owner == nullptr)
    return flagcxSuccess;
  if (owner->liveWindows != 0)
    return flagcxInvalidUsage;
  while (owner->orphanWindows != nullptr) {
    WindowOwner *orphan = owner->orphanWindows;
    owner->orphanWindows = orphan->orphanNext;
    flagcxResult_t cleanup = releaseWindow(orphan);
    if (cleanup != flagcxSuccess) {
      orphan->orphanNext = owner->orphanWindows;
      owner->orphanWindows = orphan;
      return cleanup;
    }
  }
  if (owner->signals != nullptr) {
    FLAGCXCHECK(releaseWindow(owner->signals));
    owner->signals = nullptr;
  }
  if (owner->deviceView != nullptr) {
    FLAGCXCHECK(
        deviceAdaptor->deviceFree(owner->deviceView, flagcxMemDevice, nullptr));
    owner->deviceView = nullptr;
  }
  if (owner->errorFlag != nullptr) {
    FLAGCXCHECK(
        deviceAdaptor->deviceFree(owner->errorFlag, flagcxMemDevice, nullptr));
    owner->errorFlag = nullptr;
  }
  if (owner->publishAttempted && owner->context != nullptr) {
    FLAGCXCHECK(result(
        xpu_netio_unpublish_qps_retryable(owner->context, &owner->published)));
    owner->publishAttempted = false;
  }
  for (size_t i = 0; i < owner->qps.size(); ++i) {
    if (owner->qps[i] != nullptr)
      FLAGCXCHECK(result(xpu_netio_destroy_qp_retryable(&owner->qps[i])));
  }
  if (owner->context != nullptr)
    FLAGCXCHECK(result(xpu_netio_destroy_context_retryable(&owner->context)));
  if (owner->device != nullptr)
    FLAGCXCHECK(result(xpu_netio_close_device_retryable(&owner->device)));
  delete owner;
  return flagcxSuccess;
}

flagcxResult_t registerRegion(flagcxComm_t comm, CommOwner *owner,
                              void *address, size_t bytes,
                              WindowOwner **output) {
  *output = nullptr;
  WindowOwner *window = new (std::nothrow) WindowOwner;
  flagcxResult_t allocation =
      stage(comm, window ? flagcxSuccess : flagcxSystemError);
  if (allocation != flagcxSuccess) {
    delete window;
    return allocation;
  }
  window->comm = owner;
  window->bytes = bytes;
  flagcxResult_t local = flagcxSuccess;
  try {
    window->remote.resize(comm->nranks, nullptr);
  } catch (const std::bad_alloc &) {
    local = flagcxSystemError;
  }
  flagcxResult_t common = stage(comm, local);
  if (common != flagcxSuccess) {
    delete window;
    return common;
  }

  xpu_netio_mr_attr_t attr{};
  xpu_netio_mr_attr_init(&attr);
  attr.address = address;
  attr.length = bytes;
  const XpuNetioGdaRuntime runtime = flagcxXpuNetioGdaRuntime();
  attr.memory_type = runtime.memoryType;
  if (runtime.memoryType == XPU_NETIO_MEMORY_TYPE_XPU)
    attr.gpu_platform_id = runtime.platformId;
  attr.access_bits =
      XPU_NETIO_MR_ACCESS_LOCAL_WRITE | XPU_NETIO_MR_ACCESS_REMOTE_WRITE;
  attr.flags = XPU_NETIO_MR_FLAG_STRICT_ORDERING;
  local = result(
      xpu_netio_register_memory_retryable(owner->context, &attr, &window->mr));
  xpu_netio_mr_blob_t blob{};
  if (local == flagcxSuccess)
    local = result(xpu_netio_export_memory(window->mr, &blob));
  if (local == flagcxSuccess)
    local = result(xpu_netio_get_local_region(window->mr, &window->local));
  common = stage(comm, local);
  if (common != flagcxSuccess) {
    if (releaseWindow(window) != flagcxSuccess) {
      window->orphanNext = owner->orphanWindows;
      owner->orphanWindows = window;
    }
    return common;
  }

  std::vector<xpu_netio_mr_blob_t> blobs;
  local = gather(comm, blob, blobs);
  if (local != flagcxSuccess) {
    if (releaseWindow(window) != flagcxSuccess) {
      window->orphanNext = owner->orphanWindows;
      owner->orphanWindows = window;
    }
    return local;
  }
  std::vector<xpu_netio_remote_region_t> regions;
  try {
    regions.resize(comm->nranks);
  } catch (const std::bad_alloc &) {
    local = flagcxSystemError;
  }
  common = stage(comm, local);
  if (common != flagcxSuccess) {
    if (releaseWindow(window) != flagcxSuccess) {
      window->orphanNext = owner->orphanWindows;
      owner->orphanWindows = window;
    }
    return common;
  }
  for (int peer = 0; peer < comm->nranks && local == flagcxSuccess; ++peer) {
    local = result(xpu_netio_import_remote_memory_retryable(
        owner->context, &blobs[peer], &window->remote[peer]));
    if (local == flagcxSuccess)
      local = result(
          xpu_netio_get_remote_region(window->remote[peer], &regions[peer]));
    if (local == flagcxSuccess && regions[peer].length < bytes)
      local = flagcxInvalidArgument;
  }
  if (local == flagcxSuccess) {
    local = deviceAdaptor->deviceMalloc(
        reinterpret_cast<void **>(&window->deviceRegions),
        regions.size() * sizeof(regions[0]), flagcxMemDevice, nullptr);
    if (local == flagcxSuccess)
      local = deviceAdaptor->deviceMemcpy(window->deviceRegions, regions.data(),
                                          regions.size() * sizeof(regions[0]),
                                          flagcxMemcpyHostToDevice, nullptr,
                                          nullptr);
  }
  common = stage(comm, local);
  if (common != flagcxSuccess) {
    if (releaseWindow(window) != flagcxSuccess) {
      window->orphanNext = owner->orphanWindows;
      owner->orphanWindows = window;
    }
    return common;
  }
  *output = window;
  return flagcxSuccess;
}

unsigned operationCapabilities(const xpu_netio_caps_t &caps) {
  unsigned supported = 0;
  if (caps.operation_bits & XPU_NETIO_OP_WRITE)
    supported |= DefaultNetGdaPut;
  if (caps.operation_bits & XPU_NETIO_OP_INLINE)
    supported |= DefaultNetGdaPutValue;
  if ((caps.operation_bits & (XPU_NETIO_OP_FLUSH | XPU_NETIO_OP_FLUSH_ASYNC)) ==
      (XPU_NETIO_OP_FLUSH | XPU_NETIO_OP_FLUSH_ASYNC))
    supported |= DefaultNetGdaFlush;
  // This adapter implements a single-writer Barrier slot using inline writes.
  // It does not implement the public multi-writer SignalInc/Add contract.
  if ((supported & (DefaultNetGdaPutValue | DefaultNetGdaFlush)) ==
          (DefaultNetGdaPutValue | DefaultNetGdaFlush) &&
      (caps.ordering_bits & XPU_NETIO_ORDERED_TRANSPORT))
    supported |= DefaultNetGdaBarrier;
  return supported;
}

struct Candidate {
  xpu_netio_abi_header_t provider;
  int32_t netType;
  uint32_t required;
  uint32_t capabilities;
  uint32_t reserved;
};

bool sameProvider(const xpu_netio_abi_header_t &a,
                  const xpu_netio_abi_header_t &b) {
  return a.provider_id == b.provider_id &&
         a.device_build_id == b.device_build_id &&
         a.device_abi_major == b.device_abi_major &&
         a.device_context_size == b.device_context_size &&
         a.device_qp_size == b.device_qp_size &&
         std::memcmp(a.provider_uuid, b.provider_uuid,
                     XPU_NETIO_PROVIDER_UUID_SIZE) == 0;
}

} // namespace

flagcxResult_t createCommAttempt(flagcxComm_t comm, flagcxDevComm_t devComm,
                                 const flagcxDevCommRequirements *reqs,
                                 int netType) {
  if (comm == nullptr || devComm == nullptr || reqs == nullptr)
    return flagcxInvalidArgument;
  FLAGCXCHECK(stage(comm, comm->devNetCommHandle == nullptr
                              ? flagcxSuccess
                              : flagcxInvalidUsage));
  if (netType != -1 && netType != 1)
    return flagcxInvalidArgument;
  unsigned requiredOperations =
      netType == 1 ? kAllToAllCapabilities : DefaultNetGdaRequiredCapabilities;
  if (reqs->interSignalCount > 0)
    requiredOperations |= DefaultNetGdaSignal | DefaultNetGdaSignalCompletion;
  if (reqs->interCounterCount > 0)
    requiredOperations |= DefaultNetGdaCounterCompletion;
  const unsigned implemented = kAllToAllCapabilities;
  flagcxResult_t support =
      stage(comm, (requiredOperations & ~implemented) != 0 ? flagcxNotSupported
                                                           : flagcxSuccess);
  if (support != flagcxSuccess)
    return support;

  CommOwner *owner = new (std::nothrow) CommOwner;
  flagcxResult_t allocation =
      stage(comm, owner ? flagcxSuccess : flagcxSystemError);
  if (allocation != flagcxSuccess) {
    delete owner;
    return allocation;
  }
  devComm->gdaHostOwner = owner;
  owner->comm = comm;
  xpu_netio_device_handle_init(&owner->published);

  flagcxResult_t local = flagcxSuccess;
  const char *provider = std::getenv("FLAGCX_XPU_NETIO_PROVIDER");
  local = result(
      xpu_netio_load_provider(provider ? provider : "mlx5", &owner->ops));
  if (local == flagcxSuccess)
    local = result(xpu_netio_query_retryable_cleanup(owner->ops));
  uint64_t count = 0;
  if (local == flagcxSuccess)
    local = result(xpu_netio_enumerate_devices(owner->ops, nullptr, &count));
  if (local == flagcxSuccess && (count == 0 || count > UINT32_MAX))
    local = flagcxNotSupported;
  std::vector<xpu_netio_device_info_t> devices;
  if (local == flagcxSuccess) {
    try {
      devices.resize(count);
    } catch (const std::bad_alloc &) {
      local = flagcxSystemError;
    }
  }
  if (local == flagcxSuccess) {
    local =
        result(xpu_netio_enumerate_devices(owner->ops, devices.data(), &count));
  }
  if (local == flagcxSuccess) {
    const char *deviceName = std::getenv("FLAGCX_XPU_NETIO_DEVICE");
    const xpu_netio_device_info_t *chosen = nullptr;
    for (const auto &candidate : devices)
      if (deviceName == nullptr ||
          std::strcmp(candidate.name, deviceName) == 0) {
        chosen = &candidate;
        break;
      }
    local = chosen ? result(xpu_netio_open_device_retryable(owner->ops, chosen,
                                                            &owner->device))
                   : flagcxNotSupported;
  }
  if (local == flagcxSuccess)
    local =
        result(xpu_netio_query_caps(owner->ops, owner->device, &owner->caps));
  if (local == flagcxSuccess) {
    owner->capabilities = operationCapabilities(owner->caps);
    const uint64_t requiredOps = XPU_NETIO_OP_WRITE | XPU_NETIO_OP_INLINE |
                                 XPU_NETIO_OP_FLUSH | XPU_NETIO_OP_FLUSH_ASYNC;
    const uint64_t requiredOrder = XPU_NETIO_ORDERED_TRANSPORT |
                                   XPU_NETIO_ORDER_SOURCE_REUSE |
                                   XPU_NETIO_ORDER_SYSTEM_FENCE;
    // Every Barrier channel reserves one source slot per Inter-team rank.
    // World uses the same Inter phase followed by intra-node synchronization.
    // Validate signed bounds before subtracting or widening the slot counts.
    const bool barrierStorageValid =
        devComm->nTeamRanks > 0 && devComm->interBarrierCount > 0 &&
        devComm->barrierSignalBase >= 0 &&
        devComm->signalCount >= devComm->barrierSignalBase &&
        static_cast<uint64_t>(devComm->nTeamRanks) *
                static_cast<uint64_t>(devComm->interBarrierCount) <=
            static_cast<uint64_t>(devComm->signalCount -
                                  devComm->barrierSignalBase);
    if ((owner->caps.operation_bits & requiredOps) != requiredOps ||
        (owner->caps.ordering_bits & requiredOrder) != requiredOrder ||
        (owner->capabilities & requiredOperations) != requiredOperations ||
        owner->caps.max_endpoints < static_cast<uint32_t>(comm->nranks) ||
        owner->caps.progress_mode == XPU_NETIO_PROGRESS_REQUIRED ||
        owner->caps.max_inline_bytes < sizeof(uint64_t) ||
        devComm->contextCount != 1 || devComm->signalBuffer == nullptr ||
        devComm->signalHostBuffer != nullptr || !barrierStorageValid ||
        comm->nranks >
            INT_MAX / static_cast<int>(sizeof(xpu_netio_connect_blob_t)))
      local = flagcxNotSupported;
    if (local == flagcxSuccess &&
        defaultNetSelectProvider<typename DeviceAPI::Platform>(
            kProviderId, owner->capabilities, requiredOperations, true) !=
            DefaultNetProviderGda)
      local = flagcxNotSupported;
  }
  flagcxResult_t common = stage(comm, local);
  if (common != flagcxSuccess)
    return common;

  Candidate candidate{};
  candidate.provider = owner->ops->hdr;
  candidate.netType = netType;
  candidate.required = requiredOperations;
  candidate.capabilities = owner->capabilities;
  std::vector<Candidate> candidates;
  FLAGCXCHECK(gather(comm, candidate, candidates));
  for (const Candidate &peer : candidates)
    if (!sameProvider(peer.provider, candidate.provider) ||
        peer.required != candidate.required ||
        peer.netType != candidate.netType ||
        (peer.capabilities & requiredOperations) != requiredOperations)
      return flagcxNotSupported;

  xpu_netio_context_attr_t attr{};
  xpu_netio_context_attr_init(&attr);
  attr.device = owner->device;
  const XpuNetioGdaRuntime runtime = flagcxXpuNetioGdaRuntime();
  local = runtime.ops != nullptr && runtime.memoryCapability != 0
              ? flagcxSuccess
              : flagcxNotSupported;
  common = stage(comm, local);
  if (common != flagcxSuccess)
    return common;
  attr.runtime_ops = runtime.ops;
  attr.device_context_count = 1;
  attr.endpoints_per_context = comm->nranks;
  attr.required_operation_bits = XPU_NETIO_OP_WRITE | XPU_NETIO_OP_WRITE_ASYNC |
                                 XPU_NETIO_OP_INLINE | XPU_NETIO_OP_FLUSH |
                                 XPU_NETIO_OP_FLUSH_ASYNC;
  attr.required_ordering_bits = XPU_NETIO_ORDERED_TRANSPORT |
                                XPU_NETIO_ORDER_SOURCE_REUSE |
                                XPU_NETIO_ORDER_SYSTEM_FENCE;
  attr.required_memory_bits = runtime.memoryCapability |
                              XPU_NETIO_MEMORY_REMOTE_WRITE |
                              XPU_NETIO_MEMORY_STRICT_ORDERING;
  attr.required_completion_bits = XPU_NETIO_COMPLETION_LOCAL_SEND |
                                  XPU_NETIO_COMPLETION_ASYNC_REQUEST |
                                  XPU_NETIO_COMPLETION_TIMEOUT_WAIT;
  attr.required_posting_bits = XPU_NETIO_POST_WAIT_CONTROL;
  attr.required_routing_bits =
      XPU_NETIO_ROUTING_LOGICAL_PEER | XPU_NETIO_ROUTING_SELF;
  attr.required_sharing_bits = XPU_NETIO_SHARING_THREAD;
  local = result(
      xpu_netio_create_context_retryable(owner->ops, &attr, &owner->context));
  common = stage(comm, local);
  if (common != flagcxSuccess)
    return common;

  std::vector<xpu_netio_connect_blob_t> endpoints;
  local = flagcxSuccess;
  try {
    endpoints.resize(static_cast<size_t>(comm->nranks) * comm->nranks);
    owner->qps.resize(comm->nranks, nullptr);
  } catch (const std::bad_alloc &) {
    local = flagcxSystemError;
  }
  common = stage(comm, local);
  if (common != flagcxSuccess)
    return common;
  for (int peer = 0; peer < comm->nranks && local == flagcxSuccess; ++peer) {
    xpu_netio_qp_attr_t qpAttr{};
    xpu_netio_qp_attr_init(&qpAttr);
    qpAttr.queue_depth = std::max(owner->caps.min_queue_depth,
                                  std::min(1024u, owner->caps.max_queue_depth));
    qpAttr.traffic_class = -1;
    qpAttr.ack_timeout = 20;
    qpAttr.retry_count = 7;
    qpAttr.service_level = 5;
    local = result(xpu_netio_create_qp_retryable(owner->context, &qpAttr,
                                                 &owner->qps[peer]));
    if (local == flagcxSuccess)
      local = result(xpu_netio_export_endpoint(
          owner->qps[peer], &endpoints[comm->rank * comm->nranks + peer]));
  }
  common = stage(comm, local);
  if (common != flagcxSuccess)
    return common;
  FLAGCXCHECK(bootstrapCollAllGather(comm->bootstrap, endpoints.data(),
                                     comm->nranks * sizeof(endpoints[0])));
  for (int peer = 0; peer < comm->nranks && local == flagcxSuccess; ++peer)
    local = result(xpu_netio_connect_endpoint(
        owner->qps[peer], &endpoints[peer * comm->nranks + comm->rank]));
  common = stage(comm, local);
  if (common != flagcxSuccess)
    return common;

  uint32_t auxCount = 0;
  local =
      result(xpu_netio_get_aux_endpoint_count(owner->context, 0, &auxCount));
  common = stage(comm, local);
  if (common != flagcxSuccess)
    return common;
  std::vector<uint32_t> auxCounts;
  FLAGCXCHECK(gather(comm, auxCount, auxCounts));
  for (uint32_t n : auxCounts)
    if (n != auxCount)
      return flagcxNotSupported;
  for (uint32_t index = 0; index < auxCount; ++index) {
    xpu_netio_connect_blob_t mine{};
    local =
        result(xpu_netio_export_aux_endpoint(owner->context, 0, index, &mine));
    common = stage(comm, local);
    if (common != flagcxSuccess)
      return common;
    std::vector<xpu_netio_connect_blob_t> all;
    FLAGCXCHECK(gather(comm, mine, all));
    for (int peer = 0; peer < comm->nranks && local == flagcxSuccess; ++peer)
      local = result(xpu_netio_connect_context_endpoint(
          owner->context, 0, peer, XPU_NETIO_CONTEXT_ENDPOINT_AUXILIARY, index,
          &all[peer]));
    common = stage(comm, local);
    if (common != flagcxSuccess)
      return common;
  }

  xpu_netio_publish_attr_t publish{};
  xpu_netio_publish_attr_init(&publish);
  publish.context_count = 1;
  publish.queue_pairs_per_context = comm->nranks;
  owner->publishAttempted = true;
  local = result(xpu_netio_publish_qps_retryable(
      owner->context, owner->qps.data(), owner->qps.size(), &publish,
      &owner->published));
  if (local == flagcxSuccess &&
      (owner->published.published_context_count != 1 ||
       owner->published.device_contexts == nullptr ||
       owner->published.host_device_contexts == nullptr ||
       owner->published.host_device_contexts[0].provider_id !=
           owner->ops->hdr.provider_id ||
       owner->published.needs_host_progress != 0))
    local = flagcxNotSupported;
  common = stage(comm, local);
  if (common != flagcxSuccess)
    return common;

  FLAGCXCHECK(registerRegion(comm, owner, devComm->signalBuffer,
                             static_cast<size_t>(devComm->signalCount) *
                                 devComm->contextCount * sizeof(uint64_t),
                             &owner->signals));
  local =
      deviceAdaptor->deviceMalloc(reinterpret_cast<void **>(&owner->errorFlag),
                                  sizeof(uint32_t), flagcxMemDevice, nullptr);
  if (local == flagcxSuccess)
    local = deviceAdaptor->deviceMemset(owner->errorFlag, 0, sizeof(uint32_t),
                                        flagcxMemDevice, nullptr);
  flagcxXpuNetioDeviceContext view{};
  view.contexts = owner->published.device_contexts;
  view.signalRemote = owner->signals->deviceRegions;
  view.signalLocal = owner->signals->local;
  view.signalBuffer = devComm->signalBuffer;
  view.errorFlag = owner->errorFlag;
  view.timeoutNs = kTimeoutNs;
  view.rank = comm->rank;
  view.nranks = comm->nranks;
  view.signalCount = devComm->signalCount;
  if (local == flagcxSuccess)
    local = deviceAdaptor->deviceMalloc(
        reinterpret_cast<void **>(&owner->deviceView), sizeof(view),
        flagcxMemDevice, nullptr);
  if (local == flagcxSuccess)
    local =
        deviceAdaptor->deviceMemcpy(owner->deviceView, &view, sizeof(view),
                                    flagcxMemcpyHostToDevice, nullptr, nullptr);
  common = stage(comm, local);
  if (common != flagcxSuccess)
    return common;

  devComm->gdaProvider = 1;
  devComm->gdaProviderId = kProviderId;
  devComm->gdaDeviceContext = owner->deviceView;
  return flagcxSuccess;
}

static flagcxResult_t registerExistingWindows(flagcxComm_t comm,
                                              flagcxDevComm_t devComm);
static flagcxResult_t detachOwnedWindows(CommOwner *owner);

flagcxResult_t createComm(flagcxComm_t comm, flagcxDevComm_t devComm,
                          const flagcxDevCommRequirements *reqs, int netType) {
  if (comm == nullptr || devComm == nullptr || reqs == nullptr)
    return flagcxInvalidArgument;
  flagcxResult_t status = createCommAttempt(comm, devComm, reqs, netType);
  if (status == flagcxSuccess && devComm->gdaProvider != 0)
    status = registerExistingWindows(comm, devComm);
  if (status == flagcxSuccess)
    return status;

  // A failed allocation can leave an owner on only some ranks. Every rank
  // must converge after local cleanup before AUTO may choose Proxy.
  CommOwner *owner = static_cast<CommOwner *>(devComm->gdaHostOwner);
  flagcxResult_t cleanup =
      owner != nullptr ? detachOwnedWindows(owner) : flagcxSuccess;
  if (cleanup == flagcxSuccess && owner != nullptr)
    cleanup = releaseComm(owner);
  if (cleanup == flagcxSuccess) {
    devComm->gdaHostOwner = nullptr;
    devComm->gdaDeviceContext = nullptr;
    devComm->gdaProvider = 0;
    devComm->gdaProviderId = 0;
  }
  flagcxResult_t commonCleanup = stage(comm, cleanup);
  if (commonCleanup != flagcxSuccess)
    return commonCleanup;

  if (netType == -1 && status != flagcxInvalidUsage &&
      status != flagcxInvalidArgument)
    return flagcxSuccess;
  return status;
}

flagcxResult_t destroyComm(flagcxDevComm_t devComm) {
  if (devComm == nullptr || devComm->gdaHostOwner == nullptr)
    return flagcxSuccess;
  CommOwner *owner = static_cast<CommOwner *>(devComm->gdaHostOwner);
  FLAGCXCHECK(detachOwnedWindows(owner));
  FLAGCXCHECK(releaseComm(owner));
  devComm->gdaHostOwner = nullptr;
  devComm->gdaDeviceContext = nullptr;
  devComm->gdaProvider = 0;
  devComm->gdaProviderId = 0;
  return flagcxSuccess;
}

flagcxResult_t registerWindow(flagcxComm_t comm, flagcxDevComm_t devComm,
                              flagcxWindow_t hostWindow) {
  if (devComm == nullptr || devComm->gdaProvider == 0)
    return flagcxSuccess;
  if (comm == nullptr)
    return flagcxInvalidArgument;
  flagcxResult_t local = hostWindow != nullptr &&
                                 hostWindow->userBase != nullptr &&
                                 hostWindow->userSize != 0 &&
                                 hostWindow->devNetRegistrationId != 0 &&
                                 hostWindow->gdaWindowOwner == nullptr
                             ? flagcxSuccess
                             : flagcxInvalidArgument;
  FLAGCXCHECK(stage(comm, local));
  std::vector<uint64_t> ids;
  FLAGCXCHECK(gather(comm, hostWindow->devNetRegistrationId, ids));
  for (uint64_t id : ids)
    if (id != hostWindow->devNetRegistrationId)
      return flagcxInvalidUsage;
  CommOwner *owner = static_cast<CommOwner *>(devComm->gdaHostOwner);
  WindowOwner *window = nullptr;
  WindowOwner *previousOrphans = owner->orphanWindows;
  flagcxResult_t registration = registerRegion(
      comm, owner, hostWindow->userBase, hostWindow->userSize, &window);
  if (registration != flagcxSuccess) {
    hostWindow->gdaRegistrationActive = true;
    if (owner->orphanWindows != previousOrphans) {
      WindowOwner *orphan = owner->orphanWindows;
      owner->orphanWindows = orphan->orphanNext;
      orphan->orphanNext = nullptr;
      hostWindow->gdaWindowOwner = orphan;
      ++owner->liveWindows;
    }
    return registration;
  }
  flagcxXpuNetioDeviceWindow view{};
  view.local = window->local;
  view.remote = window->deviceRegions;
  view.bytes = hostWindow->userSize;
  local = deviceAdaptor->deviceMalloc(
      reinterpret_cast<void **>(&window->deviceView), sizeof(view),
      flagcxMemDevice, nullptr);
  if (local == flagcxSuccess)
    local =
        deviceAdaptor->deviceMemcpy(window->deviceView, &view, sizeof(view),
                                    flagcxMemcpyHostToDevice, nullptr, nullptr);
  flagcxResult_t common = stage(comm, local);
  if (common != flagcxSuccess) {
    if (releaseWindow(window) != flagcxSuccess) {
      hostWindow->gdaWindowOwner = window;
      ++owner->liveWindows;
    }
    hostWindow->gdaRegistrationActive = true;
    return common;
  }
  hostWindow->gdaWindowOwner = window;
  hostWindow->gdaDeviceView = window->deviceView;
  hostWindow->gdaRegistrationActive = true;
  ++owner->liveWindows;
  return flagcxSuccess;
}

flagcxResult_t prepareWindowDeregister(flagcxWindow_t hostWindow) {
  if (hostWindow == nullptr)
    return flagcxSuccess;
  if (hostWindow->gdaDevMemRefs != 0)
    return flagcxInvalidUsage;
  if (hostWindow->gdaWindowOwner == nullptr)
    return flagcxSuccess;
  return deviceAdaptor->deviceSynchronize();
}

flagcxResult_t deregisterWindow(flagcxWindow_t hostWindow) {
  if (hostWindow == nullptr || hostWindow->gdaWindowOwner == nullptr)
    return flagcxSuccess;
  if (hostWindow->gdaDevMemRefs != 0)
    return flagcxInvalidUsage;
  WindowOwner *window = static_cast<WindowOwner *>(hostWindow->gdaWindowOwner);
  CommOwner *owner = window->comm;
  FLAGCXCHECK(releaseWindow(window));
  hostWindow->gdaWindowOwner = nullptr;
  hostWindow->gdaDeviceView = nullptr;
  --owner->liveWindows;
  return flagcxSuccess;
}

flagcxResult_t bindDevMem(flagcxWindow_t hostWindow, flagcxDevMem_t devMem,
                          size_t bytes, void *deviceWindow) {
  if (hostWindow == nullptr || devMem == nullptr || deviceWindow == nullptr ||
      hostWindow->gdaWindowOwner == nullptr ||
      hostWindow->gdaDeviceView == nullptr ||
      hostWindow->gdaDevMemRefs == UINT_MAX)
    return flagcxInvalidUsage;
  const uintptr_t base = reinterpret_cast<uintptr_t>(hostWindow->userBase);
  const uintptr_t ptr = reinterpret_cast<uintptr_t>(devMem->rawPtr);
  // Remote offsets are relative to each peer's registered Window. A local
  // subview offset cannot describe an independently created peer subview.
  if (ptr != base || bytes != hostWindow->userSize)
    return flagcxNotSupported;
  DefaultNetProviderPolicy<typename DeviceAPI::Platform>::publishWindow(
      *static_cast<DeviceAPI::Window *>(deviceWindow),
      hostWindow->gdaDeviceView);
  devMem->gdaWindowOwner = hostWindow;
  devMem->hasWindow = true;
  ++hostWindow->gdaDevMemRefs;
  return flagcxSuccess;
}

flagcxResult_t unbindDevMem(flagcxDevMem_t devMem) {
  if (devMem == nullptr || devMem->gdaWindowOwner == nullptr)
    return flagcxSuccess;
  flagcxWindow_t hostWindow =
      static_cast<flagcxWindow_t>(devMem->gdaWindowOwner);
  if (hostWindow->gdaDevMemRefs == 0)
    return flagcxInternalError;
  --hostWindow->gdaDevMemRefs;
  devMem->gdaWindowOwner = nullptr;
  return flagcxSuccess;
}

static flagcxResult_t registerExistingWindows(flagcxComm_t comm,
                                              flagcxDevComm_t devComm) {
  std::vector<flagcxWindow_t> windows;
  flagcxResult_t local = flagcxSuccess;
  try {
    for (flagcxWindow_t window = comm->devNetWindows; window != nullptr;
         window = window->devNetNext)
      windows.push_back(window);
  } catch (const std::bad_alloc &) {
    local = flagcxSystemError;
  }
  FLAGCXCHECK(stage(comm, local));
  std::vector<size_t> counts;
  FLAGCXCHECK(gather(comm, windows.size(), counts));
  for (size_t count : counts)
    if (count != windows.size())
      return flagcxInvalidUsage;

  // The list is head-inserted. Replay collective registrations in the
  // original Window order on every rank.
  std::reverse(windows.begin(), windows.end());
  // Validate the entire sequence before importing any MR. Equal list lengths
  // alone cannot detect different local deregistrations on different ranks.
  for (flagcxWindow_t window : windows) {
    std::vector<uint64_t> ids;
    FLAGCXCHECK(gather(comm, window->devNetRegistrationId, ids));
    if (window->devNetRegistrationId == 0)
      return flagcxInvalidUsage;
    for (uint64_t id : ids)
      if (id != window->devNetRegistrationId)
        return flagcxInvalidUsage;
  }
  for (flagcxWindow_t window : windows)
    FLAGCXCHECK(registerWindow(comm, devComm, window));

  // Preexisting DevMem views may already have device snapshots. Drain work
  // before replacing their copied Window state with the new GDA handle.
  FLAGCXCHECK(stage(comm, deviceAdaptor->deviceSynchronize()));
  local = flagcxSuccess;
  for (flagcxWindow_t window : windows) {
    for (flagcxDevMem_t view = window->devMemViews; view != nullptr;
         view = view->gdaWindowNext) {
      if (view->window == nullptr)
        continue;
      local = bindDevMem(window, view, view->viewSize, view->window);
      if (local == flagcxSuccess && view->cachedDevicePtr != nullptr) {
        flagcxDevMem hostCopy(*view);
        local = deviceAdaptor->deviceMemcpy(
            view->cachedDevicePtr, &hostCopy, sizeof(hostCopy),
            flagcxMemcpyHostToDevice, nullptr, nullptr);
      }
      if (local != flagcxSuccess)
        break;
    }
    if (local != flagcxSuccess)
      break;
  }
  return stage(comm, local);
}

static flagcxResult_t detachOwnedWindows(CommOwner *owner) {
  if (owner == nullptr || owner->comm == nullptr)
    return flagcxSuccess;
  FLAGCXCHECK(deviceAdaptor->deviceSynchronize());
  for (flagcxWindow_t window = owner->comm->devNetWindows; window != nullptr;
       window = window->devNetNext) {
    WindowOwner *registration =
        static_cast<WindowOwner *>(window->gdaWindowOwner);
    if (registration == nullptr || registration->comm != owner)
      continue;
    for (flagcxDevMem_t view = window->devMemViews; view != nullptr;
         view = view->gdaWindowNext) {
      if (view->window == nullptr)
        continue;
      DefaultNetProviderPolicy<typename DeviceAPI::Platform>::initWindow(
          *static_cast<DeviceAPI::Window *>(view->window));
      view->hasWindow =
          static_cast<DeviceAPI::Window *>(view->window)->hasAccess();
      if (view->gdaWindowOwner != nullptr)
        FLAGCXCHECK(unbindDevMem(view));
      if (view->cachedDevicePtr != nullptr) {
        flagcxDevMem hostCopy(*view);
        FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
            view->cachedDevicePtr, &hostCopy, sizeof(hostCopy),
            flagcxMemcpyHostToDevice, nullptr, nullptr));
      }
    }
    FLAGCXCHECK(deregisterWindow(window));
    window->gdaRegistrationActive = false;
  }
  for (flagcxWindow_t window = owner->comm->devNetWindows; window != nullptr;
       window = window->devNetNext)
    if (window->gdaWindowOwner == nullptr)
      window->gdaRegistrationActive = false;
  return flagcxSuccess;
}

flagcxResult_t queryComm(flagcxDevComm_t devComm, unsigned *capabilities,
                         uint32_t **deviceErrorFlag, uint64_t *maxChunkBytes,
                         unsigned *requiredAlignment) {
  if (devComm == nullptr || devComm->gdaProvider == 0 ||
      devComm->gdaHostOwner == nullptr || capabilities == nullptr ||
      deviceErrorFlag == nullptr || maxChunkBytes == nullptr ||
      requiredAlignment == nullptr)
    return flagcxInvalidArgument;
  const CommOwner *owner =
      static_cast<const CommOwner *>(devComm->gdaHostOwner);
  *capabilities = owner->capabilities;
  *deviceErrorFlag = owner->errorFlag;
  *maxChunkBytes = std::min(owner->caps.max_transfer_bytes,
                            owner->caps.max_write_chunk_bytes);
  *requiredAlignment = owner->caps.required_alignment;
  return flagcxSuccess;
}

flagcxResult_t clearError(flagcxDevComm_t devComm) {
  if (devComm == nullptr || devComm->gdaProvider == 0 ||
      devComm->gdaHostOwner == nullptr)
    return flagcxInvalidArgument;
  CommOwner *owner = static_cast<CommOwner *>(devComm->gdaHostOwner);
  return deviceAdaptor->deviceMemset(owner->errorFlag, 0, sizeof(uint32_t),
                                     flagcxMemDevice, nullptr);
}

flagcxResult_t pollError(flagcxDevComm_t devComm) {
  if (devComm == nullptr || devComm->gdaProvider == 0 ||
      devComm->gdaHostOwner == nullptr)
    return flagcxInvalidArgument;
  CommOwner *owner = static_cast<CommOwner *>(devComm->gdaHostOwner);
  xpu_netio_error_info_t info{};
  xpu_netio_error_info_init(&info);
  FLAGCXCHECK(result(xpu_netio_query_error(&owner->published, &info)));
  if (info.category != XPU_NETIO_SUCCESS)
    return result(info.category);
  uint32_t deviceStatus = 0;
  FLAGCXCHECK(deviceAdaptor->deviceMemcpy(
      &deviceStatus, owner->errorFlag, sizeof(deviceStatus),
      flagcxMemcpyDeviceToHost, nullptr, nullptr));
  return result(static_cast<xpu_netio_status_t>(deviceStatus));
}

} // namespace XpuNetioGda

const flagcxDevNetAdaptor flagcxDevNetGda = {
    XpuNetioGda::createComm,       XpuNetioGda::destroyComm,
    XpuNetioGda::registerWindow,   XpuNetioGda::prepareWindowDeregister,
    XpuNetioGda::deregisterWindow, XpuNetioGda::bindDevMem,
    XpuNetioGda::unbindDevMem};
