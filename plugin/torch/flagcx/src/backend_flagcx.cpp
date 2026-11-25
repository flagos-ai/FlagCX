/*************************************************************************
 * Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd.
   All Rights Reserved.
 * Copyright (c) 2025 by DU. All Rights Reserved.
 ************************************************************************/
#include "backend_flagcx.hpp"
#include "utils_flagcx.hpp"
#include <algorithm>
#include <c10/util/ScopeExit.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace c10d {
namespace {

bool heteroP2pRequested() {
  const char *useHeteroComm = std::getenv("FLAGCX_USE_HETERO_COMM");
  if (useHeteroComm != nullptr && std::stoi(useHeteroComm) == 1) {
    return true;
  }

  const char *clusterSplitInfo = std::getenv("FLAGCX_CLUSTER_SPLIT_LIST");
  if (clusterSplitInfo == nullptr) {
    return false;
  }

  std::stringstream ss(clusterSplitInfo);
  std::string token;
  int totalClusters = 0;
  try {
    while (std::getline(ss, token, ',')) {
      totalClusters += std::stoi(token);
    }
  } catch (const std::exception &) {
    // Let FlagCX core report the malformed split configuration. Avoid
    // constructing a pair communicator from a configuration intended for the
    // process-group communicator.
    return true;
  }
  return totalClusters > 1;
}

// FlagCX op mapping
const std::map<ReduceOp::RedOpType, flagcxRedOp_t> flagcxOp = {
    {ReduceOp::MIN, flagcxMin}, {ReduceOp::MAX, flagcxMax},
    {ReduceOp::SUM, flagcxSum}, {ReduceOp::PRODUCT, flagcxProd},
    {ReduceOp::AVG, flagcxAvg},
};

// Helper function that gets the FlagCX reduction operation
flagcxRedOp_t getFlagcxReduceOp(const ReduceOp &reduceOp, at::Tensor &input,
                                const flagcxDataType_t &dataType) {
  try {
    if (input.scalar_type() == at::kBool) {
      if (reduceOp == ReduceOp::SUM) {
        // For bool tensors, map sum to max, which both represent a bitwise or.
        // This is to prevent overflow issues with sum, since we use uint8 to
        // represent a bool (see ncclDataType mapping).
        return flagcxMax;
      }
      if (reduceOp == ReduceOp::AVG) {
        C10_THROW_ERROR(TypeError,
                        "Cannot use ReduceOp.AVG with boolean inputs");
      }
    }
    return flagcxOp.at(reduceOp);
  } catch (const std::out_of_range &) {
    switch (reduceOp) {
      case ReduceOp::AVG:
        C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.AVG with FlagCX");
        break;
      case ReduceOp::BAND:
        C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.BAND with FlagCX");
        break;
      case ReduceOp::BOR:
        C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.BOR with FlagCX");
        break;
      case ReduceOp::BXOR:
        C10_THROW_ERROR(ValueError, "Cannot use ReduceOp.BXOR with FlagCX");
        break;
      default:
        C10_THROW_ERROR(ValueError, "Unhandled ReduceOp");
        break;
    }
  }
}

// FlagCX type typing
std::map<at::ScalarType, flagcxDataType_t> flagcxDataType = {
    {at::kChar, flagcxInt8},
    {at::kByte, flagcxUint8},
    {at::kFloat, flagcxFloat},
    {at::kDouble, flagcxDouble},
    {at::kInt, flagcxInt32},
    {at::kLong, flagcxInt64},
    {at::kHalf, flagcxHalf},
    {at::kBool, flagcxUint8},
    {at::kFloat8_e5m2, flagcxUint8},
    {at::kFloat8_e4m3fn, flagcxUint8},
    /*
    {at::kFloat8_e4m3fnuz, flagcxUint8},
    {at::kFloat8_e5m2fnuz, flagcxUint8},
    */
    {at::kBFloat16, flagcxBfloat16},
};

// Helper function that gets the data type and issues error if not supported
flagcxDataType_t getFlagcxDataType(at::ScalarType type) {
  auto it = flagcxDataType.find(type);
  TORCH_CHECK_WITH(
      TypeError, it != flagcxDataType.end(),
      "Input tensor data type is not supported for FlagCX process group: ",
      type);
  return it->second;
}

bool check_same_size(const std::vector<at::Tensor> &inputTensors) {
  for (const auto &inputTensor : inputTensors) {
    if (!inputTensors[0].is_same_size(inputTensor)) {
      return false;
    }
  }
  return true;
}

at::Tensor newLikeFlatOnStream(std::vector<at::Tensor> &tensors,
                               flagcxStream_t stream, int deviceId) {
#ifdef FLAGCX_TORCH_BACKEND_FLAGOS
  // torch-fl's caching allocator associates new allocations with the current
  // device stream. Allocate flattened intermediates on the communication
  // stream so that releasing their Tensor handles cannot recycle the storage
  // before the queued collective and copy operations have completed.
  flagcxStreamGuard guard(stream, deviceId);
#endif
  return newLikeFlat(tensors);
}

void copyTensorOnStream(at::Tensor dst, const at::Tensor &src,
                        flagcxStream_t stream, flagcxDeviceHandle_t devHandle,
                        int deviceId) {
  TORCH_CHECK(dst.numel() == src.numel(),
              "FlagCX tensor copy requires equal element counts");
  TORCH_CHECK(dst.scalar_type() == src.scalar_type(),
              "FlagCX tensor copy requires equal data types");
#ifdef FLAGCX_TORCH_BACKEND_FLAGOS
  TORCH_CHECK(dst.is_contiguous() && src.is_contiguous(),
              "FlagOS collective intermediates must be contiguous");
  C10D_FLAGCX_CHECK(devHandle->deviceMemcpy(dst.data_ptr(), src.data_ptr(),
                                            dst.numel() * dst.element_size(),
                                            flagcxMemcpyDeviceToDevice, stream),
                    std::nullopt);
#else
  flagcxStreamGuard guard(stream, deviceId);
  // Register both buffers with the current Torch stream before queuing the
  // copy. Sunrise selects a pool stream here, which differs from the FlagCX
  // submission stream; the retained batch covers that stream separately.
  auto *guardImpl = c10::impl::getDeviceGuardImpl(dst.device().type());
  const auto copyStream = guardImpl->getStream(dst.device());
  guardImpl->recordDataPtrOnStream(src.storage().data_ptr(), copyStream);
  guardImpl->recordDataPtrOnStream(dst.storage().data_ptr(), copyStream);
  dst.copy_(src, true);
#endif
}

void check_device(at::Device dev1, at::Device dev2) {
#ifdef USE_CAMBRICON_ADAPTOR
  if (dev1.is_privateuseone() && dev2.is_privateuseone() && dev1 != dev2) {
    throw std::runtime_error(
        "flagcxBackend does not support multidevice tensors");
  }
#elif USE_ASCEND_ADAPTOR
  if (dev1.is_privateuseone() && dev2.is_privateuseone() && dev1 != dev2) {
    throw std::runtime_error(
        "flagcxBackend does not support multidevice tensors");
  }
#elif USE_TSM_ADAPTOR
  if (dev1.is_privateuseone() && dev2.is_privateuseone() && dev1 != dev2) {
    throw std::runtime_error(
        "flagcxBackend does not support multidevice tensors");
  }
#elif USE_ENFLAME_ADAPTOR
  if (dev1.is_privateuseone() && dev2.is_privateuseone() && dev1 != dev2) {
    throw std::runtime_error(
        "flagcxBackend does not support multidevice tensors");
  }
#elif USE_SUNRISE_ADAPTOR
  if (dev1.is_privateuseone() && dev2.is_privateuseone() && dev1 != dev2) {
    throw std::runtime_error(
        "flagcxBackend does not support multidevice tensors");
  }
#else
  if (dev1.is_cuda() && dev2.is_cuda() && dev1 != dev2) {
    throw std::runtime_error(
        "flagcxBackend does not support multidevice tensors");
  }
#endif
}

int64_t check_gpu_tensors_same_device(const std::vector<at::Tensor> &tensors) {
  if (tensors.empty()) {
    C10_THROW_ERROR(ValueError, "Tensor list must be nonempty");
  }

  const auto &first = tensors.front();

  int64_t totalNumel = 0;
  for (const auto &t : tensors) {
    if (t.is_sparse()) {
      C10_THROW_ERROR(ValueError, "Tensors must be dense");
    }
    if (t.scalar_type() != first.scalar_type()) {
      C10_THROW_ERROR(TypeError, "Tensors must have identical type");
    }
    if (!t.is_non_overlapping_and_dense()) {
      C10_THROW_ERROR(ValueError, "Tensors must be non-overlapping and dense");
    }
    // If we're in this function, the user called a _coalesced collective
    // on a set of tensors with potentially different sizes and strides.
    // Therefore, we don't check for matching sizes and strides,
    // but we do double-check tensors are on the same device.
    TORCH_CHECK_WITH(ValueError, t.get_device() == tensors[0].get_device(),
                     "Expected list of tensors on the same device");
    totalNumel += t.numel();
  }
  return totalNumel;
}
#if (defined(USE_NVIDIA_ADAPTOR) || defined(USE_METAX_ADAPTOR)) &&             \
    defined(TORCH_VER_GE_250)
std::string commOpToString(flagcxCommOp_t commOp) {
  switch (commOp) {
    case flagcxCommOpSend:
      return "send";
    case flagcxCommOpRecv:
      return "recv";
    case flagcxCommOpBroadcast:
      return "broadcast";
    case flagcxCommOpGather:
      return "gather";
    case flagcxCommOpScatter:
      return "scatter";
    case flagcxCommOpReduce:
      return "reduce";
    case flagcxCommOpAllReduce:
      return "allreduce";
    case flagcxCommOpAllGather:
      return "allgather";
    case flagcxCommOpReduceScatter:
      return "reducescatter";
    case flagcxCommOpAlltoAll:
      return "alltoall";
    case flagcxCommOpAlltoAllv:
      return "alltoallv";
    default:
      return "noop";
  }
}

size_t getDataSize(flagcxDataType_t dtype, size_t count) {
  return getFlagcxDataTypeSize(dtype) * count;
}

void recordFlagcxTuneObject(const flagcxBackend::TuneObjectKey &key,
                            int tuneGroupIdx) {
  using nlohmann::json;

  // Read env var ONCE — throw if missing or empty.
  static const std::string tuneFilePath = []() -> std::string {
    const char *base = std::getenv("FLAGCX_TUNE_FILE");
    if (!base || !*base) {
      throw std::runtime_error(
          "Environment variable FLAGCX_TUNE_FILE is not set or empty. "
          "TuneObject recording requires this file path.");
    }
    std::string path(base);
    // create a file for each process
    path += ".pid" + std::to_string(::getpid());
    return std::string(path);
  }();

  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);

  json root;

  // Load existing JSON file if present.
  std::ifstream in(tuneFilePath);
  if (in.good()) {
    try {
      in >> root;
    } catch (...) {
      // If the file exists but is corrupted, reset to empty object.
      root = json::object();
    }
  }

  const std::string groupKey = std::to_string(tuneGroupIdx);
  // Ensure group object exists.
  if (!root.contains(groupKey) || !root[groupKey].is_object()) {
    root[groupKey] = json::object();
  }

  // Ensure "tune_objects" is an array.
  if (!root[groupKey].contains("tune_objects") ||
      !root[groupKey]["tune_objects"].is_array()) {
    root[groupKey]["tune_objects"] = json::array();
  }

  // add new record
  root[groupKey]["tune_objects"].push_back({
      {"commOp", key.commOp},
      {"nBytes", key.nBytes},
  });

  // Write back to file.
  std::ofstream out(tuneFilePath, std::ios::trunc);
  if (!out.good()) {
    throw std::runtime_error("Failed to write tune object JSON file at: " +
                             tuneFilePath);
  }
  out << root.dump(2) << '\n';
}
#endif

} // namespace

bool flagcxWork::isCompleted() {
#if defined(FLAGCX_TORCH_BACKEND_FLAGOS) && defined(USE_ASCEND_ADAPTOR)
  int previousDevice = 0;
  C10D_FLAGCX_CHECK(devHandle_->getDevice(&previousDevice), std::nullopt);
  C10D_FLAGCX_CHECK(devHandle_->setDevice(deviceId_), std::nullopt);
  bool complete = false;
  try {
    complete = event_->query();
  } catch (...) {
    devHandle_->setDevice(previousDevice);
    throw;
  }
  C10D_FLAGCX_CHECK(devHandle_->setDevice(previousDevice), std::nullopt);
  return complete;
#else
  return future_->completed();
#endif
}

bool flagcxWork::isSuccess() const { return future_->hasValue(); }

bool flagcxWork::wait(std::chrono::milliseconds /* unused */) {
#if defined(FLAGCX_TORCH_BACKEND_FLAGOS) &&                                    \
    (defined(USE_ENFLAME_ADAPTOR) || defined(USE_ASCEND_ADAPTOR))
  int previousDevice = 0;
  C10D_FLAGCX_CHECK(devHandle_->getDevice(&previousDevice), std::nullopt);
  C10D_FLAGCX_CHECK(devHandle_->setDevice(deviceId_), std::nullopt);
  try {
    C10D_FLAGCX_CHECK(devHandle_->streamSynchronize(stream_), std::nullopt);
  } catch (...) {
    devHandle_->setDevice(previousDevice);
    throw;
  }
  C10D_FLAGCX_CHECK(devHandle_->setDevice(previousDevice), std::nullopt);
#else
  event_->block(deviceId_);
  if (isBarrierOp_) {
    C10D_FLAGCX_CHECK(devHandle_->streamSynchronize(stream_), std::nullopt);
  }
#endif
  return true;
}

c10::intrusive_ptr<c10::ivalue::Future> flagcxWork::getFuture() {
#if defined(FLAGCX_TORCH_BACKEND_FLAGOS) &&                                    \
    (defined(USE_ENFLAME_ADAPTOR) || defined(USE_ASCEND_ADAPTOR))
  wait();
#endif
  return future_;
}

// If necessary, pass store/rank/size to the ctor and exchange connection
// information here
#if (defined(USE_NVIDIA_ADAPTOR) || defined(USE_METAX_ADAPTOR)) &&             \
    defined(TORCH_VER_GE_250)
flagcxBackend::flagcxBackend(const c10::intrusive_ptr<::c10d::Store> &store,
                             int rank, int size,
                             c10::intrusive_ptr<Options> options)
    : Backend(rank, size), store_(store),
      options_(options == nullptr ? Options::create() : std::move(options)) {
  deviceId_ = 0;
  status_ = 0;
  activeGroupCounter_ = 0;
  C10D_FLAGCX_CHECK(flagcxDeviceHandleInit(&devHandle_), std::nullopt);
  C10D_FLAGCX_CHECK(devHandle_->getDeviceCount(&nDevs_), std::nullopt);
  usePairComm_ = !heteroP2pRequested();
  if (!usePairComm_) {
    // A heterogeneous communicator is initialized collectively. Do it while
    // every process-group rank is constructing the backend instead of lazily
    // from send/recv, where only the two P2P peers may participate.
    initComm();
  }
}
#else
flagcxBackend::flagcxBackend(const c10::intrusive_ptr<::c10d::Store> &store,
                             int rank, int size)
    : Backend(rank, size), store_(store) {
  deviceId_ = 0;
  status_ = 0;
  activeGroupCounter_ = 0;
  C10D_FLAGCX_CHECK(flagcxDeviceHandleInit(&devHandle_), std::nullopt);
  C10D_FLAGCX_CHECK(devHandle_->getDeviceCount(&nDevs_), std::nullopt);
  usePairComm_ = !heteroP2pRequested();
  if (!usePairComm_) {
    // See the extended-API constructor above. The caller must bind its local
    // accelerator before creating the process group.
    initComm();
  }
}
#endif

flagcxBackend::~flagcxBackend() {
  abortPairCoalescing();
  abortHeteroCoalescing();
  {
    std::lock_guard<std::mutex> lock(tensorReaperMutex_);
    stopTensorReaper_ = true;
  }
  tensorReaperCv_.notify_all();
  if (tensorReaperThread_.joinable()) {
    tensorReaperThread_.join();
  }
  // Only this thread accesses either list after the reaper has stopped.
  inflightTensorBatches_.splice(inflightTensorBatches_.end(),
                                reapingTensorBatches_);
  int previousDevice = -1;
  const bool restoreDevice =
      devHandle_->getDevice(&previousDevice) == flagcxSuccess;
  bool retainedUnconfirmedBatch = false;
  // A caller can drop the Work without waiting. Keep tensor storage alive until
  // the device has completed each submitted tensor batch.
  for (auto it = inflightTensorBatches_.begin();
       it != inflightTensorBatches_.end();) {
    auto batch = it++;
    auto result = flagcxUnhandledDeviceError;
    if (!batch->completionUncertain) {
      result = devHandle_->setDevice(batch->deviceId);
      if (result == flagcxSuccess) {
        result = batch->recorded ? devHandle_->eventSynchronize(batch->event)
                                 : devHandle_->streamSynchronize(batch->stream);
      }
    }
    if (result == flagcxSuccess) {
      discardTensorBatch(batch);
    } else {
      // Completion is unknown; do not return the tensors to the allocator.
      retainedUnconfirmedBatch = true;
      static auto *quarantine = new std::list<inflightTensorBatch>();
      quarantine->splice(quarantine->end(), inflightTensorBatches_, batch);
      try {
        TORCH_WARN(
            "Could not confirm completion of a FlagCX tensor batch before "
            "backend destruction; retaining its tensors");
      } catch (...) {
      }
    }
  }
  if (restoreDevice) {
    devHandle_->setDevice(previousDevice);
  }
  if (retainedUnconfirmedBatch) {
    // The retained batches may still be using these resources. Leave them
    // allocated along with their tensors rather than destroy active handles.
#if defined(USE_ASCEND_ADAPTOR) && !defined(FLAGCX_TORCH_BACKEND_FLAGOS)
    // A failed HCCL group can also retain a pointer to an aclStreams_ slot.
    // Moving the map preserves the addresses of its elements.
    static auto *quarantinedStreams = new std::list<decltype(aclStreams_)>();
    quarantinedStreams->emplace_back(std::move(aclStreams_));
#endif
    return;
  }
#if defined(USE_ASCEND_ADAPTOR) && !defined(FLAGCX_TORCH_BACKEND_FLAGOS)
  // On Ascend, flagcxStreams_ holds pointers into aclStreams_. The streams
  // themselves are owned by PyTorch and were never produced by streamCreate(),
  // so destroying them frees memory this backend does not own. This showed up
  // as "free(): invalid pointer" followed by SIGABRT at interpreter shutdown,
  // making every run exit non-zero even after the work had completed.
  flagcxStreams_.clear();
  aclStreams_.clear();
#else
  for (auto &s : flagcxStreams_) {
    devHandle_->streamDestroy(s.second);
  }
#endif
  // Pair communicators can be initialized lazily by send/recv, before the
  // process-group communicator is initialized.
  for (auto &kv : pairComms_) {
    auto ret = flagcxCommDestroy(kv.second);
    if (ret != flagcxSuccess) {
      TORCH_WARN("flagcxCommDestroy failed for pair-comm ", kv.first);
    }
  }
  pairComms_.clear();
  if (status_ == 1) {
    flagcxCommDestroy(comm_);
    status_ = 0;
  }
  if (status_ == 0) {
    flagcxDeviceHandleFree(devHandle_);
  }
}

flagcxStream_t flagcxBackend::getStreamByIndex(int streamId) {
  checkSubmissionHealthy();
#if defined(USE_ASCEND_ADAPTOR) && !defined(FLAGCX_TORCH_BACKEND_FLAGOS)
  // Always resolve PyTorch's *current* NPU stream; never cache it.
  //
  // The previous code captured the current stream on the first call and cached
  // the pointer for the lifetime of the backend. Whenever PyTorch switched
  // streams afterwards - communication/computation overlap, or an explicit
  // side stream - collectives were still issued on the stale stream and had no
  // ordering relationship with the stream that produced or consumed the data.
  //
  // stream(need_empty=true) additionally drains torch_npu's asynchronous task
  // queue before returning the raw ACL stream. With need_empty=false the queue
  // is not drained, so a kernel still sitting in the software queue can reach
  // the stream *after* a collective that was submitted directly to it, and the
  // collective then operates on pre-reduction data.
  //
  // Measured on Ascend 910C (4096 elements, 16 ranks): all_reduce returned a
  // wrong result on 2 of 400 calls before this change, and running the same
  // call from an explicit side stream failed on 59 of 60. HCCL under identical
  // conditions had no failures.
  //
  // Keep one submission stream for every coalesced P2P batch. In particular,
  // the completion event must be recorded on the stream used by the first
  // operation, even if PyTorch changes its current stream before groupEnd().
  if (!coalescedStreamPinned_ ||
      aclStreams_.find(streamId) == aclStreams_.end()) {
    aclStreams_[streamId] = c10_npu::getCurrentNPUStream().stream(true);
  }
  flagcxStreams_[streamId] =
      reinterpret_cast<flagcxStream_t>(&aclStreams_[streamId]);
  return flagcxStreams_[streamId];
#else
  if (auto search = flagcxStreams_.find(streamId);
      search != flagcxStreams_.end()) {
    return search->second;
  } else {
    flagcxStreams_[streamId] = nullptr;
    C10D_FLAGCX_CHECK(devHandle_->streamCreate(&flagcxStreams_[streamId]),
                      std::nullopt);
    return flagcxStreams_[streamId];
  }
#endif
}

std::unique_ptr<flagcxEvent> &flagcxBackend::getEventByIndex(int eventId) {
  if (auto search = flagcxEvents_.find(eventId);
      search != flagcxEvents_.end()) {
    return search->second;
  } else {
#if USE_NVIDIA_ADAPTOR
    flagcxEvents_[eventId] = std::make_unique<flagcxCudaEvent>();
#elif USE_PPU_ADAPTOR
    flagcxEvents_[eventId] = std::make_unique<flagcxPpuEvent>();
#elif USE_ASCEND_ADAPTOR
    flagcxEvents_[eventId] = std::make_unique<flagcxCannEvent>();
#elif USE_ILUVATAR_ADAPTOR
    flagcxEvents_[eventId] = std::make_unique<flagcxIxcudaEvent>();
#elif USE_CAMBRICON_ADAPTOR
    flagcxEvents_[eventId] = std::make_unique<flagcxMluEvent>();
#elif USE_METAX_ADAPTOR
    flagcxEvents_[eventId] = std::make_unique<flagcxMacaEvent>();
#elif USE_MUSA_ADAPTOR
    flagcxEvents_[eventId] = std::make_unique<flagcxMusaEvent>();
#elif USE_DU_ADAPTOR
    flagcxEvents_[eventId] = std::make_unique<flagcxDuEvent>();
#elif USE_KUNLUNXIN_ADAPTOR
    flagcxEvents_[eventId] = std::make_unique<flagcxXpuEvent>();
#elif USE_AMD_ADAPTOR
    flagcxEvents_[eventId] = std::make_unique<flagcxHipEvent>();
#elif USE_TSM_ADAPTOR
    flagcxEvents_[eventId] = std::make_unique<flagcxTxdaEvent>();
#elif USE_ENFLAME_ADAPTOR
    flagcxEvents_[eventId] = std::make_unique<flagcxTopsEvent>();
#elif USE_SUNRISE_ADAPTOR
    flagcxEvents_[eventId] = std::make_unique<flagcxPtpuEvent>();
#endif
    return flagcxEvents_[eventId];
  }
}

void flagcxBackend::initComm(at::Device dev) {
  if (status_ == 0) {
    deviceId_ = dev.index();
    C10D_FLAGCX_CHECK(devHandle_->setDevice(deviceId_), std::nullopt);
    // Get the unique id
    flagcxUniqueId uniqueId;
    if (rank_ == 0) {
      C10D_FLAGCX_CHECK(flagcxGetUniqueId(&uniqueId), std::nullopt);
      auto vec = std::vector<uint8_t>(reinterpret_cast<uint8_t *>(&uniqueId),
                                      reinterpret_cast<uint8_t *>(&uniqueId) +
                                          sizeof(flagcxUniqueId));
      store_->set("flagcx/unique_id", std::string(vec.begin(), vec.end()));
    } else {
      try {
        auto vec = store_->get("flagcx/unique_id");
        TORCH_CHECK_WITH(DistBackendError, vec.size() == sizeof(flagcxUniqueId),
                         "Invalide size for flagcxUniqueId");
        std::memcpy((uint8_t *)&uniqueId, vec.data(), sizeof(flagcxUniqueId));
      } catch (const std::exception &e) {
        throw std::runtime_error(
            "Failed to retrieve the unique id from the store: " +
            std::string(e.what()));
      } catch (...) {
        throw std::runtime_error("Unknown exception during the retrieving of "
                                 "unique id from the store");
      }
    }
    // Initialize the communicator
    C10D_FLAGCX_CHECK(flagcxCommInitRank(&comm_, size_, &uniqueId, rank_),
                      std::nullopt);
    status_ = 1;
  } else {
    if (dev.is_cuda() || dev.is_privateuseone()) {
      if (deviceId_ != dev.index()) {
        throw std::runtime_error(
            "flagcx communicator was initialized with different device");
      }
    }
  }
}

void flagcxBackend::initComm() {
#if defined(USE_NVIDIA_ADAPTOR) || defined(USE_ILUVATAR_ADAPTOR) ||            \
    defined(USE_METAX_ADAPTOR) || defined(USE_DU_ADAPTOR) ||                   \
    defined(USE_KUNLUNXIN_ADAPTOR) || defined(USE_AMD_ADAPTOR)
  initComm(c10::impl::getDeviceGuardImpl(at::DeviceType::CUDA)->getDevice());
#elif USE_PPU_ADAPTOR
  initComm(c10::impl::getDeviceGuardImpl(at::DeviceType::CUDA)->getDevice());
#elif defined(USE_CAMBRICON_ADAPTOR)
  initComm(
      c10::impl::getDeviceGuardImpl(at::DeviceType::PrivateUse1)->getDevice());
#elif defined(USE_ASCEND_ADAPTOR)
  initComm(
      c10::impl::getDeviceGuardImpl(at::DeviceType::PrivateUse1)->getDevice());
#elif defined(USE_MUSA_ADAPTOR)
  initComm(
      c10::impl::getDeviceGuardImpl(at::DeviceType::PrivateUse1)->getDevice());
#elif defined(USE_TSM_ADAPTOR)
  initComm(
      c10::impl::getDeviceGuardImpl(at::DeviceType::PrivateUse1)->getDevice());
#elif defined(USE_ENFLAME_ADAPTOR)
  initComm(
      c10::impl::getDeviceGuardImpl(at::DeviceType::PrivateUse1)->getDevice());
#elif defined(USE_SUNRISE_ADAPTOR)
  initComm(
      c10::impl::getDeviceGuardImpl(at::DeviceType::PrivateUse1)->getDevice());
#endif
}

flagcxComm_t flagcxBackend::getOrCreatePairComm(int peer) {
  std::string key;
  int numRanks;
  int p2pRank;

  if (peer == rank_) {
    // Self-send/recv: dedicated 1-rank comm
    key = "self:" + std::to_string(rank_);
    numRanks = 1;
    p2pRank = 0;
  } else {
    const int low = std::min(rank_, peer);
    const int high = std::max(rank_, peer);
    key = std::to_string(low) + ":" + std::to_string(high);
    numRanks = 2;
    p2pRank = (rank_ < peer) ? 0 : 1;
  }

  if (auto it = pairComms_.find(key); it != pairComms_.end()) {
    return it->second;
  }

  flagcxUniqueId uniqueId{};
  C10D_FLAGCX_CHECK(flagcxGetUniqueId(&uniqueId), std::nullopt);

  if (numRanks == 2) {
    // Exchange uniqueId via store_: lower rank (p2pRank 0) generates and
    // publishes; higher rank (p2pRank 1) reads.
    const std::string storeKey = "flagcx/p2p/" + key + "/uniqueId";
    if (p2pRank == 0) {
      auto vec = std::vector<uint8_t>(reinterpret_cast<uint8_t *>(&uniqueId),
                                      reinterpret_cast<uint8_t *>(&uniqueId) +
                                          sizeof(flagcxUniqueId));
      store_->set(storeKey, std::string(vec.begin(), vec.end()));
    } else {
      auto vec = store_->get(storeKey);
      TORCH_CHECK_WITH(DistBackendError, vec.size() == sizeof(flagcxUniqueId),
                       "Invalid size for flagcxUniqueId on p2p key '", key,
                       "'");
      std::memcpy(reinterpret_cast<uint8_t *>(&uniqueId), vec.data(),
                  sizeof(flagcxUniqueId));
    }
  }
  // numRanks == 1 (self-comm): no store exchange needed, single participant.

  flagcxComm_t pairComm = nullptr;
  C10D_FLAGCX_CHECK(flagcxCommInitRank(&pairComm, numRanks, &uniqueId, p2pRank),
                    std::nullopt);
  pairComms_.emplace(key, pairComm);
  return pairComm;
}

void flagcxBackend::syncStream(at::Device device, int index) {
  auto &event = getEventByIndex(index);
  auto stream = getStreamByIndex(index);
  event->record(device.index());
  event->block(stream, device.index());
}

void flagcxBackend::groupStart() {
  initComm();
  C10D_FLAGCX_CHECK(flagcxGroupStart(comm_), std::nullopt);
  ++activeGroupCounter_;
}

void flagcxBackend::groupEnd() {
  TORCH_CHECK(activeGroupCounter_ > 0, "FlagCX groupEnd without groupStart");
  auto clearGroup =
      c10::make_scope_exit([&]() noexcept { --activeGroupCounter_; });
  flagcxResult_t result;
  try {
    result = flagcxGroupEnd(comm_);
  } catch (...) {
    submissionUnsafe_ = true;
    throw;
  }
  if (result != flagcxSuccess) {
    submissionUnsafe_ = true;
  }
  C10D_FLAGCX_CHECK(result, std::nullopt);
}

void flagcxBackend::abortPairCoalescing() noexcept {
  if (!pairCoalesce_.active) {
    return;
  }
  // Pair operations are deferred until endCoalescing(), so a send/recv failure
  // before that point has not submitted any of these callbacks.
  pairCoalesce_.pendingOps.clear();
  pairCoalesce_.deviceId = -1;
  pairCoalesce_.active = false;
#if defined(USE_ASCEND_ADAPTOR) && !defined(FLAGCX_TORCH_BACKEND_FLAGOS)
  coalescedStreamPinned_ = false;
#endif
}

void flagcxBackend::abortHeteroCoalescing() noexcept {
  if (!heteroCoalesce_.active) {
    return;
  }
  if (heteroCoalesce_.groupOpen) {
    // Even a send/recv validation failure must close the FlagCX group. Earlier
    // operations may already have been queued inside it.
    auto result = flagcxInternalError;
    try {
      result = flagcxGroupEnd(comm_);
    } catch (...) {
      // Preserve the exception that triggered this cleanup.
    }
    --activeGroupCounter_;
    heteroCoalesce_.groupOpen = false;
    if (result != flagcxSuccess) {
      submissionUnsafe_ = true;
    }
  }
  if (heteroCoalesce_.batch) {
    finishTensorBatchOnError(*heteroCoalesce_.batch,
                             heteroCoalesce_.mayHaveSubmitted);
    heteroCoalesce_.batch.reset();
  }
  heteroCoalesce_.active = false;
  heteroCoalesce_.mayHaveSubmitted = false;
  heteroCoalesce_.deviceId = -1;
#if defined(USE_ASCEND_ADAPTOR) && !defined(FLAGCX_TORCH_BACKEND_FLAGOS)
  coalescedStreamPinned_ = false;
#endif
}

void flagcxBackend::checkSubmissionHealthy() {
  TORCH_CHECK(!submissionUnsafe_,
              "A previous FlagCX batch failed after submission; this "
              "backend cannot safely submit more work");
}

void flagcxBackend::runTensorReaper() noexcept {
  int previousDevice = -1;
  const bool restoreDevice =
      devHandle_->getDevice(&previousDevice) == flagcxSuccess;
  std::unique_lock<std::mutex> lock(tensorReaperMutex_);
  while (!stopTensorReaper_) {
    tensorReaperCv_.wait_for(lock, std::chrono::milliseconds(10), [&] {
      return stopTensorReaper_ || !reapingTensorBatches_.empty();
    });
    if (stopTensorReaper_) {
      break;
    }
    for (auto it = reapingTensorBatches_.begin();
         it != reapingTensorBatches_.end();) {
      if (it->completionUncertain) {
        ++it;
        continue;
      }
      flagcxResult_t result = devHandle_->setDevice(it->deviceId);
      if (result == flagcxSuccess) {
        result = devHandle_->eventQuery(it->event);
      }
      if (result == flagcxSuccess) {
        auto completed = it++;
        // The event has completed. Move the batch out before releasing its
        // tensors, since allocator cleanup may call back into device code.
        std::list<inflightTensorBatch> ready;
        ready.splice(ready.end(), reapingTensorBatches_, completed);
        lock.unlock();
        auto &batch = ready.front();
        if (batch.event != nullptr) {
          devHandle_->eventDestroy(batch.event);
        }
        if (batch.ownsStreamWrapper && batch.stream != nullptr) {
          devHandle_->streamFree(batch.stream);
        }
        ready.clear();
        lock.lock();
      } else if (result == flagcxInProgress) {
        ++it;
      } else {
        it->completionUncertain = true;
        submissionUnsafe_ = true;
        ++it;
      }
    }
    // Do not spin when an event is pending. The next poll happens shortly
    // even if the caller never performs another P2P operation.
    if (!stopTensorReaper_) {
      tensorReaperCv_.wait_for(lock, std::chrono::milliseconds(10));
    }
  }
  lock.unlock();
  if (restoreDevice) {
    devHandle_->setDevice(previousDevice);
  }
}

flagcxBackend::inflightTensorIter
flagcxBackend::prepareTensorBatch(int deviceId, flagcxStream_t stream) {
  checkSubmissionHealthy();
  inflightTensorIter batch;
  {
    std::lock_guard<std::mutex> lock(tensorReaperMutex_);
    batch = inflightTensorBatches_.emplace(inflightTensorBatches_.end());
  }
  batch->deviceId = deviceId;
  batch->stream = stream;
  try {
#if defined(USE_ASCEND_ADAPTOR) && !defined(FLAGCX_TORCH_BACKEND_FLAGOS)
    // The original wrapper points into a mutable aclStreams_ slot. Keep an
    // independent wrapper for completion tracking after the slot is unpinned.
    auto nativeStream = *reinterpret_cast<aclrtStream *>(stream);
    C10D_FLAGCX_CHECK(
        devHandle_->streamCopy(&batch->stream,
                               reinterpret_cast<void *>(nativeStream)),
        std::nullopt);
    batch->ownsStreamWrapper = true;
#endif
    C10D_FLAGCX_CHECK(
        devHandle_->eventCreate(&batch->event, flagcxEventDisableTiming),
        std::nullopt);
  } catch (...) {
    discardTensorBatch(batch);
    throw;
  }
  return batch;
}

void flagcxBackend::discardTensorBatch(inflightTensorIter batch) noexcept {
  std::lock_guard<std::mutex> lock(tensorReaperMutex_);
  if (batch->event != nullptr) {
    devHandle_->eventDestroy(batch->event);
  }
  if (batch->ownsStreamWrapper && batch->stream != nullptr) {
    devHandle_->streamFree(batch->stream);
  }
  inflightTensorBatches_.erase(batch);
}

void flagcxBackend::finishTensorBatch(inflightTensorIter batch) {
  const auto result = devHandle_->eventRecord(batch->event, batch->stream);
  if (result == flagcxSuccess) {
    batch->recorded = true;
    try {
      std::lock_guard<std::mutex> lock(tensorReaperMutex_);
      if (!tensorReaperThread_.joinable()) {
        tensorReaperThread_ = std::thread([this] { runTensorReaper(); });
      }
      reapingTensorBatches_.splice(reapingTensorBatches_.end(),
                                   inflightTensorBatches_, batch);
    } catch (...) {
      // The event is still valid; the destructor can synchronize it. Prevent
      // another operation from depending on an unserviced retention entry.
      submissionUnsafe_ = true;
      throw;
    }
    tensorReaperCv_.notify_one();
    return;
  }
  if (devHandle_->streamSynchronize(batch->stream) == flagcxSuccess) {
    discardTensorBatch(batch);
  } else {
    submissionUnsafe_ = true;
  }
  C10D_FLAGCX_CHECK(result, std::nullopt);
}

void flagcxBackend::finishTensorBatchOnError(inflightTensorIter batch,
                                             bool mayHaveSubmitted) noexcept {
  if (!mayHaveSubmitted) {
    discardTensorBatch(batch);
    return;
  }
  // The failed call may leave tasks queued outside the submission stream.
  // A stream event cannot prove those tasks will never access the tensor.
  batch->completionUncertain = true;
  submissionUnsafe_ = true;
}

flagcxBackend::retainedTensorBatch::retainedTensorBatch(flagcxBackend &backend,
                                                        int deviceId,
                                                        flagcxStream_t stream)
    : backend_(backend), batch_(backend.prepareTensorBatch(deviceId, stream)) {}

flagcxBackend::retainedTensorBatch::~retainedTensorBatch() noexcept {
  if (!finished_) {
    backend_.finishTensorBatchOnError(batch_, mayHaveSubmitted_);
  }
}

void flagcxBackend::retainedTensorBatch::add(const at::Tensor &tensor) {
  batch_->storages.push_back(tensor.storage());
}

void flagcxBackend::retainedTensorBatch::add(
    const std::vector<at::Tensor> &tensors) {
  for (const auto &tensor : tensors) {
    add(tensor);
  }
}

void flagcxBackend::retainedTensorBatch::finish() {
  finished_ = true;
  backend_.finishTensorBatch(batch_);
}

void flagcxBackend::startCoalescing() {
  TORCH_CHECK(!pairCoalesce_.active && !heteroCoalesce_.active,
              "Nested coalescing is not supported for FlagCX P2P operations");
  checkSubmissionHealthy();
  if (usePairComm_) {
    // Pair communicators are initialized lazily by the participating peers.
    // Initializing comm_ here would reintroduce the subgroup P2P deadlock that
    // pair communicators are intended to avoid.
    pairCoalesce_.pendingOps.clear();
    pairCoalesce_.deviceId = -1;
    pairCoalesce_.active = true;
  } else {
    TORCH_CHECK(status_ == 1,
                "Heterogeneous P2P communicator was not eagerly initialized");
    groupStart();
    heteroCoalesce_.active = true;
    heteroCoalesce_.groupOpen = true;
    heteroCoalesce_.mayHaveSubmitted = false;
    heteroCoalesce_.deviceId = -1;
    heteroCoalesce_.batch.reset();
  }
#if defined(USE_ASCEND_ADAPTOR) && !defined(FLAGCX_TORCH_BACKEND_FLAGOS)
  // Both pair and heterogeneous groups must submit on the same NPU stream
  // from their first operation through completion-event recording.
  try {
    aclStreams_[0] = c10_npu::getCurrentNPUStream().stream(true);
    coalescedStreamPinned_ = true;
  } catch (...) {
    if (usePairComm_) {
      abortPairCoalescing();
    } else {
      abortHeteroCoalescing();
    }
    throw;
  }
#endif
}

c10::intrusive_ptr<Work> flagcxBackend::endCoalescing() {
  if (usePairComm_) {
    TORCH_CHECK(pairCoalesce_.active,
                "endCoalescing called without matching startCoalescing");
  } else {
    TORCH_CHECK(heteroCoalesce_.active,
                "endCoalescing called without matching startCoalescing");
  }
  // Keep the Ascend vendor stream pinned through the Work event recording.
  // Reset batch state even when submission or Work setup throws.
  auto cleanup = c10::make_scope_exit([&]() noexcept {
    if (usePairComm_) {
      abortPairCoalescing();
    } else {
      abortHeteroCoalescing();
    }
  });

  if (usePairComm_) {
    // Sort by peer ascending: canonical (min,max) order avoids deadlock.
    std::stable_sort(
        pairCoalesce_.pendingOps.begin(), pairCoalesce_.pendingOps.end(),
        [](const auto &a, const auto &b) { return a.peer < b.peer; });
    // Submit the pending operations of each peer inside one group.
    //
    // Without a group the HCCL adaptor takes its groupDepth == 0 path and calls
    // the blocking HcclSend/HcclRecv. batch_isend_irecv posts a send before a
    // recv on both sides of a pair, so both ranks block in their send waiting
    // for the peer to post the matching recv, and the call never returns. This
    // makes any pipeline-parallel run (PP > 1) hang on its first step, because
    // Megatron's send_forward_recv_backward is built on batch_isend_irecv.
    //
    // Inside a group the adaptor accumulates the operations and submits them as
    // a single HcclBatchSendRecv, which performs the exchange in both
    // directions at once. The non-pair-communicator branch of this function
    // already wraps its operations in groupStart()/groupEnd().
    //
    // pendingOps is already sorted by peer, so group each run of equal peers.
    if (!pairCoalesce_.pendingOps.empty()) {
      deviceId_ = pairCoalesce_.deviceId;
      C10D_FLAGCX_CHECK(devHandle_->setDevice(deviceId_), std::nullopt);
      auto stream = getStreamByIndex(0);
      auto batch = prepareTensorBatch(pairCoalesce_.deviceId, stream);
      bool mayHaveSubmitted = false;
      bool finalized = false;
      auto finishOnError = c10::make_scope_exit([&]() noexcept {
        if (!finalized) {
          finishTensorBatchOnError(batch, mayHaveSubmitted);
        }
      });
      batch->storages.reserve(pairCoalesce_.pendingOps.size());
      for (const auto &op : pairCoalesce_.pendingOps) {
        batch->storages.push_back(op.tensor.storage());
      }
      size_t i = 0;
      const size_t n = pairCoalesce_.pendingOps.size();
      while (i < n) {
        const int peer = pairCoalesce_.pendingOps[i].peer;
        size_t j = i;
        while (j < n && pairCoalesce_.pendingOps[j].peer == peer) {
          ++j;
        }
        auto pairComm = getOrCreatePairComm(peer);
        flagcxGroupGuard group(pairComm);
        mayHaveSubmitted = true;
        for (size_t k = i; k < j; ++k) {
          pairCoalesce_.pendingOps[k].submit();
        }
        C10D_FLAGCX_CHECK(group.close(), std::nullopt);
        i = j;
      }
      finalized = true;
      finishTensorBatch(batch);
    }
  } else {
    // groupEnd may submit the entire heterogeneous batch. Keep the retained
    // tensors until an event on the pinned submission stream completes.
    heteroCoalesce_.groupOpen = false;
    groupEnd();
    if (heteroCoalesce_.batch) {
      auto batch = *heteroCoalesce_.batch;
      heteroCoalesce_.batch.reset();
      finishTensorBatch(batch);
    }
  }

  auto stream = getStreamByIndex(0);
  auto work =
      c10::make_intrusive<flagcxWork>(OpType::COALESCED, stream, devHandle_);
  work->event_->record(stream, deviceId_);
  work->deviceId_ = deviceId_;
  if (!usePairComm_) {
    // This path uses the heterogeneous P2P runner even when all ranks happen
    // to use the same vendor (for example, FLAGCX_USE_HETERO_COMM=1).
    work->isBarrierOp_ = true;
  }
  work->future_ = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()));
  work->future_->markCompleted(c10::IValue(0));
  return work;
}

template <typename Fn>
c10::intrusive_ptr<Work>
flagcxBackend::collectiveCoalesced(std::vector<at::Tensor> &inputs,
                                   std::vector<at::Tensor> &outputs, Fn fn,
                                   OpType opType) {
  // Currently, the API permits one scenario where inputs.size() and
  // outputs.size() are > 0.
  // 1. If the call was a _coalesced call, all inputs must be on the same
  // device.
  //    The group of flagcx calls applies the collective separately to each
  //    input, but the group as a whole should be efficient.
  auto device = inputs[0].device();
  initComm(device);

  // TODO: keep track of the coalesced state at backend side.

  // First let default flagcx stream wait for input tensor allocation stream
  syncStream(device);
  auto collectiveStream = getStreamByIndex(0);
  retainedTensorBatch retained(*this, device.index(), collectiveStream);
  retained.add(inputs);
  retained.add(outputs);
  auto work =
      c10::make_intrusive<flagcxWork>(opType, collectiveStream, devHandle_);

  {
    // isHomo decides whether the batch is grouped at all, and
    // flagcxIsHomoComm() leaves it untouched when it fails, so take its
    // result rather than branching on an uninitialized value.
    int isHomo = 0;
    C10D_FLAGCX_CHECK(flagcxIsHomoComm(comm_, &isHomo), std::nullopt);
    // The guard has to cover the loop below, not just this if-statement.
    //
    // flagcxGroupGuard is RAII - flagcxGroupStart() in the constructor,
    // flagcxGroupEnd() in the destructor. Declared as
    // `if (isHomo) { flagcxGroupGuard guard(comm_); }` it opened an empty
    // group and closed it on the very next line, so every collective issued
    // in the loop below ran outside any group and the batching that the
    // *_coalesced entry points exist for was lost. That is the path the
    // distributed optimizer takes for reduce_scatter_tensor_coalesced and
    // allgather_into_tensor_coalesced.
    std::optional<flagcxGroupGuard> guard;
    if (isHomo) {
      guard.emplace(comm_);
    }
    // multi-stream may lead to queue sync error on mlu,
    // more tests are required to confirm,
    // so we disable multi-stream support for now
    // flagcxStream_t stream;
    flagcxStream_t stream = getStreamByIndex(0);

    retained.markSubmitted();

    for (const auto i : c10::irange(inputs.size())) {
      // if (isHomo) {
      //   stream = getStreamByIndex(0);
      // } else {
      //   stream = getStreamByIndex(i + 1);
      // }
      // The retained batch above keeps both storage allocations alive until
      // its completion event has finished.
      auto inputTensor = inputs[i];
      auto outputTensor = outputs[i];
      // Perform the collective operation
      C10D_FLAGCX_CHECK(fn(inputTensor, outputTensor, comm_, stream),
                        std::nullopt);

      // if (!isHomo) {
      //   auto &event = getEventByIndex(i + 1);
      //   event->record(stream, deviceId_);
      // }
    }
    // for (const auto i : c10::irange(inputs.size())) {
    //   if (!isHomo) {
    //     auto &event = getEventByIndex(i + 1);
    //     event->block(getStreamByIndex(0), deviceId_);
    //   }
    // }

    // End the group here instead of leaving it to the guard's destructor.
    // Now that the collectives above are issued inside the group,
    // flagcxGroupEnd() is what submits them, and the Work created below
    // reports the batch as complete - so a failed submission has to surface
    // before that. The destructor still ends the group if the loop above
    // throws; it warns there rather than throwing, because it can run while
    // another exception is in flight.
    if (guard.has_value()) {
      C10D_FLAGCX_CHECK(guard->close(), std::nullopt);
    }
  }

  work->event_->record(getStreamByIndex(0), deviceId_);
  retained.finish();
  work->deviceId_ = deviceId_;
  work->isBarrierOp_ = false;
  // Create a future to track the coalesced operation
  std::vector<at::Device> devices{inputs[0].device()};
  work->future_ = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), devices);
  work->future_->markCompleted(c10::IValue(outputs[0]));
  return work;
}

c10::intrusive_ptr<Work>
flagcxBackend::allgather(std::vector<std::vector<at::Tensor>> &outputTensors,
                         std::vector<at::Tensor> &inputTensors,
                         const AllgatherOptions & /* unused */) {
  auto inputTensor = inputTensors.back();
  auto outputTensorsTmp = outputTensors.back();
  auto device = inputTensor.device();
  auto flagcxDataType = getFlagcxDataType(inputTensor.scalar_type());
  auto stream = getStreamByIndex(0);
  check_device(inputTensor.device(), outputTensorsTmp[0].device());
  initComm(device);
  syncStream(device);
  retainedTensorBatch retained(*this, device.index(), stream);
  retained.add(inputTensor);
  retained.add(outputTensorsTmp);

  if (!check_same_size(outputTensorsTmp)) {
    // Implement allgather with different sizes using broadcast
    retained.markSubmitted();
    const auto num_reduces = outputTensorsTmp.size();
    for (const int64_t i : c10::irange(static_cast<int64_t>(num_reduces))) {
      auto &output = outputTensorsTmp[i];
      auto &input = (i == rank_) ? inputTensor : output;
      // Perform out-of-place broadcast from rank i
      C10D_FLAGCX_CHECK(flagcxBroadcast(input.data_ptr(), output.data_ptr(),
                                        input.numel(), flagcxDataType, i, comm_,
                                        stream),
                        std::nullopt);
    }
  } else {
    // Flatten a vector of tensors into a single, stacked tensor.
    at::Tensor outputFlattened =
        newLikeFlatOnStream(outputTensorsTmp, stream, device.index());
    retained.add(outputFlattened);
    retained.markSubmitted();

#if (defined(USE_NVIDIA_ADAPTOR) || defined(USE_METAX_ADAPTOR)) &&             \
    defined(TORCH_VER_GE_250)
    if (needRecording()) {
      recordTuneObject(flagcxCommOpAllGather, flagcxDataType,
                       inputTensor.numel());
    }

#endif
    // Perform the allgather operation
    C10D_FLAGCX_CHECK(
        flagcxAllGather(inputTensor.data_ptr(), outputFlattened.data_ptr(),
                        inputTensor.numel(), flagcxDataType, comm_, stream),
        std::nullopt);

    // Copy the flattened tensor back into a vector of tensors.
    {
      flagcxStreamGuard guard(stream, device.index());
      for (const auto j : c10::irange(outputTensorsTmp.size())) {
        copyTensorOnStream(outputTensorsTmp[j], outputFlattened[j], stream,
                           devHandle_, device.index());
      }
    }
  }

  auto work =
      c10::make_intrusive<flagcxWork>(OpType::ALLGATHER, stream, devHandle_);
  work->event_->record(stream, deviceId_);
  retained.finish();
  work->deviceId_ = deviceId_;
  // Create a future to track the allgather operation
  std::vector<at::Device> devices{inputTensor.device()};
  work->future_ = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), devices);
  work->future_->markCompleted(c10::IValue(outputTensorsTmp));
  return work;
}

c10::intrusive_ptr<Work>
flagcxBackend::_allgather_base(at::Tensor &outputTensor,
                               at::Tensor &inputTensor,
                               const AllgatherOptions & /* unused */) {
  auto flagcxDataType = getFlagcxDataType(inputTensor.scalar_type());
  auto stream = getStreamByIndex(0);
  auto work = c10::make_intrusive<flagcxWork>(OpType::_ALLGATHER_BASE, stream,
                                              devHandle_);
  check_device(inputTensor.device(), outputTensor.device());
  initComm(inputTensor.device());
  syncStream(inputTensor.device());
  retainedTensorBatch retained(*this, inputTensor.device().index(), stream);
  retained.add(inputTensor);
  retained.add(outputTensor);

#if (defined(USE_NVIDIA_ADAPTOR) || defined(USE_METAX_ADAPTOR)) &&             \
    defined(TORCH_VER_GE_250)
  if (needRecording()) {
    recordTuneObject(flagcxCommOpAllGather, flagcxDataType,
                     inputTensor.numel());
  }

#endif

  // Perform the allgather operation
  retained.markSubmitted();
  C10D_FLAGCX_CHECK(
      flagcxAllGather(inputTensor.data_ptr(), outputTensor.data_ptr(),
                      inputTensor.numel(), flagcxDataType, comm_, stream),
      std::nullopt);

  work->event_->record(stream, deviceId_);
  retained.finish();
  work->deviceId_ = deviceId_;
  // Create a future to track the allgather operation
  std::vector<at::Device> devices{inputTensor.device()};
  work->future_ = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), devices);
  work->future_->markCompleted(c10::IValue(outputTensor));
  return work;
}

c10::intrusive_ptr<Work>
flagcxBackend::allgather_into_tensor_coalesced(std::vector<at::Tensor> &outputs,
                                               std::vector<at::Tensor> &inputs,
                                               const AllgatherOptions &opts) {
  // parameter validation
  check_gpu_tensors_same_device(inputs);

  return collectiveCoalesced(
      inputs, outputs,
      [&](at::Tensor &input, at::Tensor &output, flagcxComm_t comm,
          flagcxStream_t stream) {
        auto flagcxDataType = getFlagcxDataType(input.scalar_type());
#if (defined(USE_NVIDIA_ADAPTOR) || defined(USE_METAX_ADAPTOR)) &&             \
    defined(TORCH_VER_GE_250)
        if (options_->enableTuner && !recordingEnded) {
          recordTuneObject(flagcxCommOpAllGather, flagcxDataType,
                           input.numel());
        }

#endif
        return flagcxAllGather(input.data_ptr(), output.data_ptr(),
                               input.numel(), flagcxDataType, comm, stream);
      },
      OpType::COALESCED);
}

c10::intrusive_ptr<Work>
flagcxBackend::allreduce(std::vector<at::Tensor> &tensors,
                         const AllreduceOptions &opts) {
  auto &tensor = tensors.back();
  auto flagcxDataType = getFlagcxDataType(tensor.scalar_type());
  auto flagcxReduceOp =
      getFlagcxReduceOp(opts.reduceOp, tensor, flagcxDataType);
  auto stream = getStreamByIndex(0);
  auto work =
      c10::make_intrusive<flagcxWork>(OpType::ALLREDUCE, stream, devHandle_);
  initComm(tensor.device());
  syncStream(tensor.device());
  retainedTensorBatch retained(*this, tensor.device().index(), stream);
  retained.add(tensor);

#if (defined(USE_NVIDIA_ADAPTOR) || defined(USE_METAX_ADAPTOR)) &&             \
    defined(TORCH_VER_GE_250)
  if (needRecording()) {
    recordTuneObject(flagcxCommOpAllReduce, flagcxDataType, tensor.numel());
  }

#endif

  // Perform the allreduce operation
  retained.markSubmitted();
  C10D_FLAGCX_CHECK(flagcxAllReduce(tensor.data_ptr(), tensor.data_ptr(),
                                    tensor.numel(), flagcxDataType,
                                    flagcxReduceOp, comm_, stream),
                    std::nullopt);

  work->event_->record(stream, deviceId_);
  retained.finish();
  work->deviceId_ = deviceId_;
  // Create a future to track the allreduce operation
  std::vector<at::Device> devices{tensor.device()};
  work->future_ = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), devices);
  work->future_->markCompleted(c10::IValue(tensors));
  return work;
}

c10::intrusive_ptr<Work>
flagcxBackend::allreduce_coalesced(std::vector<at::Tensor> &tensors,
                                   const AllreduceCoalescedOptions &opts) {
  // parameter validation
  check_gpu_tensors_same_device(tensors);
  TORCH_CHECK(
      !isFloat8Type(tensors.back().scalar_type()),
      "Float8 dtypes are not currenlty supported for FlagCX reductions");

  return collectiveCoalesced(
      tensors, tensors,
      [&](at::Tensor &input, at::Tensor &output, flagcxComm_t comm,
          flagcxStream_t stream) {
        auto flagcxDataType = getFlagcxDataType(input.scalar_type());
        auto flagcxReduceOp =
            getFlagcxReduceOp(opts.reduceOp, input, flagcxDataType);
#if (defined(USE_NVIDIA_ADAPTOR) || defined(USE_METAX_ADAPTOR)) &&             \
    defined(TORCH_VER_GE_250)
        if (needRecording()) {
          recordTuneObject(flagcxCommOpAllReduce, flagcxDataType,
                           input.numel());
        }

#endif
        return flagcxAllReduce(input.data_ptr(), output.data_ptr(),
                               input.numel(), flagcxDataType, flagcxReduceOp,
                               comm, stream);
      },
      OpType::COALESCED);
}

c10::intrusive_ptr<Work>
flagcxBackend::alltoall(std::vector<at::Tensor> &outputTensors,
                        std::vector<at::Tensor> &inputTensors,
                        const AllToAllOptions & /* unused */) {
  TORCH_CHECK(inputTensors.size() == outputTensors.size(),
              "Number of input and output tensors must be equal");
  TORCH_CHECK(check_same_size(inputTensors) && check_same_size(outputTensors),
              "All input and output tensors must be the same size");

  auto count = outputTensors[0].numel();
  auto device = outputTensors[0].device();
  auto flagcxDataType = getFlagcxDataType(outputTensors[0].scalar_type());
  auto stream = getStreamByIndex(0);
  auto work =
      c10::make_intrusive<flagcxWork>(OpType::ALLTOALL, stream, devHandle_);

  for (const auto i : c10::irange(outputTensors.size())) {
    TORCH_CHECK(inputTensors[i].numel() == outputTensors[i].numel(),
                "Tensors must have the same number of elements");
    TORCH_CHECK(device == outputTensors[i].device() &&
                    device == inputTensors[i].device(),
                "Tensors must be on the same device");
    TORCH_CHECK(
        flagcxDataType == getFlagcxDataType(outputTensors[0].scalar_type()) &&
            flagcxDataType == getFlagcxDataType(inputTensors[0].scalar_type()),
        "Tensors must have the same data type");
  }

  initComm(device);
  syncStream(device);
  retainedTensorBatch retained(*this, device.index(), stream);
  retained.add(inputTensors);
  retained.add(outputTensors);

  // Flatten a vector of tensors into a single, stacked tensor.
  at::Tensor inputFlattened =
      newLikeFlatOnStream(inputTensors, stream, device.index());
  at::Tensor outputFlattened =
      newLikeFlatOnStream(outputTensors, stream, device.index());
  retained.add(inputFlattened);
  retained.add(outputFlattened);
  retained.markSubmitted();

  // Copy the input tensors to the flattened tensor.
  {
    flagcxStreamGuard guard(stream, device.index());
    for (const auto j : c10::irange(inputTensors.size())) {
      copyTensorOnStream(inputFlattened[j], inputTensors[j], stream, devHandle_,
                         device.index());
    }
  }

#if (defined(USE_NVIDIA_ADAPTOR) || defined(USE_METAX_ADAPTOR)) &&             \
    defined(TORCH_VER_GE_250)
  if (needRecording()) {
    recordTuneObject(flagcxCommOpAlltoAll, flagcxDataType, count);
  }

#endif

  // Perform the alltoall operation
  C10D_FLAGCX_CHECK(flagcxAlltoAll(inputFlattened.data_ptr(),
                                   outputFlattened.data_ptr(), count,
                                   flagcxDataType, comm_, stream),
                    std::nullopt);

  // Copy the flattened tensor back into a vector of tensors.
  {
    flagcxStreamGuard guard(stream, device.index());
    for (const auto j : c10::irange(outputTensors.size())) {
      copyTensorOnStream(outputTensors[j], outputFlattened[j], stream,
                         devHandle_, device.index());
    }
  }

  work->event_->record(stream, deviceId_);
  retained.finish();
  work->deviceId_ = deviceId_;
  // Create a future to track the alltoall operation
  std::vector<at::Device> devices{device};
  work->future_ = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), devices);
  work->future_->markCompleted(c10::IValue(outputTensors));
  return work;
}

c10::intrusive_ptr<Work>
flagcxBackend::alltoall_base(at::Tensor &outputTensor, at::Tensor &inputTensor,
                             std::vector<int64_t> &outputSplitSizes,
                             std::vector<int64_t> &inputSplitSizes,
                             const AllToAllOptions & /* unused */) {
  auto count = outputTensor.numel() / size_;
  auto device = outputTensor.device();
  auto flagcxDataType = getFlagcxDataType(outputTensor.scalar_type());
  auto stream = getStreamByIndex(0);
  auto work = c10::make_intrusive<flagcxWork>(OpType::ALLTOALL_BASE, stream,
                                              devHandle_);

  TORCH_CHECK(device == outputTensor.device() && device == inputTensor.device(),
              "Tensor must be on the same device");
  TORCH_CHECK(flagcxDataType == getFlagcxDataType(outputTensor.scalar_type()) &&
                  flagcxDataType ==
                      getFlagcxDataType(inputTensor.scalar_type()),
              "Tensor must have the same data type");

  bool isEqualSize = (outputSplitSizes.empty() && inputSplitSizes.empty());

  std::vector<size_t> inLengths(size_);
  std::vector<size_t> outLengths(size_);
  std::vector<size_t> inOffsets(size_);
  std::vector<size_t> outOffsets(size_);

  if (!isEqualSize) {
    c10d::computeLengthsAndOffsets(inputSplitSizes, inputTensor, &inLengths,
                                   &inOffsets);
    c10d::computeLengthsAndOffsets(outputSplitSizes, outputTensor, &outLengths,
                                   &outOffsets);
  }

  initComm(device);
  syncStream(device);
  retainedTensorBatch retained(*this, device.index(), stream);
  retained.add(inputTensor);
  retained.add(outputTensor);

  if (isEqualSize) {
#if (defined(USE_NVIDIA_ADAPTOR) || defined(USE_METAX_ADAPTOR)) &&             \
    defined(TORCH_VER_GE_250)
    if (needRecording()) {
      recordTuneObject(flagcxCommOpAlltoAll, flagcxDataType, count);
    }

#endif
    // Perform the alltoall operation
    retained.markSubmitted();
    C10D_FLAGCX_CHECK(flagcxAlltoAll(inputTensor.data_ptr(),
                                     outputTensor.data_ptr(), count,
                                     flagcxDataType, comm_, stream),
                      std::nullopt);
  } else {
    // currently, we do not support recording alltoallv operations for
    // flagcxTuner Perform the alltoallv operation
    retained.markSubmitted();
    C10D_FLAGCX_CHECK(flagcxAlltoAllv(inputTensor.data_ptr(), inLengths.data(),
                                      inOffsets.data(), outputTensor.data_ptr(),
                                      outLengths.data(), outOffsets.data(),
                                      flagcxDataType, comm_, stream),
                      std::nullopt);
  }

  work->event_->record(stream, deviceId_);
  retained.finish();
  work->deviceId_ = deviceId_;
  // Create a future to track the alltoall operation
  std::vector<at::Device> devices{device};
  work->future_ = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), devices);
  work->future_->markCompleted(c10::IValue(outputTensor));
  return work;
}

c10::intrusive_ptr<Work> flagcxBackend::barrier(const BarrierOptions &opts) {
  initComm();
  auto stream = getStreamByIndex(0);
  auto work =
      c10::make_intrusive<flagcxWork>(OpType::BARRIER, stream, devHandle_);

  C10D_FLAGCX_CHECK(flagcxBarrier(comm_, stream), std::nullopt);

  work->event_->record(stream, deviceId_);
  work->deviceId_ = deviceId_;
  work->isBarrierOp_ = true;
  // Create a future to track the barrier operation
  work->future_ = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()));
  work->future_->markCompleted(c10::IValue(0));
  return work;
}

c10::intrusive_ptr<Work>
flagcxBackend::broadcast(std::vector<at::Tensor> &tensors,
                         const BroadcastOptions &opts) {
  auto &tensor = tensors.back();
  auto flagcxDataType = getFlagcxDataType(tensor.scalar_type());
  auto stream = getStreamByIndex(0);
  auto work =
      c10::make_intrusive<flagcxWork>(OpType::BROADCAST, stream, devHandle_);
  initComm(tensor.device());
  syncStream(tensor.device());
  retainedTensorBatch retained(*this, tensor.device().index(), stream);
  retained.add(tensor);

  const auto root = opts.rootRank + opts.rootTensor;
#if (defined(USE_NVIDIA_ADAPTOR) || defined(USE_METAX_ADAPTOR)) &&             \
    defined(TORCH_VER_GE_250)
  if (needRecording()) {
    recordTuneObject(flagcxCommOpBroadcast, flagcxDataType, tensor.numel());
  }

#endif
  retained.markSubmitted();
  C10D_FLAGCX_CHECK(flagcxBroadcast(tensor.data_ptr(), tensor.data_ptr(),
                                    tensor.numel(), flagcxDataType, root, comm_,
                                    stream),
                    std::nullopt);

  work->event_->record(stream, deviceId_);
  retained.finish();
  work->deviceId_ = deviceId_;
  // Create a future to track the broadcast operation
  std::vector<at::Device> devices{tensor.device()};
  work->future_ = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), devices);
  work->future_->markCompleted(c10::IValue(tensors));
  return work;
}

c10::intrusive_ptr<Work>
flagcxBackend::gather(std::vector<std::vector<at::Tensor>> &outputTensors,
                      std::vector<at::Tensor> &inputTensors,
                      const GatherOptions &opts) {
  auto &inputTensor = inputTensors.back();
  auto device = inputTensor.device();
  auto flagcxDataType = getFlagcxDataType(inputTensor.scalar_type());
  auto stream = getStreamByIndex(0);
  auto work =
      c10::make_intrusive<flagcxWork>(OpType::GATHER, stream, devHandle_);
  initComm(device);
  syncStream(device);
  retainedTensorBatch retained(*this, device.index(), stream);
  retained.add(inputTensor);

  auto root = opts.rootRank;
  std::vector<at::Tensor> outputTensorsTmp;
  if (rank_ == root) {
    outputTensorsTmp = outputTensors.back();
  } else {
    outputTensorsTmp = {};
    outputTensorsTmp.emplace_back(
        at::ones({1}, at::TensorOptions().device(inputTensor.device())));
  }
  retained.add(outputTensorsTmp);

  // Flatten a vector of tensors into a single, stacked tensor.
#if defined(FLAGCX_TORCH_BACKEND_FLAGOS) && defined(USE_ENFLAME_ADAPTOR)
  auto outputTemplates = outputTensorsTmp;
  if (rank_ != root) {
    outputTemplates.resize(size_, inputTensor);
  }
  at::Tensor outputFlattened =
      newLikeFlatOnStream(outputTemplates, stream, device.index());
#else
  at::Tensor outputFlattened =
      newLikeFlatOnStream(outputTensorsTmp, stream, device.index());
#endif
  retained.add(outputFlattened);

#if (defined(USE_NVIDIA_ADAPTOR) || defined(USE_METAX_ADAPTOR)) &&             \
    defined(TORCH_VER_GE_250)
  if (needRecording()) {
    recordTuneObject(flagcxCommOpGather, flagcxDataType, inputTensor.numel());
  }

#endif
  // Perform the gather operation. The tested ECCL runtime does not populate
  // the root receive buffer for gather, so use its working all-gather primitive
  // and expose the result only on the root, preserving PyTorch semantics.
#if defined(FLAGCX_TORCH_BACKEND_FLAGOS) && defined(USE_ENFLAME_ADAPTOR)
  retained.markSubmitted();
  C10D_FLAGCX_CHECK(
      flagcxAllGather(inputTensor.data_ptr(), outputFlattened.data_ptr(),
                      inputTensor.numel(), flagcxDataType, comm_, stream),
      std::nullopt);
#else
  retained.markSubmitted();
  C10D_FLAGCX_CHECK(
      flagcxGather(inputTensor.data_ptr(), outputFlattened.data_ptr(),
                   inputTensor.numel(), flagcxDataType, root, comm_, stream),
      std::nullopt);
#endif

  // Unflatten the flattened tensor back into a vector of tensors.
  if (rank_ == root) {
    flagcxStreamGuard guard(stream, device.index());
    for (const auto j : c10::irange(outputTensorsTmp.size())) {
      copyTensorOnStream(outputTensorsTmp[j], outputFlattened[j], stream,
                         devHandle_, device.index());
    }
  }

  work->event_->record(stream, deviceId_);
  retained.finish();
  work->deviceId_ = deviceId_;
  // Create a future to track the gather operation
  std::vector<at::Device> devices{inputTensor.device()};
  work->future_ = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), devices);
  work->future_->markCompleted(c10::IValue(outputTensorsTmp));
  return work;
}

c10::intrusive_ptr<Work> flagcxBackend::reduce(std::vector<at::Tensor> &tensors,
                                               const ReduceOptions &opts) {
  auto &tensor = tensors.back();
  auto flagcxDataType = getFlagcxDataType(tensor.scalar_type());
  auto flagcxReduceOp =
      getFlagcxReduceOp(opts.reduceOp, tensor, flagcxDataType);
  auto stream = getStreamByIndex(0);
  auto work =
      c10::make_intrusive<flagcxWork>(OpType::REDUCE, stream, devHandle_);
  initComm(tensor.device());
  syncStream(tensor.device());
  retainedTensorBatch retained(*this, tensor.device().index(), stream);
  retained.add(tensor);

  const auto root = opts.rootRank + opts.rootTensor;

#if (defined(USE_NVIDIA_ADAPTOR) || defined(USE_METAX_ADAPTOR)) &&             \
    defined(TORCH_VER_GE_250)
  if (needRecording()) {
    recordTuneObject(flagcxCommOpReduce, flagcxDataType, tensor.numel());
  }

#endif

  retained.markSubmitted();
  C10D_FLAGCX_CHECK(flagcxReduce(tensor.data_ptr(), tensor.data_ptr(),
                                 tensor.numel(), flagcxDataType, flagcxReduceOp,
                                 root, comm_, stream),
                    std::nullopt);

  work->event_->record(stream, deviceId_);
  retained.finish();
  work->deviceId_ = deviceId_;
  // Create a future to track the reduce operation
  std::vector<at::Device> devices{tensor.device()};
  work->future_ = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), devices);
  work->future_->markCompleted(c10::IValue(tensors));
  return work;
}

c10::intrusive_ptr<Work> flagcxBackend::reduce_scatter(
    std::vector<at::Tensor> &outputTensors,
    std::vector<std::vector<at::Tensor>> &inputTensors,
    const ReduceScatterOptions &opts) {
  auto outputTensor = outputTensors.back();
  auto inputTensorsTmp = inputTensors.back();
  auto device = outputTensor.device();
  auto flagcxDataType = getFlagcxDataType(outputTensor.scalar_type());
  auto flagcxReduceOp =
      getFlagcxReduceOp(opts.reduceOp, outputTensor, flagcxDataType);
  auto stream = getStreamByIndex(0);
  auto work = c10::make_intrusive<flagcxWork>(OpType::REDUCE_SCATTER, stream,
                                              devHandle_);
  check_device(outputTensor.device(), inputTensorsTmp[0].device());
  initComm(device);
  syncStream(device);
  retainedTensorBatch retained(*this, device.index(), stream);
  retained.add(outputTensor);
  retained.add(inputTensorsTmp);

  if (!check_same_size(inputTensorsTmp)) {
    throw std::runtime_error(
        "flagcx only support same size reducescatter operation");
  } else {
    // Flatten a vector of tensors into a single, stacked tensor.
    at::Tensor inputFlattened =
        newLikeFlatOnStream(inputTensorsTmp, stream, device.index());
    retained.add(inputFlattened);
    retained.markSubmitted();

    // Copy the input tensors to the flattened tensor.
    {
      flagcxStreamGuard guard(stream, device.index());
      for (const auto j : c10::irange(inputTensorsTmp.size())) {
        copyTensorOnStream(inputFlattened[j], inputTensorsTmp[j], stream,
                           devHandle_, device.index());
      }
    }

#if (defined(USE_NVIDIA_ADAPTOR) || defined(USE_METAX_ADAPTOR)) &&             \
    defined(TORCH_VER_GE_250)
    if (needRecording()) {
      recordTuneObject(flagcxCommOpReduceScatter, flagcxDataType,
                       outputTensor.numel());
    }

#endif

    // Perform the reducescatter operation
    C10D_FLAGCX_CHECK(flagcxReduceScatter(inputFlattened.data_ptr(),
                                          outputTensor.data_ptr(),
                                          outputTensor.numel(), flagcxDataType,
                                          flagcxReduceOp, comm_, stream),
                      std::nullopt);
  }

  work->event_->record(stream, deviceId_);
  retained.finish();
  work->deviceId_ = deviceId_;
  // Create a future to track the reducescatter operation
  std::vector<at::Device> devices{outputTensor.device()};
  work->future_ = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), devices);
  work->future_->markCompleted(c10::IValue(outputTensor));
  return work;
}

c10::intrusive_ptr<Work>
flagcxBackend::_reduce_scatter_base(at::Tensor &outputTensor,
                                    at::Tensor &inputTensor,
                                    const ReduceScatterOptions &opts) {
  auto flagcxDataType = getFlagcxDataType(outputTensor.scalar_type());
  auto flagcxReduceOp =
      getFlagcxReduceOp(opts.reduceOp, outputTensor, flagcxDataType);
  auto stream = getStreamByIndex(0);
  auto work = c10::make_intrusive<flagcxWork>(OpType::_REDUCE_SCATTER_BASE,
                                              stream, devHandle_);
  check_device(outputTensor.device(), inputTensor.device());
  initComm(outputTensor.device());
  syncStream(outputTensor.device());
  retainedTensorBatch retained(*this, outputTensor.device().index(), stream);
  retained.add(inputTensor);
  retained.add(outputTensor);

  if (inputTensor.numel() != outputTensor.numel() * size_) {
    throw std::runtime_error(
        "Input tensor must be the same szie as output size times world size");
  } else {
#if (defined(USE_NVIDIA_ADAPTOR) || defined(USE_METAX_ADAPTOR)) &&             \
    defined(TORCH_VER_GE_250)
    if (needRecording()) {
      recordTuneObject(flagcxCommOpReduceScatter, flagcxDataType,
                       outputTensor.numel());
    }

#endif
    // Perform the reducescatter operation
    retained.markSubmitted();
    C10D_FLAGCX_CHECK(flagcxReduceScatter(inputTensor.data_ptr(),
                                          outputTensor.data_ptr(),
                                          outputTensor.numel(), flagcxDataType,
                                          flagcxReduceOp, comm_, stream),
                      std::nullopt);
  }

  work->event_->record(stream, deviceId_);
  retained.finish();
  work->deviceId_ = deviceId_;
  // Create a future to track the reducescatter operation
  std::vector<at::Device> devices{outputTensor.device()};
  work->future_ = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), devices);
  work->future_->markCompleted(c10::IValue(outputTensor));
  return work;
}

c10::intrusive_ptr<Work> flagcxBackend::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor> &outputs, std::vector<at::Tensor> &inputs,
    const ReduceScatterOptions &opts) {
  // parameter validation
  check_gpu_tensors_same_device(inputs);
  TORCH_CHECK(
      !isFloat8Type(inputs.back().scalar_type()),
      "Float8 dtypes are not currenlty supported for FlagCX reductions");

  return collectiveCoalesced(
      inputs, outputs,
      [&](at::Tensor &input, at::Tensor &output, flagcxComm_t comm,
          flagcxStream_t stream) {
        auto flagcxDataType = getFlagcxDataType(input.scalar_type());
        auto flagcxReduceOp =
            getFlagcxReduceOp(opts.reduceOp, input, flagcxDataType);
#if (defined(USE_NVIDIA_ADAPTOR) || defined(USE_METAX_ADAPTOR)) &&             \
    defined(TORCH_VER_GE_250)
        if (needRecording()) {
          recordTuneObject(flagcxCommOpReduceScatter, flagcxDataType,
                           output.numel());
        }

#endif
        return flagcxReduceScatter(input.data_ptr(), output.data_ptr(),
                                   output.numel(), flagcxDataType,
                                   flagcxReduceOp, comm, stream);
      },
      OpType::COALESCED);
}

c10::intrusive_ptr<Work>
flagcxBackend::scatter(std::vector<at::Tensor> &outputTensors,
                       std::vector<std::vector<at::Tensor>> &inputTensors,
                       const ScatterOptions &opts) {
  auto &outputTensor = outputTensors.back();
  auto device = outputTensor.device();
  auto flagcxDataType = getFlagcxDataType(outputTensor.scalar_type());
  auto stream = getStreamByIndex(0);
  auto work =
      c10::make_intrusive<flagcxWork>(OpType::SCATTER, stream, devHandle_);
  initComm(device);
  syncStream(device);
  retainedTensorBatch retained(*this, device.index(), stream);
  retained.add(outputTensor);

  auto root = opts.rootRank;
  std::vector<at::Tensor> inputTensorsTmp;
  if (rank_ == root) {
    inputTensorsTmp = inputTensors.back();
  } else {
    inputTensorsTmp = {};
    inputTensorsTmp.emplace_back(
        at::ones({1}, at::TensorOptions().device(outputTensor.device())));
  }
  retained.add(inputTensorsTmp);

  // Flatten a vector of tensors into a single, stacked tensor.
#if defined(FLAGCX_TORCH_BACKEND_FLAGOS) && defined(USE_ENFLAME_ADAPTOR)
  auto inputTemplates = inputTensorsTmp;
  if (rank_ != root) {
    inputTemplates.resize(size_, outputTensor);
  }
  at::Tensor inputFlattened =
      newLikeFlatOnStream(inputTemplates, stream, device.index());
#else
  at::Tensor inputFlattened =
      newLikeFlatOnStream(inputTensorsTmp, stream, device.index());
#endif
  retained.add(inputFlattened);

  // Copy the input tensors to the flattened tensor.
  if (rank_ == root) {
    retained.markSubmitted();
    flagcxStreamGuard guard(stream, device.index());
    for (const auto j : c10::irange(inputTensorsTmp.size())) {
      copyTensorOnStream(inputFlattened[j], inputTensorsTmp[j], stream,
                         devHandle_, device.index());
    }
  }

#if (defined(USE_NVIDIA_ADAPTOR) || defined(USE_METAX_ADAPTOR)) &&             \
    defined(TORCH_VER_GE_250)
  if (needRecording()) {
    recordTuneObject(flagcxCommOpScatter, flagcxDataType, outputTensor.numel());
  }

#endif

  // Perform the scatter operation. The tested ECCL runtime does not populate
  // scatter outputs, so broadcast each root input and retain this rank's slot.
#if defined(FLAGCX_TORCH_BACKEND_FLAGOS) && defined(USE_ENFLAME_ADAPTOR)
  retained.markSubmitted();
  for (const auto peer : c10::irange(size_)) {
    C10D_FLAGCX_CHECK(flagcxBroadcast(inputFlattened[peer].data_ptr(),
                                      inputFlattened[peer].data_ptr(),
                                      outputTensor.numel(), flagcxDataType,
                                      root, comm_, stream),
                      std::nullopt);
  }
  copyTensorOnStream(outputTensor, inputFlattened[rank_], stream, devHandle_,
                     device.index());
#else
  retained.markSubmitted();
  C10D_FLAGCX_CHECK(flagcxScatter(inputFlattened.data_ptr(),
                                  outputTensor.data_ptr(), outputTensor.numel(),
                                  flagcxDataType, root, comm_, stream),
                    std::nullopt);
#endif

  work->event_->record(stream, deviceId_);
  retained.finish();
  work->deviceId_ = deviceId_;
  // Create a future to track the scatter operation
  std::vector<at::Device> devices{outputTensor.device()};
  work->future_ = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), devices);
  work->future_->markCompleted(c10::IValue(outputTensor));
  return work;
}

c10::intrusive_ptr<Work> flagcxBackend::send(std::vector<at::Tensor> &tensors,
                                             int dstRank, int tag) {
  const bool inPairBatch = usePairComm_ && pairCoalesce_.active;
  const bool inHeteroBatch = !usePairComm_ && heteroCoalesce_.active;
  const int uncaught = std::uncaught_exceptions();
  auto abortOnError = c10::make_scope_exit([&]() noexcept {
    if (std::uncaught_exceptions() > uncaught) {
      if (inPairBatch) {
        abortPairCoalescing();
      } else if (inHeteroBatch) {
        abortHeteroCoalescing();
      }
    }
  });
  if (!inPairBatch && !inHeteroBatch) {
    checkSubmissionHealthy();
  }
  TORCH_CHECK(tensors.size() == 1, "FlagCX send expects a single tensor");
  TORCH_CHECK(dstRank >= 0 && dstRank < size_, "Invalid FlagCX send peer ",
              dstRank, " for group size ", size_);
  auto &tensor = tensors.back();
  if (inPairBatch && pairCoalesce_.deviceId != -1) {
    TORCH_CHECK(tensor.device().index() == pairCoalesce_.deviceId,
                "A coalesced FlagCX P2P batch must use one device");
  }
  if (inHeteroBatch && heteroCoalesce_.deviceId != -1) {
    TORCH_CHECK(tensor.device().index() == heteroCoalesce_.deviceId,
                "A coalesced FlagCX P2P batch must use one device");
  }
  auto flagcxDataType = getFlagcxDataType(tensor.scalar_type());
  if (usePairComm_) {
    // Only the sender and receiver participate in pair-comm initialization.
    deviceId_ = tensor.device().index();
    C10D_FLAGCX_CHECK(devHandle_->setDevice(deviceId_), std::nullopt);
  } else {
    // Eager initialization is already complete, so this call is non-
    // collective and only validates that the tensor uses the same device.
    initComm(tensor.device());
  }
  auto stream = getStreamByIndex(0);
  auto work = c10::make_intrusive<flagcxWork>(OpType::SEND, stream, devHandle_);
  syncStream(tensor.device());

#if (defined(USE_NVIDIA_ADAPTOR) || defined(USE_METAX_ADAPTOR)) &&             \
    defined(TORCH_VER_GE_250)
  if (needRecording()) {
    recordTuneObject(flagcxCommOpSend, flagcxDataType, tensor.numel());
  }

#endif

  if (usePairComm_) {
    auto doSend = [this, tensor, flagcxDataType, stream, dstRank]() {
      flagcxComm_t p2pComm = getOrCreatePairComm(dstRank);
      int peerRank = (rank_ < dstRank) ? 1 : 0;
      C10D_FLAGCX_CHECK(flagcxSend(tensor.data_ptr(), tensor.numel(),
                                   flagcxDataType, peerRank, p2pComm, stream),
                        std::nullopt);
    };
    if (pairCoalesce_.active) {
      pairCoalesce_.pendingOps.push_back({dstRank, tensor, std::move(doSend)});
      pairCoalesce_.deviceId = deviceId_;
      return nullptr;
    }
    auto pairComm = getOrCreatePairComm(dstRank);
    auto batch = prepareTensorBatch(deviceId_, stream);
    bool mayHaveSubmitted = false;
    bool finalized = false;
    auto finishOnError = c10::make_scope_exit([&]() noexcept {
      if (!finalized) {
        finishTensorBatchOnError(batch, mayHaveSubmitted);
      }
    });
    batch->storages.push_back(tensor.storage());
    const int peerRank = (rank_ < dstRank) ? 1 : 0;
    mayHaveSubmitted = true;
    C10D_FLAGCX_CHECK(flagcxSend(tensor.data_ptr(), tensor.numel(),
                                 flagcxDataType, peerRank, pairComm, stream),
                      std::nullopt);
    finalized = true;
    finishTensorBatch(batch);
  } else {
    if (inHeteroBatch) {
      if (!heteroCoalesce_.batch) {
        heteroCoalesce_.batch = prepareTensorBatch(deviceId_, stream);
        heteroCoalesce_.deviceId = deviceId_;
      }
      (*heteroCoalesce_.batch)->storages.push_back(tensor.storage());
      heteroCoalesce_.mayHaveSubmitted = true;
      C10D_FLAGCX_CHECK(flagcxSend(tensor.data_ptr(), tensor.numel(),
                                   flagcxDataType, dstRank, comm_, stream),
                        std::nullopt);
      return nullptr;
    }
    auto batch = prepareTensorBatch(deviceId_, stream);
    bool mayHaveSubmitted = false;
    bool finalized = false;
    auto finishOnError = c10::make_scope_exit([&]() noexcept {
      if (!finalized) {
        finishTensorBatchOnError(batch, mayHaveSubmitted);
      }
    });
    batch->storages.push_back(tensor.storage());
    mayHaveSubmitted = true;
    C10D_FLAGCX_CHECK(flagcxSend(tensor.data_ptr(), tensor.numel(),
                                 flagcxDataType, dstRank, comm_, stream),
                      std::nullopt);
    finalized = true;
    finishTensorBatch(batch);
  }

  work->event_->record(stream, deviceId_);
  work->deviceId_ = deviceId_;
  std::vector<at::Device> devices{tensor.device()};
  work->future_ = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), devices);
  work->future_->markCompleted(c10::IValue(tensors));
  return work;
}

c10::intrusive_ptr<Work> flagcxBackend::recv(std::vector<at::Tensor> &tensors,
                                             int srcRank, int tag) {
  const bool inPairBatch = usePairComm_ && pairCoalesce_.active;
  const bool inHeteroBatch = !usePairComm_ && heteroCoalesce_.active;
  const int uncaught = std::uncaught_exceptions();
  auto abortOnError = c10::make_scope_exit([&]() noexcept {
    if (std::uncaught_exceptions() > uncaught) {
      if (inPairBatch) {
        abortPairCoalescing();
      } else if (inHeteroBatch) {
        abortHeteroCoalescing();
      }
    }
  });
  if (!inPairBatch && !inHeteroBatch) {
    checkSubmissionHealthy();
  }
  TORCH_CHECK(tensors.size() == 1, "FlagCX recv expects a single tensor");
  TORCH_CHECK(srcRank >= 0 && srcRank < size_, "Invalid FlagCX recv peer ",
              srcRank, " for group size ", size_);
  auto &tensor = tensors.back();
  if (inPairBatch && pairCoalesce_.deviceId != -1) {
    TORCH_CHECK(tensor.device().index() == pairCoalesce_.deviceId,
                "A coalesced FlagCX P2P batch must use one device");
  }
  if (inHeteroBatch && heteroCoalesce_.deviceId != -1) {
    TORCH_CHECK(tensor.device().index() == heteroCoalesce_.deviceId,
                "A coalesced FlagCX P2P batch must use one device");
  }
  auto flagcxDataType = getFlagcxDataType(tensor.scalar_type());
  if (usePairComm_) {
    deviceId_ = tensor.device().index();
    C10D_FLAGCX_CHECK(devHandle_->setDevice(deviceId_), std::nullopt);
  } else {
    initComm(tensor.device());
  }
  auto stream = getStreamByIndex(0);
  auto work = c10::make_intrusive<flagcxWork>(OpType::RECV, stream, devHandle_);
  syncStream(tensor.device());

#if (defined(USE_NVIDIA_ADAPTOR) || defined(USE_METAX_ADAPTOR)) &&             \
    defined(TORCH_VER_GE_250)
  if (needRecording()) {
    recordTuneObject(flagcxCommOpRecv, flagcxDataType, tensor.numel());
  }

#endif

  if (usePairComm_) {
    auto doRecv = [this, tensor, flagcxDataType, stream, srcRank]() {
      flagcxComm_t p2pComm = getOrCreatePairComm(srcRank);
      int peerRank = (rank_ < srcRank) ? 1 : 0;
      C10D_FLAGCX_CHECK(flagcxRecv(tensor.data_ptr(), tensor.numel(),
                                   flagcxDataType, peerRank, p2pComm, stream),
                        std::nullopt);
    };
    if (pairCoalesce_.active) {
      pairCoalesce_.pendingOps.push_back({srcRank, tensor, std::move(doRecv)});
      pairCoalesce_.deviceId = deviceId_;
      return nullptr;
    }
    auto pairComm = getOrCreatePairComm(srcRank);
    auto batch = prepareTensorBatch(deviceId_, stream);
    bool mayHaveSubmitted = false;
    bool finalized = false;
    auto finishOnError = c10::make_scope_exit([&]() noexcept {
      if (!finalized) {
        finishTensorBatchOnError(batch, mayHaveSubmitted);
      }
    });
    batch->storages.push_back(tensor.storage());
    const int peerRank = (rank_ < srcRank) ? 1 : 0;
    mayHaveSubmitted = true;
    C10D_FLAGCX_CHECK(flagcxRecv(tensor.data_ptr(), tensor.numel(),
                                 flagcxDataType, peerRank, pairComm, stream),
                      std::nullopt);
    finalized = true;
    finishTensorBatch(batch);
  } else {
    if (inHeteroBatch) {
      if (!heteroCoalesce_.batch) {
        heteroCoalesce_.batch = prepareTensorBatch(deviceId_, stream);
        heteroCoalesce_.deviceId = deviceId_;
      }
      (*heteroCoalesce_.batch)->storages.push_back(tensor.storage());
      heteroCoalesce_.mayHaveSubmitted = true;
      C10D_FLAGCX_CHECK(flagcxRecv(tensor.data_ptr(), tensor.numel(),
                                   flagcxDataType, srcRank, comm_, stream),
                        std::nullopt);
      return nullptr;
    }
    auto batch = prepareTensorBatch(deviceId_, stream);
    bool mayHaveSubmitted = false;
    bool finalized = false;
    auto finishOnError = c10::make_scope_exit([&]() noexcept {
      if (!finalized) {
        finishTensorBatchOnError(batch, mayHaveSubmitted);
      }
    });
    batch->storages.push_back(tensor.storage());
    mayHaveSubmitted = true;
    C10D_FLAGCX_CHECK(flagcxRecv(tensor.data_ptr(), tensor.numel(),
                                 flagcxDataType, srcRank, comm_, stream),
                      std::nullopt);
    finalized = true;
    finishTensorBatch(batch);
  }

  work->event_->record(stream, deviceId_);
  work->deviceId_ = deviceId_;
  std::vector<at::Device> devices{tensor.device()};
  work->future_ = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()), devices);
  work->future_->markCompleted(c10::IValue(tensors));
  return work;
}

c10::intrusive_ptr<Work>
flagcxBackend::recvAnysource(std::vector<at::Tensor> &tensors, int tag) {
  throw std::runtime_error("flagcxBackend does not support recvAnysource");
}

#if (defined(USE_NVIDIA_ADAPTOR) || defined(USE_METAX_ADAPTOR)) &&             \
    defined(TORCH_VER_GE_250)
void flagcxBackend::checkRecordingEnded() {
  const char *configIdEnv = std::getenv("FLAGCX_TUNER_CONFIG_ID");
  const int configId = (configIdEnv != NULL) ? std::atoi(configIdEnv) : -1;
  // if configId >= 0, we have finished the recording phase, and started tuning
  // phase
  if (configId >= 0)
    recordingEnded = true;
}

void flagcxBackend::recordTuneObject(flagcxCommOp_t commOp,
                                     flagcxDataType_t dataType, size_t count) {
  checkRecordingEnded();
  struct TuneObjectKey tuneObjectKey = {commOpToString(commOp),
                                        getDataSize(dataType, count)};
  if (tuneObjectSet_.find(tuneObjectKey) == tuneObjectSet_.end()) {
    // write this to file
    recordFlagcxTuneObject(tuneObjectKey, options_->tuneGroupIdx);
    tuneObjectSet_.insert(tuneObjectKey);
  }
}

bool flagcxBackend::needRecording() {
  if (recordingEnded || !options_->enableTuner) {
    return false;
  }
  const char *curTuneGroupIdxEnv = std::getenv("FLAGCX_TUNE_GROUP_IDX");
  const int curTuneGroupIdx =
      (curTuneGroupIdxEnv != NULL) ? std::atoi(curTuneGroupIdxEnv) : -1;
  return curTuneGroupIdx == options_->tuneGroupIdx;
}

c10::intrusive_ptr<Backend> flagcxBackend::createFlagcxBackend(
    c10d::DistributedBackendOptions backendOptions,
    c10::intrusive_ptr<Options> extraOptions) {
  const c10::intrusive_ptr<::c10d::Store> &store = backendOptions.store;
  int rank = backendOptions.group_rank;
  int size = backendOptions.group_size;
  return c10::make_intrusive<flagcxBackend>(store, rank, size, extraOptions);
}

flagcxBackend::Options::Options(bool enableTuner, int tuneGroupIdx)
    : Backend::Options(FLAGCX_BACKEND_NAME), enableTuner(enableTuner),
      tuneGroupIdx(tuneGroupIdx) {}

template <typename T>
using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>>;
#else
c10::intrusive_ptr<Backend> flagcxBackend::createFlagcxBackend(
    const c10::intrusive_ptr<::c10d::Store> &store, int rank, int size,
    const std::chrono::duration<float> & /* unused */) {
  return c10::make_intrusive<flagcxBackend>(store, rank, size);
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("createFlagcxBackend", &flagcxBackend::createFlagcxBackend);

#if (defined(USE_NVIDIA_ADAPTOR) || defined(USE_METAX_ADAPTOR)) &&             \
    defined(TORCH_VER_GE_250)
  py::object dist = py::module::import("torch._C._distributed_c10d");
  auto pg_flagcx = intrusive_ptr_class_<flagcxBackend>(
      m, "ProcessGroupFlagCX",
      dist.attr("Backend") // base Python class
  );
  intrusive_ptr_class_<flagcxBackend::Options>(
      pg_flagcx, "Options",
      dist.attr("Backend").attr("Options")) // base Python class
      .def(py::init<bool, int>(), py::arg("enable_tuner") = false,
           py::arg("tune_group_idx") = 0)
      .def_readwrite("enable_tuner", &flagcxBackend::Options::enableTuner)
      .def_readwrite("tune_group_idx", &flagcxBackend::Options::tuneGroupIdx);
#endif
}

} // namespace c10d
