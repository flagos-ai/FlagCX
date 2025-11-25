/*************************************************************************
 * Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd. All
 *Rights Reserved. Copyright (c) 2025 by DU. All Rights Reserved.
 ************************************************************************/
#pragma once

#include <c10/core/DeviceGuard.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/python.h>

#include <memory>
#include <pybind11/chrono.h>
#include <unordered_map>
#include <vector>

#include "event_flagcx.hpp"
#include "stream_guard_flagcx.hpp"

namespace c10d {

constexpr const char *FLAGCX_BACKEND_NAME = "flagcx";

class flagcxWork : public Work {
  friend class flagcxBackend;

public:
  flagcxWork(OpType opType, flagcxStream_t stream = nullptr,
             flagcxDeviceHandle_t handler = nullptr,
             c10::intrusive_ptr<c10::ivalue::Future> future =
                 nullptr, // future of the output
             int deviceId = 0)
      : Work(-1, // rank, only used by recvAnySource, irrelevant in this
                 // implementation
             opType),
        stream_(stream), handler_(handler), future_(std::move(future)),
        deviceId_(deviceId), isBarrierOp_(false) {
#ifdef USE_NVIDIA_ADAPTOR
    event_ = std::make_unique<flagcxCudaEvent>();
#elif USE_ASCEND_ADAPTOR
    event_ = std::make_unique<flagcxCannEvent>();
#elif USE_ILUVATAR_COREX_ADAPTOR
    event_ = std::make_unique<flagcxIxcudaEvent>();
#elif USE_CAMBRICON_ADAPTOR
    event_ = std::make_unique<flagcxMluEvent>();
#elif USE_METAX_ADAPTOR
    event_ = std::make_unique<flagcxMacaEvent>();
#elif USE_MUSA_ADAPTOR
    event_ = std::make_unique<flagcxMusaEvent>();
#elif USE_DU_ADAPTOR
    event_ = std::make_unique<flagcxDuEvent>();
#elif USE_KUNLUNXIN_ADAPTOR
    event_ = std::make_unique<flagcxXpuEvent>();
#elif USE_AMD_ADAPTOR
    event_ = std::make_unique<flagcxHipEvent>();
#endif
  }
  bool isCompleted() override;
  bool isSuccess() const override;
  bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;
  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

private:
  flagcxStream_t stream_;
  flagcxDeviceHandle_t handler_;
  c10::intrusive_ptr<c10::ivalue::Future> future_;
  int deviceId_;
  bool isBarrierOp_;
  std::unique_ptr<flagcxEvent> event_;
};

class flagcxBackend : public Backend {
public:
  struct Options : Backend::Options {
    explicit Options(bool enableTuner = false);

    static c10::intrusive_ptr<Options> create(bool enableTuner = false) {
      return c10::make_intrusive<Options>(enableTuner);
    }

    bool enableTuner{false};
  };

  explicit flagcxBackend(
      const c10::intrusive_ptr<::c10d::Store> &store, int rank = -1,
      int size = -1, c10::intrusive_ptr<Options> options = Options::create());

  ~flagcxBackend() override;

  void startCoalescing() override;

  c10::intrusive_ptr<Work> endCoalescing() override;

  const std::string getBackendName() const override {
    static const std::string name{FLAGCX_BACKEND_NAME};
    return name;
  }

  // For specifying a composite optype, such as ALLGATHER and REDUCE_SCATTER
  c10::intrusive_ptr<Work> endCoalescing(OpType optype);

  c10::intrusive_ptr<Work>
  broadcast(std::vector<at::Tensor> &data,
            const BroadcastOptions &opts = BroadcastOptions()) override;

  c10::intrusive_ptr<Work>
  allreduce(std::vector<at::Tensor> &tensors,
            const AllreduceOptions &opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work>
  allreduce_coalesced(std::vector<at::Tensor> &tensors,
                      const AllreduceCoalescedOptions &opts =
                          AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<Work>
  reduce(std::vector<at::Tensor> &tensors,
         const ReduceOptions &opts = ReduceOptions()) override;

  c10::intrusive_ptr<Work>
  allgather(std::vector<std::vector<at::Tensor>> &outputTensors,
            std::vector<at::Tensor> &inputTensors,
            const AllgatherOptions &opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work>
  _allgather_base(at::Tensor &outputBuffer, at::Tensor &inputBuffer,
                  const AllgatherOptions &opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor> &outputs, std::vector<at::Tensor> &inputs,
      const AllgatherOptions &opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work>
  barrier(const BarrierOptions &opts = BarrierOptions()) override;

  c10::intrusive_ptr<Work>
  gather(std::vector<std::vector<at::Tensor>> &outputTensors,
         std::vector<at::Tensor> &inputTensors,
         const GatherOptions &opts = GatherOptions()) override;

  c10::intrusive_ptr<Work>
  scatter(std::vector<at::Tensor> &outputTensors,
          std::vector<std::vector<at::Tensor>> &inputTensors,
          const ScatterOptions &opts = ScatterOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor> &outputTensors,
      std::vector<std::vector<at::Tensor>> &inputTensors,
      const ReduceScatterOptions &opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> _reduce_scatter_base(
      at::Tensor &outputTensor, at::Tensor &inputTensor,
      const ReduceScatterOptions &opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor> &outputs, std::vector<at::Tensor> &inputs,
      const ReduceScatterOptions &opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work>
  alltoall_base(at::Tensor &outputTensor, at::Tensor &inputTensor,
                std::vector<int64_t> &outputSplitSizes,
                std::vector<int64_t> &inputSplitSizes,
                const AllToAllOptions &opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work>
  alltoall(std::vector<at::Tensor> &outputTensors,
           std::vector<at::Tensor> &inputTensors,
           const AllToAllOptions &opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> send(std::vector<at::Tensor> &tensors, int dstRank,
                                int tag) override;

  c10::intrusive_ptr<Work> recv(std::vector<at::Tensor> &tensors, int srcRank,
                                int tag) override;

  c10::intrusive_ptr<Work> recvAnysource(std::vector<at::Tensor> &tensors,
                                         int tag) override;

  static c10::intrusive_ptr<Backend> createFlagcxBackend(
      c10d::DistributedBackendOptions backendOptions,
      c10::intrusive_ptr<Options> extraOptions = Options::create());

  static void flagcxBackendConstructor() __attribute__((constructor)) {
    std::string devName = "cuda";
#ifdef USE_NVIDIA_ADAPTOR
    devName = "cuda";
#elif USE_ASCEND_ADAPTOR
    devName = "cann";
#elif USE_ILUVATAR_COREX_ADAPTOR
    devName = "cuda";
#elif USE_CAMBRICON_ADAPTOR
    devName = "mlu";
#elif USE_METAX_ADAPTOR
    devName = "cuda";
#elif USE_MUSA_ADAPTOR
    devName = "musa";
#elif USE_DU_ADAPTOR
    devName = "cuda";
#elif USE_KUNLUNXIN_ADAPTOR
    devName = "cuda";
#endif
    py::object module = py::module::import("torch.distributed");
    py::object registerBackend =
        module.attr("Backend").attr("register_backend");
    registerBackend("flagcx", py::cpp_function(createFlagcxBackend),
                    py::arg("extended_api") = true,
                    py::arg("devices") = py::make_tuple(devName));
  }

protected:
  flagcxStream_t getStreamByIndex(int streamId);
  std::unique_ptr<flagcxEvent> &getEventByIndex(int eventId);
  void initComm(at::Device dev);
  void initComm();
  void syncStream(at::Device device, int index = 0);
  void groupStart();
  void groupEnd();

  c10::intrusive_ptr<::c10d::Store> store_;
  int nDevs_;
  int deviceId_;
  int status_; // 0: allocated, 1: initialized
  uint64_t activeGroupCounter_;
  std::unordered_map<int, flagcxStream_t> flagcxStreams_;
  std::unordered_map<int, std::unique_ptr<flagcxEvent>> flagcxEvents_;
  flagcxHandlerGroup_t handler_ = nullptr;
  const c10::intrusive_ptr<Options> options_;
#ifdef USE_ASCEND_ADAPTOR
  aclrtStream acl_stream;
#endif

private:
  // Helper that encapsulates work shared across all collective communication
  // primitives.  The callbacks have the following signatures:
  //
  //    flagcxResult_t fn(at::Tensor& input, at::Tensor& output,
  //                    flagcxComm_t, flagcxStream_t&);
  template <typename Fn>
  c10::intrusive_ptr<Work> collectiveCoalesced(std::vector<at::Tensor> &input,
                                               std::vector<at::Tensor> &output,
                                               Fn fn, OpType opType);
};

} // namespace c10d
