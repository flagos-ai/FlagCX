/*************************************************************************
 * Copyright (c) 2025 by MetaX Integrated Circuits (Shanghai) Co., Ltd.
   All Rights Reserved.
 * Copyright (c) 2025 by DU. All Rights Reserved.
 ************************************************************************/
#pragma once

#include "flagcx.h"

#ifdef FLAGCX_TORCH_BACKEND_FLAGOS
#if defined(USE_ASCEND_ADAPTOR)
#include <acl/acl.h>
#elif defined(USE_ENFLAME_ADAPTOR)
#include <tops/tops_runtime_api.h>
#else
#error "The FlagOS torch backend supports only Ascend and Enflame"
#endif
#include <flagos.h>
#elif USE_NVIDIA_ADAPTOR
#include <ATen/cuda/CUDAEvent.h>
#include <cuda_runtime.h>
#elif USE_ASCEND_ADAPTOR
#include "torch_npu/csrc/core/npu/NPUEvent.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#elif USE_ILUVATAR_COREX_ADAPTOR
#include <ATen/cuda/CUDAEvent.h>
#include <cuda_runtime.h>
#elif USE_CAMBRICON_ADAPTOR
#include "framework/core/MLUEvent.h"
#include "framework/core/MLUStream.h"
#elif USE_METAX_ADAPTOR
#include <c10/core/Event.h>
#include <cstdint>
#include <cstring>
#elif USE_MUSA_ADAPTOR
#include "torch_musa/csrc/core/MUSAEvent.h"
#include "torch_musa/csrc/core/MUSAStream.h"
#elif USE_DU_ADAPTOR
#include <ATen/hip/HIPEvent.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#elif USE_KUNLUNXIN_ADAPTOR
#include <ATen/cuda/CUDAEvent.h>
#include <cuda_runtime.h>
#elif USE_AMD_ADAPTOR
#include <ATen/hip/HIPEvent.h>
#include <hip/hip_runtime.h>
#elif USE_TSM_ADAPTOR
#include "torch_txda/csrc/core/TXDAEvent.h"
#include "torch_txda/csrc/core/TXDAStream.h"
#include <tx_runtime.h>
#elif USE_ENFLAME_ADAPTOR
#ifdef FLAGCX_TORCH_BACKEND_FLAGOS
#include <flagos.h>
#include <tops/tops_runtime_api.h>
#else
#include <gcu/gcu_event.h>
#include <gcu/gcu_guard.h>
#include <tops/tops_runtime_api.h>
#endif
#elif USE_SUNRISE_ADAPTOR
#include <tang_runtime.h>
#include <torch_ptpu/core/Event.h>
#include <torch_ptpu/core/Stream.h>
#endif

namespace c10d {

#ifdef USE_METAX_ADAPTOR
#ifndef FLAGCX_TORCH_METAX_STREAM_HELPERS
#define FLAGCX_TORCH_METAX_STREAM_HELPERS
inline c10::StreamId flagcxStreamIdFromFlagcxStream(flagcxStream_t stream) {
  static_assert(sizeof(c10::StreamId) >= sizeof(uintptr_t),
                "stream handle does not fit in c10::StreamId");
  uintptr_t stream_handle = 0;
  std::memcpy(&stream_handle, stream, sizeof(stream_handle));
  c10::StreamId stream_id = 0;
  std::memcpy(&stream_id, &stream_handle, sizeof(stream_handle));
  return stream_id;
}

inline c10::Stream flagcxToC10Stream(flagcxStream_t stream, int device_id) {
  return c10::Stream(c10::Stream::UNSAFE,
                     c10::Device(c10::DeviceType::CUDA, device_id),
                     flagcxStreamIdFromFlagcxStream(stream));
}
#endif
#endif

class flagcxEvent {
public:
  virtual ~flagcxEvent() = default;

  virtual void record(const int deviceId) = 0;
  virtual void record(const flagcxStream_t &stream, const int deviceId) = 0;

  virtual void block(const int deviceId) = 0;
  virtual void block(const flagcxStream_t &stream, const int deviceId) = 0;
};

#ifdef USE_NVIDIA_ADAPTOR
class flagcxCudaEvent : public flagcxEvent {
public:
  flagcxCudaEvent() {
    cudaEvent_ = at::cuda::CUDAEvent(cudaEventDisableTiming);
  }

  void record(const int deviceId) override {
    cudaEvent_.record(at::cuda::getCurrentCUDAStream(deviceId));
  }

  void record(const flagcxStream_t &stream, const int deviceId) override {
    cudaEvent_.record(
        at::cuda::getStreamFromExternal(*(cudaStream_t *)stream, deviceId));
  }

  void block(const int deviceId) override {
    cudaEvent_.block(at::cuda::getCurrentCUDAStream(deviceId));
  }

  void block(const flagcxStream_t &stream, const int deviceId) override {
    cudaEvent_.block(
        at::cuda::getStreamFromExternal(*(cudaStream_t *)stream, deviceId));
  }

private:
  at::cuda::CUDAEvent cudaEvent_;
};
#elif USE_ILUVATAR_COREX_ADAPTOR
class flagcxIxcudaEvent : public flagcxEvent {
public:
  flagcxIxcudaEvent() {
    ixcuda_event = at::cuda::CUDAEvent(cudaEventDisableTiming);
  }

  void record(const int device_id) override {
    ixcuda_event.record(at::cuda::getCurrentCUDAStream(device_id));
  }

  void record(const flagcxStream_t &stream, const int device_id) override {
    ixcuda_event.record(
        at::cuda::getStreamFromExternal(*(cudaStream_t *)stream, device_id));
  }

  void block(const int device_id) override {
    ixcuda_event.block(at::cuda::getCurrentCUDAStream(device_id));
  }

  void block(const flagcxStream_t &stream, const int device_id) override {
    ixcuda_event.block(
        at::cuda::getStreamFromExternal(*(cudaStream_t *)stream, device_id));
  }

private:
  at::cuda::CUDAEvent ixcuda_event;
};
#elif USE_ASCEND_ADAPTOR
class flagcxCannEvent : public flagcxEvent {
public:
#ifdef FLAGCX_TORCH_BACKEND_FLAGOS
  flagcxCannEvent() {
    TORCH_CHECK(aclrtCreateEventWithFlag(&event_, ACL_EVENT_TIME_LINE) ==
                    ACL_SUCCESS,
                "aclrtCreateEventWithFlag failed");
  }

  ~flagcxCannEvent() override {
    if (event_ != nullptr) {
      aclrtDestroyEvent(event_);
    }
  }

  void record(const int deviceId) override {
    auto stream =
        reinterpret_cast<aclrtStream>(GetCurrentStreamForDevice(deviceId));
    TORCH_CHECK(aclrtRecordEvent(event_, stream) == ACL_SUCCESS,
                "aclrtRecordEvent failed");
  }

  void record(const flagcxStream_t &stream, const int /*deviceId*/) override {
    TORCH_CHECK(aclrtRecordEvent(event_, *reinterpret_cast<aclrtStream *>(
                                             stream)) == ACL_SUCCESS,
                "aclrtRecordEvent failed");
  }

  void block(const int deviceId) override {
    auto stream =
        reinterpret_cast<aclrtStream>(GetCurrentStreamForDevice(deviceId));
    TORCH_CHECK(aclrtStreamWaitEvent(stream, event_) == ACL_SUCCESS,
                "aclrtStreamWaitEvent failed");
  }

  void block(const flagcxStream_t &stream, const int /*deviceId*/) override {
    TORCH_CHECK(aclrtStreamWaitEvent(*reinterpret_cast<aclrtStream *>(stream),
                                     event_) == ACL_SUCCESS,
                "aclrtStreamWaitEvent failed");
  }

private:
  aclrtEvent event_ = nullptr;
#else
  flagcxCannEvent() { npu_event = c10_npu::NPUEvent(); }

  void record(const int device_id) override {
    npu_event.record(c10_npu::getCurrentNPUStream(device_id));
  }

  void record(const flagcxStream_t &stream, const int device_id) override {
    npu_event.record(c10_npu::getNPUStreamFromPool(device_id));
  }

  void block(const int device_id) override {
    npu_event.block(c10_npu::getCurrentNPUStream(device_id));
  }

  void block(const flagcxStream_t &stream, const int device_id) override {
    npu_event.block(c10_npu::getNPUStreamFromPool(device_id));
  }

private:
  c10_npu::NPUEvent npu_event;
#endif
};
#elif USE_CAMBRICON_ADAPTOR
class flagcxMluEvent : public flagcxEvent {
public:
  flagcxMluEvent() { mlu_event = torch_mlu::MLUEvent(); }

  void record(const int device_id) override {
    mlu_event.place(torch_mlu::getCurrentMLUStream(device_id));
  }

  void record(const flagcxStream_t &stream, const int device_id) override {
    mlu_event.place(
        torch_mlu::getStreamFromExternal(*(cnrtQueue_t *)stream, device_id));
  }

  void block(const int device_id) override {
    mlu_event.wait(torch_mlu::getCurrentMLUStream(device_id));
  }

  void block(const flagcxStream_t &stream, const int device_id) override {
    mlu_event.wait(
        torch_mlu::getStreamFromExternal(*(cnrtQueue_t *)stream, device_id));
  }

private:
  torch_mlu::MLUEvent mlu_event;
};
#elif USE_METAX_ADAPTOR
class flagcxMacaEvent : public flagcxEvent {
public:
  flagcxMacaEvent() : maca_event(c10::DeviceType::CUDA) {}

  void record(const int device_id) override {
    auto guard_impl = c10::impl::getDeviceGuardImpl(c10::DeviceType::CUDA);
    auto stream =
        guard_impl->getStream(c10::Device(c10::DeviceType::CUDA, device_id));
    maca_event.record(stream);
  }

  void record(const flagcxStream_t &stream, const int device_id) override {
    maca_event.record(flagcxToC10Stream(stream, device_id));
  }

  void block(const int device_id) override {
    auto guard_impl = c10::impl::getDeviceGuardImpl(c10::DeviceType::CUDA);
    auto stream =
        guard_impl->getStream(c10::Device(c10::DeviceType::CUDA, device_id));
    maca_event.block(stream);
  }

  void block(const flagcxStream_t &stream, const int device_id) override {
    maca_event.block(flagcxToC10Stream(stream, device_id));
  }

private:
  c10::Event maca_event;
};
#elif USE_MUSA_ADAPTOR
class flagcxMusaEvent : public flagcxEvent {
public:
  flagcxMusaEvent() {
    musa_event = at::musa::MUSAEvent(musaEventDisableTiming);
  }

  void record(const int device_id) override {
    musa_event.record(c10::musa::getCurrentMUSAStream(device_id));
  }

  void record(const flagcxStream_t &stream, const int device_id) override {
    musa_event.record(
        c10::musa::getStreamFromExternal(*(musaStream_t *)stream, device_id));
  }

  void block(const int device_id) override {
    musa_event.block(c10::musa::getCurrentMUSAStream(device_id));
  }

  void block(const flagcxStream_t &stream, const int device_id) override {
    musa_event.block(
        c10::musa::getStreamFromExternal(*(musaStream_t *)stream, device_id));
  }

private:
  at::musa::MUSAEvent musa_event;
};
#elif USE_DU_ADAPTOR
class flagcxDuEvent : public flagcxEvent {
public:
  flagcxDuEvent() { hipEvent_ = at::cuda::CUDAEvent(hipEventDisableTiming); }

  void record(const int deviceId) override {
    hipEvent_.record(at::hip::getCurrentHIPStreamMasqueradingAsCUDA(deviceId));
  }

  void record(const flagcxStream_t &stream, const int deviceId) override {
    hipEvent_.record(at::hip::getStreamFromExternalMasqueradingAsCUDA(
        *(hipStream_t *)stream, deviceId));
  }

  void block(const int deviceId) override {
    hipEvent_.block(at::hip::getCurrentHIPStreamMasqueradingAsCUDA(deviceId));
  }

  void block(const flagcxStream_t &stream, const int deviceId) override {
    hipEvent_.block(at::hip::getStreamFromExternalMasqueradingAsCUDA(
        *(hipStream_t *)stream, deviceId));
  }

private:
  at::cuda::CUDAEvent hipEvent_;
};
#elif USE_KUNLUNXIN_ADAPTOR
class flagcxXpuEvent : public flagcxEvent {
public:
  flagcxXpuEvent() { cudaEvent_ = at::cuda::CUDAEvent(cudaEventDisableTiming); }

  void record(const int deviceId) override {
    cudaEvent_.record(at::cuda::getCurrentCUDAStream(deviceId));
  }

  void record(const flagcxStream_t &stream, const int deviceId) override {
    cudaEvent_.record(
        at::cuda::getStreamFromExternal(*(cudaStream_t *)stream, deviceId));
  }

  void block(const int deviceId) override {
    cudaEvent_.block(at::cuda::getCurrentCUDAStream(deviceId));
  }

  void block(const flagcxStream_t &stream, const int deviceId) override {
    cudaEvent_.block(
        at::cuda::getStreamFromExternal(*(cudaStream_t *)stream, deviceId));
  }

private:
  at::cuda::CUDAEvent cudaEvent_;
};
#elif USE_AMD_ADAPTOR
class flagcxHipEvent : public flagcxEvent {
public:
  flagcxHipEvent() { hipEvent_ = at::cuda::CUDAEvent(hipEventDisableTiming); }

  void record(const int deviceId) override {
    hipEvent_.record(at::hip::getCurrentHIPStreamMasqueradingAsCUDA(deviceId));
  }

  void record(const flagcxStream_t &stream, const int deviceId) override {
    hipEvent_.record(at::hip::getStreamFromExternalMasqueradingAsCUDA(
        *(hipStream_t *)stream, deviceId));
  }

  void block(const int deviceId) override {
    hipEvent_.block(at::hip::getCurrentHIPStreamMasqueradingAsCUDA(deviceId));
  }

  void block(const flagcxStream_t &stream, const int deviceId) override {
    hipEvent_.block(at::hip::getStreamFromExternalMasqueradingAsCUDA(
        *(hipStream_t *)stream, deviceId));
  }

private:
  at::cuda::CUDAEvent hipEvent_;
};
#elif USE_TSM_ADAPTOR
class flagcxTxdaEvent : public flagcxEvent {
public:
  flagcxTxdaEvent() { txda_event = torch_txda::TXDAEvent(); }

  void record(const int device_id) override {
    txda_event.record(torch_txda::getCurrentTXDAStream(device_id));
  }

  void record(const flagcxStream_t &stream, const int device_id) override {
    txda_event.record(
        torch_txda::getStreamFromExternal(*(txStream_t *)stream, device_id));
  }

  void block(const int device_id) override {
    txda_event.block(torch_txda::getCurrentTXDAStream(device_id));
  }

  void block(const flagcxStream_t &stream, const int device_id) override {
    txda_event.block(
        torch_txda::getStreamFromExternal(*(txStream_t *)stream, device_id));
  }

private:
  torch_txda::TXDAEvent txda_event;
};
#elif USE_ENFLAME_ADAPTOR
class flagcxTopsEvent : public flagcxEvent {
public:
#ifdef FLAGCX_TORCH_BACKEND_FLAGOS
  flagcxTopsEvent() {
    TORCH_CHECK(topsEventCreateWithFlags(&event_, topsEventDisableTiming) ==
                    topsSuccess,
                "topsEventCreateWithFlags failed");
  }

  ~flagcxTopsEvent() override {
    if (event_ != nullptr) {
      topsEventDestroy(event_);
    }
  }

  void record(const int deviceId) override {
    TORCH_CHECK(topsEventRecord(event_, reinterpret_cast<topsStream_t>(
                                            GetCurrentStreamForDevice(
                                                deviceId))) == topsSuccess,
                "topsEventRecord failed");
  }

  void record(const flagcxStream_t &stream, const int /*deviceId*/) override {
    TORCH_CHECK(topsEventRecord(event_, *reinterpret_cast<topsStream_t *>(
                                            stream)) == topsSuccess,
                "topsEventRecord failed");
  }

  void block(const int deviceId) override {
    TORCH_CHECK(topsStreamWaitEvent(reinterpret_cast<topsStream_t>(
                                        GetCurrentStreamForDevice(deviceId)),
                                    event_, 0) == topsSuccess,
                "topsStreamWaitEvent failed");
  }

  void block(const flagcxStream_t &stream, const int /*deviceId*/) override {
    TORCH_CHECK(topsStreamWaitEvent(*reinterpret_cast<topsStream_t *>(stream),
                                    event_, 0) == topsSuccess,
                "topsStreamWaitEvent failed");
  }

private:
  topsEvent_t event_ = nullptr;
#else
  flagcxTopsEvent() { topsEvent_ = torch_gcu::GCUEvent(); }

  void record(const int deviceId) override {
    topsEvent_.record(torch_gcu::getCurrentGCUStream(deviceId));
  }

  void record(const flagcxStream_t &stream, const int deviceId) override {
    topsEvent_.record(
        torch_gcu::getStreamFromExternal(*(topsStream_t *)stream, deviceId));
  }

  void block(const int deviceId) override {
    topsEvent_.block(torch_gcu::getCurrentGCUStream(deviceId));
  }

  void block(const flagcxStream_t &stream, const int deviceId) override {
    topsEvent_.block(
        torch_gcu::getStreamFromExternal(*(topsStream_t *)stream, deviceId));
  }

private:
  torch_gcu::GCUEvent topsEvent_;
#endif
};
#elif USE_SUNRISE_ADAPTOR
class flagcxPtpuEvent : public flagcxEvent {
public:
  flagcxPtpuEvent() : event_(tangEventDisableTiming) {}

  void record(const int deviceId) override {
    event_.record(torchpt::get_current_stream(deviceId));
  }

  void record(const flagcxStream_t &stream, const int deviceId) override {
    ensurePrimed(deviceId);
    auto rc = tangEventRecord(event_.raw(), *(tangStream_t *)stream);
    TORCH_CHECK(rc == tangSuccess,
                "tangEventRecord on flagcx-allocated tangStream_t failed: ",
                tangGetErrorString(rc));
  }

  void block(const int deviceId) override {
    if (event_.raw() == nullptr) {
      return;
    }
    event_.block(torchpt::get_current_stream(deviceId));
  }

  void block(const flagcxStream_t &stream, const int /* deviceId */) override {
    if (event_.raw() == nullptr) {
      return;
    }
    auto rc = tangStreamWaitEvent(*(tangStream_t *)stream, event_.raw(),
                                  tangEventWaitDefault);
    TORCH_CHECK(rc == tangSuccess,
                "tangStreamWaitEvent on flagcx-allocated tangStream_t failed: ",
                tangGetErrorString(rc));
  }

private:
  // PTPUEvent lazily allocates tangEvent_t on first record(PTPUStream),
  // but the external-stream overloads bypass that. Prime by recording
  // once on the pool stream; the subsequent real tangEventRecord
  // overwrites it, so cost is one redundant enqueue on first call only.
  void ensurePrimed(int deviceId) {
    if (event_.raw() != nullptr) {
      return;
    }
    event_.record(torchpt::get_current_stream(deviceId));
  }

  torchpt::PTPUEvent event_;
};
#endif

} // namespace c10d
