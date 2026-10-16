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
#elif defined(USE_TSM_ADAPTOR)
#include <tx_runtime.h>
#else
#error "The FlagOS torch backend supports only Ascend, Enflame and TsingMicro"
#endif
#include <flagos.h>
#elif USE_NVIDIA_ADAPTOR
#include <ATen/cuda/CUDAEvent.h>
#include <cuda_runtime.h>
#elif USE_PPU_ADAPTOR
#include <ATen/cuda/CUDAEvent.h>
#include <cuda_runtime.h>
#elif USE_ASCEND_ADAPTOR
#include "torch_npu/csrc/core/npu/NPUEvent.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include <acl/acl.h>
#elif USE_ILUVATAR_ADAPTOR
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

#if USE_NVIDIA_ADAPTOR
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
#elif USE_PPU_ADAPTOR
class flagcxPpuEvent : public flagcxEvent {
public:
  flagcxPpuEvent() { ppuEvent_ = at::cuda::CUDAEvent(cudaEventDisableTiming); }

  void record(const int deviceId) override {
    ppuEvent_.record(at::cuda::getCurrentCUDAStream(deviceId));
  }

  void record(const flagcxStream_t &stream, const int deviceId) override {
    ppuEvent_.record(
        at::cuda::getStreamFromExternal(*(cudaStream_t *)stream, deviceId));
  }

  void block(const int deviceId) override {
    ppuEvent_.block(at::cuda::getCurrentCUDAStream(deviceId));
  }

  void block(const flagcxStream_t &stream, const int deviceId) override {
    ppuEvent_.block(
        at::cuda::getStreamFromExternal(*(cudaStream_t *)stream, deviceId));
  }

private:
  at::cuda::CUDAEvent ppuEvent_;
};
#elif USE_ILUVATAR_ADAPTOR
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
  // Driven as an ACL event, like the FLAGOS path above.
  //
  // The streams this backend hands out are raw ACL stream handles that
  // torch_npu does not own, and torch_npu's NPUEvent only accepts streams it
  // manages: wrapping one with getStreamFromExternal() and passing it to
  // NPUEvent::block() fails outright with "External NPU stream is not
  // supported in NPUEvent::block. This path requires a torch_npu-managed
  // stream." The ACL event API takes the stream handle as it is.
  //
  // ACL_EVENT_SYNC is the flag for stream-ordering events; events created with
  // ACL_EVENT_TIME_LINE are rejected by aclrtStreamWaitEvent on CANN 9.0.1.
  flagcxCannEvent() {
    auto ret = aclrtCreateEventWithFlag(&event_, ACL_EVENT_SYNC);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtCreateEventWithFlag failed: ", ret);
  }

  ~flagcxCannEvent() override {
    if (event_ != nullptr) {
      aclrtDestroyEvent(event_);
    }
  }

  void record(const int device_id) override {
    auto stream = c10_npu::getCurrentNPUStream(device_id).stream(true);
    auto ret = aclrtRecordEvent(event_, stream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtRecordEvent failed: ", ret);
  }

  void record(const flagcxStream_t &stream, const int /*device_id*/) override {
    // Place the event on the stream the operation was actually submitted to.
    //
    // Both stream-taking overloads used to ignore the stream argument and use
    // an unrelated stream from the pool. That stream carries no work, so the
    // event was satisfied immediately and neither ordering edge existed:
    // flagcxWork::wait() blocks on this event, and syncStream() uses the
    // block() overload below to make the submission stream wait for the
    // caller's stream. Measured on Ascend 910C, a side-stream all_reduce
    // followed only by work.wait() returned pre-reduction data on 21 of 60
    // calls.
    auto ret =
        aclrtRecordEvent(event_, *reinterpret_cast<aclrtStream *>(stream));
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtRecordEvent failed: ", ret);
  }

  void block(const int device_id) override {
    auto stream = c10_npu::getCurrentNPUStream(device_id).stream(true);
    auto ret = aclrtStreamWaitEvent(stream, event_);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtStreamWaitEvent failed: ", ret);
  }

  void block(const flagcxStream_t &stream, const int /*device_id*/) override {
    auto ret =
        aclrtStreamWaitEvent(*reinterpret_cast<aclrtStream *>(stream), event_);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtStreamWaitEvent failed: ", ret);
  }

private:
  aclrtEvent event_ = nullptr;
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
#ifdef FLAGCX_TORCH_BACKEND_FLAGOS
  // The FlagOS runtime exposes a device stream but no event type, so the event
  // is driven through the native TSM runtime, like the Ascend and Enflame
  // FlagOS paths.
  flagcxTxdaEvent() {
    TORCH_CHECK(txEventCreate(&event_) == TX_SUCCESS, "txEventCreate failed");
  }

  ~flagcxTxdaEvent() override {
    if (event_ != nullptr) {
      txEventDestroy(event_);
    }
  }

  void record(const int deviceId) override {
    TORCH_CHECK(
        txEventRecord(event_, reinterpret_cast<txStream_t>(
                                  GetCurrentStreamForDevice(deviceId))) ==
            TX_SUCCESS,
        "txEventRecord failed");
  }

  void record(const flagcxStream_t &stream, const int /*deviceId*/) override {
    TORCH_CHECK(txEventRecord(event_, *reinterpret_cast<txStream_t *>(stream)) ==
                    TX_SUCCESS,
                "txEventRecord failed");
  }

  void block(const int deviceId) override {
    TORCH_CHECK(txStreamWaitEvent(reinterpret_cast<txStream_t>(
                                      GetCurrentStreamForDevice(deviceId)),
                                  event_) == TX_SUCCESS,
                "txStreamWaitEvent failed");
  }

  void block(const flagcxStream_t &stream, const int /*deviceId*/) override {
    TORCH_CHECK(txStreamWaitEvent(*reinterpret_cast<txStream_t *>(stream),
                                  event_) == TX_SUCCESS,
                "txStreamWaitEvent failed");
  }

private:
  txEvent_t event_ = nullptr;
#else
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
#endif
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
