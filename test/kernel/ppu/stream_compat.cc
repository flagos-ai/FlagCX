// PPU Torch uses the SDK CUDA compatibility ABI. Verify ordering across that
// boundary using host callbacks, without requiring a Torch installation.
#include "ppu_adaptor.h"
#include "ppu_stream_compat.hpp"
#include <atomic>
#include <cstdio>

#define HGGC_CHECK(call)                                                       \
  do {                                                                         \
    hggcError_t result = (call);                                               \
    if (result != hggcSuccess) {                                               \
      std::fprintf(stderr, "%s failed: %d\n", #call, int(result));             \
      return 1;                                                                \
    }                                                                          \
  } while (0)
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t result = (call);                                               \
    if (result != cudaSuccess) {                                               \
      std::fprintf(stderr, "%s failed: %d\n", #call, int(result));             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

struct Callback {
  std::atomic<int> *value;
  std::atomic<bool> *failed;
  int expected;
  static void run(void *arg) {
    auto &state = *static_cast<Callback *>(arg);
    if (state.value->exchange(state.expected + 1) != state.expected)
      state.failed->store(true);
  }
};

int main() {
  flagcxStream native{};
  cudaStream_t compatibility = nullptr;
  hggcEvent_t nativeEvent = nullptr;
  cudaEvent_t compatibilityEvent = nullptr;
  HGGC_CHECK(hggcStreamCreateWithFlags(&native.base, hggcStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&compatibility, cudaStreamNonBlocking));
  HGGC_CHECK(hggcEventCreateWithFlags(&nativeEvent, hggcEventDisableTiming));
  CUDA_CHECK(
      cudaEventCreateWithFlags(&compatibilityEvent, cudaEventDisableTiming));
  std::atomic<int> value{0};
  std::atomic<bool> failed{false};
  Callback callbacks[] = {
      {&value, &failed, 0}, {&value, &failed, 1}, {&value, &failed, 2}};
  HGGC_CHECK(hggcLaunchHostFunc(native.base, Callback::run, &callbacks[0]));
  // Export a FlagCX/HGGC stream to Torch's compatibility runtime.
  CUDA_CHECK(cudaEventRecord(compatibilityEvent,
                             flagcxPpuTorch::toTorchStream(&native)));
  CUDA_CHECK(cudaStreamWaitEvent(compatibility, compatibilityEvent, 0));
  CUDA_CHECK(cudaLaunchHostFunc(compatibility, Callback::run, &callbacks[1]));
  // Import a compatibility stream as streamCopy does in the native adaptor.
  HGGC_CHECK(hggcEventRecord(nativeEvent,
                             reinterpret_cast<hggcStream_t>(compatibility)));
  HGGC_CHECK(hggcStreamWaitEvent(native.base, nativeEvent, 0));
  HGGC_CHECK(hggcLaunchHostFunc(native.base, Callback::run, &callbacks[2]));
  HGGC_CHECK(hggcStreamSynchronize(native.base));
  CUDA_CHECK(cudaStreamSynchronize(compatibility));
  HGGC_CHECK(hggcEventDestroy(nativeEvent));
  CUDA_CHECK(cudaEventDestroy(compatibilityEvent));
  HGGC_CHECK(hggcStreamDestroy(native.base));
  CUDA_CHECK(cudaStreamDestroy(compatibility));
  if (failed.load() || value.load() != 3) {
    std::fprintf(stderr, "HGGC/CUDA compatibility stream ordering failed\n");
    return 1;
  }
  std::puts("HGGC/CUDA compatibility stream ordering passed");
  return 0;
}
