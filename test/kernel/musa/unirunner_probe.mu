// Native SDK and device capability checks required by MUSA UniRunner CI.
#include <musa.h>
#include <musa/atomic>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include <cstdint>
#include <cstdio>

#define MUSA_CHECK(call)                                                       \
  do {                                                                         \
    const musaError_t result = (call);                                         \
    if (result != musaSuccess) {                                               \
      std::fprintf(stderr, "%s failed: %s\n", #call,                           \
                   musaGetErrorString(result));                                \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define DRIVER_CHECK(call)                                                     \
  do {                                                                         \
    const MUresult result = (call);                                            \
    if (result != MUSA_SUCCESS) {                                              \
      std::fprintf(stderr, "%s failed: %d\n", #call,                           \
                   static_cast<int>(result));                                  \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__global__ void probeKernel(uint64_t *word) {
  if (threadIdx.x == 0)
    musa::atomic_ref<uint64_t, musa::thread_scope_system>{*word}.store(
        42, musa::memory_order_release);
}

int main() {
  int devices = 0;
  MUSA_CHECK(musaGetDeviceCount(&devices));
  if (devices < 4) {
    std::fprintf(stderr, "MUSA experimental CI requires four visible GPUs\n");
    return 1;
  }
  for (int device = 0; device < 4; ++device) {
    MUSA_CHECK(musaSetDevice(device));
    int cooperative = 0, concurrent = 0, sms = 0, maxThreads = 0, blocks = 0;
    MUSA_CHECK(musaDeviceGetAttribute(&cooperative,
                                      musaDevAttrCooperativeLaunch, device));
    MUSA_CHECK(musaDeviceGetAttribute(&concurrent, musaDevAttrConcurrentKernels,
                                      device));
    MUSA_CHECK(
        musaDeviceGetAttribute(&sms, musaDevAttrMultiProcessorCount, device));
    MUSA_CHECK(musaDeviceGetAttribute(&maxThreads,
                                      musaDevAttrMaxThreadsPerBlock, device));
    MUSA_CHECK(musaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        &blocks, reinterpret_cast<const void *>(probeKernel), 256, 0,
        musaOccupancyDefault));
    MUdevice nativeDevice;
    int stream64 = 0;
    DRIVER_CHECK(muDeviceGet(&nativeDevice, device));
    DRIVER_CHECK(muDeviceGetAttribute(
        &stream64, MU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS,
        nativeDevice));
    std::printf("MUSA device=%d cooperative=%d concurrent=%d sms=%d "
                "max_threads=%d active_blocks=%d stream64=%d\n",
                device, cooperative, concurrent, sms, maxThreads, blocks,
                stream64);
    if (!cooperative || !concurrent || sms <= 1 || maxThreads < 256 ||
        blocks <= 0 || !stream64)
      return 1;

    // IPC operates across all local ranks, including logical cluster edges.
    for (int target = 0; target < 4; ++target) {
      if (target == device)
        continue;
      int peer = 0;
      MUSA_CHECK(musaDeviceCanAccessPeer(&peer, device, target));
      if (!peer) {
        std::fprintf(stderr, "MUSA devices %d and %d require P2P\n", device,
                     target);
        return 1;
      }
    }

    uint64_t *word = nullptr;
    musaStream_t stream;
    MUSA_CHECK(musaMalloc(reinterpret_cast<void **>(&word), sizeof(*word)));
    MUSA_CHECK(musaStreamCreate(&stream));
    void *args[] = {&word};
    MUSA_CHECK(
        musaLaunchCooperativeKernel(reinterpret_cast<const void *>(probeKernel),
                                    dim3(1), dim3(256), args, 0, stream));
    const auto address = reinterpret_cast<MUdeviceptr>(word);
    const auto nativeStream = reinterpret_cast<MUstream>(stream);
    DRIVER_CHECK(muStreamWaitValue64(nativeStream, address, 42,
                                     MU_STREAM_WAIT_VALUE_GEQ));
    DRIVER_CHECK(muStreamWriteValue64(nativeStream, address, 43,
                                      MU_STREAM_WRITE_VALUE_DEFAULT));
    MUSA_CHECK(musaStreamSynchronize(stream));
    uint64_t observed = 0;
    MUSA_CHECK(
        musaMemcpy(&observed, word, sizeof(observed), musaMemcpyDeviceToHost));
    if (observed != 43)
      return 1;
    // Verify that this allocation can be exported; multi-process open and
    // rebind correctness is covered by the experimental IPC tests.
    musaIpcMemHandle_t handle;
    MUSA_CHECK(musaIpcGetMemHandle(&handle, word));
    MUSA_CHECK(musaStreamDestroy(stream));
    MUSA_CHECK(musaFree(word));
  }
  return 0;
}
