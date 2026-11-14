// Separate regression probe; does not alter the public T1--T4 cases.
#include "flagcx.h"
#include "device_api/comm_traits.h"
#include "flagcx_device_adaptor.h"
#include "musa_fifo_errors.h"
#include <musa_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

extern "C" struct flagcxDeviceAdaptor *deviceAdaptor;

static void check(musaError_t result, const char *operation) {
  if (result != musaSuccess) {
    std::fprintf(stderr, "FAIL %s: %s\n", operation, musaGetErrorString(result));
    std::_Exit(1);
  }
}
static void expect(flagcxResult_t result, flagcxResult_t wanted,
                   const char *operation) {
  if (result != wanted) {
    std::fprintf(stderr, "FAIL %s: result=%d expected=%d\n", operation,
                 int(result), int(wanted));
    std::_Exit(1);
  }
}
#define NATIVE(op) check((op), #op)

__global__ void reportError(DeviceAPI::Comm comm, int fail) {
  comm.require(fail == 0, flagcxInvalidUsage);
  if (fail) comm.require(false, flagcxInternalError); // First error is sticky.
}

int main(int argc, char **argv) {
  NATIVE(musaSetDevice(argc > 1 ? std::atoi(argv[1]) : 0));
  musaFuncAttributes attributes{};
  NATIVE(musaFuncGetAttributes(&attributes, (const void *)reportError));
  std::printf("reportError: maxThreads=%d regs=%d local=%zu\n",
              attributes.maxThreadsPerBlock, attributes.numRegs,
              attributes.localSizeBytes);
  if (attributes.maxThreadsPerBlock < 1) return 1;

  constexpr size_t words = 3 * flagcxFifoIdxData;
  uint64_t *host = nullptr, *device = nullptr;
  NATIVE(musaHostAlloc((void **)&host, words * sizeof(uint64_t),
                       musaHostAllocMapped));
  NATIVE(musaHostGetDevicePointer((void **)&device, host, 0));
  // First word is a per-Comm error word; next two regions model actual FIFOs.
  auto *fifo0 = host + flagcxFifoIdxData;
  auto *fifo1 = host + 2 * flagcxFifoIdxData;
  for (auto *word : {host, fifo0 + flagcxFifoIdxTerminalStatus,
                    fifo1 + flagcxFifoIdxTerminalStatus})
    expect(flagcxMusaRegisterFifoError(word), flagcxSuccess, "register");

  DeviceAPI::Comm comm{};
  comm.deviceErrorStatus = device;
  // Test both an Intra-only communicator with no FIFOs and a Proxy Comm.
  for (int contexts : {0, 2}) {
    comm.contextCount = contexts;
    comm.fifoBuffers[0] = device + flagcxFifoIdxData;
    comm.fifoBuffers[1] = device + 2 * flagcxFifoIdxData;
    for (int fail : {0, 1}) {
      for (size_t i = 0; i < words; ++i) host[i] = 0;
      reportError<<<1, 1>>>(comm, fail);
      NATIVE(musaGetLastError());
      NATIVE(musaDeviceSynchronize()); // Native execution itself must succeed.
      const auto wanted = fail ? flagcxInvalidUsage : flagcxSuccess;
      expect(deviceAdaptor->deviceSynchronize(), wanted, "Host error propagation");
      expect(deviceAdaptor->deviceSynchronize(), wanted, "sticky query");
      if (host[0] != uint64_t(wanted)) return 1;
      if (contexts && (fifo0[flagcxFifoIdxTerminalStatus] != uint64_t(wanted) ||
                       fifo1[flagcxFifoIdxTerminalStatus] != uint64_t(wanted)))
        return 1;
    }
  }
  for (auto *word : {host, fifo0 + flagcxFifoIdxTerminalStatus,
                    fifo1 + flagcxFifoIdxTerminalStatus})
    flagcxMusaUnregisterFifoError(word);
  NATIVE(musaFreeHost(host));
  expect(deviceAdaptor->deviceSynchronize(), flagcxSuccess, "unregistered owner");
  std::puts("MUSA device error propagation PASS");
  return 0;
}
