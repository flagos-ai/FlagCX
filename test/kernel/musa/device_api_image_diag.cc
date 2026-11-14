// Host-only entry point: inspect the real device_api.o without Comm/IPC setup.
#include <musa_runtime.h>
#include <cstdio>
#include <cstdlib>

extern "C" int flagcxMusaDiagnoseDeviceApiImage();

int main(int argc, char **argv) {
  char *end = nullptr;
  const long ordinal = argc == 2 ? std::strtol(argv[1], &end, 10) : -1;
  if (argc != 2 || !argv[1][0] || !end || *end || ordinal < 0 || ordinal > 65535) {
    std::fprintf(stderr, "Usage: %s <GPU ordinal>\n", argv[0]);
    return 1;
  }
  musaError_t result = musaSetDevice(static_cast<int>(ordinal));
  if (result != musaSuccess) {
    std::fprintf(stderr, "setDevice=%d (%s)\n", int(result), musaGetErrorString(result));
    return 1;
  }
  // Establish the runtime context before looking up functions; no GPU kernel
  // or IPC operation is required by this allocation.
  void *allocation = nullptr;
  result = musaMalloc(&allocation, sizeof(int));
  if (result != musaSuccess) {
    std::fprintf(stderr, "context allocation=%d (%s)\n", int(result), musaGetErrorString(result));
    return 1;
  }
  const int diagnostic = flagcxMusaDiagnoseDeviceApiImage();
  result = musaFree(allocation);
  if (result != musaSuccess) {
    std::fprintf(stderr, "context allocation free=%d (%s)\n", int(result), musaGetErrorString(result));
    return 1;
  }
  std::fprintf(stderr, "Attribute query diagnostic only; no Kernel launch or functional PASS claimed.\n");
  return diagnostic;
}
