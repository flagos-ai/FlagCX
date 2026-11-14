// Driver-only diagnostic: no libflagcx, musart, host stubs or IPC dependency.
#include <musa.h>
#include <cstdio>
#include <cstdlib>

static bool report(MUresult result, const char *operation) {
  const char *description = nullptr;
  (void)muGetErrorString(result, &description);
  std::fprintf(stderr, "%s: result=%d (%s)\n", operation, int(result),
               description ? description : "no driver error text");
  std::fflush(stderr);
  return result == MUSA_SUCCESS;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::fprintf(stderr, "Usage: %s <GPU ordinal> <mubin path> <device symbol> [more symbols...]\n", argv[0]);
    return 1;
  }
  char *end = nullptr;
  const long ordinal = std::strtol(argv[1], &end, 10);
  if (!argv[1][0] || *end || ordinal < 0 || ordinal > 65535) return 1;
  if (!report(muInit(0), "muInit")) return 1;
  MUdevice device;
  if (!report(muDeviceGet(&device, static_cast<int>(ordinal)), "muDeviceGet")) return 1;
  MUcontext context = nullptr;
  if (!report(muCtxCreate(&context, 0, device), "muCtxCreate")) return 1;
  MUmodule module = nullptr;
  std::fprintf(stderr, "Loading GPU%ld image=%s\n", ordinal, argv[2]);
  const bool loaded = report(muModuleLoad(&module, argv[2]), "muModuleLoad");
  bool passed = loaded;
  if (loaded) {
    for (int i = 3; i < argc; ++i) {
      MUfunction function = nullptr;
      std::fprintf(stderr, "Looking up symbol=%s\n", argv[i]);
      if (!report(muModuleGetFunction(&function, module, argv[i]), "muModuleGetFunction"))
        passed = false;
      else
        std::fprintf(stderr, "Resolved function=%p\n", reinterpret_cast<void *>(function));
    }
    if (!report(muModuleUnload(module), "muModuleUnload")) passed = false;
  }
  if (!report(muCtxDestroy(context), "muCtxDestroy")) passed = false;
  std::fprintf(stderr, "Driver load/lookup only; no Kernel execution or functional PASS claimed.\n");
  return passed ? 0 : 1;
}
