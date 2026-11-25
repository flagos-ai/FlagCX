/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/
#include "diagnostics.h"

#include <mcr/mc_runtime.h>
#include <sys/resource.h>

// No FlagCX, MPI or verbs linkage: this is the SDK startup control, separate
// from the full workers and their library constructors.
int main(int argc, char **argv) {
  probeBoot();
  rlimit noCore = {0, 0};
  setrlimit(RLIMIT_CORE, &noCore);
  probeEmit({{"event", "main_enter"}, {"worker", "sdk_inventory"}});
  probeLibraries();
  if (argc != 2)
    return 2;
  const std::string operation = argv[1];
  if (operation == "--count") {
    int count = 0;
    const auto code = probeTimed("mcGetDeviceCount",
                                 [&] { return mcGetDeviceCount(&count); });
    probeEmit({{"event", "gpu_inventory"},
               {"gpu_count_code", int(code)},
               {"gpu_count", count}});
    return code == mcSuccess ? 0 : 1;
  }
  if (operation == "--runtime-version" || operation == "--driver-version") {
    int version = 0;
    const auto code =
        operation == "--runtime-version"
            ? probeTimed("mcRuntimeGetVersion",
                         [&] { return mcRuntimeGetVersion(&version); })
            : probeTimed("mcDriverGetVersion",
                         [&] { return mcDriverGetVersion(&version); });
    probeEmit({{"event", "sdk_version"},
               {"query", operation},
               {"code", int(code)},
               {"version", version}});
    return code == mcSuccess ? 0 : 1;
  }
  return 2;
}
