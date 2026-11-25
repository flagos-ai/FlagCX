/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 ************************************************************************/
#include "probe.h"

#include <map>
#include <sys/resource.h>

int main(int argc, char **argv) {
  rlimit noCore = {0, 0};
  setrlimit(RLIMIT_CORE, &noCore);
  Options o;
#ifdef FLAGCX_PROBE_NATIVE
  const char *worker = "native";
#else
  const char *worker = "adaptor";
#endif
  if (argc == 2 && std::string(argv[1]) == "--inventory") {
    emit(o, inventory());
    return 0;
  }
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &o.rank);
  MPI_Comm_size(MPI_COMM_WORLD, &o.ranks);
  try {
    std::map<std::string, std::string> args;
    if (argc % 2 != 1)
      fail(o, "arguments", "SETUP_ERROR", 0, "expected --option value pairs");
    for (int i = 1; i + 1 < argc; i += 2)
      args[argv[i]] = argv[i + 1];
    auto number = [&](const char *key, int fallback) {
      return args.count(key) ? std::stoi(args[key]) : fallback;
    };
    auto text = [&](const char *key, const char *fallback) {
      return args.count(key) ? args[key] : std::string(fallback);
    };
    o.layer = text("--layer", worker);
    o.operation = text("--operation", "fetch_add");
    o.hca = text("--hca", "auto");
    o.registration = text("--registration", "mr");
    o.mode = number("--mode", 0);
    o.bytes = number("--bytes", 4096);
    o.offset = number("--offset", 0);
    o.qps = number("--qps", 1);
    o.depth = number("--depth", 16);
    o.ro = number("--ro", 0);
    o.direction = number("--direction", 0);
    o.iterations = number("--iterations", 8);
    o.timeout = number("--poll-timeout", 4);
    if ((o.ranks != 2 && o.ranks != 4) || o.mode < 0 || o.mode > 9 ||
        o.qps < 1 || o.qps > kMaxQps || o.iterations < 1 || o.iterations > 32 ||
        o.depth < 1 || o.depth > 255 || o.ro < 0 || o.ro > 1 ||
        o.timeout <= 0 || o.bytes > 1024 * 1024 ||
        (o.registration != "mr" && o.registration != "iova2") ||
        o.direction < 0 || o.direction > 1 || o.offset % 8 ||
        o.offset > o.bytes || o.bytes - o.offset < 8 ||
        (o.ranks == 4 && o.operation != "fetch_add") ||
        (o.operation != "alloc" && o.operation != "write" &&
         o.operation != "read" && o.operation != "fetch_add"))
      fail(o, "arguments", "SETUP_ERROR", 0, "invalid probe geometry");
    const char *hostSignals = std::getenv("FLAGCX_SIGNAL_HOST_ENABLE");
    if (hostSignals && std::strcmp(hostSignals, "0"))
      fail(o, "environment", "SETUP_ERROR", 0,
           "HOST SIGNAL is forbidden in this probe");
    emit(o, {{"event", "start"},
             {"layer", o.layer},
             {"worker", worker},
             {"mode", o.mode},
             {"operation", o.operation},
             {"bytes", o.bytes},
             {"offset", o.offset},
             {"direction", o.direction},
             {"ranks", o.ranks}});
    int gpus = 0;
    checkMaca(o, "device_count", mcGetDeviceCount(&gpus));
    if (gpus < o.ranks)
      fail(o, "device_count", "SETUP_ERROR", gpus,
           "one distinct GPU per rank is required");
    o.gpu = o.rank;
    checkMaca(o, "set_device", mcSetDevice(o.gpu));
    char pci[64] = {};
    mcError_t pciResult = mcDeviceGetPCIBusId(pci, sizeof(pci), o.gpu);
    emit(o, {{"event", "gpu"},
             {"device", o.gpu},
             {"pci_code", int(pciResult)},
             {"pci_bus_id", pci}});
    Allocation allocation;
    allocation.create(o);
    if (o.operation == "alloc") {
      emit(o, {{"event", "verify"},
               {"actual", allocation.read(o)},
               {"expected", initial(o)}});
      if (allocation.read(o) != initial(o))
        fail(o, "initialize_readback", "VERIFY_ERROR", 0,
             "allocation copy round trip failed");
    } else {
      if (o.layer != worker)
        fail(o, "arguments", "SETUP_ERROR", 0, "wrong worker for probe layer");
#ifdef FLAGCX_PROBE_NATIVE
      runNative(o, allocation);
#else
      runAdaptor(o, allocation);
#endif
    }
    allocation.release(o);
    MPI_Barrier(MPI_COMM_WORLD);
    loadedLibraries(o);
    emit(o, {{"event", "result"}, {"status", "PASS"}});
    MPI_Finalize();
    return 0;
  } catch (const Failure &error) {
    emit(o, {{"event", "result"},
             {"status", error.status},
             {"detail", error.what()}});
  } catch (const std::exception &error) {
    emit(o, {{"event", "result"},
             {"status", "INTERNAL_ERROR"},
             {"detail", error.what()}});
  }
  // Do not enter collective cleanup after a failed rank or a poisoned QP.
  // The supervisor also bounds MPI_Abort and all surviving descendants.
  loadedLibraries(o);
  MPI_Abort(MPI_COMM_WORLD, 1);
  return 1;
}
