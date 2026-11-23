// SDK-only reproducer for the untracked-allocation IPC probe. No FlagCX/MPI.
// Raw VMM -> runtime IPC means creating/mapping memory with
// cuMemCreate/cuMemMap, then passing its address to cudaIpcGetMemHandle for
// traditional IPC export. CoreX currently crashes in that export; VMM
// allocation/mapping has succeeded.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#if defined(USE_ILUVATAR_ADAPTOR) && !FLAGCX_ILUVATAR_HAS_VMM
#define IPC_REPRO_HAS_VMM 0
#else
#define IPC_REPRO_HAS_VMM 1
#endif

template <typename F>
int stage(const char *name, F call) {
  std::fprintf(stderr, "IPC_REPRO begin %s\n", name);
  int result = static_cast<int>(call());
  std::fprintf(stderr, "IPC_REPRO end %s result=%d\n", name, result);
  return result;
}

void library(const char *name, const void *entry) {
  Dl_info info = {};
  if (dladdr(entry, &info))
    std::fprintf(stderr, "IPC_REPRO symbol=%s entry=%p library=%s\n", name,
                 entry, info.dli_fname ? info.dli_fname : "unknown");
  else
    std::fprintf(stderr, "IPC_REPRO symbol=%s entry=%p library=unresolved\n",
                 name, entry);
}

void context(const char *where) {
  // Resolve optional diagnostic APIs without adding a required SDK surface.
  using Current = CUresult (*)(CUcontext *);
  using Device = CUresult (*)(CUdevice *);
  auto current =
      reinterpret_cast<Current>(dlsym(RTLD_DEFAULT, "cuCtxGetCurrent"));
  auto device = reinterpret_cast<Device>(dlsym(RTLD_DEFAULT, "cuCtxGetDevice"));
  CUcontext ctx = nullptr;
  CUdevice ordinal = -1;
  const int ctxResult = current ? static_cast<int>(current(&ctx)) : -1;
  const int devResult = device ? static_cast<int>(device(&ordinal)) : -1;
  std::fprintf(
      stderr, "IPC_REPRO context=%s current=%p result=%d device=%d result=%d\n",
      where, reinterpret_cast<void *>(ctx), ctxResult,
      static_cast<int>(ordinal), devResult);
}

struct Allocation {
  void *ordinary = nullptr;
#if IPC_REPRO_HAS_VMM
  CUdeviceptr base = 0;
  CUmemGenericAllocationHandle physical = 0;
  size_t bytes = 0;
  bool mapped = false;
#endif
  bool cleanupAttempted = false;
  bool cleanup() {
    cleanupAttempted = true;
    if (ordinary) {
      if (stage("cudaFree", [&] { return cudaFree(ordinary); }) != 0)
        return false;
      ordinary = nullptr;
    }
#if IPC_REPRO_HAS_VMM
    if (mapped) {
      if (stage("cuMemUnmap", [&] { return cuMemUnmap(base, bytes); }) != 0)
        return false;
      mapped = false;
    }
    if (base) {
      if (stage("cuMemAddressFree",
                [&] { return cuMemAddressFree(base, bytes); }) != 0)
        return false;
      base = 0;
    }
    if (physical) {
      if (stage("cuMemRelease", [&] { return cuMemRelease(physical); }) != 0)
        return false;
      physical = 0;
    }
#endif
    return true;
  }
  ~Allocation() {
    if (!cleanupAttempted)
      cleanup();
  }
};

int main(int argc, char **argv) {
  std::setvbuf(stderr, nullptr, _IONBF, 0);
  if (argc != 2 || (std::strcmp(argv[1], "malloc") != 0 &&
                    std::strcmp(argv[1], "vmm") != 0 &&
                    std::strcmp(argv[1], "vmm-driver") != 0 &&
                    std::strcmp(argv[1], "vmm-sync") != 0)) {
    std::fprintf(stderr, "Usage: %s malloc|vmm|vmm-driver|vmm-sync\n", argv[0]);
    return 2;
  }
  const bool vmm = std::strcmp(argv[1], "malloc") != 0;
  const bool driver = std::strcmp(argv[1], "vmm-driver") == 0;
  using DriverIpc = CUresult (*)(CUipcMemHandle *, CUdeviceptr);
  auto driverIpc =
      reinterpret_cast<DriverIpc>(dlsym(RTLD_DEFAULT, "cuIpcGetMemHandle"));
  std::fprintf(
      stderr,
      "IPC_REPRO case=%s runtimeHandleBytes=%zu driverHandleBytes=%zu\n",
      argv[1], sizeof(cudaIpcMemHandle_t), sizeof(CUipcMemHandle));
  library("cudaIpcGetMemHandle",
          reinterpret_cast<const void *>(&cudaIpcGetMemHandle));
  library("cuIpcGetMemHandle", reinterpret_cast<const void *>(driverIpc));
  library("cuMemGetAddressRange",
          reinterpret_cast<const void *>(&cuMemGetAddressRange));
  int runtimeVersion = 0, driverVersion = 0;
  const auto rt = cudaRuntimeGetVersion(&runtimeVersion);
  const auto dr = cudaDriverGetVersion(&driverVersion);
  std::fprintf(
      stderr,
      "IPC_REPRO runtimeVersion=%d result=%d driverVersion=%d result=%d\n",
      runtimeVersion, static_cast<int>(rt), driverVersion,
      static_cast<int>(dr));
  if (stage("cudaSetDevice(0)", [] { return cudaSetDevice(0); }) != 0)
    return 1;
  context("before-allocation");
  Allocation allocation;
  void *pointer = nullptr;
  size_t requested = 2 * 1024 * 1024;
  if (!vmm) {
    if (stage("cudaMalloc",
              [&] { return cudaMalloc(&allocation.ordinary, requested); }) != 0)
      return 1;
    pointer = allocation.ordinary;
  } else {
#if IPC_REPRO_HAS_VMM
    CUdevice device = 0;
    if (stage("cuDeviceGet", [&] { return cuDeviceGet(&device, 0); }) != 0)
      return 1;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    if (stage("cuMemGetAllocationGranularity",
              [&] {
                return cuMemGetAllocationGranularity(
                    &requested, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
              }) != 0 ||
        requested == 0)
      return 1;
    allocation.bytes = requested;
    // Match the raw mapping created by UntrackedVmmMappingIsRejected.
    if (stage("cuMemCreate",
              [&] {
                return cuMemCreate(&allocation.physical, requested, &prop, 0);
              }) != 0 ||
        stage("cuMemAddressReserve",
              [&] {
                return cuMemAddressReserve(&allocation.base, requested,
                                           requested, 0, 0);
              }) != 0 ||
        stage("cuMemMap", [&] {
          return cuMemMap(allocation.base, requested, 0, allocation.physical,
                          0);
        }) != 0)
      return 1;
    allocation.mapped = true;
    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (stage("cuMemSetAccess", [&] {
          return cuMemSetAccess(allocation.base, requested, &access, 1);
        }) != 0)
      return 1;
    pointer = reinterpret_cast<void *>(allocation.base);
    if (std::strcmp(argv[1], "vmm-sync") == 0) {
      unsigned sync = 1;
      if (stage("cuPointerSetAttribute(SYNC_MEMOPS)", [&] {
            return cuPointerSetAttribute(
                &sync, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, allocation.base);
          }) != 0)
        return 1;
    }
#else
    std::fprintf(stderr, "IPC_REPRO VMM declarations unavailable\n");
    return 2;
#endif
  }
  context("before-range");
  CUdeviceptr base = 0;
  size_t bytes = 0;
  const int range = stage("cuMemGetAddressRange", [&] {
    return cuMemGetAddressRange(&base, &bytes,
                                reinterpret_cast<CUdeviceptr>(pointer));
  });
  std::fprintf(
      stderr,
      "IPC_REPRO pointer=%p requested=%zu rangeBase=0x%llx rangeBytes=%zu\n",
      pointer, requested, static_cast<unsigned long long>(base), bytes);
  const auto address = reinterpret_cast<CUdeviceptr>(pointer);
  bool passed = false;
  if (range != 0 || address < base || address - base > bytes ||
      requested > bytes - (address - base)) {
    // CUDA's describe path also stops here; do not claim IPC was exercised.
    passed = vmm && (range == CUDA_ERROR_INVALID_VALUE ||
                     range == CUDA_ERROR_NOT_SUPPORTED);
    std::fprintf(stderr, "IPC_REPRO rejected-before-ipc rangeResult=%d\n",
                 range);
  } else {
    context("before-ipc");
    int result = 0;
    if (driver) {
      if (!driverIpc) {
        std::fprintf(stderr, "IPC_REPRO driver IPC entry unavailable\n");
        return 2;
      }
      CUipcMemHandle handle;
      result =
          stage("cuIpcGetMemHandle", [&] { return driverIpc(&handle, base); });
    } else {
      // Output storage matches the CUDA adaptor; do not pre-initialize it.
      cudaIpcMemHandle_t handle;
      result = stage("cudaIpcGetMemHandle", [&] {
        return cudaIpcGetMemHandle(&handle, reinterpret_cast<void *>(base));
      });
    }
    passed = vmm ? result != 0 : result == 0;
    std::fprintf(stderr, "IPC_REPRO ipc-returned result=%d expected=%s\n",
                 result, vmm ? "rejection" : "success");
  }
  const bool cleaned = allocation.cleanup();
  if (!passed || !cleaned)
    return 1;
  std::fprintf(stderr, "IPC_REPRO_PASS case=%s\n", argv[1]);
  return 0;
}
