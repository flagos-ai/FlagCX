// Native MACA diagnostic. Each invocation owns one allocation and one stream.
// The Python supervisor enforces the deadline, including driver cleanup calls.
#include <mcr/mc_runtime.h>

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/resource.h>

#include "metax_alloc_config.h"

static bool record(const char *stage, mcError_t error) {
  printf("STAGE\t%s\t%d\t%s\n", stage, int(error), mcGetErrorString(error));
  return error == mcSuccess;
}

static bool finish(const char *result) {
  printf("RESULT\t%s\n", result);
  return strcmp(result, "PASS") == 0;
}

static void metadata(int device) {
#if PROBE_RUNTIME_VERSION
  int runtime = 0;
  if (record("runtime_version", mcRuntimeGetVersion(&runtime)))
    printf("INFO\truntime_version\t%d\n", runtime);
#endif
#if PROBE_DRIVER_VERSION
  int driver = 0;
  if (record("driver_version", mcDriverGetVersion(&driver)))
    printf("INFO\tdriver_version\t%d\n", driver);
#endif
  mcDeviceProp_t props = {};
  if (record("device_properties", mcGetDeviceProperties(&props, device)))
    printf("INFO\tgpu\t%s\n", props.name);
}

static mcError_t writeValue(mcStream_t stream, void *ptr, uint64_t value,
                            int width) {
  if (width == 32) {
#if PROBE_WRITE32
    return mcStreamWriteValue32(
        stream, ptr, uint32_t(value),
        mcStreamWriteValue_flags::MC_STREAM_WRITE_VALUE_DEFAULT);
#endif
  } else {
#if PROBE_WRITE64
    return mcStreamWriteValue64(
        stream, ptr, value,
        mcStreamWriteValue_flags::MC_STREAM_WRITE_VALUE_DEFAULT);
#endif
  }
  return mcErrorNotSupported;
}

static mcError_t waitValue(mcStream_t stream, void *ptr, uint64_t value,
                           int width) {
  if (width == 32) {
#if PROBE_WAIT32
    return mcStreamWaitValue32(
        stream, ptr, uint32_t(value),
        mcStreamWaitValue_flags::MC_STREAM_WAIT_VALUE_GEQ);
#endif
  } else {
#if PROBE_WAIT64
    return mcStreamWaitValue64(
        stream, ptr, value, mcStreamWaitValue_flags::MC_STREAM_WAIT_VALUE_GEQ);
#endif
  }
  return mcErrorNotSupported;
}

int main(int argc, char **argv) {
  const struct rlimit noCore = {0, 0};
  setrlimit(RLIMIT_CORE, &noCore);
  setvbuf(stdout, nullptr, _IONBF, 0);
  setvbuf(stderr, nullptr, _IONBF, 0);
  if (argc == 3 && strcmp(argv[1], "--info") == 0) {
    int device = atoi(argv[2]);
    if (!record("set_device", mcSetDevice(device)))
      return 1;
    metadata(device);
    return 0;
  }
  if (argc != 6)
    return 2;
  const int mode = atoi(argv[1]);
  const size_t bytes = strtoull(argv[2], nullptr, 10);
  const int width = atoi(argv[3]);
  const char *operation = argv[4];
  const int device = atoi(argv[5]);
  const bool doWrite = strcmp(operation, "wait") != 0;
  const bool doWait = strcmp(operation, "write") != 0;
  const size_t wordBytes = width / 8;
  // Distinct upper/lower halves detect a truncated 64-bit store.
  const uint64_t expected =
      width == 64 ? UINT64_C(0x123456789abcdef0) : UINT64_C(0x9abcdef0);
  if (!record("set_device", mcSetDevice(device))) {
    finish("DEVICE_ERROR");
    return 1;
  }
  void *ptr = nullptr;
  mcError_t allocation = allocateMode(mode, &ptr, bytes);
  const bool allocated = record("allocate", allocation);
  if (!allocated || ptr == nullptr) {
    finish("ALLOC_ERROR");
    return 1;
  }
  printf("INFO\tpointer\t%p\n", ptr);
  printf("INFO\talignment_mod8\t%zu\n", size_t(uintptr_t(ptr) % 8));
  mcPointerAttribute_t attributes = {};
  // Attribute-query failure remains visible but does not hide memops results.
  if (record("pointer_attributes", mcPointerGetAttributes(&attributes, ptr))) {
    printf("INFO\tdevice_pointer\t%p\n", attributes.devicePointer);
    printf("INFO\thost_pointer\t%p\n", attributes.hostPointer);
    printf("INFO\tpointer_device\t%d\n", attributes.device);
#if CUDART_VERSION >= 10000
    printf("INFO\tmemory_type\t%d\n", int(attributes.type));
#else
    printf("INFO\tmemory_type\t%d\n", int(attributes.memoryType));
    printf("INFO\tis_managed\t%d\n", int(attributes.isManaged));
#endif
  }
  mcStream_t stream = nullptr;
  const char *result = "PASS";
  printf("INFO\tstream_flags\t%u\n", unsigned(mcStreamNonBlocking));
  bool streamCreated = record(
      "stream_create", mcStreamCreateWithFlags(&stream, mcStreamNonBlocking));
  bool queued = false;
  bool safeToFree = true;
  if (!streamCreated) {
    result = "SETUP_ERROR";
  } else {
    uint64_t initial = doWrite ? 0 : expected;
    // Wait-only cases start satisfied. A write failure never queues a wait.
    bool initialized = record(
        "initialize", mcMemcpy(ptr, &initial, wordBytes, mcMemcpyHostToDevice));
    // Match the failing adaptor tests: establish initialization completion
    // before a nonblocking stream accesses the word.
    if (initialized)
      initialized = record("initialize_sync", mcDeviceSynchronize());
    if (!initialized) {
      result = "INIT_ERROR";
    } else {
      bool ok = true;
      if (doWrite) {
        ok = record("write", writeValue(stream, ptr, expected, width));
        queued = ok;
      }
      if (ok && doWait) {
        ok = record("wait", waitValue(stream, ptr, expected, width));
        queued = queued || ok;
      }
      if (!ok)
        result = "API_ERROR";
      if (queued) {
        safeToFree = record("synchronize", mcStreamSynchronize(stream));
        if (!safeToFree)
          result = "SYNC_ERROR";
      }
      if (ok && safeToFree) {
        uint64_t actual = 0;
        if (!record("readback",
                    mcMemcpy(&actual, ptr, wordBytes, mcMemcpyDeviceToHost))) {
          result = "COPY_ERROR";
        } else {
          printf("INFO\texpected\t0x%016" PRIx64 "\n", expected);
          printf("INFO\tactual\t0x%016" PRIx64 "\n", actual);
          if (actual != expected)
            result = "DATA_MISMATCH";
        }
      }
    }
  }
  // Save the operation outcome before cleanup, which can itself hang/fail.
  printf("OP_RESULT\t%s\n", result);
  if (safeToFree) {
    if (streamCreated && !record("stream_destroy", mcStreamDestroy(stream)))
      result = "CLEANUP_ERROR";
    if (!record("free", mcFree(ptr)))
      result = "CLEANUP_ERROR";
  } else {
    // A failed synchronization does not establish buffer lifetime completion.
    printf("INFO\tcleanup\tdeferred to process exit after sync failure\n");
  }
  return finish(result) ? 0 : 1;
}
