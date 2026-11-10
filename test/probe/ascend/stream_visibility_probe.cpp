// Standalone CANN/MPI peer-write visibility probe; no FlagCX dependency.
#include "stream_consumer.h"

#include <acl/acl.h>
#include <dlfcn.h>
#include <mpi.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

constexpr size_t kAllocationBytes = 4096;
constexpr size_t kPayloadOffset = 64;
constexpr size_t kPayloadBytes = 256;
constexpr int32_t kStreamTimeoutMs = 10000;
constexpr uint8_t kExpectedByte = 0xA7;

// Driver SDK DVattribute layout. The probe resolves the symbol at runtime so
// it can still build in toolkit images that do not ship driver headers.
struct DriverMemAttributes {
  uint32_t memType;
  uint32_t reserved1;
  uint32_t reserved2;
  uint32_t deviceId;
  uint32_t pageSize;
  uint32_t reserved[8];
};
static_assert(sizeof(DriverMemAttributes) == 52);

void logDriverAttributes(int rank, const char *role, void *ptr) {
  void *library = dlopen("libascend_hal.so", RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr)
    library = dlopen("/usr/local/Ascend/driver/lib64/driver/libascend_hal.so",
                     RTLD_NOW | RTLD_LOCAL);
  if (library == nullptr) {
    const char *reason = dlerror();
    std::printf(
        "PROBE rank=%d operation=ipc_driver_attributes status=UNAVAILABLE "
        "role=%s reason=%s\n",
        rank, role, reason == nullptr ? "libascend_hal_not_found" : reason);
    return;
  }
  void *symbol = dlsym(library, "drvMemGetAttribute");
  if (symbol == nullptr) {
    const char *reason = dlerror();
    std::printf(
        "PROBE rank=%d operation=ipc_driver_attributes status=UNAVAILABLE "
        "role=%s reason=%s\n",
        rank, role,
        reason == nullptr ? "drvMemGetAttribute_not_found" : reason);
    dlclose(library);
    return;
  }
  using GetAttribute = int (*)(uint64_t, DriverMemAttributes *);
  auto getAttribute = reinterpret_cast<GetAttribute>(symbol);
  DriverMemAttributes attr = {};
  const int error = getAttribute(reinterpret_cast<uintptr_t>(ptr), &attr);
  Dl_info libraryInfo = {};
  const char *libraryPath =
      dladdr(symbol, &libraryInfo) != 0 && libraryInfo.dli_fname != nullptr
          ? libraryInfo.dli_fname
          : "unknown";
  Dl_info aclLibraryInfo = {};
  void *aclSymbol = dlsym(RTLD_DEFAULT, "aclrtPointerGetAttributes");
  const char *aclLibraryPath =
      aclSymbol != nullptr && dladdr(aclSymbol, &aclLibraryInfo) != 0 &&
              aclLibraryInfo.dli_fname != nullptr
          ? aclLibraryInfo.dli_fname
          : "unknown";
  std::printf("PROBE rank=%d operation=ipc_driver_attributes status=%s "
              "role=%s drv_error=%d mem_type=0x%x device=%u page_size=%u "
              "svm_device=%d uvm=%d driver_library=%s acl_library=%s\n",
              rank, error == 0 ? "PASS" : "FAIL", role, error,
              error == 0 ? attr.memType : 0, error == 0 ? attr.deviceId : 0,
              error == 0 ? attr.pageSize : 0,
              error == 0 && (attr.memType & 0x0004U) != 0,
              error == 0 && (attr.memType & 0x0200U) != 0, libraryPath,
              aclLibraryPath);
  dlclose(library);
}

enum class Result : int { pass = 0, unsupported = 1, fail = 2 };
enum class Mode { valueWrite, kernelReadback, kernelWait };

const char *name(Result result) {
  switch (result) {
    case Result::pass:
      return "PASS";
    case Result::unsupported:
      return "UNSUPPORTED";
    case Result::fail:
      return "FAIL";
  }
  return "FAIL";
}

Result aclCheck(aclError error, int rank, const char *operation) {
  if (error == ACL_SUCCESS)
    return Result::pass;
  Result result = error == ACL_ERROR_RT_FEATURE_NOT_SUPPORT
                      ? Result::unsupported
                      : Result::fail;
  std::fprintf(stderr, "PROBE rank=%d operation=%s status=%s acl_error=%d\n",
               rank, operation, name(result), static_cast<int>(error));
  const char *recent = aclGetRecentErrMsg();
  if (recent != nullptr && *recent != '\0')
    std::fprintf(stderr, "PROBE rank=%d operation=%s recent_error=%s\n", rank,
                 operation, recent);
  return result;
}

Result checkpoint(int rank, const char *stage, Result local) {
  int localCode = static_cast<int>(local);
  int globalCode = 0;
  if (MPI_Allreduce(&localCode, &globalCode, 1, MPI_INT, MPI_MAX,
                    MPI_COMM_WORLD) != MPI_SUCCESS)
    return Result::fail;
  Result global = static_cast<Result>(globalCode);
  if (rank == 0)
    std::printf("PROBE stage=%s status=%s\n", stage, name(global));
  return global;
}

void combine(Result *result, Result next) {
  if (static_cast<int>(next) > static_cast<int>(*result))
    *result = next;
}

void *offset(void *base, size_t bytes) {
  return static_cast<void *>(static_cast<unsigned char *>(base) + bytes);
}

bool parseArgs(int argc, char **argv, int *first, int *second, Mode *mode) {
  if ((argc != 3 && argc != 5) || std::strcmp(argv[1], "--devices") != 0)
    return false;
  int consumed = 0;
  if (std::sscanf(argv[2], "%d,%d%n", first, second, &consumed) != 2 ||
      consumed != static_cast<int>(std::strlen(argv[2])) || *first < 0 ||
      *second < 0)
    return false;
  if (argc == 3) {
    *mode = Mode::valueWrite;
    return true;
  }
  if (std::strcmp(argv[3], "--mode") != 0)
    return false;
  if (std::strcmp(argv[4], "valuewrite") == 0)
    *mode = Mode::valueWrite;
  else if (std::strcmp(argv[4], "kernel-readback") == 0)
    *mode = Mode::kernelReadback;
  else if (std::strcmp(argv[4], "kernel-wait") == 0)
    *mode = Mode::kernelWait;
  else
    return false;
  return true;
}

class LocalProbe {
public:
  explicit LocalProbe(int rank) : rank_(rank) {}

  Result run() {
    Result result = aclCheck(
        aclrtMalloc(&buffer_, kAllocationBytes, ACL_MEM_MALLOC_HUGE_FIRST),
        rank_, "aclrtMalloc(local buffer)");
    if (result != Result::pass)
      return result;
    result =
        aclCheck(aclrtMalloc(reinterpret_cast<void **>(&status_),
                             2 * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST),
                 rank_, "aclrtMalloc(local status)");
    if (result != Result::pass)
      return result;
    result = aclCheck(aclrtCreateStream(&stream_), rank_, "aclrtCreateStream");
    if (result != Result::pass)
      return result;
    result =
        aclCheck(aclrtMemset(buffer_, kAllocationBytes, 0, kAllocationBytes),
                 rank_, "aclrtMemset(local buffer)");
    if (result != Result::pass)
      return result;
    result = aclCheck(
        aclrtMemset(status_, 2 * sizeof(uint32_t), 0, 2 * sizeof(uint32_t)),
        rank_, "aclrtMemset(local status)");
    if (result != Result::pass)
      return result;

    // Same-stream smoke test checks both public value APIs and GEQ mode.
    launchStreamProducer(offset(buffer_, kPayloadOffset), kExpectedByte,
                         stream_);
    result = aclCheck(aclrtValueWrite(buffer_, 2, 0, stream_), rank_,
                      "aclrtValueWrite(local)");
    if (result != Result::pass)
      return result;
    result =
        aclCheck(aclrtValueWait(buffer_, 1, ACL_STREAM_WAIT_VALUE_GEQ, stream_),
                 rank_, "aclrtValueWait(local GEQ)");
    if (result != Result::pass)
      return result;
    launchStreamConsumer(offset(buffer_, kPayloadOffset), status_,
                         kExpectedByte, stream_);
    result =
        aclCheck(aclrtSynchronizeStreamWithTimeout(stream_, kStreamTimeoutMs),
                 rank_, "aclrtSynchronizeStreamWithTimeout(local)");
    if (result != Result::pass)
      return result;
    synced_ = true;
    std::array<uint32_t, 2> hostStatus = {};
    result =
        aclCheck(aclrtMemcpy(hostStatus.data(), sizeof(hostStatus), status_,
                             sizeof(hostStatus), ACL_MEMCPY_DEVICE_TO_HOST),
                 rank_, "aclrtMemcpy(local status)");
    if (result != Result::pass)
      return result;
    if (hostStatus[0] != 0 || hostStatus[1] != kPayloadBytes) {
      std::fprintf(stderr,
                   "PROBE rank=%d operation=local_consumer status=FAIL "
                   "mismatches=%u checked=%u\n",
                   rank_, hostStatus[0], hostStatus[1]);
      return Result::fail;
    }
    return Result::pass;
  }

  Result cleanup() {
    Result result = Result::pass;
    if (stream_ != nullptr)
      combine(&result, aclCheck(synced_ ? aclrtDestroyStream(stream_)
                                        : aclrtDestroyStreamForce(stream_),
                                rank_, "aclrtDestroyStream(local)"));
    if (result != Result::pass)
      return result;
    if (status_ != nullptr)
      combine(&result,
              aclCheck(aclrtFree(status_), rank_, "aclrtFree(local status)"));
    if (buffer_ != nullptr)
      combine(&result,
              aclCheck(aclrtFree(buffer_), rank_, "aclrtFree(local buffer)"));
    return result;
  }

private:
  int rank_;
  bool synced_ = false;
  void *buffer_ = nullptr;
  uint32_t *status_ = nullptr;
  aclrtStream stream_ = nullptr;
};

class RemoteProbe {
public:
  RemoteProbe(int rank, Mode mode) : rank_(rank), mode_(mode) {}

  Result run() {
    Result result =
        checkpoint(rank_, "receiver_allocation_and_export", prepareReceiver());
    if (result != Result::pass)
      return result;
    if (MPI_Bcast(key_.data(), static_cast<int>(key_.size()), MPI_CHAR, 1,
                  MPI_COMM_WORLD) != MPI_SUCCESS)
      return checkpoint(rank_, "ipc_key_exchange", Result::fail);
    result = checkpoint(rank_, "ipc_key_exchange", Result::pass);
    if (result != Result::pass)
      return result;

    result = checkpoint(rank_, "sender_import_and_streams", prepareSender());
    if (result != Result::pass)
      return result;
    const Result attributes =
        checkpoint(rank_, "ipc_pointer_attributes", logAttributes());
    if (mode_ == Mode::kernelWait) {
      result = checkpoint(rank_, "remote_wait_and_consumer_enqueued",
                          enqueueWaitAndConsumer());
      if (result != Result::pass)
        return result;
    }
    result = checkpoint(rank_, "remote_write_enqueued",
                        mode_ == Mode::valueWrite ? enqueueValueWrite()
                                                  : enqueueKernelWrite());
    if (result != Result::pass)
      return result;
    if (mode_ == Mode::kernelWait) {
      // Both streams progress concurrently. Report their completion states
      // separately after the synchronization calls have returned.
      const Result localCompletion =
          rank_ == 0 ? synchronizeWriter() : synchronizeReceiver();
      Result writerCompletion =
          checkpoint(rank_, "remote_writer_completed",
                     rank_ == 0 ? localCompletion : Result::pass);
      Result waitCompletion =
          checkpoint(rank_, "remote_wait_completed",
                     rank_ == 1 ? localCompletion : Result::pass);
      combine(&writerCompletion, waitCompletion);
      if (writerCompletion != Result::pass)
        return writerCompletion;
    } else {
      result =
          checkpoint(rank_, "remote_writer_completed", synchronizeWriter());
      if (result != Result::pass)
        return result;
    }
    if (mode_ == Mode::valueWrite)
      result = checkpoint(rank_, "remote_signal_readback", verifySignal());
    else if (mode_ == Mode::kernelReadback)
      result =
          checkpoint(rank_, "remote_kernel_readback", verifyKernelReadback());
    else
      result = checkpoint(rank_, "remote_device_consumer_data", verify());
    combine(&result, attributes);
    return result;
  }

  Result cleanup() {
    Result result = Result::pass;
    if (stream_ != nullptr)
      combine(&result, aclCheck(synced_ ? aclrtDestroyStream(stream_)
                                        : aclrtDestroyStreamForce(stream_),
                                rank_, "aclrtDestroyStream(remote)"));
    if (result != Result::pass)
      return checkpoint(rank_, "remote_sender_released", result);
    if (status_ != nullptr)
      combine(&result,
              aclCheck(aclrtFree(status_), rank_, "aclrtFree(remote status)"));
    if (imported_) {
      combine(&result, aclCheck(aclrtIpcMemClose(key_.data()), rank_,
                                "aclrtIpcMemClose(import)"));
      imported_ = false;
    }
    // The exporter cannot close/free while the importer may still be using
    // its key or while a sender stream may still be writing that allocation.
    Result peerReleased = checkpoint(rank_, "remote_sender_released", result);
    if (peerReleased != Result::pass)
      return Result::fail;
    if (exported_)
      combine(&result, aclCheck(aclrtIpcMemClose(key_.data()), rank_,
                                "aclrtIpcMemClose(export)"));
    if (owned_ != nullptr)
      combine(&result,
              aclCheck(aclrtFree(owned_), rank_, "aclrtFree(remote buffer)"));
    return checkpoint(rank_, "remote_cleanup", result);
  }

private:
  Result logAttributes() {
    void *ptr = rank_ == 1 ? owned_ : importedBase_;
    const char *role = rank_ == 1 ? "owner" : "importer";
    logDriverAttributes(rank_, role, ptr);
    aclrtPtrAttributes attr = {};
    const aclError error = aclrtPointerGetAttributes(ptr, &attr);
    Result result = aclCheck(error, rank_, "aclrtPointerGetAttributes(IPC)");
    std::printf("PROBE rank=%d operation=ipc_pointer_attributes status=%s "
                "role=%s address=%p aligned8=%d location=%d device=%u "
                "page_size=%u\n",
                rank_, name(result), role, ptr,
                reinterpret_cast<uintptr_t>(ptr) % alignof(uint64_t) == 0,
                result == Result::pass ? static_cast<int>(attr.location.type)
                                       : -1,
                result == Result::pass ? attr.location.id : 0,
                result == Result::pass ? attr.pageSize : 0);
    return result;
  }

  Result prepareReceiver() {
    if (rank_ != 1)
      return Result::pass;
    Result result = aclCheck(
        aclrtMalloc(&owned_, kAllocationBytes, ACL_MEM_MALLOC_HUGE_FIRST),
        rank_, "aclrtMalloc(receiver buffer)");
    if (result != Result::pass)
      return result;
    result =
        aclCheck(aclrtMemset(owned_, kAllocationBytes, 0, kAllocationBytes),
                 rank_, "aclrtMemset(receiver buffer)");
    if (result != Result::pass)
      return result;
    result = aclCheck(aclrtIpcMemGetExportKey(
                          owned_, kAllocationBytes, key_.data(), 65,
                          ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION),
                      rank_, "aclrtIpcMemGetExportKey");
    if (result != Result::pass)
      return result;
    exported_ = true;
    if (key_[0] == '\0' || std::memchr(key_.data(), '\0', 65) == nullptr)
      return Result::fail;
    result =
        aclCheck(aclrtMalloc(reinterpret_cast<void **>(&status_),
                             4 * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST),
                 rank_, "aclrtMalloc(remote status)");
    if (result != Result::pass)
      return result;
    return aclCheck(
        aclrtMemset(status_, 4 * sizeof(uint32_t), 0, 4 * sizeof(uint32_t)),
        rank_, "aclrtMemset(remote status)");
  }

  Result prepareSender() {
    if (rank_ == 0) {
      void *peerPointer = nullptr;
      Result result = aclCheck(
          aclrtIpcMemImportByKey(&peerPointer, key_.data(),
                                 ACL_RT_IPC_MEM_IMPORT_FLAG_ENABLE_PEER_ACCESS),
          rank_, "aclrtIpcMemImportByKey");
      if (result != Result::pass)
        return result;
      imported_ = true;
      importedBase_ = peerPointer;
    }
    return aclCheck(aclrtCreateStream(&stream_), rank_,
                    "aclrtCreateStream(remote)");
  }

  Result enqueueWaitAndConsumer() {
    if (rank_ != 1)
      return Result::pass;
    Result result =
        aclCheck(aclrtValueWait(owned_, 1, ACL_STREAM_WAIT_VALUE_GEQ, stream_),
                 rank_, "aclrtValueWait(remote GEQ)");
    if (result != Result::pass)
      return result;
    // Keep the consumer adjacent to ValueWait in the same stream. There is no
    // Host copy, stream sync, or device sync between these two tasks.
    launchStreamConsumer(offset(owned_, kPayloadOffset), status_, kExpectedByte,
                         stream_);
    return Result::pass;
  }

  Result enqueueValueWrite() {
    if (rank_ != 0)
      return Result::pass;
    // Value 2 deliberately tests the GEQ 1 condition instead of equality.
    return aclCheck(aclrtValueWrite(importedBase_, 2, 0, stream_), rank_,
                    "aclrtValueWrite(peer signal)");
  }

  Result enqueueKernelWrite() {
    if (rank_ != 0)
      return Result::pass;
    launchStreamProducer(offset(importedBase_, kPayloadOffset), kExpectedByte,
                         stream_);
    launchStreamSignal(importedBase_, 2, stream_);
    return Result::pass;
  }

  Result synchronizeWriter() {
    if (rank_ != 0)
      return Result::pass;
    return synchronizeStream();
  }

  Result synchronizeReceiver() {
    if (rank_ != 1)
      return Result::pass;
    return synchronizeStream();
  }

  Result synchronizeStream() {
    Result result =
        aclCheck(aclrtSynchronizeStreamWithTimeout(stream_, kStreamTimeoutMs),
                 rank_, "aclrtSynchronizeStreamWithTimeout(remote)");
    synced_ = result == Result::pass;
    return result;
  }

  Result verify() {
    if (rank_ != 1)
      return Result::pass;
    std::array<uint32_t, 2> hostStatus = {};
    Result result =
        aclCheck(aclrtMemcpy(hostStatus.data(), sizeof(hostStatus), status_,
                             sizeof(hostStatus), ACL_MEMCPY_DEVICE_TO_HOST),
                 rank_, "aclrtMemcpy(remote status)");
    if (result != Result::pass)
      return result;
    uint64_t signal = 0;
    result = aclCheck(aclrtMemcpy(&signal, sizeof(signal), owned_,
                                  sizeof(signal), ACL_MEMCPY_DEVICE_TO_HOST),
                      rank_, "aclrtMemcpy(remote signal)");
    if (result != Result::pass)
      return result;
    if (signal != 2 || hostStatus[0] != 0 || hostStatus[1] != kPayloadBytes) {
      std::fprintf(stderr,
                   "PROBE rank=%d operation=remote_consumer status=FAIL "
                   "signal=%llu mismatches=%u checked=%u\n",
                   rank_, static_cast<unsigned long long>(signal),
                   hostStatus[0], hostStatus[1]);
      return Result::fail;
    }
    return Result::pass;
  }

  Result verifySignal() {
    if (rank_ != 1)
      return Result::pass;
    uint64_t signal = 0;
    Result result =
        aclCheck(aclrtMemcpy(&signal, sizeof(signal), owned_, sizeof(signal),
                             ACL_MEMCPY_DEVICE_TO_HOST),
                 rank_, "aclrtMemcpy(IPC signal)");
    if (result != Result::pass)
      return result;
    std::printf(
        "PROBE rank=%d operation=ipc_signal_readback status=%s value=%llu\n",
        rank_, signal == 2 ? "PASS" : "FAIL",
        static_cast<unsigned long long>(signal));
    return signal == 2 ? Result::pass : Result::fail;
  }

  Result verifyKernelReadback() {
    if (rank_ != 1)
      return Result::pass;
    // Read on the device before the diagnostic Host copies can affect caches.
    launchStreamConsumer(offset(owned_, kPayloadOffset), status_, kExpectedByte,
                         stream_);
    launchStreamSignalReader(owned_, status_, stream_);
    Result result = synchronizeReceiver();
    if (result != Result::pass)
      return result;
    std::array<uint32_t, 4> hostStatus = {};
    result =
        aclCheck(aclrtMemcpy(hostStatus.data(), sizeof(hostStatus), status_,
                             sizeof(hostStatus), ACL_MEMCPY_DEVICE_TO_HOST),
                 rank_, "aclrtMemcpy(IPC device status)");
    if (result != Result::pass)
      return result;
    const uint64_t deviceSignal = static_cast<uint64_t>(hostStatus[2]) |
                                  (static_cast<uint64_t>(hostStatus[3]) << 32);
    std::printf("PROBE rank=%d operation=ipc_device_read status=%s "
                "signal=%llu mismatches=%u checked=%u\n",
                rank_,
                deviceSignal == 2 && hostStatus[0] == 0 &&
                        hostStatus[1] == kPayloadBytes
                    ? "PASS"
                    : "FAIL",
                static_cast<unsigned long long>(deviceSignal), hostStatus[0],
                hostStatus[1]);
    Result outcome = deviceSignal == 2 && hostStatus[0] == 0 &&
                             hostStatus[1] == kPayloadBytes
                         ? Result::pass
                         : Result::fail;
    uint64_t signal = 0;
    std::array<uint8_t, kPayloadBytes> payload = {};
    const Result signalCopy =
        aclCheck(aclrtMemcpy(&signal, sizeof(signal), owned_, sizeof(signal),
                             ACL_MEMCPY_DEVICE_TO_HOST),
                 rank_, "aclrtMemcpy(IPC signal)");
    const Result payloadCopy =
        aclCheck(aclrtMemcpy(payload.data(), payload.size(),
                             offset(owned_, kPayloadOffset), payload.size(),
                             ACL_MEMCPY_DEVICE_TO_HOST),
                 rank_, "aclrtMemcpy(IPC payload)");
    combine(&outcome, signalCopy);
    combine(&outcome, payloadCopy);
    if (signalCopy != Result::pass || payloadCopy != Result::pass)
      return outcome;
    unsigned mismatches = 0;
    for (uint8_t byte : payload)
      mismatches += byte != kExpectedByte;
    std::printf("PROBE rank=%d operation=ipc_host_readback status=%s "
                "signal=%llu mismatches=%u\n",
                rank_, signal == 2 && mismatches == 0 ? "PASS" : "FAIL",
                static_cast<unsigned long long>(signal), mismatches);
    if (signal != 2 || mismatches != 0)
      outcome = Result::fail;
    return outcome;
  }

  int rank_;
  Mode mode_;
  bool exported_ = false;
  bool imported_ = false;
  bool synced_ = false;
  void *owned_ = nullptr;
  void *importedBase_ = nullptr;
  uint32_t *status_ = nullptr;
  aclrtStream stream_ = nullptr;
  // The extra byte keeps cleanup safe if the runtime returns an unterminated
  // 65-byte key despite reporting success.
  std::array<char, 66> key_ = {};
};

} // namespace

int main(int argc, char **argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
    return 2;
  int rank = -1;
  int ranks = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &ranks);
  int first = 0;
  int second = 1;
  Mode mode = Mode::valueWrite;
  if (ranks != 2 || !parseArgs(argc, argv, &first, &second, &mode)) {
    if (rank == 0)
      std::fprintf(stderr,
                   "Usage: mpirun -np 2 stream_visibility_probe --devices "
                   "0,1 [--mode valuewrite|kernel-readback|kernel-wait]\n");
    MPI_Finalize();
    return 2;
  }

  Result local = aclCheck(aclInit(nullptr), rank, "aclInit");
  bool initialized = local == Result::pass;
  Result result = checkpoint(rank, "acl_init", local);
  if (result != Result::pass) {
    if (initialized)
      aclFinalize();
    MPI_Finalize();
    return 2;
  }
  uint32_t count = 0;
  local = aclCheck(aclrtGetDeviceCount(&count), rank, "aclrtGetDeviceCount");
  int device = rank == 0 ? first : second;
  if (local == Result::pass && static_cast<uint32_t>(device) >= count)
    local = Result::fail;
  result = checkpoint(rank, "device_inventory", local);
  bool selected = false;
  if (result == Result::pass) {
    local = aclCheck(aclrtSetDevice(device), rank, "aclrtSetDevice");
    selected = local == Result::pass;
    result = checkpoint(rank, "device_select", local);
  }
  if (result == Result::pass) {
    if (rank == 0)
      std::printf("PROBE transport=IPC devices=%d,%d "
                  "visible_device_count=%u\n",
                  first, second, count);
    LocalProbe smoke(rank);
    result = smoke.run();
    Result cleanup = smoke.cleanup();
    if (cleanup != Result::pass)
      result = Result::fail;
    result = checkpoint(rank, "local_value_ops_and_consumer", result);
  }
  if (result == Result::pass) {
    RemoteProbe remote(rank, mode);
    result = remote.run();
    Result cleanup = remote.cleanup();
    if (cleanup != Result::pass)
      result = Result::fail;
  }
  if (selected)
    if (aclCheck(aclrtResetDevice(device), rank, "aclrtResetDevice") !=
        Result::pass)
      result = Result::fail;
  if (aclCheck(aclFinalize(), rank, "aclFinalize") != Result::pass)
    result = Result::fail;
  result = checkpoint(rank, "final_result", result);
  MPI_Finalize();
  return result == Result::pass ? 0 : result == Result::unsupported ? 3 : 2;
}
