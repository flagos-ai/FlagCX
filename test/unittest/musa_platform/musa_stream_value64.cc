// Host-only validation of the production adaptor's native stream operations.
// Run in a separate process with an external timeout: a broken wait may hang.
#include <musa_runtime.h>
#include "flagcx_device_adaptor.h"
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

extern "C" struct flagcxDeviceAdaptor *deviceAdaptor;

// Fail immediately instead of destroying a stream containing an unsatisfied
// wait. Process teardown reclaims the context; the external timeout guards
// native calls that themselves stop making progress.
static void check(musaError_t result, const char *operation) {
  if (result != musaSuccess) {
    std::fprintf(stderr, "FAIL %s: %s\n", operation, musaGetErrorString(result));
    std::fflush(nullptr);
    std::_Exit(1);
  }
}
static void check(flagcxResult_t result, const char *operation) {
  if (result != flagcxSuccess) {
    std::fprintf(stderr, "FAIL %s: flagcxResult=%d\n", operation, int(result));
    std::fflush(nullptr);
    std::_Exit(1);
  }
}
#define CHECK(operation) check((operation), #operation)

struct Stream {
  int device;
  musaStream_t native = nullptr;
  flagcxStream_t wrapper = nullptr;
  explicit Stream(int ordinal) : device(ordinal) {
    CHECK(musaSetDevice(device));
    CHECK(musaStreamCreateWithFlags(&native, musaStreamNonBlocking));
    CHECK(deviceAdaptor->streamCopy(&wrapper, reinterpret_cast<void *>(native)));
  }
  ~Stream() {
    CHECK(musaSetDevice(device));
    CHECK(musaStreamDestroy(native));
    CHECK(deviceAdaptor->streamFree(wrapper));
  }
};

struct Allocation {
  int device;
  void *ptr = nullptr;
  Allocation(int ordinal, size_t bytes) : device(ordinal) {
    CHECK(musaSetDevice(device));
    CHECK(musaMalloc(&ptr, bytes));
    CHECK(musaMemset(ptr, 0, bytes));
  }
  ~Allocation() {
    CHECK(musaSetDevice(device));
    CHECK(musaFree(ptr));
  }
};

// True only for an explicit NotSupported result; errors must fail the test.
static bool unsupported(flagcxResult_t result, const char *name) {
  if (result == flagcxNotSupported) {
    std::printf("SKIP %s: adaptor returned NotSupported\n", name);
    return true;
  }
  check(result, name);
  return false;
}

static bool blockingGeq() {
  Stream producer(0), consumer(0);
  Allocation signal(0, sizeof(uint64_t)), snapshot(0, sizeof(uint64_t));
  // Low-word GEQ would accept 33 >= 17, but the complete value is smaller.
  constexpr uint64_t below = (uint64_t{1} << 40) - (uint64_t{1} << 32) + 33;
  constexpr uint64_t expected = (uint64_t{1} << 40) + 17;
  constexpr uint64_t published = expected + 3;
  CHECK(musaMemcpy(signal.ptr, &below, sizeof(below), musaMemcpyHostToDevice));
  // Preflight write support before leaving any pending wait behind.
  if (unsupported(deviceAdaptor->streamWriteValue64(
                      producer.wrapper, signal.ptr, below, 0), "blocking GEQ write"))
    return false;
  CHECK(musaStreamSynchronize(producer.native));
  if (unsupported(deviceAdaptor->streamWaitValue64(
                      consumer.wrapper, signal.ptr, expected, 0), "blocking GEQ wait"))
    return false;
  CHECK(musaMemcpyAsync(snapshot.ptr, signal.ptr, sizeof(uint64_t),
                        musaMemcpyDeviceToDevice, consumer.native));
  bool premature = false;
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(100);
  do {
    const auto query = musaStreamQuery(consumer.native);
    if (query == musaSuccess) { premature = true; break; }
    if (query != musaErrorNotReady) check(query, "blocked consumer query");
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  } while (std::chrono::steady_clock::now() < deadline);
  // Always release before normal cleanup, including a premature-wait failure.
  CHECK(deviceAdaptor->streamWriteValue64(producer.wrapper, signal.ptr, published, 0));
  CHECK(musaStreamSynchronize(consumer.native));
  CHECK(musaStreamSynchronize(producer.native));
  uint64_t actual = 0;
  CHECK(musaMemcpy(&actual, snapshot.ptr, sizeof(actual), musaMemcpyDeviceToHost));
  if (premature || actual != published) {
    std::fprintf(stderr, "FAIL blocking GEQ: premature=%d snapshot=0x%llx\n",
                 premature, static_cast<unsigned long long>(actual));
    std::exit(1);
  }
  std::puts("PASS blocking GEQ: high 32 bits, below-target blocking, independent-stream progress");
  return true;
}

// MCCL transport/p2p.cc uses the same native peer capability/enable sequence.
static bool enablePeer(int from, int to) {
  CHECK(musaSetDevice(from));
  int accessible = 0;
  CHECK(musaDeviceCanAccessPeer(&accessible, from, to));
  if (!accessible) {
    std::printf("SKIP GPU%d->GPU%d: no native peer access\n", from, to);
    return false;
  }
  auto result = musaDeviceEnablePeerAccess(to, 0);
  if (result == musaErrorPeerAccessAlreadyEnabled) {
    (void)musaGetLastError(); // Clear the already-enabled status, as MCCL does.
  } else {
    CHECK(result);
  }
  return true;
}

static bool publication(int from, int to, bool flush) {
  char name[96];
  std::snprintf(name, sizeof(name), "GPU%d->GPU%d %s", from, to,
                flush ? "remote-flush P2P publication" : "default-wait P2P publication");
  if (!enablePeer(from, to)) return false;
  constexpr size_t count = 256;
  constexpr size_t bytes = count * sizeof(uint64_t);
  Stream producer(from), consumer(to);
  Allocation source(from, bytes), payload(to, bytes), snapshot(to, bytes);
  Allocation signal(to, sizeof(uint64_t));
  std::array<uint64_t, count> input{}, output{};
  const unsigned int flags = flush ? FLAGCX_STREAM_WAIT_VALUE_FLUSH_REMOTE_WRITES : 0;
  CHECK(musaSetDevice(to));
  // The satisfied preflight wait tests capability without stranding a stream.
  if (unsupported(deviceAdaptor->streamWaitValue64(
                      consumer.wrapper, signal.ptr, 0, flags), name)) return false;
  CHECK(musaStreamSynchronize(consumer.native));
  CHECK(musaSetDevice(from));
  if (unsupported(deviceAdaptor->streamWriteValue64(
                      producer.wrapper, signal.ptr, 0, 0), name)) return false;
  CHECK(musaStreamSynchronize(producer.native));

  for (unsigned int round = 1; round <= 32; ++round) {
    const uint64_t expected = (uint64_t{1} << 40) + round * 16;
    const uint64_t published = expected + (round % 2 ? 0 : 3);
    for (size_t i = 0; i < count; ++i)
      input[i] = (uint64_t{round} << 48) ^
                 (static_cast<uint64_t>(from + 1) << 40) ^ (i * 7919);
    CHECK(musaSetDevice(from));
    CHECK(musaMemcpy(source.ptr, input.data(), bytes, musaMemcpyHostToDevice));
    CHECK(musaSetDevice(to));
    CHECK(deviceAdaptor->streamWaitValue64(consumer.wrapper, signal.ptr, expected, flags));
    CHECK(musaMemcpyAsync(snapshot.ptr, payload.ptr, bytes,
                          musaMemcpyDeviceToDevice, consumer.native));
    CHECK(musaSetDevice(from));
    // No host synchronization between payload publication, signal, and read.
    CHECK(musaMemcpyAsync(payload.ptr, source.ptr, bytes,
                          musaMemcpyDeviceToDevice, producer.native));
    CHECK(deviceAdaptor->streamWriteValue64(producer.wrapper, signal.ptr, published, 0));
    CHECK(musaSetDevice(to));
    CHECK(musaStreamSynchronize(consumer.native));
    CHECK(musaMemcpy(output.data(), snapshot.ptr, bytes, musaMemcpyDeviceToHost));
    CHECK(musaSetDevice(from));
    CHECK(musaStreamSynchronize(producer.native));
    if (input != output) {
      std::fprintf(stderr, "FAIL %s: payload mismatch round=%u\n", name, round);
      std::exit(1);
    }
  }
  std::printf("PASS %s: 32 rounds, 256 full-width words/round, EQ and GEQ\n", name);
  return true;
}

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const bool flush = argc == 2 && std::strcmp(argv[1], "--flush") == 0;
  if (argc > 2 || (argc == 2 && !flush)) {
    std::fprintf(stderr, "Usage: %s [--flush]\n", argv[0]);
    return 1;
  }
  if (!deviceAdaptor || std::strcmp(deviceAdaptor->name, "MUSA") != 0 ||
      !deviceAdaptor->streamWriteValue64 || !deviceAdaptor->streamWaitValue64) {
    std::fprintf(stderr, "FAIL: requires the MUSA production adaptor\n");
    return 1;
  }
  int devices = 0;
  CHECK(musaGetDeviceCount(&devices));
  int passed = 0, skipped = 0;
  if (!flush) { if (blockingGeq()) ++passed; else ++skipped; }
  if (devices < 2) {
    std::puts("SKIP bidirectional publication: requires two GPUs");
    skipped += 2;
  } else {
    if (publication(0, 1, flush)) ++passed; else ++skipped;
    if (publication(1, 0, flush)) ++passed; else ++skipped;
  }
  std::printf("RESULT: %d PASS / %d SKIP; P2P copy-engine path only, no NIC/GDR proof\n", passed, skipped);
  // An incomplete capability run must not look like full acceptance to a script.
  return skipped ? 77 : 0;
}
