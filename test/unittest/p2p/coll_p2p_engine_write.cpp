// Cross-process correctness test for the P2P Engine WRITE path.

#include <mpi.h>

#include "flagcx.h"
#include "flagcx_p2p.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr size_t kBufferSize = 8192;

[[noreturn]] void fail(int rank, const char *message) {
  std::fprintf(stderr, "[rank %d] P2P WRITE: %s\n", rank, message);
  MPI_Abort(MPI_COMM_WORLD, 1);
  __builtin_unreachable();
}

void require(bool condition, int rank, const char *message) {
  if (!condition)
    fail(rank, message);
}

[[noreturn]] void failMismatch(int rank, const char *kind, size_t size,
                               size_t offset, unsigned char want,
                               unsigned char got) {
  std::fprintf(stderr,
               "[rank %d] P2P %s WRITE size=%zu offset=%zu expected=0x%02x "
               "actual=0x%02x\n",
               rank, kind, size, offset, static_cast<unsigned>(want),
               static_cast<unsigned>(got));
  fail(rank, "data mismatch");
}

void waitForWrite(FlagcxP2pConn *conn, uint64_t transferId, int rank) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!flagcxP2pEngineXferStatus(conn, transferId)) {
    if (std::chrono::steady_clock::now() >= deadline)
      fail(rank, "WRITE completion timed out");
    std::this_thread::yield();
  }
}

} // namespace

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = -1;
  int worldSize = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
  require(worldSize == 2, rank, "requires exactly two MPI ranks");
#ifdef USE_SHARED_P2P_ENGINE
  // More slices than request credits require background post/poll progress.
  require(setenv("FLAGCX_P2P_SLICE_SIZE", "256", 1) == 0 &&
              setenv("FLAGCX_P2P_FRAGMENT_LIMIT", "256", 1) == 0 &&
              setenv("FLAGCX_P2P_MAX_REQUESTS", "1", 1) == 0,
          rank, "configuring background-progress test failed");
#endif

  flagcxDeviceHandle_t device = nullptr;
  require(flagcxDeviceHandleInit(&device) == flagcxSuccess && device != nullptr,
          rank, "device adaptor initialization failed");
  int deviceCount = 0;
  require(device->getDeviceCount(&deviceCount) == flagcxSuccess &&
              deviceCount >= 2,
          rank, "requires at least two visible devices");
  require(device->setDevice(rank) == flagcxSuccess, rank,
          "selecting device failed");
  flagcxStream_t stream = nullptr;
  require(device->streamCreate(&stream) == flagcxSuccess, rank,
          "creating stream failed");

  void *buffer = nullptr;
  require(flagcxMemAlloc(&buffer, kBufferSize) == flagcxSuccess &&
              buffer != nullptr,
          rank, "allocating GPU buffer failed");
  std::vector<unsigned char> expected(kBufferSize);
  std::vector<unsigned char> actual(kBufferSize, 0);
  for (size_t i = 0; i < kBufferSize; ++i)
    expected[i] = static_cast<unsigned char>((i * 37 + 11) & 0xff);
  const void *initial = rank == 1 ? static_cast<const void *>(expected.data())
                                  : static_cast<const void *>(actual.data());
  require(device->deviceMemcpy(buffer, const_cast<void *>(initial), kBufferSize,
                               flagcxMemcpyHostToDevice,
                               stream) == flagcxSuccess,
          rank, "initializing GPU buffer failed");
  require(device->streamSynchronize(stream) == flagcxSuccess, rank,
          "initial GPU copy failed");

  FlagcxP2pEngine *engine = flagcxP2pEngineCreate();
  require(engine != nullptr, rank, "creating Engine failed");
  FlagcxP2pMr mr = 0;
  require(flagcxP2pEngineReg(engine, reinterpret_cast<uintptr_t>(buffer),
                             kBufferSize, mr) == 0,
          rank, "registering GPU buffer failed");
  require(flagcxP2pEngineStartRpcServer(engine) == 0, rank,
          "starting RPC server failed");

  char session[256] = {};
  if (rank == 0) {
    char *metadata = nullptr;
    require(flagcxP2pEngineGetMetadata(engine, &metadata) == 0 &&
                metadata != nullptr,
            rank, "obtaining Engine metadata failed");
    const std::string value(metadata);
    delete[] metadata;
    const size_t separator = value.find('?');
    require(separator != std::string::npos && separator < sizeof(session), rank,
            "invalid Engine metadata");
    std::memcpy(session, value.data(), separator);
  }
  MPI_Bcast(session, sizeof(session), MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

  FlagcxP2pConn *conn = nullptr;
  if (rank == 1) {
    conn = flagcxP2pEngineGetConn(engine, session);
    require(conn != nullptr, rank, "connecting to target Engine failed");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  uint64_t targetAddress = rank == 0 ? reinterpret_cast<uint64_t>(buffer) : 0;
  MPI_Bcast(&targetAddress, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
  if (rank == 1) {
    FlagcxP2pRdmaDesc invalidDesc = {};
    require(flagcxP2pEngineMakeDesc(conn, targetAddress + kBufferSize - 16, 32,
                                    &invalidDesc) != 0,
            rank, "out-of-bounds descriptor unexpectedly succeeded");
  }

  const size_t sizes[] = {1, 2048, 2052, kBufferSize};
  for (size_t bytes : sizes) {
    if (rank == 0) {
      require(device->deviceMemcpy(buffer, actual.data(), kBufferSize,
                                   flagcxMemcpyHostToDevice,
                                   stream) == flagcxSuccess,
              rank, "clearing target buffer failed");
      require(device->streamSynchronize(stream) == flagcxSuccess, rank,
              "target clear did not complete");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 1) {
      FlagcxP2pRdmaDesc desc = {};
      require(flagcxP2pEngineMakeDesc(conn, targetAddress,
                                      static_cast<uint32_t>(bytes), &desc) == 0,
              rank, "creating remote descriptor failed");
      uint64_t transferId = 0;
      require(
          flagcxP2pEngineWrite(conn, mr, buffer, bytes, desc, &transferId) == 0,
          rank, "submitting WRITE failed");
      waitForWrite(conn, transferId, rank);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      require(device->deviceMemcpy(actual.data(), buffer, kBufferSize,
                                   flagcxMemcpyDeviceToHost,
                                   stream) == flagcxSuccess,
              rank, "copying target buffer to host failed");
      require(device->streamSynchronize(stream) == flagcxSuccess, rank,
              "target verification copy failed");
      for (size_t i = 0; i < kBufferSize; ++i) {
        const unsigned char want = i < bytes ? expected[i] : 0;
        if (actual[i] != want)
          failMismatch(rank, "single", bytes, i, want, actual[i]);
      }
      std::fill(actual.begin(), actual.end(), 0);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  if (rank == 0) {
    require(device->deviceMemcpy(buffer, actual.data(), kBufferSize,
                                 flagcxMemcpyHostToDevice,
                                 stream) == flagcxSuccess,
            rank, "clearing vector target failed");
    require(device->streamSynchronize(stream) == flagcxSuccess, rank,
            "vector target clear did not complete");
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 1) {
    const size_t offsets[] = {0, 4096};
    const size_t lengths[] = {128, 2052};
    std::vector<FlagcxP2pMr> mrs;
    std::vector<void *> sources;
    std::vector<size_t> sizes;
    std::vector<FlagcxP2pRdmaDesc> descs;
    for (int i = 0; i < 2; ++i) {
      FlagcxP2pRdmaDesc desc = {};
      require(flagcxP2pEngineMakeDesc(conn, targetAddress + offsets[i],
                                      static_cast<uint32_t>(lengths[i]),
                                      &desc) == 0,
              rank, "creating vector descriptor failed");
      mrs.push_back(mr);
      sources.push_back(static_cast<char *>(buffer) + offsets[i]);
      sizes.push_back(lengths[i]);
      descs.push_back(desc);
    }
    uint64_t transferId = 0;
    require(flagcxP2pEngineWriteVector(conn, mrs, sources, sizes, descs, 2,
                                       &transferId) == 0,
            rank, "submitting vector WRITE failed");
    waitForWrite(conn, transferId, rank);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    require(device->deviceMemcpy(actual.data(), buffer, kBufferSize,
                                 flagcxMemcpyDeviceToHost,
                                 stream) == flagcxSuccess,
            rank, "copying vector target to host failed");
    require(device->streamSynchronize(stream) == flagcxSuccess, rank,
            "vector verification copy failed");
    for (size_t i = 0; i < kBufferSize; ++i) {
      const bool written = i < 128 || (i >= 4096 && i < 4096 + 2052);
      const unsigned char want = written ? expected[i] : 0;
      if (actual[i] != want)
        failMismatch(rank, "vector", kBufferSize, i, want, actual[i]);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

#ifdef USE_SHARED_P2P_ENGINE
  if (rank == 0) {
    std::fill(actual.begin(), actual.end(), 0);
    require(device->deviceMemcpy(buffer, actual.data(), kBufferSize,
                                 flagcxMemcpyHostToDevice,
                                 stream) == flagcxSuccess &&
                device->streamSynchronize(stream) == flagcxSuccess,
            rank, "clearing background-progress target failed");
  }
  MPI_Barrier(MPI_COMM_WORLD);
  uint64_t unattendedId = 0;
  if (rank == 1) {
    FlagcxP2pRdmaDesc desc = {};
    require(flagcxP2pEngineMakeDesc(conn, targetAddress, kBufferSize, &desc) ==
                    0 &&
                flagcxP2pEngineWrite(conn, mr, buffer, kBufferSize, desc,
                                     &unattendedId) == 0,
            rank, "submitting background-progress WRITE failed");
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(10);
    bool complete = false;
    while (!complete) {
      require(device->deviceMemcpy(actual.data(), buffer, kBufferSize,
                                   flagcxMemcpyDeviceToHost,
                                   stream) == flagcxSuccess &&
                  device->streamSynchronize(stream) == flagcxSuccess,
              rank, "reading background-progress target failed");
      complete = std::equal(actual.begin(), actual.end(), expected.begin());
      if (std::chrono::steady_clock::now() >= deadline)
        fail(rank, "WRITE made no progress without XferStatus");
      if (!complete)
        std::this_thread::yield();
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 1)
    waitForWrite(conn, unattendedId, rank);
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (rank == 0)
    std::puts(
        "P2P Engine WRITE correctness: single and vector transfers passed");

  flagcxP2pEngineMrDestroy(engine, mr);
  flagcxP2pEngineDestroy(engine);
  flagcxMemFree(buffer);
  device->streamDestroy(stream);
  flagcxDeviceHandleFree(device);
  MPI_Finalize();
  return 0;
}
