#include "adaptor.h"
#include "rma_test.hpp"
#include "sym_heap.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

int collectiveFailed(flagcxResult_t result) {
  int local = result == flagcxSuccess ? 0 : 1;
  int global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  return global;
}

void stage(int rank, const char *name, flagcxResult_t result) {
  std::fprintf(stderr, "IPC Get probe: rank=%d stage=%s result=%d\n", rank,
               name, static_cast<int>(result));
  std::fflush(stderr);
}

} // namespace

// This control exercises a gate that is already released by an earlier write
// on the copy stream. It checks same-stream execution and Get completion;
// only the cross-stream probe below can hold the copy to test early counters.
TEST_F(RmaTest, IpcGetSameStreamProbe) {
  ASSERT_TRUE(requireIpc);
  ASSERT_EQ(nranks, 2);
  void *gate = nullptr;
  flagcxResult_t setup = flagcxSuccess;
  if (rank == 1) {
    setup = flagcxMemAlloc(&gate, sizeof(uint64_t));
    if (setup == flagcxSuccess)
      setup = devHandle->deviceMemset(gate, 0, sizeof(uint64_t),
                                      flagcxMemDevice, nullptr);
  } else {
    std::vector<uint8_t> pattern(64, 0xA5);
    setup = devHandle->deviceMemcpy(dataBuff, pattern.data(), pattern.size(),
                                    flagcxMemcpyHostToDevice, nullptr);
  }
  if (setup == flagcxSuccess)
    setup = devHandle->deviceSynchronize();
  stage(rank, "same-stream prepare", setup);
  int setupFailed = collectiveFailed(setup);
  if (setupFailed) {
    if (gate != nullptr) {
      EXPECT_EQ(flagcxMemFree(gate), flagcxSuccess);
    }
    ASSERT_EQ(setupFailed, 0);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxResult_t op = flagcxSuccess;
  auto runStage = [&](const char *name, auto operation) {
    if (op == flagcxSuccess) {
      op = operation();
      stage(rank, name, op);
    }
  };
  uint64_t before = 0;
  uint64_t after = 0;
  if (rank == 1) {
    runStage("same-stream read counter before",
             [&] { return flagcxReadCounter(comm, &before); });
    runStage("same-stream queue write", [&] {
      return deviceAdaptor->streamWriteValue64(stream, gate, 1, 0);
    });
    runStage("same-stream queue wait", [&] {
      return deviceAdaptor->streamWaitValue64(stream, gate, 1,
                                              FLAGCX_STREAM_WAIT_VALUE_DEFAULT);
    });
    runStage("same-stream GetAsync", [&] {
      return flagcxGetAsync(dataBuff, 64, flagcxUint8, 0, dataWin, 0, 0, comm,
                            stream);
    });
    runStage("same-stream wait counter",
             [&] { return flagcxWaitCounter(comm, before + 1); });
    runStage("same-stream synchronize copy",
             [&] { return devHandle->streamSynchronize(stream); });
    runStage("same-stream read counter after",
             [&] { return flagcxReadCounter(comm, &after); });
    if (op == flagcxSuccess) {
      EXPECT_EQ(after, before + 1);
      std::vector<uint8_t> actual(64);
      EXPECT_EQ(devHandle->deviceMemcpy(actual.data(), dataBuff, actual.size(),
                                        flagcxMemcpyDeviceToHost, nullptr),
                flagcxSuccess);
      EXPECT_EQ(actual, std::vector<uint8_t>(64, 0xA5));
    }
  }
  int operationFailed = collectiveFailed(op);
  if (gate != nullptr) {
    EXPECT_EQ(flagcxMemFree(gate), flagcxSuccess);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  ASSERT_EQ(operationFailed, 0) << "Same-stream IPC Get failed; rank=" << rank
                                << " result=" << static_cast<int>(op);
}

TEST_F(RmaTest, IpcGetCounterOrderProbe) {
  ASSERT_TRUE(requireIpc) << "Probe requires the explicit IPC invocation";
  ASSERT_EQ(nranks, 2);
  int localMrAbsent = dataWin->defaultBase->mrIndex < 0 ? 1 : 0;
  int allMrsAbsent = 0;
  MPI_Allreduce(&localMrAbsent, &allMrsAbsent, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  ASSERT_EQ(allMrsAbsent, 1) << "Probe unexpectedly has a network MR";

  void *gate = nullptr;
  flagcxStream_t releaseStream = nullptr;
  flagcxResult_t setup = flagcxSuccess;
  if (rank == 1) {
    setup = flagcxMemAlloc(&gate, sizeof(uint64_t));
    stage(rank, "allocate gate", setup);
    if (setup == flagcxSuccess)
      setup = devHandle->deviceMemset(gate, 0, sizeof(uint64_t),
                                      flagcxMemDevice, nullptr);
    if (setup == flagcxSuccess)
      setup = devHandle->deviceSynchronize();
    if (setup == flagcxSuccess)
      setup = devHandle->streamCreate(&releaseStream);
  } else {
    std::vector<uint8_t> pattern(64, 0xA5);
    setup = devHandle->deviceMemcpy(dataBuff, pattern.data(), pattern.size(),
                                    flagcxMemcpyHostToDevice, nullptr);
    if (setup == flagcxSuccess)
      setup = devHandle->deviceSynchronize();
  }
  stage(rank, "prepare", setup);
  int setupFailed = collectiveFailed(setup);
  if (setupFailed) {
    if (rank == 1) {
      if (releaseStream != nullptr)
        EXPECT_EQ(devHandle->streamDestroy(releaseStream), flagcxSuccess);
      if (gate != nullptr)
        EXPECT_EQ(flagcxMemFree(gate), flagcxSuccess);
    }
    ASSERT_EQ(setupFailed, 0) << "IPC Get probe setup failed";
  }
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxResult_t gateResult = flagcxSuccess;
  if (rank == 1)
    gateResult = deviceAdaptor->streamWaitValue64(
        stream, gate, 1, FLAGCX_STREAM_WAIT_VALUE_DEFAULT);
  stage(rank, "queue wait", gateResult);
  int gateFailed = collectiveFailed(gateResult);
  if (gateFailed) {
    if (rank == 1) {
      EXPECT_EQ(devHandle->streamDestroy(releaseStream), flagcxSuccess);
      EXPECT_EQ(flagcxMemFree(gate), flagcxSuccess);
    }
    ASSERT_EQ(gateFailed, 0) << "IPC Get probe could not queue stream wait";
  }

  flagcxResult_t op = flagcxSuccess;
  const char *failedStage = "none";
  auto runStage = [&](const char *name, auto operation) {
    if (op == flagcxSuccess) {
      op = operation();
      stage(rank, name, op);
      if (op != flagcxSuccess)
        failedStage = name;
    }
  };
  uint64_t before = 0;
  uint64_t pending = 0;
  if (rank == 1) {
    runStage("read counter before",
             [&] { return flagcxReadCounter(comm, &before); });
    runStage("GetAsync", [&] {
      return flagcxGetAsync(dataBuff, 64, flagcxUint8, 0, dataWin, 0, 0, comm,
                            stream);
    });
    runStage("read pending counter",
             [&] { return flagcxReadCounter(comm, &pending); });
    std::fprintf(stderr, "IPC Get probe: rank=1 before=%llu pending=%llu\n",
                 static_cast<unsigned long long>(before),
                 static_cast<unsigned long long>(pending));
    std::fflush(stderr);

    // Always release the queued wait, including when an earlier stage fails.
    flagcxResult_t release =
        deviceAdaptor->streamWriteValue64(releaseStream, gate, 1, 0);
    stage(rank, "queue release", release);
    runStage("streamWriteValue64(release)", [&] { return release; });
    if (release != flagcxSuccess) {
      uint64_t open = 1;
      EXPECT_EQ(devHandle->deviceMemcpy(gate, &open, sizeof(open),
                                        flagcxMemcpyHostToDevice, nullptr),
                flagcxSuccess);
    } else {
      release = devHandle->streamSynchronize(releaseStream);
      stage(rank, "synchronize release", release);
      runStage("streamSynchronize(release)", [&] { return release; });
    }
    runStage("wait counter",
             [&] { return flagcxWaitCounter(comm, before + 1); });
    runStage("synchronize copy",
             [&] { return devHandle->streamSynchronize(stream); });
  }
  int operationFailed = collectiveFailed(op);
  if (rank == 1) {
    if (op == flagcxSuccess) {
      EXPECT_EQ(pending, before) << "Counter advanced before device copy";
      std::vector<uint8_t> actual(64);
      EXPECT_EQ(devHandle->deviceMemcpy(actual.data(), dataBuff, actual.size(),
                                        flagcxMemcpyDeviceToHost, nullptr),
                flagcxSuccess);
      EXPECT_EQ(actual, std::vector<uint8_t>(64, 0xA5));
    }
    EXPECT_EQ(devHandle->streamDestroy(releaseStream), flagcxSuccess);
    EXPECT_EQ(flagcxMemFree(gate), flagcxSuccess);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  ASSERT_EQ(operationFailed, 0)
      << "IPC Get probe failed on rank=" << rank << " stage=" << failedStage
      << " result=" << static_cast<int>(op);
}
