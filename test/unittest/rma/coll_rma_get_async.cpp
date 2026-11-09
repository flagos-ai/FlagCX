#include "adaptor.h"
#include "rma_test.hpp"
#include "sym_heap.h"
#include <cstdint>
#include <vector>

namespace {

int collectiveFailed(flagcxResult_t result) {
  int local = result == flagcxSuccess ? 0 : 1;
  int global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  return global;
}

} // namespace

TEST_F(RmaTest, GetAsyncOffsets) {
  if (nranks != 2)
    GTEST_SKIP() << "Requires two ranks";
  constexpr size_t transferSize = 64;
  const size_t offsets[] = {0, 0x400, size - transferSize};

  for (size_t offset : offsets) {
    flagcxResult_t setup = flagcxSuccess;
    if (rank == 0) {
      std::vector<uint8_t> pattern(transferSize);
      for (size_t i = 0; i < transferSize; ++i)
        pattern[i] = static_cast<uint8_t>((i + offset / transferSize) & 0xff);
      setup = devHandle->deviceMemcpy(static_cast<char *>(dataBuff) + offset,
                                      pattern.data(), transferSize,
                                      flagcxMemcpyHostToDevice, nullptr);
    } else {
      setup =
          devHandle->deviceMemset(dataBuff, 0, size, flagcxMemDevice, nullptr);
    }
    if (setup == flagcxSuccess)
      setup = devHandle->deviceSynchronize();
    ASSERT_EQ(collectiveFailed(setup), 0) << "GetAsync setup failed";
    MPI_Barrier(MPI_COMM_WORLD);

    flagcxResult_t op = flagcxSuccess;
    if (rank == 1) {
      uint64_t before = 0;
      op = flagcxReadCounter(comm, &before);
      if (op == flagcxSuccess)
        op =
            flagcxGetAsync(static_cast<char *>(dataBuff) + offset, transferSize,
                           flagcxUint8, 0, dataWin, offset, 0, comm, stream);
      if (op == flagcxSuccess)
        op = flagcxWaitCounter(comm, before + 1);
      if (op == flagcxSuccess)
        op = devHandle->streamSynchronize(stream);
    }
    ASSERT_EQ(collectiveFailed(op), 0) << "GetAsync operation failed";

    bool valid = true;
    if (rank == 1) {
      std::vector<uint8_t> actual(transferSize);
      op = devHandle->deviceMemcpy(
          actual.data(), static_cast<char *>(dataBuff) + offset, transferSize,
          flagcxMemcpyDeviceToHost, nullptr);
      valid = op == flagcxSuccess;
      for (size_t i = 0; i < transferSize && valid; ++i)
        valid = actual[i] ==
                static_cast<uint8_t>((i + offset / transferSize) & 0xff);
    }
    int localValid = valid ? 1 : 0;
    int allValid = 0;
    MPI_Allreduce(&localValid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    EXPECT_EQ(allValid, 1) << "GetAsync data mismatch at offset " << offset;
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

TEST_F(RmaTest, IpcGetCounterFollowsCopy) {
  if (!requireIpc)
    GTEST_SKIP() << "Runs only in the explicit IPC invocation";
  if (nranks != 2)
    GTEST_SKIP() << "Requires two ranks";
  int localMrAbsent = dataWin->defaultBase->mrIndex < 0 ? 1 : 0;
  int allMrsAbsent = 0;
  MPI_Allreduce(&localMrAbsent, &allMrsAbsent, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  ASSERT_EQ(allMrsAbsent, 1) << "IPC test unexpectedly has a network MR";

  void *gate = nullptr;
  flagcxStream_t releaseStream = nullptr;
  flagcxResult_t setup = flagcxSuccess;
  if (rank == 1) {
    setup = flagcxMemAlloc(&gate, sizeof(uint64_t));
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
  ASSERT_EQ(collectiveFailed(setup), 0) << "IPC Get gate setup failed";
  MPI_Barrier(MPI_COMM_WORLD);

  // Only the counter-ordering check requires a stream gate. GetAsyncOffsets
  // independently exercises IPC Get and validates its data without this gate.
  flagcxResult_t gateResult = flagcxSuccess;
  if (rank == 1)
    gateResult = deviceAdaptor->streamWaitValue64(
        stream, gate, 1, FLAGCX_STREAM_WAIT_VALUE_DEFAULT);
  int localGateUnsupported = 0;
#if defined(USE_ILUVATAR_ADAPTOR)
  localGateUnsupported = gateResult == flagcxNotSupported ? 1 : 0;
#endif
  int anyGateUnsupported = 0;
  MPI_Allreduce(&localGateUnsupported, &anyGateUnsupported, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  int anyGateError =
      collectiveFailed(localGateUnsupported ? flagcxSuccess : gateResult);
  if (anyGateUnsupported || anyGateError) {
    if (rank == 1) {
      EXPECT_EQ(devHandle->streamDestroy(releaseStream), flagcxSuccess);
      EXPECT_EQ(flagcxMemFree(gate), flagcxSuccess);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    ASSERT_EQ(anyGateError, 0)
        << "IPC Get counter-ordering gate failed at streamWaitValue64 on "
           "at least one rank; local result="
        << static_cast<int>(gateResult);
    GTEST_SKIP() << "CoreX streamWaitValue64 is unsupported; only IPC Get "
                    "counter ordering is unavailable. GetAsyncOffsets "
                    "independently validates IPC Get";
  }

  flagcxResult_t op = flagcxSuccess;
  const char *failedStage = "none";
  auto runStage = [&](const char *name, auto operation) {
    if (op == flagcxSuccess) {
      op = operation();
      if (op != flagcxSuccess)
        failedStage = name;
    }
  };
  uint64_t before = 0;
  uint64_t pending = 0;
  if (rank == 1) {
    runStage("flagcxReadCounter(before)",
             [&] { return flagcxReadCounter(comm, &before); });
    runStage("flagcxGetAsync", [&] {
      return flagcxGetAsync(dataBuff, 64, flagcxUint8, 0, dataWin, 0, 0, comm,
                            stream);
    });
    runStage("flagcxReadCounter(pending)",
             [&] { return flagcxReadCounter(comm, &pending); });
    // Release the gate on a different stream even when the operation fails,
    // so the test cannot leave the RMA progress thread waiting forever.
    flagcxResult_t release =
        deviceAdaptor->streamWriteValue64(releaseStream, gate, 1, 0);
    runStage("streamWriteValue64(release)", [&] { return release; });
    if (release == flagcxSuccess) {
      release = devHandle->streamSynchronize(releaseStream);
      runStage("streamSynchronize(release)", [&] { return release; });
    }
    runStage("flagcxWaitCounter",
             [&] { return flagcxWaitCounter(comm, before + 1); });
    runStage("streamSynchronize(copy)",
             [&] { return devHandle->streamSynchronize(stream); });
  }
  int anyOperationError = collectiveFailed(op);
  if (rank == 1) {
    if (op == flagcxSuccess) {
      EXPECT_EQ(pending, before) << "IPC Get completed before its D2D copy";
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
  ASSERT_EQ(anyOperationError, 0)
      << "IPC Get counter-ordering operation failed on at least one rank; "
         "local rank="
      << rank << ", stage=" << failedStage
      << ", result=" << static_cast<int>(op);
}
