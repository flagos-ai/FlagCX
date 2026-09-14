// MPI correctness tests for flagcxPut / flagcxPutSignal.
// Requires 2 local ranks with a hetero communicator; individual invocations
// select either IPC or RDMA explicitly.

#include "adaptor.h"
#include "flagcx_hetero.h"
#include "global_comm.h"
#include "rma_test.hpp"
#include "sym_heap.h"
#include <cstdint>
#include <cstring>
#include <vector>

static int collectiveOpStatus(flagcxResult_t res) {
  int localStatus =
      (res == flagcxSuccess) ? 0 : (res == flagcxNotSupported ? 1 : 2);
  int globalStatus = 0;
  MPI_Allreduce(&localStatus, &globalStatus, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);
  return globalStatus;
}

// Validate the symmetric-window IPC locator independently from the RMA wrapper.
// This uses the same 1 MiB flagcxMemAlloc allocation and IPC table entry as the
// production path, including small transfers at interior offsets.
TEST_F(RmaTest, IpcResolvedPeerPointerSupportsDirectCopy) {
  if (!requireIpc)
    GTEST_SKIP() << "Runs only in the explicit IPC invocation";

  constexpr size_t testSize = 64;
  const size_t offsets[] = {0, 0x400, size - testSize};
  flagcxStream_t s = nullptr;
  flagcxResult_t setupRes = devHandle->streamCreate(&s);
  if (setupRes == flagcxSuccess && s == nullptr)
    setupRes = flagcxInternalError;
  ASSERT_EQ(collectiveOpStatus(setupRes), 0);

  // An IPC handle describes the containing allocation. Verify that the common
  // IPC table restores each peer's user-buffer offset after opening that
  // allocation in the importing process. Address-range introspection is an
  // optional adaptor capability, so every rank must agree whether this
  // additional check can run before entering its collectives.
  void *allocationBase = nullptr;
  size_t allocationSize = 0;
  flagcxResult_t addressRangeRes = flagcxNotSupported;
  if (deviceAdaptor->getAddressRange != nullptr) {
    addressRangeRes = deviceAdaptor->getAddressRange(dataBuff, &allocationBase,
                                                     &allocationSize);
  }
  int addressRangeStatus = collectiveOpStatus(addressRangeRes);
  ASSERT_NE(addressRangeStatus, 2)
      << "Allocation-range query failed on at least one rank";

  if (addressRangeStatus == 0) {
    uintptr_t userAddress = reinterpret_cast<uintptr_t>(dataBuff);
    uintptr_t baseAddress = reinterpret_cast<uintptr_t>(allocationBase);
    uint64_t localUserOffset =
        userAddress >= baseAddress ? userAddress - baseAddress : UINT64_MAX;
    std::vector<uint64_t> peerUserOffsets(nranks, UINT64_MAX);
    MPI_Allgather(&localUserOffset, sizeof(localUserOffset), MPI_BYTE,
                  peerUserOffsets.data(), sizeof(localUserOffset), MPI_BYTE,
                  MPI_COMM_WORLD);

    int peer = nranks == 2 ? 1 - rank : -1;
    int peerLocalRank =
        peer >= 0 && comm->heteroComm->rankToLocalRank != nullptr
            ? comm->heteroComm->rankToLocalRank[peer]
            : -1;
    int slot = dataWin->defaultBase->ipcSlot;
    bool localOffsetValid = peer >= 0 && peerLocalRank >= 0 && slot >= 0 &&
                            slot < FLAGCX_MAX_IPC_ENTRIES;
    if (localOffsetValid) {
      struct flagcxIpcTableEntry *entry = &comm->ipcTable[slot];
      localOffsetValid = peerLocalRank < entry->nPeers &&
                         entry->hostPeerBasePtrs != nullptr &&
                         entry->hostPeerPtrs != nullptr &&
                         entry->hostPeerBasePtrs[peerLocalRank] != nullptr &&
                         entry->hostPeerPtrs[peerLocalRank] != nullptr;
      if (localOffsetValid) {
        uintptr_t peerBase =
            reinterpret_cast<uintptr_t>(entry->hostPeerBasePtrs[peerLocalRank]);
        uintptr_t peerUser =
            reinterpret_cast<uintptr_t>(entry->hostPeerPtrs[peerLocalRank]);
        localOffsetValid = peerUser >= peerBase &&
                           peerUser - peerBase == peerUserOffsets[peer];
      }
    }
    int localValid = localOffsetValid ? 1 : 0;
    int allValid = 0;
    MPI_Allreduce(&localValid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    ASSERT_EQ(allValid, 1)
        << "IPC peer pointer did not preserve the exported user offset";
  }

  // The direct-copy checks are deliberately outside the optional address-range
  // block. Adaptors without introspection must still prove IPC data
  // correctness.
  for (size_t offset : offsets) {
    setupRes =
        devHandle->deviceMemset(dataBuff, 0, size, flagcxMemDevice, nullptr);
    ASSERT_EQ(collectiveOpStatus(setupRes), 0);
    MPI_Barrier(MPI_COMM_WORLD);

    flagcxResult_t opRes = flagcxSuccess;
    if (rank == 0) {
      std::vector<uint8_t> pattern(testSize);
      for (size_t i = 0; i < testSize; ++i)
        pattern[i] = static_cast<uint8_t>((i * 17 + offset / testSize) & 0xff);

      void *localRange = static_cast<char *>(dataBuff) + offset;
      void *peerRange = nullptr;
      opRes = devHandle->deviceMemcpy(localRange, pattern.data(), testSize,
                                      flagcxMemcpyHostToDevice, nullptr);
      if (opRes == flagcxSuccess) {
        opRes = flagcxSymWindowResolveIpcPeerPtr(comm->heteroComm,
                                                 dataWin->defaultBase, 1,
                                                 offset, testSize, &peerRange);
      }
      if (opRes == flagcxSuccess && peerRange == nullptr)
        opRes = flagcxInternalError;
      if (opRes == flagcxSuccess) {
        opRes = devHandle->deviceMemcpy(peerRange, localRange, testSize,
                                        flagcxMemcpyDeviceToDevice, s);
      }
      if (opRes == flagcxSuccess)
        opRes = devHandle->streamSynchronize(s);
    }

    ASSERT_EQ(collectiveOpStatus(opRes), 0)
        << "Direct symmetric-window IPC copy failed at offset " << offset;
    MPI_Barrier(MPI_COMM_WORLD);

    bool localDataValid = true;
    if (rank == 1) {
      std::vector<uint8_t> actual(testSize, 0);
      void *localRange = static_cast<char *>(dataBuff) + offset;
      flagcxResult_t copyRes =
          devHandle->deviceMemcpy(actual.data(), localRange, testSize,
                                  flagcxMemcpyDeviceToHost, nullptr);
      if (copyRes == flagcxSuccess)
        copyRes = devHandle->deviceSynchronize();
      for (size_t i = 0; copyRes == flagcxSuccess && i < testSize; ++i) {
        uint8_t expected =
            static_cast<uint8_t>((i * 17 + offset / testSize) & 0xff);
        if (actual[i] != expected) {
          localDataValid = false;
          break;
        }
      }
      if (copyRes != flagcxSuccess)
        localDataValid = false;
    }

    int localValid = localDataValid ? 1 : 0;
    int allDataValid = 0;
    MPI_Allreduce(&localValid, &allDataValid, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    EXPECT_EQ(allDataValid, 1)
        << "Direct symmetric-window IPC data mismatch at offset " << offset;
  }

  EXPECT_EQ(devHandle->streamDestroy(s), flagcxSuccess);
}

// The IPC data path must use the symmetric window's IPC locator directly. It
// must not require a network MR index or an initialized RDMA sendComm.
TEST_F(RmaTest, IpcPutWithoutNetworkMr) {
  if (!requireIpc)
    GTEST_SKIP() << "Runs only in the explicit IPC invocation";
  int localMrAbsent = dataWin->defaultBase->mrIndex < 0 ? 1 : 0;
  int allMrsAbsent = 0;
  MPI_Allreduce(&localMrAbsent, &allMrsAbsent, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  ASSERT_EQ(allMrsAbsent, 1)
      << "IPC invocation unexpectedly registered a network MR";

  constexpr size_t testSize = 64;
  const size_t offsets[] = {0, 0x400, size - testSize};
  flagcxStream_t s = nullptr;
  flagcxResult_t setupRes = devHandle->streamCreate(&s);
  if (setupRes == flagcxSuccess && s == nullptr)
    setupRes = flagcxInternalError;
  ASSERT_EQ(collectiveOpStatus(setupRes), 0);

  for (size_t offset : offsets) {
    setupRes =
        devHandle->deviceMemset(dataBuff, 0, size, flagcxMemDevice, nullptr);
    ASSERT_EQ(collectiveOpStatus(setupRes), 0);
    MPI_Barrier(MPI_COMM_WORLD);

    flagcxResult_t opRes = flagcxSuccess;
    if (rank == 0) {
      std::vector<uint8_t> pattern(testSize, 0x5A);
      void *localRange = static_cast<char *>(dataBuff) + offset;
      opRes = devHandle->deviceMemcpy(localRange, pattern.data(), testSize,
                                      flagcxMemcpyHostToDevice, nullptr);
      if (opRes == flagcxSuccess) {
        uint64_t opSeq = 0;
        opRes = flagcxHeteroPutStream(comm->heteroComm, 1, offset, offset,
                                      testSize, -1, -1, dataWin->defaultBase,
                                      dataWin->defaultBase, s, &opSeq);
      }
      if (opRes == flagcxSuccess)
        opRes = devHandle->streamSynchronize(s);
    }

    int globalOpStatus = collectiveOpStatus(opRes);
    ASSERT_EQ(globalOpStatus, 0)
        << "IPC PUT failed without network MR state at offset " << offset;
    MPI_Barrier(MPI_COMM_WORLD);

    bool localDataValid = true;
    if (rank == 1) {
      std::vector<uint8_t> received(testSize, 0);
      void *localRange = static_cast<char *>(dataBuff) + offset;
      flagcxResult_t copyRes =
          devHandle->deviceMemcpy(received.data(), localRange, testSize,
                                  flagcxMemcpyDeviceToHost, nullptr);
      localDataValid = copyRes == flagcxSuccess &&
                       received == std::vector<uint8_t>(testSize, 0x5A);
    }

    int localValid = localDataValid ? 1 : 0;
    int allDataValid = 0;
    MPI_Allreduce(&localValid, &allDataValid, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    EXPECT_EQ(allDataValid, 1) << "IPC PUT data mismatch at offset " << offset;
  }
  EXPECT_EQ(devHandle->streamDestroy(s), flagcxSuccess);
}

// ---------------------------------------------------------------------------
// PutSignal: rank 0 writes known pattern to rank 1, rank 1 verifies
// ---------------------------------------------------------------------------
TEST_F(RmaTest, PutSignalSmall) {
  if (nranks < 2)
    GTEST_SKIP() << "Requires at least 2 ranks";
  ASSERT_FALSE(signalRmaSetupFailed) << signalRmaSkipReason;
  if (!signalRmaAvailable)
    GTEST_SKIP() << signalRmaSkipReason;

  const size_t testSize = 64;
  flagcxStream_t s;
  devHandle->streamCreate(&s);

  // Reset data buffer
  devHandle->deviceMemset(dataBuff, 0, size, flagcxMemDevice, nullptr);
  devHandle->deviceMemset(signalBuff, 0, signalSize, flagcxMemDevice, nullptr);
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxResult_t opRes = flagcxSuccess;
  if (rank == 0) {
    // Fill source with 0xAB pattern
    std::vector<uint8_t> pattern(testSize, 0xAB);
    devHandle->deviceMemcpy(dataBuff, pattern.data(), testSize,
                            flagcxMemcpyHostToDevice, nullptr);

    opRes = flagcxPutSignal(dataBuff, testSize, flagcxChar, 1, dataWin, 0, 0,
                            comm, s);
  }

  int globalStatus = collectiveOpStatus(opRes);
  if (globalStatus == 1) {
    devHandle->streamDestroy(s);
    GTEST_SKIP() << "RMA signal operations are not supported";
  }
  if (globalStatus == 2) {
    devHandle->streamDestroy(s);
    FAIL() << "flagcxPutSignal failed collectively with status "
           << globalStatus;
  }

  flagcxResult_t waitRes = flagcxSuccess;
  if (rank == 0) {
    waitRes = devHandle->streamSynchronize(s);
  } else if (rank == 1) {
    // Wait for signal from rank 0
    flagcxWaitSignalDesc_t desc = {1, 0};
    waitRes = flagcxWaitSignal(1, &desc, comm, s);
    if (waitRes == flagcxSuccess)
      waitRes = devHandle->streamSynchronize(s);
  }

  int globalWaitStatus = collectiveOpStatus(waitRes);
  if (globalWaitStatus == 1) {
    devHandle->streamDestroy(s);
    FAIL() << "Remote-write visibility flush became unavailable after the "
              "suite capability check";
  }
  if (globalWaitStatus == 2) {
    devHandle->streamDestroy(s);
    FAIL() << "flagcxWaitSignal failed collectively with status "
           << globalWaitStatus;
  }

  if (rank == 1) {
    // Verify data
    std::vector<uint8_t> received(testSize, 0);
    devHandle->deviceMemcpy(received.data(), dataBuff, testSize,
                            flagcxMemcpyDeviceToHost, nullptr);

    int mismatches = 0;
    for (size_t i = 0; i < testSize; ++i) {
      if (received[i] != 0xAB) {
        mismatches++;
        if (mismatches == 1) {
          EXPECT_EQ(received[i], 0xAB) << "Mismatch at byte " << i;
        }
      }
    }
    EXPECT_EQ(mismatches, 0);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  devHandle->streamDestroy(s);
}

// ---------------------------------------------------------------------------
// PutSignal large (1 MB)
// ---------------------------------------------------------------------------
TEST_F(RmaTest, PutSignalLarge) {
  if (nranks < 2)
    GTEST_SKIP() << "Requires at least 2 ranks";
  ASSERT_FALSE(signalRmaSetupFailed) << signalRmaSkipReason;
  if (!signalRmaAvailable)
    GTEST_SKIP() << signalRmaSkipReason;

  const size_t testSize = RMA_TEST_SIZE;
  flagcxStream_t s;
  devHandle->streamCreate(&s);

  devHandle->deviceMemset(dataBuff, 0, size, flagcxMemDevice, nullptr);
  devHandle->deviceMemset(signalBuff, 0, signalSize, flagcxMemDevice, nullptr);
  MPI_Barrier(MPI_COMM_WORLD);

  flagcxResult_t opRes = flagcxSuccess;
  if (rank == 0) {
    // Fill with ascending byte pattern
    std::vector<uint8_t> pattern(testSize);
    for (size_t i = 0; i < testSize; ++i)
      pattern[i] = static_cast<uint8_t>(i & 0xFF);
    devHandle->deviceMemcpy(dataBuff, pattern.data(), testSize,
                            flagcxMemcpyHostToDevice, nullptr);

    opRes = flagcxPutSignal(dataBuff, testSize, flagcxChar, 1, dataWin, 0, 0,
                            comm, s);
  }

  int globalStatus = collectiveOpStatus(opRes);
  if (globalStatus == 1) {
    devHandle->streamDestroy(s);
    GTEST_SKIP() << "RMA signal operations are not supported";
  }
  if (globalStatus == 2) {
    devHandle->streamDestroy(s);
    FAIL() << "flagcxPutSignal failed collectively with status "
           << globalStatus;
  }

  flagcxResult_t waitRes = flagcxSuccess;
  if (rank == 0) {
    waitRes = devHandle->streamSynchronize(s);
  } else if (rank == 1) {
    flagcxWaitSignalDesc_t desc = {1, 0};
    waitRes = flagcxWaitSignal(1, &desc, comm, s);
    if (waitRes == flagcxSuccess)
      waitRes = devHandle->streamSynchronize(s);
  }

  int globalWaitStatus = collectiveOpStatus(waitRes);
  if (globalWaitStatus == 1) {
    devHandle->streamDestroy(s);
    FAIL() << "Remote-write visibility flush became unavailable after the "
              "suite capability check";
  }
  if (globalWaitStatus == 2) {
    devHandle->streamDestroy(s);
    FAIL() << "flagcxWaitSignal failed collectively with status "
           << globalWaitStatus;
  }

  if (rank == 1) {
    std::vector<uint8_t> received(testSize, 0);
    devHandle->deviceMemcpy(received.data(), dataBuff, testSize,
                            flagcxMemcpyDeviceToHost, nullptr);

    int mismatches = 0;
    for (size_t i = 0; i < testSize && mismatches < 10; ++i) {
      uint8_t expected = static_cast<uint8_t>(i & 0xFF);
      if (received[i] != expected) {
        mismatches++;
        if (mismatches == 1) {
          EXPECT_EQ(received[i], expected) << "Mismatch at byte " << i;
        }
      }
    }
    EXPECT_EQ(mismatches, 0);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  devHandle->streamDestroy(s);
}
