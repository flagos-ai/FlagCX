#include "flagcx_hetero.h"
#include "global_comm.h"
#include "ipcsocket.h"
#include "mem_alloc_registry.h"
#include "onesided_types.h"
#include "probe.hpp"
#include "sym_heap.h"
#include <chrono>
#include <mpi.h>
#include <unistd.h>

class MpiVmmProbe : public VmmProbe {
protected:
  void SetUp() override {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &local);
    int localRank = 0;
    MPI_Comm_rank(local, &localRank);
    MPI_Comm_size(local, &localRanks);
    MPI_Comm_free(&local);
    ASSERT_EQ(all(deviceAdaptor && deviceAdaptor->symPhysDescribe &&
                          deviceAdaptor->symFlatMapSegmentsBegin &&
                          deviceAdaptor->symFlatMapSegmentOne &&
                          deviceAdaptor->symLsaMapBeginAt
                      ? flagcxSuccess
                      : flagcxNotSupported),
              flagcxSuccess);
    int devices = 0;
    ASSERT_EQ(all(deviceAdaptor->getDeviceCount(&devices)), flagcxSuccess);
    ASSERT_EQ(all(devices >= 2 ? flagcxSuccess : flagcxInvalidUsage),
              flagcxSuccess);
    ASSERT_EQ(all(deviceAdaptor->setDevice(localRank % devices)),
              flagcxSuccess);
    ASSERT_STREQ(deviceAdaptor->name, "IXCUDA");
    char name[MPI_MAX_PROCESSOR_NAME];
    int length = 0;
    MPI_Get_processor_name(name, &length);
    std::fprintf(
        stderr,
        "probe rank=%d/%d host=%s localRank=%d localRanks=%d device=%d\n", rank,
        ranks, name, localRank, localRanks, localRank % devices);
  }
  flagcxResult_t all(flagcxResult_t result) {
    int value = static_cast<int>(result), reduced = 0;
    MPI_Allreduce(&value, &reduced, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    return static_cast<flagcxResult_t>(reduced);
  }
  void expectCollectiveFailure(flagcxResult_t result) {
    int value = (int)result, lo = 0, hi = 0;
    MPI_Allreduce(&value, &lo, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&value, &hi, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    EXPECT_NE(lo, (int)flagcxSuccess);
    EXPECT_EQ(lo, hi) << "rank-local failure did not converge";
  }
  void initComm() {
    flagcxUniqueId id;
    flagcxResult_t result = flagcxSuccess;
    if (rank == 0)
      result = flagcxGetUniqueId(&id);
    ASSERT_EQ(all(result), flagcxSuccess);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ASSERT_EQ(all(probeCall(
                  "communicator init",
                  [&] { return flagcxCommInitRank(&comm, ranks, &id, rank); })),
              flagcxSuccess);
    ASSERT_EQ(
        all(comm && comm->heteroComm ? flagcxSuccess : flagcxInternalError),
        flagcxSuccess);
  }
  void windowAllocate() {
    ASSERT_EQ(all(probeCall(
                  "window allocation",
                  [&] { return flagcxMemAlloc(&windowBuffer, 1024 * 1024); })),
              flagcxSuccess);
  }
  void registerWindow() {
    ASSERT_EQ(all(probeCall("window register",
                            [&] {
                              return flagcxCommWindowRegister(
                                  comm, windowBuffer, 1024 * 1024, &window,
                                  FLAGCX_WIN_COLL_SYMMETRIC);
                            })),
              flagcxSuccess);
    ASSERT_EQ(all(window && window->defaultBase &&
                          window->defaultBase->hasFlatMapping &&
                          window->defaultBase->isVmmAllocation
                      ? flagcxSuccess
                      : flagcxInternalError),
              flagcxSuccess);
  }
  void TearDown() override {
    if (socket.fd >= 0)
      EXPECT_EQ(flagcxIpcSocketClose(&socket), flagcxSuccess);
    if (comm) {
      // Let commDestroy handle live/pending windows before their allocation.
      auto result = probeCall("communicator destroy",
                              [&] { return flagcxCommDestroy(comm); });
      EXPECT_EQ(result, flagcxSuccess);
      if (result != flagcxSuccess)
        return;
    }
    if (windowBuffer)
      EXPECT_EQ(flagcxMemFree(windowBuffer), flagcxSuccess);
    if (secondBuffer)
      EXPECT_EQ(flagcxMemFree(secondBuffer), flagcxSuccess);
    VmmProbe::TearDown();
  }
  int rank = 0, ranks = 0, localRanks = 0;
  flagcxComm_t comm = nullptr;
  flagcxWindow_t window = nullptr;
  void *windowBuffer = nullptr, *secondBuffer = nullptr;
  flagcxIpcSocket socket{-1, {}, nullptr};
};

// No FlagCX communicator; uses only MPI and the local IPC socket transport.
TEST_F(MpiVmmProbe, LocalFdPeerMapping) {
  ASSERT_EQ(localRanks, ranks) << "allocation FDs must stay on one host";
  ASSERT_EQ(ranks, 2);
  auto result = deviceAdaptor->gdrMemAlloc(&buffer, 1024 * 1024, nullptr);
  ASSERT_EQ(all(result), flagcxSuccess);
  result = deviceAdaptor->symPhysDescribe(
      buffer, 1024 * 1024, &bundle, &segments, &count, &offset, &alignment);
  ASSERT_EQ(all(result), flagcxSuccess);
  // The default component invocation uses one segment per allocation.
  ASSERT_EQ(all(count == 1 && offset == 0 ? flagcxSuccess : flagcxInvalidUsage),
            flagcxSuccess);
  allocationBytes = segments[0].size;
  std::vector<flagcxSymRankLayout> layouts(ranks);
  std::vector<flagcxSymSegmentMeta> geometry(ranks);
  flagcxSymRankLayout own{allocationBytes, 0, 1024 * 1024, rank, 1};
  MPI_Allgather(&own, sizeof(own), MPI_BYTE, layouts.data(), sizeof(own),
                MPI_BYTE, MPI_COMM_WORLD);
  MPI_Allgather(segments, sizeof(*segments), MPI_BYTE, geometry.data(),
                sizeof(*segments), MPI_BYTE, MPI_COMM_WORLD);
  size_t stride = 0;
  for (auto &layout : layouts)
    stride = std::max(stride, layout.mappedSize);
  result = deviceAdaptor->symFlatMapSegmentsBegin(
      ranks, rank, bundle, layouts.data(), geometry.data(), ranks, stride,
      alignment, &flat);
  ASSERT_EQ(all(result), flagcxSuccess);
  uint64_t hash = 0;
  if (rank == 0)
    hash = (static_cast<uint64_t>(getpid()) << 32) ^
           std::chrono::steady_clock::now().time_since_epoch().count();
  MPI_Bcast(&hash, sizeof(hash), MPI_BYTE, 0, MPI_COMM_WORLD);
  ASSERT_EQ(all(flagcxIpcSocketInit(&socket, rank, hash, 1)), flagcxSuccess);
  for (int owner = 0; owner < ranks; ++owner) {
    int fd = -1;
    result = flagcxSuccess;
    if (rank == owner)
      result = deviceAdaptor->symPhysExportSegment(bundle, 0, &fd);
    auto exported = all(result);
    if (exported != flagcxSuccess) {
      if (fd >= 0)
        close(fd);
    }
    ASSERT_EQ(exported, flagcxSuccess);
    if (rank == owner)
      result = flagcxIpcSocketSendFd(&socket, fd, 1 - rank, hash);
    ASSERT_EQ(all(result), flagcxSuccess);
    if (rank != owner)
      result = flagcxIpcSocketRecvFd(&socket, &fd);
    ASSERT_EQ(all(result), flagcxSuccess);
    result = deviceAdaptor->symFlatMapSegmentOne(flat, owner,
                                                 rank == owner ? -1 : fd);
    EXPECT_EQ(close(fd), 0);
    ASSERT_EQ(all(result), flagcxSuccess);
  }
  std::vector<uint8_t> pattern(4096, rank + 21), actual(4096);
  ASSERT_EQ(all(deviceAdaptor->deviceMemcpy(
                buffer, pattern.data(), pattern.size(),
                flagcxMemcpyHostToDevice, nullptr, nullptr)),
            flagcxSuccess);
  MPI_Barrier(MPI_COMM_WORLD);
  void *peer = static_cast<char *>(flat) + (1 - rank) * stride;
  ASSERT_EQ(all(deviceAdaptor->deviceMemcpy(actual.data(), peer, actual.size(),
                                            flagcxMemcpyDeviceToHost, nullptr,
                                            nullptr)),
            flagcxSuccess);
  EXPECT_EQ(actual, std::vector<uint8_t>(4096, (1 - rank) + 21));
  MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(MpiVmmProbe, LocalArenaWindowReuse) {
  initComm();
  ASSERT_FALSE(HasFatalFailure());
  windowAllocate();
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_EQ(all(flagcxMemAlloc(&secondBuffer, 1024 * 1024)), flagcxSuccess);
  registerWindow();
  ASSERT_FALSE(HasFatalFailure());
  auto *firstBase = window->defaultBase;
  const auto firstAddress = reinterpret_cast<uintptr_t>(firstBase->flatBase);
  const size_t firstOffset = firstBase->lsaWindowOffset;
  EXPECT_EQ(firstBase->resourcePath, flagcxSymResourceLsa);
  flagcxWindow_t second = nullptr;
  ASSERT_EQ(all(flagcxCommWindowRegister(comm, secondBuffer, 1024 * 1024,
                                         &second, FLAGCX_WIN_COLL_SYMMETRIC)),
            flagcxSuccess);
  ASSERT_EQ(
      all(second && second->defaultBase ? flagcxSuccess : flagcxInternalError),
      flagcxSuccess);
  EXPECT_EQ(second->defaultBase->resourcePath, flagcxSymResourceLsa);
  EXPECT_NE(firstOffset, second->defaultBase->lsaWindowOffset);
  EXPECT_EQ(firstBase->slotStride, second->defaultBase->slotStride);
  EXPECT_EQ(firstAddress - firstOffset,
            reinterpret_cast<uintptr_t>(second->defaultBase->flatBase) -
                second->defaultBase->lsaWindowOffset);
  ASSERT_EQ(all(flagcxCommWindowDeregister(comm, window)), flagcxSuccess);
  window = nullptr;
  registerWindow();
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(window->defaultBase->lsaWindowOffset, firstOffset);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(window->defaultBase->flatBase),
            firstAddress);
  ASSERT_EQ(all(flagcxCommWindowDeregister(comm, second)), flagcxSuccess);
}

TEST_F(MpiVmmProbe, RemoteWithoutNetRollsBack) {
  initComm();
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_LT(comm->heteroComm->localRanks, ranks);
  windowAllocate();
  ASSERT_FALSE(HasFatalFailure());
  int handles = comm->heteroComm->oneSideHandleCount;
  auto result = probeCall("remote window without NET", [&] {
    return flagcxCommWindowRegister(comm, windowBuffer, 1024 * 1024, &window,
                                    FLAGCX_WIN_COLL_SYMMETRIC);
  });
  int value = (int)result, lo = 0, hi = 0;
  MPI_Allreduce(&value, &lo, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&value, &hi, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  EXPECT_EQ(lo, (int)flagcxNotSupported);
  EXPECT_EQ(lo, hi);
  EXPECT_EQ(window, nullptr);
  EXPECT_EQ(comm->heteroComm->symWindows, nullptr);
  EXPECT_EQ(comm->heteroComm->pendingSymCleanup, nullptr);
  EXPECT_EQ(comm->heteroComm->oneSideHandleCount, handles);
}

TEST_F(MpiVmmProbe, WindowVaGetPut) {
  initComm();
  ASSERT_FALSE(HasFatalFailure());
  ASSERT_LT(comm->heteroComm->localRanks, ranks);
  windowAllocate();
  ASSERT_FALSE(HasFatalFailure());
  registerWindow();
  ASSERT_FALSE(HasFatalFailure());
  auto *state = window->defaultBase;
  ASSERT_EQ(all(state->mrIndex >= 0 ? flagcxSuccess : flagcxInternalError),
            flagcxSuccess);
  auto *mr = comm->heteroComm->oneSideHandles[state->mrIndex];
  ASSERT_EQ(all(mr && mr->registrationRoute == FLAGCX_VMM_MR_ROUTE_VA
                    ? flagcxSuccess
                    : flagcxInternalError),
            flagcxSuccess);
  int peer = -1;
  // Pair matching local ranks across the two nodes, giving a bijection for PUT.
  for (int i = 0; i < ranks; ++i)
    if (comm->heteroComm->rankToNode[i] != comm->heteroComm->node &&
        i % 2 == rank % 2) {
      peer = i;
      break;
    }
  ASSERT_EQ(all(peer >= 0 ? flagcxSuccess : flagcxInvalidUsage), flagcxSuccess);
  std::vector<uint8_t> pattern(4096, rank + 31), actual(4096);
  ASSERT_EQ(all(deviceAdaptor->deviceMemcpy(
                static_cast<char *>(windowBuffer) + 127, pattern.data(),
                pattern.size(), flagcxMemcpyHostToDevice, nullptr, nullptr)),
            flagcxSuccess);
  MPI_Barrier(MPI_COMM_WORLD);
  uint64_t before = 0;
  ASSERT_EQ(all(flagcxReadCounter(comm, &before)), flagcxSuccess);
  ASSERT_EQ(all(probeCall("remote VA GET",
                          [&] {
                            return flagcxGet(comm, peer, 127, 8193,
                                             pattern.size(), state->mrIndex,
                                             state->mrIndex);
                          })),
            flagcxSuccess);
  ASSERT_EQ(all(flagcxWaitCounter(comm, before + 1)), flagcxSuccess);
  ASSERT_EQ(all(deviceAdaptor->deviceMemcpy(
                actual.data(), static_cast<char *>(windowBuffer) + 8193,
                actual.size(), flagcxMemcpyDeviceToHost, nullptr, nullptr)),
            flagcxSuccess);
  EXPECT_EQ(actual, std::vector<uint8_t>(4096, peer + 31));
  ASSERT_EQ(all(flagcxReadCounter(comm, &before)), flagcxSuccess);
  ASSERT_EQ(all(probeCall("remote VA PUT",
                          [&] {
                            return flagcxPut(comm, peer, 127, 16385,
                                             pattern.size(), state->mrIndex,
                                             state->mrIndex);
                          })),
            flagcxSuccess);
  ASSERT_EQ(all(flagcxWaitCounter(comm, before + 1)), flagcxSuccess);
  MPI_Barrier(MPI_COMM_WORLD);
  ASSERT_EQ(all(deviceAdaptor->deviceMemcpy(
                actual.data(), static_cast<char *>(windowBuffer) + 16385,
                actual.size(), flagcxMemcpyDeviceToHost, nullptr, nullptr)),
            flagcxSuccess);
  EXPECT_EQ(actual, std::vector<uint8_t>(4096, peer + 31));
}

namespace {
flagcxResult_t (*savedMap)(void *, int, int) = nullptr;
flagcxResult_t (*savedUnmap)(void *) = nullptr;
flagcxResult_t (*savedRegMr)(void *, void *, size_t, int, int,
                             void **) = nullptr;
flagcxResult_t failMr(void *, void *, size_t, int, int, void **mr) {
  *mr = nullptr;
  return flagcxUnhandledDeviceError;
}
flagcxResult_t failMappedSegment(void *base, int index, int fd) {
  auto result = savedMap(base, index, fd);
  return result == flagcxSuccess ? flagcxUnhandledDeviceError : result;
}
flagcxResult_t failUnmap(void *) { return flagcxUnhandledDeviceError; }
} // namespace

TEST_F(MpiVmmProbe, MapFailureAndRetry) {
  initComm();
  ASSERT_FALSE(HasFatalFailure());
  windowAllocate();
  ASSERT_FALSE(HasFatalFailure());
  savedMap = deviceAdaptor->symFlatMapSegmentOne;
  if (rank == 0)
    deviceAdaptor->symFlatMapSegmentOne = failMappedSegment;
  auto result = probeCall("injected mapped-segment failure", [&] {
    return flagcxCommWindowRegister(comm, windowBuffer, 1024 * 1024, &window,
                                    FLAGCX_WIN_COLL_SYMMETRIC);
  });
  deviceAdaptor->symFlatMapSegmentOne = savedMap;
  expectCollectiveFailure(result);
  EXPECT_EQ(window, nullptr);
  EXPECT_EQ(comm->heteroComm->pendingSymCleanup, nullptr);
  EXPECT_EQ(comm->heteroComm->symWindows, nullptr);
  registerWindow();
  ASSERT_FALSE(HasFatalFailure());
}

TEST_F(MpiVmmProbe, DeregistrationFailureAndRetry) {
  initComm();
  ASSERT_FALSE(HasFatalFailure());
  windowAllocate();
  ASSERT_FALSE(HasFatalFailure());
  registerWindow();
  ASSERT_FALSE(HasFatalFailure());
  flagcxWindow_t original = window;
  auto *retainedState = window->defaultBase;
  const int mrIndex = retainedState->mrIndex;
  ASSERT_EQ(all(retainedState->hasAllocationLease &&
                        retainedState->hasNetworkMrRef && mrIndex >= 0
                    ? flagcxSuccess
                    : flagcxInternalError),
            flagcxSuccess);
  savedUnmap = deviceAdaptor->symFlatMappingUnmapSegments;
  if (rank == 0)
    deviceAdaptor->symFlatMappingUnmapSegments = failUnmap;
  auto result = probeCall("injected unmap failure", [&] {
    return flagcxCommWindowDeregister(comm, window);
  });
  deviceAdaptor->symFlatMappingUnmapSegments = savedUnmap;
  EXPECT_EQ(result, rank == 0 ? flagcxUnhandledDeviceError : flagcxSuccess);
  // Public deregistration is local: only the failed rank keeps its state.
  ASSERT_EQ(all((rank == 0 ? window->defaultBase != nullptr
                           : window->defaultBase == nullptr)
                    ? flagcxSuccess
                    : flagcxInternalError),
            flagcxSuccess);
  if (rank == 0) {
    EXPECT_EQ(window->defaultBase, retainedState);
    EXPECT_EQ(retainedState->state, flagcxSymWindowCleanupRequired);
    EXPECT_TRUE(retainedState->published);
    EXPECT_TRUE(retainedState->hasAllocationLease);
    EXPECT_TRUE(retainedState->flatMappingOwned);
    // MR cleanup precedes the injected mapping failure and must not repeat.
    EXPECT_FALSE(retainedState->hasNetworkMrRef);
  } else {
    EXPECT_EQ(comm->heteroComm->symWindows, nullptr);
  }
  auto *mr = comm->heteroComm->oneSideHandles[mrIndex];
  // Slot zero may retain its connections after releasing the actual MR.
  EXPECT_TRUE(mr == nullptr ||
              (mr->localMrHandle == nullptr && mr->windowRefs == 0));
  flagcxMemAllocationInfo allocation = {};
  ASSERT_EQ(all(globalMemAllocRegistry.findExact(windowBuffer, &allocation)),
            flagcxSuccess);
  EXPECT_EQ(allocation.windowRefs, rank == 0 ? 1u : 0u);

  MPI_Barrier(MPI_COMM_WORLD);
  result = flagcxSuccess;
  if (rank == 0) {
    result = probeCall("retry local window deregistration", [&] {
      return flagcxCommWindowDeregister(comm, window);
    });
  }
  // Other ranks wait here without entering deregistration again.
  ASSERT_EQ(all(result), flagcxSuccess);
  EXPECT_EQ(window->defaultBase, nullptr);
  EXPECT_EQ(comm->heteroComm->symWindows, nullptr);
  EXPECT_EQ(comm->heteroComm->pendingSymCleanup, nullptr);
  ASSERT_EQ(all(globalMemAllocRegistry.findExact(windowBuffer, &allocation)),
            flagcxSuccess);
  EXPECT_EQ(allocation.windowRefs, 0u);

  registerWindow();
  ASSERT_FALSE(HasFatalFailure());
  EXPECT_EQ(window, original);
  ASSERT_EQ(all(flagcxCommWindowDeregister(comm, window)), flagcxSuccess);
  EXPECT_EQ(window->defaultBase, nullptr);
  ASSERT_EQ(all(flagcxMemFree(windowBuffer)), flagcxSuccess);
  windowBuffer = nullptr;
  window = nullptr;
}

TEST_F(MpiVmmProbe, MrFailureAndRetry) {
  initComm();
  ASSERT_FALSE(HasFatalFailure());
  windowAllocate();
  ASSERT_FALSE(HasFatalFailure());
  auto *net = comm->heteroComm->netAdaptor;
  ASSERT_EQ(all(net && net->regMr ? flagcxSuccess : flagcxInternalError),
            flagcxSuccess);
  const int handlesBefore = comm->heteroComm->oneSideHandleCount;
  const auto exchangesBefore =
      comm->heteroComm->oneSideDataMetadataExchangeCount;
  savedRegMr = net->regMr;
  if (rank == 0)
    net->regMr = failMr;
  auto result = probeCall("injected VA MR failure", [&] {
    return flagcxCommWindowRegister(comm, windowBuffer, 1024 * 1024, &window,
                                    FLAGCX_WIN_COLL_SYMMETRIC);
  });
  net->regMr = savedRegMr;
  int value = (int)result, lo = 0, hi = 0;
  MPI_Allreduce(&value, &lo, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&value, &hi, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  EXPECT_NE(lo, (int)flagcxSuccess);
  EXPECT_EQ(lo, hi);
  EXPECT_EQ(window, nullptr);
  EXPECT_EQ(comm->heteroComm->symWindows, nullptr);
  EXPECT_EQ(comm->heteroComm->pendingSymCleanup, nullptr);
  EXPECT_EQ(comm->heteroComm->oneSideHandleCount, handlesBefore);
  EXPECT_EQ(comm->heteroComm->oneSideDataMetadataExchangeCount,
            exchangesBefore);
  registerWindow();
  ASSERT_FALSE(HasFatalFailure());
}
