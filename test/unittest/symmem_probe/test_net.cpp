#include "flagcx_net.h"
#include "flagcx_net_adaptor.h"
#include "net.h"
#include "net_test_utils.h"
#include "onesided_types.h"
#include "probe.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>

namespace {
constexpr auto kTimeout = std::chrono::seconds(30);
flagcxResult_t waitRequest(struct flagcxNetAdaptor *net, void *request,
                           int *size = nullptr) {
  const auto deadline = std::chrono::steady_clock::now() + kTimeout;
  int done = 0;
  while (!done && std::chrono::steady_clock::now() < deadline) {
    flagcxResult_t result = net->test(request, &done, size);
    if (result != flagcxSuccess)
      return result;
    if (!done)
      std::this_thread::yield();
  }
  return done ? flagcxSuccess : flagcxSystemError;
}

struct ConnectionResult {
  flagcxResult_t connectResult = flagcxSystemError;
  flagcxResult_t acceptResult = flagcxSystemError;
  void *sendComm = nullptr;
  void *recvComm = nullptr;
};

ConnectionResult connectLoopback(struct flagcxNetAdaptor *net, int dev,
                                 const char *listenHandle, void *listenComm) {
  ConnectionResult result;
  char connectHandle[FLAGCX_NET_HANDLE_MAXSIZE] = {};
  memcpy(connectHandle, listenHandle, sizeof(connectHandle));

  std::atomic<bool> cancelled(false);
  std::mutex mutex;
  std::condition_variable completion;
  int completed = 0;
  auto markComplete = [&]() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      ++completed;
    }
    completion.notify_one();
  };

  std::thread connector([&]() {
    const auto deadline = std::chrono::steady_clock::now() + kTimeout;
    result.connectResult = flagcxSuccess;
    while (result.sendComm == nullptr && !cancelled.load() &&
           std::chrono::steady_clock::now() < deadline) {
      result.connectResult = net->connect(dev, connectHandle, &result.sendComm);
      if (result.connectResult != flagcxSuccess) {
        cancelled.store(true);
        break;
      }
      if (result.sendComm == nullptr)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (result.sendComm == nullptr && result.connectResult == flagcxSuccess) {
      result.connectResult = flagcxSystemError;
      cancelled.store(true);
    }
    markComplete();
  });

  std::thread accepter([&]() {
    const auto deadline = std::chrono::steady_clock::now() + kTimeout;
    result.acceptResult = flagcxSuccess;
    while (result.recvComm == nullptr && !cancelled.load() &&
           std::chrono::steady_clock::now() < deadline) {
      result.acceptResult = net->accept(listenComm, &result.recvComm);
      if (result.acceptResult != flagcxSuccess) {
        cancelled.store(true);
        break;
      }
      if (result.recvComm == nullptr)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (result.recvComm == nullptr && result.acceptResult == flagcxSuccess) {
      result.acceptResult = flagcxSystemError;
      cancelled.store(true);
    }
    markComplete();
  });

  {
    std::unique_lock<std::mutex> lock(mutex);
    if (!completion.wait_for(lock, kTimeout + std::chrono::seconds(1),
                             [&]() { return completed == 2; })) {
      fprintf(stderr, "net adaptor loopback handshake timed out\n");
      std::abort();
    }
  }
  connector.join();
  accepter.join();
  return result;
}

struct TestWindow {
  struct flagcxOneSideHandleInfo info = {};
  uintptr_t baseVas[2] = {};
  size_t regionSizes[2] = {};
  struct flagcxNetMrInfo mrInfos[2] = {};

  flagcxResult_t init(struct flagcxNetAdaptor *net, void *buffer, size_t size,
                      int rank, void *mrHandle) {
    if (net == nullptr || net->getMrInfo == nullptr || buffer == nullptr ||
        rank < 0 || rank >= 2 || mrHandle == nullptr)
      return flagcxInvalidArgument;
    FLAGCXCHECK(net->getMrInfo(mrHandle, &mrInfos[rank]));
    baseVas[rank] = reinterpret_cast<uintptr_t>(buffer);
    regionSizes[rank] = size;
    info.baseVas = baseVas;
    info.regionSizes = regionSizes;
    info.mrInfos = mrInfos;
    info.localMrHandle = mrHandle;
    info.nRanks = 2;
    return flagcxSuccess;
  }

  void **opaque() { return reinterpret_cast<void **>(&info); }
};

} // namespace

// No FlagCX communicator: isolates provider VA MR from SymMem setup.
void VmmProbe::netReadWrite(bool multi) {
  size_t bytesToAllocate = 1024 * 1024;
  if (multi) {
    size_t totalBytes = 0, granularity = 0;
    PROBE(deviceAdaptor->symLsaGetCapacity(&totalBytes, &granularity));
    ASSERT_GT(granularity, 0u);
    ASSERT_LE(granularity, SIZE_MAX / 3);
    ASSERT_EQ(setenv("FLAGCX_VMM_SEGMENT_BYTES",
                     std::to_string(granularity).c_str(), 1),
              0);
    bytesToAllocate = 3 * granularity;
  }
  allocate(bytesToAllocate, false);
  ASSERT_FALSE(HasFatalFailure());
  auto *net = getNetAdaptor(RDMA);
  ASSERT_NE(net, nullptr);
  ASSERT_NE(net->getMrInfo, nullptr);
  ASSERT_NE(net->iput, nullptr);
  ASSERT_NE(net->iget, nullptr);
  PROBE(net->init());
  int devices = 0, netDev = -1;
  PROBE(net->devices(&devices));
  ASSERT_GT(devices, 0);
  PROBE(flagcx_test::getLocalNetDevice(net, devices, &netDev));
  struct Resources {
    flagcxNetAdaptor *net;
    void *listener = nullptr, *sender = nullptr, *receiver = nullptr;
    void *sendMr = nullptr, *recvMr = nullptr, *destination = nullptr;
    ~Resources() {
      // Keep connection/VA ownership if the provider cannot deregister.
      if (sendMr && net->deregMr(sender, sendMr) != flagcxSuccess) {
        ADD_FAILURE() << "source MR deregistration failed";
        std::_Exit(1);
      }
      if (recvMr && net->deregMr(receiver, recvMr) != flagcxSuccess) {
        ADD_FAILURE() << "destination MR deregistration failed";
        std::_Exit(1);
      }
      if (sender)
        EXPECT_EQ(net->closeSend(sender), flagcxSuccess);
      if (receiver)
        EXPECT_EQ(net->closeRecv(receiver), flagcxSuccess);
      if (listener)
        EXPECT_EQ(net->closeListen(listener), flagcxSuccess);
      if (destination)
        EXPECT_EQ(deviceAdaptor->gdrMemFree(destination, nullptr),
                  flagcxSuccess);
    }
  } r{net};
  char handle[FLAGCX_NET_HANDLE_MAXSIZE] = {};
  PROBE(net->listen(netDev, handle, &r.listener));
  auto connection = connectLoopback(net, netDev, handle, r.listener);
  r.sender = connection.sendComm;
  r.receiver = connection.recvComm;
  ASSERT_EQ(connection.connectResult, flagcxSuccess);
  ASSERT_EQ(connection.acceptResult, flagcxSuccess);
  ASSERT_NE(r.sender, nullptr);
  ASSERT_NE(r.receiver, nullptr);
  PROBE(deviceAdaptor->gdrMemAlloc(&r.destination, allocationBytes, nullptr));
  PROBE(net->regMr(r.sender, buffer, allocationBytes, FLAGCX_PTR_CUDA,
                   FLAGCX_NET_MR_FLAG_NONE, &r.sendMr));
  PROBE(net->regMr(r.receiver, r.destination, allocationBytes, FLAGCX_PTR_CUDA,
                   FLAGCX_NET_MR_FLAG_NONE, &r.recvMr));
  TestWindow source, destination;
  PROBE(source.init(net, buffer, allocationBytes, 0, r.sendMr));
  PROBE(destination.init(net, r.destination, allocationBytes, 1, r.recvMr));
  std::vector<uint8_t> pattern(allocationBytes), zero(allocationBytes, 0),
      actual(allocationBytes);
  for (size_t i = 0; i < pattern.size(); ++i)
    pattern[i] = (i * 17 + 7) & 255;
  PROBE(deviceAdaptor->deviceMemcpy(buffer, pattern.data(), pattern.size(),
                                    flagcxMemcpyHostToDevice, nullptr,
                                    nullptr));
  PROBE(deviceAdaptor->deviceMemcpy(r.destination, zero.data(), zero.size(),
                                    flagcxMemcpyHostToDevice, nullptr,
                                    nullptr));
  const size_t bytes = 4096, srcOffset = 127,
               dstOffset = multi ? alignment - 17 : 8193;
  void *request = nullptr;
  PROBE(net->iput(r.sender, srcOffset, dstOffset, bytes, 0, 1, source.opaque(),
                  destination.opaque(), &request));
  ASSERT_NE(request, nullptr);
  // A timed-out outstanding request must not be deregistered underneath DMA.
  if (probeCall("wait PUT request",
                [&] { return waitRequest(net, request); }) != flagcxSuccess)
    std::_Exit(124);
  PROBE(deviceAdaptor->deviceMemcpy(actual.data(), r.destination, actual.size(),
                                    flagcxMemcpyDeviceToHost, nullptr,
                                    nullptr));
  std::copy_n(pattern.begin() + srcOffset, bytes, zero.begin() + dstOffset);
  EXPECT_EQ(actual, zero);
  PROBE(deviceAdaptor->deviceMemset(buffer, 0, allocationBytes, flagcxMemDevice,
                                    nullptr));
  request = nullptr;
  PROBE(net->iget(r.sender, dstOffset, srcOffset, bytes, 1, 0,
                  destination.opaque(), source.opaque(), &request));
  ASSERT_NE(request, nullptr);
  if (probeCall("wait GET request",
                [&] { return waitRequest(net, request); }) != flagcxSuccess)
    std::_Exit(124);
  PROBE(deviceAdaptor->deviceMemcpy(actual.data(), buffer, actual.size(),
                                    flagcxMemcpyDeviceToHost, nullptr,
                                    nullptr));
  std::vector<uint8_t> expected(allocationBytes, 0);
  std::copy_n(pattern.begin() + srcOffset, bytes, expected.begin() + srcOffset);
  EXPECT_EQ(actual, expected);
}

TEST_F(VmmProbe, VaMrReadWrite) { netReadWrite(false); }
TEST_F(VmmProbe, VaMrMultiSegmentReadWrite) { netReadWrite(true); }
