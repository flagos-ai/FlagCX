/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 *
 * Host-only checks for the default Net Provider dispatch contract.
 ************************************************************************/

#include "device_api/comm_traits.h"
#include "device_api/default_platform_traits.h"
#include "device_api/devnet/default_net_gda_policy.h"

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <type_traits>
#include <typeinfo>

struct OtherPlatform {};
struct MockGdaPlatform {};
static int mockGdaCalls = 0;
struct MockBarrierCall {
  int contextId;
  const std::type_info *operation;
  flagcxDevNetResourceSharingMode_t sharingMode;
};
static MockBarrierCall mockBarrierCalls[8];
static int mockBarrierCallCount = 0;

// Models a participating subgroup without relying on the host thread-index
// macros, which always describe one thread in this test.
struct MockBarrierCoop {
  int rank;
  int groupSize;
  int *syncCalls;

  MockBarrierCoop(int rank = 0, int groupSize = 1, int *syncCalls = nullptr)
      : rank(rank), groupSize(groupSize), syncCalls(syncCalls) {}
  int threadRank() const { return rank; }
  int size() const { return groupSize; }
  void sync() const {
    if (syncCalls != nullptr)
      ++*syncCalls;
  }
};

template <typename T>
struct MockBarrierResult {
  static T get() { return T(); }
};
template <>
struct MockBarrierResult<void> {
  static void get() {}
};

template <>
struct PlatformTraits<MockGdaPlatform> : PlatformTraits<DefaultPlatform> {};

template <>
struct DefaultNetProviderPolicy<MockGdaPlatform> : DefaultNetGdaPolicy {};

template <>
struct DefaultNetProvider<MockGdaPlatform> {
  static constexpr bool isGda = true;
  static uint64_t beginBarrierWait(DefaultNetContext) { return 0; }
  static bool continueBarrierWait(DefaultNetContext, uint64_t) { return true; }

  static bool supportsBarrierFence(DefaultNetContext,
                                   flagcxDevNetFenceLevel fence) {
    return flagcxDevNetFenceValid(fence);
  }

  template <typename NetT>
  static bool isValid(DefaultNetContext ctx, const NetT &, bool = false) {
    return flagcxDevNetResourceSharingModeValid(ctx.resourceSharingMode) &&
           ctx.handle != nullptr;
  }

  template <typename Operation, typename... Args>
  static auto call(DefaultNetContext ctx, Args &&...args)
      -> decltype(DefaultNetProxyApi<Operation>::call(
          ctx, static_cast<Args &&>(args)...)) {
    if (ctx.handle == reinterpret_cast<void *>(0x2000)) {
      assert(mockBarrierCallCount < 8);
      mockBarrierCalls[mockBarrierCallCount++] = {
          ctx.contextId, &typeid(Operation), ctx.resourceSharingMode};
      typedef decltype(DefaultNetProxyApi<Operation>::call(
          ctx, static_cast<Args &&>(args)...)) Result;
      return MockBarrierResult<Result>::get();
    }
    ++mockGdaCalls;
    return DefaultNetProxyApi<Operation>::call(ctx,
                                               static_cast<Args &&>(args)...);
  }
};
static_assert(
    std::is_empty<DefaultNetProviderPolicy<OtherPlatform>::WindowState>::value,
    "non-NVIDIA Window must not carry Provider state");
static_assert(
    std::is_empty<DefaultNetProviderPolicy<OtherPlatform>::CommState>::value,
    "non-NVIDIA Comm must not carry Provider state");
static_assert(
    std::is_empty<DefaultNetProviderPolicy<OtherPlatform>::BarrierState>::value,
    "non-NVIDIA Barrier must not carry Provider state");
static_assert(static_cast<uint32_t>(flagcxDevNetFenceLevel::Relaxed) == 0,
              "Relaxed must retain its existing value");
static_assert(flagcxDevNetResourceSharingDevice == 0,
              "zero-initialized sharing must allow device-wide callers");
static_assert(DefaultNetContext{}.resourceSharingMode ==
                  flagcxDevNetResourceSharingDevice,
              "legacy/default context construction must use Device sharing");
static_assert(std::is_trivially_copyable<DefaultNetContext>::value,
              "dispatch contexts must remain safe to copy by value");
static_assert(!flagcxDevNetResourceSharingModeValid(
                  static_cast<flagcxDevNetResourceSharingMode_t>(3)),
              "unknown sharing modes must be rejected");
static_assert(flagcxDevNetFenceHas(flagcxDevNetFenceLevel::Put |
                                       flagcxDevNetFenceLevel::Get,
                                   flagcxDevNetFenceLevel::Put),
              "Put and Get fence bits must compose");
static_assert(flagcxDevNetFenceValid(flagcxDevNetFenceLevel::Put |
                                     flagcxDevNetFenceLevel::Get),
              "composed fence must be valid");
static_assert(!flagcxDevNetFenceValid(static_cast<flagcxDevNetFenceLevel>(4)),
              "unknown fence bits must be rejected");

struct MockNet {
  mutable int calls[11] = {};

  template <typename... Args>
  void proxyPut(Args &&...) const {
    ++calls[0];
  }
  template <typename... Args>
  void proxyGet(Args &&...) const {
    ++calls[1];
  }
  template <typename... Args>
  void proxyPutValue(Args &&...) const {
    ++calls[2];
  }
  template <typename... Args>
  void proxySignal(Args &&...) const {
    ++calls[3];
  }
  template <typename... Args>
  void proxyFlush(Args &&...) const {
    ++calls[4];
  }
  template <typename... Args>
  void proxyWaitSignal(Args &&...) const {
    ++calls[5];
  }
  template <typename... Args>
  uint64_t proxyReadSignal(Args &&...) const {
    ++calls[6];
    return 42;
  }
  template <typename... Args>
  void proxyResetSignal(Args &&...) const {
    ++calls[7];
  }
  template <typename... Args>
  void proxyWaitCounter(Args &&...) const {
    ++calls[8];
  }
  template <typename... Args>
  uint64_t proxyReadCounter(Args &&...) const {
    ++calls[9];
    return 43;
  }
  template <typename... Args>
  void proxyResetCounter(Args &&...) const {
    ++calls[10];
  }
};

int main() {
  typedef DeviceAPI::Net Net;
  typedef DefaultNetGdaPolicy Policy;
  DefaultNetContext ctx{};
  MockNet mock;
  DefaultNetContext invalid = ctx;
  invalid.resourceSharingMode =
      static_cast<flagcxDevNetResourceSharingMode_t>(3);
  assert(!DefaultNetProvider<DefaultPlatform>::isValid(invalid, mock, true));

  DefaultNetProvider<DefaultPlatform>::call<Net::PutOp>(ctx, mock);
  DefaultNetProvider<DefaultPlatform>::call<Net::GetOp>(ctx, mock);
  DefaultNetProvider<DefaultPlatform>::call<Net::PutValueOp>(ctx, mock);
  DefaultNetProvider<DefaultPlatform>::call<Net::SignalOp>(ctx, mock);
  DefaultNetProvider<DefaultPlatform>::call<Net::FlushOp>(ctx, mock);
  DefaultNetProvider<DefaultPlatform>::call<Net::WaitSignalOp>(ctx, mock);
  assert(DefaultNetProvider<DefaultPlatform>::call<Net::ReadSignalOp>(
             ctx, mock) == 42);
  DefaultNetProvider<DefaultPlatform>::call<Net::ResetSignalOp>(ctx, mock);
  DefaultNetProvider<DefaultPlatform>::call<Net::WaitCounterOp>(ctx, mock);
  assert(DefaultNetProvider<DefaultPlatform>::call<Net::ReadCounterOp>(
             ctx, mock) == 43);
  DefaultNetProvider<DefaultPlatform>::call<Net::ResetCounterOp>(ctx, mock);
  for (int i = 0; i != 11; ++i)
    assert(mock.calls[i] == 1);

  static_assert(!DefaultNetProvider<DefaultPlatform>::isGda, "None uses Proxy");
  static_assert(DefaultNetProvider<MockGdaPlatform>::isGda,
                "GDA is a type binding");
  DefaultNetContext gdaContext{reinterpret_cast<void *>(0x1000), 0,
                               flagcxDevNetResourceSharingDevice, true};
  DefaultNetProvider<MockGdaPlatform>::call<Net::PutOp>(gdaContext, mock);
  assert(mockGdaCalls == 1 && mock.calls[0] == 2);
  DefaultNetDispatch<MockGdaPlatform>::call<Net::PutOp>(ctx, mock);
  assert(mockGdaCalls == 1 && mock.calls[0] == 3);
  assert(DefaultNetDispatch<MockGdaPlatform>::isValid(ctx, mock, true));
  assert(!DefaultNetDispatch<DefaultPlatform>::isValid(gdaContext, mock, true));

  uint64_t signal = 0, counter = 0, fifo = 0;
  DeviceAPI::Comm proxyComm{};
  proxyComm.contextCount = 1;
  proxyComm.signalCount = 1;
  proxyComm.counterCount = 1;
  proxyComm.signalBuffer = &signal;
  proxyComm.counterBuffer = &counter;
  proxyComm.fifoBuffers[0] = &fifo;
  assert(DeviceAPI::Net(proxyComm, 0).isValid());
  proxyComm.fifoBuffers[0] = nullptr;
  assert(!DeviceAPI::Net(proxyComm, 0).isValid());
  assert(DeviceAPI::Net(proxyComm, 0).term(DeviceAPI::CoopThread{}) ==
         flagcxInvalidUsage);
  proxyComm.fifoBuffers[0] = &fifo;
  assert(DeviceAPI::Net(proxyComm, 0).requireProxyFifo() == flagcxSuccess);
  proxyComm.signalBuffer = nullptr;
  assert(!DeviceAPI::Net(proxyComm, 0).isValid());
  proxyComm.signalBuffer = &signal;
  proxyComm.counterBuffer = nullptr;
  assert(!DeviceAPI::Net(proxyComm, 0).isValid());
  proxyComm.counterCount = 0;
  assert(DeviceAPI::Net(proxyComm, 0).isValid());

  typedef CommTraits<DefaultBackend<MockGdaPlatform>> MockGdaApi;
  MockGdaApi::Comm gdaComm{};
  gdaComm.contextCount = 1;

  gdaComm.netUseGda = true;
  gdaComm.gdaDeviceContext = reinterpret_cast<void *>(0x1000);
  MockGdaApi::Net gdaNet(gdaComm, 0);
  assert(gdaNet.fifoBuffer == nullptr);
  assert(gdaNet.isValid());
  MockGdaApi::CoopThread gdaThread;
  MockGdaApi::Window gdaWindow{};
  assert(gdaNet.send(gdaThread, gdaWindow, 0, 0, flagcxInt32, 0) ==
         flagcxNotSupported);
  assert(gdaNet.recv(gdaThread, gdaWindow, 0, 0, flagcxInt32, 0) ==
         flagcxNotSupported);
  assert(gdaNet.term(gdaThread) == flagcxNotSupported);
  assert(gdaNet.wait(gdaThread) == flagcxNotSupported);
  gdaComm.fifoBuffers[0] = &fifo;
  MockGdaApi::Net gdaWithProxyFifo(gdaComm, 0);
  assert(gdaWithProxyFifo.fifoBuffer == &fifo);
  assert(gdaWithProxyFifo.requireProxyFifo() == flagcxNotSupported);
  assert(gdaWithProxyFifo.send(gdaThread, gdaWindow, 0, 0, flagcxInt32, 0) ==
         flagcxNotSupported);
  assert(gdaWithProxyFifo.recv(gdaThread, gdaWindow, 0, 0, flagcxInt32, 0) ==
         flagcxNotSupported);
  assert(gdaWithProxyFifo.term(gdaThread) == flagcxNotSupported);
  assert(gdaWithProxyFifo.wait(gdaThread) == flagcxNotSupported);
  gdaComm.fifoBuffers[0] = nullptr;
  gdaComm.gdaDeviceContext = nullptr;
  assert(!MockGdaApi::Net(gdaComm, 0).isValid());
  gdaComm.gdaDeviceContext = reinterpret_cast<void *>(0x1000);

  assert(MockGdaApi::Net(gdaComm, 0).isValid());
  gdaComm.netUseGda = false;
  gdaComm.fifoBuffers[0] = &fifo;
  MockGdaApi::Net fallback(gdaComm, 0);
  assert(!fallback.context().useGda && fallback.context().handle == &fifo);
  assert(fallback.requireProxyFifo() == flagcxSuccess);
  gdaComm.netUseGda = true;
  gdaComm.fifoBuffers[0] = nullptr;

  // A strong barrier dispatches only on its bound context. Put drains before
  // signaling; Get drains after all signals have been observed.
  uint64_t barrierSignals[16] = {};
  uint64_t barrierShadows[16] = {};
  MockGdaApi::Comm barrierComm{};
  barrierComm.contextCount = 2;

  barrierComm.netUseGda = true;
  barrierComm.gdaDeviceContext = reinterpret_cast<void *>(0x2000);
  barrierComm.signalBuffer = barrierSignals;
  barrierComm.shadowBuffer = barrierShadows;
  barrierComm.signalCount = 8;
  barrierComm.interBarrierCount = 3;
  barrierComm.nRanks = 2;
  barrierComm.rank = 0;
  barrierComm.nTeamRanks = 2;
  barrierComm.teamRank = 0;
  barrierComm.intraSize = 1;
  barrierComm.intraRank = 0;
  MockGdaApi::Net barrierNet(barrierComm, 1);
  MockGdaApi::Comm gdaWorldComm = barrierComm;
  gdaWorldComm.rank = 1;
  gdaWorldComm.nRanks = 4;
  gdaWorldComm.nTeamRanks = 2;
  gdaWorldComm.teamRank = 0;
  gdaWorldComm.intraSize = 2;
  gdaWorldComm.intraRank = 1;
  uint64_t gdaWorldEpochs[FLAGCX_DEVICE_CTA_COUNT] = {};
  gdaWorldComm.epochBuffer = gdaWorldEpochs;
  gdaWorldComm.nBarriers = FLAGCX_DEVICE_CTA_COUNT;
  MockGdaApi::Net gdaWorldNet(gdaWorldComm, 0);
  typedef MockGdaApi::Barrier<flagcxTeamTagWorld, MockGdaApi::CoopThread>
      MockWorldBarrier;
  MockWorldBarrier gdaWorld(MockGdaApi::CoopThread{}, flagcxTeamTagWorld{},
                            gdaWorldNet, gdaWorldComm, 0, false, 1);
  // World keeps its intra + Inter composition with either Net provider.
  assert(gdaWorld._nInterPeers == 1);
  assert(gdaWorld._inter._nTeamRanks == 2);
  assert(gdaWorld._inter._teamRank == 0);
  assert(gdaWorld._inter._stride == 2);
  assert(gdaWorld._inter._localRank == 1);
  typedef MockGdaApi::Barrier<flagcxTeamTagInter, MockGdaApi::CoopThread>
      MockBarrier;
  MockBarrier barrier(MockGdaApi::CoopThread{}, barrierNet, barrierComm,
                      MockGdaApi::Team{2, 0, 1}, 0, 1);
  assert(barrierNet.getResourceSharingMode() ==
         flagcxDevNetResourceSharingDevice);
  assert(barrier.context().resourceSharingMode ==
         flagcxDevNetResourceSharingDevice);
  // Per-Net selection must survive dispatch and both Barrier constructors;
  // other Net objects on the same communicator retain their own defaults.
  for (auto mode :
       {flagcxDevNetResourceSharingDevice, flagcxDevNetResourceSharingBlock,
        flagcxDevNetResourceSharingThread}) {
    MockGdaApi::Net selected(barrierComm, 1, mode);
    assert(selected.getResourceSharingMode() == mode);
    assert(selected.context().resourceSharingMode == mode);
    MockBarrier selectedBarrier(MockGdaApi::CoopThread{}, selected, barrierComm,
                                MockGdaApi::Team{2, 0, 1}, 0, 1);
    assert(selectedBarrier.context().resourceSharingMode == mode);
    MockGdaApi::Net selectedWorldNet(gdaWorldComm, 1, mode);
    MockWorldBarrier selectedWorld(MockGdaApi::CoopThread{},
                                   flagcxTeamTagWorld{}, selectedWorldNet,
                                   gdaWorldComm, 0, false, 1);
    assert(selectedWorld._inter.context().resourceSharingMode == mode);
    mockBarrierCallCount = 0;
    selectedBarrier.sync();
    assert(mockBarrierCallCount == 3);
    for (int i = 0; i < mockBarrierCallCount; ++i)
      assert(mockBarrierCalls[i].sharingMode == mode);
    assert(barrierNet.getResourceSharingMode() ==
           flagcxDevNetResourceSharingDevice);
  }
  mockBarrierCallCount = 0;
  barrier.sync(flagcxDeviceMemoryOrderAcqRel,
               flagcxDevNetFenceLevel::Put | flagcxDevNetFenceLevel::Get);
  assert(mockBarrierCallCount == 4);
  assert(*mockBarrierCalls[0].operation == typeid(MockGdaApi::Net::FlushOp));
  assert(*mockBarrierCalls[1].operation == typeid(MockGdaApi::Net::SignalOp));
  assert(*mockBarrierCalls[2].operation ==
         typeid(MockGdaApi::Net::WaitSignalOp));
  assert(*mockBarrierCalls[3].operation == typeid(MockGdaApi::Net::FlushOp));
  for (int i = 0; i < mockBarrierCallCount; ++i)
    assert(mockBarrierCalls[i].contextId == 1);

  mockBarrierCallCount = 0;
  MockBarrier relaxedBarrier(MockGdaApi::CoopThread{}, barrierNet, barrierComm,
                             MockGdaApi::Team{2, 0, 1}, 1, 1);
  relaxedBarrier.sync();
  assert(mockBarrierCallCount == 3);
  assert(*mockBarrierCalls[0].operation == typeid(MockGdaApi::Net::SignalOp));

  // Scalar IR reconstructs the barrier for each split call. Relaxed must not
  // depend on state kept in the first Barrier object.
  mockBarrierCallCount = 0;
  MockBarrier splitArrive(MockGdaApi::CoopThread{}, barrierNet, barrierComm,
                          MockGdaApi::Team{2, 0, 1}, 1, 1);
  MockBarrier splitWait(MockGdaApi::CoopThread{}, barrierNet, barrierComm,
                        MockGdaApi::Team{2, 0, 1}, 1, 1);
  splitArrive.arrive();
  splitWait.wait();
  assert(mockBarrierCallCount == 3);
  assert(*mockBarrierCalls[0].operation == typeid(MockGdaApi::Net::SignalOp));
  assert(*mockBarrierCalls[1].operation ==
         typeid(MockGdaApi::Net::WaitSignalOp));
  assert(*mockBarrierCalls[2].operation == typeid(MockGdaApi::Net::FlushOp));

  mockBarrierCallCount = 0;
  barrierComm.nTeamRanks = 1;
  barrierComm.nRanks = 1;
  MockGdaApi::Net singleNet(barrierComm, 1);
  MockBarrier singleBarrier(MockGdaApi::CoopThread{}, singleNet, barrierComm,
                            MockGdaApi::Team{1, 0, 1}, 2, 0);
  singleBarrier.sync(flagcxDeviceMemoryOrderAcqRel,
                     flagcxDevNetFenceLevel::Get);
  assert(mockBarrierCallCount == 1);
  assert(*mockBarrierCalls[0].operation == typeid(MockGdaApi::Net::FlushOp));
  assert(mockBarrierCalls[0].contextId == 1);
  mockBarrierCallCount = 0;
  singleBarrier.sync();
  assert(mockBarrierCallCount == 0);

  typedef MockGdaApi::Barrier<flagcxTeamTagInter, MockBarrierCoop>
      SubgroupBarrier;
  int subgroupSyncCalls = 0;
  MockBarrierCoop nonleader(1, 2, &subgroupSyncCalls);
  SubgroupBarrier nonleaderBarrier(nonleader, singleNet, barrierComm,
                                   MockGdaApi::Team{1, 0, 1}, 2, 0);
  nonleaderBarrier.sync(flagcxDeviceMemoryOrderAcqRel,
                        flagcxDevNetFenceLevel::Put |
                            flagcxDevNetFenceLevel::Get);
  assert(subgroupSyncCalls == 4);
  assert(mockBarrierCallCount == 0);

  subgroupSyncCalls = 0;
  MockBarrierCoop leader(0, 2, &subgroupSyncCalls);
  SubgroupBarrier leaderBarrier(leader, singleNet, barrierComm,
                                MockGdaApi::Team{1, 0, 1}, 2, 0);
  leaderBarrier.sync(flagcxDeviceMemoryOrderAcqRel,
                     flagcxDevNetFenceLevel::Put | flagcxDevNetFenceLevel::Get);
  assert(subgroupSyncCalls == 4);
  assert(mockBarrierCallCount == 2);
  assert(*mockBarrierCalls[0].operation == typeid(MockGdaApi::Net::FlushOp));
  assert(*mockBarrierCalls[1].operation == typeid(MockGdaApi::Net::FlushOp));

  // With four team ranks, subgroup lane one owns only the second peer.
  mockBarrierCallCount = 0;
  subgroupSyncCalls = 0;
  barrierComm.nTeamRanks = 4;
  barrierComm.nRanks = 4;
  barrierComm.interBarrierCount = 2;
  MockGdaApi::Net subgroupNet(barrierComm, 1);
  SubgroupBarrier multiPeerBarrier(nonleader, subgroupNet, barrierComm,
                                   MockGdaApi::Team{4, 0, 1}, 0, 3);
  multiPeerBarrier.sync();
  assert(subgroupSyncCalls == 5);
  assert(mockBarrierCallCount == 2);
  assert(*mockBarrierCalls[0].operation == typeid(MockGdaApi::Net::SignalOp));
  assert(*mockBarrierCalls[1].operation ==
         typeid(MockGdaApi::Net::WaitSignalOp));

  // Two independent groups use distinct channels within the allocated range.
  const uint64_t channel0Epoch = barrierShadows[8 + 1];
  const uint64_t channel1Epoch = barrierShadows[8 + 4 + 1];
  mockBarrierCallCount = 0;
  SubgroupBarrier channel0(leader, subgroupNet, barrierComm,
                           MockGdaApi::Team{4, 0, 1}, 0, 3);
  channel0.sync();
  assert(barrierShadows[8 + 1] == channel0Epoch + 1);
  assert(barrierShadows[8 + 4 + 1] == channel1Epoch);
  mockBarrierCallCount = 0;
  SubgroupBarrier channel1(leader, subgroupNet, barrierComm,
                           MockGdaApi::Team{4, 0, 1}, 1, 3);
  channel1.sync();
  assert(barrierShadows[8 + 4 + 1] == channel1Epoch + 1);

  struct HostComm : Policy::CommState {
  } hostComm;
  struct HostWindow : Policy::WindowState {
  } hostWindow;
  struct HostInput {
    int netUseGda;
    void *gdaDeviceContext;
  } hostInput{0, nullptr};
  Policy::initComm(hostComm, hostInput);
  Policy::initWindow(hostWindow);
  assert(hostComm.gdaDeviceContext == nullptr);
  assert(hostWindow.netDeviceHandle == nullptr);
  assert(Policy::makeContext(hostComm, reinterpret_cast<void *>(0x1000), 2)
             .handle == reinterpret_cast<void *>(0x1000));
  hostInput = {1, reinterpret_cast<void *>(0x2000)};
  Policy::initComm(hostComm, hostInput);
  assert(Policy::makeContext(hostComm, reinterpret_cast<void *>(0x1000), 2)
             .handle == reinterpret_cast<void *>(0x2000));
  return 0;
}
