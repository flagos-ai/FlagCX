/*************************************************************************
 * Copyright (c) 2026 BAAI. All rights reserved.
 * Real GDA host ownership and public APIs, with fallible provider releases.
 ************************************************************************/
#include <algorithm>
#include <cstdlib>
#include <gtest/gtest.h>
#include <vector>

// White-box ownership fixtures; the executable exports this real adaptor to
// libflagcx, so public API calls exercise the same implementation below.
#include "../../../flagcx/adaptor/device_api/devnet/xpu_netio_gda_host.cc"
#include "dev_api_backend.h"
#include "global_comm.h"

namespace {
std::vector<int> events;
enum { Drain, Converge, Unpublish, Qp, Mr, RemoteMr, Context, Device, Free };
flagcxResult_t peerStatus;
bool failUnpublish, failDevice, failMr, failFree;
xpu_netio_status_t providerError;
unsigned convergeCalls, failPeerAt;
uint32_t *timeoutOnDrain;
std::vector<void *> freed;
flagcxResult_t mockSynchronize() {
  events.push_back(Drain);
  if (timeoutOnDrain != nullptr) {
    *timeoutOnDrain = XPU_NETIO_TIMEOUT;
    timeoutOnDrain = nullptr;
  }
  return flagcxSuccess;
}
flagcxResult_t mockFree(void *p, flagcxMemType_t, flagcxStream_t) {
  events.push_back(Free);
  if (failFree) {
    failFree = false;
    return flagcxRemoteError;
  }
  freed.push_back(p);
  std::free(p);
  return flagcxSuccess;
}
class GdaHostControl : public ::testing::Test {
protected:
  flagcxComm comm{};
  flagcxDevCommInternal dev{};
  XpuNetioGda::CommOwner owner{};
  uint32_t control[2]{};
  flagcxDeviceAdaptor mock{};
  flagcxDeviceAdaptor *saved;
  void SetUp() override {
    events.clear();
    peerStatus = flagcxSuccess;
    failUnpublish = failDevice = failMr = failFree = false;
    providerError = XPU_NETIO_SUCCESS;
    convergeCalls = failPeerAt = 0;
    timeoutOnDrain = nullptr;
    freed.clear();
    saved = deviceAdaptor;
    mock.deviceSynchronize = mockSynchronize;
    mock.deviceFree = mockFree;
    deviceAdaptor = &mock;
    comm.commType = flagcxCommunicatorHomo;
    comm.rank = 0;
    comm.nranks = 2;
    comm.devNetCommHandle = &dev;
    dev.gdaProvider = 1;
    dev.gdaHostOwner = &owner;
    owner.comm = &comm;
    owner.hostControl = owner.deviceControl = control;
    owner.errorFlag = control;
    xpu_netio_device_handle_init(&owner.published);
  }
  void TearDown() override { deviceAdaptor = saved; }
  XpuNetioGda::WindowOwner *makeWindow(flagcxWindow &host,
                                       XpuNetioGda::CommOwner *o = nullptr) {
    if (o == nullptr)
      o = &owner;
    auto *w = new XpuNetioGda::WindowOwner;
    w->comm = o;
    w->deviceView = static_cast<flagcxXpuNetioDeviceWindow *>(
        std::calloc(1, sizeof(flagcxXpuNetioDeviceWindow)));
    w->mr = reinterpret_cast<xpu_netio_mr_t *>(0x4000);
    host.userBase = reinterpret_cast<void *>(0x5000);
    host.userSize = 4096;
    host.gdaWindowOwner = w;
    host.gdaDeviceView = w->deviceView;
    host.gdaRegistrationActive = true;
    ++o->liveWindows;
    return w;
  }
  flagcxDevComm_t heapComm() {
    auto *d = static_cast<flagcxDevComm_t>(std::calloc(1, sizeof(dev)));
    pthread_mutex_init(&d->cachedPtrMutex, nullptr);
    d->barrierIpcIndex = d->signalIpcSlot = -1;
    auto *o = new XpuNetioGda::CommOwner;
    o->comm = &comm;
    o->context = reinterpret_cast<xpu_netio_context_t *>(0x1000);
    o->device = reinterpret_cast<xpu_netio_device_t *>(0x2000);
    o->qps.push_back(reinterpret_cast<xpu_netio_qp_t *>(0x3000));
    o->publishAttempted = true;
    xpu_netio_device_handle_init(&o->published);
    o->hostControl = static_cast<uint32_t *>(std::calloc(2, sizeof(uint32_t)));
    o->deviceControl = o->errorFlag = o->hostControl;
    o->deviceView = static_cast<flagcxXpuNetioDeviceContext *>(
        std::calloc(1, sizeof(flagcxXpuNetioDeviceContext)));
    o->signals = new XpuNetioGda::WindowOwner;
    o->signals->comm = o;
    o->signals->mr = reinterpret_cast<xpu_netio_mr_t *>(0x4000);
    d->gdaProvider = 1;
    d->gdaHostOwner = o;
    d->gdaDeviceContext = o->deviceView;
    comm.devNetCommHandle = d;
    return d;
  }
};

TEST_F(GdaHostControl, PublicAsyncErrorObservesGdaAndRetainsFirstFailure) {
  flagcxResult_t error = flagcxInternalError;
  ASSERT_EQ(flagcxCommGetAsyncError(&comm, &error), flagcxSuccess);
  EXPECT_EQ(error, flagcxSuccess);
  control[0] = XPU_NETIO_INVALID_ARGUMENT;
  ASSERT_EQ(flagcxCommGetAsyncError(&comm, &error), flagcxSuccess);
  EXPECT_EQ(error, flagcxInvalidArgument);
  control[0] = XPU_NETIO_TIMEOUT;
  ASSERT_EQ(flagcxCommGetAsyncError(&comm, &error), flagcxSuccess);
  EXPECT_EQ(error, flagcxInvalidArgument);
  EXPECT_TRUE(events.empty()); // No CUDA copies or stream synchronization.
}

TEST_F(GdaHostControl, PublicAbortPublishesCancellationWithoutVendorComm) {
  ASSERT_EQ(flagcxCommAbort(&comm), flagcxSuccess);
  EXPECT_EQ(control[1], 1u);
  EXPECT_EQ(control[0], 0u); // Host never writes the GPU-owned error word.
  flagcxResult_t error;
  ASSERT_EQ(flagcxCommGetAsyncError(&comm, &error), flagcxSuccess);
  EXPECT_EQ(error, flagcxRemoteError);
  EXPECT_EQ(XpuNetioGda::clearError(&dev), flagcxInvalidUsage);
  EXPECT_TRUE(events.empty());
}

TEST_F(GdaHostControl,
       PeerDrainFailureRetainsAllRegistrationsForCollectiveRetry) {
  auto *d = heapComm();
  peerStatus = flagcxRemoteError;
  EXPECT_EQ(flagcxDevCommDestroy(&comm, d), flagcxRemoteError);
  EXPECT_EQ(events, (std::vector<int>{Drain, Converge}));
  EXPECT_EQ(comm.devNetCommHandle, nullptr);
  EXPECT_EQ(comm.pendingDevCommCleanup, d);
  EXPECT_EQ(d->gdaQuiesced, 0);
  peerStatus = flagcxSuccess;
  EXPECT_EQ(flagcxDevCommDestroy(&comm, d), flagcxSuccess);
  EXPECT_EQ(comm.pendingDevCommCleanup, nullptr);
  auto qp = std::find(events.begin(), events.end(), Qp);
  auto mr = std::find(events.begin(), events.end(), Mr);
  ASSERT_NE(qp, events.end());
  ASSERT_NE(mr, events.end());
  EXPECT_LT(qp, mr);
  EXPECT_EQ(*(qp + 1), Converge); // All ranks acknowledge NIC shutdown first.
}

TEST_F(GdaHostControl, UnpublishFailureNeverReleasesMrAndCanBeRetried) {
  auto *d = heapComm();
  failUnpublish = true;
  EXPECT_EQ(flagcxDevCommDestroy(&comm, d), flagcxRemoteError);
  EXPECT_EQ(std::count(events.begin(), events.end(), Mr), 0);
  EXPECT_EQ(std::count(events.begin(), events.end(), Free), 0);
  EXPECT_EQ(comm.devNetCommHandle, nullptr);
  EXPECT_EQ(flagcxDevCommDestroy(&comm, d), flagcxSuccess);
  EXPECT_EQ(std::count(events.begin(), events.end(), Mr), 1);
}

TEST_F(GdaHostControl, PartialReleaseIsCleanupOnlyAndDoesNotDoubleFreeOnRetry) {
  auto *d = heapComm();
  d->cachedDevicePtr = std::malloc(16);
  failDevice = true;
  EXPECT_EQ(flagcxDevCommDestroy(&comm, d), flagcxRemoteError);
  EXPECT_EQ(d->gdaQuiesced, 1);
  EXPECT_EQ(comm.devNetCommHandle, nullptr);
  void *pointer = reinterpret_cast<void *>(1);
  EXPECT_EQ(flagcxDevCommGetDevicePtr(d, &pointer), flagcxInvalidUsage);
  EXPECT_EQ(pointer, nullptr);
  EXPECT_EQ(std::count(events.begin(), events.end(), Mr), 1);
  EXPECT_EQ(std::count(events.begin(), events.end(), Free), 1);
  std::free(d->cachedDevicePtr);
  d->cachedDevicePtr = nullptr;
  EXPECT_EQ(flagcxDevCommDestroy(&comm, d), flagcxSuccess);
  EXPECT_EQ(std::count(events.begin(), events.end(), Mr), 1);
  EXPECT_EQ(std::count(events.begin(), events.end(), Free), 2);
  EXPECT_EQ(std::count(events.begin(), events.end(), Qp), 1);
}

TEST_F(GdaHostControl, WindowMrFailureRejectsRebindAndCachedViewUntilRetry) {
  flagcxWindow host{};
  auto *w = makeWindow(host);
  w->deviceRegions = static_cast<xpu_netio_remote_region_t *>(std::malloc(16));
  w->remote = {reinterpret_cast<xpu_netio_remote_mr_t *>(0x6000),
               reinterpret_cast<xpu_netio_remote_mr_t *>(0x7000)};
  void *releasedView = w->deviceView;
  failMr = true;
  ASSERT_EQ(XpuNetioGda::prepareWindowDeregister(&host), flagcxSuccess);
  EXPECT_EQ(XpuNetioGda::deregisterWindow(&host), flagcxRemoteError);
  EXPECT_TRUE(w->retiring);
  EXPECT_EQ(w->deviceView, nullptr);
  EXPECT_EQ(host.gdaDeviceView, nullptr);
  EXPECT_EQ(host.gdaWindowOwner, w);
  EXPECT_EQ(owner.liveWindows, 1u);
  flagcxDevMemInternal view{};
  DeviceAPI::Window deviceWindow{};
  view.rawPtr = host.userBase;
  view.winHandle = &host;
  EXPECT_EQ(XpuNetioGda::bindDevMem(&host, &view, 4096, &deviceWindow),
            flagcxInvalidUsage);
  EXPECT_FALSE(DefaultNetGdaPolicy::hasWindowAccess(deviceWindow));
  // Model a cached DevMem whose publication was detached during cleanup.
  view.cachedDevicePtr = std::malloc(16);
  void *pointer = reinterpret_cast<void *>(1);
  EXPECT_EQ(flagcxDevMemGetDevicePtr(&view, &pointer), flagcxInvalidUsage);
  EXPECT_EQ(pointer, nullptr);
  std::free(view.cachedDevicePtr);
  view.cachedDevicePtr = nullptr;
  flagcxDevMem_t newView = reinterpret_cast<flagcxDevMem_t>(1);
  EXPECT_EQ(flagcxDevMemCreate(&comm, host.userBase, 4096, &host, &newView),
            flagcxInvalidUsage);
  EXPECT_EQ(newView, nullptr);
  EXPECT_EQ(host.gdaDevMemRefs, 0u);
  ASSERT_EQ(XpuNetioGda::deregisterWindow(&host), flagcxSuccess);
  EXPECT_EQ(host.gdaWindowOwner, nullptr);
  EXPECT_EQ(owner.liveWindows, 0u);
  EXPECT_EQ(std::count(freed.begin(), freed.end(), releasedView), 1);
  EXPECT_EQ(std::count(events.begin(), events.end(), RemoteMr), 2);
  EXPECT_EQ(std::count(events.begin(), events.end(), Mr), 2);
  EXPECT_EQ(std::count(events.begin(), events.end(), Qp), 0);
}

TEST_F(GdaHostControl, WindowFirstFreeFailureIsAlreadyCleanupOnly) {
  flagcxWindow host{};
  auto *w = makeWindow(host);
  void *original = w->deviceView;
  failFree = true;
  EXPECT_EQ(XpuNetioGda::deregisterWindow(&host), flagcxRemoteError);
  EXPECT_EQ(w->deviceView, original); // Failed free retains its ownership.
  EXPECT_TRUE(w->retiring);
  EXPECT_EQ(host.gdaDeviceView, nullptr);
  EXPECT_TRUE(freed.empty());
  flagcxDevMemInternal view{};
  DeviceAPI::Window deviceWindow{};
  view.rawPtr = host.userBase;
  EXPECT_EQ(XpuNetioGda::bindDevMem(&host, &view, 4096, &deviceWindow),
            flagcxInvalidUsage);
  EXPECT_EQ(XpuNetioGda::deregisterWindow(&host), flagcxSuccess);
  EXPECT_EQ(std::count(freed.begin(), freed.end(), original), 1);
}

TEST_F(GdaHostControl, HealthyWindowReleaseKeepsCommunicatorAndQpsAlive) {
  flagcxWindow host{};
  makeWindow(host);
  owner.context = reinterpret_cast<xpu_netio_context_t *>(0x1000);
  owner.qps = {reinterpret_cast<xpu_netio_qp_t *>(0x3000)};
  owner.publishAttempted = true;
  EXPECT_EQ(XpuNetioGda::prepareWindowDeregister(&host), flagcxSuccess);
  EXPECT_EQ(XpuNetioGda::deregisterWindow(&host), flagcxSuccess);
  EXPECT_TRUE(owner.publishAttempted);
  EXPECT_NE(owner.qps[0], nullptr);
  EXPECT_FALSE(owner.draining);
  EXPECT_EQ(std::count(events.begin(), events.end(), Qp), 0);
  EXPECT_EQ(std::count(events.begin(), events.end(), Unpublish), 0);
}

TEST_F(GdaHostControl, ErrorWindowCannotReleaseUntilCollectiveNicRetirement) {
  for (int failure = 0; failure != 4; ++failure) {
    SCOPED_TRACE(failure);
    events.clear();
    owner.asyncError = flagcxSuccess;
    owner.draining = owner.nicQuiesced = false;
    dev.gdaDraining = dev.gdaQuiesced = 0;
    control[0] = failure == 0 ? XPU_NETIO_TIMEOUT : 0;
    control[1] = failure == 1 ? 1 : 0;
    if (failure == 2)
      owner.asyncError = flagcxRemoteError;
    providerError = failure == 3 ? XPU_NETIO_PROVIDER_ERROR : XPU_NETIO_SUCCESS;
    owner.context = reinterpret_cast<xpu_netio_context_t *>(0x1000);
    owner.qps = {reinterpret_cast<xpu_netio_qp_t *>(0x3000)};
    owner.publishAttempted = true;
    flagcxWindow host{};
    makeWindow(host);
    comm.devNetWindows = &host;
    // The public collective entry must converge failure without releasing MRs.
    EXPECT_EQ(flagcxCommWindowDeregister(&comm, &host), flagcxRemoteError);
    EXPECT_EQ(XpuNetioGda::deregisterWindow(&host), flagcxRemoteError);
    EXPECT_EQ(std::count(events.begin(), events.end(), Mr), 0);
    EXPECT_EQ(std::count(events.begin(), events.end(), Free), 0);
    EXPECT_TRUE(owner.publishAttempted);
    EXPECT_NE(owner.qps[0], nullptr);
    EXPECT_NE(host.gdaDeviceView, nullptr); // No destructive step was entered.
    ASSERT_EQ(XpuNetioGda::prepareCommDestroy(&dev), flagcxSuccess);
    EXPECT_TRUE(owner.nicQuiesced);
    ASSERT_EQ(XpuNetioGda::prepareWindowDeregister(&host), flagcxSuccess);
    ASSERT_EQ(XpuNetioGda::deregisterWindow(&host), flagcxSuccess);
    EXPECT_LT(std::find(events.begin(), events.end(), Qp),
              std::find(events.begin(), events.end(), Mr));
    comm.devNetWindows = nullptr;
  }
}

TEST_F(GdaHostControl, ErrorRaisedDuringGpuDrainCannotBeClearedOrReleased) {
  flagcxWindow host{};
  makeWindow(host);
  timeoutOnDrain = control;
  EXPECT_EQ(XpuNetioGda::prepareWindowDeregister(&host), flagcxRemoteError);
  EXPECT_EQ(XpuNetioGda::clearError(&dev), flagcxInvalidUsage);
  EXPECT_EQ(control[0], static_cast<uint32_t>(XPU_NETIO_TIMEOUT));
  EXPECT_EQ(XpuNetioGda::deregisterWindow(&host), flagcxRemoteError);
  EXPECT_EQ(std::count(events.begin(), events.end(), Mr), 0);
  ASSERT_EQ(XpuNetioGda::prepareCommDestroy(&dev), flagcxSuccess);
  EXPECT_EQ(XpuNetioGda::deregisterWindow(&host), flagcxSuccess);
}

TEST_F(GdaHostControl, UnpolledTimeoutCannotBeErasedBeforeWindowRelease) {
  flagcxWindow host{};
  makeWindow(host);
  control[0] = XPU_NETIO_TIMEOUT;
  EXPECT_EQ(owner.asyncError, flagcxSuccess);
  EXPECT_EQ(XpuNetioGda::clearError(&dev), flagcxInvalidUsage);
  EXPECT_EQ(control[0], static_cast<uint32_t>(XPU_NETIO_TIMEOUT));
  EXPECT_EQ(owner.asyncError, flagcxRemoteError);
  EXPECT_EQ(XpuNetioGda::deregisterWindow(&host), flagcxRemoteError);
  EXPECT_EQ(std::count(events.begin(), events.end(), Mr), 0);
  ASSERT_EQ(XpuNetioGda::prepareCommDestroy(&dev), flagcxSuccess);
  EXPECT_EQ(XpuNetioGda::deregisterWindow(&host), flagcxSuccess);
}

TEST_F(GdaHostControl, FailedUnpublishedSetupCanRetryWindowRollback) {
  flagcxWindow host{};
  makeWindow(host);
  // Initialization failed before the public DevComm became runnable.
  comm.devNetCommHandle = nullptr;
  providerError = XPU_NETIO_PROVIDER_ERROR;
  failMr = true;
  EXPECT_EQ(XpuNetioGda::deregisterWindow(&host), flagcxRemoteError);
  EXPECT_EQ(host.gdaDeviceView, nullptr);
  EXPECT_EQ(XpuNetioGda::deregisterWindow(&host), flagcxSuccess);
  EXPECT_EQ(owner.liveWindows, 0u);
}

TEST_F(GdaHostControl,
       LocalQpRetiredButPeerNotRetiredStillBlocksWindowRelease) {
  flagcxWindow host{};
  makeWindow(host);
  control[0] = XPU_NETIO_TIMEOUT;
  owner.context = reinterpret_cast<xpu_netio_context_t *>(0x1000);
  owner.qps = {reinterpret_cast<xpu_netio_qp_t *>(0x3000)};
  owner.publishAttempted = true;
  failPeerAt = 2; // Peer fails the QP retirement convergence, after GPU drain.
  EXPECT_EQ(XpuNetioGda::prepareCommDestroy(&dev), flagcxRemoteError);
  EXPECT_EQ(owner.qps[0], nullptr);
  EXPECT_FALSE(owner.nicQuiesced);
  EXPECT_EQ(XpuNetioGda::prepareWindowDeregister(&host), flagcxInvalidUsage);
  EXPECT_EQ(XpuNetioGda::deregisterWindow(&host), flagcxInvalidUsage);
  EXPECT_EQ(std::count(events.begin(), events.end(), Mr), 0);
  ASSERT_EQ(XpuNetioGda::prepareCommDestroy(&dev), flagcxSuccess);
  ASSERT_TRUE(owner.nicQuiesced);
  EXPECT_EQ(XpuNetioGda::deregisterWindow(&host), flagcxSuccess);
  EXPECT_EQ(std::count(events.begin(), events.end(), Qp), 1);
}

TEST_F(GdaHostControl, PublicCommDestroyRetriesPartiallyReleasedWindow) {
  auto *d = heapComm();
  auto *o = static_cast<XpuNetioGda::CommOwner *>(d->gdaHostOwner);
  flagcxWindow host{};
  auto *w = makeWindow(host, o);
  void *original = w->deviceView;
  comm.devNetWindows = &host;
  failMr = true;
  EXPECT_EQ(XpuNetioGda::deregisterWindow(&host), flagcxRemoteError);
  EXPECT_EQ(host.gdaDeviceView, nullptr);
  EXPECT_EQ(flagcxDevCommDestroy(&comm, d), flagcxSuccess);
  EXPECT_EQ(host.gdaWindowOwner, nullptr);
  EXPECT_FALSE(host.gdaRegistrationActive);
  EXPECT_EQ(std::count(freed.begin(), freed.end(), original), 1);
  EXPECT_EQ(comm.devNetCommHandle, nullptr);
  comm.devNetWindows = nullptr;
}

TEST_F(GdaHostControl, FailedDrainRevokesWindowPublicationBeforeMrRelease) {
  flagcxWindow host{};
  auto *window = makeWindow(host);
  comm.devNetWindows = &host;
  failPeerAt = 1;
  EXPECT_EQ(XpuNetioGda::prepareCommDestroy(&dev), flagcxRemoteError);
  EXPECT_TRUE(window->retiring);
  EXPECT_EQ(host.gdaDeviceView, nullptr);
  EXPECT_NE(window->deviceView, nullptr);
  EXPECT_EQ(std::count(events.begin(), events.end(), Mr), 0);
  flagcxDevMemInternal view{};
  DeviceAPI::Window deviceWindow{};
  view.rawPtr = host.userBase;
  EXPECT_EQ(XpuNetioGda::bindDevMem(&host, &view, 4096, &deviceWindow),
            flagcxInvalidUsage);
  ASSERT_EQ(XpuNetioGda::prepareCommDestroy(&dev), flagcxSuccess);
  EXPECT_EQ(XpuNetioGda::deregisterWindow(&host), flagcxSuccess);
  comm.devNetWindows = nullptr;
}
} // namespace

flagcxResult_t flagcxConvergeBootstrapStatus(flagcxComm_t, flagcxResult_t local,
                                             flagcxResult_t *common) {
  events.push_back(Converge);
  ++convergeCalls;
  *common =
      local != flagcxSuccess
          ? local
          : (failPeerAt == convergeCalls ? flagcxRemoteError : peerStatus);
  return flagcxSuccess;
}
extern "C" xpu_netio_status_t
xpu_netio_unpublish_qps_retryable(xpu_netio_context_t *,
                                  xpu_netio_device_handle_t *) {
  events.push_back(Unpublish);
  if (failUnpublish) {
    failUnpublish = false;
    return XPU_NETIO_PROVIDER_ERROR;
  }
  return XPU_NETIO_SUCCESS;
}
extern "C" xpu_netio_status_t
xpu_netio_destroy_qp_retryable(xpu_netio_qp_t **p) {
  events.push_back(Qp);
  *p = nullptr;
  return XPU_NETIO_SUCCESS;
}
extern "C" xpu_netio_status_t
xpu_netio_deregister_memory_retryable(xpu_netio_mr_t **p) {
  events.push_back(Mr);
  if (failMr) {
    failMr = false;
    return XPU_NETIO_PROVIDER_ERROR;
  }
  *p = nullptr;
  return XPU_NETIO_SUCCESS;
}
extern "C" xpu_netio_status_t
xpu_netio_destroy_context_retryable(xpu_netio_context_t **p) {
  events.push_back(Context);
  *p = nullptr;
  return XPU_NETIO_SUCCESS;
}
extern "C" xpu_netio_status_t
xpu_netio_close_device_retryable(xpu_netio_device_t **p) {
  events.push_back(Device);
  if (failDevice) {
    failDevice = false;
    return XPU_NETIO_PROVIDER_ERROR;
  }
  *p = nullptr;
  return XPU_NETIO_SUCCESS;
}

extern "C" xpu_netio_status_t
xpu_netio_destroy_remote_memory_retryable(xpu_netio_remote_mr_t **p) {
  events.push_back(RemoteMr);
  *p = nullptr;
  return XPU_NETIO_SUCCESS;
}
extern "C" xpu_netio_status_t
xpu_netio_query_error(xpu_netio_device_handle_t *,
                      xpu_netio_error_info_t *info) {
  info->category = providerError;
  return XPU_NETIO_SUCCESS;
}
