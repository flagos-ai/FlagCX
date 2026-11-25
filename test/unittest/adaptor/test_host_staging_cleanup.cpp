#include "alloc.h"
#include "flagcx.h"
#include "utils.h"
#include <chrono>
#include <cstdlib>
#include <gtest/gtest.h>
#include <map>
#include <memory>
#include <new>
#include <set>
#include <vector>

namespace HostCleanupProbe {
static std::set<void *> backing, wrappers;
static int contextsFreed, wrappersFreed, nativeFreed;
static void *allocateBacking(size_t size) {
  void *p = std::malloc(size);
  if (p)
    backing.insert(p);
  return p;
}
static void release(void *p) {
  backing.erase(p);
  if (wrappers.erase(p))
    wrappersFreed++;
  std::free(p);
}
struct NativeBuffer {
  void *allocation;
  int sends = 0, receives = 0;
  uint32_t tag = 0;
  explicit NativeBuffer(void *p) : allocation(p) {}
  ~NativeBuffer() {
    EXPECT_EQ(backing.count(allocation), 1u)
        << "backing freed before native registration";
    nativeFreed++;
  }
  void send(int, uint32_t value, size_t, size_t) {
    sends++;
    tag = value;
  }
  void recv(int, uint32_t value, size_t, size_t) {
    receives++;
    tag = value;
  }
  void waitSend(std::chrono::milliseconds) {}
  void waitRecv(std::chrono::milliseconds) {}
};
struct Context {
  ~Context() { contextsFreed++; }
  std::unique_ptr<NativeBuffer> createUnboundBuffer(void *p, size_t) {
    return std::unique_ptr<NativeBuffer>(new NativeBuffer(p));
  }
};
struct Owner {
  std::shared_ptr<Context> base;
};
static Owner *makeOwner() {
  void *memory = std::calloc(1, sizeof(Owner));
  if (!memory)
    return nullptr;
  wrappers.insert(memory);
  auto *owner = new (memory) Owner;
  owner->base = std::make_shared<Context>();
  return owner;
}
} // namespace HostCleanupProbe

namespace MpiStagingContract {
using flagcxInnerComm_t = HostCleanupProbe::Owner *;
struct stagedBuffer {
  flagcxInnerComm_t owner;
  int offset, size;
  void *buffer;
};
using stagedBuffer_t = stagedBuffer *;
static std::vector<stagedBuffer_t> sendStagedBufferList, recvStagedBufferList;
static void *malloc(size_t size) {
  return HostCleanupProbe::allocateBacking(size);
}
static void free(void *p) { HostCleanupProbe::release(p); }
#define MPI_ADAPTOR_MAX_STAGED_BUFFER_SIZE (8 * 1024 * 1024)
#include "adaptor-contracts/mpi_staging.h"
} // namespace MpiStagingContract

namespace GlooStagingContract {
using flagcxInnerComm_t = HostCleanupProbe::Owner *;
struct stagedBuffer {
  flagcxInnerComm_t owner;
  int offset, size, cnt;
  void *buffer;
  HostCleanupProbe::NativeBuffer *unboundBuffer;
};
using stagedBuffer_t = stagedBuffer *;
using bufferPtr = std::unique_ptr<HostCleanupProbe::NativeBuffer>;
static std::vector<stagedBuffer_t> sendStagedBufferList, recvStagedBufferList;
static std::vector<bufferPtr> unboundBufferStorage;
static std::map<std::pair<flagcxInnerComm_t, int>, uint32_t> sendPeerTags,
    recvPeerTags;
static int groupDepth;
static constexpr std::chrono::milliseconds flagcxGlooDefaultTimeout(10);
static void *malloc(size_t size) {
  return HostCleanupProbe::allocateBacking(size);
}
static void free(void *p) { HostCleanupProbe::release(p); }
#define GLOO_ADAPTOR_MAX_STAGED_BUFFER_SIZE (8 * 1024 * 1024)
#include "adaptor-contracts/gloo_staging.h"
} // namespace GlooStagingContract

namespace {
class HostStagingCleanup : public ::testing::Test {
protected:
  using Owner = HostCleanupProbe::Owner;
  using Operation = flagcxResult_t (*)(Owner *);
  using Stage = flagcxResult_t (*)(Owner *, void **, size_t, int);
  void SetUp() override {
    ASSERT_TRUE(HostCleanupProbe::backing.empty());
    ASSERT_TRUE(HostCleanupProbe::wrappers.empty());
    HostCleanupProbe::contextsFreed = HostCleanupProbe::wrappersFreed =
        HostCleanupProbe::nativeFreed = 0;
  }
  void TearDown() override {
    EXPECT_TRUE(HostCleanupProbe::backing.empty());
    EXPECT_TRUE(HostCleanupProbe::wrappers.empty());
  }
  void emptyOwner(Operation finalize, Operation destroy) {
    auto *owner = HostCleanupProbe::makeOwner();
    ASSERT_NE(owner, nullptr);
    EXPECT_EQ(finalize(owner), flagcxSuccess);
    EXPECT_EQ(finalize(owner), flagcxSuccess);
    EXPECT_EQ(HostCleanupProbe::contextsFreed, 1);
    EXPECT_EQ(HostCleanupProbe::wrappersFreed, 0);
    EXPECT_EQ(destroy(owner), flagcxSuccess);
    EXPECT_EQ(HostCleanupProbe::contextsFreed, 1);
    EXPECT_EQ(HostCleanupProbe::wrappersFreed, 1);
  }
  void independentOwners(Stage stage, Operation finalize, Operation destroy) {
    auto *a = HostCleanupProbe::makeOwner(), *b = HostCleanupProbe::makeOwner();
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    void *aSend, *aRecv, *bSend, *bRecv, *again;
    ASSERT_EQ(stage(a, &aSend, 64, 0), flagcxSuccess);
    ASSERT_EQ(stage(b, &bSend, 64, 0), flagcxSuccess);
    ASSERT_EQ(stage(a, &aRecv, 64, 1), flagcxSuccess);
    ASSERT_EQ(stage(b, &bRecv, 64, 1), flagcxSuccess);
    EXPECT_NE(aSend, bSend);
    EXPECT_NE(aRecv, bRecv);
    ASSERT_EQ(stage(a, &again, 64, 0), flagcxSuccess);
    EXPECT_EQ(again, aSend);
    EXPECT_EQ(finalize(a), flagcxSuccess);
    EXPECT_EQ(HostCleanupProbe::backing.size(), 2u);
    EXPECT_EQ(HostCleanupProbe::backing.count(bSend), 1u);
    EXPECT_EQ(HostCleanupProbe::backing.count(bRecv), 1u);
    EXPECT_EQ(destroy(a), flagcxSuccess);
    EXPECT_EQ(stage(b, &again, 64, 0), flagcxSuccess);
    EXPECT_EQ(again, bSend);
    EXPECT_EQ(destroy(b), flagcxSuccess);
    EXPECT_EQ(HostCleanupProbe::contextsFreed, 2);
    EXPECT_EQ(HostCleanupProbe::wrappersFreed, 2);
  }
};
} // namespace
TEST_F(HostStagingCleanup, MpiEmptyFinalizeThenDestroyReleasesExactlyOnce) {
  emptyOwner(MpiStagingContract::mpiAdaptorCommFinalize,
             MpiStagingContract::mpiAdaptorCommDestroy);
}
TEST_F(HostStagingCleanup, GlooEmptyFinalizeThenDestroyReleasesExactlyOnce) {
  emptyOwner(GlooStagingContract::glooAdaptorCommFinalize,
             GlooStagingContract::glooAdaptorCommDestroy);
}
TEST_F(HostStagingCleanup, MpiStagingSurvivesAnotherCommunicatorsDestroy) {
  independentOwners(MpiStagingContract::mpiAdaptorGetStagedBuffer,
                    MpiStagingContract::mpiAdaptorCommFinalize,
                    MpiStagingContract::mpiAdaptorCommDestroy);
}
TEST_F(HostStagingCleanup, GlooStagingAndRegistrationsFollowTheirOwner) {
  independentOwners(GlooStagingContract::glooAdaptorGetStagedBuffer,
                    GlooStagingContract::glooAdaptorCommFinalize,
                    GlooStagingContract::glooAdaptorCommDestroy);
  EXPECT_EQ(HostCleanupProbe::nativeFreed, 4);
}
TEST_F(HostStagingCleanup, GlooPendingOperationRetainsContextAndBuffers) {
  using namespace GlooStagingContract;
  auto *a = HostCleanupProbe::makeOwner(), *b = HostCleanupProbe::makeOwner();
  void *aSend, *bSend;
  ASSERT_EQ(glooAdaptorGetStagedBuffer(a, &aSend, 64, 0), flagcxSuccess);
  ASSERT_EQ(glooAdaptorGetStagedBuffer(b, &bSend, 64, 0), flagcxSuccess);
  groupDepth = 1;
  ASSERT_EQ(glooAdaptorSend(aSend, 1, flagcxInt, 1, a, nullptr), flagcxSuccess);
  EXPECT_EQ(sendStagedBufferList[0]->unboundBuffer->sends, 1);
  EXPECT_EQ(sendStagedBufferList[1]->unboundBuffer->sends, 0);
  EXPECT_EQ(glooAdaptorCommDestroy(a), flagcxInProgress);
  EXPECT_EQ(a->base != nullptr, true);
  EXPECT_EQ(HostCleanupProbe::nativeFreed, 0);
  sendStagedBufferList[0]->cnt = 0; // Simulate retired fake SDK requests.
  groupDepth = 0;
  EXPECT_EQ(glooAdaptorCommDestroy(a), flagcxSuccess);
  EXPECT_EQ(HostCleanupProbe::backing.count(bSend), 1u);
  EXPECT_EQ(glooAdaptorCommDestroy(b), flagcxSuccess);
}
TEST_F(HostStagingCleanup, GlooPeerTagsAndSendRecvSelectionArePerOwner) {
  using namespace GlooStagingContract;
  auto *a = HostCleanupProbe::makeOwner(), *b = HostCleanupProbe::makeOwner();
  void *aSend, *aRecv, *bSend, *bRecv;
  ASSERT_EQ(glooAdaptorGetStagedBuffer(a, &aSend, 64, 0), flagcxSuccess);
  ASSERT_EQ(glooAdaptorGetStagedBuffer(b, &bSend, 64, 0), flagcxSuccess);
  ASSERT_EQ(glooAdaptorGetStagedBuffer(a, &aRecv, 64, 1), flagcxSuccess);
  ASSERT_EQ(glooAdaptorGetStagedBuffer(b, &bRecv, 64, 1), flagcxSuccess);
  EXPECT_EQ(glooAdaptorSend(aSend, 1, flagcxInt, 1, a, nullptr), flagcxSuccess);
  EXPECT_EQ(glooAdaptorSend(bSend, 1, flagcxInt, 1, b, nullptr), flagcxSuccess);
  EXPECT_EQ(sendStagedBufferList[0]->unboundBuffer->tag, 0u);
  EXPECT_EQ(sendStagedBufferList[1]->unboundBuffer->tag, 0u);
  EXPECT_EQ(glooAdaptorRecv(aRecv, 1, flagcxInt, 1, a, nullptr), flagcxSuccess);
  EXPECT_EQ(glooAdaptorRecv(bRecv, 1, flagcxInt, 1, b, nullptr), flagcxSuccess);
  EXPECT_EQ(recvStagedBufferList[0]->unboundBuffer->receives, 1);
  EXPECT_EQ(recvStagedBufferList[1]->unboundBuffer->receives, 1);
  EXPECT_EQ(glooAdaptorSend(bSend, 1, flagcxInt, 1, a, nullptr),
            flagcxInvalidArgument);
  EXPECT_EQ(glooAdaptorCommDestroy(a), flagcxSuccess);
  EXPECT_EQ(sendPeerTags.count({b, 1}), 1u);
  EXPECT_EQ(recvPeerTags.count({b, 1}), 1u);
  EXPECT_EQ(glooAdaptorCommDestroy(b), flagcxSuccess);
}

namespace BootstrapStagingContract {
struct flagcxInnerComm {
  void *base;
};
using flagcxInnerComm_t = flagcxInnerComm *;
struct stagedBuffer {
  flagcxInnerComm_t owner;
  int offset, size;
  void *buffer;
};
using stagedBuffer_t = stagedBuffer *;
static std::vector<stagedBuffer_t> sendStagedBufferList, recvStagedBufferList;
#define BOOTSTRAP_ADAPTOR_MAX_STAGED_BUFFER_SIZE 1024
#include "adaptor-contracts/bootstrap_staging.h"
} // namespace BootstrapStagingContract

TEST(BootstrapHostStaging, OwnerIsolationKeepsBorrowedBootstrapAlive) {
  using namespace BootstrapStagingContract;
  auto *a = (flagcxInnerComm_t)calloc(1, sizeof(flagcxInnerComm));
  auto *b = (flagcxInnerComm_t)calloc(1, sizeof(flagcxInnerComm));
  int bootstrap = 42;
  a->base = b->base = &bootstrap;
  void *pa, *pb, *again;
  ASSERT_EQ(bootstrapAdaptorGetStagedBuffer(a, &pa, 64, 0), flagcxSuccess);
  ASSERT_EQ(bootstrapAdaptorGetStagedBuffer(b, &pb, 64, 0), flagcxSuccess);
  EXPECT_NE(pa, pb);
  EXPECT_EQ(bootstrapAdaptorCommFinalize(a), flagcxSuccess);
  EXPECT_EQ(a->base, &bootstrap);
  EXPECT_EQ(bootstrapAdaptorCommDestroy(a), flagcxSuccess);
  ASSERT_EQ(sendStagedBufferList.size(), 1u);
  EXPECT_EQ(bootstrapAdaptorGetStagedBuffer(b, &again, 64, 0), flagcxSuccess);
  EXPECT_EQ(again, pb);
  EXPECT_EQ(bootstrapAdaptorCommDestroy(b), flagcxSuccess);
  EXPECT_TRUE(sendStagedBufferList.empty());
  EXPECT_EQ(bootstrap, 42);
}
