"""Exercise error propagation, parsing and device-copy contracts behind CI warnings."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CXX = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")


def function(source, name):
    start = source.index(name + "(")
    start = source.rfind("\n", 0, start) + 1
    previous = source.rfind("\n", 0, start - 1) + 1
    if source[previous:start].strip() in ("flagcxResult_t", "static flagcxResult_t"):
        start = previous
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@unittest.skipUnless(CXX, "C++ compiler required")
class SourceWarningTest(unittest.TestCase):
    def compile_run(self, source, headers=None, flags=()):
        with tempfile.TemporaryDirectory() as directory:
            tmp = Path(directory)
            for name, content in (headers or {}).items():
                header = tmp / name
                header.parent.mkdir(parents=True, exist_ok=True)
                header.write_text(content)
            cpp = tmp / "test.cpp"
            cpp.write_text(source)
            result = subprocess.run(
                [CXX, "-std=c++11", "-Wall", "-Wextra", "-Werror",
                 "-pthread", "-I" + str(tmp),
                 "-I" + str(ROOT / "flagcx/adaptor/device"),
                 str(cpp), "-o", str(tmp / "test"), *flags],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            subprocess.run([str(tmp / "test")], check=True)

    def test_devnet_enable_and_build_binding_agreement(self):
        utils = (ROOT / "flagcx/service/utils.cc").read_text()
        common = (ROOT / "flagcx/adaptor/device_api/devnet/dev_net.cc").read_text()
        config = common[common.index("struct flagcxDevNetConfig {"):
                        common.index(function(common, "flagcxDevNetCreateComm"))]
        self.compile_run(r"""
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>
enum flagcxResult_t {
  flagcxSuccess, flagcxInvalidArgument, flagcxSystemError,
  flagcxInvalidUsage, flagcxNotSupported
};
struct Comm { int nranks, rank; void *bootstrap; };
typedef Comm *flagcxComm_t;
struct Adaptor { const char *name; bool isGda; };
static Adaptor devNetAdaptor = {"Mock GDA", false};
static int mismatch, gathers;
const char *flagcxGetEnv(const char *name) { return std::getenv(name); }
flagcxResult_t flagcxConvergeBootstrapStatus(Comm *, flagcxResult_t local,
                                            flagcxResult_t *common) {
  *common = local; return flagcxSuccess;
}
flagcxResult_t bootstrapCollAllGather(void *, void *, size_t);
#define FLAGCXCHECK(call) do { flagcxResult_t r = (call); \
  if (r != flagcxSuccess) return r; } while (0)
""" + function(utils, "flagcxReadDevNetInteger") + "\n" +
            function(utils, "flagcxReadDevNetConfig") + "\n" + config + r"""
flagcxResult_t bootstrapCollAllGather(void *, void *data, size_t bytes) {
  assert(bytes == sizeof(flagcxDevNetConfig)); ++gathers;
  flagcxDevNetConfig *all = static_cast<flagcxDevNetConfig *>(data);
  all[1] = all[0];
  switch (mismatch) {
    case 1: all[1].enabled ^= 1; break;
    case 2: all[1].gda ^= 1; break;
    case 3: all[1].wantsNet ^= 1; break;
    case 4: all[1].forceOneSidedNet ^= 1; break;
    case 5: all[1].valid = 0; break;
    case 6: all[1].binding[0] ^= 1; break;
  }
  return flagcxSuccess;
}
int main() {
  Comm comm = {1, 0, nullptr};
  bool enabled = false;
  for (int binding = 0; binding != 2; ++binding) {
    devNetAdaptor.isGda = binding != 0;
    unsetenv("FLAGCX_DEVNET_ENABLE");
    assert(flagcxDevNetConfigure(&comm, true, true, &enabled) == flagcxSuccess);
    assert(enabled == (binding != 0) && gathers == 0);
    setenv("FLAGCX_DEVNET_ENABLE", "0", 1);
    assert(flagcxDevNetConfigure(&comm, false, false, &enabled) == flagcxSuccess);
    assert(!enabled);
    assert(flagcxDevNetConfigure(&comm, true, false, &enabled) == flagcxSuccess);
    assert(!enabled);
    for (const char *text : {"-1", "2", "garbage", "1x"}) {
      setenv("FLAGCX_DEVNET_ENABLE", text, 1);
      assert(flagcxDevNetConfigure(&comm, true, false, &enabled) == flagcxInvalidArgument);
    }
  }
  setenv("FLAGCX_DEVNET_ENABLE", "1", 1);
  comm.nranks = 2;
  for (mismatch = 0; mismatch != 7; ++mismatch) {
    flagcxResult_t expected = mismatch == 0 || mismatch == 2 || mismatch == 6
                                 ? flagcxSuccess : mismatch == 5
                                 ? flagcxInvalidArgument : flagcxInvalidUsage;
    assert(flagcxDevNetConfigure(&comm, true, true, &enabled) == expected);
    if (expected == flagcxSuccess) assert(enabled == (mismatch == 0));
  }
  assert(gathers == 7);
}
""")

    def test_gda_creation_fallback_requires_collective_cleanup(self):
        source = (ROOT / "flagcx/adaptor/device_api/devnet/dev_net.cc").read_text()
        self.compile_run(r"""
#include <cassert>
#include <initializer_list>
#include <vector>
enum flagcxResult_t {
  flagcxSuccess, flagcxInvalidArgument, flagcxSystemError, flagcxInvalidUsage,
  flagcxNotSupported, flagcxInternalError, flagcxRemoteError,
  flagcxUnhandledDeviceError
};
struct Comm {};
typedef Comm *flagcxComm_t;
struct DevComm {
  int netUseGda;
  void *gdaHostOwner, *gdaDeviceContext;
};
typedef DevComm *flagcxDevComm_t;
struct flagcxDevCommRequirements {};
static flagcxResult_t createResult, destroyResult;
static bool retainOwner, missingContext;
static int destroys;
static unsigned convergenceRound;
static std::vector<flagcxResult_t> peer;
flagcxResult_t create(Comm *, DevComm *dev,
                      const flagcxDevCommRequirements *) {
  if (createResult == flagcxSuccess || retainOwner) {
    dev->gdaHostOwner = dev;
    if (!missingContext) dev->gdaDeviceContext = dev;
  }
  return createResult;
}
flagcxResult_t destroy(DevComm *dev) {
  ++destroys;
  if (destroyResult == flagcxSuccess)
    dev->gdaHostOwner = dev->gdaDeviceContext = nullptr;
  return destroyResult;
}
struct Adaptor {
  const char *name;
  flagcxResult_t (*createComm)(Comm *, DevComm *, const flagcxDevCommRequirements *);
  flagcxResult_t (*destroyComm)(DevComm *);
};
static Adaptor devNetAdaptor = {"Mock GDA", create, destroy};
flagcxResult_t flagcxConvergeBootstrapStatus(Comm *, flagcxResult_t local,
                                            flagcxResult_t *common) {
  assert(convergenceRound < peer.size());
  *common = local != flagcxSuccess ? local : peer[convergenceRound];
  ++convergenceRound;
  return flagcxSuccess;
}
#define FLAGCXCHECK(call) do { flagcxResult_t r = (call); \
  if (r != flagcxSuccess) return r; } while (0)
#define INFO(...) ((void)0)
""" + function(source, "flagcxDevNetCreateComm") + r"""
int main() {
  Comm comm;
  flagcxDevCommRequirements req;
  auto run = [&](flagcxResult_t result, std::initializer_list<flagcxResult_t> peers,
                 flagcxResult_t expected, bool useGda) {
    createResult = result; peer = peers; convergenceRound = 0; destroys = 0;
    DevComm dev = {};
    assert(flagcxDevNetCreateComm(&comm, &dev, &req) == expected);
    assert((dev.netUseGda != 0) == useGda);
    assert(convergenceRound == peer.size());
    return dev;
  };
  destroyResult = flagcxSuccess;
  run(flagcxSuccess, {flagcxSuccess, flagcxSuccess}, flagcxSuccess, true);
  // Clean capability, device, and resource failures select Proxy.
  for (auto failure : {flagcxNotSupported, flagcxSystemError, flagcxRemoteError,
                       flagcxUnhandledDeviceError})
    run(failure, {failure, flagcxSuccess}, flagcxSuccess, false);
  // Local resources or a peer's retained owner must prevent fallback.
  retainOwner = true;
  DevComm dev = run(flagcxNotSupported, {flagcxNotSupported, flagcxSuccess},
                    flagcxNotSupported, false);
  assert(dev.gdaHostOwner != nullptr);
  retainOwner = false;
  run(flagcxNotSupported, {flagcxNotSupported, flagcxRemoteError},
      flagcxRemoteError, false);
  run(flagcxInvalidArgument, {flagcxSuccess, flagcxSuccess},
      flagcxInvalidArgument, false);
  // A broken publication contract is an error, never a successful fallback.
  missingContext = true;
  dev = run(flagcxSuccess, {flagcxSuccess, flagcxSuccess, flagcxSuccess},
            flagcxInternalError, false);
  assert(destroys == 1 && dev.gdaHostOwner == nullptr);
  missingContext = false;
  destroyResult = flagcxRemoteError;
  dev = run(flagcxSuccess, {flagcxSuccess, flagcxInternalError, flagcxSuccess},
            flagcxRemoteError, false);
  assert(destroys == 1 && dev.gdaHostOwner != nullptr);
}
""")

    def test_gda_signal_storage_without_barrier_channels(self):
        source = (ROOT / "flagcx/adaptor/device_api/devnet/xpu_netio_gda_host.cc").read_text()
        start = source.index("    const bool barrierStorageValid =")
        end = source.index("    if ((owner->caps.operation_bits", start)
        self.compile_run(r"""
#include <cassert>
#include <climits>
#include <cstdint>
struct DevComm {
  int interBarrierCount, nTeamRanks, barrierSignalBase, signalCount;
};
bool valid(const DevComm *devComm) {
""" + source[start:end] + r"""
  return barrierStorageValid;
}
int main() {
  DevComm dev = {0, 0, 2, 2};
  assert(valid(&dev)); // Two indexed user signals, no Barrier storage.
  dev.interBarrierCount = 1;
  assert(!valid(&dev));
  dev.nTeamRanks = 4; dev.signalCount = 6;
  assert(valid(&dev));
  dev.signalCount = 5; assert(!valid(&dev));
  dev.interBarrierCount = -1; assert(!valid(&dev));
  dev = {INT_MAX, INT_MAX, 0, INT_MAX}; assert(!valid(&dev));
  dev = {0, 1, -1, 1}; assert(!valid(&dev));
}
""")

    def test_gda_finalize_retries_retire_qps_before_local_cleanup(self):
        lifecycle = (ROOT / "flagcx/adaptor/flagcx_device.cc").read_text()
        source = (ROOT / "flagcx/flagcx.cc").read_text()
        finalize = function(source, "flagcxCommFinalize")
        loop_start = finalize.index("  while (state->commonPhase <")
        stages = finalize[loop_start:finalize.index(
            "  __atomic_store_n(&state->lastResult", loop_start)]
        destroy = function(lifecycle[lifecycle.index(
            'extern "C" flagcxResult_t flagcxDevCommDestroy('):],
            "flagcxDevCommDestroy")
        snapshot = destroy[destroy.index("  devComm->cleanupStarted = true;"):
                           destroy.index("  // Stop the proxy")]
        header = (ROOT / "flagcx/core/include/comm_lifecycle.h").read_text()
        phases = header[header.index("enum flagcxFinalizePhase {"):
                        header.index("enum flagcxNativeFinalizeScope {")]
        program = r"""
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
enum flagcxResult_t { flagcxSuccess, flagcxRemoteError };
struct DevComm {
  bool cleanupStarted;
  int cleanupWork;
  void *gdaHostOwner;
  int gdaDraining, gdaQuiesced;
  DevComm *previousActive;
};
typedef DevComm *flagcxDevComm_t;
""" + phases + r"""
struct FinalizeState {
  flagcxFinalizePhase commonPhase;
  unsigned localDoneMask, vendorCount;
  void **vendorOwners;
};
struct Hetero {
  struct { int importsClosedConfirmed, phase; } finalizeState;
};
struct Comm {
  int rank, deviceOrdinal;
  bool deviceOrdinalValid;
  FinalizeState finalizeState;
  Hetero *heteroComm;
  void *activeDevMems;
  DevComm *devNetCommHandle, *pendingDevCommCleanup, *activeDevComms;
};
typedef Comm *flagcxComm_t;
#define FLAGCXCHECK(call) do { flagcxResult_t r = (call); \
  if (r != flagcxSuccess) return r; } while (0)
struct Rendezvous {
  std::mutex mutex;
  std::condition_variable changed;
  int arrived = 0;
  flagcxResult_t pending = flagcxSuccess;
  std::vector<flagcxResult_t> results;
  flagcxResult_t converge(flagcxResult_t local, flagcxResult_t *common) {
    std::unique_lock<std::mutex> lock(mutex);
    size_t round = results.size();
    if (local != flagcxSuccess) pending = local;
    if (++arrived == 2) {
      results.push_back(pending);
      pending = flagcxSuccess; arrived = 0;
      changed.notify_all();
    } else {
      bool ready = changed.wait_for(lock, std::chrono::seconds(5), [&] {
        return results.size() > round;
      });
      assert(ready && "ranks entered different destruction collectives");
    }
    *common = results[round];
    return flagcxSuccess;
  }
};
static Rendezvous *rendezvous;
static bool localOnly;
static int scenario, prepareCalls[2], retired[2], released[2];
static bool failed[2];
static thread_local Comm *current;
flagcxResult_t flagcxConvergeBootstrapStatus(Comm *, flagcxResult_t local,
                                            flagcxResult_t *common) {
  assert(!localOnly && "Finalize-complete cleanup must remain rank-local");
  return rendezvous->converge(local, common);
}
flagcxResult_t prepare(DevComm *dev) {
  int rank = current->rank;
  ++prepareCalls[rank];
  dev->gdaDraining = 1;
  flagcxResult_t local = flagcxSuccess, common;
  if (scenario == 1 && rank == 0 && !failed[rank]) {
    failed[rank] = true; local = flagcxRemoteError;
  }
  FLAGCXCHECK(flagcxConvergeBootstrapStatus(current, local, &common));
  if (common != flagcxSuccess) return common;
  local = flagcxSuccess;
  if (scenario == 2 && rank == 0 && !failed[rank]) {
    failed[rank] = true; local = flagcxRemoteError;
  } else if (!retired[rank]) {
    ++retired[rank];
  }
  FLAGCXCHECK(flagcxConvergeBootstrapStatus(current, local, &common));
  if (common == flagcxSuccess) dev->gdaQuiesced = 1;
  return common;
}
struct Adaptor { flagcxResult_t (*prepareCommDestroy)(DevComm *); };
static Adaptor devNetAdaptor = {prepare};
""" + function(lifecycle, "flagcxCommPrepareGdaDestroy") + r"""
flagcxResult_t flagcxCommPrepareFinalizeStage(Comm *comm, int stage) {
  if (stage != 2) return flagcxSuccess;
  int rank = comm->rank;
  // A rank-local DevMem sync/free fails before DevComm destruction.
  if (scenario == 0 && rank == 0 && !failed[rank]) {
    failed[rank] = true; return flagcxRemoteError;
  }
  // Simulate existing local destruction, including its fallback GDA drain.
  // The real scheduling must ensure this cannot introduce a collective here.
  for (DevComm *dev = comm->activeDevComms; dev != nullptr;
       dev = comm->activeDevComms) {
    if (dev->gdaHostOwner != nullptr) {
      assert(comm->pendingDevCommCleanup == nullptr || scenario == 4);
      if (!dev->gdaQuiesced) FLAGCXCHECK(prepare(dev));
      assert(retired[rank] == 1);
      ++released[rank];
      dev->gdaHostOwner = nullptr;
    }
    comm->activeDevComms = dev->previousActive;
    if (comm->pendingDevCommCleanup == dev)
      comm->pendingDevCommCleanup = nullptr;
    if (comm->devNetCommHandle == dev) comm->devNetCommHandle = nullptr;
  }
  // One rank retains a later local failure after both GDA owners are gone.
  if (scenario == 3 && rank == 1 && !failed[rank]) {
    failed[rank] = true; return flagcxRemoteError;
  }
  return flagcxSuccess;
}
typedef void *flagcxInnerComm_t;
struct RegPool {
  flagcxResult_t releaseOwner(void *, void *) { return flagcxSuccess; }
} globalRegPool;
struct DeviceAdaptor { flagcxResult_t (*setDevice)(int); };
static DeviceAdaptor *deviceAdaptor;
static const int flagcxHeteroFinalizeComplete = 1;
#define INFO(...) ((void)0)
#define FLAGCX_INIT 0
flagcxResult_t flagcxCommFinalizeStatus(Comm *comm, int, flagcxResult_t local) {
  flagcxResult_t common;
  FLAGCXCHECK(flagcxConvergeBootstrapStatus(comm, local, &common));
  return common;
}
flagcxResult_t flagcxCommCatalogNativeOwners(Comm *) { return flagcxSuccess; }
flagcxResult_t flagcxHeteroCommCloseRelayImports(Hetero *) { return flagcxSuccess; }
flagcxResult_t flagcxHeteroCommPrepareDestroy(Hetero *) { return flagcxSuccess; }
flagcxResult_t flagcxCommCleanupIpcTable(Comm *) { return flagcxSuccess; }
flagcxResult_t flagcxCommFinalizeNativeOwners(Comm *) { return flagcxSuccess; }
flagcxResult_t flagcxDevMemDestroy(Comm *, void *) { return flagcxSuccess; }
flagcxResult_t finalize(Comm *comm) {
  auto *state = &comm->finalizeState;
  flagcxResult_t result = flagcxSuccess;
""" + stages + r"""
  return result;
}
static bool failSnapshot;
flagcxResult_t flagcxCommDrainSnapshot(Comm *comm, int *) {
  return failSnapshot && comm->rank == 0 ? flagcxRemoteError : flagcxSuccess;
}
flagcxResult_t drainBeforeDestroy(Comm *comm, DevComm *devComm) {
""" + snapshot + r"""
  return flagcxSuccess;
}
int main() {
  const int firstScenario = 0;
  for (scenario = firstScenario; scenario != 6; ++scenario) {
    Rendezvous exchange; rendezvous = &exchange;
    DevComm gda[2] = {}, proxy[2] = {};
    Comm comm[2] = {};
    for (int rank = 0; rank != 2; ++rank) {
      prepareCalls[rank] = retired[rank] = released[rank] = 0;
      failed[rank] = false;
      gda[rank].gdaHostOwner = &gda[rank];
      proxy[rank].previousActive = &gda[rank];
      comm[rank].rank = rank;
      comm[rank].activeDevComms = &proxy[rank];
      comm[rank].devNetCommHandle = &gda[rank];
      if (scenario == 4) {
        // Retry a standalone GDA drain that already revoked publication.
        comm[rank].devNetCommHandle = nullptr;
        comm[rank].pendingDevCommCleanup = &gda[rank];
        gda[rank].gdaDraining = 1;
      } else if (scenario == 5) {
        // A build with GDA support can still select only Proxy.
        comm[rank].devNetCommHandle = nullptr;
        proxy[rank].previousActive = nullptr;
      }
    }
    auto run = [&](flagcxResult_t expected) {
      std::thread ranks[2];
      for (int rank = 0; rank != 2; ++rank)
        ranks[rank] = std::thread([&, rank] {
          current = &comm[rank];
          assert(finalize(current) == expected);
        });
      for (auto &rank : ranks) rank.join();
    };
    if (scenario < 4) {
      run(flagcxRemoteError);
      for (int rank = 0; rank != 2; ++rank) {
        assert(prepareCalls[rank] == 1);
        assert(released[rank] == (scenario == 0 ? rank : scenario == 3));
      }
    }
    run(flagcxSuccess);
    for (int rank = 0; rank != 2; ++rank) {
      assert(comm[rank].finalizeState.commonPhase == flagcxFinalizeComplete);
      assert(comm[rank].devNetCommHandle == nullptr);
      assert(comm[rank].pendingDevCommCleanup == nullptr);
      assert(prepareCalls[rank] == (scenario == 5 ? 0 :
                                  scenario == 1 || scenario == 2 ? 2 : 1));
      assert(retired[rank] == (scenario == 5 ? 0 : 1));
      assert(released[rank] == (scenario == 5 ? 0 : 1));
    }
  }
  // A snapshot failure must be agreed before either rank can retire GDA.
  {
    Rendezvous exchange; rendezvous = &exchange;
    Comm comm[2]{};
    DevComm dev[2]{};
    failSnapshot = true;
    std::thread ranks[2];
    for (int rank = 0; rank != 2; ++rank) {
      comm[rank].rank = rank;
      comm[rank].devNetCommHandle = &dev[rank];
      dev[rank].gdaHostOwner = &dev[rank];
      ranks[rank] = std::thread([&, rank] {
        assert(drainBeforeDestroy(&comm[rank], &dev[rank]) == flagcxRemoteError);
      });
    }
    for (auto &rank : ranks) rank.join();
    failSnapshot = false;
    for (auto &view : dev) assert(view.cleanupStarted && !view.gdaQuiesced);
  }
  // Once QPs have retired collectively, destroy can proceed serially per rank.
  // Exercise the real snapshot admission block without a peer rendezvous.
  localOnly = true;
  for (int rank = 0; rank != 2; ++rank) {
    Comm comm{};
    DevComm dev{};
    comm.devNetCommHandle = &dev;
    dev.gdaHostOwner = &dev;
    dev.gdaDraining = dev.gdaQuiesced = 1;
    assert(drainBeforeDestroy(&comm, &dev) == flagcxSuccess);
    assert(dev.cleanupStarted);
  }
}
"""
        self.compile_run(program)

    def test_unified_net_forwarding_preserves_context_coop_and_flush_guards(self):
        source = (ROOT / "bindings/ir/flagcx_device_unified_ir_impl.h").read_text()
        backend = (ROOT / "flagcx/adaptor/device_api/default_dev_api_backend.cc").read_text()
        traits = (ROOT / "flagcx/adaptor/include/device_api/default_comm_traits.h").read_text()
        proxy = backend.index("  if (!devComm->netUseGda && comm->heteroComm != nullptr &&")

        readiness = backend[backend.index("  // GDA readiness comes from its published device context") :
                            backend.index("  if (!devComm->netUseGda && oneSidedRequested && forceOneSidedNet &&", proxy)]
        copy_ready = next(line for line in traits.splitlines()
                          if "dc.netOneSidedReady = di.netOneSidedReady;" in line)
        utils = (ROOT / "flagcx/adaptor/include/device_utils.h").read_text()
        macros = utils[utils.index("#if FLAGCX_DEVICE_REQUIRES_LOCAL_NET_DESCRIPTOR"):
                       utils.index("// IR extern")]
        names = ("flagcxDevWaitSignal", "flagcxDevWaitCounter",
                 "flagcxDevReadSignal", "flagcxDevReadCounter", "flagcxDevFlush")
        bodies = "\n".join(
            ("void " if name in (names[0], names[1], names[4]) else "") +
            function(source, name) for name in names)
        harness = r"""
#include <cstdint>
#include <cassert>
#define FLAGCX_IR_EXTERN_C
#define FLAGCX_DEVICE_INLINE_DECORATOR
#define FLAGCX_IR_GLOBAL_PTR
using flagcxDevSignal_t = int;
using flagcxDevCounter_t = int;
using flagcxDevContext_t = int;
using flagcxDevCoopKind_t = int;
using flagcxDevMemoryOrder_t = int;
static int syncs;
struct flagcxCoopAny {
  int kind;
  void sync() const { ++syncs; }
};
flagcxCoopAny flagcxMakeCoopFromKind(int kind) { return {kind}; }
static int seenContext, seenCoop, seenSlot, seenBits, seenOrder, calls, fences;
static uint64_t seenLeast;
enum flagcxResult_t { flagcxSuccess, flagcxInternalError };
struct Registration { void *fullSendComms; };
struct Hetero { int oneSideHandleCount; Registration *oneSideHandles[1]; };
struct HostComm { Hetero *heteroComm; };
struct HostDevComm {
  int contextCount, netOneSidedReady, netUseGda;
  void *gdaDeviceContext, *fifoBuffers[2];
};
flagcxResult_t publishReadiness(HostComm *comm, HostDevComm *devComm) {
""" + readiness + r"""
  return flagcxSuccess;
}
struct Base {
  int netOneSidedReady;
""" + function(traits, "isOneSidedTransportReady") + r"""
  static void populateFromInternal(Base &dc, const HostDevComm &di) {
""" + copy_ready + r"""
  }
};
struct flagcxDevComm { Base _commBase; bool hasNet; };
bool flagcxDevCommHasNetContexts(const flagcxDevComm &comm) { return comm.hasNet; }
namespace DeviceAPI { namespace Intrin {
void threadfenceSystem() { ++fences; }
void threadfenceDevice() { ++fences; }
} }
struct flagcxDevNet {
  int context;
  flagcxDevNet(const flagcxDevComm &, int id) : context(id) {}
  void record(int coop, int slot, uint64_t least, int bits, int order) const {
    seenContext = context; seenCoop = coop; seenSlot = slot;
    seenLeast = least; seenBits = bits; seenOrder = order; ++calls;
  }
  void waitSignal(flagcxCoopAny coop, int slot, uint64_t least, int bits, int order) const {
    record(coop.kind, slot, least, bits, order);
  }
  void waitCounter(flagcxCoopAny coop, int slot, uint64_t least, int bits, int order) const {
    record(coop.kind, slot, least, bits, order);
  }
  uint64_t readSignal(int slot, int bits, int order) const {
    record(-1, slot, 0, bits, order); return UINT64_C(0xfedcba9876543210);
  }
  uint64_t readCounter(int slot, int bits, int order) const {
    record(-1, slot, 0, bits, order); return UINT64_C(0xabcdef0123456789);
  }
  void flush(flagcxCoopAny coop, int order) const {
    coop.sync(); record(coop.kind, -1, 0, 0, order); coop.sync();
  }
};
const void *flagcxDevNetGetFromCommS(const void *comm, int context) {
  static flagcxDevNet net(*(const flagcxDevComm *)comm, 0);
  net.context = context; return &net;
}
// Reintroducing a dependency on any legacy entry must fail this compilation.
__attribute__((deprecated)) void flagcxDevNetWaitSignalS(const void *, int, int, uint64_t, int, int);
__attribute__((deprecated)) void flagcxDevNetWaitCounterS(const void *, int, int, uint64_t, int, int);
__attribute__((deprecated)) uint64_t flagcxDevNetReadSignalS(const void *, int, int, int);
__attribute__((deprecated)) uint64_t flagcxDevNetReadCounterS(const void *, int, int, int);
__attribute__((deprecated)) void flagcxDevNetFlushS(const void *, int, int);
"""
        main = r"""
int main() {
  void *handle = reinterpret_cast<void *>(0x1000);
  Registration registration = {handle};
  Hetero hetero = {1, {&registration}};
  HostComm hostComm = {&hetero};
  HostDevComm host = {}; host.contextCount = 1;
#if TEST_GDA_BINDING
  host.netUseGda = 1; host.gdaDeviceContext = handle;
  hostComm.heteroComm = nullptr; // GDA has no Proxy mesh or FIFO.
#else
  host.fifoBuffers[0] = handle;
#endif
  assert(publishReadiness(&hostComm, &host) == flagcxSuccess);
  flagcxDevComm comm = {}; comm.hasNet = true;
  Base::populateFromInternal(comm._commBase, host);
  assert(comm._commBase.isOneSidedTransportReady());
  const uint64_t least = UINT64_C(0x123456789abcdef0);
  flagcxDevWaitSignal(&comm, 7, least, 64, 3, 2, 4);
  assert(seenContext == 3 && seenSlot == 7 && seenLeast == least);
  assert(seenCoop == 2 && seenBits == 64 && seenOrder == 4);
  flagcxDevWaitCounter(&comm, 8, least + 1, 32, 4, 3, 5);
  assert(seenContext == 4 && seenSlot == 8 && seenLeast == least + 1);
  assert(seenCoop == 3 && seenBits == 32 && seenOrder == 5);
  assert(flagcxDevReadSignal(&comm, 9, 64, 5, 6) == UINT64_C(0xfedcba9876543210));
  assert(seenContext == 5 && seenSlot == 9 && seenBits == 64 && seenOrder == 6);
  assert(flagcxDevReadCounter(&comm, 10, 32, 6, 7) == UINT64_C(0xabcdef0123456789));
  assert(seenContext == 6 && seenSlot == 10 && seenBits == 32 && seenOrder == 7);
  int before = calls;
  flagcxDevFlush(&comm, 7, 4, 8);
  assert(calls == before + FLAGCX_DEVICE_HAS_NET_TRANSPORT);
  assert(syncs == 2 * FLAGCX_DEVICE_HAS_NET_TRANSPORT);
  if (FLAGCX_DEVICE_HAS_NET_TRANSPORT) {
    assert(seenContext == 7 && seenCoop == 4 && seenOrder == 8);
  }
  before = calls;
  comm._commBase.netOneSidedReady = 0;
  flagcxDevFlush(&comm, 7, 4, 8);
  assert(calls == before);
  comm._commBase.netOneSidedReady = 1; comm.hasNet = false;
  flagcxDevFlush(&comm, 7, 4, 8);
  assert(calls == before && fences == 3);
  // Proxy still requires its mesh and every context's FIFO.
  for (int missing = 0; missing != 3; ++missing) {
    host = {}; host.contextCount = 1;
    host.fifoBuffers[0] = missing == 1 ? nullptr : handle;
    hostComm.heteroComm = missing == 2 ? nullptr : &hetero;
    assert(publishReadiness(&hostComm, &host) == flagcxSuccess);
    Base::populateFromInternal(comm._commBase, host);
    assert(comm._commBase.isOneSidedTransportReady() == (missing == 0));
    comm.hasNet = true; before = calls;
    int beforeSync = syncs;
    flagcxDevFlush(&comm, 7, 4, 8);
    int expected = (missing == 0) * FLAGCX_DEVICE_HAS_NET_TRANSPORT;
    assert(calls == before + expected && syncs == beforeSync + 2 * expected);
  }
#if TEST_GDA_BINDING
  for (int missing = 0; missing != 2; ++missing) {
    host = {}; host.netUseGda = 1;
    host.gdaDeviceContext = missing == 0 ? nullptr : handle;
    host.contextCount = missing == 0 ? 1 : 0;
    assert(publishReadiness(&hostComm, &host) == flagcxInternalError);
    assert(host.netOneSidedReady == 0);
  }
#endif
}
"""
        for local, transport, gda in ((0, 1, 1), (1, 1, 1), (1, 0, 1),
                                      (0, 1, 0), (1, 1, 0)):
            with self.subTest(local_descriptor=local, net_transport=transport, gda=gda):
                self.compile_run(harness + macros + bodies + main, flags=(
                    f"-DFLAGCX_DEVICE_REQUIRES_LOCAL_NET_DESCRIPTOR={local}",
                    f"-DFLAGCX_DEVICE_HAS_NET_TRANSPORT={transport}",
                ) + (f"-DTEST_GDA_BINDING={gda}",))

    def test_unified_barrier_dispatch_preserves_split_operations_and_parameters(self):
        source = (ROOT / "bindings/ir/flagcx_device_unified_ir_impl.h").read_text()
        scalar = (ROOT / "bindings/ir/flagcx_device_scalar_ir_impl.h").read_text()
        utils = (ROOT / "flagcx/adaptor/include/device_utils.h").read_text()
        macros = utils[utils.index("#if FLAGCX_DEVICE_REQUIRES_LOCAL_NET_DESCRIPTOR"):
                       utils.index("// IR extern")]
        names = ["flagcxDevBarrier" + op for op in ("Arrive", "Wait", "Sync")]
        legacy = "\n".join(
            f"__attribute__((deprecated)) void flagcx{team}Barrier{op}S(...);"
            for team in ("Intra", "Inter", "World") for op in ("Arrive", "Wait", "Sync"))
        self_source = r"""
#include <cassert>
#include <cstdint>
#define FLAGCX_IR_EXTERN_C
#define FLAGCX_DEVICE_INLINE_DECORATOR
#define FLAGCX_IR_GLOBAL_PTR
using flagcxDevTeamKind_t = int;
using flagcxDevContext_t = int;
using flagcxDevCoopKind_t = int;
using flagcxDevMemoryOrder_t = int;
using flagcxDevMemoryScope_t = int;
enum { FLAGCX_TEAM_INTRA, FLAGCX_TEAM_INTER, FLAGCX_TEAM_WORLD };
enum class flagcxDevNetFenceLevel { Relaxed, Put, Get };
struct flagcxTeamTagIntra { enum { kind = 0 }; };
struct flagcxTeamTagInter { enum { kind = 1 }; };
struct flagcxTeamTagWorld { enum { kind = 2 }; };
struct flagcxCoopAny { int kind; };
flagcxCoopAny flagcxMakeCoopFromKind(int kind) { return {kind}; }
struct CommBase {
  int getRank() const { return 9; }
  int getSize() const { return 16; }
  int getIntraSize() const { return 4; }
};
struct flagcxDevComm { CommBase _commBase; };
struct flagcxTeam { struct { int nRanks, rank, stride; } _teamBase; };
flagcxTeam flagcxTeamIntra(const flagcxDevComm &) { return {{4, 1, 1}}; }
struct flagcxDevNet {
  CommBase _dc;
  int context;
  flagcxDevNet(const flagcxDevComm &, int id) : context(id) {}
};
const void *flagcxDevNetGetFromCommS(const void *comm, int id) {
  static flagcxDevNet net(*(const flagcxDevComm *)comm, 0);
  net.context = id; return &net;
}
struct Trace { int team, coop, context, index, count, rank, stride, op, order;
  bool multimem; flagcxDevNetFenceLevel fence;
};
static Trace seen;
void record(int tag, int coop, int context, uint32_t index, flagcxTeam team, bool multi) {
  seen = {tag, coop, context, (int)index, team._teamBase.nRanks,
          team._teamBase.rank, team._teamBase.stride, -1, -1, multi,
          flagcxDevNetFenceLevel::Put};
}
struct Backend {
  void arrive(int order, flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    seen.op = 0; seen.order = order; seen.fence = fence;
  }
  void wait(int order, flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    seen.op = 1; seen.order = order; seen.fence = fence;
  }
  void sync(int order, flagcxDevNetFenceLevel fence = flagcxDevNetFenceLevel::Relaxed) {
    seen.op = 2; seen.order = order; seen.fence = fence;
  }
};
template <typename Tag, typename Coop> struct flagcxDevBarrier {
  Backend _impl;
  flagcxDevBarrier(Coop coop, const flagcxDevNet &net, flagcxTeam team, uint32_t index) {
    assert((int)Tag::kind == FLAGCX_TEAM_INTER);
    record(Tag::kind, coop.kind, net.context, index, team, false);
  }
  flagcxDevBarrier(Coop coop, flagcxTeamTagWorld, const flagcxDevNet &net,
                    uint32_t index, bool multi) {
    assert((int)Tag::kind == FLAGCX_TEAM_WORLD);
    record(Tag::kind, coop.kind, net.context, index, {{16, 9, 1}}, multi);
  }
  __attribute__((deprecated)) void arrive(int order, flagcxDevNetFenceLevel fence) {
    _impl.arrive(order, fence);
  }
  __attribute__((deprecated)) void wait(int order, flagcxDevNetFenceLevel fence) {
    _impl.wait(order, fence);
  }
  void sync(int order, flagcxDevNetFenceLevel fence) { _impl.sync(order, fence); }
};
template <typename Coop> struct flagcxDevBarrier<flagcxTeamTagIntra, Coop> {
  Backend _impl;
  flagcxDevBarrier(Coop coop, const flagcxDevComm &, flagcxTeam team,
                    uint32_t index, bool multi) {
    record(FLAGCX_TEAM_INTRA, coop.kind, -1, index, team, multi);
  }
  void arrive(int order) { _impl.arrive(order); }
  void wait(int order) { _impl.wait(order); }
  void sync(int order) { _impl.sync(order); }
};
"""
        main = r"""
int main() {
  flagcxDevComm comm;
  void (*ops[])(const void *, int, uint32_t, int, int, int, int) = {
      flagcxDevBarrierArrive, flagcxDevBarrierWait, flagcxDevBarrierSync};
  for (int op = 0; op < 3; ++op) for (int team = 0; team < 3; ++team)
    for (int coop = 0; coop < 3; ++coop) for (int scope = 0; scope < 4; ++scope) {
      ops[op](&comm, team, 7, 5, coop, 4, scope);
      assert(seen.team == team && seen.coop == coop && seen.op == op);
      assert(seen.context == (team == FLAGCX_TEAM_INTRA ? -1 : 5));
      assert(seen.index == 7 && !seen.multimem && seen.order == 4);
      assert(seen.fence == flagcxDevNetFenceLevel::Relaxed);
      assert(seen.count == (team == FLAGCX_TEAM_WORLD ? 16 : 4));
      assert(seen.rank == (team == FLAGCX_TEAM_INTRA ? 1 : team == FLAGCX_TEAM_INTER ? 2 : 9));
      assert(seen.stride == (team == FLAGCX_TEAM_INTER ? 4 : 1));
    }
}
"""
        bodies = "\n".join(function(source, name) for name in names)
        helper = "\nflagcxTeam " + function(scalar, "flagcxMakeInterTeamFromNet")
        for local in (0, 1):
            with self.subTest(local_descriptor=local):
                self.compile_run(self_source + macros + legacy + helper + bodies + main,
                                 flags=(f"-DFLAGCX_DEVICE_REQUIRES_LOCAL_NET_DESCRIPTOR={local}",))

    def test_size_suffix_preserves_whitespace_and_numeric_parsing(self):
        source = (ROOT / "test/tools.cc").read_text()
        self.compile_run("#include <cstdio>\n#include <cassert>\n" +
                         function(source, "parsesize") + r"""
int main() {
  assert(parsesize("1G") == 1073741824.0);
  assert(parsesize("1.5 M") == 1572864.0);
  assert(parsesize(" 2\tk ") == 2048.0);
  assert(parsesize("3") == 3.0);
  assert(parsesize("3  ") == 3.0);
  assert(parsesize("4x") == -1.0);
  assert(parsesize("invalid") == -1.0);
}
""")

    def test_torch_check_propagates_outer_result_and_evaluates_once(self):
        source = (ROOT / "plugin/torch/flagcx/include/utils_flagcx.hpp").read_text()
        macro = source[source.index("#define C10D_FLAGCX_CHECK"):
                       source.index("namespace c10d")]
        self.compile_run(r"""
#include <string>
#include <stdexcept>
#include <cassert>
enum flagcxResult_t { flagcxSuccess = 0, flagcxRemoteError = 6 };
static int seenError = -1, reasonCalls = 0, commandCalls = 0;
std::string getFlagcxErrorDetailStr(flagcxResult_t value, int reason) {
  seenError = value; assert(reason == 42); return "failure";
}
#define TORCH_CHECK_WITH(type, predicate, text) \
  do { if (!(predicate)) throw std::runtime_error(text); } while (0)
int reason() { ++reasonCalls; return 42; }
flagcxResult_t command(flagcxResult_t value) { ++commandCalls; return value; }
""" + macro + r"""
int main() {
  flagcxResult_t result = flagcxRemoteError;
  bool threw = false;
  try { C10D_FLAGCX_CHECK(result, reason()); }
  catch (const std::runtime_error &) { threw = true; }
  assert(threw && seenError == flagcxRemoteError && reasonCalls == 1);
  assert(result == flagcxRemoteError);
  C10D_FLAGCX_CHECK(command(flagcxSuccess), reason());
  assert(commandCalls == 1 && reasonCalls == 1);
  try { C10D_FLAGCX_CHECK(command(flagcxRemoteError), reason()); }
  catch (const std::runtime_error &) {}
  assert(commandCalls == 2 && reasonCalls == 2);
}
""")

    def test_staging_arithmetic_on_32_and_64_bit_size_types(self):
        source = (ROOT / "flagcx/adaptor/include/device_api/flagcx_device_internal.h").read_text()
        helpers = (function(source, "flagcxPutValueStagingLayout") + "\n" +
                   function(source, "flagcxKernelPutValueStagingOffset"))
        # Include the split return type, which is on the line before the name.
        helpers = helpers.replace("flagcxKernelPutValueStagingOffset(",
                                  "bool flagcxKernelPutValueStagingOffset(", 1)
        for size_type in ("uint32_t", "uint64_t"):
            with self.subTest(size_type=size_type):
                self.compile_run(r"""
#include <cstdint>
#include <climits>
#include <cassert>
using Size = SIZE_TYPE;
struct flagcxDevCommInternal {
  int putValueStagingContextCount, putValueStagingSlotCount;
  Size putValueStagingKernelBaseSlot, putValueStagingTotalSlotCount;
};
#define size_t Size
""".replace("SIZE_TYPE", size_type) + helpers + r"""
#undef size_t
int main() {
  Size base, slots, bytes;
  assert(flagcxPutValueStagingLayout(8, 2, 4, &base, &slots, &bytes));
  assert(base == 8 && slots == 16 && bytes == 128);
  assert(!flagcxPutValueStagingLayout(-1, 2, 4, &base, &slots, &bytes));
  assert(!flagcxPutValueStagingLayout(INT_MAX, INT_MAX, INT_MAX,
                                    &base, &slots, &bytes));
  flagcxDevCommInternal comm = {2, 4, 8, 16};
  uint64_t offset = 999;
  assert(flagcxKernelPutValueStagingOffset(&comm, 1, 3, &offset));
  assert(offset == 120);
  assert(!flagcxKernelPutValueStagingOffset(&comm, 2, 0, &offset));
  comm.putValueStagingKernelBaseSlot = Size(-2);
  comm.putValueStagingTotalSlotCount = Size(-1);
  assert(!flagcxKernelPutValueStagingOffset(&comm, 1, 3, &offset));
}
""")

    def test_inter_alltoall_verifiers_preserve_round_and_peer_scope(self):
        source = (ROOT / "test/unittest/device_api/test_device_api_inter.cpp").read_text()
        helpers = (function(source, "verifyAlltoAll") + "\n" +
                   function(source, "verifyInterAlltoAll"))
        self.compile_run(r"""
#include <cstddef>
#include <cmath>
#include <cassert>
""" + helpers + r"""
int main() {
  float payload[] = {100, 101, 102, 1100, 1101, 1102,
                     2100, 2101, 2102, 3100, 3101, 3102};
  assert(verifyAlltoAll(payload, 3, 4, 1));
  assert(verifyInterAlltoAll(payload, 3, 4, 1, 2));
  payload[0] = -1; // Local peers are excluded from the inter-only check.
  assert(!verifyAlltoAll(payload, 3, 4, 1));
  assert(verifyInterAlltoAll(payload, 3, 4, 1, 2));
  payload[0] = 100;
  payload[6] = -1;
  assert(!verifyInterAlltoAll(payload, 3, 4, 1, 2));
  payload[6] = 2100;
  for (float &value : payload) value += 7 * 17;
  assert(verifyAlltoAll(payload, 3, 4, 1, 7));
  assert(!verifyAlltoAll(payload, 3, 4, 1));
  assert(!verifyAlltoAll(payload, 3, 4, 1, 6));
}
""")

    def test_device_comm_copy_initializes_net_contexts_and_rolls_back(self):
        source = (ROOT / "flagcx/adaptor/device_api/default_dev_api_backend.cc").read_text()
        self.compile_run(r"""
#include <pthread.h>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <type_traits>
enum flagcxResult_t {
  flagcxSuccess, flagcxInvalidArgument, flagcxUnhandledDeviceError,
  flagcxInvalidUsage
};
enum { flagcxMemDevice, flagcxMemcpyHostToDevice };
struct flagcxDevCommInternal {
  pthread_mutex_t cachedPtrMutex;
  void *cachedDevicePtr, *cachedNetContextsPtr, *cachedGridBarrierPtr;
  int contexts, gdaDraining;
};
using flagcxDevComm_t = flagcxDevCommInternal *;
struct flagcxDevComm {
private:
  int padding;
public:
  int _contextCount;
  void *_netContexts;
  unsigned *_gridBarrierState;
  explicit flagcxDevComm(const flagcxDevCommInternal &comm)
      : padding(7), _contextCount(comm.contexts), _netContexts(nullptr),
        _gridBarrierState(nullptr) { assert(padding == 7); }
};
static_assert(!std::is_standard_layout<flagcxDevComm>::value, "exercise nonstandard layout");
static_assert(std::is_trivially_copyable<flagcxDevComm>::value, "byte copy contract");
static void *live[3];
static int liveCount;
bool isLive(void *ptr) {
  for (int i = 0; i < 3; ++i) if (live[i] == ptr) return true;
  return false;
}
static int allocationCount, failAllocation, constructCalls;
static bool failCopy, failMemset, failSynchronize;
struct Adaptor {
  flagcxResult_t deviceMalloc(void **ptr, size_t bytes, int, void *) {
    if (++allocationCount == failAllocation) return flagcxUnhandledDeviceError;
    *ptr = malloc(bytes); assert(*ptr);
    for (int i = 0; i < 3; ++i) if (!live[i]) {
      live[i] = *ptr; ++liveCount; return flagcxSuccess;
    }
    abort();
  }
  flagcxResult_t deviceMemset(void *ptr, int value, size_t bytes, int, void *) {
    if (failMemset) return flagcxUnhandledDeviceError;
    memset(ptr, value, bytes); return flagcxSuccess;
  }
  flagcxResult_t deviceMemcpy(void *dst, const void *src, size_t bytes,
                             int, void *, void *) {
    if (failCopy) return flagcxUnhandledDeviceError;
    memcpy(dst, src, bytes); return flagcxSuccess;
  }
  flagcxResult_t deviceFree(void *ptr, int, void *) {
    for (int i = 0; i < 3; ++i) if (live[i] == ptr) {
      live[i] = nullptr; --liveCount; free(ptr); return flagcxSuccess;
    }
    abort();
  }
  flagcxResult_t deviceSynchronize() {
    return failSynchronize ? flagcxUnhandledDeviceError : flagcxSuccess;
  }
};
static Adaptor adaptor, *deviceAdaptor = &adaptor;
size_t flagcxDevNetSizeOf() { return 64; }
void flagcxDevNetLaunchConstruct(void *net, void *comm, int count, void *) {
  flagcxDevCommInternal di = {}; flagcxDevComm snapshot(di);
  memcpy(&snapshot, comm, sizeof(snapshot));
  assert(snapshot._netContexts == net && snapshot._contextCount == count);
  assert(isLive(snapshot._gridBarrierState));
  ++constructCalls;
}
#define FLAGCXCHECKGOTO(call, result, label) \
  do { result = (call); if (result != flagcxSuccess) goto label; } while (0)
""" + function(source, "defaultDevApiCommGetDevicePtr") + r"""
int main() {
  for (int scenario = 0; scenario < 8; ++scenario) {
    flagcxDevCommInternal di = {}; di.contexts = scenario == 7 ? 0 : 2;
    pthread_mutex_init(&di.cachedPtrMutex, nullptr);
    allocationCount = constructCalls = 0;
    failAllocation = scenario >= 1 && scenario <= 3 ? scenario : 0;
    failCopy = scenario == 4; failMemset = scenario == 5;
    failSynchronize = scenario == 6;
    void *device = nullptr;
    flagcxResult_t result = defaultDevApiCommGetDevicePtr(&di, &device);
    if (scenario == 0 || scenario == 7) {
      assert(result == flagcxSuccess && device == di.cachedDevicePtr);
      int calls = allocationCount;
      assert(defaultDevApiCommGetDevicePtr(&di, &device) == flagcxSuccess);
      assert(allocationCount == calls);
      assert(constructCalls == (di.contexts ? 1 : 0));
      void *cached[3] = {di.cachedDevicePtr, di.cachedNetContextsPtr,
                         di.cachedGridBarrierPtr};
      di.gdaDraining = 1;
      device = reinterpret_cast<void *>(1);
      assert(defaultDevApiCommGetDevicePtr(&di, &device) == flagcxInvalidUsage);
      assert(device == nullptr && allocationCount == calls);
      assert(constructCalls == (di.contexts ? 1 : 0));
      assert(di.cachedDevicePtr == cached[0] && di.cachedNetContextsPtr == cached[1] &&
             di.cachedGridBarrierPtr == cached[2]);
      for (int i = 0; i < 3; ++i) {
        if (cached[i]) adaptor.deviceFree(cached[i], flagcxMemDevice, nullptr);
      }
    } else {
      assert(result != flagcxSuccess && device == nullptr);
      assert(!di.cachedDevicePtr && !di.cachedNetContextsPtr && !di.cachedGridBarrierPtr);
    }
    assert(liveCount == 0);
    pthread_mutex_destroy(&di.cachedPtrMutex);
  }
  flagcxDevCommInternal draining = {};
  draining.contexts = 2;
  draining.gdaDraining = 1;
  pthread_mutex_init(&draining.cachedPtrMutex, nullptr);
  allocationCount = constructCalls = 0;
  void *device = reinterpret_cast<void *>(1);
  assert(defaultDevApiCommGetDevicePtr(&draining, &device) == flagcxInvalidUsage);
  assert(device == nullptr && allocationCount == 0 && constructCalls == 0);
  assert(!draining.cachedDevicePtr && !draining.cachedNetContextsPtr &&
         !draining.cachedGridBarrierPtr && liveCount == 0);
  pthread_mutex_destroy(&draining.cachedPtrMutex);
}
""")



if __name__ == "__main__":
    unittest.main()
