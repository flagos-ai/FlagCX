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

    def test_failed_barrier_attachment_retains_partial_owner_outputs(self):
        source = (ROOT / "flagcx/adaptor/device_api/default_dev_api_backend.cc").read_text()
        body = function(source, "setupIpcBarriers")
        harness = r"""
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
using flagcxResult_t = int;
using flagcxShmHandle_t = void *;
constexpr int flagcxSuccess=0, flagcxSystemError=1, flagcxRemoteError=2;
constexpr int flagcxMemDevice=0, flagcxInternalBufferDevice=0;
constexpr int flagcxMemcpyHostToDevice=0, FLAGCX_DEVICE_CTA_COUNT=1;
constexpr int SHM_PATH_MAX=128;
#define FLAGCX_INIT 0
#define INFO(...) ((void)0)
#define FLAGCXCHECK(call) do { int result=(call); if(result) return result; } while(0)
#define FLAGCXCHECKGOTO(call, result, label) do { result=(call); if(result) goto label; } while(0)
struct Comm {
  int localRanks, nranks, rank, localRank;
  int *localRankToRank;
  void *bootstrap;
  uint64_t magic;
  struct { void *devPeerPtrs; } ipcTable[1];
};
using flagcxComm_t = Comm *;
struct View {
  int nLocalRanks, nBarriers, barrierIpcIndex, registeredBarrierShmCount;
  int *localRankToRank;
  uint64_t *localBarrierFlags;
  uint64_t **barrierPeers, **barrierDevPeerPtrsRaw;
  bool localBarrierFlagsDeviceAllocated;
  void *localBarrierShmPtr;
  void **peerBarrierShmPtrs;
  size_t barrierShmSize;
  flagcxShmHandle_t myShmHandle, *peerShmHandles;
};
using flagcxDevComm_t = View *;
struct Adaptor {
  int (*deviceMemset)(void *, int, size_t, int, void *);
  int (*deviceMalloc)(void **, size_t, int, void *);
  int (*deviceMemcpy)(void *, void *, size_t, int, void *, void *);
  int (*hostGetDevicePointer)(void **, void *);
} adaptor;
Adaptor *deviceAdaptor = &adaptor;
int flagcxParamSignalHostEnable() { return 1; }
template<class... A> int flagcxCommAllocInternalBuffer(A...) { return flagcxSystemError; }
template<class... A> int flagcxConvergeBootstrapStatus(A...) { return flagcxSystemError; }
struct Ipc { int slot; int result() { return flagcxSystemError; } };
template<class... A> Ipc flagcxCommWindowRegisterIpcInternal(A...) { return {0}; }
template<class... A> int bootstrapCollBarrier(A...) { return flagcxSuccess; }
int shmHostRegister(void *, size_t) { return flagcxSuccess; }
int ownStorage, partialStorage, opens=0;
int flagcxShmOpen(char *, size_t, size_t, void **mapping, void **,
                 int references, flagcxShmHandle_t *owner) {
  ++opens;
  *mapping = references>0 ? &ownStorage : &partialStorage;
  *owner = reinterpret_cast<void *>(references>0 ? 0x11 : 0x22);
  // Attachment failed and its own rollback could not close this output.
  return references>0 ? flagcxSuccess : flagcxRemoteError;
}
"""
        self.compile_run(harness + body + r"""
int main() {
  int ranks[] = {0,1};
  Comm comm = {}; comm.localRanks=comm.nranks=2; comm.localRankToRank=ranks;
  View view = {};
  assert(setupIpcBarriers(&comm, &view)==flagcxRemoteError);
  assert(opens==2 && view.nLocalRanks==2);
  assert(view.myShmHandle==reinterpret_cast<void *>(0x11));
  assert(view.peerShmHandles[1]==reinterpret_cast<void *>(0x22));
  assert(view.peerBarrierShmPtrs[1]==&partialStorage);
  assert(view.registeredBarrierShmCount==0);
  free(view.peerShmHandles); free(view.peerBarrierShmPtrs);
  free(view.localRankToRank);
}
""")

    def test_unified_net_forwarding_preserves_context_coop_and_flush_guards(self):
        source = (ROOT / "bindings/ir/flagcx_device_unified_ir_impl.h").read_text()
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
struct flagcxCoopAny { int kind; };
flagcxCoopAny flagcxMakeCoopFromKind(int kind) { return {kind}; }
static int seenContext, seenCoop, seenSlot, seenBits, seenOrder, calls, fences;
static uint64_t seenLeast;
struct Base {
  bool ready;
  bool isOneSidedTransportReady() const { return ready; }
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
  void flush(flagcxCoopAny coop, int order) const { record(coop.kind, -1, 0, 0, order); }
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
  flagcxDevComm comm = {{true}, true};
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
  if (FLAGCX_DEVICE_HAS_NET_TRANSPORT) {
    assert(seenContext == 7 && seenCoop == 4 && seenOrder == 8);
  }
  before = calls;
  comm._commBase.ready = false;
  flagcxDevFlush(&comm, 7, 4, 8);
  assert(calls == before);
  comm._commBase.ready = true; comm.hasNet = false;
  flagcxDevFlush(&comm, 7, 4, 8);
  assert(calls == before && fences == 3);
}
"""
        for local, transport in ((0, 1), (1, 1), (1, 0)):
            with self.subTest(local_descriptor=local, net_transport=transport):
                self.compile_run(harness + macros + bodies + main, flags=(
                    f"-DFLAGCX_DEVICE_REQUIRES_LOCAL_NET_DESCRIPTOR={local}",
                    f"-DFLAGCX_DEVICE_HAS_NET_TRANSPORT={transport}",
                ))

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

    def test_device_comm_copy_initializes_net_contexts_and_rolls_back(self):
        source = (ROOT / "flagcx/adaptor/device_api/default_dev_api_backend.cc").read_text()
        self.compile_run(r"""
#include <pthread.h>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <type_traits>
enum flagcxResult_t { flagcxSuccess, flagcxInvalidArgument, flagcxUnhandledDeviceError };
enum { flagcxMemDevice, flagcxMemcpyHostToDevice };
struct flagcxDevCommInternal {
  pthread_mutex_t cachedPtrMutex;
  void *cachedDevicePtr, *cachedNetContextsPtr, *cachedGridBarrierPtr;
  int contexts;
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
}
""")



if __name__ == "__main__":
    unittest.main()
