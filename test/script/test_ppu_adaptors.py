"""Host contract tests for native adaptor conversions and async allocation.

SDK stubs deliberately use different enum values from FlagCX. These tests
check API wiring and error propagation; PPU CI checks the actual SDK ABI and
device execution.
"""
import os
import importlib.util
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CXX = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
INCLUDE = ROOT / "flagcx/adaptor/include"

def production_function(source, name):
    match = re.search(r"(?:static\s+)?(?:inline\s+)?(?:flagcxResult_t|MPI_Datatype|MPI_Op|void|stagedBuffer_t)\s+"
                      + re.escape(name) + r"\s*\(", source)
    if match is None:
        raise ValueError("Missing production function " + name)
    opening = source.index("{", match.end())
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end]



def generate_native_contracts(output):
    cases = []
    text = '// SDK boundary mocks around actual production CCL callbacks.\n'
    platforms = (
        ('nccl', 'nccl', 'nccl', True), ('rccl', 'rccl', 'nccl', True),
        ('durccl', 'durccl', 'nccl', True), ('ixnccl', 'ixnccl', 'nccl', True),
        ('mccl', 'mccl', 'mccl', True), ('musa_mccl', 'mccl', 'mccl', False),
        ('ppu_pccl', 'ppupccl', 'pccl', True), ('tccl', 'tccl', 'tccl', True),
        ('hccl', 'hccl', 'Hccl', False), ('cncl', 'cncl', 'cncl', False),
        ('pccl', 'pccl', 'pccl', False), ('xccl', 'xccl', 'bkcl', False),
        ('eccl', 'eccl', 'eccl', False),
    )
    for file, adaptor, sdk, native in platforms:
        source = (ROOT / f'flagcx/adaptor/ccl/{file}_adaptor.cc').read_text()
        ns = file + 'OwnerContract'
        text += f'namespace {ns} {{\n'
        text += 'struct flagcxInnerComm { void *base; }; using flagcxInnerComm_t = flagcxInnerComm *;\n'
        text += 'static void free(void *p) { NativeOwnerProbe::release(p); }\n'
        text += 'static int finalizeSdk(void *p) { EXPECT_EQ(p, (void *)42); ++NativeOwnerProbe::finalizes; return NativeOwnerProbe::result; }\n'
        text += 'static int destroySdk(void *p) { EXPECT_EQ(p, (void *)42); ++NativeOwnerProbe::destroys; return NativeOwnerProbe::result; }\n'
        if file == 'ppu_pccl':
            text += '''struct ppupcclSymbols { int (*pcclCommFinalize)(void *); int (*pcclCommDestroy)(void *); };
static const ppupcclSymbols api = {finalizeSdk, destroySdk};
static flagcxResult_t ppupcclLoadApi(const ppupcclSymbols **p) { *p = &api; return flagcxSuccess; }
static flagcxResult_t ppupcclResultToFlagcx(int r) { return (flagcxResult_t)r; }
'''
        elif file == 'cncl':
            text += 'static int cnclDestroyComms(void **p, int n) { EXPECT_EQ(n, 1); return destroySdk(*p); }\n'
        elif file == 'xccl':
            text += 'static int bkcl_destroy_context(void *p) { return destroySdk(p); }\n'
        else:
            text += f'static int {sdk}CommDestroy(void *p) {{ return destroySdk(p); }}\n'
            if native:
                text += f'static int {sdk}CommFinalize(void *p) {{ return finalizeSdk(p); }}\n'
        if file in ('hccl', 'cncl', 'pccl'):
            name = {'hccl': 'h2f', 'cncl': 'c2f', 'pccl': 'p2f'}[file]
            text += f'static const int {name}_ret_map[] = {{0,1,2,3,4,5,6,7,8}};\n'
        if file == 'tccl':
            text += 'using tcclResult_t = int;\nstatic flagcxResult_t fromTcclResult(int r) { return (flagcxResult_t)r; }\n'
        for op in ('Finalize', 'Destroy'):
            text += production_function(source, adaptor + 'AdaptorComm' + op) + '\n'
        text += 'static void *make() { auto *p = (flagcxInnerComm *)std::calloc(1, sizeof(flagcxInnerComm)); p->base = (void *)42; NativeOwnerProbe::owners.insert(p); return p; }\n'
        text += '}\n'
        cases.append(f'''  {{"{file}", {str(native).lower()}, {ns}::make,
    [](void *p) {{ return {ns}::{adaptor}AdaptorCommFinalize(({ns}::flagcxInnerComm_t)p); }},
    [](void *p) {{ return {ns}::{adaptor}AdaptorCommDestroy(({ns}::flagcxInnerComm_t)p); }} }}''')
    text += 'static const NativeOwnerContract nativeOwners[] = {\n' + ',\n'.join(cases) + '\n};\n'
    (Path(output) / 'native_owners.h').write_text(text)


def generate_host_contracts(output):
    for vendor in ('mpi', 'gloo', 'bootstrap'):
        source = (ROOT / f'flagcx/adaptor/ccl/{vendor}_adaptor.cc').read_text()
        names = [vendor + 'CleanupOwner', vendor + 'AdaptorGetStagedBuffer',
                 vendor + 'AdaptorCommFinalize', vendor + 'AdaptorCommDestroy']
        if vendor != 'bootstrap':
            names.insert(0, vendor + 'ReleaseStagedBuffers')
        if vendor == 'gloo':
            names.insert(2, 'glooFindStagedBuffer')
            names += ['glooAdaptorSend', 'glooAdaptorRecv']
        text = '// Exact production host allocation and lifecycle functions.\n'
        text += '\n'.join(production_function(source, name) for name in names)
        (Path(output) / (vendor + '_staging.h')).write_text(text + '\n')


EVENT_CONTRACTS = (
    ("cuda", "cuda", "nvidia", "cudaEventCreateWithFlags", "cudaEventDestroy", "cudaSuccess", "cudaEventDefault", "cudaEventDisableTiming"),
    ("hip", "hip", "amd", "hipEventCreateWithFlags", "hipEventDestroy", "hipSuccess", "hipEventDefault", "hipEventDisableTiming"),
    ("duhip", "duhip", "du", "hipEventCreateWithFlags", "hipEventDestroy", "hipSuccess", "hipEventDefault", "hipEventDisableTiming"),
    ("ixcuda", "ixcuda", "iluvatar", "cudaEventCreateWithFlags", "cudaEventDestroy", "cudaSuccess", "cudaEventDefault", "cudaEventDisableTiming"),
    ("musa", "musa", "musa", "musaEventCreateWithFlags", "musaEventDestroy", "musaSuccess", "musaEventDefault", "musaEventDisableTiming"),
    ("maca", "maca", "metax", "mcEventCreateWithFlags", "mcEventDestroy", "mcSuccess", "mcEventDefault", "mcEventDisableTiming"),
    ("mlu", "mlu", "cambricon", "cnrtNotifierCreateWithFlags", "cnrtNotifierDestroy", "cnrtSuccess", "CNRT_NOTIFIER_DEFAULT", "CNRT_NOTIFIER_DISABLE_TIMING_ALL"),
    ("cann", "cann", "ascend", "aclrtCreateEventWithFlag", "aclrtDestroyEvent", "ACL_SUCCESS", "ACL_EVENT_TIME_LINE", "ACL_EVENT_SYNC"),
    ("tops", "tops", "enflame", "topsEventCreateWithFlags", "topsEventDestroy", "topsSuccess", "topsEventDefault", "topsEventDisableTiming"),
    ("ptpu", "ptpu", "sunrise", "tangEventCreateWithFlags", "tangEventDestroy", "tangSuccess", "tangEventDefault", "tangEventDisableTiming"),
    ("tsmicro", "tsmicro", "tsmicro", "txEventCreate", "txEventDestroy", "TX_SUCCESS", None, None),
    ("kunlunxin", "kunlun", "kunlunxin", "cudaEventCreateWithFlags", "cudaEventDestroy", "cudaSuccess", "cudaEventDefault", "cudaEventDisableTiming"),
    ("hggc", "hggc", "ppu", "hggcEventCreateWithFlags", "hggcEventDestroy", "hggcSuccess", "hggcEventDefault", "hggcEventDisableTiming"),
)


def error_macro(header, name):
    lines = (INCLUDE / (header + "_adaptor.h")).read_text().splitlines()
    first = next(i for i, line in enumerate(lines)
                 if line.startswith("#define " + name + "(func)"))
    last = first
    while lines[last].endswith("\\"):
        last += 1
    return "\n".join(lines[first:last + 1])


def generate_event_contracts(output):
    text = "// SDK and allocator boundary mocks around actual platform callbacks.\n"
    cases = []
    for file, adaptor, header, create, destroy, success, default, timing in EVENT_CONTRACTS:
        source = (ROOT / f"flagcx/adaptor/device/{file}_adaptor.cc").read_text()
        namespace = adaptor + "EventContract"
        text += f"namespace {namespace} {{\n"
        text += "struct flagcxEvent { uintptr_t base; };\n"
        text += "typedef flagcxEvent *flagcxEvent_t;\n"
        text += "typedef int hipError_t; typedef int tangError_t; typedef int hggcError_t;\n"
        text += "static void *calloc(size_t n, size_t bytes) { return DeviceEventMock::allocate(n, bytes); }\n"
        text += "static void free(void *p) { DeviceEventMock::release(p); }\n"
        text += f"enum {{ {success} = 0"
        if default:
            text += f", {default} = 7, {timing} = 11"
        text += " };\n"
        args = "uintptr_t *p, unsigned int flags" if default else "uintptr_t *p"
        flags = "flags" if default else "0"
        text += f"static int {create}({args}) {{ return DeviceEventMock::create(p, {flags}); }}\n"
        text += f"static int {destroy}(uintptr_t p) {{ return DeviceEventMock::destroy(p); }}\n"
        macro = "HGGCCHECK" if file == "hggc" else "DEVCHECK"
        if file == "hggc":
            text += "static flagcxResult_t hggcResultToFlagcx(int r) { return r == 0 ? flagcxSuccess : r == 70 ? flagcxInProgress : flagcxUnhandledDeviceError; }\n"
        text += error_macro(header, macro) + "\n"
        for operation in ("Create", "Destroy"):
            text += production_function(source, adaptor + "AdaptorEvent" + operation) + "\n"
        text += f"#undef {macro}\n}} // namespace {namespace}\n"
        expected = "flagcxInProgress" if file == "hggc" else "flagcxUnhandledDeviceError"
        cases.append(f'''  {{"{adaptor}", {7 if default else 0}, {11 if default else 0}, {expected},
    [](void **p, flagcxEventType_t type) {{ return {namespace}::{adaptor}AdaptorEventCreate(({namespace}::flagcxEvent_t *)p, type); }},
    [](void *p) {{ return {namespace}::{adaptor}AdaptorEventDestroy(({namespace}::flagcxEvent_t)p); }} }}''')
    text += "static const DeviceEventContract eventContracts[] = {\n" + ",\n".join(cases) + "\n};\n"
    (Path(output) / "device_events.h").write_text(text)


RUNTIME = r"""
#pragma once
#include <cstddef>
struct hggcStreamImpl;
struct hggcEventImpl;
struct hggcPoolImpl;
using hggcStream_t = hggcStreamImpl *;
using hggcEvent_t = hggcEventImpl *;
using hggcMemPool_t = hggcPoolImpl *;
struct hggcIpcMemHandle_t { char bytes[64]; };
enum hggcError_t { hggcSuccess = 0, hggcErrorInvalidValue = 1, hggcErrorNotReady = 70, hggcErrorUnknown = 99 };
enum hggcMemoryType { hggcMemoryTypeHost, hggcMemoryTypeDevice, hggcMemoryTypeManaged };
struct hggcPointerAttributes {
  int device;
#if FLAGCX_HGGC_HAS_POINTER_TYPE
  hggcMemoryType type;
#else
  hggcMemoryType memoryType;
  bool isManaged;
#endif
};
static hggcError_t pointerStatus = hggcSuccess;
inline hggcError_t hggcGetLastError() { return hggcSuccess; }
inline hggcError_t hggcPointerGetAttributes(hggcPointerAttributes *a, const void *) {
  a->device = 3;
#if FLAGCX_HGGC_HAS_POINTER_TYPE
  a->type = hggcMemoryTypeManaged;
#else
  a->memoryType = hggcMemoryTypeHost;
  a->isManaged = true;
#endif
  return pointerStatus;
}
enum hggcMemcpyKind { hggcMemcpyHostToDevice = 10, hggcMemcpyDeviceToHost,
                      hggcMemcpyDeviceToDevice };
static int memcpyCalls = 0;
static hggcMemcpyKind seenMemcpyKind = hggcMemcpyHostToDevice;
static hggcStream_t seenMemcpyStream = nullptr;
static hggcError_t memcpyStatus = hggcSuccess;
inline hggcError_t hggcMemcpy(void *, const void *, size_t, hggcMemcpyKind kind) {
  ++memcpyCalls; seenMemcpyKind = kind; seenMemcpyStream = nullptr; return memcpyStatus;
}
inline hggcError_t hggcMemcpyAsync(void *, const void *, size_t, hggcMemcpyKind kind, hggcStream_t stream) {
  ++memcpyCalls; seenMemcpyKind = kind; seenMemcpyStream = stream; return memcpyStatus;
}
enum { hggcHostAllocMapped = 1 };
static int mallocCalls = 0;
static int poolMallocCalls = 0;
static hggcError_t mallocStatus = hggcSuccess;
static hggcStream_t seenStream = nullptr;
inline hggcError_t hggcMalloc(void **p, size_t) {
  *p = reinterpret_cast<void *>(2); return mallocStatus;
}
inline hggcError_t hggcHostAlloc(void **p, size_t size, unsigned) {
  return hggcMalloc(p, size);
}
inline hggcError_t hggcHostGetDevicePointer(void **device, void *host, unsigned) {
  *device = host; return hggcSuccess;
}
inline hggcError_t hggcMallocAsync(void **p, size_t, hggcStream_t stream) {
  ++mallocCalls; seenStream = stream;
  *p = reinterpret_cast<void *>(3); return mallocStatus;
}
inline hggcError_t hggcMallocFromPoolAsync(void **p, size_t, hggcMemPool_t pool, hggcStream_t stream) {
  ++poolMallocCalls; seenStream = stream;
  if (pool == nullptr) return hggcErrorInvalidValue;
  *p = reinterpret_cast<void *>(3); return mallocStatus;
}
inline hggcError_t hggcMallocAsync(void **p, size_t size, hggcMemPool_t pool, hggcStream_t stream) {
  return hggcMallocFromPoolAsync(p, size, pool, stream);
}

"""

RUNTIME_TEMPLATES = r"""
// HGGC headers also expose typed wrappers. Taking their addresses with auto
// must not be used to determine whether the SDK provides these APIs.
template <class T> inline hggcError_t hggcMalloc(T **p, size_t size) {
  return hggcMalloc(reinterpret_cast<void **>(p), size);
}
template <class T> inline hggcError_t hggcHostAlloc(T **p, size_t size, unsigned flags) {
  return hggcHostAlloc(reinterpret_cast<void **>(p), size, flags);
}
template <class T> inline hggcError_t hggcHostGetDevicePointer(T **device, void *host, unsigned flags) {
  return hggcHostGetDevicePointer(reinterpret_cast<void **>(device), host, flags);
}
"""

DRIVER = r"""
#pragma once
enum HGresult { HGGC_SUCCESS = 0, HGGC_ERROR_NOT_SUPPORTED = 80, HGGC_ERROR_UNKNOWN = 90 };
"""

PCCL = r"""
#pragma once
#include <hggc_runtime.h>
struct pcclComm;
using pcclComm_t = pcclComm *;
struct pcclUniqueId { char bytes[128]; };
enum pcclResult_t { pcclSuccess = 0, pcclUnhandledHggcError = 21,
 pcclSystemError, pcclInternalError, pcclInvalidArgument, pcclInvalidUsage,
 pcclRemoteError, pcclInProgress, pcclTimeout };
enum pcclDataType_t { pcclInt8 = 30, pcclChar = pcclInt8, pcclUint8, pcclInt32,
 pcclUint32, pcclInt64, pcclUint64, pcclFloat16, pcclFloat32, pcclFloat64, pcclBfloat16 };
enum pcclRedOp_t { pcclSum = 50, pcclProd, pcclMax, pcclMin, pcclAvg };
static pcclResult_t pcclStatus = pcclSuccess, pcclAsyncStatus = pcclSuccess;
static pcclDataType_t seenType = pcclInt8;
static pcclRedOp_t seenOp = pcclSum;
static int collectiveCalls = 0;
extern "C" inline const char *pcclGetErrorString(pcclResult_t) { return "pccl error"; }
extern "C" inline const char *pcclGetLastError(pcclComm_t) { return "last error"; }
extern "C" inline pcclResult_t pcclCommGetAsyncError(pcclComm_t, pcclResult_t *r) {
  *r = pcclAsyncStatus; return pcclStatus;
}
extern "C" inline pcclResult_t pcclAllReduce(const void *, void *, size_t, pcclDataType_t type,
                                pcclRedOp_t op, pcclComm_t, hggcStream_t) {
  seenType = type; seenOp = op; ++collectiveCalls; return pcclStatus;
}
"""

PCCL_SIGNATURES = {
    "pcclGetVersion": "int *",
    "pcclGetUniqueId": "pcclUniqueId *",
    "pcclCommInitRank": "pcclComm_t *, int, pcclUniqueId, int",
    "pcclCommFinalize": "pcclComm_t",
    "pcclCommDestroy": "pcclComm_t",
    "pcclCommAbort": "pcclComm_t",
    "pcclCommCount": "pcclComm_t, int *",
    "pcclCommCuDevice": "pcclComm_t, int *",
    "pcclCommUserRank": "pcclComm_t, int *",
    "pcclReduce": "const void *, void *, size_t, pcclDataType_t, pcclRedOp_t, int, pcclComm_t, hggcStream_t",
    "pcclBroadcast": "const void *, void *, size_t, pcclDataType_t, int, pcclComm_t, hggcStream_t",
    "pcclReduceScatter": "const void *, void *, size_t, pcclDataType_t, pcclRedOp_t, pcclComm_t, hggcStream_t",
    "pcclAllGather": "const void *, void *, size_t, pcclDataType_t, pcclComm_t, hggcStream_t",
    "pcclSend": "const void *, size_t, pcclDataType_t, int, pcclComm_t, hggcStream_t",
    "pcclRecv": "void *, size_t, pcclDataType_t, int, pcclComm_t, hggcStream_t",
    "pcclGroupStart": "",
    "pcclGroupEnd": "",
}

PROJECT = r"""
#pragma once
#include "flagcx_ccl_adaptor.h"
#include "flagcx_device_adaptor.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#define FLAGCX_INIT 1
#define INFO(flags, ...) do { std::fprintf(stderr, __VA_ARGS__); std::fputc('\n', stderr); } while (0)
#define WARN(...) INFO(0, __VA_ARGS__)
#define FLAGCXCHECK(call) do { auto r = (call); if (r != flagcxSuccess) return r; } while (0)
#define FLAGCXCHECKGOTO(call, result, label) do { result = (call); if (result != flagcxSuccess) goto label; } while (0)
template <class T> flagcxResult_t flagcxCalloc(T **p, size_t n) {
  *p = static_cast<T *>(std::calloc(n, sizeof(T)));
  return *p ? flagcxSuccess : flagcxSystemError;
}
inline bool flagcxCCLAdaptorNeedSendrecv(size_t n) { return n != 0; }
inline size_t getFlagcxDataTypeSize(flagcxDataType_t) { return 4; }
"""



@unittest.skipUnless(CXX, "C++ compiler required")
class DeviceSelectionTest(unittest.TestCase):
    def compile_selection(self, platform, adaptor, header, mocks, checks):
        # Compile the production function and error macro, with SDK calls mocked.
        production = (ROOT / f"flagcx/adaptor/device/{platform}_adaptor.cc").read_text()
        start = production.index(f"flagcxResult_t {adaptor}AdaptorSetDevice(int dev)")
        opening = production.index("{", start)
        depth = 1
        end = opening + 1
        while depth:
            depth += (production[end] == "{") - (production[end] == "}")
            end += 1
        lines = (INCLUDE / f"{header}_adaptor.h").read_text().splitlines()
        first = next(i for i, line in enumerate(lines)
                     if line.startswith("#define DEVCHECK(func)"))
        last = first
        while lines[last].endswith("\\"):
            last += 1
        source = '#include <cassert>\n#include "flagcx.h"\n'
        source += '#define WARN(...) ((void)0)\n'
        source += mocks + "\n" + "\n".join(lines[first:last + 1]) + "\n"
        source += production[start:end] + "\nint main() {\n" + checks + "\n}\n"
        with tempfile.TemporaryDirectory() as folder:
            cpp = Path(folder) / "selection.cc"
            binary = Path(folder) / "selection"
            cpp.write_text(source)
            result = subprocess.run(
                [CXX, "-std=c++11", f"-I{ROOT / 'flagcx/include'}", str(cpp),
                 "-o", str(binary)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(binary)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_device_selection(self):
        # PPU's existing contract below tests its native error conversion too.
        platforms = (
            ("cuda", "cuda", "nvidia", "cuda", "cudaSuccess"),
            ("musa", "musa", "musa", "musa", "musaSuccess"),
            ("hip", "hip", "amd", "hip", "hipSuccess"),
            ("duhip", "duhip", "du", "hip", "hipSuccess"),
            ("ixcuda", "ixcuda", "iluvatar", "cuda", "cudaSuccess"),
            ("maca", "maca", "metax", "mc", "mcSuccess"),
            ("mlu", "mlu", "cambricon", "cnrt", "cnrtSuccess"),
            ("tops", "tops", "enflame", "tops", "topsSuccess"),
            ("ptpu", "ptpu", "sunrise", "tang", "tangSuccess"),
            ("tsmicro", "tsmicro", "tsmicro", "tx", "TX_SUCCESS"),
            ("cann", "cann", "ascend", "aclrt", "ACL_SUCCESS"),
        )
        for platform, adaptor, header, native, success in platforms:
            with self.subTest(platform=platform):
                mocks = r"""
using hipError_t = int;
using tangError_t = int;
static const int SUCCESS = 0;
static int current = 2, selections = 0;
static int selectStatus = SUCCESS;
static bool contextBound = false;
static int SELECT(int device) {
  ++selections;
  if (selectStatus == SUCCESS) {
    current = device;
    contextBound = true;
  }
  return selectStatus;
}
""".replace("SUCCESS", success)
                mocks = mocks.replace("SELECT", native + "SetDevice")
                checks = r"""
// A matching ordinal does not establish this thread's native context.
assert(SET(2) == flagcxSuccess && contextBound && selections == 1);
contextBound = false;
assert(SET(2) == flagcxSuccess && contextBound && selections == 2);
assert(SET(3) == flagcxSuccess && selections == 3 && current == 3);
current = 2; // Applications may change device without going through FlagCX.
assert(SET(3) == flagcxSuccess && selections == 4 && current == 3);
selectStatus = 98;
contextBound = false;
assert(SET(3) == flagcxUnhandledDeviceError && selections == 5 && !contextBound);
assert(SET(-1) == flagcxUnhandledDeviceError && selections == 6 && current == 3);
""".replace("SET", adaptor + "AdaptorSetDevice")
                self.compile_selection(platform, adaptor, header, mocks, checks)

    def test_kunlun_device_selection_checks_both_contexts(self):
        mocks = r"""
static const int cudaSuccess = 0;
static int compatDevice = 2, nativeDevice = 2;
static int compatSets = 0, nativeSets = 0;
static int compatSetStatus = 0, nativeSetStatus = 0;
static bool compatBindsNative = false;
static int cudaSetDevice(int device) {
  ++compatSets;
  if (compatSetStatus == 0) {
    compatDevice = device;
    if (compatBindsNative) nativeDevice = device;
  }
  return compatSetStatus;
}
static int xpu_set_device(int device) {
  ++nativeSets;
  if (nativeSetStatus == 0) nativeDevice = device;
  return nativeSetStatus;
}
"""
        checks = r"""
assert(kunlunAdaptorSetDevice(2) == flagcxSuccess);
assert(compatSets == 1 && nativeSets == 1);
nativeDevice = 1;
assert(kunlunAdaptorSetDevice(2) == flagcxSuccess && compatSets == 2 && nativeSets == 2);
assert(nativeDevice == 2);
compatDevice = 1;
assert(kunlunAdaptorSetDevice(2) == flagcxSuccess && compatSets == 3 && nativeSets == 3);
assert(compatDevice == 2);
assert(kunlunAdaptorSetDevice(3) == flagcxSuccess && compatSets == 4 && nativeSets == 4);
compatBindsNative = true;
assert(kunlunAdaptorSetDevice(4) == flagcxSuccess && compatSets == 5 && nativeSets == 5);
compatSetStatus = 98;
assert(kunlunAdaptorSetDevice(4) == flagcxUnhandledDeviceError);
assert(compatSets == 6 && nativeSets == 5);
compatSetStatus = 0;
compatBindsNative = false;
nativeDevice = 1;
nativeSetStatus = 98;
assert(kunlunAdaptorSetDevice(4) == flagcxUnhandledDeviceError && nativeSets == 6);
assert(compatDevice == 4 && nativeDevice == 1);
nativeSetStatus = 0;
assert(kunlunAdaptorSetDevice(4) == flagcxSuccess && nativeSets == 7 && nativeDevice == 4);
"""
        self.compile_selection("kunlunxin", "kunlun", "kunlunxin", mocks, checks)


@unittest.skipUnless(CXX, "C++ compiler required")
class SdkCompileTest(unittest.TestCase):
    def test_mpi_predefined_handles_accept_pointer_and_integer_sdks(self):
        production = (INCLUDE / "mpi_adaptor.h").read_text()
        for pointer in (False, True):
            with self.subTest(pointer=pointer), tempfile.TemporaryDirectory() as folder:
                tmp = Path(folder)
                names = ["CHAR", "UNSIGNED_CHAR", "INT", "UNSIGNED", "LONG_LONG",
                         "UNSIGNED_LONG_LONG", "FLOAT", "DOUBLE", "UINT16_T",
                         "DATATYPE_NULL", "SUM", "PROD", "MAX", "MIN", "OP_NULL"]
                source = '#include <cassert>\n#include <vector>\n#include "flagcx.h"\n'
                source += 'using MPI_Comm = int;\n'
                source += ('using MPI_Datatype = void *; using MPI_Op = void *;\n'
                           if pointer else 'using MPI_Datatype = int; using MPI_Op = int;\n')
                for i, name in enumerate(names):
                    source += (f'static int storage_{name}; static void *MPI_{name} = &storage_{name};\n'
                               if pointer else f'static int MPI_{name} = {i};\n')
                source += 'enum { MPI_ERR_TYPE = 17 };\n'
                for name in ['Allreduce', 'Reduce', 'Bcast', 'Gather', 'Scatter',
                             'Allgather', 'Alltoall', 'Comm_size', 'Reduce_scatter']:
                    source += f'static int MPI_{name}(...) {{ return 0; }}\n'
                source += '\n'.join(production_function(production, name) for name in
                                    ['getFlagcxToMpiDataType', 'getFlagcxToMpiOp', 'callMpi'])
                source += '\nint main() { assert(getFlagcxToMpiDataType(flagcxFloat) == MPI_FLOAT);'
                source += ' assert(getFlagcxToMpiOp(flagcxSum) == MPI_SUM);'
                source += ' assert(getFlagcxToMpiDataType(flagcxBfloat16) == MPI_DATATYPE_NULL); }\n'
                (tmp / 'mpi.cc').write_text(source)
                result = subprocess.run([CXX, '-std=c++11', f'-I{ROOT / "flagcx/include"}',
                                         str(tmp / 'mpi.cc'), '-o', str(tmp / 'test')],
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                result = subprocess.run([str(tmp / 'test')], text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_devcomm_backend_preserves_failed_owner(self):
        for vendor in ('nccl', 'eccl'):
            with self.subTest(vendor=vendor), tempfile.TemporaryDirectory() as folder:
                tmp = Path(folder)
                ccl = (ROOT / f'flagcx/adaptor/ccl/{vendor}_adaptor.cc').read_text()
                backend = (ROOT / f'flagcx/adaptor/device_api/{vendor}_dev_api_backend.cc').read_text()
                source = r'''
#include "flagcx.h"
#include <cstdlib>
#include <cassert>
struct flagcxInnerComm { int unused; };
using flagcxInnerComm_t = flagcxInnerComm *;
struct flagcxInnerDevComm { int base; };
using flagcxInnerDevComm_t = flagcxInnerDevComm *;
struct flagcxComm { flagcxInnerComm_t homoComm; };
struct flagcxDevComm { flagcxInnerDevComm_t devComm; };
using flagcxDevComm_t = flagcxDevComm *;
static int status, frees, closes;
static void releaseWrapper(void *p) { ++frees; std::free(p); }
#define NCCL_VERSION(a,b,c) ((a)*10000+(b)*100+(c))
#define NCCL_VERSION_CODE 22900
#define free releaseWrapper
'''
                if vendor == 'nccl':
                    # Actual native callback also uses comm->base.
                    source = source.replace('struct flagcxInnerComm { int unused; };',
                                            'struct flagcxInnerComm { void *base; };')
                    source += 'static flagcxResult_t ncclDevCommDestroyHelper(void *, int *) { ++closes; return (flagcxResult_t)status; }\n'
                else:
                    source += 'static flagcxResult_t flagcxEcclDevCommDestroy(flagcxInnerComm *, flagcxInnerDevComm *) { ++closes; return (flagcxResult_t)status; }\n'
                source += production_function(ccl, vendor + 'AdaptorDevCommDestroy') + '\n#undef free\n'
                source += 'struct Ccl { flagcxResult_t (*devCommDestroy)(flagcxInnerComm_t, flagcxInnerDevComm_t); };\n'
                source += f'static Ccl sdk = {{{vendor}AdaptorDevCommDestroy}}; static Ccl *cclAdaptors[] = {{nullptr, &sdk}};\n'
                source += '#define flagcxCCLAdaptorDevice 1\n'
                source += production_function(backend, vendor + 'DevApiCommDestroy')
                source += '\nint main() { flagcxInnerComm inner{}; flagcxComm comm{&inner};\n'
                source += 'auto *p = (flagcxInnerDevComm_t)std::calloc(1, sizeof(flagcxInnerDevComm)); p->base = 42; flagcxDevComm dev{p};\n'
                source += 'for (int r : {flagcxInProgress, flagcxRemoteError}) { status = r;\n'
                source += f'assert({vendor}DevApiCommDestroy(&comm, &dev) == r);\n'
                source += 'assert(dev.devComm == p && p->base == 42 && frees == 0); }\nstatus = 0;\n'
                source += f'assert({vendor}DevApiCommDestroy(&comm, &dev) == flagcxSuccess);\n'
                source += f'assert(dev.devComm == nullptr && frees == 1 && closes == 3); assert({vendor}DevApiCommDestroy(&comm, &dev) == flagcxSuccess && closes == 3); }}\n'
                source = '#include <initializer_list>\n' + source
                (tmp / 'devcomm.cc').write_text(source)
                result = subprocess.run([CXX, '-std=c++11', f'-I{ROOT / "flagcx/include"}',
                                         str(tmp / 'devcomm.cc'), '-o', str(tmp / 'test')],
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                result = subprocess.run([str(tmp / 'test')], text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_eccl_runtime_only_and_full_sdk_use_actual_adaptor(self):
        for full in (False, True):
            with self.subTest(full=full), tempfile.TemporaryDirectory() as folder:
                tmp = Path(folder)
                (tmp / "tops").mkdir()
                (tmp / "tops/tops_runtime.h").write_text(r"""
#pragma once
#include <cstdlib>
using topsDeviceProp_t = int;
using topsStream_t = void *;
using topsEvent_t = void *;
struct topsIpcMemHandle_t { char bytes[64]; };
enum { topsSuccess = 0 };
static int topsAllocs, topsFrees;
static int topsMalloc(void **p, size_t n) { ++topsAllocs; *p = malloc(n); return 0; }
static int topsFree(void *p) { ++topsFrees; free(p); return 0; }
""")
                (tmp / "eccl.h").write_text(r"""
#pragma once
#include <cstdlib>
using ecclComm_t = void *;
using ecclResult_t = int;
using ecclDataType_t = int;
using ecclRedOp_t = int;
struct ecclUniqueId { char bytes[128]; };
enum { ecclInvalidUsage = 5 };
static int ecclGetVersion(...) { return 0; }
static int ecclGetUniqueId(...) { return 0; }
static const char *ecclGetErrorString(...) { return "stub"; }
static const char *ecclGetLastError(...) { return "stub"; }
static int ecclCommInitRank(...) { return 0; }
static int ecclCommDestroy(...) { return 0; }
static int ecclCommAbort(...) { return 0; }
static int ecclCommCount(...) { return 0; }
static int ecclCommDevice(...) { return 0; }
static int ecclCommUserRank(...) { return 0; }
static int ecclCommGetAsyncError(...) { return 0; }
static int ecclReduce(...) { return 0; }
static int ecclGather(...) { return 0; }
static int ecclScatter(...) { return 0; }
static int ecclBroadcast(...) { return 0; }
static int ecclAllReduce(...) { return 0; }
static int ecclReduceScatter(...) { return 0; }
static int ecclAllGather(...) { return 0; }
static int ecclAlltoall(...) { return 0; }
static int ecclAlltoAllv(...) { return 0; }
static int ecclSend(...) { return 0; }
static int ecclRecv(...) { return 0; }
static int ecclGroupStart(...) { return 0; }
static int ecclGroupEnd(...) { return 0; }
""")
                if full:
                    (tmp / "eccl_device.h").write_text(r"""
#pragma once
struct ecclDevComm_t { int value; };
using ecclWindow_t = void *;
struct ecclCommProperties_t { int deviceApiSupport, multimemSupport; };
#define ECCL_COMM_PROPERTIES_INITIALIZER {1, 1}
struct ecclDevCommRequirements_t {
 int barrierCount, lsaBarrierCount, lsaLLA2ABlockCount, lsaLLA2ASlotCount;
 int ginContextCount, railGinBarrierCount, ginCounterCount, ginConnectionType;
};
#define ECCL_DEV_COMM_REQUIREMENTS_INITIALIZER {}
enum { ECCL_GIN_CONNECTION_FULL = 7 };
static int nativeAllocs, nativeFrees, closes, closeResult;
static ecclDevCommRequirements_t seenReqs;
static int ecclMemAlloc(void **p, size_t n) { ++nativeAllocs; *p = malloc(n); return 0; }
static int ecclMemFree(void *p) { ++nativeFrees; free(p); return 0; }
static int ecclCommWindowRegister(void *, void *, size_t, void **p, int) { *p = (void *)42; return 0; }
static int ecclCommWindowDeregister(void *, void *) { ++closes; return closeResult; }
static int ecclCommQueryProperties(void *, ecclCommProperties_t *) { return 0; }
static int ecclDevCommCreate(void *, const ecclDevCommRequirements_t *r, ecclDevComm_t *p) { seenReqs = *r; p->value = 42; return 0; }
static int ecclDevCommDestroy(void *, ecclDevComm_t *) { ++closes; return closeResult; }
""")
                (tmp / "adaptor.h").write_text(PROJECT)
                (tmp / "alloc.h").write_text('#include "adaptor.h"\n')
                (tmp / "comm.h").write_text('#define FLAGCX_DEVICE_CTA_COUNT 8\n')
                cpp = tmp / "eccl.cc"
                cpp.write_text('#include <cassert>\n#include "'
                               + str(ROOT / 'flagcx/adaptor/ccl/eccl_adaptor.cc') + '"\n'
                               + r"""
int main() {
 assert(ecclAdaptor.devCommCreate != nullptr);
 assert(ecclAdaptor.devCommDestroy != nullptr);
 void *p = nullptr;
 assert(ecclAdaptorMemAlloc(&p, 64) == flagcxSuccess && p);
 assert(ecclAdaptorMemFree(p) == flagcxSuccess);
 flagcxInnerComm comm{}; flagcxDevCommRequirements reqs{};
 flagcxInnerDevComm_t device = nullptr;
 flagcxInnerWindow_t win = nullptr;
#ifdef FLAGCX_ENFLAME_DEVICE_API
 assert(nativeAllocs == 1 && nativeFrees == 1 && topsAllocs == 0);
 assert(ecclAdaptorDevCommReqsInit(&comm, &reqs) == flagcxSuccess);
 reqs.interContextCount = 2; reqs.interBarrierCount = 3; reqs.interCounterCount = 4;
 assert(ecclAdaptorDevCommCreate(&comm, &reqs, &device) == flagcxSuccess);
 assert(seenReqs.ginContextCount == 2 && seenReqs.railGinBarrierCount == 3 && seenReqs.ginCounterCount == 4);
 closeResult = flagcxRemoteError;
 assert(ecclAdaptorDevCommDestroy(&comm, device) == flagcxRemoteError);
 assert(device->base.value == 42);
 closeResult = 0;
 assert(ecclAdaptorDevCommDestroy(&comm, device) == flagcxSuccess);
 assert(ecclAdaptorCommWindowRegister(&comm, (void *)1, 64, &win, 0) == flagcxSuccess);
 closeResult = flagcxRemoteError;
 assert(ecclAdaptorCommWindowDeregister(&comm, win) == flagcxRemoteError);
 assert(win->base == (void *)42);
 closeResult = 0;
 assert(ecclAdaptorCommWindowDeregister(&comm, win) == flagcxSuccess);
#else
 assert(topsAllocs == 1 && topsFrees == 1);
 assert(ecclAdaptorDevCommReqsInit(&comm, &reqs) == flagcxNotSupported);
 assert(ecclAdaptorDevCommCreate(&comm, &reqs, &device) == flagcxNotSupported && device == nullptr);
 assert(ecclAdaptorCommWindowRegister(&comm, (void *)1, 64, &win, 0) == flagcxNotSupported && win == nullptr);
#endif
}
""")
                cmd = [CXX, '-std=c++17', '-DUSE_ENFLAME_ADAPTOR',
                       f'-I{tmp}', f'-I{INCLUDE}', f'-I{ROOT / "flagcx/include"}',
                       str(cpp), '-o', str(tmp / 'test')]
                if full:
                    cmd.insert(1, '-DFLAGCX_ENFLAME_DEVICE_API')
                result = subprocess.run(cmd, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                result = subprocess.run([str(tmp / 'test')], text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(CXX, "C++ compiler required")
class PpuAdaptorTest(unittest.TestCase):
    def compile_and_run(self, source, missing_pccl_symbol=None, fail_open=False,
                        runtime_env=None):
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            (tmp / "hggc_runtime.h").write_text(RUNTIME + RUNTIME_TEMPLATES)
            (tmp / "hggc.h").write_text(DRIVER)
            pccl = PCCL + "\n".join(
                f'extern "C" inline pcclResult_t {name}({args}) {{ return pcclStatus; }}'
                for name, args in PCCL_SIGNATURES.items()
            )
            # Keep native symbol names visible to dlsym in this host fixture.
            names = list(PCCL_SIGNATURES) + ["pcclGetErrorString", "pcclGetLastError",
                                              "pcclCommGetAsyncError", "pcclAllReduce"]
            pccl += "\n" + "\n".join(
                f"auto volatile fixture_{name} = &{name};"
                for name in names if name != missing_pccl_symbol
            )
            (tmp / "pccl.h").write_text(pccl)
            (tmp / "adaptor.h").write_text(PROJECT)
            (tmp / "alloc.h").write_text('#include "adaptor.h"\n')
            for name in ("comm.h", "param.h"):
                (tmp / name).write_text("")
            cpp = tmp / "test.cc"
            cpp.write_text(r"""
#include <cassert>
#include <cstdlib>
#include <dlfcn.h>
static int openLibraryCalls = 0;
static int seenLoadFlags = 0;
static int kernelEnabledAtOpen = -1;
void *flagcxOpenLib(const char *path, int flags,
                   void (*error_handler)(const char *, int, const char *)) {
  ++openLibraryCalls;
  seenLoadFlags = flags;
  const char *enabled = std::getenv("PCCL_ENABLE_EXT_KERNEL");
  if (enabled == nullptr) enabled = std::getenv("NCCL_ENABLE_EXT_KERNEL");
  kernelEnabledAtOpen = enabled == nullptr ? 1 : std::atoi(enabled);
  if (""" + str(int(fail_open)) + r""") {
    error_handler(path, 0, "fixture library missing");
    return nullptr;
  }
  return dlopen(nullptr, RTLD_NOW);
}
""" + source)
            command = [CXX, "-std=c++11", "-DUSE_PPU_ADAPTOR",
                       f"-I{tmp}", f"-I{INCLUDE}",
                       f"-I{ROOT / 'flagcx/include'}", str(cpp),
                       f"-I{ROOT / 'flagcx/service/include'}", "-rdynamic",
                       "-o", str(tmp / "test")]
            if os.sys.platform == "linux":
                command += ["-ldl", "-pthread"]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            env = {key: value for key, value in os.environ.items()
                   if key not in ("PCCL_ENABLE_EXT_KERNEL", "NCCL_ENABLE_EXT_KERNEL")}
            env.update(runtime_env or {})
            result = subprocess.run([str(tmp / "test")], env=env,
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_pccl_defaults_to_builtin_kernels_before_loading(self):
        source = '#include "' + str(ROOT / 'flagcx/adaptor/ccl/ppu_pccl_adaptor.cc') + '"\n'
        settings = [
            ({}, 0),
            ({"PCCL_ENABLE_EXT_KERNEL": ""}, 0),
            ({"NCCL_ENABLE_EXT_KERNEL": ""}, 0),
            ({"PCCL_ENABLE_EXT_KERNEL": "1"}, 1),
            ({"NCCL_ENABLE_EXT_KERNEL": "1"}, 1),
            ({"PCCL_ENABLE_EXT_KERNEL": "0", "NCCL_ENABLE_EXT_KERNEL": "1"}, 0),
        ]
        for env, expected in settings:
            with self.subTest(settings=env):
                self.compile_and_run(source + f"""
int main() {{
  assert(ppupcclAdaptorGetVersion(nullptr) == flagcxSuccess);
  assert(kernelEnabledAtOpen == {expected});
}}
""", runtime_env=env)

    def test_pccl_enum_mapping_and_async_error_output(self):
        source = '#include "' + str(ROOT / 'flagcx/adaptor/ccl/ppu_pccl_adaptor.cc') + '"\n'
        self.compile_and_run(source + r"""
int main() {
  flagcxInnerComm comm{};
  flagcxStream stream{};
  assert(ppupcclAdaptorAllReduce(nullptr, nullptr, 1, flagcxFloat32,
                               flagcxMax, &comm, &stream) == flagcxSuccess);
  assert(seenType == pcclFloat32 && seenOp == pcclMax);
  assert(ppupcclAdaptorAllReduce(nullptr, nullptr, 1, flagcxNumTypes,
                               flagcxMax, &comm, &stream) == flagcxInvalidArgument);
  assert(collectiveCalls == 1);
  assert(ppupcclAdaptorAllReduce(nullptr, nullptr, 1, flagcxFloat32,
                               flagcxNumRedOps, &comm, &stream) == flagcxInvalidArgument);
  assert(collectiveCalls == 1);
  pcclStatus = pcclTimeout;
  assert(ppupcclAdaptorGetVersion(nullptr) == flagcxUnhandledCCLError);
  pcclStatus = pcclUnhandledHggcError;
  assert(ppupcclAdaptorGetVersion(nullptr) == flagcxUnhandledDeviceError);
  flagcxResult_t async = flagcxRemoteError;
  assert(ppupcclAdaptorCommGetAsyncError(&comm, &async) == flagcxUnhandledDeviceError);
  assert(async == flagcxRemoteError);
  pcclStatus = pcclSuccess;
  pcclAsyncStatus = pcclTimeout;
  assert(ppupcclAdaptorCommGetAsyncError(&comm, &async) == flagcxSuccess);
  assert(async == flagcxUnhandledCCLError);
  assert(ppupcclAdaptorCommGetAsyncError(&comm, nullptr) == flagcxInvalidArgument);
  assert(openLibraryCalls == 1);
  assert((seenLoadFlags & RTLD_NOW) != 0);
  assert((seenLoadFlags & RTLD_GLOBAL) == 0);
#ifdef RTLD_DEEPBIND
  assert((seenLoadFlags & RTLD_DEEPBIND) != 0);
#endif
}
""")

    def test_pccl_loader_reports_missing_library_and_symbol(self):
        source = '#include "' + str(ROOT / 'flagcx/adaptor/ccl/ppu_pccl_adaptor.cc') + '"\n'
        source += r"""
int main() {
  int version = -1;
  assert(ppupcclAdaptorGetVersion(&version) == flagcxSystemError);
  assert(version == -1);
  flagcxInnerComm comm{};
  flagcxStream stream{};
  assert(ppupcclAdaptorAllReduce(nullptr, nullptr, 1, flagcxFloat32,
                               flagcxSum, &comm, &stream) == flagcxSystemError);
  assert(collectiveCalls == 0);
  assert(ppupcclAdaptorGetErrorString(flagcxSystemError) != nullptr);
}
"""
        with self.subTest(failure="library"):
            self.compile_and_run(source, fail_open=True)
        with self.subTest(failure="symbol"):
            self.compile_and_run(source, missing_pccl_symbol="pcclGroupEnd")

    def test_pccl_loader_initializes_once_across_threads(self):
        source = '#include "' + str(ROOT / 'flagcx/adaptor/ccl/ppu_pccl_adaptor.cc') + '"\n'
        self.compile_and_run(source + r"""
#include <thread>
#include <vector>
int main() {
  std::vector<std::thread> threads;
  for (int i = 0; i < 16; ++i)
    threads.emplace_back([] {
      int version = 0;
      assert(ppupcclAdaptorGetVersion(&version) == flagcxSuccess);
    });
  for (auto &thread : threads) thread.join();
  assert(openLibraryCalls == 1);
}
""")

    @unittest.skipUnless(os.sys.platform == "linux", "ELF/glibc loading contract")
    def test_pccl_and_u2mm_symbol_scopes_in_both_load_orders(self):
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            (tmp / "hggc_runtime.h").write_text(RUNTIME + RUNTIME_TEMPLATES)
            (tmp / "hggc.h").write_text(DRIVER)
            pccl = PCCL + "\n".join(
                f'extern "C" inline pcclResult_t {name}({args}) {{ return pcclStatus; }}'
                for name, args in PCCL_SIGNATURES.items()
            )
            (tmp / "pccl.h").write_text(pccl)
            (tmp / "adaptor.h").write_text(PROJECT)
            (tmp / "alloc.h").write_text('#include "adaptor.h"\n')
            (tmp / "comm.h").write_text("")
            names = list(PCCL_SIGNATURES) + ["pcclGetErrorString", "pcclGetLastError",
                                              "pcclCommGetAsyncError", "pcclAllReduce"]
            (tmp / "pccl.cc").write_text('#include "pccl.h"\n' + "\n".join(
                f"auto volatile fixture_{name} = &{name};" for name in names
            ) + r"""
extern "C" {
// Reproduce the vendor data symbol that collides with a real U2MM function.
int (*u2mm_get_default_alloc_config)();
static int private_config() { return 17; }
int check_private_binding() {
  u2mm_get_default_alloc_config = private_config;
  return u2mm_get_default_alloc_config();
}
}
""")
            (tmp / "u2mm.cc").write_text(
                'extern "C" int u2mm_get_default_alloc_config() { return 42; }\n')
            (tmp / "barex.cc").write_text(r"""
extern "C" int u2mm_get_default_alloc_config();
extern "C" int register_host_mr() {
  return u2mm_get_default_alloc_config();
}
""")
            utils = (ROOT / "flagcx/service/utils.cc").read_text()
            begin = utils.index("void *flagcxOpenLib(")
            end = utils.index("\n}\n", begin) + 3
            main = r"""
#include <cassert>
#include <cerrno>
#include <cstring>
#include <dlfcn.h>
""" + utils[begin:end]
            main += '#include "' + str(ROOT / 'flagcx/adaptor/ccl/ppu_pccl_adaptor.cc') + '"\n'
            main += r"""
int main(int argc, char **argv) {
  assert(argc == 2);
  void *barex = nullptr;
  if (strcmp(argv[1], "barex-first") == 0) {
    barex = dlopen("libbarex_fixture.so", RTLD_NOW | RTLD_GLOBAL);
    assert(barex != nullptr);
  }
  int version = 0;
  assert(ppupcclAdaptorGetVersion(&version) == flagcxSuccess);
  // A second dlopen must not promote the locally opened PCCL to global.
  void *pccl = dlopen("libpccl.so.2", RTLD_NOW | RTLD_NOLOAD);
  assert(pccl != nullptr);
  auto check = reinterpret_cast<int (*)()>(dlsym(pccl, "check_private_binding"));
  assert(check != nullptr && check() == 17);
  if (barex == nullptr) {
    assert(dlsym(RTLD_DEFAULT, "u2mm_get_default_alloc_config") == nullptr);
    barex = dlopen("libbarex_fixture.so", RTLD_NOW | RTLD_GLOBAL);
    assert(barex != nullptr);
  }
  auto reg = reinterpret_cast<int (*)()>(dlsym(barex, "register_host_mr"));
  assert(reg != nullptr && reg() == 42);
  assert(dlsym(RTLD_DEFAULT, "u2mm_get_default_alloc_config") !=
         dlsym(pccl, "u2mm_get_default_alloc_config"));
}
"""
            (tmp / "main.cc").write_text(main)
            common = [CXX, "-std=c++11", "-DUSE_PPU_ADAPTOR", f"-I{tmp}",
                      f"-I{INCLUDE}", f"-I{ROOT / 'flagcx/include'}",
                      f"-I{ROOT / 'flagcx/service/include'}"]
            commands = [
                common + ["-shared", "-fPIC", str(tmp / "pccl.cc"),
                          "-o", str(tmp / "libpccl.so.2")],
                common + ["-shared", "-fPIC", str(tmp / "u2mm.cc"),
                          "-o", str(tmp / "libu2mm.so")],
                common + ["-shared", "-fPIC", str(tmp / "barex.cc"),
                          f"-L{tmp}", "-lu2mm", f"-Wl,-rpath,{tmp}",
                          "-o", str(tmp / "libbarex_fixture.so")],
                common + [str(tmp / "main.cc"), "-ldl", "-pthread",
                          "-o", str(tmp / "test")],
            ]
            for command in commands:
                result = subprocess.run(command, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            env = dict(os.environ, LD_LIBRARY_PATH=str(tmp))
            for order in ("barex-first", "pccl-first"):
                with self.subTest(order=order):
                    result = subprocess.run([str(tmp / "test"), order], env=env,
                                            text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(os.sys.platform == "linux", "ELF/glibc loading contract")
    def test_pccl_nested_plugin_registers_with_the_query_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            (tmp / "hggc_runtime.h").write_text(RUNTIME + RUNTIME_TEMPLATES)
            (tmp / "hggc.h").write_text(DRIVER)
            pccl = PCCL + "\n".join(
                f'extern "C" inline pcclResult_t {name}({args}) {{ return pcclStatus; }}'
                for name, args in PCCL_SIGNATURES.items()
                if name != "pcclCommInitRank"
            )
            pccl += '\nextern "C" pcclResult_t pcclCommInitRank(pcclComm_t *, int, pcclUniqueId, int);\n'
            (tmp / "pccl.h").write_text(pccl)
            (tmp / "adaptor.h").write_text(PROJECT)
            (tmp / "alloc.h").write_text('#include "adaptor.h"\n')
            (tmp / "comm.h").write_text("")
            (tmp / "runtime.cc").write_text(r"""
extern "C" int __hggcRegisterFunction() { return RUNTIME_VERSION; }
extern "C" int hggcFuncSetAttribute() { return RUNTIME_VERSION; }
extern "C" int fixtureLegacyRegister() { return RUNTIME_VERSION; }
""")
            # Both SDK runtimes use this symbol version. The process-global
            # native wrapper can still interpose their registration exports.
            (tmp / "runtime.map").write_text("libhggcrt1.so { global: *; };\n")
            (tmp / "wrapper.cc").write_text(r"""
extern "C" int fixtureLegacyRegister();
extern "C" int __hggcRegisterFunction() { return fixtureLegacyRegister(); }
extern "C" int fixtureWrapperLoaded() { return 1; }
""")
            names = list(PCCL_SIGNATURES) + [
                "pcclGetErrorString", "pcclGetLastError",
                "pcclCommGetAsyncError", "pcclAllReduce"]
            (tmp / "pccl.cc").write_text(
                '#include "pccl.h"\n#include <cstdlib>\n#include <cstdio>\n#include <dlfcn.h>\n'
                + "\n".join(f"auto volatile fixture_{name} = &{name};"
                            for name in names) + r"""
extern "C" int hggcFuncSetAttribute();
static const char *setting(const char *name, const char *compat) {
  const char *value = getenv(name);
  return value != nullptr ? value : getenv(compat);
}
extern "C" pcclResult_t pcclCommInitRank(pcclComm_t *comm, int, pcclUniqueId, int) {
  *comm = reinterpret_cast<pcclComm_t>(1);
  const char *enabled = setting("PCCL_ENABLE_EXT_KERNEL", "NCCL_ENABLE_EXT_KERNEL");
  if (enabled != nullptr && atoi(enabled) == 0) return pcclSuccess;
  if (setting("PCCL_ALGO", "NCCL_ALGO") != nullptr ||
      setting("PCCL_PROTO", "NCCL_PROTO") != nullptr) return pcclSuccess;
  const char *name = setting("PCCL_EXT_KERNEL_PLUGIN", "NCCL_EXT_KERNEL_PLUGIN");
  if (name == nullptr || *name == '\0') name = "libpccl-ext-kernel.so";
  // Match the vendor's second-level dlopen, without DEEPBIND.
  void *plugin = dlopen(name, RTLD_NOW | RTLD_LOCAL);
  if (plugin == nullptr) return pcclSuccess; // optional plugin
  auto registered = reinterpret_cast<int (*)()>(dlsym(plugin, "fixtureRegisteredRuntime"));
  if (registered == nullptr) return pcclInternalError;
  const int registration = registered();
  const int query = hggcFuncSetAttribute();
  fprintf(stderr, "registered=%d queried=%d\n", registration, query);
  dlclose(plugin);
  return registration == query ? pcclSuccess : pcclInternalError;
}
""")
            (tmp / "plugin.cc").write_text(r"""
extern "C" int __hggcRegisterFunction();
static int registeredRuntime;
__attribute__((constructor)) static void registerKernel() {
  registeredRuntime = __hggcRegisterFunction();
}
extern "C" int fixtureRegisteredRuntime() { return registeredRuntime; }
""")
            utils = (ROOT / "flagcx/service/utils.cc").read_text()
            begin = utils.index("void *flagcxOpenLib(")
            end = utils.index("\n}\n", begin) + 3
            main = '#include <cassert>\n#include <cerrno>\n#include <cstring>\n#include <dlfcn.h>\n'
            main += utils[begin:end]
            main += '#include "' + str(ROOT / 'flagcx/adaptor/ccl/ppu_pccl_adaptor.cc') + '"\n'
            main += r"""
extern "C" int fixtureWrapperLoaded();
int main(int argc, char **argv) {
  assert(argc == 2 && fixtureWrapperLoaded() == 1);
  pcclUniqueId id{};
  if (strcmp(argv[1], "ordinary") == 0) {
    void *core = dlopen("libpccl.so.2", RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    assert(core != nullptr);
    auto init = reinterpret_cast<decltype(&pcclCommInitRank)>(dlsym(core, "pcclCommInitRank"));
    pcclComm_t comm = nullptr;
    // Reproduce the split before applying the adaptor's plugin preload.
    assert(init(&comm, 4, id, 0) == pcclInternalError);
    return 0;
  }
  if (strcmp(argv[1], "prebound") == 0) {
    assert(dlopen("libpccl.so.2", RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND) != nullptr);
    assert(dlopen("libpccl-ext-kernel.so", RTLD_NOW | RTLD_LOCAL) != nullptr);
    flagcxInnerComm_t comm = nullptr;
    // An earlier ordinary load cannot be rebound. Preserve the native error.
    assert(ppupcclAdaptorCommInitRank(&comm, 4,
        reinterpret_cast<flagcxInnerUniqueId_t>(&id), 0, nullptr) == flagcxInternalError);
    free(comm);
    return 0;
  }
  for (int i = 0; i < 3; ++i) {
    flagcxInnerComm_t comm = nullptr;
    assert(ppupcclAdaptorCommInitRank(&comm, 4,
        reinterpret_cast<flagcxInnerUniqueId_t>(&id), 0, nullptr) == flagcxSuccess);
    assert(ppupcclAdaptorCommDestroy(comm) == flagcxSuccess);
    void *plugin = dlopen("libpccl-ext-kernel.so", RTLD_NOW | RTLD_NOLOAD);
    if (strcmp(argv[1], "optional") == 0) {
      assert(plugin == nullptr);
    } else {
      // Our lifetime reference outlives the vendor's per-communicator one.
      assert(plugin != nullptr);
      auto registered = reinterpret_cast<int (*)()>(dlsym(plugin, "fixtureRegisteredRuntime"));
      assert(registered != nullptr && registered() == 13);
      dlclose(plugin);
    }
  }
}
"""
            (tmp / "main.cc").write_text(main)
            common = [CXX, "-std=c++11", "-DUSE_PPU_ADAPTOR", f"-I{tmp}",
                      f"-I{INCLUDE}", f"-I{ROOT / 'flagcx/include'}",
                      f"-I{ROOT / 'flagcx/service/include'}"]
            shared = ["-shared", "-fPIC", f"-L{tmp}", f"-Wl,-rpath,{tmp}"]
            commands = [
                common + shared + [f"-DRUNTIME_VERSION={version}",
                    str(tmp / "runtime.cc"),
                    f"-Wl,--version-script={tmp / 'runtime.map'}",
                    f"-Wl,-soname,libhggcrt.{version}.so",
                    "-o", str(tmp / f"libhggcrt.{version}.so")]
                for version in (12, 13)
            ]
            commands += [
                common + shared + [str(tmp / "wrapper.cc"), "-l:libhggcrt.12.so",
                    "-o", str(tmp / "libwrapper.so")],
                common + shared + [str(tmp / "pccl.cc"), "-l:libhggcrt.13.so",
                    "-ldl", "-Wl,-soname,libpccl.so.2",
                    "-o", str(tmp / "libpccl.so.2")],
                common + shared + [str(tmp / "plugin.cc"), "-Wl,--no-as-needed",
                    "-l:libpccl.so.2", "-l:libhggcrt.13.so",
                    "-Wl,-soname,libpccl-ext-kernel.so",
                    "-o", str(tmp / "libpccl-ext-kernel.so")],
                common + [str(tmp / "main.cc"), f"-L{tmp}", "-lwrapper",
                    f"-Wl,-rpath,{tmp}", "-ldl", "-pthread",
                    "-o", str(tmp / "test")],
            ]
            for command in commands:
                result = subprocess.run(command, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            # Each mode needs a fresh process: existing ELF relocations cannot
            # be repaired by reopening an already bound library.
            cases = [
                ("ordinary", {}),
                ("prebound", {"PCCL_ENABLE_EXT_KERNEL": "1"}),
                ("optional", {}),
                ("fixed", {"PCCL_ENABLE_EXT_KERNEL": "1"}),
                ("fixed", {"NCCL_ENABLE_EXT_KERNEL": "1"}),
                ("fixed", {"PCCL_ENABLE_EXT_KERNEL": "1",
                           "PCCL_EXT_KERNEL_PLUGIN": str(tmp / "libpccl-ext-kernel.so")}),
                ("fixed", {"PCCL_ENABLE_EXT_KERNEL": "1",
                           "NCCL_EXT_KERNEL_PLUGIN": str(tmp / "libpccl-ext-kernel.so")}),
                ("fixed", {"PCCL_ENABLE_EXT_KERNEL": "1",
                           "PCCL_EXT_KERNEL_PLUGIN": str(tmp / "libpccl-ext-kernel.so"),
                           "NCCL_EXT_KERNEL_PLUGIN": str(tmp / "missing.so")}),
                ("optional", {"PCCL_ENABLE_EXT_KERNEL": "0"}),
                ("optional", {"PCCL_ENABLE_EXT_KERNEL": "0", "NCCL_ENABLE_EXT_KERNEL": "1"}),
                ("optional", {"NCCL_ENABLE_EXT_KERNEL": "0"}),
                ("optional", {"PCCL_ENABLE_EXT_KERNEL": "1",
                              "PCCL_EXT_KERNEL_PLUGIN": str(tmp / "missing.so")}),
                ("optional", {"PCCL_ENABLE_EXT_KERNEL": "1",
                              "PCCL_EXT_KERNEL_PLUGIN": "STATIC_PLUGIN"}),
                ("optional", {"PCCL_ENABLE_EXT_KERNEL": "1", "PCCL_ALGO": "Ring"}),
                ("optional", {"PCCL_ENABLE_EXT_KERNEL": "1", "NCCL_PROTO": "Simple"}),
            ]
            for mode, settings in cases:
                with self.subTest(mode=mode, settings=settings):
                    env = {key: value for key, value in os.environ.items()
                           if key not in ("PCCL_EXT_KERNEL_PLUGIN", "NCCL_EXT_KERNEL_PLUGIN",
                                          "PCCL_ENABLE_EXT_KERNEL", "NCCL_ENABLE_EXT_KERNEL",
                                          "PCCL_ALGO", "NCCL_ALGO", "PCCL_PROTO", "NCCL_PROTO")}
                    env.update(LD_LIBRARY_PATH=str(tmp), **settings)
                    result = subprocess.run([str(tmp / "test"), mode], env=env,
                                            text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    if mode == "ordinary":
                        self.assertIn("registered=12 queried=13", result.stderr)
                    elif mode == "prebound":
                        self.assertIn("was already loaded", result.stderr)
                        self.assertIn("registered=12 queried=13", result.stderr)
                    elif mode == "fixed":
                        self.assertEqual(result.stderr.count("registered=13 queried=13"), 3)

    def test_memcpy_dispatch_preserves_kind_stream_and_native_errors(self):
        production = (ROOT / "flagcx/adaptor/device/hggc_adaptor.cc").read_text()
        start = production.index("flagcxResult_t hggcAdaptorDeviceMemcpy(")
        end = production.index("flagcxResult_t hggcAdaptorDeviceMemset(", start)
        source = '#include "ppu_adaptor.h"\n#include "adaptor.h"\n' + production[start:end]
        self.compile_and_run(source + r"""
int main() {
  flagcxStream stream{reinterpret_cast<hggcStream_t>(4)};
  const flagcxMemcpyType_t types[] = {flagcxMemcpyHostToDevice, flagcxMemcpyDeviceToHost,
                                     flagcxMemcpyDeviceToDevice};
  const hggcMemcpyKind expected[] = {hggcMemcpyHostToDevice, hggcMemcpyDeviceToHost,
                                    hggcMemcpyDeviceToDevice};
  for (int i = 0; i < 3; i++) {
    assert(hggcAdaptorDeviceMemcpy(nullptr, nullptr, 64, types[i], nullptr, nullptr) == flagcxSuccess);
    assert(seenMemcpyKind == expected[i] && seenMemcpyStream == nullptr);
    assert(hggcAdaptorDeviceMemcpy(nullptr, nullptr, 64, types[i], &stream, nullptr) == flagcxSuccess);
    assert(seenMemcpyKind == expected[i] && seenMemcpyStream == stream.base);
  }
  assert(memcpyCalls == 6);
  assert(hggcAdaptorDeviceMemcpy(nullptr, nullptr, 64, (flagcxMemcpyType_t)-1,
                                  &stream, nullptr) == flagcxInvalidArgument);
  assert(memcpyCalls == 6);
  memcpyStatus = hggcErrorUnknown;
  assert(hggcAdaptorDeviceMemcpy(nullptr, nullptr, 64, flagcxMemcpyHostToDevice,
                                  &stream, nullptr) == flagcxUnhandledDeviceError);
}
""")

    def test_async_allocator_uses_current_pool_and_preserves_stream_and_errors(self):
        source = (ROOT / "flagcx/adaptor/device/hggc_adaptor.cc").read_text()
        start = source.index("flagcxResult_t hggcAdaptorDeviceMalloc(")
        end = source.index("flagcxResult_t hggcAdaptorDeviceFree(", start)
        source = '#include "ppu_adaptor.h"\n#include "adaptor.h"\n' + source[start:end]
        self.compile_and_run(source + r"""
int main() {
  void *ptr = nullptr;
  flagcxStream stream{reinterpret_cast<hggcStream_t>(4)};
  assert(hggcAdaptorDeviceMalloc(&ptr, 64, flagcxMemDevice, &stream) == flagcxSuccess);
  assert(mallocCalls == 1 && poolMallocCalls == 0 && seenStream == stream.base);
  mallocStatus = hggcErrorUnknown;
  assert(hggcAdaptorDeviceMalloc(&ptr, 64, flagcxMemDevice, &stream) == flagcxUnhandledDeviceError);
  assert(mallocCalls == 2 && poolMallocCalls == 0 && seenStream == stream.base);
  assert(hggcResultToFlagcx(hggcErrorNotReady) == flagcxInProgress);
  assert(hggcDriverResultToFlagcx(HGGC_ERROR_NOT_SUPPORTED) == flagcxNotSupported);
}
""")

    def test_legacy_accl_pointer_classification_uses_native_runtime(self):
        source = (ROOT / "flagcx/core/flagcx_p2p_accl.cc").read_text()
        source = source[source.index("bool classifyPtr("):source.index("class AcclNullCb")]
        for modern in (False, True):
            with self.subTest(modern=modern):
                prelude = f"#define FLAGCX_HGGC_HAS_POINTER_TYPE {int(modern)}\n"
                prelude += '#include <hggc_runtime.h>\nenum device_type { CPU, GPU };\n'
                self.compile_and_run(prelude + source + r"""
int main() {
  device_type kind = CPU;
  int ordinal = -1;
  assert(classifyPtr(nullptr, &kind, &ordinal));
  assert(kind == GPU && ordinal == 3);
  pointerStatus = hggcErrorUnknown;
  assert(classifyPtr(nullptr, &kind, &ordinal));
  assert(kind == CPU && ordinal == 0);
}
""")

    def test_native_adaptor_build_dependency_and_link_configuration(self):
        target = ROOT / "build/obj/flagcx/adaptor/device/hggc_adaptor.o"
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("USE_") and key != "CCL_HOME"}
        result = subprocess.run(["make", "-B", "-n", "USE_PPU=1",
                                 "DEVICE_HOME=/opt/ppu",
                                 f"BUILDDIR={ROOT / 'build'}", str(target)],
                                cwd=ROOT, env=environment, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("tools/probe_hggc_sdk.py", result.stdout)
        self.assertIn("hggc_sdk_features.h", result.stdout)
        self.assertIn("-lhggc_wrapper -lhg_wrapper", result.stdout)
        self.assertNotIn("-lpccl", result.stdout)
        self.assertIn("CCL_INCLUDE=/usr/local/PPU_SDK/include", result.stdout)
        self.assertIn("DEVICE_INCLUDE=/opt/ppu/include", result.stdout)
        self.assertNotIn("-lcudart", result.stdout)
        self.assertNotIn("-lnccl", result.stdout)

    @unittest.skipUnless(shutil.which("ar"), "archive tool required")
    def test_sdk_probe_links_without_executing_and_rejects_missing_abi(self):
        spec = importlib.util.spec_from_file_location("probe_hggc", ROOT / "tools/probe_hggc_sdk.py")
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            # A runtime call would abort. The capability tool must only link.
            runtime_base = '#include <cstdlib>\n' + RUNTIME.replace("++mallocCalls;", "std::abort(); ++mallocCalls;")
            runtime = runtime_base + RUNTIME_TEMPLATES
            pccl = PCCL + "\n".join(
                f"inline pcclResult_t {name}({args}) {{ return pcclStatus; }}"
                for name, args in PCCL_SIGNATURES.items()
            )
            declared = set(re.findall(r"\b(\w+)\(", runtime + pccl + DRIVER))
            required_stubs = [name for name in probe.REQUIRED if name not in declared]
            (tmp / "hggc.h").write_text(DRIVER)
            (tmp / "pccl.h").write_text(pccl)
            (tmp / "empty.cc").write_text("int sdk_fixture_anchor = 0;\n")
            result = subprocess.run([CXX, "-c", str(tmp / "empty.cc"), "-o", str(tmp / "empty.o")],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for library in ("pccl", "hggc_wrapper", "hg_wrapper"):
                result = subprocess.run(["ar", "rcs", str(tmp / f"lib{library}.a"), str(tmp / "empty.o")],
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            output = tmp / "hggc_sdk_features.h"
            command = [os.sys.executable, str(ROOT / "tools/probe_hggc_sdk.py"),
                       "--cxx", CXX,
                       "--include", str(tmp), "--library", str(tmp), "--output", str(output)]
            (tmp / "hggc_runtime.h").write_text(runtime + "\n".join(
                f"inline int {name}() {{ return 0; }}" for name in required_stubs
            ))
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("#define FLAGCX_HGGC_HAS_MANAGED 0", output.read_text())

            # Also accept public APIs exposed only as function templates,
            # forwarding to underlying runtime entry points.
            template_runtime = runtime_base
            templates = RUNTIME_TEMPLATES
            for name in ("hggcMalloc", "hggcHostAlloc", "hggcHostGetDevicePointer"):
                template_runtime = re.sub(r"\b" + name + r"(?=\()",
                                          "sdk_" + name, template_runtime)
                templates = templates.replace("return " + name + "(",
                                              "return sdk_" + name + "(")
            (tmp / "hggc_runtime.h").write_text(template_runtime + templates + "\n".join(
                f"inline int {name}() {{ return 0; }}" for name in required_stubs
            ))
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)

            # A pool-only overload cannot satisfy the current-pool contract.
            pool_only_runtime = re.sub(
                r"inline hggcError_t hggcMallocAsync\(void \*\*p, size_t, hggcStream_t stream\) \{.*?\n\}",
                "", runtime, count=1, flags=re.DOTALL,
            )
            output.unlink()
            (tmp / "hggc_runtime.h").write_text(pool_only_runtime + "\n".join(
                f"inline int {name}() {{ return 0; }}" for name in required_stubs
            ))
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("required interface compile/link check failed", result.stderr)
            self.assertIn("hggcMallocAsync", result.stderr)
            self.assertFalse(output.exists())

            # Removing a mandatory IPC entry point must stop configuration.
            (tmp / "hggc_runtime.h").write_text(runtime + "\n".join(
                f"inline int {name}() {{ return 0; }}"
                for name in required_stubs if name != "hggcIpcOpenMemHandle"
            ))
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("required interface compile/link check failed", result.stderr)
            self.assertIn("hggcIpcOpenMemHandle", result.stderr)
            self.assertFalse(output.exists())

            # A declaration without its library definition must fail linking,
            # including for the overloaded APIs checked using call expressions.
            missing_symbol_runtime = runtime.replace(
                "inline hggcError_t hggcHostGetDevicePointer(void **device, void *host, unsigned) {\n"
                "  *device = host; return hggcSuccess;\n}",
                "hggcError_t hggcHostGetDevicePointer(void **device, void *host, unsigned);",
            )
            (tmp / "hggc_runtime.h").write_text(missing_symbol_runtime + "\n".join(
                f"inline int {name}() {{ return 0; }}" for name in required_stubs
            ))
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("required interface compile/link check failed", result.stderr)
            self.assertIn("hggcHostGetDevicePointer", result.stderr)
            self.assertIn("undefined", result.stderr.lower())
            self.assertFalse(output.exists())


    def test_hggc_device_selection_always_binds_native_context(self):
        production = (ROOT / "flagcx/adaptor/device/hggc_adaptor.cc").read_text()
        start = production.index("flagcxResult_t hggcAdaptorSetDevice(int dev)")
        end = production.index("flagcxResult_t hggcAdaptorGetDevice(", start)
        source = '#include "ppu_adaptor.h"\n'
        source += r"""
static int nativeDevice = 2, selectCalls = 0;
static bool contextBound = false;
static hggcError_t selectResult = hggcSuccess;
static hggcError_t hggcSetDevice(int dev) {
  ++selectCalls;
  if (selectResult == hggcSuccess) {
    nativeDevice = dev;
    contextBound = true;
  }
  return selectResult;
}
"""
        source += production[start:end] + r"""
int main() {
  assert(hggcAdaptorSetDevice(2) == flagcxSuccess);
  assert(contextBound && selectCalls == 1);
  contextBound = false;
  assert(hggcAdaptorSetDevice(2) == flagcxSuccess);
  assert(contextBound && selectCalls == 2);
  assert(hggcAdaptorSetDevice(3) == flagcxSuccess);
  assert(selectCalls == 3 && nativeDevice == 3);
  nativeDevice = 2; // An application can switch devices outside FlagCX.
  assert(hggcAdaptorSetDevice(3) == flagcxSuccess);
  assert(selectCalls == 4 && nativeDevice == 3);
  selectResult = hggcErrorInvalidValue;
  contextBound = false;
  assert(hggcAdaptorSetDevice(3) == flagcxUnhandledDeviceError);
  assert(selectCalls == 5 && !contextBound);
  assert(hggcAdaptorSetDevice(-1) == flagcxUnhandledDeviceError);
  assert(selectCalls == 6 && nativeDevice == 3);
}
"""
        self.compile_and_run(source)



if __name__ == "__main__":
    if len(os.sys.argv) == 3 and os.sys.argv[1] == "--generate-contracts":
        Path(os.sys.argv[2]).mkdir(parents=True, exist_ok=True)
        generate_event_contracts(os.sys.argv[2])
        generate_native_contracts(os.sys.argv[2])
        generate_host_contracts(os.sys.argv[2])
    else:
        unittest.main()
