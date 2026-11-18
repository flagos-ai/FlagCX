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
enum hggcError_t { hggcSuccess = 0, hggcErrorNotReady = 70, hggcErrorUnknown = 99 };
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
enum { hggcHostAllocMapped = 1 };
static int mallocCalls = 0;
static hggcError_t mallocStatus = hggcSuccess;
static hggcMemPool_t seenPool = reinterpret_cast<hggcMemPool_t>(1);
static hggcStream_t seenStream = nullptr;
inline hggcError_t hggcMalloc(void **p, size_t) {
  *p = reinterpret_cast<void *>(2); return mallocStatus;
}
inline hggcError_t hggcHostAlloc(void **p, size_t size, unsigned) {
  return hggcMalloc(p, size);
}
inline hggcError_t hggcMallocAsync(void **p, size_t, hggcMemPool_t pool, hggcStream_t stream) {
  ++mallocCalls; seenPool = pool; seenStream = stream;
  *p = reinterpret_cast<void *>(3); return mallocStatus;
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
inline const char *pcclGetErrorString(pcclResult_t) { return "pccl error"; }
inline const char *pcclGetLastError(pcclComm_t) { return "last error"; }
inline pcclResult_t pcclCommGetAsyncError(pcclComm_t, pcclResult_t *r) {
  *r = pcclAsyncStatus; return pcclStatus;
}
inline pcclResult_t pcclAllReduce(const void *, void *, size_t, pcclDataType_t type,
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
class PpuAdaptorTest(unittest.TestCase):
    def compile_and_run(self, source):
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            (tmp / "hggc_runtime.h").write_text(RUNTIME)
            (tmp / "hggc.h").write_text(DRIVER)
            pccl = PCCL + "\n".join(
                f"inline pcclResult_t {name}({args}) {{ return pcclStatus; }}"
                for name, args in PCCL_SIGNATURES.items()
            )
            (tmp / "pccl.h").write_text(pccl)
            (tmp / "adaptor.h").write_text(PROJECT)
            (tmp / "alloc.h").write_text('#include "adaptor.h"\n')
            for name in ("comm.h", "param.h"):
                (tmp / name).write_text("")
            cpp = tmp / "test.cc"
            cpp.write_text('#include <cassert>\n' + source)
            command = [CXX, "-std=c++17", "-DUSE_PPU_ADAPTOR",
                       f"-I{tmp}", f"-I{INCLUDE}",
                       f"-I{ROOT / 'flagcx/include'}", str(cpp),
                       "-o", str(tmp / "test")]
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(tmp / "test")], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)

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
}
""")

    def test_async_allocator_uses_null_pool_and_preserves_stream_and_errors(self):
        source = (ROOT / "flagcx/adaptor/device/hggc_adaptor.cc").read_text()
        start = source.index("flagcxResult_t hggcAdaptorDeviceMalloc(")
        end = source.index("flagcxResult_t hggcAdaptorDeviceFree(", start)
        source = '#include "ppu_adaptor.h"\n#include "adaptor.h"\n' + source[start:end]
        self.compile_and_run(source + r"""
int main() {
  void *ptr = nullptr;
  flagcxStream stream{reinterpret_cast<hggcStream_t>(4)};
  assert(hggcAdaptorDeviceMalloc(&ptr, 64, flagcxMemDevice, &stream) == flagcxSuccess);
  assert(mallocCalls == 1 && seenPool == nullptr && seenStream == stream.base);
  mallocStatus = hggcErrorUnknown;
  assert(hggcAdaptorDeviceMalloc(&ptr, 64, flagcxMemDevice, &stream) == flagcxUnhandledDeviceError);
  assert(mallocCalls == 2 && seenPool == nullptr && seenStream == stream.base);
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
        self.assertIn("CCL_LINK=-lpccl", result.stdout)
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
            runtime = '#include <cstdlib>\n' + RUNTIME.replace("++mallocCalls;", "std::abort(); ++mallocCalls;")
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
            output = tmp / "features.h"
            command = [os.sys.executable, str(ROOT / "tools/probe_hggc_sdk.py"),
                       "--cxx", CXX,
                       "--include", str(tmp), "--library", str(tmp), "--output", str(output)]
            (tmp / "hggc_runtime.h").write_text(runtime + "\n".join(
                f"inline int {name}() {{ return 0; }}" for name in required_stubs
            ))
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("#define FLAGCX_HGGC_HAS_MANAGED 0", output.read_text())
            # Removing a mandatory IPC entry point must stop configuration.
            output.unlink()
            (tmp / "hggc_runtime.h").write_text(runtime + "\n".join(
                f"inline int {name}() {{ return 0; }}"
                for name in required_stubs if name != "hggcIpcOpenMemHandle"
            ))
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing a required adaptor interface", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
