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
class PpuAdaptorTest(unittest.TestCase):
    def compile_and_run(self, source, missing_pccl_symbol=None, fail_open=False):
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
#include <dlfcn.h>
static int openLibraryCalls = 0;
static int seenLoadFlags = 0;
void *flagcxOpenLib(const char *path, int flags,
                   void (*error_handler)(const char *, int, const char *)) {
  ++openLibraryCalls;
  seenLoadFlags = flags;
  if (""" + str(int(fail_open)) + r""") {
    error_handler(path, 0, "fixture library missing");
    return nullptr;
  }
  return dlopen(nullptr, RTLD_NOW);
}
""" + source)
            command = [CXX, "-std=c++17", "-DUSE_PPU_ADAPTOR",
                       f"-I{tmp}", f"-I{INCLUDE}",
                       f"-I{ROOT / 'flagcx/include'}", str(cpp),
                       f"-I{ROOT / 'flagcx/service/include'}", "-rdynamic",
                       "-o", str(tmp / "test")]
            if os.sys.platform == "linux":
                command += ["-ldl", "-pthread"]
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
            common = [CXX, "-std=c++17", "-DUSE_PPU_ADAPTOR", f"-I{tmp}",
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


if __name__ == "__main__":
    unittest.main()
