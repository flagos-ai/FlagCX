"""Hardware-free compilation and build checks for the MetaX Default API.

The runtime model checks C++ dispatch, mask width and ordering calls only.
Actual MACA code generation and peer visibility require the hardware suites.
"""

from pathlib import Path
import ast
import contextlib
import io
import multiprocessing
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
CXX = shutil.which("clang++") or shutil.which("g++")

RUNTIME = r"""
#pragma once
#include <cstdint>
#define __host__
#define __device__
#define __forceinline__ inline
#define __global__
#define __shared__ static
#define __constant__
using mcStream_t = void *;
struct Dim { unsigned x = 0; };
static Dim threadIdx, blockIdx, blockDim, gridDim;
struct int4 { int x, y, z, w; };
static uint64_t active = ~uint64_t{0}, synced = 0;
static int systemFences = 0, deviceFences = 0, blockFences = 0;
inline uint64_t __activemask() { return active; }
inline void __syncwarp(unsigned long long mask) { synced = mask; }
inline int __popcll(unsigned long long mask) { return __builtin_popcountll(mask); }
inline void __syncthreads() {}
inline void __threadfence_system() { ++systemFences; }
inline void __threadfence() { ++deviceFences; }
inline void __threadfence_block() { ++blockFences; }
inline uint64_t clock64() { static uint64_t tick = 0; return ++tick; }
"""

PRELUDE = r"""
#include "device_api/platform_traits.h"
#if defined(FLAGCX_METAX_DEVICE_COMPILE)
#include "device_api/comm_traits.h"
#endif
#include <cassert>
#include <type_traits>
using P = PlatformTraits<MetaxPlatform>;
static_assert(FLAGCX_SIMT_WIDTH == 64, "MACA SIMT width");
#if defined(FLAGCX_METAX_DEVICE_COMPILE)
static_assert(std::is_same<DeviceAPI,
              CommTraits<DefaultBackend<MetaxPlatform>>>::value,
              "MetaX must select the Default backend");
#endif
static_assert(sizeof(PlatformCompletionWord<MetaxPlatform>::type) == 8,
              "host and device completion ABI");
#ifdef FLAGCX_NVIDIA_PLATFORM_TRAITS_H_
#error MetaX must not select NVIDIA platform traits
#endif
#ifdef FLAGCX_NVIDIA_PTX_MEMORY_H_
#error MetaX must not include NVIDIA PTX memory helpers
#endif
"""


@unittest.skipUnless(CXX, "C++ compiler required")
class MetaxTraitsTest(unittest.TestCase):
    def compile_and_run(self, source, *, native=False, compiler_flags=(),
                        standard="c++11"):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            (directory / "mcr").mkdir()
            (directory / "mcr/mc_runtime.h").write_text(RUNTIME)
            cpp = directory / "test.cpp"
            cpp.write_text(PRELUDE + source)
            cmd = [CXX, f"-std={standard}", "-Wall", "-Wextra", "-Werror",
                   "-Wno-unused-variable", "-Wno-unused-parameter",
                   "-DUSE_METAX_ADAPTOR",
                   f"-I{directory}", f"-I{ROOT / 'flagcx/adaptor/include'}",
                   f"-I{ROOT / 'flagcx/include'}",
                   f"-I{ROOT / 'flagcx/core/include'}",
                   f"-I{ROOT / 'flagcx/service/include'}", str(cpp),
                   "-o", str(directory / "test")]
            if native:
                # Native MACA marker without the CUDA compatibility marker.
                cmd.insert(1, "-D__MACACC__")
            cmd[1:1] = compiler_flags
            result = subprocess.run(cmd, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(directory / "test")],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_host_dispatch_and_completion_atomics(self):
        self.compile_and_run(r"""
int main() {
  uint64_t word = (uint64_t{1} << 40);
  auto previous = P::Atomic::fetchAdd(&word, uint64_t{7},
                                     flagcxDeviceMemoryOrderAcqRel);
  assert(previous == (uint64_t{1} << 40));
  uint64_t expected = previous;
  assert(!P::Atomic::compareExchange(&word, expected, uint64_t{0},
                                     flagcxDeviceMemoryOrderSeqCst));
  assert(expected == word);
  assert(P::Atomic::compareExchange(&word, expected, uint64_t{9},
                                    flagcxDeviceMemoryOrderRelease));
  assert(P::Atomic::fetchOr(&word, uint64_t{16},
                           flagcxDeviceMemoryOrderRelaxed) == 9);
  assert(P::Atomic::fetchAnd(&word, uint64_t{15},
                            flagcxDeviceMemoryOrderRelaxed) == 25);
  assert(P::Atomic::exchange(&word, uint64_t{12},
                            flagcxDeviceMemoryOrderRelaxed) == 9);
  assert(P::Atomic::fetchSub(&word, uint64_t{3},
                            flagcxDeviceMemoryOrderRelaxed) == 12);
  assert(P::Atomic::load(&word, flagcxDeviceMemoryOrderAcquire) == 9);
  uint32_t small = 0;
  P::Atomic::store(&small, uint32_t{4}, flagcxDeviceMemoryOrderRelease);
  assert(P::Atomic::load(&small, flagcxDeviceMemoryOrderAcquire) == 4);
}
""")

    def test_explicit_and_compatibility_compiler_markers(self):
        for flags in (("-DFLAGCX_METAX_DEVICE_COMPILE=1",),
                      ("-DFLAGCX_METAX_DEVICE_COMPILE=1", "-D__CUDACC__",
                       "-D__CUDA_ARCH__=800")):
            with self.subTest(flags=flags):
                self.compile_and_run(r"""
static_assert(FLAGCX_DEVICE_COMPILE && FLAGCX_CHECK_DEVICE_CC,
              "MACA compile markers must select device implementations");
int main() {
  threadIdx.x = 63;
  assert(P::Intrin::lane() == 63);
  P::Intrin::syncwarp();
  assert(synced == ~uint64_t{0});
}
""", compiler_flags=flags)

    def test_upper_lanes_and_partial_tiles_keep_64_bit_masks(self):
        self.compile_and_run(r"""
int main() {
  static_assert(FLAGCX_DEVICE_COMPILE && FLAGCX_CHECK_DEVICE_CC,
                "native MACA device compilation must be detected");
  threadIdx.x = 63;
  assert(P::Intrin::lane() == 63);
  assert(P::Intrin::lanemaskLt() == ((uint64_t{1} << 63) - 1));
  active = 0xff00000000000000ull;
  assert(P::Intrin::activemask() == active);
  P::CoopTile<8> tile;
  assert(tile.threadRank() == 7 && tile.size() == 8);
  tile.sync();
  assert(synced == active);
  P::CoopLanes lanes(active);
  assert(lanes.threadRank() == 7 && lanes.size() == 8);
  lanes.sync();
  assert(synced == active);
  P::CoopWarp wave;
  assert(wave.size() == 64 && wave.threadRank() == 63);
  wave.sync();
  assert(synced == ~uint64_t{0});
}
""", native=True)

    def test_native_value_types_construct_from_host_descriptors(self):
        self.compile_and_run(r"""
#include "device_api/flagcx_device.h"
#include <new>
void constructNet(flagcxDevNet *net, flagcxDevComm *comm) {
  ::new (net) flagcxDevNet(*comm, 0);
}
int main() {
  flagcxDevCommInternal descriptor = {};
  descriptor.rank = 3;
  descriptor.nRanks = 4;
  descriptor.intraRank = 1;
  descriptor.intraSize = 2;
  flagcxDevComm comm(descriptor);
  assert(comm.getRank() == 3 && comm.getSize() == 4);
  assert(comm.getIntraRank() == 1 && comm.getIntraSize() == 2);
}
""", native=True, standard="c++17")

    def test_system_publication_fences_and_unaligned_cooperative_copy(self):
        self.compile_and_run(r"""
int main() {
  uint64_t word = 0;
  P::Atomic::store(&word, uint64_t{7}, flagcxDeviceMemoryOrderRelease);
  assert(systemFences >= 1);
  systemFences = 0;
  assert(P::Atomic::load(&word, flagcxDeviceMemoryOrderAcquire) == 7);
  assert(systemFences >= 1);
  P::Atomic::fetchAdd<uint64_t, flagcxDeviceScopeDevice>(
      &word, uint64_t{1}, flagcxDeviceMemoryOrderRelaxed);
  assert(deviceFences >= 1);
  unsigned char source[73], target[73] = {};
  for (int i = 0; i < 73; ++i) source[i] = static_cast<unsigned char>(i);
  for (int rank = 0; rank < 8; ++rank)
    P::Intrin::coopCopyBytes(target + 1, source + 1, 71, rank, 8);
  assert(target[0] == 0 && target[72] == 0);
  for (int i = 1; i < 72; ++i) assert(target[i] == source[i]);
}
""", native=True)

    def test_complete_ir_translation_unit_exports_every_public_symbol(self):
        if "clang" not in Path(CXX).name:
            self.skipTest("LLVM IR inspection requires Clang")
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            (directory / "mcr").mkdir()
            (directory / "mcr/mc_runtime.h").write_text(RUNTIME)
            cpp = directory / "wrapper.cpp"
            cpp.write_text('#include "flagcx_device_wrapper_impl.h"\n')
            ll = directory / "wrapper.ll"
            result = subprocess.run(
                [CXX, "-std=c++17", "-D__MACACC__", "-DUSE_METAX_ADAPTOR",
                 "-D__clang_llvm_bitcode_lib__", "-S", "-emit-llvm", "-O1",
                 f"-I{directory}", f"-I{ROOT / 'bindings/ir'}",
                 f"-I{ROOT / 'flagcx/adaptor/include'}",
                 f"-I{ROOT / 'flagcx/adaptor/include/device_api'}",
                 f"-I{ROOT / 'flagcx/include'}", str(cpp), "-o", str(ll)],
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            definitions = set(re.findall(r"^define .*?@([\w]+)\(",
                                         ll.read_text(), re.MULTILINE))
            public = {line.strip() for line in
                      (ROOT / "bindings/ir/public_symbols.txt").read_text().splitlines()
                      if line.strip() and not line.startswith("#")}
            self.assertFalse(public - definitions,
                             f"Missing IR definitions: {sorted(public - definitions)}")

    def test_allocation_policy_preserves_async_lifetimes_and_errors(self):
        source = (ROOT / "flagcx/adaptor/device/maca_adaptor.cc").read_text()
        allocation = source[source.index("flagcxResult_t macaAdaptorDeviceMalloc("):
                            source.index("flagcxResult_t macaAdaptorSetDevice(")]
        self.compile_and_run(r"""
enum flagcxMemType_t { flagcxMemHost, flagcxMemManaged, flagcxMemDevice };
enum flagcxResult_t { flagcxSuccess, flagcxUnhandledDeviceError };
struct flagcxStream { mcStream_t base; };
using flagcxStream_t = flagcxStream *;
enum { mcSuccess, mcMemAttachGlobal };
static int operation = 0, error = 0;
static mcStream_t usedStream;
int mcMallocHost(void **, size_t) { operation = 1; return error; }
int mcMallocManaged(void **, size_t, int) { operation = 2; return error; }
int mcMalloc(void **, size_t) { operation = 3; return error; }
int mcMallocAsync(void **, size_t, mcStream_t stream) {
  operation = 4; usedStream = stream; return error;
}
int mcFreeHost(void *) { operation = 5; return error; }
int mcFree(void *) { operation = 6; return error; }
int mcFreeAsync(void *, mcStream_t stream) {
  operation = 7; usedStream = stream; return error;
}
#define DEVCHECK(call) do { if ((call) != mcSuccess) return flagcxUnhandledDeviceError; } while (0)
""" + allocation + r"""
int main() {
  void *ptr = nullptr;
  flagcxStream stream{reinterpret_cast<void *>(0x1234)};
  assert(macaAdaptorDeviceMalloc(&ptr, 64, flagcxMemDevice, nullptr) == flagcxSuccess);
  assert(operation == 3);
  assert(macaAdaptorDeviceMalloc(&ptr, 64, flagcxMemDevice, &stream) == flagcxSuccess);
  assert(operation == 4 && usedStream == stream.base);
  assert(macaAdaptorDeviceFree(ptr, flagcxMemDevice, &stream) == flagcxSuccess);
  assert(operation == 7 && usedStream == stream.base);
  assert(macaAdaptorDeviceFree(ptr, flagcxMemDevice, nullptr) == flagcxSuccess);
  assert(operation == 6);
  assert(macaAdaptorDeviceMalloc(&ptr, 64, flagcxMemHost, nullptr) == flagcxSuccess);
  assert(operation == 1);
  assert(macaAdaptorDeviceFree(ptr, flagcxMemHost, nullptr) == flagcxSuccess);
  assert(operation == 5);
  assert(macaAdaptorDeviceMalloc(&ptr, 64, flagcxMemManaged, nullptr) == flagcxSuccess);
  assert(operation == 2);
  error = 1;
  assert(macaAdaptorDeviceMalloc(&ptr, 64, flagcxMemDevice, nullptr) == flagcxUnhandledDeviceError);
  assert(operation == 3); // Propagate the native allocation error.
}
""")


class MetaxBuildTest(unittest.TestCase):
    def make(self, *args, directory=ROOT, check=True):
        result = subprocess.run(["make", "-n", "-B", "-C", str(directory), *args,
                                 "DEVICE_HOME=/opt/maca", "CCL_HOME=/opt/maca"],
                                capture_output=True, text=True)
        if check:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_library_and_all_six_mpi_binaries_use_native_device_link(self):
        library = self.make("USE_METAX=1", "COMPILE_KERNEL=1").stdout
        self.assertIn("/kernel/metax/device_api_host_helpers.cu", library)
        self.assertIn("-maca-link", library)
        self.assertIn("-maca-host-input", library)
        self.assertNotIn(" -dlink ", library)
        suite = ROOT / "test/unittest/device_api"
        build = self.make("USE_METAX=1", "COMPILE_KERNEL=1", "mpi",
                          directory=suite).stdout
        for target in ("test_device_api_intra", "test_device_api_inter",
                       "test_device_ir_intra", "test_device_ir_inter",
                       "test_device_ir_unified_intra", "test_device_ir_unified_inter"):
            link = next(line for line in build.splitlines()
                        if f"/bin/{target} " in line)
            self.assertTrue(link.startswith("/opt/maca/mxgpu_llvm/bin/mxcc "), link)
            self.assertIn(" -maca-link ", link)
            self.assertIn("-maca-host-input", link)
            self.assertIn("/kernel/maca/build/obj/device_", link)
        unit = self.make("USE_METAX=1", "unit", directory=suite).stdout
        link = next(line for line in unit.splitlines()
                    if "/bin/test_dev_comm_cleanup " in line)
        self.assertTrue(link.startswith("g++ "), link)
        self.assertNotIn(" -maca-link ", link)

    def test_bitcode_uses_matching_architecture_and_complete_exports(self):
        build = self.make("USE_METAX=1", "DEVICE_BITCODE_ARCH=xcore2000",
                          "device_bitcode").stdout
        command = next(line for line in build.splitlines()
                       if "mxcc " in line and " -emit-llvm " in line)
        for flag in ("-x maca", "-maca-device-only", "-offload-arch=xcore2000",
                     "--maca-path=/opt/maca", "-fkeep-device-symbols",
                     "-DFLAGCX_COMM_TRAITS_DEFAULT", "-DFLAGCX_METAX_DEVICE_COMPILE=1"):
            self.assertIn(flag, command)
        self.assertNotIn("ppu", command.lower())
        helper = next(line for line in build.splitlines()
                      if "mxcc " in line and "device_api_host_helpers.cu" in line)
        self.assertIn("-offload-arch=xcore2000", helper)
        self.assertIn("public_symbols.txt", build)
        self.assertIn("--passes=verify", build)
        self.assertIn("/metax_platform_traits.h", build)
        for arguments, error in [(("USE_METAX=1", "device_bitcode"),
                                  "DEVICE_BITCODE_ARCH is required"),
                                 (("USE_METAX=1", "DEVICE_BITCODE_ARCH=xcore1000",
                                   "COMPILE_KERNEL=0", "device_bitcode"),
                                  "device_bitcode requires COMPILE_KERNEL=1"),
                                 (("USE_METAX=1", "DEVICE_BITCODE_ARCH=xcore1000",
                                   "METAX_ARCH=xcore2000", "device_bitcode"),
                                  "METAX_ARCH must match DEVICE_BITCODE_ARCH")]:
            rejected = self.make(*arguments, check=False)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn(error, rejected.stderr)

    def test_header_export_and_wheel_payload_follow_the_manifest(self):
        # Execute the real packaging method with compiler/build tools stubbed;
        # the fake .bc payload verifies packaging, not MACA code generation.
        tree = ast.parse((ROOT / "setup.py").read_text())
        definition = next(node for node in ast.walk(tree)
                          if isinstance(node, ast.ClassDef) and node.name == "BuildExtWithMake")

        class Base:
            def build_extensions(self):
                pass

        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            build = directory / "build"
            result = subprocess.run(
                ["make", "-C", str(ROOT / "bindings/ir/metax"),
                 f"BUILDDIR={build}", "headers"], capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            (build / "lib").mkdir()
            for filename in ("libflagcx.so", "libflagcx_device.bc"):
                (build / "lib" / filename).write_bytes(b"test payload")
            json_dir = directory / "json"
            (json_dir / "nlohmann").mkdir(parents=True)
            (json_dir / "nlohmann/json.hpp").touch()
            calls = []
            namespace = dict(BuildExtension=Base, ROOT_DIR=str(directory),
                             JSON_INCLUDE_DIR=str(json_dir),
                             VENDORED_JSON_INCLUDE_DIR=str(json_dir),
                             adaptor="metax", adaptor_make_flag="USE_METAX",
                             os=os, shutil=shutil, multiprocessing=multiprocessing,
                             subprocess=SimpleNamespace(check_call=calls.append))
            exec(compile(ast.Module(body=[definition], type_ignores=[]),
                         "setup.py", "exec"), namespace)
            builder = namespace["BuildExtWithMake"]()
            builder.extensions = []
            builder.build_lib = str(directory / "wheel")
            with mock.patch.dict(os.environ, FLAGCX_BITCODE_ARCH="xcore1000",
                                 COMPILE_KERNEL="1"), contextlib.redirect_stdout(io.StringIO()):
                builder.build_extensions()
            self.assertIn("USE_METAX=1", calls[0])
            self.assertIn("DEVICE_BITCODE_ARCH=xcore1000", calls[0])
            self.assertEqual(calls[0][-1], "device_bitcode")
            manifest = (build / "include/.flagcx_device_headers").read_text().splitlines()
            self.assertIn("metax_platform_traits.h", manifest)
            for base in (directory / "wheel", directory / "src"):
                package = base / "flagcx"
                self.assertEqual((package / "lib/libflagcx_device.bc").read_bytes(),
                                 b"test payload")
                self.assertEqual({p.name for p in (package / "include").iterdir()},
                                 set(manifest))


class MetaxCiTest(unittest.TestCase):
    def test_three_suites_use_shared_runner_and_propagate_mpi_failures(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            scripts = directory / ".github/scripts/ci"
            scripts.mkdir(parents=True)
            for filename in ("run_unit_test.sh", "ignore_test.py",
                             "filter_supplemental_tests.sh"):
                shutil.copy2(ROOT / ".github/scripts/ci" / filename, scripts)
            (directory / "test/unittest/device_api").mkdir(parents=True)
            binaries = directory / "tools"
            binaries.mkdir()
            log = directory / "commands.log"

            def executable(path, body):
                path.write_text("#!/usr/bin/env bash\n" + body)
                path.chmod(0o755)

            executable(binaries / "nproc", "echo 1\n")
            for filename in ("make", "cmake"):
                executable(binaries / filename,
                           f'printf "%s|%s\\n" {filename} "$*" >> "$METAX_TEST_LOG"\n')
            executable(scripts / "run_mpi_with_timeout.sh", r'''
printf '%s|%s\n' "$FLAGCX_CI_MPI_LABEL" "$*" >> "$METAX_TEST_LOG"
if [[ "$FLAGCX_CI_MPI_LABEL" == "${METAX_TEST_FAIL_LABEL:-}" ]]; then exit 7; fi
''')
            executable(scripts / "run_with_timeout.sh", r'''
printf '%s|%s\n' "$FLAGCX_CI_TEST_LABEL" "$*" >> "$METAX_TEST_LOG"
exec "$@"
''')
            env_script = directory / "metax.sh"
            env_script.write_text(
                f"source {shlex.quote(str(ROOT / '.github/scripts/set_env/metax.sh'))}\n"
                'flagcx_ci_prepare() { :; }\n'
                'flagcx_ci_validate_rdma() { '
                'printf "rdma|%s\\n" "$1" >> "$METAX_TEST_LOG"; }\n'
            )
            environment = dict(os.environ, GITHUB_WORKSPACE=str(directory),
                               MPI_HOME=str(directory / "mpi"),
                               PATH=f"{binaries}:{os.environ['PATH']}",
                               FLAGCX_CI_COVERAGE="0", METAX_TEST_LOG=str(log))
            for key in ("BASH_ENV", "FLAGCX_CI_UNIFIED_IR_SKIP_S16",
                        "FLAGCX_CI_DEVICE_API_INTRA_ONLY", "FLAGCX_CI_DEVICE_API_INTER",
                        "FLAGCX_CI_UNIFIED_IR_INTER", "FLAGCX_CI_UNIFIED_IR_FALLBACK",
                        "FLAGCX_CI_DEVICE_API_SHMEM_INTRANODE_ONLY"):
                environment.pop(key, None)
            cases = [
                ("device_api", "", {"device_api intra", "device_api inter",
                                      "device IR intra", "device IR inter"}),
                ("device_api_host", "", {"device_api_host unit tests"}),
                ("device_api_unified_ir", "", {"unified IR intra", "unified IR inter",
                                               "unified IR intra forced fallback",
                                               "unified IR inter forced fallback"}),
                ("device_api_unified_ir", "unified IR intra forced fallback", set()),
            ]
            for suite, failed_label, expected in cases:
                with self.subTest(suite=suite, failed_label=failed_label):
                    log.write_text("")
                    environment["METAX_TEST_FAIL_LABEL"] = failed_label
                    result = subprocess.run(
                        ["bash", str(scripts / "run_unit_test.sh"), str(env_script), suite],
                        env=environment, text=True, capture_output=True,
                    )
                    self.assertEqual(result.returncode, 7 if failed_label else 0,
                                     result.stdout + result.stderr)
                    records = [line.split("|", 1) for line in log.read_text().splitlines()]
                    labels = {label for label, _ in records}
                    self.assertTrue(expected <= labels, records)
                    self.assertNotIn("SKIPPED", result.stdout)
                    if suite != "device_api_host":
                        self.assertIn("rdma", labels)
                    for label, arguments in records:
                        if label == "make":
                            self.assertIn("COMPILE_KERNEL=1", arguments)
                            self.assertIn("FORCE_DEFAULT_PATH=1", arguments)
                    if suite == "device_api_unified_ir" and not failed_label:
                        intra = next(args for label, args in records
                                     if label == "unified IR intra")
                        self.assertIn("-np 4", intra)
                        self.assertNotIn("FLAGCX_P2P_DISABLE=1", intra)
                        fallback = next(args for label, args in records
                                        if label == "unified IR intra forced fallback")
                        self.assertIn("FLAGCX_DEVICE_ONE_SIDED_FORCE_NET=1", fallback)


if __name__ == "__main__":
    unittest.main()
