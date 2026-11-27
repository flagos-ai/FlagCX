"""Host checks of MetaX runtime mappings and error handling with SDK stubs.

These checks do not build MACA kernels or validate GPU ordering/residency.
"""

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from test_adaptor_contracts import production_function


ROOT = Path(__file__).resolve().parents[2]
CXX = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")

RUNTIME = r"""
#pragma once
#include <cassert>
#include <cstddef>
#include <cstdint>
using mcStream_t = void *;
enum mcError_t {
  mcSuccess = 0, mcErrorNotSupported = 41, mcErrorInvalidValue = 42,
  mcErrorLaunchOutOfResources = 43
};
enum class mcStreamWriteValue_flags { MC_STREAM_WRITE_VALUE_DEFAULT = 7 };
enum mcDeviceAttribute_t {
  mcDeviceAttributeCooperativeLaunch, mcDeviceAttributeConcurrentKernels,
  mcDeviceAttributeMultiProcessorCount, mcDeviceAttributeMaxThreadsPerBlock
};
inline mcError_t mcGetDevice(int *device) {
  *device = 3;
  return mcSuccess;
}
inline mcError_t mcDeviceGetAttribute(int *out, mcDeviceAttribute_t attr,
                                     int device) {
  assert(device == 3);
  const int values[] = {1, 1, 8, 1024};
  *out = values[attr];
  return mcSuccess;
}
inline mcError_t mcOccupancyMaxActiveBlocksPerMultiprocessor(
    int *out, const void *, int, size_t) {
  *out = 2;
  return mcSuccess;
}
static mcError_t writeResult = mcSuccess, launchError = mcSuccess;
static int writeCalls = 0, errorQueries = 0;
inline mcError_t mcStreamWriteValue64(mcStream_t stream, void *address,
                                     uint64_t, mcStreamWriteValue_flags flags) {
  assert(stream && address);
  assert(flags == mcStreamWriteValue_flags::MC_STREAM_WRITE_VALUE_DEFAULT);
  ++writeCalls;
  return writeResult;
}
inline mcError_t mcGetLastError() {
  ++errorQueries;
  auto result = launchError;
  launchError = mcSuccess;
  return result;
}
"""

PRELUDE = r"""
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include "flagcx_kernel_core.h"
struct flagcxStream { mcStream_t base; };
struct flagcxDevMemInternal {};
using flagcxDevMem_t = flagcxDevMemInternal *;
struct flagcxDevMem {
  flagcxDevMem() = default;
  explicit flagcxDevMem(const flagcxDevMemInternal &) {}
};
// Logging may call runtime helpers; capture the launch error before logging.
// printf also checks the production format string and argument expressions.
#define INFO(flags, ...) (launchError = mcSuccess, std::printf(__VA_ARGS__))
"""


@unittest.skipUnless(CXX, "C++ compiler required")
class MetaxUniRunnerErrorTest(unittest.TestCase):
    def compile_and_run(self, source):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mcr").mkdir()
            (root / "mcr/mc_runtime.h").write_text(RUNTIME)
            cpp = root / "test.cpp"
            cpp.write_text(PRELUDE + source)
            result = subprocess.run(
                [CXX, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                 "-Wno-unused-variable", "-Wno-unused-function",
                 "-Wno-unused-parameter",
                 "-DUSE_METAX_ADAPTOR", f"-I{root}",
                 f"-I{ROOT / 'flagcx/include'}",
                 f"-I{ROOT / 'flagcx/adaptor/include'}",
                 str(cpp), "-o", str(root / "test")],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(root / "test")], capture_output=True,
                                    text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_runtime_mapping_compiles_and_queries_residency(self):
        source = (ROOT / "flagcx/adaptor/kernel/common/flagcx_reduce_kernel_device_impl.h").read_text()
        start = source.index("#elif defined(USE_METAX_ADAPTOR)")
        end = source.index("#elif defined(USE_MUSA_ADAPTOR)", start)
        mapping = source[start:end].replace("#elif", "#if", 1) + "\n#endif\n"
        residency = production_function(
            (ROOT / "flagcx/runner/uni_runner_dag_plan.cc").read_text(),
            "resolveUniRunnerStaticExecutorResidencyBudget",
        )
        query = production_function(source, "getStaticKernelMaxExecutorBlocks")
        self.compile_and_run(mapping + residency + query + r"""
int main() {
  size_t budget = 99;
  assert(getStaticKernelMaxExecutorBlocks(nullptr, "test", 256, &budget)
         == flagcxSuccess);
  assert(budget == 7);
  assert(getStaticKernelMaxExecutorBlocks(nullptr, "test", 2048, &budget)
         == flagcxInvalidArgument);
  assert(budget == 0);
}
""")

    def test_write_preserves_only_explicit_unsupported(self):
        source = (ROOT / "flagcx/adaptor/device/maca_adaptor.cc").read_text()
        callback = production_function(source, "macaAdaptorStreamWriteValue64")
        self.compile_and_run(callback + r"""
int main() {
  flagcxStream stream{reinterpret_cast<void *>(0x1234)};
  uint64_t value = 0;
  assert(macaAdaptorStreamWriteValue64(&stream, &value, 1, 0) == flagcxSuccess);
  writeResult = mcErrorNotSupported;
  assert(macaAdaptorStreamWriteValue64(&stream, &value, 1, 0) == flagcxNotSupported);
  for (auto error : {mcErrorInvalidValue, mcErrorLaunchOutOfResources}) {
    writeResult = error;
    assert(macaAdaptorStreamWriteValue64(&stream, &value, 1, 0) == flagcxUnhandledDeviceError);
  }
  assert(writeCalls == 4);
  assert(macaAdaptorStreamWriteValue64(nullptr, &value, 1, 0) == flagcxInvalidArgument);
  assert(macaAdaptorStreamWriteValue64(&stream, nullptr, 1, 0) == flagcxInvalidArgument);
  assert(writeCalls == 4);
}
""")

    def test_dynamic_launch_returns_error_before_completion_wait(self):
        source = (ROOT / "flagcx/adaptor/kernel/common/flagcx_reduce_kernel_device_impl.h").read_text()
        launcher = production_function(source, "flagcxLaunchCollectiveKernel")
        # Replace only the GPU submission syntax. Keep real host validation,
        # error queries and return handling; no GPU kernel is executed here.
        launcher, count = re.subn(r"flagcxCollectiveKernel<<<.*?>>>\(",
                                 "flagcxCollectiveKernel(", launcher, flags=re.S)
        self.assertEqual(count, 1)
        self.compile_and_run(r"""
static int launches = 0;
void flagcxCollectiveKernel(...) { ++launches; }
""" + launcher + r"""
int main() {
  uint64_t fifo[32] = {};
  flagcxStream stream{reinterpret_cast<void *>(0x1234)};
  auto launch = [&]() {
    return flagcxLaunchCollectiveKernel(fifo, nullptr, nullptr, nullptr,
                                       nullptr, nullptr, 256, 1, 0, 1, &stream);
  };
  assert(launch() == flagcxSuccess);
  assert(launches == 1 && errorQueries == 1);
  launchError = mcErrorLaunchOutOfResources;
  assert(launch() == flagcxUnhandledDeviceError);
  assert(launches == 2 && errorQueries == 2);
  assert(launchError == mcSuccess);
  assert(launch() == flagcxSuccess);
  assert(launches == 3 && errorQueries == 3);
}
""")


if __name__ == "__main__":
    unittest.main()
