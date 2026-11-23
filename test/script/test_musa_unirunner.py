"""Host checks of MUSA dispatch and launch wiring; native CI tests GPU execution.

Compile the real reduction dispatch and host launch functions with a small SDK
shim. This does not validate mcc lowering, occupancy accuracy or GPU ordering.
"""

import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CXX = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
COMMON = (ROOT / "flagcx/adaptor/kernel/common/flagcx_reduce_kernel_device_impl.h").read_text()


def function(source, name):
    start = source.index(name + "(")
    begin = source.index("{", start)
    depth = 1
    end = begin + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


RUNTIME = r"""
#pragma once
#include <cstddef>
using musaStream_t = void *;
using musaError_t = int;
constexpr int musaSuccess = 0;
struct dim3 { unsigned x, y, z; dim3(unsigned x=1, unsigned y=1, unsigned z=1): x(x),y(y),z(z) {} };
enum musaDeviceAttr { musaDevAttrCooperativeLaunch, musaDevAttrConcurrentKernels,
                     musaDevAttrMultiProcessorCount, musaDevAttrMaxThreadsPerBlock };
static int cooperative=1, concurrent=1, sms=8, occupancy=2, runtimeError=0;
static const void *lastKernel;
static dim3 lastGrid, lastBlock;
static musaStream_t lastStream;
static int cooperativeCalls=0, recoveryCalls=0;
inline int musaGetLastError() { return runtimeError; }
inline int musaGetDevice(int *device) { *device=3; return runtimeError; }
inline int musaDeviceGetAttribute(int *out, musaDeviceAttr attr, int device) {
  if (device != 3) return 99;
  int values[] = {cooperative, concurrent, sms, 1024}; *out=values[attr]; return runtimeError;
}
inline int musaOccupancyMaxActiveBlocksPerMultiprocessor(int *out, const void *, int, size_t) {
  *out=occupancy; return runtimeError;
}
inline int musaLaunchCooperativeKernel(const void *kernel, dim3 grid, dim3 block,
                                      void **args, size_t bytes, musaStream_t stream) {
  if (!args || bytes != 0) return 99;
  lastKernel=kernel; lastGrid=grid; lastBlock=block; lastStream=stream;
  ++cooperativeCalls; return runtimeError;
}
inline int musaLaunchKernel(const void *kernel, dim3 grid, dim3 block,
                           void **args, size_t bytes, musaStream_t stream) {
  ++recoveryCalls;
  int result=musaLaunchCooperativeKernel(kernel,grid,block,args,bytes,stream);
  --cooperativeCalls; return result;
}
"""

PRELUDE = r"""
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include "flagcx_kernel_core.h"
#define FLAGCX_STEPS 8
#define FLAGCX_SIMPLE_MAX_NTHREADS 512
#define TRACE(...) ((void)0)
struct flagcxStream { musaStream_t base; };
struct flagcxDevMemInternal {};
using flagcxDevMem_t = flagcxDevMemInternal *;
struct flagcxDevMem {
  flagcxDevMem() = default;
  explicit flagcxDevMem(const flagcxDevMemInternal &) {}
};
void flagcxStaticReduceKernel() {}
void flagcxStaticCollectiveKernel() {}
void flagcxStaticRingDagKernel() {}
void flagcxStaticIpcRecoveryKernel() {}
"""


@unittest.skipUnless(CXX, "C++ compiler required")
class MusaUniRunnerTest(unittest.TestCase):
    def test_required_runner_case_uses_actual_platform_build_flags(self):
        source = (ROOT / "test/unittest/runner/coll_uni_runner_collectives.cpp").read_text()
        required_case = source[source.index("TEST_F(FlagCXCollTest, UniRunnerUnsupportedReductionTypes)"):
                               source.index("TEST_F(FlagCXCollTest, UniRunnerAllReduceStreamsAndInPlace)")]
        for platform, macro, should_run in (
            ("MUSA", "USE_MUSA_ADAPTOR", True),
            ("DU", "USE_DU_ADAPTOR", True),
            ("PPU", "USE_PPU_ADAPTOR", True),
            ("NVIDIA", "USE_NVIDIA_ADAPTOR", False),
        ):
            with self.subTest(platform=platform), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run(
                    ["make", "-n", "-B", "-C", str(ROOT / "test/unittest/runner"),
                     f"USE_{platform}=1", "COMPILE_KERNEL=1", f"BUILDDIR={directory}",
                     "DEVICE_HOME=/test/device-sdk", "CCL_HOME=/test/ccl-sdk",
                     "unirunner-experimental"],
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                commands = [shlex.split(line) for line in result.stdout.splitlines()
                            if " -c " in line and "coll_uni_runner_collectives.cpp" in line]
                self.assertEqual(len(commands), 1, result.stdout)
                command = commands[0]
                self.assertIn(f"-D{macro}", command)
                self.assertIn("-I/test/device-sdk/include", command)
                self.assertIn("-I/test/ccl-sdk/include", command)
                defines = [argument for argument in command if argument.startswith("-D")]
                result = subprocess.run(
                    [CXX, "-E", "-P", "-x", "c++", *defines, "-"],
                    input=required_case, capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual("flagcxAllReduce(" in result.stdout, should_run)
                self.assertEqual("GTEST_SKIP()" in result.stdout, not should_run)

    def compile_and_run(self, source):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "musa_runtime.h").write_text(RUNTIME)
            cpp = root / "test.cpp"
            cpp.write_text(PRELUDE + source)
            result = subprocess.run(
                [CXX, "-std=c++17", "-DUSE_MUSA_ADAPTOR", f"-I{root}",
                 f"-I{ROOT / 'flagcx/include'}", f"-I{ROOT / 'flagcx/adaptor/include'}",
                 str(cpp), "-o", str(root / "test")],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(root / "test")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_dynamic_launch_validates_inputs_and_returns_launch_error(self):
        launcher = function(COMMON, "flagcxLaunchCollectiveKernel")
        # Only replace GPU launch syntax; execute the real validation and
        # error handling around it, without pretending to execute a GPU kernel.
        launcher, count = re.subn(r"flagcxCollectiveKernel<<<.*?>>>\(",
                                 "flagcxCollectiveKernel(", launcher, flags=re.S)
        self.assertEqual(count, 1)
        self.compile_and_run(r"""
static int dynamicCalls=0;
void flagcxCollectiveKernel(...) { ++dynamicCalls; }
""" + "\nflagcxResult_t " + launcher + r"""
int main() {
  uint64_t fifo[32]={};
  flagcxStream stream{reinterpret_cast<void *>(0x1234)};
  flagcxDevMemInternal mem;
  auto launch=[&](void *red, size_t threads, flagcxStream_t selected) {
    return flagcxLaunchCollectiveKernel(red,nullptr,nullptr,nullptr,nullptr,
                                       nullptr,threads,1,0,2,selected);
  };
  assert(launch(fifo,256,&stream)==flagcxSuccess && dynamicCalls==1);
  assert(launch(nullptr,256,&stream)==flagcxInvalidArgument);
  assert(launch(fifo,0,&stream)==flagcxInvalidArgument);
  assert(launch(fifo,1025,&stream)==flagcxInvalidArgument);
  assert(launch(fifo,256,nullptr)==flagcxInvalidArgument);
  assert(dynamicCalls==1);
  assert(flagcxLaunchCollectiveKernel(fifo,fifo,&mem,&mem,&mem,nullptr,
                                     256,1,1,4,&stream)==flagcxSuccess);
  assert(flagcxLaunchCollectiveKernel(fifo,fifo,nullptr,&mem,&mem,nullptr,
                                     256,1,1,4,&stream)==flagcxInvalidArgument);
  runtimeError=17;
  assert(launch(fifo,256,&stream)==flagcxUnhandledDeviceError);
}
""")
    def test_reduce_dispatch_and_unsupported_types(self):
        start = COMMON.index("template <typename T, typename Enable = void>")
        end = COMMON.index("FLAGCX_DEVICE_INLINE_DECORATOR void\nloadComputeTriggerMetadata", start)
        guard = function((ROOT / "flagcx/runner/uni_runner.cc").read_text(), "ensureUniRunnerReduceTypeSupported")
        self.compile_and_run(COMMON[start:end] + "\nflagcxResult_t " + guard + r"""
template<class T> void check(flagcxDataType_t datatype) {
  T a[]={2,3,4}, b[]={4,5,6}, out[3]={};
  auto run=[&](flagcxRedOp_t op, uint64_t divisor=1) {
    flagcxReduceKernel(reinterpret_cast<uint64_t>(a), reinterpret_cast<uint64_t>(b),
                       reinterpret_cast<uint64_t>(out), 3, 1, datatype, op, divisor);
  };
  run(flagcxSum); assert(out[0]==6 && out[1]==8 && out[2]==10);
  run(flagcxProd); assert(out[0]==8 && out[1]==15 && out[2]==24);
  run(flagcxMin); assert(out[0]==2 && out[1]==3 && out[2]==4);
  run(flagcxMax); assert(out[0]==4 && out[1]==5 && out[2]==6);
  run(flagcxAvg, 2); assert(out[0]==3 && out[1]==4 && out[2]==5);
  assert(ensureUniRunnerReduceTypeSupported(datatype)==flagcxSuccess);
}
int main() {
  check<int8_t>(flagcxInt8); check<uint8_t>(flagcxUint8);
  check<int32_t>(flagcxInt32); check<uint32_t>(flagcxUint32);
  check<int64_t>(flagcxInt64); check<uint64_t>(flagcxUint64);
  check<float>(flagcxFloat32); check<double>(flagcxFloat64);
  assert(ensureUniRunnerReduceTypeSupported(flagcxFloat16)==flagcxNotSupported);
  assert(ensureUniRunnerReduceTypeSupported(flagcxBfloat16)==flagcxNotSupported);
  assert(uniRunnerReduceTraits<int32_t>::sum(INT32_MAX,1)==INT32_MIN);
  assert(uniRunnerReduceTraits<uint64_t>::sum(UINT64_MAX,1)==0);
}
""")

    def test_static_launches_residency_and_runtime_errors(self):
        start = COMMON.index("#elif defined(USE_MUSA_ADAPTOR)")
        end = COMMON.index("#elif defined(USE_PPU_ADAPTOR)", start)
        mapping = COMMON[start:end].replace("#elif", "#if", 1) + "\n#endif\n"
        residency = function((ROOT / "flagcx/runner/uni_runner_dag_plan.cc").read_text(), "resolveUniRunnerStaticExecutorResidencyBudget")
        launchers = COMMON[COMMON.index("static flagcxResult_t\ngetStaticKernelMaxExecutorBlocks"):]
        self.compile_and_run(mapping + "\nflagcxResult_t " + residency + launchers + r"""
int main() {
  size_t budget=99;
  assert(flagcxGetStaticReduceKernelMaxExecutorBlocks(256,&budget)==flagcxSuccess);
  assert(budget==7);
  cooperative=0;
  assert(flagcxGetStaticCollectiveKernelMaxExecutorBlocks(256,&budget)==flagcxNotSupported && budget==0);
  cooperative=1; concurrent=0;
  assert(flagcxGetStaticRingDagKernelMaxExecutorBlocks(256,&budget)==flagcxNotSupported);
  concurrent=1;
  assert(flagcxGetStaticReduceKernelMaxExecutorBlocks(2048,&budget)==flagcxInvalidArgument);
  runtimeError=17;
  assert(flagcxGetStaticReduceKernelMaxExecutorBlocks(256,&budget)==flagcxUnhandledDeviceError);
  runtimeError=0;
  uint64_t fifo[256]={}, done=0, abort=0;
  flagcxStream stream{reinterpret_cast<void *>(0x1234)};
  assert(flagcxLaunchStaticReduceKernel(fifo,2,256,2,7,4,&stream,0,0)==flagcxSuccess);
  assert(lastKernel==reinterpret_cast<void *>(flagcxStaticReduceKernel));
  assert(lastGrid.x==2 && lastBlock.x==256 && lastStream==stream.base);
  int calls=cooperativeCalls;
  assert(flagcxLaunchStaticReduceKernel(fifo,2,256,2,1,4,&stream,0,0)==flagcxInvalidArgument);
  assert(cooperativeCalls==calls);
  flagcxDevMemInternal mem;
  flagcxStaticIpcControl control{};
  control.epoch=1; control.localRanks=4; control.localRank=0;
  control.doneBase=1; control.readyDataOffset=5;
  assert(flagcxLaunchStaticCollectiveKernel(fifo,2,fifo,2,&mem,&mem,&mem,nullptr,0,
             256,2,2,7,4,&control,&stream)==flagcxSuccess);
  assert(lastKernel==reinterpret_cast<void *>(flagcxStaticCollectiveKernel) && lastGrid.x==4);
  flagcxRingLaneDesc lanes[1]{};
  flagcxRingStepTrigger triggers[1]{};
  assert(flagcxLaunchStaticRingDagKernel(lanes,1,triggers,1,fifo,fifo,&mem,&mem,&mem,
             1024,flagcxFloat32,flagcxSum,4,&done,&abort,&control,256,7,&stream)==flagcxSuccess);
  assert(lastKernel==reinterpret_cast<void *>(flagcxStaticRingDagKernel));
  assert(flagcxLaunchStaticIpcRecoveryKernel(&mem,&control,&stream)==flagcxSuccess);
  assert(lastKernel==reinterpret_cast<void *>(flagcxStaticIpcRecoveryKernel) && recoveryCalls==1);
  control.epoch=0;
  assert(flagcxLaunchStaticIpcRecoveryKernel(&mem,&control,&stream)==flagcxInvalidArgument);
  runtimeError=17;
  assert(flagcxLaunchStaticReduceKernel(fifo,2,256,2,7,4,&stream,0,0)==flagcxUnhandledDeviceError);
}
""")


if __name__ == "__main__":
    unittest.main()
