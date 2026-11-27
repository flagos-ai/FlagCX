"""Run the real NCCL wrapper against deterministic host-only device shims."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which("g++"), "C++ compiler is required")
class NcclWrapperCompletionTest(unittest.TestCase):
    def test_async_grouped_completion_and_failed_cleanup_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / "cuda_runtime.h").write_text("#pragma once\ntypedef void *cudaStream_t;\n")
            for version in (22100, 22703):
                (directory / "nccl.h").write_text('''
#pragma once
#include <cuda_runtime.h>
#define NCCL_VERSION(X,Y,Z) ((X)*10000+(Y)*100+(Z))
#define NCCL_VERSION_CODE ''' + str(version) + '''
#define NCCL_UNIQUE_ID_BYTES 128
typedef struct ncclComm *ncclComm_t;
typedef void *ncclWindow_t;
typedef struct { char internal[128]; } ncclUniqueId;
typedef struct {} ncclConfig_t;
typedef struct {} ncclSimInfo_t;
typedef int ncclScalarResidence_t;
typedef enum { ncclSuccess, ncclUnhandledCudaError, ncclSystemError,
 ncclInternalError, ncclInvalidArgument, ncclInvalidUsage, ncclRemoteError,
 ncclInProgress } ncclResult_t;
typedef enum { ncclInt8, ncclUint8, ncclInt32, ncclUint32, ncclInt64,
 ncclUint64, ncclFloat16, ncclFloat32, ncclFloat64, ncclBfloat16 } ncclDataType_t;
typedef enum { ncclSum, ncclProd, ncclMax, ncclMin, ncclAvg, ncclNumOps } ncclRedOp_t;
''')
                binary = directory / f"completion-{version}"
                link_flags = ["-Wl,-dead_strip"] if sys.platform == "darwin" else ["-Wl,--gc-sections", "-ldl"]
                built = subprocess.run(
                    ["g++", "-std=c++11", "-O2", "-pthread", "-ffunction-sections",
                     "-fdata-sections", '-DREAL_NCCL_PATH="/mock/nccl.so"',
                     f"-I{directory}", f"-I{ROOT / 'flagcx/include'}",
                     f"-I{ROOT / 'plugin/nccl/src'}",
                     str(ROOT / "test/fixtures/nccl_wrapper_completion_test.cpp"),
                     *link_flags, "-o", str(binary)], capture_output=True, text=True, timeout=45)
                self.assertEqual(built.returncode, 0, built.stderr)
                ran = subprocess.run([str(binary)], capture_output=True, text=True, timeout=15)
                self.assertEqual(ran.returncode, 0, ran.stdout + ran.stderr)
