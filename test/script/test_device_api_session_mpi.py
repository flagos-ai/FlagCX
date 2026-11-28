"""Execute actual rank-zero reporting blocks through the MPI session checker."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which("g++"), "C++ compiler is required")
class DeviceApiSessionHeaderTest(unittest.TestCase):
    def test_header_declares_its_own_standard_library_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            # C-only MPI and a minimal yield interface must not accidentally
            # supply declarations for initializer_list or free, as some SDKs do.
            (directory / "thread").write_text(
                "#pragma once\nnamespace std { namespace this_thread { "
                "inline void yield() {} } }\n")
            (directory / "mpi.h").write_text('''
#pragma once
typedef int MPI_Comm;
#define MPI_COMM_WORLD 0
#define MPI_COMM_NULL -1
#define MPI_SUCCESS 0
#define MPI_INT 0
#define MPI_MAX 0
int MPI_Initialized(int *);
int MPI_Allreduce(const void *, void *, int, int, int, MPI_Comm);
int MPI_Bcast(void *, int, int, int, MPI_Comm);
int MPI_Comm_rank(MPI_Comm, int *);
int MPI_Comm_free(MPI_Comm *);
int MPI_Finalize();
int MPI_Abort(MPI_Comm, int);
double MPI_Wtime();
''')
            source = directory / "header.cpp"
            source.write_text('#include "device_api_test_session.hpp"\n')
            result = subprocess.run(
                ["g++", "-std=c++17", "-fsyntax-only", "-DFLAGCX_COMM_TRAITS_DEFAULT",
                 f"-I{directory}", f"-I{ROOT / 'test/fixtures'}",
                 f"-I{ROOT / 'flagcx/include'}", f"-I{ROOT / 'flagcx/adaptor/include'}",
                 str(source)], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(shutil.which("mpic++") and shutil.which("mpirun"),
                     "MPI compiler and launcher are required")
class DeviceApiSessionMpiTest(unittest.TestCase):
    def test_rank_local_diagnostic_readback_converges_on_every_rank(self):
        original = (ROOT / "test/unittest/device_api/test_device_ir_unified_intra.cpp").read_text()
        start = original.index("      flagcxResult_t readbackStatus =")
        end = original.index("      resources.check(readbackStatus);", start)
        block = original[start:end + len("      resources.check(readbackStatus);")]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "diagnostic.cpp"
            source.write_text('''
#include "device_api_test_session.hpp"
#include <cstdlib>
#define usleep(value) ((void)0)
static int readbacks = 0, injected = 0;
struct ReadbackDevice {
  flagcxResult_t deviceMemcpy(void *, void *, size_t, flagcxMemcpyType_t,
                              flagcxStream_t) {
    ++readbacks;
    return (flagcxResult_t)injected;
  }
};
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  injected = std::atoi(argv[1]);
  bool s18bOk = rank != 0;
  uint64_t hostRecvV[6] = {}, recvBuff[6] = {};
  constexpr size_t kPutValueStrideBytes = sizeof(uint64_t);
  flagcxStream_t stream = nullptr;
  ReadbackDevice device, *devHandle = &device;
  DeviceApiTestSession resources;
  int observed = 0;
  try {
''' + block + '''
  } catch (const DeviceApiTestFailure &error) {
    observed = (int)error.status;
  }
  int bad = observed != injected || readbacks != (rank == 0 ? 1 : 0);
  int common = 0;
  MPI_Allreduce(&bad, &common, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return common;
}
''')
            binary = directory / "diagnostic"
            includes = [ROOT / "test/fixtures", ROOT / "flagcx/include",
                        ROOT / "flagcx/adaptor/include"]
            compiled = subprocess.run(
                ["mpic++", "-std=c++17", "-O2", "-DFLAGCX_COMM_TRAITS_DEFAULT",
                 "-DOMPI_SKIP_MPICXX", "-DMPICH_SKIP_MPICXX",
                 *[f"-I{path}" for path in includes], str(source), "-o", str(binary)],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            env = {**os.environ, "OMPI_ALLOW_RUN_AS_ROOT": "1",
                   "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
                   "OMPI_MCA_rmaps_base_oversubscribe": "1",
                   "OMPI_MCA_pml": "ob1", "OMPI_MCA_btl": "self,tcp"}
            for ranks in (2, 4):
                for status in (0, 9):
                    with self.subTest(ranks=ranks, status=status):
                        result = subprocess.run(
                            ["mpirun", "-np", str(ranks), str(binary), str(status)],
                            env=env, capture_output=True, text=True, timeout=15)
                        self.assertEqual(result.returncode, 0,
                                         result.stdout + result.stderr)

    def test_rank_zero_reports_converge_before_other_ranks_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            includes = [ROOT / "test/fixtures", ROOT / "flagcx/include",
                        ROOT / "flagcx/adaptor/include"]
            env = {**os.environ, "OMPI_ALLOW_RUN_AS_ROOT": "1",
                   "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
                   "OMPI_MCA_rmaps_base_oversubscribe": "1",
                   "OMPI_MCA_pml": "ob1", "OMPI_MCA_btl": "self,tcp"}
            for name in ("test_device_api_inter.cpp", "test_device_ir_inter.cpp"):
                original = (ROOT / "test/unittest/device_api" / name).read_text()
                start = original.index("  flagcxResult_t reportStatus =")
                end = original.index("  resources.check(reportStatus);", start)
                block = original[start:end + len("  resources.check(reportStatus);")]
                source = directory / name
                source.write_text('''
#include "device_api_test_session.hpp"
#include <cstdlib>
static int reports = 0, injected = 0;
static flagcxResult_t deviceApiReportWindowMode(flagcxComm_t, int) {
  ++reports;
  return (flagcxResult_t)injected;
}
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int proc = 0, totalProcs = 0, color = 0, winFlags = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  MPI_Comm_size(MPI_COMM_WORLD, &totalProcs);
  injected = std::atoi(argv[1]);
  flagcxComm_t comm = nullptr;
  DeviceApiTestSession resources;
  int observed = 0;
  try {
''' + block + '''
  } catch (const DeviceApiTestFailure &error) {
    observed = (int)error.status;
  }
  int bad = observed != injected || reports != (proc == 0 ? 1 : 0);
  int common = 0;
  MPI_Allreduce(&bad, &common, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
  return common;
}
''')
                binary = directory / source.stem
                compiled = subprocess.run(
                    ["mpic++", "-std=c++17", "-O2", "-DFLAGCX_COMM_TRAITS_DEFAULT",
                     "-DOMPI_SKIP_MPICXX", "-DMPICH_SKIP_MPICXX",
                     *[f"-I{path}" for path in includes], str(source),
                     "-o", str(binary)], capture_output=True, text=True,
                    timeout=30)
                self.assertEqual(compiled.returncode, 0, compiled.stderr)
                for ranks in (2, 4):
                    for status in (0, 9):
                        with self.subTest(source=name, ranks=ranks, status=status):
                            result = subprocess.run(
                                ["mpirun", "-np", str(ranks), str(binary), str(status)],
                                env=env, capture_output=True, text=True, timeout=15)
                            self.assertEqual(result.returncode, 0,
                                             result.stdout + result.stderr)
