"""Fault injection for diagnostic completion, not MACA hardware semantics."""

import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "test/diagnostics/metax_alloc_memops.py"
SPEC = importlib.util.spec_from_file_location("metax_alloc_probe", SCRIPT)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)
CXX = shutil.which("clang++") or shutil.which("g++")

# Enum constants (not macros), one deliberately absent option, and independent
# API/allocation fault modes. The real native worker is compiled against this.
RUNTIME = r"""
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <unistd.h>
using mcError_t = int;
using mcStream_t = void*;
constexpr int mcSuccess=0, mcErrorNotSupported=801;
constexpr unsigned mcStreamNonBlocking=1;
enum Alloc { mcDeviceMallocDefault=0, mcDeviceMallocFinegrained=1,
  mcMallocSignalMemory=2, mcDeviceMallocWriteCoherence=3,
  mcDeviceMallocMapPcieDefault=4, mcDeviceMallocMapPcieCoherence=5,
  mcDeviceMallocFixedMemCoherence=7 };
enum mcStreamWriteValue_flags { MC_STREAM_WRITE_VALUE_DEFAULT=0 };
enum mcStreamWaitValue_flags { MC_STREAM_WAIT_VALUE_GEQ=0 };
enum Kind { mcMemcpyHostToDevice, mcMemcpyDeviceToHost };
struct mcDeviceProp_t { char name[32]; };
struct mcPointerAttribute_t {
  void *devicePointer, *hostPointer;
  int device, memoryType=2, isManaged=0;
};
static int allocationMode=-1;
static size_t allocationBytes=0;
inline const char *mcGetErrorString(int e) { return e==0 ? "mcSuccess" : "mcErrorInvalidValue"; }
inline int mcRuntimeGetVersion(int *v) { *v=3800; return 0; }
inline int mcDriverGetVersion(int *v) { *v=3900; return 0; }
inline int mcSetDevice(int) { return 0; }
inline int mcGetDeviceProperties(mcDeviceProp_t *p,int) { strcpy(p->name,"FAKE_RUNTIME"); return 0; }
inline int mcMalloc(void **p,size_t n) { allocationMode=-1; *p=malloc(n); return 0; }
inline int mcExtMallocWithFlags(void **p,size_t n,unsigned f) {
  allocationMode=int(f); allocationBytes=n;
  if(f==3 && n>8) for(;;) pause();
  *p=malloc(n); return 0;
}
inline int mcFree(void *p) {
  if(allocationMode==5 && allocationBytes>8) for(;;) pause();
  free(p); return allocationMode==5 ? 1 : 0;
}
inline int mcPointerGetAttributes(mcPointerAttribute_t *a,void *p) {
  *a={p,nullptr,0}; return 0;
}
inline int mcStreamCreateWithFlags(mcStream_t *s,unsigned) { *s=(void*)1; return 0; }
inline int mcDeviceSynchronize() { return 0; }
inline int mcStreamDestroy(mcStream_t) { return 0; }
inline int mcStreamSynchronize(mcStream_t) { return 0; }
inline int mcMemcpy(void *d,const void *s,size_t n,Kind) { memcpy(d,s,n); return 0; }
inline int mcStreamWriteValue32(mcStream_t,void *p,uint32_t v,mcStreamWriteValue_flags) {
  if(allocationMode==3) raise(SIGABRT);
  memcpy(p,&v,4); return 0;
}
inline int mcStreamWriteValue64(mcStream_t,void *p,uint64_t v,mcStreamWriteValue_flags) {
  if(allocationMode==1) return 1;
  if(allocationMode==3) raise(SIGABRT);
  if(allocationMode==4) v=uint32_t(v);
  memcpy(p,&v,8); return 0;
}
inline int fakeWait() { if(allocationMode==2) for(;;) pause(); return 0; }
inline int mcStreamWaitValue32(mcStream_t,void*,uint32_t,mcStreamWaitValue_flags) { return fakeWait(); }
#ifndef FAKE_NO_WAIT64
inline int mcStreamWaitValue64(mcStream_t,void*,uint64_t,mcStreamWaitValue_flags) { return fakeWait(); }
#endif
"""


class SupervisorTest(unittest.TestCase):
    def test_timeout_retains_output_and_kills_descendants_without_pipe_wait(self):
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "hang.log"
            code = (
                'import os,signal,time; '
                'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                'print("before hang",flush=True); '
                'os.fork(); time.sleep(30)'
            )
            start = time.monotonic()
            outcome = PROBE.run_bounded([sys.executable, "-c", code], log, 0.15)
            self.assertTrue(outcome["timed_out"])
            self.assertTrue(outcome["reaped"])
            self.assertLess(time.monotonic() - start, 2)
            self.assertIn("before hang", log.read_text())
            next_case = PROBE.run_bounded(
                [sys.executable, "-c", 'print("next case")'],
                Path(temp) / "next.log", 2)
            self.assertEqual(next_case["returncode"], 0)

    def test_launch_failure_is_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            result = PROBE.run_bounded(
                [str(Path(temp) / "nonexistent")], Path(temp) / "launch.log", 1)
            self.assertIn("launch_error", result)

    def test_missing_sdk_still_reports_every_strategy(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--sdk", str(root / "missing"),
                 "--output-dir", str(root / "output")],
                capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            report = json.loads((root / "output/results.json").read_text())
            self.assertEqual(len(report["cases"]), 108)
            self.assertEqual({c["mode"] for c in report["cases"]}, set(PROBE.MODES))
            self.assertEqual({c["status"] for c in report["cases"]}, {"NOT_RUN_SETUP_ERROR"})
            self.assertEqual(report["launched"], 0)
            self.assertFalse(report["complete"])

    def test_hung_compiler_still_writes_complete_result_table(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            compiler = root / "hung_compiler.py"
            compiler.write_text(
                'import sys,time\n'
                'if "--version" in sys.argv: print("FAKE"); sys.exit(0)\n'
                'time.sleep(30)\n')
            start = time.monotonic()
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--cxx",
                 shlex.join([sys.executable, str(compiler)]),
                 "--compile-timeout", "0.15", "--sdk", str(root / "sdk"),
                 "--output-dir", str(root / "output")],
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertLess(time.monotonic() - start, 5)
            report = json.loads((root / "output/results.json").read_text())
            self.assertTrue(report["build_checks"]["base"]["timed_out"])
            self.assertEqual(len(report["cases"]), 108)

    @unittest.skipUnless(CXX, "C++ compiler required")
    def test_native_worker_matrix_continues_after_every_failure_type(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            headers = root / "sdk/include/mcr"
            headers.mkdir(parents=True)
            (headers / "mc_runtime.h").write_text(RUNTIME)
            # Header-only model; an empty archive satisfies the normal link.
            library = root / "sdk/lib"
            library.mkdir()
            (library / "libmcruntime.a").write_bytes(b"!<arch>\n")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--sdk", str(root / "sdk"),
                 "--cxx", CXX, "--case-timeout", "0.15",
                 "--output-dir", str(root / "output")],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            report = json.loads((root / "output/results.json").read_text())
            self.assertNotIn("setup_error", report)
            self.assertTrue(report["complete"])
            self.assertEqual(len(report["cases"]), 108)
            self.assertEqual(report["launched"], 96)
            self.assertFalse(any(c["status"].startswith("NOT_RUN") for c in report["cases"]))

            def case(mode, width, operation):
                return next(c for c in report["cases"] if c["mode"] == mode
                            and c["size"] == 8 and c["width"] == width
                            and c["operation"] == operation)

            self.assertEqual(case("mcMalloc", 64, "write_wait")["status"], "PASS")
            failure = case("mcDeviceMallocFinegrained", 64, "write_wait")
            self.assertEqual(failure["status"], "API_ERROR")
            self.assertNotIn("wait", [s["stage"] for s in failure["stages"]])
            self.assertEqual(case("mcDeviceMallocFinegrained", 64, "wait")["status"], "PASS")
            timeout = case("mcMallocSignalMemory", 64, "wait")
            self.assertEqual(timeout["status"], "TIMEOUT")
            self.assertIn("INFO\tpointer", Path(timeout["log"]).read_text())
            crash = case("mcDeviceMallocWriteCoherence", 32, "write")
            self.assertEqual(crash["status"], "CRASH_OR_EXIT_ERROR")
            self.assertLess(crash["returncode"], 0)
            self.assertEqual(case("mcDeviceMallocMapPcieDefault", 64, "write")["status"], "DATA_MISMATCH")
            self.assertEqual(case("mcDeviceMallocMapPcieCoherence", 64, "write")["status"], "CLEANUP_ERROR")
            self.assertEqual(case("mcDeviceMallocFixedMemDefault", 64, "write")["status"], "SDK_OPTION_ABSENT")
            self.assertEqual(case("mcDeviceMallocFixedMemCoherence", 64, "write_wait")["status"], "PASS")
            cleanup_timeout = next(c for c in report["cases"]
                                   if c["mode"] == "mcDeviceMallocMapPcieCoherence"
                                   and c["size"] > 8 and c["operation"] == "write")
            self.assertEqual(cleanup_timeout["status"], "TIMEOUT")
            self.assertEqual(cleanup_timeout["op_result"], "PASS")
            allocation_timeout = next(c for c in report["cases"]
                                      if c["mode"] == "mcDeviceMallocWriteCoherence"
                                      and c["size"] > 8)
            self.assertEqual(allocation_timeout["status"], "TIMEOUT")
            self.assertEqual(report["device_info"]["info"]["gpu"], "FAKE_RUNTIME")
            self.assertEqual(len(report["headers"]), 1)
            summary = (root / "output/summary.md").read_text()
            for mode in PROBE.MODES:
                self.assertIn(mode, summary)

    @unittest.skipUnless(CXX, "C++ compiler required")
    def test_missing_wait64_does_not_hide_write64_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            headers = root / "sdk/include/mcr"
            headers.mkdir(parents=True)
            (headers / "mc_runtime.h").write_text("#define FAKE_NO_WAIT64\n" + RUNTIME)
            library = root / "sdk/lib"
            library.mkdir()
            (library / "libmcruntime.a").write_bytes(b"!<arch>\n")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--sdk", str(root / "sdk"),
                 "--cxx", CXX, "--case-timeout", "0.15",
                 "--output-dir", str(root / "output")],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            report = json.loads((root / "output/results.json").read_text())
            self.assertTrue(report["complete"])
            self.assertTrue(report["features"]["WRITE64"])
            self.assertFalse(report["features"]["WAIT64"])
            baseline = [c for c in report["cases"] if c["mode"] == "mcMalloc"
                        and c["width"] == 64]
            for case in baseline:
                self.assertEqual(case["status"], "PASS" if case["operation"] == "write"
                                 else "API_BUILD_UNAVAILABLE")

    def test_ci_hook_runs_before_platform_commands_and_preserves_suite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            diagnostic = root / "test/diagnostics"
            diagnostic.mkdir(parents=True)
            (diagnostic / SCRIPT.name).write_text(
                'import os\nfrom pathlib import Path\n'
                'Path(os.environ["PROBE_MARKER"]).write_text("executed")\n'
                'raise SystemExit(1)\n')
            marker = root / "marker"
            # Override command lookup only, without requiring a local SDK/MPI.
            command = ('source "$1"; command() { return 0; }; '
                       'flagcx_ci_prepare adaptor; printf "SUITE_CONTINUES\\n"')
            result = subprocess.run(
                ["bash", "-c", command, "bash", str(ROOT / ".github/scripts/set_env/metax.sh")],
                env=dict(os.environ, PROJECT_ROOT=str(root), PROBE_MARKER=str(marker)),
                capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(marker.exists())
            self.assertIn("warning title=MetaX allocation diagnostic", result.stdout)
            self.assertIn("SUITE_CONTINUES", result.stdout)


if __name__ == "__main__":
    unittest.main()
