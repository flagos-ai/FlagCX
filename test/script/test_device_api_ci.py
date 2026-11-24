"""Check native Device API MPI coverage and the MUSA suite configuration."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CI = ROOT / ".github/scripts/ci"
SPEC = importlib.util.spec_from_file_location("device_api_checked", CI / "run_device_api_checked.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def result_log(binary, ranks, sizes):
    lines = []
    for size in sizes:
        lines.append(f"# Size = {size} bytes, count = {size // 4}")
        for rank in range(ranks):
            for case in CHECK.REQUIRED[binary]:
                lines.append(f"[rank {rank}] {case} operation: PASS")
    if "unified" in binary:
        lines.append("=== Final Result: ALL PASS ===")
    else:
        lines += [f"[rank {rank}] === Overall: PASS ===" for rank in range(ranks)]
    return "\n".join(lines) + "\n"


class DeviceApiCiTest(unittest.TestCase):
    def test_musa_configuration_rejects_non_four_gpu_layouts(self):
        for devices in ("0,1", "0,1,2,3,4,5,6,7"):
            result = subprocess.run(
                ["bash", "-eu", "-c", 'source "$ENV_SCRIPT"'],
                env={**os.environ, "ENV_SCRIPT": str(ROOT / ".github/scripts/set_env/mthreads.sh"),
                     "FLAGCX_CI_MUSA_DEVICES": devices}, capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exactly four", result.stderr)

    def test_host_isolation_gate_requires_both_cases_to_execute(self):
        source = (CI / "run_unit_test.sh").read_text()
        function = source[source.index("run_device_api_host() {"):source.index("flagcx_ci_device_api_intra_only() {")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / ".github/scripts/ci"
            scripts.mkdir(parents=True)
            (scripts / "check_required_gtests.py").write_text((CI / "check_required_gtests.py").read_text())
            timeout = root / "timeout"
            timeout.write_text('#!/bin/bash\nexec "$@"\n')
            timeout.chmod(0o755)
            binary = root / "bin"
            binary.mkdir()
            make = binary / "make"
            make.write_text(f"#!{sys.executable}\n" + '''import os, pathlib, sys, xml.etree.ElementTree as ET
prefix = "FLAGCX_CI_GTEST_XML_DIR="
path = pathlib.Path(next(a[len(prefix):] for a in sys.argv if a.startswith(prefix)))
suite = ET.Element("testsuite", name="CommStreamIsolation")
for name in ("FinalizeDoesNotDrainIndependentCommunicator", "PartialOwnerCachesDoNotDrainIndependentCommunicator"):
    if os.getenv("XML_MODE") == "missing" and name.startswith("Partial"):
        continue
    case = ET.SubElement(suite, "testcase", name=name, status="run", result="completed")
    if os.getenv("XML_MODE") == "skip": ET.SubElement(case, "skipped")
    if os.getenv("XML_MODE") == "fail": ET.SubElement(case, "failure")
ET.ElementTree(suite).write(path / "host-default.xml")
''')
            make.chmod(0o755)
            rma = root / "test/unittest/device_api/build/bin/test_dev_comm_cleanup"
            rma.parent.mkdir(parents=True)
            rma.write_text(f"#!{sys.executable}\n" + '''import os, pathlib, sys, xml.etree.ElementTree as ET
assert os.environ["FLAGCX_VMM_ENABLE"] == "0"
assert os.environ["GTEST_FILTER"] == "CommStreamIsolation.HeteroRmaNonVmmFinalizeDoesNotDrainIndependentWork"
mode = os.environ["XML_MODE"]
if mode == "rma_missing": sys.exit(0)
output = next(a.split("xml:", 1)[1] for a in sys.argv if a.startswith("--gtest_output=xml:"))
suite = ET.Element("testsuite", name="CommStreamIsolation")
case = ET.SubElement(suite, "testcase", name="HeteroRmaNonVmmFinalizeDoesNotDrainIndependentWork", status="run", result="completed")
if mode == "rma_skip": ET.SubElement(case, "skipped")
if mode == "rma_fail": ET.SubElement(case, "failure")
ET.ElementTree(suite).write(output)
''')
            rma.chmod(0o755)
            for mode in ("pass", "skip", "missing", "fail", "rma_missing", "rma_skip", "rma_fail"):
                with self.subTest(mode=mode):
                    command = '''set -euo pipefail
FLAGCX_CI_TEST_MAKE_ARGS=(USE_MUSA=1)
SET_ENV_SCRIPT=mthreads.sh
SUITE=device_api_host
flagcx_ci_phase_supported() { return 0; }
suite_log_dir=$PROJECT_ROOT/logs
mkdir -p "$suite_log_dir"
''' + function + "\nrun_device_api_host"
                    result = subprocess.run(
                        ["bash", "-c", command],
                        env={**os.environ, "PROJECT_ROOT": str(root), "TEST_RUNNER": str(timeout),
                             "PATH": f"{binary}:{os.environ['PATH']}", "XML_MODE": mode},
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode == 0, mode == "pass", result.stderr)

    def test_native_test_kernels_keep_host_cpp_header_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("device_api", "device_ir"):
                with self.subTest(kernel=name):
                    result = subprocess.run(
                        ["make", "-n", "-B", "-C", str(ROOT / "test/kernel/musa"),
                         f"{directory}/obj/{name}.device.o", f"BUILDDIR={directory}",
                         "USE_MUSA=1", "COMPILE_KERNEL=1", "DEVICE_HOME=/fixture/musa", "MUSA_RESOURCE_DIR=/fixture/resource",
                         "MUSA_TEST_CXX_ISYSTEM=-isystem /fixture/host-cxx"],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("-isystem /fixture/host-cxx", result.stdout)
                    self.assertLess(result.stdout.index("-isystem /fixture/resource/include/cuda_wrappers"),
                                    result.stdout.index("-isystem /fixture/host-cxx"))
                    self.assertIn("--cuda-device-only", result.stdout)
                    self.assertIn("-DCOMPILE_KERNEL", result.stdout)

    def test_missing_musa_compiler_fails_instead_of_skipping(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                ["bash", "-c", '''set -euo pipefail
source "$ENV_SCRIPT"
flagcx_ci_configure_suite device_api
if declare -F flagcx_ci_skip_suite >/dev/null; then exit 99; fi
mpirun() { :; }
g++() { :; }
flagcx_ci_prepare device_api
'''],
                env={**os.environ, "ENV_SCRIPT": str(ROOT / ".github/scripts/set_env/mthreads.sh"),
                     "DEVICE_HOME": directory}, capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0, result.stderr)
            self.assertNotEqual(result.returncode, 99, result.stderr)
            self.assertIn("Building native Device API", result.stdout)

    def test_per_rank_and_sweep_coverage_for_all_six_programs(self):
        for binary in CHECK.REQUIRED:
            with self.subTest(binary=binary):
                good = result_log(binary, 4, [1024, 2048])
                self.assertEqual(CHECK.check_results(good, binary, 4, [1024, 2048]), [])
                case = CHECK.REQUIRED[binary][0]
                required = f"[rank 3] {case} operation: PASS"
                for bad in ("", good.replace(required, "", 1),
                            good.replace(required, required.replace("PASS", "SKIP"), 1),
                            good.replace(required, required.replace("PASS", "FAIL"), 1),
                            good + required, good.replace("Size = 2048", "Size = 4096"),
                            good.replace("Overall: PASS", "Overall: FAIL"),
                            good.replace("ALL PASS", "NO CASES GRADED"),
                            good + "[rank 0] S1 optional: FAIL\n"):
                    if bad == good:
                        continue
                    self.assertTrue(CHECK.check_results(bad, binary, 4, [1024, 2048]))

    def test_mpi_app_contexts_must_match(self):
        first = ["-np", "2", "build/bin/test_device_api_inter", "-b", "1K", "-e", "16M", "-f", "2", "-R", "2"]
        binary, ranks, sizes = CHECK.invocation([*first, ":", *first])
        self.assertEqual((binary, ranks, len(sizes)), ("test_device_api_inter", 4, 15))
        for args in ([], first[:-4], [*first, ":", *first[:-1], "1" , "-b", "2K"],
                     [*first, ":", *["test_device_api_intra" if x.endswith("test_device_api_inter") else x for x in first]],
                     ["-np", "0", *first[2:]], [x if x != "16M" else "1" for x in first]):
            with self.subTest(args=args), self.assertRaises((ValueError, IndexError)):
                CHECK.invocation(args)

    def test_checked_runner_rounds_and_mpi_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = root / "run_device_api_checked.py"
            runner.write_text((CI / runner.name).read_text())
            mpi = root / "run_mpi_with_timeout.sh"
            mpi.write_text(f"#!{sys.executable}\n" + '''import os, pathlib, sys
with open(os.environ["CAPTURE"], "a") as out:
    out.write(os.environ["FLAGCX_CI_TEST_MAX_ATTEMPTS"] + "\\n")
print(pathlib.Path(os.environ["RESULT_LOG"]).read_text(), end="")
sys.exit(int(os.environ.get("MPI_STATUS", "0")))
''')
            mpi.chmod(0o755)
            log = root / "fixture.log"
            capture = root / "calls"
            binary = "test_device_ir_unified_inter"
            app = ["-np", "2", f"build/bin/{binary}", "-b", "1K", "-e", "2K", "-f", "2"]
            for rounds, status, missing in ((3, 0, False), (3, 19, False), (1, 0, True), (0, 0, False)):
                with self.subTest(rounds=rounds, status=status, missing=missing):
                    log.write_text("" if missing else result_log(binary, 4, [1024, 2048]))
                    capture.write_text("")
                    result = subprocess.run([sys.executable, str(runner), *app, ":", *app],
                                            env={**os.environ, "CAPTURE": str(capture), "RESULT_LOG": str(log),
                                                 "MPI_STATUS": str(status), "FLAGCX_CI_DEVICE_API_ROUNDS": str(rounds),
                                                 "FLAGCX_CI_LOG_DIR": str(root / "logs")},
                                            text=True, capture_output=True)
                    expected = 2 if rounds == 0 else status or (1 if missing else 0)
                    self.assertEqual(result.returncode, expected, result.stderr)
                    calls = capture.read_text().splitlines()
                    self.assertEqual(calls, ["1"] * (0 if rounds == 0 else 1 if status or missing else rounds))

    def test_musa_shared_runner_uses_native_programs_and_four_ranks(self):
        source = (CI / "run_unit_test.sh").read_text()
        functions = source[source.index("flagcx_ci_device_api_intra_only() {"):source.index("run_suite() {")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / ".github/scripts/ci"
            scripts.mkdir(parents=True)
            (scripts / "ignore_test.py").write_text((CI / "ignore_test.py").read_text())
            (root / "test/unittest/device_api").mkdir(parents=True)
            mpi = root / "mpi"
            mpi.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
with open(os.environ["CAPTURE"], "a") as out:
    out.write(json.dumps({"args": sys.argv[1:], "rounds": os.environ["FLAGCX_CI_DEVICE_API_ROUNDS"]}) + "\\n")
''')
            mpi.chmod(0o755)
            capture = root / "calls"
            for count in (4,):
                for suite in ("device_api", "device_api_unified_ir"):
                    with self.subTest(count=count, suite=suite):
                        capture.write_text("")
                        command = '''set -euo pipefail
source "$ENV_SCRIPT"
flagcx_ci_configure_suite "$SUITE"
MPI_RUNNER=/must-not-run
FLAGCX_CI_DEVICE_API_MPI_RUNNER=$MOCK_MPI
SET_ENV_SCRIPT=mthreads.sh
''' + functions + '\nrun_' + suite
                        env = {k: v for k, v in os.environ.items()
                               if k not in ("FLAGCX_CI_DEVICE_API_ROUNDS", "UNIFIED_IR_NP")}
                        env.update(ENV_SCRIPT=str(ROOT / ".github/scripts/set_env/mthreads.sh"),
                                   SUITE=suite, PROJECT_ROOT=str(root), MOCK_MPI=str(mpi),
                                   CAPTURE=str(capture), FLAGCX_CI_MUSA_DEVICES=",".join(map(str, range(count))))
                        result = subprocess.run(["bash", "-c", command], env=env, text=True, capture_output=True)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        calls = [json.loads(line) for line in capture.read_text().splitlines()]
                        self.assertEqual(len(calls), 4)
                        binaries = []
                        for call in calls:
                            binary, ranks, sizes = CHECK.invocation(call["args"])
                            binaries.append(binary)
                            self.assertEqual(ranks, count)
                            self.assertEqual(call["rounds"], "1")
                            self.assertIn("FLAGCX_VMM_ENABLE=0", call["args"])
                            self.assertEqual(len(sizes), 2 if "FLAGCX_DEVICE_ONE_SIDED_FORCE_NET=1" in call["args"] else 15)
                        if suite == "device_api":
                            self.assertEqual(set(binaries), set(CHECK.REQUIRED) - {b for b in CHECK.REQUIRED if "unified" in b})
                            self.assertIn("FLAGCX_P2P_DISABLE=0", calls[0]["args"])
                        else:
                            self.assertEqual(set(binaries), {b for b in CHECK.REQUIRED if "unified" in b})
                            for call in calls[2:]:
                                self.assertIn("FLAGCX_DEVICE_ONE_SIDED_FORCE_NET=1", call["args"])
