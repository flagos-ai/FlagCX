"""Exercise MUSA workload construction and completion gates without GPUs."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
CI = ROOT / ".github/scripts/ci"


class MusaCiTest(unittest.TestCase):
    def test_shared_handler_runs_both_perf_modes_and_preserves_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / ".github/scripts/ci"
            scripts.mkdir(parents=True)
            handler = scripts / "run_mthreads_workload.sh"
            handler.write_text((CI / handler.name).read_text())
            environment = root / "environment.sh"
            environment.write_text('''export MPI_HOME=/fake/mpi
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
FLAGCX_CI_PROJECT_MAKE_ARGS=(USE_MUSA=1)
FLAGCX_CI_TEST_MAKE_ARGS=(USE_MUSA=1)
flagcx_ci_configure_suite() { :; }
flagcx_ci_prepare() { :; }
flagcx_ci_validate_rdma() { :; }
''')
            fake_bin = root / "bin"
            fake_bin.mkdir()
            recorder = '''#!/usr/bin/env python3
import json, os, pathlib, sys
kind = "make" if pathlib.Path(sys.argv[0]).name == "make" else sys.argv[1]
with open(os.environ["CAPTURE"], "a") as out:
    out.write(json.dumps(kind) + "\\n")
sys.exit(19 if kind == os.environ.get("FAIL_MODE") else 0)
'''
            for target in (fake_bin / "make", scripts / "run_host_perf_suite.sh"):
                target.write_text(recorder)
                target.chmod(0o755)
            (fake_bin / "nproc").write_text("#!/bin/sh\necho 2\n")
            (fake_bin / "nproc").chmod(0o755)
            capture = root / "calls"
            for failure in ("", "homogeneous", "heterogeneous"):
                with self.subTest(failure=failure):
                    capture.write_text("")
                    result = subprocess.run(
                        ["bash", str(handler), "perf"],
                        env={**os.environ, "GITHUB_WORKSPACE": str(root),
                             "FLAGCX_CI_SET_ENV": environment.name,
                             "PATH": f"{fake_bin}:{os.environ['PATH']}",
                             "CAPTURE": str(capture), "FAIL_MODE": failure},
                        capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 19 if failure else 0, result.stderr)
                    calls = [json.loads(line) for line in capture.read_text().splitlines()]
                    expected = ["make", "make", "homogeneous"]
                    if failure != "homogeneous":
                        expected.append("heterogeneous")
                    self.assertEqual(calls, expected)

    def test_config_and_public_workflows_consume_musa(self):
        result = subprocess.run(
            ["ruby", str(CI / "load_platform_config.rb"), "mthreads"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        config = json.loads(result.stdout)
        self.assertEqual(config["set_env"], ".github/scripts/set_env/mthreads.sh")
        self.assertEqual(json.loads(config["runs_on"]), ["mt-8g-cicd-flagcx"])
        registry = yaml.safe_load((ROOT / ".github/configs/platforms.yml").read_text())["platforms"]
        self.assertIn("mthreads", registry)
        self.assertNotIn("musa", registry)
        for name in ("configs/musa.yml", "scripts/set_env/musa.sh", "scripts/ci/run_musa_workload.sh"):
            self.assertFalse((ROOT / ".github" / name).exists(), name)
        self.assertIn("--runtime=mthreads", config["docker_args"])
        self.assertIn("device_api", json.loads(config["unit_test_suites"]))
        for filename, prefix, workloads, completion in (
            ("test.yml", "perf-test", ("perf", "perf-test-experimental"), "perf_complete"),
            ("torch-api-test.yml", "torch-api-test", ("torch-api",), "torch_api_complete"),
        ):
            jobs = yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())["jobs"]
            for workload in workloads:
                key = "perf-test-experimental-mthreads" if workload == "perf-test-experimental" else f"{prefix}-mthreads"
                job = jobs[key]
                self.assertIn("'mthreads'", job["if"])
                self.assertIn("mthreads_runs_on", job["runs-on"])
                display = "perf-test-experimental" if workload == "perf-test-experimental" else prefix
                self.assertEqual(job["name"], f"{display} (musa)")
                self.assertIn(key, jobs[completion]["needs"])
                commands = "\n".join(step.get("run", "") for step in job["steps"])
                self.assertIn(f"mthreads {workload}", commands)
                self.assertTrue(any(step.get("if") == "always()" and "upload-artifact" in step.get("uses", "") for step in job["steps"]))
            self.assertIn("MTHREADS_RESULT", jobs[completion]["env"])

    def test_completion_gate_rejects_selected_musa_failure_or_skip(self):
        for experimental in (False, True):
            filename = "test.yml" if experimental else "torch-api-test.yml"
            completion = "perf_complete" if experimental else "torch_api_complete"
            jobs = yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())["jobs"]
            gate = next(step["run"] for step in jobs[completion]["steps"]
                        if step.get("name", "").startswith("Validate selected Mthreads"))
            for config in ("success", "failure"):
                for regular in ("success", "failure", "skipped", "cancelled"):
                    for extra in ("success", "failure", "skipped", "cancelled"):
                        env = {**os.environ, "PLATFORMS": '["mthreads"]',
                               "CONFIG_RESULT": config, "MTHREADS_RESULT": regular,
                               "MTHREADS_EXPERIMENTAL_RESULT": extra}
                        result = subprocess.run(
                            ["bash", "-eu", "-c", gate],
                            env=env, capture_output=True, text=True,
                        )
                        expected = config == "success" and regular == "success" and (not experimental or extra == "success")
                        self.assertEqual(result.returncode == 0, expected, result.stderr)
        # A missing result must fail closed; unselected MUSA is irrelevant.
        env = {k: v for k, v in os.environ.items() if not k.startswith("MTHREADS_")}
        for selection, expected in (("[]", True), ('["mthreads"]', False)):
            result = subprocess.run(
                ["bash", "-eu", "-c", gate],
                env={**env, "PLATFORMS": selection, "CONFIG_RESULT": "success"},
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode == 0, expected, result.stderr)

    def test_environment_enables_native_kernel_only_for_kernel_suites(self):
        for suite in ("perf", "runner", "device_api", "device_api_host", "device_api_unified_ir", "perf-test-experimental"):
            result = subprocess.run(
                ["bash", "-c", 'set -eu; source "$1"; flagcx_ci_configure_suite "$2"; printf "%s\\n" "${FLAGCX_CI_PROJECT_MAKE_ARGS[@]}" "${FLAGCX_CI_TEST_MAKE_ARGS[@]}"',
                 "bash", str(ROOT / ".github/scripts/set_env/mthreads.sh"), suite],
                capture_output=True, text=True, check=True,
            )
            self.assertIn("USE_MUSA=1", result.stdout)
            self.assertEqual(result.stdout.count("COMPILE_KERNEL=1"), 2 if suite == "perf-test-experimental" else 0)

    def run_experimental(self, directory, *, fail_label="", xml_mode="pass", fail_sdk_build=False):
        root = Path(directory)
        scripts = root / ".github/scripts/ci"
        scripts.mkdir(parents=True, exist_ok=True)
        # Execute the real workload with build/launcher shims. The real XML
        # validator still checks all requested cases on all four ranks.
        for name in ("run_mthreads_perf_experimental.sh", "check_required_gtests.py"):
            (scripts / name).write_text((CI / name).read_text())
        for name in ("run_mpi_with_timeout.sh", "run_with_timeout.sh"):
            path = scripts / name
            path.write_text('#!/usr/bin/env bash\nexec python3 "$(dirname "$0")/recorder.py" "$@"\n')
            (scripts / "recorder.py").write_text(r'''#!/usr/bin/env python3
import json, os, pathlib, sys, xml.etree.ElementTree as ET
args = sys.argv[1:]
label = os.getenv("FLAGCX_CI_MPI_LABEL", "probe")
with open(os.environ["CAPTURE"], "a") as out:
    out.write(json.dumps({"label": label, "args": args}) + "\n")
if label == os.getenv("FAIL_LABEL"):
    sys.exit(19)
filter_arg = next((a for a in args if a.startswith("--gtest_filter=")), None)
if filter_arg:
    result_dir = pathlib.Path(args[args.index(next(a for a in args if a.endswith("run_gtest_rank_xml.sh"))) + 1])
    for rank in range(4):
        if rank == 3 and os.getenv("XML_MODE") == "missing":
            continue
        suite = ET.Element("testsuite", name="FlagCXCollTest")
        for name in filter_arg.split("=", 1)[1].split(":"):
            case = ET.SubElement(suite, "testcase", name=name.split(".")[1], status="run", result="completed")
            if rank == 3 and os.getenv("XML_MODE") == "skip":
                ET.SubElement(case, "skipped")
        ET.ElementTree(suite).write(result_dir / f"rank-{rank}.xml")
''')
            path.chmod(0o755)
        capture = root / "calls.jsonl"
        capture.write_text("")
        command = r'''set -euo pipefail
make() {
  if [[ "${FAIL_SDK_BUILD:-0}" == 1 && "$*" == *unirunner-probe* ]]; then
    return 23
  fi
}
cmake() { :; }
nproc() { echo 2; }
FLAGCX_CI_PROJECT_MAKE_ARGS=(USE_MUSA=1 COMPILE_KERNEL=1)
FLAGCX_CI_TEST_MAKE_ARGS=("${FLAGCX_CI_PROJECT_MAKE_ARGS[@]}")
FLAGCX_CI_MUSA_DEVICES=0,1,2,3
source "$GITHUB_WORKSPACE/.github/scripts/ci/run_mthreads_perf_experimental.sh"
'''
        result = subprocess.run(
            ["bash", "-c", command],
            env={**os.environ, "GITHUB_WORKSPACE": str(root), "CAPTURE": str(capture),
                 "FAIL_LABEL": fail_label, "XML_MODE": xml_mode,
                 "FAIL_SDK_BUILD": "1" if fail_sdk_build else "0",
                 "FLAGCX_CI_LOG_DIR": str(root / "logs")},
            capture_output=True, text=True, timeout=30,
        )
        calls = [json.loads(line) for line in capture.read_text().splitlines()]
        return result, calls

    def test_experimental_runs_real_algorithms_and_checks_results(self):
        with tempfile.TemporaryDirectory() as directory:
            result, calls = self.run_experimental(directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(calls), 61)  # Probe, 5 GTests, 55 perf runs.
            for call in calls[1:]:
                args = call["args"]
                self.assertEqual(args[:3], ["-np", "4", "--allow-run-as-root"])
                self.assertIn("FLAGCX_CI_EXPECT_UNIRUNNER_KERNEL=1", args)
                self.assertIn("FLAGCX_CI_EXPECT_STREAM_VALUE=1", args)
                self.assertIn("MUSA_VISIBLE_DEVICES=0,1,2,3", args)
                if any("/perf_" in a for a in args):
                    self.assertIn("-v", args)
                    self.assertEqual(args[args.index("-v") + 1], "1")
            sliced = calls[-1]["args"]
            self.assertIn("FLAGCX_UNIRUNNER_ALGO=6", sliced)
            self.assertEqual(sliced[sliced.index("-e") + 1], "64M")
            self.assertEqual({c["label"].split()[-1] for c in calls if "RingAR int32" in c["label"]}, {"sum", "prod", "min", "max", "avg"})

    def test_experimental_preserves_launcher_failures(self):
        for label in ("probe", "MUSA experimental AllGatherGemm", "MUSA experimental IpcRingAR", "MUSA experimental SlicedAR"):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                result, calls = self.run_experimental(directory, fail_label=label)
                self.assertEqual(result.returncode, 19, result.stderr)
                self.assertEqual(calls[-1]["label"], label)

    def test_sdk_probe_build_failure_stops_before_gpu_work(self):
        with tempfile.TemporaryDirectory() as directory:
            result, calls = self.run_experimental(directory, fail_sdk_build=True)
            self.assertEqual(result.returncode, 23, result.stderr)
            self.assertEqual(calls, [])

    def test_experimental_rejects_missing_or_skipped_rank_tests(self):
        for mode in ("missing", "skip"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                result, calls = self.run_experimental(directory, xml_mode=mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(calls[-1]["label"], "MUSA experimental AllGatherGemm")


if __name__ == "__main__":
    unittest.main()
