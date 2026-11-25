"""Run Iluvatar workload orchestration with mocked SDK/build boundaries."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]


class IluvatarWorkloadTest(unittest.TestCase):
    def run_workload(self, workload, *, failure="", devices=4, coverage=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (
                ".github/scripts/ci/run_hardware_workload.sh",
                ".github/scripts/ci/run_iluvatar_workload.sh",
                ".github/scripts/ci/run_with_timeout.sh",
                ".github/scripts/ci/rdma_static_preflight.sh",
                ".github/scripts/set_env/iluvatar.sh",
            ):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, target)
            # Keep actual platform configuration; replace only hardware probes.
            with (root / ".github/scripts/set_env/iluvatar.sh").open("a") as output:
                output.write('\nflagcx_ci_prepare() { echo "mock CoreX SDK"; }\n'
                             'flagcx_ci_validate_rdma() { echo "mock RDMA"; }\n')
            mock_bin = root / "bin"
            mock_bin.mkdir()
            logger = f'''#!{sys.executable}
import json, os, sys, time, types
from pathlib import Path
kind = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["CALLS"], "a") as output:
    output.write(json.dumps({{"kind": kind, "args": args,
        "corex": os.environ.get("COREX_VISIBLE_DEVICES"),
        "cuda": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "kernel": os.environ.get("COMPILE_KERNEL"),
        "adaptor": os.environ.get("FLAGCX_ADAPTOR")}}) + "\\n")
failure = os.environ.get("FAILURE", "")
if kind == "make":
    if failure == "build": sys.exit(19)
    if failure == "timeout": time.sleep(5)
if kind == "python":
    if args[:2] == ["-m", "pip"]:
        sys.exit(0)
    if args[:1] == ["-c"] and "import torch" in args[1]:
        module = types.ModuleType("torch")
        module.__version__, module.__file__ = "mock CoreX", "mock/corex/torch"
        module.cuda = types.SimpleNamespace(
            is_available=lambda: failure != "no-device",
            device_count=lambda: int(os.environ["DEVICES"]),
            synchronize=lambda device: None)
        module.ones = lambda *a, **kw: types.SimpleNamespace(
            sum=lambda: types.SimpleNamespace(item=lambda: 0 if failure == "compute" else 4))
        sys.modules["torch"] = module
        exec(args[1])
    if args[:1] == ["-c"] and "import flagcx" in args[1] and failure == "import":
        sys.exit(23)
    if args[:1] == ["setup.py"] and failure == "extension":
        sys.exit(24)
'''
            for tool in ("make", "python"):
                target = mock_bin / tool
                target.write_text(logger)
                target.chmod(0o755)
            # Actual handler calls these shared suites; log calls and inject
            # failures rather than reproducing their implementation here.
            suites = {
                ".github/scripts/ci/run_host_perf_suite.sh": "perf",
                "test/script/torch_api_test.sh": "api",
                ".github/scripts/ci/run_torch_coverage.sh": "coverage",
            }
            for relative, kind in suites.items():
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.with_suffix(target.suffix + ".py").write_text(f'''#!{sys.executable}
import json, os, sys
with open(os.environ["CALLS"], "a") as output:
    output.write(json.dumps({{"kind": "{kind}", "args": sys.argv[1:]}}) + "\\n")
failure = os.environ.get("FAILURE", "")
if "{kind}" in failure.split("+") or ("{kind}" == "perf" and sys.argv[1] == failure):
    sys.exit(32 if "{kind}" == "coverage" else 31)
''')
                # These are invoked with bash, so use the selected interpreter.
                target.write_text(f'#!/bin/bash\nexec "{sys.executable}" "{target}.py" "$@"\n')
            calls = root / "calls.jsonl"
            env = os.environ.copy()
            env.update({
                "PATH": f"{mock_bin}:{env['PATH']}",
                "GITHUB_WORKSPACE": str(root), "CALLS": str(calls),
                "FAILURE": failure, "DEVICES": str(devices),
                "PYTHON_BIN": str(mock_bin / "python"),
                "FLAGCX_CI_LOG_DIR": str(root / "logs"),
                "FLAGCX_CI_TEST_TIMEOUT": "1s" if failure == "timeout" else "10s",
                "FLAGCX_CI_TEST_KILL_AFTER": "0.1s",
                "FLAGCX_CI_TORCH_COVERAGE": str(int(coverage)),
            })
            result = subprocess.run(
                ["bash", str(root / ".github/scripts/ci/run_hardware_workload.sh"),
                 "iluvatar", workload, ".github/scripts/set_env/iluvatar.sh", ""],
                env=env, text=True, capture_output=True, timeout=30)
            records = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
            logs = [log.read_text() for log in (root / "logs").glob("*/attempt-*.log")]
            return result, records, logs

    def test_perf_runs_both_modes_with_platform_build_and_visibility(self):
        result, calls, logs = self.run_workload("perf")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual([call["args"][0] for call in calls if call["kind"] == "perf"],
                         ["homogeneous", "heterogeneous"])
        builds = [call for call in calls if call["kind"] == "make"]
        self.assertEqual(len(builds), 2)
        for call in builds:
            self.assertIn("USE_ILUVATAR=1", call["args"])
            self.assertIn("COMPILE_KERNEL=1", call["args"])
            self.assertEqual(call["corex"], "0,1,2,3")
            self.assertEqual(call["cuda"], call["corex"])
        self.assertIn("host_api", builds[1]["args"])
        for call in calls:
            if call["kind"] == "perf":
                self.assertEqual(call["args"][-2:], ["-x", "FLAGCX_VMM_ENABLE=0"])
        self.assertEqual(len(logs), 2)

    def test_perf_failure_in_either_mode_is_not_hidden(self):
        for failure in ("homogeneous", "heterogeneous"):
            with self.subTest(failure=failure):
                result, calls, _ = self.run_workload("perf", failure=failure)
                self.assertEqual(result.returncode, 31)
                self.assertEqual(sum(call["kind"] == "perf" for call in calls), 2)

    def test_build_failure_and_timeout_keep_logs_and_stop_before_tests(self):
        for failure, code in (("build", 19), ("timeout", 124)):
            with self.subTest(failure=failure):
                result, calls, logs = self.run_workload("perf", failure=failure)
                self.assertEqual(result.returncode, code, result.stdout + result.stderr)
                self.assertEqual(sum(call["kind"] == "make" for call in calls), 1)
                self.assertFalse(any(call["kind"] == "perf" for call in calls))
                self.assertEqual(len(logs), 1)

    def test_torch_builds_current_extension_and_runs_api_and_coverage(self):
        result, calls, _ = self.run_workload("torch-api")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(any(call["args"] == ["setup.py", "build_ext", "--inplace"]
                            and call["adaptor"] == "iluvatar" and call["kernel"] == "1"
                            for call in calls))
        self.assertEqual([call["kind"] for call in calls if call["kind"] in ("api", "coverage")],
                         ["api", "coverage"])

    def test_torch_preflight_and_extension_failures_stop_before_api(self):
        for failure, devices, code in (("", 3, 1), ("no-device", 4, 1), ("compute", 4, 1),
                                       ("import", 4, 23), ("extension", 4, 24)):
            with self.subTest(failure=failure, devices=devices):
                result, calls, logs = self.run_workload("torch-api", failure=failure, devices=devices)
                self.assertEqual(result.returncode, code, result.stdout + result.stderr)
                self.assertFalse(any(call["kind"] == "api" for call in calls))
                self.assertTrue(logs)

    def test_torch_api_failure_still_collects_coverage_and_preserves_failure(self):
        for failure, code in (("api", 31), ("coverage", 32), ("api+coverage", 31)):
            with self.subTest(failure=failure):
                result, calls, _ = self.run_workload("torch-api", failure=failure)
                self.assertEqual(result.returncode, code)
                self.assertTrue(any(call["kind"] == "coverage" for call in calls))
        result, calls, _ = self.run_workload("torch-api", coverage=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(any(call["kind"] == "coverage" for call in calls))


class IluvatarWorkflowTest(unittest.TestCase):
    def test_jobs_use_configured_containers_and_required_results(self):
        for filename, kind, gate in (("test.yml", "perf-test", "perf_complete"),
                                     ("torch-api-test.yml", "torch-api-test", "torch_api_complete")):
            with self.subTest(filename=filename):
                jobs = yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())["jobs"]
                name = f"{kind}-iluvatar"
                job = jobs[name]
                self.assertIn("'iluvatar'", job["if"])
                self.assertIn("iluvatar_runs_on", job["runs-on"])
                self.assertIn(name, jobs[gate]["needs"])
                self.assertIn(name, jobs[gate]["env"]["ILUVATAR_RESULT"])
                steps = {step.get("name"): step for step in job["steps"]}
                workload = "perf" if kind == "perf-test" else "torch-api"
                run = steps[f"Run {kind} (iluvatar) in configured container"]
                self.assertIn("iluvatar_ci_image", run["env"]["FLAGCX_CI_IMAGE"])
                self.assertIn("pipefail", run["run"])
                self.assertIn(f"iluvatar {workload}", run["run"])
                self.assertEqual(steps["Upload test attempt logs"]["if"], "always()")
                self.assertEqual(steps[f"Cleanup {kind} (iluvatar) container"]["if"], "always()")

    def test_container_failure_survives_log_pipeline(self):
        for filename, kind in (("test.yml", "perf-test"), ("torch-api-test.yml", "torch-api-test")):
            jobs = yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())["jobs"]
            step = next(step for step in jobs[f"{kind}-iluvatar"]["steps"]
                        if step.get("name") == f"Run {kind} (iluvatar) in configured container")
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                handler = root / ".github/scripts/ci/run_hardware_container.sh"
                handler.parent.mkdir(parents=True)
                handler.write_text('echo "mock build failed"; exit 27\n')
                env = os.environ.copy()
                env.update(GITHUB_WORKSPACE=str(root), FLAGCX_CI_LOG_DIR=str(root / "logs"))
                result = subprocess.run(["bash", "-c", step["run"]], env=env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 27)
                self.assertIn("mock build failed", next((root / "logs").glob("*.log")).read_text())


if __name__ == "__main__":
    unittest.main()
