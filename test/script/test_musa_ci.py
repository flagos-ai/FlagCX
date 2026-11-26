"""Exercise MUSA workload construction and completion gates without GPUs."""

import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[2]
CI = ROOT / ".github/scripts/ci"


def copy_timeout_scripts(directory):
    for name in ("timeouts.sh", "run_with_timeout.sh", "collect_process_stacks.py"):
        shutil.copy2(CI / name, directory / name)


class MusaCiTest(unittest.TestCase):
    def test_native_mccl_logging_is_exported_and_can_be_overridden(self):
        environment = ROOT / ".github/scripts/set_env/mthreads.sh"
        for overrides, expected in (({}, ["TRACE", "ALL"]),
                                    ({"MCCL_DEBUG": "INFO", "MCCL_DEBUG_SUBSYS": "INIT"},
                                     ["INFO", "INIT"])):
            with self.subTest(overrides=overrides):
                env = {k: v for k, v in os.environ.items()
                       if k not in ("MCCL_DEBUG", "MCCL_DEBUG_SUBSYS")}
                env.update(overrides)
                env.update(ENV_SCRIPT=str(environment), TEST_PYTHON=sys.executable)
                # A child process must receive the variables, as MPI workers do.
                result = subprocess.run(["bash", "-eu", "-c", '''source "$ENV_SCRIPT"
"$TEST_PYTHON" -c 'import json, os; print(json.dumps([os.environ["MCCL_DEBUG"], os.environ["MCCL_DEBUG_SUBSYS"]]))'
'''], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout), expected)

    def test_shared_handler_runs_both_perf_modes_and_preserves_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / ".github/scripts/ci"
            scripts.mkdir(parents=True)
            handler = scripts / "run_mthreads_workload.sh"
            handler.write_text((CI / handler.name).read_text())
            copy_timeout_scripts(scripts)
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
                             "FLAGCX_CI_LOG_DIR": str(root / "logs"),
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

    def test_torch_handler_uses_one_interpreter_and_preserves_failures(self):
        cases = (("explicit", ""), ("venv", ""), ("explicit", "probe"),
                 ("explicit", "pip"), ("explicit", "build"),
                 ("explicit", "suite"), ("explicit", "coverage"))
        for selection, failure in cases:
            with self.subTest(selection=selection, failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                scripts = root / ".github/scripts/ci"
                scripts.mkdir(parents=True)
                handler = scripts / "run_mthreads_workload.sh"
                handler.write_text((CI / handler.name).read_text())
                copy_timeout_scripts(scripts)
                environment = root / "environment.sh"
                environment.write_text("""export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
FLAGCX_CI_PROJECT_MAKE_ARGS=(USE_MUSA=1)
flagcx_ci_configure_suite() { :; }
flagcx_ci_prepare() { :; }
flagcx_ci_validate_rdma() { :; }
""")
                test_scripts = root / "test/script"
                test_scripts.mkdir(parents=True)
                (test_scripts / "torch_api_test.sh").write_text('"$PYTHON_BIN" suite\n')
                (scripts / "run_torch_coverage.sh").write_text('#!/bin/bash\n"$PYTHON_BIN" coverage\n')
                (scripts / "run_torch_coverage.sh").chmod(0o755)
                fake_bin = root / "bin"
                fake_bin.mkdir()
                python = root / "torch environment/bin/python3"
                python.parent.mkdir(parents=True)
                recorder = f"#!{sys.executable}\n" + r'''import json, os, pathlib, sys
args = sys.argv[1:]
if pathlib.Path(sys.argv[0]).name == "make":
    kind = "make"
elif args[0] == "-c":
    kind = "probe" if "device_count" in args[1] else "import"
elif args[:2] == ["-m", "pip"]:
    kind = "coverage-install" if "coverage==7.10.7" in args else "pip"
elif args[0] == "setup.py":
    kind = "build"
else:
    kind = args[0]
with open(os.environ["CAPTURE"], "a") as out:
    out.write(json.dumps({"kind": kind, "executable": sys.argv[0],
                          "selected": os.environ.get("PYTHON_BIN"),
                          "autoload": os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD")}) + "\n")
sys.exit(19 if kind == os.environ.get("FAIL_STEP") else 0)
'''
                python.write_text(recorder)
                python.chmod(0o755)
                (fake_bin / "make").write_text(recorder)
                (fake_bin / "make").chmod(0o755)
                # A system interpreter without Torch must not be used by the workload.
                for name in ("python3", "python"):
                    # The common watchdog uses Python for deadlines and process
                    # cleanup. Reject Torch work on system Python only.
                    (fake_bin / name).write_text(
                        f"#!{sys.executable}\n" + '''import os, sys
args = sys.argv[1:]
if args[:1] == ["-"] or (args[:1] == ["-c"] and "import time" in args[1]):
    os.execv(sys.executable, [sys.executable, *args])
sys.exit(29)
''')
                    (fake_bin / name).chmod(0o755)
                (fake_bin / "nproc").write_text("#!/bin/sh\necho 2\n")
                (fake_bin / "nproc").chmod(0o755)
                capture = root / "calls"
                env = {k: v for k, v in os.environ.items() if k != "PYTHON_BIN"}
                env.update(GITHUB_WORKSPACE=str(root), FLAGCX_CI_SET_ENV=environment.name,
                           FLAGCX_CI_LOG_DIR=str(root / "logs"),
                           PATH=f"{fake_bin}:{os.environ['PATH']}", CAPTURE=str(capture),
                           FAIL_STEP=failure, FLAGCX_CI_TORCH_COVERAGE="1",
                           VIRTUAL_ENV=str(python.parents[1]))
                if selection == "explicit":
                    env["PYTHON_BIN"] = str(python)
                result = subprocess.run(["bash", str(handler), "torch-api"], env=env,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 19 if failure else 0, result.stderr)
                calls = [json.loads(line) for line in capture.read_text().splitlines()]
                expected = ["probe", "make", "pip", "build", "suite", "coverage-install", "coverage"]
                if failure:
                    expected = expected[:expected.index(failure) + 1]
                if selection == "venv":
                    expected.insert(0, "import")
                self.assertEqual([c["kind"] for c in calls], expected)
                for call in calls:
                    if call["kind"] != "make":
                        self.assertEqual(call["executable"], str(python))
                    if call["kind"] != "import":
                        self.assertEqual(call["selected"], str(python))
                    if call["kind"] == "build":
                        self.assertEqual(call["autoload"], "0")

    def load_mthreads_config(self):
        result = subprocess.run(
            ["ruby", str(CI / "load_platform_config.rb"), "mthreads"],
            # Read JSON on stdout without writing to the enclosing Actions step.
            env={k: v for k, v in os.environ.items() if k != "GITHUB_OUTPUT"},
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        return json.loads(result.stdout)

    def test_config_load_ignores_inherited_github_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "github-output"
            sentinel = "existing_step_output=preserve\n"
            output.write_text(sentinel)
            with mock.patch.dict(os.environ, {"GITHUB_OUTPUT": str(output)}):
                config = self.load_mthreads_config()
                self.assertEqual(os.environ["GITHUB_OUTPUT"], str(output))
            self.assertEqual(config["set_env"], ".github/scripts/set_env/mthreads.sh")
            self.assertEqual(output.read_text(), sentinel)

    def test_config_and_public_workflows_consume_musa(self):
        config = self.load_mthreads_config()
        self.assertEqual(config["set_env"], ".github/scripts/set_env/mthreads.sh")
        self.assertEqual(config["ci_image"], "harbor.baai.ac.cn/flagos-runtime/flagos-runtime-mthreads-musa5.2.0@sha256:1870e059f98d95465009f55db78d723cf77640b211ff1c5f1378d59b8e200c27")
        self.assertEqual(json.loads(config["runs_on"]), ["mt-8g-cicd-flagcx"])
        registry = yaml.safe_load((ROOT / ".github/configs/platforms.yml").read_text())["platforms"]
        self.assertIn("mthreads", registry)
        self.assertNotIn("musa", registry)
        for name in ("configs/musa.yml", "scripts/set_env/musa.sh", "scripts/ci/run_musa_workload.sh"):
            self.assertFalse((ROOT / ".github" / name).exists(), name)
        self.assertIn("--runtime=mthreads", config["docker_args"])
        self.assertEqual(json.loads(config["unit_test_suites"]),
                         ["adaptor", "core", "device_api", "device_api_host",
                          "device_api_unified_ir", "p2p", "rma", "runner", "service", "symmem"])
        unit_jobs = yaml.safe_load((ROOT / ".github/workflows/unit_tests_common.yml").read_text())["jobs"]
        self.assertEqual(unit_jobs["unit_test"]["if"],
                         "inputs.platform != 'enflame' && inputs.platform != 'kunlunxin'")
        self.assertEqual(unit_jobs["unit_test_enflame"]["if"], "inputs.platform == 'enflame'")
        self.assertEqual(unit_jobs["kunlunxin_unit_test"]["if"], "inputs.platform == 'kunlunxin'")
        for filename, prefix, workloads, completion in (
            ("test.yml", "perf-test", ("perf", "perf-test-experimental"), "perf_complete"),
            ("torch-api-test.yml", "torch-api-test", ("torch-api",), "torch_api_complete"),
        ):
            jobs = yaml.safe_load((ROOT / ".github/workflows" / filename).read_text())["jobs"]
            self.assertEqual(
                jobs["load-hardware-container-configs"]["outputs"]["mthreads_ignore_test"],
                "${{ steps.mthreads.outputs.ignore_test }}")
            for workload in workloads:
                key = "perf-test-experimental-mthreads" if workload == "perf-test-experimental" else f"{prefix}-mthreads"
                job = jobs[key]
                self.assertIn("'mthreads'", job["if"])
                expected_condition = ("${{ false && contains(fromJSON(inputs.platforms), 'mthreads') }}"
                                      if workload == "perf-test-experimental"
                                      else "contains(fromJSON(inputs.platforms), 'mthreads')")
                self.assertEqual(job["if"], expected_condition)
                self.assertIn("mthreads_runs_on", job["runs-on"])
                display = "perf-test-experimental" if workload == "perf-test-experimental" else prefix
                self.assertEqual(job["name"], f"{display} (musa)")
                self.assertIn(key, jobs[completion]["needs"])
                commands = "\n".join(step.get("run", "") for step in job["steps"])
                self.assertIn(f"mthreads {workload}", commands)
                run_step = next(step for step in job["steps"]
                                if "run_hardware_container.sh" in step.get("run", ""))
                self.assertEqual(
                    run_step["env"]["FLAGCX_CI_IGNORE_TEST"],
                    "${{ needs.load-hardware-container-configs.outputs.mthreads_ignore_test }}")
                self.assertTrue(any(step.get("if") == "always()" and "upload-artifact" in step.get("uses", "") for step in job["steps"]))
            self.assertIn("MTHREADS_RESULT", jobs[completion]["env"])

    def test_mtml_injection_arguments_reach_docker_creation(self):
        config = self.load_mthreads_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            docker = root / "docker"
            capture = root / "calls.jsonl"
            docker.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
with open(os.environ["MUSA_DOCKER_CAPTURE"], "a") as out:
    out.write(json.dumps(sys.argv[1:]) + "\\n")
''')
            docker.chmod(0o755)
            for workload, suite in (("unittest", "runner"), ("perf", ""),
                                    ("perf-test-experimental", ""), ("torch-api", "")):
                with self.subTest(workload=workload):
                    capture.write_text("")
                    env = {key: value for key, value in os.environ.items()
                           if key not in ("GITHUB_OUTPUT", "GITHUB_STEP_SUMMARY")}
                    env.update(PATH=f"{root}:{os.environ['PATH']}",
                               MUSA_DOCKER_CAPTURE=str(capture),
                               FLAGCX_CI_IMAGE=config["ci_image"],
                               FLAGCX_CI_DOCKER_ARGS=config["docker_args"],
                               FLAGCX_CI_SET_ENV=config["set_env"],
                               FLAGCX_CI_LOG_DIR=str(root / "logs" / workload))
                    result = subprocess.run(
                        ["bash", str(CI / "run_hardware_container.sh"),
                         "mthreads", workload, suite],
                        env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    calls = [json.loads(line) for line in capture.read_text().splitlines()]
                    run = next(args for args in calls if args[:2] == ["container", "run"])
                    options = run[:run.index(config["ci_image"])]
                    self.assertIn("--env=MTHREADS_VISIBLE_DEVICES=0,1,2,3", options)
                    self.assertIn("--env=MTHREADS_DRIVER_CAPABILITIES=compute,utility", options)
    def test_exported_ignore_policy_needs_no_yaml_or_ruby(self):
        config = self.load_mthreads_config()
        policy = json.loads(config["ignore_test"])
        phase = next(item for item in policy["phases"]
                     if item["name"] == "vmm-net-data")
        with tempfile.TemporaryDirectory() as directory:
            # -S disables site-packages, including PyYAML. A Ruby fallback
            # must fail, as it does in the unmodified runtime image.
            ruby = Path(directory) / "ruby"
            ruby.write_text("#!/bin/sh\necho 'Unexpected Ruby fallback' >&2\nexit 97\n")
            ruby.chmod(0o755)
            env = {**os.environ, "PATH": directory, "PYTHONPATH": "",
                   "FLAGCX_CI_PLATFORM": "mthreads",
                   "FLAGCX_CI_IGNORE_TEST": config["ignore_test"]}
            for args, expected in ((["torch", "homogeneous"], ""),
                                   (["phase", "symmem", "vmm-net-data"], phase["reason"])):
                with self.subTest(args=args):
                    result = subprocess.run(
                        [sys.executable, "-S", str(CI / "ignore_test.py"), *args],
                        env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), expected)

    def test_native_isolation_uses_yaml_policy_without_environment_override(self):
        config = self.load_mthreads_config()
        policy = json.loads(config["ignore_test"])
        for suite, expected in (("adaptor", "NetAdaptorLoopback.RegisterDmaBufMr"),
                                ("device_api_host", "")):
            excluded = subprocess.run(
                [sys.executable, str(CI / "ignore_test.py"), "gtest", suite],
                env={**os.environ, "FLAGCX_CI_IGNORE_TEST": config["ignore_test"]},
                capture_output=True, text=True, check=True)
            self.assertEqual(excluded.stdout.strip(), expected)
        self.assertEqual({item["name"] for item in policy["phases"]}, {
            "vmm-local-and-remote-without-net", "vmm-net-data"})
        result = subprocess.run(
            ["bash", "-eu", "-c", 'source "$1"; printf "%s" "$FLAGCX_CI_IGNORE_TEST"',
             "bash", str(ROOT / config["set_env"])],
            env={**os.environ, "FLAGCX_CI_IGNORE_TEST": config["ignore_test"]},
            capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), policy)

    def test_completion_gates_require_musa_standard_success_and_experimental_skipped(self):
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
                        expected = config == "success" and regular == "success" and (not experimental or extra == "skipped")
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
            self.assertEqual(result.stdout.count("COMPILE_KERNEL=1"), 2 if suite in ("device_api", "device_api_unified_ir", "perf-test-experimental") else 0)

    def test_native_device_suites_use_gpu_mr_with_net_enabled(self):
        for suite in ("device_api", "device_api_unified_ir"):
            with self.subTest(suite=suite):
                result = subprocess.run(
                    ["bash", "-eu", "-c", '''source "$1"
flagcx_ci_configure_suite "$2"
printf '%s\\n' "${FLAGCX_CI_DEVICE_API_MPI_ARGS[*]}" "${FLAGCX_CI_UNIFIED_IR_MPI_ARGS[*]}"
''', "bash", str(ROOT / ".github/scripts/set_env/mthreads.sh"), suite],
                    env={**os.environ, "FLAGCX_DMABUF_ENABLE": "1"},
                    capture_output=True, text=True, check=True,
                )
                lines = result.stdout.splitlines()
                self.assertEqual(len(lines), 2)
                for line in lines:
                    args = shlex.split(line)
                    self.assertEqual(args.count("FLAGCX_DMABUF_ENABLE=0"), 1)
                    self.assertNotIn("FLAGCX_DMABUF_ENABLE=1", args)
                    self.assertIn("FLAGCX_IB_DISABLE=0", args)

    def run_experimental(self, directory, *, fail_label="", xml_mode="pass",
                         fail_project_build=False, fail_after_reports=False, umask="022"):
        root = Path(directory)
        scripts = root / ".github/scripts/ci"
        scripts.mkdir(parents=True, exist_ok=True)
        copy_timeout_scripts(scripts)
        # Execute the real workload with build/launcher shims. The real XML
        # validator still checks all requested cases on all four ranks.
        for name in ("run_mthreads_perf_experimental.sh", "check_required_gtests.py"):
            (scripts / name).write_text((CI / name).read_text())
        for name in ("run_mpi_with_timeout.sh",):
            path = scripts / name
            path.write_text('#!/usr/bin/env bash\nexec ' + shlex.quote(sys.executable)
                            + ' "$(dirname "$0")/recorder.py" "$@"\n')
            (scripts / "recorder.py").write_text(r'''#!/usr/bin/env python3
import json, os, pathlib, sys, xml.etree.ElementTree as ET
args = sys.argv[1:]
label = os.environ["FLAGCX_CI_MPI_LABEL"]
with open(os.environ["CAPTURE"], "a") as out:
    out.write(json.dumps({"label": label, "args": args}) + "\n")
if label == os.getenv("FAIL_LABEL") and os.getenv("FAIL_AFTER_REPORTS") != "1":
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
if label == os.getenv("FAIL_LABEL"):
    sys.exit(19)
''')
            path.chmod(0o755)
        capture = root / "calls.jsonl"
        capture.write_text("")
        fake_bin = root / "bin"
        fake_bin.mkdir(exist_ok=True)
        for name in ("make", "cmake"):
            tool = fake_bin / name
            tool.write_text('''#!/usr/bin/env bash
if [[ "$(basename "$0")" == make && "${FAIL_PROJECT_BUILD:-0}" == 1 ]]; then
  exit 23
fi
''')
            tool.chmod(0o755)
        # The host container wrapper creates this directory before mounting it.
        logs = root / "logs"
        logs.mkdir(exist_ok=True)
        logs.chmod(0o755)
        command = r'''set -euo pipefail
umask "$MUSA_TEST_UMASK"
nproc() { echo 2; }
FLAGCX_CI_PROJECT_MAKE_ARGS=(USE_MUSA=1 COMPILE_KERNEL=1)
FLAGCX_CI_TEST_MAKE_ARGS=("${FLAGCX_CI_PROJECT_MAKE_ARGS[@]}")
FLAGCX_CI_MUSA_DEVICES=0,1,2,3
source "$GITHUB_WORKSPACE/.github/scripts/ci/run_mthreads_perf_experimental.sh"
'''
        result = subprocess.run(
            ["bash", "-c", command],
            env={**os.environ, "GITHUB_WORKSPACE": str(root), "CAPTURE": str(capture),
                 "PATH": f"{fake_bin}:{os.environ['PATH']}",
                 "FAIL_LABEL": fail_label, "XML_MODE": xml_mode,
                 "FAIL_PROJECT_BUILD": "1" if fail_project_build else "0",
                 "FAIL_AFTER_REPORTS": "1" if fail_after_reports else "0",
                 "MUSA_TEST_UMASK": umask,
                 "FLAGCX_CI_LOG_DIR": str(root / "logs")},
            capture_output=True, text=True, timeout=30,
        )
        calls = [json.loads(line) for line in capture.read_text().splitlines()]
        return result, calls

    def test_experimental_runs_real_algorithms_and_checks_results(self):
        with tempfile.TemporaryDirectory() as directory:
            result, calls = self.run_experimental(directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(calls), 61)  # 6 GTests, 55 perf runs.
            lifecycle = next(call for call in calls if call["label"] == "MUSA experimental IPC lifecycle")
            self.assertIn("--gtest_filter=FlagCXCollTest.UniRunnerIpcFinalizeThenRankLocalDestroy:FlagCXCollTest.GroupedCommunicatorsFinalizeTheNonLaunchStreamFirst", lifecycle["args"])
            for call in calls:
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
        for label in ("MUSA experimental AllGatherGemm", "MUSA experimental IPC lifecycle", "MUSA experimental IpcRingAR", "MUSA experimental SlicedAR"):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                result, calls = self.run_experimental(directory, fail_label=label)
                self.assertEqual(result.returncode, 19, result.stderr)
                self.assertEqual(calls[-1]["label"], label)

    def test_project_build_failure_stops_before_gpu_work(self):
        with tempfile.TemporaryDirectory() as directory:
            result, calls = self.run_experimental(directory, fail_project_build=True)
            self.assertEqual(result.returncode, 23, result.stderr)
            self.assertEqual(calls, [])

    def test_experimental_reports_readable_after_success_and_failure(self):
        for failure in ("", "MUSA experimental AllGatherGemm"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                root.chmod(0o755)
                result, _ = self.run_experimental(
                    directory, fail_label=failure, fail_after_reports=True, umask="077")
                self.assertEqual(result.returncode, 19 if failure else 0, result.stderr)
                reports = list((root / "logs").glob("musa-*"))
                self.assertEqual(len(reports), 1 if failure else 6)
                for report in reports:
                    self.assertEqual(report.stat().st_mode & 0o777, 0o755)
                    files = list(report.glob("rank-*.xml"))
                    self.assertEqual(len(files), 4)
                    for path in files:
                        self.assertEqual(path.stat().st_mode & 0o777, 0o644)
                if hasattr(os, "geteuid") and os.geteuid() == 0:
                    # Linux CI: root generates the artifacts; a different UID
                    # must be able to enumerate directories and read every XML.
                    read = subprocess.run(
                        [sys.executable, "-c", '''import pathlib, sys
files = list(pathlib.Path(sys.argv[1]).glob("musa-*/rank-*.xml"))
assert len(files) == int(sys.argv[2])
for path in files: path.read_bytes()
''', str(root / "logs"), str(4 * len(reports))],
                        user=65534, group=65534, extra_groups=(),
                        capture_output=True, text=True)
                    self.assertEqual(read.returncode, 0, read.stderr)

    def test_experimental_rejects_missing_or_skipped_rank_tests(self):
        for mode in ("missing", "skip"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                result, calls = self.run_experimental(directory, xml_mode=mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(calls[-1]["label"], "MUSA experimental AllGatherGemm")


if __name__ == "__main__":
    unittest.main()
