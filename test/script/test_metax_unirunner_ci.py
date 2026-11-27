"""Exercise the experimental CI orchestration with fake build/MPI commands."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[2]
CI = ROOT / ".github/scripts/ci"

MPI = r'''#!/usr/bin/env python3
import json, os, pathlib, sys, xml.etree.ElementTree as ET
args = sys.argv[1:]
values = {}
i = 0
while i < len(args):
    arg = args[i]
    if arg == '-x':
        key, _, value = args[i+1].partition('=')
        values[key] = value or os.environ.get(key, '')
        i += 2
    elif arg == '-np':
        values['np'] = args[i+1]
        i += 2
    elif arg == '--allow-run-as-root':
        i += 1
    else:
        break
command = args[i:]
with open(os.environ['INVOCATIONS'], 'a') as output:
    output.write(json.dumps({'env': values, 'command': command}) + '\n')
algorithm = values.get('FLAGCX_UNIRUNNER_ALGO', 'default')
if algorithm == os.environ.get('FAIL_ALGORITHM'):
    sys.exit(19)
if algorithm != 'default' and not os.environ.get('MISSING_EXECUTOR'):
    print('UniRunner actual algorithm=' + algorithm)
    if not os.environ.get('MISSING_POLICY'):
        print('MACA allocation: purpose=UNIRUNNER allocator=mcMalloc')
    if algorithm in ('2', '6', '9', '11') and not os.environ.get('MISSING_NATIVE'):
        print('UniRunner native launcher=mock-test-only')
if command[0] == 'bash':
    directory = pathlib.Path(command[2])
    selected = command[-1].split('=', 1)[1]
    names = ['AllGatherGemm', 'AllGatherGemmConvergesRankLocalSizeFailure',
             'GemmReduceScatter', 'GemmReduceScatterConvergesRankLocalSizeFailure',
             'UniRunnerDagAlgorithmsRejectOuterGroup',
             'UniRunnerIpcPreparationErrorConverges', 'AllGatherInsideGroup']
    import fnmatch
    for rank in range(4 - int(os.environ.get('MISSING_RANK', '0'))):
        root = ET.Element('testsuites')
        suite = ET.SubElement(root, 'testsuite', name='FlagCXCollTest')
        for name in names:
            full = 'FlagCXCollTest.' + name
            if any(fnmatch.fnmatchcase(full, pattern) for pattern in selected.split(':')):
                case = ET.SubElement(suite, 'testcase', name=name,
                                     status='run', result='completed')
                if os.environ.get('SKIP_CASE') == name:
                    ET.SubElement(case, 'skipped')
        ET.ElementTree(root).write(directory / ('rank-%s.xml' % rank))
'''


class MetaXUniRunnerCiTest(unittest.TestCase):
    def run_matrix(self, **overrides):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        project = Path(temporary.name)
        scripts = project / ".github/scripts/ci"
        scripts.mkdir(parents=True)
        for name in ("run_metax_perf_experimental.sh", "check_required_gtests.py"):
            shutil.copyfile(CI / name, scripts / name)
        (scripts / "timeouts.sh").write_text(
            'FLAGCX_CI_PERF_TIMEOUT=10m\n'
            'nproc() { echo 4; }\n'
            'flagcx_ci_run() { printf "%s\\n" "$*" >> "$BUILDS"; }\n'
        )
        launcher = scripts / "run_mpi_with_timeout.sh"
        launcher.write_text(MPI)
        launcher.chmod(0o755)
        env = os.environ.copy()
        env.update(
            GITHUB_WORKSPACE=str(project),
            FLAGCX_CI_LOG_DIR=str(project / "logs"),
            MPI_HOME="/fake/mpi",
            MACA_VISIBLE_DEVICES="0,2,4,6",
            FLAGCX_IB_HCA="bnxt_roce0,bnxt_roce2",
            INVOCATIONS=str(project / "invocations.jsonl"),
            BUILDS=str(project / "builds.log"),
            **overrides,
        )
        command = (
            'FLAGCX_CI_PROJECT_MAKE_ARGS=(USE_METAX=1 COMPILE_KERNEL=1 FORCE_DEFAULT_PATH=1); '
            'FLAGCX_CI_TEST_MAKE_ARGS=(USE_METAX=1 COMPILE_KERNEL=1 FORCE_DEFAULT_PATH=1); '
            'source "$GITHUB_WORKSPACE/.github/scripts/ci/run_metax_perf_experimental.sh"'
        )
        completed = subprocess.run(["bash", "-c", command], env=env, text=True,
                                   capture_output=True, timeout=30)
        report = project / "logs/metax-experimental/results.tsv"
        rows = [line.split("\t") for line in report.read_text().splitlines()[1:]]
        calls = [json.loads(line) for line in
                 (project / "invocations.jsonl").read_text().splitlines()]
        return completed, rows, calls, project

    def test_complete_matrix_and_allocation_build_flags(self):
        completed, rows, calls, project = self.run_matrix()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(len(rows), 9)
        self.assertTrue(all(row[1] == "PASS" for row in rows), rows)
        self.assertEqual(len(calls), 9)
        algorithms = {call["env"].get("FLAGCX_UNIRUNNER_ALGO") for call in calls}
        self.assertTrue({"2", "4", "6", "7", "8", "9", "11"} <= algorithms)
        for call in calls:
            self.assertEqual(call["env"]["np"], "4")
            self.assertEqual(call["env"]["MACA_VISIBLE_DEVICES"], "0,2,4,6")
            self.assertEqual(call["env"]["FLAGCX_SIGNAL_HOST_ENABLE"], "0")
            self.assertEqual(call["env"]["FLAGCX_CI_EXPECT_DEVICE_STREAM_CONTROL"], "1")
        builds = (project / "builds.log").read_text()
        self.assertIn("COMPILE_KERNEL=1 FORCE_DEFAULT_PATH=1", builds)
        self.assertIn("unirunner-experimental", builds)
        perf_builds = [line.split() for line in builds.splitlines()
                       if line.startswith("build-tests make ") and "/test/perf " in line]
        self.assertEqual(len(perf_builds), 1, builds)
        self.assertIn("host_api", perf_builds[0])
        self.assertNotIn("all", perf_builds[0])
        self.assertNotIn("device_api", perf_builds[0])
        reports = list((project / "logs").rglob("rank-*.xml"))
        self.assertEqual(len(reports), 16)
        for report in reports:
            self.assertEqual(report.stat().st_mode & 0o777, 0o644)
            self.assertEqual(report.parent.stat().st_mode & 0o777, 0o755)

    def test_failed_invocation_blocks_remaining_gpu_submissions(self):
        completed, rows, calls, _ = self.run_matrix(FAIL_ALGORITHM="7")
        self.assertEqual(completed.returncode, 19, completed.stderr)
        self.assertEqual(rows[-2][1:3], ["FAIL", "19"])
        self.assertEqual(rows[-1][0:2], ["SlicedAR", "BLOCKED"])
        self.assertEqual(len(calls), 8)

    def test_skipped_required_case_is_not_a_pass(self):
        completed, rows, calls, _ = self.run_matrix(SKIP_CASE="AllGatherGemm")
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(rows[0][1], "FAIL")
        self.assertTrue(all(row[1] == "BLOCKED" for row in rows[1:]))
        self.assertEqual(len(calls), 1)
        self.assertIn("skipped or not run", completed.stderr)

    def test_missing_execution_evidence_is_not_a_pass(self):
        completed, rows, _, _ = self.run_matrix(MISSING_EXECUTOR="1")
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(rows[0][1], "FAIL")
        self.assertIn("Missing actual UniRunner algorithm=9 evidence", completed.stderr)

    def test_missing_native_launcher_is_not_a_pass(self):
        completed, rows, _, _ = self.run_matrix(MISSING_NATIVE="1")
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(rows[0][1], "FAIL")
        self.assertIn("Missing native executor launch evidence", completed.stderr)

    def test_missing_allocator_policy_is_not_a_pass(self):
        completed, rows, _, _ = self.run_matrix(MISSING_POLICY="1")
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(rows[0][1], "FAIL")
        self.assertIn("Missing MetaX UniRunner allocation policy evidence", completed.stderr)

    def test_missing_rank_xml_is_not_a_pass(self):
        completed, rows, _, _ = self.run_matrix(MISSING_RANK="1")
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(rows[0][1], "FAIL")
        self.assertIn("expected four rank XML", completed.stderr)

    def test_experimental_preflight_does_not_require_unused_mccl_alltoall(self):
        spec = importlib.util.spec_from_file_location("metax_preflight", CI / "check_metax_devices.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for workload, expected, calls in (("perf-test-experimental", 0, 0), ("perf", 1, 1)):
            with self.subTest(workload=workload), \
                    mock.patch("sys.argv", ["preflight", workload]), \
                    mock.patch.object(module, "runtime_device_count", return_value=4), \
                    mock.patch.object(module.shutil, "which", return_value=None), \
                    mock.patch.object(module, "check_mccl_alltoall", side_effect=RuntimeError("missing")) as check, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(), expected)
                self.assertEqual(check.call_count, calls)

    def test_workflow_requires_experimental_and_always_uploads(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/test.yml").read_text())
        jobs = workflow["jobs"]
        job = jobs["perf-test-experimental-metax"]
        self.assertIn("'metax'", job["if"])
        self.assertIn("perf-test-experimental-metax", jobs["perf_complete"]["needs"])
        self.assertIn("METAX_EXPERIMENTAL_RESULT", jobs["perf_complete"]["env"])
        uploads = [step for step in job["steps"]
                   if step.get("uses", "").startswith("actions/upload-artifact@")]
        self.assertEqual(len(uploads), 1)
        self.assertEqual(uploads[0]["if"], "always()")


if __name__ == "__main__":
    unittest.main()
