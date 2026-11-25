"""Exercise lifecycle selection and rank-complete XML gating without SDKs."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which("timeout"), "GNU timeout is required")
class LifecycleCiTest(unittest.TestCase):
    def run_device_host(self, platform, inherited="*", binary=True):
        source = (ROOT / ".github/scripts/ci/run_unit_test.sh").read_text()
        policies = source[source.index('flagcx_ci_report_skipped_suite() {'):
                          source.index('\nif [[ "$(basename "$SET_ENV_SCRIPT" .sh)" == kunlunxin ]]')]
        run = source[source.index('run_device_api_host() {'):
                     source.index('flagcx_ci_device_api_intra_only() {')]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ci = root / '.github/scripts/ci'
            ci.mkdir(parents=True)
            shutil.copy(ROOT / '.github/scripts/ci/ignore_test.py', ci)
            target = root / 'test/unittest/device_api/build/bin/test_dev_comm_cleanup'
            target.parent.mkdir(parents=True)
            if binary:
                target.touch()
            runner = root / 'runner'
            runner.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
with open(os.environ['CALLS'], 'a') as out:
    out.write(json.dumps({'args': sys.argv[1:], 'filter': os.getenv('GTEST_FILTER'),
                          'vmm': os.getenv('FLAGCX_VMM_ENABLE')}) + '\\n')
if sys.argv[1] != 'make' and not Path(sys.argv[1]).is_file():
    sys.exit(127)
''')
            runner.chmod(0o755)
            harness = root / 'run.sh'
            harness.write_text('''#!/bin/bash
set -euo pipefail
source "$1"
PROJECT_ROOT=$2
TEST_RUNNER=$3
SUITE=device_api_host
if declare -F flagcx_ci_configure_suite >/dev/null; then
  flagcx_ci_configure_suite "$SUITE"
fi
ignore_filter=$(python3 "$PROJECT_ROOT/.github/scripts/ci/ignore_test.py" gtest "$SUITE")
if [[ -n "$ignore_filter" ]]; then
  if [[ "${GTEST_FILTER:-*}" == *-* ]]; then
    GTEST_FILTER+=":$ignore_filter"
  else
    GTEST_FILTER="${GTEST_FILTER:-*}-$ignore_filter"
  fi
  export GTEST_FILTER
fi
''' + policies + run + '\nrun_device_api_host\n')
            result = subprocess.run(
                ['bash', str(harness), str(ROOT / f'.github/scripts/set_env/{platform}.sh'),
                 str(root), str(runner)],
                env={**os.environ, 'GTEST_FILTER': inherited, 'CALLS': str(root / 'calls'),
                     'FLAGCX_CI_IGNORE_TEST': json.dumps(yaml.safe_load(
                         (ROOT / f'.github/configs/{platform}.yml').read_text())['ignore_test'])},
                text=True, capture_output=True, timeout=20)
            calls = [json.loads(line) for line in (root / 'calls').read_text().splitlines()]
            return result, calls

    def test_enflame_does_not_execute_the_disabled_host_binary(self):
        result, calls = self.run_device_host('enflame', binary=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['args'][0], 'make')
        self.assertIn('SKIPPED device_api_host/non-vmm-rma', result.stdout)
        self.assertIn('no test coverage', result.stdout)

    def test_enabled_host_target_must_build_its_binary(self):
        result, calls = self.run_device_host('cuda', binary=False)
        self.assertEqual(result.returncode, 127, result.stderr)
        self.assertEqual(len(calls), 2)

    def test_kunlunxin_preserves_cleanup_coverage_without_host_callbacks(self):
        result, calls = self.run_device_host('kunlunxin')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 1)
        positive, _, negative = calls[0]['filter'].partition('-')
        self.assertEqual(positive, '*')
        self.assertEqual(set(negative.split(':')), {
            'CommStreamIsolation.FinalizeDoesNotDrainIndependentCommunicator',
            'CommStreamIsolation.PartialOwnerCachesDoNotDrainIndependentCommunicator',
            'CommStreamIsolation.HeteroRmaNonVmmFinalizeDoesNotDrainIndependentWork'})
        self.assertIn('excluded by the effective GTEST_FILTER', result.stdout)

    def test_fresh_rma_honors_positive_and_negative_selection(self):
        case = 'CommStreamIsolation.HeteroRmaNonVmmFinalizeDoesNotDrainIndependentWork'
        for platform, pattern, selected in (
            ('cuda', '*', True),
            ('cuda', 'CommStreamIsolation.*', True),
            ('cuda', 'DefaultDevCommCleanupTest.*:CommStreamIsolation.*', True),
            ('cuda', 'DefaultDevCommCleanupTest.*', False),
            ('cuda', '*-' + case, False),
            ('cuda', '-CommStreamIsolation.*', False),
            ('cuda', '*-*.HeteroRma*', False),
            ('mthreads', '*', True),
            ('ascend', '*', True),
            ('iluvatar', '*', True),
            ('ppu', '*', True),
        ):
            with self.subTest(platform=platform, pattern=pattern):
                result, calls = self.run_device_host(platform, pattern)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(calls), 2 if selected else 1)
                self.assertIn(case, calls[0]['filter'].partition('-')[2])
                if selected:
                    self.assertEqual(calls[1]['filter'], case)
                    self.assertEqual(calls[1]['vmm'], '0')

    def test_compat_finalize_excludes_only_native_isolation(self):
        native_cases = {
            'CommStreamIsolation.FinalizeDoesNotDrainIndependentCommunicator',
            'CommStreamIsolation.PartialOwnerCachesDoNotDrainIndependentCommunicator'}
        for platform in ('ascend', 'mthreads', 'enflame'):
            with self.subTest(platform=platform):
                result, calls = self.run_device_host(platform)
                self.assertEqual(result.returncode, 0, result.stderr)
                excluded = set(calls[0]['filter'].partition('-')[2].split(':'))
                self.assertTrue(native_cases.issubset(excluded))
                self.assertEqual(excluded - native_cases, {
                    'CommStreamIsolation.HeteroRmaNonVmmFinalizeDoesNotDrainIndependentWork'})
                if platform != 'enflame':
                    self.assertEqual(len(calls), 2)

    def test_public_lifecycle_stays_enabled_with_compat_finalize(self):
        case = 'FlagCXCollTest.FinalizeThenRankLocalDestroy'
        for platform in ('ascend', 'mthreads', 'enflame', 'kunlunxin'):
            with self.subTest(platform=platform):
                result = subprocess.run(
                    ['bash', '-c', 'source "$1"; python3 "$2" gtest runner',
                     'bash', str(ROOT / f'.github/scripts/set_env/{platform}.sh'),
                     str(ROOT / '.github/scripts/ci/ignore_test.py')],
                    env={**os.environ, 'FLAGCX_CI_IGNORE_TEST': json.dumps(yaml.safe_load(
                        (ROOT / f'.github/configs/{platform}.yml').read_text())['ignore_test'])},
                    text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn(case, result.stdout)

    def run_wrapper(self, mode="pass", inherited="*", selected="*"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher = root / "mpirun"
            launcher.write_text('''#!/usr/bin/env python3
import os, subprocess, sys
n = int(sys.argv[2])
for rank in range(n):
    if os.getenv('TEST_XML_MODE') == 'missing' and rank == 1:
        continue
    result = subprocess.run(sys.argv[3:], env={**os.environ, 'OMPI_COMM_WORLD_RANK': str(rank)})
    if result.returncode:
        sys.exit(result.returncode)
''')
            launcher.chmod(0o755)
            binary = root / "runner_mpi_tests"
            binary.write_text('''#!/usr/bin/env python3
import os, sys
from pathlib import Path
print('FILTER ' + ' '.join(sys.argv[1:]), flush=True)
report = os.getenv('GTEST_OUTPUT', '')
if report:
    skipped = os.getenv('TEST_XML_MODE') == 'skip' and os.getenv('OMPI_COMM_WORLD_RANK') == '1'
    case = '<testcase name="FinalizeThenRankLocalDestroy" status="run" result="%s"/>' % ('skipped' if skipped else 'completed')
    Path(report.removeprefix('xml:')).write_text('<testsuites><testsuite name="FlagCXCollTest">' + case + '</testsuite></testsuites>')
'''.replace("report.removeprefix('xml:')", "report[4:]"))
            binary.chmod(0o755)
            return subprocess.run(
                ["bash", str(ROOT / ".github/scripts/ci/run_mpi_with_timeout.sh"),
                 "-np", "2", str(binary), "--gtest_filter=" + selected],
                env={**os.environ, "FLAGCX_CI_MPIRUN": str(launcher),
                     "FLAGCX_CI_LOG_DIR": str(root / "logs"),
                     "FLAGCX_CI_TEST_DIAGNOSTICS_AFTER": "0",
                     "GTEST_FILTER": inherited, "TEST_XML_MODE": mode},
                text=True, capture_output=True, timeout=20,
            )

    def test_every_rank_must_report_success(self):
        self.assertEqual(self.run_wrapper().returncode, 0)
        for mode in ("missing", "skip"):
            with self.subTest(mode=mode):
                self.assertNotEqual(self.run_wrapper(mode).returncode, 0)

    def test_platform_exclusion_survives_an_explicit_pxn_filter(self):
        case = "FlagCXCollTest.FinalizeThenRankLocalDestroy"
        result = self.run_wrapper("missing", "*-" + case, case)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--gtest_filter=" + case + "-" + case, result.stdout)
        self.assertNotIn("expected 2 lifecycle XML", result.stderr)


if __name__ == "__main__":
    unittest.main()
