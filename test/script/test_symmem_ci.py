"""Check fixed SymMem CI selection and MPI exit status handling."""

import os
import re
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def install_lifecycle_reports(ci):
    shutil.copyfile(REPO_ROOT / '.github/scripts/ci/check_required_gtests.py',
                    ci / 'check_required_gtests.py')
    (ci / 'mock_lifecycle_xml.py').write_text("""import sys
from pathlib import Path
for i, arg in enumerate(sys.argv):
    if arg.endswith('run_gtest_rank_xml.sh'):
        root = Path(sys.argv[i + 1])
        names = ('EmptyRanksAndDelayedImporterJoinBeforeRelease', 'FailedImporterKeepsAllExportersUntilCommonRetry', 'DeviceSelectionFailureIsAgreedBeforeAnyClose', 'BackingReleaseFailureRetriesWithoutRepeatingReleasedOwners', 'NativePreparationConvergesBeforeEnteringCollectiveDestroy', 'PendingEarlierNativeOwnerCompletesBeforeNextCollective', 'PendingEarlierCompatibilityCloseCompletesBeforeNextCollective', 'HostShmGenerationIsIssuedByRootAndRetained', 'CounterOnlyCreationLeavesNextShmGenerationAligned', 'PendingExplicitFlushKeepsEveryExporterUntilRetry')
        body = ''.join('<testcase name="%s" status="run" result="completed"/>' % name for name in names)
        for rank in range(4):
            report = root / ('rank-%s.xml' % rank)
            report.write_text('<testsuites><testsuite name="IpcFinalizeBootstrapTest">' + body + '</testsuite></testsuites>')
            report.chmod(0o600)
        break
""")


class SymmemCiTest(unittest.TestCase):
    def test_required_lifecycle_cases_exist_in_real_mpi_source(self):
        source = (REPO_ROOT / 'test/unittest/symmem/coll_ipc_finalize.cpp').read_text()
        actual = set(re.findall(
            r'TEST_F\(IpcFinalizeBootstrapTest,\s*(\w+)\)', source))
        script = (REPO_ROOT / 'test/script/symmem_test.sh').read_text()
        required = set(re.findall(r'--case IpcFinalizeBootstrapTest\.(\w+)', script))
        self.assertGreaterEqual(len(actual), 10)
        self.assertEqual(required, actual)
        self.assertNotIn('HeteroFinalizeMpi', script)

    def test_lifecycle_artifacts_survive_strict_umask_and_failed_mpi(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / 'test/script'
            scripts.mkdir(parents=True)
            shutil.copy(REPO_ROOT / 'test/script/symmem_test.sh', scripts)
            ci = root / '.github/scripts/ci'
            ci.mkdir(parents=True)
            install_lifecycle_reports(ci)
            shutil.copy(REPO_ROOT / '.github/scripts/ci/ignore_test.py', ci)
            for name in ('run_with_timeout.sh', 'run_mpi_with_timeout.sh'):
                wrapper = ci / name
                wrapper.write_text('''#!/bin/bash
python3 "$(dirname "$0")/mock_lifecycle_xml.py" "$@" || exit $?
if [[ "$*" == *run_gtest_rank_xml.sh* ]]; then
  exit "$CI_LIFECYCLE_STATUS"
fi
''')
                wrapper.chmod(0o755)
            for status in (0, 7):
                with self.subTest(status=status):
                    log_root = root / f'logs-{status}'
                    result = subprocess.run(
                        ['bash', '-c', 'umask 077; exec bash "$1"', 'bash',
                         str(scripts / 'symmem_test.sh')],
                        env={**os.environ, 'FLAGCX_ADAPTOR': 'enflame',
                             'FLAGCX_CI_SYMMEM_VMM_SUPPORTED': '0',
                             'FLAGCX_CI_REQUIRE_MULTICAST': '0',
                             'FLAGCX_CI_IGNORE_TEST': '{}',
                             'FLAGCX_CI_LOG_DIR': str(log_root),
                             'CI_LIFECYCLE_STATUS': str(status)},
                        text=True, capture_output=True, timeout=20)
                    self.assertEqual(result.returncode, status, result.stderr)
                    reports = list(log_root.glob('symmem-lifecycle/results.*/rank-*.xml'))
                    self.assertEqual(len(reports), 4)
                    for report in reports:
                        self.assertEqual(report.stat().st_mode & 0o777, 0o644)
                        for parent in (log_root, report.parent.parent, report.parent):
                            self.assertEqual(parent.stat().st_mode & 0o777, 0o755)

    def test_iluvatar_matches_cuda_except_untracked_vmm_ipc_with_va_only_mr(self):
        source = (REPO_ROOT / '.github/scripts/ci/run_unit_test.sh').read_text()
        start = source.index('    symmem)\n')
        end = source.index('    device_api)\n', start)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / 'test/script'
            scripts.mkdir(parents=True)
            shutil.copy(REPO_ROOT / 'test/script/symmem_test.sh', scripts)
            ci = root / '.github/scripts/ci'
            ci.mkdir(parents=True)
            install_lifecycle_reports(ci)
            shutil.copyfile(REPO_ROOT / '.github/scripts/ci/ignore_test.py',
                            ci / 'ignore_test.py')
            for name in ('run_with_timeout.sh', 'run_mpi_with_timeout.sh'):
                wrapper = ci / name
                wrapper.write_text('''#!/usr/bin/env python3
import json, os, sys, subprocess
from pathlib import Path
subprocess.check_call([sys.executable, str(Path(__file__).with_name('mock_lifecycle_xml.py')), *sys.argv[1:]])
with open(os.environ['CI_CAPTURE'], 'a') as output:
    output.write(json.dumps({'label': os.getenv('FLAGCX_CI_MPI_LABEL', os.getenv('FLAGCX_CI_TEST_LABEL')),
                             'args': sys.argv[1:]}) + '\\n')
''')
                wrapper.chmod(0o755)
            harness = root / 'run.sh'
            harness.write_text('''#!/bin/bash
set -eo pipefail
SET_ENV_SCRIPT=$1
PROJECT_ROOT=$2
SUITE=symmem
source "$SET_ENV_SCRIPT"
flagcx_ci_configure_suite symmem
MPI_RUNNER="$PROJECT_ROOT/.github/scripts/ci/run_mpi_with_timeout.sh"
TEST_RUNNER="$PROJECT_ROOT/.github/scripts/ci/run_with_timeout.sh"
''' + source[source.index('flagcx_ci_report_skipped_suite() {'):source.index('\nif [[ "$(basename "$SET_ENV_SCRIPT" .sh)" == kunlunxin ]]')] + '''
run_suite() {
  case "$SUITE" in
''' + source[start:end] + '''  esac
}
run_suite
''')
            results = {}
            for platform in ('cuda', 'iluvatar'):
                capture = root / f'{platform}.jsonl'
                env = os.environ.copy()
                for name in ('FLAGCX_CI_SYMMEM_VMM_SUPPORTED', 'FLAGCX_DMABUF_ENABLE',
                             'FLAGCX_CI_REQUIRE_MULTICAST'):
                    env.pop(name, None)
                env.update(CI_CAPTURE=str(capture), MPI_HOME=str(root / 'mpi'))
                run = subprocess.run(
                    ['bash', str(harness), str(REPO_ROOT / f'.github/scripts/set_env/{platform}.sh'), str(root)],
                    env=env, text=True, capture_output=True, timeout=30)
                self.assertEqual(run.returncode, 0, run.stderr)
                if platform == 'iluvatar':
                    self.assertIn('Skipping UntrackedVmmMappingIsRejected on Iluvatar', run.stdout)
                else:
                    self.assertNotIn('Skipping', run.stdout)
                results[platform] = [json.loads(line) for line in capture.read_text().splitlines()]
            results['cuda'] = [call for call in results['cuda']
                               if call['label'] != 'SymMemTest.UntrackedVmmMappingIsRejected']
            self.assertEqual([call['label'] for call in results['cuda']],
                             [call['label'] for call in results['iluvatar']])
            for cuda, iluvatar in zip(results['cuda'], results['iluvatar']):
                self.assertEqual([arg for arg in cuda['args'] if arg.startswith('--gtest_filter=')],
                                 [arg for arg in iluvatar['args'] if arg.startswith('--gtest_filter=')])
                self.assertNotIn('FLAGCX_CI_ALLOW_VMM_NET_UNSUPPORTED=1', iluvatar['args'])
                self.assertNotIn('FLAGCX_DMABUF_ENABLE=1', iluvatar['args'])
                if iluvatar['label'] in ('symmem VMM + NET auto', 'symmem VMM + NET route union'):
                    self.assertIn('FLAGCX_CI_EXPECT_VMM_MR_ROUTE=va', iluvatar['args'])
                    self.assertIn('FLAGCX_CI_SYMMEM_VA_ONLY=1', iluvatar['args'])
                    self.assertIn('FLAGCX_DMABUF_ENABLE=0', iluvatar['args'])

    def test_selected_cases_by_adaptor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "test/script/symmem_test.sh"
            script.parent.mkdir(parents=True)
            shutil.copyfile(REPO_ROOT / "test/script/symmem_test.sh", script)
            ci_dir = root / ".github/scripts/ci"
            ci_dir.mkdir(parents=True)
            install_lifecycle_reports(ci_dir)
            shutil.copyfile(REPO_ROOT / ".github/scripts/ci/ignore_test.py",
                            ci_dir / "ignore_test.py")
            for name in ("run_with_timeout.sh", "run_mpi_with_timeout.sh"):
                runner = ci_dir / name
                runner.write_text(
                    '#!/bin/bash\npython3 "$(dirname "$0")/mock_lifecycle_xml.py" "$@"\nprintf "%s\\n" "$*" >> "$CI_CAPTURE"\n'
                    'if [[ -n "${CI_FAIL_CASE:-}" && "$*" == *"$CI_FAIL_CASE"* ]]; '
                    'then exit 7; fi\n'
                )
                runner.chmod(0o755)

            for adaptor, multi_segment in (
                ("nvidia", True),
                ("du", False),
                ("ppu", True),
                ("metax", True),
                ("klx", False),
                ("iluvatar", True),
                ("musa", False),
            ):
                with self.subTest(adaptor=adaptor):
                    capture = root / f"{adaptor}.calls"
                    env = dict(
                        os.environ, FLAGCX_ADAPTOR=adaptor, CI_CAPTURE=str(capture)
                    )
                    if adaptor == "musa":
                        policy = yaml.safe_load(
                            (REPO_ROOT / ".github/configs/mthreads.yml").read_text()
                        )["ignore_test"]
                        env["FLAGCX_CI_IGNORE_TEST"] = json.dumps(policy)
                    result = subprocess.run(
                        ["bash", str(script)], env=env, capture_output=True, text=True
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    calls = capture.read_text().splitlines()
                    dedicated = [
                        line for line in calls
                        if "--gtest_filter=SymMemTest." in line
                    ]
                    untracked = adaptor not in ('iluvatar', 'musa')
                    self.assertEqual(len(dedicated), (2 if multi_segment else 0) + int(untracked))
                    self.assertEqual(
                        sum(
                            "UntrackedVmmMappingIsRejected" in line
                            for line in dedicated
                        ),
                        int(untracked),
                    )
                    self.assertEqual(
                        sum(
                            "FLAGCX_CI_REQUIRE_MULTI_SEGMENT=1" in line
                            for line in dedicated
                        ),
                        2 if multi_segment else 0,
                    )
                    self.assertEqual(
                        sum(
                            "FLAGCX_VMM_MULTI_SEGMENT_ENABLE=0" in line
                            for line in calls
                        ),
                        2 if adaptor == "du" else 0,
                    )
                    if adaptor == "musa":
                        self.assertTrue(any("symmem_unit_tests" in line for line in calls))
                        self.assertTrue(any("IpcFinalizeBootstrapTest.*" in line for line in calls))
                        self.assertTrue(any("FLAGCX_VMM_ENABLE=0" in line for line in calls))
                        self.assertFalse(any("FLAGCX_VMM_ENABLE=1" in line for line in calls))
                        self.assertIn("Skipping symmem VMM-local tests", result.stdout)

            generic_capture = root / "generic.calls"
            generic_env = os.environ.copy()
            generic_env.pop("FLAGCX_ADAPTOR", None)
            generic_env["CI_CAPTURE"] = str(generic_capture)
            generic = subprocess.run(
                ["bash", str(script)],
                env=generic_env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(generic.returncode, 0, generic.stderr)
            generic_calls = generic_capture.read_text().splitlines()
            self.assertEqual(
                sum("symmem_mpi_tests" in line for line in generic_calls), 3
            )
            self.assertFalse(any(
                "--gtest_filter=SymMemTest." in line for line in generic_calls
            ))
            self.assertNotIn("UntrackedVmmMappingIsRejected", generic.stdout)

            unsupported_capture = root / "vmm-unsupported.calls"
            unsupported = subprocess.run(
                ["bash", str(script)],
                env=dict(
                    generic_env,
                    FLAGCX_CI_IGNORE_TEST=(
                        '{"phases":[{"suite":"symmem",'
                        '"name":"vmm-local-and-remote-without-net",'
                        '"reason":"VMM is unavailable"}]}'
                    ),
                    CI_CAPTURE=str(unsupported_capture),
                ),
                capture_output=True,
                text=True,
            )
            self.assertEqual(unsupported.returncode, 0, unsupported.stderr)
            unsupported_calls = unsupported_capture.read_text().splitlines()
            self.assertTrue(any("symmem_unit_tests" in line for line in unsupported_calls))
            self.assertEqual(
                sum("symmem_mpi_tests" in line for line in unsupported_calls), 2
            )
            self.assertTrue(any("FLAGCX_VMM_ENABLE=0" in line for line in unsupported_calls))
            self.assertFalse(any("FLAGCX_VMM_ENABLE=1" in line for line in unsupported_calls))
            self.assertIn("Skipping symmem VMM-local tests", unsupported.stdout)

            enflame_capture = root / "enflame-vmm-unsupported.calls"
            enflame = subprocess.run(
                ["bash", str(script)],
                env=dict(
                    os.environ,
                    FLAGCX_ADAPTOR="enflame",
                    FLAGCX_CI_SYMMEM_VMM_SUPPORTED="0",
                    CI_CAPTURE=str(enflame_capture),
                ),
                capture_output=True,
                text=True,
            )
            self.assertEqual(enflame.returncode, 0, enflame.stderr)
            enflame_calls = enflame_capture.read_text().splitlines()
            self.assertTrue(any("symmem_unit_tests" in line for line in enflame_calls))
            self.assertTrue(any("symmem_mpi_tests" in line for line in enflame_calls))
            self.assertTrue(any("FLAGCX_VMM_ENABLE=0" in line for line in enflame_calls))
            self.assertFalse(any("FLAGCX_VMM_ENABLE=1" in line for line in enflame_calls))
            self.assertFalse(any("UntrackedVmmMappingIsRejected" in line for line in enflame_calls))
            self.assertIn("Skipping symmem VMM-local tests", enflame.stdout)

            invalid = subprocess.run(
                ["bash", str(script)],
                env=dict(os.environ, FLAGCX_ADAPTOR="unknown"),
                capture_output=True,
                text=True,
            )
            self.assertEqual(invalid.returncode, 2)
            self.assertIn("Unsupported FLAGCX_ADAPTOR", invalid.stderr)

            failed = subprocess.run(
                ["bash", str(script)],
                env=dict(
                    os.environ,
                    FLAGCX_ADAPTOR="du",
                    CI_CAPTURE=str(root / "failed.calls"),
                    CI_FAIL_CASE="UntrackedVmmMappingIsRejected",
                ),
                capture_output=True,
                text=True,
            )
            self.assertEqual(failed.returncode, 7)

            for adaptor, expected_status in (
                ("nvidia", 0), ("metax", 0), ("du", 1), ("ppu", 1)
            ):
                with self.subTest(multicast=adaptor):
                    capture = root / f"{adaptor}-multicast.calls"
                    env = dict(
                        os.environ,
                        FLAGCX_ADAPTOR=adaptor,
                        FLAGCX_CI_REQUIRE_MULTICAST="1",
                        CI_CAPTURE=str(capture),
                    )
                    result = subprocess.run(
                        ["bash", str(script)],
                        env=env,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, expected_status)
                    dedicated = [
                        line for line in capture.read_text().splitlines()
                        if "--gtest_filter=SymMemTest." in line
                    ]
                    self.assertEqual(
                        sum(
                            "FLAGCX_CI_REQUIRE_MULTICAST=1" in line
                            for line in dedicated
                        ),
                        3 if expected_status == 0 else 0,
                    )


if __name__ == "__main__":
    unittest.main()
