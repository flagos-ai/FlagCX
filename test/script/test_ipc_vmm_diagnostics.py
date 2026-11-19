"""Keep IPC crash diagnostics bounded, independent and unable to hide failure."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / '.github/scripts/ci/run_ipc_vmm_diagnostics.sh'


class IpcVmmDiagnosticsTest(unittest.TestCase):
    def run_diagnostics(self, **extra):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mock = root / 'bin'
            mock.mkdir()
            commands = {
                'gdb': '''import json, os, sys
from pathlib import Path
if '--version' in sys.argv:
    print('test debugger')
    sys.exit(0)
with open(os.environ['CALLS'], 'a') as out:
    out.write(json.dumps(sys.argv[1:]) + '\\n')
i = sys.argv.index('--args')
os.execv(sys.argv[i+1], sys.argv[i+1:])
''',
                'timeout': '''import os, sys
if sys.argv[-1] == os.getenv('TIMEOUT_CASE'):
    sys.exit(124)
os.execvp(sys.argv[4], sys.argv[4:])
''',
                'ldd': "print('test SDK dependencies')\n",
                'sdk-repro': '''import os, sys
case = sys.argv[1]
print('IPC_REPRO begin cudaIpcGetMemHandle', flush=True)
if case == os.getenv('CRASH_CASE'):
    print('Program received signal SIGSEGV', flush=True)
    sys.exit(0)  # A debugger can exit zero even though its inferior crashed.
if case == os.getenv('FAIL_CASE'):
    sys.exit(7)
print('IPC_REPRO_PASS case=' + case, flush=True)
''',
            }
            for name, body in commands.items():
                command = mock / name
                command.write_text(f'#!{sys.executable}\n' + body)
                command.chmod(0o755)
            env = os.environ.copy()
            env.update(PATH=f"{mock}:{env['PATH']}", CALLS=str(root / 'calls'),
                       CRASH_CASE='', FAIL_CASE='', TIMEOUT_CASE='')
            env.update(extra)
            results = root / 'results'
            result = subprocess.run(
                ['bash', str(SCRIPT), str(mock / 'sdk-repro'), str(results)],
                env=env, text=True, capture_output=True, timeout=15)
            calls = [json.loads(line) for line in (root / 'calls').read_text().splitlines()]
            logs = {p.name: p.read_text() for p in results.iterdir()}
            return result, calls, logs

    def test_all_controls_run_under_debugger(self):
        result, calls, logs = self.run_diagnostics()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([args[-1] for args in calls],
                         ['malloc', 'vmm', 'vmm-driver', 'vmm-sync'])
        for args in calls:
            self.assertIn('--return-child-result', args)
            self.assertIn(str(ROOT / '.github/scripts/ci/ipc_vmm_backtrace.gdb'), args)
        self.assertIn('vmm\t0\tPASS', logs['summary.tsv'])
        self.assertIn('test SDK dependencies', logs['libraries.log'])

    def test_debugger_zero_exit_without_pass_marker_is_failure(self):
        result, calls, logs = self.run_diagnostics(CRASH_CASE='vmm')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(calls), 4)
        self.assertEqual(logs['vmm.status'].strip(), '1')
        self.assertIn('SIGSEGV', logs['vmm.log'])
        self.assertIn('vmm-sync\t0\tPASS', logs['summary.tsv'])

    def test_sdk_error_does_not_hide_later_controls(self):
        result, calls, logs = self.run_diagnostics(FAIL_CASE='vmm')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(calls), 4)
        self.assertEqual(logs['vmm.status'].strip(), '7')
        self.assertIn('vmm-driver\t0\tPASS', logs['summary.tsv'])

    def test_timeout_does_not_hide_later_controls(self):
        result, calls, logs = self.run_diagnostics(TIMEOUT_CASE='vmm')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([args[-1] for args in calls], ['malloc', 'vmm-driver', 'vmm-sync'])
        self.assertEqual(logs['vmm.status'].strip(), '124')

    def test_original_symmem_failure_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / 'test/script/symmem_test.sh'
            script.parent.mkdir(parents=True)
            shutil.copy(ROOT / 'test/script/symmem_test.sh', script)
            ci = root / '.github/scripts/ci'
            ci.mkdir(parents=True)
            shutil.copyfile(ROOT / '.github/scripts/ci/ignore_test.py',
                            ci / 'ignore_test.py')
            for name in ('run_with_timeout.sh', 'run_mpi_with_timeout.sh'):
                path = ci / name
                path.write_text('''#!/bin/bash
if [[ "$*" == *--gtest_filter=SymMemTest.UntrackedVmmMappingIsRejected* ]]; then
  exit 139
fi
''')
                path.chmod(0o755)
            (ci / 'run_ipc_vmm_diagnostics.sh').write_text('''#!/bin/bash
printf '%s\\n' "$*" > "$DIAGNOSTIC_CALL"
exit 23
''')
            marker = root / 'diagnostic-call'
            env = dict(os.environ, FLAGCX_ADAPTOR='iluvatar',
                       FLAGCX_CI_REQUIRE_MULTICAST='0', DIAGNOSTIC_CALL=str(marker),
                       FLAGCX_CI_LOG_DIR=str(root / 'logs'))
            result = subprocess.run(['bash', str(script)], env=env,
                                    text=True, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 139, result.stderr)
            self.assertIn('ipc_vmm_repro', marker.read_text())
            self.assertIn('ipc-vmm-diagnostics', marker.read_text())
            self.assertIn('retaining the original SymMem failure', result.stderr)

    def test_reproducer_links_only_sdk_libraries(self):
        for platform in ('USE_NVIDIA=1', 'USE_ILUVATAR=1'):
            with self.subTest(platform=platform):
                result = subprocess.run(
                    ['make', '-n', '-B', '-C', str(ROOT / 'test/unittest/symmem'),
                     'ipc-repro', platform], text=True, capture_output=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stderr)
                command = next(line for line in result.stdout.splitlines()
                               if 'ipc_vmm_repro.cpp' in line)
                self.assertIn('-lcudart', command)
                self.assertIn('-lcuda', command)
                for library in ('-lflagcx', '-lmpi', '-lgtest'):
                    self.assertNotIn(library, command)


if __name__ == '__main__':
    unittest.main()
