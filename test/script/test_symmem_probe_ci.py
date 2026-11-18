"""Exercise diagnostic routing and XML gates without a GPU or MPI runtime."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]

FAKE_GTEST = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
import xml.etree.ElementTree as E
args = sys.argv[1:]
case = next(a.split('=',1)[1] for a in args if a.startswith('--gtest_filter='))
mpi = '--allow-run-as-root' in args
if mpi:
    ranks = sum(int(args[i+1]) for i,a in enumerate(args) if a == '-np')
    i = next(i for i,a in enumerate(args) if a.endswith('run_gtest_rank_xml.sh'))
    output = Path(args[i+1])
    paths = [output / f'rank-{rank}.xml' for rank in range(ranks)]
else:
    paths = [Path(next(a.split('xml:',1)[1] for a in args if a.startswith('--gtest_output=')))]
with open(os.environ['PROBE_CALLS'], 'a') as f:
    f.write(json.dumps({'case':case, 'mpi':mpi, 'args':args, 'dmabuf':os.getenv('FLAGCX_DMABUF_ENABLE'), 'vmm':os.getenv('FLAGCX_VMM_ENABLE')})+'\n')
if os.getenv('FAIL_CASE') == case: sys.exit(7)
for path in paths:
    if os.getenv('DROP_LAST_RANK') == '1' and mpi and path == paths[-1]: continue
    suite, name = case.split('.',1)
    tree = E.Element('testsuites')
    node = E.SubElement(tree, 'testsuite', name=suite)
    E.SubElement(node, 'testcase', name=name, status='run', result='completed')
    path.parent.mkdir(parents=True, exist_ok=True)
    E.ElementTree(tree).write(path)
'''


class SymmemProbeCiTest(unittest.TestCase):
    def run_probe(self, restrictive_umask=False, **extra):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ci = root / '.github/scripts/ci'
            ci.mkdir(parents=True)
            binary = root / 'test/unittest/symmem_probe/build/bin'
            binary.mkdir(parents=True)
            for name in ('check_required_gtests.py', 'run_symmem_probe.sh',
                         'run_unit_test.sh', 'filter_supplemental_tests.sh',
                         'rdma_static_preflight.sh'):
                shutil.copy(ROOT / '.github/scripts/ci' / name, ci)
            platform_dir = root / '.github/scripts/set_env'
            platform_dir.mkdir()
            platform = platform_dir / 'iluvatar.sh'
            shutil.copy(ROOT / '.github/scripts/set_env/iluvatar.sh', platform)
            # Keep the real platform configuration and dispatcher. Stub only
            # builds, hardware inventory and execution of GPU/MPI binaries.
            mock_bin = root / 'mock-bin'
            mock_bin.mkdir()
            for name in ('cmake', 'make', 'mpirun', 'clang', 'ixsmi', 'gdb'):
                command = mock_bin / name
                command.write_text('#!/bin/bash\nexit 0\n')
                command.chmod(0o755)
            (mock_bin / 'make').write_text('''#!/bin/bash
umask 022
printf '%s\\n' "$*" >> "$PROBE_MAKE_CALLS"
if [[ "$*" == *iluvatar-vmm-diagnostics* ]]; then
  for arg in "$@"; do
    if [[ "$arg" == ILUVATAR_VMM_DIAGNOSTICS_DIR=* ]]; then
      directory=${arg#*=}
      mkdir -p "$directory"
      printf 'mock SDK diagnostic\\n' > "$directory/complete.log"
    fi
  done
elif [[ "${FAIL_BUILD:-0}" == 1 ]]; then
  exit 17
fi
''')
            nproc = mock_bin / 'nproc'
            nproc.write_text('#!/bin/bash\necho 1\n')
            nproc.chmod(0o755)
            wrapper = ci / 'run_with_timeout.sh'
            wrapper.write_text('#!/bin/bash\nexec "$@"\n')
            wrapper.chmod(0o755)
            fake = binary / 'symmem_probe_unit_tests'
            fake.write_text(FAKE_GTEST)
            fake.chmod(0o755)
            shutil.copy(fake, ci / 'run_mpi_with_timeout.sh')
            calls = root / 'calls.jsonl'
            env = os.environ.copy()
            env.update(GITHUB_WORKSPACE=str(root), PROBE_CALLS=str(calls),
                       PROBE_MAKE_CALLS=str(root / 'make-calls'),
                       PATH=f"{mock_bin}:{env['PATH']}", MPI_HOME=str(root / 'mpi'),
                       FLAGCX_CI_COVERAGE='0', FLAGCX_CI_PROBE_HOSTS='',
                       FLAGCX_CI_LOG_DIR=str(root / 'logs'), BASH_ENV='', **extra)
            command = ['bash', str(ci / 'run_unit_test.sh'), str(platform), 'symmem_probe']
            if restrictive_umask:
                command = ['bash', '-c', 'umask 077; exec "$@"', 'bash', *command]
            result = subprocess.run(
                command,
                env=env, text=True, capture_output=True, timeout=30,
            )
            self.artifact_modes = {
                str(path.relative_to(root / 'logs')): path.stat().st_mode & 0o777
                for path in (root / 'logs').rglob('*')
            }
            self.make_calls = (root / 'make-calls').read_text().splitlines()
            return result, ([json.loads(line) for line in calls.read_text().splitlines()]
                            if calls.exists() else [])

    def test_dispatcher_preserves_platform_mpi_topology(self):
        result, calls = self.run_probe()
        self.assertEqual(result.returncode, 0, result.stderr)
        mpi_calls = [call for call in calls if call['mpi']]
        self.assertEqual(len(mpi_calls), 7)
        for call in mpi_calls:
            args = call['args']
            with self.subTest(case=call['case']):
                self.assertIn('COREX_VISIBLE_DEVICES=0,1', args)
                self.assertIn('CUDA_VISIBLE_DEVICES=0,1', args)
                self.assertIn('FLAGCX_HOSTID=node0', args)
                self.assertIn('NCCL_HOSTID=node0', args)
                if ':' in args:
                    left, right = args[:args.index(':')], args[args.index(':') + 1:]
                    self.assertNotIn('FLAGCX_HOSTID=node1', left)
                    for value in ('COREX_VISIBLE_DEVICES=4,5',
                                  'CUDA_VISIBLE_DEVICES=4,5',
                                  'FLAGCX_HOSTID=node1', 'NCCL_HOSTID=node1'):
                        self.assertIn(value, right)
                    self.assertNotIn('FLAGCX_HOSTID=node0', right)
                else:
                    self.assertNotIn('FLAGCX_HOSTID=node1', args)

    def test_runs_every_case_independently_with_dmabuf_disabled(self):
        result, calls = self.run_probe()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 14)
        self.assertEqual(calls[0]['case'], 'SymmemProbePrerequisites.VmmCallbacks')
        self.assertEqual(sum(c['mpi'] for c in calls), 7)
        self.assertTrue(all(c['dmabuf'] == '0' and c['vmm'] == '1' for c in calls))

    def test_missing_vmm_prerequisite_blocks_components_and_mpi(self):
        result, calls = self.run_probe(FAIL_CASE='SymmemProbePrerequisites.VmmCallbacks')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([call['case'] for call in calls],
                         ['SymmemProbePrerequisites.VmmCallbacks'])
        self.assertIn('VMM prerequisite failed', result.stderr)
        self.assertTrue(any('iluvatar-vmm-diagnostics' in call for call in self.make_calls))

    def test_artifacts_remain_readable_with_restrictive_container_umask(self):
        for fail_case in ('', 'SymmemProbePrerequisites.VmmCallbacks'):
            with self.subTest(fail_case=fail_case):
                result, _ = self.run_probe(restrictive_umask=True, FAIL_CASE=fail_case)
                self.assertEqual(result.returncode == 0, not fail_case, result.stderr)
                results = [name for name in self.artifact_modes
                           if name.startswith('symmem-probe.') and '/' not in name]
                self.assertEqual(len(results), 1)
                self.assertEqual(self.artifact_modes[results[0]], 0o755)
                for name, mode in self.artifact_modes.items():
                    if name.endswith(('.xml', '.log')):
                        self.assertEqual(mode & 0o444, 0o444, name)

    def test_sdk_diagnostics_survive_a_later_build_failure(self):
        result, calls = self.run_probe(FAIL_BUILD='1')
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertEqual(calls, [])
        self.assertIn('iluvatar-vmm-diagnostics', self.make_calls[0])
        self.assertIn('iluvatar-vmm-sdk/complete.log', self.artifact_modes)

    def test_failed_allocation_blocks_mpi_and_keeps_failure(self):
        result, calls = self.run_probe(FAIL_CASE='VmmProbe.AllocationAndRange')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(c['mpi'] for c in calls))
        self.assertIn('MPI probes blocked', result.stderr)

    def test_missing_rank_xml_cannot_pass(self):
        result, calls = self.run_probe(DROP_LAST_RANK='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sum(c['mpi'] for c in calls), 7)
        self.assertIn('cannot read GoogleTest XML', result.stderr)

    def test_net_component_failure_preserves_independent_mpi_diagnostics(self):
        result, calls = self.run_probe(FAIL_CASE='VmmProbe.VaMrReadWrite')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sum(c['mpi'] for c in calls), 7)


if __name__ == '__main__':
    unittest.main()
