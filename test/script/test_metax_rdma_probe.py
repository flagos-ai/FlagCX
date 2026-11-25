"""Hardware-free checks for the MetaX diagnostic supervisor and CI policy."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github/scripts/ci/run_rdma_atomic_probe.py"
spec = importlib.util.spec_from_file_location("metax_rdma_probe", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
snapshot_spec = importlib.util.spec_from_file_location(
    "rdma_probe_snapshot", SCRIPT.with_name("rdma_probe_snapshot.py"))
snapshot = importlib.util.module_from_spec(snapshot_spec)
snapshot_spec.loader.exec_module(snapshot)


class MetaXAtomicProbeTest(unittest.TestCase):
    def worker_arguments(self, root):
        arguments = []
        for worker, option in (("sdk", "--sdk-inventory"), ("native", "--native-binary"), ("adaptor", "--adaptor-binary"),
                               ("barrier", "--barrier")):
            binary = root / worker
            binary.touch()
            binary.chmod(0o755)
            arguments += [option, str(binary)]
        return arguments

    def inventory_child(self, command, log):
        if "--host-inventory" in command:
            record = dict(event="host_inventory", ports=["bnxt_roce0:1"], entries=[], errors=[])
        elif "--count" in command or "--inventory" in command:
            record = dict(event="gpu_inventory", gpu_count=4, gpu_count_code=0)
        elif "--runtime-version" in command or "--driver-version" in command:
            record = dict(event="sdk_version", code=0, version=1)
        else:
            return False
        log.write_text('PROBE_BOOT main_enter\nPROBE_JSON ' + json.dumps(record) + '\n')
        return True

    def test_allocation_flags_and_real_gdr_have_full_baselines(self):
        self.assertEqual(len(probe.MODES), 10)
        matrix = probe.build_matrix(["bnxt_roce0:1", "bnxt_roce1:1"], 4096)
        self.assertEqual(len({case["id"] for case in matrix}), len(matrix))
        for mode in range(10):
            with self.subTest(allocation=probe.MODES[mode]):
                for size in (8, 4096):
                    self.assertTrue(any(c["layer"] == "allocation" and c["mode"] == mode
                                        and c["bytes"] == size for c in matrix))
                    for layer in ("native", "adaptor"):
                        for operation in ("write", "read", "fetch_add"):
                            for direction in (0, 1):
                                self.assertTrue(any(
                                    c["layer"] == layer and c["mode"] == mode
                                    and c["bytes"] == size and c["operation"] == operation
                                    and c["direction"] == direction
                                    and c["hca"] == "bnxt_roce0:1" for c in matrix))
                self.assertTrue(any(c["layer"] == "adaptor" and c["mode"] == mode
                                    and c["hca"] == "bnxt_roce1:1" for c in matrix))

    def test_only_metax_enables_suite_and_host_signals_are_forced_off(self):
        for path in (ROOT / ".github/configs").glob("*.yml"):
            config = yaml.safe_load(path.read_text())
            if not isinstance(config, dict) or "unit_test_suites" not in config:
                continue
            self.assertEqual("rdma_atomic_probe" in config["unit_test_suites"],
                             path.stem == "metax", path.name)
        case = probe.build_matrix(["bnxt_roce0:1"], 4096)[-1]
        with mock.patch.dict(os.environ, FLAGCX_SIGNAL_HOST_ENABLE="1",
                             FLAGCX_DEVICE_ONE_SIDED_FORCE_NET="1",
                             FLAGCX_CI_UNIFIED_IR_SKIP_S16="1"):
            env = probe.case_environment(case)
        self.assertEqual(env["FLAGCX_SIGNAL_HOST_ENABLE"], "0")
        self.assertEqual(env["FLAGCX_VMM_ENABLE"], "0")
        self.assertNotIn("FLAGCX_CI_UNIFIED_IR_SKIP_S16", env)
        command = probe.case_command(case, Path("native"), Path("barrier"), ["mpirun"])
        self.assertEqual(command.count("FLAGCX_SIGNAL_HOST_ENABLE=0"), 2)
        self.assertEqual(command.count("FLAGCX_CI_RDMA_PROBE_BARRIER_ONLY=1"), 2)

    def test_pass_requires_every_rank_and_preserves_first_error(self):
        outcome = dict(returncode=0, timed_out=False)
        case = dict(layer="native", ranks=2)
        records = [dict(event="result", status="PASS", rank=0)]
        self.assertEqual(probe.classify(case, outcome, records), "MISSING_RESULT")
        records.append(dict(event="result", status="PASS", rank=1))
        self.assertEqual(probe.classify(case, outcome, records), "PASS")
        records.insert(0, dict(event="error", status="CQ_ERROR", rank=0))
        self.assertEqual(probe.classify(case, outcome, records), "CQ_ERROR")
        self.assertEqual(probe.classify(case, dict(outcome, timed_out=True), records), "TIMEOUT")
        self.assertEqual(probe.classify(case, dict(outcome, timed_out=True, reaped=False), records),
                         "WORKER_STUCK")

    def test_host_discovery_is_independent_and_preserves_read_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, state in (("bnxt1/ports/2", "4: ACTIVE"),
                                ("bnxt0/ports/1", "4: ACTIVE"),
                                ("bnxt0/ports/2", "2: INIT")):
                port = root / name
                port.mkdir(parents=True)
                (port / "state").write_text(state)
            (root / "bnxt2/ports/1").mkdir(parents=True)
            inventory = probe.discover_ports(root)
            self.assertEqual(inventory["ports"], ["bnxt0:1", "bnxt1:2"])
            self.assertEqual(len(inventory["entries"]), 3)
            self.assertEqual(len(inventory["errors"]), 1)

    def test_four_port_matrix_reserves_bounded_snapshot_time_inside_job(self):
        cases = probe.build_matrix([f'bnxt{i}:1' for i in range(4)], 4096)
        self.assertEqual(len(cases), 469)
        matrix_budget = sum((180 if c['layer'] == 'barrier' else 10) + probe.CLEANUP_ALLOWANCE
                            for c in cases)
        self.assertAlmostEqual(matrix_budget, 6108.7)
        # The 110-minute supervisor budget includes setup and optional metadata;
        # the enclosing 120-minute job retains ten minutes for project setup.
        self.assertLess(matrix_budget + 300 + 5 * (5 + probe.CLEANUP_ALLOWANCE) + 30, 6600)
        self.assertLess(6600, 120 * 60)

    def test_host_inventory_timeout_retains_partial_port_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = dict(page_size=4096, cases=probe.build_matrix([], 4096))
            def child(command, log, timeout):
                log.write_text('PROBE_JSON {"event":"host_port","hca":"bnxt0","port":1}\n')
                return dict(returncode=-9, timed_out=True, reaped=True)
            with mock.patch.object(probe, "run_child", side_effect=child):
                with self.assertRaises(probe.StartupFailure) as error:
                    probe.host_inventory(root, report)
            self.assertEqual(error.exception.status, "STARTUP_TIMEOUT")
            self.assertEqual(report["host_inventory_run"]["records"][0]["hca"], "bnxt0")
            self.assertTrue((root / 'results.json').is_file())

    def test_gpu_inventory_falls_back_only_after_reap_and_requires_a_real_count(self):
        for failure in ("empty", "timeout", "sdk_error", "invalid_count"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                report = dict(page_size=4096, cases=probe.build_matrix([], 4096))
                builds = {name: dict(available=True, binary=name) for name in ('sdk', 'native', 'adaptor')}
                calls = []
                def child(command, log, timeout, env):
                    name = command[0]
                    calls.append(name)
                    outcome = dict(returncode=0, timed_out=False, reaped=True)
                    if name != 'sdk':
                        self.inventory_child(command, log)
                        return outcome
                    records = []
                    if failure == 'timeout':
                        records = [dict(event='stage', stage='mcGetDeviceCount', phase='enter')]
                        outcome.update(timed_out=True, returncode=-9)
                    elif failure in ('sdk_error', 'invalid_count'):
                        records = [dict(event='gpu_inventory', gpu_count_code=1 if failure == 'sdk_error' else 0,
                                        gpu_count=999 if failure == 'sdk_error' else '4')]
                        outcome['returncode'] = int(failure == 'sdk_error')
                    log.write_text(''.join('PROBE_JSON ' + json.dumps(r) + '\n' for r in records))
                    return outcome
                with mock.patch.object(probe, 'run_child', side_effect=child):
                    result = probe.gpu_inventory(SimpleNamespace(inventory_timeout=15), root, report, builds)
                self.assertEqual(calls, ['sdk', 'native'])
                self.assertEqual(result['gpu_count'], 4)
                first, second = report['gpu_inventory_attempts']
                self.assertNotEqual(first['status'], 'PASS')
                self.assertEqual(second['status'], 'PASS')
                if failure == 'timeout':
                    self.assertEqual(first['status'], 'STARTUP_TIMEOUT')
                    self.assertEqual(first['last_stage'], 'mcGetDeviceCount/enter')

    def test_unreaped_inventory_stops_gpu_submissions_and_retains_host_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'results'
            cases = [c for c in probe.build_matrix(['bnxt_roce0:1'], 4096) if c['layer'] == 'allocation']
            arguments = [str(SCRIPT)] + self.worker_arguments(root) + ['--output', str(output)]
            calls = []
            def child(command, log, timeout, env=None):
                calls.append(command)
                self.assertTrue(self.inventory_child(command, log))
                if '--count' in command:
                    # Even a valid partial count must not override failed cleanup.
                    return dict(returncode=None, timed_out=True, reaped=False, surviving_workers=[1044])
                return dict(returncode=0, timed_out=False, reaped=True)
            with mock.patch.object(sys, 'argv', arguments), \
                 mock.patch.object(probe, 'build_matrix', return_value=cases), \
                 mock.patch.object(probe, 'collect_provenance', return_value=[]), \
                 mock.patch.object(probe, 'run_child', side_effect=child), \
                 mock.patch.object(probe.signal, 'signal'), \
                 mock.patch.dict(os.environ, GITHUB_STEP_SUMMARY=''):
                self.assertEqual(probe.main(), 1)
            self.assertEqual(len(calls), 2)  # Host discovery, then SDK; no fallback/MPI.
            report = json.loads((output / 'results.json').read_text())
            self.assertEqual(report['setup_status'], 'WORKER_STUCK')
            self.assertNotIn('inventory', report)
            self.assertTrue((output / 'host-inventory.json').is_file())
            self.assertEqual({c['mode'] for c in report['cases']}, set(range(10)))
            self.assertEqual({c['status'] for c in report['cases']}, {'BLOCKED'})

    def test_reaped_startup_timeout_does_not_prevent_allocation_matrix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'results'
            cases = [c for c in probe.build_matrix(['bnxt_roce0:1'], 4096) if c['layer'] == 'allocation']
            arguments = [str(SCRIPT)] + self.worker_arguments(root) + ['--output', str(output)]
            calls = []
            def child(command, log, timeout, env=None):
                calls.append(command)
                outcome = dict(returncode=0, timed_out=False, reaped=True)
                if self.inventory_child(command, log):
                    if '--count' in command:
                        log.write_text('PROBE_BOOT main_enter\nPROBE_JSON '
                                       '{"event":"stage","stage":"mcGetDeviceCount","phase":"enter"}\n')
                        outcome.update(timed_out=True, returncode=-9)
                    elif '--runtime-version' in command:
                        # Optional metadata is allowed to time out after all cases.
                        outcome.update(timed_out=True, returncode=-9)
                    return outcome
                log.write_text(''.join('PROBE_JSON ' + json.dumps(dict(
                    event='result', status='PASS', rank=rank)) + '\n' for rank in (0, 1)))
                return outcome
            with mock.patch.object(sys, 'argv', arguments), \
                 mock.patch.object(probe, 'build_matrix', return_value=cases), \
                 mock.patch.object(probe, 'collect_provenance', return_value=[]), \
                 mock.patch.object(probe, 'run_child', side_effect=child), \
                 mock.patch.object(probe.signal, 'signal'), \
                 mock.patch.dict(os.environ, GITHUB_STEP_SUMMARY=''):
                self.assertEqual(probe.main(), 1)  # Failed startup control remains visible.
            matrix_calls = [c for c in calls if '--mode' in c]
            self.assertEqual(len(matrix_calls), 20)
            self.assertEqual({int(c[c.index('--mode') + 1]) for c in matrix_calls}, set(range(10)))
            first_version = next(i for i,c in enumerate(calls) if '--runtime-version' in c)
            self.assertTrue(all(i < first_version for i,c in enumerate(calls) if '--mode' in c))
            report = json.loads((output / 'results.json').read_text())
            self.assertEqual({c['status'] for c in report['cases']}, {'PASS'})
            self.assertTrue(report['sdk_versions'][0]['outcome']['timed_out'])

    def test_snapshot_retains_errors_partial_reads_and_rejects_reused_pids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc, output = root / 'proc', root / 'output'
            process = proc / '123'
            process.mkdir(parents=True)
            output.mkdir()
            stat = '123 (worker) S ' + ' '.join(['0'] * 18 + ['456'] + ['0'] * 10)
            (process / 'stat').write_text(stat)
            (process / 'status').write_text('State:\tS\n')
            snapshot.snapshot_process(proc, 123, '456', output)
            record = json.loads((output / 'pid-123.json').read_text())
            self.assertTrue(record['complete'])
            self.assertIn('error', record['fields']['stack'])
            original = snapshot.read_field
            def blocked(path, limit=65536):
                if path.name == 'wchan':
                    raise RuntimeError('simulate collector termination during a read')
                return original(path, limit)
            with mock.patch.object(snapshot, 'read_field', side_effect=blocked):
                with self.assertRaises(RuntimeError):
                    snapshot.snapshot_process(proc, 123, '456', output)
            partial = json.loads((output / 'pid-123.json').read_text())
            self.assertFalse(partial['complete'])
            self.assertTrue(partial['reading'].endswith('/wchan'))
            self.assertIn('State:', partial['fields']['status']['text'])
            snapshot.snapshot_process(proc, 123, '999', output)
            reused = json.loads((output / 'pid-123.json').read_text())
            self.assertIn('reused', reused['error'])
            self.assertFalse(reused['fields'])

    def test_adaptor_cq_errors_retain_valid_provider_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "case.log"
            log.write_text(
                'PROBE_JSON {"event":"start","pid":123,"rank":1}\n'
                'te-cicd:123:456 [0] FLAGCX WARN Got completion status=2 opcode=4 len=0 '
                'vendor err 3 qp_num=1404 wr_id=0 (IPut)\n')
            record = probe.parse_log(log)[-1]
            self.assertEqual(record["rank"], 1)
            self.assertEqual(record["wc_status"], 2)
            self.assertEqual(record["vendor_err"], 3)
            self.assertEqual(record["qpn"], 1404)
            self.assertNotIn("opcode", record)
            self.assertEqual(probe.classify(dict(layer="adaptor", ranks=2),
                                           dict(returncode=0, timed_out=False),
                                           probe.parse_log(log)), "CQ_ERROR")

    def test_timeout_does_not_prevent_next_process_or_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = probe.run_child([sys.executable, "-c", "import time; time.sleep(10)"],
                                    root / "timeout.log", 0.05, grace=0.05)
            self.assertTrue(first["timed_out"])
            self.assertTrue(first["reaped"])
            self.assertEqual([d["phase"] for d in first["diagnostics"]],
                             ["before-term", "after-term", "after-kill"])
            for snapshot in first["diagnostics"]:
                self.assertTrue((Path(snapshot["directory"]) / "collector.json").is_file())
            second = probe.run_child([sys.executable, "-c", "print('NEXT_CASE')"],
                                     root / "next.log", 5)
            self.assertEqual(second["returncode"], 0)
            self.assertIn("NEXT_CASE", (root / "next.log").read_text())

    def test_failures_continue_and_failed_allocation_blocks_only_its_dependents(self):
        matrix = probe.build_matrix(["bnxt_roce0:1"], 4096)
        select = lambda layer, mode, size: dict(next(c for c in matrix
                                                   if c["layer"] == layer
                                                   and c["mode"] == mode and c["bytes"] == size))
        cases = [select("allocation", 0, 8), select("native", 0, 8),
                 select("allocation", 2, 4096), select("native", 2, 4096),
                 select("allocation", 9, 4096), select("adaptor", 9, 4096)]
        calls = []

        def child(command, log, timeout, env=None, **kwargs):
            outcome = dict(returncode=0, timed_out=False, reaped=True, elapsed_s=0)
            if self.inventory_child(command, log):
                return outcome
            mode = int(command[command.index("--mode") + 1])
            layer = command[command.index("--layer") + 1]
            calls.append((layer, mode))
            status = "ALLOC_ERROR" if mode == 0 else "CQ_ERROR" if layer == "native" else "PASS"
            outcome["returncode"] = int(status != "PASS")
            records = [dict(event="result", status=status, rank=rank) for rank in (0, 1)]
            log.write_text(''.join('PROBE_JSON ' + json.dumps(r) + '\n' for r in records))
            return outcome

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            arguments = [str(SCRIPT)] + self.worker_arguments(Path(temporary)) + [
                "--output", str(output)]
            with mock.patch.object(sys, "argv", arguments), \
                 mock.patch.object(probe, "build_matrix", return_value=cases), \
                 mock.patch.object(probe, "collect_provenance", return_value=[]), \
                 mock.patch.object(probe, "run_child", side_effect=child), \
                 mock.patch.object(probe.signal, "signal"), \
                 mock.patch.dict(os.environ, GITHUB_STEP_SUMMARY=""):
                self.assertEqual(probe.main(), 1)
            report = json.loads((output / "results.json").read_text())
            self.assertEqual([c["status"] for c in report["cases"]],
                             ["ALLOC_ERROR", "BLOCKED", "PASS", "CQ_ERROR", "PASS", "PASS"])
            self.assertNotIn(("native", 0), calls)
            self.assertIn(("adaptor", 9), calls)
            self.assertNotIn("PENDING", (output / "summary.md").read_text())

    def test_worker_build_failure_preserves_other_layers_and_build_evidence(self):
        for failed in ("sdk", "native", "adaptor", "barrier"):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                output = root / "results"
                # These executable files deliberately survive a failed build:
                # an old worker must never be mistaken for a successful build.
                arguments = [str(SCRIPT)] + self.worker_arguments(root) + [
                    "--build-root", str(root), "--output", str(output)]
                full_matrix = probe.build_matrix(["bnxt_roce0:1"], 4096)
                cases = [c for c in full_matrix if c["layer"] == "allocation"]
                cases += [next(c for c in full_matrix if c["layer"] == layer)
                          for layer in ("native", "adaptor", "barrier")]
                built, executed = [], []

                def child(command, log, timeout, env=None, **kwargs):
                    outcome = dict(returncode=0, timed_out=False, reaped=True)
                    if command[0] == "make":
                        worker = Path(command[4]).name.replace("sdk-inventory", "sdk")
                        built.append(worker)
                        log.write_text(f"build {worker}\n")
                        outcome["returncode"] = 7 if worker == failed else 0
                    elif self.inventory_child(command, log):
                        pass
                    else:
                        worker = ("barrier" if "--layer" not in command else
                                  Path(command[command.index("--layer") - 1]).name)
                        executed.append(worker)
                        if worker == "barrier":
                            log.write_text("=== Final Result: ALL PASS ===\n")
                        else:
                            log.write_text(''.join('PROBE_JSON ' + json.dumps(dict(
                                event="result", status="PASS", rank=rank)) + '\n'
                                for rank in (0, 1)))
                    return outcome

                with mock.patch.object(sys, "argv", arguments), \
                     mock.patch.object(probe, "build_matrix", return_value=cases), \
                     mock.patch.object(probe, "collect_provenance", return_value=[]), \
                     mock.patch.object(probe, "run_child", side_effect=child), \
                     mock.patch.object(probe.signal, "signal"), \
                     mock.patch.dict(os.environ, GITHUB_STEP_SUMMARY=""):
                    self.assertEqual(probe.main(), 1)
                self.assertEqual(built, ["sdk", "native", "adaptor", "barrier"])
                self.assertNotIn(failed, executed)
                self.assertTrue(set(built) - {failed, "sdk"} <= set(executed))
                builds = json.loads((output / "build.json").read_text())
                report = json.loads((output / "results.json").read_text())
                for worker in built:
                    self.assertTrue((output / "build" / f"{worker}.log").is_file())
                    self.assertEqual(builds[worker]["status"],
                                     "BUILD_ERROR" if worker == failed else "READY")
                for case in report["cases"]:
                    self.assertEqual(case["status"],
                                     "BUILD_ERROR" if case["layer"] == failed else "PASS")
                self.assertEqual(len([c for c in report["cases"]
                                      if c["layer"] == "allocation"]), 20)


if __name__ == "__main__":
    unittest.main()
