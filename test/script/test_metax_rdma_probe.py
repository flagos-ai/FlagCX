"""Hardware-free checks for the MetaX diagnostic supervisor and CI policy."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github/scripts/ci/run_rdma_atomic_probe.py"
spec = importlib.util.spec_from_file_location("metax_rdma_probe", SCRIPT)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class MetaXAtomicProbeTest(unittest.TestCase):
    def worker_arguments(self, root):
        arguments = []
        for worker, option in (("native", "--native-binary"), ("adaptor", "--adaptor-binary"),
                               ("barrier", "--barrier")):
            binary = root / worker
            binary.touch()
            binary.chmod(0o755)
            arguments += [option, str(binary)]
        return arguments

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
            if "--inventory" in command:
                log.write_text('PROBE_JSON ' + json.dumps(dict(
                    event="inventory", gpu_count=4, gpu_count_code=0,
                    ports=["bnxt_roce0:1"])) + '\n')
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
        for failed in ("native", "adaptor", "barrier"):
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
                        worker = Path(command[4]).name
                        built.append(worker)
                        log.write_text(f"build {worker}\n")
                        outcome["returncode"] = 7 if worker == failed else 0
                    elif "--inventory" in command:
                        log.write_text('PROBE_JSON ' + json.dumps(dict(
                            event="inventory", ports=["bnxt_roce0:1"],
                            gpu_count=4, gpu_count_code=0)) + '\n')
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
                self.assertEqual(built, ["native", "adaptor", "barrier"])
                self.assertNotIn(failed, executed)
                self.assertTrue(set(built) - {failed} <= set(executed))
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
