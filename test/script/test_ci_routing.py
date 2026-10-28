import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX_SCRIPT = REPO_ROOT / ".github/scripts/ci/resolve_ci_matrix.rb"
TIMEOUT_SCRIPT = REPO_ROOT / ".github/scripts/ci/run_with_timeout.sh"
GATE_SCRIPT = REPO_ROOT / ".github/scripts/ci/check_ci_results.sh"
CONFIG_DIR = REPO_ROOT / ".github/configs"


class CiMatrixRoutingTest(unittest.TestCase):
    def resolve(self, context):
        with tempfile.TemporaryDirectory() as temp_dir:
            context_path = Path(temp_dir) / "context.json"
            output_path = Path(temp_dir) / "github-output"
            context_path.write_text(json.dumps(context))
            env = os.environ.copy()
            env["GITHUB_OUTPUT"] = str(output_path)
            result = subprocess.run(
                ["ruby", str(MATRIX_SCRIPT), str(CONFIG_DIR), str(context_path)],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            outputs = dict(line.split("=", 1) for line in output_path.read_text().splitlines())
            outputs["matrix"] = json.loads(outputs["matrix"])
            outputs["platforms"] = json.loads(outputs["platforms"])
            return outputs, result.stdout

    def test_push_and_schedule_always_run_all_enabled_platforms(self):
        for event_name in ("push", "schedule"):
            with self.subTest(event=event_name):
                outputs, _ = self.resolve({"event_name": event_name})
                self.assertEqual(outputs["platforms"], ["cuda", "hygon", "iluvatar", "metax", "ppu"])

    def test_common_changes_select_all_platforms(self):
        outputs, _ = self.resolve(
            {"event_name": "pull_request", "files": ["flagcx/core/group.cc"]}
        )
        self.assertEqual(outputs["platforms"], ["cuda", "hygon", "iluvatar", "metax", "ppu"])

    def test_platform_specific_changes_select_only_matching_platform(self):
        for path, expected in (
            ("test/kernel/maca/device_api_host_helpers.cu", ["metax"]),
            ("flagcx/adaptor/device/cuda_adaptor.cc", ["cuda"]),
            ("flagcx/adaptor/device/ixcuda_adaptor.cc", ["iluvatar"]),
            ("docker/iluvatar-ci/Dockerfile", ["iluvatar"]),
            (".github/scripts/ci/run_ppu_workload.sh", ["ppu"]),
            ("flagcx/adaptor/device/duhip_adaptor.cc", ["hygon"]),
            ("flagcx/adaptor/ccl/durccl_adaptor.cc", ["hygon"]),
            ("test/kernel/du/device_ir.hip", ["hygon"]),
            ("test/kernel/du/nested/device_ir.hip", ["hygon"]),
            ("flagcx/adaptor/kernel/du/nested/kernel.hip", ["hygon"]),
            ("test/kernel/maca/nested/foo.cu", ["metax"]),
            ("flagcx/adaptor/kernel/nvidia/nested/foo.cu", ["cuda"]),
            ("test/kernel/nvidia/nested/foo.cu", ["cuda"]),
            (".github/scripts/ci/run_cuda_workload.sh", ["cuda"]),
            (".github/scripts/ci/run_cuda_legacy_perf.sh", ["cuda"]),
            (".github/scripts/ci/run_metax_workload.sh", ["metax"]),
        ):
            with self.subTest(path=path):
                outputs, _ = self.resolve(
                    {"event_name": "pull_request", "files": [path]}
                )
                self.assertEqual(outputs["platforms"], expected)

    def test_platform_labels_override_paths_and_can_select_multiple_platforms(self):
        outputs, _ = self.resolve(
            {
                "event_name": "pull_request",
                "labels": ["platform:ppu", "platform:hygon"],
                "files": ["README.md"],
            }
        )
        self.assertEqual(outputs["platforms"], ["hygon", "ppu"])

    def test_all_label_forces_full_matrix(self):
        outputs, _ = self.resolve(
            {"event_name": "pull_request", "labels": ["platform:all"]}
        )
        self.assertEqual(outputs["platforms"], ["cuda", "hygon", "iluvatar", "metax", "ppu"])

    def test_empty_pr_changes_skip_hardware_matrix(self):
        outputs, _ = self.resolve(
            {"event_name": "pull_request", "files": []}
        )
        self.assertEqual(outputs["platforms"], [])
        self.assertEqual(outputs["has_platforms"], "false")

    def test_unclassified_pr_changes_use_full_matrix(self):
        outputs, _ = self.resolve(
            {"event_name": "pull_request", "files": ["README.md"]}
        )
        self.assertEqual(outputs["platforms"], ["cuda", "hygon", "iluvatar", "metax", "ppu"])

    def test_unmapped_ci_source_change_falls_back_to_all_platforms(self):
        outputs, _ = self.resolve(
            {
                "event_name": "pull_request",
                "files": [".github/scripts/ci/new_backend_ci.sh"],
            }
        )
        self.assertEqual(outputs["platforms"], ["cuda", "hygon", "iluvatar", "metax", "ppu"])

    def test_mixed_platform_and_unknown_changes_run_all_platforms(self):
        outputs, _ = self.resolve(
            {"event_name": "pull_request", "files": [
                "flagcx/adaptor/device/cuda_adaptor.cc", "new/shared/file.cc"
            ]}
        )
        self.assertEqual(outputs["platforms"], ["cuda", "hygon", "iluvatar", "metax", "ppu"])

    def test_unregistered_vendor_paths_use_full_matrix(self):
        outputs, _ = self.resolve({
            "event_name": "pull_request",
            "files": ["flagcx/adaptor/device/mlu_adaptor.cc"],
        })
        self.assertEqual(outputs["platforms"], ["cuda", "hygon", "iluvatar", "metax", "ppu"])

    def test_fork_repository_and_immutable_ref_are_forwarded(self):
        outputs, _ = self.resolve({
            "event_name": "pull_request", "repository": "contributor/FlagCX",
            "ref": "a" * 40, "labels": ["platform:cuda"],
        })
        self.assertEqual(outputs["repository"], "contributor/FlagCX")
        self.assertEqual(outputs["ref"], "a" * 40)

    def test_comment_without_platform_respects_labels(self):
        outputs, _ = self.resolve({
            "event_name": "issue_comment", "comment": "/test", "permission": "admin",
            "is_pull_request": True, "labels": ["platform:ppu"],
            "files": ["flagcx/core/group.cc"],
        })
        self.assertEqual(outputs["platforms"], ["ppu"])

    def test_every_platform_config_has_watch_patterns(self):
        for platform in ("cuda", "hygon", "iluvatar", "metax", "ppu"):
            with self.subTest(platform=platform):
                config = (CONFIG_DIR / f"{platform}.yml").read_text()
                self.assertIn("paths_watch:", config)
                self.assertIn(f".github/configs/{platform}.yml", config)

    def test_dispatch_selection_and_comment_commands(self):
        for event_name, extra, expected in (
            ("workflow_dispatch", {"selection": "cuda,ppu"}, ["cuda", "ppu"]),
            (
                "issue_comment",
                {
                    "comment": "/test metax",
                    "permission": "write",
                    "is_pull_request": True,
                },
                ["metax"],
            ),
            (
                "issue_comment",
                {
                    "comment": "/test all",
                    "permission": "maintain",
                    "is_pull_request": True,
                },
                ["cuda", "hygon", "iluvatar", "metax", "ppu"],
            ),
            (
                "issue_comment",
                {
                    "comment": "/test",
                    "permission": "write",
                    "is_pull_request": True,
                    "files": ["flagcx/adaptor/device/cuda_adaptor.cc"],
                },
                ["cuda"],
            ),
        ):
            with self.subTest(event=event_name, extra=extra):
                outputs, _ = self.resolve({"event_name": event_name, **extra})
                self.assertEqual(outputs["platforms"], expected)

    def test_comment_trigger_requires_pull_request_command_and_write_permission(self):
        cases = (
            {"comment": "/test", "permission": "read", "is_pull_request": True},
            {"comment": "/test", "permission": "write", "is_pull_request": False},
            {"comment": "please /test metax", "permission": "write", "is_pull_request": True},
        )
        for extra in cases:
            with self.subTest(extra=extra):
                outputs, _ = self.resolve({"event_name": "issue_comment", **extra})
                self.assertEqual(outputs["active"], "false")
                self.assertEqual(outputs["platforms"], [])

    def test_unknown_label_and_selection_fail_closed(self):
        for context in (
            {"event_name": "pull_request", "labels": ["platform:unknown"]},
            {"event_name": "workflow_dispatch", "selection": "unknown"},
        ):
            with self.subTest(context=context), tempfile.TemporaryDirectory() as temp_dir:
                context_path = Path(temp_dir) / "context.json"
                context_path.write_text(json.dumps(context))
                env = os.environ.copy()
                env["GITHUB_OUTPUT"] = str(Path(temp_dir) / "output")
                result = subprocess.run(
                    ["ruby", str(MATRIX_SCRIPT), str(CONFIG_DIR), str(context_path)],
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertNotEqual(result.returncode, 0)


class CiRetryTest(unittest.TestCase):
    def run_wrapper(self, command, *, timeout="5s"):
        env = os.environ.copy()
        env.update(
            {
                "FLAGCX_CI_TEST_TIMEOUT": timeout,
                "FLAGCX_CI_TEST_KILL_AFTER": "1s",
                "FLAGCX_CI_TEST_LABEL": "retry-test",
            }
        )
        return subprocess.run(
            ["bash", str(TIMEOUT_SCRIPT), "bash", "-c", command],
            capture_output=True,
            text=True,
            env=env,
            timeout=15,
        )

    def test_transient_failure_retries_and_is_reported_as_flaky(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            marker = Path(temp_dir) / "first-attempt"
            summary = Path(temp_dir) / "summary.md"
            command = 'if [[ ! -e "$MARKER" ]]; then touch "$MARKER"; exit 7; fi'
            env = os.environ.copy()
            env["MARKER"] = str(marker)
            env["GITHUB_STEP_SUMMARY"] = str(summary)
            env["FLAGCX_CI_LOG_DIR"] = str(Path(temp_dir) / "attempt-logs")
            env.update(
                {
                    "FLAGCX_CI_TEST_TIMEOUT": "5s",
                    "FLAGCX_CI_TEST_KILL_AFTER": "1s",
                    "FLAGCX_CI_TEST_LABEL": "retry-test",
                }
            )
            result = subprocess.run(
                ["bash", str(TIMEOUT_SCRIPT), "bash", "-c", command],
                capture_output=True,
                text=True,
                env=env,
                timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("passed on retry 1", result.stdout)
            self.assertIn("::warning title=Flaky CI test::", result.stdout)
            self.assertIn("First attempt log", summary.read_text())
            attempt_dirs = list((Path(temp_dir) / "attempt-logs").glob("retry-test.*"))
            self.assertEqual(len(attempt_dirs), 1)
            self.assertTrue((attempt_dirs[0] / "attempt-1.log").exists())
            self.assertTrue((attempt_dirs[0] / "attempt-2.log").exists())

    def test_persistent_failure_retries_once_then_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            count_file = Path(temp_dir) / "count"
            command = 'n=$(cat "$COUNT_FILE" 2>/dev/null || echo 0); echo $((n + 1)) > "$COUNT_FILE"; exit 9'
            env = os.environ.copy()
            env["COUNT_FILE"] = str(count_file)
            env.update(
                {
                    "FLAGCX_CI_TEST_TIMEOUT": "5s",
                    "FLAGCX_CI_TEST_KILL_AFTER": "1s",
                    "FLAGCX_CI_TEST_LABEL": "retry-test",
                }
            )
            result = subprocess.run(
                ["bash", str(TIMEOUT_SCRIPT), "bash", "-c", command],
                capture_output=True,
                text=True,
                env=env,
                timeout=15,
            )
            attempts = count_file.read_text().strip()
        self.assertEqual(result.returncode, 9)
        self.assertEqual(attempts, "2")

    def test_timeout_is_retried_and_can_recover(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            marker = Path(temp_dir) / "first-attempt"
            command = 'if [[ ! -e "$MARKER" ]]; then touch "$MARKER"; sleep 3; else exit 0; fi'
            env = os.environ.copy()
            env["MARKER"] = str(marker)
            env.update(
                {
                    "FLAGCX_CI_TEST_TIMEOUT": "1s",
                    "FLAGCX_CI_TEST_KILL_AFTER": "1s",
                    "FLAGCX_CI_TEST_LABEL": "timeout-test",
                }
            )
            result = subprocess.run(
                ["bash", str(TIMEOUT_SCRIPT), "bash", "-c", command],
                capture_output=True,
                text=True,
                env=env,
                timeout=15,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("passed on retry 1", result.stdout)
        self.assertIn("exceeded its 1s timeout on attempt 1", result.stderr)

    def test_retry_policy_is_fixed_to_one_retry(self):
        result = self.run_wrapper("exit 8")
        self.assertEqual(result.returncode, 8)
        self.assertEqual(result.stderr.count("retrying once"), 1)

    def test_identical_labels_keep_separate_invocation_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            env = dict(os.environ, FLAGCX_CI_LOG_DIR=temporary,
                       FLAGCX_CI_TEST_LABEL="same-label")
            for message in ("first invocation", "second invocation"):
                result = subprocess.run(
                    ["bash", str(TIMEOUT_SCRIPT), "printf", "%s\\n", message],
                    capture_output=True, text=True, env=env, timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
            logs = list(Path(temporary).glob("same-label.*/attempt-1.log"))
            self.assertEqual(len(logs), 2)
            self.assertEqual({log.read_text().strip() for log in logs},
                             {"first invocation", "second invocation"})

    def test_cancellation_stops_workload_without_retrying(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "started"
            env = dict(os.environ, FLAGCX_CI_LOG_DIR=temporary,
                       FLAGCX_CI_TEST_TIMEOUT="30s", FLAGCX_CI_TEST_KILL_AFTER="1s",
                       MARKER=str(marker))
            process = subprocess.Popen(
                ["bash", str(TIMEOUT_SCRIPT), "bash", "-c",
                 'touch "$MARKER"; sleep 30'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
            )
            try:
                deadline = time.monotonic() + 5
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(marker.exists(), "test workload did not start")
                process.terminate()
                stdout, stderr = process.communicate(timeout=8)
                self.assertEqual(process.returncode, 143, stdout + stderr)
                self.assertNotIn("retrying once", stderr)
                self.assertEqual(len(list(Path(temporary).glob("*/attempt-*.log"))), 1)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=5)


class CiWorkflowWiringTest(unittest.TestCase):
    def test_single_gate_calls_all_platform_workflows(self):
        workflow = (REPO_ROOT / ".github/workflows/unit-test.yml").read_text()
        self.assertIn("name: FlagCX CI", workflow)
        self.assertIn("  ci_complete:\n    name: ci-complete", workflow)
        self.assertIn(
            "needs: [prepare, unit_tests, performance_tests, torch_api_tests, format-check]",
            workflow,
        )
        for reusable in (
            "unit_tests_common.yml",
            "test.yml",
            "torch-api-test.yml",
            "format-check.yml",
        ):
            self.assertIn(reusable, workflow)

    def test_platform_specific_jobs_are_conditioned_on_matrix_selection(self):
        for path in (
            REPO_ROOT / ".github/workflows/test.yml",
            REPO_ROOT / ".github/workflows/torch-api-test.yml",
        ):
            source = path.read_text()
            self.assertIn("workflow_call:", source)
            for platform in ("cuda", "hygon", "metax", "ppu"):
                self.assertIn(f"contains(fromJSON(inputs.platforms), '{platform}')", source)
            self.assertIn("if: always()", source)
            self.assertIn("check_platform_workflow.sh", source)

    def test_torch_coverage_upload_is_optional_and_uses_flagcicd_contract(self):
        workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/torch-api-test.yml").read_text())
        caller = yaml.safe_load((REPO_ROOT / ".github/workflows/unit-test.yml").read_text())
        token = caller["jobs"]["torch_api_tests"]["secrets"]["FLAGCICD_API_TOKEN"]
        self.assertIn("needs.prepare.outputs.repository == github.repository", token)
        job = workflow["jobs"]["torch-api-test-cuda"]
        steps = {step.get("name"): step for step in job["steps"]}
        upload = steps["Upload coverage report to FlagCICD"]
        self.assertTrue(upload["continue-on-error"])
        self.assertEqual(upload["with"]["file_type"], "coverage")
        self.assertEqual(upload["with"]["job_name"], "coverage-cuda-gpu-torch")
        self.assertEqual(upload["with"]["report_path"], "coverage-cuda-gpu-torch.json")
        self.assertEqual(upload["with"]["fail_on_error"], "false")
        self.assertIn("FLAGCICD_TOKEN_SET == 'true'", upload["if"])
        self.assertIn("hashFiles('coverage-cuda-gpu-torch.json')", upload["if"])
        self.assertIn("github.event.issue.number", upload["with"]["pr_id"])
        self.assertNotIn("container", job)
        self.assertIn("Collect Python coverage report", steps)
        workload = steps["Run torch-api-test (cuda) in configured container"]
        self.assertEqual(workload["env"]["FLAGCX_CI_TORCH_COVERAGE"], "1")
        cuda_runner = (REPO_ROOT / ".github/scripts/ci/run_cuda_workload.sh").read_text()
        self.assertIn('bash "$project_root/.github/scripts/ci/run_torch_coverage.sh"', cuda_runner)
        runner = (REPO_ROOT / ".github/scripts/ci/run_torch_coverage.sh").read_text()
        self.assertIn("-m coverage run -m unittest discover", runner)

    def test_ordinary_comments_cannot_cancel_pr_ci(self):
        workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/unit-test.yml").read_text())
        self.assertIn("github.event_name == 'issue_comment' && github.run_id", workflow["concurrency"]["group"])
        self.assertIn("startsWith(github.event.comment.body, '/test')", workflow["jobs"]["prepare"]["if"])


class CiCompletionGateTest(unittest.TestCase):
    def check(self, *results):
        return subprocess.run(
            ["bash", str(GATE_SCRIPT), *results],
            capture_output=True,
            text=True,
        )

    def test_success_and_no_platform_matrix(self):
        self.assertEqual(
            self.check("true", "true", "success", "success", "success", "success", "success").returncode,
            0,
        )
        self.assertEqual(
            self.check("true", "false", "success", "skipped", "success", "skipped", "success").returncode,
            0,
        )
        self.assertEqual(
            self.check("false", "false", "success", "skipped", "skipped", "skipped", "skipped").returncode,
            0,
        )

    def test_selected_unit_skip_or_any_failure_cannot_pass_gate(self):
        for result in (
            ("true", "true", "success", "skipped", "success", "success", "success"),
            ("true", "true", "success", "success", "failure", "success", "success"),
            ("true", "true", "success", "success", "success", "cancelled", "success"),
            ("true", "true", "success", "success", "success", "success", "failure"),
            ("true", "true", "failure", "skipped", "skipped", "skipped", "success"),
        ):
            with self.subTest(results=result):
                self.assertNotEqual(self.check(*result).returncode, 0)


class PlatformWorkflowCompletionTest(unittest.TestCase):
    def check(self, platforms, config, cuda_base, cuda, hygon, metax, ppu):
        return subprocess.run(
            [
                "bash",
                str(REPO_ROOT / ".github/scripts/ci/check_platform_workflow.sh"),
                platforms,
                config,
                cuda_base,
                cuda,
                hygon,
                metax,
                ppu,
            ],
            capture_output=True,
            text=True,
        )

    def test_selected_platform_jobs_must_succeed(self):
        self.assertEqual(
            self.check('["cuda"]', "success", "success", "success", "skipped", "skipped", "skipped").returncode,
            0,
        )
        self.assertEqual(
            self.check('["hygon","ppu"]', "success", "skipped", "skipped", "success", "skipped", "success").returncode,
            0,
        )
        for result in (
            ("[\"cuda\"]", "skipped", "success", "skipped", "skipped", "skipped", "skipped"),
            ("[\"hygon\"]", "skipped", "skipped", "skipped", "failure", "skipped", "skipped"),
            ("[\"ppu\"]", "success", "skipped", "skipped", "skipped", "skipped", "cancelled"),
        ):
            with self.subTest(result=result):
                self.assertNotEqual(self.check(*result).returncode, 0)

    def test_empty_selection_and_invalid_json(self):
        self.assertEqual(
            self.check("[]", "skipped", "skipped", "skipped", "skipped", "skipped", "skipped").returncode,
            0,
        )
        self.assertNotEqual(
            self.check("invalid", "skipped", "skipped", "skipped", "skipped", "skipped", "skipped").returncode,
            0,
        )

    def test_cuda_and_metax_require_successful_platform_configuration(self):
        for platform in ("cuda", "metax"):
            with self.subTest(platform=platform):
                result = self.check(json.dumps([platform]), "failure", "success",
                                    "success", "success", "success", "success")
                self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
