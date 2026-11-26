import json
import os
import re
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
PLATFORM_REGISTRY = yaml.safe_load(
    (CONFIG_DIR / "platforms.yml").read_text(encoding="utf-8")
)["platforms"]
ALL_ENABLED_PLATFORMS = [
    name
    for name, settings in PLATFORM_REGISTRY.items()
    if not isinstance(settings, dict) or settings.get("enabled", True)
]


def enabled_platform_or_matrix(platform):
    return [platform] if platform in ALL_ENABLED_PLATFORMS else []


class CiMatrixRoutingTest(unittest.TestCase):
    def test_musa_alias_routes_to_mthreads(self):
        for context in (
            {"event_name": "pull_request", "labels": ["platform:musa"]},
            {"event_name": "pull_request", "labels": ["platform:musa", "platform:mthreads"]},
            {"event_name": "workflow_dispatch", "selection": "musa,mthreads"},
            {"event_name": "issue_comment", "comment": "/test musa",
             "is_pull_request": True, "permission": "write"},
            {"event_name": "issue_comment", "comment": "/test",
             "labels": ["platform:musa"], "is_pull_request": True,
             "permission": "write"},
        ):
            with self.subTest(context=context):
                outputs, _ = self.resolve(context)
                self.assertEqual(outputs["platforms"], ["mthreads"])

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
            outputs["runner_matrix"] = json.loads(outputs["runner_matrix"])
            outputs["platforms"] = json.loads(outputs["platforms"])
            outputs["scheduling"] = json.loads(outputs["scheduling"])
            return outputs, result.stdout

    def test_push_routes_by_changed_paths_and_schedule_runs_full_matrix(self):
        self.assertEqual(
            ALL_ENABLED_PLATFORMS,
            [
                "cuda",
                "hygon",
                "iluvatar",
                "metax",
                "kunlunxin",
                "mthreads",
                "ppu",
                "ascend",
                "enflame",
            ],
        )
        outputs, _ = self.resolve({
            "event_name": "push", "files": [".github/scripts/set_env/metax.sh"]
        })
        self.assertEqual(outputs["platforms"], ["metax"])

        outputs, _ = self.resolve({
            "event_name": "push", "files": ["flagcx/core/group.cc"]
        })
        self.assertEqual(outputs["platforms"], ALL_ENABLED_PLATFORMS)

    def test_suite_selection_labels_and_path_fallback(self):
        outputs, _ = self.resolve({
            "event_name": "pull_request",
            "labels": ["platform:metax", "suite:p2p"],
            "files": ["test/unittest/p2p/test_p2p.cc"],
        })
        self.assertEqual(outputs["matrix"]["include"], [
            {"platform": "metax", "display_name": "MetaX Tests", "suites": ["p2p"]}
        ])

        outputs, _ = self.resolve({
            "event_name": "workflow_dispatch", "selection": "metax,cuda",
            "suites": ["p2p", "rma"],
        })
        self.assertEqual(outputs["matrix"]["include"][0]["suites"], ["p2p", "rma"])
        self.assertEqual(outputs["matrix"]["include"][1]["suites"], ["p2p", "rma"])

        outputs, _ = self.resolve({
            "event_name": "pull_request", "labels": ["platform:metax"],
            "files": ["test/unittest/p2p/test_p2p.cc"],
        })
        self.assertEqual(outputs["matrix"]["include"][0]["suites"], ["p2p"])

        outputs, _ = self.resolve({
            "event_name": "pull_request", "labels": ["platform:metax"],
            "files": ["new/unmapped/change.cc"],
        })
        self.assertEqual(outputs["matrix"]["include"][0]["suites"], [
            "adaptor", "core", "p2p", "rma", "runner", "service", "symmem"
        ])

    def test_suite_dispatch_input_and_nightly_select_all_suites(self):
        outputs, _ = self.resolve({
            "event_name": "workflow_dispatch", "selection": "metax", "suites": ["p2p"]
        })
        self.assertEqual(outputs["matrix"]["include"][0]["suites"], ["p2p"])
        outputs, _ = self.resolve({"event_name": "schedule"})
        self.assertEqual(len(outputs["matrix"]["include"]), len(ALL_ENABLED_PLATFORMS))
        for entry in outputs["matrix"]["include"]:
            expected = yaml.safe_load(
                (CONFIG_DIR / f"{entry['platform']}.yml").read_text()
            )["unit_test_suites"]
            self.assertEqual(entry["suites"], expected)

    def test_pr_and_dispatch_default_to_unit_only_and_push_to_l2(self):
        for event_name, context, expected_level in (
            ("pull_request", {"files": ["test/unittest/p2p/test_p2p.cc"]}, "L1"),
            ("push", {"files": ["test/unittest/p2p/test_p2p.cc"]}, "L2"),
            ("workflow_dispatch", {"selection": "metax"}, "L1"),
        ):
            with self.subTest(event_name=event_name):
                outputs, _ = self.resolve({"event_name": event_name, **context})
                self.assertEqual(outputs["level"], expected_level)
                self.assertEqual(outputs["execution_window"], "day")
                self.assertEqual(outputs["scheduling"]["unit"]["resource_class"], "single-gpu")

        scheduled, _ = self.resolve({"event_name": "schedule"})
        self.assertEqual(scheduled["level"], "L3")
        self.assertEqual(scheduled["execution_window"], "night")
        self.assertEqual(scheduled["scheduling"]["performance"]["gpu_count"], 2)

    def test_requested_suite_unavailable_on_selected_platform_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context_path = Path(temp_dir) / "context.json"
            output_path = Path(temp_dir) / "github-output"
            context_path.write_text(json.dumps({
                "event_name": "workflow_dispatch",
                "selection": "metax",
                "suites": ["device_api"],
            }))
            env = dict(os.environ, GITHUB_OUTPUT=str(output_path))
            result = subprocess.run(
                ["ruby", str(MATRIX_SCRIPT), str(CONFIG_DIR), str(context_path)],
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("do not provide requested suite", result.stderr)

    def test_l3_keeps_explicit_platform_selection(self):
        outputs, _ = self.resolve({
            "event_name": "pull_request",
            "labels": ["level:L3", "platform:metax"],
            "files": ["flagcx/core/group.cc"],
        })
        self.assertEqual(outputs["platforms"], ["metax"])

        outputs, _ = self.resolve({"event_name": "schedule"})
        self.assertEqual(outputs["platforms"], ALL_ENABLED_PLATFORMS)

    def test_common_changes_select_all_platforms(self):
        outputs, _ = self.resolve(
            {"event_name": "pull_request", "files": ["flagcx/core/group.cc"]}
        )
        self.assertEqual(outputs["platforms"], ALL_ENABLED_PLATFORMS)

    def test_platform_specific_changes_select_only_matching_platform(self):
        for path, expected in (
            (
                "test/kernel/maca/device_api_host_helpers.cu",
                enabled_platform_or_matrix("metax"),
            ),
            (
                "flagcx/adaptor/device/cuda_adaptor.cc",
                enabled_platform_or_matrix("cuda"),
            ),
            (
                "flagcx/adaptor/device/ixcuda_adaptor.cc",
                enabled_platform_or_matrix("iluvatar"),
            ),
            (
                "docker/iluvatar-ci/Dockerfile",
                enabled_platform_or_matrix("iluvatar"),
            ),
            (
                "flagcx/adaptor/device/cann_adaptor.cc",
                enabled_platform_or_matrix("ascend"),
            ),
            (".github/scripts/set_env/enflame.sh", ["enflame"]),
            (
                ".github/scripts/ci/run_ppu_workload.sh",
                enabled_platform_or_matrix("ppu"),
            ),
            (
                "flagcx/adaptor/device/duhip_adaptor.cc",
                enabled_platform_or_matrix("hygon"),
            ),
            (
                "flagcx/adaptor/ccl/durccl_adaptor.cc",
                enabled_platform_or_matrix("hygon"),
            ),
            ("test/kernel/du/device_ir.hip", enabled_platform_or_matrix("hygon")),
            (
                "test/kernel/du/nested/device_ir.hip",
                enabled_platform_or_matrix("hygon"),
            ),
            (
                "flagcx/adaptor/kernel/du/nested/kernel.hip",
                enabled_platform_or_matrix("hygon"),
            ),
            (
                "test/kernel/maca/nested/foo.cu",
                enabled_platform_or_matrix("metax"),
            ),
            (
                "flagcx/adaptor/kernel/nvidia/nested/foo.cu",
                enabled_platform_or_matrix("cuda"),
            ),
            (
                "flagcx/adaptor/kernel/common/nested/foo.h",
                [p for p in ("cuda", "hygon") if p in ALL_ENABLED_PLATFORMS]
                or ALL_ENABLED_PLATFORMS,
            ),
            (
                "test/kernel/nvidia/nested/foo.cu",
                enabled_platform_or_matrix("cuda"),
            ),
            (
                ".github/scripts/ci/run_cuda_workload.sh",
                enabled_platform_or_matrix("cuda"),
            ),
            (
                ".github/scripts/ci/run_cuda_perf_experimental.sh",
                enabled_platform_or_matrix("cuda"),
            ),
            (
                ".github/scripts/ci/run_hygon_perf_experimental.sh",
                enabled_platform_or_matrix("hygon"),
            ),
            (
                ".github/scripts/ci/run_metax_workload.sh",
                enabled_platform_or_matrix("metax"),
            ),
        ):
            with self.subTest(path=path):
                outputs, _ = self.resolve(
                    {"event_name": "pull_request", "files": [path]}
                )
                self.assertEqual(outputs["platforms"], expected)

    def test_platform_labels_override_paths_for_enabled_platforms(self):
        for platform in ("enflame", "kunlunxin"):
            with self.subTest(platform=platform):
                outputs, _ = self.resolve(
                    {
                        "event_name": "pull_request",
                        "labels": [f"platform:{platform}"],
                        "files": ["README.md"],
                    }
                )
                self.assertEqual(outputs["platforms"], [platform])

    def test_all_label_forces_full_matrix(self):
        for label in ("platform:all", "platform:all,cuda", "platform:cuda,all"):
            with self.subTest(label=label):
                outputs, _ = self.resolve(
                    {"event_name": "pull_request", "labels": [label]}
                )
                self.assertEqual(outputs["platforms"], ALL_ENABLED_PLATFORMS)

    def test_shared_device_api_entry_point_selects_all_suites(self):
        outputs, _ = self.resolve({
            "event_name": "pull_request",
            "files": ["flagcx/adaptor/flagcx_device.cc"],
        })
        self.assertEqual(outputs["platforms"], ALL_ENABLED_PLATFORMS)
        for entry in outputs["matrix"]["include"]:
            config = yaml.safe_load(
                (CONFIG_DIR / f"{entry['platform']}.yml").read_text()
            )
            self.assertEqual(entry["suites"], config["unit_test_suites"])

    def test_push_routes_both_sides_of_renames(self):
        outputs, _ = self.resolve({
            "event_name": "push",
            "files": [
                "test/kernel/nvidia/common.cuh",
                "test/kernel/maca/common.cuh",
            ],
        })
        self.assertEqual(
            outputs["platforms"],
            [platform for platform in ("cuda", "metax") if platform in ALL_ENABLED_PLATFORMS],
        )

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
        self.assertEqual(outputs["platforms"], ALL_ENABLED_PLATFORMS)

    def test_unmapped_ci_source_change_falls_back_to_all_platforms(self):
        outputs, _ = self.resolve(
            {
                "event_name": "pull_request",
                "files": [".github/scripts/ci/new_backend_ci.sh"],
            }
        )
        self.assertEqual(outputs["platforms"], ALL_ENABLED_PLATFORMS)

    def test_unmapped_path_fallback_is_visible_and_never_blocks_planning(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context_path = Path(temp_dir) / "context.json"
            output_path = Path(temp_dir) / "github-output"
            plan_path = Path(temp_dir) / "plan.json"
            summary_path = Path(temp_dir) / "summary.md"
            context_path.write_text(json.dumps({
                "event_name": "pull_request", "files": ["new/unknown/path.cc"]
            }))
            env = dict(os.environ, GITHUB_OUTPUT=str(output_path),
                       FLAGCX_CI_PLAN_PATH=str(plan_path),
                       GITHUB_STEP_SUMMARY=str(summary_path))
            result = subprocess.run(
                ["ruby", str(MATRIX_SCRIPT), str(CONFIG_DIR), str(context_path)],
                capture_output=True, text=True, env=env,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(plan_path.read_text())
            summary = summary_path.read_text()
            self.assertEqual(plan["platforms"], ALL_ENABLED_PLATFORMS)
            self.assertEqual(plan["blocked"], [])
            self.assertIn("Unmapped path", " ".join(plan["routing_reason"]))
            self.assertIn("Blocked work", summary)
            self.assertIn("full platform matrix", summary)

    def test_levels_select_the_expected_workloads(self):
        for level in ("L0", "L0.5", "L1", "L2", "L3"):
            with self.subTest(level=level):
                outputs, _ = self.resolve({
                    "event_name": "workflow_dispatch", "selection": "cuda", "level": level
                })
                self.assertEqual(outputs["level"], level)

    def test_l05_selects_runner_smoke_without_test_workloads(self):
        outputs, _ = self.resolve({
            "event_name": "pull_request",
            "labels": ["platform:metax", "level:L0.5"],
            "files": ["test/unittest/p2p/test_p2p.cc"],
        })
        self.assertEqual(outputs["platforms"], ["metax"])
        self.assertEqual(outputs["runner_matrix"]["include"], [
            {"platform": "metax", "display_name": "MetaX Tests"}
        ])

    def test_level_labels_escalate_without_downgrading(self):
        outputs, _ = self.resolve({
            "event_name": "pull_request", "labels": ["level:L1", "level:L3"],
            "files": ["flagcx/adaptor/device/cuda_adaptor.cc"], "level": "L0",
        })
        self.assertEqual(outputs["level"], "L3")

    def test_l3_pr_level_selects_full_matrix_without_platform_override(self):
        outputs, _ = self.resolve({
            "event_name": "pull_request", "labels": ["level:L3"],
            "files": ["flagcx/adaptor/device/cuda_adaptor.cc"],
        })
        self.assertEqual(outputs["platforms"], ALL_ENABLED_PLATFORMS)

    def test_l3_keeps_explicit_platform_routing(self):
        cases = (
            {"event_name": "pull_request", "labels": ["level:L3", "platform:cuda"]},
            {"event_name": "workflow_dispatch", "selection": "cuda", "level": "L3"},
            {"event_name": "issue_comment", "comment": "/test cuda", "level": "L3",
             "is_pull_request": True, "permission": "write"},
        )
        for context in cases:
            with self.subTest(context=context):
                outputs, _ = self.resolve(context)
                self.assertEqual(outputs["platforms"], ["cuda"])

    def test_l3_ignore_test_projection_preserves_selectors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            context_path = Path(temp_dir) / "context.json"
            output_path = Path(temp_dir) / "github-output"
            plan_path = Path(temp_dir) / "plan.json"
            summary_path = Path(temp_dir) / "summary.md"
            context_path.write_text(json.dumps({
                "event_name": "workflow_dispatch", "selection": "metax", "level": "L2"
            }))
            env = dict(os.environ, GITHUB_OUTPUT=str(output_path),
                       FLAGCX_CI_PLAN_PATH=str(plan_path),
                       GITHUB_STEP_SUMMARY=str(summary_path))
            subprocess.run(
                ["ruby", str(MATRIX_SCRIPT), str(CONFIG_DIR), str(context_path)],
                check=True, capture_output=True, text=True, env=env,
            )
            entry = json.loads(plan_path.read_text())["ignore_test"][0]
            self.assertEqual(entry["modes"], ["homogeneous"])
            self.assertEqual(entry["operations"], ["allreduce", "reducescatter"])
            self.assertIn('"modes":["homogeneous"]', summary_path.read_text())

    def test_mixed_platform_and_unknown_changes_run_all_platforms(self):
        outputs, _ = self.resolve(
            {"event_name": "pull_request", "files": [
                "flagcx/adaptor/device/cuda_adaptor.cc", "new/shared/file.cc"
            ]}
        )
        self.assertEqual(outputs["platforms"], ALL_ENABLED_PLATFORMS)

    def test_unregistered_vendor_paths_use_full_matrix(self):
        outputs, _ = self.resolve({
            "event_name": "pull_request",
            "files": ["flagcx/adaptor/device/mlu_adaptor.cc"],
        })
        self.assertEqual(outputs["platforms"], ALL_ENABLED_PLATFORMS)

    def test_fork_repository_and_immutable_ref_are_forwarded(self):
        outputs, _ = self.resolve({
            "event_name": "pull_request", "repository": "contributor/FlagCX",
            "ref": "a" * 40, "labels": ["platform:enflame"],
        })
        self.assertEqual(outputs["repository"], "contributor/FlagCX")
        self.assertEqual(outputs["ref"], "a" * 40)

    def test_comment_without_platform_respects_labels(self):
        outputs, _ = self.resolve({
            "event_name": "issue_comment", "comment": "/test", "permission": "admin",
            "is_pull_request": True, "labels": ["platform:enflame"],
            "files": ["flagcx/core/group.cc"],
        })
        self.assertEqual(outputs["platforms"], ["enflame"])

    def test_every_platform_config_has_watch_patterns(self):
        for platform in ALL_ENABLED_PLATFORMS:
            with self.subTest(platform=platform):
                config = (CONFIG_DIR / f"{platform}.yml").read_text()
                self.assertIn("paths_watch:", config)
                self.assertIn(f".github/configs/{platform}.yml", config)

    def test_dispatch_selection_and_comment_commands(self):
        for event_name, extra, expected in (
            ("workflow_dispatch", {"selection": "enflame"}, ["enflame"]),
            (
                "issue_comment",
                {
                    "comment": "/test enflame",
                    "permission": "write",
                    "is_pull_request": True,
                },
                ["enflame"],
            ),
            ("workflow_dispatch", {"selection": "kunlunxin"}, ["kunlunxin"]),
            (
                "issue_comment",
                {
                    "comment": "/test kunlunxin",
                    "permission": "write",
                    "is_pull_request": True,
                },
                ["kunlunxin"],
            ),
            (
                "issue_comment",
                {
                    "comment": "/test all",
                    "permission": "maintain",
                    "is_pull_request": True,
                },
                ALL_ENABLED_PLATFORMS,
            ),
            (
                "issue_comment",
                {
                    "comment": "/test",
                    "permission": "write",
                    "is_pull_request": True,
                    "files": ["flagcx/adaptor/device/cuda_adaptor.cc"],
                },
                enabled_platform_or_matrix("cuda"),
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
            attempt_logs = sorted((Path(temp_dir) / "attempt-logs").glob("retry-test.*/attempt-*.log"))
            self.assertEqual(len(attempt_logs), 2)
            result_lines = [
                line.split("\t")
                for log in attempt_logs
                for line in log.read_text().splitlines()
                if line.startswith("FLAGCX_CI_RESULT\t")
            ]
            self.assertEqual(
                result_lines,
                [
                    ["FLAGCX_CI_RESULT", "retry-test", "1", "7", "0"],
                    ["FLAGCX_CI_RESULT", "retry-test", "2", "0", "0"],
                ],
            )
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
            self.assertEqual({re.sub(r"FLAGCX_CI_RESULT\tsame-label\t1\t0\t\d+", "", log.read_text()).strip() for log in logs},
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
    def test_push_resolver_fetches_before_commit_and_gate_receives_suite_filter(self):
        workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/unit-test.yml").read_text())
        prepare_steps = workflow["jobs"]["prepare"]["steps"]
        checkout = next(step for step in prepare_steps if step.get("name") == "Checkout CI configuration")
        self.assertEqual(checkout["with"]["fetch-depth"], 0)
        gate = workflow["jobs"]["ci_complete"]["steps"][-1]
        self.assertEqual(
            gate["env"]["SUITE_FILTER_ACTIVE"],
            "${{ needs.prepare.outputs.suite_filter_active }}",
        )
        self.assertIn("$SUITE_FILTER_ACTIVE", gate["run"])

    def test_single_gate_calls_all_platform_workflows(self):
        workflow = (REPO_ROOT / ".github/workflows/unit-test.yml").read_text()
        self.assertIn("name: FlagCX CI", workflow)
        self.assertIn("  ci_complete:\n    name: ci-complete", workflow)
        self.assertIn(
            "needs: [prepare, unit_tests, runner_smoke, performance_tests, torch_api_tests, format-check]",
            workflow,
        )
        for reusable in (
            "unit_tests_common.yml",
            "test.yml",
            "torch-api-test.yml",
            "format-check.yml",
        ):
            self.assertIn(reusable, workflow)

    def test_l3_daytime_skips_end_to_end_jobs_but_l2_runs_them(self):
        workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/unit-test.yml").read_text())
        for job_name in ("performance_tests", "torch_api_tests"):
            condition = workflow["jobs"][job_name]["if"]
            self.assertIn("needs.prepare.outputs.execution_window == 'night'", condition)
            self.assertIn("needs.prepare.outputs.level == 'L2'", condition)

    def test_platform_specific_jobs_are_conditioned_on_matrix_selection(self):
        for path in (
            REPO_ROOT / ".github/workflows/test.yml",
            REPO_ROOT / ".github/workflows/torch-api-test.yml",
        ):
            source = path.read_text()
            self.assertIn("workflow_call:", source)
            for platform in ("cuda", "hygon", "metax", "ppu", "kunlunxin"):
                self.assertIn(f"contains(fromJSON(inputs.platforms), '{platform}')", source)
            self.assertIn("if: always()", source)
            self.assertIn("check_platform_workflow.sh", source)

    def test_kunlunxin_performance_job_uses_platform_config_and_gate(self):
        workflow = yaml.safe_load(
            (REPO_ROOT / ".github/workflows/test.yml").read_text()
        )
        jobs = workflow["jobs"]
        job = jobs["perf-test-kunlunxin"]
        self.assertEqual(
            job["if"], "contains(fromJSON(inputs.platforms), 'kunlunxin')"
        )
        self.assertIn(
            "outputs.kunlunxin_runs_on",
            job["runs-on"],
        )
        steps = {step.get("name"): step for step in job["steps"]}
        run_step = steps["Run perf-test (kunlunxin) in job container"]
        self.assertEqual(
            job["container"]["image"],
            "${{ needs.load-hardware-container-configs.outputs.kunlunxin_ci_image }}",
        )
        self.assertIn("kunlunxin perf", run_step["run"])
        self.assertIn("perf-test-kunlunxin", jobs["perf_complete"]["needs"])
        self.assertIn(
            "Validate selected KunlunXin performance job",
            [step.get("name") for step in jobs["perf_complete"]["steps"]],
        )

    def test_kunlunxin_torch_api_job_uses_platform_config_and_hardware_runner(self):
        workflow = yaml.safe_load(
            (REPO_ROOT / ".github/workflows/torch-api-test.yml").read_text()
        )
        jobs = workflow["jobs"]
        job = jobs["torch-api-test-kunlunxin"]
        self.assertEqual(
            job["if"], "contains(fromJSON(inputs.platforms), 'kunlunxin')"
        )
        self.assertIn(
            "outputs.kunlunxin_runs_on",
            job["runs-on"],
        )
        steps = {step.get("name"): step for step in job["steps"]}
        run_step = steps["Run torch-api-test (kunlunxin) in job container"]
        self.assertEqual(
            job["container"]["image"],
            "${{ needs.load-hardware-container-configs.outputs.kunlunxin_ci_image }}",
        )
        self.assertIn("kunlunxin torch-api", run_step["run"])
        self.assertIn("torch-api-test-kunlunxin", jobs["torch_api_complete"]["needs"])
        self.assertIn(
            "Validate selected KunlunXin Torch API job",
            [step.get("name") for step in jobs["torch_api_complete"]["steps"]],
        )

        handler = (
            REPO_ROOT / ".github/scripts/ci/run_kunlunxin_workload.sh"
        ).read_text()
        self.assertIn("FLAGCX_ADAPTOR=klx", handler)
        self.assertNotIn("export XTDK_HOME=", handler)
        self.assertIn("torch.cuda.device_count() >= 4", handler)
        self.assertIn("test/script/torch_api_test.sh", handler)

    def test_torch_coverage_upload_is_optional_and_uses_flagcicd_contract(self):
        workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/torch-api-test.yml").read_text())
        caller = yaml.safe_load((REPO_ROOT / ".github/workflows/unit-test.yml").read_text())
        token = caller["jobs"]["torch_api_tests"]["secrets"]["FLAGCICD_API_TOKEN"]
        self.assertIn("needs.prepare.outputs.repository == github.repository", token)
        for platform in ("cuda", "hygon", "metax", "ppu"):
            with self.subTest(platform=platform):
                job = workflow["jobs"][f"torch-api-test-{platform}"]
                steps = {step.get("name"): step for step in job["steps"]}
                report = f"coverage-{platform}-gpu-torch.json"
                upload = steps["Upload coverage report to FlagCICD"]
                self.assertTrue(upload["continue-on-error"])
                self.assertEqual(upload["with"]["file_type"], "coverage")
                self.assertEqual(upload["with"]["job_name"], report.removesuffix(".json"))
                self.assertEqual(upload["with"]["report_path"], report)
                self.assertEqual(upload["with"]["fail_on_error"], "false")
                self.assertIn("FLAGCICD_TOKEN_SET == 'true'", upload["if"])
                self.assertIn(f"hashFiles('{report}')", upload["if"])
                self.assertIn("github.event.issue.number", upload["with"]["pr_id"])
                self.assertNotIn("container", job)
                self.assertIn("Collect Python coverage report", steps)
                self.assertIn(report, steps["Collect Python coverage report"]["run"])
                self.assertEqual(steps["Archive Python coverage report"]["with"]["path"], report)
                workload = steps[f"Run torch-api-test ({platform}) in configured container"]
                self.assertEqual(workload["env"]["FLAGCX_CI_TORCH_COVERAGE"], "1")
                runner = (REPO_ROOT / f".github/scripts/ci/run_{platform}_workload.sh").read_text()
                self.assertIn('bash "$project_root/.github/scripts/ci/run_torch_coverage.sh"', runner)

        container_runner = (REPO_ROOT / ".github/scripts/ci/run_hardware_container.sh").read_text()
        self.assertIn('coverage-${platform}-gpu-torch.json', container_runner)
        runner = (REPO_ROOT / ".github/scripts/ci/run_torch_coverage.sh").read_text()
        self.assertIn("-m coverage run -m unittest discover", runner)

    def test_ordinary_comments_cannot_cancel_pr_ci(self):
        workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/unit-test.yml").read_text())
        self.assertIn("github.event_name == 'issue_comment' && github.run_id", workflow["concurrency"]["group"])
        self.assertIn("startsWith(github.event.comment.body, '/test')", workflow["jobs"]["prepare"]["if"])


class CiCompletionGateTest(unittest.TestCase):
    def check(self, *results, level=None, suite_filter_active=None, runner_smoke=None, execution_window=None):
        if level is not None:
            results += (level,)
        if suite_filter_active is not None:
            if level is None:
                results += ("L2",)
            results += (suite_filter_active,)
        if runner_smoke is not None:
            if level is None:
                results += ("L2",)
            if suite_filter_active is None:
                results += ("false",)
            results += (runner_smoke,)
        if execution_window is not None:
            if level is None:
                results += ("L2",)
            if suite_filter_active is None:
                results += ("false",)
            if runner_smoke is None:
                results += ("skipped",)
            results += (execution_window,)
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

    def test_reduced_levels_allow_intentional_job_skips_but_keep_failures_visible(self):
        for level, unit in (("L0", "skipped"), ("L1", "success")):
            result = self.check(
                "true", "true", "success", unit, "skipped", "skipped", "success", level
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        failed = self.check(
            "true", "true", "success", "skipped", "skipped", "skipped", "success", "L1"
        )
        self.assertNotEqual(failed.returncode, 0)
        cancelled = self.check(
            "true", "true", "success", "success", "cancelled", "skipped", "success", "L1"
        )
        self.assertNotEqual(cancelled.returncode, 0)

    def test_l05_requires_runner_smoke_but_allows_test_jobs_to_skip(self):
        passed = self.check(
            "true", "true", "success", "skipped", "skipped", "skipped", "success",
            level="L0.5", suite_filter_active="false", runner_smoke="success",
        )
        self.assertEqual(passed.returncode, 0, passed.stderr)
        failed = self.check(
            "true", "true", "success", "skipped", "skipped", "skipped", "success",
            level="L0.5", suite_filter_active="false", runner_smoke="failure",
        )
        self.assertNotEqual(failed.returncode, 0)

    def test_focused_suite_run_allows_perf_and_torch_skips_only(self):
        passed = self.check(
            "true", "true", "success", "success", "skipped", "skipped", "success",
            level="L2", suite_filter_active="true",
        )
        self.assertEqual(passed.returncode, 0, passed.stderr)
        failed = self.check(
            "true", "true", "success", "success", "failure", "skipped", "success",
            level="L2", suite_filter_active="true",
        )
        self.assertNotEqual(failed.returncode, 0)

    def test_daytime_l3_may_skip_l2_only_jobs_without_failure(self):
        passed = self.check(
            "true", "true", "success", "success", "skipped", "skipped", "success",
            level="L3", execution_window="day",
        )
        self.assertEqual(passed.returncode, 0, passed.stderr)
        failed = self.check(
            "true", "true", "success", "success", "skipped", "failure", "success",
            level="L3", execution_window="day",
        )
        self.assertNotEqual(failed.returncode, 0)
        night_failed = self.check(
            "true", "true", "success", "success", "skipped", "skipped", "success",
            level="L2", execution_window="night",
        )
        self.assertNotEqual(night_failed.returncode, 0)


class PlatformWorkflowCompletionTest(unittest.TestCase):
    def check(self, platforms, config, cuda_base, cuda, hygon, metax, ppu, ascend="skipped"):
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
                ascend,
            ],
            capture_output=True,
            text=True,
        )

    def test_hygon_experimental_perf_is_required(self):
        script = str(REPO_ROOT / ".github/scripts/ci/check_platform_workflow.sh")
        args = ["bash", script, '["hygon"]', "success", "skipped",
                "skipped", "success", "failure", "skipped", "skipped",
                "skipped"]
        failed = subprocess.run(args, capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("Hygon experimental workload", failed.stderr)
        args[7] = "success"
        passed = subprocess.run(args, capture_output=True, text=True)
        self.assertEqual(passed.returncode, 0, passed.stderr)

    def test_ppu_experimental_perf_is_required(self):
        script = str(REPO_ROOT / ".github/scripts/ci/check_platform_workflow.sh")
        args = ["bash", script, '["ppu"]', "success", "skipped",
                "skipped", "skipped", "skipped", "skipped", "success",
                "failure", "skipped"]
        failed = subprocess.run(args, capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("PPU experimental workload", failed.stderr)
        args[10] = "success"
        passed = subprocess.run(args, capture_output=True, text=True)
        self.assertEqual(passed.returncode, 0, passed.stderr)

    def test_selected_platform_jobs_must_succeed(self):
        self.assertEqual(
            self.check('["cuda"]', "success", "success", "success", "skipped", "skipped", "skipped").returncode,
            0,
        )
        self.assertEqual(
            self.check('["hygon","ppu"]', "success", "skipped", "skipped", "success", "skipped", "success").returncode,
            0,
        )
        self.assertEqual(
            self.check('["ascend"]', "success", "skipped", "skipped", "skipped", "skipped", "skipped", "success").returncode,
            0,
        )
        for result in (
            ("[\"cuda\"]", "skipped", "success", "skipped", "skipped", "skipped", "skipped"),
            ("[\"hygon\"]", "skipped", "skipped", "skipped", "failure", "skipped", "skipped"),
            ("[\"ppu\"]", "success", "skipped", "skipped", "skipped", "skipped", "cancelled"),
            ("[\"ascend\"]", "success", "skipped", "skipped", "skipped", "skipped", "skipped", "failure"),
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
