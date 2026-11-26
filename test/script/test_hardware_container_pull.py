"""Registry retries must not hide image or hardware workload failures."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".github/scripts/ci/run_hardware_container.sh"
REGISTRY_TIMEOUT = (
    'Error response from daemon: Get "https://harbor.baai.ac.cn/v2/": '
    "net/http: request canceled while waiting for connection "
    "(Client.Timeout exceeded while awaiting headers)"
)
FAKE_COMMAND = """import json, os, pathlib, signal, sys
root = pathlib.Path(os.environ['FAKE_ROOT'])
command = pathlib.Path(sys.argv[0]).name
with (root / 'calls.jsonl').open('a') as log:
    log.write(json.dumps([command, *sys.argv[1:]]) + '\\n')
if command == 'sleep':
    sys.exit(0)
if command == 'id':
    print(os.environ['LOG_TEST_UID' if sys.argv[1] == '-u' else 'LOG_TEST_GID'])
    sys.exit(0)
if sys.argv[1] == 'pull':
    counter = root / 'pull-count'
    attempt = int(counter.read_text()) if counter.exists() else 0
    counter.write_text(str(attempt + 1))
    errors = json.loads(os.environ['PULL_ERRORS'])
    if attempt < len(errors):
        print(errors[attempt], file=sys.stderr)
        sys.exit(17)
elif sys.argv[1:3] == ['container', 'run']:
    (root / 'workload-started').touch()
    if os.environ.get('WORKLOAD_SIGNAL'):
        # The direct parent is GNU timeout, not the container launcher.
        os.kill(int(os.environ['TEST_CONTAINER_LAUNCHER_PID']),
                getattr(signal, 'SIG' + os.environ['WORKLOAD_SIGNAL']))
    sys.exit(int(os.environ['WORKLOAD_STATUS']))
elif sys.argv[1] == 'run':
    sys.exit(int(os.environ['RESTORE_STATUS']))
elif sys.argv[1:3] == ['container', 'rm'] and (root / 'workload-started').exists():
    sys.exit(int(os.environ['CLEANUP_STATUS']))
elif sys.argv[1:3] == ['container', 'ls'] and os.environ['CONTAINER_PRESENT'] == '1':
    print('remaining-container')
"""


class HardwareContainerPullTest(unittest.TestCase):
    def run_container(self, errors=(), workload_status=0, *, restore_status=0,
                      cleanup_status=0, container_present=True, owner=None,
                      workload_signal="", log_directory=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            for name in ("docker", "sleep", "id"):
                command = bin_dir / name
                command.write_text(f"#!{sys.executable}\n" + FAKE_COMMAND)
                command.chmod(0o755)
            environment = os.environ.copy()
            environment.update(
                PATH=f"{bin_dir}:{environment['PATH']}",
                FAKE_ROOT=str(root),
                PULL_ERRORS=json.dumps(errors),
                WORKLOAD_STATUS=str(workload_status),
                RESTORE_STATUS=str(restore_status),
                CLEANUP_STATUS=str(cleanup_status),
                CONTAINER_PRESENT="1" if container_present else "0",
                LOG_TEST_UID=str(owner[0] if owner else os.getuid()),
                LOG_TEST_GID=str(owner[1] if owner else os.getgid()),
                WORKLOAD_SIGNAL=workload_signal,
                FLAGCX_CI_IMAGE="registry.example/flagcx:qualified",
                FLAGCX_CI_DOCKER_ARGS="--network=host",
                FLAGCX_CI_SET_ENV=".github/scripts/set_env/cuda.sh",
                FLAGCX_CI_LOG_DIR=str(log_directory or root / "logs"),
                GITHUB_STEP_SUMMARY=str(root / "summary.md"),
            )
            result = subprocess.run(
                ["bash", "-c", 'export TEST_CONTAINER_LAUNCHER_PID=$$; exec bash "$@"',
                 "container-launcher", str(SCRIPT), "cuda", "unittest", "device_api"],
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
            )
            calls = [
                json.loads(line)
                for line in (root / "calls.jsonl").read_text().splitlines()
            ]
            logs = {
                path.name: path.read_text()
                for path in Path(environment["FLAGCX_CI_LOG_DIR"]).glob("image-pull-attempt-*.log")
            }
            summary = (root / "summary.md").read_text() if (root / "summary.md").exists() else ""
            return result, calls, logs, summary

    def test_successful_pull_starts_the_workload_once(self):
        result, calls, logs, summary = self.run_container()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sum(call[:2] == ["docker", "pull"] for call in calls), 1)
        self.assertEqual(sum(call[:3] == ["docker", "container", "run"] for call in calls), 1)
        self.assertFalse(any(call[0] == "sleep" for call in calls))
        self.assertEqual(len(logs), 1)
        self.assertIn("succeeded on attempt 1/3", summary)

    def test_observed_harbor_timeout_retries_and_preserves_the_failure_log(self):
        result, calls, logs, summary = self.run_container([REGISTRY_TIMEOUT])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(REGISTRY_TIMEOUT, logs["image-pull-attempt-1.log"])
        self.assertEqual(len(logs), 2)
        self.assertIn(["sleep", "5"], calls)
        self.assertEqual(sum(call[:3] == ["docker", "container", "run"] for call in calls), 1)
        self.assertIn("::warning::Image pull attempt 1/3 failed", result.stderr)
        self.assertIn("succeeded on attempt 2/3", summary)

    def test_persistent_registry_outage_fails_without_starting_a_container(self):
        result, calls, logs, summary = self.run_container([REGISTRY_TIMEOUT] * 3)
        self.assertEqual(result.returncode, 17)
        self.assertEqual(len(logs), 3)
        self.assertEqual(
            [call for call in calls if call[0] == "sleep"],
            [["sleep", "5"], ["sleep", "10"]],
        )
        self.assertFalse(any(call[:3] == ["docker", "container", "run"] for call in calls))
        self.assertFalse(any(call[:2] == ["docker", "run"] for call in calls))
        self.assertIn("failed on attempt 3/3 (exit=17)", summary)

    def test_permanent_image_errors_do_not_retry_or_use_a_cached_image(self):
        for error in (
            "unauthorized: authentication required",
            "unauthorized: authentication required after timeout",
            "manifest unknown",
            "invalid reference format",
            "unexpected unclassified failure",
        ):
            with self.subTest(error=error):
                result, calls, logs, summary = self.run_container([error])
                self.assertEqual(result.returncode, 17)
                self.assertEqual(len(logs), 1)
                self.assertFalse(any(call[0] == "sleep" for call in calls))
                self.assertFalse(any(call[:3] == ["docker", "container", "run"] for call in calls))
                self.assertFalse(any(call[:2] == ["docker", "run"] for call in calls))
                self.assertIn("failed on attempt 1/3", summary)

    def test_workload_failure_is_propagated_without_repeating_it(self):
        result, calls, _, _ = self.run_container(workload_status=23)
        self.assertEqual(result.returncode, 23)
        self.assertEqual(sum(call[:2] == ["docker", "pull"] for call in calls), 1)
        self.assertEqual(sum(call[:3] == ["docker", "container", "run"] for call in calls), 1)
        self.assertFalse(any(call[0] == "sleep" for call in calls))

    def test_log_handoff_uses_host_ids_and_only_log_and_helper_mounts(self):
        for owner in ((1234, 5678), (0, 0)):
            with self.subTest(owner=owner):
                result, calls, _, _ = self.run_container(owner=owner)
                self.assertEqual(result.returncode, 0, result.stderr)
                helper = next(call for call in calls if call[:2] == ["docker", "run"])
                for option in ("--rm", "--pull=never", "--runtime=runc", "--network=none", "--user=root"):
                    self.assertIn(option, helper)
                self.assertNotIn("--privileged", helper)
                self.assertEqual(helper[-3:], ["/ci-logs", *map(str, owner)])
                mounts = [helper[i + 1] for i, value in enumerate(helper) if value == "--volume"]
                self.assertEqual(len(mounts), 2)
                self.assertTrue(any(m.endswith("/logs:/ci-logs") for m in mounts))
                self.assertTrue(any(m.endswith("/restore_ci_log_owner.sh:/restore_ci_log_owner.sh:ro") for m in mounts))
                main_index = next(i for i, c in enumerate(calls) if c[:3] == ["docker", "container", "run"])
                self.assertIn("--network=host", calls[main_index])
                helper_index = calls.index(helper)
                self.assertTrue(any(c[:3] == ["docker", "container", "rm"]
                                    for c in calls[main_index + 1:helper_index]))

    def test_workload_status_survives_successful_or_failed_handoff(self):
        for workload_status in (0, 1, 19, 124, 137):
            for restore_status in (0, 31):
                with self.subTest(workload=workload_status, restore=restore_status):
                    result, calls, _, _ = self.run_container(
                        workload_status=workload_status, restore_status=restore_status)
                    self.assertEqual(result.returncode, workload_status or (1 if restore_status else 0), result.stderr)
                    self.assertEqual(sum(c[:2] == ["docker", "run"] for c in calls), 1)
                    if restore_status:
                        self.assertIn("ownership restoration failed (exit=31)", result.stderr)

    def test_live_container_prevents_handoff_and_preserves_prior_failure(self):
        for workload_status in (0, 19):
            with self.subTest(workload_status=workload_status):
                result, calls, _, _ = self.run_container(
                    workload_status=workload_status, cleanup_status=41)
                self.assertEqual(result.returncode, workload_status or 1, result.stderr)
                self.assertFalse(any(c[:2] == ["docker", "run"] for c in calls))
                self.assertIn("container may still be running", result.stderr)

    def test_auto_removed_container_still_hands_off_logs(self):
        result, calls, _, _ = self.run_container(cleanup_status=1, container_present=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sum(c[:2] == ["docker", "run"] for c in calls), 1)

    def test_interrupts_keep_signal_status_and_hand_off_logs(self):
        for name, status in (("INT", 130), ("TERM", 143)):
            with self.subTest(signal=name):
                result, calls, _, _ = self.run_container(workload_signal=name)
                self.assertEqual(result.returncode, status, result.stderr)
                self.assertEqual(sum(c[:3] == ["docker", "container", "run"]
                                     for c in calls), 1)
                self.assertEqual(sum(c[:2] == ["docker", "run"] for c in calls), 1)
                self.assertNotIn("watchdog expired", result.stderr)

    def test_workspace_and_ancestor_log_roots_are_rejected(self):
        for path in (REPO_ROOT, REPO_ROOT.parent, Path("/")):
            with self.subTest(path=path):
                result, calls, _, _ = self.run_container(log_directory=path)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("dedicated directory", result.stderr)
                self.assertFalse(any(call[0] == "docker" for call in calls))


if __name__ == "__main__":
    unittest.main()
