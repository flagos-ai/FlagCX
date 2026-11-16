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
FAKE_COMMAND = """import json, os, pathlib, sys
root = pathlib.Path(os.environ['FAKE_ROOT'])
command = pathlib.Path(sys.argv[0]).name
with (root / 'calls.jsonl').open('a') as log:
    log.write(json.dumps([command, *sys.argv[1:]]) + '\\n')
if command == 'sleep':
    sys.exit(0)
if sys.argv[1] == 'pull':
    counter = root / 'pull-count'
    attempt = int(counter.read_text()) if counter.exists() else 0
    counter.write_text(str(attempt + 1))
    errors = json.loads(os.environ['PULL_ERRORS'])
    if attempt < len(errors):
        print(errors[attempt], file=sys.stderr)
        sys.exit(17)
elif sys.argv[1] == 'run':
    sys.exit(int(os.environ['WORKLOAD_STATUS']))
"""


class HardwareContainerPullTest(unittest.TestCase):
    def run_container(self, errors=(), workload_status=0):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            for name in ("docker", "sleep"):
                command = bin_dir / name
                command.write_text(f"#!{sys.executable}\n" + FAKE_COMMAND)
                command.chmod(0o755)
            environment = os.environ.copy()
            environment.update(
                PATH=f"{bin_dir}:{environment['PATH']}",
                FAKE_ROOT=str(root),
                PULL_ERRORS=json.dumps(errors),
                WORKLOAD_STATUS=str(workload_status),
                FLAGCX_CI_IMAGE="registry.example/flagcx:qualified",
                FLAGCX_CI_DOCKER_ARGS="--network=host",
                FLAGCX_CI_SET_ENV=".github/scripts/set_env/cuda.sh",
                FLAGCX_CI_LOG_DIR=str(root / "logs"),
                GITHUB_STEP_SUMMARY=str(root / "summary.md"),
            )
            result = subprocess.run(
                ["bash", str(SCRIPT), "cuda", "unittest", "device_api"],
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
                for path in (root / "logs").glob("image-pull-attempt-*.log")
            }
            summary = (root / "summary.md").read_text()
            return result, calls, logs, summary

    def test_successful_pull_starts_the_workload_once(self):
        result, calls, logs, summary = self.run_container()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sum(call[:2] == ["docker", "pull"] for call in calls), 1)
        self.assertEqual(sum(call[:2] == ["docker", "run"] for call in calls), 1)
        self.assertFalse(any(call[0] == "sleep" for call in calls))
        self.assertEqual(len(logs), 1)
        self.assertIn("succeeded on attempt 1/3", summary)

    def test_observed_harbor_timeout_retries_and_preserves_the_failure_log(self):
        result, calls, logs, summary = self.run_container([REGISTRY_TIMEOUT])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(REGISTRY_TIMEOUT, logs["image-pull-attempt-1.log"])
        self.assertEqual(len(logs), 2)
        self.assertIn(["sleep", "5"], calls)
        self.assertEqual(sum(call[:2] == ["docker", "run"] for call in calls), 1)
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
                self.assertFalse(any(call[:2] == ["docker", "run"] for call in calls))
                self.assertIn("failed on attempt 1/3", summary)

    def test_workload_failure_is_propagated_without_repeating_it(self):
        result, calls, _, _ = self.run_container(workload_status=23)
        self.assertEqual(result.returncode, 23)
        self.assertEqual(sum(call[:2] == ["docker", "pull"] for call in calls), 1)
        self.assertEqual(sum(call[:2] == ["docker", "run"] for call in calls), 1)
        self.assertFalse(any(call[0] == "sleep" for call in calls))


if __name__ == "__main__":
    unittest.main()
