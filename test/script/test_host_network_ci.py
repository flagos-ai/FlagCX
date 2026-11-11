"""Check job-container namespace entry without needing RDMA or root in CI."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github/scripts/ci/run_in_host_network.sh"


class HostNetworkCiTest(unittest.TestCase):
    def test_all_kunlunxin_workloads_enter_host_network(self):
        for workflow, job, step in (
            ("unit_tests_common.yml", "kunlunxin_unit_test",
             "Run unit-test suite in job container"),
            ("test.yml", "perf-test-kunlunxin",
             "Run perf-test (kunlunxin) in job container"),
            ("torch-api-test.yml", "torch-api-test-kunlunxin",
             "Run torch-api-test (kunlunxin) in job container"),
        ):
            with self.subTest(workflow=workflow):
                config = yaml.safe_load((ROOT / ".github/workflows" / workflow).read_text())
                job_config = config["jobs"][job]
                self.assertIn("container", job_config)
                command = next(item["run"] for item in job_config["steps"]
                               if item.get("name") == step)
                self.assertIn("run_in_host_network.sh", command)
                self.assertNotIn("docker run", command)

    def test_missing_mount_fails_before_workload(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "executed"
            result = subprocess.run(
                ["bash", str(SCRIPT), "touch", str(marker)],
                env={**os.environ, "FLAGCX_CI_HOST_NETNS": directory + "/missing"},
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("mount /proc:/run/flagcx-host-proc:ro", result.stderr)
            self.assertFalse(marker.exists())

    def test_namespace_mismatch_fails_before_workload(self):
        result = subprocess.run(
            ["bash", str(SCRIPT), "--in-host-network", "wrong-namespace", "true"],
            text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("namespace mismatch", result.stderr)

    def test_command_arguments_environment_and_status_are_preserved(self):
        for preflight_status in (0, 9):
            with self.subTest(preflight_status=preflight_status), \
                    tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                namespace = root / "host-netns"
                namespace.touch()
                # Model setns and the hardware inventory. Real network entry and
                # GID visibility are covered separately on the P800 machine.
                for name, body in {
                    "readlink": 'printf "net:[42]\\n"\n',
                    "unshare": 'shift 4\nexec "$@"\n',
                    "nsenter": 'test "$2" = -- || exit 90\nshift 2\nexec "$@"\n',
                    "python3": f"cat >/dev/null\nexit {preflight_status}\n",
                }.items():
                    executable = root / name
                    executable.write_text("#!/bin/bash\n" + body)
                    executable.chmod(0o755)
                marker = root / "result"
                result = subprocess.run(
                    ["bash", str(SCRIPT), "bash", "-c",
                     'printf "%s|%s" "$1" "$PROBE_ENV" > "$2"; exit 7',
                     "bash", "argument with spaces", str(marker)],
                    env={**os.environ, "PATH": f"{root}:{os.environ['PATH']}",
                         "FLAGCX_CI_HOST_NETNS": str(namespace), "PROBE_ENV": "kept"},
                    text=True, capture_output=True,
                )
                self.assertEqual(result.returncode, preflight_status or 7,
                                 result.stdout + result.stderr)
                if preflight_status:
                    self.assertFalse(marker.exists())
                else:
                    self.assertEqual(marker.read_text(), "argument with spaces|kept")


if __name__ == "__main__":
    unittest.main()
