import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(importlib.util.find_spec("coverage"), "coverage.py is required")
class TorchCoverageReportTest(unittest.TestCase):
    def run_fixture(self, root, *, fail=False, report_path=None, platform="cuda"):
        for relative in (
            ".github/configs/coverage.ini",
            ".github/scripts/ci/run_torch_coverage.sh",
            ".github/scripts/ci/run_with_timeout.sh",
                ".github/scripts/ci/timeouts.sh",
        ):
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO_ROOT / relative, destination)
        source = root / "src/flagcx"
        source.mkdir(parents=True)
        (source / "__init__.py").touch()
        (source / "example.py").write_text("def answer():\n    return 42\n")
        (source / "unimported.py").write_text("def never_called():\n    return 7\n")
        tests = root / "plugin/torch/tests"
        tests.mkdir(parents=True)
        expected = 0 if fail else 42
        (tests / "test_example.py").write_text(
            "import unittest\n"
            "from flagcx.example import answer\n"
            "class ExampleTest(unittest.TestCase):\n"
            "    def test_answer(self):\n"
            f"        self.assertEqual(answer(), {expected})\n"
        )
        env = os.environ.copy()
        env.update({
            "GITHUB_WORKSPACE": str(root),
            "PYTHON_BIN": sys.executable,
            "FLAGCX_CI_PLATFORM": platform,
            "FLAGCX_CI_LOG_DIR": str(root / "logs"),
            "FLAGCX_CI_TEST_TIMEOUT": "5s",
            "FLAGCX_CI_TEST_KILL_AFTER": "1s",
            "GITHUB_STEP_SUMMARY": str(root / "summary.md"),
        })
        if report_path is not None:
            env["FLAGCX_CI_COVERAGE_REPORT"] = str(report_path)
        return subprocess.run(
            ["bash", str(root / ".github/scripts/ci/run_torch_coverage.sh")],
            capture_output=True, text=True, env=env, timeout=15,
        )

    def assert_real_report(self, root):
        report = json.loads((root / "coverage-cuda-gpu-torch.json").read_text())
        self.assertTrue(report["meta"]["branch_coverage"])
        self.assertGreater(report["totals"]["covered_lines"], 0)
        self.assertIn("src/flagcx/example.py", report["files"])
        self.assertIn("src/flagcx/unimported.py", report["files"])
        self.assertEqual(report["files"]["src/flagcx/unimported.py"]["summary"]["percent_covered"], 0.0)
        self.assertFalse(any("tests/" in path for path in report["files"]))

    def test_generates_coverage_json_for_python_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_fixture(root)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assert_real_report(root)

    def test_failed_tests_still_generate_report_and_preserve_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_fixture(root, fail=True)
            self.assertNotEqual(result.returncode, 0)
            self.assert_real_report(root)
            self.assertEqual(len(list((root / "logs").glob("*/attempt-*.log"))), 1)

    def test_iluvatar_report_survives_failed_tests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_fixture(root, fail=True, platform="iluvatar")
            self.assertNotEqual(result.returncode, 0)
            report = json.loads((root / "coverage-iluvatar-gpu-torch.json").read_text())
            self.assertGreater(report["totals"]["covered_lines"], 0)
            self.assertEqual(len(list((root / "logs").glob("*/attempt-*.log"))), 1)

    def test_report_can_be_persisted_outside_container_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "workspace"
            persisted = Path(temporary) / "logs"
            persisted.mkdir()
            report_path = persisted / "coverage-cuda-gpu-torch.json"
            result = self.run_fixture(root, report_path=report_path)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(report_path.read_text())
            self.assertIn("src/flagcx/example.py", report["files"])
            self.assertFalse((root / "coverage-cuda-gpu-torch.json").exists())


if __name__ == "__main__":
    unittest.main()
