import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECTOR = REPO_ROOT / ".github/scripts/ci/collect_cpp_coverage.py"
AGGREGATOR = REPO_ROOT / ".github/scripts/ci/aggregate_cpp_coverage.py"
FILTER_SCRIPT = REPO_ROOT / ".github/scripts/ci/filter_supplemental_tests.sh"
HAS_GCOV_TOOLCHAIN = all(shutil.which(command) for command in ("g++", "gcov"))


class SupplementalFilterTest(unittest.TestCase):
    def make_filter(self, current):
        result = subprocess.run(
            ["bash", str(FILTER_SCRIPT), current, "exclude"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def test_default_filter_excludes_supplemental_tests(self):
        self.assertEqual(self.make_filter("*"), "*-CoverageSupplemental.*")

    def test_existing_negative_patterns_are_preserved(self):
        self.assertEqual(
            self.make_filter("Suite.*-Suite.Slow:Suite.Unsupported"),
            "Suite.*-Suite.Slow:Suite.Unsupported:CoverageSupplemental.*",
        )

    def test_only_production_library_receives_coverage_instrumentation(self):
        project_makefile = (REPO_ROOT / "Makefile").read_text()
        test_makefile = (REPO_ROOT / "test/make.inc").read_text()
        runner = (REPO_ROOT / ".github/scripts/ci/run_unit_test.sh").read_text()
        self.assertIn("HOST_CXXFLAGS += --coverage -O0", project_makefile)
        self.assertNotIn("CXXFLAGS += --coverage", test_makefile)
        self.assertIn("FLAGCX_CI_PROJECT_MAKE_ARGS+=(FLAGCX_COVERAGE=1)", runner)
        self.assertIn("run_supplemental_unit_tests", runner)
        self.assertIn("--gtest_filter='CoverageSupplemental.*'", runner)

    def test_weekly_workflow_runs_platform_suite_artifacts(self):
        workflow = (REPO_ROOT / ".github/workflows/cpp-coverage.yml").read_text()
        self.assertIn("'0 18 * * 6'", workflow)
        self.assertIn("workflow_dispatch:", workflow)
        self.assertIn("cpp_coverage: true", workflow)
        reusable = (REPO_ROOT / ".github/workflows/unit_tests_common.yml").read_text()
        self.assertIn("if: ${{ always() && inputs.cpp_coverage }}", reusable)


@unittest.skipUnless(
    HAS_GCOV_TOOLCHAIN and importlib.util.find_spec("gcovr"),
    "g++/gcov and gcovr are required for native coverage integration tests",
)
class CppCoverageReportTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source = self.root / "flagcx/core/example.cc"
        self.object_file = self.root / "build/obj/flagcx/core/example.o"
        self.source.parent.mkdir(parents=True)
        self.object_file.parent.mkdir(parents=True)
        self.source.write_text(
            "int choose(int value) {\n"
            "  if (value > 0) return 1;\n"
            "  return 0;\n"
            "}\n"
        )
        self.uninstrumented = self.root / "flagcx/core/unbuilt.cc"
        self.uninstrumented.write_text("int never_built() { return 0; }\n")
        self.device_source = self.root / "flagcx/adaptor/kernel/device.cu"
        self.device_source.parent.mkdir(parents=True)
        self.device_source.write_text("__global__ void device_only() {}\n")
        self.test_source = self.root / "test/should_not_appear.cc"
        self.test_source.parent.mkdir(parents=True)
        self.test_source.write_text("int test_only() { return 0; }\n")
        self.output = self.root / "coverage"

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_collector(self, *, skip_reason=None):
        command = [
            sys.executable, str(COLLECTOR),
            "--root", str(self.root),
            "--build-dir", str(self.root / "build"),
            "--output-dir", str(self.output),
            "--platform", "fixture",
        ]
        if skip_reason:
            command += ["--skip-reason", skip_reason]
        return subprocess.run(command, capture_output=True, text=True)

    def test_gcovr_report_has_three_metrics_and_source_filters(self):
        subprocess.run(
            ["g++", "--coverage", "-O0", "-c", str(self.source), "-o", str(self.object_file)],
            check=True,
            cwd=self.root,
        )
        harness = self.root / "main.cc"
        harness.write_text("int choose(int); int main() { return choose(1) != 1; }\n")
        binary = self.root / "build/coverage-test"
        subprocess.run(
            ["g++", str(harness), str(self.object_file), "--coverage", "-o", str(binary)],
            check=True,
            cwd=self.root,
        )
        subprocess.run([str(binary)], cwd=self.root, check=True)

        result = self.run_collector()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((self.output / "report.json").read_text())
        self.assertEqual(report["collection_status"], "collected")
        self.assertFalse(report["coverage_complete"])
        self.assertEqual(report["metrics"]["line_percent"], 66.7)
        self.assertEqual(report["metrics"]["function_percent"], 100.0)
        self.assertEqual(report["metrics"]["branch_percent"], 50.0)
        self.assertEqual(
            report["unmeasured_sources"][0]["source"],
            "flagcx/core/unbuilt.cc",
        )
        self.assertEqual(
            report["unmeasured_device_sources"][0]["source"],
            "flagcx/adaptor/kernel/device.cu",
        )
        coverage_json = json.loads((self.output / "coverage.json").read_text())
        self.assertFalse(any("test/" in entry["file"] for entry in coverage_json["files"]))

    def test_skipped_platform_has_explicit_status_not_zero_percent(self):
        result = self.run_collector(skip_reason="unsupported by platform toolchain")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((self.output / "report.json").read_text())
        self.assertEqual(report["collection_status"], "skipped")
        self.assertEqual(report["skip_reason"], "unsupported by platform toolchain")
        self.assertIsNone(report["metrics"]["line_percent"])
        self.assertIn("Lines: N/A", (self.output / "summary.md").read_text())

    def test_missing_instrumented_objects_fails_clearly(self):
        result = self.run_collector()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no instrumented C++ objects", result.stderr)
        self.assertTrue((self.output / "report.json").exists())


class CppCoverageAggregationTest(unittest.TestCase):
    def test_platform_metrics_remain_separate_until_policy_is_approved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_dir = root / "platform" / "core"
            report_dir.mkdir(parents=True)
            (report_dir / "report.json").write_text(json.dumps({
                "schema_version": 1,
                "platform": "cuda/core",
                "metrics": {
                    "line_percent": 80.0,
                    "function_percent": 75.0,
                    "branch_percent": 60.0,
                },
                "unmeasured_sources": [{"source": "flagcx/kernel.cu"}],
            }))
            output = root / "summary.md"
            result = subprocess.run(
                [sys.executable, str(AGGREGATOR), "--input-dir", str(root), "--output", str(output)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            matrix = json.loads((root / "matrix.json").read_text())
            self.assertIsNone(matrix["aggregate_metric"])
            self.assertIn("80.00%", output.read_text())
            self.assertIn("threshold is enforced", output.read_text())


if __name__ == "__main__":
    unittest.main()
