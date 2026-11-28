import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
HELPER = REPO_ROOT / ".github/scripts/ci/checkout_submodules.sh"
WORKFLOWS = ("unit_tests_common.yml", "test.yml", "torch-api-test.yml",
             "build-deb.yml", "build-rpm.yml")


def checkout_jobs():
    for filename in WORKFLOWS:
        workflow = yaml.safe_load((REPO_ROOT / ".github/workflows" / filename).read_text())
        for name, job in workflow["jobs"].items():
            if "FLAGCX_CI_SOURCE_ROOT" in job.get("env", {}):
                yield filename, name, job


class CheckoutWorkflowTest(unittest.TestCase):
    def test_every_hardware_and_packaging_checkout_is_isolated_and_logged(self):
        jobs = list(checkout_jobs())
        self.assertEqual(len(jobs), 25)
        for filename, name, job in jobs:
            with self.subTest(workflow=filename, job=name):
                env, steps = job["env"], job["steps"]
                self.assertIn("github.run_id", env["FLAGCX_CI_CHECKOUT_PATH"])
                self.assertIn("github.run_attempt", env["FLAGCX_CI_CHECKOUT_PATH"])
                self.assertIn(name, env["FLAGCX_CI_CHECKOUT_PATH"])
                self.assertNotEqual(env["FLAGCX_CI_SOURCE_ROOT"], env["FLAGCX_CI_LOG_DIR"])
                checkout = next(i for i, s in enumerate(steps)
                                if s.get("uses", "").startswith("actions/checkout@"))
                submodules = next(i for i, s in enumerate(steps)
                                  if s.get("name") == "Checkout submodules with retries")
                self.assertEqual(steps[0]["name"], "Prepare isolated checkout and diagnostics")
                self.assertLess(checkout, submodules)
                self.assertFalse(steps[checkout]["with"]["submodules"])
                self.assertEqual(steps[checkout]["with"]["path"], "${{ env.FLAGCX_CI_CHECKOUT_PATH }}")
                self.assertIn("set -euo pipefail", steps[submodules]["run"])
                self.assertNotIn("if", steps[submodules])
                uploads = [s for s in steps if s.get("uses", "").startswith("actions/upload-artifact@")
                           and s.get("with", {}).get("path") == "${{ env.FLAGCX_CI_LOG_DIR }}/"]
                self.assertEqual(len(uploads), 1)
                self.assertEqual(uploads[0]["if"], "always()")
                self.assertEqual(uploads[0]["with"]["path"], "${{ env.FLAGCX_CI_LOG_DIR }}/")
                for step in steps:
                    if step.get("name") == "Upload C++ coverage report":
                        self.assertIn("steps.workload.outcome != 'skipped'", step["if"])
                        self.assertEqual(step["with"]["if-no-files-found"], "error")

    def test_checkout_failure_has_diagnostics_before_source_exists(self):
        for filename, name, job in checkout_jobs():
            with self.subTest(workflow=filename, job=name), tempfile.TemporaryDirectory() as tmp:
                source, logs = Path(tmp) / "checkout", Path(tmp) / "logs"
                env = dict(os.environ, FLAGCX_CI_SOURCE_ROOT=str(source),
                           FLAGCX_CI_LOG_DIR=str(logs), GITHUB_JOB=name,
                           GITHUB_RUN_ID="123", GITHUB_RUN_ATTEMPT="2", RUNNER_NAME="fixture")
                subprocess.run(["bash", "-c", job["steps"][0]["run"]],
                               env=env, check=True, capture_output=True)
                self.assertTrue(source.is_dir())
                self.assertIn("run=123 attempt=2", (logs / "checkout.log").read_text())
                outcome = next(s for s in job["steps"] if s.get("name") == "Record checkout outcome")
                subprocess.run(["bash", "-c", outcome["run"]],
                               env=dict(env, CHECKOUT_OUTCOME="failure", SUBMODULE_OUTCOME="skipped"),
                               check=True, capture_output=True)
                self.assertIn("checkout=failure submodules=skipped", (logs / "checkout.log").read_text())


@unittest.skipUnless(shutil.which("timeout"), "GNU timeout required")
class SubmoduleCheckoutTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.env = dict(os.environ, GIT_ALLOW_PROTOCOL="file",
                        GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
        child = self.root / "child"
        child.mkdir()
        self.git(child, "init", "-q")
        (child / "header.hpp").write_text("// pinned dependency\n")
        self.git(child, "add", ".")
        self.git(child, "commit", "-qm", "dependency")
        parent = self.root / "parent"
        parent.mkdir()
        self.git(parent, "init", "-q")
        self.git(parent, "submodule", "add", "-q", str(child), "third-party/json")
        helper = parent / ".github/scripts/ci/checkout_submodules.sh"
        helper.parent.mkdir(parents=True)
        shutil.copy2(HELPER, helper)
        self.git(parent, "add", ".")
        self.git(parent, "commit", "-qm", "parent")
        self.checkout = self.root / "checkout"
        self.git(self.root, "clone", "-q", str(parent), str(self.checkout))

    def git(self, cwd, *args):
        return subprocess.run(["git", "-c", "user.name=CI", "-c", "user.email=ci@example.test", *args],
                              cwd=cwd, env=self.env, check=True, text=True, capture_output=True)

    def update(self, checkout=None, **overrides):
        checkout = checkout or self.checkout
        env = dict(self.env, GITHUB_WORKSPACE=str(checkout), **overrides)
        return subprocess.run(["bash", str(checkout / ".github/scripts/ci/checkout_submodules.sh")],
                              env=env, text=True, capture_output=True, timeout=20)

    def test_fresh_checkout_populates_pinned_dependency(self):
        result = self.update()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.checkout / "third-party/json/header.hpp").is_file())

    def test_existing_lock_is_preserved_and_fails_before_update(self):
        self.assertEqual(self.update().returncode, 0)
        lock = self.checkout / ".git/modules/third-party/json/shallow.lock"
        lock.write_text("owned by another git process\n")
        result = self.update()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(str(lock.relative_to(self.checkout)), result.stderr)
        self.assertNotIn("attempt 1/3", result.stdout)
        self.assertEqual(lock.read_text(), "owned by another git process\n")
        fresh = self.checkout / ".ci-checkouts/fresh-attempt"
        self.git(self.root, "clone", "-q", str(self.root / "parent"), str(fresh))
        result = self.update(fresh)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(lock.exists())

    def test_interrupted_submodule_head_is_recovered(self):
        self.assertEqual(self.update().returncode, 0)
        (self.checkout / ".git/modules/third-party/json/HEAD").write_text("ref: refs/heads/interrupted\n")
        result = self.update()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Recovering interrupted checkout", result.stdout)

    def test_fetch_failure_retries_are_bounded_and_report_original_error(self):
        self.git(self.checkout, "config", "submodule.third-party/json.url", str(self.root / "missing"))
        # sync restores .gitmodules, so change the tracked URL as a broken remote fixture.
        modules = self.checkout / ".gitmodules"
        modules.write_text(modules.read_text().replace(str(self.root / "child"), str(self.root / "missing")))
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        sleeper = bin_dir / "sleep"
        sleeper.write_text("#!/bin/sh\nexit 0\n")
        sleeper.chmod(0o755)
        result = self.update(PATH=str(bin_dir) + os.pathsep + os.environ["PATH"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("attempt 3/3", result.stdout)
        self.assertIn("does not exist", result.stderr)
        self.assertIn("failed after 3 attempts", result.stderr)


if __name__ == "__main__":
    unittest.main()
