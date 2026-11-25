"""Exercise submodule lock recovery against local Git repositories."""

import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".github/scripts/ci/checkout_submodules.sh"


class SubmoduleCheckoutTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.workspace = self.root / "workspace"
        self.source = self.root / "json-origin"
        self.env = dict(
            os.environ,
            GIT_CONFIG_GLOBAL=str(self.root / "gitconfig"),
            GIT_CONFIG_NOSYSTEM="1",
            GIT_ALLOW_PROTOCOL="file",
            GIT_AUTHOR_NAME="CI test",
            GIT_AUTHOR_EMAIL="ci@example.invalid",
            GIT_COMMITTER_NAME="CI test",
            GIT_COMMITTER_EMAIL="ci@example.invalid",
            GITHUB_WORKSPACE=str(self.workspace),
            PATH=f"{self.bin_dir}:{os.environ['PATH']}",
        )
        # Do not make negative recovery cases wait between attempts.
        self.fake_command("sleep", "exit 0")
        self.fake_command("pgrep", "exit 1")
        self.git("init", str(self.source))
        (self.source / "json.h").write_text("first version\n")
        self.git("-C", str(self.source), "add", "json.h")
        self.git("-C", str(self.source), "commit", "-m", "initial")
        self.git("init", str(self.workspace))
        self.git(
            "-C", str(self.workspace), "submodule", "add", "--depth=1",
            self.source.as_uri(), "third-party/json",
        )
        self.git("-C", str(self.workspace), "commit", "-am", "add submodule")
        # The superproject now requests a commit absent from the shallow clone.
        (self.source / "json.h").write_text("second version\n")
        self.git("-C", str(self.source), "commit", "-am", "new version")
        self.target = self.git("-C", str(self.source), "rev-parse", "HEAD").strip()
        self.git(
            "-C", str(self.workspace), "update-index", "--cacheinfo",
            f"160000,{self.target},third-party/json",
        )
        self.lock = self.workspace / ".git/modules/third-party/json/shallow.lock"

    def git(self, *args):
        return subprocess.run(
            ["git", *args], env=self.env, check=True,
            capture_output=True, text=True, timeout=15,
        ).stdout

    def fake_command(self, name, body):
        command = self.bin_dir / name
        command.write_text(f"#!/usr/bin/env bash\n{body}\n")
        command.chmod(0o755)

    def create_lock(self, age=600):
        self.lock.write_text("interrupted shallow update\n")
        timestamp = time.time() - age
        os.utime(self.lock, (timestamp, timestamp))

    def checkout(self):
        return subprocess.run(
            ["bash", str(SCRIPT)], env=self.env,
            capture_output=True, text=True, timeout=15,
        )

    def test_old_orphan_lock_is_recovered_and_pinned_commit_is_fetched(self):
        self.create_lock()
        result = self.checkout()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Recovering stale submodule shallow lock", result.stdout)
        self.assertFalse(self.lock.exists())
        self.assertEqual(
            self.git("-C", str(self.workspace / "third-party/json"), "rev-parse", "HEAD").strip(),
            self.target,
        )

    def test_recent_lock_is_preserved_and_checkout_fails(self):
        self.create_lock(age=0)
        result = self.checkout()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("lock is recent", result.stderr)
        self.assertTrue(self.lock.exists())

    def test_lock_is_preserved_while_a_git_process_is_running(self):
        self.create_lock()
        (self.bin_dir / "pgrep").unlink()
        process = subprocess.Popen(
            ["git", "hash-object", "--stdin"], env=self.env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        try:
            self.assertIsNone(process.poll())
            result = self.checkout()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Git processes are still running", result.stderr)
            self.assertTrue(self.lock.exists())
        finally:
            process.communicate(input=b"", timeout=5)

    def test_failed_process_check_preserves_lock(self):
        self.create_lock()
        self.fake_command("pgrep", "exit 2")
        result = self.checkout()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("process check failed", result.stderr)
        self.assertTrue(self.lock.exists())

    def test_checkout_without_stale_locks_does_not_require_pgrep(self):
        self.fake_command("pgrep", "exit 2")
        result = self.checkout()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.git("-C", str(self.workspace / "third-party/json"), "rev-parse", "HEAD").strip(),
            self.target,
        )


if __name__ == "__main__":
    unittest.main()
