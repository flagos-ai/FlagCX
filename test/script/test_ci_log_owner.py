"""Check actual root-created CI artifacts can be read and removed by the host."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
RESTORE = ROOT / ".github/scripts/ci/restore_ci_log_owner.sh"


@unittest.skipUnless(sys.platform.startswith("linux") and os.geteuid() == 0,
                     "Linux root container required for cross-UID ownership tests")
class CiLogOwnerTest(unittest.TestCase):
    def test_host_can_read_and_remove_all_result_layouts(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            workspace.chmod(0o755)
            os.chown(workspace, 65534, 65534)
            logs = workspace / "logs"
            logs.mkdir()
            os.chown(logs, 65534, 65534)
            layouts = ("checked-native/round-1.json", "checked-native/round-1.log",
                       "suites/musa-host.test/host-default.xml",
                       "musa-experimental/rank-0.xml", "attempt.test/attempt-1.log")
            for name in layouts:
                path = logs / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.parent.chmod(0o700)
                path.write_text("report\n")
                path.chmod(0o600)
            outside = workspace / "outside"
            outside.mkdir(mode=0o700)
            sentinel = outside / "untouched"
            sentinel.write_text("outside\n")
            sentinel.chmod(0o400)
            (logs / "outside-link").symlink_to(outside, target_is_directory=True)
            (logs / "outside-file").symlink_to(sentinel)
            result = subprocess.run(["sh", str(RESTORE), str(logs), "65534", "65534"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(sentinel.stat().st_uid, 0)
            self.assertEqual(sentinel.stat().st_mode & 0o777, 0o400)
            self.assertEqual(outside.stat().st_uid, 0)
            self.assertEqual(outside.stat().st_mode & 0o777, 0o700)
            reader = subprocess.run([sys.executable, "-c", '''import pathlib, shutil, sys
logs = pathlib.Path(sys.argv[1])
paths = ("checked-native/round-1.json", "checked-native/round-1.log",
         "suites/musa-host.test/host-default.xml", "musa-experimental/rank-0.xml",
         "attempt.test/attempt-1.log")
for name in paths: assert (logs / name).read_text() == "report\\n"
shutil.rmtree(logs)
''', str(logs)], user=65534, group=65534, extra_groups=(),
                capture_output=True, text=True)
            self.assertEqual(reader.returncode, 0, reader.stderr)
            self.assertFalse(logs.exists())
            self.assertEqual(sentinel.read_text(), "outside\n")

    def test_invalid_owner_and_symlink_root_do_not_change_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir(mode=0o700)
            link = root / "link"
            link.symlink_to(target, target_is_directory=True)
            for path, uid in ((target, "invalid"), (target, "-1"), (link, "65534")):
                with self.subTest(path=path, uid=uid):
                    result = subprocess.run(["sh", str(RESTORE), str(path), uid, "65534"],
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertEqual(target.stat().st_uid, 0)
                    self.assertEqual(target.stat().st_mode & 0o777, 0o700)


if __name__ == "__main__":
    unittest.main()
