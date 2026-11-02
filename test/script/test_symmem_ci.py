"""Check fixed SymMem CI selection and MPI exit status handling."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class SymmemCiTest(unittest.TestCase):
    def test_selected_cases_by_adaptor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "test/script/symmem_test.sh"
            script.parent.mkdir(parents=True)
            shutil.copyfile(REPO_ROOT / "test/script/symmem_test.sh", script)
            ci_dir = root / ".github/scripts/ci"
            ci_dir.mkdir(parents=True)
            for name in ("run_with_timeout.sh", "run_mpi_with_timeout.sh"):
                runner = ci_dir / name
                runner.write_text(
                    '#!/bin/bash\nprintf "%s\\n" "$*" >> "$CI_CAPTURE"\n'
                    'if [[ -n "${CI_FAIL_CASE:-}" && "$*" == *"$CI_FAIL_CASE"* ]]; '
                    'then exit 7; fi\n'
                )
                runner.chmod(0o755)

            for adaptor, multi_segment in (
                ("nvidia", True),
                ("du", False),
                ("ppu", True),
                ("metax", True),
            ):
                with self.subTest(adaptor=adaptor):
                    capture = root / f"{adaptor}.calls"
                    env = dict(
                        os.environ, FLAGCX_ADAPTOR=adaptor, CI_CAPTURE=str(capture)
                    )
                    result = subprocess.run(
                        ["bash", str(script)], env=env, capture_output=True, text=True
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    calls = capture.read_text().splitlines()
                    dedicated = [
                        line for line in calls
                        if "--gtest_filter=SymMemTest." in line
                    ]
                    self.assertEqual(len(dedicated), 3 if multi_segment else 1)
                    self.assertEqual(
                        sum(
                            "UntrackedVmmMappingIsRejected" in line
                            for line in dedicated
                        ),
                        1,
                    )
                    self.assertEqual(
                        sum(
                            "FLAGCX_CI_REQUIRE_MULTI_SEGMENT=1" in line
                            for line in dedicated
                        ),
                        2 if multi_segment else 0,
                    )
                    self.assertEqual(
                        sum(
                            "FLAGCX_VMM_MULTI_SEGMENT_ENABLE=0" in line
                            for line in calls
                        ),
                        2 if adaptor == "du" else 0,
                    )

            invalid = subprocess.run(
                ["bash", str(script)],
                env=dict(os.environ, FLAGCX_ADAPTOR="unknown"),
                capture_output=True,
                text=True,
            )
            self.assertEqual(invalid.returncode, 2)
            self.assertIn("FLAGCX_ADAPTOR must be", invalid.stderr)

            failed = subprocess.run(
                ["bash", str(script)],
                env=dict(
                    os.environ,
                    FLAGCX_ADAPTOR="du",
                    CI_CAPTURE=str(root / "failed.calls"),
                    CI_FAIL_CASE="UntrackedVmmMappingIsRejected",
                ),
                capture_output=True,
                text=True,
            )
            self.assertEqual(failed.returncode, 7)

            for adaptor, expected_status in (
                ("nvidia", 0), ("metax", 0), ("du", 1), ("ppu", 1)
            ):
                with self.subTest(multicast=adaptor):
                    capture = root / f"{adaptor}-multicast.calls"
                    env = dict(
                        os.environ,
                        FLAGCX_ADAPTOR=adaptor,
                        FLAGCX_CI_REQUIRE_MULTICAST="1",
                        CI_CAPTURE=str(capture),
                    )
                    result = subprocess.run(
                        ["bash", str(script)],
                        env=env,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, expected_status)
                    dedicated = [
                        line for line in capture.read_text().splitlines()
                        if "--gtest_filter=SymMemTest." in line
                    ]
                    self.assertEqual(
                        sum(
                            "FLAGCX_CI_REQUIRE_MULTICAST=1" in line
                            for line in dedicated
                        ),
                        3 if expected_status == 0 else 0,
                    )


if __name__ == "__main__":
    unittest.main()
