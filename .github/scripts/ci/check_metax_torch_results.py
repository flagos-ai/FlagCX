#!/usr/bin/env python3
"""Verify both four-rank MetaX Torch runs completed without skipping work."""

from pathlib import Path
import re
import sys


def validate(log):
    if "[SKIP]" in log:
        raise RuntimeError("MetaX Torch tests skipped accelerator work")
    expected = {"homogeneous", "heterogeneous"}
    completed = set()
    mode = None
    ranks = set()
    for line in log.splitlines():
        launch = re.search(r"\[INFO\] Launching PyTorch API tests in (\w+) mode", line)
        if launch:
            mode = launch[1]
            ranks = set()
        # A retry must establish a complete set of ranks independently.
        if line.startswith("Starting PyTorch API tests ("):
            ranks = set()
        if mode is not None:
            ranks.update(int(rank) for rank in re.findall(
                r"rank (\d+) after all_to_all_single with FLAGCX_GROUP1 \(with splits\)", line
            ))
        finish = re.search(r"\[INFO\] Completed PyTorch API tests in (\w+) mode", line)
        if finish:
            if mode != finish[1] or ranks != {0, 1, 2, 3}:
                raise RuntimeError(f"MetaX Torch {finish[1]} incomplete: ranks={sorted(ranks)}")
            completed.add(mode)
            mode = None
    if completed != expected:
        raise RuntimeError(f"MetaX Torch modes incomplete: {sorted(completed)}")


if __name__ == "__main__":
    validate(Path(sys.argv[1]).read_text(errors="replace"))
    print("MetaX Torch homogeneous/heterogeneous: all four ranks completed, no skips")
