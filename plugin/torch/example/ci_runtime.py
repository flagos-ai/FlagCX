"""Check which FlagCX library a CI Torch worker actually loaded."""

import os
from pathlib import Path
import sys
from functools import lru_cache
import hashlib


@lru_cache(maxsize=None)
def _digest(path):
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.digest()


def report_ci_stage(stage):
    if not os.environ.get("FLAGCX_CI_EXPECT_FLAGCX_LIBRARY"):
        return
    rank = os.environ.get("RANK", "?")
    print(f"rank {rank} stage: {stage}", flush=True)


def report_ci_imports():
    if not os.environ.get("FLAGCX_CI_EXPECT_FLAGCX_LIBRARY"):
        return
    for name in ("torch", "torch_npu", "torch_mlu", "torch_musa", "torch_txda",
                 "torch_gcu", "torch_ptpu", "flagcx", "flagcx._C"):
        module = sys.modules.get(name)
        if module is not None:
            print(f"rank {os.environ.get('RANK', '?')} module {name}: "
                  f"{getattr(module, '__file__', None)}", flush=True)


def verify_ci_flagcx_library(maps_path="/proc/self/maps", *, stage="runtime",
                             require_loaded=True):
    expected = os.environ.get("FLAGCX_CI_EXPECT_FLAGCX_LIBRARY")
    if not expected:
        return

    expected_path = Path(expected).resolve(strict=True)
    project_root = expected_path.parents[2]
    allowed = {
        expected_path,
        (project_root / "src/flagcx/lib/libflagcx.so").resolve(strict=True),
    }
    mapped = set()
    raw_lines = []
    for line in Path(maps_path).read_text().splitlines():
        if "flagcx" in line.lower():
            raw_lines.append(line)
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and "libflagcx.so" in fields[5]:
            mapped.add(fields[5])

    rank = os.environ.get("RANK", "?")
    print(f"rank {rank} {stage} mapped libflagcx.so: {sorted(mapped)}", flush=True)
    print(f"rank {rank} {stage} raw flagcx mappings: {raw_lines}", flush=True)
    if not mapped:
        if require_loaded:
            raise RuntimeError(
                f"Torch CI has no mapped libflagcx.so at {stage}; "
                f"expected one of {sorted(map(str, allowed))}"
            )
        return False
    def matches_checkout(path):
        candidate = Path(path).resolve()
        if candidate in allowed:
            return True
        try:
            return _digest(candidate) == _digest(expected_path)
        except OSError:
            return False

    if any(not matches_checkout(path) for path in mapped):
        raise RuntimeError(
            "Torch CI loaded a FlagCX library that differs from this checkout's build; "
            f"expected one of {sorted(map(str, allowed))} or a byte-identical "
            f"installed copy, got {sorted(mapped)}"
        )
    return True
