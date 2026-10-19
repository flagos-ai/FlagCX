#!/usr/bin/env python3
"""Check GPU one-sided adaptor cases against the selected platform policy."""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


REQUIRED_CASES = (
    "GpuIput",
    "GpuIget",
    "GpuBatchPutAndGet",
    "GpuBounds",
)
UNSUPPORTED_GPU_MR_REASON = "Selected net adaptor does not advertise GPU MR"


def check_results(
    path: Path,
    allowed_skips: set[str] | None = None,
    *,
    allow_gpu_mr_unsupported: bool = False,
) -> tuple[list[str], list[str]]:
    allowed_skips = allowed_skips or set()
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        return [f"Cannot read adaptor GoogleTest XML at {path}: {error}"], []

    cases = {}
    for suite in root.iter("testsuite"):
        if suite.get("name") == "NetAdaptorLoopback":
            for case in suite.findall("testcase"):
                cases.setdefault(case.get("name"), []).append(case)

    errors = []
    unsupported = []
    for name in REQUIRED_CASES:
        matches = cases.get(name, [])
        if len(matches) != 1:
            errors.append(f"NetAdaptorLoopback.{name}: expected once, found {len(matches)}")
            continue
        case = matches[0]
        if case.find("failure") is not None or case.find("error") is not None:
            errors.append(f"NetAdaptorLoopback.{name}: failed")
            continue
        skipped = case.find("skipped")
        if skipped is not None:
            reason = " ".join(
                " ".join((skipped.get("message", ""), "".join(skipped.itertext()))).split()
            )
            if allow_gpu_mr_unsupported and UNSUPPORTED_GPU_MR_REASON in reason:
                unsupported.append(name)
            elif name not in allowed_skips:
                errors.append(f"NetAdaptorLoopback.{name}: skipped or not run")
            continue
        if case.get("status") != "run" or case.get("result") != "completed":
            errors.append(f"NetAdaptorLoopback.{name}: skipped or not run")
    return errors, unsupported


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[2] not in {"ascend", "required"}:
        sys.exit("Usage: check_adaptor_gpu_tests.py <gtest-xml> <ascend|required>")
    allowed_skips = set(
        os.environ.get("FLAGCX_CI_ADAPTOR_GPU_SKIP_CASES", "").split()
    )
    failures, unsupported = check_results(
        Path(sys.argv[1]),
        allowed_skips,
        allow_gpu_mr_unsupported=sys.argv[2] == "ascend",
    )
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        sys.exit(1)
    passed = set(REQUIRED_CASES) - allowed_skips - set(unsupported)
    if passed:
        print(f"GPU one-sided adaptor cases passed: {', '.join(sorted(passed))}")
    elif not unsupported:
        print("No GPU one-sided adaptor cases are supported on this platform")
    if unsupported:
        print(
            "GPU MR coverage is unsupported on Ascend; these required cases "
            f"were explicitly skipped: {', '.join(unsupported)}"
        )
    if allowed_skips:
        print(f"Expected capability skips: {', '.join(sorted(allowed_skips))}")
