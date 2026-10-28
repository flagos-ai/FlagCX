#!/usr/bin/env python3
"""Require the shared GPU one-sided adaptor cases to run in hardware CI."""

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


def check_results(path: Path, allowed_skips: set[str] | None = None) -> list[str]:
    allowed_skips = allowed_skips or set()
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        return [f"Cannot read adaptor GoogleTest XML at {path}: {error}"]

    cases = {}
    for suite in root.iter("testsuite"):
        if suite.get("name") == "NetAdaptorLoopback":
            for case in suite.findall("testcase"):
                cases.setdefault(case.get("name"), []).append(case)

    errors = []
    for name in REQUIRED_CASES:
        matches = cases.get(name, [])
        if len(matches) != 1:
            errors.append(f"NetAdaptorLoopback.{name}: expected once, found {len(matches)}")
            continue
        case = matches[0]
        if case.find("skipped") is not None and name in allowed_skips:
            continue
        if (
            case.get("status") != "run"
            or case.get("result") != "completed"
            or case.find("skipped") is not None
        ):
            errors.append(f"NetAdaptorLoopback.{name}: skipped or not run")
        elif case.find("failure") is not None or case.find("error") is not None:
            errors.append(f"NetAdaptorLoopback.{name}: failed")
    return errors


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: check_adaptor_gpu_tests.py <gtest-xml>")
    allowed_skips = set(
        os.environ.get("FLAGCX_CI_ADAPTOR_GPU_SKIP_CASES", "").split()
    )
    failures = check_results(Path(sys.argv[1]), allowed_skips)
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        sys.exit(1)
    passed = set(REQUIRED_CASES) - allowed_skips
    if passed:
        print(f"GPU one-sided adaptor cases passed: {', '.join(sorted(passed))}")
    else:
        print("No GPU one-sided adaptor cases are supported on this platform")
    if allowed_skips:
        print(f"Expected capability skips: {', '.join(sorted(allowed_skips))}")
