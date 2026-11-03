#!/usr/bin/env python3
"""Check GPU one-sided adaptor cases against the selected platform policy."""

from __future__ import annotations

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
) -> tuple[list[str], list[str]]:
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
            if UNSUPPORTED_GPU_MR_REASON in reason:
                unsupported.append(name)
            else:
                errors.append(f"NetAdaptorLoopback.{name}: skipped or not run")
            continue
        if case.get("status") != "run" or case.get("result") != "completed":
            errors.append(f"NetAdaptorLoopback.{name}: skipped or not run")
    return errors, unsupported


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: check_adaptor_gpu_tests.py <gtest-xml>")
    failures, unsupported = check_results(Path(sys.argv[1]))
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        sys.exit(1)
    passed = set(REQUIRED_CASES) - set(unsupported)
    if passed:
        print(f"GPU one-sided adaptor cases passed: {', '.join(sorted(passed))}")
    elif not unsupported:
        print("No GPU one-sided adaptor cases are supported on this platform")
    if unsupported:
        print(
            "GPU one-sided adaptor cases skipped after runtime capability check: "
            f"{', '.join(sorted(unsupported))}"
        )
