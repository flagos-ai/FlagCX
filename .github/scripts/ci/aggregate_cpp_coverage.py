#!/usr/bin/env python3
"""Build an informational matrix summary without inventing a cross-platform denominator."""

import argparse
import json
from pathlib import Path


METRICS = (
    ("Line", "line_percent"),
    ("Function", "function_percent"),
    ("Branch", "branch_percent"),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = sorted(args.input_dir.rglob("report.json"))
    if not reports:
        raise SystemExit(f"No C++ coverage reports found under {args.input_dir}")

    rows = []
    for path in reports:
        report = json.loads(path.read_text())
        metrics = report["metrics"]
        rows.append((
            report["platform"],
            report.get("collection_status", "unknown"),
            *(metrics.get(key) for _, key in METRICS),
            len(report.get("unmeasured_sources", []))
            + len(report.get("unmeasured_device_sources", [])),
        ))

    lines = [
        "# C++ Coverage Matrix",
        "",
        "Platform and suite reports are kept separate. A cross-platform denominator "
        "is not calculated until the coverage policy is approved.",
        "",
        "| Platform / suite | Status | Lines | Functions | Branches | Unmeasured production sources |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for platform, status, *values in rows:
        formatted = ["—" if value is None else f"{value:.2f}%" for value in values[:3]]
        lines.append(
            f"| {platform} | {status} | {formatted[0]} | {formatted[1]} | "
            f"{formatted[2]} | {values[3]} |"
        )
    lines += [
        "",
        "Reports are informational; no 90% merge threshold is enforced.",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    (args.output.parent / "matrix.json").write_text(
        json.dumps(
            {"schema_version": 1, "aggregate_metric": None,
             "reason": "coverage denominator policy is pending approval",
             "reports": [json.loads(path.read_text()) for path in reports]},
            indent=2,
        ) + "\n"
    )


if __name__ == "__main__":
    main()
