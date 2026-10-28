#!/usr/bin/env python3
"""Generate FlagCX-owned host coverage reports and explicit source gaps."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


SOURCE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".hip", ".xpu",
}
DEVICE_SUFFIXES = {".cu", ".cuh", ".hip", ".xpu"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--skip-reason", default="")
    args = parser.parse_args()

    root = args.root.resolve()
    build_dir = args.build_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_json = output_dir / "coverage.json"
    summary_json = output_dir / "summary.json"
    html_report = output_dir / "index.html"

    compiler_version = subprocess.run(
        ["g++", "-dumpversion"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().splitlines()[0]
    compiler_major = compiler_version.split(".", 1)[0]
    gcov = shutil.which(f"gcov-{compiler_major}") or shutil.which("gcov")
    if not gcov:
        raise SystemExit(f"No gcov executable found for g++ {compiler_version}")

    gcno_files = list(build_dir.rglob("*.gcno")) if build_dir.exists() else []
    if gcno_files:
        gcovr = [
            sys.executable, "-m", "gcovr",
            "--root", str(root),
            "--gcov-executable", gcov,
            "--filter", "flagcx/",
            "--exclude", r".*/(third-party|test|tests|build|generated)/.*",
            "--txt-metric", "branch",
            "--json", str(raw_json),
            "--json-summary", str(summary_json),
            "--html-details", str(html_report),
            "--print-summary",
        ]
        subprocess.run(gcovr, cwd=root, check=True)
        metrics = json.loads(summary_json.read_text())
        measured = {
            (root / entry["file"]).resolve()
            if not Path(entry["file"]).is_absolute()
            else Path(entry["file"]).resolve()
            for entry in json.loads(raw_json.read_text()).get("files", [])
        }
        collection_status = "collected"
        if not measured:
            collection_status = "no-measured-production-sources"
    else:
        metrics = {
            "line_percent": None,
            "function_percent": None,
            "branch_percent": None,
        }
        measured = set()
        collection_status = "skipped" if args.skip_reason else "no-instrumented-objects"
        raw_json.write_text(json.dumps({"files": []}, indent=2) + "\n")
        summary_json.write_text(json.dumps(metrics, indent=2) + "\n")
    sources = sorted(
        path.resolve()
        for path in (root / "flagcx").rglob("*")
        if path.is_file() and path.suffix in SOURCE_SUFFIXES
        and not any(part in {"third-party", "test", "tests", "build", "generated"}
                    for part in path.parts)
    )

    gaps = [
        {
            "source": source.relative_to(root).as_posix(),
            "reason": "source was not present in gcov data for this platform build",
        }
        for source in sources
        if source not in measured and source.suffix not in DEVICE_SUFFIXES
    ]
    device_gaps = [
        {
            "source": source.relative_to(root).as_posix(),
            "reason": "device instructions are not covered by the host gcov backend",
        }
        for source in sources
        if source.suffix in DEVICE_SUFFIXES
    ]

    report = {
        "schema_version": 1,
        "platform": args.platform,
        "backend": "gcc-gcov-host",
        "scope": "FlagCX host translation units; device instructions are reported as gaps",
        "collection_status": collection_status,
        "skip_reason": args.skip_reason or None,
        "coverage_complete": collection_status == "collected" and not gaps and not device_gaps,
        "metrics": metrics,
        "unmeasured_sources": gaps,
        "unmeasured_device_sources": device_gaps,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    format_metric = lambda value: "N/A" if value is None else f"{value:.2f}%"
    (output_dir / "summary.md").write_text(
        f"# C++ Coverage: {args.platform}\n\n"
        f"- Collection status: {collection_status}\n"
        f"- Lines: {format_metric(report['metrics'].get('line_percent'))}\n"
        f"- Functions: {format_metric(report['metrics'].get('function_percent'))}\n"
        f"- Branches: {format_metric(report['metrics'].get('branch_percent'))}\n"
        f"- Unmeasured production sources: {len(gaps)}\n"
        f"- Unmeasured device sources: {len(device_gaps)}\n"
        "- 90% merge threshold: not enforced pending metric policy approval\n"
    )
    if collection_status == "no-instrumented-objects":
        raise SystemExit("Coverage run produced no instrumented C++ objects")
    if collection_status == "no-measured-production-sources":
        raise SystemExit("Coverage run produced no measured FlagCX production sources")


if __name__ == "__main__":
    main()
