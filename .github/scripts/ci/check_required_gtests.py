#!/usr/bin/env python3
"""Require named GoogleTest cases to run and pass in every supplied XML file."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def check_file(path: Path, required: list[str]) -> list[str]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        return [f"{path}: cannot read GoogleTest XML: {error}"]

    cases: dict[str, list[ET.Element]] = {}
    for suite in root.iter("testsuite"):
        suite_name = suite.get("name")
        for case in suite.findall("testcase"):
            name = f"{suite_name}.{case.get('name')}"
            cases.setdefault(name, []).append(case)

    errors = []
    for name in required:
        matches = cases.get(name, [])
        if len(matches) != 1:
            errors.append(f"{path}: {name}: expected once, found {len(matches)}")
            continue
        case = matches[0]
        if case.find("failure") is not None or case.find("error") is not None:
            errors.append(f"{path}: {name}: failed")
        elif (case.find("skipped") is not None or
              case.get("status") != "run" or
              case.get("result") != "completed"):
            errors.append(f"{path}: {name}: skipped or not run")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, dest="cases")
    parser.add_argument("xml", nargs="+", type=Path)
    args = parser.parse_args()
    errors = [error for path in args.xml for error in check_file(path, args.cases)]
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"Required GoogleTest cases passed in {len(args.xml)} report(s): "
          f"{', '.join(args.cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
