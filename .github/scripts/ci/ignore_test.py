#!/usr/bin/env python3
"""Read structured per-platform test exclusions supplied by the CI workflow."""

import argparse
import json
import os
import sys


def config():
    raw = os.environ.get("FLAGCX_CI_IGNORE_TEST", "{}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid FLAGCX_CI_IGNORE_TEST JSON: {error}")
    if not isinstance(value, dict):
        raise SystemExit("ignore_test must be a JSON object")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("gtest", "phase", "perf", "torch", "runner"))
    parser.add_argument("suite", nargs="?")
    parser.add_argument("mode", nargs="?")
    parser.add_argument("name", nargs="?")
    args = parser.parse_args()
    section = "phases" if args.kind == "phase" else args.kind
    data = config().get(section, {})

    if args.kind == "gtest":
        patterns = [
            test
            for entry in data
            if entry.get("suite") == args.suite
            for test in entry.get("tests", [])
        ]
        print(":".join(patterns))
        return

    if args.kind == "phase":
        match = next(
            (entry for entry in data
             if entry.get("suite") == args.suite and entry.get("name") == args.mode),
            None,
        )
        print(match.get("reason", "") if match else "")
        return

    if args.kind == "perf":
        if args.suite in data.get("modes", []):
            print(" ".join(data.get("operations", [])))
        return

    if args.kind == "torch":
        if args.suite in data.get("modes", []):
            print(" ".join(data.get("operations", [])))
        return

    for entry in data:
        if entry.get("test") == args.name and args.mode in entry.get("modes", []):
            print(entry.get("reason", ""))
            return
    print("")


if __name__ == "__main__":
    main()
