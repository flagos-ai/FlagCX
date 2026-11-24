#!/usr/bin/env python3
"""Read platform ignore_test, exported from configs/<platform>.yml by CI."""

import argparse
import json
import os
import sys
from pathlib import Path


ADAPTOR_PLATFORMS = {
    "nvidia": "cuda",
    "du": "hygon",
    "iluvatar": "iluvatar",
    "metax": "metax",
    "klx": "kunlunxin",
    "musa": "mthreads",
    "ppu": "ppu",
    "ascend": "ascend",
    "enflame": "enflame",
}


def config():
    raw = os.environ.get("FLAGCX_CI_IGNORE_TEST")
    if raw is None:
        # Local execution can read YAML directly. Hardware jobs receive JSON
        # from load_platform_config.rb and need no YAML package in the image.
        platform = os.environ.get("FLAGCX_CI_PLATFORM")
        if not platform:
            platform = ADAPTOR_PLATFORMS.get(os.environ.get("FLAGCX_ADAPTOR", ""))
        if not platform:
            return {}
        config_root = Path(
            os.environ.get(
                "FLAGCX_CI_CONFIG_DIR",
                Path(__file__).resolve().parents[2] / "configs",
            )
        )
        path = config_root / f"{platform}.yml"
        # Some lightweight test harnesses copy this helper without the full
        # repository config tree. Keep their generic behavior unchanged; CI
        # runners set FLAGCX_CI_CONFIG_DIR when platform exclusions are needed.
        if not path.is_file():
            return {}
        try:
            import yaml

            value = yaml.safe_load(path.read_text()).get("ignore_test", {})
        except ImportError:
            import subprocess

            raw = subprocess.check_output(
                ["ruby", "-e", 'require "yaml"; require "json"; puts JSON.generate(YAML.safe_load(File.read(ARGV.fetch(0))).fetch("ignore_test", {}))', str(path)],
                text=True,
            )
            value = json.loads(raw)
        if not isinstance(value, dict):
            raise SystemExit(f"{path}: ignore_test must be a map")
        return value
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid FLAGCX_CI_IGNORE_TEST JSON: {error}")
    if not isinstance(value, dict):
        raise SystemExit("ignore_test must be a JSON object")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "kind", choices=("gtest", "phase", "case", "perf", "torch", "runner")
    )
    parser.add_argument("suite", nargs="?")
    parser.add_argument("mode", nargs="?")
    parser.add_argument("name", nargs="?")
    args = parser.parse_args()
    section = {"phase": "phases", "case": "cases"}.get(args.kind, args.kind)
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
             if entry.get("suite") == args.suite
             and entry.get("name") == args.mode
             and all(
                 os.environ.get(key) == str(value)
                 for key, value in entry.get("when", {}).items()
             )),
            None,
        )
        print(match.get("reason", "") if match else "")
        return

    if args.kind == "case":
        match = next(
            (
                entry
                for entry in data
                if entry.get("suite") == args.suite
                and args.mode in entry.get("tests", [])
            ),
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
