#!/usr/bin/env python3
"""Detect allocation enums in the CI SDK without executing accelerator code."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess


FLAGS = (
    "mcDeviceMallocDefault", "mcDeviceMallocFinegrained",
    "mcMallocSignalMemory", "mcDeviceMallocWriteCoherence",
    "mcDeviceMallocMapPcieDefault", "mcDeviceMallocMapPcieCoherence",
    "mcDeviceMallocFixedMemDefault", "mcDeviceMallocFixedMemCoherence",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    command = shlex.split(args.cxx) + [
        "-std=c++17", "-x", "c++", "-fsyntax-only", "-",
        "-I" + str(args.sdk / "include"),
        "-I" + str(args.sdk / "include/mcr"),
    ]
    availability = {}
    base = subprocess.run(command, input="#include <mcr/mc_runtime.h>\n"
                          "void f() { void *p; mcMalloc(&p,8); }\n",
                          text=True, capture_output=True, timeout=20)
    (args.output / "base.log").write_text(base.stdout + base.stderr)
    if base.returncode:
        raise SystemExit("MACA headers/compiler unavailable; see sdk/base.log")
    for flag in FLAGS:
        source = ("#include <mcr/mc_runtime.h>\n"
                  f"void f() {{ void *p; mcExtMallocWithFlags(&p,8,{flag}); }}\n")
        try:
            result = subprocess.run(command, input=source, text=True,
                                    capture_output=True, timeout=20)
            availability[flag] = result.returncode == 0
            diagnostic = result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            availability[flag] = False
            diagnostic = "SDK enum compile check timed out\n"
        (args.output / (flag + ".log")).write_text(diagnostic)
    lines = ["#pragma once", "// Generated from the SDK installed in this image.",
             "static mcError_t probeNativeAlloc(int mode, void **ptr, size_t n) {",
             "switch (mode) {", "case 0: return mcMalloc(ptr, n);"]
    for index, flag in enumerate(FLAGS, 1):
        if availability[flag]:
            lines += [f"case {index}: return mcExtMallocWithFlags(ptr,n,{flag});"]
    lines += ["default: return mcErrorNotSupported;", "}", "}",
              "static bool probeModeAvailable(int mode) {",
              "switch (mode) {", "case 0: case 9: return true;"]
    lines += [f"case {index}: return true;"
              for index, flag in enumerate(FLAGS, 1) if availability[flag]]
    lines += ["default: return false;", "}", "}"]
    lines += ["static int64_t probeFlagValue(int mode) {", "switch (mode) {"]
    lines += [f"case {index}: return int64_t({flag});"
              for index, flag in enumerate(FLAGS, 1) if availability[flag]]
    lines += ["default: return -1;", "}", "}"]
    (args.output / "allocation_modes.h").write_text("\n".join(lines) + "\n")
    (args.output / "sdk-features.json").write_text(
        json.dumps(availability, indent=2) + "\n")


if __name__ == "__main__":
    main()
