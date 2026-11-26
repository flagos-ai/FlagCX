#!/usr/bin/env python3
"""Require the real MTML dependency used by native MCCL initialization."""

import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import sys


def library_inventory():
    roots = [*os.environ.get("LD_LIBRARY_PATH", "").split(":"),
             "/usr/lib/x86_64-linux-gnu", "/usr/local/lib"]
    for root in dict.fromkeys(path for path in roots if path):
        try:
            for path in sorted(Path(root).glob("libmtml.so*")):
                print(f"MTML candidate: {path} -> {path.resolve()}", file=sys.stderr)
        except OSError as error:
            print(f"Cannot inspect {root}: {error}", file=sys.stderr)
    ldconfig = shutil.which("ldconfig")
    if ldconfig:
        try:
            result = subprocess.run([ldconfig, "-p"], capture_output=True,
                                    text=True, timeout=5)
            entries = [line.strip() for line in result.stdout.splitlines()
                       if "libmtml.so" in line]
            print("MTML loader cache: " + ("; ".join(entries) or "<none>"),
                  file=sys.stderr)
        except (OSError, subprocess.TimeoutExpired) as error:
            print(f"Cannot inspect loader cache: {error}", file=sys.stderr)


def main():
    for name in ("MTHREADS_VISIBLE_DEVICES", "MTHREADS_DRIVER_CAPABILITIES",
                 "MUSA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"):
        print(f"MUSA runtime {name}={os.environ.get(name, '<unset>')}", flush=True)
    try:
        # Match MCCL's unversioned name and resolve transitive dependencies now.
        library = ctypes.CDLL("libmtml.so", mode=os.RTLD_NOW | os.RTLD_GLOBAL)
    except OSError as error:
        print(f"MUSA MTML preflight failed: {error}", file=sys.stderr)
        library_inventory()
        print("Check the host MTML installation and the container runtime's "
              "utility library mapping. Set MTHREADS_VISIBLE_DEVICES and "
              "MTHREADS_DRIVER_CAPABILITIES=compute,utility when creating "
              "the container.", file=sys.stderr)
        return 1
    print(f"MUSA MTML loaded: {library._name}", flush=True)
    maps = Path("/proc/self/maps")
    if maps.is_file():
        paths = {line.split()[-1] for line in maps.read_text().splitlines()
                 if "libmtml.so" in line}
        for path in sorted(paths):
            print(f"MUSA MTML mapped: {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
