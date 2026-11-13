#!/usr/bin/env python3
"""Log MetaX device state and require four usable CI devices."""

import argparse
import ctypes
import importlib.metadata
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


REQUIRED_DEVICES = 4


def log(message):
    print(f"MetaX device preflight: {message}", flush=True)


def runtime_device_count():
    candidates = (
        "libmcruntime.so",
        "/opt/maca/lib64/libmcruntime.so",
        "/opt/maca/lib/libmcruntime.so",
    )
    errors = []
    for candidate in candidates:
        try:
            runtime = ctypes.CDLL(candidate)
            break
        except OSError as error:
            errors.append(f"{candidate}: {error}")
    else:
        raise RuntimeError("cannot load MACA runtime: " + "; ".join(errors))

    get_count = runtime.mcGetDeviceCount
    get_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_count.restype = ctypes.c_int
    count = ctypes.c_int()
    result = get_count(ctypes.byref(count))
    log(f"MACA runtime={runtime._name}, mcGetDeviceCount status={result}, count={count.value}")
    if result != 0:
        raise RuntimeError(f"mcGetDeviceCount failed with status {result}")
    return count.value


def check_mccl_alltoall():
    header_candidates = (
        Path("/opt/maca/include/mccl/mccl.h"),
        Path("/opt/maca/include/mccl.h"),
    )
    header = next((path for path in header_candidates if path.is_file()), None)
    if header is None:
        raise RuntimeError("MCCL header mccl.h is missing")
    declaration = header.read_text(errors="replace")
    if re.search(r"\bmcclAllToAll\s*\(", declaration) is None:
        raise RuntimeError(f"mcclAllToAll is not declared in {header}")

    errors = []
    for candidate in ("libmccl.so", "/opt/maca/lib64/libmccl.so",
                      "/opt/maca/lib/libmccl.so"):
        try:
            library = ctypes.CDLL(candidate)
            break
        except OSError as error:
            errors.append(f"{candidate}: {error}")
    else:
        raise RuntimeError("cannot load MCCL: " + "; ".join(errors))
    if not hasattr(library, "mcclAllToAll"):
        raise RuntimeError(f"mcclAllToAll symbol is missing from {library._name}")
    get_version = library.mcclGetVersion
    get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_version.restype = ctypes.c_int
    version = ctypes.c_int()
    result = get_version(ctypes.byref(version))
    log(f"MCCL library={library._name}, version={version.value}, version status={result}")
    if result != 0:
        raise RuntimeError(f"mcclGetVersion failed with status {result}")
    log(f"mcclAllToAll declaration={header}, symbol=present")


def check_torch_devices():
    import torch

    metadata = importlib.metadata.distribution("torch")
    print(f"Torch: {torch.__version__}, path={torch.__file__}", flush=True)
    print(f"Torch direct_url: {metadata.read_text('direct_url.json')}", flush=True)
    # init() gives the actual driver/runtime error instead of only False/zero.
    torch.cuda.init()
    visible = torch.cuda.device_count()
    print(f"Torch available={torch.cuda.is_available()}, count={visible}", flush=True)
    if not torch.cuda.is_available() or visible < REQUIRED_DEVICES:
        raise RuntimeError("MetaX CI requires at least four Torch-visible devices")
    for index in range(REQUIRED_DEVICES):
        with torch.cuda.device(index):
            value = torch.arange(16, device=f"cuda:{index}")
            result = value.sum().item()
            if result != 120:
                raise RuntimeError(f"cuda:{index} read/write check failed: {result}")
            torch.cuda.synchronize(index)
            print(f"cuda:{index}: {torch.cuda.get_device_name(index)}, sum={result}: PASS", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workload", choices=("perf", "torch-api"))
    args = parser.parse_args(argv)

    log(f"workload={args.workload}, Python={sys.executable}")
    for name in ("CUDA_VISIBLE_DEVICES", "MACA_VISIBLE_DEVICES", "MACA_PATH"):
        log(f"{name}={os.environ.get(name, '<unset>')}")
    for path in ("/opt/mxdriver", "/dev/mxcd", "/dev/dri"):
        log(f"{path} present={Path(path).exists()}")
    if shutil.which("mx-smi"):
        try:
            result = subprocess.run(["mx-smi"], capture_output=True, text=True, timeout=30)
            log(f"mx-smi exit={result.returncode}\n{result.stdout}{result.stderr}")
        except (OSError, subprocess.TimeoutExpired) as error:
            log(f"mx-smi failed: {error}")
    else:
        log("mx-smi unavailable")

    failures = []
    try:
        native_count = runtime_device_count()
        if native_count < REQUIRED_DEVICES:
            raise RuntimeError(f"MACA sees {native_count} devices; four are required")
    except Exception as error:
        failures.append(f"MACA runtime: {error}")
    try:
        if args.workload == "perf":
            check_mccl_alltoall()
        else:
            check_torch_devices()
    except Exception as error:
        failures.append(f"{args.workload}: {error}")
    if failures:
        for failure in failures:
            log(f"FAILED: {failure}")
        return 1

    log("PASSED: four MetaX devices are available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
