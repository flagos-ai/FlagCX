#!/usr/bin/env python3
"""Log MetaX device state and require four usable CI devices."""

import argparse
import ctypes
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


def torch_device_count():
    import torch

    torch_dir = Path(torch.__file__).resolve().parent
    log(f"Python={sys.executable}")
    log(f"torch={torch.__version__}, path={torch.__file__}")
    log(f"torch.version.cuda={torch.version.cuda}")
    for name in ("libtorch_cuda.so", "libc10_cuda.so"):
        log(f"{name} present={(torch_dir / 'lib' / name).is_file()}")
    available = torch.cuda.is_available()
    count = torch.cuda.device_count()
    log(f"torch.cuda.is_available={available}, device_count={count}")
    if not available or count < REQUIRED_DEVICES:
        raise RuntimeError("mcPyTorch does not expose four CUDA-compatible MetaX devices")
    for device in range(REQUIRED_DEVICES):
        value = torch.ones(1, device=f"cuda:{device}")
        torch.cuda.synchronize(device)
        if value.item() != 1:
            raise RuntimeError(f"cuda:{device} tensor allocation/readback failed")
        log(f"cuda:{device} tensor allocation/readback passed")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workload", choices=("torch-api", "perf"))
    args = parser.parse_args()

    log(f"workload={args.workload}, Python={sys.executable}")
    for name in ("CUDA_VISIBLE_DEVICES", "MACA_VISIBLE_DEVICES", "MACA_PATH",
                 "MACA_CLANG_PATH", "CUDA_PATH", "CUCC_PATH"):
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
        check_mccl_alltoall()
    except Exception as error:
        failures.append(f"MCCL: {error}")
    if args.workload == "torch-api":
        try:
            torch_device_count()
        except Exception as error:
            failures.append(f"mcPyTorch: {error}")
    if failures:
        for failure in failures:
            log(f"FAILED: {failure}")
        return 1

    log("PASSED: four MetaX devices are available")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
