#!/usr/bin/env python3
"""Fail before the MetaX Torch build if MACA or Torch cannot use four GPUs."""

import ctypes
import importlib.metadata
import sys


def main():
    print(f"Python: {sys.executable}", flush=True)
    runtime = ctypes.CDLL("libmcruntime.so")
    count = ctypes.c_int()
    runtime.mcGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    runtime.mcGetDeviceCount.restype = ctypes.c_int
    status = runtime.mcGetDeviceCount(ctypes.byref(count))
    print(f"MACA mcGetDeviceCount: status={status}, count={count.value}", flush=True)
    if status != 0 or count.value < 4:
        raise RuntimeError("MetaX CI requires at least four usable MACA devices")

    import torch

    metadata = importlib.metadata.distribution("torch")
    print(f"Torch: {torch.__version__}, path={torch.__file__}", flush=True)
    print(f"Torch direct_url: {metadata.read_text('direct_url.json')}", flush=True)
    # init() gives the actual driver/runtime error instead of only False/zero.
    torch.cuda.init()
    visible = torch.cuda.device_count()
    print(f"Torch available={torch.cuda.is_available()}, count={visible}", flush=True)
    if not torch.cuda.is_available() or visible < 4:
        raise RuntimeError("MetaX CI requires at least four Torch-visible devices")
    for index in range(4):
        with torch.cuda.device(index):
            value = torch.arange(16, device=f"cuda:{index}")
            result = value.sum().item()
            if result != 120:
                raise RuntimeError(f"cuda:{index} read/write check failed: {result}")
            torch.cuda.synchronize(index)
            print(f"cuda:{index}: {torch.cuda.get_device_name(index)}, sum={result}: PASS", flush=True)


if __name__ == "__main__":
    main()
