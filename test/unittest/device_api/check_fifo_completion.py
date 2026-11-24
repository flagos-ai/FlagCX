#!/usr/bin/env python3
"""Exercise actual FIFO producers, host accessors and completion arithmetic.

Accessor and completion-consumer definitions are extracted to exclude platform
allocation/lifecycle dependencies. Only RMA submission and logging are mocked.
"""

import argparse
from pathlib import Path
import shlex
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default="c++")
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    source = (root / "flagcx/core/flagcx_kernel_host.cc").read_text()
    accessors = source[source.index("// Common accessors (trd)") :
                       source.index("// FIFO init / destroy")]
    # Drop the section separator immediately before FIFO lifecycle code.
    accessors = accessors[:accessors.rfind("// ==========")]
    with tempfile.TemporaryDirectory(prefix="flagcx-fifo-") as temporary:
        out = Path(temporary)
        (out / "trigger_accessors.inc").write_text(accessors)
        proxy = (root / "flagcx/core/proxy.cc").read_text()
        progress = proxy[proxy.index("static flagcxResult_t flagcxKernelWorkerProgressOnce") :]
        guard_start = progress.index("  if (res == flagcxSuccess) {\n    res = flagcxKernelProxyValidateFormat")
        guard_end = progress.index("    switch (pending.getPrim())", guard_start)
        signals_start = progress.index("      case flagcxDevicePrimSignal:")
        signals_end = progress.index("      case flagcxDevicePrimWaitSignal:", signals_start)
        put_start = progress.index("      case flagcxDevicePrimPutSignal:")
        put_end = progress.index("      case flagcxDevicePrimPutValue:", put_start)
        # Execute the real worker branches with only the network submission
        # mocked. This catches missing/double context expansion at the call site.
        (out / "completion_consumer.inc").write_text(
            progress[guard_start:guard_end]
            + "    switch (pending.getPrim()) {\n"
            + progress[signals_start:signals_end]
            + progress[put_start:put_end]
            + "      default: res = flagcxInvalidArgument; break;\n    }\n"
        )
        command = shlex.split(args.cxx) + [
            "-std=c++11", "-DFLAGCX_DEVICE_COMPILE=1", "-Wall", "-Wextra", "-Wno-unused-parameter",
            "-Wno-unused-function", "-g", "-I" + str(out),
        ]
        # Use the real device view with host DefaultPlatform atomics. Skip only
        # host Window construction, which pulls in unrelated allocation code.
        if args.sanitize:
            command += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
        for directory in ("flagcx/include", "flagcx/adaptor/include",
                          "flagcx/core/include", "flagcx/runner/include",
                          "flagcx/service/include", "third-party/json/single_include"):
            command += ["-I" + str(root / directory)]
        for bits in (64, 32):
            binary = out / f"fifo_completion_{bits}"
            build = command + (["-DFLAGCX_FIFO_TEST_WORD32=1"] if bits == 32 else [])
            build += [str(root / "test/unittest/device_api/test_fifo_completion.cpp"),
                      "-o", str(binary)]
            subprocess.run(build, check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
