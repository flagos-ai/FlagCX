"""Run one torch worker under GDB and save its native crash stack."""

import os
from pathlib import Path
import subprocess
import sys


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: torch_rank_backtrace.py <torch-test-script>")

    rank = int(os.environ["RANK"])
    log_dir = Path(os.environ["FLAGCX_CI_DIAGNOSTIC_DIR"])
    log_dir.mkdir(parents=True, exist_ok=True)
    trace = log_dir / f"rank-{rank}.log"
    command = [
        "gdb", "-q", "-nx", "-batch", "--return-child-result",
        "-ex", "set pagination off",
        "-ex", "set print frame-arguments all",
        "-ex", "set print elements 32",
        "-ex", "run",
        # A completed worker has no threads; avoid turning a passing test into
        # a GDB command error while still capturing every thread on a stop.
        "-ex", "python import gdb; threads = gdb.selected_inferior().threads(); gdb.execute('thread apply all bt full') if threads else None; gdb.execute('info registers') if threads else None",
        "--args", sys.executable, sys.argv[1],
    ]
    with trace.open("w") as output:
        print(f"rank={rank} local_rank={os.environ.get('LOCAL_RANK')} "
              f"worker_pid={os.getpid()}", file=output, flush=True)
        try:
            status = subprocess.run(command, stdout=output,
                                    stderr=subprocess.STDOUT, check=False).returncode
        except OSError as error:
            print(f"Failed to run GDB: {error}", file=output, flush=True)
            status = 127
        output.flush()
        trace_text = trace.read_text()
        if "received signal SIGSEGV" in trace_text:
            status = 139
        elif "exited normally" in trace_text:
            status = 0
        print(f"gdb exit={status}", file=output, flush=True)
    print(f"Rank {rank} native backtrace: {trace} (gdb exit={status})", flush=True)
    # The primary diagnostic run must fail when its child fails. The later
    # reproduction keeps the old behavior so all ranks can save their traces.
    if os.environ.get("FLAGCX_TORCH_GDB_PROPAGATE_FAILURE") == "1":
        return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
