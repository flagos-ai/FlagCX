"""Capture only the timed workload's descendants before CI kills them."""

import argparse
from pathlib import Path
import shutil
import signal
import subprocess
import time


active_debugger = None


def stop(signum, frame):
    if active_debugger is not None:
        active_debugger.kill()
        active_debugger.wait()
    raise SystemExit(0)


def read(path):
    try:
        return path.read_text().strip()
    except OSError as error:
        return str(error)


def capture(root_pid):
    global active_debugger
    processes = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            processes[int(path.name)] = int(fields[1])
        except (OSError, IndexError, ValueError):
            continue
    descendants = {root_pid}
    while True:
        children = {pid for pid, parent in processes.items() if parent in descendants}
        expanded = descendants | children
        if expanded == descendants:
            break
        descendants = expanded

    print("Pre-timeout workload diagnostics (not a postmortem stack)", flush=True)
    debugger = shutil.which("gdb")
    if debugger is None:
        print("gdb unavailable; collecting /proc thread states and kernel stacks", flush=True)
    for pid in sorted(descendants):
        process = Path("/proc") / str(pid)
        if not process.exists():
            continue
        name = read(process / "comm")
        print(f"\nPID {pid} PPID {processes.get(pid)} name={name}", flush=True)
        try:
            threads = sorted((process / "task").iterdir())
        except OSError:
            continue
        for thread in threads:
            print(f"TID {thread.name} name={read(thread / 'comm')} "
                  f"wchan={read(thread / 'wchan')}\n{read(thread / 'stack')}", flush=True)
        if debugger is not None and name.startswith("runner_mpi_test"):
            # Bound each attach so diagnostics cannot consume the job budget.
            active_debugger = subprocess.Popen(
                [debugger, "-q", "-nx", "-batch", "-p", str(pid),
                 "-ex", "set pagination off", "-ex", "thread apply all bt",
                 "-ex", "detach"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            try:
                output, _ = active_debugger.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                active_debugger.kill()
                output, _ = active_debugger.communicate()
                print("gdb attach exceeded 5s", flush=True)
            print(output, flush=True)
            active_debugger = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--delay", type=float, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    time.sleep(max(0, args.delay))
    capture(args.pid)
