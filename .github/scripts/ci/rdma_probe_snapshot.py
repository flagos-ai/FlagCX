#!/usr/bin/env python3
"""Best-effort procfs evidence, isolated and bounded by the probe supervisor."""

import argparse
import json
from pathlib import Path


def save(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def read_field(path, limit=65536):
    try:
        with path.open("rb") as source:
            data = source.read(limit + 1)
        return dict(text=data[:limit].decode(errors="replace").replace("\0", " "),
                    truncated=len(data) > limit)
    except OSError as error:
        return dict(error=f"{type(error).__name__}: {error}")


def snapshot_process(proc, pid, identity, output):
    """Persist each field before the next procfs read, even if that read hangs."""
    record = dict(pid=pid, expected_start_time=identity, fields={}, threads={},
                  complete=False)
    destination = output / f"pid-{pid}.json"
    save(destination, record)
    try:
        fields = (proc / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[19] != identity:
            record["error"] = "PID was reused; evidence skipped"
            save(destination, record)
            return
    except (OSError, IndexError) as error:
        record["error"] = f"{type(error).__name__}: {error}"
        save(destination, record)
        return

    root = proc / str(pid)
    for name in ("stat", "status", "cmdline", "wchan", "maps", "syscall", "stack"):
        record["reading"] = str(root / name)
        save(destination, record)
        record["fields"][name] = read_field(root / name, 262144 if name == "maps" else 65536)
        save(destination, record)
    try:
        tasks = sorted((p for p in (root / "task").iterdir() if p.name.isdigit()),
                       key=lambda p: int(p.name))
    except OSError as error:
        record["task_error"] = f"{type(error).__name__}: {error}"
        tasks = []
    # Keep the thread-state overview broader than the stack sample; include
    # blocked threads first, and always retain the process leader's stack.
    states = {}
    for task in tasks[:128]:
        record["reading"] = str(task / "stat")
        save(destination, record)
        states[task.name] = read_field(task / "stat")
        record["thread_states"] = states
        save(destination, record)
    def priority(task):
        text = states.get(task.name, {}).get("text", "")
        state = text.rsplit(")", 1)[-1].split()
        return (task.name != str(pid), not (state and state[0] == "D"), int(task.name))
    selected = sorted(tasks[:128], key=priority)[:16]
    record["threads_omitted"] = len(tasks) - len(selected)
    record["thread_states_omitted"] = max(0, len(tasks) - 128)
    for task in selected:
        thread = record["threads"][task.name] = {}
        for name in ("status", "wchan", "syscall", "stack"):
            record["reading"] = str(task / name)
            save(destination, record)
            thread[name] = read_field(task / name)
            save(destination, record)
    record.pop("reading", None)
    record["complete"] = True
    save(destination, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--process", action="append", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for process in args.process:
        pid, identity = process.split(":", 1)
        if not pid.isdigit() or not identity.isdigit():
            parser.error("process identity must be PID:start_time")
        snapshot_process(Path("/proc"), int(pid), identity, args.output)


if __name__ == "__main__":
    main()
