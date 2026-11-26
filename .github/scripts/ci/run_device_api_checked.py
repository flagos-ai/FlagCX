#!/usr/bin/env python3
"""Run the shared MPI wrapper and require native Device API case coverage."""

from collections import Counter
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile


REQUIRED = {
    "test_device_api_intra": ("K10",),
    "test_device_api_inter": ("K14",),
    "test_device_ir_intra": ("S10",),
    "test_device_ir_inter": ("S14",),
    "test_device_ir_unified_intra": tuple(f"S{i}" for i in range(16, 26)),
    "test_device_ir_unified_inter": tuple(f"S{i}" for i in range(16, 26)),
}


def byte_count(value):
    match = re.fullmatch(r"([1-9][0-9]*)([KMG]?)", value.upper())
    if not match:
        raise ValueError(f"Invalid byte count: {value}")
    return int(match[1]) * 1024 ** ("", "K", "M", "G").index(match[2])


def invocation(args):
    """Read the actual MPI app contexts, including the two logical nodes."""
    apps = [[]]
    for arg in args:
        if arg == ":":
            apps.append([])
        else:
            apps[-1].append(arg)
    ranks, signature = 0, None
    for app in apps:
        binaries = [i for i, arg in enumerate(app) if Path(arg).name in REQUIRED]
        if len(binaries) != 1:
            raise ValueError("Expected one Device API binary in each MPI app context")
        index = binaries[0]
        rank_flags = [i for i, arg in enumerate(app[:index]) if arg in ("-np", "-n")]
        if len(rank_flags) != 1:
            raise ValueError("Expected one MPI rank count per app context")
        count = int(app[rank_flags[0] + 1])
        if count <= 0:
            raise ValueError("MPI rank count must be positive")
        ranks += count
        flags = app[index + 1:]
        def option(name):
            if flags.count(name) != 1:
                raise ValueError(f"Expected one {name} test option")
            return flags[flags.index(name) + 1]
        current = (Path(app[index]).name, byte_count(option("-b")),
                   byte_count(option("-e")), int(option("-f")))
        if signature is not None and current != signature:
            raise ValueError("MPI app contexts must run the same binary and size sweep")
        signature = current
    binary, low, high, factor = signature
    if factor <= 1 or high < low:
        raise ValueError("Invalid size sweep")
    sizes = []
    while low <= high:
        sizes.append(low)
        low *= factor
    return binary, ranks, sizes


def check_results(text, binary, ranks, sizes):
    errors = []
    records = re.findall(r"\[rank (\d+)\]\s+([KS]\d+[ab]?)\b[^\n]*?\b(PASS|FAIL|SKIP)\b", text)
    for rank, case, result in records:
        if result == "FAIL" or (case in REQUIRED[binary] and result == "SKIP"):
            errors.append(f"rank {rank}: {case} {result}")
    # MPI stdout may interleave ranks and sizes. Each final case result must
    # carry its own size; rank-0 banners cannot identify another rank's case.
    prefix = "FLAGCX_DEVICE_API_RESULT"
    observed = Counter()
    for line in text.splitlines():
        if prefix not in line:
            continue
        try:
            if not line.startswith(prefix + " "):
                raise ValueError("invalid record prefix")
            record = json.loads(line[len(prefix) + 1:])
            if not isinstance(record, dict):
                raise ValueError("expected a result object")
            if (type(record.get("version")) is not int or record["version"] != 1
                    or record.get("binary") != binary
                    or type(record.get("rank")) is not int
                    or type(record.get("size_bytes")) is not int
                    or record.get("case") not in REQUIRED[binary]
                    or record.get("status") not in ("PASS", "FAIL", "SKIP")):
                raise ValueError("invalid result fields")
        except (ValueError, TypeError) as error:
            errors.append(f"Malformed Device API result: {error}: {line}")
            continue
        key = (record["rank"], record["case"], record["size_bytes"])
        observed[key] += 1
        if record["status"] != "PASS":
            errors.append(f"Result {key}: {record['status']}")
    expected = Counter((rank, case, size) for rank in range(ranks)
                       for case in REQUIRED[binary] for size in sizes)
    for key in sorted(expected.keys() | observed.keys()):
        if observed[key] != expected[key]:
            errors.append(f"Result {key}: count {observed[key]}, expected {expected[key]}")
    observed_sizes = Counter(map(int, re.findall(r"# Size = (\d+) bytes", text)))
    if observed_sizes != Counter(sizes):
        errors.append(f"Size sweep {dict(observed_sizes)}, expected {sizes}")
    if "unified" in binary:
        if text.count("=== Final Result: ALL PASS ===") != 1:
            errors.append("Expected one unified IR ALL PASS summary")
    else:
        overall = Counter(map(int, re.findall(r"\[rank (\d+)\]\s+=== Overall: PASS ===", text)))
        if overall != Counter(range(ranks)):
            errors.append("Expected one Overall PASS from every rank")
    return errors


def main():
    try:
        binary, ranks, sizes = invocation(sys.argv[1:])
        rounds = int(os.environ.get("FLAGCX_CI_DEVICE_API_ROUNDS", "1"))
        if rounds <= 0:
            raise ValueError("FLAGCX_CI_DEVICE_API_ROUNDS must be positive")
    except (ValueError, IndexError) as error:
        print(error, file=sys.stderr)
        return 2
    root = Path(os.environ.get("FLAGCX_CI_LOG_DIR", tempfile.gettempdir()))
    root.mkdir(parents=True, exist_ok=True)
    label = re.sub(r"[^A-Za-z0-9_.-]", "_", os.environ.get("FLAGCX_CI_MPI_LABEL", binary))
    output = Path(tempfile.mkdtemp(prefix=f"checked-{label}.", dir=root))
    runner = Path(__file__).with_name("run_mpi_with_timeout.sh")
    # Every acceptance round must pass; do not combine partial retry output
    # into apparently complete per-rank coverage.
    env = {**os.environ, "FLAGCX_CI_TEST_MAX_ATTEMPTS": "1"}
    for round_id in range(1, rounds + 1):
        print(f"Device API {binary}: round {round_id}/{rounds}; logs: {output}", flush=True)
        log = output / f"round-{round_id}.log"
        with log.open("w") as transcript:
            process = subprocess.Popen([str(runner), *sys.argv[1:]], env=env,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, errors="replace")
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    transcript.write(line)
                status = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                return 130
            finally:
                process.stdout.close()
        errors = check_results(log.read_text(), binary, ranks, sizes)
        if status:
            errors.insert(0, f"MPI exited with status {status}")
        (output / f"round-{round_id}.json").write_text(json.dumps({
            "binary": binary, "ranks": ranks, "sizes": sizes, "round": round_id,
            "exit": status, "errors": errors,
        }, indent=2) + "\n")
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return status if 0 < status < 256 else 1
    return 0


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
