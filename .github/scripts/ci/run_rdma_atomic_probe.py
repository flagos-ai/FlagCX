#!/usr/bin/env python3
"""Run the MetaX GPU-only RDMA allocation matrix with isolated MPI workers."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import time
import uuid


MODES = (
    "mcMalloc", "mcDeviceMallocDefault", "mcDeviceMallocFinegrained",
    "mcMallocSignalMemory", "mcDeviceMallocWriteCoherence",
    "mcDeviceMallocMapPcieDefault", "mcDeviceMallocMapPcieCoherence",
    "mcDeviceMallocFixedMemDefault", "mcDeviceMallocFixedMemCoherence",
    "gdrMemAlloc",
)
TERMINAL_ERRORS = {
    "BUILD_ERROR",
    "ALLOC_ERROR", "SDK_UNAVAILABLE", "INIT_ERROR", "COPY_ERROR",
    "REG_ERROR", "POST_ERROR", "CQ_ERROR", "VERIFY_ERROR", "CLEANUP_ERROR",
    "SDK_ERROR", "VERBS_ERROR", "ADAPTOR_ERROR", "CONNECT_ERROR",
    "CAPABILITY_ERROR", "SETUP_ERROR", "INTERNAL_ERROR", "TIMEOUT",
}


def process_tree(root, token=None):
    """Linux MPI workers may start new sessions; remember their PID identity."""
    processes = {}
    marked = set()
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return {}
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            processes[int(entry.name)] = (int(fields[1]), fields[19], fields[0])
            if token and (f"FLAGCX_CI_RDMA_PROBE_TOKEN={token}".encode() in
                          (entry / "environ").read_bytes().split(b"\0")):
                marked.add(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    descendants = {root} | marked
    while True:
        discovered = {pid for pid, (parent, _, _) in processes.items()
                      if parent in descendants}
        if discovered <= descendants:
            break
        descendants |= discovered
    return {pid: processes[pid][1] for pid in descendants if pid in processes}


def same_process(pid, identity):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19] == identity and fields[0] != "Z"
    except (OSError, IndexError):
        return False


def build_matrix(ports, page_size):
    """Every allocation gets both sizes and both transports before tuning."""
    cases = []

    def add(layer, mode, size=page_size, operation="fetch_add", **settings):
        case = dict(layer=layer, mode=mode, allocation=MODES[mode], bytes=size,
                    operation=operation, hca=ports[0] if ports else "unavailable",
                    offset=0, direction=0, qps=1, depth=16, ro=0,
                    registration="mr", ranks=2, iterations=8, status="PENDING")
        case.update(settings)
        case["id"] = f"{len(cases):04d}-{layer}-{MODES[mode]}-{operation}"
        cases.append(case)

    for mode in range(len(MODES)):
        for size in (8, page_size):
            add("allocation", mode, size, "alloc")
    for layer in ("native", "adaptor"):
        for mode in range(len(MODES)):
            for size in (8, page_size):
                for operation in ("write", "read", "fetch_add"):
                    for direction in (0, 1):
                        add(layer, mode, size, operation, direction=direction)
            # Isolate per-HCA atomic compatibility for every allocation mode.
            for port in ports[1:]:
                for direction in (0, 1):
                    add(layer, mode, hca=port, direction=direction)
    # Change one parameter at a time for the ordinary, IPC and GDR policies.
    for mode in (0, 2, 9):
        for layer in ("native", "adaptor"):
            for setting in ({"offset": 8}, {"depth": 1}, {"ro": 1}, {"qps": 4}):
                add(layer, mode, **setting)
            if layer == "native":
                add(layer, mode, registration="iova2")
    for layer in ("native", "adaptor"):
        for mode in range(len(MODES)):
            add(layer, mode, hca="auto")
            for direction in (0, 1):
                add(layer, mode, hca="auto", ranks=4, qps=4, direction=direction)
    # Existing kernel acceptance path, with GPU signals and no allocator override.
    for hca in (ports[0] if ports else "unavailable", "auto"):
        add("barrier", 9, ranks=4, hca=hca)
    return cases


def run_child(command, log, timeout, env=None, grace=1):
    """Bound wait/reap as well as execution; kill the whole local MPI group."""
    started = time.monotonic()
    result = dict(command=[str(x) for x in command], log=str(log),
                  timed_out=False, reaped=True, returncode=None)
    child = None
    descendants = {}
    token = uuid.uuid4().hex
    env = dict(os.environ if env is None else env)
    env["FLAGCX_CI_RDMA_PROBE_TOKEN"] = token
    with log.open("w") as output:
        try:
            child = subprocess.Popen(command, stdout=output,
                                     stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)
            end = time.monotonic() + timeout
            while child.poll() is None:
                descendants.update(process_tree(child.pid))
                remaining = end - time.monotonic()
                if remaining <= 0:
                    result["timed_out"] = True
                    break
                try:
                    child.wait(timeout=min(0.1, remaining))
                except subprocess.TimeoutExpired:
                    pass
        except OSError as error:
            result["launch_error"] = str(error)
            output.write(str(error) + "\n")
        finally:
            if child:
                # The marker also identifies reparented MPI workers that have
                # escaped mpirun's process group or session.
                descendants.update(process_tree(child.pid if child.poll() is None else -1, token))
                # Also remove descendants if mpirun exited ahead of its workers.
                for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 0.25)):
                    cleanup_deadline = time.monotonic() + wait
                    for pid, identity in descendants.items():
                        if same_process(pid, identity):
                            try:
                                os.kill(pid, sig)
                            except ProcessLookupError:
                                pass
                    group_alive = child.poll() is None
                    for pid, identity in descendants.items():
                        if same_process(pid, identity):
                            try:
                                group_alive |= os.getpgid(pid) == child.pid
                            except ProcessLookupError:
                                pass
                    if group_alive:
                        try:
                            os.killpg(child.pid, sig)
                        except ProcessLookupError:
                            pass
                    try:
                        child.wait(timeout=wait)
                    except subprocess.TimeoutExpired:
                        pass
                    while time.monotonic() < cleanup_deadline and any(
                            same_process(pid, identity) for pid, identity in descendants.items()):
                        time.sleep(0.01)
                result["returncode"] = child.returncode
                result["surviving_workers"] = [pid for pid, identity in descendants.items()
                                               if same_process(pid, identity)]
                result["reaped"] = child.poll() is not None and not result["surviving_workers"]
    result["elapsed_s"] = round(time.monotonic() - started, 3)
    return result


def parse_log(path):
    records = []
    ranks = {}
    with path.open(errors="replace") as source:
        for line in source:
            position = line.find("PROBE_JSON ")
            if position < 0:
                if "Got completion" in line:
                    match = re.search(r"status=(\d+).*vendor err (\d+) qp_num=(\d+) wr_id=(\d+)", line)
                    if match:
                        record = dict(event="provider_completion_error", wc_status=int(match[1]),
                                      vendor_err=int(match[2]), qpn=int(match[3]), wr_id=int(match[4]))
                        owner = re.search(r"\S+:(\d+):\d+ \[", line)
                        if owner:
                            record["pid"] = int(owner[1])
                            record["rank"] = ranks.get(int(owner[1]))
                        records.append(record)
                continue
            try:
                record = json.loads(line[position + len("PROBE_JSON "):])
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
                if "pid" in record and "rank" in record:
                    ranks[record["pid"]] = record["rank"]
    return records


def classify(case, outcome, records, text=""):
    if outcome["timed_out"]:
        return "TIMEOUT"
    if outcome.get("launch_error"):
        return "LAUNCH_ERROR"
    if not outcome.get("reaped", True):
        return "WORKER_STUCK"
    failures = [r["status"] for r in records if r.get("status") in TERMINAL_ERRORS]
    if failures:
        return failures[0]
    if any(r.get("event") == "provider_completion_error" and r.get("wc_status")
           for r in records):
        return "CQ_ERROR"
    if case["layer"] == "barrier":
        if outcome["returncode"] == 0 and "=== Final Result: ALL PASS ===" in text:
            return "PASS"
        if "Got completion" in text:
            return "CQ_ERROR"
        return "PROCESS_ERROR" if outcome["returncode"] else "MISSING_RESULT"
    ranks = {r.get("rank") for r in records
             if r.get("event") == "result" and r.get("status") == "PASS"}
    if outcome["returncode"] == 0 and ranks == set(range(case["ranks"])):
        return "PASS"
    return "PROCESS_ERROR" if outcome["returncode"] else "MISSING_RESULT"


def case_environment(case):
    env = os.environ.copy()
    env.update(FLAGCX_SIGNAL_HOST_ENABLE="0", FLAGCX_VMM_ENABLE="0",
               FLAGCX_USE_HETERO_COMM="1", FLAGCX_IB_DISABLE="0", FLAGCX_P2P_DISABLE="0",
               FLAGCX_DMABUF_ENABLE="0", FLAGCX_DEBUG="TRACE",
               FLAGCX_DEBUG_SUBSYS="INIT,NET,ENV",
               FLAGCX_IB_PCI_RELAXED_ORDERING=str(case["ro"]),
               FLAGCX_IB_RD_ATOMIC_DEPTH=str(case["depth"]),
               FLAGCX_IB_QPS_PER_CONNECTION=str(case["qps"]))
    if case["layer"] == "barrier":
        env["FLAGCX_CI_RDMA_PROBE_BARRIER_ONLY"] = "1"
    else:
        env.pop("FLAGCX_CI_RDMA_PROBE_BARRIER_ONLY", None)
    # Ambient test selectors must not silently change this experiment.
    for key in ("FLAGCX_DEVICE_ONE_SIDED_FORCE_NET", "FLAGCX_NET_ADAPTOR_PLUGIN",
                "FLAGCX_DEVICE_ADAPTOR_PLUGIN", "FLAGCX_CI_UNIFIED_IR_SKIP_S16",
                "FLAGCX_CI_UNIFIED_IR_SKIP_S16_REASON", "UCX_HOME"):
        env.pop(key, None)
    if case["hca"] == "auto":
        env.pop("FLAGCX_IB_HCA", None)
    elif case["hca"] != "unavailable":
        env["FLAGCX_IB_HCA"] = "=" + case["hca"]
    return env


def case_command(case, binary, barrier, mpirun):
    command = mpirun + ["--allow-run-as-root", "--oversubscribe", "--host", "localhost"]
    common = ["-x", "FLAGCX_SIGNAL_HOST_ENABLE=0", "-x", "FLAGCX_VMM_ENABLE=0",
              "-x", "FLAGCX_USE_HETERO_COMM=1", "-x", "LD_LIBRARY_PATH",
              "-x", "FLAGCX_CI_RDMA_PROBE_TOKEN"]
    if case["layer"] == "barrier":
        arguments = [str(barrier), "-b", "1K", "-e", "1K", "-f", "2", "-R", "1"]
        common += ["-x", "FLAGCX_CI_RDMA_PROBE_BARRIER_ONLY=1"]
        return command + ["-np", "2"] + common + [
            "-x", "FLAGCX_HOSTID=node0", "-x", "NCCL_HOSTID=node0",
        ] + arguments + [":", "-np", "2"] + common + [
            "-x", "FLAGCX_HOSTID=node1", "-x", "NCCL_HOSTID=node1",
        ] + arguments
    command += ["-np", str(case["ranks"])] + common + [str(binary)]
    options = dict(layer=case["layer"], operation=case["operation"], mode=case["mode"],
                   bytes=case["bytes"], offset=case["offset"], hca=case["hca"],
                   qps=case["qps"], depth=case["depth"], ro=case["ro"],
                   direction=case["direction"], registration=case["registration"],
                   iterations=case["iterations"], **{"poll-timeout": 3})
    for key, value in options.items():
        command += ["--" + key, str(value)]
    return command


def write_build_report(directory, builds):
    temporary = directory / "build.json.tmp"
    temporary.write_text(json.dumps(builds, indent=2) + "\n")
    temporary.replace(directory / "build.json")


def build_workers(args, directory, report):
    """A failed worker must not prevent another worker from being built/run."""
    components = {
        "native": args.native_binary,
        "adaptor": args.adaptor_binary,
        "barrier": args.barrier,
    }
    builds = report["builds"] = {}
    build_dir = directory / "build"
    build_dir.mkdir(exist_ok=True)
    for name, binary in components.items():
        entry = builds[name] = dict(binary=str(binary), status="PENDING", available=False)
        write_build_report(directory, builds)
        if args.build_root:
            suite = "device_api" if name == "barrier" else "rdma_atomic_probe"
            target = str(binary.resolve()) if name == "barrier" else name
            command = ["make", "-C", str(args.build_root / "test/unittest" / suite),
                       "--jobs=" + str(args.jobs), target] + args.make_arg
            entry["status"] = "BUILDING"
            write_build_report(directory, builds)
            outcome = run_child(command, build_dir / f"{name}.log", args.build_timeout)
            entry.update(outcome)
            entry["available"] = (outcome["returncode"] == 0 and not outcome["timed_out"]
                                  and outcome["reaped"] and os.access(binary, os.X_OK))
        else:
            entry["available"] = os.access(binary, os.X_OK)
            (build_dir / f"{name}.log").write_text(f"Using prebuilt binary: {binary}\n")
        entry["status"] = "READY" if entry["available"] else "BUILD_ERROR"
        write_build_report(directory, builds)
        save_report(directory, report)
        print(f'{name} worker: {entry["status"]}', flush=True)
    return builds


def case_worker(case, builds):
    if case["layer"] == "allocation":
        return next((Path(builds[name]["binary"]) for name in ("native", "adaptor")
                     if builds[name]["available"]), None)
    name = case["layer"]
    return Path(builds[name]["binary"]) if builds[name]["available"] else None


def save_report(directory, report):
    temporary = directory / "results.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(directory / "results.json")
    lines = ["# MetaX GPU RDMA allocation probe", "",
             "Remote signals are GPU memory. No HOST SIGNAL mode is used.",
             "Allocation success, transport success and IPC visibility are different results.",
             "", "| Allocation | Alloc 8 B | Alloc page | Native W/R/FAdd | Adaptor W/R/FAdd |",
             "|---|---|---|---|---|"]
    for mode, name in enumerate(MODES):
        cells = []
        for layer, size in (("allocation", 8), ("allocation", report["page_size"]),
                            ("native", None), ("adaptor", None)):
            chosen = [c for c in report["cases"] if c["mode"] == mode and c["layer"] == layer
                      and (size is None or c["bytes"] == size)]
            if size is not None:
                cells.append(", ".join(sorted({c["status"] for c in chosen})))
            else:
                operations = []
                for operation in ("write", "read", "fetch_add"):
                    counts = Counter(c["status"] for c in chosen if c["operation"] == operation)
                    operations.append(operation + ": " + ", ".join(
                        f"{status}={count}" for status, count in sorted(counts.items())))
                cells.append("<br>".join(operations))
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    lines += ["", "## Every planned case", "",
              "| Case | Layer | Allocation | Bytes/offset | HCA | Dir | Ranks/QPs | Depth/RO | Registration | Result | Evidence |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for case in report["cases"]:
        evidence = case.get("reason", "")
        errors = [r for r in case.get("records", []) if r.get("event") == "error"]
        if errors:
            first = errors[0]
            evidence = f'rank={first.get("rank")} {first.get("stage")} code={first.get("code")}: {first.get("detail")}'
        completions = [r for r in case.get("records", []) if r.get("event") == "provider_completion_error"]
        if completions:
            first = completions[0]
            evidence += f' WC status={first["wc_status"]} vendor={first["vendor_err"]} qpn={first["qpn"]}'
        evidence = evidence.replace("|", "/").replace("\n", " ")
        lines.append(
            f'| {case["id"]} | {case["layer"]}/{case["operation"]} | {case["allocation"]} '
            f'| {case["bytes"]}/{case["offset"]} | {case["hca"]} | {case["direction"]} '
            f'| {case["ranks"]}/{case["qps"]} | {case["depth"]}/{case["ro"]} '
            f'| {case["registration"]} | {case["status"]} | {evidence} |')
    lines += ["", "Counts: " + json.dumps(dict(Counter(c["status"] for c in report["cases"])), sort_keys=True),
              "", "All raw records, original return codes and launch commands are in results.json and case logs.",
              "A mixed-success diagnostic run fails CI; BLOCKED/UNAVAILABLE are never counted as PASS."]
    if report.get("setup_error"):
        lines += ["", "Setup error: " + report["setup_error"]]
    if report.get("builds"):
        lines += ["", "## Worker builds", "",
                  "| Worker | Status | Binary |", "|---|---|---|"]
        for name, build in report["builds"].items():
            lines.append(f'| {name} | {build["status"]} | {build["binary"]} |')
    temporary = directory / "summary.md.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(directory / "summary.md")


def collect_provenance(directory):
    commands = (
        ["uname", "-a"], ["ibv_devinfo", "-v"], ["rdma", "link", "show"],
        ["mx-smi"], ["lspci", "-nnvv"], ["modinfo", "bnxt_re"],
    )
    outcomes = []
    for index, command in enumerate(commands):
        outcomes.append(run_child(command, directory / f"metadata-{index}.log", 5))
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-binary", "--binary", dest="native_binary", type=Path, required=True)
    parser.add_argument("--adaptor-binary", type=Path, required=True)
    parser.add_argument("--barrier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-root", type=Path)
    parser.add_argument("--make-arg", action="append", default=[])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--build-timeout", type=float, default=300)
    parser.add_argument("--case-timeout", type=float,
                        default=float(os.environ.get("FLAGCX_CI_RDMA_PROBE_CASE_TIMEOUT", "10")))
    parser.add_argument("--suite-timeout", type=float,
                        default=float(os.environ.get("FLAGCX_CI_RDMA_PROBE_SUITE_TIMEOUT", "6000")))
    parser.add_argument("--barrier-timeout", type=float, default=180)
    parser.add_argument("--mpirun", default=os.environ.get("FLAGCX_CI_MPIRUN", "mpirun"))
    args = parser.parse_args()
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.case_timeout, args.suite_timeout, args.barrier_timeout, args.build_timeout)):
        parser.error("timeouts must be finite and positive")
    if args.jobs < 1:
        parser.error("jobs must be positive")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    page_size = os.sysconf("SC_PAGE_SIZE")
    report = dict(page_size=page_size, cases=build_matrix([], page_size), metadata=[],
                  workers=dict(native=str(args.native_binary), adaptor=str(args.adaptor_binary)),
                  host_signal=False)
    report["environment"] = {key: os.environ.get(key) for key in (
        "MACA_PATH", "MACA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "MPI_HOME",
        "LD_LIBRARY_PATH", "FLAGCX_IB_HCA", "FLAGCX_IB_GID_INDEX", "FLAGCX_SOCKET_IFNAME")}
    save_report(directory, report)
    started = time.monotonic()
    def interrupted(signum, _frame):
        raise InterruptedError(f"received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        builds = build_workers(args, directory, report)
        sdk = args.native_binary.parent.parent / "sdk"
        if sdk.is_dir():
            shutil.copytree(sdk, directory / "sdk", dirs_exist_ok=True)
        inventory_worker = case_worker(dict(layer="allocation"), builds)
        if inventory_worker is None:
            for case in report["cases"]:
                case.update(status="BUILD_ERROR", reason="no usable allocation/inventory worker; see build.json")
            raise RuntimeError("both probe workers are unavailable; see build logs")
        inventory_log = directory / "inventory.log"
        inv_outcome = run_child([str(inventory_worker), "--inventory"], inventory_log, 15)
        report["inventory_run"] = inv_outcome
        inventories = [r for r in parse_log(inventory_log) if r.get("event") == "inventory"]
        if inv_outcome["returncode"] != 0 or inv_outcome["timed_out"] or not inventories:
            raise RuntimeError("inventory failed; see inventory.log")
        inventory = report["inventory"] = inventories[0]
        ports = sorted(set(inventory["ports"]))
        report["cases"] = build_matrix(ports, page_size)
        report["metadata"] = collect_provenance(directory)
        # Reserve termination/reap time per process group and enough margin to
        # finish/save the entire plan even if every worker hits its deadline.
        required = sum((args.barrier_timeout if c["layer"] == "barrier" else args.case_timeout) + 1.25
                       for c in report["cases"] if case_worker(c, builds) is not None)
        report["worst_case_seconds"] = required
        if time.monotonic() - started + required + 30 > args.suite_timeout:
            raise RuntimeError(f"matrix requires {required:.1f}s plus setup; increase suite budget or lower case timeout")
        save_report(directory, report)
        allocation_outcomes = {}
        driver_stuck = False
        mpirun = shlex.split(args.mpirun)
        if not mpirun:
            raise RuntimeError("empty MPI launcher")
        for case in report["cases"]:
            case_log = directory / (case["id"] + ".log")
            case["log"] = str(case_log)
            reason = None
            blocked_status = "BLOCKED"
            worker = case_worker(case, builds)
            dependency = allocation_outcomes.get((case["mode"], case["bytes"]))
            if worker is None:
                blocked_status = "BUILD_ERROR"
                reason = f'{case["layer"]} worker unavailable; see build.json'
            elif driver_stuck:
                reason = "previous worker could not be reaped after SIGKILL"
            elif time.monotonic() - started > args.suite_timeout:
                reason = "suite deadline reached"
            elif inventory["gpu_count_code"] != 0 or inventory["gpu_count"] < case["ranks"]:
                reason = f'{case["ranks"]} distinct GPUs required; inventory has {inventory["gpu_count"]}'
            elif case["layer"] != "allocation" and dependency and dependency["status"] != "PASS":
                reason = "allocation prerequisite failed: " + dependency["id"]
            elif case["layer"] != "allocation" and not ports:
                reason = "no active verbs HCA/port"
            if reason:
                case.update(status=blocked_status, reason=reason)
                case_log.write_text(reason + "\n")
            else:
                env = case_environment(case)
                command = case_command(case, worker, args.barrier, mpirun)
                timeout = args.barrier_timeout if case["layer"] == "barrier" else args.case_timeout
                outcome = run_child(command, case_log, timeout, env)
                records = parse_log(case_log)
                case.update(outcome=outcome, records=records,
                            status=classify(case, outcome, records,
                                            case_log.read_text(errors="replace") if case["layer"] == "barrier" else ""))
                driver_stuck |= not outcome["reaped"]
            if case["layer"] == "allocation":
                allocation_outcomes[(case["mode"], case["bytes"])] = case
            print(f'{case["id"]}: {case["status"]}', flush=True)
            save_report(directory, report)
    except (OSError, RuntimeError, ValueError, InterruptedError) as error:
        report["setup_error"] = str(error)
        for case in report["cases"]:
            if case["status"] == "PENDING":
                case.update(status="BLOCKED", reason=str(error))
    finally:
        report["elapsed_s"] = round(time.monotonic() - started, 3)
        save_report(directory, report)
        step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if step_summary:
            with open(step_summary, "a") as output:
                output.write((directory / "summary.md").read_text())
    return 0 if report["cases"] and all(c["status"] == "PASS" for c in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
