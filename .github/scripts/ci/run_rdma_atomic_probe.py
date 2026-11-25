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
import sys
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

SNAPSHOT_TIMEOUT = 0.25
SNAPSHOT_REAP_TIMEOUT = 0.1
# Three isolated snapshots: before TERM, after TERM, after KILL.
CLEANUP_ALLOWANCE = 1.25 + 3 * (SNAPSHOT_TIMEOUT + SNAPSHOT_REAP_TIMEOUT)


class StartupFailure(RuntimeError):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status


def discover_ports(root=Path("/sys/class/infiniband"), progress=None):
    """Host-only discovery: never load MACA, MPI or a verbs provider."""
    result = dict(ports=[], entries=[], errors=[])
    try:
        devices = sorted(root.iterdir())
    except OSError as error:
        result["errors"].append(str(error))
        return result
    for device in devices:
        try:
            ports = sorted((device / "ports").iterdir())
        except OSError as error:
            result["errors"].append(str(error))
            continue
        for port in ports:
            if not port.name.isdigit():
                continue
            try:
                state = (port / "state").read_text().strip()
                result["entries"].append(dict(hca=device.name, port=int(port.name), state=state))
                if progress:
                    progress(result["entries"][-1])
                if state.split(":", 1)[0].strip() == "4":
                    result["ports"].append(f"{device.name}:{int(port.name)}")
            except OSError as error:
                result["errors"].append(str(error))
    result["ports"] = sorted(set(result["ports"]))
    return result


def host_inventory(directory, report):
    # Even sysfs can stall in a driver read. Keep that risk outside the
    # supervisor, and retain partial port records and process snapshots.
    log = directory / "host-inventory.log"
    outcome = run_child([sys.executable, str(Path(__file__)), "--host-inventory"], log, 5)
    records = parse_log(log)
    report["host_inventory_run"] = dict(outcome=outcome, records=records)
    inventories = [r for r in records if r.get("event") == "host_inventory"]
    if inventories:
        report["host_inventory"] = inventories[-1]
    save_report(directory, report)
    if not outcome["reaped"]:
        raise StartupFailure("WORKER_STUCK", "host inventory worker was not reaped")
    if outcome["timed_out"]:
        raise StartupFailure("STARTUP_TIMEOUT", "host inventory timed out; see host-inventory.log")
    if outcome["returncode"] != 0 or not inventories:
        raise StartupFailure("MISSING_INVENTORY", "host inventory failed; see host-inventory.log")
    return inventories[-1]


def capture_diagnostics(log, phase, processes):
    directory = log.parent / (log.name + ".diagnostics") / phase
    command = [sys.executable, str(Path(__file__).with_name("rdma_probe_snapshot.py")),
               "--output", str(directory)]
    for pid, identity in sorted(processes.items()):
        command += ["--process", f"{pid}:{identity}"]
    record = dict(phase=phase, directory=str(directory), processes=processes,
                  timed_out=False, reaped=True)
    child = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.pop("FLAGCX_CI_RDMA_PROBE_TOKEN", None)
        with (directory / "collector.log").open("w") as output:
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True)
            try:
                child.wait(timeout=SNAPSHOT_TIMEOUT)
            except subprocess.TimeoutExpired:
                record["timed_out"] = True
                child.kill()
                try:
                    child.wait(timeout=SNAPSHOT_REAP_TIMEOUT)
                except subprocess.TimeoutExpired:
                    record["reaped"] = False
            record.update(pid=child.pid, returncode=child.returncode)
    except OSError as error:
        record["error"] = str(error)
    finally:
        if child is not None:
            returncode = child.poll()
            record.update(pid=child.pid, returncode=returncode, reaped=returncode is not None)
    try:
        (directory / "collector.json").write_text(json.dumps(record, indent=2) + "\n")
    except OSError as error:
        record["save_error"] = str(error)
    return record


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
            result["pid"] = child.pid
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
        except InterruptedError:
            raise
        except OSError as error:
            result["launch_error"] = str(error)
            output.write(str(error) + "\n")
        finally:
            if child:
                # The marker also identifies reparented MPI workers that have
                # escaped mpirun's process group or session.
                descendants.update(process_tree(child.pid if child.poll() is None else -1, token))
                need_diagnostics = (result["timed_out"] or child.poll() != 0 or any(
                    same_process(pid, identity) for pid, identity in descendants.items()))
                diagnostics = result["diagnostics"] = []
                def snapshot(phase):
                    if need_diagnostics and all(d["reaped"] for d in diagnostics):
                        descendants.update(process_tree(
                            child.pid if child.poll() is None else -1, token))
                        diagnostics.append(capture_diagnostics(log, phase, descendants))
                snapshot("before-term")
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
                    snapshot("after-term" if sig == signal.SIGTERM else "after-kill")
                result["returncode"] = child.returncode
                result["surviving_workers"] = [pid for pid, identity in descendants.items()
                                               if same_process(pid, identity)]
                result["reaped"] = (child.poll() is not None and not result["surviving_workers"]
                                    and all(d["reaped"] for d in diagnostics))
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
    if not outcome.get("reaped", True):
        return "WORKER_STUCK"
    if outcome["timed_out"]:
        return "TIMEOUT"
    if outcome.get("launch_error"):
        return "LAUNCH_ERROR"
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
        "sdk": args.sdk_inventory,
        "native": args.native_binary,
        "adaptor": args.adaptor_binary,
        "barrier": args.barrier,
    }
    builds = report["builds"] = {
        name: dict(binary=str(binary), status="PENDING", available=False)
        for name, binary in components.items()}
    build_dir = directory / "build"
    build_dir.mkdir(exist_ok=True)
    for name, binary in components.items():
        entry = builds[name]
        write_build_report(directory, builds)
        if args.build_root:
            suite = "device_api" if name == "barrier" else "rdma_atomic_probe"
            target = (str(binary.resolve()) if name == "barrier" else
                      "sdk-inventory" if name == "sdk" else name)
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
        if not entry.get("reaped", True):
            entry["status"] = "WORKER_STUCK"
            for pending in builds.values():
                if pending["status"] == "PENDING":
                    pending.update(status="BLOCKED", reason="previous build worker was not reaped")
            write_build_report(directory, builds)
            raise StartupFailure("WORKER_STUCK", f"{name} build worker was not reaped")
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
        lines += ["", "Setup status: " + report.get("setup_status", "SETUP_ERROR"),
                  "Setup error: " + report["setup_error"]]
    if report.get("gpu_inventory_attempts"):
        lines += ["", "## GPU startup controls", "",
                  "| Worker | Result | Main entered | Last stage | Log |",
                  "|---|---|---|---|---|"]
        for attempt in report["gpu_inventory_attempts"]:
            lines.append(f'| {attempt["worker"]} | {attempt["status"]} '
                         f'| {attempt.get("main_entered")} | {attempt.get("last_stage", "")} '
                         f'| {attempt["log"]} |')
    if report.get("builds"):
        lines += ["", "## Worker builds", "",
                  "| Worker | Status | Binary |", "|---|---|---|"]
        for name, build in report["builds"].items():
            lines.append(f'| {name} | {build["status"]} | {build["binary"]} |')
    temporary = directory / "summary.md.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(directory / "summary.md")


def collect_provenance(directory, phase="host"):
    commands = ((["uname", "-a"], ["lspci", "-nnvv"], ["modinfo", "bnxt_re"])
                if phase == "host" else
                (["ibv_devinfo", "-v"], ["rdma", "link", "show"], ["mx-smi"]))
    outcomes = []
    for index, command in enumerate(commands):
        outcomes.append(run_child(command, directory / f"metadata-{phase}-{index}.log", 5))
        if not outcomes[-1]["reaped"]:
            break
    return outcomes


def gpu_inventory(args, directory, report, builds):
    """Use real counts, retaining failed controls; never reuse a live worker."""
    attempts = report["gpu_inventory_attempts"] = []
    for name in ("sdk", "native", "adaptor"):
        if not builds[name]["available"]:
            continue
        log = directory / f"inventory-{name}.log"
        attempt = dict(worker=name, log=str(log), status="RUNNING")
        attempts.append(attempt)
        save_report(directory, report)
        env = case_environment(dict(layer="allocation", hca="auto", ro=0, depth=16, qps=1))
        outcome = run_child([builds[name]["binary"], "--count" if name == "sdk" else "--inventory"],
                            log, args.inventory_timeout, env)
        records = parse_log(log)
        stages = [r for r in records if r.get("event") == "stage"]
        attempt.update(outcome=outcome, records=records,
                       main_entered="PROBE_BOOT main_enter" in log.read_text(errors="replace"),
                       last_stage=(stages[-1].get("stage", "") + "/" + stages[-1].get("phase", "")
                                   if stages else ""))
        inventories = [r for r in records if r.get("event") == "gpu_inventory"]
        if not outcome["reaped"]:
            attempt["status"] = "WORKER_STUCK"
        elif outcome["timed_out"]:
            attempt["status"] = "STARTUP_TIMEOUT"
        elif outcome.get("launch_error"):
            attempt["status"] = "LAUNCH_ERROR"
        elif inventories and inventories[-1].get("gpu_count_code") != 0:
            attempt["status"] = "SDK_ERROR"
        elif (outcome["returncode"] == 0 and inventories
              and inventories[-1].get("gpu_count_code") == 0
              and type(inventories[-1].get("gpu_count")) is int
              and inventories[-1]["gpu_count"] >= 0):
            attempt["status"] = "PASS"
        else:
            attempt["status"] = "MISSING_INVENTORY" if outcome["returncode"] == 0 else "STARTUP_ERROR"
        save_report(directory, report)
        print(f'GPU inventory {name}: {attempt["status"]} last_stage={attempt["last_stage"]}', flush=True)
        if attempt["status"] == "WORKER_STUCK":
            raise StartupFailure("WORKER_STUCK", f"{name} GPU inventory was not reaped; see {log.name}")
        if attempt["status"] == "PASS":
            return inventories[-1]
    status = attempts[-1]["status"] if attempts else "BUILD_ERROR"
    raise StartupFailure(status, "no GPU inventory control succeeded; see startup records and diagnostics")


def optional_metadata(args, directory, report, builds):
    # Run version/provider queries after the matrix, so optional information
    # cannot suppress allocation coverage. Each query is a fresh process.
    report["device_metadata"] = collect_provenance(directory, "device")
    for outcome in report["device_metadata"]:
        if not outcome["reaped"]:
            raise StartupFailure("WORKER_STUCK", "optional device metadata worker was not reaped")
    report["sdk_versions"] = []
    if not builds["sdk"]["available"]:
        report["sdk_versions_skipped"] = "SDK inventory control is unavailable; see build.json"
        return
    for option in ("--runtime-version", "--driver-version"):
        log = directory / (option[2:] + ".log")
        outcome = run_child([str(args.sdk_inventory), option], log, 5)
        report["sdk_versions"].append(dict(outcome=outcome, records=parse_log(log)))
        save_report(directory, report)
        if not outcome["reaped"]:
            raise StartupFailure("WORKER_STUCK", f"{option} worker was not reaped")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-binary", "--binary", dest="native_binary", type=Path, required=True)
    parser.add_argument("--adaptor-binary", type=Path, required=True)
    parser.add_argument("--sdk-inventory", type=Path, required=True)
    parser.add_argument("--barrier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-root", type=Path)
    parser.add_argument("--make-arg", action="append", default=[])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--build-timeout", type=float, default=300)
    parser.add_argument("--inventory-timeout", type=float, default=15)
    parser.add_argument("--case-timeout", type=float,
                        default=float(os.environ.get("FLAGCX_CI_RDMA_PROBE_CASE_TIMEOUT", "10")))
    parser.add_argument("--suite-timeout", type=float,
                        default=float(os.environ.get("FLAGCX_CI_RDMA_PROBE_SUITE_TIMEOUT", "6600")))
    parser.add_argument("--barrier-timeout", type=float, default=180)
    parser.add_argument("--mpirun", default=os.environ.get("FLAGCX_CI_MPIRUN", "mpirun"))
    args = parser.parse_args()
    if any(not math.isfinite(value) or value <= 0 for value in
           (args.case_timeout, args.suite_timeout, args.barrier_timeout,
            args.build_timeout, args.inventory_timeout)):
        parser.error("timeouts must be finite and positive")
    if args.jobs < 1:
        parser.error("jobs must be positive")
    directory = args.output.resolve()
    previous_umask = os.umask(0o022)
    directory.mkdir(parents=True, exist_ok=True)
    page_size = os.sysconf("SC_PAGE_SIZE")
    report = dict(page_size=page_size, cases=build_matrix([], page_size), metadata=[],
                  workers=dict(sdk=str(args.sdk_inventory), native=str(args.native_binary),
                               adaptor=str(args.adaptor_binary), barrier=str(args.barrier)),
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
        report["metadata"] = collect_provenance(directory)
        save_report(directory, report)
        if any(not outcome["reaped"] for outcome in report["metadata"]):
            raise StartupFailure("WORKER_STUCK", "host metadata worker was not reaped")
        report["host_inventory"] = host_inventory(directory, report)
        ports = report["host_inventory"]["ports"]
        report["cases"] = build_matrix(ports, page_size)
        (directory / "host-inventory.json").write_text(
            json.dumps(report["host_inventory"], indent=2) + "\n")
        save_report(directory, report)
        builds = build_workers(args, directory, report)
        sdk = args.native_binary.parent.parent / "sdk"
        if sdk.is_dir():
            shutil.copytree(sdk, directory / "sdk", dirs_exist_ok=True)
        inventory = report["inventory"] = gpu_inventory(args, directory, report, builds)
        # Reserve termination/reap time per process group and enough margin to
        # finish/save the entire plan even if every worker hits its deadline.
        required = sum((args.barrier_timeout if c["layer"] == "barrier" else args.case_timeout) + CLEANUP_ALLOWANCE
                       for c in report["cases"] if case_worker(c, builds) is not None)
        report["worst_case_seconds"] = required
        report["optional_metadata_budget_seconds"] = 5 * (5 + CLEANUP_ALLOWANCE)
        if (time.monotonic() - started + required + report["optional_metadata_budget_seconds"]
                + 30 > args.suite_timeout):
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
        if driver_stuck:
            report["optional_metadata_skipped"] = "a matrix worker was not reaped"
        else:
            optional_metadata(args, directory, report, builds)
    except (OSError, RuntimeError, ValueError, InterruptedError) as error:
        report["setup_error"] = str(error)
        report["setup_status"] = getattr(error, "status", "SETUP_ERROR")
        print(f'{report["setup_status"]}: {error}', flush=True)
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
        os.umask(previous_umask)
    passed = (report["cases"] and not report.get("setup_error")
              and not report.get("host_inventory", {}).get("errors")
              and all(c["status"] == "PASS" for c in report["cases"])
              and all(b["status"] == "READY" for b in report.get("builds", {}).values())
              and all(a["status"] == "PASS" for a in report.get("gpu_inventory_attempts", [])))
    return 0 if passed else 1


if __name__ == "__main__":
    if sys.argv[1:] == ["--host-inventory"]:
        print("PROBE_BOOT host_inventory_enter", flush=True)
        ports = discover_ports(progress=lambda entry: print(
            "PROBE_JSON " + json.dumps(dict(event="host_port", **entry)), flush=True))
        print("PROBE_JSON " + json.dumps(dict(event="host_inventory", **ports)), flush=True)
    else:
        raise SystemExit(main())
