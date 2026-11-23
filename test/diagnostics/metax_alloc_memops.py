#!/usr/bin/env python3
"""Bounded, exhaustive MACA allocation/memops diagnostic; no FlagCX dependency."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import subprocess
import time


MODES = (
    "mcMalloc",
    "mcDeviceMallocDefault",
    "mcDeviceMallocFinegrained",
    "mcMallocSignalMemory",
    "mcDeviceMallocWriteCoherence",
    "mcDeviceMallocMapPcieDefault",
    "mcDeviceMallocMapPcieCoherence",
    "mcDeviceMallocFixedMemDefault",
    "mcDeviceMallocFixedMemCoherence",
)
OPERATIONS = ("write", "wait", "write_wait")
FEATURES = {
    "WRITE32": "mcStreamWriteValue32(s,p,uint32_t(1),mcStreamWriteValue_flags::MC_STREAM_WRITE_VALUE_DEFAULT)",
    "WRITE64": "mcStreamWriteValue64(s,p,uint64_t(1),mcStreamWriteValue_flags::MC_STREAM_WRITE_VALUE_DEFAULT)",
    "WAIT32": "mcStreamWaitValue32(s,p,uint32_t(1),mcStreamWaitValue_flags::MC_STREAM_WAIT_VALUE_GEQ)",
    "WAIT64": "mcStreamWaitValue64(s,p,uint64_t(1),mcStreamWaitValue_flags::MC_STREAM_WAIT_VALUE_GEQ)",
    "RUNTIME_VERSION": "mcRuntimeGetVersion(&v)",
    "DRIVER_VERSION": "mcDriverGetVersion(&v)",
}


def run_bounded(command, log, timeout, env=None):
    """Never drain inherited pipes or wait indefinitely after SIGKILL."""
    started = time.monotonic()
    result = {"command": [str(x) for x in command], "log": str(log)}
    with log.open("w") as output:
        try:
            child = subprocess.Popen(
                command, stdout=output, stderr=subprocess.STDOUT,
                start_new_session=True, env=env,
            )
        except OSError as error:
            output.write(str(error) + "\n")
            return dict(result, returncode=None, launch_error=str(error),
                        timed_out=False, reaped=True, elapsed_s=0)
        timed_out = False
        reaped = True
        try:
            child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Also kill descendants after a parent exits while leaving work.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                # A task stuck in a driver cannot be synchronously reaped.
                # Its state is reported; the supervisor still progresses.
                reaped = False
    return dict(result, returncode=child.returncode, timed_out=timed_out,
                reaped=reaped, elapsed_s=round(time.monotonic() - started, 3))


def matrix(page_size):
    return [dict(mode=mode, mode_index=index, size=size, width=width,
                 operation=operation, status="PENDING")
            for index, mode in enumerate(MODES)
            for size in (8, page_size)
            for width in (32, 64)
            for operation in OPERATIONS]


def save_report(directory, report):
    temporary = directory / "results.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(directory / "results.json")
    lines = ["# MetaX allocation / stream memops", "",
             "Diagnostic outcomes only; PASS does not establish IPC/RDMA visibility.",
             "", f"Native cases launched: {report.get('launched', 0)}/108", "",
             "| Allocation | Bytes | Bits | Write | Wait | Write → Wait |",
             "|---|---:|---:|---|---|---|"]
    cells = {}
    for case in report["cases"]:
        key = (case["mode"], case["size"], case["width"])
        label = case["status"]
        failures = [s for s in case.get("stages", []) if s["code"] != 0]
        if failures:
            label += f" ({failures[-1]['stage']}: {failures[-1]['error']})"
        cells.setdefault(key, []).append(label.replace("|", "/"))
    for (mode, size, width), statuses in cells.items():
        lines.append(f"| {mode} | {size} | {width} | " + " | ".join(statuses) + " |")
    lines += ["", "Outcome counts: " + json.dumps(
        dict(Counter(case["status"] for case in report["cases"])), sort_keys=True),
        "", "See results.json and per-case logs for raw API codes and cleanup outcomes."]
    temporary = directory / "summary.md.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(directory / "summary.md")


def parse_native(log):
    parsed = {"stages": [], "info": {}}
    for line in log.read_text(errors="replace").splitlines():
        fields = line.split("\t", 3)
        if fields[0] == "STAGE" and len(fields) == 4:
            try:
                parsed["stages"].append(dict(stage=fields[1], code=int(fields[2]),
                                             error=fields[3]))
            except ValueError:
                continue
        elif fields[0] == "INFO" and len(fields) >= 3:
            parsed["info"][fields[1]] = fields[2]
        elif fields[0] in ("RESULT", "OP_RESULT") and len(fields) >= 2:
            parsed[fields[0].lower()] = fields[1]
    return parsed


def collect_headers(sdk, directory):
    pattern = re.compile("mcExtMallocWithFlags|mcDeviceMalloc|mcMallocSignalMemory|"
                         "mcStream(?:Wait|Write)Value(?:32|64)")
    records, excerpts = [], []
    for header in sorted((sdk / "include/mcr").rglob("*.h")):
        data = header.read_bytes()
        lines = data.decode(errors="replace").splitlines()
        matches = [i for i, line in enumerate(lines) if pattern.search(line)]
        if not matches:
            continue
        records.append(dict(path=str(header), sha256=hashlib.sha256(data).hexdigest()))
        selected = set()
        for i in matches:
            selected.update(range(max(0, i - 16), min(len(lines), i + 17)))
        excerpts.append("\n" + str(header))
        excerpts.extend(f"{i + 1}: {lines[i]}" for i in sorted(selected))
    (directory / "sdk-header-excerpts.txt").write_text("\n".join(excerpts) + "\n")
    return records


def build_probe(args, directory, report):
    sdk = args.sdk.resolve()
    compiler = shlex.split(args.cxx)
    if not compiler or not shutil.which(compiler[0]):
        report["setup_error"] = "C++ compiler unavailable: " + args.cxx
        return None
    report["headers"] = collect_headers(sdk, directory)
    if shutil.which("mx-smi"):
        report["mx_smi"] = run_bounded(["mx-smi"], directory / "mx-smi.log", 5)
    else:
        report["mx_smi"] = {"status": "unavailable"}
    include = ["-I" + str(sdk / "include"), "-I" + str(sdk / "include/mcr")]
    libraries = ["-L" + str(sdk / name) for name in ("lib", "lib64")]
    libraries += ["-Wl,-rpath," + str(sdk / name) for name in ("lib", "lib64")]
    common = compiler + ["-std=c++17"] + include
    version = run_bounded(compiler + ["--version"], directory / "compiler.log", 5)
    report["compiler"] = version
    source = directory / "check.cc"
    executable = directory / "check"

    def check(name, body, link=True):
        source.write_text("#include <cstdint>\n#include <mcr/mc_runtime.h>\n"
                          "int main() { void *p=nullptr; mcStream_t s=nullptr; "
                          "int v=0; " + body + "; return 0; }\n")
        command = common + [str(source)]
        command += (["-o", str(executable)] + libraries + ["-lmcruntime"]
                    if link else ["-fsyntax-only"])
        outcome = run_bounded(command, directory / (name + ".log"), args.compile_timeout)
        report.setdefault("build_checks", {})[name] = outcome
        return outcome["returncode"] == 0 and not outcome["timed_out"]

    if not check("base", "mcMalloc(&p,8); mcFree(p); mcSetDevice(0)"):
        report["setup_error"] = "SDK/compiler/link setup failed; see base.log"
        return None
    available = [True]
    for mode in MODES[1:]:
        available.append(check(mode, "unsigned flag=" + mode + "; (void)flag", link=False))
    features = {name: check(name, expression) for name, expression in FEATURES.items()}
    report["features"] = features
    report["modes_available"] = dict(zip(MODES, available))
    config = ["#pragma once"]
    config += [f"#define PROBE_{name} {int(value)}" for name, value in features.items()]
    config += ["static mcError_t allocateMode(int mode,void **p,size_t n) {",
               'switch(mode) { case 0: printf("INFO\\tallocation_api\\tmcMalloc\\n"); return mcMalloc(p,n);']
    config += [f'case {i}: printf("INFO\\tallocation_flag\\t%u\\n",unsigned({mode})); '
               f"return mcExtMallocWithFlags(p,n,{mode});"
               for i, mode in enumerate(MODES[1:], 1) if available[i]]
    config += ["default: return mcErrorNotSupported; } }"]
    (directory / "metax_alloc_config.h").write_text("\n".join(config) + "\n")
    executable = directory / "metax_alloc_memops"
    source = Path(__file__).with_suffix(".cc")
    outcome = run_bounded(common + ["-I" + str(directory), str(source), "-o",
                                   str(executable)] + libraries + ["-lmcruntime"],
                          directory / "build.log", args.compile_timeout)
    report["build"] = outcome
    if outcome["returncode"] != 0 or outcome["timed_out"]:
        report["setup_error"] = "Native diagnostic build failed; see build.log"
        return None
    return executable


def unavailable_status(case, report):
    if not report["modes_available"][case["mode"]]:
        check = report["build_checks"][case["mode"]]
        if check["timed_out"]:
            return "OPTION_CHECK_TIMEOUT"
        diagnostic = Path(check["log"]).read_text(errors="replace")
        if re.search(r"undeclared|not declared|no member named", diagnostic):
            return "SDK_OPTION_ABSENT"
        return "OPTION_CHECK_ERROR"
    for op in ("write", "wait"):
        if case["operation"] in (op, "write_wait"):
            if not report["features"][op.upper() + str(case["width"])]:
                return "API_BUILD_UNAVAILABLE"
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, default=Path(os.environ.get("MACA_PATH", "/opt/maca")))
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--case-timeout", type=float, default=5)
    parser.add_argument("--compile-timeout", type=float, default=15)
    args = parser.parse_args(argv)
    for name in ("case_timeout", "compile_timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0 or value > 60:
            parser.error(name.replace("_", "-") + " must be in (0,60]")
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version=1, sdk=str(args.sdk.resolve()), device=args.device,
                  host=platform.platform(), launched=0, complete=False,
                  case_timeout_s=args.case_timeout, compile_timeout_s=args.compile_timeout,
                  cases=matrix(os.sysconf("SC_PAGE_SIZE")))
    save_report(directory, report)
    interrupted = False

    def on_terminate(signum, frame):
        raise KeyboardInterrupt

    previous_terminate = signal.signal(signal.SIGTERM, on_terminate)
    try:
        executable = build_probe(args, directory, report)
        if executable:
            info = run_bounded([str(executable), "--info", str(args.device)],
                               directory / "device-info.log", max(5, args.case_timeout))
            report["device_info"] = dict(info, **parse_native(Path(info["log"])))
            for case in report["cases"]:
                absent = unavailable_status(case, report)
                if absent:
                    case["status"] = absent
                else:
                    name = f"{case['mode']}-{case['size']}-{case['width']}-{case['operation']}"
                    case["status"] = "RUNNING"
                    outcome = run_bounded(
                        [str(executable), str(case["mode_index"]), str(case["size"]),
                         str(case["width"]), case["operation"], str(args.device)],
                        directory / (name + ".log"), args.case_timeout)
                    report["launched"] += 1
                    case.update(outcome)
                    case.update(parse_native(Path(outcome["log"])))
                    if outcome["timed_out"]:
                        case["status"] = "TIMEOUT"
                    elif outcome.get("launch_error"):
                        case["status"] = "LAUNCH_ERROR"
                    elif outcome["returncode"] != 0:
                        case["status"] = case.get("result", "CRASH_OR_EXIT_ERROR")
                    elif case.get("result") == "PASS":
                        case["status"] = "PASS"
                    else:
                        case["status"] = "INCOMPLETE_OUTPUT"
                print(f"{case['mode']} bytes={case['size']} bits={case['width']} "
                      f"{case['operation']}: {case['status']}", flush=True)
                save_report(directory, report)
    except KeyboardInterrupt:
        interrupted = True
    except Exception as error:
        report["setup_error"] = f"{type(error).__name__}: {error}"
    finally:
        for case in report["cases"]:
            if case["status"] == "PENDING":
                case["status"] = "NOT_RUN_INTERRUPTED" if interrupted else "NOT_RUN_SETUP_ERROR"
            elif case["status"] == "RUNNING":
                case["status"] = "INTERRUPTED" if interrupted else "CASE_SUPERVISOR_ERROR"
        report["complete"] = not interrupted and "setup_error" not in report
        save_report(directory, report)
        print((directory / "summary.md").read_text(), flush=True)
        print("Full report: " + str(directory / "results.json"), flush=True)
        signal.signal(signal.SIGTERM, previous_terminate)
    if interrupted:
        return 130
    if "setup_error" in report:
        print("SETUP ERROR: " + report["setup_error"], flush=True)
        return 2
    return int(any(case["status"] != "PASS" for case in report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())
