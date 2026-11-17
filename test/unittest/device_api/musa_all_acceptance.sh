#!/usr/bin/env bash
# Run the four unchanged public programs with the MUSA platform configuration.
# Standalone: upload this single file, then pass the repository path as arg 1.
set -euo pipefail
[[ -f /.dockerenv ]] || { echo 'ERROR: run inside the Docker container.' >&2; exit 2; }
root=${1:-$PWD}
cd "$root"
root=$PWD
[[ -f flagcx/include/flagcx.h && -f test/unittest/device_api/Makefile ]] || {
  echo 'ERROR: argument 1 must be the FlagCX repository directory.' >&2; exit 2;
}
rounds=${ROUNDS:-3}
build=${BUILD:-1}
metadata_only=${METADATA_ONLY:-0}
[[ "$metadata_only" == 0 || "$metadata_only" == 1 ]] || {
  echo 'ERROR: METADATA_ONLY must be 0 or 1.' >&2; exit 2;
}
if [[ "$metadata_only" == 1 ]]; then build=0; fi
read -r -a suites <<< "${SUITES:-T1 T2 T3 T4}"
[[ "$rounds" =~ ^[1-9][0-9]*$ && ( "$build" == 0 || "$build" == 1 ) ]] || {
  echo 'ERROR: ROUNDS must be positive; BUILD must be 0 or 1.' >&2; exit 2;
}
declare -A selected=()
for suite in "${suites[@]}"; do
  case "$suite" in T1|T2|T3|T4) ;; *) echo "ERROR: unknown suite $suite" >&2; exit 2 ;; esac
  [[ ! -v "selected[$suite]" ]] || { echo "ERROR: duplicate suite $suite" >&2; exit 2; }
  selected[$suite]=1
done
(( ${#suites[@]} > 0 )) || { echo 'ERROR: empty SUITES' >&2; exit 2; }
for tool in python3 make mpirun timeout tee awk; do
  command -v "$tool" >/dev/null || { echo "ERROR: missing $tool" >&2; exit 2; }
done
out=$(mktemp -d "$root/test/unittest/device_api/all-results-$(date +%Y%m%d_%H%M%S).XXXXXXXX")
echo "OUTPUT=$out"
trap 'rc=$?; if (( rc != 0 )); then echo "REGRESSION_STOPPED exit=$rc OUTPUT=$out" >&2; fi' EXIT
device_home=${DEVICE_HOME:-/usr/local/musa}
arch=${MUSA_ARCH:-mp_31}
export LD_LIBRARY_PATH="$root/build/lib:$device_home/lib:$device_home/lib64:${LD_LIBRARY_PATH:-}"
unset LD_PRELOAD LD_DEBUG LD_DEBUG_OUTPUT FLAGCX_DEBUG_FILE PRELOAD
unset FLAGCX_DEVICE_ADAPTOR_PLUGIN
export FLAGCX_USE_HETERO_COMM=1 FLAGCX_MEM_ENABLE=1 FLAGCX_VMM_ENABLE=0
export FLAGCX_P2P_DISABLE=0 FLAGCX_DMABUF_ENABLE=1 FLAGCX_SIGNAL_HOST_ENABLE=1
export FLAGCX_DEVICE_ONE_SIDED_FORCE_NET=0
export MCCL_TOPO_ENHANCE=0 MCCL_IB_DISABLE=1
export FLAGCX_DEBUG=${FLAGCX_DEBUG:-WARN} FLAGCX_DEBUG_SUBSYS=${FLAGCX_DEBUG_SUBSYS:-ALL}
{
  printf 'root=%s\nrounds=%s\nbuild=%s\narch=%s\nsuites=%s\n' \
    "$root" "$rounds" "$build" "$arch" "${suites[*]}"
  git rev-parse HEAD 2>/dev/null || true
  env | sort | sed -n '/^FLAGCX_/p; /^MCCL_/p; /^MUSA_/p; /^MTHREADS_/p; /^LD_LIBRARY_PATH=/p'
} > "$out/environment.txt"
git diff --binary > "$out/source-diff.patch" 2>/dev/null || true
for source in flagcx/adaptor/include/device_api/default_comm_traits.h \
  flagcx/adaptor/device/musa_adaptor.cc flagcx/core/flagcx_kernel_host.cc \
  flagcx/core/proxy.cc test/kernel/musa/device_api.mu test/kernel/musa/device_ir.mu; do
  sha256sum "$source"
done > "$out/source-sha256.txt"

# Validate coverage per rank, rather than trusting one final result line.
cat > "$out/check.py" <<'FLAGCX_ALL_CHECK_PY'
from pathlib import Path
from collections import Counter
import json
import re
import sys

out = Path(sys.argv[1])
suite, round_id, rc = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
text = (out / f"{suite}-round{round_id}.log").read_text(encoding="utf-8", errors="replace")
errors = []
counts = {}
skips = Counter(re.findall(r"\[rank \d+\]\s+(K\d+[ab]?)\b[^\n]*\bSKIP\b", text))
if rc:
    errors.append(f"MPI/pipeline exit={rc}")
if suite in ("T1", "T2"):
    case, name = ("K10", "IntraAllReduce(composite)") if suite == "T1" else ("K14", "OneSidedAlltoAll")
    pattern = r"\[rank (\d+)\]\s+" + case + r"\s+" + re.escape(name) + r"\s+PASS\s*$"
    passes = Counter(int(r) for r in re.findall(pattern, text, re.M))
    overall = Counter(int(r) for r in re.findall(r"\[rank (\d+)\]\s+=== Overall: PASS ===", text))
    counts[case] = dict(passes)
    if passes != Counter({r: 15 for r in range(8)}):
        errors.append(f"{case} coverage={dict(passes)}; expected 15 per rank, ranks 0..7")
    if overall != Counter({r: 1 for r in range(8)}):
        errors.append(f"Overall PASS coverage={dict(overall)}; expected one per rank")
    if re.search(r"\[rank \d+\]\s+(?:K\d+[ab]?\b[^\n]*\bFAIL\b|=== Overall: FAIL ===)", text):
        errors.append("A public case or overall result reported FAIL")
else:
    for case in range(16, 26):
        passes = Counter(int(r) for r in re.findall(r"\[rank (\d+)\]\s+S" + str(case) + r"\b[^\n]*:\s*PASS\s*$", text, re.M))
        counts[f"S{case}"] = dict(passes)
        if passes != Counter({r: 15 for r in range(8)}):
            errors.append(f"S{case} coverage={dict(passes)}; expected 15 per rank, ranks 0..7")
    if text.count("=== Final Result: ALL PASS ===") != 1:
        errors.append("Expected exactly one ALL PASS summary")
    if re.search(r"\[rank \d+\]\s+S(?:1[6-9]|2[0-5])\b[^\n]*:\s*(?:FAIL|SKIP)\b", text):
        errors.append("A required S16-S25 case reported FAIL or SKIP")
record = {"suite": suite, "round": round_id, "rc": rc, "acceptance_passed": not errors,
          "case_pass_counts_by_rank": counts, "skipped_not_executed": dict(skips),
          "errors": errors, "log": str(out / f"{suite}-round{round_id}.log")}
(out / f"{suite}-round{round_id}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("T?-round*.json"))]
summary = {"completed_runs": len(records), "completed_runs_passed": all(r["acceptance_passed"] for r in records), "runs": records}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
with (out / "results.tsv").open("w", encoding="utf-8") as f:
    f.write("suite\tround\trc\tacceptance_result\tskipped_not_executed\n")
    for r in records:
        f.write(f'{r["suite"]}\t{r["round"]}\t{r["rc"]}\t' + ("PASS" if r["acceptance_passed"] else "FAIL") + "\t" + json.dumps(r["skipped_not_executed"], sort_keys=True) + "\n")
for error in errors:
    print("COVERAGE_ERROR: " + error)
for case, count in sorted(skips.items()):
    print(f"SKIP_NOT_EXECUTED: {suite} {case} lines={count}; this is not a functional PASS")
sys.exit(0 if not errors else 1)
FLAGCX_ALL_CHECK_PY

bins=(test_device_api_intra test_device_api_inter test_device_ir_unified_intra test_device_ir_unified_inter)
bin_dir="$root/test/unittest/device_api/build/bin"
if [[ "$build" == 1 ]]; then
  build_step() {
    local label=$1
    shift
    printf '%q ' "$@" > "$out/$label.command.txt"
    printf '\n' >> "$out/$label.command.txt"
    echo "BUILD $label: $out/$label.log"
    local rc=0
    timeout --kill-after=5s 600s "$@" > "$out/$label.log" 2>&1 || rc=$?
    if (( rc != 0 )); then tail -80 "$out/$label.log"; exit "$rc"; fi
  }
  flags=(USE_MUSA=1 COMPILE_KERNEL=1 "DEVICE_HOME=$device_home" "MUSA_ARCH=$arch")
  build_step library-build make -B -j1 "${flags[@]}"
  build_step kernel-build make -B -C test/kernel/musa all "PROJECT_ROOT=$root" "${flags[@]}"
  targets=()
  for bin in "${bins[@]}"; do targets+=("$bin_dir/$bin"); done
  build_step host-build make -B -C test/unittest/device_api "${targets[@]}" \
    "PROJECT_ROOT=$root" "${flags[@]}"
fi
for bin in "${bins[@]}"; do
  [[ -x "$bin_dir/$bin" ]] || { echo "ERROR: missing $bin_dir/$bin" >&2; exit 2; }
done
sha256sum "$root/build/lib/libflagcx.so" "${bins[@]/#/$bin_dir/}" > "$out/binary-sha256.txt"

# Capture the environment fields required by the vendor acceptance report.
python3 - "$root" "$out" "$device_home" "$arch" <<'FLAGCX_ENV_REPORT_PY'
from pathlib import Path
import ctypes
import json
import platform
import re
import subprocess
import sys

root, out, sdk = map(Path, sys.argv[1:4])
def command(args):
    try:
        p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, errors="replace", timeout=10)
        return {"command": args, "exit": p.returncode, "output": p.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": args, "error": str(error)}
def version(library, symbol):
    try:
        lib = ctypes.CDLL(str(sdk / "lib" / library))
        fn = getattr(lib, symbol)
        fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
        fn.restype = ctypes.c_int
        value = ctypes.c_int()
        result = fn(ctypes.byref(value))
        return {"library": str(sdk / "lib" / library), "result": result,
                "value": value.value if result == 0 else None}
    except (OSError, AttributeError) as error:
        return {"error": str(error)}
os_release = Path("/etc/os-release")
device_info = command(["mthreads-gmi", "-q"])
driver_file = Path("/proc/driver/mtgpu/version")
report = {
    "FlagCX": {"commit": command(["git", "rev-parse", "HEAD"]),
               "branch/tag": command(["git", "branch", "--show-current"]),
               "backend": "Default", "registration": "symmetric window (-R 2 for T1/T2)",
               "build arguments": ["USE_MUSA=1", "COMPILE_KERNEL=1",
                                   "DEVICE_HOME=" + str(sdk), "MUSA_ARCH=" + sys.argv[4]]},
    "Host": {"OS": os_release.read_text() if os_release.exists() else platform.platform(),
             "kernel": platform.release(), "CPU architecture": platform.machine(),
             "Host compiler": command(["g++", "--version"])},
    "Device": {"vendor/model and firmware": device_info,
               "device count": version("libmusart.so", "musaGetDeviceCount"),
               "topology": command(["mthreads-gmi", "topo", "-m"]),
               "Device SDK": version("libmusart.so", "musaRuntimeGetVersion"),
               "driver API": version("libmusart.so", "musaDriverGetVersion"),
               "driver": driver_file.read_text() if driver_file.exists() else device_info,
               "firmware fallback": "N/A if not exposed in device information",
               "Device compiler": command([str(sdk / "bin/mcc"), "--version"]),
               "device architecture": sys.argv[4]},
    "Communication runtime": {"MPI": command(["mpirun", "--version"]),
                              "CCL": version("libmccl.so", "mcclGetVersion"),
                              "SHMEM": "N/A (Default backend)"},
    "Artifacts": {"libflagcx.so path": str(root / "build/lib/libflagcx.so"),
                  "complete build commands": "*-build.command.txt and *-build.log",
                  "loaded libflagcx.so path": {}},
}
expected = (root / "build/lib/libflagcx.so").resolve()
errors = []
for name in ("test_device_api_intra", "test_device_api_inter",
             "test_device_ir_unified_intra", "test_device_ir_unified_inter"):
    executable = root / "test/unittest/device_api/build/bin" / name
    resolved = command(["ldd", str(executable)])
    report["Artifacts"]["loaded libflagcx.so path"][name] = resolved
    match = re.search(r"libflagcx\.so\s+=>\s+(\S+)", resolved.get("output", ""))
    if resolved.get("exit") != 0 or match is None or Path(match.group(1)).resolve() != expected:
        errors.append("Unexpected/missing libflagcx.so resolution: " + name)
(out / "environment-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
if errors:
    sys.exit("\n".join(errors))
FLAGCX_ENV_REPORT_PY
if [[ "$metadata_only" == 1 ]]; then
  echo "METADATA_SAVED=$out/environment-report.json"
  echo 'No build or tests were started.'
  exit 0
fi

mpi_env=()
for name in LD_LIBRARY_PATH FLAGCX_USE_HETERO_COMM FLAGCX_MEM_ENABLE FLAGCX_VMM_ENABLE \
  FLAGCX_IB_DISABLE FLAGCX_P2P_DISABLE FLAGCX_DMABUF_ENABLE FLAGCX_SIGNAL_HOST_ENABLE \
  FLAGCX_DEVICE_ONE_SIDED_FORCE_NET \
  MCCL_TOPO_ENHANCE MCCL_IB_DISABLE FLAGCX_DEBUG FLAGCX_DEBUG_SUBSYS; do
  mpi_env+=(-x "$name")
done
for name in FLAGCX_IB_HCA FLAGCX_IB_GID_INDEX FLAGCX_SOCKET_IFNAME \
  MUSA_VISIBLE_DEVICES MTHREADS_VISIBLE_DEVICES; do
  if [[ -v "$name" ]]; then mpi_env+=(-x "$name"); fi
done
for suite in "${suites[@]}"; do
  case "$suite" in
    T1) bin=${bins[0]}; export FLAGCX_IB_DISABLE=1; args=(-b 1K -e 16M -f 2 -R 2) ;;
    T2) bin=${bins[1]}; export FLAGCX_IB_DISABLE=0; args=(-b 1K -e 16M -f 2 -R 2) ;;
    T3) bin=${bins[2]}; export FLAGCX_IB_DISABLE=1; args=(-b 1K -e 16M -f 2) ;;
    T4) bin=${bins[3]}; export FLAGCX_IB_DISABLE=0; args=(-b 1K -e 16M -f 2) ;;
  esac
  for (( round=1; round<=rounds; round++ )); do
    log="$out/$suite-round$round.log"
    if [[ "$suite" == T1 || "$suite" == T3 ]]; then
      cmd=(mpirun --allow-run-as-root --bind-to none --tag-output -np 8
        "${mpi_env[@]}" -x FLAGCX_HOSTID=node0 -x NCCL_HOSTID=node0 "$bin_dir/$bin" "${args[@]}")
    else
      cmd=(mpirun --allow-run-as-root --bind-to none --tag-output
        -np 4 "${mpi_env[@]}" -x FLAGCX_HOSTID=node0 -x NCCL_HOSTID=node0 "$bin_dir/$bin" "${args[@]}"
        : -np 4 "${mpi_env[@]}" -x FLAGCX_HOSTID=node1 -x NCCL_HOSTID=node1 "$bin_dir/$bin" "${args[@]}")
    fi
    printf '%q ' timeout --signal=TERM --kill-after=60s 30m "${cmd[@]}" > "$out/$suite-command$round.txt"
    printf '\n' >> "$out/$suite-command$round.txt"
    echo "RUN $suite round $round/$rounds: $log (FLAGCX_IB_DISABLE=$FLAGCX_IB_DISABLE)"
    started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    set +e
    timeout --signal=TERM --kill-after=60s 30m "${cmd[@]}" 2>&1 |
      tee "$log" |
      awk '/\[rank [0-9]+\].*(K[0-9]|S[0-9])|Overall:|Final Result:|FLAGCX WARN|native=/ { print; fflush(); }'
    pipeline_rc=("${PIPESTATUS[@]}")
    set -e
    rc=${pipeline_rc[0]}
    if (( rc == 0 && pipeline_rc[1] != 0 )); then rc=${pipeline_rc[1]}; fi
    if (( rc == 0 && pipeline_rc[2] != 0 )); then rc=${pipeline_rc[2]}; fi
    echo "$rc" > "$out/$suite-round$round.rc"
    printf 'start=%s\nend=%s\nexit=%s\n' "$started" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" \
      > "$out/$suite-round$round.timing.txt"
    verified=0
    python3 "$out/check.py" "$out" "$suite" "$round" "$rc" || verified=$?
    if (( rc != 0 || verified != 0 )); then
      echo "FAIL $suite round $round: exit=$rc coverage_exit=$verified"
      tail -60 "$log"
      if (( rc != 0 )); then exit "$rc"; else exit 1; fi
    fi
    echo "$suite round $round PASS"
  done
done
python3 - "$out" "$rounds" "${suites[@]}" <<'FLAGCX_ALL_FINAL_PY'
from pathlib import Path
import json
import sys
p = Path(sys.argv[1]) / "summary.json"
s = json.loads(p.read_text(encoding="utf-8"))
s["planned_suites"] = sys.argv[3:]
s["rounds_per_suite"] = int(sys.argv[2])
s["all_selected_completed"] = s["completed_runs"] == len(s["planned_suites"]) * s["rounds_per_suite"]
s["acceptance_passed"] = s["all_selected_completed"] and s["completed_runs_passed"]
p.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
sys.exit(0 if s["acceptance_passed"] else 1)
FLAGCX_ALL_FINAL_PY
echo "REGRESSION_RESULT=PASS suites=${suites[*]} rounds=$rounds"
echo 'NOTE: documented SKIP cases are recorded separately; a suite PASS does not validate those cases.'
echo "RESULTS=$out/results.tsv"
echo "SUMMARY=$out/summary.json"
echo "OUTPUT=$out"
