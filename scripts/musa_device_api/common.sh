#!/usr/bin/env bash
# Shared configuration and helpers for FlagCX MUSA Device API W1 scripts.
#
# All scripts record real exit codes and produce machine-readable .status
# files. Nothing here ever turns a failure into a PASS: failures are FAIL,
# missing prerequisites are BLOCKED or SKIP, and only exit code 0 is PASS.
#
# Sourced, never executed directly.

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---------------------------------------------------------------------------
# Unified test parameters (fixed by the W1 plan unless overridden).
# ---------------------------------------------------------------------------
NRANKS="${NRANKS:-8}"
INTER_NODE_RANKS="${INTER_NODE_RANKS:-4}"   # per logical node; 2 x 4 = 8
MIN_BYTES="${MIN_BYTES:-1K}"
MAX_BYTES="${MAX_BYTES:-16M}"
STEP_FACTOR="${STEP_FACTOR:-2}"
REG_MODE="${REG_MODE:-2}"                    # -R 2 = window registration
RUN_TIMEOUT="${RUN_TIMEOUT:-1800}"            # 30 minutes
BUILD_TIMEOUT="${BUILD_TIMEOUT:-1800}"
BUILD_JOBS="${BUILD_JOBS:-8}"

MPI_FLAVOR="${MPI_FLAVOR:-openmpi}"
MPIRUN="${MPIRUN:-mpirun}"

# Unified environment required by the W1 acceptance plan.
export FLAGCX_USE_HETERO_COMM="${FLAGCX_USE_HETERO_COMM:-1}"
export FLAGCX_VMM_ENABLE="${FLAGCX_VMM_ENABLE:-0}"
export FLAGCX_MEM_ENABLE="${FLAGCX_MEM_ENABLE:-1}"
export FLAGCX_P2P_DISABLE="${FLAGCX_P2P_DISABLE:-1}"

# MUSA locations (overridable).
MUSA_HOME="${MUSA_HOME:-/usr/local/musa}"
MUSA_LIB="${MUSA_LIB:-$MUSA_HOME/lib}"

# Result / log layout. Kept under build/ so generated artifacts are never
# accidentally committed.
RESULT_DIR="${RESULT_DIR:-$PROJECT_ROOT/build/musa_device_api_results}"
LOG_DIR="${LOG_DIR:-$RESULT_DIR/logs}"
mkdir -p "$LOG_DIR"

FLAGCX_BUILD_LIB="${FLAGCX_BUILD_LIB:-$PROJECT_ROOT/build/lib}"
DEVICE_API_BUILD_DIR="${DEVICE_API_BUILD_DIR:-$PROJECT_ROOT/test/unittest/device_api/build}"
DEVICE_API_BIN_DIR="${DEVICE_API_BIN_DIR:-$DEVICE_API_BUILD_DIR/bin}"

# Runtime library path must contain the freshly built libflagcx plus the MUSA
# runtime. Keep any caller-provided path in front.
export LD_LIBRARY_PATH="$FLAGCX_BUILD_LIB:$MUSA_LIB:${LD_LIBRARY_PATH:-}"

# ---------------------------------------------------------------------------
# MPI flags
# ---------------------------------------------------------------------------
MPI_ROOT_ARGS=()
if [ "$MPI_FLAVOR" = "openmpi" ]; then
  MPI_ROOT_ARGS=(--allow-run-as-root)
elif [ "$MPI_FLAVOR" = "mpich" ]; then
  MPI_ROOT_ARGS=()
else
  echo "Unsupported MPI_FLAVOR '$MPI_FLAVOR' (use openmpi or mpich)" >&2
  exit 2
fi

# Env vars propagated to every MPI rank. Values are read from the current
# environment at launch time, so callers can override them.
MPI_PROPAGATED_ENV=(
  FLAGCX_USE_HETERO_COMM
  FLAGCX_VMM_ENABLE
  FLAGCX_MEM_ENABLE
  FLAGCX_P2P_DISABLE
)
[ -n "${FLAGCX_IB_DISABLE:-}" ] && MPI_PROPAGATED_ENV+=(FLAGCX_IB_DISABLE)
[ -n "${FLAGCX_SOCKET_IFNAME:-}" ] && MPI_PROPAGATED_ENV+=(FLAGCX_SOCKET_IFNAME)
[ -n "${FLAGCX_FORCE_NET_SOCKET:-}" ] && MPI_PROPAGATED_ENV+=(FLAGCX_FORCE_NET_SOCKET)
MPI_PROPAGATED_ENV+=(LD_LIBRARY_PATH)

# ---------------------------------------------------------------------------
# Transport detection
# ---------------------------------------------------------------------------
# The container used for W1 does not mount /dev/infiniband, so ibverbs sees no
# HCAs and FlagCX would silently fall back to Socket. Detect this and make the
# fallback explicit for inter-node runs instead of nondeterministic.
RDMA_AVAILABLE=0
if [ -d /dev/infiniband ] && command -v ibv_devices >/dev/null 2>&1; then
  if ibv_devices 2>/dev/null | grep -q 'mlx'; then
    RDMA_AVAILABLE=1
  fi
fi

ensure_net_env() {
  # The transport assertions below grep FlagCX's own INFO records
  # ("NET/Socket : Using", "NET/IB : libibverbs reported N"). flagcxDebugLog()
  # in flagcx/service/debug.cc drops every record when
  # flagcxDebugLevel < level, and an unset FLAGCX_DEBUG maps to
  # FLAGCX_LOG_NONE, so those lines only exist at INFO or above. Raise the
  # level here (inter-node runs only) so the assertion is satisfiable without
  # weakening it; intra-node runs keep the default level to avoid huge logs.
  # FLAGCX_DEBUG_SUBSYS keeps the extra output to INIT/NET (the two flags the
  # transport lines are logged under); the default mask is INIT|ENV already.
  case "${FLAGCX_DEBUG:-}" in
    INFO|ABORT|TRACE|info|abort|trace) : ;; # caller already verbose enough
    *)
      export FLAGCX_DEBUG=INFO
      export FLAGCX_DEBUG_SUBSYS="${FLAGCX_DEBUG_SUBSYS:-INIT,NET}"
      log "raised FlagCX log level to INFO (INIT,NET) for the transport assertion"
      ;;
  esac
  # Always forward the level/subset to the ranks: the caller may have set them
  # itself, and mpirun only propagates what is listed here.
  case " ${MPI_PROPAGATED_ENV[*]} " in
    *" FLAGCX_DEBUG "*) ;;
    *) MPI_PROPAGATED_ENV+=(FLAGCX_DEBUG) ;;
  esac
  case " ${MPI_PROPAGATED_ENV[*]} " in
    *" FLAGCX_DEBUG_SUBSYS "*) ;;
    *) MPI_PROPAGATED_ENV+=(FLAGCX_DEBUG_SUBSYS) ;;
  esac
  if [ "$RDMA_AVAILABLE" -eq 0 ] && [ -z "${FLAGCX_IB_DISABLE:-}" ]; then
    export FLAGCX_IB_DISABLE=1
    MPI_PROPAGATED_ENV+=(FLAGCX_IB_DISABLE)
    warn "no RDMA HCA visible (/dev/infiniband missing); inter-node runs use Socket, not IB"
  fi
  # FLAGCX_FORCE_NET_SOCKET=1 is the official deterministic switch in
  # flagcx/core/net.cc:819. Relying only on ibv_get_device_list() returning 0
  # makes the fallback implicit; force it when no HCA is visible.
  if [ "$RDMA_AVAILABLE" -eq 0 ] && [ -z "${FLAGCX_FORCE_NET_SOCKET:-}" ]; then
    export FLAGCX_FORCE_NET_SOCKET=1
    MPI_PROPAGATED_ENV+=(FLAGCX_FORCE_NET_SOCKET)
    log "forced Socket network (FLAGCX_FORCE_NET_SOCKET=1)"
  fi
  if [ -z "${FLAGCX_SOCKET_IFNAME:-}" ]; then
    local ifn
    if command -v ip >/dev/null 2>&1; then
      ifn="$(ip -o -4 addr show 2>/dev/null | awk '
        $2!="lo" && $2!~/(docker|virbr|veth|br-)/ {print $2}' | head -1)"
    fi
    if [ -n "${ifn:-}" ]; then
      export FLAGCX_SOCKET_IFNAME="$ifn"
      MPI_PROPAGATED_ENV+=(FLAGCX_SOCKET_IFNAME)
      log "inter-node Socket interface: $FLAGCX_SOCKET_IFNAME"
    fi
  fi
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log()  { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
warn() { printf '[%s] WARN: %s\n' "$(date '+%H:%M:%S')" "$*" >&2; }
die()  { printf '[%s] ERROR: %s\n' "$(date '+%H:%M:%S')" "$*" >&2; exit 2; }

# ---------------------------------------------------------------------------
# Status recording
# ---------------------------------------------------------------------------
# write_status <name> <status> <exit> <duration_s> <command...>
write_status() {
  local name="$1" status="$2" rc="$3" dur="$4"; shift 4
  local st="$LOG_DIR/$name.status" kept=""
  # A caller may record evidence before the run (assert_runtime_lib writes the
  # resolved libflagcx path). Those stubs carry no status= line yet, and
  # rewriting the record from the exit code used to truncate them, so a gate
  # could pass without the proof that it used this build's library. A record
  # that still has a status= line is a stale one from an earlier run and is
  # replaced entirely.
  if [ -f "$st" ] && ! grep -q '^status=' "$st" 2>/dev/null; then
    kept="$(grep -Ev '^(name|status|exit|duration_s|command)=' "$st" 2>/dev/null || true)"
  fi
  {
    printf 'name=%s\n' "$name"
    printf 'status=%s\n' "$status"
    printf 'exit=%s\n' "$rc"
    printf 'duration_s=%s\n' "$dur"
    printf 'command=%s\n' "$*"
    if [ -n "$kept" ]; then
      printf '%s\n' "$kept"
    fi
  } >"$st"
}

# set_status <name> <status> <exit> <duration_s>
# Update an existing .status file in place, preserving assertion/case/verdict
# evidence appended after the initial run_capture record. Falls back to
# write_status when the file does not exist yet.
set_status() {
  local name="$1" status="$2" rc="$3" dur="$4"
  local st="$LOG_DIR/$name.status" tmp="$LOG_DIR/$name.status.tmp"
  if [ ! -f "$st" ]; then
    write_status "$name" "$status" "$rc" "$dur" "$name"
    return
  fi
  awk -v s="$status" -v r="$rc" -v d="$dur" '
    BEGIN { hs = 0; hr = 0; hd = 0 }
    /^status=/     { print "status=" s;     hs = 1; next }
    /^exit=/       { print "exit=" r;       hr = 1; next }
    /^duration_s=/ { print "duration_s=" d; hd = 1; next }
    { print }
    END {
      if (!hs) print "status=" s
      if (!hr) print "exit=" r
      if (!hd) print "duration_s=" d
    }
  ' "$st" >"$tmp" && mv "$tmp" "$st"
}

# record_blocked <name> <reason>
record_blocked() {
  local name="$1" reason="$2"
  write_status "$name" BLOCKED 0 0 "blocked: $reason"
  printf '%s\n' "$reason" >"$LOG_DIR/$name.reason"
  warn "$name: BLOCKED - $reason"
}

# run_capture <name> <timeout_s> <command...>
# Runs the command with a hard timeout, capturing stdout/stderr separately.
# Returns the command's exit code (124 on timeout). Never masks failures.
run_capture() {
  local name="$1" tmo="$2"; shift 2
  local out="$LOG_DIR/$name.out"
  local err="$LOG_DIR/$name.err"
  local start end rc dur status
  start=$(date +%s)
  timeout --signal=TERM --kill-after=30 "$tmo" "$@" >"$out" 2>"$err"
  rc=$?
  end=$(date +%s); dur=$((end - start))
  if [ "$rc" -eq 0 ]; then
    status=PASS
  elif [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    status=TIMEOUT
  else
    status=FAIL
  fi
  write_status "$name" "$status" "$rc" "$dur" "$*"
  log "$name: $status (exit=$rc, ${dur}s)  -> $out"
  return "$rc"
}

# finalize_status <name> <verdict> <exit>
# run_capture() records a provisional status from the exit code alone, so a
# binary that exits 0 without printing anything would otherwise stay
# status=PASS. Every caller must finalise the record from the verdict once the
# output assertions have run. The real duration recorded by run_capture is
# preserved.
finalize_status() {
  local name="$1" verdict="$2" rc="$3"
  local st="$LOG_DIR/$name.status" status dur
  case "$verdict" in
    PASS) status=PASS ;;
    TIMEOUT)
      status=TIMEOUT
      case "$rc" in 124 | 137) ;; *) rc=124 ;; esac
      ;;
    BLOCKED | SKIP | NOT_RUN) status="$verdict" ;;
    *)
      status=FAIL
      [ "$rc" -eq 0 ] && rc=1
      ;;
  esac
  dur="$(sed -n 's/^duration_s=//p' "$st" 2>/dev/null | head -1)"
  set_status "$name" "$status" "$rc" "${dur:-0}"
  return 0
}

# ---------------------------------------------------------------------------
# Output classification
# ---------------------------------------------------------------------------
# Some suites report per-rank PASS/FAIL but still exit 0. Scan the captured
# output for explicit failure markers; require at least one success marker.
# A hard failure marker always wins over a success marker, so a run that
# prints "PASS" on seven ranks and "FAIL" on the eighth can never be PASS.

# count_marker <pattern> <file...>
# Sum grep -c hits across files without the "file:count" prefix pitfall.
count_marker() {
  local pattern="$1"; shift
  local total=0 f n
  for f in "$@"; do
    [ -f "$f" ] || continue
    n=$(grep -Eic -- "$pattern" "$f" 2>/dev/null || true)
    case "$n" in ''|*[!0-9]*) n=0 ;; esac
    total=$((total + n))
  done
  printf '%s\n' "$total"
}

# classify_output <name> <exit_code>
classify_output() {
  local name="$1" rc="$2"
  local out="$LOG_DIR/$name.out"
  local err="$LOG_DIR/$name.err"
  if [ ! -f "$out" ] && [ ! -f "$err" ]; then
    echo "NOT_RUN"
    return
  fi
  local fail_hits pass_hits
  fail_hits=$(count_marker \
    '(^|[^A-Za-z])(FAIL|FAILED|FAILURE|MISMATCH|ERROR:|Aborting|SOME FAILED)([^A-Za-z]|$)' \
    "$out" "$err")
  pass_hits=$(count_marker \
    '(^|[^A-Za-z])(PASS|PASSED|SUCCESS|ALL PASS)([^A-Za-z]|$)' "$out")
  if [ "$rc" -ne 0 ]; then
    echo "FAIL"
  elif [ "${fail_hits:-0}" -gt 0 ]; then
    echo "FAIL"
  elif [ "${pass_hits:-0}" -gt 0 ]; then
    echo "PASS"
  else
    echo "NOT_RUN"
  fi
}

# status_is_consistent <status> <verdict>
# A .status row is only trustworthy when its status and verdict agree. The
# provisional status written by run_capture comes from the exit code, while the
# verdict comes from classify_output + the log assertions; a mismatch (most
# importantly status=PASS with verdict!=PASS) means the evidence contradicts
# itself and must not be counted as a pass. TIMEOUT may pair with FAIL because
# classify_output only sees the exit code (124/137) and the wrapper re-labels
# such a run as TIMEOUT afterwards.
status_is_consistent() {
  local status="$1" verdict="$2"
  [ -n "$verdict" ] || return 0
  case "$status:$verdict" in
    PASS:PASS | FAIL:FAIL | TIMEOUT:TIMEOUT | FAIL:TIMEOUT | TIMEOUT:FAIL) return 0 ;;
    BLOCKED:BLOCKED | SKIP:SKIP | NOT_RUN:NOT_RUN | NOT_RUN:SKIP) return 0 ;;
    *) return 1 ;;
  esac
}

# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------
find_first_bin() {
  local b
  for b in "$@"; do
    if [ -x "$DEVICE_API_BIN_DIR/$b" ]; then
      printf '%s\n' "$DEVICE_API_BIN_DIR/$b"
      return 0
    fi
  done
  return 1
}

# Emit MPI `-x`/`--genv` flags for the runtime environment.
# NOTE: `mpirun ... env VAR=val cmd` is silently a no-op with this OpenMPI
# build, so we always use the launcher's native env option.
mpi_env_flags() {
  local hostid="${1:-}" devices="${2:-}"
  local -a parts=()
  local v
  if [ "$MPI_FLAVOR" = "openmpi" ]; then
    [ -n "$hostid" ] && parts+=(-x "FLAGCX_HOSTID=$hostid" -x "NCCL_HOSTID=$hostid")
    for v in "${MPI_PROPAGATED_ENV[@]}"; do
      # shellcheck disable=SC2163
      parts+=(-x "$v=${!v}")
    done
    [ -n "$devices" ] && parts+=(-x "MUSA_VISIBLE_DEVICES=$devices")
  else
    [ -n "$hostid" ] && parts+=(--genv "FLAGCX_HOSTID=$hostid" --genv "NCCL_HOSTID=$hostid")
    for v in "${MPI_PROPAGATED_ENV[@]}"; do
      # shellcheck disable=SC2163
      parts+=(--genv "$v=${!v}")
    done
    [ -n "$devices" ] && parts+=(--genv "MUSA_VISIBLE_DEVICES=$devices")
  fi
  printf '%s\n' "${parts[*]}"
}

# require_bin <name> <candidate...>; sets BIN or records BLOCKED and returns 1.
require_bin() {
  local name="$1"; shift
  if BIN="$(find_first_bin "$@")"; then
    return 0
  fi
  record_blocked "$name" "no device test binary among: $* (run build_all.sh; device build is blocked until A delivers musa.mk/kernels)"
  return 1
}

# ---------------------------------------------------------------------------
# Log assertions
# ---------------------------------------------------------------------------
# An exit code alone is not enough: a suite can exit 0 while reporting a wrong
# topology or silently falling back to Socket. These helpers make those
# conditions explicit FAILs instead of trusting the binary.
# STRICT_TRANSPORT_ASSERT=0 downgrades transport assertions to warnings.

# check_log_pattern <name> <extended-regex> <description>
check_log_pattern() {
  local name="$1" pattern="$2" desc="$3"
  if grep -Eq -- "$pattern" "$LOG_DIR/$name.out" 2>/dev/null; then
    printf 'assertion=PASS %s\n' "$desc" >>"$LOG_DIR/$name.status"
    return 0
  fi
  printf 'assertion=FAIL %s (pattern: %s)\n' "$desc" "$pattern" \
    >>"$LOG_DIR/$name.status"
  warn "$name: expected $desc (missing pattern: $pattern)"
  return 1
}

# assert_transport <name>: require the transport we actually selected.
assert_transport() {
  local name="$1"
  if [ "$RDMA_AVAILABLE" -eq 1 ]; then
    check_log_pattern "$name" 'NET/IB : libibverbs reported [1-9]' \
      "IB transport active (RDMA HCA visible)"
    return $?
  fi
  if [ "${STRICT_TRANSPORT_ASSERT:-1}" -eq 0 ]; then
    if grep -Eq 'NET/Socket : Using' "$LOG_DIR/$name.out" 2>/dev/null; then
      printf 'assertion=PASS explicit Socket transport\n' >>"$LOG_DIR/$name.status"
    else
      printf 'assertion=WARN no explicit Socket transport line\n' >>"$LOG_DIR/$name.status"
      warn "$name: no NET/Socket line (strict assertion disabled)"
    fi
    return 0
  fi
  check_log_pattern "$name" 'NET/Socket : Using' \
    "explicit Socket transport (no RDMA HCA present)"
}

# apply_node_assertions <name> <expected-nodes> <expected-intra>
# Verifies the test really split into the logical topology we asked for.
apply_node_assertions() {
  local name="$1" nodes="$2" intra="$3" rc=0
  check_log_pattern "$name" "Nodes: $nodes([^0-9]|$)" \
    "run reports $nodes nodes" || rc=1
  check_log_pattern "$name" "IntraSize: $intra([^0-9]|$)" \
    "run reports intra size $intra" || rc=1
  return "$rc"
}

# require_case_pass <name> <want> <pattern> <description>
# An exit code of 0 is not enough: every v3 case must actually print PASS on
# every rank. A raw line count cannot prove that, because each case is printed
# once per rank *per size* and the suites sweep many sizes (1K..16M, factor 2),
# so a single rank satisfies a ">= NRANKS hits" rule on its own. Instead count
# the distinct "[rank N]" prefixes of the matching lines and require the full
# set 0..want-1 to be present. Patterns that only match a single rank-less
# global line (e.g. "=== Final Result: ALL PASS ===") keep the plain hit-count
# semantics, because there is no per-rank coverage to prove for those.
require_case_pass() {
  local name="$1" want="$2" pattern="$3" desc="$4"
  local out="$LOG_DIR/$name.out" hits ranks=0 missing="" r seen_list=" "
  local -a seen=()
  hits=$(count_marker "$pattern" "$out")
  if [ -f "$out" ]; then
    while IFS= read -r r; do
      [ -n "$r" ] && seen+=("$r")
    done < <(grep -Ei -- "$pattern" "$out" 2>/dev/null |
      sed -n 's/^\[rank \([0-9][0-9]*\)\].*/\1/p' | sort -n -u)
  fi
  for r in "${seen[@]:-}"; do
    [ -n "$r" ] && seen_list+="$r "
  done
  ranks="$(printf '%s' "$seen_list" | wc -w)"
  ranks="${ranks//[!0-9]/}"; ranks="${ranks:-0}"

  if [ "$ranks" -eq 0 ]; then
    # No rank-prefixed match: a global result line, checked by hit count.
    if [ "$hits" -ge "$want" ]; then
      printf 'case=PASS %s (hits=%s ranks=global)\n' "$desc" "$hits" \
        >>"$LOG_DIR/$name.status"
      return 0
    fi
    printf 'case=FAIL %s (want>=%s hits=%s ranks=global)\n' "$desc" "$want" \
      "$hits" >>"$LOG_DIR/$name.status"
    warn "$name: expected PASS for $desc (want>=$want, got $hits)"
    return 1
  fi

  for ((r = 0; r < want; r++)); do
    case "$seen_list" in
      *" $r "*) ;;
      *) missing+=" $r" ;;
    esac
  done
  if [ -z "$missing" ]; then
    printf 'case=PASS %s (hits=%s ranks=%s/%s)\n' "$desc" "$hits" "$ranks" \
      "$want" >>"$LOG_DIR/$name.status"
    return 0
  fi
  printf 'case=FAIL %s (want>=%s hits=%s ranks=%s/%s missing:%s)\n' \
    "$desc" "$want" "$hits" "$ranks" "$want" "$missing" \
    >>"$LOG_DIR/$name.status"
  warn "$name: expected PASS for $desc on ranks 0..$((want - 1)); missing ranks:$missing"
  return 1
}

# total_duration <name>...
# Sum the recorded duration of the given target records so an aggregate stage
# reports the wall time of the runs it stands for instead of 0.
total_duration() {
  local n d sum=0
  for n in "$@"; do
    d="$(sed -n 's/^duration_s=//p' "$LOG_DIR/$n.status" 2>/dev/null | head -1)"
    case "$d" in '' | *[!0-9]*) d=0 ;; esac
    sum=$((sum + d))
  done
  printf '%s' "$sum"
}

# assert_runtime_lib <name> <binary>
# Called before the run by the device gates: drop any record left by an earlier
# invocation so this run owns a clean .status (run_capture would truncate it
# anyway, but the pre-run assertion below must not be appended to stale rows).
begin_status_record() { rm -f "$LOG_DIR/$1.status"; }

# Fail if the binary would load a libflagcx.so other than the freshly built one
# (stale artifact, distro copy, or a broken rpath/LD_LIBRARY_PATH).
assert_runtime_lib() {
  local name="$1" bin="$2" resolved expected
  resolved="$(ldd "$bin" 2>/dev/null | awk '/libflagcx\.so/ {print $3; exit}')"
  expected="$(readlink -f "$FLAGCX_BUILD_LIB/libflagcx.so" 2>/dev/null || true)"
  if [ -n "$resolved" ] && [ -n "$expected" ] && \
     [ "$(readlink -f "$resolved")" = "$expected" ]; then
    printf 'assertion=PASS libflagcx -> %s\n' "$resolved" \
      >>"$LOG_DIR/$name.status"
    return 0
  fi
  printf 'assertion=FAIL libflagcx resolved=%s expected=%s\n' \
    "${resolved:-none}" "${expected:-none}" >>"$LOG_DIR/$name.status"
  warn "$name: stale/wrong libflagcx (resolved=${resolved:-none}, expected=${expected:-none})"
  return 1
}

# musa_device_count: number of MUSA devices visible to this process.
# MUSA_GMI_BIN allows the harness test to inject a deterministic count.
# The vendor CLI is not installed at a fixed path: some images ship a
# zero-byte /usr/bin/mthreads-gmi stub next to the real binary in /driver.
# Fall back to the DRM render nodes only when no CLI reports anything.
musa_device_count() {
  local bin out="" n=""
  if [ -n "${MUSA_GMI_BIN:-}" ]; then
    out="$("$MUSA_GMI_BIN" -L 2>/dev/null || true)"
    printf '%s\n' "$out" | grep -c '^GPU' || true
    return 0
  fi
  for bin in /driver/usr/bin/mthreads-gmi \
             "$(command -v mthreads-gmi 2>/dev/null || true)"; do
    [ -n "$bin" ] && [ -x "$bin" ] || continue
    out="$("$bin" -L 2>/dev/null || true)"
    n="$(printf '%s\n' "$out" | grep -c '^GPU' || true)"
    [ "${n:-0}" -gt 0 ] && break
  done
  if [ "${n:-0}" -eq 0 ]; then
    n="$(ls -1 /dev/dri/renderD* 2>/dev/null | wc -l)"
  fi
  printf '%s\n' "${n:-0}"
}

# assert_device_count <name> <required>
assert_device_count() {
  local name="$1" need="$2" have
  have="$(musa_device_count)"
  if [ "${have:-0}" -ge "$need" ]; then
    printf 'assertion=PASS device count %s >= %s\n' "$have" "$need" \
      >>"$LOG_DIR/$name.status"
    return 0
  fi
  printf 'assertion=FAIL device count %s < %s\n' "$have" "$need" \
    >>"$LOG_DIR/$name.status"
  warn "$name: $have MUSA device(s) visible, $need required"
  return 1
}

# assert_device_mapping <name> <per-node> <list0> <list1>
# The two logical nodes must each get per-node unique devices and must not
# overlap, otherwise two ranks silently share one GPU.
assert_device_mapping() {
  local name="$1" per="$2" list0="$3" list1="$4"
  local -a a0 a1
  IFS=',' read -r -a a0 <<<"$list0"
  IFS=',' read -r -a a1 <<<"$list1"
  local ok=1 dups
  [ "${#a0[@]}" -eq "$per" ] && [ "${#a1[@]}" -eq "$per" ] || ok=0
  dups=$(printf '%s\n' "${a0[@]}" "${a1[@]}" | sort | uniq -d | wc -l)
  [ "$dups" -eq 0 ] || ok=0
  if [ "$ok" -eq 1 ]; then
    printf 'assertion=PASS device mapping per-node=%s (%s | %s)\n' \
      "$per" "$list0" "$list1" >>"$LOG_DIR/$name.status"
    return 0
  fi
  printf 'assertion=FAIL device mapping per-node=%s lists=%s|%s duplicates=%s\n' \
    "$per" "$list0" "$list1" "$dups" >>"$LOG_DIR/$name.status"
  warn "$name: bad device mapping ($list0 | $list1)"
  return 1
}
