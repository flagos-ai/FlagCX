#!/usr/bin/env bash
# Standalone two-process / two-GPU MUSA IPC closed-loop gate.
#
# This exercises ONLY the MUSA device-adaptor IPC path
# (flagcx/adaptor/device/musa_adaptor.cc) exported through
# flagcxDeviceHandle:
#
#   ipcMemHandleCreate / Get / Open / Close / Free
#
# It deliberately does not create a FlagCX communicator or call any collective.
# The IPC handle travels over MPI between exactly two ranks pinned to two
# distinct GPUs (default 6 and 7).
#
# Cases asserted:
#   handle_roundtrip, peer_read_after_write, peer_write_visible,
#   reopen_cycles (>= 8), invalid_handle
#
# Status conventions follow scripts/musa_device_api/common.sh: a .status file
# with command/exit/duration, 124/137 mapped to TIMEOUT, and PASS only when the
# process exit code is 0 AND every expected output line was observed.
#
# Usage: run_musa_ipc_peer.sh [--dev-a N] [--dev-b N] [--cycles N] [--build-only]
# Env overrides: DEV_A, DEV_B, CYCLES, MUSA_ARCH, MUSA_HOME, RUN_TIMEOUT.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$(cd "$HERE/.." && pwd)"
# shellcheck source=../common.sh
source "$SCRIPTS/common.sh"

TEST_NAME="musa_ipc_peer"
DEV_A="${DEV_A:-6}"
DEV_B="${DEV_B:-7}"
CYCLES="${CYCLES:-8}"
MUSA_ARCH="${MUSA_ARCH:-mp_31}"
MUSA_HOME="${MUSA_HOME:-/usr/local/musa}"
IPC_BUILD_DIR="${IPC_BUILD_DIR:-$RESULT_DIR/ipc_peer}"
IPC_BUILD_DIR="$(readlink -m "$IPC_BUILD_DIR")"
BIN="${BIN:-$IPC_BUILD_DIR/bin/musa_ipc_peer}"
BIN="$(readlink -m "$BIN")"

build_only=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dev-a)
      [ $# -ge 2 ] || die "--dev-a requires a value"
      DEV_A="$2"; shift 2 ;;
    --dev-b)
      [ $# -ge 2 ] || die "--dev-b requires a value"
      DEV_B="$2"; shift 2 ;;
    --cycles)
      [ $# -ge 2 ] || die "--cycles requires a value"
      CYCLES="$2"; shift 2 ;;
    --build-only) build_only=1; shift ;;
    --allow-blocked) shift ;;
    -h|--help)
      echo "Usage: $0 [--dev-a N] [--dev-b N] [--cycles N] [--build-only]"
      exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

is_uint() { case "$1" in ''|*[!0-9]*) return 1 ;; *) return 0 ;; esac; }
for v in "$DEV_A" "$DEV_B" "$CYCLES"; do
  is_uint "$v" || die "device/cycle values must be non-negative integers (got '$v')"
done
DEV_A=$((10#$DEV_A))
DEV_B=$((10#$DEV_B))
CYCLES=$((10#$CYCLES))
[ "$DEV_A" -ne "$DEV_B" ] || die "DEV_A and DEV_B must differ (both $DEV_A)"
[ "$CYCLES" -ge 8 ] || die "CYCLES must be >= 8 (got $CYCLES)"

NEED_DEVS=$(( (DEV_A > DEV_B ? DEV_A : DEV_B) + 1 ))

if [ ! -f "$FLAGCX_BUILD_LIB/libflagcx.so" ]; then
  record_blocked "$TEST_NAME" "libflagcx.so missing under $FLAGCX_BUILD_LIB; run build_all.sh first"
  exit 2
fi

# Best-effort device-count assertion before spending build time. If the MUSA
# query tool is absent we warn and let the in-process check decide.
if [ -n "${MUSA_GMI_BIN:-}" ] || command -v mthreads-gmi >/dev/null 2>&1; then
  HAVE_DEVS="$(musa_device_count)"
  if [ "${HAVE_DEVS:-0}" -gt 0 ] && [ "${HAVE_DEVS:-0}" -lt "$NEED_DEVS" ]; then
    record_blocked "$TEST_NAME" \
      "need at least $NEED_DEVS MUSA devices for device $DEV_A/$DEV_B, only ${HAVE_DEVS:-0} visible"
    exit 2
  fi
  if [ "${HAVE_DEVS:-0}" -eq 0 ]; then
    warn "mthreads-gmi did not report a usable GPU count; device assertion deferred to the test"
  else
    log "MUSA devices visible: $HAVE_DEVS (using $DEV_A and $DEV_B)"
  fi
else
  warn "mthreads-gmi/MUSA_GMI_BIN unavailable; device-count assertion deferred to the test"
fi

log "building MUSA IPC peer test (arch=$MUSA_ARCH)"
run_capture "${TEST_NAME}_build" "$BUILD_TIMEOUT" \
  make -C "$PROJECT_ROOT/test/musa" -f musa_ipc_peer.mk \
  BUILD_DIR="$IPC_BUILD_DIR" BIN="$BIN" \
  MUSA_HOME="$MUSA_HOME" MUSA_ARCH="$MUSA_ARCH" \
  FLAGCX_BUILD_LIB="$FLAGCX_BUILD_LIB"
build_rc=$?
if [ "$build_rc" -ne 0 ]; then
  warn "$TEST_NAME: build failed (exit=$build_rc); see $LOG_DIR/${TEST_NAME}_build.err"
  exit "$build_rc"
fi
printf 'verdict=PASS\n' >>"$LOG_DIR/${TEST_NAME}_build.status"

RESOLVED_LIB="$(ldd "$BIN" 2>/dev/null | awk '/libflagcx\.so/ {print $3; exit}')"
log "$TEST_NAME: libflagcx -> ${RESOLVED_LIB:-none}"

if [ "$build_only" -eq 1 ]; then
  log "build-only: $BIN"
  exit 0
fi

if [ "$MPI_FLAVOR" = "openmpi" ]; then
  DEV_ENVV=(-x "TEST_DEV_A=$DEV_A" -x "TEST_DEV_B=$DEV_B"
            -x "IPC_PEER_CYCLES=$CYCLES" -x "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
            -x "FLAGCX_DEVICE_ADAPTOR_PLUGIN=none"
            -x "FLAGCX_CCL_ADAPTOR_PLUGIN=none")
else
  DEV_ENVV=(--genv "TEST_DEV_A=$DEV_A" --genv "TEST_DEV_B=$DEV_B"
            --genv "IPC_PEER_CYCLES=$CYCLES" --genv "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
            --genv "FLAGCX_DEVICE_ADAPTOR_PLUGIN=none"
            --genv "FLAGCX_CCL_ADAPTOR_PLUGIN=none")
fi

log "running $TEST_NAME on 2 ranks (devices $DEV_A/$DEV_B, cycles=$CYCLES)"
run_capture "$TEST_NAME" "$RUN_TIMEOUT" \
  "$MPIRUN" "${MPI_ROOT_ARGS[@]}" -n 2 "${DEV_ENVV[@]}" "$BIN"
rc=$?

# A nonzero exit can never be PASS. Preserve run_capture's TIMEOUT record.
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  warn "$TEST_NAME: TIMEOUT after ${RUN_TIMEOUT}s"
  exit "$rc"
fi
if [ "$rc" -ne 0 ]; then
  warn "$TEST_NAME: test exited $rc (see $LOG_DIR/$TEST_NAME.err)"
  exit "$rc"
fi

OUT="$LOG_DIR/$TEST_NAME.out"
ST="$LOG_DIR/$TEST_NAME.status"
assertions_ok=1

append_status() { printf '%s\n' "$*" >>"$ST"; }

require_line() { # require_line <extended-regex> <description>
  local pattern="$1" desc="$2"
  if grep -Eq -- "$pattern" "$OUT" 2>/dev/null; then
    append_status "assertion=PASS $desc"
    return 0
  fi
  append_status "assertion=FAIL $desc (pattern: $pattern)"
  warn "$TEST_NAME: missing $desc (pattern: $pattern)"
  assertions_ok=0
  return 1
}

# Exact expected lines: the RESULT line pins the rank->device mapping, so a
# silently wrong GPU selection cannot pass.
require_line "^IPC_PEER rank=0 dev=${DEV_A} RESULT=PASS checks=5$" \
  "rank 0 device $DEV_A RESULT=PASS checks=5"
require_line "^IPC_PEER rank=1 dev=${DEV_B} RESULT=PASS checks=5$" \
  "rank 1 device $DEV_B RESULT=PASS checks=5"

# Every named case must PASS once per rank, on the exact device that rank was
# pinned to: one CASE line per rank, so a case silently attributed to the wrong
# GPU cannot satisfy the requirement.
for case_name in handle_roundtrip peer_read_after_write peer_write_visible \
                 reopen_cycles invalid_handle; do
  for pair in "0:$DEV_A" "1:$DEV_B"; do
    r="${pair%%:*}"
    d="${pair##*:}"
    if grep -Eq \
      "^IPC_PEER rank=$r dev=$d CASE=${case_name} PASS( cycles=[0-9]+)?$" \
      "$OUT" 2>/dev/null; then
      append_status "case=PASS $case_name (rank=$r dev=$d)"
    else
      append_status "case=FAIL $case_name (rank=$r dev=$d missing)"
      warn "$TEST_NAME: CASE $case_name missing for rank $r on device $d"
      assertions_ok=0
    fi
  done
done

FAIL_HITS="$(grep -Ec 'CASE=[a-z_]+ FAIL|RESULT=FAIL' "$OUT" 2>/dev/null || true)"
if [ "${FAIL_HITS:-0}" -ne 0 ]; then
  append_status "assertion=FAIL failure markers present (hits=$FAIL_HITS)"
  warn "$TEST_NAME: $FAIL_HITS failure marker(s) in output"
  assertions_ok=0
else
  append_status "assertion=PASS no failure markers"
fi

if assert_runtime_lib "$TEST_NAME" "$BIN"; then
  :
else
  assertions_ok=0
fi
append_status "libflagcx=${RESOLVED_LIB:-none}"

DUR="$(awk -F= '/^duration_s=/ {print $2; exit}' "$ST" 2>/dev/null || true)"
if [ "$assertions_ok" -eq 1 ]; then
  append_status "verdict=PASS"
  log "$TEST_NAME: PASS (device $DEV_A/$DEV_B, libflagcx=${RESOLVED_LIB:-none})"
  exit 0
fi

set_status "$TEST_NAME" FAIL 1 "${DUR:-0}"
append_status "verdict=FAIL"
warn "$TEST_NAME: FAIL (assertions did not all pass)"
exit 1
