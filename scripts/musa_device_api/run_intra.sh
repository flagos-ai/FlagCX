#!/usr/bin/env bash
# T1: intra-node Device API / Device IR tests on MUSA.
# 8 ranks, 1K..16M, factor 2, -R 2 (window registration), 30 min timeout.
#
# Usage: run_intra.sh [--allow-blocked] [-- <extra test args>]

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This suite registers symmetric windows (flagcxCommWindowRegister).  With P2P
# disabled the registration must fall back to a network MR, which the MUSA
# Socket transport cannot provide, so it returns flagcxNotSupported before the
# first case runs.  Default P2P on for this gate; common.sh keeps its
# historical default for the host-only suites.
export FLAGCX_P2P_DISABLE="${FLAGCX_P2P_DISABLE:-0}"
# shellcheck source=common.sh
source "$HERE/common.sh"

allow_blocked=0
extra=()
while [ $# -gt 0 ]; do
  case "$1" in
    --allow-blocked) allow_blocked=1; shift ;;
    --) shift; extra=("$@"); break ;;
    -h|--help) echo "Usage: $0 [--allow-blocked] [-- extra args]"; exit 0 ;;
    *) extra+=("$1"); shift ;;
  esac
done

NAME=run_intra
begin_status_record "$NAME"

read -r -a ENVV <<<"$(mpi_env_flags)"
common_args=(-b "$MIN_BYTES" -e "$MAX_BYTES" -f "$STEP_FACTOR")

rc_total=0
ran_any=0
timed_out=0

# T1 is K10 IntraAllReduce. Requiring the official case line proves the case
# really ran on every rank instead of trusting exit code 0.
assert_required_cases() {
  local tag="$1" rc=0
  case "$tag" in
    "$NAME.api")
      require_case_pass "$tag" "$NRANKS" \
        'K10 IntraAllReduce\(composite\).* PASS' "T1 K10 IntraAllReduce" || rc=1
      require_case_pass "$tag" "$NRANKS" \
        '=== Overall: PASS ===' "T1 overall PASS" || rc=1
      ;;
    "$NAME.ir")
      # The IR suite prints "S10 IntraAllReduce(Scalar): PASS" (note the
      # single space before the label), so it needs its own pattern; without
      # this branch its status row carried no proof beyond the exit code.
      require_case_pass "$tag" "$NRANKS" \
        'S10 IntraAllReduce\(Scalar\).* PASS' "T1-ir S10 IntraAllReduce" || rc=1
      require_case_pass "$tag" "$NRANKS" \
        '=== Overall: PASS ===' "T1-ir overall PASS" || rc=1
      ;;
  esac
  return "$rc"
}

run_one() {
  local tag="$1" bin="$2"; shift 2
  local args=("$@")
  log "running $tag: $bin ${args[*]}"
  begin_status_record "$tag"
  assert_runtime_lib "$tag" "$bin" || { rc_total=1; return 1; }
  run_capture "$tag" "$RUN_TIMEOUT" \
    "$MPIRUN" "${MPI_ROOT_ARGS[@]}" -n "$NRANKS" "${ENVV[@]}" "$bin" "${args[@]}"
  local rc=$?
  [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ] && timed_out=1
  local verdict
  verdict="$(classify_output "$tag" "$rc")"
  if [ "$verdict" = "PASS" ]; then
    assert_required_cases "$tag" || verdict=FAIL
  fi
  [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ] && verdict=TIMEOUT
  printf 'verdict=%s\n' "$verdict" >>"$LOG_DIR/$tag.status"
  # run_capture only saw the exit code; replace its provisional status with the
  # verdict so a zero-output exit-0 binary can never stay PASS.
  finalize_status "$tag" "$verdict" "$rc"
  [ "$verdict" = "PASS" ] || rc=1
  return "$rc"
}

if ! assert_device_count "$NAME" "$NRANKS"; then
  record_blocked "$NAME" "only $(musa_device_count) MUSA device(s) visible; $NRANKS ranks required"
  [ "$allow_blocked" -eq 1 ] && exit 0
  exit 2
fi

if require_bin "$NAME.api" test_device_api_intra; then
  ran_any=1
  run_one "$NAME.api" "$BIN" "${common_args[@]}" -R "$REG_MODE" || rc_total=1
else
  rc_total=1
fi


if require_bin "$NAME.ir" test_device_ir_intra; then
  ran_any=1
  run_one "$NAME.ir" "$BIN" "${common_args[@]}" "${extra[@]}" || rc_total=1
fi

if [ "$ran_any" -eq 0 ]; then
  record_blocked "$NAME" "no runnable device test binary (device build blocked; owner A)"
  if [ "$allow_blocked" -eq 1 ]; then
    exit 0
  fi
  exit 2
fi

# Aggregate status. TIMEOUT must stay distinct from FAIL.
STAGE_DUR="$(total_duration "$NAME.api" "$NAME.ir")"
if [ "$timed_out" -eq 1 ]; then
  set_status "$NAME" TIMEOUT 124 "$STAGE_DUR"
  printf 'verdict=TIMEOUT\n' >>"$LOG_DIR/$NAME.status"
  exit 1
elif [ "$rc_total" -eq 0 ]; then
  set_status "$NAME" PASS 0 "$STAGE_DUR"
  printf 'verdict=PASS\n' >>"$LOG_DIR/$NAME.status"
else
  set_status "$NAME" FAIL "$rc_total" "$STAGE_DUR"
  printf 'verdict=FAIL\n' >>"$LOG_DIR/$NAME.status"
fi
exit "$rc_total"
