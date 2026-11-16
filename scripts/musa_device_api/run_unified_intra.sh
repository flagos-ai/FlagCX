#!/usr/bin/env bash
# T3: unified intra-node IR suite (INTRA + WORLD, full coop matrices).
# 8 ranks, 1K..16M, factor 2, 30 min timeout.
#
# Usage: run_unified_intra.sh [--allow-blocked] [-- <extra args>]

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Unified IR windows are registered through the IPC/P2P path.  With P2P
# disabled flagcxSymWindowRegister must fall back to a network MR, which the
# MUSA Socket transport cannot provide, so registration returns
# flagcxNotSupported before any case runs.  Default P2P on for this gate only;
# common.sh keeps its historical default for the other suites.
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

NAME=run_unified_intra
begin_status_record "$NAME"
read -r -a ENVV <<<"$(mpi_env_flags)"

if ! assert_device_count "$NAME" "$NRANKS"; then
  record_blocked "$NAME" "only $(musa_device_count) MUSA device(s) visible; $NRANKS ranks required"
  [ "$allow_blocked" -eq 1 ] && exit 0
  exit 2
fi
if ! require_bin "$NAME" test_device_ir_unified_intra; then
  [ "$allow_blocked" -eq 1 ] && exit 0
  exit 2
fi
assert_runtime_lib "$NAME" "$BIN" || { set_status "$NAME" FAIL 1 0; exit 1; }

args=(-b "$MIN_BYTES" -e "$MAX_BYTES" -f "$STEP_FACTOR" "${extra[@]}")
run_capture "$NAME" "$RUN_TIMEOUT" \
  "$MPIRUN" "${MPI_ROOT_ARGS[@]}" -n "$NRANKS" "${ENVV[@]}" "$BIN" "${args[@]}"
rc=$?
raw_rc=$rc
verdict="$(classify_output "$NAME" "$rc")"
if [ "$verdict" = "PASS" ]; then
  # T3 is S16-S25 (unified IR intra). Every case must really print PASS on
  # every rank.
  local_rc=0
  for c in \
    'S16 DevBarrier\(INTRA\+WORLD\)' \
    'S17 DevTeamResolution\(INTRA\+WORLD\)' \
    'S18 DevPut\+PutValue\(INTRA\+WORLD\)' \
    'S19 DevGet\(INTRA\+WORLD\)' \
    'S20 DevSignalStandalone\(INTRA\+WORLD\)' \
    'S21 DevPutSignalWait\(INTRA\+WORLD\)' \
    'S22 DevPut_RSig\(INTRA\+WORLD\)' \
    'S23 DevPutCounter\(INTRA\+WORLD\)' \
    'S24 DevPutValue_RSig\(INTRA\+WORLD\)' \
    'S25 DevSignalShadowFlush\(INTRA\+WORLD\)'; do
    require_case_pass "$NAME" "$NRANKS" "$c.* PASS" "T3 $c" || local_rc=1
  done
  require_case_pass "$NAME" 1 '=== Final Result: ALL PASS ===' \
    "T3 final result" || local_rc=1
  # Single-node run: all ranks must be in one team of NRANKS.
  check_log_pattern "$NAME" "IntraSize: $NRANKS([^0-9]|$)" \
    "single-node intra size $NRANKS" || local_rc=1
  [ "$local_rc" -eq 0 ] || verdict=FAIL
fi
if [ "$raw_rc" -eq 124 ] || [ "$raw_rc" -eq 137 ]; then
  verdict=TIMEOUT
fi
printf 'verdict=%s\n' "$verdict" >>"$LOG_DIR/$NAME.status"
[ "$verdict" = "PASS" ] || rc=1
# run_capture only saw the exit code; replace its provisional status with the
# verdict so a zero-output exit-0 binary can never stay PASS.
finalize_status "$NAME" "$verdict" "$raw_rc"
exit "$rc"
