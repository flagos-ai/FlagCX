#!/usr/bin/env bash
# T4: unified inter-node IR suite (INTER + WORLD, full coop matrices) using
# two logical nodes (2 x 4 ranks) on a single physical host.
#
# Usage: run_unified_inter.sh [--allow-blocked] [-- <extra args>]

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

NAME=run_unified_inter
begin_status_record "$NAME"
ensure_net_env
NODE0_DEVICES="${INTER_NODE0_DEVICES:-0,1,2,3}"
NODE1_DEVICES="${INTER_NODE1_DEVICES:-4,5,6,7}"
read -r -a ENV0 <<<"$(mpi_env_flags node0 "$NODE0_DEVICES")"
read -r -a ENV1 <<<"$(mpi_env_flags node1 "$NODE1_DEVICES")"

TOTAL_RANKS=$((INTER_NODE_RANKS * 2))
if ! assert_device_mapping "$NAME" "$INTER_NODE_RANKS" \
     "$NODE0_DEVICES" "$NODE1_DEVICES"; then
  record_blocked "$NAME" "invalid MUSA device mapping ($NODE0_DEVICES | $NODE1_DEVICES)"
  [ "$allow_blocked" -eq 1 ] && exit 0
  exit 2
fi
if ! assert_device_count "$NAME" "$TOTAL_RANKS"; then
  record_blocked "$NAME" "only $(musa_device_count) MUSA device(s) visible; $TOTAL_RANKS ranks required"
  [ "$allow_blocked" -eq 1 ] && exit 0
  exit 2
fi
if ! require_bin "$NAME" test_device_ir_unified_inter; then
  [ "$allow_blocked" -eq 1 ] && exit 0
  exit 2
fi
assert_runtime_lib "$NAME" "$BIN" || { set_status "$NAME" FAIL 1 0; exit 1; }

args=(-b "$MIN_BYTES" -e "$MAX_BYTES" -f "$STEP_FACTOR" "${extra[@]}")
run_capture "$NAME" "$RUN_TIMEOUT" \
  "$MPIRUN" "${MPI_ROOT_ARGS[@]}" \
  -n "$INTER_NODE_RANKS" "${ENV0[@]}" "$BIN" "${args[@]}" \
  : -n "$INTER_NODE_RANKS" "${ENV1[@]}" "$BIN" "${args[@]}"
rc=$?
raw_rc=$rc
verdict="$(classify_output "$NAME" "$rc")"
if [ "$verdict" = "PASS" ]; then
  # T4 is S16-S25 (unified IR inter). Every case must really print PASS on
  # every rank.
  local_rc=0
  for c in \
    'S16 DevBarrier\(INTER\+WORLD\)' \
    'S17 DevTeamResolution\(INTER\+WORLD\+INTRA-locality\)' \
    'S18 DevPut\+PutValue\(INTER\+WORLD\)' \
    'S19 DevGet\(INTER\+WORLD\)' \
    'S20 DevSignalStandalone\(INTER\+WORLD\)' \
    'S21 DevPutSignalWait\(INTER\+WORLD\)' \
    'S22 DevPut_RSig\(INTER\+WORLD\)' \
    'S23 DevPutCounter\(INTER\+WORLD\)' \
    'S24 DevPutValue_RSig\(INTER\+WORLD\)' \
    'S25 DevSignalShadowFlush\(INTER\+WORLD\)'; do
    require_case_pass "$NAME" "$TOTAL_RANKS" "$c.* PASS" "T4 $c" || local_rc=1
  done
  require_case_pass "$NAME" 1 '=== Final Result: ALL PASS ===' \
    "T4 final result" || local_rc=1
  # Require the topology we actually launched (2 logical nodes) and the real
  # transport, so a Socket fallback or a collapsed topology cannot pass.
  apply_node_assertions "$NAME" 2 "$INTER_NODE_RANKS" || local_rc=1
  assert_transport "$NAME" || local_rc=1
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
