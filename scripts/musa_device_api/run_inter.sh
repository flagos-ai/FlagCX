#!/usr/bin/env bash
# T2: inter-node Device API / Device IR tests on MUSA using two logical nodes
# (2 x 4 ranks = 8 total) on a single physical host.
#
# This validates inter-node discovery, FLAGCX_HOSTID splitting and the
# window/signal relay path. It does NOT validate a real second physical node
# or a real NIC/IB transport.
#
# Usage: run_inter.sh [--allow-blocked] [-- <extra test args>]

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

NAME=run_inter
begin_status_record "$NAME"

ensure_net_env
NODE0_DEVICES="${INTER_NODE0_DEVICES:-0,1,2,3}"
NODE1_DEVICES="${INTER_NODE1_DEVICES:-4,5,6,7}"
read -r -a ENV0 <<<"$(mpi_env_flags node0 "$NODE0_DEVICES")"
read -r -a ENV1 <<<"$(mpi_env_flags node1 "$NODE1_DEVICES")"
common_args=(-b "$MIN_BYTES" -e "$MAX_BYTES" -f "$STEP_FACTOR")

rc_total=0
ran_any=0
timed_out=0
TOTAL_RANKS=$((INTER_NODE_RANKS * 2))

# T2 is K14 OneSidedAlltoAll.
assert_required_cases() {
  local tag="$1" rc=0
  case "$tag" in
    "$NAME.api")
      require_case_pass "$tag" "$TOTAL_RANKS" \
        'K14 OneSidedAlltoAll.* PASS' "T2 K14 OneSidedAlltoAll" || rc=1
      require_case_pass "$tag" "$TOTAL_RANKS" \
        '=== Overall: PASS ===' "T2 overall PASS" || rc=1
      ;;
    "$NAME.ir")
      require_case_pass "$tag" "$TOTAL_RANKS" \
        'S14 OneSidedAlltoAll: PASS' "T2-ir S14 OneSidedAlltoAll" || rc=1
      require_case_pass "$tag" "$TOTAL_RANKS" \
        '=== Overall: PASS ===' "T2-ir overall PASS" || rc=1
      ;;
  esac
  return "$rc"
}

run_one() {
  local tag="$1" bin="$2"; shift 2
  local args=("$@")
  log "running $tag: 2 x $INTER_NODE_RANKS ranks, $bin ${args[*]}"
  begin_status_record "$tag"
  assert_runtime_lib "$tag" "$bin" || { rc_total=1; return 1; }
  run_capture "$tag" "$RUN_TIMEOUT" \
    "$MPIRUN" "${MPI_ROOT_ARGS[@]}" \
    -n "$INTER_NODE_RANKS" "${ENV0[@]}" "$bin" "${args[@]}" \
    : -n "$INTER_NODE_RANKS" "${ENV1[@]}" "$bin" "${args[@]}"
  local rc=$?
  [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ] && timed_out=1
  local verdict
  verdict="$(classify_output "$tag" "$rc")"
  if [ "$verdict" = "PASS" ]; then
    assert_required_cases "$tag" || verdict=FAIL
    # An exit 0 that silently used Socket must not be reported as an RDMA run.
    assert_transport "$tag" || verdict=FAIL
  fi
  [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ] && verdict=TIMEOUT
  printf 'verdict=%s\n' "$verdict" >>"$LOG_DIR/$tag.status"
  # run_capture only saw the exit code; replace its provisional status with the
  # verdict so a zero-output exit-0 binary can never stay PASS.
  finalize_status "$tag" "$verdict" "$rc"
  [ "$verdict" = "PASS" ] || rc=1
  return "$rc"
}

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

if require_bin "$NAME.api" test_device_api_inter; then
  ran_any=1
  run_one "$NAME.api" "$BIN" "${common_args[@]}" -R "$REG_MODE" || rc_total=1
else
  rc_total=1
fi


if require_bin "$NAME.ir" test_device_ir_inter; then
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
