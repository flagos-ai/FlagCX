#!/usr/bin/env bash
# Build all C-scope artifacts and classify the device-kernel build honestly.
#
# Stages:
#   1. host library        make USE_MUSA=1
#   2. host-only unit test make -C test/unittest/device_api unit USE_MUSA=1
#   3. device/kernel tests make -C test/unittest/device_api mpi USE_MUSA=1 COMPILE_KERNEL=1
#
# Stage 3 is expected to be BLOCKED while makefiles/musa.mk lacks
# DEVICE_PLATFORM / DEVICE_COMPILER / test kernels (owner: A). BLOCKED is
# distinct from FAIL and never counted as PASS.
#
# Usage: build_all.sh [--allow-blocked]

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

allow_blocked=0
for arg in "$@"; do
  case "$arg" in
    --allow-blocked) allow_blocked=1 ;;
    -h|--help) echo "Usage: $0 [--allow-blocked]"; exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

SUMMARY="$RESULT_DIR/build_summary.env"
mkdir -p "$RESULT_DIR"
: >"$SUMMARY"

record_summary() { printf '%s=%s\n' "$1" "$2" >>"$SUMMARY"; }

log "== Stage 1: host library build (USE_MUSA=1) =="
host_rc=0
run_capture build_host "$BUILD_TIMEOUT" \
  make -C "$PROJECT_ROOT" USE_MUSA=1 -j"$BUILD_JOBS" || host_rc=$?
if [ "$host_rc" -ne 0 ]; then
  record_summary BUILD_HOST FAIL
  record_summary BUILD_UNIT SKIP
  record_summary BUILD_DEVICE SKIP
  log "host build failed; aborting build_all"
  exit "$host_rc"
fi
if [ ! -f "$FLAGCX_BUILD_LIB/libflagcx.so" ]; then
  record_summary BUILD_HOST FAIL
  record_summary BUILD_UNIT SKIP
  record_summary BUILD_DEVICE SKIP
  die "host build exited 0 but $FLAGCX_BUILD_LIB/libflagcx.so is missing"
fi
record_summary BUILD_HOST PASS
record_summary BUILD_HOST_LIB "$FLAGCX_BUILD_LIB/libflagcx.so"

log "== Stage 2: host-only unit tests build =="
unit_rc=0
run_capture build_unit "$BUILD_TIMEOUT" \
  make -C "$PROJECT_ROOT/test/unittest/device_api" unit USE_MUSA=1 -j"$BUILD_JOBS" || unit_rc=$?
if [ "$unit_rc" -ne 0 ]; then
  record_summary BUILD_UNIT FAIL
else
  record_summary BUILD_UNIT PASS
fi

log "== Stage 3: device/kernel tests build (COMPILE_KERNEL=1) =="
dev_rc=0
run_capture build_device "$BUILD_TIMEOUT" \
  make -C "$PROJECT_ROOT/test/unittest/device_api" mpi USE_MUSA=1 COMPILE_KERNEL=1 -j"$BUILD_JOBS" || dev_rc=$?
if [ "$dev_rc" -eq 0 ]; then
  record_summary BUILD_DEVICE PASS
else
  # Distinguish the known external block from a genuine regression.
  if grep -qE 'sets no DEVICE_PLATFORM|has no DEVICE_COMPILER|no test kernels for DEVICE_PLATFORM|missing test/kernel' \
       "$LOG_DIR/build_device.out" "$LOG_DIR/build_device.err" 2>/dev/null; then
    record_summary BUILD_DEVICE BLOCKED
    record_summary BUILD_DEVICE_REASON "makefiles/musa.mk lacks DEVICE_PLATFORM/DEVICE_COMPILER/test kernels (owner: A)"
    record_blocked build_device "device build blocked by makefiles/musa.mk (DEVICE_PLATFORM/DEVICE_COMPILER/test kernels missing; owner A)"
  else
    record_summary BUILD_DEVICE FAIL
  fi
fi

echo
log "== Build summary ($SUMMARY) =="
sed 's/^/  /' "$SUMMARY"

# Exit policy: host/unit failures are hard failures. Device BLOCKED is only
# tolerated with --allow-blocked, and is still recorded as BLOCKED, never PASS.
rc=0
[ "$(grep -c '=FAIL' "$SUMMARY")" -gt 0 ] && rc=1
if [ "$(grep -c 'BUILD_DEVICE=BLOCKED' "$SUMMARY")" -gt 0 ] && [ "$allow_blocked" -eq 0 ]; then
  rc=2
fi
exit "$rc"
