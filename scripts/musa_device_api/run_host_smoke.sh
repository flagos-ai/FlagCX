#!/usr/bin/env bash
# C-owned host-side lifecycle smoke on real MUSA hardware.
#
# Builds test/musa/host_lifecycle_smoke.cpp and runs it with
# NRANKS ranks. Validates:
#   flagcxDeviceHandleInit/getDeviceCount/setDevice
#   flagcxGetUniqueId + flagcxCommInitRank (real MCCL/bootstrap)
#   flagcxDevCommCreate(zero reqs) / Destroy  (real musaMalloc epoch buffer)
#   flagcxDevCommGetDevicePtr / FreeDevicePtr (real device materialization)
#   flagcxMemAlloc / flagcxMemFree
# and, with --probe-devmem, reports flagcxDevMemCreate(win=nullptr) behavior.
# With --probe-devptr it additionally gates DevMem Create -> GetDevicePtr ->
# FreeDevicePtr -> Destroy on 2 ranks.
# With --probe-barrier it reports the IPC barrier DevComm result, which must be
# flagcxNotSupported (not flagcxSystemError) when the platform lacks IPC.
# With --probe-window it gates symmetric window registration on 2 ranks: either
# a real window + deregister, or a clean flagcxNotSupported (no crash/hang).
#
# Usage: run_host_smoke.sh [--probe-devmem] [--probe-devptr] [--probe-barrier] [--probe-window] [--build-only]

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The zero-requirement devComm still establishes the intra-node P2P mesh
# (defaultDevApiCommCreate -> preconnectFullMesh), so this suite needs P2P on:
# with P2P disabled that mesh falls back to a network MR, which the MUSA Socket
# transport cannot provide, and creation fails. common.sh keeps its historical
# default for the host-only checks; the device suites set this themselves.
export FLAGCX_P2P_DISABLE="${FLAGCX_P2P_DISABLE:-0}"
# shellcheck source=common.sh
source "$HERE/common.sh"

probe=0
probe_devptr=0
probe_barrier=0
probe_window=0
churn=0
build_only=0
while [ $# -gt 0 ]; do
  case "$1" in
    --probe-devmem) probe=1; shift ;;
    --probe-devptr) probe_devptr=1; shift ;;
    --probe-barrier) probe_barrier=1; shift ;;
    --probe-window) probe_window=1; shift ;;
    --probe-churn) churn="${2:?-}"; shift 2 ;;
    --build-only) build_only=1; shift ;;
    --allow-blocked) shift ;;
    -h|--help) echo "Usage: $0 [--probe-devmem] [--probe-devptr] [--probe-barrier] [--probe-window] [--probe-churn N] [--build-only]"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
case "$churn" in
  ''|*[!0-9]*) die "--probe-churn expects a positive integer" ;;
esac

SMOKE_DIR="$RESULT_DIR/smoke"
mkdir -p "$SMOKE_DIR"
BIN="$SMOKE_DIR/host_lifecycle_smoke"
SRC="$PROJECT_ROOT/test/musa/host_lifecycle_smoke.cpp"

if [ ! -f "$FLAGCX_BUILD_LIB/libflagcx.so" ]; then
  record_blocked host_smoke_build "libflagcx.so missing; run build_all.sh first"
  exit 2
fi

# Prefer mpicxx so MPI flags match the installed MPI exactly.
if command -v mpicxx >/dev/null 2>&1; then
  read -r -a MPI_CC <<<"$(mpicxx --showme:compile 2>/dev/null)"
  read -r -a MPI_LD <<<"$(mpicxx --showme:link 2>/dev/null)"
  CXX_BIN=mpicxx
else
  if [ -f /usr/include/x86_64-linux-gnu/mpi/mpi.h ]; then
    MPI_CC=(-I/usr/include/x86_64-linux-gnu/mpi)
    MPI_LD=(-L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi)
  else
    MPI_CC=()
    MPI_LD=(-lmpi)
  fi
  CXX_BIN=g++
fi

log "compiling host lifecycle smoke"
run_capture host_smoke_build "$BUILD_TIMEOUT" \
  "$CXX_BIN" -std=c++17 -g -O2 -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX \
  "${MPI_CC[@]}" \
  -I"$PROJECT_ROOT/flagcx/include" \
  -I"$PROJECT_ROOT/flagcx/adaptor/include" \
  -I"$PROJECT_ROOT/flagcx/core/include" \
  -I"$PROJECT_ROOT/flagcx/service/include" \
  -I"$PROJECT_ROOT/third-party/json/single_include" \
  "$SRC" -o "$BIN" \
  -L"$FLAGCX_BUILD_LIB" -Wl,--no-as-needed -Wl,-rpath,"$FLAGCX_BUILD_LIB" \
  -lflagcx "${MPI_LD[@]}" || exit $?

if [ "$build_only" -eq 1 ]; then
  log "build-only: $BIN"
  exit 0
fi

read -r -a ENVV <<<"$(mpi_env_flags)"
log "running host lifecycle smoke on $NRANKS ranks"
run_capture host_smoke "$RUN_TIMEOUT" \
  "$MPIRUN" "${MPI_ROOT_ARGS[@]}" -n "$NRANKS" "${ENVV[@]}" "$BIN"
rc=$?

pass_count="$(grep -c 'HOST_LIFECYCLE_SMOKE: PASS' "$LOG_DIR/host_smoke.out" 2>/dev/null || true)"
printf 'verdict=%s\nranks_passed=%s\n' \
  "$([ "$rc" -eq 0 ] && [ "$pass_count" -eq "$NRANKS" ] && echo PASS || echo FAIL)" \
  "$pass_count" >>"$LOG_DIR/host_smoke.status"
if [ "$pass_count" -ne "$NRANKS" ]; then
  warn "host_smoke: only $pass_count/$NRANKS ranks reported PASS"
  rc=1
fi
if [ "$rc" -ne 0 ]; then
  write_status host_smoke FAIL "$rc" 0 "run_host_smoke.sh"
  exit "$rc"
fi
write_status host_smoke PASS 0 0 "run_host_smoke.sh"

if [ "$probe" -eq 1 ]; then
  log "probing flagcxDevMemCreate(win=nullptr) on 2 ranks"
  run_capture host_smoke_devmem_probe "$RUN_TIMEOUT" \
    "$MPIRUN" "${MPI_ROOT_ARGS[@]}" -n 2 "${ENVV[@]}" "$BIN" --probe-devmem || true
  # Record observed result codes without gating: this is a probe.
  grep -h 'PROBE flagcxDevMemCreate' "$LOG_DIR/host_smoke_devmem_probe.out" \
    >"$LOG_DIR/host_smoke_devmem_probe.result" 2>/dev/null || true
  cat "$LOG_DIR/host_smoke_devmem_probe.result" 2>/dev/null || true
fi

if [ "$probe_devptr" -eq 1 ]; then
  log "gating DevMem descriptor materialization on 2 ranks"
  run_capture host_smoke_devptr_probe "$RUN_TIMEOUT" \
    "$MPIRUN" "${MPI_ROOT_ARGS[@]}" -n 2 "${ENVV[@]}" "$BIN" --probe-devptr
  rc=$?
  grep -hE 'PROBE flagcxDevMem(Create|GetDevicePtr|FreeDevicePtr|Destroy)|INFO devMem device ptr' \
    "$LOG_DIR/host_smoke_devptr_probe.out" \
    >"$LOG_DIR/host_smoke_devptr_probe.result" 2>/dev/null || true
  cat "$LOG_DIR/host_smoke_devptr_probe.result" 2>/dev/null || true
  ok_count="$(grep -c 'PROBE flagcxDevMemGetDevicePtr *= 0 (flagcxSuccess)' \
    "$LOG_DIR/host_smoke_devptr_probe.out" 2>/dev/null || true)"
  pass_count="$(grep -c 'HOST_LIFECYCLE_SMOKE: PASS' \
    "$LOG_DIR/host_smoke_devptr_probe.out" 2>/dev/null || true)"
  if [ "$rc" -ne 0 ] || [ "$ok_count" -ne 2 ] || [ "$pass_count" -ne 2 ]; then
    warn "devptr probe failed (rc=$rc getDevicePtr_ok=$ok_count ranks_passed=$pass_count)"
    write_status host_smoke_devptr_probe FAIL "${rc:-1}" 0 \
      "run_host_smoke.sh --probe-devptr"
    exit 1
  fi
  write_status host_smoke_devptr_probe PASS 0 0 "run_host_smoke.sh --probe-devptr"
fi

if [ "$probe_window" -eq 1 ]; then
  log "probing symmetric window registration on 2 ranks"
  run_capture host_smoke_window_probe "$RUN_TIMEOUT" \
    "$MPIRUN" "${MPI_ROOT_ARGS[@]}" -n 2 "${ENVV[@]}" "$BIN" --probe-window
  rc=$?
  grep -hE 'PROBE flagcxCommWindow|INFO window registered' \
    "$LOG_DIR/host_smoke_window_probe.out" \
    >"$LOG_DIR/host_smoke_window_probe.result" 2>/dev/null || true
  cat "$LOG_DIR/host_smoke_window_probe.result" 2>/dev/null || true
  pass_count="$(grep -c 'HOST_LIFECYCLE_SMOKE: PASS' \
    "$LOG_DIR/host_smoke_window_probe.out" 2>/dev/null || true)"
  if [ "$rc" -ne 0 ] || [ "$pass_count" -ne 2 ]; then
    warn "window probe failed (rc=$rc ranks_passed=$pass_count)"
    write_status host_smoke_window_probe FAIL "${rc:-1}" 0 \
      "run_host_smoke.sh --probe-window"
    exit 1
  fi
  write_status host_smoke_window_probe PASS 0 0 "run_host_smoke.sh --probe-window"
fi

if [ "$probe_barrier" -eq 1 ]; then
  log "probing IPC barrier DevComm create on 2 ranks"
  run_capture host_smoke_barrier_probe "$RUN_TIMEOUT" \
    "$MPIRUN" "${MPI_ROOT_ARGS[@]}" -n 2 "${ENVV[@]}" "$BIN" --probe-barrier || true
  grep -h 'PROBE flagcxDevCommCreate(barrier)' \
    "$LOG_DIR/host_smoke_barrier_probe.out" \
    >"$LOG_DIR/host_smoke_barrier_probe.result" 2>/dev/null || true
  cat "$LOG_DIR/host_smoke_barrier_probe.result" 2>/dev/null || true
  # A capability probe must surface NotSupported, never a hard system error.
  if grep -q 'flagcxSystemError' "$LOG_DIR/host_smoke_barrier_probe.result" 2>/dev/null; then
    warn "barrier probe still reports flagcxSystemError (expected flagcxNotSupported)"
    write_status host_smoke_barrier_probe FAIL 1 0 "run_host_smoke.sh --probe-barrier"
    exit 1
  fi
  write_status host_smoke_barrier_probe PASS 0 0 "run_host_smoke.sh --probe-barrier"
fi

if [ "$churn" -gt 0 ]; then
  log "gating $churn create/destroy iterations (host alloc, window, DevComm) on 2 ranks"
  # The churn probe registers a real symmetric window.
  read -r -a CHURN_ENVV <<<"$(mpi_env_flags)"
  run_capture host_smoke_churn_probe "$RUN_TIMEOUT" \
    "$MPIRUN" "${MPI_ROOT_ARGS[@]}" -n 2 "${CHURN_ENVV[@]}" "$BIN" --probe-churn "$churn"
  rc=$?
  grep -hE 'CHURN' "$LOG_DIR/host_smoke_churn_probe.out" \
    >"$LOG_DIR/host_smoke_churn_probe.result" 2>/dev/null || true
  cat "$LOG_DIR/host_smoke_churn_probe.result" 2>/dev/null || true
  pass_count="$(grep -c 'HOST_LIFECYCLE_SMOKE: PASS' \
    "$LOG_DIR/host_smoke_churn_probe.out" 2>/dev/null || true)"
  churn_ok="$(grep -c 'CHURN .* 0 failure' \
    "$LOG_DIR/host_smoke_churn_probe.out" 2>/dev/null || true)"
  if [ "$rc" -ne 0 ] || [ "$pass_count" -ne 2 ] || [ "$churn_ok" -ne 2 ]; then
    warn "churn probe failed (rc=$rc clean=$churn_ok ranks_passed=$pass_count)"
    write_status host_smoke_churn_probe FAIL "${rc:-1}" 0 \
      "run_host_smoke.sh --probe-churn $churn"
    exit 1
  fi
  write_status host_smoke_churn_probe PASS 0 0 \
    "run_host_smoke.sh --probe-churn $churn"
fi

exit 0
