#!/usr/bin/env bash
# Run the unchanged public Unified IR Intra suite (T3), single node,
# INTRA + WORLD teams, S16-S25, 1K-16M factor 2 (15 sizes x 8 ranks).
#
# T3 = test_device_ir_unified_intra. Host driver, device kernels and the
# PASS/FAIL rules are not modified by this script.
#
# Two modes, deliberately separated:
#
#   MODE=official (default)
#     Exactly the command from 《FlagCX厂商适配（v3）》13.3.5: 8 ranks, only the
#     public baseline environment (FLAGCX_USE_HETERO_COMM=1, FLAGCX_VMM_ENABLE=0,
#     LD_LIBRARY_PATH), -b 1K -e 16M -f 2, 30 min timeout, 3 rounds.
#     MPICH/Hydra would use --genv; this box runs Open MPI, so 13.3.7 applies
#     and -x is the official equivalent (same default as
#     test/unittest/device_api/Makefile's MPI_FLAVOR=openmpi).
#     Use this mode to decide whether T3 passes.
#
#   MODE=musa
#     The MUSA diagnostic variant: adds the vendor runtime variables
#     (IB/MCCL off, DMA-BUF, MEM_ENABLE, host-mapped signal plane, rank CTA
#     rendezvous, INFO logging). It is NOT the official baseline; it exists to
#     compare against the 2-node T4 S20 failure.
#
# Note: `make -C test/unittest/device_api run-mpi-unified-intra` is the
# in-repo developer loop and uses -n 4; it is not the T3 acceptance form.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$root"
mode=${MODE:-official}
device_home=${DEVICE_HOME:-/usr/local/musa}
arch=${MUSA_ARCH:-mp_31}
build=${BUILD:-0}
case "$mode" in
  official) rounds=${ROUNDS:-3} ;;
  musa) rounds=${ROUNDS:-1} ;;
  *)
    echo "ERROR: MODE must be official or musa, got '$mode'" >&2
    exit 2
    ;;
esac
if [[ ! "$rounds" =~ ^[1-9][0-9]*$ ]] || [[ "$build" != 0 && "$build" != 1 ]]; then
  echo 'ERROR: ROUNDS must be positive; BUILD must be 0 or 1.' >&2
  exit 2
fi

out=$(mktemp -d "$root/test/unittest/device_api/t3-results-$(date +%Y%m%d_%H%M%S).XXXXXX")
echo "MODE=$mode OUTPUT=$out"
trap 'rc=$?; if (( rc != 0 )); then echo "T3 stopped: exit=$rc OUTPUT=$out" >&2; fi' EXIT

export LD_LIBRARY_PATH="$root/build/lib:$device_home/lib:$device_home/lib64:${LD_LIBRARY_PATH:-}"
unset LD_DEBUG LD_DEBUG_OUTPUT FLAGCX_DEBUG_FILE

# Common baseline shared by both modes (13.3.2 / 13.3.5).
export FLAGCX_USE_HETERO_COMM=1 FLAGCX_VMM_ENABLE=0

mpi_env=(-x LD_LIBRARY_PATH -x FLAGCX_USE_HETERO_COMM -x FLAGCX_VMM_ENABLE)
mpi_flags=(--allow-run-as-root)

if [[ "$mode" == musa ]]; then
  # Vendor runtime variables for this platform. Single node: FlagCX/MCCL IB off.
  export FLAGCX_MEM_ENABLE=1 FLAGCX_IB_DISABLE=1 FLAGCX_P2P_DISABLE=0
  export FLAGCX_DMABUF_ENABLE=1
  # The experiment: host-mapped signal plane (0 reproduces the earlier T3 pass).
  export FLAGCX_SIGNAL_HOST_ENABLE=${FLAGCX_SIGNAL_HOST_ENABLE:-1}
  # Rank-internal CTA rendezvous: validated fix for the reset-vs-write race.
  export FLAGCX_MUSA_FORCE_CTA_SYNC=${FLAGCX_MUSA_FORCE_CTA_SYNC:-1}
  export FLAGCX_DEVICE_ONE_SIDED_FORCE_NET=${FLAGCX_DEVICE_ONE_SIDED_FORCE_NET:-0}
  export MCCL_TOPO_ENHANCE=0 MCCL_IB_DISABLE=1
  # INFO keeps the signal-plane trace (sigSize / one-sided paths / CTA sync).
  export FLAGCX_DEBUG=${DEBUG:-INFO} FLAGCX_DEBUG_SUBSYS=${DEBUG_SUBSYS:-ALL}
  mpi_env+=(-x FLAGCX_MEM_ENABLE -x FLAGCX_IB_DISABLE -x FLAGCX_P2P_DISABLE
            -x FLAGCX_DMABUF_ENABLE -x FLAGCX_SIGNAL_HOST_ENABLE
            -x FLAGCX_MUSA_FORCE_CTA_SYNC -x FLAGCX_DEVICE_ONE_SIDED_FORCE_NET
            -x MCCL_TOPO_ENHANCE -x MCCL_IB_DISABLE
            -x FLAGCX_DEBUG -x FLAGCX_DEBUG_SUBSYS)
  mpi_flags+=(--bind-to none --tag-output)
  if [[ -v MUSA_VISIBLE_DEVICES ]]; then
    mpi_env+=(-x MUSA_VISIBLE_DEVICES)
  fi
fi

# Optional native-error interposer (musaStreamSynchronize wrapper), see
# docs/MUSA_T4失败分析_2026-11-14.md: PRELOAD=/path/to/runtime_errors.so
if [[ -n "${PRELOAD:-}" ]]; then
  mpi_env+=(-x "LD_PRELOAD=$PRELOAD")
fi

{
  git rev-parse HEAD 2>/dev/null || true
  git diff --stat 2>/dev/null || true
  printf 'root=%s\nmode=%s\narch=%s\ndevice_home=%s\nrounds=%s\nbuild=%s\n' \
    "$root" "$mode" "$arch" "$device_home" "$rounds" "$build"
  env | sort | sed -n '/^FLAGCX_/p; /^MCCL_/p; /^LD_LIBRARY_PATH=/p; /^MUSA_/p'
} > "$out/environment.txt"

bin="$root/test/unittest/device_api/build/bin/test_device_ir_unified_intra"
if [[ "$build" == 1 ]]; then
  timeout --kill-after=5s 600s make -B -j1 \
    USE_MUSA=1 COMPILE_KERNEL=1 DEVICE_HOME="$device_home" MUSA_ARCH="$arch" \
    > "$out/library-build.log" 2>&1
  timeout --kill-after=5s 600s make -B -C test/kernel/musa device_ir.o \
    PROJECT_ROOT="$root" USE_MUSA=1 COMPILE_KERNEL=1 \
    DEVICE_HOME="$device_home" MUSA_ARCH="$arch" \
    > "$out/kernel-build.log" 2>&1
  timeout --kill-after=5s 600s make -B -C test/unittest/device_api "$bin" \
    PROJECT_ROOT="$root" USE_MUSA=1 COMPILE_KERNEL=1 \
    DEVICE_HOME="$device_home" MUSA_ARCH="$arch" \
    > "$out/host-build.log" 2>&1
fi
if [[ ! -x "$bin" ]]; then
  echo "ERROR: missing executable $bin" >&2
  echo "Run once with BUILD=1 to build library, device_ir.o and this binary." >&2
  exit 2
fi
if [[ ! -e "$root/build/lib/libflagcx.so" ]]; then
  echo "ERROR: missing $root/build/lib/libflagcx.so (use BUILD=1)" >&2
  exit 2
fi

for (( round=1; round<=rounds; round++ )); do
  log="$out/t3-round$round.log"
  cmd=(mpirun "${mpi_flags[@]}" -np 8 "${mpi_env[@]}"
       "$bin" -b 1K -e 16M -f 2)
  printf '%q ' timeout --signal=TERM --kill-after=60s 30m "${cmd[@]}" \
    > "$out/command$round.txt"
  printf '\n' >> "$out/command$round.txt"
  echo "RUN T3 round $round: $log"
  rc=0
  timeout --signal=TERM --kill-after=60s 30m "${cmd[@]}" > "$log" 2>&1 || rc=$?
  echo "$rc" > "$out/t3-round$round.rc"

  # Self-evidence of the effective configuration, so a later reader does not
  # have to infer switches from log side effects.
  {
    echo "--- round $round (mode=$mode) ---"
    grep -E 'setupHostSignals\(SHM\)|signalBuffer sigSize=|one-sided paths' \
      "$log" | sed -e 's/^\[[0-9,]*\]<std[^>]*>://' | sort -u || true
    if grep -q 'CTA rendezvous enabled' "$log"; then
      echo 'CTA_SYNC=on'
    elif grep -q 'epochBuffer' "$log"; then
      echo 'CTA_SYNC=off'
    else
      echo 'CTA_SYNC=unknown (INFO logging not enabled)'
    fi
    grep -E 'RUNTIME_DIAG|illegal memory access|signal 11|Segmentation' "$log" \
      | sed -e 's/^\[[0-9,]*\]<std[^>]*>://' | sort -u || true
  } >> "$out/trace.txt"

  if (( rc != 0 )); then
    echo "FAIL: MPI exit=$rc"
    tail -40 "$log"
    exit "$rc"
  fi
  final=$(grep -c '=== Final Result: ALL PASS ===' "$log" || true)
  bad=$(grep -Ec 'S(1[6-9]|2[0-5]).*: (FAIL|SKIP)' "$log" || true)
  valid=1
  for (( s=16; s<=25; s++ )); do
    count=$(grep -Ec "S$s .*: PASS" "$log" || true)
    printf 'round=%s case=S%s pass=%s expected=120\n' "$round" "$s" "$count" \
      >> "$out/results.txt"
    if (( count != 120 )); then valid=0; fi
  done
  if (( final != 1 || bad != 0 || valid != 1 )); then
    echo "FAIL: incomplete coverage, final=$final FAIL/SKIP=$bad"
    grep -E 'S(1[6-9]|2[0-5]) .*: (FAIL|SKIP)' "$log" | head -20 || true
    tail -40 "$log"
    exit 1
  fi
  echo "T3 round $round PASS" | tee -a "$out/results.txt"
done
echo "T3_RESULT=PASS mode=$mode rounds=$rounds" | tee -a "$out/results.txt"
echo "OUTPUT=$out"
