#!/usr/bin/env bash
# Run the unchanged public Unified IR Inter suite with two logical nodes.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$root"
device_home=${DEVICE_HOME:-/usr/local/musa}
arch=${MUSA_ARCH:-mp_31}
rounds=${ROUNDS:-3}
build=${BUILD:-1}
sync_d2h=${FLAGCX_MUSA_SYNC_D2H:-0}
if [[ ! "$rounds" =~ ^[1-9][0-9]*$ ]] || [[ "$build" != 0 && "$build" != 1 ]]; then
  echo 'ERROR: ROUNDS must be positive; BUILD must be 0 or 1.' >&2
  exit 2
fi
if [[ "$sync_d2h" != 0 && "$sync_d2h" != 1 ]]; then
  echo 'ERROR: FLAGCX_MUSA_SYNC_D2H must be 0 or 1.' >&2
  exit 2
fi

out=$(mktemp -d "$root/test/unittest/device_api/t4-results-$(date +%Y%m%d_%H%M%S).XXXXXX")
echo "OUTPUT=$out"
trap 'rc=$?; if (( rc != 0 )); then echo "T4 stopped: exit=$rc OUTPUT=$out" >&2; fi' EXIT
export LD_LIBRARY_PATH="$root/build/lib:$device_home/lib:$device_home/lib64:${LD_LIBRARY_PATH:-}"
unset LD_PRELOAD LD_DEBUG LD_DEBUG_OUTPUT FLAGCX_DEBUG_FILE
unset FLAGCX_MUSA_IR_DIAG FLAGCX_MUSA_K1_DIAG FLAGCX_MUSA_IR_CASE
unset FLAGCX_DEVICE_ADAPTOR_PLUGIN

export FLAGCX_USE_HETERO_COMM=1 FLAGCX_MEM_ENABLE=1 FLAGCX_VMM_ENABLE=0
export FLAGCX_IB_DISABLE=0 FLAGCX_P2P_DISABLE=0
export FLAGCX_DMABUF_ENABLE=1 FLAGCX_SIGNAL_HOST_ENABLE=1
# The rank-internal CTA rendezvous is the validated fix for the reset-vs-write
# race that intermittently hangs the one-sided signal cases, so the acceptance
# run keeps it on, matching the T2 handoff configuration. Set
# FLAGCX_MUSA_FORCE_CTA_SYNC=0 in the caller's environment only for A/B runs.
export FLAGCX_MUSA_FORCE_CTA_SYNC=${FLAGCX_MUSA_FORCE_CTA_SYNC:-1}
export FLAGCX_DEVICE_ONE_SIDED_FORCE_NET=${FLAGCX_DEVICE_ONE_SIDED_FORCE_NET:-0}
export MCCL_TOPO_ENHANCE=0 MCCL_IB_DISABLE=1
export FLAGCX_DEBUG=INFO FLAGCX_DEBUG_SUBSYS=ALL

{
  git rev-parse HEAD 2>/dev/null || true
  git diff --stat 2>/dev/null || true
  printf 'root=%s\narch=%s\ndevice_home=%s\nrounds=%s\nbuild=%s\n' \
    "$root" "$arch" "$device_home" "$rounds" "$build"
  env | sort | sed -n '/^FLAGCX_/p; /^MCCL_/p; /^LD_LIBRARY_PATH=/p'
} > "$out/environment.txt"

bin="$root/test/unittest/device_api/build/bin/test_device_ir_unified_inter"
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
  exit 2
fi

mpi_env=()
for name in LD_LIBRARY_PATH FLAGCX_USE_HETERO_COMM FLAGCX_MEM_ENABLE \
  FLAGCX_VMM_ENABLE FLAGCX_IB_DISABLE FLAGCX_P2P_DISABLE \
  FLAGCX_DMABUF_ENABLE FLAGCX_SIGNAL_HOST_ENABLE FLAGCX_MUSA_FORCE_CTA_SYNC \
  FLAGCX_DEVICE_ONE_SIDED_FORCE_NET MCCL_TOPO_ENHANCE MCCL_IB_DISABLE \
  FLAGCX_DEBUG FLAGCX_DEBUG_SUBSYS; do
  mpi_env+=(-x "$name")
done
# Network choices depend on the actual machine. Forward explicit user choices.
for name in FLAGCX_IB_HCA FLAGCX_IB_GID_INDEX FLAGCX_SOCKET_IFNAME \
  FLAGCX_MUSA_SYNC_D2H; do
  if [[ -v "$name" ]]; then mpi_env+=(-x "$name"); fi
done

for (( round=1; round<=rounds; round++ )); do
  log="$out/t4-round$round.log"
  cmd=(mpirun --allow-run-as-root --bind-to none --tag-output
    -np 4 "${mpi_env[@]}" -x FLAGCX_HOSTID=node0 -x NCCL_HOSTID=node0
    "$bin" -b 1K -e 16M -f 2
    : -np 4 "${mpi_env[@]}" -x FLAGCX_HOSTID=node1 -x NCCL_HOSTID=node1
    "$bin" -b 1K -e 16M -f 2)
  printf '%q ' timeout --signal=TERM --kill-after=60s 30m "${cmd[@]}" > "$out/command$round.txt"
  printf '\n' >> "$out/command$round.txt"
  echo "RUN T4 round $round: $log"
  rc=0
  timeout --signal=TERM --kill-after=60s 30m "${cmd[@]}" > "$log" 2>&1 || rc=$?
  echo "$rc" > "$out/t4-round$round.rc"
  if (( rc != 0 )); then
    tail -60 "$log"
    exit "$rc"
  fi
  final=$(grep -c '=== Final Result: ALL PASS ===' "$log" || true)
  bad=$(grep -Ec 'S(1[6-9]|2[0-5]).*: (FAIL|SKIP)' "$log" || true)
  valid=1
  for (( s=16; s<=25; s++ )); do
    count=$(grep -Ec "S$s .*: PASS" "$log" || true)
    printf 'round=%s case=S%s pass=%s expected=120\n' "$round" "$s" "$count" >> "$out/results.txt"
    if (( count != 120 )); then valid=0; fi
  done
  if (( final != 1 || bad != 0 || valid != 1 )); then
    echo "FAIL: incomplete coverage, final=$final FAIL/SKIP=$bad"
    tail -60 "$log"
    exit 1
  fi
  echo "T4 round $round PASS" | tee -a "$out/results.txt"
done
if [[ "$sync_d2h" == 1 ]]; then
  echo "T4_RESULT=DIAGNOSTIC_PASS sync_d2h=1 rounds=$rounds" | tee -a "$out/results.txt"
else
  echo "T4_RESULT=PASS rounds=$rounds" | tee -a "$out/results.txt"
fi
echo "OUTPUT=$out"
