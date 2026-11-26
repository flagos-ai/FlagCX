#!/usr/bin/env bash
set -euo pipefail
umask 022

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=${GITHUB_WORKSPACE:-$(cd "$script_dir/../../.." && pwd)}
source "$script_dir/timeouts.sh"
# This script is a child of prepare; its deadline never reaches normal tests.
flagcx_ci_start_budget FLAGCX_CI_MODE_DEADLINE 3m
results="${FLAGCX_CI_LOG_DIR:-$project_root/.ci-logs}/iluvatar-native-copy"
mkdir -p "$results"
chmod 755 "$results"
export FLAGCX_CI_LOG_DIR="$results/attempts"
export FLAGCX_CI_TEST_MAX_ATTEMPTS=1
export FLAGCX_CI_TEST_KILL_AFTER=2s
export FLAGCX_CI_MPI_KILL_AFTER=2s
# The probe has its own finite callback/query waits; retain existing diagnostics
# for normal workloads rather than starting stack collectors for these cases.
export FLAGCX_CI_TEST_DIAGNOSTICS_AFTER=
export FLAGCX_CI_MPI_DIAGNOSTICS_AFTER=
export FLAGCX_CI_REQUIRE_GTEST_PASS=0

printf 'case\tkind\trecord_count\tranks\texit\tresult\tdirectory\n' > "$results/summary.tsv"
build_status=0
FLAGCX_CI_BUILD_TEST_TIMEOUT=60s FLAGCX_CI_TEST_LABEL="Iluvatar native copy build" \
  flagcx_ci_run build-tests make -C "$project_root/test/unittest/runner" \
    native-copy-probe USE_ILUVATAR=1 || build_status=$?
if ((build_status != 0)); then
  printf 'build\t-\t-\t0\t%s\tFAIL\t%s\n' "$build_status" "$FLAGCX_CI_LOG_DIR" >> "$results/summary.tsv"
  cat "$results/summary.tsv"
  echo 'Native copy probe build failed; ordinary runner tests were not executed.' >&2
  exit "$build_status"
fi
binary="$project_root/test/unittest/runner/build/bin/iluvatar_native_copy_probe"
{
  printf 'Binary: %s\nCOREX_VISIBLE_DEVICES=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
    "$binary" "${COREX_VISIBLE_DEVICES:-}" "${CUDA_VISIBLE_DEVICES:-}"
  ldd "$binary" || true
} > "$results/libraries.log" 2>&1

status=0
# Local D2D already reproduces the stall. Isolate post-callback record count
# before rechecking IPC/staging. Repeated counts are fresh cases, not retries.
record_counts=(0 1 2 3 3 0)
for index in "${!record_counts[@]}"; do
  case_number=$((index + 1))
  record_count=${record_counts[index]}
  case_dir=$(mktemp -d "$results/local-$case_number-records-$record_count.XXXXXX")
  chmod 755 "$case_dir"
  case_status=0
  FLAGCX_CI_MPI_TIMEOUT=15s \
    FLAGCX_CI_MPI_LABEL="Iluvatar native local case $case_number record_count=$record_count" \
    bash "$script_dir/run_mpi_with_timeout.sh" --allow-run-as-root --bind-to none \
      -np 1 "$binary" local "$record_count" "$case_dir" || case_status=$?
  if [[ "$case_status" == 130 || "$case_status" == 143 ]]; then
    exit "$case_status"
  fi
  # An exit status alone must not hide an unexecuted case or a stale report.
  if [[ ! -f "$case_dir/rank-0.result" ]] ||
     ! grep -Fxq "ILUVATAR_COPY_PROBE_PASS kind=local record_count=$record_count rank=0" \
       "$case_dir/rank-0.result"; then
    echo "local case $case_number record_count=$record_count: missing PASS report for rank 0" >&2
    if ((case_status == 0)); then case_status=1; fi
  fi
  result=PASS
  if ((case_status != 0)); then status=1; result=FAIL; fi
  printf '%s\n' "$case_status" > "$case_dir/exit.status"
  printf '%s\tlocal\t%s\t1\t%s\t%s\t%s\n' "$case_number" "$record_count" \
    "$case_status" "$result" "$case_dir" >> "$results/summary.tsv"
done
cat "$results/summary.tsv"
if ((status != 0)); then
  echo 'Native copy controls failed; ordinary runner tests were not executed.' >&2
else
  echo 'All native copy controls passed; continuing ordinary runner tests.'
fi
exit "$status"
