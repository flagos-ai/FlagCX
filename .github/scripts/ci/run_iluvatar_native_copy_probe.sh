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

printf 'case\tranks\texit\tresult\tdirectory\n' > "$results/summary.tsv"
build_status=0
FLAGCX_CI_BUILD_TEST_TIMEOUT=60s FLAGCX_CI_TEST_LABEL="Iluvatar native copy build" \
  flagcx_ci_run build-tests make -C "$project_root/test/unittest/runner" \
    native-copy-probe USE_ILUVATAR=1 || build_status=$?
if ((build_status != 0)); then
  printf 'build\t0\t%s\tFAIL\t%s\n' "$build_status" "$FLAGCX_CI_LOG_DIR" >> "$results/summary.tsv"
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
for kind in local ipc staging; do
  ranks=1
  if [[ "$kind" == ipc ]]; then ranks=2; fi
  for control in B C; do
    case_dir=$(mktemp -d "$results/$kind-$control.XXXXXX")
    chmod 755 "$case_dir"
    case_status=0
    FLAGCX_CI_MPI_TIMEOUT=15s FLAGCX_CI_MPI_LABEL="Iluvatar native $kind $control" \
      bash "$script_dir/run_mpi_with_timeout.sh" --allow-run-as-root --bind-to none \
        -np "$ranks" "$binary" "$kind" "$control" "$case_dir" || case_status=$?
    if [[ "$case_status" == 130 || "$case_status" == 143 ]]; then
      exit "$case_status"
    fi
    # An exit status alone must not hide a missing rank or an unexecuted case.
    for ((rank=0; rank<ranks; ++rank)); do
      if ! grep -Fxq "ILUVATAR_COPY_PROBE_PASS kind=$kind case=$control rank=$rank" \
        "$case_dir/rank-$rank.result"; then
        echo "$kind-$control: missing PASS report for rank $rank" >&2
        if ((case_status == 0)); then case_status=1; fi
      fi
    done
    result=PASS
    if ((case_status != 0)); then status=1; result=FAIL; fi
    printf '%s\n' "$case_status" > "$case_dir/exit.status"
    printf '%s-%s\t%s\t%s\t%s\t%s\n' "$kind" "$control" "$ranks" \
      "$case_status" "$result" "$case_dir" >> "$results/summary.tsv"
  done
done
cat "$results/summary.tsv"
if ((status != 0)); then
  echo 'Native copy controls failed; ordinary runner tests were not executed.' >&2
else
  echo 'All native copy controls passed; continuing ordinary runner tests.'
fi
exit "$status"
