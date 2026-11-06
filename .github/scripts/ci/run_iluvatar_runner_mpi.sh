#!/usr/bin/env bash
# Keep a stuck case from consuming the budget of the remaining runner tests.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
binary=
filter=${GTEST_FILTER:-*}
args=()
for arg in "$@"; do
  if [[ "$arg" == --gtest_filter=* ]]; then
    filter=${arg#--gtest_filter=}
    continue
  fi
  args+=("$arg")
  if [[ "${arg##*/}" == runner_mpi_tests ]]; then
    binary=$arg
  fi
done
if [[ -z "$binary" || ! -x "$binary" ]]; then
  echo 'Iluvatar runner binary is missing' >&2
  exit 1
fi

# GTest applies the original positive and negative filters before listing.
# Listing does not initialize MPI or run any hardware fixture.
listing=$(timeout --kill-after=5s 30s "$binary" \
  --gtest_list_tests --gtest_filter="$filter")
tests=$(awk '
  /^FlagCXCollTest\./ { in_suite=1; next }
  /^[^ ]/ { in_suite=0 }
  in_suite && /^  [A-Za-z0-9_]+/ { print "FlagCXCollTest." $1 }
' <<< "$listing")
if [[ -z "$tests" ]]; then
  echo "Iluvatar runner filter selected no collective tests: $filter" >&2
  exit 1
fi

status=0
label=${FLAGCX_CI_MPI_LABEL:-Iluvatar runner}
while IFS= read -r test; do
  test_status=0
  if [[ "$test" == FlagCXCollTest.SendRecv ]]; then
    # This case currently hangs in CoreX stream synchronization. Still run
    # it and fail on timeout, but do not repeat a known deterministic hang.
    # Capture diagnostics before timeout terminates the MPI process group.
    FLAGCX_CI_MPI_TIMEOUT="${FLAGCX_CI_RUNNER_SENDRECV_TIMEOUT:-2m}" \
      FLAGCX_CI_TEST_DIAGNOSTICS_AFTER="${FLAGCX_CI_RUNNER_DIAGNOSTICS_AFTER:-90}" \
      FLAGCX_CI_TEST_MAX_ATTEMPTS=1 \
      FLAGCX_CI_MPI_LABEL="$label $test" \
      "$script_dir/run_mpi_with_timeout.sh" "${args[@]}" \
      --gtest_filter="$test" || test_status=$?
  else
    FLAGCX_CI_MPI_LABEL="$label $test" \
      "$script_dir/run_mpi_with_timeout.sh" "${args[@]}" \
      --gtest_filter="$test" || test_status=$?
  fi
  echo "$label $test result=$test_status"
  if [[ "$test_status" == 130 || "$test_status" == 143 ]]; then
    exit "$test_status"
  fi
  if ((status == 0 && test_status != 0)); then
    status=$test_status
  fi
done <<< "$tests"
exit "$status"
