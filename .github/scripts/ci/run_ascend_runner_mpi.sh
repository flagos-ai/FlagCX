#!/usr/bin/env bash
# Isolate collectives so one permanent async error cannot skip the rest.
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mpi_runner="$script_dir/run_mpi_with_timeout.sh"
binary=
for arg in "$@"; do
  if [[ "$arg" == --gtest_filter=* ]]; then
    exec "$mpi_runner" "$@"
  fi
  if [[ "${arg##*/}" == runner_mpi_tests ]]; then
    binary=$arg
  fi
done
if [[ -z "$binary" || ! -x "$binary" ]]; then
  echo 'Ascend runner binary is missing' >&2
  exit 1
fi
# Listing tests does not initialize the GTest MPI environment or run fixtures.
listing=$(timeout --kill-after=5s 30s "$binary" --gtest_list_tests)
tests=$(awk '
  /^FlagCXCollTest\./ { in_suite=1; next }
  /^[^ ]/ { in_suite=0 }
  in_suite && /^  [A-Za-z0-9_]+/ { print "FlagCXCollTest." $1 }
' <<< "$listing")
if [[ -z "$tests" ]]; then
  echo 'Ascend runner did not discover any collective tests' >&2
  exit 1
fi
status=0
label=${FLAGCX_CI_MPI_LABEL:-Ascend runner}
while IFS= read -r test; do
  FLAGCX_CI_MPI_LABEL="$label $test" \
    "$mpi_runner" "$@" --gtest_filter="$test" || status=$?
  if [[ "$status" == 130 || "$status" == 143 ]]; then
    exit "$status"
  fi
done <<< "$tests"
exit "$status"
