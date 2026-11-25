#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
MPI_RUNNER="$PROJECT_ROOT/.github/scripts/ci/run_mpi_with_timeout.sh"
TEST_BIN=${FLAGCX_HOST_MPI_CLEANUP_BIN:-"$PROJECT_ROOT/test/unittest/adaptor/build/bin/adaptor_host_mpi_cleanup_tests"}
RESULT_ROOT=${FLAGCX_HOST_MPI_CLEANUP_RESULT_ROOT:-"$PROJECT_ROOT/.ci-results"}
mkdir -p "$RESULT_ROOT"
result_dir=$(mktemp -d "$RESULT_ROOT/host-mpi-cleanup.XXXXXX")
cases=(
  HostMpiCleanup.EmptyAndStagedOwnersDestroyIndependently
  HostMpiCleanup.AutomaticFallbackOwnerDestroysWhilePeersWait
)
filter=$(IFS=:; echo "${cases[*]}")
status=0
FLAGCX_CI_MPI_LABEL="host MPI fallback cleanup" \
  FLAGCX_CI_MPI_TIMEOUT="${FLAGCX_CI_MPI_TIMEOUT:-60s}" \
  "$MPI_RUNNER" -np 2 --allow-run-as-root \
  bash "$PROJECT_ROOT/.github/scripts/ci/run_gtest_rank_xml.sh" \
  "$result_dir" "$TEST_BIN" --gtest_filter="$filter" || status=$?
required=()
for test_case in "${cases[@]}"; do required+=( --case "$test_case" ); done
python3 "$PROJECT_ROOT/.github/scripts/ci/check_required_gtests.py" \
  "${required[@]}" "$result_dir/rank-0.xml" "$result_dir/rank-1.xml" || status=$?
exit "$status"
