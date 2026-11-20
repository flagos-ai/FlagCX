#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
MPI_RUNNER="$PROJECT_ROOT/.github/scripts/ci/run_mpi_with_timeout.sh"
TEST_BIN=${FLAGCX_RUNNER_PREFLIGHT_BIN:-"$PROJECT_ROOT/test/unittest/runner/build/bin/runner_mpi_tests"}
RESULT_ROOT=${FLAGCX_RUNNER_PREFLIGHT_RESULT_ROOT:-"$PROJECT_ROOT/.ci-results"}
mkdir -p "$RESULT_ROOT"
result_dir=$(mktemp -d "$RESULT_ROOT/runner-fused-preflight.XXXXXX")
cases=(
  FusedPreflightMpi.EventCreateFailureConvergesForBothApis
  FusedPreflightMpi.CaptureQueryFailureConvergesForBothApis
  FusedPreflightMpi.ActiveCaptureRejectsBothApisCollectively
  FusedPreflightMpi.InProgressConvergesWithoutSubmittingEitherApi
  FusedPreflightMpi.CanceledRecordReleaseCanRetryWhilePeersWait
  FusedPreflightMpi.AllGatherGemmCanRetryAfterPreflightFailure
  FusedPreflightMpi.GemmReduceScatterCanRetryAfterPreflightFailure
)
filter=$(IFS=:; echo "${cases[*]}")
status=0
FLAGCX_CI_MPI_LABEL="runner fused completion preflight" \
  FLAGCX_CI_MPI_TIMEOUT="${FLAGCX_CI_MPI_TIMEOUT:-60s}" \
  "$MPI_RUNNER" -np 3 --allow-run-as-root -x FLAGCX_IB_DISABLE=1 \
  bash "$PROJECT_ROOT/.github/scripts/ci/run_gtest_rank_xml.sh" \
  "$result_dir" "$TEST_BIN" --gtest_filter="$filter" || status=$?
required=()
for test_case in "${cases[@]}"; do
  required+=( --case "$test_case" )
done
python3 "$PROJECT_ROOT/.github/scripts/ci/check_required_gtests.py" \
  "${required[@]}" "$result_dir/rank-0.xml" "$result_dir/rank-1.xml" \
  "$result_dir/rank-2.xml" || status=$?
exit "$status"
