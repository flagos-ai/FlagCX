#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <platform> <unittest|torch-api|perf|perf-test-experimental> <set-env-script> <suite>" >&2
  exit 2
fi

platform=$1
workload=$2
set_env=$3
suite=$4
project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
export FLAGCX_CI_PLATFORM="$platform"
source "$project_root/.github/scripts/ci/timeouts.sh"
workload_timeout=${FLAGCX_CI_WORKLOAD_TIMEOUT:-55m}
if [[ "$workload" == unittest && "${FLAGCX_CI_COVERAGE:-0}" != 1 ]]; then
  workload_timeout=${FLAGCX_CI_WORKLOAD_TIMEOUT:-40m}
fi
flagcx_ci_start_budget FLAGCX_CI_WORKLOAD_DEADLINE "$workload_timeout"
export FLAGCX_CI_WORKLOAD_BOUNDED=1
export FLAGCX_CI_TEST_MAX_ATTEMPTS=1

case "$workload" in
  unittest)
    FLAGCX_CI_TEST_TIMEOUT="$workload_timeout" FLAGCX_CI_TEST_LABEL="$platform $workload" \
      exec bash "$project_root/.github/scripts/ci/run_with_timeout.sh" \
      env -u FLAGCX_CI_TEST_LABEL FLAGCX_CI_TEST_TIMEOUT="$FLAGCX_CI_TEST_TIMEOUT" bash "$project_root/.github/scripts/ci/run_unit_test.sh" \
      "$set_env" "$suite"
    ;;
  torch-api|perf|perf-test-experimental)
    handler="$project_root/.github/scripts/ci/run_${platform}_workload.sh"
    if [[ ! -f "$handler" ]]; then
      echo "No $platform handler is available for the $workload workload: $handler" >&2
      exit 2
    fi
    export FLAGCX_CI_SET_ENV="$set_env"
    FLAGCX_CI_TEST_TIMEOUT="$workload_timeout" FLAGCX_CI_TEST_LABEL="$platform $workload" \
      exec bash "$project_root/.github/scripts/ci/run_with_timeout.sh" \
      env -u FLAGCX_CI_TEST_LABEL FLAGCX_CI_TEST_TIMEOUT="$FLAGCX_CI_TEST_TIMEOUT" bash "$handler" "$workload"
    ;;
  *)
    echo "Unknown hardware workload: $workload" >&2
    exit 2
    ;;
esac
