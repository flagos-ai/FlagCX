#!/usr/bin/env bash

# Shared stage limits. Absolute deadlines survive nested wrappers and MPI modes.
if [[ "${FLAGCX_CI_COVERAGE:-0}" == 1 ]]; then
  : "${FLAGCX_CI_BUILD_TIMEOUT:=20m}"
  : "${FLAGCX_CI_TEST_TIMEOUT:=10m}"
  : "${FLAGCX_CI_MPI_TIMEOUT:=10m}"
  : "${FLAGCX_CI_SUITE_TEST_BUDGET:=25m}"
  : "${FLAGCX_CI_RUNNER_PER_TEST_TIMEOUT:=5m}"
  : "${FLAGCX_CI_RUNNER_STAGE_BUDGET:=600}"
else
  : "${FLAGCX_CI_BUILD_TIMEOUT:=15m}"
  : "${FLAGCX_CI_TEST_TIMEOUT:=3m}"
  : "${FLAGCX_CI_MPI_TIMEOUT:=5m}"
  : "${FLAGCX_CI_SUITE_TEST_BUDGET:=15m}"
  : "${FLAGCX_CI_RUNNER_PER_TEST_TIMEOUT:=2m}"
  : "${FLAGCX_CI_RUNNER_STAGE_BUDGET:=300}"
fi
: "${FLAGCX_CI_BUILD_TEST_TIMEOUT:=10m}"
: "${FLAGCX_CI_PREFLIGHT_TIMEOUT:=3m}"
: "${FLAGCX_CI_IMPORT_TIMEOUT:=2m}"
: "${FLAGCX_CI_TORCH_BUILD_TIMEOUT:=15m}"
: "${FLAGCX_CI_TORCH_MODE_TIMEOUT:=15m}"
: "${FLAGCX_CI_TORCH_COVERAGE_TIMEOUT:=15m}"
: "${FLAGCX_CI_PERF_TIMEOUT:=10m}"
: "${FLAGCX_CI_PERF_MODE_BUDGET:=20m}"
export FLAGCX_CI_TEST_TIMEOUT FLAGCX_CI_MPI_TIMEOUT
export FLAGCX_CI_RUNNER_PER_TEST_TIMEOUT FLAGCX_CI_RUNNER_STAGE_BUDGET

flagcx_ci_seconds() {
  awk -v duration="$1" 'BEGIN {
    if (duration !~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)[smhd]?$/) exit 2;
    unit = substr(duration, length(duration));
    scale = unit == "d" ? 86400 : unit == "h" ? 3600 : unit == "m" ? 60 : 1;
    seconds = (duration + 0) * scale;
    if (seconds <= 0) exit 2;
    printf "%.6f", seconds;
  }'
}

flagcx_ci_now() {
  python3 -c 'import time; print(time.time())'
}

flagcx_ci_start_budget() {
  local variable=$1 seconds deadline
  seconds=$(flagcx_ci_seconds "$2") || return 2
  deadline=$(awk -v now="$(flagcx_ci_now)" -v seconds="$seconds" \
    -v previous="${!variable:-}" 'BEGIN {
      limit = now + seconds;
      if (previous != "" && previous + 0 < limit) limit = previous;
      printf "%.6f", limit;
    }')
  printf -v "$variable" '%s' "$deadline"
  export "$variable"
}

flagcx_ci_limit_timeout() {
  local seconds
  seconds=$(flagcx_ci_seconds "$1") || return 2
  awk -v seconds="$seconds" -v now="$(flagcx_ci_now)" -v details="${2:-}" \
    -v workload="${FLAGCX_CI_WORKLOAD_DEADLINE:-}" \
    -v suite="${FLAGCX_CI_TEST_DEADLINE:-}" \
    -v mode="${FLAGCX_CI_MODE_DEADLINE:-}" \
    'BEGIN {
      source = "invocation";
      split("workload suite mode", names, " ");
      limits[1] = workload;
      limits[2] = suite;
      limits[3] = mode;
      for (i = 1; i <= 3; i++) {
        if (limits[i] == "") continue;
        if (limits[i] !~ /^[0-9]+([.][0-9]+)?$/) exit 2;
        remaining = limits[i] - now;
        if (remaining < seconds) {
          seconds = remaining;
          source = names[i];
        }
      }
      if (seconds <= 0) {
        if (details == "details") print source;
        exit 124;
      }
      printf "%.6fs", seconds;
      if (details == "details") printf " %s", source;
    }'
}

flagcx_ci_run() {
  local phase=$1 limit
  shift
  case "$phase" in
    build) limit=$FLAGCX_CI_BUILD_TIMEOUT ;;
    build-tests) limit=$FLAGCX_CI_BUILD_TEST_TIMEOUT ;;
    preflight) limit=$FLAGCX_CI_PREFLIGHT_TIMEOUT ;;
    import) limit=$FLAGCX_CI_IMPORT_TIMEOUT ;;
    torch-build) limit=$FLAGCX_CI_TORCH_BUILD_TIMEOUT ;;
    torch-coverage) limit=$FLAGCX_CI_TORCH_COVERAGE_TIMEOUT ;;
    *) echo "Unknown CI timeout phase: $phase" >&2; return 2 ;;
  esac
  FLAGCX_CI_TEST_TIMEOUT="$limit" FLAGCX_CI_TEST_MAX_ATTEMPTS=1 \
    FLAGCX_CI_TEST_LABEL="${FLAGCX_CI_TEST_LABEL:-CI $phase}" \
    bash "$(dirname "${BASH_SOURCE[0]}")/run_with_timeout.sh" "$@"
}
