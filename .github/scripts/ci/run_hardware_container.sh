#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <platform> <unittest|torch-api|perf|perf-legacy> [unit-test-suite]" >&2
  exit 2
fi

platform=$1
workload=$2
suite=${3:-}

case "$workload" in
  unittest)
    if [[ -z "$suite" ]]; then
      echo "A unit-test suite is required for the unittest workload." >&2
      exit 2
    fi
    ;;
  torch-api|perf|perf-legacy)
    if [[ -n "$suite" ]]; then
      echo "The $workload workload does not accept a unit-test suite." >&2
      exit 2
    fi
    ;;
  *)
    echo "Unknown hardware workload: $workload" >&2
    exit 2
    ;;
esac

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "$script_dir/../../.." && pwd)

# Parse platform YAML on a standard GitHub-hosted runner and pass these values
# into the hardware job. Accelerator hosts only need Bash and Docker.
: "${FLAGCX_CI_IMAGE:?FLAGCX_CI_IMAGE must be provided by the workflow}"
: "${FLAGCX_CI_DOCKER_ARGS:?FLAGCX_CI_DOCKER_ARGS must be provided by the workflow}"
: "${FLAGCX_CI_SET_ENV:?FLAGCX_CI_SET_ENV must be provided by the workflow}"

image=$FLAGCX_CI_IMAGE
docker_args_text=$FLAGCX_CI_DOCKER_ARGS
set_env=$FLAGCX_CI_SET_ENV
ignore_test=${FLAGCX_CI_IGNORE_TEST:-'{}'}
container_name=${FLAGCX_CI_CONTAINER_NAME:-flagcx-${platform}-${workload//-/_}-${suite:-job}-$$}

docker_args=()
while IFS= read -r argument; do
  if [[ -n "$argument" ]]; then
    docker_args+=("$argument")
  fi
done <<<"$docker_args_text"

for proxy_var in HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
  if [[ -n "${!proxy_var:-}" ]]; then
    docker_args+=(--env "$proxy_var")
  fi
done

cleanup() {
  docker rm --force "$container_name" >/dev/null 2>&1 || true
}
log_dir=${FLAGCX_CI_LOG_DIR:-${RUNNER_TEMP:-/tmp}/flagcx-ci-attempts}
mkdir -p "$log_dir"
summary_file=$(mktemp "$log_dir/summary.XXXXXX.md")
finish() {
  if [[ "$platform" == ascend && "$workload" == perf ]]; then
    bash "$script_dir/ascend_rdma_host_inventory.sh" \
      >"$log_dir/ascend-rdma-host-inventory-after.log" 2>&1 || true
  fi
  cleanup
  if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
    cat "$summary_file" >> "$GITHUB_STEP_SUMMARY"
  fi
}
trap finish EXIT

pull_image() {
  local attempt pull_log status
  for attempt in 1 2 3; do
    pull_log="$log_dir/image-pull-attempt-$attempt.log"
    if docker pull "$image" >"$pull_log" 2>&1; then
      cat "$pull_log"
      printf -- '- **Image pull:** succeeded on attempt %s/3\n' "$attempt" >> "$summary_file"
      return 0
    else
      status=$?
    fi
    cat "$pull_log" >&2
    # Retry registry/network outages only; authentication, missing manifests and
    # unknown errors still fail immediately. Never substitute a cached image.
    if [[ "$attempt" == 3 ]] || grep -Eiq \
        'unauthorized|authentication required|no basic auth credentials|pull access denied|manifest unknown|invalid reference format' \
        "$pull_log" || ! grep -Eiq \
        'timeout|timed out|connection reset|connection refused|temporary failure|no such host|network is unreachable|unexpected (HTTP )?status.*50[0-9]|service unavailable|bad gateway|internal server error' \
        "$pull_log"; then
      printf -- '- **Image pull:** failed on attempt %s/3 (exit=%s)\n' "$attempt" "$status" >> "$summary_file"
      return "$status"
    fi
    echo "::warning::Image pull attempt $attempt/3 failed; retrying in $((attempt * 5)) seconds. Log: $pull_log" >&2
    sleep "$((attempt * 5))"
  done
}

cleanup
pull_image

# Ascend devices are host-global and cannot be safely shared by concurrent
# containers. Serialize Ascend jobs even when they come from different
# workflows or manual runs on the same self-hosted runner. Take the lock
# after pulling so the image retries above never hold it.
if [[ "$platform" == "ascend" ]]; then
  exec 9>/var/lock/flagcx-ascend-ci.lock
  flock --exclusive --wait 7200 9
  if [[ "$workload" == "perf" ]]; then
    bash "$script_dir/ascend_rdma_host_inventory.sh" | \
      tee "$log_dir/ascend-rdma-host-inventory.log"
  fi
fi

docker run --rm \
  --name "$container_name" \
  "${docker_args[@]}" \
  --user=root \
  --volume "$project_root:/workspace-src:ro" \
  --volume "$log_dir:/ci-logs" \
  --workdir /workspace \
  --env HOME=/tmp/flagcx-ci-home \
  --env GITHUB_WORKSPACE=/workspace \
  --env GITHUB_ACTIONS=true \
  --env CI=true \
  --env FLAGCX_CI_TEST_TIMEOUT="${FLAGCX_CI_TEST_TIMEOUT:-20m}" \
  --env FLAGCX_CI_MPI_TIMEOUT="${FLAGCX_CI_MPI_TIMEOUT:-20m}" \
  --env FLAGCX_CI_LOG_DIR=/ci-logs \
  --env FLAGCX_CI_TORCH_COVERAGE="${FLAGCX_CI_TORCH_COVERAGE:-0}" \
  --env FLAGCX_CI_COVERAGE_REPORT="/ci-logs/coverage-${platform}-gpu-torch.json" \
  --env FLAGCX_CI_COVERAGE="${FLAGCX_CI_COVERAGE:-0}" \
  --env FLAGCX_CI_CPP_COVERAGE_REPORT="${FLAGCX_CI_CPP_COVERAGE_REPORT:-/ci-logs/cpp-coverage}" \
  --env GITHUB_STEP_SUMMARY="/ci-logs/$(basename "$summary_file")" \
  --env FLAGCX_CI_PLATFORM="$platform" \
  --env FLAGCX_CI_WORKLOAD="$workload" \
  --env FLAGCX_CI_SUITE="$suite" \
  --env FLAGCX_CI_SET_ENV="$set_env" \
  --env "FLAGCX_CI_IGNORE_TEST=$ignore_test" \
  --entrypoint=/bin/bash \
  "$image" -lc '
    set -euo pipefail
    mkdir -p "$HOME" /workspace
    cp -a /workspace-src/. /workspace/
    git config --global --add safe.directory /workspace
    git config --global --add safe.directory /workspace/third-party/googletest
    git config --global --add safe.directory /workspace/third-party/json
    # Local checkouts and development images can carry objects compiled with
    # older headers. The test Makefiles do not track all header dependencies.
    # Clean ignored artifacts in the writable copy before compiling this ref.
    git clean -ffdX
    bash .github/scripts/ci/run_hardware_workload.sh \
      "$FLAGCX_CI_PLATFORM" "$FLAGCX_CI_WORKLOAD" \
      "$FLAGCX_CI_SET_ENV" "$FLAGCX_CI_SUITE"
  '
