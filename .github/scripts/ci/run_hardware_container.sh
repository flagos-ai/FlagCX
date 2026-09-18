#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <platform> <unittest|torch-api|perf> [unit-test-suite]" >&2
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
  torch-api|perf)
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
trap cleanup EXIT

cleanup
docker pull "$image"
docker run --rm \
  --name "$container_name" \
  "${docker_args[@]}" \
  --user=root \
  --volume "$project_root:/workspace-src:ro" \
  --workdir /workspace \
  --env HOME=/tmp/flagcx-ci-home \
  --env GITHUB_WORKSPACE=/workspace \
  --env GITHUB_ACTIONS=true \
  --env CI=true \
  --env FLAGCX_CI_PLATFORM="$platform" \
  --env FLAGCX_CI_WORKLOAD="$workload" \
  --env FLAGCX_CI_SUITE="$suite" \
  --env FLAGCX_CI_SET_ENV="$set_env" \
  --entrypoint=/bin/bash \
  "$image" -lc '
    set -euo pipefail
    mkdir -p "$HOME" /workspace
    cp -a /workspace-src/. /workspace/
    git config --global --add safe.directory /workspace
    git config --global --add safe.directory /workspace/third-party/googletest
    git config --global --add safe.directory /workspace/third-party/json
    bash .github/scripts/ci/run_hardware_workload.sh \
      "$FLAGCX_CI_PLATFORM" "$FLAGCX_CI_WORKLOAD" \
      "$FLAGCX_CI_SET_ENV" "$FLAGCX_CI_SUITE"
  '
