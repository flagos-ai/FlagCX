#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <platform> <unittest|torch-api|perf> <set-env-script> <suite>" >&2
  exit 2
fi

platform=$1
workload=$2
set_env=$3
suite=$4
project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}

case "$workload" in
  unittest)
    exec bash "$project_root/.github/scripts/ci/run_unit_test.sh" \
      "$set_env" "$suite"
    ;;
  torch-api|perf)
    handler="$project_root/.github/scripts/ci/run_${platform}_workload.sh"
    if [[ ! -f "$handler" ]]; then
      echo "No $platform handler is available for the $workload workload: $handler" >&2
      exit 2
    fi
    exec bash "$handler" "$workload"
    ;;
  *)
    echo "Unknown hardware workload: $workload" >&2
    exit 2
    ;;
esac
