#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <image> <set-env-script> <suite>" >&2
  exit 2
fi

IMAGE=$1
SET_ENV_SCRIPT=$2
SUITE=$3
PROJECT_ROOT=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}

if [[ ! -f "$SET_ENV_SCRIPT" ]]; then
  echo "Platform environment script not found: $SET_ENV_SCRIPT" >&2
  exit 1
fi

docker run --rm \
  --network host \
  --ipc=host \
  --privileged=true \
  --shm-size=100gb \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -e PLATFORM=hygon \
  -e FLAGCX_ADAPTOR=du \
  -e CI=true \
  -e HOME=/github/home \
  -e GITHUB_WORKSPACE="$PROJECT_ROOT" \
  -v "$PROJECT_ROOT:$PROJECT_ROOT" \
  -v /usr/local/hyhal:/opt/hyhal:ro \
  -w "$PROJECT_ROOT" \
  "$IMAGE" \
  bash "$PROJECT_ROOT/.github/scripts/ci/run_unit_test.sh" "$SET_ENV_SCRIPT" "$SUITE"
