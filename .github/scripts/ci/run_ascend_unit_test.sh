#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <core|adaptor|device_api_host|p2p|rma|runner|service|symmem>" >&2
  exit 2
fi

PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
export GITHUB_WORKSPACE=$PROJECT_ROOT
exec bash "$PROJECT_ROOT/.github/scripts/ci/run_unit_test.sh" \
  "$PROJECT_ROOT/.github/scripts/set_env/ascend.sh" "$1"
