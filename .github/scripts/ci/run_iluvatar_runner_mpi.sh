#!/usr/bin/env bash
set -euo pipefail
export FLAGCX_CI_PLATFORM=${FLAGCX_CI_PLATFORM:-iluvatar}
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_runner_mpi.sh" "$@"
