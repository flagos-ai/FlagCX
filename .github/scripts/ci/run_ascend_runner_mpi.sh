#!/usr/bin/env bash
set -euo pipefail
export GTEST_FILTER=${GTEST_FILTER:-FlagCXCollTest.*}
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_runner_mpi.sh" "$@"
