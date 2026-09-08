#!/usr/bin/env bash

set -euo pipefail

mpi_timeout="${FLAGCX_CI_MPI_TIMEOUT:-20m}"
mpi_kill_after="${FLAGCX_CI_MPI_KILL_AFTER:-30s}"
mpi_launcher="${FLAGCX_CI_MPIRUN:-mpirun}"
mpi_label="${FLAGCX_CI_MPI_LABEL:-MPI invocation}"
script_dir=$(cd "$(dirname "$0")" && pwd)
test_runner="$script_dir/run_with_timeout.sh"

if ! command -v "$mpi_launcher" >/dev/null 2>&1; then
  echo "MPI launcher not found: $mpi_launcher" >&2
  exit 1
fi

FLAGCX_CI_TEST_TIMEOUT="$mpi_timeout" \
  FLAGCX_CI_TEST_KILL_AFTER="$mpi_kill_after" \
  FLAGCX_CI_TEST_LABEL="$mpi_label" \
  exec "$test_runner" "$mpi_launcher" "$@"
