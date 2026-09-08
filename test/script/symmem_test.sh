#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SYMMEM_DIR="$PROJECT_ROOT/test/unittest/symmem"
BUILD_BIN="$SYMMEM_DIR/build/bin"
MPI_RUNNER="$PROJECT_ROOT/.github/scripts/ci/run_mpi_with_timeout.sh"
TEST_RUNNER="$PROJECT_ROOT/.github/scripts/ci/run_with_timeout.sh"

export MPI_HOME="${MPI_HOME:-/usr/local/mpi}"
export PATH="$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$PROJECT_ROOT/build/lib:${LD_LIBRARY_PATH:-}"

NP="${NP:-8}"

echo "=== Running symmem unit tests (no MPI/GPU) ==="
FLAGCX_CI_TEST_LABEL="symmem unit tests" \
  "$TEST_RUNNER" "$BUILD_BIN/symmem_unit_tests"

if [[ "${FLAGCX_CI_SYMMEM_MPI_SUPPORTED:-1}" != "1" ]]; then
  echo "Skipping symmem MPI tests: ${FLAGCX_CI_SYMMEM_MPI_SKIP_REASON:-unsupported by this platform}"
  exit 0
fi

echo ""
echo "=== Running symmem MPI tests (np=$NP) ==="
FLAGCX_CI_MPI_LABEL="symmem MPI tests" \
  "$MPI_RUNNER" -np "$NP" --allow-run-as-root \
    -x FLAGCX_USE_HETERO_COMM=1 \
    -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
    -x FLAGCX_MEM_ENABLE=1 \
    -x FLAGCX_VMM_ENABLE=0 \
    "$BUILD_BIN/symmem_mpi_tests"

echo ""
echo "All symmem tests passed."
