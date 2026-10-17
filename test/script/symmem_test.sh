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

LOCAL_NP="${FLAGCX_CI_SYMMEM_LOCAL_NP:-2}"

echo "=== Running symmem unit tests (no MPI/GPU) ==="
FLAGCX_CI_TEST_LABEL="symmem unit tests" \
  "$TEST_RUNNER" "$BUILD_BIN/symmem_unit_tests"

echo ""
echo ""
echo "=== Running symmem IPC-local MPI tests (np=$LOCAL_NP) ==="
FLAGCX_CI_MPI_LABEL="symmem IPC local" \
  "$MPI_RUNNER" -np "$LOCAL_NP" --allow-run-as-root \
    -x FLAGCX_USE_HETERO_COMM=1 \
    -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
    -x FLAGCX_MEM_ENABLE=1 \
    -x FLAGCX_VMM_ENABLE=0 \
    -x FLAGCX_IB_DISABLE=1 \
    "$BUILD_BIN/symmem_mpi_tests" \
    --gtest_filter=-SymMemTest.CrossGpuReadViaPeerPtr:SymMemTest.CrossGpuWriteViaPeerPtr:SymMemTest.HybridLocalAndRemoteAccess

echo ""
echo "=== Running symmem VMM-local MPI tests (np=$LOCAL_NP) ==="
FLAGCX_CI_MPI_LABEL="symmem VMM local" \
  "$MPI_RUNNER" -np "$LOCAL_NP" --allow-run-as-root \
    -x FLAGCX_USE_HETERO_COMM=1 \
    -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
    -x FLAGCX_MEM_ENABLE=1 \
    -x FLAGCX_VMM_ENABLE=1 \
    -x FLAGCX_IB_DISABLE=1 \
    -x FLAGCX_CI_REQUIRE_VMM=1 \
    "$BUILD_BIN/symmem_mpi_tests" \
    --gtest_filter=-SymMemTest.CrossGpuReadViaIpcPeerPtr:SymMemTest.CrossGpuWriteViaIpcPeerPtr:SymMemTest.HybridLocalAndRemoteAccess

case "${FLAGCX_ADAPTOR:-}" in
  nvidia|du|metax|ppu)
    echo ""
    echo "=== Running provider multi-segment VMM MPI test (np=$LOCAL_NP) ==="
    FLAGCX_CI_MPI_LABEL="symmem provider multi-segment" \
      "$MPI_RUNNER" -np "$LOCAL_NP" --allow-run-as-root \
        -x FLAGCX_USE_HETERO_COMM=1 \
        -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
        -x FLAGCX_MEM_ENABLE=1 \
        -x FLAGCX_VMM_ENABLE=1 \
        -x FLAGCX_IB_DISABLE=1 \
        -x FLAGCX_CI_REQUIRE_MULTI_SEGMENT=1 \
        "$BUILD_BIN/symmem_mpi_tests" \
        --gtest_filter=SymMemTest.SegmentedVmmSubrangeMapsOnlyTouchedSegment:SymMemTest.SegmentedVmmWindowMapsDifferentPeerLayouts:SymMemTest.UntrackedVmmMappingIsRejected
    ;;
esac

if [[ "${FLAGCX_CI_REQUIRE_MULTICAST:-0}" == "1" ]]; then
  case "${FLAGCX_ADAPTOR:-}" in
    nvidia|metax)
      echo ""
      echo "=== Running required multicast LSA MPI tests (np=$LOCAL_NP) ==="
      FLAGCX_CI_MPI_LABEL="symmem multicast LSA" \
        "$MPI_RUNNER" -np "$LOCAL_NP" --allow-run-as-root \
          -x FLAGCX_USE_HETERO_COMM=1 \
          -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
          -x FLAGCX_MEM_ENABLE=1 \
          -x FLAGCX_VMM_ENABLE=1 \
          -x FLAGCX_IB_DISABLE=1 \
          -x FLAGCX_CI_REQUIRE_VMM=1 \
          -x FLAGCX_CI_REQUIRE_MULTICAST=1 \
          "$BUILD_BIN/symmem_mpi_tests" \
          --gtest_filter=SymMemTest.MulticastMappingWhenSupported:SymMemTest.LsaMulticastOffsetsReuseAndBindFailureFallsBack:SymMemTest.LsaMulticastUnbindFailureRetainsWindowForRetry
      ;;
    *)
      echo "multicast LSA is only available on nvidia and metax" >&2
      exit 1
      ;;
  esac
fi

if [[ "${FLAGCX_ADAPTOR:-}" == "du" ]]; then
  echo ""
  echo "=== Running DU multi-segment VMM MPI tests (np=$LOCAL_NP) ==="
  FLAGCX_CI_MPI_LABEL="symmem DU multi-segment" \
    "$MPI_RUNNER" -np "$LOCAL_NP" --allow-run-as-root \
      -x FLAGCX_USE_HETERO_COMM=1 \
      -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
      -x FLAGCX_MEM_ENABLE=1 \
      -x FLAGCX_VMM_ENABLE=1 \
      -x FLAGCX_IB_DISABLE=1 \
      -x FLAGCX_CI_REQUIRE_MULTI_SEGMENT=1 \
      "$BUILD_BIN/symmem_mpi_tests" \
      --gtest_filter=SymMemTest.DuSegmentedAllocFailureRollsBackAndCanRetry
fi

echo ""
echo "All symmem tests passed."
