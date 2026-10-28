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
vmm_provider_env=()
if [[ "${FLAGCX_ADAPTOR:-}" == "du" ]]; then
  vmm_provider_env+=( -x FLAGCX_VMM_MULTI_SEGMENT_ENABLE=0 )
fi

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
if [[ "${FLAGCX_CI_SYMMEM_SKIP_VMM_LOCAL:-0}" == 1 ]]; then
  echo "Skipping symmem VMM-local MPI tests: this platform does not expose a validated cross-GPU flat VMM mapping"
else
  echo "=== Running symmem VMM-local MPI tests (np=$LOCAL_NP) ==="
  FLAGCX_CI_MPI_LABEL="symmem VMM local" \
    "$MPI_RUNNER" -np "$LOCAL_NP" --allow-run-as-root \
      -x FLAGCX_USE_HETERO_COMM=1 \
      -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
      -x FLAGCX_MEM_ENABLE=1 \
      -x FLAGCX_VMM_ENABLE=1 \
      -x FLAGCX_IB_DISABLE=1 \
      -x FLAGCX_CI_REQUIRE_VMM=1 \
      "${vmm_provider_env[@]}" \
      "$BUILD_BIN/symmem_mpi_tests" \
      --gtest_filter=-SymMemTest.CrossGpuReadViaIpcPeerPtr:SymMemTest.CrossGpuWriteViaIpcPeerPtr:SymMemTest.HybridLocalAndRemoteAccess
fi

case "${FLAGCX_ADAPTOR:-}" in
  nvidia|metax|ppu)
    provider_multi_segment_status=0
    for provider_test in \
      SymMemTest.SegmentedVmmSubrangeMapsOnlyTouchedSegment \
      SymMemTest.SegmentedVmmWindowMapsDifferentPeerLayouts; do
      echo ""
      echo "=== Running $provider_test (np=$LOCAL_NP) ==="
      if ! FLAGCX_CI_MPI_LABEL="$provider_test" \
        "$MPI_RUNNER" -np "$LOCAL_NP" --allow-run-as-root \
          -x FLAGCX_USE_HETERO_COMM=1 \
          -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
          -x FLAGCX_MEM_ENABLE=1 \
          -x FLAGCX_VMM_ENABLE=1 \
          -x FLAGCX_IB_DISABLE=1 \
          -x FLAGCX_CI_REQUIRE_MULTI_SEGMENT=1 \
          "$BUILD_BIN/symmem_mpi_tests" --gtest_filter="$provider_test"; then
        provider_multi_segment_status=1
      fi
    done
    if [[ "$provider_multi_segment_status" != "0" ]]; then
      exit 1
    fi
    ;;
esac

case "${FLAGCX_ADAPTOR:-}" in
  nvidia|du|metax|ppu)
    echo ""
    echo "=== Running untracked VMM mapping rejection (np=$LOCAL_NP) ==="
    FLAGCX_CI_MPI_LABEL="symmem untracked VMM mapping" \
      "$MPI_RUNNER" -np "$LOCAL_NP" --allow-run-as-root \
        -x FLAGCX_USE_HETERO_COMM=1 \
        -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
        -x FLAGCX_MEM_ENABLE=1 \
        -x FLAGCX_VMM_ENABLE=1 \
        -x FLAGCX_IB_DISABLE=1 \
        -x FLAGCX_CI_REQUIRE_PROVIDER_VMM=1 \
        "${vmm_provider_env[@]}" \
        "$BUILD_BIN/symmem_mpi_tests" \
        --gtest_filter=SymMemTest.UntrackedVmmMappingIsRejected
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

echo ""
echo "All symmem tests passed."
