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
run_multi_segment=0
# Keep this fixed list aligned with the provider cases exercised by hardware CI.
# DU multi-segment mapping is still under driver validation.
case "${FLAGCX_ADAPTOR:-}" in
  nvidia|metax|ppu)
    run_multi_segment=1
    ;;
  du)
    vmm_provider_env+=( -x FLAGCX_VMM_MULTI_SEGMENT_ENABLE=0 )
    ;;
  *)
    echo "FLAGCX_ADAPTOR must be nvidia, du, ppu, or metax for SymMem CI" >&2
    exit 2
    ;;
esac

run_required_case() {
  local test_case=$1
  shift
  FLAGCX_CI_MPI_LABEL="$test_case" \
    "$MPI_RUNNER" -np "$LOCAL_NP" --allow-run-as-root \
      -x FLAGCX_USE_HETERO_COMM=1 \
      -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
      -x FLAGCX_MEM_ENABLE=1 \
      -x FLAGCX_VMM_ENABLE=1 \
      -x FLAGCX_IB_DISABLE=1 \
      "${vmm_provider_env[@]+"${vmm_provider_env[@]}"}" "$@" \
      "$BUILD_BIN/symmem_mpi_tests" --gtest_filter="$test_case"
}

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
    "${vmm_provider_env[@]+"${vmm_provider_env[@]}"}" \
    "$BUILD_BIN/symmem_mpi_tests" \
    --gtest_filter=-SymMemTest.CrossGpuReadViaIpcPeerPtr:SymMemTest.CrossGpuWriteViaIpcPeerPtr:SymMemTest.HybridLocalAndRemoteAccess

if ((run_multi_segment)); then
  provider_multi_segment_status=0
  for provider_test in \
    SymMemTest.SegmentedVmmSubrangeMapsOnlyTouchedSegment \
    SymMemTest.SegmentedVmmWindowMapsDifferentPeerLayouts; do
    echo ""
    echo "=== Running $provider_test (np=$LOCAL_NP) ==="
    if ! run_required_case "$provider_test" -x FLAGCX_CI_REQUIRE_MULTI_SEGMENT=1; then
      provider_multi_segment_status=1
    fi
  done
  if ((provider_multi_segment_status)); then
    exit 1
  fi
fi

echo ""
echo "=== Running untracked VMM mapping rejection (np=$LOCAL_NP) ==="
run_required_case SymMemTest.UntrackedVmmMappingIsRejected \
  -x FLAGCX_CI_REQUIRE_PROVIDER_VMM=1

if [[ "${FLAGCX_CI_REQUIRE_MULTICAST:-0}" == "1" ]]; then
  case "${FLAGCX_ADAPTOR:-}" in
    nvidia|metax)
      echo ""
      echo "=== Running required multicast LSA MPI tests (np=$LOCAL_NP) ==="
      for provider_test in \
        SymMemTest.MulticastMappingWhenSupported \
        SymMemTest.LsaMulticastOffsetsReuseAndBindFailureFallsBack \
        SymMemTest.LsaMulticastUnbindFailureRetainsWindowForRetry; do
        run_required_case "$provider_test" \
          -x FLAGCX_CI_REQUIRE_VMM=1 \
          -x FLAGCX_CI_REQUIRE_MULTICAST=1
      done
      ;;
    *)
      echo "multicast LSA is only available on nvidia and metax" >&2
      exit 1
      ;;
  esac
fi

echo ""
echo "All symmem tests passed."
