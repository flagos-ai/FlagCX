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
test_status=0
vmm_provider_env=()
run_multi_segment=0
# Keep this fixed list aligned with the provider cases exercised by hardware CI.
# DU multi-segment mapping is still under driver validation.
case "${FLAGCX_ADAPTOR:-}" in
  nvidia|metax|ppu|ascend|iluvatar)
    run_multi_segment=1
    ;;
  enflame)
    ;;
  du)
    vmm_provider_env+=( -x FLAGCX_VMM_MULTI_SEGMENT_ENABLE=0 )
    ;;
  klx)
    ;;
  "")
    ;;
  *)
    echo "Unsupported FLAGCX_ADAPTOR '${FLAGCX_ADAPTOR}'; expected nvidia, du, ppu, metax, ascend, klx, or iluvatar" >&2
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
      -x FLAGCX_DMABUF_ENABLE="${FLAGCX_DMABUF_ENABLE:-1}" \
      -x FLAGCX_IB_DISABLE=1 \
      "${vmm_provider_env[@]+"${vmm_provider_env[@]}"}" "$@" \
      "$BUILD_BIN/symmem_mpi_tests" --gtest_filter="$test_case"
}

echo "=== Running symmem unit tests (no MPI/GPU) ==="
FLAGCX_CI_TEST_LABEL="symmem unit tests" \
  "$TEST_RUNNER" "$BUILD_BIN/symmem_unit_tests" || test_status=$?

# Host resource shims + real bootstrap; three processes, one rank each. These
# cases need no accelerator, and the third process exercises an empty node.
mkdir -p "$PROJECT_ROOT/.ci-results"
lifecycle_xml=$(mktemp -d "$PROJECT_ROOT/.ci-results/symmem-lifecycle.XXXXXX")
lifecycle_cases=(
  HeteroFinalizeMpi.RmaSyncAllocationsReleaseLocallyAfterFinalize
  HeteroFinalizeMpi.ProxyWorkWaitConvergesAndLocalReleaseCanRetry
  HeteroFinalizeMpi.PendingWorkKeepsDestroyOwnerUntilCompletion
  HeteroFinalizeMpi.WorkWaitFailureConvergesBeforeAnyImportCloses
  HeteroFinalizeMpi.IndependentCommunicatorWorkDoesNotDelayClosure
  HeteroFinalizeMpi.RelayCloseFailureKeepsEveryServiceAliveUntilRetry
  HeteroFinalizeMpi.DelayedImportAndEmptyRankConfirmBeforeLocalDestroy
  HeteroFinalizeMpi.FailedImportRetainsExporterAndLocalReleaseCanRetry
  HeteroFinalizeMpi.ClosureDoesNotWaitForAnotherNode
  HeteroFinalizeMpi.DeviceSelectionFailureConvergesBeforeClosure
  SymWindowRollback.NativeOnlyImplicitDestroyFinishesWhilePeersWait
  SymWindowRollback.TransportDeclarationIncludesEmptyRanksAtInit
  SymWindowRollback.IpcDeclarationSurvivesRankLocalAllocationFailure
  SymWindowRollback.IpcDeclarationAndClosureAreNodeLocal
  SymWindowRollback.PairCommunicatorsCloseInCanonicalOrder
  SymWindowRollback.FinalizeCloseFailureConvergesIncludingEmptyRanks
  SymWindowRollback.FinalizedDestroyDoesNotNeedPeers
  SymWindowRollback.CommDestroyPendingWindowKeepsLeaseUntilImportsClose
  SymWindowRollback.MixedNativeScopesConvergeBeforeNativeClose
)
lifecycle_filter=$(IFS=:; echo "${lifecycle_cases[*]}")
FLAGCX_CI_MPI_LABEL="symmem lifecycle bootstrap" \
  "$MPI_RUNNER" -np 3 --allow-run-as-root \
    -x FLAGCX_USE_HETERO_COMM=1 -x FLAGCX_IB_DISABLE=1 \
    -x FLAGCX_P2P_DISABLE=0 -x FLAGCX_DEVICE_ONE_SIDED_FORCE_NET=0 \
    -x FLAGCX_VMM_ENABLE=0 \
    bash "$PROJECT_ROOT/.github/scripts/ci/run_gtest_rank_xml.sh" \
    "$lifecycle_xml" "$BUILD_BIN/symmem_mpi_tests" \
    --gtest_filter="$lifecycle_filter" || test_status=$?
lifecycle_required=()
for test_case in "${lifecycle_cases[@]}"; do
  lifecycle_required+=( --case "$test_case" )
done
python3 "$PROJECT_ROOT/.github/scripts/ci/check_required_gtests.py" \
  "${lifecycle_required[@]}" "$lifecycle_xml/rank-0.xml" \
  "$lifecycle_xml/rank-1.xml" "$lifecycle_xml/rank-2.xml" || test_status=$?

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
    --gtest_filter=-SymMemTest.CrossGpuReadViaPeerPtr:SymMemTest.CrossGpuWriteViaPeerPtr:SymMemTest.HybridLocalAndRemoteAccess:CoverageSupplemental.* || test_status=$?

echo ""
vmm_skip_reason=$(python3 "$PROJECT_ROOT/.github/scripts/ci/ignore_test.py" \
  phase symmem vmm-local-and-remote-without-net)
if [[ "${FLAGCX_CI_SYMMEM_VMM_SUPPORTED:-1}" != 1 && -z "$vmm_skip_reason" ]]; then
  vmm_skip_reason="VMM is unavailable on this runtime"
fi
echo "=== Running symmem VMM-local MPI tests (np=$LOCAL_NP) ==="
if [[ -z "$vmm_skip_reason" ]]; then
  FLAGCX_CI_MPI_LABEL="symmem VMM local" \
    "$MPI_RUNNER" -np "$LOCAL_NP" --allow-run-as-root \
      -x FLAGCX_USE_HETERO_COMM=1 \
      -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
      -x FLAGCX_MEM_ENABLE=1 \
      -x FLAGCX_VMM_ENABLE=1 \
      -x FLAGCX_DMABUF_ENABLE="${FLAGCX_DMABUF_ENABLE:-1}" \
      -x FLAGCX_IB_DISABLE=1 \
      -x FLAGCX_CI_REQUIRE_VMM=1 \
      "${vmm_provider_env[@]+"${vmm_provider_env[@]}"}" \
      "$BUILD_BIN/symmem_mpi_tests" \
      --gtest_filter=-SymMemTest.CrossGpuReadViaIpcPeerPtr:SymMemTest.CrossGpuWriteViaIpcPeerPtr:SymMemTest.HybridLocalAndRemoteAccess:CoverageSupplemental.* || test_status=$?
else
  echo "Skipping symmem VMM-local tests: $vmm_skip_reason"
fi

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
    test_status=1
  fi
fi

if [[ "${FLAGCX_ADAPTOR:-}" == iluvatar ]]; then
  # CoreX crashes when the raw VMM address reaches cudaIpcGetMemHandle.
  echo "Skipping UntrackedVmmMappingIsRejected on Iluvatar: CoreX VMM-to-runtime-IPC export crashes"
elif [[ -z "$vmm_skip_reason" && -n "${FLAGCX_ADAPTOR:-}" &&
        "${FLAGCX_ADAPTOR}" != ascend ]]; then
  echo ""
  echo "=== Running untracked VMM mapping rejection (np=$LOCAL_NP) ==="
  untracked_status=0
  run_required_case SymMemTest.UntrackedVmmMappingIsRejected \
    -x FLAGCX_CI_REQUIRE_PROVIDER_VMM=1 || untracked_status=$?
  if ((untracked_status != 0)); then
    if [[ "$FLAGCX_ADAPTOR" == nvidia ]]; then
      bash "$PROJECT_ROOT/.github/scripts/ci/run_ipc_vmm_diagnostics.sh" \
        "$BUILD_BIN/ipc_vmm_repro" \
        "${FLAGCX_CI_LOG_DIR:-$SYMMEM_DIR/build/results}/ipc-vmm-diagnostics" || \
        echo 'IPC reproducer diagnostics failed; retaining the original SymMem failure' >&2
    fi
    test_status=$untracked_status
  fi
fi

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
          -x FLAGCX_CI_REQUIRE_MULTICAST=1 || test_status=$?
      done
      ;;
    *)
      echo "multicast LSA is only available on nvidia and metax" >&2
      exit 1
      ;;
  esac
fi

echo ""
if ((test_status != 0)); then
  echo "Symmem tests failed (exit=$test_status)" >&2
  exit "$test_status"
fi
echo "All symmem tests passed."
