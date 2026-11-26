#!/bin/bash
set -euo pipefail
umask 022

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
  enflame|musa)
    ;;
  du)
    vmm_provider_env+=( -x FLAGCX_VMM_MULTI_SEGMENT_ENABLE=0 )
    ;;
  klx)
    ;;
  "")
    ;;
  *)
    echo "Unsupported FLAGCX_ADAPTOR '${FLAGCX_ADAPTOR}'; expected nvidia, du, ppu, metax, ascend, klx, iluvatar, enflame, or musa" >&2
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

echo ""
echo ""
echo "=== Running bootstrap IPC lifetime fault tests (four independent processes) ==="
lifecycle_log_root="${FLAGCX_CI_LOG_DIR:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}/flagcx-ci-attempts}"
lifecycle_root="$lifecycle_log_root/symmem-lifecycle"
mkdir -p "$lifecycle_root"
lifecycle_xml=$(mktemp -d "$lifecycle_root/results.XXXXXX")
# Explicit modes are needed for the host artifact uploader: mktemp ignores
# umask for directory permissions and the tests run in a root container.
chmod 0755 "$lifecycle_log_root" "$lifecycle_root" "$lifecycle_xml"
publish_lifecycle_xml() {
  local original_status=$? permission_status=0 report
  trap - EXIT
  for report in "$lifecycle_xml"/rank-*.xml; do
    [[ -f "$report" ]] || continue
    chmod 0644 "$report" || permission_status=$?
  done
  if ((original_status != 0)); then
    exit "$original_status"
  fi
  exit "$permission_status"
}
trap publish_lifecycle_xml EXIT
FLAGCX_CI_MPI_LABEL="symmem bootstrap lifetime" \
  "$MPI_RUNNER" -np 4 --allow-run-as-root \
    bash "$PROJECT_ROOT/.github/scripts/ci/run_gtest_rank_xml.sh" "$lifecycle_xml" \
    "$BUILD_BIN/symmem_mpi_tests" --gtest_filter=HeteroFinalizeMpi.* || test_status=$?
reports=("$lifecycle_xml"/rank-*.xml)
if [[ ! -f "${reports[0]}" || ${#reports[@]} -ne 4 ]]; then
  echo "Missing per-rank IPC lifetime reports in $lifecycle_xml" >&2
  if ((test_status == 0)); then test_status=1; fi
else
  lifecycle_check_status=0
  python3 "$PROJECT_ROOT/.github/scripts/ci/check_required_gtests.py" \
    --case HeteroFinalizeMpi.DelayedImportAndEmptyRanksConfirmBeforeLocalDestroy \
    --case HeteroFinalizeMpi.FailedImportRetainsExporterUntilCollectiveRetry \
    --case HeteroFinalizeMpi.DeviceSelectionFailureConvergesBeforeImportClosure \
    --case HeteroFinalizeMpi.SparseNodeClosureDoesNotWaitForAnotherNode \
    --case HeteroFinalizeMpi.InitializationDeclaresIpcAndNativeScopeOnEmptyRanks \
    --case HeteroFinalizeMpi.CollectiveCreationDeclaresClosureBeforePublication \
    --case HeteroFinalizeMpi.OuterFinalizeRetriesClosureThenEveryRankDestroysLocally \
    --case HeteroFinalizeMpi.CrossNodeRelayDrainFailureConvergesBeforeCleanup \
    --case HeteroFinalizeMpi.CrossNodeRelayDeviceFailureConvergesBeforeCleanup \
    "${reports[@]}" || lifecycle_check_status=$?
  if ((test_status == 0)); then test_status=$lifecycle_check_status; fi
fi

echo "=== Running symmem IPC-local MPI tests (np=$LOCAL_NP) ==="
FLAGCX_CI_MPI_LABEL="symmem IPC local" \
  "$MPI_RUNNER" -np "$LOCAL_NP" --allow-run-as-root \
    -x FLAGCX_USE_HETERO_COMM=1 \
    -x FLAGCX_CLUSTER_SPLIT_LIST=2 \
    -x FLAGCX_MEM_ENABLE=1 \
    -x FLAGCX_VMM_ENABLE=0 \
    -x FLAGCX_IB_DISABLE=1 \
    "$BUILD_BIN/symmem_mpi_tests" \
    --gtest_filter=-SymMemTest.CrossGpuReadViaPeerPtr:SymMemTest.CrossGpuWriteViaPeerPtr:SymMemTest.HybridLocalAndRemoteAccess:HeteroFinalizeMpi.*:CoverageSupplemental.* || test_status=$?

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
      --gtest_filter=-SymMemTest.CrossGpuReadViaIpcPeerPtr:SymMemTest.CrossGpuWriteViaIpcPeerPtr:SymMemTest.HybridLocalAndRemoteAccess:HeteroFinalizeMpi.*:CoverageSupplemental.* || test_status=$?
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
