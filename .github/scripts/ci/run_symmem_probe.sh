#!/usr/bin/env bash
set -euo pipefail
# Logs/XML are uploaded by the non-root host runner after the container exits.
umask 022
project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
suite_dir="$project_root/test/unittest/symmem_probe"
runner="$project_root/.github/scripts/ci/run_with_timeout.sh"
mpi_runner="$project_root/.github/scripts/ci/run_mpi_with_timeout.sh"
rank_xml="$project_root/.github/scripts/ci/run_gtest_rank_xml.sh"
checker="$project_root/.github/scripts/ci/check_required_gtests.py"
log_root=${FLAGCX_CI_LOG_DIR:-"$suite_dir/build/results"}
mkdir -p "$log_root"
results=$(mktemp -d "$log_root/symmem-probe.XXXXXX")
chmod 755 "$results"
export FLAGCX_VMM_ENABLE=1 FLAGCX_DMABUF_ENABLE=0 FLAGCX_MEM_ENABLE=1
export FLAGCX_USE_HETERO_COMM=1 FLAGCX_DEBUG=INFO FLAGCX_DEBUG_SUBSYS=ALL
export FLAGCX_CI_TEST_MAX_ATTEMPTS=1 FLAGCX_CI_REQUIRE_GTEST_PASS=1
export FLAGCX_CI_TEST_DIAGNOSTICS_AFTER=60
status=0
local_allocation_status=0
# Inventory goes into the artifact even when the first provider call fails.
{
  uname -a
  command -v ixsmi >/dev/null && ixsmi -L
  command -v mpirun >/dev/null && mpirun --version
  command -v gdb >/dev/null && gdb --version
  command -v ibv_devinfo >/dev/null && ibv_devinfo
  printf 'CoreX header version declarations:\n'
  grep -E '^#define (CUDA_VERSION|CUDART_VERSION)' /usr/local/corex/include/cuda.h /usr/local/corex/include/cuda_runtime_api.h || true
} > "$results/inventory.log" 2>&1 || true

run_unit_case() {
  local test_case=$1 test_timeout=${2:-3m} case_status=0
  local case=${test_case#*.}
  FLAGCX_CI_TEST_LABEL="$test_case" FLAGCX_CI_TEST_TIMEOUT="$test_timeout" \
    "$runner" "$suite_dir/build/bin/symmem_probe_unit_tests" \
    "--gtest_filter=$test_case" "--gtest_output=xml:$results/$case.xml" || case_status=$?
  if ((case_status == 0)); then
    python3 "$checker" --case "$test_case" "$results/$case.xml" || case_status=$?
  fi
  return "$case_status"
}

if ! run_unit_case SymmemProbePrerequisites.VmmCallbacks 30s; then
  echo 'VMM prerequisite failed; component and MPI probes blocked. See iluvatar-vmm-sdk logs and VmmCallbacks.xml.' >&2
  exit 1
fi

# Older published images may not contain gdb. Bound the optional install;
# diagnostics still retain per-call markers and /proc state if it fails.
if ! command -v gdb >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
  timeout 90s bash -c 'apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends gdb' \
    > "$results/gdb-install.log" 2>&1 || echo 'gdb unavailable; retaining /proc diagnostics' >&2
fi

for case in AllocationAndRange MultiSegmentSubrange AliasAndRetryableTeardown LsaArenaOffsetsAndFdLifetime VaMrReadWrite VaMrMultiSegmentReadWrite; do
  if ! run_unit_case "VmmProbe.$case"; then
    status=1
    if [[ "$case" == AllocationAndRange ]]; then local_allocation_status=1; fi
  fi
done

if ((local_allocation_status != 0)); then
  echo 'MPI probes blocked by failed local VMM allocation; diagnostic suite failed.' >&2
  exit 1
fi

run_mpi_case() {
  local case=$1 mode=$2
  local test_case="MpiVmmProbe.$case" case_status=0 ranks=2
  local xml_dir="$results/$case"
  mkdir -p "$xml_dir"
  local -a common=(
    -x FLAGCX_USE_HETERO_COMM=1 -x FLAGCX_MEM_ENABLE=1
    -x FLAGCX_VMM_ENABLE=1 -x FLAGCX_DMABUF_ENABLE=0
    -x FLAGCX_DEBUG=INFO -x FLAGCX_DEBUG_SUBSYS=ALL -x LD_LIBRARY_PATH
  )
  local -a command=(--allow-run-as-root)
  local binary="$suite_dir/build/bin/symmem_probe_mpi_tests"
  if [[ "$mode" == local ]]; then
    command+=(-np 2 "${common[@]}" -x FLAGCX_IB_DISABLE=1 -x FLAGCX_P2P_DISABLE=0
      "${FLAGCX_CI_NODE1_MPI_ARGS[@]}" bash "$rank_xml" "$xml_dir" "$binary" "--gtest_filter=$test_case")
  else
    ranks=4
    local ib_disabled=0
    if [[ "$mode" == no-net ]]; then ib_disabled=1; fi
    if [[ -n "${FLAGCX_CI_PROBE_HOSTS:-}" ]]; then
      # Optional real two-host run: exactly two named hosts, two ranks each.
      # Both hosts must have the same executable/library/XML paths available.
      local host_a host_b extra
      IFS=, read -r host_a host_b extra <<< "$FLAGCX_CI_PROBE_HOSTS"
      if [[ -z "$host_a" || -z "$host_b" || -n "$extra" || "$host_a" == "$host_b" ]]; then
        echo 'FLAGCX_CI_PROBE_HOSTS must name two distinct hosts' >&2
        return 2
      fi
      command+=(--host "$host_a:2,$host_b:2" --map-by ppr:2:node -np 4
        "${common[@]}" -x FLAGCX_IB_DISABLE="$ib_disabled" -x FLAGCX_P2P_DISABLE=1
        -x COREX_VISIBLE_DEVICES=0,1 -x CUDA_VISIBLE_DEVICES=0,1
        bash "$rank_xml" "$xml_dir" "$binary" "--gtest_filter=$test_case")
    else
      command+=(-np 2 "${common[@]}" -x FLAGCX_IB_DISABLE="$ib_disabled" -x FLAGCX_P2P_DISABLE=1
        "${FLAGCX_CI_NODE1_MPI_ARGS[@]}" bash "$rank_xml" "$xml_dir" "$binary" "--gtest_filter=$test_case"
        : -np 2 "${common[@]}" -x FLAGCX_IB_DISABLE="$ib_disabled" -x FLAGCX_P2P_DISABLE=1
        "${FLAGCX_CI_NODE2_MPI_ARGS[@]}" bash "$rank_xml" "$xml_dir" "$binary" "--gtest_filter=$test_case")
    fi
  fi
  FLAGCX_CI_MPI_LABEL="$test_case" FLAGCX_CI_MPI_TIMEOUT=5m \
    "$mpi_runner" "${command[@]}" || case_status=$?
  if ((case_status == 0)); then
    local rank
    local -a xml=()
    for ((rank=0; rank<ranks; rank++)); do xml+=("$xml_dir/rank-$rank.xml"); done
    python3 "$checker" --case "$test_case" "${xml[@]}" || case_status=$?
  fi
  if ((case_status != 0)); then status=1; fi
  return 0
}
run_mpi_case LocalFdPeerMapping local
run_mpi_case LocalArenaWindowReuse local
run_mpi_case MapFailureAndRetry local
run_mpi_case RemoteWithoutNetRollsBack no-net
run_mpi_case WindowVaGetPut net
run_mpi_case MrFailureAndRetry net
run_mpi_case DeregistrationFailureAndRetry net
exit "$status"
