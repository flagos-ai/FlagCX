#!/usr/bin/env bash

set -euo pipefail
umask 022

project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
script_dir="$project_root/.github/scripts/ci"
source "$script_dir/timeouts.sh"
export FLAGCX_CI_MPI_TIMEOUT=$FLAGCX_CI_PERF_TIMEOUT
export FLAGCX_CI_TEST_MAX_ATTEMPTS=1
export FLAGCX_DEBUG=INFO
export FLAGCX_DEBUG_SUBSYS=INIT,NET,P2P,ENV,UNIRUNNER
export FLAGCX_SIGNAL_HOST_ENABLE=0
perf_bin="$project_root/test/perf/host_api/build/bin"
runner_bin="$project_root/test/unittest/runner/build/bin/runner_unirunner_experimental_mpi_tests"
mpi_runner="$script_dir/run_mpi_with_timeout.sh"
result_root="${FLAGCX_CI_LOG_DIR:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}/flagcx-ci-attempts}/metax-experimental"
mkdir -p "$result_root"
chmod 0755 "$result_root"
results="$result_root/results.tsv"
printf 'case\tresult\texit_code\telapsed_seconds\n' > "$results"
labels=(AllGatherGemm GemmReduceScatter group-and-ipc-errors P2pAG IpcRingAR ipc-rebind RingAG GroupedAG SlicedAR)
completed=0
publish_results() {
  local status=$? index
  trap - EXIT
  if ((status == 0 && completed != ${#labels[@]})); then status=1; fi
  for ((index=completed; index<${#labels[@]}; ++index)); do
    printf '%s\tBLOCKED\t-\t-\n' "${labels[index]}" >> "$results"
  done
  chmod 0644 "$results"
  cat "$results"
  exit "$status"
}
trap publish_results EXIT

flagcx_ci_run preflight mxcc --version
flagcx_ci_run build make -C "$project_root" --jobs="$(nproc)" \
  "${FLAGCX_CI_PROJECT_MAKE_ARGS[@]}"
flagcx_ci_run build-tests make -C "$project_root/test/perf" --jobs="$(nproc)" \
  "${FLAGCX_CI_TEST_MAKE_ARGS[@]}" MPI_HOME="$MPI_HOME" CCL_HOME=/opt/maca host_api
flagcx_ci_run build-tests cmake -S "$project_root/third-party/googletest" \
  -B "$project_root/third-party/googletest/build"
flagcx_ci_run build-tests cmake --build "$project_root/third-party/googletest/build" --parallel "$(nproc)"
flagcx_ci_run build-tests make -C "$project_root/test/unittest/runner" --jobs="$(nproc)" \
  "${FLAGCX_CI_TEST_MAKE_ARGS[@]}" unirunner-experimental

common_env=(
  -x PATH -x LD_LIBRARY_PATH -x FLAGCX_DEBUG -x FLAGCX_DEBUG_SUBSYS
  -x MACA_VISIBLE_DEVICES -x FLAGCX_IB_HCA -x FLAGCX_IB_GID_INDEX
  -x FLAGCX_SOCKET_IFNAME -x FLAGCX_SIGNAL_HOST_ENABLE
  -x FLAGCX_USE_HETERO_COMM=1 -x FLAGCX_MEM_ENABLE=1 -x FLAGCX_VMM_ENABLE=0
  -x FLAGCX_CLUSTER_SPLIT_LIST=2 -x FLAGCX_P2P_DISABLE=0 -x FLAGCX_CUSTOM_OP_ENABLE=0
  -x FLAGCX_CI_EXPECT_RUNNER_MODE=HYBRID
  -x FLAGCX_CI_EXPECT_UNIRUNNER_KERNEL=1
  -x FLAGCX_CI_EXPECT_DEVICE_STREAM_CONTROL=1
  -x FLAGCX_CI_EXPECT_LOCAL_P2P=1
)

# Stop on the first failed invocation. Even a normally exited MPI process can
# leave a poisoned GPU context; no subsequent test may obscure that failure.
run_case() {
  local label=$1 start=$SECONDS status=0
  shift
  FLAGCX_CI_MPI_LABEL="MetaX experimental $label" "$@" || status=$?
  if ((status == 0)); then
    printf '%s\tPASS\t0\t%s\n' "$label" "$((SECONDS-start))" >> "$results"
  else
    printf '%s\tFAIL\t%s\t%s\n' "$label" "$status" "$((SECONDS-start))" >> "$results"
  fi
  completed=$((completed+1))
  return "$status"
}

require_execution_log() {
  local log=$1 algorithm=$2
  if ! grep -Eq "UniRunner actual algorithm=$algorithm([^0-9]|$)" "$log"; then
    echo "Missing actual UniRunner algorithm=$algorithm evidence: $log" >&2
    return 1
  fi
  if ! grep -q "purpose=UNIRUNNER allocator=mcMalloc" "$log"; then
    echo "Missing MetaX UniRunner allocation policy evidence: $log" >&2
    return 1
  fi
  case "$algorithm" in
    2|6|9|11)
      if ! grep -q 'UniRunner native launcher=' "$log"; then
        echo "Missing native executor launch evidence: $log" >&2
        return 1
      fi
      ;;
  esac
}

run_runner_test() {
  local label=$1 algorithm=$2 filter=$3 xml_dir status=0
  shift 3
  local -a algorithm_env=() required=() reports=()
  if [[ "$algorithm" != default ]]; then
    algorithm_env=(-x FLAGCX_UNIRUNNER_ALGO="$algorithm")
  fi
  xml_dir=$(mktemp -d "$result_root/$label.XXXXXX") || return $?
  chmod 0755 "$xml_dir" || return $?
  env -u FLAGCX_UNIRUNNER_ALGO "$mpi_runner" -np 4 --allow-run-as-root \
    "${common_env[@]}" ${algorithm_env[@]+"${algorithm_env[@]}"} \
    bash "$script_dir/run_gtest_rank_xml.sh" "$xml_dir" \
    "$runner_bin" --gtest_filter="$filter" 2>&1 | tee "$xml_dir/invocation.log" || status=$?
  reports=("$xml_dir"/rank-*.xml)
  for report in "${reports[@]}"; do
    if [[ -f "$report" ]]; then chmod 0644 "$report" || return $?; fi
  done
  if ((status != 0)); then return "$status"; fi
  if [[ ${#reports[@]} -ne 4 || ! -f "${reports[0]}" ]]; then
    echo "MetaX $label: expected four rank XML reports" >&2
    return 1
  fi
  for case_name in "$@"; do required+=(--case "$case_name"); done
  python3 "$script_dir/check_required_gtests.py" "${required[@]}" "${reports[@]}" || return $?
  if [[ "$algorithm" != default ]]; then
    require_execution_log "$xml_dir/invocation.log" "$algorithm"
  fi
}

run_perf_test() {
  local algorithm=$1 binary=$2
  shift 2
  local -a algorithm_env=()
  if [[ "$algorithm" == 7 ]]; then
    algorithm_env=(-x FLAGCX_UNIRUNNER_GROUPSIZE=1)
  elif [[ "$algorithm" == 6 ]]; then
    algorithm_env=(
      -x FLAGCX_UNIRUNNER_GROUPSIZE=4 -x FLAGCX_UNIRUNNER_NSLICES=4
      -x FLAGCX_UNIRUNNER_LOCAL_NSLICES=2 -x FLAGCX_UNIRUNNER_NREDSLICES=0
      -x FLAGCX_UNIRUNNER_REDSLICESIZE=1048576
      -x FLAGCX_UNIRUNNER_MAX_BLOCKS=64 -x FLAGCX_UNIRUNNER_MAX_THREADS=512
    )
  fi
  "$mpi_runner" -np 4 --allow-run-as-root "${common_env[@]}" \
    ${algorithm_env[@]+"${algorithm_env[@]}"} -x FLAGCX_UNIRUNNER_ALGO="$algorithm" \
    "$perf_bin/$binary" "$@" 2>&1 | tee "$result_root/$binary-algorithm$algorithm.log" || return $?
  require_execution_log "$result_root/$binary-algorithm$algorithm.log" "$algorithm"
}

run_case AllGatherGemm run_runner_test AllGatherGemm 9 \
  'FlagCXCollTest.AllGatherGemm*' \
  FlagCXCollTest.AllGatherGemm FlagCXCollTest.AllGatherGemmConvergesRankLocalSizeFailure
run_case GemmReduceScatter run_runner_test GemmReduceScatter 11 \
  'FlagCXCollTest.GemmReduceScatter*' \
  FlagCXCollTest.GemmReduceScatter FlagCXCollTest.GemmReduceScatterConvergesRankLocalSizeFailure
run_case group-and-ipc-errors run_runner_test group-and-ipc-errors default \
  'FlagCXCollTest.UniRunnerDagAlgorithmsRejectOuterGroup:FlagCXCollTest.UniRunnerIpcPreparationErrorConverges' \
  FlagCXCollTest.UniRunnerDagAlgorithmsRejectOuterGroup FlagCXCollTest.UniRunnerIpcPreparationErrorConverges
run_case P2pAG run_runner_test P2pAG 8 FlagCXCollTest.AllGatherInsideGroup \
  FlagCXCollTest.AllGatherInsideGroup
run_case IpcRingAR run_perf_test 2 perf_allreduce -b 64K -e 4M -f 2 -w 2 -n 5
run_case ipc-rebind run_perf_test 2 perf_ipc_rebind \
  -b 4K -e 4K -d float32 -o sum -R 0 -w 0 -n 1
run_case RingAG run_perf_test 4 perf_allreduce -b 40 -e 40 -d float32 -w 1 -n 1
run_case GroupedAG run_perf_test 7 perf_allgather -b 64K -e 4M -f 2 -w 2 -n 5
run_case SlicedAR run_perf_test 6 perf_allreduce \
  -b 64K -e "${FLAGCX_CI_METAX_SLICED_AR_MAX_BYTES:-64M}" -f 2 -w 2 -n 5
