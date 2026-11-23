#!/usr/bin/env bash
set -euo pipefail

project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
perf_bin="$project_root/test/perf/host_api/build/bin"
runner_bin="$project_root/test/unittest/runner/build/bin/runner_unirunner_experimental_mpi_tests"
mpi_runner="$project_root/.github/scripts/ci/run_mpi_with_timeout.sh"
log_dir=${FLAGCX_CI_LOG_DIR:-$project_root/.ci-logs}
mkdir -p "$log_dir"

echo "Checking MUSA SDK support for Device API (musa::atomic_ref)"
make -C "$project_root/test/kernel/musa" --jobs="$(nproc)" \
  "${FLAGCX_CI_TEST_MAKE_ARGS[@]}" unirunner-probe
bash "$project_root/.github/scripts/ci/run_with_timeout.sh" \
  "$project_root/test/kernel/musa/build/bin/unirunner_probe"
make -C "$project_root" --jobs="$(nproc)" "${FLAGCX_CI_PROJECT_MAKE_ARGS[@]}"
make -C "$project_root/test/perf" host_api --jobs="$(nproc)" \
  "${FLAGCX_CI_TEST_MAKE_ARGS[@]}"
cmake -S "$project_root/third-party/googletest" \
  -B "$project_root/third-party/googletest/build"
cmake --build "$project_root/third-party/googletest/build" --parallel "$(nproc)"
make -C "$project_root/test/unittest/runner" --jobs="$(nproc)" \
  "${FLAGCX_CI_TEST_MAKE_ARGS[@]}" unirunner-experimental

common_env=(
  -x PATH -x LD_LIBRARY_PATH -x FLAGCX_DEBUG -x FLAGCX_DEBUG_SUBSYS
  -x FLAGCX_IB_HCA
  -x MUSA_VISIBLE_DEVICES="$FLAGCX_CI_MUSA_DEVICES"
  -x FLAGCX_USE_HETERO_COMM=1 -x FLAGCX_MEM_ENABLE=1 -x FLAGCX_VMM_ENABLE=0
  -x FLAGCX_CLUSTER_SPLIT_LIST=2 -x FLAGCX_P2P_DISABLE=0 -x FLAGCX_IB_DISABLE=0
  -x FLAGCX_CUSTOM_OP_ENABLE=0
  -x FLAGCX_CI_EXPECT_RUNNER_MODE=HYBRID
  -x FLAGCX_CI_EXPECT_UNIRUNNER_KERNEL=1 -x FLAGCX_CI_EXPECT_STREAM_VALUE=1
  -x FLAGCX_CI_EXPECT_LOCAL_P2P=1
)

run_runner_test() {
  local label=$1 algorithm=$2 filter=$3
  local result_dir rank case_name
  local -a algorithm_env=() required_cases=() xml_files=() cases=()
  if [[ "$algorithm" != default ]]; then
    algorithm_env=(-x FLAGCX_UNIRUNNER_ALGO="$algorithm")
  fi
  result_dir=$(mktemp -d "$log_dir/musa-${label// /_}.XXXXXX")
  FLAGCX_CI_MPI_LABEL="MUSA experimental $label" \
    env -u FLAGCX_UNIRUNNER_ALGO "$mpi_runner" -np 4 --allow-run-as-root \
      "${common_env[@]}" ${algorithm_env[@]+"${algorithm_env[@]}"} \
      bash "$project_root/.github/scripts/ci/run_gtest_rank_xml.sh" \
      "$result_dir" "$runner_bin" --gtest_filter="$filter"
  IFS=: read -r -a cases <<< "$filter"
  for case_name in "${cases[@]}"; do
    required_cases+=(--case "$case_name")
  done
  # Require every rank's report, including named tests that must not skip.
  for rank in 0 1 2 3; do
    xml_files+=("$result_dir/rank-$rank.xml")
  done
  python3 "$project_root/.github/scripts/ci/check_required_gtests.py" \
    "${required_cases[@]}" "${xml_files[@]}"
}

run_perf_test() {
  local label=$1 algorithm=$2 binary=$3
  shift 3
  local -a algorithm_env=()
  if [[ "$algorithm" != default ]]; then
    algorithm_env=(-x FLAGCX_UNIRUNNER_ALGO="$algorithm")
  fi
  if [[ "$algorithm" == 7 ]]; then
    algorithm_env+=(-x FLAGCX_UNIRUNNER_GROUPSIZE=1)
  fi
  FLAGCX_CI_MPI_LABEL="MUSA experimental $label" \
    env -u FLAGCX_UNIRUNNER_ALGO "$mpi_runner" -np 4 --allow-run-as-root \
      "${common_env[@]}" ${algorithm_env[@]+"${algorithm_env[@]}"} "$perf_bin/$binary" -v 1 "$@"
}

run_runner_test AllGatherGemm 9 \
  'FlagCXCollTest.AllGatherGemm:FlagCXCollTest.AllGatherGemmConvergesRankLocalSizeFailure'
run_runner_test GemmReduceScatter 11 \
  'FlagCXCollTest.GemmReduceScatter:FlagCXCollTest.GemmReduceScatterConvergesRankLocalSizeFailure'
run_runner_test 'group and IPC errors' default \
  'FlagCXCollTest.UniRunnerDagAlgorithmsRejectOuterGroup:FlagCXCollTest.UniRunnerIpcPreparationErrorConverges:FlagCXCollTest.UniRunnerUnsupportedReductionTypes'
run_runner_test 'P2pAG outer group' 8 'FlagCXCollTest.AllGatherInsideGroup'
run_runner_test 'AllReduce streams and in-place' default \
  'FlagCXCollTest.UniRunnerAllReduceStreamsAndInPlace'

run_perf_test IpcRingAR 2 perf_allreduce -b 64K -e 4M -f 2 -w 2 -n 5
run_perf_test 'IPC buffer rebinding' 2 perf_ipc_rebind \
  -b 4K -e 4K -d float32 -o sum -R 0 -w 0 -n 1
run_perf_test 'RingAG uneven count' 4 perf_allreduce \
  -b 40 -e 40 -d float32 -w 1 -n 1
run_perf_test GroupedAG 7 perf_allgather -b 64K -e 4M -f 2 -w 2 -n 5

# The native reduction executor must be exercised for each supported type and
# operation. Keep this correctness sweep small; IpcRingAR above measures sizes.
for datatype in int8 uint8 int32 uint32 int64 uint64 float32 float64; do
  for operation in sum prod min max avg; do
    run_perf_test "RingAR $datatype $operation" 0 perf_allreduce \
      -b 128 -e 128 -d "$datatype" -o "$operation" -w 1 -n 2
  done
done
for operation in sum prod min max avg; do
  run_perf_test "Reduce $operation" default perf_reduce \
    -b 4K -e 4K -d float32 -o "$operation" -r 3 -w 1 -n 2
  run_perf_test "ReduceScatter $operation" default perf_reducescatter \
    -b 4K -e 4K -d float32 -o "$operation" -w 1 -n 2
done

FLAGCX_CI_MPI_LABEL='MUSA experimental SlicedAR' \
  env -u FLAGCX_UNIRUNNER_ALGO "$mpi_runner" -np 4 --allow-run-as-root \
    "${common_env[@]}" -x FLAGCX_UNIRUNNER_ALGO=6 \
    -x FLAGCX_UNIRUNNER_GROUPSIZE=4 -x FLAGCX_UNIRUNNER_NSLICES=4 \
    -x FLAGCX_UNIRUNNER_LOCAL_NSLICES=2 -x FLAGCX_UNIRUNNER_NREDSLICES=0 \
    -x FLAGCX_UNIRUNNER_REDSLICESIZE=1048576 \
    -x FLAGCX_UNIRUNNER_MAX_BLOCKS=64 -x FLAGCX_UNIRUNNER_MAX_THREADS=512 \
    "$perf_bin/perf_allreduce" -v 1 \
    -b 64K -e "${FLAGCX_CI_MUSA_SLICED_AR_MAX_BYTES:-64M}" -f 2 -w 20 -n 200
