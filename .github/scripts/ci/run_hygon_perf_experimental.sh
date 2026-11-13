#!/usr/bin/env bash

set -euo pipefail

project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
perf_bin="$project_root/test/perf/host_api/build/bin"
runner_bin="$project_root/test/unittest/runner/build/bin/runner_unirunner_experimental_mpi_tests"
mpi_runner="$project_root/.github/scripts/ci/run_mpi_with_timeout.sh"

make -C "$project_root" --jobs="$(nproc)" \
  "${FLAGCX_CI_PROJECT_MAKE_ARGS[@]}"
make -C "$project_root/test/perf" --jobs="$(nproc)" \
  "${FLAGCX_CI_TEST_MAKE_ARGS[@]}"
cmake -S "$project_root/third-party/googletest" \
  -B "$project_root/third-party/googletest/build"
cmake --build "$project_root/third-party/googletest/build" --parallel "$(nproc)"
make -C "$project_root/test/unittest/runner" --jobs="$(nproc)" \
  "${FLAGCX_CI_TEST_MAKE_ARGS[@]}" unirunner-experimental

common_env=(
  -x PATH
  -x LD_LIBRARY_PATH
  -x HIP_VISIBLE_DEVICES="$FLAGCX_CI_HYGON_FOUR_GPU_DEVICES"
  -x FLAGCX_IB_HCA="$FLAGCX_CI_HYGON_CONNECTED_HCAS"
  -x FLAGCX_USE_HETERO_COMM=1
  -x FLAGCX_MEM_ENABLE=1
  -x FLAGCX_CLUSTER_SPLIT_LIST=2
  -x FLAGCX_P2P_DISABLE=0
  -x FLAGCX_CUSTOM_OP_ENABLE=0
  -x FLAGCX_CI_EXPECT_RUNNER_MODE=HYBRID
  -x FLAGCX_CI_EXPECT_UNIRUNNER_KERNEL=1
  -x FLAGCX_CI_EXPECT_STREAM_VALUE=1
  -x FLAGCX_CI_EXPECT_LOCAL_P2P=1
)

run_runner_test() {
  local label=$1 algorithm=$2 filter=$3
  FLAGCX_CI_MPI_LABEL="Hygon experimental $label" \
    "$mpi_runner" -np 4 --allow-run-as-root \
      "${common_env[@]}" -x FLAGCX_UNIRUNNER_ALGO="$algorithm" \
      "$runner_bin" --gtest_filter="$filter"
}

run_perf_test() {
  local label=$1 algorithm=$2 binary=$3
  shift 3
  local -a algorithm_env=()
  if [[ "$algorithm" == 7 ]]; then
    algorithm_env=(-x FLAGCX_UNIRUNNER_GROUPSIZE=1)
  fi
  FLAGCX_CI_MPI_LABEL="Hygon experimental $label" \
    "$mpi_runner" -np 4 --allow-run-as-root \
      "${common_env[@]}" "${algorithm_env[@]}" \
      -x FLAGCX_UNIRUNNER_ALGO="$algorithm" \
      "$perf_bin/$binary" "$@"
}

run_runner_test 'AllGatherGemm' 9 \
  'FlagCXCollTest.AllGatherGemm*'
run_runner_test 'GemmReduceScatter' 11 \
  'FlagCXCollTest.GemmReduceScatter*'
FLAGCX_CI_MPI_LABEL='Hygon experimental UniRunner group and IPC errors' \
  env -u FLAGCX_UNIRUNNER_ALGO "$mpi_runner" -np 4 --allow-run-as-root \
    "${common_env[@]}" "$runner_bin" \
    --gtest_filter='FlagCXCollTest.UniRunnerDagAlgorithmsRejectOuterGroup:FlagCXCollTest.UniRunnerIpcPreparationErrorConverges'
run_runner_test 'P2pAG outer group' 8 \
  'FlagCXCollTest.AllGatherInsideGroup'

run_perf_test 'IpcRingAR' 2 perf_allreduce \
  -b 64K -e 4M -f 2 -w 2 -n 5
run_perf_test 'IPC buffer rebinding' 2 perf_ipc_rebind \
  -b 4K -e 4K -d float32 -o sum -R 0 -w 0 -n 1
run_perf_test 'RingAG uneven count' 4 perf_allreduce \
  -b 40 -e 40 -f 2 -d float32 -w 1 -n 1
run_perf_test 'GroupedAG' 7 perf_allgather \
  -b 64K -e 4M -f 2 -w 2 -n 5

sliced_env=(
  -x FLAGCX_UNIRUNNER_GROUPSIZE=4
  -x FLAGCX_UNIRUNNER_NSLICES=4
  -x FLAGCX_UNIRUNNER_LOCAL_NSLICES=2
  -x FLAGCX_UNIRUNNER_NREDSLICES=0
  -x FLAGCX_UNIRUNNER_REDSLICESIZE=1048576
  -x FLAGCX_UNIRUNNER_MAX_BLOCKS=64
  -x FLAGCX_UNIRUNNER_MAX_THREADS=512
)
# Keep the default sweep within the DCU CI memory budget; the upper bound
# remains configurable for a dedicated large-message run.
FLAGCX_CI_MPI_LABEL='Hygon experimental SlicedAR' \
  "$mpi_runner" -np 4 --allow-run-as-root \
    "${common_env[@]}" "${sliced_env[@]}" -x FLAGCX_UNIRUNNER_ALGO=6 \
    "$perf_bin/perf_allreduce" \
    -b 64K -e "${FLAGCX_CI_HYGON_SLICED_AR_MAX_BYTES:-64M}" -f 2 -w 20 -n 200
