#!/usr/bin/env bash

set -euo pipefail

project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
export PATH="$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$project_root/build/lib:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}"
perf_bin="$project_root/test/perf/host_api/build/bin"
device_api_bin="$project_root/test/perf/device_api/build/bin"
cuda_extended_runner="$project_root/test/unittest/runner/build/bin/runner_cuda_extended_mpi_tests"
mpi_runner="$project_root/.github/scripts/ci/run_mpi_with_timeout.sh"

make -C "$project_root" --jobs="$(nproc)" \
  USE_NVIDIA=1 COMPILE_KERNEL=1 FORCE_DEFAULT_PATH=1
make -C "$project_root/test/perf" --jobs="$(nproc)" \
  USE_NVIDIA=1 COMPILE_KERNEL=1 FORCE_DEFAULT_PATH=1
cmake -S "$project_root/third-party/googletest" \
  -B "$project_root/third-party/googletest/build"
cmake --build "$project_root/third-party/googletest/build" --parallel "$(nproc)"
make -C "$project_root/test/unittest/runner" --jobs="$(nproc)" \
  USE_NVIDIA=1 COMPILE_KERNEL=1 FORCE_DEFAULT_PATH=1 cuda-extended

source "$project_root/test/script/_gpu_check.sh"
wait_for_gpu

# Run UniRunner-only GPU tests here, outside the cross-platform runner suite.
cuda_extended_runner_env=(
  -x PATH
  -x LD_LIBRARY_PATH
  -x CUDA_VISIBLE_DEVICES=0,1,2,3
  -x FLAGCX_USE_HETERO_COMM=1
  -x FLAGCX_MEM_ENABLE=1
  -x FLAGCX_CLUSTER_SPLIT_LIST=2
  -x FLAGCX_P2P_DISABLE=0
  -x FLAGCX_CI_EXPECT_RUNNER_MODE=HYBRID
)
FLAGCX_CI_MPI_LABEL='CUDA extended UniRunner fused and group tests' \
  env -u FLAGCX_UNIRUNNER_ALGO "$mpi_runner" -np 4 --allow-run-as-root \
    "${cuda_extended_runner_env[@]}" "$cuda_extended_runner" \
    --gtest_filter='FlagCXCollTest.AllGatherGemm*:FlagCXCollTest.GemmReduceScatter*:FlagCXCollTest.UniRunnerDagAlgorithmsRejectOuterGroup:FlagCXCollTest.UniRunnerIpcPreparationErrorConverges'

FLAGCX_CI_MPI_LABEL='CUDA extended UniRunner P2P outer group' \
  "$mpi_runner" -np 4 --allow-run-as-root \
    "${cuda_extended_runner_env[@]}" -x FLAGCX_UNIRUNNER_ALGO=8 \
    "$cuda_extended_runner" \
    --gtest_filter=FlagCXCollTest.AllGatherInsideGroup

# Preserve the original extended perf-test coverage.
export FLAGCX_CI_PERF_PLATFORM='CUDA extended'
"$project_root/.github/scripts/ci/run_host_perf_suite.sh" homogeneous "$perf_bin"

"$mpi_runner" -np 4 --allow-run-as-root "$perf_bin/perf_allreduce" \
  -b 128M -e 1G -f 2 -p 1 -R 1

# Exercise uniRunner sliced AllReduce across four GPUs on the CUDA runner.
# Use OpenMPI's -x syntax; PATH and LD_LIBRARY_PATH were set above for this job.
unirunner_sliced_ar_env=(
  -x PATH
  -x LD_LIBRARY_PATH
  -x NCCL_NVLS_ENABLE=0
  -x FLAGCX_IB_HCA=mlx5
  -x CUDA_VISIBLE_DEVICES=0,1,2,3
  -x FLAGCX_ENABLE_TOPO_DETECT=TRUE
  -x FLAGCX_DEBUG=
  -x FLAGCX_DEBUG_SUBSYS=
  -x FLAGCX_USE_HETERO_COMM=1
  -x FLAGCX_CUSTOM_OP_ENABLE=0
  -x FLAGCX_UNIRUNNER_GROUPSIZE=4
  -x FLAGCX_UNIRUNNER_NSLICES=4
  -x FLAGCX_UNIRUNNER_LOCAL_NSLICES=2
  -x FLAGCX_UNIRUNNER_NREDSLICES=0
  -x FLAGCX_UNIRUNNER_REDSLICESIZE=1048576
  -x FLAGCX_UNIRUNNER_MAX_BLOCKS=64
  -x FLAGCX_UNIRUNNER_MAX_THREADS=512
  -x FLAGCX_UNIRUNNER_ALGO=6
)
FLAGCX_CI_MPI_LABEL='CUDA extended uniRunner sliced AllReduce' \
  "$mpi_runner" -np 4 --allow-run-as-root \
    "${unirunner_sliced_ar_env[@]}" \
    "$perf_bin/perf_allreduce" -b 64K -e 4G -f 2 -w 20 -n 200

FLAGCX_CI_MPI_LABEL='CUDA extended uniRunner IPC Ring AllReduce' \
  "$mpi_runner" -np 4 --allow-run-as-root \
    -x PATH -x LD_LIBRARY_PATH \
    -x CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -x FLAGCX_USE_HETERO_COMM=1 \
    -x FLAGCX_UNIRUNNER_ALGO=2 \
    "$perf_bin/perf_allreduce" -b 64K -e 4M -f 2 -w 2 -n 5

FLAGCX_CI_MPI_LABEL='CUDA extended uniRunner IPC buffer rebinding' \
  "$mpi_runner" -np 4 --allow-run-as-root \
    -x PATH -x LD_LIBRARY_PATH \
    -x CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -x FLAGCX_USE_HETERO_COMM=1 \
    -x FLAGCX_UNIRUNNER_ALGO=2 \
    "$perf_bin/perf_ipc_rebind" -b 4K -e 4K -d float32 -o sum -R 0 -w 0 -n 1

# RingAG splits remainder elements among the first ranks.
FLAGCX_CI_MPI_LABEL='CUDA extended uniRunner RingAG uneven count' \
  "$mpi_runner" -np 4 --allow-run-as-root \
    -x PATH -x LD_LIBRARY_PATH \
    -x CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -x FLAGCX_USE_HETERO_COMM=1 \
    -x FLAGCX_UNIRUNNER_ALGO=4 \
    "$perf_bin/perf_allreduce" -b 40 -e 40 -f 2 -d float32 -w 1 -n 1

# Four groups of one rank exercise GroupedAG's later inter-group rounds.
FLAGCX_CI_MPI_LABEL='CUDA extended uniRunner grouped AllGather' \
  "$mpi_runner" -np 4 --allow-run-as-root \
    -x PATH -x LD_LIBRARY_PATH \
    -x CUDA_VISIBLE_DEVICES=0,1,2,3 \
    -x FLAGCX_USE_HETERO_COMM=1 \
    -x FLAGCX_UNIRUNNER_ALGO=7 \
    -x FLAGCX_UNIRUNNER_GROUPSIZE=1 \
    "$perf_bin/perf_allgather" -b 64K -e 4M -f 2 -w 2 -n 5

for operation in sendrecv alltoall; do
  "$mpi_runner" -np 4 --allow-run-as-root \
    -x FLAGCX_MEM_ENABLE=1 -x FLAGCX_VMM_ENABLE=0 \
    -x FLAGCX_USE_HETERO_COMM=1 \
    "$perf_bin/perf_$operation" -b 128M -e 1G -f 2 -p 1 -R 1

  "$mpi_runner" -np 4 --allow-run-as-root \
    -x FLAGCX_MEM_ENABLE=1 -x FLAGCX_VMM_ENABLE=0 \
    -x FLAGCX_USE_HETERO_COMM=1 -x FLAGCX_P2P_DISABLE=1 \
    "$perf_bin/perf_$operation" -b 128M -e 1G -f 2 -p 1 -R 1
done

for operation in put get; do
  "$mpi_runner" -np 2 --allow-run-as-root \
    -x FLAGCX_MEM_ENABLE=1 -x FLAGCX_VMM_ENABLE=0 \
    -x FLAGCX_USE_HETERO_COMM=1 \
    "$perf_bin/perf_$operation" -b 8 -e 1M -f 2 -R 2
done

"$mpi_runner" -np 4 --allow-run-as-root \
  -x FLAGCX_USE_HETERO_COMM=1 -x FLAGCX_MEM_ENABLE=1 \
  -x FLAGCX_VMM_ENABLE=0 -x FLAGCX_P2P_DISABLE=1 \
  -x FLAGCX_DEBUG=INFO -x FLAGCX_DEBUG_SUBSYS=ALL -x LD_LIBRARY_PATH \
  "$device_api_bin/perf_allreduce_intranode" -b 1M -e 64M -f 2 -R 1

common_env=(
  -x FLAGCX_USE_HETERO_COMM=1
  -x FLAGCX_MEM_ENABLE=1
  -x FLAGCX_VMM_ENABLE=0
  -x FLAGCX_P2P_DISABLE=1
  -x LD_LIBRARY_PATH
)
"$mpi_runner" --allow-run-as-root \
  -np 2 "${common_env[@]}" "${FLAGCX_CI_NODE1_MPI_ARGS[@]}" \
  "$device_api_bin/perf_internode_onesided" -b 1M -e 64M -f 2 -R 1 \
  : -np 2 "${common_env[@]}" "${FLAGCX_CI_NODE2_MPI_ARGS[@]}" \
  "$device_api_bin/perf_internode_onesided" -b 1M -e 64M -f 2 -R 1

FLAGCX_P2P_PERF_OP=write \
  "$mpi_runner" -np 2 --allow-run-as-root "$perf_bin/perf_p2p_engine" \
    -b 4K -e 64M -f 2 -n 10

shared_project_build="$project_root/build-p2p-shared"
shared_test_build="$project_root/test/perf/host_api/build-p2p-shared"
make -C "$project_root" --jobs="$(nproc)" \
  USE_NVIDIA=1 COMPILE_KERNEL=1 FORCE_DEFAULT_PATH=1 \
  USE_SHARED_P2P_ENGINE=1 BUILDDIR="$shared_project_build"
make -C "$project_root/test/perf/host_api" \
  USE_NVIDIA=1 COMPILE_KERNEL=1 FORCE_DEFAULT_PATH=1 \
  USE_SHARED_P2P_ENGINE=1 BUILDDIR="$shared_test_build" \
  FLAGCX_LIB="$shared_project_build/lib" \
  "$shared_test_build/bin/perf_p2p_engine"

LD_LIBRARY_PATH="$shared_project_build/lib:$LD_LIBRARY_PATH" \
  FLAGCX_P2P_PERF_OP=read \
  "$mpi_runner" -np 2 --allow-run-as-root \
    "$shared_test_build/bin/perf_p2p_engine" \
    -b 4K -e 64M -f 2 -n 10
