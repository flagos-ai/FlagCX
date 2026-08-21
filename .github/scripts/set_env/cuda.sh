#!/usr/bin/env bash

# Everything in this file is CUDA/NVIDIA specific. The common workflow and
# runner deliberately do not inspect the platform name.

FLAGCX_CI_MPI_BASE_HOME=${MPI_HOME:-/usr/local/mpi}

# The FlagScale training image ships a launcher wrapper that passes "$@" as a
# literal argument. Use the real OpenMPI launcher through an isolated MPI_HOME
# so Makefiles and test scripts can keep invoking plain `mpirun`.
if [[ -x "$FLAGCX_CI_MPI_BASE_HOME/bin/mpirun.real" ]]; then
  FLAGCX_CI_MPI_HOME=$(mktemp -d)
  mkdir -p "$FLAGCX_CI_MPI_HOME/bin"
  ln -s "$FLAGCX_CI_MPI_BASE_HOME/bin/mpirun.real" \
    "$FLAGCX_CI_MPI_HOME/bin/mpirun"
  ln -s "$FLAGCX_CI_MPI_BASE_HOME/include" "$FLAGCX_CI_MPI_HOME/include"
  ln -s "$FLAGCX_CI_MPI_BASE_HOME/lib" "$FLAGCX_CI_MPI_HOME/lib"
  export MPI_HOME=$FLAGCX_CI_MPI_HOME
else
  export MPI_HOME=$FLAGCX_CI_MPI_BASE_HOME
fi

FLAGCX_CI_PROJECT_MAKE_ARGS=(USE_NVIDIA=1)
FLAGCX_CI_TEST_MAKE_ARGS=(USE_NVIDIA=1)
FLAGCX_CI_INTRA_NP=8
FLAGCX_CI_NODE_NP=4
FLAGCX_CI_RUNNER_NP=8
export NP=8

# Two logical four-GPU nodes on the eight-GPU CUDA runner.
FLAGCX_CI_NODE1_MPI_ARGS=(
  -x CUDA_VISIBLE_DEVICES=0,1,2,3
  -x FLAGCX_HOSTID=node0
  -x NCCL_HOSTID=node0
  -x FLAGCX_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
  -x NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
)
FLAGCX_CI_NODE2_MPI_ARGS=(
  -x CUDA_VISIBLE_DEVICES=4,5,6,7
  -x FLAGCX_HOSTID=node1
  -x NCCL_HOSTID=node1
  -x FLAGCX_IB_HCA=mlx5_4,mlx5_5,mlx5_6,mlx5_7
  -x NCCL_IB_HCA=mlx5_4,mlx5_5,mlx5_6,mlx5_7
)

flagcx_ci_configure_suite() {
  local suite=$1
  case "$suite" in
    runner)
      FLAGCX_CI_PROJECT_MAKE_ARGS+=(COMPILE_KERNEL=1)
      ;;
    device_api|device_api_unified_ir)
      FLAGCX_CI_PROJECT_MAKE_ARGS+=(COMPILE_KERNEL=1 FORCE_DEFAULT_PATH=1)
      FLAGCX_CI_TEST_MAKE_ARGS+=(FORCE_DEFAULT_PATH=1)
      ;;
  esac
}

flagcx_ci_prepare() {
  local suite=$1
  echo "Preparing CUDA environment for unit-test suite: $suite"
  command -v nvcc
  command -v mpirun
  mpirun --version
  nvidia-smi
}
