#!/usr/bin/env bash

# Iluvatar CoreX-specific unit-test environment setup.

FLAGCX_CI_ENV_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
source "$FLAGCX_CI_ENV_DIR/../ci/rdma_static_preflight.sh"

export MPI_HOME=${MPI_HOME:-/usr/local/openmpi}

export PATH="$MPI_HOME/bin:/usr/local/corex/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/corex/lib64:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}"

FLAGCX_CI_PROJECT_MAKE_ARGS=(USE_ILUVATAR=1)
FLAGCX_CI_TEST_MAKE_ARGS=(USE_ILUVATAR=1)
FLAGCX_CI_ENABLE_SHARED_P2P_ENGINE=1
FLAGCX_CI_INTRA_NP=4
FLAGCX_CI_NODE_NP=2
FLAGCX_CI_RUNNER_NP=4
FLAGCX_CI_PXN_VISIBLE_DEVICES=0,3,4,7
export NP=4

FLAGCX_CI_NODE1_MPI_ARGS=(
  -x COREX_VISIBLE_DEVICES=0,1
  -x CUDA_VISIBLE_DEVICES=0,1
  -x FLAGCX_HOSTID=node0
  -x NCCL_HOSTID=node0
)
FLAGCX_CI_NODE2_MPI_ARGS=(
  -x COREX_VISIBLE_DEVICES=4,5
  -x CUDA_VISIBLE_DEVICES=4,5
  -x FLAGCX_HOSTID=node1
  -x NCCL_HOSTID=node1
)

flagcx_ci_configure_suite() {
  local suite=$1

  case "$suite" in
    adaptor)
      export FLAGCX_VMM_ENABLE=0
      export FLAGCX_CI_ADAPTOR_GPU_SKIP_CASES="GpuIput GpuIget GpuBatchPutAndGet GpuBounds"
      ;;
    runner)
      FLAGCX_CI_PROJECT_MAKE_ARGS+=(COMPILE_KERNEL=1)
      ;;
    device_api|device_api_host|device_api_unified_ir)
      FLAGCX_CI_PROJECT_MAKE_ARGS+=(COMPILE_KERNEL=1 FORCE_DEFAULT_PATH=1)
      FLAGCX_CI_TEST_MAKE_ARGS+=(COMPILE_KERNEL=1 FORCE_DEFAULT_PATH=1)
      ;;
  esac
}

flagcx_ci_prepare() {
  local suite=$1
  echo "Preparing Iluvatar CoreX environment for CI workload: $suite"
  command -v mpirun
  command -v clang
  command -v ixsmi
  mpirun --version
  ixsmi -L
}

flagcx_ci_validate_rdma() {
  local suite=$1
  flagcx_ci_validate_rdma_static Iluvatar "$suite" \
    "/sys/class/infiniband/mlx5_*"
}
