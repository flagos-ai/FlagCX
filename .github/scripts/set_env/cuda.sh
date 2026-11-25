#!/usr/bin/env bash

# Everything in this file is CUDA/NVIDIA specific. The common workflow and
# runner deliberately do not inspect the platform name.

FLAGCX_CI_ENV_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
source "$FLAGCX_CI_ENV_DIR/../ci/rdma_static_preflight.sh"

export FLAGCX_ADAPTOR=nvidia

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

# Make the selected launcher effective in the current shell as well as in
# subsequent workflow steps. The CUDA image's default mpirun wrapper forwards
# "$@" literally, so flagcx_ci_prepare() must not resolve that stale wrapper
# before the workflow writes the updated PATH to GITHUB_ENV.
export PATH="$MPI_HOME/bin:$PATH"
hash -r

FLAGCX_CI_PROJECT_MAKE_ARGS=(USE_NVIDIA=1)
FLAGCX_CI_TEST_MAKE_ARGS=(USE_NVIDIA=1)
FLAGCX_CI_UNIFIED_IR_MPI_ARGS=(
  -x FLAGCX_IB_GID_INDEX=3
  -x NCCL_DEBUG=INFO
  -x NCCL_DEBUG_SUBSYS=INIT
  -x NCCL_NVLS_ENABLE=0
  -x NCCL_IB_GID_INDEX=3
)
FLAGCX_CI_ENABLE_IBUC=1
FLAGCX_CI_ENABLE_SHARED_P2P_ENGINE=1
FLAGCX_CI_INTRA_NP=4
FLAGCX_CI_NODE_NP=2
FLAGCX_CI_RUNNER_NP=4
export NP=4
# Keep one GPU near and one away from each logical node's NIC for PXN.
FLAGCX_CI_PXN_VISIBLE_DEVICES=0,3,4,7

# Two logical two-GPU nodes on the CUDA runner.
FLAGCX_CI_NODE1_MPI_ARGS=(
  -x CUDA_VISIBLE_DEVICES=0,1
  -x FLAGCX_HOSTID=node0
  -x NCCL_HOSTID=node0
  -x FLAGCX_IB_HCA=mlx5
  -x NCCL_IB_HCA=mlx5
)
FLAGCX_CI_NODE2_MPI_ARGS=(
  -x CUDA_VISIBLE_DEVICES=4,5
  -x FLAGCX_HOSTID=node1
  -x NCCL_HOSTID=node1
  -x FLAGCX_IB_HCA=mlx5
  -x NCCL_IB_HCA=mlx5
)

flagcx_ci_configure_suite() {
  local suite=$1
  case "$suite" in
    adaptor)
      # Keep the allocation-range unit test on the cudaMalloc-backed GDR path,
      # matching the adaptor IPC MPI invocation. Symmem owns the explicit VMM
      # matrix, whose POSIX-FD allocations do not require CUDA IMEX/FABRIC.
      export FLAGCX_VMM_ENABLE=0
      ;;
    runner)
      FLAGCX_CI_PROJECT_MAKE_ARGS+=(COMPILE_KERNEL=1)
      ;;
    device_api|device_api_host|device_api_unified_ir|device_traits)
      FLAGCX_CI_PROJECT_MAKE_ARGS+=(COMPILE_KERNEL=1 FORCE_DEFAULT_PATH=1)
      FLAGCX_CI_TEST_MAKE_ARGS+=(COMPILE_KERNEL=1 FORCE_DEFAULT_PATH=1)
      ;;
    device_api_gda)
      export GDA_PROVIDER_HOME=${GDA_PROVIDER_HOME:-/usr/local}
      FLAGCX_CI_PROJECT_MAKE_ARGS+=(COMPILE_KERNEL=1 USE_XPUNETIO=1
        "GDA_PROVIDER_HOME=$GDA_PROVIDER_HOME")
      FLAGCX_CI_TEST_MAKE_ARGS+=(COMPILE_KERNEL=1 USE_XPUNETIO=1
        "GDA_PROVIDER_HOME=$GDA_PROVIDER_HOME")
      export LD_LIBRARY_PATH="$GDA_PROVIDER_HOME/lib:$GDA_PROVIDER_HOME/lib64:${LD_LIBRARY_PATH:-}"
      ;;
  esac
}

flagcx_ci_prepare() {
  local suite=$1
  echo "Preparing CUDA environment for CI workload: $suite"
  command -v nvcc
  command -v mpirun
  mpirun --version
  nvidia-smi
  if [[ "$suite" == device_api_gda ]]; then
    if [[ ! -f "$GDA_PROVIDER_HOME/include/xpu_netio/device_api.cuh" ||
          ( ! -f "$GDA_PROVIDER_HOME/lib/libxpu_netio.so" &&
            ! -f "$GDA_PROVIDER_HOME/lib64/libxpu_netio.so" ) ]]; then
      echo "device_api_gda requires an installed XPU NetIO SDK in GDA_PROVIDER_HOME=$GDA_PROVIDER_HOME (headers and libxpu_netio.so)." >&2
      return 1
    fi
  fi
}

flagcx_ci_validate_rdma() {
  local suite=$1
  flagcx_ci_validate_rdma_static CUDA "$suite" \
    "/sys/class/infiniband/mlx5_*"
}
