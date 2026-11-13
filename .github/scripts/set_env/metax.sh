#!/usr/bin/env bash

# MetaX-specific unit-test environment setup.

FLAGCX_CI_ENV_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=/dev/null
source "$FLAGCX_CI_ENV_DIR/../ci/rdma_static_preflight.sh"

export FLAGCX_ADAPTOR=metax

FLAGCX_CI_MPI_BASE_HOME=${MPI_HOME:-/usr/local/mpi}

# Use the real OpenMPI launcher if the image provides a wrapper.
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

export PATH="/opt/maca/mxgpu_llvm/bin:$PATH"
export LD_LIBRARY_PATH="/opt/mxdriver/lib:/opt/maca/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"

# The MetaX RoCEv2 GID is index 3 on the CI Broadcom adapters. MPI is used
# only for process coordination by these single-node perf jobs; keep its
# bootstrap on the container-visible TCP interface instead of UCX's default
# RoCE GID 0, which is not routable from the job Pod network namespace.
export FLAGCX_IB_GID_INDEX=${FLAGCX_IB_GID_INDEX:-3}
export OMPI_MCA_pml=${OMPI_MCA_pml:-ob1}
export OMPI_MCA_btl=${OMPI_MCA_btl:-self,tcp}

FLAGCX_CI_PROJECT_MAKE_ARGS=(USE_METAX=1)
FLAGCX_CI_TEST_MAKE_ARGS=(USE_METAX=1)
FLAGCX_CI_ENABLE_SHARED_P2P_ENGINE=1
FLAGCX_CI_INTRA_NP=4
FLAGCX_CI_NODE_NP=2
FLAGCX_CI_RUNNER_NP=4
export NP=4
FLAGCX_CI_PXN_VISIBLE_DEVICES=0,2,4,7

FLAGCX_CI_NODE1_MPI_ARGS=(
  -x FLAGCX_HOSTID=node0
  -x NCCL_HOSTID=node0
)
FLAGCX_CI_NODE2_MPI_ARGS=(
  -x FLAGCX_HOSTID=node1
  -x NCCL_HOSTID=node1
)

flagcx_ci_configure_suite() {
  local suite=$1

  case "$suite" in
    rma)
      export RMA_PLATFORM_ENV="-x FLAGCX_USE_TUNER=1 -x TUNNING_WITH_SINGLE_COMM=1 -x FLAGCX_USE_HOST_COMM=1"
      FLAGCX_CI_TEST_MAKE_ARGS+=(
        "RMA_PLATFORM_ENV=$RMA_PLATFORM_ENV"
      )
      ;;
  esac
}

flagcx_ci_prepare() {
  local suite=$1
  echo "Preparing MetaX environment for CI workload: $suite"
  command -v mpirun
  command -v mxcc

  if [[ "$suite" == "adaptor" || "$suite" == "p2p" ||
        "$suite" == "rma" || "$suite" == "runner" ||
        "$suite" == "symmem" || "$suite" == "perf" ||
        "$suite" == "torch-api" ]]; then
    if [[ -d /sys/class/net/bond0 ]]; then
      export FLAGCX_SOCKET_IFNAME=${FLAGCX_SOCKET_IFNAME:-bond0}
    fi

    export FLAGCX_DEBUG=${FLAGCX_DEBUG:-INFO}
    export FLAGCX_DEBUG_SUBSYS=${FLAGCX_DEBUG_SUBSYS:-INIT,NET,P2P,ENV}

    echo "MetaX network diagnostics:"
    echo "FLAGCX_IB_HCA=${FLAGCX_IB_HCA:-<unset>}"
    echo "FLAGCX_IB_GID_INDEX=${FLAGCX_IB_GID_INDEX:-<unset>}"
    echo "FLAGCX_SOCKET_IFNAME=${FLAGCX_SOCKET_IFNAME:-<unset>}"
    echo "Network interfaces visible inside the CI container:"
    ls /sys/class/net 2>/dev/null || true
    echo "RDMA devices visible inside the CI container:"
    ls /sys/class/infiniband 2>/dev/null || true
    ls /dev/infiniband 2>/dev/null || true
    ip -o addr show 2>/dev/null || true
  fi
}

flagcx_ci_select_rdma() {
  local suite=$1
  local rdma_sysfs_root=${FLAGCX_CI_METAX_RDMA_SYSFS_ROOT:-/sys/class/infiniband}
  local -a hca_paths=()
  local -a hca_names=()
  local -a pxn_hca_names=()
  local -a port_paths=()
  local hca_path

  shopt -s nullglob
  # HCA names are driver- and platform-specific (for example bnxt_roce and
  # mlx5). Discover every HCA that exposes at least one port instead of
  # assuming a vendor name.
  for hca_path in "$rdma_sysfs_root"/*; do
    [[ -d "$hca_path" ]] || continue
    port_paths=("$hca_path"/ports/*)
    [[ ${#port_paths[@]} -gt 0 ]] || continue
    hca_paths+=("$hca_path")
  done
  shopt -u nullglob

  if [[ ${#hca_paths[@]} -eq 0 ]]; then
    echo "MetaX $suite tests require an RDMA HCA with at least one port under $rdma_sysfs_root." >&2
    return 1
  fi

  for hca_path in "${hca_paths[@]}"; do
    hca_names+=("${hca_path##*/}")
  done
  local IFS=,
  if [[ -z "${FLAGCX_IB_HCA:-}" ]]; then
    export FLAGCX_IB_HCA="${hca_names[*]}"
  fi
  # Keep an explicit PXN override working, while deriving the default from
  # the same discovered HCA set used by FLAGCX_IB_HCA.
  if [[ -z "${FLAGCX_CI_PXN_HCAS:-}" ]]; then
    # The first HCA is directly adjacent to GPU 0 on these runners. Keep it
    # out of PXN so the configured GPU topology can relay through the others.
    if [[ ${#hca_names[@]} -gt 1 ]]; then
      pxn_hca_names=("${hca_names[@]:1}")
    else
      pxn_hca_names=("${hca_names[@]}")
    fi
    FLAGCX_CI_PXN_HCAS="${pxn_hca_names[*]}"
  fi
  export FLAGCX_CI_METAX_RDMA_PATTERN="$rdma_sysfs_root/*"
}

flagcx_ci_validate_rdma() {
  local suite=$1
  flagcx_ci_select_rdma "$suite" || return
  flagcx_ci_validate_rdma_static MetaX "$suite" \
    "$FLAGCX_CI_METAX_RDMA_PATTERN" || return
  echo "MetaX RDMA selection: FLAGCX_IB_HCA=$FLAGCX_IB_HCA"
}

flagcx_ci_build_suite_override() {
  local suite=$1
  local suite_dir=$2
  shift 2
  local -a args=("$@")

  if [[ "$suite" == "symmem" ]]; then
    FLAGCX_CI_BUILD_SUITE_OVERRIDE_HANDLED=1
    cmake -S "$PROJECT_ROOT/third-party/googletest" \
      -B "$PROJECT_ROOT/third-party/googletest/build"
    cmake --build "$PROJECT_ROOT/third-party/googletest/build" --parallel "$(nproc)"
    make -C "$suite_dir" --jobs="$(nproc)" "${args[@]}"
    return
  fi

  FLAGCX_CI_BUILD_SUITE_OVERRIDE_HANDLED=0
}
