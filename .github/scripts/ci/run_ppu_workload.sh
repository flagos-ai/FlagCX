#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <torch-api|perf>" >&2
  exit 2
fi

workload=$1
project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
mpi_runner="$project_root/.github/scripts/ci/run_mpi_with_timeout.sh"
ppu_env="$project_root/.github/scripts/set_env/ppu.sh"

# shellcheck source=/dev/null
source "$ppu_env"
flagcx_ci_prepare "$workload"
flagcx_ci_validate_rdma "$workload"

export PATH="$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$project_root/build/lib:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}"
export FLAGCX_IB_DISABLE=0
export FLAGCX_DEBUG=${FLAGCX_DEBUG:-INFO}
export FLAGCX_DEBUG_SUBSYS=${FLAGCX_DEBUG_SUBSYS:-INIT,NET,P2P,PROXY}

build_flagcx() {
  make -C "$project_root" --jobs="$(nproc)" USE_PPU=1 USE_ACCL_BAREX=1
}

run_perf() {
  local mode=$1
  local operation=$2
  shift 2
  local binary="$project_root/test/perf/host_api/build/bin/perf_$operation"
  local -a clean_mode_env=(
    env
    -u FLAGCX_USE_HOST_COMM
    -u FLAGCX_USE_HETERO_COMM
    -u FLAGCX_CLUSTER_SPLIT_LIST
    -u FLAGCX_MEM_ENABLE
    -u FLAGCX_VMM_ENABLE
    -u FLAGCX_P2P_TRANSPORT
    -u FLAGCX_P2P_DISABLE
  )
  local -a mode_env=()

  if [[ "$mode" == heterogeneous ]]; then
    mode_env=(
      -x FLAGCX_USE_HETERO_COMM=1
      -x FLAGCX_MEM_ENABLE=1
      -x FLAGCX_VMM_ENABLE=0
      -x FLAGCX_P2P_TRANSPORT=accl
      -x FLAGCX_IB_DISABLE=0
    )
  fi

  FLAGCX_CI_MPI_LABEL="PPU $mode perf: $operation" \
    "${clean_mode_env[@]}" "$mpi_runner" -np 8 --allow-run-as-root \
    -x LD_LIBRARY_PATH -x LD_PRELOAD -x FLAGCX_DEBUG \
    -x FLAGCX_DEBUG_SUBSYS "${mode_env[@]}" "$binary" "$@"
}

run_perf_suite() {
  local mode=$1
  local begin=$2
  local end=$3
  local -a common_args=(-b "$begin" -e "$end" -f 2 -p 1)
  local -a operations
  local operation

  case "$mode" in
    homogeneous)
      operations=(
        alltoall alltoallv sendrecv allreduce allgather reducescatter
        broadcast gather scatter reduce
      )
      ;;
    heterogeneous)
      # Match CUDA uniRunner coverage. Reduction collectives are not supported
      # by uniRunner and must be exercised through homoRunner or hybridRunner.
      operations=(alltoall alltoallv sendrecv allgather broadcast gather scatter)
      ;;
    *)
      echo "Unknown PPU perf mode: $mode" >&2
      return 2
      ;;
  esac

  for operation in "${operations[@]}"; do
    local -a operation_args=("${common_args[@]}")
    case "$operation" in
      broadcast|gather|scatter|reduce)
        operation_args+=(-r 0)
        ;;
    esac
    run_perf "$mode" "$operation" "${operation_args[@]}"
  done
}

case "$workload" in
  torch-api)
    command -v python3
    python3 -c 'import torch; print("torch", torch.__version__, "devices", torch.cuda.device_count()); assert torch.cuda.device_count() >= 8'
    build_flagcx
    (
      # The build script is the repo root's; plugin/torch holds only the native
      # backend sources now, and the Python package lives in src/.
      cd "$project_root"
      export TORCH_DEVICE_BACKEND_AUTOLOAD=0
      export FLAGCX_ADAPTOR=ppu
      export USE_PPU=1
      python3 setup.py build_ext --inplace
    )

    export PYTHON_BIN=python3
    # The package moved out of plugin/torch, so the tests import it from src/ the
    # same way the other backends' jobs do.
    export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
    export FLAGCX_ADAPTOR=ppu
    unset FLAGCX_USE_HETERO_COMM
    export FLAGCX_CLUSTER_SPLIT_LIST=2
    export FLAGCX_MEM_ENABLE=1
    export FLAGCX_VMM_ENABLE=0
    export FLAGCX_P2P_TRANSPORT=accl
    bash "$project_root/test/script/torch_api_test.sh"
    ;;
  perf)
    build_flagcx
    make -C "$project_root/test/perf" --jobs="$(nproc)" \
      USE_PPU=1 USE_ACCL_BAREX=1

    # Homogeneous coverage aligned with the CUDA/Hygon/MetaX platform jobs.
    run_perf_suite homogeneous 128M 1G

    run_perf_suite heterogeneous 128M 1G
    ;;
  *)
    echo "Unknown PPU workload: $workload" >&2
    exit 2
    ;;
esac
