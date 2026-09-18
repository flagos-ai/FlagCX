#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <torch-api|perf>" >&2
  exit 2
fi

workload=$1
project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
perf_runner="$project_root/.github/scripts/ci/run_host_perf_suite.sh"
ppu_env="$project_root/.github/scripts/set_env/ppu.sh"

# shellcheck source=/dev/null
source "$ppu_env"
flagcx_ci_configure_suite "$workload"
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

case "$workload" in
  torch-api)
    command -v python3
    python3 -c 'import torch; print("torch", torch.__version__, "devices", torch.cuda.device_count()); assert torch.cuda.device_count() >= 8'
    build_flagcx
    (
      cd "$project_root/plugin/torch"
      export TORCH_DEVICE_BACKEND_AUTOLOAD=0
      export FLAGCX_ADAPTOR=ppu
      export USE_PPU=1
      python3 setup.py build_ext --inplace
    )

    export PYTHON_BIN=python3
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

    export FLAGCX_CI_PERF_PLATFORM=PPU
    perf_bin="$project_root/test/perf/host_api/build/bin"
    "$perf_runner" homogeneous "$perf_bin" \
      -x LD_LIBRARY_PATH -x LD_PRELOAD -x FLAGCX_DEBUG \
      -x FLAGCX_DEBUG_SUBSYS
    "$perf_runner" heterogeneous "$perf_bin" \
      -x LD_LIBRARY_PATH -x LD_PRELOAD -x FLAGCX_DEBUG \
      -x FLAGCX_DEBUG_SUBSYS -x FLAGCX_VMM_ENABLE=0 \
      -x FLAGCX_P2P_TRANSPORT=accl -x FLAGCX_IB_DISABLE=0
    ;;
  *)
    echo "Unknown PPU workload: $workload" >&2
    exit 2
    ;;
esac
