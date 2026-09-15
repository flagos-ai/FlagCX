#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <torch-api|perf>" >&2
  exit 2
fi

workload=$1
project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
perf_runner="$project_root/.github/scripts/ci/run_host_perf_suite.sh"
hygon_env="$project_root/.github/scripts/set_env/hygon.sh"

# shellcheck source=/dev/null
source "$hygon_env"
flagcx_ci_configure_suite "$workload"
flagcx_ci_prepare "$workload"
flagcx_ci_validate_rdma "$workload"

export PATH="$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$project_root/build/lib:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}"
export FLAGCX_DEBUG=${FLAGCX_DEBUG:-INFO}
export FLAGCX_DEBUG_SUBSYS=${FLAGCX_DEBUG_SUBSYS:-INIT,NET,P2P,PROXY}

python3 -c 'import torch; print("torch", torch.__version__, "devices", torch.cuda.device_count()); assert torch.cuda.device_count() >= 8'

build_flagcx() {
  make -C "$project_root" --jobs="$(nproc)" \
    "${FLAGCX_CI_PROJECT_MAKE_ARGS[@]}"
}

case "$workload" in
  torch-api)
    export FLAGCX_DEBUG=TRACE
    export FLAGCX_DEBUG_SUBSYS=ALL
    build_flagcx
    (
      cd "$project_root/plugin/torch"
      export TORCH_DEVICE_BACKEND_AUTOLOAD=0
      python3 setup.py build_ext --inplace
    )

    export PYTHONPATH="$project_root/plugin/torch${PYTHONPATH:+:$PYTHONPATH}"
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    unset FLAGCX_SKIP_HETERO
    bash "$project_root/test/script/torch_api_test.sh"
    ;;
  perf)
    build_flagcx
    make -C "$project_root/test/perf" --jobs="$(nproc)" \
      "${FLAGCX_CI_TEST_MAKE_ARGS[@]}"

    export FLAGCX_CI_PERF_PLATFORM=Hygon
    perf_bin="$project_root/test/perf/host_api/build/bin"
    "$perf_runner" homogeneous "$perf_bin"
    "$perf_runner" heterogeneous "$perf_bin"
    ;;
  *)
    echo "Unknown Hygon workload: $workload" >&2
    exit 2
    ;;
esac
