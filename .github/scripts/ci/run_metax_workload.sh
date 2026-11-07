#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <torch-api|perf>" >&2
  exit 2
fi

workload=$1
project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
: "${FLAGCX_CI_SET_ENV:?The platform configuration must provide set_env}"
source "$project_root/$FLAGCX_CI_SET_ENV"
flagcx_ci_configure_suite "$workload"
flagcx_ci_prepare "$workload"
flagcx_ci_validate_rdma "$workload"

export PATH="/opt/maca/mxgpu_llvm/bin:$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$project_root/build/lib:/opt/mxdriver/lib:/opt/maca/lib:/usr/local/lib:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}"
export FLAGCX_ADAPTOR=metax
export TORCH_DISTRIBUTED_BACKEND=mccl

case "$workload" in
  perf)
    python3 "$project_root/.github/scripts/ci/check_metax_devices.py" perf
    make -C "$project_root" --jobs="$(nproc)" USE_METAX=1
    make -C "$project_root/test/perf" --jobs="$(nproc)" \
      USE_METAX=1 MPI_HOME="$MPI_HOME" CCL_HOME=/opt/maca

    export FLAGCX_CI_PERF_PLATFORM=MetaX
    export FLAGCX_CI_PERF_VERIFY=1
    perf_bin="$project_root/test/perf/host_api/build/bin"
    MCCL_DEBUG=TRACE MCCL_DEBUG_SUBSYS=ALL \
      "$project_root/.github/scripts/ci/run_host_perf_suite.sh" homogeneous "$perf_bin"
    "$project_root/.github/scripts/ci/run_host_perf_suite.sh" heterogeneous "$perf_bin"
    ;;
  torch-api)
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export FLAGCX_DEBUG=INFO
    export FLAGCX_DEBUG_SUBSYS=ALL
    export USE_METAX=1
    export PYTHON_BIN=python3

    "$PYTHON_BIN" "$project_root/.github/scripts/ci/check_metax_devices.py" torch-api

    make -C "$project_root" --jobs="$(nproc)" USE_METAX=1
    "$PYTHON_BIN" -m pip install --break-system-packages 'setuptools_scm>=8'
    (
      cd "$project_root"
      export TORCH_DEVICE_BACKEND_AUTOLOAD=0
      "$PYTHON_BIN" setup.py build_ext --inplace
    )

    export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
    torch_log="${FLAGCX_CI_LOG_DIR:-/tmp}/metax-torch-api.log"
    bash "$project_root/test/script/torch_api_test.sh" 2>&1 | tee "$torch_log"
    bash "$project_root/.github/scripts/ci/run_metax_torch_assertions.sh"
    if [[ "${FLAGCX_CI_TORCH_COVERAGE:-0}" == 1 ]]; then
      "$PYTHON_BIN" -m pip install --break-system-packages --quiet coverage==7.10.7
      bash "$project_root/.github/scripts/ci/run_torch_coverage.sh"
    fi
    ;;
  *)
    echo "Unknown MetaX workload: $workload" >&2
    exit 2
    ;;
esac
