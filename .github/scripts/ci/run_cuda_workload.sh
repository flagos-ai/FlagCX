#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <torch-api|perf|perf-experimental>" >&2
  exit 2
fi

workload=$1
project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
: "${FLAGCX_CI_SET_ENV:?The platform configuration must provide set_env}"
source "$project_root/$FLAGCX_CI_SET_ENV"

if [[ "$workload" == perf-experimental ]]; then
  # The experimental perf job keeps its own test selection and shares the CUDA
  # runner and image with the regular perf job.
  source "$project_root/.github/scripts/ci/run_cuda_experimental_perf.sh"
  exit 0
fi

flagcx_ci_configure_suite "$workload"
flagcx_ci_prepare "$workload"
flagcx_ci_validate_rdma "$workload"

export PATH="$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$project_root/build/lib:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}"

case "$workload" in
  perf)
    make -C "$project_root" --jobs="$(nproc)" \
      USE_NVIDIA=1 COMPILE_KERNEL=1 FORCE_DEFAULT_PATH=1
    make -C "$project_root/test/perf" --jobs="$(nproc)" \
      USE_NVIDIA=1 COMPILE_KERNEL=1 FORCE_DEFAULT_PATH=1

    export FLAGCX_CI_PERF_PLATFORM=CUDA
    perf_bin="$project_root/test/perf/host_api/build/bin"
    "$project_root/.github/scripts/ci/run_host_perf_suite.sh" homogeneous "$perf_bin"
    "$project_root/.github/scripts/ci/run_host_perf_suite.sh" heterogeneous "$perf_bin"
    ;;
  torch-api)
    export FLAGCX_ADAPTOR=nvidia
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export FLAGCX_DEBUG=INFO
    export FLAGCX_DEBUG_SUBSYS=ALL

    make -C "$project_root" --jobs="$(nproc)" USE_NVIDIA=1
    export PATH="/root/miniconda3/bin:$PATH"
    eval "$(/root/miniconda3/bin/conda shell.bash hook)"
    conda activate flagscale-train
    export PYTHON_BIN=/root/miniconda3/envs/flagscale-train/bin/python
    "$PYTHON_BIN" -m pip install 'setuptools_scm>=8'
    (
      cd "$project_root"
      export TORCH_DEVICE_BACKEND_AUTOLOAD=0
      "$PYTHON_BIN" setup.py build_ext --inplace
    )

    export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
    bash "$project_root/test/script/torch_api_test.sh"
    if [[ "${FLAGCX_CI_TORCH_COVERAGE:-0}" == 1 ]]; then
      "$PYTHON_BIN" -m pip install --quiet coverage==7.10.7
      bash "$project_root/.github/scripts/ci/run_torch_coverage.sh"
    fi
    ;;
  *)
    echo "Unknown CUDA workload: $workload" >&2
    exit 2
    ;;
esac
