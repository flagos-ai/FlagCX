#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 || "$1" != perf ]]; then
  echo "Usage: $0 perf" >&2
  exit 2
fi

workload=$1
project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
perf_runner="$project_root/.github/scripts/ci/run_host_perf_suite.sh"
: "${FLAGCX_CI_SET_ENV:?The platform configuration must provide set_env}"

# shellcheck source=/dev/null
source "$project_root/$FLAGCX_CI_SET_ENV"
flagcx_ci_prepare "$workload"
flagcx_ci_validate_rdma "$workload"

export PATH="$MPI_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$project_root/build/lib:$MPI_HOME/lib:${LD_LIBRARY_PATH:-}"
export FLAGCX_DEBUG=${FLAGCX_DEBUG:-INFO}
export FLAGCX_DEBUG_SUBSYS=${FLAGCX_DEBUG_SUBSYS:-INIT,NET,P2P,PROXY}

make -C "$project_root" --jobs="$(nproc)" \
  "${FLAGCX_CI_PROJECT_MAKE_ARGS[@]}"
make -C "$project_root/test/perf" --jobs="$(nproc)" \
  "${FLAGCX_CI_TEST_MAKE_ARGS[@]}"

export FLAGCX_CI_PERF_PLATFORM=Mthreads
perf_bin="$project_root/test/perf/host_api/build/bin"
"$perf_runner" homogeneous "$perf_bin"
"$perf_runner" heterogeneous "$perf_bin"
