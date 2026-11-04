#!/usr/bin/env bash
set -euo pipefail

project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
: "${DEVICE_HOME:?Ascend toolkit path is required}"
probe_bin="$project_root/test/perf/host_api/build/bin/ascend_rdma_probe"
mkdir -p "$(dirname "$probe_bin")"
g++ -std=c++11 -O2 -Wall -Wextra \
  -I "$DEVICE_HOME/include" \
  "$project_root/test/script/ascend_rdma_probe.cc" \
  -L "$DEVICE_HOME/lib64" -lascendcl -libverbs -ldl \
  -o "$probe_bin"

result=0
probe_timeout=${FLAGCX_CI_RDMA_PROBE_TIMEOUT:-${FLAGCX_CI_TEST_TIMEOUT:-3m}}
for hca in mlx5_0 mlx5_1; do
  for alignment in 4096 2097152; do
    echo "Running Ascend RDMA probe: hca=$hca alignment=$alignment"
    FLAGCX_CI_IB_HCA="$hca" \
      FLAGCX_CI_RDMA_ALIGNMENT="$alignment" \
      FLAGCX_CI_TEST_TIMEOUT="$probe_timeout" \
      FLAGCX_CI_TEST_LABEL="Ascend_RDMA_${hca}_align_${alignment}" \
      "$project_root/.github/scripts/ci/run_with_timeout.sh" "$probe_bin" || result=1
  done
done
exit "$result"
