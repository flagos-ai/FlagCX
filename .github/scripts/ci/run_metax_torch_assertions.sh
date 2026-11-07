#!/usr/bin/env bash

set -euo pipefail

project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
python_bin=${PYTHON_BIN:-python3}
runner="$project_root/.github/scripts/ci/run_with_timeout.sh"
check_script="$project_root/.github/scripts/ci/metax_torch_assertions.py"
assert_log="${FLAGCX_CI_LOG_DIR:-/tmp}/metax-torch-assertions.log"
: > "$assert_log"

# Match the two environments used by test/script/torch_api_test.sh without
# changing that shared launcher or its example program.
mode_env=(
  env
  -u FLAGCX_USE_HOST_COMM
  -u FLAGCX_USE_HETERO_COMM
  -u FLAGCX_CLUSTER_SPLIT_LIST
  -u FLAGCX_MEM_ENABLE
  -u FLAGCX_VMM_ENABLE
  -u FLAGCX_P2P_TRANSPORT
  -u FLAGCX_P2P_DISABLE
)

for mode in homogeneous heterogeneous; do
  port=$("$python_bin" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')
  if [[ "$mode" == heterogeneous ]]; then
    mode_env+=(FLAGCX_CLUSTER_SPLIT_LIST=2 FLAGCX_MEM_ENABLE=1)
  fi
  FLAGCX_CI_TEST_LABEL="MetaX Torch data assertions ($mode)" \
    "$runner" "${mode_env[@]}" \
    "$python_bin" -m torch.distributed.run \
    --nproc_per_node=4 --nnodes=1 --node_rank=0 \
    --master_addr=localhost "--master_port=$port" \
    "$check_script" "$mode" 2>&1 | tee -a "$assert_log"
  echo "[INFO] Completed MetaX Torch data assertions in $mode mode"
done
