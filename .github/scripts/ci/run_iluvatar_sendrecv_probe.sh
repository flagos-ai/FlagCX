#!/usr/bin/env bash
# Collect a controlled SendRecv matrix without changing the functional gate.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 <probe-binary> <nranks> [pxn-ranks-per-node] [pxn-visible-devices]" >&2
  exit 2
fi

binary=$1
nranks=$2
pxn_ranks_per_node=${3:-2}
pxn_visible_devices=${4:-}
if [[ ! -x "$binary" || "$nranks" != 4 ]]; then
  echo "Iluvatar SendRecv probe requires an executable binary and four ranks" >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
runner="$script_dir/run_iluvatar_runner_mpi.sh"
log_dir=${FLAGCX_CI_LOG_DIR:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}/flagcx-ci-attempts}
mkdir -p "$log_dir"
summary="$log_dir/sendrecv-probe-summary.tsv"
printf 'transport\tpattern\tcopy_order\texit_status\n' > "$summary"
infrastructure_failures=0

run_case() {
  local transport=$1 pattern=$2 order=$3
  local status=0
  local -a transport_env=(
    -x FLAGCX_MEM_ENABLE=1
    -x FLAGCX_CLUSTER_SPLIT_LIST=2
    -x FLAGCX_VMM_ENABLE=0
    -x FLAGCX_CI_RUNNER_BYTES=4194304
    -x FLAGCX_CI_EXPECT_RUNNER_MODE=HYBRID
    -x FLAGCX_CI_SENDRECV_PROBE_PATTERN="$pattern"
    -x FLAGCX_CI_SENDRECV_PROBE_COPY_ORDER="$order"
  )
  local -a launcher=("$runner")
  if [[ "$transport" == P2P ]]; then
    transport_env+=(
      -x FLAGCX_P2P_DISABLE=0
      -x FLAGCX_PXN_DISABLE=1
      -x FLAGCX_CI_EXPECT_PEER_TRANSPORT=P2P
    )
  elif [[ "$transport" == NET || "$transport" == PXN ]]; then
    transport_env+=(
      -x FLAGCX_P2P_DISABLE=1
      -x FLAGCX_CI_EXPECT_PEER_TRANSPORT=NET
      -x FLAGCX_CI_EXPECT_NET_ADAPTOR=IB
      -x FLAGCX_IB_QPS_PER_CONNECTION=2
      -x FLAGCX_IB_SPLIT_DATA_ON_QPS=0
      -x FLAGCX_IBUC_SPLIT_DATA_ON_QPS=0
    )
    if [[ "$transport" == NET ]]; then
      transport_env+=(-x FLAGCX_PXN_DISABLE=1)
    fi
  else
    echo "Unknown SendRecv probe transport: $transport" >&2
    return 2
  fi
  if [[ "$transport" == PXN ]]; then
    transport_env+=(
      -x FLAGCX_PXN_DISABLE=0
      -x FLAGCX_CI_EXPECT_PXN=1
      -x FLAGCX_CI_PXN_RANKS_PER_NODE="$pxn_ranks_per_node"
    )
    if [[ -n "$pxn_visible_devices" ]]; then
      launcher=(env CUDA_VISIBLE_DEVICES="$pxn_visible_devices" "$runner")
      transport_env+=(-x CUDA_VISIBLE_DEVICES="$pxn_visible_devices")
    fi
  fi

  echo "SendRecv probe transport=$transport pattern=$pattern copy_order=$order"
  FLAGCX_CI_MPI_LABEL="SendRecv probe $transport $pattern $order" \
    FLAGCX_CI_RUNNER_SENDRECV_TIMEOUT=40s \
    FLAGCX_CI_RUNNER_DIAGNOSTICS_AFTER=12 \
    FLAGCX_CI_RUNNER_STAGE_BUDGET=60 \
    "${launcher[@]}" -np "$nranks" --allow-run-as-root \
    "${transport_env[@]}" "$binary" \
    --gtest_filter=FlagCXCollTest.SendRecvProbe || status=$?
  printf '%s\t%s\t%s\t%d\n' "$transport" "$pattern" "$order" "$status" \
    >> "$summary"
  if [[ "$status" == 2 || "$status" == 126 || "$status" == 127 ]]; then
    infrastructure_failures=$((infrastructure_failures + 1))
  fi
}

for transport in P2P NET; do
  for pattern in F R D S RING; do
    for order in EARLY LATE; do
      run_case "$transport" "$pattern" "$order"
    done
  done
done
if [[ "${FLAGCX_CI_ENABLE_PXN:-1}" == 1 ]]; then
  for order in EARLY LATE; do
    run_case PXN PXN_D "$order"
  done
fi

cat "$summary"
# These are diagnostic observations. The original SendRecv cases remain the
# required CI gate, so an expected probe timeout must not hide later cases.
if ((infrastructure_failures != 0)); then
  echo "SendRecv probe encountered $infrastructure_failures infrastructure failures" >&2
  exit 1
fi
