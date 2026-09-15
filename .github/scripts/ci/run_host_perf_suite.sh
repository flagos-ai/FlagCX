#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <homogeneous|heterogeneous> <perf-bin-dir> [extra MPI arguments...]" >&2
  exit 2
fi

mode=$1
perf_bin=$2
shift 2

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mpi_runner="$script_dir/run_mpi_with_timeout.sh"
platform=${FLAGCX_CI_PERF_PLATFORM:-platform}
mpi_np=${FLAGCX_CI_PERF_NP:-8}

case "$mode" in
  homogeneous)
    operations=(
      alltoall alltoallv sendrecv allreduce allgather reducescatter
      broadcast gather scatter reduce
    )
    mode_mpi_args=()
    ;;
  heterogeneous)
    # uniRunner does not implement reduction collectives. Keep this list in
    # one place so every hardware-specific perf job exercises the same API set.
    operations=(alltoall alltoallv sendrecv allgather broadcast gather scatter)
    mode_mpi_args=(
      -x FLAGCX_MEM_ENABLE=1
      -x FLAGCX_USE_HETERO_COMM=1
    )
    ;;
  *)
    echo "Unknown perf mode: $mode" >&2
    exit 2
    ;;
esac

# Platform setup scripts may export mode selectors globally. Clear them before
# every invocation, then pass back only the selectors required by this mode.
clean_mode_env=(
  env
  -u FLAGCX_USE_HOST_COMM
  -u FLAGCX_USE_HETERO_COMM
  -u FLAGCX_CLUSTER_SPLIT_LIST
  -u FLAGCX_MEM_ENABLE
  -u FLAGCX_VMM_ENABLE
  -u FLAGCX_P2P_TRANSPORT
  -u FLAGCX_P2P_DISABLE
)
mpi_args=(-np "$mpi_np" --allow-run-as-root)
if [[ "$mode" == heterogeneous ]]; then
  mpi_args+=("${mode_mpi_args[@]}")
fi
mpi_args+=("$@")

for operation in "${operations[@]}"; do
  operation_args=(-b 128M -e 1G -f 2 -p 1)
  case "$operation" in
    broadcast|gather|scatter|reduce)
      operation_args+=(-r 0)
      ;;
  esac

  binary="$perf_bin/perf_$operation"
  if [[ ! -x "$binary" ]]; then
    echo "Perf binary is missing or not executable: $binary" >&2
    exit 1
  fi

  FLAGCX_CI_MPI_LABEL="$platform $mode perf: $operation" \
    "${clean_mode_env[@]}" "$mpi_runner" \
    "${mpi_args[@]}" "$binary" "${operation_args[@]}"
done
