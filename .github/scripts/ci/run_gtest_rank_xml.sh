#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <result-dir> <test-binary> [args...]" >&2
  exit 2
fi

result_dir=$1
shift
rank=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-}}
if [[ -z "$rank" || ! "$rank" =~ ^[0-9]+$ ]]; then
  echo "Cannot identify MPI rank for GoogleTest XML output" >&2
  exit 1
fi
export GTEST_OUTPUT="xml:$result_dir/rank-$rank.xml"
exec "$@"
