#!/usr/bin/env bash
# Bounded, failure-only reproduction. Its status never replaces the suite result.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}

if [[ "${1:-}" == --rank ]]; then
  shift
  : "${FLAGCX_CI_DIAGNOSTIC_DIR:?Diagnostic directory is required}"
  rank=${OMPI_COMM_WORLD_RANK:-unit}
  # Do not overwrite XML from the real test invocation during reproduction.
  unset GTEST_OUTPUT
  trace=$(mktemp "$FLAGCX_CI_DIAGNOSTIC_DIR/rank-$rank.XXXXXX.log")
  exec gdb --batch --return-child-result \
    -ex 'set pagination off' -ex 'set print frame-arguments all' \
    -ex run -ex 'thread apply all bt full' -ex 'info registers' \
    --args "$@" > "$trace" 2>&1
fi

case "${1:-}" in
  p2p|rma|symmem) suite=$1 ;;
  *) exit 0 ;;
esac
log_root=${FLAGCX_CI_LOG_DIR:-${RUNNER_TEMP:-/tmp}/flagcx-ci-attempts}
crashed=0
while IFS= read -r -d '' log; do
  if grep -Eq 'SIGSEGV|Segmentation fault|signal 11' "$log"; then
    crashed=1
    break
  fi
done < <(find "$log_root" -type f -name 'attempt-*.log' -print0)
if ((crashed == 0)); then
  echo "No SIGSEGV recorded; $suite failed without a reproduced crash. See attempt logs."
  exit 0
fi
command -v gdb >/dev/null || { echo 'Ascend diagnostics require gdb' >&2; exit 1; }
export FLAGCX_CI_DIAGNOSTIC_DIR
FLAGCX_CI_DIAGNOSTIC_DIR=$(mktemp -d "$log_root/ascend-$suite-backtrace.XXXXXX")
export FLAGCX_CI_TEST_TIMEOUT=2m FLAGCX_CI_MPI_TIMEOUT=2m
export FLAGCX_CI_TEST_KILL_AFTER=10s FLAGCX_CI_MPI_KILL_AFTER=10s
export FLAGCX_USE_HETERO_COMM=1 FLAGCX_MEM_ENABLE=1 FLAGCX_VMM_ENABLE=0

binary="$project_root/test/unittest/$suite/build/bin/${suite}_mpi_tests"
status=0
if [[ "$suite" == p2p ]]; then
  FLAGCX_CI_TEST_LABEL="Ascend p2p unit backtrace" \
    "$script_dir/run_with_timeout.sh" bash "$0" --rank \
    "$project_root/test/unittest/p2p/build/bin/p2p_unit_tests" || status=$?
  filter='*'
elif [[ "$suite" == rma ]]; then
  filter='RmaTest.GetSmall'
else
  filter='SymMemTest.RegisterSymmetricWindow'
fi
FLAGCX_CI_MPI_LABEL="Ascend $suite MPI backtrace" \
  "$script_dir/run_mpi_with_timeout.sh" --allow-run-as-root -np 2 \
    -x FLAGCX_CI_DIAGNOSTIC_DIR -x FLAGCX_USE_HETERO_COMM \
    -x FLAGCX_MEM_ENABLE -x FLAGCX_VMM_ENABLE \
    bash "$0" --rank "$binary" --gtest_filter="$filter" || status=$?
if [[ "$suite" == symmem ]]; then
  FLAGCX_CI_MPI_LABEL="Ascend symmem NET backtrace" \
    "$script_dir/run_mpi_with_timeout.sh" --allow-run-as-root \
      -np 2 -x FLAGCX_CI_DIAGNOSTIC_DIR -x FLAGCX_USE_HETERO_COMM \
      -x FLAGCX_MEM_ENABLE -x FLAGCX_VMM_ENABLE \
      -x FLAGCX_CLUSTER_SPLIT_LIST=2 -x FLAGCX_HOSTID=node0 \
      -x FLAGCX_IB_DISABLE=0 -x FLAGCX_P2P_DISABLE=1 \
      -x FLAGCX_CI_REQUIRE_NET_MR=1 -x FLAGCX_CI_EXPECT_NET_ADAPTOR=IB \
      bash "$0" --rank "$binary" \
      --gtest_filter=SymMemTest.FullMeshRoundFailureConvergesAndRetries \
      : -np 2 -x FLAGCX_CI_DIAGNOSTIC_DIR -x FLAGCX_USE_HETERO_COMM \
      -x FLAGCX_MEM_ENABLE -x FLAGCX_VMM_ENABLE \
      -x FLAGCX_CLUSTER_SPLIT_LIST=2 -x FLAGCX_HOSTID=node1 \
      -x FLAGCX_IB_DISABLE=0 -x FLAGCX_P2P_DISABLE=1 \
      -x FLAGCX_CI_REQUIRE_NET_MR=1 -x FLAGCX_CI_EXPECT_NET_ADAPTOR=IB \
      bash "$0" --rank "$binary" \
      --gtest_filter=SymMemTest.FullMeshRoundFailureConvergesAndRetries || status=$?
fi
echo "Ascend $suite diagnostic exit=$status; per-rank backtraces: $FLAGCX_CI_DIAGNOSTIC_DIR"
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  printf '\nAscend %s crash reproduction exit=%s; backtraces: `%s` (see attempts artifact).\n' \
    "$suite" "$status" "$FLAGCX_CI_DIAGNOSTIC_DIR" >> "$GITHUB_STEP_SUMMARY"
fi
exit "$status"
