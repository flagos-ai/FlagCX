#!/usr/bin/env bash
# Harness-only mock of `mpirun`. NEVER used for real acceptance.
#
# Simulates the official test binaries' observable output so the C wrappers'
# exit-code, timeout, log-assertion and aggregation logic can be red-teamed
# without MUSA kernels or real MPI. Scenario selected by MOCK_SCENARIO:
#   pass | rank_fail | launcher_err | signal | timeout | partial_fail |
#   missing_case | no_marker | subset_ranks
# MOCK_FAIL_ROUNDS=2 makes run_repeat's round 2 behave like rank_fail.

set -uo pipefail

scenario="${MOCK_SCENARIO:-pass}"
n="${MOCK_NRANKS:-8}"

if [ -n "${MOCK_FAIL_ROUNDS:-}" ] && [ -n "${REPEAT_ROUND:-}" ]; then
  case ",${MOCK_FAIL_ROUNDS}," in
    *",${REPEAT_ROUND},"*) scenario=rank_fail ;;
  esac
fi

bin=""
for a in "$@"; do
  case "$a" in
    *test_device_*) bin="$a"; break ;;
  esac
done
base="$(basename "${bin:-unknown}")"

case "$scenario" in
  signal) kill -SEGV "$$" ;;             # launcher killed by a signal
  timeout) sleep 300 ;;                  # wrapper timeout must fire
  launcher_err) echo "mock mpirun: launch failed" >&2; exit 7 ;;
  no_marker) exit 0 ;;
esac

emit() { # emit <case-name> [FAIL]
  local name="$1" res="${2:-PASS}" r
  for ((r = 0; r < n; r++)); do
    printf '[rank %d] %-35s %s\n' "$r" "$name" "$res"
  done
}

api_intra_cases=(
  "K1 CommQueries" "K2 CoopGroups" "K3 Team" "K4 LocalPointer"
  "K5 IntraPointer" "K6 PeerPointer(team)" "K8 IntraBarrierSync"
  "K9 IntraBarrierArriveWait" "K10 IntraAllReduce(composite)"
)
api_inter_cases=(
  "K2 Reset" "K3 PutSigInc" "K4 PutSigAdd" "K4b MultiContextPutRelease"
  "K5 CounterPipeline" "K6 FlushDecouple" "K7 PutValue" "K8 Get (SKIP)"
  "K9 Signal" "K11 WaitSignal+Flush" "K12 InterBarrier" "K13 WorldBarrier"
  "K14 OneSidedAlltoAll"
)
ir_intra_cases=(
  "S1  CommQueries(Scalar)" "S2  CoopGroups(Scalar)" "S3  TeamQueries(Scalar)"
  "S4  LocalPointer(Scalar)" "S5  IntraPointer(Scalar)"
  "S6  PeerPointer(Scalar)" "S8  IntraBarrierSync(Scalar)"
  "S9  IntraBarrierTwoEpochs(Scalar)" "S10 IntraAllReduce(Scalar)"
)
unified_intra_cases=(
  "S16 DevBarrier(INTRA+WORLD)" "S17 DevTeamResolution(INTRA+WORLD)"
  "S18 DevPut+PutValue(INTRA+WORLD)" "S19 DevGet(INTRA+WORLD)"
  "S20 DevSignalStandalone(INTRA+WORLD)" "S21 DevPutSignalWait(INTRA+WORLD)"
  "S22 DevPut_RSig(INTRA+WORLD)" "S23 DevPutCounter(INTRA+WORLD)"
  "S24 DevPutValue_RSig(INTRA+WORLD)" "S25 DevSignalShadowFlush(INTRA+WORLD)"
)
unified_inter_cases=(
  "S16 DevBarrier(INTER+WORLD)"
  "S17 DevTeamResolution(INTER+WORLD+INTRA-locality)"
  "S18 DevPut+PutValue(INTER+WORLD)" "S19 DevGet(INTER+WORLD)"
  "S20 DevSignalStandalone(INTER+WORLD)" "S21 DevPutSignalWait(INTER+WORLD)"
  "S22 DevPut_RSig(INTER+WORLD)" "S23 DevPutCounter(INTER+WORLD)"
  "S24 DevPutValue_RSig(INTER+WORLD)" "S25 DevSignalShadowFlush(INTER+WORLD)"
)

# The transport lines below are emitted by FlagCX through INFO(), which is
# dropped unless FLAGCX_DEBUG is at least INFO (see flagcx/service/debug.cc).
# Mirror that here so the harness catches a wrapper that forgets to propagate
# FLAGCX_DEBUG to the ranks.
info_enabled() {
  case "${FLAGCX_DEBUG:-}" in
    INFO|ABORT|TRACE|info|abort|trace) return 0 ;;
  esac
  return 1
}

emit_unified_meta() {
  case "$base" in
    test_device_ir_unified_intra)
      echo "Nodes: 1 IntraSize: $n" ;;
    test_device_ir_unified_inter|test_device_api_inter|test_device_ir_inter)
      echo "Nodes: 2 IntraSize: 4"
      if info_enabled; then
        echo "NET/Socket : Using mock0"
        echo "NET/IB : libibverbs reported 1 device(s)"
      fi ;;
  esac
}

emit_all() {
  local c
  case "$base" in
    test_device_api_intra) for c in "${api_intra_cases[@]}"; do emit "$c"; done ;;
    test_device_api_inter) for c in "${api_inter_cases[@]}"; do emit "$c"; done ;;
    test_device_ir_intra) for c in "${ir_intra_cases[@]}"; do emit "$c"; done ;;
    test_device_ir_unified_intra)
      for c in "${unified_intra_cases[@]}"; do emit "$c"; done ;;
    test_device_ir_unified_inter)
      for c in "${unified_inter_cases[@]}"; do emit "$c"; done ;;
    test_device_ir_inter)
      # The IR inter suite ends with S14 and prints the result after a colon.
      for ((r = 0; r < n; r++)); do
        printf '[rank %d] S14 OneSidedAlltoAll: PASS\n' "$r"
      done ;;
    *) emit "S1 MockCase" ;;
  esac
}

case "$scenario" in
  pass)
    emit_all
    emit_unified_meta
    for ((r = 0; r < n; r++)); do
      printf '[rank %d] === Overall: PASS ===\n' "$r"
    done
    echo "=== Final Result: ALL PASS ==="
    exit 0
    ;;
  rank_fail)
    emit_all
    emit_unified_meta
    printf '[rank %d] === Overall: FAIL ===\n' "$((n - 1))"
    echo "=== Final Result: SOME FAILED ==="
    exit 1
    ;;
  partial_fail)
    emit_all
    emit_unified_meta
    printf '[rank %d] K10 IntraAllReduce(composite)        FAIL\n' "$((n - 1))"
    for ((r = 0; r < n; r++)); do
      printf '[rank %d] === Overall: PASS ===\n' "$r"
    done
    echo "=== Final Result: ALL PASS ==="
    exit 0                        # deliberately wrong exit code
    ;;
  missing_case)
    # Exit 0 and print PASS for everything except the required v3 case.
    case "$base" in
      test_device_api_intra)
        for c in "${api_intra_cases[@]}"; do
          [ "$c" = "K10 IntraAllReduce(composite)" ] && continue
          emit "$c"
        done ;;
      test_device_ir_unified_inter)
        for c in "${unified_inter_cases[@]}"; do
          [ "${c%% *}" = "S25" ] && continue
          emit "$c"
        done ;;
      *) emit_all ;;
    esac
    emit_unified_meta
    for ((r = 0; r < n; r++)); do
      printf '[rank %d] === Overall: PASS ===\n' "$r"
    done
    echo "=== Final Result: ALL PASS ==="
    exit 0
    ;;
  subset_ranks)
    # The required composite case is printed by half of the ranks only, but
    # twice each -- exactly what the real suites produce across the 1K..16M
    # sweep, where one rank emits one line per size. The total hit count is
    # therefore still >= NRANKS, so a hit-counting check would call this PASS
    # while four ranks never ran the case. Everything else, including the
    # per-rank "=== Overall: PASS ===" line, is complete.
    case "$base" in
      test_device_api_intra)
        for c in "${api_intra_cases[@]}"; do
          [ "$c" = "K10 IntraAllReduce(composite)" ] && continue
          emit "$c"
        done
        for ((r = 0; r < n / 2; r++)); do
          printf '[rank %d] %-35s %s\n' "$r" "K10 IntraAllReduce(composite)" PASS
          printf '[rank %d] %-35s %s\n' "$r" "K10 IntraAllReduce(composite)" PASS
        done ;;
      *) emit_all ;;
    esac
    emit_unified_meta
    for ((r = 0; r < n; r++)); do
      printf '[rank %d] === Overall: PASS ===\n' "$r"
    done
    echo "=== Final Result: ALL PASS ==="
    exit 0
    ;;
  *) echo "unknown MOCK_SCENARIO=$scenario" >&2; exit 9 ;;
esac
