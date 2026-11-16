#!/usr/bin/env bash
# Harness red-team tests for the C-owned MUSA Device API wrappers.
#
# These tests use mock `mpirun`/`mthreads-gmi` processes and fake ELF test
# binaries on purpose. They validate ONLY the wrapper state machine:
#   PASS/FAIL/TIMEOUT/BLOCKED/NOT_RUN, exit-code propagation, log assertions,
#   stale-log isolation and repeated-round aggregation.
#
# They are NOT Device API correctness tests and must never be presented as
# T1-T4 acceptance evidence.
#
# Usage: tests/run_harness_tests.sh

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$(cd "$HERE/.." && pwd)"
PROJECT_ROOT="$(cd "$SCRIPTS/../.." && pwd)"
MOCK_MPIRUN="$HERE/mock_mpirun.sh"
MOCK_GMI="$HERE/mock_gmi.sh"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/flagcx-musa-harness.XXXXXX")"
export RESULT_DIR="$WORK/results"
export LOG_DIR="$RESULT_DIR/logs"
mkdir -p "$LOG_DIR"

# Fake freshly-built libflagcx so assert_runtime_lib sees a real ELF and the
# expected absolute path.
FAKE_LIB="$WORK/fake-lib"
mkdir -p "$FAKE_LIB"
printf 'int flagcx_harness_fake(void) { return 0; }\n' >"$WORK/fake.c"
gcc -shared -fPIC -o "$FAKE_LIB/libflagcx.so" "$WORK/fake.c" || {
  echo "harness: cannot build fake libflagcx" >&2
  exit 1
}
export FLAGCX_BUILD_LIB="$FAKE_LIB"

BIN="$WORK/bin"
mkdir -p "$BIN"
printf 'int main(void) { return 0; }\n' >"$WORK/main.c"
for b in test_device_api_intra test_device_api_inter test_device_ir_intra \
         test_device_ir_inter test_device_ir_unified_intra \
         test_device_ir_unified_inter; do
  gcc -o "$BIN/$b" "$WORK/main.c" -L"$FAKE_LIB" \
    -Wl,--no-as-needed -lflagcx -Wl,-rpath,"$FAKE_LIB" || exit 1
done
EMPTY_BIN="$WORK/empty-bin"
mkdir -p "$EMPTY_BIN"

export MPIRUN="$MOCK_MPIRUN"
export MUSA_GMI_BIN="$MOCK_GMI"

# shellcheck source=../common.sh
source "$SCRIPTS/common.sh"

PASS_N=0
FAIL_N=0
FAILED_NAMES=()
ok()  { PASS_N=$((PASS_N + 1)); printf '  ok   %s\n' "$1"; }
bad() { FAIL_N=$((FAIL_N + 1)); FAILED_NAMES+=("$1"); printf '  FAIL %s\n' "$1"; }
eq()  { # eq <actual> <expected> <label>
  if [ "$1" = "$2" ]; then ok "$3"; else bad "$3 (expected [$2], got [$1])"; fi
}
ne()  { if [ "$1" != "$2" ]; then ok "$3"; else bad "$3 (unexpected [$2])"; fi; }
has() { # has <file> <literal> <label>
  if grep -qF -- "$2" "$1" 2>/dev/null; then ok "$3"; else bad "$3 (missing: $2)"; fi
}
hasnt() { if ! grep -qF -- "$2" "$1" 2>/dev/null; then ok "$3"; else bad "$3 (unexpected: $2)"; fi; }

newcase() { local d="$WORK/$1"; mkdir -p "$d/logs"; printf '%s' "$d"; }
status_of() { sed -n 's/^status=//p' "$1/logs/$2.status" 2>/dev/null | head -1; }

CASE_SCENARIO=pass
CASE_BIN="$BIN"
CASE_DEVICES=8
CASE_TIMEOUT=1800
CASE_FAIL_ROUNDS=""
CASE_NODE0="0,1,2,3"
CASE_NODE1="4,5,6,7"

# NOTE: this environment has a non-functional `env` wrapper in PATH, so all
# injection below uses a subshell with exported variables instead of `env`.
run_script() { # run_script <case-dir> <script> [args...]
  local dir="$1" script="$2"; shift 2
  (
    export RESULT_DIR="$dir" LOG_DIR="$dir/logs"
    export DEVICE_API_BIN_DIR="$CASE_BIN" MPIRUN="$MOCK_MPIRUN"
    export MUSA_GMI_BIN="$MOCK_GMI" MOCK_SCENARIO="$CASE_SCENARIO"
    export MOCK_NRANKS=8 MOCK_DEVICES="$CASE_DEVICES"
    export MOCK_FAIL_ROUNDS="$CASE_FAIL_ROUNDS" RUN_TIMEOUT="$CASE_TIMEOUT"
    export INTER_NODE0_DEVICES="$CASE_NODE0" INTER_NODE1_DEVICES="$CASE_NODE1"
    bash "$SCRIPTS/$script" "$@"
  )
}

reset_case() {
  CASE_SCENARIO=pass; CASE_BIN="$BIN"; CASE_DEVICES=8; CASE_TIMEOUT=1800
  CASE_FAIL_ROUNDS=""; CASE_NODE0="0,1,2,3"; CASE_NODE1="4,5,6,7"
}

echo "== Harness: wrapper state machine =="

# T01 rank exits non-zero ------------------------------------------------
reset_case; CASE_SCENARIO=rank_fail
d="$(newcase t01)"; run_script "$d" run_intra.sh >/dev/null 2>&1; rc=$?
eq "$rc" 1 "T01 nonzero rank -> overall FAIL exit"
eq "$(status_of "$d" run_intra)" FAIL "T01 run_intra status FAIL"

# T02 rank killed by signal ----------------------------------------------
reset_case; CASE_SCENARIO=signal
d="$(newcase t02)"; run_script "$d" run_intra.sh >/dev/null 2>&1; rc=$?
eq "$rc" 1 "T02 signalled rank -> overall FAIL exit"
eq "$(status_of "$d" run_intra)" FAIL "T02 run_intra status FAIL"

# T03 launcher returns non-zero ------------------------------------------
reset_case; CASE_SCENARIO=launcher_err
d="$(newcase t03)"; run_script "$d" run_intra.sh >/dev/null 2>&1; rc=$?
eq "$rc" 1 "T03 launcher error -> overall FAIL exit"
eq "$(status_of "$d" run_intra)" FAIL "T03 run_intra status FAIL"

# T04 timeout ------------------------------------------------------------
reset_case; CASE_SCENARIO=timeout; CASE_TIMEOUT=1
d="$(newcase t04)"; run_script "$d" run_intra.sh >/dev/null 2>&1; rc=$?
eq "$rc" 1 "T04 timeout -> non-zero exit"
eq "$(status_of "$d" run_intra)" TIMEOUT "T04 run_intra status TIMEOUT"
has "$d/logs/run_intra.api.status" "status=TIMEOUT" "T04 per-target status TIMEOUT"

# T05 local PASS but overall FAIL ---------------------------------------
reset_case; CASE_SCENARIO=partial_fail
d="$(newcase t05)"; run_script "$d" run_intra.sh >/dev/null 2>&1; rc=$?
eq "$rc" 1 "T05 mixed PASS/FAIL -> FAIL exit"
eq "$(status_of "$d" run_intra)" FAIL "T05 mixed markers classified FAIL"
hasnt "$d/logs/run_intra.api.status" "verdict=PASS" "T05 verdict never PASS"

# T06 a required case never ran ----------------------------------------
reset_case; CASE_SCENARIO=missing_case
d="$(newcase t06)"; run_script "$d" run_intra.sh >/dev/null 2>&1; rc=$?
eq "$rc" 1 "T06 missing K10 -> FAIL exit"
eq "$(status_of "$d" run_intra)" FAIL "T06 missing K10 status FAIL"
has "$d/logs/run_intra.api.status" "case=FAIL T1 K10" "T06 K10 assertion FAIL"

d="$(newcase t06b)"; run_script "$d" run_unified_inter.sh >/dev/null 2>&1; rc=$?
eq "$rc" 1 "T06b missing S25 -> FAIL exit"
has "$d/logs/run_unified_inter.status" "case=FAIL T4 S25" "T06b S25 assertion FAIL"

# T07 test binary missing ----------------------------------------------
reset_case; CASE_BIN="$EMPTY_BIN"
d="$(newcase t07a)"; run_script "$d" run_intra.sh --allow-blocked >/dev/null 2>&1; rc=$?
eq "$rc" 0 "T07a missing binary + allow-blocked -> exploration exit 0"
eq "$(status_of "$d" run_intra)" BLOCKED "T07a status BLOCKED"
d="$(newcase t07b)"; run_script "$d" run_intra.sh >/dev/null 2>&1; rc=$?
eq "$rc" 2 "T07b missing binary without allow-blocked -> exit 2"

# T08 stale / wrong libflagcx ------------------------------------------
LIB_A="$WORK/libA"; LIB_B="$WORK/libB"
mkdir -p "$LIB_A" "$LIB_B"
gcc -shared -fPIC -o "$LIB_A/libflagcx.so" "$WORK/fake.c"
gcc -shared -fPIC -o "$LIB_B/libflagcx.so" "$WORK/fake.c"
OLD_BIN="$WORK/old-bin"
gcc -o "$OLD_BIN" "$WORK/main.c" -L"$LIB_A" \
  -Wl,--no-as-needed -lflagcx -Wl,-rpath,"$LIB_A"
if ( unset LD_LIBRARY_PATH; FLAGCX_BUILD_LIB="$LIB_B" \
       assert_runtime_lib oldlib "$OLD_BIN" ) >/dev/null 2>&1; then
  bad "T08 stale libflagcx must be detected"
else
  ok "T08 stale libflagcx detected"
fi
if ( unset LD_LIBRARY_PATH; FLAGCX_BUILD_LIB="$LIB_A" \
       assert_runtime_lib goodlib "$OLD_BIN" ) >/dev/null 2>&1; then
  ok "T08 matching libflagcx accepted"
else
  bad "T08 matching libflagcx rejected"
fi

# T09 insufficient devices ---------------------------------------------
reset_case; CASE_DEVICES=2
d="$(newcase t09)"; run_script "$d" run_intra.sh --allow-blocked >/dev/null 2>&1; rc=$?
eq "$rc" 0 "T09 insufficient devices + allow-blocked -> exploration exit 0"
eq "$(status_of "$d" run_intra)" BLOCKED "T09 insufficient devices BLOCKED"
has "$d/logs/run_intra.reason" "MUSA device" "T09 block reason recorded"

# T10 duplicate / overlapping device mapping ---------------------------
reset_case; CASE_NODE1="2,3,4,5"
d="$(newcase t10)"; run_script "$d" run_inter.sh --allow-blocked >/dev/null 2>&1; rc=$?
eq "$rc" 0 "T10 duplicate mapping + allow-blocked -> exploration exit 0"
eq "$(status_of "$d" run_inter)" BLOCKED "T10 duplicate mapping BLOCKED"

# T11 exit code survives redirection/timeout wrapper -------------------
run_capture harness_pipe 10 bash -c 'echo captured; exit 7' >/dev/null 2>&1; rc=$?
eq "$rc" 7 "T11 run_capture preserves child exit code"
eq "$(status_of "$RESULT_DIR" harness_pipe)" FAIL "T11 run_capture records FAIL"
has "$RESULT_DIR/logs/harness_pipe.out" "captured" "T11 stdout captured"

# T12 BLOCKED never aggregates to PASS --------------------------------
d="$(newcase t12)"; mkdir -p "$d/logs"
printf 'name=run_intra\nstatus=PASS\nexit=0\nduration_s=1\ncommand=x\n' >"$d/logs/run_intra.status"
printf 'name=run_inter\nstatus=BLOCKED\nexit=0\nduration_s=0\ncommand=y\n' >"$d/logs/run_inter.status"
RESULT_DIR="$d" LOG_DIR="$d/logs" COLLECT_REQUIRED_STAGES="run_intra run_inter" \
  bash "$SCRIPTS/collect_results.sh" >/dev/null 2>&1; rc=$?
eq "$rc" 2 "T12 BLOCKED in acceptance mode -> exit 2"
has "$d/SUMMARY.md" "ACCEPTANCE: NOT_ACCEPTED" "T12 summary NOT_ACCEPTED"
has "$d/SUMMARY.md" "BLOCKED: 1" "T12 blocked count reported"
RESULT_DIR="$d" LOG_DIR="$d/logs" COLLECT_REQUIRED_STAGES="run_intra run_inter" \
  bash "$SCRIPTS/collect_results.sh" --allow-blocked >/dev/null 2>&1; rc=$?
eq "$rc" 0 "T12 exploration mode tolerates BLOCKED exit"
has "$d/SUMMARY.md" "ACCEPTANCE: NOT_ACCEPTED" "T12 exploration still NOT_ACCEPTED"

# T15 NOT_RUN and FAIL cannot pass ------------------------------------
d="$(newcase t15)"; mkdir -p "$d/logs"
printf 'name=run_intra\nstatus=PASS\nexit=0\nduration_s=1\ncommand=x\n' >"$d/logs/run_intra.status"
printf 'name=run_unified_inter\nstatus=NOT_RUN\nexit=0\nduration_s=0\ncommand=y\n' >"$d/logs/run_unified_inter.status"
RESULT_DIR="$d" LOG_DIR="$d/logs" bash "$SCRIPTS/collect_results.sh" --allow-blocked >/dev/null 2>&1; rc=$?
eq "$rc" 2 "T15 NOT_RUN not accepted even with allow-blocked"
has "$d/SUMMARY.md" "NOT_ACCEPTED" "T15 NOT_RUN summary NOT_ACCEPTED"
d="$(newcase t15b)"; mkdir -p "$d/logs"
printf 'name=run_intra\nstatus=FAIL\nexit=1\nduration_s=1\ncommand=x\n' >"$d/logs/run_intra.status"
RESULT_DIR="$d" LOG_DIR="$d/logs" bash "$SCRIPTS/collect_results.sh" --allow-blocked >/dev/null 2>&1; rc=$?
eq "$rc" 1 "T15b FAIL always exit 1"

# T16 one required case printed by only half of the ranks ---------------
# A hit count alone cannot prove per-rank coverage: the real suites print the
# case once per rank *per size*, so four ranks over two sizes produce eight
# hits and would satisfy a ">= NRANKS" rule.
reset_case; CASE_SCENARIO=subset_ranks
d="$(newcase t16)"; run_script "$d" run_intra.sh >/dev/null 2>&1; rc=$?
eq "$rc" 1 "T16 one case on 4/8 ranks -> FAIL exit"
eq "$(status_of "$d" run_intra)" FAIL "T16 run_intra status FAIL"
has "$d/logs/run_intra.api.status" "case=FAIL T1 K10" "T16 K10 per-rank assertion FAIL"
has "$d/logs/run_intra.api.status" "missing: 4 5 6 7" "T16 missing ranks reported"
hasnt "$d/logs/run_intra.api.status" "case=PASS T1 K10" "T16 K10 never counted as PASS"

# T17 a binary that exits 0 without printing anything -------------------
# run_capture() provisionally recorded status=PASS from the exit code alone;
# the per-target record must be rewritten from the verdict, while the aggregate
# stage is FAIL because a target that produced no result cannot pass.
reset_case; CASE_SCENARIO=no_marker
d="$(newcase t17)"; run_script "$d" run_intra.sh >/dev/null 2>&1; rc=$?
eq "$rc" 1 "T17 silent exit 0 -> non-zero exit"
eq "$(status_of "$d" run_intra.api)" NOT_RUN "T17 silent target NOT_RUN"
eq "$(status_of "$d" run_intra)" FAIL "T17 silent stage FAIL"
hasnt "$d/logs/run_intra.api.status" "status=PASS" "T17 silent target never recorded PASS"

# T18 status and verdict disagree --------------------------------------
# A record whose status was derived from the exit code but whose verdict says
# FAIL is self-contradicting evidence and must never be aggregated as a pass.
d="$(newcase t18)"; mkdir -p "$d/logs"
printf 'name=run_intra\nstatus=PASS\nexit=0\nduration_s=1\ncommand=x\nverdict=FAIL\n' \
  >"$d/logs/run_intra.status"
RESULT_DIR="$d" LOG_DIR="$d/logs" COLLECT_REQUIRED_STAGES="run_intra" \
  bash "$SCRIPTS/collect_results.sh" >/dev/null 2>&1; rc=$?
eq "$rc" 1 "T18 inconsistent record -> exit 1"
has "$d/SUMMARY.md" "INCONSISTENT" "T18 inconsistency reported in summary"
has "$d/SUMMARY.md" "INCONSISTENT (status/verdict disagree): 1" \
  "T18 inconsistency counted as blocking"

# T19 pre-run evidence survives the record rewrite ----------------------
# assert_runtime_lib() writes the resolved libflagcx path before the binary
# runs; run_capture() rewrites the same file from the exit code and used to
# truncate that assertion, so a gate could win without the library proof.
d="$(newcase t19)"; mkdir -p "$d/logs"
printf 'assertion=PASS libflagcx -> /build/lib/libflagcx.so\n' \
  >"$d/logs/pre.status"
(
  export LOG_DIR="$d/logs"
  run_capture pre 10 bash -c 'exit 0'
) >/dev/null 2>&1
eq "$(status_of "$d" pre)" PASS "T19 run_capture still records PASS"
has "$d/logs/pre.status" "assertion=PASS libflagcx" "T19 pre-run assertion preserved"

# T20 a stale record is replaced, not merged ---------------------------
# Evidence from an earlier invocation must never leak into the new record:
# a surviving verdict=PASS would outvote the fresh verdict because readers
# take the first key.
d="$(newcase t20)"; mkdir -p "$d/logs"
printf 'name=st\nstatus=PASS\nexit=0\nduration_s=1\ncommand=old\nverdict=PASS\ncase=PASS stale\n' \
  >"$d/logs/st.status"
(
  export LOG_DIR="$d/logs"
  run_capture st 10 bash -c 'exit 3'
) >/dev/null 2>&1
eq "$(status_of "$d" st)" FAIL "T20 rerun records FAIL"
hasnt "$d/logs/st.status" "verdict=PASS" "T20 stale verdict not carried over"
hasnt "$d/logs/st.status" "case=PASS stale" "T20 stale assertion not carried over"

# T13 stale status files are archived, not aggregated ----------------
reset_case
d="$(newcase t13)"; mkdir -p "$d/logs"
printf 'name=stale_stage\nstatus=PASS\nexit=0\nduration_s=0\ncommand=old\n' >"$d/logs/stale_stage.status"
(
  export RESULT_DIR="$d" LOG_DIR="$d/logs" DEVICE_API_BIN_DIR="$CASE_BIN"
  export MPIRUN="$MOCK_MPIRUN" MUSA_GMI_BIN="$MOCK_GMI"
  export MOCK_SCENARIO=pass MOCK_NRANKS=8 MOCK_DEVICES=8 RUN_TIMEOUT=1800
  bash "$SCRIPTS/run_all.sh" --skip-env --skip-build --skip-host \
    --allow-blocked >/dev/null 2>&1
); rc=$?
eq "$rc" 2 "T13 skipped required stages still exit 2 (NOT_RUN)"
hasnt "$d/SUMMARY.md" "stale_stage" "T13 stale stage not aggregated"
n_arch="$(find "$d" -maxdepth 1 -type d -name 'archive-*' | wc -l)"
eq "$n_arch" 1 "T13 previous logs archived once"
has "$(find "$d" -maxdepth 1 -type d -name 'archive-*' | head -1)/stale_stage.status" \
  "name=stale_stage" "T13 stale status preserved in archive"

# Static drift guard: hardcoded case patterns still exist in official tests.
echo "== Harness: official case-pattern drift guard =="
check_literal() { # check_literal <file> <literal> <label>
  if grep -qF -- "$2" "$PROJECT_ROOT/$1"; then ok "$3"; else bad "$3 (source drift: $1)"; fi
}
check_literal test/unittest/device_api/test_device_api_intra.cpp \
  'K10 IntraAllReduce(composite)' "K10 source literal"
check_literal test/unittest/device_api/test_device_api_inter.cpp \
  'K14 OneSidedAlltoAll' "K14 source literal"
for s in 16 17 18 19 20 21 22 23 24 25; do
  check_literal test/unittest/device_api/test_device_ir_unified_intra.cpp \
    "S$s " "unified intra S$s literal"
  check_literal test/unittest/device_api/test_device_ir_unified_inter.cpp \
    "S$s " "unified inter S$s literal"
done

echo
echo "== Harness summary: PASS=$PASS_N FAIL=$FAIL_N =="
if [ "$FAIL_N" -ne 0 ]; then
  printf 'failed: %s\n' "${FAILED_NAMES[*]}"
  exit 1
fi
echo "HARNESS: PASS ($PASS_N checks)"
rm -rf "$WORK"
exit 0
