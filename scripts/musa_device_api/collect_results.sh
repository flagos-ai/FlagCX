#!/usr/bin/env bash
# Collect all recorded stage results into a single table and emit an explicit
# acceptance verdict.
#
# Reads $LOG_DIR/*.status (written by the other scripts) and
# $RESULT_DIR/build_summary.env. Emits Markdown to stdout and
# $RESULT_DIR/SUMMARY.md.
#
# Acceptance is PASS only when every required C stage is present and PASS.
# BLOCKED, NOT RUN, UNKNOWN, FAIL and TIMEOUT all yield NOT_ACCEPTED.
#
# Usage: collect_results.sh [--allow-blocked] [--strict]
#   --allow-blocked  exploration mode: exit 0 despite BLOCKED/NOT_RUN, but the
#                    summary still says NOT_ACCEPTED and reports the counts.
#                    FAIL/TIMEOUT/UNKNOWN always exit non-zero.
#   --strict         kept for compatibility; acceptance is the default.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

allow_blocked=0
for arg in "$@"; do
  case "$arg" in
    --allow-blocked) allow_blocked=1 ;;
    --strict) allow_blocked=0 ;;
    -h|--help) echo "Usage: $0 [--allow-blocked] [--strict]"; exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

md="$RESULT_DIR/SUMMARY.md"
mkdir -p "$RESULT_DIR"

kv() { # kv <file> <key>
  sed -n "s/^$2=//p" "$1" 2>/dev/null | head -1
}

# Stages this C work package must produce for acceptance. Presence of a
# status file is not enough: it has to be PASS.
REQUIRED_STAGES=(
  env_check
  build_host
  build_unit
  build_device
  run_intra
  run_inter
  run_unified_intra
  run_unified_inter
  run_real_inter
)
# COLLECT_REQUIRED_STAGES narrows the required set for a single-stage rerun.
if [ -n "${COLLECT_REQUIRED_STAGES:-}" ]; then
  read -r -a REQUIRED_STAGES <<<"$COLLECT_REQUIRED_STAGES"
fi

declare -a rows=() present=() missing=()
fail_count=0 timeout_count=0 blocked_count=0
notrun_count=0 unknown_count=0 pass_count=0 inconsistent_count=0

printf '# MUSA Device API — collected results\n\n' >"$md"
printf '| stage | status | exit | duration | verdict | detail |\n' >>"$md"
printf '|---|---|---|---|---|---|\n' >>"$md"

while IFS= read -r st; do
  [ -n "$st" ] || continue
  name="$(kv "$st" name)"; name="${name:-$(basename "$st" .status)}"
  status="$(kv "$st" status)"; status="${status:-UNKNOWN}"
  rc="$(kv "$st" exit)"; rc="${rc:-?}"
  dur="$(kv "$st" duration_s)"; dur="${dur:-?}"
  verdict="$(kv "$st" verdict)"; verdict="${verdict:-}"
  cmd="$(kv "$st" command)"
  present+=("$name")
  # run_capture() records a provisional status from the exit code while the
  # verdict comes from classify_output() plus the log assertions. A row whose
  # status and verdict disagree (in particular status=PASS with a non-PASS
  # verdict) is not evidence of anything and must never be counted as a pass.
  if ! status_is_consistent "$status" "$verdict"; then
    inconsistent_count=$((inconsistent_count + 1))
    printf '| %s | INCONSISTENT | %s | %ss | %s | %s |\n' \
      "$name" "$rc" "$dur" "$verdict" "${cmd//|/\\|}" >>"$md"
    rows+=("$name|INCONSISTENT|$rc|$dur|$verdict")
    warn "$name: status=$status disagrees with verdict=$verdict (counted as INCONSISTENT)"
    continue
  fi
  case "$status" in
    PASS) pass_count=$((pass_count + 1)) ;;
    FAIL|ERROR) fail_count=$((fail_count + 1)) ;;
    TIMEOUT) timeout_count=$((timeout_count + 1)) ;;
    BLOCKED) blocked_count=$((blocked_count + 1)) ;;
    SKIP|NOT_RUN) notrun_count=$((notrun_count + 1)) ;;
    *) unknown_count=$((unknown_count + 1)) ;;
  esac
  printf '| %s | %s | %s | %ss | %s | %s |\n' \
    "$name" "$status" "$rc" "$dur" "$verdict" "${cmd//|/\\|}" >>"$md"
  rows+=("$name|$status|$rc|$dur|$verdict")
done < <(find "$LOG_DIR" -maxdepth 1 -name '*.status' -type f | sort)

for want in "${REQUIRED_STAGES[@]}"; do
  found=0
  for got in "${present[@]:-}"; do
    [ "$got" = "$want" ] && { found=1; break; }
  done
  [ "$found" -eq 0 ] && missing+=("$want")
done

if [ -f "$RESULT_DIR/build_summary.env" ]; then
  printf '\n## Build summary\n\n```\n' >>"$md"
  cat "$RESULT_DIR/build_summary.env" >>"$md"
  printf '```\n' >>"$md"
fi

not_ok=$((fail_count + timeout_count + blocked_count + notrun_count + unknown_count))
not_ok=$((not_ok + ${#missing[@]} + inconsistent_count))

printf '\n## Totals\n\n' >>"$md"
printf -- '- PASS: %d\n- FAIL: %d\n- TIMEOUT: %d\n- BLOCKED: %d\n' \
  "$pass_count" "$fail_count" "$timeout_count" "$blocked_count" >>"$md"
printf -- '- NOT RUN (incl. missing required stages): %d\n- UNKNOWN: %d\n' \
  "$((notrun_count + ${#missing[@]}))" "$unknown_count" >>"$md"
printf -- '- INCONSISTENT (status/verdict disagree): %d\n' \
  "$inconsistent_count" >>"$md"
printf -- '- Missing required stages: %s\n' \
  "${missing[*]:-<none>}" >>"$md"

if [ "$not_ok" -eq 0 ]; then
  printf '\n## Acceptance\n\n- ACCEPTANCE: PASS\n' >>"$md"
  acceptance=PASS
else
  printf '\n## Acceptance\n\n- ACCEPTANCE: NOT_ACCEPTED\n' >>"$md"
  printf -- '- Blocking items: FAIL=%d TIMEOUT=%d BLOCKED=%d NOT_RUN=%d UNKNOWN=%d MISSING=%d INCONSISTENT=%d\n' \
    "$fail_count" "$timeout_count" "$blocked_count" \
    "$((notrun_count + ${#missing[@]}))" "$unknown_count" "${#missing[@]}" \
    "$inconsistent_count" >>"$md"
  acceptance=NOT_ACCEPTED
fi

cat "$md"
echo
log "summary written to $md (acceptance=$acceptance)"

# FAIL/TIMEOUT/UNKNOWN are hard failures in every mode.
if [ "$fail_count" -gt 0 ] || [ "$timeout_count" -gt 0 ] || [ "$unknown_count" -gt 0 ]; then
  exit 1
fi
# A contradicting status/verdict pair is a hard failure: the evidence itself
# is broken, so no mode may treat it as acceptable.
if [ "$inconsistent_count" -gt 0 ]; then
  exit 1
fi
# NOT RUN (including a missing required stage) is never acceptable, not even
# in exploration mode: it means the evidence does not exist.
if [ "$((notrun_count + ${#missing[@]}))" -gt 0 ]; then
  exit 2
fi
# BLOCKED is tolerated only in explicit exploration mode, and never turns into
# an ACCEPTANCE PASS.
if [ "$blocked_count" -gt 0 ] && [ "$allow_blocked" -eq 0 ]; then
  exit 2
fi
exit 0
