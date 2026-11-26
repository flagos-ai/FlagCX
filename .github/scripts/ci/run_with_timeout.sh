#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/timeouts.sh"
test_timeout="$FLAGCX_CI_TEST_TIMEOUT"
test_kill_after="${FLAGCX_CI_TEST_KILL_AFTER:-30s}"
test_label="${FLAGCX_CI_TEST_LABEL:-test invocation}"
max_attempts="${FLAGCX_CI_TEST_MAX_ATTEMPTS:-1}"
diagnostics_after="${FLAGCX_CI_TEST_DIAGNOSTICS_AFTER:-}"
log_root="${FLAGCX_CI_LOG_DIR:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}/flagcx-ci-attempts}"
safe_label=$(printf '%s' "$test_label" | tr -c 'A-Za-z0-9_.-' '_')

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <command> [arguments...]" >&2
  exit 2
fi
if [[ "$max_attempts" != 1 && "$max_attempts" != 2 ]]; then
  echo "FLAGCX_CI_TEST_MAX_ATTEMPTS must be 1 or 2" >&2
  exit 2
fi
if ! command -v timeout >/dev/null 2>&1; then
  echo "GNU timeout is required to run CI tests" >&2
  exit 1
fi
mkdir -p "$log_root"
log_dir=$(mktemp -d "$log_root/$safe_label.XXXXXX")
chmod 755 "$log_dir"

active_pid=
diagnostic_pid=
cleanup_workload() {
  if [[ -n "$diagnostic_pid" ]]; then
    kill -TERM "$diagnostic_pid" 2>/dev/null || true
    wait "$diagnostic_pid" 2>/dev/null || true
    diagnostic_pid=
  fi
  if [[ -n "$active_pid" ]]; then
    # Nested timeout wrappers own separate process groups. Snapshot all of
    # them before stopping the root, while their parent links still exist.
    local -a workload_groups=()
    local groups_text
    groups_text=$(python3 - "$active_pid" <<'PYGROUPS'
import os
from pathlib import Path
import sys
root = int(sys.argv[1])
parents = {}
proc = Path("/proc")
if proc.is_dir():
    for process in proc.iterdir():
        if not process.name.isdigit():
            continue
        try:
            fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
            parents[int(process.name)] = int(fields[1])
        except (OSError, IndexError, ValueError):
            pass
owned = {root}
while True:
    expanded = owned | {pid for pid, parent in parents.items() if parent in owned}
    if expanded == owned:
        break
    owned = expanded
groups = {root}
for pid in owned:
    try:
        groups.add(os.getpgid(pid))
    except ProcessLookupError:
        pass
groups.discard(os.getpgrp())
for group in sorted(groups):
    print(group)
PYGROUPS
    )
    while IFS= read -r group_id; do
      [[ -n "$group_id" ]] && workload_groups+=("$group_id")
    done <<< "$groups_text"
    for group_id in "${workload_groups[@]}"; do
      kill -TERM -- "-$group_id" 2>/dev/null || true
    done
    timeout --signal=TERM --kill-after=1s "$test_kill_after" \
      bash -c 'for group_id; do
        while kill -0 -- "-$group_id" 2>/dev/null; do sleep 0.05; done
      done' bash "${workload_groups[@]}" >/dev/null 2>&1 || true
    for group_id in "${workload_groups[@]}"; do
      kill -KILL -- "-$group_id" 2>/dev/null || true
    done
    wait "$active_pid" 2>/dev/null || true
    active_pid=
  fi
}
cancel_workload() {
  trap '' INT TERM
  cleanup_workload
  exit "$1"
}
trap cleanup_workload EXIT
trap 'cancel_workload 130' INT
trap 'cancel_workload 143' TERM

attempt=1
while ((attempt <= max_attempts)); do
  timeout_info=$(flagcx_ci_limit_timeout "$test_timeout" details) || {
    status=$?
    if [[ "$status" == 124 ]]; then
      echo "$test_label: $timeout_info cumulative budget exhausted; invocation not run" >&2
    else
      echo "$test_label: invalid timeout or deadline; invocation not run" >&2
    fi
    exit "$status"
  }
  read -r effective_timeout timeout_source <<< "$timeout_info"
  attempt_log="$log_dir/attempt-$attempt.log"
  watchdog_log="$log_dir/watchdog-attempt-$attempt.log"
  echo "Starting $test_label (attempt $attempt/$max_attempts, timeout=$test_timeout, kill-after=$test_kill_after, effective-timeout=$effective_timeout, limit-source=$timeout_source)"
  exec {log_fd}> >(tee "$attempt_log")
  log_pid=$!
  # Preserve heredoc input for Python preflights launched in the background.
  exec {input_fd}<&0
  # Only timeout writes to watchdog_log. The child keeps both output streams
  # in attempt_log, so a nested timeout cannot claim this watchdog expired.
  timeout --verbose --signal=TERM --kill-after="$test_kill_after" \
    "$effective_timeout" bash -c 'exec "$@" 2>&1' bash "$@" \
    <&"$input_fd" >&"$log_fd" 2>"$watchdog_log" &
  active_pid=$!
  exec {input_fd}<&-
  if [[ -n "$diagnostics_after" ]]; then
    diagnostic_delay=$(awk -v requested="$diagnostics_after" -v limit="${effective_timeout%s}" \
      'BEGIN { ceiling = limit * 0.75; print requested < ceiling ? requested : ceiling }')
    python3 "$script_dir/collect_process_stacks.py" \
      --pid "$active_pid" --delay "$diagnostic_delay" \
      > "$log_dir/attempt-$attempt-stacks.log" 2>&1 &
    diagnostic_pid=$!
  fi
  status=0
  wait "$active_pid" || status=$?
  cleanup_workload
  cat "$watchdog_log" >&"$log_fd"
  exec {log_fd}>&-
  wait "$log_pid"
  if [[ "$status" == 0 && "${FLAGCX_CI_REQUIRE_GTEST_PASS:-0}" == 1 ]] &&
     ! grep -Eq '\[ *PASSED *\] +[1-9][0-9]* tests?\.' "$attempt_log"; then
    echo "$test_label exited successfully without a passing GTest case" >&2
    status=1
  fi
  if [[ "$status" == 0 ]]; then
    if ((attempt > 1)); then
      echo "::warning title=Flaky CI test::$test_label passed on retry $((attempt - 1))"
      echo "$test_label passed on retry $((attempt - 1)); classify as flaky"
      if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        {
          echo "### Flaky CI test"
          echo "- **Test:** $test_label"
          echo "- **Result:** passed on retry $((attempt - 1))"
          printf '%s\n' "- **First attempt log:** $log_dir/attempt-1.log"
        } >> "$GITHUB_STEP_SUMMARY"
      fi
    fi
    exit 0
  fi
  if [[ ( "$status" == 124 || "$status" == 137 ) && -s "$watchdog_log" ]]; then
    echo "$test_label watchdog expired on attempt $attempt (effective-timeout=$effective_timeout, limit-source=$timeout_source, configured-timeout=$test_timeout)" >&2
  elif [[ "$status" == 124 || "$status" == 137 ]]; then
    echo "$test_label command exited with status=$status; this invocation's watchdog did not expire" >&2
  fi
  echo "$test_label attempt $attempt failed with exit=$status; log saved to $attempt_log" >&2
  if [[ -n "${FLAGCX_CI_ATTEMPT_FAILURE_HOOK:-}" ]]; then
    diagnostic_log="$log_dir/diagnostics-attempt-$attempt.log"
    diagnostic_status=0
    bash "$FLAGCX_CI_ATTEMPT_FAILURE_HOOK" "$test_label" "$attempt" \
      "$status" >"$diagnostic_log" 2>&1 || diagnostic_status=$?
    cat "$diagnostic_log"
    if ((diagnostic_status == 86)); then
      echo "$test_label: RDMA devices disappeared before retry; diagnostics saved to $diagnostic_log" >&2
      exit "$status"
    fi
  fi
  if ((attempt == max_attempts)) || [[ "$status" == 130 || "$status" == 143 || "$status" == 126 || "$status" == 127 ]]; then
    exit "$status"
  fi
  echo "$test_label failed with exit=$status; retrying once" >&2
  attempt=$((attempt + 1))
done
