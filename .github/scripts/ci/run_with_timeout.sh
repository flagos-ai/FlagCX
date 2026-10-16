#!/usr/bin/env bash

set -euo pipefail

test_timeout="${FLAGCX_CI_TEST_TIMEOUT:-20m}"
test_kill_after="${FLAGCX_CI_TEST_KILL_AFTER:-30s}"
test_label="${FLAGCX_CI_TEST_LABEL:-test invocation}"
log_root="${FLAGCX_CI_LOG_DIR:-${RUNNER_TEMP:-${TMPDIR:-/tmp}}/flagcx-ci-attempts}"
safe_label=$(printf '%s' "$test_label" | tr -c 'A-Za-z0-9_.-' '_')

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <command> [arguments...]" >&2
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
cleanup_workload() {
  if [[ -n "$active_pid" ]]; then
    # timeout owns a process group; stop leftover ranks before another attempt.
    if kill -0 -- "-$active_pid" 2>/dev/null; then
      kill -TERM -- "-$active_pid" 2>/dev/null || true
      timeout --signal=TERM --kill-after=1s "$test_kill_after" \
        bash -c 'while kill -0 -- "-$1" 2>/dev/null; do sleep 0.05; done' \
        bash "$active_pid" >/dev/null 2>&1 || true
      kill -KILL -- "-$active_pid" 2>/dev/null || true
    fi
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
max_attempts=2
while ((attempt <= max_attempts)); do
  attempt_log="$log_dir/attempt-$attempt.log"
  echo "Starting $test_label (attempt $attempt/$max_attempts, timeout=$test_timeout, kill-after=$test_kill_after)"
  exec {log_fd}> >(tee "$attempt_log")
  log_pid=$!
  timeout --signal=TERM --kill-after="$test_kill_after" \
    "$test_timeout" "$@" >&"$log_fd" 2>&1 &
  active_pid=$!
  status=0
  wait "$active_pid" || status=$?
  cleanup_workload
  exec {log_fd}>&-
  wait "$log_pid"
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
  if [[ "$status" == 124 || "$status" == 137 ]]; then
    echo "$test_label exceeded its $test_timeout timeout on attempt $attempt" >&2
  fi
  echo "$test_label attempt $attempt failed with exit=$status; log saved to $attempt_log" >&2
  if ((attempt == max_attempts)) || [[ "$status" == 130 || "$status" == 143 || "$status" == 126 || "$status" == 127 ]]; then
    exit "$status"
  fi
  echo "$test_label failed with exit=$status; retrying once" >&2
  attempt=$((attempt + 1))
done
