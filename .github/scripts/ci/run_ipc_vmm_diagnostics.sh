#!/usr/bin/env bash
set -euo pipefail
umask 022
if [[ $# != 2 ]]; then
  echo "Usage: $0 <SDK-only-reproducer> <result-directory>" >&2
  exit 2
fi
binary=$1
results=$2
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$results"
if [[ ! -x "$binary" ]] || ! command -v timeout >/dev/null; then
  echo 'IPC diagnostics require the SDK-only reproducer and GNU timeout' >&2
  exit 2
fi
{
  uname -a
  printf 'Reproducer: %s\n' "$binary"
  ldd "$binary" || true
  command -v gdb >/dev/null && gdb --version || true
  # Record the exact runtime/driver objects, including host-mounted libraries.
  while IFS= read -r object; do
    [[ -f "$object" ]] || continue
    readlink -f "$object" || true
    sha256sum "$object" || true
    readelf -n "$object" || true
  done < <(ldd "$binary" 2>/dev/null | awk '$2 == "=>" && $3 ~ /^\// && $1 ~ /^lib(cuda|ix)/ { print $3 }')
} > "$results/libraries.log" 2>&1

status=0
debugger=()
if command -v gdb >/dev/null; then
  debugger=(gdb -q -nx -batch --return-child-result -x "$script_dir/ipc_vmm_backtrace.gdb" --args)
else
  echo 'gdb unavailable; IPC diagnostics will retain stage logs without a live backtrace' >&2
  status=1
fi
printf 'case\texit\tresult\n' > "$results/summary.tsv"
for control in malloc vmm vmm-driver vmm-sync; do
  case_status=0
  # Each potentially crashing control has its own process and bounded lifetime.
  timeout --signal=TERM --kill-after=5s 30s \
    "${debugger[@]}" "$binary" "$control" > "$results/$control.log" 2>&1 || case_status=$?
  if ((case_status == 0)) && ! grep -Fxq "IPC_REPRO_PASS case=$control" "$results/$control.log"; then
    case_status=1
  fi
  result=PASS
  if ((case_status != 0)); then status=1; result=FAIL; fi
  printf '%s\n' "$case_status" > "$results/$control.status"
  printf '%s\t%s\t%s\n' "$control" "$case_status" "$result" >> "$results/summary.tsv"
done
cat "$results/summary.tsv"
exit "$status"
