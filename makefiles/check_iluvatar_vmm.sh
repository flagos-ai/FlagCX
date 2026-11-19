#!/usr/bin/env bash
set -euo pipefail

# check emits only 0/1 for Make. diagnose keeps every error without requiring
# VMM to be available, so the legacy library can still build on older SDKs.
if (( $# < 4 )); then
  echo "Usage: $0 <check|diagnose> <device-home> [log-dir] -- <compiler> [flags...]" >&2
  exit 2
fi
mode=$1
device_home=$2
shift 2
if [[ "$mode" == diagnose ]]; then
  log_dir=$1
  shift
elif [[ "$mode" != check ]]; then
  echo "Unknown VMM check mode: $mode" >&2
  exit 2
fi
if [[ "$1" != -- ]] || (( $# < 2 )); then
  echo 'Expected -- followed by the host compiler and its flags' >&2
  exit 2
fi
shift
compiler=("$@")
source_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_file="$source_dir/iluvatar_vmm_check.cpp"
common=(-std=c++17 "-I$device_home/include" -x c++ -fsyntax-only)

if [[ "$mode" == check ]]; then
  if "${compiler[@]}" "${common[@]}" "$source_file" >/dev/null 2>&1; then
    echo 1
  else
    echo 0
  fi
  exit 0
fi

umask 022
mkdir -p "$log_dir"
chmod 755 "$log_dir"
cp "$source_file" "$log_dir/iluvatar_vmm_check.cpp"
{
  printf 'DEVICE_HOME=%s\n' "$device_home"
  printf 'Compiler: '; printf '%q ' "${compiler[@]}"; printf '\n'
  "${compiler[@]}" --version || true
  ls -l "$device_home/include/cuda.h" || true
  "${compiler[@]}" -std=c++17 "-I$device_home/include" -dM -E -x c++ "$source_file" \
    2> "$log_dir/preprocess.log" \
    | grep -E '^#define (CUDA_VERSION|CUDART_VERSION|COREX_VERSION|IXCUDA_VERSION)([[:space:]]|_)' || true
} > "$log_dir/environment.log" 2>&1
cat "$log_dir/environment.log"

supported=1
for stage in types basic posix_fd complete; do
  case "$stage" in
    types) number=1 ;;
    basic) number=2 ;;
    posix_fd) number=3 ;;
    complete) number=0 ;;
  esac
  command=("${compiler[@]}" "${common[@]}" "-DFLAGCX_VMM_CHECK_STAGE=$number" "$source_file")
  {
    printf '%q ' "${command[@]}"
    printf '\n'
  } > "$log_dir/$stage.command"
  status=0
  "${command[@]}" > "$log_dir/$stage.log" 2>&1 || status=$?
  printf 'IXCUDA VMM SDK check: %s exit=%s\n' "$stage" "$status"
  cat "$log_dir/$stage.command" "$log_dir/$stage.log"
  printf '%s\n' "$status" > "$log_dir/$stage.status"
  if ((status != 0)); then supported=0; fi
done
printf 'FLAGCX_ILUVATAR_HAS_VMM=%s\n' "$supported" | tee "$log_dir/result.env"
echo "Compile-only check; CoreX runtime and hardware capabilities remain unverified."
