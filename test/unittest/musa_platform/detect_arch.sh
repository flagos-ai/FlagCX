#!/usr/bin/env bash
# Read GPU/runtime properties and compile SDK-only architecture probes.
# Does not launch kernels or change the project's MUSA_ARCH configuration.
set -u
set -o pipefail

device_home=${DEVICE_HOME:-/usr/local/musa}
host_cxx=${CXX:-g++}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$script_dir/build" || exit 1
report_dir=$(mktemp -d "$script_dir/build/arch_detect.XXXXXX") || exit 1
exec > >(tee "$report_dir/report.log") 2>&1
printf 'REPORT_DIR=%s\nDEVICE_HOME=%s\n' "$report_dir" "$device_home"
printf 'Compiler executable: '
readlink -f "$device_home/bin/mcc"
timeout --kill-after=5s 20s "$device_home/bin/mcc" --version
printf '\n'

cat > "$report_dir/device_properties.cc" <<'CPP'
#include <musa_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(call) do { \
  musaError_t error = (call); \
  if (error != musaSuccess) { \
    std::fprintf(stderr, "%s failed: %s\n", #call, musaGetErrorString(error)); \
    return EXIT_FAILURE; \
  } \
} while (0)

int main() {
  int runtime = 0, driver = 0, count = 0;
  CHECK(musaRuntimeGetVersion(&runtime));
  CHECK(musaDriverGetVersion(&driver));
  CHECK(musaGetDeviceCount(&count));
  std::printf("runtime_version_raw=%d\ndriver_version_raw=%d\ndevice_count=%d\n",
              runtime, driver, count);
  if (!count) return EXIT_FAILURE;
  for (int i = 0; i < count; ++i) {
    musaDeviceProp prop{};
    CHECK(musaGetDeviceProperties(&prop, i));
    std::printf("device_%d_name=%s\n", i, prop.name);
    std::printf("device_%d_compute_capability=%d.%d\n", i, prop.major, prop.minor);
    std::printf("device_%d_arch=mp_%d%d\n", i, prop.major, prop.minor);
    std::printf("device_%d_warp_size=%d\n", i, prop.warpSize);
    std::printf("device_%d_max_threads_per_block=%d\n", i, prop.maxThreadsPerBlock);
  }
  return EXIT_SUCCESS;
}
CPP

export LD_LIBRARY_PATH="$device_home/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
property_status=1
printf '%s\n' '=== Runtime and actual visible GPU properties ==='
if timeout --kill-after=5s 45s "$host_cxx" -std=c++17 \
    -I"$device_home/include" "$report_dir/device_properties.cc" \
    -L"$device_home/lib" -lmusart -lmusa -o "$report_dir/device_properties" \
    > "$report_dir/host_build.log" 2>&1; then
  printf '%s\n' 'Actual dynamic library resolution:'
  ldd "$report_dir/device_properties" > "$report_dir/ldd.log" 2>&1
  cat "$report_dir/ldd.log"
  timeout --kill-after=5s 30s "$report_dir/device_properties" \
    > "$report_dir/device_properties.log" 2>&1
  property_status=$?
  cat "$report_dir/device_properties.log"
else
  property_status=$?
  head -c 8192 "$report_dir/host_build.log"
  printf '\n'
fi
printf 'PROPERTY_PROBE_EXIT=%d\n\n' "$property_status"

cat > "$report_dir/arch_compile.cu" <<'CPP'
#include <musa_runtime.h>
extern "C" __global__ void archCompileProbe(unsigned *out) {
  out[threadIdx.x] = threadIdx.x;
}
CPP

printf '%s\n' '=== Compiler target checks (compile only; no kernel execution) ==='
arch_candidates=(mp_31 mp_43 "$@")
if [[ -f "$report_dir/device_properties.log" ]]; then
  while IFS= read -r detected_arch; do
    arch_candidates+=("$detected_arch")
  done < <(awk -F= '/^device_[0-9]+_arch=mp_[0-9]+$/ {print $2}' "$report_dir/device_properties.log")
fi
declare -A seen_arch=()
for arch in "${arch_candidates[@]}"; do
  if [[ ! "$arch" =~ ^mp_[0-9]+$ ]]; then
    printf 'Ignoring invalid target name: %s\n' "$arch"
    continue
  fi
  [[ ${seen_arch[$arch]+yes} ]] && continue
  seen_arch[$arch]=1
  if timeout --kill-after=5s 45s "$device_home/bin/mcc" \
      -std=c++17 -O2 -c -mtgpu --musa-path="$device_home" \
      --offload-arch="$arch" -fgpu-rdc -fPIC --cuda-device-only \
      "$report_dir/arch_compile.cu" -o "$report_dir/$arch.o" \
      > "$report_dir/$arch.log" 2>&1; then
    if [[ -s "$report_dir/$arch.o" ]]; then
      printf '%s compile=PASS (nonempty device object; runtime compatibility NOT tested)\n' "$arch"
    else
      printf '%s compile=FAIL (compiler returned success but no nonempty object)\n' "$arch"
    fi
  else
    compile_status=$?
    printf '%s compile=FAIL exit=%d (inspect diagnostic; not necessarily unsupported architecture)\n' \
      "$arch" "$compile_status"
    head -c 4096 "$report_dir/$arch.log"
    printf '\n'
  fi
done
printf '\nFull report: %s/report.log\n' "$report_dir"
printf '%s\n' 'No GPU kernels were launched. Compiler acceptance is separate from hardware compatibility.'
# The compile results are reported individually; a failed property query must
# remain visible to scripts consuming this report.
exit "$property_status"
