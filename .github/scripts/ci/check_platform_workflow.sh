#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 8 && $# -ne 9 && $# -ne 10 ]]; then
  echo "Usage: METAX_EXPERIMENTAL_RESULT=<result> ILUVATAR_RESULT=<result> $0 <platforms-json> <config> <cuda-first> <cuda-second> <hygon> [hygon-experimental] <metax> <ppu> [ppu-experimental] <ascend>" >&2
  exit 2
fi

platforms=$1
config_result=$2
shift 2

if ! python3 -c 'import json, sys; value = json.load(sys.stdin); sys.exit(0 if isinstance(value, list) and all(isinstance(item, str) for item in value) else 1)' <<< "$platforms"; then
  echo "Platform selection must be a JSON array" >&2
  exit 2
fi

selected() {
  python3 -c 'import json, sys; sys.exit(0 if sys.argv[1] in json.load(sys.stdin) else 1)' "$1" <<< "$platforms"
}

require_success() {
  local label=$1 result=$2
  if [[ "$result" != success ]]; then
    echo "$label was selected but ended with result: $result" >&2
    exit 1
  fi
}

if selected cuda || selected metax || selected hygon || selected ppu || selected ascend || selected iluvatar; then
  require_success "Hardware config loading" "$config_result"
fi

cuda_first=$1
cuda_specific=$2
hygon=$3
if [[ $# -eq 8 ]]; then
  hygon_experimental=$4
  metax=$5
  ppu=$6
  ppu_experimental=$7
  ascend=$8
  require_hygon_experimental=1
  require_ppu_experimental=1
elif [[ $# -eq 7 ]]; then
  hygon_experimental=$4
  metax=$5
  ppu=$6
  ascend=$7
  require_hygon_experimental=1
  require_ppu_experimental=0
else
  metax=$4
  ppu=$5
  ascend=$6
  require_hygon_experimental=0
  require_ppu_experimental=0
fi
if selected cuda; then
  require_success "CUDA first workload" "$cuda_first"
  require_success "CUDA platform workload" "$cuda_specific"
fi
if selected hygon; then
  require_success "Hygon workload" "$hygon"
  if [[ "$require_hygon_experimental" == 1 ]]; then
    require_success "Hygon experimental workload" "$hygon_experimental"
  fi
fi
if selected metax; then
  require_success "MetaX workload" "$metax"
  # Extended argument layouts identify performance workflows. Torch uses
  # the original layout and has no experimental job.
  if [[ "$require_hygon_experimental" == 1 ]]; then
    require_success "MetaX experimental workload" "${METAX_EXPERIMENTAL_RESULT:-skipped}"
  fi
fi
if selected ppu; then
  require_success "PPU workload" "$ppu"
  if [[ "$require_ppu_experimental" == 1 ]]; then
    require_success "PPU experimental workload" "$ppu_experimental"
  fi
fi
if selected ascend; then
  require_success "Ascend workload" "$ascend"
fi
if selected iluvatar; then
  require_success "Iluvatar workload" "${ILUVATAR_RESULT:-skipped}"
fi

echo "All selected platform workloads succeeded."
