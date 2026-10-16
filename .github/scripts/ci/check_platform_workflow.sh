#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "Usage: $0 <platforms-json> <config> <cuda-base> <cuda-specific> <hygon> <metax> <ppu>" >&2
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

if selected hygon || selected ppu; then
  require_success "Hardware config loading" "$config_result"
fi

cuda_base=$1
cuda_specific=$2
hygon=$3
metax=$4
ppu=$5
if selected cuda; then
  require_success "CUDA base workload" "$cuda_base"
  require_success "CUDA platform workload" "$cuda_specific"
fi
if selected hygon; then
  require_success "Hygon workload" "$hygon"
fi
if selected metax; then
  require_success "MetaX workload" "$metax"
fi
if selected ppu; then
  require_success "PPU workload" "$ppu"
fi

echo "All selected platform workloads succeeded."
