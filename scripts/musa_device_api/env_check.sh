#!/usr/bin/env bash
# D1 environment audit for MUSA Device API work.
#
# Verifies toolchain, runtime libraries, devices, MPI, submodule headers and
# the built library. Emits key=value facts and a PASS/FAIL summary.
# Exit 0 only if every critical prerequisite is present.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

NAME=env_check
out="$LOG_DIR/$NAME.out"
mkdir -p "$LOG_DIR"
: >"$out"

fail=0
check() {
  local label="$1" ok="$2" detail="${3:-}"
  if [ "$ok" -eq 1 ]; then
    printf 'PASS %-28s %s\n' "$label" "$detail" | tee -a "$out"
  else
    printf 'FAIL %-28s %s\n' "$label" "$detail" | tee -a "$out"
    fail=$((fail + 1))
  fi
}

has_cmd() { command -v "$1" >/dev/null 2>&1; }

log "== FlagCX MUSA Device API environment =="

# --- toolchain ---------------------------------------------------------------
check "cmd:mcc"      "$(has_cmd mcc && echo 1 || echo 0)"      "$(command -v mcc || echo missing)"
check "cmd:mpirun"   "$(has_cmd mpirun && echo 1 || echo 0)"   "$(mpirun --version 2>/dev/null | head -1 || echo missing)"
check "cmd:timeout"  "$(has_cmd timeout && echo 1 || echo 0)"  "$(command -v timeout || echo missing)"
check "cmd:g++"      "$(has_cmd g++ && echo 1 || echo 0)"      "$(g++ --version 2>/dev/null | head -1 || echo missing)"

if has_cmd mcc; then
  mcc_ver="$(mcc --version 2>&1 | tr '\n' ' ' | sed 's/  */ /g')"
  printf 'INFO mcc_version=%s\n' "$mcc_ver" | tee -a "$out"
fi

# --- MUSA runtime / CCL ------------------------------------------------------
# libmusa.so.1 may live in the system loader path while libmccl/libmusart come
# from the MUSA SDK. Search both before declaring a miss.
find_lib() {
  local lib="$1" d
  for d in "$MUSA_LIB" /usr/local/musa/lib /usr/lib/x86_64-linux-gnu /usr/lib /lib/x86_64-linux-gnu; do
    if [ -e "$d/$lib" ]; then printf '%s/%s\n' "$d" "$lib"; return 0; fi
  done
  if command -v ldconfig >/dev/null 2>&1; then
    ldconfig -p 2>/dev/null | awk -v l="$lib" '$1==l {print $NF; exit}'
  fi
}
for lib in libmusa.so.1 libmusart.so.4 libmccl.so.2; do
  path="$(find_lib "$lib")"
  if [ -n "$path" ]; then
    check "lib:$lib" 1 "$path"
  else
    check "lib:$lib" 0 "not found in MUSA/system loader paths"
  fi
done

# --- devices -----------------------------------------------------------------
gmi=""
if has_cmd mthreads-gmi; then gmi="$(command -v mthreads-gmi)"; fi
if [ -n "$gmi" ]; then
  dev_list="$("$gmi" -L 2>/dev/null || "$gmi" 2>/dev/null || true)"
  ndev="$(printf '%s\n' "$dev_list" | grep -c '^GPU' || true)"
  printf 'INFO musa_device_count=%s\n' "${ndev:-0}" | tee -a "$out"
  printf '%s\n' "$dev_list" | sed 's/^/INFO gmi: /' | tee -a "$out"
  [ "${ndev:-0}" -ge 2 ] && check "musa_devices>=2" 1 "count=${ndev}" || check "musa_devices>=2" 0 "count=${ndev:-0}"
else
  check "cmd:mthreads-gmi" 0 "no MUSA device management tool found"
fi
dri_count="$(ls /dev/dri/card* 2>/dev/null | wc -l)"
printf 'INFO dri_card_count=%s\n' "$dri_count" | tee -a "$out"
[ "$dri_count" -ge 2 ] && check "dri_cards>=2" 1 "count=$dri_count" || check "dri_cards>=2" 0 "count=$dri_count"

# --- submodules / headers ----------------------------------------------------
check "json_header" \
  "$([ -f "$PROJECT_ROOT/third-party/json/single_include/nlohmann/json.hpp" ] && echo 1 || echo 0)" \
  "third-party/json/single_include/nlohmann/json.hpp"
if [ -f "$PROJECT_ROOT/third-party/googletest/googletest/include/gtest/gtest.h" ]; then
  check "gtest_header" 1 "vendored"
elif [ -f /usr/src/googletest/googletest/include/gtest/gtest.h ]; then
  check "gtest_header" 1 "system /usr/src/googletest"
else
  check "gtest_header" 0 "missing"
fi

# --- built library -----------------------------------------------------------
if [ -f "$FLAGCX_BUILD_LIB/libflagcx.so" ]; then
  check "libflagcx" 1 "$FLAGCX_BUILD_LIB/libflagcx.so"
  needed="$(readelf -d "$FLAGCX_BUILD_LIB/libflagcx.so" 2>/dev/null | grep -oE 'lib(mccl|musa|musart)[^]]*' | sort -u | tr '\n' ' ')"
  printf 'INFO libflagcx_needed=%s\n' "$needed" | tee -a "$out"
else
  check "libflagcx" 0 "not built (run build_all.sh)"
fi

# --- git ---------------------------------------------------------------------
git_head="$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
git_branch="$(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
printf 'INFO git_head=%s\nINFO git_branch=%s\n' "$git_head" "$git_branch" | tee -a "$out"

printf '\n' | tee -a "$out"
if [ "$fail" -eq 0 ]; then
  log "env_check: PASS"
  write_status "$NAME" PASS 0 0 "env_check.sh"
  exit 0
fi
log "env_check: FAIL ($fail critical checks)"
write_status "$NAME" FAIL 1 0 "env_check.sh"
exit 1
