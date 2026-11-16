#!/usr/bin/env bash
# End-to-end C-scope driver: env check, build, all run stages, collect.
#
# Stages that are blocked by missing A/B deliverables are recorded BLOCKED and
# do not abort the rest of the run. Genuine failures abort and return non-zero.
#
# Usage: run_all.sh [--allow-blocked] [--skip-env]

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

allow=()
skip_env=0
skip_build=0
skip_host=0
for arg in "$@"; do
  case "$arg" in
    --allow-blocked) allow=(--allow-blocked) ;;
    --skip-env) skip_env=1 ;;
    --skip-build) skip_build=1 ;;
    --skip-host) skip_host=1 ;;
    -h|--help)
      echo "Usage: $0 [--allow-blocked] [--skip-env] [--skip-build] [--skip-host]"
      exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

rc=0

# Archive the previous run so stale .status files can never be aggregated into
# the current verdict. This is what makes repeated rounds independent.
if [ -d "$LOG_DIR" ] && [ -n "$(ls -A "$LOG_DIR" 2>/dev/null)" ]; then
  archive="$RESULT_DIR/archive-$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$RESULT_DIR"
  if mv "$LOG_DIR" "$archive" 2>/dev/null; then
    mkdir -p "$LOG_DIR"
    log "archived previous logs -> $archive"
  fi
fi

if [ "$skip_env" -eq 0 ]; then
  "$HERE/env_check.sh" || { log "env_check failed"; exit 1; }
fi

if [ "$skip_build" -eq 0 ]; then
  "$HERE/build_all.sh" "${allow[@]}" || rc=$?
else
  log "skip-build: reusing existing artifacts"
fi

if [ "$skip_host" -eq 0 ]; then
  # Real MUSA host-side lifecycle smoke (independent of A/B kernel/IPC work).
  "$HERE/run_host_smoke.sh" --probe-devmem --probe-devptr --probe-barrier --probe-window || rc=$?

else
  log "skip-host: skipping host smoke and topology probe"
fi

# Always attempt the run stages; each one records its own status.
"$HERE/run_intra.sh"           "${allow[@]}" || rc=$?
"$HERE/run_inter.sh"           "${allow[@]}" || rc=$?
"$HERE/run_unified_intra.sh"   "${allow[@]}" || rc=$?
"$HERE/run_unified_inter.sh"   "${allow[@]}" || rc=$?

"$HERE/collect_results.sh" "${allow[@]}" || rc=$?

if [ "${#allow[@]}" -gt 0 ]; then
  log "exploration mode (--allow-blocked): see ACCEPTANCE line in SUMMARY.md; blocked stages are NOT a pass"
fi
exit "$rc"
