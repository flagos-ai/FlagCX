#!/usr/bin/env bash

# Compile both CoreX Device API translation units at each mainline boundary.
# This diagnostic does not change or replace the normal Iluvatar test suites.
set -euo pipefail

repo_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
log_root=${FLAGCX_CI_LOG_DIR:-/tmp}/iluvatar-device-api-probe
scratch_root=$(mktemp -d /tmp/flagcx-device-api-probe.XXXXXX)
header=flagcx/adaptor/include/device_api/default_comm_traits.h

mkdir -p "$log_root"
source "$repo_root/.github/scripts/set_env/iluvatar.sh"
git -C "$repo_root" rev-parse HEAD > "$log_root/probe-head.txt"
"${COREX_CLANG:-/usr/local/corex/bin/clang}" --version > "$log_root/corex-clang-version.txt"
printf 'case\tcommit\ttarget\tresult\tfunction\n' > "$log_root/results.tsv"

mainline=(pre_694 after_694 after_696 after_692 merged_670)
commits=(4a3dca96 50984d4f f626a08c 14282a1a 9f350cf8)

for i in "${!commits[@]}"; do
  git -C "$repo_root" cat-file -e "${commits[$i]}^{commit}"
  if ((i > 0)); then
    git -C "$repo_root" merge-base --is-ancestor "${commits[$((i - 1))]}" "${commits[$i]}" || {
      echo "Probe commits are not ordered along one mainline" >&2
      exit 2
    }
  fi
done

copy_crash_files() {
  local target=$1 marker=$2 destination=$3
  mkdir -p "$destination"
  # CoreX sometimes writes reproducer files to /tmp even when TMPDIR is set.
  find /tmp -maxdepth 1 -type f -name "$target-*" -newer "$marker" \
    -exec cp -p {} "$destination/" \;
}

restore_inter_barrier_class() {
  local snapshot=$1
  local old_class="$snapshot/parent-inter-barrier-class.h"
  git -C "$repo_root" show "50984d4f:$header" > "$old_class"
  python3 - "$old_class" "$snapshot/$header" <<'PY'
import pathlib
import sys

old = pathlib.Path(sys.argv[1]).read_text()
path = pathlib.Path(sys.argv[2])
new = path.read_text()
start = "// ---- Barrier<DefaultBackend<P>, flagcxTeamTagInter, Coop> ----"
end = "// ---- Barrier<DefaultBackend<P>, flagcxTeamTagWorld, Coop> ----"
for text in (old, new):
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit("Inter barrier class markers changed; probe cannot apply")
old_section = old[old.index(start):old.index(end)]
path.write_text(new[:new.index(start)] + old_section + new[new.index(end):])
PY
}

force_proxy_dispatch() {
  local path=$1
  python3 - "$path" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text()
dispatch = text.index("struct DefaultNetDispatch {")
start = text.index("  template <typename Op, typename... Args>\n", dispatch)
end = text.index("\n  }\n};", start) + len("\n  }")
replacement = """  template <typename Op, typename... Args>
  FLAGCX_DEVICE_INLINE_DECORATOR static auto call(DefaultNetContext ctx,
                                                  Args &&...args)
      -> decltype(DefaultNetApi<DefaultNetProviderProxy, Op>::call(
          ctx, static_cast<Args &&>(args)...)) {
    return DefaultNetApi<DefaultNetProviderProxy, Op>::call(
        ctx, static_cast<Args &&>(args)...);
  }"""
path.write_text(text[:start] + replacement + text[end:])
PY
}

run_case() {
  local label=$1 commit=$2 variant=${3:-none}
  local snapshot="$scratch_root/$label"
  mkdir -p "$snapshot" "$log_root/$label"
  git -C "$repo_root" archive "$commit" | tar -xf - -C "$snapshot"
  # git archive records submodule references, not their files. The pinned JSON
  # revision is the same at every boundary in this probe.
  if [[ -d "$repo_root/third-party/json" ]]; then
    cp -a "$repo_root/third-party/json/." "$snapshot/third-party/json/"
  fi
  cp "$snapshot/$header" "$log_root/$label/original-default_comm_traits.h"

  case "$variant" in
    none) ;;
    parent_header)
      local parent=$4
      git -C "$repo_root" show "$parent:$header" > "$snapshot/$header"
      ;;
    parent_inter_barrier)
      restore_inter_barrier_class "$snapshot"
      ;;
    proxy_dispatch)
      force_proxy_dispatch "$snapshot/$header"
      ;;
    *) echo "Unknown probe variant: $variant" >&2; exit 2 ;;
  esac

  diff -u "$log_root/$label/original-default_comm_traits.h" \
    "$snapshot/$header" > "$log_root/$label/source-variant.patch" || true
  printf '%s\n' "$commit" > "$log_root/$label/source-commit.txt"

  local target case_dir marker rc result function
  for target in device_api device_ir; do
    case_dir="$log_root/$label/$target"
    mkdir -p "$case_dir/tmp"
    marker="$case_dir/start.marker"
    touch "$marker"
    make -C "$snapshot/test/kernel/iluvatar" -n "$target.o" \
      USE_ILUVATAR=1 COMPILE_KERNEL=1 FORCE_DEFAULT_PATH=1 \
      > "$case_dir/reproduce-make-command.txt" 2>&1

    rc=0
    TMPDIR="$case_dir/tmp" make -C "$snapshot/test/kernel/iluvatar" \
      --jobs=1 "$target.o" USE_ILUVATAR=1 COMPILE_KERNEL=1 \
      FORCE_DEFAULT_PATH=1 > "$case_dir/compile.log" 2>&1 || rc=$?
    printf '%s\n' "$rc" > "$case_dir/exit-code.txt"
    if ((rc == 0)); then
      result=PASS
    elif grep -Fq 'LLVM ERROR: failed to annotate CFG' "$case_dir/compile.log" && \
         grep -Fq 'BI annotate control flow' "$case_dir/compile.log"; then
      result=BI_CFG_CRASH
      copy_crash_files "$target" "$marker" "$case_dir/crash-files"
    else
      result=COMPILE_ERROR
      copy_crash_files "$target" "$marker" "$case_dir/crash-files"
    fi
    function=$(sed -n "s/.*BI annotate control flow' on function '\([^']*\)'.*/\1/p" \
      "$case_dir/compile.log" | head -1)
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$label" "$commit" "$target" "$result" "$function" >> "$log_root/results.tsv"
    printf '%-28s %-10s %-13s %s\n' "$label" "$commit" "$target" "$result"
  done
}

for i in "${!commits[@]}"; do
  run_case "${mainline[$i]}" "${commits[$i]}"
done

# File and control-flow probes use private source snapshots. They are compile
# diagnostics only; a passing variant is never treated as a production fix.
run_case after_694_parent_header 50984d4f parent_header 4a3dca96
run_case after_694_proxy_dispatch 50984d4f proxy_dispatch
run_case after_696_parent_header f626a08c parent_header 50984d4f
run_case after_696_parent_inter_barrier f626a08c parent_inter_barrier
run_case after_696_proxy_dispatch f626a08c proxy_dispatch

python3 - "$log_root/results.tsv" "$log_root/summary.md" <<'PY'
import csv
import pathlib
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline=""), delimiter="\t"))
labels = ["pre_694", "after_694", "after_696", "after_692", "merged_670"]
lines = ["# Iluvatar Device API compile probe", "",
         "All cases used the same CoreX image and kernel Makefile flags.", "",
         "| Source case | device_api | device_ir |", "|---|---|---|"]
results = {(row["case"], row["target"]): row for row in rows}
for label in labels + ["after_694_parent_header", "after_694_proxy_dispatch",
                       "after_696_parent_header", "after_696_parent_inter_barrier",
                       "after_696_proxy_dispatch"]:
    lines.append("| {} | {} | {} |".format(
        label, *(results[(label, target)]["result"] for target in
                 ("device_api", "device_ir"))))
lines.extend(["", "## First failing mainline boundary", ""])
for target in ("device_api", "device_ir"):
    if results[(labels[0], target)]["result"] != "PASS":
        lines.append(f"- {target}: already fails at pre_694; earlier history is needed")
        continue
    previous = "pre_694"
    for label in labels[1:]:
        before = results[(previous, target)]["result"]
        after = results[(label, target)]["result"]
        if before == "PASS" and after != "PASS":
            function = results[(label, target)]["function"]
            lines.append(f"- {target}: {previous} → {label} ({after}); "
                         f"CoreX function: `{function or 'not reported'}`")
            if label == "after_694" and after == "BI_CFG_CRASH":
                if results[("after_694_parent_header", target)]["result"] == "PASS":
                    lines.append("  - Replacing #694 default_comm_traits.h with its parent passes: "
                                 "the regression is in that header.")
                if results[("after_694_proxy_dispatch", target)]["result"] == "PASS":
                    lines.append("  - Direct Proxy dispatch passes: inspect the #694 "
                                 "DefaultNetDispatch::call switch and its inlined CFG.")
            if label == "after_696" and after == "BI_CFG_CRASH":
                if results[("after_696_parent_header", target)]["result"] == "PASS":
                    lines.append("  - Replacing #696 default_comm_traits.h with its parent passes: "
                                 "the regression is in that header.")
                if results[("after_696_parent_inter_barrier", target)]["result"] == "PASS":
                    lines.append("  - Restoring the parent Inter barrier class passes: "
                                 "inspect its #696 control-flow changes.")
                if results[("after_696_proxy_dispatch", target)]["result"] == "PASS":
                    lines.append("  - Direct Proxy dispatch passes at #696: "
                                 "inspect its interaction with the #694 switch.")
            break
        previous = label
    else:
        lines.append(f"- {target}: no failure in the selected mainline commits")
lines.extend(["", "Each source variant is a compile-only probe, not a production fix.",
              "COMPILE_ERROR in a variant provides no causal evidence. The regular",
              "Iluvatar device_api and device_api_unified_ir suites remain required.", ""])
pathlib.Path(sys.argv[2]).write_text("\n".join(lines))
print("\n".join(lines))
PY

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  cat "$log_root/summary.md" >> "$GITHUB_STEP_SUMMARY"
fi
