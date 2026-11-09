#!/usr/bin/env bash

# Compare the complete #696 header against the same tree with only #4/#5 undone.
set -euo pipefail

repo_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
log_root=${FLAGCX_CI_LOG_DIR:-/tmp}/iluvatar-device-api-probe
scratch_root=$(mktemp -d /tmp/flagcx-device-api-probe.XXXXXX)
header=flagcx/adaptor/include/device_api/default_comm_traits.h
commit=f626a08c
variant_dir="$log_root/barrier-variants"
snapshot="$scratch_root/after-696"

mkdir -p "$log_root" "$variant_dir" "$snapshot"
source "$repo_root/.github/scripts/set_env/iluvatar.sh"
git -C "$repo_root" rev-parse HEAD > "$log_root/probe-head.txt"
git -C "$repo_root" cat-file -e "50984d4f^{commit}"
git -C "$repo_root" cat-file -e "$commit^{commit}"
"${COREX_CLANG:-/usr/local/corex/bin/clang}" --version \
  > "$log_root/corex-clang-version.txt"
printf 'case\tcommit\ttarget\tresult\tfunction\n' > "$log_root/results.tsv"

git -C "$repo_root" show "50984d4f:$header" > "$variant_dir/after_694.h"
git -C "$repo_root" show "$commit:$header" > "$variant_dir/after_696.h"
python3 "$repo_root/.github/scripts/ci/probe_iluvatar_barrier_variants.py" \
  --old "$variant_dir/after_694.h" --new "$variant_dir/after_696.h" \
  --output-dir "$variant_dir"

git -C "$repo_root" archive "$commit" | tar -xf - -C "$snapshot"
# git archive records submodule references, not their files.
if [[ -d "$repo_root/third-party/json" ]]; then
  cp -a "$repo_root/third-party/json/." "$snapshot/third-party/json/"
fi
cp "$snapshot/test/kernel/iluvatar/device_api.cu" \
  "$scratch_root/original-device_api.cu"

copy_crash_files() {
  local target=$1 marker=$2 destination=$3
  mkdir -p "$destination"
  # CoreX sometimes writes reproducers to /tmp even when TMPDIR is set.
  find /tmp -maxdepth 1 -type f -name "$target-*" -newer "$marker" \
    -exec cp -p {} "$destination/" \;
}

mask_k12_for_world_barrier() {
  local path=$1 destination=$2
  cp "$path" "$destination/original_device_api.cu"
  python3 - "$path" <<'PY'
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
source = path.read_text()
start = "// K12: Inter Barrier — stress test (N iterations of inter barrier sync)"
end = "// K13: World Barrier — sync + arrive/wait split"
if source.count(start) != 1 or source.count(end) != 1:
    raise SystemExit("Cannot isolate K13: K12/K13 markers changed")
first = source.index(start)
last = source.index(end, first)
path.write_text(source[:first] + "#if 0 // Compile-only K13 isolation\n" +
                source[first:last] + "#endif\n" + source[last:])
PY
  diff -u "$destination/original_device_api.cu" "$path" \
    > "$destination/diagnostic-source.patch" || true
}

compile_target() {
  local label=$1 target=$2 case_name=${3:-$2}
  local case_dir="$log_root/$label/$case_name"
  local marker="$case_dir/start.marker" rc=0 result function
  mkdir -p "$case_dir/tmp"
  touch "$marker"
  make -C "$snapshot/test/kernel/iluvatar" -B -n "$target.o" \
    USE_ILUVATAR=1 COMPILE_KERNEL=1 FORCE_DEFAULT_PATH=1 \
    > "$case_dir/reproduce-make-command.txt" 2>&1
  TMPDIR="$case_dir/tmp" make -C "$snapshot/test/kernel/iluvatar" \
    -B --jobs=1 "$target.o" USE_ILUVATAR=1 COMPILE_KERNEL=1 \
    FORCE_DEFAULT_PATH=1 > "$case_dir/compile.log" 2>&1 || rc=$?
  printf '%s\n' "$rc" > "$case_dir/exit-code.txt"
  if ((rc == 0)); then
    result=PASS
  elif grep -Fq 'LLVM ERROR: failed to annotate CFG' "$case_dir/compile.log" && \
       grep -Fq 'BI annotate control flow' "$case_dir/compile.log"; then
    result=BI_CFG_CRASH
  elif grep -Fq 'LLVM ERROR: Broken function found' "$case_dir/compile.log"; then
    result=LLVM_VERIFIER_CRASH
  elif grep -Fq 'LLVM ERROR:' "$case_dir/compile.log"; then
    result=OTHER_LLVM_CRASH
  else
    result=COMPILE_ERROR
  fi
  if ((rc != 0)); then
    copy_crash_files "$target" "$marker" "$case_dir/crash-files"
  fi
  function=$(sed -n "s/.*on function '\([^']*\)'.*/\1/p" \
    "$case_dir/compile.log" | head -1)
  if [[ -z "$function" ]]; then
    function=$(sed -n 's/^in function \([^[:space:]]*\).*/\1/p' \
      "$case_dir/compile.log" | head -1)
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$commit" "$case_name" "$result" "$function" \
    >> "$log_root/results.tsv"
  printf '%-12s %-24s %-21s %s\n' \
    "$label" "$case_name" "$result" "$function"
}

run_case() {
  local label=$1
  local case_dir="$log_root/$label"
  mkdir -p "$case_dir"
  cp "$variant_dir/$label.h" "$snapshot/$header"
  cp "$scratch_root/original-device_api.cu" \
    "$snapshot/test/kernel/iluvatar/device_api.cu"
  diff -u "$variant_dir/after_696.h" "$snapshot/$header" \
    > "$case_dir/source-variant.patch" || true
  printf '%s\n' "$commit" > "$case_dir/source-commit.txt"

  compile_target "$label" device_api
  compile_target "$label" device_ir
  local diagnostic="$case_dir/device_api_without_k12"
  mkdir -p "$diagnostic"
  mask_k12_for_world_barrier \
    "$snapshot/test/kernel/iluvatar/device_api.cu" "$diagnostic"
  compile_target "$label" device_api device_api_without_k12
}

run_case after_696
run_case undo_04_05

summary_rc=0
python3 - "$log_root/results.tsv" "$log_root/summary.md" <<'PY' || summary_rc=$?
import csv
import pathlib
import sys

rows = list(csv.DictReader(open(sys.argv[1], newline=""), delimiter="\t"))
results = {(row["case"], row["target"]): row for row in rows}
targets = ("device_api", "device_ir", "device_api_without_k12")
lines = [
    "# Iluvatar #696 constructor probe", "",
    "Both cases use commit f626a08c and the same CoreX image. The second",
    "case restores only constructor changes #4 and #5 from 50984d4f.", "",
    "| Case | Target | Result | CoreX function |", "|---|---|---|---|",
]
for label in ("after_696", "undo_04_05"):
    for target in targets:
        row = results[(label, target)]
        lines.append(f"| {label} | {target} | {row['result']} | "
                     f"`{row['function']}` |")

baseline_reproduced = all(
    results[("after_696", target)]["result"] == "BI_CFG_CRASH"
    for target in targets
)
candidate_compiles = all(
    results[("undo_04_05", target)]["result"] == "PASS"
    for target in targets
)
lines.extend(["", f"Baseline BI CFG crash reproduced: {baseline_reproduced}.",
              f"Undo #4/#5 compiles all three targets: {candidate_compiles}.",
              "This probe checks compilation only; runtime tests are still required.",
              "The artifact includes compile logs, make commands, source patch,",
              "and CoreX crash files when available.", ""])
pathlib.Path(sys.argv[2]).write_text("\n".join(lines))
print("\n".join(lines))
sys.exit(0 if baseline_reproduced and candidate_compiles else 1)
PY

if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  cat "$log_root/summary.md" >> "$GITHUB_STEP_SUMMARY"
fi
exit "$summary_rc"
