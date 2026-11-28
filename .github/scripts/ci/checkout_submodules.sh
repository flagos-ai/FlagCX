#!/usr/bin/env bash

set -euo pipefail

project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
cd "$project_root"

safe_git=(git -c "safe.directory=$project_root"
  -c "safe.directory=$project_root/third-party/googletest"
  -c "safe.directory=$project_root/third-party/json")

network_git=("${safe_git[@]}" -c http.version=HTTP/1.1
  -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60)

# A lock may belong to a live Git process. Never delete it or repeatedly fetch
# against it; isolated job checkouts avoid inheriting another job's metadata.
check_submodule_locks() {
  local entry path metadata lock
  while IFS=$'\t' read -r -d '' entry path; do
    [[ "$entry" == "160000 "* ]] || continue
    metadata=$("${safe_git[@]}" rev-parse --git-path "modules/$path") || return $?
    for lock in "$metadata"/shallow.lock "$metadata"/index.lock \
        "$metadata"/config.lock "$metadata"/packed-refs.lock; do
      if [[ -e "$lock" ]]; then
        echo "Submodule checkout blocked by existing Git lock: $lock" >&2
        ls -ld "$lock" >&2 || true
        echo 'Lock retained; check its owner or use a fresh checkout before retrying.' >&2
        return 1
      fi
    done
  done < <("${safe_git[@]}" ls-files --stage -z)
}

recover_incomplete_checkouts() {
  local entry path commit
  while IFS=$'\t' read -r -d '' entry path; do
    [[ "$entry" == "160000 "* ]] || continue
    [[ -e "$path/.git" ]] || continue
    if "${safe_git[@]}" -C "$path" rev-parse --verify HEAD >/dev/null 2>&1; then
      continue
    fi
    commit=${entry#160000 }
    commit=${commit%% *}
    echo "Recovering interrupted checkout of $path at $commit"
    timeout --signal=TERM --kill-after=10s 2m \
      "${network_git[@]}" -C "$path" fetch --depth=1 origin "$commit" || return $?
    "${safe_git[@]}" -C "$path" checkout --detach "$commit" || return $?
  done < <("${safe_git[@]}" ls-files --stage -z)
}

# Apply HTTP/1.1 to the update and its child fetches, including inside job
# containers whose Git configuration is separate from the runner host.
# Explicit depth also limits the fetch of a pinned commit outside remote HEAD.
check_submodule_locks || exit $?
"${safe_git[@]}" submodule sync --recursive
for attempt in 1 2 3; do
  check_submodule_locks || exit $?
  echo "Checking out submodules (attempt $attempt/3, HTTP/1.1, depth=1)"
  if recover_incomplete_checkouts && \
      timeout --signal=TERM --kill-after=10s 5m \
      "${network_git[@]}" submodule update \
      --init --force --recursive --checkout --depth=1 --jobs=1; then
    "${safe_git[@]}" submodule status --recursive
    exit 0
  else
    status=$?
  fi
  check_submodule_locks || exit $?

  if [[ "$attempt" == 3 ]]; then
    echo "Submodule checkout failed after 3 attempts (exit=$status)." >&2
    exit "$status"
  fi
  echo "Submodule checkout failed; retrying in $((attempt * 5)) seconds." >&2
  sleep "$((attempt * 5))"
done
