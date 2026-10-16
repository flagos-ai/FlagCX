#!/usr/bin/env bash

set -euo pipefail

project_root=${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}
cd "$project_root"

git config --global --add safe.directory "$project_root"
git config --global --add safe.directory "$project_root/third-party/googletest"
git config --global --add safe.directory "$project_root/third-party/json"
git submodule sync --recursive

network_git=(git -c http.version=HTTP/1.1
  -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60)

recover_incomplete_checkouts() {
  local entry path commit
  while IFS=$'\t' read -r -d '' entry path; do
    [[ "$entry" == "160000 "* ]] || continue
    [[ -e "$path/.git" ]] || continue
    if git -C "$path" rev-parse --verify HEAD >/dev/null 2>&1; then
      continue
    fi
    commit=${entry#160000 }
    commit=${commit%% *}
    echo "Recovering interrupted checkout of $path at $commit"
    timeout --signal=TERM --kill-after=10s 2m \
      "${network_git[@]}" -C "$path" fetch --depth=1 origin "$commit" || return $?
    git -C "$path" checkout --detach "$commit" || return $?
  done < <(git ls-files --stage -z)
}

# Apply HTTP/1.1 to the update and its child fetches, including inside job
# containers whose Git configuration is separate from the runner host.
# Explicit depth also limits the fetch of a pinned commit outside remote HEAD.
for attempt in 1 2 3; do
  echo "Checking out submodules (attempt $attempt/3, HTTP/1.1, depth=1)"
  if recover_incomplete_checkouts && \
      timeout --signal=TERM --kill-after=10s 5m \
      "${network_git[@]}" submodule update \
      --init --force --recursive --checkout --depth=1 --jobs=1; then
    git submodule status --recursive
    exit 0
  else
    status=$?
  fi

  if [[ "$attempt" == 3 ]]; then
    echo "Submodule checkout failed after 3 attempts (exit=$status)." >&2
    exit "$status"
  fi
  echo "Submodule checkout failed; retrying in $((attempt * 5)) seconds." >&2
  sleep "$((attempt * 5))"
done
