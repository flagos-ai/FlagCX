#!/usr/bin/env bash

# Copyright (c) 2026 BAAI. All rights reserved.
#
# See LICENSE for license information.

# Run the repository's pre-commit hooks locally.
# Usage: bash .github/scripts/ci/format.sh

set -euo pipefail

FLAGCX_PATH=${FLAGCX_PATH:-${GITHUB_WORKSPACE:-$(git rev-parse --show-toplevel)}}

cd "$FLAGCX_PATH"

# clang-format comes from the hook environment pinned in .pre-commit-config.yaml,
# so pre-commit provisions the version the repository is formatted with; whatever
# clang-format happens to be on PATH is neither used nor required.
python3 -m pip install pre-commit
python3 -m pre_commit run --all-files
